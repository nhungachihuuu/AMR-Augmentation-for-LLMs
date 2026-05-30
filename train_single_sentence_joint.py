import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import wandb
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset, Subset

from src.utils.seed import seed_everything
from src.utils.lr_schedule import adjust_lr_step
from torch.nn.utils import clip_grad_norm_
from src.config import parse_args_llama
from src.utils.ckpt import _save_checkpoint, _reload_best_model_search
from src.model import load_model, llm_model_path
from src.dataset import load_dataset
from src.utils.post_processing import post_processing_func
import datetime
import numpy as np
import math
from contextlib import contextmanager


def print_snapshot():
    torch.cuda.synchronize()
    print(f"alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
          f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB  "
          f"peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB")


def reset_peak():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


@contextmanager
def peak_region():
    reset_peak()
    yield
    torch.cuda.synchronize()


def make_subset(dataset, max_size, random=False):
    n = min(max_size, len(dataset))
    indices = np.random.choice(len(dataset), size=n, replace=False).tolist() if random else list(range(n))
    return Subset(dataset, indices)


def main(args):
    time = datetime.datetime.today().strftime('%m%d%H%M')
    seed = args.seed
    job_id = os.environ.get("JOB_ID")
    wandb.init(project=f"{args.project}",
               name=f"all_data_{args.model_name}_{args.llm_model_name}_{seed}_{args.lr}_{job_id}_{time}_{args.phase2}",
               config=args)
    seed_everything(seed=args.seed)

    print(args)
    print(f"create dataset: {args.dataset}")

    train_datasets, dev_datasets, test_datasets = [], [], {}
    for ds_name in args.dataset:
        train, dev, test = load_dataset[ds_name].load(args.prompt_type)
        print(train[0])
        train_datasets.append(train)
        dev_datasets.append(dev)
        test_datasets[ds_name] = test

    train_dataset = ConcatDataset([make_subset(ds, args.train_subset, random=True)  for ds in train_datasets])
    val_dataset   = ConcatDataset([make_subset(ds, args.dev_subset)                 for ds in dev_datasets])
    new_test_datasets = {ds_name: make_subset(ds, args.test_subset) for ds_name, ds in test_datasets.items()}

    print(f"Length for running: {len(train_dataset)}, {len(val_dataset)}")

    def mixed_collate(batch):
        keys = set().union(*(sample.keys() for sample in batch))
        return {k: [sample.get(k, "") for sample in batch] for k in keys}

    print("create dataloader")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,    drop_last=True,  pin_memory=True, shuffle=True,  collate_fn=mixed_collate)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,    drop_last=True,  pin_memory=True, shuffle=False, collate_fn=mixed_collate)
    test_loaders = {
        ds_name: DataLoader(ds, batch_size=args.eval_batch_size, drop_last=False, pin_memory=True, shuffle=False, collate_fn=mixed_collate)
        for ds_name, ds in new_test_datasets.items()
    }
    print("create dataloader successfully")

    # Step 3: Build Model
    print("create model")
    args.llm_model_path = llm_model_path[args.llm_model_name]

    torch.cuda.empty_cache()
    torch.cuda.init()
    reset_peak()

    model = load_model[args.model_name](args=args)
    model.to("cuda:0")
    print("create model successfully")
    print_snapshot()

    # Step 4: Set Optimizer
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{'params': params, 'lr': args.lr, 'weight_decay': args.wd}],
        betas=(0.9, 0.95)
    )
    trainable_params, all_param = model.print_trainable_params()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
    print_snapshot()

    # Step 5: Training
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_steps)
    total_update_steps = args.num_epochs * updates_per_epoch
    warmup_steps = min(args.warmup_steps, int(round(args.warmup_ratio * total_update_steps)))
    global_update_step = 0

    progress_bar = tqdm(range(args.num_epochs * len(train_loader)))
    best_val_loss = float('inf')
    best_epoch = 0

    print("start training")
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss, accum_loss = 0., 0.

        for step, batch in enumerate(train_loader):
            print(batch.keys())

            with peak_region():
                loss = model(batch)
                if torch.isnan(loss):
                    print(f"NaN loss at step {step}, batch: {batch}")
                    raise ValueError("NaN loss encountered")

            epoch_loss += loss.item()
            accum_loss += loss.item()
            print('forward mem:'); print_snapshot()

            with peak_region():
                (loss / args.grad_steps).backward()
            print('backward mem:'); print_snapshot()

            if (step + 1) % args.grad_steps == 0:
                clip_grad_norm_(optimizer.param_groups[0]['params'], 1.0)
                adjust_lr_step(
                    optimizer.param_groups[0],
                    update_step=global_update_step,
                    total_steps=total_update_steps,
                    base_lr=args.lr,
                    min_lr=args.min_lr,
                    warmup_steps=warmup_steps,
                )
                optimizer.step()
                optimizer.zero_grad()
                global_update_step += 1

                wandb.log({'Lr': optimizer.param_groups[0]["lr"]})
                wandb.log({'Accum Loss': accum_loss / args.grad_steps})
                accum_loss = 0.

            print_snapshot()
            progress_bar.update(1)

        print(f"Epoch: {epoch}|{args.num_epochs}: Train Loss (Epoch Mean): {epoch_loss / len(train_loader)}")
        wandb.log({'Train Loss (Epoch Mean)': epoch_loss / len(train_loader)})

        val_loss = 0.
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                val_loss += model(batch).item()
        val_loss /= len(val_loader)
        print(f"Epoch: {epoch}|{args.num_epochs}: Val Loss: {val_loss}")
        wandb.log({'Val Loss': val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(model, optimizer, epoch, job_id, time, args, is_best=True)
            best_epoch = epoch

        print(f'Epoch {epoch} Val Loss {val_loss} Best Val Loss {best_val_loss} Best Epoch {best_epoch}')

        if epoch - best_epoch >= args.patience:
            print(f'Early stop at epoch {epoch}')
            break

    print('Finish training!')
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

    # Step 6: Evaluate
    model = _reload_best_model_search(model, job_id, time, args)
    model.eval()

    for ds_name, test_loader in test_loaders.items():
        eval_outputs = []
        for step, batch in enumerate(tqdm(test_loader)):
            with torch.no_grad():
                eval_outputs.append(model.inference(batch))

        prefix = ds_name.split("_")[0]
        post_processing_func[prefix](eval_outputs, args, job_id, time)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args_llama()
    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()



# import os 
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# import torch
# import wandb
# import copy
# import gc
# from tqdm import tqdm
# from torch.utils.data import DataLoader

# from src.utils.seed import seed_everything
# from src.utils.lr_schedule import adjust_learning_rate,adjust_lr_step
# from torch.nn.utils import clip_grad_norm_
# from src.config import parse_args_llama
# from src.utils.ckpt import _save_checkpoint,_reload_model, _reload_best_model, _reload_best_model_search
# from src.model import load_model, llm_model_path
# from src.dataset import load_dataset
# from src.utils.evaluate import eval_funcs
# from src.utils.post_processing import post_processing_func
# from src.utils.collate import collate_funcs
# from torch.utils.data import Subset
# import datetime
# import os
# from torch.utils.data import ConcatDataset
# from contextlib import contextmanager
# import pickle
# import numpy as np 
# import math

# def print_snapshot():
#     torch.cuda.synchronize()
#     print(f"alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
#           f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB  "
#           f"peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB")



# def reset_peak():
#     torch.cuda.reset_peak_memory_stats()
#     torch.cuda.synchronize()

# @contextmanager
# def peak_region():
#     reset_peak()
#     yield
#     torch.cuda.synchronize()


# def main(args):
#     time = datetime.datetime.today().strftime('%m%d%H%M')
#     seed = args.seed
#     job_id = os.environ.get("JOB_ID")
#     wandb.init(project=f"{args.project}",
#                name=f"all_data_{args.model_name}_{args.llm_model_name}_{seed}_{args.lr}_{job_id}_{time}_{args.phase2}",
#                config=args)
#     seed_everything(seed=args.seed)

#     print(args)
#     print(f"create dataset: {args.dataset}")

#     train_datasets, dev_datasets, test_datasets = [], [], {}
#     len_test_dataset = []
#     for ds_name in args.dataset:

#         train, dev, test = load_dataset[ds_name].load(args.prompt_type)
#         print(train[0])
#         train_datasets.append(train)
#         dev_datasets.append(dev)
#         len_test_dataset.append(len(test))
#         test_datasets[ds_name] = test
    


#     class SubsetWithAttrs(Subset):
#         def __getattr__(self, name):
#             # Forward attribute lookups to the original dataset
#             return getattr(self.dataset, name) 

#     # Create subset for testing if needed:
#     train_datasets_subset = []
#     for train in train_datasets:

#         train_indices = np.random.choice(len(train), 
#                                     size=min(args.train_subset, len(train)), 
#                                     replace=False).tolist()
#         train_dataset = SubsetWithAttrs(train, train_indices)
#         train_datasets_subset.append(train_dataset)

#     dev_datasets_subset = []
#     for dev in dev_datasets:

#         dev_indices = list(range(min(args.dev_subset, len(dev))))
#         dev_dataset = SubsetWithAttrs(dev, dev_indices)
#         dev_datasets_subset.append(dev_dataset)

#     #no graph needed
#     class ConcatDatasetWithGraph(ConcatDataset):
#         def __init__(self, datasets,add_graph=False):
#             super().__init__(datasets)
#             self.add_graph = add_graph
#             if self.add_graph:
#                self.graph = self._merge_graphs([ds.graph for ds in datasets])

#         def _merge_graphs(self, graphs):
#             # All graphs are dicts with identical keys
#             merged = {}
#             for key in graphs[0].keys():
#                 merged[key] = []
#                 for g in graphs:
#                     merged[key].extend(g[key])  # concatenate lists
#             return merged

#     #Final train and dev dataset
#     train_dataset = ConcatDatasetWithGraph(train_datasets_subset)
#     val_dataset   = ConcatDatasetWithGraph(dev_datasets_subset)
#     test_indices_list = []
#     if args.test_subset < min(len_test_dataset):

#         new_test_datasets = {}
#         for ds_name in test_datasets.keys():
#             test_indices = list(range(args.test_subset))
#             test_indices_list.append(test_indices)
#             new_ds = SubsetWithAttrs(test_datasets[ds_name], test_indices) #in here I assume this wont happen !!
#             new_test_datasets[ds_name]= new_ds
#     else:
#         # new_test_datasets = test_datasets
#         new_test_datasets = {}
#         for ds_name in test_datasets.keys():

#             test_indices = list(range(len(test_datasets[ds_name]))) 
#             test_indices_list.append(test_indices)
#             new_ds = SubsetWithAttrs(test_datasets[ds_name], test_indices) #in here I assume this wont happen !!
#             new_test_datasets[ds_name]= new_ds
    

#     print(f"Length for running : {len(train_dataset)}, {len(val_dataset)}" )

#     # collate_fn = collate_funcs[args.dataset](dataset.graph)
#     print("create dataloader")
#     # collate_fn_train = collate_funcs['single'](train_dataset.graph)
#     # collate_fn_dev = collate_funcs['single'](val_dataset.graph)
#     # collate_fn_test_dict = {}
#     # for ds_name in new_test_datasets.keys():

#     #     collate_fn_test_dict[ds_name] = collate_funcs['single'](new_test_datasets[ds_name].graph)

#     # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, pin_memory=True, shuffle=True, collate_fn = collate_fn_train)
#     # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=False, pin_memory=True, shuffle=False, collate_fn = collate_fn_dev)

#     # put this near your DataLoader setup
#     def mixed_collate(batch):
#         # union of all keys across samples
#         keys = set().union(*(sample.keys() for sample in batch))
#         out = {}
#         for k in keys:
#             vals = []
#             for sample in batch:
#                 vals.append(sample.get(k, ""))  # placeholder when missing
#             out[k] = vals
#         return out


#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, pin_memory=True, shuffle=True,collate_fn=mixed_collate)
#     val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, pin_memory=True, shuffle=False,collate_fn=mixed_collate)

#     test_loaders = {}
#     for ds_name in new_test_datasets.keys():
#         test_loader = DataLoader(new_test_datasets[ds_name], batch_size=args.eval_batch_size, drop_last=False, pin_memory=True, shuffle=False,collate_fn=mixed_collate)
#         test_loaders[ds_name] = test_loader

#     print("create dataloader successfully")


#     # Step 3: Build Model
#     print("create model")
#     args.llm_model_path = llm_model_path[args.llm_model_name]
    
#     # Block for inspect memories
    
#     torch.cuda.empty_cache()
#     torch.cuda.init()   # ← add this
#     reset_peak()


#     model = load_model[args.model_name](args=args)
#     # if args.phase2:
#     #     model = _reload_model(model, checkpoint_path=args.phase2_checkpoint)        
#     #     print('reload model from phase 1 ')
#     model.to("cuda:0")
#     print("create model succesfully")

#     # Block for inspect memories
#     print_snapshot()

#     # Step 4 Set Optimizer
#     params = [p for _, p in model.named_parameters() if p.requires_grad]
#     optimizer = torch.optim.AdamW(
#         [{'params': params, 'lr': args.lr, 'weight_decay': args.wd},],
#         betas=(0.9, 0.95)
#     )
#     trainable_params, all_param = model.print_trainable_params()
#     print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
    
    
#     # Block for inspect memories
#     print_snapshot()

#     # Step 5. Training

#     # Before training loop
#     updates_per_epoch = math.ceil(len(train_loader) / args.grad_steps)  # use ceil to be safe
#     total_update_steps = args.num_epochs * updates_per_epoch
#     warmup_ratio = args.warmup_ratio # set between 0.03 and 0.10
#     warmup_steps = min(args.warmup_steps, int(round(warmup_ratio * total_update_steps)))    
#     global_update_step = 0  # do NOT reset per epoch


#     num_training_steps = args.num_epochs * len(train_loader)
#     progress_bar = tqdm(range(num_training_steps))
#     best_val_loss, best_val_acc = float('inf'), -float('inf')
#     best_epoch = 0
#     print("start training")
#     for epoch in range(args.num_epochs):
        

#         model.train()
#         epoch_loss, accum_loss = 0., 0.

#         for step, batch in enumerate(train_loader):
#             print(batch.keys())
            
#             reset_peak()
#             with peak_region():
#                 loss = model(batch)
#                 if torch.isnan(loss):
#                     print(f"NaN loss at step {step}")
#                     print(batch)
#                     raise  # Skip this batch
            
#             epoch_loss += loss.item()
#             accum_loss += loss.item()
#             print('forward mem:')
#             print_snapshot()
#             with peak_region():
#                 (loss / args.grad_steps).backward()
#             print('backward mem:')
#             print_snapshot()

            

#             if (step + 1) % args.grad_steps == 0:
#                 clip_grad_norm_(optimizer.param_groups[0]['params'], 1.0)
#                 # adjust_learning_rate(optimizer.param_groups[0], step / len(train_loader) + epoch, args)
#                 adjust_lr_step(
#                     optimizer.param_groups[0],
#                     update_step=global_update_step,
#                     total_steps=total_update_steps,
#                     base_lr=args.lr,
#                     min_lr=args.min_lr,
#                     warmup_steps=warmup_steps,
#                 )
#                 optimizer.step()
#                 optimizer.zero_grad()
#                 global_update_step += 1

#                 #logging
#                 lr = optimizer.param_groups[0]["lr"]
#                 wandb.log({'Lr': lr})
#                 wandb.log({'Accum Loss': accum_loss / args.grad_steps})
#                 accum_loss = 0.

            
#             # Block for inspect memories
#             print_snapshot()

#             progress_bar.update(1)

#         print(f"Epoch: {epoch}|{args.num_epochs}: Train Loss (Epoch Mean): {epoch_loss / len(train_loader)}")
#         wandb.log({'Train Loss (Epoch Mean)': epoch_loss / len(train_loader)})

        
#         val_loss = 0.
#         model.eval()
#         with torch.no_grad():
#             for step, batch in enumerate(val_loader):
#                 loss = model(batch)
#                 val_loss += loss.item()
#             val_loss = val_loss/len(val_loader)
#             print(f"Epoch: {epoch}|{args.num_epochs}: Val Loss: {val_loss}")
#             wandb.log({'Val Loss': val_loss})


#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             _save_checkpoint(model, optimizer, epoch, job_id,time,  args, is_best=True)
#             best_epoch = epoch

#         print(f'Epoch {epoch} Val Loss {val_loss} Best Val Loss {best_val_loss} Best Epoch {best_epoch}')


#         if epoch - best_epoch >= args.patience:
#             print(f'Early stop at epoch {epoch}')
#             break
#     print('Finish training !')
#     torch.cuda.empty_cache()
#     torch.cuda.reset_max_memory_allocated()

    
    
#     # Step 5. Evaluating and post processing for each dataset
#     model = _reload_best_model_search(model, job_id,time, args)        
#     model.eval()

#     #Get result for each dataset: 
#     for ds_name in test_loaders.keys():
#         test_loader = test_loaders[ds_name]
#         eval_outputs = [] 
#         progress_bar_test = tqdm(range(len(test_loader)))
#         for step, batch in enumerate(test_loader):
#             with torch.no_grad():
#                 output = model.inference(batch)
#                 eval_outputs.append(output)
        
#             progress_bar_test.update(1)
        
#         prefix = ds_name.split("_")[0]
#         post_processing_func[prefix](eval_outputs, args, job_id, time)





# if __name__ == "__main__":
#     os.environ["TOKENIZERS_PARALLELISM"] = "false"

#     args = parse_args_llama()

#     main(args)
#     torch.cuda.empty_cache()
#     torch.cuda.reset_max_memory_allocated()
#     gc.collect()
