import os 
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import wandb
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset, ConcatDataset

from src.utils.seed import seed_everything
from src.utils.lr_schedule import adjust_lr_step
from torch.nn.utils import clip_grad_norm_
from src.config import parse_args_llama
from src.utils.ckpt import _save_checkpoint, _reload_best_model_search
from src.model import load_model, llm_model_path
from src.dataset import load_dataset
from src.utils.post_processing import post_processing_func
from contextlib import contextmanager
import numpy as np
import math
import datetime
import json


# ------------------------------------------------------------------ #
# Memory utilities
# ------------------------------------------------------------------ #

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


# ------------------------------------------------------------------ #
# Collate
# ------------------------------------------------------------------ #

def mixed_collate(batch):
    keys = set().union(*(sample.keys() for sample in batch))
    out = {}
    for k in keys:
        out[k] = [sample.get(k, "") for sample in batch]
    return out


# ------------------------------------------------------------------ #
# Dataset helpers
# ------------------------------------------------------------------ #

class SubsetWithAttrs(Subset):
    def __getattr__(self, name):
        return getattr(self.dataset, name)


# ------------------------------------------------------------------ #
# Single dataset: train + eval
# ------------------------------------------------------------------ #

def train_and_eval_single_dataset(ds_name, args, time, job_id):
    """Full training + evaluation pipeline for one dataset, one seed."""

    seed_everything(seed=args.seed)

    # -------- Data ------------------------------------------------- #
    print(f"[{ds_name}] Loading dataset...")

    train, dev, test = load_dataset[ds_name].load(args.prompt_type)
    print(train[0])

    train_indices = np.random.choice(
        len(train), size=min(args.train_subset, len(train)), replace=False
    ).tolist()
    dev_indices  = list(range(min(args.dev_subset,  len(dev))))
    test_indices = list(range(min(args.test_subset, len(test))))

    train_dataset = SubsetWithAttrs(train, train_indices)
    val_dataset   = SubsetWithAttrs(dev,   dev_indices)
    test_dataset  = SubsetWithAttrs(test,  test_indices)

    print(f"[{ds_name}] train={len(train_dataset)}, "
          f"val={len(val_dataset)}, test={len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        drop_last=True,  pin_memory=True, shuffle=True,  collate_fn=mixed_collate
    )
    val_loader = DataLoader(
        val_dataset,   batch_size=args.batch_size,
        drop_last=True,  pin_memory=True, shuffle=False, collate_fn=mixed_collate
    )
    test_loader = DataLoader(
        test_dataset,  batch_size=args.eval_batch_size,
        drop_last=False, pin_memory=True, shuffle=False, collate_fn=mixed_collate
    )

    # -------- Model ------------------------------------------------ #
    torch.cuda.empty_cache()
    gc.collect()
    reset_peak()

    args.llm_model_path = llm_model_path[args.llm_model_name]
    model = load_model[args.model_name](args=args)
    model.to("cuda:0")
    print(f"[{ds_name}] Model loaded:")
    print_snapshot()

    # -------- Optimizer & LR schedule ------------------------------ #
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{'params': params, 'lr': args.lr, 'weight_decay': args.wd}],
        betas=(0.9, 0.95)
    )
    trainable_params, all_param = model.print_trainable_params()
    print(f"[{ds_name}] trainable: {trainable_params} / {all_param} "
          f"({100 * trainable_params / all_param:.2f}%)")

    updates_per_epoch  = math.ceil(len(train_loader) / args.grad_steps)
    total_update_steps = args.num_epochs * updates_per_epoch
    warmup_steps       = min(
        args.warmup_steps,
        int(round(args.warmup_ratio * total_update_steps))
    )
    global_update_step = 0

    # -------- Training loop ---------------------------------------- #
    progress_bar  = tqdm(range(args.num_epochs * len(train_loader)),
                         desc=f"[{ds_name}|seed={args.seed}] training")
    best_val_loss = float('inf')
    best_epoch    = 0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss, accum_loss = 0., 0.

        for step, batch in enumerate(train_loader):
            with peak_region():
                loss = model(batch)
                if torch.isnan(loss):
                    print(f"[{ds_name}] NaN loss at step {step}")
                    raise ValueError("NaN loss encountered")

            epoch_loss += loss.item()
            accum_loss += loss.item()
            (loss / args.grad_steps).backward()

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

                wandb.log({
                    'lr':         optimizer.param_groups[0]["lr"],
                    'accum_loss': accum_loss / args.grad_steps,
                })
                accum_loss = 0.

            progress_bar.update(1)

        mean_train_loss = epoch_loss / len(train_loader)
        print(f"[{ds_name}] Epoch {epoch}/{args.num_epochs}: "
              f"Train Loss = {mean_train_loss:.4f}")
        wandb.log({'train_loss': mean_train_loss, 'epoch': epoch})

        # -- Validation -- #
        val_loss = 0.
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                val_loss += model(batch).item()
        val_loss /= len(val_loader)
        print(f"[{ds_name}] Epoch {epoch}/{args.num_epochs}: "
              f"Val Loss = {val_loss:.4f}")
        wandb.log({'val_loss': val_loss, 'epoch': epoch})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(model, optimizer, epoch, job_id, time, args, is_best=True)
            best_epoch = epoch

        print(f"[{ds_name}] Best Val Loss {best_val_loss:.4f} @ epoch {best_epoch}")

        if epoch - best_epoch >= args.patience:
            print(f"[{ds_name}] Early stop at epoch {epoch}")
            break

    print(f"[{ds_name}] Training finished.")
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

    # -------- Evaluation ------------------------------------------- #
    model = _reload_best_model_search(model, job_id, time, args)
    model.eval()

    eval_outputs = []
    for batch in tqdm(test_loader, desc=f"[{ds_name}|seed={args.seed}] eval"):
        with torch.no_grad():
            output = model.inference(batch)
            eval_outputs.append(output)

    prefix  = ds_name.split("_")[0]
    metrics = post_processing_func[prefix](eval_outputs, args, job_id, time)

    # -------- Cleanup ---------------------------------------------- #
    del model, optimizer, params
    del train_loader, val_loader, test_loader
    del train_dataset, val_dataset, test_dataset
    torch.cuda.empty_cache()
    gc.collect()
    print(f"[{ds_name}] Cleanup done.\n")

    return metrics


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main(args):
    time   = datetime.datetime.today().strftime('%m%d%H%M')
    job_id = os.environ.get("JOB_ID")
    seeds  = args.seeds  # e.g. [42, 43, 44, 45] — add --seeds to your arg parser

    # all_results[ds_name] = [metrics_seed0, metrics_seed1, ...]
    all_results = {ds_name: [] for ds_name in args.dataset}

    for ds_name in args.dataset:
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*60}")

        for seed in seeds:
            args.seed = seed
            print(f"\n{'-'*40}")
            print(f"  Seed: {seed}")
            print(f"{'-'*40}\n")

            wandb.init(
                project=f"{args.project}",
                name=f"{ds_name}_{args.model_name}_{args.llm_model_name}"
                     f"_seed{seed}_{args.lr}_{job_id}_{time}_{args.phase2}",
                config=vars(args),
                reinit=True
            )

            metrics = train_and_eval_single_dataset(ds_name, args, time, job_id)
            all_results[ds_name].append(metrics)

            wandb.finish()

        # ---- Summary across seeds for this dataset ---------------- #
        print(f"\n{'='*60}")
        print(f"  [{ds_name}] Summary across {len(seeds)} seeds")
        print(f"{'='*60}")

        metric_keys = all_results[ds_name][0].keys()
        summary = {}
        for k in metric_keys:
            vals = [r[k] for r in all_results[ds_name]]
            summary[k] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals)),
                "runs": vals,
            }
            print(f"  {k}: mean={summary[k]['mean']:.4f}, "
                  f"std={summary[k]['std']:.4f}  (runs={vals})")

        # Log summary to a dedicated wandb run
        wandb.init(
            project=f"{args.project}",
            name=f"{ds_name}_SUMMARY_{args.model_name}_{args.llm_model_name}_{args.phase2}_{job_id}_{time}",
            config=vars(args),
            reinit=True
        )
        flat_summary = {}
        for k, v in summary.items():
            flat_summary[f"{k}_mean"] = v["mean"]
            flat_summary[f"{k}_std"]  = v["std"]
        wandb.log(flat_summary)
        wandb.finish()

        # Save summary to disk
        summary_path = f"prediction/{ds_name}_summary_{job_id}_{time}.json"
        os.makedirs("prediction", exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary saved to {summary_path}")

    # ---- Final cross-dataset summary ------------------------------ #
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY — ALL DATASETS")
    print(f"{'='*60}")
    for ds_name, seed_results in all_results.items():
        print(f"\n  {ds_name}:")
        for k in seed_results[0].keys():
            vals = [r[k] for r in seed_results]
            print(f"    {k}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args_llama()
    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()