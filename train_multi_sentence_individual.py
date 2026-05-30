import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import gc
import datetime
import math

import numpy as np
import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torch.nn.utils import clip_grad_norm_

from src.config import parse_args_llama
from src.dataset import load_dataset
from src.model import load_model, llm_model_path
from src.utils.ckpt import _save_checkpoint, _reload_best_model_search
from src.utils.evaluate import eval_funcs
from src.utils.lr_schedule import adjust_lr_step
from src.utils.post_processing import post_processing_func
from src.utils.seed import seed_everything


# ── memory helpers ────────────────────────────────────────────────────────────

def print_snapshot():
    torch.cuda.synchronize()
    print(
        f"alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
        f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB  "
        f"peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB"
    )

def reset_peak():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


# ── dataset helpers ───────────────────────────────────────────────────────────

class SubsetWithAttrs(Subset):
    """Subset that forwards attribute access to the wrapped dataset."""
    def __getattr__(self, name):
        return getattr(self.dataset, name)


def _filtered_indices(ds, max_sentences):
    """Return relative indices whose 'sentence' list is shorter than max_sentences."""
    if hasattr(ds, "selected_indices") and hasattr(ds, "text") and "sentence" in ds.text:
        return [
            rel for rel, orig in enumerate(ds.selected_indices)
            if ds.text["sentence"][orig] is not None
            and len(ds.text["sentence"][orig]) < max_sentences
        ]
    return [
        i for i in range(len(ds))
        if (item := ds[i]).get("sentence") is not None
        and len(item["sentence"]) < max_sentences
    ]


def _make_subset(ds, max_sentences, max_size, sequential=False):
    if max_sentences:
        keep = _filtered_indices(ds, max_sentences)
    else:
        keep = list(range(len(ds)))

    if sequential:
        indices = keep[:max_size]
    else:
        indices = np.random.choice(keep, size=min(max_size, len(keep)), replace=False).tolist()

    return SubsetWithAttrs(ds, indices)


def mixed_collate(batch):
    """Collate heterogeneous dicts — fills missing keys with empty string."""
    keys = set().union(*(s.keys() for s in batch))
    return {k: [s.get(k, "") for s in batch] for k in keys}


# ── per-dataset training run ──────────────────────────────────────────────────

def run_dataset(ds_name, args, job_id, time_tag):
    """Full train → validate → test cycle for a single dataset."""

    print(f"\n{'='*60}")
    print(f"  Dataset: {ds_name}")
    print(f"{'='*60}\n")

    # Tag used for checkpoints and W&B — unique per dataset + job
    run_tag = f"{ds_name}_{args.model_name}_{args.llm_model_name}_{job_id}_{time_tag}_seed{args.seed}_{args.lr}_{args.phase2}"

    wandb.init(
        project=args.project,
        name=run_tag,
        config={**vars(args), "current_dataset": ds_name},
        reinit=True,
    )

    # ── datasets ──────────────────────────────────────────────────────────────


    train_ds, dev_ds, test_ds = load_dataset[ds_name].load(args.prompt_type)

    train_subset = _make_subset(train_ds, args.max_sentences, args.train_subset, sequential=False)
    dev_subset   = _make_subset(dev_ds,   args.max_sentences, args.dev_subset,   sequential=True)
    test_subset  = _make_subset(test_ds,  args.max_sentences, args.test_subset,  sequential=True)

    print(f"Sizes — train: {len(train_subset)}, val: {len(dev_subset)}, test: {len(test_subset)}")

    # ── dataloaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size,
        shuffle=True, drop_last=True, pin_memory=True, collate_fn=mixed_collate,
    )
    val_loader = DataLoader(
        dev_subset, batch_size=args.eval_batch_size,
        shuffle=False, drop_last=True, pin_memory=True, collate_fn=mixed_collate,
    )
    test_loader = DataLoader(
        test_subset, batch_size=args.eval_batch_size,
        shuffle=False, drop_last=False, pin_memory=True, collate_fn=mixed_collate,
    )

    # ── model ─────────────────────────────────────────────────────────────────
    print("Building model …")
    args.llm_model_path = llm_model_path[args.llm_model_name]
    torch.cuda.empty_cache()
    reset_peak()

    model = load_model[args.model_name](args=args)
    model = model.to("cuda")
    print_snapshot()

    # ── optimiser ─────────────────────────────────────────────────────────────
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": params, "lr": args.lr, "weight_decay": args.wd}],
        betas=(0.9, 0.95),
    )

    trainable, total = model.print_trainable_params()
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    print_snapshot()

    # ── LR schedule bookkeeping ───────────────────────────────────────────────
    updates_per_epoch  = math.ceil(len(train_loader) / args.grad_steps)
    total_update_steps = args.num_epochs * updates_per_epoch
    warmup_steps       = max(1, round(0.03 * total_update_steps))
    global_update_step = 0

    # ── training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_epoch    = 0
    progress_bar  = tqdm(total=args.num_epochs * len(train_loader), desc=f"training [{ds_name}]")

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = accum_loss = 0.0

        for step, batch in enumerate(train_loader):
            loss = model(batch)
            if torch.isnan(loss):
                raise RuntimeError(f"NaN loss at epoch {epoch}, step {step}")

            epoch_loss += loss.item()
            accum_loss += loss.item()

            (loss / args.grad_steps).backward()

            if (step + 1) % args.grad_steps == 0:
                clip_grad_norm_(params, 1.0)
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
                    "lr": optimizer.param_groups[0]["lr"],
                    "accum_loss": accum_loss / args.grad_steps,
                    "global_update_step": global_update_step,
                })
                accum_loss = 0.0

            progress_bar.update(1)

        mean_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch}/{args.num_epochs} — train loss: {mean_train_loss:.4f}")
        wandb.log({"train_loss_epoch": mean_train_loss, "epoch": epoch})

        # ── validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                val_loss += model(batch).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch}/{args.num_epochs} — val loss: {val_loss:.4f} "
              f"(best: {best_val_loss:.4f} @ epoch {best_epoch})")
        wandb.log({"val_loss": val_loss, "epoch": epoch})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            # Pass ds_name so checkpoints are namespaced per dataset
            _save_checkpoint(model, optimizer, epoch, job_id, f"{ds_name}_{time_tag}", args, is_best=True)

        if epoch - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    progress_bar.close()
    print(f"Training complete for {ds_name}.")
    torch.cuda.empty_cache()

    # ── evaluation ────────────────────────────────────────────────────────────
    model = _reload_best_model_search(model, job_id, f"{ds_name}_{time_tag}", args)
    model.eval()

    eval_outputs = []
    for batch in tqdm(test_loader, desc=f"inference [{ds_name}]"):
        with torch.no_grad():
            eval_outputs.append(model.inference(batch))

    prefix = ds_name.split("_")[0]
    post_processing_func[prefix](eval_outputs, args, job_id, f"{ds_name}_{time_tag}")

    wandb.finish()

    # Free GPU memory before the next dataset run
    del model, optimizer, params
    torch.cuda.empty_cache()
    gc.collect()


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    time_tag = datetime.datetime.today().strftime("%m%d%H%M")
    job_id   = os.environ.get("JOB_ID", "local")

    seed_everything(args.seed)

    print(args)
    print(f"Running {len(args.dataset)} dataset(s) independently: {args.dataset}")

    for ds_name in args.dataset:
        run_dataset(ds_name, args, job_id, time_tag)

    print("\nAll datasets complete.")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args_llama()
    main(args)
    torch.cuda.empty_cache()
    gc.collect()