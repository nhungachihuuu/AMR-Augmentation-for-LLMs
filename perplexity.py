#must run with batch = 1 

import pandas as pd
import torch
import wandb
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.utils.seed import seed_everything
from src.config import parse_args_llama
from src.utils.ckpt import _reload_best_model_search
from src.model import load_model, llm_model_path
from src.dataset import load_dataset
import datetime
import os
from contextlib import contextmanager
import numpy as np

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


def build_test_loaders(args):
    """Build test datasets and dataloaders. Called once and reused across seeds."""
    print(f"create dataset: {args.dataset}")

    test_datasets = {}
    len_test_dataset = []
    for ds_name in args.dataset:
        train, val, test = load_dataset[ds_name].load(args.prompt_type)
        print("first sample:")
        print(test[0])
        print('length of dataset')
        print(len(test))
        len_test_dataset.append(len(test))
        test_datasets[ds_name] = test

    print("create dataset successfully")

    test_loaders = {}
    for ds_name in test_datasets.keys():
        test_dataset = test_datasets[ds_name]
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            drop_last=False,
            pin_memory=True,
            shuffle=False,
        )
        test_loaders[ds_name] = test_loader

    print("create dataloader successfully")
    return test_loaders


def run_single_seed(model, test_loaders, args):
    """
    Run evaluation for a single seed.
    Returns:
        mean_ppl_per_ds  : dict {ds_name: float}
        rows             : list of dicts (one per sample, all datasets)
    """
    model.eval()
    rows = []
    mean_ppl_per_ds = {}

    for ds_name, test_loader in test_loaders.items():
        perplexities = []
        progress_bar = tqdm(range(len(test_loader)), desc=f"  [{ds_name}]")
        for step, batch in enumerate(test_loader):
            with torch.no_grad():
                perplexity, prompt, user_mes_text, labels = model.get_perplexity(
                    batch, args.prompt_type_ppl
                )
                perplexities.append(perplexity)
                rows.append({
                    "dataset":     ds_name,
                    "prompt":      prompt,
                    "user_message": user_mes_text,
                    "label":       labels,
                    "perplexity":  perplexity,
                })
            progress_bar.update(1)

        mean_ppl_per_ds[ds_name] = float(np.mean(perplexities))
        print(f"  [{ds_name}] mean perplexity: {mean_ppl_per_ds[ds_name]:.4f}")

    return mean_ppl_per_ds, rows


def main(args):
    time_inf  = datetime.datetime.today().strftime('%m%d%H%M')
    job_id_inf = os.environ.get("JOB_ID")

    # ------------------------------------------------------------------ #
    #  Build dataloaders ONCE — they are shared across all seeds           #
    # ------------------------------------------------------------------ #
    test_loaders = build_test_loaders(args)

    args.llm_model_path = llm_model_path[args.llm_model_name]

    # ------------------------------------------------------------------ #
    #  Single wandb run covering all seeds                                 #
    # ------------------------------------------------------------------ #
    with wandb.init(
        project=args.project,
        name=(
            f"{args.dataset}_{args.llm_model_name}_{args.note}_"
            f"{args.job_id}_{args.time}_{args.prompt_type_ppl}"
        ),
        notes=f"{job_id_inf}_{time_inf}",
        config=args,
    ) as run:

        # seed -> {ds_name: mean_ppl}
        all_seed_results: dict[int, dict[str, float]] = {}
        all_rows = []  # every sample from every seed

        for seed in range(5):  # seeds 0, 1, 2, 3, 4
            print(f"\n{'='*60}")
            print(f"  Running seed {seed}")
            print(f"{'='*60}")

            # ---- set seed -------------------------------------------- #
            args.seed = seed
            seed_everything(seed=seed)

            # ---- fresh model for this seed ---------------------------- #
            torch.cuda.empty_cache()
            reset_peak()
            model = load_model[args.model_name](args=args)
            model = _reload_best_model_search(model, args.job_id, args.time, args)  # ← add this
            model.to("cuda:0")
            print_snapshot()

            # ---- evaluate -------------------------------------------- #
            mean_ppl_per_ds, rows = run_single_seed(model, test_loaders, args)

            # tag each row with its seed
            for r in rows:
                r["seed"] = seed
            all_rows.extend(rows)
            all_seed_results[seed] = mean_ppl_per_ds

            # ---- log per-seed metrics --------------------------------- #
            seed_log = {f"seed{seed}/{ds_name}/mean_perplexity": v
                        for ds_name, v in mean_ppl_per_ds.items()}
            seed_log[f"seed{seed}/overall_mean_perplexity"] = float(
                np.mean(list(mean_ppl_per_ds.values()))
            )
            run.log(seed_log)

            # ---- free GPU memory before next seed -------------------- #
            del model
            torch.cuda.empty_cache()
            gc.collect()

        # -------------------------------------------------------------- #
        #  Aggregate across seeds                                          #
        # -------------------------------------------------------------- #
        ds_names = list(test_loaders.keys())
        summary_log = {}

        for ds_name in ds_names:
            seed_ppls = [all_seed_results[s][ds_name] for s in range(4)]
            mean_ppl  = float(np.mean(seed_ppls))
            std_ppl   = float(np.std(seed_ppls))
            summary_log[f"summary/{ds_name}/mean_perplexity"] = mean_ppl
            summary_log[f"summary/{ds_name}/std_perplexity"]  = std_ppl
            print(f"[{ds_name}] mean±std over seeds: {mean_ppl:.4f} ± {std_ppl:.4f}")

        # Overall (across all datasets and seeds)
        all_per_seed_means = [
            np.mean(list(all_seed_results[s].values())) for s in range(4)
        ]
        overall_mean = float(np.mean(all_per_seed_means))
        overall_std  = float(np.std(all_per_seed_means))
        summary_log["summary/overall_mean_perplexity"] = overall_mean
        summary_log["summary/overall_std_perplexity"]  = overall_std
        print(f"\nOverall mean±std over seeds: {overall_mean:.4f} ± {overall_std:.4f}")

        run.log(summary_log)

        # -------------------------------------------------------------- #
        #  Log full predictions table (all seeds, all datasets)           #
        # -------------------------------------------------------------- #
        df = pd.DataFrame(all_rows)
        run.log({"predictions_table": wandb.Table(dataframe=df)})


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = parse_args_llama()

    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()