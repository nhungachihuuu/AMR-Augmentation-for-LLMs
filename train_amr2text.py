import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
import wandb
from paths import HOMEPATH, TRAIN_CSV, VAL_CSV, MODEL_LLAMA_DIR, MODEL_QWEN_DIR

# ── argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Fine-tune an LLM for AMR-to-text generation.")
parser.add_argument(
    "--model", type=str, required=True, choices=["qwen", "llama"],
    help="Which base model to use: 'qwen' (Qwen3-8B) or 'llama' (Llama-3.1-8B-Instruct).",
)
args = parser.parse_args()

# ── paths ─────────────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "qwen": {
        "llm_path":   MODEL_QWEN_DIR,
        "output_dir": f"{HOMEPATH}/model_pretrain/qwen_lora_ldc",
        "wandb_name":  "qwen3-8b-amr2text-phase1",
        "wandb_model": "Qwen3-8B",
    },
    "llama": {
        "llm_path":   MODEL_LLAMA_DIR,
        "output_dir": f"{HOMEPATH}/model_pretrain/llama_lora_ldc",
        "wandb_name":  "llama3.1-8b-amr2text-phase2",
        "wandb_model": "Llama-3.1-8B-Instruct",
    },
}

cfg        = MODEL_CONFIGS[args.model]
llm_path   = cfg["llm_path"]
output_dir = cfg["output_dir"]

device = torch.device("cuda:0")


# ── helpers ───────────────────────────────────────────────────────────────────
def gpu_mem_log(label: str):
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved  = torch.cuda.memory_reserved(device)  / 1024**3
    peak      = torch.cuda.max_memory_allocated(device) / 1024**3
    print(f"[GPU MEM | {label}]  allocated: {allocated:.2f} GB  |  reserved: {reserved:.2f} GB  |  peak: {peak:.2f} GB")


# ── tokenizer ─────────────────────────────────────────────────────────────────
llm_tokenizer = AutoTokenizer.from_pretrained(llm_path)
llm_tokenizer.pad_token = llm_tokenizer.eos_token


# ── dataset ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = "Generate a sentence for the given Abstract Meaning Representation."

class AMR2TextDataset(Dataset):
    """
    Reads a CSV with columns `snt` (target sentence) and `amr` (linearised AMR graph),
    and formats each sample using the model's chat template.

    Tokenization strategy differs slightly per model family:
      - Qwen:  requires enable_thinking=False to suppress chain-of-thought tokens
      - Llama: omit enable_thinking=False
    """

    def __init__(self, csv_path: str, tokenizer, max_length: int = 2048, model_type: str = "llama"):
        print(f"Loading data from {csv_path} …")
        df = pd.read_csv(csv_path)
        assert "snt" in df.columns, "Missing column: snt"
        assert "amr" in df.columns, "Missing column: amr"

        df = df.dropna(subset=["snt", "amr"]).reset_index(drop=True)

        self.sentences   = df["snt"].tolist()
        self.amr_texts   = df["amr"].tolist()
        self.tokenizer   = tokenizer
        self.max_length  = max_length
        self.model_type  = model_type
        print(f"  → {len(self.sentences)} samples loaded.")

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        amr_text = str(self.amr_texts[idx])
        target   = str(self.sentences[idx])

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": amr_text},
        ]

        # Qwen requires enable_thinking=False to suppress chain-of-thought tokens;
        # for all other models the kwarg is simply omitted.
        extra_kwargs = {"enable_thinking": False} if self.model_type == "qwen" else {}

        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None,
            **extra_kwargs,
        )
        messages.append({"role": "assistant", "content": target})
        full_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors=None,
            **extra_kwargs,
        )
        full_ids   = full_ids[:self.max_length]
        input_ids  = torch.LongTensor(full_ids)
        prefix_len = min(len(prompt_ids), len(input_ids))

        attention_mask = torch.ones_like(input_ids)

        # Supervise only the assistant response; mask the prompt from the loss
        labels = input_ids.clone()
        labels[:prefix_len] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


# ── collator ──────────────────────────────────────────────────────────────────
class DataCollatorForAMR2Text:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len        = max(x["input_ids"].size(0) for x in batch)
        input_ids_list = []
        attn_mask_list = []
        labels_list    = []

        for x in batch:
            pad_len = max_len - x["input_ids"].size(0)
            input_ids_list.append(
                torch.cat([x["input_ids"],
                           torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
            )
            attn_mask_list.append(
                torch.cat([x["attention_mask"],
                           torch.zeros(pad_len, dtype=torch.long)])
            )
            labels_list.append(
                torch.cat([x["labels"],
                           torch.full((pad_len,), -100, dtype=torch.long)])
            )

        return {
            "input_ids":      torch.stack(input_ids_list),
            "attention_mask": torch.stack(attn_mask_list),
            "labels":         torch.stack(labels_list),
        }


# ── LoRA model ────────────────────────────────────────────────────────────────
def build_lora_model():
    gpu_mem_log("before model load")
    model = AutoModelForCausalLM.from_pretrained(
        llm_path,
        torch_dtype=torch.bfloat16,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()
    gpu_mem_log("after LoRA wrap")
    return model


# ── hyperparameters ───────────────────────────────────────────────────────────
batch_size   = 8
num_epochs   = 10
warmup_ratio = 0.03
lr           = 1e-4
max_length   = 1024

# ── dataloaders ───────────────────────────────────────────────────────────────
dataset_train = AMR2TextDataset(TRAIN_CSV, llm_tokenizer, max_length, model_type=args.model)
dataset_val   = AMR2TextDataset(VAL_CSV,   llm_tokenizer, max_length, model_type=args.model)
collator      = DataCollatorForAMR2Text(llm_tokenizer.pad_token_id)

trainloader = DataLoader(
    dataset_train, batch_size=batch_size, shuffle=True,
    collate_fn=collator, num_workers=2, pin_memory=True, drop_last=True,
)
valloader = DataLoader(
    dataset_val, batch_size=batch_size, shuffle=False,
    collate_fn=collator, num_workers=2, pin_memory=True, drop_last=False,
)

# ── model + optimizer ─────────────────────────────────────────────────────────
os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)

model = build_lora_model().to(device)

num_update_steps_per_epoch = len(trainloader)
max_train_steps            = num_epochs * num_update_steps_per_epoch

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr,
    weight_decay=0.01,
)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(max_train_steps * warmup_ratio),
    num_training_steps=max_train_steps,
)

# ── wandb ─────────────────────────────────────────────────────────────────────
wandb.init(
    project="amr-to-text",
    name=cfg["wandb_name"],
    config={
        "model":          cfg["wandb_model"],
        "peft":           "LoRA",
        "lora_r":         64,
        "lora_alpha":     128,
        "lora_dropout":   0.05,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "epochs":         num_epochs,
        "batch_size":     batch_size,
        "lr":             lr,
        "warmup_ratio":   warmup_ratio,
        "max_length":     max_length,
    },
)


# ── eval ──────────────────────────────────────────────────────────────────────
def evaluate():
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in valloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss  = model(**batch).loss
            total_loss += loss.detach().float().item()
            n += 1
    return total_loss / max(n, 1)


# ── training loop ─────────────────────────────────────────────────────────────
best_val = float("inf")

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for step, batch in enumerate(trainloader):
        torch.cuda.reset_peak_memory_stats()
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        loss = model(**batch).loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )

        optimizer.step()
        lr_scheduler.step()

        epoch_loss += loss.detach().item()

        if step % 10 == 0:
            wandb.log({
                "train/step_loss": loss.detach().item(),
                "train/lr":        lr_scheduler.get_last_lr()[0],
            })
            print(
                f"[Epoch {epoch} | Step {step}/{len(trainloader)}] "
                f"loss={loss.detach().item():.4f}  lr={lr_scheduler.get_last_lr()[0]:.2e}"
            )

    # ── checkpoint ────────────────────────────────────────────────────────────
    ckpt_dir = f"{output_dir}/checkpoints/epoch_{epoch}"
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    llm_tokenizer.save_pretrained(ckpt_dir)

    val_loss       = evaluate()
    avg_train_loss = epoch_loss / len(trainloader)

    wandb.log({
        "epoch":            epoch,
        "train/epoch_loss": avg_train_loss,
        "val/epoch_loss":   val_loss,
    })
    print(f"[Epoch {epoch}] train_loss={avg_train_loss:.4f}  val_loss={val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        best_dir = f"{output_dir}/checkpoints/best"
        os.makedirs(best_dir, exist_ok=True)
        model.save_pretrained(best_dir)
        llm_tokenizer.save_pretrained(best_dir)
        print(f"  ↳ New best val loss: {best_val:.4f} — saved to {best_dir}")

wandb.finish()