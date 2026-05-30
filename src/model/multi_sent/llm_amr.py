import contextlib
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

ignore_index = -100


class LLM_amr_multi(torch.nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.max_new_tokens = args.max_new_tokens

        # ── LoRA config ────────────────────────────────────────────────────────
        if args.lora == 7:
            lora_r, lora_alpha, lora_dropout = 64, 128, 0.05
            lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
        else:
            lora_r, lora_alpha, lora_dropout = 8, 16, 0.05
            lora_target_modules = ["q_proj", "v_proj"]

        # ── Tokenizer ─────────────────────────────────────────────────────────
        print("Loading LLM ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.llm_model_path, use_fast=False
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        # ── Base model ────────────────────────────────────────────────────────
        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            revision="main",
        )
        model.config.use_cache = False

        # ── LoRA / freeze ─────────────────────────────────────────────────────
        if args.llm_frozen:
            print("Freezing LLM weights.")
            for param in model.parameters():
                param.requires_grad = False
        else:
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            if args.phase2:
                print(f"Phase-2: loading LoRA adapter from {args.phase2_checkpoint}")
                model = PeftModel.from_pretrained(model, args.phase2_checkpoint)
                for name, param in model.named_parameters():
                    if "lora_" in name:
                        param.requires_grad = True
            else:
                print("Attaching fresh LoRA adapters.")
                model = get_peft_model(model, lora_cfg)

        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
        self.model = model
        print("LLM loaded.")

        # Detect once at init whether this tokenizer supports enable_thinking (Qwen).
        # Avoids try/except overhead on every single sample during training.
        try:
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "test"}],
                enable_thinking=False,
                tokenize=False,
            )
            self._thinking_supported = True
            print("Qwen thinking mode detected -- will disable during template calls.")
        except TypeError:
            self._thinking_supported = False

    # ── device / autocast helpers ─────────────────────────────────────────────

    @property
    def device(self):
        return next(self.parameters()).device

    def maybe_autocast(self, dtype=torch.bfloat16):
        if self.device != torch.device("cpu"):
            return torch.cuda.amp.autocast(dtype=dtype)
        return contextlib.nullcontext()

    # ── chat-template wrapper ─────────────────────────────────────────────────

    def _apply_chat_template(self, messages, **kwargs):
        """Thin wrapper that disables Qwen thinking mode when supported."""
        if self._thinking_supported:
            return self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    # ── text-building helpers ─────────────────────────────────────────────────

    @staticmethod
    def _build_passage(sentences: list, amrs: list) -> str:
        """
        Interleave sentences and their AMR graphs:
            sent1 <amr> amr1 </amr> sent2 <amr> amr2 </amr> ...
        """
        if len(sentences) != len(amrs):
            raise AssertionError(
                f"Sentence/AMR count mismatch: {len(sentences)} sentences vs {len(amrs)} AMRs"
            )
        return " ".join(f"{s} <amr> {a} </amr>" for s, a in zip(sentences, amrs))

    @staticmethod
    def _build_user_content(passage: str, task_info, is_cnn: bool) -> str:
        """CNN gets the passage only; other datasets get passage + task header."""
        if is_cnn:
            return passage
        return f"\n-Passage:\n{passage}\n{task_info}"

    # ── core input builder (shared by forward / inference) ────────────────────

    def _build_inputs(self, samples: dict, include_labels: bool):
        """
        Returns (input_ids, attention_mask, label_ids).
        label_ids is None when include_labels=False (inference).

        Training strategy:
          prompt_ids -- system + user + generation-prompt header
          full_ids   -- same conversation with assistant turn appended
          label_ids  -- full_ids with the prompt prefix masked to ignore_index
        """
        batch_size = len(samples["index"])
        all_input_ids, all_attn_mask, all_labels = [], [], []

        for i in range(batch_size):
            sentences = samples["sentence"][i]
            amrs      = samples["amr"][i]
            is_cnn    = samples["dataset"][i] == "cnn"
            task_info = None if is_cnn else samples["task_infos"][i]

            passage      = self._build_passage(sentences, amrs)
            user_content = self._build_user_content(passage, task_info, is_cnn)

            messages = [
                {"role": "system", "content": samples["prompt"][i]},
                {"role": "user",   "content": user_content},
            ]

            # Prompt-only ids (ends with the generation-prompt header)
            prompt_ids = self._apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_tensors=None,
            )

            if include_labels:
                # Full conversation -- let the tokenizer handle eot and all
                # special tokens correctly for whichever model family is in use
                full_messages = messages + [
                    {"role": "assistant", "content": samples["assistant"][i]}
                ]
                full_ids = self._apply_chat_template(
                    full_messages, add_generation_prompt=False, tokenize=True, return_tensors=None,
                )
                # Mask the prompt prefix; only the assistant response contributes to loss
                label_ids = [ignore_index] * len(prompt_ids) + full_ids[len(prompt_ids):]
                input_ids = full_ids
            else:
                input_ids = prompt_ids
                label_ids = None

            all_input_ids.append(input_ids)
            all_attn_mask.append([1] * len(input_ids))
            if include_labels:
                all_labels.append(label_ids)

        # ── left-pad to uniform length ────────────────────────────────────────
        max_len = max(len(x) for x in all_input_ids)
        pad_id  = self.tokenizer.pad_token_id

        for i in range(batch_size):
            pad_len = max_len - len(all_input_ids[i])
            if pad_len:
                all_input_ids[i] = [pad_id]       * pad_len + all_input_ids[i]
                all_attn_mask[i] = [0]             * pad_len + all_attn_mask[i]
                if include_labels:
                    all_labels[i] = [ignore_index] * pad_len + all_labels[i]

        input_ids      = torch.tensor(all_input_ids, device=self.device)
        attention_mask = torch.tensor(all_attn_mask,  device=self.device)
        label_ids      = torch.tensor(all_labels,     device=self.device) if include_labels else None

        return input_ids, attention_mask, label_ids

    # ── forward (training) ────────────────────────────────────────────────────

    def forward(self, samples: dict) -> torch.Tensor:
        input_ids, attention_mask, label_ids = self._build_inputs(
            samples, include_labels=True
        )
        print(f"batch max_len = {input_ids.shape[1]}")

        with self.maybe_autocast():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=label_ids,
                return_dict=True,
            )
        return outputs.loss

    # ── inference ─────────────────────────────────────────────────────────────

    def inference(self, samples: dict) -> dict:

        patched = samples

        input_ids, attention_mask, _ = self._build_inputs(patched, include_labels=False)

        with self.maybe_autocast():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
                use_cache=True,
            )

        # Decode only the newly generated tokens (strip the prompt)
        gen_ids    = outputs[:, input_ids.shape[1]:]
        pred       = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        input_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)

        result = {
            "Index":    samples["index"],
            "Doc_keys": samples["doc_keys"],
            "Input":    input_text,
            "Output":   pred,
            "Gold":     samples["assistant"],
        }
        if "event_id" in samples:
            result["Event_id"] = samples["event_id"]
        if "chunk_id" in samples:
            result["Chunk_id"] = samples["chunk_id"]
        return result

    # ── utils ─────────────────────────────────────────────────────────────────

    def print_trainable_params(self) -> tuple:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        return trainable, total
