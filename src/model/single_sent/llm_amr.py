import contextlib
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

ignore_index = -100


class LLM_AMR_single(torch.nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.max_new_tokens = args.max_new_tokens

        if args.lora == 7:
            lora_r: int = 64
            lora_alpha: int = 128
            lora_dropout: float = 0.05
            lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"]
        else:
            lora_r: int = 8
            lora_alpha: int = 16
            lora_dropout: float = 0.05
            lora_target_modules = ["q_proj", "v_proj"]

        print('Loading LLAMA')
        kwargs = {"revision": "main"}

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.llm_model_path, use_fast=False, revision=kwargs["revision"]
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


        self.tokenizer.padding_side = 'left'

        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            **kwargs
        )
        model.config.use_cache = False

        if args.llm_frozen:
            print("Freezing LLAMA!")
            for param in model.parameters():
                param.requires_grad = False
        else:
            print("Training LLAMA with LORA!")
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            if args.phase2:
                model = PeftModel.from_pretrained(model, args.phase2_checkpoint)
                for name, param in model.named_parameters():
                    if "lora_" in name:
                        param.requires_grad = True
            else:
                model = get_peft_model(model, config)

        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        self.model = model
        print('Finish loading LLAMA!')

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.bfloat16):
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()


    def _apply_chat_template(self, messages, **kwargs):
        try:
            return self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            # Llama and other models don't have this parameter
            return self.tokenizer.apply_chat_template(messages, **kwargs)


    # ------------------------------------------------------------------
    # User message builder — kept intact, just moved here cleanly
    # ------------------------------------------------------------------
    def generate_user_mes_text(self, dataset, train_label, samples, i):
        if dataset == 'paws':
            if train_label == 'amr_text':
                return (f"Sentence 1: {samples['sentence1'][i]}\nSentence 1 AMR: {samples['sentence1_amr_scramble'][i]}\n"
                        f"Sentence 2: {samples['sentence2'][i]}\nSentence 2 AMR: {samples['sentence2_amr_scramble'][i]}\n")
            elif train_label == 'text_only':
                return f"Sentence 1: {samples['sentence1'][i]}\nSentence 2: {samples['sentence2'][i]}\n"
            elif train_label == 'amr_only':
                return f"Sentence 1 AMR: {samples['sentence1_amr'][i]}\nSentence 2 AMR: {samples['sentence2_amr'][i]}\n"

        elif dataset == 'snli':
            if train_label == 'amr_text':
                return (f"Premise: {samples['premise'][i]}\nPremise AMR: {samples['premise_amr'][i]}\n"
                        f"Hypothesis: {samples['hypothesis'][i]}\nHypothesis AMR: {samples['hypothesis_amr'][i]}\n")
            elif train_label == 'text_only':
                return f"Premise: {samples['premise'][i]}\nHypothesis: {samples['hypothesis'][i]}\n"
            elif train_label == 'amr_only':
                return f"Premise AMR: {samples['premise_amr'][i]}\nHypothesis AMR: {samples['hypothesis_amr'][i]}\n"

        elif dataset == 'agnews':
            if train_label == 'amr_text':
                return (f"Title: {samples['title'][i]}\nDescription: {samples['description'][i]}\n"
                        f"Description AMR: {samples['description_amr'][i]}\n")
            elif train_label == 'text_only':
                return f"Title: {samples['title'][i]}\nDescription: {samples['description'][i]}\n"
            elif train_label == 'amr_only':
                return f"Title: {samples['title'][i]}\nDescription AMR: {samples['description_amr'][i]}\n"

        elif dataset == 'sst':
            if train_label == 'amr_text':
                return f"Sentence: {samples['sentence'][i]}\nSentence AMR: {samples['sentence_amr'][i]}\n"
            elif train_label == 'text_only':
                return f"Sentence: {samples['sentence'][i]}\n"
            elif train_label == 'amr_only':
                return f"Sentence AMR: {samples['sentence_amr'][i]}\n"

        elif dataset == 'pubmed':
            if train_label == 'amr_text':
                return (f"Sentence: {samples['sentence'][i]}\nSentence AMR: {samples['sentence_amr'][i]}\n"
                        f"Interaction tuple: {samples['interaction_tuple'][i]}\n")
            elif train_label == 'text_only':
                return f"Sentence: {samples['sentence'][i]}\nInteraction tuple: {samples['interaction_tuple'][i]}\n"
            elif train_label == 'amr_only':
                return f"Sentence AMR: {samples['sentence_amr'][i]}\nInteraction tuple: {samples['interaction_tuple'][i]}\n"

        elif dataset == 'wic':
            if train_label == 'amr_text':
                return (f"Target: {samples['target'][i]}\nSentence 1: {samples['sentence1'][i]}\n"
                        f"Sentence 1 AMR: {samples['sentence1_amr'][i]}\n"
                        f"Sentence 2: {samples['sentence2'][i]}\nSentence 2 AMR: {samples['sentence2_amr'][i]}\n")
            elif train_label == 'text_only':
                return (f"Target: {samples['target'][i]}\nSentence 1: {samples['sentence1'][i]}\n"
                        f"Sentence 2: {samples['sentence2'][i]}\n")
            elif train_label == 'amr_only':
                return (f"Target: {samples['target'][i]}\nSentence 1 AMR: {samples['sentence1_amr'][i]}\n"
                        f"Sentence 2 AMR: {samples['sentence2_amr'][i]}\n")

        elif dataset == 'wmt':
            if train_label == 'amr_text':
                return f"Sentence: {samples['sentence'][i]}\nSentence AMR: {samples['sentence_amr'][i]}\n"
            elif train_label == 'text_only':
                return f"Sentence: {samples['sentence'][i]}\n"
            elif train_label == 'amr_only':
                return f"Sentence AMR: {samples['sentence_amr'][i]}\n"

        elif dataset == 'conll':
            if train_label == 'amr_text':
                return (f"Sentence: {samples['sentence'][i]}\nSentence AMR: {samples['sentence_amr'][i]}\n"
                        f"Tokens: {samples['tokens'][i]}\n")
            elif train_label == 'text_only':
                return f"Sentence: {samples['sentence'][i]}\nTokens: {samples['tokens'][i]}\n"
            elif train_label == 'amr_only':
                return f"Sentence AMR: {samples['sentence_amr'][i]}\nTokens: {samples['tokens'][i]}\n"

        elif dataset == 'spider':
            if train_label == 'amr_text':
                return (f"Database: {samples['db_id'][i]}\nSchema description: {samples['schema_description'][i]}\n"
                        f"Question: {samples['question'][i]}\nQuestion AMR: {samples['question_amr'][i]}\n")
            elif train_label == 'text_only':
                return (f"Database: {samples['db_id'][i]}\nSchema description: {samples['schema_description'][i]}\n"
                        f"Question: {samples['question'][i]}\n")
            elif train_label == 'amr_only':
                return (f"Database: {samples['db_id'][i]}\nSchema description: {samples['schema_description'][i]}\n"
                        f"Question AMR: {samples['question_amr'][i]}\n")

        raise ValueError(f"Unknown dataset: {dataset}")

    # ------------------------------------------------------------------
    # Core tokenisation helper — shared by all methods
    # ------------------------------------------------------------------
    def _build_inputs(self, samples, include_response: bool, train_label_per_sample=None):
        """
        Tokenize a batch using apply_chat_template.

        Args:
            samples:                batch dict
            include_response:       True  → full conversation + labels (train / val / ppl)
                                    False → prompt only (inference)
            train_label_per_sample: list[str] | None
                                    If provided, each element is the train_label for that
                                    sample (used in forward / get_perplexity).
                                    If None, 'amr_text' is used for every sample
                                    (used in inference / forward_val).
        Returns:
            input_ids      : (B, L) LongTensor
            attention_mask : (B, L) LongTensor
            labels         : (B, L) LongTensor  or  None when include_response=False
        """
        batch_size = len(samples['idx'])
        all_input_ids = []
        all_labels = []

        for i in range(batch_size):
            t_label = train_label_per_sample[i] if train_label_per_sample else 'amr_text'
            user_text = self.generate_user_mes_text(samples['dataset'][i], t_label, samples, i)

            messages = [
                {"role": "system", "content": samples["prompt"][i]},
                {"role": "user",   "content": user_text},
            ]

            if include_response:
                # Tokenize prompt-only to determine the prefix length for label masking
                prompt_ids = self._apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors=None,
                )
                messages.append({"role": "assistant", "content": samples["assistant"][i]})
                full_ids = self._apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_tensors=None,
                )
                prefix_len = len(prompt_ids)
                label_ids = [ignore_index] * prefix_len + full_ids[prefix_len:]
                all_input_ids.append(full_ids)
                all_labels.append(label_ids)
            else:
                prompt_ids = self._apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors=None,
                )
                all_input_ids.append(prompt_ids)

        # Pad to max length within batch
        max_len = max(len(ids) for ids in all_input_ids)
        pad_id = self.tokenizer.pad_token_id

        input_ids_tensor = []
        attention_mask_tensor = []
        labels_tensor = [] if include_response else None

        for i in range(batch_size):
            ids = all_input_ids[i]
            pad_len = max_len - len(ids)

            if include_response:
                # Right-pad for training
                input_ids_tensor.append(ids + [pad_id] * pad_len)
                attention_mask_tensor.append([1] * len(ids) + [0] * pad_len)
                labels_tensor.append(all_labels[i] + [ignore_index] * pad_len)
            else:
                # Left-pad for inference (matches tokenizer.padding_side = 'left')
                input_ids_tensor.append([pad_id] * pad_len + ids)
                attention_mask_tensor.append([0] * pad_len + [1] * len(ids))

        input_ids = torch.tensor(input_ids_tensor, dtype=torch.long).to(self.device)
        attention_mask = torch.tensor(attention_mask_tensor, dtype=torch.long).to(self.device)
        labels = torch.tensor(labels_tensor, dtype=torch.long).to(self.device) if include_response else None

        return input_ids, attention_mask, labels

    # ------------------------------------------------------------------
    # Training forward  (train_label varies per sample)
    # ------------------------------------------------------------------
    def forward(self, samples):
        input_ids, attention_mask, labels = self._build_inputs(
            samples,
            include_response=True,
            train_label_per_sample=samples['train_label'],
        )
        print(f"new batch, max len {input_ids.shape[1]}")

        with self.maybe_autocast():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
        return outputs.loss

    # ------------------------------------------------------------------
    # Validation forward  (always amr_text)
    # ------------------------------------------------------------------
    def forward_val(self, samples):
        input_ids, attention_mask, labels = self._build_inputs(
            samples,
            include_response=True,
            train_label_per_sample=None,   # defaults to 'amr_text'
        )
        print(f"new batch, max len {input_ids.shape[1]}")

        with self.maybe_autocast():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
        return outputs.loss

    # ------------------------------------------------------------------
    # Inference  (always amr_text, no response)
    # ------------------------------------------------------------------
    def inference(self, samples,):
        input_ids, attention_mask, _ = self._build_inputs(
            samples,
            include_response=False,
            train_label_per_sample=None,   # defaults to 'amr_text'
        )

        input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)

        with self.maybe_autocast():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
                use_cache=True,
            )

        # Decode only the newly generated tokens
        new_tokens = outputs[:, input_ids.shape[1]:]
        pred = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        return {
            'Idx':    samples['idx'],
            'Input':  input_texts,
            'Output': pred,
            'Gold':   samples['assistant'],
        }


    # ------------------------------------------------------------------
    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            n = param.numel()
            all_param += n
            if param.requires_grad:
                trainable_params += n
        return trainable_params, all_param