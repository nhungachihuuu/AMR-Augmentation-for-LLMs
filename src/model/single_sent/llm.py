import contextlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import math

ignore_index = -100


class LLM_single(torch.nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.max_txt_len = args.max_txt_len
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
            model = get_peft_model(model, config)

        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        self.model = model
        print('Finish loading LLAMA!')

    @property
    def device(self):
        return list(self.parameters())[0].device

    def _apply_chat_template(self, messages, **kwargs):
        try:
            return self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            # Llama and other models don't have this parameter
            return self.tokenizer.apply_chat_template(messages, **kwargs)


    def maybe_autocast(self, dtype=torch.bfloat16):
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def _build_inputs(self, samples, include_response: bool):
        """
        Tokenize a batch using apply_chat_template.

        When include_response=True  (training / perplexity):
            full conversation is tokenised; labels mask out the prompt prefix.
        When include_response=False (inference):
            only the prompt up to the assistant turn is tokenised.

        Returns:
            input_ids      : (B, L) LongTensor
            attention_mask : (B, L) LongTensor
            labels         : (B, L) LongTensor  or  None when include_response=False
        """
        batch_size = len(samples['idx'])

        all_input_ids = []
        all_labels = []

        for i in range(batch_size):
            messages = [
                {"role": "system",    "content": samples["prompt"][i]},
                {"role": "user",      "content": samples["user"][i]},
            ]

            if include_response:
                # Full conversation — we need the token length of the prompt
                # portion so we can mask it out in labels.
                prompt_ids = self._apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,   # adds the assistant header
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

            # print(f"sample {i}")
            # print("prefix decoded:", self.tokenizer.decode(full_ids[:len(prompt_ids)]))
            # print("response decoded:", self.tokenizer.decode(full_ids[len(prompt_ids):]))
            # print("lengths match?", full_ids[:len(prompt_ids)] == prompt_ids)
            # print("-------------------------------")
            else:
                # Inference — just the prompt up to the assistant turn
                prompt_ids = self._apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors=None,
                )
                all_input_ids.append(prompt_ids)



        # Pad to the same length within the batch
        max_len = max(len(ids) for ids in all_input_ids)
        pad_id = self.tokenizer.pad_token_id

        input_ids_tensor = []
        attention_mask_tensor = []
        labels_tensor = [] if include_response else None

        for i in range(batch_size):
            ids = all_input_ids[i]
            pad_len = max_len - len(ids)

            if include_response:
                # Right-pad (training)
                padded_ids = ids + [pad_id] * pad_len
                mask = [1] * len(ids) + [0] * pad_len
                lbl = all_labels[i] + [ignore_index] * pad_len
                labels_tensor.append(lbl)
            else:
                # Left-pad (inference, consistent with tokenizer.padding_side)
                padded_ids = [pad_id] * pad_len + ids
                mask = [0] * pad_len + [1] * len(ids)

            input_ids_tensor.append(padded_ids)
            attention_mask_tensor.append(mask)

        input_ids = torch.tensor(input_ids_tensor, dtype=torch.long).to(self.device)
        attention_mask = torch.tensor(attention_mask_tensor, dtype=torch.long).to(self.device)
        labels = torch.tensor(labels_tensor, dtype=torch.long).to(self.device) if include_response else None

        return input_ids, attention_mask, labels

    # ------------------------------------------------------------------
    # Training forward pass
    # ------------------------------------------------------------------
    def forward(self, samples):
        input_ids, attention_mask, labels = self._build_inputs(samples, include_response=True)
        max_length = input_ids.shape[1]
        print(f"new batch, max len {max_length}")

        with self.maybe_autocast():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
        return outputs.loss

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def inference(self, samples):
        input_ids, attention_mask, _ = self._build_inputs(samples, include_response=False)

        # Keep a decoded copy of the prompt for logging
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

        # Decode only the newly generated tokens (strip the prompt)
        new_tokens = outputs[:, input_ids.shape[1]:]
        pred = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        return {
            'Idx':    samples['idx'],
            'Input':  input_texts,
            'Output': pred,
            'Gold':   samples['assistant'],
        }

    # ------------------------------------------------------------------
    # Perplexity evaluation
    # ------------------------------------------------------------------
    def get_perplexity(self, samples, prompt_type_ppl):
        """Compute perplexity of the NLD response under different prompt types."""

        # --- Build system prompt and user message text ---
        if prompt_type_ppl == "text":
            system_prompt = (
                "You are a semantic analysis assistant. Your task is to deconstruct a sentence into its underlying logical structure, "
                "expressed as plain natural language sentences — no bullet points, no headers, no markdown formatting of any kind.\n"
                "Describe the meaning by identifying the core events or states described, the entities involved (who or what), "
                "and how those entities relate to one another.\n\n"
                "Example:\n\n"
                "    Input Sentence: \"The NBA season of 1975-76 was the 30th season of the National Basketball Association.\"\n\n"
                "    Output: This refers to a season of the National Basketball Association. "
                "It was the 30th season of the league. The season took place during the 1975-76 time period."
            )
            user_texts = [f"Sentence: {x}" for x in samples["sentence1"]]

        elif prompt_type_ppl == "amr":
            system_prompt = (
                "You are a semantic analysis assistant. Your task is to deconstruct a sentence into its underlying logical structure, "
                "expressed as plain natural language sentences — no bullet points, no headers, no markdown formatting of any kind.\n"
                "Describe the meaning by identifying the core events or states described, the entities involved (who or what), "
                "and how those entities relate to one another. You may use the provided AMR (Abstract Meaning Representation) "
                "as a supplement to guide you through generation.\n\n"
                "Example:\n\n"
                "    Input Sentence: \"The NBA season of 1975-76 was the 30th season of the National Basketball Association.\"\n\n"
                "    Input AMR: \"(s / season~2\n    :ord (o / ordinal-entity~10\n        :value 30~9)\n    :poss (l / league~13\n"
                "        :name (n / name~13\n            :op1 \"National\"~13\n            :op2 \"Basketball\"~14\n"
                "            :op3 \"Association\"~15))\n    :time (d3 / date-interval~4\n        :op1 (d / date-entity~4\n"
                "            :year 1975~4)\n        :op2 (d2 / date-entity~6\n            :year 76~6)))\"\n\n"
                "    Output: This refers to a season of the National Basketball Association. "
                "It was the 30th season of the league. The season took place during the 1975-76 time period."
            )
            user_texts = [
                f"Sentence: {x}\nAMR: {y}"
                for x, y in zip(samples["sentence1"], samples["sentence1_amr"])
            ]

        elif prompt_type_ppl == "amr_only":
            system_prompt = (
                "You are a semantic analysis assistant. Your task is to deconstruct an AMR (Abstract Meaning Representation) "
                "into its underlying logical structure, expressed as plain natural language sentences — no bullet points, no headers, no markdown formatting of any kind.\n"
                "Describe the meaning by identifying the core events or states described, the entities involved (who or what), "
                "and how those entities relate to one another.\n\n"
                "Example:\n\n"
                "    Input AMR: \"(s / season~2\n    :ord (o / ordinal-entity~10\n        :value 30~9)\n    :poss (l / league~13\n"
                "        :name (n / name~13\n            :op1 \"National\"~13\n            :op2 \"Basketball\"~14\n"
                "            :op3 \"Association\"~15))\n    :time (d3 / date-interval~4\n        :op1 (d / date-entity~4\n"
                "            :year 1975~4)\n        :op2 (d2 / date-entity~6\n            :year 76~6)))\"\n\n"
                "    Output: This refers to a season of the National Basketball Association. "
                "It was the 30th season of the league. The season took place during the 1975-76 time period."
            )
            user_texts = [f"AMR: {y}" for _, y in zip(samples["sentence1"], samples["sentence1_amr"])]

        elif prompt_type_ppl == "amr_scramble":
            system_prompt = self._amr_system_prompt()
            user_texts = [
                f"Sentence: {x}\nAMR: {y}"
                for x, y in zip(samples["sentence1"], samples["sentence1_amr_scramble"])
            ]

        elif prompt_type_ppl == "amr_only_scramble":
            system_prompt = self._amr_only_system_prompt()
            user_texts = [
                f"AMR: {y}"
                for _, y in zip(samples["sentence1"], samples["sentence1_amr_scramble"])
            ]

        elif prompt_type_ppl == "amr_scramble_edge":
            system_prompt = self._amr_system_prompt()
            user_texts = [
                f"Sentence: {x}\nAMR: {y}"
                for x, y in zip(samples["sentence1"], samples["sentence1_amr_scramble_edge"])
            ]

        elif prompt_type_ppl == "amr_node_list":
            system_prompt = (
                "You are a semantic analysis assistant. Your task is to deconstruct a sentence into its underlying logical structure, "
                "expressed as plain natural language sentences — no bullet points, no headers, no markdown formatting of any kind.\n"
                "Describe the meaning by identifying the core events or states described, the entities involved (who or what), "
                "and how those entities relate to one another. You may use the supplement provided if you find it helpful.\n\n"
                "Example:\n\n"
                "    Input Sentence: \"The NBA season of 1975-76 was the 30th season of the National Basketball Association.\"\n\n"
                "    Supplement: season, ordinal-entity, 30, league, name, \"National\", "
                "\"Basketball\", \"Association\", date-interval, date-entity, 1975, date-entity, 76\n\n"
                "    Output: This refers to a season of the National Basketball Association. "
                "It was the 30th season of the league. The season took place during the 1975-76 time period."
            )
            user_texts = [
                f"Sentence: {x}\nAMR: {y}"
                for x, y in zip(samples["sentence1"], samples["sentence1_amr_node_list"])
            ]

        elif prompt_type_ppl == "amr_random":
            system_prompt = self._amr_system_prompt()
            user_texts = [
                f"Sentence: {x}\nAMR: {y}"
                for x, y in zip(samples["sentence1"], samples["pseudo_amr"])
            ]

        else:
            raise ValueError(f"Unknown prompt_type_ppl: {prompt_type_ppl}")

        # Build a temporary samples dict that reuses _build_inputs
        ppl_samples = {
            "idx":       samples["idx"],
            "prompt":    [system_prompt] * len(samples["idx"]),
            "user":      user_texts,
            "assistant": samples["nld"],
        }

        input_ids, attention_mask, labels = self._build_inputs(ppl_samples, include_response=True)
        max_length = input_ids.shape[1]
        print(f"new batch, max len {max_length}")

        with self.maybe_autocast():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )

        loss = outputs.loss
        perplexity = math.exp(loss.item())
        return perplexity, system_prompt, user_texts[0], samples["nld"][0]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _amr_system_prompt(self):
        return (
            "Task: Deconstruct the following sentence into its underlying logical structure. "
            "Describe the meaning by identifying core events, the entities involved, and how "
            "they relate to one another. You may use the provided AMR (Abstract Meaning "
            "Representation) as a supplement to guide you through generation."
        )

    def _amr_only_system_prompt(self):
        return (
            "Task: Given the Abstract Meaning Representation (AMR), describe its meaning by "
            "identifying core events, the entities involved, and how they relate to one another."
        )

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        return trainable_params, all_param
    


    def ppl_inference(self, samples, prompt_type_ppl):
        """
        Run generation using exactly the same prompt construction as get_perplexity.
        Returns decoded model outputs alongside the prompt and gold labels for inspection.
        """

        # ------------------------------------------------------------------ #
        # Build system prompt and user texts — identical to get_perplexity    #
        # ------------------------------------------------------------------ #
        if prompt_type_ppl == "text":
            system_prompt = (
                "You are a semantic analysis assistant. Your task is to deconstruct a sentence into its underlying logical structure, "
                "expressed as plain natural language sentences — no bullet points, no headers, no markdown formatting of any kind.\n"
                "Describe the meaning by identifying the core events or states described, the entities involved (who or what), "
                "and how those entities relate to one another.\n\n"
                "Example:\n\n"
                "    Input Sentence: \"The NBA season of 1975-76 was the 30th season of the National Basketball Association.\"\n\n"
                "    Output: This refers to a season of the National Basketball Association. "
                "It was the 30th season of the league. The season took place during the 1975-76 time period."
            )
            user_texts = [f"Sentence: {x}" for x in samples["sentence1"]]

        elif prompt_type_ppl == "amr":
            system_prompt = (
                "You are a semantic analysis assistant. Your task is to deconstruct a sentence into its underlying logical structure, "
                "expressed as plain natural language sentences — no bullet points, no headers, no markdown formatting of any kind.\n"
                "Describe the meaning by identifying the core events or states described, the entities involved (who or what), "
                "and how those entities relate to one another. You may use the provided AMR (Abstract Meaning Representation) "
                "as a supplement to guide you through generation.\n\n"
                "Example:\n\n"
                "    Input Sentence: \"The NBA season of 1975-76 was the 30th season of the National Basketball Association.\"\n\n"
                "    Input AMR: \"(s / season~2\n    :ord (o / ordinal-entity~10\n        :value 30~9)\n    :poss (l / league~13\n"
                "        :name (n / name~13\n            :op1 \"National\"~13\n            :op2 \"Basketball\"~14\n"
                "            :op3 \"Association\"~15))\n    :time (d3 / date-interval~4\n        :op1 (d / date-entity~4\n"
                "            :year 1975~4)\n        :op2 (d2 / date-entity~6\n            :year 76~6)))\"\n\n"
                "    Output: This refers to a season of the National Basketball Association. "
                "It was the 30th season of the league. The season took place during the 1975-76 time period."
            )
            user_texts = [
                f"Sentence: {x}\nAMR: {y}"
                for x, y in zip(samples["sentence1"], samples["sentence1_amr"])
            ]

        elif prompt_type_ppl == "amr_only":
            system_prompt = (
                "You are a semantic analysis assistant. Your task is to deconstruct an AMR (Abstract Meaning Representation) "
                "into its underlying logical structure, expressed as plain natural language sentences — no bullet points, no headers, no markdown formatting of any kind.\n"
                "Describe the meaning by identifying the core events or states described, the entities involved (who or what), "
                "and how those entities relate to one another.\n\n"
                "Example:\n\n"
                "    Input AMR: \"(s / season~2\n    :ord (o / ordinal-entity~10\n        :value 30~9)\n    :poss (l / league~13\n"
                "        :name (n / name~13\n            :op1 \"National\"~13\n            :op2 \"Basketball\"~14\n"
                "            :op3 \"Association\"~15))\n    :time (d3 / date-interval~4\n        :op1 (d / date-entity~4\n"
                "            :year 1975~4)\n        :op2 (d2 / date-entity~6\n            :year 76~6)))\"\n\n"
                "    Output: This refers to a season of the National Basketball Association. "
                "It was the 30th season of the league. The season took place during the 1975-76 time period."
            )
            user_texts = [f"AMR: {y}" for _, y in zip(samples["sentence1"], samples["sentence1_amr"])]

        elif prompt_type_ppl == "amr_scramble":
            system_prompt = self._amr_system_prompt()
            user_texts = [
                f"Sentence: {x}\nAMR: {y}"
                for x, y in zip(samples["sentence1"], samples["sentence1_amr_scramble"])
            ]

        elif prompt_type_ppl == "amr_only_scramble":
            system_prompt = self._amr_only_system_prompt()
            user_texts = [
                f"AMR: {y}"
                for _, y in zip(samples["sentence1"], samples["sentence1_amr_scramble"])
            ]

        elif prompt_type_ppl == "amr_scramble_edge":
            system_prompt = self._amr_system_prompt()
            user_texts = [
                f"Sentence: {x}\nAMR: {y}"
                for x, y in zip(samples["sentence1"], samples["sentence1_amr_scramble_edge"])
            ]

        elif prompt_type_ppl == "amr_node_list":
            system_prompt = (
                "You are a semantic analysis assistant. Your task is to deconstruct a sentence into its underlying logical structure, "
                "expressed as plain natural language sentences — no bullet points, no headers, no markdown formatting of any kind.\n"
                "Describe the meaning by identifying the core events or states described, the entities involved (who or what), "
                "and how those entities relate to one another. You may use the supplement provided if you find it helpful.\n\n"
                "Example:\n\n"
                "    Input Sentence: \"The NBA season of 1975-76 was the 30th season of the National Basketball Association.\"\n\n"
                "    Supplement: season, ordinal-entity, 30, league, name, \"National\", "
                "\"Basketball\", \"Association\", date-interval, date-entity, 1975, date-entity, 76\n\n"
                "    Output: This refers to a season of the National Basketball Association. "
                "It was the 30th season of the league. The season took place during the 1975-76 time period."
            )
            user_texts = [
                f"Sentence: {x}\nAMR: {y}"
                for x, y in zip(samples["sentence1"], samples["sentence1_amr_node_list"])
            ]

        elif prompt_type_ppl == "amr_random":
            system_prompt = self._amr_system_prompt()
            user_texts = [
                f"Sentence: {x}\nAMR: {y}"
                for x, y in zip(samples["sentence1"], samples["pseudo_amr"])
            ]

        else:
            raise ValueError(f"Unknown prompt_type_ppl: {prompt_type_ppl}")

        # ------------------------------------------------------------------ #
        # Tokenise (inference mode — left-padded, no labels)                  #
        # ------------------------------------------------------------------ #
        ppl_samples = {
            "idx":       samples["idx"],
            "prompt":    [system_prompt] * len(samples["idx"]),
            "user":      user_texts,
            "assistant": samples["nld"],   # kept so _build_inputs signature is happy
        }

        input_ids, attention_mask, _ = self._build_inputs(ppl_samples, include_response=False)
        input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)

        # ------------------------------------------------------------------ #
        # Generate                                                             #
        # ------------------------------------------------------------------ #

        gen_kwargs = dict(
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0.0,
        )
        with self.maybe_autocast():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                # do_sample=False,
                # use_cache=True,
                **gen_kwargs
            )

        new_tokens = outputs[:, input_ids.shape[1]:]
        pred = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        return {
            "idx":         samples["idx"],
            "prompt":      system_prompt,
            "user":        user_texts,
            "prediction":  pred,
            "gold":        samples["nld"],
        }


    def get_sentence_perplexity(self, samples):
        """
        Compute unconditional PPL on the raw input sentence only.
        No chat template, no prompt, no label — just the sentence tokens.
        
        Returns:
            List of (sentence, ppl) tuples, one per sample in the batch.
        """
        sentences = samples["sentence1"]

        for sentence in sentences:
            encoding = self.tokenizer(
                sentence,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_txt_len,
            )
            input_ids = encoding.input_ids.to(self.device)

            with torch.no_grad():
                with self.maybe_autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        labels=input_ids,
                        return_dict=True,
                    )

            ppl = math.exp(outputs.loss.item())

        return {"input_sentence":sentences[0],
                "ppl" : ppl}