# On the (In)effectiveness of AMR Augmentation for Large Language Models

This repository contains the code and data for our EMNLP 2026 submission: **"On the (In)effectiveness of AMR Augmentation for Large Language Models"**.

We investigate whether augmenting LLM inputs with Abstract Meaning Representation (AMR) improves downstream performance. Through experiments across two model families (Llama-3.1-8B-Instruct and Qwen3-8B), thirteen tasks spanning single- and multi-sentence settings, and multiple fine-tuning strategies, we find no consistent improvement over text-only baselines. A perplexity-based relational knowledge analysis further provides evidence suggesting that AMR augmentation does not provides relational knowledge beyond what LLM can infer from text alone. 

---

## Table of Contents

- [Environment Setup](#environment-setup)
- [Datasets](#datasets)
- [Single-Sentence Tasks](#single-sentence-tasks)
- [Multi-Sentence Tasks](#multi-sentence-tasks)
- [AMR-to-Text Intermediate Training](#amr-to-text-intermediate-training)
- [Perplexity Analysis](#perplexity-analysis)
- [Citation](#citation)

---

## Environment Setup

### 1. Create conda environment

```bash
conda create -n amr_aug python=3.10
conda activate amr_aug
```

### 2. Install PyTorch (CUDA 12.4)

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install flash-attention

```bash
pip install flash-attn --no-build-isolation
```

### 5. Configure paths

Open `path.py` and set `HOMEPATH` to the root directory of this repository.

---

## Datasets

AMR-augmented versions of all datasets used in our experiments are available for download at [OSF (anonymous link)](https://osf.io/3vu2q/overview?view_only=69f398f59ba348b3b1a86139587a3670). Download and extract the zip file into the dataset/ folder at the root of this repository.

> **Licensing notice:** The datasets used in this work are redistributed here solely for research reproducibility. Each dataset is subject to its own license — please see [`DATASET_LICENSES.md`](DATASET_LICENSES.md) for details on all 13 datasets and their respective terms of use before using this data.

---

## Single-Sentence Tasks

We experiment with nine single-sentence datasets: AG News, CoNLL-2003, PAWS, PubMed45, SNLI, SPIDER, SST-2, WiC, and WMT16.

**Key arguments:**

| Argument | Options | Description |
|---|---|---|
| `--model_name` | `llm_amr_single`, `llm_single` | Use `llm_amr_single` for AMR-augmented, `llm_single` for text-only |
| `--prompt_type` | `user_amr`, `user` | Must match `model_name`: use `user_amr` with `llm_amr_single` and `user` with `llm_single` |
| `--llm_model_name` | `qwen-8b`, `llama-8b` | Toggle between model families |

### Individual fine-tuning

```bash
python train_single_sentence_individual.py \
    --dataset agnews conll paws pubmed snli spider sst wic wmt \
    --model_name llm_amr_single \
    --llm_model_name qwen-8b \
    --prompt_type user_amr
```

### Joint fine-tuning


```bash
python train_single_sentence_joint.py \
    --dataset agnews conll paws pubmed snli spider sst wic wmt \
    --model_name llm_amr_single \
    --llm_model_name qwen-8b \
    --prompt_type user_amr
```

---

## Multi-Sentence Tasks

We experiment with four multi-sentence datasets: RAMS, CNN/DailyMail, ANLI, and LogiQA.

**Key arguments:**

| Argument | Options | Description |
|---|---|---|
| `--model_name` | `llm_amr_multi`, `llm_multi` | Use `llm_amr_multi` for AMR-augmented, `llm_multi` for text-only |
| `--prompt_type` | `user_amr_seq`, `user` | Must match `model_name`: use `user_amr_seq` with `llm_amr_multi` and `user` with `llm_multi` |
| `--llm_model_name` | `qwen-8b`, `llama-8b` | Toggle between model families |

### Individual fine-tuning

```bash
python train_multi_sentence_individual.py \
    --model_name llm_multi \
    --llm_model_name llama-8b \
    --prompt_type user \
    --dataset rams cnn logiqa anli
```

### Joint fine-tuning

```bash
python train_multi_sentence_joint.py \
    --model_name llm_multi \
    --llm_model_name llama-8b \
    --prompt_type user \
    --dataset rams cnn logiqa anli
```

---

## AMR-to-Text Intermediate Training

We use the [AMR 3.0 corpus (LDC2020T02)](https://catalog.ldc.upenn.edu/LDC2020T02) for the intermediate AMR-to-text fine-tuning phase. Due to LDC licensing restrictions, this data cannot be redistributed here. To reproduce this step:

1. Obtain the AMR 3.0 corpus from the [LDC catalog](https://catalog.ldc.upenn.edu/LDC2020T02).
2. Process the data into three CSV files — `train.csv`, `dev.csv`, and `test.csv` — each with two columns:

| Column | Description |
|---|---|
| `snt` | The natural language sentence |
| `amr` | The corresponding PENMAN-linearized AMR graph |

3. Run the training script:

```bash
python train_amr2text.py --model <model_name>
```

---

## Perplexity Analysis

The data used for the relational knowledge analysis is provided at `dataset/paws/nld.jsonl`.

To measure perplexity, run:

```bash
python perplexity.py \
    --prefinetune \                        # omit to use off-the-shelf model
    --dataset paws_nld \
    --model_name <llm_single|llm_amr_single> \
    --llm_model_name <qwen-8b|llama-8b> \
    --prompt_type_ppl <text|amr|amr_node_list> \
    --job_id <job_id> \                    # from fine-tuning run, used to reload checkpoint
    --time <time> \                        # from fine-tuning run, used to reload checkpoint
    --lora 7 \
    --eval_batch_size 1                    # must be 1
```

**Notes:**
- `--eval_batch_size` must be set to `1`.
- `--prompt_type_ppl` can be one of: `text`, `amr`, or `amr_node_list`.
- `--job_id` and `--time` are used to reload a specific fine-tuned model checkpoint.
- Omit `--prefinetune` to evaluate with an off-the-shelf (non-fine-tuned) model.

---
