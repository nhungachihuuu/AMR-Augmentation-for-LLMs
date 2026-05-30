import json
import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
def train_label(train_size_total):

    # Create train_label field: 50% amr_text, 50% text_only
    train_size_total = train_size_total
    half_size = train_size_total // 2
    
    # Create labels
    train_labels = []
    train_labels.extend(['amr_text'] * half_size)
    train_labels.extend(['text_only'] * (train_size_total - half_size))
    
    # Randomly shuffle the labels
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(train_labels)
    
    print(f"\nTrain label distribution:")
    print(f"  amr_text: {train_labels.count('amr_text')} samples")
    print(f"  text_only: {train_labels.count('text_only')} samples")
    return train_labels  # <-- ADD THIS

class RamsDataset(Dataset):
    def __init__(self, data_path, prompt_path, prompt_type,mix_amr_text,data_format="auto"):
        super().__init__()
        self.data_path = data_path

        self.prompt_path = prompt_path
        with open(self.prompt_path, 'r', encoding='utf-8') as f:
            self.prompt = json.load(f)[prompt_type]

        self.text = self._load_data(self.data_path, data_format=data_format)
        
        # create train labels
        if mix_amr_text:
            train_labels = train_label(len(self.text['doc_keys']))
            self.text['train_label'] = train_labels
    def _load_data(self, data_path: str, data_format: str = "auto"):
        """
        Returns data normalized to dict-of-lists:
          {
            "doc_keys": [...],
            "sentence": [...],
            "task_infos": [...],
            "assistant": [...],
            "index": [...],
            "train_label": [...],   # optional
            "amr": [...],           # optional depending on your data
            ...
          }
        """
        if data_format not in {"auto", "pkl", "jsonl"}:
            raise ValueError(f"Unsupported data_format={data_format}. Use auto|pkl|jsonl")

        suffix = Path(data_path).suffix.lower()
        if data_format == "auto":
            if suffix in {".pkl", ".pickle"}:
                data_format = "pkl"
            elif suffix == ".jsonl":
                data_format = "jsonl"
            else:
                raise ValueError(
                    f"Cannot infer data format from extension {suffix}. "
                    f"Please pass data_format='pkl' or 'jsonl'."
                )

        if data_format == "pkl":
            with open(data_path, 'rb') as file:
                data = pickle.load(file)
            if not isinstance(data, dict):
                raise TypeError("Pickle input must be a dict of lists.")
            return data

        # jsonl
        rows = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_num} of {data_path}: {e}") from e

        if not rows:
            raise ValueError(f"No records found in jsonl file: {data_path}")

        if not isinstance(rows[0], dict):
            raise TypeError("JSONL input must be a list of dicts (one dict per line).")

        # Convert list-of-dicts -> dict-of-lists
        # We take the union of keys across rows and fill missing with "" (or None if you prefer).
        keys = set()
        for r in rows:
            keys.update(r.keys())

        data = {k: [] for k in keys}
        for r in rows:
            for k in keys:
                data[k].append(r.get(k, ""))

        # If your pickle always has "index" but jsonl may not, optionally generate it:
        if "index" not in data:
            data["index"] = list(range(len(rows)))

        return data

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text['index'])

    def __getitem__(self, index):
        if isinstance(index, int):
            return {
                'doc_keys': self.text['doc_keys'][index],
                'sentence': self.text['sentence'][index],
                'task_infos': self.text['task_infos'][index],
                'assistant': self.text['assistant'][index],
                'index': self.text['index'][index],
                'train_label': self.text['train_label'][index] if 'train_label' in self.text else '',
                'amr_linearized': '',
                'amr': self.text['amr'][index] if 'amr' in self.text else '',
                'train_label': self.text['train_label'][index],
                'prompt': self.prompt,
                'event_id': "",
                "dataset":"rams",
            }


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-dp', '--data_path', help='Path to the input pkl/jsonl file', type=str,
                        default='dataset/rams/rams_processed_dev.pkl')
    parser.add_argument('-df', '--data_format', help='auto|pkl|jsonl', type=str, default='auto')
    parser.add_argument('-pp', '--prompt_path', help='Path to the list of prompt', type=str,
                        default='dataset/rams/system_message.json')
    parser.add_argument('-pt', '--prompt_type', type=str, help='type of prompt', required=True)
    args = parser.parse_args()

    dataset = RamsDataset(
        data_path=args.data_path,
        prompt_path=args.prompt_path,
        prompt_type=args.prompt_type,
        data_format=args.data_format,
    )

    print(dataset.prompt)
    print(dataset[0])




# import json
# import pandas as pd
# import torch
# from torch.utils.data import Dataset
# import pickle
# from argparse import ArgumentParser

# class RamsDataset(Dataset):
#     def __init__(self,data_path, prompt_path, prompt_type, graph_path):
#         super().__init__()
#         self.data_path = data_path
        
#         self.prompt_path = prompt_path
#         with open(self.data_path, 'rb') as file:
#             # print('Load data')
#             self.text = pickle.load(file)
#             # print('Load data successfully')
#             # print(f"length of dataset : {len(self.text['index'])}")
#         with open(self.prompt_path, 'r', encoding='utf-8') as f:
#             # print('Load prompts')
#             self.prompt = json.load(f)[prompt_type]
#             # print('Load prompts succesfully')
#         # self.graph = torch.load(graph_path, weights_only=False)

#     def __len__(self):
#         """Return the len of the dataset."""
#         return len(self.text['index'])

#     def __getitem__(self, index):
#         if isinstance(index, int):
#             return {
#                 'doc_keys': self.text['doc_keys'][index],
#                 'sentence': self.text['sentence'][index],
#                 'task_infos': self.text['task_infos'][index],
#                 'assistant': self.text['assistant'][index],
#                 'index': self.text['index'][index],
#                 'train_label': self.text['train_label'][index] if 'train_label' in self.text else '',
#                 # 'amr_linearized': self.text['amr_linearized'][index],
#                 'amr_linearized': '',
#                 'amr': self.text['amr'][index],
#                 'prompt': self.prompt,
#                 'event_id': "",

#                 # 'lineared_amr_tokens': self.text['lineared_amr_tokens'][index],
#                 # 'amr_adjacencies': self.text['amr_adjacencies'][index]
#             }

#     # def get_idx_split(self):

#     #     # Load the saved indices
#     #     with open('dataset/tape_arxiv/split/train_indices.txt', 'r') as file:
#     #         train_indices = [int(line.strip()) for line in file]

#     #     with open('dataset/tape_arxiv/split/val_indices.txt', 'r') as file:
#     #         val_indices = [int(line.strip()) for line in file]

#     #     with open('dataset/tape_arxiv/split/test_indices.txt', 'r') as file:
#     #         test_indices = [int(line.strip()) for line in file]
#     #     return {'train': train_indices, 'val': val_indices, 'test': test_indices}


# if __name__ == '__main__':
#     parser = ArgumentParser()
#     parser.add_argument('-dp', '--data_path', help='Path to the input pkl file',type = str, default = 'dataset/rams/rams_processed_dev.pkl')
#     parser.add_argument('-pp', '--prompt_path', help='Path to the list of prompt', type = str, default = 'dataset/rams/system_message.json')
#     parser.add_argument('-pt', '--prompt_type', type = str, help ='type of prompt')
#     args = parser.parse_args()
#     dataset = RamsDataset(data_path=args.data_path,prompt_path=args.prompt_path, prompt_type=args.prompt_type)

#     print(dataset.prompt)
#     print(dataset[0])
