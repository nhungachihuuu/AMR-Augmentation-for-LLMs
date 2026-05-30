import json
import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
from argparse import ArgumentParser
import json
import pickle
import random
from collections import defaultdict
import csv
import numpy as np

def train_label(train_size_total):
    """Helper function to create a dataset from a DataFrame"""

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
    
    return train_labels


class SnliDataset(Dataset):
    def __init__(self,data_path, prompt_path, prompt_type,mix_amr_text):
        super().__init__()
        self.data_path = data_path
        
        self.prompt_path = prompt_path
        with open(data_path, newline='', encoding="utf-8") as file:
            reader = csv.DictReader(file)
            # Convert to dict of lists where each key is a column name
            self.text = {}
            for row in reader:
                for key, value in row.items():
                    if key not in self.text:
                        self.text[key] = []
                    self.text[key].append(value)
        # Filter out entries with label == -1
        original_size = len(self.text['idx'])
        valid_indices = [i for i, label in enumerate(self.text['label']) if label != '-1']
        
        print(f"Filtering dataset: {original_size} -> {len(valid_indices)} samples")
        print(f"Removed {original_size - len(valid_indices)} samples with label=-1")
        
        # Filter all columns to keep only valid indices
        for key in self.text.keys():
            self.text[key] = [self.text[key][i] for i in valid_indices]


        if mix_amr_text:
            train_labels = train_label(len(self.text['idx']))
            self.text['train_label'] = train_labels

        with open(self.prompt_path, 'r', encoding='utf-8') as f:
            # print('Load prompts')
            self.prompt = json.load(f)[prompt_type]
            # print('Load prompts succesfully')
        
        

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text['idx'])

    def __getitem__(self, index):
        if isinstance(index, int):
            label_dict = {'0': 'entailment' , '1': 'neutral' , '2': 'contradiction'}
            return {
                'dataset': 'snli',
                'idx': self.text['idx'][index],
                'user': f"Premise :{self.text['premise'][index]}\nHypothesis : {self.text['hypothesis'][index]}\n",
                'premise' : self.text['premise'][index], 
                'hypothesis' : self.text['hypothesis'][index], 
                'assistant': label_dict[str(self.text['label'][index])],
                'hypothesis_amr':self.text['hypothesis_amr'][index] if 'hypothesis_amr' in self.text else '',
                'premise_amr':self.text['premise_amr'][index] if 'premise_amr' in self.text else '',
                'prompt': self.prompt,
                # 'train_label':'amr_text',
                'train_label': self.text['train_label'][index],
                
                }


