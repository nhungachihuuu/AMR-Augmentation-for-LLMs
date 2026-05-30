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



class ConllDataset(Dataset):
    def __init__(self,data_path, prompt_path, prompt_type,mix_amr_text,):
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




        
        with open(self.prompt_path, 'r', encoding='utf-8') as f:
            # print('Load prompts')
            self.prompt = json.load(f)[prompt_type]
            # print('Load prompts succesfully')        
            
        # Now create train labels based on final filtered size
        if mix_amr_text:
            train_labels = train_label(len(self.text['idx']))
            self.text['train_label'] = train_labels


        
        

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text['idx'])

    def __getitem__(self, index):
        if isinstance(index, int):

            return {
                'dataset': 'conll',
                'idx': self.text['idx'][index],
                'user': f"Sentence : {self.text['sentence'][index]}\nTokens: {self.text['tokens'][index]}\n",
                'sentence' : self.text['sentence'][index], 
                'assistant': str(self.text['label'][index]),
                'sentence_amr':self.text['sentence_amr'][index],
                'tokens': self.text['tokens'][index],
                'prompt': self.prompt,
                # 'train_label': 'amr_text',
                'train_label': self.text['train_label'][index],

            }


