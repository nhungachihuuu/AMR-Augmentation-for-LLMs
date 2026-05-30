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


class AgnewsDataset(Dataset):
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
            
        # Now create train labels 
        if mix_amr_text:
            train_labels = train_label(len(self.text['idx']))
            self.text['train_label'] = train_labels


            
        

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text['idx'])

    def __getitem__(self, index):
        if isinstance(index, int):
            label_dict = {'1': 'World', '2': 'Sports', '3': 'Business', '4': 'Sci/Tech'}
            return {
                'dataset': 'agnews',
                'idx': self.text['idx'][index],
                'user': f"Title :{self.text['title'][index]}\nDescription : {self.text['description'][index]}\n",
                'title' : self.text['title'][index], 
                'description' : self.text['description'][index], 
                'assistant': label_dict[str(self.text['label'][index])],
                'description_amr':self.text['description_amr'][index],
                'prompt': self.prompt,
                # 'train_label':'amr_text',
                'train_label': self.text['train_label'][index],
            
                # # Dummy fields
                # 'sentence': self.text['sentence'][index] if 'sentence' in self.text else '',
                # 'sentence_amr': self.text['sentence_amr'][index] if 'sentence_amr' in self.text else '',
                # 'tokens': self.text['tokens'][index] if 'tokens' in self.text else '',
                # 'sentence1': self.text['sentence1'][index] if 'sentence1' in self.text else '',
                # 'sentence1_amr': self.text['sentence1_amr'][index] if 'sentence1_amr' in self.text else '',
                # 'sentence2': self.text['sentence2'][index] if 'sentence2' in self.text else '',
                # 'sentence2_amr': self.text['sentence2_amr'][index] if 'sentence2_amr' in self.text else '',
                # 'interaction_tuple': self.text['interaction_tuple'][index] if 'interaction_tuple' in self.text else '',
                # 'premise': self.text['premise'][index] if 'premise' in self.text else '',
                # 'premise_amr': self.text['premise_amr'][index] if 'premise_amr' in self.text else '',
                # 'hypothesis': self.text['hypothesis'][index] if 'hypothesis' in self.text else '',
                # 'hypothesis_amr': self.text['hypothesis_amr'][index] if 'hypothesis_amr' in self.text else '',
                # 'db_id': self.text['db_id'][index] if 'db_id' in self.text else '',
                # 'question': self.text['question'][index] if 'question' in self.text else '',
                # 'question_amr': self.text['question_amr'][index] if 'question_amr' in self.text else '',
                # 'target': self.text['target'][index] if 'target' in self.text else '',        
            
            }


