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


import pandas as pd
import re
import random


import penman

def shuffle_amr_concepts(amr_text):
    """
    Shuffle AMR concepts while maintaining the graph structure.
    """
    try:
        # Decode AMR string into a Penman graph
        graph = penman.decode(amr_text)
        
        # Extract node concepts
        concepts = [tgt for src, rel, tgt in graph.triples if rel == ':instance']
        
        # Shuffle the concepts
        shuffled = concepts[:]
        random.shuffle(shuffled)
        
        # Replace concepts in the graph
        new_triples = []
        i = 0
        for triple in graph.triples:
            if triple[1] == ':instance':
                new_triples.append((triple[0], triple[1], shuffled[i]))
                i += 1
            else:
                new_triples.append(triple)
        
        # Create a new graph with the updated triples
        new_graph = penman.Graph(new_triples, top=graph.top)
        
        # Return the new AMR string
        return penman.encode(new_graph, indent=3)
    
    except Exception as e:
        print(f"⚠️  Warning: Could not shuffle AMR content: {e}")
        print(f"Original AMR text: {amr_text[:100]}...")
        # Return original text if shuffling fails
        return amr_text

def shuffle_amr_concepts_and_edges(amr_text):
    """
    Shuffle AMR concepts while maintaining the graph structure,
    and replace all edge names with 'edge'.
    """
    try:
        # Decode AMR string into a Penman graph
        graph = penman.decode(amr_text)

        # Extract node concepts
        concepts = [tgt for src, rel, tgt in graph.triples if rel == ':instance']

        # Shuffle the concepts
        shuffled = concepts[:]
        random.shuffle(shuffled)

        # Replace concepts in the graph and rename all edges to ':edge'
        new_triples = []
        i = 0
        for triple in graph.triples:
            if triple[1] == ':instance':
                # Replace concept with shuffled concept, keep ':instance' relation
                new_triples.append((triple[0], triple[1], shuffled[i]))
                i += 1
            else:
                # Replace all non-instance edge names with ':edge'
                new_triples.append((triple[0], ':edge', triple[2]))

        # Create a new graph with the updated triples
        new_graph = penman.Graph(new_triples, top=graph.top)

        # Return the new AMR string
        return penman.encode(new_graph, indent=3)

    except Exception as e:
        print(f"⚠️  Warning: Could not shuffle AMR content: {e}")
        print(f"Original AMR text: {amr_text[:100]}...")
        # Return original text if shuffling fails
        return amr_text

def amr_to_node_list(amr_text):
    """
    Extract all nodes (concepts after /) and leaves (literal values)
    line by line as they appear in the AMR.
    """
    tokens = []
    for line in amr_text.split('\n'):
        # Match nodes: concept after '/'
        match = re.search(r'/\s*(\S+)', line)
        if match:
            concept = match.group(1).split('~')[0]  # strip alignment markers
            tokens.append(concept)
        else:
            # Match leaves: values after ':relation'
            match = re.search(r':\S+\s+(.+)', line)
            if match:
                value = match.group(1).strip().rstrip(')')  # strip trailing ')'
                value = re.sub(r'~\d+', '', value).strip()  # strip alignment markers
                tokens.append(value)
    return ', '.join(tokens)



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

def train_label_third(train_size_total):
    """Helper function to create a dataset from a DataFrame"""

    # Create train_label field: 50% amr_text, 30% text_only, 20% amr
    train_size_total = train_size_total
    half_size = train_size_total // 2
    third_size = train_size_total // 3
    
    
    # Create labels
    train_labels = []
    train_labels.extend(['amr_only'] * third_size)
    train_labels.extend(['amr_text'] * (train_size_total - third_size))
    # train_labels.extend(['text_only'] * (train_size_total - half_size - third_size))
    
    # Randomly shuffle the labels
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(train_labels)
    
    print(f"\nTrain label distribution:")
    print(f"  amr_text: {train_labels.count('amr_text')} samples")
    print(f"  text_only: {train_labels.count('text_only')} samples")
    
    return train_labels


class PawsDataset(Dataset):
    def __init__(self,data_path, prompt_path, prompt_type, nld_path=None,mix_amr_text=False):
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

            
        # Now create train labels based on final filtered size
        if mix_amr_text:
            train_labels = train_label(len(self.text['idx']))
            self.text['train_label'] = train_labels

        with open(self.prompt_path, 'r', encoding='utf-8') as f:
            # print('Load prompts')
            self.prompt = json.load(f)[prompt_type]
            # print('Load prompts succesfully')

        if nld_path is not None:
            data = []
            with open(nld_path, 'r', encoding='utf-8') as f:
                
                for line in f:
                    data.append(json.loads(line))
                
                valid_nld_indices = []
                self.nld_text = {}
                for nld_item in data:
                    if not nld_item.get('amr_nld'):
                        continue  # skip this entry
                    valid_nld_indices.append(nld_item['idx'])
                    self.nld_text[nld_item['idx']] = nld_item['amr_nld'] 
                valid_nld_indices_converted = []
                for idx in valid_nld_indices:
                    valid_nld_indices_converted.append(self.text['idx'].index(idx))
                # Filter text data
                for key in self.text.keys():

                    self.text[key] = [self.text[key][i] for i in valid_nld_indices_converted]
        
        else: 
            # self.nld = None
            self.nld_text = None
        

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text['idx'])

    def __getitem__(self, index):
        if isinstance(index, int):
            label_dict = {'0': 'No' , '1': 'Yes'}
            return {
                'dataset': 'paws',
                'idx': self.text['idx'][index],
                'user': f"Sentence1 :{self.text['sentence1'][index]}\nSentence2 : {self.text['sentence2'][index]}\n",
                'sentence1' : self.text['sentence1'][index], 
                'sentence2' : self.text['sentence2'][index], 
                'assistant': label_dict[str(self.text['label'][index])],
                'sentence1_amr':self.text['sentence1_amr'][index],
                'sentence1_amr_scramble': shuffle_amr_concepts(self.text['sentence1_amr'][index]),
                'sentence2_amr':self.text['sentence2_amr'][index],
                'sentence2_amr_scramble': shuffle_amr_concepts(self.text['sentence2_amr'][index]),
                'prompt': self.prompt,
                'train_label': self.text['train_label'][index],
                'nld': self.nld_text[self.text['idx'][index]] if self.nld_text else "",
                'sentence1_amr_scramble_edge': shuffle_amr_concepts_and_edges(self.text['sentence1_amr'][index]),
                'sentence1_amr_node_list': amr_to_node_list(self.text['sentence1_amr'][index]),

            }



