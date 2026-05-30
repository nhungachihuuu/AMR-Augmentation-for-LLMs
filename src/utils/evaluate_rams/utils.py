
#v3 - Fixed version without redundant functions and proper boundary checking

import json
import re
import csv
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd 
import numpy as np 

def load_model_outputs_from_csv(csv_file: str, output_column: str = "Extracted_output") -> List[str]:
    """
    Load model outputs from CSV file.
    
    Args:
        csv_file: Path to CSV file containing model outputs
        output_column: Name of the column containing model outputs
    
    Returns:
        List of model output strings
    """
    model_outputs = []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        if output_column not in reader.fieldnames:
            raise ValueError(f"Column '{output_column}' not found in CSV. Available columns: {reader.fieldnames}")
        
        for row in reader:
            model_output = row[output_column]
            model_outputs.append(model_output)
    
    return model_outputs

def parse_model_output(model_output: str) -> List[Tuple[str, str]]:
    """
    Parse model output from text format to role-span pairs.
    Filters out roles with empty spans.
    
    Args:
        model_output: String like "transporter: ISIS\nvehicle: oil\norigin:\ndestination: Turkey"
    
    Returns:
        List of (role, span) tuples with non-empty spans only
    """
    if not model_output.strip() or model_output.strip() == "No argument roles found":
        return []
    
    role_span_pairs = []
    lines = model_output.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if ':' in line:
            # Split on first colon only
            parts = line.split(':', 1)
            if len(parts) == 2:
                role = parts[0].strip()
                span = parts[1].strip()
                
                # Only include roles with non-empty spans
                if span and span != "":
                    role_span_pairs.append((role, span))
    
    return role_span_pairs


def extract_span_text_from_positions(start_pos: int, end_pos: int, sentences: List[List[str]]) -> str:
    """
    Extract properly formatted text from span positions using the same formatting rules 
    as extract_passage_from_sentences.
    
    Args:
        start_pos: Start token position (inclusive)
        end_pos: End token position (inclusive)
        sentences: List of tokenized sentences
    
    Returns:
        Properly formatted span text
    """
    # Flatten all sentences to get all tokens
    all_tokens = []
    for sentence in sentences:
        all_tokens.extend(sentence)
    
    # Extract the span tokens
    if start_pos < len(all_tokens) and end_pos < len(all_tokens) and start_pos <= end_pos:
        span_tokens = all_tokens[start_pos:end_pos + 1]
        
        # Apply the same formatting rules as extract_passage_from_sentences
        span_text = ""
        for i, token in enumerate(span_tokens):
            if i == 0:
                span_text += token
            elif token in [".", ",", "!", "?", ":", ";", ")", "]", "}", "'s", "n't", "'re", "'ve", "'ll", "'d"]:
                # No space before punctuation and contractions
                span_text += token
            elif span_tokens[i-1] in ["(", "[", "{"]:
                # No space after opening brackets
                span_text += token
            else:
                span_text += " " + token
        
        return span_text.strip()
    
    return ""

def check_gold_span_match(target_span: str, sentences: List[List[str]], 
                         gold_evt_links: List) -> Optional[Tuple[int, int]]:
    """
    Check if target_span matches any gold argument spans and return their positions.
    Only returns spans that are within sentence boundaries.
    
    Args:
        target_span: Text span to find
        sentences: List of tokenized sentences  
        gold_evt_links: Gold event links in format [[[trigger_start, trigger_end], [arg_start, arg_end], role], ...]
    
    Returns:
        (start_pos, end_pos) tuple if match found within sentence boundaries, None otherwise
    """
    # Create mapping from global position to sentence index
    global_to_sent = []
    for sent_idx, sentence in enumerate(sentences):
        for _ in sentence:
            global_to_sent.append(sent_idx)
    
    # Extract all argument spans from gold_evt_links
    for link in gold_evt_links:
        if len(link) >= 3:  # [trigger_span, arg_span, role]
            arg_span = link[1]  # [start_pos, end_pos]
            if len(arg_span) == 2:
                start_pos, end_pos = arg_span
                
                # Check if span crosses sentence boundaries
                if (start_pos < len(global_to_sent) and end_pos < len(global_to_sent)):
                    if global_to_sent[start_pos] != global_to_sent[end_pos]:
                        continue  # Skip cross-sentence spans
                
                # Extract the properly formatted text from the gold span positions
                gold_text = extract_span_text_from_positions(start_pos, end_pos, sentences)
                
                if gold_text:
                    # Compare with target_span (case-insensitive)
                    if target_span.lower().strip() == gold_text.lower().strip():
                        return (start_pos, end_pos)
                    
                    # Also try comparing without extra spaces
                    target_normalized = " ".join(target_span.split())
                    gold_normalized = " ".join(gold_text.split())
                    if target_normalized.lower() == gold_normalized.lower():
                        return (start_pos, end_pos)
    
    return None

def tokenize_like_sentence(text: str) -> List[str]:
    """
    Tokenize text similar to how sentences are tokenized.
    Separates punctuation from words.
    
    Args:
        text: Input text to tokenize
    
    Returns:
        List of tokens with punctuation separated
    """
    # Split on whitespace first
    words = text.split()
    tokens = []
    
    for word in words:
        # Use regex to separate punctuation from words
        # This pattern captures: word characters, then punctuation, then word characters, etc.
        parts = re.findall(r'\w+|[^\w\s]', word)
        tokens.extend(parts)
    
    return tokens

def flexible_span_search(target_span: str, all_tokens: List[str], 
                        global_to_sent: List[int]) -> Optional[Tuple[int, int]]:
    """
    Flexible fallback search for spans that handles various tokenization issues.
    Only returns spans within sentence boundaries.
    
    Args:
        target_span: Text span to find
        all_tokens: Flattened list of all tokens
        global_to_sent: Mapping from token position to sentence index
    
    Returns:
        (start_pos, end_pos) tuple if found within sentence boundaries, None otherwise
    """
    # Try single token matching with punctuation handling
    target_lower = target_span.lower()
    for i, token in enumerate(all_tokens):
        token_lower = token.lower()
        # Handle punctuation attachments
        if (token_lower.rstrip('.,!?;:') == target_lower or
            token_lower.lstrip('"\'(') == target_lower or
            token_lower.strip('"\'().,!?;:') == target_lower):
            return (i, i)  # Single token always within sentence boundaries
    
    # Try approximate matching by removing punctuation from both target and tokens
    target_words = re.findall(r'\w+', target_span.lower())
    if len(target_words) == 1:
        # Single word - look for it anywhere
        target_word = target_words[0]
        for i, token in enumerate(all_tokens):
            token_words = re.findall(r'\w+', token.lower())
            if token_words and token_words[0] == target_word:
                return (i, i)  # Single token always within sentence boundaries
    
    # Try subsequence matching for multi-word spans
    # This finds the words of the target span even if punctuation differs
    if len(target_words) > 1:
        token_words = []
        token_positions = []
        
        for i, token in enumerate(all_tokens):
            words_in_token = re.findall(r'\w+', token.lower())
            for word in words_in_token:
                token_words.append(word)
                token_positions.append(i)
        
        # Look for target_words as a subsequence in token_words
        for i in range(len(token_words) - len(target_words) + 1):
            if token_words[i:i+len(target_words)] == target_words:
                start_pos = token_positions[i]
                end_pos = token_positions[i + len(target_words) - 1]
                
                # Check if within same sentence
                if (start_pos < len(global_to_sent) and end_pos < len(global_to_sent) and
                    global_to_sent[start_pos] == global_to_sent[end_pos]):
                    return (start_pos, end_pos)
    
    return None

def find_span_positions(target_span: str, sentences: List[List[str]], 
                       gold_evt_links: List = None) -> Optional[Tuple[int, int]]:
    """
    Find the token positions of a target span in the sentences.
    Rejects spans that cross sentence boundaries.
    First checks against gold data if available.
    
    Args:
        target_span: Text span to find (e.g., "ISIS")
        sentences: List of tokenized sentences
        gold_evt_links: Gold event links from RAMS data
    
    Returns:
        (start_pos, end_pos) tuple if found within a single sentence, None if crosses boundaries
    """
    # First, check if target_span matches any gold argument spans
    if gold_evt_links:
        gold_match = check_gold_span_match(target_span, sentences, gold_evt_links)
        if gold_match:
            start_pos, end_pos = gold_match
            # print(f"Found gold match for '{target_span}' at positions [{start_pos}, {end_pos}]")
            return (start_pos, end_pos)
    
    # Create mapping from global position to sentence index
    global_to_sent = []
    global_pos = 0
    
    for sent_idx, sentence in enumerate(sentences):
        for _ in sentence:
            global_to_sent.append(sent_idx)
            global_pos += 1
    
    # Flatten all sentences into a single list
    all_tokens = []
    for sentence in sentences:
        all_tokens.extend(sentence)
    
    # Tokenize the target span using the same method as sentences
    target_tokens = tokenize_like_sentence(target_span)
    
    # Search for the target tokens in the flattened token list
    for i in range(len(all_tokens) - len(target_tokens) + 1):
        # Check if tokens match (case-insensitive)
        if all(all_tokens[i + j].lower() == target_tokens[j].lower() 
               for j in range(len(target_tokens))):
            start_pos = i
            end_pos = i + len(target_tokens) - 1
            
            # Check if span crosses sentence boundaries
            if (start_pos < len(global_to_sent) and end_pos < len(global_to_sent) and
                global_to_sent[start_pos] == global_to_sent[end_pos]):
                return (start_pos, end_pos)
            else:
                # Span crosses sentence boundary - reject it
                # print(f"Rejecting cross-sentence span '{target_span}' at positions [{start_pos}, {end_pos}]")
                continue  # Keep looking for other matches
    
    # Fallback to flexible matching
    result = flexible_span_search(target_span, all_tokens, global_to_sent)
    if result:
        return result
    
    # Try finding within each sentence individually
    for sent_idx, sentence in enumerate(sentences):
        sent_start_pos = sum(len(sentences[j]) for j in range(sent_idx))
        
        # Look for target in this sentence
        for i in range(len(sentence) - len(target_tokens) + 1):
            if all(sentence[i + j].lower() == target_tokens[j].lower() 
                   for j in range(len(target_tokens))):
                start_pos = sent_start_pos + i
                end_pos = sent_start_pos + i + len(target_tokens) - 1
                return (start_pos, end_pos)
        
        # Single token match in this sentence
        target_lower = target_span.lower()
        for i, token in enumerate(sentence):
            if token.lower() == target_lower:
                pos = sent_start_pos + i
                return (pos, pos)
    
    return None

def process_single_prediction(model_output: str, rams_data: Dict[str, Any], 
                             confidence: float = 1.0) -> Dict[str, Any]:
    """
    Process a single model prediction and convert to scorer format.
    Rejects spans that cross sentence boundaries.
    
    Args:
        model_output: Model's text output
        rams_data: Original RAMS data for this document
        confidence: Confidence score for predictions
    
    Returns:
        Dictionary with doc_key and predictions in scorer format
    """
    # Parse model output
    role_span_pairs = parse_model_output(model_output)
    
    # Get document info
    doc_key = rams_data['doc_key']
    sentences = rams_data['sentences']
    evt_triggers = rams_data['evt_triggers']
    
    # Get trigger positions (assuming first trigger for now)
    if not evt_triggers:
        return {"doc_key": doc_key, "predictions": []}
    
    trigger_start, trigger_end = evt_triggers[0][0], evt_triggers[0][1]
    
    # Build predictions list
    predictions = []
    
    # Start with trigger position
    event_prediction = [[trigger_start, trigger_end]]
    
    # Add argument predictions - only those within sentence boundaries
    for role, span_text in role_span_pairs:
        if span_text == "N/A":
            continue
        span_position = find_span_positions(span_text, sentences, 
                                          rams_data.get('gold_evt_links', []))
        if span_position:
            start_pos, end_pos = span_position
            event_prediction.append([start_pos, end_pos, role, confidence])
            # print(f"Found span '{span_text}' -> [{start_pos}, {end_pos}] for role '{role}' in {doc_key}")
    
    predictions.append(event_prediction)
    
    return {
        "doc_key": doc_key,
        "predictions": predictions
    }

def process_batch_predictions(model_outputs: List[str], rams_data_list: List[Dict[str, Any]], 
                            output_file: str, confidence: float = 1.0):
    """
    Process a batch of model predictions and save to file for scorer.
    
    Args:
        model_outputs: List of model text outputs
        rams_data_list: List of corresponding RAMS data
        output_file: Output file path for scorer
        confidence: Confidence score for all predictions
    """
    if len(model_outputs) != len(rams_data_list):
        raise ValueError("Number of model outputs must match number of RAMS data entries")
    
    processed_predictions = []
    
    for i, (model_output, rams_data) in enumerate(zip(model_outputs, rams_data_list)):
        try:
            prediction = process_single_prediction(model_output, rams_data, confidence)
            processed_predictions.append(prediction)
        except Exception as e:
            print(f"Error processing prediction {i}: {e}")
            # Add empty prediction to maintain alignment
            processed_predictions.append({
                "doc_key": rams_data.get('doc_key', f'doc_{i}'),
                "predictions": []
            })
    
    # Save to file (one JSON per line)
    with open(output_file, 'w', encoding='utf-8') as f:
        for prediction in processed_predictions:
            f.write(json.dumps(prediction) + '\n')
    
    print(f"Saved {len(processed_predictions)} predictions to {output_file}")

def load_rams_data(rams_file: str) -> List[Dict[str, Any]]:
    """
    Load RAMS data from JSONL file.
    
    Args:
        rams_file: Path to RAMS JSONL file
    
    Returns:
        List of RAMS data dictionaries
    """
    data = []
    with open(rams_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def process_csv_predictions(csv_file: str, rams_file: str, output_file: str, 
                          output_column: str = "Extracted_output", confidence: float = 1.0):
    """
    Process model predictions from CSV file and save to file for scorer.
    
    Args:
        csv_file: Path to CSV file containing model outputs
        rams_file: Path to RAMS JSONL file
        output_file: Output file path for scorer
        output_column: Name of the column containing model outputs
        confidence: Confidence score for all predictions
    """
    # Load model outputs from CSV
    print(f"Loading model outputs from {csv_file}...")
    model_outputs = load_model_outputs_from_csv(csv_file, output_column)
    print(f"Loaded {len(model_outputs)} model outputs")
    
    # Load RAMS data
    print(f"Loading RAMS data from {rams_file}...")
    rams_data_list = load_rams_data(rams_file)
    print(f"Loaded {len(rams_data_list)} RAMS data entries")
    
    # Check alignment
    if len(model_outputs) != len(rams_data_list):
        print(f"Warning: Number of model outputs ({len(model_outputs)}) != number of RAMS entries ({len(rams_data_list)})")
        min_len = min(len(model_outputs), len(rams_data_list))
        model_outputs = model_outputs[:min_len]
        rams_data_list = rams_data_list[:min_len]
        print(f"Using first {min_len} entries from both")
    
    # Process predictions
    process_batch_predictions(model_outputs, rams_data_list, output_file, confidence)

