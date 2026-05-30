import pandas as pd
import os 
import json
import pandas as pd
import re
import wandb
from src.utils.evaluate import eval_funcs
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
from seqeval.metrics import f1_score as f1_score_seqeval
from seqeval.metrics import precision_score as precision_score_seqeval
from seqeval.metrics import recall_score as recall_score_seqeval
from sacrebleu import corpus_bleu
from path import OUTPUT_DIR, RAMS_EVAL_PATH, SPIDER_TABLE_DIR, SPIDER_DB_DIR, SPIDER_GOLD_DIR 

os.makedirs(f"{OUTPUT_DIR}/evaluation/", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/prediction/", exist_ok=True)


def post_processing_rams(eval_outputs, args, job_id, time):
    all_rows = []
    # Decide globally whether to include AMR
    include_amr = all('AMR' in batch for batch in eval_outputs)
    for batch in eval_outputs:
        print(batch.keys())

        indices = batch['Index']
        doc_keys = batch['Doc_keys']
        input = batch['Input']
        # input_amr = batch['AMR']
        outputs = batch['Output']
        golds = batch['Gold']
        if include_amr:
            amrs = batch['AMR']  # safe because include_amr ensures presence
            for i, d, inp, amrs, outp, gold in zip(indices, doc_keys, input, amrs, outputs, golds):
                all_rows.append({
                    'Index': i,
                    'Doc_keys': d,
                    'Input': inp,
                    'AMR': amrs,
                    'Output': outp,
                    'Gold': gold
                    })
        else: 
            # Create individual rows
            for i, d, inp,  o, g in zip(indices, doc_keys, input, outputs, golds):
                all_rows.append({
                    'Index': i,
                    'Doc_keys': d,
                    'Input': inp, 
                    'Output': o,
                    'Gold': g
                })

    # Create DataFrame and save
    df = pd.DataFrame(all_rows)
    csv_out_path = f"{OUTPUT_DIR}/prediction/rams_{job_id}_{time}.csv"
    results_save_path = f"{OUTPUT_DIR}/prediction/rams_full_res_{job_id}_{time}.csv"
    metric_save_path = f"{OUTPUT_DIR}/evaluation/rams_{job_id}_{time}.json"
    # parent_dir = os.path.dirname(args.csv_out_path_test)
    # os.makedirs(parent_dir, exist_ok=True)
    df.to_csv(csv_out_path, index=False)
    results, metrics = eval_funcs['rams'](csv_out_path,RAMS_EVAL_PATH, f"{OUTPUT_DIR}/prediction/preds_for_scorer_test_{job_id}_{time}.jsonl")

    with open(results_save_path, 'w') as f:
        json.dump(results, f)  # Pretty-printed JSON
    with open(metric_save_path, 'w') as f:
        json.dump(metrics, f)  # Pretty-printed JSON
    
    print(f"\n✓ Pipeline ended - Evaluation results saved to: {results_save_path}")
    print(metrics)
    wandb.log({
    "rams/test/f1": metrics['f1'],
    "rams/test/precision": metrics['precision'], 
    "rams/test/recall": metrics['recall']
    })
    return metrics 


def post_processing_cnn(eval_outputs, args, job_id, time):
    all_rows = []
    # Decide globally whether to include AMR
    include_amr = all('AMR' in batch for batch in eval_outputs)
    for batch in eval_outputs:
        print(batch.keys())

        indices = batch['Index']
        doc_keys = batch['Doc_keys']
        input = batch['Input']
        # input_amr = batch['AMR']
        outputs = batch['Output']
        golds = batch['Gold']
        if include_amr:
            amrs = batch['AMR']  # safe because include_amr ensures presence
            for i, d, inp, amrs, outp, gold in zip(indices, doc_keys, input, amrs, outputs, golds):
                all_rows.append({
                    'Index': i,
                    'Doc_keys': d,
                    'Input': inp,
                    'AMR': amrs,
                    'Output': outp,
                    'Gold': gold
                    })
        else: 
            # Create individual rows
            for i, d, inp,  o, g in zip(indices, doc_keys, input, outputs, golds):
                all_rows.append({
                    'Index': i,
                    'Doc_keys': d,
                    'Input': inp, 
                    'Output': o,
                    'Gold': g
                })

    # Create DataFrame and save
    df = pd.DataFrame(all_rows)


    csv_out_path = f"{OUTPUT_DIR}/prediction/cnn_{job_id}_{time}.csv"
    results_save_path = f"{OUTPUT_DIR}/prediction/cnn_full_res_{job_id}_{time}.csv"
    # parent_dir = os.path.dirname(args.csv_out_path_test)
    # os.makedirs(parent_dir, exist_ok=True)


    df.to_csv(csv_out_path, index=False)
    detailed_results, summary_metrics = eval_funcs['cnn'](csv_out_path, "Gold", "Output")

    detailed_results.to_csv(results_save_path, index=False)
    
    print(f"\n✓ Pipeline ended - Evaluation results saved to: {results_save_path}")
    wandb.log({
    "cnn/test/rouge1": summary_metrics['rouge1'],
    "cnn/test/rouge2": summary_metrics['rouge2'], 
    "cnn/test/rougeL": summary_metrics['rougeL']
    })  
    return summary_metrics

def post_processing_anli(eval_outputs, args, job_id, time):
    all_rows = []
    # Decide globally whether to include AMR
    include_amr = all('AMR' in batch for batch in eval_outputs)
    for batch in eval_outputs:
        print(batch.keys())

        indices = batch['Index']
        doc_keys = batch['Doc_keys']
        input = batch['Input']
        # input_amr = batch['AMR']
        outputs = batch['Output']
        golds = batch['Gold']
        if include_amr:
            amrs = batch['AMR']  # safe because include_amr ensures presence
            for i, d, inp, amrs, outp, gold in zip(indices, doc_keys, input, amrs, outputs, golds):
                all_rows.append({
                    'Index': i,
                    'Doc_keys': d,
                    'Input': inp,
                    'AMR': amrs,
                    'Output': outp,
                    'Gold': gold
                    })
        else: 
            # Create individual rows
            for i, d, inp,  o, g in zip(indices, doc_keys, input, outputs, golds):
                all_rows.append({
                    'Index': i,
                    'Doc_keys': d,
                    'Input': inp, 
                    'Output': o,
                    'Gold': g
                })

    # Create DataFrame and save
    df = pd.DataFrame(all_rows)


    csv_out_path = f"{OUTPUT_DIR}/anli_{job_id}_{time}.csv"
    results_save_path = f"{OUTPUT_DIR}/anli_full_res_{job_id}_{time}.csv"
    # parent_dir = os.path.dirname(args.csv_out_path_test)
    # os.makedirs(parent_dir, exist_ok=True)


    df.to_csv(csv_out_path, index=False)
    references = df['Gold'].to_list()
    predictions = df['Output'].to_list()
    acc = accuracy_score(references, predictions)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        references, predictions, average='macro', zero_division=0
    )
    results =  {
        'accuracy': acc,
        'macro_precision': macro_p,
        'macro_recall': macro_r,
        'macro_f1': macro_f1
    }

    with open(results_save_path, 'w') as f:
        json.dump(results, f)  # Pretty-printed JSON
    
    print(f"\n✓ Pipeline ended - Evaluation results saved to: {results_save_path}")
    wandb.log({
    "anli/test/f1": results['macro_f1'],
    "anli/test/precision": results['macro_precision'], 
    "anli/test/recall": results['macro_recall'],
    "anli/test/acc": results['accuracy'],
    })  
    return results


def post_processing_logiqa(eval_outputs, args, job_id, time):
    all_rows = []
    # Decide globally whether to include AMR
    include_amr = all('AMR' in batch for batch in eval_outputs)
    for batch in eval_outputs:
        print(batch.keys())

        indices = batch['Index']
        doc_keys = batch['Doc_keys']
        input = batch['Input']
        # input_amr = batch['AMR']
        outputs = batch['Output']
        golds = batch['Gold']
        if include_amr:
            amrs = batch['AMR']  # safe because include_amr ensures presence
            for i, d, inp, amrs, outp, gold in zip(indices, doc_keys, input, amrs, outputs, golds):
                all_rows.append({
                    'Index': i,
                    'Doc_keys': d,
                    'Input': inp,
                    'AMR': amrs,
                    'Output': outp,
                    'Gold': gold
                    })
        else: 
            # Create individual rows
            for i, d, inp,  o, g in zip(indices, doc_keys, input, outputs, golds):
                all_rows.append({
                    'Index': i,
                    'Doc_keys': d,
                    'Input': inp, 
                    'Output': o,
                    'Gold': g
                })

    # Create DataFrame and save
    df = pd.DataFrame(all_rows)


    csv_out_path = f"{OUTPUT_DIR}/prediction/logiqa_{job_id}_{time}.csv"
    results_save_path = f"{OUTPUT_DIR}/prediction/logiqa_full_res_{job_id}_{time}.csv"
    # parent_dir = os.path.dirname(args.csv_out_path_test)
    # os.makedirs(parent_dir, exist_ok=True)


    df.to_csv(csv_out_path, index=False)
    references = df['Gold'].str.lower().to_list()
    predictions = df['Output'].str.lower().to_list()
    accuracy = sum(p == r for p, r in zip(predictions, references)) / len(predictions)

    results =  {
        'accuracy': accuracy,
    }

    with open(results_save_path, 'w') as f:
        json.dump(results, f)  # Pretty-printed JSON
    
    print(f"\n✓ Pipeline ended - Evaluation results saved to: {results_save_path}")
    wandb.log({
    "logiqa/accuracy": results['accuracy'],
    })  
    return results


def post_processing_snli(eval_outputs, args, job_id, time):
    all_rows = []
    # Decide globally whether to include AMR
    include_amr = all('AMR' in batch for batch in eval_outputs)
    for batch in eval_outputs:
        print(batch.keys())

        indices = batch['Idx']
        input = batch['Input']
        # input_amr = batch['AMR']
        outputs = batch['Output']
        golds = batch['Gold']
        if include_amr:
            amrs = batch['AMR']  # safe because include_amr ensures presence
            for i, inp, amrs, outp, gold in zip(indices, input, amrs, outputs, golds):
                all_rows.append({
                    'Idx': i,
                    'Input': inp,
                    'AMR': amrs,
                    'Output': outp,
                    'Gold': gold
                    })
        else: 
            # Create individual rows
            for i, inp,  o, g in zip(indices, input, outputs, golds):
                all_rows.append({
                    'Idx': i,
                    'Input': inp, 
                    'Output': o,
                    'Gold': g
                })

    # Create DataFrame and save
    df = pd.DataFrame(all_rows)


    csv_out_path = f"{OUTPUT_DIR}/snli_{job_id}_{time}.csv"
    results_save_path = f"{OUTPUT_DIR}/snli_full_res_{job_id}_{time}.csv"
    # parent_dir = os.path.dirname(args.csv_out_path_test)
    # os.makedirs(parent_dir, exist_ok=True)


    df.to_csv(csv_out_path, index=False)
    references = df['Gold'].str.lower().str.strip().to_list()
    predictions = df['Output'].str.lower().str.strip().to_list()
    acc = accuracy_score(references, predictions)
    f1_macro = f1_score(references, predictions, average='macro')
    results =  {
        'accuracy': acc,
        'f1': f1_macro
    }

    with open(results_save_path, 'w') as f:
        json.dump(results, f)  # Pretty-printed JSON
    
    print(f"\n✓ Pipeline ended - Evaluation results saved to: {results_save_path}")
    wandb.log({
    "snli/test/accuracy": results['accuracy'],
    "snli/test/f1": results['f1'],
    })  
    return results


def post_processing_paws_fsdp(all_predictions,test_dataset, prompts, job_id, time):

    results = []
    for i, output in enumerate(all_predictions):
        result = {
            "Input": prompts[i],    
            "Output": output,
            "Gold": test_dataset[i]["assistant"],
        }
        # print(result)
        results.append(result)

    df = pd.DataFrame(results)

    csv_out_path = f"{OUTPUT_DIR}/prediction/paws_{job_id}_{time}.csv"
    results_save_path = f"{OUTPUT_DIR}/prediction/paws_full_res_{job_id}_{time}.csv"

    df.to_csv(csv_out_path, index=False)
    references = df['Gold'].str.lower().str.strip().to_list()
    predictions = df['Output'].str.lower().str.strip().to_list()
    acc = accuracy_score(references, predictions)
    metrics =  {
        'accuracy': acc,
    }

    with open(results_save_path, 'w') as f:
        json.dump(metrics, f)  # Pretty-printed JSON
    
    print(f"\n✓ Pipeline ended - Evaluation results saved to: {results_save_path}")
    return df, metrics

def post_processing_paws(eval_outputs, args, job_id, time):
    all_rows = []
    # Decide globally whether to include AMR
    include_amr = all('AMR' in batch for batch in eval_outputs)
    for batch in eval_outputs:
        print(batch.keys())

        indices = batch['Idx']
        input = batch['Input']
        # input_amr = batch['AMR']
        outputs = batch['Output']
        golds = batch['Gold']
        if include_amr:
            amrs = batch['AMR']  # safe because include_amr ensures presence
            for i, inp, amrs, outp, gold in zip(indices, input, amrs, outputs, golds):
                all_rows.append({
                    'Idx': i,
                    'Input': inp,
                    'AMR': amrs,
                    'Output': outp,
                    'Gold': gold
                    })
        else: 
            # Create individual rows
            for i, inp,  o, g in zip(indices, input, outputs, golds):
                all_rows.append({
                    'Idx': i,
                    'Input': inp, 
                    'Output': o,
                    'Gold': g
                })

    # Create DataFrame and save
    df = pd.DataFrame(all_rows)


    csv_out_path = f"{OUTPUT_DIR}/prediction/paws_{job_id}_{time}.csv"
    results_save_path = f"{OUTPUT_DIR}/prediction/paws_full_res_{job_id}_{time}.csv"

    df.to_csv(csv_out_path, index=False)
    references = df['Gold'].str.lower().str.strip().to_list()
    predictions = df['Output'].str.lower().str.strip().to_list()
    acc = accuracy_score(references, predictions)
    f1_macro = f1_score(references, predictions, average='macro')
    results =  {
        'accuracy': acc,
        'f1':f1_macro
    }

    with open(results_save_path, 'w') as f:
        json.dump(results, f)  # Pretty-printed JSON
    
    print(f"\n✓ Pipeline ended - Evaluation results saved to: {results_save_path}")
    wandb.log({
    "paws/test/accuracy": results['accuracy'],
    "paws/test/f1": results['f1'],
    })  
    return results




def post_processing_agnews(eval_outputs, args, job_id, time):
    all_rows = []
    # Decide globally whether to include AMR
    include_amr = all('AMR' in batch for batch in eval_outputs)
    for batch in eval_outputs:
        print(batch.keys())

        indices = batch['Idx']
        input = batch['Input']
        # input_amr = batch['AMR']
        outputs = batch['Output']
        golds = batch['Gold']
        if include_amr:
            amrs = batch['AMR']  # safe because include_amr ensures presence
            for i, inp, amrs, outp, gold in zip(indices, input, amrs, outputs, golds):
                all_rows.append({
                    'Idx': i,
                    'Input': inp,
                    'AMR': amrs,
                    'Output': outp,
                    'Gold': gold
                    })
        else: 
            # Create individual rows
            for i, inp,  o, g in zip(indices, input, outputs, golds):
                all_rows.append({
                    'Idx': i,
                    'Input': inp, 
                    'Output': o,
                    'Gold': g
                })

    # Create DataFrame and save
    df = pd.DataFrame(all_rows)


    csv_out_path =      f"{OUTPUT_DIR}/prediction/agnews_{job_id}_{time}.csv"
    results_save_path = f"{OUTPUT_DIR}/prediction/agnews_full_res_{job_id}_{time}.csv"

    df.to_csv(csv_out_path, index=False)
    references = df['Gold'].str.lower().str.strip().to_list()
    predictions = df['Output'].str.lower().str.strip().to_list()
    acc = accuracy_score(references, predictions)
    # Calculate macro F1-score
    f1_macro = f1_score(references, predictions, average='macro')
    results =  {
        'accuracy': acc,
        'f1': f1_macro
    }

    with open(results_save_path, 'w') as f:
        json.dump(results, f)  # Pretty-printed JSON
    
    print(f"\n✓ Pipeline ended - Evaluation results saved to: {results_save_path}")
    wandb.log({
    "agnews/test/accuracy": results['accuracy'],
    "agnews/test/f1": results['f1'],
    })  
    return results



def post_processing_sst(eval_outputs, args, job_id, time):
    all_rows = []
    # Decide globally whether to include AMR
    include_amr = all('AMR' in batch for batch in eval_outputs)
    for batch in eval_outputs:
        print(batch.keys())

        indices = batch['Idx']
        input = batch['Input']
        # input_amr = batch['AMR']
        outputs = batch['Output']
        golds = batch['Gold']
        if include_amr:
            amrs = batch['AMR']  # safe because include_amr ensures presence
            for i, inp, amrs, outp, gold in zip(indices, input, amrs, outputs, golds):
                all_rows.append({
                    'Idx': i,
                    'Input': inp,
                    'AMR': amrs,
                    'Output': outp,
                    'Gold': gold
                    })
        else: 
            # Create individual rows
            for i, inp,  o, g in zip(indices, input, outputs, golds):
                all_rows.append({
                    'Idx': i,
                    'Input': inp, 
                    'Output': o,
                    'Gold': g
                })

    # Create DataFrame and save
    df = pd.DataFrame(all_rows)


    csv_out_path =      f"{OUTPUT_DIR}/prediction/sst_{job_id}_{time}.csv"
    results_save_path = f"{OUTPUT_DIR}/prediction/sst_full_res_{job_id}_{time}.csv"

    df.to_csv(csv_out_path, index=False)
    references = df['Gold'].str.lower().str.strip().to_list()
    predictions = df['Output'].str.lower().str.strip().to_list()
    acc = accuracy_score(references, predictions)
    # Calculate macro F1-score
    f1_macro = f1_score(references, predictions, average='macro')
    results =  {
        'accuracy': acc,
        'f1': f1_macro
    }

    with open(results_save_path, 'w') as f:
        json.dump(results, f)  # Pretty-printed JSON
    
    print(f"\n✓ Pipeline ended - Evaluation results saved to: {results_save_path}")
    wandb.log({
    "sst/test/accuracy": results['accuracy'],
    "sst/test/f1": results['f1'],
    })  
    return results

def post_processing_pubmed(eval_outputs, args, job_id, time):
    all_rows = []
    # Decide globally whether to include AMR
    include_amr = all('AMR' in batch for batch in eval_outputs)
    for batch in eval_outputs:
        print(batch.keys())

        indices = batch['Idx']
        input = batch['Input']
        # input_amr = batch['AMR']
        outputs = batch['Output']
        golds = batch['Gold']
        if include_amr:
            amrs = batch['AMR']  # safe because include_amr ensures presence
            for i, inp, amrs, outp, gold in zip(indices, input, amrs, outputs, golds):
                all_rows.append({
                    'Idx': i,
                    'Input': inp,
                    'AMR': amrs,
                    'Output': outp,
                    'Gold': gold
                    })
        else: 
            # Create individual rows
            for i, inp,  o, g in zip(indices, input, outputs, golds):
                all_rows.append({
                    'Idx': i,
                    'Input': inp, 
                    'Output': o,
                    'Gold': g
                })

    # Create DataFrame and save
    df = pd.DataFrame(all_rows)


    csv_out_path =      f"{OUTPUT_DIR}/prediction/pubmed_{job_id}_{time}.csv"
    results_save_path = f"{OUTPUT_DIR}/prediction/pubmed_full_res_{job_id}_{time}.csv"

    df.to_csv(csv_out_path, index=False)
    references = df['Gold'].str.lower().str.strip().to_list()
    predictions = df['Output'].str.lower().str.strip().to_list()
    acc = accuracy_score(references, predictions)
    # Calculate macro F1-score
    f1_macro = f1_score(references, predictions, average='macro')
    results =  {
        'accuracy': acc,
        'f1': f1_macro
    }

    with open(results_save_path, 'w') as f:
        json.dump(results, f)  # Pretty-printed JSON
    
    print(f"\n✓ Pipeline ended - Evaluation results saved to: {results_save_path}")
    wandb.log({
    "pubmed/test/accuracy": results['accuracy'],
    "pubmed/test/f1": results['f1'],
    })  
    return results

def post_processing_wic(eval_outputs, args, job_id, time):
    all_rows = []
    # Decide globally whether to include AMR
    include_amr = all('AMR' in batch for batch in eval_outputs)
    for batch in eval_outputs:
        print(batch.keys())

        indices = batch['Idx']
        input = batch['Input']
        # input_amr = batch['AMR']
        outputs = batch['Output']
        golds = batch['Gold']
        if include_amr:
            amrs = batch['AMR']  # safe because include_amr ensures presence
            for i, inp, amrs, outp, gold in zip(indices, input, amrs, outputs, golds):
                all_rows.append({
                    'Idx': i,
                    'Input': inp,
                    'AMR': amrs,
                    'Output': outp,
                    'Gold': gold
                    })
        else: 
            # Create individual rows
            for i, inp,  o, g in zip(indices, input, outputs, golds):
                all_rows.append({
                    'Idx': i,
                    'Input': inp, 
                    'Output': o,
                    'Gold': g
                })

    # Create DataFrame and save
    df = pd.DataFrame(all_rows)


    csv_out_path =      f"{OUTPUT_DIR}/prediction/wic_{job_id}_{time}.csv"
    results_save_path = f"{OUTPUT_DIR}/prediction/wic_full_res_{job_id}_{time}.csv"

    df.to_csv(csv_out_path, index=False)
    references = df['Gold'].str.lower().str.strip().to_list()
    predictions = df['Output'].str.lower().str.strip().to_list()
    acc = accuracy_score(references, predictions)
    # Calculate macro F1-score
    f1_macro = f1_score(references, predictions, average='macro')
    results =  {
        'accuracy': acc,
        'f1': f1_macro
    }

    with open(results_save_path, 'w') as f:
        json.dump(results, f)  # Pretty-printed JSON
    
    print(f"\n✓ Pipeline ended - Evaluation results saved to: {results_save_path}")
    wandb.log({
    "wic/test/accuracy": results['accuracy'],
    "wic/test/f1": results['f1'],
    })  
    return results

def post_processing_wmt(eval_outputs, args, job_id, time):
    all_rows = []
    # Decide globally whether to include AMR
    include_amr = all('AMR' in batch for batch in eval_outputs)
    for batch in eval_outputs:
        print(batch.keys())

        indices = batch['Idx']
        input = batch['Input']
        # input_amr = batch['AMR']
        outputs = batch['Output']
        golds = batch['Gold']
        if include_amr:
            amrs = batch['AMR']  # safe because include_amr ensures presence
            for i, inp, amrs, outp, gold in zip(indices, input, amrs, outputs, golds):
                all_rows.append({
                    'Idx': i,
                    'Input': inp,
                    'AMR': amrs,
                    'Output': outp,
                    'Gold': gold
                    })
        else: 
            # Create individual rows
            for i, inp,  o, g in zip(indices, input, outputs, golds):
                all_rows.append({
                    'Idx': i,
                    'Input': inp, 
                    'Output': o,
                    'Gold': g
                })

    # Create DataFrame and save
    df = pd.DataFrame(all_rows)


    csv_out_path =      f"{OUTPUT_DIR}/prediction/wmt_{job_id}_{time}.csv"
    results_save_path = f"{OUTPUT_DIR}/prediction/wmt_full_res_{job_id}_{time}.csv"

    df.to_csv(csv_out_path, index=False)
    predictions = df['Output'].str.strip().tolist()

    # ✅ Correct format: list of reference streams
    # If you have 1 reference per sentence, wrap all of them in one list
    references = [df['Gold'].tolist()]  # shape: [num_refs, num_sentences]
    bleu = corpus_bleu(predictions, references)
    # Calculate macro F1-score
    results =  {
        'bleu': float(bleu.score)
    }

    with open(results_save_path, 'w') as f:
        json.dump(results, f)  # Pretty-printed JSON
    
    print(f"\n✓ Pipeline ended - Evaluation results saved to: {results_save_path}")
    wandb.log({
    "wmt/test/bleu": results['bleu'],
    }) 
    return results



def post_processing_conll(eval_outputs, args, job_id, time):
    all_rows = []
    # Decide globally whether to include AMR
    include_amr = all('AMR' in batch for batch in eval_outputs)
    for batch in eval_outputs:
        print(batch.keys())

        indices = batch['Idx']
        input = batch['Input']
        # input_amr = batch['AMR']
        outputs = batch['Output']
        golds = batch['Gold']
        if include_amr:
            amrs = batch['AMR']  # safe because include_amr ensures presence
            for i, inp, amrs, outp, gold in zip(indices, input, amrs, outputs, golds):
                all_rows.append({
                    'Idx': i,
                    'Input': inp,
                    'AMR': amrs,
                    'Output': outp,
                    'Gold': gold
                    })
        else: 
            # Create individual rows
            for i, inp,  o, g in zip(indices, input, outputs, golds):
                all_rows.append({
                    'Idx': i,
                    'Input': inp, 
                    'Output': o,
                    'Gold': g
                })

    # Create DataFrame and save
    df = pd.DataFrame(all_rows)




    csv_out_path =      f"{OUTPUT_DIR}/prediction/conll_{job_id}_{time}.csv"
    results_save_path = f"{OUTPUT_DIR}/prediction/conll_full_res_{job_id}_{time}.csv"

    df.to_csv(csv_out_path, index=False)


    # Count and filter problematic samples
    # problematic_count = 0
    predictions = [x.strip().split(',') for x in df['Output']]
    references  = [x.strip().split(',') for x in df['Gold']]

    valid_refs = []
    valid_preds = []
    skipped = 0

    for i, (r, p) in enumerate(zip(references, predictions)):
        if len(r) == len(p):
            valid_refs.append(r)
            valid_preds.append(p)
        else:
            skipped += 1

    print(f"Skipping {skipped} mismatched samples, evaluating on {len(valid_refs)} samples")

    f1        = f1_score_seqeval(valid_refs, valid_preds)
    precision = precision_score_seqeval(valid_refs, valid_preds)
    recall    = recall_score_seqeval(valid_refs, valid_preds)
    print(f"\nMetrics (on {len(predictions)} valid samples):")
    print(f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    results =  {
        'f1': f1,
        'precision': precision,
        'recall': recall,

    }

    with open(results_save_path, 'w') as f:
        json.dump(results, f)  # Pretty-printed JSON
    
    print(f"\n✓ Pipeline ended - Evaluation results saved to: {results_save_path}")
    wandb.log({
    "conll/test/f1": results['f1'],
    "conll/test/precision": results['precision'],
    "conll/test/recall": results['recall'],
    }) 
    return results

def post_processing_spider(eval_outputs, args, job_id, time):
    all_rows = []
    # Decide globally whether to include AMR
    include_amr = all('AMR' in batch for batch in eval_outputs)
    for batch in eval_outputs:
        print(batch.keys())

        indices = batch['Idx']
        input = batch['Input']
        # input_amr = batch['AMR']
        outputs = batch['Output']
        golds = batch['Gold']
        if include_amr:
            amrs = batch['AMR']  # safe because include_amr ensures presence
            for i, inp, amrs, outp, gold in zip(indices, input, amrs, outputs, golds):
                all_rows.append({
                    'Idx': i,
                    'Input': inp,
                    'AMR': amrs,
                    'Output': outp,
                    'Gold': gold
                    })
        else: 
            # Create individual rows
            for i, inp,  o, g in zip(indices, input, outputs, golds):
                all_rows.append({
                    'Idx': i,
                    'Input': inp, 
                    'Output': o,
                    'Gold': g
                })

    # Create DataFrame and save
    df = pd.DataFrame(all_rows)

    df['sort_key'] = df['Idx'].apply(lambda x: int(x.split('_')[0]))
    df = df.sort_values('sort_key').reset_index(drop=True)
    df = df.drop(columns='sort_key')  # clean up after sorting
    # Clean newlines from Output
    df['Output'] = df['Output'].apply(lambda x: str(x).strip().replace("\n", " ").replace("\r", " "))

    csv_out_path =      f"{OUTPUT_DIR}/prediction/spider_{job_id}_{time}.csv"
    results_save_path = f"{OUTPUT_DIR}/prediction/spider_full_res_{job_id}_{time}.csv"

    df.to_csv(csv_out_path, index=False)

    with open(f"{OUTPUT_DIR}/prediction/spider_{job_id}_{time}.sql", "w") as f:
        for _, row in df.iterrows():
            query = str(row["Output"]).strip()
            f.write(f"{query}\n")

    etype = 'match'
    
    pred_dir = f"{OUTPUT_DIR}/prediction/spider_{job_id}_{time}.sql"
    new_score_dict = eval_funcs['spider'](SPIDER_GOLD_DIR,pred_dir,SPIDER_DB_DIR, etype,SPIDER_TABLE_DIR)

    with open(results_save_path, 'w') as f:
        json.dump(new_score_dict, f)  # Pretty-printed JSON

    print(f"\n✓ Pipeline ended - Evaluation results saved to: {results_save_path}")
    wandb.log({
    "spider/all/EM": new_score_dict['all'],
    }) 
    return new_score_dict

post_processing_func = {
    'rams': post_processing_rams,
    'cnn' : post_processing_cnn,
    'anli' : post_processing_anli,
    'logiqa': post_processing_logiqa,
    'snli': post_processing_snli,
    'paws': post_processing_paws,
    'agnews': post_processing_agnews,
    'sst': post_processing_sst,
    'pubmed': post_processing_pubmed,
    'wic': post_processing_wic,
    'wmt': post_processing_wmt,
    'conll':post_processing_conll,
    'spider': post_processing_spider,
}