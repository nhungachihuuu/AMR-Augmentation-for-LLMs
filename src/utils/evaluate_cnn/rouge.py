import pandas as pd
from rouge_score import rouge_scorer

def evaluate_rouge_from_csv(csv_path, gold_column='Gold', pred_column='Extracted_output'):
    """
    Evaluate ROUGE scores from a CSV file containing gold and predicted summaries.

    Args:
        csv_path (str): Path to the CSV file.
        gold_column (str): Column name for gold/reference summaries.
        pred_column (str): Column name for predicted/generated summaries.

    Returns:
        tuple:
            - pd.DataFrame: Detailed ROUGE scores for each entry.
            - pd.DataFrame: Summary (mean) of ROUGE scores.
    """
    # Initialize the scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'],
                                       use_stemmer=True,
                                       split_summaries=True)

    # Load CSV and clean text
    df = pd.read_csv(csv_path)
    golds = [text.replace('\n', ' ') for text in df[gold_column]]
    preds = [text.replace('\n', ' ') for text in df[pred_column]]

    # Score each pair
    scores_r1, scores_r2, scores_rl = [], [], []
    for gold, pred in zip(golds, preds):
        scores = scorer.score(gold, pred)
        scores_r1.append(scores['rouge1'].fmeasure)
        scores_r2.append(scores['rouge2'].fmeasure)
        scores_rl.append(scores['rougeL'].fmeasure)

    # Combine results
    detailed_results = pd.DataFrame({
        'gold': golds,
        'prediction': preds,
        'rouge1': scores_r1,
        'rouge2': scores_r2,
        'rougeL': scores_rl
    })

    summary_metrics = {
        'rouge1': float(detailed_results['rouge1'].mean() * 100),
        'rouge2': float(detailed_results['rouge2'].mean() * 100),
        'rougeL': float(detailed_results['rougeL'].mean() * 100),
    }


    return detailed_results, summary_metrics
