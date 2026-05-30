import json
import pandas as pd
import re
from src.utils.evaluate_rams.evaluate_rams import run_evaluation_pipeline_rams
from src.utils.evaluate_cnn.rouge import evaluate_rouge_from_csv
from src.utils.spider.evaluation import get_metrics


def get_metrics_rams(mode_answer_file_path, rams_file, processed_file_path, output_column = 'Output', results_save_path = None):
    results, metrics = run_evaluation_pipeline_rams(mode_answer_file_path, rams_file, processed_file_path, results_save_path = results_save_path, output_column = output_column)
    return results, metrics

def get_metrics_cnn(csv_out_path, gold_column_name, output_column_name):
    return evaluate_rouge_from_csv(csv_out_path, gold_column_name, output_column_name)

def get_metrics_spider(spider_gold_dir,pred_dir,spider_db_dir, etype,spider_table_dir):
    return get_metrics(spider_gold_dir,pred_dir,spider_db_dir, etype,spider_table_dir)


eval_funcs = {
    'rams': get_metrics_rams,
    'cnn' : get_metrics_cnn,
    'spider': get_metrics_spider,
}
