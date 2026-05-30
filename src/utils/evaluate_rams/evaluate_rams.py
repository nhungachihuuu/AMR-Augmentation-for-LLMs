from src.utils.evaluate_rams.utils import process_csv_predictions
from src.utils.evaluate_rams.scorer import run_evaluation
import os 
import json

def run_evaluation_pipeline_rams(mode_answer_file_path, rams_file, processed_file_path, output_column, results_save_path):
    """
    Run the complete evaluation pipeline: post-processing + scoring
    
    Args:
        csv_file_path (str): Path to the generated CSV file with model outputs
        rams_file (str): Path to the RAMS data file
        ontology_file (str, optional): Path to ontology file
    
    Returns:
        This function also : save results into json file. 
        dict: Evaluation results
    """
    
    # Step 1: Run post-processing
    print("\n1. Pipeline started - Running post-processing...")

    # Create the new path
    processed_file = processed_file_path
    
    # Call the post-processing function directly
    process_csv_predictions(
            csv_file=mode_answer_file_path,
            rams_file=rams_file,
            output_file=processed_file,
            output_column=output_column,
            confidence=1.0
        )
    # print(f"✓ Post-processing completed. Output saved to: {processed_file}")
         
    # Step 2: Run scorer
    print("\n2. Running scorer...")
    
     # Create arguments object for scorer
    class Args:
        def __init__(self):
            self.gold_file = rams_file
            self.pred_file = processed_file
            self.reuse_gold_format = False
            self.ontology_file = None
            self.cd = False
            self.do_all = True
            self.metrics = True
            self.distance = True
            self.role_table = True
            self.confusion = True
    
    args = Args()
    results = run_evaluation(args)
    metrics = results['metrics']

        
    print("✓ Scoring completed !")

    def log_evaluation_results(results, results_save_path):
        # Save results to file as JSON
        
        with open(results_save_path, 'w') as f:
            json.dump(results, f)  # Pretty-printed JSON
        
        print(f"\n✓ Pipeline ended - Evaluation results saved to: {results_save_path}")

    #Save results to json file if path provided
    if results_save_path:
        log_evaluation_results(results,results_save_path)


    return results, metrics

