import argparse
from path import OUTPUT_DIR

def parse_args_llama(return_parser=False):
    parser = argparse.ArgumentParser(description="graph_llm")

    #wandb 
    parser.add_argument("--project", type=str, default="GASP") #wandb project name
    parser.add_argument("--note", type=str, default="")        #wandb name note
    parser.add_argument("--seed", type=int, default=0)         #for run with one seed
    parser.add_argument('--seeds', type=int, nargs='+', default=[0,1,2,3,4])  #for run with multiple seeds

#obsolete
#amr2text
    # parser.add_argument("--amr2text_using_paws_when_testing", action="store_true")
    # parser.add_argument("--amr_percentage", type=float, default=None)


    #ppl 
    parser.add_argument("--prompt_type_ppl", default=None)
    parser.add_argument("--prefinetune", action="store_true")

    #dataset
    parser.add_argument("--dataset", type=str,choices=["wmt", "conll" ,"spider", "pubmed" , "wic","sst", "agnews" ,"paws","snli","logiqa","rams","anli","cnn",'paws_nld'],  nargs='+', default=None)
    parser.add_argument("--train_subset", type=int, default=10000000)
    parser.add_argument("--dev_subset", type=int, default=10000000)
    parser.add_argument("--test_subset", type=int, default=10000000)    


    # Model Training
    #amr2text
    parser.add_argument("--prompt_type", type=str, default='user_amr')
    parser.add_argument("--phase2", action="store_true")
    parser.add_argument("--phase2_checkpoint", type=str, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--warmup_steps", type=float, default=150)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--patience", type=float, default=3)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--resume", type=str, default='')    
    parser.add_argument('--resume_checkpoint', type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_steps", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=10)


    # Inference
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--job_id", type=str) #needed for reload model
    parser.add_argument("--time", type=str)    #needed for reload model

    # LLM related
    parser.add_argument("--model_name", type=str, choices=["llm_single", "llm_amr_single" ,"llm_multi", "llm_amr_multi"]) #type of model
    parser.add_argument("--llm_model_name", type=str, choices=["qwen-8b","llama-8b"]) #
    parser.add_argument("--llm_model_path", type=str, default='')

    #check which is used
    parser.add_argument("--llm_frozen", action="store_true")
    # parser.add_argument("--frozen", action="store_true") 


    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    # parser.add_argument("--output_data_dir", type=str, default='prediction')
    parser.add_argument("--max_txt_len", type=int, default=1000)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--lora", type=int, default=7)
    # parser.add_argument("--disable_flash_atten", action="store_true")

    # llm adapter
    # parser.add_argument("--adapter_len", type=int, default=10)

    # parser.add_argument("--adapter_layer", type=int, default=30)


    # distributed training parameters
    # parser.add_argument("--log_dir", type=str, default='logs/')
    # parser.add_argument("--device", type=str, default='cuda')
    # parser.add_argument("--world_size", default=4, type=int, help="number of distributed processes")
    # parser.add_argument("--local_rank", default=-1, type=int)
    # parser.add_argument("--gpu", default='0,1,2,3', type=str)
    # parser.add_argument("--rank", default=0, type=int)
    # parser.add_argument("--dist_on_itp", action="store_true")
    # parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")

    # parser.add_argument("--num_workers", default=8, type=int)

    # GNN related
    # parser.add_argument("--gnn_model_name", type=str, default='gat')
    # parser.add_argument("--gnn_type", type=str, default='rgcn_v2')
    # parser.add_argument("--collate_type", type=str, default='rgcn_v2')
    # parser.add_argument("--gnn_num_layers", type=int, default=4)
    # parser.add_argument("--gnn_in_dim", type=int, default=1024)
    # parser.add_argument("--gnn_hidden_dim", type=int, default=1024)
    # parser.add_argument("--gnn_out_dim", type=int, default=1024)
    # parser.add_argument("--projector_hid_dim", type=int, default=2048)
    # parser.add_argument("--hidden_size", type=int, default=1024)


    # parser.add_argument("--gnn_num_heads", type=int, default=4)
    # parser.add_argument("--gnn_dropout", type=float, default=0.0)
    # parser.add_argument("--commpressed_tokens", type=int, default=5)
    # parser.add_argument("--amr_extract_method", type=str, choices=["last_token", "attention", "all_amr_tokens", "pooling"], default="all_amr_tokens", help="AMR extraction method to use")
    # parser.add_argument("--amr_arrange", type=str, choices=[ "mixed" ,"in-between", "after", "before", "amr_only"], default="in-between", help="How to arrange amr")
    # parser.add_argument("--inf_amr_arrange", type=str, choices=[ "mixed" ,"in-between", "after", "before", "amr_only"], default="in-between", help="How to arrange amr")
    # parser.add_argument("--gate_reg_weight", type=float, default=0.001)

    




    # parser.add_argument("--AMR_model_path", type=str, default="xfbai/AMRBART-large-finetuned-AMR3.0-AMR2Text-v2")
    
    # parser.add_argument("--csv_out_path_test", type=str, default="prediction/test_output")


    
    # parser.add_argument("--rams_file_dev", type=str, default="dataset/rams/dev_filtered_filtered.jsonlines")
    # parser.add_argument("--rams_file_devtest", type=str, default="dataset/rams/devtest.jsonlines")

    #evaluation 
    #rams
    parser.add_argument("--rams_file_test", type=str, default="dataset/rams/test_filtered_filtered.jsonlines")
    
    # parser.add_argument("--csv_out_path_dev", type=str, default="prediction/dev_output")
    # parser.add_argument("--processed_file_path_test", type=str, default="prediction/dev_output.csv")
    # parser.add_argument("--train_dataset_path", type=str, default='dataset/rams/rams_processed_train.pkl')
    # parser.add_argument("--dev_dataset_path", type=str, default='dataset/rams/rams_processed_dev.pkl')
    # parser.add_argument("--test_dataset_path", type=str, default='dataset/rams/rams_processed_test.pkl')
    


    

    parser.add_argument("--max_sentences", type=int, default=10)
    # parser.add_argument("--projector_hid_dim", type=int, default=2048)
    # parser.add_argument("--model_emb_dim", type=int, default=4096)

    #inference 
    # parser.add_argument("--pretrained", action="store_true")

    # parser.add_argument("--disable_amr", action="store_true") #If --disable_amr is provided on the command line, set the corresponding variable to True
    # parser.add_argument("--save_amr", action="store_true") #If --save_amr is provided on the command line, set the corresponding variable to True


#transition 
    # Curriculum learning arguments
    # parser.add_argument('--transition_start', type=int, default=2,
    #                    help='Epoch to start transition from Stage 1 to Stage 2')
    # parser.add_argument('--transition_length', type=int, default=1,
    #                    help='Number of epochs for gradual transition')
    # parser.add_argument('--sparsity_weight', type=float, default=0.01,
    #                    help='Weight for sparsity loss on gate')

    # parser.add_argument('--gnn_lr', type=float, default=5e-5,
    #                    help='Learning rate for GNN')
    # parser.add_argument("--freeze_gnn", action="store_true") #If --frozen is provided on the command line, set the corresponding variable to True

    if return_parser:
        return parser          # ← caller will add more args and parse themselves

    args = parser.parse_args()

    return args
