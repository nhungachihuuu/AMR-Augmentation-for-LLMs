import os
import torch


def print_trainable_params(model):
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        num_params = param.numel()

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param

def check_lora_in_checkpoint(checkpoint_path):
    """
    Check if a saved checkpoint contains LoRA parameters.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"]
    
    # LoRA parameters typically have 'lora' in their names
    lora_keys = [k for k in state_dict.keys() if 'lora' in k.lower()]
    
    if lora_keys:
        print(f"Found {len(lora_keys)} LoRA parameters:")
        for key in lora_keys[:10]:  # Show first 10
            print(f"  - {key}")
        if len(lora_keys) > 10:
            print(f"  ... and {len(lora_keys) - 10} more")
        return True
    else:
        print("No LoRA parameters found in checkpoint")
        return False


def _save_checkpoint(model, optimizer1, cur_epoch,job_id, time, args, is_best=False):
    """
    Save the checkpoint at the current epoch.
    job_id is the job_id of the ht condor, imported from evn ( this param should be obsoleted if the env doesnt have job_id)
    """

    os.makedirs(args.output_dir, exist_ok=True)

    param_grad_dic = {
        k: v.requires_grad for (k, v) in model.named_parameters()
    }
    state_dict = model.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            # delete parameters that do not require gradient
            del state_dict[k]
    save_obj = {
        "model": state_dict,
        "optimizer": optimizer1.state_dict(),
        "config": args,
        "epoch": cur_epoch,
    }

    path = f'{args.dataset}_{args.model_name}_{args.llm_model_name}_{args.seed}_{job_id}_{time}'
    save_to = os.path.join(
        args.output_dir,
        path+"_checkpoint_{}.pth".format("best" if is_best else cur_epoch),
    )

    print("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
    torch.save(save_obj, save_to)



def _reload_best_model_search(model, job_id, time, args):
    """
    Load the best checkpoint for evaluation by searching for files matching job_id and time pattern.
    """
    import glob
    
    # Search for files matching the pattern with job_id and time
    pattern = f'*{args.seed}_{job_id}_{time}_checkpoint_best.pth'
    search_path = os.path.join(args.output_dir, pattern)
    
    matching_files = glob.glob(search_path)
    
    if not matching_files:
        raise FileNotFoundError(f"No checkpoint file found matching pattern: {pattern} in {args.output_dir}")
    
    if len(matching_files) > 1:
        print(f"Warning: Multiple matching files found: {matching_files}")
        print(f"Using the first match: {matching_files[0]}")
    
    checkpoint_path = matching_files[0]
    print("Loading checkpoint from {}.".format(checkpoint_path))
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    state_dict = checkpoint["model"]
    # LoRA parameters typically have 'lora' in their names
    lora_keys = [k for k in state_dict.keys() if 'lora' in k.lower()]
    
    if lora_keys:
        print(f"Found {len(lora_keys)} LoRA parameters:")
        for key in lora_keys[:10]:  # Show first 10
            print(f"  - {key}")
        if len(lora_keys) > 10:
            print(f"  ... and {len(lora_keys) - 10} more")
    else:
        print("No LoRA parameters found in checkpoint")
    
    return model

