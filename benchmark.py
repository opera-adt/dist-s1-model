import io
import time
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from einops import rearrange
from torch.utils.data import DataLoader
import copy
from pathlib import Path
import pandas as pd
from datetime import datetime

from src.utils import load_models_and_configs
from src.dataset import StreamShardDataset

BENCHMARK_VARIABLE_SEQ_LEN = True
NUM_EVALUATION_RUNS = 1

# --- Core model (assuming it's in a sibling src folder) ---
from src.dist_model import SpatioTemporalTransformer

# --- Utilities from your src/utils.py file ---
from src.utils import (
    load_checkpoint,
    nll_gaussian,
    nll_gaussian_stable,
    setup_warnings,
)

# --- HELPER FUNCTIONS ---

def get_model(config):
    """Factory function to create a model based on the config."""
    model_type = config['model_config']['type']
    if model_type == 'transformer':
        # Pass the whole model_config dict to the constructor
        return SpatioTemporalTransformer(config['model_config'])
    # --- Add other models here ---
    # elif model_type == 'YourOtherModel':
    #     return YourOtherModel(config['model_config'])
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_dataloaders(config, batch_size):
    """Factory function to load datasets based on the config."""
    data_config = config['data']
    loader_type = data_config.get('type', 'dataset_v0')
    
    print(f"INFO: Using dataloader type '{loader_type}' for dataset '{data_config.get('name', 'N/A')}'")

    train_path = data_config.get('train_path')
    test_path = data_config.get('test_path')
    
    if not train_path or not test_path:
        raise ValueError("Data loader requires 'train_path' and 'test_path'.")

    if loader_type == 'dataset_v0':        
        train_dataset = torch.load(train_path, weights_only=False)
        test_dataset = torch.load(test_path, weights_only=False)
    elif loader_type == 'dataset_v1':
        # train_dataset =  StreamShardDataset(Path(train_path).glob("train_*.pt"))
        # test_dataset = StreamShardDataset(Path(test_path).glob("test_*.pt"))
        train_dataset =  StreamShardDataset(train_path, prefix="train")
        test_dataset = StreamShardDataset(train_path, prefix="test")

    else:
        raise ValueError(f"Unknown dataloader type: '{loader_type}'")

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader

def get_model_size_info(model):
    """Calculates model parameter count and size in megabytes."""
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_mb = buffer.tell() / (1024 * 1024)
    return param_count, size_mb

def safe_logit(tensor, eps=1e-5):
    """Applies a logit transformation with clamping for numerical stability."""
    return torch.special.logit(torch.clamp(tensor, min=eps, max=1.0 - eps))

def run_evaluation(dataloader, model, device, accelerator, pi, config):
    """
    Performs a single evaluation pass, automatically aligning sequence lengths
    based on the target_seq_len specified in the eval_config.
    """
    model.eval()
    batches_processed = 0
    nll_total, mse_total, naive_nll_total, naive_mse_total = 0, 0, 0, 0
    input_size = getattr(run_evaluation, '_input_size', 16)

    with torch.no_grad():
        for batch_idx, (batch, target) in enumerate(dataloader):
            batch = batch.to(device)
            target = target.to(device)

            batch = rearrange(batch, 'b t c (h ph) (w pw) -> (b h w) t c ph pw', ph=input_size, pw=input_size)
            target = rearrange(target, 'b c (h ph) (w pw) -> (b h w) c ph pw', ph=input_size, pw=input_size)
            
            # --- START: SEQUENCE ALIGNMENT BLOCK ---
            input_seq_len = batch.shape[1]
            
            # Get the target length from the eval_config, not the model_config
            eval_config = config.get('eval_config', {})
            target_seq_len = eval_config.get('target_seq_len')

            if target_seq_len is not None and input_seq_len != target_seq_len:
                # Case 1: Data sequence is LONGER than our target length
                if input_seq_len > target_seq_len:
                    if accelerator.is_main_process and batch_idx == 0: # Print only once per run
                        print(f"INFO: Data seq length ({input_seq_len}) > target ({target_seq_len}). Truncating data.")
                    batch = batch[:, -target_seq_len:, ...]

                # Case 2: Data sequence is SHORTER than our target length
                elif input_seq_len < target_seq_len:
                    if accelerator.is_main_process and batch_idx == 0: # Print only once per run
                        print(f"INFO: Data seq length ({input_seq_len}) < target ({target_seq_len}). Padding data.")
                    pad_amount = target_seq_len - input_seq_len
                    # Pad format is (left, right) for each dimension, starting from the last dim
                    batch = F.pad(batch, (0, 0, 0, 0, 0, 0, pad_amount, 0), "constant", 0)
            
            # --- END: SEQUENCE ALIGNMENT BLOCK ---

            batch = safe_logit(batch)
            target = safe_logit(target)

            # The forward pass will now work because the batch sequence length is correct
            pred_means, pred_logvars = model(batch)
            loss = nll_gaussian(pred_means, pred_logvars, target, pi=pi)
            mse_loss = F.mse_loss(pred_means, target)

            # --- Metric calculation and gathering ---
            pre_image_mean = torch.mean(batch, dim=1)
            pre_image_var = torch.var(batch, dim=1, unbiased=False).clamp(min=1e-4)
            naive_nll_loss = nll_gaussian_stable(pre_image_mean, pre_image_var, target, pi)
            naive_mse_loss = F.mse_loss(pre_image_mean, target)

            nll_total += accelerator.gather(loss.detach()).mean().item()
            mse_total += accelerator.gather(mse_loss.detach()).mean().item()
            naive_nll_total += accelerator.gather(naive_nll_loss.detach()).mean().item()
            naive_mse_total += accelerator.gather(naive_mse_loss.detach()).mean().item()
            
            batches_processed += 1

    # --- Metric averaging ---
    if batches_processed > 0:
        nll_avg = nll_total / batches_processed
        mse_avg = mse_total / batches_processed
        naive_nll_avg = naive_nll_total / batches_processed
        naive_mse_avg = naive_mse_total / batches_processed
    else:
        nll_avg, mse_avg, naive_nll_avg, naive_mse_avg = 0, 0, 0, 0

    return nll_avg, mse_avg, naive_nll_avg, naive_mse_avg

# --- MAIN EXPERIMENT RUNNER ---
def run_experiment(config):
    """
    Loads a model and runs repeated evaluations, returning a list of raw
    results, one for each individual run.
    """
    # --- 1. Initial Setup ---
    accelerator = Accelerator()
    setup_warnings()

    # --- 2. Model and Data Loading ---
    if accelerator.is_main_process:
        print(f"Loading model: {config['model_config'].get('name', 'N/A')}")
    model = get_model(config)
    train_loader, test_loader = get_dataloaders(config, config['train_config']['batch_size'])

    # --- 3. Accelerator Prepare ---
    model, train_loader, test_loader = accelerator.prepare(model, train_loader, test_loader)
    
    # --- 4. Load Pre-trained Weights ---
    if config.get('resume_checkpoint'):
        if accelerator.is_main_process:
            print(f"INFO: Loading weights from: {config['resume_checkpoint']}")
        load_checkpoint(config.get('resume_checkpoint'), model, accelerator=accelerator)
    else:
        if accelerator.is_main_process:
            print("Warning: No 'resume_checkpoint' path specified. Evaluating with initial weights.")

    # --- 5. Run Repeated Evaluations ---
    eval_config = config.get('eval_config', {})
    num_runs = eval_config.get('num_runs', 1)
    
    run_results_list = []

    # Get static model info once
    param_count, model_size_mb = 0, 0
    if accelerator.is_main_process:
        param_count, model_size_mb = get_model_size_info(model)

    if accelerator.is_main_process:
        print(f"Starting evaluation ({num_runs} runs)...")

    for i in range(num_runs):
        start_time = time.time()
        
        run_evaluation._input_size = config.get('train_config', {}).get('input_size', 16)
        pi = torch.FloatTensor([np.pi]).to(accelerator.device)
        
        # Evaluate on the TEST and TRAIN sets
        test_nll, test_mse, _, _ = run_evaluation(test_loader, model, accelerator.device, accelerator, pi, config)
        train_nll, train_mse, _, _ = run_evaluation(train_loader, model, accelerator.device, accelerator, pi, config)
        
        # On the main process, create a dictionary for this single run
        if accelerator.is_main_process:
            run_time_sec = time.time() - start_time
            peak_gpu_mem_mb = torch.cuda.max_memory_allocated(accelerator.device) / 1e6 if torch.cuda.is_available() else 0
            
            single_run_result = {
                "model": config['model_config'].get('name', config['model_config']['type']),
                "dataset": config['data']['name'],
                "target_seq_len": eval_config.get('target_seq_len', config.get('model_config', {}).get('max_seq_len')),
                "run_number": i + 1,
                "test_nll": test_nll,
                "test_mse": test_mse,
                "train_nll": train_nll,
                "train_mse": train_mse,
                "run_time_sec": run_time_sec,
                "peak_gpu_memory_mb": peak_gpu_mem_mb,
                "num_parameters_m": param_count / 1e6,
                "model_size_mb": model_size_mb,
            }
            run_results_list.append(single_run_result)
            print(f"Run {i+1}/{num_runs} complete. Test NLL: {test_nll:.6f}")

    # --- 6. Cleanup and Return ---
    accelerator.end_training()
    return run_results_list


# --- 1. CONFIGURE YOUR BENCHMARK ---
# Point this to the folder containing your .pth and .yml files.
MODELS_DIR = "model_data"

# Define the datasets you want to test each model against.
DATASETS_TO_TEST = [
    {
        "type": "dataset_v0",
        "name": "V0",
        "train_path": "PytorchData/train_12813.pt",
        "test_path": "PytorchData/test_3204.pt",
        "seq_len": 10 
    },
    {
        "type": "dataset_v1",
        "name": "V1",
        "train_path": "opera-dist-ml/data/v1/",
        "test_path": "opera-dist-ml/data/v1/",
        "seq_len": 20 
    },
]

# --- BENCHMARK EXECUTION ---
def main():
    models_to_test = load_models_and_configs(MODELS_DIR)
    if not models_to_test:
        print("No models found in the specified directory. Exiting.")
        return

    EXPERIMENTS = []
    for model_info in models_to_test:
        for dataset_def in DATASETS_TO_TEST:
            # Get the max length from the model's config
            model_max_seq_len = model_info['config'].get('model_config', {}).get('max_seq_len')
            # Get the sequence length from our new dataset definition
            dataset_seq_len = dataset_def.get('seq_len')

            # --- START: NEW LOGIC TO DETERMINE THE TEST RANGE ---
            upper_bound = 0
            if BENCHMARK_VARIABLE_SEQ_LEN:
                if model_max_seq_len and dataset_seq_len:
                    # The cap is the SMALLER of the model's max or the dataset's length
                    upper_bound = min(model_max_seq_len, dataset_seq_len)
                    print(f"INFO: For [{model_info['name']}] on [{dataset_def['name']}], capping benchmark at seq_len={upper_bound} (min(model={model_max_seq_len}, data={dataset_seq_len})).")
                else:
                    # If one is missing, use whichever value is available
                    upper_bound = model_max_seq_len or dataset_seq_len
            else:
                # If not benchmarking variable lengths, just use the model's default
                upper_bound = model_max_seq_len

            if not upper_bound:
                 # Fallback if no length info is available at all
                seq_len_to_test = [None]
            else:
                seq_len_to_test = range(1, upper_bound + 1) if BENCHMARK_VARIABLE_SEQ_LEN else [upper_bound]
            # --- END: NEW LOGIC ---

            for target_seq_len in seq_len_to_test:
                exp_config = copy.deepcopy(model_info['config'])
                # ... (the rest of the experiment setup is the same as before) ...
                exp_config['resume_checkpoint'] = model_info['model_path']
                exp_config['data'] = dataset_def
                if 'name' not in exp_config['model_config']:
                    exp_config['model_config']['name'] = model_info['name']

                if 'eval_config' not in exp_config:
                    exp_config['eval_config'] = {}
                exp_config['eval_config']['num_runs'] = NUM_EVALUATION_RUNS
                exp_config['eval_config']['target_seq_len'] = target_seq_len

                EXPERIMENTS.append(exp_config)

    # --- Run all generated experiments ---
    all_results = []
    total_experiments = len(EXPERIMENTS)
    for i, config in enumerate(EXPERIMENTS):
        model_name = config['model_config']['name']
        dataset_name = config['data']['name']
        target_len = config['eval_config']['target_seq_len']
        
        print(f"\n{'='*80}")
        print(f"RUNNING EXPERIMENT {i+1}/{total_experiments}: [{model_name}] on [{dataset_name}] with target_seq_len=[{target_len}]")
        print(f"{'='*80}")
        
        # run_experiment now returns a LIST of dictionaries
        list_of_run_results = run_experiment(config)
        
        # Use .extend() to add all items from the list to our main results
        if list_of_run_results:
            all_results.extend(list_of_run_results)
            
    # --- Save the final, detailed results to a single CSV file ---
    if not all_results:
        print("\nBenchmark finished, but no results were collected (this is expected on non-main processes).")
        return

    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = Path(f'benchmark_results_{now}')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n\n{'='*80}")
    print(f"Benchmark finished! Saving all raw run data to: {output_dir}")

    df = pd.DataFrame(all_results)
    output_path = output_dir / 'benchmark_raw_results.csv'
    df.to_csv(output_path, index=False)

    print(f"Results saved to '{output_path}'.")
    print(f"{'='*80}")


if __name__ == '__main__':
    dummy_dir = Path(MODELS_DIR)
    if not dummy_dir.exists():
        print(f"Creating dummy directory '{dummy_dir}' for testing...")
        dummy_dir.mkdir(exist_ok=True)
        (dummy_dir / "model_A.pth").touch()
        (dummy_dir / "model_A.yml").write_text(
            "model_config:\n  type: 'SpatioTemporalTransformer'\n  name: 'Model_A_Small'\n  d_model: 128\n"
            "train_config:\n  num_epochs: 1\n  batch_size: 4\n  learning_rate: 0.001\n  seed: 42\n  step_size: 1\n  gamma: 0.1\n"
        )
    
    main()