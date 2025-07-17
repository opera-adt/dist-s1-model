import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import math
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from einops import rearrange

# Dataset loader
from src.dataset import DistS1Dataset

# Core model and utilities
from src.dist_model_redux import SpatioTemporalTransformerRedux as SpatioTemporalTransformer
from src.utils import (
    GracefulKiller,
    WandBManager,
    load_checkpoint,
    load_config,
    load_validation_data,
    nll_gaussian,
    nll_gaussian_stable,
    run_final_validation,
    save_checkpoint,
    save_emergency_state,
    setup_warnings,
    validate_visual,
)
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split


def run_epoch_tf(dataloader, model, optimizer, device, pi, epoch, killer, accelerator, train_batch=True):
    """Perform one epoch of training by looping through the dataset once."""
    if train_batch:
        model.train()
    else:
        model.eval()

    nll_total = 0
    mse_total = 0
    naive_nll = 0
    naive_mse = 0

    num_batches = len(dataloader)
    batches_processed = 0

    for batch_idx, batch in enumerate(dataloader):
        # Check for interrupt signal
        if killer.kill_now:
            if accelerator.is_main_process:
                print(f'\nInterrupted at batch {batch_idx}/{num_batches}')
            break

        if batch_idx % 50 == 0 and accelerator.is_main_process:
            print(f'Batch {batch_idx}/{num_batches}')

        # Get input size from config or use default
        input_size = getattr(run_epoch_tf, '_input_size', 16)

        train_batch = batch['pre_imgs']
        target_batch = batch['post_img']
        # batch['acq_dts_float']

        train_batch = train_batch.to(device, non_blocking=True)
        target_batch = target_batch.to(device, non_blocking=True)

        train_batch.clamp_(0, math.pi)
        target_batch.clamp_(0, math.pi)

        # Data goes from Batch x Time X Channels X H x W -> (B h w) time channel ph pw, h = w = # of patches
        train_batch = rearrange(train_batch, 'b t c (h ph) (w pw) -> (b h w) t c ph pw', ph=input_size, pw=input_size)
        target_batch = rearrange(target_batch, 'b c (h ph) (w pw) -> (b h w) c ph pw', ph=input_size, pw=input_size)

        # target_batch = torch.special.logit(target_batch)
        # train_batch = torch.special.logit(train_batch)

        # mask_out_idx = random.randrange(train_batch.size(1))
        # train_batch = train_batch[:, mask_out_idx:, ...]

        # Mark the random masking as a graph break to prevent compilation issues
        # torch._dynamo.graph_break()

        if train_batch.shape[0] > 0:
            # Clear cache to prevent memory accumulation
            if batch_idx % 50 == 0:  # Clear every 50 batches
                torch.cuda.empty_cache()
            
            mask = ~torch.isnan(target_batch)  # True where value is NOT NaN

            pred_means, pred_logvars = model(train_batch)
            loss = nll_gaussian(pred_means, pred_logvars, target_batch, pi=pi)
            mse_loss = F.mse_loss(pred_means, target_batch)

            optimizer.zero_grad()
            accelerator.backward(loss)  # Use accelerator's backward
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

            # Gather losses from all processes for proper averaging
            loss_gathered = accelerator.gather(loss.detach())
            mse_gathered = accelerator.gather(mse_loss.detach())

            nll_total += loss_gathered.mean().cpu().item()
            mse_total += mse_gathered.mean().cpu().item()

            # Clean up intermediate tensors to save memory
            del pred_means, pred_logvars, loss, mse_loss, loss_gathered, mse_gathered

        else:
            with torch.no_grad():
                # Clear cache to prevent memory accumulation
                if batch_idx % 20 == 0:  # Clear more frequently during validation
                    torch.cuda.empty_cache()

                # Compute baseline
                pre_image_mean = torch.mean(train_batch, dim=1)
                pre_image_var = torch.var(train_batch, dim=1, unbiased=False)

                # Add larger epsilon for numerical stability
                eps = 1e-4  # Increased from 1e-8
                pre_image_var = pre_image_var + eps

                # Clamp variance to avoid extreme values that could cause NaN in log
                pre_image_var = torch.clamp(pre_image_var, min=1e-4, max=1e4)

                # Check for any remaining numerical issues
                if torch.any(torch.isnan(pre_image_var)) or torch.any(pre_image_var <= 0):
                    if accelerator.is_main_process:
                        print(
                            f'Warning: Invalid variance detected. Min: {pre_image_var.min()}, Max: {pre_image_var.max()}'
                        )
                    pre_image_var = torch.ones_like(pre_image_var) * 0.1  # Fallback to reasonable default

                naive_nll_loss = nll_gaussian_stable(pre_image_mean, pre_image_var, target_batch, pi)
                naive_mse_loss = F.mse_loss(pre_image_mean, target_batch)

                # Get prediction
                pred_means, pred_logvars = model(train_batch)
                loss = nll_gaussian(pred_means, pred_logvars, target_batch)
                mse_loss = F.mse_loss(pred_means, target_batch)

                # Gather losses from all processes for proper averaging
                loss_gathered = accelerator.gather(loss.detach())
                mse_gathered = accelerator.gather(mse_loss.detach())
                naive_nll_gathered = accelerator.gather(naive_nll_loss.detach())
                naive_mse_gathered = accelerator.gather(naive_mse_loss.detach())

                nll_total += loss_gathered.mean().cpu().item()
                mse_total += mse_gathered.mean().cpu().item()
                naive_nll += naive_nll_gathered.mean().cpu().item()
                naive_mse += naive_mse_gathered.mean().cpu().item()

                # Clean up intermediate tensors to save memory
                del pred_means, pred_logvars, loss, mse_loss
                del loss_gathered, mse_gathered, naive_nll_gathered, naive_mse_gathered
                del pre_image_mean, pre_image_var, naive_nll_loss, naive_mse_loss

        batches_processed += 1
       

    # Calculate averages based on processed batches
    if batches_processed > 0:
        nll_average = nll_total / batches_processed
        mse_average = mse_total / batches_processed
        naive_nll_average = naive_nll / batches_processed
        naive_mse_average = naive_mse / batches_processed
    else:
        nll_average = mse_average = naive_nll_average = naive_mse_average = 0

    return nll_average, mse_average, naive_nll_average, naive_mse_average


def left_pad_sequences(sequences, T_max, nodata_value=np.nan):
    batch_size = len(sequences)
    sample_shape = sequences[0].shape[1:] if sequences[0].ndim > 1 else ()

    padded = np.full((batch_size, T_max, *sample_shape), nodata_value, dtype=sequences[0].dtype)
    lengths = np.zeros(batch_size, dtype=np.int64)

    for i, seq in enumerate(sequences):
        T = seq.shape[0]
        lengths[i] = T
        padded[i, T_max - T :] = seq

    return padded, lengths


def custom_collate_fn(batch, T_max=21):
    pre_imgs = [item['pre_imgs'] for item in batch]
    post_img = np.stack([item['post_img'] for item in batch])
    dts = [item['acq_dts_float'] for item in batch]

    padded_pre_imgs, _ = left_pad_sequences(pre_imgs, T_max, nodata_value=-9999)
    # dts include the post-img date, so we add one to T_max
    padded_dts, _ = left_pad_sequences(dts, T_max + 1)

    return {
        'pre_imgs': torch.from_numpy(padded_pre_imgs),
        'post_img': torch.from_numpy(post_img),
        'acq_dts_float': torch.from_numpy(padded_dts),
    }


def main():
    # Initialize accelerator first
    accelerator = Accelerator(mixed_precision="fp16")

    # Setup warnings and debug info
    setup_warnings()

    # Debug multi-GPU setup
    print(f'Process {accelerator.process_index}: Accelerator state:')
    print(f'  - Device: {accelerator.device}')
    print(f'  - Num processes: {accelerator.num_processes}')
    print(f'  - Process index: {accelerator.process_index}')
    print(f'  - Local process index: {accelerator.local_process_index}')
    print(f'  - Available GPUs: {torch.cuda.device_count()}')
    print(f'  - Distributed type: {accelerator.distributed_type}')

    # Synchronize all processes before continuing
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print('All processes initialized successfully!')

    # Load configuration
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    config = load_config(config_path)

    # Check if we should disable dynamo compilation for this model
    if config.get('disable_dynamo_compilation', False):
        if accelerator.is_main_process:
            print('Disabling Dynamo compilation as requested in config')
        torch._dynamo.config.disable = True

    # Initialize graceful shutdown handler
    killer = GracefulKiller()

    # Initialize wandb (only on main process)
    wandb_manager = WandBManager(config, accelerator, enabled=config.get('use_wandb', True))

    # Set random seeds
    torch.manual_seed(config['train_config']['seed'])
    np.random.seed(config['train_config']['seed'])

    # Load data
    dist_dataset = DistS1Dataset(config['data']['data_dir_path'])
    train_size = int(0.8 * len(dist_dataset))
    test_size = len(dist_dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dist_dataset, [train_size, test_size], generator=generator)

    train_loader = DataLoader(
        train_dataset, batch_size=config['train_config']['batch_size'], shuffle=True, collate_fn=custom_collate_fn, pin_memory = True, 
        num_workers = 6, persistent_workers=True,  prefetch_factor=4, multiprocessing_context='fork' 
    )

    test_loader = DataLoader(
        test_dataset, batch_size=config['train_config']['batch_size'], shuffle=True, collate_fn=custom_collate_fn, pin_memory = True,
        num_workers = 6, persistent_workers=True,  prefetch_factor=4, multiprocessing_context='fork' 
    )


    # Debug dataset sizes
    if accelerator.is_main_process:
        print('Dataset sizes:')
        print(f'  - Train dataset: {len(train_dataset)} samples')
        print(f'  - Test dataset: {len(test_dataset)} samples')

    # Debug dataloader sizes before distributed setup
    if accelerator.is_main_process:
        print('DataLoader info before accelerator.prepare:')
        print(f'  - Train batches: {len(train_loader)}')
        print(f'  - Test batches: {len(test_loader)}')
        print(f'  - Train batch size: {config["train_config"]["batch_size"]}')
        print(f'  - Effective batch size per GPU: {config["train_config"]["batch_size"] // accelerator.num_processes}')

    # Initialize model
    model = SpatioTemporalTransformer(config['model_config'])
    if accelerator.is_main_process:
        print(f'Number of parameters: {model.num_parameters()}')

    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train_config']['learning_rate'])
    scheduler = StepLR(optimizer, step_size=config['train_config']['step_size'], gamma=config['train_config']['gamma'])

    # Prepare everything with accelerator
    model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, scheduler
    )

    # Debug dataloader sizes after distributed setup
    print(f'Process {accelerator.process_index} - DataLoader info after accelerator.prepare:')
    print(f'  - Train batches per GPU: {len(train_loader)}')
    print(f'  - Test batches per GPU: {len(test_loader)}')

    # Synchronize and calculate totals on main process
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f'Expected total train batches across all GPUs: {len(train_loader) * accelerator.num_processes}')
        print(f'Expected total test batches across all GPUs: {len(test_loader) * accelerator.num_processes}')

    # Verify model is on correct device
    print(f'Process {accelerator.process_index}: Model device = {next(model.parameters()).device}')
    accelerator.wait_for_everyone()  # Synchronize before continuing

    # Load checkpoint if resuming
    start_epoch = 1
    metrics_history = {
        'train_loss': [],
        'train_mse': [],
        'test_loss': [],
        'test_mse': [],
        'test_naive_nll': [],
        'test_naive_mse': [],
    }

    if config.get('resume_checkpoint'):
        if accelerator.is_main_process:
            print(f'Resuming from checkpoint: {config["resume_checkpoint"]}')
        start_epoch, metrics_history = load_checkpoint(
            config['resume_checkpoint'], model, optimizer, scheduler, accelerator
        )
        start_epoch += 1

    # Get input_size from config or use default
    input_size = config['model_config']['input_size']
    
    # Store input_size for run_epoch_tf to access
    run_epoch_tf.input_size = input_size

    if accelerator.is_main_process:
        print(f'Using input_size: {input_size}')

    # Training setup
    pi = torch.FloatTensor([np.pi]).to(accelerator.device)
    now = datetime.now().strftime('%m-%d-%Y_%H-%M')

    # Create directories (only on main process)
    if accelerator.is_main_process:
        Path(config['save_dir']['models']).mkdir(parents=True, exist_ok=True)
        Path(config['save_dir']['checkpoints']).mkdir(parents=True, exist_ok=True)
        if config.get('validation', {}).get('enable_visual_validation', False):
            Path(config['save_dir']['visualizations']).mkdir(parents=True, exist_ok=True)

    # Load validation data (only on main process)
    val_data = {}
    if accelerator.is_main_process:
        val_data = load_validation_data(config)

    try:
        # Training loop
        for epoch in range(start_epoch, config['train_config']['num_epochs'] + 1):
            if killer.kill_now:
                if accelerator.is_main_process:
                    print('\nReceived interrupt signal. Saving checkpoint and exiting...')
                break

            if accelerator.is_main_process:
                print(f'--- EPOCH [{epoch}/{config["train_config"]["num_epochs"]}] ---')
            epoch_start_time = time.time()

            # Debug GPU utilization at start of epoch
            if epoch == start_epoch:
                print(f'Process {accelerator.process_index} starting epoch {epoch} on device {accelerator.device}')
                if torch.cuda.is_available():
                    print(
                        f'Process {accelerator.process_index} GPU memory before training: {torch.cuda.memory_allocated(accelerator.device) / 1e9:.2f} GB'
                    )
                    print(
                        f'Process {accelerator.process_index} GPU memory cached: {torch.cuda.memory_reserved(accelerator.device) / 1e9:.2f} GB'
                    )
                accelerator.wait_for_everyone()

                if accelerator.is_main_process:
                    print('All processes synchronized, starting training...')

            # Add a small delay to see if it helps with process coordination
            if epoch == start_epoch:
                import time as time_module

                time_module.sleep(2)  # Give processes time to settle

            # Train
            train_loss, train_mse, _, _ = run_epoch_tf(
                train_loader, model, optimizer, accelerator.device, pi, epoch, killer, accelerator, train_batch=True
            )

            # Check again after training epoch
            if killer.kill_now:
                if accelerator.is_main_process:
                    print('\nInterrupted during training. Saving progress...')
                    # Still update metrics for partial epoch
                    if train_loss > 0:  # Only if we processed some batches
                        metrics_history['train_loss'].append(train_loss)
                        metrics_history['train_mse'].append(train_mse)

                # Save emergency checkpoint but skip validation during interruption
                last_epoch = save_emergency_state(
                    model,
                    optimizer,
                    scheduler,
                    len(metrics_history['train_loss']) + start_epoch - 1,
                    config,
                    metrics_history,
                    accelerator,
                    'training interruption',
                )
                break

            # Test
            test_loss, test_mse, test_naive_nll, test_naive_mse = run_epoch_tf(
                test_loader, model, optimizer, accelerator.device, pi, epoch, killer, accelerator, train_batch=False
            )

            # Update metrics history (only on main process to avoid duplication)
            if accelerator.is_main_process:
                metrics_history['train_loss'].append(train_loss)
                metrics_history['train_mse'].append(train_mse)
                metrics_history['test_loss'].append(test_loss)
                metrics_history['test_mse'].append(test_mse)
                metrics_history['test_naive_nll'].append(test_naive_nll)
                metrics_history['test_naive_mse'].append(test_naive_mse)

                # Log to wandb
                wandb_manager.log(
                    {
                        'epoch': epoch,
                        'train/nll': train_loss,
                        'train/mse': train_mse,
                        'test/nll': test_loss,
                        'test/mse': test_mse,
                        'test/naive_nll': test_naive_nll,
                        'test/naive_mse': test_naive_mse,
                        'learning_rate': scheduler.get_last_lr()[0],
                        'epoch_time_minutes': (time.time() - epoch_start_time) / 60,
                    },
                    step=epoch,
                )

                print(f'Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}')
                print(f'Train MSE: {train_mse:.6f}, Test MSE: {test_mse:.6f}')
                print(f'Test Naive NLL: {test_naive_nll:.6f}, Test Naive MSE: {test_naive_mse:.6f}')
                print(f'Time: {(time.time() - epoch_start_time) / 60:.2f} minutes\n')

            # Wait for all processes to finish the epoch
            accelerator.wait_for_everyone()

            # Run intermediate visual validation if enabled
            if (
                config.get('validation', {}).get('enable_intermediate_validation', False)
                and epoch % config.get('validation', {}).get('intermediate_validation_freq', 10) == 0
                and val_data
            ):
                if accelerator.is_main_process:
                    print(f'\nRunning intermediate visual validation at epoch {epoch}...')

                apply_smoothing = config.get('validation', {}).get('apply_smoothing', True)
                smooth_sigma = config.get('validation', {}).get('smooth_sigma', 0.5)
                blend_mode = config.get('validation', {}).get('blend_mode', 'gaussian')

                validate_visual(
                    model,
                    val_data,
                    epoch,
                    config['save_dir']['visualizations'],
                    wandb_manager,
                    accelerator,
                    apply_smoothing=apply_smoothing,
                    smooth_sigma=smooth_sigma,
                    blend_mode=blend_mode,
                    chip_size=input_size,  # Use configurable input_size
                )

            # Save checkpoint (only from main process)
            if epoch % config['train_config']['checkpoint_freq'] == 0:
                checkpoint_path = Path(config['save_dir']['checkpoints']) / f'checkpoint_epoch_{epoch}_{now}.pth'
                save_checkpoint(
                    model, optimizer, scheduler, epoch, config, metrics_history, checkpoint_path, accelerator
                )

                # Save model (only from main process)
                if accelerator.is_main_process:
                    model_path = (
                        Path(config['save_dir']['models']) / f'{config["model_config"]["type"]}_{now}_epoch_{epoch}.pth'
                    )
                    torch.save(accelerator.get_state_dict(model), model_path)

            scheduler.step()

        # Run visual validation at the end (before cleanup)
        if not killer.kill_now and epoch == config['train_config']['num_epochs']:
            run_final_validation(model, val_data, epoch, config, wandb_manager, accelerator)

    finally:
        # Wait for all processes before cleanup
        accelerator.wait_for_everyone()

        # Clean up wandb (only from main process)
        wandb_manager.finish()

        # Clean up distributed training properly
        accelerator.end_training()

        # Save emergency checkpoint if interrupted (only from main process)
        if killer.kill_now:
            # Save emergency state if we haven't already during training loop
            last_epoch = len(metrics_history['train_loss']) + start_epoch - 1
            save_emergency_state(
                model, optimizer, scheduler, last_epoch, config, metrics_history, accelerator, 'final interruption'
            )

        # Save final checkpoint only if training completed normally (only from main process)
        if not killer.kill_now and 'epoch' in locals() and epoch == config['train_config']['num_epochs']:
            if accelerator.is_main_process:
                final_checkpoint_path = Path(config['save_dir']['checkpoints']) / f'final_checkpoint_{now}.pth'
                save_checkpoint(
                    model, optimizer, scheduler, epoch, config, metrics_history, final_checkpoint_path, accelerator
                )

                # Save final model
                final_model_path = (
                    Path(config['save_dir']['models']) / f'{config["model_config"]["type"]}_{now}_final.pth'
                )
                torch.save(accelerator.get_state_dict(model), final_model_path)

        if accelerator.is_main_process:
            print('Training completed or interrupted. All resources cleaned up.')


if __name__ == '__main__':
    main()
