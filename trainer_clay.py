"""
Trainer for Clay Temporal Prediction Model.

This trainer uses pre-computed Clay embeddings to train a temporal model that
predicts future SAR images. The model:
1. Takes Clay embeddings of 15 pre-images + their acquisition dates
2. Uses the post-image acquisition date (but NOT the actual post-image)
3. Predicts the Clay embedding at the post-date
4. Decodes the predicted embedding back to SAR space

Loss = α * L_embedding + β * L_SAR
- L_embedding: MSE between predicted and actual Clay embedding
- L_SAR: Asymmetric NLL between predicted and actual SAR image

Based on trainer_redux.py architecture.
"""

import math
import random
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LambdaLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)
from torch.utils.data import DataLoader, random_split

# Local imports
import yaml
from src.clay_dataset import ClayEmbeddingDataset, clay_collate_fn
from src.clay_temporal_model import ClayTemporalPredictor
from src.utils import (
    GracefulKiller,
    WandBManager,
    get_test_batch_clay,
    nll_gaussian_asymmetric,
    nll_gaussian_regularized,
    save_checkpoint,
    save_emergency_state,
    setup_warnings,
    show_prediction_vs_groundtruth_wandb,
)


def load_clay_config(config_path: str) -> dict:
    """Load and validate Clay model configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Train config validation
    train_cfg = config['train_config']
    train_cfg['learning_rate'] = float(train_cfg['learning_rate'])
    train_cfg['batch_size'] = int(train_cfg['batch_size'])
    train_cfg['num_epochs'] = int(train_cfg['num_epochs'])
    train_cfg['seed'] = int(train_cfg['seed'])
    train_cfg['checkpoint_freq'] = int(train_cfg['checkpoint_freq'])

    if 'eta_min' in train_cfg:
        train_cfg['eta_min'] = float(train_cfg['eta_min'])

    # Model config validation
    model_cfg = config['model_config']
    model_cfg['d_model'] = int(model_cfg['d_model'])
    model_cfg['nhead'] = int(model_cfg['nhead'])
    model_cfg['num_encoder_layers'] = int(model_cfg['num_encoder_layers'])
    model_cfg['dim_feedforward'] = int(model_cfg['dim_feedforward'])
    model_cfg['dropout'] = float(model_cfg['dropout'])
    model_cfg['clay_dim'] = int(model_cfg.get('clay_dim', 1024))
    model_cfg['clay_spatial'] = int(model_cfg.get('clay_spatial', 32))
    model_cfg['patch_grid'] = int(model_cfg.get('patch_grid', 8))
    model_cfg['fourier_freqs'] = int(model_cfg.get('fourier_freqs', 64))
    model_cfg['fourier_max_freq'] = float(model_cfg.get('fourier_max_freq', 10.0))

    return config


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.0):
    """
    Create a learning rate scheduler with linear warmup and cosine decay.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)

    return LambdaLR(optimizer, lr_lambda)


def run_epoch(
    dataloader,
    model,
    optimizer,
    device,
    epoch,
    killer,
    accelerator,
    config,
    scheduler=None,
    scheduler_type=None,
    is_training=True,
):
    """
    Run one epoch of training or validation.

    Returns:
        tuple: (total_loss, embedding_loss, sar_loss, sar_mse)
    """
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss_sum = 0.0
    emb_loss_sum = 0.0
    sar_nll_sum = 0.0
    sar_mse_sum = 0.0
    grad_loss_sum = 0.0
    num_batches = len(dataloader)
    batches_processed = 0

    # Loss weights
    emb_weight = config['train_config'].get('embedding_loss_weight', 1.0)
    sar_weight = config['train_config'].get('sar_loss_weight', 1.0)

    # Asymmetric loss config
    use_asymmetric_loss = config['train_config'].get('use_asymmetric_loss', True)
    positive_weight = config['train_config'].get('positive_weight', 1.5)
    negative_weight = config['train_config'].get('negative_weight', 1.0)
    variance_penalty = config['train_config'].get('variance_penalty', 0.5)
    min_variance = config['train_config'].get('min_variance', 1.5)

    # Pi for NLL loss
    pi = torch.tensor([np.pi], device=device)

    context = torch.no_grad() if not is_training else torch.enable_grad()

    with context:
        for batch_idx, batch in enumerate(dataloader):
            # Check for interrupt
            if killer.kill_now:
                if accelerator.is_main_process:
                    print(f'\nInterrupted at batch {batch_idx}/{num_batches}')
                break

            if batch_idx % 10 == 0 and accelerator.is_main_process:
                mode = "Train" if is_training else "Val"
                print(f'{mode} Batch {batch_idx}/{num_batches}')

            # Move data to device
            pre_embeddings = batch['pre_embeddings'].to(device, non_blocking=True)  # (B, 15, 1024, 32, 32)
            post_embedding = batch['post_embedding'].to(device, non_blocking=True)  # (B, 1024, 32, 32)
            pre_dates = batch['pre_dates'].to(device, non_blocking=True)  # (B, 15)
            post_date = batch['post_date'].to(device, non_blocking=True)  # (B, 1)
            post_sar = batch['post_sar'].to(device, non_blocking=True)  # (B, 2, 256, 256)

            # Clear cache periodically
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()

            # Forward pass
            pred_embedding, pred_sar_mean, pred_sar_logvar = model(
                pre_embeddings, pre_dates, post_date
            )

            # --- Embedding Loss (MSE) ---
            emb_loss = F.mse_loss(pred_embedding, post_embedding)

            # --- SAR Loss (NLL) ---
            # Create mask for valid (non-NaN) SAR values
            mask = ~torch.isnan(post_sar)

            if use_asymmetric_loss:
                sar_loss = nll_gaussian_asymmetric(
                    pred_sar_mean, pred_sar_logvar, post_sar,
                    mask=mask, pi=pi,
                    variance_penalty=variance_penalty,
                    min_variance=min_variance,
                    positive_weight=positive_weight,
                    negative_weight=negative_weight,
                )
            else:
                sar_loss = nll_gaussian_regularized(
                    pred_sar_mean, pred_sar_logvar, post_sar,
                    mask=mask, pi=pi,
                    variance_penalty=variance_penalty,
                    min_variance=min_variance,
                )

            # SAR MSE (for logging)
            sar_mse = F.mse_loss(pred_sar_mean[mask], post_sar[mask])

            # --- Combined Loss ---
            total_loss = emb_weight * emb_loss + sar_weight * sar_loss

            # --- Gradient sharpness loss (L1 on spatial finite differences, NaN-aware) ---
            # Speckle gradients cancel across SGD batches; structural edge gradients reinforce.
            lambda_grad = config['train_config'].get('lambda_grad', 0.0)
            if lambda_grad > 0:
                mask_dx = mask[:, :, :, 1:] & mask[:, :, :, :-1]
                mask_dy = mask[:, :, 1:, :] & mask[:, :, :-1, :]
                pred_dx = pred_sar_mean[:, :, :, 1:] - pred_sar_mean[:, :, :, :-1]
                pred_dy = pred_sar_mean[:, :, 1:, :] - pred_sar_mean[:, :, :-1, :]
                tgt_dx  = post_sar[:, :, :, 1:] - post_sar[:, :, :, :-1]
                tgt_dy  = post_sar[:, :, 1:, :] - post_sar[:, :, :-1, :]
                dx_loss = F.l1_loss(pred_dx[mask_dx], tgt_dx[mask_dx]) if mask_dx.any() else torch.tensor(0.0, device=total_loss.device)
                dy_loss = F.l1_loss(pred_dy[mask_dy], tgt_dy[mask_dy]) if mask_dy.any() else torch.tensor(0.0, device=total_loss.device)
                grad_loss = dx_loss + dy_loss
                total_loss = total_loss + lambda_grad * grad_loss
            else:
                grad_loss = torch.tensor(0.0, device=total_loss.device)

            # Debug NaN detection
            if torch.isnan(total_loss):
                print(f"NaN detected in loss at batch {batch_idx}")
                print(f"  emb_loss: {emb_loss.item()}, sar_loss: {sar_loss.item()}")
                continue

            # Backward pass (training only)
            if is_training:
                optimizer.zero_grad()
                accelerator.backward(total_loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                # Step scheduler per batch for OneCycleLR and WarmupCosine
                if scheduler is not None and scheduler_type in ['OneCycleLR', 'WarmupCosine']:
                    scheduler.step()

            # Gather losses from all processes
            total_gathered = accelerator.gather(total_loss.detach())
            emb_gathered = accelerator.gather(emb_loss.detach())
            sar_gathered = accelerator.gather(sar_loss.detach())
            mse_gathered = accelerator.gather(sar_mse.detach())
            grad_gathered = accelerator.gather(grad_loss.detach())

            total_loss_sum += total_gathered.mean().cpu().item()
            emb_loss_sum += emb_gathered.mean().cpu().item()
            sar_nll_sum += sar_gathered.mean().cpu().item()
            sar_mse_sum += mse_gathered.mean().cpu().item()
            grad_loss_sum += grad_gathered.mean().cpu().item()

            batches_processed += 1

            # Log batch loss
            if batch_idx % 10 == 0 and accelerator.is_main_process:
                print(f"  Loss: {total_loss.item():.4f} (emb: {emb_loss.item():.4f}, sar_nll: {sar_loss.item():.4f}, grad: {grad_loss.item():.4f})")

            # Clean up
            del pred_embedding, pred_sar_mean, pred_sar_logvar
            del total_loss, emb_loss, sar_loss, sar_mse, grad_loss
            del total_gathered, emb_gathered, sar_gathered, mse_gathered, grad_gathered

    # Calculate averages
    if batches_processed > 0:
        total_avg = total_loss_sum / batches_processed
        emb_avg = emb_loss_sum / batches_processed
        sar_nll_avg = sar_nll_sum / batches_processed
        sar_mse_avg = sar_mse_sum / batches_processed
        grad_avg = grad_loss_sum / batches_processed
    else:
        total_avg = emb_avg = sar_nll_avg = sar_mse_avg = grad_avg = 0.0

    return total_avg, emb_avg, sar_nll_avg, sar_mse_avg, grad_avg


def main():
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision="bf16")

    # Setup warnings
    setup_warnings()

    # Debug info
    print(f'Process {accelerator.process_index}: Device={accelerator.device}, '
          f'Num processes={accelerator.num_processes}')
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print('All processes initialized successfully!')

    # Load configuration
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config_clay.yml'
    config = load_clay_config(config_path)

    # Initialize graceful shutdown handler
    killer = GracefulKiller()

    # Initialize WandB
    wandb_manager = WandBManager(config, accelerator, enabled=config.get('use_wandb', True))

    # Set random seeds
    torch.manual_seed(config['train_config']['seed'])
    np.random.seed(config['train_config']['seed'])

    # Load dataset
    if accelerator.is_main_process:
        print('Loading Clay embedding dataset...')

    dataset = ClayEmbeddingDataset(
        embeddings_dir=config['data']['embeddings_dir'],
        original_data_dir=config['data']['original_data_dir'],
        apply_db_transform=config['data'].get('apply_db_transform', True),
        db_epsilon=config['data'].get('db_epsilon', 1e-10),
        db_min=config['data'].get('db_min', -30.0),
        db_max=config['data'].get('db_max', 10.0),
    )

    # Train/test split (80/20)
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

    if accelerator.is_main_process:
        print(f'Dataset sizes: Train={len(train_dataset)}, Test={len(test_dataset)}')

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train_config']['batch_size'],
        shuffle=True,
        collate_fn=clay_collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['train_config']['batch_size'],
        shuffle=False,
        collate_fn=clay_collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # Initialize model
    model = ClayTemporalPredictor(config['model_config'])
    if accelerator.is_main_process:
        print(f'Model parameters: {model.num_parameters():,}')

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['train_config']['learning_rate'],
        weight_decay=0.01,
    )

    # Initialize scheduler
    scheduler_type = config['train_config'].get('scheduler_type', 'CosineAnnealingLR')

    if scheduler_type == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config['train_config']['num_epochs'],
            eta_min=config['train_config'].get('eta_min', 1e-6),
        )
    elif scheduler_type == 'WarmupCosine':
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * config['train_config']['num_epochs']
        warmup_epochs = config['train_config'].get('warmup_epochs', 5)
        warmup_steps = warmup_epochs * steps_per_epoch
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            min_lr_ratio=config['train_config'].get('min_lr_ratio', 0.0),
        )
    elif scheduler_type == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['train_config'].get('plateau_factor', 0.5),
            patience=config['train_config'].get('plateau_patience', 10),
            min_lr=config['train_config'].get('eta_min', 1e-7),
        )
    else:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config['train_config']['num_epochs'],
            eta_min=1e-6,
        )

    if accelerator.is_main_process:
        print(f'Using scheduler: {scheduler_type}')

    # Prepare with accelerator
    model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, scheduler
    )

    # Create directories
    if accelerator.is_main_process:
        Path(config['save_dir']['models']).mkdir(parents=True, exist_ok=True)
        Path(config['save_dir']['checkpoints']).mkdir(parents=True, exist_ok=True)

    # Training state
    start_epoch = 1
    metrics_history = {
        'train_total_loss': [],
        'train_emb_loss': [],
        'train_sar_nll': [],
        'train_sar_mse': [],
        'train_grad_loss': [],
        'test_total_loss': [],
        'test_emb_loss': [],
        'test_sar_nll': [],
        'test_sar_mse': [],
        'test_grad_loss': [],
    }

    # Resume from checkpoint if specified
    if config.get('resume_checkpoint'):
        if accelerator.is_main_process:
            print(f'Resuming from: {config["resume_checkpoint"]}')
        checkpoint = torch.load(config['resume_checkpoint'], map_location=accelerator.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        metrics_history = checkpoint.get('metrics', metrics_history)

    now = datetime.now().strftime('%m-%d-%Y_%H-%M')

    try:
        # Training loop
        for epoch in range(start_epoch, config['train_config']['num_epochs'] + 1):
            if killer.kill_now:
                if accelerator.is_main_process:
                    print('\nReceived interrupt signal. Saving and exiting...')
                break

            if accelerator.is_main_process:
                print(f'\n=== EPOCH [{epoch}/{config["train_config"]["num_epochs"]}] ===')
            epoch_start = time.time()

            # Train
            train_total, train_emb, train_sar_nll, train_sar_mse, train_grad = run_epoch(
                train_loader, model, optimizer, accelerator.device,
                epoch, killer, accelerator, config,
                scheduler=scheduler, scheduler_type=scheduler_type, is_training=True,
            )

            if killer.kill_now:
                break

            # Validate
            test_total, test_emb, test_sar_nll, test_sar_mse, test_grad = run_epoch(
                test_loader, model, optimizer, accelerator.device,
                epoch, killer, accelerator, config,
                scheduler=None, scheduler_type=None, is_training=False,
            )

            # Update metrics (main process only)
            if accelerator.is_main_process:
                metrics_history['train_total_loss'].append(train_total)
                metrics_history['train_emb_loss'].append(train_emb)
                metrics_history['train_sar_nll'].append(train_sar_nll)
                metrics_history['train_sar_mse'].append(train_sar_mse)
                metrics_history['train_grad_loss'].append(train_grad)
                metrics_history['test_total_loss'].append(test_total)
                metrics_history['test_emb_loss'].append(test_emb)
                metrics_history['test_sar_nll'].append(test_sar_nll)
                metrics_history['test_sar_mse'].append(test_sar_mse)
                metrics_history['test_grad_loss'].append(test_grad)

                # Log to WandB
                lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else config['train_config']['learning_rate']
                wandb_manager.log({
                    'epoch': epoch,
                    'train/total_loss': train_total,
                    'train/emb_loss': train_emb,
                    'train/sar_nll': train_sar_nll,
                    'train/sar_mse': train_sar_mse,
                    'train/grad_loss': train_grad,
                    'test/total_loss': test_total,
                    'test/emb_loss': test_emb,
                    'test/sar_nll': test_sar_nll,
                    'test/sar_mse': test_sar_mse,
                    'test/grad_loss': test_grad,
                    'learning_rate': lr,
                    'epoch_time_minutes': (time.time() - epoch_start) / 60,
                }, step=epoch)

                # Print summary
                print(f'\nEpoch {epoch} Summary:')
                print(f'  Train - Total: {train_total:.4f}, Emb: {train_emb:.4f}, SAR NLL: {train_sar_nll:.4f}, SAR MSE: {train_sar_mse:.4f}, Grad: {train_grad:.4f}')
                print(f'  Test  - Total: {test_total:.4f}, Emb: {test_emb:.4f}, SAR NLL: {test_sar_nll:.4f}, SAR MSE: {test_sar_mse:.4f}, Grad: {test_grad:.4f}')
                print(f'  Time: {(time.time() - epoch_start) / 60:.2f} min, LR: {lr:.2e}')

            accelerator.wait_for_everyone()

            # Save checkpoint
            if epoch % config['train_config']['checkpoint_freq'] == 0:
                lambda_grad = config['train_config'].get('lambda_grad', 0.0)
                grad_tag = f'_grad{lambda_grad}' if lambda_grad > 0 else ''
                checkpoint_path = Path(config['save_dir']['checkpoints']) / f'checkpoint_clay{grad_tag}_epoch_{epoch}_{now}.pth'
                save_checkpoint(
                    model, optimizer, scheduler, epoch, config, metrics_history,
                    checkpoint_path, accelerator,
                )

                if accelerator.is_main_process:
                    model_path = Path(config['save_dir']['models']) / f'clay_temporal{grad_tag}_epoch_{epoch}_{now}.pth'
                    torch.save(accelerator.get_state_dict(model), model_path)
                    print(f'Saved checkpoint and model at epoch {epoch}')

                    # Visualize and log test samples to WandB
                    pred_mean, pred_logvar, post_sar_batch, pre_sar_batch = get_test_batch_clay(
                        test_loader, model, accelerator.device, dataset
                    )
                    batch_size = pred_mean.shape[0]
                    sample_indices = random.sample(range(batch_size), min(10, batch_size))

                    for idx in sample_indices:
                        show_prediction_vs_groundtruth_wandb(
                            pred=pred_mean[idx],
                            log_var=pred_logvar[idx],
                            truth=post_sar_batch[idx],
                            pre_imgs=pre_sar_batch[idx],  # (2, 2, 256, 256) — last 2 pre-SAR
                            idx=idx,
                            pad_val=-9999.0,
                            wandb_manager=wandb_manager,
                        )

            # Step scheduler (per epoch)
            if scheduler_type == 'ReduceLROnPlateau':
                scheduler.step(test_total)
            elif scheduler_type not in ['OneCycleLR', 'WarmupCosine']:
                scheduler.step()

    finally:
        accelerator.wait_for_everyone()
        wandb_manager.finish()
        accelerator.end_training()

        # Save emergency checkpoint if interrupted
        if killer.kill_now:
            last_epoch = len(metrics_history['train_total_loss']) + start_epoch - 1
            save_emergency_state(
                model, optimizer, scheduler, last_epoch, config,
                metrics_history, accelerator, 'interruption',
            )

        if accelerator.is_main_process:
            print('Training completed or interrupted. Cleaned up.')


if __name__ == '__main__':
    main()
