import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import expit, logit
import random
import yaml
import wandb
import signal
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import time

import os
from datetime import datetime
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import ToTensor

from src.dist_model import SpatioTemporalTransformer
from src.utils import nll_gaussian, nll_gaussian_stable, visualize_reconstruction

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class WandBManager:
    def __init__(self, config, enabled=True):
        self.enabled = enabled
        self.run = None
        if self.enabled:
            self.run = wandb.init(
                entity=config.get('wandb_entity', None),
                project=config.get('wandb_project', 'spatiotemporal-transformer'),
                name=config.get('wandb_run_name', None),
                config=config,
                resume=config.get('resume_wandb_run_id', None),
                settings=wandb.Settings(
                    start_method="thread",
                    _disable_stats=False,  # Disable system stats collection
                    _disable_meta=True,   # Disable metadata collection
                    save_code=False,      # Don't save code/diffs
                    anonymous="never"     # Don't upload anonymous data
                )
            )
            
    def log(self, metrics, step=None):
        if self.enabled and self.run:
            wandb.log(metrics, step=step)
            
    def finish(self):
        if self.enabled and self.run:
            wandb.finish()


class GracefulKiller:
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        print('\nReceived signal to terminate. Finishing current epoch...')
        self.kill_now = True


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure numeric values are properly typed
    config['train_config']['learning_rate'] = float(config['train_config']['learning_rate'])
    config['train_config']['batch_size'] = int(config['train_config']['batch_size'])
    config['train_config']['num_epochs'] = int(config['train_config']['num_epochs'])
    config['train_config']['seed'] = int(config['train_config']['seed'])
    config['train_config']['step_size'] = int(config['train_config']['step_size'])
    config['train_config']['gamma'] = float(config['train_config']['gamma'])
    config['train_config']['checkpoint_freq'] = int(config['train_config']['checkpoint_freq'])
    
    # Model config
    config['model_config']['patch_size'] = int(config['model_config']['patch_size'])
    config['model_config']['num_patches'] = int(config['model_config']['num_patches'])
    config['model_config']['data_dim'] = int(config['model_config']['data_dim'])
    config['model_config']['d_model'] = int(config['model_config']['d_model'])
    config['model_config']['nhead'] = int(config['model_config']['nhead'])
    config['model_config']['num_encoder_layers'] = int(config['model_config']['num_encoder_layers'])
    config['model_config']['dim_feedforward'] = int(config['model_config']['dim_feedforward'])
    config['model_config']['max_seq_len'] = int(config['model_config']['max_seq_len'])
    config['model_config']['dropout'] = float(config['model_config']['dropout'])
    
    return config


def save_checkpoint(model, optimizer, scheduler, epoch, config, metrics, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config,
        'metrics': metrics
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def load_validation_data(config):
    """Load validation datasets for visual evaluation"""
    val_data = {}
    
    if config.get('validation', {}).get('enable_visual_validation', False):
        val_config = config['validation']
        
        # Landslide
        if val_config.get('landslide', {}).get('enabled', False):
            target = Image.open(val_config['landslide']['mask_path'])
            target = ToTensor()(target)
            target[target<-1] = -1
            target_landslide = target[0,1248:1440,448:640]
            
            landslide = torch.load(val_config['landslide']['data_path'])
            landslide = landslide[9:20,:,:,:]
            landslide_logit = torch.special.logit(landslide).float()
            
            val_data['landslide'] = {
                'pre': landslide_logit[:-1, :, : , :].unsqueeze(dim=0).to(device),
                'post': landslide_logit[-1, :, : , :].unsqueeze(dim=0),
                'target': target_landslide,
                'stride': val_config['landslide'].get('stride', 1)
            }
        
        # Fire
        if val_config.get('fire', {}).get('enabled', False):
            row_min = 1500
            row_max = row_min+224*7
            col_min = 1000
            col_max = col_min+224*5
            
            target_fire = torch.load(val_config['fire']['mask_path'], weights_only=True)
            target_fire = target_fire[row_min:row_max,col_min:col_max]
            
            fire_data = torch.load(val_config['fire']['data_path'], weights_only=True)
            fire_data = fire_data[:,:,row_min:row_max,col_min:col_max]
            fire_data_logit = torch.special.logit(fire_data).float()
            
            val_data['fire'] = {
                'pre': fire_data_logit[:7, :, : , :].unsqueeze(dim=0).to(device),
                'post': fire_data_logit[-1, :, : , :].unsqueeze(dim=0),
                'target': target_fire,
                'stride': val_config['fire'].get('stride', 4)
            }
        
        # Flood
        if val_config.get('flood', {}).get('enabled', False):
            target_bangladesh = torch.load(val_config['flood']['mask_path'])
            target_bangladesh = target_bangladesh[1248:,1248:]
            
            bangladesh_data = torch.load(val_config['flood']['data_path'])
            bangladesh_data = bangladesh_data[:,:,1248:,1248:]
            bangladesh_logit = torch.special.logit(bangladesh_data).float()
            
            val_data['flood'] = {
                'pre': bangladesh_logit[:4, :, : , :].unsqueeze(dim=0).to(device),
                'post': bangladesh_logit[-1, :, : , :].unsqueeze(dim=0),
                'target': target_bangladesh,
                'stride': val_config['flood'].get('stride', 4)
            }
    
    return val_data


def make_preds_sliding(model, pre_imgs, chip_size=16, stride=1, flatten=False):
    """Generate predictions using sliding window approach"""
    assert stride <= chip_size and stride > 0
    
    data_dim_row = pre_imgs.shape[-2]
    data_dim_col = pre_imgs.shape[-1]
    
    assert data_dim_row % chip_size == 0
    assert data_dim_col % chip_size == 0
    
    pred_means = torch.zeros((1,2,data_dim_row,data_dim_col), device=pre_imgs.device)
    pred_logvars = torch.zeros_like(pred_means)
    div_mat = torch.zeros_like(pred_means)
    
    index_range_row = int(data_dim_row - chip_size + 1)
    index_range_col = int(data_dim_col - chip_size + 1)
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0,index_range_row,stride), desc="Rows", leave=False):
            for j in range(0,index_range_col,stride):
                if flatten:
                    chip_mean, chip_logvar = model(pre_imgs[:,:,:,i:(i+chip_size),j:(j+chip_size)].flatten(start_dim=2))
                else:
                    chip_mean, chip_logvar = model(pre_imgs[:,:,:,i:(i+chip_size),j:(j+chip_size)])
                
                pred_means[:,:,i:(i+chip_size),j:(j+chip_size)] += chip_mean.reshape((1,2,chip_size,chip_size))
                pred_logvars[:,:,i:(i+chip_size),j:(j+chip_size)] += chip_logvar.reshape((1,2,chip_size,chip_size))
                div_mat[:,:,i:(i+chip_size),j:(j+chip_size)] += 1
    
    pred_means = pred_means / div_mat
    pred_logvars = pred_logvars / div_mat
    
    return pred_means, pred_logvars


def validate_visual(model, val_data, epoch, save_dir, wandb_manager=None):
    """Perform visual validation on test cases"""
    if not val_data:
        return
    
    val_results = {}
    
    for event_name, data in val_data.items():
        print(f"\nValidating {event_name}...")
        
        # Make predictions
        pred_means, pred_logvars = make_preds_sliding(
            model, 
            data['pre'][:, :-1, ...],  # Exclude last pre-image
            chip_size=16, 
            flatten=False, 
            stride=data['stride']
        )
        
        # Generate visualization
        damage_map, log_ratio_im = visualize_reconstruction(
            data['pre'][0,...], 
            data['post'][0,...], 
            pred_means[0,...], 
            pred_logvars[0,...], 
            'Transformer',  # Changed from f'Transformer - {event_name}'
            is_logit=True, 
            vmax_vv=.3, 
            vmax_vh=.1
        )
        
        # Save figure
        save_path = Path(save_dir) / f'{event_name}_epoch_{epoch}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Log to wandb if enabled
        if wandb_manager:
            wandb_manager.log({
                f'validation/{event_name}_visualization': wandb.Image(str(save_path))
            }, step=epoch)
    
    model.train()  # Set back to training mode


def run_epoch_tf(dataloader, model, optimizer, device, pi, epoch, killer, train=True):
    """Perform one epoch of training by looping through the dataset once."""
    
    if train:
        model.train()
    else:
        model.eval()

    nll_total = 0
    mse_total = 0
    naive_nll = 0
    naive_mse = 0
    
    num_batches = len(dataloader)
    batches_processed = 0

    for batch_idx, (batch, target) in enumerate(dataloader):
        
        # Check for interrupt signal
        if killer.kill_now:
            print(f"\nInterrupted at batch {batch_idx}/{num_batches}")
            break
            
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{num_batches}")

        input_size = 16
        
        batch = batch.to(device)
        target = target.to(device)

        # Keep the exact data loading logic as in original
        batch = batch.unfold(3, input_size, input_size).unfold(4, input_size, input_size)
        target = target.unfold(2, input_size, input_size).unfold(3, input_size, input_size)

        batch = batch.permute(0, 3, 4, 1, 2, 5, 6).reshape(-1, 10, 2, input_size, input_size)
        target = target.permute(0, 2, 3, 1, 4, 5).reshape(-1, 2, input_size, input_size)

        target = torch.special.logit(target)
        batch = torch.special.logit(batch)

        mask_out_idx = random.randrange(batch.size(1))
        batch = batch[:, mask_out_idx:, ...]

        if train:
            pred_means, pred_logvars = model(batch)
            loss = nll_gaussian(pred_means, pred_logvars, target, pi=pi)
            mse_total += F.mse_loss(pred_means, target).detach().cpu().item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

        else:
            with torch.no_grad():
                # Compute baseline
                pre_image_mean = torch.mean(batch, dim=1)
                pre_image_var = torch.var(batch, dim=1, unbiased=False)
                
                # Add larger epsilon for numerical stability
                eps = 1e-4  # Increased from 1e-8
                pre_image_var = pre_image_var + eps
                
                # Clamp variance to avoid extreme values that could cause NaN in log
                pre_image_var = torch.clamp(pre_image_var, min=1e-4, max=1e4)
                
                # Check for any remaining numerical issues
                if torch.any(torch.isnan(pre_image_var)) or torch.any(pre_image_var <= 0):
                    print(f"Warning: Invalid variance detected. Min: {pre_image_var.min()}, Max: {pre_image_var.max()}")
                    pre_image_var = torch.ones_like(pre_image_var) * 0.1  # Fallback to reasonable default
                
                naive_nll += nll_gaussian_stable(pre_image_mean, pre_image_var, target, pi)
                naive_mse += F.mse_loss(pre_image_mean, target).detach().cpu().item()

                # Get prediction
                pred_means, pred_logvars = model(batch)
                loss = nll_gaussian(pred_means, pred_logvars, target)
                mse_total += F.mse_loss(pred_means, target).detach().cpu().item()

        nll_total += loss.detach().cpu().item()
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


def main():
    # Load configuration
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    config = load_config(config_path)
    
    # Initialize graceful shutdown handler
    killer = GracefulKiller()
    
    # Initialize wandb
    wandb_manager = WandBManager(config, enabled=config.get('use_wandb', True))
    
    # Set random seeds
    torch.manual_seed(config['train_config']['seed'])
    np.random.seed(config['train_config']['seed'])
    
    # Load data
    train_dataset = torch.load(config['data']['train_path'], weights_only=False)
    test_dataset = torch.load(config['data']['test_path'], weights_only=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['train_config']['batch_size'], 
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['train_config']['batch_size'], 
        shuffle=True
    )
    
    # Initialize model
    model = SpatioTemporalTransformer(config['model_config']).to(device)
    print(f"Number of parameters: {model.num_parameters()}")
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['train_config']['learning_rate']
    )
    scheduler = StepLR(
        optimizer, 
        step_size=config['train_config']['step_size'], 
        gamma=config['train_config']['gamma']
    )
    
    # Load checkpoint if resuming
    start_epoch = 1
    metrics_history = {
        'train_loss': [],
        'train_mse': [],
        'test_loss': [],
        'test_mse': [],
        'test_naive_nll': [],
        'test_naive_mse': []
    }
    
    if config.get('resume_checkpoint'):
        print(f"Resuming from checkpoint: {config['resume_checkpoint']}")
        start_epoch, metrics_history = load_checkpoint(
            config['resume_checkpoint'], 
            model, 
            optimizer, 
            scheduler
        )
        start_epoch += 1
    
    # Training setup
    pi = torch.FloatTensor([np.pi]).to(device)
    now = datetime.now().strftime("%m-%d-%Y_%H-%M")
    
    # Create directories
    Path(config['save_dir']['models']).mkdir(parents=True, exist_ok=True)
    Path(config['save_dir']['checkpoints']).mkdir(parents=True, exist_ok=True)
    if config.get('validation', {}).get('enable_visual_validation', False):
        Path(config['save_dir']['visualizations']).mkdir(parents=True, exist_ok=True)
    
    # Load validation data
    val_data = load_validation_data(config)
    
    try:
        # Training loop
        for epoch in range(start_epoch, config['train_config']['num_epochs'] + 1):
            if killer.kill_now:
                print("\nReceived interrupt signal. Saving checkpoint and exiting...")
                break
                
            print(f'--- EPOCH [{epoch}/{config["train_config"]["num_epochs"]}] ---')
            epoch_start_time = time.time()
            
            # Train
            train_loss, train_mse, _, _ = run_epoch_tf(
                train_loader, model, optimizer, device, pi, epoch, killer, train=True
            )
            
            # Check again after training epoch
            if killer.kill_now:
                print("\nInterrupted during training. Saving progress...")
                # Still update metrics for partial epoch
                if train_loss > 0:  # Only if we processed some batches
                    metrics_history['train_loss'].append(train_loss)
                    metrics_history['train_mse'].append(train_mse)
                break
            
            # Test
            test_loss, test_mse, test_naive_nll, test_naive_mse = run_epoch_tf(
                test_loader, model, optimizer, device, pi, epoch, killer, train=False
            )
            
            # Update metrics history
            metrics_history['train_loss'].append(train_loss)
            metrics_history['train_mse'].append(train_mse)
            metrics_history['test_loss'].append(test_loss)
            metrics_history['test_mse'].append(test_mse)
            metrics_history['test_naive_nll'].append(test_naive_nll)
            metrics_history['test_naive_mse'].append(test_naive_mse)
            
            # Log to wandb
            wandb_manager.log({
                'epoch': epoch,
                'train/nll': train_loss,
                'train/mse': train_mse,
                'test/nll': test_loss,
                'test/mse': test_mse,
                'test/naive_nll': test_naive_nll,
                'test/naive_mse': test_naive_mse,
                'learning_rate': scheduler.get_last_lr()[0],
                'epoch_time_minutes': (time.time() - epoch_start_time) / 60
            }, step=epoch)
            
            print(f"Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
            print(f"Train MSE: {train_mse:.6f}, Test MSE: {test_mse:.6f}")
            print(f"Test Naive NLL: {test_naive_nll:.6f}, Test Naive MSE: {test_naive_mse:.6f}")
            print(f"Time: {(time.time() - epoch_start_time)/60:.2f} minutes\n")
            
            # Save checkpoint
            if epoch % config['train_config']['checkpoint_freq'] == 0:
                checkpoint_path = Path(config['save_dir']['checkpoints']) / f'checkpoint_epoch_{epoch}_{now}.pth'
                save_checkpoint(
                    model, optimizer, scheduler, epoch, 
                    config, metrics_history, checkpoint_path
                )
                
                # Save model
                model_path = Path(config['save_dir']['models']) / f'{config["model_config"]["type"]}_{now}_epoch_{epoch}.pth'
                torch.save(model.state_dict(), model_path)
            
            scheduler.step()
        
        # Run visual validation at the end (before cleanup)
        if not killer.kill_now and epoch == config['train_config']['num_epochs']:
            if config.get('validation', {}).get('enable_visual_validation', False):
                print("\nRunning final visual validation...")
                validate_visual(
                    model, 
                    val_data, 
                    epoch, 
                    config['save_dir']['visualizations'],
                    wandb_manager
                )
    
    finally:
        # Save emergency checkpoint if interrupted
        if killer.kill_now:
            print("\nSaving emergency checkpoint...")
            emergency_checkpoint_path = Path(config['save_dir']['checkpoints']) / f'emergency_checkpoint_{now}.pth'
            # Use the last completed epoch number for the checkpoint
            last_epoch = len(metrics_history['train_loss']) + start_epoch - 1
            save_checkpoint(
                model, optimizer, scheduler, last_epoch, 
                config, metrics_history, emergency_checkpoint_path
            )
            print(f"Emergency checkpoint saved. Resume from epoch {last_epoch + 1}")
        
        # Clean up wandb
        wandb_manager.finish()
        
        # Save final checkpoint only if training completed normally
        if not killer.kill_now and 'epoch' in locals() and epoch == config['train_config']['num_epochs']:
            final_checkpoint_path = Path(config['save_dir']['checkpoints']) / f'final_checkpoint_{now}.pth'
            save_checkpoint(
                model, optimizer, scheduler, epoch, 
                config, metrics_history, final_checkpoint_path
            )
            
            # Save final model
            final_model_path = Path(config['save_dir']['models']) / f'{config["model_config"]["type"]}_{now}_final.pth'
            torch.save(model.state_dict(), final_model_path)
        
        print("Training completed or interrupted. All resources cleaned up.")


if __name__ == "__main__":
    main()