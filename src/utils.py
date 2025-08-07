"""
Utility functions for SAR image processing and spatiotemporal transformer model.

This module contains functions for:
- Model loading and inference
- Image processing and damage detection
- Training utilities
- Visualization
"""

import json
import math
import os
import platform
import signal
import warnings
from collections.abc import Generator
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.mps
import torch.nn.functional as F
import wandb
import yaml
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.special import logit
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm

# Optional imports
try:
    from einops._torch_specific import allow_ops_in_compiled_graph
except ImportError:
    allow_ops_in_compiled_graph = None

from src.dist_model import SpatioTemporalTransformer


# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

# Model configurations
TRANSFORMER_LATEST_CONFIG = {
    'type': 'transformer',
    'patch_size': 8,
    'num_patches': 4,
    'data_dim': 128,
    'd_model': 256,
    'nhead': 4,
    'num_encoder_layers': 4,
    'dim_feedforward': 768,
    'max_seq_len': 10,
    'dropout': 0.2,
    'activation': 'relu',
}

TRANSFORMER_CONFIG = {
    'type': 'transformer (space and time pos encoding)',
    'patch_size': 8,
    'num_patches': 4,
    'data_dim': 128,  # 2 * patch_size * patch_size
    'd_model': 256,
    'nhead': 4,
    'num_encoder_layers': 2,
    'dim_feedforward': 512,
    'max_seq_len': 10,
    'dropout': 0.2,
    'activation': 'relu',
}

# Dtype mapping
_DTYPE_MAP = {
    'float32': torch.float32,
    'float': torch.float32,
    'bfloat16': torch.bfloat16,
}
DEV_DTYPE = _DTYPE_MAP.get(os.environ.get('DEV_DTYPE', 'float32').lower(), torch.float32)

# Model paths
MODEL_DATA = Path(__file__).parent.resolve() / 'model_data'
TRANSFORMER_WEIGHTS_PATH_LATEST = MODEL_DATA / 'transformer_latest.pth'
TRANSFORMER_WEIGHTS_PATH_ORIGINAL = MODEL_DATA / 'transformer.pth'

# Global model cache
_MODEL = None


# =============================================================================
# DEVICE AND SYSTEM UTILITIES
# =============================================================================

def get_device() -> str:
    """Get the best available device for computation."""
    if torch.cuda.is_available():
        return 'cuda'
    elif platform.system() == 'Darwin' and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def control_flow_for_device(device: str | None = None) -> str:
    """Validate and return device string."""
    if device is None:
        device = get_device()
    elif isinstance(device, str):
        if device not in ['cpu', 'cuda', 'mps']:
            raise ValueError('device must be one of cpu, cuda, mps')
    return device


def setup_warnings():
    """Setup warning filters for cleaner output."""
    # Filter out TensorRT-related warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.autograd.graph")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch_tensorrt.dynamo.utils")
    warnings.filterwarnings("ignore", message=".*tensorrt::execute_engine.*")
    warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*")
    
    # Filter out torch._dynamo recompilation warnings
    warnings.filterwarnings("ignore", message=".*torch._dynamo hit config.recompile_limit.*")
    warnings.filterwarnings("ignore", message=".*To log all recompilation reasons.*")
    warnings.filterwarnings("ignore", message=".*To diagnose recompilation issues.*")


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def nll_gaussian(mean, logvar, value, mask=None, pi=None):
    """
    Compute masked negative log-likelihood loss under a Gaussian.
    mean, logvar, value: tensors of shape (...), same size
    mask: bool tensor, True where valid data exists
    """
    assert mean.shape == logvar.shape == value.shape
    
    if pi is None:
        pi = torch.FloatTensor([np.pi]).to(value.device)

    # Default mask: everything is valid
    if mask is None:
        mask = torch.ones_like(value, dtype=torch.bool)
    else:
        mask = mask.bool()  # Make sure it's boolean

    # Only compute on valid entries
    valid_mean = mean[mask]
    valid_logvar = logvar[mask]
    valid_value = value[mask]

    # Compute NLL only on valid entries
    nll_element = ((valid_value - valid_mean) ** 2) / torch.exp(valid_logvar) + valid_logvar + torch.log(2 * pi)
    loss = 0.5 * nll_element.mean()

    return loss



def nll_gaussian_stable(mean, variance, value, mask=None, pi=None, eps=1e-6):
    """
    Numerically stable negative log-likelihood of Gaussian with masking,
    avoiding any computation at invalid (masked out) positions.
    
    Args:
        mean: Mean tensor
        variance: Variance tensor (must be > 0)
        value: Target tensor
        mask: Boolean tensor, True where data is valid
        pi: Optional precomputed pi tensor
        eps: Small epsilon to stabilize log/variance
    Returns:
        Scalar loss: average NLL over valid entries
    """
    assert mean.size() == variance.size() == value.size()

    if pi is None:
        pi = torch.FloatTensor([np.pi]).to(value.device)

    if mask is not None:
        # Mask out all tensors before computation
        mask = mask.to(dtype=torch.bool)
        mean = mean[mask]
        variance = variance[mask].clamp(min=eps)
        value = value[mask]
    else:
        variance = variance.clamp(min=eps)

    logvar = torch.log(variance)
    nll_element = (value - mean).pow(2) / variance + logvar + torch.log(2 * pi)
    return loss

def spatial_smoothness_loss(logvar, weight=0.1):
    """Penalize large differences between neighboring pixels"""
    # Horizontal differences
    h_diff = torch.abs(logvar[:, :, :, 1:] - logvar[:, :, :, :-1])
    # Vertical differences  
    v_diff = torch.abs(logvar[:, :, 1:, :] - logvar[:, :, :-1, :])
    return weight * (torch.mean(h_diff) + torch.mean(v_diff))


def anisotropic_smoothness_loss(logvar, h_weight=0.1, v_weight=0.1):
    h_diff = torch.abs(logvar[:, :, :, 1:] - logvar[:, :, :, :-1])
    v_diff = torch.abs(logvar[:, :, 1:, :] - logvar[:, :, :-1, :])
    return h_weight * torch.mean(h_diff) + v_weight * torch.mean(v_diff)


# =============================================================================
# IMAGE PROCESSING UTILITIES
# =============================================================================

def _transform_pre_arrs(
    pre_arrs_vv: list[np.ndarray], 
    pre_arrs_vh: list[np.ndarray], 
    logit_transformed: bool = False
) -> np.ndarray:
    """
    Transform and stack VV and VH pre-event arrays.
    
    Args:
        pre_arrs_vv: List of VV polarization arrays
        pre_arrs_vh: List of VH polarization arrays
        logit_transformed: Whether to apply logit transformation
    
    Returns:
        Stacked array of shape (T, 2, H, W)
    """
    if len(pre_arrs_vh) != len(pre_arrs_vv):
        raise ValueError('Both vv and vh pre-arrays must have the same length')
    
    dual_pol = [np.stack([vv, vh], axis=0) for (vv, vh) in zip(pre_arrs_vv, pre_arrs_vh)]
    ts = np.stack(dual_pol, axis=0)
    
    if logit_transformed:
        ts = logit(ts)
    
    return ts


def log_ratio(pre_imgs, target, is_logit=True, method='mean'):
    """
    Compute the log ratio damage map.
    
    Args:
        pre_imgs: Pre-event images (T, C, H, W)
        target: Post-event image (C, H, W)
        is_logit: Whether inputs are in logit space
        method: Aggregation method ('mean' or 'median')
    
    Returns:
        Log ratio damage map
    """
    assert len(pre_imgs.shape) == 4
    assert len(target.shape) == 3
    assert pre_imgs.shape[1:] == target.shape
    
    # Convert from logit space to original SAR space if needed
    if is_logit:
        pre_imgs = torch.sigmoid(pre_imgs)
        target = torch.sigmoid(target)
    
    assert torch.max(pre_imgs) < 1 and torch.min(pre_imgs) > 0
    assert torch.max(target) < 1 and torch.min(target) > 0
    
    if method == 'mean':
        pred = torch.mean(pre_imgs, 0)
    elif method == 'median':
        pred = torch.median(pre_imgs, 0)[0]
    else:
        raise ValueError('Invalid method')
    
    # Add small perturbation to avoid log(0)
    return 10 * torch.abs(torch.log10(target + 1e-8) - torch.log10(pred + 1e-8))


# =============================================================================
# MODEL LOADING AND OPTIMIZATION
# =============================================================================

def _optimize_model(
    transformer: torch.nn.Module, 
    dtype: str, 
    device: str, 
    batch_size: int, 
    cuda_latest: bool = False
) -> torch.nn.Module:
    """Optimize model for inference using torch.compile or TensorRT."""
    if allow_ops_in_compiled_graph:
        allow_ops_in_compiled_graph()
    
    if device == 'cuda' and cuda_latest:
        try:
            import torch_tensorrt
            
            # Get dimensions for TensorRT
            total_pixels = transformer.num_patches * (transformer.patch_size**2)
            wh = math.isqrt(total_pixels)
            channels = transformer.data_dim // (transformer.patch_size**2)
            expected_dims = (batch_size, transformer.max_seq_len, channels, wh, wh)
            
            transformer = torch_tensorrt.compile(
                transformer,
                inputs=[
                    torch_tensorrt.Input(
                        min_shape=(1,) + expected_dims[1:],
                        opt_shape=expected_dims,
                        max_shape=expected_dims,
                        dtype=dtype,
                    )
                ],
                enabled_precisions={dtype},
                truncate_long_and_double=True,
            )
        except ImportError:
            print("torch_tensorrt not available, using standard compilation")
            transformer = torch.compile(transformer, backend='inductor')
    elif device == 'cuda':
        transformer = torch.compile(transformer, backend='inductor')
    else:
        transformer = torch.compile(transformer, mode='max-autotune-no-cudagraphs', dynamic=False)
    
    return transformer


def load_transformer_model(
    model_token: str = 'latest',
    model_cfg_path: Path | None = None,
    model_wts_path: Path | None = None,
    device: str | None = None,
    optimize: bool = False,
    batch_size: int = 32,
    dtype: str = 'float32',
) -> SpatioTemporalTransformer:
    """
    Load and optionally optimize a transformer model.
    
    Args:
        model_token: Which model to load ('latest', 'original', or 'external')
        model_cfg_path: Path to config file (for external models)
        model_wts_path: Path to weights file (for external models)
        device: Device to load model on
        optimize: Whether to optimize model for inference
        batch_size: Batch size for optimization
        dtype: Data type for model
    
    Returns:
        Loaded transformer model
    """
    global _MODEL
    
    if dtype not in _DTYPE_MAP.keys():
        raise ValueError(f'dtype must be one of {_DTYPE_MAP.keys()}')
    
    if _MODEL is not None:
        return _MODEL
    
    if model_token not in ['latest', 'original', 'external']:
        raise ValueError('model_token must be one of latest, original, or external')
    
    # Select configuration and weights
    if model_token == 'latest':
        config = TRANSFORMER_LATEST_CONFIG
        weights_path = TRANSFORMER_WEIGHTS_PATH_LATEST
    elif model_token == 'original':
        config = TRANSFORMER_CONFIG
        weights_path = TRANSFORMER_WEIGHTS_PATH_ORIGINAL
    else:
        with Path.open(model_cfg_path) as cfg:
            config = json.load(cfg)
        weights_path = model_wts_path
    
    # Load model
    device = control_flow_for_device(device)
    weights = torch.load(weights_path, map_location=device, weights_only=True)
    transformer = SpatioTemporalTransformer(config).to(device)
    transformer.load_state_dict(weights)
    transformer = transformer.eval()
    
    # Optimize if requested
    if optimize:
        transformer = _optimize_model(transformer, dtype=dtype, device=device, batch_size=batch_size)
    
    _MODEL = transformer
    return transformer


# =============================================================================
# INFERENCE UTILITIES
# =============================================================================

def unfolding_stream(
    image_st: torch.Tensor, 
    kernel_size: int, 
    stride: int, 
    batch_size: int
) -> Generator[torch.Tensor, None, None]:
    """
    Generate patches from an image using sliding window with batching.
    
    Yields:
        Tuple of (batch of patches, slice indices)
    """
    _, _, H, W = image_st.shape
    
    patches = []
    slices = []
    
    n_patches_y = int(np.floor((H - kernel_size) / stride) + 1)
    n_patches_x = int(np.floor((W - kernel_size) / stride) + 1)
    
    for i in range(n_patches_y):
        for j in range(n_patches_x):
            # Handle boundary cases
            if i == (n_patches_y - 1):
                s_y = slice(H - kernel_size, H)
            else:
                s_y = slice(i * stride, i * stride + kernel_size)
            
            if j == (n_patches_x - 1):
                s_x = slice(W - kernel_size, W)
            else:
                s_x = slice(j * stride, j * stride + kernel_size)
            
            patch = image_st[..., s_y, s_x]
            patches.append(patch)
            slices.append((s_y, s_x))
            
            # Yield patches in batches
            if len(patches) == batch_size:
                yield torch.stack(patches, dim=0), slices
                patches = []
                slices = []
    
    # Yield remaining patches
    if patches:
        yield torch.stack(patches, dim=0), slices


@torch.inference_mode()
def _estimate_logit_params_via_streamed_patches(
    model: torch.nn.Module,
    imgs_copol: list[np.ndarray],
    imgs_crosspol: list[np.ndarray],
    patch_size: int = 16,
    stride: int = 2,
    batch_size: int = 32,
    max_nodata_ratio: float = 0.1,
    tqdm_enabled: bool = True,
    device: str | None = None,
) -> tuple[np.ndarray]:
    """
    Estimate mean and sigma using low-memory streaming strategy.
    
    This method processes patches sequentially, requiring less GPU memory
    but more time due to data transfers.
    
    Args:
        model: Trained transformer model
        imgs_copol: List of co-polarization images
        imgs_crosspol: List of cross-polarization images
        patch_size: Size of patches to extract (default: 16)
        stride: Stride for sliding window
        batch_size: Batch size for processing
        max_nodata_ratio: Maximum ratio of no-data pixels allowed in a patch
        tqdm_enabled: Whether to show progress bar
        device: Device to run on
    
    Returns:
        Tuple of (mean, sigma) arrays
    """
    assert stride <= patch_size and stride > 0
    
    device = control_flow_for_device(device)
    
    # Stack and prepare data
    pre_imgs_stack = _transform_pre_arrs(imgs_copol, imgs_crosspol)
    pre_imgs_stack = pre_imgs_stack.astype('float32')
    
    # Create mask
    mask_stack = np.isnan(pre_imgs_stack)
    mask_spatial = torch.from_numpy(np.any(mask_stack, axis=(0, 1))).to(device, dtype=DEV_DTYPE)
    
    # Apply logit transformation
    pre_imgs_stack[mask_stack] = 1e-7
    pre_imgs_logit = logit(pre_imgs_stack)
    pre_imgs_stack_t = torch.from_numpy(pre_imgs_logit).to(device, dtype=DEV_DTYPE)
    
    # Get dimensions
    C, H, W = pre_imgs_logit.shape[-3:]
    n_patches_y = int(np.floor((H - patch_size) / stride) + 1)
    n_patches_x = int(np.floor((W - patch_size) / stride) + 1)
    n_patches = n_patches_y * n_patches_x
    n_batches = math.ceil(n_patches / batch_size)
    
    # Initialize accumulators
    target_shape = (C, H, W)
    count = torch.zeros(*target_shape).to(device, dtype=DEV_DTYPE)
    pred_means = torch.zeros(*target_shape).to(device, dtype=DEV_DTYPE)
    pred_logvars = torch.zeros(*target_shape).to(device, dtype=DEV_DTYPE)
    
    # Process patches
    unfold_gen = unfolding_stream(pre_imgs_stack_t, patch_size, stride, batch_size)
    
    for patch_batch, slices in tqdm(
        unfold_gen,
        total=n_batches,
        desc='Processing patches',
        mininterval=2,
        disable=(not tqdm_enabled),
        dynamic_ncols=True,
    ):
        patch_batch = patch_batch.to(device, dtype=DEV_DTYPE)
        chip_mean, chip_logvar = model(patch_batch)
        
        for k, (sy, sx) in enumerate(slices):
            chip_mask = mask_spatial[sy, sx]
            if (chip_mask).sum().item() / chip_mask.nelement() <= max_nodata_ratio:
                pred_means[:, sy, sx] += chip_mean[k, ...]
                pred_logvars[:, sy, sx] += chip_logvar[k, ...]
                count[:, sy, sx] += 1
    
    # Average predictions
    pred_means /= count
    pred_logvars /= count
    
    # Apply mask
    M_3d = mask_spatial.unsqueeze(dim=0).expand(pred_means.shape)
    pred_means[M_3d] = torch.nan
    pred_logvars[M_3d] = torch.nan
    
    # Convert to numpy
    pred_means = pred_means.cpu().numpy().squeeze()
    pred_logvars = pred_logvars.cpu().numpy().squeeze()
    pred_sigmas = np.sqrt(np.exp(pred_logvars))
    
    return pred_means, pred_sigmas


@torch.inference_mode()
def _estimate_logit_params_via_folding(
    model: torch.nn.Module,
    imgs_copol: list[np.ndarray],
    imgs_crosspol: list[np.ndarray],
    patch_size: int = 16,
    stride: int = 2,
    batch_size: int = 32,
    device: str | None = None,
    tqdm_enabled: bool = True,
) -> tuple[np.ndarray]:
    """
    Estimate mean and sigma using high-memory folding strategy.
    
    This method uses F.unfold/fold which stores pixels redundantly
    but is very fast on GPU.
    
    Args:
        model: Trained transformer model
        imgs_copol: List of co-polarization images
        imgs_crosspol: List of cross-polarization images
        patch_size: Size of patches to extract (default: 16)
        stride: Stride for sliding window
        batch_size: Batch size for processing
        device: Device to run on
        tqdm_enabled: Whether to show progress bar
    
    Returns:
        Tuple of (mean, sigma) arrays
    """
    assert stride <= patch_size and stride > 0
    
    device = control_flow_for_device(device)
    
    # Stack and prepare data
    pre_imgs_stack = _transform_pre_arrs(imgs_copol, imgs_crosspol)
    pre_imgs_stack = pre_imgs_stack.astype('float32')
    
    # Create mask
    mask_stack = np.isnan(pre_imgs_stack)
    mask_spatial = torch.from_numpy(np.any(mask_stack, axis=(0, 1)))
    
    # Apply logit transformation
    pre_imgs_stack[mask_stack] = 1e-7
    pre_imgs_logit = logit(pre_imgs_stack)
    
    # Get dimensions
    H, W = pre_imgs_logit.shape[-2:]
    T = pre_imgs_logit.shape[0]
    C = pre_imgs_logit.shape[1]
    
    # Calculate number of patches
    n_patches_y = int(np.floor((H - patch_size) / stride) + 1)
    n_patches_x = int(np.floor((W - patch_size) / stride) + 1)
    n_patches = n_patches_y * n_patches_x
    
    # Convert to tensor and unfold
    pre_imgs_stack_t = torch.from_numpy(pre_imgs_logit).to(device, dtype=DEV_DTYPE)
    patches = F.unfold(pre_imgs_stack_t, kernel_size=patch_size, stride=stride)
    patches = patches.permute(2, 0, 1).to(device, dtype=DEV_DTYPE)
    patches = patches.view(n_patches, T, C, patch_size**2)
    
    # Process in batches
    n_batches = math.ceil(n_patches / batch_size)
    target_chip_shape = (n_patches, C, patch_size, patch_size)
    pred_means_p = torch.zeros(*target_chip_shape).to(device, dtype=DEV_DTYPE)
    pred_logvars_p = torch.zeros_like(pred_means_p).to(device, dtype=DEV_DTYPE)
    
    for i in tqdm(
        range(n_batches),
        desc='Processing batches',
        mininterval=2,
        disable=(not tqdm_enabled),
        dynamic_ncols=True,
    ):
        batch_s = slice(batch_size * i, batch_size * (i + 1))
        patch_batch = patches[batch_s, ...].view(-1, T, C, patch_size, patch_size)
        chip_mean, chip_logvar = model(patch_batch)
        pred_means_p[batch_s, ...] += chip_mean
        pred_logvars_p[batch_s, ...] += chip_logvar
    
    # Clean up memory
    del patches
    torch.cuda.empty_cache()
    
    # Fold predictions back
    pred_logvars_p_reshaped = pred_logvars_p.view(n_patches, C * patch_size**2).permute(1, 0)
    pred_logvars = F.fold(pred_logvars_p_reshaped, output_size=(H, W), kernel_size=patch_size, stride=stride)
    del pred_logvars_p
    
    pred_means_p_reshaped = pred_means_p.view(n_patches, C * patch_size**2).permute(1, 0)
    pred_means = F.fold(pred_means_p_reshaped, output_size=(H, W), kernel_size=patch_size, stride=stride)
    del pred_means_p_reshaped
    
    # Count overlapping patches
    input_ones = torch.ones(1, H, W).to(device, dtype=DEV_DTYPE)
    count_patches = F.unfold(input_ones, kernel_size=patch_size, stride=stride)
    count = F.fold(count_patches, output_size=(H, W), kernel_size=patch_size, stride=stride)
    del count_patches
    torch.cuda.empty_cache()
    
    # Average predictions
    pred_means /= count
    pred_logvars /= count
    
    # Apply mask
    mask_3d = mask_spatial.unsqueeze(dim=0).expand(pred_means.shape)
    pred_means[mask_3d] = torch.nan
    pred_logvars[mask_3d] = torch.nan
    
    # Convert to numpy
    pred_means = pred_means.cpu().numpy().squeeze()
    pred_logvars = pred_logvars.cpu().numpy().squeeze()
    pred_sigmas = np.sqrt(np.exp(pred_logvars))
    
    return pred_means, pred_sigmas


def estimate_normal_params_of_logits(
    model: torch.nn.Module,
    imgs_copol: list[np.ndarray],
    imgs_crosspol: list[np.ndarray],
    patch_size: int = 16,
    stride: int = 2,
    batch_size: int = 32,
    tqdm_enabled: bool = True,
    memory_strategy: str = 'high',
    device: str | None = None,
) -> tuple[np.ndarray]:
    """
    Estimate normal distribution parameters of logit-transformed images.
    
    Args:
        model: Trained transformer model
        imgs_copol: List of co-polarization images
        imgs_crosspol: List of cross-polarization images
        patch_size: Size of patches to extract (default: 16)
        stride: Stride for sliding window
        batch_size: Batch size for processing
        tqdm_enabled: Whether to show progress bar
        memory_strategy: 'high' for fast GPU processing, 'low' for memory-efficient
        device: Device to run on
    
    Returns:
        Tuple of (mean, sigma) arrays
    """
    if memory_strategy not in ['high', 'low']:
        raise ValueError('memory_strategy must be high or low')
    
    estimate_func = (
        _estimate_logit_params_via_folding 
        if memory_strategy == 'high' 
        else _estimate_logit_params_via_streamed_patches
    )
    
    return estimate_func(
        model, imgs_copol, imgs_crosspol, 
        patch_size=patch_size, stride=stride, batch_size=batch_size, 
        tqdm_enabled=tqdm_enabled, device=device
    )


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_reconstruction(
    pre_imgs, post_img, pred_mean, pred_logvar, 
    model_type, is_logit=True, 
    vmin_vv=None, vmax_vv=None, vmin_vh=None, vmax_vh=None
):
    """
    Create visualization comparing predictions to ground truth.
    
    Args:
        pre_imgs: Pre-event images (T, C, H, W)
        post_img: Post-event image (C, H, W)
        pred_mean: Predicted mean (C, H, W)
        pred_logvar: Predicted log variance (C, H, W)
        model_type: Type of model ('Transformer' or 'RNN')
        is_logit: Whether inputs are in logit space
        vmin_vv, vmax_vv: Min/max values for VV visualization
        vmin_vh, vmax_vh: Min/max values for VH visualization
    
    Returns:
        Tuple of (damage_map, log_ratio_map)
    """
    assert post_img.shape == pred_mean.shape == pred_logvar.shape
    assert len(pre_imgs.shape) == 4
    assert len(post_img.shape) == 3
    
    if model_type not in ['Transformer', 'RNN']:
        raise ValueError('Not a valid model type')
    
    # Ensure everything is on CPU for plotting
    pre_imgs = pre_imgs.cpu()
    post_img = post_img.cpu()
    pred_mean = pred_mean.cpu()
    pred_logvar = pred_logvar.cpu()
    
    pred_std_image = torch.sqrt(torch.exp(pred_logvar))
    
    # Compute metrics
    pred_vs_true_mse = F.mse_loss(pred_mean, post_img)
    pred_vs_true_nll = nll_gaussian(pred_mean, pred_logvar, post_img)
    
    mean_image = torch.mean(pre_imgs, 0)
    std_image = torch.std(pre_imgs, 0)
    
    naive_vs_true_mse = F.mse_loss(mean_image, post_img)
    naive_vs_true_nll = nll_gaussian_stable(mean_image, torch.var(pre_imgs, 0) + 1e-8, post_img)
    
    # Compute damage maps
    damage_map = torch.absolute((post_img - pred_mean) / pred_std_image)
    log_ratio_im = log_ratio(pre_imgs, post_img, is_logit=is_logit)
    
    # Convert from logit space for visualization
    if is_logit:
        pre_imgs = torch.sigmoid(pre_imgs)
        post_img = torch.sigmoid(post_img)
        pred_mean = torch.sigmoid(pred_mean)
        mean_image = torch.sigmoid(mean_image)
    
    # Create figure
    fig, axs = plt.subplots(7, 2, figsize=(10, 30))
    
    # Set color limits
    if vmin_vv is None:
        vmin_vv = torch.min(post_img[0,...])
    if vmin_vh is None:
        vmin_vh = torch.min(post_img[1,...])
    if vmax_vv is None:
        vmax_vv = torch.max(post_img[0,...])
    if vmax_vh is None:
        vmax_vh = torch.max(post_img[1,...])
    
    # Row 0: Model predictions
    pred_mean_vv = axs[0,0].imshow(pred_mean[0, ...], vmin=vmin_vv, vmax=vmax_vv, interpolation='None')
    axs[0,0].set_title(f'{model_type} Pred Mean VV, MSE={pred_vs_true_mse:.4f}')
    axs[0,0].axis('off')
    plt.colorbar(pred_mean_vv, ax=axs[0,0])
    
    pred_mean_vh = axs[0,1].imshow(pred_mean[1, ...], vmin=vmin_vh, vmax=vmax_vh, interpolation='None')
    axs[0,1].set_title(f'{model_type} Pred Mean VH, NLL={pred_vs_true_nll:.4f}')
    axs[0,1].axis('off')
    plt.colorbar(pred_mean_vh, ax=axs[0,1])
    
    # Row 1: Naive average
    mean_vv = axs[1,0].imshow(mean_image[0, ...], vmin=vmin_vv, vmax=vmax_vv, interpolation='None')
    axs[1,0].set_title(f'Avg Pre Img VV, MSE={naive_vs_true_mse:.4f}')
    axs[1,0].axis('off')
    plt.colorbar(mean_vv, ax=axs[1,0])
    
    mean_vh = axs[1,1].imshow(mean_image[1, ...], vmin=vmin_vh, vmax=vmax_vh, interpolation='None')
    axs[1,1].set_title(f'Avg Pre Img VH, NLL={naive_vs_true_nll:.4f}')
    axs[1,1].axis('off')
    plt.colorbar(mean_vh, ax=axs[1,1])
    
    # Row 2: Ground truth
    post_img_vv = axs[2,0].imshow(post_img[0, ...], vmin=vmin_vv, vmax=vmax_vv, interpolation='None')
    axs[2,0].set_title('Ground Truth VV')
    axs[2,0].axis('off')
    plt.colorbar(post_img_vv, ax=axs[2,0])
    
    post_img_vh = axs[2,1].imshow(post_img[1, ...], vmin=vmin_vh, vmax=vmax_vh, interpolation='None')
    axs[2,1].set_title('Ground Truth VH')
    axs[2,1].axis('off')
    plt.colorbar(post_img_vh, ax=axs[2,1])
    
    # Row 3: Model predicted std
    std_vmin_vv = torch.min(pred_std_image[0,...])
    std_vmax_vv = torch.max(pred_std_image[0,...])
    std_vmin_vh = torch.min(pred_std_image[1,...])
    std_vmax_vh = torch.max(pred_std_image[1,...])
    
    pred_std_vv = axs[3,0].imshow(pred_std_image[0, ...], vmin=std_vmin_vv, vmax=std_vmax_vv, interpolation='None')
    axs[3,0].set_title(f'{model_type} Pred Std VV')
    axs[3,0].axis('off')
    plt.colorbar(pred_std_vv, ax=axs[3,0])
    
    pred_std_vh = axs[3,1].imshow(pred_std_image[1, ...], vmin=std_vmin_vh, vmax=std_vmax_vh, interpolation='None')
    axs[3,1].set_title(f'{model_type} Pred Std VH')
    axs[3,1].axis('off')
    plt.colorbar(pred_std_vh, ax=axs[3,1])
    
    # Row 4: Numerical std
    std_vv = axs[4,0].imshow(std_image[0, ...], vmin=std_vmin_vv, vmax=std_vmax_vv, interpolation='None')
    axs[4,0].set_title('Numerical Pre Img Std VV')
    axs[4,0].axis('off')
    plt.colorbar(std_vv, ax=axs[4,0])
    
    std_vh = axs[4,1].imshow(std_image[1, ...], vmin=std_vmin_vh, vmax=std_vmax_vh, interpolation='None')
    axs[4,1].set_title('Numerical Pre Img Std VH')
    axs[4,1].axis('off')
    plt.colorbar(std_vh, ax=axs[4,1])
    
    # Row 5: Z-score damage map
    damage_vmax = torch.max(damage_map) * .75
    
    dam_vv = axs[5,0].imshow(damage_map[0, ...], cmap='plasma', interpolation='None', vmax=damage_vmax)
    axs[5,0].set_title(f'{model_type} Z-score Damage Map VV')
    axs[5,0].axis('off')
    plt.colorbar(dam_vv, ax=axs[5,0])
    
    dam_vh = axs[5,1].imshow(damage_map[1, ...], cmap='plasma', interpolation='None', vmax=damage_vmax)
    axs[5,1].set_title(f'{model_type} Z-score Damage Map VH')
    axs[5,1].axis('off')
    plt.colorbar(dam_vh, ax=axs[5,1])
    
    # Row 6: Log ratio damage map
    lr_vmax = torch.max(log_ratio_im) * .75
    
    lr_vv = axs[6,0].imshow(log_ratio_im[0, ...], cmap='plasma', interpolation='None', vmax=lr_vmax)
    axs[6,0].set_title('Log Ratio Damage Map VV')
    axs[6,0].axis('off')
    plt.colorbar(lr_vv, ax=axs[6,0])
    
    lr_vh = axs[6,1].imshow(log_ratio_im[1, ...], cmap='plasma', interpolation='None', vmax=lr_vmax)
    axs[6,1].set_title('Log Ratio Damage Map VH')
    axs[6,1].axis('off')
    plt.colorbar(lr_vh, ax=axs[6,1])
    
    plt.subplots_adjust(hspace=.3)
    
    return damage_map, log_ratio_im


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class WandBManager:
    """Manages WandB logging with multi-GPU safety."""
    
    def __init__(self, config, accelerator, enabled=True):
        self.enabled = enabled
        self.run = None
        self.accelerator = accelerator
        
        # Only initialize wandb on the main process
        if self.enabled and accelerator.is_main_process:
            self.run = wandb.init(
                entity=config.get('wandb_entity', None),
                project=config.get('wandb_project', 'spatiotemporal-transformer'),
                name=config.get('wandb_run_name', None),
                config=config,
                resume=config.get('resume_wandb_run_id', None),
                settings=wandb.Settings(
                    start_method="thread",
                    _disable_stats=False,
                    _disable_meta=True,
                    save_code=False,
                    anonymous="never"
                )
            )
    
    def log(self, metrics, step=None):
        """Log metrics to WandB (only from main process)."""
        if self.enabled and self.run and self.accelerator.is_main_process:
            wandb.log(metrics, step=step)
    
    def finish(self):
        """Finish WandB run."""
        if self.enabled and self.run and self.accelerator.is_main_process:
            wandb.finish()


class GracefulKiller:
    """Handles graceful shutdown on interrupt signals."""
    
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        print('\nReceived signal to terminate. Finishing current epoch...')
        self.kill_now = True


def load_config(config_path):
    """Load and validate configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Train config
    train_cfg = config['train_config']
    train_cfg['learning_rate'] = float(train_cfg['learning_rate'])
    train_cfg['batch_size'] = int(train_cfg['batch_size'])
    train_cfg['num_epochs'] = int(train_cfg['num_epochs'])
    train_cfg['seed'] = int(train_cfg['seed'])
    train_cfg['step_size'] = int(train_cfg['step_size'])
    train_cfg['gamma'] = float(train_cfg['gamma'])
    train_cfg['checkpoint_freq'] = int(train_cfg['checkpoint_freq'])
    
    # Model config
    model_cfg = config['model_config']
    model_cfg['patch_size'] = int(model_cfg['patch_size'])
    model_cfg['num_patches'] = int(model_cfg.get('num_patches', 0))
    model_cfg['data_dim'] = int(model_cfg['data_dim'])
    model_cfg['d_model'] = int(model_cfg['d_model'])
    model_cfg['nhead'] = int(model_cfg['nhead'])
    model_cfg['num_encoder_layers'] = int(model_cfg['num_encoder_layers'])
    model_cfg['dim_feedforward'] = int(model_cfg['dim_feedforward'])
    model_cfg['max_seq_len'] = int(model_cfg['max_seq_len'])
    model_cfg['dropout'] = float(model_cfg['dropout'])
    
    return config


def save_checkpoint(model, optimizer, scheduler, epoch, config, metrics, checkpoint_path, accelerator):
    """Save training checkpoint (only from main process)."""
    if accelerator.is_main_process:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': accelerator.get_state_dict(model),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': config,
            'metrics': metrics
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, accelerator):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=accelerator.device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['metrics']


def save_emergency_state(model, optimizer, scheduler, epoch, config, metrics_history, accelerator, reason="interruption"):
    """Save emergency checkpoint on interruption."""
    if accelerator.is_main_process:
        print(f"\nSaving emergency state due to {reason}...")
        now = datetime.now().strftime("%m-%d-%Y_%H-%M")
        emergency_checkpoint_path = Path(config['save_dir']['checkpoints']) / f'emergency_checkpoint_{now}.pth'
        save_checkpoint(
            model, optimizer, scheduler, epoch, 
            config, metrics_history, emergency_checkpoint_path, accelerator
        )
        print(f"Emergency checkpoint saved. Resume from epoch {epoch + 1}")
    return epoch


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def load_validation_data(config):
    """Load validation datasets for visual evaluation."""
    val_data = {}
    
    if not config.get('validation', {}).get('enable_visual_validation', False):
        return val_data
    
    val_config = config['validation']
    
    # Landslide dataset
    if val_config.get('landslide', {}).get('enabled', False):
        target = Image.open(val_config['landslide']['mask_path'])
        target = ToTensor()(target)
        target[target < -1] = -1
        target_landslide = target[0, 1248:1440, 448:640]
        
        landslide = torch.load(val_config['landslide']['data_path'])
        landslide = landslide[9:20, :, :, :]
        landslide_logit = torch.special.logit(landslide).float()
        
        val_data['landslide'] = {
            'pre': landslide_logit[:-1, :, :, :].unsqueeze(dim=0),
            'post': landslide_logit[-1, :, :, :].unsqueeze(dim=0),
            'target': target_landslide,
            'stride': val_config['landslide'].get('stride', 1)
        }
    
    # Fire dataset
    if val_config.get('fire', {}).get('enabled', False):
        row_min = 1500
        row_max = row_min + 224 * 7
        col_min = 1000
        col_max = col_min + 224 * 5
        
        target_fire = torch.load(val_config['fire']['mask_path'], weights_only=True)
        target_fire = target_fire[row_min:row_max, col_min:col_max]
        
        fire_data = torch.load(val_config['fire']['data_path'], weights_only=True)
        fire_data = fire_data[:, :, row_min:row_max, col_min:col_max]
        fire_data_logit = torch.special.logit(fire_data).float()
        
        val_data['fire'] = {
            'pre': fire_data_logit[:7, :, :, :].unsqueeze(dim=0),
            'post': fire_data_logit[-1, :, :, :].unsqueeze(dim=0),
            'target': target_fire,
            'stride': val_config['fire'].get('stride', 4)
        }
    
    # Flood dataset
    if val_config.get('flood', {}).get('enabled', False):
        target_bangladesh = torch.load(val_config['flood']['mask_path'])
        target_bangladesh = target_bangladesh[1248:, 1248:]
        
        bangladesh_data = torch.load(val_config['flood']['data_path'])
        bangladesh_data = bangladesh_data[:, :, 1248:, 1248:]
        bangladesh_logit = torch.special.logit(bangladesh_data).float()
        
        val_data['flood'] = {
            'pre': bangladesh_logit[:4, :, :, :].unsqueeze(dim=0),
            'post': bangladesh_logit[-1, :, :, :].unsqueeze(dim=0),
            'target': target_bangladesh,
            'stride': val_config['flood'].get('stride', 4)
        }
    
    return val_data


def make_preds_sliding(model, pre_imgs, chip_size=16, stride=1, flatten=False, accelerator=None, blend_mode='gaussian'):
    """
    Generate predictions using sliding window with improved blending.
    
    Args:
        model: Trained model
        pre_imgs: Pre-event images
        chip_size: Size of sliding window (default: 16)
        stride: Stride for sliding window
        flatten: Whether to flatten patches
        accelerator: Accelerator for distributed training
        blend_mode: Blending mode ('gaussian', 'cosine', 'triangular', 'uniform')
    
    Returns:
        Tuple of (predicted means, predicted log variances)
    """
    assert stride <= chip_size and stride > 0
    
    data_dim_row = pre_imgs.shape[-2]
    data_dim_col = pre_imgs.shape[-1]
    
    assert data_dim_row % chip_size == 0
    assert data_dim_col % chip_size == 0
    
    pred_means = torch.zeros((1, 2, data_dim_row, data_dim_col), device=pre_imgs.device)
    pred_logvars = torch.zeros_like(pred_means)
    
    # Create weight matrix based on blend mode
    if blend_mode == 'gaussian':
        # Gaussian weights - smooth falloff from center
        weight_1d = torch.exp(-0.5 * ((torch.arange(chip_size, dtype=torch.float32) - chip_size//2) / (chip_size//6))**2)
        weight_2d = weight_1d.unsqueeze(0) * weight_1d.unsqueeze(1)
        weight_2d = weight_2d / weight_2d.max()
        weights = weight_2d.to(pre_imgs.device)
    elif blend_mode == 'cosine':
        # Cosine weights
        x = torch.linspace(0, torch.pi, chip_size)
        weight_1d = (torch.cos(x) + 1) / 2
        weight_2d = weight_1d.unsqueeze(0) * weight_1d.unsqueeze(1)
        weights = weight_2d.to(pre_imgs.device)
    elif blend_mode == 'triangular':
        # Triangular weights
        center = chip_size // 2
        weight_1d = 1 - torch.abs(torch.arange(chip_size, dtype=torch.float32) - center) / center
        weight_2d = weight_1d.unsqueeze(0) * weight_1d.unsqueeze(1)
        weights = weight_2d.to(pre_imgs.device)
    else:  # uniform
        weights = torch.ones((chip_size, chip_size), device=pre_imgs.device)
    
    # Weight accumulator
    weight_sum = torch.zeros((1, 2, data_dim_row, data_dim_col), device=pre_imgs.device)
    
    index_range_row = int(data_dim_row - chip_size + 1)
    index_range_col = int(data_dim_col - chip_size + 1)
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, index_range_row, stride), desc="Rows", leave=False):
            for j in range(0, index_range_col, stride):
                if flatten:
                    chip_mean, chip_logvar = model(
                        pre_imgs[:, :, :, i:(i+chip_size), j:(j+chip_size)].flatten(start_dim=2)
                    )
                else:
                    chip_mean, chip_logvar = model(
                        pre_imgs[:, :, :, i:(i+chip_size), j:(j+chip_size)]
                    )
                
                # Reshape predictions
                chip_mean_2d = chip_mean.reshape((1, 2, chip_size, chip_size))
                chip_logvar_2d = chip_logvar.reshape((1, 2, chip_size, chip_size))
                
                # Apply weighted blending
                weight_mask = weights.unsqueeze(0).unsqueeze(0)
                
                pred_means[:, :, i:(i+chip_size), j:(j+chip_size)] += chip_mean_2d * weight_mask
                pred_logvars[:, :, i:(i+chip_size), j:(j+chip_size)] += chip_logvar_2d * weight_mask
                weight_sum[:, :, i:(i+chip_size), j:(j+chip_size)] += weight_mask
    
    # Normalize by accumulated weights
    pred_means = pred_means / (weight_sum + 1e-8)
    pred_logvars = pred_logvars / (weight_sum + 1e-8)
    
    return pred_means, pred_logvars


def validate_visual(model, val_data, epoch, save_dir, wandb_manager=None, accelerator=None, 
                   apply_smoothing=True, smooth_sigma=0.5, blend_mode='gaussian', chip_size=16):
    """
    Perform visual validation on test cases.
    
    Only runs on main process to avoid conflicts.
    
    Args:
        model: Trained model
        val_data: Validation datasets
        epoch: Current epoch number
        save_dir: Directory to save visualizations
        wandb_manager: WandB manager for logging
        accelerator: Accelerator for distributed training
        apply_smoothing: Whether to apply post-processing smoothing
        smooth_sigma: Sigma for Gaussian smoothing
        blend_mode: Blending mode for sliding window
        chip_size: Size of sliding window patches (default: 16)
    """
    if not val_data or not accelerator.is_main_process:
        return
    
    for event_name, data in val_data.items():
        print(f"\nValidating {event_name}...")
        print(f"Generating comparison: uniform baseline vs {blend_mode} blend with smoothing")
        
        # Move data to device
        data_pre = data['pre'].to(accelerator.device)
        
        # Generate baseline predictions
        print("Generating baseline (uniform blend, no smoothing)...")
        pred_means_baseline, pred_logvars_baseline = make_preds_sliding(
            model, 
            data_pre[:, :-1, ...],
            chip_size=chip_size, 
            flatten=False, 
            stride=data['stride'],
            accelerator=accelerator,
            blend_mode='uniform'
        )
        
        # Generate improved predictions
        print(f"Generating improved ({blend_mode} blend)...")
        pred_means_improved, pred_logvars_improved = make_preds_sliding(
            model, 
            data_pre[:, :-1, ...],
            chip_size=chip_size, 
            flatten=False, 
            stride=data['stride'],
            accelerator=accelerator,
            blend_mode=blend_mode
        )
        
        # Apply post-processing smoothing if requested
        if apply_smoothing:
            print(f"Applying post-processing smoothing: sigma={smooth_sigma}")
            
            pred_means_cpu = pred_means_improved.cpu().numpy()
            pred_logvars_cpu = pred_logvars_improved.cpu().numpy()
            
            pred_means_smooth = np.zeros_like(pred_means_cpu)
            pred_logvars_smooth = np.zeros_like(pred_logvars_cpu)
            
            for batch in range(pred_means_cpu.shape[0]):
                for channel in range(pred_means_cpu.shape[1]):
                    pred_means_smooth[batch, channel] = gaussian_filter(
                        pred_means_cpu[batch, channel], 
                        sigma=smooth_sigma, 
                        mode='reflect'
                    )
                    pred_logvars_smooth[batch, channel] = gaussian_filter(
                        pred_logvars_cpu[batch, channel], 
                        sigma=smooth_sigma, 
                        mode='reflect'
                    )
            
            pred_means_final = torch.tensor(pred_means_smooth)
            pred_logvars_final = torch.tensor(pred_logvars_smooth)
        else:
            pred_means_final = pred_means_improved.cpu()
            pred_logvars_final = pred_logvars_improved.cpu()
        
        # Generate visualizations
        print("Generating baseline visualization...")
        plt.figure(figsize=(10, 30))
        damage_map_baseline, log_ratio_baseline = visualize_reconstruction(
            data['pre'][0,...], 
            data['post'][0,...], 
            pred_means_baseline[0,...].cpu(), 
            pred_logvars_baseline[0,...].cpu(), 
            'Transformer',
            is_logit=True, 
            vmax_vv=.3, 
            vmax_vh=.1
        )
        
        baseline_path = Path(save_dir) / f'{event_name}_epoch_{epoch}_baseline.png'
        plt.suptitle(f'{event_name} - Baseline (Uniform Blend)', fontsize=16, fontweight='bold', y=0.98)
        plt.savefig(baseline_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Generating improved visualization...")
        improved_title = f'{blend_mode.capitalize()}'
        if apply_smoothing:
            improved_title += f' + Smooth({smooth_sigma})'
        
        plt.figure(figsize=(10, 30))
        damage_map_improved, log_ratio_improved = visualize_reconstruction(
            data['pre'][0,...], 
            data['post'][0,...], 
            pred_means_final[0,...], 
            pred_logvars_final[0,...], 
            'Transformer',
            is_logit=True, 
            vmax_vv=.3, 
            vmax_vh=.1
        )
        
        improved_path = Path(save_dir) / f'{event_name}_epoch_{epoch}_improved.png'
        plt.suptitle(f'{event_name} - {improved_title}', fontsize=16, fontweight='bold', y=0.98)
        plt.savefig(improved_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Log to wandb
        if wandb_manager:
            wandb_manager.log({
                f'validation/{event_name}_baseline': wandb.Image(str(baseline_path)),
                f'validation/{event_name}_improved': wandb.Image(str(improved_path))
            }, step=epoch)
        
        print(f"Saved baseline: {baseline_path}")
        print(f"Saved improved: {improved_path}")
    
    model.train()


def run_final_validation(model, val_data, epoch, config, wandb_manager, accelerator):
    """Run visual validation at end of training."""
    if config.get('validation', {}).get('enable_visual_validation', False) and val_data:
        if accelerator.is_main_process:
            print("\nRunning final visual validation...")
        
        apply_smoothing = config.get('validation', {}).get('apply_smoothing', True)
        smooth_sigma = config.get('validation', {}).get('smooth_sigma', 0.5)
        blend_mode = config.get('validation', {}).get('blend_mode', 'gaussian')
        chip_size = config.get('train_config', {}).get('input_size', 16)
        
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
            chip_size=chip_size
        )
        
        if accelerator.is_main_process:
            print("Visual validation completed!")
    elif accelerator.is_main_process and config.get('validation', {}).get('enable_visual_validation', False):
        print("\nSkipping visual validation - no validation data loaded")