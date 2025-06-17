import json
import math
import os
import platform
from collections.abc import Generator
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.mps
import torch.nn.functional as F
from einops._torch_specific import allow_ops_in_compiled_graph
from scipy.special import logit
from tqdm.auto import tqdm

from src.dist_model import SpatioTemporalTransformer


transformer_latest_config = {
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

transformer_config = {
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


def _transform_pre_arrs(
    pre_arrs_vv: list[np.ndarray], pre_arrs_vh: list[np.ndarray], logit_transformed: bool = False
) -> np.ndarray:
    if len(pre_arrs_vh) != len(pre_arrs_vv):
        raise ValueError('Both vv and vh pre-arrays must have the same length')
    dual_pol = [np.stack([vv, vh], axis=0) for (vv, vh) in zip(pre_arrs_vv, pre_arrs_vh)]
    ts = np.stack(dual_pol, axis=0)
    if logit_transformed:
        ts = logit(ts)
    return ts


# Dtype selection
_DTYPE_MAP = {
    'float32': torch.float32,
    'float': torch.float32,
    'bfloat16': torch.bfloat16,
}
DEV_DTYPE = _DTYPE_MAP.get(os.environ.get('DEV_DTYPE', 'float32').lower(), torch.float32)

MODEL_DATA = Path(__file__).parent.resolve() / 'model_data'
TRANSFORMER_WEIGHTS_PATH_LATEST = MODEL_DATA / 'transformer_latest.pth'
TRANSFORMER_WEIGHTS_PATH_ORIGINAL = MODEL_DATA / 'transformer.pth'

_MODEL = None


def nll_gaussian(mean, logvar, value, pi=None):
    """Compute negative log-likelihood of Gaussian."""
    assert mean.size() == logvar.size() == value.size()

    # this is just for efficiency reasons - the below line is actually quite slow
    #  if you're calling this function each batch
    if pi is None:
        pi = torch.FloatTensor([np.pi]).to(value.device)

    nll_element = (value - mean).pow(2) / torch.exp(logvar) + logvar + torch.log(2 * pi)
    out = torch.mean(0.5 * nll_element)
    return out


def nll_gaussian_stable(mean, variance, value, pi=None):
    """Compute negative log-likelihood of Gaussian."""
    assert mean.size() == variance.size() == value.size()

    if pi is None:
      pi = torch.FloatTensor([np.pi]).to(value.device)

    logvar = torch.log(variance)

    nll_element = (value - mean).pow(2) / variance + logvar + torch.log(2*pi)

    return torch.mean(0.5*nll_element)


def unfolding_stream(
    image_st: torch.Tensor, kernel_size: int, stride: int, batch_size: int
) -> Generator[torch.Tensor, None, None]:
    _, _, H, W = image_st.shape

    patches = []
    slices = []

    n_patches_y = int(np.floor((H - kernel_size) / stride) + 1)
    n_patches_x = int(np.floor((W - kernel_size) / stride) + 1)

    for i in range(0, n_patches_y):
        for j in range(0, n_patches_x):
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

    if patches:
        yield torch.stack(patches, dim=0), slices


def get_device() -> str:
    if torch.cuda.is_available():
        device = 'cuda'
    elif platform.system() == 'Darwin' and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device


def control_flow_for_device(device: str | None = None) -> str:
    if device is None:
        device = get_device()
    elif isinstance(device, str):
        if device not in ['cpu', 'cuda', 'mps']:
            raise ValueError('device must be one of cpu, cuda, mps')
    return device


def _optimize_model(
    transformer: torch.nn.Module, dtype: str, device: str, batch_size: int, cuda_latest: bool = False
) -> torch.nn.Module:
    allow_ops_in_compiled_graph()

    if device == 'cuda' and cuda_latest:
        import torch_tensorrt

        # Get dimensions
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
            enabled_precisions={dtype},  # e.g., {torch.float}, {torch.float16}
            truncate_long_and_double=True,  # Optional: helps prevent type issues
        )
    elif device == 'cuda' and not cuda_latest:
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
    global _MODEL

    if dtype not in _DTYPE_MAP.keys():
        raise ValueError(f'dtype must be one of {_DTYPE_MAP.keys()}')

    if _MODEL is not None:
        return _MODEL

    if model_token not in ['latest', 'original', 'external']:
        raise ValueError('model_token must be one of latest, original, or external')

    if model_token == 'latest':
        config = transformer_latest_config
        weights_path = TRANSFORMER_WEIGHTS_PATH_LATEST
    elif model_token == 'original':
        config = transformer_config
        weights_path = TRANSFORMER_WEIGHTS_PATH_ORIGINAL
    else:
        with Path.open(model_cfg_path) as cfg:
            config = json.load(cfg)
        weights_path = model_wts_path

    device = control_flow_for_device(device)
    weights = torch.load(weights_path, map_location=device, weights_only=True)
    transformer = SpatioTemporalTransformer(config).to(device)
    transformer.load_state_dict(weights)
    transformer = transformer.eval()

    if optimize:
        transformer = _optimize_model(transformer, dtype=dtype, device=device)

    _MODEL = transformer

    return transformer


@torch.inference_mode()
def _estimate_logit_params_via_streamed_patches(
    model: torch.nn.Module,
    imgs_copol: list[np.ndarray],
    imgs_crosspol: list[np.ndarray],
    stride: int = 2,
    batch_size: int = 32,
    max_nodata_ratio: float = 0.1,
    tqdm_enabled: bool = True,
    device: str | None = None,
) -> tuple[np.ndarray]:
    """Estimate the mean and sigma of the normal distribution of logit input images using low-memory strategy.

    This streams the data in chunks *on the CPU* and requires less GPU memory, but is slower due to data transfer.

    Parameters
    ----------
    model : torch.nn.Module
        transformer with chip (or patch size) 16
    pre_imgs_vv : list[np.ndarray]
    pre_imgs_vh : list[np.ndarray]
        _description_
    stride : int, optional
        Should be between 1 and 16, by default 2.
    batch_size : int, optional
        How to batch chips.

    Returns
    -------
    tuple[np.ndarray]
        pred_mean, pred_sigma (as logits)

    Notes
    -----
    - Applied model to images where mask values are assigned 1e-7
    """
    P = 16
    assert stride <= P
    assert stride > 0

    device = control_flow_for_device(device)

    # stack to T x 2 x H x W
    pre_imgs_stack = _transform_pre_arrs(imgs_copol, imgs_crosspol)
    pre_imgs_stack = pre_imgs_stack.astype('float32')

    # Mask
    mask_stack = np.isnan(pre_imgs_stack)
    # Remove T x 2 dims
    mask_spatial = torch.from_numpy(np.any(mask_stack, axis=(0, 1))).to(device, dtype=DEV_DTYPE)
    assert len(mask_spatial.shape) == 2, 'spatial mask should be 2d'

    # Logit transformation
    pre_imgs_stack[mask_stack] = 1e-7
    pre_imgs_logit = logit(pre_imgs_stack)
    pre_imgs_stack_t = torch.from_numpy(pre_imgs_logit).to(device, dtype=DEV_DTYPE)

    # C x H x W
    C, H, W = pre_imgs_logit.shape[-3:]

    # Sliding window
    n_patches_y = int(np.floor((H - P) / stride) + 1)
    n_patches_x = int(np.floor((W - P) / stride) + 1)
    n_patches = n_patches_y * n_patches_x

    n_batches = math.ceil(n_patches / batch_size)

    target_shape = (C, H, W)
    count = torch.zeros(*target_shape).to(device, dtype=DEV_DTYPE)
    pred_means = torch.zeros(*target_shape).to(device, dtype=DEV_DTYPE)
    pred_logvars = torch.zeros(*target_shape).to(device, dtype=DEV_DTYPE)

    unfold_gen = unfolding_stream(pre_imgs_stack_t, P, stride, batch_size)

    for patch_batch, slices in tqdm(
        unfold_gen,
        total=n_batches,
        desc='Chips Traversed',
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
    pred_means /= count
    pred_logvars /= count

    M_3d = mask_spatial.unsqueeze(dim=0).expand(pred_means.shape)
    pred_means[M_3d] = torch.nan
    pred_logvars[M_3d] = torch.nan

    pred_means = pred_means.cpu().numpy().squeeze()
    pred_logvars = pred_logvars.cpu().numpy().squeeze()
    pred_sigmas = np.sqrt(np.exp(pred_logvars))
    return pred_means, pred_sigmas


@torch.inference_mode()
def _estimate_logit_params_via_folding(
    model: torch.nn.Module,
    imgs_copol: list[np.ndarray],
    imgs_crosspol: list[np.ndarray],
    stride: int = 2,
    batch_size: int = 32,
    device: str | None = None,
    tqdm_enabled: bool = True,
) -> tuple[np.ndarray]:
    """Estimate the mean and sigma of the normal distribution of logit input images using high-memory strategy.

    This uses folding/unfolding which stores pixels reduntly in memory, but is very fast on the GPU.

    Parameters
    ----------
    model : torch.nn.Module
        transformer with chip (or patch size) 16, make sure your model is in evaluation mode
    pre_imgs_vv : list[np.ndarray]
    pre_imgs_vh : list[np.ndarray]
        _description_
    stride : int, optional
        Should be between 1 and 16, by default 2.
    stride : int, optional
        How to batch chips.
    device : str | None, optional
        Device to run the model on. If None, will use the best device available.
        Acceptable values are 'cpu', 'cuda', 'mps'. Defaults to None.

    Returns
    -------
    tuple[np.ndarray]
        pred_mean, pred_sigma (as logits)

    Notes
    -----
    - May apply model to chips of slightly smaller size around boundary
    - Applied model to images where mask values are assigned 1e-7
    """
    P = 16
    assert stride <= P
    assert stride > 0

    device = control_flow_for_device(device)

    # stack to T x 2 x H x W
    pre_imgs_stack = _transform_pre_arrs(imgs_copol, imgs_crosspol)
    pre_imgs_stack = pre_imgs_stack.astype('float32')

    # Mask
    mask_stack = np.isnan(pre_imgs_stack)
    # Remove T x 2 dims
    mask_spatial = torch.from_numpy(np.any(mask_stack, axis=(0, 1)))
    assert len(mask_spatial.shape) == 2, 'spatial mask should be 2d'

    # Logit transformation
    pre_imgs_stack[mask_stack] = 1e-7
    pre_imgs_logit = logit(pre_imgs_stack)

    # H x W
    H, W = pre_imgs_logit.shape[-2:]
    T = pre_imgs_logit.shape[0]
    C = pre_imgs_logit.shape[1]

    # Sliding window
    n_patches_y = int(np.floor((H - P) / stride) + 1)
    n_patches_x = int(np.floor((W - P) / stride) + 1)
    n_patches = n_patches_y * n_patches_x

    # Shape (T x 2 x H x W)
    pre_imgs_stack_t = torch.from_numpy(pre_imgs_logit).to(device, dtype=DEV_DTYPE)
    # T x (2 * P**2) x n_patches
    patches = F.unfold(pre_imgs_stack_t, kernel_size=P, stride=stride)
    # n_patches x T x (C * P**2)
    patches = patches.permute(2, 0, 1).to(device, dtype=DEV_DTYPE)
    # n_patches x T x C x P**2
    patches = patches.view(n_patches, T, C, P**2)

    n_batches = math.ceil(n_patches / batch_size)

    target_chip_shape = (n_patches, C, P, P)
    pred_means_p = torch.zeros(*target_chip_shape).to(device, dtype=DEV_DTYPE)
    pred_logvars_p = torch.zeros(*target_chip_shape).to(device, dtype=DEV_DTYPE)

    for i in tqdm(
        range(n_batches),
        desc='Chips Traversed',
        mininterval=2,
        disable=(not tqdm_enabled),
        dynamic_ncols=True,
    ):
        # change last dimension from P**2 to P, P; use -1 because won't always have batch_size as 0th dimension
        batch_s = slice(batch_size * i, batch_size * (i + 1))
        patch_batch = patches[batch_s, ...].view(-1, T, C, P, P)
        chip_mean, chip_logvar = model(patch_batch)
        pred_means_p[batch_s, ...] += chip_mean
        pred_logvars_p[batch_s, ...] += chip_logvar
    del patches
    torch.cuda.empty_cache()

    # n_patches x C x P x P -->  (C * P**2) x n_patches
    pred_logvars_p_reshaped = pred_logvars_p.view(n_patches, C * P**2).permute(1, 0)
    pred_logvars = F.fold(pred_logvars_p_reshaped, output_size=(H, W), kernel_size=P, stride=stride)
    del pred_logvars_p

    pred_means_p_reshaped = pred_means_p.view(n_patches, C * P**2).permute(1, 0)
    pred_means = F.fold(pred_means_p_reshaped, output_size=(H, W), kernel_size=P, stride=stride)
    del pred_means_p_reshaped

    input_ones = torch.ones(1, H, W).to(device, dtype=DEV_DTYPE)
    count_patches = F.unfold(input_ones, kernel_size=P, stride=stride)
    count = F.fold(count_patches, output_size=(H, W), kernel_size=P, stride=stride)
    del count_patches
    torch.cuda.empty_cache()

    pred_means /= count
    pred_logvars /= count

    mask_3d = mask_spatial.unsqueeze(dim=0).expand(pred_means.shape)
    pred_means[mask_3d] = torch.nan
    pred_logvars[mask_3d] = torch.nan

    pred_means = pred_means.cpu().numpy().squeeze()
    pred_logvars = pred_logvars.cpu().numpy().squeeze()
    pred_sigmas = np.sqrt(np.exp(pred_logvars))
    return pred_means, pred_sigmas


def estimate_normal_params_of_logits(
    model: torch.nn.Module,
    imgs_copol: list[np.ndarray],
    imgs_crosspol: list[np.ndarray],
    stride: int = 2,
    batch_size: int = 32,
    tqdm_enabled: bool = True,
    memory_strategy: str = 'high',
    device: str | None = None,
) -> tuple[np.ndarray]:
    if memory_strategy not in ['high', 'low']:
        raise ValueError('memory strategy must be high or low')

    estimate_logits = (
        _estimate_logit_params_via_folding if memory_strategy == 'high' else _estimate_logit_params_via_streamed_patches
    )

    mu, sigma = estimate_logits(
        model, imgs_copol, imgs_crosspol, stride=stride, batch_size=batch_size, tqdm_enabled=tqdm_enabled, device=device
    )
    return mu, sigma


def log_ratio(pre_imgs, target, is_logit=True, method='mean'):
  '''
  Compute the log ratio damage map

  Pre_imgs: (T, C, H, W)
  Target: (C, H, W)
  '''
  assert len(pre_imgs.shape) == 4
  assert len(target.shape) == 3
  assert pre_imgs.shape[1] == target.shape[0]
  assert pre_imgs.shape[2] == target.shape[1]
  assert pre_imgs.shape[3] == target.shape[2]

  # convert to original SAR space - Sigmoid is inverse of Logit
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

  # add a small perturbation to avoid log(0) or division by 0
  return 10*torch.abs(torch.log10(target + 1e-8) - torch.log10(pred + 1e-8))


def visualize_reconstruction(pre_imgs, post_img, pred_mean, pred_logvar, model_type, is_logit=True, vmin_vv=None, vmax_vv=None, vmin_vh=None, vmax_vh=None):
  '''
  Create a series of images to visualize the performance of a model

  Assume pre imgs has shape (T, C, H, W)
  Assume post_img, pred_mean, and pred_logvar has shape (C, H, W)
  '''
  assert post_img.shape == pred_mean.shape == pred_logvar.shape
  assert len(pre_imgs.shape) == 4
  assert len(post_img.shape) == 3

  if model_type not in ['Transformer', 'RNN']:
    raise ValueError('Not a valid model type')

  # make sure everything is on CPU (required for plotting)
  pre_imgs = pre_imgs.cpu()
  post_img = post_img.cpu()
  pred_mean = pred_mean.cpu()
  pred_logvar = pred_logvar.cpu()

  pred_std_image = torch.sqrt(torch.exp(pred_logvar))

  # compute MSE and NLL
  pred_vs_true_mse = F.mse_loss(pred_mean, post_img)
  pred_vs_true_nll = nll_gaussian(pred_mean, pred_logvar, post_img)

  mean_image = torch.mean(pre_imgs, 0)
  std_image = torch.std(pre_imgs, 0)

  #compute mse and nll
  naive_vs_true_mse = F.mse_loss(mean_image, post_img)
  naive_vs_true_nll = nll_gaussian_stable(mean_image, torch.var(pre_imgs, 0) + 1e-8, post_img)

  # z_score-based damage map
  damage_map = torch.absolute((post_img - pred_mean) / pred_std_image)

  # log ratio damage map
  log_ratio_im = log_ratio(pre_imgs, post_img, is_logit=is_logit)
  print(torch.max(log_ratio_im))
  print(torch.min(log_ratio_im))

  if is_logit:
    pre_imgs = torch.sigmoid(pre_imgs)
    post_img = torch.sigmoid(post_img)
    pred_mean = torch.sigmoid(pred_mean)
    mean_image = torch.sigmoid(mean_image)

  fig, axs = plt.subplots(7, 2, figsize=(10, 30))

  if vmin_vv is None:
    vmin_vv = torch.min(post_img[0,...])
  if vmin_vh is None:
    vmin_vh = torch.min(post_img[1,...])
  if vmax_vv is None:
    vmax_vv = torch.max(post_img[0,...])
  if vmax_vh is None:
    vmax_vh = torch.max(post_img[1,...])

  pred_mean_vv = axs[0,0].imshow(pred_mean[0, ...], vmin=vmin_vv, vmax=vmax_vv, interpolation='None')
  axs[0,0].set_title(f'{model_type} ' + 'Pred Mean VV, MSE={:.4f}'.format(pred_vs_true_mse))
  axs[0,0].axis('off')
  pred_mean_vh = axs[0,1].imshow(pred_mean[1, ...], vmin=vmin_vh, vmax=vmax_vh, interpolation='None')
  axs[0,1].set_title(f'{model_type} ' + 'Pred Mean VH, NLL={:.4f}'.format(pred_vs_true_nll))
  axs[0,1].axis('off')
  plt.colorbar(pred_mean_vv, ax=axs[0,0])
  plt.colorbar(pred_mean_vh, ax=axs[0,1])

  plt.subplots_adjust(hspace=.3)

  mean_vv = axs[1,0].imshow(mean_image[0, ...],  vmin=vmin_vv, vmax=vmax_vv, interpolation='None')
  axs[1,0].set_title('Avg Pre Img VV, MSE={:.4f}'.format(naive_vs_true_mse))
  axs[1,0].axis('off')
  mean_vh = axs[1,1].imshow(mean_image[1, ...],  vmin=vmin_vh, vmax=vmax_vh, interpolation='None')
  axs[1,1].set_title('Avg Pre Img VH, NLL={:.4f}'.format(naive_vs_true_nll))
  axs[1,1].axis('off')
  plt.colorbar(mean_vv, ax=axs[1,0])
  plt.colorbar(mean_vh, ax=axs[1,1])

  plt.subplots_adjust(hspace=.3)

  post_img_vv = axs[2,0].imshow(post_img[0, ...], vmin=vmin_vv, vmax=vmax_vv, interpolation='None')
  axs[2,0].set_title('Ground Truth VV')
  axs[2,0].axis('off')
  post_img_vh = axs[2,1].imshow(post_img[1, ...], vmin=vmin_vh, vmax=vmax_vh, interpolation='None')
  axs[2,1].set_title('Ground Truth VH')
  axs[2,1].axis('off')
  plt.colorbar(post_img_vv, ax=axs[2,0])
  plt.colorbar(post_img_vh, ax=axs[2,1])

  plt.subplots_adjust(hspace=.3)

  # share colorbar between std plots
  std_vmin_vv = torch.min(pred_std_image[0,...])
  std_vmax_vv = torch.max(pred_std_image[0,...])
  std_vmin_vh = torch.min(pred_std_image[1,...])
  std_vmax_vh = torch.max(pred_std_image[1,...])

  print(std_vmax_vv)
  print(std_vmin_vv)
  print(std_vmax_vh)
  print(std_vmin_vh)

  pred_std_vv = axs[3,0].imshow(pred_std_image[0, ...], vmin=std_vmin_vv, vmax=std_vmax_vv, interpolation='None')
  axs[3,0].set_title(f'{model_type} Pred Std VV')
  axs[3,0].axis('off')
  pred_std_vh = axs[3,1].imshow(pred_std_image[1, ...], vmin=std_vmin_vh, vmax=std_vmax_vh, interpolation='None')
  axs[3,1].set_title(f'{model_type} Pred Std VH')
  axs[3,1].axis('off')
  plt.colorbar(pred_std_vv, ax=axs[3,0])
  plt.colorbar(pred_std_vh, ax=axs[3,1])

  plt.subplots_adjust(hspace=.3)

  std_vv = axs[4,0].imshow(std_image[0, ...], vmin=std_vmin_vv, vmax=std_vmax_vv, interpolation='None')
  axs[4,0].set_title('Numerical Pre Img Std VV')
  axs[4,0].axis('off')
  std_vh = axs[4,1].imshow(std_image[1, ...], vmin=std_vmin_vh, vmax=std_vmax_vh, interpolation='None')
  axs[4,1].set_title('Numerical Pre Img Std VH')
  axs[4,1].axis('off')
  plt.colorbar(std_vv, ax=axs[4,0])
  plt.colorbar(std_vh, ax=axs[4,1])

  plt.subplots_adjust(hspace=.3)

  damage_vmax = torch.max(damage_map) * .75
  print(damage_vmax)

  dam_vv = axs[5,0].imshow(damage_map[0, ...], cmap='plasma', interpolation='None', vmax=damage_vmax)
  axs[5,0].set_title(f'{model_type} Z-score Damage Map VV')
  axs[5,0].axis('off')
  dam_vh = axs[5,1].imshow(damage_map[1, ...], cmap='plasma', interpolation='None', vmax=damage_vmax)
  axs[5,1].set_title(f'{model_type} Z-score Damage Map VH')
  axs[5,1].axis('off')

  plt.colorbar(dam_vv,ax=axs[5,0])
  plt.colorbar(dam_vh,ax=axs[5,1])

  lr_vmax = torch.max(log_ratio_im) * .75

  lr_vv = axs[6,0].imshow(log_ratio_im[0, ...], cmap='plasma', interpolation='None', vmax=lr_vmax)
  axs[6,0].set_title('Log Ratio Damage Map VV')
  axs[6,1].axis('off')
  lr_vh = axs[6,1].imshow(log_ratio_im[1, ...], cmap='plasma', interpolation='None', vmax=lr_vmax)
  axs[6,1].set_title('Log Ratio Damage Map VH')
  axs[6,1].axis('off')

  plt.colorbar(lr_vv,ax=axs[6,0])
  plt.colorbar(lr_vh,ax=axs[6,1])

  return damage_map, log_ratio_im