import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np
from torch.utils.data import DataLoader, random_split, RandomSampler, SequentialSampler
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset, ConcatDataset
import time
import math

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

def animate_data(pre_imgs, post_img, is_logit=True, interval=250, vmin=None, vmax=None):
  '''
  Function that animates a data sample

  Pre_imgs: (T, C, H, W)
  Post_img: (C, H, W)
  '''
  assert pre_imgs.shape[1] == post_img.shape[0]
  assert pre_imgs.shape[2] == post_img.shape[1]
  assert pre_imgs.shape[3] == post_img.shape[2]
  assert len(pre_imgs.shape) == 4
  assert len(post_img.shape) == 3

  if is_logit:
    pre_imgs = torch.sigmoid(pre_imgs)
    post_img = torch.sigmoid(post_img)

  fig, (ax1, ax2) = plt.subplots(1, 2)

  vv = pre_imgs[0, 0, ...]
  vh = pre_imgs[0, 1, ...]

  vv_plot = ax1.imshow(vv, interpolation='None', vmin=vmin, vmax=vmax)
  ax1.set_title('VV: Frame 1')
  vh_plot = ax2.imshow(vh, interpolation='None', vmin=vmin, vmax=vmax)
  ax2.set_title('VH: Frame 1')
  plt.colorbar(vv_plot, ax=ax1,fraction=0.046, pad=0.04)
  plt.colorbar(vh_plot, ax=ax2,fraction=0.046, pad=0.04)
  plt.subplots_adjust(wspace=0.50)

  def update(frame):

      if frame == pre_imgs.shape[0]:
        vv_plot.set_array(post_img[0, ...])
        ax1.set_title(f'VV: Frame {frame} (Target)')
        vh_plot.set_array(post_img[1, ...])
        ax2.set_title(f'VH: Frame {frame} (Target)')
        return [vv_plot, vh_plot]

      vv_plot.set_array(pre_imgs[frame, 0, ...])
      ax1.set_title(f'VV: Frame {frame}')
      vh_plot.set_array(pre_imgs[frame, 1, ...])
      ax2.set_title(f'VH: Frame {frame}')
      return [vv_plot, vh_plot]

  ani = animation.FuncAnimation(fig=fig, func=update, frames=pre_imgs.shape[0]+1, interval=interval, blit=True)

  return ani

def nll_gaussian(mean, logvar, value, pi=None):
    """Compute negative log-likelihood of Gaussian."""
    assert mean.size() == logvar.size() == value.size()

    # this is just for efficiency reasons - the below line is actually quite slow if you're calling this function each batch
    if pi is None:
        pi = torch.FloatTensor([np.pi]).to(value.device)
    
    nll_element = (value - mean).pow(2) / torch.exp(logvar) + logvar + torch.log(2*pi)
    out = torch.mean(0.5*nll_element)
    return out

def nll_gaussian_stable(mean, variance, value):
    """Compute negative log-likelihood of Gaussian."""
    assert mean.size() == variance.size() == value.size()
    pi = torch.FloatTensor([np.pi]).to(value.device)

    logvar = torch.log(variance)

    nll_element = (value - mean).pow(2) / variance + logvar + torch.log(2*pi)

    return torch.mean(0.5*nll_element)