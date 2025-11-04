from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from distmetrics.despeckle import interpolate_arr


def despeckle_image(img: np.ndarray) -> np.ndarray:
    """
    Despeckle a single image (C, H, W) by interpolating NaN values using bilinear interpolation.

    Args:
        img: Image array of shape (C, H, W)

    Returns:
        Despeckled image of same shape
    """
    despeckled = np.zeros_like(img)
    for c in range(img.shape[0]):
        despeckled[c] = interpolate_arr(
            img[c],
            interp_method='bilinear',
            preserve_exterior_mask=True,
            n_iter_bilinear=10
        )
    return despeckled


def to_db(img: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Convert image to dB scale: 10 * log10(img).

    Args:
        img: Image array
        eps: Small epsilon to avoid log(0)

    Returns:
        Image in dB scale
    """
    return 10 * np.log10(np.maximum(img, eps))


class DistS1Dataset(Dataset):
    def __init__(self, root_dir=Path('.'), transform=None, apply_despeckle=True, apply_db_transform=True):
        self.root_dir = Path(root_dir)
        self._parquet_dir = self.root_dir / 'npz_paths'
        self._dataset_dir = self.root_dir / 'dataset_samples_npz'
        self.apply_despeckle = apply_despeckle
        self.apply_db_transform = apply_db_transform

        # Load and concatenate all Parquet files
        self.df = self._load_parquet_files()

        # Validate the presence of npz_path column
        if 'npz_path' not in self.df.columns:
            raise ValueError("'npz_path' column is required in the parquet files.")

    def _load_parquet_files(self):
        parquet_files = sorted(self._parquet_dir.glob('*.parquet'))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {str(self.root_dir / self._parquet_dir)}")
        df_list = [pd.read_parquet(pf) for pf in parquet_files]
        df = pd.concat(df_list, ignore_index=True)
        df = df.drop_duplicates().reset_index(drop=True)
        # the paths are relative to the two directories being parallel in the current working directory
        # so we add the root
        df['npz_path'] = f'{str(self.root_dir)}/' + df['npz_path']
        print(f'there were {df.shape[0]:,} samples found')
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npz_path = row['npz_path']

        # Load the .npz file
        with np.load(npz_path, allow_pickle=False) as npz:
            sample = {key: npz[key] for key in npz.files}

        # Apply despeckling before dB transform
        if self.apply_despeckle:
            # Despeckle pre images (T, C, H, W)
            if 'pre_imgs' in sample:
                pre_imgs = sample['pre_imgs']
                despeckled_pre = np.zeros_like(pre_imgs)
                for t in range(pre_imgs.shape[0]):
                    despeckled_pre[t] = despeckle_image(pre_imgs[t])
                sample['pre_imgs'] = despeckled_pre

            # Despeckle post image (C, H, W)
            if 'post_img' in sample:
                sample['post_img'] = despeckle_image(sample['post_img'])

        # Apply dB transform after despeckling
        if self.apply_db_transform:
            if 'pre_imgs' in sample:
                sample['pre_imgs'] = to_db(sample['pre_imgs'])
            if 'post_img' in sample:
                sample['post_img'] = to_db(sample['post_img'])

        # Calculate relative acquisition times (relative to post image)
        if 'acq_dts_float' in sample:
            acq_dts_float = sample['acq_dts_float']  # Shape: (T+1,) where last element is post time
            post_time = acq_dts_float[-1]
            pre_times = acq_dts_float[:-1]

            # Calculate relative times: post_time - pre_time (positive values, larger = further in past)
            relative_times = post_time - pre_times  # Shape: (T,)

            # Append 0 for post image (its relative time from itself is 0)
            relative_times_with_post = np.append(relative_times, 0.0)  # Shape: (T+1,)

            # Store both original and relative times
            sample['acq_dts_float_original'] = acq_dts_float
            sample['acq_dts_float'] = relative_times_with_post

        return sample