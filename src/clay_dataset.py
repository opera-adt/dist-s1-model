"""
Dataset for loading pre-computed Clay embeddings with original SAR data for supervision.

The Clay embeddings were computed by compute_clay_embeddings.py and stored as:
    - pre_embeddings: (15, 1024, 32, 32) float16
    - post_embedding: (1024, 32, 32) float16
    - acq_dts_float: (16,) float64 — first 15 are pre-dates, last is post-date

For training, we also need the original SAR post-image for the decoder loss.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ClayEmbeddingDataset(Dataset):
    """
    Dataset that loads pre-computed Clay embeddings and original SAR data.

    Returns:
        pre_embeddings: (15, 1024, 32, 32) - Clay embeddings of pre-images
        post_embedding: (1024, 32, 32) - Clay embedding of post-image (for embedding loss)
        pre_dates: (15,) - acquisition dates of pre-images (fractional years)
        post_date: (1,) - acquisition date of post-image (fractional years)
        post_sar: (2, 256, 256) - original SAR post-image (for SAR decoder loss)
    """

    def __init__(
        self,
        embeddings_dir: str = "clay_embeddings",
        original_data_dir: str = "/scratch/opera-dist-ml/users/jmauro/dist-s1-model",
        apply_db_transform: bool = True,
        db_epsilon: float = 1e-10,
        db_min: float = -30.0,
        db_max: float = 10.0,
    ):
        """
        Args:
            embeddings_dir: Directory containing clay embeddings and manifest.csv
            original_data_dir: Directory containing npz_paths.parquet for original SAR data
            apply_db_transform: Whether to convert SAR to dB scale
            db_epsilon: Epsilon for log transform
            db_min: Minimum dB value for clipping
            db_max: Maximum dB value for clipping
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.original_data_dir = Path(original_data_dir)
        self.apply_db_transform = apply_db_transform
        self.db_epsilon = db_epsilon
        self.db_min = db_min
        self.db_max = db_max

        # Load manifest (maps subset_idx -> dataset_idx -> npz_path)
        manifest_path = self.embeddings_dir / "manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")

        self.manifest = pd.read_csv(manifest_path)
        print(f"Loaded {len(self.manifest):,} clay embedding samples from {manifest_path}")

        # Load original dataset parquet to get SAR paths
        parquet_path = self.original_data_dir / "npz_paths.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet not found at {parquet_path}")

        self.original_df = pd.read_parquet(parquet_path)

        # Pre-filter to samples with >= 15 temporal frames (cache this for efficiency)
        # This matches the filtering done in DistS1Dataset
        self.filtered_df = self.original_df[self.original_df['temporal_dim'] >= 15].reset_index(drop=True)
        print(f"Filtered to {len(self.filtered_df):,} samples with >= 15 temporal frames")

        # Build path to original SAR data
        self.sar_base_dir = Path("/scratch/opera-dist-ml/dist-s1-data-updated")

    def __len__(self):
        return len(self.manifest)

    def _to_db(self, x: np.ndarray) -> np.ndarray:
        """Convert raw intensity to dB scale."""
        nan_mask = np.isnan(x)
        x_db = 10.0 * np.log10(np.clip(x, self.db_epsilon, None))
        x_db = np.clip(x_db, self.db_min, self.db_max)
        x_db = np.where(nan_mask, np.nan, x_db)
        return x_db

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        subset_idx = row['subset_idx']
        dataset_idx = row['dataset_idx']
        emb_path = self.embeddings_dir / f"{subset_idx:06d}.npz"

        # Load clay embeddings
        with np.load(emb_path, allow_pickle=False) as npz:
            pre_embeddings = npz['pre_embeddings'].astype(np.float32)  # (15, 1024, 32, 32)
            post_embedding = npz['post_embedding'].astype(np.float32)  # (1024, 32, 32)
            acq_dts_float = npz['acq_dts_float']  # (16,)

        # Split dates: first 15 are pre-dates, last is post-date
        pre_dates = acq_dts_float[:15].astype(np.float32)  # (15,)
        post_date = acq_dts_float[15:16].astype(np.float32)  # (1,)

        # Load original SAR post-image for decoder supervision
        # Get path from original dataset using dataset_idx
        # Note: dataset_idx corresponds to the FILTERED dataset (temporal_dim >= 15)
        # We need to reload the filtered dataset to get the correct mapping
        original_row = self._get_original_sar_path(dataset_idx)
        sar_path = self.sar_base_dir / original_row['npz_path']

        with np.load(sar_path, allow_pickle=False) as npz:
            post_sar = npz['post_img'].astype(np.float32)  # (2, 256, 256)

        # Apply dB transform to SAR if requested
        if self.apply_db_transform:
            post_sar = self._to_db(post_sar)

        return {
            'pre_embeddings': pre_embeddings,   # (15, 1024, 32, 32)
            'post_embedding': post_embedding,   # (1024, 32, 32)
            'pre_dates': pre_dates,             # (15,)
            'post_date': post_date,             # (1,)
            'post_sar': post_sar,               # (2, 256, 256)
            'dataset_idx': dataset_idx,         # int — for loading pre-SAR on demand
        }

    def _get_original_sar_path(self, dataset_idx: int):
        """
        Get the original SAR data path for a given dataset index.

        The dataset_idx from the manifest corresponds to the index in the
        FILTERED dataset (temporal_dim >= 15), not the raw parquet.
        We use the pre-cached filtered_df for efficiency.
        """
        if dataset_idx >= len(self.filtered_df):
            raise IndexError(f"dataset_idx {dataset_idx} out of range for filtered dataset ({len(self.filtered_df)} samples)")

        return self.filtered_df.iloc[dataset_idx]


def clay_collate_fn(batch):
    """
    Collate function for ClayEmbeddingDataset.

    Stacks all tensors and moves to appropriate format.
    """
    pre_embeddings = torch.stack([torch.from_numpy(item['pre_embeddings']) for item in batch])
    post_embedding = torch.stack([torch.from_numpy(item['post_embedding']) for item in batch])
    pre_dates = torch.stack([torch.from_numpy(item['pre_dates']) for item in batch])
    post_date = torch.stack([torch.from_numpy(item['post_date']) for item in batch])
    post_sar = torch.stack([torch.from_numpy(item['post_sar']) for item in batch])
    dataset_idxs = [item['dataset_idx'] for item in batch]

    return {
        'pre_embeddings': pre_embeddings,   # (B, 15, 1024, 32, 32)
        'post_embedding': post_embedding,   # (B, 1024, 32, 32)
        'pre_dates': pre_dates,             # (B, 15)
        'post_date': post_date,             # (B, 1)
        'post_sar': post_sar,               # (B, 2, 256, 256)
        'dataset_idxs': dataset_idxs,       # list of int
    }
