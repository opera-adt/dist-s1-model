"""
Compute Clay v1.5 encoder embeddings for the 5% training subset.

This script runs each image (15 pre + 1 post per sample) through the frozen
Clay foundation model encoder and saves the resulting (1024, 32, 32) feature
maps to disk. These embeddings can then be used as input features for the
temporal disturbance model.

Usage:
    # First install Clay:
    #   pip install git+https://github.com/Clay-foundation/model.git
    #
    # Download the checkpoint:
    #   wget https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt

    # Test on 5 samples:
    python compute_clay_embeddings.py --max-samples 5

    # Full run:
    python compute_clay_embeddings.py

    # Custom output dir and batch size:
    python compute_clay_embeddings.py --output-dir /scratch/clay_emb --batch-size 32
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.dataset_v3 import DistS1Dataset


# ---------------------------------------------------------------------------
# Clay S1-RTC metadata (from Clay configs/metadata.yaml)
# ---------------------------------------------------------------------------
S1_MEAN = torch.tensor([-12.113, -18.673]).view(1, 2, 1, 1)  # dB
S1_STD = torch.tensor([8.314, 8.017]).view(1, 2, 1, 1)
S1_WAVES = torch.tensor([3.5, 4.0], dtype=torch.float32)  # cm
S1_GSD = torch.tensor(10.0, dtype=torch.float32)


def fractional_year_to_time_encoding(acq_dts_float: np.ndarray) -> np.ndarray:
    """Convert fractional-year timestamps to Clay's [sin(w), cos(w), sin(h), cos(h)] format.

    Clay expects time as 4-d sinusoidal encoding of (week-of-year, hour-of-day).
    SAR acquisitions don't include hour info, so hour is set to 0.

    Args:
        acq_dts_float: array of shape (...,) with fractional years (e.g. 2023.456)

    Returns:
        array of shape (..., 4)
    """
    flat = np.asarray(acq_dts_float, dtype=np.float64).ravel()
    out = np.zeros((flat.shape[0], 4), dtype=np.float32)

    for i, fy in enumerate(flat):
        year = int(fy)
        frac = fy - year
        # Approximate day-of-year (ignore leap year nuance)
        day_of_year = frac * 365.25
        week = (day_of_year / 7.0) % 52
        week_rad = week * 2 * np.pi / 52
        # Hour unknown for SAR — default to 0
        hour_rad = 0.0
        out[i] = [math.sin(week_rad), math.cos(week_rad),
                  math.sin(hour_rad), math.cos(hour_rad)]

    target_shape = acq_dts_float.shape + (4,)
    return out.reshape(target_shape)


def intensity_to_db(x: torch.Tensor, eps: float = 1e-10,
                    db_min: float = -30.0, db_max: float = 10.0) -> torch.Tensor:
    """Convert raw SAR intensity to dB scale, matching the training pipeline."""
    nan_mask = torch.isnan(x)
    x_db = 10.0 * torch.log10(torch.clamp(x, min=eps))
    x_db = torch.clamp(x_db, db_min, db_max)
    x_db = torch.where(nan_mask, torch.zeros_like(x_db), x_db)
    return x_db


def load_clay_model(checkpoint_path: str, device: torch.device,
                    metadata_path: str = "configs/metadata.yaml"):
    """Load Clay v1.5 encoder in eval / frozen mode."""
    from claymodel.module import ClayMAEModule

    model = ClayMAEModule.load_from_checkpoint(
        checkpoint_path,
        model_size="large",
        mask_ratio=0.0,
        shuffle=False,
        metadata_path=metadata_path,
    )
    model.eval()
    model = model.to(device)

    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad = False

    print(f"Clay model loaded on {device}  "
          f"({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)")
    return model


@torch.no_grad()
def encode_batch(model, pixels_db: torch.Tensor, time_enc: torch.Tensor,
                 device: torch.device) -> torch.Tensor:
    """Run a batch of images through the Clay encoder.

    Args:
        model: loaded ClayMAEModule
        pixels_db: (B, 2, 256, 256) in dB scale (NOT yet normalized)
        time_enc: (B, 4) sinusoidal time encoding
        device: torch device

    Returns:
        patch_features: (B, 1024, 32, 32) float16 feature maps
    """
    # Normalize for Clay
    mean = S1_MEAN.to(device)
    std = S1_STD.to(device)
    pixels_norm = (pixels_db - mean) / std

    datacube = {
        "pixels": pixels_norm,
        "time": time_enc.to(device),
        "latlon": torch.zeros(pixels_norm.shape[0], 4, device=device),
        "gsd": S1_GSD.to(device),
        "waves": S1_WAVES.to(device),
    }

    # Encoder returns (unmsk_patch, unmsk_idx, msk_idx, msk_matrix)
    # unmsk_patch shape: (B, 1025, 1024) — index 0 is CLS token
    unmsk_patch, _, _, _ = model.model.encoder(datacube)

    # Drop CLS token, reshape to spatial feature map
    patch_emb = unmsk_patch[:, 1:, :]  # (B, 1024, 1024)
    patch_emb = patch_emb.reshape(-1, 32, 32, 1024)  # (B, 32, 32, 1024)
    patch_emb = patch_emb.permute(0, 3, 1, 2)  # (B, 1024, 32, 32)

    return patch_emb.half()


def main():
    parser = argparse.ArgumentParser(description="Compute Clay embeddings for the training subset")
    parser.add_argument("--checkpoint", type=str, default="clay-v1.5.ckpt",
                        help="Path to Clay v1.5 checkpoint")
    parser.add_argument("--metadata-path", type=str, default="configs/metadata.yaml",
                        help="Path to Clay metadata.yaml")
    parser.add_argument("--data-dir", type=str,
                        default="/scratch/opera-dist-ml/users/jmauro/dist-s1-model",
                        help="Root dir containing npz_paths.parquet")
    parser.add_argument("--output-dir", type=str, default="clay_embeddings",
                        help="Directory to save embeddings")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Number of images to encode at once (not samples)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples (for testing)")
    parser.add_argument("--subset-fraction", type=float, default=0.10,
                        help="Fraction of dataset to process (default 0.05 = 5%%)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load dataset and create the same 5% subset as trainer_redux.py
    # ------------------------------------------------------------------
    print("Loading dataset...")
    dataset = DistS1Dataset(
        root_dir=args.data_dir,
        temporal_length=15,
        random_selection=False,
    )

    subset_size = int(args.subset_fraction * len(dataset))
    generator = torch.Generator().manual_seed(42)
    subset_indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()

    if args.max_samples is not None:
        subset_indices = subset_indices[:args.max_samples]
        subset_size = len(subset_indices)

    print(f"Dataset size: {len(dataset):,}")
    print(f"Subset size:  {subset_size:,}")
    print(f"Images to encode: {subset_size * 16:,} (15 pre + 1 post each)")

    # ------------------------------------------------------------------
    # 2. Load Clay encoder
    # ------------------------------------------------------------------
    print("Loading Clay model...")
    model = load_clay_model(args.checkpoint, device, args.metadata_path)

    # ------------------------------------------------------------------
    # 3. Process each sample
    # ------------------------------------------------------------------
    manifest_rows = []
    t0 = time.time()

    for i, dataset_idx in enumerate(subset_indices):
        out_path = output_dir / f"{i:06d}.npz"

        # Skip if already computed (resumable)
        if out_path.exists():
            manifest_rows.append({
                "subset_idx": i,
                "dataset_idx": dataset_idx,
                "npz_path": str(out_path),
            })
            if i % 500 == 0:
                print(f"[{i}/{subset_size}] Already exists, skipping")
            continue

        # Load sample from dataset
        sample = dataset[dataset_idx]
        pre_imgs = torch.from_numpy(sample["pre_imgs"]).float()    # (15, 2, 256, 256)
        post_img = torch.from_numpy(sample["post_img"]).float()    # (2, 256, 256)
        acq_dts = sample["acq_dts_float"]                          # (16,) float64

        # Stack all images: (16, 2, 256, 256)
        all_imgs = torch.cat([pre_imgs, post_img.unsqueeze(0)], dim=0)

        # Convert to dB
        all_imgs_db = intensity_to_db(all_imgs)

        # Build time encodings for all 16 images
        time_enc_np = fractional_year_to_time_encoding(acq_dts)  # (16, 4)
        time_enc = torch.from_numpy(time_enc_np).float()

        # Encode in batches (16 images, maybe more than batch_size)
        all_embeddings = []
        n_imgs = all_imgs_db.shape[0]

        for start in range(0, n_imgs, args.batch_size):
            end = min(start + args.batch_size, n_imgs)
            batch_pixels = all_imgs_db[start:end].to(device)
            batch_time = time_enc[start:end].to(device)

            emb = encode_batch(model, batch_pixels, batch_time, device)
            all_embeddings.append(emb.cpu())

        all_embeddings = torch.cat(all_embeddings, dim=0)  # (16, 1024, 32, 32) float16

        # Split back into pre and post
        pre_embeddings = all_embeddings[:15].numpy()   # (15, 1024, 32, 32)
        post_embedding = all_embeddings[15].numpy()    # (1024, 32, 32)

        # Save
        np.savez(
            out_path,
            pre_embeddings=pre_embeddings,
            post_embedding=post_embedding,
            acq_dts_float=acq_dts,
        )

        manifest_rows.append({
            "subset_idx": i,
            "dataset_idx": dataset_idx,
            "npz_path": str(out_path),
        })

        # Progress
        if (i + 1) % 100 == 0 or (i + 1) == subset_size:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (subset_size - i - 1) / rate if rate > 0 else 0
            print(f"[{i + 1}/{subset_size}] "
                  f"{rate:.1f} samples/s, "
                  f"ETA {eta / 3600:.1f}h, "
                  f"file: {out_path.name}")

        # Periodic cache clearing
        if (i + 1) % 500 == 0:
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 4. Save manifest
    # ------------------------------------------------------------------
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\nDone. Manifest saved to {manifest_path}")
    print(f"Total time: {(time.time() - t0) / 3600:.2f} hours")
    print(f"Output dir: {output_dir}")
    print(f"Sample shapes: pre_embeddings=(15,1024,32,32), post_embedding=(1024,32,32)")


if __name__ == "__main__":
    main()
