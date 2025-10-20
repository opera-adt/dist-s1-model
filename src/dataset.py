import math
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info


class DistS1Dataset(Dataset):
    def __init__(self, root_dir=Path('.'), transform=None):
        self.root_dir = Path(root_dir)
        self._parquet_dir = self.root_dir / 'npz_paths'
        self._dataset_dir = self.root_dir / 'dataset_samples_npz'

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

        return sample
    

class StreamShardDataset(IterableDataset):
    def __init__(self, shard_paths, prefix='train', shuffle=True):
        self.shard_paths = list(Path(shard_paths).glob(f"{prefix}_*.pt"))
        self.shuffle     = shuffle

    def _shard_iterator(self, paths):
        for p in paths:
            x, y = torch.load(p, map_location='cpu', weights_only=False)
            for xi, yi in zip(x, y):
                yield xi, yi
            del x, y

    def __iter__(self):
        # Worker-aware sharding
        info = get_worker_info()
        paths = self.shard_paths
        if self.shuffle:
            random.shuffle(paths)

        if info is not None:
            n = len(paths)
            per_worker = int(math.ceil(n / info.num_workers))
            paths = paths[info.id*per_worker : (info.id+1)*per_worker]

        return self._shard_iterator(paths)

    def __len__(self):
        total = 0
        for p in self.shard_paths:
            x, _ = torch.load(p, map_location='cpu', weights_only=False)
            total += x.size(0)
            del x
        return total
