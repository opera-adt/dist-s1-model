import bisect
import multiprocessing as mp
from functools import lru_cache
from pathlib import Path

import backoff
import numpy as np
import pandas as pd
import rasterio
import torch
from mpire import WorkerPool
from rasterio.profiles import DefaultGTiffProfile
from rasterio.windows import Window
from requests.exceptions import HTTPError
from skimage.restoration import denoise_tv_bregman
from torch.utils.data import Dataset
from tqdm import tqdm
import time
import concurrent.futures

def despeckle_one(
    X: np.ndarray, reg_param=5, noise_floor_db=-22, preserve_nans=False
) -> np.ndarray:
    X_c = np.clip(X, 1e-7, 1)
    X_db = 10 * np.log10(X_c, out=np.full(X_c.shape, np.nan), where=(~np.isnan(X_c)))
    X_db[np.isnan(X_c)] = noise_floor_db
    X_db_dspkl = denoise_tv_bregman(
        X_db, weight=1.0 / reg_param, isotropic=True, eps=1e-3
    )
    X_dspkl = np.power(10, X_db_dspkl / 10.0)
    if preserve_nans:
        X_dspkl[np.isnan(X)] = np.nan
    X_dspkl = np.clip(X_dspkl, 0, 1)
    return X_dspkl


@lru_cache
def open_rtc_table() -> pd.DataFrame:
    return pd.read_parquet("rtc_s1_tables/rtc_s1_data_val_bursts.parquet")


def read_raster_from_slice(
    url: str, x_start, y_start, x_stop, y_stop
) -> tuple[np.ndarray, dict]:
    rows = (y_start, y_stop)
    cols = (x_start, x_stop)
    window = Window.from_slices(rows=rows, cols=cols)

    with rasterio.open(url) as ds:
        X = ds.read(window=window).astype(np.float32)
        t_window = ds.window_transform(window)
        crs = ds.crs
        count = ds.count

    p = DefaultGTiffProfile()
    p["dtype"] = "float32"
    p["transform"] = t_window
    p["crs"] = crs
    p["nodata"] = np.nan
    p["count"] = count
    _, p["height"], p["width"] = X.shape

    return X, p


@backoff.on_exception(
    backoff.expo,
    [ConnectionError, HTTPError],
    max_tries=10,
    max_time=60,
    jitter=backoff.full_jitter,
)
def localize_one_rtc(
    url: str | Path | list | tuple,
    ts_dir: str | Path = Path("."),
    with_despeckling=True,
    preserve_nans=True,
) -> Path:
    if isinstance(url, (list, tuple)):
        local_fn = url[0].split("/")[-1]
        local_fn = f"{local_fn[:-7]}_stack.tif"
    else:
        local_fn = url.split("/")[-1]
    burst_id = local_fn.split("_")[3]
    out_dir = Path(ts_dir) / burst_id
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / local_fn

    if out_path.exists():
        return out_path

    def open_one(url: str) -> tuple:
        with rasterio.open(url) as ds:
            X = ds.read(1).astype(np.float32)
            p = ds.profile
        return X, p

    if not isinstance(url, (list, tuple)):
        arrs, ps = zip(*[open_one(url)])
    else:
        arrs, ps = zip(*list(map(open_one, url)))

    if with_despeckling:

        def despeckle_one_p(arr: np.ndarray) -> np.ndarray:
            return despeckle_one(arr, preserve_nans=preserve_nans)

        arrs = list(map(despeckle_one, arrs))

    X_out = np.stack(arrs, axis=0)
    p = ps[0].copy()
    p["count"] = X_out.shape[0]
    with rasterio.open(out_path, "w", **p) as ds:
        ds.write(X_out)
    return out_path


def localize_one_rtc_with_slice(
    url: str | Path,
    x_start: int,
    y_start: int,
    window_size: int,
    local_fn_suffix: str = None,
    rtc_dir: str | Path = Path("."),
    with_despeckling: bool = True,
) -> Path:
    local_fn = Path(url.split("/")[-1])
    if local_fn_suffix is not None:
        local_fn = f"{local_fn.stem} + {local_fn_suffix} + {local_fn.sufix}"
    burst_id = local_fn.split("_")[3]
    out_dir = Path(rtc_dir) / burst_id
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / local_fn

    if out_path.exists():
        return out_path

    X_w, p_w = read_raster_from_slice(
        url, x_start, y_start, x_start + window_size, y_start + window_size
    )

    X_d = X_w[0, ...].copy()
    if with_despeckling:
        X_d = despeckle_one(X_d, preserve_nans=False)

    with rasterio.open(out_path, "w", **p_w) as ds:
        ds.write(X_d, 1)
    return out_path


class SeqDistDataset(Dataset):
    def __init__(
        self,
        rtc_s1_parquet_table_path: str | Path = None,
        patch_size: int = 16,
        n_pre_imgs: int = 4,
        root: Path | str = Path("opera_rtc_data"),
        n_workers_for_download: int = 5,
    ):
        self.root = root

        self.patch_data_dir = Path("patch_data")
        if not self.patch_data_dir.exists():
            raise ValueError(f"{self.patch_data_dir} does not exist")
        self.patch_data_sizes = sorted(
            [int(p.stem.split("_")[-1]) for p in self.patch_data_dir.glob("*.parquet")]
        )
        if patch_size not in self.patch_data_sizes:
            raise ValueError(
                f'{patch_size} is not in {",".join(self.patch_data_sizes)}'
            )
        self.patch_size = patch_size

        self.df_rtc_meta = (
            pd.read_parquet(rtc_s1_parquet_table_path)
            if rtc_s1_parquet_table_path is not None
            else open_rtc_table()
        )
        self.df_patch = pd.read_parquet(
            self.patch_data_dir / f"burst_patch_table_{patch_size}.parquet"
        )

        self.n_pre_imgs = n_pre_imgs

        # Requested CPU count or available CPUs, whichever smaller
        target_cpus = min(mp.cpu_count(), n_workers_for_download)
        self.n_workers_for_download = target_cpus

        self.patch_by_burst_data_dir = Path("patch_by_burst_tables")
        self.patch_by_burst_data_dir.mkdir(exist_ok=True, parents=True)

        self.rtc_by_burst_data_dir = Path("rtc_by_burst_tables")
        self.rtc_by_burst_data_dir.mkdir(exist_ok=True, parents=True)

        self.download_rtc_data()
        self.generate_spatio_temporal_sample_data()

    def generate_spatio_temporal_sample_data(self):
        df_date_count_by_burst = (
            self.df_rtc_meta.groupby("jpl_burst_id")
            .size()
            .reset_index(drop=False)
            .rename(columns={0: "acq_per_burst"})
        )
        df_patch_count_by_burst = (
            self.df_patch.groupby("jpl_burst_id")
            .size()
            .reset_index(drop=False)
            .rename(columns={0: "patches_per_burst"})
        )

        df_count_rtc = pd.merge(
            df_date_count_by_burst,
            df_patch_count_by_burst,
            on="jpl_burst_id",
            how="inner",
        )
        df_count_rtc["total_samples_per_burst"] = (
            df_count_rtc["acq_per_burst"] * df_count_rtc["patches_per_burst"]
        )
        df_count_rtc["total_samples_cumul"] = (
            df_count_rtc.total_samples_per_burst.cumsum()
        )

        self.df_st_sample = df_count_rtc
        self.burst_ids = self.df_st_sample.jpl_burst_id.unique().tolist()
        self.df_rtc_meta = self.df_rtc_meta[
            self.df_rtc_meta.jpl_burst_id.isin(self.burst_ids)
        ].reset_index(drop=True)

        self.df_patch = None
        self.df_rtc_meta = None

    def download_rtc_data(self):
        def localize_one_rtc_p(*urls):
            return localize_one_rtc(urls, ts_dir=self.root)

        input_data = [
            [vv_path, vh_path]
            for (vv_path, vh_path) in zip(
                self.df_rtc_meta.rtc_s1_vv_url.tolist(),
                self.df_rtc_meta.rtc_s1_vh_url.tolist(),
            )
        ]
        with WorkerPool(n_jobs=self.n_workers_for_download, use_dill=True) as pool:
            dual_loc_path = pool.map(
                localize_one_rtc_p,
                input_data,
                progress_bar=True,
                progress_bar_style="std",
                concatenate_numpy_output=False,
            )

        self.df_rtc_meta["rtc_s1_stack_loc_path"] = [str(p) for p in dual_loc_path]

        self.patch_by_burst_data_dir.mkdir(exist_ok=True, parents=True)
        burst_ids = self.df_rtc_meta.jpl_burst_id.unique().tolist()
        for burst_id in tqdm(burst_ids, desc="localize patch tables"):
            patch_out_path = self.patch_by_burst_data_dir / f"patch_{burst_id}_{self.patch_size}.parquet"
            if patch_out_path.exists():
                continue
            else:
                df_burst_patches = self.df_patch[
                    self.df_patch.jpl_burst_id == burst_id
                ].reset_index(drop=True)
                df_burst_patches.to_parquet(
                    patch_out_path, compression="snappy", engine="pyarrow"
                )

            rtc_out_path = self.rtc_by_burst_data_dir / f"rtc_{burst_id}.parquet"
            if rtc_out_path.exists():
                continue
            else:
                df_rtc_ts = self.df_rtc_meta[
                    self.df_rtc_meta.jpl_burst_id == burst_id
                ].reset_index(drop=True)
                df_rtc_ts.to_parquet(
                    rtc_out_path, compression="snappy", engine="pyarrow"
                )

    def __len__(self):
        return self.df_st_sample.total_samples_per_burst.sum()

    def __getitem__(self, idx):

        # start_time = time.time()
        
        burst_idx = bisect.bisect_left(
            self.df_st_sample.total_samples_cumul.tolist(), idx
        )
        assert self.df_st_sample.iloc[burst_idx].total_samples_cumul >= idx

        burst_id = self.df_st_sample.iloc[burst_idx].jpl_burst_id
        # As we did for df_count, we need to subtract the n preimages and 1 pre-image
        acq_for_burst_lookup = (
            self.df_st_sample.iloc[burst_idx].acq_per_burst - self.n_pre_imgs - 1
        )
        patches_for_burst = self.df_st_sample.iloc[burst_idx].patches_per_burst

        # Key here is we need the n samples < idx to determine how to sample patch and acq time
        total_samples_running_idx = (
            self.df_st_sample.iloc[burst_idx - 1].total_samples_cumul
            if burst_idx > 0
            else 0
        )
        assert total_samples_running_idx <= idx

        acq_idx = (idx - total_samples_running_idx) % acq_for_burst_lookup
        rtc_ts_path = self.rtc_by_burst_data_dir / f"rtc_{burst_id}.parquet"

        # read_parquet_start = time.time()
        df_ts_t = pd.read_parquet(rtc_ts_path)
        # print(f"Time to read parquet file {rtc_ts_path}: {time.time() - read_parquet_start:.4f} seconds")

        # Add 1 to index to get post image
        # df_ts is a dataframe of n_preimgs + 1 in dim 0 to provide metadata for one sample of sequence
        df_ts = df_ts_t.iloc[acq_idx: acq_idx + self.n_pre_imgs + 1].reset_index(
            drop=True
        )
        assert df_ts.shape[0] == (
            self.n_pre_imgs + 1
        ), f"Issue with {idx=}, {burst_idx=}"

        patch_idx = (idx - total_samples_running_idx) % patches_for_burst

        patch_path = self.patch_by_burst_data_dir / f"patch_{burst_id}_{self.patch_size}.parquet"

        # read_patch_start = time.time()  
        df_patch_burst = pd.read_parquet(patch_path)
        # print(f"Time to read patch file {patch_path}: {time.time() - read_patch_start:.4f} seconds")
        
        patch_data = df_patch_burst.iloc[patch_idx].to_dict()

        # return torch.zeros((3, 3))
        def read_window_p(url: str):
            X, p = read_raster_from_slice(
                url,
                patch_data["x_start"],
                patch_data["y_start"],
                patch_data["x_start"] + self.patch_size,
                patch_data["y_start"] + self.patch_size,
            )
            return X, p

        img_paths = df_ts.rtc_s1_stack_loc_path.tolist()

        # img_loading_start = time.time()

        ###
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            arrs = list(executor.map(read_window_p, img_paths))

        arrs, _ = zip(*arrs)
        ###
        
        # arrs, _ = zip(*[read_window_p(p) for p in img_paths])
        # print(f"Time to load images: {time.time() - img_loading_start:.4f} seconds")

        # Input for modeling
        # pre img is n_pre_imgs X 2 X H X W
        pre_imgs = np.stack(arrs[: self.n_pre_imgs], axis=0)
        # post img is 2 X H X W
        post_img = arrs[-1]

        # print(f"Data loading time: {time.time() - start_time} seconds")

        # additional metadata
        return {
            "pre_imgs": torch.from_numpy(pre_imgs),
            "post_img": torch.from_numpy(post_img),
        }

from dateutil.relativedelta import relativedelta

class MultiYearDistDataset(Dataset):
    def __init__(
        self,
        rtc_s1_parquet_table_path: str | Path = None,
        patch_size: int = 16,
        n_pre_imgs: int = 4,
        root: Path | str = Path("opera_rtc_data"),
        n_workers_for_download: int = 5,
        download_flag=True
    ):
        self.root = root

        self.year_offsets  = [2, 1, 0]        # “years ago”
        self.samples_each  = [4, 3, 3]        # how many per offset
        self.window_days   = 90          # width of the seasonal window
        self.n_pre_imgs    = sum(self.samples_each)   # 9 in this scheme

        self.patch_data_dir = Path("jk_patch_data")
        if not self.patch_data_dir.exists():
            raise ValueError(f"{self.patch_data_dir} does not exist")
        self.patch_data_sizes = sorted(
            [int(p.stem.split("_")[-1]) for p in self.patch_data_dir.glob("*.parquet")]
        )

        if patch_size not in self.patch_data_sizes:
            raise ValueError(
                f'{patch_size} is not in {",".join(self.patch_data_sizes)}'
            )
        self.patch_size = patch_size

        self.df_rtc_meta = (
            pd.read_parquet(rtc_s1_parquet_table_path)
            if rtc_s1_parquet_table_path is not None
            else open_rtc_table()
        )
        self.df_patch = pd.read_parquet(
            self.patch_data_dir / f"burst_patch_data_{patch_size}.parquet"
        )

        # Requested CPU count or available CPUs, whichever smaller
        target_cpus = min(mp.cpu_count(), n_workers_for_download)
        self.n_workers_for_download = target_cpus

        self.patch_by_burst_data_dir = Path("patch_by_burst_tables")
        self.patch_by_burst_data_dir.mkdir(exist_ok=True, parents=True)

        self.rtc_by_burst_data_dir = Path("rtc_by_burst_tables3")
        self.rtc_by_burst_data_dir.mkdir(exist_ok=True, parents=True)

        self.download_rtc_data()
        self.generate_spatio_temporal_sample_data()

    def _collect_history(self, post_date, df_ts):
        """Return list[indices] for the 10-scene history or None if impossible."""
        hist_idx = []  
        for yrs, n_pick in zip(self.year_offsets, self.samples_each):
            hi = post_date - relativedelta(years=yrs)
            lo = hi - pd.Timedelta(days=self.window_days)
            window = df_ts[(df_ts.acq_datetime > lo) & (df_ts.acq_datetime < hi)]
            hist_idx.extend(window.tail(n_pick).index.tolist())

        # drop duplicates but keep order
        hist_idx = list(dict.fromkeys(hist_idx))
        return hist_idx if len(hist_idx) == self.n_pre_imgs else None

    # def generate_spatio_temporal_sample_data(self):
    #     df_date_count_by_burst = (
    #         self.df_rtc_meta.groupby("jpl_burst_id")
    #         .size()
    #         .reset_index(drop=False)
    #         .rename(columns={0: "acq_per_burst"})
    #     )
    #     df_patch_count_by_burst = (
    #         self.df_patch.groupby("jpl_burst_id")
    #         .size()
    #         .reset_index(drop=False)
    #         .rename(columns={0: "patches_per_burst"})
    #     )

    #     df_count_rtc = pd.merge(
    #         df_date_count_by_burst,
    #         df_patch_count_by_burst,
    #         on="jpl_burst_id",
    #         how="inner",
    #     )
    #     df_count_rtc["total_samples_per_burst"] = (
    #         df_count_rtc["acq_per_burst"] * df_count_rtc["patches_per_burst"]
    #     )
    #     df_count_rtc["total_samples_cumul"] = (
    #         df_count_rtc.total_samples_per_burst.cumsum()
    #     )

    #     self.df_st_sample = df_count_rtc
    #     self.burst_ids = self.df_st_sample.jpl_burst_id.unique().tolist()
    #     self.df_rtc_meta = self.df_rtc_meta[
    #         self.df_rtc_meta.jpl_burst_id.isin(self.burst_ids)
    #     ].reset_index(drop=True)

    #     self.df_patch = None
    #     self.df_rtc_meta = None


    def generate_spatio_temporal_sample_data(self):
        """Pre-compute valid post-images *and* their 10-scene histories."""
        post2hist = {}      # (burst_id, post_row_idx) ➜ hist_idx list
        rows       = []     # summary rows for df_st_sample

        for bid, df_burst in self.df_rtc_meta.groupby("jpl_burst_id"):
            df_burst = df_burst.sort_values("acq_datetime").reset_index(drop=True)
            if df_burst.acq_datetime.dtype == object:
                df_burst["acq_datetime"] = pd.to_datetime(df_burst.acq_datetime)

            valid_posts = []
            for i, ts in enumerate(df_burst.acq_datetime):
                hist = self._collect_history(ts, df_burst)
                if hist is not None:              # keep only the good ones
                    valid_posts.append(i)
                    post2hist[(bid, i)] = hist

            if not valid_posts:                   # skip bursts with no history
                continue

            patches = (self.df_patch
                    .query("jpl_burst_id == @bid")
                    .shape[0])
            rows.append({
                "jpl_burst_id":          bid,
                "valid_posts_per_burst": len(valid_posts),
                "patches_per_burst":     patches,
                "total_samples_per_burst":
                    len(valid_posts) * patches
            })

        df = pd.DataFrame(rows)
        df["total_samples_cumul"] = df.total_samples_per_burst.cumsum()

        self.df_st_sample = df.reset_index(drop=True)
        self.post2hist    = post2hist    # (burst, post_row_idx) → list[hist_idx]

    def generate_spatio_temporal_sample_data_three_old(self):
        """Pre-compute, for every burst, which acquisitions have full history."""
        valid_posts = {}          # burst_id → list[row indices in df_ts]
        rows = []                 # records for df_st_sample

        for burst_id, df_burst in self.df_rtc_meta.groupby("jpl_burst_id"):
            df_burst = df_burst.sort_values("acq_datetime").reset_index(drop=True)
            if df_burst.acq_datetime.dtype == object:
                df_burst["acq_datetime"] = pd.to_datetime(df_burst.acq_datetime)

            good = [i for i, d in enumerate(df_burst.acq_datetime)
                    if self._has_full_history(d, df_burst)]
            if not good:                  # skip burst that has no valid post-images
                continue

            valid_posts[burst_id] = good
            patches = (self.df_patch
                    .query("jpl_burst_id == @burst_id")
                    .shape[0])
            rows.append({
                "jpl_burst_id":          burst_id,
                "valid_posts_per_burst": len(good),
                "patches_per_burst":     patches,
                "total_samples_per_burst":
                    len(good) * patches
            })

        df = pd.DataFrame(rows)
        df["total_samples_cumul"] = df.total_samples_per_burst.cumsum()

        df = df[(df.patches_per_burst > 0) & (df.valid_posts_per_burst > 0)]
        df["total_samples_per_burst"] = df.valid_posts_per_burst * df.patches_per_burst
        df["total_samples_cumul"]     = df.total_samples_per_burst.cumsum()
        
        self.df_st_sample = df.reset_index(drop=True)
        self.df_st_sample = df.reset_index(drop=True)
        self.valid_posts   = valid_posts           # store for later use
    # def generate_spatio_temporal_sample_data(self):
    #     """
    #     Build self.df_st_sample with, for every burst:
    #     - acq_per_burst
    #     - patches_per_burst
    #     - valid_posts_per_burst
    #     - total_samples_per_burst
    #     """
    #     # 1. basic counts
    #     df_acq = (self.df_rtc_meta.groupby("jpl_burst_id")
    #             .size().reset_index(name="acq_per_burst"))
    #     df_patch = (self.df_patch.groupby("jpl_burst_id")
    #                 .size().reset_index(name="patches_per_burst"))

    #     df = pd.merge(df_acq, df_patch, on="jpl_burst_id", how="inner")

    #     # 2. which acquisitions have the full 3-year history?
    #     valid_posts = {}
    #     v_posts_col = []
    #     for bid in df.jpl_burst_id:
    #         ts = (self.df_rtc_meta
    #                 .query("jpl_burst_id == @bid")
    #                 .sort_values("acq_datetime")
    #                 .reset_index(drop=True))
    #         if ts.acq_datetime.dtype == object:
    #             ts["acq_datetime"] = pd.to_datetime(ts.acq_datetime)

    #         good = [i for i, t in enumerate(ts.acq_datetime)
    #                 if self._has_full_history(t, ts)]
    #         valid_posts[bid] = good
    #         v_posts_col.append(len(good))

    #     df["valid_posts_per_burst"] = v_posts_col
    #     df["total_samples_per_burst"] = (
    #         df.valid_posts_per_burst * df.patches_per_burst
    #     )
    #     df = df[df.total_samples_per_burst > 0]           # keep useful bursts
    #     df["total_samples_cumul"] = df.total_samples_per_burst.cumsum()

    #     self.df_st_sample = df.reset_index(drop=True)
    #     self.valid_posts = valid_posts

    def download_rtc_data(self):
        def localize_one_rtc_p(*urls):
            return localize_one_rtc(urls, ts_dir=self.root)

        input_data = [
            [vv_path, vh_path]
            for (vv_path, vh_path) in zip(
                self.df_rtc_meta.rtc_s1_vv_url.tolist(),
                self.df_rtc_meta.rtc_s1_vh_url.tolist(),
            )
        ]
        with WorkerPool(n_jobs=self.n_workers_for_download, use_dill=True) as pool:
            dual_loc_path = pool.map(
                localize_one_rtc_p,
                input_data,
                progress_bar=False,
                progress_bar_style="std",
                concatenate_numpy_output=False,
            )

        self.df_rtc_meta["rtc_s1_stack_loc_path"] = [str(p) for p in dual_loc_path]

        self.patch_by_burst_data_dir.mkdir(exist_ok=True, parents=True)
        burst_ids = self.df_rtc_meta.jpl_burst_id.unique().tolist()
        for burst_id in tqdm(burst_ids, desc="localize patch tables"):
            patch_out_path = self.patch_by_burst_data_dir / f"patch_{burst_id}_{self.patch_size}.parquet"
            if patch_out_path.exists():
                print('okay')
                # continue
            else:
                df_burst_patches = self.df_patch[
                    self.df_patch.jpl_burst_id == burst_id
                ].reset_index(drop=True)
                df_burst_patches.to_parquet(
                    patch_out_path, compression="snappy", engine="pyarrow"
                )

            rtc_out_path = self.rtc_by_burst_data_dir / f"rtc_{burst_id}.parquet"
            if rtc_out_path.exists():
                continue
            else:
                df_rtc_ts = self.df_rtc_meta[
                    self.df_rtc_meta.jpl_burst_id == burst_id
                ].reset_index(drop=True)
                df_rtc_ts.to_parquet(
                    rtc_out_path, compression="snappy", engine="pyarrow"
                )

    def __len__(self):
        return self.df_st_sample.total_samples_per_burst.sum()

    def __getitem__(self, idx: int):
        """
        Parameters
        ----------
        idx : int
            Global sample index (0 … len(self)-1).

        Returns
        -------
        dict
            {
                "pre_imgs" : torch.Tensor  [n_pre_imgs, 2, H, W]   # here n_pre_imgs = 10
                "post_img" : torch.Tensor  [2, H, W]
            }

        Sampling rule
        -------------
        * post-image: acquisition chosen by `_index_logic` (slides through the burst)
        * pre-images (10 total):
            • last-4 scenes within (post-3 yr − 48 d , post-3 yr]
            • last-3 scenes within (post-2 yr − 48 d , post-2 yr]
            • last-3 scenes within (post-1 yr − 48 d , post-1 yr]
        """

        # ------------------------------------------------------------------
        # 1.  translate global idx ➜ burst, patch, time-series indices
        # ------------------------------------------------------------------
        burst_id, df_ts, ts_indices, patch_idx = self._index_logic(idx)
        img_paths = df_ts.loc[ts_indices, "rtc_s1_stack_loc_path"].tolist()
        print(img_paths)
        # ------------------------------------------------------------------
        # 2.  locate the spatial window for this patch
        # ------------------------------------------------------------------
        patch_path = (self.patch_by_burst_data_dir /
                    f"patch_{burst_id}_{self.patch_size}.parquet")
        patch_table = pd.read_parquet(patch_path)
        patch_info = patch_table.iloc[patch_idx].to_dict()
        def read_window_p(url: str):
            X, _ = read_raster_from_slice(
                url,
                patch_info["x_start"],
                patch_info["y_start"],
                patch_info["x_start"] + self.patch_size,
                patch_info["y_start"] + self.patch_size,
            )
            return X

        # ------------------------------------------------------------------
        # 3.  load the 11 images in parallel (10 pre + 1 post)
        # ------------------------------------------------------------------
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            arrs = list(pool.map(read_window_p, img_paths))
        pre_imgs = np.stack(arrs[:-1], axis=0)   # [10, 2, H, W]
        post_img = arrs[-1]                      # [2, H, W]
        return {
            "pre_imgs": torch.from_numpy(pre_imgs),
            "post_img": torch.from_numpy(post_img),
        }

    def _index_logic(self, idx):
        # ----- which burst? -----
        bidx      = bisect.bisect_left(self.df_st_sample.total_samples_cumul, idx)
        burst_row = self.df_st_sample.iloc[bidx]
        bid       = burst_row.jpl_burst_id

        # ----- which patch? -----
        patches_for_burst = burst_row.patches_per_burst
        local_idx = idx - (self.df_st_sample.iloc[bidx-1].total_samples_cumul
                        if bidx else 0)
        patch_idx = local_idx % patches_for_burst

        # ----- which post-image row? -----
        valid_rows   = [key[1] for key in self.post2hist.keys() if key[0] == bid]
        post_row_idx = valid_rows[(local_idx // patches_for_burst) % len(valid_rows)]

        rtc_ts_path = self.rtc_by_burst_data_dir / f"rtc_{bid}.parquet"
        df_ts = (pd.read_parquet(rtc_ts_path)
                .sort_values("acq_datetime")
                .reset_index(drop=True))

        hist_idx  = self.post2hist[(bid, post_row_idx)]
        ts_indices = hist_idx + [post_row_idx]          # 10 pre + 1 post

        return bid, df_ts, ts_indices, patch_idx
        
    # def _index_logic_old(self, idx):
    #     # ---------- which burst? ----------
    #     burst_idx = bisect.bisect_left(
    #         self.df_st_sample.total_samples_cumul.tolist(), idx
    #     )
    #     burst_row = self.df_st_sample.iloc[burst_idx]
    #     burst_id  = burst_row.jpl_burst_id

    #     # ---------- which patch? ----------
    #     patches_for_burst = burst_row.patches_per_burst
    #     local_idx = idx - (self.df_st_sample.iloc[burst_idx - 1].total_samples_cumul
    #                     if burst_idx > 0 else 0)
    #     patch_idx = local_idx % patches_for_burst

    #     # ---------- which post-image row? ----------
    #     valid_rows = self.valid_posts[burst_id]          # list of df_ts indices
    #     post_row_idx = valid_rows[(local_idx // patches_for_burst) % len(valid_rows)]

    #     # ---------- load df_ts & derive history ----------
    #     rtc_ts_path = self.rtc_by_burst_data_dir / f"rtc_{burst_id}.parquet"
    #     df_ts = (pd.read_parquet(rtc_ts_path)
    #             .sort_values("acq_datetime")
    #             .reset_index(drop=True))
    #     if df_ts.acq_datetime.dtype == object:
    #         df_ts["acq_datetime"] = pd.to_datetime(df_ts.acq_datetime)

    #     post_row  = df_ts.iloc[post_row_idx]
    #     post_date = post_row.acq_datetime

    #     hist_idx = []
    #     for yrs, n_pick in zip(self.year_offsets, self.samples_each):
    #         hi = post_date - relativedelta(years=yrs)
    #         lo = hi - pd.Timedelta(days=self.window_days)
    #         window = df_ts[(df_ts.acq_datetime > lo) & (df_ts.acq_datetime < hi)]
    #         hist_idx.extend(window.tail(n_pick).index.tolist())

    #     hist_idx = list(dict.fromkeys(hist_idx))
    #     if len(hist_idx) != self.n_pre_imgs:
    #         raise RuntimeError("history length drifted; should never happen")
    #     # ts_indices.append(post_row_idx)
    #     ts_indices = sorted(hist_idx, key=lambda i: df_ts.at[i, "acq_datetime"])
    #     ts_indices.append(post_row_idx)

    #     return burst_id, df_ts, ts_indices, patch_idx
    # # ------------------------------------------------------------------

    # 2️⃣  Tiny convenience wrapper  ----------------------------------
    def get_acq_dates(self, idx):
        """Return the list of 10 `Timestamp`s for quick sanity checks."""
        _, df_ts, ts_indices, _ = self._index_logic(idx)
        # print(df_ts)
        return df_ts.loc[ts_indices, "acq_datetime"].tolist()
    # ------------

    def _has_full_history(self, post_date, df_ts):
        """True if df_ts contains enough scenes in every offset window."""
        wanted = 0
        for yrs, n_pick in zip(self.year_offsets, self.samples_each):
            hi = post_date - relativedelta(years=yrs)
            lo = hi - pd.Timedelta(days=self.window_days)
            mask = (df_ts.acq_datetime > lo) & (df_ts.acq_datetime < hi)
            # count = ((df_ts.acq_datetime > lo) & (df_ts.acq_datetime <= hi)).sum()
            count = mask.sum()
            wanted += n_pick
            if count < n_pick:
                return False
        return True