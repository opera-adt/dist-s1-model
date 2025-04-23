import random
from functools import lru_cache
from pathlib import Path

import backoff
import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.profiles import DefaultGTiffProfile
from rasterio.windows import Window
from requests.exceptions import HTTPError
from skimage.restoration import denoise_tv_bregman
from torch.utils.data import Dataset
from tqdm import tqdm
from mpire import WorkerPool
import multiprocessing as mp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    return pd.read_parquet("rtc_data_subset.parquet")


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
        df_rtc_meta: str = None,
        patch_data_path: str = None,
        n_pre_imgs: int = 4,
        root: Path | str = Path("opera_rtc_data/chips/"),
        n_workers_for_download: int = 10,
    ):
        self.root = root
        # Hard-coded because that's what is in the patch table
        self.patch_size = 224

        target_cpus = min(mp.cpu_count(), n_workers_for_download)
        self.n_workers_for_download = target_cpus

        self.df_rtc_meta = df_rtc_meta if df_rtc_meta is not None else open_rtc_table()
        self.burst_ids = self.df_rtc_meta.jpl_burst_id.unique().tolist()

        self.n_pre_imgs = n_pre_imgs

        self.patch_table_path = patch_data_path or Path("burst_patch_data.pqt")
        self.rtc_geotiff_dir = Path(self.root)  # / 'geotiff'
        self.rtc_geotiff_dir.mkdir(exist_ok=True, parents=True)

        self.rtc_patch_dir = Path(self.root) / "patch_time_series"
        self.rtc_patch_dir.mkdir(exist_ok=True, parents=True)

        # If data exists, stores the path
        self.download_rtc_data()

        self.spatio_temporal_path = Path("spatial_temporal_dataset.parquet")

    def download_rtc_data(self):
        def fetch_slices_for_burst(jpl_burst_id: str) -> pd.DataFrame:
            df_burst_patches = pd.read_parquet(
                self.patch_table_path, filters=[("jpl_burst_id", "=", jpl_burst_id)]
            )
            return df_burst_patches

        def localize_one_rtc_p(url):
            return localize_one_rtc(url, ts_dir=self.rtc_geotiff_dir)

        def localize_numpy_data_to_torch_tensor(X: np.array, dst_path: Path) -> Path:
            if dst_path.exists():
                return dst_path
            X_pt = torch.from_numpy(X.astype(np.float32))
            torch.save(X_pt, dst_path)
            return dst_path

        def read_raster_from_slice_p(
            url: str, x_start: int, y_start: int, x_stop: int, y_stop: int
        ) -> np.ndarray:
            X, _ = read_raster_from_slice(url, x_start, y_start, x_stop, y_stop)
            return X[0, ...]

        with WorkerPool(n_jobs=self.n_workers_for_download, use_dill=True) as pool:
            vv_loc_paths = pool.map(
                localize_one_rtc_p,
                self.df_rtc_meta.rtc_s1_vv_url.tolist(),
                progress_bar=True,
                progress_bar_style="std",
                concatenate_numpy_output=False,
            )

        with WorkerPool(n_jobs=self.n_workers_for_download, use_dill=True) as pool:
            vh_loc_paths = pool.map(
                localize_one_rtc_p,
                self.df_rtc_meta.rtc_s1_vh_url.tolist(),
                progress_bar=True,
                progress_bar_style="std",
                concatenate_numpy_output=False,
            )
        self.df_rtc_meta["rtc_s1_vv_loc_path"] = [str(p) for p in vv_loc_paths]
        self.df_rtc_meta["rtc_s1_vh_loc_path"] = [str(p) for p in vh_loc_paths]

        self.dataset_tensor_paths = []
        for burst_id, df_burst_ts in tqdm(
            self.df_rtc_meta.groupby("jpl_burst_id"), desc="burst_id"
        ):
            df_patches_for_burst = fetch_slices_for_burst(burst_id)
            n_acqs = df_burst_ts.shape[0]
            rtc_ids = df_burst_ts.rtc_s1_id.tolist()
            x_start_l, y_start_l = (
                df_patches_for_burst.x_start.tolist(),
                df_patches_for_burst.y_start.tolist(),
            )
            for patch_idx, (x_start, y_start) in enumerate(
                zip(tqdm(x_start_l, desc="patches"), y_start_l)
            ):
                # Open arrays and stack into dual polarization img
                vv_arrs = [
                    read_raster_from_slice_p(
                        vv_path,
                        x_start,
                        y_start,
                        x_start + self.patch_size,
                        y_start + self.patch_size,
                    )
                    for vv_path in df_burst_ts.rtc_s1_vv_loc_path
                ]
                vh_arrs = [
                    read_raster_from_slice_p(
                        vh_path,
                        x_start,
                        y_start,
                        x_start + self.patch_size,
                        y_start + self.patch_size,
                    )
                    for vh_path in df_burst_ts.rtc_s1_vh_loc_path
                ]
                dual_pol_data = [
                    np.stack([vv, vh], axis=0) for (vv, vh) in zip(vv_arrs, vh_arrs)
                ]

                # Organize temporal dim
                time_slices = [
                    (idx_start, idx_start + self.n_pre_imgs + 1)
                    for idx_start in range(n_acqs - self.n_pre_imgs - 1)
                ]
                tensor_paths = [
                    self.rtc_patch_dir
                    / f"start{rtc_ids[idx_start]}__patch-id-{patch_idx}.pt"
                    for idx_start, _ in time_slices
                ]
                # Form into T x 2 x H x W imagery
                np_data = [
                    np.stack(dual_pol_data[idx_start:idx_stop], axis=0)
                    for (idx_start, idx_stop) in time_slices
                ]
                _ = [
                    localize_numpy_data_to_torch_tensor(X, pt_path)
                    for (pt_path, X) in zip(tensor_paths, np_data)
                ]
                self.dataset_tensor_paths.extend(tensor_paths)

    def __len__(self):
        return self.dataset_tensor_paths

    def __getitem__(self, idx):
        sample = torch.load(self.dataset_tensor_paths[idx])

        return {
            "pre_imgs": sample[:-1, ...],
            "post_img": sample[-1, ...],
        }


###################################
# Stream Patches
###################################


class SeqDistDatasetStreamPatches(Dataset):
    def __init__(
        self,
        df_rtc_meta: str = None,
        patch_data_path: str = None,
        n_pre_imgs: int = 4,
        root: Path | str = Path("opera_rtc_data"),
        n_workers_for_download: int = 20,
    ):
        self.root = root
        self.patch_size = 224

        self.df_rtc_meta = df_rtc_meta if df_rtc_meta is not None else open_rtc_table()
        self.burst_ids = self.df_rtc_meta.jpl_burst_id.unique().tolist()
        self.patch_data_path = patch_data_path or Path("burst_patch_data.pqt")
        self.n_pre_imgs = n_pre_imgs
        self.df_patch = pd.read_parquet(self.patch_data_path).set_index("jpl_burst_id")

        # Requested CPU count or available CPUs, whichever smaller
        target_cpus = min(mp.cpu_count(), n_workers_for_download)
        self.n_workers_for_download = target_cpus

        self.patch_data_dir = Path("patch_data_dir")

        self.download_rtc_data()

    def download_rtc_data(self):
        # n = self.df_rtc_meta.shape[0]

        def localize_one_rtc_p(*urls):
            return localize_one_rtc(urls, ts_dir=self.root)

        input_data = [[vv_path, vh_path] for (vv_path, vh_path) in zip(self.df_rtc_meta.rtc_s1_vv_url.tolist(), self.df_rtc_meta.rtc_s1_vh_url.tolist())]
        with WorkerPool(n_jobs=self.n_workers_for_download, use_dill=True) as pool:
            dual_loc_path = pool.map(
                localize_one_rtc_p,
                input_data,
                progress_bar=True,
                progress_bar_style="std",
                concatenate_numpy_output=False,
            )

        self.df_rtc_meta["rtc_s1_stack_loc_path"] = [str(p) for p in dual_loc_path]

        self.patch_data_dir.mkdir(exist_ok=True, parents=True)
        for burst_id in tqdm(self.burst_ids, desc="localize patch tables"):
            out_path = self.patch_data_dir / f"{burst_id}.parquet"
            if out_path.exists():
                continue
            else:
                df_burst_patches = pd.read_parquet(
                    self.patch_data_path, filters=[("jpl_burst_id", "=", burst_id)]
                ).reset_index(drop=True)
                df_burst_patches.to_parquet(
                    out_path, compression="snappy", engine="pyarrow"
                )

    def __len__(self):
        return len(self.burst_ids)

    def __getitem__(self, idx):
        burst_id = self.burst_ids[idx]

        df_ts_full = self.df_rtc_meta[
            self.df_rtc_meta.jpl_burst_id == burst_id
        ].reset_index(drop=True)
        df_burst_patches = pd.read_parquet(self.patch_data_dir / f"{burst_id}.parquet")

        N = df_ts_full.shape[0] - self.n_pre_imgs - 1
        # randint includes both endpoints
        i = random.randint(0, N - 1)
        df_ts = df_ts_full.iloc[i : i + self.n_pre_imgs + 1].reset_index(drop=True)

        M = df_burst_patches.shape[0]
        # randint includes both endpoints
        j = random.randint(0, M - 1)
        patch_data = df_burst_patches.iloc[j].to_dict()

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

        arrs, _ = zip(*[read_window_p(p) for p in img_paths])

        # Input for modeling
        # pre img is n_pre_imgs X 2 X H X W
        pre_imgs = np.stack(arrs[: self.n_pre_imgs], axis=0)
        # post img is 2 X H X W
        post_img = arrs[-1]

        # additional metadata
        return {
            "pre_imgs": torch.from_numpy(pre_imgs),
            "post_img": torch.from_numpy(post_img),
        }
