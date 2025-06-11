import numpy as np
from dem_stitcher.rio_tools import reproject_arr_to_match_profile
from tqdm import tqdm


def resample_multiclass_arr_to_percent(
    label_arr: np.ndarray, src_profile: dict, dst_profile: dict, nodata_val: int = 0
) -> tuple[np.ndarray, dict]:
    dst_aff = dst_profile["transform"]
    src_aff = src_profile["transform"]

    x_ratio = dst_aff.a / src_aff.a
    y_ratio = dst_aff.e / dst_aff.e

    if (x_ratio < 1) or (y_ratio < 1):
        raise ValueError(
            "Only works with down sampling i.e. lowering spatial resolution"
        )

    unique_labels = sorted(np.unique(label_arr[label_arr != nodata_val]))
    n_labels = len(unique_labels)
    ind2label = {i: l for (i, l) in enumerate(unique_labels)}

    percent_mat = np.full(
        (n_labels, dst_profile["height"], dst_profile["width"]), nodata_val
    )
    p_perc = src_profile.copy()
    p_perc["dtype"] = "float32"

    for ind, label in tqdm(ind2label.items()):
        bin_arr = (label_arr == label).astype(np.float32)
        bin_arr_r, _ = reproject_arr_to_match_profile(
            bin_arr, p_perc, dst_profile, resampling="average"
        )
        bin_arr_r = bin_arr_r[0, ...]
        percent_mat[ind, ...] = bin_arr_r
    return percent_mat


def resample_multiclass_arr_to_majority_label(
    label_arr: np.ndarray, src_profile: dict, dst_profile: dict, nodata_val: int = 0
) -> np.ndarray:
    unique_labels = sorted(np.unique(label_arr[label_arr != nodata_val]))
    ind2label = {i: label for (i, label) in enumerate(unique_labels)}

    percent_mat = resample_multiclass_arr_to_percent(
        label_arr, src_profile, dst_profile
    )
    max_label_ind = np.argmax(percent_mat, axis=0)

    label_arr_r = np.full((dst_profile["height"], dst_profile["width"]), nodata_val)
    for ind, label in ind2label.items():
        label_arr_r[max_label_ind == ind] = label
    return label_arr_r
