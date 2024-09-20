#C3: Aggregation Strategy

# Patch Level
# Threshold
# Image Level
import numpy as np
import matplotlib.patches as patches
from scipy.signal import convolve


def patch_level_aggregation(image, patch_size, mean=False, **kwargs):
    if type(patch_size) == int:
        patch_size = len(image.shape) * [patch_size]
    kernel = np.ones(patch_size)
    patch_aggragated = convolve(image, kernel, mode="valid")
    if mean:
        patch_aggragated = patch_aggragated / (np.prod(patch_size))
    all_max_indices = np.where(np.isclose(patch_aggragated, np.max(patch_aggragated)))
    max_indices = []
    for indices in all_max_indices:
        max_indices.append(indices[0])

    max_indices_slice = []
    for idx, index in enumerate(max_indices):
        max_indices_slice.append((int(index), int(index + patch_size[idx])))
    return {
        "max_score": float(np.max(patch_aggragated)),
        "bounding_box": max_indices_slice,
    }


def image_level_aggregation(image, mean=False, **kwargs):
    if mean:
        return float(np.sum(image) / image.size)
    return {"max_score": float(np.sum(image))}