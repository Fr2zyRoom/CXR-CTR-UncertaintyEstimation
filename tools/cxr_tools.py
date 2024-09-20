import numpy as np
from skimage.measure import label, regionprops

from tools.segmentation_tools import *

def cal_ct_ratio(heart_mask,lung_mask):
    """
    Calculate Cardiothoracic ratio from heart mask and lung mask
    
    Parameters:
        heart_mask (np.array) -- heart mask
        lung_mask (np.array) -- lung mask(left+right)
    
    Return:
        cardiac_size, thoracic_size, ct_ratio (int, int, float) -- cardiac_size, thoracic_size are pixel-wise size
    
    """
    heart_islands = label(heart_mask)
    
    # denoise
    lbl_heart_mask = split_island(heart_mask)
    heart_largest_island = np.where(np.where(lbl_heart_mask>1, 0, lbl_heart_mask)>0, 1, 0)
    
    heart_horizontal_indicies = np.where(np.any(heart_largest_island, axis=0))[0]
    if len(heart_horizontal_indicies)>1:
        cardiac_size = heart_horizontal_indicies[-1] - heart_horizontal_indicies[0]
    else:
        cardiac_size = 0
    
    # denoise
    lbl_lung_mask = split_island(lung_mask)
    filtered_mask = np.where(np.where(lbl_lung_mask>2, 0, lbl_lung_mask)>0, 1, 0)
    
    left_lung, right_lung = split_lung(filtered_mask)
    lung_corrected = np.logical_or(left_lung, right_lung).astype(np.uint8)
    lung_horizontal_indicies = np.where(np.any(lung_corrected, axis=0))[0]
    thoracic_size = lung_horizontal_indicies[-1] - lung_horizontal_indicies[0]
    
    ct_ratio = cardiac_size/thoracic_size
    if ct_ratio > 1:
        ct_ratio = 1
    return cardiac_size, thoracic_size, ct_ratio
