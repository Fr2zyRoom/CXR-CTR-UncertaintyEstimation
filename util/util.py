FILE_EXTENSION = ['.img', '.IMG', '.jpg', '.JPG', '.png', '.PNG', 'dcm', 'DCM', '.csv', '.CSV']
DCM_EXTENSION = ['.dcm', '.DCM']
IMG_EXTENSION = ['.img', '.IMG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.gif', '.GIF']
PNG_EXTENSION = ['.png', '.PNG', '.jpg', '.JPG']
EXTENSION_NAME_TO_LS = {'png':PNG_EXTENSION,
                        'enhanced_png':PNG_EXTENSION,
                        'dcm':DCM_EXTENSION}
import os
import numpy as np
import pandas as pd
from PIL import Image
import zipfile

def check_extension(filename, extension_ls=FILE_EXTENSION):
    return any(filename.endswith(extension) for extension in extension_ls)


def load_file_path(folder_path, extension_ls=FILE_EXTENSION, all_sub_folders=False):
    """find 'IMG_EXTENSION' file paths in folder.
    
    Parameters:
        folder_path (str) -- folder directory
        extension_ls (list) -- list of extensions
        all_sub_folders(bool) -- check all sub directories
    
    Return:
        file_paths (list) -- list of 'extension_ls' file paths
    """
    
    file_paths = []
    assert os.path.isdir(folder_path), f'{folder_path} is not a valid directory'

    for root, _, fnames in sorted(os.walk(folder_path)):
        for fname in fnames:
            if check_extension(fname, extension_ls):
                path = os.path.join(root, fname)
                file_paths.append(path)
        if not all_sub_folders:
            break

    return sorted(file_paths[:])


def gen_fname_path_dict(paths):
    """create path dictionary
    Parameters:
        paths (str list) -- a list of paths
        
    Return:
        fname_path_dict (dict) -- dictionary key-fname /value-path
    """
    
    if isinstance(paths, list) and not isinstance(paths, str):
        fname_path_dict = {os.path.splitext(os.path.basename(p))[0]:p 
                           for p in paths}
    else:
        fname_path_dict = {os.path.splitext(os.path.basename(paths))[0]:paths}
        
    return fname_path_dict


def gen_new_dir(new_dir):
    """make new directory
    """
    try: 
        if not os.path.exists(new_dir): 
            os.makedirs(new_dir) 
            #print(f"New directory!: {new_dir}")
    except OSError: 
        print("Error: Failed to create the directory.")   


def normalize(arr, bit8=False):
    """min-max normalization
    Parameters:
        arr (np.array) -- image array
        bit8 (bool) -- uint8(0-255) or float(0-1)
    
    Return:
        arr_norm (np.array) -- normalized image array
    """
    arr_norm = arr - np.min(arr)
    if np.max(arr_norm) != 0:
        arr_norm /= np.max(arr_norm)
    if bit8 == True:
        arr_norm = np.array(arr_norm*255).astype(np.uint8)
    return arr_norm


def save_arr_to_png(arr, save_dir, fname):
    """save np.array to png file
    Parameters:
        arr (np.array) -- image array
        save_dir (str) -- directory to save array
        fname (str) -- file name
    """
    save_path = os.path.join(save_dir, fname+'.png')
    Image.fromarray(arr.astype(np.uint8)).save(save_path)


def unzip(zip_file_path, extract_dir):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)