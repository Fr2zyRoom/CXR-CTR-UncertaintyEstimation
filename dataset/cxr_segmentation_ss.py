import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

import numpy as np
import pandas as pd
import cv2
from PIL import Image
from skimage import exposure
from util.util import *
from tools.img_tools import *

def match_data_and_mask(img_path_ls, 
                        mask_name_path_dict, 
                        dataset_fname_ls=None):
    """match img(img_path_ls) and masks(mask_path_ls)
    Parameters:
        img_path_ls (list) -- the list of data paths
        mask_path_dict (dict) -- mask name:the list of mask paths
        dataset_fname_ls (list) -- the list of selected data file names
        
    Return:
        data_label_ls (list) -- a list of data_path and labels(matched)
    """
    img_path_dict = gen_fname_path_dict(img_path_ls)
    mask_path_dict = {mask_name:gen_fname_path_dict(mask_path_ls) for mask_name, mask_path_ls in mask_name_path_dict.items()}
    
    if dataset_fname_ls is None:
        img_mask_ls = [[img_path_dict[dataset_fname], [mask_path_dict[mask_name][dataset_fname] for mask_name in mask_path_dict.keys()]] 
                       for dataset_fname in list(img_path_dict.keys())]
    else:
        img_mask_ls = [[img_path_dict[dataset_fname], [mask_path_dict[mask_name][dataset_fname] for mask_name in mask_path_dict.keys()]] 
                       for dataset_fname in dataset_fname_ls]
    return img_mask_ls


def gen_cxr_seg_dataset_ls(data_dir, 
                           mask_name_ls,
                           csv_path=None, 
                           extension_ls=FILE_EXTENSION, 
                           fname_col='FilePath', 
                           split=None):
    """match data(data_dir) and masks and split dataset from .csv file
    Parameters:
        data_path_ls (list) -- the list of data paths
        label_csv_path (str) -- a path of csv file(for split)
        fname_col (str) -- a filename(data) column in the csv file 
        split (str) -- 'train' / 'val' / 'test'
        
    Return:
        img_mask_ls (list) -- a list of data_path and mask_path(matched)
    """
    
    if csv_path is None:
        dataset_ls = None
    else:
        split_df = pd.read_csv(csv_path)
        if split is None:
            dataset_ls = split_df[fname_col].values
            mask_ls = split_df[mask_name_ls].values
        else:
            dataset_ls = split_df[split_df.split==split][fname_col].values
            mask_ls = split_df[split_df.split==split][mask_name_ls].values
    
    img_mask_ls = list(zip(dataset_ls, mask_ls))
    
    return img_mask_ls


def img_loader(img_path):
    return np.array(Image.open(img_path).convert('L'))


def img_loader_cosine_gamma_correction(img_path):
    img = np.array(Image.open(img_path).convert('L'))
    return cosine_gamma_correction(img)


def img_loader_gamma_correction(img_path):
    img = np.array(Image.open(img_path).convert('L'))
    return gamma_correction(img)


def img_loader_clahe(img_path):
    img = np.array(Image.open(img_path).convert('L'))
    return clahe(img)


def img_loader_gaussian_blur(img_path):
    img = np.array(Image.open(img_path).convert('L'))
    return gaussian_blur(img, sigma=5)


def img_rgb(img):
    return np.stack([img]*3, axis=-1)


def img_grayscale(img):
    return np.expand_dims(img, -1)


def mask_loader(mask_path):
    return (np.array(Image.open(mask_path).convert('L'))>0).astype(np.uint8)


def get_transform(params=None, 
                  mode=None, 
                  resize_factor=None, 
                  gray_scale=False, 
                  convert=True):
    transform_list = []
    ## resize
    transform_list.append(
        A.Resize(resize_factor, resize_factor)
    )
    if mode == 'train':
        ## brightness or contrast
        transform_list.append(
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, 
                                           contrast_limit=0.2),
                A.RandomGamma(p=1),
                #for chest x-ray
                A.CLAHE(p=1),
                A.Equalize(p=1)
            ], p=.3)
        )
        ## blur or sharpen
        transform_list.append(
            A.OneOf([
                A.Blur(blur_limit=3, p=1),
                A.MedianBlur(blur_limit=3, p=1)
            ], p=.2)
        )
        transform_list.append(
            A.OneOf([
                A.GaussNoise(0.002, p=.5),
            ], p=.2)
        )
        transform_list.append(
            A.HorizontalFlip(p=.5)
        )
        transform_list.append(
            A.ShiftScaleRotate(shift_limit=0.03, 
                               scale_limit=0.01, 
                               rotate_limit=10, 
                               p=0.4, 
                               border_mode = cv2.BORDER_CONSTANT)
        )
    ## normalize
    if convert:
        if gray_scale:
            transform_list.append(
                A.Normalize(mean=[0.5,],
                            std=[0.5,],),
            )
        else:
            transform_list.append(
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],),

            )
        transform_list.append(ToTensorV2())
    
    return A.Compose(transform_list)


def get_tta_transform(params=None, 
                      mode=None, 
                      resize_factor=None, 
                      gray_scale=False, 
                      convert=True):
    transform_list = []
    ## resize
    transform_list.append(
        A.Resize(resize_factor, resize_factor)
    )
    if mode == 'train':
        ## brightness or contrast
        transform_list.append(
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, 
                                           contrast_limit=0.2),
                A.RandomGamma(p=1),
                #for chest x-ray
                A.CLAHE(p=1),
                A.Equalize(p=1)
            ], p=.3)
        )
        ## blur or sharpen
        transform_list.append(
            A.OneOf([
                A.Blur(blur_limit=3, p=1),
                A.GaussianBlur(blur_limit=0, sigma_limit=6, p=1),
                A.MedianBlur(blur_limit=3, p=1)
            ], p=.3)
        )
        # transform_list.append(
        #     A.OneOf([
        #         # A.Blur(blur_limit=3, p=1),
        #         A.Blur(blur_limit=15, p=1),
        #         A.GaussianBlur(blur_limit=0, sigma_limit=6, p=1),
        #         A.MedianBlur(blur_limit=3, p=1)
        #     ], p=.2)
        # )
        transform_list.append(
            A.OneOf([
                A.GaussNoise(0.002, p=1),
            ], p=.5)
        )
        transform_list.append(
            A.HorizontalFlip(p=.5)
        )
        transform_list.append(
            A.RandomRotate90(p=.5)
        )
        transform_list.append(
            A.ShiftScaleRotate(shift_limit=0.03, 
                               scale_limit=0.01, 
                               rotate_limit=10, 
                               p=0.4, 
                               border_mode = cv2.BORDER_CONSTANT)
        )
    ## normalize
    if convert:
        if gray_scale:
            transform_list.append(
                A.Normalize(mean=[0.5,],
                            std=[0.5,],),
            )
        else:
            transform_list.append(
                A.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5],),

            )
        transform_list.append(ToTensorV2())
    
    return A.Compose(transform_list)


class CXRSegDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 data_dir,
                 mask_name_ls,
                 csv_path,
                 img_loader=img_loader, 
                 mask_loader=mask_loader,
                 resize_factor=512,
                 gray_scale=False,
                 background=False,
                 transform=get_transform, 
                 mode=None, 
                 convert=True):
        self.data_dir = data_dir
        self.mask_name_ls = mask_name_ls
        self.csv_path = csv_path
        self.img_loader = img_loader
        self.mask_loader = mask_loader
        self.resize_factor = resize_factor
        self.gray_scale = gray_scale
        self.background = background
        self.mode = mode
        self.convert = convert
        
        if self.gray_scale == True:
            self.img_transform = img_grayscale
        else:
            self.img_transform = img_rgb
        
        img_mask_ls = gen_cxr_seg_dataset_ls(self.data_dir, 
                                             mask_name_ls=self.mask_name_ls,
                                             csv_path=self.csv_path, 
                                             extension_ls=PNG_EXTENSION,
                                             split=self.mode)
            
        self.transform = transform(mode=self.mode, 
                                   resize_factor=self.resize_factor, 
                                   gray_scale=self.gray_scale, 
                                   convert=self.convert)
        
        self.img_path, self.mask_path = list(zip(*img_mask_ls))
        
    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, index):
        image = self.img_transform(self.img_loader(self.img_path[index]))
        masks = [self.mask_loader(p) for p in self.mask_path[index]]
        
        sample = self.transform(image=image, masks=masks)
        image, masks = sample['image'], sample['masks']
        
        masks = np.stack(masks, axis=0)
        if self.background:
            masks = np.concatenate([masks, np.expand_dims(1- np.sum(masks, axis=0), axis=0)]).astype(np.uint8)
        else:
            pass
        return image, masks


class CXRSegInferDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 data_dir,
                 img_loader=img_loader, 
                 mask_loader=mask_loader,
                 resize_factor=512,
                 gray_scale=False,
                 background=False,
                 transform=get_transform, 
                 convert=True):
        self.data_dir = data_dir
        self.img_loader = img_loader
        self.mask_loader = mask_loader
        self.resize_factor = resize_factor
        self.gray_scale = gray_scale
        self.convert = convert
         
        if self.gray_scale == True:
            self.img_transform = img_grayscale
        else:
            self.img_transform = img_rgb    
        
        self.transform = transform(mode='test', 
                                   resize_factor=self.resize_factor, 
                                   gray_scale=self.gray_scale, 
                                   convert=self.convert)
        
        self.img_path = load_file_path(self.data_dir, PNG_EXTENSION)
        
    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, index):
        image = self.img_transform(self.img_loader(self.img_path[index]))
        
        sample = self.transform(image=image)
        image = sample['image']
        
        return image
