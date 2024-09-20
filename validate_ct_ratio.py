import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import argparse

from util.util import *
from tools.segmentation_tools import *
from tools.cxr_tools import *

class setup_config():
    def __init__(self):
        self.initilized = False
    
    def initialize(self, parser):
        parser.add_argument('--gt_ct_ratio', required=True, help='path to ground truth CT Ratio(.csv)')
        parser.add_argument('--pred_heart_maskdir', required=True, help='path to heart data(pred)')
        parser.add_argument('--pred_lung_maskdir', required=True, help='path to lung data(pred)')
        parser.add_argument('--savepoint', required=True, help='path to save results')
        parser.add_argument('--name', required=True, help='file name_CT_Ratio.csv')
        parser.add_argument('--print_options', action="store_true", help='print option?')

        self.initialized = True
        return parser
    
    
    def gather_options(self):
        if not self.initilized:  # check if it has been initialized
            parser = argparse.ArgumentParser(description='Heart segmentation model inference')
            parser = self.initialize(parser)
            self.parser = parser
        return parser.parse_args()
    
    
    def print_options(self, opt):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        
        # save to the disk
#         expr_dir = os.path.join(opt.checkpoint, opt.name)
#         gen_new_dir(expr_dir)
#         file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.name))
#         with open(file_name, 'wt') as opt_file:
#             opt_file.write(message)
#             opt_file.write('\n')
    
    
    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        
#         if opt.name is None:
#             opt.name = ''.join(datetime.datetime.now(pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d-%H-%M-%S'))
        
#         # process opt.suffix
#         if opt.suffix:
#             suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
#             opt.name = opt.name + suffix
        if opt.print_options:
            self.print_options(opt)
        self.opt = opt
        return self.opt

def mask_loader(mask_path):
    return (np.array(Image.open(mask_path).convert('L'))>0).astype(np.uint8)

def run(opt, return_mae=False, do_save=True):
    gt_ct_ratio = pd.read_csv(opt.gt_ct_ratio)
    
    pred_heart_path_ls = load_file_path(opt.pred_heart_maskdir, PNG_EXTENSION)
    pred_lung_path_ls = load_file_path(opt.pred_lung_maskdir, PNG_EXTENSION)
    
    pred_lung_path_dict = gen_fname_path_dict(pred_lung_path_ls)
    pred_heart_path_dict = gen_fname_path_dict(pred_heart_path_ls)
    
    pred_heart_lung_path_ls = [[pred_heart_path_dict[dataset_fname], pred_lung_path_dict[dataset_fname]] 
                             for dataset_fname in list(pred_heart_path_dict.keys()) if pred_lung_path_dict.get(dataset_fname) is not None]
    
    fname_ct_ratio_ls = []
    for heart_path, lung_path in pred_heart_lung_path_ls:
        fname = os.path.splitext(os.path.basename(heart_path))[0]
        heart_mask = mask_loader(heart_path)
        lung_mask = mask_loader(lung_path)
        _, _, ct_ratio = cal_ct_ratio(heart_mask, lung_mask)
        fname_ct_ratio_ls.append([fname, ct_ratio])
    
    pred_ct_ratio = pd.DataFrame(fname_ct_ratio_ls,columns=['FileName', 'Pred_CT_Ratio'])
    ct_ratio_result = pd.merge(gt_ct_ratio, pred_ct_ratio)
    ct_ratio_result['AbsoluteError'] = ct_ratio_result.apply(lambda x: np.abs(x['CT_Ratio']-x['Pred_CT_Ratio']), axis=1)
    MAE = np.mean(ct_ratio_result['AbsoluteError'].values)
    if opt.print_options:
        print(f'mean MAE of CT Ratio: {MAE}')
        print(f'max MAE of CT Ratio: {np.max(ct_ratio_result.AbsoluteError.values)}')
        print(f'min MAE of CT Ratio: {np.min(ct_ratio_result.AbsoluteError.values)}')
    
    if do_save:
        createDirectory(opt.savepoint)
        ct_ratio_result.to_csv(os.path.join(opt.savepoint, f'{opt.name}_CT_Ratio_Result.csv'), index=False)
    
    if return_mae:
        return MAE

def getValidCtrMAE(gt_ct_ratio, pred_heart_maskdir, pred_lung_maskdir, savepoint, name, print_options):
    opt = setup_config()
    opt.gt_ct_ratio = gt_ct_ratio
    opt.pred_heart_maskdir = pred_heart_maskdir
    opt.pred_lung_maskdir = pred_lung_maskdir
    opt.savepoint = savepoint
    opt.name = name
    opt.print_options = print_options

    return run(opt, do_save=False, return_mae=True)

if __name__=='__main__':
    opt = setup_config().parse()
    run(opt)
