import numpy as np
import cv2
import pandas as pd
from PIL import Image
from tqdm import tqdm
import argparse

from util.util import *
from tools.segmentation_tools import *
from tools.cxr_tools import *
from metrics.segmentation_metrics import *

class setup_config():
    def __init__(self):
        self.initilized = False
    
    def initialize(self, parser):
        parser.add_argument('--gt_maskdir', required=True, help='path to ground truth mask')
        parser.add_argument('--pred_maskdir', required=True, help='path to pred mask')
        parser.add_argument('--savepoint', required=True, help='path to save results')
        parser.add_argument('--name', required=True, help='file name_DICE.csv')
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
        
        self.print_options(opt)
        self.opt = opt
        return self.opt

def mask_loader(mask_path):
    return (np.array(Image.open(mask_path).convert('L'))>0).astype(np.uint8)

def run(opt):
    gt_path_ls = load_file_path(opt.gt_maskdir, PNG_EXTENSION)
    pred_path_ls = load_file_path(opt.pred_maskdir, PNG_EXTENSION)
    
    gt_path_dict = gen_fname_path_dict(gt_path_ls)
    pred_path_dict = gen_fname_path_dict(pred_path_ls)
    
    gt_pred_path_ls = [[gt_path_dict[dataset_fname], pred_path_dict[dataset_fname]] 
                             for dataset_fname in list(gt_path_dict.keys()) if pred_path_dict.get(dataset_fname) is not None]
    
    fname_dice_ls = []
    for gt_path, pred_path in gt_pred_path_ls:
        fname = os.path.splitext(os.path.basename(gt_path))[0]
        gt_mask = cv2.resize(mask_loader(gt_path), dsize=(256,256), interpolation=cv2.INTER_AREA)
        pred_mask = cv2.resize(mask_loader(pred_path), dsize=(256,256), interpolation=cv2.INTER_AREA)
        fname_dice_ls.append([fname, dice(gt_mask, pred_mask)])
    
    dice_result = pd.DataFrame(fname_dice_ls,columns=['FileName', 'Dice'])
    print(f'mean Dice score: {np.mean(dice_result.Dice.values)}')
    print(f'max Dice score: {np.max(dice_result.Dice.values)}')
    print(f'min Dice score: {np.min(dice_result.Dice.values)}')
    dice_result.to_csv(os.path.join(opt.savepoint, f'{opt.name}_DICE.csv'), index=False)
if __name__=='__main__':
    opt = setup_config().parse()
    run(opt)
