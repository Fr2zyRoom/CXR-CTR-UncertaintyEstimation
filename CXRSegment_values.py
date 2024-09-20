import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils

import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import label
from tqdm import tqdm
import argparse
import copy

from util.util import *
from metrics.segmentation_metrics import *
from dataset.cxr_segmentation import *
from tools.cxr_tools import *
from tools.img_tools import *
from tools.mask_tools import *
from values.c1 import *
from values.c2 import *
from values.c3 import *

class setup_config():
    def __init__(self):
        self.initilized = False
    
    def initialize(self, parser):
        parser.add_argument('--dataroot', required=True, help='path to data')
        parser.add_argument('--mask_name_ls', required=True, nargs="+", help='list of masks')
        parser.add_argument('--savepoint', required=True, help='path to save results')
        parser.add_argument('--weights', required=True, help='path to model weights')
        ##data##
        parser.add_argument('--resize_factor', type=int, default=256, help='model input size')
        parser.add_argument('--gray_scale', action='store_true', help='grayscale')
        parser.add_argument('--background', action='store_true', help='background')
        parser.add_argument('--enhancement', type=str, default=None, help='image enhancement: None | CLAHE | Gamma | CosineGamma')
        parser.add_argument('--gaussian', action='store_true', help='Gaussian Blur')
        
        ##model##
        parser.add_argument('--patch_size', type=int, default=64, help='aggregation patch size')
        parser.add_argument('--model', required=True, help='UNet | FPN | DeepLabV3Plus')
        parser.add_argument('--encoder', required=True, help='backbone encoder')
        parser.add_argument('--activation', required=True, help='sigmoid/softmax2d')
        ##optimize##
        parser.add_argument('--batch_size', type=int, default=16, help='batch size')
        
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

def chunk_list(ls, n):
    return [ls[i:i + n] for i in range(0, len(ls), n)]

def run(opt):
    assert os.path.exists(opt.dataroot)
    
    if opt.enhancement == 'CLAHE':
        img_loader_func = img_loader_clahe
    elif opt.enhancement == 'Gamma':
        img_loader_func = img_loader_gamma_correction
    elif opt.enhancement == 'CosineGamma':
        img_loader_func = img_loader_cosine_gamma_correction
    else:
        img_loader_func = img_loader
    
    if opt.gray_scale is True:
        in_channels=1
        img_transform = img_grayscale
    else:
        in_channels=3
        img_transform = img_rgb

    if opt.background is True:
        n_class = len(opt.mask_name_ls)+1
    else:
        n_class = len(opt.mask_name_ls)

    transform = get_tta_transform(mode='test', 
                  resize_factor=opt.resize_factor, 
                  gray_scale=opt.gray_scale, 
                  convert=True)

    if opt.model == 'Unet':
        model = smp.Unet(
            encoder_name=opt.encoder, 
            encoder_weights=None, 
            in_channels=in_channels,
            classes=n_class, 
            activation=opt.activation,
        )
    elif opt.model == 'FPN':
        model = smp.FPN(
            encoder_name=opt.encoder, 
            encoder_weights=None, 
            in_channels=in_channels,
            classes=n_class, 
            activation=opt.activation,
        )
    elif opt.model == 'DeepLabV3Plus':
        model = smp.DeepLabV3Plus(
            encoder_name=opt.encoder, 
            encoder_weights=None, 
            in_channels=in_channels,
            classes=n_class, 
            activation=opt.activation,
        )
    else:
        model = smp.DeepLabV3Plus(
            encoder_name=opt.encoder, 
            encoder_weights=None, 
            in_channels=in_channels,
            classes=n_class, 
            activation=opt.activation,
        )
    
    gen_new_dir(opt.savepoint)

    # data_df = pd.read_csv(opt.csvpath)

    # if opt.name is not None:
    #     weight_name = '_'.join([opt.model, opt.encoder, opt.activation, opt.name])
    # else:
    #     weight_name = 'trial'
        
    ### VALUES: A FRAMEWORK FOR SYSTEMATIC VALIDATION OF UNCERTAINTY ESTIMATION IN SEMANTIC SEGMENTATION

    model1 = copy.deepcopy(model)
    # load best saved checkpoint
    model1.load_state_dict(torch.load(opt.weights))

    filename_ls = [n for n in os.listdir(opt.dataroot) if n.endswith('.png')]
    
    columns = ['FileName', 'CardiacDiameter', 'ThoracicDiameter', 'CTR',
               'SoftmaxTTA_PU_image', 'SoftmaxTTA_PU_patch', 'SoftmaxTTA_AU_image', 'SoftmaxTTA_AU_patch', 'SoftmaxTTA_EU_image', 'SoftmaxTTA_EU_patch', 'CardiacSilhouetteObscurationRatio']
    values_df = pd.DataFrame(columns=columns)
    len_df = len(filename_ls)

    filename_batch_ls = chunk_list(filename_ls, opt.batch_size)
    
    for idx, filename_batch in enumerate(filename_batch_ls):
        cxr_img_batch = []
        for batch_idx, filename in enumerate(filename_batch):
            print(f'Extracting VALUES from {filename} ({idx*opt.batch_size+batch_idx}/{len_df})')
            cxr_img = img_transform(img_loader_func(os.path.join(opt.dataroot, filename)))
            norm_img = cv2.resize(cxr_img, (512, 512), interpolation=cv2.INTER_AREA)
            norm_img[:50,:50,:] = 0
            norm_img = normalize(norm_img.astype(np.float16),True)
            if opt.gaussian:
                norm_img = cv2.GaussianBlur(norm_img, sigmaX=5, ksize=(0, 0))
            cxr_img_batch.append(norm_img)

        print('C1 Prediction Model')

        # Test time augmentation (TTA)
        softmax_pred_tta_batch = get_batch_softmax_pred_with_tta_from(cxr_img_batch, transform, model1)
        
        for batch_idx, softmax_pred_tta in enumerate(softmax_pred_tta_batch):
            print('C2 Uncertainty Measure')
            ## TTA - random augmentation variable T
            # PU Measure: predictive entropy
            # EU Measure: MI(Y,T|x)
            # AU Measure: expected entropy
            softmax_pred_tta_uncertainty = calculate_uncertainty(softmax_pred_tta)
            
            print('Calculate CT Ratio')
            pred_mask = np.argmax(np.mean(softmax_pred_tta,0),0)
            
            lung_mask = pred_mask == 0
            heart_mask = pred_mask == 1
            cardiac_size, thoracic_size, ct_ratio = cal_ct_ratio(heart_mask, lung_mask)

            print('C3 Aggregation Strategy')
            #C3: Aggregation Strategy
            (rmin, rmax, cmin, cmax) = find_bounding_box(np.logical_or(heart_mask>0,lung_mask>0))
            softmax_pred_tta_pu_image_agg = image_level_aggregation(softmax_pred_tta_uncertainty["pred_entropy"][rmin:rmax,cmin:cmax],True)
            softmax_pred_tta_pu_patch_agg = patch_level_aggregation(softmax_pred_tta_uncertainty["pred_entropy"],opt.patch_size)
    
            softmax_pred_tta_au_image_agg = image_level_aggregation(softmax_pred_tta_uncertainty["aleatoric_uncertainty"][rmin:rmax,cmin:cmax],True)
            softmax_pred_tta_au_patch_agg = patch_level_aggregation(softmax_pred_tta_uncertainty["aleatoric_uncertainty"],opt.patch_size)
    
            softmax_pred_tta_eu_image_agg = image_level_aggregation(softmax_pred_tta_uncertainty["epistemic_uncertainty"][rmin:rmax,cmin:cmax],True)
            softmax_pred_tta_eu_patch_agg = patch_level_aggregation(softmax_pred_tta_uncertainty["epistemic_uncertainty"],opt.patch_size)

            print('Calculate CSOR')
            # 값이 1인 요소의 위치를 찾음
            y_indices, _ = np.where(pred_mask == 1)
            if y_indices.shape[0] != 0:
                # y축 최저점과 최고점 찾기
                heart_y_min = np.min(y_indices)
                heart_y_max = np.max(y_indices)
        
                zero_to_one_transition_ls = find_zero_to_one_transition_per_row(pred_mask)
                one_to_zero_transition_ls = find_one_to_zero_transition_per_row(pred_mask)
                
                right_bottom_point_x, right_bottom_point_y = zero_to_one_transition_ls[-1]
                left_bottom_point_x, left_bottom_point_y = one_to_zero_transition_ls[-1]
                # CSOR: Cardiac Silhouette Obscuration Ratio
                csor = (heart_y_max-left_bottom_point_y) / (heart_y_max-heart_y_min) if right_bottom_point_y > left_bottom_point_y else (heart_y_max-right_bottom_point_y) / (heart_y_max-heart_y_min)
            else:
                csor = .5
            values_df.loc[idx*opt.batch_size+batch_idx] = [filename_batch[batch_idx], cardiac_size, thoracic_size, ct_ratio,
                              softmax_pred_tta_pu_image_agg, softmax_pred_tta_pu_patch_agg['max_score'], 
                              softmax_pred_tta_au_image_agg, softmax_pred_tta_au_patch_agg['max_score'], 
                              softmax_pred_tta_eu_image_agg, softmax_pred_tta_eu_patch_agg['max_score'], csor]
    
            values_df.to_csv(os.path.join(opt.savepoint, '_'.join(opt.mask_name_ls) + '_values_dataframe.csv'), index=False)
        
    
if __name__=='__main__':
    opt = setup_config().parse()
    run(opt)
