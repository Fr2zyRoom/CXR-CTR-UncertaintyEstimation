#C1: Prediction Model

# Softmax (Plain deterministic softmax)
# Softmax - Temperature scaling
# MC-Dropout at test-time (TTD)
# Ensemble of 5 models
# Test time augmentation (TTA)
# Stochastic Segmentation Network (SSN) - Not Ready

# N x C x W x H
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils


# Softmax (Plain deterministic softmax)
def get_softmax_pred_from(img, transform, model):
    img = torch.unsqueeze(transform(image=img)['image'],0)
    model_copy = deepcopy(model)
    with torch.no_grad():
        model_copy.eval()
        model_copy = model_copy.cuda()
        DEVICE = 'cuda'
        out = model_copy(img.to(DEVICE))
        softmax_pred_prob = out.cpu().detach().numpy()
    del model_copy
    torch.cuda.empty_cache()
    return softmax_pred_prob


# Softmax - Temperature scaling (deterministic softmax)
def get_softmax_pred_with_temp_scal_from(img, transform, model, T=1000.0):
    img = torch.unsqueeze(transform(image=img)['image'],0)
    model_copy = deepcopy(model)
    with torch.no_grad():
        model_copy.segmentation_head[2] = segmentation_models_pytorch.base.modules.Activation(None)
        model_copy.eval()
        model_copy = model_copy.cuda()
        DEVICE = 'cuda'
        out = model_copy(img.to(DEVICE))
        softmax_pred_prob = F.softmax(out / T, dim=1).cpu().detach().numpy()
    del model_copy
    torch.cuda.empty_cache()
    return softmax_pred_prob


# MC-Dropout at test-time (TTD)- bayesian
def encode_with_dropout(x, encoder_model):
    m = torch.nn.Dropout(p=0.5)
    x = encoder_model.conv_stem(x)
    x = encoder_model.bn1(x)
    x = encoder_model.act1(x)
    if encoder_model.feature_hooks is None:
        features = []
        if 0 in encoder_model._stage_out_idx:
            features.append(x)  # add stem out
        for i, b in enumerate(encoder_model.blocks):
            x = b(x)
            # dropout
            x = m(x)
            if i + 1 in encoder_model._stage_out_idx:
                features.append(x)
        return features
    else:
        encoder_model.blocks(x)
        out = encoder_model.feature_hooks.get_output(x.device)
        return list(out.values())

def decode_with_dropout(features, decoder_model):
    m = torch.nn.Dropout(p=0.5)
    features = features[1:]    # remove first skip with same spatial resolution
    features = features[::-1]  # reverse channels to start from head of encoder

    head = features[0]
    skips = features[1:]

    x = decoder_model.center(head)
    for i, decoder_block in enumerate(decoder_model.blocks):
        skip = skips[i] if i < len(skips) else None
        x = decoder_block(x, skip)
        # dropout
        x = m(x)
    return x

def get_softmax_pred_with_MC_dropout_from(img, transform, model):
    img = torch.unsqueeze(transform(image=img)['image'],0)
    model_copy = deepcopy(model)
    with torch.no_grad():
        model_copy.eval()
        model_copy = model_copy.cuda()
        DEVICE = 'cuda'
        features = encode_with_dropout(img.to(DEVICE), model_copy.encoder.model)
        features = [img.to(DEVICE),] + features
        
        #decoder_output = model_copy.decoder(*features)
        decoder_output = decode_with_dropout(features, model_copy.decoder)
        masks = model_copy.segmentation_head(decoder_output)
        
        softmax_pred_prob = masks.cpu().detach().numpy()
    del model_copy
    torch.cuda.empty_cache()
    return softmax_pred_prob

def get_N_softmax_pred_with_MC_dropout_from(img, transform, model, N=10):
    softmax_pred_ls = []
    for _ in range(N):
        softmax_pred_ls.append(get_softmax_pred_with_MC_dropout_from(img, transform, model)[0])
    softmax_pred_prob = np.stack(softmax_pred_ls)
    return softmax_pred_prob


# Ensemble of 5 models
def get_softmax_pred_with_ensemble_from(img, transform, model_ls):
    softmax_pred_ls = []
    for model in model_ls:
        softmax_pred_ls.append(get_softmax_pred_from(img, transform, model)[0])
    softmax_pred_prob = np.stack(softmax_pred_ls)
    return softmax_pred_prob


# Test time augmentation (TTA)
# def tta_aug_transform(img):
#     return [img, np.fliplr(img), np.rot90(img,1), np.rot90(img,2), np.rot90(img,3), np.rot90(np.fliplr(img),1), np.rot90(np.fliplr(img),2), np.rot90(np.fliplr(img),3)]

# def tta_deaug_transform(aug_img_ls):
#     return [np.squeeze(aug_img_ls[0]), 
#             np.stack([np.fliplr(img) for img in np.squeeze(aug_img_ls[1])]),
#             np.stack([np.rot90(img,3) for img in np.squeeze(aug_img_ls[2])]),
#             np.stack([np.rot90(img,2) for img in np.squeeze(aug_img_ls[3])]),
#             np.stack([np.rot90(img,1) for img in np.squeeze(aug_img_ls[4])]),
#             np.stack([np.fliplr(np.rot90(img,3)) for img in np.squeeze(aug_img_ls[5])]),
#             np.stack([np.fliplr(np.rot90(img,2)) for img in np.squeeze(aug_img_ls[6])]),
#             np.stack([np.fliplr(np.rot90(img,1)) for img in np.squeeze(aug_img_ls[7])])]

def tta_aug_transform(img):
    gaussian_blur_img = cv2.GaussianBlur(img, sigmaX=2, ksize=(0, 0))
    return [img, np.fliplr(img), np.rot90(img,1), np.rot90(img,2), np.rot90(img,3), np.rot90(np.fliplr(img),1), np.rot90(np.fliplr(img),2), np.rot90(np.fliplr(img),3),
           gaussian_blur_img, np.fliplr(gaussian_blur_img), np.rot90(gaussian_blur_img,1), np.rot90(gaussian_blur_img,2), np.rot90(gaussian_blur_img,3), np.rot90(np.fliplr(gaussian_blur_img),1), np.rot90(np.fliplr(gaussian_blur_img),2), np.rot90(np.fliplr(gaussian_blur_img),3)]

def tta_deaug_transform(aug_img_ls):
    return [np.squeeze(aug_img_ls[0]), 
            np.stack([np.fliplr(img) for img in np.squeeze(aug_img_ls[1])]),
            np.stack([np.rot90(img,3) for img in np.squeeze(aug_img_ls[2])]),
            np.stack([np.rot90(img,2) for img in np.squeeze(aug_img_ls[3])]),
            np.stack([np.rot90(img,1) for img in np.squeeze(aug_img_ls[4])]),
            np.stack([np.fliplr(np.rot90(img,3)) for img in np.squeeze(aug_img_ls[5])]),
            np.stack([np.fliplr(np.rot90(img,2)) for img in np.squeeze(aug_img_ls[6])]),
            np.stack([np.fliplr(np.rot90(img,1)) for img in np.squeeze(aug_img_ls[7])]),
            np.squeeze(aug_img_ls[8]), 
            np.stack([np.fliplr(img) for img in np.squeeze(aug_img_ls[9])]),
            np.stack([np.rot90(img,3) for img in np.squeeze(aug_img_ls[10])]),
            np.stack([np.rot90(img,2) for img in np.squeeze(aug_img_ls[11])]),
            np.stack([np.rot90(img,1) for img in np.squeeze(aug_img_ls[12])]),
            np.stack([np.fliplr(np.rot90(img,3)) for img in np.squeeze(aug_img_ls[13])]),
            np.stack([np.fliplr(np.rot90(img,2)) for img in np.squeeze(aug_img_ls[14])]),
            np.stack([np.fliplr(np.rot90(img,1)) for img in np.squeeze(aug_img_ls[15])])]


def get_softmax_pred_with_tta_from(img, transform, model):
    softmax_pred_prob_ls = [get_softmax_pred_from(aug_img, transform, model) for aug_img in tta_aug_transform(img)]
    return np.stack(tta_deaug_transform(softmax_pred_prob_ls))



# Softmax (Plain deterministic softmax)
def get_batch_softmax_pred_from(img_batch, transform, model):
    transformed_imgs = [transform(image=img)['image'] for img in img_batch]
    transformed_batch = torch.stack(transformed_imgs)
    model_copy = deepcopy(model)
    model_copy.eval()
    model_copy = model_copy.cuda()
    
    DEVICE = 'cuda'
    
    with torch.no_grad():
        out = model_copy(transformed_batch.to(DEVICE))
        softmax_pred_prob = out.cpu().detach().numpy()
    del model_copy
    torch.cuda.empty_cache()
    
    return softmax_pred_prob
    

def get_batch_softmax_pred_with_tta_from(img_batch, transform, model):
    tta_img_bacth = list(zip(*[tta_aug_transform(img) for img in img_batch]))
    batch_softmax_pred_prob_ls = [get_batch_softmax_pred_from(tta_img, transform, model) for tta_img in tta_img_bacth]
    
    return np.stack([np.stack(tta_deaug_transform(softmax_pred_prob_ls)) for softmax_pred_prob_ls in list(zip(*batch_softmax_pred_prob_ls))])