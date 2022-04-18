from re import L
import yaml
import warnings
import torch
import numpy as np
import torch.nn as nn
import math
from torch.utils import data
import os
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import logging

def get_option(opt_path):
    with open(opt_path, 'r') as f:
        option = yaml.load(f)
    return option

def build_optimizer(opt, model):
    optimizer_name = opt['optimizer']
    try:
        optimizer_class = getattr(torch.optim, optimizer_name)
        optimizer = optimizer_class(model.parameters(), lr=opt['lr'])
    except:
        raise NotImplementedError('Unable to load optimizer: \'%s\' ' % optimizer_name)
    
    return optimizer

def build_lr_scheduler(opt, optimizer):
    lr_scheduler_name = opt['lr_schedule']
    if lr_scheduler_name:
        try:
            lr_scheduler_class = getattr(getattr(torch.optim, 'lr_scheduler'), lr_scheduler_name)
            lr_scheduler = lr_scheduler_class(optimizer, **opt['lr_s_args'])
        except:
            raise NotImplementedError('Unable to load lr_scheduler: \'%s\', please check if there are any spelling errors ' % lr_scheduler_name)
        
        return lr_scheduler
    else:
        return None

def build_dataloader(opt, type='train'):
    dataset_name = opt['dataset_name']
    module = __import__('dataset.dataset')
    dataset_class = getattr(module, dataset_name)
    dataset = dataset_class(opt, type)
    dataloader = data.DataLoader(dataset,
                                 batch_size=opt['bs'] if type == 'train' else 1,
                                 num_workers=opt['num_workers'],
                                 shuffle=True if type == 'train' else False)
    return dataloader

def build_model(opt):
    model_name = opt['model_name']
    module = __import__('all_model.' + model_name + '.model')
    model_class = getattr(module, model_name)

    # load model args
    all_args = list(opt.keys())
    model_args = {}
    for i in range(len(all_args) - 4):
        model_args[all_args[i + 4]] = opt.get(all_args[i + 4])
    model = model_class(**model_args)

    if opt['cuda']:
        model = model.cuda()
    if opt['parallel']:
        model = torch.nn.DataParallel(model)

    # load pretrained dict
    if opt['resume_ckpt_path']:
        ckpt_dict = torch.load(opt['resume_ckpt_path'])['net']
        model.load_state_dict(ckpt_dict)

    return model

def build_logger(opt):
    make_dir(os.path.join(opt['save_root'], opt['log']))
    log_path = os.path.join(opt['save_root'], opt['log'], 'logs.log')
    log_format = "%(asctime)s - %(message)s"
    logging.basicConfig(filename=log_path, level=logging.DEBUG, format=log_format)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    return logger

def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        paths = path.split('/')
        now_path = ''
        for temp_path in paths:
            now_path = os.path.join(now_path, temp_path)
            if not os.path.exists(now_path):
                os.mkdir(now_path)
        return 

def calc_psnr(pred, gt, is_for_torch=True):
    if is_for_torch:
        pred = pred[0].permute(1,2,0).detach().numpy()
        gt = gt[0].premute(1,2,0).detach().numpy()

        psnr = peak_signal_noise_ratio(gt, pred)
    else:
        psnr = peak_signal_noise_ratio(gt, pred)

    return psnr

def calc_ssim(pred, gt, is_for_torch=True):
    if is_for_torch:
        pred = pred[0].permute(1,2,0).detach().numpy()
        gt = gt[0].premute(1,2,0).detach().numpy()

        ssim = structural_similarity(gt, pred, multichannel=True)
    else:
        ssim = structural_similarity(gt, pred, multichannel=True)

    return ssim

def normalize_img(img):
    if torch.max(img) > 1 or torch.min(img) < 0:
        # img: b x c x h x w
        b, c, h, w = img.shape
        temp_img = img.view(b, c, h*w)
        im_max = torch.max(temp_img, dim=2)[0].view(b, c, 1)
        im_min = torch.min(temp_img, dim=2)[0].view(b, c, 1)
        # im_min[im_min > 0] = 0.0
        # im_max[im_max < 1] = 1.0

        temp_img = (temp_img - im_min) / (im_max - im_min + 1e-7)
        
        img = temp_img.view(b, c, h, w)
    # img[img > 1] = 1.0
    # img[img < 0] = 0.0
    
    return img