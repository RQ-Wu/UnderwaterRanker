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
        option = yaml.safe_load(f)
    
    option.setdefault('seed', 2022)

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
    lr_scheduler_name = opt['lr_scheduler'] if 'lr_scheduler' in opt.keys() else None
    if lr_scheduler_name:
        try:
            lr_scheduler_class = getattr(getattr(torch.optim, 'lr_scheduler'), lr_scheduler_name)
        except:
            raise NotImplementedError('Unable to load lr_scheduler: \'%s\', please check if there are any spelling errors ' % lr_scheduler_name)
        try:
            lr_scheduler = lr_scheduler_class(optimizer, **opt['lr_scheduler_arg'])
        except:
             raise NotImplementedError('Failed to load optimizer')
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
        im_max = torch.max(img)
        im_min = torch.min(img)

        img = (img - im_min) / (im_max - im_min + 1e-7)
    
    return img

def preprocessing(d_img_org):     
        d_img_org = padding_img(d_img_org)
        x_his = build_historgram(d_img_org)
        return {
            'x': d_img_org,
            'x_his': x_his
        }
        
def padding_img(img):
    b, c, h, w = img.shape
    h_out = math.ceil(h / 32) * 32
    w_out = math.ceil(w / 32) * 32
    
    left_pad = (w_out- w) // 2
    right_pad = w_out - w - left_pad
    top_pad  = (h_out - h) // 2
    bottom_pad = h_out - h - top_pad
    
    img = nn.ZeroPad2d((left_pad, right_pad, top_pad, bottom_pad))(img)
    
    return img

def build_historgram(img):
    with torch.no_grad():
        b, _, _, _ = img.shape

        r_his = torch.histc(img[0][0], 64, min=0.0, max=1.0)
        g_his = torch.histc(img[0][1], 64, min=0.0, max=1.0)
        b_his = torch.histc(img[0][2], 64, min=0.0, max=1.0)

        historgram = torch.cat((r_his, g_his, b_his)).unsqueeze(0).unsqueeze(0)

        for i in range(1, b):
            r_his = torch.histc(img[i][0], 64, min=0.0, max=1.0)
            g_his = torch.histc(img[i][1], 64, min=0.0, max=1.0)
            b_his = torch.histc(img[i][2], 64, min=0.0, max=1.0)

            historgram_temp = torch.cat((r_his, g_his, b_his)).unsqueeze(0).unsqueeze(0)
            historgram = torch.cat((historgram, historgram_temp), dim=0)

    return historgram