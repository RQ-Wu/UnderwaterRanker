from asyncio.log import logger
from pickletools import read_uint2
import plistlib
from tabnanny import check
from unittest import result
from tqdm import tqdm
import utils
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr, kendalltau
import numpy as np
import os
import loss
import skimage.io as io
from datetime import datetime
import time
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import random
import cv2
def manual_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)            # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子

class UIE_Runner():
    def __init__(self, options, type='train'):
        # manual_seed(options['seed'])

        self.type = type
        self.dataset_opt = options['dataset']
        self.model_opt = options['model']
        self.training_opt = options['train']
        self.experiments_opt = options['experiments']
        self.test_opt = options['test']

        self.model = utils.build_model(self.model_opt)

        self.train_dataloader = utils.build_dataloader(self.dataset_opt, type='train')
        self.test_dataloader = utils.build_dataloader(self.dataset_opt, type='test')
        
        self.optimizer = utils.build_optimizer(self.training_opt, self.model)
        self.lr_scheduler = utils.build_lr_scheduler(self.training_opt, self.optimizer)
        
        self.tb_writer = SummaryWriter(os.path.join(self.experiments_opt['save_root'], 'tensorboard'))
        self.logger = utils.build_logger(self.experiments_opt)

    def main_loop(self):
        psnr_list = []
        ssim_list = []
        start_epoch = 0
        if self.model_opt['resume_ckpt_path']:
            ckpt = torch.load(self.model_opt['resume_ckpt_path'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            psnr_list.append(ckpt['max_psnr'])
            ssim_list.append(ckpt['max_ssim'])
            start_epoch = ckpt['epoch'] + 1
            for _ in range(start_epoch * 50):
                self.lr_scheduler.step()

        print(self.model)
        for epoch in range(start_epoch, self.training_opt['epoch']):
            print('================================ %s %d / %d ================================' % (self.experiments_opt['save_root'].split('/')[-1], epoch, self.training_opt['epoch']))
            loss = self.train_loop(epoch)
            torch.cuda.empty_cache()
            psnr, ssim = self.test_loop(epoch_num=epoch)

            psnr_list.append(psnr)
            ssim_list.append(ssim)

            self.logger.info(
                f"Epoch: {epoch}/{self.training_opt['epoch']}\t"
                f"Loss: {loss}\t"
                f"PSNR: {psnr} (max: {np.max(np.array(psnr_list))})\t"
                f"SSIM: {ssim} (max: {np.max(np.array(ssim_list))})\t"
            )
            if np.max(np.array(psnr_list)) == psnr or np.max(np.array(ssim_list)) == ssim:
                self.logger.warning(f"After {epoch+1} epochs trainingg, model achecieves best performance ==> PSNR: {psnr}, SSIM: {ssim}\n")
                # if epoch > 50:
                self.save(epoch, psnr, ssim)
            print()

    def main_test_loop(self):
        if self.test_opt['start_epoch'] >=0 and self.test_opt['end_epoch'] >=0 and self.test_opt['test_ckpt_path'] is None:
            for i in range(self.test_opt['start_epoch'], self.test_opt['end_epoch']):
                checkpoint_name = os.path.join(self.experiments_opt['save_root'], self.experiments_opt['checkpoints'], f'checkpoint_{i}.pth')
                self.test_loop(checkpoint_name, i)
        else:
            self.test_loop(self.test_opt['test_ckpt_path'])
        
    def train_loop(self, epoch_num):
        total_loss = 0
        ranker_model = utils.build_model(self.training_opt['ranker_args']) if self.training_opt['loss_rank'] else None

        with tqdm(total=len(self.train_dataloader)) as t_bar:
            for iter_num, data in enumerate(self.train_dataloader):
                # put data to cuda device
                if self.model_opt['cuda']:
                    data = {key:value.cuda() for key, value in data.items()}
                
                # model prediction
                result = self.model(**data)
                
                # # loss and bp
                loss = self.build_loss(result, data['gt_img'], ranker_model)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                
                # # print log
                # self.print()
                total_loss = (total_loss * iter_num + loss) / (iter_num + 1)
                t_bar.set_description('Epoch:%d/%d, loss:%.6f' % (epoch_num, self.training_opt['epoch'], total_loss))
                t_bar.update(1)

        
        self.tb_writer.add_scalar('train/loss', total_loss, epoch_num + 1)
        if self.training_opt['loss_rank']:
            ranker_model=ranker_model.cpu()
        return total_loss
            
    def test_loop(self, checkpoint_path=None, epoch_num=-1):
        if checkpoint_path == None:
            raise NotImplementedError('checkpoint_name can not be NoneType!')

        with torch.no_grad():
            psnr_meter = AverageMeter()
            ssim_meter = AverageMeter()
            if checkpoint_path:
                ckpt_dict = torch.load(checkpoint_path)['net']
                self.model.load_state_dict(ckpt_dict)
            if self.test_opt['save_image']:
                save_root = os.path.join(self.experiments_opt['save_root'], 'results')
                utils.make_dir(save_root)

            with tqdm(total=len(self.test_dataloader)) as t_bar:
                for iter_num, data in enumerate(self.test_dataloader):
                    _, _, h, w = data['gt_img'].shape
                    gt_img = data['gt_img'][0].permute(1, 2, 0).detach().numpy()
                    if self.model_opt['cuda']:
                        data = {key:value.cuda() for key, value in data.items()}

                    upsample = nn.UpsamplingBilinear2d((h, w))
                    pred_img = upsample(utils.normalize_img(self.model(**data)))
                    pred_img = pred_img[0].permute(1, 2, 0).detach().cpu().numpy()
                    if self.test_opt['save_image']:
                        cv2.imwrite(os.path.join(self.experiments_opt['save_root'], 'results', str(iter_num)+'.png'), pred_img[:, :, ::-1] * 255.0)

                    psnr = utils.calc_psnr(pred_img, gt_img, is_for_torch=False)
                    ssim = utils.calc_ssim(pred_img, gt_img, is_for_torch=False)

                    psnr_meter.update(psnr)
                    ssim_meter.update(ssim)

                    # update bar
                    if checkpoint_path:
                        t_bar.set_description('checkpoint: %s, psnr:%.6f, ssim:%.6f' % (checkpoint_path.split('/')[-1], psnr_meter.avg, ssim_meter.avg))
                    if epoch_num >= 0:
                        t_bar.set_description('Epoch:%d/%d, psnr:%.6f, ssim:%.6f' % (epoch_num, self.training_opt['epoch'], psnr_meter.avg, ssim_meter.avg))

                    t_bar.update(1)
        if epoch_num >= 0:
            self.tb_writer.add_scalar('valid/psnr', psnr_meter.avg, epoch_num + 1)
            self.tb_writer.add_scalar('valid/ssim', ssim_meter.avg, epoch_num + 1)

        return psnr_meter.avg, ssim_meter.avg
            
    
    def save(self, epoch_num, psnr, ssim):
        # path for saving
        path = os.path.join(self.experiments_opt['save_root'], self.experiments_opt['checkpoints'])
        utils.make_dir(path)
            
        checkpoint = {
        "net": self.model.state_dict(),
        'optimizer':self.optimizer.state_dict(),
        "epoch": epoch_num,
        "max_psnr": psnr,
        "max_ssim": ssim
        }
        torch.save(checkpoint, os.path.join(path, f'checkpoint_{epoch_num}.pth'))
    
    def build_loss(self, pred, gt, ranker_model):
        loss_total = 0
        Loss_L1 = nn.L1Loss().cuda()
        loss_total = loss_total + self.training_opt['loss_coff'][0] * Loss_L1(pred, gt)

        if self.training_opt['loss_vgg']:
            Loss_VGG = loss.make_perception_loss(self.training_opt.get('loss_vgg_args')).cuda()
            loss_total = loss_total + self.training_opt['loss_coff'][1] * Loss_VGG(pred, gt)
            del Loss_VGG
        if self.training_opt['loss_rank']:
            loss_total = loss_total + self.training_opt['loss_coff'][2] * loss.ranker_loss(ranker_model, pred)

        return loss_total

class Ranker_Runner():
    def __init__(self, options, type='train'):
        self.type = type
        self.dataset_opt = options['dataset']
        self.model_opt = options['model']
        self.training_opt = options['train']
        self.experiments_opt = options['experiments']
        self.test_opt = options['test']

        self.model = utils.build_model(self.model_opt)

        self.train_dataloader = utils.build_dataloader(self.dataset_opt, type='train')
        self.test_dataloader = utils.build_dataloader(self.dataset_opt, type='test')
        
        self.optimizer = utils.build_optimizer(self.training_opt, self.model)
        self.lr_scheduler = utils.build_lr_scheduler(self.training_opt, self.optimizer)
        
        self.tb_writer = SummaryWriter(os.path.join(self.experiments_opt['save_root'], 'tensorboard'))
        self.logger = utils.build_logger(self.experiments_opt)

    def main_loop(self):
        srocc_list = []
        krocc_list = []
        start_epoch = 0

        if self.model_opt['resume_ckpt_path']:
            ckpt = torch.load(self.model_opt['resume_ckpt_path'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt['epoch'] + 1
        for epoch in range(start_epoch, self.training_opt['epoch']):
            print('================================ %s %d / %d ================================' % (self.experiments_opt['save_root'].split('/')[-1], epoch, self.training_opt['epoch']))
            loss = self.train_loop(epoch)
            torch.cuda.empty_cache()
            srocc, krocc = self.test_loop(epoch_num=epoch)

            srocc_list.append(srocc)
            krocc_list.append(krocc)

            self.logger.info(
                f"Epoch: {epoch}/{self.training_opt['epoch']}\t"
                f"Loss: {loss}\t"
                f"SROCC: {srocc} (max: {np.max(np.array(srocc_list))})\t"
                f"KROCC: {krocc} (max: {np.max(np.array(krocc_list))})\t"
            )
            if np.max(np.array(srocc_list)) == srocc or np.max(np.array(krocc_list)) == krocc:
                self.logger.warning(f"After {epoch+1} epochs trainingg, model achecieves best performance ==> SROCC: {srocc}, KROCC: {krocc}\n")
                self.save(epoch)
            print()

    def main_test_loop(self):
        if self.test_opt['start_epoch'] >=0 and self.test_opt['end_epoch'] >=0 and self.test_opt['test_ckpt_path'] is None:
            for i in range(self.test_opt['start_epoch'], self.test_opt['end_epoch']):
                checkpoint_name = os.path.join(self.experiments_opt['save_root'], self.experiments_opt['checkpoints'], f'checkpoint_{i}.pth')
                self.test_loop(checkpoint_name, i)
        else:
            self.test_loop(self.test_opt['test_ckpt_path'])
        
    def train_loop(self, epoch_num):
        loss_meter = AverageMeter()
        with tqdm(total=len(self.train_dataloader)) as t_bar:
            for iter_num, data in enumerate(self.train_dataloader):
                # put data to cuda device
                if self.model_opt['cuda']:
                    data = {key:value.cuda() for key, value in data.items()}
                
                # pre-processing
                pre_input_high = utils.preprocessing(data['high_img'])
                pre_input_low = utils.preprocessing(data['low_img'])
                
                # model prediction
                pred_high = self.model(**pre_input_high)
                pred_low = self.model(**pre_input_low)
                
                # # loss and bp
                loss = self.build_loss(pred_high, pred_low)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                
                # # print log
                # self.print()
                loss_meter.update(loss)
                t_bar.set_description('Epoch:%d/%d, loss:%.6f' % (epoch_num, self.training_opt['epoch'], loss_meter.avg))
                t_bar.update(1)
        
        self.tb_writer.add_scalar('loss/ranker_loss', loss_meter.avg, epoch_num + 1)
        return loss_meter.avg  
            
    def test_loop(self, checkpoint_path=None, epoch_num=-1):
        if checkpoint_path == None:
            raise NotImplementedError('checkpoint_name can not be NoneType!')

        with torch.no_grad():
            srocc_meter = AverageMeter()
            krocc_meter = AverageMeter()
            if checkpoint_path:
                ckpt_dict = torch.load(checkpoint_path)['net']
                self.model.load_state_dict(ckpt_dict)

            preds = np.zeros(10)
            with tqdm(total=len(self.test_dataloader) // 10) as t_bar:
                score = np.linspace(1, 10, num=10)[::-1]
                for iter_num, data in enumerate(self.test_dataloader):
                    img = data['img'].cuda()
                    pre_input = utils.preprocessing(img)
                    pred = self.model(**pre_input)['final_result'][0][0]
                    preds[iter_num % 10] = pred.cpu().detach().numpy()
                    torch.cuda.empty_cache()
                    # calculate score
                    if (iter_num + 1) % 10 == 0:
                        step_num = iter_num // 10
                        srocc, _ = spearmanr(preds, score)
                        krocc, _ = kendalltau(preds, score)

                        # update score
                        srocc_meter.update(srocc)
                        krocc_meter.update(krocc)

                        # update bar
                        if checkpoint_path:
                            t_bar.set_description('checkpoint:%s, SROCC:%.6f, KROCC:%.6f' % (checkpoint_path, srocc_meter.avg, krocc_meter.avg))
                        if epoch_num >= 0:
                            t_bar.set_description('Epoch:%d/%d, SROCC:%.6f, KROCC:%.6f' % (epoch_num, self.training_opt['epoch'], srocc_meter.avg, krocc_meter.avg))
                        t_bar.update(1)
        if epoch_num >= 0:
            self.tb_writer.add_scalar('valid/srocc', srocc_meter.avg, epoch_num + 1)
            self.tb_writer.add_scalar('valid/krocc', krocc_meter.avg, epoch_num + 1)

        return srocc_meter.avg, krocc_meter.avg
            
    
    def save(self, epoch_num):
        best_range = (epoch_num // 50 + 1) * 50
        # path for saving
        path = os.path.join(self.experiments_opt['save_root'], self.experiments_opt['checkpoints'])
        utils.make_dir(path)
            
        checkpoint = {
        "net": self.model.state_dict(),
        'optimizer':self.optimizer.state_dict(),
        "epoch": epoch_num
        }
        torch.save(checkpoint, os.path.join(path, f'checkpoint_{best_range}best.pth'))
    
    def build_loss(self, result1, result2):
        L_final = loss.rank_loss(result1['final_result'], result2['final_result'])
        L_rank = L_final
        
        return L_rank

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger():
    def __init__(self, experiments_opt):
        self.expreiments_opt = experiments_opt
        self.loss_list = []
        self.srocc_list = []
        self.acc_list = []
        self.path = os.path.join(self.expreiments_opt['logger_root'], self.expreiments_opt['exp_name'])
        
    def write(self, loss, srocc, acc, epoch_num):
        self.loss_list.append(loss)
        self.srocc_list.append(srocc)
        self.acc_list.append(acc)
        
        if not os.path.exists(self.path):
            os.mkdir(self.path)
            
        # draw loss and score 
        plt.plot(self.loss_list)
        plt.savefig(os.path.join(self.path, 'loss.png'))
        plt.clf()
        plt.plot(self.srocc_list)
        plt.plot(self.acc_list)
        plt.savefig(os.path.join(self.path, 'scores.png'))
        plt.clf()
        
        # saving log
        if np.max(np.array(self.srocc_list)) == srocc:
            with open(os.path.join(self.path, 'log.txt'), 'a') as f:
                now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                f.write('%s saving best checkpoint: Ecpoh:%d loss:%.6f, SROCC:%.3f, ACC:%.3f\n' % (now, epoch_num, loss, srocc, acc))
            return True
        else:
            return False