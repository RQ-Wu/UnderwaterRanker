from asyncio.log import logger
from pickletools import read_uint2
from tabnanny import check
from unittest import result
from click import option
from tqdm import tqdm
import utils
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr
import numpy as np
import os
import loss
import skimage.io as io
from datetime import datetime
import time
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

class UIE_Runner():
    def __init__(self, opt_path, type='train'):
        options = utils.get_option(opt_path)

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
        for epoch in range(self.training_opt['epoch']):
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
                self.logger.warning(f"After {epoch+1} epochs trainingg, model achecieves best performance ==> PSNR: {psnr}, SSIM: {ssim}")
                if epoch > 80:
                    self.save(epoch, psnr, ssim)
            print()

    def main_test_loop(self):
        if self.test_opt['start_epoch'] >=0 and self.test_opt['end_epoch'] >=0:
            for i in range(self.test_opt['start_epoch'], self.test_opt['end_epoch']):
                checkpoint_name = os.path.join(self.experiments_opt['save_root'], self.experiments_opt['checkpoints'], f'checkpoint_{i}.pth')
                self.test_loop(checkpoint_name, i)
        else:
            checkpoint_name = os.path.join(self.experiments_opt['save_root'], self.experiments_opt['checkpoints'], self.test_opt['test_ckpt_path'])
            self.test_loop(checkpoint_name)
        
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
        ranker_model=ranker_model.cpu()
        return total_loss
            
    def test_loop(self, checkpoint_path=None, epoch_num=None):
        if checkpoint_path == None and epoch_num == None:
            raise NotImplementedError('checkpoint_name and epoch_num can not both be NoneType!')

        with torch.no_grad():
            psnr_meter = AverageMeter()
            ssim_meter = AverageMeter()
            if checkpoint_path:
                ckpt_dict = torch.load(checkpoint_path)['net']
                self.model.load_state_dict(ckpt_dict)

            with tqdm(total=len(self.test_dataloader)) as t_bar:
                for iter_num, [data, filename] in enumerate(self.test_dataloader):
                    gt_img = data['gt_img'][0].permute(1, 2, 0).detach().numpy()
                    pred_img = utils.normalize_img(self.model(**data))[0].permute(1, 2, 0).detach().cpu().numpy()
                    # pred_img = io.imread(os.path.join(self.experiments_opt['save_root'], self.experiments_opt['results'], filename[0])) / 255.0

                    psnr = utils.calc_psnr(pred_img, gt_img, is_for_torch=False)
                    ssim = utils.calc_ssim(pred_img, gt_img, is_for_torch=False)

                    psnr_meter.update(psnr)
                    ssim_meter.update(ssim)

                    if self.test_opt['save_img']:
                        io.imsave(os.path.join(self.experiments_opt['save_root'], self.experiments_opt['results'], filename[0]),
                                result[0].permute(1, 2, 0).cpu().detach().numpy())
                    # update bar
                    if checkpoint_path:
                        t_bar.set_description('checkpoints: %s, psnr:%.6f, ssim:%.6f' % (checkpoint_path.split('/')[-1], psnr_meter.avg, ssim_meter.avg))
                    elif epoch_num >= 0:
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
        "epoch": epoch_num
        }
        torch.save(checkpoint, os.path.join(path, f'checkpoint_{epoch_num}_psnr{psnr}_ssim{ssim}.pth'))
    
    def build_loss(self, pred, gt, ranker_model):
        loss_total = 0
        Loss_L2 = nn.MSELoss().cuda()
        loss_total = loss_total + self.training_opt['loss_coff'][0] * Loss_L2(pred, gt)

        if self.training_opt['loss_vgg']:
            Loss_VGG = loss.perception_loss().cuda()
            loss_total = loss_total + self.training_opt['loss_coff'][1] * Loss_VGG(pred, gt)
            del Loss_VGG
        if self.training_opt['loss_rank']:
            loss_total = loss_total + self.training_opt['loss_coff'][2] * loss.ranker_loss(ranker_model, pred)

        return loss_total

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