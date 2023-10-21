import os
from config.config import args
os.environ['CUDA_VISIBLE_DEVICES'] = "%s" % args.GPU_ID
import numpy as np
import torch
import argparse
import cv2
import torch.utils.data as data
import torchvision
import random
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from tqdm import tqdm
import lpips
from model.VDM_PCD import VDM_PCD, model_fn_decorator
from data.load_video_temporal import *
from utils.loss_util import *
from utils.common import *


def val_epoch(args, ValImgLoader, model, model_fn_val, net_metric, epoch, save_path):
    save_path = save_path + '/' + '%04d' % epoch
    mkdir(save_path)
    #tbar = tqdm(ValImgLoader)
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    total_lpips = 0

    for batch_idx, data in enumerate(ValImgLoader):
        loss, cur_psnr, cur_ssim, cur_lpips = model_fn_val(args, data, model, net_metric, save_path)

        total_loss += loss.item()
        avg_val_loss = total_loss / (batch_idx + 1)
        total_psnr += cur_psnr
        avg_val_psnr = total_psnr / (batch_idx + 1)
        total_ssim += cur_ssim
        avg_val_ssim = total_ssim / (batch_idx + 1)
        total_lpips += cur_lpips
        avg_val_lpips = total_lpips / (batch_idx + 1)
        #tbar.set_description(desc)
        #tbar.update()
    desc = 'Validation: Epoch %d, Avg. LPIPS = %.4f, Avg. PSNR = %.4f and SSIM = %.4f, Avg. Loss = %.5f' % (
            epoch, avg_val_lpips, avg_val_psnr, avg_val_ssim, avg_val_loss)
    print(desc)

    return avg_val_loss, avg_val_psnr, avg_val_ssim, avg_val_lpips


def train_epoch(args, TrainImgLoader, model, model_fn, optimizer, epoch, iters, lr_scheduler):
    #tbar = tqdm(TrainImgLoader)
    num_batch = len(TrainImgLoader)
    total_loss = 0
    total_loss_temporal = 0
    total_loss_reg = 0
    
    # lr_scheduler
    lr_scheduler.step()
    print('lr ================= ',optimizer.state_dict()['param_groups'][0]['lr'])
    logger.add_scalar('Train/learning_rate', optimizer.state_dict()['param_groups'][0]['lr'], epoch)

    for batch_idx, data in enumerate(TrainImgLoader):
        loss, loss_temporal, loss_reg, in_img, ref, out_img_, out_img, label = model_fn(args, data, model, iters, epoch)
        optimizer.zero_grad()
        loss.backward()
        
        # test grad
        # for name, param in model.named_parameters():
        #     print(name, param.requires_grad)

        optimizer.step()
        #lr_scheduler.step()
        iters += 1
        lr = optimizer.state_dict()['param_groups'][0]['lr']

        total_loss += loss.item()
        avg_train_loss = total_loss / (batch_idx + 1)
        total_loss_temporal += loss_temporal.item()
        avg_train_loss_temporal = total_loss_temporal / (batch_idx + 1)
        total_loss_reg += loss_reg.item()
        avg_train_loss_reg = total_loss_reg / (batch_idx + 1)
        if batch_idx % args.PRINT_FEQ ==0:
            desc = 'Training  : lr %e, Epoch %d, iter %d/%d, AvgLoss:%.5f, AvgLossTemporal:%.5f, AvgLossReg:%.5f' % (
                lr, epoch, batch_idx, num_batch, avg_train_loss, avg_train_loss_temporal, avg_train_loss_reg)
            print(desc)
            logger.add_scalar('Train/avg_loss', avg_train_loss, (epoch-1)*num_batch+batch_idx)
            logger.add_scalar('Train/avg_temp_loss', avg_train_loss_temporal, (epoch-1)*num_batch+batch_idx)
            logger.add_scalar('Train/avg_reg_loss', avg_train_loss_reg, (epoch-1)*num_batch+batch_idx)
            in_save = in_img.detach().cpu()[0:1, [0,1,3], :, :]
            ref_save = ref.detach().cpu()[0:1, :, ::2, ::2]
            out_save_ = out_img_.detach().cpu()[0:1, :, ::2, ::2]
            out_save = out_img.detach().cpu()[0:1, :, ::2, ::2]
            gt_save = label.detach().cpu()[0:1, :, ::2, ::2]
            res_save = torch.cat((in_save, ref_save, out_save, gt_save), 3)
            logger.add_images('res_save',res_save,(epoch-1)*num_batch+batch_idx)
        #print(desc)
        #tbar.set_description(desc)
        #tbar.update()

    return lr, avg_train_loss, iters


def init():
    """
    Initialize settings
    """

    # Make dirs
    mkdir(args.MODEL_DIR)
    mkdir(args.VAL_RESULT_DIR)
    mkdir(args.LOGS_DIR)
    mkdir(args.VISUALS_DIR)
    mkdir(args.NETS_DIR)

    # GPU devices
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID#"%d" % args.GPU_ID

    # logger
    logger = SummaryWriter(args.LOGS_DIR)

    # LPIPS
    net_metric_alex = lpips.LPIPS(net='alex').cuda()

    # random seed
    random.seed(args.SEED)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed_all(args.SEED)
    if args.SEED == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True#不需要复现结果，尽可能提升网络性能，设成True

    return logger, net_metric_alex


def load_checkpoint(model, load_epoch):
    # import shutil
    # shutil.copy('blend_vdm_pcd_shuffle_f3_i1_t2_mixup/model_dir/nets/checkpoint_000049.tar', args.NETS_DIR)

    load_dir = args.NETS_DIR + '/checkpoint' + '_' + '%06d' % load_epoch + '.tar'
    print('Loading pre-trained checkpoint %s' % load_dir)
    avg_lpips = torch.load(load_dir)['avg_val_lpips']
    avg_psnr = torch.load(load_dir)['avg_val_psnr']
    avg_ssim = torch.load(load_dir)['avg_val_ssim']
    print('Avg. LPIPS, PSNR and SSIM values recorded from the checkpoint: %f, %f, %f' % (avg_lpips, avg_psnr, avg_ssim))
    model_state_dict = torch.load(load_dir)['state_dict']
    # 多gpu
    for i in list(model_state_dict.keys()):
        if 'module.' not in i:
            new_name = 'module.'+i
            model_state_dict[new_name] = model_state_dict.pop(i)
    #
    model.load_state_dict(model_state_dict)
    learning_rate = torch.load(load_dir)['learning_rate']
    iters = torch.load(load_dir)['iters']
    print('Learning rate recorded from the checkpoint: %s' % str(learning_rate))

    return learning_rate, iters


def load_pretrain(model, load_dir):
    print('Loading pre-trained checkpoint %s' % load_dir)
    model_state_dict = torch.load(load_dir)['state_dict']
    dict_keys = list(model_state_dict.keys())
    #change name
    for name in dict_keys:
        if 'module.' not in name:
            new_name = 'module.'+name
            model_state_dict[new_name] = model_state_dict.pop(name)
    model.load_state_dict(model_state_dict,strict=False)


if __name__ == '__main__':
    logger, net_metric = init()
    learning_rate = args.BASE_LR
    iters = 0
    # model = VDM_PCD(args).cuda()
    # 多gpu
    gpus = eval('['+args.GPU_ID+']')
    if len(gpus)>1:
        model = nn.DataParallel(VDM_PCD(args).cuda(), device_ids=gpus, output_device=gpus[0])
    elif len(gpus)==1:
        model = VDM_PCD(args).cuda()
    if args.LOAD_EPOCH != 0:
        _, iters = load_checkpoint(model, args.LOAD_EPOCH)

    # load pre-trained model
    # if args.fhd_pretrain is not None:
    #     load_pretrain(model, args.fhd_pretrain)
            
    #print requires_grad
    # for name, param in model.named_parameters():
    #     if 'swin' not in name and 'embed' not in name:
    #         param.requires_grad = False
    #     print(name, param.requires_grad)

    loss_fn = multi_VGGPerceptualLoss(lam_p=args.WEIGHT_PEC, lam_l=args.WEIGHT_L1).cuda()#, lam_c=args.WEIGHT_COLOR
    ## no deep supervision loss
    # loss_fn = single_VGGPerceptualLoss(lam_p=args.WEIGHT_PEC, lam_l=args.WEIGHT_L1).cuda()
    optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': learning_rate}], lr=learning_rate, betas=(0.9, 0.999))
    #lr_scheduler = CosineAnnealingLR_warmup(args, optimizer, base_lr=args.BASE_LR, last_epoch=iters - 1, min_lr=1e-7)
    lr_scheduler = SubsectionLR(optimizer,update_iter=args.UPDATE_ITER,update_weights=args.UPDATE_WEIGHTS,last_epoch=args.LOAD_EPOCH-1)

    # model_fn(criterion)
    model_fn = model_fn_decorator(loss_fn=loss_fn)
    model_fn_val = model_fn_decorator(loss_fn=loss_fn, mode='val')

    # create dataloader
    tr_input_list = sorted([file for file in os.listdir(args.TRAIN_DATASET + 'gt_rgb/') if (file.endswith('.jpg') or file.endswith('.png'))])#[:10]
    val_input_list = sorted([file for file in os.listdir(args.TEST_DATASET + 'gt_rgb/') if (file.endswith('.jpg') or file.endswith('.png'))])[0:-1:10]
    TrainImgLoader = data.DataLoader(data_loader(args, tr_input_list, mode='train'),
                                     batch_size=args.BATCH_SIZE,
                                     shuffle=True,
                                     num_workers=0,
                                     pin_memory=True)
    ValImgLoader = data.DataLoader(data_loader(args, val_input_list, mode='val'),
                                   batch_size=1,
                                   shuffle=False,
                                   num_workers=0)

    # train and val metrics
    avg_train_loss = 0
    avg_val_psnr = 0
    avg_val_ssim = 0
    avg_val_lpips = 0
    avg_val_loss = 0

    for epoch in range(args.LOAD_EPOCH + 1, args.EPOCHS + 1):
        # manual set lr
        # for p in optimizer.param_groups:
        #     p['lr'] = 5e-5
        #optimizer.state_dict()['param_groups'][0]['lr'] = 1e-7#———————————————————————————————————————————————————————————————————————————————
        
        learning_rate, avg_train_loss, iters = train_epoch(args, TrainImgLoader, model, model_fn,
                                                           optimizer, epoch, iters, lr_scheduler)
        if epoch % args.VAL_TIME == args.VAL_TIME - 1:
            avg_val_loss, avg_val_psnr, avg_val_ssim, avg_val_lpips = val_epoch(args, ValImgLoader, model, model_fn_val,
                                                                                net_metric, epoch, args.VISUALS_DIR)
        #logger.add_scalar('Train/avg_loss', avg_train_loss, epoch)
        logger.add_scalar('Validation/avg_psnr', avg_val_psnr, epoch)
        logger.add_scalar('Validation/avg_ssim', avg_val_ssim, epoch)
        logger.add_scalar('Validation/avg_lpips', avg_val_lpips, epoch)
        logger.add_scalar('Validation/avg_val_loss', avg_val_loss, epoch)
        logger.add_scalar('Train/learning_rate', learning_rate, epoch)

        # Save the network per epoch with performance metrics as well
        savefilename = args.NETS_DIR + '/checkpoint' + '_' + '%06d' % epoch + '.tar'
        torch.save({
            'learning_rate': learning_rate,
            'iters': iters,
            'avg_val_lpips': avg_val_lpips,
            'avg_val_psnr': avg_val_psnr,
            'avg_val_ssim': avg_val_ssim,
            'state_dict': model.state_dict()
        }, savefilename)

    logger.close()








