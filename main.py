import os
import sys
import time
import datetime
import argparse
import logging
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import distributed as dist
from apex import amp

from configs.default_img import get_img_config
from configs.default_vid import get_vid_config
from data import build_dataloader
from models import build_model
from losses import build_losses
from tools.utils import save_checkpoint, set_seed, get_logger
from train import train_cal, train_cal_with_memory
from test import test, test_prcc


VID_DATASET = ['ccvid']


def parse_option():
    parser = argparse.ArgumentParser(description='Train clothes-changing re-id model with clothes-based adversarial loss')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    # Datasets
    parser.add_argument('--root', type=str, help="your root path to data directory")
    parser.add_argument('--dataset', type=str, default='ltcc', help="ltcc, prcc, vcclothes, ccvid, last, deepchange")
    # Miscs
    parser.add_argument('--output', type=str, help="your output path to save model and logs")
    parser.add_argument('--resume', type=str, metavar='PATH')
    parser.add_argument('--amp', action='store_true', help="automatic mixed precision")
    parser.add_argument('--eval', action='store_true', help="evaluation only")
    parser.add_argument('--tag', type=str, help='tag for log file')
    parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

    args, unparsed = parser.parse_known_args()
    if args.dataset in VID_DATASET:
        config = get_vid_config(args)
    else:
        config = get_img_config(args)

    return config


def main(config):
    # Build dataloader
    if config.DATA.DATASET == 'prcc':
        trainloader, queryloader_same, queryloader_diff, galleryloader, dataset, train_sampler = build_dataloader(config)
    else:
        trainloader, queryloader, galleryloader, dataset, train_sampler = build_dataloader(config)
    # Define a matrix pid2clothes with shape (num_pids, num_clothes). 
    # pid2clothes[i, j] = 1 when j-th clothes belongs to i-th identity. Otherwise, pid2clothes[i, j] = 0.
    pid2clothes = torch.from_numpy(dataset.pid2clothes)

    # Build model
    model, classifier, clothes_classifier = build_model(config, dataset.num_train_pids, dataset.num_train_clothes)
    # Build identity classification loss, pairwise loss, clothes classificaiton loss, and adversarial loss.
    criterion_cla, criterion_pair, criterion_clothes, criterion_adv = build_losses(config, dataset.num_train_clothes)
    # Build optimizer
    parameters = list(model.parameters()) + list(classifier.parameters())
    if config.TRAIN.OPTIMIZER.NAME == 'adam':
        optimizer = optim.Adam(parameters, lr=config.TRAIN.OPTIMIZER.LR, 
                               weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        optimizer_cc = optim.Adam(clothes_classifier.parameters(), lr=config.TRAIN.OPTIMIZER.LR, 
                                  weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
    elif config.TRAIN.OPTIMIZER.NAME == 'adamw':
        optimizer = optim.AdamW(parameters, lr=config.TRAIN.OPTIMIZER.LR, 
                               weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        optimizer_cc = optim.AdamW(clothes_classifier.parameters(), lr=config.TRAIN.OPTIMIZER.LR, 
                                  weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
    elif config.TRAIN.OPTIMIZER.NAME == 'sgd':
        optimizer = optim.SGD(parameters, lr=config.TRAIN.OPTIMIZER.LR, momentum=0.9, 
                              weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY, nesterov=True)
        optimizer_cc = optim.SGD(clothes_classifier.parameters(), lr=config.TRAIN.OPTIMIZER.LR, momentum=0.9, 
                              weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY, nesterov=True)
    else:
        raise KeyError("Unknown optimizer: {}".format(config.TRAIN.OPTIMIZER.NAME))
    # Build lr_scheduler
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config.TRAIN.LR_SCHEDULER.STEPSIZE, 
                                         gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE)

    start_epoch = config.TRAIN.START_EPOCH
    if config.MODEL.RESUME:
        logger.info("Loading checkpoint from '{}'".format(config.MODEL.RESUME))
        checkpoint = torch.load(config.MODEL.RESUME)
        model.load_state_dict(checkpoint['model_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        if config.LOSS.CAL == 'calwithmemory':
            criterion_adv.load_state_dict(checkpoint['clothes_classifier_state_dict'])
        else:
            clothes_classifier.load_state_dict(checkpoint['clothes_classifier_state_dict'])
        start_epoch = checkpoint['epoch']

    local_rank = dist.get_rank()
    model = model.cuda(local_rank)
    classifier = classifier.cuda(local_rank)
    if config.LOSS.CAL == 'calwithmemory':
        criterion_adv = criterion_adv.cuda(local_rank)
    else:
        clothes_classifier = clothes_classifier.cuda(local_rank)
    torch.cuda.set_device(local_rank)

    if config.TRAIN.AMP:
        [model, classifier], optimizer = amp.initialize([model, classifier], optimizer, opt_level="O1")
        if config.LOSS.CAL != 'calwithmemory':
            clothes_classifier, optimizer_cc = amp.initialize(clothes_classifier, optimizer_cc, opt_level="O1")

    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    classifier = nn.parallel.DistributedDataParallel(classifier, device_ids=[local_rank], output_device=local_rank)
    if config.LOSS.CAL != 'calwithmemory':
        clothes_classifier = nn.parallel.DistributedDataParallel(clothes_classifier, device_ids=[local_rank], output_device=local_rank)

    if config.EVAL_MODE:
        logger.info("Evaluate only")
        with torch.no_grad():
            if config.DATA.DATASET == 'prcc':
                test_prcc(model, queryloader_same, queryloader_diff, galleryloader, dataset)
            else:
                test(config, model, queryloader, galleryloader, dataset)
        return

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    logger.info("==> Start training")
    for epoch in range(start_epoch, config.TRAIN.MAX_EPOCH):
        train_sampler.set_epoch(epoch)
        start_train_time = time.time()
        if config.LOSS.CAL == 'calwithmemory':
            train_cal_with_memory(config, epoch, model, classifier, criterion_cla, criterion_pair, 
                criterion_adv, optimizer, trainloader, pid2clothes)
        else:
            train_cal(config, epoch, model, classifier, clothes_classifier, criterion_cla, criterion_pair, 
                criterion_clothes, criterion_adv, optimizer, optimizer_cc, trainloader, pid2clothes)
        train_time += round(time.time() - start_train_time)        
        
        if (epoch+1) > config.TEST.START_EVAL and config.TEST.EVAL_STEP > 0 and \
            (epoch+1) % config.TEST.EVAL_STEP == 0 or (epoch+1) == config.TRAIN.MAX_EPOCH:
            logger.info("==> Test")
            torch.cuda.empty_cache()
            if config.DATA.DATASET == 'prcc':
                rank1 = test_prcc(model, queryloader_same, queryloader_diff, galleryloader, dataset)
            else:
                rank1 = test(config, model, queryloader, galleryloader, dataset)
            torch.cuda.empty_cache()
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            model_state_dict = model.module.state_dict()
            classifier_state_dict = classifier.module.state_dict()
            if config.LOSS.CAL == 'calwithmemory':
                clothes_classifier_state_dict = criterion_adv.state_dict()
            else:
                clothes_classifier_state_dict = clothes_classifier.module.state_dict()
            if local_rank == 0:
                save_checkpoint({
                    'model_state_dict': model_state_dict,
                    'classifier_state_dict': classifier_state_dict,
                    'clothes_classifier_state_dict': clothes_classifier_state_dict,
                    'rank1': rank1,
                    'epoch': epoch,
                }, is_best, osp.join(config.OUTPUT, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))
        scheduler.step()

    logger.info("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    logger.info("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
    

if __name__ == '__main__':
    config = parse_option()
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
    # Init dist
    dist.init_process_group(backend="nccl", init_method='env://')
    local_rank = dist.get_rank()
    # Set random seed
    set_seed(config.SEED + local_rank)
    # get logger
    if not config.EVAL_MODE:
        output_file = osp.join(config.OUTPUT, 'log_train_.log')
    else:
        output_file = osp.join(config.OUTPUT, 'log_test.log')
    logger = get_logger(output_file, local_rank, 'reid')
    logger.info("Config:\n-----------------------------------------")
    logger.info(config)
    logger.info("-----------------------------------------")

    main(config)