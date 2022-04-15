import os
import yaml
from yacs.config import CfgNode as CN


_C = CN()
# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Root path for dataset directory
_C.DATA.ROOT = '/home/guxinqian/data'
# Dataset for evaluation
_C.DATA.DATASET = 'ccvid'
# Whether split each full-length video in the training set into some clips
_C.DATA.DENSE_SAMPLING = True
# Sampling step of dense sampling for training set
_C.DATA.SAMPLING_STEP = 64
# Workers for dataloader
_C.DATA.NUM_WORKERS = 4
# Height of input image
_C.DATA.HEIGHT = 256
# Width of input image
_C.DATA.WIDTH = 128
# Batch size for training
_C.DATA.TRAIN_BATCH = 16
# Batch size for testing
_C.DATA.TEST_BATCH = 128
# The number of instances per identity for training sampler
_C.DATA.NUM_INSTANCES = 4
# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Random erase prob
_C.AUG.RE_PROB = 0.0
# Temporal sampling mode for training, 'tsn' or 'stride'
_C.AUG.TEMPORAL_SAMPLING_MODE = 'stride'
# Sequence length of each input video clip
_C.AUG.SEQ_LEN = 8
# Sampling stride of each input video clip
_C.AUG.SAMPLING_STRIDE = 4
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model name. All supported model can be seen in models/__init__.py
_C.MODEL.NAME = 'c2dres50'
# The stride for laery4 in resnet
_C.MODEL.RES4_STRIDE = 1
# feature dim
_C.MODEL.FEATURE_DIM = 2048
# Model path for resuming
_C.MODEL.RESUME = ''
# Params for AP3D
_C.MODEL.AP3D = CN()
# Temperature for APM
_C.MODEL.AP3D.TEMPERATURE = 4
# Contrastive attention
_C.MODEL.AP3D.CONTRACTIVE_ATT = True
# -----------------------------------------------------------------------------
# Losses for training 
# -----------------------------------------------------------------------------
_C.LOSS = CN()
# Classification loss
_C.LOSS.CLA_LOSS = 'crossentropy'
# Clothes classification loss
_C.LOSS.CLOTHES_CLA_LOSS = 'cosface'
# Scale for classification loss
_C.LOSS.CLA_S = 16.
# Margin for classification loss
_C.LOSS.CLA_M = 0.
# Pairwise loss
_C.LOSS.PAIR_LOSS = 'triplet'
# The weight for pairwise loss
_C.LOSS.PAIR_LOSS_WEIGHT = 0.0
# Scale for pairwise loss
_C.LOSS.PAIR_S = 16.
# Margin for pairwise loss
_C.LOSS.PAIR_M = 0.3
# Clothes-based adversarial loss
_C.LOSS.CAL = 'cal'
# Epsilon for clothes-based adversarial loss
_C.LOSS.EPSILON = 0.1
# Momentum for clothes-based adversarial loss with memory bank
_C.LOSS.MOMENTUM = 0.
# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.MAX_EPOCH = 150
# Start epoch for clothes classification
_C.TRAIN.START_EPOCH_CC = 50
# Start epoch for adversarial training
_C.TRAIN.START_EPOCH_ADV = 50
# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adam'
# Learning rate
_C.TRAIN.OPTIMIZER.LR = 0.00035
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 5e-4
# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
# Stepsize to decay learning rate
_C.TRAIN.LR_SCHEDULER.STEPSIZE = [40, 80, 120]
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# Using amp for training
_C.TRAIN.AMP = False
# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Perform evaluation after every N epochs (set to -1 to test after training)
_C.TEST.EVAL_STEP = 10
# Start to evaluate after specific epoch
_C.TEST.START_EVAL = 0
# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Fixed random seed
_C.SEED = 1
# Perform evaluation only
_C.EVAL_MODE = False
# GPU device ids for CUDA_VISIBLE_DEVICES
_C.GPU = '0, 1'
# Path to output folder, overwritten by command line argument
_C.OUTPUT = '/data/guxinqian/logs/'
# Tag of experiment, overwritten by command line argument
_C.TAG = 'res50-ce-cal'


def update_config(config, args):
    config.defrost()
    config.merge_from_file(args.cfg)

    # merge from specific arguments
    if args.root:
        config.DATA.ROOT = args.root
    if args.output:
        config.OUTPUT = args.output

    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.eval:
        config.EVAL_MODE = True
    
    if args.tag:
        config.TAG = args.tag

    if args.dataset:
        config.DATA.DATASET = args.dataset
    if args.gpu:
        config.GPU = args.gpu
    if args.amp:
        config.TRAIN.AMP = True

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.DATA.DATASET, config.TAG)

    config.freeze()


def get_vid_config(args):
    """Get a yacs CfgNode object with default values."""
    config = _C.clone()
    update_config(config, args)

    return config
