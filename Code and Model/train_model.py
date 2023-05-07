#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np
import random
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# get the current path
cwd = Path.cwd()
#root = cwd.parent
root = '/cluster/home/tiayou/DeepSTPPTraining'
sys.path.insert(0, str(root))

from src import *

# use Argparse to take in the training parameters
import argparse

parser = argparse.ArgumentParser(description='Process command line arguments.')

parser.add_argument('--REGION_NAME', type=str, required=True, help='Folder name containing the dataset of that region')
parser.add_argument('--RUN_NAME', type=str, default='Base', help='Name prefix of the folder containing the run')
parser.add_argument('--USE_ALL_DATA', type=bool, default=False, help='Whether to train the model using all 10 years of data')
parser.add_argument('--FEATURE_NAME', type=str, default='noFeat', help='Length of historical events in the sequence')
parser.add_argument('--HISTORY_LEN', type=int, default=50, help='Length of historical events in the sequence')
parser.add_argument('--LOOK_AHEAD', type=int, default=1, help='Number of events to predict')
parser.add_argument('--NUM_BACK_POINTS', type=int, default=100, help='Number of background points')
parser.add_argument('--BATCH_SIZE', type=int, default=512, help='Batch size')
parser.add_argument('--NUM_EPOCHS', type=int, default=200, help='Number of epochs')
parser.add_argument('--EVAL_EPOCH', type=int, default=5, help='Evaluation epoch')
parser.add_argument('--PATIENCE', type=int, default=50, help='Patience')

args = parser.parse_args()

REGION_NAME = args.REGION_NAME
RUN_NAME = args.RUN_NAME
USE_ALL_DATA = args.USE_ALL_DATA
FEATURE_NAME = args.FEATURE_NAME
HISTORY_LEN = args.HISTORY_LEN
NUM_BACK_POINTS = args.NUM_BACK_POINTS
LOOK_AHEAD = args.LOOK_AHEAD
BATCH_SIZE = args.BATCH_SIZE
NUM_EPOCHS = args.NUM_EPOCHS
EVAL_EPOCH = args.EVAL_EPOCH
PATIENCE = args.PATIENCE

device = get_device()

# if we are to use all 10 years of data, then go through the npz files
# in the list and train sequentially
if USE_ALL_DATA:
    SEQ_LIST = [f'2022-06-30_90_90_720_10var_{FEATURE_NAME}_split',
                f'2020-07-10_90_90_720_10var_{FEATURE_NAME}_split',
                f'2018-07-21_90_90_720_10var_{FEATURE_NAME}_split',
                f'2016-07-31_90_90_720_10var_{FEATURE_NAME}_split']
else:
    SEQ_LIST = [f'2022-06-30_90_90_720_10var_{FEATURE_NAME}_split']

for index, SEQ_NAME in enumerate(SEQ_LIST):

    fpath = CACHE_DIR / REGION_NAME / f'{SEQ_NAME}.npz' 
    dataset = np.load(fpath, allow_pickle=True)

    # prepare train, validation and test set
    train_seqs = dataset['train_seqs']
    valid_seqs = dataset['valid_seqs']
    test_seqs = dataset['test_seqs']

    shared_opts = dict(normalized=True, 
                    lookback=HISTORY_LEN,
                    lookahead=LOOK_AHEAD)
    
    trainset = SlidingWindowWrapper(train_seqs, **shared_opts)
    valset   = SlidingWindowWrapper(valid_seqs, min=trainset.min, max=trainset.max, **shared_opts)
    testset  = SlidingWindowWrapper(test_seqs,  min=trainset.min, max=trainset.max, **shared_opts)

    # identify number of marks
    NUM_MARKS = train_seqs.shape[-1] - 3

    # get the scales and biases of the data
    scales = (trainset.max - trainset.min).cpu().numpy()
    biases = trainset.min.cpu().numpy()

    # read the npz file containing 1000 background points, and sample NUM_BACK_POINTS.
    ip_path = CACHE_DIR / REGION_NAME / 'init_pts.npz' 
    ip = np.load(ip_path, allow_pickle=True)
    ip = ip['init_points']
    init_points = np.array(random.sample(list(ip), NUM_BACK_POINTS))
    init_points = (init_points - biases[:2]) / scales[:2]

    if USE_ALL_DATA:
        model_name = SEQ_NAME + "_acc"
    else:
        model_name = SEQ_NAME

    run_config = DeepSTPPConfig(RUN_NAME,
                                REGION_NAME,
                                SEQ_NAME, 
                                num_marks=NUM_MARKS,
                                seq_len=HISTORY_LEN,
                                num_points=NUM_BACK_POINTS,
                                lookahead=LOOK_AHEAD,
                                point_inits=init_points,
                                batch=BATCH_SIZE,
                                epochs=NUM_EPOCHS,
                                patience=PATIENCE, 
                                )

    shared_opts = dict(batch_size=run_config.batch)
    train_loader = DataLoader(trainset, shuffle=True, **shared_opts)
    val_loader   = DataLoader(valset, shuffle=False, **shared_opts)
    test_loader  = DataLoader(testset, shuffle=False, **shared_opts)

    if index == 0:
        model = DeepSTPP(run_config, device)
    else:
        model = best_model
    
    best_model = train(model, train_loader, val_loader, run_config)
