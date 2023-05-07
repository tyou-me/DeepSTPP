#!/usr/bin/env python
# coding: utf-8


import os
import sys

# enable relative imports from within this notebook
cwd = Path.cwd()
root = cwd.parent.parent   # NEED TO CHANGE THIS
sys.path.insert(0, str(root))


import logging

import numpy as np
import random
import torch
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
import glob
import json

from datetime import datetime, timedelta

from src import *

#from src.common import SEQ_CACHE_DIR, MODEL_CACHE_DIR, TMP_DIR
#from modules.steda import get_eastern_drc_admins, squash_gdf, sample_from_poly
#from modules.steda.steda.common.plotting_utils import fully_despine_axis

get_ipython().system('pip install pyemd')

# Define key notebook parameters

REGION_NAME = 'ACLED-COD'
MODEL_NAME = f'{REGION_NAME}'

device = get_device()

folder_path = CACHE_DIR / REGION_NAME
dir_list = os.listdir(folder_path)

for path in glob.glob(f'{folder_path}/*/'):

    curr_path = path #'/Users/tyou/Projects/conflict-stpp/store/ACLED_COD/AllData_0504114256_00/'

    # get the config file
    with open(f'{curr_path}/config.json', 'r') as f:  
        rg = json.load(f) 
    run_config = DeepSTPPConfig(*rg) 
    
    # get the model
    model_path = glob.glob(f'{curr_path}/*best_model.mod')[0]
    model_name = model_path.split('/')[-1]

    model = DeepSTPP(run_config, device)
    model.load_state_dict(torch.load(f'{curr_path}/best_model.mod', map_location=device))

    # load the data
    SEQ_NAME = run_config.seq_name

    LAST_DATE, TEST_PERIOD_LEN, VALID_PERIOD_LEN, TRAIN_PERIOD_LEN, NUM_VARIANTS, FEAT, _ = SEQ_NAME.split('_')
    LAST_DATE = datetime.strptime(LAST_DATE, '%Y-%m-%d')
    TEST_PERIOD_LEN = int(TEST_PERIOD_LEN)
    VALID_PERIOD_LEN = int(VALID_PERIOD_LEN)
    TRAIN_PERIOD_LEN = int(TRAIN_PERIOD_LEN)
    NUM_VARIANTS = int(NUM_VARIANTS[:len(NUM_VARIANTS)-3])

    test_start = LAST_DATE - timedelta(days=(TEST_PERIOD_LEN-1))

    dataset = np.load(f'{curr_path}/{SEQ_NAME}.npz', allow_pickle=True)
    train_seqs = dataset['train_seqs']
    valid_seqs = dataset['valid_seqs']
    test_seqs = dataset['test_seqs']

    shared_opts = dict(normalized=True, 
                   lookback=run_config.seq_len,
                   lookahead=run_config.lookahead)

    trainset = SlidingWindowWrapper(train_seqs, **shared_opts)
    valset   = SlidingWindowWrapper(valid_seqs, min=trainset.min, max=trainset.max, **shared_opts)
    testset  = SlidingWindowWrapper(test_seqs,  min=trainset.min, max=trainset.max, **shared_opts)

    scales = (trainset.max - trainset.min).cpu().numpy()
    biases = trainset.min.cpu().numpy()
    data_info = DataInfo(biases, scales, test_start)

    # Read background points file
    init_pts_file = np.load(folder_path/'init_pts.npz', allow_pickle=True)
    init_points = init_pts_file['init_points']
    init_points = np.array(random.sample(list(init_points), run_config.num_points))
    init_points = (init_points - data_info.biases[:2]) /data_info.scales[:2]

    shared_opts = dict(batch_size=run_config.batch)
    train_loader = DataLoader(trainset, shuffle=True, **shared_opts)
    val_loader   = DataLoader(valset, shuffle=False, **shared_opts)
    test_loader  = DataLoader(testset, shuffle=False, **shared_opts)
    
    ## Information about the hexagons
    HEX_NUM = 297
    hi_path = CACHE_DIR / REGION_NAME / f'hex{HEX_NUM}_info.npz' 
    hi = np.load(hi_path, allow_pickle=False)

    mesh_info = MeshInfo(torch.from_numpy(hi['mesh_centre']), 
                        hi['mesh_area'][0],
                        torch.from_numpy(hi['mesh_region']))


    predictor = ModelPredictor(model, run_config, device, data_info, mesh_info)


    mesh_count, mesh_count_avg = predictor.pred_mesh_count(test_loader)
    agg_count = predictor.pred_agg_count(mesh_count_avg)



    y_true_admin2 = np.load(curr_path / f'count_admin2.pkl', allow_pickle=True)
    y_true_hex = np.load(curr_path / MODEL_NAME / f'count_hex{HEX_NUM}.pkl', allow_pickle=True)

    calc_score(agg_count, y_true_admin2, )

