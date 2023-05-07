import copy
import datetime
import json
import logging
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from copy import deepcopy
from dataclasses import dataclass
from typing import Union
from tqdm.auto import tqdm, trange
from tqdm.contrib import tenumerate


class AverageMeter(object):
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


def visualize_diff(outputs, targets, portion=1):
    outputs = outputs[:int(len(outputs) * portion)]
    targets = targets[:int(len(targets) * portion)]

    plt.figure(figsize=(14, 10), dpi=180)
    plt.subplot(2, 2, 1)

    n = outputs.shape[0]
    lookahead = outputs.shape[1]

    for i in range(lookahead):
        plt.plot(range(i, n), outputs[:n - i, i, 0], "-o", label=f"Predicted {i} step")
    plt.plot(targets[:, 0, 0], "-o", color="b", label="Actual")
    plt.ylabel('Latitude')
    plt.legend()

    plt.subplot(2, 2, 2)
    for i in range(lookahead):
        plt.plot(range(i, n), outputs[:n - i, i, 1], "-o", label=f"Predicted {i} step")
    plt.plot(targets[:, 0, 1], "-o", color="b", label="Actual")
    plt.ylabel('Longitude')
    plt.legend()

    plt.subplot(2, 2, 3)
    for i in range(lookahead):
        plt.plot(range(i, n), outputs[:n - i, i, 2], "-o", label=f"Predicted {i} step")
    plt.plot(targets[:, 0, 2], "-o", color="b", label="Actual")
    plt.ylabel('delta_t (hours)')
    plt.legend()
    plt.savefig('result.png')


def eval_loss(model, test_loader):
    """
    Custom eval function for the DeepSTPP model, which has been amended
    to avoid memory leaks on the GPU.
    """
    model.eval()
    sll_meter = AverageMeter()
    tll_meter = AverageMeter()
    loss_meter = AverageMeter()
    for data in test_loader:
        st_x, st_y, _, _, _ = data
        loss, sll, tll = model.loss(st_x, st_y)
        loss_meter.update(loss.item())
        sll_meter.update(sll.mean().item())
        tll_meter.update(tll.mean().item())
    return loss_meter.avg, sll_meter.avg, tll_meter.avg


def train(model, train_loader, val_loader, config):
    """
    Custom training function for the DeepSTPP model, which supports
    early stopping as well as check-pointing.
    """
    prepare_run(config)
    logger = get_logger(config)

    scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, step_size=50, gamma=0.2)
    best_eval = np.infty
    best_epoch = 0
    sll_meter = AverageMeter()
    tll_meter = AverageMeter()
    loss_meter = AverageMeter()

    since_improvement = 0
    for epoch in trange(config.epochs):
        if epoch == 0:
            best_model = copy.deepcopy(model)

        model.train()
        for index, data in tenumerate(train_loader):
            st_x, st_y, _, _, _ = data

            model.optimizer.zero_grad()
            loss, sll, tll = model.loss(st_x, st_y)
            if torch.isnan(loss):
                print('Numerical error, quiting...')
                return best_model

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            model.optimizer.step()

            loss_meter.update(loss.item())
            sll_meter.update(sll.mean().item())
            tll_meter.update(tll.mean().item())

        scheduler.step()
        msg = get_log_msg(loss_meter.avg, sll_meter.avg, tll_meter.avg)
        logger.info(f'In training epoch {epoch} | {msg}')

        check_epoch = (epoch+1) % config.eval_epoch == 0
        last_epoch = (epoch+1) == config.eval_epoch
        if check_epoch or last_epoch:
            print('Evaluate')
            valloss, valspace, valtime = eval_loss(model, val_loader)
            msg = get_log_msg(valloss, valspace, valtime)
            logger.info(f'VALIDATION | {msg}')
            #log_background_points(model, config, epoch)

            # Checkpointing and early-stopping
            if valloss < best_eval:
                best_eval = valloss
                best_epoch = epoch
                best_model = copy.deepcopy(model)
                since_improvement = 0
                save_model_state(best_model, config, f'{config.seq_name}_best_model')
                update_valloss_file(valloss, config, epoch)
            else:
                if since_improvement > config.patience:
                    if not last_epoch:
                        print('Stopping early with best Val Loss '
                              f'{best_eval:5f} from epoch {best_epoch}')
                        break
                since_improvement += config.eval_epoch
    save_model_state(model, config, f'{config.seq_name}_final_model')
    print("Training done!")
    return best_model


'''
The following is added by Tian You
'''

def load_best_state(region_name, seq_name, run_num):
    """
    Return the state of the best model checkpoint on a
    user-provided run number.
    """
    """
    if run_num is None:
        run_num = get_highest_run_number(model_name)
        if run_num == 0:
            raise ValueError(
                f"No run data available for model '{model_name}'."
            )
    """

    run_cache_dir = CACHE_DIR / region_name / f'{run_num}'
    fpath = run_cache_dir / f'{seq_name}_best_model.mod'
    if not fpath.is_file():
        raise ValueError(
            'Best checkpoint not found for model '
            f"'Model trained based on {seq_name}' from run (={run_num})"
        )
    print(f"Loading best checkpoint for run {run_num} of model '{seq_name}'...")
    return torch.load(fpath)


def prepare_run(config):
    """
    Make caching and logging directories for the present model run,
    and save the config file to disk for documentation.
    """

    #config.run_num = get_highest_run_number(config.model_name) + 1
    ct = datetime.datetime.now()
    temp_run_num = config.run_name + "_" + ct.strftime("%m%d%H%M%S")
    temp_path = CACHE_DIR / config.region_name / temp_run_num
    path_exists = os.path.exists(temp_path)
    run_count = 0
    while path_exists:
        run_count = run_count + 1
        path_exists = os.path.exists(temp_path+str(run_count))
    
    config.run_num = temp_run_num + "_{:02d}".format(run_count)
    config.run_cache_dir = CACHE_DIR / config.region_name / f'{config.run_num}'
    config.run_log_dir = CACHE_DIR / config.region_name / f'{config.run_num}'
    config.run_cache_dir.mkdir(exist_ok=False, parents=True)
    #config.run_log_dir.mkdir(exist_ok=False, parents=True)
    save_config(config)

def get_logger(config):
    logger = logging.getLogger('full_lookahead{}batch{}'.format(config.lookahead, config.batch))
    logger.setLevel(logging.DEBUG)

    # Make log file handler
    log_fname = f'{config.region_name}_run{config.run_num}_batch{config.batch}_lr{config.lr}'
    hdlr = logging.FileHandler(config.run_log_dir / log_fname)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)

    # Make console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def get_log_msg(total_loss, s_metric, t_metric):
    return (f'Total loss: {total_loss:5f} | '
            f'Space: {s_metric:5f} | '
            f'Time: {t_metric:5f}')


def log_background_points(model, config, epoch):
    points = model.background.detach().cpu()
    plt.figure(figsize=(3, 6))
    plt.scatter(*points.T)
    plt.savefig(config.run_log_dir / f'background_points_epoch{epoch}.png')
    plt.close()


def update_valloss_file(current_best, config, epoch):
    # To avoid going through the full logs, let's write the
    # best validation loss to a single file.
    file = config.run_log_dir / 'best_valloss.txt'
    with open(file, 'w') as f:
        f.write(f'Best validation loss in epoch {epoch}:\n')
        f.write(f'{current_best:.5f}')


def save_model_state(model, config, name):
    torch.save(
        model.state_dict(),
        config.run_cache_dir / f'{name}.mod')


def save_config(config):
    # We need to make paths JSON-serialisable or drop them
    config_ = deepcopy(config)
    config_.run_cache_dir = str(config.run_cache_dir)
    config_.run_log_dir = str(config.run_log_dir)

    # Drop initialisation values for background points
    del config_.point_inits

    with open(config.run_log_dir / 'config.json', 'w') as f:
        f.write(json.dumps(config_.__dict__))


def get_device():
    if torch.cuda.is_available():
        print("You are using GPU acceleration.")
        print("Device name: ", torch.cuda.get_device_name(0))
        print("Number of CUDAs(cores): ", torch.cuda.device_count())
        return torch.device('cuda:0')

    print("CUDA is not Available. You are using CPU only.")
    print("Number of cores: ", os.cpu_count())
    return torch.device("cpu")

    

'''
Functions for train and evaluate rmtpp by Zhou, which isnot relavent to Tian's thesis and
moved to the very end
'''
def eval_loss_rmtpp(model, test_loader, device):
    model.eval()
    loss_total = 0
    loss_meter = AverageMeter()
    
    for index, data in enumerate(test_loader):
        st_x, st_y, _, _, _ = data
        loss = model.loss(st_x, st_y)
        
        loss_meter.update(loss.item())
        
    return loss_meter.avg


def train_rmtpp(model, train_loader, val_loader, config, logger, device):
    scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, step_size=50, gamma=0.2)
    best_eval = np.infty
    loss_meter = AverageMeter()
    
    for epoch in trange(config.epochs):
        loss_total = 0
        model.train()
        for index, data in tenumerate(train_loader):
            st_x, st_y, _, _, _ = data

            model.optimizer.zero_grad()
            loss = model.loss(st_x, st_y)

            if torch.isnan(loss):
                print("Numerical error, quiting...")
                return best_model

            loss.backward()
            model.optimizer.step()

            loss_meter.update(loss.item())

        scheduler.step()

        logger.info("In epochs {} | Loss: {:5f}".format(
            epoch, loss_meter.avg
        ))
        if (epoch+1)%config.eval_epoch==0:
            print("Evaluate")
            valloss = eval_loss_rmtpp(model, val_loader, device)
            logger.info("Val Loss {:5f} ".format(valloss))
            if valloss < best_eval:
                best_eval = valloss
                best_model = copy.deepcopy(model)

    print("training done!")
    return best_model



def mult_eval(models, n_eval, dataset, test_loader, config, device, scales, rmtpp=False):
    time_scale = np.log(scales[-1])
    space_scale = np.log(np.prod(scales[:2]))

    sll_list = []
    tll_list = []
    with torch.no_grad():
        for model in models:
            model.eval()
            for _ in trange(n_eval):
                if rmtpp:
                    tll = eval_loss_rmtpp(model, test_loader, device)
                    sll_list.append(0.0)
                    tll_list.append(-tll - time_scale)
                else:
                    _, sll, tll = eval_loss(model, test_loader, device)
                    sll_list.append(sll.item() - space_scale)
                    tll_list.append(tll.item() - time_scale)

    print("%.4f" % np.mean(sll_list), '±', "%.4f" % np.std(sll_list))
    print("%.4f" % np.mean(tll_list), '±', "%.4f" % np.std(tll_list))