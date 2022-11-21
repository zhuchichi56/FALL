# %%
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-05-27 15:00
# @Author  : Xiaoke Huang
# @Email   : xiaokehuang@foxmail.com
from modeling.vectornet import HGNN
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import pandas as pd
from utils.viz_utils import show_predict_result
import matplotlib.pyplot as plt
import numpy as np
import pdb
import os
from dataset import GraphDataset
# from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.loader import DataLoader
from utils.eval import get_eval_metric_results
from tqdm import tqdm
import torch_geometric.nn as nn
import time
from utils.config import INTERMEDIATE_DATA_DIR
import getpass

# %%
# train related
# 走之前一定要检查运行的文件夹
TRAIN_DIR = os.path.join(INTERMEDIATE_DATA_DIR, 'train_intermediate')
VAL_DIR = os.path.join(INTERMEDIATE_DATA_DIR, 'val_intermediate')
gpus = [2, 3]
SEED = 13

decay_lr_factor = 0.5

lr = 0.003
in_channels, out_channels = 8, 60
show_every = 20
val_every = 50  # 单位是epoch
small_dataset = True
if getpass.getuser() == "zhuhe":
    batch_size = 768  # 900是4G显卡的上限
else:
    batch_size = 4096

if small_dataset is True:
    epochs = 3000
    show_every = 50  # 这个和step绑定的
    val_every = 100  # 单位是epoch，现在看来这个test的时间实在是比较长
    decay_lr_every = 1000  # 每多少epoch衰减 # 这些都是要和数据集大小有关的
if getpass.getuser() == "zhuhe" and small_dataset is True:
    small_dataset_train = batch_size * 50 - 1  # processed 大约在14万会卡死，结果6万也会被杀死
    small_dataset_test = batch_size * 10 - 1
else:
    small_dataset_train = -1
    small_dataset_test = -1
end_epoch = 0

best_minade = float('inf')
global_step = 0
date = '20221111'
save_dir = 'trained_params' + date[:4] + "_" + date[4:6] + "_" + date[6:]
# eval related
max_n_guesses = 1
horizon = 30
miss_threshold = 2.0
max_save_files = 5


# %%
def save_checkpoint(checkpoint_dir, model, optimizer, end_epoch, val_minade, date):
    # state_dict: a Python dictionary object that:
    # - for a model, maps each layer to its parameter tensor;
    # - for an optimizer, contains info about the optimizerâ€™s states and hyperparameters used.
    os.makedirs(checkpoint_dir, exist_ok=True)
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'end_epoch': end_epoch,
        'val_minade': val_minade
    }
    filenames = os.listdir(checkpoint_dir)
    if len(filenames) > max_save_files:
        filenames = sorted(filenames, key=lambda x: float(x.split(".")[1].split("_")[1] + "." + x.split(".")[2]))

        os.remove(os.path.join(checkpoint_dir, filenames[-1]))
        print(f"remove file {filenames[-1]}")

    checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{end_epoch}.valminade_{val_minade:.3f}.{date}.{"zly"}.pth')
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)
    # return checkpoint_path['end_epoch']


# %%
if __name__ == "__main__":
    # training envs
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # prepare dara
    if small_dataset is True:
        train_data = GraphDataset(TRAIN_DIR, small_dataset_train).shuffle()
        val_data = GraphDataset(VAL_DIR, small_dataset_test)
    else:
        raise "this will Out of memory"
    if small_dataset is True:
        train_loader = DataLoader(train_data[:small_dataset_train], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data[:small_dataset_test], batch_size=batch_size)
    else:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)

    model = HGNN(in_channels, out_channels)
    print(model.eval())
    # model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter('tensorboard/'+ date)
    # data_sample = next(iter(train_loader))
    # writer.add_graph(model,data_sample) # 这2行确实跑不了，因为需要额外的插件
    model = model.to(device=device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=decay_lr_every, gamma=decay_lr_factor)

    load_checkpoint("./trained_params2022_11_11/epoch_2499.valminade_2.335.20221111.zly.pth", model, optimizer)

    # training loop
    model.train()
    for epoch in range(epochs):
        acc_loss = .0
        num_samples = 0
        start_tic = time.time()
        for data in train_loader:
            data.to(device)
            if epoch < end_epoch: break
            # y = torch.cat([i.y for i in data], 0).view(-1, out_channels).to(device)
            # print(data.y)
            y = torch.cat([data.y], 0).view(-1, out_channels).to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.mse_loss(out, y)
            loss.backward()
            acc_loss += batch_size * loss.item()
            num_samples += y.shape[0]
            optimizer.step()
            global_step += 1
            if (global_step + 1) % show_every == 0:
                print(
                    f"loss at epoch {epoch} step {global_step}:{loss.item():3f}, lr:{optimizer.state_dict()['param_groups'][0]['lr']: .6f}, time:{time.time() - start_tic: 4f}sec")
        scheduler.step()
        print(
            f"loss at epoch {epoch}:{acc_loss / num_samples:.3f}, lr:{optimizer.state_dict()['param_groups'][0]['lr']: .6f}, time:{time.time() - start_tic: 4f}sec")
        writer.add_scalar('training loss', acc_loss / num_samples, epoch)

        writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], epoch)

        writer.add_scalar("spend time", time.time() - start_tic, epoch)

        if (epoch + 1) % val_every == 0 and (not epoch < end_epoch):
            print("eval as epoch:{epoch}")
            metrics = get_eval_metric_results(model, val_loader, device, out_channels, max_n_guesses, horizon,
                                              miss_threshold, whether_plot=False)  # 调用这个函数可能非常花时间
            curr_minade = metrics["minADE"]
            print(f"minADE:{metrics['minADE']:3f}, minFDE:{metrics['minFDE']:3f}, MissRate:{metrics['MR']:3f}")
            writer.add_scalar('test minADE', metrics['minADE'], epoch // val_every)
            writer.add_scalar('test minFDE', metrics['minFDE'], epoch // val_every)
            writer.add_scalar('test MissRate', metrics['MR'], epoch // val_every)

            if curr_minade < best_minade:
                best_minade = curr_minade
                save_checkpoint(save_dir, model, optimizer, epoch, best_minade, date)

    # eval result on the identity dataset
    metrics = get_eval_metric_results(model, val_loader, device, out_channels,
                                      max_n_guesses, horizon, miss_threshold, whether_plot=False)
    curr_minade = metrics["minADE"]
    if curr_minade < best_minade:
        best_minade = curr_minade
        save_checkpoint(save_dir, model, optimizer, -1, best_minade, date)

    writer.close()

# %%
