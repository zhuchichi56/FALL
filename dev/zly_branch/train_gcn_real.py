#!/usr/bin/env python
# coding: utf-8

# # Lyft: Training with multi-mode confidence
# 
# ![](http://www.l5kit.org/_images/av.jpg)
# <cite>The image from L5Kit official document: <a href="http://www.l5kit.org/README.html">http://www.l5kit.org/README.html</a></cite>
# 
# Continued from the previous kernel:
#  - [Lyft: Comprehensive guide to start competition](https://www.kaggle.com/corochann/lyft-comprehensive-guide-to-start-competition)
#  - [Lyft: Deep into the l5kit library](https://www.kaggle.com/corochann/lyft-deep-into-the-l5kit-library)
# 
# In this kernel, I will run **pytorch CNN model training**. Especially, followings are new items to try:
#  - Predict **multi-mode with confidence**: As written in [evaluation metric](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/overview/evaluation) page, we can predict **3 modes** of motion trajectory.
#  - Training loss with **competition evaluation metric**
#  - Use Training abstraction library **`pytorch-ignite` and `pytorch-pfn-extras`**.
# 
# 
# [Update 2020/9/6]<br/>
# Published prediction kernel: [Lyft: Prediction with multi-mode confidence](https://www.kaggle.com/corochann/lyft-prediction-with-multi-mode-confidence)<br/>
# Try yourself how good score you can get using only single model without ensemble! :)
#         *

# # Environment setup
# 
#  - Please add [pestipeti/lyft-l5kit-unofficial-fix](https://www.kaggle.com/pestipeti/lyft-l5kit-unofficial-fix) as utility script.
#     - Official utility script "[philculliton/kaggle-l5kit](https://www.kaggle.com/mathurinache/kaggle-l5kit)" does not work with pytorch GPU.
#  - Please add [lyft-config-files](https://www.kaggle.com/jpbremer/lyft-config-files) as dataset
#  
# See previous kernel [Lyft: Comprehensive guide to start competition](https://www.kaggle.com/corochann/lyft-comprehensive-guide-to-start-competition) for details.

# In[1]:


# https://github.com/pfnet/pytorch-pfn-extras/releases/tag/v0.3.1
# !pip install pytorch-pfn-extras==0.6.1


# 

# In[2]:


import gc
import os
from pathlib import Path
import random
import sys

from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import scipy as sp


import matplotlib.pyplot as plt
import seaborn as sns

from IPython.core.display import display, HTML

# --- plotly ---
from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
pio.templates.default = "plotly_dark"

# --- models ---
from sklearn import preprocessing
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# --- setup ---
# pd.set_option('max_columns', 50)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'


# In[3]:


import zarr

import l5kit
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset, AgentDataset

from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data
from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR
from l5kit.geometry import transform_points
from tqdm import tqdm
from collections import Counter
from l5kit.data import PERCEPTION_LABELS
from prettytable import PrettyTable

from matplotlib import animation, rc
from IPython.display import HTML

rc('animation', html='jshtml')
print("l5kit version:", l5kit.__version__)


# In[4]:


import torch
from pathlib import Path

import pytorch_pfn_extras as ppe
from math import ceil
from pytorch_pfn_extras.training import IgniteExtensionsManager
from pytorch_pfn_extras.training.triggers import MinValueTrigger

from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import pytorch_pfn_extras.training.extensions as E


# In[5]:


# !pip upgrea


# In[6]:


# --- Dataset utils ---
from typing import Callable

from torch.utils.data.dataset import Dataset


class TransformDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: Callable):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        batch = self.dataset[index]
        return self.transform(batch)

    def __len__(self):
        return len(self.dataset)


# ## Function
# 
# To define loss function to calculate competition evaluation metric **in batch**.<br/>
# It works with **pytorch tensor, so it is differentiable** and can be used for training Neural Network.

# In[7]:


# --- Function utils ---
# Original code from https://github.com/lyft/l5kit/blob/20ab033c01610d711c3d36e1963ecec86e8b85b6/l5kit/l5kit/evaluation/metrics.py
import numpy as np

import torch
from torch import Tensor


def pytorch_neg_multi_log_likelihood_batch(
    gt: Tensor, pred: Tensor, confidences: Tensor, avails: Tensor
) -> Tensor:
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:
    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    https://leimao.github.io/blog/LogSumExp/
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    assert len(pred.shape) == 4, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    batch_size, num_modes, future_len, num_coords = pred.shape
    # print(f"pred.shape : {pred.shape}")
    assert gt.shape == (batch_size, future_len, num_coords), f"wrong shape for gt, got {gt.shape}"
    assert confidences.shape == (batch_size, num_modes), f"expected 1D (Modes) array for gt, got {confidences.shape}"
    assert torch.allclose(torch.sum(confidences, dim=1), confidences.new_ones((batch_size,))), "confidences should sum to 1"
    assert avails.shape == (batch_size, future_len), f"expected 1D (Time) array for gt, got {avails.shape}"
    # assert all data are valid
    assert torch.isfinite(pred).all(), "invalid value found in pred"
    assert torch.isfinite(gt).all(), "invalid value found in gt"
    assert torch.isfinite(confidences).all(), "invalid value found in confidences"
    assert torch.isfinite(avails).all(), "invalid value found in avails"

    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)  # reduce coords and use availability

    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # reduce time

    # use max aggregator on modes for numerical stability
    # error (batch_size, num_modes)
    max_value, _ = error.max(dim=1, keepdim=True)  # error are negative at this point, so max() gives the minimum one
    error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True)) - max_value  # reduce modes
    # print("error", error)
    return torch.mean(error)


def pytorch_neg_multi_log_likelihood_single(
    gt: Tensor, pred: Tensor, avails: Tensor
) -> Tensor:
    """

    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(time)x(2D coords)
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    # pred (bs)x(time)x(2D coords) --> (bs)x(mode=1)x(time)x(2D coords)
    # create confidence (bs)x(mode=1)
    batch_size, future_len, num_coords = pred.shape
    confidences = pred.new_ones((batch_size, 1))
    return pytorch_neg_multi_log_likelihood_batch(gt, pred.unsqueeze(1), confidences, avails)


# ## Model
# 
# pytorch model definition. Here model outputs both **multi-mode trajectory prediction & confidence of each trajectory**.

# In[8]:


# --- Model utils ---
import torch
from torchvision.models import resnet18
from torch import nn
from typing import Dict



# In[9]:


# --- Utils ---
import yaml


def save_yaml(filepath, content, width=120):
    with open(filepath, 'w') as f:
        yaml.dump(content, f, width=width)


def load_yaml(filepath):
    with open(filepath, 'r') as f:
        content = yaml.safe_load(f)
    return content


class DotDict(dict):
    """dot.notation access to dictionary attributes

    Refer: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary/23689767#23689767
    """  # NOQA

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    


# ## Configs

# In[10]:


# --- Lyft configs ---
cfg = {
    'format_version': 4,
    'model_params': {
        'model_architecture': 'resnet50',
        'history_num_frames': 10,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1,
        'render_ego_history':True,
        'step_time':0.1
    },

    'raster_params': {
        'raster_size': [224, 224],
        'pixel_size': [0.5, 0.5],
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'satellite_map_key': 'aerial_map/aerial_map.png',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        'filter_agents_threshold': 0.5,

        'set_origin_to_bottom': True,
        'disable_traffic_light_faces':False
    },

    'train_data_loader': {
        'key': 'scenes/train.zarr',
        'batch_size': 12,
        'shuffle': True,
        'num_workers': 4
    },

    'valid_data_loader': {
        'key': 'scenes/validate.zarr',
        'batch_size': 32,
        'shuffle': False,
        'num_workers': 4
    },

    'train_params': {
        'max_num_steps': 10000,
        'checkpoint_every_n_steps': 5000,

        # 'eval_every_n_steps': -1
    }
}





# In[11]:


flags_dict = {
    "debug": True,
    # --- Data configs ---
    "l5kit_data_folder": "/home/zhuhe/kaggle/input/lyft-motion-prediction-autonomous-vehicles",
    # --- Model configs ---
    "pred_mode": "multi",
    # --- Training configs ---
    "device": "cuda:0",
    "out_dir": "results/multi_train",
    "epoch": 2,
    "snapshot_freq": 50,
}


# # Main script
# 
# Now finished defining all the util codes. Let's start writing main script to train the model!

# ## Loading data
# 
# Here we will only use the first dataset from the sample set. (sample.zarr data is used for visualization, please use train.zarr / validate.zarr / test.zarr for actual model training/validation/prediction.)<br/>
# We're building a `LocalDataManager` object. This will resolve relative paths from the config using the `L5KIT_DATA_FOLDER` env variable we have just set.

# In[12]:


flags = DotDict(flags_dict)
out_dir = Path(flags.out_dir)
os.makedirs(str(out_dir), exist_ok=True)
print(f"flags: {flags_dict}")
save_yaml(out_dir / 'flags.yaml', flags_dict)
save_yaml(out_dir / 'cfg.yaml', cfg)
debug = flags.debug



# In[13]:


# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = flags.l5kit_data_folder
dm = LocalDataManager(None)

print("Load dataset...")
train_cfg = cfg["train_data_loader"]
valid_cfg = cfg["valid_data_loader"]

# Rasterizer
rasterizer = build_rasterizer(cfg, dm)
train_path = "scenes/train.zarr" if debug else train_cfg["key"]
train_zarr = ChunkedDataset(dm.require(train_path)).open()
print("train_zarr", type(train_zarr))
train_ego_dataset = EgoDataset(cfg, train_zarr, rasterizer)
train_agent_dataset = AgentDataset(cfg, train_zarr, rasterizer)

# valid_path = "scenes/sample.zarr" if debug else valid_cfg["key"]
# valid_zarr = ChunkedDataset(dm.require(valid_path)).open()
# print("valid_zarr", type(train_zarr))
# valid_agent_dataset = AgentDataset(cfg, valid_zarr, rasterizer)
# valid_dataset = TransformDataset(valid_agent_dataset, transform)


# In[14]:


def get_distance(centroid1, centroid2):
    return np.sqrt(np.square(centroid1[0]-centroid2[0]) + np.square(centroid1[1]-centroid2[1]))
# def change_all_list_to_ndarray(dic_in):

def transform_one_scene_dataset(scene_ego_dataset, scene_agent_dataset, debug: False):
    def add_element_into(ele_dict, ele):
        temp_x = []
        temp_x.extend(ele["curr_speed"].flatten())
        temp_x.extend(ele["history_positions"].flatten())
        temp_x.extend(ele["history_yaws"].flatten())
        temp_x.extend(ele["history_availabilities"].flatten())
        ele_dict["x"].append(temp_x)
        if len(ele_dict["index"])== 0:
            cur_index = 0
        else:
            cur_index = ele_dict["index"][-1] + 1
        for i in range(0,cur_index):
            ele_dict["edge_attr"].append(get_distance(ele["centroid"], ele_dict["centroid"][i])) # 双向图
            ele_dict["edge_index"][0].append(i)
            ele_dict["edge_index"][1].append(cur_index)
            ele_dict["edge_attr"].append(get_distance(ele["centroid"], ele_dict["centroid"][i]))
            ele_dict["edge_index"][1].append(i)
            ele_dict["edge_index"][0].append(cur_index)
        ele_dict["target_positions"].append(ele["target_positions"])
        ele_dict["target_availabilities"].append(ele["target_availabilities"])
        ele_dict["centroid"].append(ele["centroid"])  # 这个元素是为了建图方便
        ele_dict["index"].append(cur_index)  # 这个元素是为了建图方便

    return_np = [] # 这个的长度应该是247或者248的样子，并且这个就是frame_index

    for ele in scene_ego_dataset:
        ele_dict = {}
        for name in ["x","edge_index","edge_attr","centroid","index","target_positions","target_availabilities"]:
            if name == "edge_index":
                ele_dict[name] = [[],[]]
            else:
                ele_dict[name] = []
        add_element_into(ele_dict,ele)
        return_np.append(ele_dict)
    if debug:
        print(f"len(return_np) : {len(return_np)}")
    for ele in scene_agent_dataset:
        curr_frame_index = ele["frame_index"]
        if debug:
            print(ele["frame_index"])
        cur_dict = return_np[curr_frame_index]
        add_element_into(cur_dict, ele)

    for ele in return_np:
        for key_ in ele.keys():
            ele[key_] = np.array(ele[key_])

    return return_np


# In[15]:


# frame_dic_array = transform_one_scene_dataset(scene_ego_dataset,scene_agent_dataset)


# In[16]:


# print(frame_dic_array[0])


# ## Prepare model & optimizer

# ## Write training code
# 
# pytorch-ignite & pytorch-pfn-extras are used here.
# 
#  - [pytorch/ignite](https://github.com/pytorch/ignite): It provides abstraction for writing training loop.
#  - [pfnet/pytorch-pfn-extras](https://github.com/pfnet/pytorch-pfn-extras): It provides several "extensions" useful for training. Useful for **logging, printing, evaluating, saving the model, scheduling the learning rate** during training.
#  
# **[Note] Why training abstraction library is used?**
# 
# You may feel understanding training abstraction code below is a bit unintuitive compared to writing "raw" training loop.<br/>
# The advantage of abstracting the code is that we can re-use implemented handler class for other training, other competition.<br/>
# You don't need to write code for saving models, logging training loss/metric, show progressbar etc.
# These are done by provided util classes in `pytorch-pfn-extras` library!
# 
# You may refer my other kernel in previous competition too: [Bengali: SEResNeXt training with pytorch](https://www.kaggle.com/corochann/bengali-seresnext-training-with-pytorch)

# In[17]:


import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False, whether_dropout= False):
        # TODO: Implement this function that initializes self.convs,
        # self.bns, and self.softmax.

        super(GCN, self).__init__()

        # A list of GCNConv layers
        self.convs = None

        # A list of 1D batch normalization layers
        self.bns = None

        # The log softmax layer
        self.softmax = None
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()

        for i in range(num_layers - 1):
            self.convs.append(GCNConv(input_dim, hidden_dim))
            input_dim = hidden_dim

        self.convs.append(GCNConv(input_dim, output_dim))
        self.whether_dropout = whether_dropout

        self.softmax=torch.nn.LogSoftmax()
        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    # def reset_parameters(self):
    #     for conv in self.convs:
    #         conv.reset_parameters()
    #     for bn in self.bns:
    #         bn.reset_parameters()

    def forward(self, x, edge_index,edge_attr):
        # TODO: Implement this function that takes the feature tensor x,
        # edge_index tensor adj_t and returns the output tensor as
        # shown in the figure.
        x = x.to(device)
        edge_attr = edge_attr.to(device)
        edge_index = edge_index.to(device)
        out = None
        for layer in range(len(self.convs)-1):  #layer：层数
            x=self.convs[layer](x,edge_index,edge_attr).to(torch.float)   #叠GCNConv
            # x= x.to(torch.float)# 这个是因为他这的输出搞成了float64,导致大家数据形式不兼容
            x=F.relu(x)  #叠relu,这个不会导致数据被转化成float,
            if self.whether_dropout is True:
                x=F.dropout(x,self.dropout,self.training)  #叠dropout。这个self.dropout看下文是概率。
        #最后一层
        out=self.convs[-1](x,edge_index,edge_attr)  #GCNVonv
        if not self.return_embeds:
            out=self.softmax(out)

        return out




# In[18]:


args = {
    'num_layers': 2,
    'hidden_dim': 256,
    'dropout': 0,
    'lr': 0.001,
    'epochs': 30,
}


# In[18]:





# In[19]:


scene_ego_dataset = train_ego_dataset.get_scene_dataset(0)
scene_agent_dataset = train_agent_dataset.get_scene_dataset(0)
frame_dic_array = transform_one_scene_dataset(scene_ego_dataset,scene_agent_dataset,debug=False)

# print(type(frame_dic_array[0]['edge_attr']))


# In[29]:


scene_ego_dataset[100].keys()


# In[20]:


scene_ego_dataset


# In[21]:


from tqdm import tqdm,trange

input_dim = len(frame_dic_array[0]["x"][0])
model_gcn = GCN(input_dim=input_dim, hidden_dim=args["hidden_dim"], output_dim = cfg["model_params"]["future_num_frames"] * 2,num_layers=args["num_layers"],dropout=args["dropout"],return_embeds = True, whether_dropout=False)

model_gcn = model_gcn.to(device=device)

print(model_gcn.eval())
epoch = flags.epoch

optimizer = torch.optim.Adam(model_gcn.parameters(), lr=args["lr"])

def train(model, value_dic, train_idx, optimizer, loss_fn):
    model.train()
    loss = 0
    optimizer.zero_grad()

    out=model(torch.tensor(value_dic["x"]),edge_index = torch.tensor(value_dic["edge_index"],dtype=torch.long), edge_attr = torch.tensor(value_dic["edge_attr"]))
    if len(out.shape) <= 2:
        out = torch.unsqueeze(out,0)

    train_output=out.view([-1,50,2]).to(device)  # 这里暂时是全部训练
    train_label= torch.tensor(value_dic["target_positions"], dtype = torch.float).to(device)
    train_availabilities = torch.tensor(value_dic["target_availabilities"],dtype= torch.int).to(device)
    loss=loss_fn(train_label,train_output,train_availabilities) # 只预测一条路
    loss.backward()
    optimizer.step()

    return loss.item()

for epoch in trange(1, 1 + args["epochs"]):
    loss_whole_scene_list = []
    loss_this_epoch = 0
    for scene_index in trange(0,99):
        scene_ego_dataset = train_ego_dataset.get_scene_dataset(scene_index)
        scene_agent_dataset = train_agent_dataset.get_scene_dataset(scene_index)
        frame_dic_array = transform_one_scene_dataset(scene_ego_dataset,scene_agent_dataset,debug=False)
        print(frame_dic_array)
        break
        loss_each_frame = []

        index = 0
        for each_frame_dic in frame_dic_array:
            index = index + 1
            loss = train(model_gcn, each_frame_dic, [], optimizer , pytorch_neg_multi_log_likelihood_single)
            loss_each_frame.append(loss)
        loss_this_scene = np.mean(np.array(loss_each_frame))
        loss_this_epoch = loss_this_epoch + loss_this_scene
    print(loss_this_epoch)






# In[ ]:


print(frame_dic_array[11])


# 
# 
# 
# 

# You can obtrain training history results really easily by just accessing `LogReport` class, which is useful for managing a lot of experiments during kaggle competitions.

# The history log and model's weight are saved by "extensions" (`LogReport` and `E.snapshot_object` respectively) easily, which is a benefit of using training abstration.

# In[ ]:


# Let's see training results directory

get_ipython().system('ls results/multi_train')


# # Items to try
# 
# This kernel shows demonstration run of the training (`debug=True`). You can try these things to see how the score changes at first
#  - set debug=False to train with actual training dataset
#  - change training hyperparameters (training epoch, change optimizer, scheduler learning rate etc...)
#    - Especially, just training much longer time may improve the score.
#  
# To go further, these items may be nice to try:
#  - Change the cnn model (now simple resnet18 is used as baseline modeling)
#  - Training the model using full dataset: [lyft-full-training-set](https://www.kaggle.com/philculliton/lyft-full-training-set)
#  - Write your own rasterizer to prepare input image as motivation explained in previous kernel.
#  - Consider much better scheme to predict multi-trajectory
#     - The model just predicts multiple trajectory at the same time in this kernel, but it is possible to collapse "trivial" solution where all trajectory converges to same. How to avoid this?

# # Next to go
# 
# [Update 2020/9/6]<br/>
# Published prediction kernel: [Lyft: Prediction with multi-mode confidence](https://www.kaggle.com/corochann/lyft-prediction-with-multi-mode-confidence)<br/>
# Try yourself how good score you can get using only single model without ensemble! :)
# 
# To understand the competition in more detail, please refer my other kernels too.
#  - [Lyft: Comprehensive guide to start competition](https://www.kaggle.com/corochann/lyft-comprehensive-guide-to-start-competition)
#  - [Lyft: Deep into the l5kit library](https://www.kaggle.com/corochann/lyft-deep-into-the-l5kit-library)
#  - [Save your time, submit without kernel inference](https://www.kaggle.com/corochann/save-your-time-submit-without-kernel-inference)
#  - [Lyft: pytorch implementation of evaluation metric](https://www.kaggle.com/corochann/lyft-pytorch-implementation-of-evaluation-metric)

# # Further reference
# 
#  - Paper of this Lyft Level 5 prediction dataset: [One Thousand and One Hours: Self-driving Motion Prediction Dataset](https://arxiv.org/abs/2006.14480)
#  - [jpbremer/lyft-scene-visualisations](https://www.kaggle.com/jpbremer/lyft-scene-visualisations)

# <h3 style="color:red">If this kernel helps you, please upvote to keep me motivated :)<br>Thanks!</h3>

# In[ ]:




