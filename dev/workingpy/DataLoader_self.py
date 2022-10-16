import os

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

import torch
from matplotlib import animation, rc
import torch
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader

rc('animation', html='jshtml')
print("l5kit version:", l5kit.__version__)

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
        'render_ego_history': True,
        'step_time': 0.1
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
        'disable_traffic_light_faces': False
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


def get_distance(centroid1, centroid2):
    return np.sqrt(np.square(centroid1[0] - centroid2[0]) + np.square(centroid1[1] - centroid2[1]))


# def change_all_list_to_ndarray(dic_in):

def transform_one_scene_dataset(scene_ego_dataset, scene_agent_dataset, debug: False):
    def add_element_into(ele_dict, ele):
        temp_x = []
        temp_x.extend(ele["curr_speed"].flatten())
        temp_x.extend(ele["history_positions"].flatten())
        temp_x.extend(ele["history_yaws"].flatten())
        temp_x.extend(ele["history_availabilities"].flatten())
        ele_dict["x"].append(temp_x)
        if len(ele_dict["index"]) == 0:
            cur_index = 0
        else:
            cur_index = ele_dict["index"][-1] + 1
        for i in range(0, cur_index):
            ele_dict["edge_attr"].append(get_distance(ele["centroid"], ele_dict["centroid"][i]))  # 双向图
            ele_dict["edge_index"][0].append(i)
            ele_dict["edge_index"][1].append(cur_index)
            ele_dict["edge_attr"].append(get_distance(ele["centroid"], ele_dict["centroid"][i]))
            ele_dict["edge_index"][1].append(i)
            ele_dict["edge_index"][0].append(cur_index)
        ele_dict["target_positions"].append(ele["target_positions"])
        ele_dict["target_availabilities"].append(ele["target_availabilities"])
        ele_dict["centroid"].append(ele["centroid"])  # 这个元素是为了建图方便
        ele_dict["index"].append(cur_index)  # 这个元素是为了建图方便

    return_np = []  # 这个的长度应该是247或者248的样子，并且这个就是frame_index

    for ele in scene_ego_dataset:
        ele_dict = {}
        for name in ["x", "edge_index", "edge_attr", "centroid", "index", "target_positions", "target_availabilities"]:
            if name == "edge_index":
                ele_dict[name] = [[], []]
            else:
                ele_dict[name] = []
        add_element_into(ele_dict, ele)
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

        ele["x"] = torch.tensor(ele["x"])
        ele["edge_index"] = torch.tensor(ele["edge_index"], dtype=torch.long)
        ele["edge_attr"] = torch.tensor(ele["edge_attr"])
        ele["target_positions"] = torch.tensor(ele["target_positions"])
        ele["target_availabilities"] = torch.tensor(ele["target_availabilities"])

    return return_np


def get_dataset():
    flags = DotDict(flags_dict)
    out_dir = Path(flags.out_dir)
    os.makedirs(str(out_dir), exist_ok=True)
    print(f"flags: {flags_dict}")
    save_yaml(out_dir / 'flags.yaml', flags_dict)
    save_yaml(out_dir / 'cfg.yaml', cfg)
    debug = flags.debug
    os.environ["L5KIT_DATA_FOLDER"] = flags.l5kit_data_folder
    dm = LocalDataManager(None)

    print("Load dataset...")
    train_cfg = cfg["train_data_loader"]
    valid_cfg = cfg["valid_data_loader"]

    # Rasterizer
    rasterizer = build_rasterizer(cfg, dm)
    train_path = "scenes/train.zarr" if debug else train_cfg["key"]
    test_path = "scenes/test.zarr"
    train_zarr = ChunkedDataset(dm.require(train_path)).open()
    test_zarr = ChunkedDataset(dm.require(test_path)).open()
    print("train_zarr", type(train_zarr))
    print("test_zarr", type(test_zarr))
    train_ego_dataset = EgoDataset(cfg, train_zarr, rasterizer)
    train_agent_dataset = AgentDataset(cfg, train_zarr, rasterizer)
    test_ego_dataset = EgoDataset(cfg, test_zarr, rasterizer)
    test_agent_dataset = AgentDataset(cfg, test_zarr, rasterizer)
    return train_ego_dataset, train_agent_dataset, test_ego_dataset, test_agent_dataset
