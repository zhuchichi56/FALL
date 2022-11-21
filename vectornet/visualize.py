# 从vectornet那个文件过来的
from modeling.predmlp import TrajPredMLP
from modeling.selfatten import SelfAttentionLayer
from modeling.subgraph import SubGraph
import os
import pdb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch_geometric.nn import MessagePassing, max_pool
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch_geometric.data import Data, DataLoader
from utils.viz_utils import show_predict_result
from dataset import GraphDataset
import sys
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from vectornet.modeling.vectornet import HGNN
from train import load_checkpoint
from argoverse.visualization.visualize_sequences import viz_sequence
import cv2
def get_data_path_ls(dir_):
    return [os.path.join(dir_, data_path) for data_path in os.listdir(dir_)]


# data preparation
arg_test = {
    'batch_size': 32,
    'how_many_visualize': 1000
}
in_channels, out_channels = 8, 60
if __name__ == "__main__":
    property_ = "train"

    DIR = '/home/zhuhe/Dataset/interm_data/' + property_ + '_intermediate/'
    model = HGNN(in_channels, out_channels)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    load_checkpoint("./trained_params2022_11_11/epoch_2499.valminade_2.335.20221111.zly.pth", model, optimizer)
    model = model.to('cuda:0')
    model.eval()
    data_path_ls = get_data_path_ls(DIR)
    data_path_ls = sorted(data_path_ls)

    data_path_ls = data_path_ls[:arg_test['how_many_visualize']]
    dataset = GraphDataset(DIR, max_load_num=arg_test['how_many_visualize'])

    data_loader = DataLoader(dataset, batch_size=arg_test['batch_size'])
    out_ls = []
    # 初始化
    root_dir = "/home/zhuhe/Dataset/data/train"
    afl = ArgoverseForecastingLoader(root_dir)

    for data in data_loader:
        data = data.to('cuda:0')
        out_ls.append(model(data).cpu().detach())
    with torch.no_grad():
        accum_loss = .0
        for which_to_visual, data_p in enumerate(data_path_ls):
            print(f"sample id: {which_to_visual}")
            data = pd.read_pickle(data_path_ls[which_to_visual])
            print(data)


            print(f"real pkl {data_path_ls[which_to_visual]}")
            y = data['GT'].values[0].reshape(-1).astype(np.float32)
            y = torch.from_numpy(y)

            out = out_ls[which_to_visual // arg_test['batch_size']][
                which_to_visual % arg_test['batch_size']]  # 第一个batch的第一个
            loss = F.mse_loss(out, y)

            accum_loss += loss.item()
            print(f"loss for sample {which_to_visual}: {loss.item():.3f}")
            if loss < 1:
                image_files = []
                real_index = int(data_p.split("_")[-1].split(".")[0])
                which_to_vis = real_index
                seq_path = f"{root_dir}/{which_to_vis}.csv"
                viz_sequence(afl.get(seq_path).seq_df, show=False)
                plt.savefig("./visualize/result/" + f"{property_}/{which_to_visual}_scene.png")
                img1 = cv2.imread("./visualize/result/" + f"{property_}/{which_to_visual}_scene.png")
                plt.show()
                show_predict_result(data, out, y, data['TARJ_LEN'].values[0])
                plt.savefig("./visualize/result/" + f"{property_}/{which_to_visual}_result.png")
                plt.show()  # 清空
                img2 = cv2.imread("./visualize/result/" + f"{property_}/{which_to_visual}_result.png")
                img2 = cv2.resize(img2, [img1.shape[1], img1.shape[0]])
                im_h = cv2.hconcat([img1, img2])
                cv2.imwrite("./visualize/result/" + f"{property_}/{which_to_visual}.png", im_h)

    print(f"eval overall loss: {accum_loss / len(data_path_ls):.3f}")
# print(y)

# edge_index = torch.tensor(
#     [[1, 2, 0, 2, 0, 1],
#         [0, 0, 1, 1, 2, 2]], dtype=torch.long)
# x = torch.tensor([[3, 1, 2], [2, 3, 1], [1, 2, 3]], dtype=torch.float)
# y = torch.tensor([1, 2, 3, 4], dtype=torch.float)
# data = Data(x=x, edge_index=edge_index, y=y)

# data = pd.read_pickle('./input_data/features_4791.pkl')
# all_in_features_, y = data['POLYLINE_FEATURES'].values[0], data['GT'].values[0].reshape(-1).astype(np.float32)
# traj_mask_, lane_mask_ = data["TRAJ_ID_TO_MASK"].values[0], data['LANE_ID_TO_MASK'].values[0]
# y = torch.from_numpy(y)
# in_channels, out_channels = all_in_features_.shape[1], y.shape[0]
# print(f"all feature shape: {all_in_features_.shape}, gt shape: {y.shape}")
# print(f"len of trajs: {traj_mask_}, len of lanes: {lane_mask_}")
