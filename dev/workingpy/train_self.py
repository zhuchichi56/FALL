import DataLoader_self
import torch
import numpy as np
import model
import metric_self

args = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # 'device' = 'cpu'
    'num_layers': 2,
    'hidden_dim': 256,
    'dropout': 0,
    'lr': 0.001,
    'epochs': 1000,
    'train_scene_start': 10,  # 这个该有多少，还要好好想想，因为我们存储的数据并不多，主要是egodataset的load速度那边被限制住了
    'train_scene_end': 100,
    'test_scene_start': 10,
    'test_scene_end': 30,
    'batch_size_gcn': 512,
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


from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def convert_framearray_to_dataloader(frame_dic_array):
    pyg_data_list = []
    for ele in frame_dic_array:
        pyg_data_list.append(Data(x=ele["x"], edge_index=ele["edge_index"], edge_attr=ele["edge_attr"], y=torch.cat(
            [ele["target_positions"], torch.unsqueeze(ele["target_availabilities"], dim=-1)], dim=-1)))
    return DataLoader(pyg_data_list, batch_size=args['batch_size_gcn'])


def load_all_data_into_memory(ego_dataset, agent_dataset, start_index, end_index):
    return_array = []
    for index_ in range(start_index, end_index):
        scene_ego_dataset = ego_dataset.get_scene_dataset(index_)
        scene_agent_dataset = agent_dataset.get_scene_dataset(index_)
        frame_dic_array = transform_one_scene_dataset(scene_ego_dataset, scene_agent_dataset, debug=False)
        dataloader = convert_framearray_to_dataloader(frame_dic_array)
        return_array.append(dataloader)
    return return_array


def prepare_dataloader_array():
    train_dataloader = load_all_data_into_memory(train_ego_dataset, train_agent_dataset, args['train_scene_start'],
                                                 args['train_scene_end'])
    test_dataloader = load_all_data_into_memory(test_ego_dataset, test_agent_dataset, args['test_scene_start'],
                                                args['test_scene_end'])
    return train_dataloader, test_dataloader


if __name__ == '__main__':
    train_ego_dataset, train_agent_dataset, test_ego_dataset, test_agent_dataset = DataLoader_self.get_dataset()
    scene_ego_dataset = train_ego_dataset.get_scene_dataset(60)
    scene_agent_dataset = train_agent_dataset.get_scene_dataset(60)
    frame_dic_array = transform_one_scene_dataset(scene_ego_dataset, scene_agent_dataset, debug=False)
    from tqdm import tqdm, trange

    input_dim = len(frame_dic_array[0]["x"][0])
    model_gcn = model.GCN(input_dim=input_dim, hidden_dim=args["hidden_dim"],
                          output_dim=DataLoader_self.cfg["model_params"]["future_num_frames"] * 2,
                          num_layers=args["num_layers"],
                          dropout=args["dropout"], return_embeds=True, whether_dropout=False)

    model_gcn = model_gcn.to(device=args['device'])

    print(model_gcn.eval())

    optimizer = torch.optim.Adam(model_gcn.parameters(), lr=args["lr"])
    train_dataloader, test_dataloader = prepare_dataloader_array()
    train_file = open('train_output.txt', 'w+', encoding='utf-8')
    test_file = open('test_output.txt', 'w+', encoding='utf-8')
    for epoch in trange(1, 1 + args["epochs"]):
        loss_whole_scene_list = []
        loss_this_epoch = 0
        # for train_scene_index in trange(0,args['train_scene_end']-args['train_scene_start']):
        for train_scene_index in range(0, args['train_scene_end'] - args['train_scene_start']):
            dataloader = train_dataloader[train_scene_index]

            loss_each_frame = model.train_with_batch(model_gcn, args["device"], dataloader, optimizer,
                                                     metric_self.pytorch_neg_multi_log_likelihood_single)
            loss_this_scene = np.mean(np.array(loss_each_frame))
            loss_this_epoch = loss_this_epoch + loss_this_scene
        loss_whole_scene_list.append(loss_this_epoch)
        print(f"train loss this epoch{loss_this_epoch}")


        if epoch % 10 == 1:
            loss_list = []
            for test_scene_index in range(0, args['test_scene_end'] - args['test_scene_start']):
                dataloader = test_dataloader[test_scene_index]
                y_true, y_pred, loss = model.test_with_batch(model_gcn, args["device"], dataloader,
                                                             metric_self.pytorch_neg_multi_log_likelihood_single)
                loss_list.append(loss)
            loss_list = np.array(loss_list)
            test_file.write(str(loss_list.mean()))
            print(str(loss_list.mean()))
            test_file.write("\n")

    test_file.close()
