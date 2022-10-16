import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch
import numpy as np


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False, whether_dropout=False):
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

        self.softmax = torch.nn.LogSoftmax()
        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    # def forward(self, x, edge_index,edge_attr):
    #     # x = x.to(device) # 在模型to_divice很慢？
    #     edge_attr = edge_attr.to(device)
    #     edge_index = edge_index.to(device)
    #     out = None
    #     for layer in range(len(self.convs)-1):  #layer：层数
    #         x=self.convs[layer](x,edge_index,edge_attr).to(torch.float)   #叠GCNConv
    #         # x= x.to(torch.float)# 这个是因为他这的输出搞成了float64,导致大家数据形式不兼容
    #         x=F.relu(x)  #叠relu,这个不会导致数据被转化成float,
    #         if self.whether_dropout is True:
    #             x=F.dropout(x,self.dropout,self.training)  #叠dropout。这个self.dropout看下文是概率。
    #     #最后一层
    #     out=self.convs[-1](x,edge_index,edge_attr)  #GCNVonv
    #     if not self.return_embeds:
    #         out=self.softmax(out)
    def forward(self, batch_data):
        out = None
        x = batch_data.x
        for layer in range(len(self.convs) - 1):  # layer：层数
            x = self.convs[layer](x, batch_data.edge_index, batch_data.edge_attr).to(torch.float)  # 叠GCNConv

            # x= x.to(torch.float)# 这个是因为他这的输出搞成了float64,导致大家数据形式不兼容
            x = F.relu(x)  # 叠relu,这个不会导致数据被转化成float,
            if self.whether_dropout is True:
                x = F.dropout(x, self.dropout, self.training)  # 叠dropout。这个self.dropout看下文是概率。
        # 最后一层

        out = self.convs[-1](x, batch_data.edge_index, batch_data.edge_attr)  # GCNVonv
        if not self.return_embeds:
            out = self.softmax(out)

        return out


def train_with_batch(model, device, data_loader, optimizer, loss_fn):
    model.train()
    loss = 0
    loss_all = 0
    length = 0
    # for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
    for batch in data_loader:
        length = length + 1
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            optimizer.zero_grad()
            out = model(batch)
            train_output = out.view([-1, 50, 2]).to(device)
            train_label = batch.y[:, :, 0:2]
            train_availabilities = torch.squeeze(batch.y[:, :, 0:1], dim=-1)
            loss = loss_fn(train_label, train_output, train_availabilities)
            loss_all = loss_all + loss.item()
            loss.backward()
            optimizer.step()

    return loss_all / length


@torch.no_grad()
def test_with_batch(model, device, data_loader, loss_fn):
    model.eval()
    y_true = []
    y_pred = []
    loss_array = []

    # for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
    for batch in data_loader:
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            out = model(batch)
            train_output = out.view([-1, 50, 2]).to(device)
            train_label = batch.y[:, :, 0:2]
            train_availabilities = torch.squeeze(batch.y[:, :, 0:1], dim=-1)
            loss = loss_fn(train_label, train_output, train_availabilities)
            y_true.append(train_label.detach().cpu())
            y_pred.append(train_output.detach().cpu())
            loss_array.append(loss.item())

    loss_array = np.array(loss_array)

    return y_true, y_pred, loss_array
