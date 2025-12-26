import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
from models.beltrami import BELTRAMI
from models.MeanCurv import MEANCURV
import numpy as np

class BeltramiDiffusion(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=4, dropout=0.5, input_dim=None):
        super(BeltramiDiffusion, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_dim = input_dim

        # 初始化特征转换层
        if input_dim is not None:
            self.feature_transform = nn.Linear(input_dim, hidden_dim)

        # 使用 BELTRAMI 和 MEAN CURVATURE FLOW
        self.beltrami_layer = BELTRAMI(
            in_features=hidden_dim,
            out_features=hidden_dim,
            hidden_features=hidden_dim,
            n_layers=num_layers,
            adj_norm_func=None,
            dropout=dropout
        )
        self.meancurv_layer = MEANCURV(
            in_features=hidden_dim,
            out_features=hidden_dim,
            hidden_features=hidden_dim,
            n_layers=num_layers,
            adj_norm_func=None,
            dropout=dropout
        )

    def forward(self, x, coords, input_dim):
        # 获取输入张量的设备
        device = x.device

        # 根据坐标生成邻接矩阵
        adj = self.create_adjacency_matrix(coords, k=5)

        # 确保邻接矩阵在与特征相同的设备上并且是稀疏格式
        adj = adj.to_sparse().to(device)

        # 动态初始化特征转换层，将输入特征映射到隐藏层维度
        self.feature_transform = nn.Linear(input_dim, self.hidden_dim).to(device)
        # 添加 InstanceNorm1d 层，动态根据输入的特征维度设置 num_features
        self.instance_norm = nn.InstanceNorm1d(num_features=self.hidden_dim, affine=False).to(device)
        # 调整输入维度以适应特征转换
        batch_size, feature_dim, num_nodes = x.shape
        x = x.permute(0, 2, 1).contiguous().view(-1, feature_dim)  # 调整维度为 (batch_size * num_nodes, feature_dim)

        # 通过特征转换层，将输入特征映射到隐藏层维度
        x = self.feature_transform(x)
        x = F.relu(x)

        # 将特征恢复到 (batch_size, num_nodes, hidden_dim)
        x = x.view(batch_size, num_nodes, self.hidden_dim)

        # 调整 x 以适应 BELTRAMI FLOW 和 MEAN CURVATURE FLOW 的输入格式
        x = x.permute(1, 0, 2).contiguous().view(-1, self.hidden_dim)  # 调整为 (num_nodes * batch_size, hidden_dim)

        # 第一个扩散模块：BELTRAMI FLOW
        x = self.beltrami_layer(x, adj)
        x = F.relu(x)

        # 第二个扩散模块：MEAN CURVATURE FLOW
        x = self.meancurv_layer(x, adj)
        x = F.relu(x)

        # 最后再恢复为 (batch_size, hidden_dim, num_nodes)
        x = x.view(batch_size, num_nodes, self.hidden_dim).permute(0, 2, 1).contiguous()

        return x

    def create_adjacency_matrix(self, coords, k=5):
        # 使用 sklearn 生成 KNN 图的邻接矩阵
        num_nodes = coords.shape[1]
        adj = kneighbors_graph(coords.T.cpu().numpy(), n_neighbors=min(k, num_nodes), mode='connectivity',
                               include_self=True)
        # 转为 COO 格式
        adj_coo = adj.tocoo()
        indices = torch.tensor(np.array([adj_coo.row, adj_coo.col]), dtype=torch.long, device=coords.device)
        values = torch.tensor(adj_coo.data, dtype=torch.float32, device=coords.device)
        adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes), device=coords.device)
        return adj

