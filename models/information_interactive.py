import copy
import torch
import torch.nn as nn
from utils import square_dists, gather_points, sample_and_group, angle
from models.hilbert import encode as hilbert_encode
from models.z_order import xyz2key
import numpy as np
from flash_attn import flash_attn_varlen_qkvpacked_func
import torch.nn.functional as F
class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        # 定义用于计算 Query, Key 和 Value 的线性层
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        # 输出层
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        # 假设输入为二维张量 (B * N * k, M * 2C)
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 分成 Q, K, V
        q, k, v = qkv  # 直接得到 Q, K, V，不再解包为三维张量
        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B * N * k, N)
        attn = attn.softmax(dim=-1)
        # 加权求和
        out = attn @ v  # (B * N * k, 2C)
        return self.to_out(out)

def get_graph_features(feats, coords, k=10, methods=['hilbert', 'z_order'], shuffle_orders=False):
    '''
    :param feats: (B, N, C), 输入的点云特征，形状为 (批次大小, 点数, 特征数)。
    :param coords: (B, N, 3), 点云坐标，形状为 (批次大小, 点数, 坐标维度)。
    :param k: int, 表示要构建邻域的大小。
    :param methods: list, 序列化方法列表，例如 ['hilbert', 'z_order']，可以包含多个不同的序列化方式。
    :param shuffle_orders: bool, 是否对序列化的结果进行随机打乱。
    :return: (B, N, k, 2C), 输出融合后的邻域特征。
    '''
    device = feats.device
    B, N, C = feats.shape
    all_sorted_indices = []
    all_features = []

    # 1. 使用多种序列化方式对点云进行编码
    for method in methods:
        if method == 'hilbert':
            # 使用 Hilbert 曲线进行编码
            hilbert_indices = []
            for b in range(B):
                coords_tensor = coords[b]  # 确保 coords 是张量
                hilbert_indices.append(hilbert_encode(coords_tensor, num_dims=3, num_bits=10))
            hilbert_indices = torch.stack(hilbert_indices, dim=0)  # (B, N)
            sorted_indices = torch.argsort(hilbert_indices, dim=1)
        elif method == 'Trans_hilbert':
            # 使用 Hilbert 曲线进行编码
            hilbert_indices = []
            for b in range(B):
                coords_tensor = coords[b, :, [1, 0, 2]]  # 确保 coords 是张量
                hilbert_indices.append(hilbert_encode(coords_tensor, num_dims=3, num_bits=10))
            hilbert_indices = torch.stack(hilbert_indices, dim=0)  # (B, N)
            sorted_indices = torch.argsort(hilbert_indices, dim=1)
        elif method == 'z_order':
            # 使用 Z-Order 曲线进行编码
            sorted_indices = []
            for b in range(B):
                x = coords[b, :, 0]
                y = coords[b, :, 1]
                z = coords[b, :, 2]
                z_keys = xyz2key(x, y, z, depth=16)
                sorted_indices.append(torch.argsort(z_keys))
            sorted_indices = torch.stack(sorted_indices, dim=0)  # (B, N)
        elif method == 'Trans_z_order':
            # 使用 Z-Order 曲线进行编码
            sorted_indices = []
            for b in range(B):
                x = coords[b, :, 1]
                y = coords[b, :, 0]
                z = coords[b, :, 2]
                z_keys = xyz2key(x, y, z, depth=16)
                sorted_indices.append(torch.argsort(z_keys))
            sorted_indices = torch.stack(sorted_indices, dim=0)  # (B, N)
        else:
            raise ValueError(f"Unsupported method for neighborhood construction: {method}")

        all_sorted_indices.append(sorted_indices)

    # 2. 将多种序列化方式的结果进行融合，生成多种特征
    for sorted_indices in all_sorted_indices:
        neighborhoods = []
        for b in range(B):
            serialized_neighbors = []
            for i in range(N):
                # 使用序列化结果直接构建邻域
                start_idx = max(0, i - k // 2)
                end_idx = min(N, i + k // 2 + 1)
                neighbors = sorted_indices[b][start_idx:end_idx][:k]
                # 如果邻居不足 k 个，用最后一个邻居填充
                if len(neighbors) < k:
                    neighbors = torch.nn.functional.pad(neighbors, (0, k - len(neighbors)), mode='constant', value=neighbors[-1])
                serialized_neighbors.append(neighbors)
            neighborhoods.append(torch.stack(serialized_neighbors))

        inds = torch.stack(neighborhoods)  # (B, N, k)
        inds = inds.to(coords.device)  # 将邻域索引移动到相应设备上
        neigh_feats = gather_points(feats, inds)  # (B, N, k, C)
        feats_expanded = feats.unsqueeze(2).repeat(1, 1, k, 1)  # (B, N, k, C)

        all_features.append(torch.cat([feats_expanded, neigh_feats - feats_expanded], dim=-1))  # (B, N, k, 2C)

    all_features = torch.stack(all_features, dim=-1)  # (B, N, k, 2C, M)
    B, N, k, two_C, M = all_features.shape

    # Shuffle orders if required
    if shuffle_orders:
        perm = torch.randperm(M).to(device)
        all_features = all_features[:, :, :, :, perm]

    # Reshape for flash attention
    all_features_reshaped = all_features.view(B * N * k, M * 2 * C)  # (B * N * k, M * 2 * C)

    # Convert to fp16
    all_features_reshaped = all_features_reshaped.to(torch.float16)

    # Define number of heads and head dimension
    total_dim = M * 2 * C
    headdim = min(256, total_dim)  # Ensure headdim is at most 256
    nheads = total_dim // headdim

    # Ensure the total dimension matches
    assert nheads * headdim == total_dim, "nheads * headdim must equal M * 2 * C"

    # Prepare QKV for flash attention
    qkv = torch.stack([all_features_reshaped, all_features_reshaped, all_features_reshaped],
                      dim=1)  # (B * N * k, 3, M * 2 * C)
    qkv = qkv.view(B * N * k, 3, nheads, headdim)  # (B * N * k, 3, nheads, headdim)

    # Prepare cumulative sequence lengths
    cu_seqlens = torch.arange(0, B * N * k + 1, step=k, dtype=torch.int32, device=device)  # (B * N + 1)

    # Apply flash attention
    attn_output = flash_attn_varlen_qkvpacked_func(
        qkv=qkv,
        cu_seqlens=cu_seqlens,
        max_seqlen=k,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False
    )  # (B * N * k, nheads, headdim)

    # Reshape back to original shape
    attn_output = attn_output.view(B * N * k, nheads * headdim)  # (B * N * k, M * 2 * C)

    # Convert back to fp32 for further processing
    attn_output = attn_output.to(torch.float32)

    # Linear transformation to reduce dimensionality
    linear_out = nn.Linear(M * 2 * C, 512).to(device)
    fused_features = linear_out(attn_output).view(B, N, k, -1)  # (B, N, k, 512)

    return fused_features  # (B, N, k, 512)

class LocalFeatureFused(nn.Module):
    def __init__(self, in_dim, out_dims):
        super(LocalFeatureFused, self).__init__()
        self.blocks = nn.Sequential()
        for i, out_dim in enumerate(out_dims):
            self.blocks.add_module(f'conv2d_{i}',
                                   nn.Conv2d(in_dim, out_dim, 1, bias=False))
            self.blocks.add_module(f'in_{i}',
                                   nn.InstanceNorm2d(out_dim))
            self.blocks.add_module(f'relu_{i}', nn.ReLU(inplace=True))
            in_dim = out_dim

    def forward(self, x):
        '''
        :param x: (B, C1, K, M)
        :return: (B, C2, M)
        '''
        x = self.blocks(x)
        x = torch.max(x, dim=2)[0]
        return x


class PPF(nn.Module):
    def __init__(self, feats_dim, k, radius):
        super().__init__()
        self.k = k
        self.radius = radius
        self.local_feature_fused = LocalFeatureFused(in_dim=10,
                                                     out_dims=feats_dim)
    
    def forward(self, coords, feats):
        '''

        :param coors: (B, 3, N)
        :param feats: (B, 3, N)
        :param k: int
        :return: (B, C, N)
        '''

        feats = feats.permute(0, 2, 1).contiguous()
        coords = coords.permute(0, 2, 1).contiguous()
        new_xyz, new_points, grouped_inds, grouped_xyz = \
            sample_and_group(xyz=coords,
                             points=feats,
                             M=-1,
                             radius=self.radius,
                             K=self.k)
        nr_d = angle(feats[:, :, None, :], grouped_xyz)
        ni_d = angle(new_points[..., 3:], grouped_xyz)
        nr_ni = angle(feats[:, :, None, :], new_points[..., 3:])
        d_norm = torch.norm(grouped_xyz, dim=-1)
        ppf_feat = torch.stack([nr_d, ni_d, nr_ni, d_norm], dim=-1) # (B, N, K, 4)
        new_points = torch.cat([new_points[..., :3], ppf_feat], dim=-1)

        coords = torch.unsqueeze(coords, dim=2).repeat(1, 1, min(self.k, new_points.size(2)), 1)
        new_points = torch.cat([coords, new_points], dim=-1)
        feature_local = new_points.permute(0, 3, 2, 1).contiguous() # (B, C1 + 3, K, M)
        feature_local = self.local_feature_fused(feature_local)
        return feature_local


class GCN(nn.Module):
    def __init__(self, feats_dim, k):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(feats_dim * 2, feats_dim, 1, bias=False),
            nn.InstanceNorm2d(feats_dim),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(feats_dim * 2, feats_dim * 2, 1, bias=False),
            nn.InstanceNorm2d(feats_dim * 2),
            nn.LeakyReLU(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(feats_dim * 4, feats_dim, 1, bias=False),
            nn.InstanceNorm1d(feats_dim),
            nn.LeakyReLU(0.2)
        )
        self.k = k

    def forward(self, coords, feats):
        '''

        :param coors: (B, 3, N)
        :param feats: (B, C, N)
        :param k: int
        :return: (B, C, N)
        '''
        feats1 = get_graph_features(feats=feats.permute(0, 2, 1).contiguous(),
                                    coords=coords.permute(0, 2, 1).contiguous(),
                                    k=self.k, methods=['z_order','Trans_z_order', 'hilbert', 'Trans_hilbert'], shuffle_orders=True)
        feats1 = self.conv1(feats1.permute(0, 3, 1, 2).contiguous())
        feats1 = torch.max(feats1, dim=-1)[0]

        feats2 = get_graph_features(feats=feats1.permute(0, 2, 1).contiguous(),
                                    coords=coords.permute(0, 2, 1).contiguous(),
                                    k=self.k,methods=['z_order','Trans_z_order', 'hilbert', 'Trans_hilbert'], shuffle_orders=True)
        feats2 = self.conv2(feats2.permute(0, 3, 1, 2).contiguous())
        feats2 = torch.max(feats2, dim=-1)[0]

        feats3 = torch.cat([feats, feats1, feats2], dim=1)
        feats3 = self.conv3(feats3)

        return feats3


class GGE(nn.Module):
    def __init__(self, feats_dim, gcn_k, ppf_k, radius, bottleneck):
        super().__init__()
        self.gcn = GCN(feats_dim, gcn_k)
        if bottleneck:
            self.ppf = PPF([feats_dim // 2, feats_dim, feats_dim // 2], ppf_k, radius)
            self.fused = nn.Sequential(
                nn.Conv1d(feats_dim + feats_dim // 2, feats_dim + feats_dim // 2, 1),
                nn.InstanceNorm1d(feats_dim + feats_dim // 2),
                nn.LeakyReLU(0.2),
                nn.Conv1d(feats_dim + feats_dim // 2, feats_dim, 1),
                nn.InstanceNorm1d(feats_dim),
                nn.LeakyReLU(0.2)
                )
        else:
            self.ppf = PPF([feats_dim, feats_dim*2, feats_dim], ppf_k, radius)
            self.fused = nn.Sequential(
                nn.Conv1d(feats_dim * 2, feats_dim * 2, 1),
                nn.InstanceNorm1d(feats_dim * 2),
                nn.LeakyReLU(0.2),
                nn.Conv1d(feats_dim * 2, feats_dim, 1),
                nn.InstanceNorm1d(feats_dim),
                nn.LeakyReLU(0.2)
                )
    
    def forward(self, coords, feats, normals):
        feats_ppf = self.ppf(coords, normals)
        feats_gcn = self.gcn(coords, feats)
        feats_fused = self.fused(torch.cat([feats_ppf, feats_gcn], dim=1))
        return feats_fused


def multi_head_attention(query, key, value):
    '''

    :param query: (B, dim, nhead, N)
    :param key: (B, dim, nhead, M)
    :param value: (B, dim, nhead, M)
    :return: (B, dim, nhead, N)
    '''
    dim = query.size(1)
    scores = torch.einsum('bdhn, bdhm->bhnm', query, key) / dim**0.5
    attention = torch.nn.functional.softmax(scores, dim=-1)
    feats = torch.einsum('bhnm, bdhm->bdhn', attention, value)
    return feats


class Cross_Attention(nn.Module):
    def __init__(self, feat_dims, nhead):
        super().__init__()
        assert feat_dims % nhead == 0
        self.feats_dim = feat_dims
        self.nhead = nhead
        # self.q_conv = nn.Conv1d(feat_dims, feat_dims, 1, bias=True)
        # self.k_conv = nn.Conv1d(feat_dims, feat_dims, 1, bias=True)
        # self.v_conv = nn.Conv1d(feat_dims, feat_dims, 1, bias=True)
        self.conv = nn.Conv1d(feat_dims, feat_dims, 1)
        self.q_conv, self.k_conv, self.v_conv = [copy.deepcopy(self.conv) for _ in range(3)] # a good way than better ?
        self.mlp = nn.Sequential(
            nn.Conv1d(feat_dims * 2, feat_dims * 2, 1),
            nn.InstanceNorm1d(feat_dims * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(feat_dims * 2, feat_dims, 1),
        )

    def forward(self, feats1, feats2):
        '''

        :param feats1: (B, C, N)
        :param feats2: (B, C, M)
        :return: (B, C, N)
        '''
        b = feats1.size(0)
        dims = self.feats_dim // self.nhead
        query = self.q_conv(feats1).reshape(b, dims, self.nhead, -1)
        key = self.k_conv(feats2).reshape(b, dims, self.nhead, -1)
        value = self.v_conv(feats2).reshape(b, dims, self.nhead, -1)
        feats = multi_head_attention(query, key, value)
        feats = feats.reshape(b, self.feats_dim, -1)
        feats = self.conv(feats)
        cross_feats = self.mlp(torch.cat([feats1, feats], dim=1))
        return cross_feats


class InformationInteractive(nn.Module):
    def __init__(self, layer_names, feat_dims, gcn_k, ppf_k, radius, bottleneck, nhead):
        super().__init__()
        self.layer_names = layer_names
        self.blocks = nn.ModuleList()
        #self.diffusion = DiffusionModule(feats_dim=feat_dims, num_steps=1)
        for layer_name in layer_names:
            if layer_name == 'gcn':
                self.blocks.append(GCN(feat_dims, gcn_k))
            elif layer_name == 'gge':
                self.blocks.append(GGE(feat_dims, gcn_k, ppf_k, radius, bottleneck))
            elif layer_name == 'cross_attn':
                self.blocks.append(Cross_Attention(feat_dims, nhead))
            else:
                raise NotImplementedError

    def forward(self, coords1, feats1, coords2, feats2, normals1, normals2):
        '''

        :param coords1: (B, 3, N)
        :param feats1: (B, C, N)
        :param coords2: (B, 3, M)
        :param feats2: (B, C, M)
        :return: feats1=(B, C, N), feats2=(B, C, M)
        '''

        for layer_name, block in zip(self.layer_names, self.blocks):
            if layer_name == 'gcn':
                feats1 = block(coords1, feats1)
                feats2 = block(coords2, feats2)
            elif layer_name == 'gge':
                feats1 = block(coords1, feats1, normals1)
                feats2 = block(coords2, feats2, normals2)
            elif layer_name == 'cross_attn':
                feats1 = feats1 + block(feats1, feats2)
                feats2 = feats2 + block(feats2, feats1)
            else:
                raise NotImplementedError

        return feats1, feats2
