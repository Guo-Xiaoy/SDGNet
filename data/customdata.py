import numpy as np
import os
import pickle
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
import pandas as pd
CUR = os.path.dirname(os.path.abspath(__file__))
from utils import npy2pcd, pcd2npy, vis_plys, get_correspondences, format_lines, normal
import torch

myscaler = 100.0
class OnUnitCube:
    def __init__(self):
        pass

    def method1(self, tensor):
        m = tensor.mean(dim=0, keepdim=True)  # [N, D] -> [1, D]
        v = tensor - m
        s = torch.max(v.abs())
        v = v / s * 0.5
        return v

    def method2(self, tensor):
        c = torch.max(tensor, dim=0)[0] - torch.min(tensor, dim=0)[0]  # [N, D] -> [D]
        s = torch.max(c)  # -> scalar
        v = tensor / s
        return v - v.mean(dim=0, keepdim=True)

    def __call__(self, tensor):
        # 使用method2进行归一化
        return self.method2(tensor)

class CustomData(Dataset):
    def __init__(self, root, split, aug, overlap_radius, noise_scale=0.005, max_points=1024):
        super().__init__()
        self.root = root
        self.split = split  # 'train', 'val', or 'test'
        self.aug = aug
        self.noise_scale = noise_scale
        self.overlap_radius = overlap_radius
        self.max_points = max_points

        # 修改这里以适应新的.pkl文件名
        pkl_path = os.path.join(CUR, 'dataset', f'{split}.pkl')
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"File {pkl_path} does not exist.")

        with open(pkl_path, 'rb') as f:
            self.infos = pickle.load(f)

        # 检查.pkl文件中是否包含'points'键值对
        for info in self.infos:
            if 'points' not in info:
                raise KeyError(f"Key 'points' not found in the .pkl file entry.")

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, item):
        # 从.pkl文件中读取原始点云数据
        on_unit_cube = OnUnitCube()
        points = on_unit_cube(torch.tensor(self.infos[item]['points'].copy(), dtype=torch.float32)).numpy()
        #points = self.infos[item]['points'].copy() / myscaler  # 根据您的需求调整缩放因子

        # 如果需要，可以随机分割成源点云和目标点云
        # 这里我们简单地将整个点云作为源点云，并且复制一份作为目标点云
        src_points = points.copy()
        tgt_points = points.copy()
        # 创建随机旋转和平移矩阵（仅用于增强）
        if self.aug:
            # 将旋转角度限制在-5到5度
            euler_ab = np.random.uniform(-0.087, 0.087, size=3)  # -5 to 5 degrees in radians
            rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
            # 将平移限制在-1到1之间
            trans_ab = np.random.uniform(-50/myscaler, 50/myscaler, size=3)  # -1 to 1

            # 变换目标点云
            tgt_points = (rot_ab @ tgt_points.T).T + trans_ab
            # 更新旋转和平移矩阵
            T = np.eye(4).astype(np.float32)
            T[:3, :3] = rot_ab
            T[:3, 3:] = trans_ab[:, None]
        else:
            # 如果不进行增强，则使用单位矩阵作为变换矩阵
            T = np.eye(4).astype(np.float32)

        # 添加噪声
        if self.aug:
            src_points += (np.random.rand(src_points.shape[0], 3) - 0.5) * self.noise_scale
            tgt_points += (np.random.rand(tgt_points.shape[0], 3) - 0.5) * self.noise_scale

        # 点云下采样
        if (src_points.shape[0] > self.max_points):
            idx = np.random.permutation(src_points.shape[0])[:self.max_points]
            src_points = src_points[idx]
        if (tgt_points.shape[0] > self.max_points):
            idx = np.random.permutation(tgt_points.shape[0])[:self.max_points]
            tgt_points = tgt_points[idx]

        coors = get_correspondences(npy2pcd(src_points),
                                    npy2pcd(tgt_points),
                                    T,
                                    self.overlap_radius)

        src_feats = np.ones_like(src_points[:, :1], dtype=np.float32)
        tgt_feats = np.ones_like(tgt_points[:, :1], dtype=np.float32)

        src_pcd, tgt_pcd = normal(npy2pcd(src_points)), normal(npy2pcd(tgt_points))
        src_normals = np.array(src_pcd.normals).astype(np.float32)
        tgt_normals = np.array(tgt_pcd.normals).astype(np.float32)

        pair = dict(
            src_points=src_points,
            tgt_points=tgt_points,
            src_feats=src_feats,
            tgt_feats=tgt_feats,
            src_normals=src_normals,
            tgt_normals=tgt_normals,
            transf=T,
            coors=coors,
            src_points_raw=src_points,
            tgt_points_raw=tgt_points)
        #print(len(pair['src_points']))
        return pair


class CustomDataForTest(Dataset):
    def __init__(self, root, split, aug, overlap_radius, noise_scale=0.0025, max_points=1024):
        super().__init__()
        self.root = root
        self.split = split
        self.aug = aug
        self.noise_scale = noise_scale
        self.overlap_radius = overlap_radius
        self.max_points = max_points

        # 加载点云数据
        pkl_path = os.path.join(CUR, 'dataset', f'{split}.pkl')
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"File {pkl_path} does not exist.")

        with open(pkl_path, 'rb') as f:
            self.infos = pickle.load(f)

        # 加载位姿数据
        test_gt_path = os.path.join(CUR, 'dataset', f'{split}_gt.csv')
        if not os.path.exists(test_gt_path):
            raise FileNotFoundError(f"File {test_gt_path} does not exist.")
        self.pose_data = pd.read_csv(test_gt_path, header=None)
        #print(len(self.pose_data))
        if len(self.pose_data) != len(self.infos):
            raise ValueError("The number of poses must match the number of point clouds.")

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, item):
        #on_unit_cube = OnUnitCube()
        #points = on_unit_cube(torch.tensor(self.infos[item]['points'].copy(), dtype=torch.float32)).numpy()
        points = self.infos[item]['points'].copy() / 100  # 根据您的需求调整缩放因子
        # 从CSV文件中获取旋转和平移信息
        rpy = self.pose_data.iloc[item, :3].values
        xyz = self.pose_data.iloc[item, 3:].values

        # 转换为旋转矩阵和平移向量
        rot = Rotation.from_euler('xyz', rpy).as_matrix()
        trans = xyz

        # 创建变换矩阵
        T = np.eye(4).astype(np.float32)
        T[:3, :3] = rot
        T[:3, 3] = trans
        # 应用变换
        src_points = points.copy()
        tgt_points = (rot @ points.T).T + trans

        # 添加噪声
        if self.aug:
            src_points += (np.random.rand(src_points.shape[0], 3) - 0.5) * self.noise_scale
            tgt_points += (np.random.rand(tgt_points.shape[0], 3) - 0.5) * self.noise_scale

        # 点云下采样
        if src_points.shape[0] > self.max_points:
            idx = np.random.choice(src_points.shape[0], self.max_points, replace=False)
            src_points = src_points[idx]
        if tgt_points.shape[0] > self.max_points:
            idx = np.random.choice(tgt_points.shape[0], self.max_points, replace=False)
            tgt_points = tgt_points[idx]

        coors = get_correspondences(npy2pcd(src_points), npy2pcd(tgt_points), T, self.overlap_radius)

        src_feats = np.ones_like(src_points[:, :1], dtype=np.float32)
        tgt_feats = np.ones_like(tgt_points[:, :1], dtype=np.float32)

        src_pcd, tgt_pcd = normal(npy2pcd(src_points)), normal(npy2pcd(tgt_points))
        src_normals = np.array(src_pcd.normals).astype(np.float32)
        tgt_normals = np.array(tgt_pcd.normals).astype(np.float32)

        pair = {
            'src_points': src_points,
            'tgt_points': tgt_points,
            'src_feats': src_feats,
            'tgt_feats': tgt_feats,
            'src_normals': src_normals,
            'tgt_normals': tgt_normals,
            'transf': T,
            'coors': coors,
            'src_points_raw': src_points,
            'tgt_points_raw': tgt_points
        }

        return pair