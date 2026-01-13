import argparse
import copy
import numpy as np
import os
import torch
from easydict import EasyDict as edict
import sys
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import open3d as o3d
from data import collate_fn
from models import architectures, NgeNet, vote
from utils import decode_config, npy2pcd, pcd2npy, execute_global_registration, \
    npy2feat, setup_seed, get_blue, get_yellow, voxel_ds, normal, \
    read_cloud, vis_plys

CUR = os.path.dirname(os.path.abspath(__file__))

scaler = 100.0
class NgeNet_pipeline():
    def __init__(self, ckpt_path, voxel_size, vote_flag, cuda=True):
        self.voxel_size_3dmatch = 0.025
        self.voxel_size = voxel_size
        self.scale = self.voxel_size  # / self.voxel_size_3dmatch
        self.cuda = cuda
        self.vote_flag = vote_flag
        config = self.prepare_config()
        self.neighborhood_limits = [8, 15, 22, 24]
        model = NgeNet(config)
        if self.cuda:
            model = model.cuda()
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path)
                state_dict = checkpoint
                # 删除与 feature_transform 相关的权重
                filtered_state_dict = {k: v for k, v in state_dict.items() if 'feature_transform' not in k}
                # 加载过滤后的 state_dict
                model.load_state_dict(filtered_state_dict, strict=False)
                print("Loaded pre-trained model.")
        else:
            model.load_state_dict(
                torch.load(ckpt_path, map_location=torch.device('cpu')))
        self.model = model
        self.config = config
        self.model.eval()

    def prepare_config(self):
        config = decode_config(os.path.join(CUR, 'configs', 'customdata.yaml'))
        config = edict(config)
        # config.first_subsampling_dl = self.voxel_size
        config.architecture = architectures[config.dataset]
        return config

    def prepare_inputs(self, source, target):
        src_pcd_input = pcd2npy(voxel_ds(copy.deepcopy(source), self.voxel_size))
        tgt_pcd_input = pcd2npy(voxel_ds(copy.deepcopy(target), self.voxel_size))

        src_pcd_input /= self.scale
        tgt_pcd_input /= self.scale

        src_feats = np.ones_like(src_pcd_input[:, :1])
        tgt_feats = np.ones_like(tgt_pcd_input[:, :1])

        src_pcd = normal(npy2pcd(src_pcd_input), radius=4 * self.voxel_size_3dmatch, max_nn=30, loc=(0, 0, 0))
        tgt_pcd = normal(npy2pcd(tgt_pcd_input), radius=4 * self.voxel_size_3dmatch, max_nn=30, loc=(0, 0, 0))
        src_normals = np.array(src_pcd.normals).astype(np.float32)
        tgt_normals = np.array(tgt_pcd.normals).astype(np.float32)

        T = np.eye(4)
        coors = np.array([[0, 0], [1, 1]])
        src_pcd = pcd2npy(source)
        tgt_pcd = pcd2npy(target)

        pair = dict(
            src_points=src_pcd_input,
            tgt_points=tgt_pcd_input,
            src_feats=src_feats,
            tgt_feats=tgt_feats,
            src_normals=src_normals,
            tgt_normals=tgt_normals,
            transf=T,
            coors=coors,
            src_points_raw=src_pcd,
            tgt_points_raw=tgt_pcd)

        dict_inputs = collate_fn([pair], self.config, self.neighborhood_limits)
        if self.cuda:
            for k, v in dict_inputs.items():
                if isinstance(v, list):
                    for i in range(len(v)):
                        dict_inputs[k][i] = dict_inputs[k][i].cuda()
                else:
                    dict_inputs[k] = dict_inputs[k].cuda()

        return dict_inputs

    def pipeline(self, source, target, npts=20000):
        inputs = self.prepare_inputs(source, target)

        batched_feats_h, batched_feats_m, batched_feats_l = self.model(inputs)
        stack_points = inputs['points']
        stack_lengths = inputs['stacked_lengths']
        coords_src = stack_points[0][:stack_lengths[0][0]]
        coords_tgt = stack_points[0][stack_lengths[0][0]:]
        feats_src_h = batched_feats_h[:stack_lengths[0][0]]
        feats_tgt_h = batched_feats_h[stack_lengths[0][0]:]
        feats_src_m = batched_feats_m[:stack_lengths[0][0]]
        feats_tgt_m = batched_feats_m[stack_lengths[0][0]:]
        feats_src_l = batched_feats_l[:stack_lengths[0][0]]
        feats_tgt_l = batched_feats_l[stack_lengths[0][0]:]

        source_npy = coords_src.detach().cpu().numpy() * self.scale
        target_npy = coords_tgt.detach().cpu().numpy() * self.scale

        source_feats_h = feats_src_h[:, :-2].detach().cpu().numpy()
        target_feats_h = feats_tgt_h[:, :-2].detach().cpu().numpy()
        source_feats_m = feats_src_m.detach().cpu().numpy()
        target_feats_m = feats_tgt_m.detach().cpu().numpy()
        source_feats_l = feats_src_l.detach().cpu().numpy()
        target_feats_l = feats_tgt_l.detach().cpu().numpy()

        source_overlap_scores = feats_src_h[:, -2].detach().cpu().numpy()
        target_overlap_scores = feats_tgt_h[:, -2].detach().cpu().numpy()
        source_scores = source_overlap_scores
        target_scores = target_overlap_scores

        npoints = npts
        if npoints > 0:
            if source_npy.shape[0] > npoints:
                p = source_scores / np.sum(source_scores)
                idx = np.random.choice(len(source_npy), size=npoints, replace=False, p=p)
                source_npy = source_npy[idx]
                source_feats_h = source_feats_h[idx]
                source_feats_m = source_feats_m[idx]
                source_feats_l = source_feats_l[idx]

            if target_npy.shape[0] > npoints:
                p = target_scores / np.sum(target_scores)
                idx = np.random.choice(len(target_npy), size=npoints, replace=False, p=p)
                target_npy = target_npy[idx]
                target_feats_h = target_feats_h[idx]
                target_feats_m = target_feats_m[idx]
                target_feats_l = target_feats_l[idx]

        if self.vote_flag:
            after_vote = vote(source_npy=source_npy,
                              target_npy=target_npy,
                              source_feats=[source_feats_h, source_feats_m, source_feats_l],
                              target_feats=[target_feats_h, target_feats_m, target_feats_l],
                              voxel_size=self.voxel_size * 2,
                              use_cuda=self.cuda)
            source_npy, target_npy, source_feats_npy, target_feats_npy = after_vote
        else:
            source_feats_npy, target_feats_npy = source_feats_h, target_feats_h
        source, target = npy2pcd(source_npy), npy2pcd(target_npy)
        source_feats, target_feats = npy2feat(source_feats_npy), npy2feat(target_feats_npy)
        pred_T, estimate = execute_global_registration(source=source,
                                                       target=target,
                                                       source_feats=source_feats,
                                                       target_feats=target_feats,
                                                       voxel_size=self.voxel_size * 2)

        torch.cuda.empty_cache()
        return pred_T


def rotation_matrix_to_rpy(R):
    # 提取3x3旋转矩阵
    rotation_matrix = R[:3, :3]

    # 计算Yaw (ψ)
    psi = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    # 计算Pitch (θ)
    theta = np.arcsin(-rotation_matrix[2, 0])

    # 计算Roll (φ)
    phi = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])

    rpy = (np.degrees(phi), np.degrees(theta), np.degrees(psi))

    return rpy


def generate_data(args, gt, num_runs=100):
    data = []
    source, target = read_cloud(args.src_path), read_cloud(args.tgt_path)
    source_ori, target_ori = read_cloud(args.src_path), read_cloud(args.tgt_path)
    source.points = o3d.utility.Vector3dVector(np.asarray(source.points) / scaler)
    target.points = o3d.utility.Vector3dVector(np.asarray(target.points) / scaler)

    for _ in range(num_runs):
        cuda = not args.no_cuda
        vote_flag = not args.no_vote
        model = NgeNet_pipeline(
            ckpt_path=args.checkpoint,
            voxel_size=args.voxel_size,
            vote_flag=vote_flag,
            cuda=cuda)

        T = model.pipeline(source, target, npts=args.npts)
        T = T.copy()
        T[0:3, 3] *= scaler
        rpy = rotation_matrix_to_rpy(T)
        print(rpy, T[0:3, 3])
        # 确保 rpy 和 T[0:3,3] 都是列表，并且可以连接
        rpy_list = list(rpy)
        translation_list = list(T[0:3, 3])
        predicted_values = rpy_list + translation_list

        # 计算绝对误差
        abs_error = np.abs(np.array(predicted_values) - np.array(gt))

        # 将rpy、T[0:3,3]、gt和绝对误差添加到数据列表
        data.append(np.concatenate((predicted_values, gt, abs_error)))

    df = pd.DataFrame(data, columns=['Roll', 'Pitch', 'Yaw', 'X', 'Y', 'Z',
                                     'GT_Roll', 'GT_Pitch', 'GT_Yaw', 'GT_X', 'GT_Y', 'GT_Z',
                                     'Error_Roll', 'Error_Pitch', 'Error_Yaw', 'Error_X', 'Error_Y', 'Error_Z'])

    # 计算每列误差的统计信息
    error_columns = ['Error_Roll', 'Error_Pitch', 'Error_Yaw', 'Error_X', 'Error_Y', 'Error_Z']
    stats = df[error_columns].agg(['mean', 'max', 'min']).transpose().reset_index()
    stats.columns = ['Error_Metric', 'Mean_Error', 'Max_Error', 'Min_Error']

    # 创建新的DataFrame来保存原始数据和统计信息
    result_df = pd.concat([df, stats], axis=0, ignore_index=True)

    # 创建Excel文件名
    excel_filename = f"./registration/{args.src_path.split('/')[-1].split('.')[0]}_{args.tgt_path.split('/')[-1].split('.')[0]}_results.xlsx"

    # 将DataFrame写入Excel
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        result_df.to_excel(writer, index=False, header=True)
    if not args.no_vis:
        # voxelization for fluent visualization
        source_ori, target_ori = read_cloud(args.src_path), read_cloud(args.tgt_path)
        estimate_ori = copy.deepcopy(source_ori).transform(T)
        # 设置颜色
        source_ori.paint_uniform_color([1, 0, 0])  # 红色
        target_ori.paint_uniform_color([0, 1, 0])  # 绿色
        estimate_ori.paint_uniform_color([0, 0, 1])  # 蓝色

        # 可视化
        o3d.visualization.draw_geometries([source_ori, target_ori, estimate_ori])


if __name__ == "__main__":
    setup_seed(1212)
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    
    parser.add_argument('--src_path', default='./demo_data/demo1.csv', help='source point cloud path')
    parser.add_argument('--tgt_path', default='./demo_data/demo2.csv', help='target point cloud path')
    parser.add_argument('--checkpoint', default='./train_logs/customdata/checkpoints/best_loss.pth',
                        help='checkpoint path')
    parser.add_argument('--voxel_size', type=float, default=0.05, help='voxel size')
    parser.add_argument('--npts', type=int, default=10240, help='the number of sampled points for registration')
    parser.add_argument('--no_vote', action='store_true', help='whether to use multi-level consistent voting')
    parser.add_argument('--no_vis', action='store_true', help='whether to visualize the point clouds')
    parser.add_argument('--no_cuda', action='store_true', help='whether to use cuda')
    args = parser.parse_args()

    # gt
    if 'demo1' in args.src_path and 'demo2' in args.tgt_path:
        gt = [0, 0, 0, -1.5, 1.5, -2.0]
    else:
        raise ValueError("Unsupported file combination for src_path and tgt_path")

    # save to Excel
    generate_data(args, gt, num_runs=1)

   

