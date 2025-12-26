import warnings
warnings.filterwarnings('ignore')
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from easydict import EasyDict as edict
from data import get_dataset, get_dataloader
from models import NgeNet
from losses import Loss
import open3d as o3d
from models import architectures, NgeNet, vote
import numpy as np
import os
from utils import decode_config, npy2pcd, pcd2npy, execute_global_registration, \
    npy2feat, setup_seed, get_blue, get_yellow, voxel_ds, normal, \
    read_cloud, vis_plys
import pandas as pd
global EXCEL_PATH
EXCEL_PATH = "./test_result/test_RT_result_0.0025.xlsx"
def save_summary(writer, loss_dict, global_step, tag, lr=None):
    for k, v in loss_dict.items():
        writer.add_scalar(f'{tag}/{k}', v, global_step)
    if lr is not None:
        writer.add_scalar('lr', lr, global_step)

def rotation_matrix_to_rpy(R):
    # 提取3x3旋转矩阵
    rotation_matrix = R[:3, :3]
    # 计算Yaw (ψ)
    psi = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    # 计算Pitch (θ)
    theta = np.arcsin(-rotation_matrix[2, 0])
    # 计算Roll (φ)
    phi = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    rpy = (phi, theta, psi)

    return rpy


def display_point_clouds(source, target, estimate):

    def paint_point_cloud(point_cloud, color):
        point_cloud.paint_uniform_color(color)

    # 上色
    paint_point_cloud(source, [1, 0, 0])  # 红色
    paint_point_cloud(target, [0, 1, 0])  # 绿色
    paint_point_cloud(estimate, [0, 0, 1])  # 蓝色

    # 显示点云
    o3d.visualization.draw_geometries([source, target, estimate])

def test(model, dataloader, config, writer, epoch):
    model.eval()  # 切换到评估模式
    total_test_loss = 0.0
    total_circle_loss, total_recall, total_recall_sum = [], [], []
    data = []
    with torch.no_grad():  # 关闭梯度计算
        for inputs in tqdm(dataloader):
            for k, v in inputs.items():
                if isinstance(v, list):
                    for i in range(len(v)):
                        inputs[k][i] = inputs[k][i].cuda()
                else:
                    inputs[k] = inputs[k].cuda()

            # 调用 pipeline 逻辑
            batched_feats_h, batched_feats_m, batched_feats_l = model(inputs)
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

            source_npy = coords_src.detach().cpu().numpy()
            target_npy = coords_tgt.detach().cpu().numpy()

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

            npoints = 1024  # 可以根据需要调整这个值
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


            after_vote = vote(source_npy=source_npy,
                              target_npy=target_npy,
                              source_feats=[source_feats_h, source_feats_m, source_feats_l],
                              target_feats=[target_feats_h, target_feats_m, target_feats_l],
                              voxel_size=0.005,
                              use_cuda=True)
            source_npy, target_npy, source_feats_npy, target_feats_npy = after_vote

            # 执行全局配准
            source, target = npy2pcd(source_npy), npy2pcd(target_npy)
            source_feats, target_feats = npy2feat(source_feats_npy), npy2feat(target_feats_npy)
            pred_T, estimate = execute_global_registration(
                source=source,
                target=target,
                source_feats=source_feats,
                target_feats=target_feats,
                voxel_size=0.005
            )
            #display_point_clouds(source, target, estimate)
            rpy = rotation_matrix_to_rpy(pred_T)
            rpy_list = list(rpy)
            translation_list = list(pred_T[0:3, 3])
            predicted_values = rpy_list + translation_list
            data.append(predicted_values)
            # 计算损失
            coors = inputs['coors'][0]  # 准备处理 batch size > 1 的情况
            transf = inputs['transf'][0]  # (1, 4, 4)，准备处理 batch size > 1 的情况
            points_raw = inputs['batched_points_raw']
            coords_src = points_raw[:stack_lengths[0][0]]
            coords_tgt = points_raw[stack_lengths[0][0]:]

            loss_dict = model_loss(coords_src=coords_src,
                                   coords_tgt=coords_tgt,
                                   feats_src=feats_src_h,
                                   feats_tgt=feats_tgt_h,
                                   feats_src_m=feats_src_m,
                                   feats_tgt_m=feats_tgt_m,
                                   feats_src_l=feats_src_l,
                                   feats_tgt_l=feats_tgt_l,
                                   coors=coors,
                                   transf=transf,
                                   w_saliency=config.w_saliency_loss)

            loss = loss_dict['circle_loss'] + loss_dict['circle_loss_m'] + loss_dict['circle_loss_l']
            total_test_loss += loss.item()
            total_circle_loss.append(loss_dict['circle_loss'].detach().cpu().numpy())
            total_recall.append(loss_dict['recall'].detach().cpu().numpy())
            total_recall_sum.append((loss_dict['recall'] + loss_dict['recall_m'] + loss_dict['recall_l']).detach().cpu().numpy())

            global_step = epoch * len(dataloader) + 1

            if global_step % config.log_freq == 0:
                save_summary(writer, loss_dict, global_step, 'test')

            # 如果 GPU 内存允许，建议不要添加此行代码或每轮结束后添加
            torch.cuda.empty_cache()
    df = pd.DataFrame(data, columns=['Roll', 'Pitch', 'Yaw', 'X', 'Y', 'Z'])
    excel_filename = EXCEL_PATH

    result_df = pd.concat([df], axis=0, ignore_index=True)
    # 将DataFrame写入Excel
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        result_df.to_excel(writer, index=False, header=True)
    avg_test_loss = total_test_loss / len(dataloader)
    print(f"Epoch {epoch}: Average Test Loss: {avg_test_loss:.4f}")
    return avg_test_loss, np.mean(total_circle_loss), np.mean(total_recall), np.mean(total_recall_sum)

if __name__ == "__main__":
    # 配置文件路径
    argv = './configs/customdata.yaml'
    #setup_seed(1234)
    config = decode_config(argv)
    config = edict(config)
    config.architecture = architectures[config.dataset]

    # 模型和数据加载
    model = NgeNet(config).cuda()
    model_loss = Loss(config)

    neighborhood_limits = [8, 15, 22, 23]
    # 测试集数据加载器
    train_dataset, val_dataset,test_dataset = get_dataset(config.dataset, config)  # 假设支持指定 split 参数来获取不同数据集
    test_dataloader, _ = get_dataloader(config=config,
                                        dataset=test_dataset,
                                        batch_size=1,
                                        num_workers=config.num_workers,
                                        shuffle=False,
                                        neighborhood_limits=None)

    # TensorBoard 日志
    saved_logs_path = os.path.join(config.exp_dir, 'summary')
    writer = SummaryWriter(saved_logs_path)

    # 加载预训练模型（如果有）
    checkpoint_path = os.path.join(config.exp_dir, 'checkpoints', 'best_loss.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint
        # 删除与 feature_transform 相关的权重
        filtered_state_dict = {k: v for k, v in state_dict.items() if 'feature_transform' not in k}
        # 加载过滤后的 state_dict
        model.load_state_dict(filtered_state_dict, strict=False)
        print("Loaded pre-trained model.")

    # 进行测试
    avg_test_loss, avg_test_circle_loss, avg_test_recall, avg_test_recall_sum = test(model, test_dataloader, config, writer, 0)
    print(f"Test Circle Loss: {avg_test_circle_loss:.4f}, Recall: {avg_test_recall:.4f}, Recall Sum: {avg_test_recall_sum:.4f}")