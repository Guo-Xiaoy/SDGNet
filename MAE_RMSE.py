import pandas as pd
import numpy as np
import os
import _1_test

# 假设这是你的项目名称
project_name = 'SDGCNet'

# 读取预测结果
excel_filename = _1_test.EXCEL_PATH
pred_df = pd.read_excel(excel_filename, header=0)

# 读取真实值，没有表头
gt_df = pd.read_csv('./data/dataset/test_gt.csv', header=None)

# 确保两者的索引对齐
assert all(pred_df.index == gt_df.index), "索引不匹配，确保数据按相同的顺序排列"

# 定义列索引与名称的映射
column_names = ['Roll', 'Pitch', 'Yaw', 'X', 'Y', 'Z']
column_indices = [0, 1, 2, 3, 4, 5]

# 计算 MAE 和 RMSE
mae = {}
rmse = {}

for i, column_name in zip(column_indices, column_names):
    diff = pred_df[column_name] - gt_df[i]
    mae[column_name] = np.mean(np.abs(diff))
    rmse[column_name] = np.sqrt(np.mean(diff**2))

# 将结果转换为 DataFrame
errors_df = pd.DataFrame([mae, rmse], index=['MAE', 'RMSE'])

# 构建输出文件名
base_name = os.path.basename(_1_test.EXCEL_PATH)  # 获取文件名 test_RT_result_NoNoise.xlsx
project_name = base_name.split("_")[1]  # 假设 project_name 是从文件名中提取的部分
suffix = base_name.split("_")[-1].replace('.xlsx', '')  # 提取_NoNoise部分

# 动态生成输出文件名
output_filename = f'./test_result/{project_name}_test_error_{suffix}.csv'

# 保存结果到 CSV 文件
errors_df.to_csv(output_filename)

# 打印结果
print("Mean Absolute Error (MAE):")
for key, value in mae.items():
    print(f"{key}: {value:.4f}")

print("\nRoot Mean Square Error (RMSE):")
for key, value in rmse.items():
    print(f"{key}: {value:.4f}")

print(f"\nResults have been saved to {output_filename}")
