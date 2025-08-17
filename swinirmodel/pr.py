import pandas as pd
import cv2
import pywt
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# --- 1. 设置基本参数 ---
CSV_FILE_PATH = '1mmSHARP.csv'
PATH_PREFIX = 'Preprocessed_512x512'

# 定义四个输出文件夹的名称
output_dirs = {
    'Full Dose': {
        'lf': 'Full Dose lf',
        'hf': 'Full Dose hf'
    },
    'Quarter Dose': {
        'lf': 'Quarter Dose lf',
        'hf': 'Quarter Dose hf'
    }
}

# --- 2. 创建输出文件夹 ---
print("正在创建输出文件夹...")
for dose_type in output_dirs.values():
    for dir_path in dose_type.values():
        os.makedirs(dir_path, exist_ok=True)
print("文件夹创建完成。")

# --- 3. 读取CSV文件 ---
try:
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"成功读取CSV文件: {CSV_FILE_PATH}")
except FileNotFoundError:
    print(f"错误: 找不到CSV文件 '{CSV_FILE_PATH}'。请确保脚本与CSV文件在同一目录下。")
    exit()

# 定义要处理的列和对应的文件夹映射
column_mapping = {
    'Full Dose Filepath': output_dirs['Full Dose'],
    'Quarter Dose Filepath': output_dirs['Quarter Dose']
}

# --- 4. 循环处理每一列的图像 ---
for column_name, dirs in column_mapping.items():
    print(f"\n--- 开始处理列: {column_name} ---")
    
    # 检查列是否存在
    if column_name not in df.columns:
        print(f"警告: CSV文件中找不到列 '{column_name}'，将跳过。")
        continue

    # 使用tqdm创建进度条
    for original_path in tqdm(df[column_name].dropna(), desc=f"处理 {column_name}"):
        try:
            # --- a. 拼接完整路径 ---
            full_image_path = os.path.join(PATH_PREFIX, original_path)

            # --- b. 读取图像 ---
            # 以灰度模式读取
            img = cv2.imread(full_image_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"\n警告: 无法读取图像，路径可能无效: {full_image_path}")
                continue

            # --- c. 进行Haar小波分解 ---
            coeffs = pywt.dwt2(img, 'haar')
            cA, (cH, cV, cD) = coeffs

            # --- d. 准备保存路径和文件名 ---
            # 获取不带扩展名的原始文件名，例如 L067_FD_1_SHARP_1...
            base_filename = Path(original_path).stem
            
            # --- e. 保存低频分量 (LL) ---
            ll_save_path = os.path.join(dirs['lf'], f"{base_filename}.npy")
            np.save(ll_save_path, cA)

            # --- f. 打包并保存高频分量 (HF) ---
            # 将三个高频分量沿着最后一个维度堆叠，形成 (H, W, 3) 的形状
            hf_stacked = np.stack([cH, cV, cD], axis=-1)
            hf_save_path = os.path.join(dirs['hf'], f"{base_filename}.npy")
            np.save(hf_save_path, hf_stacked)

        except Exception as e:
            print(f"\n处理文件时发生错误: {original_path}。错误信息: {e}")

print("\n--- 所有处理已完成！ ---")
