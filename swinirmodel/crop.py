import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import imageio
from sklearn.model_selection import train_test_split

# ==============================================================================
# 最终版脚本：
# 整合所有需求，从两个总文件夹和一个CSV规则文件，生成6个包含真实NumPy数组的PKL文件。
# !!! 警告：此脚本在最后一步会加载所有有效数据到内存，可能会消耗大量RAM !!!
# ==============================================================================

def build_filepath_map(root_dir: str, extension: str) -> Dict[str, str]:
    """
    递归扫描目录，创建一个从【不带扩展名的基础文件名】到【完整路径】的映射字典。
    这可以极大地加速后续的匹配过程。
    """
    print(f"正在扫描 {root_dir} 下的所有 {extension} 文件...")
    path_map = {}
    
    # 使用 os.walk 进行高效的递归文件查找
    for dirpath, _, filenames in tqdm(os.walk(root_dir), desc=f"扫描 {extension} 文件"):
        for filename in filenames:
            if filename.endswith(extension):
                full_path = os.path.join(dirpath, filename)
                basename_stem = Path(filename).stem
                path_map[basename_stem] = full_path
                
    print(f"扫描完成，找到 {len(path_map)} 个 {extension} 文件。")
    return path_map

def load_data_from_paths(paths: List[str], file_type: str) -> List[np.ndarray]:
    """根据路径列表，将所有图像或NPY文件加载为NumPy数组列表。"""
    print(f"正在从磁盘加载 {len(paths)} 个 {file_type} 文件到内存...")
    data_arrays = []
    for path in tqdm(paths, desc=f"加载 {file_type} 数据"):
        try:
            if file_type == 'png':
                data = imageio.imread(path)
            elif file_type == 'npy':
                data = np.load(path)
            else:
                continue # 不支持的类型
            data_arrays.append(data)
        except Exception as e:
            print(f"\n[警告] 读取文件 '{path}' 时出错，已跳过。错误: {e}")
    return data_arrays


def main():
    # --- 1. 请在这里配置您的所有输入和输出路径 ---
    GT_FOLDER = "/data/coding/Preprocessed_512x512/512/Full Dose/1mm/Sharp Kernel (D45)"
    LQ_FOLDER = "/data/coding/Quarter Dose lf"
    CSV_MATCHING_FILE = "/data/coding/1mmSHARP.csv"
    OUTPUT_DIRECTORY = "./datasets_final_numpy_pkl"
    RANDOM_SEED = 42
    
    # --- 2. 预处理：扫描文件夹，建立文件名到完整路径的索引 ---
    # 这一步非常快，而且节省内存
    gt_path_map = build_filepath_map(GT_FOLDER, extension=".png")
    lq_path_map = build_filepath_map(LQ_FOLDER, extension=".npy")

    # --- 3. 核心匹配逻辑：以CSV为准，筛选真实存在的图像对 ---
    print(f"\n--- 正在根据 {CSV_MATCHING_FILE} 进行匹配 ---")
    try:
        df = pd.read_csv(CSV_MATCHING_FILE)
    except FileNotFoundError:
        print(f"❌ 错误：找不到CSV文件 '{CSV_MATCHING_FILE}'。")
        return

    # 这两个列表将只存储最终被验证为有效的图像对的【完整路径】
    final_gt_paths: List[str] = []
    final_lq_paths: List[str] = []
    
    gt_col_name = df.columns[4]
    lq_col_name = df.columns[5]
    print(f"   将使用CSV列 '{gt_col_name}' (GT) 和 '{lq_col_name}' (LQ) 进行匹配。")

    for index, row in tqdm(df.iterrows(), total=len(df), desc="匹配CSV规则"):
        lq_relative_path = row[lq_col_name]
        gt_relative_path = row[gt_col_name]
        
        lq_basename_stem = Path(lq_relative_path).stem
        gt_basename_stem = Path(gt_relative_path).stem

        # 核心匹配条件：检查CSV中提取的基础名是否存在于我们扫描到的文件索引中
        if lq_basename_stem in lq_path_map and gt_basename_stem in gt_path_map:
            # 匹配成功！将这对文件的【完整路径】添加到最终列表中
            final_lq_paths.append(lq_path_map[lq_basename_stem])
            final_gt_paths.append(gt_path_map[gt_basename_stem])

    if not final_gt_paths:
        print("\n❌ 错误：未能根据CSV规则从您的文件夹中匹配到任何有效的图像对。")
        return
        
    print(f"✅ 匹配完成！共找到 {len(final_gt_paths)} 对有效的图像。")

    # --- 4. 按 7:2:1 比例同步切分【路径列表】---
    print("\n--- 正在切分数据集 ---")
    gt_train_val_paths, gt_test_paths, lq_train_val_paths, lq_test_paths = train_test_split(
        final_gt_paths, final_lq_paths, test_size=0.1, random_state=RANDOM_SEED
    )
    gt_train_paths, gt_val_paths, lq_train_paths, lq_val_paths = train_test_split(
        gt_train_val_paths, lq_train_val_paths, test_size=(2/9), random_state=RANDOM_SEED
    )
    print("✅ 路径列表切分完成。")
    print(f"   - 训练: {len(gt_train_paths)}, 验证: {len(gt_val_paths)}, 测试: {len(gt_test_paths)}")

    # --- 5. 加载数据并保存为 6 个独立的【Numpy数组列表】PKL文件 ---
    print("\n--- 正在加载数据并打包保存 (这一步会消耗大量内存) ---")
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    # 加载和保存训练集
    train_gt_arrays = load_data_from_paths(gt_train_paths, 'png')
    train_lq_arrays = load_data_from_paths(lq_train_paths, 'npy')
    with open(os.path.join(OUTPUT_DIRECTORY, "train_GT.pklv4"), 'wb') as f: pickle.dump(train_gt_arrays, f, protocol=4)
    with open(os.path.join(OUTPUT_DIRECTORY, "train_LQ.pklv4"), 'wb') as f: pickle.dump(train_lq_arrays, f, protocol=4)
    print("✅ 训练集 (GT & LQ) 保存完毕。")

    # 加载和保存验证集
    val_gt_arrays = load_data_from_paths(gt_val_paths, 'png')
    val_lq_arrays = load_data_from_paths(lq_val_paths, 'npy')
    with open(os.path.join(OUTPUT_DIRECTORY, "val_GT.pklv4"), 'wb') as f: pickle.dump(val_gt_arrays, f, protocol=4)
    with open(os.path.join(OUTPUT_DIRECTORY, "val_LQ.pklv4"), 'wb') as f: pickle.dump(val_lq_arrays, f, protocol=4)
    print("✅ 验证集 (GT & LQ) 保存完毕。")
    
    # 加载和保存测试集
    test_gt_arrays = load_data_from_paths(gt_test_paths, 'png')
    test_lq_arrays = load_data_from_paths(lq_test_paths, 'npy')
    with open(os.path.join(OUTPUT_DIRECTORY, "test_GT.pklv4"), 'wb') as f: pickle.dump(test_gt_arrays, f, protocol=4)
    with open(os.path.join(OUTPUT_DIRECTORY, "test_LQ.pklv4"), 'wb') as f: pickle.dump(test_lq_arrays, f, protocol=4)
    print("✅ 测试集 (GT & LQ) 保存完毕。")

    print("\n🎉 全部 6 个文件生成完毕！")


if __name__ == '__main__':
    main()
