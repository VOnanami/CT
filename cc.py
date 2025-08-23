import os
import pandas as pd
from pathlib import Path
import torch
from monai.data import DataLoader, Dataset, CacheDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityd,
    ToTensord,
)
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
from monai.metrics import PSNRMetric, SSIMMetric
from tqdm import tqdm

# ==============================================================================
# 函数 1: 创建 Train, Val, Test 三个 DataLoader
# ==============================================================================
def create_train_val_test_dataloaders(
    csv_path: str,
    data_prefix: str,
    batch_size: int,
    num_workers: int,
    random_seed: int
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    根据CSV文件和数据前缀，创建训练、验证和测试的DataLoader。
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"错误: 找不到CSV文件 '{csv_path}'。跳过此数据集。")
        return None, None, None

    data_dicts = []
    required_cols = ['Quarter Dose Filepath', 'Full Dose Filepath']
    if not all(col in df.columns for col in required_cols):
        print(f"错误: CSV文件 '{csv_path}' 必须包含 {required_cols} 列。")
        return None, None, None
        
    for index, row in df.iterrows():
        full_low_dose_path = os.path.join(data_prefix, row['Quarter Dose Filepath'])
        full_high_dose_path = os.path.join(data_prefix, row['Full Dose Filepath'])
        
        if os.path.exists(full_low_dose_path) and os.path.exists(full_high_dose_path):
            data_dicts.append({
                "low_dose_img": full_low_dose_path,
                "high_dose_img": full_high_dose_path,
            })

    if not data_dicts:
        print(f"错误: 在路径 '{data_prefix}' 下未能根据CSV '{csv_path}' 找到任何配对的图像文件。")
        return None, None, None

    # 按照 7:2:1 的比例切分数据集
    train_val_files, test_files = train_test_split(
        data_dicts, test_size=0.1, random_state=random_seed
    )
    train_files, val_files = train_test_split(
        train_val_files, test_size=(2/9.0), random_state=random_seed # 2/9 of the remaining 90% is 20% of the total
    )
    
    print(f"数据集切分完成: {len(data_dicts)} 个总样本")
    print(f" -> 训练集: {len(train_files)} | 验证集: {len(val_files)} | 测试集: {len(test_files)}")

    # 对所有子集使用相同的基础变换 (无数据增强)
    transforms = Compose([
        LoadImaged(keys=["low_dose_img", "high_dose_img"], image_only=True, ensure_channel_first=True),
        ScaleIntensityd(keys=["low_dose_img", "high_dose_img"]),
        ToTensord(keys=["low_dose_img", "high_dose_img"]),
    ])

    train_ds = CacheDataset(data=train_files, transform=transforms, cache_rate=1.0)
    val_ds = CacheDataset(data=val_files, transform=transforms, cache_rate=1.0)
    test_ds = CacheDataset(data=test_files, transform=transforms, cache_rate=1.0)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

# ==============================================================================
# 函数 2: 计算基线指标 (无需修改)
# ==============================================================================
def calculate_baseline_metrics(loader: DataLoader, device: torch.device):
    """
    遍历DataLoader，计算低剂量图像和高剂量图像之间的平均PSNR和SSIM。
    """
    total_psnr, total_ssim, num_samples = 0.0, 0.0, 0
    ssim_metric = SSIMMetric(data_range=1.0, spatial_dims=2)
    psnr_metric = PSNRMetric(max_val=1.0)
    
    pbar = tqdm(loader, desc="计算指标中", leave=False)

    with torch.no_grad():
        for batch_data in pbar:
            low_dose_imgs = batch_data["low_dose_img"].to(device)
            high_dose_imgs = batch_data["high_dose_img"].to(device)
            
            batch_size = low_dose_imgs.shape[0]
            psnr_val = psnr_metric(y_pred=low_dose_imgs, y=high_dose_imgs)
            ssim_val = ssim_metric(y_pred=low_dose_imgs, y=high_dose_imgs)
            
            total_psnr += psnr_val.mean().item() * batch_size
            total_ssim += ssim_val.mean().item() * batch_size
            num_samples += batch_size

    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    
    print(f"  -> 平均 PSNR: {avg_psnr:.2f} dB")
    print(f"  -> 平均 SSIM: {avg_ssim:.4f}")

# ==============================================================================
# 3. 主程序入口
# ==============================================================================
if __name__ == "__main__":
    
    # --- 1. 在这里配置您所有的数据集 ---
    datasets_to_process = [
        {
            "name": "1mm Sharp Kernel (D45)",
            "csv_path": "/data/coding/1mmSHARP.csv",
            "data_prefix": "/data/coding/Preprocessed_512x512"
        },
        # {
        #     "name": "另一个数据集 (例如: 1mm Soft)",
        #     "csv_path": "/data/coding/1mmSOFT.csv",  # <--- 替换为您的第二个CSV路径
        #     "data_prefix": "/data/coding/Preprocessed_512x512" # <--- 通常前缀相同
        # },
        # {
        #     "name": "第三个数据集 (例如: 3mm Sharp)",
        #     "csv_path": "/data/coding/3mmSHARP.csv", # <--- 替换为您的第三个CSV路径
        #     "data_prefix": "/data/coding/Preprocessed_512x512"
        # },
    ]
    
    # --- 2. 配置通用参数 ---
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    RANDOM_SEED = 42
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")

    # --- 3. 循环处理所有配置好的数据集 ---
    for dataset_info in datasets_to_process:
        print(f"========== 处理数据集: {dataset_info['name']} ==========")
        
        train_loader, val_loader, test_loader = create_train_val_test_dataloaders(
            csv_path=dataset_info['csv_path'],
            data_prefix=dataset_info['data_prefix'],
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            random_seed=RANDOM_SEED
        )

        if train_loader and val_loader and test_loader:
            print("\n[1/3] 计算 训练集 (Train Set) 基线指标...")
            calculate_baseline_metrics(loader=train_loader, device=device)
            
            print("\n[2/3] 计算 验证集 (Validation Set) 基线指标...")
            calculate_baseline_metrics(loader=val_loader, device=device)

            print("\n[3/3] 计算 测试集 (Test Set) 基线指标...")
            calculate_baseline_metrics(loader=test_loader, device=device)
        
        print(f"========== '{dataset_info['name']}' 处理完成 ==========\n")
