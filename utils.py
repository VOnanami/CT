#generally utils
import os
import pandas as pd
from pathlib import Path
import torch
from monai.data import DataLoader, Dataset, CacheDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ToTensord,
    ScaleIntensityd
)
from sklearn.model_selection import train_test_split
from typing import Tuple

# === 函数签名已简化 ===
def create_train_val_test_dataloaders_from_csv(
    csv_path: str,
    data_prefix: str,  # <--- 只需要一个统一的前缀路径
    batch_size: int,
    num_workers: int,
    random_seed: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    根据CSV文件中的路径和统一的数据前缀，配对图像并创建DataLoader。
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"错误: 找不到CSV文件 '{csv_path}'。")
        return None, None, None

    data_dicts = []
    print("正在根据CSV和数据前缀拼接完整路径...")
    
    for index, row in df.iterrows():
        # === 核心逻辑：直接拼接前缀和CSV路径 ===
        low_dose_path_from_csv = row['Quarter Dose Filepath']
        high_dose_path_from_csv = row['Full Dose Filepath']
        
        # 构建完整路径，例如: '/data/coding/Preprocessed_512x512/' + '512/Quarter Dose/...'
        full_low_dose_path = os.path.join(data_prefix, low_dose_path_from_csv)
        full_high_dose_path = os.path.join(data_prefix, high_dose_path_from_csv)
        
        # 检查拼接后的完整路径是否存在
        if os.path.exists(full_low_dose_path) and os.path.exists(full_high_dose_path):
            data_dicts.append({
                "low_dose_img": full_low_dose_path,
                "high_dose_img": full_high_dose_path,
                "filename": Path(full_low_dose_path).name # 提取纯文件名用于保存
            })

    if not data_dicts:
        print(f"数据前缀: {data_prefix}")
        print("CSV中的一个路径示例: ", df.iloc[0]['Quarter Dose Filepath'])
        print("拼接后的一个路径示例: ", os.path.join(data_prefix, df.iloc[0]['Quarter Dose Filepath']))
        raise ValueError("未能找到任何配对的图像文件。请仔细检查'data_prefix'是否正确，以及CSV路径与实际文件是否匹配。")
    image_keys=["low_dose_img", "high_dose_img"]
    # (后续所有代码保持不变)
    # 步骤2: 按 7:2:1 比例切分数据集
    train_val_files, test_files = train_test_split(
        data_dicts, test_size=0.1, random_state=random_seed
    )
    train_files, val_files = train_test_split(
        train_val_files, test_size=(2/9), random_state=random_seed
    )
    
    print(f"数据集切分完成：")
    print(f" - 找到 {len(data_dicts)} 个有效样本对。")
    print(f" - 训练样本: {len(train_files)}")
    print(f" - 验证样本: {len(val_files)}")
    print(f" - 测试样本: {len(test_files)}")

    # 步骤3: 定义MONAI数据变换
    train_transforms = Compose([
      LoadImaged(keys=image_keys, image_only=True, ensure_channel_first=True),
      ScaleIntensityd(keys=image_keys),
      ToTensord(keys=image_keys),
    ])
    val_test_transforms = Compose([
      LoadImaged(keys=image_keys, image_only=True, ensure_channel_first=True),
      ScaleIntensityd(keys=image_keys),
      ToTensord(keys=image_keys),
    ])

    # 步骤4 & 5: 创建Dataset和DataLoader
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
    val_ds = CacheDataset(data=val_files, transform=val_test_transforms, cache_rate=1.0)
    test_ds = CacheDataset(data=test_files, transform=val_test_transforms, cache_rate=1.0)
    torch.manual_seed(random_seed)
    generator = torch.Generator().manual_seed(random_seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
