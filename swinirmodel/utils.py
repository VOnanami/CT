"""
本模块用于从【已分割好】的训练、验证、测试数据集中加载数据。

核心功能:
1. 分别为训练、验证、测试集指定不同的低剂量(LQ)和高剂量(GT)的 .pkl 文件。
2. 为每个数据集加载并配对数据。
3. 使用 MONAI 构建统一的数据预处理管道。
4. 创建并返回可直接用于模型训练的、独立的训练、验证和测试 PyTorch DataLoaders。
"""

import pickle
import torch
from typing import Tuple, List, Dict

# MONAI 的导入
from torch.utils.data import DataLoader
from monai.data import CacheDataset
from monai.transforms import Compose, ScaleIntensityd,  LoadImaged, ToTensord, EnsureChannelFirstd

# 注意：我们不再需要 scikit-learn 的 train_test_split
# from sklearn.model_selection import train_test_split


def _load_and_pair_data_from_pkl(lq_pkl_path: str, gt_pkl_path: str) -> List[Dict]:
    """
    一个内部辅助函数，用于从一对LQ和GT的Pickle文件中加载和配对数据。
    """
    print(f"  -> 正在加载 LQ 数据: {lq_pkl_path}")
    print(f"  -> 正在加载 GT 数据: {gt_pkl_path}")
    
    try:
        with open(lq_pkl_path, 'rb') as f:
            lq_data = pickle.load(f)
        with open(gt_pkl_path, 'rb') as f:
            gt_data = pickle.load(f)
    except FileNotFoundError as e:
        print(f"错误: 找不到文件 {e.filename}。请检查路径。")
        raise e # 抛出异常，中断程序

    if len(lq_data) != len(gt_data):
        raise ValueError(f"数据列表长度不匹配！文件: {lq_pkl_path} ({len(lq_data)}) 与 {gt_pkl_path} ({len(gt_data)})")

    data_dicts = [
        {
            "lq_img": lq_array,
            "gt_img": gt_array,
            "filename": f"item_{i}"
        }
        for i, (lq_array, gt_array) in enumerate(zip(lq_data, gt_data))
    ]

    if not data_dicts:
        print(f"警告: 在文件 {lq_pkl_path} 和 {gt_pkl_path} 中未能配对任何数据。")
        
    return data_dicts


def create_dataloaders_from_split_pkl(
    data_paths: Dict[str, Dict[str, str]],
    batch_size: int,
    num_workers: int = 4,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    从已分割好的Pickle文件创建训练、验证和测试的DataLoader。

    Args:
        data_paths (Dict): 一个包含'train', 'val', 'test'三个键的字典。
                           每个键的值是另一个字典，包含'lq'和'gt'的路径。
        batch_size (int): 每个批次的数据量。
        num_workers (int): 用于数据加载的子进程数。
        random_seed (int): 用于训练集随机打乱的种子。

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: 分别返回训练、验证、测试的DataLoader。
    """
    # --- 步骤 1: 分别为训练、验证、测试集加载数据 ---
    print("步骤 1: 正在加载已分割的数据集...")
    print("加载训练数据...")
    train_files = _load_and_pair_data_from_pkl(data_paths['train']['lq'], data_paths['train']['gt'])
    print("加载验证数据...")
    val_files = _load_and_pair_data_from_pkl(data_paths['val']['lq'], data_paths['val']['gt'])
    print("加载测试数据...")
    test_files = _load_and_pair_data_from_pkl(data_paths['test']['lq'], data_paths['test']['gt'])
    
    print("\n数据集加载完成:")
    print(f"  - 训练样本: {len(train_files)}")
    print(f"  - 验证样本: {len(val_files)}")
    print(f"  - 测试样本: {len(test_files)}")

    # --- 步骤 2: 定义MONAI数据变换 ---
    print("\n步骤 2: 正在定义数据变换流程...")
    image_keys = ["lq_img", "gt_img"]
    
    # 训练和验证/测试使用相同的变换流程
    transforms = Compose([
        EnsureChannelFirstd(keys=image_keys, channel_dim="no_channel"),
        ScaleIntensityd(keys=image_keys),
        ToTensord(keys=image_keys),
    ])

    # --- 步骤 3: 创建Dataset和DataLoader ---
    print("\n步骤 3: 正在创建 Dataset 和 DataLoader...")
    train_ds = CacheDataset(data=train_files, transform=transforms, cache_rate=1.0)
    val_ds = CacheDataset(data=val_files, transform=transforms, cache_rate=1.0)
    test_ds = CacheDataset(data=test_files, transform=transforms, cache_rate=1.0)

    torch.manual_seed(random_seed)
    generator = torch.Generator().manual_seed(random_seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, generator=generator
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print("\n数据加载器已成功创建！")
    return train_loader, val_loader, test_loader


# ================== 主程序入口和使用示例 ==================
if __name__ == '__main__':
    # --- 在这里配置你的参数 ---
    # 这是一个示例结构，请将其修改为您自己的实际文件路径
    DATA_PATHS = {
        'train': {
            'lq': '/data/coding/datasets_final_numpy_pkl/train_LQ.pklv4',
            'gt': '/data/coding/datasets_final_numpy_pkl/train_GT.pklv4'
        },
        'val': {
            'lq': '/data/coding/datasets_final_numpy_pkl/val_LQ.pklv4',
            'gt': '/data/coding/datasets_final_numpy_pkl/val_GT.pklv4'
        },
        'test': {
            'lq': '/data/coding/datasets_final_numpy_pkl/test_LQ.pklv4',
            'gt': '/data/coding/datasets_final_numpy_pkl/test_GT.pklv4'
        }
    }
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    RANDOM_SEED = 42

    print("="*50)
    print("开始执行数据预处理流程 (已分割数据集)...")
    print("="*50)

    # 获取数据加载器
    train_loader, val_loader, test_loader = create_dataloaders_from_split_pkl(
        data_paths=DATA_PATHS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        random_seed=RANDOM_SEED
    )

    # --- 完整性检查 ---
    if train_loader and val_loader and test_loader:
        print(f"\n训练集批次数: {len(train_loader)}")
        print(f"验证集批次数: {len(val_loader)}")
        print(f"测试集批次数: {len(test_loader)}")

        print("\n--- 正在检查一个训练批次的数据 ---")
        sample_batch = next(iter(train_loader))
        print(f"LQ批次尺寸: {sample_batch['lq_img'].shape}")
        print(f"GT批次尺寸: {sample_batch['gt_img'].shape}")
