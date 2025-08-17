# train.py
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from typing import Dict
from monai.metrics import SSIMMetric, PSNRMetric
import numpy as np
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
@torch.no_grad()
def save_test_outputs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
):
    """
    对测试集进行前向传播，并将模型的输出结果直接保存为 .npy 文件。

    Args:
        model (nn.Module): 训练好的模型。
        loader (DataLoader): 测试集的DataLoader。
        device (torch.device): 计算设备 (cpu 或 cuda)。
        output_dir (Path): 保存输出.npy文件的文件夹路径。
    """
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"测试集输出结果将保存为 .npy 文件到: {output_dir}")

    pbar = tqdm(loader, desc="正在保存测试集输出 (.npy)")

    for batch in pbar:
        input_images = batch["low_dose_img"].to(device)
        original_filenames = batch["filename"]

        with autocast(device_type=device.type if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda')):
            reconstructed_images = model(input_images)

        reconstructed_images_np = reconstructed_images[-1].cpu().numpy()

        for i in range(reconstructed_images_np.shape[0]):
            # 获取单张图像的Numpy数组和对应的文件名
            # 数组形状通常是 (C, H, W)，例如 (1, 256, 256)
            img_np = reconstructed_images_np[i] 
            filename = original_filenames[i]
            
            # --- 主要改动在这里 ---
            # 1. 无需进行任何后处理，直接保存原始的浮点数数组
            
            # 2. 构建新的保存路径，扩展名为 .npy
            save_path = output_dir / f"{Path(filename).stem}_reconstructed.npy"
            
            # 3. 使用 np.save 保存数组
            np.save(save_path, img_np)
def train_epoch_swin(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    scaler: GradScaler,
) -> None:
    model.train()
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}")
    
    # 关键：让最终输出（最后一个元素）权重最大，中间层权重递减
    # 假设模型输出4个结果（可根据实际输出数量调整）
    num_outputs = 4  # 需与模型实际输出数量一致
    # 权重分配：最后一个输出占比 > 60%，前面按0.1, 0.2, 0.3...递增
    deep_sup_weights = [0.1, 0.2, 0.3, 0.4]  # 总和为1.0，最后一个权重最大
    # 若输出数量不同，例如3个输出：[0.1, 0.2, 0.7]（最后一个占70%）

    for step, x in pbar:
        noisy_images = x["low_dose_img"].to(device)
        target_images = x["high_dose_img"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda')):
            reconstructions = model(noisy_images)  # 列表：[out1, out2, ..., outN]
            
            # 自动适配输出数量（推荐！无需手动修改num_outputs）
            num_outputs = len(reconstructions)
            deep_sup_weights = [i / sum(range(1, num_outputs+1)) for i in range(1, num_outputs+1)]
            # 上面一行等价于：输出n个时，权重为[1,2,...,n]/(1+2+...+n)，最后一个权重最大
            # 例：4个输出 → [1/10, 2/10, 3/10, 4/10]，最后一个占40%
            
            total_loss = 0.0
            for out, weight in zip(reconstructions, deep_sup_weights):
                # 确保中间输出与目标尺寸一致（如需上采样）
                if out.shape[2:] != target_images.shape[2:]:
                    out = F.interpolate(
                        out, 
                        size=target_images.shape[2:], 
                        mode='bilinear' if target_images.ndim == 4 else 'trilinear',
                        align_corners=True
                    )
                loss_i = F.l1_loss(out.float(), target_images.float())
                total_loss += weight * loss_i

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if writer is not None:
            writer.add_scalar("lr", get_lr(optimizer), epoch * len(loader) + step)
            writer.add_scalar("train/loss_mae", total_loss.item(), epoch * len(loader) + step)
            # 可选：记录最终输出的损失（方便观察关键指标）
            final_loss = F.l1_loss(reconstructions[-1].float(), target_images.float())
            writer.add_scalar("train/final_output_loss", final_loss.item(), epoch * len(loader) + step)
        
        pbar.set_postfix(
            total_mae=f"{total_loss.item():.4f}",
            final_mae=f"{final_loss.item():.4f}",  # 显示最终输出的损失
            lr=f"{get_lr(optimizer):.6f}"
        )

@torch.no_grad()
def eval_or_test(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    step: int = 0,
    writer: SummaryWriter = None,
    eval_mode: str = "val"
) -> Dict[str, float]:
    model.eval()
    
    total_mae = 0.0
    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    
    pbar_desc = "Validating" if eval_mode == "val" else "Testing"
    pbar = tqdm(loader, desc=pbar_desc)

    for x in pbar:
        noisy_images = x["low_dose_img"].to(device)
        target_images = x["high_dose_img"].to(device)

        with autocast(device_type=device.type if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda')):
            reconstruction = model(noisy_images)
            reconstruction = reconstruction[-1]
            # --- 核心修改在这里 ---
            # 1. 动态计算当前批次目标图像的动态范围
            data_range = target_images.max() - target_images.min()
            
            # 2. 安全检查：如果 data_range 几乎为0 (图像是平坦的)，则该批次的PSNR/SSIM无意义
            if data_range < 1e-6:
                # 对于这个特殊的批次，我们跳过PSNR/SSIM计算，或者给一个合理的值
                # 这里我们简单地认为这个批次的PSNR/SSIM贡献为0
                psnr = torch.tensor(0.0, device=device)
                ssim = torch.tensor(0.0, device=device)
            else:
                # 只有在data_range有效时才进行计算
                psnr_metric = PSNRMetric(max_val=data_range.item())
                ssim_metric = SSIMMetric(data_range=data_range, spatial_dims=2)
                psnr = psnr_metric(reconstruction, target_images)
                ssim = ssim_metric(reconstruction, target_images)
            
            # MAE 和 MSE 的计算不受影响
            mae = F.l1_loss(reconstruction, target_images)
            mse = F.mse_loss(reconstruction, target_images)
            # --- 结束修改 ---
            
        # 累加每个批次的结果
        batch_size = noisy_images.shape[0]
        total_mae += mae.item() * batch_size
        total_mse += mse.item() * batch_size
        total_psnr += psnr.mean().item() * batch_size
        total_ssim += ssim.mean().item() * batch_size

    # 计算最终的平均值
    n_samples = len(loader.dataset)
    avg_mae = total_mae / n_samples
    avg_mse = total_mse / n_samples
    avg_psnr = total_psnr / n_samples
    avg_ssim = total_ssim / n_samples

    if writer is not None:
        writer.add_scalar(f"{eval_mode}/mae", avg_mae, step)
        writer.add_scalar(f"{eval_mode}/mse", avg_mse, step)
        writer.add_scalar(f"{eval_mode}/psnr", avg_psnr, step)
        writer.add_scalar(f"{eval_mode}/ssim", avg_ssim, step)

    return {
        "mae": avg_mae, "mse": avg_mse, "psnr": avg_psnr, "ssim": avg_ssim
    }
def train_swin(
    model: nn.Module,
    start_epoch: int,
    best_mae_loss: float,
    train_loader: DataLoader,
    val_loader: DataLoader,     # <--- 我们将从这里获取验证集
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    eval_freq: int,
    scheduler: _LRScheduler,
    writer_train: SummaryWriter,
    writer_val: SummaryWriter,
    device: torch.device,
    run_dir: Path,
) -> float:
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    raw_model = model.module if hasattr(model, "module") else model

    # ==============================================================================
    # --- 新增：在训练开始前，对验证集进行一次性的健康扫描 ---
    # ==============================================================================
    print("\n--- [启动检查] 正在扫描验证集中的潜在问题数据 ---")
    
    # 从 DataLoader 中获取其底层的 Dataset 对象
    val_ds = val_loader.dataset 
    
    problematic_files_info = []
    # 使用 tqdm 创建进度条
    for i in tqdm(range(len(val_ds)), desc="扫描验证集"):
        try:
            # val_ds[i] 会自动应用 val_test_transforms
            sample = val_ds[i]
            
            # 检查 sample 中的每一个张量
            for key, tensor in sample.items():
                if isinstance(tensor, torch.Tensor): # 只检查张量
                    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                        # 从 val_ds.data 访问原始的文件路径字典
                        problem_info = val_ds.data[i]
                        problematic_files_info.append(problem_info)
                        # 找到一个问题就足够了，跳出内层循环
                        break 
        except Exception as e:
            problem_info = val_ds.data[i]
            print(f"\n处理 Index {i} 时遇到加载或变换错误: {e}, 文件信息: {problem_info}")
            problematic_files_info.append(problem_info)

    if problematic_files_info:
        print("\n\n" + "="*80)
        print(f"🚨 [启动失败] 检查到 {len(problematic_files_info)} 组问题数据，训练已停止。")
        print("请从您的CSV文件中移除以下这些数据对应的行，然后重新开始训练：")
        print("="*80)
        for info in problematic_files_info:
            print(info)
        print("="*80)
        # 抛出异常，中断整个程序
        raise ValueError("验证集中发现损坏或无法处理的数据，请清理后重试。")
    else:
        print("✅ [启动成功] 数据集健康检查通过，未发现问题。开始正式训练...")
    # --- 扫描逻辑结束 ---


    # --- 原有的训练循环保持不变 ---
    for epoch in range(start_epoch, n_epochs):
        train_epoch_swin(
            model=model, loader=train_loader, optimizer=optimizer,
            device=device, epoch=epoch, writer=writer_train, scaler=scaler,
        )

        if (epoch + 1) % eval_freq == 0:
            val_metrics = eval_or_test(
                model=model, loader=val_loader, device=device,
                step=len(train_loader) * (epoch + 1), writer=writer_val, eval_mode="val"
            )
            val_mae = val_metrics["mae"]
            
            # 如果验证过程中仍然出现NaN，说明问题可能来自模型内部（如BatchNorm）
            if torch.isnan(torch.tensor(val_mae)):
                print(f"Epoch {epoch + 1} | 验证时出现NaN！这可能是一个模型内部问题（如BatchNorm）。训练停止。")
                # 可以在这里提前结束训练
                break

            print(
                f"Epoch {epoch + 1} | Val MAE: {val_mae:.4f} | Val MSE: {val_metrics['mse']:.4f} | "
                f"Val PSNR: {val_metrics['psnr']:.2f} dB | Val SSIM: {val_metrics['ssim']:.4f}"
            )
            
            # ... (后续的保存模型、学习率调度等逻辑完全保持不变) ...
            if val_mae < best_mae_loss:
                print(f"✅ New best val MAE: {val_mae:.4f}. Saving model...")
                best_mae_loss = val_mae
                torch.save(raw_model.state_dict(), run_dir / "best_model.pth")
            
            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_loss": best_mae_loss,
            }
            torch.save(checkpoint, run_dir / "checkpoint.pth")

            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_mae) 
            else:
                scheduler.step()

    # ... (训练结束后的最终测试和保存逻辑也完全保持不变) ...
    print("🏁 Training finished!")
    print("Saving final model...")
    torch.save(raw_model.state_dict(), run_dir / "final_model.pth")

    print("\n--- Running Final Evaluation on Test Set ---")
    best_model_path = run_dir / "best_model.pth"
    if best_model_path.exists():
        print(f"Loading best model from {best_model_path}")
        raw_model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("No best model found, using final model for testing.")

    test_metrics = eval_or_test(model=model, loader=test_loader, device=device, eval_mode="test")
    print("\n--- Test Set Results ---")
    print(f"  PSNR: {test_metrics['psnr']:.2f} dB")
    print(f"  SSIM: {test_metrics['ssim']:.4f}")
    print(f"  MAE:  {test_metrics['mae']:.4f}")
    print(f"  MSE:  {test_metrics['mse']:.4f}")
    print("------------------------\n")
    output_save_dir = run_dir / "test_outputs"
    
    # 调用函数，传入最好的模型、测试集加载器、设备和输出路径
    save_test_outputs(
        model=model, # 使用已加载了最佳权重的模型
        loader=test_loader,
        device=device,
        output_dir=output_save_dir
    )
    print("✅ All test outputs have been saved.")
    return best_mae_loss
