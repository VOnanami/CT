#generally train
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

        reconstructed_images_np = reconstructed_images.cpu().numpy()

        for i in range(reconstructed_images_np.shape[0]):
          
            img_np = reconstructed_images_np[i] 
            filename = original_filenames[i]
            
           
            save_path = output_dir / f"{Path(filename).stem}_reconstructed.npy"
            
           
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
    for step, x in pbar:
        noisy_images = x["low_dose_img"].to(device)
        target_images = x["high_dose_img"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda')):
            reconstruction = model(noisy_images)
            loss = F.l1_loss(reconstruction.float(), target_images.float())

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # <--- 新增
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if writer is not None:
            writer.add_scalar("lr", get_lr(optimizer), epoch * len(loader) + step)
            writer.add_scalar("train/loss_mae", loss.item(), epoch * len(loader) + step)
        
        pbar.set_postfix(mae_loss=f"{loss.item():.4f}", lr=f"{get_lr(optimizer):.6f}")

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
            
            # --- 核心修改在这里 ---
            # 1. 动态计算当前批次目标图像的动态范围
            data_range = target_images.max() - target_images.min()
            
            # 2. 安全检查：如果 data_range 几乎为0 (图像是平坦的)，则该批次的PSNR/SSIM无意义
            if data_range < 1e-6:
               
                psnr = torch.tensor(0.0, device=device)
                ssim = torch.tensor(0.0, device=device)
            else:
             
                psnr_metric = PSNRMetric(max_val=data_range.item())
                ssim_metric = SSIMMetric(data_range=data_range, spatial_dims=2)
                psnr = psnr_metric(reconstruction, target_images)
                ssim = ssim_metric(reconstruction, target_images)
            
        
            mae = F.l1_loss(reconstruction, target_images)
            mse = F.mse_loss(reconstruction, target_images)
            # --- 结束修改 ---
            
      
        batch_size = noisy_images.shape[0]
        total_mae += mae.item() * batch_size
        total_mse += mse.item() * batch_size
        total_psnr += psnr.mean().item() * batch_size
        total_ssim += ssim.mean().item() * batch_size

  
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
      
        raise ValueError("验证集中发现损坏或无法处理的数据，请清理后重试。")
    else:
        print("✅ [启动成功] 数据集健康检查通过，未发现问题。开始正式训练...")

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
            
           
            if torch.isnan(torch.tensor(val_mae)):
                print(f"Epoch {epoch + 1} | 验证时出现NaN！这可能是一个模型内部问题（如BatchNorm）。训练停止。")
                # 可以在这里提前结束训练
                break

            print(
                f"Epoch {epoch + 1} | Val MAE: {val_mae:.4f} | Val MSE: {val_metrics['mse']:.4f} | "
                f"Val PSNR: {val_metrics['psnr']:.2f} dB | Val SSIM: {val_metrics['ssim']:.4f}"
            )
            
          
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
    
  
    save_test_outputs(
        model=model, # 使用已加载了最佳权重的模型
        loader=test_loader,
        device=device,
        output_dir=output_save_dir
    )
    print("✅ All test outputs have been saved.")
    return best_mae_loss
