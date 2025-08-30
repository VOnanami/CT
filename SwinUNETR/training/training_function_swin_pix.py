""" Training functions for the different models. """
from collections import OrderedDict
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from generative.losses.adversarial_loss import PatchAdversarialLoss
# from pynvml.smi import nvidia_smi

from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from util import log_ldm_sample_unconditioned, log_reconstructions, log_ddpm_sample
from generative.losses.perceptual import PerceptualLoss

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
# ----------------------------------------------------------------------------------------------------------------------
# SWINUNET
# ----------------------------------------------------------------------------------------------------------------------

def train_epoch_swin(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    scaler: GradScaler, # Pass scaler to the function
    input_key: str,
    target_key: str,
) -> None:
    model.train()
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}")
    for step, batch in pbar:
        inputs = batch[input_key].to(device)
        targets = batch[target_key].to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=(device.type == 'cuda')):
            reconstruction = model(inputs)
            # The only loss being calculated is L1
            loss = F.l1_loss(reconstruction.float(), targets.float())

        scaler.scale(loss).backward()
        # Optional: Gradient clipping
        # scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if writer is not None:
            writer.add_scalar("lr", get_lr(optimizer), epoch * len(loader) + step)
            writer.add_scalar("train/loss", loss.item(), epoch * len(loader) + step)

        pbar.set_postfix(loss=f"{loss.item():.6f}", lr=f"{get_lr(optimizer):.6f}")

@torch.no_grad()
def eval_swin(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    step: int,
    writer: SummaryWriter,
    input_key: str,
    target_key: str,

) -> float:
    model.eval()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Validation"):
        inputs = batch[input_key].to(device)
        targets = batch[target_key].to(device)
        #print(f"DEBUG: Shape of '{input_key}' tensor fed to model: {inputs.shape}")

        with autocast(enabled=(device.type == 'cuda')):
            reconstruction = model(inputs)
            loss = F.l1_loss(reconstruction.float(), targets.float())

        total_loss += loss.item() * inputs.shape[0]

    avg_loss = total_loss / len(loader.dataset)

    if writer is not None:
        writer.add_scalar("val/loss", avg_loss, step)
        log_reconstructions(
            image=targets,
            reconstruction=reconstruction,
            writer=writer,
            step=step,
            title=f"RECONSTRUCTION_{target_key.upper()}"
        )

    return avg_loss

def train_swin(
    model: nn.Module,
    start_epoch: int,
    best_loss: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    eval_freq: int,
    scheduler: _LRScheduler,
    writer_train: SummaryWriter,
    writer_val: SummaryWriter,
    device: torch.device,
    run_dir: Path, # FIX 1: Type hint is now Path
    input_key: str,
    target_key: str,
) -> float:
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    raw_model = model.module if hasattr(model, "module") else model

    # Initial evaluation before training starts
    val_loss = eval_swin(
        model=model, loader=val_loader, device=device, step=0, writer=writer_val,input_key=input_key, target_key=target_key
    )
    print(f"Initial val loss: {val_loss:.6f}")

    for epoch in range(start_epoch, n_epochs):
        train_epoch_swin(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            writer=writer_train,
            scaler=scaler,
            input_key=input_key,
            target_key=target_key
        )

        if (epoch + 1) % eval_freq == 0:
            val_loss = eval_swin(
                model=model,
                loader=val_loader,
                device=device,
                step=len(train_loader) * (epoch + 1),
                writer=writer_val,
                input_key=input_key,
                target_key=target_key
            )
            print(f"Epoch {epoch + 1} val loss: {val_loss:.6f}")
            #print_gpu_memory_report()

            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_loss": best_loss,
            }
            torch.save(checkpoint, run_dir / "checkpoint.pth")

            if val_loss < best_loss:
                print(f"✅ New best val loss {val_loss:.6f}. Saving model...")
                best_loss = val_loss
                torch.save(raw_model.state_dict(), run_dir / "best_model.pth")

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

    print("🏁 Training finished!")
    print("Saving final model...")
    torch.save(raw_model.state_dict(), run_dir / "final_model.pth")

    return best_loss


# ----------------------------------------------------------------------------------------------------------------------
# PIX2PIX
# ----------------------------------------------------------------------------------------------------------------------
def train_epoch_gan(
        model: nn.Module,
        d_model: nn.Module,
        loader: DataLoader,
        optimizer_g: torch.optim.Optimizer,
        optimizer_d: torch.optim.Optimizer,
        loss_gan_criterion: nn.Module,
        lambda_l1: float,
        device: torch.device,
        epoch: int,
        writer: SummaryWriter,
        scaler_g: GradScaler,
        scaler_d: GradScaler,
) -> None:
    model.train()
    d_model.train()
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"GAN Epoch {epoch}")

    for step, x in pbar:
        low_dose_images = x["lowct"].to(device)
        high_dose_images = x["highct"].to(device)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_d.zero_grad(set_to_none=True)

        with autocast(enabled=(device.type == 'cuda')):
            fake_images = model(x=low_dose_images).detach()

            # --- 关键修正: 处理嵌套列表输出 ---
            # Real image pair
            real_pair = torch.cat((low_dose_images, high_dose_images), 1)
            # d_model output is like [[tensor1, tensor2, ...]]
            # We need the inner list of tensors, which is the first element.
            pred_real_list = d_model(real_pair)[0]
            loss_d_real = 0
            for pred_real in pred_real_list:
                loss_d_real += loss_gan_criterion(pred_real, torch.ones_like(pred_real))

            # Fake image pair
            fake_pair = torch.cat((low_dose_images, fake_images), 1)
            pred_fake_list = d_model(fake_pair)[0]
            loss_d_fake = 0
            for pred_fake in pred_fake_list:
                loss_d_fake += loss_gan_criterion(pred_fake, torch.zeros_like(pred_fake))

            loss_d = (loss_d_real + loss_d_fake) * 0.5

        scaler_d.scale(loss_d).backward()
        scaler_d.step(optimizer_d)
        scaler_d.update()

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_g.zero_grad(set_to_none=True)

        with autocast(enabled=(device.type == 'cuda')):
            generated_images = model(x=low_dose_images)

            fake_pair_for_g = torch.cat((low_dose_images, generated_images), 1)
            pred_fake_list_for_g = d_model(fake_pair_for_g)[0]
            loss_g_gan = 0
            for pred_fake_for_g in pred_fake_list_for_g:
                loss_g_gan += loss_gan_criterion(pred_fake_for_g, torch.ones_like(pred_fake_for_g))

            loss_g_l1 = F.l1_loss(generated_images.float(), high_dose_images.float())
            loss_g = loss_g_gan + (loss_g_l1 * lambda_l1)

        scaler_g.scale(loss_g).backward()
        scaler_g.step(optimizer_g)
        scaler_g.update()

        # --- Logging ---
        if writer is not None:
            global_step = epoch * len(loader) + step
            writer.add_scalar("lr/generator", get_lr(optimizer_g), global_step)
            writer.add_scalar("lr/discriminator", get_lr(optimizer_d), global_step)
            writer.add_scalar("train/loss_g", loss_g.item(), global_step)
            writer.add_scalar("train/loss_d", loss_d.item(), global_step)
            writer.add_scalar("train/loss_g_gan", loss_g_gan.item(), global_step)
            writer.add_scalar("train/loss_g_l1", loss_g_l1.item(), global_step)

        pbar.set_postfix(
            G_loss=f"{loss_g.item():.4f}",
            D_loss=f"{loss_d.item():.4f}",
            LR_G=f"{get_lr(optimizer_g):.6f}"
        )


# ===================================================================
# UNCHANGED: eval_model remains the same as it only evaluates the generator
# ===================================================================
@torch.no_grad()
def eval_model(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        step: int,
        writer: SummaryWriter,
) -> float:
    model.eval()
    total_loss = 0.0
    for x in loader:
        noisy_images = x["lowct"].to(device)
        target_images = x["highct"].to(device)

        with autocast(enabled=(device.type == 'cuda')):
            reconstruction = model(x=noisy_images)
            loss = F.l1_loss(reconstruction.float(), target_images.float())

        total_loss += loss.item() * noisy_images.shape[0]

    avg_loss = total_loss / len(loader.dataset)

    if writer is not None:
        writer.add_scalar("val/l1_loss", avg_loss, step)  # Renamed for clarity
        log_reconstructions(
            image=target_images,
            reconstruction=reconstruction,
            writer=writer,
            step=step,
        )

    return avg_loss


# ===================================================================
# MODIFIED: The main training driver now handles GAN components
# ===================================================================
def train_model(
        model: nn.Module,
        d_model: nn.Module,
        start_epoch: int,
        best_loss: float,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer_g: torch.optim.Optimizer,
        optimizer_d: torch.optim.Optimizer,
        lambda_l1: float,
        n_epochs: int,
        eval_freq: int,
        scheduler_g: _LRScheduler,
        scheduler_d: _LRScheduler,
        writer_train: SummaryWriter,
        writer_val: SummaryWriter,
        device: torch.device,
        run_dir: Path,
) -> float:
    # Separate scalers for generator and discriminator
    scaler_g = GradScaler(enabled=(device.type == 'cuda'))
    scaler_d = GradScaler(enabled=(device.type == 'cuda'))

    # Define the GAN loss criterion
    loss_gan_criterion = nn.BCEWithLogitsLoss()

    raw_model = model.module if hasattr(model, "module") else model
    raw_d_model = d_model.module if hasattr(d_model, "module") else d_model

    # Initial evaluation before training starts
    val_loss = eval_model(
        model=model, loader=val_loader, device=device, step=0, writer=writer_val
    )
    print(f"Initial val L1 loss: {val_loss:.6f}")

    for epoch in range(start_epoch, n_epochs):
        train_epoch_gan(
            model=model,
            d_model=d_model,
            loader=train_loader,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            loss_gan_criterion=loss_gan_criterion,
            lambda_l1=lambda_l1,
            device=device,
            epoch=epoch,
            writer=writer_train,
            scaler_g=scaler_g,
            scaler_d=scaler_d,
        )

        if (epoch + 1) % eval_freq == 0:
            val_loss = eval_model(
                model=model,
                loader=val_loader,
                device=device,
                step=len(train_loader) * (epoch + 1),
                writer=writer_val,
            )
            print(f"Epoch {epoch + 1} val L1 loss: {val_loss:.6f}")


            checkpoint = {
                "epoch": epoch + 1,
                "state_dict_g": raw_model.state_dict(),
                "state_dict_d": raw_d_model.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "scheduler_g": scheduler_g.state_dict(),
                "scheduler_d": scheduler_d.state_dict(),
                "best_loss": best_loss,
            }
            torch.save(checkpoint, run_dir / "checkpoint.pth")

            if val_loss < best_loss:
                print(f"✅ New best val loss {val_loss:.6f}. Saving model...")
                best_loss = val_loss
                torch.save(raw_model.state_dict(), run_dir / "best_model.pth")

        # Step schedulers for both optimizers
        if isinstance(scheduler_g, ReduceLROnPlateau):
            scheduler_g.step(val_loss)
        else:
            scheduler_g.step()

        if isinstance(scheduler_d, ReduceLROnPlateau):
            # Note: ReduceLROnPlateau is not ideal for GANs.
            # This is just for structural consistency.
            scheduler_d.step(val_loss)
        else:
            scheduler_d.step()

    print("🏁 Training finished!")
    print("Saving final model...")
    torch.save(raw_model.state_dict(), run_dir / "final_model.pth")

    return best_loss