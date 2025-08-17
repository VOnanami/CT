""" Main training script for the SwinUNETR model. """
import argparse
import warnings
from pathlib import Path
import os
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
sys.path.append('/data/coding/SwinIR/models')
from network_swinir import SwinIR
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
from train import train_swin  # Assuming your updated train_swin is in train.py
# MODIFIED: Import the new dataloader function
from utils import create_dataloaders_from_split_pkl
import torch.nn as nn

# Ignore unnecessary warnings
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed to use.")
    parser.add_argument('--run_dir', type=str, default='default_run', help='Directory for this run')
    parser.add_argument("--config_file", required=True, help="Location of config file.")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--n_epochs", type=int, default=200, help="Number of epochs to train.")
    parser.add_argument("--eval_freq", type=int, default=10, help="Number of epochs between evaluations.")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of loader workers")
    parser.add_argument("--experiment", default="swinir_sr", help="Experiment name (e.g., for MLflow).")
    args = parser.parse_args()
    return args

def main(args):
    # --- Setup ---
    set_determinism(seed=args.seed)
    print_config()
    output_dir = Path("./training_runs") # Main output directory
    output_dir.mkdir(exist_ok=True, parents=True)
    run_dir = output_dir / args.run_dir
    
    # --- Checkpoint Resumption ---
    checkpoint_path = run_dir / "checkpoint.pth"
    if checkpoint_path.exists():
        resume = True
        print(f"✅ Checkpoint found. Resuming from: {checkpoint_path}")
    else:
        resume = False
        run_dir.mkdir(exist_ok=True)
        print("📂 No checkpoint found. Starting a new run.")
        
    writer_train = SummaryWriter(log_dir=str(run_dir / "train"))
    writer_val = SummaryWriter(log_dir=str(run_dir / "val"))
    
    config = OmegaConf.load(args.config_file)
    
    # --- Print Configuration ---
    print("\n" + "="*40)
    print("               RUN CONFIG               ")
    print("="*40)
    for k, v in vars(args).items():
        print(f"  - {k}: {v}")
    print(f"  - run_dir (absolute): {run_dir.resolve()}")
    print("="*40 + "\n")
    DATA_PATHS = config['data_paths']
    # --- Data Loading ---
    print("Setting up model, optimizer, and scheduler...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = SwinIR(**config["model"])
    print("Getting data using CSV-based split...")
    optimizer = optim.AdamW(model.parameters(), lr=config.training.lr)
    scheduler_config = config.training.lr_scheduler
    if scheduler_config.name == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, **scheduler_config.params)
    else:
        # You can add other schedulers here if needed
        raise NotImplementedError(f"Scheduler {scheduler_config.name} not implemented.")
    # MODIFIED: Call the new dataloader function and get all three loaders
    train_loader, val_loader, test_loader = create_dataloaders_from_split_pkl(
        data_paths=DATA_PATHS,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_seed=args.seed
    )
    # --- Model, Optimizer, Scheduler, and Device Setup ---
    
   
    
    if torch.cuda.device_count() > 1:
        print(f"✅ Activating DataParallel for {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

   
    
    # NEW: Dynamically create scheduler from config
    

    # --- Load from Checkpoint or Initialize for New Run ---
    if resume:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Using a raw_model reference to handle DataParallel wrappers easily
        raw_model = model.module if hasattr(model, "module") else model
        raw_model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        
        start_epoch = checkpoint['epoch']
        # MODIFIED: Use the correct variable name for the metric
        best_mae_loss = checkpoint['best_loss']
        print(f"✅ Successfully loaded checkpoint. Starting from epoch {start_epoch}.")
    else:
        start_epoch = 0
        # MODIFIED: Initialize best_mae_loss to infinity for a new run
        best_mae_loss = np.inf
        print("🚀 Initializing variables for a new training run.")

    # --- Start Training ---
    print("\n🚀 Starting training process...")
    train_swin(
        model=model,
        start_epoch=start_epoch,
        # MODIFIED: Pass the correct variable
        best_mae_loss=best_mae_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        # MODIFIED: Pass the new test_loader
        test_loader=test_loader,
        optimizer=optimizer,
        n_epochs=args.n_epochs,
        eval_freq=args.eval_freq,
        scheduler=scheduler,
        writer_train=writer_train,
        writer_val=writer_val,
        device=device,
        run_dir=run_dir,
    )
    
    # --- Final Cleanup ---
    writer_train.close()
    writer_val.close()
    print("🎉 Training complete and TensorBoard writers closed.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
