""" Training script for the autoencoder with KL regulization. """
import argparse
import warnings
from pathlib import Path
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
from training_function_swin import  train_swin
from util import log_mlflow, create_wavelet_dataloaders,create_swin_dataloaders
import torch.nn as nn


warnings.filterwarnings("ignore")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, required=True,
        choices=["swin_ll", "swin_hf", "swin_lh", "swin_hl", "swin_hh","pure_swinunet"],
        help="Specify model to train: low-freq ('swin_ll'), stacked high-freq ('swin_hf'), or individual high-freqs ('swin_lh', 'swin_hl', 'swin_hh')."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed to use.")
    parser.add_argument('--run_dir', type=str, default='default_run', help='Directory for this run')
    parser.add_argument("--config_file", required=True, help="Location of configs file.")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--n_epochs", type=int, default=200, help="Number of epochs to train.")
    parser.add_argument("--eval_freq", type=int, default=10, help="Number of epochs between evaluations.")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of loader workers")
    parser.add_argument("--experiment", default="unet_denoise", help="Mlflow experiment name.")
    args = parser.parse_args()
    return args


def main(args):
    # --- Setup ---
    set_determinism(seed=args.seed)
    print_config()
    output_dir = Path("your output dir")
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
    writer_test = SummaryWriter(log_dir=str(run_dir / "test"))

    config = OmegaConf.load(args.config_file)

    print("\n" + "=" * 40)
    print("                RUN CONFIG                ")
    print("=" * 40)
    for k, v in vars(args).items():
        print(f"  - {k}: {v}")
    print(f"  - run_dir (absolute): {run_dir.resolve()}")
    print("=" * 40 + "\n")

    print(f"\n[CONFIG] Training Model: {args.model_type}")
    if args.model_type == 'swin_ll':
        input_key, target_key = "input_LL", "target_LL"
    elif args.model_type == 'swin_lh':
        input_key, target_key = "input_LH", "target_LH"
    elif args.model_type == 'swin_hl':
        input_key, target_key = "input_HL", "target_HL"
    elif args.model_type == 'swin_hh':
        input_key, target_key = "input_HH", "target_HH"
    elif args.model_type == 'swin_hf':
        # This trains on the 3-channel stacked high-frequency components
        input_key, target_key = "input_HF", "target_HF"
    elif args.model_type == 'pure_swinunet':
        # This trains on the 3-channel stacked high-frequency components
        input_key, target_key = "low_dose", "high_dose"

    print(f"  - Input Tensor Key: '{input_key}'")
    print(f"  - Target Tensor Key: '{target_key}'")

    # --- Data Loading ---
    print("Getting data...")
    # Assumes your configs file has a 'data' section with folder paths
    if args.model_type in ['swin_ll', 'swin_lh', 'swin_hl', 'swin_hh','swin_hf']  :
        train_loader, val_loader = create_wavelet_dataloaders(
        train_tsv_path=config.data.train_tsv,
        val_tsv_path=config.data.val_tsv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_seed=args.seed
    )
    elif args.model_type == 'pure_swinunet':
        train_loader, val_loader = create_swin_dataloaders(
            train_tsv_path=config.data.train_tsv,
            val_tsv_path=config.data.val_tsv,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            random_seed=args.seed
        )


    # --- Model, Optimizer, Scheduler, and Device Setup ---
    print("Setting up model, optimizer, and scheduler...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define model based on configs file
    model = SwinUNETR(**config["model"])

    if torch.cuda.device_count() > 1:
        print(f"✅ Activating DataParallel for {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model.to(device)
    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.training.lr)

    # Define scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10,  min_lr=1e-6)

    # --- Load from Checkpoint or Initialize for New Run ---
    if resume:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # FIX: Adjust state_dict keys to handle single/multi-GPU mismatches
        state_dict = checkpoint['state_dict']

        is_dataparallel_model = isinstance(model, nn.DataParallel)
        is_dataparallel_checkpoint = list(state_dict.keys())[0].startswith('module.')

        # Case 1: Load single-GPU checkpoint into a multi-GPU model
        if is_dataparallel_model and not is_dataparallel_checkpoint:
            print("🔧 Adjusting single-GPU checkpoint for DataParallel model by adding 'module.' prefix.")
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = 'module.' + k  # Add 'module.' prefix
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        # Case 2: Load multi-GPU checkpoint into a single-GPU model
        elif not is_dataparallel_model and is_dataparallel_checkpoint:
            print("🔧 Adjusting DataParallel checkpoint for single-GPU model by removing 'module.' prefix.")
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.removeprefix('module.')  # Remove 'module.' prefix
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        # Case 3: Keys match, load directly
        else:
            model.load_state_dict(state_dict)

        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])

        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        print(f"✅ Successfully loaded model and optimizer states. Starting from epoch {start_epoch}.")


    else:

        start_epoch = 0
        best_loss = float('inf')
        print("🚀 Initializing variables for a new training run.")
    # ==========================================================

    # --- Start Training ---
    print(f"\n🚀 Starting training process for: {args.model_type}")
    train_swin(
        model=model,
        start_epoch=start_epoch,
        best_loss=best_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        n_epochs=args.n_epochs,
        eval_freq=args.eval_freq,
        scheduler=scheduler,
        writer_train=writer_train,
        writer_val=writer_val,
        device=device,
        run_dir=run_dir,
        input_key=input_key,
        target_key=target_key
    )

    # --- Final Cleanup ---
    writer_train.close()
    writer_val.close()
    writer_test.close()
    print("🎉 Training complete and TensorBoard writers closed.")


if __name__ == "__main__":
    args = parse_args()
    main(args)