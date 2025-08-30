"""Utility functions for training."""
from pathlib import Path
from typing import Tuple, Union
import pywt
from torch.utils.data import Dataset
import os
from PIL import Image
import matplotlib.pyplot as plt
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from custom_transforms import ApplyTokenizerd
from mlflow import start_run
from monai import transforms
from monai.data import DataLoader, Dataset, CacheDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    RandRotate90d,
    RandFlipd,
    Resized,
    ToTensord,
    MapTransform,

    ScaleIntensityd,



)
from monai.data import PersistentDataset
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm


# ----------------------------------------------------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------------------------------------------------
def get_datalist(
    ids_path: str,
    extended_report: bool = False,
):
    """Get data dicts for data loaders."""
    df = pd.read_csv(ids_path, sep="\t")

    data_dicts = []
    for index, row in df.iterrows():
        data_dicts.append(
            {
                "t1w": f"{row['t1w']}",
                "flair": f"{row['flair']}",
                "report": "T1-weighted image of a brain.",
            }
        )

    print(f"Found {len(data_dicts)} subjects.")
    return data_dicts


# In your util.py or a similar file

import pandas as pd
from pathlib import Path
import torch
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    RandRotate90d,
    RandFlipd,
    RandAffined,
    RandShiftIntensityd,
    RandAdjustContrastd,
    ThresholdIntensityd,
    ToTensord,
)



def get_dataloader(
        batch_size: int,
        training_ids: str,
        validation_ids: str,
        num_workers: int,
        model_type: str,
) -> (DataLoader, DataLoader):
    """
    Creates training and validation data loaders based on the training stage.

    This function reads .tsv files created by `create_datalist.py`.

    Args:
        batch_size: The batch size for the data loaders.
        training_ids: Path to the train.tsv file.
        validation_ids: Path to the validation.tsv file.
        num_workers: Number of worker processes for loading data.
        model_type: The stage of training. Must be one of ["diffusion_nos1", "controlnet_nos1", "autoencoder"].

    Returns:
        A tuple containing the training DataLoader and validation DataLoader.
    """
    print(f"Creating dataloader for model_type: '{model_type}'")

    if model_type == "diffusion":
        # Stage A (Base DDPM): We only need the target (high-dose) images.
        keys_to_load = ["high_dose"]
        train_transforms = Compose(
            [
                LoadImaged(keys=keys_to_load),
                EnsureChannelFirstd(keys=keys_to_load),

                # --- MERGED TRANSFORMS ---
                # 1. Scale intensity to the [-1, 1] range, which is optimal for DDPMs.
                ScaleIntensityRanged(keys=keys_to_load, a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0, clip=True),

                # 2. Add rich spatial and intensity augmentations.
                # All instances of "t1w" are replaced with `keys_to_load`.
                RandFlipd(keys=keys_to_load, prob=0.5, spatial_axis=0),
                RandAffined(
                    keys=keys_to_load,
                    translate_range=(-5, 5),
                    scale_range=(-0.1, 0.1),
                    spatial_size=[256, 256],
                    prob=0.5,  
                ),
                RandShiftIntensityd(keys=keys_to_load, offsets=0.05, prob=0.1),
                RandAdjustContrastd(keys=keys_to_load, gamma=(0.97, 1.03), prob=0.1),

                # 3. Final conversion to tensor
                ToTensord(keys=keys_to_load),
            ]
        )

        # For validation, we ONLY perform the necessary steps: loading, scaling, resizing. NO random augmentations.
        val_transforms = Compose(
            [
                LoadImaged(keys=keys_to_load),
                EnsureChannelFirstd(keys=keys_to_load),
                ScaleIntensityRanged(keys=keys_to_load, a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0, clip=True),

                Resized(keys=keys_to_load, spatial_size=[256, 256]),
                ToTensord(keys=keys_to_load),
            ]
        )
    elif model_type == "controlnet":
        # Stage B (ControlNet): We need the input (low-dose) and target (high-dose).
        keys_to_load = ["low_dose", "high_dose"]
        # Use the same transforms as diffusion, but applied to both keys
        train_transforms = Compose(
            [
                LoadImaged(keys=keys_to_load),
                EnsureChannelFirstd(keys=keys_to_load),
                ScaleIntensityRanged(keys=keys_to_load, a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0, clip=True),
                RandRotate90d(keys=keys_to_load, prob=0.5, spatial_axes=(0, 1)),
                RandFlipd(keys=keys_to_load, prob=0.5, spatial_axis=0),
                RandAffined(
                    keys=keys_to_load,
                    translate_range=(-5, 5),
                    scale_range=(-0.1, 0.1),
                    spatial_size=[256, 256],  # Ensure a consistent size
                    prob=0.5,
                ),
                RandShiftIntensityd(keys=keys_to_load, offsets=0.1, prob=0.5),
                RandAdjustContrastd(keys=keys_to_load, gamma=(0.9, 1.1), prob=0.5),

                # 3. Final conversion to tensor
                ToTensord(keys=keys_to_load),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=keys_to_load),
                EnsureChannelFirstd(keys=keys_to_load),
                ScaleIntensityRanged(keys=keys_to_load, a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0, clip=True),
                Resized(keys=keys_to_load, spatial_size=[256, 256]),
                ToTensord(keys=keys_to_load),
            ]
        )

    # NEW: Add the logic for the "autoencoder" model_type
    elif model_type == "autoencoder":
        # Autoencoder learns to reconstruct high-quality images.
        # So we only need to load the 'high_dose' images.
        keys_to_load = ["high_dose"]

        # We replace every instance of "t1w" from your example with "high_dose"
        train_transforms = Compose(
            [
                LoadImaged(keys=keys_to_load),
                EnsureChannelFirstd(keys=keys_to_load),
                # Note: The original transforms scaled to [0,1]. Sticking to that for consistency.
                ScaleIntensityRanged(keys=keys_to_load, a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
                RandFlipd(keys=keys_to_load, prob=0.5, spatial_axis=0),
                RandAffined(
                    keys=keys_to_load,
                    translate_range=(-2, 2),
                    scale_range=(-0.05, 0.05),
                    spatial_size=[160, 224],  # Note: This resizes the image
                    prob=0.5,
                ),
                RandShiftIntensityd(keys=keys_to_load, offsets=0.05, prob=0.1),
                RandAdjustContrastd(keys=keys_to_load, gamma=(0.97, 1.03), prob=0.1),
                ThresholdIntensityd(keys=keys_to_load, threshold=1, above=False, cval=1.0),
                ThresholdIntensityd(keys=keys_to_load, threshold=0, above=True, cval=0),
                ToTensord(keys=keys_to_load),
            ]
        )

        # For validation, we use a simpler set of transforms without random augmentations
        val_transforms = Compose(
            [
                LoadImaged(keys=keys_to_load),
                EnsureChannelFirstd(keys=keys_to_load),
                ScaleIntensityRanged(keys=keys_to_load, a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
                # Note: We must still resize to match the training input size
                # This could also be done with Resized transform

                ToTensord(keys=keys_to_load),
            ]
        )

    else:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. Must be one of ['diffusion_nos1', 'controlnet_nos1', 'autoencoder']")

    # --- The rest of the function is the same for all model types ---

    # 1. Read the .tsv files using pandas
    print(f"Reading training data list from: {training_ids}")
    train_df = pd.read_csv(training_ids, sep='\t')
    print(f"Reading validation data list from: {validation_ids}")
    val_df = pd.read_csv(validation_ids, sep='\t')

    # Convert pandas DataFrames to a list of dictionaries, which MONAI expects
    train_files = train_df.to_dict('records')
    val_files = val_df.to_dict('records')

    # Create MONAI Datasets
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    # Create PyTorch DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader

# ----------------------------------------------------------------------------------------------------------------------
# LOGS
# ----------------------------------------------------------------------------------------------------------------------
def recursive_items(dictionary, prefix=""):
    for key, value in dictionary.items():
        if type(value) in [dict, DictConfig]:
            yield from recursive_items(value, prefix=str(key) if prefix == "" else f"{prefix}.{str(key)}")
        else:
            yield (str(key) if prefix == "" else f"{prefix}.{str(key)}", value)


def log_mlflow(
    model,
    config,
    args,
    experiment: str,
    run_dir: Path,
    val_loss: float,
):
    """Log model and performance on Mlflow system"""
    config = {**OmegaConf.to_container(config), **vars(args)}
    print(f"Setting mlflow experiment: {experiment}")
    mlflow.set_experiment(experiment)

    with start_run():
        print(f"MLFLOW URI: {mlflow.tracking.get_tracking_uri()}")
        print(f"MLFLOW ARTIFACT URI: {mlflow.get_artifact_uri()}")

        for key, value in recursive_items(config):
            mlflow.log_param(key, str(value))

        mlflow.log_artifacts(str(run_dir / "train"), artifact_path="events_train")
        mlflow.log_artifacts(str(run_dir / "val"), artifact_path="events_val")
        mlflow.log_metric(f"loss", val_loss, 0)

        raw_model = model.module if hasattr(model, "module") else model
        mlflow.pytorch.log_model(raw_model, "final_model")


def get_figure(
    img: torch.Tensor,
    recons: torch.Tensor,
):
    img_npy_0 = np.clip(a=img[0, 0, :, :].cpu().numpy(), a_min=0, a_max=1)
    recons_npy_0 = np.clip(a=recons[0, 0, :, :].cpu().numpy(), a_min=0, a_max=1)
    img_npy_1 = np.clip(a=img[1, 0, :, :].cpu().numpy(), a_min=0, a_max=1)
    recons_npy_1 = np.clip(a=recons[1, 0, :, :].cpu().numpy(), a_min=0, a_max=1)

    img_row_0 = np.concatenate(
        (
            img_npy_0,
            recons_npy_0,
            img_npy_1,
            recons_npy_1,
        ),
        axis=1,
    )

    fig = plt.figure(dpi=300)
    plt.imshow(img_row_0, cmap="gray")
    plt.axis("off")
    return fig


def log_reconstructions(
    image: torch.Tensor,
    reconstruction: torch.Tensor,
    writer: SummaryWriter,
    step: int,
    title: str = "RECONSTRUCTION",
) -> None:
    fig = get_figure(
        image,
        reconstruction,
    )
    writer.add_figure(title, fig, step)


@torch.no_grad()
def log_ldm_sample_unconditioned(
    model: nn.Module,
    stage1: nn.Module,
    scheduler: nn.Module,
    spatial_shape: Tuple,
    writer: SummaryWriter,
    step: int,
    device: torch.device,
    scale_factor: float = 1.0,
) -> None:
    latent = torch.randn((1,) + spatial_shape)
    latent = latent.to(device)

    # prompt_embeds = torch.cat((49406 * torch.ones(1, 1), 49407 * torch.ones(1, 76)), 1).long()
    # prompt_embeds = text_encoder(prompt_embeds.squeeze(1))
    # prompt_embeds = prompt_embeds[0]

    for t in tqdm(scheduler.timesteps, ncols=70):
        noise_pred = model(x=latent, timesteps=torch.asarray((t,)).to(device))
        latent, _ = scheduler.step(noise_pred, t, latent)

    x_hat = stage1.model.decode(latent / scale_factor)
    img_0 = np.clip(a=x_hat[0, 0, :, :].cpu().numpy(), a_min=0, a_max=1)
    fig = plt.figure(dpi=300)
    plt.imshow(img_0, cmap="gray")
    plt.axis("off")
    writer.add_figure("SAMPLE", fig, step)


@torch.no_grad()
def log_ddpm_sample(
    model: nn.Module,
    scheduler: nn.Module,
    spatial_shape: tuple,
    writer: SummaryWriter,
    step: int,
    device: torch.device,
        scale_factor: float = 1.0,
) -> None:
    """
    Samples an image from a pure, unconditional, pixel-space DDPM
    and logs it to TensorBoard.
    """
    model.eval() # Set model to evaluation mode

    # 1. Start with pure random noise in the shape of the desired image
    # spatial_shape should be like (channels, height, width), e.g., (1, 256, 256)
    image = torch.randn((1,) + spatial_shape).to(device)

    # 2. Set the number of inference steps in the scheduler
    # More steps are higher quality but slower. 1000 is common for DDPM.
    scheduler.set_timesteps(1000)

    # 3. Iteratively denoise the image
    for t in tqdm(scheduler.timesteps, desc="Sampling image", ncols=80):
        # Predict the noise (or v-prediction) from the current noisy image
        # No context is passed to the model
        noise_pred = model(x=image, timesteps=torch.asarray((t,), device=device))

        # Use the scheduler to compute the previous image state (denoise one step)
        image = scheduler.step(noise_pred, t, image).prev_sample

    # 4. Post-process the final image
    # The model was trained on data in [-1, 1], so we reverse that for visualization
    image = (image.clamp(-1, 1) + 1) / 2.0
    image = (image * 255.0).to(torch.uint8)  # Convert to 0-255 range

    # 5. Log the image to TensorBoard
    img_np = image[0, 0, :, :].cpu().numpy()
    fig = plt.figure(dpi=200)
    plt.imshow(img_np, cmap="gray")
    plt.axis("off")
    writer.add_figure("Generated Sample", fig, global_step=step)
    plt.close(fig)  # Prevent memory leaks





class WaveletDecompositiond(MapTransform):
    """
    A MONAI transform to apply 2D Discrete Wavelet Transform (DWT)
    to a pair of images and separate the coefficients into individual keys.
    This works on tensors.
    """
    # ============================ THE FIX ============================
    def __init__(self, key_map: dict, wavelet='haar'):
        """
        Args:
            key_map (dict): A dictionary mapping input keys to output prefixes.
                            Example: {"low_dose": "input", "high_dose": "target"}
            wavelet (str): The name of the wavelet to use (e.g., 'haar').
        """
        # 1. Pass ONLY the input keys to the parent class.
        #    The parent needs to know which keys to look for in the data dict.
        super().__init__(keys=list(key_map.keys()))

        # 2. Store your custom mapping in a separate attribute to avoid conflicts.
        self.key_map = key_map
        self.wavelet = wavelet

    def __call__(self, data):
        d = dict(data)
        # 3. Iterate over your custom mapping dictionary.
        for key_in, key_out_prefix in self.key_map.items():
            # Make sure the key exists before processing
            if key_in not in d:
                continue

            # Ensure tensor is on CPU and in numpy format for pywt
            image_np = d[key_in].squeeze(0).cpu().numpy()

            # Perform 2D DWT
            coeffs = pywt.dwt2(image_np, self.wavelet)
            LL, (LH, HL, HH) = coeffs

            # Create a 3-channel stack for the combined 'HF' target
            HF = np.stack([LH, HL, HH], axis=0)

            # Add each coefficient back to the dictionary as a tensor with a channel dim
            d[f"{key_out_prefix}_LL"] = torch.from_numpy(LL).unsqueeze(0)
            d[f"{key_out_prefix}_LH"] = torch.from_numpy(LH).unsqueeze(0)
            d[f"{key_out_prefix}_HL"] = torch.from_numpy(HL).unsqueeze(0)
            d[f"{key_out_prefix}_HH"] = torch.from_numpy(HH).unsqueeze(0)
            d[f"{key_out_prefix}_HF"] = torch.from_numpy(HF) # Already has channel dim from stack
        return d
    # =================================================================

def create_wavelet_dataloaders(
    train_tsv_path: str,
    val_tsv_path: str,
    batch_size: int,
    num_workers: int,
    random_seed: int,
):
    """
    Creates training and validation dataloaders for the wavelet-based models.
    """
    load_keys = ["low_dose", "high_dose"]
    augmentation_keys = [
        "input_LL", "target_LL", "input_LH", "target_LH",
        "input_HL", "target_HL", "input_HH", "target_HH",
        "input_HF", "target_HF"
    ]

    train_transforms = Compose([
        LoadImaged(keys=load_keys, image_only=False),
        EnsureChannelFirstd(keys=load_keys),
        ScaleIntensityRanged(keys=load_keys, a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),


        WaveletDecompositiond(key_map={"low_dose": "input", "high_dose": "target"}),

        RandFlipd(keys=augmentation_keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=augmentation_keys, prob=0.5, spatial_axis=1),
        RandRotate90d(keys=augmentation_keys, prob=0.5, max_k=3),
    ])

    val_transforms = Compose([
        LoadImaged(keys=load_keys, image_only=False),
        EnsureChannelFirstd(keys=load_keys),
        ScaleIntensityRanged(keys=load_keys, a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        # ==================== FIX THE CALL HERE ====================
        WaveletDecompositiond(key_map={"low_dose": "input", "high_dose": "target"}),
        # =========================================================
    ])

    # --- Create DataLoaders (this part remains the same) ---
    train_df = pd.read_csv(train_tsv_path, sep='\t')
    train_files = train_df.to_dict('records')
    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),generator=torch.Generator().manual_seed(random_seed)
    )

    val_df = pd.read_csv(val_tsv_path, sep='\t')
    val_files = val_df.to_dict('records')
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader





def create_swin_dataloaders(
    train_tsv_path: str,
    val_tsv_path: str,
    batch_size: int,
    num_workers: int,
    random_seed: int,
):
    """
    Creates training and validation dataloaders for the wavelet-based models.
    """
    load_keys = ["low_dose", "high_dose"]


    train_transforms = Compose([
        LoadImaged(keys=load_keys, image_only=False),
        EnsureChannelFirstd(keys=load_keys),
        ScaleIntensityRanged(keys=load_keys, a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),



        RandFlipd(keys=load_keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=load_keys, prob=0.5, spatial_axis=1),
        RandRotate90d(keys=load_keys, prob=0.5, max_k=3),
        ToTensord(keys=load_keys),
    ])

    val_transforms = Compose([
        LoadImaged(keys=load_keys, image_only=False),
        EnsureChannelFirstd(keys=load_keys),
        ScaleIntensityRanged(keys=load_keys, a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        # ==================== FIX THE CALL HERE ====================
        ToTensord(keys=load_keys),
        # =========================================================
    ])

    # --- Create DataLoaders (this part remains the same) ---
    train_df = pd.read_csv(train_tsv_path, sep='\t')
    train_files = train_df.to_dict('records')
    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),generator=torch.Generator().manual_seed(random_seed)
    )

    val_df = pd.read_csv(val_tsv_path, sep='\t')
    val_files = val_df.to_dict('records')
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader