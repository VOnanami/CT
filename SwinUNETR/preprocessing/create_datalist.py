import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse


def create_paired_list(base_dir: Path, thicknesses: list, kernels: list) -> pd.DataFrame:
    """
    Scans the dataset to find all corresponding low-dose and high-dose image pairs
    by matching them based on their sorted order.
    """
    all_pairs = []
    print("Scanning dataset to find all valid image pairs...")

    for thickness in thicknesses:
        for kernel in kernels:
            print(f"--> Processing: {thickness} / {kernel}")

            quarter_dose_base = base_dir / "Quarter Dose" / thickness / kernel
            full_dose_base = base_dir / "Full Dose" / thickness / kernel

            if not quarter_dose_base.exists() or not full_dose_base.exists():
                print(f"  - Warning: Path not found for this combination, skipping.")
                continue

            patient_folders = [p for p in quarter_dose_base.iterdir() if p.is_dir()]

            for patient_dir in tqdm(patient_folders, desc=f"  - Patients ({kernel})"):
                patient_id = patient_dir.name

                # 1. Get a sorted list of all quarter-dose images for this patient
                quarter_dose_images = sorted(patient_dir.glob("*.png"))

                # 2. Construct the path to the corresponding full-dose patient directory
                full_dose_patient_dir = full_dose_base / patient_id

                if not full_dose_patient_dir.exists():
                    continue  # Skip if the corresponding full-dose folder doesn't exist

                # 3. Get a sorted list of all full-dose images
                full_dose_images = sorted(full_dose_patient_dir.glob("*.png"))

                # 4. Check if the number of slices matches. If not, something is wrong.
                if len(quarter_dose_images) != len(full_dose_images):
                    print(f"\n  - Warning: Mismatch in file count for patient {patient_id}. Skipping.")
                    continue

                # 5. Pair the images by their index in the sorted lists
                for i in range(len(quarter_dose_images)):
                    all_pairs.append({
                        "patient_id": patient_id,
                        "low_dose": str(quarter_dose_images[i]),
                        "high_dose": str(full_dose_images[i])
                    })

    return pd.DataFrame(all_pairs)


def main(args):
    """Main function to generate and save datalists."""
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all pairs in the dataset based on the specified criteria
    all_data_df = create_paired_list(base_dir, args.thicknesses, args.kernels)

    if all_data_df.empty:
        print("No image pairs were found! Please check your base_dir and dataset structure.")
        return

    print(f"\nFound a total of {len(all_data_df)} image pairs.")

    # --- Splitting Data by Patient ID ---
    print("Splitting data by patient ID to prevent data leakage...")

    # Get a unique list of all patient IDs
    patient_ids = all_data_df['patient_id'].unique()

    # Split patient IDs into train (80%) and a temporary set (20%)
    train_ids, temp_ids = train_test_split(
        patient_ids, test_size=0.2, random_state=42
    )

    # Split the temporary set into validation (50% of temp -> 10% of total) and test (10% of total)
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.5, random_state=42
    )

    print(f"Total Patients: {len(patient_ids)}")
    print(f"Training Patients: {len(train_ids)}")
    print(f"Validation Patients: {len(val_ids)}")
    print(f"Test Patients: {len(test_ids)}")

    # Create the final DataFrames by filtering the master list
    train_df = all_data_df[all_data_df['patient_id'].isin(train_ids)]
    val_df = all_data_df[all_data_df['patient_id'].isin(val_ids)]
    test_df = all_data_df[all_data_df['patient_id'].isin(test_ids)]


    train_df = train_df.drop(columns=['patient_id'])
    val_df = val_df.drop(columns=['patient_id'])
    test_df = test_df.drop(columns=['patient_id'])

    print(f"\nTotal training samples: {len(train_df)}")
    print(f"Total validation samples: {len(val_df)}")
    print(f"Total test samples: {len(test_df)}")

    # Save the datalists as tab-separated files (.tsv)
    train_df.to_csv(output_dir / "train.tsv", index=False, sep="\t")
    val_df.to_csv(output_dir / "validation.tsv", index=False, sep="\t")
    test_df.to_csv(output_dir / "test.tsv", index=False, sep="\t")

    print(f"\nSuccessfully created datalists in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create paired datalists for the Mayo CT Challenge dataset.")
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        default="/datasets/andrewmvd/ct-low-dose-reconstruction/versions/2/Preprocessed_512x512/512",
        help="The base directory of the preprocessed data (e.g., '.../Preprocessed_256x256/256')."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/datasets/andrewmvd/ct-low-dose-reconstruction/versions/2/Preprocessed_512x512/512/output",
        help="The directory where the output .tsv files will be saved."
    )
    parser.add_argument(
        "--thicknesses",
        nargs='+',
        default=["1mm", "3mm"],
        help="List of thicknesses to include (e.g., '1mm' '3mm')."
    )
    parser.add_argument(
        "--kernels",
        nargs='+',
        default=["Soft Kernel (B30)", "Sharp Kernel (D45)"],
        help="List of kernel types to include."
    )

    args = parser.parse_args()
    main(args)