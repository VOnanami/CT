# Seed for reproducibility
seed=42

run_dir="swinunet_generator/4_generators/swinunetr_ll_pretrain_LL"

config_file="configs/swinunet.yaml"

# Batch size.
batch_size=64

# Total number of epochs to train for.
n_epochs=100

eval_freq=10

# Number of parallel workers for the DataLoader.
num_workers=16

# Path to your python script
PYTHON_SCRIPT_PATH="/training/swin_train.py"

# --- Execution ---

export TRANSFORMERS_CACHE="/work3/s242644/swinunet"
mkdir -p $TRANSFORMERS_CACHE # Create the directory if it doesn't exist

      python3 ${PYTHON_SCRIPT_PATH} \
      --model_type "swin_ll" \   # you can also set to swin_hh, swin_lh, or swin_hf etc.
      --seed ${seed} \
      --run_dir ${run_dir} \
      --config_file ${config_file} \
      --batch_size ${batch_size} \
      --n_epochs ${n_epochs} \
      --eval_freq ${eval_freq} \
      --num_workers ${num_workers}