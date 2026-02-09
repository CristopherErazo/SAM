#!/bin/bash

# Meta parameters
n_prints=50 # Number of prints during training
num_epochs=25 # Number of training epochs

# Parameters
vocab_size=256 # Vocabulary size
seq_len=128  # Sequence length
batch_size=64 # Batch size
dataset_size=30000 # Dataset size
train_fraction=0.8 # Fraction of data used for training
dropout=0.00 # Dropout rate
lr=0.005 # Learning rate

alpha=0.3
gamma=0.0 # Weight decay for sgd

alpha_load=0.2 # Alpha parameter for loading model checkpoint
lr_load=0.0001 # Learning rate for loading model checkpoint

# Loop over various configurations = (opt rho)
configurations=(
    'SGD 0.0'
    'SAM 0.1'
    'SAM 0.2'
    'SAM 0.3'
    'SAM 0.4'
    # 'adam 0.0'
)


for config in "${configurations[@]}"; do
    read -r  opt rho <<< "$config"
    python -u ./scripts/induction_optimizers.py \
        --vocab_size $vocab_size \
        --seq_len $seq_len \
        --batch_size $batch_size \
        --dataset_size $dataset_size \
        --train_fraction $train_fraction \
        --lr $lr \
        --num_epochs $num_epochs \
        --n_prints $n_prints \
        --dropout $dropout \
        --alpha $alpha \
        --gamma $gamma \
        --rho $rho \
        --opt $opt \
        --alpha_load $alpha_load \
        --lr_load $lr_load 
        
    echo "Completed: at $(date)"
    echo "---"
    echo ""
done
echo "All training runs completed!"