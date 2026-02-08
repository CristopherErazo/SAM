#!/bin/bash

# Meta parameters
n_prints=50 # Number of prints during training
num_epochs=5 # Number of training epochs

# Parameters
vocab_size=128 # Vocabulary size
seq_len=32  # Sequence length
batch_size=16 # Batch size
dataset_size=1000 # Dataset size
train_fraction=0.8 # Fraction of data used for training
dropout=0.00 # Dropout rate
alpha=0.3  # Noise level for parameter initialization (interpolation parameter)
beta_1=5.0  # Induction head beta_1 parameter
beta_2=1.0  # Induction head beta_2 parameter
beta_out=1.0  # Induction head beta_out parameter

# Loop over various configurations = (lr)
configurations=(
    '0.0001'
)


for config in "${configurations[@]}"; do
    read -r  lr <<< "$config"
    python -u ./scripts/induction_head.py \
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
        --beta_1 $beta_1 \
        --beta_2 $beta_2 \
        --beta_out $beta_out
        
    echo "Completed: at $(date)"
    echo "---"
    echo ""

done

echo "All training runs completed!"