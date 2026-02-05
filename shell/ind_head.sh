#!/bin/bash

# Meta parameters
n_prints=50 # Number of prints during training
num_epochs=10 # Number of training epochs

# Parameters
vocab_size=40 # Vocabulary size
seq_len=16  # Sequence length
batch_size=16 # Batch size
dataset_size=10000 # Dataset size
train_fraction=0.8 # Fraction of data used for training
dropout=0.0 # Dropout rate
noise=0.0  # Noise level for parameter initialization

# Loop over various configurations = (lr)
configurations=(
    '0.00001'
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
        
    echo "Completed: at $(date)"
    echo "---"
    echo ""

done

echo "All training runs completed!"