#!/bin/bash

# Meta parameters
n_prints=50 # Number of prints during training
num_epochs=200 # Number of training epochs

# Parameters
vocab_size=64 # Vocabulary size
seq_len=64  # Sequence length
batch_size=64 # Batch size
dataset_size=30000 # Dataset size
train_fraction=0.8 # Fraction of data used for training
dropout=0.00 # Dropout rate
beta_1=1.0  # Induction head beta_1 parameter
beta_2=1.0  # Induction head beta_2 parameter
beta_out=1.0  # Induction head beta_out parameter
lr=0.02 # Learning rate
sigma=0.3 # Sigma for interpolation initialization
cV=1.1 # Coefficient for WV1

# Loop over various configurations = (alpha)
configurations=(
    '0.0'
    '0.2'
    '0.4'
    '0.6'
    '0.8'
    '1.0'
)


for config in "${configurations[@]}"; do
    read -r  alpha <<< "$config"
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
        --beta_out $beta_out \
        --sigma $sigma\
        --cV $cV

        
    echo "Completed: at $(date)"
    echo "---"
    echo ""
done



echo "All training runs completed!"