#!/bin/bash

# Meta parameters
n_prints=50 # Number of prints during training
num_epochs=100 # Number of training epochs

# Parameters
d_model=32 # Model dimension
d_eff=16    # Effective dimension
vocab_size=20 # Vocabulary size
seq_len=16  # Sequence length
batch_size=64 # Batch size
dataset_size=10000 # Dataset size
train_fraction=0.8 # Fraction of data used for training
n_heads=1  # Number of attention heads
n_layers=2 # Number of transformer layers
sigma=0.2  # Std of parameter initialization
fr_emb=True # Freeze embeddings during training
dropout=0.1 # Dropout rate
sparsity=0.0 # Sparsity level for initialization

# Loop over various configurations = (sigma)
configurations=(
    '0.00005'
)


for config in "${configurations[@]}"; do
    read -r  lr <<< "$config"
    python -u ./scripts/copy_task.py \
        --d_model $d_model \
        --d_eff $d_eff \
        --vocab_size $vocab_size \
        --seq_len $seq_len \
        --batch_size $batch_size \
        --dataset_size $dataset_size \
        --train_fraction $train_fraction \
        --sigma $sigma \
        --lr $lr \
        --n_heads $n_heads \
        --n_layers $n_layers \
        --num_epochs $num_epochs \
        --n_prints $n_prints \
        --fr_emb $fr_emb \
        --dropout $dropout \
        --sparsity $sparsity
        
    echo "Completed: at $(date)"
    echo "---"
    echo ""

done

echo "All training runs completed!"