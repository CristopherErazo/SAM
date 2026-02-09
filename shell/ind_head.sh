#!/bin/bash

# Meta parameters
n_prints=80 # Number of prints during training
num_epochs=50 # Number of training epochs

# Parameters
vocab_size=256 # Vocabulary size
# seq_len=64  # Sequence length
batch_size=64 # Batch size
dataset_size=30000 # Dataset size
train_fraction=0.8 # Fraction of data used for training
dropout=0.00 # Dropout rate
# alpha=0.5  # (interpolation parameter) - 0: planted, 1: random
beta_1=5.0  # Induction head beta_1 parameter
beta_2=2.0  # Induction head beta_2 parameter
beta_out=3.0  # Induction head beta_out parameter
lr=0.0001 # Learning rate

# Loop over various configurations = (alpha)
configurations=(
    '0.45'
    # '0.1'
    # '0.2'
    # '0.3'
    # '0.4'
    # '0.5'
    # '0.6'
    # '0.7'
    # '0.8'
    # '0.9'
    # '1.0'
)

for seq_len in 128
do 
echo "Running with seq_len = $seq_len"
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

            
        echo "Completed: at $(date)"
        echo "---"
        echo ""
    done
done


echo "All training runs completed!"