#!/bin/bash

# Make log directory if it doesn't exist
mkdir -p logs

# Fix Parameters
vocab_size=64 # Vocabulary size
seq_len=64  # Sequence length
batch_size=32 # Batch size
dataset_size=5000 # Dataset size
train_fraction=0.8 # Fraction of data used for training
beta_1=1.0  # Induction head beta_1 parameter
beta_2=1.0  # Induction head beta_2 parameter
beta_out=1.0  # Induction head beta_out parameter

# --------------------------------------------------

# print / run parameters
n_prints=50 # Number of prints during training
n_prints_model=5 # Number of times to save model checkpoints during training.
print_scale='log' # Scale for printing steps: log or linear
num_epochs=200 # Number of training epochs

# Variable Parameters
lr=0.1 # Learning rate
cV=1.1     # Coefficient for WV1  
alpha=0.5 # Coefficient for interpolation 
opt='SAM' # Optimizer choice = 'SGD' or 'adam' or 'SAM'
p_error=0.0 # Probability of introducing noise in the target for the induction task
rho=0.0 # Rho parameter for SAM optimizer
attn='linear' # Type of attention: linear or softmax
loss='CE' # Type of loss function: CE or MSE
experiment_name='opt_lin_CE' # Name of the experiment for saving results


# Loop over various configurations = (rho)
configurations=(
    # '0.0'
    # '0.1'
    # '0.2'
    # '0.3'
    '0.4'
)


for config in "${configurations[@]}"; do
    read -r  rho <<< "$config"
    python -u ./scripts/linear_vs_sfm.py \
        --vocab_size $vocab_size \
        --seq_len $seq_len \
        --batch_size $batch_size \
        --dataset_size $dataset_size \
        --train_fraction $train_fraction \
        --beta_1 $beta_1 \
        --beta_2 $beta_2 \
        --beta_out $beta_out \
        --n_prints $n_prints \
        --n_prints_model $n_prints_model \
        --print_scale $print_scale \
        --num_epochs $num_epochs \
        --lr $lr \
        --cV $cV\
        --alpha $alpha \
        --opt $opt \
        --p_error $p_error\
        --rho $rho \
        --attn $attn \
        --loss $loss \
        --experiment_name $experiment_name
        
    echo "Completed: at $(date)"
    echo "---"
    echo ""
done

echo "All training runs completed!"