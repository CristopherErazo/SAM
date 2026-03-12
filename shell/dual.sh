#!/bin/bash

# Make log directory if it doesn't exist
mkdir -p logs
jobs=1 # Number of parallel jobs to run

# Fix Parameters
d_model=1000 # Model dimension
vocab_size=64 # Vocabulary size
seq_len=64  # Sequence length
batch_size=10 # Batch size
dataset_size=100 # Dataset size
train_fraction=0.8 # Fraction of data used for training
save_data='False' # Whether to save dataset
mode='uniform' # Data distribution mode: uniform or random
K=1 # Number of trigger tokens

# --------------------------------------------------

# print / run parameters
n_prints=100 # Number of prints during training
n_prints_model=5 # Number of times to save model checkpoints during training.
print_scale='log' # Scale for printing steps: log or linear
num_epochs=1500 # Number of training epochs

# Variable Parameters
lr=0.05 # Learning rate
experiment_name='tmp' # Name of the experiment for saving results


# Loop over various configurations 
configurations=(
    '20 15' 
)


run_config() {
    local vocab_size="$1"
    local seq_len="$2"
    local log_file
    # log_file="logs/${experiment_name}_${vocab_size}_${seq_len}.log"
    log_file="logs/${experiment_name}.log"

    echo "Logging to ${log_file}"

    {
        python -u ./scripts/dual_task.py \
            --vocab_size $vocab_size \
            --seq_len $seq_len \
            --batch_size $batch_size \
            --dataset_size $dataset_size \
            --train_fraction $train_fraction \
            --n_prints $n_prints \
            --n_prints_model $n_prints_model \
            --print_scale $print_scale \
            --num_epochs $num_epochs \
            --lr $lr \
            --experiment_name $experiment_name \
            --save_data $save_data\
            --K $K \
            --mode $mode \
            --d_model $d_model


        echo "Completed: at $(date)"
        echo "---"
        echo ""
    } >"$log_file" 2>&1
}

export -f run_config
export vocab_size seq_len batch_size dataset_size train_fraction
export n_prints n_prints_model print_scale num_epochs lr experiment_name save_data K mode d_model

parallel -j "${jobs}" --colsep ' ' run_config {1} {2} ::: "${configurations[@]}"

echo "All training runs completed!"