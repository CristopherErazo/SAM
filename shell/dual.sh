#!/bin/bash

# Make log directory if it doesn't exist
mkdir -p logs
jobs=1 # Number of parallel jobs to run

# Fix Parameters
d_model=2048 # Model dimension
vocab_size=64 # Vocabulary size
seq_len=64  # Sequence length
batch_size=64 # Batch size
K=10 # Number of trigger tokens
alpha=2.0 # exponent for zipf distribution 

# --------------------------------------------------

# print / run parameters
n_prints=100 # Number of prints during training
n_prints_model=5 # Number of times to save model checkpoints during training.
print_scale='log' # Scale for printing steps: log or linear
steps=20 # Number of training epochs

# Variable Parameters
lr=0.05 # Learning rate
experiment_name='tmp' # Name of the experiment for saving results


# Loop over various configurations 
configurations=(
    '42 256' 
)


run_config() {
    local vocab_size="$1"
    local seq_len="$2"
    local log_file
    # log_file="logs/${experiment_name}_${vocab_size}_${seq_len}.log"
    log_file="logs/${experiment_name}.log"

    echo "Logging to ${log_file}"
    time_init=$(date +%s)

    {
        python -u ./scripts/dual_task.py \
            --vocab_size $vocab_size \
            --seq_len $seq_len \
            --batch_size $batch_size \
            --n_prints $n_prints \
            --n_prints_model $n_prints_model \
            --print_scale $print_scale \
            --lr $lr \
            --experiment_name $experiment_name \
            --K $K \
            --alpha $alpha \
            --d_model $d_model \
            --steps $steps


        echo "Completed: at $(date)"
        echo "---"
        echo ""
    } >"$log_file" 2>&1

    time_end=$(date +%s)
    elapsed=$((time_end - time_init))
    echo "Elapsed time  ${elapsed} seconds"
}

export -f run_config
export vocab_size seq_len batch_size steps alpha
export n_prints n_prints_model print_scale lr experiment_name K mode d_model

parallel -j "${jobs}" --colsep ' ' run_config {1} {2} ::: "${configurations[@]}"

echo "All training runs completed!"