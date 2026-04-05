#!/bin/bash

# Make log directory if it doesn't exist
mkdir -p logs

# Meta parameters
dropout=0.0 # Dropout rate
n_prints=60 # Number of prints during training
steps=600 # Number of training epochs
experiment_name='dual_task_new' # Name of the experiment for saving results
n_prints_model=6 # Number of times to save model checkpoints during training.
print_scale='linear' # Scale for printing steps: log or linear
init='random' # initialization method: planted or random
momentum=0.9 # Momentum for SGD optimizer
weight_decay=0.0 # Weight decay for optimizers

# Fix parameters
vocab_size=64 # Vocabulary size
seq_len=256  # Sequence length
d_model=128 # Model dimension
batch_size=64 # Batch size
opt='adam' # Optimizer
test_size=200 # Number of samples in the test set

# Variable parameters
K=20 # Number of trigger tokens
lr=0.008 # Learning rate
b_type='spiked' # P_b distribution type: dirichlet or spiked
u_type='zipf' # P_u distribution type: dirichlet or zipf (only used if b_type is spiked)
alpha=1 # Dirichlet concentration parameter or exponent for the Zipf's law
beta=0.9 # Beta parameter for spiked bigram distribution (only used if b_type is spiked)
fix_trig='True' # Whether to fix the trigger tokens across all experiments.
trig_type='freq' # Whether the trigger tokens should be the most freq, rare or random according to P_u. Only used if fix_trig is True.

# Configurations to loop over
configurations=(
    'dirichlet 15'
    'spiked 20'
)

# Function to run a single configuration
run_config() {
    local b_type="$1"
    local K="$2"
    local log_file
    log_file="logs/${experiment_name}_btype${b_type}_K${K}_utype${u_type}.log"

    echo "Logging to ${log_file}"
    time_init=$(date +%s)

    {
        python -u ./scripts/dual_task_train.py \
            --vocab_size $vocab_size \
            --seq_len $seq_len \
            --K $K \
            --d_model $d_model \
            --dropout $dropout \
            --batch_size $batch_size \
            --lr $lr \
            --n_prints $n_prints \
            --steps $steps \
            --opt $opt \
            --experiment_name $experiment_name \
            --n_prints_model $n_prints_model \
            --print_scale $print_scale \
            --b_type $b_type \
            --u_type $u_type \
            --alpha $alpha \
            --beta $beta \
            --fix_trig $fix_trig \
            --trig_type $trig_type \
            --init $init \
            --test_size $test_size \
            --momentum $momentum \
            --weight_decay $weight_decay

        echo "Completed: at $(date)"
        echo "---"
        echo ""
    } >"$log_file" 2>&1

    time_end=$(date +%s)
    elapsed=$((time_end - time_init))
    echo "Elapsed time  ${elapsed} seconds"
}

# Loop over configurations
for config in "${configurations[@]}"; do
    IFS=' ' read -r b_type K <<< "$config"
    run_config "$b_type" "$K"
done

echo "All training runs completed!"