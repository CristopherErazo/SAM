#!/bin/bash

jobs=4
# Parameters
d=50                # Input dimension
tch_act='He4'        # Activation function for teacher
std_act='He4'        # Activation function for student
n_test=8000           # Number of test samples
bs=1                # Batch size for training
lr=0.05                   # Learning rate
eps=0.0               # Noise level for labels
q=2.0                 # q-norm for SAM
nprints=30            # Number of prints during training
nsteps=10
k=4.0                 # Information exponent
gamma=0.0             # Momentum for optimizers
  
# # Loop over various configurations = (opt  rho  lr)

configurations=(
    # 'SAM 0.0 0.01'
    # 'SAM 0.0 0.001'
    'SAM 0.0001'
    'SAM 0.001'
    'SAM 0.01'
    'SAM 0.1'
)


run_config() {
    local opt="$1"
    local rho="$2"
    local log_file
    log_file="logs/${opt}_${rho}.log"

    echo "Logging to ${log_file}"

    {
        python -u ./scripts/main.py \
            --d $d \
            --tch_act $tch_act \
            --std_act $std_act \
            --n_test $n_test \
            --bs $bs \
            --opt $opt \
            --lr $lr \
            --gamma $gamma \
            --eps $eps \
            --q $q \
            --rho $rho \
            --nprints $nprints \
            --n_steps $nsteps\
            --k $k


        echo "Completed: at $(date)"
        echo "---"
        echo ""
    } >"$log_file" 2>&1
}

export -f run_config
export d tch_act std_act n_test bs lr gamma eps q rho nprints nsteps k opt


parallel -j "${jobs}" --colsep ' ' run_config {1} {2} ::: "${configurations[@]}"

echo "All training runs completed!"





# for config in "${configurations[@]}"; do
#     read -r  opt gamma rho  <<< "$config"
#     python -u ./scripts/main.py \
#         --d $d \
#         --tch_act $tch_act \
#         --std_act $std_act \
#         --n_test $n_test \
#         --bs $bs \
#         --opt $opt \
#         --lr $lr \
#         --gamma $gamma \
#         --eps $eps \
#         --q $q \
#         --rho $rho \
#         --nprints $nprints \
#         --n_steps $nsteps\
#         --k $k


#     echo "Completed: at $(date)"
#     echo "---"
#     echo ""

# done

# echo "All training runs completed!"