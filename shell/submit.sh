#!/bin/bash

# Parameters
d=1000                  # Input dimension
tch_act='tanh'        # Activation function for teacher
std_act='tanh'        # Activation function for student
n_test=8000           # Number of test samples
bs=10                # Batch size for training
lr=0.5                # Learning rate
eps=0.0               # Noise level for labels
q=2.0                 # q-norm for SAM
nprints=1000            # Number of prints during training
nsteps=100000
  
# # Loop over various configurations = (opt , gamma , rho)

configurations=(
    'SAM 0.0 0.01'
    'SAM 0.0 0.001'
    'SGD 0.0 0.0'
    'SGD 0.000001 0.0'
    'SGD 0.0000001 0.0'
)


for config in "${configurations[@]}"; do
    read -r  opt gamma rho  <<< "$config"
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
        --n_steps $nsteps


    echo "Completed: at $(date)"
    echo "---"
    echo ""

done

echo "All training runs completed!"