#!/bin/bash

# Parameters
d=200                  # Input dimension
tch_act='tanh'        # Activation function for teacher
std_act='tanh'        # Activation function for student
n_train=1000          # Number of training samples
n_test=100           # Number of test samples
bs=10                # Batch size for training
# opt='SGD'             # Type of optimizer: SGD, adam, or SAM
lr=0.1                # Learning rate
eps=0.0               # Noise level for labels
q=2.0                 # q-norm for SAM
rho=0.1               # Radius for SAM
nprints=20            # Number of prints during training

  
# Loop over various configurations = opt
configurations=(
    'SGD'
    'SAM'
)

for config in "${configurations[@]}"; do
    read -r  opt <<< "$config"
    python -u ./scripts/main.py \
        --d $d \
        --tch_act $tch_act \
        --std_act $std_act \
        --n_train $n_train \
        --n_test $n_test \
        --bs $bs \
        --opt $opt \
        --lr $lr \
        --eps $eps \
        --q $q \
        --rho $rho \
        --nprints $nprints


    echo "Completed: at $(date)"
    echo "---"
    echo ""

done

echo "All training runs completed!"