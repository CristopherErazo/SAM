#!/bin/bash

# Make log directory if it doesn't exist
mkdir -p logs

experiment_name='dual_task_new' # Name of the experiment for saving results

# paths=() # Paths to run another: full_trigg
paths=('full' 'induction' 'bigram' 'full_trigg')
Ks=(15) # Values of K to loop over (if needed for different configurations)
btypes=('dirichlet' 'spiked_zipf' )

# Loop over btypes, paths, and K values
for btype in "${btypes[@]}"; do
    for path in "${paths[@]}"; do
        for K in "${Ks[@]}"; do
            file_name="${btype}_${path}_K${K}"
            log_file="logs/${file_name}.log"
            echo "Logging to ${log_file}"
            time_init=$(date +%s)
            # If btype='dirichlet', define 'b_type' as 'dirichlet'
            # If btype='spiked_dirichlet', define 'b_type' as 'spiked' and 'u_type' as 'dirichlet'
            # If btype='spiked_zipf', define 'b_type' as 'spiked' and 'u_type' as 'zipf'
            if [ "$btype" == "dirichlet" ]; then
                b_type="dirichlet"
                u_type=""
            elif [ "$btype" == "spiked_dirichlet" ]; then
                b_type="spiked"
                u_type="dirichlet"
            elif [ "$btype" == "spiked_zipf" ]; then
                b_type="spiked"
                u_type="zipf"
            else
                echo "Invalid btype: $btype. Skipping."
                continue
            fi
            echo "Running configuration: btype=${btype}, path=${path}, K=${K} which corresponds to b_type=${b_type} and u_type=${u_type}"

            {
                python -u ./scripts/dual_task_train_new.py\
                    extra_args.path=$path \
                    extra_args.experiment_name=$experiment_name \
                    extra_args.file_name=$file_name \
                    data_args.b_type=$b_type \
                    data_args.u_type=$u_type \
                    data_args.K=$K

                echo "Completed: at $(date)"
                echo "---"
                echo ""
            } >"$log_file" 2>&1
        done
    done
done

