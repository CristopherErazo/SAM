import yaml
import pickle
import numpy as np
# import torch
from configurations import load_data , make_data_paths, set_font_sizes, apply_general_styles, make_params_dict




def get_config():
    with open('./data/dual_task_new/config.yaml') as f:
        params = yaml.safe_load(f)
    return params
    

def get_scalar_measurements(params, Ks,configs,metrics):
    experiment_name = params['fixed']['experiment_name']
    del(params['fixed']['experiment_name'])
    results = {}
    for conf in configs:
        if conf == 'dirichlet_full':
            params['variable']['b_type'] = 'dirichlet'
            params['variable']['u_type'] = 'dirichlet'
        elif conf == 'dirichlet_spiked':
            params['variable']['b_type'] = 'spiked'
            params['variable']['u_type'] = 'dirichlet'
        elif conf == 'zipf_spiked':
            params['variable']['b_type'] = 'spiked'
            params['variable']['u_type'] = 'zipf'
            params['variable']['alpha'] = 1.0
        else:
            raise ValueError(f"Unknown configuration: {conf}")
        
        results[conf] = {}
        for K in Ks:
            params['variable']['K'] = K
            data = load_data('dual_task_train',
                                experiment_name=f"{experiment_name}/measures",
                                params=params,base_dir='./data',show=False)
            
            result = {}
            for key in metrics.keys():
                result[key] = {}
                for name in metrics[key]:
                    result[key][name] = data[name]
                    
            results[conf][K] = result

    return results



def get_attn_patterns(results_matrix, batch_id=0):
    trigg_set = results_matrix['test_batch']['trigger_set'][batch_id]
    out_set = results_matrix['test_batch']['output_set'][batch_id]
    seq = results_matrix['test_batch']['sequence'][batch_id][:-1]
    is_trigg = results_matrix['test_batch']['is_trigg'][batch_id]
    counts = results_matrix['test_batch']['counts'][batch_id] 
    steps = results_matrix['step']
    print('shapes of test batch components: ')
    print(f"trigg_set: {trigg_set.shape}, out_set: {out_set.shape}, seq: {seq.shape}, is_trigg: {is_trigg.shape}, counts: {counts.shape}")

    attn1 = results_matrix[f'attn1'][:,batch_id] # (10,256,256)
    attn2 = results_matrix[f'attn2'][:,batch_id]
    print(f"Attention patterns loaded with shapes: att1 {attn1.shape}, att2 {attn2.shape}")
    correct_attn = []
    leng = attn1.shape[1]
    for it , t in enumerate(steps):
        correct_attention = np.zeros(leng)
        for mu , tau in enumerate(seq):
            if is_trigg[mu] == 1 and counts[mu] > 1:
                # Map trigger tokens to their corresponding output tokens
                output_token = out_set[(trigg_set == tau).nonzero(as_tuple=True)[0].item()]
                # For the mu-th row of the attention pattern get the attention weigth of all output tokens and sum them up
                attention_weights = attn2[it,mu] # (256,)
                correct_attention[mu] = attention_weights[seq == output_token].sum()
                # # Compute the total attention in second pattern from tau to output_token
                # total_attention = results_matrix[f'attn2'][step_id,batch_id][tau.item(),output_token.item()]
                # print(f"Trigger token {tau.item()} attends to output token {output_token.item()} with total attention {total_attention:.4f}")
        correct_attn.append(correct_attention)
    
    correct_attn = np.array(correct_attn)
    return attn1, attn2, correct_attn, steps
# attn1: (10, 5, 256, 256)
# attn2: (10, 5, 256, 256)

def get_new_data_scalars(paths,Ks,btypes):
    print('Loading data...')
    results_scalar = {}
    for btype in btypes:
        results_scalar[btype] = {}
        for K in Ks:
            results_scalar[btype][K] = {}
            for path in paths:
                with open(f"./data/dual_task_new/{btype}_{path}_K{K}_scalar.pkl", "rb") as f:
                    results_scalar[btype][K][path] =  pickle.load(f)
    print('Data loaded successfully.')
    return results_scalar

def get_new_data_matrix(paths,Ks,btypes,batch_id=0):
    print('Loading data...')
    results_matrix = {}
    for btype in btypes:
        results_matrix[btype] = {}
        for K in Ks:
            results_matrix[btype][K] = {}
            for path in paths:
                with open(f"./data/dual_task_new/{btype}_{path}_K{K}_matrix.pkl", "rb") as f:
                    res = pickle.load(f)
                att1, att2, correct_attn, steps = get_attn_patterns(res, batch_id=batch_id)
                results_matrix[btype][K][path] = {
                    'att1': att1,
                    'att2': att2,
                    'correct_attn': correct_attn,
                    'steps': steps  }
                    # results_matrix[btype][K][path] = pickle.load(f)     
    print('Data loaded successfully.')
    return results_matrix