import yaml
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



# ,'fix_trig','freq_trig','K'