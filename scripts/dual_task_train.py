import argparse
import torch
import torch.nn as nn
import numpy as np
import time
import math


from sam.dual_model import DualModel , initialize_dual_model
from sam.dataset import get_dataloader_dual_task , get_distributions , get_sample_dual_task, generate_dual_task_batch, compute_entropies_and_dkl, get_triggers
from sam.evaluation import evaluate_induction_bigram , evaluate_dual_model, optimal_pop_losses
from sam.optimizers import SAM_Optimizer

from configurations import save_data , make_data_paths, make_params_dict



def main():
    parser = argparse.ArgumentParser(description="Training Attention-Only Transformer on Copying Task")

    parser.add_argument('--vocab_size', type=int, default=100, help='Vocabulary size')
    parser.add_argument('--seq_len', type=int, default=64, help='Sequence length')
    parser.add_argument('--K', type=int, default=4, help='Number of Trigger Tokens')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for each step')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--n_prints', type=int, default=50, help='Number of times to print during training.')
    parser.add_argument('--steps', type=int, default=5, help='Number of training steps.')
    parser.add_argument('--opt', type=str, default='sgd', help='Type of optimizer: SGD, adam, or adamW')
    parser.add_argument('--experiment_name', type=str, default='tmp', help='Name of the experiment for saving results')
    parser.add_argument('--n_prints_model', type=int, default=5, help='Number of times to save model checkpoints during training.')
    parser.add_argument('--print_scale',type=str, default='log', help='Scale for printing steps: log or linear')
    parser.add_argument('--b_type', type=str, default='dirichlet', help='P_b distribution type: dirichlet or spiked')
    parser.add_argument('--u_type', type=str, default='zipf', help='P_u distribution type: dirichlet or zipf (only used if b_type is spiked)')
    parser.add_argument('--alpha', type=float, default=1.0, help='Dirichlet concentration parameter or exponent for the Zipf\'s law')
    parser.add_argument('--beta', type=float, default=0.5, help='Beta parameter for spiked bigram distribution (only used if b_type is spiked)')
    parser.add_argument('--fix_trig',type=str,default='True', help='Whether to fix the trigger tokens across all experiments. If False, trigger tokens will be randomly sampled at each sequence')
    parser.add_argument('--trig_type', type=str, default='rare', help='Whether the trigger tokens should be the most freq, rare or random according to P_u. Only used if fix_trig is True.')
    parser.add_argument('--init', type=str, default='random', help='Initialization method: planted or random')
    parser.add_argument('--test_size', type=int, default=1000, help='Number of samples in the test set')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizers')
    

   
    args = parser.parse_args()
    config = vars(args)

    fix_params = { key : config[key] for key in ['vocab_size','seq_len','d_model','batch_size','opt','test_size']}
    variable_params = { key : config[key] for key in ['K','lr','b_type','u_type','alpha','beta','fix_trig','trig_type']}

    
    params = {'fixed' : fix_params,
              'variable': variable_params}

    print("Configuration:")
    print(config)

    for key in ['fix_trig']:
        config[key] = config[key].lower() == 'true' # Convert string to boolean

 
    # Define device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Define distributions
    P_b, P_u, P_o, P_t = get_distributions(config['vocab_size'], config['b_type'],config['alpha'],device,config['u_type'],config['beta'])
    if config['b_type'] == 'spiked':
        if config['u_type'] == 'dirichlet':
            print(f"Generated spiked bigram distribution where P_u is sampled from a Dirichlet distribution\n with alpha = {config['alpha']} and bigram distribution P_b is generated with beta = {config['beta']}")
        elif config['u_type'] == 'zipf':
            print(f"Generated spiked bigram distribution where P_u is generated from a Zipf distribution with\n exponent alpha = {config['alpha']} and bigram distribution P_b is generated with beta = {config['beta']}")
    else:
        print(f"Generated bigram distribution P_b sampled from a Dirichlet distribution with\n alpha = {config['alpha']} and unigram distribution P_u as the stationary distribution of P_b")
    # Compute entropies and reference KL divergences
    kl_Pb_uniform, kl_Pu_uniform, entropy_Pb, entropy_Pu, max_entropy = compute_entropies_and_dkl(P_b, P_u)

    print(f"KL(P_b || uniform): {kl_Pb_uniform:.4f}, KL(P_u || uniform): {kl_Pu_uniform:.4f}, Entropy(P_b): {entropy_Pb:.4f}, Entropy(P_u): {entropy_Pu:.4f}, Max Entropy = log V: {max_entropy:.4f}")


    # Initialize Model
    model = DualModel(config)

    trigger_set = get_triggers(config['fix_trig'], config['trig_type'], config['K'], P_t)
    

    print(f"Initializing...")
    model , device = initialize_dual_model(model , P_b, trigger_set=trigger_set,init=config['init'])
    print("Model initialized at device: ", device)

    # Freeze the parameters exept for some
    # no_freeze = ['WF.weight']
    no_freeze = ['WQK1.weight','WQK2.weight','WOV2.weight','WF.weight']
    for name, param in model.named_parameters():
        if not any([nf in name for nf in no_freeze]):
            param.requires_grad = False

    # Print trainable parameters
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    # Define loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    if config['opt'] == 'sgd':
        print("Using SGD optimizer")
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay'])
    elif config['opt'] == 'adam':
        print("Using Adam optimizer")
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'],weight_decay=config['weight_decay'])
    elif config['opt'] == 'adamW':
        print("Using AdamW optimizer")
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'],weight_decay=config['weight_decay'])
    else:
        raise ValueError("Invalid optimizer type. Options are 'SGD', 'adam', and 'adamW'.")
    

    test_batch = generate_dual_task_batch(config['test_size'], config['seq_len'], config['K'], P_b, P_u, P_o, P_t, trigger_set=trigger_set)

    phi1, phi2, phi3, trigg_per_seq = optimal_pop_losses(test_batch, P_b=P_b, p0=0.999)
 
    print(f"Optimal population losses on the test set: Loss_dual = {phi1:.4f}, Loss_ind = {phi2:.4f}, Loss_b = {phi3:.4f}\nAverage triggers per sequence = {trigg_per_seq:.4f}")

    results = {
        'model_steps': [],
        'step': [],
        'loss': [],
        'accuracy_top3': [],
        'accuracy': [],
        'kl_WF_b': [],
        'kl_b_WF': [],
        'kl_full_b': [],
        'kl_b_full': [],
        'kl_full_u': [],
        'loss_b_tot': [],
        'loss_b_filt': [],
        'loss_ind_tot': [],
        'loss_ind_filt': [],
        'entropy_Pb': entropy_Pb,
        'entropy_Pu': entropy_Pu,
        'kl_Pb_uniform': kl_Pb_uniform,
        'kl_Pu_uniform': kl_Pu_uniform,
        'max_entropy': max_entropy,
        'P_u': P_u.cpu().numpy(),
        'P_b': P_b.cpu().numpy(),
        'opt_loss_dual': phi1,
        'opt_loss_ind': phi2,
        'opt_loss_b': phi3,
        'trigg_per_seq': trigg_per_seq
    }




    # Training loop parameters
    tot_global_steps = config['steps']
    nprints = config['n_prints']
    nprints_model = config['n_prints_model']


    
    if config['print_scale'] == 'log':
        print_steps = np.unique(np.logspace(-0.01, np.log10(tot_global_steps-1), num=nprints).astype(int))
        print_steps_model = np.unique(np.logspace(-0.01, np.log10(tot_global_steps-1), num=nprints_model).astype(int))
    elif config['print_scale'] == 'linear':
        print_steps = np.linspace(0, tot_global_steps-1, num=nprints).astype(int)
        print_steps_model = np.linspace(0, tot_global_steps-1, num=nprints_model).astype(int)

    t0 = time.time()
    
    for step in range(config['steps']):

        if step in print_steps:  # return accuracy, loss_tot.item(), kl_WF.item(), accuracy_top3, kl_full_b.item() , kl_full_u.item()

            acc , loss ,kl_b_WF, kl_WF_b , acc3, kl_full_b , kl_b_full, kl_full_u, loss_b_tot, loss_b_filt, loss_ind_tot, loss_ind_filt = evaluate_dual_model(model,test_batch,loss_fn,P_b,P_u)
            
            results['step'].append(step)
            results['loss'].append(loss)
            results['accuracy'].append(acc)
            results['accuracy_top3'].append(acc3)
            results['kl_WF_b'].append(kl_WF_b)
            results['kl_b_WF'].append(kl_b_WF)
            results['kl_full_b'].append(kl_full_b)
            results['kl_b_full'].append(kl_b_full)
            results['kl_full_u'].append(kl_full_u)
            results['loss_b_tot'].append(loss_b_tot)
            results['loss_b_filt'].append(loss_b_filt)
            results['loss_ind_tot'].append(loss_ind_tot)
            results['loss_ind_filt'].append(loss_ind_filt)
            
            print(f"Step {step}/{config['steps']}: Loss = {loss:.4f}, Accuracy = {acc:.4f}, Top-3 Accuracy = {acc3:.4f}, KL(WF||P_b) = {kl_WF_b:.4f}, KL(full||P_b) = {kl_full_b:.4f}, KL(full||P_u) = {kl_full_u:.4f}, Loss_b_tot = {loss_b_tot:.4f}, Loss_b_filt = {loss_b_filt:.4f}, Loss_ind_tot = {loss_ind_tot:.4f}, Loss_ind_filt = {loss_ind_filt:.4f}")

        if step in print_steps_model:
            results['model_steps'].append(step)
            model_path = make_data_paths(f'dual_task_train_step{step}',f"{config['experiment_name']}/model",params=params,ext='pt')[0]
            torch.save(model.state_dict(), model_path)
            print(f"Saved model checkpoint at step {step} to {model_path}")

            
        batch = generate_dual_task_batch(config['batch_size'], config['seq_len'], config['K'], P_b, P_u, P_o, P_t, trigger_set=trigger_set)

        # Evaluate model on the dual task
        sequence = batch['sequence'].to(device) # shape (batch_size, seq_len + 1)
        input = sequence[:, :-1] # shape (batch_size, seq_len)
        target = sequence[:, 1:] # shape (batch_size, seq_len)
        mask = batch['mask'].to(device) # shape (batch_size, seq_len, seq_len)

        logits = model(input, mask, attn=True, fc=True) # shape (batch_size, seq_len, vocab_size)

        # Compute loss and update model
        loss = loss_fn(logits.reshape(-1, config['vocab_size']), target.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
    t1 = time.time()
    print(f"Training completed in {t1-t0:.2f} seconds = {((t1-t0)/60):.2f} minutes")

    for key in results:
        results[key] = np.array(results[key])
        print(f"{key}: {results[key].shape}")


    
    # Save results
    save_data(results, 'dual_task_train',f"{config['experiment_name']}/measures",params=params)



if __name__ == "__main__":
    main()