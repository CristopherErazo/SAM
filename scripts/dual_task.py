import argparse
import torch
import torch.nn as nn
import numpy as np
import time


from sam.dual_model import DualModel , initialize_dual_model
from sam.dataset import get_dataloader_dual_task , get_distributions , get_sample_dual_task, generate_dual_task_batch
from sam.evaluation import evaluate_induction_bigram
from sam.optimizers import SAM_Optimizer

from configurations import save_data , make_data_paths



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
    parser.add_argument('--opt', type=str, default='SGD', help='Type of optimizer: SGD, adam, or SAM')
    parser.add_argument('--experiment_name', type=str, default='linear_attention', help='Name of the experiment for saving results')
    parser.add_argument('--n_prints_model', type=int, default=5, help='Number of times to save model checkpoints during training.')
    parser.add_argument('--print_scale',type=str, default='log', help='Scale for printing steps: log or linear')
    parser.add_argument('--mode', type=str, default='uniform', help='Data distribution mode: uniform or random')
    parser.add_argument('--alpha', type=float, default=1.0, help='Exponent for the unigram distribution (Zipf\'s law)')

    args = parser.parse_args()
    config = vars(args)

    print("Configuration:")
    print(config)

 
    # Define the distributions for the dual task    
    # P_b, P_u, P_o, P_t = get_distributions(config['vocab_size'], config['mode'], beta=2.0)

    P_t = torch.ones(config['vocab_size']) / config['vocab_size'] # Uniform distribution over triggers
    P_o = torch.ones((config['vocab_size'],config['vocab_size'])) / (config['vocab_size']-1) # Uniform distribution over outputs given triggers apart from the trigger token itself
    P_o.fill_diagonal_(0) # Set diagonal to 0 since output cannot be

    # Unigram distribution follows zipf's law with exponent alpha    
    ranks = torch.arange(1, config['vocab_size'] + 1)
    P_u = 1.0 / ranks**config['alpha']
    P_u /= P_u.sum() # Normalize to get a valid probability distribution

    # Bigram distribution is the same as unigram for each token
    P_b = P_u.unsqueeze(0).repeat(config['vocab_size'], 1)

    # Initialize Model
    model = DualModel(config)
    model , device = initialize_dual_model(model , P_b)
    print("Model initialized at device: ", device) 

    # Define loss
    loss_fn = nn.CrossEntropyLoss()

    # List of V values
    V_values = np.logspace(4, 9, num=12, base=2, dtype=int)


    # Test several trigger numbers
    K_values = np.unique(np.linspace(1, config['vocab_size'], num=15, dtype=int))

    results = {
        'K_values': K_values,
        'loss_bi': [],
        'loss_ind': [],
        'loss_tot': [],
    }
    results['entropy_P_u']= (-(P_u * torch.log(P_u + 1e-10)).sum().item()) # Add small constant for numerical stability
    results['max_entropy']= (np.log(config['vocab_size']))

    for K in K_values:
        config['K'] = K

        print(f"\nRunning experiment with K: {K}, initializing...")
       

        # Test planted model
        loss_bi = 0.0
        loss_ind = 0.0
        loss_tot = 0.0
        model.eval()
        with torch.no_grad():
            for step in range(config['steps']):
                if step % 5 == 0:
                    print(f"Step {step}/{config['steps']}")
                # Generate batch of data for the dual task
                batch = generate_dual_task_batch(config['batch_size'], config['seq_len'], config['K'], P_b, P_u, P_o, P_t)
                l_b , l_ind, l_tot = evaluate_induction_bigram(model, batch, loss_fn)
                loss_bi += l_b
                loss_ind += l_ind
                loss_tot += l_tot

                
                
        results['loss_bi'].append(loss_bi / config['steps'])
        results['loss_ind'].append(loss_ind / config['steps'])
        results['loss_tot'].append(loss_tot / config['steps'])
        print(f"Experiment with K: {K} completed. Loss Bigram: {results['loss_bi'][-1]:.4f}, Loss Induction: {results['loss_ind'][-1]:.4f}")

    # Save results
    save_data(results, 'dual_task_compare_loss','tmp',params={'d_model': config['d_model'], 'alpha': config['alpha']})



if __name__ == "__main__":
    main()