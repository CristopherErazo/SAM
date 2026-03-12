import argparse
import torch
import torch.nn as nn
import numpy as np
import time


from sam.dual_model import DualModel , initialize_dual_model
from sam.dataset import get_dataloader_dual_task , get_distributions , get_sample_dual_task
from sam.evaluation import evaluate_model_lin_sfm
from sam.optimizers import SAM_Optimizer

from configurations import save_data , make_data_paths



def main():
    parser = argparse.ArgumentParser(description="Training Attention-Only Transformer on Copying Task")

    parser.add_argument('--vocab_size', type=int, default=100, help='Vocabulary size')
    parser.add_argument('--seq_len', type=int, default=64, help='Sequence length')
    parser.add_argument('--K', type=int, default=4, help='Number of Trigger Tokens')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--dataset_size', type=int, default=1000, help='Total dataset size')
    parser.add_argument('--train_fraction', type=float, default=0.8, help='Fraction of data for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--n_prints', type=int, default=50, help='Number of times to print during training.')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs.')
    parser.add_argument('--opt', type=str, default='SGD', help='Type of optimizer: SGD, adam, or SAM')
    parser.add_argument('--experiment_name', type=str, default='linear_attention', help='Name of the experiment for saving results')
    parser.add_argument('--n_prints_model', type=int, default=5, help='Number of times to save model checkpoints during training.')
    parser.add_argument('--print_scale',type=str, default='log', help='Scale for printing steps: log or linear')
    parser.add_argument('--save_data', type=str, default='False', help='Whether to save training data and model checkpoints')
    parser.add_argument('--mode', type=str, default='uniform', help='Data distribution mode: uniform or random')

    args = parser.parse_args()
    config = vars(args)

    print("Configuration:")
    print(config)

    for key in ['save_data']:
        config[key] = config[key].lower() == 'true'

    
    P_b, P_u, P_o, P_t = get_distributions(config['vocab_size'], config['mode'], beta=2.0)

    # Order tokens ranks with descending P_u
    P_tot = P_b * P_u[:,None]  # shape (V, V)
    P_marginal = P_tot.sum(axis=0)  # shape (V,)
    idx = torch.argsort(P_marginal, descending=True)
    P_u = P_u[idx]
    P_b = P_b[:,idx]


    # Conditional entropy
    H = - torch.sum(P_b * P_u[:,None] * torch.log(P_u[:,None] + 1e-10))
    print(f"Conditional entropy H(Y|X) = {H.item():.4f} bits")
    print(f'Loss at random chance (log(vocab_size)) : {np.log(config["vocab_size"]):.4f}')

    # trigger_set = np.random.choice(config['vocab_size'], size=config['K'], replace=False).tolist()
    trigger_set = [0]
    # P_o[0] = 0 # Set the probabilities of the first trigger token to 0 for all input tokens to ensure it is not selected as a trigger token
    # P_o[0,1] = 1.0 # Set the probability of the first trigger token to 1 for the first input token to ensure it is selected as a trigger token

    train_loader , val_loader = get_dataloader_dual_task(config,P_b,P_u,P_o,trigger_set)


    model = DualModel(config['d_model'], config['vocab_size'], config['seq_len'], config['dropout'])
    model , device = initialize_dual_model(model , P_b, trigger_set)
    print("Model initialized at device: ", device)  

    loss_fn = nn.CrossEntropyLoss()

    # Check trainable parameters
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Trainable: {param.requires_grad}")

    # Compute loss before training
    running_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            input = batch['input'].to(device) # shape (batch_size, seq_len)
            mask = batch['mask'].to(device) # shape (batch_size, seq_len, seq_len)
            target = batch['target'].to(device) # shape (batch_size, seq_len)
            # logits = model(input, mask) # shape (batch_size, seq_len, vocab_size)
            output = model.full_output(input, mask) # shape (batch_size, seq_len, vocab_size)
            output['input'] = input.cpu().numpy()
            save_data(output,'temp','tmp')
            break
     
            # Evaluate the loss only from the second token onwards, since the first token is only used to generate the second token and does not have a meaningful target
            loss = loss_fn(logits[:, 1:, :].reshape(-1, config['vocab_size']), target[:, 1:].reshape(-1))

            # Evaluate the loss only at the last token
            # loss = loss_fn(logits[:, -1, :], target[:, -1])

            
            print('loss',loss.item())
            running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"Initial validation loss before training: {avg_loss:.4f}")
    print(f'Loss at random chance (log(vocab_size)) : {np.log(config["vocab_size"]):.4f}')

if __name__ == "__main__":
    main()