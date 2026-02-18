import argparse
import torch
import torch.nn as nn
import numpy as np
import time

# from sam.models import create_induction_head , planted_initialization, interpolation_initialization, InductionHeadAttentionSmaller, planted_initialization_small
# from sam.small_models import InductionHeadAttention, warm_initialization, interpolation_initialization
from sam.small_models import noisy_initialization
from sam.dataset import get_dataloader
from sam.evaluation import evaluate_model
from sam.optimizers import SAM_Optimizer

from configurations import save_data , make_data_paths



def main():
    parser = argparse.ArgumentParser(description="Training Attention-Only Transformer on Copying Task")

    parser.add_argument('--vocab_size', type=int, default=100, help='Vocabulary size')
    parser.add_argument('--seq_len', type=int, default=64, help='Sequence length')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--dataset_size', type=int, default=1000, help='Total dataset size')
    parser.add_argument('--train_fraction', type=float, default=0.8, help='Fraction of data for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--n_prints', type=int, default=50, help='Number of times to print during training.')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs.')
    parser.add_argument('--alpha', type=float, default=0.05, help='Noise level for parameter initialization.')
    parser.add_argument('--beta_1', type=float, default=1.0, help='Induction head beta_1 parameter.')
    parser.add_argument('--beta_2', type=float, default=1.0, help='Induction head beta_2 parameter.')   
    parser.add_argument('--beta_out', type=float, default=1.0, help='Induction head beta_out parameter.')
    parser.add_argument('--sigma', type=float, default=0.5, help='Sigma for interpolation initialization.')
    parser.add_argument('--cV', type=float, default=1.0, help='Coefficient for WV1.')
    parser.add_argument('--gamma',type=float,default=0.01,help='Weigth decay for sgd')
    parser.add_argument('--rho',type=float,default=0.05,help='Rho parameter for SAM optimizer')
    parser.add_argument('--opt', type=str, default='SGD', help='Type of optimizer: SGD, adam, or SAM')
    parser.add_argument('--p_error', type=float, default=0.0, help='Probability of introducing noise in the target for the permutation task')
    
    args = parser.parse_args()
    config = vars(args)

    print("Configuration:")
    print(config)

    # Create model and move to device
    train_list = ['WQ1.weight', 'WK1.weight','WV1.weight','WQ2.weight', 'WK2.weight']
    model , device = noisy_initialization(config, train_list=train_list)
    print("Model created on device:", device)

    # Freeze some parameters if needed
    # list_parameters = ['beta_1', 'beta_2', 'beta_out', 
    # 'embedding.weight', 'positions.weight', 
    # 'WQ1.weight', 'WK1.weight', 'WV1.weight', 
    # 'WQ2.weight', 'WK2.weight', 'WV2.weight']

    # Freeze all parameters except the ones on the train_list
    for name, param in model.named_parameters():
        if name not in train_list:
            param.requires_grad = False


    # Get dataloaders
    train_loader , val_loader = get_dataloader(config)

    # Define loss function and optimizer
    CE_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    # optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'])
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'])

    if config['opt'] == 'SGD':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'],weight_decay=config['gamma'])
        closure = None
    elif config['opt'] == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'])
        closure = None  
    elif config['opt'] == 'SAM':
        optimizer = SAM_Optimizer(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'], q=2, rho=config['rho'])
        print('Using SAM optimizer with rho = ', config['rho'])
        def closure():
            logits = model(input) # (batch_size, vocab_size)
            loss = CE_loss(logits.view(-1, config['vocab_size']), target.view(-1))
            return loss

    # Training loop parameters
    tot_global_steps = config['num_epochs']*len(train_loader)
    nprints = config['n_prints']
    print_every = max(1,tot_global_steps // nprints)
    global_step = 0


    # Dictionary to store process
    summary = {
        'step': [],
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'target_mass': []
    }

    # Save data
    fix_params = { key : config[key] for key in ['vocab_size','seq_len','lr']}
    variable_params = { key : config[key] for key in ['alpha','cV','opt','rho','gamma','p_error']}

    params = {'fixed' : fix_params,
              'variable': variable_params}

    # Save checkpoint of model at the beggining of training
    file_path , _, _ = make_data_paths('model_init', experiment_name= 'linear_attention', params=params,ext='pt') 
    print('Saving model checkpoint to ', file_path)
    torch.save(model.state_dict(), file_path)


    print(f'\n Loss with uniform initialization = logV = {np.log(config["vocab_size"]):.4f}\n')

    # Example: Iterate through the training dataloader
    time_start = time.time()
    for epoch in range(config['num_epochs']):
        for batch in train_loader:
            input = batch['input'].to(device) # (batch_size, seq_len)
            target = batch['target'].to(device) # (batch_size, )
            logits = model(input) # (batch_size, vocab_size)
            loss = CE_loss(logits.view(-1, config['vocab_size']), target.view(-1))
            # loss.backward()

            # Check for print condition to evaluate and print
            if global_step % print_every == 0:
                summary['step'].append(global_step)
                val_loss , val_accuracy , target_mass = evaluate_model(model, val_loader, device, CE_loss)

                text = ''
                for key , variable in zip(['val_loss','val_accuracy','target_mass','train_loss',],
                                          [val_loss, val_accuracy, target_mass,loss.item()]):
                    text += f'{key}: {variable:.4f}  '
                    summary[key].append(variable)
                
                print(f'Step {global_step}/{tot_global_steps}  ' + text )
            
            # if global_step == 23000:
            #     print(' switch')
            #     optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'],weight_decay=config['gamma'])

            

            global_step += 1
            if closure is None:
                loss.backward()
                optimizer.step()
            else:
                optimizer.step(closure)
            optimizer.zero_grad()
    
    

    print('Training completed.')
    print('Total training time (min): ', (time.time() - time_start)/60)
    print('Total training time (h): ', (time.time() - time_start)/3600)

    for key in summary:
        summary[key] = np.array(summary[key])
        print(f'{key} : {summary[key].shape}')


    save_data(summary,'summary',experiment_name='linear_attention', params=params)

    # Save checkpoint of model at the end of training

    # Save checkpoint of model at the beggining of training
    file_path , _, _ = make_data_paths('model_fin', experiment_name= 'linear_attention', params=params,ext='pt') 
    print('Saving model checkpoint to ', file_path)
    torch.save(model.state_dict(), file_path)
    
    

if __name__ == "__main__":
    main()    