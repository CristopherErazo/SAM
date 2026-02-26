import argparse
import torch
import torch.nn as nn
import numpy as np
import time

# from sam.models import create_induction_head , planted_initialization, interpolation_initialization, InductionHeadAttentionSmaller, planted_initialization_small
# from sam.small_models import InductionHeadAttention, warm_initialization, interpolation_initialization
from sam.small_models import noisy_initialization
from sam.dataset import get_dataloader
from sam.evaluation import evaluate_model_lin_sfm
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
    parser.add_argument('--sigma', type=float, default=0.3, help='Sigma for interpolation initialization.')
    parser.add_argument('--cV', type=float, default=1.0, help='Coefficient for WV1.')
    parser.add_argument('--gamma',type=float,default=0.0,help='Weigth decay for sgd')
    parser.add_argument('--rho',type=float,default=0.05,help='Rho parameter for SAM optimizer')
    parser.add_argument('--opt', type=str, default='SGD', help='Type of optimizer: SGD, adam, or SAM')
    parser.add_argument('--p_error', type=float, default=0.0, help='Probability of introducing noise in the target for the permutation task')
    parser.add_argument('--attn', type=str, default='softmax', help='Type of attention: linear or softmax')
    parser.add_argument('--loss', type=str, default='CE', help='Type of loss function: CE or MSE')
    parser.add_argument('--experiment_name', type=str, default='linear_attention', help='Name of the experiment for saving results')
    parser.add_argument('--n_prints_model', type=int, default=5, help='Number of times to save model checkpoints during training.')
    parser.add_argument('--print_scale',type=str, default='log', help='Scale for printing steps: log or linear')
    parser.add_argument('--save_data', type=str, default='False', help='Whether to save training data and model checkpoints')


    args = parser.parse_args()
    config = vars(args)

    print("Configuration:")
    print(config)

    for key in ['save_data']:
        config[key] = config[key].lower() == 'true'
        
    # Create model and move to device
    train_list = ['WQ1.weight', 'WK1.weight','WV1.weight','WQ2.weight', 'WK2.weight']
    model , device = noisy_initialization(config, train_list=train_list)
    print("Model created on device:", device)

    # Freeze all parameters except the ones on the train_list
    for name, param in model.named_parameters():
        if name not in train_list:
            param.requires_grad = False


    # Get dataloaders
    train_loader , val_loader = get_dataloader(config)

    # Define loss function and optimizer
    if config['loss'] == 'CE':
        CE_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        loss_fn = CE_loss
    elif config['loss'] == 'MSE':
        MSE_loss = nn.MSELoss()
        loss_fn = MSE_loss

    
    if config['opt'] == 'SGD' or config['rho'] == 0.0:
        print('Using SGD optimizer with weight decay = ', config['gamma'])  
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'],weight_decay=config['gamma'])
        closure = None
    elif config['opt'] == 'adam':
        print('Using Adam optimizer')
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'])
        closure = None  
    elif config['opt'] == 'SAM':
        print('Using SAM optimizer with rho = ', config['rho'], ' and weight decay = ', config['gamma'])
        optimizer = SAM_Optimizer(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'], q=2, rho=config['rho'])
        print('Using SAM optimizer with rho = ', config['rho'])
        def closure():
            output = model(input) # (batch_size, vocab_size)
            if config['loss'] == 'CE':
                loss = loss_fn(output.view(-1, config['vocab_size']), target.view(-1))
            elif config['loss'] == 'MSE':
                 targ_emb = model.embedding(target) # (batch_size, vocab_size)
                 loss = loss_fn(output, targ_emb)
            return loss

    # Training loop parameters
    tot_global_steps = config['num_epochs']*len(train_loader)
    nprints = config['n_prints']
    nprints_model = config['n_prints_model']
    global_step = 0
    if config['print_scale'] == 'log':
        print_steps = np.unique(np.logspace(-0.01, np.log10(tot_global_steps-1), num=nprints).astype(int))
        print_steps_model = np.unique(np.logspace(-0.01, np.log10(tot_global_steps-1), num=nprints_model).astype(int))
    elif config['print_scale'] == 'linear':
        print_steps = np.linspace(0, tot_global_steps-1, num=nprints).astype(int)
        print_steps_model = np.linspace(0, tot_global_steps-1, num=nprints_model).astype(int)


    # Dictionary to store process
    summary = {
        'model_step': [],
        'global_step': [],
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'target_mass': []
    }


    # Save data
    fix_params = { key : config[key] for key in ['vocab_size','seq_len','lr','num_epochs']}
    variable_params = { key : config[key] for key in ['alpha','cV','opt','rho','p_error','attn','loss']}

    params = {'fixed' : fix_params,
              'variable': variable_params}
    
    # save_data(summary,'summary',experiment_name=config['experiment_name'], params=params)
    

    # Save train and val loaders for reproducibility
    if config['save_data']:
        save_data(train_loader, 'train_loader', experiment_name=config['experiment_name'], params=params)
        save_data(val_loader, 'val_loader', experiment_name=config['experiment_name'], params=params)

    print(f'\n Loss with uniform initialization = logV = {np.log(config["vocab_size"]):.4f}\n')

    # Example: Iterate through the training dataloader
    time_start = time.time()
    for epoch in range(config['num_epochs']):
        for batch in train_loader:
            input = batch['input'].to(device) # (batch_size, seq_len)
            target = batch['target'].to(device) # (batch_size, )
            output = model(input) # (batch_size, vocab_size)

            if config['loss'] == 'CE':
                loss = loss_fn(output.view(-1, config['vocab_size']), target.view(-1))
            elif config['loss'] == 'MSE':
                 targ_emb = model.embedding(target) # (batch_size, vocab_size)
                 loss = loss_fn(output, targ_emb)

            # Check for exaluation print condition to evaluate and print
            measure_condition = global_step in print_steps
            if measure_condition:
                summary['global_step'].append(global_step)
                
                val_loss , val_accuracy , target_mass =  evaluate_model_lin_sfm(model, val_loader, device, loss_fn,loss_type=config['loss'])

                text = ''
                for key , variable in zip(['val_loss','val_accuracy','target_mass','train_loss',],
                                          [val_loss, val_accuracy, target_mass,loss.item()]):
                    text += f'{key}: {variable:.4f}  '
                    summary[key].append(variable)
                
                print(f'Step {global_step}/{tot_global_steps}  ' + text )
            
            # Check for model checkpoint print condition to save model
            if global_step in print_steps_model:
                summary['model_step'].append(global_step)
                file_path , _, _ = make_data_paths(f'model_{global_step}', experiment_name= config['experiment_name'], params=params,ext='pt') 
                print('Saving model checkpoint to ', file_path)
                torch.save(model.state_dict(), file_path)


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

    summary['epc_step'] = summary['global_step']/len(train_loader)
    summary['epc_model_step'] = summary['model_step']/len(train_loader)

    save_data(summary,'summary',experiment_name=config['experiment_name'], params=params)
    

if __name__ == "__main__":
    main()    