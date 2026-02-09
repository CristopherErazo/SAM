import argparse
import torch
import torch.nn as nn
import numpy as np
import time

from sam.models import create_induction_head , planted_initialization, interpolation_initialization
from sam.models import InductionHeadAttentionSmaller, planted_initialization_small, interpolation_initialization_smaller
from sam.dataset import get_dataloader
from sam.evaluation import evaluate_model , evaluate_overlap_with_teacher
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
    parser.add_argument('--alpha', type=float, default=0.5, help='Noise level for parameter initialization.')
    parser.add_argument('--gamma',type=float,default=0.01,help='Weigth decay for sgd')
    parser.add_argument('--rho',type=float,default=0.05,help='Rho parameter for SAM optimizer')
    parser.add_argument('--opt', type=str, default='SGD', help='Type of optimizer: SGD, adam, or SAM')
    parser.add_argument('--alpha_load', type=float, default=0.1, help='Alpha parameter for loading model checkpoint')
    parser.add_argument('--lr_load', type=float, default=0.00005, help='Learning rate for loading model checkpoint')

    args = parser.parse_args()
    config = vars(args)

    print("Configuration:")
    print(config)

    # Load reate model in device
    fix_params = { key : config[key] for key in ['vocab_size','seq_len']}
    fix_params['lr'] = config['lr_load']
    variable_params = {'alpha': config['alpha_load']}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    teacher_model = InductionHeadAttentionSmaller(config['vocab_size'], config['seq_len']).to(device)
    # teacher_model , device = create_induction_head(config)
    params = {'fixed' : fix_params,
              'variable': variable_params}
    file_path , _, _ = make_data_paths(f'model_fin', experiment_name= 'small_induction_head', params=params,ext='pt',base_dir='./data')     
    teacher_model.load_state_dict(torch.load(file_path, map_location=device))
    
    # Create a copy of the teacher model to be trained
    model = InductionHeadAttentionSmaller(config['vocab_size'], config['seq_len']).to(device)
    model.load_state_dict(teacher_model.state_dict())
    model.to(device)
    
    # Interpolation initialization between teacher and random

    interpolation_initialization_smaller(model, alpha=config['alpha'])

    print("Model created on device:", device)

    # Freeze some parameters if needed
    # list_parameters = ['beta_1', 'beta_2', 'beta_out', 
    # 'embedding.weight', 'positions.weight', 
    # 'WQ1.weight', 'WK1.weight', 'WV1.weight', 
    # 'WQ2.weight', 'WK2.weight', 'WV2.weight']

    train_list = ['beta_1','beta_2','beta_out',
                  'positions.weight',
                  'WQ1.weight', 'WK1.weight', 'WV1.weight', 
                  'WQ2.weight', 'WK2.weight', 'WV2.weight'] # List of parameters to train (unfreeze)

    # Freeze all parameters except the ones on the train_list
    for name, param in model.named_parameters():
        if name not in train_list:
            param.requires_grad = False

    # Print trainable parameters
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name , param.shape)

    print(f'\n Loss with uniform initialization = logV = {np.log(config["vocab_size"]):.4f}\n')

    # Get dataloaders
    train_loader , val_loader = get_dataloader(config)

    # Define loss function and optimizer
    CE_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    # optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'])

    if config['opt'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'],weight_decay=config['gamma'])
        closure = None
    elif config['opt'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        closure = None  
    elif config['opt'] == 'SAM':
        optimizer =SAM_Optimizer(model.parameters(), lr=config['lr'], q=2, rho=config['rho'])
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

    intervene_acc = [1.9] # List of val_accuracy_sample thresholds for intervention
    n_interventions = 0
    tot_interventions = len(intervene_acc)

    # Dictionary to store process
    summary = {
        'step': [],
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_accuracy_sample': []
    }

    aligment = { 
        'overlap': {name : [] for (name, _) in model.named_parameters() },
        'difference': {name : [] for (name, _) in model.named_parameters() }}

    # Save data
    fix_params = { key : config[key] for key in ['vocab_size','seq_len','lr']}
    variable_params = { key : config[key] for key in ['alpha','rho','gamma','opt']}

    params = {'fixed' : fix_params,
              'variable': variable_params}

    # Save checkpoint of model at the beggining of training
    file_path , _, _ = make_data_paths('model_init', experiment_name= 'post_small_induction_head', params=params,ext='pt') 
    print('Saving model checkpoint to ', file_path)
    torch.save(model.state_dict(), file_path)

    # Example: Iterate through the training dataloader
    time_start = time.time()
    for epoch in range(config['num_epochs']):

        for batch in train_loader:
            input = batch['input'].to(device) # (batch_size, seq_len)
            target = batch['target'].to(device) # (batch_size, )
            # mask = batch['mask'].to(device) # (1, seq_len, seq_len)

            logits = model(input) # (batch_size, vocab_size)
            loss = CE_loss(logits.view(-1, config['vocab_size']), target.view(-1))
            # loss.backward()


            # Check for print condition to evaluate and print
            if global_step % print_every == 0:
                summary['step'].append(global_step)
                val_loss , val_accuracy , val_accuracy_sample = evaluate_model(model, val_loader, device, CE_loss)
                overlaps_step = evaluate_overlap_with_teacher(model, teacher_model, device)
                # train_loss , train_accuracy = evaluate_model(model, train_loader, device, CE_loss)
                
                text = ''
                for key , variable in zip(['val_loss','val_accuracy','val_accuracy_sample','train_loss',],
                                          [val_loss, val_accuracy, val_accuracy_sample,loss.item()]):
                    text += f'{key}: {variable:.4f}  '
                    summary[key].append(variable)


                for name in overlaps_step:
                    aligment['overlap'][name].append(overlaps_step[name][0])
                    aligment['difference'][name].append(overlaps_step[name][1])
                

                # Print norm of trainable parameters with name
                text2 = ''
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        param_norm = param.data.norm(2).item()
                        text2 += f'{name}: {param_norm:.4f}  '
                print(f'Step {global_step}/{tot_global_steps}  ' + text )#+ '  ' + text2)
            

            condition_intervention = n_interventions < tot_interventions and val_accuracy >= intervene_acc[n_interventions]
            if condition_intervention:
                print(f'Intervention at step {global_step} with val_accuracy {val_accuracy:.4f}')
                interpolation_initialization(model, alpha=0.1)
                n_interventions += 1


            global_step += 1
            optimizer.zero_grad()
            if closure is None:
                loss.backward()
                optimizer.step()
            else:
                optimizer.step(closure)
            # optimizer.step()
            
    

    print('Training completed.')
    print('Total training time (min): ', (time.time() - time_start)/60)
    print('Total training time (h): ', (time.time() - time_start)/3600)

    for key in summary:
        summary[key] = np.array(summary[key])
        print(f'{key} : {summary[key].shape}')

    for name in aligment['overlap']:
        aligment['overlap'][name] = np.array(aligment['overlap'][name])
        print(f"{name} : { aligment['overlap'][name].shape }")
    
    for name in aligment['difference']:
        aligment['difference'][name] = np.array(aligment['difference'][name])
        print(f"{name} : {aligment['difference'][name].shape}")


    save_data(summary,'summary',experiment_name='post_small_induction_head', params=params)
    save_data(aligment,'aligment',experiment_name='post_small_induction_head', params=params)

    # Save checkpoint of model at the end of training

    file_path , _, _ = make_data_paths('model_fin', experiment_name= 'post_small_induction_head', params=params,ext='pt') 
    print('Saving model checkpoint to ', file_path)
    torch.save(model.state_dict(), file_path)
    
    

if __name__ == "__main__":
    main()