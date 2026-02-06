import argparse
import torch
import torch.nn as nn
import numpy as np
import time

from sam.models import create_induction_head , planted_initialization, interpolation_initialization
from sam.dataset import get_dataloader
from sam.evaluation import evaluate_model

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

    args = parser.parse_args()
    config = vars(args)

    print("Configuration:")
    print(config)
    # for key in ['fr_emb']:
    #     config[key] = True if config[key] == 'True' else False

    # Create model in device
    model , device = create_induction_head(config)
    planted_initialization(model,betas=(config['beta_1'], config['beta_2'], config['beta_out']))
    interpolation_initialization(model, alpha=config['alpha'])

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
            print(name)


    # Get dataloaders
    train_loader , val_loader = get_dataloader(config)

    # Define loss function and optimizer
    CE_loss = nn.CrossEntropyLoss(label_smoothing=0.15)
    optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'],eps=1e-9,weight_decay=0.03)


    # Training loop parameters
    tot_global_steps = config['num_epochs']*len(train_loader)
    nprints = config['n_prints']
    print_every = max(1,tot_global_steps // nprints)
    global_step = 0

    intervene_acc = [ 0.9] # List of val_accuracy_sample thresholds for intervention
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

    # Example: Iterate through the training dataloader
    time_start = time.time()
    for epoch in range(config['num_epochs']):

        for batch in train_loader:
            input = batch['input'].to(device) # (batch_size, seq_len)
            target = batch['target'].to(device) # (batch_size, )
            # mask = batch['mask'].to(device) # (1, seq_len, seq_len)

            logits = model(input) # (batch_size, vocab_size)
            loss = CE_loss(logits.view(-1, config['vocab_size']), target.view(-1))
            loss.backward()

            # Check for print condition to evaluate and print
            if global_step % print_every == 0:
                summary['step'].append(global_step)
                val_loss , val_accuracy , val_accuracy_sample = evaluate_model(model, val_loader, device, CE_loss)
                # train_loss , train_accuracy = evaluate_model(model, train_loader, device, CE_loss)
                
                text = ''
                for key , variable in zip(['val_loss','val_accuracy','val_accuracy_sample','train_loss',],
                                          [val_loss, val_accuracy, val_accuracy_sample,loss.item()]):
                    text += f'{key}: {variable:.4f}  '
                    summary[key].append(variable)
                

                # Print norm of trainable parameters with name
                text2 = ''
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        param_norm = param.data.norm(2).item()
                        text2 += f'{name}: {param_norm:.4f}  '
                print(f'Step {global_step}/{tot_global_steps}  ' + text + '  ' + text2)
            

            condition_intervention = n_interventions < tot_interventions and val_accuracy >= intervene_acc[n_interventions]
            if condition_intervention:
                print(f'Intervention at step {global_step} with val_accuracy {val_accuracy:.4f}')
                interpolation_initialization(model, alpha=0.1)
                n_interventions += 1


            global_step += 1
            optimizer.step()
            optimizer.zero_grad()
    

    print('Training completed.')
    print('Total training time (min): ', (time.time() - time_start)/60)
    print('Total training time (h): ', (time.time() - time_start)/3600)

    for key in summary:
        summary[key] = np.array(summary[key])
        print(f'{key} : {summary[key].shape}')

    # Save data
    fix_params = { key : config[key] for key in ['vocab_size','seq_len']}
    variable_params = { key : config[key] for key in ['lr','alpha']}

    params = {'fixed' : fix_params,
              'variable': variable_params}
    save_data(summary,'summary',experiment_name='induction_head', params=params)

    # Save checkpoint of model at the end of training

    file_path , _, _ = make_data_paths('model', experiment_name= 'induction_head', params=params,ext='pt') 
    print('Saving model checkpoint to ', file_path)
    torch.save(model.state_dict(), file_path)
    
    

if __name__ == "__main__":
    main()