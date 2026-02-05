import argparse
import torch
import torch.nn as nn
import numpy as np
import time

from sam.models import create_transformer
from sam.dataset import get_dataloader
from sam.evaluation import evaluate_model

from configurations import save_data



def main():
    parser = argparse.ArgumentParser(description="Training Attention-Only Transformer on Copying Task")

    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--d_eff', type=int, default=32, help='Effective dimension per head')
    parser.add_argument('--vocab_size', type=int, default=100, help='Vocabulary size')
    parser.add_argument('--seq_len', type=int, default=64, help='Sequence length')
    parser.add_argument('--n_heads', type=int, default=1, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--dataset_size', type=int, default=1000, help='Total dataset size')
    parser.add_argument('--train_fraction', type=float, default=0.8, help='Fraction of data for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--n_prints', type=int, default=50, help='Number of times to print during training.')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs.')
    parser.add_argument('--sigma', type=float, default=1.0, help='Variance for parameter initialization.')
    parser.add_argument('--fr_emb', type=str, default='False', help='Freeze embeddings during training?')
    parser.add_argument('--sparsity', type=float, default=0.0, help='Sparsity level for initializaition of parameters.')

    args = parser.parse_args()
    config = vars(args)

    print("Configuration:")
    print(config)
    for key in ['fr_emb']:
        config[key] = True if config[key] == 'True' else False

    # Create model in device
    model , device = create_transformer(config)
    print("Model created on device:", device)

    # Initialize embeddings like one-hot and freeze if specified
    # if config['fr_emb']:
    #     with torch.no_grad():
    #         model.embedding.embedding.weight.copy_(torch.eye(config['vocab_size'], config['d_model']))
    #     model.embedding.embedding.weight.requires_grad = False
    

    # Get dataloaders
    train_loader , val_loader = get_dataloader(config)

    # Define loss function and optimizer
    CE_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'],eps=1e-9,weight_decay=0.01)


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
        'train_accuracy': [],
        'val_accuracy': []
    }

    # Example: Iterate through the training dataloader
    time_start = time.time()
    for epoch in range(config['num_epochs']):

        for batch in train_loader:
            input = batch['input'].to(device) # (batch_size, seq_len)
            target = batch['target'].to(device) # (batch_size, )
            mask = batch['mask'].to(device) # (1, seq_len, seq_len)

            logits = model(input, mask) # (batch_size, vocab_size)
            loss = CE_loss(logits.view(-1, config['vocab_size']), target.view(-1))
            loss.backward()

            # Check for print condition to evaluate and print
            if global_step % print_every == 0:
                summary['step'].append(global_step)
                val_loss , val_accuracy = evaluate_model(model, val_loader, device, CE_loss)
                # train_loss , train_accuracy = evaluate_model(model, train_loader, device, CE_loss)
                
                text = ''
                for key , variable in zip(['val_loss','val_accuracy','train_loss',],
                                          [val_loss, val_accuracy, loss.item()]):
                    text += f'{key}: {variable:.4f}  '
                    summary[key].append(variable)
                print(f'Step {global_step}/{tot_global_steps}  ' + text)
                


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
    
    fix_params = { key : config[key] for key in ['d_model','d_eff','vocab_size','seq_len','batch_size','dataset_size','train_fraction','fr_emb']}
    variable_params = { key : config[key] for key in ['sigma','lr','n_heads','n_layers','dropout','sparsity']}

    params = {'fixed' : fix_params,
              'variable': variable_params}

    save_data(summary,'summary',experiment_name='copy_task', params=params)
    
    

if __name__ == "__main__":
    main()