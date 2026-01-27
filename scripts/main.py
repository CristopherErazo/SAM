import argparse
import torch
import torch.nn.functional as F
import numpy as np

from sam.models import init_teacher_student
from sam.optimizers import SAM_Optimizer
from sam.utils import generate_dataset

from configurations import save_data 


def main():
    parser = argparse.ArgumentParser(description="Training Single Index Model")

    parser.add_argument('--d', type=int, default=200, help='Input dimension')
    parser.add_argument('--tch_act', type=str, default='tanh', help='Activation function for teacher')
    parser.add_argument('--std_act', type=str, default='tanh', help='Activation function for student')
    parser.add_argument('--n_train', type=int, default=1000, help='Number of training samples')
    parser.add_argument('--n_test', type=int, default=1000, help='Number of test samples')
    parser.add_argument('--bs', type=int, default=100, help='Batch size for training')
    parser.add_argument('--opt', type=str, default='SGD', help='Type of optimizer: SGD, adam, or SAM')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--eps', type=float, default=0.0, help='Noise level for labels')
    parser.add_argument('--q', type=float, default=2.0, help='q-norm for SAM')
    parser.add_argument('--rho', type=float, default=0.1, help='Radius for SAM')
    parser.add_argument('--nprints', type=int, default=20, help='Number of prints during training')

    args = parser.parse_args()
    config = vars(args)

    # Extract config parameters
    bs = config['bs']
    

    # Set device and initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    device = torch.device(device)

    # Initialize teacher and student models
    teacher, student, w_teacher = init_teacher_student(config['d'], 
                                                       config['tch_act'],
                                                       config['std_act'],
                                                       device)
    
    if config['opt'] == 'SGD':
        optimizer = torch.optim.SGD(student.parameters(), lr=config['lr'])
        closure = None
    elif config['opt'] == 'adam':
        optimizer = torch.optim.Adam(student.parameters(), lr=config['lr'])
        closure = None  
    elif config['opt'] == 'SAM':
        optimizer =SAM_Optimizer(student.parameters(), lr=config['lr'], q=config['q'], rho=config['rho'])
        def closure():
            y_pred = student(x)
            loss = F.mse_loss(y_pred, y)
            return loss
    else:
        raise ValueError(f'Unknown optimizer type: {config["opt"]}')                                                      
                                                

    # Generate datasets for test and train
    x_train, y_train = generate_dataset(teacher, config['n_train'], config['d'], config['eps'], device)
    x_test, y_test = generate_dataset(teacher, config['n_test'], config['d'], config['eps'], device)

    # Dictionary to save training info
    n_steps = config['n_train'] // bs
    print_every = max(1, n_steps // config['nprints'])
    summary = {
        'step': [],
        'train_loss': [],
        'test_loss': [],
        'overlap': [], 
        'norm_student': []
    }


    
    for step in range(n_steps): # each step sample a batch from training set 
        idx = step * bs
        x = x_train[idx:idx+bs]
        y = y_train[idx:idx+bs]
        y_pred = student(x)
        loss = F.mse_loss(y, y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(closure);
 
        print_condition = (step % print_every == 0) or (step == n_steps - 1)
        if print_condition:
            # Compute generalization error
            with torch.no_grad():
                y_test_pred = student(x_test)
                test_loss = F.mse_loss(y_test, y_test_pred)
                # Compute overlap and norm
                w_student = torch.cat([p.view(-1) for p in student.parameters()])
                overlap = torch.dot(w_teacher, w_student) / (torch.norm(w_teacher) * torch.norm(w_student))
                norm_student = torch.norm(w_student)
            # Save to summary
            summary['step'].append(step)
            summary['train_loss'].append(loss.item())
            summary['test_loss'].append(test_loss.item())
            summary['overlap'].append(overlap.item())
            summary['norm_student'].append(norm_student.item())
            print(f'Step {step+1}/{n_steps}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Overlap: {overlap.item():.4f}, Norm Student: {norm_student.item():.4f}')


    for key in summary.keys():
        summary[key] = np.array(summary[key])
        print(f"{key}: {summary[key].shape}")

    # Save data
    
    fix_params = { key : config[key] for key in ['d','tch_act','std_act','n_train','n_test']}
    variable_params = { key : config[key] for key in ['bs','opt','lr','q','rho']}

    params = {'fixed' : fix_params,
              'variable': variable_params}
    
    save_data(summary,'first_tests',experiment_name='compare_optimizers', params=params)


if __name__ == "__main__":
    main()