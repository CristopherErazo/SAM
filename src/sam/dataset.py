from torch.utils.data import Dataset, DataLoader , random_split
import torch


################################################################
################################################################
############### DATASET FOR COPYING TASK #######################
################################################################
################################################################

def get_sample(V,L):
    """Generate a single sample for the copying task."""
    l = 0
    nsteps = 0
    while l < 2:
        input = torch.randint(0,V,(L,))
        index = (input == input[-1]).nonzero(as_tuple=True)[0]
        l = len(index)
        nsteps += 1
    target = input[index[-2]+1]

    # Return input.shape = (L,) and target.shape = () for compatibility with CrossEntropyLoss
    return input, target, nsteps

def get_sample_permut(V,L,p_error = 0.0):
    assert V >= L-1, "Vocabulary size must be greater than or equal to sequence length - 1 for permutation task."
    input = torch.randperm(V)[:L-1]
    # Get random location on the sequence 
    loc = torch.randint(0,L-3,(1,)).item()
    input = torch.cat([input,input[loc].unsqueeze(0)],dim=0) # shape (L,)
    target = input[loc+1] # shape () 
    # With probability p_error, replace target with a random token from the vocabulary (introduce noise)
    if torch.rand(1).item() < p_error:
        target = torch.randint(0,V,(1,))[0] # shape () 
    # Return input.shape = (L,) and target.shape = () for compatibility with CrossEntropyLoss
    return input, target, 0

def generate_copying_data(num_samples:int, seq_len:int, vocab_size:int,p_error:float=0.0) -> list[dict]:
    """Generate a dataset for the copying task."""
    data = []
    for _ in range(num_samples):
        input , target, nsteps = get_sample_permut(vocab_size, seq_len,p_error)
        data.append({
            'input' : input,
            'target' : target,
            'nsteps' : nsteps
        })
    return data

class InContextDataset(Dataset):
    """
    Dataset for the copying task. 
    Task: Given a sequence of random tokens, look at the last token,
    find the second last occurrence of that token in the sequence,
    and return the token that follows it.
    eg: input = [2,5,1,3,5,4,5] -> target = 4
    2nd last occurrence of 5 is at index 4, token after it is 4.
    """
    def __init__(self, data,seq_len:int) -> None:
        self.data = data
        self.L = seq_len
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self,idx:int):
        item = self.data[idx]
        # causal mask: lower triangular (allow attend to <= position)
        mask = torch.tril(torch.ones((self.L, self.L), dtype=torch.bool)).unsqueeze(0)  # (1, L, L) for multi-head attention
        return {
            "input": item['input'],
            "target": item['target'],
            "nsteps": item['nsteps'],
            "mask": mask
        }

def get_dataloader(config:dict) -> tuple[DataLoader,DataLoader]:
    """Generate train and test dataloaders for the copying task. 
    Args:
        config (dict): Configuration dictionary with keys:
            - dataset_size (int): Total dataset size
            - train_fraction (float): Fraction of data for training
            - seq_len (int): Length of each input sequence
            - vocab_size (int): Size of the vocabulary
            - batch_size (int): Batch size for dataloaders
            - p_error (float): Probability of introducing noise in the target for the permutation task
    Returns:
        tuple: (train_dataloader, val_dataloader)    
    """
    n = config['dataset_size']
    L = config['seq_len']
    V = config['vocab_size']
    p_error = config['p_error']

    train_size = int(config['train_fraction'] * n)
    val_size = n - train_size

    train_ds = generate_copying_data(train_size,L,V,p_error)
    val_ds = generate_copying_data(val_size,L,V,p_error=0.0)

  
    # Create InContextDataset instances
    train_dataset = InContextDataset(train_ds,L)
    val_dataset = InContextDataset(val_ds,L)

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True )
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False )
    return train_dataloader, val_dataloader

    

