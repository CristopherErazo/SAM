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



def get_sample_dual_task(L:int,P_b:torch.Tensor,P_u:torch.Tensor,P_o:torch.Tensor,trigger_set:list):
    """
    Generate a single sample for the dual task.
    Args:
        L (int): Sequence length
        P_b (torch.tensor): Bigram probability matrix of shape (V,V) P_b[i,j]  = P(token j | token i)
        P_u (torch.tensor): Unigram probability vector of shape (V,) P_u[i] = P(token i)
        P_o (torch.tensor): Probability for output tokens of a trigger token, shape (V,V) P_o[i,j] = P(output token j | trigger token i)
        trigger_set (list): Set of trigger tokens for the dual task
        
    Returns:
        input (torch.tensor): Input sequence of shape (L,)
        target (torch.tensor): Target token of shape ()

    First we sample a list of output tokens for each trigger token in the sequence according to P_o.
    Then the first token in the sequence is sampled from P_u, and each subsequent token is sampled 
    depending on the previous token. 

    - If the previour token is a trigger token, the next token is deterministically the output token corresponding to that trigger token.
    - If the previous token is not a trigger token, the next token is sampled from P_b depending on the previous token.
    """

    # Sample an output token for each trigger token in the sequence according to P_o
    output_set = [ torch.multinomial(P_o[trigger_token], num_samples=1).item() for trigger_token in trigger_set]
    # print("Output set for trigger tokens: ", output_set)
    sequence = torch.zeros(L+1, dtype=torch.long)
    # Sample the first token from P_u
    sequence[0] = torch.multinomial(P_u, num_samples=1).item()
    for i in range(1,L):
        prev_token = sequence[i-1].item()
        if prev_token in trigger_set:
            # If the previous token is a trigger token, the next token is deterministically the output token corresponding to that trigger token
            trigger_index = trigger_set.index(prev_token)
            sequence[i] = output_set[trigger_index]
        else:
            # If the previous token is not a trigger token, the next token is sampled from P_b depending on the previous token
            sequence[i] = torch.multinomial(P_b[prev_token], num_samples=1).item()

    input = sequence[:-1] # shape (L,)
    target = sequence[1:] # shape (L,) target is the next token for each position in the input sequence
    return input, target , output_set


def generate_dual_task_data(num_samples:int, seq_len:int,P_b:torch.Tensor,P_u:torch.Tensor,P_o:torch.Tensor,trigger_set:list) -> list[dict]:
    """Generate a dataset for the copying task."""
    data = []
    for _ in range(num_samples):
        input , target  , output_set = get_sample_dual_task(seq_len,P_b,P_u,P_o,trigger_set)
        data.append({
            'input' : input,
            'target' : target,
            'output_set' : output_set
        })
    return data


def get_distributions(vocab_size:int,mode:str='uniform',beta:float=1.0) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    """Get probability distributions P_b, P_u, P_o and P_t for the dual task data generation.
    Args:
        vocab_size (int): Size of the vocabulary
        mode (str): Type of distribution to generate. Options are 'uniform' and 'random'.
    Returns:
        P_b (torch.tensor): Bigram probability matrix of shape (V,V) P_b[i,j]  = P(token j | token i)
        P_u (torch.tensor): Unigram probability vector of shape (V,) P_u[i] = P(token i)
        P_o (torch.tensor): Probability for output tokens of a trigger token, shape (V,V) P_o[i,j] = P(output token j | trigger token i)
        P_t (torch.tensor): Probability for trigger tokens, shape (V,) P_t[i] = P(trigger token i)
    """
    if mode == 'uniform':
        P_b = torch.ones((vocab_size, vocab_size)) / vocab_size # Uniform distribution over next tokens
        P_u = torch.ones(vocab_size) / vocab_size # Uniform distribution over first token
        P_o = torch.ones((vocab_size, vocab_size)) 
        # Mask out the diagonal to avoid repeating the same trigger token
        P_o = P_o.masked_fill(torch.eye(vocab_size, dtype=torch.bool), float('-inf'))
        P_o = P_o.softmax(dim=-1) # Uniform distribution over output tokens for each
        P_t = torch.ones(vocab_size) / vocab_size # Uniform distribution over trigger tokens
    elif mode == 'random':
        P_b = torch.randn((vocab_size, vocab_size)) * beta
        P_b = torch.softmax(P_b, dim=-1) # Normalize to get probabilities
        P_u = torch.randn(vocab_size) * beta
        P_u = torch.softmax(P_u, dim=-1) # Normalize to get probabilities
        P_o = torch.randn((vocab_size, vocab_size)) * beta
        P_o = P_o.masked_fill(torch.eye(vocab_size, dtype=torch.bool), float('-inf')) # Mask out the diagonal to avoid repeating the same trigger token
        P_o = torch.softmax(P_o, dim=-1) # Normalize to get probabilities
        P_t = torch.randn(vocab_size) * beta
        P_t = torch.softmax(P_t, dim=-1) # Normalize to get probabilities
    else:
        raise ValueError("Invalid mode. Options are 'uniform' and 'random'.")
    return P_b, P_u, P_o, P_t

    

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
        mask = torch.tril(torch.ones((self.L, self.L), dtype=torch.bool)) # (1, L, L) for multi-head attention
        return {
            "input": item['input'],
            "target": item['target'],
            "output_set": item['output_set'],
            "mask": mask
        }


def get_dataloader_dual_task(config:dict,P_b:torch.tensor,P_u:torch.tensor,P_o:torch.tensor,trigger_set:list) -> tuple[DataLoader,DataLoader]:
    """Generate train and test dataloaders for the dual task. 
    Args:
        config (dict): Configuration dictionary with keys:
            - dataset_size (int): Total dataset size
            - train_fraction (float): Fraction of data for training
            - seq_len (int): Length of each input sequence
            - batch_size (int): Batch size for dataloaders
    Returns:
        tuple: (train_dataloader, val_dataloader)    
    """
    n = config['dataset_size']
    L = config['seq_len']

    train_size = int(config['train_fraction'] * n)
    val_size = n - train_size

    train_ds = generate_dual_task_data(train_size,L,P_b,P_u,P_o,trigger_set)
    val_ds = generate_dual_task_data(val_size,L,P_b,P_u,P_o,trigger_set)

  
    # Create InContextDataset instances
    train_dataset = InContextDataset(train_ds,L)
    val_dataset = InContextDataset(val_ds,L)

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True )
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False )
    return train_dataloader, val_dataloader



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

    

