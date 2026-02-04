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
    return input, target, nsteps


def generate_copying_data(num_samples:int, seq_len:int, vocab_size:int):
    """Generate a dataset for the copying task."""
    data = []
    for _ in range(num_samples):
        input , target, nsteps = get_sample(vocab_size, seq_len)
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
    """Generate train and test dataloaders for the copying task."""
    n = config['dataset_size']
    L = config['seq_len']
    V = config['vocab_size']
    dataset = generate_copying_data(n,L,V)

    # Split dataset into train and validation sets
    train_size = int(config['train_fraction'] * n)
    val_size = n - train_size
    train_ds , val_ds = random_split(dataset,[train_size,val_size])

    # Create InContextDataset instances
    train_dataset = InContextDataset(train_ds,L)
    val_dataset = InContextDataset(val_ds,L)

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True )
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False )
    return train_dataloader, val_dataloader

    

