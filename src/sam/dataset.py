from torch.utils.data import Dataset, DataLoader , random_split
import torch
import math
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from line_profiler import profile

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


def get_triggers(fix_trig:bool,trig_type:str,K:int,P_t:torch.Tensor):
    vocab_size = P_t.shape[0]
    trigger_set = None
    if fix_trig:
        if trig_type == 'freq':
            trigger_set = [k for k in range(K)]
        elif trig_type == 'rare':
            trigger_set = [vocab_size - 1 - k for k in range(K)] 
        elif trig_type == 'rand':
            trigger_set = torch.multinomial(P_t, num_samples=K, replacement=False).tolist()
        print(f"Using fixed {trig_type} trigger set")
        print(f"Length of trigger set: {len(trigger_set)}, Trigger set: {trigger_set}")
    else:
        print(f"Using random trigger set at each sequence")
    return torch.tensor(trigger_set)

def get_sample_dual_task(L:int, K: int ,P_b:torch.Tensor,P_u:torch.Tensor,P_o:torch.Tensor,P_t:torch.Tensor, trigger_set:list=None):
    """
    Generate a single sample for the dual task.
    Args:
        K (int): Number of trigger tokens in the sequence
        L (int): Sequence length
        P_b (torch.tensor): Bigram probability matrix of shape (V,V) P_b[i,j]  = P(token j | token i)
        P_u (torch.tensor): Unigram probability vector of shape (V,) P_u[i] = P(token i)
        P_o (torch.tensor): Probability for output tokens of a trigger token, shape (V,V) P_o[i,j] = P(output token j | trigger token i)
        P_t (torch.tensor): Probability for trigger tokens, shape (V,) P_t[i] = P(trigger token i)
        trigger_set (list): List of trigger tokens to use in the sequence. If None, trigger tokens will be sampled from P_t.
        
    Returns:
        input (torch.tensor): Input sequence of shape (L,)
        target (torch.tensor): Target token of shape ()

    First we sample a list of output tokens for each trigger token in the sequence according to P_o.
    Then the first token in the sequence is sampled from P_u, and each subsequent token is sampled 
    depending on the previous token. 

    - If the previour token is a trigger token, the next token is deterministically the output token corresponding to that trigger token.
    - If the previous token is not a trigger token, the next token is sampled from P_b depending on the previous token.
    """
    if trigger_set is None:
        # Sample K trigger tokens from P_t without replacement
        trigger_set = torch.multinomial(P_t, num_samples=K, replacement=False).tolist()
    else:
        assert len(trigger_set) == K, f"Length of trigger_set must be equal to K. Instead got trigger_set of length , {len(trigger_set)}, and K = , {K}"

    # Sample an output token for each trigger token in the sequence according to P_o
    output_set = [ torch.multinomial(P_o[trigger_token], num_samples=1).item() for trigger_token in trigger_set]
    # print("Output set for trigger tokens: ", output_set)
    sequence = torch.zeros(L+1, dtype=torch.long)
    # Sample the first token from P_u
    sequence[0] = torch.multinomial(P_u, num_samples=1).item()

    cnts = {} # dictionary to count occurrences of each token in the sequence, initialized with the first token
    counts = []
    is_trigg = []

    for i in range(1,L):
        prev_token = sequence[i-1].item()
        cnts[prev_token] = cnts.get(prev_token,0) + 1
        counts.append(cnts[prev_token])
        if prev_token in trigger_set:
            # If the previous token is a trigger token, the next token is deterministically the output token corresponding to that trigger token
            trigger_index = trigger_set.index(prev_token)
            sequence[i] = output_set[trigger_index]
            is_trigg.append(1)
        else:
            # If the previous token is not a trigger token, the next token is sampled from P_b depending on the previous token
            sequence[i] = torch.multinomial(P_b[prev_token], num_samples=1).item()
            is_trigg.append(0)

    prev_token = sequence[L].item()
    cnts[prev_token] = cnts.get(prev_token,0) + 1
    counts.append(cnts[prev_token])
    is_trigg.append(1 if prev_token in trigger_set else 0)

    return sequence , torch.tensor(trigger_set.copy()) , torch.tensor(output_set) , torch.tensor(counts), torch.tensor(is_trigg)

def generate_dual_task_batch(batch_size:int, seq_len:int,K:int, P_b:torch.Tensor,P_u:torch.Tensor,P_o:torch.Tensor,P_t:torch.Tensor,trigger_set: torch.Tensor | None = None,) -> dict[str, torch.Tensor]:
    """Generate a batch of data for the dual task."""
    sequences = []
    trigger_sets = []
    output_sets = []
    counts_list = []
    is_trigg_list = []
    for _ in range(batch_size):
        sequence , trigg_set  , output_set , counts , is_trigg = get_sample_dual_task(seq_len,K,P_b,P_u,P_o,P_t,trigger_set=trigger_set)
        sequences.append(sequence)
        trigger_sets.append(trigg_set)
        output_sets.append(output_set)
        counts_list.append(counts)
        is_trigg_list.append(is_trigg)

    mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool)) # (1, L, L) for multi-head attention
    batch_mask = mask.unsqueeze(0).expand(batch_size, -1, -1) # (batch_size, seq_len, seq_len)
    return {
        'sequence' : torch.stack(sequences), # shape (batch_size, seq_len)
        'trigger_set' : torch.stack(trigger_sets), # shape (batch_size, K)
        'output_set' : torch.stack(output_sets), # shape (batch_size, K)
        'counts' : torch.stack(counts_list), # shape (batch_size, seq_len)
        'mask' : batch_mask, # shape (batch_size, seq_len, seq_len)
        'is_trigg' : torch.stack(is_trigg_list) # shape (batch_size, seq_len)


    }

@profile
def generate_dual_task_batch_fast(
    num_samples: int,
    L: int,
    K: int,
    P_b: torch.Tensor,
    P_u: torch.Tensor,
    P_o: torch.Tensor,
    P_t: torch.Tensor,
    trigger_set: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
):
    """
    Fully vectorized batch generator for the dual task.

    Returns:
        dict with:
            sequence     : (B, L)
            trigger_set  : (B, K)
            output_set   : (B, K)
            counts       : (B, L)
            is_trigg     : (B, L)
    """

    B = num_samples
    V = P_u.shape[0]

    # տեղափոխ device
    P_b = P_b.to(device)
    P_u = P_u.to(device)
    P_o = P_o.to(device)
    P_t = P_t.to(device)

    # print('devices:')
    # print(f'{P_b.device=}, {P_t.device=},{P_o.device=},{P_u.device=   }')

    # ---- sample trigger sets ----
    if trigger_set is None:
        trigger_sets = torch.stack([
            torch.multinomial(P_t, K, replacement=False)
            for _ in range(B)
        ], dim=0)
    else:
        trigger_set = torch.tensor(trigger_set, device=device)
        assert trigger_set.numel() == K
        trigger_sets = trigger_set.unsqueeze(0).repeat(B, 1)

    # ---- sample output tokens ----
    # shape: (B, K)
    output_sets = torch.zeros(B, K, dtype=torch.long, device=device)
    for k in range(K):
        output_sets[:, k] = torch.multinomial(
            P_o[trigger_sets[:, k]], 1
        ).squeeze(-1)

    # ---- build trigger mask ----
    trigger_mask = torch.zeros(B, V, dtype=torch.bool, device=device)
    trigger_mask.scatter_(1, trigger_sets, True)

    # ---- build mapping trigger -> output ----
    mapping = torch.full((B, V), -1, dtype=torch.long, device=device)
    mapping.scatter_(1, trigger_sets, output_sets)

    # ---- initialize sequence ----
    sequence = torch.zeros(B, L+1, dtype=torch.long, device=device)
    sequence[:, 0] = torch.multinomial(P_u, B, replacement=True)

    # ---- outputs ----
    is_trigg = torch.zeros(B, L+1, dtype=torch.long, device=device)
    counts = torch.zeros(B, L+1, dtype=torch.long, device=device)

    # for counting occurrences
    token_counts = torch.zeros(B, V, dtype=torch.long, device=device)

    # ---- main loop over sequence length ----
    for t in range(L+1):
        current = sequence[:, t]

        # update counts
        token_counts.scatter_add_(
            1,
            current.unsqueeze(1),
            torch.ones(B, 1, dtype=torch.long, device=device),
        )
        counts[:, t] = token_counts[
            torch.arange(B, device=device), current
        ]

        # mark trigger
        is_trigg[:, t] = trigger_mask[
            torch.arange(B, device=device), current
        ].long()

        if t == L:
            break

        # ---- next token ----
        prev = current

        # sample bigram transitions
        next_tokens = torch.multinomial(
            P_b[prev], 1
        ).squeeze(-1)

        # override if trigger
        mapped = mapping[
            torch.arange(B, device=device), prev
        ]

        trigger_positions = mapped != -1
        next_tokens[trigger_positions] = mapped[trigger_positions]

        sequence[:, t + 1] = next_tokens

    mask = torch.tril(torch.ones((L, L), dtype=torch.bool)) # (1, L, L) for multi-head attention
    batch_mask = mask.unsqueeze(0).expand(B, -1, -1) # (batch_size, seq_len, seq_len)
    
    return {
        "sequence": sequence, # shape (B, L+1)
        "trigger_set": trigger_sets, # shape (B, K)
        "output_set": output_sets, # shape (B, K)
        "counts": counts[:,:L],    # shape (B, L)
        "is_trigg": is_trigg[:,:L], # shape (B, L)
        "mask": batch_mask # shape (B, L, L)
    }


def get_spike_vector(vocab_size:int , P_u:torch.Tensor) -> torch.Tensor:    
    u = torch.empty_like(P_u)
    s_pos = 0.0
    s_neg = 0.0
    for i in range(vocab_size):
        if s_pos <= s_neg:
            u[i] = 1.0
            s_pos += P_u[i].item()
        else:
            u[i] = -1.0
            s_neg += P_u[i].item()
    return u


def get_distributions(vocab_size:int,b_type:str='dirichlet',alpha:float=1.0,device:str='cpu',u_type:str=None, beta:float=None) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    """Get probability distributions P_b, P_u, P_o and P_t for the dual task data generation.

    Args:
        - vocab_size (int): Size of the vocabulary
        - b_type (str): Type of bigram distribution to generate. Options are 'dirichlet' and 'spiked'. 'dirichlet' samples the bigram distribution from a dirichlet distribution with concentration parameter alpha. 'spiked' generates a bigram distribution that is correlated with the unigram distribution, with more frequent tokens having higher probabilities of being followed by other tokens, using the formula P_b[i,j] = P_u[j] * (1 - beta * u[i] * u[j]) where u[i] is 1 for more frequent tokens and -1 for less frequent tokens.
        - alpha (float): Concentration parameter for the Dirichlet distribution used to generate the big
        unigram distribution P_u if b_type is 'dirichlet', or exponent for the zipf distribution if u_type is 'zipf' and b_type is 'spiked'.
        - u_type (str): Type of unigram distribution to generate if b_type is 'spiked'. Options are 'dirichlet' and 'zipf'. If 'dirichlet', the unigram distribution is sampled from a dirichlet distribution with concentration parameter alpha. If 'zipf', the unigram distribution is generated using a zipf distribution with exponent alpha.
        - beta (float): Concentration parameter for the Dirichlet distribution used to generate the bigram distribution P_b from the unigram distribution P_u using the formula P_b[i,j] = P_u[j] * (1 - beta * u[i] * u[j]) where u[i] is 1 for more frequent tokens and -1 for less frequent tokens. This creates a bigram distribution that is correlated with the unigram distribution, with more frequent tokens having higher probabilities of being followed by other tokens. Only used if b_type is 'spiked'.

    Returns:
        - P_b (torch.tensor): Bigram probability matrix of shape (V,V) P_b[i,j]  = P(token j | token i)
        - P_u (torch.tensor): Unigram probability vector of shape (V,) P_u[i] = P(token i)
        - P_o (torch.tensor): Probability for output tokens of a trigger token, shape (V,V) P_o[i,j] = P(output token j | trigger token i)
        - P_t (torch.tensor): Probability for trigger tokens, shape (V,) P_t[i] = P(trigger token i)
    """

    if b_type == 'dirichlet': 
        print("Sampling bigram distribution from a Dirichlet distribution with concentration parameter alpha = ", alpha)
        # Sample bigram distribution from a dirichlet distribution with concentration parameter alpha
        concentration = torch.ones((vocab_size, vocab_size)) * alpha
        P_b = torch.distributions.dirichlet.Dirichlet(concentration).sample()
        # Compute unigram distribution as the stationary distribution of P_b
        eigenvalues, eigenvectors = torch.linalg.eig(P_b.T)
        P_u = eigenvectors[:, torch.isclose(eigenvalues.real, torch.tensor(1.0))].real.squeeze()
        P_u /= P_u.sum()  # Normalize
        # Sort P_u and P_b in descending order of P_u for better interpretability (most frequent tokens first)
        sorted_indices = torch.argsort(P_u, descending=True)
        P_u = P_u[sorted_indices]
        P_b = P_b[sorted_indices][:, sorted_indices]

    elif b_type == 'spiked':
        print("Generating spiked bigram distribution with alpha = ", alpha, " and beta = ", beta)
        assert u_type is not None and beta is not None, "u_type and beta must be specified for spiked bigram distribution."
        if u_type == 'dirichlet':
            print("Sampling unigram distribution from a Dirichlet distribution with concentration parameter alpha = ", alpha)
            P_u = torch.distributions.dirichlet.Dirichlet(torch.ones(vocab_size) * alpha).sample()
        elif u_type == 'zipf':
            print("Generating unigram distribution from a Zipf distribution with exponent alpha = ", alpha)
            ranks = torch.arange(1, vocab_size + 1)
            P_u = 1.0 / ranks**alpha
            P_u /= P_u.sum() # Normalize to get a valid probability distribution
        else:
            raise ValueError("Invalid u_type. Options are 'dirichlet' and 'zipf'.")
        
        P_u = P_u.sort(descending=True).values # Sort unigram distribution in descending order for better interpretability (most frequent tokens first)
        u = get_spike_vector(vocab_size, P_u)
        P_b = P_u[None,:] * (1 - beta * u[:,None] * u[None,:]) # Generate bigram distribution P_b from unigram distribution P_u using the formula P_b[i,j] = P_u[j] * (1 - beta * u[i] * u[j]) where u[i] is 1 for more frequent tokens and -1 for less frequent tokens. This creates a bigram distribution that is correlated with the unigram distribution, with more frequent tokens having higher probabilities of being followed by other tokens.
        

    assert torch.all(P_b >= 0), "P_b contains negative probabilities."
    assert torch.allclose(P_b.sum(dim=-1), torch.ones(vocab_size),rtol=0.0001), "Rows of P_b do not sum to 1."
    assert torch.all(P_u >= 0), "P_u contains negative probabilities."
    assert torch.isclose(P_u.sum(), torch.tensor(1.0)), "P_u does not sum to 1."    
    # Renormalize
    P_b /= P_b.sum(dim=-1, keepdim=True)    


    P_t = P_u.clone() # Trigger distribution is the same as unigram distribution
    P_o = torch.ones((vocab_size,vocab_size)) / (vocab_size-1) # Uniform distribution over outputs given triggers apart from the trigger token itself
    P_o.fill_diagonal_(0) # Set diagonal to 0 since output cannot be the same as the trigger token

    return P_b.to(device), P_u.to(device), P_o.to(device), P_t.to(device)
    

def compute_entropies_and_dkl(P_b:torch.Tensor,P_u:torch.Tensor):
    vocab_size = P_u.shape[0]
    # Average dkl between bigram distribution and uniform 1/V
    kl_Pb_uniform = (P_b * (torch.log(P_b + 1e-10) - math.log(1.0/vocab_size + 1e-10))).sum(dim=-1).mean().item()
    
    # dkl between unigram and uniform 1/V
    kl_Pu_uniform = (P_u * (torch.log(P_u + 1e-10) - math.log(1.0/vocab_size + 1e-10))).sum().item()
   
    # Average entropy of bigram distribution
    entropy_Pb = -(P_b * torch.log(P_b + 1e-10)).sum(dim=-1).mean().item()
    entropy_Pu = -(P_u * torch.log(P_u + 1e-10)).sum().item()
    max_entropy = math.log(vocab_size)
    return kl_Pb_uniform, kl_Pu_uniform, entropy_Pb, entropy_Pu, max_entropy
    

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
            "sequence": item['sequence'],
            "trigger_set": item['trigger_set'],
            "output_set": item['output_set'],
            "counts": item['counts'],
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
    V = vocab_size
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



