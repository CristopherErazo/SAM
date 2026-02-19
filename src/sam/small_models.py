from torch import nn
import torch
import math

if False:
    def full_output(self, input: torch.Tensor) -> torch.Tensor:
        """ X: (batch_size, seq_len) list of token ids """
        with torch.no_grad():
            output = {}
            # Input embedding + positional encoding
            E = self.embedding(input)  # (batch_size, seq_len, vocab_size)
            P = self.positions(torch.arange(self.seq_len, device=input.device)) # (seq_len, seq_len)
            output['E'] = E.detach().cpu().numpy()
            output['P'] = P.detach().cpu().numpy()
            # FIRST LAYER
            # Compute queries and keys
            Q1 = self.WQ1(P)  # ( seq_len, seq_len)
            K1 = self.WK1(P)  # ( seq_len, seq_len)
            V1 = self.WV1(E)  # (batch_size, seq_len, vocab_size)
            output['Q1'] = Q1.detach().cpu().numpy()
            output['K1'] = K1.detach().cpu().numpy()
            output['V1'] = V1.detach().cpu().numpy()

            # Compute attention scores and attention weights with masking
            S1 = math.sqrt(self.seq_len)*self.beta_1*torch.matmul(Q1, K1.transpose(-2, -1)) * (self.d_model/self.seq_len)**2  # ( seq_len, seq_len)
            S1 = S1.masked_fill(self.mask1.to(S1.device), float('-inf'))
            output['S1'] = S1.detach().cpu().numpy()
            A1 = S1.softmax(dim=-1).unsqueeze(0).expand(input.size(0), -1, -1)   # (batch_size, seq_len, seq_len)
            # Zero the first attention weight
            A1[:,0,:] = 0.0
            output['A1'] = A1.detach().cpu().numpy()
            A1 = self.dropout(A1)
            Y1 = torch.matmul(A1, V1)  # (batch_size, seq_len, vocab_size)
            output['Y1'] = Y1.detach().cpu().numpy()
            Z1 = E + Y1  # (batch_size, seq_len, vocab_size)
            output['Z1'] = Z1.detach().cpu().numpy()
            # SECOND LAYER
            # Compute queries and keys
            q2 = self.WQ2(Z1[:,-1]).unsqueeze(1)  # (batch_size, 1 , vocab_size)
            K2 = self.WK2(Z1)  # (batch_size, seq_len, vocab_size)
            
            output['q2'] = q2.detach().cpu().numpy()
            output['K2'] = K2.detach().cpu().numpy()

            # Compute attention scores and attention weights with masking
            S2 = math.sqrt(self.seq_len)*self.beta_2*torch.matmul(q2, K2.transpose(-2, -1)) * (self.d_model/self.vocab_size)**2 # (batch_size, 1, seq_len)
            # Mask out last position
            S2[:,:,-1] = float('-inf')
            output['S2'] = S2.detach().cpu().numpy()
            A2 = S2.softmax(dim=-1)  # (batch_size, 1, seq_len)
            output['A2'] = A2.detach().cpu().numpy()
            A2 = self.dropout(A2)

            Y2 = torch.matmul(A2, V2)  # (batch_size, 1, vocab_size)
            output['Y2'] = Y2.detach().cpu().numpy()

            # Compute logit outputs as projection onto embeddings
            logits = math.sqrt(self.vocab_size)*self.beta_out*torch.matmul(Y2, self.embedding.weight.t()) * (self.d_model/self.vocab_size)**1.5 # (batch_size, 1, vocab_size)
            output['logits'] = logits.detach().cpu().numpy()
        return output

class InductionHeadAttention(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            seq_len: int,
            dropout: float = 0.0,
            ) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.d_model = vocab_size + seq_len
        self.d_eff = self.seq_len
        self.mask1 = torch.tril(torch.ones((seq_len,seq_len)),diagonal=0) == 0
        self.mask2 = torch.tensor([[[False]*(self.seq_len-2) + [True,True]]])

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        self.positions = nn.Embedding(seq_len,seq_len)

        # First layer weights
        self.WQ1 = nn.Linear(self.seq_len, self.seq_len, bias=False)
        self.WK1 = nn.Linear(self.seq_len, self.seq_len, bias=False)
        self.WV1 = nn.Linear(self.vocab_size, self.vocab_size, bias=False)

        # Second layer weights
        self.WQ2 = nn.Linear(self.vocab_size, self.vocab_size, bias=False)
        self.WK2 = nn.Linear(self.vocab_size, self.vocab_size, bias=False)

        # Dropout
        self.dropout=nn.Dropout(dropout)

        # Scalar Parameters
        self.beta_1 = nn.Parameter(torch.tensor(1.0))
        self.beta_2 = nn.Parameter(torch.tensor(1.0))
        self.beta_out = nn.Parameter(torch.tensor(1.0))

    def full_output(self, input: torch.Tensor) -> torch.Tensor:
        """ X: (batch_size, seq_len) list of token ids """
        output = {}
        # Input embedding + positional encoding
        E = self.embedding(input)  # (batch_size, seq_len, vocab_size)
        P = self.positions(torch.arange(self.seq_len, device=input.device)) # (seq_len, seq_len)

        # FIRST LAYER
        # Compute queries and keys
        Q1 = self.WQ1(P)  # ( seq_len, seq_len)
        K1 = self.WK1(P)  # ( seq_len, seq_len)
        V1 = self.WV1(E)  # (batch_size, seq_len, vocab_size)
        output['E'] = E[0]
        output['P'] = P
        output['Q1'] = Q1
        output['K1'] = K1
        output['V1'] = V1[0]

        # Compute attention scores and attention weights with masking
        S1 = self.beta_1*torch.matmul(Q1, K1.transpose(-2, -1)) /math.sqrt(self.seq_len)  # ( seq_len, seq_len)
        output['S1'] = S1
        S1 = S1.masked_fill(self.mask1.to(S1.device), float('-inf'))
        A1 = S1.softmax(dim=-1).unsqueeze(0).expand(input.size(0), -1, -1).clone()   # (batch_size, seq_len, seq_len)
        # Zero the first attention weight
        A1[:,0,:] = 0.0
        output['A1'] = A1[0]
        A1 = self.dropout(A1)
        Y1 = torch.matmul(A1, V1)  # (batch_size, seq_len, vocab_size)
        Z1 = E + Y1  # (batch_size, seq_len, vocab_size)
        output['Y1'] = Y1[0]
        output['Z1'] = Z1[0]

        # SECOND LAYER
        # Compute queries and keys
        q2 = self.WQ2(Z1[:,-1]).unsqueeze(1)  # (batch_size, 1 , vocab_size)
        K2 = self.WK2(Z1)  # (batch_size, seq_len, vocab_size)
        output['q2'] = q2[0,0,:]
        output['K2'] = K2[0]
        

        # Compute attention scores and attention weights with masking
        S2 = self.beta_2*torch.matmul(q2, K2.transpose(-2, -1)) / math.sqrt(self.vocab_size) # (batch_size, 1, seq_len)
        output['S2'] = S2[0,0,:]
        # Mask out last position
        S2 = S2.masked_fill(self.mask2.to(S2.device), float('-inf'))
        A2 = S2.softmax(dim=-1)  # (batch_size, 1, seq_len)
        output['A2'] = A2[0,0,:]
        A2 = self.dropout(A2)
        Y2 = torch.matmul(A2, E)  # (batch_size, 1, vocab_size)
        output['Y2'] = Y2[0,0,:]

        # Compute logit outputs
        logits = self.beta_out * Y2  # (batch_size, 1, vocab_size)
        output['logits'] = logits[0,0,:]
        output['probs'] = torch.softmax(logits, dim=-1)[0,0,:]

        for key in output:
            output[key] = output[key].detach().cpu().numpy()
        return output   



    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ X: (batch_size, seq_len) list of token ids """

        # Input embedding + positional encoding
        E = self.embedding(input)  # (batch_size, seq_len, vocab_size)
        P = self.positions(torch.arange(self.seq_len, device=input.device)) # (seq_len, seq_len)

        # FIRST LAYER
        # Compute queries and keys
        Q1 = self.WQ1(P)  # ( seq_len, seq_len)
        K1 = self.WK1(P)  # ( seq_len, seq_len)
        V1 = self.WV1(E)  # (batch_size, seq_len, vocab_size)

        # Compute attention scores and attention weights with masking
        S1 = self.beta_1*torch.matmul(Q1, K1.transpose(-2, -1)) /math.sqrt(self.seq_len)  # ( seq_len, seq_len)
        S1 = S1.masked_fill(self.mask1.to(S1.device), float('-inf'))
        A1 = S1.softmax(dim=-1).unsqueeze(0).expand(input.size(0), -1, -1).clone()   # (batch_size, seq_len, seq_len)
        # Zero the first attention weight
        A1[:,0,:] = 0.0
        # A1 = S1 /math.sqrt(self.seq_len)
        A1 = self.dropout(A1)
        Y1 = torch.matmul(A1, V1) # (batch_size, seq_len, vocab_size)
        Z1 = E + Y1  # (batch_size, seq_len, vocab_size)

        # SECOND LAYER
        # Compute queries and keys
        q2 = self.WQ2(Z1[:,-1]).unsqueeze(1)  # (batch_size, 1 , vocab_size)
        K2 = self.WK2(Z1)  # (batch_size, seq_len, vocab_size)
        

        # Compute attention scores and attention weights with masking
        S2 = self.beta_2*torch.matmul(q2, K2.transpose(-2, -1)) / math.sqrt(self.vocab_size) # (batch_size, 1, seq_len)
        # Mask out last position
        S2 = S2.masked_fill(self.mask2.to(S2.device), float('-inf'))
        A2 = S2.softmax(dim=-1)  # (batch_size, 1, seq_len)
        # A2 = S2/math.sqrt(self.seq_len) 
        A2 = self.dropout(A2)
        Y2 = torch.matmul(A2, E) # (batch_size, 1, vocab_size)

        # Compute logit outputs
        logits = self.beta_out * Y2  # (batch_size, 1, vocab_size)

        # Return logits at last position only
        return  logits[:,0,:]  # (batch_size, vocab_size)


class Lin_Sfm_Attention(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            seq_len: int,
            dropout: float = 0.0,
            attn: str = 'softmax'
            ) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.d_model = vocab_size + seq_len
        self.d_eff = self.seq_len
        self.attn = attn
        self.mask1 = torch.tril(torch.ones((seq_len,seq_len)),diagonal=0) == 0
        self.mask2 = torch.tensor([[[False]*(self.seq_len-2) + [True,True]]])
        if attn not in ['softmax', 'linear']:
            raise ValueError("Invalid attention type")
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        self.positions = nn.Embedding(seq_len,seq_len)

        # First layer weights
        self.WQ1 = nn.Linear(self.seq_len, self.seq_len, bias=False)
        self.WK1 = nn.Linear(self.seq_len, self.seq_len, bias=False)
        self.WV1 = nn.Linear(self.vocab_size, self.vocab_size, bias=False)

        # Second layer weights
        self.WQ2 = nn.Linear(self.vocab_size, self.vocab_size, bias=False)
        self.WK2 = nn.Linear(self.vocab_size, self.vocab_size, bias=False)

        # Dropout
        self.dropout=nn.Dropout(dropout)

        # Scalar Parameters
        self.beta_1 = nn.Parameter(torch.tensor(1.0))
        self.beta_2 = nn.Parameter(torch.tensor(1.0))
        self.beta_out = nn.Parameter(torch.tensor(1.0))



    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ X: (batch_size, seq_len) list of token ids """

        # Input embedding + positional encoding
        E = self.embedding(input)  # (batch_size, seq_len, vocab_size)
        P = self.positions(torch.arange(self.seq_len, device=input.device)) # (seq_len, seq_len)

        # FIRST LAYER
        # Compute queries and keys
        Q1 = self.WQ1(P)  # ( seq_len, seq_len)
        K1 = self.WK1(P)  # ( seq_len, seq_len)
        V1 = self.WV1(E)  # (batch_size, seq_len, vocab_size)

        # Compute attention scores and attention weights with masking
        S1 = self.beta_1*torch.matmul(Q1, K1.transpose(-2, -1)) /math.sqrt(self.seq_len)  # ( seq_len, seq_len)
        if self.attn == 'softmax':
            S1 = S1.masked_fill(self.mask1.to(S1.device), float('-inf'))
            A1 = S1.softmax(dim=-1).unsqueeze(0).expand(input.size(0), -1, -1).clone()   # (batch_size, seq_len, seq_len)
            # Zero the first attention weight
            A1[:,0,:] = 0.0
        elif self.attn == 'linear':
            A1 = S1 /math.sqrt(self.seq_len)
          
        A1 = self.dropout(A1)
        Y1 = torch.matmul(A1, V1) # (batch_size, seq_len, vocab_size)
        Z1 = E + Y1  # (batch_size, seq_len, vocab_size)

        # SECOND LAYER
        # Compute queries and keys
        q2 = self.WQ2(Z1[:,-1]).unsqueeze(1)  # (batch_size, 1 , vocab_size)
        K2 = self.WK2(Z1)  # (batch_size, seq_len, vocab_size)
        

        # Compute attention scores and attention weights with masking
        S2 = self.beta_2*torch.matmul(q2, K2.transpose(-2, -1)) / math.sqrt(self.vocab_size) # (batch_size, 1, seq_len)
        if self.attn == 'softmax':
            # Mask out last position
            S2 = S2.masked_fill(self.mask2.to(S2.device), float('-inf'))
            A2 = S2.softmax(dim=-1)  # (batch_size, 1, seq_len)
        elif self.attn == 'linear':
            A2 = S2/math.sqrt(self.seq_len) 
             
        A2 = self.dropout(A2)
        Y2 = torch.matmul(A2, E) # (batch_size, 1, vocab_size)

        # Compute logit outputs
        output = self.beta_out * Y2  # (batch_size, 1, vocab_size)

        # Return logits at last position only
        return  output[:,0,:]  # (batch_size, vocab_size)

def warm_initialization(model: InductionHeadAttention, alpha:float = 0.0 ,betas:tuple = (1.0,1.0,1.0), sigma:float = 0.5) -> None:

    V = model.vocab_size
    L = model.seq_len
    d_model = model.d_model
    beta_1, beta_2, beta_out = betas

    sig = sigma / math.sqrt(d_model)
    a = (1-alpha)#**(0.25)

    # Constants for scaling
    Ce = math.sqrt(V/d_model)
    Cp = math.sqrt(L/d_model)
    C1 = math.sqrt(L/d_model)
    Cv1 = math.sqrt(V/d_model)

    C2 = math.sqrt(V/d_model)


    # Matrices
    IdV = torch.eye(V)
    IdL = torch.eye(L)
    Zeros_LV = torch.zeros((L, V))
    shift = torch.zeros((L, L))
    shift[1:,:-1] = torch.eye(L-1)

    # V1 = torch.zeros((d_model, d_model))
    aux_V1 = torch.eye(V)
    aux_V1 = a*Cv1*aux_V1 + (1-a)*sig*torch.randn_like(aux_V1)
    M = aux_V1*math.sqrt(d_model/V)

    K = torch.cat((torch.eye(V), torch.zeros((V,L))), dim=1)

    with torch.no_grad():
        # Embeddings as one-hot encodding
        model.embedding.weight.data.copy_(Ce*torch.eye(V))
        # Positional encoding as one-hot encodding
        model.positions.weight.data.copy_(a*Cp*IdL + (1-a)*sig*torch.randn_like(IdL))

        # First layer
        model.WQ1.weight.data.copy_(a*C1*IdL + (1-a)*sig*torch.randn_like(IdL)) 
        model.WK1.weight.data.copy_(a*C1*shift + (1-a)*sig*torch.randn_like(IdL))
        model.WV1.weight.data.copy_(M)

        # Second layer
        model.WQ2.weight.data.copy_(a*C2*IdV + (1-a)*sig*torch.randn_like(IdV))
        model.WK2.weight.data.copy_(a*C2*IdV + (1-a)*sig*torch.randn_like(IdV))
        model.WV2.weight.data.copy_(a*C2*IdV + (1-a)*sig*torch.randn_like(IdV))

        # Scalar parameters
        model.beta_1.data.copy_(torch.tensor(beta_1))
        model.beta_2.data.copy_(torch.tensor(beta_2))
        model.beta_out.data.copy_(torch.tensor(beta_out))



def planted_initialization(model, betas:tuple = (1.0,1.0,1.0), cV:float = 1.0) -> None:
    V = model.vocab_size
    L = model.seq_len
    beta_1, beta_2, beta_out = betas

    # Matrices
    shift = torch.zeros((L, L))
    shift[1:,:-1] = torch.eye(L-1)

    with torch.no_grad():
        # Embeddings and positional encoding as one-hot 
        model.embedding.weight.data.copy_(math.sqrt(V)*torch.eye(V))
        model.positions.weight.data.copy_(math.sqrt(L)*torch.eye(L))

        # First layer
        model.WQ1.weight.data.copy_(torch.eye(L)) 
        model.WK1.weight.data.copy_(shift)
        model.WV1.weight.data.copy_(cV*torch.eye(V))

        # Second layer
        model.WQ2.weight.data.copy_(torch.eye(V))
        model.WK2.weight.data.copy_(torch.eye(V))
        
        # Scalar parameters
        model.beta_1.data.copy_(torch.tensor(beta_1))
        model.beta_2.data.copy_(torch.tensor(beta_2))
        model.beta_out.data.copy_(torch.tensor(beta_out))


def random_initialization(model, betas:tuple = (1.0,1.0,1.0), sigma:float = 0.5, cV:float = 1.0) -> None:
    V = model.vocab_size
    L = model.seq_len
    beta_1, beta_2, beta_out = betas
    sig = sigma / math.sqrt(V)

    # Matrices
    shift = torch.zeros((L, L))
    shift[1:,:-1] = torch.eye(L-1)

    with torch.no_grad():
        # Embeddings and positional encoding as one-hot 
        model.embedding.weight.data.copy_(math.sqrt(V)*sig*torch.randn(V, V))
        model.positions.weight.data.copy_(math.sqrt(V)*sig*torch.randn(L, L))

        # First layer
        model.WQ1.weight.data.copy_(math.sqrt(V/L)*sig*torch.randn(L, L)) 
        model.WK1.weight.data.copy_(math.sqrt(V/L)*sig*torch.randn(L, L))
        model.WV1.weight.data.copy_(cV*sig*torch.randn(V, V))

        # Second layer
        model.WQ2.weight.data.copy_(sig*torch.randn(V, V))
        model.WK2.weight.data.copy_(sig*torch.randn(V, V))
        

        # Scalar parameters
        model.beta_1.data.copy_(torch.tensor(beta_1))
        model.beta_2.data.copy_(torch.tensor(beta_2))
        model.beta_out.data.copy_(torch.tensor(beta_out))

def noisy_initialization(config,train_list=None) -> None:
    """
    Initialize the Induction Head Attention model with interpolation initialization between planted and random initialization.
    
    Input:
    ------
    config (dict): Configuration dictionary containing the following keys:
        - vocab_size (int): Vocabulary size.
        - seq_len (int): Sequence length.
        - dropout (float): Dropout rate.
        - alpha (float): Interpolation parameter between 0 and 1.   
        - beta_1 (float): Induction head beta_1 parameter.
        - beta_2 (float): Induction head beta_2 parameter.
        - beta_out (float): Induction head beta_out parameter.
        - sigma (float): Sigma for random initialization.
        - cV (float): Scaling factor for V matrices in planted and random initialization.
        - attn (str): Type of attention: 'softmax' or 'linear'.
    train_list (list, optional): List of parameter names to apply interpolation initialization. If None, interpolation is applied to all parameters. Default is None.

    Returns:
    --------
    model (InductionHeadAttention): The initialized model.
    device (torch.device): The device on which the model is located.
    """
    # Create model in device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Initialize model with planted parameters
    planted_model = Lin_Sfm_Attention(vocab_size=config['vocab_size'], seq_len=config['seq_len'], dropout=config['dropout'],attn=config['attn']).to(device)
    planted_initialization(planted_model,betas=(config['beta_1'], config['beta_2'], config['beta_out']), cV=config['cV'])
    # Initialize model with random parameters
    random_model = Lin_Sfm_Attention(vocab_size=config['vocab_size'], seq_len=config['seq_len'], dropout=config['dropout'],attn=config['attn']).to(device)
    random_initialization(random_model,betas=(config['beta_1'], config['beta_2'], config['beta_out']), sigma=config['sigma'], cV=config['cV'])

    # Interpolate between planted and random parameters just for the parameters in train_list
    with torch.no_grad():
        for (name, param), (_, random_param) in zip(planted_model.named_parameters(), random_model.named_parameters()):
            if (train_list is None or name in train_list) and param.ndim > 0:
                param.copy_(config['alpha']*random_param + (1-config['alpha'])*param)
    
    return planted_model, device

def _interpolation_initialization(model: InductionHeadAttention,  alpha: float = 0.0, sigma: float = 0.5) -> None:
    """Initialize the Induction Head Attention model with interpolation initialization.
    
    The interpolation is between the model's current parameters and random 
    gaussian parameters with std = 1/sqrt(d_model). The interpolation is controlled by the alpha parameter, where:
    - alpha = 0 corresponds to the original parameters (no change)
    - alpha = 1 corresponds to completely random parameters (full change)
    Args:        
        model (InductionHeadAttention): The model to initialize.
        alpha (float): Interpolation parameter between 0 and 1.
    """


    # Create temporary model with random initialization and interpolate
    temp_model = InductionHeadAttention(
        vocab_size = model.vocab_size,
        seq_len = model.seq_len,
    ).to(next(model.parameters()).device)

    # Random model
    warm_initialization(temp_model, alpha=1.0, betas=(model.beta_1.item(), model.beta_2.item(), model.beta_out.item()), sigma=sigma)

    with torch.no_grad():
        # Interpolate only parameters that required gradients (trainable parameters) apart from scalar parameters
        for param, temp_param in zip(model.parameters(), temp_model.parameters()):
            if param.requires_grad and param.ndim > 0:  # Check if the parameter is trainable and not a scalar
                param.copy_((1-alpha)*param + alpha*temp_param)