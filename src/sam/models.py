from torch import nn
import torch
import math
from .activations import get_activation



################################################################
################################################################
################# SINGLE INDEX MODELS ##########################
################################################################
################################################################

class SingleIndex(nn.Module):
    """Single Index Model:  y = f(Wx/sqrt(d)) """
    def __init__(self, d: int, function_specs):
        super().__init__()
        self.W = nn.Linear(d, 1 , bias=False)
        self.activation = get_activation(function_specs)[0]
        self.d = d
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Single Index Model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, d).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        return self.activation(self.W(x)/ math.sqrt(self.d))
    
def init_teacher_student(d : int=10, teacher_act = 'He3', student_act = 'relu', device="cpu"):
    """Teacher is frozen; student is trainable."""
    teacher = SingleIndex(d,teacher_act).to(device)
    student = SingleIndex(d,student_act).to(device)

    with torch.no_grad():
        teacher.W.weight.normal_(0, 1)
        student.W.weight.normal_(0, 1)
        # Normalize both teacher and student weights
        teacher.W.weight /= torch.norm(teacher.W.weight)/math.sqrt(d)
        student.W.weight /= torch.norm(student.W.weight)/math.sqrt(d)

    for p in teacher.parameters():
        p.requires_grad_(False)

    # Normalize teacher weights
    w_teacher = torch.cat([p.view(-1) for p in teacher.parameters()])


    return teacher, student, w_teacher


################################################################
################################################################
################## TRANSFORMER MODELS ##########################
################################################################
################################################################


class InputEmbeddings(nn.Module):
    """
    Embedding layer to convert token IDs to dense vectors.
    
    Args:
        d_model (int): dimension of the embeddings
        vocab_size (int): size of the vocabulary 
        freeze (bool): whether to freeze the embeddings during training
    """
    def __init__(self, d_model: int, vocab_size: int, freeze:bool = False) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        if freeze:
            self.embedding.weight.requires_grad = False

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):
    """
    Positional Encoding layer to add positional information to token embeddings.
    
    The encodings are random and learned during training.
    """ 
    def __init__(self, d_model: int, seq_len: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.pos_embedding = nn.Embedding(seq_len, d_model)

    def forward(self, x):
        """ Add positional encodings to input embeddings.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_encodings = self.pos_embedding(positions)  # (batch_size, seq_len, d_model)
        return x + pos_encodings

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_eff: int, seq_len: int = 64):
        super().__init__()
        assert d_eff % 2 == 0, "d_eff must be even"

        inv_freq = 1.0 / (
            500 ** (torch.arange(0, d_eff, 2).float() / d_eff)
        )

        positions = torch.arange(seq_len).float()
        freqs = torch.einsum("i,j->ij", positions, inv_freq)

        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x):
        """
        x: (batch, n_heads, L, d_eff)
        """
        L = x.size(2)

        cos = self.cos[:L].unsqueeze(0).unsqueeze(0)  # (1, 1, L, d_eff/2)
        sin = self.sin[:L].unsqueeze(0).unsqueeze(0)

        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]

        x_rot = torch.stack(
            [
                x_even * cos - x_odd * sin,
                x_even * sin + x_odd * cos,
            ],
            dim=-1,
        )

        return x_rot.flatten(-2) # (batch, n_heads, L, d_eff)


class MultiHeadAttentionLayer(nn.Module):
    """
    Self-attention layer with low-rank factorization.
    
    Args:
        d_model (int): dimension of the embeddings
        seq_len (int): maximum sequence length
        dropout (float): dropout rate 
        d_eff (int): dimension of the Key, Query  and Value projections
        n_heads (int): number of attention heads
    """
    def __init__(self,d_model:int,seq_len:int,d_eff:int,n_heads:int,dropout:float=0.0) -> None:
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.d_eff = d_eff
        self.n_heads = n_heads
        self.d_full = d_eff * n_heads

        self.Wq = nn.Linear(d_model, self.d_full, bias=False)
        self.Wk = nn.Linear(d_model, self.d_full, bias=False)
        self.Wv = nn.Linear(d_model, self.d_full, bias=False)
        self.Wo = nn.Linear(self.d_full, d_model, bias=False)
        self.dropout=nn.Dropout(dropout)

        self.RoPE = RotaryPositionalEncoding(d_eff, seq_len)

    def attention_scores(self, Q, K):
        """ Compute attention scores.
            Q: (batch_size,n_heads,L,deff)
            K: (batch_size,n_heads,L,deff)
        """
        S = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_eff) # (batch_size,nheads,L,L)
        return S

    def forward(self,X,mask=None):
        """ X: (batch_size,L,d_model) 
            mask: (batch_size,L,L)
        """
        batch_size, _, _ = X.shape
        # Projections
        Q = self.Wq(X)  # (batch_size, seq_len, dfull)
        K = self.Wk(X)  # (batch_size, seq_len, dfull)
        V = self.Wv(X)  # (batch_size, seq_len, dfull)

        # Reshape for multi-head attention
        Q = Q.view(batch_size,self.seq_len,self.n_heads,self.d_eff).permute(0,2,1,3)  # (batch_size,n_heads,L,deff)
        K = K.view(batch_size,self.seq_len,self.n_heads,self.d_eff).permute(0,2,1,3)  # (batch_size,n_heads,L,deff)
        V = V.view(batch_size,self.seq_len,self.n_heads,self.d_eff).permute(0,2,1,3)  # (batch_size,n_heads,L,deff)
        
        # Apply Rotary Positional Encoding
        Q = self.RoPE(Q)  # (batch_size,n_heads,L,deff)
        K = self.RoPE(K)  # (batch_size,n_heads,L,deff

        # Dot-product attention
        S = self.attention_scores(Q,K)  # (batch_size,nheads,L,L)

        if mask is not None:
            S = S.masked_fill(mask == 0, -1e9)
        A = torch.softmax(S,dim=1)  # (batch_size,nheads,L,L)
        if self.dropout is not None:
            A = self.dropout(A)

        # Value aggregation and output projection
        Y = torch.matmul(A,V) # (batch_size,nheads,L,deff)
        Y = Y.permute(0,2,1,3).contiguous().view(batch_size,self.seq_len,self.d_full)  # (batch_size,L,d_full)
        Y = self.Wo(Y)  # (batch_size, d_model)

        return Y * math.sqrt(self.d_model/self.d_full)



class AttentionOnlyTransformer(nn.Module):

    def __init__(
            self,
            d_model: int,
            d_eff: int,
            vocab_size: int,
            seq_len: int,
            n_heads: int,
            n_layers: int,
            dropout: float = 0.0,
            freeze_embeddings: bool = False  
            ) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.embedding = InputEmbeddings(d_model, vocab_size, freeze=freeze_embeddings)
        self.positional_encoding = PositionalEncoding(d_model, seq_len)
        self.layers = nn.ModuleList([
            MultiHeadAttentionLayer(d_model, seq_len, d_eff, n_heads, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, input:torch.tensor, mask=None) -> torch.Tensor:
        """ Forward pass of the Attention-Only Transformer.
        
        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, seq_len).
            mask (torch.Tensor, optional): Attention mask of shape (batch_size, seq_len, seq_len). Defaults to None.
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Input embedding
        X = self.embedding(input)  # (batch_size, seq_len, d_model)
        # X = self.positional_encoding(X)  # (batch_size, seq_len, d_model)

        # Pass through each attention layer with residual connections
        for layer in self.layers:
            X = X + layer(X, mask)  # (batch_size, seq_len, d_model)
        
        # Last position output
        X = X[:, -1, :]  # (batch_size, d_model)

        # Project over embedding matrix to get logits
        logits = torch.matmul(X,self.embedding.embedding.weight.t())  # (batch_size, vocab_size)

        return logits
    

def create_transformer(config:dict) -> tuple[AttentionOnlyTransformer, torch.device]:
    """Create an Attention-Only Transformer model based on the provided configuration.
    
    Args:
        config (dict): Configuration dictionary containing model parameters.
                - d_model (int): dimension of the embeddings
                - d_eff (int): dimension of the Key, Query  and Value projections
                - vocab_size (int): size of the vocabulary 
                - seq_len (int): maximum sequence length
                - n_heads (int): number of attention heads
                - n_layers (int): number of attention layers
                - dropout (float): dropout rate 
                - sigma (float): standard deviation for parameter initialization
                - freeze_embeddings (bool): whether to freeze the embeddings during training
    
    Returns:
        AttentionOnlyTransformer: The created transformer model.
        str: A string representation of the model configuration.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = AttentionOnlyTransformer(
        d_model = config['d_model'],
        d_eff = config['d_eff'],
        vocab_size = config['vocab_size'],
        seq_len = config['seq_len'],
        n_heads = config['n_heads'],
        n_layers = config['n_layers'],
        dropout = config['dropout'],
        freeze_embeddings = config['fr_emb']
    ).to(device)

    # Initialize the parameters with sparse initialization
    std_init = config['sigma'] / math.sqrt(config['d_model'])
    for name, param in model.named_parameters():
        if 'weight' in name:
            torch.nn.init.normal_(param, mean=0.0, std=std_init)
            # Introduce sparsity: set 50% of weights to zero
            mask = torch.rand_like(param) > config['sparsity'] 
            param.data.mul_(mask)
        elif 'bias' in name:
            torch.nn.init.constant_(param, 0.0)

    # std_init = config['sigma'] / math.sqrt(config['d_model'])
    # for param in model.parameters():
    #     if param.dim() > 1:
    #         torch.nn.init.normal_(param, mean=0.0, std=std_init)


    return model, device









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
        self.mask2 = torch.tril(torch.ones((seq_len,seq_len)),diagonal=0) == 0

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        self.positions = nn.Embedding(seq_len,seq_len)

        # First layer weights
        self.WQ1 = nn.Linear(self.d_model, self.d_eff, bias=False)
        self.WK1 = nn.Linear(self.d_model, self.d_eff, bias=False)
        self.WV1 = nn.Linear(self.d_model, self.d_model, bias=False)

        # Second layer weights
        self.WQ2 = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.WK2 = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.WV2 = nn.Linear(self.d_model, self.vocab_size, bias=False)

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
        P = P.unsqueeze(0).expand(input.size(0), -1, -1)  # (batch_size, seq_len, seq_len)
        X = torch.cat([E, P], dim=-1)  # (batch_size, seq_len, d_model)

        # FIRST LAYER
        # Compute queries and keys
        Q1 = self.WQ1(X)  # (batch_size, seq_len, d_eff)
        K1 = self.WK1(X)  # (batch_size, seq_len, d_eff)
        V1 = self.WV1(X)  # (batch_size, seq_len, d_model)

        # Compute attention scores and attention weights with masking
        S1 = self.beta_1*torch.matmul(Q1, K1.transpose(-2, -1)) * (self.d_model/self.seq_len)**2  # (batch_size, seq_len, seq_len)
        S1 = S1.masked_fill(self.mask1.unsqueeze(0).to(S1.device), float('-inf'))
        A1 = S1.softmax(dim=-1)  # (batch_size, seq_len, seq_len)
        A1 = self.dropout(A1)
        Y1 = torch.matmul(A1, V1)  # (batch_size, seq_len, d_model)
        Z1 = X + Y1  # (batch_size, seq_len, d_model)

        # SECOND LAYER
        # Compute queries and keys
        q2 = self.WQ2(Z1[:,-1]).unsqueeze(1)  # (batch_size, 1 , vocab_size)
        K2 = self.WK2(Z1)  # (batch_size, seq_len, vocab_size)
        V2 = self.WV2(Z1)  # (batch_size, seq_len, vocab_size)

        # Compute attention scores and attention weights with masking
        S2 = self.beta_2*torch.matmul(q2, K2.transpose(-2, -1)) * (self.d_model/self.vocab_size)**2 # (batch_size, 1, seq_len)
        # Mask out all positions whose token is the same as the last token
        # last_tokens = input[:, -1].unsqueeze(1)  # (batch_size, 1)
        # token_mask = (input != last_tokens).unsqueeze(1)  # (batch_size, 1, seq_len)
        # S2 = S2.masked_fill(~token_mask.to(S2.device), float('-inf'))
        A2 = S2.softmax(dim=-1)  # (batch_size, 1, seq_len)
        A2 = self.dropout(A2)
        Y2 = torch.matmul(A2, V2)  # (batch_size, 1, vocab_size)

        # Compute logit outputs as projection onto embeddings
        logits = self.beta_out*torch.matmul(Y2, self.embedding.weight.t()) * (self.d_model/self.vocab_size)**1.5 # (batch_size, 1, vocab_size)

        # Return logits at last position only
        return  logits[:,0,:]  # (batch_size, vocab_size)
    
def interpolation_initialization(model: InductionHeadAttention, alpha: float = 0.0) -> None:
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
        dropout = 0.0,
    ).to(next(model.parameters()).device)

    planted_initialization(temp_model, alpha=1.0, betas=(model.beta_1.item(), model.beta_2.item(), model.beta_out.item()))

    with torch.no_grad():
        for param, temp_param in zip(model.parameters(), temp_model.parameters()):
            # Skip scalar parameters (beta_1, beta_2, beta_out)
            if param.shape == torch.Size([]):
                continue
            param.copy_( (1 - alpha) * param + alpha * temp_param )

def planted_initialization(model: InductionHeadAttention, alpha:float,betas:tuple) -> None:

    V = model.vocab_size
    L = model.seq_len
    d_model = model.d_model
    beta_1, beta_2, beta_out = betas

    sigma = 1.0 / math.sqrt(d_model)
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
    aux_V1 = a*Cv1*aux_V1 + (1-a)*sigma*torch.randn_like(aux_V1)
    M = torch.zeros((d_model, d_model))
    M[:V,:V] = aux_V1*math.sqrt(d_model/V)

    K = torch.cat((torch.eye(V), torch.zeros((V,L))), dim=1)

    with torch.no_grad():
        # Embeddings as one-hot encodding
        model.embedding.weight.copy_(Ce*torch.eye(V))
        # Positional encoding as one-hot encodding
        model.positions.weight.copy_(a*Cp*IdL + (1-a)*sigma*torch.randn_like(IdL))

        # First layer
        model.WQ1.weight.copy_( torch.cat([Zeros_LV, a*C1*IdL + (1-a)*sigma*torch.randn_like(IdL)] , dim=1)) 
        model.WK1.weight.copy_( torch.cat([Zeros_LV, a*C1*shift + (1-a)*sigma*torch.randn_like(IdL)], dim=1))
        model.WV1.weight.copy_(M)

        # Second layer
        model.WQ2.weight.copy_(torch.cat([a*C2*IdV + (1-a)*sigma*torch.randn_like(IdV), Zeros_LV.T], dim=1))
        model.WK2.weight.copy_(torch.cat([a*C2*IdV + (1-a)*sigma*torch.randn_like(IdV), Zeros_LV.T], dim=1))
        model.WV2.weight.copy_(torch.cat([a*C2*IdV + (1-a)*sigma*torch.randn_like(IdV), Zeros_LV.T], dim=1))

        # Scalar parameters
        model.beta_1.copy_(torch.tensor(beta_1))
        model.beta_2.copy_(torch.tensor(beta_2))
        model.beta_out.copy_(torch.tensor(beta_out))

        # Freeze all components that include Zeros_LV to maintain the planted structure
        # model.WQ1.weight[:,:V].requires_grad = False
        # model.WK1.weight[:,:V].requires_grad = False
        # model.WV1.weight.requires_grad = False
        # model.WV1.weight[:V,:V].requires_grad = False

        # model.WQ2.weight[:,V:].requires_grad = False
        # model.WK2.weight[:,V:].requires_grad = False
        # model.WV2.weight[:,V:].requires_grad = False

        # model.embedding.weight.requires_grad = False
    

def create_induction_head(config:dict) -> tuple[InductionHeadAttention, torch.device]:

    """Create an Induction Head Attention model based on the provided configuration.
    Args:
        config (dict): Configuration dictionary containing model parameters.
                - vocab_size (int): size of the vocabulary 
                - seq_len (int): maximum sequence length
                - dropout (float): dropout rate 
    Returns:
        InductionHeadAttention: The created induction head attention model.
        str: A string representation of the model device.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = InductionHeadAttention(
        vocab_size = config['vocab_size'],
        seq_len = config['seq_len'],
        dropout = config['dropout'],
    ).to(device)

    return model, device





class InductionHeadAttentionSmaller(nn.Module):
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
        self.mask2 = torch.tensor([[[False]*(self.seq_len-1) + [True]]])

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
        self.WV2 = nn.Linear(self.vocab_size, self.vocab_size, bias=False)

        # Dropout
        self.dropout=nn.Dropout(dropout)

        # Scalar Parameters
        self.beta_1 = nn.Parameter(torch.tensor(1.0))
        self.beta_2 = nn.Parameter(torch.tensor(1.0))
        self.beta_out = nn.Parameter(torch.tensor(1.0))
    

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
            V2 = self.WV2(E)  # (batch_size, seq_len, vocab_size)
            output['q2'] = q2.detach().cpu().numpy()
            output['K2'] = K2.detach().cpu().numpy()
            output['V2'] = V2.detach().cpu().numpy()

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
    
        # Return logits at last position only
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
        S1 = math.sqrt(self.seq_len)*self.beta_1*torch.matmul(Q1, K1.transpose(-2, -1)) * (self.d_model/self.seq_len)**2  # ( seq_len, seq_len)
        S1 = S1.masked_fill(self.mask1.to(S1.device), float('-inf'))
        A1 = S1.softmax(dim=-1).unsqueeze(0).expand(input.size(0), -1, -1).clone()   # (batch_size, seq_len, seq_len)
        # Zero the first attention weight
        A1[:,0,:] = 0.0
        A1 = self.dropout(A1)
        Y1 = torch.matmul(A1, V1)  # (batch_size, seq_len, vocab_size)
        Z1 = E + Y1  # (batch_size, seq_len, vocab_size)

        # SECOND LAYER
        # Compute queries and keys
        q2 = self.WQ2(Z1[:,-1]).unsqueeze(1)  # (batch_size, 1 , vocab_size)
        K2 = self.WK2(Z1)  # (batch_size, seq_len, vocab_size)
        V2 = self.WV2(E)  # (batch_size, seq_len, vocab_size)

        # Compute attention scores and attention weights with masking
        S2 = math.sqrt(self.seq_len)*self.beta_2*torch.matmul(q2, K2.transpose(-2, -1)) * (self.d_model/self.vocab_size)**2 # (batch_size, 1, seq_len)
        # Mask out last position
        S2 = S2.masked_fill(self.mask2.to(S2.device), float('-inf'))

        A2 = S2.softmax(dim=-1)  # (batch_size, 1, seq_len)
        
        A2 = self.dropout(A2)

        Y2 = torch.matmul(A2, V2)  # (batch_size, 1, vocab_size)

        # Compute logit outputs as projection onto embeddings
        logits = math.sqrt(self.vocab_size)*self.beta_out*torch.matmul(Y2, self.embedding.weight.t()) * (self.d_model/self.vocab_size)**1.5 # (batch_size, 1, vocab_size)

        # Return logits at last position only
        return  logits[:,0,:]  # (batch_size, vocab_size)
    


def planted_initialization_small(model: InductionHeadAttentionSmaller, alpha:float,betas:tuple) -> None:

    V = model.vocab_size
    L = model.seq_len
    d_model = model.d_model
    beta_1, beta_2, beta_out = betas

    sigma = 1.0 / math.sqrt(d_model)
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
    aux_V1 = a*Cv1*aux_V1 + (1-a)*sigma*torch.randn_like(aux_V1)
    M = aux_V1*math.sqrt(d_model/V)

    K = torch.cat((torch.eye(V), torch.zeros((V,L))), dim=1)

    with torch.no_grad():
        # Embeddings as one-hot encodding
        model.embedding.weight.data.copy_(Ce*torch.eye(V))
        # Positional encoding as one-hot encodding
        model.positions.weight.data.copy_(a*Cp*IdL + (1-a)*sigma*torch.randn_like(IdL))

        # First layer
        model.WQ1.weight.data.copy_(a*C1*IdL + (1-a)*sigma*torch.randn_like(IdL)) 
        model.WK1.weight.data.copy_(a*C1*shift + (1-a)*sigma*torch.randn_like(IdL))
        model.WV1.weight.data.copy_(M)

        # Second layer
        model.WQ2.weight.data.copy_(a*C2*IdV + (1-a)*sigma*torch.randn_like(IdV))
        model.WK2.weight.data.copy_(a*C2*IdV + (1-a)*sigma*torch.randn_like(IdV))
        model.WV2.weight.data.copy_(a*C2*IdV + (1-a)*sigma*torch.randn_like(IdV))

        # Scalar parameters
        model.beta_1.data.copy_(torch.tensor(beta_1))
        model.beta_2.data.copy_(torch.tensor(beta_2))
        model.beta_out.data.copy_(torch.tensor(beta_out))


def interpolation_initialization_smaller(model: InductionHeadAttentionSmaller, alpha: float = 0.0) -> None:
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
    temp_model = InductionHeadAttentionSmaller(
        vocab_size = model.vocab_size,
        seq_len = model.seq_len,
        dropout = 0.0,
    ).to(next(model.parameters()).device)

    planted_initialization_small(temp_model, alpha=1.0, betas=(model.beta_1.item(), model.beta_2.item(), model.beta_out.item()))

    with torch.no_grad():
        for param, temp_param in zip(model.parameters(), temp_model.parameters()):
            # Skip scalar parameters (beta_1, beta_2, beta_out)
            if param.shape == torch.Size([]):
                continue
            param.copy_( (1 - alpha) * param + alpha * temp_param )