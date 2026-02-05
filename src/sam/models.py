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