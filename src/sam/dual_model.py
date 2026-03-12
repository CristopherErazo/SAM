from torch import nn
import torch
import math

class DualModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            vocab_size: int,
            seq_len: int,
            dropout: float = 0.0,
            ) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Embeddings
        self.E = nn.Embedding(vocab_size, d_model)
        self.P = nn.Embedding(seq_len,d_model)
        
        # First attention
        self.WQK1 = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W0V1 = nn.Linear(self.d_model, self.d_model, bias=False)

        # First attention
        self.WQK2 = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W0V2 = nn.Linear(self.d_model, self.d_model, bias=False)

        # Linear layer
        self.WF = nn.Linear(self.d_model, self.d_model, bias=False)

        # Unembedding
        self.U = nn.Linear(self.d_model, vocab_size, bias=False)

        # Dropout
        self.dropout=nn.Dropout(dropout)

        # Extras
        self.positions = torch.arange(self.seq_len)

    def full_output(self, input: torch.Tensor,mask:torch.Tensor) -> torch.Tensor:
        """ X: (batch_size, seq_len) list of token ids """
        output = {}
        with torch.no_grad():
            # Input embedding + positional encoding
            X0 = self.E(input) + self.P(self.positions.to(input.device))  # (batch_size, seq_len, d_model)
            print('X0',X0.std())
            # FIRST LAYER
            S1 = self.WQK1(X0)  @ X0.transpose(-2, -1) * math.sqrt(self.d_model)  # (batch_size, seq_len, seq_len)
            output['S1'] = S1
            print('S1',S1.std())
            S1 = S1.masked_fill(~mask, float('-inf'))
            A1 = S1.softmax(dim=-1)  # (batch_size, seq_len, seq_len)
            output['A1'] = A1
            A1 = self.dropout(A1)        
            Y1 = A1 @ self.W0V1(X0)  # (batch_size, seq_len, d_model)
            output['Y1'] = Y1
            print('Y1',Y1.std())
            X1 = X0 + Y1  # ( batch_size, seq_len, d_model)
            print('X1',X1.std())
            # SECOND LAYER
            S2 = self.WQK2(X1) @ X1.transpose(-2, -1) * math.sqrt(self.d_model)   # (batch_size, seq_len, seq_len)
            print('S2',S2.std())
            output['S2'] = S2
            S2 = S2.masked_fill(~mask, float('-inf'))
            A2 = S2.softmax(dim=-1)  # (batch_size, seq_len, seq_len)
            output['A2'] = A2        
            A2 = self.dropout(A2)
            Y2 = A2 @ self.W0V2(X1) * 2  # (batch_size, seq_len, d_model)
            print('Y2',Y2.std())
            output['Y2'] = Y2
            X2 = X1 + Y2  # ( batch_size, seq_len, d_model)
            output['X2'] = X2
            print('X2',X2.std())
            # Linear layer
            Y3 = self.WF(X2)  # (batch_size, seq_len, d_model)
            print('Y3',Y3.std())
            output['Y3'] = Y3
            X3 = X2 + Y3  # (batch_size, seq_len, d_model)
            print('X3',X3.std())
            output['X3'] = X3

            # Unembedding
            logits = self.U(X3)  # (batch_size, seq_len, vocab_size)
            output['logits'] = logits #* math.sqrt(self.d_model)
            output['logits_X2'] = (self.U(X2) * math.sqrt(self.d_model))
            output['logits_Y3'] = (self.U(Y3) * math.sqrt(self.d_model))
        for key in output:
            output[key] = output[key].cpu().detach().numpy()
        return output
    

    def forward(self, input: torch.Tensor,mask:torch.Tensor) -> torch.Tensor:
        """ X: (batch_size, seq_len) list of token ids """

        # Input embedding + positional encoding
        X0 = self.E(input) + self.P(self.positions.to(input.device))  # (batch_size, seq_len, d_model)
       
        # FIRST LAYER
        S1 = self.WQK1(X0)  @ X0.transpose(-2, -1) * math.sqrt(self.d_model)  # (batch_size, seq_len, seq_len)
        S1 = S1.masked_fill(~mask, float('-inf'))
        A1 = S1.softmax(dim=-1)  # (batch_size, seq_len, seq_len)
        A1 = self.dropout(A1)        
        Y1 = A1 @ self.W0V1(X0)  # (batch_size, seq_len, d_model)
        X1 = X0 + Y1  # ( batch_size, seq_len, d_model)

        # SECOND LAYER
        S2 = self.WQK2(X1) @ X1.transpose(-2, -1) * math.sqrt(self.d_model)   # (batch_size, seq_len, seq_len)
        S2 = S2.masked_fill(~mask, float('-inf'))
        A2 = S2.softmax(dim=-1)  # (batch_size, seq_len, seq_len)   
        A2 = self.dropout(A2)
        Y2 = A2 @ self.W0V2(X1) * 3  # (batch_size, seq_len, d_model)
        X2 = X1 + Y2  # ( batch_size, seq_len, d_model)

        # Linear layer
        Y3 = self.WF(X2)  # (batch_size, seq_len, d_model)
        X3 = X2 + Y3  # (batch_size, seq_len, d_model)

        # Unembedding
        logits = self.U(X3)  # (batch_size, seq_len, vocab_size)
        return logits


def initialize_dual_model(model , P_b, trigger_set):
    with torch.no_grad():
        # Initialize E , U ~ N(0, 1/sqrt(d_model)) and freeze them
        model.E.weight.copy_(torch.randn_like(model.E.weight) / math.sqrt(model.d_model))
        model.P.weight.copy_(torch.randn_like(model.P.weight) / math.sqrt(model.d_model))
        model.U.weight.copy_(torch.randn_like(model.U.weight) / math.sqrt(model.d_model))
        model.E.weight.requires_grad = False
        model.P.weight.requires_grad = False
        model.U.weight.requires_grad = False

        # Initialize WOV1 ~ N(0, 1/sqrt(d_model)) and freeze it
        model.W0V1.weight.copy_(torch.randn_like(model.W0V1.weight) / math.sqrt(model.d_model))
        model.W0V1.weight.requires_grad = False 

        # Initialize WQK1 as sum_mu=2^L P[mu]*P[mu-1]^T where P is the positional embedding matrix 
        # Remembed that model.P.weight.shape = (seq_len, d_model) 
        WQK1_init = torch.zeros_like(model.WQK1.weight)
        for mu in range(1,model.seq_len):
            WQK1_init += torch.outer(model.P.weight[mu], model.P.weight[mu-1]) 
        model.WQK1.weight.copy_(WQK1_init.T)

        # Initialize WQK2 as sum_t E[t] * (W0V1 @ E[t])^T where E is the token embedding matrix and W0V1 is the value projection of the first attention and t in trigger_set
        WQK2_init = torch.zeros_like(model.WQK2.weight)
        for t in trigger_set:
            WQK2_init += torch.outer(model.E.weight[t], model.E.weight[t])
        model.WQK2.weight.copy_( ( model.W0V1.weight @ WQK2_init))



        # Initialize W0V2 as sum t U[t] * E[t]^T where U is the unembedding matrix and E is the token embedding matrix and t is all vocab tokens
        # Remember that model.U.weight.shape = (vocab_size, d_model) and model.E.weight.shape = (vocab_size, d_model)
        W0V2_init = torch.zeros_like(model.W0V2.weight)
        for t in range(model.vocab_size):
            W0V2_init += torch.outer(model.U.weight[t], model.E.weight[t])
        model.W0V2.weight.copy_(W0V2_init)

        # Initialize WF as sum_t1,t2 log P_b [t1,t2] * U[t2] * E[t1]^T where 
        # P_b[t1,t2]  = Prob(next token = t2 | current token = t1)
        WF_init = torch.zeros_like(model.WF.weight)
        for t1 in range(model.vocab_size):
            for t2 in range(model.vocab_size):
                WF_init += math.log(P_b[t1,t2]) * torch.outer(model.U.weight[t2], model.E.weight[t1])
        model.WF.weight.copy_(WF_init)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        return model , device
