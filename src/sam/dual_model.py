from torch import nn
import torch
import math

class DualModel(nn.Module):
    def __init__(
            self,
            config: dict,
            ) -> None:
        super().__init__()

        self.seq_len = config['seq_len']
        self.vocab_size = config['vocab_size']
        self.d_model = config['d_model']
        self.drop = config['dropout']

        # Embeddings
        self.E = nn.Embedding(self.vocab_size, self.d_model)
        self.P = nn.Embedding(self.seq_len,self.d_model)
        
        # First attention
        self.WQK1 = nn.Linear(self.d_model, self.d_model, bias=False)
        self.WOV1 = nn.Linear(self.d_model, self.d_model, bias=False)

        # First attention
        self.WQK2 = nn.Linear(self.d_model, self.d_model, bias=False)
        self.WOV2 = nn.Linear(self.d_model, self.d_model, bias=False)

        # Linear layer
        self.WF = nn.Linear(self.d_model, self.d_model, bias=False)

        # Unembedding
        self.U = nn.Linear(self.d_model, self.vocab_size, bias=False)

        # Dropout
        self.dropout=nn.Dropout(self.drop)

        # Extras
        self.positions = torch.arange(self.seq_len)
        # self.gamma = nn.Parameter(torch.tensor(4.0))

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
            Y1 = A1 @ self.WOV1(X0)  # (batch_size, seq_len, d_model)
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
            Y2 = A2 @ self.WOV2(X1) * 2  # (batch_size, seq_len, d_model)
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
    

    def forward(self, input: torch.Tensor,mask:torch.Tensor,attn:bool=True,fc:bool=True) -> torch.Tensor:
        """ X: (batch_size, seq_len) list of token ids """

        # Input embedding + positional encoding
        X = self.E(input) + self.P(self.positions.to(input.device))  # (batch_size, seq_len, d_model)

        if attn:
            # FIRST LAYER
            S = self.WQK1(X)  @ X.transpose(-2, -1) * math.sqrt(self.d_model)  # (batch_size, seq_len, seq_len)
            S = S.masked_fill(~mask, float('-inf'))
            A = S.softmax(dim=-1)  # (batch_size, seq_len, seq_len)
            A = self.dropout(A)        
            Y = A @ self.WOV1(X)  # (batch_size, seq_len, d_model)
            X = X + Y  # ( batch_size, seq_len, d_model)

            # SECOND LAYER
            S = self.WQK2(X) @ X.transpose(-2, -1) * math.sqrt(self.d_model)   # (batch_size, seq_len, seq_len)
            S = S.masked_fill(~mask, float('-inf'))
            A = S.softmax(dim=-1)  # (batch_size, seq_len, seq_len)   
            A = self.dropout(A)
            Y = A @ self.WOV2(X) #* self.gamma # (batch_size, seq_len, d_model)
            X = X + Y  # ( batch_size, seq_len, d_model)

        if fc:
            # Linear layer
            Y = self.WF(X)  # (batch_size, seq_len, d_model)
            X = X + Y  # (batch_size, seq_len, d_model)

        # Unembedding
        logits = self.U(X)  # (batch_size, seq_len, vocab_size)
        return logits


def initialize_dual_model(model , P_b , trigger_set=None, init='planted'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    with torch.no_grad():
        # Initialize E , U ~ N(0, 1/sqrt(d_model)) and freeze them
        model.E.weight.copy_(torch.randn_like(model.E.weight) / math.sqrt(model.d_model))
        model.P.weight.copy_(torch.randn_like(model.P.weight) / math.sqrt(model.d_model))
        model.U.weight.copy_(torch.randn_like(model.U.weight) / math.sqrt(model.d_model))
        model.E.weight.requires_grad = False
        model.P.weight.requires_grad = False
        model.U.weight.requires_grad = False

        if init == 'planted':

            # Initialize WOV1 ~ N(0, 1/sqrt(d_model)) and freeze it
            model.WOV1.weight.copy_(torch.randn_like(model.WOV1.weight) / math.sqrt(model.d_model))
            model.WOV1.weight.requires_grad = False 

            # Initialize WQK1 as sum_mu=2^L P[mu]*P[mu-1]^T where P is the positional embedding matrix 
            # Remembed that model.P.weight.shape = (seq_len, d_model) 
            # WQK1_init = torch.zeros_like(model.WQK1.weight)
            # for mu in range(1,model.seq_len):
            #     WQK1_init += torch.outer(model.P.weight[mu], model.P.weight[mu-1]) 
            # model.WQK1.weight.copy_(WQK1_init.T) #(d_model, d_model)
            P = model.P.weight
            model.WQK1.weight.copy_(P[:-1].T @ P[1:]) #(d_model, d_model)

            # Initialize WQK2 as sum_t E[t] * (WOV1 @ E[t])^T where E is the token embedding matrix and WOV1 is the value projection of the first attention and t in trigger_set
            # WQK2_init = torch.zeros_like(model.WQK2.weight)
            # rnge = range(model.vocab_size) if trigger_set is None else trigger_set
            # for t in rnge:
            #     WQK2_init += torch.outer(model.E.weight[t], model.E.weight[t])
            # model.WQK2.weight.copy_(model.WOV1.weight @ WQK2_init) #(d_model, d_model) 
            E = model.E.weight
            if trigger_set is None:
                E_sub = E
            else:
                E_sub = E[trigger_set]
            gram_E = E_sub.T @ E_sub #(d_model, d_model)
            model.WQK2.weight.copy_( ( model.WOV1.weight @ gram_E))

            # Initialize WOV2 as sum t U[t] * E[t]^T where U is the unembedding matrix and E is the token embedding matrix and t is all vocab tokens
            # Remember that model.U.weight.shape = (vocab_size, d_model) and model.E.weight.shape = (vocab_size, d_model)
            # WOV2_init = torch.zeros_like(model.WOV2.weight)
            # for t in range(model.vocab_size):
            #     WOV2_init += torch.outer(model.U.weight[t], model.E.weight[t])
            # model.WOV2.weight.copy_(WOV2_init) #(d_model, d_model)
            U = model.U.weight
            model.WOV2.weight.copy_((U.T @ E)) #(d_model, d_model)

            # Initialize WF as sum_t1,t2 log P_b [t1,t2] * U[t2] * E[t1]^T where 
            # P_b[t1,t2]  = Prob(next token = t2 | current token = t1)
            # WF_init = torch.zeros_like(model.WF.weight)
            # for t1 in range(model.vocab_size):
            #     for t2 in range(model.vocab_size):
            #         WF_init += math.log(P_b[t1,t2]) * torch.outer(model.U.weight[t2], model.E.weight[t1])
            # model.WF.weight.copy_(WF_init) #(d_model, d_model)
            logPb = torch.log(P_b.clamp(min=1e-10)) #(vocab_size, vocab_size)
            model.WF.weight.copy_((U.T @ logPb.T @ E)) #(d_model, d_model)

        elif init == 'random':
            # Random initialization for all weights (except E, P, U which are already initialized and frozen)
            for param in model.parameters():
                if param.requires_grad:
                    param.copy_(torch.randn_like(param) / math.sqrt(model.d_model))
        
        return model , device
