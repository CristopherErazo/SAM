import torch
import math

def compute_kl(P, Q):
    "P and Q are distributions over the same support, shape (..., vocab_size) = (B,L,V) usually"
    return (P * (torch.log(P + 1e-10) - torch.log(Q + 1e-10))).sum(dim=-1).mean()



def compute_entropies_and_dkl(P_b:torch.Tensor,P_u:torch.Tensor):
    """ 
    Compute KL divergences between P_b and uniform distribution, P_u and uniform distribution, as well as the entropies of P_b and P_u.
    """
    vocab_size = P_u.shape[0]
    # Average dkl between bigram distribution and uniform 1/V
    kl_Pb_uniform = (P_b * (torch.log(P_b + 1e-10) - math.log(1.0/vocab_size + 1e-10))).sum(dim=-1).mean().item()
    
    # dkl between unigram and uniform 1/V
    kl_Pu_uniform = (P_u * (torch.log(P_u + 1e-10) - math.log(1.0/vocab_size + 1e-10))).sum().item()
   
    # Average entropy of bigram distribution
    entropy_Pb = -(P_b * torch.log(P_b + 1e-10)).sum(dim=-1).mean().item()
    entropy_Pu = -(P_u * torch.log(P_u + 1e-10)).sum().item()
    max_entropy = math.log(vocab_size)
    # return kl_Pb_uniform, kl_Pu_uniform, entropy_Pb, entropy_Pu, max_entropy
    # Return a dicitonary of the computed values for better readability
    return {
        "kl_Pb_uniform": kl_Pb_uniform,
        "kl_Pu_uniform": kl_Pu_uniform,
        "entropy_Pb": entropy_Pb,
        "entropy_Pu": entropy_Pu,
        "max_entropy": max_entropy
    }



def optimal_pop_losses(test_batch,P_b,p0=0.99):
    vocab_size = P_b.shape[-1]
    device = P_b.device
    input = test_batch['sequence'][:,1:-1].to(device) # shape (batch_size, seq_len-2)
    output = test_batch['sequence'][:,2:].to(device) # shape (batch_size, seq_len-2)
    is_trigg = test_batch['is_trigg'][:,1:].to(device) # shape (batch_size, seq_len-2)
    seq_len = input.shape[1]+2

    trigg_per_seq = is_trigg.sum(dim=-1).float().mean().item()/(seq_len-2)

    H_cond = -torch.sum(P_b * torch.log(P_b + 1e-10), dim=-1)  # shape (seq_len,)

    # Case 1: As if the model makes prediction with the dual 'teacher' model (up to p0 mass to avoid log(0) issues)
    input_eval = input[is_trigg==0] # shape (num_non_trigg_tokens,)
    loss = H_cond[input_eval]  # shape (num_non_trigg_tokens,)
    loss = loss.mean().item()
    pop_loss_1 = trigg_per_seq*(-math.log(p0)) + (1-trigg_per_seq)*loss

    # Case 2: As if the model makes prediction with induction head only, agnostic about frequencies
    pop_loss_2 = trigg_per_seq*(-math.log(p0)) + (1-trigg_per_seq)*math.log(vocab_size)

    # Case 3: As if the model makes predicitons with bigram statistics only
    loss_no_trig = H_cond[input]  # shape (batch_size, seq_len-2)

    # For loss2 evaluate the condition probability for each input,output pair 
    loss_trig = -torch.log(P_b[input,output]+1e-10)  # shape (batch_size, seq_len-2)

    # Where input is a trigger set loss = loss_trig, else loss = loss_no_trig
    loss = loss_no_trig.clone()
    loss[is_trigg==1] = loss_trig[is_trigg==1]
    pop_loss_3 = loss.mean().item()

    # return pop_loss_1, pop_loss_2, pop_loss_3, trigg_per_seq
    return {
        "pop_loss_dual": pop_loss_1,
        "pop_loss_induction": pop_loss_2,
        "pop_loss_bigram": pop_loss_3,
        "trigg_per_seq": trigg_per_seq
    }


class EvalContext:
    def __init__(self, model, batch, loss_fn, P_b, P_u):
        device = next(model.parameters()).device

        # Ground Variables
        self.sequence = batch['sequence'].to(device) # shape (B, L+1)
        self.input = self.sequence[:, :-1] # shape (B, L)
        self.target = self.sequence[:, 1:] # shape (B, L)
        self.counts = batch['counts'].to(device) # shape (B, L)
        self.mask = batch['mask'].to(device) # shape (B, L, L)
        self.trigger_set = batch['trigger_set'].to(device) # shape (B, K)
        self.only_triggers = (self.input.unsqueeze(-1) == self.trigger_set.unsqueeze(1)).any(-1) & (self.counts >= 2) # shape (B, L)
        self.only_non_triggers = ~( (self.input.unsqueeze(-1) == self.trigger_set.unsqueeze(1)).any(-1) ) # shape (B, L)
        self.all = torch.ones_like(self.input, dtype=torch.bool) # shape (B, L)
        
        self.model = model
        self.loss_fn = loss_fn
        self.P_b = P_b.to(device) # shape (V, V)
        self.P_u = P_u.to(device) # shape (V,)

        with torch.no_grad():
            self.logits = model(self.input, self.mask, path='full') # shape (B, L, V)
            self.logits_bigram = model(self.input, self.mask, path='bigram') # shape (B, L, V)
            self.logits_induction = model(self.input, self.mask, path='induction') # shape (B, L, V)

            self.model_prob = torch.softmax(self.logits, dim=-1) # shape (B, L, V)
            self.model_prob_bigram = torch.softmax(self.logits_bigram, dim=-1) # shape (B, L, V)
            self.std_logits = self.logits.std().item()




class IC_TopKAccuracy:
    def __init__(self, k):
        self.k = k
        self.name = f"top{self.k}_accuracy"

    def __call__(self, ctx):
        logits = ctx.logits[ctx.only_triggers]  # shape (num_masked_positions, vocab_size)
        targets = ctx.target[ctx.only_triggers] # shape (num_masked_positions,)

        topk = logits.topk(self.k, dim=-1).indices # shape (num_masked_positions, k)
        correct = (topk == targets.unsqueeze(-1)).any(dim=-1)

        return correct.float().mean().item()

class KLMetric:
    def __init__(self, name = 'kl_b_full', P_fn = lambda ctx: ctx.P_b[ctx.input] , Q_fn = lambda ctx: ctx.model_prob):
        """
        P_fn, Q_fn: functions that take ctx and return distributions
        """
        self.name = name
        self.P_fn = P_fn
        self.Q_fn = Q_fn

    def __call__(self, ctx):
        P = self.P_fn(ctx)
        Q = self.Q_fn(ctx)

        kl = compute_kl(P, Q)
        return kl.item()


class LossMetric:
    def __init__(self, name = 'loss', logits_fn = lambda ctx: ctx.logits[ctx.all], target_fn = lambda ctx: ctx.target[ctx.all], rescale=False):
        """
        logits_fn: function(ctx) -> logits tensor (num_masked_positions,V)
        target_fn: function(ctx) -> target tensor (num_masked_positions,)
        rescale: whether to normalize logits to match global std
        """
        self.name = name
        self.logits_fn = logits_fn
        self.target_fn = target_fn
        self.rescale = rescale

    def __call__(self, ctx):
        logits_masked = self.logits_fn(ctx)
        targets_masked = self.target_fn(ctx)

        # Optional rescaling
        if self.rescale:
            std_global = ctx.std_logits
            std_masked = logits_masked.std()
            logits_masked = logits_masked * (std_global / std_masked)
    
        # Compute loss
        return ctx.loss_fn(logits_masked, targets_masked).item()

class LogitMeanMetric:
    def __init__(self, name = 'logit', logits_fn = lambda ctx: ctx.logits[ctx.all]):
        self.name = name
        self.logits_fn = logits_fn

    def __call__(self, ctx):
        logits_masked = self.logits_fn(ctx)
        
        return logits_masked.mean().item()

class LogitStdMetric:
    def __init__(self, name = 'logit_std', logits_fn = lambda ctx: ctx.logits[ctx.all]):
        self.name = name
        self.logits_fn = logits_fn

    def __call__(self, ctx):
        logits_masked = self.logits_fn(ctx)
        
        return logits_masked.std().item()

class Evaluator:
    def __init__(self, metrics):
        self.metrics = metrics

    def evaluate(self, model, batch, loss_fn, P_b, P_u):
        model.eval()

        ctx = EvalContext(model, batch, loss_fn, P_b, P_u)

        results = {}
        for metric in self.metrics:
            results[metric.name] = metric(ctx)

        return results
    

def get_attention_patterns(model, test_batch, path, device, n_test = 5):
    """ 
    Get the attention patterns of the model on the given batch. This function is used for evaluation during training.    
    """
    sequence = test_batch['sequence'][:n_test].to(device) # shape (n_test, seq_len + 1)
    input = sequence[:, :-1] # shape (n_test, seq_len)
    mask = test_batch['mask'][:n_test].to(device) # shape (n_test, seq_len, seq_len)

    with torch.no_grad():
        output = model.full_output(input,mask, path = path)
        attn1 = output.get('A1', None) # shape (n_test, seq_len, seq_len)
        attn2 = output.get('A2', None) # shape (n_test, seq_len, seq_len)
    return {'attn1': attn1, 'attn2': attn2}
    


def evaluate_model(model,batch,loss_fn,path,device):
    """ 
    Evaluate the model on the given batch and return the computed loss. This function is used for evaluation during training.    
    """
    # Evaluate model on the dual 
    sequence = batch['sequence'].to(device) # shape (batch_size, seq_len + 1)
    input = sequence[:, :-1] # shape (batch_size, seq_len)
    target = sequence[:, 1:] # shape (batch_size, seq_len)
    mask = batch['mask'].to(device) # shape (batch_size, seq_len, seq_len)
    counts = batch['counts'].to(device) # shape (batch_size, seq_len)
    trigg_set = batch['trigger_set'].to(device) # shape (batch_size, K)

    logits = model(input, mask, path='full' if path=='full_trigg' else path) # shape (batch_size, seq_len, vocab_size)
    
    if path == 'full':
        all = torch.ones_like(input, dtype=torch.bool) # shape (B, L)
        logits_masked = logits[all] # shape (num_masked_positions, vocab_size)
        target_masked = target[all] # shape (num_masked_positions,)
    elif path == 'bigram':
        only_non_triggers = ~( (input.unsqueeze(-1) == trigg_set.unsqueeze(1)).any(-1) ) # shape (B, L)
        logits_masked = logits[only_non_triggers] # shape (num_masked_positions, vocab_size)
        target_masked = target[only_non_triggers] # shape (num_masked_positions,)
    elif path == 'induction':
        only_triggers = (input.unsqueeze(-1) == trigg_set.unsqueeze(1)).any(-1) & (counts >= 2) # shape (B, L)
        logits_masked = logits[only_triggers] # shape (num_masked_positions, vocab_size)
        target_masked = target[only_triggers] # shape (num_masked_positions,)
    elif path == 'full_trigg':
        only_triggers = (input.unsqueeze(-1) == trigg_set.unsqueeze(1)).any(-1) & (counts >= 2) # shape (B, L)
        logits_masked = logits[only_triggers] # shape (num_masked_positions, vocab_size)
        target_masked = target[only_triggers] # shape (num_masked_positions,)
    else:
        raise ValueError("Invalid path type. Options are 'full', 'bigram', 'induction', 'full_trigg'.")

    # Compute loss and update model
    loss_trigg = loss_fn(logits_masked, target_masked)
    return loss_trigg