import torch
import math


def evaluate_model(model:torch.nn.Module, dataloader:torch.utils.data.DataLoader, device:torch.device, CE_loss:torch.nn.Module):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_target_mass = 0.0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)  # (batch_size, seq_len)
            targets = batch['target'].to(device)  # (batch_size) 
            logits = model(inputs)  # (batch_size, vocab_size)

            # Compute loss
            loss = CE_loss(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item() * inputs.size(0)

            # Compute Accuracy with top-1 prediction
            prediction = torch.argmax(logits, dim=-1)  # (batch_size)
            correct = (prediction == targets).sum().item()
            total_accuracy += correct

            # Compute Accuracy sampling from the distribution
            probs = torch.softmax(logits, dim=-1)  # (batch_size, vocab_size)
            mass_at_target = probs[torch.arange(probs.size(0)), targets].sum().item() # probability mass assigned to the correct target token
            total_target_mass += mass_at_target


    avg_loss = total_loss / len(dataloader.dataset)
    avg_accuracy = total_accuracy / len(dataloader.dataset)
    avg_accuracy_sample = total_target_mass / len(dataloader.dataset)
    return avg_loss , avg_accuracy, avg_accuracy_sample



def evaluate_model_lin_sfm(model:torch.nn.Module, dataloader:torch.utils.data.DataLoader, device:torch.device, loss_fn:torch.nn.Module,loss_type:str):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_target_mass = 0.0
    V = model.vocab_size
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)  # (batch_size, seq_len)
            target = batch['target'].to(device)  # (batch_size) 
            output = model(inputs) # (batch_size, vocab_size)

            if loss_type == 'CE':
                loss = loss_fn(output.view(-1, V), target.view(-1))
            elif loss_type == 'MSE':
                 targ_emb = model.embedding(target) # (batch_size, vocab_size)
                 loss = loss_fn(output, targ_emb)

            total_loss += loss.item() * inputs.size(0)

            # Compute Accuracy with top-1 prediction
            prediction = torch.argmax(output, dim=-1)  # (batch_size)
            correct = (prediction == target).sum().item()
            total_accuracy += correct

            # Compute Accuracy sampling from the distribution
            probs = torch.softmax(output, dim=-1)  # (batch_size, vocab_size)
            mass_at_target = probs[torch.arange(probs.size(0)), target].sum().item() # probability mass assigned to the correct target token
            total_target_mass += mass_at_target


    avg_loss = total_loss / len(dataloader.dataset)
    avg_accuracy = total_accuracy / len(dataloader.dataset)
    avg_accuracy_sample = total_target_mass / len(dataloader.dataset)
    return avg_loss , avg_accuracy, avg_accuracy_sample



def evaluate_overlap_with_teacher(model:torch.nn.Module, teacher_model:torch.nn.Module, device:torch.device):
    model.eval()
    teacher_model.eval()
    # For each parameter in the model independently, compute the cosine similarity with the corresponding parameter in the teacher model
    overlaps = {}
    for (name, param), (teacher_name, teacher_param) in zip(model.named_parameters(), teacher_model.named_parameters()):
        if param.requires_grad:
            param_flat = param.view(-1)
            teacher_param_flat = teacher_param.view(-1)
            cosine_similarity = torch.nn.functional.cosine_similarity(param_flat, teacher_param_flat, dim=0).item()
            square_distance = torch.sum((param_flat - teacher_param_flat) ** 2).item()
            overlaps[name] = [cosine_similarity,square_distance]
    return overlaps    


def evaluate_induction_bigram(model,batch,loss_fn):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        sequence = batch['sequence'].to(device) # shape (batch_size, seq_len + 1)
        input = sequence[:, :-1] # shape (batch_size, seq_len)
        target = sequence[:, 1:] # shape (batch_size, seq_len)
        counts = batch['counts'].to(device) # shape (batch_size, seq_len)
        mask = batch['mask'].to(device) # shape (batch_size, seq_len, seq_len)
        trigger_set = batch['trigger_set'].to(device) # shape (batch_size, K)
        
        # Case 1: evaluate only attention mechanism which should perform induction head

        # Evaluate Model
        logits = model(input, mask, attn=True, fc=False) # shape (batch_size, seq_len, vocab_size)

        # mask outputs only when counts >=2 and input token is in trigger set of correponding example in batch
        #   ( (B,L)->(B,L,1) -- (B,K) -> (B,1,K) ) --> (B,L,K) -any-> (B,L)  & (B,L) ==>> (B,L) 
        # valid = (input.unsqueeze(-1) == trigger_set.unsqueeze(1)).any(-1) & (counts >= 2) # for only triggers
        valid = (counts >= 2) # for all tokens with counts >=2
        
        # Extract logits and targets at masked positions
        masked_logits = logits[valid] # shape (num_masked_positions, vocab_size)
        predictions = masked_logits.argmax(dim=-1) # shape (num_masked_positions)
        masked_targets = target[valid] # shape (num_masked_positions)
        loss_ind = loss_fn(masked_logits, masked_targets) 
        # accuracy = (predictions == masked_targets).float().mean().item()

        # Case 2: evaluate only linear layer which should perform bigram statistics
        logits = model(input, mask, attn=False, fc=True) # shape (batch_size, seq_len, vocab_size)

        # Mask out the trigger tokens
        # valid = ~(input.unsqueeze(-1) == trigger_set.unsqueeze(1)).any(-1) # shape (batch_size, seq_len) # only non-triggers
       
        valid = torch.ones_like(input, dtype=torch.bool) # Valid for all positions
        valid[:,0] = False # mask out the first position since it has no bigram context

        masked_logits = logits[valid] # shape (num_masked_positions, vocab_size)
        masked_targets = target[valid] # shape (num_masked_positions)
        loss_bi = loss_fn(masked_logits, masked_targets)

         # Case 3: evaluate both
        logits = model(input, mask, attn=True, fc=True) # shape (batch_size, seq_len, vocab_size)

        # Mask out the trigger tokens
        # valid = ~(input.unsqueeze(-1) == trigger_set.unsqueeze(1)).any(-1) # shape (batch_size, seq_len) # only non-triggers
       
        valid = torch.ones_like(input, dtype=torch.bool) # Valid for all positions
        valid[:,0] = False # mask out the first position since it has no bigram context

        masked_logits = logits[valid] # shape (num_masked_positions, vocab_size)
        masked_targets = target[valid] # shape (num_masked_positions)
        loss_tot = loss_fn(masked_logits, masked_targets)

    return loss_bi.item(), loss_ind.item() , loss_tot.item()




def evaluate_dual_model(model,test_batch,loss_fn,P_b,P_u):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        sequence = test_batch['sequence'].to(device) # shape (batch_size, seq_len + 1)
        input = sequence[:, :-1] # shape (batch_size, seq_len)
        target = sequence[:, 1:] # shape (batch_size, seq_len)
        counts = test_batch['counts'].to(device) # shape (batch_size, seq_len)
        mask = test_batch['mask'].to(device) # shape (batch_size, seq_len, seq_len)
        trigg_set = test_batch['trigger_set'].to(device) # shape (batch_size, K)

        logits = model(input, mask, attn=True, fc=True) # shape (batch_size, seq_len, vocab_size)
        std_logits = logits.std()
        
        # 1: evaluate in context accuracy

        # mask outputs only when counts >=2 and input token is in trigger set of correponding example in batch
        #   ( (B,L)->(B,L,1) -- (B,K) -> (B,1,K) ) --> (B,L,K) -any-> (B,L)  & (B,L) ==>> (B,L) 
        valid = (input.unsqueeze(-1) == trigg_set.unsqueeze(1)).any(-1) & (counts >= 2) # for only triggers
        # valid = (counts >= 2) # for all tokens with counts >=2
        
        # Extract logits and targets at masked positions
        masked_logits = logits[valid] # shape (num_masked_positions, vocab_size)
        predictions = masked_logits.argmax(dim=-1) # shape (num_masked_positions)
        masked_targets = target[valid] # shape (num_masked_positions)
        # loss_tot = loss_fn(masked_logits, masked_targets) 
        accuracy = (predictions == masked_targets).float().mean().item()
        

        # Evaluate in context top 3 accuracy
        top3_predictions = masked_logits.topk(3, dim=-1).indices # shape (num_masked_positions, 3)
        accuracy_top3 = (top3_predictions == masked_targets.unsqueeze(-1)).any(dim=-1).float().mean().item()    



        # 2: evaluate KL divergence of the 'bigram' path of the model Dkl(p_model || P_b)
        

        # seq = torch.arange(model.vocab_size).to(device) # (vocab_size)
        # x = model.E(seq) # (vocab_size, d_model)
        # x = model.WF(x) # (vocab_size, d_model)
        # logits_WF = model.U(x) # (vocab_size, vocab_size)
        X0 = model.E(input) # + model.P(model.positions.to(input.device)) # (batch_size, seq_len, d_model)
        X3 = model.WF(X0) # (batch_size, seq_len, d_model)
        logits_WF = model.U(X3) # (batch_size, seq_len, vocab_size)
        std_logits_WF = logits_WF.std()
        logits_WF = logits_WF * (std_logits / std_logits_WF) # rescale logits to be on the same scale as full model logits
    
        P_model = torch.softmax(logits_WF, dim=-1) # (batch_size, seq_len, vocab_size)
        P_target = P_b[input] # (batch_size, seq_len, vocab_size)

        kl_Pb_WF = P_target * (torch.log(P_target + 1e-10) - torch.log(P_model + 1e-10)) # (batch_size, seq_len, vocab_size)
        kl_Pb_WF = kl_Pb_WF.sum(axis=-1).mean()  # scalar

        kl_WF_Pb = P_model * (torch.log(P_model + 1e-10) - torch.log(P_target + 1e-10)) # (batch_size, seq_len, vocab_size)
        kl_WF_Pb = kl_WF_Pb.sum(axis=-1).mean()
        # kl_Pb_WF = torch.sum(P_target * (torch.log(P_target + 1e-10) - torch.log(P_model + 1e-10)))/model.vocab_size # Add small constant for numerical stability
        # kl_WF_Pb = torch.sum(P_model * (torch.log(P_model + 1e-10) - torch.log(P_target + 1e-10)))/model.vocab_size # Add small constant for numerical stability

        # Evalaute kl of full logit output with bigram distribution
        model_prob = torch.softmax(logits, dim=-1) # (batch_size, seq_len, vocab_size)
        P_bigram_input = P_b[input]  # (batch_size, seq_len, vocab_size)
        kl_b_full = P_bigram_input * ( torch.log(P_bigram_input + 1e-10) - torch.log(model_prob + 1e-10)) # (batch_size, seq_len, vocab_size)
        kl_b_full = kl_b_full.sum(axis=-1).mean()  # scalar

        kl_full_b = model_prob * ( torch.log(model_prob + 1e-10) - torch.log(P_bigram_input + 1e-10)) # (batch_size, seq_len, vocab_size)
        kl_full_b = kl_full_b.sum(axis=-1).mean()  # scalar
        
        # Evalaute kl of full logit output with unigram distribution
        kl_full_u = P_u[None, None, :] * ( torch.log(P_u[None, None, :] + 1e-10) - torch.log(model_prob + 1e-10)) # (batch_size, seq_len, vocab_size)
        kl_full_u = kl_full_u.sum(axis=-1).mean()  # scalar

        # 3: evaluate loss
        # Full Loss
        valid = torch.ones_like(input, dtype=torch.bool) # Valid for all positions
        valid[:,0] = False # mask out the first position since it has no bigram context

        masked_logits = logits[valid] # shape (num_masked_positions, vocab_size)
        masked_targets = target[valid] # shape (num_masked_positions)
        std_masked_logits = masked_logits.std()
        masked_logits = masked_logits * (std_logits / std_masked_logits) # rescale logits to be on the same scale as full model logits
        loss_tot = loss_fn(masked_logits, masked_targets)

        # Loss of bigram path only: total and only focused on non-triggers
        X0 = model.E(input) + model.P(model.positions.to(input.device))
        X3 = model.WF(X0)
        logits_bigram_path = model.U(X3)
        masked_logits_bigram = logits_bigram_path[valid] # shape (num_masked_positions, vocab_size)
        std_masked_logits_bigram = masked_logits_bigram.std()
        masked_logits_bigram = masked_logits_bigram * (std_logits / std_masked_logits_bigram) # rescale logits to be on the same scale as full model logits
        loss_bigram_path_total = loss_fn(masked_logits_bigram, masked_targets)

        valid_non_triggers = valid & ~( (input.unsqueeze(-1) == trigg_set.unsqueeze(1)).any(-1) ) # only non-triggers
        masked_logits_bigram_non_trig = logits_bigram_path[valid_non_triggers]
        std_masked_logits_bigram_non_trig = masked_logits_bigram_non_trig.std()
        masked_logits_bigram_non_trig = masked_logits_bigram_non_trig * (std_logits / std_masked_logits_bigram_non_trig) # rescale logits to be on the same scale as full model logits
        masked_targets_non_trig = target[valid_non_triggers]
        loss_bigram_path_non_trig = loss_fn(masked_logits_bigram_non_trig, masked_targets_non_trig)

        # Loss of induction path only: total and focused only on triggers with counts >=2
        # First Layer
        X0 = model.E(input) + model.P(model.positions.to(input.device))
        S = model.WQK1(X0)  @ X0.transpose(-2, -1) * math.sqrt(model.d_model)  # (batch_size, seq_len, seq_len)
        S = S.masked_fill(~mask, float('-inf'))
        A = S.softmax(dim=-1)  # (batch_size, seq_len, seq_len)
        Y = A @ model.WOV1(X0)  # (batch_size, seq_len, d_model)
        X1 = X0 + Y  # ( batch_size, seq_len, d_model)
        # Second Layer
        S = model.WQK2(X1) @ X1.transpose(-2, -1) * math.sqrt(model.d_model)   # (batch_size, seq_len, seq_len)
        S = S.masked_fill(~mask, float('-inf'))
        A = S.softmax(dim=-1)  # (batch_size, seq_len, seq_len)   
        X2 = A @ model.WOV2(X1) #* model.gamma # (batch_size, seq_len, d_model)
        
        logits_induction_path = model.U(X2)
        masked_logits_induction = logits_induction_path[valid] # shape (num_masked_positions, vocab_size)
        std_masked_logits_induction = masked_logits_induction.std()
        masked_logits_induction = masked_logits_induction * (std_logits / std_masked_logits_induction) # rescale logits to be on the same scale as full model logits
        loss_induction_path_total = loss_fn(masked_logits_induction, masked_targets)

        valid_triggers = (input.unsqueeze(-1) == trigg_set.unsqueeze(1)).any(-1) & (counts >= 2) # only triggers with counts >=2
        masked_logits_induction_trig = logits_induction_path[valid_triggers]
        std_masked_logits_induction_trig = masked_logits_induction_trig.std()
        masked_logits_induction_trig = masked_logits_induction_trig * (std_logits / std_masked_logits_induction_trig) # rescale logits to be on the same scale as full model logits
        masked_targets_trig = target[valid_triggers]
        loss_induction_path_trig = loss_fn(masked_logits_induction_trig, masked_targets_trig)


    # loss_bigram_path_total = torch.tensor(0.0)
    # loss_bigram_path_non_trig = torch.tensor(0.0)
    # loss_induction_path_total = torch.tensor(0.0)
    # loss_induction_path_trig = torch.tensor(0.0)

    return accuracy, loss_tot.item(), kl_Pb_WF.item(), kl_WF_Pb.item(), accuracy_top3, kl_full_b.item(), kl_b_full.item() , kl_full_u.item() , loss_bigram_path_total.item(), loss_bigram_path_non_trig.item(), loss_induction_path_total.item(), loss_induction_path_trig.item()



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

    return pop_loss_1, pop_loss_2, pop_loss_3, trigg_per_seq

    
