import torch


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