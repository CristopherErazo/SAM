import torch


def evaluate_model(model:torch.nn.Module, dataloader:torch.utils.data.DataLoader, device:torch.device, CE_loss:torch.nn.Module):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
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

    avg_loss = total_loss / len(dataloader.dataset)
    avg_accuracy = total_accuracy / len(dataloader.dataset)
    return avg_loss , avg_accuracy