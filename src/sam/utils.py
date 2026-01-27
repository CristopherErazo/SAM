from torch import nn
import torch

def generate_dataset(teacher: nn.Module, n:int=1000, d:int=20, eps:float = 0.0, device="cpu"):
    """Generate a dataset (x,y) where y = teacher(x) + noise
    
    Args:
        teacher (nn.Module): the teacher network
        n (int): number of samples
        d (int): input dimension
        eps (float): noise standard deviation
        device (str): device to use
    Returns:
        x (torch.Tensor): input data of shape (n,d)
        y (torch.Tensor): output data of shape (n,1)
    """
    x = torch.randn(n, d, device=device)
    with torch.no_grad():
        y = teacher(x) 
        y = y + eps * torch.randn_like(y)
    return x, y