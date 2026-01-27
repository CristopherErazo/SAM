from torch import nn
import torch
import math

from .activations import get_activation

class SingleIndex(nn.Module):
    """Single Index Model:  y = f(Wx/sqrt(d)) """
    def __init__(self, d: int, function_specs):
        super().__init__()
        self.W = nn.Linear(d, 1 , bias=False)
        self.activation = get_activation(function_specs)
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

    for p in teacher.parameters():
        p.requires_grad_(False)

    # w_teacher = teacher.W.weight.data.clone().to(device)
    w_teacher = torch.cat([p.view(-1) for p in teacher.parameters()])

    return teacher, student, w_teacher