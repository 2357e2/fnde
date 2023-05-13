import torch
import torch.nn as nn

class RelativeLoss(nn.Module):
    """Takes the mean fractional relative loss between tensor-valued predicted and actual."""
    def __init__(self):
        super(RelativeLoss, self).__init__()
    
    def forward(self, pred, actual):
        abs_loss = torch.abs((pred - actual)/actual)
        
        return torch.mean(abs_loss)