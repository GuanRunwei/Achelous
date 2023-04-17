import torch
import torch.nn as nn
import torch.nn.functional as F



class NllLoss(nn.Module):
    def __init__(self):
        super(NllLoss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)
        return total_loss
