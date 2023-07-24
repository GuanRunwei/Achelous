import torch
import torch.nn as nn
import math


class HUncertainty(nn.Module):
    def __init__(self, task_num):
        super().__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros(task_num))

    def forward(self, loss_items):
        assert len(loss_items) == self.task_num

        total_loss = [torch.exp(-self.log_vars[i])*loss_items[i] for i in range(len(loss_items))]
        total_loss = sum(total_loss)

        # precision0 = torch.exp(-self.log_vars[0])
        # loss0 = precision0 * loss_det
        # # loss0 = precision0 * loss_det + self.log_vars[0] ** 2
        #
        # precision1 = torch.exp(-self.log_vars[1])
        # loss1 = precision1 * loss_seg
        #
        # precision2 = torch.exp(-self.log_vars[2])
        # loss2 = precision2 * loss_seg_wl

        return total_loss
