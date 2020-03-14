import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['mlp']


class TINY(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        n_hid = 1
        n_out = 2
        self.l1 = nn.Linear(1, n_hid, bias=False)
        self.l2 = nn.Linear(n_hid, n_out, bias=False)

        s1 = torch.nn.Parameter(torch.tensor([2.]))
        s2 = torch.nn.Parameter(torch.tensor([[-1., -2.]]).T)

        # self.l1.weight = s1
        # self.l2.weight = s2

    def forward(self, x: torch.Tensor):
        x1 =self.l1(x)
        x2 = self.l2(x1)
        return x2


