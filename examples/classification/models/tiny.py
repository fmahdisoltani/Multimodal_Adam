import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsso.utils.generate_data import *



__all__ = ['mlp']


def log_normalize(x):
    return x - torch.logsumexp(x, 0)

def gaussian_sample(mean, std, num_samples):
    D = 1
    eps = torch.randn(size=(num_samples, D))
    z = eps * std + mean
    return z

def gmm_sample(means, stds, log_pais, num_samples): #TODO: change std's to log_std's
    samples = torch.cat([gaussian_sample(mean, std, num_samples)[:, np.newaxis, :]
                         for mean, std in zip(means, stds)], axis=1)
    weights = torch.exp(log_normalize(log_pais))
    ixs = torch.multinomial(weights, num_samples, replacement=True)
    return torch.stack([samples[i, ix, :] for i, ix in enumerate(ixs)])

class TINY(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        n_hid = 1
        n_out = 2
        self.l1 = nn.Linear(1, n_hid, bias=False)
        # self.l2 = nn.Linear(n_hid, n_out, bias=False)

        s1 = torch.nn.Parameter(torch.tensor([[1.]]).T)
        # s2 = torch.nn.Parameter(torch.tensor([[1.]]).T)

        self.l1.weight = s1
        # self.l2.weight = s2

    def forward(self, x: torch.Tensor):
        x1 = self.l1(x)
        # x2 = self.l2(x1)
        return x1


class TINY_GMM(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        n_hid = 1
        n_out = 2
        self.l1 = nn.Linear(1, n_hid, bias=False)
        # s1 = torch.nn.Parameter(torch.tensor([[1.]]).T)
        # self.l1.weight = s1

        # self.l2 = nn.Linear(n_hid, n_out, bias=False)

        self.s1_mean = torch.tensor([3.])
        self.s1_std = torch.tensor([10.])
        self.s1_pai = torch.tensor([1.])

    def sample_weight(self):
        s1 = torch.nn.Parameter(
            gmm_sample(self.s1_mean, self.s1_std, self.s1_pai, 1))
        print("weights sampled:{}".format(s1.item()))
    #     s1 = torch.nn.Parameter(torch.tensor([[1.]]).T)
        self.l1.weight = s1

    def forward(self, x: torch.Tensor, gen_data=False):
        x1 = self.l1(x)
        return x1


