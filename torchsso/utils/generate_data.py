import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset


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


class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        n_hid = 1
        n_out = 1
        self.l1 = nn.Linear(1, n_hid, bias=False)
        self.l2 = nn.Linear(n_hid, n_out, bias=False)
        # s1 = torch.nn.Parameter(gmm_sample(torch.tensor([-2., 2.]), torch.tensor([.8, .8]),
        #                             torch.log(torch.tensor([.5, .5])),1))
        # s2 = torch.nn.Parameter(gmm_sample(torch.tensor([-1., 1.]), torch.tensor([1.8, 1.4]),
        #                             torch.log(torch.tensor([.6, .4])), num_samples=1))

        s1 = torch.nn.Parameter(torch.tensor([[2.]]), requires_grad=False)
        s2 = torch.nn.Parameter(torch.tensor([[1.]]), requires_grad=False)

        print("D" * 20)
        print(s1)
        print(s2)
        print("D"*20)
        self.l1.weight = s1
        self.l2.weight = s2

    def forward(self, x: torch.Tensor):
        x1 = F.relu(self.l1(x))
        print(self.l1(x))
        x2 = self.l2(x1)
        return x2


def mlp(**kwargs):
    model = MLP(**kwargs)
    return model


class TinyDataset(Dataset):
    def __init__(self):
        self.mlp = MLP()
        self.samples = self.mlp(torch.FloatTensor(1000, 1).uniform_(-1, 1))
        self.labels = torch.tensor([int(s > 0) for s in self.samples])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (self.samples[idx], self.labels[idx])


if __name__ == '__main__':
    q_means = torch.nn.Parameter(torch.tensor([-2., 2.]))
    q_stds = torch.nn.Parameter(torch.tensor([.8, .8]))
    q_log_pais = torch.nn.Parameter(torch.log(torch.tensor([.5, .5])))
    model = mlp()

    # sample_params
    y = model(torch.Tensor([1.]))
    print(y)
