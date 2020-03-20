import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

from examples.classification.models.tiny import TINY


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


def mlp(**kwargs):
    model = TINY(**kwargs)
    return model


class TinyDataset(Dataset):
    def __init__(self):
        self.mlp = TINY()
        self.samples = torch.FloatTensor(1000, 1).uniform_(-1, 1)
        mlp_output = self.mlp(self.samples)
        # self.labels = torch.tensor([s for s in mlp_output])
        self.labels = torch.tensor([int(s[1] > s[0]) for s in mlp_output])

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
