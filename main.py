#main file to run for pca and AE comparison

import numpy as np
from PCA import PCA
from autoencoder import Autoencoder
import torch
import torchvision
from torchvision import datasets, transforms


SEED = 42

train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=transforms.ToTensor())

def mse(xi, xhat, ntest):
    return np.sum((xi-xhat)**2)/ntest
def relative_gap(mse_pca, mse_ae):
    return (mse_pca - mse_ae)/mse_pca

def reconstuction_error(xi, xhat, xbar):
    return 1-((np.sum(xi-xhat)**2)/np.sum(xi-xbar)**2)

latent_dims = [2,4,8,16,32,64]
for d in latent_dims:
    pca = PCA(d)
    for _ in range(10):
        np.random.seed(SEED)
        linear_ae = Autoencoder()