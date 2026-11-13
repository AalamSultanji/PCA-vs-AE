#main file to run for pca and AE comparison

import numpy as np
import os
import time
import json
import copy
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader 
from torchvision import datasets
from data import load_dataset
from PCA import PCA
from autoencoder import Autoencoder
from metrics import mse, downstream_acc, r2
from train import train_ae, evaluate, EarlyStopping

CONFIG = {
    'input_dims': 784,
    'latent_dims': [2, 4, 8, 16, 32, 64],
    'seeds': [42, 123, 1123, 11123, 111123],
    'h1': 128,
    'h2': 64,
    'lr': 0.001,
    'batch_size': 256,
    'epochs': 50,
    'train_size': 0.9,
    'val_size': 0.1,
    'SPLIT_SEED': 42
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def run_pca(d, data): 
    pca = PCA(n_components=d)
    pca.fit(data)
    Z_train, Z_test = pca.transform(), pca.transform(X_test)
    X_hat = pca.inverse_transform(Z_test)
    metrics = {
        'mse_pca': mse(X_test, X_hat),
        'r2_pca': r2(X_test, X_hat),
        'acc_pca': downstream_acc(Z_train, Y_train, Z_test, Y_test)
    }
    return metrics

def run_ae(input_dims, hidden_dims, seed, activation, X_train, X_val, X_test, Y_train, Y_test):
    set_seed(seed)
    model = Autoencoder(input_dims, hidden_dims, CONFIG['h1'], CONFIG['h2'], activation).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.MSELoss()
    trained_model, history = train_ae(model, X_train, X_val, criterion, optimizer, device, epochs = CONFIG['epochs'])
    model.eval()
    with torch.no_grad():
        Z_train = model.encode(X_train).to(device)
        Z_test = model.encode(X_test).to(device)
        X_hat = model.decode(Z_test).cpu().numpy()
        metrics = {
            'mse_pca': mse(X_test, X_hat),
            'r2_pca': r2(X_test, X_hat),
            'acc_pca': downstream_acc(Z_train, Y_train, Z_test, Y_test)
        }
    return {model:'AE', **metrics, d:hidden_dims,    }

def main():
    X_train, X_val, X_test, Y_train, Y_val, Y_test = load_dataset(0.9, 0.1, CONFIG['SPLIT_SEED'])
    train_data = TensorDataset(X_train, Y_train)

    for d in CONFIG['latent_dims']:
        run_pca(d, X_train, X_test, Y_train, Y_test)
        for seed in CONFIG['seeds']:
            run_ae(CONFIG['input_dims'], seed, False, X_train, X_val, X_test, Y_train, Y_test)
            run_ae(CONFIG['input_dims'], seed, True, X_train, X_val, X_test, Y_train, Y_test)

if __name__ == "__main__":
    main()