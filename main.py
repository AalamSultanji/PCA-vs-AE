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
from data import load_dataset
from PCA import PCA
from autoencoder import Autoencoder
from metrics import mse, downstream_acc, r2
from train import train_ae

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

SMOKE = True   # flip to False for the real run

SMOKE_TEST = {
    'latent_dims': [4],
    'seeds': [42],
    'epochs': 2,
    'n_subset': 5000,
}

if SMOKE:
    CONFIG.update(SMOKE_TEST)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def run_pca(d, data): 
    pca = PCA(n_components=d)
    start = time.perf_counter()
    pca.fit(data['X_train'])
    elapsed = time.perf_counter()-start
    Z_train, Z_test = pca.transform(data['X_train']), pca.transform(data['X_test'])
    X_hat = pca.inverse_transform(Z_test)
    metrics = {
        'mse': mse(data['X_test'], X_hat),
        'r2': r2(data['X_test'], X_hat),
        'acc': downstream_acc(Z_train, data['Y_train'], Z_test, data['Y_test'])
    }
    return {'model':'PCA', **metrics, 'd' :d, 'seed': None, 'epochs': None, 'best_val_loss': None, 'fit_time': elapsed }

def run_ae(input_dims, hidden_dims, seed, activation, loaders, data):
    set_seed(seed)
    model = Autoencoder(input_dims, hidden_dims, CONFIG['h1'], CONFIG['h2'], activation).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.MSELoss()
    start = time.perf_counter()
    trained_model, best_val_loss, epochs_run = train_ae(model, loaders['train_loader'], loaders['val_loader'], criterion, optimizer, device, epochs = CONFIG['epochs'])
    elapsed = time.perf_counter()-start
    model.eval()
    with torch.no_grad():
        Z_train = trained_model.encode(torch.from_numpy(data['X_train'])).cpu().numpy()
        Z_test = trained_model.encode(torch.from_numpy(data['X_test']))
        X_hat = trained_model.decode(Z_test).cpu().numpy()
        metrics = {
        'mse': mse(data['X_test'], X_hat),
        'r2': r2(data['X_test'], X_hat),
        'acc': downstream_acc(Z_train, data['Y_train'], Z_test, data['Y_test'])
    }
    return {'model':'nonlin AE' if activation else 'lin AE', **metrics, 'd' :hidden_dims, 'seed': seed, 'epochs': epochs_run, 'best_val_loss': best_val_loss, 'fit_time': elapsed }

def main(CONFIG):
    os.makedirs("results", exist_ok=True)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = load_dataset(0.9, 0.1, CONFIG['SPLIT_SEED'])
    if 'n_subset' in CONFIG:
        n = CONFIG['n_subset']
        X_train, Y_train = X_train[:n], Y_train[:n]
        X_val, Y_val = X_val[:n], Y_val[:n]
        X_test, Y_test = X_test[:n], Y_test[:n]

    data = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'Y_train': Y_train,
        'Y_val': Y_val,
        'Y_test': Y_test
    }

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train)),
        batch_size = CONFIG['batch_size'],
        shuffle = True)
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val)),
        batch_size = CONFIG['batch_size'],
        shuffle = False)

    loaders = {
        'train_loader': train_loader,
        'val_loader': val_loader
    }
    
    rows = []
    for d in CONFIG['latent_dims']:
        rows.append(run_pca(d, data))
        df = pd.DataFrame(rows)
        df.to_csv('results/results.csv', index = False)
        for seed in CONFIG['seeds']:
            rows.append(run_ae(CONFIG['input_dims'], d, seed, False, loaders, data))
            df = pd.DataFrame(rows)
            df.to_csv('results/results.csv', index = False)
            rows.append(run_ae(CONFIG['input_dims'], d , seed, True, loaders, data))
            df = pd.DataFrame(rows)
            df.to_csv('results/results.csv', index = False)

if __name__ == "__main__":
    main(CONFIG)