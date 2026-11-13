import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split



SPLIT_SEED = 42
def load_dataset(train_size=0.9, val_size=0.1, split_seed=SPLIT_SEED):
        #can be changed to any other dataset
    train_set = datasets.FashionMNIST(root='data', train=True, download=True, transform=None)
    test_set = datasets.FashionMNIST(root='data', train=False, download=True, transform=None)

    train_set.data = (train_set.data)/255.0.float().reshape(-1, 28*28)
    test_set.data = (test_set.data)/255.0.float().reshape(-1, 28*28)

    #splitting the sets
    total = len(train_set)
    indices = list(range(total))
    np.random.seed(split_seed)
    np.random.shuffle(indices)
    train_len = int(train_size * total)
    val_len = int(val_size * total)

    train_index = indices[:train_len]
    val_index = indices[train_len:val_len+train_len]

    #creating the splits
    X_train = train_set.data[train_index]
    X_val = train_set.data[val_index]
    X_test = test_set.data
    Y_train = train_set.targets[train_index]
    Y_val = train_set.targets[val_index]
    Y_test = test_set.targets

    return X_train, X_val, X_test, Y_train, Y_val, Y_test