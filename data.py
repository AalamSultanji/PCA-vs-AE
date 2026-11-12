import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset



SPLIT_SEED = 42
def load_dataset(batch_size=256, train_size=0.9, val_size=0.1, split_seed=SPLIT_SEED):

    #can be changed to any other dataset
    train_set = datasets.FashionMNIST(root='data', train=True, download=True, transform=None)
    test_set = datasets.FashionMNIST(root='data', train=False, download=True, transform=None)

    train_set.data = (float(train_set.data)/255.0).reshape(-1, 28*28).flatten()
    test_set.data = (float(test_set.data)/255.0).reshape(-1, 28*28).flatten()

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
    train_split = Subset(train_set, train_index)
    val_split = Subset(train_set, val_index)

    #creating the dataloaders
    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)
    return train_loader, val_loader, test_loader