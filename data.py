import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np


SPLIT_SEED = 42
def load_dataset(val_frac, split_seed = SPLIT_SEED):
    train_transform=transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0], std = [1])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0], std = [1])
    ])
    
    train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=train_transform)
    test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=test_transform)
    
    #splitting the training data into training and validation sets
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(val_frac * num_train))
    
    np.random.seed(split_seed)
    np.random.shuffle(indices)
    
    train_idx, val_idx = indices[split:], indices[:split]
    
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
    
    return train_data, test_data, train_sampler, val_sampler