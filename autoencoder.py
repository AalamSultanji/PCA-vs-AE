#manually writing the autoencoder class 

import numpy as np
import torch 
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, activation = True):
        super(Autoencoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, latent_dim)
        self.delinear1 = nn.Linear(latent_dim, 64)
        self.delinear2 = nn.Linear(64, 128)
        self.delinear3 = nn.Linear(128, input_dim)
        self.activation = activation

    def forward(self, x):
        if self.activation:
            x = torch.relu(self.linear1(x))
            x = torch.relu(self.linear2(x))
            x = self.linear3(x)
            decoded = torch.relu(self.delinear1(x))
            decoded = torch.relu(self.delinear2(decoded))
            decoded = self.delinear3(decoded)
        else:
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.linear3(x)
            decoded = self.delinear1(x)
            decoded = self.delinear2(decoded)
            decoded = self.delinear3(decoded)
        return decoded

