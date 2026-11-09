#manually writing the autoencoder class 

import numpy as np
import torch 
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim,  h1, h2, activation = False,):
        super(Autoencoder, self).__init__()
        act = nn.ReLU() if activation else nn.Identity() #to account for nonlinearity 
        h1 = max(h1, latent_dim) #to ensure that the bottleneck does not happen before the latent dimension, else curves flatten
        h2 = max(h2, latent_dim)
        #encoder block
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            act,
            nn.Linear(h1, h2),
            act,
            nn.Linear(h2, latent_dim)
        )
        #decoder block
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            act,
            nn.Linear(h2, h1),
            act,
            nn.Linear(h1, input_dim)
        )
    

    def encode(self, x):
        x = self.encoder(x)
        return x
    def decode(self, x):
        x = self.decoder(x)
        return x
    def forward(self, x):
        decoded = self.decode(self.encode(x))
        return decoded


