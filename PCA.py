'''
Author: Aalam Sultanji
Date: 5/11/26

This code is a simple implementation of PCA using first principles.

'''

import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        #mean standardising: 
        self.mean = np.mean(X, axis = 0)
        #self.std = np.std(X, axis = 0)
        X_centered = (X - self.mean)

        #covariance matrix calculation:
        cov_matrix = np.cov(X_centered.T)

        #eigen decomposition:
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        #sort eval and evec in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[sorted_indices]
        self.eigenvectors = eigenvectors[:, sorted_indices]
        self.principal_components = self.eigenvectors[:, :self.n_components]
        self.ratio = np.sum(self.eigenvalues[:self.n_components]) / np.sum(self.eigenvalues)
        return self


    def transform(self, X):
        transform = np.dot((X - self.mean), self.principal_components)
        return transform
    def inverse_transform(self, transform):
        inverse_transform = np.dot(transform, self.principal_components.T) + self.mean
        return inverse_transform

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        

