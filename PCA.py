import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        #mean standardising: 
        self.mean = np.mean(X, axis = 0)
        X_centered = X- self.mean

        #covariance matrix calculation:
        cov_matrix = np.cov(X_centered, rowvar = False)

        #eigen decomposition:
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        #sort eval and evec in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[sorted_indices]
        self.eigenvectors = eigenvectors[:, sorted_indices]
        self.principal_components = self.eigenvectors[:, :self.n_components]


    def fit_transform(self, X):
        self.fit(X)
