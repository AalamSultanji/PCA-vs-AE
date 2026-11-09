#defining the metrics used for performance. 
import numpy as np

def mse(xi, xhat, ntest):
    return np.sum((xi-xhat)**2)/ntest
def relative_gap(mse_pca, mse_ae):
    return (mse_pca - mse_ae)/mse_pca

def reconstuction_error(xi, xhat, xbar):
    return 1-((np.sum(xi-xhat)**2)/np.sum(xi-xbar)**2)