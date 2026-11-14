#defining the metrics used for performance. 
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def mse(xi, xhat):
    return mean_squared_error(xi, xhat)

def r2(xi, xhat):
    return r2_score(xi, xhat, multioutput='variance_weighted')

def downstream_acc(z_train, y_train, z_test, y_test):
    scaler = StandardScaler()
    logmod = LogisticRegression(max_iter=1000)
    logmod.fit(scaler.fit_transform(z_train), y_train)
    acc = logmod.score(scaler.transform(z_test), y_test)
    return acc
