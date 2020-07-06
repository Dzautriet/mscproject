# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:38:14 2020

@author: Zhe Cao
"""

import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from train import call_train, call_train_blobs

k = 10
n_features = 2
X, y = make_blobs(n_samples=12000, centers=k, n_features=n_features, random_state=42)
# y = np.eye(k)[y]
X_train, X_vali, X_test = X[:10000], X[10000:11000], X[11000:]
y_train, y_vali, y_test = y[:10000], y[10000:11000], y[11000:]
valid_range = np.arange(10000)

# pred_train, pred_vali, vali_acc, test_acc, model = call_train_blobs(X_train, valid_range, y_train, X_vali, y_vali, y_vali, X_test, y_test)
# pred_train, pred_vali, vali_acc, test_acc, model = call_train_blobs(X_train, valid_range, y_train_wmv, X_vali, y_vali_corrupt, y_vali, X_test, y_test)


plt.scatter(x=X_train[:, 0], y=X_train[:, 1], c=y_train)
plt.show()
plt.scatter(x=X_vali[:, 0], y=X_vali[:, 1], c=y_vali)
plt.show()
plt.scatter(x=X_test[:, 0], y=X_test[:, 1], c=y_test)
plt.show()