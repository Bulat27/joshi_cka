import os
import numpy as np
import time
import random
import math

import torch

from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances

def pairwise_euclidean_dist(A, eps=1e-10):
    # Fast pairwise euclidean distance on GPU
    # TODO why is this not fast?
	sqrA = torch.sum(torch.pow(A, 2), 1, keepdim=True).expand(A.shape[0], A.shape[0])
	return torch.sqrt(
		sqrA - 2*torch.mm(A, A.t()) + sqrA.t() + eps
	)


def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    return np.dot(np.dot(H, K), H)  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    # return np.dot(H, K)  # KH


def rbf(X, sigma=None):
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = np.exp(KX)
    return KX


def kernel_HSIC(X, Y, sigma):
    return np.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))


def linear_HSIC(X, Y):
    L_X = np.dot(X, X.T)
    L_Y = np.dot(Y, Y.T)
    return np.sum(centering(L_X) * centering(L_Y))


def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = np.sqrt(linear_HSIC(X, X))
    var2 = np.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)


def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = np.sqrt(kernel_HSIC(X, X, sigma))
    var2 = np.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2)


def fast_linear_CKA(X, Y):
    L_X = centering(np.dot(X, X.T))
    L_Y = centering(np.dot(Y, Y.T))
    hsic = np.sum(L_X * L_Y)
    var1 = np.sqrt(np.sum(L_X * L_X))
    var2 = np.sqrt(np.sum(L_Y * L_Y))

    return hsic / (var1 * var2)
