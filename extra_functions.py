import numpy as np
import scipy.signal as signal


class FilterBlock(object):
    def __init__(self, h):
        """
        Filter object to facilitate filter bank design
        :param h: 1-D numpy array representing filter response
        """
        assert(isinstance(h, np.ndarray)), 'h must be a numpy array'
        assert(h.ndim == 1), 'filter response must be one-dimensional'

        self.h = h
        self.length = len(h)

    def filter(self, x):
        """
        convolve signal x with impulse response h
        :param x: 1-D signal
        :return: filtered signal
        """
        assert(isinstance(x, np.ndarray)), 'x must be numpy array'
        assert(x.ndim == 1), 'x must be one dimensional'
        assert(len(x) >= self.length), 'x must be at least as long as filter response'

        return signal.oaconvolve(x, self.h, mode='same')


def PCA(X):
    """
    Perform Principal Component Analysis.
    This version uses SVD for better numerical performance when d >> n.

    Parameters
    --------------------
        X      -- numpy array of shape (n,d), features

    Returns
    --------------------
        U      -- numpy array of shape (d,d), d d-dimensional eigenvectors
                  each column is a unit eigenvector; columns are sorted by eigenvalue
        mu     -- numpy array of shape (d,), mean of input data X
    """
    n, d = X.shape
    mu = np.mean(X, axis=0)
    x, l, v = np.linalg.svd(X - mu)
    l = np.hstack([l, np.zeros(v.shape[0] - l.shape[0], dtype=float)])
    U = np.array([vi / 1.0 for (li, vi) in sorted(zip(l, v), reverse=True, key=lambda q: q[0])]).T
    return U, mu


def apply_PCA_from_Eig(X, U, l, mu):
    """
    Project features into lower-dimensional space.

    Parameters
    --------------------
        X  -- numpy array of shape (n,d), n d-dimensional features
        U  -- numpy array of shape (d,d), d d-dimensional eigenvectors
              each column is a unit eigenvector; columns are sorted by eigenvalue
        l  -- int, number of principal components to retain
        mu -- numpy array of shape (d,), mean of input data X

    Returns
    --------------------
        Z   -- numpy matrix of shape (n,l), n l-dimensional features
               each row is a sample, each column is one dimension of the sample
        Ul  -- numpy matrix of shape (d,l), l d-dimensional eigenvectors
               each column is a unit eigenvector; columns are sorted by eigenvalue
               (Ul is a subset of U, specifically the d-dimensional eigenvectors
                of U corresponding to largest l eigenvalues)
    """
    Ul = np.mat(U[:, :l])
    Z = (X - mu) * Ul
    # Z = X*Ul
    return Z, Ul

def get_kfolds(data, y, inds):
    data_train = data[inds[0], :]
    data_test = data[inds[1], :]
    y_train = y[inds[0]]
    y_test = y[inds[1]]
    return data_train, y_train, data_test, y_test


