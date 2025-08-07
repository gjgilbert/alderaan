__all__ = ["psi_N", "psi_T", "psi_G", "F_iter", "z_iter", "emus_weights"]


import numpy as np
import scipy.linalg as linalg


def psi_N(x, norm=1.0):
    """
    Bias funciton for 'N' (non-grazing) umbrella; see Gilbert 2022
    """
    x_ = np.atleast_1d(x)
    psi = np.zeros_like(x_)
    psi[(x_ < 2) * (x_ >= 1)] = x_[(x_ < 2) * (x_ >= 1)] - 1
    psi[(x_ >= 2)] = 1.0
    psi = psi.clip(0, 1)

    return psi / norm


def psi_T(x, norm=1.0):
    """
    Bias funciton for 'T' (transition) umbrella; see Gilbert 2022
    """
    x_ = np.atleast_1d(x)
    psi = np.zeros_like(x_)
    psi[(x_ >= 0) * (x_ < 1)] = x_[(x_ >= 0) * (x_ < 1)]
    psi[(x_ >= 1) * (x_ < 2)] = 2 - x_[(x_ >= 1) * (x_ < 2)]
    psi = psi.clip(0, 1)

    return psi / norm


def psi_G(x, norm=1.0):
    """
    Bias funciton for 'G' (grazing) umbrella; see Gilbert 2022
    """
    x_ = np.atleast_1d(x)
    psi = np.zeros_like(x_)
    psi[(x_ < 0) * (x_ >= -1)] = 1 + x_[(x_ < 0) * (x_ >= -1)]
    psi[(x_ >= 0)] = 1.0 - x_[x_ >= 0]
    psi = psi.clip(0, 1)

    return psi / norm


def F_iter(z, psi_fxns, coordinates, weights=None):
    """
    Helper function for .emus_weights
    """
    Nwin = len(psi_fxns)
    F = np.zeros((Nwin, Nwin))

    if weights is None:
        weights = []
        for j, c_ in enumerate(coordinates):
            weights.append(np.ones(len(c_)) / len(c_))

    for i in range(Nwin):
        denom = 0.0
        for k in range(Nwin):
            denom += psi_fxns[k](coordinates[i]) / z[k]
        for j in range(Nwin):
            num = psi_fxns[j](coordinates[i]) / z[i]
            F[i, j] = np.sum(weights[i] * num / denom)

    return F


def z_iter(F, tol=1.0e-10, max_iter=100):
    """
    Helper function for emus_weights
    """
    # stationary distribution is the last column of QR factorization
    M = np.eye(len(F)) - F
    q, r = linalg.qr(M)
    z = q[:, -1]
    z /= np.sum(z)

    # polish solution using power method.
    for itr in range(max_iter):
        znew = np.dot(z, F)
        tv = np.abs(znew[z > 0] - z[z > 0])
        tv = tv / z[z > 0]

        maxresid = np.max(tv)
        if maxresid < tol:
            break
        else:
            z = znew

    # return normalized (by convention)
    return z / np.sum(z)


def emus_weights(
    psi_fxns, coordinates, weights=None, nMBAR=20, tol=1e-10, max_iter=100
):
    """
    Calculate umbrella weights using the EMUS algorithm
    See Matthews+ 2017 (https://ui.adsabs.harvard.edu/abs/2018MNRAS.480.4069M/abstract)

    Parameters
    ----------
    psi_fxns : list
        list of callable 1D functions, each takes a 1D vector of coordinates and returns normalized psi
    coordinates : list
        list of 1D arrays, each array is the coordinates for samples from a single umbrella
    weights : list
        list of 1D arrays, each array is sample weights corresponding to coordinates (default = None)
    nMBAR : int
        number of MBAR iterations to perform
    tol : float
        convergence tolerance
    max_iter
        maximum number of polishing iterations to perform on z

    Returns
    -------
    z : ndarray
        normalized window weights
    """
    Nwin = len(psi_fxns)

    z = np.zeros(Nwin)

    for i, psi in enumerate(psi_fxns):
        if weights is None:
            z[i] = np.mean(psi(coordinates[i]))
        else:
            if np.abs(np.sum(weights[i]) - 1) > 1e-8:
                raise ValueError("sample weights must sum to unity")
            else:
                z[i] = np.sum(psi(coordinates[i]) * weights[i])

    z /= np.sum(z)

    for n in range(nMBAR):
        F = F_iter(z, psi_fxns, coordinates, weights)
        z = z_iter(F)

        # perform self-consistent polishing until convergence
        for n in range(nMBAR):
            z_old = np.copy(z)
            z_old[z_old < 1e-100] = 1e-100

            F = F_iter(z, psi_fxns, coordinates, weights)
            z = z_iter(F, tol, max_iter)

            # check if we have converged
            if np.max(np.abs(z - z_old) / z_old) < tol:
                break

    return z
