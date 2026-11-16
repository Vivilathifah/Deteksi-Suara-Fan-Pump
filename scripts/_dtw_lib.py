import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True)
def dtw_pair(a, b):
    """DTW exact, biaya lokal euclidean, dinormalisasi dengan (n+m)."""
    n, m, d = a.shape[0], b.shape[0], a.shape[1]
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0.0
            for k in range(d):
                t = a[i - 1, k] - b[j - 1, k]
                c += t * t
            D[i, j] = np.sqrt(c) + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[n, m] / (n + m)


@njit(parallel=True, cache=True)
def dtw_matrix(X):
    N = X.shape[0]
    M = np.zeros((N, N))
    for i in prange(N):
        for j in range(i + 1, N):
            M[i, j] = dtw_pair(X[i], X[j])
    return M + M.T


def fastdtw_pair(a, b, radius=1):
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    d, _ = fastdtw(a, b, radius=radius, dist=euclidean)
    return d / (len(a) + len(b))


def _row(X, i, radius):
    return [fastdtw_pair(X[i], X[j], radius) for j in range(i + 1, len(X))]


def fastdtw_matrix(X, radius=1, n_jobs=-1):
    from joblib import Parallel, delayed
    N = len(X)
    rows = Parallel(n_jobs=n_jobs)(delayed(_row)(X, i, radius) for i in range(N - 1))
    M = np.zeros((N, N))
    for i, r in enumerate(rows):
        M[i, i + 1:] = r
    return M + M.T
