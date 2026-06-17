from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def center(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return X - X.mean(axis=0, keepdims=True)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = center(X)
    Y = center(Y)
    XTY = X.T @ Y
    XTX = X.T @ X
    YTY = Y.T @ Y
    denom = np.linalg.norm(XTX, ord="fro") * np.linalg.norm(YTY, ord="fro")
    if denom <= 0:
        return 0.0
    return float((np.linalg.norm(XTY, ord="fro") ** 2) / denom)


def _pca_reduce(X: np.ndarray, variance_threshold: float = 0.99, max_components: int = 50) -> np.ndarray:
    X = center(X)
    n_samples, n_features = X.shape
    n_components = min(n_samples, n_features)
    if n_components <= 1:
        return X
    pca = PCA(n_components=n_components, svd_solver="full", random_state=0)
    Z = pca.fit_transform(X)
    evr = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(evr, variance_threshold) + 1)
    k = max(1, min(k, max_components, Z.shape[1]))
    return Z[:, :k]


def svcca(X: np.ndarray, Y: np.ndarray, variance_threshold: float = 0.99, max_components: int = 50) -> float:
    Xr = _pca_reduce(X, variance_threshold=variance_threshold, max_components=max_components)
    Yr = _pca_reduce(Y, variance_threshold=variance_threshold, max_components=max_components)
    n = min(Xr.shape[0], Yr.shape[0], Xr.shape[1], Yr.shape[1])
    if n < 2:
        return 0.0
    Xr = Xr[:, :n]
    Yr = Yr[:, :n]
    cca = CCA(n_components=n, max_iter=1000)
    try:
        cca.fit(Xr, Yr)
        U, V = cca.transform(Xr, Yr)
        corrs = []
        for i in range(U.shape[1]):
            u = U[:, i]
            v = V[:, i]
            if np.std(u) < 1e-12 or np.std(v) < 1e-12:
                continue
            c = np.corrcoef(u, v)[0, 1]
            if np.isfinite(c):
                corrs.append(abs(float(c)))
        return float(np.mean(corrs)) if corrs else 0.0
    except Exception:
        return 0.0


def twonn_intrinsic_dimension(X: np.ndarray, centered: bool = True) -> float:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 5:
        return 0.0
    if centered:
        X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    nbrs = NearestNeighbors(n_neighbors=3, algorithm="auto").fit(X)
    dists, _ = nbrs.kneighbors(X)
    r1 = dists[:, 1]
    r2 = dists[:, 2]
    mask = (r1 > 1e-12) & (r2 > r1)
    mu = r2[mask] / r1[mask]
    mu = mu[np.isfinite(mu) & (mu > 1.0)]
    if mu.size < 5:
        return 0.0
    mu = np.sort(mu)
    n = mu.size
    F = np.arange(1, n + 1, dtype=np.float64) / (n + 1.0)
    x = np.log(mu)
    y = -np.log(np.clip(1.0 - F, 1e-12, 1.0))
    # Robust fit on the central bulk
    lo = int(0.1 * n)
    hi = max(lo + 5, int(0.9 * n))
    x_fit = x[lo:hi]
    y_fit = y[lo:hi]
    if x_fit.size < 5:
        x_fit, y_fit = x, y
    try:
        slope, _ = np.polyfit(x_fit, y_fit, 1)
        if np.isfinite(slope) and slope > 0:
            return float(slope)
    except Exception:
        pass
    # fallback: slope through origin
    denom = np.dot(x_fit, x_fit)
    if denom <= 0:
        return 0.0
    slope = float(np.dot(x_fit, y_fit) / denom)
    return max(slope, 0.0)


def pairwise_matrix(
    names: List[str],
    data: Dict[str, np.ndarray],
    fn,
) -> np.ndarray:
    n = len(names)
    out = np.zeros((n, n), dtype=np.float64)
    for i, a in enumerate(names):
        out[i, i] = 1.0 if fn in {linear_cka, svcca} else 0.0
        for j in range(i + 1, n):
            val = fn(data[a], data[names[j]])
            out[i, j] = out[j, i] = float(val)
    return out


def adjacent_series(names: List[str], data: Dict[str, np.ndarray], fn) -> np.ndarray:
    vals = []
    for i in range(len(names) - 1):
        vals.append(fn(data[names[i]], data[names[i + 1]]))
    return np.asarray(vals, dtype=np.float64)
