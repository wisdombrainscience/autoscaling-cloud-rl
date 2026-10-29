"""Ridge-AR forecasts (forecaster-swap experiment), variance calibrated on validation."""
import numpy as np
from common import *
D = load(); Z = D["Z"]; T = D["T"]; s = D["split"]; H = H_FC
tr = np.arange(*s["train"]); va = np.arange(*s["val"]); ALL = np.arange(288, T)
def feats(ii):
    f = [Z[ii - l, 0] for l in range(1, 13)] + [Z[ii - 1, j] for j in range(1, 5)]
    f += [Z[ii - 288, 0], Z[ii - 287, 0], np.ones(len(ii))]
    return np.stack(f, 1)
def tgt(ii): return np.stack([Z[np.minimum(ii + h, T - 1), 0] for h in range(H)], 1)
X = feats(tr); W = np.linalg.solve(X.T @ X + np.eye(X.shape[1]), X.T @ tgt(tr))
rv = ((tgt(tr) - X @ W) ** 2).mean(0)
tau = float(np.mean((tgt(va) - feats(va) @ W) ** 2 / rv))
mu = np.full((T, H), np.nan, np.float32); var = mu.copy()
mu[ALL] = feats(ALL) @ W; var[ALL] = rv * tau
np.savez(os.path.join(OUT, "forecast_cache_ridge.npz"), mu=mu, var=var, tau=tau)
print("ridge tau", tau)
