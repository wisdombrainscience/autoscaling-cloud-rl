"""Shared data utilities: loading, chronological split, train-only normalisation."""
import os, json, random
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "mu.csv")
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)

STEPS_PER_DAY = 288          # 300 s sampling
L_HIST = 12                  # 1 h history (selected on validation NLL from {12,24,48,72,144,288})
H_FC = 3                     # forecast horizon (15 min)

# Chronological split (index of the *target* time step)
#   day 0            : history only
#   days 1-4  [288,1440)  : training   (forecasters + policy)
#   day 5     [1440,1728) : validation (early stopping, calibration, model selection)
#   days 6-7  [1728,2243) : test       (never touched before final evaluation)
SPLIT = {"train": (288, 1440), "val": (1440, 1728), "test": (1728, None)}


def set_seed(s):
    random.seed(s); np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
    except ImportError:
        pass


def load():
    df = pd.read_csv(DATA)
    cols = ["cpu_util_percent", "mem_util_percent", "net_in", "net_out", "disk_io_percent"]
    df = df[cols].interpolate(limit_direction="both").ffill().bfill()
    X = df.values.astype(np.float64)
    T = len(X)
    tr0, tr1 = SPLIT["train"]
    # normalisation statistics from the training span only (history day + train days)
    mu = X[:tr1].mean(0); sd = X[:tr1].std(0, ddof=1) + 1e-6
    Z = ((X - mu) / sd).astype(np.float32)
    split = {k: (a, b if b is not None else T) for k, (a, b) in SPLIT.items()}
    return dict(X=X, Z=Z, mu=mu, sd=sd, cols=cols, T=T, split=split)


def dump(obj, name):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=2, default=float)
