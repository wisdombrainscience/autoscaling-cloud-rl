"""Probabilistic forecasters: DAPM and CPUForecaster (ANP-inspired), plus training.

Both follow the architectures of the original notebook; differences are listed in the
paper (history length is a parameter, trend input uses L, multi-horizon output H).
"""
import math, time, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(2)


class DAPM(nn.Module):
    """Decomposition-based Attentive Probabilistic Model.
    Seasonal branch: embed -> LSTM -> attention with learned horizon queries -> (mu_s, logvar_s)
    Trend branch   : flatten window -> MLP -> mu_t (deterministic)
    Output anchored to last observed demand; variances add (independence)."""

    def __init__(self, d_in, L, H, hid=64, heads=4, layers=2, drop=0.15):
        super().__init__()
        self.H = H
        self.emb = nn.Linear(d_in, hid)
        self.lstm = nn.LSTM(hid, hid, layers, batch_first=True, dropout=drop)
        self.q = nn.Parameter(torch.randn(H, hid) * 0.1)      # learned query per horizon step
        self.att = nn.MultiheadAttention(hid, heads, batch_first=True, dropout=drop)
        self.ln = nn.LayerNorm(hid)
        self.sdec = nn.Sequential(nn.Linear(hid, 2 * hid), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * hid, 2))
        self.tenc = nn.Sequential(nn.Linear(L * d_in, hid), nn.GELU(), nn.Dropout(drop))
        self.tdec = nn.Sequential(nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, H))

    def forward(self, x):                        # x [B,L,D]
        B = x.shape[0]
        last = x[:, -1:, 0]                       # [B,1]
        e, _ = self.lstm(F.gelu(self.emb(x)))
        q = F.gelu(self.q).unsqueeze(0).expand(B, -1, -1)
        a, _ = self.att(q, e, e)
        o = self.ln(q + a)
        s = self.sdec(o)                          # [B,H,2]
        mu_s, lv_s = s[..., 0], s[..., 1].clamp(-8, 4)
        mu_t = self.tdec(self.tenc(x.reshape(B, -1)))
        mu = last + mu_s + mu_t
        var = lv_s.exp() + 1e-4                    # trend branch deterministic: no variance term (orig. fixed log-var=0 => +1)
        return mu, var.log()


class CPUForecaster(nn.Module):
    """ANP-inspired: ReLU embed -> LSTM context encoder -> cross-attention -> separate mean/var heads.
    Mean head predicts a delta added to the last observed demand."""

    def __init__(self, d_in, H, hid=64, heads=4, layers=2, drop=0.2):
        super().__init__()
        self.H = H
        self.emb = nn.Linear(d_in, hid)
        self.lstm = nn.LSTM(hid, hid, layers, batch_first=True, dropout=drop)
        self.q = nn.Parameter(torch.randn(H, hid) * 0.1)
        self.att = nn.MultiheadAttention(hid, heads, batch_first=True)
        self.ln = nn.LayerNorm(hid)
        self.mh = nn.Sequential(nn.Linear(hid, 2 * hid), nn.ReLU(), nn.Linear(2 * hid, 1))
        self.vh = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))
        self.log_std_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B = x.shape[0]
        last = x[:, -1:, 0]
        e, _ = self.lstm(F.relu(self.emb(x)))
        q = self.q.unsqueeze(0).expand(B, -1, -1)
        a, _ = self.att(q, e, e)
        z = self.ln(a)
        mu = last + self.mh(z).squeeze(-1)
        lv = (self.vh(z).squeeze(-1) + self.log_std_scale).clamp(-8, 4)
        return mu, lv


def gauss_nll(mu, lv, y):
    return 0.5 * (lv + (y - mu) ** 2 / lv.exp() + math.log(2 * math.pi))


def windows(Z, idx, L, H):
    """Inputs Z[t-L:t], targets Z[t:t+H, 0] for each t in idx (targets truncated at series end)."""
    X = np.stack([Z[t - L:t] for t in idx])
    T = len(Z)
    Y = np.stack([np.array([Z[min(t + h, T - 1), 0] for h in range(H)]) for t in idx])
    M = np.stack([np.array([1.0 if t + h < T else 0.0 for h in range(H)]) for t in idx])
    return (torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32),
            torch.tensor(M, dtype=torch.float32))


def train_model(model, Z, tr_idx, va_idx, L, H, epochs=60, bs=64, lr=1e-3, patience=8, seed=0, log=None):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, Ytr, Mtr = windows(Z, tr_idx, L, H)
    Xva, Yva, Mva = windows(Z, va_idx, L, H)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state, bad = 1e9, None, 0
    hist = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        tl = 0.0
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]
            mu, lv = model(Xtr[b])
            loss = (gauss_nll(mu, lv, Ytr[b]) * Mtr[b]).sum() / Mtr[b].sum()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tl += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            mu, lv = model(Xva)
            vl = ((gauss_nll(mu, lv, Yva) * Mva).sum() / Mva.sum()).item()
        hist.append((tl / len(Xtr), vl))
        if log: log(f"  ep{ep:02d} train_nll={tl/len(Xtr):.4f} val_nll={vl:.4f}")
        if vl < best - 1e-4:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state)
    model.eval()
    return model, hist, best


@torch.no_grad()
def predict(model, Z, idx, L, H, bs=256):
    mus, lvs = [], []
    for i in range(0, len(idx), bs):
        X, _, _ = windows(Z, idx[i:i + bs], L, H)
        mu, lv = model(X)
        mus.append(mu.numpy()); lvs.append(lv.numpy())
    return np.concatenate(mus), np.exp(np.concatenate(lvs))  # mean, variance (z-scale)


def mix(mu_d, var_d, mu_a, var_a, w=0.5):
    """Moment-matched equal-weight Gaussian mixture (law of total variance)."""
    mu = w * mu_d + (1 - w) * mu_a
    var = w * var_d + (1 - w) * var_a + w * (1 - w) * (mu_d - mu_a) ** 2
    return mu, var
