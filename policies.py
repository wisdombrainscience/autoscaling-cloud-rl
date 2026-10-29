"""Rule-based / predictive baselines and the recurrent actor-critic."""
import numpy as np
import torch, torch.nn as nn
from scipy.stats import norm


# ------------------------------------------------------------------ baselines
class Threshold:
    """Reactive step-scaling (EC2 step scaling / original-paper baseline, corrected)."""
    name = "Threshold"

    def __init__(self, up=0.9, low=0.7, cooldown=3):
        self.up, self.low, self.cd = up, low, cooldown

    def reset(self, env):
        self.last = np.full(env.B, -99)

    def act(self, env):
        a = np.ones(env.B, int)
        rho, n, tau = env.rho_prev, env.n, env.tau
        pend = env.pend.sum(1)
        out = (rho > self.up * tau) & (pend == 0)
        rho_after = rho * n / np.maximum(n - 1, 1)
        inn = (~out) & (n > 1) & (rho_after < self.low * tau) & (env.k - self.last >= self.cd)
        a[out] = 2; a[inn] = 0
        self.last[out | inn] = env.k
        return a


class HPA:
    """Kubernetes HPA target tracking: desired = ceil(n * rho / target), 10% tolerance,
    scale-down stabilisation window; limited to +-1 VM per step like all methods."""
    name = "HPA"

    def __init__(self, target=0.85, tol=0.1, window=5):
        self.target, self.tol, self.w = target, tol, window

    def reset(self, env):
        self.hist = []

    def act(self, env):
        tgt = self.target * env.tau
        ratio = env.rho_prev / tgt
        desired = np.where(np.abs(ratio - 1) <= self.tol, env.n, np.ceil(env.n * ratio)).astype(int)
        self.hist = (self.hist + [desired])[-self.w:]
        desired_down = np.max(np.stack(self.hist), 0)
        cap = env.n + env.pend.sum(1)
        a = np.ones(env.B, int)
        a[desired > cap] = 2
        a[(desired_down < env.n) & (desired <= cap)] = 0
        return a


class PredictiveQuantile:
    """Forecast-driven provisioning (AWS-predictive-scaling / MPC-style one-step rule):
    keep committed capacity >= q-quantile of forecast demand / tau."""
    name = "PredQuantile"

    def __init__(self, q=0.95, q_in=0.995):
        self.z, self.z_in = norm.ppf(q), norm.ppf(q_in)   # hysteresis: scale in only if safe at q_in

    def reset(self, env):
        pass

    def _need(self, env, z, hs=(0, 1)):
        yprev = np.maximum(env.y[env.t - 1], 1e-3)
        q = np.max(np.stack([env._quantile_real(h, z) for h in hs]), 0)
        return np.ceil(env.D_prev * q / yprev / env.tau).astype(int)

    def act(self, env):
        need = self._need(env, self.z)
        need_in = self._need(env, self.z_in, (0, 1, 2))
        cap = env.n + env.pend.sum(1)
        a = np.ones(env.B, int)
        a[cap < need] = 2
        a[(cap - 1 >= need_in) & (env.n - 1 >= need_in) & (cap >= need)] = 0
        return a


class StaticPeak:
    """Static provisioning sized for the episode peak in hindsight (oracle upper bound on cost)."""
    name = "StaticPeak"

    def reset(self, env):
        D = np.stack([env._demand_at(env.t + k, k) for k in range(env.T)], 1)
        env.n = np.clip(np.ceil(D.max(1) / env.tau), 1, 40).astype(int)

    def act(self, env):
        return np.ones(env.B, int)


# ------------------------------------------------------------------ recurrent actor-critic
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim=3, hid=64, recurrent=True):
        super().__init__()
        self.recurrent = recurrent
        self.base = nn.Sequential(nn.Linear(obs_dim, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh())
        if recurrent:
            self.rnn = nn.GRU(hid, hid, batch_first=True)
        else:
            self.rnn = None
            self.mid = nn.Sequential(nn.Linear(hid, hid), nn.Tanh())
        self.pi = nn.Linear(hid, act_dim)
        self.v = nn.Linear(hid, 1)
        self.hid = hid
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2)); nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.orthogonal_(self.v.weight, 1.0)

    def init_h(self, B):
        return torch.zeros(1, B, self.hid)

    def forward(self, x, h):
        """x [B,T,obs] -> logits [B,T,A], v [B,T], h, features"""
        f = self.base(x)
        if self.recurrent:
            y, h = self.rnn(f, h)
        else:
            y = self.mid(f)
        return self.pi(y), self.v(y).squeeze(-1), h, y


class RLPolicy:
    def __init__(self, net, greedy=True, name="RL"):
        self.net, self.greedy, self.name = net, greedy, name

    def reset(self, env):
        self.h = self.net.init_h(env.B)
        self.feats = []

    @torch.no_grad()
    def act(self, env, obs=None):
        o = torch.tensor(env.obs() if obs is None else obs).unsqueeze(1)
        lg, _, self.h, y = self.net(o, self.h)
        self.feats.append(y[:, 0].numpy())
        if self.greedy:
            return lg[:, 0].argmax(-1).numpy()
        return torch.distributions.Categorical(logits=lg[:, 0]).sample().numpy()
