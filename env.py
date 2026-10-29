"""Vectorised trace-driven autoscaling environment.

Workload: CPU-utilisation trace y_t (%) of the Alibaba 2018 machine; service demand in
VM-units D_t = kappa * y_t / 10 * (1 + b_t) * (1 + eps_t), with hidden task parameters
  kappa  (workload-to-capacity mapping), d (scale-out provisioning delay, steps),
  sigma  (multiplicative noise), burst process b_t.
Visible task parameters: SLA target tau, cost weight c.
Capacity n_t VMs; load ratio rho_t = D_t / n_t; SLA violated when rho_t > tau.
Actions: 0 scale-in (-1 VM, immediate), 1 hold, 2 scale-out (+1 VM after d steps).
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.stats import norm

Z99 = norm.ppf(0.99)
N_MIN, N_MAX = 1, 40


@dataclass
class Task:
    kappa: float
    tau: float
    delay: int
    sigma: float
    burst_p: float
    cost: float
    burst_amp: tuple = (0.3, 0.8)
    seed: int = 0


def sample_train_task(rng):
    return Task(kappa=rng.uniform(0.8, 1.6), tau=float(rng.choice([0.6, 0.7, 0.8])),
                delay=int(rng.choice([0, 1, 2])), sigma=rng.uniform(0.02, 0.06),
                burst_p=rng.uniform(0.0, 0.01), cost=float(rng.choice([0.2, 0.3, 0.4])),
                seed=int(rng.integers(1 << 30)))


def fixed_train_task(rng):
    return Task(1.2, 0.7, 1, 0.04, 0.005, 0.3, seed=int(rng.integers(1 << 30)))


def test_tasks(kind, n=24, seed=777):
    rng = np.random.default_rng(seed + (0 if kind == "id" else 1))
    out = []
    for i in range(n):
        if kind == "id":
            t = Task(kappa=[0.9, 1.2, 1.5][i % 3], tau=[0.6, 0.7, 0.8][(i // 3) % 3],
                     delay=[0, 1, 2][(i // 9) % 3] if i < 18 else int(rng.choice([0, 1, 2])),
                     sigma=0.04, burst_p=0.005, cost=0.3, seed=1000 + i)
        else:  # distribution shift: larger amplitude, longer delay, heavier bursts/noise
            t = Task(kappa=[1.8, 2.2][i % 2], tau=[0.6, 0.7, 0.8][(i // 2) % 3],
                     delay=3, sigma=0.08, burst_p=0.03, cost=0.3,
                     burst_amp=(0.5, 1.0), seed=2000 + i)
        out.append(t)
    return out


class VecEnv:
    """B parallel episodes over trace interval [t0, t0+T)."""
    feat_names = ["y_prev", "dmu1", "dmu2", "dmu3", "sig1", "sig2", "sig3", "rho_prev", "n", "pending",
                  "tau", "cost", "risk_m1", "risk_0", "risk_p1", "a_prev", "r_prev", "tod_sin", "tod_cos"]

    def __init__(self, y, mu_z, var_z, mu0, sd0, risk_w=0.5, feature_mask=None, point_forecast=False,
                 oracle=False):
        self.y = y                      # real-scale CPU %, length T
        self.mu_z, self.sd_z = mu_z, np.sqrt(var_z)
        self.mu0, self.sd0 = mu0, sd0
        self.risk_w = risk_w
        self.mask = feature_mask        # np.array of 0/1 for obs features, or None
        self.point = point_forecast
        self.oracle = oracle
        self.obs_dim = len(self.feat_names) + (2 if oracle else 0)

    # ---------------------------------------------------------------- setup
    def reset(self, tasks, starts, T):
        B = len(tasks); self.B, self.T = B, T
        self.tasks = tasks
        self.t = np.array(starts, dtype=np.int64)
        self.k = 0
        g = lambda a: np.array([getattr(tk, a) for tk in tasks])
        self.kappa, self.tau, self.delay = g("kappa"), g("tau"), g("delay").astype(int)
        self.sigma, self.cost = g("sigma"), g("cost")
        # exogenous randomness for the whole episode (common random numbers across policies)
        self.eps = np.zeros((B, T)); self.burst = np.zeros((B, T))
        for i, tk in enumerate(tasks):
            r = np.random.default_rng(tk.seed)
            self.eps[i] = r.normal(0, tk.sigma, T)
            j = 0
            while j < T:
                if r.random() < tk.burst_p:
                    ln = int(r.integers(3, 13)); amp = r.uniform(*tk.burst_amp)
                    prof = amp * np.sin(np.linspace(0, np.pi, ln))
                    self.burst[i, j:j + ln] += prof[:max(0, min(ln, T - j))]
                    j += ln
                else:
                    j += 1
        # initial capacity: provisioned for the last observed demand (all methods identical)
        D_prev = self._demand_at(self.t - 1, None)
        self.n = np.clip(np.ceil(D_prev / self.tau), N_MIN, N_MAX).astype(int)
        self.pend = np.zeros((B, 4), dtype=int)       # pending[:, j] arrives in j+1 steps
        self.D_prev = D_prev
        self.rho_prev = D_prev / self.n
        self.a_prev = np.ones(B, dtype=int); self.r_prev = np.zeros(B)
        return self.obs()

    def _demand_at(self, t, k):
        base = self.kappa * self.y[t] / 10.0
        if k is None:
            return base
        return base * (1 + self.burst[:, k]) * (1 + self.eps[:, k])

    # ---------------------------------------------------------------- observation
    def _quantile_real(self, h, z=Z99):
        m = self.mu_z[self.t, h]
        s = 0.0 if self.point else self.sd_z[self.t, h]
        return (m + z * s) * self.sd0 + self.mu0

    def projected_risk(self, cap):
        """q99 projected load ratio for capacity cap minus tau, using forecast ratio
        q99(y_{t+h}) / y_{t-1} applied to the observed demand D_{t-1} (max over h=1,2)."""
        yprev = np.maximum(self.y[self.t - 1], 1e-3)
        q = np.maximum(self._quantile_real(0), self._quantile_real(1))
        Dq = self.D_prev * q / yprev
        return Dq / np.maximum(cap, 1) - self.tau

    def obs(self):
        tz = (self.y[self.t - 1] - self.mu0) / self.sd0
        dmu = [self.mu_z[self.t, h] - tz for h in range(3)]
        sig = [(0 * self.sd_z[self.t, h] if self.point else self.sd_z[self.t, h]) for h in range(3)]
        cap = self.n + self.pend.sum(1)
        risks = [self.projected_risk(cap + d) for d in (-1, 0, 1)]
        tod = 2 * np.pi * (self.t % 288) / 288
        f = [tz, *dmu, *sig, self.rho_prev, self.n / 10, self.pend.sum(1) / 10, self.tau, self.cost,
             *risks, self.a_prev - 1, self.r_prev, np.sin(tod), np.cos(tod)]
        o = np.stack(f, 1).astype(np.float32)
        if self.mask is not None:
            o = o * self.mask[None]
        if self.oracle:
            o = np.concatenate([o, np.stack([self.kappa - 1.2, self.delay / 2], 1).astype(np.float32)], 1)
        return o

    # ---------------------------------------------------------------- dynamics
    def step(self, a):
        a = np.asarray(a, dtype=int)
        # arrivals of previously requested capacity
        self.n = np.minimum(self.n + self.pend[:, 0], N_MAX)
        self.pend = np.concatenate([self.pend[:, 1:], np.zeros((self.B, 1), int)], 1)
        cap = self.n + self.pend.sum(1)
        inn = (a == 0) & (self.n > N_MIN)
        self.n = self.n - inn
        out = (a == 2) & (cap < N_MAX)
        imm = out & (self.delay == 0)
        self.n = self.n + imm
        for i in np.where(out & (self.delay > 0))[0]:
            self.pend[i, self.delay[i] - 1] += 1
        cap = self.n + self.pend.sum(1)
        risk = np.maximum(0, self.projected_risk(cap)) / self.tau
        D = self._demand_at(self.t, self.k)
        rho = D / self.n
        over = np.maximum(0, rho - self.tau) / self.tau
        viol = (rho > self.tau).astype(float)
        r_sla = -(viol + 4.0 * over)
        r_cost = -self.cost * self.n / 10.0
        r_sw = -0.05 * (a != 1)
        r_risk = -self.risk_w * risk
        r = r_sla + r_cost + r_sw + r_risk
        info = dict(rho=rho, viol=viol, over=over, n=self.n.copy(), act=a.copy(), D=D,
                    r_sla=r_sla, r_cost=r_cost, r_sw=r_sw, r_risk=r_risk)
        self.D_prev, self.rho_prev, self.a_prev, self.r_prev = D, rho, a, r
        self.t = self.t + 1; self.k += 1
        done = self.k >= self.T
        return (None if done else self.obs()), r, done, info
