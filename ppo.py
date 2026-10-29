"""PPO with full-episode BPTT for the recurrent policy; task-randomised (RL^2-style) training."""
import time
import numpy as np
import torch, torch.nn as nn
from env import VecEnv, sample_train_task, fixed_train_task
from policies import ActorCritic

torch.set_num_threads(1)

HP = dict(iters=260, B=16, T=288, epochs=4, n_mb=4, clip=0.2, vf=0.5, ent0=0.01, ent1=0.001,
          lr=3e-4, gamma=0.99, lam=0.95, grad_clip=0.5, hid=64, bc_epochs=0)


class RetNorm:
    """Scale rewards by the running std of discounted returns (Engstrom et al., 2020)."""
    def __init__(self, gamma):
        self.g, self.n, self.m, self.s2 = gamma, 1e-4, 0.0, 1.0
    def __call__(self, rew):                      # rew [B,T]
        B, T = rew.shape; ret = np.zeros(B); rets = []
        for k in range(T):
            ret = ret * self.g + rew[:, k]; rets.append(ret.copy())
        x = np.concatenate(rets); bm, bv, bn = x.mean(), x.var(), len(x)
        d = bm - self.m; tot = self.n + bn
        self.m += d * bn / tot
        self.s2 = (self.s2 * self.n + bv * bn + d * d * self.n * bn / tot) / tot; self.n = tot
        return rew / (np.sqrt(self.s2) + 1e-8)


def behaviour_clone(net, env, expert, train_range, rng, epochs, B=16, T=288, lr=1e-3, log=None):
    """Warm start: imitate a forecast-driven expert on randomised training tasks (full-sequence CE)."""
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lo, hi = train_range
    for ep in range(epochs):
        tasks = [sample_train_task(rng) for _ in range(B)]
        o = env.reset(tasks, rng.integers(lo, hi - T, B), T); expert.reset(env)
        O = np.zeros((B, T, env.obs_dim), np.float32); A = np.zeros((B, T), int)
        for k in range(T):
            a = expert.act(env); O[:, k] = o; A[:, k] = a
            o, _, _, _ = env.step(a)
        lg, _, _, _ = net(torch.tensor(O), net.init_h(B))
        loss = nn.functional.cross_entropy(lg.reshape(-1, 3), torch.tensor(A).reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
        if log and ep % 25 == 0: log(f"  bc{ep:03d} ce={loss.item():.3f}")


def train(make_env, train_range, seed, recurrent=True, fixed_task=False, hp=HP, log=None, expert=None, val_fn=None):
    import copy
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    env = make_env()
    net = ActorCritic(env.obs_dim, 3, hp["hid"], recurrent)
    opt = torch.optim.Adam(net.parameters(), lr=hp["lr"], eps=1e-5)
    B, T = hp["B"], hp["T"]
    lo, hi = train_range
    curve = []
    t_start = time.time()
    rn = RetNorm(hp["gamma"]); best = None
    if expert is not None and hp["bc_epochs"] > 0:
        behaviour_clone(net, env, expert, train_range, rng, hp["bc_epochs"], B, T, log=log)
    for it in range(hp["iters"]):
        frac = it / hp["iters"]
        for g in opt.param_groups: g["lr"] = hp["lr"] * (1 - 0.9 * frac)
        ent_c = hp["ent0"] + (hp["ent1"] - hp["ent0"]) * frac
        tasks = [(fixed_train_task if fixed_task else sample_train_task)(rng) for _ in range(B)]
        starts = rng.integers(lo, hi - T, B)
        o = env.reset(tasks, starts, T)
        obs = np.zeros((B, T, env.obs_dim), np.float32); act = np.zeros((B, T), int)
        rew = np.zeros((B, T), np.float32); logp = np.zeros((B, T), np.float32); val = np.zeros((B, T), np.float32)
        viol = np.zeros((B, T)); nvm = np.zeros((B, T))
        h = net.init_h(B)
        with torch.no_grad():
            for k in range(T):
                ot = torch.tensor(o).unsqueeze(1)
                lg, v, h, _ = net(ot, h)
                dist = torch.distributions.Categorical(logits=lg[:, 0])
                a = dist.sample()
                obs[:, k] = o; act[:, k] = a.numpy(); logp[:, k] = dist.log_prob(a).numpy(); val[:, k] = v[:, 0].numpy()
                o, r, done, info = env.step(a.numpy())
                rew[:, k] = r; viol[:, k] = info["viol"]; nvm[:, k] = info["n"]
        # GAE; episode truncated at T -> bootstrap with last value (time-limit, not terminal)
        rew_raw = rew; rew = rn(rew).astype(np.float32)
        adv = np.zeros_like(rew); last = 0.0
        for k in reversed(range(T)):
            nv = val[:, k + 1] if k + 1 < T else val[:, k]
            delta = rew[:, k] + hp["gamma"] * nv - val[:, k]
            last = delta + hp["gamma"] * hp["lam"] * last
            adv[:, k] = last
        ret = adv + val
        tO, tA, tLP, tAdv, tRet, tV = map(torch.tensor, (obs, act, logp, adv, ret, val))
        idx = np.arange(B)
        for ep in range(hp["epochs"]):
            rng.shuffle(idx)
            for mb in np.array_split(idx, hp["n_mb"]):
                lg, v, _, _ = net(tO[mb], net.init_h(len(mb)))
                dist = torch.distributions.Categorical(logits=lg)
                lp = dist.log_prob(tA[mb])
                A = tAdv[mb]; A = (A - A.mean()) / (A.std() + 1e-8)
                ratio = torch.exp(lp - tLP[mb])
                pl = -torch.min(ratio * A, ratio.clamp(1 - hp["clip"], 1 + hp["clip"]) * A).mean()
                vl = 0.5 * ((v - tRet[mb]) ** 2).mean()
                loss = pl + hp["vf"] * vl - ent_c * dist.entropy().mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), hp["grad_clip"]); opt.step()
        curve.append(dict(it=it, ret=float(rew_raw.sum(1).mean()), viol=float(viol.mean() * 100),
                          vms=float(nvm.mean()), t=time.time() - t_start))
        if val_fn is not None and (it % 20 == 19 or it == hp["iters"] - 1):
            sc = val_fn(net); curve[-1]["val"] = sc
            if best is None or sc > best[0]:
                best = (sc, copy.deepcopy(net.state_dict()), it)
        if log and (it % 20 == 0 or it == hp["iters"] - 1):
            log(f"  it{it:03d} ret={curve[-1]['ret']:.1f} viol={curve[-1]['viol']:.2f}% vms={curve[-1]['vms']:.2f} "
                f"[{curve[-1]['t']:.0f}s]")
    if best is not None:
        net.load_state_dict(best[1]); curve.append(dict(selected_it=best[2], val=best[0]))
        if log: log(f"  selected it{best[2]} val={best[0]:.2f}")
    return net, curve
