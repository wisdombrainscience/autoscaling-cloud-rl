"""Stage 2: train RL variants (multi-seed) and evaluate all controllers on held-out days.

usage: python run_control.py <variant> <seed>   (trains + evaluates one RL run)
       python run_control.py baselines          (evaluates rule/predictive baselines)
"""
import sys, os, time, json, pickle
import numpy as np, torch
from common import *
from env import VecEnv, test_tasks, Task
from policies import *
import ppo

D = load(); T_all = D["T"]
C = np.load(os.path.join(OUT, "forecast_cache.npz"))
Y = D["X"][:, 0]
MU0, SD0 = D["mu"][0], D["sd"][0]
TR = (D["split"]["train"][0], D["split"]["train"][1])
TEST = (D["split"]["test"][0], T_all)
VAL = D["split"]["val"]

FEATS = VecEnv.feat_names
def mask_without(names):
    return np.array([0.0 if f in names else 1.0 for f in FEATS], np.float32)

FC_FEATS = ["dmu1", "dmu2", "dmu3", "sig1", "sig2", "sig3", "risk_m1", "risk_0", "risk_p1"]
VARIANTS = {
    # name: (env kwargs, recurrent, fixed_task)
    "Ours":          (dict(), True, False),
    "MLP":           (dict(), False, False),
    "NoForecast":    (dict(feature_mask=mask_without(FC_FEATS), risk_w=0.0), True, False),
    "PointForecast": (dict(point_forecast=True), True, False),
    "NoRiskShaping": (dict(risk_w=0.0), True, False),
    "SingleTask":    (dict(), True, True),
    "Oracle":        (dict(oracle=True), True, False),
    "NoBC":          (dict(), True, False),
    "RidgeForecast": (dict(cache="ridge"), True, False),
}
NO_BC = {"NoBC"}
ppo.HP["bc_epochs"] = int(os.environ.get("BC_EPOCHS", 150))


CR = np.load(os.path.join(OUT, "forecast_cache_ridge.npz"))
def make_env(cache="ens", **kw):
    c = CR if cache == "ridge" else C
    return lambda: VecEnv(Y, c["mu"], c["var"], MU0, SD0, **kw)


def evaluate(policy, env_fn, tasks, rng_range, record=False):
    env = env_fn()
    T = rng_range[1] - rng_range[0]
    env.reset(tasks, [rng_range[0]] * len(tasks), T)
    policy.reset(env)
    o = env.obs()
    keys = ["rho", "viol", "over", "n", "act", "D", "r_sla", "r_cost", "r_sw", "r_risk"]
    rec = {k: [] for k in keys}; R = []
    for k in range(T):
        a = policy.act(env)
        o, r, done, info = env.step(a)
        R.append(r)
        for kk in keys: rec[kk].append(info[kk])
    rec = {k: np.stack(v, 1) for k, v in rec.items()}; R = np.stack(R, 1)
    util = np.minimum(rec["rho"], 1.0)
    m = dict(
        ret=R.sum(1), viol=rec["viol"].mean(1) * 100, over=rec["over"].mean(1) * 100,
        vms=rec["n"].mean(1), util=util.mean(1) * 100,
        actions=(rec["act"] != 1).sum(1), rho_std=rec["rho"].std(1) * 100,
        cost=-rec["r_cost"].sum(1),
        # over-provisioning: capacity above the ideal ceil(D/tau)
        overprov=(rec["n"] - np.ceil(rec["D"] / np.array([t.tau for t in tasks])[:, None])).mean(1),
    )
    m = {k: v.astype(float).tolist() for k, v in m.items()}
    if record:
        m["trace"] = {k: rec[k].tolist() for k in ["rho", "n", "D", "act", "viol"]}
        if isinstance(policy, RLPolicy) and policy.feats:
            m["hidden"] = np.stack(policy.feats, 1).astype(np.float32)
    return m


def eval_all(policy, env_fn, tag):
    out = {}
    for kind in ["id", "ood"]:
        out[kind] = evaluate(policy, env_fn, test_tasks(kind), TEST)
    # validation (model selection only)
    out["val"] = evaluate(policy, env_fn, test_tasks("id", n=12, seed=5), VAL)
    return out


if __name__ == "__main__":
    os.makedirs(os.path.join(OUT, "runs"), exist_ok=True)
    if sys.argv[1] == "baselines":
        res = {}; grid_log = {}
        val_tasks = test_tasks("id", n=12, seed=5)
        def tune(name, cands):
            sc = [(float(np.mean(evaluate(p, make_env(), val_tasks, VAL)["ret"])), i) for i, p in enumerate(cands)]
            grid_log[name] = [(desc, s_) for (s_, _), desc in zip(sc, [getattr(c, "desc", "") for c in cands])]
            return cands[max(sc)[1]]
        def mk(cls, desc, **kw):
            p = cls(**kw); p.desc = desc; return p
        cands = {
          "Threshold": [mk(Threshold, f"up={u},low={l}", up=u, low=l) for u in (0.8, 0.9, 1.0) for l in (0.5, 0.6, 0.7, 0.8) if l < u],
          "HPA": [mk(HPA, f"target={t},tol={tl}", target=t, tol=tl) for t in (0.7, 0.8, 0.9, 1.0) for tl in (0.05, 0.1)],
          "PredQuantile": [mk(PredictiveQuantile, f"q={q},q_in={qi}", q=q, q_in=qi) for q in (0.8, 0.9, 0.95, 0.99) for qi in (0.99, 0.995, 0.999) if qi > q],
        }
        pols = [tune(k, v) for k, v in cands.items()] + [StaticPeak()]
        for pol in pols:
            nm = pol.name
            res[nm] = eval_all(pol, make_env(), nm); res[nm]["selected"] = getattr(pol, "desc", "")
            print(nm, {k: round(float(np.mean(v)), 3) for k, v in res[nm]["id"].items()},
                  "| OOD viol", round(float(np.mean(res[nm]["ood"]["viol"])), 2), flush=True)
        res["_grid"] = grid_log
        dump(res, "runs/baselines.json")
        sys.exit()

    variant, seed = sys.argv[1], int(sys.argv[2])
    kw, rec, fixed = VARIANTS[variant]
    kw = dict(kw)
    t0 = time.time()
    val_tasks = test_tasks("id", n=12, seed=5)
    def val_fn(net):   # greedy return on validation day (model selection only)
        return float(np.mean(evaluate(RLPolicy(net, greedy=True), make_env(**kw), val_tasks, VAL)["ret"]))
    if variant in NO_BC: ppo.HP["bc_epochs"] = 0
    net, curve = ppo.train(make_env(**kw), TR, seed, recurrent=rec, fixed_task=fixed, val_fn=val_fn,
                           expert=PredictiveQuantile(0.95, 0.999),
                           log=lambda m: print(f"[{variant}/{seed}]{m}", flush=True))
    torch.save(net.state_dict(), os.path.join(OUT, "runs", f"{variant}_s{seed}.pt"))
    res = eval_all(RLPolicy(net, greedy=True), make_env(**kw), variant)
    res["curve"] = curve; res["train_time_s"] = time.time() - t0
    dump(res, f"runs/{variant}_s{seed}.json")
    print(variant, seed, "ID viol", np.mean(res["id"]["viol"]), "vms", np.mean(res["id"]["vms"]),
          "OOD viol", np.mean(res["ood"]["viol"]), f"{time.time()-t0:.0f}s", flush=True)
