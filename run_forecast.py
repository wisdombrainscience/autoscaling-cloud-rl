"""Stage 1: train forecasters, evaluate on held-out days, and cache forecasts for the control stage.

Outputs
  results/forecast_metrics.json   per-model metrics on test days (3 seeds) + baselines
  results/forecast_cache.npz      ensemble mean/std (z-scale) for every t >= L, out-of-fold on train days
  results/forecast_curves.json    training curves (seed 0)
"""
import sys, time, json
import numpy as np, torch
from scipy.stats import norm
from common import *
from forecasters import *

log = lambda m: (print(m, flush=True))
D = load(); Z = D["Z"]; s = D["split"]; T = D["T"]
L, H = L_HIST, H_FC
sd0, mu0 = D["sd"][0], D["mu"][0]
tr = np.arange(*s["train"]); va = np.arange(*s["val"]); te = np.arange(*s["test"])
ALL = np.arange(L, T)
SEEDS = [0, 1, 2]


def crps_g(y, m, sdv):
    z = (y - m) / sdv
    return sdv * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def metrics(y, m, var, mask):
    """y,m,var in z-scale [N,H]; returned in %CPU units (NLL in %CPU units too)."""
    out = {}
    for h in range(H):
        k = mask[:, h] > 0
        yy, mm, ss = y[k, h] * sd0 + mu0, m[k, h] * sd0 + mu0, np.sqrt(var[k, h]) * sd0
        z90 = norm.ppf(0.95)
        out[f"h{h+1}"] = dict(
            MAE=float(np.mean(np.abs(yy - mm))),
            RMSE=float(np.sqrt(np.mean((yy - mm) ** 2))),
            NLL=float(np.mean(0.5 * np.log(2 * np.pi * ss ** 2) + 0.5 * (yy - mm) ** 2 / ss ** 2)),
            CRPS=float(np.mean(crps_g(yy, mm, ss))),
            PICP90=float(np.mean(np.abs(yy - mm) <= z90 * ss) * 100),
            PIW90=float(np.mean(2 * z90 * ss)),
            COV99U=float(np.mean(yy <= mm + norm.ppf(0.99) * ss) * 100),   # one-sided q99 coverage
        )
    return out


def targets(idx):
    _, Y, M = windows(Z, idx, L, H)
    return Y.numpy(), M.numpy()


def temp_fit(y, m, var, mask):
    """scalar variance temperature minimising validation NLL (closed form)."""
    k = mask > 0
    return float(np.mean(((y - m) ** 2 / var)[k]))


# ------------------------------------------------------------------ baselines
def baseline_preds(idx):
    y_last = Z[idx - 1, 0]
    persist = np.repeat(y_last[:, None], H, 1)
    seas = np.stack([Z[idx + h - 288, 0] for h in range(H)], 1)
    return persist, seas


def ridge_ar(train_idx, idx, lags=12, lam=1.0):
    def feats(ii):
        f = [Z[ii - l, 0] for l in range(1, lags + 1)]
        f += [Z[ii - 1, j] for j in range(1, Z.shape[1])]
        f += [Z[ii - 288, 0], Z[ii - 287, 0], np.ones(len(ii))]
        return np.stack(f, 1)
    Ytr, Mtr = targets(train_idx)
    Xtr = feats(train_idx)
    W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ Ytr)
    res_var = ((Ytr - Xtr @ W) ** 2 * Mtr).sum(0) / Mtr.sum(0)
    return feats(idx) @ W, np.repeat(res_var[None], len(idx), 0)


results = {"test": {}, "val_temperature": {}, "notes": {}}
Yte, Mte = targets(te); Yva, Mva = targets(va); Ytr_, Mtr_ = targets(tr)

# naive baselines with homoscedastic variance estimated on train residuals
p_tr, s_tr = baseline_preds(tr); p_te, s_te = baseline_preds(te)
for name, ptr, pte in [("Persistence", p_tr, p_te), ("SeasonalNaive", s_tr, s_te)]:
    rv = ((Ytr_ - ptr) ** 2 * Mtr_).sum(0) / Mtr_.sum(0)
    results["test"][name] = metrics(Yte, pte, np.repeat(rv[None], len(te), 0), Mte)
m_ar, v_ar = ridge_ar(tr, te)
results["test"]["RidgeAR"] = metrics(Yte, m_ar, v_ar, Mte)
log("baselines done")

# ------------------------------------------------------------------ neural models, 3 seeds
curves = {}
cache_full = {}
per_seed = {k: [] for k in ["DAPM", "DAPM-origvar", "CPUForecaster", "Ensemble-naive", "Ensemble", "Ensemble-cal"]}
for seed in SEEDS:
    t0 = time.time()
    md, hd, _ = train_model(DAPM(5, L, H), Z, tr, va, L, H, seed=seed)
    ma, ha, _ = train_model(CPUForecaster(5, H), Z, tr, va, L, H, seed=seed)
    if seed == 0:
        curves = {"DAPM": hd, "CPUForecaster": ha}
        torch.save({"dapm": md.state_dict(), "cpuf": ma.state_dict()}, os.path.join(OUT, "forecasters_seed0.pt"))
    out = {}
    for split_name, idx in [("val", va), ("test", te), ("all", ALL)]:
        out[split_name] = (*predict(md, Z, idx, L, H), *predict(ma, Z, idx, L, H))
    mdv, vdv, mav, vav = out["val"]; mdt, vdt, mat, vat = out["test"]
    per_seed["DAPM"].append(metrics(Yte, mdt, vdt, Mte))
    per_seed["DAPM-origvar"].append(metrics(Yte, mdt, vdt + 1.0, Mte))
    per_seed["CPUForecaster"].append(metrics(Yte, mat, vat, Mte))
    # original notebook ensemble: average of variances, no disagreement term
    per_seed["Ensemble-naive"].append(metrics(Yte, 0.5 * (mdt + mat), 0.5 * (vdt + vat), Mte))
    mm_t, vv_t = mix(mdt, vdt, mat, vat)
    mm_v, vv_v = mix(mdv, vdv, mav, vav)
    tau = temp_fit(Yva, mm_v, vv_v, Mva)
    results["val_temperature"][f"seed{seed}"] = tau
    per_seed["Ensemble"].append(metrics(Yte, mm_t, vv_t, Mte))
    per_seed["Ensemble-cal"].append(metrics(Yte, mm_t, tau * vv_t, Mte))
    if seed == 0:
        cache_full = dict(all=out["all"], tau=tau)
    log(f"seed {seed} done in {time.time()-t0:.0f}s  temp={tau:.3f}  ens-cal h1 {per_seed['Ensemble-cal'][-1]['h1']}")


def agg(lst):
    o = {}
    for h in lst[0]:
        o[h] = {k: (float(np.mean([d[h][k] for d in lst])), float(np.std([d[h][k] for d in lst]))) for k in lst[0][h]}
    return o


for k, v in per_seed.items():
    results["test"][k] = agg(v)
dump(results, "forecast_metrics.json")
dump(curves, "forecast_curves.json")

# ------------------------------------------------------------------ out-of-fold forecasts for policy training days
# For each training day d in 1..4, retrain (seed 0) on the other training days, forecast day d.
tau = cache_full["tau"]
mu_all, var_all = mix(*cache_full["all"])
var_all = var_all * tau
mu_cache = np.full((T, H), np.nan, np.float32); var_cache = np.full((T, H), np.nan, np.float32)
mu_cache[ALL] = mu_all; var_cache[ALL] = var_all
mu_dm = np.full((T, H), np.nan, np.float32); var_dm = mu_dm.copy(); mu_cp = mu_dm.copy(); var_cp = mu_dm.copy()
mu_dm[ALL], var_dm[ALL], mu_cp[ALL], var_cp[ALL] = cache_full["all"]
oof_scores = []
for d in range(1, 5):
    held = np.arange(d * 288, (d + 1) * 288)
    rest = np.setdiff1d(tr, held)
    md, _, _ = train_model(DAPM(5, L, H), Z, rest, va, L, H, seed=0)
    ma, _, _ = train_model(CPUForecaster(5, H), Z, rest, va, L, H, seed=0)
    a = (*predict(md, Z, held, L, H), *predict(ma, Z, held, L, H))
    m, v = mix(*a)
    mu_cache[held] = m; var_cache[held] = v * tau
    mu_dm[held], var_dm[held], mu_cp[held], var_cp[held] = a
    Yh, Mh = targets(held)
    oof_scores.append(metrics(Yh, m, v * tau, Mh)["h1"])
    log(f"OOF fold day {d}: {oof_scores[-1]}")
results["oof_train_days_h1"] = oof_scores
dump(results, "forecast_metrics.json")
np.savez(os.path.join(OUT, "forecast_cache.npz"), mu=mu_cache, var=var_cache,
         mu_dapm=mu_dm, var_dapm=var_dm, mu_cpuf=mu_cp, var_cpuf=var_cp, tau=tau)
log("DONE")
