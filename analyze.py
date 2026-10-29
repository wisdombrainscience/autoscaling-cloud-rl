"""Stage 3: aggregate results -> LaTeX tables (paper/tables/*.tex), figures (paper/figs/*.pdf), summary JSON.
Every number in the manuscript's results section is produced here."""
import json, glob, time, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
import run_control as rc
from policies import *
from env import test_tasks

P = os.path.join(ROOT, "paper"); os.makedirs(f"{P}/tables", exist_ok=True)
R = os.path.join(OUT, "runs")
C = {"Ours": "#2a78d6", "PredQuantile": "#eb6834", "HPA": "#1baf7a", "Threshold": "#eda100",
     "StaticPeak": "#8a8984", "MLP": "#e87ba4", "NoForecast": "#6250d6", "NoBC": "#008300", "Oracle": "#0b0b0b"}
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.5, "legend.frameon": False,
                     "savefig.bbox": "tight", "savefig.dpi": 300, "axes.edgecolor": "#52514e"})
rng = np.random.default_rng(0)
METRICS = ["ret", "viol", "over", "vms", "util", "actions", "rho_std"]
LABEL = {"Ours": "Ours (GRU-PPO, full)", "MLP": "MLP-PPO (no memory)", "NoForecast": "GRU-PPO, no forecast",
         "PointForecast": "GRU-PPO, point forecast", "NoRiskShaping": "GRU-PPO, no risk shaping",
         "NoBC": "GRU-PPO, no BC warm start", "SingleTask": "GRU-PPO, single task", "Oracle": "GRU-PPO + true $\\theta$",
         "RidgeForecast": "GRU-PPO, ridge forecaster", "Threshold": "Threshold (step scaling)", "HPA": "HPA target tracking",
         "PredQuantile": "Predictive quantile rule", "StaticPeak": "Static peak (hindsight)"}

def boot(x, n=5000):
    x = np.asarray(x, float)
    if len(x) < 2: return (float(x.mean()), float("nan"), float("nan"))
    b = rng.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

# ------------------------------------------------------------------ load
base = json.load(open(f"{R}/baselines.json"))
runs = {}
for f in sorted(glob.glob(f"{R}/*_s*.json")):
    v, s = os.path.basename(f)[:-5].rsplit("_s", 1)
    runs.setdefault(v, {})[int(s)] = json.load(open(f))
print({k: sorted(v) for k, v in runs.items()})

def per_seed(v, split, m):          # list over seeds of task-mean
    return [np.mean(r[split][m]) for r in runs[v].values()]
def per_task(v, split, m):          # task vector averaged over seeds
    return np.mean([r[split][m] for r in runs[v].values()], 0)

summary = {"baselines": {}, "rl": {}, "paired": {}}
for b in ["Threshold", "HPA", "PredQuantile", "StaticPeak"]:
    summary["baselines"][b] = {sp: {m: float(np.mean(base[b][sp][m])) for m in METRICS} for sp in ["id", "ood"]}
    summary["baselines"][b]["selected"] = base[b].get("selected", "")
for v in runs:
    summary["rl"][v] = {sp: {m: boot(per_seed(v, sp, m)) for m in METRICS} for sp in ["id", "ood"]}
    summary["rl"][v]["n_seeds"] = len(runs[v])
    summary["rl"][v]["train_time_s"] = float(np.mean([r["train_time_s"] for r in runs[v].values()]))
    summary["rl"][v]["selected_it"] = [r["curve"][-1].get("selected_it") for r in runs[v].values()]

# paired task-level comparison Ours vs each baseline (difference of task means, bootstrap over tasks)
for b in ["Threshold", "HPA", "PredQuantile"]:
    for sp in ["id", "ood"]:
        for m in ["ret", "viol", "vms", "actions"]:
            d = per_task("Ours", sp, m) - np.array(base[b][sp][m])
            summary["paired"][f"{b}|{sp}|{m}"] = boot(d)

# ------------------------------------------------------------------ main table
def fmt(t, d=2):
    m, lo, hi = t
    return f"{m:.{d}f}" if np.isnan(lo) else f"{m:.{d}f}\\,{{\\scriptsize[{lo:.{d}f},{hi:.{d}f}]}}"
def row(name, src, sp, rl):
    cells = []
    for m, d in [("ret", 1), ("viol", 2), ("over", 2), ("vms", 2), ("actions", 0)]:
        cells.append(fmt(summary["rl"][src][sp][m], d) if rl else f"{summary['baselines'][src][sp][m]:.{d}f}")
    return f"{name} & " + " & ".join(cells) + " \\\\"

for sp, cap in [("id", "in-distribution test tasks"), ("ood", "distribution-shift test tasks")]:
    L = ["\\begin{tabular}{l" + "r" * 5 + "}", "\\toprule",
         "Controller & Return $\\uparrow$ & Viol.\\ (\\%) $\\downarrow$ & Overload (\\%) $\\downarrow$ & VMs $\\downarrow$ & Actions $\\downarrow$ \\\\", "\\midrule"]
    for b in ["StaticPeak", "Threshold", "HPA", "PredQuantile"]:
        L.append(row(LABEL[b], b, sp, False))
    L.append("\\midrule")
    for v in ["MLP", "Ours"]:
        if v in runs: L.append(row(LABEL[v] + f" (n={len(runs[v])})", v, sp, True))
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{P}/tables/main_{sp}.tex", "w").write("\n".join(L))

# ablation table (ID and OOD return/viol/vms/actions)
L = ["\\begin{tabular}{lrrrr}", "\\toprule",
     " & \\multicolumn{2}{c}{In-distribution} & \\multicolumn{2}{c}{Distribution shift} \\\\ \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
     "Variant & Return & Viol.\\,(\\%) & Return & Viol.\\,(\\%) \\\\", "\\midrule"]
for v in ["Ours", "NoForecast", "PointForecast", "NoRiskShaping", "MLP", "NoBC", "SingleTask", "RidgeForecast", "Oracle"]:
    if v not in runs: continue
    c = [fmt(summary["rl"][v][sp][m], d) for sp in ["id", "ood"] for m, d in [("ret", 1), ("viol", 2)]]
    L.append(f"{LABEL[v]} (n={len(runs[v])}) & " + " & ".join(c) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(f"{P}/tables/ablation.tex", "w").write("\n".join(L))

# forecasting table
F = json.load(open(f"{OUT}/forecast_metrics.json"))["test"]
order = [("Persistence", "Persistence"), ("SeasonalNaive", "Seasonal naive (24 h)"), ("RidgeAR", "Ridge AR"),
         ("DAPM-origvar", "DAPM, fixed trend log-var.\\ (original)"), ("DAPM", "DAPM"), ("CPUForecaster", "CPUForecaster"),
         ("Ensemble-naive", "Ensemble, variance average (original)"), ("Ensemble", "Ensemble, moment-matched"),
         ("Ensemble-cal", "Ensemble, moment-matched + calibrated (used)")]
L = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
     " & \\multicolumn{5}{c}{$h=1$ (5 min)} & $h=3$ (15 min) \\\\ \\cmidrule(lr){2-6}\\cmidrule(lr){7-7}",
     "Model & MAE & RMSE & CRPS & PICP$_{90}$ & PIW$_{90}$ & CRPS \\\\", "\\midrule"]
for k, nm in order:
    g = lambda h, m: F[k][h][m][0] if isinstance(F[k][h][m], list) else F[k][h][m]
    sdv = lambda h, m: F[k][h][m][1] if isinstance(F[k][h][m], list) else None
    cells = []
    for h, m in [("h1", "MAE"), ("h1", "RMSE"), ("h1", "CRPS"), ("h1", "PICP90"), ("h1", "PIW90"), ("h3", "CRPS")]:
        s_ = sdv(h, m); cells.append(f"{g(h, m):.2f}" + (f"\\,{{\\scriptsize$\\pm${s_:.2f}}}" if s_ is not None else ""))
    L.append(f"{nm} & " + " & ".join(cells) + " \\\\")
    if k in ("RidgeAR",): L.append("\\midrule")
L += ["\\bottomrule", "\\end{tabular}"]
open(f"{P}/tables/forecast.tex", "w").write("\n".join(L))

# ------------------------------------------------------------------ Pareto figure
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
for ax, sp, tt in [(axs[0], "id", "(a) In-distribution"), (axs[1], "ood", "(b) Distribution shift")]:
    for b, mk in [("Threshold", "v"), ("HPA", "^"), ("PredQuantile", "D")]:
        x, y = summary["baselines"][b][sp]["vms"], summary["baselines"][b][sp]["viol"]
        ax.scatter(x, y, color=C[b], marker=mk, s=28, zorder=3, label=LABEL[b], edgecolor="white", linewidth=0.6)
    for v, mk in [("MLP", "o"), ("NoForecast", "P"), ("NoBC", "X"), ("Ours", "*")]:
        if v not in runs: continue
        xs, ys = per_seed(v, sp, "vms"), per_seed(v, sp, "viol")
        ax.scatter(xs, ys, color=C[v], marker=mk, s=10, alpha=0.45, zorder=2, linewidth=0)
        bx, by = boot(xs), boot(ys)
        ax.errorbar(bx[0], by[0], xerr=[[bx[0] - bx[1]], [bx[2] - bx[0]]], yerr=[[by[0] - by[1]], [by[2] - by[0]]],
                    color=C[v], lw=0.8, capsize=1.5, zorder=3)
        ax.scatter(np.mean(xs), np.mean(ys), color=C[v], marker=mk, s=60 if v == "Ours" else 30, zorder=4,
                   label=LABEL[v], edgecolor="white", linewidth=0.6)
    ax.set_xlabel("Mean provisioned VMs (cost)"); ax.set_ylabel("SLA violations (% of steps)"); ax.set_title(tt, loc="left", fontsize=8)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.2), fontsize=6.5)
fig.savefig(f"{P}/figs/fig_pareto.pdf"); plt.close(fig)

# ------------------------------------------------------------------ learning curves
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.2))
for v in ["Ours", "NoBC", "MLP", "NoForecast"]:
    if v not in runs: continue
    cs = [[c["ret"] for c in r["curve"] if "ret" in c] for r in runs[v].values()]
    n = min(map(len, cs)); a = np.array([c[:n] for c in cs])
    k = 10; sm = np.array([np.convolve(x, np.ones(k) / k, "valid") for x in a])
    it = np.arange(sm.shape[1]) + k - 1
    ax[0].plot(it, sm.mean(0), color=C[v], lw=1.1, label=LABEL[v]); ax[0].fill_between(it, sm.min(0), sm.max(0), color=C[v], alpha=0.15, lw=0)
    vs = [[(c["it"], c["val"]) for c in r["curve"] if "val" in c and "it" in c] for r in runs[v].values()]
    n = min(map(len, vs)); va = np.array([[x[1] for x in s_[:n]] for s_ in vs]); vi = [x[0] for x in vs[0][:n]]
    ax[1].plot(vi, va.mean(0), color=C[v], lw=1.1, marker="o", ms=2.5)
ax[1].axhline(np.mean(json.load(open(f"{R}/baselines.json"))["_grid"]["PredQuantile"][-3][1]) if False else max(s for _, s in base["_grid"]["PredQuantile"]),
              color=C["PredQuantile"], ls="--", lw=0.8, label="Pred. quantile rule (tuned)")
ax[0].set_ylim(-170, -85); ax[0].set_xlabel("PPO iteration"); ax[0].set_ylabel("Training return (288-step episode)"); ax[0].set_title("(a) Training episodes", loc="left", fontsize=8)
ax[1].set_xlabel("PPO iteration"); ax[1].set_ylabel("Validation return (greedy)"); ax[1].set_title("(b) Validation day", loc="left", fontsize=8)
h, l = ax[0].get_legend_handles_labels(); h2, l2 = ax[1].get_legend_handles_labels()
fig.legend(h + h2, l + l2, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.17), fontsize=6.5)
fig.savefig(f"{P}/figs/fig_learning.pdf"); plt.close(fig)

# ------------------------------------------------------------------ recorded rollouts: traces, adaptation, probe, latency
def load_net(v, s):
    kw, rec, _ = rc.VARIANTS[v]; kw = dict(kw)
    env = rc.make_env(**kw)()
    net = ActorCritic(env.obs_dim, 3, 64, rec); net.load_state_dict(torch.load(f"{R}/{v}_s{s}.pt")); net.eval()
    return net, rc.make_env(**kw)

tasks_ood = test_tasks("ood"); tasks_id = test_tasks("id")
rec = {}
for nm in ["Ours", "MLP"]:
    if nm not in runs: continue
    rec[nm] = []
    for s in sorted(runs[nm]):
        net, ef = load_net(nm, s)
        rec[nm].append({sp: rc.evaluate(RLPolicy(net), ef, tk, rc.TEST, record=True) for sp, tk in [("id", tasks_id), ("ood", tasks_ood)]})
for b, pol in [("PredQuantile", PredictiveQuantile(0.95, 0.999)), ("HPA", HPA(target=0.8, tol=0.05))]:
    rec[b] = [{sp: rc.evaluate(pol, rc.make_env(), tk, rc.TEST, record=True) for sp, tk in [("id", tasks_id), ("ood", tasks_ood)]}]

# per-delay breakdown (ID tasks d=0,1,2; OOD tasks d=3)
dl = np.array([t.delay for t in tasks_id])
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
groups = [("HPA", None), ("PredQuantile", None), ("MLP", "rl"), ("Ours", "rl"), ("Oracle", "rl")]
groups = [g_ for g_ in groups if g_[1] is None or g_[0] in runs]
wid = 0.8 / len(groups); per_delay = {}
for gi, (nm, kind) in enumerate(groups):
    vio, vms = [], []
    for d_ in [0, 1, 2, 3]:
        sp = "ood" if d_ == 3 else "id"; mask = np.ones(24, bool) if d_ == 3 else (dl == d_)
        if kind is None:
            vio.append(np.mean(np.array(base[nm][sp]["viol"])[mask])); vms.append(np.mean(np.array(base[nm][sp]["vms"])[mask]))
        else:
            vio.append(np.mean(per_task(nm, sp, "viol")[mask])); vms.append(np.mean(per_task(nm, sp, "vms")[mask]))
    per_delay[nm] = {"viol": vio, "vms": vms}
    xs = np.arange(4) + (gi - (len(groups) - 1) / 2) * wid
    ax[0].bar(xs, vio, wid * 0.9, color=C.get(nm, "#888"), label=LABEL[nm])
    ax[1].bar(xs, vms, wid * 0.9, color=C.get(nm, "#888"))
for a_, yl in zip(ax, ["SLA violations (%)", "Mean VMs"]):
    a_.set_xticks(range(4)); a_.set_xticklabels(["$d$=0", "$d$=1", "$d$=2", "$d$=3\n(shift)"]); a_.set_ylabel(yl)
ax[1].set_ylim(bottom=8)
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.13), fontsize=6.5)
fig.savefig(f"{P}/figs/fig_delay.pdf"); plt.close(fig)
summary["per_delay"] = per_delay

# hidden-state probe on ID tasks: leave-one-task-out ridge read-out of the hidden task parameters
probe = {}; steps = [1, 6, 12, 24, 48, 96, 192, 384]
for nm in ["Ours", "MLP"]:
    if nm not in rec: continue
    out = {"kappa": [], "delay": []}
    for r in rec[nm]:
        Hs = r["id"]["hidden"]
        for tgt in ["kappa", "delay"]:
            yv = np.array([getattr(t, tgt) for t in tasks_id], float); r2 = []
            for k in steps:
                Xk = Hs[:, k]; Xk = (Xk - Xk.mean(0)) / (Xk.std(0) + 1e-6); pred = np.zeros(len(yv))
                for i in range(len(yv)):
                    tr_ = np.arange(len(yv)) != i; Xa = np.c_[Xk, np.ones(len(Xk))]
                    W = np.linalg.solve(Xa[tr_].T @ Xa[tr_] + 1.0 * np.eye(Xa.shape[1]), Xa[tr_].T @ yv[tr_])
                    pred[i] = Xa[i] @ W
                r2.append(1 - np.mean((yv - pred) ** 2) / np.var(yv))
            out[tgt].append(r2)
    probe[nm] = {k: np.mean(v, 0).tolist() for k, v in out.items()}
summary["probe_steps"] = steps; summary["probe"] = probe

# trace figure: one OOD task, Ours vs PredQuantile vs HPA
ti = 3
fig, axs = plt.subplots(2, 1, figsize=(7.0, 3.2), sharex=True)
tr = rec["PredQuantile"][0]["ood"]["trace"]; Dd = np.array(tr["D"][ti]); tau = tasks_ood[ti].tau
W0, W1 = 0, 16 * 12
Dd = Dd[W0:W1]; tt = np.arange(len(Dd)) / 12
axs[0].fill_between(tt, 0, Dd / tau, color="#d9d8d3", lw=0, step="post", label="Required capacity $D_t/\\tau$")
for nm in ["HPA", "PredQuantile", "Ours"]:
    axs[0].step(tt, np.array(rec[nm][0]["ood"]["trace"]["n"][ti])[W0:W1], where="post", color=C[nm], lw=1.0, label=LABEL[nm])
    axs[1].plot(tt, np.array(rec[nm][0]["ood"]["trace"]["rho"][ti])[W0:W1] * 100, color=C[nm], lw=0.8)
axs[1].axhline(tau * 100, color="#e34948", ls="--", lw=0.8, label="SLA target $\\tau$")
axs[0].set_ylabel("VMs"); axs[1].set_ylabel("Load ratio $\\rho_t$ (%)"); axs[1].set_xlabel("Hours (test span)")
axs[0].legend(ncol=4, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0)); axs[1].legend(fontsize=6.5, loc="upper right")
fig.savefig(f"{P}/figs/fig_trace.pdf"); plt.close(fig)
summary["trace_task"] = vars(tasks_ood[ti])

# inference latency (CPU, single env, one decision incl. feature construction)
net, ef = load_net("Ours", sorted(runs["Ours"])[0]); env = ef(); env.reset(tasks_id[:1], [rc.TEST[0]], 200)
pol = RLPolicy(net); pol.reset(env); ts = []
for k in range(200):
    t0 = time.perf_counter(); a = pol.act(env); ts.append(time.perf_counter() - t0); env.step(a)
summary["latency_ms"] = {"median": float(np.median(ts) * 1e3), "p99": float(np.percentile(ts, 99) * 1e3)}
summary["params"] = int(sum(p.numel() for p in net.parameters()))
fc = torch.load(f"{OUT}/forecasters_seed0.pt")
summary["forecaster_params"] = {k: int(sum(v.numel() for v in sd.values())) for k, sd in fc.items()}
dump(summary, "summary.json")
print(json.dumps({k: summary[k] for k in ["latency_ms", "params", "forecaster_params", "probe"]}, indent=1))

# ------------------------------------------------------------------ LaTeX macros for numbers quoted in the text
def mac(name, val): return f"\\newcommand{{\\{name}}}{{{val}}}"
M = []
S = summary
M.append(mac("PolicyParams", f"{S['params']:,}".replace(",", "\\,")))
M.append(mac("PPOMinutes", f"{S['rl']['Ours']['train_time_s']/60:.0f}"))
M.append(mac("SelThreshold", S["baselines"]["Threshold"]["selected"].replace("up=", "$u{=}").replace(",low=", "$, $l{=}") + "$"))
M.append(mac("SelHPA", S["baselines"]["HPA"]["selected"].replace("target=", "$g{=}").replace(",tol=", "$, tol.\\ ") ))
M.append(mac("SelPred", S["baselines"]["PredQuantile"]["selected"].replace("q=", "$q{=}").replace(",q_in=", "$, $q_{\\mathrm{in}}{=}") + "$"))
M.append(mac("LatMed", f"{S['latency_ms']['median']:.2f}")); M.append(mac("LatPnn", f"{S['latency_ms']['p99']:.2f}"))
for nm, key in [("Ours", "Ours"), ("Mlp", "MLP"), ("NoFc", "NoForecast"), ("NoBc", "NoBC"), ("Point", "PointForecast"),
                ("NoRisk", "NoRiskShaping"), ("Single", "SingleTask"), ("Oracle", "Oracle"), ("Ridge", "RidgeForecast")]:
    if key not in S["rl"]: continue
    for sp in ["id", "ood"]:
        for m, d in [("ret", 1), ("viol", 2), ("vms", 2), ("actions", 0), ("util", 1), ("over", 2), ("rho_std", 1)]:
            v = S["rl"][key][sp][m]
            M.append(mac(f"{nm}{sp.upper()}{m.replace('_','')}", f"{v[0]:.{d}f}"))
            if not np.isnan(v[1]): M.append(mac(f"{nm}{sp.upper()}{m.replace('_','')}CI", f"[{v[1]:.{d}f}, {v[2]:.{d}f}]"))
for nm, key in [("Thr", "Threshold"), ("Hpa", "HPA"), ("Pred", "PredQuantile"), ("Static", "StaticPeak")]:
    for sp in ["id", "ood"]:
        for m, d in [("ret", 1), ("viol", 2), ("vms", 2), ("actions", 0), ("util", 1), ("over", 2), ("rho_std", 1)]:
            M.append(mac(f"{nm}{sp.upper()}{m.replace('_','')}", f"{S['baselines'][key][sp][m]:.{d}f}"))
for k, v in S["paired"].items():
    b, sp, m = k.split("|"); nm = {"Threshold": "Thr", "HPA": "Hpa", "PredQuantile": "Pred"}[b]
    d = 0 if m == "actions" else 2
    M.append(mac(f"D{nm}{sp.upper()}{m}", f"{v[0]:+.{d}f}"))
    M.append(mac(f"D{nm}{sp.upper()}{m}CI", f"[{v[1]:+.{d}f}, {v[2]:+.{d}f}]"))
# relative action reduction vs predictive rule
for sp in ["id", "ood"]:
    M.append(mac(f"ActRed{sp.upper()}", f"{100*(1-S['rl']['Ours'][sp]['actions'][0]/S['baselines']['PredQuantile'][sp]['actions']):.0f}"))
# forecasting numbers
Fm = json.load(open(f"{OUT}/forecast_metrics.json"))
g = lambda k, h, m: (Fm["test"][k][h][m][0] if isinstance(Fm["test"][k][h][m], list) else Fm["test"][k][h][m])
for k, nm in [("Persistence", "Per"), ("RidgeAR", "Rid"), ("DAPM", "Dapm"), ("DAPM-origvar", "DapmO"), ("CPUForecaster", "Cpuf"),
              ("Ensemble-naive", "EnsN"), ("Ensemble-cal", "Ens"), ("SeasonalNaive", "Sea")]:
    for h in ["h1", "h3"]:
        for m in ["MAE", "RMSE", "CRPS", "PICP90", "PIW90"]:
            M.append(mac(f"F{nm}{ {'h1':'One','h3':'Three'}[h] }{ {'PICP90':'Cov','PIW90':'Wid'}.get(m, m.capitalize()) }", f"{g(k, h, m):.{1 if m.startswith('PI') else 2}f}"))
M.append(mac("FTemp", f"{np.mean(list(Fm['val_temperature'].values())):.2f}"))
M.append(mac("FMaeGainPer", f"{100*(1-g('Ensemble-cal','h1','MAE')/g('Persistence','h1','MAE')):.0f}"))
M.append(mac("FCrpsGainPer", f"{100*(1-g('Ensemble-cal','h1','CRPS')/g('Persistence','h1','CRPS')):.0f}"))
M.append(mac("ForecastMinutes", os.environ.get("FC_MIN", "5")))
open(f"{P}/tables/macros.tex", "w").write("\n".join(M) + "\n")
print("macros:", len(M))
