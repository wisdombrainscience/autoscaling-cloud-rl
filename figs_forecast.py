import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import norm
from common import *
P = os.path.join(ROOT, "paper", "figs")
C1, C2, C3, C4, GR = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.5, "axes.edgecolor": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e", "legend.frameon": False,
                     "savefig.bbox": "tight", "savefig.dpi": 300, "lines.linewidth": 1.0})
D = load(); X = D["X"]; T = D["T"]; s = D["split"]; mu0, sd0 = D["mu"][0], D["sd"][0]
days = np.arange(T) / 288

# ---- Fig: dataset overview with split shading
names = ["CPU (%)", "Memory (%)", "Net in", "Net out", "Disk I/O (%)"]
fig, axs = plt.subplots(5, 1, figsize=(7.0, 4.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1, 1, 1, 1]})
for i, ax in enumerate(axs):
    ax.plot(days, X[:, i], color=C1 if i == 0 else "#52514e", lw=0.7 if i == 0 else 0.5)
    ax.set_ylabel(names[i], fontsize=7)
    for (a, b), c, lab in [((0, 288), "#f2f1ec", "history"), (s["train"], "#e7f0fb", "train"), (s["val"], "#fdf0e6", "validation"), (s["test"], "#e6f6ef", "test")]:
        ax.axvspan(a / 288, b / 288, color=c, zorder=-1, lw=0)
        if i == 0: ax.text((a + b) / 2 / 288, 82, lab, ha="center", fontsize=7, color="#52514e")
axs[0].set_ylim(0, 90); axs[-1].set_xlabel("Time (days)")
fig.align_ylabels(axs); fig.savefig(f"{P}/fig_data.pdf"); plt.close(fig)

# ---- Fig: forecast fan chart on test + reliability
Cc = np.load(os.path.join(OUT, "forecast_cache.npz"))
te = np.arange(*s["test"]); y = X[te, 0]
m = Cc["mu"][te, 0] * sd0 + mu0; sdv = np.sqrt(Cc["var"][te, 0]) * sd0
fig = plt.figure(figsize=(7.0, 2.7)); gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1], wspace=0.28)
ax = fig.add_subplot(gs[0]); tt = te / 288
for z, al in [(norm.ppf(0.995), 0.18), (norm.ppf(0.95), 0.3)]:
    ax.fill_between(tt, m - z * sdv, m + z * sdv, color=C1, alpha=al, lw=0)
ax.plot(tt, y, color="#0b0b0b", lw=0.6, label="Observed")
ax.plot(tt, m, color=C1, lw=0.9, label="Ensemble mean")
ax.fill_between([], [], [], color=C1, alpha=0.3, label="90% interval"); ax.fill_between([], [], [], color=C1, alpha=0.18, label="99% interval")
ax.set_xlabel("Time (days, test span)"); ax.set_ylabel("CPU utilisation (%)"); ax.legend(ncol=4, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2))
ax.set_title("(a) One-step-ahead probabilistic forecast", fontsize=8, loc="left")
ax2 = fig.add_subplot(gs[1]); lv = np.linspace(0.05, 0.99, 25)
def cov(mm, ss): return [np.mean(np.abs(y - mm) <= norm.ppf(0.5 + l / 2) * ss) for l in lv]
md = Cc["mu_dapm"][te, 0] * sd0 + mu0; sdd = np.sqrt(Cc["var_dapm"][te, 0]) * sd0
mc = Cc["mu_cpuf"][te, 0] * sd0 + mu0; sdc = np.sqrt(Cc["var_cpuf"][te, 0]) * sd0
sdo = np.sqrt(Cc["var_dapm"][te, 0] + 1) * sd0
ax2.plot([0, 1], [0, 1], color=GR, lw=0.6, ls="--"); ax2.set_xlim(0, 1.02); ax2.set_ylim(0, 1.04)
for mm, ss, c, lab, mk in [(m, sdv, C1, "Ensemble (cal.)", "o"), (md, sdd, C2, "DAPM", "s"), (mc, sdc, C3, "CPUForecaster", "^"), (md, sdo, C4, "DAPM, fixed trend var.", "D")]:
    ax2.plot(lv, cov(mm, ss), color=c, lw=1.0, marker=mk, ms=2.5, label=lab)
ax2.set_xlabel("Nominal central coverage"); ax2.set_ylabel("Empirical coverage"); ax2.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
ax2.set_title("(b) Reliability (test)", fontsize=8, loc="left")
fig.savefig(f"{P}/fig_forecast.pdf"); plt.close(fig)
print("ok")
