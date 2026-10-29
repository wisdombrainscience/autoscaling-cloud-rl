# Risk-aware recurrent RL autoscaler — reproducible code

Code for *Risk-Aware Recurrent Reinforcement Learning with Calibrated Probabilistic Forecasts
for Predictive Cloud Autoscaling* (submitted to Future Generation Computer Systems).
Everything runs on a CPU; no GPU is needed.

## Layout
| File | Role |
|---|---|
| `common.py` | data loading, chronological split (train days 1–4, val day 5, test days 6–7), train-only z-scoring, constants (`L_HIST=12`, `H_FC=3`) |
| `forecasters.py` | DAPM, CPUForecaster, Gaussian NLL training with early stopping, moment-matched mixture |
| `run_forecast.py` | Stage 1: trains forecasters (3 seeds), naive/ridge baselines, validation temperature, out-of-fold forecasts for the training days, writes `results/forecast_cache.npz` and `forecast_metrics.json` |
| `ridge_cache.py` | ridge-AR forecasts for the forecaster-swap ablation |
| `env.py` | vectorised trace-driven environment: hidden task parameters (κ, delay, noise, bursts), capacity-aware risk features, reward |
| `policies.py` | Threshold, HPA, predictive-quantile and static baselines; GRU actor–critic |
| `ppo.py` | PPO with full-episode BPTT, return scaling, BC warm start, validation checkpointing |
| `run_control.py` | Stage 2: `python run_control.py baselines` (tunes baselines on the validation day) and `python run_control.py <Variant> <seed>` |
| `jobs.txt` | the 35 RL runs in the paper (variant, seed) |
| `analyze.py` | Stage 3: all tables (`paper/tables/*.tex`), figures (`paper/figs/*.pdf`), and the LaTeX macros for every number quoted in the text |
| `figs_forecast.py` | data and forecasting figures |

## Reproduce
```bash
pip install torch numpy pandas scipy matplotlib
# data/mu.csv = machine_usage_days_1_to_8_grouped_300_seconds.csv (Alibaba cluster-trace-v2018, 300-s means)
python run_forecast.py                     # ~5 min on 2 CPU cores
python ridge_cache.py
python run_control.py baselines
cat jobs.txt | xargs -P 2 -L 1 sh -c 'python run_control.py $0 $1'   # ~7 min per run
python analyze.py && python figs_forecast.py
cd ../paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
Seeds are fixed. Exogenous noise and bursts use common random numbers, so every controller sees the same test conditions.

## Differences from the earlier notebook (`dapm-anp-rl.ipynb`)
* No train/test leakage: forecasters are evaluated on unseen days, and the scaler is fit on training data only.
* DAPM trend log-variance is no longer fixed at 0, which had added a unit variance to every prediction.
* The ensemble variance includes the disagreement term, and the ensemble is temperature-calibrated on validation.
* The history length is selected on validation (1 h instead of 24 h).
* The tail-risk term depends on committed capacity, so the action affects it.
* Scale-out has a provisioning delay.
* `cost_per_vm` is used.
* The rule baseline is fixed: it had compared a z-score to a percentage. All baselines are tuned on validation.
* PPO uses return scaling and an unclipped value loss (value clipping on unnormalised returns had stalled the critic).
* Checkpoints are selected on the validation day.
* Evaluation covers 24 in-distribution and 24 shifted tasks, 5 seeds and bootstrap CIs.
