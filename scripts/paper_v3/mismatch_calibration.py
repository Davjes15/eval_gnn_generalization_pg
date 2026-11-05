"""Mismatch calibration: label floor, angle-noise sensitivity and V/theta decomposition.

Usage: python scripts/paper_v3/mismatch_calibration.py gcn IEEE39 0
Appends rows to results_v3/analysis/paper_v3/mismatch_calibration.csv.
"""
import os, sys, csv, numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, REPO); os.chdir(REPO)
import torch
from ac_feasibility import build_cases
from physics_metrics import predicted_mask
from eval_checkpoints import _scaler_for, _test_sets, _predict
from experiments import _build_model, _load_all, load_arch_config
sys.path.insert(0, os.path.join(REPO, "scripts", "paper_v3")); from step3_analyses import per_sample
MODEL, GRID, S = sys.argv[1], sys.argv[2], int(sys.argv[3])
DA = "data_regime_a_fixed_topology_v3"
data = _load_all(DA, ["IEEE24", "IEEE39", "IEEE118", "UK"])
sc = _scaler_for("within", data, GRID, "pu_zscore")
m = _build_model(MODEL, load_arch_config("configs/arch_config.json", [MODEL])[MODEL], "cpu")
m.load_state_dict(torch.load(os.path.join(REPO, "ckpt_v3", f"within_{MODEL}", f"within_{MODEL}_{GRID}_s{S}.pt"), map_location="cpu"))
yt, yp, x = (a.numpy() for a in _predict(m, "cpu", _test_sets("within", data, GRID, sc)[GRID], 32, sc))
cases = build_cases(GRID, f"{DA}/{GRID}/test"); n = yt.shape[0] // len(cases)
mask = predicted_mask(torch.tensor(x)).numpy()
def mm(y): return np.nanmean(per_sample(yt, y, cases, n)[0])
th_err = np.abs(yp[:, 3] - yt[:, 3])[mask[:, 3]]
rows = [("model_prediction", mm(yp)), ("label_floor", mm(yt))]
y = yt.copy(); y[:, 2] = yp[:, 2]; rows.append(("true_state_model_V_only", mm(y)))
y = yt.copy(); y[:, 3] = yp[:, 3]; rows.append(("true_state_model_theta_only", mm(y)))
rng = np.random.default_rng(0)
for sig in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]:
    y = yt.copy(); y[:, 3] = yt[:, 3] + np.where(mask[:, 3], rng.normal(0, sig, len(yt)), 0)
    rows.append((f"true_state_theta_noise_sd_{sig}_deg", mm(y)))
rows.append(("model_theta_mae_deg", float(th_err.mean())))
out = os.path.join(REPO, "results_v3", "analysis", "paper_v3", "mismatch_calibration.csv")
new = not os.path.exists(out)
with open(out, "a", newline="") as f:
    w = csv.writer(f)
    if new: w.writerow(["model", "grid", "seed", "state", "ac_dp_pct_load"])
    for name, val in rows:
        w.writerow([MODEL, GRID, S, name, f"{val:.6g}"]); print(MODEL, GRID, S, name, round(val, 4))
