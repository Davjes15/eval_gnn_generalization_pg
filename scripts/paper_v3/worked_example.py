"""Worked example: ARMA at the median-mismatch snapshot of one within-grid test set.

Usage: python scripts/paper_v3/worked_example.py IEEE39 1000
"""
import os, sys, json, numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, REPO); os.chdir(REPO)
import torch
from ac_feasibility import build_cases
from physics_metrics import predicted_mask
OUT = os.path.join(REPO, "results_v3", "analysis", "paper_v3")
GRID, s = sys.argv[1], int(sys.argv[2])
from eval_checkpoints import _scaler_for, _test_sets, _predict
from experiments import _build_model, _load_all, load_arch_config
sys.path.insert(0, os.path.join(REPO, "scripts", "paper_v3")); from step3_analyses import per_sample
DA = "data_regime_a_fixed_topology_v3"
data = _load_all(DA, ["IEEE24", "IEEE39", "IEEE118", "UK"])
sc = _scaler_for("within", data, GRID, "pu_zscore")
m = _build_model("arma_gnn", load_arch_config("configs/arch_config.json", ["arma_gnn"])["arma_gnn"], "cpu")
m.load_state_dict(torch.load(os.path.join(REPO, "ckpt_v3", "within_arma_gnn", f"within_arma_gnn_{GRID}_s{s}.pt"), map_location="cpu"))
yt_, yp_, x_ = _predict(m, "cpu", _test_sets("within", data, GRID, sc)[GRID], 32, sc)
cases = build_cases(GRID, f"{DA}/{GRID}/test")
n_bus = yt_.shape[0] // len(cases)
dps, _, _ = per_sample(yt_.numpy(), yp_.numpy(), cases, n_bus)
r = {"yt": yt_.numpy(), "yp": yp_.numpy(), "x": x_.numpy()}
print("run mean", np.nanmean(dps)); med = np.nanmedian(dps); i = int(np.nanargmin(np.abs(dps - med)))
yt = r["yt"].reshape(len(cases), n_bus, 4)[i]; yp = r["yp"].reshape(len(cases), n_bus, 4)[i]
mask = predicted_mask(torch.tensor(r["x"])).numpy().reshape(len(cases), n_bus, 4)[i]
c = cases[i]
names = ["P_MW", "Q_Mvar", "V_pu", "theta_deg"]
err = {}
for j, n in enumerate(names):
    e = np.abs(yp[:, j] - yt[:, j])[mask[:, j]]
    err[n] = {"n_predicted": int(mask[:, j].sum()), "mean_abs": float(e.mean()) if e.size else None,
              "max_abs": float(e.max()) if e.size else None}
rp, rq = c.mismatch(*yp.T); rpt, _ = c.mismatch(*yt.T)
load = float(np.abs(yt[:, 0].clip(min=0)).sum())
lt = c.loading_percent(yt[:, 2], yt[:, 3]); lp = c.loading_percent(yp[:, 2], yp[:, 3])
top = np.argsort(-np.abs(rp))[:3]
res_extra = {"share_dP_at_buses_with_known_P": float(np.abs(rp)[~mask[:, 0]].sum() / np.abs(rp).sum()), "n_bus": int(n_bus)}
res = {"grid": GRID, "seed": s, "sample": i, "run_median_pct": float(med), "sample_pct": float(dps[i]),
       "errors_predicted_entries": err,
       "sum_abs_dP_MW": float(np.abs(rp).sum()), "served_load_MW": load,
       "pct_load": float(np.abs(rp).sum() / load * 100), "true_state_sum_abs_dP_MW": float(np.abs(rpt).sum()),
       "top3_buses_dP_MW": [(int(b), float(rp[b]), names[0] + "_known" if not mask[b, 0] else "P_predicted") for b in top],
       "top3_bus_theta_err_deg": [float(yp[b, 3] - yt[b, 3]) for b in top],
       "top3_bus_V_err_pu": [float(yp[b, 2] - yt[b, 2]) for b in top],
       "branch_loading_abs_err_pct": {"mean": float(np.abs(lp - lt).mean()), "max": float(np.abs(lp - lt).max())},
       "n_branches": int(lt.size), "true_overloaded": int((lt > 100).sum()), "pred_overloaded": int((lp > 100).sum()),
       "missed": int(((lt > 100) & (lp <= 100)).sum()), "false": int(((lt <= 100) & (lp > 100)).sum())}
res.update(res_extra); print(json.dumps(res, indent=1)); json.dump(res, open(f"{OUT}/worked_example_{GRID}_arma_s{s}.json", "w"), indent=1)
