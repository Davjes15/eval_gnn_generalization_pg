"""Step 3 analyses (predictions only, v3 release checkpoints; nothing trained).

1. Regime B outage overlap: error and AC mismatch by k and seen/new outage set (cc diagonal).
2. Overload severity: missed overloads by true loading band, false alarms by true loading band.

Usage (from the repo root, with the v3.0.0 release checkpoints unpacked in ckpt_v3/):
    python scripts/paper_v3/step3_analyses.py --models gcn arma_gnn gat gin nnconv transformer
    python scripts/paper_v3/step3_analyses.py --dc --models none
    python scripts/paper_v3/aggregate.py
"""
import argparse, ast, json, os, sys
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np, pandas as pd, torch
from eval_checkpoints import find_checkpoints, _scaler_for, _test_sets, _predict
from experiments import _build_model, _load_all, load_arch_config
from ac_feasibility import build_cases, OVERLOAD_PCT
from training_utils import get_device, nrmse_range

GRIDS = ("IEEE24", "IEEE39", "IEEE118", "UK")
DA, DB = "data_regime_a_fixed_topology_v3", "data_regime_b_contingencies_v3"
MISS_BINS = [(100, 110), (110, 150), (150, np.inf)]
FA_BINS = [(90, 100), (-np.inf, 90)]


def outage_key(s):
    return tuple(sorted(map(str, ast.literal_eval(str(s)))))


def per_sample(y_true, y_pred, cases, n_bus):
    yt = y_true.reshape(len(cases), n_bus, 4); yp = y_pred.reshape(len(cases), n_bus, 4)
    dp_share, tl, pl = [], [], []
    for i, c in enumerate(cases):
        p, q, vm, va = yp[i].T
        rp, _ = c.mismatch(p, q, vm, va)
        load = np.abs(yt[i, :, 0].clip(min=0)).sum()
        dp_share.append(np.abs(rp).sum() / load * 100 if load else np.nan)
        tl.append(c.loading_percent(yt[i, :, 2], yt[i, :, 3]))
        pl.append(c.loading_percent(vm, va))
    return np.array(dp_share), tl, pl


def severity_row(tl, pl):
    t = np.concatenate(tl); p = np.concatenate(pl)
    over_t, over_p = t > OVERLOAD_PCT, p > OVERLOAD_PCT
    r = {"n_branch_samples": t.size, "n_true_over": int(over_t.sum()),
         "n_missed": int((over_t & ~over_p).sum()),
         "n_true_secure": int((~over_t).sum()), "n_false": int((~over_t & over_p).sum())}
    for lo, hi in MISS_BINS:
        m = over_t & (t > lo) & (t <= hi)
        r[f"true_over_{lo}_{hi}"] = int(m.sum()); r[f"missed_{lo}_{hi}"] = int((m & ~over_p).sum())
    for lo, hi in FA_BINS:
        m = ~over_t & (t > lo) & (t <= hi)
        r[f"secure_{lo}_{hi}"] = int(m.sum()); r[f"false_{lo}_{hi}"] = int((m & over_p).sum())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", default=os.path.join(REPO, "ckpt_v3"))
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(REPO, "results_v3", "analysis", "paper_v3"))
    ap.add_argument("--dc", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(1)
    data = {DA: _load_all(DA, GRIDS), DB: _load_all(DB, GRIDS)}
    cache = {}
    def cases(d, g):
        if (d, g) not in cache:
            cache[(d, g)] = build_cases(g, os.path.join(d, g, "test"))
        return cache[(d, g)]
    src = {(d, g, s): pd.read_csv(os.path.join(d, g, s, "dataset_src.csv"))
           for d in (DA, DB) for g in GRIDS for s in ("train", "test")}
    tag = "_".join(a.models)
    sev, ovl, samples = [], [], []
    if a.dc:
        from dc_feasibility import _states
        from training_utils import load_grid_dataset
        for d, reg in ((DA, "regime_a"), (DB, "regime_b")):
            for g in GRIDS:
                cs = cases(d, g)
                yt, yd = _states(load_grid_dataset(d, g, "test"))
                n_bus = yt.shape[0] // len(cs)
                for name, st in (("dc_pf", yd), ("floor", yt)):
                    dps, tl, pl = per_sample(yt, st, cs, n_bus)
                    r = {"setting": reg, "model": name, "train_grid": g, "test_grid": g, "seed": -1,
                         "dp_pct_mean": np.nanmean(dps)}
                    r.update(severity_row(tl, pl)); sev.append(r)
        tag = "dc"
    device = get_device()
    ckpts = [c for c in find_checkpoints(a.ckpt_root) if c[2] in a.models] if not a.dc else []
    arch = load_arch_config("configs/arch_config.json", sorted({c[2] for c in ckpts})) if ckpts else {}
    for path, arm, model, grid, seed in ckpts:
        d = DA if arm == "within" else DB
        sc = _scaler_for(arm, data[d], grid, "pu_zscore")
        m = _build_model(model, arch[model], device)
        m.load_state_dict(torch.load(path, map_location=device))
        for tg, ds in _test_sets(arm, data[d], grid, sc).items():
            yt, yp, x = _predict(m, device, ds, 32, sc)
            cs = cases(d, tg); n_bus = yt.shape[0] // len(cs)
            setting = arm if arm != "cross" else ("cc_diagonal" if tg == grid else "cc_unseen")
            setting = {"within": "within"}.get(setting, setting)
            base = {"setting": setting, "arm": arm, "model": model, "train_grid": grid,
                    "test_grid": tg, "seed": seed, "nrmse": nrmse_range(yt, yp)}
            finite = bool(torch.isfinite(yp).all())
            if not finite:
                sev.append({**base, "finite": False}); continue
            dps, tl, pl = per_sample(yt.numpy(), yp.numpy(), cs, n_bus)
            r = {**base, "finite": True, "dp_pct_mean": np.nanmean(dps)}
            r.update(severity_row(tl, pl)); sev.append(r)
            if setting == "cc_diagonal":
                te = src[(d, tg, "test")]; tr = set(src[(d, tg, "train")].out_lines.map(outage_key))
                k = te.k.values; seen = te.out_lines.map(outage_key).isin(tr).values
                ytr = yt.reshape(len(cs), n_bus, 4); ypr = yp.reshape(len(cs), n_bus, 4)
                avg_range = (yt.max(0).values - yt.min(0).values).mean()
                for kk in (0, 1, 2):
                    for sn in ((True,) if kk == 0 else (True, False)):
                        mk = (k == kk) & (seen == sn)
                        if not mk.any(): continue
                        rmse = torch.sqrt(torch.mean((ytr[mk] - ypr[mk]) ** 2))
                        ovl.append({**base, "k": kk, "seen": sn, "n_samples": int(mk.sum()),
                                    "nrmse_fullrange": float(rmse / avg_range),
                                    "dp_pct_mean": float(np.nanmean(dps[mk]))})
            print(arm, model, grid, tg, seed, round(base["nrmse"], 5), round(r["dp_pct_mean"], 2), flush=True)
    pd.DataFrame(sev).to_csv(os.path.join(a.out, f"severity_{tag}.csv"), index=False)
    if ovl: pd.DataFrame(ovl).to_csv(os.path.join(a.out, f"overlap_{tag}.csv"), index=False)
    print("done", tag)


if __name__ == "__main__":
    main()
