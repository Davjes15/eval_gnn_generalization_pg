"""Summaries for the outage-overlap and overload-severity analyses (void rule applied)."""
import glob, os, numpy as np, pandas as pd
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.join(REPO, "results_v3", "analysis", "paper_v3"))
pd.set_option("display.width", 250)
sev = pd.concat([pd.read_csv(f) for f in glob.glob("severity_*.csv") if "dc" not in f and "summary" not in f])
dc = pd.read_csv("severity_dc.csv")
# void rule: whole (arm, model, train_grid, seed) group if any non-finite
bad = sev[(sev.finite == False) | ~np.isfinite(sev.nrmse)][["arm", "model", "train_grid", "seed"]].drop_duplicates()
print("voided groups:\n", bad)
key = ["arm", "model", "train_grid", "seed"]
sev = sev.merge(bad.assign(void=True), on=key, how="left"); sev["void"] = sev["void"].fillna(False).astype(bool)
v = sev[~sev.void].copy()
# validation vs committed
c = pd.read_csv(os.path.join(REPO, "results_v3", "physics", "physics_metrics.csv"))
m = v.merge(c, on=["arm", "model", "train_grid", "test_grid", "seed"], suffixes=("", "_c"))
print("validated rows", len(m), "of", len(v), "max |dnrmse|", (m.nrmse - m.nrmse_c).abs().max(),
      "max |ddp|", (m.dp_pct_mean - m.ac_dp_pct_load).abs().max(),
      "max |dmiss|", (m.n_missed / m.n_true_over - m.missed_overload_rate).abs().max(),
      "max |dfa|", (m.n_false / m.n_true_secure - m.false_overload_rate).abs().max())
def add_rates(d):
    d = d.copy()
    for b in ["100_110", "110_150", "150_inf"]:
        d[f"share_of_misses_{b}"] = d[f"missed_{b}"] / d.n_missed
        d[f"miss_rate_{b}"] = d[f"missed_{b}"] / d[f"true_over_{b}"]
        d[f"true_over_share_{b}"] = d[f"true_over_{b}"] / d.n_true_over
    for b in ["90_100", "-inf_90"]:
        d[f"share_of_false_{b}"] = d[f"false_{b}"] / d.n_false
        d[f"fa_rate_{b}"] = d[f"false_{b}"] / d[f"secure_{b}"]
    d["miss_rate"] = d.n_missed / d.n_true_over; d["fa_rate"] = d.n_false / d.n_true_secure
    return d
v = add_rates(v); dc = add_rates(dc)
cols = ["miss_rate", "share_of_misses_100_110", "share_of_misses_110_150", "share_of_misses_150_inf",
        "miss_rate_100_110", "miss_rate_110_150", "miss_rate_150_inf",
        "fa_rate", "share_of_false_90_100", "share_of_false_-inf_90", "fa_rate_90_100", "fa_rate_-inf_90",
        "true_over_share_100_110", "true_over_share_110_150", "true_over_share_150_inf"]
v = v[v.setting.isin(["within", "cc_unseen", "ood", "cc_diagonal"])]
agg = v.groupby(["setting", "model"])[cols].mean(); agg["n_runs"] = v.groupby(["setting", "model"]).size()
agg = pd.concat([agg, dc.groupby(["setting", "model"])[cols].mean().assign(n_runs=4)])
agg.round(4).to_csv("severity_summary.csv")
print(agg[["n_runs", "miss_rate", "share_of_misses_100_110", "share_of_misses_110_150", "share_of_misses_150_inf",
           "miss_rate_100_110", "miss_rate_150_inf", "fa_rate", "share_of_false_90_100", "fa_rate_90_100", "fa_rate_-inf_90"]].round(3).to_string())
print(agg[["true_over_share_100_110", "true_over_share_110_150", "true_over_share_150_inf"]].round(3).drop_duplicates().to_string())
# overlap
o = pd.concat([pd.read_csv(f) for f in glob.glob("overlap_*.csv") if "summary" not in f])
o = o.merge(bad.assign(void=True), on=key, how="left"); o = o[o["void"].isna()]
o["group"] = o.apply(lambda r: "k0" if r.k == 0 else f"k{r.k}_{'seen' if r.seen else 'new'}", axis=1)
print("samples per group per grid:\n", o.groupby(["test_grid", "group"]).n_samples.first().unstack())
oa = o.groupby(["model", "group"])[["nrmse_fullrange", "dp_pct_mean"]].mean().unstack()
oa.round(5).to_csv("overlap_summary.csv"); print(oa.round(4).to_string())
# within-run ratio new/seen, then mean over runs
p = o.pivot_table(index=key + ["test_grid"], columns="group", values=["nrmse_fullrange", "dp_pct_mean"])
for met in ["nrmse_fullrange", "dp_pct_mean"]:
    for kk in ("k1", "k2"):
        r = p[(met, f"{kk}_new")] / p[(met, f"{kk}_seen")]
        print(met, kk, "new/seen ratio: median", round(r.median(), 3), "mean", round(r.mean(), 3),
              "share of runs new>seen", round((r > 1).mean(), 3), "n", r.notna().sum())
    r = p[(met, "k1_seen")] / p[(met, "k0")]; print(met, "k1_seen/k0 median", round(r.median(), 3))
    r = p[(met, "k2_seen")] / p[(met, "k0")]; print(met, "k2_seen/k0 median", round(r.median(), 3))
print("per grid median new/seen nrmse:")
for kk in ("k1", "k2"):
    r = (p[("nrmse_fullrange", f"{kk}_new")] / p[("nrmse_fullrange", f"{kk}_seen")]).groupby(level="test_grid").median()
    print(kk, r.round(3).to_dict())
