"""Assemble a human-readable results report from the experiment CSVs."""
import pandas as pd

RES = "/home/ubuntu/full_run/results"
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)

def sec(t):
    return f"\n{'='*78}\n{t}\n{'='*78}\n"

out = []
out.append("TRANSMISSION GNN GENERALIZATION - FULL RUN RESULTS REPORT")
out.append("Task: AC power-flow node regression [P, Q, V, theta] (per-unit).")
out.append("Grids: IEEE24, IEEE39, IEEE118, UK.  6 architectures.")
out.append("Datasets: 800 train / 100 val / 100 test per grid (random N-1/N-2 contingencies + demand).")

out.append(sec("1. MMD topological distance (Laplacian-spectrum) between grids"))
out.append(pd.read_csv(f"{RES}/mmd_laplacian.csv", index_col=0).round(4).to_string())
out.append("\n(degree-histogram MMD in mmd_degree.csv)")

out.append(sec("2. DC power-flow baseline (per test grid, per-quantity NRMSE)"))
out.append(pd.read_csv(f"{RES}/dc_baseline.csv").round(4).to_string(index=False))

out.append(sec("3. Cross-context transfer (train on one grid, test on all)"))
cc = pd.read_csv(f"{RES}/cross_context.csv")
out.append(cc.round(4).to_string(index=False))

out.append(sec("3b. Cross-context aggregate-NRMSE transfer matrices (rows=train, cols=test)"))
for m in cc.model.unique():
    piv = cc[cc.model == m].pivot(index="train_grid", columns="test_grid", values="nrmse")
    out.append(f"\n[{m}]")
    out.append(piv.round(4).to_string())

out.append(sec("4. Out-of-distribution (leave-one-grid-out; train on 3, test on held-out)"))
out.append(pd.read_csv(f"{RES}/ood.csv").round(4).to_string(index=False))

out.append(sec("5. g-score -- ORIGINAL (ENGAGE percentile trim, bounds=2)"))
out.append("NOTE: with only 3 unseen grids the 2/98 percentile trim keeps 1 point,")
out.append("so std_nrmse=mmd_range=0. This is the documented small-sample fragility.")
out.append(pd.read_csv(f"{RES}/gscore.csv").round(4).to_string(index=False))

out.append(sec("5b. g-score -- SMALL-N (no trim, all 3 unseen grids) [recommended here]"))
out.append(pd.read_csv(f"{RES}/gscore_smallN.csv").round(4).to_string(index=False))

report = "\n".join(str(x) for x in out)
with open(f"{RES}/results_report.txt", "w") as f:
    f.write(report + "\n")
print(report)
