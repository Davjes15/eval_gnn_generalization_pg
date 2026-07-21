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

out.append(sec("5b. Cross-context g-score -- SMALL-N (no trim, all 3 unseen grids) [recommended here]"))
out.append(pd.read_csv(f"{RES}/gscore_smallN.csv").round(4).to_string(index=False))

out.append(sec("5d. Cross-context g-score -- ENGAGE Table-3 format (AGGREGATED per model)"))
out.append("ENGAGE reports ONE aggregated row per model (its Table 3), pooling ALL")
out.append("train->test pairs into a single g-score, not per training grid. Reproduced")
out.append("here for paper-comparability; the 5b breakdown is kept for the source-grid")
out.append("mechanism. NO percentile trim (bounds=0), consistent with the OOD table (5/6)")
out.append("-- with only 4 grids the ENGAGE 2/98 trim is degenerate. GAT best / GIN worst")
out.append("regardless of trim; the trim only reorders the middle (see Findings 6.1).")
out.append("Laplacian MMD; DC-PF g-score is an artifact (Dmmd=0, Q==0 bookkeeping) -- a")
out.append("reference bar, not a competitor.")
out.append(pd.read_csv(f"{RES}/gscore_cc_aggregate.csv").round(4).to_string(index=False))

out.append(sec("5c. OOD topological distance (held-out grid -> POOLED training grids)"))
out.append("This is the x-axis the OOD g-score uses: for each leave-one-grid-out split")
out.append("the 3 training grids are POOLED into one distribution and a single MMD is")
out.append("computed to the held-out grid -- MMD(held, A u B u C), matching ENGAGE's")
out.append("evaluate_cc_mmd (NOT a mean of pairwise MMDs; see design decision D14).")
out.append(pd.read_csv(f"{RES}/ood_distance.csv").round(4).to_string(index=False))

out.append(sec("6. OOD g-score (per model, over held-out grids; no trim) [better-posed at N=4]"))
out.append("Distance = POOLED Laplacian-MMD from each held-out grid to its training mixture.")
out.append("One point per held-out grid (up to 4); NaN cells (e.g. arma_gnn/UK) dropped.")
out.append("This is the generalization measure most aligned with 'train on several grids,")
out.append("deploy on a new one', and unlike the cross-context g-score it is NOT degenerate.")
out.append(pd.read_csv(f"{RES}/gscore_ood.csv").round(4).to_string(index=False))

report = "\n".join(str(x) for x in out)
with open(f"{RES}/results_report.txt", "w") as f:
    f.write(report + "\n")
print(report)
