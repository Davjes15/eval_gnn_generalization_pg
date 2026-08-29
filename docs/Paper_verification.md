# Verification of this study's claims against the two published papers

Until now every claim in this repository about "what ENGAGE does" or "what
PowerGraph does" came from the **released code** (`repos/engage/graph_gen.py`,
`training_utils.py`, `environment.yaml`; the PowerGraph-Node repository), because
this session's network allowlist blocks the publication sites. The two papers
were supplied directly on 2026-07-18 and every such claim is now checked against
them:

- **ENGAGE** — Okoyomon & Goebel, *A Framework for Assessing the Generalizability
  of GNN-Based AC Power Flow Models*, E-Energy '25, Rotterdam.
- **PowerGraph** — *PowerGraph: A Power Grid Benchmark Dataset for Graph Neural
  Networks*, NeurIPS 2024 Datasets & Benchmarks Track.

Page numbers are PDF pages of the supplied files.

## 1. The DC baseline — confirmed, and the paper adds numbers we can compare to

| Claim made in this repo | Paper evidence | Verdict |
|---|---|---|
| ENGAGE reports a DC power-flow baseline | `dc_pf` is a model in every results figure and in Table 3 (pp. 6–9) | **confirmed** |
| DC is scored on the same four quantities as the GNNs, i.e. it is charged for Q | Table 3 lists DC PF's `μNRMSE` in the same column as the GCN variants, and §3.3.1 defines one NRMSE over the `D` output dimensions (p. 5) | **confirmed** |
| DC is an analytical model whose error is topology-agnostic | *"DC PF is an analytical model, not a learned model, so its test performance for a particular grid will be consistent regardless of training data"* (p. 6) | **confirmed** |
| DC's g-score is a reference bar, not a competitor, because its MMD range is 0 | Table 3: DC PF has `ΔMMD = 0.0000` and `𝔤score = μNRMSE` in both experiments (p. 9) | **confirmed — this is ENGAGE's own table, not our interpretation** |
| Our motivating premise (GNNs lose to DC on unseen grids) is ENGAGE's finding | *"It is clear from both the mean and the distribution that the DC PF model is far superior in this context"* (cross-context, p. 7); the OOD case is *"precisely the case in which all models from [11] failed to generalize as well as DC PF"* (p. 6) | **confirmed** |
| PowerGraph publishes no DC baseline | PowerGraph's only non-GNN baseline is **Gradient Boosted Trees**, and only on the project website, not in the paper (pp. 5–6). DC power flow is never mentioned | **confirmed** |

What the paper does **not** state is the `NaN`→0 mechanism. The Q ≡ 0 convention
is documented only in ENGAGE's code (`graph_gen.py`: *"Convert to tensor and
replace nan (q_mvar) with 0"*, with `pandapower==2.14.11` pinned in
`environment.yaml`). So the sentence "Q ≡ 0 is ENGAGE's convention" is supported
by their code and consistent with the paper scoring DC over all `D` dimensions,
but it is a code-level fact, not a paper-level one. That distinction is now
stated wherever the claim appears.

**ENGAGE's DC numbers, for calibration** (Table 3, p. 9; mean NRMSE over their
test cases):

| | reference GCN | DC PF | GNN ÷ DC |
|---|---:|---:|---:|
| Cross-context | 0.0464 | 0.0044 | 10.5× |
| OOD | 0.0094 | 0.0044 | 2.1× |

Our corrected four-quantity ratios are **3.8–54× cross-context** and **1.9–23×
OOD**. Same direction, and the OOD figure is the same order of magnitude as
ENGAGE's 2.1×; our cross-context penalty is larger, which is expected because our
grids are four *transmission* systems spanning 24→2 224 buses, while ENGAGE's are
ten SimBench *distribution* feeders. The discarded "8–224×" figure was, notably,
outside ENGAGE's range in the wrong direction — a second reason to distrust it.

Our absolute DC error (0.051–0.098 aggregate NRMSE) is 12–22× ENGAGE's 0.0044.
That is a grid-class difference, not a bug: on transmission systems with large
reactive flows, Q ≡ 0 is a much worse prediction relative to the mean per-dimension
range than it is on distribution feeders.

## 2. The metric — our implementation matches ENGAGE's Equation 3

ENGAGE (p. 5) defines

```
NRMSE = sqrt( (1/N) Σ_i (1/D) Σ_j (Y_ij - Ŷ_ij)² )  /  ( (1/D) Σ_j (y_max,j - y_min,j) )
```

i.e. **one pooled RMSE over all output dimensions, divided by the mean of the
per-dimension ranges** — explicitly *"in order to balance the difference of scale
across feature dimension"*. `training_utils.nrmse_range` is exactly this. So the
audit's criticism A3 (the aggregate mixes MW, Mvar, p.u. and degrees, so voltage
is invisible in it) is a criticism of **ENGAGE's published metric**, which we
reproduced faithfully; it is not an implementation error on our side. The remedy
stands regardless: report the per-quantity table alongside it, which we do (§5 of
`Regime_comparison_results.md`), and which ENGAGE does not.

One estimator caveat this check exposed, now stated in §7 of the results doc: our
`PVtheta` column is the **mean of the three per-quantity NRMSEs** (each normalised
by its own range) on both sides of the comparison, *not* Equation 3 restricted to
three columns. Those are different numbers — for DC on Regime A, 0.084 for the
mean-of-quantities versus 0.018 for pooled Equation 3 (`dc_nrmse_PVtheta` in
`results/analysis/dc_baseline_regime_*.csv`). The mean-of-quantities version is
dominated by voltage magnitude, because V's own range is tiny. The comparison is
apples-to-apples (a unit test enforces the same estimator on both sides) but the
ratios must be read as "V-dominated", and the pooled Equation 3 version cannot be
computed for the GNNs without re-scoring their predictions, which for ARMA and
NNConv would mean retraining.

## 3. Known values at inference — both papers do what we do

The audit's A3 also noted that 2 of the 4 target columns per bus are known inputs
re-injected into the output. Both source papers do exactly this:

- ENGAGE: *"We train the model freely for 500 epochs, backpropagating with an MSE
  loss; however, we incorporate the known values at inference time"* (p. 5).
- PowerGraph: *"if a variable is known, we mask it during training, and masked
  values are indicated with grey cells"* (Figure 1, p. 4).

So the protocol is standard in this literature and our masking follows it. What
remains valid from A3 is only the reporting problem: an aggregate that includes
re-injected ground truth flatters every model, so the fraction of genuinely
predicted entries must be stated (it is, in §5) and the per-quantity table must
carry the interpretation.

## 4. Normalization — CORRECTED: PowerGraph-Node does normalize, in code

The first version of this section said "neither paper normalizes node features".
That is right for ENGAGE and **wrong for PowerGraph-Node**. Neither *paper text*
mentions a scaler — ENGAGE relies on the metric's range normalization to
*"balance the difference of scale across feature dimension"* (p. 5), and
PowerGraph reports MSE per quantity (pp. 5–6, Fig. 3) — but PowerGraph-Node's
released code max-abs-normalizes both X and Y per dimension
(`code/dataset/powergrid.py`: `N_norm = N / maxsX`, `Y_norm = Y_o / maxsY`),
trains in normalized space, and de-normalizes only for reporting
(`code/train_gnn.py`: `batch_preds.squeeze() * batch.maxs`).

So A2 *is* a deviation from the benchmark our grids come from, our documentation
error stands (`transmission_graph_gen.py`'s header and design decision D9 claim
per-unit node features, which is false of the node tensors), and the substantive
concern is larger than "cross-grid magnitude": in raw units voltage magnitude
receives ~1e-8 of the training gradient and is not learned at all. Full
assessment, measurements and options: `docs/Normalization_assessment.md`.

## 5. Training-protocol deviations from both papers (deliberate, now stated)

| | ENGAGE | PowerGraph | This study |
|---|---|---|---|
| Grids | 10 SimBench distribution feeders, 300 samples each (p. 4) | IEEE24, IEEE39, IEEE118, UK transmission (p. 3) | the 4 PowerGraph transmission grids |
| Architectures | one 8-layer GCN + 3 positional-encoding variants (p. 5) | GCN, GAT, GINe, Transformer (p. 5) | those 4 + ARMA + NNConv |
| Search space | none reported (8 layers, hidden 64 "as determined by hyperparameter tuning", p. 5) | MPL ∈ {1,2,3} × hidden ∈ {8,16,32}, lr 1e-3 (p. 5) | layers ∈ {2,3,8} × hidden ∈ {32,64,128} × lr ∈ {1e-3, 3e-4} |
| Epochs | 500 | 50 with LR scheduler | 200, no scheduler |
| Batch size | 64 cross-context, 256 OOD (p. 6) | 32 node-level | 32 within/cross-context, 96 OOD |
| Split | not reported | 85 / 5 / 10 (p. 5) | 800 / 100 / 100 per grid |
| Loss | *"MSE loss"* (p. 5) — but their code uses inverse-target-norm-weighted MSE | MSE | inverse-target-norm-weighted MSE (follows their **code**) |
| Seeds | not reported | 5 (p. 5) | 5 (NNConv 3) |

Two things worth flagging in that table. First, ENGAGE's paper says plain MSE
while their released `training_utils.py` weights each sample by
`1/‖target‖`; we followed the code, so our loss does not match the paper's text,
and that is why plain MSE is also reported as a metric. Second, PowerGraph's
hidden dimensions top out at 32 and ours start at 32 — our models are
substantially larger than the PowerGraph baselines, so our in-distribution errors
are not directly comparable to their Table 4.

## 6. Net effect on the study's claims

| Claim | After reading the papers |
|---|---|
| "DC beats every GNN out of distribution" | **stands**, and is ENGAGE's own headline finding; our 1.9–23× brackets their 2.1× |
| "Q ≡ 0 is the ENGAGE-comparable convention" | **stands**, sourced from their code plus the paper's single all-dimension NRMSE; the code-level provenance is now labelled |
| "PowerGraph has no DC baseline to match" | **stands** (their non-GNN baseline is GBT, website only) |
| "DC's g-score is an artifact of ΔMMD = 0" | **stands**, and is literally ENGAGE's Table 3 |
| "Our aggregate metric hides the physics" | **stands as a limitation**, but it is ENGAGE's published metric reproduced faithfully, not a coding error |
| "Known-value re-injection inflates the numbers" | **stands as a reporting caveat**; both papers do the same thing |
| "Node features are per-unit" (old docstring / D9) | **false**; ENGAGE does not normalize either, but PowerGraph-Node's code does (max-abs on X and Y) — see `docs/Normalization_assessment.md` |
| "DC beats the GNNs even in-distribution on P/V/θ by 23–108×" | **valid only for the mean-of-per-quantity estimator**, which is V-dominated and is *not* ENGAGE's Equation 3; now labelled as such |
