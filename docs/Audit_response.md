# Response to the external audit (verified point by point)

Every finding was re-tested against the code and the committed/on-disk artifacts on this
machine (pandapower 3.5.4, the same `data_a` / `data_full` used for the reported tables).

**Bottom line: the audit is correct on all eight points.** Three of them (A1, A2, A3) do
invalidate statements currently written in `docs/Regime_comparison_results.md`. Two of the
findings (A3, A6) are results rather than bugs. One point (A5) I can resolve more precisely
than the auditor could, and it is *worse* than stated for the arm that matters. One point
(A8) I strengthened with a test the auditor did not run, and it goes further against our
current wording.

Nothing found here changes the training runs: no model needs retraining for A1, A4, A6, A7,
A8. A2 and A3 need an eval/regeneration pass.

**Update 2026-07-18.** The first version of this response argued about ENGAGE's and
PowerGraph's conventions from their *released code only*, because the publication sites are
outside this session's network allowlist. Both papers were then supplied directly and every
such claim is now verified against them in [`Paper_verification.md`](Paper_verification.md).
Net effect: the DC-convention reasoning holds and is strengthened (ENGAGE publishes DC ratios
of 10.5× cross-context and 2.1× OOD, bracketing ours), A3 turns out to describe *standard
practice in both papers* rather than a deviation — which sharpens it into a reporting duty
rather than a bug — and one number I quoted (23–108×) had to be relabelled as an estimator
artefact.

**Update 2026-07-18 (second pass, A2).** I first wrote that A2 was also standard practice in
both papers. That is wrong: PowerGraph-Node's released code normalizes X and Y (max-abs per
dimension), and so do the other AC-power-flow GNN code bases I could reach. A2 stands as a real
defect, its critical consequence is that voltage magnitude receives ~1e-8 of the training
gradient, and the audit's suggested remedy (per unit) would not have fixed it. See the A2
section below and [`Normalization_assessment.md`](Normalization_assessment.md).

---

## Remediation status (updated as work lands)

| # | Finding | Status | What closes it | Retraining? |
|---|---|---|---|---|
| A1 | DC baseline carried AC reactive power | **closed** | Q ≡ 0 enforced at generation *and* at scoring (`apply_dc_convention`); `nrmse_PVtheta` added as the fair-to-DC secondary aggregate; `pandapower==3.5.4` pinned; regression tests | no |
| A2 | Node features/targets never scaled | **closed** | `normalization.py` + `experiments.py --normalize pu_zscore`, de-normalized before scoring (Decision 20). Final campaign complete: 6 architectures × 3 arms, 336 checkpoints, tables regenerated in `results_norm/analysis/` and reported in `docs/Normalization_results.md` §4. In-distribution voltage NRMSE 5.8–27 → 0.001–0.015 | **done** |
| A3 | Aggregate metric hides the physics | **closed** | `physics_metrics.py` (per-quantity, predicted-entries-only, p95/p99/max tails, V-violation / false-secure / false-alarm rates) + `eval_checkpoints.py` replay driver + `tests/test_physics_metrics.py`. All 336 checkpoints replayed → 672 rows in `results_norm/physics/physics_metrics.csv`, summarised in `docs/tables/physics_summary_norm.csv` | no |
| A4 | Not reproducible from the repository | **closed** | `docs/Reproducibility.md`: pinned versions, exact generation/training commands, `dataset_src.csv` provenance + realised split windows, `checkpoint_index.py` (path, size, SHA-256, parameter count) and one-command replay via `eval_checkpoints.py`; the 21 tuning artifacts the configuration tables cite are now committed. Remaining limits (one data realization, four grids, NNConv seeds) are stated, not fixed | no |
| A5 | Split hygiene (shared demand snapshots in Regime B) | **closed** (data and training) | `--time_split blocked` (Decision 21), gate H in `validate.py`, `tests/test_split_hygiene.py`; `data_full_v2` generated and gate-passed (800/100/100 per grid, disjoint windows, one-day gap) | **yes** (Regime B only, once) |
| A6 | g-score affine in (mean, sd) here | **closed** | `docs/Generalization_score_and_MMD.md` §1: the identity `g = μ + 0.806σ` (cross-context) / `μ + 0.857σ` (OOD) verified against the committed artifacts to 1e-4, and rank-by-g ≡ rank-by-mean (τ = 1.0) in both arms | no |
| A7 | MMD estimator details unstated | **closed** | `docs/Generalization_score_and_MMD.md` §2: biased V-statistic named as such, `unbiased=True` U-statistic added, per-pair median bandwidth stated, electrical `reactance_histogram` descriptor added, all 16 grid pairs × 3 descriptors × 2 estimators tabulated in `docs/tables/mmd_data_full_v2.csv` | no |
| A8 | Statistics weaker than claimed | **closed** (with L2 accepted) | permutation test moved into `rank_analysis.py` (exact, all 720 relabellings) and re-run on the normalized campaign: A↔cross-context τ = 0.067, p = 0.72; A↔OOD τ = 0.000, p = 1.00 (`results_norm/analysis/rank_permutation_test.csv`); wording corrected. The NNConv 3 → 5 seed expansion is declined as accepted limitation L2 | additive only, declined |

**Non-finite accounting for the final tables.** Of the 672 replayed rows, exactly two are
non-finite — `gcn`, seed 1000, cross-context IEEE39→IEEE118 and IEEE118→IEEE24 — and they are a
reproducible architecture failure, not a numerical accident: GCN's learned scalar edge weight can
go negative on an unseen grid, so `GCNConv`'s symmetric normalization takes `deg^(-1/2)` of a
negative weighted degree (`docs/Normalization_results.md` §4bis). They are listed in
`results_norm/analysis/nonfinite_runs.csv`, and `rank_analysis.py` voids the whole
(model, train grid, seed) cell rather than averaging the surviving test grids, which would have
reported the run as better than the transfer where it broke. The other 168 non-finite *cells* are
`vm_false_secure` on splits with zero true voltage violations, i.e. 0/0, reported as undefined.

**Ordering constraint that drove the plan:** A5 changes the Regime B *data*, and A2 changes the
*training representation*. Doing them in either order separately would mean training the same six
architectures twice on the cross-context and OOD arms. They are therefore bundled: Regime A (which
uses `data_a`, unaffected by A5) trains under A2 immediately; Regime B waits for `data_full_v2` and
then trains once. A3/A4/A6/A7 are evaluation and documentation passes over the resulting
checkpoints, which is why checkpointing every final run was made a requirement.

---

**A second audit (B1–B8) followed this one.** Its point-by-point response, the fixes applied
without retraining, and the limitations accepted knowingly are at the end of this document,
under *Second audit*.

---

## A1 — DC baseline reactive power is the AC ground truth. CONFIRMED (critical)

Verified empirically:

```
runpp(net) -> deepcopy -> rundcpp:  res_bus.q_mvar identical to AC at every bus (allclose True)
fresh net  -> rundcpp:              res_bus.q_mvar is all NaN
```

`rundcpp` never writes `q_mvar`, so `deepcopy` of an already-AC-solved net carries the AC
answer through. P, V and θ in `dc_pf` are genuine DC quantities (P differs from AC by up to
44 MW on case39; V is gen setpoints + flat elsewhere) — only the Q column is contaminated.

This is the true reason `dc_nrmse_Q = 0.0`. The explanation in
`Regime_comparison_results.md` §7 ("DC-PF carries no reactive power, Q ≡ 0") is wrong: the
stored value is not zero, the *error* is zero because DC was handed the label. Same wrong
explanation in `docs/Findings.md`.

Impact, measured on the Regime A test sets by re-scoring the stored `dc_pf` with the Q
column replaced (no regeneration needed):

| Grid | DC aggregate NRMSE as reported | with Q ≡ 0 | with Q excluded (P/V/θ) |
|---|---:|---:|---:|
| IEEE24 | 0.0157 | 0.0651 | 0.0177 |
| IEEE39 | 0.0091 | 0.0979 | 0.0129 |
| IEEE118 | 0.0244 | 0.0515 | 0.0315 |
| UK | 0.0177 | 0.0665 | 0.0197 |

So the reported DC baseline is optimistic by ~2–10× under the Q ≡ 0 convention. Direction of
the consequences:

- within-grid ("every GNN beats DC"): the bar was *too high*, so the claim survives and gets
  stronger;
- cross-context / OOD ("DC beats every GNN by 8–224×"): the gap shrinks by roughly the same
  factor as DC's error grows (to roughly 2–56×). DC still wins every OOD fold, but the
  headline number as written is not defensible.

**FIXED 2026-07-18.** Convention decided by looking at what ENGAGE actually did rather than
by preference: their `graph_gen.py` uses the same deepcopy→`rundcpp` pattern with the comment
*"replace nan (q_mvar) with 0"* and pins `pandapower==2.14.11`, where the column really did
arrive as NaN. So Q ≡ 0 is ENGAGE's convention and our bug is a version regression of that
one line. PowerGraph-Node publishes no DC baseline, so there is no competing convention.

Both papers have since been read directly and every claim of that kind is checked
line by line in [`Paper_verification.md`](Paper_verification.md). Summary for this
item: ENGAGE's paper scores DC PF over all output dimensions in the same column as
the GNNs (Table 3), reports DC beating the GNNs by 10.5× cross-context and 2.1×
OOD, and gives DC `ΔMMD = 0` so its g-score equals its mean — all consistent with
what is written here. The `NaN`→0 mechanism itself appears only in their code, so
that part of the provenance is now labelled code-level rather than paper-level.
PowerGraph's only non-GNN baseline is Gradient Boosted Trees, website-only.

Implemented:

- `transmission_graph_gen._build_sample` zeroes the reactive column explicitly after
  `rundcpp`, so the convention no longer depends on the pandapower version;
- `training_utils.apply_dc_convention` re-applies it at scoring time, which corrects the
  already-generated `data_a` / `data_full` without regeneration (P, V and θ are provably
  untouched: a DC solve on a fresh net and on a copy of an AC-solved net agree on those
  three columns);
- `training_utils.test_dc_pf(full=True)` additionally reports `nrmse_PVtheta`, the Q-excluded
  aggregate, so both conventions are available;
- `recompute_dc_baseline.py` regenerates the baseline per dataset — the DC table is now
  **per arm**, because Regime A (`data_a`) and Regime B (`data_full`) are different data and
  the single shard table was being used for both;
- `recompute_tables.py` takes `--dc_regime_a` / `--dc_regime_b` and emits a `PVtheta` row
  next to the four-quantity rows; the stale per-shard `dc_baseline.csv` files are no longer
  used for values;
- `pandapower==3.5.4` pinned in `requirements.txt`;
- `docs/Regime_comparison_results.md` §7 rewritten, `docs/Findings.md` §3 marked superseded.

The recomputed result confirms the direction stated above and adds one finding: under the
Q-excluded (P/V/θ) convention **DC beats every GNN in every arm, including in-distribution**,
because the per-quantity V-NRMSE is where the GNNs are weak. The four-quantity aggregate is
the only view under which the GNNs win in-distribution. Both are now printed side by side so
neither can be quoted alone.

One correction to my own wording after reading ENGAGE's Equation 3: the "23–108×" figure I
quoted for the Q-excluded convention comes from averaging the three separately
range-normalised per-quantity NRMSEs, which is **dominated by voltage magnitude** and is not
Equation 3 restricted to three columns (that pooled form gives 0.018 rather than 0.084 for DC
on Regime A). The comparison is still apples-to-apples — a unit test enforces the same
estimator on both sides — but the multiplier is an artefact of the estimator, so the claim is
now stated as "the GNNs lose to DC on voltage magnitude even in-distribution", which the
per-quantity table shows directly.

## A2 — Node features and targets are never normalized. CONFIRMED (critical)

`engage_contract.get_node_features` emits raw `p_mw, q_mvar, vm_pu, va_degree`. Only the edge
attributes are per-unit (`z = vn_kv² / sn_mva`). There is no scaler anywhere in the repo.

The header of `transmission_graph_gen.py` claims "features are ENGAGE per-unit" and D9
demands a per-unit basis for cross-grid comparison. **Both statements are false of the node
tensors, and I wrote both.**

Measured target ranges on the Regime A test sets (native units):

| Grid | P range (MW) | Q range (Mvar) | θ range (deg) |
|---|---|---|---|
| IEEE24 | −660 … 1,556 | −563 … 128 | −8.6 … 54.8 |
| IEEE39 | −830 … 2,088 | −1,725 … 184 | −7.8 … 97.4 |
| IEEE118 | −5,877 … 295 | −3,042 … 102 | −178.7 … 0.0 |
| UK | −11,306 … 36,527 | −12,786 … 766 | −1.7 … 165.5 |

UK is an order of magnitude outside every other grid's support. So cross-context and
leave-one-grid-out error mixes topology shift with an input/target magnitude shift of ~20–30×,
and the sentence "cross-grid degradation measures topology generalization" is not currently
supportable. This is the audit's most consequential point for the paper's story.

Reassessment (2026-07-18), full evidence in
[`Normalization_assessment.md`](Normalization_assessment.md). Three corrections to what is
written above, one of them to my own paper check:

1. **Per-unit conversion is not the fix.** All four cases carry `net.sn_mva = 100.0`, so
   converting P and Q to p.u. divides them by a single constant: no ratio, no learnability and
   no cross-grid spread changes. Worth doing for convention (Hansen et al. do it), useless as a
   remedy.
2. **The decisive defect is intra-sample, not cross-grid.** Share of `weighted_mse_loss`
   contributed by each target dimension at the mean predictor: IEEE24 P 0.831 / Q 0.148 /
   θ 0.021 / **V 5.3e−08**; UK **V 1.2e−11**. Voltage magnitude is not optimized, which is
   exactly what the per-quantity table shows (V NRMSE 5.8–27 in-distribution, i.e. worse than
   the constant `V ≡ 1.0`). This makes A2 the *cause* of half of A3's finding.
3. **My earlier paper check was half wrong.** ENGAGE indeed does not normalize (and gets away
   with it: its SimBench LV/MV feeders have MW injections of the same order as `vm_pu ≈ 1`, and
   its `1/‖y‖` loss weight equalizes buses). But **PowerGraph-Node's released code does
   normalize** — max-abs per dimension on both X and Y, training in normalized space,
   de-normalized only for reporting. So does PowerFlowNet (z-score, train statistics) and the
   KIT-IAI pretrained-GNN work (p.u. *and* z-score). Standard practice in the field is p.u. plus
   a per-dimension scaler with physical-unit reporting; ENGAGE is the exception.

The cross-grid confound above still stands (nominal load 2,850 MW / 24 buses for IEEE24,
6,254 MW / 39 buses for IEEE39, 3,733 MW / 118 buses for IEEE118, **56,326 MW** / 29 buses for
UK, read from `transmission/cases/*.mat` via `transmission_grids.load_case`),
but no unit system removes it — the UK system genuinely moves an order of magnitude more power.
Isolating topology needs a per-grid physical base (e.g. nominal load, which is input data and so
leak-free); a train-grid-fitted scaler keeps the size shift inside the OOD shift, which is
honest but means the claim must read "generalization to an unseen system".

Recommended sequence (not implemented, no retraining started): scaler behind a flag defaulting
to today's behaviour → pilot on gcn + arma, seeds 0/100, within + cross-context (hours) → full
retrain as the primary protocol only if the pilot confirms the mechanism → per-grid load base as
the OOD-isolating ablation.

## A3 — The aggregate metric hides the physics. CONFIRMED (critical, and a result)

Two mechanisms, both confirmed in code:

**(i) Known-value re-injection.** `models.py::inference` overwrites, per bus type, exactly two
of the four target columns with ground-truth inputs. Precisely:

| Bus type | Copied at eval | Genuinely predicted |
|---|---|---|
| PQ | P, Q | V, θ |
| PV | P, V | Q, θ |
| slack | V, θ | P, Q |

Paper check (2026-07-18): re-injection is what both source papers do — ENGAGE *"incorporate[s]
the known values at inference time"* (p. 5) and PowerGraph masks known variables during
training (Fig. 1, p. 4) — and the pooled mixed-unit aggregate **is** ENGAGE's Equation 3,
reproduced faithfully. So neither mechanism is an implementation error; what stands is the
reporting duty (state the predicted fraction, always show the per-quantity table), which is
why this item is filed as a result rather than a bug. Details in `Paper_verification.md`.

So in the **P** column only the slack bus (1 of N) is ever predicted; **Q** is predicted only
at PV + slack, **V** only at PQ buses, **θ** at PQ + PV. Metrics are then computed over all
buses, i.e. `nrmse_P = 2.1e-4` is largely a count of copied labels, not a measure of learning.
The decoder is ENGAGE's and is fine; scoring over the re-injected entries is not.

**(ii) Mixed-unit pooling.** `nrmse_range` takes one RMSE over MW / Mvar / p.u. / degrees and
divides by the *mean* of the four ranges, which the MW/Mvar columns dominate. Voltage is
numerically invisible. The same imbalance is in the loss: `weighted_mse_loss` weights each bus
by `1/‖y_row‖`, and that norm is set by P and Q in MW, so voltage contributes ~10⁻⁶ of the
gradient. The models are trained almost entirely to predict megawatts.

The auditor's constant-predictor comparison holds. Measured `MAE(V ≡ 1.0)` on the Regime A
test sets: IEEE24 0.0266, IEEE39 0.0439, IEEE118 0.0256, UK 0.0020 (mean ≈ 0.0245 p.u.). The
best GNN is ARMA at 0.0829, the worst GAT at 0.302 — **every architecture is 3–12× worse than
a constant on voltage, in-distribution, on its own training grid**, while the DC baseline is
0.002–0.039. The claim "every architecture fits power flow well within grid" must go.

## A4 — Not reproducible from the repository. CONFIRMED (high)

`.push_branch.py` commits an explicit file list through the GitHub Data API, so only what I
named was ever pushed. On disk but *not* in the repo: `data_a` (40 MB), `data_full`
(symlink to `full_run/data`), `ckpt_a`/`ckpt_b` (141 MB, gcn/gat/gin/transformer only), and
every `tuning.csv` / `tuning_summary.csv` / `tuning_per_grid_argmin.csv` that
`Model_configurations.md` §1 cites. Those citations are dead references as committed — the
selection tables are currently unbacked. The tuning CSVs are a few kB each and can simply be
pushed; the data and checkpoints need a release plan, not a git commit.

`requirements.txt` is also unpinned (`torch>=2.2`, `pandapower>=2.14`), which matters exactly
because A1 depends on `rundcpp` behaviour and the cases emit `from_mpc` trafo-reclassification
warnings on load. Pinning to the versions actually used (pandapower 3.5.4) is trivial.

## A5 — Split hygiene. CONFIRMED, and I can be more precise than the audit

The auditor found the leak in `full_run/data` and could not check the final datasets. I can:

| Dataset | Arms it feeds | Shared demand snapshots train↔test | Duplicate train rows |
|---|---|---|---|
| `data_a` | Regime A (within-grid) | 0 on all four grids | 0 |
| `data_full` | Regime B (cross-context, OOD) | IEEE39 4, IEEE118 3, UK 1, IEEE24 0 | 1–4 per grid |

So Regime A was generated with `--unique_demand` and is clean; **the leak is in exactly the
Regime B data, i.e. the arms the paper's thesis rests on.** It is small (≤4 % of a test split,
and the contingency differs) but it has to be stated, and `--unique_demand` defaulting to
false is a footgun. Adjacent-timestep near-duplicates are not controlled at all; a blocked
time split would be the right fix.

## A6 — The g-score is affine in (mean, sd) here. CONFIRMED exactly

`Δ_MMD` is a property of the data, so it is one constant per arm: in the raw-unit campaign
0.516485 (cross-context) and 0.352837 (OOD), repeated down the whole column (the normalized
campaign's are 0.521637 and 0.349052, giving 0.805 and 0.858 below). Hence

```
g = μ + σ·log(Δ+1)/Δ = μ + 0.8064·σ  (cross-context)
                     = μ + 0.8563·σ  (OOD)
```

Checked against the stored value: ARMA OOD seed 0, 0.127976 + 0.8563 × 0.024694 = 0.14912 =
`g_score`. With one fixed dataset the MMD term cannot reorder architectures; it only does work
when comparing dataset designs or protocols. ENGAGE never claimed otherwise, and I did not say
it out loud. This belongs in the paper as a protocol observation.

**Closed.** Now checked over *all* stored rows rather than one: the identity holds to 1.0e-4
(cross-context, 33 rows) and 3.5e-5 (OOD, 28 rows), and ranking the six architectures by
`g_score` versus by `mean_nrmse` gives Kendall τ = 1.0 in both arms — the distance term moves no
position. Per train grid (`gscore.csv`) `Δ = 0`, so there `g = μ` exactly. Written up with the
derivation and a reproduction snippet in `docs/Generalization_score_and_MMD.md` §1.

## A7 — MMD details. CONFIRMED (low–medium)

`mmd()` uses full Gram means including the diagonal, i.e. the **biased V-statistic**. The
docstring says "unbiased-ish", which is not a defensible description — either name it biased
or switch to the U-statistic. Descriptors are purely topological (degree histogram, normalized
Laplacian spectrum), so they are blind to the ~20× electrical/scale shift in A2 — a distance
that cannot see the dominant shift is a weak covariate. Median-heuristic bandwidth is refit per
pair; defensible, but must be stated. Pooled OOD MMD (D14) is implemented as documented.

**Closed.** The estimator is now named as the biased V-statistic, `mmd(..., unbiased=True)` adds
the U-statistic (default unchanged so the committed CSVs stay reproducible), and the per-pair
bandwidth is stated. The measured bias: ~0.005 on different-grid pairs, but the entire value on
same-grid pairs (IEEE39 self-distance 0.0713 biased vs 0.0000 unbiased) — so any within-grid MMD
*floor* is an estimator artifact, while cross-grid comparisons are unaffected. For the blindness
to electrical scale, `mmd_utils.reactance_histogram` adds a `log10(x_pu)` branch-reactance
descriptor; `mmd_report.py` tabulates all 16 grid pairs under all three descriptors and both
estimators (`docs/tables/mmd_data_full_v2.csv`). The pair that makes the point: UK → IEEE24 is the
closest off-diagonal pair topologically (degree MMD 0.276) yet far apart electrically (0.833).
Details in `docs/Generalization_score_and_MMD.md` §2.

## A8 — Statistics are weaker than the doc implies. CONFIRMED, and worse

The auditor noted τ = 0.32 at n = 6 is inside the null sd (0.355). I ran the test that was
missing: a permutation test over model labels, on the 12 (grid, seed) cells where all six
architectures exist, mean Kendall τ across cells as the statistic, all 720 relabelings as the
null. It is no longer an ad-hoc script — `rank_analysis.permutation_test` runs it as part of
the standard analysis and writes `rank_permutation_test.csv`, with two regression tests
(`tests/test_rank_analysis.py`) pinning the exact p-value on a known reversal.

| Comparison | Observed mean τ | Permutation p | campaign |
|---|---:|---:|---|
| Regime A ↔ cross-context | 0.067 | 0.72 | normalized (final) |
| Regime A ↔ OOD | 0.000 | 1.00 | normalized (final) |
| Regime A ↔ cross-context | −0.022 | 0.91 | raw-unit (ablation) |
| Regime A ↔ OOD | 0.222 | 0.21 | raw-unit (ablation) |

So neither correlation is **distinguishable from zero**, and the doc's earlier "weak but
positive information" reading of the raw-unit A↔OOD τ is an over-read that must be removed —
under the final protocol that τ is exactly 0.00. Note this makes the paper's thesis cleaner,
not weaker: fixed-topology ranking predicts neither transfer arm.

The 12 cells are 4 grids × 3 seeds: only the three seeds NNConv was run at have all six
architectures present, which is the concrete cost of the reduced NNConv seed count.

Also confirmed: n = 4 grids with size, density and load scale all confounded; NNConv has 3
seeds while being the highest-variance model; and seeds vary training init only — there is no
data-generation seed variance anywhere, so all error bars understate total uncertainty.

---

# Second audit (B1–B8): response, fixes and knowingly-accepted limitations

Every one of the eight new findings was re-checked against the code, the committed CSVs and
what the training campaign actually did. **All eight are correct**; two carry small numerical
slips that do not affect the conclusion (the auditor's B2 τ values recompute as 0.69 / 0.55 /
0.60 rather than 0.73 / 0.60 / 0.60, a tie-handling difference; and A3's "partly closed"
status was charitable — absent AC residuals it was open, and is now closed by B1).

The decision taken on this round, deliberately and on the record: **no retuning and no
retraining.** Everything that can be fixed from the 336 saved checkpoints and the committed
tables is fixed; the two items that would need training are accepted as limitations and are
listed as such below, so a later reviewer sees a choice rather than an oversight.

| # | Finding | Status | What closes it |
|---|---|---|---|
| B1 | No AC feasibility check; the promised residual/thermal metrics were never delivered | **closed** | `ac_feasibility.py` (Ybus P/Q residual with the shunt double-count removed, loading of every in-service branch against its rating), `eval_checkpoints.py --feasibility`, `tests/test_ac_feasibility.py` (calibrated against pandapower's own solution: true-state residual ≤ 2.8e-2 MW, loading matches `res_line` and `res_trafo` to 1e-12). All 336 checkpoints replayed; results in `docs/Normalization_results.md` §4.6 and `docs/tables/ac_feasibility_norm.csv` |
| B2 | Regime A → Regime B confounds protocol and grid | **closed** | `rank_analysis.py` now carries four arms (`regime_a`, `regime_b_diagonal`, `cross_context`, `ood`) and a `protocol_decomposition.csv` splitting the gap into a same-grid factor (1.5–10.5×, which bundles two design changes — see L8) and an unseen-grid factor (34–382×) |
| B3 | The rank inference is stated backwards | **closed** | pooled leaderboard τ (0.60 / 0.20) reported alongside the per-cell mean τ in `rank_correlation_pooled.csv`; wording changed to rank *instability* and "no reliable guarantee", and the claim that p = 1.00 proves a zero correlation is removed |
| B4 | Hyperparameters selected under the raw-unit loss | **accepted limitation** | see L1 below — not retuned |
| B5 | Non-finite policy inconsistent; a diverging model was rewarded | **closed** | `training_utils.gscore_row` voids an incomplete cell and records `n_expected` / `n_finite` / `finite_rate`, the same void-the-cell policy the ranking uses (`tests/test_gscore_policy.py`) |
| B6 | `gscore.csv` `mean_nrmse` was a trimmed median, `std_nrmse` identically 0 | **closed** | trimming removed (`bounds=0` on three unseen grids); the DC row is labelled `basis = one_point_per_grid` against the models' `unseen_pairs` |
| B7 | Only active demand varies; Q pinned at base case, undisclosed | **closed** | stated in `transmission_graph_gen._apply_demand` and in the limitations below; it mirrors PowerGraph's `gendataopf.m`, which also varies PD only |
| B8 | Merged metadata omits the objective; masked-input semantics undocumented | **closed** | `gather_results.py` refuses shards with different `--normalize` and records `normalize`, `models` and per-architecture `arch_config` in the merged `summary.json`; the mask sentinel's mode-dependent meaning is documented in `normalization.py` |

## Knowingly-accepted limitations

These are choices, not omissions. Each states what was not done and why.

**L1 — hyperparameters were selected under a different objective (B4).** The sweep in
`tune_budget.py` scored candidates in raw units, where the loss is dominated by MW/Mvar and
voltage contributes ~1e-8 of the gradient; the final campaign trains and reports under
`pu_zscore`, where the four quantities count roughly equally. The configurations were reused
rather than re-selected. Re-tuning all six architectures costs ~1 day, and any change it
produced would require retraining the affected architectures across three arms (1.5–4 days);
that was declined. Mitigating evidence, which is reasoning and not measurement: the procedure
was identical for all six architectures, so it does not favour any one of them, and all six
selected the boundary of the search grid (hidden = 128, lr = 1e-3) with only depth differing,
which is the pattern of a criterion that was not discriminating finely. **A reported number
may therefore be slightly worse than that architecture's best achievable under the final
objective.** The paper's claim is about the ranking's transfer, not about each architecture's
optimum.

**L2 — NNConv is replicated at 3 seeds, not 5.** A deliberate compute trade-off recorded as
Decision 17 in `PowerGraph_to_ENGAGE_design_decisions.md`: NNConv emits a full 128×128
transform per edge, an IEEE118 run takes ~3 h, and five seeds across three arms would be
~1.5–2 days of extra wall clock for a variance estimate. The consequence is concrete and is
stated wherever it bites: NNConv contributes 12 rows per arm where the other five contribute 20
(48 vs 80 for cross-context; `results_norm/all_within/within_grid.csv`,
`all_cross/cross_context.csv`, `all_ood/ood.csv`), so every pooled mean and every ranking it
appears in rests on three seeds; the permutation test has 12 complete (grid, seed) cells rather
than 20, because only the three NNConv seeds have all six architectures present; and NNConv's
spread is the least reliable in the set.

**L3 — GCN's unseen-grid NaN is reported, not fixed.** Two of 448 cross-context rows are
non-finite from a real architectural defect (negative learned edge weight → `deg^(-1/2)` of a
negative weighted degree; `Normalization_results.md` §4bis). The softplus that fixed the same
defect in ARMA would remove it, at the cost of re-tuning and retraining GCN. Left as ENGAGE
ships it, because an architecture that can return an undefined answer on an unseen grid
without warning is one of the more informative results here. The affected cells are voided,
never averaged away.

**L4 — four grids, and the g-score is underpowered.** IEEE24 / IEEE39 / IEEE118 / UK differ in
size, density and load scale simultaneously (24 / 39 / 118 / 29 buses at 2,850 / 6,254 / 3,733 /
56,326 MW nominal load), so "unseen grid" is not isolated topology transfer but transfer to an
unseen system — scale, structure and the Regime B protocol change together. Each per-training-grid g-score cell has three unseen points and each OOD cell has
one grid, so those spreads are descriptive rather than inferential, and the ENGAGE-style
g-score is affine in (mean, sd) at this n (A6).

**L5 — five seeds measure training randomness only.** They are repeated training runs, not a
random sample from a population: they support "the gap is larger than run-to-run noise" and
they do not support classical significance statements. There is no data-generation seed
anywhere, so every error bar understates total uncertainty. Only the rank correlation is
tested against a null, and that test is underpowered at 12 cells (it cannot detect a modest
true correlation, which is why the wording is "no reliable guarantee", not "no correlation").

**L6 — one data realization, and only active demand varies (B7).** Reactive demand stays at
each case's base value — `transmission_graph_gen._apply_demand` writes `net.load.p_mw` and
nothing else — mirroring PowerGraph's `gendataopf.m`; the load trajectory is one
realization of the PowerGraph-Node demand series, so nothing here estimates variability across
alternative demand histories.

**L7 — artifacts, now published (closes the open half of A4).** Both generated datasets
(~79 MB) are committed, and the 336 checkpoints ship as the `ckpt_norm.tar.gz` asset of
release `v1.0.0` (593 MB compressed, SHA-256 in `docs/tables/artifact_manifest.csv`), so an
outside reader can replay our exact weights on our exact tensors rather than only repeating
the procedure. What remains outside the repository is the PowerGraph-Node demand and case
files, which are the upstream dataset's to distribute and are required only for the
`--feasibility` pass and for regenerating data.

---

## What I got wrong, plainly

1. I wrote that DC has no reactive power and reported a zero that was a leak (A1).
2. I asserted per-unit node features in a module docstring and in D9's remedy, and never
   implemented them (A2) — then interpreted a scale-shift result as a topology-shift result.
3. I promoted a pooled mixed-unit NRMSE to headline and stated that all architectures fit
   power flow well in-distribution, when they are worse than a constant on voltage (A3).
4. I cited tuning artifacts in the docs that were never pushed (A4).
5. I did not check `--unique_demand` on the Regime B generation, and did not test the rank
   correlation against a null (A5, A8).

## Recommended order of work

| # | Item | Cost | Needs |
|---|---|---|---|
| 1 | Fix DC baseline + recompute `dc_baseline.csv`, `dc_comparison.csv`, correct §7 | ~1 h | a convention decision |
| 2 | Push tuning CSVs, pin `requirements.txt`, remove/repair dead references, release plan for data + checkpoints | ~1 h | — |
| 3 | State the g-score degeneracy and the biased MMD estimator; replace the τ over-read with the permutation test | ~1 h | — |
| 4 | Re-score metrics restricted to genuinely predicted entries, promote per-quantity physical units to primary | ~half a day | checkpoints exist for gcn/gat/gin/transformer; ARMA + NNConv need a re-run |
| 5 | AC feasibility: per-bus P/Q mismatch from predicted (V, θ), V-limit and line-loading violation rates, p95/p99/max tails | ~1 day | same eval pass as 4 |
| 6 | p.u. normalization ablation (regenerate + re-run 1–2 architectures, CC + OOD) — decides whether the OOD story is topology or units | ~1 day | compute |
| 7 | Regenerate Regime B with `--unique_demand` + blocked time split; commit `dataset_src.csv` | bundle with 6 | compute |
| 8 | NNConv to 5 seeds | hours | compute |

Items 1–3 are pure bookkeeping and change no experiment. Item 6 is the one that decides
whether the paper's central claim stands as written.

---

# Third audit (C1–C6): response and fixes

All six findings are correct. Three of them (C1–C3) are not disagreements about method: they
report that files the previous section cites were **never pushed to the branch**, which was
true. `.push_branch.py` commits an explicitly listed set of paths and the audit-2 push listed
only the documents; the code, the tests and the regenerated replay CSV existed in the working
tree and passed, which is exactly the state that produces a document citing an artifact a
reviewer cannot open. `.diff_branch.py` now reports what the working tree has that a branch
does not, and it is run after every push.

| # | Finding | Verdict | What closes it |
|---|---|---|---|
| C1 | `tests/test_ac_feasibility.py` is cited but absent from the branch | correct | pushed; the file was working-tree only |
| C2 | The committed `results_norm/physics/physics_metrics.csv` predates the feasibility pass, so the AC table cannot be regenerated | correct | pushed after the replay, with the `ac_*` columns present; `summarize_feasibility.py` fails loudly if they are missing |
| C3 | B8 is marked closed but neither change is in the tree | correct | `gather_results.py`, `experiments.py`, `normalization.py` and `tests/test_gather_results.py` pushed |
| C4 | The thermal check covers lines only | correct (his branch counts are slightly off) | `ac_feasibility.build_case` now carries every in-service branch, with a rating **per end**, and the test validates against `res_trafo.loading_percent` as well as `res_line`, including under a transformer outage |
| C5 | The table has no DC row and no reconstruction-floor row | correct, and the most useful item in the report | `dc_feasibility.py` scores the stored DC state and the labels through the same checker; both appear in `docs/tables/ac_feasibility_norm.csv` |
| C6 | `protocol_factor` bundles the blocked split and the contingencies | correct | the column is renamed `same_grid_factor` and both `rank_analysis.py` and §4.6 state that it bounds the two changes jointly |

Two corrections to the report itself, neither affecting its conclusions:

* **C4's transformer counts.** They are 5/38 (IEEE24), 11/46 (IEEE39), **9/184** (IEEE118) and
  **4/90** (UK), not 11/186 and 13/99. The UK case therefore lost ~4 % of its branches from the
  screen, not 13 %.
* **C5's framing of the DC row.** DC power flow is scored, but its *reactive* residual is not a
  measure of the linearisation: DC fixes |V| = 1 and Q ≡ 0, so its Q residual is essentially the
  snapshot's reactive demand by construction. The comparable columns are the active-power
  residual and the thermal screening, and the previous section's claim that DC could not be
  scored at all was wrong.

Nothing in this round required retraining, and none was done: C4 and C5 are replay passes over
the same 336 checkpoints and the stored DC states, and C6 is a rename plus a stated limitation.
The limitation added by C6 is recorded as **L8** below.

**L8 — the same-grid step is not a clean protocol effect (C6).** Regime B differs from Regime A
in two ways at once: blocked temporal splits with a one-day gap, and N-1/N-2 line contingencies
(L9) instead of the intact topology. `same_grid_factor` therefore bounds their combined cost
(1.5–10.5× across the six architectures,
`results_norm/analysis/protocol_decomposition.csv`) and attributes it to neither. Isolating them
needs two further datasets (blocked-split-only and contingencies-only) and a retrained campaign
on each, which was declined along with the other training items.

---

## Fourth audit (verification pass) — D1–D4

The fourth report re-verified C1–C6 and closed all six. It raised three minor findings, all
correct and all handled here; none required retraining.

| # | Finding | Verdict | What closes it |
|---|---|---|---|
| D1 | The void-the-cell policy (B5) is not applied to the feasibility table: the two GCN cross-context cells with a non-finite NRMSE still report finite `ac_*` values, because `feasibility_metrics` averages over buses with `nanmean`, and they are averaged into `cc_unseen` | correct | `summarize_feasibility.valid_mask` applies the ranking's rule — the whole (arm, model, train grid, seed) group is voided — and the table carries `n_rows` / `n_voided`, so a cell whose every row was voided is reported as NaN with its count rather than as a mean |
| D2 | Means only; `cc_unseen`'s mean residual is driven by GIN, so the aggregate reads as typical | correct | the table adds a median and a max of both residual columns; §4.6 quotes the unseen-grid median (4,323 % of load) against the mean (8,174 %) and names GIN as the cause |
| D3 | `build_case` assumes the post-contingency network stays connected and raises `IndexError` otherwise; unreachable through the pipeline but undocumented | correct | `build_case` states the precondition and rejects an islanding outage with a `ValueError` naming it, with a regression test on the radial IEEE24 line |

Effect on reported numbers: D1 moves only the settings that contained the two voided GCN rows —
Regime B same-grid `dP` 380 → 354 % of load, unseen-grid GCN 5,825 → 4,769 % — and leaves every
other row unchanged. D2 and D3 change no reported value.

Two notes on the report's own framing:

* **D3 is unreachable for a stronger reason than given.** The generator does reject islanding
  before solving, but the default sampler also only ever removes **lines**: `dataset_src.csv`
  contains no transformer outage in either regime, and the 9-of-11 islanding transformer outages
  on IEEE39 are therefore unreachable twice over. (One of those 9 raises an `AttributeError` on
  the Ybus rather than an `IndexError`; both are now the same `ValueError`.)
* **Contingency wording.** Several documents said "N-1/N-2 contingencies" where the sampler
  draws random N-1/N-2 **line** outages; transformer outages are supported by the harvest path
  (`contingency_harvest.py`) but were not used. The wording is now explicit wherever the datasets
  are described. This narrows the topology distribution the results speak to and is recorded as
  **L9**.

**L9 — the topology distribution is line outages only.** Regime B's contingencies are random
N-1/N-2 outages of in-service *lines*; transformer and generator outages, busbar splits and
switching actions do not appear, and islanding cases are discarded rather than modelled as
separated systems. Transformers are covered by the thermal screen (C4) but never taken out.
