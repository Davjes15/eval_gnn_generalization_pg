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

The cross-grid confound above still stands (nominal load: 2,850 / 6,254 / 3,733 / **56,326** MW),
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

`Δ_MMD` is a property of the data, so it is one constant per arm: 0.516485 (cross-context),
0.352837 (OOD), repeated down the whole column. Hence

```
g = μ + σ·log(Δ+1)/Δ = μ + 0.8064·σ  (cross-context)
                     = μ + 0.8563·σ  (OOD)
```

Checked against the stored value: ARMA OOD seed 0, 0.127976 + 0.8563 × 0.024694 = 0.14912 =
`g_score`. With one fixed dataset the MMD term cannot reorder architectures; it only does work
when comparing dataset designs or protocols. ENGAGE never claimed otherwise, and I did not say
it out loud. This belongs in the paper as a protocol observation.

## A7 — MMD details. CONFIRMED (low–medium)

`mmd()` uses full Gram means including the diagonal, i.e. the **biased V-statistic**. The
docstring says "unbiased-ish", which is not a defensible description — either name it biased
or switch to the U-statistic. Descriptors are purely topological (degree histogram, normalized
Laplacian spectrum), so they are blind to the ~20× electrical/scale shift in A2 — a distance
that cannot see the dominant shift is a weak covariate. Median-heuristic bandwidth is refit per
pair; defensible, but must be stated. Pooled OOD MMD (D14) is implemented as documented.

## A8 — Statistics are weaker than the doc implies. CONFIRMED, and worse

The auditor noted τ = 0.32 at n = 6 is inside the null sd (0.355). I ran the test that was
missing: a permutation test over model labels, on the 12 (grid, seed) cells where all six
architectures exist, mean Kendall τ across cells as the statistic, all 720 relabelings as the
null.

| Comparison | Observed mean τ | Permutation p |
|---|---:|---:|
| Regime A ↔ cross-context | −0.022 | 0.91 |
| Regime A ↔ OOD | 0.222 | 0.21 |

So the A↔OOD correlation is **not distinguishable from zero**, and the doc's "weak but
positive information" is an over-read that must be removed. The cross-context claim ("no
information") is safe. Note this makes the paper's thesis cleaner, not weaker: fixed-topology
ranking predicts neither transfer arm.

Also confirmed: n = 4 grids with size, density and load scale all confounded; NNConv has 3
seeds while being the highest-variance model; and seeds vary training init only — there is no
data-generation seed variance anywhere, so all error bars understate total uncertainty.

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
