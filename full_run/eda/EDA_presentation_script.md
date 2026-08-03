# EDA — Presentation Script (~3 minutes)
### Exploring the data before and after generation, and how it shaped the pipeline

*Time markers in [brackets]; ~150 wpm. File/format description is deliberately short so the time goes to findings + implications.*

---

## [0:00–0:20] What the raw data is (kept brief)

"The raw data comes in two forms. First, a **static grid model** per grid —
buses, lines and transformers with their electrical parameters, one fixed
topology, no time. Second, a **demand time series** — for each bus, its active
power load at 15-minute resolution over a full year, 35,040 time steps. So: one
fixed graph, plus a per-bus load profile over time. A training sample only
exists once we pick a time step, apply that load, and solve the physics. I ran
two rounds of EDA — one on these raw inputs, one on the generated samples — and
each finding directly shaped a data-generation decision."

---

## [0:20–1:40] Raw-input EDA — findings and how they informed generation

"**Finding 1 — the four grids are genuinely heterogeneous, by orders of
magnitude.** UK's demand reaches about 13.6 gigawatts; IEEE24 tops out near 330
megawatts. Their structure differs too — UK is dense and generator-dominated,
IEEE118 is large with long electrical paths, the smaller IEEE grids are sparse.
*Implication for generation:* this is the fact that makes a leave-one-grid-out
study meaningful — an unseen grid is genuinely out-of-distribution. It also told
me up front that I must **normalize** targets and errors, and that I need a
principled **distance metric (MMD)**, because the grids are not on a common
scale.

**Finding 2 — the bus-type mix differs per grid.** The IEEE grids are
load-dominated — mostly PQ buses, where voltage and angle are unknown and must
be predicted — whereas UK is generator-dominated, where voltage is largely a
known input. *Implication for generation:* the prediction task itself changes
shape across grids, so the per-bus-type **masking** — hiding exactly the
unknowns for slack, PV and PQ buses — has to be built into every sample, not
bolted on later. It also warned me that UK would be the hard transfer target.

**Finding 3 — connectivity is fragile in the sparse grids.** Every base grid is
connected, but the IEEE grids are sparse and sit close to the connectivity edge
— and I found that two IEEE118 buses connect only through series-impedance
branches, which are easy to miss. *Implication for generation:* when I remove
lines to create contingencies, I *must* check the grid hasn't split into islands
and reject those cases — otherwise I'd be trying to solve an unsolvable network.
That islanding-rejection gate came directly from this observation.

**Finding 4 — the demand has real temporal structure.** Clear seasonal and
daily peak/off-peak cycles, not a flat signal. *Implication for generation:*
sampling demand snapshots genuinely varies the loading condition, so crossing
demand with topology gives a rich distribution rather than near-duplicate
samples — and it suggests stratifying snapshots across seasons so peak-stress
cases are represented."

---

## [1:40–2:40] Generated-data EDA — did generation do what we intended?

"After generating the samples, a second EDA checked the *product*.

**Finding 5 — the topology distribution is real and controlled.** Each grid has
a deliberate mix of intact, single-outage and double-outage cases, and the
number of in-service branches genuinely varies sample to sample. *Implication:*
the graph structure carries information — a contingency actually propagates
through the model — which is the whole premise of the study.

**Finding 6 — targets live on very different scales, and voltage is special.**
Active and reactive power and angle span wide, grid-dependent ranges, while
voltage is tightly bounded around 1 per-unit. *Implication:* a single aggregate
error is misleading, so I report **per-quantity, range-normalized** error — P,
Q, V and angle separately — otherwise the well-behaved voltage would flatter the
score.

**Finding 7 — the DC baseline is fair but beatable, and precisely where it
matters.** DC power flow reproduces active and reactive power almost exactly —
they're essentially known injections — but it's *blind to voltage*, because it
assumes a flat 1-per-unit profile. *Implication:* storing a DC baseline per
sample gives a non-trivial reference, and it pinpoints where a learned AC model
earns its keep — recovering voltage and angle, the quantities DC cannot
represent.

**Finding 8 — the masking is correct, with no leakage.** The unknown inputs are
missing exactly according to bus-type physics, and the prediction targets are
always fully observed. *Implication:* this is the check that lets us trust every
accuracy number — no to-be-predicted quantity leaked into the inputs.

**Finding 9 — the structural distance survives generation.** Computing MMD on
the *sampled* graphs, UK is still the clear outlier. *Implication:* the distance
our generalization metric relies on is a genuine property of the data, not an
artifact of the base topology."

---

## [2:40–3:00] Closing — EDA as the 'why' behind the pipeline

"The takeaway is that the EDA wasn't a formality — it *drove* the design. Grid
heterogeneity justified normalization and the MMD distance; the bus-type mix
drove the masking; fragile connectivity forced islanding rejection; the voltage
range and the DC gap dictated per-quantity metrics and the baseline; and the
leakage check is what makes the results trustworthy. In short, every
data-generation decision traces back to something we saw in the data first."

---

### One-line summary per finding (for a backup slide)
| # | Finding | Informed |
|---|---|---|
| 1 | Grids differ by orders of magnitude | Normalization + MMD distance |
| 2 | Bus-type mix differs per grid | Per-bus-type masking; UK is hard |
| 3 | Sparse grids fragment easily | Islanding rejection |
| 4 | Demand has seasonal/daily cycles | Snapshot sampling / stratification |
| 5 | Topology genuinely varies | Contingency-aware `edge_index` |
| 6 | Voltage tightly bounded, others wide | Per-quantity, range-normalized NRMSE |
| 7 | DC exact on P/Q, blind to voltage | DC baseline; value is in V/θ |
| 8 | Masking correct, no leakage | Trustworthy accuracy numbers |
| 9 | UK outlier survives generation | Distance-aware g-score is valid |
