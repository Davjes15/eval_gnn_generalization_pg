#!/usr/bin/env bash
# Rebuild the v3 merged results and downstream tables from campaign jobs/checkpoints.
set -u

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON=${PYTHON:-python}
RESULTS_ROOT="${RESULTS_ROOT:-results_v3}"
CKPT_ROOT="${CKPT_ROOT:-ckpt_v3}"
DATA_A="${DATA_A:-data_regime_a_fixed_topology_v3}"
DATA_B="${DATA_B:-data_regime_b_contingencies_v3}"
DOWNSTREAM_ROOT="${DOWNSTREAM_ROOT:-results_v3}"

if [[ -z "${POWERGRAPH_NODE_DIR:-}" ]]; then
    printf 'POWERGRAPH_NODE_DIR must point to PowerGraph-Node/13_Power_system.\n' >&2
    exit 2
fi
if [[ ! -d "$POWERGRAPH_NODE_DIR" ]]; then
    printf 'POWERGRAPH_NODE_DIR is not a directory: %s\n' "$POWERGRAPH_NODE_DIR" >&2
    exit 2
fi
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    printf 'Python executable not found: %s\n' "$PYTHON" >&2
    exit 2
fi

export OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export POWERGRAPH_NODE_DIR
mkdir -p "$DOWNSTREAM_ROOT/logs" "$DOWNSTREAM_ROOT/mmd_report" \
    "$DOWNSTREAM_ROOT/topology" "$DOWNSTREAM_ROOT/physics" \
    "$DOWNSTREAM_ROOT/analysis"
: > "$DOWNSTREAM_ROOT/step_timings.tsv"

FAILED=0
run_step() {
    local name="$1"
    shift
    local log_file="$DOWNSTREAM_ROOT/logs/$name.log"
    local start_ns
    local end_ns
    local elapsed_ms
    local rc

    {
        printf 'COMMAND: OMP_NUM_THREADS=%q PYTHONDONTWRITEBYTECODE=%q ' \
            "$OMP_NUM_THREADS" "$PYTHONDONTWRITEBYTECODE"
        printf 'POWERGRAPH_NODE_DIR=%q ' "$POWERGRAPH_NODE_DIR"
        printf '%q ' "$@"
        printf '\n'
    } > "$log_file"
    start_ns=$(date +%s%N)
    if "$@" >> "$log_file" 2>&1; then
        rc=0
    else
        rc=$?
        FAILED=1
    fi
    end_ns=$(date +%s%N)
    elapsed_ms=$(((end_ns - start_ns) / 1000000))
    printf '%s\t%s\t%s\n' "$name" "$elapsed_ms" "$rc" \
        >> "$DOWNSTREAM_ROOT/step_timings.tsv"
    printf '%s: exit=%s elapsed_ms=%s log=%s\n' \
        "$name" "$rc" "$elapsed_ms" "$log_file"
}

MODELS=(gcn arma_gnn gat gin transformer nnconv)

run_step gather_within "$PYTHON" gather_results.py \
    --shards "$RESULTS_ROOT/within_*" --file within_grid.csv \
    --out "$DOWNSTREAM_ROOT/all_within" --models "${MODELS[@]}" --seed_shards
run_step gather_cross "$PYTHON" gather_results.py \
    --shards "$RESULTS_ROOT/cross_*" --file cross_context.csv \
    --out "$DOWNSTREAM_ROOT/all_cross" --models "${MODELS[@]}" --seed_shards
run_step gather_ood "$PYTHON" gather_results.py \
    --shards "$RESULTS_ROOT/ood_*" --file ood.csv \
    --out "$DOWNSTREAM_ROOT/all_ood" --models "${MODELS[@]}" --seed_shards

run_step mmd_report "$PYTHON" mmd_report.py --data_dir "$DATA_B" \
    --out "$DOWNSTREAM_ROOT/mmd_report" --mmd_graph multi \
    --mmd_bandwidth reference --mmd_bandwidth_file configs/mmd_bandwidth.json
run_step topology_mmd_and_pooled_ood "$PYTHON" experiments.py \
    --only_topology --experiment ood --data_dir "$DATA_B" \
    --out "$DOWNSTREAM_ROOT/topology" --regime_tag B --models gcn \
    --arch_config configs/arch_config.json --normalize pu_zscore \
    --mmd_graph multi --mmd_bandwidth reference \
    --mmd_bandwidth_file configs/mmd_bandwidth.json

run_step checkpoint_index "$PYTHON" checkpoint_index.py \
    --ckpt_root "$CKPT_ROOT" --out "$DOWNSTREAM_ROOT/checkpoint_index.csv"
run_step physics_replay "$PYTHON" eval_checkpoints.py \
    --ckpt_root "$CKPT_ROOT" --data_a "$DATA_A" --data_b "$DATA_B" \
    --normalize pu_zscore --out "$DOWNSTREAM_ROOT/physics" --feasibility
run_step dc_feasibility "$PYTHON" dc_feasibility.py \
    --data "$DATA_A" "$DATA_B" \
    --out "$DOWNSTREAM_ROOT/physics/dc_feasibility.csv"
run_step summarize_feasibility "$PYTHON" summarize_feasibility.py \
    --physics "$DOWNSTREAM_ROOT/physics/physics_metrics.csv" \
    --baselines "$DOWNSTREAM_ROOT/physics/dc_feasibility.csv" \
    --out "$DOWNSTREAM_ROOT/analysis/ac_feasibility_norm.csv"

run_step recompute_dc_baseline_a "$PYTHON" recompute_dc_baseline.py \
    --data_dir "$DATA_A" --out "$DOWNSTREAM_ROOT/dc_baseline_regime_a.csv"
run_step rank_analysis "$PYTHON" rank_analysis.py \
    --regime_a "$DOWNSTREAM_ROOT/all_within/within_grid.csv" \
    --cross "$DOWNSTREAM_ROOT/all_cross/cross_context.csv" \
    --ood "$DOWNSTREAM_ROOT/all_ood/ood.csv" \
    --out "$DOWNSTREAM_ROOT/analysis"
run_step recompute_tables "$PYTHON" recompute_tables.py \
    --within "$DOWNSTREAM_ROOT/all_within/within_grid.csv" \
    --cross "$DOWNSTREAM_ROOT/all_cross/cross_context.csv" \
    --ood "$DOWNSTREAM_ROOT/all_ood/ood.csv" \
    --topology "$DOWNSTREAM_ROOT/topology" \
    --dc_regime_a "$DOWNSTREAM_ROOT/dc_baseline_regime_a.csv" \
    --dc_regime_b "$DOWNSTREAM_ROOT/topology/dc_baseline.csv" \
    --out "$DOWNSTREAM_ROOT/analysis"

if [ "$FAILED" -ne 0 ]; then
    printf 'One or more downstream steps failed; see logs and step_timings.tsv.\n' >&2
    exit 1
fi
