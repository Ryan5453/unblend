#!/usr/bin/env bash
# MPS benchmark campaign driver.
#
# Runs the tiers from BENCHMARK.md §4 back to back in one process so the
# machine stays busy unattended. Each tier writes its own output directory and
# logs to benchmarks/logs/<tier>.log. A tier that fails is recorded and the
# campaign continues -- one bad config must not cost the whole day.
#
# Deliberately sequential: these are timing measurements, so nothing else may
# contend for the GPU while a tier runs.

set -uo pipefail

ROOT="/Users/ryan/Developer/unblend"
PY="$ROOT/.venv/bin/python"
MUSDB="/Users/ryan/Music/musdb18hq/test"
OUT="$ROOT/benchmarks"
LOGS="$OUT/logs"
STATUS="$OUT/campaign_status.tsv"

mkdir -p "$LOGS"
cd "$ROOT" || exit 1

ALL_MODELS=(htdemucs htdemucs_ft htdemucs_6s bs_roformer_sw bs_roformer_anvuew
            melband_roformer_kim scnet_small scnet_xl_wide_v5
            roformer_vocals_ensemble htdemucs_scnet_ensemble)

# scnet_xl_wide_v5 and the ensemble containing it run at <1x realtime on MPS
# and together are ~66% of the full-matrix cost. They stay in tier 1 (where the
# headline number matters) and are dropped from the sweep tiers.
SWEEP_MODELS=(htdemucs htdemucs_ft htdemucs_6s bs_roformer_sw bs_roformer_anvuew
              melband_roformer_kim scnet_small roformer_vocals_ensemble)

model_flags() {
    for m in "$@"; do printf -- "--model %s " "$m"; done
}

note() {
    printf '%s\t%s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$STATUS"
}

run_tier() {
    # run_tier <name> <command...>
    local name="$1"; shift
    local log="$LOGS/$name.log"
    note "START $name"
    local started
    started=$(date +%s)
    if "$@" >"$log" 2>&1; then
        note "OK    $name ($(( ($(date +%s) - started) / 60 ))m)"
    else
        note "FAIL  $name (exit $?) -- see $log"
    fi
}

note "campaign begin"

# --- Tier 1: headline. All models, fp16, full test split, with SDR. ----------
run_tier tier1_headline \
    "$PY" benchmark.py --musdb-root "$MUSDB" --device mps \
        --output-dir "$OUT/tier1_headline" --limit 15 \
        $(model_flags "${ALL_MODELS[@]}") \
        --precision fp16 --compile-mode false --shifts 1 --split-overlap 0.25 \
        --sdr --seed 1234

# --- Tier 2: precision sweep. Throughput only; SDR comes from tier 1. -------
run_tier tier2_precision \
    "$PY" benchmark.py --musdb-root "$MUSDB" --device mps \
        --output-dir "$OUT/tier2_precision" --limit 5 \
        $(model_flags "${SWEEP_MODELS[@]}") \
        --precision fp16 --precision fp32 --precision bf16 \
        --compile-mode false --shifts 1 --split-overlap 0.25 \
        --no-sdr --seed 1234

# --- Tier 3: chunk_batch_size. Output is bit-identical across values, so this
#     is a pure throughput/memory knob and needs no SDR. -----------------------
for cbs in 1 2 4 8 16; do
    run_tier "tier3_cbs${cbs}" \
        "$PY" benchmark.py --musdb-root "$MUSDB" --device mps \
            --output-dir "$OUT/tier3_cbs/cbs${cbs}" --limit 5 \
            --model htdemucs --model melband_roformer_kim --model scnet_small \
            --precision fp16 --compile-mode false --shifts 1 --split-overlap 0.25 \
            --chunk-batch-size "$cbs" --no-sdr --seed 1234
done

# --- Tier 4: shifts and overlap. Both trade wall time for quality, so both
#     need SDR to be interpretable. ---------------------------------------------
run_tier tier4_shifts_overlap \
    "$PY" benchmark.py --musdb-root "$MUSDB" --device mps \
        --output-dir "$OUT/tier4_shifts_overlap" --limit 5 \
        --model htdemucs --model melband_roformer_kim \
        --precision fp16 --compile-mode false \
        --shifts 1 --shifts 2 --shifts 4 \
        --split-overlap 0.1 --split-overlap 0.25 --split-overlap 0.5 \
        --sdr --seed 1234

# --- Tier 5: fused Metal kernels, interleaved A/B. --------------------------
run_tier tier5_kernels \
    "$PY" "$OUT/ab_kernels.py" --out "$OUT/tier5_kernels.json" \
        --precision fp16 --precision bf16 --seconds 20 --rounds 6 \
        --model htdemucs --model htdemucs_6s --model melband_roformer_kim \
        --model bs_roformer_sw --model scnet_small --model scnet_xl_wide_v5

# --- Tier 6: single-stem extraction. htdemucs_ft's specialist shortcut. ------
run_tier tier6_fullbag \
    "$PY" benchmark.py --musdb-root "$MUSDB" --device mps \
        --output-dir "$OUT/tier6_singlestem/full" --limit 15 \
        --model htdemucs_ft --precision fp16 --compile-mode false \
        --shifts 1 --split-overlap 0.25 --sdr --seed 1234

for stem in vocals bass other drums; do
    run_tier "tier6_only_${stem}" \
        "$PY" benchmark.py --musdb-root "$MUSDB" --device mps \
            --output-dir "$OUT/tier6_singlestem/only_${stem}" --limit 15 \
            --model htdemucs_ft --precision fp16 --compile-mode false \
            --shifts 1 --split-overlap 0.25 --use-only-stem "$stem" \
            --sdr --seed 1234
done

# --- Tier 7: batched vs per-track dispatch. ---------------------------------
run_tier tier7_dataset_throughput \
    "$PY" benchmark.py --musdb-root "$MUSDB" --device mps \
        --output-dir "$OUT/tier7_throughput" --limit 16 \
        --model htdemucs --model melband_roformer_kim \
        --precision fp16 --compile-mode false --shifts 1 --split-overlap 0.25 \
        --dataset-throughput --no-sdr --seed 1234

# --- Tier 8: upstream adefossez/demucs, HTDemucs family only. ---------------
# Provisions an isolated venv on first run (torch 2.1.2, python 3.11).
run_tier tier8_upstream_demucs \
    "$PY" benchmark.py --musdb-root "$MUSDB" --device mps \
        --output-dir "$OUT/tier8_upstream" --limit 15 \
        --model htdemucs --model htdemucs_ft --model htdemucs_6s \
        --precision fp32 --compile-mode false --shifts 1 --split-overlap 0.25 \
        --include-upstream --sdr --seed 1234

note "campaign end"
