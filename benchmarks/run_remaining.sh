#!/usr/bin/env bash
# Remaining MPS work after the first campaign. Every tier now flushes its CSVs
# after each config, so an interrupt keeps whatever finished.
set -uo pipefail
ROOT="/Users/ryan/Developer/unblend"; PY="$ROOT/.venv/bin/python"
MUSDB="/Users/ryan/Music/musdb18hq/test"; OUT="$ROOT/benchmarks"
STATUS="$OUT/campaign_status.tsv"; mkdir -p "$OUT/logs"; cd "$ROOT" || exit 1
SWEEP=(htdemucs htdemucs_ft htdemucs_6s bs_roformer_sw bs_roformer_anvuew
       melband_roformer_kim scnet_small roformer_vocals_ensemble)
mf() { for m in "$@"; do printf -- "--model %s " "$m"; done; }
note() { printf '%s\t%s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$STATUS"; }
run_tier() { local n="$1"; shift; note "START $n"; local s=$(date +%s)
  if "$@" >"$OUT/logs/$n.log" 2>&1; then note "OK    $n ($(( ($(date +%s)-s)/60 ))m)"
  else note "FAIL  $n (exit $?)"; fi; }

note "remaining begin"

# Precision sweep, the gap left by the killed run. Timing only; SDR is in tier 1.
run_tier tier2_precision "$PY" benchmark.py --musdb-root "$MUSDB" --device mps \
    --output-dir "$OUT/tier2_precision" --limit 5 $(mf "${SWEEP[@]}") \
    --precision fp16 --precision fp32 --precision bf16 \
    --compile-mode false --shifts 1 --split-overlap 0.25 --no-sdr --seed 1234

# The two models excluded from the sweep for cost; run separately so a slow
# tier cannot delay the cheap one.
run_tier tier2b_precision_heavy "$PY" benchmark.py --musdb-root "$MUSDB" --device mps \
    --output-dir "$OUT/tier2b_precision_heavy" --limit 5 \
    --model scnet_xl_wide_v5 --model htdemucs_scnet_ensemble \
    --precision fp16 --precision fp32 --precision bf16 \
    --compile-mode false --shifts 1 --split-overlap 0.25 --no-sdr --seed 1234

# Completes the cbs sweep; cbs=16 died pre-fix.
run_tier tier3_cbs16 "$PY" benchmark.py --musdb-root "$MUSDB" --device mps \
    --output-dir "$OUT/tier3_cbs/cbs16" --limit 5 \
    --model htdemucs --model melband_roformer_kim --model scnet_small \
    --precision fp16 --compile-mode false --shifts 1 --split-overlap 0.25 \
    --chunk-batch-size 16 --no-sdr --seed 1234

# Kernel A/B rerun: scnet_xl and htdemucs-bf16 had >10% spread last time and
# were not quotable. More rounds, and scnet_xl gets a shorter clip so the
# round count is affordable.
run_tier tier5b_kernels_noisy "$PY" "$OUT/ab_kernels.py" \
    --out "$OUT/tier5b_kernels_noisy.json" \
    --precision fp16 --precision bf16 --seconds 10 --rounds 12 \
    --model htdemucs --model scnet_xl_wide_v5

note "remaining end"
