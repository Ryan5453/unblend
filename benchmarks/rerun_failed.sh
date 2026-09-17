#!/usr/bin/env bash
# Re-run the tiers that died on the pre-fix RoFormer MPS attention path.
set -uo pipefail
ROOT="/Users/ryan/Developer/unblend"; PY="$ROOT/.venv/bin/python"
MUSDB="/Users/ryan/Music/musdb18hq/test"; OUT="$ROOT/benchmarks"
STATUS="$OUT/campaign_status.tsv"; mkdir -p "$OUT/logs"; cd "$ROOT" || exit 1
ALL=(htdemucs htdemucs_ft htdemucs_6s bs_roformer_sw bs_roformer_anvuew
     melband_roformer_kim scnet_small scnet_xl_wide_v5
     roformer_vocals_ensemble htdemucs_scnet_ensemble)
SWEEP=(htdemucs htdemucs_ft htdemucs_6s bs_roformer_sw bs_roformer_anvuew
       melband_roformer_kim scnet_small roformer_vocals_ensemble)
mf() { for m in "$@"; do printf -- "--model %s " "$m"; done; }
note() { printf '%s\t%s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$STATUS"; }
run_tier() { local n="$1"; shift; note "START $n"; local s=$(date +%s)
  if "$@" >"$OUT/logs/$n.log" 2>&1; then note "OK    $n ($(( ($(date +%s)-s)/60 ))m)"
  else note "FAIL  $n (exit $?)"; fi; }

note "rerun begin (post attention fix)"
run_tier tier1_headline "$PY" benchmark.py --musdb-root "$MUSDB" --device mps \
    --output-dir "$OUT/tier1_headline" --limit 15 $(mf "${ALL[@]}") \
    --precision fp16 --compile-mode false --shifts 1 --split-overlap 0.25 --sdr --seed 1234
run_tier tier2_precision "$PY" benchmark.py --musdb-root "$MUSDB" --device mps \
    --output-dir "$OUT/tier2_precision" --limit 5 $(mf "${SWEEP[@]}") \
    --precision fp16 --precision fp32 --precision bf16 \
    --compile-mode false --shifts 1 --split-overlap 0.25 --no-sdr --seed 1234
run_tier tier3_cbs16 "$PY" benchmark.py --musdb-root "$MUSDB" --device mps \
    --output-dir "$OUT/tier3_cbs/cbs16" --limit 5 \
    --model htdemucs --model melband_roformer_kim --model scnet_small \
    --precision fp16 --compile-mode false --shifts 1 --split-overlap 0.25 \
    --chunk-batch-size 16 --no-sdr --seed 1234
note "rerun end"
