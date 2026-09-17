#!/usr/bin/env bash
# scnet_xl_wide_v5 is the one upstream model that would run ~2.6 h at fp32 with
# a >20 GB peak. Swap was already 94% full and the OS had started killing
# processes, so it gets a reduced track count instead of the full 15. Realtime
# factor normalises over track count, so the speed comparison stays valid; only
# its SDR is over fewer tracks, which is recorded alongside the number.
set -uo pipefail
export PATH=/usr/bin:/bin:$PATH
ROOT="/Users/ryan/Developer/unblend"; HERE="$ROOT/benchmarks/msst_upstream"
STATUS="$ROOT/benchmarks/campaign_status.tsv"
note() { printf '%s\t%s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$STATUS"; }

# Wait for scnet_small to land, then stop before the heavy model starts.
while ! grep -q '"model": "scnet_small"' "$HERE/results/summary.json" 2>/dev/null; do sleep 30; done
note "scnet_small banked; capping scnet_xl_wide_v5 to 5 tracks"
pkill -f "_worker_materialised" 2>/dev/null
pkill -f "msst_upstream/run.py" 2>/dev/null
sleep 5

note "START msst_scnet_xl_limit5"
if "$ROOT/.venv/bin/python" "$HERE/run.py" --model scnet_xl_wide_v5 --limit 5 \
        --device mps --out "$HERE/results_scnet_xl" \
        > "$ROOT/benchmarks/logs/msst_scnet_xl.log" 2>&1; then
    note "OK    msst_scnet_xl_limit5"
else
    note "FAIL  msst_scnet_xl_limit5"
fi
note "msst chain end"
