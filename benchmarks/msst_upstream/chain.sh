#!/usr/bin/env bash
# Wait for the MPS campaign to finish, then provision and run the MSST
# upstream comparison. Sequential on purpose: both are timing measurements and
# must not contend for the GPU.
set -uo pipefail
export PATH=/usr/bin:/bin:/usr/local/bin:$HOME/.local/bin:$PATH
ROOT="/Users/ryan/Developer/unblend"; HERE="$ROOT/benchmarks/msst_upstream"
STATUS="$ROOT/benchmarks/campaign_status.tsv"
note() { printf '%s\t%s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$STATUS"; }

while ! grep -q "remaining end" "$STATUS" 2>/dev/null; do sleep 60; done
note "START msst_setup"
if bash "$HERE/setup.sh" > "$ROOT/benchmarks/logs/msst_setup.log" 2>&1; then
    note "OK    msst_setup"
else
    note "FAIL  msst_setup -- see logs/msst_setup.log"; exit 1
fi

note "START msst_check_configs"
"$ROOT/.venv/bin/python" "$HERE/check_configs.py" \
    > "$ROOT/benchmarks/logs/msst_check_configs.log" 2>&1
note "DONE  msst_check_configs (advisory; see log)"

note "START msst_upstream_run"
if "$ROOT/.venv/bin/python" "$HERE/run.py" --limit 15 --device mps \
        > "$ROOT/benchmarks/logs/msst_upstream_run.log" 2>&1; then
    note "OK    msst_upstream_run"
else
    note "FAIL  msst_upstream_run -- see logs/msst_upstream_run.log"
fi
note "msst chain end"
