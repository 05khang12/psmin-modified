#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/run_demokan.sh MODEL [LABEL]

MODEL:
  hwak
  hm_idelta
  hmr

Examples:
  scripts/run_demokan.sh hwak baseline
  PSMIN_C=10 PSMIN_D=0.00015 PSMIN_NU=0.00015 \
    scripts/run_demokan.sh hwak C10
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
    exit 2
fi

model="$1"
label="${2:-run}"

case "$model" in
    hwak)
        program="codes/simulation_codes/hwak_min.py"
        ;;
    hm_idelta)
        program="codes/simulation_codes/hm_idelta_min.py"
        ;;
    hmr)
        program="codes/simulation_codes/hmr_min.py"
        ;;
    *)
        printf 'Unknown model: %s\n\n' "$model" >&2
        usage >&2
        exit 2
        ;;
esac

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

if [[ ! -f "$program" ]]; then
    printf 'Simulation program not found: %s\n' "$root/$program" >&2
    exit 1
fi

timestamp="$(date +%Y-%m-%d_%H-%M-%S)"
safe_label="$(printf '%s' "$label" | tr -c 'A-Za-z0-9._-' '_')"
run_name="${timestamp}_${model}_${safe_label}"
run_dir="$root/outputs/$run_name"
mkdir -p "$run_dir"

export MLSARRAY_BACKEND="${MLSARRAY_BACKEND:-numpy}"
export PSMIN_FLNAME="$run_dir/out_${model}.h5"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"

{
    printf 'run_name=%s\n' "$run_name"
    printf 'model=%s\n' "$model"
    printf 'program=%s\n' "$program"
    printf 'host=%s\n' "$(hostname)"
    printf 'start_time=%s\n' "$(date --iso-8601=seconds)"
    printf 'working_directory=%s\n' "$root"
    printf 'python=%s\n' "$(command -v python3)"
    python3 --version
    printf 'MLSARRAY_BACKEND=%s\n' "$MLSARRAY_BACKEND"
    printf 'PSMIN_FLNAME=%s\n' "$PSMIN_FLNAME"
    env | LC_ALL=C sort | grep '^PSMIN_' || true
    git rev-parse HEAD 2>/dev/null | sed 's/^/git_commit=/' || true
    git status --short 2>/dev/null | sed 's/^/git_status=/' || true
} > "$run_dir/metadata.txt"

printf 'Run directory: %s\n' "$run_dir"
printf 'Starting %s at %s\n' "$model" "$(date --iso-8601=seconds)"

set +e
python3 -u "$program" 2>&1 | tee "$run_dir/run.log"
status=${PIPESTATUS[0]}
set -e

{
    printf 'end_time=%s\n' "$(date --iso-8601=seconds)"
    printf 'exit_status=%s\n' "$status"
} >> "$run_dir/metadata.txt"

if [[ $status -eq 0 ]]; then
    touch "$run_dir/SUCCESS"
    printf 'Run completed successfully: %s\n' "$run_dir"
else
    touch "$run_dir/FAILED"
    printf 'Run failed with status %s: %s\n' "$status" "$run_dir" >&2
fi

exit "$status"
