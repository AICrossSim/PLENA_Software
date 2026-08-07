#!/usr/bin/env bash
# Publication pipeline launcher.
#
# The lazy compiler-trace path imports `asm_templates` as a top-level package
# (compiler frontend), which requires PLENA_Simulator/compiler and
# PLENA_Simulator/PLENA_Tools on PYTHONPATH; the in-process sys.path injection
# covers only the simulator root and analytic_models/performance. Launching
# through this script makes runs independent of ambient shell state.
set -euo pipefail

SOFTWARE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SIMULATOR_ROOT="${PLENA_SIMULATOR_PATH:-$(dirname "$SOFTWARE_ROOT")/PLENA_Simulator}"

if [[ ! -d "$SIMULATOR_ROOT/analytic_models/performance" ]]; then
    echo "PLENA_Simulator not found at $SIMULATOR_ROOT; set PLENA_SIMULATOR_PATH" >&2
    exit 2
fi

export PLENA_SIMULATOR_PATH="$SIMULATOR_ROOT"
export PYTHONPATH="$SIMULATOR_ROOT/compiler:$SIMULATOR_ROOT/PLENA_Tools${PYTHONPATH:+:$PYTHONPATH}"

cd "$SOFTWARE_ROOT"
exec "$SOFTWARE_ROOT/.venv/bin/python" -m decode_dse.software.sweep "$@"
