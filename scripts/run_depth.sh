#!/usr/bin/env bash
# One entry point for the camera policy, existing full Astra model and PX4 SITL.
set -eo pipefail
package="$(cd "$(dirname "$0")/.." && pwd)"
workspace="$(cd "${package}/../.." && pwd)"
auto=false
gui=false
record=false
weights="${GAP_WEIGHTS:-${package}/results/depth_current/best.onnx}"
seconds=120
while [[ $# -gt 0 ]]; do
  case "$1" in
    --auto) auto=true; shift;;
    --gui) gui=true; shift;;
    --test) record=true; auto=true; shift;;
    --weights) weights="$2"; shift 2;;
    --seconds) seconds="$2"; shift 2;;
    -h|--help) echo 'Usage: run_depth.sh [--auto] [--gui] [--test] [--weights PATH] [--seconds 120]'; exit 0;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done
if [[ ! -f "$weights" ]]; then echo "Camera ONNX not found: $weights. Run scripts/train_depth.sh first." >&2; exit 1; fi
source /opt/ros/foxy/setup.bash
if [[ ! -f "${workspace}/install/shade_challenger/share/shade_challenger/package.xml" ]]; then
  (cd "$workspace" && colcon build --packages-select shade_challenger --symlink-install)
fi
source "${workspace}/install/setup.bash"
if [[ "$record" == true ]]; then
  options=()
  if [[ "$gui" == true ]]; then options+=(--gui); fi
  exec /usr/bin/python3 "${package}/scripts/sitl_test.py" --sensor depth --seconds "$seconds" --weights "$weights" --name "camera_$(date +%Y%m%d_%H%M%S)" "${options[@]}"
fi
exec "${package}/scripts/deploy.sh" --sensor depth weights:="$weights" auto_arm:="$auto" gui:="$gui"
