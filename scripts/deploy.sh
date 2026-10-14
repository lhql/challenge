#!/usr/bin/env bash
set -eo pipefail
package="$(cd "$(dirname "$0")/.." && pwd)"
workspace="$(cd "${package}/../.." && pwd)"
if [[ "${1:-}" != --sensor || ( "${2:-}" != depth && "${2:-}" != mid360 ) ]]; then echo 'Usage: deploy.sh --sensor depth|mid360 [ROS launch arguments]'; exit 2; fi
sensor="$2";shift 2
source /opt/ros/foxy/setup.bash
source "${workspace}/install/setup.bash"
weights="${package}/results/${sensor}_dual/best.onnx"
if [[ "$sensor" == depth ]]; then weights="${GAP_WEIGHTS:-${package}/results/depth_current/best.onnx}"; fi
exec ros2 launch shade_challenger sim.launch.py sensor:="$sensor" weights:="$weights" "$@"
