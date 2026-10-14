#!/usr/bin/env bash
set -eo pipefail
package="$(cd "$(dirname "$0")/.." && pwd)"
workspace="$(cd "${package}/../.." && pwd)"
source /opt/ros/foxy/setup.bash
if [[ ! -f "$workspace/install/shade_challenger/share/shade_challenger/weights/depth.onnx" ]]; then
  (cd "$workspace" && colcon build --packages-select shade_challenger --symlink-install)
fi
source "$workspace/install/setup.bash"
exec ros2 launch shade_challenger demo.launch.py "$@"
