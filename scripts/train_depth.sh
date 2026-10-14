#!/usr/bin/env bash
# Offline camera training; no Gazebo/PX4 processes are launched.
set -euo pipefail
package="$(cd "$(dirname "$0")/.." && pwd)"
workspace="$(cd "${package}/../.." && pwd)"
python="${GAP_PYTHON:-/home/li/miniconda3/envs/torch/bin/python}"
output="${GAP_RUN_DIR:-${package}/results/depth_train_$(date +%Y%m%d_%H%M%S)}"
if [[ -e "$output" ]]; then echo "Output exists; choose another GAP_RUN_DIR: $output" >&2; exit 1; fi
mkdir -p "$output"
export PYTHONPATH="${package}${PYTHONPATH:+:${PYTHONPATH}}"
export OPENBLAS_NUM_THREADS=1
initial=()
if [[ -n "${GAP_INITIAL:-}" ]]; then initial=(--initial "$GAP_INITIAL"); fi
"$python" "${package}/scripts/depth_dataset.py" --output "$output/views.pt" --views "${GAP_VIEWS:-4096}" 2>&1 | tee "$output/data.log"
"$python" "${package}/scripts/augment_states.py" --input "$output/views.pt" --output "$output/samples.pt" --states "${GAP_STATES:-100000}" --vertical-goal-std .8 --reverse-goal-fraction .25 2>&1 | tee "$output/states.log"
"$python" -m gap_nav.learning.train --sensor depth --data "$output/samples.pt" --output "$output/base" --epochs "${GAP_BASE_EPOCHS:-30}" --batch-size 64 --max-speed 2 --max-accel 1.5 "${initial[@]}" 2>&1 | tee "$output/base.log"
"$python" -m gap_nav.shadow.coordinator --sensor depth --data "$output/samples.pt" --actor "$output/base/best.pt" --output "$output/shadows" --fraction "${GAP_SHADOW_FRACTION:-.01}" --iterations "${GAP_MPC_ITERATIONS:-4}" --max-batch 256 --max-speed 2 --max-accel 1.5 2>&1 | tee "$output/shadows.log"
"$python" -m gap_nav.learning.train --sensor depth --data "$output/samples.pt" --initial "$output/base/best.pt" --shadows "$output/shadows/shadows.pt" --output "$output/dual" --epochs "${GAP_DUAL_EPOCHS:-5}" --batch-size 64 --max-speed 2 --max-accel 1.5 2>&1 | tee "$output/dual.log"
"$python" "${package}/scripts/export.py" "$output/dual/best.pt" 2>&1 | tee "$output/export.log"
ln -sfn "$output/dual" "$package/results/depth_current"
echo "Trained camera model: $output/dual/best.onnx"
echo 'Run scripts/run_depth.sh --test to validate this new model in SITL.'
