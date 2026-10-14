#!/usr/bin/env bash
set -euo pipefail
package="$(cd "$(dirname "$0")/.." && pwd)"
workspace="$(cd "${package}/../.." && pwd)"
python="${GAP_PYTHON:-/home/li/miniconda3/envs/torch/bin/python}"
sensor="${1:-depth}"
if [[ "$sensor" != depth && "$sensor" != mid360 ]]; then echo 'Usage: train.sh depth|mid360'; exit 2; fi
if [[ "$sensor" == depth ]]; then exec "${package}/scripts/train_depth.sh"; fi
export PYTHONPATH="${package}${PYTHONPATH:+:${PYTHONPATH}}"
export OPENBLAS_NUM_THREADS=1
if [[ ! -f "${package}/results/baseline/metrics.json" ]]; then "$python" "${package}/scripts/baseline.py"; fi
if [[ ! -f "${package}/results/data/samples.pt" ]]; then "$python" -m gap_nav.data --workspace "$workspace" --output "${package}/results/data"; fi
"$python" -m gap_nav.learning.train --sensor "$sensor" --data "${package}/results/data/samples.pt" --output "${package}/results/${sensor}_base" --epochs "${GAP_EPOCHS:-30}"
"$python" -m gap_nav.shadow.coordinator --sensor "$sensor" --data "${package}/results/data/samples.pt" --actor "${package}/results/${sensor}_base/best.pt" --output "${package}/results/${sensor}_shadows" --fraction "${GAP_SHADOW_FRACTION:-0.03}"
"$python" -m gap_nav.learning.train --sensor "$sensor" --data "${package}/results/data/samples.pt" --output "${package}/results/${sensor}_dual" --initial "${package}/results/${sensor}_base/best.pt" --shadows "${package}/results/${sensor}_shadows/shadows.pt" --epochs "${GAP_EPOCHS:-30}"
"$python" "${package}/scripts/export.py" "${package}/results/${sensor}_dual/best.pt"
