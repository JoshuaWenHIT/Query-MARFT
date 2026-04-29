#!/usr/bin/env bash
# ------------------------------------------------------------------------
# Copyright (c) 2026 JoshuaWenHIT. All Rights Reserved.
# ------------------------------------------------------------------------
# Convenience wrapper around tools/postprocess_tracks.py.
#
# Usage:
#   bash tools/postprocess_tracks.sh <input_tracker_dir> [output_dir] [extra_args]
#
# Examples:
#   # Default pipeline on MOTRv2 output:
#   bash tools/postprocess_tracks.sh exps/run1/tracker
#
#   # Ablation — only interpolation + short filter:
#   bash tools/postprocess_tracks.sh exps/run1/tracker exps/run1/post_interp \
#       --stages interp,short
#
#   # Tighter merge for DanceTrack val (more aggressive stitching):
#   bash tools/postprocess_tracks.sh exps/run1/tracker exps/run1/post_v2 \
#       --merge_t_max 80 --merge_iou 0.3 --interp_max_gap 80
# ------------------------------------------------------------------------

set -e
set -o pipefail

if [ $# -lt 1 ]; then
    echo "usage: bash tools/postprocess_tracks.sh <input_tracker_dir> [output_dir] [extra_args...]"
    exit 2
fi

INPUT_DIR="$1"
shift

# Default output_dir = sibling "post_<input_basename>"
if [ $# -ge 1 ] && [[ "$1" != --* ]]; then
    OUTPUT_DIR="$1"
    shift
else
    INPUT_BASENAME=$(basename "$INPUT_DIR")
    INPUT_PARENT=$(dirname "$INPUT_DIR")
    OUTPUT_DIR="${INPUT_PARENT}/post_${INPUT_BASENAME}"
fi

echo "[postprocess] input_dir  = ${INPUT_DIR}"
echo "[postprocess] output_dir = ${OUTPUT_DIR}"

python3 tools/postprocess_tracks.py "${INPUT_DIR}" "${OUTPUT_DIR}" "$@"

echo "[postprocess] results written to: ${OUTPUT_DIR}/tracker"
