#!/usr/bin/env bash
# ------------------------------------------------------------------------
# Copyright (c) 2026 JoshuaWenHIT. All Rights Reserved.
# ------------------------------------------------------------------------
# MOTRv2 inference on MOT20 (mirrors tools/simple_inference.sh for DanceTrack).
#
# Usage:
#   bash tools/simple_inference_mot20.sh <checkpoint.pth> [--split train|test] [extra py args]
#
# Defaults to --split test. Pass --split train to run on the 4 MOT20 train
# sequences (useful for offline validation with HOTA/IDF1 against public gt).
# ------------------------------------------------------------------------

set -x
set -o pipefail

if [ $# -lt 1 ]; then
    echo "usage: bash tools/simple_inference_mot20.sh <checkpoint.pth> [extra py args]"
    exit 2
fi

args=$(cat configs/motrv2_mot20.args)
python3 submit_mot20.py ${args} \
    --exp_name tracker \
    --resume "$1" \
    "${@:2}"
