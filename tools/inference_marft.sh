#!/usr/bin/env bash
# ------------------------------------------------------------------------
# Copyright (c) 2026 JoshuaWenHIT. All Rights Reserved.
# ------------------------------------------------------------------------
# Query-MARFT inference launcher (mirrors tools/simple_inference.sh).
#
# Usage:
#   bash tools/inference_marft.sh <path/to/checkpoint.pth> [extra py args]
#   e.g.  bash tools/inference_marft.sh exps/Query-MARFT-marft/run1/checkpoint.pth
#
# The script reuses ALL base MOTRv2 CLI args from configs/Query-MARFT-marft.args
# (dataset paths, query_denoise, num_queries, etc.) plus the four-agent and
# LoRA settings needed to correctly reconstruct the MARFT wrapper before
# loading the checkpoint.
# ------------------------------------------------------------------------


set -x
set -o pipefail

args=$(cat configs/Query-MARFT-v2.args)
python3 scripts/inference_marft.py ${args} --exp_name tracker --resume $1 ${@:2}
