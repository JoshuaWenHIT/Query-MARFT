#!/usr/bin/env bash
# ------------------------------------------------------------------------
# Copyright (c) 2026 JoshuaWenHIT. All Rights Reserved.
# ------------------------------------------------------------------------
# Native-MOTRv2 inference with the MOTRWithStability wrapper (score EMA +
# ref_pts motion extrapolation + same-frame dedup).  No MARFT, no LoRA,
# no retraining — purely an inference-time overlay for DanceTrack IDF1.
#
# Usage:
#   bash tools/inference_stability.sh <motrv2_ckpt.pth> [extra py args]
#
# Current defaults (SAFE preset after 2026-04-20 HOTA-regression fix):
#   * stab_score_ema_alpha = 0.3 , ema_mode = asymmetric_down
#       -> only dampens score DIPS, does NOT hold stale IDs alive
#   * stab_motion_k         = 1   (motion extrapolation OFF)
#   * stab_dedup_iou_thresh = 1.1 (same-frame dedup OFF)
#
# Examples:
#   # Default (EMA-only, asymmetric).  Expected: small IDF1 gain, HOTA >= baseline.
#   bash tools/inference_stability.sh motrv2_dancetrack.pth
#
#   # Ablation A: turn the wrapper into a complete no-op (MUST reproduce the
#   # native submit_dance number; if not, there is an integration bug):
#   bash tools/inference_stability.sh motrv2_dancetrack.pth \
#       --stab_ema_mode off
#
#   # Ablation B: symmetric EMA (the OLD default that caused the regression):
#   bash tools/inference_stability.sh motrv2_dancetrack.pth \
#       --stab_ema_mode symmetric --stab_score_ema_alpha 0.3
#
#   # Ablation C: enable motion extrapolation alone:
#   bash tools/inference_stability.sh motrv2_dancetrack.pth \
#       --stab_ema_mode off --stab_motion_k 5 --stab_motion_max_step 0.03
#
#   # Ablation D: enable dedup alone (use a very conservative IoU=0.95):
#   bash tools/inference_stability.sh motrv2_dancetrack.pth \
#       --stab_ema_mode off --stab_dedup_iou_thresh 0.95
# ------------------------------------------------------------------------


set -x
set -o pipefail

if [ $# -lt 1 ]; then
    echo "usage: bash tools/inference_stability.sh <motrv2_ckpt.pth> [extra py args]"
    exit 2
fi

args=$(cat configs/motrv2.args)
python3 scripts/inference_stability.py ${args} \
    --exp_name tracker \
    --resume "$1" \
    "${@:2}"
