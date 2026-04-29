# ------------------------------------------------------------------------
# Copyright (c) 2026 JoshuaWenHIT. All Rights Reserved.
# ------------------------------------------------------------------------
"""
Native-MOTRv2 inference entry point that wraps the model with
``MOTRWithStability`` (see ``models/motr_stability.py``) to improve IDF1
on DanceTrack.  No MARFT, no LoRA, no training — pure inference with
three stability mechanisms (score EMA / ref_pts motion extrapolation /
same-frame cross-ID dedup).

Mirrors ``submit_dance.py`` exactly for the data-loading / iteration
logic (reuses its ``Detector`` and ``RuntimeTrackerBase``) and uses the
original MOTRv2 CLI args parser from ``main.py``, plus a few
``--stab_*`` flags for the stability knobs.

Usage
-----
    bash tools/inference_stability.sh <path/to/motrv2_dancetrack.pth>

or directly::

    python3 scripts/inference_stability.py \\
        $(cat configs/motrv2.args) \\
        --exp_name tracker --resume <ckpt> \\
        --score_threshold 0.5 --update_score_threshold 0.5 \\
        --miss_tolerance 20 \\
        --stab_score_ema_alpha 0.3 \\
        --stab_motion_k 5 \\
        --stab_motion_score_thresh 0.3 \\
        --stab_motion_max_step 0.03 \\
        --stab_dedup_iou_thresh 0.9
"""

import argparse
import os
import sys
from pathlib import Path

import torch

# Make the project root importable before we pull in project modules.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import get_args_parser as get_base_args_parser
from models import build_model
from models.motr_stability import MOTRWithStability, build_stability_model
from submit_dance import Detector, RuntimeTrackerBase
from util.tool import load_model


def get_args_parser() -> argparse.ArgumentParser:
    parser = get_base_args_parser()
    g = parser.add_argument_group('Stability (inference-only)')

    g.add_argument('--stab_score_ema_alpha', type=float, default=0.3,
                   help='EMA blend for track scores: score_t = alpha*prev + '
                        '(1-alpha)*cur.  0 disables the score smoother.')
    g.add_argument('--stab_ema_mode',
                   choices=['off', 'symmetric', 'asymmetric_down'],
                   default='asymmetric_down',
                   help='EMA direction.  "asymmetric_down" (default) only '
                        'smooths when cur < prev — dampens single-frame dips '
                        'but does NOT hold stale IDs alive.  "symmetric" is '
                        'the original bidirectional behaviour (can hurt HOTA '
                        'on DanceTrack).  "off" disables.')
    g.add_argument('--stab_motion_k', type=int, default=1,
                   help='Tail window (frames) used to estimate per-track '
                        'velocity.  <2 disables motion extrapolation '
                        '(DEFAULT OFF — known to compete with YOLOX '
                        'proposals and cause ID churn on DanceTrack).')
    g.add_argument('--stab_motion_score_thresh', type=float, default=0.3,
                   help='Only extrapolate ref_pts for tracks whose current '
                        'score is below this value.')
    g.add_argument('--stab_motion_max_step', type=float, default=0.03,
                   help='Max per-frame xy center offset (as fraction of the '
                        'image) applied during extrapolation.')
    g.add_argument('--stab_dedup_iou_thresh', type=float, default=1.1,
                   help='Cross-ID same-frame dedup IoU threshold.  >=1 '
                        'disables dedup (DEFAULT OFF — genuine occlusion '
                        'in DanceTrack can produce IoU>0.9 between truly '
                        'different dancers; hard-killing one is usually '
                        'catastrophic for IDF1/HOTA).')

    # Inference-specific knobs already used by submit_dance.py.
    g.add_argument('--score_threshold', default=0.5, type=float)
    g.add_argument('--update_score_threshold', default=0.5, type=float)
    g.add_argument('--miss_tolerance', default=20, type=int)
    return parser


def main(args) -> None:
    # ---------- Build native MOTR (unmodified) ----------
    base_model, _, _ = build_model(args)

    # Runtime tracker thresholds must be set BEFORE wrapping, because the
    # stability wrapper installs its EMA hook on the track_base instance
    # at construction time.
    base_model.track_embed.score_thr = args.update_score_threshold
    base_model.track_base = RuntimeTrackerBase(
        args.score_threshold, args.score_threshold, args.miss_tolerance)

    # Load the MOTRv2 checkpoint (same helper used by submit_dance.py).
    assert args.resume, 'Please pass --resume <motrv2_checkpoint.pth>'
    base_model = load_model(base_model, args.resume)

    # ---------- Wrap with stability overlay ----------
    model = build_stability_model(base_model, args)
    model.eval()
    model = model.cuda()

    print('[inference-stability] config =', {
        'score_ema_alpha': model.score_ema_alpha,
        'ema_mode': model.ema_mode,
        'motion_k': model.motion_k,
        'motion_score_thresh': model.motion_score_thresh,
        'motion_max_step': model.motion_max_step,
        'dedup_iou_thresh': model.dedup_iou_thresh,
        'score_threshold': args.score_threshold,
        'update_score_threshold': args.update_score_threshold,
        'miss_tolerance': args.miss_tolerance,
    })

    # ---------- DanceTrack test-split iteration (identical to submit_dance) ----------
    sub_dir = 'DanceTrack/test'
    seq_nums = os.listdir(os.path.join(args.mot_path, sub_dir))
    if 'seqmap' in seq_nums:
        seq_nums.remove('seqmap')
    vids = [os.path.join(sub_dir, seq) for seq in seq_nums]

    rank = int(os.environ.get('RLAUNCH_REPLICA', '0'))
    ws = int(os.environ.get('RLAUNCH_REPLICA_TOTAL', '1'))
    vids = vids[rank::ws]

    for vid in vids:
        # Each sequence begins with a clean tracker state + empty stability
        # memory so max_obj_id starts at 0 and no cross-sequence score/
        # position history leaks in.
        model.clear()
        det = Detector(args, model=model, vid=vid)
        det.detect(args.score_threshold)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
