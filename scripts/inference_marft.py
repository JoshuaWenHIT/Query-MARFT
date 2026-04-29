# ------------------------------------------------------------------------
# Copyright (c) 2026 JoshuaWenHIT. All Rights Reserved.
# ------------------------------------------------------------------------
# Query-MARFT inference entry — mirrors ``submit_dance.py`` but wraps the
# MOTR base model with ``MOTRWithMARFT`` (LoRA + four agents) before
# loading the MARFT checkpoint.  Reuses 100% of MOTRv2's
# ``ListImgDataset`` / ``Detector`` / ``RuntimeTrackerBase`` logic.
# ------------------------------------------------------------------------

import argparse
import os
import sys
from pathlib import Path

import torch

# Ensure project root is on sys.path before importing project modules.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model
from models.motr_marft import MOTRWithMARFT
from submit_dance import Detector, RuntimeTrackerBase  # reuse original classes
from scripts.train_marft import (
    get_args_parser as get_marft_args_parser,
    _build_lora_strategy,
)


def main(args):
    # ---------- Build base MOTR (native, unmodified) ----------
    base_model, _, _ = build_model(args)

    # ---------- Wrap with MARFT (LoRA + agents) ----------
    args.marft_use_lora = bool(args.marft_use_lora)
    args.marft_use_agents = bool(args.marft_use_agents)
    # Pull inference-time safety knobs straight from the MARFT args parser.
    # These directly mitigate the "ID explosion" failure mode where
    # under-trained agents destabilise track scores and the native
    # RuntimeTrackerBase rebirths a new ID every time a track dips below
    # ``filter_score_thresh`` for ``miss_tolerance`` frames.
    infer_safety = dict(
        det_delta_scale=getattr(args, 'marft_infer_det_delta_scale', 1.0),
        assoc_alpha_gamma=getattr(args, 'marft_infer_assoc_alpha_gamma', 1.0),
        corr_soft_factor=getattr(args, 'marft_infer_corr_soft_factor', 0.0),
        corr_consec_terminate=getattr(
            args, 'marft_infer_corr_consec_terminate', 1),
    )
    model = MOTRWithMARFT(
        base_model,
        hidden_dim=getattr(args, 'hidden_dim', 256),
        use_lora=args.marft_use_lora,
        lora_strategy=_build_lora_strategy(args) if args.marft_use_lora else None,
        use_agents=args.marft_use_agents,
        agent_config={
            'corr': {'corr_conf_threshold': args.marft_corr_threshold},
        },
        infer_safety=infer_safety,
    )
    print(f'[inference] infer_safety = {model.infer_safety}')
    # Per-agent ablation toggles (respect CLI flags)
    model.agent_manager.set_agent_enabled('det', bool(args.marft_det_enabled))
    model.agent_manager.set_agent_enabled('assoc', bool(args.marft_assoc_enabled))
    model.agent_manager.set_agent_enabled('update', bool(args.marft_update_enabled))
    model.agent_manager.set_agent_enabled('corr', bool(args.marft_corr_enabled))

    # ---------- Runtime tracker configuration on base_model ----------
    base_model.track_embed.score_thr = args.update_score_threshold
    model.track_base = RuntimeTrackerBase(
        args.score_threshold, args.score_threshold, args.miss_tolerance)

    # ---------- Load MARFT checkpoint ----------
    assert args.resume, 'Please pass --resume <checkpoint.pth>'
    ckpt = torch.load(args.resume, map_location='cpu')
    state_dict = ckpt.get('model', ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # Filter expected-missing keys (e.g. cuda-op profiling buffers from thop).
    unexpected = [k for k in unexpected
                  if not (k.endswith('total_params') or k.endswith('total_ops'))]
    if missing:
        print(f'[inference] Missing Keys ({len(missing)}): {missing[:10]}'
              f'{" ..." if len(missing) > 10 else ""}')
    if unexpected:
        print(f'[inference] Unexpected Keys ({len(unexpected)}): {unexpected[:10]}'
              f'{" ..." if len(unexpected) > 10 else ""}')

    model.eval()
    model = model.cuda()

    # ---------- DanceTrack test-split iteration (identical to submit_dance) ----------
    sub_dir = 'DanceTrack/test'
    seq_nums = os.listdir(os.path.join(args.mot_path, sub_dir))
    if 'seqmap' in seq_nums:
        seq_nums.remove('seqmap')
    vids = [os.path.join(sub_dir, seq) for seq in seq_nums]

    rank = int(os.environ.get('RLAUNCH_REPLICA', '0'))
    ws = int(os.environ.get('RLAUNCH_REPLICA_TOTAL', '1'))
    vids = vids[rank::ws]

    reset_per_seq = bool(getattr(args, 'marft_infer_reset_per_seq', 1))
    for vid in vids:
        # Each DanceTrack sequence must start with a clean tracker state so
        # that ``max_obj_id`` begins from 0 (otherwise IDs accumulate across
        # sequences, e.g. dancetrack0003 starting at ID 9013 because earlier
        # sequences already consumed 0..9012).  ``model.clear()`` also
        # resets the CorrAgent debounce counter.
        if reset_per_seq:
            model.clear()
        det = Detector(args, model=model, vid=vid)
        det.detect(args.score_threshold)


if __name__ == '__main__':
    # Reuse the MARFT parser directly (which inherits all original MOTRv2 args
    # via main.py and adds the --marft_* flags), then tack on the three
    # inference-specific knobs used by submit_dance.py.
    #
    # NOTE: we don't wrap via ``parents=[get_marft_args_parser()]`` because
    # the returned parser already has ``add_help=True``, and nesting another
    # parser with the same flag triggers::
    #   argparse.ArgumentError: argument -h/--help: conflicting option strings
    parser = get_marft_args_parser()
    parser.add_argument('--score_threshold', default=0.5, type=float)
    parser.add_argument('--update_score_threshold', default=0.5, type=float)
    parser.add_argument('--miss_tolerance', default=20, type=int)
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
