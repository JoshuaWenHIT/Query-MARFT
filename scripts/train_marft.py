# ------------------------------------------------------------------------
# Copyright (c) 2026 JoshuaWenHIT. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Query-MARFT training entry — two-phase training that fully reuses
MOTRv2's native dataset loading, model init, and checkpoint logic.

Phase 1 (warm-up):  Supervised training with LoRA adapters only.
Phase 2 (MARFT):    Multi-agent reinforced fine-tuning with GRPO advantage.

All MARFT settings are passed as CLI arguments via an .args file (mirroring
the MOTRv2 ``configs/Query-MARFT.args`` style).
"""

import argparse
import datetime
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset
from models import build_model
from models.motr_marft import build_marft_model
from models.core.reward_fn import RewardConfig
from engine_marft import train_one_epoch_warmup, train_one_epoch_marft
from util.tool import load_model
from main import get_args_parser as base_args_parser


# ======================================================================
# Argument parsing — extends the original MOTRv2 parser with MARFT flags
# ======================================================================
def get_args_parser():
    parser = argparse.ArgumentParser(
        'Query-MARFT training', parents=[base_args_parser()], add_help=True)
    g = parser.add_argument_group('MARFT')

    # ---- High-level switches ----
    g.add_argument('--marft_use_lora', type=int, default=1)
    g.add_argument('--marft_use_agents', type=int, default=1)
    g.add_argument('--marft_det_enabled', type=int, default=1)
    g.add_argument('--marft_assoc_enabled', type=int, default=1)
    g.add_argument('--marft_update_enabled', type=int, default=1)
    g.add_argument('--marft_corr_enabled', type=int, default=1)
    g.add_argument('--marft_corr_threshold', type=float, default=0.4)
    g.add_argument('--marft_scene_adaptive', type=int, default=1)
    g.add_argument('--marft_sequential_optim', type=int, default=1)
    g.add_argument('--marft_grad_efficient_grpo', type=int, default=1,
                   help='1 (default): GRPO baseline samples run under '
                        'no_grad and only the LAST rollout carries the '
                        'autograd graph (REINFORCE-with-G-sample baseline). '
                        'Caps Phase-2 activation memory at ~1.05x Phase-1. '
                        '0: legacy full-graph GRPO (~G x activations, OOMs '
                        'on 32GB GPUs with G>=4 and 5-frame clips).')

    # ---- Training schedule ----
    g.add_argument('--marft_warmup_epochs', type=int, default=3)
    g.add_argument('--marft_epochs', type=int, default=20)
    g.add_argument('--marft_lambda_sup', type=float, default=0.3)
    g.add_argument('--marft_lambda_rl', type=float, default=1.0)

    # ---- Optimiser ----
    g.add_argument('--marft_lr_lora', type=float, default=1e-4,
                   help='Phase-1 LoRA learning rate')
    g.add_argument('--marft_lr_lora_phase2', type=float, default=1e-5,
                   help='Phase-2 LoRA learning rate (smaller, prevent destruction)')
    g.add_argument('--marft_lr_agents', type=float, default=1e-5,
                   help='Phase-2 agent learning rate')

    # ---- LoRA per-region ranks ----
    g.add_argument('--marft_lora_backbone_r', type=int, default=8)
    g.add_argument('--marft_lora_encoder_r', type=int, default=16)
    g.add_argument('--marft_lora_decoder_r', type=int, default=24)
    g.add_argument('--marft_lora_query_embed_r', type=int, default=24)

    # ---- Hierarchical reward — per-step ----
    g.add_argument('--marft_lam_det_conf', type=float, default=0.5)
    g.add_argument('--marft_lam_det_loc', type=float, default=0.3)
    g.add_argument('--marft_lam_assoc_match', type=float, default=1.0)
    g.add_argument('--marft_lam_assoc_idsw', type=float, default=-5.0)
    g.add_argument('--marft_lam_update_smooth', type=float, default=0.3)
    g.add_argument('--marft_lam_update_consist', type=float, default=0.3)
    g.add_argument('--marft_lam_corr_recover', type=float, default=2.0)
    g.add_argument('--marft_lam_corr_fp', type=float, default=-1.0)
    g.add_argument('--marft_lam_corr_fn', type=float, default=-1.5)

    # ---- Hierarchical reward — episode-level ----
    g.add_argument('--marft_w_mota', type=float, default=10.0)
    g.add_argument('--marft_w_idsw', type=float, default=-20.0)
    g.add_argument('--marft_w_idf1', type=float, default=5.0)
    g.add_argument('--marft_w_frag', type=float, default=-3.0)
    g.add_argument('--marft_w_mt', type=float, default=2.0)

    # ---- Inference-time safety knobs (ID-explosion mitigation) ----
    # Purpose:
    #   MARFT agents operating on an under-trained checkpoint destabilise
    #   the track scores that RuntimeTrackerBase uses to decide rebirth.
    #   These flags bound each agent's intervention at inference WITHOUT
    #   modifying any training dynamics (defaults = original behaviour).
    g.add_argument('--marft_infer_det_delta_scale', type=float, default=1.0,
                   help='Scale DetAgent Δp at inference (0=no ref_pt perturbation).')
    g.add_argument('--marft_infer_assoc_alpha_gamma', type=float, default=1.0,
                   help="Blend AssocAgent α to neutral: α'=1+γ(α-1). "
                        "0=logits untouched, 1=original agent output.")
    g.add_argument('--marft_infer_corr_soft_factor', type=float, default=0.0,
                   help='When CorrAgent TERMINATEs, scores*=factor instead of =0. '
                        '0=original hard kill; 0.5..0.8 recommended at inference.')
    g.add_argument('--marft_infer_corr_consec_terminate', type=int, default=1,
                   help='Require N consecutive TERMINATE decisions (same obj_idx) '
                        'before acting. 1=original; 3 gives strong debouncing.')
    g.add_argument('--marft_infer_reset_per_seq', type=int, default=1,
                   help='At inference, call model.clear() before each sequence to '
                        'reset RuntimeTrackerBase.max_obj_id (per-sequence ID space).')

    return parser


def _build_lora_strategy(args) -> dict:
    """Build LoRA strategy from per-region rank args.

    Target short-names are chosen to match the Linear layers that ACTUALLY
    EXIST in MOTRv2 (MSDeformAttn uses ``value_proj`` / ``output_proj``;
    FFN uses ``linear1`` / ``linear2``; QIMv2 has ``linear*`` / ``linear_feat*``).
    nn.MultiheadAttention children are automatically filtered inside
    ``inject_lora`` (see ``lora_layers.py``).
    """
    r_bb = args.marft_lora_backbone_r
    r_enc = args.marft_lora_encoder_r
    r_dec = args.marft_lora_decoder_r
    r_qe = args.marft_lora_query_embed_r

    msdeform_targets = {'value_proj', 'output_proj'}
    ffn_targets = {'linear1', 'linear2'}
    return {
        # Backbone: ResNet is pure Conv2d; no Linear to inject.  Entry kept
        # for symmetry; will match no module and be a no-op.
        'backbone': dict(apply=True, r=r_bb, alpha=r_bb * 2, dropout=0.05,
                         targets=msdeform_targets),
        # Encoder MSDeformAttn + FFN
        'transformer.encoder': dict(apply=True, r=r_enc, alpha=r_enc * 2,
                                    dropout=0.05,
                                    targets=msdeform_targets | ffn_targets),
        # Decoder cross-attn (MSDeformAttn) + FFN.  Self-attn children are
        # auto-skipped as MultiheadAttention-descendants.
        'transformer.decoder': dict(apply=True, r=r_dec, alpha=r_dec * 2,
                                    dropout=0.1,
                                    targets=msdeform_targets | ffn_targets),
        # Output heads operating on Track Query embeddings
        'class_embed': dict(apply=True, r=r_qe, alpha=r_qe * 2, dropout=0.1,
                            targets={'*'}),
        'bbox_embed': dict(apply=True, r=r_qe, alpha=r_qe * 2, dropout=0.1,
                           targets={'*'}),
        # QIMv2 (track_embed) Linear layers
        'track_embed': dict(apply=True, r=r_qe, alpha=r_qe * 2, dropout=0.1,
                            targets={'linear1', 'linear2',
                                     'linear_feat1', 'linear_feat2',
                                     'linear_pos1', 'linear_pos2'}),
    }


def _build_reward_config(args) -> RewardConfig:
    return RewardConfig(
        lam_det_conf=args.marft_lam_det_conf,
        lam_det_loc=args.marft_lam_det_loc,
        lam_assoc_match=args.marft_lam_assoc_match,
        lam_assoc_idsw=args.marft_lam_assoc_idsw,
        lam_update_smooth=args.marft_lam_update_smooth,
        lam_update_consist=args.marft_lam_update_consist,
        lam_corr_recover=args.marft_lam_corr_recover,
        lam_corr_fp=args.marft_lam_corr_fp,
        lam_corr_fn=args.marft_lam_corr_fn,
        w_mota=args.marft_w_mota,
        w_idsw=args.marft_w_idsw,
        w_idf1=args.marft_w_idf1,
        w_frag=args.marft_w_frag,
        w_mt=args.marft_w_mt,
    )


# ======================================================================
# Main
# ======================================================================
def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    warmup_epochs = args.marft_warmup_epochs
    marft_epochs = args.marft_epochs
    total_epochs = warmup_epochs + marft_epochs

    reward_config = _build_reward_config(args)

    # ---------- Build base MOTR model (native, unmodified) ----------
    base_model, criterion, postprocessors = build_model(args)
    base_model.to(device)

    if args.pretrained is not None:
        base_model = load_model(base_model, args.pretrained)

    # ---------- Wrap with MARFT ----------
    args.marft_use_lora = bool(args.marft_use_lora)
    args.marft_use_agents = bool(args.marft_use_agents)

    # Build with per-agent enable flags + per-region LoRA strategy
    from models.motr_marft import MOTRWithMARFT
    model = MOTRWithMARFT(
        base_model,
        hidden_dim=getattr(args, 'hidden_dim', 256),
        use_lora=args.marft_use_lora,
        lora_strategy=_build_lora_strategy(args) if args.marft_use_lora else None,
        use_agents=args.marft_use_agents,
        agent_config={
            'corr': {'corr_conf_threshold': args.marft_corr_threshold},
        },
    )
    # Per-agent ablation toggles
    model.agent_manager.set_agent_enabled('det', bool(args.marft_det_enabled))
    model.agent_manager.set_agent_enabled('assoc', bool(args.marft_assoc_enabled))
    model.agent_manager.set_agent_enabled('update', bool(args.marft_update_enabled))
    model.agent_manager.set_agent_enabled('corr', bool(args.marft_corr_enabled))
    model.to(device)

    # ---------- Dataset (100 % reuse) ----------
    dataset_train = build_dataset(image_set='train', args=args)

    if args.distributed:
        sampler_train = (samplers.NodeDistributedSampler(dataset_train)
                         if args.cache_mode
                         else samplers.DistributedSampler(dataset_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    pin_memory = args.batch_size == 1
    data_loader_train = DataLoader(
        dataset_train, batch_sampler=batch_sampler_train,
        collate_fn=utils.mot_collate_fn, num_workers=args.num_workers,
        pin_memory=pin_memory)

    # ---------- Optimiser: separate param groups ----------
    lora_params = model.get_lora_params()
    agent_params = model.get_agent_params()

    param_groups_warmup = [{'params': lora_params, 'lr': args.marft_lr_lora}]
    param_groups_warmup = [g for g in param_groups_warmup if g['params']]

    param_groups_marft = [
        {'params': lora_params, 'lr': args.marft_lr_lora_phase2},
        {'params': agent_params, 'lr': args.marft_lr_agents},
    ]
    param_groups_marft = [g for g in param_groups_marft if g['params']]

    optimizer_warmup = torch.optim.AdamW(
        param_groups_warmup, lr=args.marft_lr_lora,
        weight_decay=args.weight_decay
    ) if param_groups_warmup else None

    optimizer_marft = torch.optim.AdamW(
        param_groups_marft, lr=args.marft_lr_agents,
        weight_decay=args.weight_decay
    ) if param_groups_marft else None

    lr_scheduler_warmup = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_warmup, T_max=max(warmup_epochs, 1))
        if optimizer_warmup else None
    )
    lr_scheduler_marft = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_marft, T_max=max(marft_epochs, 1))
        if optimizer_marft else None
    )

    # ---------- DDP ----------
    # Only wrap with DistributedDataParallel when there is genuine work
    # for it to do (world_size > 1).  Under a single-process launch
    # (``--nproc_per_node=1``) ``args.distributed`` is True but
    # ``args.world_size == 1``: DDP performs no cross-GPU sync, yet its
    # reducer + ``find_unused_parameters=True`` machinery still runs.
    #
    # That machinery has a documented edge case with PyTorch 1.7 in
    # MARFT Phase 2: ``_find_tensors`` traces every tensor returned by
    # ``forward()``, and the Phase-2 outputs include several tensors
    # (``pred_logits``, the Bernoulli sampling ``log_prob``, the
    # ``agent_log_probs`` dict, and the criterion's accumulated
    # ``losses_dict``) that share overlapping subgraphs through the
    # MOTR / LoRA / AssocAgent parameters.  When ``losses.backward()``
    # then traverses only a *subset* of those subgraphs, the reducer's
    # synthetic "unused parameter" mark and the autograd-hook real mark
    # collide on the same parameter, raising::
    #
    #     RuntimeError: Expected to mark a variable ready only once.
    #
    # For single-process training the entire DDP layer is unnecessary,
    # so we simply skip wrapping.  Multi-process training (world_size
    # >= 2) still goes through DDP unchanged; users running multi-GPU
    # MARFT should additionally consider upgrading to a PyTorch version
    # that supports ``static_graph=True`` (>=1.11).
    model_without_ddp = model
    world_size = getattr(args, 'world_size', 1)
    if args.distributed and world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # ---------- Resume ----------
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        missing, unexpected = model_without_ddp.load_state_dict(
            ckpt['model'], strict=False)
        if missing:
            print(f'Missing Keys: {missing}')
        if unexpected:
            print(f'Unexpected Keys: {unexpected}')
        if 'epoch' in ckpt:
            args.start_epoch = ckpt['epoch'] + 1

    output_dir = Path(args.output_dir)
    print("Start Query-MARFT training")
    start_time = time.time()
    dataset_train.set_epoch(args.start_epoch)

    # ============================
    #  Phase 1: Supervised warm-up
    # ============================
    for epoch in range(args.start_epoch,
                       min(args.start_epoch + warmup_epochs, total_epochs)):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        if optimizer_warmup is None:
            break

        train_one_epoch_warmup(
            model, criterion, data_loader_train, optimizer_warmup,
            device, epoch, args.clip_max_norm, use_amp=args.use_amp)

        if lr_scheduler_warmup is not None:
            lr_scheduler_warmup.step()

        _save_checkpoint(model_without_ddp, optimizer_warmup,
                         lr_scheduler_warmup, epoch, args, output_dir,
                         total_epochs)
        dataset_train.step_epoch()

    # ============================
    #  Phase 2: MARFT RL training
    # ============================
    marft_start = args.start_epoch + warmup_epochs
    for epoch in range(marft_start, marft_start + marft_epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        if optimizer_marft is None:
            break

        train_one_epoch_marft(
            model, criterion, data_loader_train, optimizer_marft,
            device, epoch, args.clip_max_norm,
            lambda_sup=args.marft_lambda_sup,
            lambda_rl=args.marft_lambda_rl,
            grpo_group_size=args.grpo_group_size,
            id_switch_penalty=args.grpo_id_switch_penalty,
            id_stable_reward=args.grpo_id_stable_reward,
            fp_penalty=args.grpo_fp_penalty,
            grpo_loss_weight=args.grpo_loss_weight,
            sequential_optim=bool(args.marft_sequential_optim),
            use_amp=args.use_amp,
            reward_config=reward_config,
            grad_efficient_grpo=bool(args.marft_grad_efficient_grpo),
        )

        if lr_scheduler_marft is not None:
            lr_scheduler_marft.step()

        _save_checkpoint(model_without_ddp, optimizer_marft,
                         lr_scheduler_marft, epoch, args, output_dir,
                         total_epochs)
        dataset_train.step_epoch()

    total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f'Training time {total_time_str}')


def _save_checkpoint(model, optimizer, scheduler, epoch, args, output_dir,
                     total_epochs):
    if not args.output_dir:
        return
    checkpoint_paths = [output_dir / 'checkpoint.pth']
    if ((epoch + 1) % 5 == 0) or (epoch + 1 == total_epochs):
        checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
    for p in checkpoint_paths:
        utils.save_on_master({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': scheduler.state_dict() if scheduler else None,
            'epoch': epoch,
            'args': args,
        }, p)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
