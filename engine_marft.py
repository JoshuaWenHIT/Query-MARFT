# ------------------------------------------------------------------------
# Copyright (c) 2026 JoshuaWenHIT. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Query-MARFT training engine.

Provides two training loops that **reuse** the native MOTRv2 data loading,
criterion, and model-init pipeline:

  * ``train_one_epoch_warmup``  — Phase 1 supervised warm-up (LoRA only).
  * ``train_one_epoch_marft``   — Phase 2 multi-agent RL fine-tuning.

Both loops follow the same iterator pattern as the original
``train_one_epoch_mot`` in engine.py.
"""

import math
import sys
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

import util.misc as utils
from datasets.data_prefetcher import data_dict_to_cuda
from util.reward_mechanisms import compute_reward_from_obj_idxes
from models.core.reward_fn import (
    HierarchicalRewardFn, RewardConfig, compute_step_result_from_frame,
)
from models.agents.agent_manager import AgentManager


# ======================================================================
# Phase 1  —  Supervised warm-up (LoRA adapters only)
# ======================================================================
def train_one_epoch_warmup(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    use_amp: bool = False,
):
    """
    Identical to ``train_one_epoch_mot`` but the model is a
    ``MOTRWithMARFT`` whose agents are disabled — only LoRA adapters
    are trainable.  The function signature and logging are kept compatible
    so that ``main.py`` can call either loop transparently.
    """
    model.train()
    criterion.train()
    # disable agents during warm-up
    if hasattr(model, 'module'):
        model.module.use_agents = False
    else:
        model.use_agents = False

    scaler = GradScaler(enabled=use_amp)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = f'Epoch: [{epoch}] MARFT-Warmup'
    print_freq = 10

    for data_dict in metric_logger.log_every(data_loader, print_freq, header):
        data_dict = data_dict_to_cuda(data_dict, device)
        clip_dicts = list(utils.iter_mot_clip_dicts(data_dict))
        n_clip = len(clip_dicts)
        loss_dict = None
        losses = None

        for sub in clip_dicts:
            with autocast(enabled=use_amp):
                outputs = model(sub)
                ld = criterion(outputs, sub)
                weight_dict = criterion.weight_dict
                li = sum(ld[k] * weight_dict[k] for k in ld if k in weight_dict)
            if loss_dict is None:
                loss_dict = {k: ld[k].clone() for k in ld}
                losses = li
            else:
                for k in ld:
                    loss_dict[k] = loss_dict[k] + ld[k]
                losses = losses + li

        for k in list(loss_dict.keys()):
            loss_dict[k] /= n_clip
        losses = losses / n_clip

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        weight_dict = criterion.weight_dict
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k] for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        loss_value = sum(loss_dict_reduced_scaled.values()).item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            grad_total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm) if max_norm > 0 else \
                utils.get_total_grad_norm(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            grad_total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm) if max_norm > 0 else \
                utils.get_total_grad_norm(model.parameters(), max_norm)
            optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ======================================================================
# Phase 2  —  Multi-agent RL fine-tuning (MARFT)
# ======================================================================
def train_one_epoch_marft(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    # --- MARFT hyper-params ---
    lambda_sup: float = 0.3,
    lambda_rl: float = 1.0,
    grpo_group_size: int = 4,
    id_switch_penalty: float = 15.0,
    id_stable_reward: float = 1.0,
    fp_penalty: float = 2.0,
    grpo_loss_weight: float = 1.0,
    sequential_optim: bool = True,
    use_amp: bool = False,
    reward_config: Optional[RewardConfig] = None,
    grad_efficient_grpo: bool = True,
):
    """
    Phase 2 training loop.

    Loss = λ_sup · L_supervised + λ_rl · L_RL

    L_RL is computed per-agent using GRPO-style group advantage and the
    hierarchical reward function. When ``sequential_optim`` is True the
    agents are updated one at a time in topological order.

    Memory model (the OOM-fix from 2026-04-25)
    -------------------------------------------
    The original implementation kept the autograd graph of *every* GRPO
    sample alive (each sample's ``total_log_prob`` referenced its forward
    activations, and ``outputs_for_criterion`` retained the last one).
    With ``grpo_group_size=4`` and 5-frame clips, this multiplied the
    Phase-1 activation budget by ~4× and reliably OOM-ed on 32 GB V100s
    on the very first MARFT step after warm-up.

    When ``grad_efficient_grpo=True`` (default), Phase 2 instead runs
    ``G-1`` rollouts under ``torch.no_grad()`` purely to estimate the
    GRPO baseline ``mean(r_k)``, then performs one final rollout *with*
    autograd whose log-prob carries the policy gradient.  This is the
    REINFORCE-with-G-sample-baseline estimator (Williams 1992; a strict
    subset of Shao et al. 2024 GRPO recovered by zeroing the gradient
    contribution of all but the last sample).  It is unbiased, has the
    same baseline-induced variance reduction as full GRPO, and caps
    Phase-2 activation memory at ~1.05× Phase-1.

    Setting ``grad_efficient_grpo=False`` reverts to the original (high
    memory) path for ablation studies.

    DDP compatibility (PyTorch 1.7, find_unused_parameters=True)
    -----------------------------------------------------------
    DistributedDataParallel runs ``prepare_for_backward`` inside every
    ``forward()`` call when ``find_unused_parameters=True``.  Issuing
    ``G`` consecutive forwards before a single backward inflates the
    per-parameter "expected ready count" to ``G`` while the backward
    only fires the gradient hook once per parameter, triggering
    ``RuntimeError: Expected to mark a variable ready only once``.

    To stay equivalent to the Phase-1 contract (one DDP forward per
    backward), the ``G-1`` no_grad rollouts are dispatched directly to
    ``model.module`` (bypassing the DDP wrapper) and only the final,
    grad-bearing rollout goes through ``model(...)``.  Bypassing DDP is
    safe under ``no_grad`` because the wrapper's only side-effect is to
    arrange the upcoming backward, which never happens for those calls.
    """
    model.train()
    criterion.train()

    # ensure agents are active
    base = model.module if hasattr(model, 'module') else model
    base.use_agents = True

    reward_fn = HierarchicalRewardFn(reward_config)
    scaler = GradScaler(enabled=use_amp)

    # Drop fragmentation accumulated during Phase 1.  Cheap (one call per
    # epoch) and avoids the case where the allocator holds reserved-but-
    # unused blocks that would otherwise force Phase-2 cudaMalloc to fail
    # even though the working set fits.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('reward', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('loss_rl', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Epoch: [{epoch}] MARFT'
    print_freq = 10

    agent_names = AgentManager.EXEC_ORDER

    # Underlying nn.Module — used for the no_grad rollouts so the DDP
    # wrapper sees exactly one forward call per backward (matching the
    # Phase-1 contract).  PyTorch 1.7's DDP runs ``prepare_for_backward``
    # inside every ``forward()`` call when ``find_unused_parameters=True``;
    # invoking ``model(sub)`` G times before a single backward inflates
    # the per-parameter "expected ready count", which is what triggers
    # ``RuntimeError: Expected to mark a variable ready only once`` even
    # when G-1 of those calls are wrapped in ``torch.no_grad()``.
    inner_module = model.module if hasattr(model, 'module') else model

    for data_dict in metric_logger.log_every(data_loader, print_freq, header):
        data_dict = data_dict_to_cuda(data_dict, device)
        clip_dicts = list(utils.iter_mot_clip_dicts(data_dict))
        n_clip = len(clip_dicts)
        weight_dict = criterion.weight_dict
        loss_dict = None
        losses_sup = None
        loss_rl_acc = torch.tensor(0.0, device=device)

        for sub in clip_dicts:
            sub['run_mode'] = 'sampling'

            # ----- GRPO group sampling -----
            # In grad-efficient mode (default) only the LAST rollout
            # carries an autograd graph; the previous (G-1) rollouts run
            # under no_grad and contribute *only* their reward to the
            # group-mean baseline.  In legacy mode every rollout retains
            # its graph (high memory; reproduces the pre-fix behaviour
            # for ablation).
            rewards_list: List[torch.Tensor] = []
            total_log_prob_grad: Optional[torch.Tensor] = None
            agent_lps_grad: Dict[str, torch.Tensor] = {}
            outputs_for_criterion = None
            # Legacy-mode buffers (only populated when not grad_efficient).
            legacy_total_log_probs: List[torch.Tensor] = []
            legacy_agent_log_probs: List[Dict[str, torch.Tensor]] = []

            last_idx = grpo_group_size - 1
            for sample_idx in range(grpo_group_size):
                sub['skip_loss'] = sample_idx < last_idx
                is_grad_sample = (sample_idx == last_idx) or (
                    not grad_efficient_grpo)
                # Activations are NOT stored when no_grad is active, so
                # the no_grad rollouts cost only the parameter-resident
                # memory (~constant) regardless of clip length.
                #
                # IMPORTANT (2026-04-25 DDP fix):
                # Only the grad-bearing rollout goes through the DDP
                # wrapper.  The no_grad rollouts are dispatched directly
                # to the underlying ``inner_module`` so DDP sees exactly
                # **one** ``forward()`` per ``backward()`` — matching the
                # Phase-1 contract.  Without this guard, PyTorch 1.7's
                # DDP runs ``prepare_for_backward`` inside every wrapper
                # ``forward`` (find_unused_parameters=True), inflates the
                # per-parameter "expected ready count" to G, and the
                # single backward (which only fires hooks once per param)
                # trips ``RuntimeError: Expected to mark a variable ready
                # only once``.
                grad_ctx = (torch.enable_grad() if is_grad_sample
                            else torch.no_grad())
                forward_target = model if is_grad_sample else inner_module
                with grad_ctx, autocast(enabled=use_amp):
                    outputs = forward_target(sub)

                obj_idxes_seq = outputs.get('obj_idxes_seq', [])
                scores_seq = outputs.get('scores_seq', [])
                log_probs = outputs.get('log_prob', [])

                if obj_idxes_seq and log_probs:
                    reward = compute_reward_from_obj_idxes(
                        obj_idxes_seq,
                        id_switch_penalty=id_switch_penalty,
                        id_stable_reward=id_stable_reward,
                        fp_penalty=fp_penalty,
                        scores_seq=scores_seq,
                    )
                    # Detach reward — it depends on sampled discrete
                    # obj_idxes which are non-differentiable anyway, but
                    # being explicit avoids ever propagating a graph
                    # reference through the rewards list.
                    rewards_list.append(reward.detach())

                if is_grad_sample:
                    # The single grad-bearing forward owns:
                    #   - the supervised loss target (criterion input)
                    #   - the policy-gradient log_prob factor
                    outputs_for_criterion = outputs
                    if log_probs:
                        total_log_prob_grad = sum(
                            lp.sum() for lp in log_probs if lp is not None)
                    for infos in outputs.get('agent_log_probs', []):
                        for name, info in infos.items():
                            lp = (info.log_prob if hasattr(info, 'log_prob')
                                  else None)
                            if lp is not None:
                                agent_lps_grad[name] = agent_lps_grad.get(
                                    name,
                                    torch.tensor(0.0, device=device)) + lp.sum()
                    if not grad_efficient_grpo:
                        legacy_total_log_probs.append(total_log_prob_grad)
                        legacy_agent_log_probs.append(dict(agent_lps_grad))
                else:
                    if not grad_efficient_grpo:
                        if log_probs:
                            legacy_total_log_probs.append(
                                sum(lp.sum() for lp in log_probs
                                    if lp is not None))
                        per_agent_lps: Dict[str, torch.Tensor] = {}
                        for infos in outputs.get('agent_log_probs', []):
                            for name, info in infos.items():
                                lp = (info.log_prob
                                      if hasattr(info, 'log_prob') else None)
                                if lp is not None:
                                    per_agent_lps[name] = per_agent_lps.get(
                                        name,
                                        torch.tensor(0.0,
                                                     device=device)) + lp.sum()
                        legacy_agent_log_probs.append(per_agent_lps)
                    # Drop the no_grad outputs immediately so the inner
                    # tensors (track_instances, hs, etc.) become
                    # garbage-collectable before the next forward.
                    del outputs

            sub['skip_loss'] = False

            # ----- Supervised loss (uses the grad-bearing forward) -----
            with autocast(enabled=use_amp):
                ld = criterion(outputs_for_criterion, sub)
                sup_i = sum(ld[k] * weight_dict[k] for k in ld
                            if k in weight_dict)

            # ----- RL loss (GRPO advantage) -----
            loss_rl = torch.tensor(0.0, device=device)
            reward_val = 0.0

            if not grad_efficient_grpo:
                # Legacy path — reproduces the original (high-memory)
                # full-graph GRPO update for ablation studies only.
                if len(rewards_list) >= 2:
                    rewards_stack = torch.stack(rewards_list)
                    baseline = rewards_stack.mean()
                    advantages = rewards_stack - baseline
                    if sequential_optim:
                        for name in agent_names:
                            agent_lps = torch.stack([
                                alp.get(name,
                                        torch.tensor(0.0, device=device))
                                for alp in legacy_agent_log_probs
                            ])
                            loss_rl = loss_rl - (
                                advantages.detach() * agent_lps).sum()
                    else:
                        total_lps = torch.stack(legacy_total_log_probs)
                        loss_rl = -(advantages.detach() * total_lps).sum()
                    reward_val = rewards_stack.mean().item()
                elif len(rewards_list) == 1:
                    loss_rl = -rewards_list[0] * legacy_total_log_probs[0]
                    reward_val = rewards_list[0].item()
            else:
                # Grad-efficient path — REINFORCE with G-sample baseline.
                # advantage = r_last - mean(r_1..G); only sample G's
                # log_prob is differentiable.
                if rewards_list:
                    rewards_stack = torch.stack(rewards_list)
                    baseline = rewards_stack.mean().detach()
                    advantage = (rewards_list[-1] - baseline).detach()
                    reward_val = rewards_stack.mean().item()
                    if sequential_optim and agent_lps_grad:
                        for name in agent_names:
                            lp = agent_lps_grad.get(name)
                            if lp is not None:
                                loss_rl = loss_rl - advantage * lp
                    elif total_log_prob_grad is not None:
                        loss_rl = loss_rl - advantage * total_log_prob_grad

            metric_logger.update(reward=reward_val,
                                 loss_rl=loss_rl.item())

            if loss_dict is None:
                loss_dict = {k: ld[k].clone() for k in ld}
                losses_sup = sup_i
            else:
                for k in ld:
                    loss_dict[k] = loss_dict[k] + ld[k]
                losses_sup = losses_sup + sup_i
            loss_rl_acc = loss_rl_acc + loss_rl

        # average over clips
        for k in list(loss_dict.keys()):
            loss_dict[k] /= n_clip
        losses_sup = losses_sup / n_clip
        loss_rl_acc = loss_rl_acc / n_clip

        # total = λ_sup * supervised + λ_rl * RL
        losses = lambda_sup * losses_sup + lambda_rl * grpo_loss_weight * loss_rl_acc

        # reduce for logging
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k] for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        loss_sup_reduced = sum(loss_dict_reduced_scaled.values())
        loss_rl_reduced = loss_rl_acc
        if utils.is_dist_avail_and_initialized():
            loss_rl_reduced = loss_rl_acc.detach().clone()
            torch.distributed.all_reduce(loss_rl_reduced)
            loss_rl_reduced /= utils.get_world_size()
        loss_value = (lambda_sup * loss_sup_reduced
                      + lambda_rl * grpo_loss_weight * loss_rl_reduced).item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            grad_total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm) if max_norm > 0 else \
                utils.get_total_grad_norm(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            grad_total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm) if max_norm > 0 else \
                utils.get_total_grad_norm(model.parameters(), max_norm)
            optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
