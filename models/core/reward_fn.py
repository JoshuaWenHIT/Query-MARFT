# ------------------------------------------------------------------------
# Copyright (c) 2026 JoshuaWenHIT. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Hierarchical Reward Function for Query-MARFT.

Two layers:
  1. Per-step (per-frame) immediate reward  — fast, per-agent, partially
     differentiable.
  2. Episode-level global reward  — CLEAR-MOT-style metrics, distributed
     back to each step via credit assignment.

This module reuses the existing ``compute_reward_from_obj_idxes`` for the
global signal and adds per-agent decomposition on top.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


# ======================================================================
# Reward hyper-parameters (dataclass for easy YAML serialisation)
# ======================================================================
@dataclass
class RewardConfig:
    # --- per-step weights ---
    lam_det_conf: float = 0.5
    lam_det_loc: float = 0.3
    lam_assoc_match: float = 1.0
    lam_assoc_idsw: float = -5.0
    lam_update_smooth: float = 0.3
    lam_update_consist: float = 0.3
    lam_corr_recover: float = 2.0
    lam_corr_fp: float = -1.0
    lam_corr_fn: float = -1.5

    # --- episode-level weights ---
    w_mota: float = 10.0
    w_idsw: float = -20.0
    w_idf1: float = 5.0
    w_frag: float = -3.0
    w_mt: float = 2.0


# ======================================================================
# Per-step reward
# ======================================================================
class HierarchicalRewardFn:
    """Compute both per-step and episode-level rewards."""

    def __init__(self, config: Optional[RewardConfig] = None):
        self.cfg = config or RewardConfig()

    # ------------------------------------------------------------------
    # Layer 1: immediate per-frame reward
    # ------------------------------------------------------------------
    def compute_step_reward(
        self,
        agent_name: str,
        result: Dict[str, Any],
    ) -> Tensor:
        """
        Compute the immediate reward for *one* agent on *one* frame.

        ``result`` is populated by the MARFT engine with whatever tensors
        are available; missing keys gracefully evaluate to zero.
        """
        device = _get_device(result)

        if agent_name == 'det':
            r_conf = result.get('detection_confidence_gain',
                                torch.tensor(0.0, device=device))
            r_loc = result.get('localization_giou',
                               torch.tensor(0.0, device=device))
            return self.cfg.lam_det_conf * r_conf + self.cfg.lam_det_loc * r_loc

        if agent_name == 'assoc':
            r_match = result.get('association_accuracy',
                                 torch.tensor(0.0, device=device))
            r_idsw = result.get('num_id_switches',
                                torch.tensor(0.0, device=device))
            return (self.cfg.lam_assoc_match * r_match
                    + self.cfg.lam_assoc_idsw * r_idsw)

        if agent_name == 'update':
            r_smooth = result.get('trajectory_smoothness',
                                  torch.tensor(0.0, device=device))
            r_consist = result.get('temporal_consistency',
                                   torch.tensor(0.0, device=device))
            return (self.cfg.lam_update_smooth * r_smooth
                    + self.cfg.lam_update_consist * r_consist)

        if agent_name == 'corr':
            r_rec = result.get('correct_recoveries',
                               torch.tensor(0.0, device=device))
            r_fp = result.get('false_positives',
                              torch.tensor(0.0, device=device))
            r_fn = result.get('false_negatives',
                              torch.tensor(0.0, device=device))
            return (self.cfg.lam_corr_recover * r_rec
                    + self.cfg.lam_corr_fp * r_fp
                    + self.cfg.lam_corr_fn * r_fn)

        raise ValueError(f"Unknown agent: {agent_name}")

    # ------------------------------------------------------------------
    # Layer 2: episode-level global reward
    # ------------------------------------------------------------------
    def compute_episode_reward(
        self,
        episode_stats: Dict[str, float],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute the global reward from CLEAR MOT statistics collected over
        an entire episode (video clip).

        ``episode_stats`` should contain keys like 'mota', 'idf1',
        'num_switches', 'num_fragments', 'mostly_tracked'.
        """
        c = self.cfg
        mota = episode_stats.get('mota', 0.0)
        idf1 = episode_stats.get('idf1', 0.0)
        nsw = episode_stats.get('num_switches', 0.0)
        nfrag = episode_stats.get('num_fragments', 0.0)
        mt = episode_stats.get('mostly_tracked', 0.0)

        R_mota = c.w_mota * mota
        R_idf1 = c.w_idf1 * idf1
        R_idsw = c.w_idsw * nsw
        R_frag = c.w_frag * nfrag
        R_mt = c.w_mt * mt

        R_global = R_mota + R_idf1 + R_idsw + R_frag + R_mt
        detail = dict(mota=R_mota, idf1=R_idf1, idsw=R_idsw,
                       frag=R_frag, mt=R_mt)
        return R_global, detail

    # ------------------------------------------------------------------
    # Credit assignment
    # ------------------------------------------------------------------
    @staticmethod
    def assign_credit(
        step_rewards: Dict[Tuple[int, str], float],
        episode_reward: float,
        episode_length: int,
        method: str = 'uniform',
        gamma: float = 0.99,
    ) -> Dict[Tuple[int, str], float]:
        """
        Distribute the episode-level reward back to per-(step, agent) slots.

        Methods:
            'uniform'      — equal share per step.
            'time_decay'   — exponential decay favouring recent steps.
        """
        if episode_length <= 0:
            return step_rewards

        agent_names = ['det', 'assoc', 'update', 'corr']

        if method == 'time_decay':
            tau = max(episode_length / 3.0, 1.0)
            raw_w = [math.exp((t - episode_length) / tau)
                     for t in range(episode_length)]
            total_w = sum(raw_w) or 1.0
            shares = [episode_reward * w / total_w for w in raw_w]
        else:
            per_step = episode_reward / episode_length
            shares = [per_step] * episode_length

        total_rewards: Dict[Tuple[int, str], float] = {}
        for t in range(episode_length):
            for name in agent_names:
                key = (t, name)
                r_step = step_rewards.get(key, 0.0)
                total_rewards[key] = r_step + shares[t]
        return total_rewards


# ======================================================================
# Helpers
# ======================================================================
def _get_device(d: Dict[str, Any]) -> torch.device:
    for v in d.values():
        if isinstance(v, Tensor):
            return v.device
    return torch.device('cpu')


def compute_step_result_from_frame(
    pred_boxes: Tensor,
    gt_boxes: Tensor,
    pred_scores: Tensor,
    obj_idxes: Tensor,
    prev_obj_idxes: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """
    Build the ``result`` dict expected by :meth:`compute_step_reward` from
    raw per-frame model outputs and ground truth.  Reuses MOTRv2-native
    tensor formats (cxcywh normalised boxes, obj_idxes from ClipMatcher).
    """
    device = pred_boxes.device
    result: Dict[str, Tensor] = {}

    # DetAgent metrics
    result['detection_confidence_gain'] = pred_scores.mean() if pred_scores.numel() else torch.tensor(0.0, device=device)

    active = obj_idxes >= 0
    if active.any() and gt_boxes.numel() > 0:
        matched_pred = box_cxcywh_to_xyxy(pred_boxes[active])
        matched_gt_idx = obj_idxes[active].clamp(min=0, max=gt_boxes.shape[0] - 1)
        matched_gt = box_cxcywh_to_xyxy(gt_boxes[matched_gt_idx])
        giou = generalized_box_iou(matched_pred, matched_gt)
        result['localization_giou'] = giou.diag().mean()
    else:
        result['localization_giou'] = torch.tensor(0.0, device=device)

    # AssocAgent metrics
    total_queries = obj_idxes.shape[0]
    n_matched = active.sum().float()
    result['association_accuracy'] = n_matched / max(total_queries, 1)
    n_idsw = torch.tensor(0.0, device=device)
    if prev_obj_idxes is not None:
        min_len = min(obj_idxes.shape[0], prev_obj_idxes.shape[0])
        cur = obj_idxes[:min_len]
        prev = prev_obj_idxes[:min_len]
        switched = (cur > 0) & (prev > 0) & (cur != prev)
        n_idsw = switched.float().sum()
    result['num_id_switches'] = n_idsw

    # UpdateAgent metrics (placeholders populated by engine)
    result['trajectory_smoothness'] = torch.tensor(0.0, device=device)
    result['temporal_consistency'] = torch.tensor(0.0, device=device)

    # CorrAgent metrics
    result['correct_recoveries'] = torch.tensor(0.0, device=device)
    result['false_positives'] = torch.tensor(0.0, device=device)
    result['false_negatives'] = torch.tensor(0.0, device=device)

    return result
