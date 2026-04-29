# ------------------------------------------------------------------------
# Copyright (c) 2026 JoshuaWenHIT. All Rights Reserved.
# ------------------------------------------------------------------------
"""
MOTRWithStability — inference-time stability wrapper around native MOTR.

Purpose
-------
Reduce ID churn on DanceTrack (and similar MOT scenarios with heavy
occlusion and near-identical appearance) WITHOUT retraining and WITHOUT
modifying any MOTRv2 source file.  The wrapper targets the dominant
failure mode on these datasets: a single-frame score dip (caused by
occlusion / pose ambiguity) pushes a live track below
``RuntimeTrackerBase.filter_score_thresh``, the track becomes
"disappeared", after ``miss_tolerance`` frames its ``obj_idx`` is reset
to -1, and when the same real object re-appears it is assigned a brand
new ID — killing IDF1 / HOTA.

Three cooperating stability mechanisms:

1. **Track-score EMA** — intercept ``track_base.update`` (installed via
   a runtime hook, *not* a source edit) and apply an exponential moving
   average to each track's ``scores`` using a per-``obj_idx`` memory,
   BEFORE the rebirth / disappearance decisions run.  Single-frame dips
   no longer cross the threshold.

2. **ref_pts motion extrapolation** — for incoming tracks whose score
   is low, push the xy-center of their reference points one frame along
   the last-K-frame velocity *before* the Decoder runs.  Cross-attention
   still samples features near the true object, so the next-frame score
   has a chance to recover rather than flat-lining.

3. **Same-frame cross-ID dedup** — when two different ``obj_idx`` boxes
   overlap with IoU ≥ ``dedup_iou_thresh`` in the same frame, drop the
   track with the shorter position history.  This removes the common
   case of MOTRv2 "forking" a single real object into two IDs.

Design principles
-----------------
* The native MOTR model is wrapped, never edited.  Instance-level
  method replacement on ``track_base.update`` is consistent with the
  same pattern used in ``models/amp_patches.py``.
* No learnable parameters.  Pure inference.
* Every mechanism is independently toggleable via a scalar:
  ``score_ema_alpha=0``, ``motion_k<2``, ``dedup_iou_thresh>=1`` each
  disables its own feature.  Default config is mild and intended to be
  non-regressive on DanceTrack baselines.
* Per-sequence state (``_prev_scores``, ``_pos_history``) MUST be
  cleared between videos via ``clear()`` (auto-called by the provided
  inference script).
"""

from __future__ import annotations

import functools
from collections import deque
from typing import Any, Deque, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from models.structures import Instances
from util.misc import NestedTensor, nested_tensor_from_tensor_list


class MOTRWithStability(nn.Module):
    """Inference-only stability overlay around an *unmodified* MOTR instance."""

    def __init__(
        self,
        base_model: nn.Module,
        score_ema_alpha: float = 0.3,
        ema_mode: str = 'asymmetric_down',
        motion_k: int = 1,           # <2 disables extrapolation (default OFF)
        motion_score_thresh: float = 0.3,
        motion_max_step: float = 0.03,
        dedup_iou_thresh: float = 1.1,  # >=1 disables dedup (default OFF)
    ):
        super().__init__()
        self.base_model = base_model
        self.score_ema_alpha = float(score_ema_alpha)
        assert ema_mode in ('symmetric', 'asymmetric_down', 'off'), ema_mode
        self.ema_mode = str(ema_mode)
        self.motion_k = int(motion_k)
        self.motion_score_thresh = float(motion_score_thresh)
        self.motion_max_step = float(motion_max_step)
        self.dedup_iou_thresh = float(dedup_iou_thresh)

        # Per-obj_idx stability state; reset on clear().
        self._prev_scores: Dict[int, float] = {}
        self._pos_history: Dict[int, Deque] = {}

        # Install runtime hooks on the current track_base.
        self._install_score_ema_hook()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Must be called before each new sequence."""
        self.base_model.clear()
        self._prev_scores.clear()
        self._pos_history.clear()

    def set_stability(self, **kwargs) -> None:
        """Tune any stability knob at runtime."""
        for k, v in kwargs.items():
            if v is None or not hasattr(self, k):
                continue
            setattr(self, k, type(getattr(self, k))(v))

    # ------------------------------------------------------------------
    # Transparent delegation so submit_dance-style scripts work verbatim
    # ------------------------------------------------------------------
    @property
    def track_embed(self):
        return self.base_model.track_embed

    @property
    def track_base(self):
        return self.base_model.track_base

    @track_base.setter
    def track_base(self, value):
        self.base_model.track_base = value
        # A new track_base means we need to re-install the EMA hook.
        self._install_score_ema_hook()

    @property
    def post_process(self):
        return self.base_model.post_process

    # ------------------------------------------------------------------
    # (1) Score EMA — monkey-patch track_base.update on this instance.
    #     We replace the bound method of ONE object; the RuntimeTrackerBase
    #     class in models/motr.py is untouched.
    # ------------------------------------------------------------------
    def _install_score_ema_hook(self) -> None:
        track_base = self.base_model.track_base
        # If already hooked (e.g., re-install on same instance), unwrap first
        # so we don't stack hooks.
        current = getattr(track_base, 'update', None)
        if getattr(current, '_stability_hooked', False):
            original_update = getattr(current, '_stability_original', current)
        else:
            original_update = current

        wrapper_self = self

        @functools.wraps(original_update)
        def hooked_update(track_instances: Instances) -> None:
            n = len(track_instances)
            # 1) EMA-smooth scores for existing (obj_idx >= 0) tracks ONLY.
            #    Newly-born tracks (obj_idx == -1) keep their raw scores so
            #    the original birth logic still works.
            #
            # ``ema_mode`` controls WHEN smoothing applies:
            #   * ``off``             — never smooth (disables the feature).
            #   * ``symmetric``       — smooth every frame (original behaviour).
            #   * ``asymmetric_down`` — smooth only when cur < prev.  This is
            #     the SAFE default: it dampens single-frame score dips
            #     (preventing spurious rebirth) but lets genuinely-dying
            #     tracks die at their natural rate, and lets recovering
            #     tracks recover at full speed.  Crucially, it does NOT
            #     artificially keep stale IDs alive, which was the main
            #     source of HOTA regression with the symmetric default.
            alpha = wrapper_self.score_ema_alpha
            mode = wrapper_self.ema_mode
            if alpha > 0.0 and mode != 'off' and n > 0:
                obj_idxes = track_instances.obj_idxes
                scores = track_instances.scores
                prev = wrapper_self._prev_scores
                for i in range(n):
                    oid = int(obj_idxes[i].item())
                    if oid < 0:
                        continue
                    p = prev.get(oid)
                    if p is None:
                        continue
                    cur = float(scores[i].item())
                    if mode == 'asymmetric_down' and cur >= p:
                        continue  # let upward recovery pass through
                    scores[i] = alpha * p + (1.0 - alpha) * cur

            # 2) Delegate to the original RuntimeTrackerBase.update using
            #    the (possibly smoothed) scores.  This is where rebirth /
            #    disappear decisions happen — now stabilised.
            original_update(track_instances)

            # 3) Record scores for next-frame EMA, keyed by FINAL obj_idx
            #    (newly born IDs are also captured here).
            if n > 0:
                obj_idxes = track_instances.obj_idxes
                scores = track_instances.scores
                for i in range(n):
                    oid = int(obj_idxes[i].item())
                    if oid >= 0:
                        wrapper_self._prev_scores[oid] = float(
                            scores[i].item())

        hooked_update._stability_hooked = True  # type: ignore[attr-defined]
        hooked_update._stability_original = original_update  # type: ignore[attr-defined]
        track_base.update = hooked_update  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # (2) ref_pts motion extrapolation — applied BEFORE the Decoder runs,
    #     i.e. on the incoming track_instances carried over from frame N-1.
    # ------------------------------------------------------------------
    def _apply_motion_extrapolation(self, track_instances: Instances) -> None:
        if self.motion_k < 2 or len(track_instances) == 0:
            return
        if not hasattr(track_instances, 'scores'):
            return

        obj_idxes = track_instances.obj_idxes
        scores = track_instances.scores
        ref_pts = track_instances.ref_pts  # [N, 2] center or [N, 4] cxcywh
        step_cap = self.motion_max_step
        score_gate = self.motion_score_thresh

        for i in range(len(track_instances)):
            oid = int(obj_idxes[i].item())
            if oid < 0:
                continue
            if float(scores[i].item()) >= score_gate:
                continue
            hist = self._pos_history.get(oid)
            if hist is None or len(hist) < 2:
                continue
            tail = list(hist)[-self.motion_k:]
            dt = max(1, len(tail) - 1)
            vx = (tail[-1][0] - tail[0][0]) / dt
            vy = (tail[-1][1] - tail[0][1]) / dt
            # Cap step to avoid sending the query off the image.
            vx = max(-step_cap, min(step_cap, vx))
            vy = max(-step_cap, min(step_cap, vy))
            new_cx = float(ref_pts[i, 0].item()) + vx
            new_cy = float(ref_pts[i, 1].item()) + vy
            ref_pts[i, 0] = max(0.0, min(1.0, new_cx))
            ref_pts[i, 1] = max(0.0, min(1.0, new_cy))

    def _record_position_history(self, track_instances: Instances) -> None:
        """After the full frame is processed, archive pred_boxes (normalized
        cxcywh) of every live track for next-frame velocity estimation."""
        if not hasattr(track_instances, 'pred_boxes'):
            return
        if not hasattr(track_instances, 'obj_idxes'):
            return
        obj_idxes = track_instances.obj_idxes
        pred_boxes = track_instances.pred_boxes
        maxlen = max(self.motion_k, 5)
        for i in range(len(track_instances)):
            oid = int(obj_idxes[i].item())
            if oid < 0:
                continue
            # Store as CPU-side floats — avoid retaining GPU tensors.
            box = pred_boxes[i].detach().float().cpu().tolist()
            hist = self._pos_history.setdefault(oid, deque(maxlen=maxlen))
            hist.append(box)

    # ------------------------------------------------------------------
    # (3) Same-frame cross-ID dedup.
    # ------------------------------------------------------------------
    def _same_frame_dedup(self, track_instances: Instances) -> Instances:
        if self.dedup_iou_thresh >= 1.0 or self.dedup_iou_thresh <= 0.0:
            return track_instances
        n = len(track_instances)
        if n < 2:
            return track_instances
        if not hasattr(track_instances, 'pred_boxes'):
            return track_instances

        # Consider only tracks that already have a stable obj_idx AND a
        # reasonable score (avoid suppressing brand-new tentative tracks).
        obj_idxes = track_instances.obj_idxes
        scores = track_instances.scores
        valid = (obj_idxes >= 0) & (scores >= self.motion_score_thresh)
        if int(valid.sum().item()) < 2:
            return track_instances

        idx = torch.nonzero(valid, as_tuple=False).flatten()
        boxes_cxcywh = track_instances.pred_boxes[idx]
        # cxcywh -> xyxy for IoU
        cx, cy, w, h = boxes_cxcywh.unbind(-1)
        x1, y1 = cx - 0.5 * w, cy - 0.5 * h
        x2, y2 = cx + 0.5 * w, cy + 0.5 * h
        xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

        # Pairwise IoU
        A = xyxy[:, None, :]  # [M, 1, 4]
        B = xyxy[None, :, :]  # [1, M, 4]
        inter_x1 = torch.max(A[..., 0], B[..., 0])
        inter_y1 = torch.max(A[..., 1], B[..., 1])
        inter_x2 = torch.min(A[..., 2], B[..., 2])
        inter_y2 = torch.min(A[..., 3], B[..., 3])
        iw = (inter_x2 - inter_x1).clamp_min(0)
        ih = (inter_y2 - inter_y1).clamp_min(0)
        inter = iw * ih
        area_a = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        union = area_a[:, None] + area_a[None, :] - inter
        iou = torch.where(union > 0, inter / union, torch.zeros_like(union))
        iou.fill_diagonal_(0.0)

        thresh = self.dedup_iou_thresh
        # History length per obj_idx (longer = more stable = kept)
        M = idx.shape[0]
        hist_len = torch.zeros(M, device=iou.device)
        for k in range(M):
            oid = int(obj_idxes[idx[k]].item())
            hist_len[k] = len(self._pos_history.get(oid, ()))

        drop_mask_full = torch.zeros(n, dtype=torch.bool,
                                     device=track_instances.pred_boxes.device)
        # Greedy pass — drop the side with shorter history.
        for a in range(M):
            if drop_mask_full[idx[a]]:
                continue
            for b in range(a + 1, M):
                if drop_mask_full[idx[b]]:
                    continue
                if float(iou[a, b].item()) < thresh:
                    continue
                # Tie-break on history length, then on score.
                keep_a = (hist_len[a] > hist_len[b]) or (
                    hist_len[a] == hist_len[b]
                    and float(scores[idx[a]].item())
                    >= float(scores[idx[b]].item()))
                if keep_a:
                    drop_mask_full[idx[b]] = True
                else:
                    drop_mask_full[idx[a]] = True
                    break  # a is gone; move on
        if not bool(drop_mask_full.any()):
            return track_instances
        keep_mask = ~drop_mask_full
        return track_instances[keep_mask]

    # ------------------------------------------------------------------
    # Main entry points used by the inference script
    # ------------------------------------------------------------------
    def forward(self, data: dict):
        # The stability wrapper is strictly inference-only.  If someone
        # calls forward() in training mode we pass through untouched so
        # the existing training path still works.
        return self.base_model(data)

    @torch.no_grad()
    def inference_single_image(
        self,
        img,
        ori_img_size,
        track_instances: Optional[Instances] = None,
        proposals: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        base = self.base_model

        if not isinstance(img, NestedTensor):
            img = nested_tensor_from_tensor_list(img)

        # ---- (2) PRE-Decoder: motion extrapolation on incoming tracks ----
        if track_instances is not None and len(track_instances) > 0:
            self._apply_motion_extrapolation(track_instances)

        if track_instances is None:
            track_instances = base._generate_empty_tracks(proposals)
        else:
            track_instances = Instances.cat([
                base._generate_empty_tracks(proposals),
                track_instances])

        # Backbone + Encoder + Decoder (native)
        frame_res = base._forward_single_image(
            img, track_instances=track_instances)

        # Matching / QIM / track_base.update.
        # (1) Our EMA hook on track_base.update runs inside here, just
        #     before the rebirth/disappear decision is made.
        frame_res = base._post_process_single_image(
            frame_res, track_instances, is_last=False, run_mode='inference')

        track_instances = frame_res['track_instances']

        # ---- (3) POST: same-frame cross-ID dedup ----
        track_instances = self._same_frame_dedup(track_instances)

        # Record history for next-frame motion extrapolation.  Must happen
        # BEFORE the upcoming base.post_process(), because that call mutates
        # boxes into pixel coords; we record normalized pred_boxes instead.
        self._record_position_history(track_instances)

        # Native box-scaling to pixel coords + scores = sigmoid(logits).
        track_instances = base.post_process(track_instances, ori_img_size)

        ret = {'track_instances': track_instances}
        if 'ref_pts' in frame_res:
            ref_pts = frame_res['ref_pts']
            img_h, img_w = ori_img_size
            scale_fct = torch.Tensor([img_w, img_h]).to(ref_pts)
            ret['ref_pts'] = ref_pts * scale_fct[None]
        return ret


# ======================================================================
# Convenience builder — parses the stability CLI flags and wraps the
# already-built base MOTR in one call.
# ======================================================================
def build_stability_model(base_model: nn.Module, args) -> MOTRWithStability:
    return MOTRWithStability(
        base_model,
        score_ema_alpha=getattr(args, 'stab_score_ema_alpha', 0.3),
        ema_mode=getattr(args, 'stab_ema_mode', 'asymmetric_down'),
        motion_k=getattr(args, 'stab_motion_k', 1),
        motion_score_thresh=getattr(args, 'stab_motion_score_thresh', 0.3),
        motion_max_step=getattr(args, 'stab_motion_max_step', 0.03),
        dedup_iou_thresh=getattr(args, 'stab_dedup_iou_thresh', 1.1),
    )
