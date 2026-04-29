# ------------------------------------------------------------------------
# Copyright (c) 2026 JoshuaWenHIT. All Rights Reserved.
# ------------------------------------------------------------------------
import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class UnitrackCriterion(nn.Module):
    """
    UniTrack loss with three components:
    - tracking score loss
    - spatial consistency loss
    - temporal consistency loss
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (1920, 1080),
        iou_threshold: float = 0.5,
        alpha_tracking: float = 2.0,
        alpha_spatial: float = 1.5,
        alpha_temporal: float = 1.8,
        beta_fp: float = 0.9,
        beta_fn: float = 0.9,
        gamma_switch: float = 1.5,
    ):
        super().__init__()
        self.iou_threshold = iou_threshold
        self.img_diagonal = math.sqrt(img_size[0] ** 2 + img_size[1] ** 2)
        self.scale_factor = 1.0 / max(self.img_diagonal, 1.0)

        # Keep scalar weights as plain floats.
        # In DDP, buffers can be broadcast every forward and cause autograd
        # version mismatch when one optimizer step accumulates multiple forwards.
        self.alpha_tracking = float(alpha_tracking)
        self.alpha_spatial = float(alpha_spatial)
        self.alpha_temporal = float(alpha_temporal)
        self.beta_fp = float(beta_fp)
        self.beta_fn = float(beta_fn)
        self.gamma_switch = float(gamma_switch)

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        device = outputs["pred_boxes"].device
        pred_boxes = outputs["pred_boxes"]
        pred_track_ids = outputs["track_ids"]

        loss_tracking = torch.tensor(0.0, device=device)
        loss_spatial = torch.tensor(0.0, device=device)
        loss_temporal = torch.tensor(0.0, device=device)
        num_tracks = 0

        batch_size = pred_boxes.shape[0]
        for b in range(batch_size):
            cur_boxes = pred_boxes[b]
            cur_track_ids = pred_track_ids[b]
            gt_boxes = targets[b]["boxes"]
            gt_track_ids = targets[b]["track_ids"]

            valid_mask = cur_track_ids > 0
            if not valid_mask.any() or gt_boxes.numel() == 0:
                continue

            cur_boxes = cur_boxes[valid_mask]
            cur_track_ids = cur_track_ids[valid_mask]
            ious = self._box_iou(cur_boxes, gt_boxes)

            loss_tracking = loss_tracking + self._compute_tracking_loss(ious, cur_track_ids, gt_track_ids)
            loss_spatial = loss_spatial + self._compute_spatial_consistency(cur_boxes, cur_track_ids)
            loss_temporal = loss_temporal + self._compute_temporal_consistency(cur_boxes, cur_track_ids)
            num_tracks += len(torch.unique(cur_track_ids))

        if num_tracks > 0:
            norm = float(num_tracks)
            loss_tracking = loss_tracking / norm
            loss_spatial = loss_spatial / norm
            loss_temporal = loss_temporal / norm

        total = (
            self.alpha_tracking * loss_tracking
            + self.alpha_spatial * loss_spatial
            + self.alpha_temporal * loss_temporal
        )
        return {
            "loss_unitrack": total,
            "loss_unitrack_tracking": loss_tracking,
            "loss_unitrack_spatial": loss_spatial,
            "loss_unitrack_temporal": loss_temporal,
        }

    def _box_area(self, boxes: torch.Tensor) -> torch.Tensor:
        return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)

    def _box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        area1 = self._box_area(boxes1)
        area2 = self._box_area(boxes2)
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        union = area1[:, None] + area2 - inter
        return inter / union.clamp(min=1e-6)

    def _compute_tracking_loss(
        self,
        ious: torch.Tensor,
        pred_track_ids: torch.Tensor,
        gt_track_ids: torch.Tensor,
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=ious.device)
        unique_gt_tracks = torch.unique(gt_track_ids)

        for gt_tid in unique_gt_tracks:
            gt_mask = gt_track_ids == gt_tid
            gt_indices = torch.where(gt_mask)[0]
            pred_mask = pred_track_ids == gt_tid

            if not pred_mask.any():
                loss = loss + self.beta_fn * len(gt_indices)
                continue

            pred_indices = torch.where(pred_mask)[0]
            track_ious = ious[pred_indices][:, gt_indices]
            max_ious, _ = track_ious.max(dim=1)
            loss = loss + torch.sum(1.0 - max_ious)

            other_pred_mask = ~pred_mask
            if other_pred_mask.any():
                other_pred_ious = ious[other_pred_mask][:, gt_indices]
                wrong_matches = other_pred_ious > self.iou_threshold
                loss = loss + self.gamma_switch * wrong_matches.sum()

        for pred_tid in torch.unique(pred_track_ids):
            if (pred_tid == unique_gt_tracks).any():
                continue
            loss = loss + self.beta_fp * (pred_track_ids == pred_tid).sum()

        return loss

    def _compute_spatial_consistency(self, pred_boxes: torch.Tensor, pred_track_ids: torch.Tensor) -> torch.Tensor:
        if len(pred_boxes) <= 1:
            return torch.tensor(0.0, device=pred_boxes.device)
        loss = torch.tensor(0.0, device=pred_boxes.device)
        unique_tracks = torch.unique(pred_track_ids)
        for tid in unique_tracks:
            boxes = pred_boxes[pred_track_ids == tid]
            if len(boxes) <= 1:
                continue
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            loss = loss + (torch.abs(widths - widths.mean()) * self.scale_factor).mean()
            loss = loss + (torch.abs(heights - heights.mean()) * self.scale_factor).mean()
        return loss / max(len(unique_tracks), 1)

    def _compute_temporal_consistency(self, pred_boxes: torch.Tensor, pred_track_ids: torch.Tensor) -> torch.Tensor:
        if len(pred_boxes) <= 2:
            return torch.tensor(0.0, device=pred_boxes.device)
        loss = torch.tensor(0.0, device=pred_boxes.device)
        unique_tracks = torch.unique(pred_track_ids)
        for tid in unique_tracks:
            boxes = pred_boxes[pred_track_ids == tid]
            if len(boxes) <= 2:
                continue
            centers = torch.stack(
                [(boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5],
                dim=1,
            )
            velocities = centers[1:] - centers[:-1]
            accelerations = (velocities[1:] - velocities[:-1]) * self.scale_factor
            loss = loss + torch.norm(accelerations, dim=1).mean()
        return loss / max(len(unique_tracks), 1)
