# ------------------------------------------------------------------------
# Copyright (c) 2026 JoshuaWenHIT. All Rights Reserved.
# ------------------------------------------------------------------------

"""
SceneAnalyzer — real-time scene characterisation that feeds the Flex-MG
dependency-graph adapter.  All computations are lightweight and run on CPU
tensors so they never add to the GPU compute graph.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from util.box_ops import box_cxcywh_to_xyxy


@dataclass
class SceneInfo:
    target_density: float = 0.0
    occlusion_ratio: float = 0.0
    avg_speed: float = 0.0
    track_conf_mean: float = 0.5
    scene_type: str = 'normal'


class SceneAnalyzer:
    """Stateless analyser — call :meth:`analyze` once per frame."""

    DENSITY_SPARSE = 0.1
    DENSITY_DENSE = 0.5
    SPEED_FAST = 0.05       # normalised coords per frame
    OCCLUSION_HIGH = 0.4

    def __init__(self, img_w: int = 1920, img_h: int = 1080):
        self.img_w = img_w
        self.img_h = img_h

    # ------------------------------------------------------------------
    @torch.no_grad()
    def analyze(
        self,
        pred_boxes: Optional[Tensor],
        scores: Optional[Tensor],
        prev_boxes: Optional[Tensor] = None,
    ) -> SceneInfo:
        """
        Args:
            pred_boxes: [N, 4] in cxcywh normalised format.
            scores:     [N] confidence scores.
            prev_boxes: [N, 4] from the previous frame (for speed estimate).
        """
        if pred_boxes is None or pred_boxes.numel() == 0:
            return SceneInfo()

        N = pred_boxes.shape[0]
        density = N / (self.img_w * self.img_h) * 1e6  # targets / mega-pixel

        # occlusion estimate via pairwise IoU
        xyxy = box_cxcywh_to_xyxy(pred_boxes)
        occlusion = self._estimate_occlusion(xyxy)

        # speed from displacement of box centres
        avg_speed = 0.0
        if prev_boxes is not None and prev_boxes.shape[0] == N:
            disp = (pred_boxes[:, :2] - prev_boxes[:, :2]).norm(dim=-1)
            avg_speed = disp.mean().item()

        conf_mean = scores.mean().item() if scores is not None else 0.5

        scene_type = self._classify(density, occlusion, avg_speed)
        return SceneInfo(
            target_density=density,
            occlusion_ratio=occlusion,
            avg_speed=avg_speed,
            track_conf_mean=conf_mean,
            scene_type=scene_type,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _estimate_occlusion(xyxy: Tensor) -> float:
        """Fraction of box pairs whose IoU > 0.3."""
        N = xyxy.shape[0]
        if N < 2:
            return 0.0
        lt = torch.max(xyxy[:, None, :2], xyxy[None, :, :2])
        rb = torch.min(xyxy[:, None, 2:], xyxy[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        union = area[:, None] + area[None, :] - inter
        iou = inter / union.clamp(min=1e-6)
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=xyxy.device), diagonal=1)
        num_pairs = mask.sum().item()
        if num_pairs == 0:
            return 0.0
        return float((iou[mask] > 0.3).sum().item() / num_pairs)

    def _classify(self, density: float, occlusion: float, speed: float) -> str:
        if occlusion > self.OCCLUSION_HIGH:
            return 'occluded'
        if speed > self.SPEED_FAST:
            return 'fast'
        if density > self.DENSITY_DENSE:
            return 'dense'
        if density < self.DENSITY_SPARSE:
            return 'sparse'
        return 'normal'
