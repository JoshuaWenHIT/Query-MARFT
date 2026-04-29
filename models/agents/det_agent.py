# ------------------------------------------------------------------------
# Copyright (c) 2026 JoshuaWenHIT. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Detection Agent — controls query attention focus via sampling-offset deltas.

Integration point: applied *before* the decoder cross-attention in the
Deformable DETR pipeline. The agent outputs Δp offsets that are added to the
original reference points so that queries attend to more informative regions.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .base_agent import BaseAgent, ActionInfo


class DetAgent(BaseAgent):
    """
    Lightweight MLP that outputs per-query spatial offset Δp ∈ R^{N×2}
    added to the reference points before decoder feature sampling.
    """

    def __init__(self, hidden_dim: int, config: Optional[Dict[str, Any]] = None):
        super().__init__(hidden_dim, config)
        self.offset_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 2),
        )
        # scale factor keeps initial offsets near zero
        self.offset_scale = nn.Parameter(torch.tensor(0.01))
        self._init_weights()

    def _init_weights(self):
        for m in self.offset_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        obs: Dict[str, Tensor],
        hidden_state: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], ActionInfo]:
        """
        obs keys:
            'query_embed': [N, D]  — current track query embeddings
            'ref_pts':     [N, 2/4] — current reference points
        Returns:
            delta_p: [N, 2] offset to add to ref_pts[:, :2]
        """
        query_embed = obs['query_embed']  # [N, D]
        delta_p = self.offset_head(query_embed) * self.offset_scale  # [N, 2]

        if self.training:
            noise = torch.randn_like(delta_p) * 0.005
            delta_p = delta_p + noise
            log_prob = -0.5 * (noise ** 2).sum(dim=-1)
        else:
            log_prob = torch.zeros(delta_p.shape[0], device=delta_p.device)

        info = ActionInfo(log_prob=log_prob)
        return delta_p, hidden_state, info

    def get_deterministic_action(self, obs: Dict[str, Tensor]) -> Tensor:
        query_embed = obs['query_embed']
        return self.offset_head(query_embed) * self.offset_scale
