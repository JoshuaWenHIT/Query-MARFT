# ------------------------------------------------------------------------
# Copyright (c) 2026 JoshuaWenHIT. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Association Agent — modulates the matching confidence between detections
and historical track queries.

The agent outputs a per-query modulation factor α ∈ [0.5, 1.5] that scales
the raw association score: final_score = raw_score × α.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .base_agent import BaseAgent, ActionInfo


class AssocAgent(BaseAgent):
    """
    Input: concatenation of detection embedding and track-query embedding.
    Output: per-pair modulation factor α.
    """

    def __init__(self, hidden_dim: int, config: Optional[Dict[str, Any]] = None):
        super().__init__(hidden_dim, config)
        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.alpha_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # bias last layer to output ~0 → sigmoid(0)=0.5, scaled to α=1.0
        nn.init.zeros_(self.alpha_head[-1].bias)

    def forward(
        self,
        obs: Dict[str, Tensor],
        hidden_state: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], ActionInfo]:
        """
        obs keys:
            'det_embed':   [N, D]  current frame detection output embeddings
            'track_embed': [N, D]  historical track query embeddings
        Returns:
            alpha: [N, 1]  modulation factor in [0.5, 1.5]
        """
        det_embed = obs['det_embed']      # [N, D]
        track_embed = obs['track_embed']  # [N, D]
        cat_feat = torch.cat([det_embed, track_embed], dim=-1)  # [N, 2D]

        raw = self.alpha_head(cat_feat)    # [N, 1]
        alpha = torch.sigmoid(raw) + 0.5  # range [0.5, 1.5]

        if self.training:
            noise = torch.randn_like(raw) * 0.01
            alpha_noisy = torch.sigmoid(raw + noise) + 0.5
            log_prob = -0.5 * (noise ** 2).sum(dim=-1, keepdim=True)
            info = ActionInfo(log_prob=log_prob.squeeze(-1))
            return alpha_noisy, hidden_state, info

        info = ActionInfo(log_prob=torch.zeros(alpha.shape[0], device=alpha.device))
        return alpha, hidden_state, info

    def get_deterministic_action(self, obs: Dict[str, Tensor]) -> Tensor:
        det_embed = obs['det_embed']
        track_embed = obs['track_embed']
        cat_feat = torch.cat([det_embed, track_embed], dim=-1)
        raw = self.alpha_head(cat_feat)
        return torch.sigmoid(raw) + 0.5
