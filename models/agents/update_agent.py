# ------------------------------------------------------------------------
# Copyright (c) 2026 JoshuaWenHIT. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Update Agent — adaptively controls the temporal update of track query
hidden states across frames.

Original MOTRv2: h_t = QIM(h_{t-1}, current_content)
With UpdateAgent: h_t = α · QIM(h_{t-1}, current_content) + β · current_content

α and β are dynamically predicted per query per frame.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .base_agent import BaseAgent, ActionInfo


class UpdateAgent(BaseAgent):
    """
    Takes the pre-update and post-update query embeddings as input and
    outputs gating parameters (α, β) controlling the update mixture.
    """

    def __init__(self, hidden_dim: int, config: Optional[Dict[str, Any]] = None):
        super().__init__(hidden_dim, config)
        self.gate_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 2),  # outputs (α_raw, β_raw)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.gate_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # initialise so that sigmoid → 0.5, giving α=β≈0.5 at start
        nn.init.zeros_(self.gate_head[-1].bias)

    def forward(
        self,
        obs: Dict[str, Tensor],
        hidden_state: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], ActionInfo]:
        """
        obs keys:
            'pre_update_embed':  [N, D] query embedding before QIM update
            'post_update_embed': [N, D] query embedding after QIM update
        Returns:
            gates: [N, 2]  where gates[:, 0]=α and gates[:, 1]=β, both in (0, 1)
        """
        pre = obs['pre_update_embed']
        post = obs['post_update_embed']
        cat_feat = torch.cat([pre, post], dim=-1)  # [N, 2D]

        raw = self.gate_head(cat_feat)              # [N, 2]
        gates = torch.sigmoid(raw)                  # (α, β) ∈ (0,1)

        if self.training:
            noise = torch.randn_like(raw) * 0.01
            gates_noisy = torch.sigmoid(raw + noise)
            log_prob = -0.5 * (noise ** 2).sum(dim=-1)
            info = ActionInfo(log_prob=log_prob)
            return gates_noisy, hidden_state, info

        info = ActionInfo(log_prob=torch.zeros(gates.shape[0], device=gates.device))
        return gates, hidden_state, info

    def get_deterministic_action(self, obs: Dict[str, Tensor]) -> Tensor:
        pre = obs['pre_update_embed']
        post = obs['post_update_embed']
        cat_feat = torch.cat([pre, post], dim=-1)
        return torch.sigmoid(self.gate_head(cat_feat))

    @staticmethod
    def apply_gates(qim_output: Tensor, current_content: Tensor, gates: Tensor) -> Tensor:
        """
        Merge QIM output and raw content using agent-predicted gates.
            h_t = α · qim_output + β · current_content
        """
        alpha = gates[:, 0:1]  # [N, 1]
        beta = gates[:, 1:2]   # [N, 1]
        return alpha * qim_output + beta * current_content
