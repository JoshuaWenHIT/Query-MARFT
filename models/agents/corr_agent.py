# ------------------------------------------------------------------------
# Copyright (c) 2026 JoshuaWenHIT. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Correction Agent — performs adaptive correction on low-confidence tracks.

Discrete action space: {keep, interpolate-recover, terminate}.
Applied as a post-processing step on tracks whose confidence falls below
a configurable threshold.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base_agent import BaseAgent, ActionInfo

# Action indices
ACTION_KEEP = 0
ACTION_RECOVER = 1
ACTION_TERMINATE = 2
NUM_ACTIONS = 3


class CorrAgent(BaseAgent):
    """
    Classification network over low-confidence track features + global context.
    Outputs a 3-class probability for each candidate track.
    """

    def __init__(self, hidden_dim: int, config: Optional[Dict[str, Any]] = None):
        super().__init__(hidden_dim, config)
        self.conf_threshold = self.config.get('corr_conf_threshold', 0.4)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, NUM_ACTIONS),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.classifier:
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
            'track_embed':   [N, D]  embeddings of low-confidence tracks
            'global_context': [D]    mean-pooled scene context vector
            'scores':         [N]    track confidence scores (used for masking)
        Returns:
            action: [N] integer actions (0=keep, 1=recover, 2=terminate)
        """
        track_embed = obs['track_embed']     # [N, D]
        global_ctx = obs['global_context']   # [D]
        scores = obs['scores']               # [N]

        N = track_embed.shape[0]
        if N == 0:
            empty = torch.empty(0, dtype=torch.long, device=track_embed.device)
            info = ActionInfo(log_prob=torch.empty(0, device=track_embed.device))
            return empty, hidden_state, info

        low_conf_mask = scores < self.conf_threshold
        if not low_conf_mask.any():
            actions = torch.full((N,), ACTION_KEEP, dtype=torch.long,
                                 device=track_embed.device)
            info = ActionInfo(log_prob=torch.zeros(N, device=track_embed.device))
            return actions, hidden_state, info

        ctx_expanded = global_ctx.unsqueeze(0).expand(N, -1)  # [N, D]
        feat = torch.cat([track_embed, ctx_expanded], dim=-1)  # [N, 2D]
        logits = self.classifier(feat)  # [N, 3]

        actions = torch.full((N,), ACTION_KEEP, dtype=torch.long,
                             device=track_embed.device)

        if self.training:
            probs = F.softmax(logits[low_conf_mask], dim=-1)
            dist = torch.distributions.Categorical(probs)
            sampled = dist.sample()
            log_prob_all = torch.zeros(N, device=track_embed.device)
            log_prob_all[low_conf_mask] = dist.log_prob(sampled)
            actions[low_conf_mask] = sampled
            info = ActionInfo(
                log_prob=log_prob_all,
                entropy=torch.zeros(N, device=track_embed.device),
            )
            ent = torch.zeros(N, device=track_embed.device)
            ent[low_conf_mask] = dist.entropy()
            info.entropy = ent
        else:
            actions[low_conf_mask] = logits[low_conf_mask].argmax(dim=-1)
            info = ActionInfo(log_prob=torch.zeros(N, device=track_embed.device))

        return actions, hidden_state, info

    def get_deterministic_action(self, obs: Dict[str, Tensor]) -> Tensor:
        track_embed = obs['track_embed']
        global_ctx = obs['global_context']
        scores = obs['scores']
        N = track_embed.shape[0]
        if N == 0:
            return torch.empty(0, dtype=torch.long, device=track_embed.device)

        actions = torch.full((N,), ACTION_KEEP, dtype=torch.long,
                             device=track_embed.device)
        low_conf_mask = scores < self.conf_threshold
        if not low_conf_mask.any():
            return actions

        ctx_expanded = global_ctx.unsqueeze(0).expand(N, -1)
        feat = torch.cat([track_embed, ctx_expanded], dim=-1)
        logits = self.classifier(feat)
        actions[low_conf_mask] = logits[low_conf_mask].argmax(dim=-1)
        return actions
