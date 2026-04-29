# ------------------------------------------------------------------------
# Copyright (c) 2026 JoshuaWenHIT. All Rights Reserved.
# ------------------------------------------------------------------------

"""
AgentManager — orchestrates the four role-based agents in Query-MARFT.

Execution follows the DAG topological order:
    DetAgent → AssocAgent → UpdateAgent → CorrAgent

Each subsequent agent may observe the intermediate results produced by
all preceding agents.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .det_agent import DetAgent
from .assoc_agent import AssocAgent
from .update_agent import UpdateAgent
from .corr_agent import CorrAgent
from .base_agent import ActionInfo


class AgentManager(nn.Module):
    """
    Manages the four agents' lifecycle, parameter collection, and
    sequential execution within a single frame.
    """

    EXEC_ORDER: List[str] = ['det', 'assoc', 'update', 'corr']

    def __init__(self, hidden_dim: int, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        config = config or {}
        self.hidden_dim = hidden_dim
        # Four agents — assigning as attributes automatically registers
        # them as submodules; their parameters are tracked by PyTorch.
        self.det_agent = DetAgent(hidden_dim, config.get('det'))
        self.assoc_agent = AssocAgent(hidden_dim, config.get('assoc'))
        self.update_agent = UpdateAgent(hidden_dim, config.get('update'))
        self.corr_agent = CorrAgent(hidden_dim, config.get('corr'))

        # NOTE: plain dict, NOT nn.ModuleDict — using key 'update' in a
        # ModuleDict would collide with its own .update() method and raise
        # `KeyError: "attribute 'update' already exists"`.  Parameter
        # registration is already handled via the attribute assignments
        # above, so a regular dict is safe and sufficient for lookup.
        self._agent_map: Dict[str, nn.Module] = {
            'det': self.det_agent,
            'assoc': self.assoc_agent,
            'update': self.update_agent,
            'corr': self.corr_agent,
        }

    def get_agent(self, name: str):
        return self._agent_map[name]

    def set_agent_enabled(self, name: str, flag: bool):
        self._agent_map[name].set_enabled(flag)

    # ------------------------------------------------------------------
    # Core forward — run the four-agent pipeline on a single frame
    # ------------------------------------------------------------------
    def forward(
        self,
        frame_features: Dict[str, Tensor],
        track_queries: Dict[str, Tensor],
        track_states: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
        """
        Execute all enabled agents in topological order.

        Args:
            frame_features: dict with at least 'query_embed' [N,D] and
                            'output_embedding' [N,D].
            track_queries:  dict with 'ref_pts' [N,2/4], 'scores' [N],
                            'pred_boxes' [N,4], etc.
            track_states:   optional per-agent hidden states carried across
                            frames.
        Returns:
            updated_queries: dict mirroring track_queries with agent
                             modifications applied.
            agent_outputs:   dict of per-agent action/info for reward
                             computation and logging.
        """
        track_states = track_states or {}
        intermediate: Dict[str, Any] = {}
        joint_actions: Dict[str, Tensor] = {}
        joint_infos: Dict[str, ActionInfo] = {}

        updated_queries = {k: v.clone() if isinstance(v, Tensor) else v
                          for k, v in track_queries.items()}
        query_embed = frame_features.get('query_embed',
                                         frame_features.get('output_embedding'))
        N = query_embed.shape[0]

        # --- DetAgent ---
        if self.det_agent.enabled:
            det_obs = {
                'query_embed': query_embed,
                'ref_pts': updated_queries['ref_pts'],
            }
            delta_p, det_h, det_info = self.det_agent(
                det_obs, track_states.get('det'))
            updated_queries['ref_pts'] = updated_queries['ref_pts'].clone()
            updated_queries['ref_pts'][:, :2] = (
                updated_queries['ref_pts'][:, :2] + delta_p
            ).clamp(0.0, 1.0)
            joint_actions['det'] = delta_p
            joint_infos['det'] = det_info
            intermediate['det'] = {'delta_p': delta_p}

        # --- AssocAgent ---
        if self.assoc_agent.enabled:
            det_embed = frame_features.get('output_embedding', query_embed)
            assoc_obs = {
                'det_embed': det_embed,
                'track_embed': query_embed,
            }
            alpha, assoc_h, assoc_info = self.assoc_agent(
                assoc_obs, track_states.get('assoc'))
            if 'scores' in updated_queries:
                updated_queries['scores'] = updated_queries['scores'] * alpha.squeeze(-1)
            joint_actions['assoc'] = alpha
            joint_infos['assoc'] = assoc_info
            intermediate['assoc'] = {'alpha': alpha}

        # --- UpdateAgent ---
        if self.update_agent.enabled:
            pre_embed = query_embed
            post_embed = frame_features.get('output_embedding', query_embed)
            update_obs = {
                'pre_update_embed': pre_embed,
                'post_update_embed': post_embed,
            }
            gates, upd_h, upd_info = self.update_agent(
                update_obs, track_states.get('update'))
            joint_actions['update'] = gates
            joint_infos['update'] = upd_info
            intermediate['update'] = {'gates': gates}

        # --- CorrAgent ---
        if self.corr_agent.enabled:
            global_ctx = query_embed.mean(dim=0)
            scores = updated_queries.get(
                'scores', torch.ones(N, device=query_embed.device) * 0.5)
            corr_obs = {
                'track_embed': query_embed,
                'global_context': global_ctx,
                'scores': scores,
            }
            corr_actions, corr_h, corr_info = self.corr_agent(
                corr_obs, track_states.get('corr'))
            joint_actions['corr'] = corr_actions
            joint_infos['corr'] = corr_info
            intermediate['corr'] = {'actions': corr_actions}

        return updated_queries, {
            'joint_actions': joint_actions,
            'joint_infos': joint_infos,
            'intermediate': intermediate,
        }

    def get_agent_params_by_name(self, name: str):
        return list(self._agent_map[name].parameters())

    def get_all_agent_params(self):
        params = []
        for name in self.EXEC_ORDER:
            params.extend(self.get_agent_params_by_name(name))
        return params
