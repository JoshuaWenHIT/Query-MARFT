# ------------------------------------------------------------------------
# Copyright (c) 2026 JoshuaWenHIT. All Rights Reserved.
# ------------------------------------------------------------------------

"""
BaseAgent — abstract parent for all role-based agents in Query-MARFT.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class ActionInfo:
    """Container returned alongside the action tensor."""
    log_prob: Optional[Tensor] = None
    value: Optional[Tensor] = None
    entropy: Optional[Tensor] = None
    extra: Optional[Dict[str, Any]] = None


class BaseAgent(ABC, nn.Module):
    """Abstract base class shared by all four Query-MARFT agents."""

    def __init__(self, hidden_dim: int, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = config or {}
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, flag: bool):
        self._enabled = flag

    @abstractmethod
    def forward(
        self,
        obs: Dict[str, Tensor],
        hidden_state: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], ActionInfo]:
        """
        Args:
            obs: observation dict built by AgentManager for this agent.
            hidden_state: optional RNN-style hidden state carried across frames.
        Returns:
            action, new_hidden_state, action_info
        """
        ...

    @abstractmethod
    def get_deterministic_action(self, obs: Dict[str, Tensor]) -> Tensor:
        """Deterministic (greedy) action for evaluation."""
        ...
