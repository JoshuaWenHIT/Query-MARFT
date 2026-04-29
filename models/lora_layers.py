# ------------------------------------------------------------------------
# Copyright (c) 2026 JoshuaWenHIT. All Rights Reserved.
# ------------------------------------------------------------------------

"""
LoRA (Low-Rank Adaptation) layer implementations for Query-MARFT.

Original:  y = W·x
LoRA:      y = W·x + (B·A)·x       where A ∈ R^{r×d_in}, B ∈ R^{d_out×r}
           W is frozen; only A, B are trained.

A is initialised with small Gaussian noise, B with zeros so that the
injected path is a no-op at initialisation.

Important compatibility notes for MOTRv2:
  * ``nn.MultiheadAttention`` internally calls
    ``F.multi_head_attention_forward(..., self.out_proj.weight, ...)``,
    accessing ``.weight`` directly instead of going through the child
    module's ``forward()``.  Wrapping its internal Linear layers with
    LoRALinear therefore **breaks** MultiheadAttention.  We explicitly
    skip any Linear whose parent is an ``nn.MultiheadAttention`` (or
    subclass).
  * MOTRv2 uses ``MSDeformAttn`` (custom deformable attention) and
    ``nn.MultiheadAttention``.  Only MSDeformAttn's Linear submodules
    (``value_proj``, ``output_proj``, ``sampling_offsets``,
    ``attention_weights``) are safe targets.  MultiheadAttention's
    ``out_proj`` is NOT.
"""

from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class LoRALinear(nn.Module):
    """Drop-in LoRA wrapper around an existing ``nn.Linear``."""

    def __init__(
        self,
        original_linear: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert isinstance(original_linear, nn.Linear)
        self.original_linear = original_linear
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        self.r = r
        self.scaling = alpha / r

        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        for p in original_linear.parameters():
            p.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        original_out = self.original_linear(x)
        lora_out = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        return original_out + lora_out * self.scaling

    def extra_repr(self) -> str:
        return (f"r={self.r}, scaling={self.scaling:.2f}, "
                f"in={self.original_linear.in_features}, "
                f"out={self.original_linear.out_features}")


# ======================================================================
# Strategy config — targets are matched against the short name of each
# Linear submodule.  These names are the ones that ACTUALLY EXIST in
# MOTRv2 (verified by inspecting MSDeformAttn, QIMv2, and MOTR heads).
# ======================================================================
DEFAULT_LORA_STRATEGY: Dict[str, Dict] = {
    # Backbone is pure Conv2d (ResNet); no Linear to inject.  Kept as a
    # no-op entry for forward compatibility.
    'backbone': dict(apply=True, r=8, alpha=16, dropout=0.05,
                     targets={'value_proj', 'output_proj'}),
    # transformer.encoder.*  — MSDeformAttn + FFN Linear layers
    'transformer.encoder': dict(apply=True, r=16, alpha=32, dropout=0.05,
                                targets={'value_proj', 'output_proj',
                                         'linear1', 'linear2'}),
    # transformer.decoder.*  — MSDeformAttn (cross-attn) + FFN
    # NOTE: self_attn (nn.MultiheadAttention) children are filtered out
    # automatically by `_is_child_of_multihead_attn`.
    'transformer.decoder': dict(apply=True, r=24, alpha=48, dropout=0.1,
                                targets={'value_proj', 'output_proj',
                                         'linear1', 'linear2'}),
    # Heads on top of Track Query output embeddings (class / bbox)
    'class_embed': dict(apply=True, r=24, alpha=48, dropout=0.1,
                        targets={'*'}),
    'bbox_embed': dict(apply=True, r=24, alpha=48, dropout=0.1,
                       targets={'*'}),
    # QIMv2 (track_embed) — Linear layers that update Track Query state
    'track_embed': dict(apply=True, r=16, alpha=32, dropout=0.1,
                        targets={'linear1', 'linear2', 'linear_feat1',
                                 'linear_feat2', 'linear_pos1', 'linear_pos2'}),
}


# ======================================================================
# Injection helpers
# ======================================================================
def inject_lora(
    model: nn.Module,
    strategy: Optional[Dict[str, Dict]] = None,
    verbose: bool = False,
) -> nn.ModuleDict:
    """
    Walk *model*, find ``nn.Linear`` layers that match the strategy, and
    replace them in-place with ``LoRALinear`` wrappers.

    Any Linear whose *parent* is an ``nn.MultiheadAttention`` (or subclass)
    is unconditionally skipped to avoid the known ``.weight`` AttributeError
    (see module docstring).
    """
    strategy = strategy or DEFAULT_LORA_STRATEGY
    lora_modules = nn.ModuleDict()
    n_injected, n_skipped_mha = 0, 0

    # Collect first so we don't mutate the tree while walking.
    candidates: List[Tuple[str, nn.Linear]] = [
        (name, m) for name, m in model.named_modules()
        if isinstance(m, nn.Linear)
    ]

    for full_name, module in candidates:
        cfg = _match_strategy(full_name, strategy)
        if cfg is None or not cfg.get('apply', False):
            continue

        targets: Set[str] = cfg.get('targets', {'*'})
        short_name = full_name.rsplit('.', 1)[-1] if '.' in full_name else full_name
        if '*' not in targets and short_name not in targets:
            continue

        # Critical safety filter: skip Linear children of MultiheadAttention.
        parent, attr = _get_parent_attr(model, full_name)
        if _is_child_of_multihead_attn(parent):
            n_skipped_mha += 1
            if verbose:
                print(f'[LoRA] SKIP (MultiheadAttention child): {full_name}')
            continue

        lora_layer = LoRALinear(
            module,
            r=cfg.get('r', 8),
            alpha=cfg.get('alpha', 16),
            dropout=cfg.get('dropout', 0.0),
        )
        setattr(parent, attr, lora_layer)

        safe_key = full_name.replace('.', '_')
        lora_modules[safe_key] = lora_layer
        n_injected += 1
        if verbose:
            print(f'[LoRA] INJECT: {full_name}  (r={cfg["r"]}, α={cfg["alpha"]})')

    print(f'[LoRA] Injected into {n_injected} Linear layers '
          f'(skipped {n_skipped_mha} MultiheadAttention children).')
    return lora_modules


def _match_strategy(name: str, strategy: Dict[str, Dict]) -> Optional[Dict]:
    """Return the *longest-prefix-match* strategy entry for a module path.

    Using longest-match avoids ``'transformer.encoder'`` being accidentally
    swallowed by a generic ``'encoder'`` key (or vice-versa).
    """
    best_key, best_len = None, -1
    for key in strategy:
        if key in name and len(key) > best_len:
            best_key, best_len = key, len(key)
    return strategy[best_key] if best_key is not None else None


def _get_parent_attr(root: nn.Module, dotted_name: str) -> Tuple[nn.Module, str]:
    """Return (parent_module, attribute_name) for a dotted module path."""
    parts = dotted_name.split('.')
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _is_child_of_multihead_attn(parent: nn.Module) -> bool:
    """
    Return True if *parent* is an ``nn.MultiheadAttention`` (or subclass).

    ``nn.MultiheadAttention`` calls ``F.multi_head_attention_forward`` which
    accesses ``out_proj.weight`` directly, bypassing our ``LoRALinear.forward``.
    Wrapping its children therefore raises ``AttributeError: 'LoRALinear'
    object has no attribute 'weight'`` at runtime.
    """
    return isinstance(parent, nn.MultiheadAttention)


# ======================================================================
# Utility: print parameter statistics
# ======================================================================
def print_lora_param_stats(model: nn.Module, tag: str = ''):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    pct = trainable / total * 100 if total else 0
    prefix = f'[{tag}] ' if tag else ''
    print(f'{prefix}Parameters: total={total:,}  trainable={trainable:,}  '
          f'frozen={frozen:,}  fine-tune={pct:.2f}%')
