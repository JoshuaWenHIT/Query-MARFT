# ------------------------------------------------------------------------
# Copyright (c) 2026 JoshuaWenHIT. All Rights Reserved.
# ------------------------------------------------------------------------

"""
AMP compatibility patches for MOTRv2 original CUDA ops.

Problem
-------
The ``MSDeformAttn`` CUDA kernel under ``models/ops/`` is compiled for
``float32`` only.  Running it inside ``torch.cuda.amp.autocast()`` raises::

    RuntimeError: "ms_deform_attn_forward_cuda" not implemented for 'Half'

Fix
---
PyTorch provides ``torch.cuda.amp.custom_fwd / custom_bwd`` decorators
that tell autocast to cast specific ``autograd.Function`` inputs back to
a chosen dtype (here fp32), run the op, then cast outputs per autocast
rules.

Rather than editing the original ``ms_deform_attn_func.py`` (which would
violate the "never modify MOTRv2 native code" principle), this module
applies the decorators to ``MSDeformAttnFunction`` at runtime via class
attribute reassignment.  The patch is idempotent — safe to call
multiple times.

Usage
-----
Import and call :func:`apply_msdeform_amp_patch` once before any forward
pass that may run under ``autocast``.  This module is auto-imported by
``models/motr_marft.py`` so MARFT training picks it up transparently.
"""

from __future__ import annotations

import functools
from typing import Any, Dict

import torch


_PATCHED = False
_POSTPROC_PATCHED = False


def apply_msdeform_amp_patch(verbose: bool = True) -> bool:
    """Patch ``MSDeformAttnFunction`` with ``custom_fwd`` / ``custom_bwd``.

    Returns True if the patch was applied (or was already applied);
    False if patching could not happen (e.g. import failure).
    """
    global _PATCHED
    if _PATCHED:
        return True

    try:
        from torch.cuda.amp import custom_fwd, custom_bwd
    except ImportError:
        if verbose:
            print('[amp_patches] torch.cuda.amp.custom_fwd unavailable — '
                  'patch skipped.')
        return False

    try:
        from models.ops.functions.ms_deform_attn_func import (
            MSDeformAttnFunction,
        )
    except ImportError as exc:
        if verbose:
            print(f'[amp_patches] MSDeformAttnFunction import failed: {exc}')
        return False

    # Access underlying callables (staticmethod auto-unwrapped via class).
    orig_fwd = MSDeformAttnFunction.forward
    orig_bwd = MSDeformAttnFunction.backward

    wrapped_fwd = custom_fwd(cast_inputs=torch.float32)(orig_fwd)
    wrapped_bwd = custom_bwd(orig_bwd)

    # Re-wrap in staticmethod so torch.autograd.Function conventions hold.
    MSDeformAttnFunction.forward = staticmethod(wrapped_fwd)
    MSDeformAttnFunction.backward = staticmethod(wrapped_bwd)

    _PATCHED = True
    if verbose:
        print('[amp_patches] Patched MSDeformAttnFunction for AMP '
              '(fp32 cast_inputs).')
    return True


def apply_post_process_amp_patch(verbose: bool = True) -> bool:
    """Patch ``MOTR._post_process_single_image`` for AMP safety.

    Problem
    -------
    Under ``autocast``, the Decoder outputs (``pred_logits``, ``pred_boxes``,
    ``hs``) are fp16, but long-lived state tensors inside ``track_instances``
    (``ref_pts`` from ``nn.Embedding.weight``, ``query_pos``, etc.) remain
    fp32.  Operations that require strict dtype match (indexed assignment in
    QIMv2, Hungarian matching cost, memory bank updates) then crash with::

        RuntimeError: Index put requires the source and destination dtypes
        match, got Float for the destination and Half for the source.

    Fix
    ---
    Wrap ``MOTR._post_process_single_image`` so that:
      1. Fp16 activations in ``frame_res`` are upcast to fp32 at entry.
      2. The body runs under ``autocast(enabled=False)`` — all downstream
         ops (ClipMatcher, QIMv2, MemoryBank, Bernoulli sampling) see
         consistent fp32 tensors.

    Backbone / Encoder / Decoder forward still enjoy the fp16 speedup;
    only the numerically-sensitive post-processing is forced to fp32.
    """
    global _POSTPROC_PATCHED
    if _POSTPROC_PATCHED:
        return True

    try:
        from models.motr import MOTR
    except ImportError as exc:
        if verbose:
            print(f'[amp_patches] MOTR import failed: {exc}')
        return False

    original = MOTR._post_process_single_image

    def _upcast_fp16(d: Dict[str, Any]) -> None:
        """In-place upcast of fp16 tensors in ``frame_res`` to fp32."""
        for k in ('pred_logits', 'pred_boxes', 'hs'):
            v = d.get(k)
            if isinstance(v, torch.Tensor) and v.dtype == torch.float16:
                d[k] = v.float()
        if 'aux_outputs' in d and isinstance(d['aux_outputs'], list):
            for aux in d['aux_outputs']:
                for k in ('pred_logits', 'pred_boxes'):
                    v = aux.get(k)
                    if isinstance(v, torch.Tensor) and v.dtype == torch.float16:
                        aux[k] = v.float()
        if 'ps_outputs' in d and isinstance(d['ps_outputs'], list):
            for ps in d['ps_outputs']:
                for k in ('pred_logits', 'pred_boxes'):
                    v = ps.get(k)
                    if isinstance(v, torch.Tensor) and v.dtype == torch.float16:
                        ps[k] = v.float()

    @functools.wraps(original)
    def wrapped(self, frame_res, track_instances, is_last, run_mode='supervised'):
        _upcast_fp16(frame_res)
        with torch.cuda.amp.autocast(enabled=False):
            return original(self, frame_res, track_instances, is_last,
                            run_mode=run_mode)

    MOTR._post_process_single_image = wrapped

    _POSTPROC_PATCHED = True
    if verbose:
        print('[amp_patches] Patched MOTR._post_process_single_image for '
              'AMP (fp32 post-processing).')
    return True
