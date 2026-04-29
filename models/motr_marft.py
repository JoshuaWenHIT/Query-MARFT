# ------------------------------------------------------------------------
# Copyright (c) 2026 JoshuaWenHIT. All Rights Reserved.
# ------------------------------------------------------------------------

"""
MOTRWithMARFT — incremental wrapper around the *unmodified* MOTR model.

Design principles:
  * The original MOTR class is **never edited**.  This module wraps it,
    injects LoRA adapters into its Linear layers, attaches an
    AgentManager, and overrides ``forward()`` to insert multi-agent
    decision logic at the correct integration points.
  * When ``use_agents=False`` the forward path is numerically identical
    to the original MOTR.
  * Supports toggling individual agents on/off for ablation studies.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from util.misc import NestedTensor, nested_tensor_from_tensor_list
from models.structures import Instances

from .agents import AgentManager
from .agents.update_agent import UpdateAgent
from .agents.corr_agent import ACTION_KEEP, ACTION_RECOVER, ACTION_TERMINATE
from .lora_layers import inject_lora, print_lora_param_stats, DEFAULT_LORA_STRATEGY

# Apply AMP-compatibility patches:
#   1. MSDeformAttn CUDA op -> force fp32 inputs (kernel has no fp16).
#   2. MOTR._post_process_single_image -> upcast fp16 activations to fp32
#      and disable autocast, so ClipMatcher / QIMv2 / MemoryBank never
#      mix fp16 Decoder outputs with fp32 Embedding-derived state.
# Neither patch modifies any original MOTRv2 source file.
from .amp_patches import (
    apply_msdeform_amp_patch as _apply_msdeform_patch,
    apply_post_process_amp_patch as _apply_postproc_patch,
)
_apply_msdeform_patch()
_apply_postproc_patch()


class MOTRWithMARFT(nn.Module):
    """
    Thin wrapper: ``self.base_model`` is the original MOTR instance.
    New trainable components live as siblings so that ``state_dict()``
    cleanly separates MARFT parameters from MOTR parameters.
    """

    def __init__(
        self,
        base_model: nn.Module,
        hidden_dim: int = 256,
        use_lora: bool = True,
        lora_strategy: Optional[Dict] = None,
        use_agents: bool = True,
        agent_config: Optional[Dict[str, Any]] = None,
        infer_safety: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.hidden_dim = hidden_dim
        self.use_agents = use_agents
        self.use_lora = use_lora

        # 1. Freeze all original parameters
        for p in self.base_model.parameters():
            p.requires_grad = False

        # 2. Inject LoRA adapters (unfreezes only the A/B matrices)
        self.lora_modules = nn.ModuleDict()
        if use_lora:
            self.lora_modules = inject_lora(
                self.base_model, lora_strategy or DEFAULT_LORA_STRATEGY)

        # 3. Four-agent module (fully trainable, no LoRA)
        self.agent_manager = AgentManager(hidden_dim, agent_config)

        # 4. Inference-time safety knobs.  Defaults reproduce the original
        #    (pre-patch) agent behaviour exactly so training loops are
        #    unaffected.  At inference the scripts can dial these down to
        #    stabilise tracking when agents are under-trained, which is the
        #    root cause of ID explosion in MARFT inference.
        #
        # Keys:
        #   det_delta_scale       — multiplier on DetAgent Δp (0 = no ref_pt perturbation)
        #   assoc_alpha_gamma     — blend factor for AssocAgent α: α' = 1 + γ(α-1)
        #                           (0 = α≡1 i.e. logits untouched; 1 = original)
        #   corr_soft_factor      — when CorrAgent TERMINATEs, scores *= factor
        #                           (0 = original hard kill "scores=0")
        #   corr_consec_terminate — N consecutive TERMINATE decisions required
        #                           before actually acting (1 = act immediately)
        self._default_infer_safety = dict(
            det_delta_scale=1.0,
            assoc_alpha_gamma=1.0,
            corr_soft_factor=0.0,
            corr_consec_terminate=1,
        )
        self.infer_safety = dict(self._default_infer_safety)
        if infer_safety:
            self.infer_safety.update(
                {k: v for k, v in infer_safety.items()
                 if k in self._default_infer_safety})

        # Per-obj_idx counter for the CorrAgent debounce filter.  Reset in
        # ``clear()`` so each new sequence starts fresh.
        self._corr_term_streak: Dict[int, int] = {}

        print_lora_param_stats(self, tag='MOTRWithMARFT')

    # ------------------------------------------------------------------
    # Public helper — inference scripts call this to tune safety knobs
    # without touching the rest of the pipeline.
    # ------------------------------------------------------------------
    def set_infer_safety(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if k in self._default_infer_safety and v is not None:
                self.infer_safety[k] = v

    # ------------------------------------------------------------------
    # Convenience accessors (delegate to base)
    # ------------------------------------------------------------------
    @property
    def criterion(self):
        return self.base_model.criterion

    @property
    def num_classes(self):
        return self.base_model.num_classes

    def clear(self):
        # Reset both native tracker state AND our CorrAgent debounce counter
        # so each new video starts with a clean slate.
        self.base_model.clear()
        self._corr_term_streak.clear()

    def _generate_empty_tracks(self, proposals=None):
        return self.base_model._generate_empty_tracks(proposals)

    # ------------------------------------------------------------------
    # Main forward — mirrors MOTR.forward() but injects agents
    # ------------------------------------------------------------------
    def forward(self, data: dict):
        """
        Drop-in replacement for MOTR.forward().

        When ``use_agents`` is False the call is forwarded verbatim to the
        base model so that supervised-only warm-up works transparently.
        """
        if not self.use_agents:
            return self.base_model(data)

        # --- initialise criterion as the original code does ---
        if self.training:
            self.base_model.criterion.initialize_for_single_clip(
                data['gt_instances'],
                skip_loss=data.get('skip_loss', False),
            )
        frames = data['imgs']
        skip_loss = data.get('skip_loss', False)
        run_mode = data.get('run_mode', 'supervised' if self.training else 'inference')
        collect_outputs = not (self.training and skip_loss and run_mode == 'sampling')

        outputs: Dict[str, Any] = {
            'pred_logits': [] if collect_outputs else None,
            'pred_boxes': [] if collect_outputs else None,
        }

        track_instances = None
        keys = list(self._generate_empty_tracks()._fields.keys())
        prev_boxes = None

        for frame_index, (frame, gt, proposals) in enumerate(
                zip(frames, data['gt_instances'], data['proposals'])):
            frame.requires_grad = False
            is_last = frame_index == len(frames) - 1

            # query-denoise (reused from base)
            if self.base_model.query_denoise > 0:
                l_1 = l_2 = self.base_model.query_denoise
                gtboxes = gt.boxes.clone()
                _rs = torch.rand_like(gtboxes) * 2 - 1
                gtboxes[..., :2] += gtboxes[..., 2:] * _rs[..., :2] * l_1
                gtboxes[..., 2:] *= 1 + l_2 * _rs[..., 2:]
            else:
                gtboxes = None

            if track_instances is None:
                track_instances = self._generate_empty_tracks(proposals)
            else:
                track_instances = Instances.cat([
                    self._generate_empty_tracks(proposals),
                    track_instances])

            # ============================================================
            #  PRE-DECODER AGENT: DetAgent modifies ref_pts BEFORE Decoder
            # ============================================================
            det_info = self._apply_det_agent(track_instances)

            # ---- original single-image forward (backbone + transformer) ----
            # Decoder now uses DetAgent-adjusted ref_pts for cross-attention
            frame_tensor = nested_tensor_from_tensor_list([frame])
            frame_res = self.base_model._forward_single_image(
                frame_tensor, track_instances, gtboxes)

            # ============================================================
            #  POST-DECODER AGENTS: Assoc / Update / Corr on Decoder output
            # ============================================================
            frame_res, track_instances = self._apply_post_decoder_agents(
                frame_res, track_instances, prev_boxes, det_info)
            prev_boxes = track_instances.pred_boxes.detach().clone() \
                if hasattr(track_instances, 'pred_boxes') else None

            # ---- original post-processing (matching, loss, QIM) ----
            frame_res = self.base_model._post_process_single_image(
                frame_res, track_instances, is_last, run_mode=run_mode)

            # collect sampling outputs for GRPO / MARFT RL loss
            if run_mode == 'sampling':
                if 'log_prob' not in outputs:
                    outputs['log_prob'] = []
                outputs['log_prob'].append(frame_res.get('log_prob'))
                if 'frame_obj_idxes' in frame_res:
                    outputs.setdefault('obj_idxes_seq', []).append(
                        frame_res['frame_obj_idxes'])
                if 'frame_scores' in frame_res:
                    outputs.setdefault('scores_seq', []).append(
                        frame_res['frame_scores'])

            # collect agent log_probs for per-agent policy gradient
            if self.training and 'agent_infos' in frame_res:
                outputs.setdefault('agent_log_probs', []).append(
                    frame_res['agent_infos'])
                # --------------------------------------------------------
                # DDP-visibility tap (fix for "Expected to mark a variable
                # ready only once" in Phase 2).
                #
                # ``agent_infos`` is a dict[str, ActionInfo].  PyTorch 1.7's
                # ``DistributedDataParallel._find_tensors`` only recurses
                # into ``list / tuple / dict`` and therefore CANNOT reach
                # the ``log_prob`` tensors buried inside the ActionInfo
                # dataclass fields.
                #
                # As a consequence ``prepare_for_backward`` marks every
                # agent parameter (most importantly CorrAgent.classifier)
                # as **expected unused**, and schedules a synthetic
                # "ready" at the end of backward.  The actual backward of
                # ``-advantage * agent_lps_grad['corr']`` later fires a
                # real AccumulateGrad hook on the same parameter →
                # DDP counts TWO "ready" signals per iteration → crash.
                #
                # Exposing the log_prob tensors as a plain list inside
                # ``outputs`` lets ``_find_tensors`` walk their grad_fn
                # chain and register the agent parameters as genuinely
                # used.  The tap is read-only for the engine (it never
                # appears in any loss expression), so it changes
                # nothing about the gradient computation itself.
                # --------------------------------------------------------
                tap_bucket = outputs.setdefault('_ddp_visibility_tap', [])
                for _info in frame_res['agent_infos'].values():
                    lp = getattr(_info, 'log_prob', None)
                    if lp is not None and lp.requires_grad:
                        tap_bucket.append(lp)
                    ent = getattr(_info, 'entropy', None)
                    if ent is not None and ent.requires_grad:
                        tap_bucket.append(ent)

            track_instances = frame_res['track_instances']
            if collect_outputs:
                outputs['pred_logits'].append(frame_res['pred_logits'])
                outputs['pred_boxes'].append(frame_res['pred_boxes'])

        if not self.training:
            outputs['track_instances'] = track_instances
        elif not skip_loss:
            outputs['losses_dict'] = self.base_model.criterion.losses_dict
        return outputs

    # ------------------------------------------------------------------
    #  PRE-DECODER: DetAgent adjusts ref_pts before Decoder sampling
    # ------------------------------------------------------------------
    def _apply_det_agent(
        self,
        track_instances: Instances,
    ) -> Dict[str, Any]:
        """
        Run DetAgent BEFORE the Decoder so that Deformable Cross-Attention
        samples features from agent-adjusted reference points.

        Returns det_info dict (log_prob etc.) for later RL loss collection.
        """
        if not self.agent_manager.det_agent.enabled:
            return {}

        query_embed = track_instances.query_pos
        N = query_embed.shape[0]
        device = query_embed.device

        # Detach observations for the same reason as the post-decoder
        # agents (see ``_apply_post_decoder_agents``): the RL log-prob
        # gradient must not propagate back through MOTR / QIM, otherwise
        # ``losses.backward()`` traverses the previous frame's
        # checkpointed Transformer a second time → DDP "marked ready
        # twice".  ``delta_p`` itself remains differentiable so the
        # supervised loss can still update DetAgent parameters.
        det_obs = {
            'query_embed': query_embed.detach(),
            'ref_pts': track_instances.ref_pts.detach().clone(),
        }
        delta_p, _, det_action_info = self.agent_manager.det_agent(
            det_obs, None)

        # Inference-time safety: shrink Δp to avoid violent ref_pts
        # displacement when DetAgent is under-trained.  scale=0 disables
        # the perturbation structurally while keeping the agent enabled.
        det_scale = self.infer_safety.get('det_delta_scale', 1.0)
        if det_scale != 1.0:
            delta_p = delta_p * det_scale

        # Apply offset to ref_pts — Decoder will use these adjusted points
        adjusted_ref = track_instances.ref_pts.clone()
        adjusted_ref[:, :2] = (adjusted_ref[:, :2] + delta_p).clamp(0.0, 1.0)
        track_instances.ref_pts = adjusted_ref

        return {
            'delta_p': delta_p,
            'action_info': det_action_info,
        }

    # ------------------------------------------------------------------
    #  POST-DECODER: Assoc / Update / Corr on Decoder output
    # ------------------------------------------------------------------
    def _apply_post_decoder_agents(
        self,
        frame_res: Dict[str, Any],
        track_instances: Instances,
        prev_boxes: Optional[Tensor],
        det_info: Dict[str, Any],
    ) -> tuple:
        """
        Run AssocAgent, UpdateAgent, CorrAgent on Decoder outputs.
        DetAgent already ran pre-Decoder; its info is merged here.

        Handles the ``query_denoise > 0`` case correctly: the Decoder
        actually processes ``n_track + n_gt_boxes`` queries (the last
        ``n_gt_boxes`` are auxiliary denoising queries injected by MOTR
        in ``_forward_single_image``).  Agents must act ONLY on the
        first ``n_track`` queries; the denoise portion is left intact
        so that ``_post_process_single_image`` can still split it out
        into ``ps_outputs`` for the PS loss.
        """
        hs_full = frame_res['hs']  # [1, N_full, D]  (may include denoise)
        if hs_full.dim() == 3:
            hs_full = hs_full[0]   # [N_full, D]

        # Upcast fp16 Decoder activations to fp32 so that agent MLPs
        # (fp32 weights) and track_instances.query_pos (fp32, from
        # nn.Embedding) can safely participate in torch.cat / matmul.
        if hs_full.dtype == torch.float16:
            hs_full = hs_full.float()

        n_track = len(track_instances)     # real track queries only
        n_full = hs_full.shape[0]
        hs = hs_full[:n_track]              # [n_track, D] — matches query_pos
        device = hs.device

        query_embed = track_instances.query_pos  # [n_track, D], fp32

        # IMPORTANT (2026-04-25 DDP+checkpoint fix):
        #   The post-decoder agents READ from ``hs`` (Decoder output of
        #   the checkpointed Transformer) and from ``query_embed``.  If we
        #   let the RL log-prob gradient flow back through these tensors,
        #   ``losses.backward()`` will traverse the checkpoint segment a
        #   SECOND time (the first traversal comes from the supervised
        #   loss).  Each traversal triggers ``torch.utils.checkpoint``'s
        #   reentrant backward, which fires the DDP gradient hook on every
        #   shared MOTR / LoRA parameter twice → DistributedDataParallel
        #   ``"Expected to mark a variable ready only once"`` error.
        #
        # The standard MARFT design treats agents as actors built on top
        # of the MOTR feature extractor: the RL signal should ONLY update
        # agent parameters, while LoRA / MOTR are updated by the
        # supervised loss path.  Detaching here enforces exactly that
        # separation, fixes the DDP error, and is fully compatible with
        # ``--use_checkpoint`` and ``find_unused_parameters=True``.
        hs_obs = hs.detach()
        query_embed_obs = query_embed.detach()

        joint_infos: Dict[str, Any] = {}

        # Carry over DetAgent info from pre-decoder phase
        if det_info and 'action_info' in det_info:
            joint_infos['det'] = det_info['action_info']

        # --- AssocAgent: modulate pred_logits for TRACK queries only ---
        if self.agent_manager.assoc_agent.enabled and n_track > 0:
            assoc_obs = {
                'det_embed': hs_obs,            # [n_track, D] — detached
                'track_embed': query_embed_obs,  # [n_track, D] — detached
            }
            alpha, _, assoc_info = self.agent_manager.assoc_agent(assoc_obs, None)
            # Inference-time safety: blend α towards the neutral value 1.0.
            # α' = 1 + γ*(α - 1), so γ=0 ⇒ α'≡1 (logits untouched),
            # γ=1 ⇒ original α.  This bounds score jitter caused by an
            # under-trained AssocAgent which is the dominant amplifier of
            # score-dip -> ID-rebirth events in RuntimeTrackerBase.
            gamma = self.infer_safety.get('assoc_alpha_gamma', 1.0)
            if gamma != 1.0:
                alpha = 1.0 + gamma * (alpha - 1.0)
            pred_logits = frame_res['pred_logits']  # [1, N_full, C]
            # Apply alpha only to the first n_track queries; denoise passes through.
            new_logits = pred_logits.clone()
            new_logits[:, :n_track] = (
                pred_logits[:, :n_track] * alpha.unsqueeze(0).to(pred_logits.dtype)
            )
            frame_res['pred_logits'] = new_logits
            joint_infos['assoc'] = assoc_info

        # --- UpdateAgent: adaptive gating ---
        if self.agent_manager.update_agent.enabled and n_track > 0:
            update_obs = {
                'pre_update_embed': query_embed_obs,  # detached
                'post_update_embed': hs_obs,           # detached
            }
            gates, _, upd_info = self.agent_manager.update_agent(update_obs, None)
            frame_res['update_gates'] = gates
            joint_infos['update'] = upd_info

        # --- CorrAgent: track correction (operates on track_instances only) ---
        if self.agent_manager.corr_agent.enabled and n_track > 0:
            global_ctx = hs_obs.mean(dim=0)            # [D] — detached
            scores = track_instances.scores.detach().clone() if hasattr(
                track_instances, 'scores') else torch.ones(n_track, device=device) * 0.5
            corr_obs = {
                'track_embed': hs_obs,              # [n_track, D] — detached
                'global_context': global_ctx,
                'scores': scores,
            }
            corr_actions, _, corr_info = self.agent_manager.corr_agent(
                corr_obs, None)
            terminate_mask = corr_actions == ACTION_TERMINATE

            # ---- Inference-time safety: debounce + soft-kill ----
            # (i)  Debounce: require the CorrAgent to output TERMINATE for
            #      the *same* obj_idx for ``consec`` consecutive frames before
            #      we actually act on it.  Without this, a single noisy
            #      prediction immediately triggers score=0 -> disappearance
            #      -> new ID.
            # (ii) Soft-kill: when we do act, multiply the score by a factor
            #      in (0, 1] instead of hard-setting it to 0.  Scores that
            #      remain above ``filter_score_thresh`` keep their obj_idx,
            #      so a short blip cannot cause ID-rebirth.  factor=0 (the
            #      default during training) reproduces the original hard-kill
            #      behaviour used by the RL reward calculation.
            consec_req = max(1, int(self.infer_safety.get(
                'corr_consec_terminate', 1)))
            soft_factor = float(self.infer_safety.get('corr_soft_factor', 0.0))

            act_mask = terminate_mask
            if consec_req > 1 and hasattr(track_instances, 'obj_idxes'):
                obj_idxes = track_instances.obj_idxes
                streak = self._corr_term_streak
                keep_keys = set()
                act_flags = torch.zeros_like(terminate_mask)
                for i in range(n_track):
                    oid = int(obj_idxes[i].item())
                    if oid < 0:
                        continue  # untracked yet, skip debounce
                    keep_keys.add(oid)
                    if bool(terminate_mask[i].item()):
                        streak[oid] = streak.get(oid, 0) + 1
                        if streak[oid] >= consec_req:
                            act_flags[i] = True
                    else:
                        streak[oid] = 0
                # prune entries for obj_idxes that have left the scene
                for k in list(streak.keys()):
                    if k not in keep_keys:
                        del streak[k]
                act_mask = act_flags

            if act_mask.any() and hasattr(track_instances, 'scores'):
                if soft_factor <= 0.0:
                    track_instances.scores[act_mask] = 0.0
                else:
                    track_instances.scores[act_mask] = \
                        track_instances.scores[act_mask] * soft_factor
            joint_infos['corr'] = corr_info

        frame_res['agent_infos'] = joint_infos
        return frame_res, track_instances

    # ------------------------------------------------------------------
    # Inference — mirror of MOTR.inference_single_image with agents
    # injected at the same points as training (pre-Decoder DetAgent,
    # post-Decoder Assoc/Update/Corr).
    # ------------------------------------------------------------------
    @torch.no_grad()
    def inference_single_image(self, img, ori_img_size,
                               track_instances=None, proposals=None):
        base = self.base_model

        # If agents disabled, short-circuit to the original MOTR path
        # (keeps behaviour identical to baseline MOTRv2 inference).
        if not self.use_agents:
            return base.inference_single_image(
                img, ori_img_size, track_instances, proposals)

        if not isinstance(img, NestedTensor):
            img = nested_tensor_from_tensor_list(img)

        if track_instances is None:
            track_instances = base._generate_empty_tracks(proposals)
        else:
            track_instances = Instances.cat([
                base._generate_empty_tracks(proposals),
                track_instances])

        # ----- Pre-Decoder: DetAgent adjusts ref_pts -----
        det_info = self._apply_det_agent(track_instances)

        # ----- Backbone + Encoder + Decoder -----
        frame_res = base._forward_single_image(
            img, track_instances=track_instances)

        # ----- Post-Decoder: Assoc / Update / Corr -----
        frame_res, track_instances = self._apply_post_decoder_agents(
            frame_res, track_instances, None, det_info)

        # ----- Matching / QIM / TrackerBase via original post-process -----
        frame_res = base._post_process_single_image(
            frame_res, track_instances, is_last=False, run_mode='inference')

        track_instances = frame_res['track_instances']
        track_instances = base.post_process(track_instances, ori_img_size)
        ret = {'track_instances': track_instances}
        if 'ref_pts' in frame_res:
            ref_pts = frame_res['ref_pts']
            img_h, img_w = ori_img_size
            scale_fct = torch.Tensor([img_w, img_h]).to(ref_pts)
            ret['ref_pts'] = ref_pts * scale_fct[None]
        return ret

    # ------------------------------------------------------------------
    # Convenience accessors so submit_dance-style scripts that set
    # ``model.track_embed.score_thr`` / ``model.track_base = ...``
    # work transparently on the wrapped model.
    # ------------------------------------------------------------------
    @property
    def track_embed(self):
        return self.base_model.track_embed

    @property
    def track_base(self):
        return self.base_model.track_base

    @track_base.setter
    def track_base(self, value):
        self.base_model.track_base = value

    @property
    def post_process(self):
        return self.base_model.post_process

    # ------------------------------------------------------------------
    # Parameter group helpers
    # ------------------------------------------------------------------
    def get_lora_params(self) -> List[nn.Parameter]:
        return [p for p in self.lora_modules.parameters() if p.requires_grad]

    def get_agent_params(self) -> List[nn.Parameter]:
        return list(self.agent_manager.parameters())

    def get_all_trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]


# ======================================================================
# Builder — called from engine / main
# ======================================================================
def build_marft_model(base_model: nn.Module, args) -> MOTRWithMARFT:
    """
    Wrap an already-built MOTR model with the MARFT overlay.

    Reads MARFT-specific flags from ``args`` (added by
    ``engine_marft.add_marft_args``).
    """
    hidden_dim = getattr(args, 'hidden_dim', 256)
    use_lora = getattr(args, 'marft_use_lora', True)
    use_agents = getattr(args, 'marft_use_agents', True)
    agent_config = {
        'corr': {'corr_conf_threshold': getattr(args, 'marft_corr_threshold', 0.4)},
    }
    infer_safety = dict(
        det_delta_scale=getattr(args, 'marft_infer_det_delta_scale', 1.0),
        assoc_alpha_gamma=getattr(args, 'marft_infer_assoc_alpha_gamma', 1.0),
        corr_soft_factor=getattr(args, 'marft_infer_corr_soft_factor', 0.0),
        corr_consec_terminate=getattr(args, 'marft_infer_corr_consec_terminate', 1),
    )
    return MOTRWithMARFT(
        base_model,
        hidden_dim=hidden_dim,
        use_lora=use_lora,
        use_agents=use_agents,
        agent_config=agent_config,
        infer_safety=infer_safety,
    )
