# ------------------------------------------------------------------------
# Copyright (c) 2026 JoshuaWenHIT. All Rights Reserved.
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch
from torch.cuda.amp import autocast, GradScaler
import util.misc as utils
from util.reward_mechanisms import compute_reward_from_obj_idxes

from datasets.data_prefetcher import data_dict_to_cuda


def train_one_epoch_mot(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, use_amp: bool = False):
    model.train()
    criterion.train()
    scaler = GradScaler(enabled=use_amp)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for data_dict in metric_logger.log_every(data_loader, print_freq, header):
        data_dict = data_dict_to_cuda(data_dict, device)
        clip_dicts = list(utils.iter_mot_clip_dicts(data_dict))
        n_clip = len(clip_dicts)
        loss_dict = None
        losses = None
        for sub in clip_dicts:
            with autocast(enabled=use_amp):
                outputs = model(sub)
                ld = criterion(outputs, sub)
                weight_dict = criterion.weight_dict
                li = sum(ld[k] * weight_dict[k] for k in ld.keys() if k in weight_dict)
            if loss_dict is None:
                loss_dict = {k: ld[k].clone() for k in ld}
                losses = li
            else:
                for k in ld:
                    loss_dict[k] = loss_dict[k] + ld[k]
                losses = losses + li
        for k in list(loss_dict.keys()):
            loss_dict[k] = loss_dict[k] / n_clip
        losses = losses / n_clip

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        weight_dict = criterion.weight_dict
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            if max_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            else:
                grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            if max_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            else:
                grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
            optimizer.step()

        # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        # gather the stats from all processes

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_grpo(model: torch.nn.Module, criterion: torch.nn.Module,
                         data_loader: Iterable, optimizer: torch.optim.Optimizer,
                         device: torch.device, epoch: int, max_norm: float = 0,
                         id_switch_penalty: float = 15.0, id_stable_reward: float = 1.0,
                         fp_penalty: float = 2.0,
                         grpo_loss_weight: float = 1.0, grpo_group_size: int = 4,
                         use_amp: bool = False):
    """
    GRPO 组采样训练流程，继承 train_one_epoch_mot 的监督 loss，并加入基于 obj_idxes 的奖励 loss：
    - 监督 loss：criterion 的 loss_dict（同 train_one_epoch_mot）
    - 奖励 loss：GRPO 组内相对优势 * log_prob（符合 GRPO 原理）
    - 总 loss = 监督 loss + grpo_loss_weight * 奖励 loss

    GRPO 核心：对同一输入采样 grpo_group_size 次，计算 advantage_i = reward_i - mean(rewards)，
    使用 advantage 而非 raw reward 做 policy gradient，降低方差。
    """
    model.train()
    criterion.train()
    scaler = GradScaler(enabled=use_amp)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('reward', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('reward_std', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('loss_grpo', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Epoch: [{}] GRPO'.format(epoch)
    print_freq = 10

    for data_dict in metric_logger.log_every(data_loader, print_freq, header):
        data_dict = data_dict_to_cuda(data_dict, device)
        clip_dicts = list(utils.iter_mot_clip_dicts(data_dict))
        n_clip = len(clip_dicts)
        weight_dict = criterion.weight_dict
        loss_dict = None
        losses_sup = None
        loss_grpo_acc = None

        for sub in clip_dicts:
            sub['run_mode'] = 'sampling'

            # 组采样：对同一输入采样 grpo_group_size 次（每个 clip 独立，与 batch_size=1 一致）
            rewards_list = []
            total_log_probs_list = []
            outputs_for_criterion = None
            for sample_idx in range(grpo_group_size):
                # For GRPO samples that are not used for supervised loss, skip loss
                # construction inside ClipMatcher to save compute.
                sub['skip_loss'] = sample_idx < (grpo_group_size - 1)
                with autocast(enabled=use_amp):
                    outputs = model(sub)
                outputs_for_criterion = outputs  # 取最后一组用于监督 loss
                obj_idxes_seq = outputs.get('obj_idxes_seq', [])
                scores_seq = outputs.get('scores_seq', [])
                log_probs = outputs.get('log_prob', [])
                if obj_idxes_seq and log_probs:
                    reward = compute_reward_from_obj_idxes(
                        obj_idxes_seq,
                        id_switch_penalty=id_switch_penalty,
                        id_stable_reward=id_stable_reward,
                        fp_penalty=fp_penalty,
                        scores_seq=scores_seq,
                    )
                    total_log_prob = sum(lp.sum() for lp in log_probs)
                    rewards_list.append(reward)
                    total_log_probs_list.append(total_log_prob)

            sub['skip_loss'] = False
            # 监督 loss：用最后一组 outputs（与原始逻辑一致）
            with autocast(enabled=use_amp):
                ld = criterion(outputs_for_criterion, sub)
                sup_i = sum(ld[k] * weight_dict[k] for k in ld.keys() if k in weight_dict)

            # GRPO 奖励 loss：组内优势 = reward_i - mean(rewards)
            loss_grpo = torch.tensor(0., device=device)
            reward_val = 0.
            reward_std_val = 0.
            if len(rewards_list) >= 2:  # 至少 2 组才能计算组内优势
                rewards_stack = torch.stack(rewards_list)
                total_log_probs_stack = torch.stack(total_log_probs_list)
                baseline = rewards_stack.mean()
                advantages = rewards_stack - baseline
                loss_grpo = -(advantages * total_log_probs_stack).sum()
                reward_val = rewards_stack.mean().item()
                reward_std_val = rewards_stack.std().item()
            elif len(rewards_list) == 1:
                # 退化为 REINFORCE
                loss_grpo = -rewards_list[0] * total_log_probs_list[0]
                reward_val = rewards_list[0].item()
            metric_logger.update(reward=reward_val, reward_std=reward_std_val, loss_grpo=loss_grpo.item())

            if loss_dict is None:
                loss_dict = {k: ld[k].clone() for k in ld}
                losses_sup = sup_i
                loss_grpo_acc = loss_grpo
            else:
                for k in ld:
                    loss_dict[k] = loss_dict[k] + ld[k]
                losses_sup = losses_sup + sup_i
                loss_grpo_acc = loss_grpo_acc + loss_grpo

        for k in list(loss_dict.keys()):
            loss_dict[k] = loss_dict[k] / n_clip
        losses_sup = losses_sup / n_clip
        loss_grpo = loss_grpo_acc / n_clip

        # 总 loss = 监督 loss + grpo_loss_weight * 奖励 loss
        losses = losses_sup + grpo_loss_weight * loss_grpo

        # reduce losses over all GPUs for logging
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_sup_reduced = sum(loss_dict_reduced_scaled.values())
        loss_grpo_reduced = loss_grpo
        if utils.is_dist_avail_and_initialized():
            loss_grpo_reduced = loss_grpo.detach().clone()
            torch.distributed.all_reduce(loss_grpo_reduced)
            loss_grpo_reduced = loss_grpo_reduced / utils.get_world_size()
        loss_value = (losses_sup_reduced + grpo_loss_weight * loss_grpo_reduced).item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            if max_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            else:
                grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            if max_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            else:
                grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
            optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
