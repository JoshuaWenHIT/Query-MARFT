import torch
from typing import List


def compute_reward_from_obj_idxes(
    obj_idxes_seq: List[torch.Tensor],
    id_switch_penalty: float = 15.0,
    id_stable_reward: float = 1.0,
    group_indices=None,
):
    """
    基于 ClipMatcher 更新后的 track_instances.obj_idxes 计算 Reward。
    obj_idxes_seq: 每帧的 obj_idxes tensor 列表，由 model forward 在 run_mode='sampling' 时收集。
        每个 tensor 形状 [N]，表示该帧每个 query 匹配到的 GT 目标 ID（-1 表示未匹配）。
    对比连续两帧之间同一 query 索引的 obj_idxes：
    - 从一个正值变为另一个不同的正值 -> ID Switch，扣除 id_switch_penalty 分（默认 15）
    - 保持一致 -> 给予 id_stable_reward 分

    Returns:
        reward: scalar tensor (higher is better)
    """
    if not obj_idxes_seq or len(obj_idxes_seq) < 2:
        return torch.tensor(0., device=obj_idxes_seq[0].device if obj_idxes_seq else 'cpu')

    device = obj_idxes_seq[0].device
    if group_indices is not None:
        time_indices = group_indices
    else:
        time_indices = list(range(len(obj_idxes_seq)))

    # Stack: [T, N], T=num_frames, N=num_queries
    obj_idxes = torch.stack([obj_idxes_seq[t] for t in time_indices], dim=0)
    T, N = obj_idxes.shape

    total_reward = torch.tensor(0., device=device)
    for qi in range(N):
        prev_obj_id = -1
        for t in range(T):
            curr_obj_id = obj_idxes[t, qi].item()
            if curr_obj_id == -1:
                prev_obj_id = -1
                continue
            if prev_obj_id == -1:
                prev_obj_id = curr_obj_id
                continue
            if curr_obj_id == prev_obj_id:
                total_reward = total_reward + id_stable_reward
            else:
                # ID Switch: 从一个正值变为另一个不同的正值
                total_reward = total_reward - id_switch_penalty
                prev_obj_id = curr_obj_id

    return total_reward


def compute_mot_reward(
    pred_tracks, 
    gt_tracks, 
    matcher, 
    id_switch_penalty=10.0, 
    id_stable_reward=1.0,
    group_indices=None,
):
    """
    pred_tracks: List of length T, each item is Instances (predicted), len N
        Each Instances must at least have .boxes [N, 4], and optionally .ids
    gt_tracks: List of length T, each item is Instances (gt), len M
        Each Instances must at least have .boxes [M, 4] and .obj_idxes [M]
    matcher: callable matching function with API like models/motr.py::ClipMatcher (returns indices list)
    id_switch_penalty: float, penalty for a single ID Switch
    id_stable_reward: float, reward for a query keeping the same obj_id in adjacent frames
    group_indices: Optional[List[int]], supports group sampling for GRPO (sampled frame indices)
    
    Returns:
        reward: scalar tensor (higher is better)
    """
    device = pred_tracks[0].boxes.device
    # If group_indices given, run only on the selected frames; otherwise use all
    if group_indices is not None:
        time_indices = group_indices
    else:
        time_indices = list(range(len(pred_tracks)))
    T = len(time_indices)

    matched_query_gt = [] # size [T][N], for each frame, for each query: matched GT id or -1

    # Perform assignment using matcher for every frame
    for t_i, t in enumerate(time_indices):
        preds = pred_tracks[t]
        gts = gt_tracks[t]

        pred_boxes = preds.boxes      # [N, 4]
        gt_boxes = gts.boxes          # [M, 4]
        N = pred_boxes.shape[0]
        M = gt_boxes.shape[0]
        if hasattr(preds, 'ids'):
            query_indices = preds.ids
        else:
            query_indices = torch.arange(N, device=pred_boxes.device)
        if hasattr(gts, 'obj_idxes'):
            gt_ids = gts.obj_idxes
        else:
            gt_ids = torch.arange(M, device=gt_boxes.device)

        # Use matcher to get indices
        if N > 0 and M > 0:
            indices = matcher(
                [preds], [gts])  # motr/matcher返回格式: indices=list(tuple((idx1_tensor, idx2_tensor)))
            if isinstance(indices, list):
                indices = indices[0]  # Only one batch element
            pred_matched, gt_matched = indices  # tensors, shape (K,)
            # Build matched_gt: size N, for each query_idx: matched obj_idx or -1 if not matched
            matched_gt = torch.full((N,), -1, dtype=torch.long, device=device)
            if pred_matched.numel():
                matched_gt[pred_matched] = gt_ids[gt_matched]
        else:
            matched_gt = torch.full((N,), -1, dtype=torch.long, device=device)

        matched_query_gt.append(matched_gt)

    # Compute reward/penalty over queries for adjacent frames
    # Assumes queries are consistent across frames (same ordering or ids).
    # [T][N] -> transpose to [N][T]
    if len(matched_query_gt) == 0:
        return torch.tensor(0., device=device)
    matched_query_gt = torch.stack(matched_query_gt, dim=0) # [T, N]
    matched_query_gt_tr = matched_query_gt.transpose(0, 1)  # [N, T]

    total_reward = torch.tensor(0., device=device)
    num_id_switches = 0
    num_stable = 0

    for query_idx in range(matched_query_gt_tr.shape[0]):
        prev_obj_id = -1
        for t in range(matched_query_gt_tr.shape[1]):
            curr_obj_id = matched_query_gt_tr[query_idx, t].item()
            if curr_obj_id == -1:
                prev_obj_id = -1
                continue
            if prev_obj_id == -1:
                prev_obj_id = curr_obj_id
                continue
            if curr_obj_id == prev_obj_id:
                # stable id, positive reward
                total_reward += id_stable_reward
                num_stable += 1
            else:
                # id switch, penalty
                total_reward -= id_switch_penalty
                num_id_switches += 1
                prev_obj_id = curr_obj_id

    return total_reward
