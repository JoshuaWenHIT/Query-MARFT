# GRPO 实现检查报告

## 1. 计算图检查：Reward Loss 是否参与反向传播？

**结论：✅ 是，梯度正确流入模型。**

- `loss_grpo = -reward * total_log_prob`
- `reward`：来自 `compute_reward_from_obj_idxes` 的标量张量，不含 `requires_grad`（是常量）
- `total_log_prob`：由 `log_probs` 求和得到，`log_probs` 来自模型 Bernoulli 采样的 `log_prob`，**含梯度**
- `losses = losses_sup + grpo_loss_weight * loss_grpo` → `losses.backward()` 时梯度会经由 `total_log_prob` 传回模型

---

## 2. 是否实现 GRPO 的「组内优势值」？

**结论：❌ 否，当前实现缺少 GRPO 的核心步骤。**

标准 GRPO 需要：
1. **组采样**：对同一输入采样 K 个轨迹（K≥2，如 16）
2. **组内均值作为 baseline**：`baseline = mean(reward_1, ..., reward_K)`
3. **相对优势**：`advantage_i = reward_i - baseline` 或 `(reward_i - baseline) / std`
4. **策略梯度**：`loss = -sum(advantage_i * log_prob_i)`，而非直接用 `-reward * log_prob`

当前实现：
- 每个 batch 只 **1 次** forward，即 1 个样本
- 未做组采样、未计算 baseline
- 直接用 **reward** 而非 **advantage**，更接近 **REINFORCE / 简单策略梯度**，而非 GRPO

---

## 3. 与 GRPO 原理的符合度

| 维度 | GRPO 要求 | 当前实现 | 符合？ |
|------|-----------|----------|--------|
| 组采样 | 同一 prompt 采样多组轨迹 | 每 batch 仅 1 次采样 | ❌ |
| Baseline | 组内平均 reward | 无 baseline | ❌ |
| 优势 | advantage = reward - baseline | 使用 raw reward | ❌ |
| 方差 | 相对 reward 降低方差 | 未做 | ❌ |
| 策略梯度 | 用 advantage 加权 log_prob | 用 raw reward 加权 | 部分 ✅ |

当前形式本质上是 **REINFORCE**，reward 的尺度与方差直接影响训练稳定性，未体现 GRPO 的「相对优势」思想。

---

## 4. 已实现的 GRPO 改动

已在 `engine.py` 中实现：

1. **组采样**：对同一 `data_dict` 做 `grpo_group_size` 次 forward，得到 K 组 `(reward_i, total_log_prob_i)`。
2. **组内优势**：`baseline = mean(rewards)`，`advantage_i = reward_i - baseline`。
3. **GRPO loss**：`loss_grpo = -sum(advantage_i * total_log_prob_i)`。
4. **退化**：当 `grpo_group_size=1` 时退化为 REINFORCE；当 `grpo_group_size>=2` 时使用 GRPO 优势。

参数 `--grpo_group_size` 默认 4，可调。
