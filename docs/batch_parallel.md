# MOTR Batch Parallel 重构总结

## 背景

原 MOTR 代码主要为 `batch_size=1` 设计，当修改 `batch_size > 1` 时出现了一系列 `'list' object has no attribute 'xxx'` 错误。

## 重构过程

### 第一阶段：基础兼容性修复
- 修复 `frame.requires_grad = False`（frame 可能为 list）
- 修复 `gt.boxes.clone()`（gt 可能为 list）
- 修复 `nested_tensor_from_tensor_list([frame])` 中的 ndim 错误
- 修复 `gt_instances_i.obj_ids` 访问问题

**核心策略**：在访问属性前增加 `isinstance(x, list)` 判断，并取 `x[0]`

### 第二阶段：结构化重构
- 重构 `MOTR.forward()` 方法，使用清晰的 `clip_idx`/`b_idx` 命名
- 为 `ClipMatcher` 添加 `_get_gt_for_frame()` 辅助方法
- 优化 `calc_loss_for_track_scores` 和 `match_for_single_frame`
- 修复循环导入问题（使用函数内延迟导入）

### 第三阶段：代码清理和文档
- 添加详细的中文文档字符串
- 统一变量命名和注释风格
- 优化错误处理逻辑

## 当前实现分析

**优点：**
- ✅ 可以正常以 `batch_size=2/4` 训练
- ✅ 代码结构清晰，可读性好
- ✅ 显存占用随 batch_size 合理增加
- ✅ 保持了原有单 clip 行为的兼容性

**局限性：**
- **不是真正意义的 batch 并行**：仍然采用 `for b_idx in range(B):` 的循环方式
- 每个 clip 仍然是串行处理的，只是同时加载了多个 clip 的数据
- `track_instances` 的管理仍然是 per-clip 的

## 未来优化方向

### 1. 真正 (B*T) 并行（推荐）
- 将所有帧展平为 `(B*T, C, H, W)`
- Backbone 和 Transformer 一次性处理所有帧
- 只在 track management 阶段区分不同 clip

### 2. 数据加载优化
- 改进 `mot_collate_fn` 支持 tensor 形式的 batched 数据
- 优化 `data_prefetcher` 对复杂嵌套结构的支持

### 3. 内存优化
- 减少 `Instances.cat()` 的频繁调用
- 优化 checkpoint 的使用策略

## 使用方法

```bash
# 当前支持的训练命令
python main.py --batch_size 4 --epochs 1

# 推荐先用小 batch 测试
python main.py --batch_size 2 --epochs 1 --lr 1e-5
```

## 注意事项

1. `pin_memory` 已禁用（因为复杂嵌套数据结构会导致 CUDA pin memory 错误）
2. `use_checkpoint` 仍然可用，但对 batch>1 的加速效果有限
3. GRPO 训练模式（`--use_grpo`）需要进一步测试

## 重构时间线

- 2026.03：发现 batch_size>1 兼容性问题
- 多轮迭代：修复各种 list attribute 错误
- 最终版本：结构清晰、可维护的 batch 支持实现



