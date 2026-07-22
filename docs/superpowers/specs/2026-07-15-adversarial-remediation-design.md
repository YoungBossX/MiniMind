# 对抗性审查修复设计

> 状态（2026-07-22）：历史设计记录。后续提交 `09196d7` 已扩展本设计的非目标，包括 checkpoint schema v2 和每 rank RNG 状态；现行行为以代码、测试和根 README 为准。

## 目标

按风险顺序修复训练续训、评估可靠性和数据校验问题，保持现有训练与推理接口兼容。

## 范围与顺序

1. 续训数据顺序：pretrain、LoRA、DPO 的正常与恢复路径必须从同一确定性批次序列取数；恢复后跳过的批次仅来自该序列的前缀。
2. 评估编排：`evals/run_all.py` 必须汇总子评估状态、写入报告，并在任一请求任务失败时以非零状态退出。
3. PPO critic：value head 只消费 backbone 的一次最终归一化输出。
4. 显式 LoRA 路径：缺失时失败，而不是静默评估基座模型。
5. DPO 数据审计：验证 chosen/rejected 的共同上下文、角色和最终 assistant 回复。
6. 评估与性能：向速度评估透传 batch size；为 padding 下的注意力性能添加可观测性或长度分桶准备。

## 设计

### 确定性训练批次

新增共享的 epoch 批次构造接口。单卡使用显式、带固定 epoch 种子的随机排列；DDP 继续使用已 `set_epoch(epoch)` 的 `DistributedSampler`。正常训练和恢复训练都使用同一 `SkipBatchSampler`，其中正常路径 `skip_batches=0`。这样恢复路径只移除同一批次序列的前缀，不再切换为顺序索引。

本设计阶段未把完整 RNG 状态作为“样本序列等价”承诺的一部分；后续实现已将每 rank RNG 状态纳入 schema v2 checkpoint。

### 评估状态

每个 evaluator 产生 `success`、`failed` 或 `skipped` 状态。汇总报告显示各状态；任一实际执行的 evaluator 失败时，编排器以非零状态退出，避免 CI 误绿。

### 训练与数据防线

PPO critic 删除重复 RMSNorm。显式的 LoRA 路径须存在。DPO 审计对 pair 进行结构化验证：两侧非空、角色/content 有效、均以 assistant 收尾，且去除最终 assistant 后的上下文一致。

## 验收

- 每项修复均先有能在修复前失败的回归测试。
- 单卡恢复测试证明已恢复批次等于无中断批次序列的后缀。
- `run_all` 的子进程失败传播为非零退出码，成功时为零。
- PPO critic、LoRA 缺失路径、DPO pair 审计有定向测试。
- 全量 pytest、语法编译和适用的 GPU smoke test 通过。

## 非目标

- 原始设计不改变 checkpoint 格式，也不实现跨进程 RNG 状态快照；后续实现已由 schema v2 和每 rank RNG sidecar 覆盖这些限制。
- 不重构 Reward Model。
- 不删除用户已有的未提交改动。
