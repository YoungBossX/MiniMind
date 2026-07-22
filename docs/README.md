# MiniMind 文档索引

## 现行入口

- [项目 README](../README.md)：安装、训练、推理和评测命令。
- [评测系统说明](../evals/README.md)：单项评测、汇总评测、结果对比和输出文件。
- `scripts/audit_data.py`：训练前的数据结构、角色、重复样本和 SHA-256 审计。
- `feedback/run_feedback_loop.py`：把评测失败整理为必须人工审核的 SFT 候选。

## 历史设计记录

`superpowers/plans/` 和 `superpowers/specs/` 保存实现过程中的设计与执行记录。
其中的复选框和中间方案是历史上下文，不是当前待办，也不覆盖现行代码行为。
需要确认实际行为时，以当前代码、测试和上面的现行入口为准。
