import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.MiniMindModel import MiniMindConfig
from dataset.llm_dataset import DPODataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')

def logits_to_log_probs(logits, labels):
    """
    把模型输出的 logits 转换为每个 token 位置的 log 概率。
 
    公式：
        log_prob(token_t) = log( softmax(logits_t)[label_t] )
                          = log_softmax(logits_t)[label_t]
 
    参数：
        logits: [B, seq_len, vocab_size]  模型对每个位置所有词的原始分数
        labels: [B, seq_len]              每个位置的真实 token id
 
    返回：
        log_probs_per_token: [B, seq_len]  每个位置真实 token 的 log 概率
    """
 
    # 对 vocab 维度做 log_softmax，把原始分数转成 log 概率
    # log_probs shape: [B, seq_len, vocab_size]
    log_probs = F.log_softmax(logits, dim=2)
 
    # torch.gather：从 vocab 维度（dim=2）取出 label 对应位置的 log 概率
    # labels.unsqueeze(2) shape: [B, seq_len, 1]  → 作为索引
    # gather 结果 shape: [B, seq_len, 1]
    # squeeze(-1) → [B, seq_len]
    log_probs_per_token = torch.gather(
        log_probs, dim=2, index=labels.unsqueeze(2)
    ).squeeze(-1)
 
    return log_probs_per_token

def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """
    计算 DPO（Direct Preference Optimization）损失。
 
    ==================== DPO 公式 ====================
 
    原始公式：
        L_DPO = -E[ log σ( β · (
            log π(y⁺|x)/π_ref(y⁺|x) - log π(y⁻|x)/π_ref(y⁻|x)
        ))]
 
    其中：
        π       = 策略模型（policy model，正在训练的模型）
        π_ref   = 参考模型（reference model，冻结的 SFT 模型）
        y⁺      = chosen（人类偏好的回答）
        y⁻      = rejected（次优回答）
        β       = 温度超参，控制偏离参考模型的惩罚力度
        σ       = sigmoid 函数
 
    利用对数性质展开：
        log π(y⁺|x)/π_ref(y⁺|x) = log π(y⁺|x) - log π_ref(y⁺|x)
 
    调换顺序后等价形式（代码实现方式）：
        L_DPO = -E[ log σ( β · (
            [log π(y⁺|x) - log π(y⁻|x)]         ← pi_logratios
          - [log π_ref(y⁺|x) - log π_ref(y⁻|x)] ← ref_logratios
        ))]
 
    ==================================================
 
    参数：
        ref_log_probs:    [B, seq_len]  参考模型每个 token 的 log 概率
        policy_log_probs: [B, seq_len]  策略模型每个 token 的 log 概率
        mask:             [B, seq_len]  1=有效token，0=PAD
        beta:             float         温度超参
 
    返回：
        loss: 标量
    """
    # ── Step 1：计算每条序列的有效长度 ─────────────────────────────
    # clamp_min(1e-8) 防止全 PAD 时除以 0
    seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)
    # seq_lengths shape: [B, 1]
 
    # ── Step 2：计算每条序列的平均 log 概率 ────────────────────────
    # mask 把 PAD 位置清零，只累加有效 token 的 log 概率，再除以有效长度
    # 结果 shape: [B]
    ref_log_probs = (ref_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    policy_log_probs = (policy_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()
 
    # ── Step 3：切分 chosen 和 rejected ────────────────────────────
    # batch 的前一半是 chosen，后一半是 rejected
    # 例：batch_size=4 → [chosen_0, chosen_1, rejected_0, rejected_1]
    batch_size = ref_log_probs.shape[0]
 
    chosen_ref_log_probs = ref_log_probs[: batch_size // 2] # [B/2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2 :] # [B/2]
    chosen_policy_log_probs = policy_log_probs[: batch_size // 2]  # [B/2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2 :]  # [B/2]
 
    # ── Step 4：计算 log-ratio ──────────────────────────────────────
    # 策略模型：chosen 和 rejected 的 log 概率之差
    # = log π(y⁺|x) - log π(y⁻|x)
    # = log π(y⁺|x) / π(y⁻|x)
    pi_logratios  = chosen_policy_log_probs - reject_policy_log_probs  # [B/2]
 
    # 参考模型：chosen 和 rejected 的 log 概率之差
    # = log π_ref(y⁺|x) - log π_ref(y⁻|x)
    ref_logratios = chosen_ref_log_probs    - reject_ref_log_probs     # [B/2]
 
    # ── Step 5：计算 DPO 损失 ───────────────────────────────────────
    # logits = pi_logratios - ref_logratios
    # 含义：策略模型相对于参考模型，多偏好 chosen 的程度
    #   > 0：策略模型比参考模型更偏好 chosen → loss 较小（好）
    #   < 0：策略模型比参考模型更偏好 rejected → loss 较大（需要纠正）
    logits = pi_logratios - ref_logratios   # [B/2]
 
    # loss = -log σ(β · logits)
    # F.logsigmoid(x) = log(1 / (1 + e^(-x)))，数值稳定
    # 负号：最大化 σ(β·logits) 等价于最小化 -log σ(β·logits)
    loss = -F.logsigmoid(beta * logits)     # [B/2]
 
    # .mean() 对应公式中的期望 E[...]
    return loss.mean()

def train_epoch(epoch, loader, iters, ref_model, lm_config, start_step=0, wandb=None, beta=None):
    """
    DPO 训练一个 epoch。
 
    DPO 和 SFT 的核心区别：
        SFT：只有一个模型，最小化交叉熵 loss
        DPO：同时前向传播策略模型和参考模型
             用两者的 log 概率差计算偏好 loss
    """
    start_time = time.time()
    local_step = 0

    for step, batch in enumerate(loader, start=start_step + 1):
        local_step += 1
        # ── 加载数据到设备 ────────────────────────────────────────────
        # chosen：人类偏好的回答
        # rejected：次优回答
        # mask：标记 assistant 回答部分（loss 只计算这部分）
        # attention_mask：标记非 PAD 位置（传给模型做注意力）
        x_chosen  = batch["x_chosen"].to(args.device)   # [B/2, seq_len] 输入
        x_rejected= batch["x_rejected"].to(args.device) # [B/2, seq_len] 输入
        y_chosen  = batch["y_chosen"].to(args.device)   # [B/2, seq_len] 目标
        y_rejected= batch["y_rejected"].to(args.device) # [B/2, seq_len] 目标
        mask_chosen  = batch["mask_chosen"].to(args.device)    # [B/2, seq_len]
        mask_rejected= batch["mask_rejected"].to(args.device)  # [B/2, seq_len]
        attention_mask_chosen  = batch["attention_mask_chosen"].to(args.device)
        attention_mask_rejected= batch["attention_mask_rejected"].to(args.device)

        # ── 合并 chosen 和 rejected，一次前向传播处理两种数据 ────────
        # 前一半是 chosen，后一半是 rejected
        x = torch.cat([x_chosen, x_rejected],   dim=0)  # [B, seq_len]
        y = torch.cat([y_chosen, y_rejected],   dim=0)  # [B, seq_len]
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)  # [B, seq_len]
        attention_mask = torch.cat([attention_mask_chosen, attention_mask_rejected], dim=0)  # [B, seq_len]

        # ── 动态学习率调整 ────────────────────────────────────────────
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
 
        with autocast_ctx:
            # ── 参考模型前向传播（冻结，不计算梯度）────────────────────
            # 参考模型是 SFT 初始化后冻结的模型，作为 baseline
            # 防止策略模型过度偏离 SFT 模型导致语言能力退化
            with torch.no_grad():
                ref_outputs = ref_model(x, attention_mask=attention_mask)
                ref_logits = ref_outputs.logits # [B, seq_len, vocab_size]
            # 转换为每个 token 的 log 概率
            ref_log_probs = logits_to_log_probs(ref_logits, y) # [B, seq_len]
 
            # ── 策略模型前向传播（需要梯度）──────────────────────────
            outputs = model(x, attention_mask=attention_mask)
            logits = outputs.logits # [B, seq_len, vocab_size]
            policy_log_probs = logits_to_log_probs(logits, y) # [B, seq_len]

            # ── 计算 DPO loss ─────────────────────────────────────────
            # dpo_loss 内部会把 [B, seq_len] 聚合成标量
            dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=args.beta)
 
            # 加上 MoE 辅助损失（非 MoE 模型此项为 0）
            # aux_loss 用于保证 MoE 各专家负载均衡
            loss = dpo_loss_val + outputs.aux_loss
 
            # 梯度累积：把 loss 除以累积步数，等效于更大的 batch
            loss = loss / args.accumulation_steps
 
        # ── 反向传播 ──────────────────────────────────────────────────
        scaler.scale(loss).backward()

        # ── 梯度累积满足条件时更新参数 ───────────────────────────────
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # ── 训练日志 ──────────────────────────────────────────────────
        if step % args.log_interval == 0 or step == iters:
            spend_time   = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr   = optimizer.param_groups[-1]["lr"]
            # ETA 估算：平均每步耗时 × 剩余步数
            eta_min = spend_time / local_step * (iters - step) // 60
 
            Logger(f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:")

            if wandb:
                wandb.log({"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min})

        # ── 模型保存 ──────────────────────────────────────────────────
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = "_moe" if lm_config.use_moe else ""
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
 
            # 取出原始模型（DDP 封装时需要 .module）
            state_dict = (
                model.module.state_dict()
                if isinstance(model, torch.nn.parallel.DistributedDataParallel)
                else model.state_dict()
            )
            # 半精度保存，减小文件体积
            torch.save({k: v.half() for k, v in state_dict.items()}, ckp)
 
            # 保存完整训练状态（用于断点续训）
            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir="../checkpoints",
            )
            model.train()

if __name__ == "__main__":
    # 📚 命令行参数解析
    # argparse: Python标准库，用于解析命令行参数
    # 提供默认值和帮助信息，便于用户配置训练参数
    parser = argparse.ArgumentParser(description="MiniMind DPO")

    # 📚 模型保存相关参数
    # save_dir: 指定LoRA权重和检查点的保存目录
    # lora_name: LoRA权重的标识符，用于区分不同任务的LoRA适配器
    parser.add_argument("--save_dir", type=str, default="../out", help="DPO权重保存目录")
    parser.add_argument("--save_weight", type=str, default="dpo", help="保存权重的前缀名")

    # 📚 训练设备和精度配置
    # device: 指定训练使用的设备（GPU/CPU）
    # dtype: 混合精度训练的数据类型，bfloat16更稳定，float16更高效
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")

    # 📚 训练超参数
    # epochs: 训练的总轮数，控制模型训练的完整程度
    # batch_size: 每个批次的样本数量，影响显存使用和训练稳定性
    # learning_rate: 初始学习率，控制参数更新的步长
    # DPO 学习率要远小于 SFT（建议 ≤ 5e-8）
    # 太大会导致策略模型遗忘 SFT 阶段学到的知识
    parser.add_argument("--learning_rate", type=float, default=4e-8, help="初始学习率")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    # beta：控制偏离参考模型的惩罚力度
    # beta 越大：离参考模型越近，但偏好学习效果越弱
    # beta 越小：偏好学习更激进，但可能遗忘 SFT 知识
    # 常用范围：0.1 ~ 0.5
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta 超参")

    # 📚 数据加载和训练优化
    # num_workers: 数据加载的并行进程数，提高数据读取效率
    # accumulation_steps: 梯度累积步数，模拟更大的batch size
    # grad_clip: 梯度裁剪阈值，防止梯度爆炸
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")

    # 📚 日志和保存配置
    # log_interval: 每多少步打印一次训练日志
    # save_interval: 每多少步保存一次模型检查点
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")

    # 📚 模型架构参数
    # hidden_size: 模型隐藏层维度，影响模型容量和计算复杂度
    # num_hidden_layers: Transformer层数，层数越多模型越深
    # max_seq_len: 训练时序列的最大长度，影响显存使用
    # use_moe: 是否使用Mixture of Experts架构，提高模型效率
    parser.add_argument("--hidden_size", type=int, default=512, help="模型隐藏层维度")
    parser.add_argument("--num_hidden_layers", type=int, default=8, help="Transformer层数")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="训练的最大截断长度")
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")

    # 📚 数据和权重配置
    # data_path: 训练数据的文件路径，通常是JSONL格式
    # from_weight: 基于哪个预训练权重进行LoRA微调
    # from_resume: 是否从检查点恢复训练，支持断点续训
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl", help="训练数据路径")
    # DPO 基于 SFT 模型进行对齐优化，from_weight 通常是 "full_sft"
    parser.add_argument("--from_weight", default="full_sft", type=str, help="基于哪个权重训练，默认full_sft")
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")

    # 📚 实验跟踪配置
    # use_wandb: 是否启用WandB/SwanLab进行实验跟踪
    # wandb_project: WandB项目的名称，用于组织实验
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-LoRA", help="wandb项目名")

    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # 📚 分布式训练初始化
    # init_distributed_mode(): 初始化多GPU分布式训练环境
    # 如果使用多卡，会设置进程组和本地rank
    local_rank = init_distributed_mode()

    # 📚 设备分配
    # 在分布式训练中，每个进程使用不同的GPU
    # dist.get_rank(): 获取当前进程的全局rank
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"

    # 📚 随机种子设置
    # setup_seed(): 设置随机种子，确保训练的可复现性
    # 不同进程使用不同的种子，避免生成相同的数据
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    # 📚 创建保存目录
    # os.makedirs: 递归创建目录，如果已存在则忽略
    os.makedirs(args.save_dir, exist_ok=True)

    # 📚 模型配置初始化
    # MiniMindConfig: 定义模型的超参数，如隐藏维度、层数、是否使用MoE
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )

    # 📚 检查点检测
    # lm_checkpoint(): 检查是否存在可用的检查点
    # 如果from_resume=1，则尝试加载之前的训练状态
    ckp_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir="../checkpoints")
        if args.from_resume == 1
        else None
    )

    # ========== 3. 设置混合精度 ==========
    # 📚 设备类型判断
    # 根据设备字符串判断是CPU还是GPU
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # 📚 数据类型选择
    # bfloat16: 更好的数值稳定性，适合现代GPU
    # float16: 更高的性能，但可能有精度损失
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    # 📚 自动混合精度上下文
    # autocast: 自动选择合适的精度进行计算
    # CPU模式下使用nullcontext（无操作）
    autocast_ctx = (nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=dtype))

    # ========== 4. 配置wandb ==========
    # 📚 实验跟踪初始化
    # SwanLab: 类似WandB的实验管理工具
    # 支持实验重启和指标记录
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        # 📚 WandB运行ID
        # 从检查点恢复时使用相同的ID，保持实验连续性
        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None

        # 📚 实验名称生成
        # 包含关键参数，便于识别不同的实验配置
        wandb_run_name = f"MokioMind-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化策略模型和参考模型 ==========
    # 📚 模型初始化
    # init_model(): 加载预训练模型和tokenizer
    # from_weight指定基础权重文件
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    Logger(f"策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M")

    # 参考模型（reference model）：冻结的 baseline
    # 与策略模型初始权重完全相同，但整个训练过程不更新
    # 作用：正则化项，防止策略模型过度优化偏好而遗忘语言能力
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model.eval() # 切换到推理模式（关闭 dropout 等）
    ref_model.requires_grad_(False) # 冻结所有参数，不参与反向传播
    Logger(f"参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M")

    # ========== 6. 数据集和优化器 ==========
    # DPODataset 返回 chosen/rejected 配对数据
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.amp.GradScaler(device=device_type, enabled=(args.dtype == "float16"))
    # DPO 只优化策略模型，参考模型不传给 optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 7. 从检查点恢复训练状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step  = ckp_data.get("step", 0)

    # ========== 8. DDP 包装策略模型 ==========
    # 参考模型不需要 DDP（不计算梯度，不需要跨卡同步）
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 9. 开始训练 ==========
    # 📚 训练循环
    # 遍历每个epoch，执行训练过程
    # 支持从检查点恢复，继续未完成的训练
    for epoch in range(start_epoch, args.epochs):
        # 📚 采样器epoch设置
        # set_epoch(): 确保分布式采样器的随机性
        train_sampler and train_sampler.set_epoch(epoch)
        
        # 第一个epoch且存在检查点
        if epoch == start_epoch and start_step > 0:
            # 📚 跳过已完成的step
            # SkipBatchSampler: 自定义采样器，跳过前N个batch
            # 用于断点续训时从指定step开始
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始")
            train_epoch(epoch, loader, len(loader), lora_params, start_step=start_step, wandb=wandb)
        else:
            # 📚 默认从头开始
            # 标准数据加载器
            # DataLoader: PyTorch的数据加载器
            # shuffle: 单GPU时随机打乱，多GPU时由sampler控制
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            train_epoch(epoch, loader, len(loader), lora_params, 0, wandb)