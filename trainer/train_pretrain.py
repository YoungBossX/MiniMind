import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
# 计时，用于计算每个 epoch 的预计剩余时间
import time
import warnings
import torch
# 分布式训练通信库
import torch.distributed as dist
# 空上下文管理器（CPU 不支持 autocast 时使用）
from contextlib import nullcontext
# 优化器模块
from torch import optim
# DDP 多卡并行封装
from torch.nn.parallel import DistributedDataParallel
# 数据加载和分布式采样
from torch.utils.data import DataLoader, DistributedSampler
# 模型配置类
from model.MiniMindModel import MiniMindConfig
# 预训练数据集类
from dataset.llm_dataset import PretrainDataset
# 动态计算学习率（通常是 warmup + cosine decay）
# 日志打印（只在主进程打印，避免多卡重复输出）
# 判断当前进程是否是主进程（rank 0）
# 保存 / 加载完整训练状态（断点续训）
# 初始化分布式环境（读取 RANK、LOCAL_RANK 等环境变量）
# 设置随机种子（torch / numpy / random）
# 根据配置创建并初始化模型
# 自定义采样器，用于跳过已经训练过的 batch（断点续训）
from trainer.trainer_utils import (
    get_lr,
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    init_model,
    SkipBatchSampler,
)

# 忽略警告信息，保持输出清洁
warnings.filterwarnings("ignore")

# ============================================================
# 核心训练函数：执行单个 epoch 的前向传播、反向传播、参数更新
# ============================================================
def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    Args:
        epoch      : 当前 epoch 编号（从 0 开始）
        loader     : DataLoader 对象，迭代产出 (input_ids, labels, attention_mask)
        iters      : 本 epoch 总 step 数（用于进度日志 & ETA 计算）
        start_step : 断点续训时从哪个 step 开始（第一次训练为 0）
        wandb      : 实验跟踪对象（swanlab / wandb），None 表示不使用
    """
    # 记录 epoch 开始时间，用于计算 ETA
    start_time = time.time()
    local_step = 0
    
    # enumerate(loader, start=start_step+1)：
    #   - loader 每次迭代返回一个 batch
    #   - start 参数让 step 从 start_step+1 开始计数，与断点续训对齐
    # DataLoader 把多条样本拼成 batch，每条样本包含 input_ids、labels、attention_mask
    for step, (input_ids, labels, loss_mask, attention_mask) in enumerate(
        loader, start=start_step + 1
    ):  
        local_step += 1
        # 将数据搬到指定设备（GPU 或 CPU）
        input_ids = input_ids.to(args.device) # 形状: (batch_size, seq_len)
        labels = labels.to(args.device) # 形状: (batch_size, seq_len)，是 input_ids 向右移一位
        loss_mask = loss_mask.to(args.device) # 形状: (batch_size, seq_len)，1 = 需要计算 loss，0 = 跳过
        attention_mask = attention_mask.to(args.device) # 形状: (batch_size, seq_len)，1 = 有效 token，0 = padding
        # ── 动态学习率计算 ──────────────────────────────────────────────
        # get_lr 通常实现 warmup + cosine decay 调度：
        #   - 前 warmup 步线性从 0 升到 learning_rate
        #   - 之后按余弦曲线下降到 min_lr
        # epoch * iters + step = 全局 step 编号
        # args.epochs * iters  = 总 step 数（用于 cosine decay 终点）
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)

        # 将新学习率写入优化器所有参数组（AdamW 通常只有一个参数组）
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        # ── 前向传播（在混合精度上下文中执行）──────────────────────────
        # autocast_ctx：
        #   - GPU: torch.amp.autocast，自动把合适的运算降为 bfloat16/float16
        #   - CPU: nullcontext（空操作），不做任何精度转换
        with autocast_ctx:
            # 调用模型前向，返回包含 .loss 的输出对象
            # labels 不为 None 时，模型内部会计算交叉熵损失
            res = model(input_ids, attention_mask=attention_mask, labels=labels, loss_mask=loss_mask)

            # 预训练任务的总 loss = 主 loss + 辅助 loss
            loss = (res.loss + res.aux_loss)
            
            # 梯度累积：把 loss 除以累积步数
            # 目的：模拟更大 batch_size，节省显存
            loss = loss / args.accumulation_steps

        # ── 反向传播 ──────────────────────────────────────────────────
        # scaler.scale(loss)：将 loss 乘以缩放因子（防止 float16 下梯度下溢）
        # .backward()：计算所有参数的梯度，并累加到 .grad 上
        scaler.scale(loss).backward()

        # ── 参数更新（每 accumulation_steps 步执行一次）───────────────
        if step % args.accumulation_steps == 0:
            # unscale_：将梯度除回缩放因子，还原真实梯度值
            # 必须在 clip_grad_norm_ 之前调用，否则裁剪的是放大后的梯度
            scaler.unscale_(optimizer)

            # 梯度裁剪：将所有参数梯度的 L2 范数限制在 grad_clip 以内
            # 防止梯度爆炸，稳定训练
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # scaler.step(optimizer)：
            #   - 检查梯度是否含有 inf/nan（float16 溢出时出现）
            #   - 若正常：调用 optimizer.step() 更新参数
            #   - 若异常：跳过本次更新，避免参数被污染
            scaler.step(optimizer)

            # scaler.update()：
            #   - 根据上一步是否发生溢出，动态调整缩放因子
            #   - 溢出 → 缩小缩放因子；连续正常 → 逐渐放大缩放因子
            scaler.update()

            # 清空梯度，set_to_none=True 比 zero_grad() 更省显存
            # （直接释放梯度张量，而非填充 0）
            optimizer.zero_grad(set_to_none=True)

        # ── 日志打印 ─────────────────────────────────────────────────
        if step % args.log_interval == 0 or step == iters:
            # 已用时间（秒）
            spend_time = time.time() - start_time
            # 恢复真实 loss（之前除以了 accumulation_steps）
            current_loss = loss.item() * args.accumulation_steps
            # 当前实际学习率
            current_lr = optimizer.param_groups[-1]["lr"]

            # 预计剩余时间（分钟）：
            # spend_time / (step+1) = 每 step 平均耗时
            # * iters = 本 epoch 总耗时估计
            # // 60 - spend_time // 60 = 剩余分钟数
            eta_min = spend_time / local_step * (iters - step) // 60

            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:"
            )

            # 上报指标到实验跟踪系统（如 SwanLab / WandB）
            if wandb:
                wandb.log(
                    {"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min}
                )

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval() # 切换到评估模式

            # 构建保存文件路径
            # MoE 模型额外加 "_moe" 后缀，方便区分
            moe_suffix = (
                "_moe" if hasattr(lm_config, "use_moe") and lm_config.use_moe else ""
            )
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"

            # DDP 模型有额外封装层，真正的模型在 .module 属性里
            # 非 DDP 模型直接调用 state_dict()
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            # 将所有参数从 float32 转为 float16（半精度）再保存
            # 优点：文件大小减半，加载更快
            # 代价：精度略有损失（推理时通常可以接受）
            state_dict = {k: v.half() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)

            # 保存完整训练状态（用于断点续训）：
            # 包含 model weights、optimizer state、scaler state、epoch、step、wandb_id
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
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")

    # ========== 基础训练参数 ==========
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument("--save_weight", default="pretrain", type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数（建议1轮zero或2-6轮充分训练）")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")

    # ========== 硬件和性能参数 ==========
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")

    # ========== 训练策略参数 ==========
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")

    # ========== 模型架构参数 ==========
    parser.add_argument("--hidden_size", default=512, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument("--max_seq_len", default=512, type=int, help="训练的最大截断长度")
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")

    # ========== 数据和恢复参数 ==========
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl", help="预训练数据路径",)
    parser.add_argument("--from_weight", default="none", type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument("--from_resume", default=1, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")

    # ========== 实验跟踪参数 ==========
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")

    # 解析命令行参数
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    """
    📚 分布式训练初始化知识点：
    - local_rank: 当前进程在本机上的GPU编号
    - 随机种子: 确保不同进程有不同但可复现的随机序列
    - 这样既保证了随机性，又保证了可复现性
    """
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"  # 分布式训练时使用对应的GPU

    # 📚 随机种子设置知识点
    # 不同进程使用不同的种子，避免数据采样完全相同
    # 42是基础种子，每个进程加上自己的rank保证不同
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查点 ==========
    """
    📚 模型配置和检查点管理：
    - 创建保存目录
    - 构建模型配置对象
    - 尝试加载断点续训数据
    """
    os.makedirs(args.save_dir, exist_ok=True) # 确保保存目录存在

    # 创建MiniMind模型配置
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )

    # 📚 断点续训知识点
    # 如果开启了断点续训，尝试加载之前的训练状态
    ckp_data = (
        lm_checkpoint(
            lm_config, weight=args.save_weight, save_dir="../checkpoints"
        )
        if args.from_resume == 1
        else None
    )

    # ========== 3. 设置混合精度 ==========
    """
    📚 混合精度训练知识点：
    - bfloat16: Google开发，数值范围大，更稳定
    - float16: 标准半精度，节省内存但可能溢出
    - autocast: 自动选择精度，关键运算用float32
    """
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    # 📚 上下文管理器知识点
    # CPU不支持autocast，使用nullcontext作为空操作
    autocast_ctx = (nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=dtype))

    # ========== 4. 配置WandB实验跟踪 ==========
    """
    📚 实验跟踪系统知识点：
    - WandB: 实验管理平台，记录训练过程
    - SwanLab: 国产替代方案
    - 支持断点续训时恢复到同一个实验
    """
    wandb = None
    if args.use_wandb and is_main_process():
        # 使用SwanLab作为WandB的替代
        import swanlab as wandb

        # 📚 实验恢复知识点
        # 如果有检查点数据，获取之前的wandb_id来恢复实验
        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        # 必须恢复到指定实验
        resume = "must" if wandb_id else None

        # 构建实验名称，包含关键超参数
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(
            project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume
        )

    # ========== 5. 定义模型、数据、优化器 ==========
    """
    📚 训练组件初始化：
    - 模型: 根据配置创建MiniMind模型
    - 数据集: 加载预训练数据
    - 采样器: 分布式训练的数据分配
    - 优化器: AdamW优化器
    - 缩放器: 混合精度训练的梯度缩放
    """
    # 初始化模型和分词器
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)

    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    scaler = torch.amp.GradScaler(device=device_type, enabled=(args.dtype == "float16"))

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_epoch, start_step = 0, 0
    if ckp_data:
        # 恢复模型参数
        model.load_state_dict(ckp_data["model"])
        # 恢复优化器状态（动量、方差估计等）
        optimizer.load_state_dict(ckp_data["optimizer"])
        # 恢复梯度缩放器状态
        scaler.load_state_dict(ckp_data["scaler"])
        # 恢复训练进度
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    if dist.is_initialized():
        # 📚 RoPE位置编码特殊处理
        # freqs_cos, freqs_sin是位置编码缓存，不需要梯度同步
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    for epoch in range(start_epoch, args.epochs):
        # 📚 分布式采样器epoch设置
        # 每个epoch设置不同的随机种子，确保数据顺序随机化
        if train_sampler:
            train_sampler.set_epoch(epoch)

        # 📚 断点续训逻辑
        if epoch == start_epoch and start_step > 0: # 第一个epoch且存在检查点
            # 使用跳批采样器，跳过已训练的数据
            batch_sampler = SkipBatchSampler(
                train_sampler or range(len(train_ds)), args.batch_size, start_step
            )
            loader = DataLoader(
                train_ds,
                batch_sampler=batch_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            Logger(
                f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始"
            )
            train_epoch(epoch, loader, len(loader) + start_step, start_step, wandb)
        else:  # 默认从头开始
            loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            train_epoch(epoch, loader, len(loader), 0, wandb)