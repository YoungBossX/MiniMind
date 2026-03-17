"""
=============================================================================
推理模型数据蒸馏训练脚本 (train_reason_improved.py)
=============================================================================

目标: 让模型学会 DeepSeek-R1 风格的 "先思考再回答" 格式:`
      <think>\n思考过程\n</think>\n<answer>\n最终回答\n</answer>

方法: 本质上就是 SFT (有监督微调), 只是:
  1. 训练数据来自大模型的推理输出 (数据蒸馏, 不是模型蒸馏)
  2. 对 <think>/<answer> 等格式标签施加更高的 loss 权重,
     强迫模型学会输出正确的格式

训练链路: pretrain → full_sft → dpo → reason (本脚本)
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.MiniMindModel import MiniMindConfig
from dataset.llm_dataset import SFTDataset
from trainer.trainer_utils import (
    get_lr, Logger, is_main_process, lm_checkpoint,
    init_distributed_mode, setup_seed, init_model, SkipBatchSampler
)

warnings.filterwarnings('ignore')


# ==========================================================================
#  序列匹配工具函数
# ==========================================================================
def find_subsequence_positions(sequence, pattern):
    """
    在 1D tensor sequence 中查找 pattern 子序列的所有出现位置.

    例如:
      sequence = [1, 2, 30, 450, 1573, 32, 5, 6]    ← 完整的 input token 序列
      pattern  = [30, 450, 1573, 32]                  ← <think> 的 token ids
      返回: [[2, 3, 4, 5]]                            ← 这 4 个位置构成 <think>

    Args:
        sequence: [T] 1D tensor (一条序列的 token ids)
        pattern:  [P] 1D tensor (要匹配的标签 token ids)

    Returns:
        list[list[int]]: 每次匹配的位置列表
    """
    positions = []
    seq_len = len(sequence)
    pat_len = len(pattern)
    if pat_len == 0 or seq_len < pat_len:
        return positions

    for i in range(seq_len - pat_len + 1):
        if torch.equal(sequence[i:i + pat_len], pattern):
            positions.append(list(range(i, i + pat_len)))

    return positions


def build_tag_penalty_mask(shift_labels, tag_id_seqs, penalty_weight=10.0, device='cpu'):
    """
    构建标签惩罚 mask: 只在完整标签序列出现的位置施加额外权重.

    Args:
        shift_labels:   [B, T] 移位后的 labels (已经是 input_ids[1:])
        tag_id_seqs:    list[tensor], 每个元素是一个标签的完整 token id 序列
                        例如 [tensor([30,450,1573,32]), tensor([5540,450,1573,32]), ...]
        penalty_weight: 标签位置的 loss 权重倍数
        device:         设备

    Returns:
        penalty_mask: [B, T] float tensor, 正常位置=1.0, 标签位置=penalty_weight
        tag_hit_count: int, 命中的标签 token 总数 (用于日志)
    """
    B, T = shift_labels.shape
    penalty_mask = torch.ones(B, T, device=device)
    tag_hit_count = 0

    for b in range(B):
        seq = shift_labels[b]  # [T]
        for pattern in tag_id_seqs:
            matches = find_subsequence_positions(seq, pattern)
            for pos_list in matches:
                for p in pos_list:
                    if p < T:
                        penalty_mask[b, p] = penalty_weight
                        tag_hit_count += 1

    return penalty_mask, tag_hit_count


# ==========================================================================
#  训练一个 Epoch
# ==========================================================================
def train_epoch(epoch, loader, iters, lm_config, tag_id_seqs,
                start_step=0, wandb=None):
    """
    训练循环. 与普通 SFT 的区别仅在于: 对格式标签施加更高的 loss 权重.

    Args:
        tag_id_seqs: 预计算的标签 token id 序列列表 (在 main 中创建, 传进来复用)
    """
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()

    for step, (input_ids, labels, loss_mask, attention_mask) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        loss_mask = loss_mask.to(args.device)
        attention_mask = attention_mask.to(args.device)
        # 手动调整学习率 (余弦退火)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ---- Forward ----
        with autocast_ctx:
            res = model(input_ids)
            # 语言模型预测下一个 token, 所以 logits 和 labels 要错开一位
            shift_logits = res.logits[..., :-1, :].contiguous()  # [B, T, V]
            shift_labels = labels[..., 1:].contiguous()           # [B, T]

            # 计算 per-token cross entropy loss
            loss_raw = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).view(shift_labels.size())  # [B, T]

            # ---- 构建 loss mask ----
            # 基础 mask: labels=-100 的位置不参与 loss (prompt 部分和 padding)
            loss_mask = (shift_labels != -100).float()  # [B, T]

            penalty_mask, tag_hit_count = build_tag_penalty_mask(
                shift_labels, tag_id_seqs,
                penalty_weight=args.tag_penalty_weight,
                device=args.device
            )

            # 合并: 基础 mask × 惩罚 mask
            # 效果: prompt/pad 位置 = 0, 正常 response 位置 = 1, 标签位置 = penalty_weight
            weighted_mask = loss_mask * penalty_mask  # [B, T]

            # 计算加权 loss
            # 分母用未加权的 token 数, 这样标签的 penalty_weight 倍效果才能体现
            valid_token_count = loss_mask.sum()
            logits_loss = (loss_raw * weighted_mask).sum() / (valid_token_count + 1e-8)

            # 加上 MoE 的负载均衡 loss (非 MoE 模型为 0)
            loss = logits_loss + res.aux_loss
            loss = loss / args.accumulation_steps

        # ---- Backward ----
        scaler.scale(loss).backward()

        # ---- 梯度步 (支持梯度累积) ----
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # ---- 日志 ----
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = logits_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60

            # 标签命中率: 每个 batch 平均命中多少个标签 token
            tag_ratio = tag_hit_count / max(valid_token_count.item(), 1)

            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                f'loss:{current_loss:.4f}, logits:{current_logits_loss:.4f}, '
                f'aux:{current_aux_loss:.4f}, '
                f'tag_hits:{tag_hit_count}, tag_ratio:{tag_ratio:.4f}, '
                f'lr:{current_lr:.8f}, eta:{eta_min:.1f}min'
            )
            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "logits_loss": current_logits_loss,
                    "aux_loss": current_aux_loss,
                    "tag_hit_count": tag_hit_count,
                    "tag_hit_ratio": tag_ratio,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min
                })

        # ---- 保存 checkpoint ----
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(
                lm_config, weight=args.save_weight, model=model,
                optimizer=optimizer, scaler=scaler, epoch=epoch, step=step,
                wandb=wandb, save_dir='../checkpoints'
            )
            model.train()
            del state_dict

        del input_ids, labels, res, loss


# ==========================================================================
#  入口
# ==========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Reasoning Distillation (Improved)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='reason', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=2, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=720, type=int, help="训练的最大截断长度 (中文 1 token ≈ 1.5~1.7 字符)")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument("--data_path", type=str, default="../dataset/r1_mix_1024.jsonl", help="推理蒸馏数据路径")
    parser.add_argument('--from_weight', default='dpo', type=str, help="基于哪个权重训练 (默认 dpo, 即 pretrain→sft→dpo 之后)")
    parser.add_argument('--tag_penalty_weight', default=10.0, type=float, help="格式标签的 loss 惩罚倍数 (默认 10, 越大越强迫模型输出正确格式)")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Reasoning")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1])
    args = parser.parse_args()

    # ========== 1. 初始化 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置 ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe)
    )
    ckp_data = (lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints')
                if args.from_resume == 1 else None)

    # ========== 3. 混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Reason-E{args.epochs}-BS{args.batch_size}-LR{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')

    # 每个标签被 tokenizer 拆成的完整 id 序列
    TAG_STRINGS = ['<think>', '</think>', '<answer>', '</answer>']
    tag_id_seqs = []
    for tag in TAG_STRINGS:
        ids = tokenizer(tag, add_special_tokens=False).input_ids
        tag_id_seqs.append(torch.tensor(ids, dtype=torch.long, device=args.device))
        Logger(f'  Tag {tag:15s} → ids={ids} (len={len(ids)})')
    Logger(f'  Tag penalty weight: {args.tag_penalty_weight}x')

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.amp.GradScaler(device=device_type, enabled=(args.dtype == "float16"))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 恢复训练状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. DDP ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler,
                            num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}步, 从 step {start_step + 1} 开始')
            train_epoch(epoch, loader, len(loader) + skip, lm_config,
                        tag_id_seqs, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), lm_config,
                        tag_id_seqs, 0, wandb)

    if is_main_process():
        Logger("Training finished. Saving final checkpoint...")
        model.eval()
        moe_suffix = '_moe' if lm_config.use_moe else ''
        ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
        lm_checkpoint(
            lm_config, weight=args.save_weight, model=model,
            optimizer=optimizer, scaler=scaler, epoch=args.epochs, step=0,
            wandb=wandb, save_dir='../checkpoints'
        )
        Logger(f"Final model saved to {ckp}")

    # ========== 9. 清理 ==========
    if dist.is_initialized():
        dist.destroy_process_group()