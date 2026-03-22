import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import re
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.MiniMindModel import MiniMindConfig, MiniMindForCausalLM
from dataset.llm_dataset import RLAIFDataset
from trainer.trainer_utils import (
    Logger, is_main_process, lm_checkpoint,
    init_distributed_mode, setup_seed, SkipBatchSampler, init_model
)
 
warnings.filterwarnings('ignore')

# ==========================================================================
#  Per-token log-prob 计算 
# ==========================================================================
def get_per_token_logps(mdl, input_ids, num_completion_tokens):
    """
    计算 completion 部分每个 token 的 log-probability.
    
    Args:
        mdl: 模型
        input_ids: [B*G, P+R] 完整序列 (prompt+response)
        num_completion_tokens: R, response 部分的 token 数
    Returns:
        per_token_logps: [B*G, R] 每个 response token 的 log-prob
    """
    ids = input_ids.detach().clone()
    logits = mdl(ids, logits_to_keep=num_completion_tokens + 1).logits
    logits = logits[:, :-1, :]
    completion_ids = ids[:, -num_completion_tokens:]
    log_probs = F.log_softmax(logits, dim=-1)
    per_token_logps = log_probs.gather(2, completion_ids.unsqueeze(-1)).squeeze(-1)
    entropy_per_token = -(log_probs.exp() * log_probs).sum(dim=-1)
    return per_token_logps, entropy_per_token

# ==========================================================================
#  Reward 计算
# ==========================================================================
def calculate_rewards(prompts, responses, reward_model, reward_tokenizer, args):
    rewards = torch.zeros(len(responses), device=args.device)
    
    # 格式奖励 (仅 reasoning 模式)
    if args.reasoning == 1:
        pattern1 = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"
        for idx, resp in enumerate(responses):
            if re.match(pattern1, resp, re.S) or re.match(pattern2, resp, re.S):
                rewards[idx] += 0.5
            for tag in ["<think>", "</think>", "<answer>", "</answer>"]:
                if resp.count(tag) == 1:
                    rewards[idx] += 0.25

    # Reward model 打分
    with torch.no_grad():
        batch_size = len(prompts)
        scale = 3.0
        for i in range(batch_size):
            for j in range(args.num_generations):
                resp_idx = i * args.num_generations + j
                prompt = prompts[i]
                response = responses[resp_idx]
                
                pat = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pat, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                
                tmp_chat = messages + [{"role": "assistant", "content": response}]
                score = reward_model.get_score(reward_tokenizer, tmp_chat)
                score = max(min(score, scale), -scale)
                
                if args.reasoning == 1:
                    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                    if answer_match:
                        answer_content = answer_match.group(1).strip()
                        tmp_chat2 = messages + [{"role": "assistant", "content": answer_content}]
                        answer_score = reward_model.get_score(reward_tokenizer, tmp_chat2)
                        answer_score = max(min(answer_score, scale), -scale)
                        score = score * 0.4 + answer_score * 0.6
                
                rewards[resp_idx] += score
 
    return rewards

# ==========================================================================
#  GRPO 训练一个 Epoch
# ==========================================================================
def grpo_train_epoch(epoch, loader, iters, ref_model, reward_model, reward_tokenizer,
                     start_step=0, wandb=None):
    model.train()
    
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch['prompt']  # list[str], length B
        
        # ---- 1. 编码 prompt ----
        prompt_inputs = tokenizer(
            prompts, return_tensors="pt", padding=True,
            return_token_type_ids=False, padding_side="left",
            add_special_tokens=False
        ).to(args.device)
        
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]
 
        # ---- 2. 生成 N 个回答 ----
        with torch.no_grad():
            model_for_gen = model.module if isinstance(model, DistributedDataParallel) else model
            # num_return_sequences=G: 每个 prompt 生成 G 个不同的回答
            # 输出 shape: [B*G, P+R]
            outputs = model_for_gen.generate(
                **prompt_inputs,
                max_new_tokens=args.max_gen_len,
                do_sample=True,
                temperature=0.8,
                num_return_sequences=args.num_generations,
                pad_token_id=tokenizer.pad_token_id
            )
 
        prompt_len = prompt_inputs["input_ids"].size(1)
        completion_ids = outputs[:, prompt_len:]  # [B*G, R]
        R = completion_ids.size(1)
        
        if R == 0:
            del prompt_inputs, outputs, completion_ids
            continue
 
        # ---- 3. 计算 actor 和 ref 的 per-token log-prob ----
        with autocast_ctx:
            actor_logps, entropy_per_token = get_per_token_logps(model, outputs, R)
            res = model(outputs) if lm_config.use_moe else None
            aux_loss = res.aux_loss if res is not None else torch.tensor(0.0, device=args.device)
 
        with torch.no_grad():
            ref_logps, _ = get_per_token_logps(ref_model, outputs, R)  # [B*G, R]
 
        # ---- 4. 计算 reward 和 advantage ----
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        rewards = calculate_rewards(prompts, completions, reward_model, reward_tokenizer, args)  # [B*G]
 
        # 组内相对 advantage: 每组 G 个回答, 用组内均值和标准差标准化
        G = args.num_generations
        grouped_rewards = rewards.view(-1, G)  # [B, G]
        mean_r = grouped_rewards.mean(dim=1, keepdim=True)  # [B, 1]
        std_r = grouped_rewards.std(dim=1, keepdim=True)    # [B, 1]
 
        # 此时 advantage 无意义, 应该置零 (不学习)
        degenerate_mask = (std_r < 1e-4).squeeze(1)  # [B] bool
        degenerate_ratio = degenerate_mask.float().mean().item()
        
        # 组内标准化
        advantages = (grouped_rewards - mean_r) / (std_r + 1e-4)  # [B, G]
        advantages = advantages.clamp(-10, 10)
        # 退化组的 advantage 置零
        advantages[degenerate_mask] = 0.0
        advantages = advantages.view(-1)  # [B*G]
 
        # ---- 5. Completion mask (只在有效 token 上计算 loss) ----
        is_eos = (completion_ids == tokenizer.eos_token_id)
        eos_idx = torch.full((is_eos.size(0),), R, dtype=torch.long, device=args.device)
        has_eos = is_eos.any(dim=1)
        eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
        # mask: 从第一个 token 到 eos (含) 为 1, 之后为 0
        completion_mask = (torch.arange(R, device=args.device).unsqueeze(0) <= eos_idx.unsqueeze(1)).float()
 
        # ---- 6. KL 散度 ----
        log_ratio = actor_logps - ref_logps  # log(π/π_ref)
        per_token_kl = torch.exp(log_ratio) - log_ratio - 1  # ≥ 0
 
        # ---- 7. Policy loss (带 clip) ----
        # ratio = π_new / π_old, 由于 GRPO 每步只 forward 一次, ratio 通过
        # exp(logp - logp.detach()) 实现, 梯度只流过分子
        ratio = torch.exp(actor_logps - actor_logps.detach())  # [B*G, R]
        adv = advantages.unsqueeze(1)  # [B*G, 1] broadcast to [B*G, R]
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * adv
        per_token_loss = -torch.min(surr1, surr2) + args.beta * per_token_kl

        entropy = (entropy_per_token * completion_mask).sum() / (completion_mask.sum() + 1e-8)
 
        # 加权平均 (每个序列除以自己的有效 token 数, 再取 batch 均值)
        seq_lengths = completion_mask.sum(dim=1).clamp(min=1)
        policy_loss = (per_token_loss * completion_mask).sum(dim=1) / seq_lengths
        policy_loss = policy_loss.mean()
 
        loss = (policy_loss - args.entropy_coef * entropy + aux_loss) / args.accumulation_steps
        loss.backward()
 
        # ---- 8. 梯度步 ----
        if (step + 1) % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
 
        # ---- 9. 日志 ----
        if step % args.log_interval == 0 or step == iters:
            # 诊断指标
            with torch.no_grad():
                clip_frac = ((ratio - 1.0).abs() > args.clip_epsilon).float()
                clip_frac = (clip_frac * completion_mask).sum() / (completion_mask.sum() + 1e-8)
                avg_kl = (per_token_kl * completion_mask).sum() / (completion_mask.sum() + 1e-8)
 
            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                f'Loss:{policy_loss.item():.4f}, Aux:{aux_loss.item():.4f}, '
                f'Reward:{rewards.mean().item():.4f}, '
                f'KL:{avg_kl.item():.4f}, ClipFrac:{clip_frac.item():.3f}, '
                f'Entropy:{entropy.item():.4f}, '
                f'DegenerateRatio:{degenerate_ratio:.3f}, '
                f'AvgLen:{seq_lengths.mean().item():.1f}, '
                f'LR:{optimizer.param_groups[0]["lr"]:.2e}'
            )
 
            if wandb and is_main_process():
                wandb.log({
                    "policy_loss": policy_loss.item(),
                    "aux_loss": aux_loss.item(),
                    "reward": rewards.mean().item(),
                    "kl": avg_kl.item(),
                    "clip_fraction": clip_frac.item(),
                    "entropy": entropy.item(),
                    "degenerate_ratio": degenerate_ratio,
                    "avg_response_len": seq_lengths.mean().item(),
                    "advantages_mean": advantages.mean().item(),
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
 
        # ---- 10. 保存 ----
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                          epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints',
                          scheduler=scheduler)
            model.train()
            del state_dict
 
        del prompt_inputs, outputs, completion_ids, actor_logps, ref_logps
        del completions, rewards, advantages, completion_mask
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind GRPO")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument('--save_weight', default='grpo', type=str)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument('--max_seq_len', default=66, type=int, help="Prompt 最大长度")
    parser.add_argument("--max_gen_len", type=int, default=512, help="生成最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif-mini.jsonl")
    parser.add_argument("--num_generations", type=int, default=4,  help="每个 prompt 生成几个回答 (G). 越大 advantage 越稳定, 但越慢")
    parser.add_argument("--beta", type=float, default=0.02, help="KL 惩罚系数")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO 风格 clip 参数")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy bonus 系数")
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1])
    parser.add_argument("--reward_model_path", type=str, default="../internlm2-1_8b-reward")
    parser.add_argument('--from_resume', default=1, type=int, choices=[0, 1])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-GRPO")
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
        wandb_run_name = f"MiniMind-GRPO-E{args.epochs}-BS{args.batch_size}-G{args.num_generations}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
 
    # ========== 5. 模型 ==========
    base_weight = "reason" if args.reasoning == 1 else "full_sft"
    
    # GRPO 只需要 2+1 个模型
    model, tokenizer = init_model(lm_config, base_weight, save_dir=args.save_dir, device=args.device)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    
    ref_model, _ = init_model(lm_config, base_weight, save_dir=args.save_dir, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    
    reward_model = AutoModel.from_pretrained(args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True)
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)
 
    # ========== 6. 数据和优化器 ==========
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len))
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_steps = (iters // args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.learning_rate / 10)
 
    # ========== 7. 恢复 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
 
    # ========== 8. DDP ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
 
    # ========== 9. 训练 ==========
    Logger("=" * 70)
    Logger(f"GRPO Training | Epochs:{args.epochs} | Batch:{args.batch_size} | G:{args.num_generations}")
    Logger(f"Clip:{args.clip_epsilon} | Beta:{args.beta} | Entropy:{args.entropy_coef}")
    Logger("=" * 70)
 
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler,
                            num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}步')
            grpo_train_epoch(epoch, loader, len(loader) + skip, ref_model,
                             reward_model, reward_tokenizer, start_step, wandb)
        else:
            grpo_train_epoch(epoch, loader, len(loader), ref_model,
                             reward_model, reward_tokenizer, 0, wandb)
            
    if is_main_process():
        Logger("Training finished. Saving final checkpoint...")
        model.eval()
        moe_suffix = '_moe' if lm_config.use_moe else ''
        ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
        lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                      epoch=args.epochs, step=0, wandb=wandb, save_dir='../checkpoints',
                      scheduler=scheduler)
        Logger(f"Final model saved to {ckp}")
 
    if dist.is_initialized():
        dist.destroy_process_group()