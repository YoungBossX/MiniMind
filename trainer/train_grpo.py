import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import re
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.MiniMindModel import MiniMindConfig, MiniMindForCausalLM
from dataset.llm_dataset import RLAIFDataset
from trainer.trainer_utils import (
    Logger, is_main_process, lm_checkpoint,
    init_distributed_mode, setup_seed, SkipBatchSampler, init_model, checkpoint_due,
    build_checkpoint_metadata, load_model_state, save_inference_weights,
    clamp_log_ratio, gather_rng_states, restore_rng_state_for_rank,
    clip_gradients, init_reference_model, synchronize_model_state,
    load_reward_components, resolve_checkpoint_dir,
    coordinated_checkpoint_save, tokenize_rl_prompts, build_rollout_masks,
    chatml_prompt_messages
)
from trainer.path_utils import resolve_project_paths
 
warnings.filterwarnings('ignore')


def validate_args(args):
    if not args.temperature > 0:
        raise ValueError("GRPO requires --temperature to be greater than 0.")
    if args.dtype != "bfloat16":
        raise ValueError("GRPO only supports --dtype=bfloat16; float16 is not numerically safe.")
    if args.num_generations < 2:
        raise ValueError("GRPO requires --num_generations to be at least 2.")
    if args.grpo_epochs < 1:
        raise ValueError("GRPO requires --grpo_epochs to be at least 1.")
    if args.accumulation_steps != 1:
        raise ValueError(
            "GRPO requires --accumulation_steps=1 because each GRPO epoch performs an optimizer step."
        )


def sampled_k3_kl(policy_logp, ref_logp):
    """Sampled k3 KL estimator using d = log(pi_ref) - log(pi_policy)."""
    d = clamp_log_ratio(ref_logp - policy_logp)
    return torch.exp(d) - d - 1.0


def clipped_surrogate_loss(actor_logps, old_logps, advantages, clip_epsilon):
    ratio = torch.exp(clamp_log_ratio(actor_logps - old_logps))
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    return -torch.min(unclipped, clipped), ratio

# ==========================================================================
#  Per-token log-prob 计算 
# ==========================================================================
def get_per_token_logps(
    mdl, input_ids, num_completion_tokens, attention_mask, temperature=1.0,
    return_outputs=False,
):
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
    if not temperature > 0:
        raise ValueError("temperature must be greater than 0")
    outputs = mdl(
        ids,
        attention_mask=attention_mask,
        logits_to_keep=num_completion_tokens + 1,
    )
    logits = outputs.logits[:, :-1, :] / temperature
    completion_ids = ids[:, -num_completion_tokens:]
    log_probs = F.log_softmax(logits, dim=-1)
    per_token_logps = log_probs.gather(2, completion_ids.unsqueeze(-1)).squeeze(-1)
    entropy_per_token = -(log_probs.exp() * log_probs).sum(dim=-1)
    if return_outputs:
        return per_token_logps, entropy_per_token, outputs
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
                
                messages = chatml_prompt_messages(prompt)
                
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
        prompt_inputs, actor_prompts = tokenize_rl_prompts(
            tokenizer, prompts, max_length=args.max_seq_len, device=args.device
        )
 
        # ---- 2. 生成 N 个回答 ----
        model_for_gen = model.module if isinstance(model, DistributedDataParallel) else model
        was_training = model.training
        model.eval()
        with torch.no_grad():
            # num_return_sequences=G: 每个 prompt 生成 G 个不同的回答
            # 输出 shape: [B*G, P+R]
            outputs = model_for_gen.generate(
                **prompt_inputs,
                max_new_tokens=args.max_gen_len,
                do_sample=True,
                temperature=args.temperature,
                top_k=0,
                num_return_sequences=args.num_generations,
                pad_token_id=tokenizer.pad_token_id
            )
 
        prompt_len = prompt_inputs["input_ids"].size(1)
        completion_ids = outputs[:, prompt_len:]  # [B*G, R]
        R = completion_ids.size(1)
        
        if R == 0:
            del prompt_inputs, outputs, completion_ids
            continue
 
        # ---- 3. 冻结 rollout actor 和 ref 的 per-token log-prob ----
        full_attention_mask, completion_mask = build_rollout_masks(
            prompt_inputs["attention_mask"], outputs, tokenizer.eos_token_id
        )
        completion_mask = completion_mask.float()
        with torch.no_grad():
            with autocast_ctx:
                old_logps, _ = get_per_token_logps(
                    model, outputs, R, full_attention_mask, args.temperature
                )
                ref_logps, _ = get_per_token_logps(
                    ref_model, outputs, R, full_attention_mask, args.temperature
                )  # [B*G, R]
        old_logps = old_logps.detach()
        ref_logps = ref_logps.detach()
        if was_training:
            model.train()
 
        # ---- 4. 计算 reward 和 advantage ----
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        rewards = calculate_rewards(actor_prompts, completions, reward_model, reward_tokenizer, args)  # [B*G]
 
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
 
        # ---- 6. 对冻结 rollout 做多轮 clipped policy update ----
        seq_lengths = completion_mask.sum(dim=1).clamp(min=1)
        adv = advantages.unsqueeze(1)  # [B*G, 1], broadcast 到 completion token
        did_optimizer_step = False

        for grpo_ep in range(args.grpo_epochs):
            optimizer.zero_grad()
            with autocast_ctx:
                actor_logps, entropy_per_token, actor_outputs = get_per_token_logps(
                    model, outputs, R, full_attention_mask, args.temperature,
                    return_outputs=True,
                )
                aux_loss = (
                    actor_outputs.aux_loss
                    if lm_config.use_moe
                    else torch.tensor(0.0, device=args.device)
                )

            per_token_kl = sampled_k3_kl(actor_logps, ref_logps)
            surrogate_loss, ratio = clipped_surrogate_loss(
                actor_logps, old_logps, adv, args.clip_epsilon
            )
            per_token_loss = surrogate_loss + args.beta * per_token_kl
            entropy = (entropy_per_token * completion_mask).sum() / (
                completion_mask.sum() + 1e-8
            )

            # 每条序列先按有效 token 归一化，再对 batch 求均值。
            policy_loss = (per_token_loss * completion_mask).sum(dim=1) / seq_lengths
            policy_loss = policy_loss.mean()
            loss = policy_loss - args.entropy_coef * entropy + aux_loss
            loss.backward()

            clip_gradients(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            did_optimizer_step = True
 
        # ---- 7. 日志 ----
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
        if did_optimizer_step and checkpoint_due(
            step, iters, args.accumulation_steps, args.save_interval
        ):
            rng_state_by_rank = gather_rng_states(args.device)
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            coordinated_checkpoint_save(
                primary_save=lambda: lm_checkpoint(
                    lm_config, weight=args.save_weight, model=model,
                    optimizer=optimizer, epoch=epoch, step=step, wandb=wandb,
                    save_dir=args.checkpoint_dir, scheduler=scheduler,
                    metadata=checkpoint_metadata,
                    rng_state_by_rank=rng_state_by_rank, ref_model=ref_model,
                    save_inference=False,
                ),
                derived_save=lambda: save_inference_weights(model, ckp),
            )
            model.train()
 
        del prompt_inputs, outputs, completion_ids, actor_logps, old_logps, ref_logps
        del completions, rewards, advantages, completion_mask
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind GRPO")
    parser.add_argument("--save_dir", type=str, default="out")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="断点续训目录（默认 <save_dir>/checkpoints）")
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
    parser.add_argument("--temperature", type=float, default=0.8, help="采样及策略统计温度")
    parser.add_argument("--data_path", type=str, default="dataset/rlaif-mini.jsonl")
    parser.add_argument("--num_generations", type=int, default=4,  help="每个 prompt 生成几个回答 (G). 越大 advantage 越稳定, 但越慢")
    parser.add_argument("--grpo_epochs", type=int, default=2, help="每批 rollout 的更新轮数")
    parser.add_argument("--beta", type=float, default=0.02, help="KL 惩罚系数")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO 风格 clip 参数")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy bonus 系数")
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1])
    parser.add_argument("--reward_model_path", type=str, default="internlm2-1_8b-reward")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否显式从检查点续训")
    parser.add_argument('--allow_legacy_resume', default=0, type=int, choices=[0, 1], help="允许恢复缺少安全元数据的旧检查点")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-GRPO")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1])
    args = parser.parse_args()
    args = resolve_project_paths(args, "save_dir", "data_path", "reward_model_path", "checkpoint_dir")
    args.checkpoint_dir = resolve_checkpoint_dir(args.save_dir, args.checkpoint_dir)
    validate_args(args)
 
    # ========== 1. 初始化 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    setup_seed(42 + rank)
 
    # ========== 2. 配置 ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe)
    )
    checkpoint_metadata = build_checkpoint_metadata(args, lm_config, "grpo")
    ckp_data = (lm_checkpoint(
                    lm_config,
                    weight=args.save_weight,
                    save_dir=args.checkpoint_dir,
                    expected_metadata=checkpoint_metadata,
                    allow_legacy_resume=bool(args.allow_legacy_resume),
                )
                if args.from_resume == 1 else None)
 
    # ========== 3. 混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16
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
    model, tokenizer = init_model(
        lm_config, base_weight, save_dir=args.save_dir,
        resume_dir=args.checkpoint_dir, device=args.device,
        allow_legacy_resume=bool(args.allow_legacy_resume),
    )
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    
    ref_model, restored_ref = init_reference_model(
        lm_config,
        base_weight,
        checkpoint=ckp_data,
        save_dir=args.save_dir,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        allow_legacy_resume=bool(args.allow_legacy_resume),
    )
    if ckp_data and not restored_ref:
        Logger("Legacy checkpoint has no frozen reference snapshot.")
    synchronize_model_state(ref_model)
    ref_model = ref_model.eval().requires_grad_(False)
    
    reward_model, reward_tokenizer = load_reward_components(
        args.reward_model_path, args.device, dtype=torch.bfloat16
    )
 
    # ========== 6. 数据和优化器 ==========
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len))
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_steps = iters * args.grpo_epochs * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.learning_rate / 10)
 
    # ========== 7. 恢复 ==========
    start_epoch, start_step = 0, 0
    resume_rng_state_by_rank = None
    if ckp_data:
        load_model_state(model, ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
        resume_rng_state_by_rank = ckp_data.get('rng_state_by_rank')
 
    # ========== 8. DDP ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
 
    # ========== 9. 训练 ==========
    Logger("=" * 70)
    Logger(
        f"GRPO Training | Epochs:{args.epochs} | Batch:{args.batch_size} | "
        f"G:{args.num_generations} | Updates/Rollout:{args.grpo_epochs}"
    )
    Logger(f"Clip:{args.clip_epsilon} | Beta:{args.beta} | Entropy:{args.entropy_coef}")
    Logger("=" * 70)
 
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        train_ds.set_epoch(epoch)
        setup_seed(42 + epoch * world_size + rank)
        index_generator = torch.Generator()
        index_generator.manual_seed(42 + epoch)
        indices = torch.randperm(
            len(train_ds), generator=index_generator
        ).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader_generator = torch.Generator()
        loader_generator.manual_seed(10_000 + epoch + rank)
        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            generator=loader_generator,
        )
        if ckp_data and epoch == start_epoch:
            restored = restore_rng_state_for_rank(
                resume_rng_state_by_rank,
                device=args.device,
                allow_missing=bool(args.allow_legacy_resume),
            )
            if not restored:
                Logger("Legacy checkpoint has no RNG state; resume is not exact.")
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}步')
            grpo_train_epoch(epoch, loader, len(loader) + skip, ref_model,
                             reward_model, reward_tokenizer, start_step, wandb)
        else:
            grpo_train_epoch(epoch, loader, len(loader), ref_model,
                             reward_model, reward_tokenizer, 0, wandb)
            
    rng_state_by_rank = gather_rng_states(args.device)
    Logger("Training finished. Saving final checkpoint...")
    model.eval()
    moe_suffix = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    coordinated_checkpoint_save(
        primary_save=lambda: lm_checkpoint(
            lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
            epoch=args.epochs, step=0, wandb=wandb,
            save_dir=args.checkpoint_dir, scheduler=scheduler,
            metadata=checkpoint_metadata, rng_state_by_rank=rng_state_by_rank,
            ref_model=ref_model, save_inference=False,
        ),
        derived_save=lambda: save_inference_weights(model, ckp),
    )
    Logger(f"Final model saved to {ckp}")
 
    if dist.is_initialized():
        dist.destroy_process_group()
