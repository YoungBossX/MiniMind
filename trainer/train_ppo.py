"""
PPO (Proximal Policy Optimization) 是一种 Actor-Critic 强化学习算法.
核心思想: 用 "新策略/旧策略" 的概率比率来更新参数, 同时用 clip 限制更新幅度,
         防止策略变化太大导致训练崩溃.
 
在 LLM RLHF/RLAIF 场景下:
  - Actor  = LLM 本身, 负责生成回答
  - Critic = 一个 value network, 预测 "当前状态能拿到多少 reward"
  - Reward Model = 外部打分模型, 给生成的回答打分
  - Reference Model = 冻结的初始 LLM, 用来计算 KL 散度防止策略跑太远
 
训练流程 (每个 step):
  1. Actor 根据 prompt 生成回答 (rollout)
  2. Reward Model 给回答打分
  3. Critic 估计每个 token 位置的 value
  4. 用 GAE 算法计算每个 token 的 advantage (这个 action 比平均好多少)
  5. 用 PPO 的 clipped objective 更新 Actor 和 Critic (K 轮)
"""
import os
import sys
 
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
 
import argparse
import re
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.MiniMindModel import MiniMindConfig, MiniMindForCausalLM
from dataset.llm_dataset import RLAIFDataset
from trainer.trainer_utils import (
    Logger, is_main_process, lm_checkpoint,
    init_distributed_mode, setup_seed, SkipBatchSampler, init_model
)
 
warnings.filterwarnings('ignore')
 
# ============================================================================
#  Critic 模型: 继承 MiniMindForCausalLM
# ============================================================================
#
#  PPO 需要一个 Value Network (Critic) 来估计 "在当前 token 位置，未来还能拿到多少 reward". 这个估计值 V(s) 用于计算 advantage = "实际拿到的" - "预估能拿到的"。
#
#  实现方式: 复用模型的 Transformer backbone (共享相同的语言理解能力)，
#  只是把最后的 lm_head (输出词表概率) 换成 value_head (输出一个标量)。
#
#  为什么不单独训一个网络？
#  LLM 的 Transformer backbone 已经学会了语言理解, 复用它可以让 Critic 快速学会评估文本质量, 不需要从零学起。
class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        self.value_head = nn.Linear(params.hidden_size, 1)
    
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])
        # 通过 value_head 计算状态值
        values = self.value_head(hidden_states).squeeze(-1) # 形状为 (batch_size, seq_len)
        return values
    
# ============================================================================
#  GAE: Generalized Advantage Estimation (per-token)
# ============================================================================
#
#  为什么需要 GAE? 直接用 reward - V(s) 做 advantage 有什么问题?
#
#  简单做法:  A = R - V(last_token)
#    问题: 只看最后一个 token 的 value, 完全忽略了中间过程.
#          就像只看期末考试成绩评价学生, 不看平时表现.
#
#  GAE 做法: 逐 token 反向递推, 综合考虑每一步的 TD error
#    TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
#              = "实际拿到的 + 下一步预估的" - "当前预估的"
#              > 0 说明比预期好, < 0 说明比预期差
#
#    GAE:     A_t = δ_t + (γλ) * δ_{t+1} + (γλ)² * δ_{t+2} + ...
#              = δ_t + γλ * A_{t+1}  (递推形式)
#
#    λ 控制 bias-variance tradeoff:
#      λ=0: A_t = δ_t, 只看一步, 方差小但偏差大
#      λ=1: A_t = R - V(s_t), 看所有步, 偏差小但方差大
#      λ=0.95: 折中, 实践中最常用
def compute_gae(rewards_scalar, values_seq, resp_mask, gamma=1.0, lam=0.95):
    B, T = values_seq.shape 
    device = values_seq.device

    # 构造 per-token reward: 只在每个序列最后一个 response token 放 reward
    per_token_rewards = torch.zeros(B, T, device=device)
    # 找到每个序列最后一个 response token 的位置
    last_resp_indices = (resp_mask * torch.arange(T, device=device).unsqueeze(0)).argmax(dim=1)  # [B]
    per_token_rewards[torch.arange(B, device=device), last_resp_indices] = rewards_scalar

    # 反向递推计算 GAE
    advantages = torch.zeros(B, T, device=device)
    gae = torch.zeros(B, device=device)
 
    for t in reversed(range(T)):
        mask_t = resp_mask[:, t].float()  # [B]
        if t == T - 1:
            next_value = torch.zeros(B, device=device)
        else:
            next_value = values_seq[:, t + 1]
        # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        delta = per_token_rewards[:, t] + gamma * next_value - values_seq[:, t]
        # GAE: A_t = δ_t + γλ * A_{t+1}
        gae = delta + gamma * lam * gae
        # 只在 response token 上累积 advantage
        advantages[:, t] = gae * mask_t
 
    returns = advantages + values_seq  # TD(λ) return target
    return advantages, returns

# ============================================================================
#  Reward 计算
# ============================================================================
#  奖励信号由两部分组成:
#  1) 规则奖励 (仅 reasoning 模式): 检查 <think>/<answer> 格式是否正确
#  2) Reward Model 打分: 用 InternLM2-1.8B-Reward 给回答质量打分
def calculate_rewards(prompts, responses, reward_model, reward_tokenizer, args):
    """
    计算每个 response 的总奖励.
 
    Args:
        prompts:   list[str], 原始 prompt 文本
        responses: list[str], 生成的回答文本
        reward_model: 预训练的 reward model (frozen)
        reward_tokenizer: reward model 对应的 tokenizer
        args: 命令行参数
 
    Returns:
        rewards: [B] 每个 response 的总奖励分数
    """
    def reasoning_model_reward(rewards):
        """
        推理模型的格式奖励: 鼓励模型输出 <think>...</think><answer>...</answer> 格式.
 
        为什么需要这个?
        → 类似 DeepSeek-R1, 我们希望模型先思考再回答.
          如果模型输出了正确的格式, 给 0.5 分的额外奖励.
          即使格式不完全正确, 只要出现了部分标签, 也给部分奖励 (防止奖励过于稀疏).
        """
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"
        matches1 = [re.match(pattern, r, re.S) for r in responses]
        matches2 = [re.match(pattern2, r, re.S) for r in responses]
        format_rewards = [0.5 if (m1 or m2) else 0.0 for m1, m2 in zip(matches1, matches2)]
        rewards += torch.tensor(format_rewards, device=args.device)
 
        # 标记奖励: 每个标签出现恰好 1 次各给 0.25 分
        # 这是一个 "阶梯式" 奖励, 防止模型完全拿不到分导致学不动
        def mark_num(text):
            r = 0
            for tag in ["<think>", "</think>", "<answer>", "</answer>"]:
                if text.count(tag) == 1:
                    r += 0.25
            return r
 
        mark_rewards = [mark_num(r) for r in responses]
        rewards += torch.tensor(mark_rewards, device=args.device)
        return rewards
 
    rewards = torch.zeros(len(responses), device=args.device)
    
    # 1) 格式奖励 (仅 reasoning 模式)
    if args.reasoning == 1:
        rewards = reasoning_model_reward(rewards)
 
    # 2) Reward Model 打分
    with torch.no_grad():
        scores = []
        for prompt, response in zip(prompts, responses):
            # 从 prompt 中解析出 chat 格式的 messages
            # prompt 长这样: <|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]
            # 拼上 response, 送给 reward model 打分
            tmp_chat = messages + [{"role": "assistant", "content": response}]
            score = reward_model.get_score(reward_tokenizer, tmp_chat)
            # 裁剪到 [-3, 3], 防止极端值干扰训练
            scale = 3.0
            score = max(min(score, scale), -scale)
            # reasoning 模式: 额外给 <answer> 内容单独打分
            # 因为 <think> 部分可能很长但质量不高, 最终答案的质量更重要
            if args.reasoning == 1:
                answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                if answer_match:
                    answer_content = answer_match.group(1).strip()
                    tmp_chat2 = messages + [{"role": "assistant", "content": answer_content}]
                    answer_score = reward_model.get_score(reward_tokenizer, tmp_chat2)
                    answer_score = max(min(answer_score, scale), -scale)
                    score = score * 0.4 + answer_score * 0.6
 
            scores.append(score)
 
        rewards += torch.tensor(scores, device=args.device)
 
    return rewards

# ============================================================================
#  Rollout 数据收集 (生成 + 打分 + 计算 advantage)
# ============================================================================
#  这是 PPO 的 "数据收集阶段", 对应 RL 中的 "和环境交互".
#  在 LLM 场景下, "和环境交互" = "用当前 Actor 生成回答, 然后用 Reward Model 打分".
#
#  所有计算都在 @torch.no_grad() 下完成, 因为这里只是收集数据,
#  真正的梯度计算在后面的 ppo_update() 里.
@torch.no_grad()
def collect_rollout(prompts, actor_model, critic_model, old_actor_model, ref_model,
                    reward_model, reward_tokenizer, tokenizer, args, autocast_ctx):
    """
    收集一个 batch 的完整 rollout 数据.
 
    流程:
    1. Actor 生成回答
    2. Reward Model 打分
    3. Critic 估计 value
    4. Old Actor 和 Ref Model 计算 log-prob (用于后续计算 ratio 和 KL)
    5. GAE 计算 advantage
    6. Advantage 标准化
 
    Returns:
        dict: 包含后续 PPO 更新所需的所有张量
    """
    # ---- 1. 编码 prompt ----
    # padding_side="left" 是因为生成时需要从右边续写
    # 如果右边 pad 了, generate 会在 pad token 后面续写, 不对
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                    max_length=args.max_seq_len, padding_side="left").to(args.device)
    prompt_length = enc.input_ids.shape[1]
 
    # ---- 2. Actor 生成回答 ----
    # DDP 包装的模型需要 .module 才能调用 generate
    # gen_out 形状: [B, P+R], 其中 P=prompt长度, R=response长度 (可变)
    model_for_gen = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
    gen_out = model_for_gen.generate(
        input_ids=enc.input_ids, attention_mask=enc.attention_mask,
        max_new_tokens=args.max_gen_len, do_sample=True, temperature=0.8,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id
    )
 
    # ---- 3. 解码文本 & 计算 reward ----
    responses_text = [tokenizer.decode(gen_out[i, prompt_length:], skip_special_tokens=True) for i in range(len(prompts))]
    rewards = calculate_rewards(prompts, responses_text, reward_model, reward_tokenizer, args)  # [B]

    # ---- 4. 构造各种 mask ----
    # full_mask: 哪些位置不是 padding (用于 attention)
    full_mask = (gen_out != tokenizer.pad_token_id).long()  # [B, P+R]
    # labels: 用于计算 log-prob 的目标 token (右移一位)
    # 因为语言模型预测的是 "下一个 token", 所以 labels = input[1:]
    # 例如 input = [A, B, C, D], labels = [B, C, D]
    # 模型在位置 A 预测 B, 在位置 B 预测 C, ...
    seq_len = gen_out.size(1) - 1  # labels = gen_out[:, 1:], 所以 token 数量 - 1
    labels = gen_out[:, 1:].clone()  # [B, T] 其中 T = P+R-1
    
    # resp_mask: 标记哪些位置属于 response (排除 prompt 和 padding)
    # prompt_length-1 是因为 labels 比 input 少一个 token
    resp_mask = torch.arange(seq_len, device=gen_out.device).unsqueeze(0) >= (prompt_length - 1)
    pad_mask = ~labels.eq(tokenizer.pad_token_id)
    resp_mask = resp_mask & pad_mask  # [B, T]
 
    # ---- 5. 计算各模型的 per-token log-prob ----
    with autocast_ctx:
        # Critic: 估计每个 token 位置的 value
        values_seq = critic_model(input_ids=gen_out, attention_mask=full_mask)  # [B, P+R]
        # 取 shifted 的 values (对齐 labels)
        values_seq = values_seq[:, :-1]  # [B, T]
        # Old Actor: 冻结的上一版 actor, 用于计算 importance sampling ratio
        old_logits = old_actor_model(input_ids=gen_out, attention_mask=full_mask).logits[:, :-1]  # [B, T, V]
        # Reference Model: 冻结的初始模型, 用于计算 KL 散度
        ref_logits = ref_model(input_ids=gen_out, attention_mask=full_mask).logits[:, :-1]  # [B, T, V]
    
    # 计算 per-token log-probability
    # log_softmax 得到 log(概率分布), gather 取出 label 对应的那个 log-prob
    # 例如 logits=[0.1, 0.8, 0.1], label=1, 则 log_softmax→[-2.4, -0.4, -2.4], gather→-0.4
    old_logp = F.log_softmax(old_logits, dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, T]
    ref_logp = F.log_softmax(ref_logits, dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, T]
 
    # ---- 6. 用 GAE 计算 per-token advantage ----
    advantages, returns = compute_gae(
        rewards, values_seq.float(), resp_mask.float(),
        gamma=args.gamma, lam=args.gae_lambda
    )
 
    # ---- 7. Advantage 标准化 ----
    # 只在 response token 上做标准化, 使 advantage 均值≈0, 标准差≈1
    # 这样正 advantage (好的 action) 和负 advantage (差的 action) 大致对称,
    # 防止 policy gradient 被 advantage 的绝对大小主导
    adv_resp = advantages[resp_mask]
    if adv_resp.numel() > 1:
        advantages = (advantages - adv_resp.mean()) / (adv_resp.std() + 1e-8)
 
    return {
        'gen_out': gen_out,           # [B, P+R]
        'full_mask': full_mask,       # [B, P+R]
        'labels': labels,             # [B, T]
        'resp_mask': resp_mask,       # [B, T]
        'old_logp': old_logp,         # [B, T]
        'ref_logp': ref_logp,         # [B, T]
        'advantages': advantages,     # [B, T]
        'returns': returns,           # [B, T]  (GAE return = adv + V)
        'old_values': values_seq,     # [B, T]
        'rewards': rewards,           # [B]
        'responses_text': responses_text,
        'prompt_length': prompt_length,
    }

# ============================================================================
#  PPO Update: 对收集到的 rollout 数据做 K 轮更新
# ============================================================================
#  这是 PPO 算法的精髓. 对同一批 rollout 数据做 K 轮梯度更新.
#
#  为什么可以用同一批数据更新多次?
#  → 这就是 PPO 比普通 Policy Gradient 高效的地方.
#    通过 importance sampling (用 ratio = π_new/π_old 修正) 和
#    clipping (限制 ratio 不能偏离 1 太远), PPO 允许在同一批数据上
#    安全地做多次更新, 大幅提高数据利用率.
#
#  Loss 由四部分组成:
#  1. Policy Loss:  PPO clipped surrogate objective (更新 Actor)
#  2. Value Loss:   MSE(V_new, GAE_returns) with clipping (更新 Critic)
#  3. KL Penalty:   KL(π_current || π_reference), 防止偏离初始模型太远
#  4. Entropy Bonus: -H(π), 鼓励探索, 防止策略过早坍缩到确定性输出
def ppo_update(rollout, actor_model, critic_model, actor_optimizer, critic_optimizer,
               actor_scheduler, critic_scheduler, tokenizer, args, autocast_ctx, lm_config):
    """
    标准 PPO 的 K-epoch 更新.
    对同一批 rollout 数据做多轮梯度更新, 通过 clip 和 early stopping 保证稳定性.
    """
    gen_out = rollout['gen_out']
    full_mask = rollout['full_mask']
    labels = rollout['labels']
    resp_mask = rollout['resp_mask'].float()
    old_logp = rollout['old_logp']
    ref_logp = rollout['ref_logp']
    advantages = rollout['advantages']
    returns = rollout['returns']
    old_values = rollout['old_values']
 
    metrics = {}
 
    for ppo_ep in range(args.ppo_epochs):
        # ================================================================
        #  Actor Forward: 计算当前策略下每个 token 的 log-prob
        # ================================================================
        with autocast_ctx:
            res = actor_model(input_ids=gen_out, attention_mask=full_mask)
            logits = res.logits[:, :-1]  # [B, T, V]
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
 
        log_probs_all = F.log_softmax(logits, dim=-1)  # [B, T, V]
        actor_logp = log_probs_all.gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, T]
 
        # ================================================================
        #  Entropy Bonus: 防止策略过早收敛
        # ================================================================
        # H(π) = -Σ_a π(a) * log π(a)
        # 如果 entropy 很低, 说明模型对输出非常确定, 不再探索其他可能性
        # 加上 entropy bonus 鼓励模型保持一定的随机性
        probs_all = log_probs_all.exp()
        entropy_per_token = -(probs_all * log_probs_all).sum(dim=-1)  # [B, T]
        # 只在 response token 上计算平均 entropy
        entropy = (entropy_per_token * resp_mask).sum() / (resp_mask.sum() + 1e-8)
 
        # ================================================================
        #  Critic Forward: 计算当前 Critic 的 value 预测
        # ================================================================
        with autocast_ctx:
            values_new = critic_model(input_ids=gen_out, attention_mask=full_mask)[:, :-1]  # [B, T]
 
        # ================================================================
        #  Policy Loss: PPO Clipped Surrogate Objective
        # ================================================================
        #
        # 核心公式:
        #   ratio = π_new(a|s) / π_old(a|s) = exp(logp_new - logp_old)
        #   L_clip = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
        #
        # 直觉:
        #   - ratio = 1: 新旧策略一样, 普通 policy gradient
        #   - ratio > 1: 新策略更倾向这个 action
        #   - ratio < 1: 新策略不太倾向这个 action
        #   - clip 限制 ratio 在 [1-ε, 1+ε] 范围内, 防止更新太大
        #
        # 如果 advantage > 0 (好 action):
        #   min(ratio*A, clip(ratio)*A) 会在 ratio > 1+ε 时停止增长
        #   防止对好 action 过度增加概率
        #
        # 如果 advantage < 0 (坏 action):
        #   min(ratio*A, clip(ratio)*A) 会在 ratio < 1-ε 时停止下降
        #   防止对坏 action 过度减少概率
        ratio = torch.exp(actor_logp - old_logp)  # [B, T]
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages
        policy_loss_per_token = -torch.min(surr1, surr2)  # [B, T]
        policy_loss = (policy_loss_per_token * resp_mask).sum() / (resp_mask.sum() + 1e-8)
 
        # Clip fraction (诊断指标)
        clip_frac = ((ratio - 1.0).abs() > args.clip_epsilon).float()
        clip_frac = (clip_frac * resp_mask).sum() / (resp_mask.sum() + 1e-8)
 
        # ================================================================
        #  Value Loss: 训练 Critic 更准确地预测 return
        # ================================================================
        # 基本: MSE(V_new, returns)
        # 改进: Value Function Clipping (PPO 论文建议)
        #   V_clipped = V_old + clip(V_new - V_old, -ε, +ε)
        #   L_vf = max(MSE(V_new, returns), MSE(V_clipped, returns))
        #
        # 这防止 Critic 在一次更新中变化太大 (类似 policy clipping 的思想)
        values_clipped = old_values + torch.clamp(
            values_new - old_values, -args.clip_epsilon, args.clip_epsilon
        )
        vf_loss1 = (values_new - returns) ** 2
        vf_loss2 = (values_clipped - returns) ** 2
        value_loss_per_token = torch.max(vf_loss1, vf_loss2)
        value_loss = 0.5 * (value_loss_per_token * resp_mask).sum() / (resp_mask.sum() + 1e-8)
 
        # ================================================================
        #  KL Penalty: 防止策略偏离 Reference Model 太远
        # ================================================================
        # 为什么需要 KL(π || π_ref)?
        #   如果只优化 reward, 模型可能学会 "hack" reward model,
        #   输出 reward model 喜欢但实际质量很差的内容 (reward hacking).
        #   加上 KL 惩罚, 让模型不能偏离初始策略太远, 保持一定的 "正常".
        #
        # 公式: KL ≈ r - 1 - log(r), 其中 r = exp(logp_new - logp_ref)
        # 这是 KL 散度的二阶近似, 非负, 在 r=1 时为 0
        log_ratio_ref = actor_logp - ref_logp  # [B, T]
        ratio_ref = torch.exp(log_ratio_ref)
        kl_ref_per_token = ratio_ref - 1 - log_ratio_ref  # [B, T], 非负
        kl_ref = (kl_ref_per_token * resp_mask).sum() / (resp_mask.sum() + 1e-8)
 
        # ================================================================
        #  Early Stopping: 当新旧策略差异过大时停止当前 epoch
        # ================================================================
        # approx_kl 衡量当前 actor 和 old_actor 之间的 KL
        # 如果太大, 说明这批数据已经被 "用完了", 继续更新反而有害
        log_ratio_old = actor_logp - old_logp  # [B, T]
        approx_kl = (log_ratio_old.exp() - 1 - log_ratio_old)  # [B, T]
        approx_kl = (approx_kl * resp_mask).sum() / (resp_mask.sum() + 1e-8)
        if ppo_ep > 0 and approx_kl.item() > args.target_kl:
            Logger(f"  PPO epoch {ppo_ep}: early stop, approx_kl={approx_kl.item():.4f} > target_kl={args.target_kl}")
            break
 
        # ================================================================
        #  Total Loss: 加权求和
        # ================================================================
        #
        #  L = L_policy                       ← 核心: 让 actor 输出更好的 response
        #    + vf_coef * L_value              ← 让 critic 预测更准
        #    + kl_coef * KL(π, π_ref)         ← 别跑太远
        #    - entropy_coef * H(π)            ← 负号! 最大化 entropy = 最小化 -entropy
        #    + aux_loss                       ← MoE load balancing (如果用 MoE)
        loss = (
            policy_loss
            + args.vf_coef * value_loss
            + args.kl_coef * kl_ref
            - args.entropy_coef * entropy  # entropy bonus (负号=最大化 entropy)
            + aux_loss
        ) / args.accumulation_steps
 
        loss.backward()
 
    # 返回最后一轮的 metrics (用于日志)
    metrics = {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': entropy.item(),
        'kl_ref': kl_ref.item(),
        'approx_kl': approx_kl.item(),
        'clip_frac': clip_frac.item(),
        'aux_loss': aux_loss.item(),
        'reward': rollout['rewards'].mean().item(),
        'ppo_epochs_actual': ppo_ep + 1,
    }
 
    return metrics

# ============================================================================
#  训练一个 Epoch
# ============================================================================
def ppo_train_epoch(epoch, loader, iters, old_actor_model, ref_model,
                    actor_scheduler, critic_scheduler,
                    reward_model, reward_tokenizer, start_step=0, wandb=None):
    """
    训练一个完整的 epoch.
 
    每个 step 的流程:
    1. collect_rollout: Actor 生成回答, 收集所有数据
    2. ppo_update:      用 PPO 算法更新 Actor 和 Critic (K 轮)
    3. 梯度步:          应用梯度 (可能有梯度累积)
    4. 日志:            记录训练指标
    5. 同步 Old Actor:  定期把当前 Actor 的参数复制给 Old Actor
    6. 保存 checkpoint: 定期保存模型
    """
    actor_model.train()
    critic_model.train()
 
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]
 
        # ---- Step 1: 收集 rollout ----
        # 这一步: 生成回答 → 打分 → 计算 value → 计算 GAE advantage
        # 全程 no_grad, 不消耗计算图内存
        rollout = collect_rollout(
            prompts, actor_model, critic_model, old_actor_model, ref_model,
            reward_model, reward_tokenizer, tokenizer, args, autocast_ctx
        )
 
        # ---- Step 2: PPO 更新 ----
        # 用收集到的数据做 K 轮梯度更新
        # 这一步会调用 loss.backward(), 在 actor 和 critic 上累积梯度
        metrics = ppo_update(
            rollout, actor_model, critic_model,
            actor_optimizer, critic_optimizer,
            actor_scheduler, critic_scheduler,
            tokenizer, args, autocast_ctx, lm_config
        )
 
        # ---- Step 3: 应用梯度 ----
        # 如果使用梯度累积 (accumulation_steps > 1), 等累积够了再 step
        if (step + 1) % args.accumulation_steps == 0:
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            actor_optimizer.step()
            critic_optimizer.step()
            actor_scheduler.step()
            critic_scheduler.step()
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
 
        # ---- Step 4: 日志 ----
        if is_main_process():
            # 统计生成 response 的平均长度 (监控 reward hacking: 长度暴涨可能有问题)
            gen_out = rollout['gen_out']
            prompt_length = rollout['prompt_length']
            response_ids = gen_out[:, prompt_length:]
            is_eos = (response_ids == tokenizer.eos_token_id)
            eos_indices = torch.argmax(is_eos.int(), dim=1)
            has_eos = is_eos.any(dim=1)
            lengths = torch.where(has_eos, eos_indices + 1,
                                  torch.tensor(response_ids.shape[1], device=is_eos.device))
            avg_len = lengths.float().mean().item()
 
            actor_lr = actor_optimizer.param_groups[0]['lr']
            critic_lr = critic_optimizer.param_groups[0]['lr']

            # 记录到 wandb/swanlab
            if wandb is not None:
                wandb.log({
                    "actor_loss": metrics['policy_loss'],
                    "critic_loss": metrics['value_loss'],
                    "entropy": metrics['entropy'],
                    "aux_loss": metrics['aux_loss'],
                    "reward": metrics['reward'],
                    "kl_ref": metrics['kl_ref'],
                    "approx_kl": metrics['approx_kl'],
                    "clip_fraction": metrics['clip_frac'],
                    "avg_response_len": avg_len,
                    "actor_lr": actor_lr,
                    "ppo_epochs_actual": metrics['ppo_epochs_actual'],
                })
            # 终端输出
            if step % args.log_interval == 0:
                Logger(
                    f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                    f"πLoss:{metrics['policy_loss']:.4f}, VLoss:{metrics['value_loss']:.4f}, "
                    f"Entropy:{metrics['entropy']:.4f}, Reward:{metrics['reward']:.4f}, "
                    f"KL_ref:{metrics['kl_ref']:.4f}, ApproxKL:{metrics['approx_kl']:.4f}, "
                    f"ClipFrac:{metrics['clip_frac']:.3f}, PPO_ep:{metrics['ppo_epochs_actual']}, "
                    f"AvgLen:{avg_len:.1f}, aLR:{actor_lr:.2e}, cLR:{critic_lr:.2e}"
                )
 
        # ---- Step 5: 同步 Old Actor ----
        # 每隔几步把当前 Actor 的参数复制到 Old Actor
        # Old Actor 用于计算 importance sampling ratio
        # 如果更新太频繁, ratio 总是 ≈1, PPO 退化为普通 PG
        # 如果更新太不频繁, ratio 偏离 1 太远, 训练不稳定
        if (step + 1) % args.update_old_actor_freq == 0:
            # 在 device 上直接 clone, 不走 CPU
            raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)
            for p_old, p_new in zip(old_actor_model.parameters(), raw_actor.parameters()):
                p_old.data.copy_(p_new.data)
 
        # ---- Step 6: 保存 checkpoint ----
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            actor_model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)
            actor_state = raw_actor.state_dict()
            torch.save({k: v.half().cpu() for k, v in actor_state.items()}, ckp)
 
            lm_checkpoint(lm_config, weight=args.save_weight, model=actor_model, optimizer=actor_optimizer,
                          epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints',
                          scheduler=actor_scheduler, critic_model=critic_model,
                          critic_optimizer=critic_optimizer, critic_scheduler=critic_scheduler)
            actor_model.train()
            del actor_state
 
        # 清理显存
        del rollout

# ============================================================================
#  Main
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind PPO (Improved)")
    # --- 基础参数 ---
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument('--save_weight', default='ppo_actor', type=str)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="Actor 学习率")
    parser.add_argument("--critic_learning_rate", type=float, default=8e-8, help="Critic 学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=10)
 
    # --- 模型参数 ---
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument('--max_seq_len', default=66, type=int, help="Prompt 最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1536, help="生成最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif-mini.jsonl")
 
    # --- PPO 核心超参数 ---
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO clip 参数 (原版 0.1, 建议 0.2)")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function loss 系数")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL(ref) 惩罚系数")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy bonus 系数")
    parser.add_argument("--gamma", type=float, default=1.0, help="GAE discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--ppo_epochs", type=int, default=4, help="每批数据的 PPO 更新轮数")
    parser.add_argument("--target_kl", type=float, default=0.02, help="KL early stopping 阈值")
 
    # --- 其他 ---
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1])
    parser.add_argument("--update_old_actor_freq", type=int, default=4)
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-PPO-Improved")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1])
    args = parser.parse_args()
 
    # ========== 1. 初始化环境 ==========
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
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume == 1 else None
 
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
        wandb_run_name = f"MiniMind-PPO-Imp-E{args.epochs}-BS{args.batch_size}-LR{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
 
    # ========== 5. 初始化模型 ==========
    # PPO 训练中同时需要维护以下模型:
    #
    #   ┌─────────────────┬──────────────┬─────────────────────────────────────┐
    #   │ 模型            │ 是否更新     │ 作用                                │
    #   ├─────────────────┼──────────────┼─────────────────────────────────────┤
    #   │ Actor           │ ✅ 训练      │ 生成回答, 核心要优化的模型           │
    #   │ Old Actor       │ ❌ 定期同步  │ 计算 ratio=π_new/π_old 的分母       │
    #   │ Ref Model       │ ❌ 永远冻结  │ 计算 KL 散度, 防止 reward hacking   │
    #   │ Critic          │ ✅ 训练      │ 估计 value, 用于 GAE 计算 advantage │
    #   │ (Reward Model)  │ ❌ 永远冻结  │ 给 response 打分                    │
    #   └─────────────────┴──────────────┴─────────────────────────────────────┘
    #
    # 这就是为什么 PPO 比 DPO/GRPO 更吃显存 (约 1.5-2 倍)
    base_weight = "reason" if args.reasoning == 1 else "full_sft"
 
    # Actor
    actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    if args.use_compile == 1:
        actor_model = torch.compile(actor_model)
        Logger('torch.compile enabled')
 
    # Old Actor (frozen)
    old_actor_model, _ = init_model(lm_config, base_weight, device=args.device)
    old_actor_model = old_actor_model.eval().requires_grad_(False)
 
    # Reference (frozen)
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
 
    # Critic (从同样的 base weight 初始化, value_head 随机初始化)
    moe_suffix = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/{base_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    critic_model = CriticModel(lm_config)
    critic_model.load_state_dict(state_dict, strict=False)
    critic_model = critic_model.to(args.device)
 
    # Reward Model (frozen)
    reward_model = AutoModel.from_pretrained(
        args.reward_model_path, device_map="cuda", torch_dtype=torch.float16, trust_remote_code=True
    )
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)
 
    # ========== 6. 数据和优化器 ==========
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len))
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
 
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)
 
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_optimizer_steps, eta_min=args.critic_learning_rate / 10)
 
    # ========== 7. 恢复训练状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        actor_model.load_state_dict(ckp_data['model'])
        critic_model.load_state_dict(ckp_data['critic_model'])
        actor_optimizer.load_state_dict(ckp_data['optimizer'])
        critic_optimizer.load_state_dict(ckp_data['critic_optimizer'])
        actor_scheduler.load_state_dict(ckp_data['scheduler'])
        critic_scheduler.load_state_dict(ckp_data['critic_scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
 
    # ========== 8. DDP ==========
    if dist.is_initialized():
        actor_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        critic_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])
        old_actor_model.to(args.device)
 
    # ========== 9. 训练 ==========
    Logger("=" * 70)
    Logger(f"PPO Improved Training | Epochs: {args.epochs} | Batch: {args.batch_size}")
    Logger(f"PPO epochs/batch: {args.ppo_epochs} | GAE(γ={args.gamma}, λ={args.gae_lambda})")
    Logger(f"Clip: {args.clip_epsilon} | Entropy coef: {args.entropy_coef} | Target KL: {args.target_kl}")
    Logger("=" * 70)
 
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
 
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}步, 从 step {start_step + 1} 开始')
            ppo_train_epoch(epoch, loader, len(loader) + skip, old_actor_model, ref_model,
                            actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, start_step, wandb)
        else:
            ppo_train_epoch(epoch, loader, len(loader), old_actor_model, ref_model,
                            actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, 0, wandb)
 
    # 训练结束后总是保存一次
    if is_main_process():
        Logger("Training finished. Saving final checkpoint...")
        actor_model.eval()
        moe_suffix = '_moe' if lm_config.use_moe else ''
        ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
        raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)
        actor_state = raw_actor.state_dict()
        torch.save({k: v.half().cpu() for k, v in actor_state.items()}, ckp)
        lm_checkpoint(lm_config, weight=args.save_weight, model=actor_model, optimizer=actor_optimizer,
                      epoch=args.epochs, step=0, wandb=wandb, save_dir='../checkpoints',
                      scheduler=actor_scheduler, critic_model=critic_model,
                      critic_optimizer=critic_optimizer, critic_scheduler=critic_scheduler)
        Logger(f"Final model saved to {ckp}")
 
    # ========== 10. 清理 ==========
    if dist.is_initialized():
        dist.destroy_process_group()