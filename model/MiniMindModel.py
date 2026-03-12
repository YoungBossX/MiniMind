import torch
import math
import torch.nn as nn
from torch.nn import init
from typing import Optional, Tuple, List, Union
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        #------------ MoE ------------
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling else None
        )

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x) * self.weight


def precompute_freqs(
    dim: int,
    end: int,
    rope_base: float,
    rope_scaling: Optional[dict],
):
    # 1. 初始化标准 RoPE 频率。
    # torch.arange(0, dim, 2) 生成 [0, 2, 4, ... dim-2]
    # 计算出的 freqs 就是标准的 1 / (base ** (2i / d))
    freqs, attn_factor = (
        1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)),
        1.0,
    )

    if rope_scaling is not None:
        # 2. 从配置字典中提取 YaRN 的超参数
        # orig_max: 模型预训练时的原始最大长度（例如 Llama-2 是 2048 或 4096）
        # factor: 要扩展的倍数 s (比如从 2k 扩展到 32k，factor 就是 16)
        # beta_fast (对应论文中的 α): 高频边界，波长比例大于此值的维度不缩放
        # beta_slow (对应论文中的 β): 低频边界，波长比例小于此值的维度全量缩放
        # attn_factor: 注意力温度补偿，由于距离拉长导致注意力分布发散（变平缓），需要乘上一个系数让注意力重新“聚焦”
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0),
        )

        # 只有当要推断的长度大于原始训练长度时，才应用缩放
        if end / orig_max > 1.0:
            # 3. 使用前文推导的公式，定义波长比例 b 到维度索引 i 的映射函数
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (
                2 * math.log(rope_base)
            )

            # 4. 计算高频区和低频区的维度切分点
            # low: 不需要缩放的高频部分的最高索引
            # high: 需要完全缩放的低频部分的最低索引
            low, high = (
                max(math.floor(inv_dim(beta_fast)), 0),
                min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1),
            )

            # 5. 计算混合因子 γ (Ramp)
            # 在 low 之前，ramp 为 0；在 high 之后，ramp 为 1；在 low 和 high 之间，线性过渡。
            # clamp 函数限制了数值只能在 [0, 1] 之间。
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )

            # 6. 频率融合公式：f'(i) = f(i) * ((1-γ) + γ/s)
            # 当 ramp=0 时（高频）：系数为 1，保持原频率不变。
            # 当 ramp=1 时（低频）：系数为 1/factor，即对频率进行线性插值缩放。
            # ramp在0-1之间时：平滑过渡。
            freqs = freqs * (1 - ramp + ramp / factor)

    # 7. 根据目标长度 end，生成位置索引向量 t
    t = torch.arange(end, device=freqs.device)

    # 8. 计算外积：将位置 t 与处理好的频率 freqs 相乘，得到每个位置的旋转角度 θ
    freqs = torch.outer(t, freqs).float()

    # 9. 计算 Cos 和 Sin，并应用注意力补偿系数 (attn_factor)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # 1. 如果提供了 position_ids，则根据 position_ids 从预计算的 cos 和 sin 中选择对应的位置编码
    def rotate_half(x):
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )

    # 2. 对 q 和 k 应用 RoPE 旋转公式
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # x 的形状是 (batch_size, seq_len, num_key_value_heads, head_dim)
    batch_size, seq_len, num_key_value_heads, head_dim = x.shape
    # 如果 n_rep=1，直接返回原始张量，不进行任何复制。
    if n_rep == 1:
        return x

    return (
        x[:, :, :, None, :]
        .expand(batch_size, seq_len, num_key_value_heads, n_rep, head_dim)
        .reshape(batch_size, seq_len, num_key_value_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(
        self,
        args: MiniMindConfig,
    ):
        super().__init__()
        self.config = args

        self.num_key_value_heads = (
            args.num_attention_heads
            if args.num_key_value_heads is None
            else args.num_key_value_heads
        )
        
        assert args.num_attention_heads % self.num_key_value_heads == 0

        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = args.num_attention_heads // self.num_key_value_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        self.q_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)

        self.dropout = args.dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and args.flash_attention
        )

    def forward(
            self, 
            x: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache=False,
            attention_mask: Optional[torch.Tensor] = None,
        ):
        # 1. 获取输入的 batch_size 和 seq_len
        batch_size, seq_len, _ = x.size()
        # 2. 线性变换得到 q, k, v，q_proj 输出形状 (batch_size, seq_len, hidden_size)，k_proj 和 v_proj 输出形状 (batch_size, seq_len, num_key_value_heads * head_dim)
        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)
        # 3. 重复 k 和 v 以匹配总的注意力头数
        xq = xq.view(batch_size, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        # 4. 应用 RoPE 位置编码
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        # 5. kv_cache 处理：如果提供了 past_key_value，则将新的 k 和 v 与缓存中的 k 和 v 进行拼接
        # xk 和 xv 的形状在拼接前是 (batch_size, seq_len, num_key_value_heads, head_dim)，拼接后是 (batch_size, seq_len_kv, num_key_value_heads, head_dim)，其中 seq_len_kv = seq_len + past_seq_len
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
        # 6. 计算注意力输出
        # 如果满足使用 Flash Attention 的条件（支持且训练模式下），则直接调用 PyTorch 内置的 scaled_dot_product_attention 函数
        if (
            self.flash
            and (seq_len > 1)
            and (past_key_value is None)
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        # 否则，手动计算注意力分数，并应用掩码和 softmax
        else:
            # 计算注意力分数：xq 的形状是 (batch_size, n_local_heads, seq_len, head_dim)，xk 的形状是 (batch_size, n_local_heads, seq_len_kv, head_dim)，因此 scores 的形状是 (batch_size, n_local_heads, seq_len, seq_len_kv)
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores[:, :, :, -seq_len:] += torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1,
            )
            # 如果提供了 attention_mask，则将其扩展到与 scores 形状匹配，并将 mask 应用到 scores 上。通常，attention_mask 的形状是 (batch_size, seq_len)，其中 1 表示有效位置，0 表示需要掩盖的位置。通过 unsqueeze 和广播机制，将其转换为 (batch_size, 1, 1, seq_len)，然后乘以 -1e9（一个很大的负数）来掩盖无效位置，使得 softmax 后的概率接近于零
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv
    
class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        # 1. 根据配置计算 intermediate_size，如果配置中没有指定，则使用 hidden_size 的 8/3 倍，并向上调整到最接近的 64 的倍数
        # * 8/3 是为了抵消 SwiGLU 双路带来的额外参数，对齐到64倍数 是为了让 GPU 矩阵运算更高效
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        # 2. 定义前馈网络的升维线性层分支
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        # 2. 定义前馈网络有SiLU激活函数分支的线性层
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        # 4. 定义前馈网络的降维线性层分支
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        # 5. 定义前馈网络的 dropout 层
        self.dropout = nn.Dropout(config.dropout)
        # 6. 获取激活函数，ACT2FN 是一个字典，，让激活函数可以通过配置文件动态指定，而不是硬编码在代码里
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 计算前馈网络的升维输出和门控分支输出
        gated = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        # 2. 计算前馈网络的降维输出
        output = self.down_proj(gated)
        # 3. 应用 dropout
        output = self.dropout(output)
        return output
    
class MoEGate(nn.Module):
    """
    MoE 路由门控模块：决定每个 token 应该被送到哪些专家处理

    核心职责：
        1. 对每个 token 计算它与每个专家的匹配分数
        2. 选出 top_k 个专家
        3. 训练时附加负载均衡辅助损失，防止所有 token 都涌向同一个专家
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # 每个 token 选几个专家
        self.top_k = config.num_experts_per_tok
        # 专家总数
        self.n_routed_experts = config.n_routed_experts
        # 打分函数
        self.scoring_func = config.scoring_func
        # 辅助损失的权重系数，控制负载均衡强度
        self.alpha = config.aux_loss_alpha
        # True=按序列维度算辅助损失，False=按token维度
        self.seq_aux = config.seq_aux
        # 是否对 top_k 权重重新归一化
        self.norm_topk_prob = config.norm_topk_prob
        # 门控输入维度
        self.gating_dim = config.hidden_size
        # 门控权重矩阵：[hidden_size, n_routed_experts]
        # 每个 token 的隐藏状态乘以此矩阵，得到对每个专家的原始打分 logits
        self.weight = nn.Parameter(torch.empty(self.n_routed_experts, self.gating_dim))
        # 权重初始化，使用 Kaiming 均匀分布，适合线性层的权重
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, hidden_states):
        # hidden_states shape: [batch_size, sequence_len, hidden_size]
        batch_size, sequence_len, hidden_size = hidden_states.shape
        # 把 batch 和 seq_len 合并，把每个 token 都当成独立样本处理
        # shape: [batch_size * sequence_len, hidden_size]
        hidden_states = hidden_states.view(-1, hidden_size)

        # ── 1. 计算每个 token 对每个专家的原始分数 ─────────────────────
        # F.linear(x, W) = x @ W.T
        # [batch_size * sequence_len, hidden_size] @ [hidden_size, n_routed_experts]
        # → logits shape: [batch_size * sequence_len, n_routed_experts]
        logits = F.linear(hidden_states, self.weight, None)

        # ── 2. 归一化：把 logits 转成概率分布 ───────────────────────────
        if self.scoring_func == "softmax":
            # softmax 使所有专家的分数之和为 1
            # scores shape: [batch_size * sequence_len, n_routed_experts]
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        # ── 3. 选出 top_k 个专家 ─────────────────────────────────────────
        # topk_weight shape: [batch_size * sequence_len, top_k]  每个专家的路由权重
        # topk_idx shape: [batch_size * sequence_len, top_k]  专家编号（0 ~ n_routed_experts-1）
        # sorted=False：不要求按分数排序，速度更快
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # ── 4. 对 top_k 权重重新归一化（可选）────────────────────────────
        # 只取了 top_k 个专家，它们的权重之和不再是 1
        # 归一化后：这 top_k 个专家的权重之和 = 1，保证输出幅度稳定
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # ── 5. 计算辅助负载均衡损失（仅训练时）──────────────────────────
        # 问题背景：如果不加约束，门控网络会倾向于总是选同几个专家
        # 因为这些专家更新更多，变得更强，形成马太效应
        # 辅助损失的目标：让每个专家被选中的频率尽量均等
        if self.training and self.alpha > 0.0:
            # 保存完整的 scores 用于辅助损失计算
            scores_for_aux = scores
            aux_topk = self.top_k
            # topk_idx_for_aux_loss shape: [batch_size, sequence_len * top_k]
            # 把所有 token 选中的专家编号展开成一个列表
            topk_idx_for_aux_loss = topk_idx.view(batch_size, -1)
            if self.seq_aux:
                # ── 方法A：序列级辅助损失（seq_aux=True，粒度更细）──────
                # scores_for_seq_aux shape: [batch_size, sequence_len, n_routed_experts]
                scores_for_seq_aux = scores_for_aux.view(batch_size, sequence_len, -1)
                # ce：统计每个 batch 中每个专家被选中了多少次
                # shape: [batch_size, n_routed_experts]，初始化为全 0
                ce = torch.zeros(
                    batch_size, self.n_routed_experts, device=hidden_states.device
                )
                # scatter_add_：在 dim=1（专家维度）上累加计数
                # topk_idx_for_aux_loss 是每个 token 选中的专家编号
                # 对应位置 +1，统计每个专家被选中的总次数
                # 除以 (sequence_len * top_k / n_experts)，得到相对负载
                # 理想情况下每个专家负载均等，相对负载都为 1.0
                # 若某专家被选了很多次，相对负载 > 1，会产生更大的惩罚
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(batch_size, sequence_len * aux_topk, device=hidden_states.device),
                ).div_(sequence_len * aux_topk / self.n_routed_experts)
                # aux_loss = mean(每个专家的平均路由概率 × 相对负载)
                # 专家被选得越多 且 得分越高 → aux_loss 越大 → 梯度会推动门控分散
                # scores_for_seq_aux.mean(dim=1) shape: [batch_size, n_routed_experts]
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean() * self.alpha
            else:
                # ── 方法B：token 级辅助损失（seq_aux=False，DeepSeek 原版）─
                # mask_ce：one-hot 编码，每行代表一个（token, 专家）的选中情况
                # shape: [batch_size * sequence_len * top_k, n_routed_experts]
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                # ce：每个专家被选中的频率（在所有 token 中的比例）
                # shape: [n_routed_experts]
                ce = mask_ce.float().mean(0)
                # Pi：每个专家的平均路由概率（软概率，可微）
                # shape: [n_routed_experts]
                Pi = scores_for_aux.mean(0)
                # fi：实际负载（硬选择频率 × 专家数 = 归一化后的负载系数）
                # 理想情况每个 fi = 1.0，如果某个专家被选得过多，fi 会大于 1，产生更大的惩罚
                fi = ce * self.n_routed_experts
                # aux_loss = sum(Pi * fi) × alpha
                # 直觉：Pi 可微，fi 不可微（来自 argmax）
                # 用 Pi（软概率）作为代理来传递梯度，推动负载均衡
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            # 推理阶段或 alpha=0：不计算辅助损失，直接返回零
            aux_loss = scores.new_zeros(1).squeeze()
        # topk_idx    : [bsz*seq_len, top_k]  每个 token 被路由到的专家编号
        # topk_weight : [bsz*seq_len, top_k]  对应的路由权重
        # aux_loss    : 标量，负载均衡损失，需加到总 loss 中一起反向传播
        return topk_idx, topk_weight, aux_loss

class MoEFeedForward(nn.Module):
    """
    MoE（Mixture of Experts）前馈层。

    标准 FFN 的升级版：把一个大 FFN 拆成 N 个小专家 FFN，
    每个 token 只激活其中 top_k 个专家，计算量不变但模型容量大幅提升。

    结构：
        routed experts（路由专家）：由门控动态选择，每个 token 只用其中 top_k 个
        shared experts （共享专家）：所有 token 都会经过，用于学习通用知识
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config

        # 路由专家列表：n_routed_experts 个独立的 FeedForward 模块
        # 每个专家结构相同，但参数完全独立，各自学习不同的知识
        self.experts = nn.ModuleList(
            [FeedForward(config) for _ in range(config.n_routed_experts)]
        )

        # 门控模块：决定每个 token 路由到哪 top_k 个专家
        self.gate = MoEGate(config)

        # 共享专家（可选）：不参与路由，所有 token 都会经过
        # 用于捕获跨 token 的通用模式，补充路由专家的专业化能力
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [FeedForward(config) for _ in range(config.n_shared_experts)]
            )

    def forward(self, x):
        identity = x # 保存原始输入，用于后面加共享专家的输出
        orig_shape = x.shape # 记录原始形状，最后需要恢复
        batch_size, sequence_len, hidden_size = orig_shape  # batch_size, 序列长度, 隐藏维度

        # ── 1. 门控：为每个 token 选出 top_k 个专家 ────────────────────
        # topk_idx    shape: [batch_size * sequence_len, top_k]  被选中的专家编号
        # topk_weight shape: [batch_size * sequence_len, top_k]  对应的路由权重（归一化后之和=1）
        # aux_loss    : 标量，负载均衡损失
        topk_idx, topk_weight, aux_loss = self.gate(x)

        # 将 x 展平：[batch_size, sequence_len, hidden_size] → [batch_size * sequence_len, hidden_size]
        # 之后把每个 token 当作独立样本处理
        x = x.view(-1, x.shape[-1])

        # flat_topk_idx shape: [batch_size * sequence_len * top_k]
        # 把所有 token 的所有专家选择展平成一维，方便后续按专家编号筛选
        # 例：top_k=2, 3个token → [专家a, 专家b, 专家c, 专家d, 专家e, 专家f]
        flat_topk_idx = topk_idx.view(-1)

        # ── 2A. 训练阶段：逐专家循环处理 ───────────────────────────────
        if self.training:
            # repeat_interleave：把每个 token 复制 top_k 份
            # [token0, token1, token2] → [token0, token0, token1, token1, token2, token2]
            # 这样第 i 份 token 就对应 flat_topk_idx[i] 号专家
            # x shape: [batch_size * sequence_len * top_k, hidden_size]
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)

            # 预分配输出缓冲区，形状与 repeat 后的 x 相同
            y = torch.empty_like(x, dtype=x.dtype)

            for i, expert in enumerate(self.experts):
                # flat_topk_idx == i：找出所有被路由到专家 i 的位置（bool 掩码）
                # x[flat_topk_idx == i]：取出这些位置对应的 token
                expert_out = expert(x[flat_topk_idx == i])

                if expert_out.shape[0] > 0:
                    # 正常情况：把专家输出写回 y 对应位置
                    y[flat_topk_idx == i] = expert_out.to(y.dtype)
                else:
                    # 该专家本轮没有被分配到任何 token（负载不均时可能发生）
                    # 加上 0 * sum(参数) 的目的：
                    #   - 数值上不影响结果（乘 0）
                    #   - 但让这个专家的参数出现在计算图中，梯度得以流过
                    #   - 避免"死专家"的参数完全没有梯度，保证所有专家都在训练
                    y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(
                        p.sum() for p in expert.parameters()
                    )

            # ── 加权求和，合并 top_k 个专家的输出 ──────────────────────
            # y 当前 shape: [batch_size * sequence_len * top_k, hidden_size]
            # → view(*topk_weight.shape, -1): [batch_size * sequence_len, top_k, hidden_size]
            # → * topk_weight.unsqueeze(-1) : [batch_size * sequence_len, top_k, hidden_size]（逐专家加权）
            # → .sum(dim=1)                 : [batch_size * sequence_len, hidden_size]（top_k 个专家输出求和）
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)

            # 恢复原始形状 [batch_size, sequence_len, hidden_size]
            y = y.view(*orig_shape)

        # ── 2B. 推理阶段：按专家批量处理（更高效）──────────────────────
        # 训练时逐专家循环没问题（反向传播需要计算图），
        # 推理时用 moe_infer，把同一专家的所有 token 打包一次处理，减少重复调用开销
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(
                *orig_shape
            )

        # ── 3. 加入共享专家的输出 ───────────────────────────────────────
        # 共享专家处理原始输入 identity（不是路由后的 x），结果直接相加
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)

        # 把辅助损失挂在模块属性上，外部训练循环可以通过 model.aux_loss 取到并加入总 loss
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()  # 推理阶段不需要计算梯度，节省显存和计算
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        推理专用的高效 MoE 分发方法。

        核心思路：
            训练时每次只处理一个专家的 token（for i in experts），调用次数 = 专家数
            推理时先排序，把同一专家的 token 聚集在一起，再批量处理，减少碎片化调用

        Args:
            x                  : [batch_size * sequence_len, hidden_size]，展平的所有 token
            flat_expert_indices: [batch_size * sequence_len * top_k]，每个（token, 专家）对的专家编号
            flat_expert_weights: [batch_size * sequence_len * top_k, 1]，对应的路由权重
        """
        # 预分配输出缓冲区，初始化为 0
        # 后面用 scatter_add_ 累加每个专家的加权输出
        expert_cache = torch.zeros_like(x)

        # ── 排序：把相同专家的 token 聚集在一起 ─────────────────────────
        idxs = flat_expert_indices.argsort()

        # ── 统计每个专家分到的 token 数，计算分段边界 ────────────────────
        # bincount：统计 0~n_experts-1 每个值出现次数
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)

        # ── 计算每个排序后位置对应的原始 token 编号 ─────────────────────
        # flat_expert_indices 是 token 被 repeat_interleave(top_k) 展开后的结果
        # 第 j 个位置对应第 j // top_k 个 token
        token_idxs = idxs // self.config.num_experts_per_tok

        # ── 按专家分段处理 ───────────────────────────────────────────────
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]

            # 该专家没有被分配到任何 token，跳过
            if start_idx == end_idx:
                continue

            expert = self.experts[i] # 取出第 i 个专家

            # 取出属于该专家的原始 token 编号（在 x 中的行索引）
            # shape: [该专家负责的token数]
            exp_token_idx = token_idxs[start_idx:end_idx]

            # 根据 token 编号从 x 中取出对应的隐藏状态
            # shape: [该专家负责的token数, h]
            expert_tokens = x[exp_token_idx]

            # 批量过专家 FFN，一次处理该专家所有 token（高效）
            # shape: [该专家负责的token数, h]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)

            # 乘以路由权重（inplace 操作，节省显存）
            # flat_expert_weights[idxs[start_idx:end_idx]] 取出对应位置的权重
            # shape: [该专家负责的token数, 1]，广播到 [该专家负责的token数, h]
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])

            # scatter_add_：把加权结果累加回 expert_cache 对应 token 的位置
            # exp_token_idx.view(-1,1).repeat(1, h)：把 token 编号扩展到每个维度
            # 如果一个 token 被两个专家处理，两次 scatter_add_ 会自动累加（加权求和）
            expert_cache.scatter_add_(
                0,
                exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]),
                expert_out,
            )

        # expert_cache shape: [bsz*seq_len, h]
        # 每个 token 的位置已经累加了所有分配给它的专家的加权输出
        return expert_cache
        
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.attention = Attention(config)
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = (
            FeedForward(config) if not config.use_moe else MoEFeedForward(config)
        )

    def forward(
        self,
        hidden_states,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # 1. 注意力子层（Pre-Norm + 残差）
        res = hidden_states
        hidden_states, present_key_value = self.attention(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        hidden_states = res + hidden_states

        # 2. FFN 子层（Pre-Norm + 残差）
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )

        return hidden_states, present_key_value
    
class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = (config.vocab_size, config.num_hidden_layers)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.block_layers = nn.ModuleList(
            [MiniMindBlock(l, config) for l in range(self.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        # 1. 获取输入的 batch_size 和 seq_length
        batch_size, seq_length = input_ids.shape

        # 2. 创建缓存列表，如果 past_key_values 没有提供，则初始化为 None 的列表，长度与层数相同
        if hasattr(past_key_values, "layers"):
            past_key_values = None
        # past_key_values = [
        #   (k0, v0),   # 第0层的 KV cache
        #   (k1, v1),   # 第1层的 KV cache
        #   (k2, v2),   # 第2层的 KV cache
        #   ...
        # ]
        past_key_values = past_key_values or [None] * len(self.block_layers)
        # 3. 计算start_pos：如果存在past，则start_pos为已有past序列长度，计算当前 token 的起始位置索引，以便从预计算的 RoPE 频率表中取出正确的位置编码
        # k.shape = [batch, past_seq_len, num_kv_heads, head_dim]， v.shape = [batch, past_seq_len, num_kv_heads, head_dim]
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        # 4. Embedding + dropout
        hidden_states = self.dropout(
            self.embed_tokens(input_ids)
        )

        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_length],
            self.freqs_sin[start_pos : start_pos + seq_length],
        )
        
        presents = []
        for layer_idx, (block_layer, past_key_value) in enumerate(
            zip(self.block_layers, past_key_values)
        ):
            hidden_states, present = block_layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        aux_loss = sum(
            [
                block_layer.mlp.aux_loss
                for block_layer in self.block_layers
                if isinstance(
                    block_layer.mlp, MoEFeedForward
                )
            ],
            hidden_states.new_zeros(1).squeeze(),
        )
        
        return hidden_states, presents, aux_loss

class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    def __init__(self, config: MiniMindConfig):
        super().__init__(config)
        # 主体模型，包含 Embedding + 所有 Transformer 层
        self.model = MiniMindModel(config)
        # 输出头：将隐藏层维度映射到词汇表大小，不使用偏置
        # hidden_size → vocab_size，用于预测下一个 token
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 权重共享：让 embed_tokens 和 lm_head 共享同一份权重
        # 输入 Embedding 和输出投影互为转置，节省参数量
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **args,
    ):
        # -------------------------------------------------------
        # 1. 主体模型前向传播
        #    输入 token ids → 经过 Embedding + N 层 Transformer
        #    输出每个位置的隐藏状态 hidden_states
        # hidden_states shape: [B, seq_len, hidden_size]
        # -------------------------------------------------------
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args,
        )

        # -------------------------------------------------------
        # 2. 决定对哪些位置计算 logits
        #    logits_to_keep=0：保留所有位置（训练时）
        #    logits_to_keep=1：只保留最后1个位置（推理时只需预测下一个token）
        #    logits_to_keep=n：只保留最后 n 个位置
        # -------------------------------------------------------
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        # lm_head：hidden_size → vocab_size
        # logits shape: [B, seq_len, vocab_size]，每个位置对应词汇表中每个词的得分
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # -------------------------------------------------------
        # 3. 计算损失（训练时才有 labels）
        # -------------------------------------------------------
        loss = None
        if labels is not None:
            # 语言模型的训练目标：用当前token预测下一个token
            # 所以 logits 和 labels 需要错位对齐：
            #
            # input:  [t0, t1, t2, t3, t4]
            # logits: [t0, t1, t2, t3, t4] → 去掉最后一个 → [t0, t1, t2, t3]
            # labels: [t0, t1, t2, t3, t4] → 去掉第一个  → [t1, t2, t3, t4]
            #
            # 含义：用 t0 预测 t1，用 t1 预测 t2，以此类推
            shift_logits = logits[..., :-1, :].contiguous() # shape: [B, seq_len-1, vocab_size]
            shift_labels = labels[..., 1:].contiguous() # shape: [B, seq_len-1]
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), # [B×(seq_len-1), vocab_size]
                shift_labels.view(-1), # [B×(seq_len-1)]
                reduction='none',
            ).view(shift_labels.size()) # [B, seq_len-1]

            if loss_mask is not None:
                loss_mask = loss_mask[:, 1:] # 只对有效位置计算损失
                loss = (loss * loss_mask).sum() / loss_mask.sum()
            else :
                loss = loss.mean() # 平均所有位置的损失
        # -------------------------------------------------------
        # 4. 封装输出，统一返回格式
        # 注意力权重只在可视化分析时有用，正常训练推理不需要，且保存它会浪费大量显存，所以 MiniMind 直接不收集，设为默认 None
        # -------------------------------------------------------
        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )

        output.aux_loss = aux_loss

        return output