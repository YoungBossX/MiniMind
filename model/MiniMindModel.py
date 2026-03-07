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
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
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
            if self.num_key_value_heads is None
            else self.num_key_value_heads
        )
        
        assert args.num_attention_heads % self.num_key_value_heads == 0

        self.n_local_heads = args.num_attention_heads
        self.n_local_heads = self.num_key_value_heads
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
        # 2. 线性变换得到 q, k, v
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
    
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.attention = Attention(config)
        self.layer_id = layer_id
        self.intput_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = (
            FeedForward(config)
        )

    def forward(
        self,
        hidden_states,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # 1. 对输入的 hidden_states 进行层归一化，得到 normed_hidden_states
        res = hidden_states
        hidden_states, present_key_value = self.attention(
            self.intput_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        # 2. 将注意力输出与残差连接相加，得到 attn_output
        hidden_states = res + hidden_states
        # 3. 对 attn_output 进行层归一化，得到 normed_attn_output
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
        self.layers = nn.ModuleList(
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
        
        past_key_values = past_key_values or [None] * len(self.layers)
        # 3. 计算start_pos：如果存在past，则start_pos为已有past序列长度，计算当前 token 的起始位置索引，以便从预计算的 RoPE 频率表中取出正确的位置编码
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
        for layer_idx, (layer, past_key_value) in enumerate(
            zip(self.layers, past_key_values)
        ):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)
        
        return hidden_states, presents

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
        hidden_states, past_key_values = self.model(
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
                ignore_index=-100, # labels 中 -100 的位置不计算损失（padding 或 prompt 部分）
            )
            
        # -------------------------------------------------------
        # 4. 封装输出，统一返回格式
        # -------------------------------------------------------
        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )

        return output