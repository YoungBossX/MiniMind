"""统一模型/Tokenizer/Checkpoint 加载

支持两种模式：
  - native: 加载项目原生 torch checkpoint（MiniMindForCausalLM）
  - huggingface: 加载 HuggingFace transformers 格式权重
"""

import os
import sys
import torch
from typing import Tuple, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from model.MiniMindModel import MiniMindConfig, MiniMindForCausalLM


def load_model_and_tokenizer(
    checkpoint_path: str = "",
    tokenizer_path: str = "",
    hidden_size: int = 512,
    num_hidden_layers: int = 8,
    use_moe: bool = False,
    device: str = "cpu",
    dtype: str = "auto",
    lora_path: Optional[str] = None,
    lora_rank: int = 8,
    lora_alpha: int = 16,
) -> Tuple:
    """统一加载模型和 tokenizer

    Args:
        checkpoint_path: .pth 权重文件路径。为空则只初始化未训练模型。
        tokenizer_path: tokenizer 目录路径。为空则使用项目 model/ 目录。
        hidden_size: 隐藏维度
        num_hidden_layers: Transformer 层数
        use_moe: 是否使用 MoE 架构
        device: 运行设备
        dtype: "auto" | "fp32" | "fp16" | "bf16"
        lora_path: LoRA 权重路径（可选）
        lora_rank: LoRA 秩
        lora_alpha: LoRA alpha 参数

    Returns:
        (model, tokenizer)
    """
    from transformers import AutoTokenizer

    if not tokenizer_path:
        tokenizer_path = os.path.join(_PROJECT_ROOT, "model")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    config = MiniMindConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        use_moe=use_moe,
    )
    model = MiniMindForCausalLM(config)

    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)

    if lora_path and os.path.exists(lora_path):
        from model.model_lora import apply_lora, load_lora
        apply_lora(model, lora_rank, lora_alpha, ["q_proj", "v_proj", "k_proj", "o_proj"])
        load_lora(model, lora_path)

    if dtype == "auto":
        dtype = "fp32"
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    model = model.to(device=device, dtype=dtype_map.get(dtype, torch.float32))
    model.eval()

    return model, tokenizer
