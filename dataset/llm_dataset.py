from torch.utils.data import Dataset
import torch
import os
import json
import random
from datasets import load_dataset

# 禁用 HuggingFace tokenizer 的多进程并行，避免在 DataLoader 多进程环境中产生死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

_CHATML_RESERVED_DELIMITERS = ("<|im_start|>", "<|im_end|>")


def _approximate_text_length(text):
    """Cheap length signal used only to locally group offline training samples."""
    return max(1, len(str(text)))


def _approximate_conversation_length(conversations):
    """Estimate ChatML size without rendering or tokenizing the conversation."""
    return max(
        1,
        sum(
            _approximate_text_length(turn.get("content", ""))
            for turn in conversations
            if isinstance(turn, dict)
        ),
    )


def _find_last_subsequence(sequence, pattern):
    if not pattern or len(pattern) > len(sequence):
        return -1
    for index in range(len(sequence) - len(pattern), -1, -1):
        if sequence[index:index + len(pattern)] == pattern:
            return index
    return -1


def _reject_reserved_chatml_content(conversations, sample_name):
    """Reject unescaped delimiters before the chat template adds boundaries."""
    for turn_index, turn in enumerate(conversations):
        if not isinstance(turn, dict):
            continue
        content = turn.get("content")
        if not isinstance(content, str):
            continue
        for delimiter in _CHATML_RESERVED_DELIMITERS:
            if delimiter in content:
                raise ValueError(
                    f"{sample_name} turn {turn_index} contains reserved "
                    f"ChatML delimiter {delimiter!r}"
                )


def _validate_chatml_start_boundaries(
    input_ids, message_start_id, eos_id, sample_name
):
    """Require every rendered ChatML start marker to follow a legal boundary."""
    for index in range(len(input_ids) - len(message_start_id) + 1):
        if input_ids[index:index + len(message_start_id)] != message_start_id:
            continue
        if index == 0 or (
            index >= len(eos_id)
            and input_ids[index - len(eos_id):index] == eos_id
        ):
            continue
        raise ValueError(
            f"{sample_name} contains an unbounded reserved ChatML delimiter"
        )


def _assistant_loss_mask(input_ids, assistant_start_id, eos_id, max_length):
    """Mask assistant spans whose headers begin at valid ChatML boundaries."""
    loss_mask = [0] * len(input_ids)
    index = 0
    while index < len(input_ids):
        is_assistant_header = (
            input_ids[index:index + len(assistant_start_id)]
            == assistant_start_id
        )
        has_valid_boundary = index == 0 or (
            index >= len(eos_id)
            and input_ids[index - len(eos_id):index] == eos_id
        )
        if not is_assistant_header or not has_valid_boundary:
            index += 1
            continue

        content_start = index + len(assistant_start_id)
        content_end = content_start
        while content_end < len(input_ids):
            if input_ids[content_end:content_end + len(eos_id)] == eos_id:
                break
            content_end += 1
        span_end = min(content_end + len(eos_id), max_length, len(input_ids))
        for token_index in range(content_start, span_end):
            loss_mask[token_index] = 1
        index = span_end if span_end > index else index + 1
    return loss_mask


def _find_last_bounded_subsequence(sequence, pattern, eos_id):
    """Find a marker that begins the sequence or follows a ChatML EOS."""
    if not pattern or len(pattern) > len(sequence):
        return -1
    for index in range(len(sequence) - len(pattern), -1, -1):
        if sequence[index:index + len(pattern)] != pattern:
            continue
        if index == 0 or (
            eos_id
            and index >= len(eos_id)
            and sequence[index - len(eos_id):index] == eos_id
        ):
            return index
    return -1


def _chatml_message_parts(message, message_start_id, header_end_id, eos_id):
    """Return a complete ChatML message's header and content token slices."""
    message = list(message)
    if (
        not message_start_id
        or not header_end_id
        or not eos_id
        or message[:len(message_start_id)] != message_start_id
        or message[-len(eos_id):] != eos_id
    ):
        return None

    search_start = max(len(message_start_id) - len(header_end_id), 0)
    eos_start = len(message) - len(eos_id)
    for index in range(search_start, eos_start - len(header_end_id) + 1):
        if message[index:index + len(header_end_id)] == header_end_id:
            content_start = index + len(header_end_id)
            return message[:content_start], message[content_start:eos_start]
    return None


def _chatml_message_min_length(message, message_start_id, header_end_id, eos_id):
    parts = _chatml_message_parts(
        message, message_start_id, header_end_id, eos_id
    )
    if parts is None:
        return 0
    header, content = parts
    return len(header) + len(eos_id) + int(bool(content))


def _truncate_chatml_message_tail(
    message,
    message_start_id,
    header_end_id,
    eos_id,
    max_length,
):
    """Keep a valid message header, its newest content tokens, and full EOS."""
    parts = _chatml_message_parts(
        message, message_start_id, header_end_id, eos_id
    )
    if parts is None:
        return []
    header, content = parts
    content_budget = max_length - len(header) - len(eos_id)
    minimum_content = int(bool(content))
    if content_budget < minimum_content:
        return []
    if len(content) > content_budget:
        content = content[-content_budget:] if content_budget else []
    return header + content + eos_id


def _bounded_message_starts(sequence, message_start_id, eos_id):
    starts = []
    for index in range(len(sequence) - len(message_start_id) + 1):
        if sequence[index:index + len(message_start_id)] != message_start_id:
            continue
        if index == 0 or (
            eos_id
            and index >= len(eos_id)
            and sequence[index - len(eos_id):index] == eos_id
        ):
            starts.append(index)
    return starts


def _latest_chatml_message_min_length(
    prefix, message_start_id, header_end_id, eos_id
):
    starts = _bounded_message_starts(prefix, message_start_id, eos_id)
    if not starts:
        return 0
    return _chatml_message_min_length(
        prefix[starts[-1]:], message_start_id, header_end_id, eos_id
    )


def _complete_chatml_prefix_suffix(
    prefix,
    message_start_id,
    header_end_id,
    eos_id,
    max_length,
):
    """Return the newest complete ChatML context that fits ``max_length``."""
    if max_length <= 0:
        return []

    starts = _bounded_message_starts(prefix, message_start_id, eos_id)
    if not starts:
        return []

    best_suffix = []
    for index in reversed(starts):
        candidate = prefix[index:]
        if len(candidate) > max_length:
            break
        best_suffix = candidate
    if best_suffix:
        return best_suffix

    return _truncate_chatml_message_tail(
        prefix[starts[-1]:],
        message_start_id,
        header_end_id,
        eos_id,
        max_length,
    )


def _split_final_assistant(input_ids, bos_id, eos_id):
    input_ids = list(input_ids)
    assistant_start = _find_last_bounded_subsequence(input_ids, bos_id, eos_id)
    final_message_start = _find_last_bounded_subsequence(
        input_ids, bos_id[:1], eos_id
    )
    if assistant_start < 0 or assistant_start != final_message_start:
        return input_ids, None

    content_start = assistant_start + len(bos_id)
    eos_start = _find_last_subsequence(input_ids, eos_id)

    target_content = (
        input_ids[content_start:]
        if eos_start < content_start
        else input_ids[content_start:eos_start]
    )
    assistant_segment = bos_id + target_content + eos_id
    return input_ids[:assistant_start], assistant_segment


def _truncate_assistant_segment(assistant_segment, bos_id, eos_id, max_length):
    truncated = _truncate_chatml_message_tail(
        assistant_segment,
        bos_id,
        bos_id[-1:],
        eos_id,
        max_length,
    )
    if not truncated:
        raise ValueError(
            "max_length is too small for assistant content and ChatML boundaries"
        )
    return truncated


def _truncate_preserving_final_assistant(input_ids, bos_id, eos_id, max_length):
    """Keep the final assistant target and as much recent context as fits."""
    input_ids = list(input_ids)
    if len(input_ids) <= max_length:
        return input_ids

    prefix, assistant_segment = _split_final_assistant(input_ids, bos_id, eos_id)
    if assistant_segment is None:
        return input_ids[:max_length]

    message_start_id = bos_id[:1]
    header_end_id = bos_id[-1:]
    prefix_min_length = _latest_chatml_message_min_length(
        prefix, message_start_id, header_end_id, eos_id
    )
    assistant_min_length = _chatml_message_min_length(
        assistant_segment, bos_id, header_end_id, eos_id
    )
    prefix_reserve = (
        prefix_min_length
        if (
            prefix_min_length
            and max_length >= prefix_min_length + assistant_min_length
        )
        else 0
    )
    assistant_segment = _truncate_assistant_segment(
        assistant_segment, bos_id, eos_id, max_length - prefix_reserve
    )
    prefix_budget = max_length - len(assistant_segment)
    complete_prefix = _complete_chatml_prefix_suffix(
        prefix,
        message_start_id,
        header_end_id,
        eos_id,
        prefix_budget,
    )
    return complete_prefix + assistant_segment


def _truncate_dpo_pair(chosen_ids, rejected_ids, bos_id, eos_id, max_length):
    chosen_prefix, chosen_segment = _split_final_assistant(
        chosen_ids, bos_id, eos_id
    )
    rejected_prefix, rejected_segment = _split_final_assistant(
        rejected_ids, bos_id, eos_id
    )
    if chosen_segment is None or rejected_segment is None:
        return (
            _truncate_preserving_final_assistant(
                chosen_ids, bos_id, eos_id, max_length
            ),
            _truncate_preserving_final_assistant(
                rejected_ids, bos_id, eos_id, max_length
            ),
        )
    if chosen_prefix != rejected_prefix:
        raise ValueError("DPO chosen and rejected branches must share one prompt")

    message_start_id = bos_id[:1]
    header_end_id = bos_id[-1:]
    prefix_min_length = _latest_chatml_message_min_length(
        chosen_prefix, message_start_id, header_end_id, eos_id
    )
    assistant_min_length = max(
        _chatml_message_min_length(
            chosen_segment, bos_id, header_end_id, eos_id
        ),
        _chatml_message_min_length(
            rejected_segment, bos_id, header_end_id, eos_id
        ),
    )
    prefix_reserve = (
        prefix_min_length
        if (
            prefix_min_length
            and max_length >= prefix_min_length + assistant_min_length
        )
        else 0
    )
    assistant_budget = max_length - prefix_reserve
    chosen_segment = _truncate_assistant_segment(
        chosen_segment, bos_id, eos_id, assistant_budget
    )
    rejected_segment = _truncate_assistant_segment(
        rejected_segment, bos_id, eos_id, assistant_budget
    )
    prefix_budget = max_length - max(len(chosen_segment), len(rejected_segment))
    common_prefix = _complete_chatml_prefix_suffix(
        chosen_prefix,
        message_start_id,
        header_end_id,
        eos_id,
        prefix_budget,
    )
    return common_prefix + chosen_segment, common_prefix + rejected_segment


def _pad_token_ids(input_ids, max_length, pad_token_id):
    input_ids = list(input_ids[:max_length])
    attention_mask = [1] * len(input_ids)
    padding = max_length - len(input_ids)
    if padding:
        input_ids.extend([pad_token_id] * padding)
        attention_mask.extend([0] * padding)
    return input_ids, attention_mask


def dynamic_padding_collate(batch):
    """Trim right padding in fixed-shape offline-training samples before stacking.

    Datasets keep their fixed-length ``__getitem__`` contract. This collate
    function only removes columns that are padding for every sample in a batch.
    """
    first_sample = batch[0]
    if isinstance(first_sample, tuple):
        max_length = max(1, max(int(sample[3].sum().item()) for sample in batch))
        return tuple(
            torch.stack([sample[field_index][:max_length] for sample in batch])
            for field_index in range(4)
        )

    if isinstance(first_sample, dict):
        result = {}
        for branch in ("chosen", "rejected"):
            max_length = max(
                1,
                max(
                    int(sample[f"attention_mask_{branch}"].sum().item())
                    for sample in batch
                ),
            )
            for field_name in (
                f"x_{branch}",
                f"y_{branch}",
                f"mask_{branch}",
                f"attention_mask_{branch}",
            ):
                result[field_name] = torch.stack(
                    [sample[field_name][:max_length] for sample in batch]
                )
        return result

    raise TypeError(f"Unsupported offline batch sample type: {type(first_sample)!r}")


def pre_processing_chat(conversations, add_system_ratio=0.2):
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    if conversations and conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.05, rng=None):
    rng = random if rng is None else rng
    if '<think>\n\n</think>\n\n' in prompt_content and rng.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

# ──────────────────────────────────────────────────────────────────────────────
# 1. PretrainDataset —— 自回归预训练数据集
# ──────────────────────────────────────────────────────────────────────────────
# 训练目标：Next-Token Prediction（下一个 token 预测）
# 数据格式：{"text": "一段原始文本"}
# 训练特点：
#   - 模型对整段文本的每个位置都进行预测，没有"只学回复"的区分。
#   - 使用 BOS/EOS 标记文本边界，让模型学会文本的起止。
#   - PAD token 对应的 label 置 -100，不参与 loss 计算，节省无效梯度。
#   - labels 直接 clone 自 input_ids（即 X 和 Y 错位一格：Y[t] = X[t+1]）。
# ──────────────────────────────────────────────────────────────────────────────
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        max_length = int(max_length)
        if max_length < 2:
            raise ValueError("Pretrain max_length must be at least 2")
        self.tokenizer = tokenizer      # 分词器，用于将文本转为token ID
        self.max_length = max_length    # 每条样本的最大token长度
        self.samples = self.load_data(data_path)  # 加载数据
        self.lengths = [
            self._estimate_length(sample['text']) for sample in self.samples
        ]

    def _estimate_length(self, text):
        return _approximate_text_length(text)

    def load_data(self, path):
        """从文件中加载数据，每一行为一条JSON格式的样本"""
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # 读取每一行，解析成字典结构
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        """返回样本数量"""
        return len(self.samples)

    def __getitem__(self, index):
        """
        返回第 index 个样本：
        - X: 模型输入（input_ids[:-1]）
        - Y: 目标输出（input_ids[1:]）
        - loss_mask: 哪些token位置参与loss计算（去除padding部分）
        """
        sample = self.samples[index]

        # 将样本中的文本字段进行 tokenize
        input_ids = self.tokenizer(
            str(sample['text']), add_special_tokens=False
        ).input_ids
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("pretraining requires a tokenizer EOS token")
        if not input_ids or input_ids[-1] != eos_token_id:
            input_ids.append(eos_token_id)
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length - 1] + [eos_token_id]
        input_ids, attention_mask = _pad_token_ids(
            input_ids, self.max_length, self.tokenizer.pad_token_id
        )
        
        # 计算 loss_mask：pad 的位置不参与 loss
        loss_mask = [
            token_id != self.tokenizer.pad_token_id for token_id in input_ids
        ]

        # 语言模型是自回归的，使用前一个 token 预测下一个
        X = torch.tensor(input_ids[:-1], dtype=torch.long)         # 输入：[0, ..., n-2]
        Y = torch.tensor(input_ids[1:], dtype=torch.long)          # 目标：[1, ..., n-1]
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # loss_mask 对齐目标 Y
        # 对其 X ，提供注意力机制中的掩码
        attention_mask = torch.tensor(attention_mask[:-1], dtype=torch.long)

        return X, Y, loss_mask, attention_mask
    
# ──────────────────────────────────────────────────────────────────────────────
# 2. SFTDataset —— 有监督微调（Supervised Fine-Tuning）数据集
# ──────────────────────────────────────────────────────────────────────────────
# 训练目标：让模型学会"只预测 assistant 回复"，忽略 user/system 输入
# 数据格式：{"conversations": [{"role": "user"/"assistant"/"system", "content": "..."}]}
# 训练特点：
#   - 通过 generate_labels 扫描 bos_id（assistant 回复起始标记）定位每段回复，
#     仅将 assistant 回复的 token 位置设为有效 label，其余全部为 0。
#   - 这样做的意义：让 loss 只反映模型对"正确回答"的拟合，不浪费梯度在
#     用户输入的复现上（用户输入只作为 context，不是预测目标）。
#   - 支持 function calling：若 system 消息携带 "functions" 字段，
#     会透传给 apply_chat_template，生成带工具描述的提示词。
#   - 与 PretrainDataset 的关键区别：标签是"稀疏"的，只有 assistant 部分非 -100。
# ──────────────────────────────────────────────────────────────────────────────
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer                  # 分词器
        self.max_length = max_length                # 最大输入长度（会进行截断或填充）
        self.samples = self.load_data(jsonl_path)   # 加载数据样本
        self.bos_id = tokenizer('<|im_start|>assistant\n', add_special_tokens=False).input_ids # [1, 1078, 538, 501]， [1]是<|im_start|>这个特殊token的id，[1078, 538, 501]是assistant的分词id
        self.eos_id = tokenizer('<|im_end|>\n', add_special_tokens=False).input_ids # [2]
        self.lengths = [
            self._estimate_length(sample['conversations'])
            for sample in self.samples
        ]

    def __len__(self):
        return len(self.samples)  # 返回样本数量

    def load_data(self, path):
        """从 jsonl 文件加载对话数据"""
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())  # 每行为一个 JSON 对象
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        """
        将对话轮构造成符合 ChatML 格式的字符串：
        每一轮用户/助手对话被标注为 'user' / 'assistant'
        最终用 tokenizer 的 apply_chat_template 统一构造 prompt。
        """
        messages = []
        for i, turn in enumerate(conversations):
            messages.append({"role": turn['role'], "content": turn['content']})

        tools = (
            conversations[0]["functions"]
            if (
                conversations
                and conversations[0]["role"] == "system"
                and conversations[0].get("functions")
            )
            else None
        )

        # 返回字符串形式的 prompt，而非直接 tokenize
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def _estimate_length(self, conversations):
        return _approximate_conversation_length(conversations)

    def _generate_loss_mask(self, input_ids):
        """
        构建损失掩码，只有 assistant 的回答部分才参与 loss 计算。
        找出每一段 assistant 的响应，在其 <|im_start|>assistant 和 <|im_end|> 之间设置 loss_mask 为 1。
        """
        return _assistant_loss_mask(
            input_ids, self.bos_id, self.eos_id, self.max_length
        )

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = sample['conversations']
        _reject_reserved_chatml_content(
            conversations, f"SFT sample {index}"
        )

        # 构建 ChatML 格式 prompt（字符串）
        prompt = self._create_chat_prompt(conversations)

        input_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        _validate_chatml_start_boundaries(
            input_ids,
            self.bos_id[:1],
            self.eos_id,
            f"SFT sample {index}",
        )
        input_ids = _truncate_preserving_final_assistant(
            input_ids, self.bos_id, self.eos_id, self.max_length
        )
        input_ids, attention_mask = _pad_token_ids(
            input_ids, self.max_length, self.tokenizer.pad_token_id
        )

        # 生成动态 loss mask，仅对 assistant 响应位置计算 loss
        loss_mask = self._generate_loss_mask(input_ids)

        # 构建训练样本：
        # 模型输入为前 n-1 个 token，预测目标为第 2 到第 n 个 token
        X = torch.tensor(input_ids[:-1], dtype=torch.long)         # 输入序列
        Y = torch.tensor(input_ids[1:], dtype=torch.long)          # 目标标签（shifted）
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐 Y 的位置（从第一个预测 token 开始）
        label_attention_mask = torch.tensor(attention_mask[1:], dtype=torch.long)
        loss_mask = loss_mask * label_attention_mask
        if loss_mask.sum().item() == 0:
            raise ValueError(
                f"SFT sample {index} has no supervised assistant tokens after tokenization"
            )
        # 对其 X ，提供注意力机制中的掩码
        attention_mask = torch.tensor(attention_mask[:-1], dtype=torch.long)

        return X, Y, loss_mask, attention_mask

# ──────────────────────────────────────────────────────────────────────────────
# 3. DPODataset —— 比较学习（Direct Preference Optimization）数据集
# ──────────────────────────────────────────────────────────────────────────────
# 训练目标：让模型学会区分"更好的回答"（chosen ）和"较差的回答"（rejected），通过比较学习优化模型的偏好。
# 数据格式：{"chosen": [{"role": "user"/"assistant"/"system", "content": "..."}], "rejected": [{"role": "user"/"assistant"/"system", "content": "..."}]}
# 训练特点：
#   - 每条样本包含一对对话：一个是"更好的回答"（chosen），另一个是"较差的回答"（rejected）。
#   - 对 chosen 和 rejected 分别构建输入序列，并生成对应的 loss mask。
#   - 通过比较 chosen 和 rejected 的 loss，优化模型更倾向于生成"更好的回答"。
#   - 与 SFTDataset 的关键区别：每条样本包含两套输入输出（chosen 和 rejected），训练时需要同时处理两者，并通过 loss 差异进行优化。
#   - 适用于强化学习前的预训练阶段，让模型学会区分优劣回答，为后续的 RLHF 打下基础。
#   - 注意：DPODataset 生成的训练样本结构更复杂，包含两套输入输出（chosen 和 rejected），需要在训练循环中同时处理两者，并通过 loss 差异进行优化。
# ──────────────────────────────────────────────────────────────────────────────
class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        # 特殊标记 <|im_start|>assistant 和 <|im_end|> 的 token ids（一般是开头和结尾的边界符）
        self.bos_id = tokenizer('<|im_start|>assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>\n', add_special_tokens=False).input_ids

        # 加载 JSONL 格式数据：每行为一个 dict，有 chosen 和 rejected
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = []
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                self.data.append(obj)
        self.lengths = [self._estimate_length(item) for item in self.data]

    def __len__(self):
        return len(self.data)

    def _estimate_length(self, item):
        return max(
            _approximate_conversation_length(item[branch])
            for branch in ('chosen', 'rejected')
        )

    def __getitem__(self, index):
        item = self.data[index]

        chosen = item['chosen']
        rejected = item['rejected']
        invalid_final_assistant = [
            branch_name
            for branch_name, conversation in (
                ("chosen", chosen),
                ("rejected", rejected),
            )
            if (
                not isinstance(conversation, list)
                or not conversation
                or not isinstance(conversation[-1], dict)
                or conversation[-1].get("role") != "assistant"
            )
        ]
        if invalid_final_assistant:
            branches = ", ".join(invalid_final_assistant)
            raise ValueError(
                f"DPO sample {index} {branches} must end with a final assistant message"
            )
        _reject_reserved_chatml_content(chosen, f"DPO sample {index} chosen")
        _reject_reserved_chatml_content(rejected, f"DPO sample {index} rejected")

        # 拼接成字符串（不 tokenize，只生成 prompt 文本）
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )

        chosen_input_ids = self.tokenizer(
            chosen_prompt, add_special_tokens=False
        ).input_ids
        rejected_input_ids = self.tokenizer(
            rejected_prompt, add_special_tokens=False
        ).input_ids
        _validate_chatml_start_boundaries(
            chosen_input_ids,
            self.bos_id[:1],
            self.eos_id,
            f"DPO sample {index} chosen",
        )
        _validate_chatml_start_boundaries(
            rejected_input_ids,
            self.bos_id[:1],
            self.eos_id,
            f"DPO sample {index} rejected",
        )
        chosen_input_ids, rejected_input_ids = _truncate_dpo_pair(
            chosen_input_ids,
            rejected_input_ids,
            self.bos_id,
            self.eos_id,
            self.max_length,
        )
        chosen_input_ids, chosen_attention_mask = _pad_token_ids(
            chosen_input_ids, self.max_length, self.padding
        )
        rejected_input_ids, rejected_attention_mask = _pad_token_ids(
            rejected_input_ids, self.max_length, self.padding
        )

        # 构造 loss mask：仅在 assistant 段落（<|im_start|>assistant ... <|im_end|>）中的 token 参与损失
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)     # shape: (max_length,)
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids) # shape: (max_length,)


        # 构造训练数据：左移一位预测（即 y 是 x 的下一位）
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)      # shape: (max_length - 1,)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)       # shape: (max_length - 1,)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long) * torch.tensor(chosen_attention_mask[1:], dtype=torch.long) # shape: (max_length - 1,)

        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)  # shape: (max_length - 1,)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)   # shape: (max_length - 1,)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long) * torch.tensor(rejected_attention_mask[1:], dtype=torch.long) # shape: (max_length - 1,)
        if mask_chosen.sum().item() == 0:
            raise ValueError(
                f"DPO sample {index} chosen branch has no supervised assistant tokens"
            )
        if mask_rejected.sum().item() == 0:
            raise ValueError(
                f"DPO sample {index} rejected branch has no supervised assistant tokens"
            )

        # X = input_ids[:-1]，attention_mask 也取 [:-1]
        attention_mask_chosen = torch.tensor(chosen_attention_mask[:-1],   dtype=torch.long)
        attention_mask_rejected = torch.tensor(rejected_attention_mask[:-1], dtype=torch.long)

        return {
            'x_chosen': x_chosen,           # shape: (max_length - 1,)
            'y_chosen': y_chosen,           # shape: (max_length - 1,)
            'mask_chosen': mask_chosen,     # shape: (max_length - 1,)
            'attention_mask_chosen': attention_mask_chosen, # shape: (max_length - 1,)

            'x_rejected': x_rejected,       # shape: (max_length - 1,)
            'y_rejected': y_rejected,       # shape: (max_length - 1,)
            'mask_rejected': mask_rejected,  # shape: (max_length - 1,)
            'attention_mask_rejected': attention_mask_rejected # shape: (max_length - 1,)
        }

    def _generate_loss_mask(self, input_ids):
        return _assistant_loss_mask(
            input_ids, self.bos_id, self.eos_id, self.max_length
        )
    
# ──────────────────────────────────────────────────────────────────────────────
# 4. RLAIFDataset —— 基于 AI 反馈的强化学习数据集（用于 PPO / GRPO）
# ──────────────────────────────────────────────────────────────────────────────
# 训练目标：为 RL 训练提供"问题-参考答案"对，由 actor 在线采样生成回复，
#           再由 reward model 或规则函数打分优化
# 数据格式：{"conversations": [{"content": "..."}, {"content": "..."}]}
#   - 奇数索引 (0,2,4...) 为 user 发言
#   - 偶数索引 (1,3,5...) 为 assistant 发言（最后一条为参考答案）
# 训练特点（与前三个 Dataset 的核心区别）：
#   - **不做离线 tokenize**：只返回原始字符串 prompt 和 answer，
#     让 RL trainer（PPO/GRPO）在线 rollout 时自行 tokenize，
#     因为 RL 需要动态生成回复并实时打分，无法预先固定 token 序列。
#   - create_chat_prompt 会剥离最后一条 assistant 消息，
#     将其余对话渲染为带 add_generation_prompt=True 的 prompt，
#     供 actor 模型续写；answer 保存为参考答案用于奖励计算。
#   - bos_id / eos_id 在此类中被定义但目前未用于 mask 计算，
#     保留以备后续扩展（如 reward shaping）需要。
#   - 返回值是 dict{"prompt": str, "answer": str}，而非 tensor，
#     这是 RL 数据集与 SL 数据集（返回 tensor）的最显著差异。
# ──────────────────────────────────────────────────────────────────────────────
class RLAIFDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=1024, seed=42):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.seed = int(seed)
        self.epoch = 0
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        # 特殊标记 <|im_start|>assistant 和 <|im_end|> 的 token ids（一般是开头和结尾的边界符）
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids

        # 加载 JSONL 格式数据：每行为一个 dict，有 chosen 和 rejected
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = []
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                self.data.append(obj)

    def __len__(self):
        return len(self.data)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def create_chat_prompt(self, conversations, index=0):
        messages = []
        answer = ""
        for i, turn in enumerate(conversations):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": turn["content"]})
            answer = turn["content"] # 持续更新，最终保留最后一条 assistant 内容
        # messages[:-1]：去掉最后一条 assistant 回复，只保留上下文
        # add_generation_prompt=True：在末尾追加续写引导 token，告诉模型"现在开始生成"
        prompt = self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        sample_rng = random.Random(
            self.seed + self.epoch * max(len(self.data), 1) + int(index)
        )
        prompt = post_processing_chat(prompt, rng=sample_rng)
        return prompt, answer
    
    def __getitem__(self, index):
        sample = self.data[index]

        if isinstance(sample.get("prompt"), str) and sample["prompt"].strip():
            messages = [{"role": "user", "content": sample["prompt"]}]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return {"prompt": prompt, "answer": sample.get("answer", "")}

        prompt, answer = self.create_chat_prompt(sample["conversations"], index)

        return {"prompt": prompt, "answer": answer}

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("../model")
    print(tokenizer("\n", add_special_tokens=False).input_ids)
    print(tokenizer.convert_ids_to_tokens(tokenizer("\n", add_special_tokens=False).input_ids))

    print(tokenizer("assistant\n", add_special_tokens=False).input_ids)
    print(tokenizer.convert_ids_to_tokens(tokenizer("assistant\n", add_special_tokens=False).input_ids))

    print(tokenizer("<|im_start|>assistant\n", add_special_tokens=False).input_ids)
    print(tokenizer.convert_ids_to_tokens(tokenizer("<|im_start|>assistant\n", add_special_tokens=False).input_ids))
