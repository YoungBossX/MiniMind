from torch.utils.data import Dataset
import torch
import os
import json
import random
from datasets import load_dataset

# 禁用 HuggingFace tokenizer 的多进程并行，避免在 DataLoader 多进程环境中产生死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

def post_processing_chat(prompt_content, empty_think_ratio=0.05):
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
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
        self.tokenizer = tokenizer      # 分词器，用于将文本转为token ID
        self.max_length = max_length    # 每条样本的最大token长度
        self.samples = self.load_data(data_path)  # 加载数据

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
        encoding = self.tokenizer(
            str(sample['text']),                 # 转为字符串（确保数据类型一致）
            max_length=self.max_length,          # 限制最大长度
            padding='max_length',                # 不足部分补pad
            truncation=True,                     # 超出部分截断
            return_tensors='pt'                  # 返回PyTorch tensor形式（包含batch维度）
        )

        # 获取 input_ids 张量，并去除 batch 维度（变成一维）
        input_ids = encoding.input_ids.squeeze()  # shape: [max_length]
        
        # 计算 loss_mask：pad 的位置不参与 loss
        loss_mask = (input_ids != self.tokenizer.pad_token_id)  # shape: [max_length]，bool类型

        # 语言模型是自回归的，使用前一个 token 预测下一个
        X = torch.tensor(input_ids[:-1], dtype=torch.long)         # 输入：[0, ..., n-2]
        Y = torch.tensor(input_ids[1:], dtype=torch.long)          # 目标：[1, ..., n-1]
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # loss_mask 对齐目标 Y

        return X, Y, loss_mask
    
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
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids # [1, 1078, 538, 501]， [1]是<|im_start|>这个特殊token的id，[1078, 538, 501]是assistant的分词id
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids # [2]

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
            # 偶数轮为用户，奇数轮为助手
            role = 'user' if i % 2 == 0 else 'assistant'
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
            add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids):
        """
        构建损失掩码，只有 assistant 的回答部分才参与 loss 计算。
        找出每一段 assistant 的响应，在其 <|im_start|>assistant 和 <|im_end|> 之间设置 loss_mask 为 1。
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # 找 assistant 开头标志
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id) # 答案起点
                end = start
                while end < len(input_ids):
                    # 查找 assistant 的回答终止符 <|im_end|>
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 为 assistant 回答部分（从 start + 1 到 end 之间）设置 loss mask
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                # 跳过到下一个 segment
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        sample = self.samples[index]

        # 构建 ChatML 格式 prompt（字符串）
        prompt = self._create_chat_prompt(sample['conversations'])

        # 分词并截断，确保长度 <= max_length
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]

        # 右侧填充 pad_token 直到 max_length 长度
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 生成动态 loss mask，仅对 assistant 响应位置计算 loss
        loss_mask = self._generate_loss_mask(input_ids)

        # 构建训练样本：
        # 模型输入为前 n-1 个 token，预测目标为第 2 到第 n 个 token
        X = torch.tensor(input_ids[:-1], dtype=torch.long)         # 输入序列
        Y = torch.tensor(input_ids[1:], dtype=torch.long)          # 目标标签（shifted）
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐 Y 的位置（从第一个预测 token 开始）

        return X, Y, loss_mask

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

    def __getitem__(self, index):
        item = self.data[index]

        chosen = item['chosen']
        rejected = item['rejected']

        # 拼接成字符串（不 tokenize，只生成 prompt 文本）
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )

        # 编码为 input_ids（截断 + 填充）
        chosen_encoding = self.tokenizer(
            chosen_prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length'
        )

        # 转换为 token ID 列表，长度为 max_length
        chosen_input_ids = chosen_encoding['input_ids']           # shape: (max_length,)
        rejected_input_ids = rejected_encoding['input_ids']       # shape: (max_length,)

        # 构造 loss mask：仅在 assistant 段落（<|im_start|>assistant ... <|im_end|>）中的 token 参与损失
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)     # shape: (max_length,)
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids) # shape: (max_length,)


        # 构造训练数据：左移一位预测（即 y 是 x 的下一位）
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)      # shape: (max_length - 1,)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)       # shape: (max_length - 1,)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)    # shape: (max_length - 1,)

        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)  # shape: (max_length - 1,)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)   # shape: (max_length - 1,)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)# shape: (max_length - 1,)

        return {
            'x_chosen': x_chosen,           # shape: (max_length - 1,)
            'y_chosen': y_chosen,           # shape: (max_length - 1,)
            'mask_chosen': mask_chosen,     # shape: (max_length - 1,)

            'x_rejected': x_rejected,       # shape: (max_length - 1,)
            'y_rejected': y_rejected,       # shape: (max_length - 1,)
            'mask_rejected': mask_rejected  # shape: (max_length - 1,)
        }

    def _generate_loss_mask(self, input_ids):
        """
        根据 <|im_start|>assistant 和 <|im_end|> 的位置标记哪些 token 应该参与损失计算。
        返回一个和 input_ids 等长的 0/1 mask。
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # 匹配一个 assistant 段落开头
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    # 查找 assistant 的回答终止符 <|im_end|>
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 在 <|im_start|>assistant 和 <|im_end|> 之间部分启用 loss
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask
    
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
    def __init__(self, file_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
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

    def create_chat_prompt(self, conversations):
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
        prompt = post_processing_chat(prompt)
        return prompt, answer
    
    def __getitem__(self, index):
        sample = self.data[index]
        prompt, answer = self.create_chat_prompt(sample["conversations"])

        return {"prompt": prompt, "answer": answer}