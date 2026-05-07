"""数据与结果读写工具

支持格式：JSONL（读取+写入）、TXT（读取）、JSON（读取+写入）
"""

import json
import os


def read_jsonl(path: str) -> list:
    """读取 JSONL 文件，返回 list of dict"""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def write_jsonl(path: str, samples: list):
    """写入 JSONL 文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def read_txt(path: str) -> list:
    """读取 TXT 文件，每行一条文本，跳过空行"""
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def read_json(path: str) -> dict:
    """读取 JSON 文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: dict):
    """写入 JSON 文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def make_serializable(obj):
    """递归转换对象为 JSON 可序列化类型"""
    import torch
    import numpy as np

    if isinstance(obj, (torch.Tensor,)):
        obj = obj.detach().cpu()
        if obj.numel() == 1:
            return obj.item()
        return obj.tolist()
    if isinstance(obj, np.ndarray):
        if obj.size == 1:
            return obj.item()
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    return obj
