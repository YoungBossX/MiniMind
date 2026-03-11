import os
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler

# 检查是否是主进程
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

# 日志
def Logger(content):
    if is_main_process():
        print(content)

# 动态学习率计算
def get_lr(current_step, total_steps, lr):
    return (
        lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))
    )

# 初始化分布式
def init_distributed_mode():
    # 非DDP模式
    if int(os.environ.get("RANK", -1)) == -1:
        return 0

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


# 设置种子
def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置检查点
def lm_checkpoint(
    lm_config,
    weight=None,
    model=None,
    optimizer=None,
    epoch=0,
    step=0,
    wandb=None,
    save_dir=None,
    **kwargs,
):
    # 确保保存目录存在，不存在则创建
    os.makedirs(save_dir, exist_ok=True)
    # 构建文件名后缀：MoE 模型加 "_moe"，普通模型不加
    moe_path = "_moe" if hasattr(lm_config, "use_moe") and lm_config.use_moe else ""
    # ckp_path：只保存模型权重（半精度），用于推理加载
    ckp_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth"
    # resume_path：保存完整训练状态（模型+优化器+进度），用于断点续训
    resume_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth"

    # ================================================================
    # 保存模式（model 不为 None）
    # ================================================================
    if model is not None:
        from torch.nn.parallel import DistributedDataParallel
        
        # DDP 模型有额外封装，真正的模型在 .module 里
        # 非 DDP 模型直接调用 state_dict()
        if isinstance(model, DistributedDataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        # ── 保存推理用的模型权重（半精度）────────────────────────────
        # 先写到 .tmp 临时文件，写完再用 os.replace 原子替换
        # 目的：防止写到一半程序崩溃导致文件损坏
        # os.replace 是原子操作，不会出现"替换到一半"的中间状态
        ckp_tmp = ckp_path + ".tmp"
        torch.save({k: v.half() for k, v in state_dict.items()}, ckp_tmp)
        # half()：float32 → float16，文件大小减半，推理时精度损失可接受
        os.replace(ckp_tmp, ckp_path)

        # ── 获取 wandb 实验 id（用于续训时恢复到同一个实验）──────────
        wandb_id = None
        if wandb:
            # SwanLab 的 API：通过 get_run() 获取当前实验对象
            if hasattr(wandb, "get_run"):
                run = wandb.get_run()
                wandb_id = getattr(run, "id", None) if run else None
            else:
                # WandB 的 API：直接从 wandb 对象取 id
                wandb_id = getattr(wandb, "id", None)

        # ── 构建完整训练状态字典 ─────────────────────────────────────
        resume_data = {
            # 模型权重（float32，用于续训）
            "model": state_dict,
            # 优化器状态（动量m、方差v等）
            "optimizer": optimizer.state_dict(),
            # 当前 epoch 编号
            "epoch": epoch,
            # 当前 step 编号
            "step": step,
            # 保存当前 GPU 数量，续训时 GPU 数量变化时用于换算 step
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            # 实验 id，续训时恢复实验曲线
            "wandb_id": wandb_id,
        }

        # ── 处理额外的可选状态（如 scaler）──────────────────────────
        # 调用时传入 scaler=scaler，kwargs = {"scaler": scaler}
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, "state_dict"):
                    # 有 state_dict 方法的对象（如 GradScaler、模型）
                    if isinstance(value, DistributedDataParallel):
                        resume_data[key] = value.module.state_dict()
                    else:
                        # scaler → resume_data["scaler"] = scaler.state_dict()
                        resume_data[key] = value.state_dict()
                else:
                    # 普通值（如整数、字符串）直接存
                    resume_data[key] = value
        # ── 保存完整训练状态（同样用 tmp + replace 原子写入）─────────
        resume_tmp = resume_path + ".tmp"
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)

    # ================================================================
    # 加载模式（model 为 None，只传了 lm_config 和 weight）
    # ================================================================
    else:
        # resume_path 存在 → 有上次的训练状态，可以续训
        if os.path.exists(resume_path):
            # map_location="cpu"：先加载到 CPU，避免 GPU 显存直接被占用
            # 后续训练循环会手动把参数移到对应设备
            ckp_data = torch.load(resume_path, map_location="cpu")
            # ── 处理 GPU 数量变化的情况 ──────────────────────────────
            # 上次用 4 张卡训练，这次只有 2 张卡
            # 每张卡处理的数据量不同，step 编号需要换算
            saved_ws = ckp_data.get("world_size", 1) # 上次训练的 GPU 数量
            current_ws = dist.get_world_size() if dist.is_initialized() else 1

            if saved_ws != current_ws:
                # 换算公式：新 step = 旧 step × 旧GPU数 ÷ 新GPU数
                # 例：旧 step=100，4卡→2卡
                # 新 step = 100 × 4 // 2 = 200
                # 含义：4卡跑了100步 = 2卡跑了200步（处理了同样多的数据）
                ckp_data["step"] = ckp_data["step"] * saved_ws // current_ws
                Logger(
                    f"GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data['step']}"
                )
            # 返回完整训练状态，调用方用它恢复 model/optimizer/scaler/epoch/step
            return ckp_data
        # resume_path 不存在 → 没有checkpoint，返回 None，从头开始训练
        return None

# 初始化模型
def init_model(
    lm_config,
    from_weight=None,
    tokenizer_path=None,
    save_dir="../out",
    device="cuda",
):
    from transformers import AutoTokenizer
    from model.MiniMindModel import MiniMindForCausalLM

    # 如果没有指定 tokenizer_path，使用项目根目录下的 model 文件夹
    if tokenizer_path is None:
        # 获取当前文件所在目录的父目录（项目根目录）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        tokenizer_path = os.path.join(project_root, "model")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    model = MiniMindForCausalLM(lm_config)

    if from_weight != "none":
        moe_suffix = (
            "_moe" if hasattr(lm_config, "use_moe") and lm_config.use_moe else ""
        )
        weight_path = (
            f"{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
        )

        weights = torch.load(weight_path, map_location=device)

        model.load_state_dict(weights, strict=False)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger(f"所加载Model可训练参数：{total_params / 1e6:.3f} 百万")

    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        # sampler：底层采样器，决定数据的遍历顺序
        #   - 分布式训练时是 DistributedSampler（确保不同GPU拿不同数据）
        #   - 单卡训练时是 indices（torch.randperm 生成的随机索引列表）
        self.sampler = sampler
        # 每个 batch 包含多少条样本
        self.batch_size = batch_size
        # 要跳过的 batch 数量，对应上次已训练的 step 数
        # skip_batches=0 表示从头开始，不跳过任何数据
        self.skip_batches = skip_batches

    def __iter__(self):
        # 当前正在积累的 batch，存放样本索引
        batch = []
        # 已经跳过的 batch 数量
        skipped = 0

        for idx in self.sampler:
            # 把当前样本索引加入 batch
            batch.append(idx)
            # batch 积累满了
            if len(batch) == self.batch_size:
                # 还没跳够，丢弃这个 batch
                if skipped < self.skip_batches:
                    # 跳过计数 +1
                    skipped += 1
                    # 清空，重新积累下一个 batch
                    batch = []
                    # 继续遍历下一个 idx
                    continue

                # 已跳过足够的 batch，正常产出
                yield batch
                # 清空，准备积累下一个 batch
                batch = []
                
        # 处理最后一个不完整的 batch（样本数 < batch_size）
        # 必须满足 skipped >= skip_batches，即跳过阶段已经结束
        # 如果跳过阶段还没结束，说明整个数据集都被跳过了，不产出任何数据
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size

        return max(0, total_batches - self.skip_batches)