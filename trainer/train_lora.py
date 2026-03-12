import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.MiniMindModel import MiniMindConfig
from dataset.llm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')

def train_epoch(epoch, loader, iters, lora_params, start_step=0, wandb=None):
    for step, (input_ids, labels, loss_mask) in enumerate(dataloader, start_step + 1):
        input_ids, labels, loss_mask = input_ids.to(args.device), labels.to(args.device), loss_mask.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        with autocast_ctx:
            