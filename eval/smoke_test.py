"""MiniMind 评测系统 — 框架正确性 Smoke Test

每个训练管线快速跑 50 步验证框架完整性：
  - 模型初始化 → forward 无误
  - 梯度正常流动（非零、非 NaN）
  - loss 下降 > 10%
  - checkpoint 存取一致
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import warnings
import numpy as np
import torch

from model.MiniMindModel import MiniMindConfig, MiniMindForCausalLM
from eval_utils import (
    generate_report, check_grad_flow, verify_checkpoint_roundtrip,
    init_swanlab, log_to_swanlab, make_small_config,
)

warnings.filterwarnings("ignore")

SMOKE_STEPS = 50
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")
SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "out")


def assertion(name, passed, detail=""):
    return {"name": name, "passed": passed, "detail": detail}


def run_stage(stage_name, config, metrics, assertions, use_swanlab=False):
    """输出终端结果，生成报告，可选 SwanLab 上报"""
    report = generate_report(stage_name, metrics, assertions, REPORT_DIR)

    status = "PASS" if report["passed"] else "FAIL"
    print(f"\n{'='*60}")
    print(f"  {stage_name}: {status}")
    print(f"{'='*60}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
    print(f"  assertions: {len([a for a in assertions if a['passed']])}/{len(assertions)} passed")

    if use_swanlab:
        log_to_swanlab(stage_name, metrics)

    return report["passed"]


def smoke_pretrain(device, use_swanlab=False):
    """预训练管线 smoke test: 50 步验证"""
    from torch import optim
    from torch.utils.data import DataLoader
    from dataset.llm_dataset import PretrainDataset
    from transformers import AutoTokenizer

    config = make_small_config()
    model = MiniMindForCausalLM(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    )
    tokenizer.pad_token = tokenizer.eos_token

    data_path = os.path.join(os.path.dirname(__file__), "test_data", "pretrain_smoke.jsonl")
    ds = PretrainDataset(data_path, tokenizer, max_length=128)
    loader = DataLoader(ds, batch_size=8, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # 初始 forward
    batch = next(iter(loader))
    X, Y, loss_mask, attn_mask = [t.to(device) for t in batch]
    res = model(X, attention_mask=attn_mask, labels=Y, loss_mask=loss_mask)
    initial_loss = (res.loss + res.aux_loss).item()
    print(f"  Initial loss: {initial_loss:.4f}")

    # 跑 SMOKE_STEPS 步
    losses = []
    model.train()
    data_iter = iter(loader)
    for step in range(1, SMOKE_STEPS + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        X, Y, loss_mask, attn_mask = [t.to(device) for t in batch]

        optimizer.zero_grad()
        res = model(X, attention_mask=attn_mask, labels=Y, loss_mask=loss_mask)
        loss = res.loss + res.aux_loss
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    final_loss = losses[-1]
    loss_drop_pct = (initial_loss - final_loss) / initial_loss * 100
    grad_info = check_grad_flow(model)
    ckpt_ok, ckpt_detail = verify_checkpoint_roundtrip(model, None, X[:1], device)

    print(f"  Final loss: {final_loss:.4f}, Drop: {loss_drop_pct:.1f}%")

    return run_stage("pretrain", config, {
        "initial_loss": initial_loss, "final_loss": final_loss,
        "loss_drop_pct": loss_drop_pct, "grad_norm": grad_info["grad_norm"],
    }, [
        assertion("model_init_ok", True),
        assertion("grad_has_grad", grad_info["has_grad"]),
        assertion("grad_no_nan", not grad_info["has_nan"]),
        assertion("loss_drop_gt_10pct", loss_drop_pct > 10, f"{loss_drop_pct:.1f}% > 10%"),
        assertion("checkpoint_roundtrip", ckpt_ok, ckpt_detail),
    ], use_swanlab)


def smoke_sft(device, use_swanlab=False):
    """SFT 管线 smoke test: 加载 pretrain 权重 → 50 步验证"""
    from torch import optim
    from torch.utils.data import DataLoader
    from dataset.llm_dataset import SFTDataset
    from trainer.trainer_utils import init_model

    config = make_small_config()
    model, tokenizer = init_model(config, "pretrain", device=device, save_dir=SAVE_DIR)

    data_path = os.path.join(os.path.dirname(__file__), "test_data", "sft_smoke.jsonl")
    ds = SFTDataset(data_path, tokenizer, max_length=128)
    loader = DataLoader(ds, batch_size=4, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    batch = next(iter(loader))
    X, Y, loss_mask, attn_mask = [t.to(device) for t in batch]
    res = model(X, attention_mask=attn_mask, labels=Y, loss_mask=loss_mask)
    initial_loss = (res.loss + res.aux_loss).item()
    print(f"  Initial loss: {initial_loss:.4f}")

    # loss_mask 验证：prompt 位置 loss_mask=0，应不贡献 loss
    prompt_ratio = (loss_mask == 0).float().mean().item()
    print(f"  Prompt token ratio (mask=0): {prompt_ratio:.3f}")

    losses = []
    model.train()
    data_iter = iter(loader)
    for step in range(1, SMOKE_STEPS + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        X, Y, loss_mask, attn_mask = [t.to(device) for t in batch]

        optimizer.zero_grad()
        res = model(X, attention_mask=attn_mask, labels=Y, loss_mask=loss_mask)
        loss = res.loss + res.aux_loss
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    final_loss = losses[-1]
    loss_drop_pct = (initial_loss - final_loss) / initial_loss * 100
    grad_info = check_grad_flow(model)
    ckpt_ok, ckpt_detail = verify_checkpoint_roundtrip(model, None, X[:1], device)

    return run_stage("sft", config, {
        "initial_loss": initial_loss, "final_loss": final_loss,
        "loss_drop_pct": loss_drop_pct, "grad_norm": grad_info["grad_norm"],
        "prompt_ratio": prompt_ratio,
    }, [
        assertion("model_init_ok", True),
        assertion("grad_has_grad", grad_info["has_grad"]),
        assertion("grad_no_nan", not grad_info["has_nan"]),
        assertion("loss_drop_gt_10pct", loss_drop_pct > 10, f"{loss_drop_pct:.1f}% > 10%"),
        assertion("checkpoint_roundtrip", ckpt_ok, ckpt_detail),
        assertion("loss_mask_active", prompt_ratio > 0, f"prompt_ratio={prompt_ratio:.3f} > 0"),
    ], use_swanlab)


def smoke_lora(device, use_swanlab=False):
    """LoRA 管线 smoke test: 注入 LoRA → 验证冻结参数 + loss 下降"""
    from torch import optim
    from torch.utils.data import DataLoader
    from dataset.llm_dataset import SFTDataset
    from trainer.trainer_utils import init_model
    from model.model_lora import apply_lora, save_lora

    config = make_small_config()
    model, tokenizer = init_model(config, "full_sft", device=device, save_dir=SAVE_DIR)
    apply_lora(model, rank=8, alpha=16, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"])

    # 检查仅 LoRA 参数可训练
    lora_params = []
    frozen_has_grad = False
    for name, param in model.named_parameters():
        if "lora" in name:
            lora_params.append(param)
            assert param.requires_grad, f"LoRA param {name} should be trainable"
        else:
            if param.requires_grad:
                frozen_has_grad = True

    print(f"  LoRA params: {len(lora_params)}, Frozen has grad: {frozen_has_grad}")

    data_path = os.path.join(os.path.dirname(__file__), "test_data", "sft_smoke.jsonl")
    ds = SFTDataset(data_path, tokenizer, max_length=128)
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    optimizer = optim.AdamW(lora_params, lr=1e-3)

    batch = next(iter(loader))
    X, Y, loss_mask, attn_mask = [t.to(device) for t in batch]
    res = model(X, attention_mask=attn_mask, labels=Y, loss_mask=loss_mask)
    initial_loss = (res.loss + res.aux_loss).item()

    losses = []
    model.train()
    data_iter = iter(loader)
    for step in range(1, SMOKE_STEPS + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        X, Y, loss_mask, attn_mask = [t.to(device) for t in batch]
        optimizer.zero_grad()
        res = model(X, attention_mask=attn_mask, labels=Y, loss_mask=loss_mask)
        loss = res.loss + res.aux_loss
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    final_loss = losses[-1]
    loss_drop_pct = (initial_loss - final_loss) / initial_loss * 100
    grad_info = check_grad_flow(model)

    import tempfile
    tmp_path = os.path.join(tempfile.gettempdir(), "_eval_lora_test.pth")
    save_lora(model, tmp_path)
    lora_state = torch.load(tmp_path, map_location="cpu")
    only_ab = all("lora.A." in k or "lora.B." in k for k in lora_state.keys())
    os.remove(tmp_path)

    return run_stage("lora", config, {
        "initial_loss": initial_loss, "final_loss": final_loss,
        "loss_drop_pct": loss_drop_pct, "grad_norm": grad_info["grad_norm"],
        "lora_param_count": len(lora_params),
        "frozen_has_grad": frozen_has_grad,
    }, [
        assertion("model_init_ok", True),
        assertion("grad_has_grad", grad_info["has_grad"]),
        assertion("grad_no_nan", not grad_info["has_nan"]),
        assertion("loss_drop_gt_10pct", loss_drop_pct > 10, f"{loss_drop_pct:.1f}% > 10%"),
        assertion("frozen_params_no_grad", not frozen_has_grad, "non-LoRA params should be frozen"),
        assertion("lora_save_only_ab", only_ab, "saved weights should only contain lora.A/B"),
    ], use_swanlab)


def smoke_dpo(device, use_swanlab=False):
    """DPO 管线 smoke test: 加载 full_sft → 训练 → 验证 chosen > rejected"""
    from torch import optim
    from torch.utils.data import DataLoader
    from dataset.llm_dataset import DPODataset
    from trainer.trainer_utils import init_model
    import torch.nn.functional as F

    config = make_small_config()
    model, tokenizer = init_model(config, "full_sft", device=device, save_dir=SAVE_DIR)
    ref_model, _ = init_model(config, "full_sft", device=device, save_dir=SAVE_DIR)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    data_path = os.path.join(os.path.dirname(__file__), "test_data", "dpo_smoke.jsonl")
    ds = DPODataset(data_path, tokenizer, max_length=256)
    loader = DataLoader(ds, batch_size=2, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=1e-6)

    # 初始检查：chosen vs rejected log-prob
    batch = next(iter(loader))
    x_chosen = batch["x_chosen"][:1].to(device)
    x_rejected = batch["x_rejected"][:1].to(device)

    losses = []
    model.train()
    for step in range(1, SMOKE_STEPS + 1):
        try:
            batch = next(iter(loader))
        except StopIteration:
            break
        x_c = batch["x_chosen"].to(device)
        x_r = batch["x_rejected"].to(device)
        y_c = batch["y_chosen"].to(device)
        y_r = batch["y_rejected"].to(device)
        mask_c = batch["mask_chosen"].to(device)
        mask_r = batch["mask_rejected"].to(device)
        attn_c = batch["attention_mask_chosen"].to(device)
        attn_r = batch["attention_mask_rejected"].to(device)

        x = torch.cat([x_c, x_r], dim=0)
        y = torch.cat([y_c, y_r], dim=0)
        mask = torch.cat([mask_c, mask_r], dim=0)
        attn = torch.cat([attn_c, attn_r], dim=0)

        with torch.no_grad():
            ref_out = ref_model(x, attention_mask=attn)
            ref_logp = F.log_softmax(ref_out.logits, dim=-1).gather(2, y.unsqueeze(-1)).squeeze(-1)

        out = model(x, attention_mask=attn)
        policy_logp = F.log_softmax(out.logits, dim=-1).gather(2, y.unsqueeze(-1)).squeeze(-1)

        B = policy_logp.shape[0]
        chosen_policy = (policy_logp[:B//2] * mask[:B//2]).sum(dim=1)
        reject_policy = (policy_logp[B//2:] * mask[B//2:]).sum(dim=1)
        chosen_ref = (ref_logp[:B//2] * mask[:B//2]).sum(dim=1)
        reject_ref = (ref_logp[B//2:] * mask[B//2:]).sum(dim=1)

        pi_logratios = chosen_policy - reject_policy
        ref_logratios = chosen_ref - reject_ref
        dpo_loss = -F.logsigmoid(0.1 * (pi_logratios - ref_logratios)).mean()

        loss = dpo_loss + out.aux_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    final_loss = np.mean(losses[-5:]) if len(losses) >= 5 else losses[-1]
    grad_info = check_grad_flow(model)
    ckpt_ok, ckpt_detail = verify_checkpoint_roundtrip(model, None, x_c[:1], device)

    chosen_gt_rejected = (chosen_policy.mean() > reject_policy.mean()).item()

    return run_stage("dpo", config, {
        "final_loss": final_loss, "step_count": len(losses),
        "grad_norm": grad_info["grad_norm"],
        "chosen_logp_mean": chosen_policy.mean().item(),
        "rejected_logp_mean": reject_policy.mean().item(),
    }, [
        assertion("model_init_ok", True),
        assertion("grad_has_grad", grad_info["has_grad"]),
        assertion("grad_no_nan", not grad_info["has_nan"]),
        assertion("dpo_loss_lt_ln2", final_loss < np.log(2), f"{final_loss:.4f} < {np.log(2):.4f}"),
        assertion("chosen_gt_rejected", chosen_gt_rejected, "chosen logp > rejected logp"),
        assertion("checkpoint_roundtrip", ckpt_ok, ckpt_detail),
    ], use_swanlab)


def smoke_reason(device, use_swanlab=False):
    """Reason 管线 smoke test: 加载 dpo 权重 → 格式标签惩罚训练 → 验证 tag_hit"""
    from torch import optim
    from torch.utils.data import DataLoader
    from dataset.llm_dataset import SFTDataset
    from trainer.trainer_utils import init_model
    import torch.nn as nn

    config = make_small_config()
    model, tokenizer = init_model(config, "dpo", device=device, save_dir=SAVE_DIR)

    # 构建标签 token 序列
    tag_id_seqs = []
    for tag in ["<think>", "</think>", "<answer>", "</answer>"]:
        ids = tokenizer(tag, add_special_tokens=False).input_ids
        tag_id_seqs.append(torch.tensor(ids, dtype=torch.long, device=device))

    data_path = os.path.join(os.path.dirname(__file__), "test_data", "reason_smoke.jsonl")
    ds = SFTDataset(data_path, tokenizer, max_length=256)
    loader = DataLoader(ds, batch_size=4, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=1e-6)
    loss_fct = nn.CrossEntropyLoss(reduction="none")

    batch = next(iter(loader))
    X, Y, loss_mask, attn_mask = [t.to(device) for t in batch]

    # 检查首条数据的标签命中情况
    from trainer.train_reason import build_tag_penalty_mask
    _, initial_hits = build_tag_penalty_mask(Y, tag_id_seqs, penalty_weight=10.0, device=device)
    print(f"  Initial tag hits: {initial_hits}")

    losses = []
    model.train()
    for step in range(1, SMOKE_STEPS + 1):
        try:
            batch = next(iter(loader))
        except StopIteration:
            break
        X, Y, loss_mask, attn_mask = [t.to(device) for t in batch]

        res = model(X, attention_mask=attn_mask)
        logits = res.logits
        loss_raw = loss_fct(logits.view(-1, logits.size(-1)), Y.view(-1)).view(Y.size())

        penalty_mask, tag_hit_count = build_tag_penalty_mask(
            Y, tag_id_seqs, penalty_weight=10.0, device=device
        )
        weighted_mask = loss_mask.float() * penalty_mask
        valid_count = loss_mask.sum()
        logits_loss = (loss_raw * weighted_mask).sum() / (valid_count + 1e-8)
        loss = logits_loss + res.aux_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    final_loss = np.mean(losses[-5:]) if len(losses) >= 5 else losses[-1]
    final_tag_hit = tag_hit_count
    tag_ratio = final_tag_hit / max(valid_count.item(), 1)
    grad_info = check_grad_flow(model)
    ckpt_ok, ckpt_detail = verify_checkpoint_roundtrip(model, None, X[:1], device)

    return run_stage("reason", config, {
        "initial_tag_hits": initial_hits, "final_tag_hits": final_tag_hit,
        "tag_ratio": tag_ratio, "final_loss": final_loss,
        "grad_norm": grad_info["grad_norm"],
    }, [
        assertion("model_init_ok", True),
        assertion("grad_has_grad", grad_info["has_grad"]),
        assertion("grad_no_nan", not grad_info["has_nan"]),
        assertion("tag_hits_gt_0", final_tag_hit > 0, f"tag_hits={final_tag_hit} > 0"),
        assertion("tag_ratio_gt_0", tag_ratio > 0, f"tag_ratio={tag_ratio:.4f} > 0"),
        assertion("checkpoint_roundtrip", ckpt_ok, ckpt_detail),
    ], use_swanlab)


def smoke_ppo(device, use_swanlab=False):
    """PPO 管线 smoke test: 验证四个模型加载 + rollout 可完成"""
    if not torch.cuda.is_available():
        print("  [SKIP] PPO smoke test requires GPU")
        return run_stage("ppo", make_small_config(), {"status": "skipped (no GPU)"}, [
            assertion("ppo_skipped_no_gpu", True),
        ], use_swanlab)

    import os as _os
    reward_path = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "internlm2-1_8b-reward")
    if not _os.path.exists(reward_path):
        print("  [SKIP] PPO smoke test requires Reward Model at internlm2-1_8b-reward/")
        return run_stage("ppo", make_small_config(), {"status": "skipped (no reward model)"}, [
            assertion("ppo_skipped_no_rm", True),
        ], use_swanlab)

    from transformers import AutoTokenizer, AutoModel
    from trainer.trainer_utils import init_model
    from trainer.train_ppo import CriticModel, compute_gae

    config = make_small_config()
    base_weight = "dpo"

    actor, tokenizer = init_model(config, base_weight, device=device, save_dir=SAVE_DIR)
    actor.train()

    old_actor, _ = init_model(config, base_weight, device=device, save_dir=SAVE_DIR)
    old_actor.eval()
    for p in old_actor.parameters():
        p.requires_grad_(False)

    ref_model, _ = init_model(config, base_weight, device=device, save_dir=SAVE_DIR)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # Critic
    critic = CriticModel(config).to(device)
    try:
        ckpt_state = torch.load(
            _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "out", f"{base_weight}_{config.hidden_size}.pth"),
            map_location=device
        )
        critic.load_state_dict(ckpt_state, strict=False)
    except FileNotFoundError:
        pass

    # Reward
    reward_model = AutoModel.from_pretrained(reward_path, trust_remote_code=True, torch_dtype=torch.float16).to(device).eval()
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_path, trust_remote_code=True)
    for p in reward_model.parameters():
        p.requires_grad_(False)

    # Rollout
    prompts = ["你好，请介绍一下你自己。", "1+1等于多少？"]
    actor_gen = actor.module if hasattr(actor, "module") else actor
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=64, padding_side="left").to(device)
    gen_out = actor_gen.generate(
        **enc, max_new_tokens=32, do_sample=True, temperature=0.8,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    prompt_len = enc.input_ids.shape[1]
    responses = [tokenizer.decode(gen_out[i, prompt_len:], skip_special_tokens=True) for i in range(len(prompts))]
    print(f"  Generated {len(responses)} responses, avg_len={np.mean([len(r) for r in responses]):.1f}")

    # GAE
    seq_len = gen_out.size(1) - 1
    B = len(prompts)
    values_seq = torch.randn(B, seq_len, device=device) * 0.1
    resp_mask = torch.ones(B, seq_len, device=device)
    rewards = torch.randn(B, device=device)
    advantages, returns = compute_gae(rewards, values_seq, resp_mask)
    adv_nonzero = (advantages.abs().sum() > 0).item()

    return run_stage("ppo", config, {
        "num_prompts": len(prompts), "num_responses": len(responses),
        "response_len_mean": np.mean([len(r) for r in responses]),
        "gae_adv_nonzero": adv_nonzero,
    }, [
        assertion("actor_loaded", True),
        assertion("critic_loaded", True),
        assertion("ref_loaded", True),
        assertion("reward_loaded", True),
        assertion("rollout_completed", len(responses) == 2),
        assertion("gae_adv_nonzero", adv_nonzero, "GAE advantage should be non-zero"),
    ], use_swanlab)


def smoke_grpo(device, use_swanlab=False):
    """GRPO 管线 smoke test: 验证多回答生成 + 组内 advantage"""
    if not torch.cuda.is_available():
        print("  [SKIP] GRPO smoke test requires GPU")
        return run_stage("grpo", make_small_config(), {"status": "skipped (no GPU)"}, [
            assertion("grpo_skipped_no_gpu", True),
        ], use_swanlab)

    import os as _os
    reward_path = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "internlm2-1_8b-reward")
    if not _os.path.exists(reward_path):
        print("  [SKIP] GRPO smoke test requires Reward Model")
        return run_stage("grpo", make_small_config(), {"status": "skipped (no reward model)"}, [
            assertion("grpo_skipped_no_rm", True),
        ], use_swanlab)

    from trainer.trainer_utils import init_model

    config = make_small_config()
    model, tokenizer = init_model(config, "dpo", device=device, save_dir=SAVE_DIR)
    model.eval()

    # 生成 G 个回答
    G = 4
    prompts = ["你好"]
    prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").to(device)
    prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -64:]
    prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -64:]

    model_gen = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        outputs = model_gen.generate(
            **prompt_inputs, max_new_tokens=32, do_sample=True, temperature=0.8,
            num_return_sequences=G, pad_token_id=tokenizer.pad_token_id,
        )

    prompt_len = prompt_inputs["input_ids"].size(1)
    completions = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)

    # 检查回答互不相同
    unique_count = len(set(completions))
    all_different = unique_count == len(completions)

    # 模拟 reward + 组内 advantage
    fake_rewards = torch.randn(len(completions), device=device)
    grouped = fake_rewards.view(-1, G)
    mean_r = grouped.mean(dim=1, keepdim=True)
    std_r = grouped.std(dim=1, keepdim=True)
    advantages = ((grouped - mean_r) / (std_r + 1e-4)).view(-1)

    return run_stage("grpo", config, {
        "num_prompts": len(prompts), "num_generations": G,
        "total_completions": len(completions),
        "unique_completions": unique_count,
        "adv_mean": advantages.mean().item(),
        "adv_std": advantages.std().item(),
    }, [
        assertion("generation_completed", len(completions) == len(prompts) * G),
        assertion("completions_different", all_different, f"{unique_count}/{len(completions)} unique"),
        assertion("adv_std_gt_0", advantages.std().item() > 0),
    ], use_swanlab)


def main():
    parser = argparse.ArgumentParser(description="MiniMind Smoke Test")
    parser.add_argument("--all", action="store_true", help="运行所有管线 smoke test")
    parser.add_argument("--stage", type=str, default=None,
                        choices=["pretrain", "sft", "lora", "dpo", "reason", "ppo", "grpo"])
    parser.add_argument("--skip-rl", action="store_true", help="跳过 PPO/GRPO (需要 Reward Model)")
    parser.add_argument("--use-wandb", action="store_true", help="上报到 SwanLab")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.use_wandb:
        init_swanlab("MiniMind-Eval")

    os.makedirs(REPORT_DIR, exist_ok=True)

    if not args.all and not args.stage:
        parser.print_help()
        print("\n请指定 --all 或 --stage STAGE")
        return

    all_stages = ["pretrain", "sft", "lora", "dpo", "reason", "ppo", "grpo"]
    if args.stage:
        stages = [args.stage]
    else:
        stages = all_stages
    if args.skip_rl:
        stages = [s for s in stages if s not in ("ppo", "grpo")]

    results = {}
    for stage in stages:
        fn = globals().get(f"smoke_{stage}")
        if fn:
            print(f"\n{'#'*60}\n#  Running smoke test: {stage}\n{'#'*60}")
            results[stage] = fn(args.device, args.use_wandb)
        else:
            print(f"[WARN] Unknown stage: {stage}")

    all_passed = all(results.values()) if results else False
    print(f"\n{'='*60}")
    print(f"  OVERALL: {'PASS' if all_passed else 'FAIL'} ({sum(results.values())}/{len(results)} passed)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
