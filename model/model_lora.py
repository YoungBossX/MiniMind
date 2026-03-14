import torch
import math
from torch import optim, nn
from torch.nn.init import kaiming_uniform_

class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x)) * self.scaling
    
def apply_lora(model, rank, alpha, target_modules):
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        module_name = name.split(".")[-1]
        if module_name not in target_modules:
            continue

        if hasattr(module, "lora"):
            continue

        lora = LoRA(
            in_features=module.in_features,
            out_features=module.out_features,
            rank=rank,
            alpha=alpha,
        ).to(module.weight.device, dtype=module.weight.dtype)

        module.lora = lora
        original_forward = module.forward

        def forward_with_lora(x, layer1=original_forward, layer2=lora):
            return layer1(x) + layer2(x)

        module.forward = forward_with_lora
    
    return model

def load_lora(model, path):
    device = next(model.parameters()).device
    state_dict = torch.load(path, map_location=device)
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)

def save_lora(model, path):
    # torch.compile(model) 会把模型封装一层，有 _orig_mod：说明用了 torch.compile，取出原始模型，没有 _orig_mod：没有用 torch.compile，直接用 model 本身
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            # 去掉前缀，统一格式：
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {f'{clean_name}.lora.{k}': v for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)