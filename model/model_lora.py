import torch
import math
from torch import optim, nn
from torch.nn.init import kaiming_uniform_

class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank 
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        # 矩阵 A 使用 Kaiming Uniform 初始化
        kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        # 矩阵 B 全 0 初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))
    
def apply_lora(model, rank=8):
    device = next(model.parameters()).device
    for name, module in model.named_modules():
        if (isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]):
            lora = LoRA(module.weight.shape[1], module.weight.shape[0], rank=rank).to(device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora

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