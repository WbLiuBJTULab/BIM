import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class BIMLoRALinear(nn.Linear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            config,
            r: int = None,
            lora_alpha: int = None,
            lora_dropout: float = None,
            fan_in_fan_out: bool = False,
            merge_weights: bool = True,
            block_size: int = None,
            mask_temperature: float = None,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)

        self.config = config
        self.r = r if r is not None else config.lora_r
        self.lora_alpha = lora_alpha if lora_alpha is not None else config.lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout if lora_dropout is not None else config.lora_dropout)
        self.fan_in_fan_out = fan_in_fan_out
        self.merge_weights = merge_weights
        self.merged = False
        self.block_size = block_size if block_size is not None else config.block_size
        self.mask_temperature = mask_temperature if mask_temperature is not None else config.mask_temperature

        assert out_features % self.block_size == 0, f"out_features {out_features} must be divisible by block_size {self.block_size}"
        assert in_features % self.block_size == 0, f"in_features {in_features} must be divisible by block_size {self.block_size}"

        self.n_blocks_out = out_features // self.block_size
        self.n_blocks_in = in_features // self.block_size
        self.n_blocks_total = self.n_blocks_out * self.n_blocks_in

        self.lora_A = nn.Parameter(torch.zeros((self.r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, self.r)))
        self.scaling = self.lora_alpha / self.r

        self.register_buffer('block_scores',
                             torch.ones(self.n_blocks_out, self.n_blocks_in) * config.initial_block_score)

        self.register_buffer('hard_mask', None)

        self.register_buffer('grad_mean', torch.zeros(self.n_blocks_out, self.n_blocks_in))
        self.register_buffer('grad_var', torch.zeros(self.n_blocks_out, self.n_blocks_in))
        self.register_buffer('act_var', torch.zeros(self.n_blocks_out, self.n_blocks_in))
        self.register_buffer('weight_norm', torch.zeros(self.n_blocks_out, self.n_blocks_in))

        self.enable_masking = True

        self.reset_parameters()

        self.use_ste = False

        self.enable_temperature_annealing = config.enable_temperature_annealing
        self.temperature_decay_rate = config.temperature_decay_rate
        self.temperature_decay_steps = config.temperature_decay_steps
        self.min_temperature = config.min_temperature

    def reset_parameters(self):
        """Initialize LoRA parameters."""
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def get_block_mask(self, soft: bool = True) -> torch.Tensor:
        if not self.enable_masking:
            return torch.ones_like(self.weight)

        if self.hard_mask is not None:
            mask = self.hard_mask
        else:
            logits = self.block_scores

            mask = torch.sigmoid(logits / self.mask_temperature)


        if mask.shape != self.weight.shape:
            if mask.shape == (self.n_blocks_out, self.n_blocks_in):
                mask = mask.repeat_interleave(self.block_size, dim=0)
                mask = mask.repeat_interleave(self.block_size, dim=1)
            else:
                if mask.numel() == self.weight.numel():
                    mask = mask.view(self.weight.shape)
                else:
                    raise ValueError(f"Mask shape {mask.shape} incompatible with weight shape {self.weight.shape}")

        return mask.to(self.weight.device, dtype=self.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        result = F.linear(x, T(self.weight), bias=self.bias)
        if self.r > 0 and not self.merged:

            delta_w = self.lora_B @ self.lora_A

            if self.enable_masking:

                block_mask = torch.sigmoid(self.block_scores / self.mask_temperature)
                mask_full = block_mask.repeat_interleave(self.block_size, dim=0)
                mask_full = mask_full.repeat_interleave(self.block_size, dim=1)
                mask_full = mask_full.to(delta_w.device, dtype=delta_w.dtype)
                delta_w = delta_w * mask_full

            x_drop = self.lora_dropout(x)

            lora_output = F.linear(x_drop, delta_w, bias=None)

            result = result + lora_output * self.scaling

        return result

    def compute_block_features(self, grad: Optional[torch.Tensor] = None,
                               activation: Optional[torch.Tensor] = None):
        """Compute block-wise statistics for feature collection."""
        with torch.no_grad():
            if grad is not None and grad.shape == self.weight.shape:

                grad = torch.clamp(grad, -10.0, 10.0)

                grad_blocks = grad.view(self.n_blocks_out, self.block_size,
                                        self.n_blocks_in, self.block_size)

                self.grad_mean = grad_blocks.abs().mean(dim=(1, 3))

                block_mean = grad_blocks.mean(dim=(1, 3), keepdim=True)
                self.grad_var = ((grad_blocks - block_mean) ** 2).mean(dim=(1, 3))
                self.grad_var = torch.clamp(self.grad_var, min=1e-8)

            if activation is not None:
                if activation.dim() == 1 and activation.size(0) == self.in_features:
                    activation = torch.clamp(activation, min=1e-8, max=1e3)

                    act_var_blocks = activation.view(self.n_blocks_in, self.block_size)
                    self.act_var = act_var_blocks.mean(dim=1)
                else:
                    print(f"Warning: Unexpected activation shape: {activation.shape}")
                    self.act_var = torch.ones(self.n_blocks_in, device=activation.device) * 1e-6

            weight_blocks = self.weight.view(self.n_blocks_out, self.block_size,
                                             self.n_blocks_in, self.block_size)
            self.weight_norm = weight_blocks.norm(dim=(1, 3))
            self.weight_norm = torch.clamp(self.weight_norm, min=1e-8)


def mark_only_bim_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    """Mark only BIM-LoRA parameters as trainable."""
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
        else:
            p.requires_grad = True

    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, BIMLoRALinear) and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise ValueError(f"Invalid bias option: {bias}")