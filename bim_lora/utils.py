import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
import json
import os
from transformers import PreTrainedModel
from bim_lora.layers import BIMLoRALinear
from bim_lora.implicit_mlp import ImplicitMLP
import re


def get_bim_lora_model(
        model: PreTrainedModel,
        config,
        r: int = None,
        lora_alpha: int = None,
        lora_dropout: float = None,
        target_modules: Optional[List[str]] = None,
        block_size: int = None,
        mask_temperature: float = None
) -> nn.Module:

    r = r if r is not None else config.lora_r
    lora_alpha = lora_alpha if lora_alpha is not None else config.lora_alpha
    lora_dropout = lora_dropout if lora_dropout is not None else config.lora_dropout
    block_size = block_size if block_size is not None else config.block_size
    mask_temperature = mask_temperature if mask_temperature is not None else config.mask_temperature

    if target_modules is None:
        target_modules = config.target_modules

    target_patterns = [re.compile(f".*{module}$") for module in target_modules]

    def replace_with_bim_lora(model, prefix=""):

        for name, module in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(module, nn.Linear):

                for pattern in target_patterns:
                    if pattern.match(full_name):

                        in_features = module.in_features
                        out_features = module.out_features

                        if (out_features % block_size == 0 and
                                in_features % block_size == 0):

                            new_module = BIMLoRALinear(
                                in_features=in_features,
                                out_features=out_features,
                                config=config,
                                r=r,
                                lora_alpha=lora_alpha,
                                lora_dropout=lora_dropout,
                                bias=module.bias is not None,
                                block_size=block_size,
                                mask_temperature=mask_temperature
                            )

                            new_module.weight.data = module.weight.data.clone()
                            if module.bias is not None:
                                new_module.bias.data = module.bias.data.clone()

                            setattr(model, name, new_module)
                            print(f"Replaced {full_name}: {in_features}x{out_features}")
                        else:
                            print(f"Skipped {full_name}: dimensions not divisible by {block_size}")
                        break
            else:

                replace_with_bim_lora(module, full_name)

    replace_with_bim_lora(model)

    total_params = sum(p.numel() for p in model.parameters())
    lora_params = sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n)

    print(f"\nModel conversion complete:")
    print(f"Total parameters: {total_params:,}")
    print(f"LoRA parameters: {lora_params:,} ({100 * lora_params / total_params:.2f}%)")

    return model


def save_bim_lora_model(
        model: nn.Module,
        mlp: ImplicitMLP,
        save_path: str,
        metadata: Optional[Dict] = None
):
    """Save BIM-LoRA model checkpoint."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    lora_state_dict = {}
    block_scores = {}
    hard_masks = {}
    variational_params = {}

    for name, module in model.named_modules():
        if isinstance(module, BIMLoRALinear):

            lora_state_dict[f"{name}.lora_A"] = module.lora_A
            lora_state_dict[f"{name}.lora_B"] = module.lora_B

            block_scores[name] = module.block_scores

            if module.hard_mask is not None:
                hard_masks[name] = module.hard_mask

            if hasattr(module, 'q_logits'):
                variational_params[name] = {
                    'q_logits': module.q_logits,
                    'prior_alpha': module.prior_alpha,
                    'prior_beta': module.prior_beta,
                    'variational_pruning': module.variational_pruning
                }

    checkpoint = {
        'lora_state_dict': lora_state_dict,
        'block_scores': block_scores,
        'hard_masks': hard_masks,
        'mlp_state_dict': mlp.state_dict(),
        'metadata': metadata or {},
        'variational_params': variational_params
    }

    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")

    metadata_path = save_path.replace('.pt', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json_metadata = {
            'n_lora_modules': len(lora_state_dict) // 2,
            'n_parameters': sum(p.numel() for p in lora_state_dict.values()),
            **checkpoint['metadata']
        }
        json.dump(json_metadata, f, indent=2)


def load_bim_lora_model(
        model: nn.Module,
        mlp: ImplicitMLP,
        load_path: str,
        strict: bool = True
) -> Dict:
    """Load BIM-LoRA model checkpoint."""
    checkpoint = torch.load(load_path, map_location='cpu', weights_only=False)

    missing_keys = []
    for name, param in checkpoint['lora_state_dict'].items():
        try:
            model.get_parameter(name).data = param.data
        except AttributeError:
            if strict:
                raise KeyError(f"Parameter {name} not found in model")
            missing_keys.append(name)

    for name, module in model.named_modules():
        if isinstance(module, BIMLoRALinear):
            if name in checkpoint['block_scores']:
                module.block_scores.data = checkpoint['block_scores'][name].data

            if name in checkpoint.get('hard_masks', {}):
                module.hard_mask = checkpoint['hard_masks'][name]

            if 'variational_params' in checkpoint and name in checkpoint['variational_params']:
                var_params = checkpoint['variational_params'][name]
                module.q_logits.data = var_params['q_logits'].data
                module.prior_alpha.data = var_params['prior_alpha'].data
                module.prior_beta.data = var_params['prior_beta'].data
                module.variational_pruning = var_params['variational_pruning']

    mlp.load_state_dict(checkpoint['mlp_state_dict'])

    if missing_keys:
        print(f"Warning: {len(missing_keys)} keys not found in model")

    return checkpoint.get('metadata', {})
