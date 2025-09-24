import torch
import torch.nn as nn
import numpy as np
from typing import Dict


from bim_lora import BIMLoRALinear


class MaskAllocator:

    def __init__(self, model: nn.Module, config):
        self.model = model
        self.config = config

        self.bim_modules = {}
        for name, module in model.named_modules():
            if isinstance(module, BIMLoRALinear):
                self.bim_modules[name] = module

        print(f"MaskAllocator managing {len(self.bim_modules)} BIM-LoRA layers")

    def get_sparsity_stats(self) -> Dict[str, float]:

        stats = {
            'modules': len(self.bim_modules),
        }

        all_scores = []
        for module in self.bim_modules.values():
            probs = torch.sigmoid(module.block_scores)
            all_scores.extend(probs.detach().cpu().numpy().flatten())

        if all_scores:
            all_scores = np.array(all_scores)
            stats['mean_score'] = float(all_scores.mean())
            stats['std_score'] = float(all_scores.std())
            stats['min_score'] = float(all_scores.min())
            stats['max_score'] = float(all_scores.max())
            stats['active_ratio'] = float((all_scores > 0.5).mean())

        return stats
