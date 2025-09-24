import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict
import math
from bim_lora import BIMLoRALinear


class FeatureCollector:
    """
    Collects block-wise statistics during initial training steps.
    Implements Step-0 of BIM-LoRA: feature collection.
    """

    def __init__(self, model: nn.Module, config):
        self.model = model
        self.config = config
        self.block_size = config.block_size
        self.features = defaultdict(list)
        self.handles = []
        self.step_count = 0
        self.collection_steps = config.collection_steps
        self.device = next(model.parameters()).device

        try:
            self.total_layers = model.config.num_hidden_layers
        except (AttributeError, KeyError):
            self.total_layers = config.default_num_layers

        self.global_max = {
            'grad_mean': 0.0,
            'grad_var': 0.0,
            'act_var': 0.0,
            'weight_norm': 0.0
        }

        self.module_types = {

            'query': 0.0, 'q_proj': 0.0, 'self.query': 0.0,
            'key': 0.167, 'k_proj': 0.167, 'self.key': 0.167,
            'value': 0.333, 'v_proj': 0.333, 'self.value': 0.333,
            'output': 0.5, 'o_proj': 0.5, 'out_proj': 0.5, 'output.dense': 0.5,

            'intermediate': 0.667, 'fc1': 0.667, 'intermediate.dense': 0.667,
            'output.dense': 0.833, 'fc2': 0.833,

            'classifier': 1.0, 'classifier.dense': 1.0, 'pooler': 1.0
        }

    def register_hooks(self):
        print("Registering feature collection hooks...")

        for name, module in self.model.named_modules():
            if isinstance(module, BIMLoRALinear):

                module._accumulated_grad_sum = None
                module._accumulated_grad_sq_sum = None
                module._accumulated_act_sum = None
                module._accumulated_act_sq_sum = None
                module._accumulated_count = 0

                def forward_hook(module, input, output, name=name):
                    if self.step_count < self.collection_steps:
                        with torch.no_grad():
                            act = input[0].detach()
                            if act.dim() >= 2:

                                flat = act.flatten(start_dim=0, end_dim=act.dim() - 2)

                                if module._accumulated_act_sum is None:
                                    module._accumulated_act_sum = flat.sum(dim=0)
                                    module._accumulated_act_sq_sum = (flat ** 2).sum(dim=0)
                                    module._accumulated_count = flat.size(0)
                                else:
                                    module._accumulated_act_sum += flat.sum(dim=0)
                                    module._accumulated_act_sq_sum += (flat ** 2).sum(dim=0)
                                    module._accumulated_count += flat.size(0)

                def backward_hook(module, grad_input, grad_output, name=name):
                    if self.step_count < self.collection_steps:
                        with torch.no_grad():
                            grad = module.weight.grad
                            if grad is None:
                                return

                            if module._accumulated_grad_sum is None:
                                module._accumulated_grad_sum = grad.abs()
                                module._accumulated_grad_sq_sum = grad ** 2
                            else:
                                module._accumulated_grad_sum += grad.abs()
                                module._accumulated_grad_sq_sum += grad ** 2

                            if self.step_count == self.collection_steps - 1:

                                n_steps = self.collection_steps
                                grad_mean = module._accumulated_grad_sum / n_steps
                                grad_sq_mean = module._accumulated_grad_sq_sum / n_steps
                                grad_var = grad_sq_mean - grad_mean ** 2
                                grad_var = torch.clamp(grad_var, min=1e-8)

                                if module._accumulated_act_sum is not None:
                                    act_mean = module._accumulated_act_sum / module._accumulated_count
                                    act_sq_mean = module._accumulated_act_sq_sum / module._accumulated_count
                                    act_var = act_sq_mean - act_mean ** 2
                                    act_var = torch.clamp(act_var, min=1e-8)
                                else:
                                    act_var = torch.ones(self.in_features, device=grad.device) * 1e-6

                                module.grad_mean = grad_mean.view(module.n_blocks_out, module.block_size,
                                                                  module.n_blocks_in, module.block_size).mean(
                                    dim=(1, 3))
                                module.grad_var = grad_var.view(module.n_blocks_out, module.block_size,
                                                                module.n_blocks_in, module.block_size).mean(dim=(1, 3))
                                module.act_var = act_var.view(module.n_blocks_in, module.block_size).mean(dim=1)

                                weight_blocks = module.weight.view(module.n_blocks_out, module.block_size,
                                                                   module.n_blocks_in, module.block_size)
                                module.weight_norm = weight_blocks.norm(dim=(1, 3))

                                self._store_block_features(name, module)

                            module.weight.grad = None

                h_f = module.register_forward_hook(forward_hook)
                h_b = module.register_full_backward_hook(backward_hook)
                self.handles += [h_f, h_b]

        print(f"Registered {len(self.handles) // 2} BIMLoRALinear hooks")

    def _get_module_info(self, name: str) -> Tuple[float, float]:

        parts = name.split('.')
        layer_idx = -1

        for i, part in enumerate(parts):
            if part == 'layer' and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_idx = int(parts[i + 1])
                break

        if layer_idx >= 0:

            layer_pos = (layer_idx + 1) / self.total_layers
        elif 'classifier' in name.lower() or 'pooler' in name.lower():

            layer_pos = 1.0
        elif 'embeddings' in name.lower():

            layer_pos = 0.0
        else:

            layer_pos = 0.5

        module_type = None
        name_lower = name.lower()

        if 'attention.self.query' in name_lower or name_lower.endswith('.query'):
            module_type = 0.0
        elif 'attention.self.key' in name_lower or name_lower.endswith('.key'):
            module_type = 0.167
        elif 'attention.self.value' in name_lower or name_lower.endswith('.value'):
            module_type = 0.333
        elif 'attention.output.dense' in name_lower:
            module_type = 0.5
        elif 'intermediate.dense' in name_lower:
            module_type = 0.667
        elif 'output.dense' in name_lower and 'attention' not in name_lower:
            module_type = 0.833
        elif 'classifier' in name_lower:
            module_type = 1.0

        if module_type is None:
            last_part = parts[-1].lower() if parts else ""
            if last_part in ['query', 'q_proj']:
                module_type = 0.0
            elif last_part in ['key', 'k_proj']:
                module_type = 0.167
            elif last_part in ['value', 'v_proj']:
                module_type = 0.333
            elif last_part in ['dense', 'out_proj'] and len(parts) > 1:

                if 'attention' in name_lower:
                    module_type = 0.5
                elif 'classifier' in name_lower:
                    module_type = 1.0
                else:
                    module_type = 0.833
            else:

                module_type = 0.5
                print(f"Warning: Using default module_type for {name}")

        return layer_pos, module_type

    def _store_block_features(self, module_name: str, module: BIMLoRALinear):
        """Store block features from a module - WITH TASK GRADIENTS"""
        layer_pos, module_type = self._get_module_info(module_name)

        with torch.no_grad():
            grad_mean = module.grad_mean.detach()
            grad_var = module.grad_var.detach()
            act_var = module.act_var.detach()
            weight_norm = module.weight_norm.detach()

            self.global_max['grad_mean'] = max(self.global_max['grad_mean'],
                                               float(grad_mean.max().item()))
            self.global_max['grad_var'] = max(self.global_max['grad_var'],
                                              float(grad_var.max().item()))
            self.global_max['act_var'] = max(self.global_max['act_var'],
                                             float(act_var.max().item()))
            self.global_max['weight_norm'] = max(self.global_max['weight_norm'],
                                                 float(weight_norm.max().item()))

        n_blocks_out, n_blocks_in = grad_mean.shape

        features_batch = {
            'module_name': module_name,
            'layer_pos': layer_pos,
            'module_type': module_type,
            'grad_mean': grad_mean.cpu().numpy(),
            'grad_var': grad_var.cpu().numpy(),
            'act_var': act_var.cpu().numpy(),
            'weight_norm': weight_norm.cpu().numpy(),
            'task_gradients': grad_mean.cpu().numpy(),
            'n_blocks_out': n_blocks_out,
            'n_blocks_in': n_blocks_in
        }

        if module_name in self.features and len(self.features[module_name]) > 0:

            self.features[module_name][0] = features_batch
        else:
            self.features[module_name].append(features_batch)

    def update(self):
        """Update step counter and clear processed modules set."""
        self.step_count += 1

        if hasattr(self, 'processed_modules'):
            self.processed_modules.clear()

    def finalize(self):

        print("\nFinalizing feature collection (with enhanced normalization)")

        for h in self.handles:
            h.remove()
        self.handles.clear()

        if not self.features:
            print("WARNING: No features collected!")
            return np.empty((0, self.config.mlp_input_dim), dtype=np.float32), []

        all_feats, all_meta = [], []
        for module_name in sorted(self.features):
            batch = self.features[module_name][0]
            grad_mean = torch.from_numpy(batch['grad_mean']).to(self.device)
            grad_var = torch.from_numpy(batch['grad_var']).to(self.device)
            act_var = torch.from_numpy(batch['act_var']).to(self.device)
            weight_n = torch.from_numpy(batch['weight_norm']).to(self.device)

            dyn = torch.stack([
                torch.log1p(grad_mean),
                torch.log1p(grad_var),
                torch.log1p(act_var).unsqueeze(0).expand_as(grad_mean),
                torch.log1p(weight_n)
            ], dim=-1)

            gmax = torch.tensor([
                math.log1p(self.global_max['grad_mean']),
                math.log1p(self.global_max['grad_var']),
                math.log1p(self.global_max['act_var']),
                math.log1p(self.global_max['weight_norm'])
            ], device=self.device).clamp_min(1e-8)

            dyn = (dyn / gmax).clamp(0, 1)

            layer_pos = torch.full_like(grad_mean, batch['layer_pos'])
            module_t = torch.full_like(grad_mean, batch['module_type'])
            feats = torch.stack([*dyn.unbind(-1), layer_pos, module_t], dim=-1)

            all_feats.append(feats.reshape(-1, self.config.mlp_input_dim).cpu())

            n_out, n_in = grad_mean.shape
            all_meta.extend({
                                'module_name': module_name,
                                'block_idx': (i, j)
                            } for i in range(n_out) for j in range(n_in))

        feature_array = torch.cat(all_feats, dim=0).numpy().astype(np.float32)
        self.features.clear()
        return feature_array, all_meta

    def save_features(self, save_path: str):
        """Save collected features to disk."""
        if not hasattr(self, 'feature_metadata'):
            raise ValueError("No features collected yet. Call finalize() first.")

        feature_array, _ = self.finalize()

        torch.save({
            'features': feature_array,
            'metadata': self.feature_metadata,
            'global_max': self.global_max,
            'block_size': self.block_size
        }, save_path)

        print(f"Features saved to {save_path}")

    def load_features(self, load_path: str) -> Tuple[np.ndarray, List[Dict]]:
        """Load features from disk, allowing NumPy arrays in PyTorch 2.6+."""
        import torch

        try:

            checkpoint = torch.load(load_path, weights_only=False)
        except TypeError:

            checkpoint = torch.load(load_path)
        except Exception as e:
            print(f"Error loading features from {load_path}: {e}")
            raise

        self.feature_metadata = checkpoint.get('metadata', [])
        self.global_max = checkpoint.get('global_max', {})
        self.block_size = checkpoint.get('block_size', self.config.block_size)

        return checkpoint['features'], checkpoint.get('metadata', [])