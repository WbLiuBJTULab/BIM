import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm


class ImplicitMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        input_dim = config.mlp_input_dim
        hidden_dim = config.mlp_hidden_dim
        dropout_p = config.mlp_dropout

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def predict_batch(self, features: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        """Predict importance scores for a large batch of features - FULLY GPU OPTIMIZED."""
        if batch_size is None:
            batch_size = self.config.mlp_predict_batch_size

        self.eval()
        device = next(self.parameters()).device
        n_samples = features.shape[0]

        if features.device != device:
            features = features.to(device)

        predictions = torch.zeros(n_samples, device=device)

        with torch.no_grad():

            actual_batch_size = min(batch_size * self.config.mlp_batch_multiplier, n_samples)

            if n_samples <= actual_batch_size:
                predictions = self.forward(features).squeeze(-1)
            else:

                for i in tqdm(range(0, n_samples, actual_batch_size), desc="MLP prediction"):
                    end_idx = min(i + actual_batch_size, n_samples)
                    batch = features[i:end_idx]

                    predictions[i:end_idx] = self.forward(batch).squeeze(-1)

        return predictions


class MLPTrainer:
    """Trainer for the implicit MLP with soft label generation."""

    def __init__(self, mlp: ImplicitMLP, device: torch.device):
        self.mlp = mlp.to(device)
        self.device = device
        self.config = mlp.config
        self.optimizer = torch.optim.Adam(
            self.mlp.parameters(),
            lr=self.config.mlp_lr,
            betas=(self.config.mlp_adam_beta1, self.config.mlp_adam_beta2)
        )

    def generate_soft_labels(self, features: np.ndarray,
                             weights: Optional[List[float]] = None,
                             task_gradients: Optional[np.ndarray] = None) -> np.ndarray:

        if weights is None:

            weights = self.config.mlp_feature_weights

        dynamic_features = features[:, :4].copy()

        if task_gradients is not None:
            task_grads = task_gradients.astype(np.float32)
            task_grads = (task_grads - task_grads.min()) / (task_grads.max() - task_grads.min() + 1e-8)
            dynamic_features[:, 0] *= (1.0 + task_grads)

        transformed_features = np.zeros_like(dynamic_features)

        transformed_features[:, 0] = np.sqrt(dynamic_features[:, 0] + 1e-8)

        transformed_features[:, 1] = np.cbrt(dynamic_features[:, 1] + 1e-8)

        transformed_features[:, 2] = np.log(dynamic_features[:, 2] + 1e-8)
        transformed_features[:, 3] = np.log(dynamic_features[:, 3] + 1e-8)

        normalized_features = np.zeros_like(transformed_features)
        for i in range(4):
            feat = transformed_features[:, i]

            if i == 0:
                q_low = np.percentile(feat, self.config.mlp_grad_percentile_low)
                q_high = np.percentile(feat, self.config.mlp_grad_percentile_high)
            else:
                q_low = np.percentile(feat, self.config.mlp_other_percentile_low)
                q_high = np.percentile(feat, self.config.mlp_other_percentile_high)

            if q_high - q_low > 1e-8:
                normalized_features[:, i] = np.clip((feat - q_low) / (q_high - q_low), 0, 1)
            else:
                normalized_features[:, i] = 0.5

        weighted_scores = np.sum(normalized_features * np.array(weights), axis=1)

        layer_pos = features[:, 4]
        module_type = features[:, 5]

        layer_weight = 1.0 + self.config.mlp_layer_weight_scale * np.exp(
            -((layer_pos - 0.5) ** 2) / self.config.mlp_layer_weight_sigma
        )

        type_weight = np.ones_like(module_type)

        module_weights = self.config.mlp_module_type_weights
        type_weight[np.abs(module_type - 0.5) < 0.01] = module_weights.get("attention_output", 1.8)
        type_weight[np.abs(module_type - 0.667) < 0.01] = module_weights.get("ffn_intermediate", 1.6)
        type_weight[np.abs(module_type - 0.833) < 0.01] = module_weights.get("ffn_output", 1.5)
        type_weight[np.abs(module_type - 0.333) < 0.01] = module_weights.get("value", 1.4)
        type_weight[np.abs(module_type - 0.167) < 0.01] = module_weights.get("key", 1.2)
        type_weight[np.abs(module_type - 0.0) < 0.01] = module_weights.get("query", 1.1)
        type_weight[np.abs(module_type - 1.0) < 0.01] = module_weights.get("classifier", 0.3)

        final_scores = weighted_scores * layer_weight * type_weight

        n_samples = len(final_scores)

        sorted_indices = np.argsort(final_scores)
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(n_samples)

        percentiles = ranks / (n_samples - 1)

        steepness = self.config.mlp_sigmoid_steepness
        midpoint = self.config.mlp_sigmoid_midpoint

        labels = 1 / (1 + np.exp(-steepness * (percentiles - midpoint)))

        labels = self.config.mlp_label_min + self.config.mlp_label_range * labels

        middle_mask = (percentiles > 0.25) & (percentiles < 0.75)
        noise = np.random.normal(0, self.config.mlp_noise_std, size=n_samples)
        labels[middle_mask] += noise[middle_mask]
        labels = np.clip(labels, self.config.mlp_label_clip_min, self.config.mlp_label_clip_max)

        if not hasattr(self, '_label_stats_printed'):
            print(f"\nGenerated soft labels statistics:")
            print(f"  Mean: {labels.mean():.3f}")
            print(f"  Std: {labels.std():.3f}")
            print(f"  Min: {labels.min():.3f}")
            print(f"  Max: {labels.max():.3f}")
            print(f"  10%: {np.percentile(labels, 10):.3f}")
            print(f"  25%: {np.percentile(labels, 25):.3f}")
            print(f"  50%: {np.percentile(labels, 50):.3f}")
            print(f"  75%: {np.percentile(labels, 75):.3f}")
            print(f"  90%: {np.percentile(labels, 90):.3f}")

            if labels.std() < self.config.mlp_min_std_threshold:
                print(f"⚠️  WARNING: Label standard deviation is too low! Adjusting...")
                labels = self._enhance_variance(labels)
                print(f"  Adjusted Std: {labels.std():.3f}")

            self._label_stats_printed = True

        return labels

    def _enhance_variance(self, labels: np.ndarray) -> np.ndarray:

        p10 = np.percentile(labels, 10)
        p90 = np.percentile(labels, 90)

        if p90 - p10 > 1e-8:
            labels = (labels - p10) / (p90 - p10)
            labels = self.config.mlp_variance_enhance_min + self.config.mlp_variance_enhance_range * labels
            labels = np.clip(labels, self.config.mlp_label_clip_min, self.config.mlp_label_clip_max)

        return labels

    def train_epoch(self, features: torch.Tensor, labels: torch.Tensor,
                    batch_size: Optional[int] = None) -> float:
        """Train one epoch - FULLY GPU OPTIMIZED VERSION."""
        if batch_size is None:
            batch_size = self.config.mlp_train_batch_size

        self.mlp.train()
        n_samples = features.shape[0]
        device = features.device

        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    param.grad = param.grad.to(device)

        indices = torch.randperm(n_samples, device=device)
        total_loss = 0.0
        n_batches = 0

        from tqdm import tqdm
        pbar = tqdm(range(0, n_samples, batch_size),
                    desc="MLP training",
                    ncols=100,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        for i in pbar:
            batch_idx = indices[i:i + batch_size]
            batch_features = features[batch_idx]
            batch_labels = labels[batch_idx]

            # Forward pass
            predictions = self.mlp(batch_features)
            loss = F.binary_cross_entropy(predictions, batch_labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss
            n_batches += 1

            pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'avg_loss': f'{total_loss / n_batches:.4f}'
            })

        return total_loss / n_batches

    def pretrain(self, features: np.ndarray, n_epochs: Optional[int] = None,
                 val_split: Optional[float] = None) -> Dict[str, List[float]]:

        if n_epochs is None:
            n_epochs = self.config.mlp_epochs
        if val_split is None:
            val_split = self.config.mlp_val_split

        device = next(self.mlp.parameters()).device
        print(f"MLP training on device: {device}")

        print("Generating soft labels...")
        labels = self.generate_soft_labels(features)

        features_tensor = torch.FloatTensor(features).to(device)
        labels_tensor = torch.FloatTensor(labels).to(device)

        print(f"Data moved to {device}")
        print(f"Features shape: {features_tensor.shape}, on device: {features_tensor.device}")
        print(f"Labels shape: {labels_tensor.shape}, on device: {labels_tensor.device}")

        n_samples = features_tensor.shape[0]
        n_val = int(n_samples * val_split)
        indices = torch.randperm(n_samples, device=device)

        train_features = features_tensor[indices[n_val:]]
        train_labels = labels_tensor[indices[n_val:]]
        val_features = features_tensor[indices[:n_val]]
        val_labels = labels_tensor[indices[:n_val]]

        print(f"Train set: {train_features.shape[0]} samples")
        print(f"Val set: {val_features.shape[0]} samples")

        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        patience = self.config.mlp_patience if self.config.mlp_patience is not None else self.config.mlp_early_stop_patience

        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch + 1}/{n_epochs}")

            train_loss = self.train_epoch(train_features, train_labels)
            history['train_loss'].append(train_loss)

            self.mlp.eval()
            with torch.no_grad():
                val_pred = self.mlp(val_features)
                val_loss = F.binary_cross_entropy(val_pred, val_labels).item()
            history['val_loss'].append(val_loss)

            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = self.mlp.state_dict()
                print("  ✓ New best model")
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

        if best_state is not None:
            self.mlp.load_state_dict(best_state)

        return history

    def validate_health(self, predictions: torch.Tensor) -> Dict[str, float]:
        """Check if MLP predictions are in healthy range - ENHANCED VERSION."""
        predictions = predictions.detach().cpu()

        health_metrics = {
            'mean': predictions.mean().item(),
            'std': predictions.std().item(),
            'min': predictions.min().item(),
            'max': predictions.max().item(),
            'zeros': (predictions < self.config.mlp_health_zero_threshold).sum().item() / len(predictions),
            'ones': (predictions > self.config.mlp_health_one_threshold).sum().item() / len(predictions),
            'q25': predictions.quantile(0.25).item(),
            'q50': predictions.quantile(0.50).item(),
            'q75': predictions.quantile(0.75).item()
        }

        # Check health criteria using configuration
        health_status = []
        mean_range = self.config.mlp_health_mean_range
        if not (mean_range[0] <= health_metrics['mean'] <= mean_range[1]):
            health_status.append(f"Mean {health_metrics['mean']:.3f} outside healthy range {mean_range}")

        std_range = self.config.mlp_health_std_range
        if not (std_range[0] <= health_metrics['std'] <= std_range[1]):
            health_status.append(f"Std {health_metrics['std']:.3f} outside healthy range {std_range}")

        if health_metrics['zeros'] > self.config.mlp_health_max_zeros:
            health_status.append(f"Too many near-zero predictions: {health_metrics['zeros']:.1%}")

        if health_metrics['ones'] > self.config.mlp_health_max_ones:
            health_status.append(f"Too many near-one predictions: {health_metrics['ones']:.1%}")

        iqr = health_metrics['q75'] - health_metrics['q25']
        if iqr < self.config.mlp_health_min_iqr:
            health_status.append(f"Predictions too concentrated: IQR = {iqr:.3f}")

        if health_status:
            print("⚠️ MLP Health Warnings:")
            for warning in health_status:
                print(f"  - {warning}")
            print(
                f"\n📊 Distribution: Q25={health_metrics['q25']:.3f}, Q50={health_metrics['q50']:.3f}, Q75={health_metrics['q75']:.3f}")
        else:
            print("✅ MLP predictions are healthy")
            print(f"📊 Distribution: Mean={health_metrics['mean']:.3f}, Std={health_metrics['std']:.3f}")
            print(
                f"   Quartiles: Q25={health_metrics['q25']:.3f}, Q50={health_metrics['q50']:.3f}, Q75={health_metrics['q75']:.3f}")

        return health_metrics