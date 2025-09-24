import os
import json
from typing import Dict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

from transformers import (
    get_linear_schedule_with_warmup,
    set_seed,
)
from transformers.trainer_utils import EvalPrediction

from bim_lora import (
    BIMLoRALinear,
    ImplicitMLP,
    FeatureCollector,
    MaskAllocator,
    MLPTrainer,
    mark_only_bim_lora_as_trainable,
    save_bim_lora_model,
)

from torch.utils.tensorboard import SummaryWriter

class BIMLoRATrainer:


    def __init__(
            self,
            config,
            model: nn.Module,
            train_dataset,
            eval_dataset,
            tokenizer,
            compute_metrics
    ):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.compute_metrics = compute_metrics

        set_seed(config.seed)

        self.setup_distributed()

        self.model = self.model.to(self.config.device)

        self.implicit_mlp = ImplicitMLP(self.config).to(self.config.device)

        if self.config.world_size > 2:
            self.feature_collector = self.model
        else:
            self.feature_collector = FeatureCollector(
                self.model,
                config
            )

        self.mask_allocator = MaskAllocator(
            self.model,
            config
        )

        self.setup_data_loaders()

        self.setup_optimization()

        self.scaler = GradScaler() if config.fp16 else None

        if config.is_main_process:
            self.writer = SummaryWriter(config.logging_dir)
        else:
            self.writer = None

        self.global_step = 0
        self.epoch = 0
        self.best_metric = None
        self.features_collected = False
        self.mlp_pretrained = False

    def setup_distributed(self):

        if self.config.local_rank >= 0:
            torch.cuda.set_device(self.config.local_rank)

            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')

            self.model = DDP(
                self.model,
                device_ids=[self.config.local_rank],
                output_device=self.config.local_rank,
                find_unused_parameters=True
            )

        print(f"Process rank: {self.config.process_rank}/{self.config.world_size}")

    def setup_data_loaders(self):

        if self.config.world_size > 1:
            train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.config.world_size,
                rank=self.config.process_rank,
                shuffle=True
            )
        else:
            train_sampler = None

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )

        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.per_device_eval_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    def setup_optimization(self):
        mark_only_bim_lora_as_trainable(self.model, bias='none')

        lora_params = [p for n, p in self.model.named_parameters() if p.requires_grad]

        mlp_params = list(self.implicit_mlp.parameters())

        self.optimizer = torch.optim.AdamW([
            {'params': lora_params, 'lr': self.config.learning_rate},
            {'params': mlp_params, 'lr': self.config.mlp_lr}
        ],
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay
        )

        steps_per_epoch = len(self.train_loader) // self.config.gradient_accumulation_steps
        self.total_steps = steps_per_epoch * self.config.num_train_epochs

        if self.config.warmup_steps > 0:
            warmup_steps = self.config.warmup_steps
        else:
            warmup_steps = int(self.total_steps * self.config.warmup_ratio)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps
        )

    def train(self):

        print(f"\n{'=' * 50}")
        print(f"Starting BIM-LoRA Training")
        print(f"{'=' * 50}")
        print(f"Total epochs: {self.config.num_train_epochs}")
        print(f"Total steps: {self.total_steps}")
        print(f"Device: {self.config.device}")
        print(f"World size: {self.config.world_size}")
        print(f"{'=' * 50}\n")

        progress_file = os.path.join(self.config.output_dir, "training_progress.json")
        progress = {
            "step0_completed": False,
            "step1_completed": False,
            "step2_completed": False,
        }

        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                saved_progress = json.load(f)
                progress.update(saved_progress)
            print(f"Found existing progress: {progress}")

        if not progress["step0_completed"]:
            self.step0_feature_collection()
            progress["step0_completed"] = True
            with open(progress_file, 'w') as f:
                json.dump(progress, f)
            print("✓ Step 0 completed and saved")
        else:
            print("✓ Step 0 already completed, skipping...")
            self.features_collected = True

        if not progress["step1_completed"]:
            self.step1_mlp_pretraining()
            progress["step1_completed"] = True
            with open(progress_file, 'w') as f:
                json.dump(progress, f)
            print("✓ Step 1 completed and saved")
        else:
            print("✓ Step 1 already completed, loading MLP...")
            if os.path.exists(self.config.mlp_save_path):
                self.implicit_mlp.load_state_dict(torch.load(self.config.mlp_save_path))
                self.mlp_pretrained = True
            else:
                print("WARNING: MLP checkpoint not found, retraining...")
                self.step1_mlp_pretraining()

        if not progress["step2_completed"]:
            best_metric = self.step2_joint_finetuning()
            progress["step2_completed"] = True
            with open(progress_file, 'w') as f:
                json.dump(progress, f)
            print("✓ Step 2 completed and saved")
            print(f"✓ Final best accuracy: {best_metric:.4f}")
        else:
            print("✓ Step 2 already completed, loading best model...")
            best_model_path = os.path.join(self.config.output_dir, "checkpoint-best", "bim_lora_model.pt")
            if os.path.exists(best_model_path):
                from bim_lora import load_bim_lora_model
                metadata = load_bim_lora_model(self.model, self.implicit_mlp, best_model_path)
                print(f"Loaded best model from step 2")
                if 'best_metric' in metadata:
                    self.best_metric = metadata['best_metric']
                    print(f"Best metric: {self.best_metric:.4f}")

        if self.config.is_main_process:
            self.save_model("final")

        print(f"\n{'=' * 50}")
        print(f"Training Complete!")
        print(f"Best accuracy achieved: {self.best_metric:.4f}")
        print(f"{'=' * 50}\n")

    def step0_feature_collection(self):

        print(f"\n{'=' * 30}")
        print("Step 0: Feature Collection")
        print(f"{'=' * 30}")

        if os.path.exists(self.config.features_save_path) and not self.config.debug_mode:
            print(f"Features already exist at {self.config.features_save_path}, skipping collection.")
            self.features_collected = True
            return

        print("Disabling masking for feature collection...")
        for name, module in self.model.named_modules():
            if isinstance(module, BIMLoRALinear):
                module.enable_masking = False

                layer_idx = -1
                parts = name.split('.')
                for i, part in enumerate(parts):
                    if part == 'layer' and i + 1 < len(parts):
                        try:
                            layer_idx = int(parts[i + 1])
                            break
                        except (ValueError, IndexError):
                            pass

                module._layer_idx = layer_idx
                module._module_name = name

        self.feature_collector.register_hooks()

        for name, param in self.model.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = True

        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
            weight_decay=0.0
        )

        self.model.train()
        collection_bar = tqdm(
            total=self.config.collection_steps,
            desc="Collecting features",
            disable=not self.config.is_main_process
        )

        step = 0
        data_iter = iter(self.train_loader)

        while step < self.config.collection_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            batch = {k: v.to(self.config.device) for k, v in batch.items()}

            with autocast(enabled=self.config.fp16):
                outputs = self.model(**batch)
                loss = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss at step {step}: {loss.item()}")
                continue

            optimizer.zero_grad()

            if self.config.fp16:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0
                )
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0
                )
                optimizer.step()

            self.feature_collector.update()

            step += 1
            collection_bar.update(1)

            if step % 50 == 0:
                collection_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'collected': len(self.feature_collector.features)
                })


            if step % self.config.logging_steps == 0 and self.writer:
                self.writer.add_scalar('feature_collection/loss', loss.item(), step)

        collection_bar.close()

        for name, param in self.model.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False

        print("Re-enabling masking after feature collection...")
        for name, module in self.model.named_modules():
            if isinstance(module, BIMLoRALinear):
                module.enable_masking = True

        print("\nFinalizing and saving features...")

        if self.config.is_main_process:
            features, metadata = self.feature_collector.finalize()

            if len(features) == 0:
                raise ValueError("No features collected! Check hooks and model forward pass.")

            os.makedirs(os.path.dirname(self.config.features_save_path), exist_ok=True)
            torch.save({
                'features': features,
                'metadata': metadata,
                'global_max': self.feature_collector.global_max,
                'block_size': self.feature_collector.block_size
            }, self.config.features_save_path)

            print(f"✓ Features saved to {self.config.features_save_path}")
            print(f"✓ Total features collected: {len(features)}")
            self.features_collected = True

        if self.config.world_size > 1:
            dist.barrier()

    def step1_mlp_pretraining(self):

        print(f"\n{'=' * 30}")
        print("Step 1: MLP Pretraining")
        print(f"{'=' * 30}")

        print(f"MLP training epochs: {self.config.mlp_epochs}")

        if os.path.exists(self.config.mlp_save_path) and not self.config.debug_mode:
            print(f"MLP model already exists at {self.config.mlp_save_path}")
            self.implicit_mlp.load_state_dict(torch.load(self.config.mlp_save_path))
            self.mlp_pretrained = True
            return

        features, _ = self.feature_collector.load_features(self.config.features_save_path)
        print(f"Loaded {len(features)} features")

        if self.config.is_main_process:

            self.implicit_mlp = self.implicit_mlp.to(self.config.device)
            print(f"MLP moved to {self.config.device}")

            mlp_trainer = MLPTrainer(self.implicit_mlp, self.config.device)

            print(f"Starting MLP pretraining for {self.config.mlp_epochs} epochs...")
            history = mlp_trainer.pretrain(
                features,
                n_epochs=self.config.mlp_epochs,
                val_split=0.1
            )

            torch.save(self.implicit_mlp.state_dict(), self.config.mlp_save_path)
            print(f"✓ MLP model saved to {self.config.mlp_save_path}")

            print("\nValidating MLP predictions...")
            with torch.no_grad():

                features_tensor = torch.FloatTensor(features).to(self.config.device)

                predictions = self.implicit_mlp.predict_batch(
                    features_tensor,
                    batch_size=self.config.mlp_predict_batch_size
                )

            health_metrics = mlp_trainer.validate_health(predictions)

        if self.config.world_size > 1:
            dist.barrier()
            if not self.config.is_main_process:
                self.implicit_mlp.load_state_dict(torch.load(self.config.mlp_save_path))

        self.mlp_pretrained = True
        print("✓ Step 1 completed")

    def _analyze_layer_importance(self):

        print("\nAnalyzing layer importance dynamically...")

        self.model.train()
        layer_gradients = {}

        for name, param in self.model.named_parameters():
            if 'weight' in name and 'lora_' not in name:
                param.requires_grad = True

        data_iter = iter(self.train_loader)
        num_analysis_steps = 50

        for step in range(num_analysis_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            batch = {k: v.to(self.config.device) for k, v in batch.items()}

            with autocast(enabled=self.config.fp16):
                outputs = self.model(**batch)
                loss = outputs.loss

            if self.config.fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            for name, module in self.model.named_modules():
                if isinstance(module, BIMLoRALinear):
                    if module.weight.grad is not None:
                        grad_norm = module.weight.grad.abs().mean().item()

                        if name not in layer_gradients:
                            layer_gradients[name] = []
                        layer_gradients[name].append(grad_norm)

                        module.weight.grad = None

        for name, param in self.model.named_parameters():
            if 'weight' in name and 'lora_' not in name:
                param.requires_grad = False

        layer_importance = {}
        for name, grads in layer_gradients.items():
            layer_importance[name] = np.mean(grads)

        if layer_importance:
            min_grad = min(layer_importance.values())
            max_grad = max(layer_importance.values())
            grad_range = max_grad - min_grad + 1e-8

            for name in layer_importance:
                layer_importance[name] = (layer_importance[name] - min_grad) / grad_range

        print("\nLayer importance scores (based on gradient magnitude):")
        for name, score in sorted(layer_importance.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {name}: {score:.4f}")

        return layer_importance

    def step2_joint_finetuning(self):

        print(f"\n{'=' * 30}")
        print("Step 2: Joint Training with Better Initialization (MRPC)")
        print(f"{'=' * 30}")

        layer_importance = self._analyze_layer_importance()

        print("\n1. Loading features to GPU...")
        checkpoint = torch.load(self.config.features_save_path,
                                map_location=self.config.device,
                                weights_only=False)
        features = checkpoint['features']
        metadata = checkpoint.get('metadata', [])
        features_tensor = torch.FloatTensor(features).to(self.config.device)
        print(f"Features loaded to {self.config.device}: {features_tensor.shape}")

        module_indices = {}
        for idx, meta in enumerate(metadata):
            module_name = meta['module_name']
            block_idx = meta['block_idx']
            if module_name not in module_indices:
                module_indices[module_name] = []
            module_indices[module_name].append((idx, block_idx))

        print("\n2. MLP predictions on GPU...")
        self.implicit_mlp = self.implicit_mlp.to(self.config.device)
        with torch.no_grad():
            self.implicit_mlp.eval()
            mlp_predictions = self.implicit_mlp(features_tensor).squeeze(-1)
        print(f"Predictions shape: {mlp_predictions.shape}, device: {mlp_predictions.device}")

        print("\n3. Better initialization of block_scores... [MRPC]")

        for module_name, indices in module_indices.items():
            if module_name in self.mask_allocator.bim_modules:
                module = self.mask_allocator.bim_modules[module_name]

                new_scores = torch.zeros_like(module.block_scores)
                feature_indices = [idx for idx, _ in indices]
                block_i_indices = [i for _, (i, j) in indices]
                block_j_indices = [j for _, (i, j) in indices]

                batch_predictions = mlp_predictions[feature_indices]

                if module_name in layer_importance:
                    importance = layer_importance[module_name]

                    if 'attention.output.dense' in module_name:
                        min_score, max_score = 0.20, 0.60
                    elif 'output.dense' in module_name and 'attention' not in module_name:
                        min_score, max_score = 0.20, 0.55
                    elif 'intermediate.dense' in module_name:
                        min_score, max_score = 0.10, 0.40
                    elif 'value' in module_name:
                        min_score, max_score = 0.12, 0.45
                    elif 'key' in module_name:
                        min_score, max_score = 0.05, 0.25
                    elif 'query' in module_name:
                        min_score, max_score = 0.02, 0.20
                    else:
                        min_score, max_score = 0.10, 0.50

                    importance_boost = importance * 0.12
                    min_score = min(0.95, min_score + 0.5 * importance_boost)
                    max_score = min(0.99, max_score + 1.0 * importance_boost)

                    batch_predictions = min_score + (max_score - min_score) * batch_predictions

                    print(f"  {module_name.split('.')[-1]}: imp={importance:.3f}, "
                          f"range=[{min_score:.2f}, {max_score:.2f}]")
                else:
                    batch_predictions = 0.05 + 0.5 * batch_predictions

                batch_predictions = torch.clamp(batch_predictions, 0.01, 0.99)
                logits = torch.log(batch_predictions / (1 - batch_predictions))

                new_scores[block_i_indices, block_j_indices] = logits
                module.block_scores.data = new_scores.to(module.block_scores.device)

                with torch.no_grad():
                    mean_score = torch.sigmoid(new_scores).mean().item()
                    std_score = torch.sigmoid(new_scores).std().item()
                    print(f"    Final(init-dist): mean={mean_score:.3f}, std={std_score:.3f}")

        print("\n4. Setting up parameters on GPU...")

        block_score_params, lora_params, clf_params = [], [], []
        for name, module in self.model.named_modules():
            if isinstance(module, BIMLoRALinear):
                if not module.block_scores.is_cuda:
                    module.block_scores = module.block_scores.to(self.config.device)
                if not isinstance(module.block_scores, nn.Parameter):
                    module.block_scores = nn.Parameter(module.block_scores.data)
                module.block_scores.requires_grad = True
                block_score_params.append(module.block_scores)

                if hasattr(module, 'lora_A'):
                    if not module.lora_A.is_cuda:
                        module.lora_A = module.lora_A.to(self.config.device)
                    if module.lora_A.requires_grad:
                        lora_params.append(module.lora_A)
                if hasattr(module, 'lora_B'):
                    if not module.lora_B.is_cuda:
                        module.lora_B = module.lora_B.to(self.config.device)
                    if module.lora_B.requires_grad:
                        lora_params.append(module.lora_B)

                module.mask_temperature = 1.5
                module.enable_masking = True

        for n, p in self.model.named_parameters():
            if 'classifier.out_proj' in n:
                p.requires_grad = True
                clf_params.append(p)

        print(f"  LoRA parameters: {len(lora_params)} (on {self.config.device})")
        print(f"  Block score parameters: {len(block_score_params)} (on {self.config.device})")
        print(f"  Classifier head parameters: {len(clf_params)} (on {self.config.device})")

        optimizer = torch.optim.AdamW(
            [
                {'params': lora_params, 'lr': 5e-4, 'weight_decay': 0.01},
                {'params': block_score_params, 'lr': 5e-4, 'weight_decay': 0.00},
                {'params': clf_params, 'lr': 1.5e-3, 'weight_decay': 0.00},
            ],
            betas=(0.9, 0.98)
        )

        steps_per_epoch = len(self.train_loader)
        total_steps = steps_per_epoch * self.config.num_train_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=1e-6
        )

        print("\n5. Initial evaluation...")
        initial_results = self.evaluate()
        best_metric = initial_results.get('eval_accuracy', 0.0)
        print(f"Initial accuracy: {best_metric:.4f}")
        self.best_metric = best_metric

        print(f"\n6. Training for {self.config.num_train_epochs} epochs...")

        patience_counter = 0
        max_patience = 15

        for epoch in range(self.config.num_train_epochs):
            self.model.train()
            epoch_loss, epoch_steps = 0.0, 0

            if epoch < 6:
                target_temperature = 1.4
            elif epoch < 12:
                target_temperature = 1.0
            elif epoch < 20:
                target_temperature = 0.8
            elif epoch < 26:
                target_temperature = 0.6
            else:
                target_temperature = 0.45

            for m in self.model.modules():
                if isinstance(m, BIMLoRALinear):
                    m.mask_temperature = target_temperature

            progress_bar = tqdm(self.train_loader,
                                desc=f"Epoch {epoch + 1}/{self.config.num_train_epochs}")

            for step, batch in enumerate(progress_bar):
                batch = {k: v.to(self.config.device, non_blocking=True) for k, v in batch.items()}

                with autocast(enabled=self.config.fp16):
                    outputs = self.model(**batch)
                    loss = outputs.loss

                loss = loss / self.config.gradient_accumulation_steps

                if self.config.fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.fp16:
                        self.scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(lora_params, self.config.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(block_score_params, 5.0)

                    if self.config.fp16:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad()
                    self.global_step += 1

                    epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                    epoch_steps += 1

                    if step % 50 == 0:
                        with torch.no_grad():
                            all_scores = []
                            for m in self.model.modules():
                                if isinstance(m, BIMLoRALinear):
                                    s = torch.sigmoid(m.block_scores / m.mask_temperature)
                                    all_scores.append(s.flatten())
                            if all_scores:
                                all_scores_tensor = torch.cat(all_scores)
                                mean_score = all_scores_tensor.mean().item()
                                std_score = all_scores_tensor.std().item()
                                active_ratio = (all_scores_tensor > 0.5).float().mean().item()
                            else:
                                mean_score, std_score, active_ratio = 0.5, 0.0, 0.5

                        progress_bar.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'score': f'{mean_score:.3f}±{std_score:.3f}',
                            'active': f'{active_ratio:.2%}',
                            'temp': f'{target_temperature:.2f}',
                            'best': f'{best_metric:.4f}'
                        })

                    if self.global_step % self.config.eval_steps == 0:
                        eval_results = self.evaluate()
                        eval_metric = eval_results.get('eval_accuracy', 0.0)

                        print(f"\nStep {self.global_step}:")
                        print(f"  Accuracy: {eval_metric:.4f}")
                        print(f"  Mean score: {mean_score:.3f} (std: {std_score:.3f})")
                        print(f"  Active ratio: {active_ratio:.2%}")
                        print(f"  Temperature: {target_temperature:.2f}")

                        if eval_metric > best_metric + 0.0001:
                            improvement = eval_metric - best_metric
                            best_metric = eval_metric
                            self.best_metric = best_metric
                            patience_counter = 0
                            self.save_model("best")
                            print(f"  ✓ New best: {best_metric:.4f} (+{improvement:.4f})")
                        else:
                            patience_counter += 1
                            print(f"  No improvement ({patience_counter}/{max_patience})")

            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Loss: {epoch_loss / max(1, epoch_steps):.4f}")
            print(f"  Best accuracy: {best_metric:.4f}")
            print(f"  Current temperature: {target_temperature:.2f}")

            with torch.no_grad():
                print("\nScore distribution by module type:")
                module_types = {'query': [], 'key': [], 'value': [], 'dense': []}
                for name, m in self.model.named_modules():
                    if isinstance(m, BIMLoRALinear):
                        s = torch.sigmoid(m.block_scores / m.mask_temperature)
                        mean_val = s.mean().item()
                        if 'query' in name:
                            module_types['query'].append(mean_val)
                        elif 'key' in name:
                            module_types['key'].append(mean_val)
                        elif 'value' in name:
                            module_types['value'].append(mean_val)
                        elif 'dense' in name:
                            module_types['dense'].append(mean_val)
                for t, vals in module_types.items():
                    if vals:
                        print(f"  {t}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

            if patience_counter >= max_patience and epoch >= 9:
                print("Early stopping triggered")
                break

        print(f"\n{'=' * 30}")
        print(f"Step 2 Complete! Best accuracy: {best_metric:.4f}")
        print(f"{'=' * 30}")

        return best_metric

    def evaluate(self) -> Dict[str, float]:

        self.model.eval()

        all_predictions = []
        all_labels = []
        total_loss = 0.0
        total_steps = 0

        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating", disable=not self.config.is_main_process):

                batch = {k: v.to(self.config.device, non_blocking=True)
                         for k, v in batch.items()}
                labels = batch.get('labels')

                with autocast(enabled=self.config.fp16):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    logits = outputs.logits

                total_loss += loss.item()
                total_steps += 1

                if labels is not None:

                    all_predictions.append(logits)
                    all_labels.append(labels)

        results = {"eval_loss": total_loss / total_steps}

        if all_labels:

            predictions_tensor = torch.cat(all_predictions, dim=0).cpu()
            labels_tensor = torch.cat(all_labels, dim=0).cpu()

            predictions_array = predictions_tensor.numpy()
            labels_array = labels_tensor.numpy()

            eval_pred = EvalPrediction(
                predictions=predictions_array,
                label_ids=labels_array
            )
            metrics = self.compute_metrics(eval_pred)

            for key, value in metrics.items():
                if not key.startswith('eval_'):
                    results[f"eval_{key}"] = value
                else:
                    results[key] = value

        self.model.train()
        return results

    def save_model(self, suffix: str = ""):

        output_dir = os.path.join(self.config.output_dir, f"checkpoint-{suffix}")
        os.makedirs(output_dir, exist_ok=True)

        self.config.save(os.path.join(output_dir, "config.json"))

        save_path = os.path.join(output_dir, "bim_lora_model.pt")

        model_to_save = self.model
        if hasattr(self.model, 'module'):
            model_to_save = self.model.module

        metadata = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "config": self.config.to_dict()
        }

        save_bim_lora_model(
            model_to_save,
            self.implicit_mlp,
            save_path,
            metadata
        )

        self.tokenizer.save_pretrained(output_dir)

        print(f"Model saved to {output_dir}")