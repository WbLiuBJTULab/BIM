import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import torch
import json


@dataclass
class BIMLoRAConfig:

    model_name_or_path: str = "models/roberta-base"
    task_name: str = "sst2"
    max_seq_length: int = 128

    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: [
        "query", "key", "value", "dense",
        "attention.output.dense",
        "intermediate.dense",
        "output.dense"
    ])

    block_size: int = 4
    target_sparsity: float = 0.08
    collection_steps: int = 3000
    mlp_hidden_dim: int = 64
    mlp_dropout: float = 0.10
    mlp_lr: float = 1e-3
    mlp_epochs: int = 3
    mlp_patience: Optional[int] = None
    mask_temperature: float = 1.0
    initial_block_score: float = 0.5
    default_num_layers: int = 12
    word_dropout: int = 0
    step2_base_lr: float = 1.5e-4
    step2_patience: int = 40
    step2_warmup_epochs: int = 2

    step2_lora_lr: float = 2.8e-4
    step2_score_lr: float = 6.8e-4

    skip_step3: bool = True
    skip_step4: bool = True

    mlp_input_dim: int = 6
    mlp_adam_beta1: float = 0.9
    mlp_adam_beta2: float = 0.999

    mlp_train_batch_size: int = 67349
    mlp_predict_batch_size: int = 67349
    mlp_batch_multiplier: int = 4
    mlp_val_split: float = 0.1
    mlp_early_stop_patience: int = 10

    mlp_feature_weights: List[float] = field(default_factory=lambda: [0.60, 0.20, 0.15, 0.05])
    mlp_grad_percentile_low: float = 2.0
    mlp_grad_percentile_high: float = 98.0
    mlp_other_percentile_low: float = 5.0
    mlp_other_percentile_high: float = 95.0

    mlp_layer_weight_scale: float = 1.0
    mlp_layer_weight_sigma: float = 0.03

    mlp_module_type_weights: Dict[str, float] = field(default_factory=lambda: {
        "attention_output": 1.8,
        "ffn_intermediate": 1.6,
        "ffn_output": 1.5,
        "value": 1.4,
        "key": 1.2,
        "query": 1.1,
        "classifier": 0.3
    })

    mlp_sigmoid_steepness: float = 8.0
    mlp_sigmoid_midpoint: float = 0.5

    mlp_label_min: float = 0.05
    mlp_label_range: float = 0.9
    mlp_noise_std: float = 0.02
    mlp_label_clip_min: float = 0.01
    mlp_label_clip_max: float = 0.99
    mlp_min_std_threshold: float = 0.15

    mlp_variance_enhance_min: float = 0.1
    mlp_variance_enhance_range: float = 0.8

    mlp_health_mean_range: List[float] = field(default_factory=lambda: [0.40, 0.60])
    mlp_health_std_range: List[float] = field(default_factory=lambda: [0.15, 0.30])
    mlp_health_zero_threshold: float = 0.01
    mlp_health_one_threshold: float = 0.99
    mlp_health_max_zeros: float = 0.15
    mlp_health_max_ones: float = 0.15
    mlp_health_min_iqr: float = 0.2

    beta_init: float = 0.0
    beta_final: float = 1e-3
    beta_warmup_steps: int = 500

    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    num_train_epochs: int = 10
    warmup_ratio: float = 0.06
    warmup_steps: int = 0
    patience: int = 10

    hard_mask_threshold: float = 0.5
    use_8x8_blocks: bool = True
    final_finetune_steps: int = 4000

    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_accuracy"
    greater_is_better: bool = True

    data_dir: str = "data/glue"
    output_dir: str = "outputs"
    cache_dir: str = "cache"
    features_save_path: str = "outputs/features/features.pt"
    mlp_save_path: str = "outputs/features/mlp_init.pt"

    no_cuda: bool = False
    fp16: bool = True
    fp16_opt_level: str = "O1"
    local_rank: int = -1

    seed: int = 42

    logging_dir: Optional[str] = None
    logging_first_step: bool = True
    log_level: str = "info"
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])

    debug_mode: bool = False
    fast_dev_run: int = 0

    block_lr: float = 3e-3
    retention_reg_weight: float = 0.05
    final_lr: float = 5e-6
    lora_alpha_multiplier: int = 8
    enable_temperature_annealing: bool = True
    min_temperature: float = 0.1
    temperature_decay_rate: float = 0.95
    temperature_decay_steps: int = 100

    skip_visualization: bool = False

    def __post_init__(self):

        if self.logging_dir is None:
            self.logging_dir = os.path.join(self.output_dir, "logs")

        if self.fast_dev_run > 0:
            self.collection_steps = min(self.collection_steps, self.fast_dev_run * 10)
            self.eval_steps = self.fast_dev_run * 5
            self.save_steps = self.fast_dev_run * 10
            self.logging_steps = self.fast_dev_run
            self.final_finetune_steps = self.fast_dev_run * 5

    @property
    def device(self) -> torch.device:

        if self.no_cuda:
            return torch.device("cpu")
        elif self.local_rank >= 0:
            return torch.device("cuda", self.local_rank)
        else:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def n_gpu(self) -> int:

        if self.no_cuda:
            return 0
        elif self.local_rank >= 0:
            return 1
        else:
            return torch.cuda.device_count()

    @property
    def world_size(self) -> int:

        if self.local_rank >= 0:
            return int(os.environ.get("WORLD_SIZE", 1))
        return 1

    @property
    def process_rank(self) -> int:

        if self.local_rank >= 0:
            return int(os.environ.get("RANK", 0))
        return 0

    @property
    def is_main_process(self) -> bool:

        return self.process_rank == 0

    def to_dict(self) -> Dict[str, Any]:

        return {k: v for k, v in self.__dict__.items()}

    def save(self, save_path: str):

        with open(save_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, load_path: str) -> "BIMLoRAConfig":

        with open(load_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


def get_default_config(task_name: str = "sst2") -> BIMLoRAConfig:

    config = BIMLoRAConfig(task_name=task_name)

    task_configs = {
        "sst2": {

            "num_train_epochs": 15,
            "per_device_train_batch_size": 32,
            "learning_rate": 1e-4,
            "max_seq_length": 128,
            "warmup_ratio": 0.06,
            "target_sparsity": 0.08,
            "lora_r": 16,
            "lora_alpha": 16,
            "mlp_epochs": 3,
            "mlp_lr": 1e-3,
            "eval_steps": 100,
            "gradient_accumulation_steps": 1,
            "collection_steps": 3000,
            "save_steps": 1000,
            "lora_dropout": 0.0,

            "mask_temperature": 1.0,
            "beta_final": 1e-3,
            "beta_warmup_steps": 500,

            "block_lr": 3e-3,
            "retention_reg_weight": 0.05,
            "enable_temperature_annealing": True,
            "min_temperature": 1.0,
            "temperature_decay_rate": 0.998,
            "temperature_decay_steps": 2000,
            "patience": 30,

        },
        "mrpc": {
            "num_train_epochs": 22,
            "per_device_train_batch_size": 32,
            "per_device_eval_batch_size": 64,
            "learning_rate": 1e-5,
            "max_seq_length": 128,
            "warmup_ratio": 0.06,
            "target_sparsity": 0.08,
            "lora_r": 16,
            "lora_alpha": 16,
            "mlp_epochs": 12,
            "mlp_lr": 1e-3,
            "eval_steps": 10,
            "gradient_accumulation_steps": 1,
            "collection_steps": 1200,
            "save_steps": 1000,
            "lora_dropout": 0.10,

            "mask_temperature": 1.0,
            "beta_final": 1e-3,
            "beta_warmup_steps": 500,

            "block_lr": 3e-3,
            "retention_reg_weight": 0.05,
            "enable_temperature_annealing": True,
            "min_temperature": 1.0,
            "temperature_decay_rate": 0.998,
            "temperature_decay_steps": 2000,
            "patience": 30,

        },
        "cola": {
            "num_train_epochs": 22,
            "per_device_train_batch_size": 32,
            "per_device_eval_batch_size": 64,
            "learning_rate": 1e-5,
            "max_seq_length": 128,
            "warmup_ratio": 0.06,
            "target_sparsity": 0.08,
            "lora_r": 16,
            "lora_alpha": 16,
            "mlp_epochs": 12,
            "mlp_lr": 1e-3,
            "eval_steps": 10,
            "gradient_accumulation_steps": 1,
            "collection_steps": 1200,
            "save_steps": 1000,
            "lora_dropout": 0.10,

            "mask_temperature": 1.0,
            "beta_final": 1e-3,
            "beta_warmup_steps": 500,

            "block_lr": 3e-3,
            "retention_reg_weight": 0.05,
            "enable_temperature_annealing": True,
            "min_temperature": 1.0,
            "temperature_decay_rate": 0.998,
            "temperature_decay_steps": 2000,
            "patience": 30,

            "metric_for_best_model": "eval_mcc",
            "greater_is_better": True,

        },
        "mnli": {
            "num_train_epochs": 3,

        },
        "qnli": {

            "num_train_epochs": 15,
            "per_device_train_batch_size": 32,
            "learning_rate": 3e-5,
            "max_seq_length": 256,
            "warmup_ratio": 0.06,
            "target_sparsity": 0.08,
            "lora_r": 16,
            "lora_alpha": 16,
            "mlp_epochs": 4,
            "mlp_lr": 1e-3,
            "eval_steps": 200,
            "gradient_accumulation_steps": 1,
            "collection_steps": 5000,
            "save_steps": 1000,
            "lora_dropout": 0.05,

            "mask_temperature": 1.0,
            "beta_final": 1e-3,
            "beta_warmup_steps": 500,

            "block_lr": 3e-3,
            "retention_reg_weight": 0.05,
            "enable_temperature_annealing": True,
            "min_temperature": 1.0,
            "temperature_decay_rate": 0.998,
            "temperature_decay_steps": 2000,
            "patience": 30,
        },
        "qqp": {

            "num_train_epochs": 15,
            "per_device_train_batch_size": 32,
            "per_device_eval_batch_size": 128,
            "learning_rate": 3e-4,
            "max_seq_length": 128,
            "warmup_ratio": 0.1,
            "target_sparsity": 0.08,
            "lora_r": 16,
            "lora_alpha": 16,
            "mlp_epochs": 10,
            "mlp_lr": 8e-4,
            "mlp_dropout": 0.08,
            "eval_steps": 800,
            "gradient_accumulation_steps": 1,
            "collection_steps": 5000,
            "save_steps": 1000,
            "lora_dropout": 0.05,

            "mask_temperature": 1.0,
            "beta_final": 1e-3,
            "beta_warmup_steps": 500,

            "block_lr": 3e-3,
            "retention_reg_weight": 0.05,
            "enable_temperature_annealing": True,
            "min_temperature": 2.0,
            "temperature_decay_rate": 0.998,
            "temperature_decay_steps": 2000,
            "patience": 30,
        },
        "rte": {

            "num_train_epochs": 22,
            "per_device_train_batch_size": 32,
            "per_device_eval_batch_size": 64,
            "learning_rate": 2e-5,
            "max_seq_length": 128,
            "warmup_ratio": 0.06,
            "target_sparsity": 0.08,
            "lora_r": 16,
            "lora_alpha": 16,
            "mlp_epochs": 3,
            "mlp_lr": 8e-4,
            "eval_steps": 20,
            "gradient_accumulation_steps": 1,
            "collection_steps": 1200,
            "save_steps": 1000,
            "lora_dropout": 0.10,

            "mask_temperature": 1.0,
            "beta_final": 1e-3,
            "beta_warmup_steps": 500,

            "block_lr": 3e-3,
            "retention_reg_weight": 0.05,
            "enable_temperature_annealing": True,
            "min_temperature": 1.0,
            "temperature_decay_rate": 0.998,
            "temperature_decay_steps": 2000,
            "patience": 30,

        },
        "stsb": {
            "num_train_epochs": 18,
            "per_device_train_batch_size": 32,
            "per_device_eval_batch_size": 64,
            "learning_rate": 2e-5,
            "max_seq_length": 128,
            "warmup_ratio": 0.1,
            "target_sparsity": 0.08,
            "lora_r": 16,
            "lora_alpha": 16,
            "mlp_epochs": 5,
            "mlp_lr": 1e-3,
            "eval_steps": 20,
            "gradient_accumulation_steps": 1,
            "collection_steps": 1500,
            "save_steps": 200,
            "lora_dropout": 0.05,

            "metric_for_best_model": "eval_pearson",
            "greater_is_better": True,

            "mask_temperature": 1.0,
            "beta_final": 1e-3,
            "beta_warmup_steps": 200,

            "block_lr": 3e-3,
            "retention_reg_weight": 0.05,
            "enable_temperature_annealing": True,
            "min_temperature": 1.0,
            "temperature_decay_rate": 0.998,
            "temperature_decay_steps": 500,
            "patience": 15,

        }
    }

    if task_name in task_configs:
        for key, value in task_configs[task_name].items():
            setattr(config, key, value)

    return config