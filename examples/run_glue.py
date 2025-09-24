import os
import sys
import json
import argparse
from typing import Dict, List, Optional, Tuple, Any
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed
)

from transformers.trainer_utils import EvalPrediction
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bim_lora import get_bim_lora_model
from examples.config import BIMLoRAConfig, get_default_config
from examples.trainer import BIMLoRATrainer

TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

LABEL_TO_ID = {
    "cola": {"unacceptable": 0, "acceptable": 1, "0": 0, "1": 1},
    "mnli": {"entailment": 0, "neutral": 1, "contradiction": 2, "0": 0, "1": 1, "2": 2},
    "mrpc": {"not_equivalent": 0, "equivalent": 1, "0": 0, "1": 1},
    "qnli": {"entailment": 0, "not_entailment": 1, "0": 0, "1": 1},
    "qqp": {"not_duplicate": 0, "duplicate": 1, "0": 0, "1": 1},
    "rte": {"entailment": 0, "not_entailment": 1, "0": 0, "1": 1},
    "sst2": {"negative": 0, "positive": 1, "0": 0, "1": 1},
    "stsb": None,
    "wnli": {"not_entailment": 0, "entailment": 1, "0": 0, "1": 1},
}


class GLUEDataset(Dataset):


    def __init__(self, data_path: str, tokenizer, config, augment: bool = False):
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.max_seq_length
        self.task_name = config.task_name.lower()
        self.augment = augment
        self.word_dropout = getattr(config, 'word_dropout', 0.0)

        self.is_regression = (self.task_name == "stsb")

        self.sentence1_key, self.sentence2_key = TASK_TO_KEYS[self.task_name]

        self.label_to_id = LABEL_TO_ID.get(self.task_name, {})

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        self.data = self._load_data(data_path)

        if len(self.data) == 0:
            raise ValueError(f"No valid data found in {data_path}")

        print(f"Loaded {len(self.data)} examples from {data_path} for {self.task_name}")

        self._print_data_stats()

    def _load_data(self, data_path: str) -> List[Dict]:

        data = []

        file_ext = os.path.splitext(data_path)[1].lower()

        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content.startswith('[') and content.endswith(']'):

                    json_data = json.loads(content)
                    for item in json_data:
                        processed = self._process_example(item)
                        if processed is not None:
                            data.append(processed)
                    if data:
                        print(f"Successfully loaded as JSON array format")
                        return data
        except:
            pass

        data = []
        error_count = 0
        total_lines = 0

        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                total_lines += 1
                line = line.strip()

                if not line:
                    continue

                if line.startswith('#') or line.startswith('//'):
                    continue

                try:

                    item = json.loads(line)
                    processed = self._process_example(item)
                    if processed is not None:
                        data.append(processed)
                except json.JSONDecodeError:
                    error_count += 1

                    if error_count <= 10:
                        if error_count == 10:
                            print(f"... (suppressing further JSON error messages)")
                except Exception as e:
                    error_count += 1
                    if error_count <= 10:
                        print(f"Line {line_num}: Error processing - {str(e)[:50]}")

        if error_count > 0:
            print(f"Skipped {error_count}/{total_lines} invalid lines")

        if data:
            print(f"Successfully loaded {len(data)} examples in JSONL format")

        return data

    def _process_example(self, item: Any) -> Optional[Dict]:

        if not isinstance(item, dict):
            return None

        try:
            processed = {}

            text_a = None
            text_b = None

            for key in [self.sentence1_key, 'text', 'text_a', 'sentence', 'premise', 'question']:
                if key in item:
                    text_a = str(item[key]).strip()
                    if text_a:
                        break

            if not text_a:
                return None

            processed['text_a'] = text_a

            if self.sentence2_key:
                for key in [self.sentence2_key, 'text_b', 'hypothesis', 'sentence2']:
                    if key in item:
                        text_b = str(item[key]).strip()
                        if text_b:
                            break
                processed['text_b'] = text_b
            else:
                processed['text_b'] = None

            label = None
            for key in ['label', 'labels', 'target', 'gold_label']:
                if key in item:
                    label = item[key]
                    break

            if label is not None:
                if self.is_regression:

                    try:
                        label = float(label)
                        label = max(0.0, min(5.0, label))
                        processed['label'] = label
                    except:
                        processed['label'] = 0.0
                else:

                    if isinstance(label, str):

                        if label in self.label_to_id:
                            processed['label'] = self.label_to_id[label]
                        else:

                            try:
                                processed['label'] = int(label)
                            except:

                                processed['label'] = 0
                    else:
                        try:
                            processed['label'] = int(label)
                        except:
                            processed['label'] = 0
            else:

                processed['label'] = 0 if not self.is_regression else 0.0

            return processed

        except Exception as e:
            return None

    def _print_data_stats(self):

        if not self.data:
            return

        print(f"Data statistics for {self.task_name}:")
        print(f"  Total examples: {len(self.data)}")

        sample = self.data[0]
        print(f"  Data keys: {list(sample.keys())}")

        if self.is_regression:
            labels = [d['label'] for d in self.data]
            print(f"  Label range: [{min(labels):.2f}, {max(labels):.2f}]")
            print(f"  Label mean: {np.mean(labels):.2f}")
        else:
            label_counts = {}
            for d in self.data:
                label = d['label']
                label_counts[label] = label_counts.get(label, 0) + 1
            print(f"  Label distribution: {label_counts}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        text_a = item['text_a']
        text_b = item.get('text_b', None)

        if self.augment and self.word_dropout > 0:
            if text_a:
                words = text_a.split()
                words = [w for w in words if np.random.random() > self.word_dropout]
                text_a = ' '.join(words) if words else item['text_a']

            if text_b:
                words = text_b.split()
                words = [w for w in words if np.random.random() > self.word_dropout]
                text_b = ' '.join(words) if words else item['text_b']

        if text_b is None:

            encoding = self.tokenizer(
                text_a,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
        else:

            encoding = self.tokenizer(
                text_a,
                text_b,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )

        encoding = {key: val.squeeze(0) for key, val in encoding.items()}

        if self.is_regression:
            encoding["labels"] = torch.tensor(item['label'], dtype=torch.float)
        else:
            encoding["labels"] = torch.tensor(item['label'], dtype=torch.long)

        return encoding


def get_compute_metrics(task_name: str):

    task_name = task_name.lower()

    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids

        if task_name == "stsb":

            predictions = predictions[:, 0] if len(predictions.shape) > 1 else predictions
            pearson_corr = float(pearsonr(predictions, labels)[0])
            spearman_corr = float(spearmanr(predictions, labels)[0])
            return {
                "eval_pearson": pearson_corr,
                "eval_spearmanr": spearman_corr,
                "eval_corr": (pearson_corr + spearman_corr) / 2,
            }

        elif task_name == "cola":

            predictions = np.argmax(predictions, axis=1)
            mcc = float(matthews_corrcoef(labels, predictions))
            acc = float((predictions == labels).mean())
            return {
                "eval_matthews_correlation": mcc,
                "eval_mcc": mcc,
                "eval_accuracy": acc,
            }

        elif task_name in ["mrpc", "qqp"]:

            predictions = np.argmax(predictions, axis=1)
            acc = float((predictions == labels).mean())
            f1 = float(f1_score(labels, predictions, average='binary'))
            return {
                "eval_accuracy": acc,
                "eval_f1": f1,
                "eval_combined": (acc + f1) / 2,
            }

        else:

            predictions = np.argmax(predictions, axis=1)
            acc = float((predictions == labels).mean())
            return {"eval_accuracy": acc}

    return compute_metrics


def find_data_files(data_dir: str, task_name: str) -> Tuple[str, str, Optional[str]]:

    task_name = task_name.lower()
    task_dir = os.path.join(data_dir, task_name)

    if not os.path.exists(task_dir):
        for name in [task_name.upper(), task_name.capitalize()]:
            candidate = os.path.join(data_dir, name)
            if os.path.exists(candidate):
                task_dir = candidate
                break

    if not os.path.exists(task_dir):
        raise ValueError(f"Task directory not found: {task_dir}")

    train_file = None
    for name in ["train.json", "train.jsonl", "train.txt", "train.tsv"]:
        candidate = os.path.join(task_dir, name)
        if os.path.exists(candidate):
            train_file = candidate
            break

    if train_file is None:
        raise ValueError(f"Training file not found in {task_dir}")

    eval_file = None
    for name in ["validation.json", "dev.json", "valid.json",
                 "validation.jsonl", "dev.jsonl", "valid.jsonl",
                 "validation.txt", "dev.txt", "dev.tsv"]:
        candidate = os.path.join(task_dir, name)
        if os.path.exists(candidate):
            eval_file = candidate
            break

    if task_name == "mnli" and eval_file is None:
        for name in ["validation_matched.json", "dev_matched.json",
                     "validation_matched.jsonl", "dev_matched.jsonl"]:
            candidate = os.path.join(task_dir, name)
            if os.path.exists(candidate):
                eval_file = candidate
                break

    if eval_file is None:
        raise ValueError(f"Validation file not found in {task_dir}")

    test_file = None
    for name in ["test.json", "test.jsonl", "test.txt", "test.tsv"]:
        candidate = os.path.join(task_dir, name)
        if os.path.exists(candidate):
            test_file = candidate
            break

    print(f"\nData files for {task_name}:")
    print(f"  Train: {os.path.basename(train_file)}")
    print(f"  Eval:  {os.path.basename(eval_file)}")
    if test_file:
        print(f"  Test:  {os.path.basename(test_file)}")

    return train_file, eval_file, test_file


def main():
    parser = argparse.ArgumentParser(description="BIM-LoRA for GLUE Tasks")

    parser.add_argument("--task_name", type=str, required=True,
                        choices=["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"],
                        help="The name of the GLUE task")

    parser.add_argument("--model_name_or_path", type=str, default="models/roberta-base",
                        help="Path to pretrained model")
    parser.add_argument("--data_dir", type=str, default="data/glue",
                        help="The input data directory")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="The output directory")

    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--max_seq_length", type=int, default=None)

    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--target_sparsity", type=float, default=None)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--collection_steps", type=int, default=None)
    parser.add_argument("--mlp_epochs", type=int, default=None)
    parser.add_argument("--mask_temperature", type=float, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--skip_visualization", action="store_true")
    parser.add_argument("--fast_dev_run", type=int, default=0)

    args = parser.parse_args()

    args.task_name = args.task_name.lower()

    config = get_default_config(args.task_name)

    for key, value in vars(args).items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)

    config.task_name = args.task_name
    config.model_name_or_path = args.model_name_or_path
    config.data_dir = args.data_dir
    config.seed = args.seed
    config.fp16 = args.fp16
    config.local_rank = args.local_rank
    config.fast_dev_run = args.fast_dev_run
    config.skip_visualization = args.skip_visualization

    config.output_dir = os.path.join(
        args.output_dir,
        f"{args.task_name}_lora{config.lora_r}_sparse{config.target_sparsity}"
    )

    config.features_save_path = os.path.join(config.output_dir, "features", "features.pt")
    config.mlp_save_path = os.path.join(config.output_dir, "features", "mlp_init.pt")
    config.logging_dir = os.path.join(config.output_dir, "logs")

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config.features_save_path), exist_ok=True)
    os.makedirs(config.logging_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("BIM-LoRA Training for GLUE")
    print("=" * 70)
    print(f"Task: {config.task_name.upper()}")
    print(f"Model: {config.model_name_or_path}")
    print(f"Output: {config.output_dir}")
    print(f"Device: {config.device}")
    print("-" * 70)

    set_seed(config.seed)

    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        train_file, eval_file, test_file = find_data_files(config.data_dir, config.task_name)
    except Exception as e:
        print(f"Error finding data files: {e}")
        return 1

    print(f"\nLoading datasets...")
    try:
        train_dataset = GLUEDataset(train_file, tokenizer, config, augment=True)
        eval_dataset = GLUEDataset(eval_file, tokenizer, config, augment=False)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        import traceback
        traceback.print_exc()
        return 1

    num_labels_map = {
        "stsb": 1,
        "mnli": 3,
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "qnli": 2,
        "rte": 2,
        "wnli": 2
    }
    num_labels = num_labels_map[config.task_name]

    print(f"\nLoading model...")
    try:
        model_config = AutoConfig.from_pretrained(
            config.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=config.task_name
        )

        if config.task_name == "stsb":
            model_config.problem_type = "regression"
        else:
            model_config.problem_type = "single_label_classification"

        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name_or_path,
            config=model_config
        )

        print("\nConverting to BIM-LoRA...")
        model = get_bim_lora_model(model, config=config)

    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return 1

    compute_metrics = get_compute_metrics(config.task_name)

    print("\nInitializing trainer...")
    try:
        trainer = BIMLoRATrainer(
            config=config,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
    except Exception as e:
        print(f"Error creating trainer: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print(f"\n{'=' * 70}")
    print(f"Starting training for {config.task_name.upper()}")
    print(f"{'=' * 70}\n")

    try:
        trainer.train()

        print("\n" + "=" * 70)
        print("Training Complete - Final Evaluation")
        print("=" * 70)

        final_results = trainer.evaluate()

        print(f"\nFinal Results for {config.task_name.upper()}:")
        for key, value in sorted(final_results.items()):
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")

        results_file = os.path.join(config.output_dir, "final_results.json")
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)

        print(f"\nResults saved to {results_file}")

        if hasattr(trainer, 'best_metric') and trainer.best_metric is not None:
            print(f"Best performance: {trainer.best_metric:.4f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if config.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()

    print("\n" + "=" * 70)
    print("SUCCESS: Training completed!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())