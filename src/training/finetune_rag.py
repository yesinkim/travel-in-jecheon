"""
Finetune-RAG Training Script for KANANA1.5-8B

Based on the paper: "Finetune-RAG: Fine-Tuning Language Models to Resist
Hallucination in Retrieval-Augmented Generation"

This script fine-tunes KANANA1.5-8B using QLoRA for efficient training on
the Jecheon tourism RAG dataset.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import torch
from dataclasses import dataclass, field

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Transformers and training libraries
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_from_disk, Dataset
import wandb


@dataclass
class ModelConfig:
    """Model configuration"""
    name: str
    type: str = "causal_lm"
    trust_remote_code: bool = False
    use_flash_attention_2: bool = True
    torch_dtype: str = "bfloat16"


@dataclass
class QLoRAConfig:
    """QLoRA configuration"""
    enabled: bool = True
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


class FinetuneRAGTrainer:
    """
    Trainer for Finetune-RAG methodology
    """

    def __init__(self, config_path: str):
        """
        Initialize trainer with config file

        Args:
            config_path: Path to YAML config file
        """
        self.config = self.load_config(config_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.trainer = None

        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config

    def setup_model_and_tokenizer(self):
        """
        Setup model and tokenizer with QLoRA configuration
        """
        model_config = self.config["model"]
        qlora_config = self.config["qlora"]

        print(f"\n=== Loading Model: {model_config['name']} ===")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config["name"],
            trust_remote_code=model_config["trust_remote_code"],
        )

        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Set padding side to right for training
        self.tokenizer.padding_side = "right"

        print(f"Tokenizer loaded: {len(self.tokenizer)} tokens")
        print(f"PAD token: {self.tokenizer.pad_token}")
        print(f"EOS token: {self.tokenizer.eos_token}")

        # Configure quantization
        if qlora_config["enabled"]:
            print("\n=== Configuring QLoRA (4-bit quantization) ===")

            # Get dtype
            compute_dtype = getattr(torch, qlora_config["bnb_4bit_compute_dtype"])

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=qlora_config["load_in_4bit"],
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=qlora_config["bnb_4bit_quant_type"],
                bnb_4bit_use_double_quant=qlora_config["bnb_4bit_use_double_quant"],
            )

            print(f"Quantization type: {qlora_config['bnb_4bit_quant_type']}")
            print(f"Compute dtype: {compute_dtype}")
            print(f"Double quantization: {qlora_config['bnb_4bit_use_double_quant']}")
        else:
            bnb_config = None

        # Load model
        model_dtype = getattr(torch, model_config["torch_dtype"])

        load_kwargs = {
            "pretrained_model_name_or_path": model_config["name"],
            "quantization_config": bnb_config,
            "device_map": self.config["hardware"]["device_map"],
            "trust_remote_code": model_config["trust_remote_code"],
            "torch_dtype": model_dtype,
        }

        # Add Flash Attention 2 if supported
        if model_config.get("use_flash_attention_2", False):
            try:
                load_kwargs["attn_implementation"] = "flash_attention_2"
                print("Flash Attention 2: Enabled")
            except Exception as e:
                print(f"Flash Attention 2 not available: {e}")

        self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

        # Prepare model for k-bit training
        if qlora_config["enabled"]:
            self.model = prepare_model_for_kbit_training(self.model)

        # Configure LoRA
        if qlora_config["enabled"]:
            print("\n=== Configuring LoRA ===")

            lora_config = LoraConfig(
                r=qlora_config["lora_r"],
                lora_alpha=qlora_config["lora_alpha"],
                lora_dropout=qlora_config["lora_dropout"],
                target_modules=qlora_config["target_modules"],
                bias=qlora_config["bias"],
                task_type=qlora_config["task_type"],
            )

            print(f"LoRA rank (r): {qlora_config['lora_r']}")
            print(f"LoRA alpha: {qlora_config['lora_alpha']}")
            print(f"LoRA dropout: {qlora_config['lora_dropout']}")
            print(f"Target modules: {qlora_config['target_modules']}")

            self.model = get_peft_model(self.model, lora_config)

            # Print trainable parameters
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")

        print("\n=== Model Setup Complete ===\n")

    def load_dataset(self) -> tuple:
        """
        Load training and evaluation datasets

        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        dataset_config = self.config["dataset"]
        dataset_path = dataset_config["train_dataset_path"]

        print(f"\n=== Loading Dataset: {dataset_path} ===")

        # Load dataset
        dataset = load_from_disk(dataset_path)

        print(f"Dataset loaded:")
        print(f"  Splits: {list(dataset.keys())}")
        for split_name, split_data in dataset.items():
            print(f"  {split_name}: {len(split_data)} examples")

        # Get train and eval datasets
        train_dataset = dataset["train"]
        eval_dataset = dataset.get("test", None)

        # Sample example
        print(f"\n=== Sample Training Example ===")
        sample = train_dataset[0]
        if "messages" in sample:
            for msg in sample["messages"]:
                print(f"[{msg['role']}]:")
                print(f"{msg['content'][:200]}...")
                print()

        return train_dataset, eval_dataset

    def setup_training_args(self) -> TrainingArguments:
        """
        Setup training arguments

        Returns:
            TrainingArguments object
        """
        train_config = self.config["training"]

        # Setup W&B if enabled
        if self.config["wandb"]["enabled"]:
            wandb.init(
                project=self.config["wandb"]["project"],
                name=self.config["wandb"]["name"],
                entity=self.config["wandb"].get("entity"),
                tags=self.config["wandb"].get("tags", []),
                notes=self.config["wandb"].get("notes", ""),
                config=self.config,
            )

        training_args = TrainingArguments(
            output_dir=train_config["output_dir"],
            num_train_epochs=train_config["num_train_epochs"],
            per_device_train_batch_size=train_config["per_device_train_batch_size"],
            per_device_eval_batch_size=train_config["per_device_eval_batch_size"],
            gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
            learning_rate=train_config["learning_rate"],
            warmup_ratio=train_config["warmup_ratio"],
            lr_scheduler_type=train_config["lr_scheduler_type"],
            optim=train_config["optim"],
            weight_decay=train_config["weight_decay"],
            max_grad_norm=train_config["max_grad_norm"],
            bf16=train_config["bf16"],
            fp16=train_config["fp16"],
            logging_steps=train_config["logging_steps"],
            logging_first_step=train_config["logging_first_step"],
            evaluation_strategy=train_config["evaluation_strategy"],
            eval_steps=train_config["eval_steps"],
            save_strategy=train_config["save_strategy"],
            save_steps=train_config["save_steps"],
            save_total_limit=train_config["save_total_limit"],
            load_best_model_at_end=train_config["load_best_model_at_end"],
            metric_for_best_model=train_config["metric_for_best_model"],
            greater_is_better=train_config["greater_is_better"],
            seed=train_config["seed"],
            dataloader_num_workers=train_config["dataloader_num_workers"],
            remove_unused_columns=train_config["remove_unused_columns"],
            group_by_length=train_config["group_by_length"],
            gradient_checkpointing=train_config["gradient_checkpointing"],
            report_to=["wandb"] if self.config["wandb"]["enabled"] else [],
            run_name=self.config["wandb"]["name"] if self.config["wandb"]["enabled"] else None,
        )

        # Add gradient checkpointing kwargs if specified
        if train_config.get("gradient_checkpointing_kwargs"):
            training_args.gradient_checkpointing_kwargs = train_config["gradient_checkpointing_kwargs"]

        return training_args

    def formatting_func(self, example):
        """
        Format examples for training using chat template

        Args:
            example: Dataset example with 'messages' field

        Returns:
            Formatted text
        """
        if "messages" in example:
            # Use tokenizer's chat template
            text = self.tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            text = example["text"]

        return text

    def train(self):
        """
        Main training function
        """
        # Setup model and tokenizer
        self.setup_model_and_tokenizer()

        # Load datasets
        train_dataset, eval_dataset = self.load_dataset()

        # Setup training arguments
        training_args = self.setup_training_args()

        print(f"\n=== Training Configuration ===")
        print(f"Output dir: {training_args.output_dir}")
        print(f"Epochs: {training_args.num_train_epochs}")
        print(f"Batch size: {training_args.per_device_train_batch_size}")
        print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
        print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
        print(f"Learning rate: {training_args.learning_rate}")
        print(f"LR scheduler: {training_args.lr_scheduler_type}")
        print(f"Optimizer: {training_args.optim}")
        print(f"Max sequence length: {self.config['dataset']['max_seq_length']}")

        # Create trainer
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            formatting_func=self.formatting_func,
            max_seq_length=self.config["dataset"]["max_seq_length"],
            packing=False,  # Don't pack multiple examples into one sequence
        )

        # Train
        print(f"\n=== Starting Training ===\n")
        self.trainer.train()

        # Save final model
        print(f"\n=== Saving Final Model ===")
        self.trainer.save_model(training_args.output_dir)
        self.tokenizer.save_pretrained(training_args.output_dir)

        print(f"\nModel saved to: {training_args.output_dir}")

        # Push to HuggingFace Hub if configured
        if self.config["hub"].get("push_to_hub", False):
            print(f"\n=== Pushing to HuggingFace Hub ===")
            self.trainer.push_to_hub(
                commit_message="Fine-tuned KANANA1.5-8B with Finetune-RAG methodology"
            )

        print("\n=== Training Complete ===")

    def evaluate(self):
        """
        Evaluate the trained model
        """
        if self.trainer is None:
            raise ValueError("Model not trained yet. Call train() first.")

        print(f"\n=== Evaluating Model ===")
        metrics = self.trainer.evaluate()

        print(f"\nEvaluation Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

        return metrics

    def test_inference(self, num_examples: int = 3):
        """
        Test inference with sample examples

        Args:
            num_examples: Number of examples to test
        """
        print(f"\n=== Testing Inference ===")

        # Get eval prompts from config
        eval_prompts = self.config["finetune_rag"].get("eval_prompts", [])

        if not eval_prompts:
            print("No eval prompts configured in config file")
            return

        inference_config = self.config["inference"]

        for i, prompt_config in enumerate(eval_prompts[:num_examples]):
            print(f"\n--- Example {i+1}/{num_examples} ---")
            print(f"Question: {prompt_config['question']}")
            print(f"Context: {prompt_config['context'][:100]}...")
            print(f"Expected type: {prompt_config['expected_type']}")

            # Format input
            messages = [
                {"role": "system", "content": self.config["finetune_rag"]["system_prompt"]},
                {"role": "user", "content": f"""<document>
<source>제천시 관광정보</source>
<context>
{prompt_config['context']}
</context>
</document>

<question>{prompt_config['question']}</question>

<answer>"""}
            ]

            # Tokenize
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=inference_config["max_new_tokens"],
                    temperature=inference_config["temperature"],
                    top_p=inference_config["top_p"],
                    top_k=inference_config["top_k"],
                    repetition_penalty=inference_config["repetition_penalty"],
                    do_sample=inference_config["do_sample"],
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            print(f"\nGenerated Answer:")
            print(generated_text)
            print("-" * 80)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Finetune-RAG Training for KANANA1.5-8B")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/finetune_rag_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only run evaluation (requires trained model)"
    )
    parser.add_argument(
        "--test_inference",
        action="store_true",
        help="Test inference with sample examples"
    )

    args = parser.parse_args()

    # Create trainer
    trainer = FinetuneRAGTrainer(args.config)

    # Run training or evaluation
    if args.eval_only:
        trainer.setup_model_and_tokenizer()
        trainer.evaluate()
    elif args.test_inference:
        trainer.setup_model_and_tokenizer()
        trainer.test_inference()
    else:
        trainer.train()
        trainer.evaluate()
        trainer.test_inference()


if __name__ == "__main__":
    main()
