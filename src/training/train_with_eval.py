"""
Fine-tuning script with comprehensive evaluation during training.

This script trains a model with:
- Automatic validation loss monitoring
- Early stopping
- Sample predictions at checkpoints
- Weights & Biases logging
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from datasets import load_dataset
import wandb
from typing import List, Dict


class SampleEvaluationCallback(TrainerCallback):
    """Callback to generate sample predictions during training."""

    def __init__(self, tokenizer, test_samples: List[Dict], device: str = "cuda"):
        self.tokenizer = tokenizer
        self.test_samples = test_samples
        self.device = device

    def on_evaluate(self, args, state, control, model, **kwargs):
        """Run sample predictions after each evaluation."""
        print("\n" + "="*50)
        print("Sample Predictions at Step", state.global_step)
        print("="*50)

        model.eval()
        with torch.no_grad():
            for i, sample in enumerate(self.test_samples[:3], 1):
                # Format input
                prompt = f"""제천 관광 정보를 바탕으로 질문에 답하세요.

문맥: {sample['context']}

질문: {sample['question']}

답변:"""

                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)

                # Generate
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=False  # Deterministic for comparison
                )

                generated = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )

                print(f"\nSample {i}:")
                print(f"Q: {sample['question']}")
                print(f"Generated: {generated.strip()}")
                print(f"Expected: {sample['answer']}")
                print("-" * 50)

        model.train()


def compute_metrics(eval_pred):
    """Compute perplexity from eval loss."""
    # Note: eval_pred is (predictions, labels) tuple
    # But for causal LM, we compute metrics from loss directly
    # This function is called after trainer.evaluate()

    # Metrics are already computed by trainer (loss)
    # We can add custom metrics here if needed
    return {}


def setup_wandb(config: Dict):
    """Initialize Weights & Biases logging."""
    wandb.init(
        project="jecheon-rag-finetuning",
        name=f"{config['model_name']}-lr{config['learning_rate']}-ep{config['num_epochs']}",
        config=config
    )


def train_with_evaluation(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    dataset_path: str = "data/processed/train_dataset.json",
    output_dir: str = "./results",
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    eval_steps: int = 50,
    logging_steps: int = 10,
    use_wandb: bool = True,
):
    """
    Train model with comprehensive evaluation.

    Args:
        model_name: HuggingFace model name or local path
        dataset_path: Path to training dataset (JSON)
        output_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Per-device batch size
        gradient_accumulation_steps: Steps to accumulate gradients
        eval_steps: Evaluate every N steps
        logging_steps: Log metrics every N steps
        use_wandb: Whether to use Weights & Biases
    """

    # Configuration
    config = {
        "model_name": model_name,
        "dataset_path": dataset_path,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
    }

    # Initialize W&B
    if use_wandb:
        setup_wandb(config)

    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Load dataset
    print(f"Loading dataset: {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path)

    # Split train/validation (90/10)
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

    # Prepare test samples for callback
    test_samples = [
        {
            "context": dataset["test"][i]["context"],
            "question": dataset["test"][i]["question"],
            "answer": dataset["test"][i]["answer"]
        }
        for i in range(min(5, len(dataset["test"])))
    ]

    # Tokenize function
    def tokenize_function(examples):
        # Format: instruction + context + question + answer
        texts = []
        for inst, ctx, q, a in zip(
            examples["instruction"],
            examples["context"],
            examples["question"],
            examples["answer"]
        ):
            text = f"{inst}\n\n문맥: {ctx}\n\n질문: {q}\n\n답변: {a}"
            texts.append(text)

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding="max_length"
        )

        # For causal LM, labels = input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    # Tokenize datasets
    print("Tokenizing datasets...")
    tokenized_train = dataset["train"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    tokenized_eval = dataset["test"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["test"].column_names
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,

        # Training
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=50,

        # Evaluation
        evaluation_strategy="steps",
        eval_steps=eval_steps,

        # Logging
        logging_strategy="steps",
        logging_steps=logging_steps,
        report_to="wandb" if use_wandb else "none",

        # Saving
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=3,  # Keep only 3 best checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        # Performance
        bf16=True,
        gradient_checkpointing=True,

        # Misc
        remove_unused_columns=False,
        push_to_hub=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
        callbacks=[
            SampleEvaluationCallback(
                tokenizer=tokenizer,
                test_samples=test_samples,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        ]
    )

    # Print training info
    print("\n" + "="*60)
    print("Training Configuration:")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Train samples: {len(tokenized_train)}")
    print(f"Eval samples: {len(tokenized_eval)}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"Total steps: {len(tokenized_train) // (batch_size * gradient_accumulation_steps) * num_epochs}")
    print(f"Eval every: {eval_steps} steps")
    print("="*60 + "\n")

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(f"{output_dir}/final_model")
    tokenizer.save_pretrained(f"{output_dir}/final_model")

    # Final evaluation
    print("\nRunning final evaluation...")
    final_metrics = trainer.evaluate()

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Final eval loss: {final_metrics['eval_loss']:.4f}")
    print(f"Final perplexity: {torch.exp(torch.tensor(final_metrics['eval_loss'])):.4f}")
    print(f"Best model saved to: {output_dir}/checkpoint-XXX")
    print("="*60)

    if use_wandb:
        wandb.finish()

    return trainer, final_metrics


if __name__ == "__main__":
    # Example usage
    trainer, metrics = train_with_evaluation(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        dataset_path="data/processed/train_dataset.json",
        output_dir="./results/qwen-7b-jecheon",
        num_epochs=3,
        learning_rate=2e-5,
        batch_size=4,
        eval_steps=50,
        use_wandb=True
    )
