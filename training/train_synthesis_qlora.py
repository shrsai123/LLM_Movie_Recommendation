"""QLoRA supervised fine-tuning for grounded recommendation synthesis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "synthesis",
    )
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "recommendation-synthesis-adapter",
    )
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()

    # Training dependencies are optional for the normal FastAPI application, so
    # import them only when this training entry point is executed.
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            "Install the necessary packages"
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit("QLoRA training requires a CUDA-capable GPU")

    # TRL supports conversational prompt/completion records. Extra fields are
    # retained in the JSONL for evaluation but removed from the training view.
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(args.data_dir / "train.jsonl"),
            "validation": str(args.data_dir / "dev.jsonl"),
        },
    )

    # `reference_answer` is the single human-editable target. Rebuild completion
    # from it here so reviewers never have to update two duplicate fields.
    dataset = dataset.map(
        lambda row: {
            "completion": [
                {"role": "assistant", "content": row["reference_answer"]}
            ]
        }
    )
    keep_columns = {"prompt", "completion"}
    removable_columns = [
        name for name in dataset["train"].column_names if name not in keep_columns
    ]
    dataset = dataset.remove_columns(removable_columns)

    # NF4 stores the frozen base model in four bits. LoRA parameters remain
    # trainable, which greatly reduces the memory needed for fine-tuning.
    compute_dtype = (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
    )
    base_model = prepare_model_for_kbit_training(base_model)
    base_model.config.use_cache = False

    # `all-linear` applies LoRA to every linear transformer layer. The original
    # Gemma weights remain frozen; the output directory stores only the adapter.
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    training_config = SFTConfig(
        output_dir=str(args.out),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        max_length=2048,
        # Only assistant completion tokens contribute to the supervised loss.
        completion_only_loss=True,
        packing=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=5,
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        report_to="none",
        seed=17,
    )

    trainer = SFTTrainer(
        model=base_model,
        args=training_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    result = trainer.train()
    trainer.save_model(str(args.out))
    tokenizer.save_pretrained(args.out)

    # Save scalar training metrics alongside the adapter for reproducibility.
    metrics = {
        key: float(value)
        for key, value in result.metrics.items()
        if isinstance(value, (int, float))
    }
    (args.out / "train_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"Saved synthesis adapter to {args.out}")


if __name__ == "__main__":
    main()
