"""
06_train.py — QLoRA fine-tuning of Qwen3-Coder-30B-A3B-Instruct on H100 80 GB.

Called by scripts/06_train.sh; can also be run directly.
Reads data/{train,valid}.jsonl (ChatML messages format produced by 05_prepare.py).

Dependencies:
    pip install torch transformers peft trl bitsandbytes accelerate flash-attn
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


def load_jsonl(path: Path) -> Dataset:
    records = [json.loads(l) for l in path.open() if l.strip()]
    return Dataset.from_list(records)


def format_messages(example: dict, tokenizer) -> dict:
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",                     default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--data_dir",          type=Path,   default=Path("data"))
    parser.add_argument("--output_dir",        type=Path,   default=Path("models/icongenai-qlora"))
    parser.add_argument("--per_device_train_batch_size", type=int,   default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int,   default=4)
    parser.add_argument("--lora_r",            type=int,    default=64)
    parser.add_argument("--lora_alpha",        type=int,    default=128)
    parser.add_argument("--lora_dropout",      type=float,  default=0.05)
    parser.add_argument("--max_seq_length",    type=int,    default=4096)
    parser.add_argument("--num_train_epochs",  type=int,    default=3)
    parser.add_argument("--learning_rate",     type=float,  default=2e-4)
    parser.add_argument("--save_steps",        type=int,    default=500)
    parser.add_argument("--warmup_ratio",      type=float,  default=0.05)
    args = parser.parse_args()

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_ds = load_jsonl(args.data_dir / "train.jsonl")
    valid_ds = load_jsonl(args.data_dir / "valid.jsonl")

    fmt = lambda ex: format_messages(ex, tokenizer)
    train_ds = train_ds.map(fmt, remove_columns=["messages"])
    valid_ds = valid_ds.map(fmt, remove_columns=["messages"])

    # ── QLoRA: 4-bit base + LoRA adapters ────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Completion-only loss mask ─────────────────────────────────────────────
    # Mask loss on system/user tokens; only train on assistant (SVG) tokens.
    # Qwen chat template: <|im_start|>assistant\n ... <|im_end|>
    collator = DataCollatorForCompletionOnlyLM(
        instruction_template="<|im_start|>user\n",
        response_template="<|im_start|>assistant\n",
        tokenizer=tokenizer,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        tf32=True,
        max_grad_norm=1.0,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=4,
        group_by_length=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=True,
        data_collator=collator,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"\nAdapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
