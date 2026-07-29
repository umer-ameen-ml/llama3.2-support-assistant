"""
train.py
--------
Fine-tunes meta-llama/Llama-3.2-3B on a customer-support instruction
dataset using 4-bit QLoRA (LoRA on q_proj, v_proj).

Usage:
    export HF_TOKEN="your_huggingface_token"
    python train.py --dataset_path path/to/hybrid_oasst_cs_8200.json

Requirements: see requirements.txt
"""

import os
import argparse

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3.2 with QLoRA")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--dataset_path", type=str, required=True,
                         help="Path to the training JSON dataset")
    parser.add_argument("--output_dir", type=str, default="./llama3.2-finetuned-final1",
                         help="Where to save the final LoRA adapter + tokenizer")
    parser.add_argument("--checkpoints_dir", type=str, default="./results",
                         help="Where trainer checkpoints are written during training")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=512)
    return parser.parse_args()


def load_base_model_and_tokenizer(model_name: str, hf_token: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": torch.cuda.current_device()},
        torch_dtype=torch.bfloat16,
        token=hf_token,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    print("Model loaded successfully")
    print("Device:", model.device)
    return model, tokenizer


def formatting(example):
    """Builds the instruction/context/response prompt template used both
    for training and inference. Keep this identical in inference.py."""
    prompt = f"""### Instruction:
{example['instruction']}

### Context:
{example['context']}

### Response:
"""
    return {"prompt": prompt, "completion": example["response"]}


def build_dataset(dataset_path: str):
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.map(formatting, remove_columns=dataset.column_names)
    print(dataset)
    print(dataset[0])
    return dataset


def apply_lora(model):
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    return model


def train(args):
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN environment variable not set. "
            "Run: export HF_TOKEN='your_huggingface_token'"
        )

    model, tokenizer = load_base_model_and_tokenizer(args.model_name, hf_token)
    dataset = build_dataset(args.dataset_path)

    # Data collator is defined for completeness / future custom loops;
    # SFTTrainer below handles batching internally.
    _ = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model = apply_lora(model)

    training_args = SFTConfig(
        output_dir=args.checkpoints_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=500,
        fp16=False,
        bf16=True,
        max_grad_norm=0.0,
        max_length=args.max_length,
        report_to="none",
        loss_type="nll",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved fine-tuned adapter + tokenizer to {args.output_dir}")


if __name__ == "__main__":
    train(parse_args())
