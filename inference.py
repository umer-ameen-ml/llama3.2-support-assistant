"""
inference.py
------------
Multi-turn, ChatGPT-style CLI inference for the fine-tuned
customer-support model (base: meta-llama/Llama-3.2-3B,
adapter: local LoRA folder produced by train.py).

Usage:
    export HF_TOKEN="your_huggingface_token"
    python inference.py --adapter_path path/to/llama3.2-finetuned-final1

Behavior:
    - Har turn pe user apna message type karta hai (jaise ChatGPT).
    - Purani conversation "context" ke through model ko yaad rehti hai,
      taake multi-turn chat natural lage.
    - "exit" ya "quit" type karke conversation khatam ki ja sakti hai.
    - "reset" type karne se conversation history clear ho jati hai.

Prompt format train.py's formatting() jaisa hi hai:

    ### Instruction:
    {instruction}

    ### Context:
    {context}

    ### Response:
"""

from __future__ import annotations

import os
import argparse
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
from peft import PeftModel

logger = logging.getLogger("inference")
logging.basicConfig(level=logging.WARNING)

# Previous (user, assistant) turns kept as rolling "context".
# Kept small so the prompt doesn't exceed the model's training max_length (512).
MAX_HISTORY_TURNS = 4

EXIT_COMMANDS = {"exit", "quit", "q"}
RESET_COMMANDS = {"reset", "clear"}

# Training data's "response" fields contained transcript-style markers
# (Customer:/Agent:, ###, Reference #), so the model can learn to keep
# generating a fake conversation. These stop sequences cut it off as
# soon as it tries to start a new fake turn.
STOP_SEQUENCES = [
    "\nYou:",
    "\nUser:",
    "\nCustomer:",
    "\nAssistant:",
    "\n###",
    "\n##",
    "\nReference",
]


class StopOnSequences(StoppingCriteria):
    """Stops generation as soon as any stop_sequence appears in the newly
    generated text (checked on decoded text, since a phrase can span
    multiple tokens)."""

    def __init__(self, tokenizer, prompt_len_tokens: int, stop_sequences: List[str]):
        self.tokenizer = tokenizer
        self.prompt_len_tokens = prompt_len_tokens
        self.stop_sequences = stop_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated_ids = input_ids[0][self.prompt_len_tokens:]
        if generated_ids.numel() == 0:
            return False
        tail_ids = generated_ids[-20:]
        tail_text = self.tokenizer.decode(tail_ids, skip_special_tokens=True)
        return any(stop in tail_text for stop in self.stop_sequences)


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.1


class InferenceEngine:
    """Loads the base model + LoRA adapter once, then serves multi-turn
    chat generations. Conversation history is kept in-memory per session."""

    def __init__(self, base_model_name: str, adapter_path: str, hf_token: Optional[str]):
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.hf_token = hf_token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        print(f"Loading tokenizer from {self.adapter_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.adapter_path,
            token=self.hf_token,
            clean_up_tokenization_spaces=False,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading base model {self.base_model_name} on device={self.device} ...")

        if self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                token=self.hf_token,
                quantization_config=bnb_config,
                device_map={"": torch.cuda.current_device()},
            )
        else:
            print("CUDA not available — loading in full precision on CPU (slower).")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                token=self.hf_token,
                torch_dtype=torch.float32,
                device_map={"": "cpu"},
            )

        base_model.resize_token_embeddings(len(self.tokenizer))
        base_model.config.pad_token_id = self.tokenizer.pad_token_id

        print(f"Attaching LoRA adapter from {self.adapter_path} ...")
        self.model = PeftModel.from_pretrained(
            base_model, self.adapter_path, token=self.hf_token,
        )
        self.model.eval()
        print("Model ready.\n")

    @staticmethod
    def _history_to_context(history: List[Tuple[str, str]]) -> str:
        if not history:
            return ""
        recent = history[-MAX_HISTORY_TURNS:]
        lines = []
        for user_msg, assistant_msg in recent:
            lines.append(f"User: {user_msg}")
            lines.append(f"Assistant: {assistant_msg}")
        return "\n".join(lines)

    def build_prompt(self, instruction: str, history: List[Tuple[str, str]]) -> str:
        context = self._history_to_context(history)
        return f"""### Instruction:
{instruction}

### Context:
{context}

### Response:
"""

    def generate(
        self,
        instruction: str,
        history: List[Tuple[str, str]],
        gen_config: Optional[GenerationConfig] = None,
    ) -> str:
        if not instruction or not instruction.strip():
            raise ValueError("Message cannot be empty.")

        gen_config = gen_config or GenerationConfig()
        prompt = self.build_prompt(instruction, history)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len_tokens = inputs["input_ids"].shape[1]

        stopping_criteria = StoppingCriteriaList(
            [StopOnSequences(self.tokenizer, prompt_len_tokens, STOP_SEQUENCES)]
        )

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=gen_config.max_new_tokens,
                temperature=gen_config.temperature,
                top_p=gen_config.top_p,
                do_sample=gen_config.do_sample,
                repetition_penalty=gen_config.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
            )

        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        if full_text.startswith(prompt):
            response = full_text[len(prompt):].strip()
        else:
            response = full_text.strip()

        response = self._truncate_at_stop_sequence(response)
        return response

    @staticmethod
    def _truncate_at_stop_sequence(text: str) -> str:
        cut_points = [text.find(stop) for stop in STOP_SEQUENCES if text.find(stop) != -1]
        if cut_points:
            text = text[: min(cut_points)]
        return text.strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with the fine-tuned model")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--adapter_path", type=str, required=True,
                         help="Path (local folder or HF repo id) of the LoRA adapter")
    return parser.parse_args()


def run_chat(args) -> None:
    hf_token = os.environ.get("HF_TOKEN")
    engine = InferenceEngine(args.base_model, args.adapter_path, hf_token)
    history: List[Tuple[str, str]] = []

    print("Customer Support Assistant — type 'exit' to quit, 'reset' to clear chat.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAssistant: Chat ended. Take care!")
            break

        if not user_input:
            continue

        if user_input.lower() in EXIT_COMMANDS:
            print("Assistant: Chat ended. Take care!")
            break

        if user_input.lower() in RESET_COMMANDS:
            history.clear()
            print("Assistant: Conversation reset.\n")
            continue

        try:
            response = engine.generate(instruction=user_input, history=history)
        except ValueError as exc:
            print(f"Assistant: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"Assistant: Sorry, something went wrong ({exc}). Please try again.")
            continue

        print(f"Assistant: {response}\n")
        history.append((user_input, response))


if __name__ == "__main__":
    run_chat(parse_args())
