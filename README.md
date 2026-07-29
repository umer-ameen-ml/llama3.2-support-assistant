# Llama 3.2 — Fine-Tuned Customer Support Assistant

A QLoRA fine-tuned version of **meta-llama/Llama-3.2-3B**, trained to respond
in a customer-support assistant tone/style. Fine-tuning was done with
4-bit quantization (BitsAndBytes) and LoRA adapters on `q_proj` and `v_proj`.

## Project Structure

```
.
├── train.py              # Training script (QLoRA fine-tuning)
├── inference.py          # Multi-turn CLI chat script
├── requirements.txt      # Python dependencies
├── adapter/              # Fine-tuned LoRA adapter + tokenizer files
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer_config.json
│   └── tokenizer.json
└── README.md
```

## Model Details

- **Base model:** `meta-llama/Llama-3.2-3B`
- **Method:** QLoRA (4-bit NF4 quantization + LoRA)
- **LoRA config:** r=16, alpha=32, target_modules=["q_proj", "v_proj"], dropout=0.05
- **Training data:** Customer-support instruction dataset (`hybrid_oasst_cs_8200.json`, ~8,200 examples)
- **Epochs:** 1 (2007 steps)
- **Prompt format:**
  ```
  ### Instruction:
  {instruction}

  ### Context:
  {context}

  ### Response:
  ```

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
pip install -r requirements.txt
```

You'll need a Hugging Face access token with access to the gated
`meta-llama/Llama-3.2-3B` model:

```bash
export HF_TOKEN="your_huggingface_token"
```

## Training

```bash
python train.py --dataset_path path/to/hybrid_oasst_cs_8200.json --output_dir ./adapter
```

Optional flags: `--model_name`, `--num_train_epochs`, `--per_device_train_batch_size`,
`--learning_rate`, `--max_length`. Run `python train.py --help` for the full list.

## Inference / Chat

```bash
python inference.py --adapter_path ./adapter
```

This starts an interactive, multi-turn chat session in the terminal:

```
You: I did not receive my parcel.
Assistant: I'm sorry about that. Let's fix this together. What is your order number?
You: exit
Assistant: Chat ended. Take care!
```

Commands:
- `exit` / `quit` / `q` — end the chat
- `reset` / `clear` — clear conversation history and start fresh

## Notes & Limitations

- This model was trained for **English** customer-support conversations only;
  it does not reliably understand or generate Urdu.
- With only ~8,200 examples and 1 epoch of LoRA fine-tuning on 2 projection
  matrices, responses are generally on-tone but not perfectly accurate for
  every query type — some responses may fall back to generic base-model
  behavior for underrepresented query patterns in the training data.
- A `StoppingCriteria` guard is used during inference to stop generation as
  soon as the model tries to fabricate a new fake conversation turn
  (a pattern it picked up from transcript-style formatting in the training data).

## License

Add your preferred license here (e.g. MIT). Note that usage of the base
Llama 3.2 model is subject to Meta's Llama license terms.
