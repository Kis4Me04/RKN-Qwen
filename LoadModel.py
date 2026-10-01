import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # используем одну карту

import torch
from unsloth import FastLanguageModel

max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    dtype = None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)
