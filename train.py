import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

MODEL = "unsloth/Qwen3-8B"
OUT = "output/qwen3-8b-law-unsloth"
MAX_SEQ_LENGTH = 1024

if not torch.cuda.is_available():
    raise RuntimeError("CUDA недоступна")

dataset = load_dataset(
    "json",
    data_files={
        "train": "data/train.jsonl",
        "validation": "data/val.jsonl",
    },
)

# Проверяем формат диалогов.
for split in ("train", "validation"):
    for item in dataset[split]:
        messages = item["messages"]
        assert messages[-1]["role"] == "assistant"
        assert all(isinstance(m["content"], str) for m in messages)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
)

args = SFTConfig(
    output_dir=OUT,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    max_length=MAX_SEQ_LENGTH,
    bf16=True,
    optim="adamw_8bit",
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    assistant_only_loss=True,
    report_to="none",
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model(OUT)
tokenizer.save_pretrained(OUT)

print("Адаптер сохранён в:", OUT)
