import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from trl import SFTConfig, SFTTrainer

# ==========================
# НАСТРОЙКИ
# ==========================
MODEL_NAME = "Qwen/Qwen3-8B"
OUTPUT_DIR = "/kaggle/working/qwen3-8b-law-lora"
MAX_LENGTH = 1024
EPOCHS = 3

# ==========================
# ПРОВЕРКА GPU
# ==========================
if not torch.cuda.is_available():
    raise RuntimeError(
        "GPU не включён. Открой Settings -> Accelerator -> GPU."
    )

print("GPU:", torch.cuda.get_device_name(0))
print("VRAM:", round(
    torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
), "GB")

# ==========================
# ПОИСК ДАТАСЕТА
# ==========================
def find_dataset_file(filename):
    for root in [Path("/kaggle/input"), Path("/kaggle/working")]:
        if root.exists():
            for path in root.rglob(filename):
                if path.is_file():
                    return str(path)

    raise FileNotFoundError(
        f"Не найден {filename}. Добавь Dataset через Add Input."
    )

train_file = find_dataset_file("train.jsonl")
val_file = find_dataset_file("val.jsonl")

print("Train:", train_file)
print("Validation:", val_file)

dataset = load_dataset(
    "json",
    data_files={
        "train": train_file,
        "validation": val_file,
    },
)

for split in ["train", "validation"]:
    if len(dataset[split]) == 0:
        raise ValueError(f"Датасет {split} пустой")

    for example in dataset[split]:
        assert "messages" in example
        assert example["messages"][-1]["role"] == "assistant"
        assert all(
            isinstance(message["content"], str)
            for message in example["messages"]
        )

print("Обучающих примеров:", len(dataset["train"]))
print("Проверочных примеров:", len(dataset["validation"]))

# ==========================
# ЗАГРУЗКА QWEN3-8B
# ==========================
use_bf16 = torch.cuda.is_bf16_supported()
compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    torch_dtype=compute_dtype,
    device_map={"": 0},
)

model.config.use_cache = False
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)

# ==========================
# НАСТРОЙКА LoRA
# ==========================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==========================
# ПАРАМЕТРЫ ОБУЧЕНИЯ
# ==========================
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    max_length=MAX_LENGTH,
    bf16=use_bf16,
    fp16=not use_bf16,
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
    gradient_checkpointing=True,
)

# ==========================
# ЗАПУСК ОБУЧЕНИЯ
# ==========================
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
)

trainer.train()

# ==========================
# СОХРАНЕНИЕ АДАПТЕРА
# ==========================
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nОБУЧЕНИЕ ЗАВЕРШЕНО")
print("Результат:", OUTPUT_DIR)

for path in Path(OUTPUT_DIR).iterdir():
    print(path.name)
