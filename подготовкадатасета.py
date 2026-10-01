from datasets import load_dataset

train_ds = load_dataset("json", data_files="/kaggle/input/qwends/train.jsonl", split="train")
val_ds   = load_dataset("json", data_files="/kaggle/input/qwenval/val.jsonl", split="train")

def to_text(ex):
    return {"text": tokenizer.apply_chat_template(
        ex["messages"], tokenize=False,
        add_generation_prompt=False, enable_thinking=False)}

train_ds = train_ds.map(to_text)
val_ds   = val_ds.map(to_text)
print(len(train_ds), len(val_ds))
print(train_ds[0]["text"][:700])
