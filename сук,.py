import os

print("--- Содержимое папки input ---")
if os.path.exists("/kaggle/input"):
    for root, dirs, files in os.walk("/kaggle/input"):
        for file in files:
            if file.endswith(".jsonl"):
                print(f"Найден файл: {os.path.join(root, file)}")
else:
    print("Папка /kaggle/input вообще пуста или не подключена!")
