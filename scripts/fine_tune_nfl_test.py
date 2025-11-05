from datasets import load_dataset
from transformers import AutoTokenizer

# Load CSV
dataset = load_dataset("csv", data_files="data/nfl_savant.csv")
print("Dataset loaded!")

# Initialize Qwen tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

# Tokenize only the 'Description' column
def tokenize(example):
    return tokenizer(example["Description"], truncation=True, padding="max_length", max_length=512)

# Optional: Use a small subset to test quickly
small_dataset = dataset["train"].select(range(500))  # first 500 rows
tokenized_dataset = small_dataset.map(tokenize, batched=True)

print("Tokenization complete!")
print(tokenized_dataset)
