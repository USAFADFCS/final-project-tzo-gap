from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# 1️⃣ Load the full CSV
dataset = load_dataset("csv", data_files="data/nfl_savant.csv")
print("Full dataset loaded!")

# 2️⃣ Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

# 3️⃣ Tokenize descriptions
def tokenize(example):
    return tokenizer(example["Description"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset["train"].map(tokenize, batched=True, batch_size=1000)
print("Tokenization complete!")

# 4️⃣ Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

# 5️⃣ Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 6️⃣ Training arguments
training_args = TrainingArguments(
    output_dir="./nfl_finetuned_full",
    overwrite_output_dir=True,
    per_device_train_batch_size=1,  # adjust if GPU allows
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    save_steps=500,
    save_total_limit=3,
    logging_steps=50,
    fp16=False,
    report_to="none",
)

# 7️⃣ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 8️⃣ Start training
trainer.train()
print("Full fine-tuning complete!")
