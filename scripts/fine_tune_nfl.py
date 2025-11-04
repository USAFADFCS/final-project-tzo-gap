from datasets import load_dataset

# Load CSV
dataset = load_dataset("csv", data_files="data/nfl_savant.csv")
print(dataset)
