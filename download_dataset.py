from datasets import load_dataset
import pandas as pd

dataset = load_dataset("dair-ai/emotion")

train = pd.DataFrame(dataset["train"])
validation = pd.DataFrame(dataset["validation"])
test = pd.DataFrame(dataset["test"])

train.to_csv("data/train.csv", index=False)
validation.to_csv("data/validation.csv", index=False)
test.to_csv("data/test.csv", index=False)

print("Dataset downloaded successfully!")
print("Training:", len(train))
print("Validation:", len(validation))
print("Testing:", len(test))