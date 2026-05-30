# Example of how to implement

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

model_path = "YOUR MODEL PATH INCLUDING ALL FILES LISTED ABOVE"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

clf = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    truncation=True,
    max_length=256,
)

results = clf(["We deployed our internal AI assistant last quarter."])
print(results)
