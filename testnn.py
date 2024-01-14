import pandas as pd
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AdamW
from transformers import Trainer, TrainingArguments



df = pd.read_csv("./datasets/cleaned.csv")
df = df[:10000].copy()
# Assuming 'df' is your DataFrame
texts = df['clean_text'].tolist()
labels = df['category'].tolist()

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

encoding = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {'input_ids': self.encodings['input_ids'][idx],
                'attention_mask': self.encodings['attention_mask'][idx],
                'labels': self.labels[idx]}

dataset = CustomDataset(encoding, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=13)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    save_steps=100,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataloader,
)

trainer.train()

trainer.evaluate()

# Save the model
model.save_pretrained('./saved_model')

