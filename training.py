from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

# Load model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_8bit=True
)

def data_collator(features):
    input_ids = [torch.tensor(f["input_ids"]) for f in features]
    attention_mask = [torch.tensor(f["attention_mask"]) for f in features]
    labels = [torch.tensor(f["input_ids"]) for f in features]
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }

# Dataset
dataset = load_dataset("csv", data_files={"train": "polite_dataset.csv"}, encoding="utf-8")["train"]

def format(sample):
    text = f"### Instruction:\n{sample['instruction']}\n\n### Input:\n{sample['input']}\n\n### Response:\n{sample['output']}"
    return {"text": text}

dataset = dataset.map(format)

tokenized = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=256))

# LoRA config
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)

# Training args
args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    max_steps=100,   # small run
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    output_dir="./polite-lora"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    data_collator=data_collator
)

trainer.train()
model.save_pretrained("./polite-lora")