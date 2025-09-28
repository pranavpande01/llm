# qlora_finetune.py (sketch)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_name = "mistralai/Mistral-7B-Instruct"   # example — pick a suitable instruct model allowed in your infra
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# Load 4-bit quantized model (bitsandbytes)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# prepare for k-bit training (necessary hooks)
model = prepare_model_for_kbit_training(model)

# LoRA config (tweak r and alpha)
lora_config = LoraConfig(
    r=16,                # rank (try 8-32; step1 may need higher)
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # typical
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Datasets: use Hugging Face datasets or a local dataset
from datasets import load_dataset
dataset = load_dataset("path/to/your/step1-cleaning-dataset")  # expect {input_text, target_text}

def preprocess(ex):
    tok = tokenizer(ex["input_text"], truncation=True, padding="max_length", max_length=512)
    labels = tokenizer(ex["target_text"], truncation=True, padding="max_length", max_length=256).input_ids
    tok["labels"] = labels
    return tok

train_ds = dataset["train"].map(preprocess, batched=True)
val_ds = dataset["validation"].map(preprocess, batched=True)

training_args = TrainingArguments(
    output_dir="./qlora_step1",
    per_device_train_batch_size=4,    # lower for Codespaces; use gradient_accumulation if needed
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=50,
    fp16=True,
    optim="adamw_torch",
    evaluation_strategy="steps",
    save_strategy="epoch",
    gradient_accumulation_steps=8,   # simulate larger batch
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer
)

trainer.train()
# Save LoRA adapter only:
model.save_pretrained("./qlora_step1/lora_adapter")
