# Refer to this link given any questions. https://replicate.com/blog/fine-tune-alpaca-with-lora

import os
import sys

import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers

from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, MT5ForConditionalGeneration
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)


torch.cuda.empty_cache()

print(torch.cuda.memory_summary())

MICRO_BATCH_SIZE = 64
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 3  
LEARNING_RATE = 3e-4  # the Karpathy constant
CUTOFF_LEN = 256  # 256 accounts for about 96% of the data
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = 200
TARGET_MODULES = [
    "query",
    "value",
]

DATA_PATH = "datasets/test/small_sample_adj.json" # Change this accordingly
OUTPUT_DIR = "out/lora-alpaca" # The directory to save the model. Change accordingly

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
print(f"world_size: {world_size}")
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size
print(f"device_map: {device_map}")
# model = LlamaForCausalLM.from_pretrained(
#     "drive/My Drive/output/",
#     load_in_8bit=True,
#     device_map=device_map,
# )
# tokenizer = LlamaTokenizer.from_pretrained(
#     "drive/My Drive/output/", add_eos_token=True
# )

PATH_TO_CONVERTED_WEIGHTS = "bigscience/mt0-small" # The path to the saved weights. Change accordingly
PATH_TO_CONVERTED_TOKENIZER = "bigscience/mt0-small" # The path to the tokenizer. Change accordingly

model_llama = MT5ForConditionalGeneration.from_pretrained(PATH_TO_CONVERTED_WEIGHTS) # change the class according to the model type
tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER) # To be honest, I'm not sure about the difference between autotokenizer and llamatokenizer...

model = prepare_model_for_int8_training(model_llama)

print("Tokenizer loaded.")
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    # target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
print("Loading data...")
data = load_dataset("json", data_files=DATA_PATH)


def generate_prompt(data_point):
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""


def tokenize(prompt):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )
    return {
        "input_ids": result["input_ids"][:-1],
        "attention_mask": result["attention_mask"][:-1],
    }


def generate_and_tokenize_prompt(data_point):
    prompt = generate_prompt(data_point)
    return tokenize(prompt)


print("Tokenizing data...")
if VAL_SET_SIZE > 0:
    train_val = data["train"].train_test_split(
        test_size=VAL_SET_SIZE, shuffle=True, seed=42
    )
    train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
else:
    train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = None

print("Building trainer...")
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=20,
        evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
        save_strategy="steps",
        eval_steps=200 if VAL_SET_SIZE > 0 else None,
        save_steps=200,
        output_dir=OUTPUT_DIR,
        save_total_limit=3,
        load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = True

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

if __name__ == "__main__":
    print("Training...")
    trainer.train()
    
    print("Saving model...")
    model.save_pretrained(OUTPUT_DIR)

    # Test
    prompt = "The medical note of a patient."
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate
    generate_ids = model.generate(input_ids=inputs.input_ids, max_length=60)
    tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
