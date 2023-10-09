import os
import sys

import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers

from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, LlamaConfig
from modules import LlamaT

MICRO_BATCH_SIZE = 2
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 3
LEARNING_RATE = 3e-4  # the Karpathy constant
CUTOFF_LEN = 256  # 256 accounts for about 96% of the data
VAL_SET_SIZE = 200

DATA_PATH = "drive/My Drive/small_sample_adj.json" # Change this accordingly
OUTPUT_DIR = "drive/My Drive/lora-alpaca" # The directory to save the model. Change accordingly

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size


PATH_TO_CONVERTED_WEIGHTS = "drive/My Drive/output/" # The path to the saved weights. Change accordingly
PATH_TO_CONVERTED_TOKENIZER = "drive/My Drive/output/" # The path to the tokenizer. Change accordingly

model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER) # To be honest, I'm not sure about the difference between autotokenizer and llamatokenizer...

# This is to freeze all of the parameters except the last linear layer
for param in model.parameters():
    param.requires_grad = False

config = LlamaConfig(vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=1,
        num_attention_heads=32,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False)
transfer = LlamaT(config)

tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
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


if VAL_SET_SIZE > 0:
    train_val = data["train"].train_test_split(
        test_size=VAL_SET_SIZE, shuffle=True, seed=42
    )
    train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
else:
    train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = None

trainer = transformers.Trainer(
    model=transfer,
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
        # save_strategy="steps",
        eval_steps=200 if VAL_SET_SIZE > 0 else None,
        save_steps=20000, # I change this to a fairly large number otherwise it will save the checkpoint everytime... I believe there is a way to only save the trained part. TBD
        output_dir=OUTPUT_DIR,
        save_total_limit=3,
        load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

if __name__ == "__main__":
    trainer.train()
    
    # Test
    prompt = "Write a response for a patient."
    inputs = tokenizer1(prompt, return_tensors="pt")
    
    model = model.to('cpu')
    # Generate
    generate_ids = model.generate(input_ids=inputs.input_ids, max_length=60)
    tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
