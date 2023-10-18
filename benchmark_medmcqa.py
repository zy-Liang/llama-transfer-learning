import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    # HfArgumentParser,
    # TrainingArguments,
    pipeline,
    # logging,
)

model_name = "akaiDing/llama-2-7b-medmcqa"
dataset_name = "bigbio/med_qa"

dataset = load_dataset(dataset_name, split="test")

# Load the entire model on the GPU 0
device_map = {"": 0}

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

print("==========================")
# dataset = dataset[0:10]
correct = 0
for i in range(len(dataset)):
    choice_text = ""
    for option in dataset['options'][i]:
        choice_text += f"{option['key']}. {option['value']}\n"
    prompt = (f"Example Input:\n"
        f"### Question: What's the result of 1 + 1?\n"
        f"### Choices:\n"
        f"A. 1\n"
        f"B. 2\n"
        f"C. 3\n"
        f"D. 4\n"
        f"E. 5\n\n"
        f"Example Output:\n"
        f"### Answer: B. 2\n"
        f"Input:\n"
        f"### Question: {dataset['question'][i]}\n"
        f"### Choices:\n{choice_text}\n"
        f"Output:\n"
        "### Answer:"
    )
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
    result = pipe(f"{prompt}")
    result = result[0]['generated_text']
    # print(prompt)
    # print('--------------------------')
    print(result)
    print('--------------------------')
    index = result.rfind("### Answer:")
    index += len("### Answer: ")
    result = result[index]
    print(f"model answer: {result}, correct answer: {dataset['answer_idx'][i]}")
    print("==========================")
    if result == dataset['answer_idx'][i]:
        correct += 1
print(f"Questions: {len(dataset)}")
print(f"Correct questions: {correct}")
print(f"Accuracy: {correct / len(dataset)}")
