import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel

# Fine-tuned model name
new_model = "llama-2-7b-med_qa"

model_name = "meta-llama/Llama-2-7b-chat-hf"

device_map = {"": 0}

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model.push_to_hub(new_model, use_temp_dir=False)
tokenizer.push_to_hub(new_model, use_temp_dir=False)