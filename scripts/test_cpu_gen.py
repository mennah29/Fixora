import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Testing loading Qwen2.5-3B-Instruct on CPU and generating...")
start = time.time()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="cpu"
)
load_time = time.time() - start
print(f"Model loaded in {load_time:.2f}s!")

prompt = "<|im_start|>user\nWhat is error code E37 in ultrasound equipment?<|im_end|>\n<|im_start|>assistant\n"
inputs = tok([prompt], return_tensors="pt")
start_gen = time.time()
out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
gen_time = time.time() - start_gen
response = tok.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
print(f"Generation took {gen_time:.2f}s:")
print("Response:", response)
