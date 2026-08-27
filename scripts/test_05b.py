import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(8)

print("Testing Qwen2.5-0.5B-Instruct...")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    device_map="cpu",
)

prompt = "<|im_start|>user\nWhat is error code E37 in ultrasound equipment?<|im_end|>\n<|im_start|>assistant\n"
inputs = tok([prompt], return_tensors="pt")

start = time.time()
out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
elapsed = time.time() - start
print(f"Generated 100 tokens in {elapsed:.2f} seconds ({100/elapsed:.2f} tok/s)!")
print("Response:\n", tok.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True))
