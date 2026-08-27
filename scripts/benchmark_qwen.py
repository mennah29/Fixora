import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(8)
torch.set_num_interop_threads(4)

print("Benchmarking Qwen2.5-3B-Instruct with 8 CPU threads...")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="cpu",
)

prompt = "<|im_start|>user\nWhat is error code E37 in ultrasound equipment?<|im_end|>\n<|im_start|>assistant\n"
inputs = tok([prompt], return_tensors="pt")

start = time.time()
out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
elapsed = time.time() - start
print(f"Generated 60 tokens in {elapsed:.2f} seconds ({60/elapsed:.2f} tok/s)!")
print("Response:", tok.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True))
