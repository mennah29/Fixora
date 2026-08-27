import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(8)

print("Testing torch.bfloat16 on CPU...")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="cpu",
)

prompt = "<|im_start|>user\nSay hi in 5 words!<|im_end|>\n<|im_start|>assistant\n"
inputs = tok([prompt], return_tensors="pt")

start = time.time()
out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
elapsed = time.time() - start
print(f"bfloat16 generated 20 tokens in {elapsed:.2f}s ({20/elapsed:.2f} tok/s)!")
print("Response:", tok.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True))
