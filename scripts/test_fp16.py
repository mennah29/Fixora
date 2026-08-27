import sys
import traceback
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Testing FP16 device_map='auto'...")
try:
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print("Model loaded successfully in FP16 with device_map='auto'!")
    print("Device map:", model.hf_device_map)
    inputs = tok(["<|im_start|>user\nSay hello!<|im_end|>\n<|im_start|>assistant\n"], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=20)
    print("Output:", tok.decode(out[0], skip_special_tokens=True))
except Exception as e:
    traceback.print_exc()
