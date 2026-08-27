import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Testing max_memory allocation...")
try:
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    max_memory = {0: "2.5GiB", "cpu": "16GiB"}
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )
    print("Model loaded with max_memory split successfully!")
    print("Device map:", model.hf_device_map)
    inputs = tok(["<|im_start|>user\nWhat is error code E37 in ultrasound equipment?<|im_end|>\n<|im_start|>assistant\n"], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=40)
    print("Output:", tok.decode(out[0], skip_special_tokens=True))
except Exception as e:
    import traceback
    traceback.print_exc()
