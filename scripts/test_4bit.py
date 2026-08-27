import traceback
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print("Testing bitsandbytes 4-bit on GPU...")
try:
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        quantization_config=quant,
        device_map={"": 0},
    )
    print("4-bit GPU model loaded successfully!")
    print("VRAM in use:", round(torch.cuda.memory_allocated()/1024**3, 2), "GB")
    
    # Test generation
    inputs = tok(["<|im_start|>user\nSay hi in 5 words!<|im_end|>\n<|im_start|>assistant\n"], return_tensors="pt").to(0)
    out = model.generate(**inputs, max_new_tokens=20)
    print("Output:", tok.decode(out[0], skip_special_tokens=True))
except Exception as e:
    traceback.print_exc()
