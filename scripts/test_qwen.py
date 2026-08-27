import traceback
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    print("Initializing BitsAndBytesConfig...")
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        device_map="auto",
        quantization_config=quant,
        attn_implementation="eager"
    )
    print("Model loaded successfully! VRAM:", round(torch.cuda.memory_allocated()/1024**3, 2), "GB")
    inputs = tok(["<|im_start|>user\nSay hello!<|im_end|>\n<|im_start|>assistant\n"], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=30)
    print("Generated:", tok.decode(out[0], skip_special_tokens=True))
except Exception as e:
    traceback.print_exc()
