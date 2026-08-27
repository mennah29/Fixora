import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("Testing CPU load...")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    print("Tokenizer loaded!")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu"
    )
    print("Model loaded successfully on CPU!")
except Exception as e:
    import traceback
    traceback.print_exc()
