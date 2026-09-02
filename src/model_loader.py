import gc
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

def flush_gpu_memory():
    """Aggressive memory flush to prevent CUDA OOM on 16GB GPUs (Colab T4)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def load_llava_model(
    model_id="llava-hf/llava-1.5-7b-hf",
    attn_implementation="sdpa",
    load_in_4bit=True,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Loads LLaVA-1.5-7B with 4-bit quantization and configurable attention.
    
    Args:
        model_id: HuggingFace model repo identifier.
        attn_implementation: 'sdpa' (fast flash attention) or 'eager' (for attention map extraction).
        load_in_4bit: Whether to quantize weights to NF4.
        device: Target execution device.
    """
    flush_gpu_memory()
    
    quant_config = None
    if load_in_4bit and device != "cpu":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

    print(f"Loading {model_id} (attn_implementation='{attn_implementation}', 4-bit={load_in_4bit})...")
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto" if device != "cpu" else None,
        attn_implementation=attn_implementation,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32
    )
    
    model.eval()
    return model, processor

def find_language_decoder_layers(model):
    """
    Introspectively traverses model modules to locate the 32 decoder layers of Vicuna.
    """
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        if hasattr(model.language_model.model, "layers"):
            return model.language_model.model.layers
    
    # Fallback recursive search
    import torch.nn as nn
    for name, child in model.named_children():
        if hasattr(child, "layers") and isinstance(child.layers, nn.ModuleList) and len(child.layers) >= 30:
            return child.layers
        for sub_name, sub_child in child.named_children():
            if hasattr(sub_child, "layers") and isinstance(sub_child.layers, nn.ModuleList) and len(sub_child.layers) >= 30:
                return sub_child.layers
                
    raise RuntimeError("Could not locate the 32 language decoder layers in the model.")
