"""Raw decoder-output extraction; layers are zero based and exclude embeddings."""
import torch
from PIL import Image
from .model_loader import find_language_decoder_layers

def extract_prompt_terminal_state(model, processor, image, prompt_text, target_layer=None):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    inputs = processor(text=f"USER: <image>\n{prompt_text}\nASSISTANT:", images=image, return_tensors="pt").to(model.device)
    layers = find_language_decoder_layers(model)
    selected = list(range(len(layers))) if target_layer is None else [target_layer]
    states, handles = {}, []
    def capture(i):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            states[i] = h[0, -1].detach().float().cpu()
        return hook
    try:
        for i in selected:
            handles.append(layers[i].register_forward_hook(capture(i)))
        with torch.inference_mode():
            model(**inputs, use_cache=False, return_dict=True)
    finally:
        for h in handles:
            h.remove()
    return states[target_layer] if target_layer is not None else torch.stack([states[i] for i in selected])

def logit_lens_token_divergence(model, processor, image, prompt_text, truth_token_str, hallucinated_token_str, target_layers=None):
    ids = [processor.tokenizer.encode(" " + s.strip(), add_special_tokens=False) for s in [truth_token_str, hallucinated_token_str]]
    if any(len(x) != 1 for x in ids):
        raise ValueError("Logit lens requires single-token candidates; never select the last subtoken silently")
    states = extract_prompt_terminal_state(model, processor, image, prompt_text)
    decoder = model.model.language_model if hasattr(model, 'model') and hasattr(model.model, 'language_model') else model.language_model.model
    head = model.get_output_embeddings()
    results = []
    for i in (range(len(states)) if target_layers is None else target_layers):
        h = states[i].to(device=decoder.norm.weight.device, dtype=decoder.norm.weight.dtype)
        with torch.inference_mode():
            normalized = decoder.norm(h)
            logits = head(normalized.to(device=head.weight.device, dtype=head.weight.dtype)).float()
            probs = logits.softmax(-1)
        results.append(dict(layer=i, p_truth=probs[ids[0][0]].item(), p_hallucinated=probs[ids[1][0]].item(), log_ratio=(logits[ids[0][0]]-logits[ids[1][0]]).item()))
    return results
