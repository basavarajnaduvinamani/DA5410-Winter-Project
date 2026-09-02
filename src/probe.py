import torch
import torch.nn.functional as F
from PIL import Image

def extract_prompt_terminal_state(model, processor, image, prompt_text, target_layer=None):
    """
    Extracts the hidden representation at the EXACT prompt boundary ('ASSISTANT:')
    before any completion tokens are generated.
    
    This eliminates the Lexical Semantic Confound (where probing on 'Yes' vs 'No'
    simply decodes the meaning of affirmation/negation words).
    """
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
        
    formatted_prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"
    inputs = processor(text=formatted_prompt, images=image, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        
    hidden_states = outputs.hidden_states
    
    # Last token index is the colon in 'ASSISTANT:'
    if target_layer is not None:
        return hidden_states[target_layer][0, -1, :].detach().cpu()
    else:
        # Return all decoder layers: shape (num_layers, hidden_dim)
        return torch.stack([layer[0, -1, :].detach().cpu() for layer in hidden_states])

def logit_lens_token_divergence(model, processor, image, prompt_text, truth_token_str, hallucinated_token_str, target_layers=None):
    """
    Projects intermediate hidden states directly onto the vocabulary using the unembedding head (lm_head).
    Tracks when the model's internal probability flips from truth to hallucination.
    """
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
        
    tokenizer = processor.tokenizer
    truth_id = tokenizer.encode(" " + truth_token_str.strip(), add_special_tokens=False)[-1]
    hallucinated_id = tokenizer.encode(" " + hallucinated_token_str.strip(), add_special_tokens=False)[-1]
    
    all_states = extract_prompt_terminal_state(model, processor, image, prompt_text)
    num_layers = all_states.shape[0]
    
    if target_layers is None:
        target_layers = list(range(num_layers))
        
    results = []
    for l in target_layers:
        h = all_states[l].unsqueeze(0).to(model.device)
        with torch.no_grad():
            logits = model.lm_head(h)
            probs = F.softmax(logits[0], dim=-1)
            p_truth = probs[truth_id].item()
            p_lie = probs[hallucinated_id].item()
            
        results.append({
            "layer": l,
            "p_truth": p_truth,
            "p_hallucinated": p_lie,
            "log_ratio": torch.log(torch.tensor(p_truth + 1e-9) / torch.tensor(p_lie + 1e-9)).item()
        })
        
    return results
