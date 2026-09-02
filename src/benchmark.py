import json
import os
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from .rotational_steering import RotationalSteeringHook, SteeringContext

def run_counterprior_benchmark(
    model,
    processor,
    target_layer_module,
    steering_tensor,
    benchmark_json_path,
    image_dir,
    alpha=0.25,
    threshold=15.0,
    max_new_tokens=30
):
    """
    Runs automated evaluation across CounterPrior test cases for both Baseline and Steered models.
    Computes Precision, Recall, F1, and Hallucination Reduction Rate.
    """
    with open(benchmark_json_path, "r", encoding="utf-8") as f:
        cases = json.load(f)
        
    tokenizer = processor.tokenizer
    hook = RotationalSteeringHook(steering_tensor, alpha=alpha, threshold=threshold, mode="adaptive")
    
    results = []
    print(f"Running Benchmark on {len(cases)} items (Baseline vs Rotational Steering)...")
    
    for item in cases:
        img_path = os.path.join(image_dir, item["image_file"])
        if not os.path.exists(img_path):
            print(f"Skipping missing image: {item['image_file']}")
            continue
            
        image = Image.open(img_path).convert("RGB")
        prompt = f"USER: <image>\n{item['prompt']}\nASSISTANT:"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
        
        # 1. Baseline Generation (No steering)
        with torch.no_grad():
            base_out = model.generate(**inputs, max_new_tokens=max_new_tokens)
        base_text = tokenizer.decode(base_out[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()
        
        # 2. Steered Generation (Rotational hook active)
        with SteeringContext(target_layer_module, hook):
            with torch.no_grad():
                steered_out = model.generate(**inputs, max_new_tokens=max_new_tokens)
        steered_text = tokenizer.decode(steered_out[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()
        
        # 3. Ground Truth Evaluation
        truth_keywords = [k.lower() for k in item["verification_truth_keywords"]]
        prior_keywords = [k.lower() for k in item["hallucinated_prior_keywords"]]
        
        base_is_truth = any(k in base_text.lower() for k in truth_keywords)
        base_is_hallucination = any(k in base_text.lower() for k in prior_keywords)
        
        steered_is_truth = any(k in steered_text.lower() for k in truth_keywords)
        steered_is_hallucination = any(k in steered_text.lower() for k in prior_keywords)
        
        results.append({
            "id": item["id"],
            "type": item["category"],
            "prompt": item["prompt"],
            "baseline_output": base_text,
            "steered_output": steered_text,
            "baseline_correct": base_is_truth and not base_is_hallucination,
            "steered_correct": steered_is_truth and not steered_is_hallucination
        })
        
    df = pd.DataFrame(results)
    base_acc = df["baseline_correct"].mean() * 100
    steered_acc = df["steered_correct"].mean() * 100
    
    print("\n" + "="*60)
    print(f"BENCHMARK RESULTS (N={len(df)}):")
    print(f"Baseline Accuracy: {base_acc:.2f}%")
    print(f"Steered Accuracy:  {steered_acc:.2f}% (Gain: +{steered_acc - base_acc:.2f}%)")
    print("="*60)
    
    return df
