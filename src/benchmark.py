"""Strict legacy compatibility; new runs use case-aware adjudication."""
import re
from .adjudication import contains

def evaluate_response_truth(text, truth_keywords, prior_keywords):
    text = text.lower().strip()
    return (any(re.fullmatch(re.escape(k.lower()) + r"[.! ]*", text) for k in truth_keywords)
            and not any(contains(text, k.lower()) for k in prior_keywords)
            and not re.search(r"\b(not|no|without)\b", text))

def run_counterprior_benchmark(*args, **kwargs):
    raise RuntimeError("Use python -m scripts.run_experiment --config configs/llava.json; legacy unpersisted scoring is disabled")
