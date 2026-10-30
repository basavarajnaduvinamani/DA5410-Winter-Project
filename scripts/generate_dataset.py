"""Verify existing assets; never silently substitute label-invalidating fallback URLs."""
import json
from pathlib import Path

def generate_images_with_ai_studio(api_key=None):
    root = Path(__file__).resolve().parents[1]
    cases = json.loads((root / "dataset/counterprior_benchmark.json").read_text())
    missing = [c["image_file"] for c in cases if not (root / "dataset/images" / c["image_file"]).is_file()]
    if missing:
        raise RuntimeError("Missing images: " + repr(missing) + "; supply verified generation prompts and audit replacement pixels before changing labels")
    print("Existing image files verified. Pixel/label exclusions: dataset/audit_2026-10-30.json")

if __name__ == "__main__":
    generate_images_with_ai_studio()
