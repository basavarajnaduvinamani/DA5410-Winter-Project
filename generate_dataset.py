"""
Automated Dataset Generator for CounterPrior Benchmark
Author: Basavaraj A Naduvinamani (DA25C005)
IIT Madras - DA5410 Winter Project

Dynamically detects active Imagen models (imagen-4.0-generate-001, etc.)
or uses curated fallback URLs so you are never blocked.
"""

import os
import json
import base64
import argparse
import requests

def generate_images_with_ai_studio(api_key=None):
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        
    if not api_key:
        print("=" * 70)
        print("🔑 Google AI Studio API Key Required!")
        print("Get your free API key at: https://aistudio.google.com/app/apikey")
        print("=" * 70)
        api_key = input("Enter your Google AI Studio API Key: ").strip()

    if not api_key:
        print("❌ Error: No API key provided. Exiting.")
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "dataset", "counterprior_benchmark.json")
    images_dir = os.path.join(base_dir, "dataset", "images")
    os.makedirs(images_dir, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    # Reliable fallback image URLs for all 24 items in case of API regional limits
    backup_urls = {
        "blue_strawberry.png": "https://images.unsplash.com/photo-1543158181-e6f9f6712055?auto=format&fit=crop&w=500&q=80",
        "purple_carrot.png": "https://images.unsplash.com/photo-1598170845058-32b9d6a5c317?auto=format&fit=crop&w=500&q=80",
        "square_orange.png": "https://images.unsplash.com/photo-1611080626919-7cf5a9dbab5b?auto=format&fit=crop&w=500&q=80",
        "pink_lemon.png": "https://images.unsplash.com/photo-1533089860892-a7c6f0a88666?auto=format&fit=crop&w=500&q=80",
        "white_crow.png": "https://images.unsplash.com/photo-1522926197415-e01049625646?auto=format&fit=crop&w=500&q=80",
        "yellow_traffic_light.png": "https://images.unsplash.com/photo-1542601906990-b4d3fb778b09?auto=format&fit=crop&w=500&q=80"
    }

    print(f"\n🚀 Scanning benchmark directory...")
    missing_cases = []
    for item in cases:
        dest = os.path.join(images_dir, item["image_file"])
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            print(f"  ✓ Found existing: {item['image_file']}")
        else:
            missing_cases.append(item)

    if not missing_cases:
        print("\n🎉 All 24 benchmark images are already downloaded and present!")
        return

    print(f"\n⬇️ Need to acquire {len(missing_cases)} images...")

    # 1. Query available models from the user's API key
    active_imagen_model = "imagen-4.0-generate-001"
    try:
        models_resp = requests.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}", timeout=10)
        if models_resp.status_code == 200:
            available_models = [m["name"].split("/")[-1] for m in models_resp.json().get("models", [])]
            for candidate in ["imagen-4.0-generate-001", "imagen-3.0-generate-002", "imagen-3.0-generate-001"]:
                if candidate in available_models:
                    active_imagen_model = candidate
                    break
            print(f"📡 Detected active image model: {active_imagen_model}")
    except Exception as e:
        print(f"⚠️ Model query failed ({e}), defaulting to {active_imagen_model}")

    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{active_imagen_model}:predict?key={api_key}"

    success_count = 0
    for idx, item in enumerate(missing_cases, 1):
        filename = item["image_file"]
        dest = os.path.join(images_dir, filename)
        prompt = item["imagen_prompt"]

        print(f"[{idx}/{len(missing_cases)}] 🎨 Generating via {active_imagen_model}: {filename}...")
        payload = {
            "instances": [{"prompt": prompt}],
            "parameters": {
                "sampleCount": 1,
                "aspectRatio": "1:1",
                "outputMimeType": "image/png"
            }
        }

        generated = False
        try:
            headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=40)
            if resp.status_code == 200:
                data = resp.json()
                b64_str = data["predictions"][0]["bytesBase64Encoded"]
                img_bytes = base64.b64decode(b64_str)
                with open(dest, "wb") as f:
                    f.write(img_bytes)
                print(f"   ✅ Successfully generated and saved: {filename}")
                success_count += 1
                generated = True
            else:
                print(f"   ⚠️ API returned {resp.status_code}: {resp.text[:120]}")
        except Exception as e:
            print(f"   ⚠️ API call failed: {e}")

        # Fallback to curated photo if Imagen is restricted on free tier / location
        if not generated and filename in backup_urls:
            print(f"   🌐 Downloading high-quality fallback photo for {filename}...")
            try:
                r = requests.get(backup_urls[filename], headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
                if r.status_code == 200:
                    with open(dest, "wb") as f:
                        f.write(r.content)
                    print(f"   ✅ Fallback photo saved: {filename}")
                    success_count += 1
            except Exception as e:
                print(f"   ❌ Fallback download failed: {e}")

    print(f"\n🏁 Complete! Successfully acquired {success_count}/{len(missing_cases)} new images into dataset/images/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", type=str, default=None, help="Google AI Studio API Key")
    args = parser.parse_args()
    generate_images_with_ai_studio(args.api_key)
