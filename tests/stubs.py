"""Offline fixtures: random-weight tiny LLaVA, stub processor/tokenizer, synthetic POPE files."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGE_TOKEN = 63


def tiny_llava(seed=20261030, layers=2):
    from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration
    torch.manual_seed(seed)
    config = LlavaConfig(text_config=LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
                                                 num_attention_heads=4, num_key_value_heads=2),
                         vision_config=CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                                        num_attention_heads=4, image_size=16, patch_size=8),
                         image_token_index=IMAGE_TOKEN, image_seq_length=4, vision_feature_layer=-1)
    return LlavaForConditionalGeneration(config).eval()


class StubTokenizer:
    """Even ids decode to 'Yes', odd ids to 'No'; enough to exercise the official POPE parser."""
    pad_token_id = 0
    VOCAB = {'Yes': 10, 'yes': 10, ' Yes': 10, ' yes': 10, 'No': 11, 'no': 11, ' No': 11, ' no': 11}

    def encode(self, word, add_special_tokens=False):
        return [self.VOCAB.get(word, 12)]

    def decode(self, ids, skip_special_tokens=True):
        ids = [int(i) for i in ids]
        return ('Yes' if ids[0] % 2 == 0 else 'No') if ids else ''


class StubProcessor:
    """Mimics the llava-hf processor: image placeholder tokens + 16x16 pixel tensor."""
    def __init__(self):
        self.tokenizer = StubTokenizer()
        self.prompts = []

    def __call__(self, text, images, return_tensors='pt'):
        from transformers.feature_extraction_utils import BatchFeature
        text = text[0] if isinstance(text, list) else text
        image = images[0] if isinstance(images, list) else images
        self.prompts.append(text)
        digest = hashlib.sha256(text.encode()).digest()
        prompt_ids = [1 + b % 60 for b in digest[:6]]
        pixels = np.asarray(image.convert('RGB').resize((16, 16)), dtype=np.float32) / 127.5 - 1
        ids = torch.tensor([[1] + [IMAGE_TOKEN] * 4 + prompt_ids])
        return BatchFeature(dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                                 pixel_values=torch.tensor(pixels).permute(2, 0, 1)[None]))


class StubChatProcessor(StubProcessor):
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        parts = []
        for m in messages:
            body = ''.join('<image>' if c['type'] == 'image' else c['text'] for c in m['content'])
            parts.append(f"<|im_start|>{m['role']}\n{body}<|im_end|>\n")
        return ''.join(parts) + ('<|im_start|>assistant\n' if add_generation_prompt else '')


def write_synthetic_pope(root, n_images=20, seed=0):
    """POPE-format files for three splits sharing images; one yes and one no question per image."""
    root = Path(root); images = root / 'images'; images.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    objects = ['dog', 'cat', 'car', 'chair', 'cup', 'bench']
    for k in range(n_images):
        Image.fromarray(rng.integers(0, 255, size=(24, 24, 3), dtype=np.uint8)).save(images / f'img_{k:03d}.jpg')
    for s, split in enumerate(['random', 'popular', 'adversarial']):
        lines, qid = [], 1
        for k in range(n_images):
            for label in ('yes', 'no'):
                obj = objects[(k + s + (label == 'no') * 3) % len(objects)]
                lines.append(json.dumps(dict(question_id=qid, image=f'img_{k:03d}.jpg',
                                             text=f'Is there a {obj} in the image?', label=label)))
                qid += 1
        (root / f'coco_pope_{split}.json').write_text('\n'.join(lines) + '\n')
    return root, images


def synthetic_config(**over):
    config = dict(adapter='llava', model_id='stub', revision=None, quantization='none', attn_implementation='eager',
                  seed=20261030, split_fractions=dict(calibration=0.5, validation=0.25, test=0.25), max_images=None,
                  max_new_tokens=2, state_layers='all', detection=dict(target='error', cv_folds=3, C=1.0),
                  steering=dict(layer='probe', gate='calibration_cosine', slerp_alphas=[0.1, 0.5], addition_betas=[0.5],
                                controls=['random_direction', 'shuffled_labels']),
                  bootstrap_iterations=50, run_id='synthetic')
    config.update(over)
    return config
