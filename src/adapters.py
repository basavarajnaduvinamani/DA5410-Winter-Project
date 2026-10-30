"""Model adapters: one interface for answer generation, state capture and steering.

LLaVA-1.5 is the default and the only adapter with a pinned revision in
configs/. Qwen2.5-VL and InternVL (transformers-native "-hf" checkpoints) sit
behind `--adapter`; they are exercised offline with stubs only, so treat any
cross-model run as unverified until it has been executed on a GPU.
"""
from contextlib import ExitStack

import numpy as np
import torch
import torch.nn as nn

from .model_loader import flush_gpu_memory
from .rotational_steering import SteeringContext

POPE_INSTRUCTION = 'Answer the question using a single word or phrase.'


def find_decoder_layers(model):
    """Known paths first (LLaVA), then the longest non-vision ModuleList named 'layers'."""
    for path in ('model.language_model.layers', 'language_model.model.layers', 'language_model.layers'):
        node = model
        for part in path.split('.'):
            node = getattr(node, part, None)
        if isinstance(node, nn.ModuleList):
            return node
    best = None
    for name, module in model.named_modules():
        if name.endswith('layers') and isinstance(module, nn.ModuleList):
            if any(v in name for v in ('vision', 'visual', 'vit')):
                continue
            if best is None or len(module) > len(best):
                best = module
    if best is None:
        raise RuntimeError('Could not locate language decoder layers')
    return best


def quantization_config(quantization):
    if quantization in (None, 'none'):
        return None
    from transformers import BitsAndBytesConfig
    if quantization == '4bit':
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                  bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True)
    if quantization == '8bit':
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f'quantization must be 4bit, 8bit or none, not {quantization!r}')


class VLMAdapter:
    """Subclasses implement `prompt` and `inputs`; `run` is shared."""
    name = 'base'

    def __init__(self, model, processor, tokenizer=None):
        self.model, self.processor = model, processor
        self.tokenizer = tokenizer or getattr(processor, 'tokenizer', processor)
        self.layers = find_decoder_layers(model)
        self.yes_ids = self._first_ids(['Yes', 'yes', ' Yes', ' yes'])
        self.no_ids = self._first_ids(['No', 'no', ' No', ' no'])

    def _first_ids(self, words):
        ids = set()
        for w in words:
            enc = self.tokenizer.encode(w, add_special_tokens=False)
            if enc:
                ids.add(int(enc[0]))
        return sorted(ids)

    @property
    def num_layers(self):
        return len(self.layers)

    def prompt(self, question):
        raise NotImplementedError

    def inputs(self, image, question):
        raise NotImplementedError

    def device(self):
        return next(self.model.parameters()).device

    def run(self, image, question, max_new_tokens=3, capture_layers=None, steering=None):
        """Greedy answer plus first-token statistics; optional prefill terminal states and steering hook.

        `steering` is (layer_index, hook). States are captured on the first
        forward call only (the prefill), at the last prompt position.
        """
        inputs = self.inputs(image, question)
        prompt_len = inputs['input_ids'].shape[1]
        states = {}

        def capture(i):
            def hook(module, args, output):
                if i not in states:
                    h = output[0] if isinstance(output, tuple) else output
                    states[i] = h[0, -1].detach().float().cpu()
            return hook
        with ExitStack() as stack:
            if steering is not None:
                stack.enter_context(SteeringContext(self.layers[steering[0]], steering[1]))
            for i in capture_layers or []:
                stack.enter_context(SteeringContext(self.layers[i], capture(i)))
            with torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                          output_scores=True, return_dict_in_generate=True,
                                          pad_token_id=getattr(self.tokenizer, 'pad_token_id', None) or 0)
        logp = torch.log_softmax(out.scores[0][0].float(), -1)
        chosen = int(out.sequences[0, prompt_len])
        text = self.tokenizer.decode(out.sequences[0, prompt_len:], skip_special_tokens=True).strip()
        p = logp.exp()
        result = dict(answer=text, first_token_id=chosen, logprob_first=float(logp[chosen]),
                      entropy_first=float(-(p * logp).nan_to_num().sum()),
                      p_yes=float(p[self.yes_ids].sum()) if self.yes_ids else None,
                      p_no=float(p[self.no_ids].sum()) if self.no_ids else None)
        if capture_layers:
            result['states'] = torch.stack([states[i] for i in capture_layers]).numpy().astype(np.float16)
        return result


class LlavaAdapter(VLMAdapter):
    name = 'llava'

    def prompt(self, question):
        # llava-hf model-card format with the LLaVA-1.5 POPE instruction.
        return f'USER: <image>\n{question}\n{POPE_INSTRUCTION} ASSISTANT:'

    def inputs(self, image, question):
        batch = self.processor(text=self.prompt(question), images=image, return_tensors='pt')
        return batch.to(self.device())


class ChatTemplateAdapter(VLMAdapter):
    """Qwen2.5-VL / InternVL-hf: the processor's own chat template inserts image tokens."""

    def prompt(self, question):
        messages = [dict(role='user', content=[dict(type='image'), dict(type='text', text=f'{question}\n{POPE_INSTRUCTION}')])]
        return self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    def inputs(self, image, question):
        batch = self.processor(text=[self.prompt(question)], images=[image], return_tensors='pt')
        return batch.to(self.device())


class QwenVLAdapter(ChatTemplateAdapter):
    name = 'qwen2_5_vl'


class InternVLAdapter(ChatTemplateAdapter):
    name = 'internvl'


ADAPTERS = {a.name: a for a in (LlavaAdapter, QwenVLAdapter, InternVLAdapter)}

DEFAULT_CHECKPOINTS = {
    'llava': 'llava-hf/llava-1.5-7b-hf',
    'qwen2_5_vl': 'Qwen/Qwen2.5-VL-3B-Instruct',
    'internvl': 'OpenGVLab/InternVL3-2B-hf',
}


def load_adapter(name, model_id=None, revision=None, quantization='4bit', attn_implementation='sdpa'):
    """Download and wrap a pretrained checkpoint (GPU runs only; tests inject stub models)."""
    if name not in ADAPTERS:
        raise ValueError(f'Unknown adapter {name!r}; choose from {sorted(ADAPTERS)}')
    from transformers import AutoProcessor
    flush_gpu_memory()
    model_id = model_id or DEFAULT_CHECKPOINTS[name]
    cuda = torch.cuda.is_available()
    kwargs = dict(revision=revision, attn_implementation=attn_implementation,
                  quantization_config=quantization_config(quantization) if cuda else None,
                  device_map='auto' if cuda else None, torch_dtype=torch.float16 if cuda else torch.float32)
    if name == 'llava':
        from transformers import LlavaForConditionalGeneration as cls
    else:
        from transformers import AutoModelForImageTextToText as cls
    model = cls.from_pretrained(model_id, **kwargs).eval()
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    return ADAPTERS[name](model, processor)
