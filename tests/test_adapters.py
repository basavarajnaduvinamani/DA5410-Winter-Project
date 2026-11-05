import os
os.environ.setdefault('USE_TF', '0')
import unittest

import torch
import torch.nn as nn
from PIL import Image

from src.adapters import ADAPTERS, DEFAULT_CHECKPOINTS, InternVLAdapter, LlavaAdapter, QwenVLAdapter, find_decoder_layers, load_adapter, quantization_config
from src.rotational_steering import RotationalSteeringHook
from stubs import StubChatProcessor, StubProcessor, tiny_llava


class QwenLikeStub(nn.Module):
    """Module tree shaped like Qwen2.5-VL: a deeper vision block list must not be picked."""
    def __init__(self):
        super().__init__()
        self.visual = nn.Module(); self.visual.layers = nn.ModuleList(nn.Identity() for _ in range(40))
        self.model = nn.Module(); self.model.decoder = nn.Module()
        self.model.decoder.layers = nn.ModuleList(nn.Linear(2, 2) for _ in range(6))


class Adapters(unittest.TestCase):
    def test_registry_and_flags(self):
        self.assertEqual(set(ADAPTERS), {'llava', 'qwen2_5_vl', 'internvl'})
        self.assertEqual(set(DEFAULT_CHECKPOINTS), set(ADAPTERS))
        with self.assertRaisesRegex(ValueError, 'Unknown adapter'):
            load_adapter('gpt4v')
        with self.assertRaisesRegex(ValueError, 'quantization'):
            quantization_config('3bit')
        self.assertIsNone(quantization_config('none'))

    def test_layer_finder_skips_vision_tower(self):
        stub = QwenLikeStub()
        self.assertIs(find_decoder_layers(stub), stub.model.decoder.layers)

    def test_llava_adapter_runs_tiny_model_with_capture_and_steering(self):
        torch.set_num_threads(1)
        adapter = LlavaAdapter(tiny_llava(), StubProcessor())
        self.assertEqual(adapter.num_layers, 2)
        self.assertEqual((adapter.yes_ids, adapter.no_ids), ([10], [11]))
        image = Image.new('RGB', (20, 20), (200, 30, 30))
        out = adapter.run(image, 'Is there a dog in the image?', max_new_tokens=2, capture_layers=[0, 1])
        self.assertIn(out['answer'], {'Yes', 'No'})
        self.assertEqual(out['states'].shape, (2, 32))
        self.assertLessEqual(out['logprob_first'], 0)
        self.assertGreaterEqual(out['entropy_first'], 0)
        self.assertTrue(0 <= out['p_yes'] <= 1)
        self.assertIn('Answer the question using a single word or phrase. ASSISTANT:', adapter.processor.prompts[-1])
        hook = RotationalSteeringHook(torch.ones(32), alpha=.9, mode='constant')
        again = adapter.run(image, 'Is there a dog in the image?', max_new_tokens=2, steering=(1, hook))
        self.assertNotAlmostEqual(again['logprob_first'] + again['entropy_first'], out['logprob_first'] + out['entropy_first'], places=6)
        self.assertTrue(all(len(l._forward_hooks) == 0 for l in adapter.layers))

    def test_chat_template_adapters_with_stub_processor(self):
        for cls in (QwenVLAdapter, InternVLAdapter):
            adapter = cls(tiny_llava(), StubChatProcessor())
            prompt = adapter.prompt('Is there a cat in the image?')
            self.assertTrue(prompt.startswith('<|im_start|>user\n<image>Is there a cat'))
            self.assertTrue(prompt.endswith('<|im_start|>assistant\n'))
            out = adapter.run(Image.new('RGB', (16, 16)), 'Is there a cat in the image?', max_new_tokens=1)
            self.assertIn(out['answer'], {'Yes', 'No'})


if __name__ == '__main__':
    unittest.main()
