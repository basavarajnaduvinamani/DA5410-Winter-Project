import os
os.environ['USE_TF']='0'
os.environ['USE_FLAX']='0'
import unittest
import numpy as np
import torch
from src.protocol import make_splits, cluster_bootstrap, select_calibration
from src.rotational_steering import RotationalSteeringHook, SteeringContext, slerp_rotate


class ProtocolTests(unittest.TestCase):
    def test_group_disjoint_and_reproducible(self):
        cases=[dict(id=str(i),image_group=str(i//2),category='counter_existence') for i in range(24)]
        split=make_splits(cases,20261030)
        self.assertEqual(split,make_splits(cases,20261030))
        for c in cases:
            self.assertEqual(len({split[d['id']] for d in cases if d['image_group']==c['image_group']}),1)
        self.assertEqual(set(split.values()),{'calibration','validation','test'})

    def test_paired_cluster_resampling(self):
        rows=[dict(image_group=str(i),method=m,value=i) for i in range(5) for m in ['baseline','steered']]
        def diff(rs):
            b=[r['value'] for r in rs if r['method']=='baseline'];s=[r['value'] for r in rs if r['method']=='steered']
            return np.mean(s)-np.mean(b)
        result=cluster_bootstrap(rows,diff,20261030,100)
        self.assertEqual(result['ci95'],[0.,0.])

    def test_calibration_requires_class_support(self):
        with self.assertRaisesRegex(ValueError,'one class'):
            select_calibration(np.ones((4,2,3)),[0,0,0,1],list('abcd'),20261030)

    def test_calibration_selection_fits_only_supplied_images(self):
        rng=np.random.default_rng(20261030)
        states=rng.normal(size=(8,2,4));y=np.array([0,1]*4)
        states[:,0,0]=(1-2*y)*10
        selected,vec,probe=select_calibration(states,y,list('abcdefgh'),20261030)
        self.assertEqual(selected['selection'],'calibration_only')
        self.assertEqual(selected['layer'],0)
        self.assertGreater(vec @ (states[y==0,0].mean(0)-states[y==1,0].mean(0)),0)
        self.assertTrue(np.all(np.isfinite(probe.predict_proba(states[:,0]))))

    def test_hook_per_sample_terminal_and_removal(self):
        x=torch.tensor([[[1.,0.],[1.,0.]],[[0.,1.],[0.,1.]]])
        hook=RotationalSteeringHook(torch.tensor([1.,0.]),threshold=.5)
        output=hook(None,None,(x,'cache'))
        self.assertEqual(output[1],'cache')
        self.assertTrue(torch.equal(output[0][:,0],x[:,0]))
        self.assertTrue(torch.equal(output[0][0],x[0]))
        self.assertFalse(torch.equal(output[0][1,-1],x[1,-1]))
        self.assertTrue(torch.equal(x[1,-1],torch.tensor([0.,1.])))
        self.assertTrue(torch.allclose(output[0].norm(dim=-1),x.norm(dim=-1),atol=1e-6))
        module=torch.nn.Identity()
        with self.assertRaisesRegex(RuntimeError,'test'):
            with SteeringContext(module,hook):
                self.assertEqual(len(module._forward_hooks),1)
                raise RuntimeError('test')
        self.assertEqual(len(module._forward_hooks),0)

    def test_slerp_edge_norms(self):
        torch.manual_seed(20261030)
        h=torch.randn(3,4,16);v=torch.randn(16)
        self.assertTrue(torch.allclose(slerp_rotate(h,v,.25).norm(dim=-1),h.norm(dim=-1),atol=1e-5))
        self.assertTrue(torch.allclose(slerp_rotate(h,v,0),h,atol=1e-5))


class TinyLlavaIntegration(unittest.TestCase):
    def test_random_weight_tiny_model_no_download(self):
        torch.set_num_threads(1)
        from transformers import LlavaConfig, LlavaForConditionalGeneration, LlamaConfig, CLIPVisionConfig
        from transformers.feature_extraction_utils import BatchFeature
        from src.model_loader import find_language_decoder_layers
        from src.probe import extract_prompt_terminal_state
        torch.manual_seed(20261030)
        config=LlavaConfig(text_config=LlamaConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2),vision_config=CLIPVisionConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,image_size=16,patch_size=8),image_token_index=63,image_seq_length=4,vision_feature_layer=-1)
        model=LlavaForConditionalGeneration(config).eval()
        payload=dict(input_ids=torch.tensor([[1,63,63,63,63,2,3]]),pixel_values=torch.randn(1,3,16,16),attention_mask=torch.ones(1,7,dtype=torch.long))
        class Processor:
            def __call__(self,**kwargs):return BatchFeature(dict(payload))
        states=extract_prompt_terminal_state(model,Processor(),object(),'test')
        self.assertEqual(tuple(states.shape),(2,32))
        layers=find_language_decoder_layers(model)
        self.assertTrue(all(len(layer._forward_hooks)==0 for layer in layers))
        with SteeringContext(layers[0],RotationalSteeringHook(torch.ones(32),mode='constant')):
            changed=extract_prompt_terminal_state(model,Processor(),object(),'test')
            with torch.inference_mode():
                answer=model.generate(**payload,max_new_tokens=2,do_sample=False,pad_token_id=0)
        self.assertFalse(torch.allclose(states,changed))
        self.assertEqual(answer.shape[1],9)


if __name__=='__main__':unittest.main()
