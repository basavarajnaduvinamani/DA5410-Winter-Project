"""Run the CPU suite and save exact code/config/seed/commit provenance."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

os.environ['USE_TF']='0'
os.environ['USE_FLAX']='0'
os.environ['OMP_NUM_THREADS']='1'


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    notebook=json.loads(Path('notebooks/CounterPrior_Mechanistic_Steering.ipynb').read_text())
    cells=[dict(cell=i,statements=len(ast.parse(''.join(c['source'])).body)) for i,c in enumerate(notebook['cells']) if c['cell_type']=='code']
    suite=unittest.defaultTestLoader.discover('tests')
    with (out/'tests.log').open('x',encoding='utf8') as f:
        result=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
    paths=[*Path('src').glob('*.py'),*Path('scripts').glob('*.py'),*Path('tests').glob('*.py'),Path('configs/llava.json'),Path('requirements-colab.lock'),Path('notebooks/CounterPrior_Mechanistic_Steering.ipynb')]
    record=dict(command=[sys.executable,*sys.argv],commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                seed=20261030,config=json.loads(Path('configs/llava.json').read_text()),
                test_count=result.testsRun,failures=len(result.failures),errors=len(result.errors),success=result.wasSuccessful(),
                scope='CPU unit tests and randomly initialized tiny LLaVA; no pretrained VLM metrics',
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                notebook_code_cells=cells,
                packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()})
    (out/'verification.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:record[k] for k in ['commit','test_count','failures','errors','success','scope']}))
    if not result.wasSuccessful():sys.exit(1)


if __name__=='__main__':main()
