"""Headless report generation must not inherit Colab's inline backend."""
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

class ReportBackendTests(unittest.TestCase):
    def test_figures_with_inherited_colab_backend(self):
        from scripts.make_report import figures
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'MPLBACKEND': 'module://matplotlib_inline.backend_inline'}):
                result=figures(SimpleNamespace(dir=Path(tmp)),Path(tmp)/'figures',None,None)
                self.assertEqual(result,[])
                import matplotlib
                self.assertEqual(matplotlib.get_backend().lower(),'agg')

if __name__=='__main__':unittest.main()
