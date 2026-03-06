from . import ppo_roa

import importlib.util
import sys
from pathlib import Path

_ppo_roa_test_path = Path(__file__).with_name("ppo_roa-test.py")
if _ppo_roa_test_path.exists():
    _module_name = f"{__name__}.ppo_roa_test"
    if _module_name not in sys.modules:
        _spec = importlib.util.spec_from_file_location(_module_name, _ppo_roa_test_path)
        if _spec is not None and _spec.loader is not None:
            _module = importlib.util.module_from_spec(_spec)
            sys.modules[_module_name] = _module
            _spec.loader.exec_module(_module)
