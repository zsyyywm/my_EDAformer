#!/usr/bin/env python3
"""DataB 冒烟（等价于 ``smoke_test_wire_data.py --wire-scheme datab``）。"""
import os.path as osp
import subprocess
import sys

_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
cmd = [
    sys.executable,
    osp.join(osp.dirname(__file__), 'smoke_test_wire_data.py'),
    '--wire-scheme',
    'datab',
] + sys.argv[1:]
raise SystemExit(subprocess.call(cmd, cwd=_ROOT))
