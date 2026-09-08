#!/usr/bin/env python3
"""DataB 电线二分类训练入口（等价于 ``train.py --wire-scheme datab``）。

用法::

    python tools/train_edaformer_wire_datab.py
    python tools/train_edaformer_wire_datab.py -- --seed 0 --gpus 1

``--`` 之后参数原样传给 ``tools/train.py``。
"""
import os
import os.path as osp
import subprocess
import sys

_TOOLS_DIR = osp.dirname(osp.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
from wire_paths import edaformer_project_root


def main():
    root = edaformer_project_root()
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)

    cmd = [
        sys.executable,
        osp.join(root, 'tools', 'train.py'),
        '--wire-scheme',
        'datab',
    ]
    if '--' in sys.argv:
        i = sys.argv.index('--')
        cmd.extend(sys.argv[i + 1:])
    else:
        cmd.extend([a for a in sys.argv[1:] if a != '--'])

    raise SystemExit(subprocess.call(cmd, cwd=root))


if __name__ == '__main__':
    main()
