#!/usr/bin/env python3
"""训练方案一：什么也不改（与 wire 配置原文一致：随机缩放+裁剪、验证 argmax）。

用法（在 ``EDAFormer-main/EDAFormer-main`` 下）::

    python tools/train_wire_scheme1.py dataa
    python tools/train_wire_scheme1.py datab -- --seed 0 --gpus 1

第一个参数必须是 ``dataa`` 或 ``datab``；其后参数原样传给 ``tools/train.py``。
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
    if len(sys.argv) < 2 or sys.argv[1] not in ('dataa', 'datab'):
        print(
            '用法: python tools/train_wire_scheme1.py <dataa|datab> [train.py 参数...]',
            file=sys.stderr)
        sys.exit(2)
    ds = sys.argv[1]
    rest = sys.argv[2:]
    root = edaformer_project_root()
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    cmd = [
        sys.executable,
        osp.join(root, 'tools', 'train.py'),
        '--wire-scheme',
        ds,
        '--train-scheme',
        'legacy',
    ] + rest
    raise SystemExit(subprocess.call(cmd, cwd=root))


if __name__ == '__main__':
    main()
