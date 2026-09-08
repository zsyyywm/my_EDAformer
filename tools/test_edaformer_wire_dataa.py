#!/usr/bin/env python3
"""DataA 电线二分类测试入口；在 ``data/checkpoints1`` 下搜 ``best_IoU*.pth``；未传 ``--out`` 时
预测 pkl 写入 ``data/checkpoints2/test_<时间戳>/``（与训练 ``checkpoints1/train_*`` 对称）。

用法::

    python tools/test_edaformer_wire_dataa.py
    python tools/test_edaformer_wire_dataa.py /path/to/epoch_x.pth
    python tools/test_edaformer_wire_dataa.py --best --work-dir data/checkpoints1/train_20260101_120000
"""
import os
import os.path as osp
import subprocess
import sys

_TOOLS_DIR = osp.dirname(osp.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
from wire_paths import dataa_root, edaformer_project_root

CONFIG = 'local_configs/wire/edaformer_tiny_dataa_512x512_wire_iou.py'


def main():
    root = edaformer_project_root()
    os.chdir(root)

    dr = dataa_root()
    if not osp.isdir(dr):
        print(f'错误: 未找到 DataA 目录: {dr}', file=sys.stderr)
        sys.exit(1)

    dr_norm = dr.replace('\\', '/')
    ck_root = osp.join(root, 'data', 'checkpoints1').replace('\\', '/')

    argv = sys.argv[1:]
    extra = []
    if argv and not argv[0].startswith('-'):
        extra.append(argv[0])
        argv = argv[1:]
    elif '--best' in argv:
        pass
    else:
        extra.append('--best')

    if '--work-dir' not in argv and '-w' not in argv:
        extra.extend(['--work-dir', ck_root])

    cmd = [
        sys.executable,
        osp.join(root, 'tools', 'test.py'),
        CONFIG,
        *extra,
        '--eval',
        'mIoU',
        '--options',
        f'data.test.data_root={dr_norm}',
        f'work_dir={ck_root}',
    ]
    cmd.extend(argv)
    raise SystemExit(subprocess.call(cmd, cwd=root))


if __name__ == '__main__':
    main()
