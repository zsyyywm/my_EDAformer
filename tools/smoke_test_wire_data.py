#!/usr/bin/env python3
"""冒烟：仅构建 DataLoader / Dataset，不训练。

三种用法（与 ``train.py`` 对齐）::

    python tools/smoke_test_wire_data.py --wire-scheme dataa
    python tools/smoke_test_wire_data.py --wire-scheme datab
    python tools/smoke_test_wire_data.py local_configs/wire/edaformer_tiny_dataa_512x512_wire_iou.py
"""
import argparse
import os.path as osp
import sys

from mmcv.utils import Config

from mmseg.datasets import build_dataloader, build_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        'config',
        nargs='?',
        default=None,
        help='方案三：配置文件路径；方案一/二可省略（配合 --wire-scheme）')
    p.add_argument(
        '--wire-scheme',
        choices=('dataa', 'datab'),
        default=None,
        help='方案一=dataa、方案二=datab：自动数据根与默认 wire 配置')
    p.add_argument('--split', choices=('train', 'val', 'test'), default='train')
    return p.parse_args()


def main():
    args = parse_args()
    _tools = osp.dirname(osp.abspath(__file__))
    if _tools not in sys.path:
        sys.path.insert(0, _tools)
    root = osp.abspath(osp.join(_tools, '..'))
    if root not in sys.path:
        sys.path.insert(0, root)

    from wire_paths import (
        default_config_for_wire_scheme,
        merge_wire_scheme_into_cfg,
        resolve_train_config_path,
    )

    if args.wire_scheme:
        rel = args.config or default_config_for_wire_scheme(args.wire_scheme)
        cfg = Config.fromfile(resolve_train_config_path(rel))
        merge_wire_scheme_into_cfg(cfg, args.wire_scheme)
    elif args.config:
        cfg = Config.fromfile(resolve_train_config_path(args.config))
    else:
        print(
            '错误: 请传配置文件，或使用 --wire-scheme dataa|datab',
            file=sys.stderr)
        sys.exit(2)
    if getattr(cfg, 'wire_seg_experiment', False):
        import mmseg.datasets.wire_binary  # noqa: F401

    key = f'data.{args.split}'
    sub = cfg.data[args.split]
    ds = build_dataset(sub)
    print(f'[{args.split}] len={len(ds)} type={sub.type} root={sub.data_root}')

    loader = build_dataloader(
        ds,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False,
    )
    it = iter(loader)
    batch = next(it)
    print('batch keys:', list(batch.keys()))
    if 'img' in batch:
        print('img shape:', batch['img'].data[0].shape)
    print('smoke_test_wire_data: OK')


if __name__ == '__main__':
    main()
