import argparse
import copy
import os
import os.path as osp
import shutil
import sys
import time
import warnings

_TOOLS_DIR = osp.dirname(osp.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import mmcv
import torch
from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument(
        'config',
        nargs='?',
        default=None,
        help='配置文件路径。方案三（与 MambaVision 一致）：必传；方案一/二：与 '
        '--wire-scheme 联用时可省略，将使用默认 dataa/datab 的 wire 配置')
    parser.add_argument(
        '--wire-scheme',
        choices=['dataa', 'datab'],
        default=None,
        help='dataa / datab：自动注入对应 data_root 与 work_dir=data/checkpoints1；'
        '不设此项则须显式传入 config 路径（与 MambaVision 一致）')
    parser.add_argument(
        '--train-scheme',
        choices=['legacy', 'fix512_th055', 'fix256_th050'],
        default='legacy',
        help='课题三种训练配方：legacy=配置原样；fix512_th055=固定512输入+前景阈值0.55；'
        'fix256_th050=固定256输入+前景阈值0.5（仅 wire_seg_experiment）')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():

    args = parse_args()

    if args.wire_scheme:
        from wire_paths import (
            default_config_for_wire_scheme,
            merge_wire_scheme_into_cfg,
            resolve_train_config_path,
        )
        rel_or_abs = args.config or default_config_for_wire_scheme(
            args.wire_scheme)
        config_path = resolve_train_config_path(rel_or_abs)
    else:
        if not args.config:
            print(
                '错误: 请传入配置文件，或使用 --wire-scheme dataa|datab',
                file=sys.stderr)
            sys.exit(2)
        from wire_paths import resolve_train_config_path

        config_path = resolve_train_config_path(args.config)

    cfg = Config.fromfile(config_path)
    if args.wire_scheme:
        from wire_paths import merge_wire_scheme_into_cfg

        merge_wire_scheme_into_cfg(cfg, args.wire_scheme)

    from wire_paths import apply_train_scheme_to_cfg

    apply_train_scheme_to_cfg(cfg, args.train_scheme)

    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(config_path))[0])

    # 电线二分类课题：注册模块；单次实验子目录 train_<时间戳>（与 TransNeXt 文档一致）
    if cfg.get('wire_seg_experiment'):
        import mmseg.core.hooks.wire_seg_hooks  # noqa: F401
        import mmseg.datasets.wire_binary  # noqa: F401
        _ts = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        cfg.timestamp = f'train_{_ts}'
        base = cfg.work_dir.rstrip(osp.sep)
        if not base.endswith(cfg.timestamp):
            cfg.work_dir = osp.join(base, cfg.timestamp)
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)


    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config（mmcv 1.2.x 依赖旧版 yapf 的 FormatCode(..., verify=)；新版 yapf 会报错）
    _cfg_dst = osp.join(cfg.work_dir, osp.basename(config_path))
    try:
        cfg.dump(_cfg_dst)
    except TypeError as e:
        if 'verify' in str(e) or 'FormatCode' in str(e):
            shutil.copy2(config_path, _cfg_dst)
            warnings.warn(
                'cfg.dump() 因 yapf 版本不兼容失败，已改为复制原始配置文件到 work_dir；'
                '若需完整展开后的配置文本，请执行: pip install "yapf==0.31.0"',
                UserWarning,
                stacklevel=2)
        else:
            raise
    # init the logger：电线课题与 TransNeXt 一致，日志文件名与 train_<时间戳> 目录对应
    if cfg.get('wire_seg_experiment') and getattr(cfg, 'timestamp', None):
        timestamp = str(cfg.timestamp).strip()
        log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    else:
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    if cfg.get('wire_seg_experiment') and getattr(cfg, 'timestamp', None):
        _which = 'DataA/DataB 电线二分类'
        if getattr(args, 'wire_scheme', None):
            _which = 'DataA' if args.wire_scheme == 'dataa' else 'DataB'
        logger.info(
            '本次 run 主输出目录（日志、val_metrics.csv、按 IoU 的 best 权重、曲线图）:\n'
            f'  {osp.abspath(cfg.work_dir)}\n'
            f'  {_which} 下 IoU = 前景类 IoU（0–1 存盘，终端与 CSV 中百分比为×100）。')

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: '
                    f'{args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(config_path)

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    logger.info(model)

    datasets = [build_dataset(cfg.data.train)]

    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
