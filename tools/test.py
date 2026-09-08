import argparse
import glob
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor


def _collect_best_checkpoints(search_root):
    if not search_root or not osp.isdir(search_root):
        return []
    hits = []
    patterns = ('best_IoU*.pth', 'best_mIoU*.pth', 'best_*.pth')
    for pat in patterns:
        hits.extend(glob.glob(osp.join(search_root, pat)))
    for sub in glob.glob(osp.join(search_root, '*')):
        if osp.isdir(sub):
            for pat in patterns:
                hits.extend(glob.glob(osp.join(sub, pat)))
    out = []
    seen = set()
    for p in hits:
        p = osp.abspath(p)
        if osp.isfile(p) and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def find_best_checkpoint_path(cfg, work_dir_override=None):
    roots = []
    wd = work_dir_override or cfg.get('work_dir')
    if wd:
        roots.append(osp.abspath(wd))
    hits = []
    for d in roots:
        hits.extend(_collect_best_checkpoints(d))
    iouish = [p for p in hits if 'IoU' in osp.basename(p)]
    hits = iouish if iouish else hits
    if not hits:
        raise FileNotFoundError(
            f'未找到 best 权重（best_IoU*.pth 等）。已搜索: {roots}')
    hits.sort(key=lambda p: osp.getmtime(p), reverse=True)
    return hits[0]


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'checkpoint',
        nargs='?',
        default=None,
        help='checkpoint .pth；与 --best 二选一')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', default='work_dirs/res.pkl', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default='mIoU',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--backbone_reduction_ratios', default=None, type=list, help='reduction ratios for backbone')
    parser.add_argument('--decoder_reduction_ratios', default=None, type=list, help='reduction ratios for decoder')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--best',
        action='store_true',
        help='在 --work-dir 或 cfg.work_dir 下自动选用最新的 best_IoU*.pth')
    parser.add_argument(
        '--work-dir',
        default=None,
        help='配合 --best：指定含 train_<时间戳> 的根或单次实验目录')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if not args.best and not args.checkpoint:
        parser.error('请提供 checkpoint 路径，或使用 --best')
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if 'None' in args.eval:
        args.eval = None
    if args.eval and args.format_only:

        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if getattr(cfg, 'wire_seg_experiment', False):
        import mmseg.datasets.wire_binary  # noqa: F401
    if args.best:
        wd = args.work_dir
        if wd is None and getattr(cfg, 'wire_seg_experiment', False):
            _tools = osp.dirname(osp.abspath(__file__))
            wd = osp.join(osp.dirname(_tools), 'data', 'checkpoints1')
        args.checkpoint = find_best_checkpoint_path(cfg, wd)

    # 电线实验：默认 pkl 与训练对称——训练在 data/checkpoints1/train_*，测试在 data/checkpoints2/test_*
    _default_test_out = 'work_dirs/res.pkl'
    if getattr(cfg, 'wire_seg_experiment', False) and args.out and (
            osp.normpath(args.out) == osp.normpath(_default_test_out)):
        _tools = osp.dirname(osp.abspath(__file__))
        _repo = osp.dirname(_tools)
        _ts = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        test_run_dir = osp.join(_repo, 'data', 'checkpoints2', f'test_{_ts}')
        mmcv.mkdir_or_exist(test_run_dir)
        args.out = osp.join(test_run_dir, 'seg_results.pkl')
        cfg.work_dir = test_run_dir

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test and len(cfg.data.test.pipeline) > 1 and cfg.data.test.pipeline[
            1].get('type') == 'MultiScaleFlipAug':
        if cfg.data.test.type == 'CityscapesDataset':
            cfg.data.test.pipeline[1].img_ratios = [
                0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
            ]
            cfg.data.test.pipeline[1].flip = True
        elif cfg.data.test.type == 'ADE20KDataset':
            cfg.data.test.pipeline[1].img_ratios = [
                0.75, 0.875, 1.0, 1.125, 1.25
            ]
            cfg.data.test.pipeline[1].flip = True
        else:
            cfg.data.test.pipeline[1].img_ratios = [
                0.5, 0.75, 1.0, 1.25, 1.5, 1.75
            ]
            cfg.data.test.pipeline[1].flip = True

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    if 'EFT' in cfg.model.backbone.type and args.backbone_reduction_ratios is not None:
        cfg.model.backbone.reduction_ratios[0] = int(args.backbone_reduction_ratios[0])
        cfg.model.backbone.reduction_ratios[1] = int(args.backbone_reduction_ratios[1])
        cfg.model.backbone.reduction_ratios[2] = int(args.backbone_reduction_ratios[2])
        cfg.model.backbone.reduction_ratios[3] = int(args.backbone_reduction_ratios[3])

    if 'EDAFormer' in cfg.model.decode_head.type and args.decoder_reduction_ratios is not None:
        cfg.model.decode_head.reduction_ratios[0] = int(args.decoder_reduction_ratios[0])
        cfg.model.decode_head.reduction_ratios[1] = int(args.decoder_reduction_ratios[1])
        cfg.model.decode_head.reduction_ratios[2] = int(args.decoder_reduction_ratios[2])

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    meta = checkpoint.get('meta') or {}
    model.CLASSES = meta.get('CLASSES', dataset.CLASSES)
    model.PALETTE = meta.get('PALETTE', getattr(dataset, 'PALETTE', None))

    efficient_test = True #False
    if args.eval_options is not None:
        efficient_test = args.eval_options.get('efficient_test', False)

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  efficient_test)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect, efficient_test)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            _out_parent = osp.dirname(osp.abspath(args.out))
            if _out_parent:
                mmcv.mkdir_or_exist(_out_parent)
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else dict(args.eval_options)
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            ev_kwargs = dict(kwargs)
            wire_dump_dir = None
            if getattr(cfg, 'wire_seg_experiment', False) and (
                    type(dataset).__name__ == 'WireBinaryDataset'):
                if args.out:
                    wire_dump_dir = osp.dirname(osp.abspath(args.out))
                elif cfg.get('work_dir'):
                    wire_dump_dir = osp.abspath(cfg.work_dir)
                if wire_dump_dir is not None:
                    ev_kwargs['result_dump_dir'] = wire_dump_dir
            dataset.evaluate(outputs, args.eval, **ev_kwargs)
            if wire_dump_dir is not None:
                meta = {
                    'eval_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                    'checkpoint': osp.abspath(args.checkpoint)
                    if args.checkpoint else None,
                    'config': osp.abspath(args.config)
                    if osp.isabs(args.config) else osp.abspath(
                        osp.join(os.getcwd(), args.config)),
                }
                mmcv.mkdir_or_exist(wire_dump_dir)
                mmcv.dump(meta, osp.join(wire_dump_dir, 'test_meta.json'),
                          file_format='json')
                print(f'Wire eval artifacts: {wire_dump_dir}/'
                      f'eval_metrics.json, eval_report.txt, test_meta.json')


if __name__ == '__main__':
    main()
