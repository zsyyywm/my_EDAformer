# Copyright (c) OpenMMLab. All rights reserved.
"""电线课题路径：与 TransNeXt mask2former 一致，解析 DataA-B 与工程根。"""
import os
import os.path as osp


def edaformer_project_root():
    """``EDAFormer-main/EDAFormer-main``（含 ``tools/``、``mmseg/`` 的根）。"""
    return osp.abspath(osp.join(osp.dirname(__file__), '..'))


def resolve_data_ab_root():
    """返回含 ``DataA`` / ``DataB`` 子目录的 ``DataA-B`` 根路径。"""
    v = os.environ.get('WIRE_SEG_DATA_ROOT')
    if v and osp.isdir(v):
        return osp.abspath(v)
    root = edaformer_project_root()
    cur = root
    for _ in range(10):
        cand = osp.join(cur, 'DataA-B')
        if osp.isdir(cand):
            return cand
        parent = osp.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return osp.join(osp.dirname(root), 'DataA-B')


def dataa_root():
    """``WIRE_SEG_DATAA_ROOT`` 可直指 ``.../DataA``（与 reapear.md 一致）。"""
    v = os.environ.get('WIRE_SEG_DATAA_ROOT')
    if v and osp.isdir(v):
        return osp.abspath(v)
    return osp.join(resolve_data_ab_root(), 'DataA')


def datab_root():
    v = os.environ.get('WIRE_SEG_DATAB_ROOT')
    if v and osp.isdir(v):
        return osp.abspath(v)
    return osp.join(resolve_data_ab_root(), 'DataB')


# 数据集快捷入口：dataa / datab（``--wire-scheme``）；训练配方 legacy/fix512/fix256 见 ``TRAIN_SCHEMES``。
WIRE_SCHEMES = ('dataa', 'datab')
_WIRE_SCHEME_DEFAULT_CONFIG = {
    'dataa': 'local_configs/wire/edaformer_tiny_dataa_512x512_wire_iou.py',
    'datab': 'local_configs/wire/edaformer_tiny_datab_512x512_wire_iou.py',
}


def checkpoints1_dir():
    return osp.join(edaformer_project_root(), 'data', 'checkpoints1')


def default_config_for_wire_scheme(scheme):
    if scheme not in _WIRE_SCHEME_DEFAULT_CONFIG:
        raise ValueError(f'unknown wire scheme: {scheme!r}, expected one of {WIRE_SCHEMES}')
    return _WIRE_SCHEME_DEFAULT_CONFIG[scheme]


def data_root_for_wire_scheme(scheme):
    if scheme == 'dataa':
        return dataa_root()
    if scheme == 'datab':
        return datab_root()
    raise ValueError(scheme)


def resolve_train_config_path(config_arg):
    """``config_arg`` 为相对路径时，优先相对工程根，其次相对 cwd。"""
    if not config_arg:
        return None
    if osp.isabs(config_arg) and osp.isfile(config_arg):
        return osp.abspath(config_arg)
    root = edaformer_project_root()
    cand = osp.join(root, config_arg)
    if osp.isfile(cand):
        return cand
    cand2 = osp.abspath(config_arg)
    if osp.isfile(cand2):
        return cand2
    raise FileNotFoundError(
        f'找不到配置文件: {config_arg!r}（已尝试工程根下路径与当前目录）')


def merge_wire_scheme_into_cfg(cfg, scheme):
    """将 DataA 或 DataB 的 ``data_root`` 与 ``work_dir=data/checkpoints1`` 写入 cfg（可被后续 ``--options`` 覆盖）。"""
    dr = data_root_for_wire_scheme(scheme)
    if not osp.isdir(dr):
        raise FileNotFoundError(
            f'未找到 {scheme} 数据目录: {dr}；请放置数据或设置 WIRE_SEG_DATAA_ROOT / '
            f'WIRE_SEG_DATAB_ROOT / WIRE_SEG_DATA_ROOT')
    dr_norm = dr.replace('\\', '/')
    ck = checkpoints1_dir().replace('\\', '/')
    cfg.data.train.data_root = dr_norm
    cfg.data.val.data_root = dr_norm
    cfg.data.test.data_root = dr_norm
    cfg.work_dir = ck


# ---------------------------------------------------------------------------
# 三种「训练配方」（与数据 dataa/datab 正交）：原样 / 固定边长+前景阈值评测
# ---------------------------------------------------------------------------
TRAIN_SCHEMES = ('legacy', 'fix512_th055', 'fix256_th050')


def _extract_img_norm_cfg(cfg):
    """从 ``img_norm_cfg`` 或 train pipeline 里的 ``Normalize`` 取参数字典。"""
    raw = cfg.get('img_norm_cfg')
    if raw is not None:
        d = dict(raw) if hasattr(raw, 'items') else dict(raw)
        return {k: v for k, v in d.items() if k != 'type'}
    for step in cfg.data.train.pipeline:
        if isinstance(step, dict) and step.get('type') == 'Normalize':
            return {k: v for k, v in step.items() if k != 'type'}
    raise ValueError('apply_train_scheme: 需要配置顶层 img_norm_cfg 或 train pipeline 中含 Normalize')


def _fixed_train_pipeline(crop, img_norm_cfg):
    return [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=False),
        dict(type='Resize', img_scale=(crop, crop), keep_ratio=False),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    ]


def _fixed_test_pipeline(crop, img_norm_cfg):
    return [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=False),
        dict(type='Resize', img_scale=(crop, crop), keep_ratio=False),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    ]


def apply_train_scheme_to_cfg(cfg, train_scheme):
    """按课题三种训练配方改写 pipeline 与 ``model.test_cfg.fg_threshold``（验证/测试时生效）。

    - ``legacy``：不改配置（原随机缩放+裁剪与 argmax 评测）。
    - ``fix512_th055``：Resize 到 512×512（``keep_ratio=False``），前景概率阈值 0.55。
    - ``fix256_th050``：Resize 到 256×256，前景概率阈值 0.5。

    仅当 ``wire_seg_experiment`` 为真时允许非 legacy，以免误伤通用分割配置。
    """
    if train_scheme is None or train_scheme == 'legacy':
        return
    if train_scheme not in ('fix512_th055', 'fix256_th050'):
        raise ValueError(f'unknown train_scheme: {train_scheme!r}')
    if not cfg.get('wire_seg_experiment'):
        raise ValueError(
            '--train-scheme fix512_th055 / fix256_th050 仅用于 wire_seg_experiment 配置')
    if train_scheme == 'fix512_th055':
        crop, thr = 512, 0.55
    else:
        crop, thr = 256, 0.5
    img_norm_cfg = _extract_img_norm_cfg(cfg)
    tr = _fixed_train_pipeline(crop, img_norm_cfg)
    te = _fixed_test_pipeline(crop, img_norm_cfg)
    cfg.crop_size = (crop, crop)
    cfg.train_pipeline = tr
    cfg.data.train.pipeline = tr
    cfg.data.val.pipeline = te
    cfg.data.test.pipeline = te
    prev = cfg.model.get('test_cfg', None)
    base = dict(prev) if prev is not None else {}
    base.setdefault('mode', 'whole')
    base['fg_threshold'] = float(thr)
    # 须为 ConfigDict：EncoderDecoder.inference 使用 ``self.test_cfg.mode`` 属性访问
    from mmcv import ConfigDict  # 延迟导入，便于无 mmcv 环境仅 import wire_paths（如薄封装脚本）

    cfg.model.test_cfg = ConfigDict(base)
