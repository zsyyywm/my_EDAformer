# Copyright (c) OpenMMLab. All rights reserved.
"""与 TransNeXt ``mask2former/train.py`` 课题对齐（旧 mmcv Runner 等价物）。

Hook 优先级（数值越小越早执行）：Epoch 横幅 → 内置 Eval → 控制台表 → 早停 → 曲线/CSV。
"""
import csv
import logging
import numbers
import os
import os.path as osp
import sys

import torch
from mmcv.runner import HOOKS, Hook
from mmcv.runner.hooks.logger.text import TextLoggerHook
from mmcv.runner import get_dist_info


def _is_main():
    rank, _ = get_dist_info()
    return rank == 0


def _wire_ansi(text, color):
    """与 TransNeXt ``mask2former/train.py`` 中 ``ConsoleSummaryHook._c`` 一致。"""
    colors = {
        'blue': '\033[94m',
        'cyan': '\033[96m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'magenta': '\033[95m',
        'bold': '\033[1m',
        'white': '\033[97m',
    }
    return f"{colors.get(color, '')}{text}\033[0m"


def _wire_cell(text, width, color):
    """与 TransNeXt ``ConsoleSummaryHook._cell`` 一致。"""
    s = str(text)
    if len(s) > width:
        s = s[: max(1, width - 2)] + '..'
    s = s.ljust(width)
    return _wire_ansi(s, color)


def _wire_unwrap_segmentor(runner):
    m = getattr(runner, 'model', None)
    if m is None:
        return None
    return m.module if hasattr(m, 'module') else m


def _wire_last_img_hw_str(runner):
    """从 BaseSegmentor.train_step 写入的 ``_wire_last_img_hw`` 取当前步输入尺寸。"""
    seg = _wire_unwrap_segmentor(runner)
    if seg is None:
        return '?'
    hw = getattr(seg, '_wire_last_img_hw', None)
    if isinstance(hw, (tuple, list)) and len(hw) == 2:
        return f'{int(hw[0])}x{int(hw[1])}'
    return '?'


_MAX_REASONABLE_EPOCH_LEN = 100000


@HOOKS.register_module()
class WireEpochBannerHook(Hook):
    """每轮训练结束、验证开始前打印一行（对齐 ``ConsoleSummaryHook.after_train_epoch`` 语义）。

    优先级由配置里 ``priority`` 传入（须 ``register_hook_from_cfg``），勿在类上定义 ``priority``（mmcv Runner 会报错）。
    """

    def after_train_epoch(self, runner):
        if not _is_main():
            return
        me = getattr(runner, '_max_epochs', None)
        ep_done = runner.epoch + 1
        ep_tag = f'{ep_done}/{me}' if me is not None else str(ep_done)
        it = runner.iter + 1
        loss_s, lr_s = '?', '?'
        outs = getattr(runner, 'outputs', None) or {}
        lv = outs.get('log_vars') if isinstance(outs, dict) else None
        if isinstance(lv, dict):
            if 'loss' in lv:
                try:
                    loss_s = f'{float(lv["loss"]):.6f}'
                except (TypeError, ValueError, KeyError):
                    pass
        if runner.optimizer is not None and len(runner.optimizer.param_groups):
            try:
                lr_s = f'{float(runner.optimizer.param_groups[0]["lr"]):.8f}'
            except (TypeError, ValueError, KeyError, IndexError):
                pass
        # 与 TransNeXt ``ConsoleSummaryHook.after_train_epoch`` 逐字一致（仅 ``[Epoch]`` 上色）
        print(
            _wire_ansi('[Epoch]', 'magenta')
            + f' 第 {ep_tag} 轮训练阶段结束 | global_iter={it} | '
            f'loss≈{loss_s} | lr={lr_s} | 随后验证集…',
            flush=True)
        print(flush=True)


@HOOKS.register_module()
class WireTrainEpochEndBlockHook(Hook):
    """与 TransNeXt ``ConsoleSummaryHook._print_train_block`` 一致：epoch 末最后一 iter 后打表。

    须在 ``TextLoggerHook``（mmcv 中为 priority=VERY_LOW=90）之后执行，默认 ``priority=92``。
    """

    def before_train(self, runner):
        runner._wire_train_block_idx = 0

    def after_train_iter(self, runner):
        if not _is_main():
            return
        try:
            nb = len(runner.data_loader)
        except TypeError:
            return
        if nb <= 0 or nb >= _MAX_REASONABLE_EPOCH_LEN:
            return
        if runner._inner_iter + 1 != nb:
            return
        _tty = sys.stdout.isatty()
        _force = os.environ.get('WIRE_FORCE_COLOR', '').lower() in ('1', 'true', 'yes')
        if not (_tty or _force):
            return

        outs = getattr(runner, 'outputs', None) or {}
        lv = outs.get('log_vars') if isinstance(outs, dict) else None
        loss = None
        if isinstance(lv, dict) and 'loss' in lv:
            try:
                loss = float(lv['loss'])
            except (TypeError, ValueError, KeyError):
                pass
        lr = None
        if runner.optimizer is not None and len(runner.optimizer.param_groups):
            try:
                lr = float(runner.optimizer.param_groups[0]['lr'])
            except (TypeError, ValueError, KeyError, IndexError):
                pass
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / (1024**2)
        else:
            gpu_mem = 0.0

        me = getattr(runner, '_max_epochs', None)
        ep = runner.epoch + 1
        it = runner.iter + 1
        bi = runner._inner_iter + 1
        tags = [f'epoch={ep}' + (f'/{me}' if me is not None else '')]
        tags.append(f'batch={bi}/{nb}')
        tags.append(f'global_iter={it}')
        prog = ' | '.join(tags)

        loss_str = f'{loss:.8f}' if loss is not None else '?'
        lr_str = f'{lr:.8f}' if lr is not None else '?'
        dn = f'{bi}/{nb}'
        ep_lab = f'{ep}/{me}' if me is not None else str(ep)

        blk = getattr(runner, '_wire_train_block_idx', 0)
        if blk > 0:
            print(flush=True)
        print(_wire_ansi(f'[本轮训练结束] {prog}', 'cyan'), flush=True)

        w_ep, w_dn, w_mem, w_loss, w_lr, w_img = 10, 12, 14, 14, 14, 10
        row1 = (
            _wire_cell('Epoch', w_ep, 'green')
            + _wire_cell('data_num', w_dn, 'yellow')
            + _wire_cell('GPU Mem', w_mem, 'yellow')
            + _wire_cell('Loss', w_loss, 'yellow')
            + _wire_cell('LR', w_lr, 'yellow')
            + _wire_cell('Image_size', w_img, 'yellow'))
        img_sz = _wire_last_img_hw_str(runner)
        row2 = (
            _wire_cell(ep_lab, w_ep, 'bold')
            + _wire_cell(dn, w_dn, 'white')
            + _wire_cell(f'{gpu_mem:.2f} MB', w_mem, 'white')
            + _wire_cell(loss_str, w_loss, 'white')
            + _wire_cell(lr_str, w_lr, 'white')
            + _wire_cell(img_sz, w_img, 'white'))
        print(row1, flush=True)
        print(row2, flush=True)
        runner._wire_train_block_idx = blk + 1


@HOOKS.register_module()
class WireIoUPatienceEarlyStopHook(Hook):
    """对齐 ``ValLossPatienceEarlyStopHook`` / ``mask2former_iou_early_stop_patience``。"""

    def __init__(self, monitor='IoU', patience=50, rule='greater', min_delta=0.0):
        self.monitor = monitor
        self.patience = int(patience)
        self.rule = str(rule).lower()
        self.min_delta = float(min_delta)
        self._best = None
        self._bad_epochs = 0

    def after_train_epoch(self, runner):
        _rank, world_size = get_dist_info()
        if world_size > 1:
            if _rank == 0:
                runner.logger.warning(
                    'WireIoUPatienceEarlyStopHook: 多卡未做跨进程同步，已跳过早停；'
                    '与 TransNeXt 一致请单卡训练或后续扩展 broadcast。')
            return
        out = getattr(runner.log_buffer, 'output', None) or {}
        if self.monitor not in out:
            return
        try:
            cur = float(out[self.monitor])
        except (TypeError, ValueError):
            return
        if self._best is None:
            self._best = cur
            self._bad_epochs = 0
            return
        if self.rule == 'less':
            improved = cur < (self._best - self.min_delta)
        else:
            improved = cur > (self._best + self.min_delta)
        if improved:
            self._best = cur
            self._bad_epochs = 0
        else:
            self._bad_epochs += 1
        if self._bad_epochs < self.patience:
            return
        runner.logger.warning(
            f'WireIoUPatienceEarlyStopHook: 连续 {self.patience} 次验证 '
            f'「{self.monitor}」未优于当前最优 {self._best:.4f}，触发早停。')
        runner._max_epochs = runner.epoch + 1


@HOOKS.register_module()
class WireConsoleSummaryHook(Hook):
    """验证后打印前景指标表（与 TransNeXt ``ConsoleSummaryHook.after_val_epoch`` 一致：红表头 + 白数值）。"""

    @staticmethod
    def _pick(out, keys):
        for k in keys:
            if k in out:
                return out[k]
        for k in keys:
            for mk, mv in out.items():
                if isinstance(mk, str) and mk.endswith(k):
                    return mv
        return None

    @staticmethod
    def _fmt_pct(v):
        if not isinstance(v, numbers.Real):
            return 'N/A'
        x = float(v)
        if x <= 1.0 + 1e-6:
            x *= 100.0
        return f'{x:.2f}'

    def after_train_epoch(self, runner):
        if not _is_main():
            return
        out = getattr(runner.log_buffer, 'output', None) or {}
        if not isinstance(out, dict) or 'IoU' not in out:
            return
        ep = runner.epoch + 1
        me = getattr(runner, '_max_epochs', None)
        it = runner.iter + 1
        # TransNeXt ``_progress_tags``：验证时 ``epoch=…/… | global_iter=…``
        vtag = (
            f'epoch={ep}' + (f'/{me}' if me is not None else '')
            + f' | global_iter={it}')
        print(_wire_ansi(f'[验证] {vtag}', 'red'), flush=True)

        # 与 TransNeXt 相同列宽；表头红、数值行全白
        w_dn, w_iou, w_f1, w_p, w_r, w_a = 12, 12, 10, 12, 12, 10
        row_hdr = (
            _wire_cell('data_num', w_dn, 'red')
            + _wire_cell('IoU(fg)', w_iou, 'red')
            + _wire_cell('F1', w_f1, 'red')
            + _wire_cell('Precision', w_p, 'red')
            + _wire_cell('Recall', w_r, 'red')
            + _wire_cell('aAcc', w_a, 'red'))

        miou = self._pick(out, ['IoU', 'mIoU'])
        mf1 = self._pick(out, ['F1', 'mFscore', 'mF1'])
        mp = self._pick(out, ['Precision', 'mPrecision'])
        mr = self._pick(out, ['Recall', 'mRecall'])
        aacc = self._pick(out, ['aAcc'])

        # TransNeXt：``len(val_dataloader)``；mmcv 从 EvalHook.dataloader 取 batch 数
        val_n = None
        for attr in ('val_dataloader', '_val_dataloader'):
            dl = getattr(runner, attr, None)
            if dl is not None:
                try:
                    val_n = len(dl)
                    break
                except TypeError:
                    pass
        if val_n is None:
            for h in getattr(runner, 'hooks', []) or []:
                if h.__class__.__name__ not in ('EvalHook', 'DistEvalHook'):
                    continue
                dl = getattr(h, 'dataloader', None)
                if dl is None:
                    continue
                try:
                    val_n = len(dl)
                    break
                except TypeError:
                    pass
        dn_val = f'{val_n}/{val_n}' if val_n is not None else '?/?'

        row_val = (
            _wire_cell(dn_val, w_dn, 'white')
            + _wire_cell(self._fmt_pct(miou), w_iou, 'white')
            + _wire_cell(self._fmt_pct(mf1), w_f1, 'white')
            + _wire_cell(self._fmt_pct(mp), w_p, 'white')
            + _wire_cell(self._fmt_pct(mr), w_r, 'white')
            + _wire_cell(self._fmt_pct(aacc), w_a, 'white'))
        print(row_hdr, flush=True)
        print(row_val, flush=True)
        print(flush=True)


@HOOKS.register_module()
class WirePlotMetricsHook(Hook):
    """对齐 ``PlotMetricsHook``：采样训练 loss、写 ``val_metrics.csv``、``train_curves.png``、``val_foreground_trends.png``。"""

    def __init__(self, sample_interval=50):
        self.sample_interval = max(1, int(sample_interval))
        self._t_iters = []
        self._t_loss = []
        self._t_lr = []
        self._v_epoch = []
        self._v_step = []
        self._v_iou = []
        self._v_f1 = []
        self._v_precision = []
        self._v_recall = []

    def before_train(self, runner):
        self._t_iters.clear()
        self._t_loss.clear()
        self._t_lr.clear()
        self._v_epoch.clear()
        self._v_step.clear()
        self._v_iou.clear()
        self._v_f1.clear()
        self._v_precision.clear()
        self._v_recall.clear()

    def after_train_iter(self, runner):
        if not _is_main():
            return
        if (runner.iter + 1) % self.sample_interval != 0:
            return
        outs = getattr(runner, 'outputs', None) or {}
        lv = outs.get('log_vars') if isinstance(outs, dict) else None
        if not isinstance(lv, dict):
            return
        loss = lv.get('loss')
        it = runner.iter + 1
        self._t_iters.append(it)
        try:
            self._t_loss.append(float(loss) if loss is not None else float('nan'))
        except (TypeError, ValueError):
            self._t_loss.append(float('nan'))
        lr = float('nan')
        if runner.optimizer is not None and len(runner.optimizer.param_groups):
            try:
                lr = float(runner.optimizer.param_groups[0]['lr'])
            except (TypeError, ValueError, KeyError, IndexError):
                pass
        self._t_lr.append(lr)

    def _append_val_csv(self, runner, out):
        path = osp.join(runner.work_dir, 'val_metrics.csv')
        ep = runner.epoch + 1
        step = runner.iter + 1

        def _f(k):
            v = out.get(k)
            if not isinstance(v, numbers.Real):
                return ''
            x = float(v)
            if x <= 1.0 + 1e-6:
                x *= 100.0
            return round(x, 4)

        row = [
            ep,
            step,
            _f('IoU'),
            _f('F1'),
            _f('Precision'),
            _f('Recall'),
            _f('aAcc'),
            '',
        ]
        write_header = not osp.isfile(path)
        with open(path, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow([
                    'epoch', 'global_iter', 'IoU_fg', 'F1', 'Precision', 'Recall',
                    'aAcc', 'val_loss',
                ])
            w.writerow(row)

    def after_train_epoch(self, runner):
        if not _is_main():
            return
        out = getattr(runner.log_buffer, 'output', None) or {}
        if 'IoU' not in out:
            self._save_figure(runner)
            return
        step = runner.iter + 1
        ep = runner.epoch + 1

        def _pick_float(k):
            v = out.get(k)
            if isinstance(v, numbers.Real):
                return float(v)
            return float('nan')

        self._append_val_csv(runner, out)
        self._v_epoch.append(ep)
        self._v_step.append(step)
        self._v_iou.append(_pick_float('IoU'))
        self._v_f1.append(_pick_float('F1'))
        self._v_precision.append(_pick_float('Precision'))
        self._v_recall.append(_pick_float('Recall'))
        self._save_figure(runner)

    def after_run(self, runner):
        if _is_main():
            self._save_figure(runner)

    def _save_figure(self, runner):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            if _is_main():
                logging.getLogger('mmseg').warning(
                    'WirePlotMetricsHook: 未安装 matplotlib，跳过绘图。')
            return
        logd = runner.work_dir
        os.makedirs(logd, exist_ok=True)

        if self._t_iters:
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
            fig.suptitle('Train (sampled)', fontsize=11)
            if any(v == v for v in self._t_loss):
                axes[0].plot(self._t_iters, self._t_loss, '-', lw=1, alpha=0.9)
            axes[0].set_title('loss')
            axes[0].set_xlabel('global_iter')
            axes[0].grid(True, alpha=0.3)
            if any(v == v for v in self._t_lr):
                axes[1].plot(self._t_iters, self._t_lr, 'g-', lw=1, alpha=0.9)
            axes[1].set_title('lr')
            axes[1].set_xlabel('global_iter')
            axes[1].grid(True, alpha=0.3)
            plt.tight_layout()
            p1 = osp.join(logd, 'train_curves.png')
            plt.savefig(p1, dpi=150, bbox_inches='tight')
            plt.close()
            runner.logger.info(f'WirePlotMetricsHook: 已写入 {p1}')

        if self._v_step:
            fig2, ax2 = plt.subplots(2, 2, figsize=(10, 8))
            fig2.suptitle('Val: foreground IoU / F1 / Precision / Recall (0–1)',
                          fontsize=11)
            xsv = self._v_step
            ax2[0, 0].plot(xsv, self._v_iou, 'r-o', ms=3, lw=1)
            ax2[0, 0].set_title('IoU (fg)')
            ax2[0, 0].set_xlabel('global_iter @ val')
            ax2[0, 0].grid(True, alpha=0.3)
            ax2[0, 1].plot(xsv, self._v_f1, 'm-s', ms=3, lw=1)
            ax2[0, 1].set_title('F1 (fg)')
            ax2[0, 1].set_xlabel('global_iter @ val')
            ax2[0, 1].grid(True, alpha=0.3)
            ax2[1, 0].plot(xsv, self._v_precision, 'b-^', ms=3, lw=1)
            ax2[1, 0].set_title('Precision (fg)')
            ax2[1, 0].set_xlabel('global_iter @ val')
            ax2[1, 0].grid(True, alpha=0.3)
            ax2[1, 1].plot(xsv, self._v_recall, 'g-d', ms=3, lw=1)
            ax2[1, 1].set_title('Recall (fg)')
            ax2[1, 1].set_xlabel('global_iter @ val')
            ax2[1, 1].grid(True, alpha=0.3)
            plt.tight_layout()
            p2 = osp.join(logd, 'val_foreground_trends.png')
            plt.savefig(p2, dpi=150, bbox_inches='tight')
            plt.close()
            runner.logger.info(
                f'WirePlotMetricsHook: 已写入 {p2}；验证指标见 val_metrics.csv')


@HOOKS.register_module()
class WireCompactTextLoggerHook(TextLoggerHook):
    """电线课题：缩短 mmcv 默认超长 ``TextLoggerHook`` 行，并用 ``print`` 输出 TransNeXt 式彩色短行。

    默认 ``TextLoggerHook`` 在一行里拼 ``eta/time/data_time/decode.*``，在约 80 列终端上会被硬折行，
    出现 ``time:`` 被截成 ``t`` + 下一行 ``ime:`` 的错位。本 Hook 训练态只打短行；验证态仍走父类。
    """

    def _log_info(self, log_dict, runner):
        if log_dict.get('mode') != 'train':
            return super()._log_info(log_dict, runner)

        if runner.meta is not None and 'exp_name' in runner.meta:
            if (self.every_n_iters(runner, self.interval_exp_name)) or (
                    self.by_epoch and self.end_of_epoch(runner)):
                runner.logger.info(f'Exp name: {runner.meta["exp_name"]}')

        lr = log_dict['lr']
        if isinstance(lr, dict):
            lr = next(iter(lr.values()))
        cur_inner = log_dict['iter']
        ep = log_dict['epoch']
        try:
            nb = len(runner.data_loader)
        except TypeError:
            nb = '?'
        loss = log_dict.get('loss')
        acc = log_dict.get('decode.acc_seg')
        mem = log_dict.get('memory')
        loss_s = f'{float(loss):.4f}' if isinstance(loss, float) else '?'
        acc_s = f'{float(acc):.2f}' if isinstance(acc, float) else '?'
        lr_s = f'{float(lr):.3e}' if isinstance(lr, float) else str(lr)
        mem_s = str(int(mem)) if mem is not None else '?'
        plain = (
            f'Epoch [{ep}][{cur_inner}/{nb}] lr={lr_s} loss={loss_s} '
            f'dec_acc={acc_s} mem={mem_s}')
        if not _is_main():
            return
        _tty = sys.stdout.isatty()
        _force = os.environ.get('WIRE_FORCE_COLOR', '').lower() in ('1', 'true', 'yes')
        if _tty or _force:
            # 与 TransNeXt ``ConsoleSummaryHook.after_train_iter`` 单行格式一致
            me = getattr(runner, '_max_epochs', None)
            git = runner.iter + 1
            ep_part = f'epoch={ep}' + (f'/{me}' if me is not None else '')
            prog = f'{ep_part} | batch={cur_inner}/{nb} | global_iter={git}'
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / (1024**2)
            else:
                gpu_mem = 0.0
            loss_6 = (
                f'{float(loss):.6f}'
                if isinstance(loss, numbers.Real) and not isinstance(loss, bool)
                else '?')
            lr_8 = (
                f'{float(lr):.8f}'
                if isinstance(lr, numbers.Real) and not isinstance(lr, bool)
                else str(lr))
            img_sz = _wire_last_img_hw_str(runner)
            print(
                f'{_wire_ansi("[训练]", "green")} {prog} | '
                f'{_wire_ansi("batch进度", "yellow")} {cur_inner}/{nb} | '
                f'{_wire_ansi("GPU Mem", "magenta")} {gpu_mem:.2f} MB | '
                f'{_wire_ansi("Loss", "red")} {loss_6} | '
                f'{_wire_ansi("LR", "yellow")} {lr_8} | '
                f'{_wire_ansi("Img", "blue")} {img_sz}',
                flush=True)
        else:
            runner.logger.info(plain)
