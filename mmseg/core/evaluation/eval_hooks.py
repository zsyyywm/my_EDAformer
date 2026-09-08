# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import warnings
from math import inf

from mmcv.runner import Hook
from torch.utils.data import DataLoader

try:
    from mmcv.utils import is_seq_of
except ImportError:  # very old mmcv
    def is_seq_of(seq, expected_type):
        return (isinstance(seq, (list, tuple))
                and all(isinstance(item, expected_type) for item in seq))


class EvalHook(Hook):
    """Evaluation hook（与 OpenMMLab 新版语义对齐：可选 ``save_best`` / ``rule`` / ``out_dir``）。

    兼容旧调用：``EvalHook(dataloader, interval=1, by_epoch=False, metric='mIoU')``。
    """

    rule_map = {'greater': lambda x, y: x > y, 'less': lambda x, y: x < y}
    init_value_map = {'greater': -inf, 'less': inf}
    _default_greater_keys = [
        'acc', 'top', 'AR@', 'auc', 'precision', 'mAP', 'mDice', 'mIoU',
        'mAcc', 'aAcc', 'IoU', 'F1', 'Recall',
    ]
    _default_less_keys = ['loss']

    def __init__(self,
                 dataloader,
                 interval=1,
                 by_epoch=False,
                 save_best=None,
                 rule=None,
                 out_dir=None,
                 greater_keys=None,
                 less_keys=None,
                 **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got '
                            f'{type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.by_epoch = by_epoch

        assert isinstance(save_best, str) or save_best is None, (
            f'save_best should be str or None, got {type(save_best)}')
        self.save_best = save_best
        self.out_dir = out_dir
        self.eval_kwargs = eval_kwargs

        if greater_keys is None:
            self.greater_keys = self._default_greater_keys
        else:
            if not isinstance(greater_keys, (list, tuple)):
                greater_keys = (greater_keys, )
            assert is_seq_of(greater_keys, str)
            self.greater_keys = greater_keys

        if less_keys is None:
            self.less_keys = self._default_less_keys
        else:
            if not isinstance(less_keys, (list, tuple)):
                less_keys = (less_keys, )
            assert is_seq_of(less_keys, str)
            self.less_keys = less_keys

        if self.save_best is not None:
            self.best_ckpt_path = None
            self._init_rule(rule, self.save_best)
        else:
            self.rule = None
            self.key_indicator = None
            self.compare_func = None

        self.initial_flag = True

    def _init_rule(self, rule, key_indicator):
        if rule is not None and rule not in self.rule_map:
            raise KeyError(f'rule must be greater, less or None, but got {rule}.')

        if key_indicator == 'auto':
            self.rule = rule or 'greater'
            self.key_indicator = 'auto'
            self.compare_func = self.rule_map[self.rule]
            return

        if rule is None and key_indicator != 'auto':
            key_indicator_lc = key_indicator.lower()
            greater_keys = [k.lower() for k in self.greater_keys]
            less_keys = [k.lower() for k in self.less_keys]
            if key_indicator_lc in greater_keys:
                rule = 'greater'
            elif key_indicator_lc in less_keys:
                rule = 'less'
            elif any(k in key_indicator_lc for k in greater_keys):
                rule = 'greater'
            elif any(k in key_indicator_lc for k in less_keys):
                rule = 'less'
            else:
                raise ValueError(
                    f'Cannot infer rule for key {key_indicator}; '
                    f'specify evaluation.rule explicitly.')

        self.rule = rule
        self.key_indicator = key_indicator
        if self.rule is not None:
            self.compare_func = self.rule_map[self.rule]

    def before_run(self, runner):
        if self.out_dir is None:
            self.out_dir = runner.work_dir
        else:
            basename = osp.basename(runner.work_dir.rstrip(osp.sep))
            if basename:
                self.out_dir = osp.join(self.out_dir, basename)
        os.makedirs(self.out_dir, exist_ok=True)

        if self.save_best is not None:
            if runner.meta is None:
                warnings.warn('runner.meta is None, creating empty dict.')
                runner.meta = dict()
            runner.meta.setdefault('hook_msgs', dict())
            self.best_ckpt_path = runner.meta['hook_msgs'].get(
                'best_ckpt', None)

    def _should_evaluate(self, runner):
        if self.by_epoch:
            return self.every_n_epochs(runner, self.interval)
        return self.every_n_iters(runner, self.interval)

    def _save_ckpt(self, runner, key_score):
        if self.save_best is None or key_score is None:
            return

        if self.by_epoch:
            current = f'epoch_{runner.epoch + 1}'
            cur_type, cur_time = 'epoch', runner.epoch + 1
        else:
            current = f'iter_{runner.iter + 1}'
            cur_type, cur_time = 'iter', runner.iter + 1

        best_score = runner.meta['hook_msgs'].get(
            'best_score', self.init_value_map[self.rule])
        if not self.compare_func(key_score, best_score):
            return

        best_score = key_score
        runner.meta['hook_msgs']['best_score'] = best_score

        if self.best_ckpt_path and osp.isfile(self.best_ckpt_path):
            try:
                os.remove(self.best_ckpt_path)
            except OSError:
                pass
            runner.logger.info(
                f'Removed previous best checkpoint {self.best_ckpt_path}')

        # mmcv 1.2: save_checkpoint(out_dir, filename_tmpl) 且 filename_tmpl 须含「{{}}」占位
        tmpl = f'best_{self.key_indicator}_e{{}}.pth'
        runner.save_checkpoint(
            self.out_dir,
            filename_tmpl=tmpl,
            save_optimizer=True,
            create_symlink=False)
        fname = tmpl.format(runner.epoch + 1)
        self.best_ckpt_path = osp.join(self.out_dir, fname)
        runner.meta['hook_msgs']['best_ckpt'] = self.best_ckpt_path
        runner.logger.info(f'Best checkpoint saved as {fname}.')
        runner.logger.info(
            f'Best {self.key_indicator} is {best_score:0.4f} '
            f'at {cur_time} {cur_type}.')

    def after_train_iter(self, runner):
        if self.by_epoch or not self._should_evaluate(runner):
            return
        from mmseg.apis import single_gpu_test
        runner.log_buffer.clear()
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        key_score = self.evaluate(runner, results)
        if self.save_best:
            self._save_ckpt(runner, key_score)

    def after_train_epoch(self, runner):
        if not self.by_epoch or not self._should_evaluate(runner):
            return
        from mmseg.apis import single_gpu_test
        runner.log_buffer.clear()
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        key_score = self.evaluate(runner, results)
        if self.save_best:
            self._save_ckpt(runner, key_score)

    def evaluate(self, runner, results):
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True

        if self.save_best is not None:
            if self.key_indicator == 'auto':
                self._init_rule(self.rule, list(eval_res.keys())[0])
            return eval_res[self.key_indicator]
        return None


class DistEvalHook(EvalHook):
    """Distributed evaluation hook."""

    def __init__(self,
                 dataloader,
                 interval=1,
                 gpu_collect=False,
                 by_epoch=False,
                 tmpdir=None,
                 save_best=None,
                 rule=None,
                 out_dir=None,
                 greater_keys=None,
                 less_keys=None,
                 **eval_kwargs):
        super().__init__(
            dataloader,
            interval=interval,
            by_epoch=by_epoch,
            save_best=save_best,
            rule=rule,
            out_dir=out_dir,
            greater_keys=greater_keys,
            less_keys=less_keys,
            **eval_kwargs)
        self.gpu_collect = gpu_collect
        self.tmpdir = tmpdir

    def after_train_iter(self, runner):
        if self.by_epoch or not self._should_evaluate(runner):
            return
        from mmseg.apis import multi_gpu_test
        runner.log_buffer.clear()
        tmpdir = self.tmpdir or osp.join(runner.work_dir, '.eval_hook')
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            key_score = self.evaluate(runner, results)
            if self.save_best:
                self._save_ckpt(runner, key_score)

    def after_train_epoch(self, runner):
        if not self.by_epoch or not self._should_evaluate(runner):
            return
        from mmseg.apis import multi_gpu_test
        runner.log_buffer.clear()
        tmpdir = self.tmpdir or osp.join(runner.work_dir, '.eval_hook')
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            key_score = self.evaluate(runner, results)
            if self.save_best:
                self._save_ckpt(runner, key_score)
