# Copyright (c) OpenMMLab. All rights reserved.
"""电线二分类数据集：与 TransNeXt/Mamba 线一致的评测键（前景 IoU / F1 / P / R / aAcc）。"""
import json
import os
import os.path as osp
from collections import OrderedDict
from functools import reduce

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmseg.core.evaluation.metrics import total_intersect_and_union
from mmseg.datasets import CustomDataset
from mmseg.datasets.builder import DATASETS


@DATASETS.register_module()
class WireBinaryDataset(CustomDataset):
    """CustomDataset 布局 + 二分类前景指标。

    ``evaluate`` 返回标量均为 **0–1** 浮点，便于 ``EvalHook(save_best='IoU')`` 比较。
    """

    def prepare_test_img(self, idx):
        """与 ``prepare_train_img`` 一致带上 ``ann_info``。

        电线配置的 val/test pipeline 含 ``LoadAnnotations``（与 TransNeXt 对齐需 gt），
        而基类 ``prepare_test_img`` 只传 ``img_info``，会在 worker 里触发 ``KeyError: ann_info``。
        """
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        # ``Collect`` 默认 meta_keys 含 flip；无 RandomFlip 的 test/val pipeline 须先占位
        results.setdefault('flip', False)
        results.setdefault('flip_direction', 'horizontal')
        return self.pipeline(results)

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        if isinstance(metric, str):
            metric = [metric]

        result_dump_dir = kwargs.pop('result_dump_dir', None)
        if result_dump_dir:
            mmcv.mkdir_or_exist(result_dump_dir)

        gt_seg_maps = self.get_gt_seg_maps(efficient_test)
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)

        total_area_intersect, total_area_union, total_area_pred_label, \
            total_area_label = total_intersect_and_union(
                results,
                gt_seg_maps,
                num_classes,
                self.ignore_index,
                self.label_map or {},
                self.reduce_zero_label)

        def _nan_div(a, b):
            with np.errstate(divide='ignore', invalid='ignore'):
                out = np.divide(a.astype(np.float64), b.astype(np.float64))
            return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        iou = _nan_div(total_area_intersect, total_area_union)
        acc = _nan_div(total_area_intersect, total_area_label)
        precision = _nan_div(total_area_intersect, total_area_pred_label)
        recall = _nan_div(total_area_intersect, total_area_label)
        denom = precision + recall
        f1 = np.where(
            denom > 1e-12, 2.0 * precision * recall / denom, 0.0)

        all_acc = float(
            np.sum(total_area_intersect) / max(np.sum(total_area_label), 1))

        fg = 1 if num_classes > 1 else 0
        fg_iou = float(np.nan_to_num(iou[fg], nan=0.0))
        fg_f1 = float(np.nan_to_num(f1[fg], nan=0.0))
        fg_p = float(np.nan_to_num(precision[fg], nan=0.0))
        fg_r = float(np.nan_to_num(recall[fg], nan=0.0))

        class_names = self.CLASSES if self.CLASSES is not None else tuple(
            range(num_classes))
        iou_pct = np.round(iou * 100, 2)
        acc_pct = np.round(acc * 100, 2)
        f1_pct = np.round(f1 * 100, 2)
        p_pct = np.round(precision * 100, 2)
        r_pct = np.round(recall * 100, 2)
        # 与 TransNeXt / 常见 mmseg 验证表对齐：每类 IoU、Acc、Fscore、Precision、Recall
        class_table_data = [[
            'Class', 'IoU', 'Acc', 'Fscore', 'Precision', 'Recall'
        ]]
        for i in range(num_classes):
            class_table_data.append([
                class_names[i],
                iou_pct[i],
                acc_pct[i],
                f1_pct[i],
                p_pct[i],
                r_pct[i],
            ])
        print_log('per class results:', logger)
        table = AsciiTable(class_table_data)
        print_log('\n' + table.table, logger=logger)

        summary = [['Scope', 'aAcc', 'IoU(fg)', 'F1', 'Precision', 'Recall'],
                   ['global',
                    round(all_acc * 100, 2),
                    round(fg_iou * 100, 2),
                    round(fg_f1 * 100, 2),
                    round(fg_p * 100, 2),
                    round(fg_r * 100, 2)]]
        print_log('Summary (wire binary):', logger)
        st = AsciiTable(summary)
        print_log('\n' + st.table, logger=logger)

        out = OrderedDict()
        out['aAcc'] = all_acc
        out['IoU'] = fg_iou
        out['F1'] = fg_f1
        out['Precision'] = fg_p
        out['Recall'] = fg_r
        out['mIoU'] = float(np.nanmean(iou))

        if result_dump_dir:
            per_class = []
            for i in range(num_classes):
                per_class.append({
                    'class': str(class_names[i]),
                    'IoU': float(iou[i]),
                    'Acc': float(acc[i]),
                    'F1': float(f1[i]),
                    'Precision': float(precision[i]),
                    'Recall': float(recall[i]),
                })
            payload = {
                'summary': {k: float(out[k]) for k in out},
                'summary_percent': {
                    'aAcc': round(all_acc * 100, 2),
                    'IoU_fg': round(fg_iou * 100, 2),
                    'F1': round(fg_f1 * 100, 2),
                    'Precision': round(fg_p * 100, 2),
                    'Recall': round(fg_r * 100, 2),
                    'mIoU_mean': round(float(np.nanmean(iou)) * 100, 2),
                },
                'per_class': per_class,
            }
            with open(osp.join(result_dump_dir, 'eval_metrics.json'), 'w') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            report_lines = [
                'per class results:', '',
                table.table,
                '',
                'Summary (wire binary):',
                '',
                st.table,
                '',
            ]
            with open(osp.join(result_dump_dir, 'eval_report.txt'), 'w') as f:
                f.write('\n'.join(report_lines))

        if mmcv.is_list_of(results, str):
            for file_name in results:
                if osp.isfile(file_name):
                    os.remove(file_name)
        return out
