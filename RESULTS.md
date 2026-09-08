# 电线分割测试结果记录

本文件汇总 **测试集** 指标。与当次测试目录内 **`eval_metrics.json` / `eval_report.txt`** 一致（由 `tools/test.py` 在 **`--eval`** 且 `wire_seg_experiment` 时自动写入 `data/checkpoints2/test_<时间戳>/`）。

**本表数据来源**：`data/checkpoints2/test_20260427_180008` … `test_20260427_180157` 目录下的 **`eval_metrics.json`**（评测时间约 **2026-04-27 18:00–18:02**）。百分数与终端 **`Summary (wire binary)`** 对齐。

---

## 字段约定

| 字段 | 含义 |
|------|------|
| **数据集** | **DataA** / **DataB** |
| **训练方案** | **legacy** / **fix512_th055** / **fix256_th050**（由对应 `train_*` 内快照 `model.test_cfg` 判定） |
| **IoU (fg)** | 前景类 IoU（`IoU_fg`，%）；**不是** `mIoU_mean`。 |
| **其余列** | `eval_metrics.json` 中 **`summary_percent`**（%） |

---

## DataA

| 日期 | 训练方案 | 训练目录 | 测试目录 | 权重 | IoU (fg) | aAcc | F1 | Precision | Recall | 备注 |
|------|----------|----------|----------|------|-----------|------|-----|-----------|--------|------|
| 2026-04-27 | **legacy** | `train_20260427_155943` | `test_20260427_180008` | `best_IoU_e46.pth` | 43.55 | 99.25 | 60.68 | 56.03 | 66.16 | `eval_metrics.json` |
| 2026-04-27 | **fix512_th055** | `train_20260427_155955` | `test_20260427_180026` | `best_IoU_e68.pth` | 45.42 | 99.34 | 62.47 | 62.01 | 62.95 | 同上 |
| 2026-04-27 | **fix256_th050** | `train_20260427_160003` | `test_20260427_180054` | `best_IoU_e144.pth` | 30.43 | 99.28 | 46.66 | 66.99 | 35.80 | 同上 |

```bash
conda activate edaformer
cd /root/my_TransNext/EDAFormer-main/EDAFormer-main
python tools/test_edaformer_wire_dataa.py --best --work-dir data/checkpoints1/train_20260427_155943
python tools/test_edaformer_wire_dataa.py --best --work-dir data/checkpoints1/train_20260427_155955
python tools/test_edaformer_wire_dataa.py --best --work-dir data/checkpoints1/train_20260427_160003
```

---

## DataB

| 日期 | 训练方案 | 训练目录 | 测试目录 | 权重 | IoU (fg) | aAcc | F1 | Precision | Recall | 备注 |
|------|----------|----------|----------|------|-----------|------|-----|-----------|--------|------|
| 2026-04-27 | **legacy** | `train_20260427_160210` | `test_20260427_180108` | `best_IoU_e113.pth` | 77.29 | 98.92 | 87.19 | 87.93 | 86.46 | `eval_metrics.json` |
| 2026-04-27 | **fix512_th055** | `train_20260427_160234` | `test_20260427_180138` | `best_IoU_e66.pth` | 77.20 | 98.88 | 87.13 | 85.02 | 89.35 | 同上 |
| 2026-04-27 | **fix256_th050** | `train_20260427_160240` | `test_20260427_180157` | `best_IoU_e70.pth` | 73.98 | 98.76 | 85.04 | 87.42 | 82.79 | 同上 |

```bash
conda activate edaformer
cd /root/my_TransNext/EDAFormer-main/EDAFormer-main
python tools/test_edaformer_wire_datab.py --best --work-dir data/checkpoints1/train_20260427_160210
python tools/test_edaformer_wire_datab.py --best --work-dir data/checkpoints1/train_20260427_160234
python tools/test_edaformer_wire_datab.py --best --work-dir data/checkpoints1/train_20260427_160240
```

---

## 每类明细

各测试目录内 **`eval_metrics.json`** 的 **`per_class`** 含 background / foreground 的 IoU、Acc、F1、Precision、Recall（0–1 浮点）；**`eval_report.txt`** 为与终端一致的 ASCII 表。

---

*与 `SETUP.md` 中训练配方及测试命令说明一致。*
