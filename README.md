# EDAFormer 电线二分类项目总览

本目录基于 **EDAFormer** 官方 MMSeg 代码，面向 **DataA / DataB** 做二类语义分割复现与对比实验。  
如果你是来搭环境或跑命令，先看 [`SETUP.md`](SETUP.md)。

---

## 文档入口

| 文档 | 用途 |
|------|------|
| [`README.md`](README.md) | 项目总览、目录约定、结果入口 |
| [`SETUP.md`](SETUP.md) | 环境、权重放置、训练/测试命令 |
| [`RESULTS.md`](RESULTS.md) | 历史结果与测试指标 |

---

## 项目概况

- 任务：DataA / DataB 电线前景二分类语义分割
- 主干：EDAFormer-Tiny
- 框架：MMSegmentation
- 数据约定：`/root/sci/DataA-B/DataA`、`/root/sci/DataA-B/DataB`
- 结果目录：`data/checkpoints1/`（训练）与 `data/checkpoints2/`（测试）

---

## 目录约定

| 位置 | 内容 |
|------|------|
| `EFT_b0.pth` | EDAFormer 预训练主干权重，放在工程根目录 |
| `data/checkpoints1/train_<时间戳>/` | 训练日志、`best_IoU*.pth`、`val_metrics.csv`、曲线图 |
| `data/checkpoints2/test_<时间戳>/` | 测试输出、`eval_metrics.json`、`eval_report.txt` |
| `DataA-B/DataA`、`DataA-B/DataB` | 数据集根目录 |

---

## 路线对照

| 路线 | 主干与框架 | 文档 |
|------|------------|------|
| 路线一 | Mask2Former + TransNeXt | `my_TransNext/README.md` |
| 路线二 | MambaVision + UPerNet | `MambaVision-main/MambaVision-main/MambaVision.md` |
| 路线三 | EDAFormer-Tiny + EDAFormerHead | 本文 + [`SETUP.md`](SETUP.md) |

---

## Citation

```bibtex
@article{yu2024embedding,
  title={Embedding-Free Transformer with Inference Spatial Reduction for Efficient Semantic Segmentation},
  author={Yu, Hyunwoo and Cho, Yubin and Kang, Beoungwoo and Moon, Seunghun and Kong, Kyeongbo and Kang, Suk-Ju},
  journal={arXiv preprint arXiv:2407.17261},
  year={2024}
}
```
