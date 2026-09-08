# EDAFormer 电线二分类复现环境

> 本文只写今天这套实际可复现的环境与命令，目标是让别人照着就能跑通 DataA / DataB。
> 任务目录：`/root/sci/my_EDAformer-main/my_EDAformer-main/`

## 1. Conda 与 Python

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda create -n edaformer python=3.8 -y
conda activate edaformer
python -m pip install --upgrade pip setuptools wheel
```

## 2. PyTorch 与基础依赖

```bash
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 \
  -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python==4.5.1.48
```

## 3. MMCV / MMSeg / 其他依赖

```bash
pip install mmcv-full==1.2.7 \
  -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
pip install timm==0.3.2
```

进入工程根后：

```bash
cd /root/sci/my_EDAformer-main/my_EDAformer-main
pip install -r requirements/runtime.txt
pip install -e .
```

## 4. 预训练权重

下载地址：

```text
https://drive.google.com/drive/u/0/folders/1hiAFQcfH9qd37WOc1_HMB0vKzbY-IWrO
```

需要的文件：

```text
EFT_b0.pth
```

放置位置：

```text
/root/sci/my_EDAformer-main/my_EDAformer-main/EFT_b0.pth
```

## 5. 数据集位置

约定为：

```text
/root/sci/DataA-B/DataA
/root/sci/DataA-B/DataB
```

每个数据集内结构为：

```text
image/train|val|test
mask/train|val|test
```

如需显式指定，可用：

```bash
export WIRE_SEG_DATA_ROOT=/root/sci/DataA-B
# 或
export WIRE_SEG_DATAA_ROOT=/root/sci/DataA-B/DataA
export WIRE_SEG_DATAB_ROOT=/root/sci/DataA-B/DataB
```

## 6. 冒烟检查

```bash
cd /root/sci/my_EDAformer-main/my_EDAformer-main
conda activate edaformer
python tools/smoke_test_wire_data.py --wire-scheme dataa
python tools/smoke_test_wire_data.py --wire-scheme datab
```

## 7. 训练命令

### DataA

```bash
cd /root/sci/my_EDAformer-main/my_EDAformer-main
conda activate edaformer
python tools/train_wire_scheme2.py dataa
```

### DataB

```bash
python tools/train_wire_scheme2.py datab
```

说明：

- 这里使用的是今天复现时对应的 `fix512_th055` 配方
- 若想复现旧版 `legacy`，把 `train_wire_scheme2.py` 换成 `train_wire_scheme1.py`
- 训练产物会写入 `data/checkpoints1/train_<时间戳>/`

## 8. 测试命令

### DataA

```bash
cd /root/sci/my_EDAformer-main/my_EDAformer-main
conda activate edaformer
python tools/test_edaformer_wire_dataa.py --best
```

### DataB

```bash
python tools/test_edaformer_wire_datab.py --best
```

### 按训练目录测试

```bash
python tools/test_edaformer_wire_dataa.py --best --work-dir data/checkpoints1/train_<时间戳>
python tools/test_edaformer_wire_datab.py --best --work-dir data/checkpoints1/train_<时间戳>
```

`train_<时间戳>` 就是训练时自动生成的目录名。复制命令时把它替换成你自己的那次训练目录。

### 指定某个权重

```bash
python tools/test_edaformer_wire_dataa.py \
  /root/sci/my_EDAformer-main/my_EDAformer-main/data/checkpoints1/train_<时间戳>/best_IoU_xxx.pth
```
