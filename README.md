# obs2gt — 观测 → 真值 图像重建（地面仿真）

从 SKIRT 星系立方体出发，前向模拟地面 r 波段观测，并训练一个变分自编码器
（VAE）从 **含噪观测（OBS）** 重建 **无噪真值（GT）**。

本仓库包含复现重建结果所需的全部**仿真**与**训练**代码，以及所依赖的模型库。
（结果评估 / 绘图脚本不在本仓库内。）

## 目录结构

```
obs2gt/
├── main.py                       # 训练入口（PyTorch Lightning / OmegaConf）
├── configs/generation/           # 三组实验配置（同一网络，仅损失不同）
│   ├── ground_mae_l.yaml         #   L1/MAE 监督
│   ├── ground_mse_l.yaml         #   MSE 监督
│   └── ground_chi2_l_obsivar.yaml#   观测方差加权的 χ² 监督
├── ground_sim/                   # 前向仿真（生成数据集）
│   ├── forward_model.py          #   SKIRT → 地面观测前向模型（OBS/GT/SIGMA）
│   └── build_dataset.py          #   批量构建 train/valid/test 数据集
└── sgm/                          # 模型库（AutoencodingEngine、Encoder/Decoder、
                                  # Regularizer、DiscVAELoss、DataModule、Dataset 等）
```

## 流程

### 1. 生成仿真数据集

`ground_sim/forward_model.py` 实现完整前向链：光度距离流量缩放、角尺寸缩放、
Moffat PSF 卷积、天空背景（含倾斜残差）、多次曝光 Poisson + 读出噪声，以及
**从数据本身估计**的背景与方差（构造 SIGMA 时从不使用 GT）。

```bash
python -m ground_sim.build_dataset --out-root <数据集输出目录> --workers 250
```

每个观测取 4 个随机 64×64 cutout。数据布局：

```
<root>/{train,valid,test}/GROUND_R/
    BGSUB/<name>.fits     # HDU0 = OBS（nanomaggy，背景已减）
                          # test 另有 HDU1 "GT" + HDU2 "COVERAGE"
    INVVAR/<name>.fits    # HDU0 = 1/SIGMA²
```

> train/valid 的 BGSUB **不含 GT**，因此重建误差在训练/验证阶段物理上无法泄漏。

### 2. 训练

`sgm/models/autoencoder.py` 的 `AutoencodingEngine` 以 64×64 单通道 OBS 为输入，
输出重建图；编码器为 `cat_invvar=False`（SIGMA 只进入损失加权，不进入网络输入）。
large 规模：`ch=128, z_channels=80, ch_mult=[1,2,4]`。

```bash
python main.py --base configs/generation/ground_mae_l.yaml  --train --gpus 0,
python main.py --base configs/generation/ground_mse_l.yaml  --train --gpus 0,
python main.py --base configs/generation/ground_chi2_l_obsivar.yaml --train --gpus 0,
```

三组配置仅 `loss_type`（`mae` / `mse` / `chi2`）不同，网络结构与数据一致。

## 说明

- 实验中的三组 large 模型用同一套网络、同一份数据、仅更换损失，用于对比
  MAE / MSE / 观测方差加权 χ² 对重建质量的影响（对应 seed 42）。
- 训练配置中的数据集路径为实验机器上的绝对路径，迁移时请按需修改
  `data.params.train.dataset.path` 与 `data.params.validation.dataset.path`。

## 环境

见 `requirements.txt`（Python 3.10+，CUDA 版 PyTorch）。
