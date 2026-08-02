# AbdomenSeg_3D：基于 3D U-Net 的腹部多器官分割系统
[English](./README.md) | [简体中文](./README_zh-CN.md)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
![UNet](https://img.shields.io/badge/Model-U--Net-success?style=flat-square)
![MONAI|94](https://img.shields.io/badge/MONAI-v1.5.2-blue)

## 项目简介

本项目基于 **MONAI** 与 **PyTorch** 实现 3D 腹部 CT 多器官分割，使用 BTCV 数据集训练 3D U-Net，支持 NIfTI 数据读取、三维预处理、patch-based 训练、滑动窗口推理、Dice 评估与 NIfTI mask 导出。
  
项目重点是完整搭建医学影像分割训练与推理 pipeline，覆盖数据处理、模型训练、断点续训、TensorBoard 监控、测试评估与独立推理脚本。

## 亮点

- **完整 3D 医学影像分割流程**：支持 BTCV NIfTI 数据读取、spacing 重采样、HU 归一化、前景裁剪与 3D patch 采样。
- **MONAI 训练管线**：基于 3D U-Net、DiceCE Loss、AdamW 与 ReduceLROnPlateau 实现多器官分割训练。
- **显存受限场景适配**：使用 patch-based training、AMP 混合精度与小 batch 设置，在 8GB 显存环境下完成训练。
- **完整推理与评估支持**：实现 sliding-window inference、Dice 测试评估、NIfTI mask 导出、checkpoint 保存、断点续训与 TensorBoard 监控。

## 快速预览

### 分割效果

> 左侧为原始 CT，中间为 Ground Truth，右侧为 Prediction。

![Segmentation comparison](./assets/seg_compare.gif)

### 训练损失下降可视化

![TensorBoard training curve](./assets/tensorboard.png)

## 项目结构

```text
AbdomenSeg_3D/
├── assets/
│   ├── seg_compare.gif                 # 分割效果对比 GIF
│   └── tensorboard.png                 # TensorBoard 训练曲线截图
├── config/
│   └── config.yaml                     # 数据路径、训练参数、模型权重路径与推理配置
├── data/
│   ├── images/                         # 原始 BTCV CT 图像
│   ├── labels/                         # 原始 BTCV 多器官标签
│   ├── train/                          # 训练集
│   ├── val/                            # 验证集
│   ├── test/                           # 测试集
│   └── inference/                      # 待推理 CT 图像
├── dataset/
│   └── dataset_3d.py                   # MONAI 数据读取、预处理、数据增强与 DataLoader 构建
├── output/
│   ├── logs/                           # 训练与评估日志输出目录
│   ├── tensorboard/                    # TensorBoard 日志目录
│   ├── weights/                        # 模型权重与 checkpoint 保存目录
│   └── predictions/                    # 推理结果 NIfTI mask 输出目录
├── utils/
│   ├── config_utils.py                 # YAML 配置文件读取与点号访问工具
│   ├── logger_utils.py                 # 日志记录工具
│   └── split_dataset.py                # BTCV 数据集训练/验证/测试集划分脚本
├── .gitignore                          # Git 忽略规则
├── LICENSE                             # MIT 开源协议
├── README.md                           # 项目说明文档
├── requirements.txt                    # Python 项目依赖
├── train.py                            # 3D U-Net 模型训练与验证主脚本
├── evaluate.py                         # 测试集评估与 Dice 指标计算脚本
└── inference.py                        # 滑动窗口推理与 NIfTI mask 导出脚本
```

## 网络架构

本项目使用 MONAI 提供的 3D U-Net 作为 baseline 分割网络，用于 13 类腹部器官分割。

主要配置如下：

- 输入：单通道腹部 CT，`in_channels=1`
- 输出：背景 + 13 个器官类别，`out_channels=14`
- 网络：`channels=(16, 32, 64, 128, 256)`，`strides=(2, 2, 2, 2)`
- 归一化：InstanceNorm，适配 3D 小 batch 训练
- 损失函数：DiceCE Loss
- 优化器：AdamW
- 学习率调度：ReduceLROnPlateau
- 训练加速：PyTorch AMP 混合精度

## 结果与性能

当前模型在验证集上取得的最佳平均 Dice 系数（Mean Dice）为 **78.00%**。

| Split | Cases | Mean Dice | Notes                     |
| ----- | ----: | --------: | ------------------------- |
| Train |    21 |         - | patch-based training      |
| Val   |     3 |    78.00% | best checkpoint selection |
| Test  |     6 |    69.32% | final evaluation          |

## 环境配置

本项目已在以下环境完成运行测试：

| 操作系统                      | 计算设备 / GPU               | 硬件后端 | 版本                  |
| :----------------------------- | :------------------------- | :--- | :--------------------- |
| **Windows 11**                 | NVIDIA RTX 5060 8G         | CUDA | PyTorch-2.8.0+cu128    |
| **Linux (Ubuntu 24.04.4 LTS)** | AMD Radeon RX 7900 XTX 24G | ROCm | PyTorch-2.11.0+rocm7.2 |

_在当前配置下，使用 `patch_size=[96, 96, 96]` 并开启 AMP 混合精度训练时，8GB 显存环境可以运行较小 batch size 的训练。不同 GPU、PyTorch 版本和数据预处理设置可能会影响实际显存占用。_

### 核心依赖项

详细的环境要求在 `requirements.txt` 中，核心库要求如下：

- **Python** = 3.10
- **PyTorch** = 2.x
- **MONAI** = 1.5.2
- **numpy** <= 2.0

我们推荐使用 Conda 管理环境，具体命令如下：

### 1、克隆仓库

```bash
git clone https://github.com/nbplus12345/AbdomenSeg_3D.git
cd AbdomenSeg_3D
```

### 2、创建激活conda环境

```bash
conda create -n abdomenseg_3d python=3.10 -y
conda activate abdomenseg_3d
```

### 3. 安装核心深度学习框架 (PyTorch)

请根据你电脑的硬件情况，选择以下【其中一种】方式安装 PyTorch：

- NVIDIA CUDA 用户可参考：

```Bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

- AMD ROCm 用户请根据 PyTorch 官方安装页面选择与本机驱动和 ROCm 版本匹配的安装命令，例如：

```Bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
```

- CPU 或 Mac 用户可以直接安装 CPU 版本 PyTorch，但本项目主要面向 CUDA / ROCm GPU 环境测试。CPU 环境理论上可运行部分流程，但训练速度极慢，且 AMP 相关代码未作为主要测试路径。

### 4、安装项目剩余依赖

```bash
pip install -r requirements.txt
```

## 数据集准备

本项目使用经典的 **BTCV (Beyond the Cranial Vault)** 腹部多器官分割数据集，其中，训练集包含共 **30** 例带有专家标注的患者数据。

1. 请前往 [Kaggle - BTCV Dataset](https://www.kaggle.com/) 或官方 [Synapse 平台](https://www.synapse.org/) 下载 BTCV 的训练集数据（请确认下载的是 BTCV training raw data，标签应为 0-13 的多类别 label，而不是二值标签）。
2. 下载并解压后，请将图像文件夹与标签文件夹分别重命名为 `images` 和 `labels`，并放入项目的 `data/` 目录下，初始数据目录结构应如下所示：

```Plaintext
data/
├── images/
│   ├── img0001.nii.gz
│   ├── ...
└── labels/
    ├── label0001.nii.gz
    ├── ...
```

1. 随后运行数据集切分脚本：

```Bash
python utils/split_dataset.py
```

1. 脚本会按照默认 70% 训练集，10% 验证集，20% 测试集的比例划分，并生成如下目录结构：

```Plaintext
data/
├── test/
│   ├── images/
│   └── labels/
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

_切分完成后，原始 `images/` 和 `labels/` 可以保留作为备份，也可以根据需要自行删除。_

## 训练与测试

### 1. 训练 (Training)

项目中的主要参数集中在 `config/config.yaml` 中，包括数据路径、patch size、batch size、训练轮数、学习率、checkpoint 路径以及推理输入输出路径等。修改配置文件后，可以分别运行训练、评估和推理脚本。训练命令如下：

```Bash
python train.py --config ./config/config.yaml
```

本模型带有 **断点续训** 的功能，每轮自动保存 checkpoint ，但训练中断需要重新训练时，需要在 config.yaml 中修改 **resume_training** 为 true 。

### 2. 测试与评估 (Evaluation)

评估脚本会自动计算平均 Dice (DSC) 指标：

```Bash
python evaluate.py --config ./config/config.yaml
```

### 3. 查看分割结果（Segmentation）

在 config/config.yaml 中配置待分割的 CT 文件路径以及输出路径，运行分割脚本：

```Bash
python inference.py --config ./config/config.yaml
```

### 4. 实时训练监控（TensorBoard）

本项目集成 TensorBoard 记录 loss 与 Dice，用于实时监控训练/验证 Loss 以及 Dice 分数的爬升曲线。
在训练开始后，重新打开一个终端并运行：

```Bash
tensorboard --logdir=./output/tensorboard --port=6006
```

打开浏览器访问 `http://localhost:6006` 即可查看。

## 局限

- BTCV 数据规模较小，当前结果主要用于验证完整 3D 分割 pipeline。
- 当前模型为 3D U-Net baseline，尚未加入 nnU-Net 自动配置策略或 Transformer-based 模型。
- 当前未进行交叉验证和外部数据集验证，后续可扩展 per-organ analysis、模型权重发布与部署流程。
  
因此，本项目更侧重于展示医学影像分割 pipeline 的搭建能力、工程实现能力和实验复现能力。

## 开源协议

本项目基于 MIT License 开源，允许自由使用、修改和分发。详细条款请见 [LICENSE](./LICENSE) 文件。
