# AbdomenSeg_3D: 3D U-Net-Based Abdominal Multi-Organ Segmentation System
[English](./README.md) | [简体中文](./README_zh-CN.md)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
![UNet](https://img.shields.io/badge/Model-U--Net-success?style=flat-square)
![MONAI|94](https://img.shields.io/badge/MONAI-v1.5.2-blue)

## Project Overview

This project implements 3D abdominal CT multi-organ segmentation with **MONAI** and **PyTorch**. It trains a 3D U-Net on the BTCV dataset and supports NIfTI data loading, 3D preprocessing, patch-based training, sliding-window inference, Dice evaluation, and NIfTI mask export.
  
The project focuses on building a complete medical image segmentation training and inference pipeline, covering data processing, model training, checkpoint-based training resumption, TensorBoard monitoring, test evaluation, and a standalone inference script.

## Highlights

- **Complete 3D medical image segmentation workflow**: Supports BTCV NIfTI data loading, spacing resampling, HU normalization, foreground cropping, and 3D patch sampling.
- **MONAI training pipeline**: Implements multi-organ segmentation training with 3D U-Net, DiceCE Loss, AdamW, and ReduceLROnPlateau.
- **Support for memory-constrained environments**: Uses patch-based training, AMP mixed precision, and small batches to train with 8 GB of VRAM.
- **Complete inference and evaluation support**: Implements sliding-window inference, Dice test evaluation, NIfTI mask export, checkpoint saving, training resumption, and TensorBoard monitoring.

## Quick Preview

### Segmentation Results

> The original CT is shown on the left, Ground Truth in the middle, and Prediction on the right.

![Segmentation comparison](./assets/seg_compare.gif)

### Training Loss Visualization

![TensorBoard training curve](./assets/tensorboard.png)

## Project Structure

```text
AbdomenSeg_3D/
├── assets/
│   ├── seg_compare.gif                 # Segmentation comparison GIF
│   └── tensorboard.png                 # Screenshot of TensorBoard training curves
├── config/
│   └── config.yaml                     # Data paths, training parameters, model weight paths, and inference configuration
├── data/
│   ├── images/                         # Original BTCV CT images
│   ├── labels/                         # Original BTCV multi-organ labels
│   ├── train/                          # Training set
│   ├── val/                            # Validation set
│   ├── test/                           # Test set
│   └── inference/                      # CT images awaiting inference
├── dataset/
│   └── dataset_3d.py                   # MONAI data loading, preprocessing, augmentation, and DataLoader construction
├── output/
│   ├── logs/                           # Training and evaluation log output
│   ├── tensorboard/                    # TensorBoard log directory
│   ├── weights/                        # Model weights and checkpoint directory
│   └── predictions/                    # Inference NIfTI mask output
├── utils/
│   ├── config_utils.py                 # YAML configuration reader with dot notation access
│   ├── logger_utils.py                 # Logging utilities
│   └── split_dataset.py                # BTCV train/validation/test splitting script
├── .gitignore                          # Git ignore rules
├── LICENSE                             # MIT License
├── README.md                           # Project documentation
├── requirements.txt                    # Python project dependencies
├── train.py                            # Main 3D U-Net training and validation script
├── evaluate.py                         # Test evaluation and Dice calculation script
└── inference.py                        # Sliding-window inference and NIfTI mask export script
```

## Network Architecture

This project uses the 3D U-Net provided by MONAI as the baseline segmentation network for 13 abdominal organ classes.

Main configuration:

- Input: Single-channel abdominal CT, `in_channels=1`
- Output: Background + 13 organ classes, `out_channels=14`
- Network: `channels=(16, 32, 64, 128, 256)`, `strides=(2, 2, 2, 2)`
- Normalization: InstanceNorm, suitable for 3D training with small batches
- Loss function: DiceCE Loss
- Optimizer: AdamW
- Learning-rate scheduler: ReduceLROnPlateau
- Training acceleration: PyTorch AMP mixed precision

## Results and Performance

The best mean Dice coefficient achieved by the current model on the validation set is **78.00%**.

| Split | Cases | Mean Dice | Notes                     |
| ----- | ----: | --------: | ------------------------- |
| Train |    21 |         - | patch-based training      |
| Val   |     3 |    78.00% | best checkpoint selection |
| Test  |     6 |    69.32% | final evaluation          |

## Environment Setup

This project has been tested in the following environments:

| Operating System | Compute Device / GPU | Hardware Backend | Version |
| :----------------------------- | :------------------------- | :--- | :--------------------- |
| **Windows 11**                 | NVIDIA RTX 5060 8G         | CUDA | PyTorch-2.8.0+cu128    |
| **Linux (Ubuntu 24.04.4 LTS)** | AMD Radeon RX 7900 XTX 24G | ROCm | PyTorch-2.11.0+rocm7.2 |

_With the current configuration, training with a small batch size can run on 8 GB of VRAM when `patch_size=[96, 96, 96]` and AMP mixed precision are enabled. Actual VRAM usage may vary with the GPU, PyTorch version, and data preprocessing settings._

### Core Dependencies

Detailed environment requirements are provided in `requirements.txt`. The core library requirements are:

- **Python** = 3.10
- **PyTorch** = 2.x
- **MONAI** = 1.5.2
- **numpy** <= 2.0

We recommend using Conda to manage the environment. The commands are as follows:

### 1. Clone the Repository

```bash
git clone https://github.com/nbplus12345/AbdomenSeg_3D.git
cd AbdomenSeg_3D
```

### 2. Create and Activate the Conda Environment

```bash
conda create -n abdomenseg_3d python=3.10 -y
conda activate abdomenseg_3d
```

### 3. Install the Core Deep Learning Framework (PyTorch)

Choose **one** of the following PyTorch installation methods based on your computer hardware:

- NVIDIA CUDA users can use:

```Bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

- AMD ROCm users should select an installation command from the official PyTorch installation page that matches their local driver and ROCm version. For example:

```Bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
```

- CPU or Mac users can install the CPU build of PyTorch directly, but this project is primarily tested in CUDA/ROCm GPU environments. Some workflows may theoretically run on a CPU, but training is extremely slow and the AMP-related code is not a primary test path.

### 4. Install the Remaining Project Dependencies

```bash
pip install -r requirements.txt
```

## Data Preparation

This project uses the classic **BTCV (Beyond the Cranial Vault)** abdominal multi-organ segmentation dataset. Its training set contains **30** expert-annotated patient cases.

1. Download the BTCV training data from [Kaggle - BTCV Dataset](https://www.kaggle.com/) or the official [Synapse platform](https://www.synapse.org/) (make sure to download the BTCV training raw data; labels should be multiclass values from 0 to 13, not binary labels).
2. After downloading and extracting the data, rename the image and label folders to `images` and `labels`, respectively, and place them under the project's `data/` directory. The initial directory structure should be:

```Plaintext
data/
├── images/
│   ├── img0001.nii.gz
│   ├── ...
└── labels/
    ├── label0001.nii.gz
    ├── ...
```

1. Then run the dataset splitting script:

```Bash
python utils/split_dataset.py
```

1. By default, the script splits the data into 70% training, 10% validation, and 20% testing, and generates the following directory structure:

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

_After splitting, the original `images/` and `labels/` folders may be retained as a backup or deleted as needed._

## Training and Testing

### 1. Training

The main project parameters are centralized in `config/config.yaml`, including data paths, patch size, batch size, number of training epochs, learning rate, checkpoint path, and inference input/output paths. After modifying the configuration, run the training, evaluation, and inference scripts separately. Run training with:

```Bash
python train.py --config ./config/config.yaml
```

This model supports **resuming interrupted training** and automatically saves a checkpoint after every epoch. To resume after an interruption, set **resume_training** to true in config.yaml.

### 2. Testing and Evaluation

The evaluation script automatically calculates the mean Dice (DSC):

```Bash
python evaluate.py --config ./config/config.yaml
```

### 3. View Segmentation Results

Configure the input CT path and output path in config/config.yaml, then run the segmentation script:

```Bash
python inference.py --config ./config/config.yaml
```

### 4. Real-Time Training Monitoring (TensorBoard)

This project integrates TensorBoard to record loss and Dice, enabling real-time monitoring of training/validation Loss and the rising Dice curve.
After training begins, open another terminal and run:

```Bash
tensorboard --logdir=./output/tensorboard --port=6006
```

Open `http://localhost:6006` in a browser to view it.

## Limitations

- The BTCV dataset is small, and the current results are intended mainly to validate the complete 3D segmentation pipeline.
- The current model is a 3D U-Net baseline and does not yet include nnU-Net automatic configuration strategies or Transformer-based models.
- Cross-validation and external dataset validation have not yet been performed. Future work could add per-organ analysis, model weight publication, and a deployment workflow.
  
This project therefore focuses more on demonstrating the ability to build a medical image segmentation pipeline, implement it as an engineering system, and reproduce experiments.

## License

This project is open-sourced under the MIT License and may be freely used, modified, and distributed. See the [LICENSE](./LICENSE) file for details.
