# SDGNet: Serialized Geometry-Guided Network for Point Cloud Registration

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Under%20Review-red)]()

This repository contains the official PyTorch implementation of **SDGNet**.

> **📢 Note:** The pre-trained models and full model resources will be made publicly available upon the acceptance of the paper.

**SDGNet** is a high-precision point cloud registration framework tailored for **industrial 3D measurement** and **robot hand-eye calibration** scenarios. Unlike general-purpose registration methods that prioritize video-rate speed, SDGNet is engineered to maximize **geometric accuracy** and **edge preservation** under the "stop-and-measure" operational mode typical in industrial automation.

By integrating **Beltrami Diffusion** with **Curve-based Serialization**, our framework solves a unified geometric flow via an operator splitting scheme. This design effectively suppresses sensor noise while strictly preserving high-frequency geometric details (e.g., sharp edges and corners) critical for high-precision pose estimation.

## ✨ Key Features

* **Targeted for Precision Measurement**: Specifically optimized for scenarios requiring sub-millimeter accuracy, such as industrial quality inspection and robotic assembly, balancing robust registration with acceptable inference latency (~300ms).
* **Beltrami Diffusion Module**: A physics-informed geometric evolution module that simultaneously minimizes Polyakov action and total curvature energy. It acts as a selective filter, smoothing out noise on flat surfaces while sharpening geometric edges to prevent "feature blurring" common in standard diffusion.
* **Curve-based Serialization**: Utilizes Space-Filling Curves (e.g., Z-order, Hilbert) to unroll irregular 3D point clouds into structure-preserving 1D sequences, enabling efficient global interaction via FlashAttention without quadratic complexity.
* **SGGE Module**: A Serialized Geometry-Guided Encoding module that explicitly fuses rigid-invariant geometric features (PPF) with learned semantic contexts, ensuring robustness against partial overlaps and clutter in complex workcells.

## ✨ Key Features

* **Beltrami Diffusion Module**: A unified geometric evolution equation that simultaneously minimizes Polyakov action and total curvature energy, achieving robust feature denoising while preserving high-frequency geometric details.
* **Curve-based Serialization**: Utilizes Space-Filling Curves (e.g., Z-order, Hilbert) to transform 3D point clouds into 1D sequences, maintaining topological locality for efficient sequence-based processing.
* **SGGE Module**: A Serialized Geometry-Guided Encoding module that fuses explicit geometric invariants (PPF) with implicit semantic features extracted by the backbone.
* **Efficiency**: Designed for high-precision measurement scenarios, offering a balanced trade-off between accuracy and inference latency.

## 🛠️ Requirements

* Python 3.x
* PyTorch (CUDA support required)
* `torch-geometric`, `torch-scatter`, `torch-sparse` (Must match your CUDA/PyTorch versions)
* Other dependencies listed in `requirementlist.txt`

### Installation

1.  **Install Python dependencies:**
    ```bash
    pip install -r requirementlist.txt
    ```

2.  **Compile C++ Extensions:**
    (Required for radius neighbors search and subsampling)
    ```bash
    python setup.py build_ext --inplace
    ```

## 📂 Directory Structure

* `configs/`: Configuration files (e.g., `customdata.yaml`, `kitti.yaml`, `threedmatch.yaml`).
* `data/`: Dataset loaders and preprocessing scripts.
* `models/`: Core network architecture (Beltrami Diffusion, SGGE, Attention modules).
* `losses/`: Loss functions.
* `utils/`: Utility functions.
* `cpp_wrappers/`: C++ CUDA extensions.
* `0_train.py`: Main entry point for training.
* `_1_test.py`: Main entry point for evaluation/testing.
* `benchmark.py`: Script for evaluating model parameters and inference latency.

## 🚀 Quick Start

### 1. Data Preparation
Modify the dataset paths in the configuration file. By default, the system uses `configs/customdata.yaml`.

### 2. Training
To train the model from scratch:
```bash
python 0_train.py