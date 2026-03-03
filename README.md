# Brain-OCT
# BrainOCT-PVT: Exploiting Radial Priors and Boundary Attention for Intracranial OCT Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

This repository contains the official PyTorch implementation of **BrainOCT-PVT**, a novel framework designed for the highly accurate segmentation of intracranial blood vessel lumens from Optical Coherence Tomography (OCT) images. 

## 📢 Introduction
Intracranial OCT imaging provides high-resolution, non-ionizing visualization of blood vessels. However, segmenting the lumen accurately is challenging due to inherent **speckle noise**, highly variable cavity shapes, and the need for extremely precise boundary delineation. 

Building upon the Pyramid Vision Transformer (PVT), **BrainOCT-PVT** introduces specialized modules that leverage the unique imaging priors of OCT (e.g., dark lumen, bright wall, radial gradient) to achieve state-of-the-art performance.

### ✨ Main Contributions
1. **Single-Channel Stem Adaptation:** Optimized the PVT backbone to process single-channel grayscale OCT inputs, drastically reducing parameters and memory usage without sacrificing accuracy.
2. **Radial Intensity Module (RIM):** A novel low-level feature enhancement module that transforms features into polar coordinates to capture the "black-bright-dark-bright" radial intensity pattern of the vessel wall, coupled with a DCT-based frequency attention mechanism to suppress speckle noise.
3. **Deformable Cascaded Feature Multiplexing (D-CFM):** Replaces standard convolutions with Deformable Convolutions to better align multi-scale features of curved and morphing vessels across spatial dimensions.
4. **Boundary-aware Attention Module (BAM):** Extracts Laplace/Sobel edge heatmaps to guide a Windowed Swin-Attention mechanism, ensuring long-range structural coherence and crisp boundary predictions.
5. **Composite OCT-Loss:** A domain-specific loss function carefully balancing regional overlap, boundary precision, and foreground-background imbalance:
   $$L = 0.5 \cdot L_{Dice} + 0.3 \cdot L_{BTIoU} + 0.2 \cdot L_{FocalTversky}$$

## 📊 Benchmark Results

Our model was evaluated on a clinical intracranial OCT dataset (13 patients, axial plane). BrainOCT-PVT significantly outperforms existing state-of-the-art segmentation networks, particularly in the Hausdorff Distance (HD95) metric, proving its superior boundary localization.

| Model | Mean Dice ↑ | Mean HD95 ↓ |
| :--- | :---: | :---: |
| U-Net | 0.9097 | 0.4347 |
| U-Net++ | 0.9093 | 0.4465 |
| TransUNet | 0.9058 | 0.4409 |
| Swin-Unet | 0.8542 | 0.7645 |
| Polyp-PVT (Baseline)| 0.9319 | 0.3073 |
| **BrainOCT-PVT (Ours)** | **0.9506** | **0.2686** |
