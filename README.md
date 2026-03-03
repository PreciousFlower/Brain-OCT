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

通过 **输入通道简化 + RIM(低层径向) + Deform-CFM(多尺度对齐) + BAM(边界注意) + OCT-Loss**，Polyp-PVT 能够更充分利用脑 OCT 数据集“暗腔/亮壁/径向层+speckle” 的成像特性，在 Dice、HD95 两大核心指标上实现可观提升，并保持推理速度基本不变。

| 组件 | Polyp-PVT 原实现 | 脑 OCT 适配点 | 改进后方案 |
| :--- | :--- | :--- | :--- |
| **输入适配** | 3-channel RGB | 实际是单通道灰度；3 通道仅复制 | - 将 PVT stem 的第一层 Conv2d(3,64,7,4,3) 改为 Conv2d(1,64,7,4,3)；<br>- dataloader 仅读灰度，节省 2/3 参数与显存 |
| **CIM(低层特征)** | CBAM-like Channel + Spatial | 腔体总是最暗；亮壁呈 环状径向梯度；speckle 噪声 | **RIM - Radial Intensity Module：**<br>1. 将特征转极坐标(grid_sample)<br>2. 对 θ 维做 1×k 深度可分离卷积 → 捕获环状梯度<br>3. 频域 DCT-attention 抑 speckle 高频<br>4. 和原特征残差相加 |
| **CFM(多尺度融合)** | 三层逐级上采 + concat | 跨尺度对齐困难；弯曲血管发生平移 | **D-CFM (Deform-CFM)：**<br>- 使用 DeformableConv2d 替代 3×3 Conv<br>- 加入可学习 offset → 让粗尺度特征依形变对齐<br>- 额外引入 Edge-guided gating (来自 Sobel 边图)，降低背景融合 |
| **SAM(语义自注意)** | 4×4 Anchor + GCN | 需要对 边界条带 保持长程一致 | **BAM – Boundary-aware Attention：**<br>- 在 CFM 输出上再做一次 Laplace(edge) 得到边缘热图<br>- 以边缘热图作 query，采用 Window Swin-Attention 代替原 GCN → 同时获得长程和局部边界连贯性 |
| **损失** | 自适应加权 BCE + WIoU | HD95 是评价重点；腔体像素远 < 背景 | **Dice + BoundaryIoU + Focal Tversky**<br>L = 0.5·Dice + 0.3·BTIoU + 0.2·FTV<br>- BoundaryIoU 直接用形态学膨胀得到 2 像素宽边带；<br>- Focal Tversky(α=.3,β=.7,γ=4/3) 解决前后景不平衡 |
| **训练细节** | 352² crop | OCT 本身 512×512；大量无用背景 | 224² 随机裁剪 + 随机极坐标裁切<br>Mix-speckle (将相邻帧 speckle 随机复制) 

### 1. 基础信息与数据来源

| 维度 | 内容要点 |
| :--- | :--- |
| **成像模态** | **光学相干断层扫描（OCT）** —— 近红外干涉成像，非电离辐射、无需造影剂、金属伪影基本为零；典型单 B-scan 分辨率可达 5–15 µm，帧率高，可快速获得 3D 体数据 |
| **器官 / 目标** | **颅内血管（脑动脉）**；研究焦点是血管腔 (lumen) 轮廓与内中外膜（IEM / media / EEM）三层结构 |
| **数据集结构** | 来自厦门大学附属第一医院<br>• 13 名患者的脑血管 OCT 扫描<br>• 每例 3 个正交平面（本研究选 **轴向**）<br>• 每人约 400 帧，原始 `.dicom` → `.jpg` + 匹配掩码<br>• 80% 训练 / 20% 验证；额外提供三维体素间距 ≈ 33.6 µm（文中 HD95 ✕ 0.033586） |

### 2. 图像与解剖特征

**图像外观特征**
* **高对比度层次**：血管腔为低反射暗区，内膜高反射，媒体低反射，外膜高且粗糙反射（Fig 2）
* **Speckle 噪声**：固有散斑导致颗粒状纹理，不同于 CT 的泊松噪声
* **纵向条带**：扫描束导致的周期性强度纹理
* **各帧强配准**：同一次 pull-back，血管走向基本一致，跨帧几何连续

**器官/解剖特点**
* 颅内动脉相对直径小（1–4 mm），壁厚仅数百微米
* 弯曲、分叉多，腔体形状高度可变
* 病变（斑块）会破坏“三层”分界并造成信号衰减；这可作为病理特征

### 3. 模型设计先验与启示

**对分割可利用的性质**
1. **强度先验** —— 腔体总是最暗，且与内膜形成高梯度，可用简单阈值或边缘 loss 进行引导
2. **层状结构先验** —— “黑–亮–暗–亮”径向强度模式可通过径向卷积(1D ring conv) 或极坐标 U-Net 强化
3. **形状连续性** —— 邻近 B-scan 腔轮廓缓变，3D 卷积或 ConvLSTM 可平滑预测
4. **小视场 & 稀疏背景** —— 绝大多数像素非血管壁，可用 hard negative mining 聚焦正样本
5. **金属无伪影** —— 不必考虑 CT 常见的金属条纹，允许更激进的对比增强
6. **成像尺寸统一** —— 400 帧/病例 & 相似像素分辨率，便于固定输入尺寸、减少重采样误差

**深度模型启示**
* **U-形网络适配**：Encoder 抽象纹理，Decoder 通过 skip-connection 保留边缘
* **引入注意力门**：可抑制 speckle 噪声，仅关注壁-腔界面
* **长程依赖**：Transformer/Swin-Unet 的长程依赖帮助在 speckle 覆盖时保持轮廓连贯
* **双重损失**：结合 `Dice + HD95` 双重损失，利用层状边界的 hausdorff 敏感性精细化边缘
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

|
