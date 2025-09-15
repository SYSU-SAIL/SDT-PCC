
# Slide Deformable Transformer for High-Precision LiDAR Point Cloud Compression

This repository provides the **official implementation (partial release)** of our paper:

> **Slide Deformable Transformer for High-Precision LiDAR Point Cloud Compression**
> *Under Review*

## 🔍 Introduction

Dynamic LiDAR point cloud compression with range images aims to reduce storage and transmission costs while preserving both **spatial accuracy** and **temporal consistency** across frames.

To tackle the limitations of existing Transformer-based methods—such as feature misalignment under motion displacement and inefficiency of global attention—we propose a new framework, **SDT-PCC**, which integrates:

* **Slide Deformable Transformer (SDT)**:

  * Restricts attention to **local sliding windows**, capturing fine-grained correspondences.
  * Incorporates **deformable convolution** into cross-frame attention for adaptive motion alignment.

* **Radix-Decomposition Multi-Channel Quantizer (RDMCQ)**:

  * Decomposes 16-bit range values into multiple channels.
  * Progressively refines precision across radix levels, mitigating quantization loss.

Experiments on the **SemanticKITTI** dataset demonstrate that SDT-PCC achieves **higher efficiency, temporal coherence, and reconstruction accuracy** compared to existing methods.

![image](img/fig_pipeline.jpg)

## 📊 Experimental Results

We provide partial experimental results on the **SemanticKITTI** dataset to showcase the effectiveness of SDT-PCC.
Detailed comparisons and ablations can be found in the full paper.

---

⚠️ **Note**
This repository currently contains a **partial release** of our implementation.
👉 The **full codebase, trained models, and detailed instructions will be made publicly available after the paper is accepted.**
