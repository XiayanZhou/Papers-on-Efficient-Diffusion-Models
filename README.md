# Papers-on-Efficient-Diffusion-Models


&emsp; &emsp; **Purpose:** We aim to provide a summary of **Efficient Diffusion Models**. More papers will be summarized.

&emsp; &emsp; **Homepage:** University of Science and Technology of China **(USTC)**, [Intelligent Media Computing Lab **(IMCL)**](https://faculty.ustc.edu.cn/chenzhibo)**.

&emsp; &emsp; **Email:** xxxxxxx@mail.ustc.edu.cn

&emsp; &emsp; &emsp; &emsp; If you have any suggestions or find our work helpful, feel free to contact us.
  
&emsp; &emsp; &emsp; &emsp; If you find our survey is useful in your research or applications, please consider giving us a star 🌟.


## Table of Contents

- [**1_U-Net_Architecture**](#_1_U-Net_Architecture)
- - [1-1_Quantization](#_1-1_Quantization)
- - - [1-1-1_PTQ_(Post-Traning-Quantization)](#_1-1-1_Post-Traning-Quantization)
- - - [1-1-2_QAT_(Quantization-Aware-Traning)](#_1-1-2_Quantization-Aware-Traning)
- - [1-2_Pruning_&_Structure_Distillation](#_1-2_Pruning_and_Structure_Distillation)
- - - [1-2-1_Pruning](#_1-2-1_Pruning)
- - - [1-2-2_Structure_Distillation](#_1-2-2_Structure_Distillation)
- - [1-3_Feature_Cache_&_Reuse](#_1-3_Feature_Cache_and_Reuse)
- - [1-4_Timestep_Distallation](#_1-4_Timestep_Distallation)
- - - [1-4-1_ODE_Preserving_Distallation](#_1-4-1_ODE_Preserving_Distallation)
- - - [1-4-2_ODE_Reconstructing_Distallation](#_1-4-2_ODE_Reconstructing_Distallation)
- - [1-5_Fast_Sampling_Solver](#_1-.5_Fast_Sampling_Solver)
- - [1-6_GAN-Based_Method](#_1-6_GAN-Based_Method)
- - [1-7_Efficient_Training](#_1-7_Efficient_Training)
- - [1-8_New_Architecture_Design](#_1-8_New_Architecture_Design)
- - [1-9_Other_Methods](#_1-9_Other_Methods)
- [**2_Transformer Architecture**](#_2_Transformer_Architecture)
- - [2-1_Quantization](#_2-1_Quantization)
- - - [2-1-1_PTQ_(Post-Traning-Quantization)](#_2-1-1_Post-Traning-Quantization)
- - - [2-1-2_QAT_(Quantization-Aware-Traning)](#_2-1-2_Quantization-Aware-Traning)
- - [2-2_Pruning & Structure Distillation](#_2-2_Pruning_and_Structure_Distillation)
- - - [2-1_Pruning](#_2-2-1_Pruning)
- - - [2-2-2_Structure Distillation](#_2-2-2_Structure_Distillation)
- - [2-3_Feature Cache & Reuse](#_2-3_Feature_Cache_and_Reuse)
- - [2-4_Timestep Distallation](#_2-4_Timestep_Distallation)
- - [2-5_New Architecture Design](#_2-5_New_Architecture_Design)
- - [2-6_Other Methods](#_2-6_Other_Methods)


# _1_U-Net_Architecture

## _1-1_Quantization

### _1-1-1_Post-Traning-Quantization

| Title | arXiv | Github| WebSite | Pub. & Date |
|:-----:|:-----:|:-----:|:-----:|:-----:|
| [Post-training Quantization on Diffusion Models](https://arxiv.org/abs/2211.15736) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.15736) | [![Star](https://img.shields.io/github/stars/42Shawn/PTQ4DM.svg?style=social&label=Star)](https://github.com/42Shawn/PTQ4DM) | - | **CVPR 2023** |
| [Q-Diffusion: Quantizing Diffusion Models](https://arxiv.org/abs/2302.04304) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2302.04304) | [![Star](https://img.shields.io/github/stars/Xiuyu-Li/q-diffusion.svg?style=social&label=Star)](https://github.com/Xiuyu-Li/q-diffusion) | [![Website](https://img.shields.io/badge/Website-9cf)](https://xiuyuli.com/qdiffusion/) | **ICCV 2023** |
| [PTQD: Accurate Post-Training Quantization for Diffusion Models](https://arxiv.org/abs/2305.10657) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.10657) | [![Star](https://img.shields.io/github/stars/ziplab/PTQD.svg?style=social&label=Star)](https://github.com/ziplab/PTQD) | - | **NeurIPS 2023** |
| [Towards Accurate Post-training Quantization for Diffusion Models](https://arxiv.org/abs/2305.18723) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.18723) | [![Star](https://img.shields.io/github/stars/junhyukso/tdq.svg?style=social&label=Star)](https://github.com/junhyukso/tdq) | - | **NeurIPS 2023** |
| [Temporal Dynamic Quantization for Diffusion Models](https://arxiv.org/abs/2306.02316) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.02316) | - | - | Jun 2023 |
| [Softmax Bias Correction for Quantized Generative Models](https://arxiv.org/abs/2309.01729) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2309.01729) | - | - | **ICCV 2023** |
| [EfficientDM: Efficient Quantization-Aware Fine-Tuning of Low-Bit Diffusion Models](https://arxiv.org/abs/2310.03270) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.03270) | [![Star](https://img.shields.io/github/stars/ThisisBillhe/EfficientDM.svg?style=social&label=Star)](https://github.com/ThisisBillhe/EfficientDM) | - | **ICLR 2024** |
| [Post-training Quantization for Text-to-Image Diffusion Models with Progressive Calibration and Activation Relaxing](https://arxiv.org/abs/2311.06322) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2311.06322) | [![Star](https://img.shields.io/github/stars/tsa18/PCR.svg?style=social&label=Star)](https://github.com/tsa18/PCR) | - | **ECCV 2024** |
| [TFMQ-DM: Temporal Feature Maintenance Quantization for Diffusion Models](https://arxiv.org/abs/2311.16503) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2311.16503) | [![Star](https://img.shields.io/github/stars/ModelTC/TFMQ-DM.svg?style=social&label=Star)](https://github.com/ModelTC/TFMQ-DM) | [![Website](https://img.shields.io/badge/Website-9cf)](https://modeltc.github.io/TFMQ-DM/) | **CVPR 2024** |
| [Efficient Quantization Strategies for Latent Diffusion Models](https://arxiv.org/abs/2312.05431) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.05431) | [![Star](https://img.shields.io/github/stars/ThisisBillhe/EfficientDM.svg?style=social&label=Star)](https://github.com/ThisisBillhe/EfficientDM) | - | **ICLR 2024** |
| [EDA-DM: Enhanced Distribution Alignment for Post-Training Quantization of Diffusion Models](https://arxiv.org/abs/2401.04585) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2401.04585) | [![Star](https://img.shields.io/github/stars/BienLuky/EDA-DM.svg?style=social&label=Star)](https://github.com/BienLuky/EDA-DM) | - | Jan 2024 |
| [QNCD: Quantization Noise Correction for Diffusion Models](https://arxiv.org/abs/2403.19140) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2403.19140) | [![Star](https://img.shields.io/github/stars/huanpengchu/QNCD.svg?style=social&label=Star)](https://github.com/huanpengchu/QNCD) | - | **ACM MM 2024** |
| [TMPQ-DM: Joint Timestep Reduction and Quantization Precision Selection for Efficient Diffusion Models](https://arxiv.org/abs/2404.09532) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2404.09532) | [![Star](https://img.shields.io/github/stars/sihouzi21c/TMPQ-DM.svg?style=social&label=Star)](https://github.com/sihouzi21c/TMPQ-DM) | - | Apr 2024 |
| [MixDQ: Memory-Efficient Few-Step Text-to-Image Diffusion Models with Metric-Decoupled Mixed Precision Quantization](https://arxiv.org/abs/2405.17873) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2405.17873) | [![Star](https://img.shields.io/github/stars/thu-nics/MixDQ.svg?style=social&label=Star)](https://github.com/thu-nics/MixDQ) | [![Website](https://img.shields.io/badge/Website-9cf)](https://a-suozhang.xyz/mixdq.github.io/) | **ECCV 2024** |
| [BitsFusion: 1.99 bits Weight Quantization of Diffusion Model](https://arxiv.org/abs/2406.05723) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.05723) | [![Star](https://img.shields.io/github/stars/zhengchen1999/BI-DiffSR.svg?style=social&label=Star)](https://github.com/zhengchen1999/BI-DiffSR) | [![Website](https://img.shields.io/badge/Website-9cf)](https://zhengchen1999.github.io/BI-DiffSR-Web/) | **NeurIPS 2024** |
| [Binarized Diffusion Model for Image Super-Resolution](https://arxiv.org/abs/2406.04333) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.04333) | [![Star](https://img.shields.io/github/stars/snap-research/BitsFusion.svg?style=social&label=Star)](https://github.com/snap-research/BitsFusion) | [![Website](https://img.shields.io/badge/Website-9cf)](https://snap-research.github.io/BitsFusion/) | **NeurIPS 2024** |
| [Timestep-Aware Correction for Quantized Diffusion Models](https://arxiv.org/abs/2407.03917) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2407.03917) | - | - | **ECCV 2024** |
| [QVD: Post-training Quantization for Video Diffusion Models](https://arxiv.org/abs/2407.11585) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2407.11585) | - | - | **ACM MM 2024** |
| [Temporal Feature Matters: A Framework for Diffusion Model Quantization](https://arxiv.org/abs/2407.19547) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2407.19547) | - | - | Jul 2024 |
| [Low-Bitwidth Floating Point Quantization for Efficient High-Quality Diffusion Models](https://arxiv.org/abs/2408.06995) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2408.06995) | - | - | Aug 2024 |
| [PTQ4ADM: Post-Training Quantization for Efficient Text Conditional Audio Diffusion Models](https://arxiv.org/abs/2409.13894) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2409.13894) | - | - | Sep 2024 |
| [DilateQuant: Accurate and Efficient Diffusion Quantization via Weight Dilation](https://arxiv.org/abs/2409.14307) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2409.14307) | [![Star](https://img.shields.io/github/stars/BienLuky/DilateQuant.svg?style=social&label=Star)](https://github.com/BienLuky/DilateQuant) | - | Sep 2024 |


### _1-1-2_Quantization-Aware-Traning

| Title | arXiv | Github| WebSite | Pub. & Date |
|:-----:|:-----:|:-----:|:-----:|:-----:|
| [Q-DM: An Efficient Low-bit Quantized Diffusion Model](https://openreview.net/forum?id=sFGkL5BsPi) | [![arXiv](https://img.shields.io/badge/OpenReview-8c1b13.svg)](https://openreview.net/forum?id=sFGkL5BsPi) | - | - | **NeurIPS 2023** |
| [Effective Quantization for Diffusion Models on CPUs](https://arxiv.org/abs/2311.16133) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2311.16133) | [![Star](https://img.shields.io/github/stars/intel/intel-extension-for-transformers.svg?style=social&label=Star)](https://github.com/intel/intel-extension-for-transformers) | - | **NeurIPS 2023 Workshop** |
| [QuEST: Low-bit Diffusion Model Quantization via Efficient Selective Finetuning](https://arxiv.org/abs/2402.03666) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.03666) | [![Star](https://img.shields.io/github/stars/hatchetProject/QuEST.svg?style=social&label=Star)](https://github.com/hatchetProject/QuEST) | - | Feb 2024 |
| [BinaryDM: Accurate Weight Binarization for Efficient Diffusion Models](https://arxiv.org/abs/2404.05662) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2404.05662) | [![Star](https://img.shields.io/github/stars/Xingyu-Zheng/BinaryDM.svg?style=social&label=Star)](https://github.com/Xingyu-Zheng/BinaryDM) | - | Apr 2024 |


## _1-2_Pruning_and_Structure_Distillation

### _1-2-1_Pruning 

| Title | arXiv | Github | WebSite | Pub. & Date |
|---|---|---|---|---|
| [Diffusion Probabilistic Model Made Slim](https://arxiv.org/abs/2211.17106) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.17106) | - | - | **CVPR 2023** |
| [Structural Pruning for Diffusion Models](https://arxiv.org/abs/2305.10924) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.10924) | [![Star](https://img.shields.io/github/stars/VainF/Diff-Pruning.svg?style=social&label=Star)](https://github.com/VainF/Diff-Pruning) | - | **NeurIPS 2023** |
| [BK-SDM: A Lightweight, Fast, and Cheap Version of Stable Diffusion](https://arxiv.org/abs/2305.15798) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.15798) | [![Star](https://img.shields.io/github/stars/Nota-NetsPresso/BK-SDM.svg?style=social&label=Star)](https://github.com/Nota-NetsPresso/BK-SDM) | - | **ECCV 2024** |
| [SnapFusion: Text-to-Image Diffusion Model on Mobile Devices within Two Seconds](https://arxiv.org/abs/2306.00980) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.00980) | - | [![Website](https://img.shields.io/badge/Website-9cf)](https://snap-research.github.io/SnapFusion/) | **NeurIPS 2023** |
| [Squeezing Large-Scale Diffusion Models for Mobile](https://arxiv.org/abs/2307.01193) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2307.01193) | [![Star](https://img.shields.io/github/stars/Xingyu-Zheng/BinaryDM.svg?style=social&label=Star)](https://github.com/Xingyu-Zheng/BinaryDM) | - | **ICML 2023 Workshop** |
| [AutoDiffusion: Training-Free Optimization of Time Steps and Architectures for Automated Diffusion Model Acceleration](https://arxiv.org/abs/2309.10438) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2309.10438) | [![Star](https://img.shields.io/github/stars/lilijiangg/autodiffusion.svg?style=social&label=Star)](https://github.com/lilijiangg/autodiffusion) | - | **ICCV 2023** |
| [Lightweight Diffusion Models with Distillation-Based Block Neural Architecture Search](https://arxiv.org/abs/2311.04950) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2311.04950) | [![Star](https://img.shields.io/github/stars/Xingyu-Zheng/BinaryDM.svg?style=social&label=Star)](https://github.com/Xingyu-Zheng/BinaryDM) | - | Nov 2023 |
| [MobileDiffusion: Instant Text-to-Image Generation on Mobile Devices](https://arxiv.org/abs/2311.16567) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2311.16567) | [![Star](https://img.shields.io/github/stars/Xingyu-Zheng/BinaryDM.svg?style=social&label=Star)](https://github.com/Xingyu-Zheng/BinaryDM) | - | **ECCV 2024** |
| [KOALA: Empirical Lessons Toward Memory-Efficient and Fast Diffusion Models for Text-to-Image Synthesis](https://arxiv.org/abs/2312.04005) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.04005) | [![Star](https://img.shields.io/github/stars/youngwanLEE/sdxl-koala.svg?style=social&label=Star)](https://github.com/youngwanLEE/sdxl-koala) | [![Website](https://img.shields.io/badge/Website-9cf)](https://youngwanlee.github.io/KOALA/) | **NeurIPS 2024** |
| [Not All Steps are Equal: Efficient Generation with Progressive Diffusion Models](https://arxiv.org/abs/2312.13307) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.13307) | - | - | Dec 2023 |
| [A-SDM: Accelerating Stable Diffusion through Redundancy Removal and Performance Optimization](https://arxiv.org/abs/2312.15516) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.15516) | - | [![Website](https://img.shields.io/badge/Website-9cf)](https://youngwanlee.github.io/KOALA/) | Dec 2023 |


### _1-2-2_Structure_Distillation

| Title | arXiv | Github | WebSite | Pub. & Date |
|---|---|---|---|---|


## _1-3_Feature_Cache_and_Reuse

| Title | arXiv | Github | WebSite | Pub. & Date |
|---|---|---|---|---|


## _1-4_Timestep_Distallation

### _1-4-1_ODE_Preserving_Distallation

| Title | arXiv | Github | WebSite | Pub. & Date |
|---|---|---|---|---|


### _1-4-2_ODE_Reconstructing_Distallation

| Title | arXiv | Github | WebSite | Pub. & Date |
|---|---|---|---|---|


## _1-5_Fast_Sampling_Solver

| Title | arXiv | Github | WebSite| Pub. & Date |
|---|---|---|---|---|


## _1-6_GAN-Based_Method

| Title | arXiv | Github | WebSite| Pub. & Date |
|---|---|---|---|---|


## _1-7_Efficient_Training

| Title | arXiv | Github | WebSite| Pub. & Date |
|---|---|---|---|---|


## _1-8_New_Architecture_Design

| Title | arXiv | Github | WebSite| Pub. & Date |
|---|---|---|---|---|


## _1-9_Other_Methods

| Title | arXiv | Github | WebSite| Pub. & Date |
|---|---|---|---|---|




# _2_Transformer_Architecture

## _2-1_Quantization

### _2-1-1_Post-Traning-Quantization

| Title | arXiv | Github| WebSite | Pub. & Date
|:-----:|:-----:|:-----:|:-----:|:-----:|


### _2-1-2_Quantization-Aware-Traning

| Title | arXiv | Github| WebSite | Pub. & Date
|:-----:|:-----:|:-----:|:-----:|:-----:|


## _2-2_Pruning_and_Structure_Distillation

### _2-2-1_Pruning 

| Title | arXiv | Github | WebSite | Pub. & Date |
|---|---|---|---|---|


### _2-2-1_Structure_Distillation

| Title | arXiv | Github | WebSite | Pub. & Date |
|---|---|---|---|---|


## _2-3_Feature_Cache_and_Reuse

| Title | arXiv | Github | WebSite | Pub. & Date |
|---|---|---|---|---|


## _2-4_Timestep_Distallation

| Title | arXiv | Github | WebSite | Pub. & Date |
|---|---|---|---|---|


## _2-5_New_Architecture_Design

| Title | arXiv | Github | WebSite| Pub. & Date |
|---|---|---|---|---|


## _2-6_Other_Methods

| Title | arXiv | Github | WebSite| Pub. & Date |
|---|---|---|---|---|

