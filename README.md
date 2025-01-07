# 🚀 Papers-on-Efficient-Diffusion-Models


&emsp; &emsp; ***Purpose:** We aim to provide a summary of **Efficient Diffusion Models**. More papers will be summarized.*

&emsp; &emsp; ***Homepage:** University of Science and Technology of China **(USTC)**, [Intelligent Media Computing Lab **(IMCL)**](https://faculty.ustc.edu.cn/chenzhibo).*

&emsp; &emsp; ***Email:** xxxxxxx@mail.ustc.edu.cn*

&emsp; &emsp; &emsp; &emsp; *If you have any suggestions or find our work helpful, feel free to contact us.*
  
&emsp; &emsp; &emsp; &emsp; *If you find our survey is useful in your research or applications, please consider giving us a star 🌟.*


## 📚 Contents

- [**1_U-Net_Architecture**](#_1_U-Net_Architecture)
- - [1-1_Quantization](#_1-1_Quantization)
- - - [1-1-1_PTQ_(Post-Traning-Quantization)](#_1-1-1_Post-Traning-Quantization)
- - - [1-1-2_QAT_(Quantization-Aware-Traning)](#_1-1-2_Quantization-Aware-Traning)
- - [1-2_Pruning_&_Structure_Distillation](#_1-2_Pruning_and_Structure_Distillation)
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
- - [2-3_Feature Cache & Reuse](#_2-3_Feature_Cache_and_Reuse)
- - [2-4_Timestep Distallation](#_2-4_Timestep_Distallation)
- - [2-5_New Architecture Design](#_2-5_New_Architecture_Design)
- - [2-6_Other Methods](#_2-6_Other_Methods)


# _1_U-Net_Architecture

## _1-1_Quantization

### _1-1-1_Post-Traning-Quantization

| No. | Title | arXiv | Github | WebSite | Pub. or Date |
|:-----:|-----|:-----:|:-----:|:-----:|:-----:|
| [1] | [Post-training Quantization on Diffusion Models](https://arxiv.org/abs/2211.15736) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.15736) | [![Star](https://img.shields.io/github/stars/42Shawn/PTQ4DM.svg?style=social&label=Star)](https://github.com/42Shawn/PTQ4DM) | - | **CVPR 2023** |
| [2] | [Q-Diffusion: Quantizing Diffusion Models](https://arxiv.org/abs/2302.04304) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2302.04304) | [![Star](https://img.shields.io/github/stars/Xiuyu-Li/q-diffusion.svg?style=social&label=Star)](https://github.com/Xiuyu-Li/q-diffusion) | [![Website](https://img.shields.io/badge/Website-9cf)](https://xiuyuli.com/qdiffusion/) | **ICCV 2023** |
| [3] | [PTQD: Accurate Post-Training Quantization for Diffusion Models](https://arxiv.org/abs/2305.10657) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.10657) | [![Star](https://img.shields.io/github/stars/ziplab/PTQD.svg?style=social&label=Star)](https://github.com/ziplab/PTQD) | - | **NeurIPS 2023** |
| [4] | [Towards Accurate Post-training Quantization for Diffusion Models](https://arxiv.org/abs/2305.18723) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.18723) | [![Star](https://img.shields.io/github/stars/junhyukso/tdq.svg?style=social&label=Star)](https://github.com/junhyukso/tdq) | - | **NeurIPS 2023** |
| [5] | [Temporal Dynamic Quantization for Diffusion Models](https://arxiv.org/abs/2306.02316) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.02316) | - | - | Jun 2023 |
| [6] | [Softmax Bias Correction for Quantized Generative Models](https://arxiv.org/abs/2309.01729) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2309.01729) | - | - | **ICCV 2023** |
| [7] | [EfficientDM: Efficient Quantization-Aware Fine-Tuning of Low-Bit Diffusion Models](https://arxiv.org/abs/2310.03270) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.03270) | [![Star](https://img.shields.io/github/stars/ThisisBillhe/EfficientDM.svg?style=social&label=Star)](https://github.com/ThisisBillhe/EfficientDM) | - | **ICLR 2024** |
| [8] | [Post-training Quantization for Text-to-Image Diffusion Models with Progressive Calibration and Activation Relaxing](https://arxiv.org/abs/2311.06322) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2311.06322) | [![Star](https://img.shields.io/github/stars/tsa18/PCR.svg?style=social&label=Star)](https://github.com/tsa18/PCR) | - | **ECCV 2024** |
| [9] | [TFMQ-DM: Temporal Feature Maintenance Quantization for Diffusion Models](https://arxiv.org/abs/2311.16503) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2311.16503) | [![Star](https://img.shields.io/github/stars/ModelTC/TFMQ-DM.svg?style=social&label=Star)](https://github.com/ModelTC/TFMQ-DM) | [![Website](https://img.shields.io/badge/Website-9cf)](https://modeltc.github.io/TFMQ-DM/) | **CVPR 2024** |
| [10] | [Efficient Quantization Strategies for Latent Diffusion Models](https://arxiv.org/abs/2312.05431) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.05431) | [![Star](https://img.shields.io/github/stars/ThisisBillhe/EfficientDM.svg?style=social&label=Star)](https://github.com/ThisisBillhe/EfficientDM) | - | **ICLR 2024** |
| [11] | [EDA-DM: Enhanced Distribution Alignment for Post-Training Quantization of Diffusion Models](https://arxiv.org/abs/2401.04585) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2401.04585) | [![Star](https://img.shields.io/github/stars/BienLuky/EDA-DM.svg?style=social&label=Star)](https://github.com/BienLuky/EDA-DM) | - | Jan 2024 |
| [12] | [QNCD: Quantization Noise Correction for Diffusion Models](https://arxiv.org/abs/2403.19140) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2403.19140) | [![Star](https://img.shields.io/github/stars/huanpengchu/QNCD.svg?style=social&label=Star)](https://github.com/huanpengchu/QNCD) | - | **ACM MM 2024** |
| [13] | [TMPQ-DM: Joint Timestep Reduction and Quantization Precision Selection for Efficient Diffusion Models](https://arxiv.org/abs/2404.09532) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2404.09532) | [![Star](https://img.shields.io/github/stars/sihouzi21c/TMPQ-DM.svg?style=social&label=Star)](https://github.com/sihouzi21c/TMPQ-DM) | - | Apr 2024 |
| [14] | [MixDQ: Memory-Efficient Few-Step Text-to-Image Diffusion Models with Metric-Decoupled Mixed Precision Quantization](https://arxiv.org/abs/2405.17873) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2405.17873) | [![Star](https://img.shields.io/github/stars/thu-nics/MixDQ.svg?style=social&label=Star)](https://github.com/thu-nics/MixDQ) | [![Website](https://img.shields.io/badge/Website-9cf)](https://a-suozhang.xyz/mixdq.github.io/) | **ECCV 2024** |
| [15] | [BitsFusion: 1.99 bits Weight Quantization of Diffusion Model](https://arxiv.org/abs/2406.05723) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.05723) | [![Star](https://img.shields.io/github/stars/zhengchen1999/BI-DiffSR.svg?style=social&label=Star)](https://github.com/zhengchen1999/BI-DiffSR) | [![Website](https://img.shields.io/badge/Website-9cf)](https://zhengchen1999.github.io/BI-DiffSR-Web/) | **NeurIPS 2024** |
| [16] | [Binarized Diffusion Model for Image Super-Resolution](https://arxiv.org/abs/2406.04333) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.04333) | [![Star](https://img.shields.io/github/stars/snap-research/BitsFusion.svg?style=social&label=Star)](https://github.com/snap-research/BitsFusion) | [![Website](https://img.shields.io/badge/Website-9cf)](https://snap-research.github.io/BitsFusion/) | **NeurIPS 2024** |
| [17] | [Timestep-Aware Correction for Quantized Diffusion Models](https://arxiv.org/abs/2407.03917) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2407.03917) | - | - | **ECCV 2024** |
| [18] | [QVD: Post-training Quantization for Video Diffusion Models](https://arxiv.org/abs/2407.11585) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2407.11585) | - | - | **ACM MM 2024** |
| [19] | [Temporal Feature Matters: A Framework for Diffusion Model Quantization](https://arxiv.org/abs/2407.19547) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2407.19547) | - | - | Jul 2024 |
| [20] | [Low-Bitwidth Floating Point Quantization for Efficient High-Quality Diffusion Models](https://arxiv.org/abs/2408.06995) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2408.06995) | - | - | Aug 2024 |
| [21] | [PTQ4ADM: Post-Training Quantization for Efficient Text Conditional Audio Diffusion Models](https://arxiv.org/abs/2409.13894) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2409.13894) | - | - | Sep 2024 |
| [22] | [DilateQuant: Accurate and Efficient Diffusion Quantization via Weight Dilation](https://arxiv.org/abs/2409.14307) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2409.14307) | [![Star](https://img.shields.io/github/stars/BienLuky/DilateQuant.svg?style=social&label=Star)](https://github.com/BienLuky/DilateQuant) | - | Sep 2024 |
| [23] | [MPQ-Diff: Mixed Precision Quantization for Diffusion Models](https://arxiv.org/abs/2412.00144) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2412.00144) | - | - | Nov 2024 |


### _1-1-2_Quantization-Aware-Traning

| No. | Title | arXiv | Github | WebSite | Pub. or Date |
|:-----:|-----|:-----:|:-----:|:-----:|:-----:|
| [1] | [Q-DM: An Efficient Low-bit Quantized Diffusion Model](https://openreview.net/forum?id=sFGkL5BsPi) | [![arXiv](https://img.shields.io/badge/OpenReview-8c1b13.svg)](https://openreview.net/forum?id=sFGkL5BsPi) | - | - | **NeurIPS 2023** |
| [2] | [Effective Quantization for Diffusion Models on CPUs](https://arxiv.org/abs/2311.16133) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2311.16133) | [![Star](https://img.shields.io/github/stars/intel/intel-extension-for-transformers.svg?style=social&label=Star)](https://github.com/intel/intel-extension-for-transformers) | - | **NeurIPS 2023 Workshop** |
| [3] | [QuEST: Low-bit Diffusion Model Quantization via Efficient Selective Finetuning](https://arxiv.org/abs/2402.03666) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.03666) | [![Star](https://img.shields.io/github/stars/hatchetProject/QuEST.svg?style=social&label=Star)](https://github.com/hatchetProject/QuEST) | - | Feb 2024 |
| [4] | [BinaryDM: Accurate Weight Binarization for Efficient Diffusion Models](https://arxiv.org/abs/2404.05662) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2404.05662) | [![Star](https://img.shields.io/github/stars/Xingyu-Zheng/BinaryDM.svg?style=social&label=Star)](https://github.com/Xingyu-Zheng/BinaryDM) | - | **NeurIPS 2024 Workshop** |


## _1-2_Pruning_and_Structure_Distillation

| No. | Title | arXiv | Github | WebSite | Pub. or Date |
|:-----:|-----|:-----:|:-----:|:-----:|:-----:|
| [1] | [Diffusion Probabilistic Model Made Slim](https://arxiv.org/abs/2211.17106) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.17106) | - | - | **CVPR 2023** |
| [2] | [Structural Pruning for Diffusion Models](https://arxiv.org/abs/2305.10924) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.10924) | [![Star](https://img.shields.io/github/stars/VainF/Diff-Pruning.svg?style=social&label=Star)](https://github.com/VainF/Diff-Pruning) | - | **NeurIPS 2023** |
| [3] | [BK-SDM: A Lightweight, Fast, and Cheap Version of Stable Diffusion](https://arxiv.org/abs/2305.15798) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.15798) | [![Star](https://img.shields.io/github/stars/Nota-NetsPresso/BK-SDM.svg?style=social&label=Star)](https://github.com/Nota-NetsPresso/BK-SDM) | - | **ECCV 2024** |
| [4] | [SnapFusion: Text-to-Image Diffusion Model on Mobile Devices within Two Seconds](https://arxiv.org/abs/2306.00980) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.00980) | - | [![Website](https://img.shields.io/badge/Website-9cf)](https://snap-research.github.io/SnapFusion/) | **NeurIPS 2023** |
| [5] | [Squeezing Large-Scale Diffusion Models for Mobile](https://arxiv.org/abs/2307.01193) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2307.01193) | [![Star](https://img.shields.io/github/stars/Xingyu-Zheng/BinaryDM.svg?style=social&label=Star)](https://github.com/Xingyu-Zheng/BinaryDM) | - | **ICML 2023 Workshop** |
| [6] | [AutoDiffusion: Training-Free Optimization of Time Steps and Architectures for Automated Diffusion Model Acceleration](https://arxiv.org/abs/2309.10438) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2309.10438) | [![Star](https://img.shields.io/github/stars/lilijiangg/autodiffusion.svg?style=social&label=Star)](https://github.com/lilijiangg/autodiffusion) | - | **ICCV 2023** |
| [7] | [Lightweight Diffusion Models with Distillation-Based Block Neural Architecture Search](https://arxiv.org/abs/2311.04950) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2311.04950) | [![Star](https://img.shields.io/github/stars/Xingyu-Zheng/BinaryDM.svg?style=social&label=Star)](https://github.com/Xingyu-Zheng/BinaryDM) | - | Nov 2023 |
| [8] | [MobileDiffusion: Instant Text-to-Image Generation on Mobile Devices](https://arxiv.org/abs/2311.16567) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2311.16567) | [![Star](https://img.shields.io/github/stars/Xingyu-Zheng/BinaryDM.svg?style=social&label=Star)](https://github.com/Xingyu-Zheng/BinaryDM) | - | **ECCV 2024** |
| [9] | [KOALA: Empirical Lessons Toward Memory-Efficient and Fast Diffusion Models for Text-to-Image Synthesis](https://arxiv.org/abs/2312.04005) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.04005) | [![Star](https://img.shields.io/github/stars/youngwanLEE/sdxl-koala.svg?style=social&label=Star)](https://github.com/youngwanLEE/sdxl-koala) | [![Website](https://img.shields.io/badge/Website-9cf)](https://youngwanlee.github.io/KOALA/) | **NeurIPS 2024** |
| [10] | [Not All Steps are Equal: Efficient Generation with Progressive Diffusion Models](https://arxiv.org/abs/2312.13307) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.13307) | - | - | Dec 2023 |
| [11] | [A-SDM: Accelerating Stable Diffusion through Redundancy Removal and Performance Optimization](https://arxiv.org/abs/2312.15516) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.15516) | - | - | Dec 2023 |
| [12] | [Progressive Knowledge Distillation Of Stable Diffusion XL Using Layer Level Loss](https://arxiv.org/abs/2401.02677) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2401.02677) | - | - | Jan 2024 |
| [13] | [Diffusion Model Compression for Image-to-Image Translation](https://arxiv.org/abs/2401.17547) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2401.17547) | [![Star](https://img.shields.io/github/stars/KIMGEONUNG/ID-compression.svg?style=social&label=Star)](https://github.com/KIMGEONUNG/ID-compression) | [![Website](https://img.shields.io/badge/Website-9cf)](https://kimgeonung.github.io/id-compression/) | **ACCV 2024** |
| [14] | [SparseDM: Toward Sparse Efficient Diffusion Models](https://arxiv.org/abs/2404.10445) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2404.10445) | - | - | Apr 2024 |
| [15] | [LAPTOP-Diff: Layer Pruning and Normalized Distillation for Compressing Diffusion Models](https://arxiv.org/abs/2404.11098) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2404.11098) | - | - | Apr 2024 |
| [16] | [LD-Pruner: Efficient Pruning of Latent Diffusion Models using Task-Agnostic Insights](https://arxiv.org/abs/2404.11936) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2404.11936) | - | - | **CVPR 2024** |
| [17] | [Hybrid SD: Edge-Cloud Collaborative Inference for Stable Diffusion Models](https://arxiv.org/abs/2408.06646) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2408.06646) | - | - | Aug 2024 |
| [18] | [DKDM: Data-Free Knowledge Distillation for Diffusion Models with Any Architecture](https://arxiv.org/abs/2409.03550) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2409.03550) | - | - | Sep 2024 |
| [19] | [Pruning then Reweighting: Towards Data-Efficient Training of Diffusion Models](https://arxiv.org/abs/2409.19128) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2409.19128) | [![Star](https://img.shields.io/github/stars/Yeez-lee/Data-Selection-and-Reweighting-for-Diffusion-Models.svg?style=social&label=Star)](https://github.com/Yeez-lee/Data-Selection-and-Reweighting-for-Diffusion-Models) | - | Sep 2024 |
| [20] | [DiP-GO: A Diffusion Pruner via Few-step Gradient Optimization](https://arxiv.org/abs/2410.16942) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.16942) | - | - | **NeurIPS 2024** |


## _1-3_Feature_Cache_and_Reuse

| No. | Title | arXiv | Github | WebSite | Pub. or Date |
|:-----:|-----|:-----:|:-----:|:-----:|:-----:|
| [1] | [Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models](https://arxiv.org/abs/2211.02048) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.02048) | [![Star](https://img.shields.io/github/stars/lmxyy/sige.svg?style=social&label=Star)](https://github.com/lmxyy/sige) | [![Website](https://img.shields.io/badge/Website-9cf)](https://www.cs.cmu.edu/~sige/) | **NeurIPS 2022<br>T-PAMI 2023** |
| [2] | [Accelerating Text-to-Image Editing via Cache-Enabled Sparse Diffusion Inference](https://arxiv.org/abs/2305.17423) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.17423) | [![Star](https://img.shields.io/github/stars/pku-dair/hetu.svg?style=social&label=Star)](https://github.com/pku-dair/hetu) | - | **AAAI 2024** |
| [3] | [DeepCache: Accelerating Diffusion Models for Free](https://arxiv.org/abs/2312.00858) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.00858) | [![Star](https://img.shields.io/github/stars/horseee/DeepCache.svg?style=social&label=Star)](https://github.com/horseee/DeepCache) | - | **CVPR 2024** |
| [4] | [Cache Me if You Can: Accelerating Diffusion Models through Block Caching](https://arxiv.org/abs/2312.03209) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.03209) | - | [![Website](https://img.shields.io/badge/Website-9cf)](https://fwmb.github.io/blockcaching/) | **CVPR 2024** |
| [5] | [FRDiff: Feature Reuse for Universal Training-free Acceleration of Diffusion Models](https://arxiv.org/abs/2312.03517) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.03517) | [![Star](https://img.shields.io/github/stars/ECoLab-POSTECH/FRDiff.svg?style=social&label=Star)](https://github.com/ECoLab-POSTECH/FRDiff) | - | **ECCV 2024** |
| [6] | [Approximate Caching for Efficiently Serving Diffusion Models](https://arxiv.org/abs/2312.04429) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.04429) | [![Star](https://img.shields.io/github/stars/ECoLab-POSTECH/FRDiff.svg?style=social&label=Star)](https://github.com/ECoLab-POSTECH/FRDiff) | - | **NSDI 2024** |
| [7] | [Clockwork Diffusion: Efficient Generation With Model-Step Distillation](https://arxiv.org/abs/2312.08128) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.08128) | [![Star](https://img.shields.io/github/stars/Qualcomm-AI-research/clockwork-diffusion.svg?style=social&label=Star)](https://github.com/Qualcomm-AI-research/clockwork-diffusion) | - | **CVPR 2024** |
| [8] | [Faster Diffusion: Rethinking the Role of UNet Encoder in Diffusion Models](https://arxiv.org/abs/2312.09608) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.09608) | [![Star](https://img.shields.io/github/stars/hutaiHang/Faster-Diffusion.svg?style=social&label=Star)](https://github.com/hutaiHang/Faster-Diffusion) | [![Website](https://img.shields.io/badge/Website-9cf)](https://sen-mao.github.io/FasterDiffusion/) | **NeurIPS 2024** |
| [9] | [Fast Sampling through the Reuse of Attention Maps in Diffusion Models](https://arxiv.org/abs/2401.01008) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2401.01008) | - | - | Dec 2023 |
| [10] | [DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models](https://arxiv.org/abs/2402.19481) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.19481) | [![Star](https://img.shields.io/github/stars/mit-han-lab/distrifuser.svg?style=social&label=Star)](https://github.com/mit-han-lab/distrifuser) | [![Website](https://img.shields.io/badge/Website-9cf)](https://hanlab.mit.edu/projects/distrifusion) | **CVPR 2024 Highlight** |
| [11] | [Faster Diffusion via Temporal Attention Decomposition](https://arxiv.org/abs/2404.02747) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2404.02747) | [![Star](https://img.shields.io/github/stars/HaozheLiu-ST/T-GATE.svg?style=social&label=Star)](https://github.com/HaozheLiu-ST/T-GATE) | - | Apr 2024 |
| [12] | [Hash3D: Training-free Acceleration for 3D Generation](https://arxiv.org/abs/2404.06091) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2404.06091) | [![Star](https://img.shields.io/github/stars/Adamdad/hash3D.svg?style=social&label=Star)](https://github.com/Adamdad/hash3D) | [![Website](https://img.shields.io/badge/Website-9cf)](https://adamdad.github.io/hash3D/) | Apr 2024 |
| [13] | [PFDiff: Training-free Acceleration of Diffusion Models through the Gradient Guidance of Past and Future](https://arxiv.org/abs/2408.08822) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2408.08822) | - | - | Aug 2024 |
| [14] | [DiP-GO: A Diffusion Pruner via Few-step Gradient Optimization](https://arxiv.org/abs/2410.16942) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.16942) | - | - | **NeurIPS 2024** |
| [15] | [Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing](https://arxiv.org/abs/2411.16375) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2411.16375) | [![Star](https://img.shields.io/github/stars/Dawn-LX/CausalCache-VDM.svg?style=social&label=Star)](https://github.com/Dawn-LX/CausalCache-VDM) | - | Nov 2024 |

## _1-4_Timestep_Distallation

### _1-4-1_ODE_Preserving_Distallation

| No. | Title | arXiv | Github | WebSite | Pub. or Date |
|:-----:|-----|:-----:|:-----:|:-----:|:-----:|
| [1] | [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2010.02502) | [![Star](https://img.shields.io/github/stars/ermongroup/ddim.svg?style=social&label=Star)](https://github.com/ermongroup/ddim) | - | **ICLR 2021** |
| [2] | [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2011.13456) | [![Star](https://img.shields.io/github/stars/yang-song/score_sde.svg?style=social&label=Star)](https://github.com/yang-song/score_sde) | - | **ICLR 2021 Oral** |

### _1-4-2_ODE_Reconstructing_Distallation

| No. | Title | arXiv | Github | WebSite | Pub. or Date |
|:-----:|-----|:-----:|:-----:|:-----:|:-----:|
| [1] | [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2010.02502) | [![Star](https://img.shields.io/github/stars/ermongroup/ddim.svg?style=social&label=Star)](https://github.com/ermongroup/ddim) | - | **ICLR 2021** |
| [2] | [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2011.13456) | [![Star](https://img.shields.io/github/stars/yang-song/score_sde.svg?style=social&label=Star)](https://github.com/yang-song/score_sde) | - | **ICLR 2021 Oral** |


## _1-5_Fast_Sampling_Solver

| No. | Title | arXiv | Github | WebSite | Pub. or Date |
|:-----:|-----|:-----:|:-----:|:-----:|:-----:|
| [1] | [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2010.02502) | [![Star](https://img.shields.io/github/stars/ermongroup/ddim.svg?style=social&label=Star)](https://github.com/ermongroup/ddim) | - | **ICLR 2021** |
| [2] | [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2011.13456) | [![Star](https://img.shields.io/github/stars/yang-song/score_sde.svg?style=social&label=Star)](https://github.com/yang-song/score_sde) | - | **ICLR 2021 Oral** |
| [3] | [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2102.09672) | [![Star](https://img.shields.io/github/stars/openai/improved-diffusion.svg?style=social&label=Star)](https://github.com/openai/improved-diffusion) | - | **ICLR 2021** |
| [4] | [Gotta Go Fast When Generating Data with Score-Based Models](https://arxiv.org/abs/2105.14080) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2105.14080) | [![Star](https://img.shields.io/github/stars/AlexiaJM/score_sde_fast_sampling.svg?style=social&label=Star)](https://github.com/AlexiaJM/score_sde_fast_sampling) | - | May 2021 |
| [5] | [On Fast Sampling of Diffusion Probabilistic Models](https://arxiv.org/abs/2106.00132) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2106.00132) | [![Star](https://img.shields.io/github/stars/zhifengkong/FastDPM_pytorch.svg?style=social&label=Star)](https://github.com/zhifengkong/FastDPM_pytorch) | - | **ICML 2021 Workshop Spotlight** |
| [6] | [Learning to Efficiently Sample from Diffusion Probabilistic Models](https://arxiv.org/abs/2106.03802) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2106.03802) | - | - | Jun 2021 |
| [7] | [Score-Based Generative Modeling with Critically-Damped Langevin Diffusion](https://arxiv.org/abs/2112.07068) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2112.07068) | [![Star](https://img.shields.io/github/stars/nv-tlabs/CLD-SGM.svg?style=social&label=Star)](https://github.com/nv-tlabs/CLD-SGM) | [![Website](https://img.shields.io/badge/Website-9cf)](https://research.nvidia.com/labs/toronto-ai/CLD-SGM/) | **ICLR 2022 Spotlight** |
| [8] | [Learning Fast Samplers for Diffusion Models by Differentiating Through Sample Quality](https://arxiv.org/abs/2202.05830) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2202.05830) | - | - | **ICLR 2022** |
| [9] | [Pseudo Numerical Methods for Diffusion Models on Manifolds](https://arxiv.org/abs/2202.09778) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2202.09778) | [![Star](https://img.shields.io/github/stars/luping-liu/PNDM.svg?style=social&label=Star)](https://github.com/luping-liu/PNDM) | - | **ICLR 2022** |
| [10] | [Fast Sampling of Diffusion Models with Exponential Integrator](https://arxiv.org/abs/2204.13902) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2204.13902) | [![Star](https://img.shields.io/github/stars/qsh-zh/deis.svg?style=social&label=Star)](https://github.com/qsh-zh/deis) | [![Website](https://img.shields.io/badge/Website-9cf)](https://qsh-zh.github.io/deis/) | **ICLR 2023** |
| [11] | [DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://arxiv.org/abs/2202.09778) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2202.09778) | [![Star](https://img.shields.io/github/stars/LuChengTHU/dpm-solver.svg?style=social&label=Star)](https://github.com/LuChengTHU/dpm-solver) | - | **NeurIPS 2022 Oral** |
| [12] | [Improving Diffusion Models for Inverse Problems using Manifold Constraints](https://arxiv.org/abs/2206.00941) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2206.00941) | [![Star](https://img.shields.io/github/stars/HJ-harry/MCG_diffusion.svg?style=social&label=Star)](https://github.com/HJ-harry/MCG_diffusion) | - | **NeurIPS 2022** |
| [13] | [gDDIM: Generalized Denoising Diffusion Implicit Models](https://arxiv.org/abs/2206.05564) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2206.05564) | [![Star](https://img.shields.io/github/stars/qsh-zh/gDDIM.svg?style=social&label=Star)](https://github.com/qsh-zh/gDDIM) | - | **ICLR 2023 Spotlight** |
| [14] | [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2209.03003) | [![Star](https://img.shields.io/github/stars/gnobitab/RectifiedFlow.svg?style=social&label=Star)](https://github.com/gnobitab/RectifiedFlow) | - | **ICLR 2023 Spotlight** |
| [15] | [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.02747) | [![Star](https://img.shields.io/github/stars/aleksandrinvictor/flow-matching.svg?style=social&label=Star)](https://github.com/aleksandrinvictor/flow-matching) | - | **ICLR 2023** |
| [16] | [GENIE: Higher-Order Denoising Diffusion Solvers](https://arxiv.org/abs/2210.05475) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.05475) | [![Star](https://img.shields.io/github/stars/nv-tlabs/GENIE.svg?style=social&label=Star)](https://github.com/nv-tlabs/GENIE) | [![Website](https://img.shields.io/badge/Website-9cf)](https://research.nvidia.com/labs/toronto-ai/GENIE/) | **NeurIPS 2022** |
| [17] | [Deep Equilibrium Approaches to Diffusion Models](https://arxiv.org/abs/2210.12867) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.12867) | [![Star](https://img.shields.io/github/stars/locuslab/deq-ddim.svg?style=social&label=Star)](https://github.com/locuslab/deq-ddim) | - | **NeurIPS 2022** |
| [18] | [DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models](https://arxiv.org/abs/2211.01095) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.01095) | [![Star](https://img.shields.io/github/stars/LuChengTHU/dpm-solver.svg?style=social&label=Star)](https://github.com/LuChengTHU/dpm-solver) | - | Nov 2022 |
| [19] | [UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2302.04867) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2302.04867) | [![Star](https://img.shields.io/github/stars/wl-zhao/unipc.svg?style=social&label=Star)](https://github.com/wl-zhao/unipc) | [![Website](https://img.shields.io/badge/Website-9cf)](https://unipc.ivg-research.xyz/) | **NeurIPS 2023** |
| [20] | [On Accelerating Diffusion-Based Sampling Process via Improved Integration Approximation](https://arxiv.org/abs/2304.11328) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.11328) | - | - | **ICLR 2023** |
| [21] | [SEEDS: Exponential SDE Solvers for Fast High-Quality Sampling from Diffusion Models](https://arxiv.org/abs/2305.14267) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.14267) | [![Star](https://img.shields.io/github/stars/nfsrules/seeds.svg?style=social&label=Star)](https://github.com/nfsrules/seeds) | - | **NeurIPS 2023** |
| [22] | [Restart Sampling for Improving Generative Processes](https://arxiv.org/abs/2306.14878) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.14878) | [![Star](https://img.shields.io/github/stars/Newbeeer/diffusion_restart_sampling.svg?style=social&label=Star)](https://github.com/Newbeeer/diffusion_restart_sampling) | - | **NeurIPS 2023** |
| [23] | [Learning to Schedule in Diffusion Probabilistic Models](https://dl.acm.org/doi/abs/10.1145/3580305.3599412) | [![arXiv](https://img.shields.io/badge/ACM-000000.svg)](https://dl.acm.org/doi/abs/10.1145/3580305.3599412) | - | - | **ACM SIGKDD 2023** |
| [24] | [SciRE-Solver: Accelerating Diffusion Models Sampling by Score-integrand Solver with Recursive Difference](https://arxiv.org/abs/2308.07896) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2308.07896) | [![Star](https://img.shields.io/github/stars/ShiguiLi/scire-solver.svg?style=social&label=Star)](https://github.com/ShiguiLi/scire-solver) | - | Aug 2023 |
| [25] | [SA-Solver: Stochastic Adams Solver for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2309.05019) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2309.05019) | [![Star](https://img.shields.io/github/stars/scxue/SA-Solver.svg?style=social&label=Star)](https://github.com/scxue/SA-Solver) | - | **NeurIPS 2023** |
| [26] | [DPM-Solver-v3: Improved Diffusion ODE Solver with Empirical Model Statistics](https://arxiv.org/abs/2310.13268) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.13268) | [![Star](https://img.shields.io/github/stars/thu-ml/DPM-Solver-v3.svg?style=social&label=Star)](https://github.com/thu-ml/DPM-Solver-v3) | [![Website](https://img.shields.io/badge/Website-9cf)](https://ml.cs.tsinghua.edu.cn/dpmv3/) | **NeurIPS 2023** |
| [27] | [AdaDiff: Adaptive Step Selection for Fast Diffusion](https://arxiv.org/abs/2311.14768) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2311.14768) | [![Star](https://img.shields.io/github/stars/Tangshengku/AdaDiff.svg?style=social&label=Star)](https://github.com/Tangshengku/AdaDiff) | - | **ECCV 2024** |
| [28] | [A Unified Sampling Framework for Solver Searching of Diffusion Probabilistic Models](https://arxiv.org/abs/2312.07243) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.07243) | [![Star](https://img.shields.io/github/stars/thu-nics/USF.svg?style=social&label=Star)](https://github.com/thu-nics/USF) | - | **ICLR 2024** |
| [29] | [Accelerating Diffusion Sampling with Optimized Time Steps](https://arxiv.org/abs/2402.17376) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.17376) | [![Star](https://img.shields.io/github/stars/scxue/DM-NonUniform.svg?style=social&label=Star)](https://github.com/scxue/DM-NonUniform) | - | **CVPR 2024** |
| [30] | [Bespoke Non-Stationary Solvers for Fast Sampling of Diffusion and Flow Models](https://arxiv.org/abs/2403.01329) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2403.01329) | - | - | Mar 2024 |
| [31] | [PFDiff: Training-free Acceleration of Diffusion Models through the Gradient Guidance of Past and Future](https://arxiv.org/abs/2408.08822) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2408.08822) | - | - | Aug 2024 |
| [32] | [DC-Solver: Improving Predictor-Corrector Diffusion Sampler via Dynamic Compensation](https://arxiv.org/abs/2409.03755) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2409.03755) | [![Star](https://img.shields.io/github/stars/wl-zhao/dc-solver.svg?style=social&label=Star)](https://github.com/wl-zhao/dc-solver) | - | **ECCV 2024** |
| [33] | [*Jump Your Steps*: Optimizing Sampling Schedule of Discrete Diffusion Models](https://arxiv.org/abs/2410.07761) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.07761) | - | - | Oct 2024 |


## _1-6_GAN-Based_Method

| No. | Title | arXiv | Github | WebSite | Pub. or Date |
|:-----:|-----|:-----:|:-----:|:-----:|:-----:|
| [1] | [Tackling the Generative Learning Trilemma with Denoising Diffusion GANs](https://arxiv.org/abs/2112.07804) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2112.07804) | [![Star](https://img.shields.io/github/stars/NVlabs/denoising-diffusion-gan.svg?style=social&label=Star)](https://github.com/NVlabs/denoising-diffusion-gan) | - | **ICLR 2022 Spotlight** |
| [2] | [Diffusion-GAN: Training GANs with Diffusion](https://arxiv.org/abs/2206.02262) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2206.02262) | [![Star](https://img.shields.io/github/stars/Zhendong-Wang/Diffusion-GAN.svg?style=social&label=Star)](https://github.com/Zhendong-Wang/Diffusion-GAN) | - | **ICLR 2023** |
| [3] | [Semi-Implicit Denoising Diffusion Models (SIDDMs)](https://arxiv.org/abs/2306.12511) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.12511) | [![Star](https://img.shields.io/github/stars/xuyanwu/SIDDMs-UFOGen.svg?style=social&label=Star)](https://github.com/xuyanwu/SIDDMs-UFOGen) | - | **NeurIPS 2023** |
| [4] | [Consistency Trajectory Models: Learning Probability Flow ODE Trajectory of Diffusion](https://arxiv.org/abs/2310.02279) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.02279) | [![Star](https://img.shields.io/github/stars/sony/ctm.svg?style=social&label=Star)](https://github.com/sony/ctm) | [![Website](https://img.shields.io/badge/Website-9cf)](https://consistencytrajectorymodel.github.io/CTM/) | **ICLR 2024** |
| [5] | [UFOGen: You Forward Once Large Scale Text-to-Image Generation via Diffusion GANs](https://arxiv.org/abs/2311.09257) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2311.09257) | [![Star](https://img.shields.io/github/stars/xuyanwu/SIDDMs-UFOGen.svg?style=social&label=Star)](https://github.com/xuyanwu/SIDDMs-UFOGen) | - | **CVPR 2024** |
| [6] | [Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2311.17042) | [![Star](https://img.shields.io/github/stars/Stability-AI/generative-models.svg?style=social&label=Star)](https://github.com/Stability-AI/generative-models) | - | **ECCV 2025** |
| [7] | [SDXL-Lightning: Progressive Adversarial Diffusion Distillation](https://arxiv.org/abs/2402.13929) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.13929) | [![Star](https://img.shields.io/github/stars/inferless/SDXL-Lightning.svg?style=social&label=Star)](https://github.com/inferless/SDXL-Lightning) | - | Feb 2024 |
| [8] | [Fast High-Resolution Image Synthesis with Latent Adversarial Diffusion Distillation](https://arxiv.org/abs/2403.12015) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2403.12015) | - | - | **ACM SIGGRAPH Asia 2024** |
| [9] | [UniFL: Improve Latent Diffusion Model via Unified Feedback Learning](https://arxiv.org/abs/2404.05595) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2404.05595) | - | [![Website](https://img.shields.io/badge/Website-9cf)](https://uni-fl.github.io/) | **NeurIPS 2024** |
| [10] | [Improved Distribution Matching Distillation for Fast Image Synthesis](https://arxiv.org/abs/2405.14867) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2405.14867) | [![Star](https://img.shields.io/github/stars/tianweiy/DMD2.svg?style=social&label=Star)](https://github.com/tianweiy/DMD2) | [![Website](https://img.shields.io/badge/Website-9cf)](https://tianweiy.github.io/dmd2/) | **NeurIPS 2024 Oral** |
| [11] | [Latent Denoising Diffusion GAN: Faster Sampling, Higher Image Quality](https://arxiv.org/abs/2406.11713) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.11713) | [![Star](https://img.shields.io/github/stars/Zhendong-Wang/Diffusion-GAN.svg?style=social&label=Star)](https://github.com/Zhendong-Wang/Diffusion-GAN) | - | **IEEE Access 2024** |
| [12] | [NitroFusion: High-Fidelity Single-Step Diffusion through Dynamic Adversarial Training](https://arxiv.org/abs/2412.02030) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2412.02030) | [![Star](https://img.shields.io/github/stars/ChenDarYen/NitroFusion.svg?style=social&label=Star)](https://github.com/ChenDarYen/NitroFusion) | [![Website](https://img.shields.io/badge/Website-9cf)](https://chendaryen.github.io/NitroFusion.github.io/) | Dec 2024 |
| [13] | [Accelerating Video Diffusion Models via Distribution Matching](https://arxiv.org/abs/2412.05899) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2412.05899) | - | - | Dec 2024 |
| [14] | [From Slow Bidirectional to Fast Causal Video Generators](https://arxiv.org/abs/2412.07772) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2412.07772) | - | [![Website](https://img.shields.io/badge/Website-9cf)](https://causvid.github.io/) | Dec 2024 |


## _1-7_Efficient_Training

| No. | Title | arXiv | Github | WebSite | Pub. or Date |
|:-----:|-----|:-----:|:-----:|:-----:|:-----:|


## _1-8_New_Architecture_Design

| No. | Title | arXiv | Github | WebSite | Pub. or Date |
|:-----:|-----|:-----:|:-----:|:-----:|:-----:|


## _1-9_Other_Methods

| No. | Title | arXiv | Github | WebSite | Pub. or Date |
|:-----:|-----|:-----:|:-----:|:-----:|:-----:|




# _2_Transformer_Architecture

## _2-1_Quantization

### _2-1-1_Post-Traning-Quantization

| No. | Title | arXiv | Github | WebSite | Pub. or Date |
|:-----:|-----|:-----:|:-----:|:-----:|:-----:|
| [1] | [TerDiT: Ternary Diffusion Models with Transformers](https://arxiv.org/abs/2405.14854) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2405.14854) | [![Star](https://img.shields.io/github/stars/Lucky-Lance/TerDiT.svg?style=social&label=Star)](https://github.com/Lucky-Lance/TerDiT) | - | May 2024 |
| [2] | [PTQ4DiT: Post-training Quantization for Diffusion Transformers](https://arxiv.org/abs/2405.16005) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2405.16005) | [![Star](https://img.shields.io/github/stars/adreamwu/PTQ4DiT.svg?style=social&label=Star)](https://github.com/adreamwu/PTQ4DiT) | - | **NeurIPS 2024** |
| [3] | [HQ-DiT: Efficient Diffusion Transformer with FP4 Hybrid Quantization](https://arxiv.org/abs/2405.19751) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2405.19751) | - | - | May 2024 |
| [4] | [ViDiT-Q: Efficient and Accurate Quantization of Diffusion Transformers for Image and Video Generation](https://arxiv.org/abs/2406.02540) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.02540) | [![Star](https://img.shields.io/github/stars/thu-nics/ViDiT-Q.svg?style=social&label=Star)](https://github.com/thu-nics/ViDiT-Q) | [![Website](https://img.shields.io/badge/Website-9cf)](https://a-suozhang.xyz/viditq.github.io/) | Jun 2024 |
| [5] | [An Analysis on Quantizing Diffusion Transformers](https://arxiv.org/abs/2406.11100) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.11100) | [![Star](https://img.shields.io/github/stars/adreamwu/PTQ4DiT.svg?style=social&label=Star)](https://github.com/adreamwu/PTQ4DiT) | - | **CVPR 2024 Workshop** |
| [6] | [Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers](https://arxiv.org/abs/2406.17343) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.17343) | [![Star](https://img.shields.io/github/stars/Juanerx/Q-DiT.svg?style=social&label=Star)](https://github.com/Juanerx/Q-DiT) | - | **NeurIPS 2024** |
| [7] | [DiTAS: Quantizing Diffusion Transformers via Enhanced Activation Smoothing](https://arxiv.org/abs/2409.07756) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2409.07756) | [![Star](https://img.shields.io/github/stars/DZY122/DiTAS.svg?style=social&label=Star)](https://github.com/DZY122/DiTAS) | - | **WACV 2025** |
| [8] | [SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models](https://arxiv.org/abs/2411.05007) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2411.05007) | [![Star](https://img.shields.io/github/stars/mit-han-lab/nunchaku.svg?style=social&label=Star)](https://github.com/mit-han-lab/nunchaku) | [![Website](https://img.shields.io/badge/Website-9cf)](https://hanlab.mit.edu/projects/svdquant) | Nov 2024 |
| [9] | [TaQ-DiT: Time-aware Quantization for Diffusion Transformers](https://arxiv.org/abs/2411.14172) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2411.14172) | - | - | Nov 2024 |
| [10] | [Qua²SeDiMo: Quantifiable Quantization Sensitivity of Diffusion Models](https://arxiv.org/abs/2412.14628) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2412.14628) | [![Star](https://img.shields.io/github/stars/Ascend-Research/Qua2SeDiMo.svg?style=social&label=Star)](https://github.com/Ascend-Research/Qua2SeDiMo) | [![Website](https://img.shields.io/badge/Website-9cf)](https://kgmills.github.io/projects/qua2sedimo/) | **AAAI 2025** |


### _2-1-2_Quantization-Aware-Traning

| No. | Title | arXiv | Github | WebSite | Pub. or Date |
|:-----:|-----|:-----:|:-----:|:-----:|:-----:|


## _2-2_Pruning_and_Structure_Distillation

| No. | Title | arXiv | Github | WebSite | Pub. or Date |
|:-----:|-----|:-----:|:-----:|:-----:|:-----:|
| [1] | [DiP-GO: A Diffusion Pruner via Few-step Gradient Optimization](https://arxiv.org/abs/2410.16942) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.16942) | - | - | **NeurIPS 2024** |
| [2] | [TinyFusion: Diffusion Transformers Learned Shallow](https://arxiv.org/abs/2412.01199) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2412.01199) | [![Star](https://img.shields.io/github/stars/VainF/TinyFusion.svg?style=social&label=Star)](https://github.com/VainF/TinyFusion) | - | Dec 2024 |


## _2-3_Feature_Cache_and_Reuse

| No. | Title | arXiv | Github | WebSite | Pub. or Date |
|:-----:|-----|:-----:|:-----:|:-----:|:-----:|
| [1] | [Δ-DiT: A Training-Free Acceleration Method Tailored for Diffusion Transformers](https://arxiv.org/abs/2406.01125) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.01125) | - | - | Jun 2024 |
| [2] | [Learning-to-Cache: Accelerating Diffusion Transformer via Layer Caching](https://arxiv.org/abs/2406.01733) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.01733) | [![Star](https://img.shields.io/github/stars/horseee/learning-to-cache.svg?style=social&label=Star)](https://github.com/horseee/learning-to-cache) | - | **NeurIPS 2024** |
| [3] | [DiTFastAttn: Attention Compression for Diffusion Transformer Models](https://arxiv.org/abs/2406.08552) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.08552) | [![Star](https://img.shields.io/github/stars/thu-nics/DiTFastAttn.svg?style=social&label=Star)](https://github.com/thu-nics/DiTFastAttn) | [![Website](https://img.shields.io/badge/Website-9cf)](https://nics-effalg.com/DiTFastAttn) | **NeurIPS 2024** |
| [4] | [FORA: Fast-Forward Caching in Diffusion Transformer Acceleration](https://arxiv.org/abs/2407.01425) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2407.01425) | [![Star](https://img.shields.io/github/stars/prathebaselva/FORA.svg?style=social&label=Star)](https://github.com/prathebaselva/FORA) | - | Jul 2024 |
| [5] | [Real-Time Video Generation with Pyramid Attention Broadcast](https://arxiv.org/abs/2408.12588) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2408.12588) | [![Star](https://img.shields.io/github/stars/NUS-HPC-AI-Lab/VideoSys.svg?style=social&label=Star)](https://github.com/NUS-HPC-AI-Lab/VideoSys) | [![Website](https://img.shields.io/badge/Website-9cf)](https://oahzxl.github.io/PAB/) | Aug 2024 |
| [6] | [Token Caching for Diffusion Transformer Acceleration](https://arxiv.org/abs/2409.18523) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2409.18523) | - | - | Sep 2024 |
| [7] | [HarmoniCa: Harmonizing Training and Inference for Better Feature Cache in Diffusion Transformer Acceleration](https://arxiv.org/abs/2410.01723) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.01723) | - | - | Oct 2024 |
| [8] | [Accelerating Diffusion Transformers with Token-wise Feature Caching](https://arxiv.org/abs/2410.05317) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.05317) | [![Star](https://img.shields.io/github/stars/Shenyi-Z/ToCa.svg?style=social&label=Star)](https://github.com/Shenyi-Z/ToCa) | [![Website](https://img.shields.io/badge/Website-9cf)](https://toca2024.github.io/ToCa/) | Oct 2024 |
| [9] | [MD-DiT: Step-aware Mixture-of-Depths for Efficient Diffusion Transformers](https://openreview.net/forum?id=1jWhiakK7N) | [![arXiv](https://img.shields.io/badge/OpenReview-8c1b13.svg)](https://openreview.net/forum?id=1jWhiakK7N) | - | - | **NeurIPS 2024 Workshop** |
| [10] | [DiP-GO: A Diffusion Pruner via Few-step Gradient Optimization](https://arxiv.org/abs/2410.16942) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.16942) | - | - | **NeurIPS 2024** |
| [11] | [FasterCache: Training-Free Video Diffusion Model Acceleration with High Quality](https://arxiv.org/abs/2410.19355) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.19355) | [![Star](https://img.shields.io/github/stars/Vchitect/FasterCache.svg?style=social&label=Star)](https://github.com/Vchitect/FasterCache) | [![Website](https://img.shields.io/badge/Website-9cf)](https://vchitect.github.io/FasterCache/) | Oct 2024 |
| [12] | [Adaptive Caching for Faster Video Generation with Diffusion Transformers](https://arxiv.org/abs/2411.02397) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2411.02397) | [![Star](https://img.shields.io/github/stars/AdaCache-DiT/AdaCache.svg?style=social&label=Star)](https://github.com/AdaCache-DiT/AdaCache) | [![Website](https://img.shields.io/badge/Website-9cf)](https://adacache-dit.github.io/) | Nov 2024 |
| [13] | [SmoothCache: A Universal Inference Acceleration Technique for Diffusion Transformers](https://arxiv.org/abs/2411.10510) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2411.10510) | [![Star](https://img.shields.io/github/stars/Roblox/SmoothCache.svg?style=social&label=Star)](https://github.com/Roblox/SmoothCache) | - | Nov 2024 |
| [14] | [Unveiling Redundancy in Diffusion Transformers (DiTs): A Systematic Study](https://arxiv.org/abs/2411.13588) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2411.13588) | [![Star](https://img.shields.io/github/stars/xdit-project/DiTCacheAnalysis.svg?style=social&label=Star)](https://github.com/xdit-project/DiTCacheAnalysis) | - | Nov 2024 |
| [15] | [Accelerating Vision Diffusion Transformers with Skip Branches](https://arxiv.org/abs/2411.17616) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2411.17616) | [![Star](https://img.shields.io/github/stars/OpenSparseLLMs/Skip-DiT.svg?style=social&label=Star)](https://github.com/OpenSparseLLMs/Skip-DiT) | - | Nov 2024 |
| [16] | [AsymRnR: Video Diffusion Transformers Acceleration with Asymmetric Reduction and Restoration](https://arxiv.org/abs/2412.11706) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2412.11706) | - | - | Dec 2024 |


## _2-4_Timestep_Distallation

| No. | Title | arXiv | Github | WebSite | Pub. or Date |
|:-----:|-----|:-----:|:-----:|:-----:|:-----:|


## _2-5_New_Architecture_Design

| No. | Title | arXiv | Github | WebSite | Pub. or Date |
|:-----:|-----|:-----:|:-----:|:-----:|:-----:|


## _2-6_Other_Methods

| No. | Title | arXiv | Github | WebSite | Pub. or Date |
|:-----:|-----|:-----:|:-----:|:-----:|:-----:|

