# Research-on-Efficient-Diffusion-Model-Research
Research on Efficient Diffusion Model Research

**Purpose:** We aim to provide a summary of neural image compression. More papers will be summarized. 


## Table of Contents

- [No.1_U-Net_Architecture](#No.1_U-Net_Architecture)
- - [1.1. Quantization](#1.1. Quantization)
- - - [1.1.1. PTQ (Post-Traning-Quantization)](#1.1.1. PTQ (Post-Traning-Quantization))
- - - [1.1.2. QAT (Quantization-Aware-Traning)](#1.1.2. QAT (Quantization-Aware-Traning))
- - [1.2. Pruning & Structure Distillation](#1.2. Pruning & Structure Distillation)
- - - [1.2.1. Pruning ](#1.2.1. Pruning )
- - - [1.2.2. Structure Distillation](#1.2.2. Structure Distillation)
- - [1.3. Feature Cache & Reuse](#1.3. Feature Cache & Reuse)
- - [1.4. Timestep Distallation](#1.4. Timestep Distallation)
- - - [1.4.1. ODE Preserving Distallation](#1.4.1. ODE Preserving Distallation)
- - - [1.4.2. ODE Reconstructing Distallation](#1.4.2. ODE Reconstructing Distallation)
- - [1.5. Fast Sampling Solver](#1.5. Fast Sampling Solver)
- - [1.6. GAN-Based Method](#1.6. GAN-Based Method)
- - [1.7. Efficient Training](#1.7. Efficient Training)
- - [1.8. New Architecture Design](#1.8. New Architecture Design)
- - [1.9. Other Methods](#1.9. Other Methods)
- [2. Transformer Architecture](#2. Transformer Architecture)
- - [2.1. Quantization](#2.1. Quantization)
- - - [2.1.1. PTQ (Post-Traning-Quantization)](#2.1.1. PTQ (Post-Traning-Quantization))
- - - [2.1.2. QAT (Quantization-Aware-Traning)](#2.1.2. QAT (Quantization-Aware-Traning))
- - [2.2. Pruning & Structure Distillation](#2.2. Pruning & Structure Distillation)
- - - [2.2.1. Pruning ](#2.2.1. Pruning )
- - - [2.2.2. Structure Distillation](#2.2.2. Structure Distillation)
- - [2.3. Feature Cache & Reuse](#2.3. Feature Cache & Reuse)
- - [2.4. Timestep Distallation](#2.4. Timestep Distallation)
- - [2.5. New Architecture Design](#2.5. New Architecture Design)
- - [2.6. Other Methods](#2.6. Other Methods)
- [Contact](#Contact)




# No.1_U-Net_Architecture

## 1.1. Quantization

### 1.1.1. PTQ (Post-Traning-Quantization)

| Title | arXiv | Github| WebSite | Pub. & Date
|:-----:|:-----:|:-----:|:-----:|:-----:|
|[Panda-70M: Captioning 70M Videos with Multiple Cross-Modality Teachers](https://arxiv.org/abs/2402.19479) |  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.19479)|   [![Star](https://img.shields.io/github/stars/snap-research/Panda-70M.svg?style=social&label=Star)](https://github.com/snap-research/Panda-70M)| [![Website](https://img.shields.io/badge/Website-9cf)](https://snap-research.github.io/Panda-70M/)  | CVPR, 2024 |
|[CelebV-Text: A Large-Scale Facial Text-Video Dataset](https://arxiv.org/pdf/2303.14717.pdf)|  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2303.14717.pdf) |   [![Star](https://img.shields.io/github/stars/CelebV-Text/CelebV-Text.svg?style=social&label=Star)](https://github.com/CelebV-Text/CelebV-Text) | - | CVPR, 2023
|[InternVid: A Large-scale Video-Text Dataset for Multimodal Understanding and Generation](https://arxiv.org/abs/2307.06942) |  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2307.06942)|   [![Star](https://img.shields.io/github/stars/OpenGVLab/InternVideo.svg?style=social&label=Star)](https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid)| - | May, 2023 |
|[VideoFactory: Swap Attention in Spatiotemporal Diffusions for Text-to-Video Generation](https://arxiv.org/abs/2305.10874)| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.10874) | - | - | May, 2023|
|[Advancing High-Resolution Video-Language Representation with Large-Scale Video Transcriptions](https://arxiv.org/abs/2111.10337)  |  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2111.10337)|- |- |Nov, 2021 |
| [Frozen in Time: A Joint Video and Image Encoder for End-to-End Retrieval](https://arxiv.org/abs/2104.00650) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2104.00650)  | - | - |ICCV, 2021 |
|[MSR-VTT: A Large Video Description Dataset for Bridging Video and Language](https://openaccess.thecvf.com/content_cvpr_2016/html/Xu_MSR-VTT_A_Large_CVPR_2016_paper.html) |  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://openaccess.thecvf.com/content_cvpr_2016/html/Xu_MSR-VTT_A_Large_CVPR_2016_paper.html) | -| -| CVPR, 2016|



### 1.1.2. QAT (Quantization-Aware-Traning)

| Title | arXiv | Github| WebSite | Pub. & Date
|:-----:|:-----:|:-----:|:-----:|:-----:|
|[UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild](https://arxiv.org/abs/1212.0402)|  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1212.0402) |   - | - | Dec., 2012
|[First Order Motion Model for Image Animation](https://arxiv.org/abs/2003.00196) |     [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2003.00196) | -|- | May, 2023 |
|[Learning to Generate Time-Lapse Videos Using Multi-Stage Dynamic Generative Adversarial Networks](https://arxiv.org/abs/1709.07592) |   [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1709.07592) | -| -| CVPR,2018|



## 1.2. Pruning & Structure Distillation

### 1.2.1. Pruning 

| Title | arXiv | Github | WebSite | Pub. & Date |
|---|---|---|---|---|
| [MagicTime: Time-lapse Video Generation Models as Metamorphic Simulators](https://arxiv.org/abs/2404.05014) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2404.05014) |[![Star](https://img.shields.io/github/stars/PKU-YuanGroup/MagicTime.svg?style=social&label=Star)](https://github.com/PKU-YuanGroup/MagicTime)|[![Website](https://img.shields.io/badge/Website-9cf)](https://pku-yuangroup.github.io/MagicTime/) | Apr., 2024
| [Mora: Enabling Generalist Video Generation via A Multi-Agent Framework](https://arxiv.org/abs/2403.13248) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2403.13248) |-|- | Mar., 2024
| [VSTAR: Generative Temporal Nursing for Longer Dynamic Video Synthesis](https://arxiv.org/abs/2403.13501) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2403.13501) |-|- | Mar., 2024
| [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.15391) |-|[![Website](https://img.shields.io/badge/Website-9cf)](https://sites.google.com/view/genie-2024/home) | Feb., 2024
| [Snap Video: Scaled Spatiotemporal Transformers for Text-to-Video Synthesis](https://arxiv.org/abs/2402.14797) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.14797) |-|[![Website](https://img.shields.io/badge/Website-9cf)](https://snap-research.github.io/snapvideo/) | Feb., 2024
| [Lumiere: A Space-Time Diffusion Model for Video Generation](https://arxiv.org/abs/2401.12945) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2401.12945) |-|[![Website](https://img.shields.io/badge/Website-9cf)](https://lumiere-video.github.io/) | Jan, 2024
| [UNIVG: TOWARDS UNIFIED-MODAL VIDEO GENERATION](https://arxiv.org/abs/2401.09084) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2401.09084) |-| [![Website](https://img.shields.io/badge/Website-9cf)](https://univg-baidu.github.io/) | Jan, 2024
| [VideoCrafter2: Overcoming Data Limitations for High-Quality Video Diffusion Models](https://arxiv.org/abs/2401.09047) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2401.09047) |[![Star](https://img.shields.io/github/stars/VideoCrafter/VideoCrafter.svg?style=social&label=Star)](https://github.com/VideoCrafter/VideoCrafter)|[![Website](https://img.shields.io/badge/Website-9cf)](https://ailab-cvc.github.io/videocrafter/) | Jan, 2024
| [360DVD: Controllable Panorama Video Generation with 360-Degree Video Diffusion Model](https://arxiv.org/abs/2401.06578) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2401.06578) |-|[![Website](https://img.shields.io/badge/Website-9cf)](https://akaneqwq.github.io/360DVD/) | Jan, 2024
| [MagicVideo-V2: Multi-Stage High-Aesthetic Video Generation](https://arxiv.org/abs/2401.04468) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2401.04468) |-|[![Website](https://img.shields.io/badge/Website-9cf)](https://magicvideov2.github.io/) | Jan, 2024
| [VideoDrafter: Content-Consistent Multi-Scene Video Generation with LLM](https://arxiv.org/abs/2401.01256) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2401.01256) |-|[![Website](https://img.shields.io/badge/Website-9cf)](https://videodrafter.github.io/) | Jan, 2024
| [A Recipe for Scaling up Text-to-Video Generation with Text-free Videos](https://arxiv.org/abs/2312.15770) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.15770) |[![Star](https://img.shields.io/github/stars/damo-vilab/i2vgen-xl.svg?style=social&label=Star)](https://github.com/damo-vilab/i2vgen-xl)|[![Website](https://img.shields.io/badge/Website-9cf)](https://tf-t2v.github.io/) | Dec, 2023
| [InstructVideo: Instructing Video Diffusion Models with Human Feedback](https://arxiv.org/abs/2312.12490) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.12490) |[![Star](https://img.shields.io/github/stars/damo-vilab/i2vgen-xl.svg?style=social&label=Star)](https://github.com/damo-vilab/i2vgen-xl)|[![Website](https://img.shields.io/badge/Website-9cf)](https://instructvideo.github.io/) | Dec, 2023
| [VideoLCM: Video Latent Consistency Model](https://arxiv.org/abs/2312.09109) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.09109) |-|- | Dec, 2023
| [Photorealistic Video Generation with Diffusion Models](https://arxiv.org/abs/2312.06662) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.06662) |-|[![Website](https://img.shields.io/badge/Website-9cf)](https://walt-video-diffusion.github.io/) | Dec, 2023
| [Hierarchical Spatio-temporal Decoupling for Text-to-Video Generation](https://arxiv.org/abs/2312.04483) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.04483) |[![Star](https://img.shields.io/github/stars/damo-vilab/i2vgen-xl.svg?style=social&label=Star)](https://github.com/damo-vilab/i2vgen-xl)|[![Website](https://img.shields.io/badge/Website-9cf)](https://higen-t2v.github.io/) | Dec, 2023
| [Delving Deep into Diffusion Transformers for Image and Video Generation](https://arxiv.org/abs/2312.04557) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.04557) |-|[![Website](https://img.shields.io/badge/Website-9cf)](https://www.shoufachen.com/gentron_website/) | Dec, 2023



### 1.2.2. Structure Distillation

| Title | arXiv | Github | WebSite | Pub. & Date |
|---|---|---|---|---|
|[VideoElevator: Elevating Video Generation Quality with Versatile Text-to-Image Diffusion Models](https://arxiv.org/abs/2403.05438) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2403.05438) | [![Star](https://img.shields.io/github/stars/YBYBZhang/VideoElevator.svg?style=social&label=Star)](https://github.com/YBYBZhang/VideoElevator)  | [![Website](https://img.shields.io/badge/Website-9cf)](https://videoelevator.github.io/) | Mar, 2024 |
|[TRAILBLAZER: TRAJECTORY CONTROL FOR DIFFUSION-BASED VIDEO GENERATION](https://arxiv.org/abs/2401.00896) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2401.00896) | [![Star](https://img.shields.io/github/stars/hohonu-vicml/Trailblazer.svg?style=social&label=Star)](https://github.com/hohonu-vicml/Trailblazer)  | [![Website](https://img.shields.io/badge/Website-9cf)](https://hohonu-vicml.github.io/Trailblazer.Page/) | Jan, 2024 |
|[FreeInit: Bridging Initialization Gap in Video Diffusion Models](https://arxiv.org/abs/2312.07537) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.07537) | [![Star](https://img.shields.io/github/stars/TianxingWu/FreeInit.svg?style=social&label=Star)](https://github.com/TianxingWu/FreeInit)  | [![Website](https://img.shields.io/badge/Website-9cf)](https://tianxingwu.github.io/pages/FreeInit/) | Dec, 2023 |
|[MTVG : Multi-text Video Generation with Text-to-Video Models](https://arxiv.org/abs/2312.04086) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.04086) | -  | [![Website](https://img.shields.io/badge/Website-9cf)](https://kuai-lab.github.io/mtvg-page/) | Dec, 2023 |
|[F3-Pruning: A Training-Free and Generalized Pruning Strategy towards Faster and Finer Text-to-Video Synthesis](https://arxiv.org/abs/2312.03459) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.03459) | -  | - | Nov, 2023 |
|[AdaDiff: Adaptive Step Selection for Fast Diffusion](https://arxiv.org/abs/2311.14768) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2311.14768) | -  | - | Nov, 2023 |
|[FlowZero: Zero-Shot Text-to-Video Synthesis with LLM-Driven Dynamic Scene Syntax](https://arxiv.org/abs/2311.15813) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2311.15813) | [![Star](https://img.shields.io/github/stars/aniki-ly/FlowZero.svg?style=social&label=Star)](https://github.com/aniki-ly/FlowZero)  | [![Website](https://img.shields.io/badge/Website-9cf)](https://flowzero-video.github.io/) | Nov, 2023 |
|[🏀GPT4Motion: Scripting Physical Motions in Text-to-Video Generation via Blender-Oriented GPT Planning](https://arxiv.org/abs/2311.12631) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2311.12631) | [![Star](https://img.shields.io/github/stars/jiaxilv/GPT4Motion.svg?style=social&label=Star)](https://github.com/jiaxilv/GPT4Motion) | [![Website](https://img.shields.io/badge/Website-9cf)](https://gpt4motion.github.io/) | Nov, 2023 |
|[FreeNoise: Tuning-Free Longer Video Diffusion Via Noise Rescheduling](https://arxiv.org/abs/2310.15169) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.15169) | [![Star](https://img.shields.io/github/stars/arthur-qiu/LongerCrafter.svg?style=social&label=Star)](https://github.com/arthur-qiu/LongerCrafter)  | [![Website](https://img.shields.io/badge/Website-9cf)](http://haonanqiu.com/projects/FreeNoise.html) | Oct, 2023 |
|[ConditionVideo: Training-Free Condition-Guided Text-to-Video Generation](https://arxiv.org/abs/2310.07697) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.07697) | [![Star](https://img.shields.io/github/stars/pengbo807/ConditionVideo.svg?style=social&label=Star)](https://github.com/pengbo807/ConditionVideo)  | [![Website](https://img.shields.io/badge/Website-9cf)](https://pengbo807.github.io/conditionvideo-website/) | Oct, 2023 |
|[LLM-grounded Video Diffusion Models](https://arxiv.org/abs/2309.17444) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2309.17444) | [![Star](https://img.shields.io/github/stars/TonyLianLong/LLM-groundedVideoDiffusion.svg?style=social&label=Star)](https://github.com/TonyLianLong/LLM-groundedVideoDiffusion) | [![Website](https://img.shields.io/badge/Website-9cf)](https://llm-grounded-video-diffusion.github.io/) | Oct, 2023 |
|[Free-Bloom: Zero-Shot Text-to-Video Generator with LLM Director and LDM Animator](https://arxiv.org/abs/2309.14494) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2309.14494) | [![Star](https://img.shields.io/github/stars/SooLab/Free-Bloom.svg?style=social&label=Star)](https://github.com/SooLab/Free-Bloom) | - | NeurIPS, 2023 |
|[DiffSynth: Latent In-Iteration Deflickering for Realistic Video Synthesis](https://arxiv.org/abs/2308.03463) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2308.03463) | [![Star](https://img.shields.io/github/stars/alibaba/EasyNLP/tree/master/diffusion/DiffSynth.svg?style=social&label=Star)](https://github.com/alibaba/EasyNLP/tree/master/diffusion/DiffSynth) | [![Website](https://img.shields.io/badge/Website-9cf)](https://anonymous456852.github.io/) | Aug, 2023 |
|[Large Language Models are Frame-level Directors for Zero-shot Text-to-Video Generation](https://arxiv.org/abs/2305.14330) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.14330) | [![Star](https://img.shields.io/github/stars/KU-CVLAB/DirecT2V.svg?style=social&label=Star)](https://github.com/KU-CVLAB/DirecT2V) | - | May, 2023 |
|[Text2video-Zero: Text-to-Image Diffusion Models Are Zero-Shot Video Generators](https://arxiv.org/abs/2303.13439) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.13439) | [![Star](https://img.shields.io/github/stars/Picsart-AI-Research/Text2Video-Zero.svg?style=social&label=Star)](https://github.com/Picsart-AI-Research/Text2Video-Zero) | [![Website](https://img.shields.io/badge/Website-9cf)](https://text2video-zero.github.io/) | Mar., 2023 |
| [PEEKABOO: Interactive Video Generation via Masked-Diffusion](https://arxiv.org/abs/2312.07509) 🫣 |  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.07509) | [![Star](https://img.shields.io/github/stars/microsoft/Peekaboo.svg?style=social&label=Star)](https://github.com/microsoft/Peekaboo) | [![Website](https://img.shields.io/badge/Website-9cf)](https://jinga-lala.github.io/projects/Peekaboo/) | CVPR, 2024 |



## 1.3. Feature Cache & Reuse

| Title | arXiv | Github | WebSite | Pub. & Date |
|---|---|---|---|---|
|[Champ: Controllable and Consistent Human Image Animation with 3D Parametric Guidance](https://arxiv.org/abs/2403.14781) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2403.14781) | [![Star](https://img.shields.io/github/stars/fudan-generative-vision/champ.svg?style=social&label=Star)](https://github.com/fudan-generative-vision/champ) | [![Website](https://img.shields.io/badge/Website-9cf)](https://fudan-generative-vision.github.io/champ/) | Mar., 2024 |
|[Action Reimagined: Text-to-Pose Video Editing for Dynamic Human Actions](https://arxiv.org/abs/2403.07198) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2403.07198) | - | - | Mar., 2024 |
|[Do You Guys Want to Dance: Zero-Shot Compositional Human Dance Generation with Multiple Persons](https://arxiv.org/abs/2401.13363) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2401.13363) | - | - | Jan., 2024 |
|[DreaMoving: A Human Dance Video Generation Framework based on Diffusion Models](https://arxiv.org/abs/2312.05107) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.05107) | - | [![Website](https://img.shields.io/badge/Website-9cf)](https://dreamoving.github.io/dreamoving/) | Dec., 2023 |
|[MagicAnimate: Temporally Consistent Human Image Animation using Diffusion Model](https://arxiv.org/abs/2311.16498) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2311.16498) | [![Star](https://img.shields.io/github/stars/magic-research/magic-animate.svg?style=social&label=Star)](https://github.com/magic-research/magic-animate) | [![Website](https://img.shields.io/badge/Website-9cf)](https://showlab.github.io/magicanimate/) | Nov., 2023 |
|[Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation](https://arxiv.org/abs/2311.17117) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2311.17117) | [![Star](https://img.shields.io/github/stars/HumanAIGC/AnimateAnyone.svg?style=social&label=Star)](https://github.com/HumanAIGC/AnimateAnyone) | [![Website](https://img.shields.io/badge/Website-9cf)](https://humanaigc.github.io/animate-anyone/) | Nov., 2023 |
|[MagicDance: Realistic Human Dance Video Generation with Motions & Facial Expressions Transfer](https://arxiv.org/abs/2311.12052) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2311.12052) | [![Star](https://img.shields.io/github/stars/Boese0601/MagicDance.svg?style=social&label=Star)](https://github.com/Boese0601/MagicDance) | [![Website](https://img.shields.io/badge/Website-9cf)](https://boese0601.github.io/magicdance/) | Nov., 2023 |
|[DisCo: Disentangled Control for Referring Human Dance Generation in Real World](https://arxiv.org/abs/2307.00040) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2307.00040) | [![Star](https://img.shields.io/github/stars/Wangt-CN/DisCo.svg?style=social&label=Star)](https://github.com/Wangt-CN/DisCo) | [![Website](https://img.shields.io/badge/Website-9cf)](https://disco-dance.github.io/) | Jul., 2023 |
|[Dancing Avatar: Pose and Text-Guided Human Motion Videos Synthesis with Image Diffusion Model](https://arxiv.org/pdf/2308.07749.pdf) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2308.07749.pdf) | - | - | Aug., 2023 |
|[DreamPose: Fashion Image-to-Video Synthesis via Stable Diffusion](https://arxiv.org/abs/2304.06025) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.06025) | [![Star](https://img.shields.io/github/stars/johannakarras/DreamPose.svg?style=social&label=Star)](https://github.com/johannakarras/DreamPose) | [![Website](https://img.shields.io/badge/Website-9cf)](https://grail.cs.washington.edu/projects/dreampose/) | Apr., 2023 |
|[Follow Your Pose: Pose-Guided Text-to-Video Generation using Pose-Free Videos](https://arxiv.org/abs/2304.01186) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.01186) | [![Star](https://img.shields.io/github/stars/mayuelala/FollowYourPose.svg?style=social&label=Star)](https://github.com/mayuelala/FollowYourPose) | [![Website](https://img.shields.io/badge/Website-9cf)](https://follow-your-pose.github.io/) | Apr., 2023 |



## 1.4. Timestep Distallation

### 1.4.1. ODE Preserving Distallation

| Title | arXiv | Github | WebSite | Pub. & Date |
|---|---|---|---|---|
|[UniCtrl: Improving the Spatiotemporal Consistency of Text-to-Video Diffusion Models via Training-Free Unified Attention Control](https://arxiv.org/abs/2403.02332) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2403.02332) | - | - | Mar., 2024 |
|[Magic-Me: Identity-Specific Video Customized Diffusion](https://arxiv.org/abs/2402.09368) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.09368) | - | [![Website](https://img.shields.io/badge/Website-9cf)](https://github.com/Zhen-Dong/Magic-Me) | Feb., 2024 |
|[InteractiveVideo: User-Centric Controllable Video Generation with Synergistic Multimodal Instructions](https://arxiv.org/abs/2402.03040) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.03040) | - | [![Website](https://img.shields.io/badge/Website-9cf)](https://invictus717.github.io/InteractiveVideo/) | Feb., 2024 |
|[Direct-a-Video: Customized Video Generation with User-Directed Camera Movement and Object Motion](https://arxiv.org/abs/2402.03162) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.03162) | - | [![Website](https://img.shields.io/badge/Website-9cf)](https://direct-a-video.github.io/) | Feb., 2024 |
|[Boximator: Generating Rich and Controllable Motions for Video Synthesis](https://arxiv.org/abs/2402.01566) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.01566) | - | [![Website](https://img.shields.io/badge/Website-9cf)](https://boximator.github.io/) | Feb., 2024 |
|[AnimateLCM: Accelerating the Animation of Personalized Diffusion Models and Adapters with Decoupled Consistency Learning](https://arxiv.org/abs/2402.00769) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.00769) | - | -| Jan., 2024 |


### 1.4.2. ODE Reconstructing Distallation

| Title | arXiv | Github | WebSite | Pub. & Date |
|---|---|---|---|---|
|[UniCtrl: Improving the Spatiotemporal Consistency of Text-to-Video Diffusion Models via Training-Free Unified Attention Control](https://arxiv.org/abs/2403.02332) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2403.02332) | - | - | Mar., 2024 |
|[Magic-Me: Identity-Specific Video Customized Diffusion](https://arxiv.org/abs/2402.09368) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.09368) | - | [![Website](https://img.shields.io/badge/Website-9cf)](https://github.com/Zhen-Dong/Magic-Me) | Feb., 2024 |
|[InteractiveVideo: User-Centric Controllable Video Generation with Synergistic Multimodal Instructions](https://arxiv.org/abs/2402.03040) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.03040) | - | [![Website](https://img.shields.io/badge/Website-9cf)](https://invictus717.github.io/InteractiveVideo/) | Feb., 2024 |
|[Direct-a-Video: Customized Video Generation with User-Directed Camera Movement and Object Motion](https://arxiv.org/abs/2402.03162) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.03162) | - | [![Website](https://img.shields.io/badge/Website-9cf)](https://direct-a-video.github.io/) | Feb., 2024 |
|[Boximator: Generating Rich and Controllable Motions for Video Synthesis](https://arxiv.org/abs/2402.01566) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.01566) | - | [![Website](https://img.shields.io/badge/Website-9cf)](https://boximator.github.io/) | Feb., 2024 |
|[AnimateLCM: Accelerating the Animation of Personalized Diffusion Models and Adapters with Decoupled Consistency Learning](https://arxiv.org/abs/2402.00769) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.00769) | - | -| Jan., 2024 |



## 1.5. Fast Sampling Solver

| Title | arXiv | Github | WebSite| Pub. & Date |
|---|---|---|---|---|
|[Hybrid Video Diffusion Models with 2D Triplane and 3D Wavelet Representation](https://arxiv.org/abs/2402.13729) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.13729) | [![Star](https://img.shields.io/github/stars/hxngiee/HVDM.svg?style=social&label=Star)](https://github.com/hxngiee/HVDM)     | [![Website](https://img.shields.io/badge/Website-9cf)](https://hxngiee.github.io/HVDM/) | Feb. 2024 |
|[Video Probabilistic Diffusion Models in Projected Latent Space](https://arxiv.org/abs/2302.07685) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2302.07685) | [![Star](https://img.shields.io/github/stars/sihyun-yu/PVDM.svg?style=social&label=Star)](https://github.com/sihyun-yu/PVDM) | [![Website](https://img.shields.io/badge/Website-9cf)](https://sihyun.me/PVDM/) | CVPR 2023 |
|[VIDM: Video Implicit Diffusion Models](https://arxiv.org/abs/2212.00235) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.00235) | [![Star](https://img.shields.io/github/stars/MKFMIKU/VIDM.svg?style=social&label=Star)](https://github.com/MKFMIKU/VIDM)     | [![Website](https://img.shields.io/badge/Website-9cf)](https://kfmei.page/vidm/) | AAAI 2023 |
|[GD-VDM: Generated Depth for better Diffusion-based Video Generation](https://arxiv.org/abs/2306.11173) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.11173) | [![Star](https://img.shields.io/github/stars/lapid92/GD-VDM.svg?style=social&label=Star)](https://github.com/lapid92/GD-VDM) | - | Jun., 2023 |
|[LEO: Generative Latent Image Animator for Human Video Synthesis](https://arxiv.org/pdf/2305.03989.pdf) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2305.03989.pdf) | [![Star](https://img.shields.io/github/stars/wyhsirius/LEO.svg?style=social&label=Star)](https://github.com/wyhsirius/LEO)   | [![Website](https://img.shields.io/badge/Website-9cf)](https://wyhsirius.github.io/LEO-project/) | May., 2023 |



## 1.6. GAN-Based Method

| Title | arXiv | Github | WebSite| Pub. & Date |
|---|---|---|---|---|


## 1.7. Efficient Training

| Title | arXiv | Github | WebSite| Pub. & Date |
|---|---|---|---|---|


## 1.8. New Architecture Design

| Title | arXiv | Github | WebSite| Pub. & Date |
|---|---|---|---|---|


## 1.9. Other Methods

| Title | arXiv | Github | WebSite| Pub. & Date |
|---|---|---|---|---|




# 2. Transformer Architecture

## 2.1. Quantization

### 2.1.1. PTQ (Post-Traning-Quantization)

| Title | arXiv | Github| WebSite | Pub. & Date
|:-----:|:-----:|:-----:|:-----:|:-----:|


### 2.1.2. QAT (Quantization-Aware-Traning)

| Title | arXiv | Github| WebSite | Pub. & Date
|:-----:|:-----:|:-----:|:-----:|:-----:|


## 2.2. Pruning & Structure Distillation

### 2.2.1. Pruning 

| Title | arXiv | Github | WebSite | Pub. & Date |
|---|---|---|---|---|


### 2.2.1. Structure Distillation

| Title | arXiv | Github | WebSite | Pub. & Date |
|---|---|---|---|---|


## 2.3. Feature Cache & Reuse

| Title | arXiv | Github | WebSite | Pub. & Date |
|---|---|---|---|---|


## 2.4. Timestep Distallation

| Title | arXiv | Github | WebSite | Pub. & Date |
|---|---|---|---|---|


## 2.5. New Architecture Design

| Title | arXiv | Github | WebSite| Pub. & Date |
|---|---|---|---|---|


## 2.6. Other Methods

| Title | arXiv | Github | WebSite| Pub. & Date |
|---|---|---|---|---|



# Contact

If you have any suggestions or find our work helpful, feel free to contact us.

Homepage: University of Science and Technology of China (USTC), [Intelligent Media Computing Lab](https://faculty.ustc.edu.cn/chenzhibo).

Email: ???.com

If you find our survey is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.
