# 镜头眩光去除（Flare Removal）论文分析报告合集

> 来源：[cranbs/Awesome-Flare-Removal](https://github.com/cranbs/Awesome-Flare-Removal) 仓库收录的全部论文。
> 每篇报告包含：解决的问题、核心创新点、模型结构与方案（附原论文配图）、实验结果、专家锐评（价值与不足）。
> 付费墙论文（无公开 PDF）的报告基于摘要、官方代码仓库与公开资料撰写，已在文中注明。

共 **47** 篇报告。

| # | 报告 | 发表 | 配图数 |
|---|------|------|--------|
| 01 | [SLCFormer: Spectral-Local Context Transformer with Physics-Grounded Flare Synthesis for Nighttime Flare Removal](01_SLCFormer_AAAI2026.md) | AAAI 2026 | 4 |
| 02 | [Nighttime Flare Removal via Wavelet-Guided and Gated-Enhanced Spatial-Frequency Fusion Network（WGSF-Net）](02_WGSF-Net_AAAI2026.md) | AAAI 2026 | 3 |
| 03 | [CAST-LUT: Tokenizer-Guided HSV Look-Up Tables for Purple Flare Removal](03_CAST-LUT_AAAI2026.md) | AAAI 2026 | 4 |
| 04 | [FlareX: A Physics-Informed Dataset for Lens Flare Removal via 2D Synthesis and 3D Rendering](04_FlareX_NeurIPS2025.md) | NeurIPS 2025 Datasets & Benchmarks Track | 4 |
| 05 | [DeflareMamba: Hierarchical Vision Mamba for Contextually Consistent Lens Flare Removal](05_DeflareMamba_ACMMM2025.md) | ACM MM 2025 | 4 |
| 06 | [Removing Out-of-Focus Reflective Flares via Color Alignment](06_ColorAlignment_ICCV2025.md) | ICCV 2025 | 4 |
| 07 | [LightsOut: Diffusion-based Outpainting for Enhanced Lens Flare Removal](07_LightsOut_ICCV2025.md) | ICCV 2025 | 3 |
| 08 | [PBFG: A New Physically-Based Dataset and Removal of Lens Flares and Glares](08_PBFG_ICCV2025.md) | ICCV 2025 | 4 |
| 09 | [FlareGS: 4D Flare Removal using Gaussian Splatting for Urban Scenes](09_FlareGS_ICCVW2025.md) | ICCV Workshop 2025 | 3 |
| 10 | [Disentangle Nighttime Lens Flares: Self-supervised Generation-based Lens Flare Removal](10_DisentangleFlares_AAAI2025.md) | AAAI 2025 | 4 |
| 11 | [MIPI 2023 Challenge on Nighttime Flare Removal: Methods and Results](11_MIPI2023_CVPRW2023.md) | CVPR Workshop (MIPI) 2023 | 3 |
| 12 | [Toward Flare-Free Images: A Survey](12_FlareFreeSurvey_arXiv2023.md) | arXiv 2023（arXiv:2310.14354） | 4 |
| 13 | [Toward Real Flare Removal: A Comprehensive Pipeline and A New Benchmark](13_TowardRealFlareRemoval_arXiv2023.md) | arXiv 2023（arXiv:2306.15884） | 4 |
| 14 | [Flare7K++: Mixing Synthetic and Real Datasets for Nighttime Flare Removal and Beyond](14_Flare7Kpp_TPAMI2024.md) | TPAMI 2024 | 4 |
| 15 | [Flare7K: A Phenomenological Nighttime Flare Removal Dataset](15_Flare7K_NeurIPS2022.md) | NeurIPS 2022 Datasets & Benchmarks Track | 4 |
| 16 | [Tackling Scattering and Reflective Flare in Mobile Camera Systems: A Raw Image Dataset for Enhanced Flare Removal](16_RawFlareDataset_arXiv2023.md) | arXiv 2023（v2 于 2024-11 更新，题为 "Understanding and Tackling Scattering and Reflective Flare for Mobile Camera Systems"） | 4 |
| 17 | [Light Source Guided Single-Image Flare Removal From Unpaired Data](17_LightSourceGuided_ICCV2021.md) | ICCV 2021 | 4 |
| 19 | [Harmonizing Light and Darkness: A Symphony of Prior-guided Data Synthesis and Adaptive Focus for Nighttime Flare Removal](19_HarmonizingLightDarkness_arXiv2024.md) | arXiv 2024（arXiv:2404.00313） | 4 |
| 20 | [Difflare: Removing Image Lens Flare with Latent Diffusion Model](20_Difflare_BMVC2024.md) | BMVC 2024 | 3 |
| 21 | [GN-FR: Generalizable Neural Radiance Fields for Flare Removal](21_GN-FR_BMVC2024.md) | BMVC 2024 | 3 |
| 22 | [Improving Lens Flare Removal with General-Purpose Pipeline and Multiple Light Sources Recovery](22_ImprovingLensFlareRemoval_ICCV2023.md) | ICCV 2023 | 3 |
| 23 | [Hard-Negative Sampling with Cascaded Fine-Tuning Network to Boost Flare Removal Performance in the Nighttime Images](23_HardNegativeSampling_CVPRW2023.md) | CVPR Workshop (MIPI) 2023 | 4 |
| 24 | [FF-Former: Swin Fourier Transformer for Nighttime Flare Removal](24_FFFormer_CVPRW2023.md) | CVPR Workshop (MIPI) 2023 | 4 |
| 25 | [Nighttime Smartphone Reflective Flare Removal Using Optical Center Symmetry Prior](25_BracketFlare_CVPR2023.md) | CVPR 2023 | 4 |
| 26 | [How to Train Neural Networks for Flare Removal](26_HowToTrainNN_ICCV2021.md) | ICCV 2021 | 4 |
| 27 | [Learning to See Through Flare](27_SeeThroughFlare_ICCVW2025.md) | ICCV Workshop (Responsible Imaging) 2025 | 4 |
| 28 | [Integrating Spatial and Frequency Information for Under-Display Camera Image Restoration](28_UDC_SpatialFrequency_arXiv2025.md) | arXiv 2025（arXiv:2501.18517，2025-01-30 提交） | 4 |
| 29 | [MIPI 2024 Challenge on Nighttime Flare Removal: Methods and Results](29_MIPI2024_CVPRW2024.md) | CVPR Workshop (MIPI) 2024 | 3 |
| 30 | [Image Lens Flare Removal Using Adversarial Curve Learning](30_AdversarialCurveLearning_TPAMI2025.md) | IEEE TPAMI 2025（vol. 47, no. 9, pp. 7396–7409，2025-05-06 上线，DOI: 10.1109/TPAMI.2025.3567308） | 4 |
| 31 | [Nighttime Flare Removal via Frequency Decoupling](31_FrequencyDecoupling_PRL2026.md) | Pattern Recognition Letters 2026（Vol. 201, pp. 73–79，DOI: 10.1016/j.patrec.2026.01.012） | 3 |
| 32 | [Flare Detection and Detail Compensation for Nighttime Flare Removal](32_FlareDetectionDetailComp_EAAI2025.md) | Engineering Applications of Artificial Intelligence（EAAI），Vol. 163, Article 112926，2026 年 1 月卷（2025 年录用，DOI: 10.1016/j.engappai.2025.112926，参考文献 63 篇） | 0 |
| 33 | [SAFAformer: Scale-Aware Frequency-Adaptive Guidance for Nighttime Flare Removal](33_SAFAformer_TCSVT2025.md) | IEEE TCSVT 2025（DOI: 10.1109/TCSVT.2025.3595933，2025 年 8 月 5 日在线发表，正式刊期为 2026 年第 36 卷第 1 期，pp. 93–105） | 0 |
| 34 | [LUFormer: A Luminance-informed Localized Transformer with Frequency Augmentation for Nighttime Flare Removal](34_LUFormer_NN2025.md) | Neural Networks 2025（Vol. 191, 2025 年 10 月刊；DOI: 10.1016/j.neunet.2025.107660） | 1 |
| 35 | [Nighttime Glare Removal for Consumer Electronics via Latent Space Transformation and Feature-Enhanced Attention Mechanism](35_GlareRemovalTCE2025.md) | IEEE Transactions on Consumer Electronics (TCE) 2025，Vol. 71, No. 2, pp. 6719–6733（DOI: 10.1109/TCE.2025.3570806） | 0 |
| 36 | [IllumiNet: A two-stage model for effective flare removal and light enhancement under complex lighting conditions](36_IllumiNet_ESWA2025.md) | Expert Systems with Applications (ESWA), Vol. 282, Article 127638, 2025（2025-04-19 在线发表） | 4 |
| 37 | [Flare-Aware RWKV for Flare Removal](37_FlareAwareRWKV_ICASSP2025.md) | ICASSP 2025（2025 年 4 月 6–11 日，印度海得拉巴，pp. 1–5） | 0 |
| 38 | [When low-light meets flares: Towards Synchronous Flare Removal and Brightness Enhancement](38_LowLightMeetsFlares_NN2025.md) | Neural Networks 2025（Volume 185, Article 107149） | 0 |
| 39 | [Mask-Q attention network for flare removal](39_MaskQAttention_EAAI2025.md) | Neurocomputing 2025，Volume 637，Article 130100（DOI: 10.1016/j.neucom.2025.130100；publicationDate 2025-03，由 Crossref / Unpaywall / Semantic Scholar 三方一致确认）。 | 0 |
| 40 | [A self-prompt based dual-domain network for nighttime flare removal](40_SelfPromptDualDomain_Neurocomputing2025.md) | Engineering Applications of Artificial Intelligence (EAAI), Vol. 144, 2025（注：见文末"异常说明"——本文的权威检索结果指向 EAAI 而非 Neurocomputing） | 0 |
| 41 | [Self-prior Guided Spatial and Fourier Transformer for Nighttime Flare Removal](41_SGSFT_TASE2025.md) | IEEE Transactions on Automation Science and Engineering (TASE) 2025 | 2 |
| 42 | [LPFSformer: Location Prior Guided Frequency and Spatial Interactive Learning for Nighttime Flare Removal](42_LPFSformer_TCSVT2025.md) | IEEE TCSVT 2025（CCF B） | 0 |
| 43 | [Flare-Free Vision: Empowering Uformer with Depth Insights](43_UformerDepth_ICASSP2024.md) | ICASSP 2024（IEEE International Conference on Acoustics, Speech and Signal Processing，韩国首尔） | 2 |
| 44 | [MFDNet: Multi-Frequency Deflare Network for Efficient Nighttime Flare Removal](44_MFDNet_TVC2024.md) | The Visual Computer 2024（CGI 2024） | 3 |
| 45 | [GR-GAN: A Unified Adversarial Framework for Single Image Glare Removal and Denoising](45_GRGAN_PR2024.md) | Pattern Recognition 2024（Vol. 156, Article 110815，2024 年 12 月卷） | 0 |
| 46 | [Understanding and Tackling Scattering and Reflective Flare for Mobile Camera Systems](46_ScatteringReflectiveFlare_ACMMM2024.md) | ACM MM 2024（pp. 8768–8776） | 4 |
| 47 | [Toward Blind Flare Removal Using Knowledge-Driven Flare-Level Estimator](47_BlindFlareRemoval_TIP2024.md) | IEEE Transactions on Image Processing (TIP) 2024（Vol. 33, pp. 6114–6128） | 0 |
| 48 | [Adversarial Lens Flares: A Threat to Camera-Based Systems in Smart Devices](48_AdversarialLensFlares_IoTJ2025.md) | IEEE Internet of Things Journal (IoTJ) 2025 | 0 |
