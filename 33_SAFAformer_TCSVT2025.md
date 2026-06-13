# SAFAformer: Scale-Aware Frequency-Adaptive Guidance for Nighttime Flare Removal

> **发表**：IEEE TCSVT 2025（DOI: 10.1109/TCSVT.2025.3595933，2025 年 8 月 5 日在线发表，正式刊期为 2026 年第 36 卷第 1 期，pp. 93–105）　|　**作者**：Wei Dong（福州大学，第一作者）, Guodong Fan（山东工商学院，通讯）, Fan Zhang（浪潮·无锡）, Min Gan（青岛大学）, Guang-Yong Chen（福州大学）, C. L. Philip Chen（华南理工大学）——单位信息来自 Unpaywall/Crossref 著录元数据
> **论文**：https://ieeexplore.ieee.org/document/11113259　|　**代码**：摘要声称开源于 GitHub，但截至撰稿未检索到公开仓库

> 说明：原文 PDF 不可公开获取（IEEE 付费墙），本报告基于 Semantic Scholar 收录的完整官方摘要、Crossref 元数据与公开检索信息撰写。**整体动机与模块名称可靠（来自摘要原文），模块内部结构与实验数字的细节置信度有限**，不确定处已注明。未找到可用的论文配图，本报告不插图。

## 一、解决的问题

夜间去 flare 的根本困难是真实配对数据几乎不可得，主流方法全部依赖 Flare7K/Flare7K++ 类合成管线造训练对。本文指出这条合成路线一个被长期忽视的结构性缺陷：**现有合成管线几乎只生成单 flare 场景，而真实夜景里多光源、多 flare 交叠以及复合型 flare（散射+反射+泛光混合）是常态**。城市夜景中路灯、车灯、霓虹灯同时存在，flare 彼此叠加形成的退化模式远比"一张背景 + 一个 flare 素材"复杂，且这种复合场景在素材层面就很难仿真。训练分布与真实分布的这种系统性错配，直接导致在合成集上刷高分的模型一到真实多 flare 场景就失灵。

本文没有走"造更真的多 flare 数据"这条重路线，而是从频域统计中找到了一条绕开数据瓶颈的途径。作者通过分析发现一个**跨域不变量**：无论是合成单 flare、真实单 flare 还是真实多 flare 场景，flare 退化信息在各频率子带上呈现**相似的分布规律——主要集中于低频区域，少量存在于高频区域；并且眩光（glare）效应越严重，能量越向低频集中**。既然退化的频域分布在合成域与真实域之间是对齐的，那么让网络在频域上针对性建模退化，就能把在合成数据上学到的去 flare 能力迁移到真实复杂场景，从原理上弥合 synthetic-to-real 的泛化鸿沟。

## 二、核心创新点

1. **频域跨域不变性的发现与论证**：首次（就该文表述而言）系统分析并指出 flare 退化在频率子带上的分布规律对"合成单 flare / 真实单 flare / 真实多 flare"三种场景具有一致性，且眩光强度与低频集中度正相关。这把"如何泛化到真实多 flare"从数据问题转化为表示问题，是全文的理论支点。
2. **频率自适应引导模块（FAGM，Frequency-Adaptive Guidance Module)**：利用上述频域先验，在训练中对不同频率子带施加自适应引导，使网络聚焦于 flare 能量集中的低频成分（内部机制推断：按子带自适应加权/调制特征或损失，置信度低，摘要未给细节）。
3. **尺度感知 Transformer 块（SATB，Scale-Aware Transformer Block）**：在 Transformer 骨干中引入多尺度感知设计，呼应真实场景中 flare 大小、形状、覆盖范围差异巨大（多光源远近不一）的特点（内部机制推断：多尺度窗口/多分支注意力，置信度低）。
4. 整体构成 **SAFAformer**：以 FAGM 提供频域引导、SATB 提供尺度自适应表征，在不改动合成数据管线的前提下提升对真实复杂 flare 场景的泛化，实验称达到 SOTA。

## 三、模型结构与方案

基于摘要与同类工作惯例的整体推断（置信度中等偏低）：

- **整体 pipeline**:输入夜间 flare 图像，经由以 SATB 为基本块的 Transformer 编解码网络（该团队及 TCSVT 同类工作多用 U 形结构，推断如此）逐尺度处理；FAGM 在训练过程中将特征/监督信号分解到频率子带（小波或 Fourier 子带，具体变换不可考），按"低频为主、高频为辅"的退化分布对各子带施加自适应引导，使网络的修复能力对准 flare 能量所在的频段。
- **尺度感知**：SATB 命名中的 scale-aware 与摘要强调的多 flare/复合 flare 场景对应——不同距离/亮度的光源产生的 flare 空间尺度差异极大，单一感受野的窗口注意力难以同时覆盖，推断 SATB 通过多尺度窗口、金字塔注意力或动态感受野实现尺度自适应（未证实）。
- **训练数据与损失**：训练数据推断为 Flare7K++ 标准合成管线（摘要批评的正是这类管线的单 flare 局限，方法是在其上做频域引导而非替换数据），损失函数细节不可考。
- **代码**：摘要称"code and pre-trained models are available on GitHub"，但在 GitHub 检索"SAFAformer"及作者名下仓库均未找到，可能尚未公开或仓库名不含方法名；社区维护的 Awesome-Flare-Removal 文献清单中该文条目同样未挂任何代码链接。
- **团队脉络**：本文是该团队夜间去 flare 系列的第二篇 TCSVT——前作 LPFSformer（Location Prior Guided Frequency and Spatial Interactive Learning, TCSVT 2025, 35(4): 3706–3718, Chen GY, Dong W, Fan G 等）已经走"频域+空间交互"路线，本文将引导信号从"光源位置先验"换成"频率子带自适应"，并把骨干换为尺度感知 Transformer。

原文结构图在付费墙内，无可用配图。

## 四、实验与结果

原文实验部分不可公开获取。可确认的信息：

- 摘要声称"extensive experiments demonstrate that SAFAformer achieves state-of-the-art performance in flare removal compared to existing methods"，评测基准推断为 Flare7K++ 真实测试集及真实多 flare 场景定性比较（未证实）；
- 具体 PSNR/SSIM/LPIPS 数字、对比方法清单、FAGM/SATB 消融结果均无法核实，**本报告不引用任何无法溯源的数字**；
- Semantic Scholar 显示该文已有 2 次引用（2026 年 6 月查询）：FlareDiffusion（JVCIR 2026）与 DeflareMambav2（arXiv 2026）。值得注意的是，DeflareMambav2 在相关工作中以一句"Dong et al. introduced SAFAformer, integrating frequency-adaptive guidance with scale-aware Transformer modeling"提及本文，但其定量对比表（FlareX、Flare7K-real 基准）并未纳入 SAFAformer——侧面印证其代码/权重在社区中不可得。

## 五、专家锐评

**价值**：这篇论文最值钱的不是网络，而是那个**经验发现**——flare 退化的频谱分布在合成域与真实域、单 flare 与多 flare 之间近似不变。如果该统计观察经得起独立复核，它为整个"合成训练→真实部署"的夜间去 flare 范式提供了一个可操作的域对齐锚点，比堆数据或堆 GAN 域适应都优雅，也解释了为什么近两年频域方法（MFDNet、FF-Former、WGSF-Net 等）在真实集上普遍比纯空间域方法稳。把多 flare 泛化问题从数据侧搬到表示侧，是问题表述层面的真贡献。

**不足与质疑**：

1. **核心发现与既有共识的边界模糊**。"flare/光照能量集中在低频"在 MFDNet、频率解耦系列等多篇前作中已是公开先验；本文的增量主张在于"该分布跨单/多 flare、跨合成/真实一致"。但多 flare 场景的频谱不变性某种程度上是傅里叶变换线性性的直接推论（多个低频为主的退化叠加仍以低频为主），把它包装成需要"detailed analysis"才能"uncover"的发现，有夸大叙事之嫌；真正非平凡的部分（复合 flare 中高频 streak 成分的占比变化）摘要反而未量化。
2. **频域引导对高频 flare 成分的覆盖存疑**。光条纹（streak）、星芒是显著的高频定向伪影，恰恰是多光源夜景里最碍眼的成分；"主要盯低频"的 FAGM 如何避免在这类伪影上欠修复，是方法逻辑上必须回答而摘要只字未提的问题。
3. **声称开源但查无此仓**。摘要白纸黑字写了"available on GitHub for validation"，截至撰稿（2026 年 6 月）在 GitHub 上检索不到任何 SAFAformer 仓库，社区维护的 Awesome-Flare-Removal 清单中该条目也无代码链接；更扎眼的是，引用它的 DeflareMambav2 在相关工作里点了名，做定量对比时却把它排除在表外。对一篇把"可验证"写进摘要的论文，这直接削弱可信度，也让 SOTA 声明暂时只能记为"作者自述"。
4. **泛化主张缺乏配对证据的先天困境**：真实多 flare 场景没有 GT，对"弥合 synthetic-to-real 鸿沟"的验证大概率只能依赖无参考指标或视觉对比/用户研究；用单 flare 配对测试集的 PSNR 提升来背书多 flare 泛化，在逻辑上是错位的。论文是否构建了多 flare 评测协议无从知晓，但这是该主张成立与否的关键。
5. **同团队前作叠影下的边际新颖性**。同一批作者一年内在 TCSVT 已发表 LPFSformer（35(4): 3706–3718），同样是"夜间去 flare + 频域/空间建模 + Transformer"的配方，本文只是把引导信号从光源位置先验换成频率子带自适应、骨干换成尺度感知块；该团队在 FDCE-Net（水下增强）、低照度遥感增强等多篇论文中也反复使用"频域引导模块 + X-Aware 块"的组织模板。FAGM/SATB 相对这一族自家模块的实质差异有多大，需要拿到全文逐一比对才能下结论——审稿人理应要求论文与 LPFSformer 做正面对比与消融，而摘要层面看不到这种自我切割。
