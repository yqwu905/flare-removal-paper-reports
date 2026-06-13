# Nighttime Glare Removal for Consumer Electronics via Latent Space Transformation and Feature-Enhanced Attention Mechanism

> **发表**：IEEE Transactions on Consumer Electronics (TCE) 2025，Vol. 71, No. 2, pp. 6719–6733（DOI: 10.1109/TCE.2025.3570806）　|　**作者**：Jiazheng Lian, Jiaming Liu, Ming-e Jing, Xiaoyang Zeng, Zherong Liu, Jun Zhou, Yibo Fan（作者次序据 DBLP `journals/tce/LianLJZLZF25`；通讯/单位以正文 PDF 为准，Yibo Fan、Xiaoyang Zeng、Ming-e Jing 为复旦大学专用集成电路与系统国家重点实验室常见署名作者，推断本文主体单位为复旦大学，置信度中等）
> **论文**：https://ieeexplore.ieee.org/document/11006158　|　**代码**：摘要声称公开于 https://github.com/Leo-student/Flare7K ，但该链接实为 Flare7K 数据集的 fork 仓库，**未发现 FBNet 本身的实现/权重**（见下文核查）

> 说明：原文 PDF 不可公开获取（IEEE 付费墙；Unpaywall 显示 `is_oa: false / oa_status: closed`，无任何开放副本，亦未检索到预印本）。本报告基于 Semantic Scholar 收录的**完整官方摘要**、Crossref/DBLP 元数据与公开检索信息撰写。**方法命名（FBNet）、核心机制描述（mapping-based self-attention）、评测规模（四个数据集、10 款消费级相机的多设备集）均来自摘要原文，可靠；模块内部结构与所有定量指标（PSNR/SSIM/LPIPS）无法溯源，本报告一律不编造、不引用。** 因付费墙且无配套结构图可得，本报告不插图。

## 一、解决的问题

夜间成像中的眩光（glare/flare）来源于强光源经镜头折射、散射与反射在传感器上形成的伪影。本文摘要的物理描述是：光线经过镜头发生弯折（light bending through the lens），在明亮光源与暗背景之间制造出极高的对比度，从而显著劣化成像质量。这一退化在夜间尤为突出——路灯、车灯、霓虹等点光源在暗场景中能量集中，产生条纹、光晕、鬼影等多形态伪影。

本文的切入视角带有鲜明的"消费电子"工程导向，这也契合发表期刊 IEEE Transactions on Consumer Electronics 的定位。与多数夜间去 flare 论文只在 Flare7K/Flare7K++ 合成与真实测试集上刷分不同，本文把问题进一步具体化为**消费级相机在真实使用场景中的鲁棒去眩光**：摘要强调评测覆盖"a multi-device dataset featuring 10 consumer-grade cameras"（一个含 10 款消费级相机的多设备数据集），并同时考察白天与夜间（both daytime and nighttime）两种光照条件。换言之，作者关心的核心痛点不只是"能不能去掉合成 flare"，而是"换一台手机/相机、换一种光照，模型还灵不灵"——即跨设备、跨光照的泛化能力，这是真正落到消费电子产品上时最致命的工程瓶颈。

为应对这一痛点，本文提出 **FBNet（Flare Basis Latent Space Transformation Network，眩光基底潜空间变换网络）**，主张通过潜空间（latent space）的变换与一种"基于映射的自注意力（mapping-based self-attention）"机制来检测并消除 flare 伪影，目标是在保持高效（efficient）的前提下取得稳定的真实场景表现。

## 二、核心创新点

以下条目均严格对应摘要原文用语，未作机制扩写处用"推断"标注，置信度从严：

1. **眩光基底潜空间变换（Flare Basis Latent Space Transformation）**：方法名 FBNet 中的 "Flare Basis" 暗示作者将 flare 退化建模为某种"基底（basis）"的组合，并在潜空间中对其进行变换/分离。这区别于主流"端到端直接回归干净图"的范式，意图是把 flare 成分显式地表示为可分解、可操纵的潜变量结构（具体是字典学习式基底、低秩分解还是可学习 token 基，摘要未给，**置信度低**）。
2. **基于映射的自注意力机制（mapping-based self-attention）**：摘要明确 "FBNet employs a mapping-based self-attention mechanism to detect and eliminate flare artifacts"。即注意力不在原始特征上直接计算，而是先经某种映射（推断为到上述潜空间/基底空间的投影）再做注意力，使注意力聚焦于 flare 相关成分的检测与剔除（**内部 Q/K/V 构造细节不可考**）。
3. **特征增强注意力（Feature-Enhanced Attention）**：见诸标题，与摘要的 mapping-based self-attention 共同构成"潜空间变换 + 增强注意力"的双支撑（标题给名、摘要给机制，二者应为同一注意力体系的两面，**置信度中等**）。
4. **面向消费电子的强泛化评测设计**：在四个数据集、含 10 款消费级相机的多设备集、昼夜双光照上系统评测，把"跨设备/跨光照鲁棒性"作为一等公民来论证——这一评测覆盖面在夜间去 flare 文献里是少见的，可视为方法论层面的贡献而非单纯刷指标。

## 三、模型结构与方案

由于正文与结构图在付费墙内，以下基于摘要与同类工作惯例做有节制的推断（**整体置信度中等偏低，结构细节请以原文为准**）：

- **整体范式**：输入夜间含 flare 图像，FBNet 不直接在像素/原特征域回归干净图，而是先把特征送入一个潜空间（latent space），在该空间内对"眩光基底（flare basis）"进行变换与分离，再借由 mapping-based self-attention 定位并抑制 flare 成分，最后重建干净图。"Flare Basis" 一词强烈暗示存在一组（可学习或解析的）基向量/原型，flare 被表示为这些基的线性/非线性组合，从而把"去 flare"转化为"在基底空间中削减 flare 子空间的能量"。
- **mapping-based self-attention 的作用**：常规自注意力在高分辨率特征上既贵又难以聚焦退化区域；本文先做映射再注意力，推断是为了（a）降低计算量以满足摘要强调的 "efficient"，（b）让注意力在 flare 基底空间里更精准地"检测并消除"伪影。具体映射函数、是否跨尺度、窗口化与否，摘要均未交代。
- **训练数据与合成策略**：摘要未点名训练管线，但参与作者长期使用 Flare7K/Flare7K++ 体系（代码链接即指向 Flare7K fork），**推断训练对仍由 Flare7K++ 类合成管线（背景图 + 散射/反射 flare 素材）生成**（置信度中等）。真正的新意在评测端而非数据合成端：多设备 + 昼夜数据用于检验泛化。
- **损失函数**：不可考。同类工作多用 L1/感知损失/flare 区域加权损失的组合，本文是否针对"基底分离"额外设计正交性/稀疏性约束未知（纯推断，不作为事实）。
- **代码可得性核查**：摘要写明 "The code is publicly available at: https://github.com/Leo-student/Flare7K"。经核查，该仓库 README 实为 *Flare7K: A Phenomenological Nighttime Flare Removal Dataset (NeurIPS 2022)* 的官方实现 fork（Uformer 骨干、Flare7K/Flare7K++ 数据），**并非 FBNet 的代码**；该账号下 19 个仓库均为低层视觉相关项目的 fork，**无任何 FBNet 专属仓库或权重**。因此摘要的"代码公开"声明与实际可得资源存在错位，FBNet 的复现资料目前在社区不可得。

因付费墙且无任何配套结构图可下载（官方未公开 FBNet 仓库、无预印本、ResearchGate 与 IEEE 页面均不可抓取），本报告不插入网络结构图，以免硬塞无关图误导读者。

## 四、实验与结果

可确认的信息（均来自官方摘要）：

- **评测规模**：在**四个数据集**上综合评测，覆盖合成图与真实图；包含一个含 **10 款消费级相机**的多设备数据集；并跨**白天与夜间**两种光照条件。这一"4 数据集 + 10 相机 + 昼夜"的组合是摘要给出的硬信息，也是本文相对同类工作最突出的卖点。
- **结论性表述**：摘要称 "FBNet outperforms existing methods, achieving significant improvements in image clarity and quality"，并强调 "robust generalization"。
- **定量指标**：具体 PSNR/SSIM/LPIPS 数字、对比方法清单（是否含 Flare7K++ baseline、Uformer、Restormer、MPRNet 等）、以及 Flare Basis / mapping-based attention 的消融结果，均在付费墙内，**无法核实，本报告不引用任何无法溯源的数字**。
- **引用情况**：Semantic Scholar 显示该文截至 2026 年 6 月被引 1 次，社区影响尚处早期。

## 五、专家锐评

**价值**：本文最实在的贡献不在网络结构的玄学，而在**评测取向**。绝大多数夜间去 flare 论文止步于 Flare7K++ 单一管线的合成+少量真实测试，泛化性论证薄弱；本文明确把"10 款消费级相机 + 昼夜双光照 + 四数据集"作为核心论据，直击产品落地最怕的跨设备/跨光照失效问题，这种问题设定本身就有工程价值，也契合 TCE 期刊的定位。其次，"Flare Basis 潜空间 + mapping-based self-attention"把 flare 显式建模为可分离的基底子空间，相比黑箱回归在可解释性上有想象空间。这两点叠加，使其在消费电子图像增强这一应用细分里站得住脚。

**不足与质疑**（资深审稿人视角，至少三条，均基于可核查信息）：

1. **代码声明与实际不符，可复现性存疑（实锤）**。摘要白纸黑字写 "code is publicly available at github.com/Leo-student/Flare7K"，但该仓库经核查是 Flare7K 数据集（NeurIPS 2022）的 fork，骨干是 Uformer，**根本不含 FBNet 的实现与权重**；该账号 19 个仓库全是低层视觉项目 fork，无 FBNet 专属库。一篇声称"高效、强泛化"的方法，若连作者自报的代码链接都指错地方，外部既无法验证其"efficient"也无法复现其"outperforms"，这是审稿层面应当直接质询的硬伤。

2. **核心机制描述过于含糊，新颖性难以评估**。"Flare Basis"、"latent space transformation"、"mapping-based self-attention"三个名词在摘要里全是名字、没有定义：基底是字典/低秩/可学习 token？映射是线性投影还是网络？注意力在哪个维度做？在 flare 领域，"频域分解 + 注意力"、"潜空间分离 flare"早有 Flare7K++、Difflare、各类 Transformer 工作铺垫，若本文只是把已知组件换个"basis"包装，其新颖性可能高度边际。摘要不给一句机制定义，读者无从判断这是真创新还是术语翻新。

3. **"强泛化"的论证设计存在隐患**。10 款消费级相机听起来很硬，但关键问题摘要没回答：这 10 款相机的数据是**真实配对**（同场景有/无 flare）还是**仍用合成 flare 贴到这些相机的背景图上**？若仍是合成管线产生训练/测试对，那么"多设备泛化"很可能只是"多设备背景的合成泛化"，与真实光学 flare 的设备差异（不同镜头镀膜/光圈叶片数决定 flare 形态）并不等价，"robust generalization"的成色会大打折扣。在缺少真实多设备配对数据细节的情况下，这一卖点应被审慎看待。

4. **效率主张缺乏证据，与机制存在张力**。摘要反复强调 "efficient"，但自注意力（即使是 mapping-based）在去 flare 这种需要高分辨率输出的任务上通常是计算瓶颈；论文未在摘要中给出任何 FLOPs/参数量/推理时延，而"消费电子"恰恰是对端侧算力最敏感的场景。把 "efficient" 作为卖点却不提供可核查的复杂度数字，是面向 TCE 读者群的一处明显空白。

综合看，FBNet 的问题设定（消费级跨设备去眩光）有价值、评测面铺得广，但**代码指向错误、机制描述空泛、泛化与效率主张均缺可核查证据**三处叠加，使其当前的可信度与可复现性明显受限。在原文 PDF 与真实 FBNet 代码公开之前，对其"SOTA + 强泛化 + 高效"的三重声明应保持保留态度。
