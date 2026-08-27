# 大模型技术进展日报（2026-08-27）

> 覆盖区间：2026-08-25 至 2026-08-27（arXiv 部分为 v1 提交日 2026-08-24 至 2026-08-26）
> 五条主线：A. 训练 / B. 推理与服务算法 / C. AI Infra 系统与工程 / D. 理论与基础研究 / E. 硬件与算力生态

---

## 数据源状态说明

**① 历史报告读取：未能读取，本期不保证跨天去重。**
- GitHub 路径失败：本会话未加载到任何 GitHub 连接器工具（`mcp__github__*` 检索无结果）；改用公开 API 探测 `suhaibo666/tracker` 的 `llm-training-daily/` 目录，返回 `404 Not Found`（仓库为私有或路径不存在，未授权状态下二者不可区分）。
- 本地回退失败：outputs 目录为空，无任何 `llm-training-daily-*.md` 历史文件。
- 结论：本期**没有排除清单**，所有条目均按"首次出现"处理。若与往日报告有重复，属已知限制。建议在连接器设置中授权 GitHub 后恢复跨天去重能力。
- **事后补充（GitHub 授权完成后回填）**：授权后确认 `suhaibo666/tracker` 仓库根目录此前仅有 `README.md`，`llm-training-daily/` 目录并不存在——即**此前所有运行都未成功推送过任何历史报告**。本文件为该目录下的第一份报告，因此本期"无排除清单"是客观事实而非工具故障导致的信息缺失。从下一期起跨天去重可正常工作。

**② 因与往日重复而被排除的条目数量：0（无排除清单可用）。**

**③ 覆盖度与窗口说明：**
- arXiv 最新 announcement 批次为 **Wednesday, 26 August 2026**（`cs.LG/new` 页头确认），2026-08-27 批次尚未发布。因此 arXiv 实际可核实窗口为 **v1 = 2026-08-24 ~ 2026-08-26**。
- 已知覆盖度缺口：arXiv `/new` 列表页只保留最新一批。08-24 当天美东 14:00 之前投稿的论文出现在 **08-25 批次**中，该批次已被覆盖掉、无法回溯枚举，因此 v1=08-24 的论文可能有少量遗漏。
- HuggingFace Daily Papers 最新可用日期为 **2026-08-26**（`/api/daily_papers?date=2026-08-27` 返回 `date must be less than or equal to 2026-08-26`）。08-26 与 08-25 两天榜单共 81 条，其中 `publishedAt >= 2026-08-24` 的仅 36 条——**其余 45 条为老论文被重新推榜**，已全部剔除。
- 已枚举的 arXiv 分类：cs.CL、cs.LG、cs.AI、cs.DC、cs.PF、cs.AR、cs.CV、cs.NI、stat.ML、cs.NE 的 08-26 批次（New + Cross 全量）。Replacement 条目一律未收。
- 未完整枚举：华为昇腾、寒武纪、Groq、AWS Trainium 的中文闭源官方渠道。本轮英文检索未发现其在窗口内有实质发布，但不排除遗漏。

---

## 今日要点

1. **智谱 GLM-5.3-Flash 发布**（08-26）：GLM-5 系列首个原生多模态模型，320B 总参 / 18B 激活 / 45 层，首次采用**稀疏注意力 + 线性注意力混合架构** + mHC，相对 GLM-5.3 注意力算力降 3.0×、KV cache 降 4.4×。
2. **阿里 Qwen3.8-Flash-Next 开源**（08-26）：Qwen4 架构先导预览，GDN + QSA 混合注意力、Gated Residual 四分支、N-gram Embedding、Muon 优化器；125B 主模型 + 51B N-gram Embedding，每 token 激活 6B，**训练开销约为 Qwen3.7-Plus 的 1/9**。
3. **微软公开 Maia 200 加速器完整系统论文**（arXiv v1 08-25，Torsten Hoefler 等）：单芯片 10,145 Tflop/s FP4 / 750W，7 TiB/s HBM；6,144 芯片系统最多 62 exaflop/s FP4；论文称内部数据显示相对微软机队中任何其他 AI 加速器**省 30% TCO、15% 能耗**。
4. **Hot Chips 2026 Day 2 集中披露自研芯片**（08-25/26）：OpenAI × Broadcom 的 Jalapeño 推理 ASIC、Google TPU v8（8t 训练 + 8i 推理同年双发）、Meta MTIA 400、Cerebras CS-4 机架平台、SambaNova SN50、d-Matrix Raptor 3D DRAM 堆叠。
5. **vLLM v0.28.0 发布**（08-26，584 commits / 270 contributors）：E/P/D 分离进入 Model Runner V2 主线、KV cache 分层 offload 到磁盘、Kimi-K3 全栈优化、DeepSeek V4 sparse MLA 端到端打通。
6. **on-policy distillation 成为今日最密集的后训练主题**：OPDVR（与 RLVR 融合、零新增超参）、DiffusionOPSD（扩散后训练，省 40–63% GPU-hours）、OPDSearch+（先蒸馏后 RL 的搜索增强推理）三篇同窗口出现。
7. **critic 回潮**：Best Practice Critic Optimization 在 1.5B ~ 30B-A3B MoE 上把 critic 做对后，**每 prompt 只采 1 条回复**即可匹配或超过 GRPO 类 group-based 基线。
8. **KV cache 压缩出现两条新路径**：Minima-KV 的 page 粒度混合格式（FP8 + TQ3，online-softmax 合并、不留 dense shadow）与 PuzzleKV 的 page 内低秩分解（约 60% 存储保持 >96% Full-KV）。
9. **一篇诚实的负面结果**：AWS 的弹性 KV cache 论文实现了完整的 userspace CUDA VMM 回收机制，然后自证其价值有限——chunked prefill 已经把差距抹平（chunk 8192 vs 32768 的中位 TTFT 只差约 1%）。
10. **理论侧补上一个长期缺口**：Every Layer Counts 证明了 ReLU 网络在两个固定深度（浅者深度 ≥ 3）之间的首个 $L_2$ 指数分离，以及跨所有相邻深度的指数层级。

---

## 重点发布与 technical report

### 1. GLM-5.3-Flash
**机构**：智谱 / Z.ai ｜ **日期**：2026-08-26 ｜ **标签**：`[Pretraining]` `[Post-training]` `[Inference]` `[Infra]`

GLM-5 系列首个原生多模态模型，**320B 总参数 / 18B 激活 / 45 层**（对比 GLM-4.5 的 355B / 32B / 92 层）。官方称在基准与真实负载上全面超过 GLM-5.2，而价格为其十分之一，编码与 agentic 基准"approaching Claude Opus 4.8"。架构上首次引入**稀疏注意力 + 线性注意力混合结构**：线性注意力通过状态建模捕捉局部依赖，稀疏注意力通过轻量 indexer 检索全局上下文；另采用 Manifold-Constrained Hyper-Connections (mHC)。为压低 1M 上下文下 indexer 的时延与显存，提出 **IndexPool**——把 4 个 indexer key 向量加权池化压成 1 个。官方给出的口径是"每头每层注意力算力 + 每层平均 KV cache（BF16）"：相对 GLM-5.3，**注意力算力降 3.0×、KV cache 降 4.4×**；其注意力算力在所有对比模型中最低，但 KV cache 仍略大于 Kimi-K3 与 DeepSeek-V4-Flash。预训练语料为最新 **30T token 多模态语料**。发布前以匿名代号在第三方平台灰度。

公开分数（官方表格）：Terminal Bench 2.1 **84.3**（GLM-5.2 81.0 / Opus 4.8 85.0 / GPT-5.6 Terra 87.4）、DeepSWE v1.1 **63.4**（GLM-5.2 46.2）、Toolathlon Verified **78.4**（GLM-5.2 59.9）、AutomationBench v1.0.6 **48.8**（GLM-5.2 26.2）、GDPval-AA v2 **1773**（GLM-5.2 1504）、HLE w/ Tools 55.3。

- 官方博客：https://z.ai/blog/glm-5.3-flash
- 权重：https://huggingface.co/zai-org/GLM-5.3-Flash

### 2. Qwen3.8-Flash-Next
**机构**：阿里 Qwen ｜ **日期**：2026-08-26 ｜ **标签**：`[Pretraining]` `[Infra]`

开源权重多模态 MoE，官方定位为 **Qwen4 模型结构的先导预览**（角色对标 Qwen3-Next 之于 Qwen3.5）。四项系统性升级：**Attention** 用 GDN + QSA Hybrid（Gated DeltaNet 压缩历史信息，Qwen Sparse Attention 通过压缩式轻量 Indexer 在 micro-block 粒度筛选上下文）；**Residual** 引入 Gated Residual，把 residual stream 扩成 4 条分支、用动态 Gate 控制读写；**Embedding** 引入 N-gram Embedding，Embedding Table 可卸载到 Host Memory 并通过异步 Prefetch 与计算重叠；**Optimization** 采用 Muon 优化器，围绕正交化精度、Muon 与 AdamW 的参数分工、融合参数拆分做优化，并针对新架构重新拟合 Scaling Law。

规模：主模型 **125B + 额外 51B N-gram Embedding，每 token 激活 6B**。原生 262,144 token 上下文，YaRN 可扩至 1,000,000。官方称相比 Qwen3.7-Plus（397B / 17B 激活）**训练开销仅约为其 1/9**，但编码与办公任务能力更强。公开分数：DeepSWE 1.1 **58.7**（Qwen3.7-Plus 16.5）、SWE-bench Pro **62.5**、SWE-bench Multilingual **81.0**、NL2Repo-Bench 48.1、CoWorkBench **73.9**、JobBench **55.7**、Toolathlon Verified 73.5、LiveCodeBench v6 91.9、GPQA Diamond 91.7。生产版以 Qwen3.8-Flash 名义服务，输入 ¥1 / 输出 ¥3 每百万 token。

- 官方博客：https://qwen.ai/blog?id=qwen3.8-flash-next
- 权重：https://huggingface.co/Qwen/Qwen3.8-Flash-Next
- 配套 SGLang day-0 工程文见 C 节第 2 条。

### 3. Apodex 1.1: Scaling Agentic Intelligence for Complex Work
**机构**：Apodex ｜ **arXiv v1**：2026-08-24（v2 08-25）｜ **标签**：`[Post-training]`

以 "working capability"（可持续、可验证地推进真实目标）为目标的通用模型 + 执行系统技术报告。两条 scaling 路径：**Environment Scaling**（围绕文件/搜索/代码等可执行环境构建数千万量级训练任务）与 **Agentic Coordination Scaling**（模型自行决定任务是否拆分、拆成几个 Subagent、何时汇合，形成异步 Agent Team），二者共享同一执行底座 AgentOS。训练为 SFT + agentic RL，核心方法 **PIVOT-RL**：用 Hindsight-Guided Trajectory Localization 在数十万条轨迹上回溯定位 pivot 关键决策点，保留有效前缀并构造带短提示的局部续写任务（提示仅训练期存在、非预测目标、推理时不出现），有状态任务还会恢复可执行环境状态。另有 Statement Review（生成与审查分离）。35B 的 Apodex 1.1 Mini 可本地部署，配套开源 harness FrontierAgent。AI4AI 案例：以 Apodex 1.1 全自动做 Teacher，10 轮迭代把 Qwen3.5-0.8B 在 200 题的三类检索任务上从 **51.0% 提到 56.0%**。

> ⚠️ 官方博客与 arXiv 摘要只给"进入领先性能带（leading performance band）"这类限定表述，**未给出与具体前沿模型逐项可比的 benchmark 分数**，网络流传的逐项分数本期未能核实，不予采用。

- arXiv：https://arxiv.org/abs/2608.23283
- 代码：https://github.com/ApodexAI/FrontierAgent

### 4. WeMM-Embedding: WeChat Multi-Modal Embedding Technical Report
**机构**：WeChat Vision, Tencent Inc.（HTML v1 作者块）｜ **arXiv v1**：2026-08-25 ｜ **标签**：`[Pretraining]`

通用多模态 embedding 模型族，统一对齐文本、图像、视频、视觉文档与任意交错多模态输入，输出维度可变，含 **2B / 4B / 9B** 三档。两阶段训练：大规模多模态对齐 → 精修阶段（精选数据、细粒度相关性监督、cross-scale knowledge transfer）。**2B 版即超过此前领先的 8B 开源基线**；在 26 项内部任务基准上有大幅提升，并在 **14 次线上 A/B 测试**中一致改善，已在视频号、公众号、朋友圈与电商规模化部署。权重与代码开源。当日 HF Daily Papers 榜首。

- arXiv：https://arxiv.org/abs/2608.24053
- 权重：https://huggingface.co/tencent/WeMM-Embedding-9B

### 5. Prime Agent: A Self-Improving RLM Harness
**机构**：Prime Intellect ｜ **arXiv v1**：2026-08-24 ｜ **标签**：`[Inference]` `[Post-training]`

开源的长时程评测与编码 agent harness。持久化 IPython REPL 落实 **Recursive Language Model** 抽象（把 context 当变量、把 subagent 委派当 REPL 里的函数调用）；**Continual Harness** 把历史、记忆、技能、提示词与 subagent 规格作为持久状态跨轨迹保留，并允许模型以小步、有证据支撑的方式自行更新。递归 subagent 之间直接 agent-to-agent 通信。核心主张是"低摩擦膜"——避免 harness 失败被误记为模型失败。可验证数字：**ARC-AGI-3 RHAE Best@1 从 30% 提升到 95.5%**；在长上下文编码、GPU kernel 生成、模拟器构建、nanoGPT 自主 speedrun 上匹配或超过原生及主流 harness。

- arXiv：https://arxiv.org/abs/2608.23552
- 代码：https://github.com/PrimeIntellect-ai/prime-agent

### 6. EchoWM: Open and Enterable Omnimodal World Models
**机构**：HKUST / PKU / 京东 Joy Future Academy / HKU / THU / USTC / FDU / 北航 / Stanford ｜ **arXiv v1**：2026-08-24 ｜ **标签**：`[Pretraining]`

面向"可进入式生成媒体"的全模态世界模型：响应连续导航的同时**联合生成 720p 视频、环境音、音乐与语音**。交互围绕 camera intent 组织——第一人称场景中它指定观察者运动，第三人称场景则从数据中直接学习相机–角色动力学，不需要视角专用控制器。离散指令与连续位姿被统一映射到共享的**米制尺度相对 6-DoF 轨迹**，并通过数据集级标定在异构数据间保持运动幅度一致。训练用互补数据引擎 + 渐进式训练 + 自回归后训练以支持长时程生成。42 页 24 图，代码开源。

> ⚠️ 网络二手摘要流传的 "WBench Navigation 81.7 / Interaction 87.9" 等分数**未能在 arXiv 摘要与 HTML 首屏核实**，本报告不采用。

- arXiv：https://arxiv.org/abs/2608.23189
- 代码：https://github.com/jd-opensource/JoyAI-Echo

---

## A. 训练（Training）

### A1. Data Mixing as Mixture Experiment: Response Surface Methodology and Optimal Design for Large Language Model Pretraining
`[Pretraining]` ｜ University of Calgary（数学与统计系）+ University of Virginia（系统与信息工程系）（HTML v1 作者块直接核实）｜ v1：2026-08-24

把"用小 proxy 模型试配比 → 拟合响应面 → 外推到大模型"重新表述为经典的**混料实验（mixture experiment）**：数据域 = 混合成分，token 占比 = 成分比例，proxy run = 实验设计点，验证 loss = 概率单纯形上的响应曲面。用稀疏二阶 Scheffé 响应面模型 + model-robust $\mathcal{I}$-optimal 设计。以 RegMix 为实证案例，Scheffé 分析显示域价值是强**关系性**的：若干在加性效应下表现弱的域，通过与 web 文本的**两两交互**变得有利；稀疏 Scheffé 模型在不同模型规模间保持配比排序。仿真中 model-robust $\mathcal{I}$-optimal 设计在**去掉约 25% 的原始 proxy run** 后仍能恢复正确的配比排序。

**为什么值得看**：数据配比搜索第一次被放进有几十年积累的实验设计框架，回答的是 proxy 实验该怎么**设计**，而不只是怎么拟合。
https://arxiv.org/abs/2608.23922

### A2. Effective Learning Rate Governs Loss Dynamics in Language Model Pretraining
`[Pretraining]` `[Theory]` ｜ 北京大学 + 蚂蚁集团（HTML v1 作者块）｜ v1：2026-08-25

提出 **ELR collapse**：学习率与参数范数主要通过其比值（effective learning rate, ELR）影响 loss 动力学；ELR 对齐后，即使 LR 与参数范数差异很大，loss 轨迹也会全程重合。跨优化器、架构、数据集与模型规模，平均 collapse 误差**通常在几 × 10⁻³**，低于代表性配置下测得的 seed-to-seed 波动。消融指出 normalization 设计与 LR-norm 变化的时间尺度是 collapse 精度的关键；weight decay 与 Hyperball 主要通过它们诱导的 ELR schedule 起作用。把 LR 换成 ELR 后，拟合的 functional scaling law 可以在不同 norm-control 方法之间迁移，并解释 norm control 常见的 "delayed acceleration"。

**为什么值得看**：给 LR / weight decay / norm control 的调参与 scaling law 拟合提供了一个跨优化器统一的单标量坐标。
https://arxiv.org/abs/2608.24814

### A3. The Mask Is Not the Model: Auditing Prefix Invariance in Attention, State-Space, and Hybrid Sequence Models
`[Pretraining]` ｜ VIDRAFT AI Research · QuantumOS, Seoul（HTML v1 作者块）｜ v1：2026-08-24（v2 08-25）

形式化 "prefix invariance"（位置 t 的表示不得依赖未来输入），给出**两次前向、无需训练与梯度**的轻量审计，产出逐层分数定位因果性在哪一层被破坏。指出业界默认的 attention-mask 检查不完备：因果性是图级性质，泄漏可经由 scan、聚合或 normalization 发生。8 个 checkpoint 上的 **192 次注入故障试验中，mask 检查一个都没发现，而该审计把 192/192 全部定位到确切层**；对 transformers 库中 chunked-scan 代码的静态/动态分析在 **Zamba2 与 Nemotron-H** 中发现同类缺陷（inter-chunk 轴错误，已按参考实现修复）。

**为什么值得看**：混合/线性注意力架构正在变成主流（见本期 GLM-5.3-Flash 与 Qwen3.8-Flash-Next），这是一个能直接接进 CI 的因果性回归测试，且真在两个知名开源模型里抓到了 bug。
https://arxiv.org/abs/2608.22876

### A4. Delayed Optimizer-State Transport Shapes Short-Horizon Training Decisions
`[Pretraining]` `[Theory]` ｜ 北京航空航天大学物理学院（HTML v1 作者块）｜ v1：2026-08-25

研究 Adam 类优化器 moment 变量中的"梯度历史"是否大到足以改变**短程**训练决策。在已确定的未来 minibatch 序列上，对 8 步 AdamW 轨迹按完整的 model–optimizer 状态求导，选择 exposure-matched 的 Math–Code loss 调度。在 **12 段未使用过的 0.3M 参数 Transformer 历史**上，full transport 相对"optimizer-aware 即时导数"在 **10/12** 段上降低了 token-disjoint loss（平均收益 4.71×10⁻⁴，精确单边符号检验 p = 0.0193）；两个控制器动作频次相同但在 **60/96** 个窗口中选出不同调度。

**为什么值得看**：数据调度/课程设计的在线决策普遍用即时梯度近似，这里量化了忽略优化器状态记忆的代价——注意规模很小（0.3M 参数），结论的外推性待验证。
https://arxiv.org/abs/2608.24593

### A5. LAION-BVD: A 10-Million-Hour Open Video Dataset for Multimodal Pre-training
`[Pretraining]` ｜ 机构未核实（HTML v1 不可得；作者含 Christoph Schuhmann、Jenia Jitsev、Matthias Bethge、Bernhard Schölkopf）｜ v1：2026-08-25

从 CommonCrawl 收集 **13 亿个平台视频 URL**，实际下载 **8000 万个视频、总时长 1000 万小时**，面向 video/audio/image 三模态预训练。用 content-aware 场景检测切片，并合成生成视频与音频 caption。在标准 video-text / audio-text benchmark 上训练出的模型表现有竞争力，且随训练量与模型规模增大持续提升。另把场景切换帧当作 image-text 数据的替代来源，其视觉分布与常规 web 图文语料明显不同，训练出的模型在 image-text 检索上表现强。数据集已开放。

**为什么值得看**：开放视频预训练数据在规模上跨了一个量级，且给出了"视频帧作为图文数据源"的分布差异证据。
https://arxiv.org/abs/2608.24845

### A6. Giga-Embeddings: Mixture-of-Experts Encoders for High-Throughput Text Embeddings
`[Pretraining]` ｜ MIPT、SaluteDevices（HTML v1 作者块）｜ v1：2026-08-24

最大成员是 **10B 参数稀疏 MoE encoder，每 token 约 1.8B 激活参数**；在英语、俄语、多语种、代码四套 MTEB 上取得族内最强综合表现。vLLM 基准（1024-token 输入）下达 **114.5k tokens/s**，比同族 dense 3B 模型高 **25%** 吞吐，为所评估外部系统的 **1.56–2.65×**。族内还有 dense 3B 与蒸馏出的 480M encoder；后者用"维度无关"目标对齐师生相似度分布，在 Russian MTEB 上取得 **70.98**。

**为什么值得看**：把 MoE 用在 encoder / embedding 侧并给出完整吞吐数字，是少见的 MoE 编码器工程报告。
https://arxiv.org/abs/2608.23806

### A7. Towards a Densing Law for User Representation Learning at Billion-Scale Capacity
`[Pretraining]` ｜ Ant Group、浙江大学（**推断**：HTML v1 机构字段未正确渲染，依据作者通讯邮箱 `@antgroup.com` / `@zju.edu.cn`）｜ v1：2026-08-24

指出十亿级用户表征学习中原始行为文本继续扩容会收益递减，可用 tokenization 缓解；提出 **User Behavioral Densing Law**，从"跨来源冗余 / 源内独特性"两个维度对 tokenization 如何分配容量给出量化刻画。据此设计 **ALGN**，一种自适应变长 tokenization 方法以改善容量分配；跨多种数据源、tokenization 方法与下游任务的实验验证了该定律的普适性。

**为什么值得看**：把 scaling / densing law 的思路搬到 tokenizer 设计上，且是工业级用户行为序列场景。
https://arxiv.org/abs/2608.23392

### A8. On-policy Distillation with Verifiable Reward
`[Post-training]` ｜ 机构未核实（HTML v1 无作者单位块；作者含 Shenzhi Wang、Bingxiang He、Gao Huang）｜ v1：2026-08-25

RLVR 反馈稀疏（任务级），on-policy distillation (OPD) 有 dense token 级信号但忽略轨迹正确性、性能被教师封顶。**OPDVR** 先把 sampled-token OPD 的隐式 reward 按轨迹正确性重构，再用 **ReLU gating** 保证正确轨迹得到非负 reward、错误轨迹得到非正 reward，从而在保留教师分布指导的同时对齐任务成功信号；**不引入任何额外超参数**。该改造把 sampled-token OPD 变成一个真正的 RLVR 方法，可与任意 policy gradient 算法（如 GRPO）组合。六个推理 benchmark 上一致优于标准 OPD，代码已开源。

**为什么值得看**：OPD 与 RLVR 的融合此前多靠加权或启发式切换，这里给出零新增超参的统一形式，很容易接进现有 GRPO 流程。
https://arxiv.org/abs/2608.24696

### A9. Best Practice Critic Optimization
`[Post-training]` ｜ National University of Singapore（HTML v1 作者块）｜ v1：2026-08-24（v2 08-25）

针对 GRPO 类"无 critic、每 prompt 采多条回复"的做法，回头把 critic 做对：**BPCO** 组合 DPPO、把 value 预测**限制在 reward 值域内**、Monte Carlo value target、未归一化的 policy advantage、以及 length-adaptive GAE。由于 critic 只在训练时使用，还可以让它看到策略看不到的信息（参考答案、评分 rubric）。受控实验逐项隔离各设计的贡献；在数学推理任务上、**从 1.5B 到 30B-A3B MoE** 的模型上，BPCO 一致改善强 critic 基线，并在**每 prompt 只采一条回复**的情况下匹配或超过 group-based 基线；同一配方在 rubric-based reward 下也有提升。代码已开源。

**为什么值得看**：如果 critic 能稳，采样成本可以降到 1 条/prompt——这是对 GRPO 主导范式的一次实打实的反击，且有逐项消融。
https://arxiv.org/abs/2608.23566

### A10. Beyond the Stability-Exploration Dilemma: Environmental Regularization for LLM Policy Optimization
`[Post-training]` ｜ AMAP, Alibaba Group；西安交通大学（HTML v1 作者块）｜ v1：2026-08-24（v2 08-25）

论点：动作侧的 Policy-KL 正则让实践者两难——留着它会约束回复行为并吃掉探索预算，去掉它又没有漂移控制。**ERPO** 把正则移到**输入侧**：引入 Query-KL (QKL) 约束当前策略诱导的训练 query 分布相对 pre-RL 参考分布的漂移，并加一个数据集静态的、由参考分布导出的 per-query 权重。关键性质是 QKL 的梯度**只经由 query 似然**流动，policy-gradient 估计量里的 response score function 不出现在 QKL 项中，因此对回复分布没有直接梯度压力，探索得以保留。可无额外前向开销地插入 GRPO / PPO / REINFORCE 流程；六个数学推理 benchmark 上替代标准 Policy-KL，在**高温解码与长训练**下精度更高、行为明显更稳。

**为什么值得看**：一个可以直接替换掉 KL 项的改动，且给出了"为什么不伤探索"的梯度层面论证。
https://arxiv.org/abs/2608.23311

### A11. SPO++: Stream-Aligned Policy Optimization for Asynchronous Agentic RL
`[Post-training]` `[Infra]` ｜ 中国人民大学高瓴人工智能学院等（HTML v1 作者块）｜ v1：2026-08-25

Group-relative RL 需要等同一 prompt 的兄弟 rollout，对长且长度可变的工具调用轨迹代价很高；SPO 用持久的 prompt 级 value 估计去掉了这一依赖，但其配方是"先对每条轨迹的单个 advantage 做 whitening，再优化 token-mean actor loss"。本文指出**轨迹级 centering 一般并不能 center actor 实际消耗的 token 加权量**，改为在 **action-token measure 下标准化终局 advantage**；并把 prompt evidence 按"产生它的策略事件"而非"learner 接收顺序"组织。在 ALFWorld（两个模型规模）与 Math-TIR 的 matched run 上，SPO++ 的在线学习效率优于 SPO。

**为什么值得看**：异步 agentic RL 里一个具体且容易被忽略的测度不匹配 bug，修法明确。
https://arxiv.org/abs/2608.24870

### A12. IAPO: Influence-Aware Policy Optimization for Credit Assignment in Multi-Turn Service Agents
`[Post-training]` ｜ 复旦大学；WeChat, Tencent Inc.（HTML v1 作者块）｜ v1：2026-08-25

多轮服务 agent 中信息随对话展开，最终 reward 无法指明哪一步动作有贡献。IAPO 的观察是：**一条已完成的 rollout 本身就记录了信息与错误在动作之间的流动**，于是把 rollout 表示为可训练 agent 动作之上的**带类型的 influence-dependency 图**（用户与工具的 observation 作为证据），把 support-use / failed-use 结构转成动作级 advantage。Qwen3-4B 与 Qwen3-8B 上，在 τ²-Bench、UserBench、AgentChangeBench 三个服务 agent benchmark 上优于多轮 RL 基线；BFCL-v4 Multi-Turn 显示收益不以牺牲多轮函数调用能力为代价。

**为什么值得看**：不需要额外采样或过程标注，只从已有 rollout 结构里榨出 credit assignment 信号。
https://arxiv.org/abs/2608.24588

### A13. Contrastive Branch Policy Optimization
`[Post-training]` ｜ Alibaba Group（杭州）；哈尔滨工业大学（HTML v1 作者块）｜ v1：2026-08-25

指出 branch sampling 类方法混淆了两个不同问题：**固定 rollout 预算的分配**，与**把分支结果翻译成 token 级 credit**。CBPO 拆开这两者，只在**非重叠的 credit 段**上做归因，从而避免共享 token 上的重复梯度；只需 outcome reward，不需要任何过程级标注。在十个 benchmark（五个数学推理 + 五个知识密集检索）、两个模型规模上，CBPO 一致优于 SOTA 的 policy-optimization 与 branch-based 方法，两个领域的 macro-average 准确率均最高。

**为什么值得看**：把"预算分配"和"credit 翻译"解耦，是分支采样类 RLVR 方法一个清晰的设计澄清。
https://arxiv.org/abs/2608.24300

### A14. OPDSearch+: On-Policy Distillation with RL Refinement for Search-Augmented Reasoning
`[Post-training]` ｜ 机构未核实（HTML v1 仅有作者姓名）｜ v1：2026-08-25

两阶段方案：**阶段一**学生与**真实搜索引擎**交互，用 per-position forward KL 目标蒸馏，无需任何任务专用的教师训练即可迁移推理分解与证据整合能力；**阶段二**在这个更丰富的行为基础上做 RL 精修，达到纯 RL 从零达不到的水平。七个 QA benchmark 上，**3B 模型**一致优于所有此前的 3B RL 基线，HotpotQA **+13.1%**、2WikiMultihopQA **+8.5%**。

**为什么值得看**："先 on-policy 蒸馏、再 RL"的两段式配方在小模型 agentic 检索上给出了明确增量数字。
https://arxiv.org/abs/2608.24310

### A15. On-Policy Self-Distillation in Diffusion Models
`[Post-training]` ｜ 机构未核实（HTML v1 不可得）｜ v1：2026-08-25

RL 能把扩散模型对齐到人类偏好，但**终点 reward 并不告诉中间去噪预测该往哪改**。**DiffusionOPSD** 把 image-level 的 reward 指导转成对采样 query 处 clean-output 预测的显式目标：每个外层迭代由冻结的 behavior policy 生成轨迹并提供 query state 与 anchor，reward 梯度在 anchor 周围构造**有界的正/负目标**；可训练策略以 detached supervision 的方式做有限次拟合，再用 EMA 刷新 behavior policy。相比最强竞争方法**最多提升 44.0%**；相对 DiffusionNFT，在 SD 3.5-M 上减少 **40%** 训练 GPU-hours、在 Z-Image-Turbo 上减少 **63%**。

**为什么值得看**：把 on-policy distillation 的思路搬进扩散后训练，省算力的幅度很大。
https://arxiv.org/abs/2608.24646

### A16. Preference Data Selection for Mitigating the Alignment Tax in Large Language Models
`[Post-training]` ｜ Microsoft Research Asia（HTML v1 作者脚注，一作实习期间完成）｜ v1：2026-08-25

从**偏好数据本身的属性**（而非优化器或架构）切入 alignment tax / 灾难性遗忘问题。**BALIGN** 用几个正交特征——包括 chosen 与 rejected 回复之间的关系、以及与通用能力语料的 **TF-IDF 相似度**——聚合成统一的 composite risk score，系统性过滤掉那些会扰动模型内在参数、或对齐效用极低的高风险偏好样本。在标准人类偏好数据集上，BALIGN 在不牺牲对齐收益的前提下保住基础能力，以极小的额外计算开销取得最优 Pareto 前沿。

**为什么值得看**：alignment tax 的数据侧解法，成本低、可直接叠加在现有 DPO / RLHF 数据管线前。
https://arxiv.org/abs/2608.24192

### A17. Robust Code RL via Faulty-Code-Driven Test case Synthesis and Dense Reward Shaping
`[Post-training]` ｜ 浙江大学；Ant Group（HTML v1 作者块）｜ v1：2026-08-25

代码 RLVR 的效果受限于测试用例覆盖度：覆盖不足导致 false positive，进而 reward hacking 与策略退化。**RobustTests** 用**接近正确的错误代码**驱动测试用例合成，引导模型精确捕捉潜在逻辑差异；再用 validator agent + 行为特征聚类做细粒度过滤，剔除无效与冗余用例。产出一个扩充了 CodeContests 测试用例的高质量数据集。用 CodeContests 中中等难度子集对 **Qwen3-32B** 做 RL 微调，在 **LiveCodeBench 上相对基线绝对提升 3%**。

**为什么值得看**：直指代码 RLVR 里最脏的一环——验证器质量本身，并给出可复用的数据集产物。
https://arxiv.org/abs/2608.24135

### A18. Task-Adaptive Rubrics for GUI Reward Modeling
`[Post-training]` ｜ 浙江大学；MiLM Plus, Xiaomi Inc.（HTML v1 作者块）｜ v1：2026-08-25

**AdaptRubric** 两步走：先做**类别级粗 rubric 检索**（把指令路由到某个 GUI 任务族，取出可复用的任务族判据），再做**实例级细 rubric 生成**（针对当前指令中的具体取值、范围与约束生成紧凑线索）。在离线 reward 评测与在线 RL 优化上都一致优于此前的 reward agent：在**图像预算对齐**的条件下 F1 比基线平均高 **3.6 点**，任务成功率提升 **4.23 点**。

**为什么值得看**：rubric reward 在 GUI / agent 场景的具体工程化方案，离线与在线两侧都有数字。
https://arxiv.org/abs/2608.24174

### A19. Mitigating Exploration Bias in RL for Multi-Instruction Following
`[Post-training]` ｜ UT Dallas；UT Austin；UC Santa Barbara（HTML v1 作者块）｜ v1：2026-08-24

发现当一个 prompt 里含多条指令时，现有 RL 配方会**偏向探索简单指令**：策略模型对难指令的初始能力太低、RL 中触发不了成功探索，进而使优化被简单指令主导。两个对策：**Behavioral Bootstrapping**（RL 前一段轻量的 rejection sampling 微调，用于"激活"难指令）与 **Scarcity-Aware Rewards**（按指令的经验稀缺度分配 reward）。所提指标与模型表现高度相关，最佳模型在**三个可验证指令跟随 benchmark** 上显著超过基线。

**为什么值得看**：多指令 prompt 是真实产品里的常态，这里把"RL 只学会了简单那条"的失效模式量化并给了解法。
https://arxiv.org/abs/2608.23830

### A20. SkillForge: Evolving Verifiable Skills for Reinforcement Learning Agents
`[Post-training]` ｜ AMAP, Alibaba Group（HTML v1 作者块）｜ v1：2026-08-25

现有 RL 训练的 agent 多是 episodic 的，跨 episode 无法积累可复用知识；SkillRL 之类方法从原始轨迹里抽 skill，但把 skill bank 当成只增不删的仓库，从不验证存下来的 skill 是否仍然有效。SkillForge 让 **skill 调用在 agent 交互中显式化**，从而 RL 可以同时优化环境动作与 skill 调用决策；再加上**基于证据的 skill 验证**与**多路径 skill 归纳**，让 skill bank 在持续增长的同时保持质量。ALFWorld、WebShop、AppWorld 上一致优于 SkillRL。

**为什么值得看**：给"agent 记忆/技能库"这类容易变成堆垃圾的机制加上了可验证的淘汰环节。
https://arxiv.org/abs/2608.24747

### A21. BrowserForge: Scaling Web Episode via Parallel Browser Sandboxes
`[Post-training]` ｜ 浙江大学；LLM Department, Tencent（HTML v1 作者块）｜ v1：2026-08-25

并行驱动大量浏览器沙箱在开放 web 上产数据，三个组件：开放 web 取样（暴露给 agent **数十万个真实可达站点**）、调度**数百个并发浏览器**的沙箱集群管理器、以及 Proposer-Solver 双 agent 回路。可达性树只作为**合成期**信号，发布的 agent 纯粹从截图行动。语料含 **203,238 条轨迹，每条来自不同站点**。在其上微调紧凑多模态模型，live Online-Mind2Web 成功率从 **25.66% 提升到 33.33%**，static Multimodal-Mind2Web 的 step accuracy 也一致提升且随语料规模增长。

**为什么值得看**：把 web agent 训练数据的站点多样性提高了一个量级，并有明确的 scaling 趋势证据。
https://arxiv.org/abs/2608.24848

### A22. PROOF-Gen: From Optimized Data to Better Distillation
`[Post-training]` ｜ Apple（HTML v1 作者块）｜ v1：2026-08-24

观察：在 τ²-bench 上**57% 的教师尝试失败，其中三分之二是 near-miss**（大部分工具调用正确，被一个决定性错误毁掉）。标准的 generate-and-filter 每个迭代周期都重复付出前沿教师的成本、且总是留下同一批硬场景。PROOF-Gen 用**逐场景的 prompt 优化**从失败里回收黄金轨迹：reflector 分析执行 trace 与评测反馈、写出纠正性指导把教师引导到通过的轨迹；训练前把这些指导剥掉，学生学到的是干净示范。τ²-bench 上逐场景优化**回收了 93% 的失败场景**；在合并数据上微调，Qwen3-4B-Instruct-2507 的 Pass@1 从 **0.132 提升到 0.529**，Gemma 4 E4B-it 在 BFCL v4 multi-turn 上 **+7.2pp**；在已部署管线中把轨迹质量提升 **6.3pp goal completion**。

**为什么值得看**：教师失败样本的回收率与学生提升幅度都很大，对周期性重跑蒸馏管线的团队直接省钱。
https://arxiv.org/abs/2608.23911

### A23. Industrial-Instruction: An End-to-End Framework for Building Instruction-Tuning and Benchmark Datasets from Industrial Technical Reports
`[Post-training]` ｜ 机构未核实 ｜ v1：2026-08-24

工业技术报告结构异质（密集正文、规格表、表格），标准检索与 QA 管线难以索引与推理，且此前**没有**从这类文档构建的公开 instruction-tuning 或 benchmark 数据集。本文填补该空缺，贡献包括从这类文档构建的两个开放 QA 数据集。

**为什么值得看**：垂直领域指令数据的空白填补；技术含量以数据集构建为主，价值判断建议直接读原文。
https://arxiv.org/abs/2608.22817

---

## B. 推理与服务算法（Inference & Serving）

### B1. Serving Masked Diffusion LLMs: Characterization and Design Principles from Real Hardware
`[Inference]` ｜ Virginia Tech（HTML v1 作者块）｜ v1：2026-08-24

首个在真实并发负载下对掩码扩散 LLM 做服务侧刻画的工作（LLaDA-8B-Instruct + D2F LoRA，单张 H200，GSM8K / HumanEval）。三个发现：请求难度（去噪步数）是**离散**的，落在 11 个固定档位，且生成前无任何信号可预测；单请求 wall-clock 只有 **24% 是 GPU 计算**，其余是 CPU 侧 dispatch 开销；batching 的主要收益是摊薄该开销——**batch=16 时共享每步前向使吞吐相对 per-request dispatch 基线提升 16.0×**。并给出 Poisson 到达下 fixed-fill 同步批处理的 batch-timeout 规则。

**为什么值得看**：dLLM 服务系统正在被大量提出，但这是第一篇先测量再定设计原则的，直接指出"AR serving 的假设不能照搬"的具体位置。
https://arxiv.org/abs/2608.23807

### B2. Elastic KV Cache for LLM Serving: A Working Reclamation Mechanism, and Why Chunked Prefill Already Closes the Gap
`[Inference]` ｜ Amazon Web Services（HTML v1 作者块）｜ v1：2026-08-24

用纯用户态 CUDA VMM 路径实现"弹性 KV cache"：每层两个 physical handle 映射进一段连续虚拟地址，decode 期把 prefill 预留显存借给 KV pool、prefill 前归还，attention kernel 不改、无需 driver patch，decommit 数毫秒 / recommit 数十毫秒。**然后诚实报告负面结果**：该机制只有在小 prefill chunk 严重伤害 TTFT 时才划算，而实测 chunk size 8192 与 32768 的**中位 TTFT 只差约 1%**；直接调低 `max_num_batched_tokens` 回收的 KV 比控制器还多。预留显存还会被 TP 稀释，从 TP1 的 16% KV 降到 TP4 的 2.7%。

**为什么值得看**：一篇把"看起来显然该做的优化"证伪的工程报告，省下别人重复造轮子的时间；同时留下了可复用的 userspace elastic-VMM allocator。
https://arxiv.org/abs/2608.23658

### B3. More GPUs or a Smaller Cache? Tensor Parallelism versus KV Compression for Memory-Bound LLM Serving
`[Inference]` `[Infra]` ｜ HTML v1 作者块标注 "MIT and Vizuara"（未经第三方交叉验证）｜ v1：2026-08-25

把 TP（degree 1–8）与 KV 压缩（16/8/4-bit，keep-ratio 低至 0.25）放在同一条"每百万 token 成本 vs 延迟"轴上，用在 A100 / A40 / H100 上标定的 profiled simulator 比较。结论：压缩便宜 **1.20×–2.00×**；决定分界的是模型大小相对于显存，**80GB 卡上约在 36B 参数处**；分界以下 TP 基本是浪费，以上 TP 变成入场券。TP 是唯一能改善延迟的杠杆（压缩因 batching 争用反而使每 token 延迟恶化 8–93%），压缩是唯一能倍增单位成本容量的杠杆（**16.5× vs 八倍 GPU 支出的 1.21×**）。

**为什么值得看**：压缩论文报内存比、并行论文报吞吐曲线，几乎没人放同一成本轴上比——这篇给了容量规划可直接引用的分界点。
https://arxiv.org/abs/2608.23962

### B4. Minima-KV: Retention-Preserving KV Cache Compression with Mixed-Format Paged Attention
`[Inference]` ｜ Minima AI, Inc.（abs 页作者标注）｜ v1：2026-08-24

混合格式分页 attention：近期与受保护的 anchor page 保持 FP8，较老的 non-anchor page 打包成 TQ3，所有在途请求的 page 仍可寻址、不驱逐。格式专用 kernel 各算部分 attention state，再用全局归一化的 online-softmax 合并，从而直接异构 decode，**不需要 cache 大小的 dense shadow**。单张 96GB RTX PRO 6000 Blackwell 上跑 Qwen3.6-27B：每活跃 token 18.3 KiB attention KV，相对 BF16 压缩 **3.50×**、相对 FP8 压缩 **1.75×**；两条 59,008-token 请求的 decode canary 实测 3.625× 活跃 KV 压缩、吞吐为对照的 **0.9821×**，16 个 full-attention 层全部走压缩路径无 fallback。

**为什么值得看**：把"量化 KV"从整池统一格式做成 page 粒度的混合格式 + online-softmax 合并，是 paged attention 之后 KV 内存管理比较自然的下一步。
https://arxiv.org/abs/2608.23834

### B5. PuzzleKV: Page-Wise Low-Rank Decomposition for KV Cache Compression
`[Inference]` ｜ 机构未核实（HTML v1 作者块无单位）｜ v1：2026-08-24

观察到把每个 head 的 KV cache 切成定长逻辑 page 后，**单个 page 内部**存在显著低秩结构（优于从权重导出的固定投影空间或跨大区域的共享基）。据此提出免训练、免校准的 PuzzleKV：对每个写满的 page 单独分解，dense page 与 factorized page 混合存放，decode 过程中增量压缩新达标的 page。在**约 60% 原始 KV 存储下，两个模型的全部 benchmark 设置都保持 >96% Full-KV 性能**，RULER 上大幅优于 Global SVD；叠加量化后用 **18.7% 存储仍保留 >93%**。

**为什么值得看**：低秩 KV 压缩长期卡在"固定基不够贴合"，page 粒度分解 + 增量压缩是能直接嵌进 paged attention 的落地形态。
https://arxiv.org/abs/2608.23843

### B6. AgentSpec: Speculative Decoding for Batch Inference of LLM Agents
`[Inference]` ｜ The Ohio State University + Microsoft Research + University of Michigan（HTML v1 作者块）｜ v1：2026-08-25

系统分析了 agent workload 下投机解码在大 batch 时崩掉的两个主因：投机 token 拒绝率高、动态 token 预算利用不足（现有 SD 算法在 agent batch 推理下甚至**慢于**普通自回归解码）。AgentSpec 用 structure-isolated drafting 把投机限制在结构化片段上以压低拒绝率，加 redundancy-aware budget allocation 利用 agent 级信息填满动态空闲预算。在 vLLM 上跨 4 个模型家族、5 种 workload 评测；**相对自回归解码最多 2.02× 加速**，Spec-Bench 上 1.14×。

**为什么值得看**：agent 场景是 SD 最该赢的地方，但恰恰在大 batch 下失效，这篇给出了原因分解与在 vLLM 里可复现的修法。
https://arxiv.org/abs/2608.24004

### B7. ResiSpec: Enhancing Multi-Candidate Speculative Sampling via Residual Distribution Shaping
`[Inference]` ｜ 机构未核实（HTML v1 作者块无单位）｜ v1：2026-08-25

指出多候选投机采样的瓶颈是 **Residual Drift**：前面的候选被拒后，残差目标分布偏离 draft 模型预测，使后续候选失效并被迫走昂贵的重采样。ResiSpec 在验证阶段重塑提案分布，把残差目标质量锚定在 draft 模型高置信区域，在不破坏输出精确性（output exactness）的前提下重新对齐验证过程，**相对 SOTA 多候选方法最多 1.92× 加速**，已开源。

**为什么值得看**：多候选 SD 的加速上限一直上不去，这篇把失效机理讲清楚了，且属于保持分布无损的那类修改。
https://arxiv.org/abs/2608.24411

### B8. Parason: Revealing Subtask and Trial Parallelism in LLM Reasoning
`[Inference]` `[Post-training]` ｜ 清华大学 + NVIDIA + MIT（HTML v1 作者块）｜ v1：2026-08-25

区分推理并行的两种形态：已被大量研究的 Subtask Parallelism，和被忽视的 **Trial Parallelism**（多路投机尝试并行探索、验证、聚合）。分析显示 Trial Parallelism 才是可并行推理计算的大头（DeepSeek-* 上 **65.5%**）。用 Parallelism-Aware GRPO（奖励同时平衡准确率、延迟和两类并行比例）让模型学出并行结构，推理时通过 tool call 执行，AIME24 / AIME25 上**平均约 1.7× 实测 wall-clock 加速**且准确率有竞争力。

**为什么值得看**：test-time scaling 的延迟问题目前多靠系统侧调度解决，这篇把并行结构训进模型再交给运行时执行，是训推协同的路子。
https://arxiv.org/abs/2608.24658

### B9. Selective Regenerative Decoding: Trajectory-Level Intervention for Inference-Time Reasoning
`[Inference]` ｜ Amazon Science 为主（合作：UC Santa Barbara、Netflix、Meta；HTML v1 作者块）｜ v1：2026-08-25

现有 inference-time decoding 把每条候选轨迹当原子对象，要么整条保留要么整条丢弃，浪费了"好前缀 + 坏后缀"的候选。SRD 对每条候选做 discard / keep / refine-suffix 三路路由，只重生成退化的后缀而保留有用前缀，且不需要更大的 target model。在温和假设下相对 rejection sampling 有可证的 **1.28–1.36 倍采样效率增益**，且随候选池增大而增大；MATH500 / GPQA Diamond / HotpotQA / AlpacaEval 上以显著更少的生成 token 匹配 Best-of-N，低算力区间优于 speculative rejection。

**为什么值得看**：把 Best-of-N / speculative rejection 的粒度从轨迹级降到片段级，是 test-time compute 成本曲线上一块没人动过的区域。
https://arxiv.org/abs/2608.24338

### B10. VisCache: Visual KV Cache Pruning for Efficient Vision Large Language Model Inference
`[Inference]` ｜ 香港中文大学（深圳）/ 深圳市大数据研究院 / 深圳国际工业与应用数学中心（HTML v1 作者块）｜ v1：2026-08-25

免训练、即插即用的两阶段视觉 KV 压缩：先用轻量 VLM 过滤时序冗余、只前传语义信息量大的关键帧；再用 PruneKV 做抛物线形的逐层预算分配 + 非对称更新（**只剪 key、对 value 做融合**），避免统一剪枝造成的信息损失。**最多 2.35× 加速**，KV cache 只保留 **19–28%** 时仍保持有竞争力的性能，代码开源。

**为什么值得看**：逐层非均匀预算 + key/value 非对称处理是可以直接搬到现有 KV 压缩实现里的两个改动。
https://arxiv.org/abs/2608.24063

### B11. HAP: Head-Adaptive Visual Token Pruning via Cross-Modal Alignment
`[Inference]` ｜ 上海交通大学（HTML v1 作者块）｜ v1：2026-08-24

把 transformer 层分组并给每组分配视觉 token 预算，组内用 PAQ 加权 softmax 聚合各 head 的 attention map 成组级矩阵，再按幅值打分保留预算内 token——核心是按 PAQ 给 head 加权，避免均匀平均稀释掉真正反映 prompt 相关性的注意力信号。18 个 benchmark 上取得 SOTA 权衡：LLaVA-1.5-7B（9 个任务）**只保留 5.6% token 仍保持 99.1% 原始性能**，比最强基线 AutoPrune 高 4.2 分。

**为什么值得看**：VLM prefill 成本主要由视觉 token 数决定，5.6% 保留率下几乎无损是目前比较激进的点。
https://arxiv.org/abs/2608.23921

### B12. Pipeline-Native Transformers: Co-Designing Model Architecture and CPU Inference for Bandwidth-Efficient Autoregressive Decode
`[Inference]` `[Infra]` ｜ 机构未核实（独作 Tom Poperszky，HTML v1 无单位）｜ v1：2026-08-24

面向 CPU 单 token 自回归解码的带宽瓶颈（约 1 TFLOP/s 算力 vs 约 50 GB/s 内存带宽）做模型与运行时协同设计：cflow 引擎按计算消费顺序把权重存成 L2 大小的 tile、每层 MoE 只读 top-k expert、融合投影；配套的 pipeline-native 架构让层间依赖图允许纵向 stage-major 调度。实测关键路径权重带宽降 **2.00×**（9.00 → 4.50 MB/token）且困惑度距最佳候选 0.24 以内；tile 布局的 L1-data read miss 比 row-major 少 **7.29×**；**30.9B pipeline-native MoE 在 32 vCPU Ice Lake 上 decode 5.94 tok/s**（对比同量级 dense 模型 llama.cpp 4.75、vLLM CPU 后端 1.65）；把 expert 延迟窗口做成磁盘常驻 expert 层的异步 I/O overlap 再净赢**最多 1.68×**。

**为什么值得看**：作者明确写出"八条设计主张中一条被实测证伪、一条不确定"并全文报告；tile 布局与 stage-major 调度对 GPU 侧 offload 场景也有借鉴价值。
https://arxiv.org/abs/2608.23841

---

## C. AI Infra 系统与工程（Systems & Engineering）

### C1. vLLM v0.28.0
`[Infra]` `[Inference]` ｜ vLLM project ｜ 发布：2026-08-26（GitHub API `published_at` = 2026-08-26T09:46:30Z）

**584 commits / 270 contributors（76 名新贡献者）** 的大版本。推理系统侧要点：

- **Kimi-K3 全栈优化**：Decode Context Parallel (#50484)、融合 FlashKDA decode / prefill kernel、MegaMoE 的 SiTU 激活支持、sequence parallelism 的 GEMM-RS、**合并 all-gather 带来 1.5~3× kernel 级加速**（#51070）、**自适应投机 token 预算带来约 60% 更好的 DSpark TTFT**（#51725）、可选 shared-expert sharding **每 GPU 省约 17 GiB**（#50912）；Kimi-K3 亦可在 ROCm 上用 V2 model runner 运行。
- **DeepSeek V4**：sparse MLA 端到端打通 plain decode、MTP 与 DSpark 投机解码（#51538），并加入 AMD Quark NVFP4 支持。
- **Model Runner V2 成熟化**：含 E/P/D 分离（#38390）、权重 offload、多层 MTP KV cache。
- **分层 KV offload**：磁盘 offload、树外 secondary tier manager、分层指标、并行无关的 canonical CPU layout；Mooncake store group 语义 / tenant ID。
- **prefix caching**：细粒度前缀匹配与部分尾部复用。
- **默认值调整**：`max_num_batched_tokens` 8192 → 16384，Mamba 默认开 prefix caching，Blackwell CUDA graph 捕获默认上调到 1024。
- **破坏性变更**：bitsandbytes 移出树外插件；移除 `calculate_kv_scales` 与 `override_attention_dtype`。

**为什么值得看**：E/P/D 分离进入 Model Runner V2 主线 + KV 分层 offload 到磁盘，是这个版本对生产部署形态影响最大的两处。
https://github.com/vllm-project/vllm/releases/tag/v0.28.0

### C2. Qwen3.8-Flash-Next: Day-0 Support in SGLang
`[Infra]` `[Inference]` ｜ SGLang（RadixArk）+ Qwen + NVIDIA + AMD（文末致谢）｜ 发布：2026-08-26

名为 day-0 支持，实为一篇有硬核 kernel / 服务数字的工程文。四块内容：

- **QSA**：压缩比 4，indexer 扫约 L/4 个压缩 key，选 512 个 block 展开到 2048 逻辑位置，最终 sparse attention 最多看 2051 个位置，softmax / value 聚合仍用未压缩 K/V；每 4 token 只加一个 BF16 压缩 index key + 四槽 ring，使 **index-cache 开销降 80%**，page 对齐寻址让压缩 cache 跟随 Radix Cache 生命周期。
- **IndexShare MTP**：draft decode 步完全跳过 indexer，复用 draft-extend 那一次的 top-k 选择，**每次 MTP 迭代的 draft indexer 调用从 N 次降到 1 次**，accept length 不变。
- **HyperConnection kernel**：B300 上 M=4 时 Mix 12.36 → 6.03 µs（2.05×），端到端投机解码吞吐 **+7.6%**；Combine 4.17 → 2.13 µs（1.96×），端到端 **+5.49%**；大 M 下 fused kernel 比 cuBLAS 基线**最快 2.54×**、有效带宽 6144 GB/s。
- **PLE 稀疏 pinned-host offload**：51.2B 嵌入约 95.4 GiB BF16 放宿主内存，每 token 只 gather 16 行，Triton UVA kernel + 独立 CUDA stream 与首个 decoder block 重叠。H200 TP4 + MTP-213 下目标模型权重从 83.91 → **60.45 GiB/GPU（-23.46 GiB）**，同 memory fraction 下 KV 容量 1.84M → **3.28M token（+78.54%）**，1/2/4 并发下吞吐几何均值变化 **-0.07%**，输出 token ID 逐位一致。B200 TP4 下 NVFP4 checkpoint batch=1 + MTP decode **540 tok/s**，accept length 3.3。

**为什么值得看**："indexer 而非 attention 才是长上下文成本主项"这一观察，以及 IndexShare 这个几乎零成本的修法，对任何做 sparse-attention + MTP 的引擎都通用。
https://www.lmsys.org/blog/2026-08-26-qwen-flash-next

### C3. Restore LLM Inference Capacity in Seconds with Shadow Engine Recovery in NVIDIA Dynamo
`[Infra]` `[Inference]` ｜ NVIDIA ｜ 发布：2026-08-25

Dynamo 的 preview 特性：在同一批 GPU 上常驻一个已完全初始化但空闲的 shadow engine，由 **GPU Memory Service（GMS）** 在两个引擎间共享已有权重、**不在 HBM 中复制第二份**（vLLM / SGLang / TensorRT-LLM 各自通过自定义 torch allocator 接入 GMS，vLLM 为主要支持后端）。active 进程失败后 shadow 秒级接管，权重加载、kernel 编译、CUDA graph 捕获都移出服务路径。实测：两 worker 服务 GLM-5.2 NVFP4、B200 单节点单 worker、TP=8、200K 最大上下文、FP8 KV cache，32K 输入 / 1K 输出的合成负载，注入 SIGKILL 后"第二个 worker 重新服务"的时间从冷启动的 **283 s 降到 7.3 s（约 39×）**；冷启动基线的 TTFT p50 跃到 84 ms 并在整个 283 s 内保持高位，shadow 方案只在故障瞬间短暂上升后回落。

**为什么值得看**：推理侧故障恢复长期只有"冷重启 + 上游承压"一条路，GMS 这种把权重从进程生命周期里剥离出来的做法是可迁移的通用机制。
https://developer.nvidia.com/blog/restore-llm-inference-capacity-in-seconds-with-shadow-engine-recovery-in-nvidia-dynamo/

### C4. CUDA Python 1.0: Stable APIs, One Foundation, Full Platform Access
`[Infra]` ｜ NVIDIA ｜ 发布：2026-08-25

CUDA Python 1.0 同批落地：`cuda.core` 1.0（统一的底层对象与流 / 内存语义）、`cuda.compute` 1.0（CCCL 并行算法的 Python 调用）、`cuda.bindings` 13.x（与 CUDA Toolkit 版本对齐的 1:1 C API 绑定）、`cuda-pathfinder`、`nvmath-python` 1.0。1.0 的实质承诺是**语义化版本**，以及把此前各库（PyTorch / CuPy / RAPIDS）各自到达 CUDA 的路径收敛到一个基础层，解决跨库在同一块显存、同一条 stream 上协作需要走交换协议的问题。

**为什么值得看**：写自定义 kernel / 做跨框架零拷贝互操作的人，从此有了带稳定性承诺的底座，对 Triton 之外的 Python-first kernel 工具链是结构性变化。
https://developer.nvidia.com/blog/cuda-python-1-0-stable-apis-one-foundation-full-platform-access/

### C5. Simthesizer: An Agent-Driven Simulation Framework for LLM Serving Systems
`[Infra]` `[Inference]` ｜ KAIST（HTML v1 作者块）｜ v1：2026-08-25

针对"服务系统演进快过模拟器开发"的问题，提出 Borg：一套可组合的模拟器基础设施，把完整服务流程（含控制决策）统一表达为动态图；Synthesizer agent（受约束的 coding agent）在模拟器专属 guardrail 与保真度校验下，把自然语言特性需求 lower 到该抽象上，从而演化**同一个**模拟器而非每个特性造一个新的。同一 coding agent 与 harness 下，基于 Borg 的扩展相对 vLLM 真实系统的**平均吞吐误差 2.51%**，而基于现有模拟器的扩展是 6.03%；同 workload 下 Borg 比 LLMServingSim2.0 **快最多 284.96×**、比 Vidur **快最多 23.19×**。

**为什么值得看**：PD 分离、agentic workload 正在把 Vidur / LLMServingSim 这代模拟器逼到需要侵入式重写，这篇给了一个抽象层 + agent 演化的答案，且误差与速度都有对照数字。
https://arxiv.org/abs/2608.24650

### C6. ShardMeter: Sharded and Geo-Distributed Training Without the Guesswork
`[Infra]` ｜ Technical University of Darmstadt + PanocularAI（abs 页作者标注）｜ v1：2026-08-24

轻量解析式性能模型，给定模型特征与目标硬件拓扑，预测 transformer 训练在任意 sharded / 分布式 / 去中心化配置下的端到端运行时间，输出 per-GPU 与 per-island 训练成本、总 wall-clock 并定位瓶颈。分析给出：island 规模增大时的边际递减区间、compute-bound 与 communication-bound 缩放之间的转换点、超参权衡，以及大规模去中心化训练的成本-吞吐建模。

**为什么值得看**：跨数据中心 / 去中心化训练的配置空间大到无法穷举 benchmark，一个能快速圈定近优部署方案的解析模型是这类工作的刚需前置件。
https://arxiv.org/abs/2608.23840

### C7. Thermal Tuning Overhead in Wafer-Scale Optical Interconnects for LLM MoE Training: A Cross-Layer Analysis and Ferroelectric-Based Mitigation
`[Infra]` `[Hardware]` ｜ Georgia Institute of Technology（HTML v1 作者块，含 Shimeng Yu）｜ v1：2026-08-25

把 workload profiling、包级网络仿真、瞬态热分析串成跨层分析，评估基于 DWDM 的晶圆级光互连在 MoE 专家并行（通信密集）负载下的实际表现。发现 workload 引起的热波动超出传统微环谐振器热光调谐控制环的跟踪能力，在通信阶段反复引入调谐停顿——**停顿时长由热模型直接导出而非假定**，再注入网络仿真。用铁电电光调谐机制去掉持续热调谐需求后，四层代理仿真中三个 MoE 模型分别获得 **2.7×（Mixtral 8x7B）、3.8×（Qwen-MoE 14.3B）、3.3×（LLaMA-MoE 6.7B）** 加速。

**为什么值得看**：光互连常被当作"更高带宽"的免费午餐，这篇量化了热调谐这一被忽略的隐性开销。
https://arxiv.org/abs/2608.24637

### C8. TorchMorph: CUDA-accelerated Morphological Transforms
`[Infra]` ｜ 上海大学（HTML v1 作者块）｜ v1：2026-08-25

轻量 PyTorch 扩展，用融合 CUDA kernel 提供 **22 个算子**（二值形态学、灰度形态学、精确与近似距离变换、熵正则化最优传输），直接作用于 (B, C, Spatial...) CUDA tensor、最多 8 个空间维度，API 对齐 `scipy.ndimage` 以便改 import 即迁移。相对单线程 CPU 参考：灰度形态学批量执行**最高约 1.1e3 倍吞吐**，精确欧氏距离变换**最高 350×**，Sinkhorn 求解器比 POT **快最多 42×**；二值与 chamfer 算子与 SciPy 逐位一致，所有浮点算子与 CPU 参考绝对误差在 1.8e-6 内。MIT 协议开源。

**为什么值得看**：把 mask / 形状处理留在 GPU 上、免掉训练循环里的 device-to-host 往返，属于数据管线上的实际瓶颈消除；正确性对照做得很规整。
https://arxiv.org/abs/2608.24738

### C9. Mooncake v0.3.13
`[Infra]` ｜ kvcache-ai ｜ 发布：2026-08-26

增量特性版本，与 KV 传输 / 存储相关的主要几项：RDMA 层通过 mlx5dv 实现 **UDP source port 维度的 QP 路径分集与 LAG 端口均衡**（#2175）、TENT 的增强 QoS 与 slice spraying（#2048）、structured object store helper（#2140）、`WITH_NVIDIA_PEERMEM` 从 CMake flag 改为运行时环境变量并默认开启、NPU wheel 构建与 release 流程。

**为什么值得看**：PD 分离部署里 KV 传输的尾延迟常受单路径 QP 影响，路径分集 + LAG 均衡是直接相关的改动。⚠️ **本次 release notes 未给任何性能数字**，收益需自行验证。
https://github.com/kvcache-ai/Mooncake/releases/tag/v0.3.13

### C10. WiCi: Wireless GPU Computing Infrastructure
`[Infra]` `[Inference]` ｜ 机构未核实（HTML v1 作者块仅有姓名与联系邮箱）｜ v1：2026-08-25

让移动设备通过 WiFi 无线访问服务器级 GPU：推理任务跑在移动客户端，GPU 相关计算 offload 到附近的 GPU。实测相对移动端本地推理同模型，**TTFT 最多降 90%、token 速率约 39×**，并能支撑大得多的模型；跨应用可达服务器级 GPU 原生性能的**近 80%**。

**为什么值得看**：介于 edge 推理与云推理之间的第三条路，数字给得完整；但缺机构信息，建议结合后续版本再判断。
https://arxiv.org/abs/2608.24204

---

## D. 理论与基础研究（Theory）

### D1. Every Layer Counts: An Exponential $L_2$ Depth Hierarchy for ReLU Networks
`[Theory]` ｜ 机构未核实（单作者 Itay Safran，无 HTML v1）｜ v1：2026-08-24

证明 ReLU 网络的深度层级定理——每增加一层可指数级节省神经元。对每个 $\ell \geq 3$，存在一个全局取值于 $[0,1]$、1-Lipschitz 的函数，可由宽度 $\mathcal{O}(d^4)$ 的深度-$\ell$ 网络实现；而任何深度-$(\ell-1)$、权重无限制且宽度至多 $2^d/[2d(\ell-2)]$ 的网络，在某绝对连续分布下**平方 $L_2$ 误差至少为 $1/24$**。作者称这是首个在两个固定深度（浅者深度 $\geq 3$）之间的指数分离，也是首个跨所有相邻深度的指数层级。

**为什么值得看**：表达能力理论里一个长期悬而未决的缺口被补上，且是 $L_2$ 意义下的强分离而非 sup-norm 取巧。
https://arxiv.org/abs/2608.23877

### D2. Revenge of Monosemanticity: Specialized Neurons Improve Data Efficiency in MLPs
`[Theory]` ｜ 机构未核实（无 HTML v1；作者含 Amirhesam Abedsoltan、Enric Boix-Adsera、Fivos Kalogiannis、Mikhail Belkin）｜ v1：2026-08-25

指出主流特征学习理论"网络学到一个全局低维预测几何"的图景不完整。在聚类数据的回归问题中，MLP 会自然长出**单义（monosemantic）的专用神经元**——单个神经元与输入空间某个局部区域相关的特定预测特征强对齐；网络学到的是一组**局部**低维表示，其并集可张成高维空间。作者证明这种特化相对基于全局低维特征学习的方法带来数据效率优势。

**为什么值得看**：把可解释性里"单义性 vs. 叠加"的经验争论接到了特征学习理论上，并给出单义性**有用**（而非只是副产品）的可证明理由。
https://arxiv.org/abs/2608.24007

### D3. Shortcut Before Circuit: Document Statistics Time In-Context Conflict Resolution
`[Theory]` ｜ 机构未核实（HTML v1 作者块只给邮箱；其中一位为 `@mail.scut.edu.cn`，**推断**为华南理工大学）｜ v1：2026-08-25

针对"上下文中同一事实出现两个冲突取值时模型依据哪条线索"这一问题，在 26M 参数 transformer 上用合成语言训练，使 recency 与 rarity 精确共延，再用一个最小因果编辑把二者分离。**全部 75 个 run 准确率 ≥ 0.999**，held-in 评测无法区分；但在干预下 per-cell readout 不可复现——25 个 cell 中有 13 个的 sign fraction 跨三个 seed 差异超过 0.3，最大者达 0.879（标准误 0.025）。可复现的是**时序**：shortcut 的逃逸有闭式上界且随冗余单调；在逃逸之前探测，**75 个 run 中有 32 个归因符号翻转**（准确率不变）；circuit 形成是归因可用的必要非充分条件。

**为什么值得看**：给"什么时候机制性归因到数据才是有意义的"提供了一个可操作判据，同时是对当前 circuit 类结论可复现性的一记有分量的警告。
https://arxiv.org/abs/2608.24460

### D4. Across the Loss Landscape with Progressive Growth
`[Theory]` `[Pretraining]` ｜ MILES Team, LAMSADE, Université Paris Dauphine-PSL + LORIA CNRS（HTML v1 作者块）｜ v1：2026-08-25

把渐进式"生长-优化"训练策略解释为**渐进约束松弛**，以此分析它为何把训练偏置到更平坦的极小值区域。具体做法是从一个低维子模型出发，通过逐步解锁嵌套随机子空间来扩张可训练参数，同时把正交补冻结在网络初始化处，每次扩张后重新优化，直到恢复完整架构。

**为什么值得看**：把 model growth / 渐进扩容这类越来越常见的工程做法，与 loss landscape 平坦性和泛化的经典理论对上了号。
https://arxiv.org/abs/2608.24568

### D5. Mechanistic Circuit Identification for Controllable Data Generation
`[Theory]` `[Pretraining]` ｜ Seoul National University（HTML v1 作者块）｜ v1：2026-08-25

提出一个 circuit-grounded 框架，把基于训练动力学的数据价值评估与机制可解释性连起来。沿 **learnability、challenge、alignment** 三条互补的效用轴刻画数据质量，先找出因果性地支配这三类效用信号的模型内部专用 circuit，再据此进行可控数据生成，替代当前普遍的启发式 prompt 控制这一黑盒范式。

**为什么值得看**：数据合成流水线目前几乎全靠 prompt 调，这是少见的把"这条数据对模型学习动力学做了什么"落到可干预内部结构上的尝试。
https://arxiv.org/abs/2608.24065

### D6. Discovering Cross-Language Reasoning Invariance in LLMs with Geometry-Invariant Sparse Autoencoders
`[Theory]` ｜ Carleton University（HTML v1 作者块）｜ v1：2026-08-24

研究多语言模型解同一道数学题时，究竟共享特征还是各语言各自计算而只是输出相似。在 5 个模型上使用 MGSM，保留在英/德/法/西/俄/中六种语言下都有有效推理链的题目，回放推理链并记录多层表示；先用 CKA 定位跨语言对齐的层，在该层训练两个 SAE：仅重构的 baseline，以及本文提出的对比变体 **Geometry-Invariant SAE**（在重构损失上加 InfoNCE 项，使同一问题不同语言、不同 token 位置的激活相似）。再通过在前向过程中**跨语言互换**这些共享特征的取值、测量输出变化，来检验其是否功能上可互换。

**为什么值得看**：SAE 特征"是否真的是同一个计算单元"通常只有相关性证据，这里用跨语言互换做了功能性因果检验。
https://arxiv.org/abs/2608.23809

### D7. Beyond Static Interpretability: Anticipating Post-SFT Mechanisms from Pre-SFT Parameters for Better Tuning
`[Theory]` `[Post-training]` ｜ 南洋理工大学（HTML v1 作者块）｜ v1：2026-08-25

指出 "locating-then-tuning"（先用可解释性定位关键参数，再做参数高效 SFT）范式有一个根本缺陷：机制可解释性本质是**回溯性**的，直接解释 SFT 前的模型会得出误导结论。尤其在新任务上，初始定位到的神经元与最终支配模型行为的神经元差别极大，这种偏差会主动干扰 SFT。作者据此提出从 pre-SFT 参数**预测** post-SFT 机制的方法，并称该方法在模型规模增大时保持稳健。

**为什么值得看**：直接质疑了一整类"可解释性指导微调"工作的方法论前提，并给出了偏差存在的实证。
https://arxiv.org/abs/2608.24482

### D8. Generalization, memorization, and overfitting for diffusion models trained in the lazy high-dimensional regime
`[Theory]` ｜ Yale University, Department of Statistics and Data Science（HTML v1 作者块，含 Sinho Chewi）｜ v1：2026-08-25

score-based 生成模型把分布学习归约为一串回归问题，若在有限数据上精确求解最终只会复现训练样本，因此泛化必然来自训练中的隐式/显式正则化。本文为过参数化网络的 supervised lazy-training regime 建立了 **benign overfitting 与算法正则化理论的生成式对应版本**，在高维 lazy régime 下研究 denoising score matching。

**为什么值得看**：扩散模型"为什么不只是记住训练集"至今主要靠经验解释，这是把监督学习那套 benign overfitting 工具正式搬过来的工作。
https://arxiv.org/abs/2608.23938

### D9. Parameter-Level Attribution of Symmetry in Trained Networks Though Parameter-Wise Functional Sensitivity
`[Theory]` ｜ University of Oxford, Mathematical Institute（HTML v1 作者块）｜ v1：2026-08-25

提出问题——当网络学到了一个具有已知对称性的函数，该对称性能否在参数化中被"搬运"，即参数空间是否存在实现该函数空间群作用的运动。形式化为实现映射 $\Phi:\theta\mapsto f_\theta$ 的 lifting 问题，证明光滑参数空间作用存在**当且仅当**函数对称轨道的切空间落在 $\mathrm{d}\Phi_\theta$ 的像中（其列即各参数的 functional sensitivity）；该条件对逐点一阶 lifting 也是充分的，并在最小二乘意义下给出松弛版本。

**为什么值得看**：给"网络的哪些参数负责了哪部分对称性"一个干净的微分几何判据，对 loss landscape 的对称性 / 模式连通性讨论有用。
https://arxiv.org/abs/2608.24700

---

## E. 硬件与算力生态（Hardware & Compute）

> 本节主体为 **Hot Chips 2026（HC38）Day 2** 的技术披露，以及微软 Maia 200 的完整系统论文。

### E1. Maia 200: A Software Defined Dataflow System for Large-scale AI Acceleration
`[Hardware]` `[Infra]` ｜ Microsoft Corporation（HTML v1 作者块；作者含 Torsten Hoefler、Matthew Mattina、Ofer Dekel）｜ arXiv v1：2026-08-25

微软自研 AI 加速器 Maia 200 的完整系统论文。单芯片 **10,145 Tflop/s FP4 / 5,072 Tflop/s FP8，750W TDP（13.3 / 6.7 Tflop/W），7 TiB/s HBM 带宽**。提出 **SDLA（Software Defined Locally Accessed Dataflow Architectures）** 这一类架构——显式编程 dataflow engine 来编排专用存储与数据搬运引擎，从 thread-centric 转向 data-movement-centric；并给出受 Flynn 分类启发的数据管理 taxonomy。**6,144 芯片**的分布式系统**最多提供 62 exaflop/s FP4、43 PiB/s 内存带宽、8.6 PiB/s 以太网网络带宽**。论文原文表述：内部数据显示 Maia 200 相对**微软机队中任何其他 AI 加速器**节省 **30% 成本（TCO）与 15% 能耗**，归因于 AI workload 与架构概念的激进协同设计。

**为什么值得看**：今日最重要的一篇。超大规模自研芯片罕有这种带完整架构论证 + 机队级 TCO / 能耗数字的公开文档。
https://arxiv.org/abs/2608.24664

### E2. OpenAI Jalapeño 自研推理 ASIC
`[Hardware]` `[Inference]` ｜ OpenAI × Broadcom ｜ 披露：2026-08-25（Hot Chips 2026）

OpenAI 首次公开自研推理 ASIC 的完整技术细节。规格：**13.4 PFLOP/s mxfp4×mxfp4 矩阵算力、15.4 TB/s HBM4 带宽、216 GiB、700W 封装**；2,048 芯片系统聚合 27 EFLOP/s、432 TiB，网络拆成本地 128-ASIC 域 600 GB/s、全局 2,048-ASIC 域 200 GB/s（基于 Broadcom Tomahawk6 的半展平两级 Clos）。基准用 SemiAnalysis **InferenceX** 功率归一化对比（Jalapeño 700W vs GB200 1.2kW、GB300/MI355X 1.4kW，对齐 2026 年 7 月 Pareto 前沿）：GPT-OSS 120B 上约 **1.9× 峰值 token/s/kW、1.7× 更低端到端时延**；DeepSeek R1 670B MXFP4 上约 **1.7× / 3.6×**；1T 参数 Kimi K2.5 上约 **1.5× / 3.4×**。架构为 spatial 架构 + Gluon 框架（每个物理 core 即一个 thread block），每个 core slice 配一个 HBM slice 形成快速本地视图，另加低时延专用 collective 网络。工程亮点：从架构概念（2024 末）到 RTL 冻结、2025 末 tapeout，**约 9 个月完成 RTL→tapeout**；用内部 AI 模型 + XLS 硬件描述语言，BF16 乘法器相对人类基线 PPA 提升 **56%**、矩阵单元面积小 **10%**；AI 优化后的 attention / MoE kernel 比专家手写快 **1.5–1.8×**。定位为 Gen 1，已规划 Gen 2 / Gen 3。

**为什么值得看**：模型公司自研推理芯片第一次给出与 GB300 同功率轴的公开对比数字；"9 个月 RTL→tapeout"与 AI 辅助 RTL 的 PPA 数字同样值得关注。
https://www.servethehome.com/openai-jalapeno-asic-at-hot-chips-2026/

### E3. Google TPU v8 家族（TPU 8t 训练 / TPU 8i 推理）
`[Hardware]` `[Infra]` ｜ Google ｜ 披露：2026-08-25（Hot Chips 2026）

第八代 TPU 首次展开。**同一年内同时出两颗芯片**（而非以往训练/推理交替），理由是 MoE 带来的通信瓶颈与 agentic AI 的长上下文需求。8t（训练）6 个 HBM stack、8i（推理）8 个 HBM stack——即**推理每单位算力需要更多 HBM 与更高比例 SRAM**。8i 与 Google 自研 Arm CPU **Axion 以 2:1 配比**成节点（此前用 x86）；采用 **BoardFly 拓扑**（8 托盘 × 4 TPU、共 36 组），**最大 7 跳**（3D Torus 为 16 跳）；in-network collective 放在靠近网络的 ICI I/O die 上处理。8t 侧：superpod **9,600 芯片、聚合 2 PB 共享 HBM、121 EFLOPS FP4**，能效约为 TPU v7 Ironwood 的 **2 倍**；用 OCS 做 9,600 芯片 scale-up 域的动态切片与坏芯片替换恢复；新增专用集群网络 **Virgo**，单域支持 **134K TPU、47 Pbit/s** 带宽；新增 in-field unit testing；**首次对光模块也做液冷**。软件侧现有代码库完全兼容，Pallas 自定义 kernel 语言给出最佳性能。

**为什么值得看**："训练芯片与推理芯片的 HBM/SRAM 配比该不同"这一判断第一次被产品化，且 Virgo 集群网络的单域规模（134K）是当前公开数字里的上限。
https://www.servethehome.com/googles-tpuv8s-for-training-and-inference-at-hot-chips-2026/

### E4. Meta MTIA 400
`[Hardware]` ｜ Meta ｜ 披露：2026-08-25（Hot Chips 2026）

MTIA 路线图为 300 / 400 / 450 / 500 四代。MTIA 300（DLRM 训练芯片）：72 PE + 16 ME、216GB HBM3e、compute die 3nm / IO die 5nm，前反向传播相对 GPU 有 **1.8× 以上加速**。MTIA 400 回到推理定位并扩展到广义推理：多类型 chiplet（compute / SoC / I/O），**硬件 MXFP4**（另支持 MS8 与 16×16 FP8 块格式 MS8S 用于训练）；compute chiplet 为 **8×6 PE 阵列**外加一行冗余 PE，每 PE 内含两颗 RISC-V CPU-P 核与 SFU（单周期 gather 64 元素）；**8 stack HBM3e、9.4 TB/s** 带宽，并引入两类 DLRM embedding cache（含 host embedding cache）；NoC 为带拥塞控制的 2D mesh。系统侧：PCIe Gen6 连主机，**基于以太网的 scale-up fabric 1.2 TB/s**，单托盘 4 卡，**72 ASIC 单 scale-up 域**。单卡 **12 PFLOPS FP4**；相对上一代推理卡 MTIA 200 为 **>15× FP16 算力、46× DRAM 带宽、5× SRAM 带宽**。

**为什么值得看**：代际提升幅度（46× DRAM 带宽）与"以太网做 scale-up fabric"的选择，都与 NVLink 路线形成明确对照。
https://www.servethehome.com/metas-mtia-custom-ai-silicon-at-hot-chips-2026/

### E5. Cerebras CS-4 / WSE-3T 与 Nexus 机架平台
`[Hardware]` `[Infra]` ｜ Cerebras ｜ 披露：2026-08-25（Hot Chips 2026）

Cerebras 首个真正的机架级系统，单机架 3 颗 WSE，运行在单一 scale-up 域内。WSE-3T 相对前代基本全面**翻倍时钟**；完全依赖片上 SRAM，提供 **43,000 TB/s 内存带宽**，晶圆内部 fabric **53 PB/s**。CS-4 相对 CS-3：**2× token 吞吐、10× tokens/watt**。Nexus 平台把电源放前、算力以可插拔 "backpack" 置后，backpack 同时承担 I/O、供电与液冷；相对 CS-3 提供 **2× 供电与散热能力、组件数少 50%**；每 backpack 最多 30 个 PSU 模块、输入最高 277V AC。互连：backpack 间直连晶圆链路，**晶圆间时延低至 2 μs、每晶圆聚合 2.4 Tb/s**。路线图：CS-5（2027，宣称最高 **3M tokens/MW** 或单用户最高 10,000 tokens/s）、**CS-6 将首次在 WSE 上堆叠 DRAM**，用堆叠 DRAM 换出 SRAM 面积以塞入更多算力。已在推进 WSE 与 AMD Instinct MI455X 的混合部署。

**为什么值得看**：CS-6 在晶圆上堆 DRAM 是晶圆级路线的一次范式转向——承认纯 SRAM 容量走不下去。
https://www.servethehome.com/cerebras-talks-going-rack-scale-with-their-wses-at-hot-chips-2026/

### E6. SambaNova SN50 RDU
`[Hardware]` `[Inference]` ｜ SambaNova ｜ 披露：2026-08-25（Hot Chips 2026）

第五代可重构数据流单元。论点建立在 **MBU（Model Bandwidth Utilization）** 而非裸 HBM 带宽利用率上：agentic 推理时间绝大部分在 decode（DeepSeek V3 上 **97% decode / 3% prefill**），而 decode 是带宽受限；GPU 的 MBU 随卡数增加显著下滑，模型带宽约 **30 TB/s 见顶**。SN50 相对 SN40 提供 **5× FLOPS**，scale-up 域可达 **256+ 芯片**；封装只有两颗 max reticle 逻辑 die + HBM，无 I/O die，但 HBM 仍用**较旧的 HBM2e**（文中点出这是未来供应风险）。MoE 的 token dispatch 给出 all-to-all 与 broadcast+filter 两条路径。实测：GEMM 在 32 socket 下利用率**持续 ≥70%**；64-socket DeepSeek MoE 专家加载**接近 80%** 带宽利用率；**256 卡时 MBU 仍保持 45%**，512 卡聚合模型带宽 **>350 TB/s**、MBU 仍在 40% 区间。部署形态为异构解耦：NVIDIA H200 做 prefill/midfill + SN50 做 decode，经 RoCE 传输；Artificial Analysis 实测 MiniMax M2.7 上 **>750 tokens/s**。

**为什么值得看**：把"GPU 集群规模一大 MBU 就塌"这件事量化，并给出 H200 prefill + SN50 decode 的异构 PD 分离生产形态。
https://www.servethehome.com/sambanovas-sn50-rdu-for-ai-at-hot-chips-2026/

### E7. d-Matrix Raptor：3D DRAM 堆叠推理加速器
`[Hardware]` ｜ d-Matrix ｜ 披露：2026-08-26（Hot Chips 2026）

号称首个面向生成式推理的 3D DRAM 加速器：TSMC N4P 计算 die 以 **36 μm pitch 面对面键合**在定制 DRAM die 之上，**单卡 32GB 提供 100 TB/s 带宽**。反转常规堆叠顺序——逻辑 die 在上（冷板直接压在计算硅上），DRAM 在下并兼作 interposer。垂直接口能耗 **0.37 pJ/bit**（对比进 HBM4 base die 约 2.4 pJ/bit），CTO 称是来自可工作硅片的"实测值"；配套 ISCA 2026 论文（与 UBC 合作）**预测**单卡吞吐约为 HBM 方案的 **4.7 倍**。工程细节：单层堆叠功率密度约 0.5 W/mm²，DRAM 按 105°C 结温设计（保持时间从 32ms 塌到 4ms，刷新频率 ×8），通过把 microbank 缩到 1,366 行 / 约 5.33MB 把全刷新代价压到总带宽的 **1.37%**；每 chiplet **840 bank**，其中约 9% 为备用；逻辑 die 上用 **[132,128] Reed-Solomon** + CRC；接口无 PHY / 无 burst / 无 sideband，改用逐 flit 与前一 flit 比较的 1-bit 反转标记，找回约 20% 的 I/O 功耗。

> ⚠️ **需要注意的限定**：垂直接口满负荷仍吃掉 422W 封装预算中的 **296W**；die 密度 11.4 MB/mm² 只有 HBM4 的 21.9–26.3 的约一半；32GB/卡对比 HBM4 卡的 192–288GB。**所有性能数字（含 2.8T 参数 Kimi K3 上 988 tokens/s/user @1M 上下文）均为 d-Matrix 基于早期硅片的推算，无第三方实测**；且 DRAM 代工方至今未披露。

**为什么值得看**：0.37 pJ/bit 的垂直接口能耗若属实，是对 HBM 能效模型的实质挑战；但整包数字目前只有厂商推算。
https://www.tomshardware.com/tech-industry/semiconductors/d-matrix-stacks-its-ai-accelerator-directly-on-custom-dram-for-100-tbs-per-card

### E8. NVIDIA NVLink Fusion Brings NVHBM to Next-Generation AI Infrastructure
`[Hardware]` ｜ NVIDIA ｜ 发布：2026-08-26

NVLink Fusion 向定制 XPU / CPU 客户开放经内存厂商验证的 **NVHBM base die**。相对标准 HBM4 给出的三项平台级指标：**内存带宽最多提升 30%**、更高效的接口连接使**可用计算 die 面积最多增加 25%**、**HBM 功耗最多降低 15%**。

> ⚠️ **这是一篇厂商产品发布稿，无可复现方法**，收录仅因为它给出了明确的 NVHBM 规格数字，且直接关系到自研加速器阵营（含 E1 的 Maia 200 这类项目）未来的显存带宽上限。
https://developer.nvidia.com/blog/nvidia-nvlink-fusion-brings-nvhbm-to-next-generation-ai-infrastructure/

### E9. Micron：HBM 相对 DDR5 的硅片惩罚正在逐代拉大
`[Hardware]` ｜ Micron ｜ **讲座 2026-08-23（窗口外）/ 报道 2026-08-25（窗口内）**

Micron HBM 设计架构 Fellow 给出的量化：同容量下 HBM 约占 **DDR5 三倍的晶圆面积**，且"definitely not getting better"。原因是 HBM4 单 die **256 bank**（DDR5 为 32），靠并行拿带宽因而需要更多数据通路、供电与 TSV 面积；单颗 HBM3E die 可喂 256 GB/s，单颗 DDR5 die 约 8 GB/s。双 GPU 封装中**内存约占硅面积 90%，约为 GPU die 面积的 8 倍**；HBM 每 bit 售价约为 DDR5 的 **5 倍**。趋势数字：算力约每两年 3×，HBM 带宽不到 2×。HBM4 主机接口翻倍到 **2,048 I/O**，Micron 产品跑到 **>11 Gb/s/pin、单 stack >2.8 TB/s**（超过 JEDEC HBM4 的 8 Gb/s / 2 TB/s 基线）。层数上"到 16 层有清晰路径，16 层之后还有大量工作"。可靠性佐证：引用 Meta Llama 3 论文——16,384 张 H100、54 天训练中 **17.2% 的非预期中断归因于 HBM3**。

> ⚠️ **日期口径**：Hot Chips 讲座在 08-23（窗口外），首篇公开报道在 08-25（窗口内）。按"首次公开可读文档"口径收录，读者按"披露日期"口径可自行剔除。

**为什么值得看**：memory wall 的量化版本，直接解释了为什么 E1/E7 这类"堆 DRAM / 换接口"的路线在同时出现。
https://www.tomshardware.com/tech-industry/semiconductors/micron-says-the-silicon-gap-between-hbm-and-ddr5-is-widening-with-every-generation

### E10. 国产 AI 芯片集群上的前沿模型规模化推理
`[Infra]` `[Hardware]` ｜ 智谱 / Z.ai ｜ 发布：2026-08-26

与「重点发布」第 1 条 GLM-5.3-Flash 同源，此处仅列算力生态相关部分。官方称过去一周在**大规模国产 AI 芯片集群**上服务 GLM-5.3-Flash，并在 SGLang 之上自建专用推理引擎。针对国产芯片显存容量与带宽受限（尤其 1M 上下文），栈内组合了：Linear Attention 与 LM head 的节点内张量并行、**ReplaySSM**、W8A8 量化、**INT8/FP8/BF16 混合 cache 量化**、Layer Split，以及"以算换带宽 / 以通信换带宽"技巧。集群级采用生产级 **Encode–Prefill–Decode (EPD) 解耦架构**，把多模态编码、prefill、decode 拆成独立调度的 worker pool，跨"数以万计"的国产加速器服务。原文表述：**相对同硬件上的初始基线取得 3× 端到端服务性能提升**，达到与主流 NVIDIA GPU "comparable" 的硬件效率与每 token 成本。另披露该优化过程由其 GLM-5.3 驱动的 infra agent 协助完成 kernel 开发与瓶颈诊断。

**为什么值得看**：EPD 三段解耦 + 混合精度 KV 是国产卡上跑 1M 上下文的一套完整公开配方；"模型帮忙优化服务自己的系统"这个反馈回路也第一次被官方点名。
https://z.ai/blog/glm-5.3-flash

---

## 一句话总结今日趋势

今天的主线是**"混合稀疏/线性注意力架构从论文进入旗舰产品"与"自研加速器集体亮牌"这两件事撞在了同一天**——GLM-5.3-Flash 与 Qwen3.8-Flash-Next 同日把稀疏 + 线性混合注意力做进 320B / 125B 级模型并公布 3.0×~4.4× 的算力与 KV 收益，SGLang 的 day-0 工程文顺势指出"长上下文的成本主项已经从 attention 转移到 indexer"，而 Hot Chips 与 Maia 200 论文则从另一端给出同一个约束的硬件答案：内存带宽与容量（而非算力）才是这一代推理系统的定价因子。

---

## 本期未能证实的传闻（不予收录）

| 传闻 | 核实状态 |
|---|---|
| EchoWM 在 WBench Navigation 取得 81.7 平均分、EchoWM-Flash 交互分 87.9、SANA-WM-Bench 上 VBench Overall 最高 | **未证实**。数字仅出现在搜索引擎生成的摘要中，arXiv 摘要与 HTML 首屏均无此表述。 |
| Apodex 1.1 相对具体前沿模型的逐项 benchmark 分数 | **未能完整核实**。官方博客与 arXiv 摘要只给"leading performance band"这类限定表述。本报告仅引用可核实的 AI4AI 数字（51.0% → 56.0%）与 35B 参数量。 |
| NVIDIA 将在年底前向中国出货基于 Groq 的 LPU | **已被否认**（Tom's Hardware 标题为 "Nvidia denies report..."），且日期未落在窗口内。 |
| NVIDIA 正在测试仅配 192GB 的 Rubin Ultra 配置，因 HBM4E 供应不足 | **未证实**，为 d-Matrix 报道中引用的 "reportedly"，无一手来源。 |
| Samsung 为 OpenAI Jalapeño 供应 HBM4 | **未证实**。仅见于 TrendForce 转述；OpenAI 在 Hot Chips 上未点名 HBM 供应商，仅致谢 Broadcom 与 Celestica。 |
| d-Matrix Raptor 的定制 DRAM 由谁代工 | **明确未披露**。已具名 TSMC（N4P 逻辑 die）与 Alchip（ASIC 设计 / 封装），两者均无 DRAM fab。 |

---

## 引用来源

### 官方发布与博客
- [GLM-5.3-Flash: Frontier Intelligence, Flash Cost（Z.ai, 2026-08-26）](https://z.ai/blog/glm-5.3-flash)
- [zai-org/GLM-5.3-Flash（HuggingFace）](https://huggingface.co/zai-org/GLM-5.3-Flash)
- [Qwen3.8-Flash-Next：全新架构，迈向极致性价比（Qwen, 2026-08-26）](https://qwen.ai/blog?id=qwen3.8-flash-next)
- [Qwen/Qwen3.8-Flash-Next（HuggingFace）](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- [Restore LLM Inference Capacity in Seconds with Shadow Engine Recovery in NVIDIA Dynamo（NVIDIA, 2026-08-25）](https://developer.nvidia.com/blog/restore-llm-inference-capacity-in-seconds-with-shadow-engine-recovery-in-nvidia-dynamo/)
- [CUDA Python 1.0: Stable APIs, One Foundation, Full Platform Access（NVIDIA, 2026-08-25）](https://developer.nvidia.com/blog/cuda-python-1-0-stable-apis-one-foundation-full-platform-access/)
- [NVIDIA NVLink Fusion Brings NVHBM to Next-Generation AI Infrastructure（NVIDIA, 2026-08-26）](https://developer.nvidia.com/blog/nvidia-nvlink-fusion-brings-nvhbm-to-next-generation-ai-infrastructure/)
- [Apodex 1.1 官方技术报告页](https://www.apodex.com/blog/apodex-1.1-scaling-agentic-intelligence-for-complex-work)

### 论文（arXiv，均已核实 v1 日期）
- [2608.23922 Data Mixing as Mixture Experiment](https://arxiv.org/abs/2608.23922)
- [2608.24814 Effective Learning Rate Governs Loss Dynamics in Language Model Pretraining](https://arxiv.org/abs/2608.24814)
- [2608.22876 The Mask Is Not the Model](https://arxiv.org/abs/2608.22876)
- [2608.24593 Delayed Optimizer-State Transport Shapes Short-Horizon Training Decisions](https://arxiv.org/abs/2608.24593)
- [2608.24845 LAION-BVD](https://arxiv.org/abs/2608.24845)
- [2608.23806 Giga-Embeddings](https://arxiv.org/abs/2608.23806)
- [2608.23392 Towards a Densing Law for User Representation Learning at Billion-Scale Capacity](https://arxiv.org/abs/2608.23392)
- [2608.24053 WeMM-Embedding](https://arxiv.org/abs/2608.24053)
- [2608.24696 On-policy Distillation with Verifiable Reward](https://arxiv.org/abs/2608.24696)
- [2608.23566 Best Practice Critic Optimization](https://arxiv.org/abs/2608.23566)
- [2608.23311 Beyond the Stability-Exploration Dilemma](https://arxiv.org/abs/2608.23311)
- [2608.24870 SPO++](https://arxiv.org/abs/2608.24870)
- [2608.24588 IAPO](https://arxiv.org/abs/2608.24588)
- [2608.24300 Contrastive Branch Policy Optimization](https://arxiv.org/abs/2608.24300)
- [2608.24310 OPDSearch+](https://arxiv.org/abs/2608.24310)
- [2608.24646 On-Policy Self-Distillation in Diffusion Models](https://arxiv.org/abs/2608.24646)
- [2608.24192 Preference Data Selection for Mitigating the Alignment Tax](https://arxiv.org/abs/2608.24192)
- [2608.24135 Robust Code RL via Faulty-Code-Driven Test case Synthesis](https://arxiv.org/abs/2608.24135)
- [2608.24174 Task-Adaptive Rubrics for GUI Reward Modeling](https://arxiv.org/abs/2608.24174)
- [2608.23830 Mitigating Exploration Bias in RL for Multi-Instruction Following](https://arxiv.org/abs/2608.23830)
- [2608.24747 SkillForge](https://arxiv.org/abs/2608.24747)
- [2608.24848 BrowserForge](https://arxiv.org/abs/2608.24848)
- [2608.23911 PROOF-Gen](https://arxiv.org/abs/2608.23911)
- [2608.22817 Industrial-Instruction](https://arxiv.org/abs/2608.22817)
- [2608.23807 Serving Masked Diffusion LLMs](https://arxiv.org/abs/2608.23807)
- [2608.23658 Elastic KV Cache for LLM Serving](https://arxiv.org/abs/2608.23658)
- [2608.23962 More GPUs or a Smaller Cache?](https://arxiv.org/abs/2608.23962)
- [2608.23834 Minima-KV](https://arxiv.org/abs/2608.23834)
- [2608.23843 PuzzleKV](https://arxiv.org/abs/2608.23843)
- [2608.24004 AgentSpec](https://arxiv.org/abs/2608.24004)
- [2608.24411 ResiSpec](https://arxiv.org/abs/2608.24411)
- [2608.24658 Parason](https://arxiv.org/abs/2608.24658)
- [2608.24338 Selective Regenerative Decoding](https://arxiv.org/abs/2608.24338)
- [2608.24063 VisCache](https://arxiv.org/abs/2608.24063)
- [2608.23921 HAP](https://arxiv.org/abs/2608.23921)
- [2608.23841 Pipeline-Native Transformers](https://arxiv.org/abs/2608.23841)
- [2608.24650 Simthesizer](https://arxiv.org/abs/2608.24650)
- [2608.23840 ShardMeter](https://arxiv.org/abs/2608.23840)
- [2608.24637 Thermal Tuning Overhead in Wafer-Scale Optical Interconnects](https://arxiv.org/abs/2608.24637)
- [2608.24738 TorchMorph](https://arxiv.org/abs/2608.24738)
- [2608.24204 WiCi](https://arxiv.org/abs/2608.24204)
- [2608.23877 Every Layer Counts](https://arxiv.org/abs/2608.23877)
- [2608.24007 Revenge of Monosemanticity](https://arxiv.org/abs/2608.24007)
- [2608.24460 Shortcut Before Circuit](https://arxiv.org/abs/2608.24460)
- [2608.24568 Across the Loss Landscape with Progressive Growth](https://arxiv.org/abs/2608.24568)
- [2608.24065 Mechanistic Circuit Identification for Controllable Data Generation](https://arxiv.org/abs/2608.24065)
- [2608.23809 Discovering Cross-Language Reasoning Invariance with Geometry-Invariant SAE](https://arxiv.org/abs/2608.23809)
- [2608.24482 Beyond Static Interpretability](https://arxiv.org/abs/2608.24482)
- [2608.23938 Generalization, memorization, and overfitting for diffusion models](https://arxiv.org/abs/2608.23938)
- [2608.24700 Parameter-Level Attribution of Symmetry in Trained Networks](https://arxiv.org/abs/2608.24700)
- [2608.24664 Maia 200](https://arxiv.org/abs/2608.24664)
- [2608.23283 Apodex 1.1](https://arxiv.org/abs/2608.23283)
- [2608.23552 Prime Agent](https://arxiv.org/abs/2608.23552)
- [2608.23189 EchoWM](https://arxiv.org/abs/2608.23189)

### 框架与工程
- [vLLM v0.28.0 release notes（2026-08-26）](https://github.com/vllm-project/vllm/releases/tag/v0.28.0)
- [Qwen3.8-Flash-Next: Day-0 Support in SGLang（LMSYS / RadixArk, 2026-08-26）](https://www.lmsys.org/blog/2026-08-26-qwen-flash-next)
- [Mooncake v0.3.13（2026-08-26）](https://github.com/kvcache-ai/Mooncake/releases/tag/v0.3.13)
- [PrimeIntellect-ai/prime-agent](https://github.com/PrimeIntellect-ai/prime-agent)
- [ApodexAI/FrontierAgent](https://github.com/ApodexAI/FrontierAgent)
- [jd-opensource/JoyAI-Echo](https://github.com/jd-opensource/JoyAI-Echo)
- [tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B)

### 硬件报道（Hot Chips 2026）
- [OpenAI Jalapeño ASIC at Hot Chips 2026（ServeTheHome）](https://www.servethehome.com/openai-jalapeno-asic-at-hot-chips-2026/)
- [Google's TPUv8s for Training and Inference at Hot Chips 2026（ServeTheHome）](https://www.servethehome.com/googles-tpuv8s-for-training-and-inference-at-hot-chips-2026/)
- [Meta's MTIA Custom AI Silicon at Hot Chips 2026（ServeTheHome）](https://www.servethehome.com/metas-mtia-custom-ai-silicon-at-hot-chips-2026/)
- [Cerebras Talks Going Rack Scale with their WSEs at Hot Chips 2026（ServeTheHome）](https://www.servethehome.com/cerebras-talks-going-rack-scale-with-their-wses-at-hot-chips-2026/)
- [SambaNova's SN50 RDU for AI at Hot Chips 2026（ServeTheHome）](https://www.servethehome.com/sambanovas-sn50-rdu-for-ai-at-hot-chips-2026/)
- [d-Matrix stacks its AI accelerator directly on custom DRAM（Tom's Hardware）](https://www.tomshardware.com/tech-industry/semiconductors/d-matrix-stacks-its-ai-accelerator-directly-on-custom-dram-for-100-tbs-per-card)
- [Micron says the silicon gap between HBM and DDR5 is widening（Tom's Hardware）](https://www.tomshardware.com/tech-industry/semiconductors/micron-says-the-silicon-gap-between-hbm-and-ddr5-is-widening-with-every-generation)
- [OpenAI says its Jalapeño chip beats Nvidia's GB300 in first published benchmarks（Tom's Hardware）](https://www.tomshardware.com/tech-industry/semiconductors/openai-says-its-jalapeno-chip-beats-nvidias-gb300-in-first-published-benchmarks)

### 已核查但无窗口内更新的索引页
- [arXiv cs.LG new listings（批次：Wednesday, 26 August 2026）](https://arxiv.org/list/cs.LG/new)
- [arXiv cs.DC new listings](https://arxiv.org/list/cs.DC/new)
- [arXiv cs.AR new listings](https://arxiv.org/list/cs.AR/new)
- [arXiv cs.PF new listings](https://arxiv.org/list/cs.PF/new)
- [HuggingFace Daily Papers 2026-08-26](https://huggingface.co/papers/date/2026-08-26)
- SGLang / Megatron-LM / DeepSpeed / TensorRT-LLM / JAX / TransformerEngine / FlashAttention / Ray：最新 release 均早于 2026-08-25，或为无实质内容的 rc / patch bump，本期不收。
- Mistral / Ai2(AllenAI) / Meta / DeepSeek / Moonshot / MiniMax / ByteDance-Seed / 百度 / xAI / Microsoft / OpenAI 的 HuggingFace 组织仓库：`createdAt` 均无 2026-08-25 之后的新模型。
