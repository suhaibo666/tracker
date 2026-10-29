# LLM 技术动态日报 · 2026-10-29

> 时区：Asia/Shanghai（UTC+8）。增量扫描窗口：**2026-10-28 09:02:09 → 2026-10-29 09:02:28**；72小时回溯：**自2026-10-25 09:02:09起**。实际生成时间：**2026-10-29T09:29:14+08:00**。有UTC精确时间的来源按原值保留；仅日期的公告不推定具体时刻。
>
> 已读取自动化memory、持久化state/sources、22篇本地历史日报及认证远端main的索引和递归树。原有日报与索引23个Git blob均与本地吻合，归档读取基线为`32f138a8ce4a62c0460c07a35390254a4c61c55a`；本轮无旧文件待补传。只维护llm-monitor与tracker/llm-training-daily，未改知识库wiki。
>
> 性能与评测均区分作者报告、公开代码/数据证据和本轮静态检查。**本轮未做模型训练、GPU/NPU/TPU性能或完整数据污染复现。** 新作、修订、权重交付和代码合入各按自身时间记账；来源清单枚举完成不代表全文/依赖/运行语义覆盖完成。

## 本次重要变化

- **先评估升级契约，再测速度。** PyTorch2.15正式发布，默认NCCL实现、FSDP2梯度dtype和DCP格式都有迁移影响；TorchTitan0.4.0也已发布。两者同日main的新接口与算子修复分别记账，不能直接算进发行包。
- **预训练研究开始更细分优化器的稳定与效率。** MuonGS的token等效收益、EnergyCal的spike抑制有可复现的小规模证据，但不等于等墙钟/大规模训练已获益；ZeRO-Prefill修订补出了物理通信和常驻slot成本。
- **训推协同重点看真实存储和概率契约。** DeepSpeed从活跃ZeRO存储规划同步，默认副本策略可能更慢；TRL修min_p候选集、LoRA请求版本与零advantage过滤，停机和统计边界仍须验证。Megatron混dtype归约、无padding分片及MoK接入各有明确限制。
- **低精度与缓存正确性值得优先回归。** FA4 beta37交付此前尾部NaN/TMA修复；FlashInfer的PDL标量时序、mHC跨长度工作区、GQA页表ring与W4A16范围修复均可静默影响结果。KDA KCP首步峰值下降应按workspace与per-call分账。
- **模型/数据的新证据主要来自报告与固定工件。** Nemotron3.5 Transcribe披露SFT/GSPO与离线边界，EmbeddingGemma2补训练目标消融；LemonDecider v3换权重与8K分组契约，GLM NVFP4和WeVisOne则是工件修订。Granary公开标签量不等于全部处理音频，演示评测集也不等于私有正式集开放。
- **服务优化要用自身负载决定是否采用。** Helix投影前移缩通信但增权重显存并重排数值，Lamport小步通信增加workspace，MLX稀疏prefill和Agent session调度的收益随batch/上下文变化；均未把局部kernel倍数外推到整模SLO。

目录：[模型与技术报告](#models) · [预训练与架构](#pretraining) · [后训练](#posttraining) · [推理与系统](#inference) · [数据工件](#data) · [核心开源](#core) · [扩展工程](#extended) · [硬件实践](#hardware) · [评估与可靠性](#evaluation) · [重要修订](#revisions) · [覆盖与缺口](#coverage)

<a id="models"></a>

## 模型与技术报告

### Nemotron 3.5 Transcribe：把说话人信息注入语音表示，公开报告区分开源集成与优化服务

**NVIDIA；官网日期10-28，主窗HF于13:44 UTC新增报告链接；BF16权重补录：首个可见含权重提交为10-27 12:22 UTC，当前固定`9d8cd6b7`。** [官方说明](https://research.nvidia.com/labs/nemotron/NVIDIA-Nemotron-3.5-Transcribe/)、[67页报告](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Transcribe-Technical-Report.pdf)、[固定工件](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B/tree/9d8cd6b79bfdd427b6558a5fdbf76fcaf8c75341)。模型将0.6B Transformer ASR encoder、冻结Sortformer说话人分离器接到Nemotron3.5 Lightning混合Mamba–Transformer MoE；说话人活动经固定sinusoidal kernel加到声学表示，LLM学习序列化说话人标签，生成后用约50M CTC adapter对齐词时间戳。整包约34B、激活约4.3B，40种转写语言、最长训练录音1小时，**针对离线转写，无原生流式转写**。

报告§3披露按答案token质量平衡训练混合，256×H100、28,500步SFT后，以20,000筛选录音做400步GSPO；RL只更新约1B参数，MoE/路由/MTP等冻结。§4/Table6在九个公开英语ASR测试集的非加权均值WER为**4.71%**，对公开Qwen3-ASR-1.7B为4.94%，但不同系统提示与推理设置并非完全统一；长录音结果含重复输出后处理。§4.13的**2,238秒音频/秒**是单RTX PRO6000 Blackwell、30秒语音、NVFP4+DFlash2、最高并发256的NIM优化服务；报告明确这些优化尚未进入开源发布，发布日开发集成预期约慢2倍。量化精度评估也显示NVFP4的多语言退化高于FP8，不能只报英语均值。

标OpenMDW1.1；BF16与DFlash2工件可核，FP8/NVFP4/GGUF API本轮均401，**不能据官网列表确认所有量化权重已公开可取**。未下载整包、未复现训练/榜单；完整语料、总训练token及耗时仍不足。建议深入读训练混合与检查点恢复配方，并先复测中文远场、多说话人、循环输出、时间戳和实际服务版本。

稳定ID：`nvidia:nemotron35-transcribe:technical-report:2026-10-28`；`hf:nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B@9d8cd6b79bfdd427b6558a5fdbf76fcaf8c75341`。

### EmbeddingGemma 2补充训练报告：先蒸馏跨模态空间，再用对比学习摆脱教师空间限制

**Google DeepMind；报告`2610.37364v1`于10-27 15:00:42 UTC提交，属72小时回溯；HF于主窗10-28 02:47 UTC新增报告链接，旧模型不是本日新权重。** [报告全文](https://arxiv.org/html/2610.37364v1)、[固定模型卡/配置](https://huggingface.co/google/embeddinggemma-2/tree/f9972ac22e696a105c482202646979e33e4b5a1e)。新增说明了约**2.9T训练tokens**：300B文本预微调、2T多模态预微调、600B多模态微调。前两阶段主要/完全蒸馏Gemini Embedding2，末阶段换为对比学习、spread-out与MRL；冻结音频encoder、联合训练视觉encoder，最终平均同一训练轨迹的早晚两个检查点。PLE保留主embedding投影与门控，移除独立查表，避免为非文本增加无直接信号的参数。

Table3用**中间配方、近似等计算预算**比较：继续蒸馏的MMEB-v2总体52.28，换微调目标后55.46；这不是最终发布模型的完整消融。Table4视觉encoder训练/冻结为52.27/48.18。配置实见24层、512宽、5个滑窗层配1个全局层；744,371,512个BF16存储参数，模块化vision/audio可独立不加载。已在Transformers固定`144ecf00`核`EmbeddingGemma2TextModel.forward`的双向mask、PLE投影，以及`EmbeddingGemma2Model.__init__`的可选encoder构造；没有重跑训练。

Table1量化结果对视觉/音频明确使用**不同harness**，不能把float与QAT差全归因于量化；端侧191/447/567MB只对应Pixel11 Pro与Q4_C的特定测量。训练数据明细、采样率、绝对学习率/批大小和总算力未完整公开，配置的262144位置上限也不能替代发布支持的8192共享输入预算。Apache2.0。建议复现目标切换的消融，并检查任务前缀、模态token预算、MRL截断后的归一化及实际检索数据。

稳定ID：`arxiv:2610.37364v1`；`hf:google/embeddinggemma-2:technical-report-link@26711154b6`。

### LemonDecider v3换权重与服务契约：联合问题读出，容量回到8K，超长列表先完整规划再分组

**inclusionAI；10-28 02:06–05:15 UTC工件更新，固定`81556cfa`，相对`9a2f41d5`的两个权重分片LFS标识均改变。** [固定卡/配置/代码](https://huggingface.co/inclusionAI/LemonDecider/tree/81556cfa823e315d7ed3e027e44905c4cce45398)。基于Qwen3.5-4B的typed decision模型从v2独立问题路径改为共享state的native joint-question readout；输入合适时一次前向，超长问题列表递归二分，每组保留全部state与选项。`code/serve_312k.py::plan/predict`先规划所有组，单问题仍超限即整体422，避免先返回部分答案；**每组容量8192，不能沿用旧卡256K或把多组都叫单次前向**。

作者披露1,199,692训练行、同一SFT run四检查点FP32平均，再用1,448条留出校准标量温度；16个accelerators、4,690步/一epoch，未公开完整语料与精确训练硬件。当前Decision Index公共0.3自评60.03，140,178可评分请求中30项unsupported保留在覆盖率与评分分母中；53.3ms中位延迟是四PPU-ZW810E workers的HTTP墙钟，不是单NVIDIA卡控制基准。结果数据集未在模型卡给出可核URL，命名整理后未重跑GPU输出一致性，不能当独立私测榜分。权重标Apache2.0、部分代码保留MIT，数据另有来源条款。建议复现分组、失败原子性、概率校准与自有分布，而非只对比不同硬件的延迟。

稳定ID：`hf:inclusionAI/LemonDecider@81556cfa823e315d7ed3e027e44905c4cce45398`。

### GLM-5.3-Flash NVFP4更新全部33个分片：应固定新量化工件，旧新分数不能直接当质量提升

**NVIDIA/Z.ai；10-28 06:23 UTC更新，固定`1d007386`；9月已有模型的量化checkpoint修订。** [固定工件/卡](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/tree/1d00738646cbb0c6d0a9eb403dac5f366f1473a7)。相对`da920bb0`，树中33个safetensors分片与index均改变，支持确有权重内容更新；没有下载张量比较。发布配置是group16 NVFP4，并显式排除attention、MoE router/shared experts、embedding/head等，卡正文称shared experts也被量化，与`config.ignore`排除`shared_experts`存在明确冲突，本报以配置边界为准；不能读成全模型所有参数4bit。

当前表在GB200、temperature1/top-p0.95、一般327680最大输出token（Terminal-Bench不封顶）下，BF16/NVFP4的GPQA为92.17/91.89、Terminal-Bench2.1为82.58/82.02，另有项目量化略高。**本次连BF16基线列也改了**，因此不能把旧表→新表差值归因于新量化权重；缺重跑方差与匹配日志，没有吞吐基准。MIT。适合已有Blackwell部署者固定新SHA后复测数值质量、显存与推理后端，尚不足以宣布普遍精度无损或新架构。

稳定ID：`hf:nvidia/GLM-5.3-Flash-NVFP4@1d00738646cbb0c6d0a9eb403dac5f366f1473a7`。

### WeVisOne-VL 2B存储精度更正：626个张量从FP16换为BF16

**腾讯微信视觉；10-28 13:18 UTC，`fcf8c8ca`；10-23模型的工件修订。** [固定工件](https://huggingface.co/tencent/WeVisOne-VL-2B/tree/fcf8c8ca7fa481b514005ef591c8b72bed39618a)。对比`f9fda870`只变`model.safetensors`；本轮用HTTP Range独立读取两个版本的safetensors头，确认626个张量由全F16变成全BF16，参数形状与总数不变，解决此前“存储FP16、示例BF16”的疑点。模型卡、配置和核心`mask_decoder.pt`没有随此提交改变；没有新训练或新质量分数证据，转换本身不能保证精度提升。Apache2.0及第三方原条款。建议既有用户固定新版并复测密集感知/像素decoder输出；本轮仅核头部，未加载模型或比较转换数值。

稳定ID：`hf:tencent/WeVisOne-VL-2B@fcf8c8ca7fa481b514005ef591c8b72bed39618a`。

### Granite-Speech 5.0新增浏览器ONNX工件：量化图从旧结构迁移，旋转位置缓存按5.0重建

**IBM Granite；10-28 21:56 UTC，`a9cf1d13`。** [固定卡/工件](https://huggingface.co/ibm-granite/granite-speech-5.0-2b-ONNX/tree/a9cf1d13d045d65813b4bc80bde8fdd62db4c35c)。这是旧5.0 ASR的Transformers.js/WebGPU交付，三个ONNX图采用block32非对称4bit：encoder激活FP32，embedding/decoder为FP16；按5.0权重重新做BatchNorm折叠、Q-Former常量折叠，以及YaRN factor4、16384位置的cos/sin缓存，不能直接复用4.1缓存。作者CPU ONNX Runtime、73句LibriSpeech dummy下WER2.43%，旧4.1同管线2.70%；小样本与CPU结果不能验证浏览器速度或完整多语精度。卡称decoder logits为Transformers的1/8，greedy不变不等于采样/概率完全一致。Apache2.0。适合先复现不同浏览器、音频长度、贪心一致性与缓存边界。

稳定ID：`hf:ibm-granite/granite-speech-5.0-2b-ONNX@a9cf1d13d045d65813b4bc80bde8fdd62db4c35c`。

### Gemini 4 Carbon受限预览：公布长任务与网络安全评测，训练实现仍未披露

**Google DeepMind；10-28 17:00 UTC公告，18:02更新。** [官方原文](https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-gemini-4-carbon/)、[评测方法](https://deepmind.google/models/evals-methodology/gemini-4-carbon)。当前是Fairwind向部分防御机构/可信伙伴逐步开放，公众产品/API仍属后续计划。官方表相对Argon的SEC-bench Pro为83.4→88.2、ExploitGym42.0→54.8；SEC取183个JS任务、三次独立尝试的pass@1均值，ExploitGym用Antigravity、无互联网/mitigations关闭并列6/10小时预算，**不能把这些未经guardrails的结果直接当部署能力**。参数、架构、训练数据/算力及可复现权重未披露；Terminal-Bench4.0成功作弊率0.0%、尝试率0.4%只限该评测的检测口径；内部10倍补丁比较缺可重建对照。建议观察公开model/API版本与带部署限制的复测，暂不当完整技术配方。

稳定ID：`google:gemini4-carbon:limited-preview:2026-10-28`。

<a id="pretraining"></a>

## 论文：预训练、架构与低精度

本节以实际打开的指定版本全文为依据；数值均为作者报告，本轮核对方法、实验设置、表格与局限，没有独立复现或冻结实现 commit 的机制审计。当前 arXiv new 列表的日期不能代替首发日期：以下新作主要首发于 10-26／10-27，均按回溯首次收录标记。

### MuonGS：去掉径向更新，放松固定权重范数约束

**2026-10-27 v1；回溯首次收录；`arxiv:2610.37647v1`；机构：University of Minnesota。** [原始论文全文](https://arxiv.org/html/2610.37647v1)。MuonGS 将 Muon 更新在当前权重方向上的分量投影掉，使范数增长受控，同时允许范数继续变化；论文用整段训练的累计有效学习率解释它与 weight decay、MuonH 固定范数的差别（§3–4）。

160M／400M GPT、FineWeb、BF16，覆盖约 1–8 倍 Chinchilla token 预算；每种优化器、规模、时长分别扫学习率。§5.2 的约 **7–15%** 是拟合 MuonH loss-token 曲线得到的 **token 等效节省**，没有直接证明 wall-clock 加速。135M active／1B total MoE 的两种预算也报告更低的最优验证 PPL，但仍属初步小规模证据（§5.3）。

**实践与建议：阅读，随后小规模复现。** 对齐学习率搜索预算、实际更新范数和长训练预算，分别验收 token 效率与每步开销，暂不外推到数百 B MoE。

### EnergyCal：用 post-NS 能量衰减专家更新，减轻 loss spike

**2026-10-27 v1；回溯首次收录；`arxiv:2610.37062v1`；机构：Fast Accounting／FA Research。** [原始论文全文](https://arxiv.org/html/2610.37062v1)。对每个 routed expert 逻辑矩阵独立计算有限步 Newton–Schulz 方向的归一化平方 Frobenius 范数，将其转为不超过 1 的更新系数；不改 NS 变换、router，也不新增持久 optimizer state（§3）。能量并不是 stable rank 或已验证的“更新可信度”。

自建 4B DeepSeekMoE 风格模型、8×H100、1.31B tokens、三个新初始化 seed。匹配学习率和 weight decay 的 plain Muon 比较中，p=1 的 spike burden 为 **2.57±2.46 对 84.79±33.13**，最终 PPL **24.191 对 24.283**（Table 4）。MuonH 对比中的最终质量差异还混有不同学习率；其吞吐 65,977 对 67,061 token/s，约低 1.6%（Tables 1–3）。两个保存事件的单步干预均没消除随后 spike，论文也缺少 routed-only 统一缩放对照，不能据此宣布找到 spike 因果根源（§6–7）。

**实践与建议：复现。** 优先比较同 LR／WD 的 plain Muon、统一衰减与能量衰减，分别记录 spike burden、最终质量、等时间质量和实测吞吐。

### Merlin：用 key 的二阶统计聚合多个线性注意力状态

**2026-10-27 v1；回溯首次收录；`arxiv:2610.37645v1`；机构：Princeton University／Google DeepMind。** [原始论文全文](https://arxiv.org/html/2610.37645v1)。把上下文切成 256-token 段，每段保存 memory state 及低维 key 的一、二阶矩；query 同时读取段内平均匹配与匹配方差，经可学习 gate 混合后分配段权重，比仅用位置或 mean-pooling 多出“少数 key 很相关”的信号（§3）。

Gated DeltaNet-1.3B 和 Mamba-2-780M 经两阶段共约 2.06B token mid-training；16K LongBench 平均为 **21.9／20.6**，对 meanpool **19.3／18.4**；RULER 为 **62.7／76.2**，对 **57.9／71.2**（Tables 2–3）。训练包含 NIAH、HotpotQA、SQuADQA 风格合成数据；这些结果衡量特定 mid-training 后的长上下文能力。保存多个段状态及二阶矩使内存随段数增长，Table 1 不是固定常数状态，也没有给出端到端服务速度。

**实践与建议：阅读。** 适合探索 hybrid／linear attention 的分段检索；复现时加上等状态内存、等训练数据的对照，避免只比较检索分数。

### CoS-VQ：联合选择共享 scale 下的多个码字

**2026-10-27 v1；回溯首次收录；`arxiv:2610.37336v1`；机构：Houmo AI／北京大学／东南大学。** [原始论文全文](https://arxiv.org/html/2610.37336v1)。固定 codebook 和已有 scale 分组，用残差感知 majorizer、多个 scale probe 与 lower-envelope 搜索联合更新 hard indices 和共享 scale；每次候选都用真实 calibration layer-output error 验收（§3–4）。

Qwen3／Llama-3.1 六模型、320×4096-token calibration；Qwen3-8B 的六任务 ACC 为 **68.61**，对 LLVQ **67.35**、BF16 **70.37**，但 14B ACC **71.01** 低于 LLVQ **71.28**（Table 1），并非各项都胜出。Llama-3.1-8B、RTX4090、2K context decode **36.7 对 BF16 21.9 token/s**，prefill 基本持平（Table 4）。协议称 2-bit indices 外另有 FP16 scales，而 Table 5 默认配置写 2.04 bpw；实际 checkpoint 的 scale／codebook／padding 计费仍应核对。理论最优针对有限候选 surrogate，非全局原始误差最优（Limitations）。

**实践与建议：观察并复现。** 先冻结实际 bpw 和量化层范围，验收序列化体积、质量及目标硬件 lookup kernel，再采用吞吐倍数。

### 无校准 MoE channel selection：预算要对齐实际读字节

**2026-10-27 v1；回溯首次收录；`arxiv:2610.37386v1`；机构：KAIST／DeepAuto.ai。** [原始论文全文](https://arxiv.org/html/2610.37386v1)。用权重构建旋转 ternary sketch，逐 token 以 top-k 选择 routed expert channels，并用权重派生低秩图恢复跳过部分；无需 calibration 样本。论文同时把“校准阈值目标预算”与测试时真实读取量拆开比较（§4–6）。

三种至多 35B MoE、九个模型任务组合，在 **17.8% routed-expert reads** 下保留平均 **98.1%**、最差 **95.0%** 的全模型准确率（Table 1）。单 RTX PRO 6000 Blackwell、vLLM0.29，128-token prompt／64-token output：整模 decode 在 batch1 仅 **1.15–1.36×**，batch8–16 峰值 **1.74–1.88×**，batch128降到 **1.00–1.24×**；不能把 MoE block 的 2.69×当整模结果（Appendix J、Table 20）。sketch 与恢复图还增加约 BF16 checkpoint 的 7% 和0.5%内存；只修改 decode routed experts，prefill不变。

**实践与建议：复现。** 同时测真实读字节、长生成循环／质量、常驻辅助内存和并发曲线；量化模型须重新计费。

### NLD-Uni：跨范式蒸馏先对齐 student 的 AR 视图

**2026-10-27 v1；回溯首次收录；`arxiv:2610.36466v1`；团队：NVIDIA Nemotron Labs。** [原始论文全文](https://arxiv.org/html/2610.36466v1)、[固定HF卡／配置／文件树](https://huggingface.co/nvidia/NLD-Uni/tree/5db755a2eb0ab84654e3dcf80c7d0245bfe36a50)、[固定LICENSE](https://huggingface.co/nvidia/NLD-Uni/blob/5db755a2eb0ab84654e3dcf80c7d0245bfe36a50/LICENSE)。统一离散扩散理解／生成双塔中，直接将 AR teacher 分数赋给已部分揭示的 diffusion context 会产生不一致监督。它先把 teacher 蒸馏给 student 的 block-size=1 AR 视图，再让 diffusion 视图匹配 stop-gradient 的 student AR 分布；图像编辑则用 flow teacher 对 student 图像加噪、再去噪细化，以SFT学习teacher-refined目标；作者说明这并非传统意义的on-policy（§3.3–3.4）。另以 token age 和 embedding ball-tree 引入逐步 corruption／self-refinement（§3.2）。

64×H100训练26天，其中最终OPD5天；理解 teacher 为 Nemotron-3 Nano Omni、编辑 teacher 为Qwen-Image，T2I仍混SFT而未使用OPD teacher。Geometry3k 从 base **54.4→67.9**，naive OPD只有 **23.0**；We-Math **59.8→74.9**（Table2）。原文Figure2／AppendixD明确按任务加载三种视图：VLM8B，T2I为4B理解加8B生成即12B，interleaved16B；Table3的12B针对T2I，不能当全模型总参。模型分支核固定HF SHA `5db755a2eb0ab84654e3dcf80c7d0245bfe36a50` 的API整包存储计数约18.27B，release manifest称generation-only-step-10000；存储整包与任务加载视图是不同口径，本轮尚未核清整包组成及差额。其同tokenizer要求仍在，低NFE时增加refinement反而会变差，细粒度文字渲染仍不足（Appendix C/F）。

**实践与建议：阅读并跟进工件复现。** 优先检验 context 对齐及 AR-view→diffusion-view 的损失消融；质量提升不能单独证明同计算预算优势。模型分支打开的LICENSE §3.3将用途限制为research/evaluation非商用（NVIDIA及affiliate例外），不能因组织或“open”措辞推断自由商用。

<a id="posttraining"></a>

## 论文：后训练与 Agent RL

### EnvSched：环境覆盖与奖励差异联合调度训练任务

**2026-10-27 v1；回溯首次收录；`arxiv:2610.37609v1`；机构：北京大学／西安交通大学／清华大学／Samsung Research／中科院自动化所／香港中文大学。** [原始论文全文](https://arxiv.org/html/2610.37609v1)。以环境–任务对为节点，同环境边利用工具／状态／verifier结构，跨环境边利用抽象操作拓扑，传播历史 reward 统计；每批一半用于覆盖欠采样环境，一半用于高信息量任务，并降低重复曝光（§2）。

EnvScaler的51环境、2550任务，Qwen3系列共200步、每步64任务×8rollouts，三个seed。4B的 BFCL-v3 Multi-Turn **31.88 对 random 28.83／LEEPS30.00**，τ-Bench **41.01 对37.78／38.79**；40步已达到30.17／41.01（Table1）。这是相同trajectory预算下的学习效率证据，非等token／等wall-clock证明；8B的40步τ-Bench50.71还略高于200步50.30，更多训练不总更好。工具用户模拟器与pool结构也是适用条件（§3.1）。

**实践与建议：复现。** 环境分布变化时联看覆盖、reward方差、最终质量与实际rollout时长，不能只优化非零advantage比例。

### AdaRoll-GRPO：离线求 rollout 分配表，在线按结果继续采样

**2026-10-27 v1；回溯首次收录；`arxiv:2610.37429v1`；团队：AdaRoll-GRPO作者团队（本轮未从一手正文确认机构）。** [原始论文全文](https://arxiv.org/html/2610.37429v1)。把每题未知成功率写成Beta belief，在固定总rollout数下用Bayes-adaptive MDP最大化组内reward差异等utility；value iteration离线完成，训练期间查表选择下一道题（§4–6）。

H10080GB，8B用两卡、其余一张；Llama3.1-8B、MATH，vanilla GRPO **42.60→48.60**，DAPO **46.60→49.40**（Table2）。不过离线复杂度 **O(B·N^(2B))**，演示可行例只有全局4prompt、每GPU16rollouts；“离线”不代表对大batch免费。主utility还依据14个模型任务配置的测试结果选择，且未见多seed误差，结果宜视为探索证据；等rollout数也非等token／时间。

**实践与建议：阅读。** 先检查小batch查表方案能否嵌入现有异步采样，再与更简单的自适应采样及独立验证集选utility比较。

### ConCal：从相同 incoming transition 借用 continuation，保持 visit 自身证据

**2026-10-27 v1；回溯首次收录；`arxiv:2610.37195v1`；机构：中山大学／Shenzhen Loop Area Institute。** [原始论文全文](https://arxiv.org/html/2610.37195v1)。针对长轨迹局部动作credit，把多个visit的continuation部分共享，递归校准，再按实际reward与successor形成action value；不把所有匹配visit完全压成一个状态（§4）。

Qwen2.5-1.5B、150updates、每步16任务×8轨迹、三个训练run：ALFWorld **94.27±1.97**，对GiGPO90.88；WebShop成功率 **79.69±3.41**，对73.83（Table1）。主要baseline取自先前论文，不能当作本轮同harness重跑；内部partial-pooling消融才是更直接的机制证据（Table2）。匹配并不保证历史、hidden state或剩余预算相同，借用错误continuation会引入偏差（Limitations）。

**实践与建议：复现。** 在可重放环境里同时测匹配覆盖和匹配误差，保留own-MC／fully-pooled／single-pass三个对照。

### MPO：turn 级传播保留长程 credit，turn 内再细分 token

**2026-10-26 v1；回溯首次收录；`arxiv:2610.35013v1`；团队：Tao Ren等MPO作者团队（机构未核）。** [原始论文全文](https://arxiv.org/html/2610.35013v1)。先沿完整assistant turn做GAE，再把turn内token trace去均值后加回；使每轮平均credit仍等于turn advantage。更新用完整turn importance ratio，同时阻断其他token比率的梯度，保留局部token差异（§5）。

Qwen3-4B BrowseComp-Plus pass@1 **48.08**，对SAO43.75、PPO40.85；30B-A3B为 **50.32**，对46.73 VAPO（Table1）。每设置一次训练、取最佳checkpoint再重复离线评测五次；不是五个独立训练seed。Claw／Automation训练数据专有，无法按公开论文完全复现，其余结果也没有等总GPU时间论证（§6.1、AppendixB）。

**实践与建议：阅读并做公开search子集复现。** 将turn边界、截断bootstrap、token mask及整turn比率稳定性列为首要验收项。

### Guided-RLVR：在高熵句界定向探索，代价与采样偏移仍需分开

**2026-10-27 v1；回溯首次收录；`arxiv:2610.37041v1`；机构：北京航空航天大学。** [原始论文全文](https://arxiv.org/html/2610.37041v1)。小controller在高熵句界选择反思／推进等steering token，提升其logit；每组2条普通加8条guided rollout，只对正确且实际触发的轨迹添加log-perplexity bonus（§3）。

Qwen3-4B、GRPO，AIME2024 pass@1 **41.17**，对vanilla15.94、REP37.32；Qwen2.5-7B在AIME2025则 **5.03** 低于vanilla5.99（Table1），不能只报混合pass@1／pass@K的AVG。headline模型还用DeepSeek-R1轨迹监督cold-start controller；另行无cold-start的RTX PRO6000 LoRA计时为 **17.92对15.92 GPU-h**（Table4），不等于headline全成本。更新对guided样本使用unboosted概率，论文将其定位为surrogate，非完整消除采样策略偏移。

**实践与建议：观察。** 先独立拆开额外监督、定向采样、bonus和importance correction，再验收同总token预算结果。

### RPSD：检索证据只在训练时提供给 self-distillation teacher

**2026-10-27 v1；回溯首次收录；`arxiv:2610.37281v1`；机构：名古屋大学／深圳大学／四川农业大学／广东人工智能与数字经济实验室／同济大学。** [原始论文全文](https://arxiv.org/html/2610.37281v1)。同模型teacher看到真实下一item和相似用户检索证据，重新给student的on-policy token打分；optimal transport用于对检索到的用户历史窗口重排；teacher／student的token置信差只重分配GRPO advantage，推理时移除特权信息（§3）。

Qwen3-4B，100候选Netflix HR@5 **0.3030**，对GRPO0.1706；MovieLens HR@5 **0.2740对0.2440**（Table2）。候选由真实item和随机未交互负例构成，训练输出单item、评测Top-k分别生成；这些并非全库排序或线上业务收益。训练期teacher直接获知标签是方法前提，也应与有同等额外信息的蒸馏对照比较。

**实践与建议：阅读。** 适合推荐任务的特权上下文蒸馏实验，先补全库／hard-negative和检索泄漏检查再部署。

<a id="inference"></a>

## 论文：推理、系统与资源调度

### TAO：联合选择与放置 Agent session，把工具返回和 KV 复用写进决策

**2026-10-27 v1；回溯首次收录；`arxiv:2610.37230v1`；机构：上海交通大学／Stanford。** [原始论文全文](https://arxiv.org/html/2610.37230v1)。按衰减KV cache credit、工具返回readiness、token footprint构造字典序价值，在容量约束下联合决定哪些session上哪些节点；异步解snapshot后重新核验session签名、资格与容量，过期计划可丢弃或仅执行可行前缀（§3、AppendixB）。

DeepSeek-V4 Flash、2节点各4×H100、SWE-Lite64workers，平均latency **185.410秒**，对Dynamo269.801、cache-aware vLLM253.228，TTFT **0.224对0.351／0.410秒**（Table1）。收益随负载变：Qwen3-32B、SWE-verified128workers，TAO933.678秒反而慢于ThunderAgent911.097秒；Terminal-Bench部分decode吞吐也下降。精确DP只对冻结、离散化snapshot目标成立，预算超限会BFD fallback，不是全局最优在线服务保证。

**实践与建议：复现。** 先重放本地工具时长分布，并同时验收任务完成时间、TTFT、cache命中、计划年龄／丢弃率，而非单独最大化decode TPS。

### INC-MoE：交换机 collective 与专家计算分带重叠，硬件仍是模拟器

**2026-10-26 v1；回溯首次收录／推进旧候选；`arxiv:2610.36339v1`；机构：Huawei Switzerland／Huawei Beijing。** [原始论文全文](https://arxiv.org/html/2610.36339v1)。采用message语义的in-network AllGather／ReduceScatter，persistent MoE kernel把线程块分为gather、FFN、reduce角色，按token bands流水；既重叠通信与计算，也利用不同链路方向重叠两种collective（§3–4）。

四块加速器模拟四节点EP rank，第五块模拟INC switch；RoCE共享400Gbps上行折成每rank100Gbps，baseline NIC也限到100Gbps。OLMoE-1B-7B、FP16、DP4／EP4、24K–126K prefill tokens/rank：TTFT下降 **20.2–33.1%**，吞吐提高 **25.3–49.5%**（§5.1–5.3）；这两项源于同一组时间测量。正文只称“高性能加速器”，本轮不能补成具体GPU型号；也没有证明商用交换机、decode或跨更多节点的收益。

**实践与建议：观察。** 网络协同方向值得读，但落地前需真实INC硬件、并发隔离、失败恢复及服务质量验证。

### AutoProfiler：让 deterministic tree 选择交给 kernel agent 的 profiling 指标

**2026-10-27 v1；回溯首次收录／推进旧候选；`arxiv:2610.36576v1`；机构：中国人民大学／对外经济贸易大学。** [原始论文全文](https://arxiv.org/html/2610.36576v1)。只在CUDA-Agent-Ops-6K构造profiling tree，以SM／DRAM指标先粗分瓶颈，再按已读计数器确定下一组指标；推理阶段由树路由，完整profile只交给rewriter一次，不再让LLM决定每个profiling步骤（§4–5）。

GPT-5.6-Sol、4×RTX4090D、NCU、正确性预检的Triton seed，在KernelBench平均speedup **1.704×**、正确率 **87.2%**，CudaForge为1.570×／78.4%（Table1）。速度聚合只覆盖各自正确题，不是相同题集合；正文metric描述与表注也不完全一致，应把正确率单独看。48-kernel控制实验里动态profile1.379×、静态1.314×，而oracle1.573×，这组更直接反映指标选择增量（Table2）。不是已证明对其他GPU普适的profiling政策。

**实践与建议：复现。** 优先复现同seed、同rewriter、同题集的blind／static／adaptive对照，并计入树构造、NCU采集与失败任务成本。

### PipeLink：把共享内存、barrier和warp要求变成可链接的组件契约

**2026-10-27 v1；回溯首次收录；`arxiv:2610.37200v1`；机构：北京大学。** [原始论文全文](https://arxiv.org/html/2610.37200v1)。组件独立编译为带contract的LTO object，以handle函数表达待链接的SM资源位置；linker核对layout／protocol／facts，以SMT联合放置资源，并生成精确的数据搬运，nvJitLink再把位置折成常数（§3–4）。

H200／B300／RTX5090、锁SM频率、51配置，linked与相同源码／放置的single-unit control性能差在0.2%内（§5.1）。B300上Llama1B megakernel换入CuTe attention后4096-context **3.516→1.630ms/token（2.16×）**，16-context却慢5%（Table3）。收益来自可组合的更好组件，linker本身不让组件更快。契约只彼此检查、不证明实现遵约；生成搬运限一个thread block，跨block／GPU通信仍由组件负责，LTO包也绑定架构和toolkit（§7）。

**实践与建议：阅读并复现。** 适合多团队维护融合kernel，验收重点是contract拒绝、sanitizer与数值测试，而非仅证明链接成功。

### EdgePace：统一内存设备上为前台deadline让出推理带宽

**2026-10-27 v1；回溯首次收录；`arxiv:2610.37307v1`；机构：清华AIR／复旦／KAUST／香港中文大学（深圳）。** [原始论文全文](https://arxiv.org/html/2610.37307v1)。把prefill和decode拆到安全暂停边界，监控PMU内存压力，按前台deadline和LLM输入新鲜度选择继续、批准下一quantum或Hold，兼顾避免前台失期与后台输出过期（§3）。

Jetson三平台及OnePlus13，平台内模型／负载相同。四摄像头Orin Nano，Qwen2.5-1.5B的deadline miss **18.33%→7.667%**，吞吐 **8.175→5.848 token/s**；OnePlus13同模型miss **66.78%→5.83%**、保留90.4%吞吐（§4.3）。输入仅246tokens、输出上限16，手机用NPU；跨平台负载不同，不可拼成硬件性能榜。机器人例子中LLM输出并不改变控制动作，质量是被消费状态的新鲜度，不是机器人策略智能提升（§4.1／4.4）。

**实践与建议：阅读。** 适合边缘多任务共存；先定义deadline与staleness contract，再把牺牲的后台吞吐列入SLO。

### Metacognitive Decoding：训练 controller 决定扩散 agent 哪些位置继续计算

**2026-10-27 v1；回溯首次收录；`arxiv:2610.37642v1`；团队：Mingyu Zhao等作者团队（正文机构字段只呈邮箱，未完整确认）。** [原始论文全文](https://arxiv.org/html/2610.37642v1)。在已做tool-use SFT的扩散backbone上冻结主模型，用两层Transformer controller按置信度、格式结构与交互进展决定token提交及active范围；GRPO奖励同时考虑正确性与含controller在内的整轨迹FLOPs（§3）。

LLaDA-8B，BFCL Live **70.96对无controller66.52**，FLOPs **18.48对32.49×10¹²**；HotpotQA **46.6对45.7**，FLOPs131.91对175.00；MuSiQue准确率却 **20.3对20.7**。Dream也有2Wiki／MuSiQue质量回退（Tables1–3）。controller训练用8×A100、每阶段40轮；所报成本是FLOPs，全文未给可直接对齐的端到端wall-clock加速。

**实践与建议：观察。** 先以相同SFT权重对照实测延迟、controller开销和工具等待，再决定FLOPs节省是否能转为服务收益。

### DuplexTune：双流宽度桥与状态重放，尚缺数值恢复的直接验收

**2026-10-26 v1；回溯首次收录／推进旧候选；`arxiv:2610.36009v1`；机构：Texas A&M University-Commerce／Michigan State University。** [原始论文全文](https://arxiv.org/html/2610.36009v1)。把两个半宽residual流接到原宽attention／MoE模块，先训width bridges再更新backbone、保持router权重冻结；反向重放其中一流可直接相减，另一流因query依赖自身而须fixed-point恢复（§3）。

Qwen1.5-MoE-A2.7B、Dolly15K、单H80080GB：峰值 **38.8对checkpointed SFT64.8GB**，吞吐25.1对19.9samples/s，但各arm按容量调batch，处理token并未匹配；GaLore仍更快35.5samples/s（Table1）。MMLU67.2对SFT66.3，没有multi-seed不确定性；多语言列反而低于base，数据组成也未说明（Table2）。恢复收敛要求并非additive coupling自动保证，冻结router也不冻结dispatch，任务分数不验证gradient replay一致性。

**实践与建议：观察。** 首先要求与stored-activation reference的恢复残差／梯度一致性，再比较等token训练质量与内存。

<a id="data"></a>

## 数据与训练工件

### Granary-v2：同一音频生成多种监督，公开的是46.3万小时标签清单

**NVIDIA；清单首个公开含数据提交10-27 19:51 UTC，10-28新报告与许可澄清，当前`81402b75`。** [官方说明](https://research.nvidia.com/labs/nemotron/Granary-v2/)、[49页报告](https://research.nvidia.com/labs/nemotron/files/granary-v2.pdf)、[固定清单](https://huggingface.co/datasets/nvidia/Granary-v2/tree/81402b751b451b5fe4f4c29e8b40f92892f1e440)。针对“一段录音只有一种转写标准”，管线分metadata、语言特定ASR、文本增广三部分，保留spoken-form后派生ITN、去口吃、code-switch、context、QA和翻译，再打包prompt-target；质量门控融合结构检查与转写一致性，不能当人工真值或隐私标记。

报告Table10的完整处理集**843,579小时**包含不公开的YODAS2、YouTube-Commons、ReazonSpeech-v2；实际公开16个source entries、48语言，约**463,062小时对应标签**，无源音频。英语同架构约600M Transformer-TDT、300K相同步数的过滤消融中，九集WER均值4.91→4.67，六集改善、三集稍差；训练混合还含处理集外语料，不能将结果完全归因于公开子集。翻译/多个目标不增加独立音频小时，present translation key也可能空字符串；需先按有效target筛选。

标签和instruction pairs标OpenMDW1.1，`people_speech`例外为CC BY-SA4.0，源音频按各分发方条款。已核card/schema/文件树，未全量profile记录、审核音频或重跑管线；作者称源训练split与验证/测试分离。建议深入读监督粒度与采样，复现同预算过滤消融，并验证source audio join、空target、语言/来源采样和分组去重。

稳定ID：`hf-dataset:nvidia/Granary-v2@81402b751b451b5fe4f4c29e8b40f92892f1e440`；`nvidia:granary-v2:technical-report:2026-10-28`。

### Granary-HiFi：为VoxPopuli增加声音质量、截断与上下文筛选，保留31.4%小时

**NVIDIA；已核10-28 11:23 UTC含数据工件，仓创建11:20不作数据交付，当前`89c693f2`。** [固定卡/工件](https://huggingface.co/datasets/nvidia/Granary-HiFi/tree/89c693f2bd02d485ad6971b20f061962cf9da828)。新增SED语音覆盖、单说话人标签、UTMOS、CTC/MFA早截断检测，以及近邻同说话人低文本重叠context。卡报22语言的46,055,576条/354,389.9小时中保留14,994,925条/**111,170.3小时（31.4%）**；是同源VoxPopuli派生元数据，不能再与Granary-v2累计成新增独立音频。main含全部记录与`HiFi`标志，HiFi分支是过滤子集，字段也少两个；音频仍另取。标OpenMDW1.1，论文尚在准备，未给受控下游质量消融；本轮核清单/模式与首次树，未独立验全量行数。适合先抽查语言偏差、感知分数、截断误杀和context泄漏，再评估训练收益。

稳定ID：`hf-dataset:nvidia/Granary-HiFi@89c693f2bd02d485ad6971b20f061962cf9da828`。

### ExplorationBench新增HF demo数据工件：27行开发材料，不是私有正式评测集开放

**复旦/腾讯混元/清华；10-28 06:47 UTC建立HF仓，08:46同步到GitHub`8892fd0`，固定`ebe9d3d8`。** [固定数据卡/文件](https://huggingface.co/datasets/tencent/ExplorationBench/tree/ebe9d3d8c559f416b5855a6387f0c9a4aa6bf7bd)、[官方代码](https://github.com/Tencent-Hunyuan/ExplorationBench)。这是9月论文与10-08已公开demo worlds的新分发工件，规则与私有评估world不同。本轮逐行解析六个JSONL，确认为5/4/5/5/5/3行，共27行，含worked examples、held-out问题、参考解与规则答案。答案键只能用于验收，不能送入探索上下文；demo分数不与正式140题榜单比较。Apache2.0；未执行解释器/证明检查器，正式world仍私有。建议用它验证harness的探索/工具禁用评估隔离和答案泄漏，而不宣称新增正式评测分数。

稳定ID：`hf-dataset:tencent/ExplorationBench@ebe9d3d8c559f416b5855a6387f0c9a4aa6bf7bd`。

<a id="core"></a>

## 核心开源框架：本轮核实条目

> 主窗 2026-10-28T01:02:09Z → 2026-10-29T01:02:28Z；72h回溯自2026-10-25T01:02:09Z。除明确发布条目外，main合入不等同已发公开release。所有运行/性能数均为作者报告；本轮只做正文、diff与固定源码核查，未跑GPU/NPU/TPU。

### Megatron KDA 与 mHC 数值契约进入 main，完整 GLM 训练仍需逐项验收
**NVIDIA/Megatron-LM；2026-10-28；main 已合入。** [#7054](https://github.com/NVIDIA/Megatron-LM/pull/7054)（`57103cf06330`）；[#7521](https://github.com/NVIDIA/Megatron-LM/pull/7521)（`8e134118baf7`）

HybridModel 新增 K 层与 GLM 式两阶段低秩 decay/output gate，A_log/dt_bias 保留 FP32，允许 KDA 投影在 FP8 actor 中单独保持 BF16；mHC 可选 epsilon 放平方根内、FP32 映射/残差混合及 mean 输出收缩，并放开完整激活重计算。新精度选项与 fused mHC 组合直接拒绝。作者本 PR 只验证 KDA CP=1，recurrent inference 仍 NotImplemented；调用 FLA 的 chunk_kda/gated norm 不等于本轮审计了依赖 kernel。相关 KPool/NoPE #7522 早于本窗已合入，而 blockwise FP8 wgrad #7523 已关闭未合入，不能把该组解释为全部 GLM-5.3-Flash 训练边界闭环。建议优先回归混合精度、packed THD、重计算和 checkpoint 对齐。

源码阅读：`megatron/core/ssm/gated_delta_net/kda.py::KimiDeltaAttention` → `megatron/core/transformer/hyper_connection.py::_proj_rms_impl` → `megatron/core/transformer/transformer_config.py::__post_init__`。

### Megatron MoK 后端接入原生参数与 DDP，适用范围限 TP/ETP=1
**NVIDIA/Megatron-LM；2026-10-28；main 已合入。** [#8210](https://github.com/NVIDIA/Megatron-LM/pull/8210)（`a153d5d3f543`）

moe_megakernel_backend=mok 让外部 MoK 执行 dispatch/expert/combine，MCore 继续持有路由、参数、optimizer 和 checkpoint；MXFP8 payload 零拷贝别名、转换 scale 在更新后刷新，kernel 写 main_grad 后通过无数值读取的 alias sentinel 触发既有 DDP hook。支持 BF16/MXFP8 routed experts 与 BF16 shared/gate，但限制 TP=ETP=1，并排除 latent/GTP、native shared overlap 与逐层 MoE graph 等组合。依赖固定 MoK 7a8cfb7；作者4×GB200的 Qwen3.5 形状 proxy 验证训练/恢复/全轮 graph，不是完整 Qwen3.5 性能结果，未资格验证多节点/EP8+/routed DP>1/Muon。建议把参数别名、scale 刷新和保存恢复作为接入验收重点。

源码阅读：`megatron/core/transformer/moe/megakernel/parameter_bridge.py::main_grad_buffer、finish_weight_gradient` → `megatron/core/transformer/moe/megakernel/mok/runtime.py` → `megatron/core/transformer/transformer_config.py`。

### Megatron MFSDP v1 拒绝混合通信 dtype 的除法融合
**NVIDIA/Megatron-LM；2026-10-28；main 已合入。** [#7751](https://github.com/NVIDIA/Megatron-LM/pull/7751)（`b9d4e2b2c899`）

同一 coalesced reduction group 混用 BF16 的预缩放 SUM 与 FP32 的 PREMUL_SUM 时，PyTorch 使用首 bucket 的算子可能把 BF16 再缩放一次。初始化现检查有效 grad communication dtype，混合时要求关闭 gradient_reduce_div_fusion 或统一 dtype；两份 DeepSeek FP8 recipe 补 --bf16。作者4×GB300/节点、1与2节点、NGC26.08/26.09各50步 mock-data 验证通过，但 canonical GB200/Pile golden 尚未重建。它是显式拒绝错误组合，未实现任意混合 dtype 融合。建议低精度 FSDP 升级优先检查梯度范数与配置 guard。

源码阅读：`megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py::gradient_reduce_div_fusion initialization guard`。

### Megatron MFSDP v2 可选无 padding 分片，以不等长集体换存储
**NVIDIA/Megatron-LM；2026-10-28；main 已合入。** [#8116](https://github.com/NVIDIA/Megatron-LM/pull/8116)（`47715bb029f5`）

UNPADDED 将张量连续打包，rank 边界就近落在完整 row、MXFP8 的32-row block上，支持不等长甚至空 shard；DBuffer list collectives 按 peer range 切片，量化 scale 平面沿数据布局派生。默认 padded 不变，--no-megatron-fsdp-padded-shards 显式开启，nccl_ub 因等长对称内存要求被排除。13%/80% 是特定 DeepSeek-V3 形状组在 DP4/16 的旧 padding 占比，不是训练整机显存节省；不等长 NCCL 路径可能更慢。作者4×GB200 单测覆盖布局/量化/HSDP等，建议按自身组宽、通信和空 shard 回归。

源码阅读：`megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/layout_builder.py::build_for_row_atomic_unpadding` → `megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/dbuffer.py::_peer_chunks`。

### Megatron optimizer checkpoint 允许纯尾 padding 的空 shard
**NVIDIA/Megatron-LM；2026-10-28；main 已合入。** [#6191](https://github.com/NVIDIA/Megatron-LM/pull/6191)（`d5cc5e6d816a`）

保存时空 optimizer state、加载时无 param_map 不再一律当损坏：只在 DP rank×本地 shard 长度已越过未 padding 的 payload 时跳过，覆盖真实参数的空 shard 仍 assert。作者四 rank 各8个保存/加载/错误拒绝用例通过。建议高 DP、强对齐 padding 的训练恢复优先回归；这一规则不允许忽略真实 optimizer 状态缺失。

源码阅读：`megatron/core/optimizer/distrib_optimizer.py::_is_trailing_padding_only_shard`。

### Megatron GDP 在 FP32 求 beta sigmoid，保留 BF16 饱和区梯度
**NVIDIA/Megatron-LM；2026-10-28；main 已合入。** [#8529](https://github.com/NVIDIA/Megatron-LM/pull/8529)（`a296523510e1`）

beta 从 b.sigmoid() 改为 b.float().sigmoid().to(b.dtype)：BF16 约6.25以上 logits 的前向仍可舍入为1，但反向使用未舍入的 FP32 sigmoid，避免梯度先归零。CuTeDSL/非CuTeDSL布局测试比对 BF16 forward bitwise 与 FP64 梯度参考；推理 prepare kernel 原已用 FP32。建议 GDP 训练检查 gate 饱和与梯度，不据此声称吞吐或最终收敛已提升。

源码阅读：`megatron/core/ssm/gated_delta_product.py::GatedDeltaProductMixer._compute_gating` → `tests/unit_tests/ssm/test_gdp_beta_sigmoid.py`。

### Megatron streamwise 反向独立调 tile，减少 FP32 寄存器溢出
**NVIDIA/Megatron-LM；2026-10-28；main 已合入。** [#8527](https://github.com/NVIDIA/Megatron-LM/pull/8527)（`26020f4e3cf9`）

宽度可被512整除时 read/write backward 改16×512/4warps，其余形状保留原32×256；forward不动，保持重计算 replay。作者GB300/Triton3.7.1、24576 tokens、width3072、3 streams、learned retention 的 FP32 write backward 2.987→0.522ms，spill1296→50 words；这不是576-GPU整步的同幅加速。参数 logit 梯度归约顺序会变，报告约1e-7相对FP64误差；activation梯度bitwise保持。建议 --fp32-residual-connection 用户 profile 反向并核 partial-reduction 上限。

源码阅读：`megatron/core/transformer/streamwise_residual_ops.py` → `tests/unit_tests/transformer/test_streamwise_residual_ops.py`。

### Megatron sequence packing 在所有 PP stage 构建 dataset
**NVIDIA/Megatron-LM；2026-10-28；main 已合入。** [#8592](https://github.com/NVIDIA/Megatron-LM/pull/8592)（`e4a643d84bab`）

有 sequence-packing scheduler 时，TP rank0在每个 pipeline stage 建 dataset，修 PP≥3中间stage因缺 batch/dataset 路径崩溃；普通模式仍保持原构建规则。补充 get_batch 回归，未给端到端性能数字。建议 packed PP/VPP 用户至少做3stage以上训练与eval冒烟，避免只测首尾stage。

源码阅读：`pretrain_gpt.py::is_dataset_built_on_rank` → `pretrain_hybrid.py::is_dataset_built_on_rank` → `tests/unit_tests/data/test_get_batch.py`。

### Megatron batch-size schedule 按 global batch 变化先保存再退出
**NVIDIA/Megatron-LM；2026-10-28；main 已合入。** [#8531](https://github.com/NVIDIA/Megatron-LM/pull/8531)（`d1eec51b8563`）

旧条件只看取整后的 microbatch 数，会漏掉8192→12288、DP8192/MBS1仍为1个microbatch的跃迁，继而在一致性检查前未存盘即崩溃。schedule模式现同时跟踪 global batch；RL动态batch/sequence-packing跳过保存路径保留。重启仍要求新batch能被MBS×DP整除。正文称测试未跑，最终diff已有回归测试；本轮仅检查源码，未重现训练跨档。建议把保存/退出/重启配置当完整验收链。

源码阅读：`megatron/training/training.py::train` → `tests/unit_tests/training/test_batch_size_transition.py`。

### Megatron TileLang sparse MLA 接受 NoPE512 与非64对齐 top-k
**NVIDIA/Megatron-LM；2026-10-28；main 已合入。** [#7617](https://github.com/NVIDIA/Megatron-LM/pull/7617)（`f77f360c96b6`）

adapter把512维Q/K补零到576，top-k尾部补-1，保持原softmax scale、输出形状与autograd连接，避免这些合法输入退回fallback。作者H200/TileLang0.1.12报告276相关检查通过；反向atomic只验数值、并非bitwise确定性，未跑完整训练或给性能。建议 NoPE/KPool 路径回归全invalid行、65/127 slots与Q/K梯度，不将adapter补齐等同新indexer能力。

源码阅读：`megatron/core/transformer/experimental_attention_variant/ops/tilelang_dsa.py::fused_sparse_mla_absorbed` → `tests/unit_tests/transformer/experimental_attention_variant/test_dsa_tilelang_shapes.py`。

### Megatron TP shared expert 的TE FC1使用本地shard尺寸
**NVIDIA/Megatron-LM；2026-10-28；main 已合入。** [#8145](https://github.com/NVIDIA/Megatron-LM/pull/8145)（`3e47f31ec222`）

TE GroupedLinear无TP模式，FC1 out_features现用本地weight行数，避免TP>1融合失效后scale行数assert。作者DSv3 proxy、16×GB200节点、TP2/PP4/EP16、MXFP8跑10步与eval；eval loss9.698接近未融合9.696，不是完整模型收敛证明。建议TP shared expert用户核融合选择及实际shard维度。

源码阅读：`megatron/core/transformer/moe/shared_experts.py::_make_fused_grouped_swiglu_ops`。

### Megatron refit 在单DP group也轮转source replica
**NVIDIA/Megatron-LM；2026-10-28；main 已合入。** [#8502](https://github.com/NVIDIA/Megatron-LM/pull/8502)（`4a55e6fee610`）

删除所有候选source同DP组时固定挑replica0的捷径；TP=PP=1来源现按dst_rank在replica间轮转，保留collocated本地拷贝优先，source/destination仍从相同metadata独立导出计划。nccl_m2n可能因此新建更多mesh-pair communicator。仅见规划回归，无端到端速度数字；建议测source流量分布与communicator数量。

源码阅读：`megatron/core/resharding/utils.py::select_src_metadata_balanced`。

### Megatron FlashInfer CUTLASS 接tanh-clamped squared-ReLU MoE
**NVIDIA/Megatron-LM；2026-10-28；main 已合入。** [#8013](https://github.com/NVIDIA/Megatron-LM/pull/8013)（`a7147091851d`）

FlashInfer推理后端接ClampedRelu2并添加CUTLASS MXFP8权重包装，沿用原payload/scale存储的view，scale按4个byte重看为int32接口。新路径只接受non-gated squared-ReLU、有限clamp、无bias；MXFP8 hidden/FFN须128整除，batch-invariant和专用routed token-cap配置拒绝。外部FlashInfer#5696已合，但本轮只核接口/guard与新增数值测试代码，无独立运行或端到端性能报告。建议使用该激活的模型先固定FlashInfer依赖，验证refit后scale布局及实际选中后端。

源码阅读：`megatron/core/inference/moe/flashinfer_mxfp8.py::prepare_cutlass_mxfp8_weights、flashinfer_cutlass_mxfp8_moe` → `megatron/core/transformer/transformer_config.py::__post_init__`。

### TorchTitan 0.4.0 正式发布，main 同日新 API 另行分账
**pytorch/torchtitan；2026-10-28 18:05:38Z；正式release v0.4.0；SHA `45e33cd413b64cb37a7840cfd3c9765d3c639ae1`。** [官方release](https://github.com/pytorch/torchtitan/releases/tag/v0.4.0)

v0.4.0于10-28 18:05:38Z公开，tag固定45e33cd413b64cb37a7840cfd3c9765d3c639ae1、release/0.4。release notes汇总模型覆盖、Dist-MoE/MXFP8/实验NVFP4、共享TrainingEngine的TitanRL、Grain与checkpoint；强调PyTorch2.15、MXFP8需TorchAO nightly，0.18缺所需kernel，DSv4全规模checkpoint/收敛和多模态CUDA graph仍有边界。此为发布汇总，未重复当新性能实测。已打开tag源码：仍BaseLoss、无LocalCompileConfig，因此下面main的loss/local_compile变更不归入0.4.0。建议升级先对release配置和optional依赖做固定版本冒烟。

源码阅读：`v0.4.0::torchtitan/components/loss.py::BaseLoss` → `v0.4.0::torchtitan/distributed/local_compile.py`。

### TorchTitan main 让内部 local compile 可替换 backend
**pytorch/torchtitan；2026-10-28；main 已合入。** [#5463](https://github.com/pytorch/torchtitan/pull/5463)（`0f9fd2bd6403`）

model.local_compile_regions改为model.local_compile: LocalCompileConfig(regions=...)，apply统一绑定命名区域及总会编译的FlexAttention/block-mask helper；外部设备可覆盖_compile，不再只改变调用者自己的torch.compile。默认仍Inductor。作者CPU focused43例通过，未给新设备硬件性能；v0.4.0 tag仍旧API。建议自定义backend维护者同时迁移recipe与内部编译边界，防只改顶层compile。

源码阅读：`torchtitan/distributed/local_compile.py::LocalCompileConfig.apply、_compile`。

### TorchTitan main 重构 loss 的 tensor/tuple 与执行包装契约
**pytorch/torchtitan；2026-10-28；main 已合入。** [#4582](https://github.com/pytorch/torchtitan/pull/4582)（`740d5f755a36`）

Loss泛型区分tensor与tuple输入，MTP cross-entropy以组合表达多个目标，ChunkedLossWrapper继续负责chunk/lm_head/反向连接，策略loss迁移到统一接口。最终源码Loss继承Function[LossResult]，与正文声称直接继承Configurable不同；本条按diff，不引用旧设计表述。作者固定seed、禁graph、MTP+chunked在1卡及TP2/CP2/SP四卡各10步loss/grad-norm差0；并非所有模型/graph全覆盖。v0.4.0仍旧BaseLoss，建议main自定义loss先迁移并验证token-count归一化。

源码阅读：`torchtitan/components/loss.py::Loss、ChunkedLossWrapper` → `torchtitan/models/deepseek_v3/mtp.py::MTPCrossEntropyLoss`。

### TorchTitan 高精度 RMSNorm override 保留 checkpoint 布局
**pytorch/torchtitan；2026-10-28；main 已合入。** [#5642](https://github.com/pytorch/torchtitan/pull/5642)（`bf88ad0bb4f3`）

recipe override默认以FMA误差分解和TwoSum补偿FP32累加平方和，可选真正FP64；rstd每行仍用FP64标量计算，Kahan模式最终norm乘法仍FP32，不能称整个前后向等价FP64。普通dtype反向FP32、dw两阶段确定归约；FP64输入需关闭Kahan。input_dtype/out_dtype可单独设置，参数布局兼容nn.RMSNorm。作者新增GPU回归但无通用速度结论，FP64吞吐弱的GPU可能更贵。建议数值敏感模型用它定位归约误差，并单独测精度/吞吐。

源码阅读：`torchtitan_recipes/overrides/higher_precision_rms_norm.py::HigherPrecisionRMSNorm、rms_norm_forward_kernel`。

### TorchTitan 提供 Qwen3 typed-decision recipe，稀疏监督只算答案行
**pytorch/torchtitan；2026-10-28；main 已合入。** [#5569](https://github.com/pytorch/torchtitan/pull/5569)（`f5f3d2900277`）

BEV问题变为letter选项，只监督答案token；choice/yes-no顺序随机化，varlen文档隔离同state各问题，AnswerTokenLoss先收集监督位置再复用chunked lm_head/loss。作者2×GB300、Qwen3-1.7B、600步，1500个同分布且与校准/训练state隔离的问题accuracy0.611→0.873；非跨域能力证明。1卡68.7K→105.5K tok/s同时改变AC、loader与文档上限，不能全归因只算答案行；PP因tied embedding/head不支持。建议关注选项顺序敏感性、校准与静态buffer上限。

源码阅读：`torchtitan_recipes/decision/qwen3.py::DecisionLetterProcessor、AnswerTokenLoss` → `torchtitan_recipes/decision/prepare_data.py` → `torchtitan_recipes/decision/evaluate.py`。

### verl VeOmni ROCm recipe 修量化导出、host optimizer 内存与权重布局
**verl-project/verl；2026-10-28；main 已合入。** [#8369](https://github.com/verl-project/verl/pull/8369)（`6f3f449e113b`）；[#8370](https://github.com/verl-project/verl/pull/8370)（`b65da71c1745`）；[#8371](https://github.com/verl-project/verl/pull/8371)（`6d04f398f1d5`）

VeOmni接rollout quantizer，但checkpoint的HF导出保持原trainer权重；拒绝DeepSeek-V4手动rollout.quantization覆盖及quantized delta_sharded。FSDP2 CPU offload默认fused AdamW以避免foreach同参数大小临时量，显式override仍优先；加载可选每rank读/EP分片。ROCm不再默认DeepGEMM，并给rollout传VLLM_ROCM_DSV4_BPRESHUFFLE=0，防AITER预shuffle后同步原布局静默错读，gfx94x还检查vLLM是否识别该开关。作者gfx942/gfx950各2节点跑通，但VeOmni#1363目前仍open，实测用了等效外部补丁及特定TileLang/TileKernels；不能称任意已发布镜像直接可用。建议把后端pin、host峰值与每次同步logprob一致性一起验收。

源码阅读：`verl/workers/engine/veomni/transformer_impl.py::_get_hf_export_per_tensor_param、_build_optimizer` → `verl/plugin/platform/platform_rocm.py` → `examples/grpo_trainer/run_deepseek_v4_veomni.sh`。

### verl Qwen3.5 MoE GRPO 交付 TPU v7x 固定配方
**verl-project/verl；2026-10-28；main 已合入。** [#8382](https://github.com/verl-project/verl/pull/8382)（`2ab35292039a`）

仅改examples，依赖verl-hardware-plugin 1eb3a61、TorchTitan89ef29b5、vLLM0.32.0/vllm-torchtpu732c4346；trainer/rollout各8 TensorCore devices、SimpleFSDP8/TP1/CP1与rollout TP8，16prompts×4、512/2048长度，文本Qwen3.5-35B-A3B开启thinking、unified block pool。libtpu内部PROMOTE_F32用于sigmoid等链的BF16数值对齐。作者100步平均28.56s、128题GSM8K55.47→96.09%，不是完整benchmark或多seed；新增CP/QKV开关不等于此实测用了CP。建议完整复现这些pin与同步误差，再扩大模型/拓扑。

源码阅读：`examples/tpu/grpo/run_qwen3_5_torchtitan.sh` → `examples/tpu/grpo/Dockerfile` → `examples/tpu/grpo/README.md`。

### verl weight-sync admission barrier 在提交引擎时释放
**verl-project/verl；2026-10-28；main 已合入。** [#8397](https://github.com/verl-project/verl/pull/8397)（`081c76bba2a9`）

默认abort路径过去等每请求首token才减_admitting，队列超过max_num_seqs时连WAITING请求也要prefill后才能pause。改直接调用AsyncLLM.add_request，返回后利用与pause同一有序socket保证覆盖，再释放barrier；取消/错误仍abort且平衡计数。作者Qwen3-8B/TPUv6e-8、256请求/max_num_seqs32观察旧每同步5–14s等待，CPU20例通过；GPU同代码路径但无独立GPU时延数据。建议异步RL测满WAITING队列和取消时的权重切换。

源码阅读：`verl/workers/rollout/vllm_rollout/vllm_async_server.py::vLLMHttpServer._generate_admitted`。

### vLLM P2P 源端 teardown 保留在途 WRITE，30秒强制上限仍有风险
**vllm-project/vllm；2026-10-28；main 已合入。** [#64250](https://github.com/vllm-project/vllm/pull/64250)（`a1cfd2c1b584`）；[#63779](https://github.com/vllm-project/vllm/pull/63779)（`0beab6c251e8`）

源端关闭以wait取消，仍PROC/PEND的transfer保留handle、source pins和旧registration，retired项不发给新session，并在保留期间拒绝同peer重连。正文写120s，merge a1cfd2c1固定源码明确_RETAINED_HOLD_CAP_S=30.0；到时仍强制释放并记error，不能视为transport fencing保证。另tiering JobResult.deferred_release把“请求失败”与“chunk可复用”拆开，但本PR时无in-tree tier设置event，P2P consumer后续接线未在此合入。#61667仍open原head4fd16ddb，#63715关闭未合入；源端CPU338例和generic tier测试不证明真实RDMA迟到WRITE安全。建议回归断连/超时/重连与source/consumer两侧pins，保留旧缺口。

源码阅读：`vllm/v1/kv_offload/tiering/p2p/manager.py::_poll_retained、_retained_peer` → `vllm/v1/kv_offload/tiering/p2p/data/nixl.py::poll_retired` → `vllm/v1/kv_offload/tiering/manager.py::_complete_deferred_releases`。

### vLLM HiSparse 将异步 KV load 延后到在途 step 完成
**vllm-project/vllm；2026-10-28；main 已合入。** [#63827](https://github.com/vllm-project/vllm/pull/63827)（`d482020608af`）

HiSparse早归还的clean resident page仍可能被上一forward读取，CPU发起NIXL load可抢先覆盖。scheduler在sched_step_seq>processed_step_seq、确有在途step时，对HiSparse consumer的异步load记录step fence，update_from_output处理完在途step再注册；abort未提交load也按同fence释放，无device sync。受延迟请求通常多等一步，无在途step则直接注册。正文GLM-5.3压力结果针对更早“仅延迟复用页”版本，最终实现范围更广，不能将旧TTFT数当final实测。建议P/D多轮长答案同时测重复token、MTP接受率与取消。

源码阅读：`vllm/v1/core/sched/scheduler.py::_drain_deferred_loads、update_from_output` → `tests/v1/core/test_deferred_block_free.py`。

### vLLM sparse MLA 分块规避65535 launch上限，防长prefill静默漏算
**vllm-project/vllm；2026-10-28；main 已合入。** [#63791](https://github.com/vllm-project/vllm/pull/63791)（`4da2a82a24b3`）

FlashInfer trtllm-gen把每query token当request映射grid.z，65536 token步会launch失败却返回未初始化out。现每次最多65535行写预分配out/LSE切片，DCP workspace也按单次launch上限预算，避免torch.cat额外峰值。≤65535仍单launch。作者SM107长prompt/69,635行测试与GLM-5.3-Flash NVFP4回归；短5-shot GSM8K可能走dense路径而漏检。建议大batched-token、DCP与长上下文质量优先回归，不把修复解释为通用吞吐提升。

源码阅读：`vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py::_TRTLLM_GEN_MAX_REQUESTS、_forward_mqa_kernel`。

### vLLM V2 DBO 放开 spec 与 DCP，PCP/LoRA等仍拒绝
**vllm-project/vllm；2026-10-28；main 已合入。** [#61464](https://github.com/vllm-project/vllm/pull/61464)（`cdd2750cd5e7`）

目标microbatch后为draft重建完整batch metadata/slot mapping，uniform多token verification按request边界切分、使用DP一致token数与graph descriptor，阈值至少num_ubatches×query_len；避免奇数request切成非uniform路径。DCP既有分片/通信buffer基础上移除guard。PCP、PP、LoRA、多模态/encoder等仍不支持，draft本身仍全batch运行。新增ngram与DCP GSM8K E2E测试，未给普适加速。建议固定V2开关测试奇数batch、空DP rank及FULL graph。

源码阅读：`vllm/config/vllm.py::_get_dbo_unsupported_features、_raise_dbo_thresholds_for_spec_decode` → `vllm/v1/worker/gpu/ubatch_utils.py::create_ubatch_slices` → `vllm/v1/worker/gpu/spec_decode/speculator.py::_build_drafter_attn_metadata`。

### vLLM AV draft graph沿用DP已一致的query几何
**vllm-project/vllm；2026-10-28；main 已合入。** [#62796](https://github.com/vllm-project/vllm/pull/62796)（`5118d990fb9d`）

AV draft-prefill现捕获varlen并排除含prefill的decode-only图，DPSyncState保留各rank已一致的max_query_len（含None），避免peer prefill时本地错误padding或额外collective。来源验证主要CPU图选择和DP回归，未重新做GPU端到端。建议多DP rank混合prefill、长短decode时检查draft/target共享metadata与FULL replay。

源码阅读：`vllm/v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py` → `vllm/v1/worker/gpu/dp_utils.py::DPSyncState`。

### vLLM MRV2 lm_head LoRA 按bonus与draft行分配adapter
**vllm-project/vllm；2026-10-28；main 已合入。** [#59341](https://github.com/vllm-project/vllm/pull/59341)（`06e172b5cf88`）

make_lora_inputs按diff(cu_num_logits)重复每request ID，使bonus+各draft logits行都对应自身adapter，避免只映射一行或污染邻请求。来源两个CPU回归覆盖mixed batch与sampled行；AV/PCP仍拒绝LoRA，不等于放开这些组合。建议LoRA+spec服务对照每request真实logits行数做GPU输出回归。

源码阅读：`vllm/v1/worker/gpu/lora_utils.py::make_lora_inputs`。

### vLLM MRV2 预热覆盖 DFlash/DSpark 与 AV 晚发 JIT
**vllm-project/vllm；2026-10-28；main 已合入。** [#63819](https://github.com/vllm-project/vllm/pull/63819)（`458334f00f7c`）

DFlash输入准备固定128 tile并按多block处理，AV accumulate不再按batch特化，warmup主动编译refit并清统计，防第100步后或新bucket线上编译、jit-monitor=error中断。作者B300、Qwen3-4B+DSpark及Qwen3-8B+DFlash、k7/多batch/超过100步未见这三kernel新JIT；不是全服务无JIT保证。prepare输出bitwise相同，AV统计仍可能因FP32归约顺序不同。建议上线前扫实际prompt/batch范围及refit周期。

源码阅读：`vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py::warmup` → `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py` → `vllm/v1/worker/gpu/warmup.py::warmup_kernels`。

### vLLM DSA indexer 直接排 BF16 logits，并预算 decode 最坏峰值
**vllm-project/vllm；2026-10-28；main 已合入。** [#62240](https://github.com/vllm-project/vllm/pull/62240)（`c105af429241`）；[#62860](https://github.com/vllm-project/vllm/pull/62860)（`317081fb6dc8`）

opt-in deep_select prefill直接读DeepGEMM BF16，跳过FP32上转；DCP/XPU/不兼容stride等保持fallback，tie索引未定义的边界仍在。作者GB200/DSv4.1-Flash dummy、DEP2、每rank8×4096新token+128K prefix、无spec，step429.8→401.5ms（6.6%），非全服务收益且来自含额外DSv4.1改动分支。另profile按max decode rows×对齐context×BF16与必要FP32副本预留，防1M/5draft等高并发运行时OOM，可能减少可分KV容量。建议同时验top-k质量、eager/mixed峰值与实际内存余量。

源码阅读：`vllm/model_executor/layers/sparse_attn_indexer.py::sparse_attn_indexer`。

### vLLM NVFP4 compressed-K gather 按8token分块，整模收益有界
**vllm-project/vllm；2026-10-28；main 已合入。** [#63018](https://github.com/vllm-project/vllm/pull/63018)（`6abdff59301f`）

将单token依赖load链改为每轮8token向量load，用精确bit placement解e2m1；入口、flags和输出位不变。作者B300 kernel在8×64K rows341→128us，偏斜batch181.2→33.8us；16–32K上下文端到端没有可测变化，因为该gather不到prefill1%。180例通过含随机cache bytes/subnormal/NaN scales等，不等于新格式支持。建议只在长/偏斜gather占比高时按profile判断价值。

源码阅读：`vllm/models/deepseek_v41/common/ops/cache_utils.py::_dequantize_and_gather_k_nvfp4_kernel`。

### vLLM SM120 NoPE FP8 MLA 采用528字节紧凑行
**vllm-project/vllm；2026-10-28；main 已合入。** [#58305](https://github.com/vllm-project/vllm/pull/58305)（`dbcf56882182`）

kv_lora_rank512且RoPE维0时行只含512FP8 latent+4FP32 scales，656→528 bytes；backend统一给KV spec与HiSparse注册实际宽度，writer区分逻辑宽度/物理stride，固定576 gather拒绝紧凑输入。真实RoPE64仍656，没有新dtype/CLI或外部cache布局协商迁移。作者4×RTX PRO6000 Blackwell、GLM-5.3-Flash-NVFP4 TP4、BF16/text/no-spec验证；外部connector未端到端测试。建议算容量时只对该行格式计算，并回归swap/eviction和connector兼容。

源码阅读：`vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py::get_fp8_ds_mla_row_bytes` → `csrc/cache_kernels.cu` → `vllm/v1/hisparse/runtime.py`。

### vLLM Mooncake store 按group有效bytes保存混合KV
**vllm-project/vllm；2026-10-29；main 已合入。** [#63513](https://github.com/vllm-project/vllm/pull/63513)（`f79a3a7d48da`）

区分block_stride与block_len，只注册本group层，并把Mamba各state view独立成region，跳过layer槽和state尾padding；GPU原有几何不变，store值变为有效pieces串联。Kimi-K3 DCP8例每KDA key14.97→10.53MB、整组359→253MB是特定布局，不是全KV或HBM省30%。key未变，单组/无padding不变；本轮未验证旧store混用迁移。作者376相关测试含clobber/restore保持邻接padding，建议升级测跨进程store读写与布局一致性。

源码阅读：`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py::RankLocalStoreLayout.register_kv_caches` → `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py::register_kv_caches`。

### vLLM PWAL 后保留 checkpoint 参数名与对象身份
**vllm-project/vllm；2026-10-28；main 已合入。** [#62407](https://github.com/vllm-project/vllm/pull/62407)（`e10ef091baf0`）

move_parameter让checkpoint/runtime两名别名同一Parameter，retire保留小的已消费参数而kernel枚举跳过；转换后大源张量可保留注册但storage置空，防复制整份权重。kernel_parameters去重并用runtime名，避免post-load删除/改名破坏reload。覆盖多量化loader但不是所有量化格式可任意原位热更新的证明；CPU tied lm_head等仍有例外。建议RL/IPC-cache维护者按具体loader检查对象身份、alias与重处理顺序。

源码阅读：`vllm/model_executor/utils.py::move_parameter、retire_parameter、kernel_parameters` → `vllm/model_executor/layers/quantization/compressed_tensors`。

### vLLM ROCm fused shared expert 可按层选择
**vllm-project/vllm；2026-10-28；main 已合入。** [#61457](https://github.com/vllm-project/vllm/pull/61457)（`af4db2a675f7`）

layer-scoped loader仅把权重导到真正分配FSE slot的层，可保留MXFP8 shared与MXFP4 routed等混合精度，不再全模型all-or-nothing。混合FSE仍拒绝EPLB、EP和nixl_ep等统一expert几何假设；MiniMaxM3/DSv4现在也要求VLLM_ROCM_USE_AITER=1。作者MI350单测扩展到13架构，但无本条通用吞吐数字。建议以startup逐层policy/skip reason核量化例外，不能用配置开关推断所有层已融合。

源码阅读：`vllm/model_executor/layers/fused_moe/utils.py::resolve_layer_fused_shared_expert` → `vllm/model_executor/layers/fused_moe/routed_experts.py`。

### vLLM NCCL weight transfer 支持每参数首owner作为source
**vllm-project/vllm；2026-10-28；main 已合入。** [#62557](https://github.com/vllm-project/vllm/pull/62557)（`2b678ac73916`）

同份ParamMeta派生各broadcaster turn，rank0驱动control，PP stage首owner发本阶段权重，worker按src缓存metadata；单sender wire保持兼容。每rank若跨两个source归属、多个source却无owners等在collective前拒绝。作者4×A100/Qwen3-0.6B两source→TP2、packed/unpacked及跨版本组合通过；某source失败仍可令peer阻塞NCCL，需要整个job终止。建议PP trainer对接先验证ownership计划、turn顺序与故障清理，未宣称容错重试。

源码阅读：`vllm/distributed/weight_transfer/nccl_engine.py::_broadcast_plan、trainer_init` → `vllm/distributed/weight_transfer/nccl_common.py`。

### vLLM 实验 CUDA checkpoint sleep 可保留原graph，边界严格
**vllm-project/vllm；2026-10-28；main 已合入。** [#64186](https://github.com/vllm-project/vllm/pull/64186)（`88d3995ea18d`）

默认关闭CUDACheckpointExecutor，level1 sleep后checkpoint CUDA workers，再按stage并发driver操作/全rank barrier恢复原状态、最后wake tags。要求Linux/NVIDIA、V2/cumem/标准worker、单节点TP，禁止EP、外部cache/weight transfer，并有NCCL/driver/通信环境限制；须完全wake tags再resume，失败重启且管理调用需外部timeout。最终split只跑57helper/executor+22engine例，5真GPU例跳过；旧H20整合版延迟与graph复用结果不等于新split复测。建议排他设备验证反复sleep/wake、真实graph replay计数及restore OOM。

源码阅读：`vllm/v1/executor/cuda_checkpoint_executor.py::CUDACheckpointExecutor` → `vllm/utils/cuda_process_checkpoint.py` → `docs/features/sleep_mode.md`。

### vLLM Transformers backend 接标准 MTP draft helpers
**vllm-project/vllm；2026-10-28；main 已合入。** [#61270](https://github.com/vllm-project/vllm/pull/61270)（`6ba5dbeeb0c1`）

注册TransformersForMTP/MoEForMTP，num_mtp_layers模型用shared MtpModel helpers逐层draft与权重映射，spec_step_idx选择多MTP层。依赖Transformers≥5.20.0相关接口；#49544已10-14合main，不是仍open路线。GLM-4.5-Air/B200/TP4接受率数据早于最后helper重构，需重跑，tiny checkpoint仅CPU加载验证；不能泛称任何HF模型都有MTP。建议先核config层数、版本gate与checkpoint映射，再测真实接受长度/吞吐。

源码阅读：`vllm/model_executor/models/transformers/mtp.py` → `vllm/config/speculative.py` → `vllm/model_executor/models/registry.py`。

### vLLM UMBP 新增 MORI embedded DRAM runtime
**vllm-project/vllm；2026-10-28；main 已合入。** [#61793](https://github.com/vllm-project/vllm/pull/61793)（`6cfb375fd6a2`）

UMBPStoreConnector注册embedded runtime，worker私有DRAM经MORI bulk/ranged I/O直连注册GPU KV，省connector自带host staging；worker-local lookup socket按model/layout及engine实例隔离，带timeout/cancel/eviction retry。PP显式拒绝，本PR主要实现，正文把主要contract/integration覆盖留后续；未提供可继承的端到端性能或多节点容错证据。建议先验证同机多个engine的lookup身份、取消后传输和MORI版本，不把接口接入当生产资格验证。

源码阅读：`vllm/distributed/kv_transfer/kv_connector/v1/umbp/runtime/embedded.py::EmbeddedRuntime、_lookup_socket_path` → `vllm/distributed/kv_transfer/kv_connector/factory.py`。

### vLLM PP stages 取共同KV布局，修异构attention启动
**vllm-project/vllm；2026-10-28；main 已合入。** [#61294](https://github.com/vllm-project/vllm/pull/61294)（`edad8b03c528`）

不再assert每worker支持列表完全相同，而按所有stage支持集求交并合并偏好，未声明布局者不投票，无交集仍报错。作者2×B200、重建10层MiniMax-M3-MXFP8、PP2选LBHNC并服务；只关闭原issue60791的failure1，其余4类未动，且cut未证明与原reporter张量一致。建议混合dense/sparse的PP先验各stage布局及显式环境覆盖。

源码阅读：`vllm/v1/attention/backends/utils.py::_merge_supported_layouts、resolve_kv_cache_layout`。

### 回溯补录：vLLM IPC weight cache key 加入kernel配置
**vllm-project/vllm；2026-10-26；main 已合入；72h/旧候选补录。** [#63506](https://github.com/vllm-project/vllm/pull/63506)（`5a224c1ef349`）

10-26合入，来自旧semantic pending而非本日新增。预处理权重布局取决于MoE/linear backend，key现加入moe_backend、linear_backend及per-quant映射，mismatch在映射tensor前报错或默认回disk，修CUTLASS daemon→auto/TRTLLM engine健康200但乱码。作者B300/Qwen3-30B-A3B-NVFP4验证。key用configured值：双方都auto却因env解析不同仍漏检；auto与显式等效backend还会保守miss。建议Fast Start维护者同步显式配置并清点resolver边界，未关闭整个cache-key audit。

源码阅读：`vllm/model_executor/model_loader/weight_cache/protocol.py::WeightCacheKey` → `vllm/model_executor/model_loader/weight_cache/ipc_loader.py::_fetch_entries`。

### 回溯补录：vLLM 小CPU-tier命中可在同step同步load
**vllm-project/vllm；2026-10-26；main 已合入；72h/旧候选补录。** [#61961](https://github.com/vllm-project/vllm/pull/61961)（`fe5fe8ba5fb9`）

10-26合入旧候选；sync_load_budget_bytes默认0，按每GPU每step有效KV bytes限额，小hit用expedited独立stream并在forward前wait，避免仅1mscopy却排队多步；超预算和无剩余计算继续async。作者Qwen3/GLM的TTFT收益来自更早per-request token阈值+共享stream版本，最终byte-budget/独立stream未端到端重测。建议profile宽EP长step、CPU命中与整batch被同步copy阻塞的权衡，勿直接套用旧219→1ms。

源码阅读：`vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py::SchedulerOffloadConfig` → `vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py::start_kv_transfers`。

### SGLang unified KV 重构收拢布局、分配身份与容量定价
**sgl-project/sglang；2026-10-28；main 已合入。** [#46987](https://github.com/sgl-project/sglang/pull/46987)（`ca8184e13a92`）；[#46988](https://github.com/sgl-project/sglang/pull/46988)（`53592ea5ff50`）；[#46990](https://github.com/sgl-project/sglang/pull/46990)（`d5bc823ad58a`）；[#46976](https://github.com/sgl-project/sglang/pull/46976)（`ccc71fa51486`）；[#46977](https://github.com/sgl-project/sglang/pull/46977)（`5f6075fa2f31`）；[#46975](https://github.com/sgl-project/sglang/pull/46975)（`3355d668ce02`）；[#46971](https://github.com/sgl-project/sglang/pull/46971)（`77936f7b250e`）；[#46972](https://github.com/sgl-project/sglang/pull/46972)（`8044ed4e282c`）；[#47071](https://github.com/sgl-project/sglang/pull/47071)（`5cc9c933fc8a`）

byte storage、logical allocation ID、typed layer view与request-row生命周期逐层分离，shared chain allocator及host band snapshot统一prefill/state-slot容量定价；并修三成员pool遗漏conv slice axis、HiCache completion未传recurrent member，DFLASH unified MHA prefix commit接stride-aware writer/FP8。int8 Mamba checkpoint+unified memory现显式拒绝。主体stack是广泛接口重构，旧H200/GSM8K+0.2–0.9%等数据早于rebase，不能称最终大模型/多硬件性能验证；部分最终CI状态未全绿且本轮未重跑。建议外部backend按ID space、view写位置与completion事件升级，优先回归转移中compact/evict及draft prefix commit。

源码阅读：`python/sglang/srt/mem_cache/kv/alloc` → `python/sglang/srt/mem_cache/prefill_budget.py::ChainPrefillBudget` → `python/sglang/srt/mem_cache/allocator/unified_hybrid_swa.py::set_hicache_transfer_done_event` → `python/sglang/srt/mem_cache/kv/layout/mha.py::set_kv_buffer_prefix_valid`。

### SGLang HiCache page_unified 首接L3 file，拒绝spec与DCP
**sgl-project/sglang；2026-10-28；main 已合入。** [#39728](https://github.com/sgl-project/sglang/pull/39728)（`f65af1cd8377`）

page_unified串起GPU→host staged write-back与JIT load-back，统一key用model namespace与layer/head chunk坐标替代rank拓扑后缀。关键guard：有storage时必须unified key scheme，目前只file backend、完整page/DCP1且spec draft不支持。正文概述的topology-independent key不等于任意拓扑、任意storage都已恢复兼容；有round-trip测试但无性能数据。建议先做相同布局的冷/热restore、命名空间和byte order校验，再考虑异构拓扑。

源码阅读：`python/sglang/srt/arg_groups/hicache_hook.py` → `python/sglang/srt/mem_cache/hicache_storage.py::HiCacheStorageConfig` → `python/sglang/srt/managers/cache_controller.py`。

### SGLang linker load 等在途forward，运行期tail graph改线程局部capture
**sgl-project/sglang；2026-10-28；main 已合入。** [#45580](https://github.com/sgl-project/sglang/pull/45580)（`e105742296df`）；[#47146](https://github.com/sgl-project/sglang/pull/47146)（`88fea372d47b`）

旧pending #45580已合main：record_offload_ready改record_transfer_ready返回TransferReady，Mooncake/UMBP线程与MLA广播在读写页前等待当前/forward完成，防read-done后scheduler先行造成load覆盖仍在读的页。此项源自代码审阅，未实际复现。另DSv4.1运行期row-bucket tail graph改thread_local capture，防linker线程event.synchronize触发global capture错误；作者GB300/DSv4.1-Flash TP2+DSpark复现修复。建议外部linker迁移返回类型/方法名，并分别验证page寿命与首次tail capture。

源码阅读：`python/sglang/srt/mem_cache/unified_cache/unified_cache_linker.py::TransferReady、record_transfer_ready` → `python/sglang/srt/model_executor/runner/row_bucket_graph_runner.py`。

### SGLang Engram auto 优先同NVLink域peer_read，host表继续collective
**sgl-project/sglang；2026-10-28；main 已合入。** [#46955](https://github.com/sgl-project/sglang/pull/46955)（`1825a09dbb25`）；[#47033](https://github.com/sgl-project/sglang/pull/47033)（`54d2f6554084`）

模型构造时一次解析auto并记录：GPU表、TP>1且同完整NVLink域选择CUDA VMM映射peer shard直接读，host表/其他拓扑留collective；all_to_all扩展decode和CUDA graph，非均匀prefill graph保留旧路径。作者GB300/B300/H200测单层，GPU表DP8 decode16时GB30047.7→9.8us；这是lookup层，不是整模倍数。旧all_to_all小DP/小batch可能更慢；peer_read不改变每GPU表shard大小。建议看解析日志/拓扑、跨节点是否同NVLink fabric和CUDA graph输出一致性，勿外推普通网络。

源码阅读：`python/sglang/srt/layers/engram.py::resolve_engram_config、_dp_all_to_all_lookup` → `python/sglang/srt/distributed/device_communicators/custom_all_reduce_utils.py::in_one_nvlink_domain`。

### SGLang ROCm Triton DSA 可用DCP，容量收益不保证交互延迟
**sgl-project/sglang；2026-10-28；main 已合入。** [#44322](https://github.com/sgl-project/sglang/pull/44322)（`edd270fd694a`）

sparse MLA返回LSE合并rank partial，HIP FP8写入与DCP读行一致，MTP draft跟随target sharding/addressing，修GPU fault。作者MI355X、GLM-5.2-MXFP4、TP4/EP4、FP8KV、真实MTP，DCP4与无DCP的400题GSM8K均0.953，容量2.83M→8.81M tokens。AgentX性能用模拟AL3.61、不同日期基线且DCP不兼容HiCache；c16吞吐近持平而交互变差，c48 P90仅原0.41×。建议按工作集是否装HBM、并发与tail latency验收，不照搬最高4.8×吞吐。

源码阅读：`python/sglang/kernels/ops/attention/dsa/triton_sparse_mla_decode.py::triton_sparse_mla_decode_splitk` → `python/sglang/srt/layers/attention/dsa/dsa_dcp.py` → `python/sglang/kernels/ops/kvcache/mla_buffer.py`。

### SGLang DSv4.1 DSpark PD 启动校验放行MoRI
**sgl-project/sglang；2026-10-28；main 已合入。** [#42457](https://github.com/sgl-project/sglang/pull/42457)（`b85d6a8893d7`）

复用已有通用state传输，只把mori加入DSv4.1 DSpark PD允许列表；并行布局/TP仍在CommonKVManager校验。作者同一8×MI355X节点、prefill/decode各TP4、CP1、Flash+DSpark+MoRI，GSM8K400题0.907、AL3.46，110K restore与cold greedy一致。prefill CP未测，本补丁没有新增传输kernel或速度结论。建议沿现有ROCm部署回归target+draft SWA状态与warm restore。

源码阅读：`python/sglang/srt/arg_groups/deepseek_v4_hook.py::validate_deepseek_v41_features`。

### SGLang ROCm KV 存储回避共享allocation与translated-window写错
**sgl-project/sglang；2026-10-28；main 已合入。** [#46861](https://github.com/sgl-project/sglang/pull/46861)（`86f72e5b5c5c`）；[#46983](https://github.com/sgl-project/sglang/pull/46983)（`14fea751244f`）；[#41696](https://github.com/sgl-project/sglang/pull/41696)（`aa8e0a9b6ad5`）

ROCm恢复K/V两份allocation，避免fused allocation后同节点Mooncake PD乱码；作者有9组前后CI二分证据，但未定位HIP IPC精确错地址步骤、该PR正文未独立MI35x复测。unified滑窗需physical translation时另禁fused store，必须经pool写入口；该guard仅CPU验证。AITER DSA decode只传kv_indptr[:bs+1]而非容量，作者MI355X TP4/FP8KV、60K输入/600输出及graph复现fault消失。三处都应优先当正确性回归，非吞吐优化。

源码阅读：`python/sglang/srt/mem_cache/memory_pool.py::MHATokenToKVPool` → `python/sglang/srt/models/utils.py::_window_writes_translated` → `python/sglang/srt/layers/attention/dsa_backend.py`。

### SGLang deterministic RMSNorm融合与gfx950 matmul调优
**sgl-project/sglang；2026-10-28；main 已合入。** [#46159](https://github.com/sgl-project/sglang/pull/46159)（`c6c7533957ab`）；[#44045](https://github.com/sgl-project/sglang/pull/44045)（`1e7e19285d78`）

batch-invariant residual add+RMSNorm按row固定归约、FP32加法/统计融合，特殊cast/variance/padded/FSDP-RL仍fallback，GemmaRMSNorm的Qwen3.5/3.8等不在此路径。另仅gfx950 BF16 matmul改128×64×128/MFMA16，需要约147KB LDS，非AMD其他卡通用配置。作者MI350X、Qwen3-8B BF16/TP1 deterministic，matmul优化1024/1024、c256中3370→3585output tok/s、TTFT略退；两个PR不能把各自增益相乘。建议按固定seed/token/logprob检查batch invariance，再测端到端延迟。

源码阅读：`python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py::_fused_add_rms_norm_kernel、_matmul_persistent_configs` → `python/sglang/srt/layers/layernorm.py`。

### SGLang Triton 3.9 兼容修复覆盖FP8 scalar scale与FLA block pointer
**sgl-project/sglang；2026-10-28；main 已合入。** [#45889](https://github.com/sgl-project/sglang/pull/45889)（`faa56e2b5c5c`）；[#45470](https://github.com/sgl-project/sglang/pull/45470)（`0b158f75e796`）

swap_ab后的scalar权重scale不再按二维索引，修SM90 block-FP8 MoE编译失败；FLA则在advance缺失时用aggregate/plain pointers shim恢复make_block_ptr/advance/load/store，旧Triton不启用。shim是迁移过渡，非所有kernel已移植；作者B300的FLA测试在3.8/3.9同有2个既存KDA bit-exact失败，不能称全绿。R1两节点H100旧基线的单次吞吐差约-0.8%到+0.4%，无加速结论。建议升级torch/Triton时分别回归编译、KDA确定性与full model数值。

源码阅读：`python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py::fused_moe_kernel` → `python/sglang/kernels/ops/attention/triton_block_ptr_compat.py`。

### SGLang Ascend Ops MoE 按prefill/decode选择通信域
**sgl-project/sglang；2026-10-28；main 已合入。** [#46282](https://github.com/sgl-project/sglang/pull/46282)（`bacf65f8f1cd`）

ascend_ops替代ascend_tp入口，prefill可TP AllGather/ReduceScatter或AllToAllV，decode用EP MC2低延迟dispatch/combine；主模型/draft共享解析与NPU-only校验，comm_alg可配置。最终字段是--moe-ascend-ops-config，正文示例较短--ascend-ops-config不应直接复制；fused_op虽在枚举说明出现，代码明确未实现并报错。来源给启动命令和截图，未读取截图数值，故无可靠吞吐/精度数字。建议按实际operator版本、layout/quant mode做prefill/decode分开验证。

源码阅读：`python/sglang/srt/layers/moe/utils.py::AscendOpsConfig` → `python/sglang/srt/arg_groups/moe_hook.py::validate_ascend_ops_config` → `python/sglang/srt/layers/moe/moe_runner/ascend.py`。

### SGLang DSv4 NPU decode 去掉动态shape与D2H同步泡
**sgl-project/sglang；2026-10-28；main 已合入。** [#46805](https://github.com/sgl-project/sglang/pull/46805)（`f59cc7e45fd0`）

C4 loc以cumsum/masked_fill/scatter压入固定上界buffer，invalid行落trash slot再切掉，避免boolean index的aclnnNonzeroV2读回count；request IDs用已有CPU mirror。只在可证明上界的verify/plain decode用新路径，multi-slot reserve/extend保留旧boolean。作者16×NPU/TP16+DP16、bs2 profile非spec50.2→39.2ms；256请求16K/1K分别测spec/non-spec吞吐+15.4%/+9.7%，硬件具体型号未在已核正文列出，不能外推所有Ascend。建议按这些branch测数值、zero-tail与scheduler idle。

源码阅读：`python/sglang/srt/hardware_backend/npu/dsv4/dsv4_common_hooks.py::_compact_c4_loc、maybe_write_dsv4_decode` → `python/sglang/srt/hardware_backend/npu/dsv4/dsv4_allocator.py`。

### SGLang LoRA graph launch 按token预算segment，减少空program
**sgl-project/sglang；2026-10-28；main 已合入。** [#46743](https://github.com/sgl-project/sglang/pull/46743)（`6b697b9aad38`）

CUDA graph原按最大metadata容量launch所有segment，现上界min(capacity,num_tokens//16+adapter_slots)，eager仍实际segment数。作者1×H200/Llama3.1-8B-Instruct、6adapter均匀、400随机512/512 prompts、c32，output2122→2767tok/s（30.4%）；这是单配置作者对比，未报告多次CI或其他rank/长序列。建议多LoRA服务按capture shape/最小chunk做回归，不能外推全模型无adapter场景。

源码阅读：`python/sglang/srt/lora/utils.py::csgmv_segment_grid_size` → `python/sglang/kernels/ops/lora/dense/chunked_sgmv_expand.py`。

### SGLang split-expert MXFP4 checkpoint 接weight-only FP4 Marlin
**sgl-project/sglang；2026-10-28；main 已合入。** [#41289](https://github.com/sgl-project/sglang/pull/41289)（`bb4bb0138172`）

compressed-tensors per-expert *_packed参数有独立scheme，Granite显式退出fused GPT-OSS布局，修加载KeyError；w4a4 checkpoint先注册input scale再丢弃activation quant，实际是W4A16/BF16 activation，不能称保留W4A4计算。显式校验hidden%128、intermediate_per_partition%64，保持既有fused MXFP4默认。作者H10080GB多Granite/TP组合验证加载与文本，无完整质量/速度证据。建议压缩权重用户核实际执行精度和格式，别只看checkpoint标签。

源码阅读：`python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a16_mxfp4_moe.py::CompressedTensorsW4A16Mxfp4MoE` → `python/sglang/srt/models/granitemoe.py`。

### SGLang MegaGate prefill 重新放行，token cap按真实DP份额校验
**sgl-project/sglang；2026-10-28；main 已合入。** [#47148](https://github.com/sgl-project/sglang/pull/47148)（`76d5b3ce02be`）；[#46267](https://github.com/sgl-project/sglang/pull/46267)（`15d39bd8406b`）

图capture不执行break后，opt-in SGLANG_OPT_DEEPGEMM_MEGA_GATE=1可用于extend/target verify，draft extend仍普通router，受每rankMegaMoE buffer cap限制。另修已被DP分割的chunked_prefill_size再次除DP导致错误放行小cap，例如16384/DP2实际8192，不再误认4096。作者GB300/DSv4.1-Flash DP4 graph4096/8192启动通过；本轮未见prefill耗时/质量测量。建议先检查cap与capture/replay，再衡量router收益。

源码阅读：`python/sglang/srt/layers/moe/mega_gate.py::should_use_mega_gate` → `python/sglang/srt/arg_groups/mega_moe_hook.py::validate_mega_moe_token_budget`。

### 分支观察：SGLang NIXL 权重发布仅在sglang-miles
**sgl-project/sglang；2026-10-28；sglang-miles 已合入。** [#35712](https://github.com/sgl-project/sglang/pull/35712)（`6136e6d2bd92`）

已合入sglang-miles，非默认main。target/draft各自注册VRAM并在注册后发布agent metadata/weight table，rollout被动接受peer写入；remote_instance_transfer_engine_info连Mooncake也改transport-tagged dict，外部trainer需同步迁移。禁止同时Mooncake seed、缺nixl与memory-saver，NIXL seed主动加载仍NotImplemented。作者H200测试及Qwen3-30B-A3B TP2/EP2 p2p权重相等依赖miles#2721。建议只作为分支协议变化阅读，不写main/RL rollout普遍支持NIXL。

源码阅读：`python/sglang/srt/model_executor/model_runner_components/remote_instance_weight_transporter.py` → `python/sglang/srt/entrypoints/engine_info_bootstrap_server.py`。

### 本轮覆盖与仍未关闭的范围

六仓main commit及所有base合入搜索已翻页到窗口结束/末页，release集合已读全；候选列表不等于逐条源码审计。slime主窗没有main commit，72h两条中的外部engine重复注册已昨日报道，straw CI修复过滤。其余普通CI、格式、oncall、无新机制的cookbook与重复条目过滤；未选重要候选、巨大kernel/外部依赖仍保留semantic backlog。旧P2P consumer #61667、SGLang非CUDA early-send #43299、VeOmni ROCm #1363仍open；SGLang #45580本次只关闭该PR未合入状态，不关闭所有异步KV寿命风险。

<a id="extended"></a>

## 扩展开源工程与硬件软件协同

以下区分正式release、默认分支commit与功能分支。测试/微测均为来源作者结果，本轮仅核原始正文、补丁和固定版本，未做GPU/NPU/TPU复现。

### PyTorch 2.15 正式发布：先验收通信、梯度与检查点的迁移契约

**pytorch/pytorch；本轮增量。** 本轮新证据是正式发布，不把此前已报道的开发提交重新算作新算法。默认 `nccl` 切到重写后端；每组绑定单个 CUDA device，创建组会提前分配通信资源，依赖 `torch.cuda.set_device()` 或单进程多卡的旧程序需复核。独立 peer P2P communicator 可选 `nccl-lazy`；`nccl-legacy` / `TORCH_DIST_USE_NCCL2=0` 只保留本版，官方计划下一版移除。flight recorder 也覆盖新后端失败。来源：[完整正式 release notes](https://github.com/pytorch/pytorch/releases/tag/v2.15.0)，tag 基线 `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`，发布于10-28 17:09:49 UTC。

最直接的训练行为变化：FP32 原参数配 BF16 `param_dtype`、且 `reduce_dtype=None` 时，FSDP2 现在按原始 `grad_dtype` 进行梯度累积与 reduce-scatter；典型通信量从 BF16 变 FP32、字节数翻倍。显式设 `reduce_dtype` 才能保持旧口径。DCP `.metadata` 增加 strides，**2.15 写出的检查点不能被2.14及更早版本读取，且无旧格式开关**；新版本仍可读旧文件；已核tag的[TensorProperties序列化](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/distributed/checkpoint/metadata.py)与[MixedPrecisionPolicy](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/distributed/fsdp/_fully_shard/_fsdp_api.py)。手动 `finalize_backward()`、保留 unsharded storage、流水激活生命周期分析和多轮 schedule 梯度累积提供更细的峰值/重叠控制，但每项都有额外持有内存或调用时序责任。

推理与平台侧有 NVGEMM NVFP4、默认 nested-reduction fusion（如 RMSNorm+量化）、实验 NIXL 单边传输与图捕获；Mac 提供 FP8、grouped_mm、公开 Stream API，MPS 最低升至 macOS15；默认 CUDA13.2，另提供13.4实验 Rubin wheel，ROCm10 wheels以TheRock包作依赖。各平台与微测收益不能相互外推。**建议深入阅读并做升级回归**：优先检查进程组 device、通信显存、梯度dtype、恢复/回退路径及编译数值，随后再评估速度；本轮未安装发行包或运行GPU。


### DeepSpeed：从活跃 ZeRO 存储直接规划训推权重同步

**deepspeedai/DeepSpeed；本轮增量。** `zero_affine.py`为ZeRO0/1/2低精度副本、ZeRO3 `ds_tensor`以及FP32 master/optimizer states描述全局rank上的仿射存储映射；`nest_affine_maps`将TP/EP布局和实际存储组合，`plan_transfer`只接收在flat shard中连续的N维piece，避免先聚合完整参数。每片用至多 `2·ndim−1` 个box表达连续区间，降低按行枚举的规划爆炸；不改变ZeRO算法或executor。

作者Qwen3-8B、8×L20、训练AutoTP2→rollout AutoTP4、ZeRO2/3同步逐次bitwise一致。ZeRO2 gather重复同步0.78–0.82s，示例balanced选择0.49–0.55s、默认lowest却1.05–1.35s；新增设备内存3.48GB→0.16GB或0.30GB，另有一次map+plan约1.2–1.4s成本。**收益依赖副本选择，不是任意映射自动提速**；optimizer offload明确NotImplemented，MiCS/hpZ不覆盖。建议考虑复现真实训推拓扑、规划复用和同步bitwise，再决定采用。

来源：[8885 / f8185049](https://github.com/deepspeedai/DeepSpeed/pull/8885)（2026-10-28T16:16:50Z）。

### TRL AsyncGRPO：过滤零 advantage 后，epoch 与恢复必须同时计入被丢弃组

**huggingface/trl；本轮增量。** 可评分rollout少于2条或其奖励全相同的组在worker端返回空sample，改传裸group_id；trainer继续取到 `samples_per_step`，但consumed按trained∪dropped计数，checkpoint恢复也找首个未消费prompt。这是不回填固定组数的动态采样；每步实际保留组数与生成消耗会变化，新增 `num_generations≥2` guard。

已读最终 `_score_group`、队列协议和epoch/save patch：`reward`统计所有可评分结果，`reward_kept`才对应进入训练的sample，`frac_reward_zero_std`记录过滤比例。**原PR前文仍说reward只覆盖训练组，已与最终代码不符；本条以代码为准。** 代码审阅还发现epoch只在step_end检查、worker可循环数据，因此持续全过滤/凑不满step时未必能自动停止；这是静态推断，未运行复现。建议考虑采用并回归全过滤停机、部分NaN、恢复和耗尽场景，重新建立reward与prompt预算统计；没有独立完整训练质量或提速证明。

来源：[7609 / e29597af](https://github.com/huggingface/trl/pull/7609)（2026-10-28T16:00:01Z）。

### TRL：截断采样的概率支持集与 LoRA rollout 模型名开始强制对齐

**huggingface/trl；本轮增量。** AsyncGRPO拒绝 `min_p>0` 且 `top_k=0`（此前top_p已有同类限制）：vLLM按截断候选集归一化，trainer若仍按全词表归一化会使importance ratio有偏，需通过top_k返回并重放候选集。另给外部注入的Harbor rollout worker写入LoRA名称，并在每次session前设置版本化模型ID，修复训练已同步adapter、rollout却一直请求基座的静默错误。

这是两项训练正确性修复；guard不等于所有采样参数已无偏，LoRA容量仍需容纳允许staleness内的版本及正在加载的版本。建议优先核查历史实验的请求model字段、候选集与概率分母，再考虑采用；来源是配置/worker代码和单元回归，无本轮GPU训练复现。

来源：[7937 / c8a5b644](https://github.com/huggingface/trl/pull/7937)（2026-10-28T14:02:26Z）；[7948 / 7fdad81d](https://github.com/huggingface/trl/pull/7948)（2026-10-28T10:18:09Z）。

### PyTorch main：RCCL 对称内存与 TokenSwitch 路由权重梯度

**pytorch/pytorch；本轮增量。** 默认分支实际导出commit新增ROCm上的NCCL/RCCL symmetric-memory路径：需RCCL≥2.30.4且device header通过host编译probe；capture内新分配再需≥2.30.7，NCCL_CUMEM_ENABLE/NCCL_WIN_ENABLE须在comm初始化前设置，缺device API的handle仅host侧可用。另让TokenSwitchNCCL把接收端top-k权重梯度经native backward combine送回源router的 `(token,slot)`。expert ID仍离散；其他backend的默认hook不自动获得同等权重梯度能力。已读 `_DispatchAutograd.backward`、NCCL binding与对称内存改动。

两项PR页面仍显示未合入/ghstack状态，而本轮默认分支commit已实际存在，因此以下**以导出commit为准，不能写成PR已merge**；也未确认进入2.15 tag。建议深入阅读并回归MoE router梯度、对称分配/窗口及多卡错误路径；无通用性能数字。

来源：[b8c7d2c25 / b8c7d2c2](https://github.com/pytorch/pytorch/commit/b8c7d2c256dc8a4d4f2619ed07423c3d152d38af)（2026-10-28T19:39:52Z）；[c5c9a0071 / c5c9a007](https://github.com/pytorch/pytorch/commit/c5c9a00715c54fdf9e3498c529cd8e7fe714d636)（2026-10-28T08:52:51Z）。

### PyTorch Inductor：PDL 等待按原始 buffer 身份处理原地修改

**pytorch/pytorch；本轮增量。** 此前把original→version的 `mutation_renames` 当反向映射，导致依赖等待没覆盖前一kernel的原地写；现在通过 `mutation_real_name` 对读写双方归一，修复load-after-store及store-after-store竞态，并把reduction store传入正确buffer名。建议相关PDL用户考虑采用并做链式原地更新数值回归；这是实际main commit，PR页面仍open，未验证2.15正式包包含。

来源：[bb922d74b / bb922d74](https://github.com/pytorch/pytorch/commit/bb922d74b9b171cbcb1534cd481347e402877a50)（2026-10-28T16:27:08Z）。

### PyTorch Inductor：导出模块保留编译期 autotune 的实际选择

**pytorch/pytorch；本轮增量。** `triton.autotune_at_compile_time`曾在编译时调优，导出Python module后却再次调优，可能换config乃至归约顺序。`KERNEL_CONFIGS`现在携带每个module-level kernel的已选配置，并在资源不足时回退重新搜索；template与用户Triton kernel不强制pin。调优执行块不再混入运行模块，captured TMA样本还须包装为descriptor才供调优使用。

建议深入阅读并回归模块独立执行、TMA别名、多GPU资源不匹配及数值一致性；不是跨硬件永不调优的保证。本轮以实际main导出commit为证，原ghstack PR状态不能替代该证据，未确认正式发行包内容。

来源：[e5d366e72 / e5d366e7](https://github.com/pytorch/pytorch/commit/e5d366e72c1db38ee16c7e449202795dced551b0)（2026-10-28T05:07:32Z）；[1570da445 / 1570da44](https://github.com/pytorch/pytorch/commit/1570da445593ed017b5bf417a5a277f9515520d4)（2026-10-28T05:07:19Z）；[b8d638946 / b8d63894](https://github.com/pytorch/pytorch/commit/b8d6389461eb57f5b21c2b6fddcb18622702fd90)（2026-10-28T05:54:47Z）。

### TransformerEngine：packed CP 避免共享内存超限，Hopper FA4 提前拒绝已知挂起组合

**NVIDIA/TransformerEngine；本轮增量。** THD CP七类helper按设备默认每block共享内存上限选择缓存 `cu_seqlens`，过大时直接读原global表并显式除数，避免文档数多时launch失败，不另分配global scratch。共享路径仍预缩放，不能遗漏两种表示的除数差异。

另在SM90、FA4、需要反向、Quack≥0.6.0且<0.6.4、CUTLASS DSL≥4.6.2时明确RuntimeError，含cache命中选择，防已知backward挂起；推理或其他组合不触发该guard。作者H100本地134检查但测试留本地，并非仓内可复现全套。建议考虑采用并固定兼容依赖，同时回归多文档CP与checkpoint/compile组合；该guard不等于上游内核本身修好。

来源：[3698 / e619cea1](https://github.com/NVIDIA/TransformerEngine/pull/3698)（2026-10-28T20:40:44Z）；[3762 / 54ecaabc](https://github.com/NVIDIA/TransformerEngine/pull/3762)（2026-10-28T21:11:40Z）。

### TransformerEngine：scale swizzle 改布局搬运，新增16字节对齐契约

**NVIDIA/TransformerEngine；本轮增量。** 列方向chunk kernel扩到宽scale；行方向full-width用于窄scale，17–30KiB共享内存区按行tile数量与GPU SM数决策，更宽回旧kernel，NVFP4也接入。最终接口强制input/output scale指针16-byte对齐，外部view调用可能从过去未检查变为报错；grouped列路径仍保留旧宽kernel，不能概括全部替换。

作者Blackwell/Rubin微测中，4096×1792 MXFP8 rowwise 25.265→1.806µs，4096×7168却约1.00×；数字是布局算子而非GEMM或训练。建议考虑复现实际scale宽度、alignment、SM数量与全步开销，不能用最高13.99×代表常见LLM吞吐。

来源：[3682 / e614b7a9](https://github.com/NVIDIA/TransformerEngine/pull/3682)（2026-10-28T18:50:20Z）。

### TensorRT-LLM 1.4.0rc2：本轮新增的是预发布交付边界

**NVIDIA/TensorRT-LLM；本轮增量。** 10-28 15:51:34 UTC 发布 [v1.4.0rc2](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.4.0rc2)，tag `b96ba434e8a05e2ab52d2082bd8d70b1d844739b`。changelog汇集部分此前已报的混合精度KV、模型/多模态与MoE修复；此次不把它们重新计为新算法。它是RC，随后main的Kimi/Helix/GLM和native库改动须按各自commit核对，不能自动继承到这个tag。建议考虑固定RC做兼容验证，并核实际wheel/container及未解决限制；无一项统一的版本提速结论。


### TensorRT-LLM Helix：先做 V 投影再 all-to-all，减少通信也改变数值重排

**NVIDIA/TensorRT-LLM；本轮增量。** MLA的按head V投影与CP加权求和可交换，交换前把每head512 latent降到128，传输/合并payload降4倍；FP8 block-scale权重需要额外BF16 V拷贝，例如DeepSeekV3 tp1/cp8约854MiB/rank，且运算重排非bit-exact。FIFO发送端新增generic→async shared fence，防复用buffer被下一bulk load抢先覆盖；该fence是新投影路径正确性的必要条件。prompt按本rank块索引拆分也由O(prompt)降至O(prompt/cp)。

作者GB300小CP层微测有0.52–1.01的新/旧耗时；部分形状无加速。DeepSeekV3-Lite在8×B200四参数化评估均过阈值，但不是逐token相等。整组8PR的+23.6%请求/s不能归因于本项。建议深入阅读并复现sender fence、额外权重显存、CP拓扑与实际质量。

来源：[20939 / b613fbe3](https://github.com/NVIDIA/TensorRT-LLM/pull/20939)（2026-10-28T06:14:04Z）。

### TensorRT-LLM：P/D 重试、共享前缀等待与异步状态写入一起收紧

**NVIDIA/TensorRT-LLM；本轮增量。** generation leg仅在明确未送达时重试并保持request ID，避免服务器已接收的请求变成收不到KV的僵尸；context-first无TxSession项过120s TTL后发FAILED，generation-first仍静默逐出。V2 scheduler对正在prefill且比现有reuse深≥64块的prefix有界等待，默认block reuse时启用，上限128 scheduler passes，多模态排除。ConversationRouter按特异的共享前缀放置新会话（默认32块），公共system prompt仍负载均衡。

all-layer KDA replay初始化合批，slot清零改stream-ordered操作，大page H2D/D2H最多16CTA；均应按最终diff与回归核实。**正文旧标题中的KV-balanced attention-DP admission已移到#21073，不属于本PR**；本条不把全8PR系列性能表归因到这个合入。建议优先回归重复投递、过期、PP scheduler次数及prefix owner离开等故障，关注默认等待对尾延迟的影响。

来源：[20938 / deebe2be](https://github.com/NVIDIA/TensorRT-LLM/pull/20938)（2026-10-28T16:25:51Z）。

### TensorRT-LLM：跨 step 专家预取改循环计划，slot 数换取 KV 容量

**NVIDIA/TensorRT-LLM；本轮增量。** DWDP按连续forward的MoE层构成循环预取计划，处理重复MTP层；`num_prefetch_slots`默认2、最少2，N槽提前N−1层。KimiK3 DWDP16每多一槽约需一层远端专家15.6GB，在KV估算前分配，因此减少KV池。权重更新与sleep/wakeup必须invalidate，DWDP明确拒绝CUDA graph配置。KDA prefill按常驻gather占用保留SM，MTP verify新增v2/双CTA调度及受限MXFP8输出。

作者GB300冷L2、dep16/b8/MTP7、BF16 state的单call 52.24→33.24µs；属于kernel，不是整模。Rubin跨step预取与不同slot的吞吐有特定配置结果，整8PR AgentX表不作为单PR收益。建议深入阅读并复现SM reserve生命周期、预取失效和容量/并发权衡；显存紧张时增加slot可能反而影响吞吐。

来源：[20941 / 5be33342](https://github.com/NVIDIA/TensorRT-LLM/pull/20941)（2026-10-29T00:42:38Z）。

### TensorRT-LLM modeling_v2：Kimi 原生目标进 main，native 库开始按模型家族隔离

**NVIDIA/TensorRT-LLM；本轮增量。** KimiK3剩余native modeling迁至模型家族下，共享kernel单份、目标特有SSM/MoE保留本地；当前MXFP4/SM100/TP16两目标为MoE TP4×EP4或TP16×EP1，text-only加载和形状/精度限制继续存在。重放计数先compact再回写原aliased storage，避免非单位stride误读。

另将native C++/CUDA按family建库、目标命名空间注册、按arch打包并lazy load；缺库/未build为NoTarget可fallback，已安装但加载损坏为NativeLoadError不可静默fallback。all-rank目标resolve先达成一致，避免部分rank走其他目标后collective挂起；`modeling_v2`仍默认off。K3作者16×GB200、1024/1024、DSpark7、**强制AL6**测试相对既有优化stack，TPOT反而高1.4–5.0%，whole-run吞吐约−0.7%至+3.4%；自然接受/完整精度与部分目标未完成。建议深入阅读架构隔离与失败语义，继续观察完整qualification，不据强制接受proxy宣称自然语言部署加速。

来源：[20893 / b72f1c1e](https://github.com/NVIDIA/TensorRT-LLM/pull/20893)（2026-10-28T16:59:46Z）；[20896 / 7723a3b3](https://github.com/NVIDIA/TensorRT-LLM/pull/20896)（2026-10-28T12:56:07Z）。

### TensorRT-LLM GLM-5.3-Flash：默认 MoE 与 FP8 indexer 选择带来有条件收益

**NVIDIA/TensorRT-LLM；本轮增量。** Blackwell上AUTO将相应NVFP4 Glm5Next优先送TRTLLM-Gen，不能服务的unquantized MTP层仍回退；packed context indexer改DeepGEMM FP8打分，generation/verify路径不改，`TRTLLM_GLM_INDEXER_BF16=1`可还原。共享row metadata、去transpose和每层小kernel，且补workspace reserve避免KV估算过量。

同节点4×B300、TP4、MTP3、>10K prompt/短回答、FP8 KV、三轮：NVFP4 EP4 C1 TTFT260→230ms、用户输出364→436tok/s，C8吞吐137.4→153.8；FP8 EP1仅约2–3%提升。LongBench-v2 short180题FP8 indexer56.11 vs BF1657.78，作者sign test不显著，不能称质量绝对无损。低M GEMM保留opt-in，因NVFP4启动多约6.5min而未默认开。建议考虑同负载复现吞吐、长上下文质量与启动成本，不外推所有GLM版本/负载。

来源：[20755 / 2fe91cdf](https://github.com/NVIDIA/TensorRT-LLM/pull/20755)（2026-10-28T08:22:46Z）。

### TensorRT-LLM：NVFP4 权重语义与大 prefill 索引两项正确性变化

**NVIDIA/TensorRT-LLM；本轮增量。** MiniMaxM3加载compressed-tensors时把weight_packed改名，global scales取倒数以匹配TRT约定，保留ConsumableWeightsDict的释放所有权。零/负/非有限global scale置0并汇总警告；零权重scale会令相应专家输出归零，负或NaN应查checkpoint，不能掩作正常校准。部分量化的MIXED_PRECISION也按实际键名转换。作者验证以CPU/stub为主，未做GPU serving。

Kimi persistent attention-residual候选偏移改64bit，解除N9时42799-token上限；此前有guard，会报错而非已证明静默损坏。仅显式persistent/split拓扑相关，默认per_token不走；仍限B1/H7168/N2..9等契约。建议考虑采用，分别回归完整模型加载和跨旧边界prefill；本轮未验证所有硬件。

来源：[20694 / 1f9b7072](https://github.com/NVIDIA/TensorRT-LLM/pull/20694)（2026-10-28T18:07:33Z）；[21011 / e2b15441](https://github.com/NVIDIA/TensorRT-LLM/pull/21011)（2026-10-28T10:33:29Z）。

### TensorRT-LLM：多 frontend 单一服务所有权与 GDN 输出缓冲

**NVIDIA/TensorRT-LLM；本轮增量。** 多frontend监听同一端口时，iteration stats、profile、batch store与liveness转由launcher持有并转发，共享Prometheus multiprocess目录，避免每次scrape随机读到1/K片段、launcher失效后其他frontend仍健康。作者8frontend/64请求中success计数从多种6/7/8/10变为32次scrape皆64；部分路线只在旧container得到launcher404，不代表已跑通新版profile业务。建议优先回归采集、转发与kill/freeze故障；它修观测完整性而非模型速度。

另一独立GDN改动让eager mixed prefill/decode写同一输出buffer，省cat拷贝；B200、4096 prefill+32单token decode，隔离core span降4.10%，普通eager仅0.6–1.5%。建议按真实提交间隙/后端测端到端，不把captured core当请求时延。

来源：[19394 / 0cd0a183](https://github.com/NVIDIA/TensorRT-LLM/pull/19394)（2026-10-28T10:39:24Z）；[21132 / 99466ea9](https://github.com/NVIDIA/TensorRT-LLM/pull/21132)（2026-10-28T08:49:56Z）。

### FlashAttention4 beta37：此前 K/V 尾部修复终于有预发布 tag

**Dao-AILab/flash-attention；本轮增量。** 10-28 08:14:55 UTC的 [beta37 release](https://github.com/Dao-AILab/flash-attention/releases/tag/fa4-v4.0.0.beta37)与[beta36→37逐commit比较](https://github.com/Dao-AILab/flash-attention/compare/fa4-v4.0.0.beta36...fa4-v4.0.0.beta37)已打开。新tag `98c1cee0ed59307833841cdbf567fe336ceec13a`包含#2983/#3012/#3013/#3071：此前日报已报的SM100/SM90未使用K/V NaN污染及ragged TMA descriptor边界闭环，增量是可固定版本安装的交付；不重算这些旧PR的新颖性。

另包含HD256 GQA backward pack及2CTA CLC坐标修正；本轮未逐项复现，不把所有head/layout性能一概承诺。它仍是FA4 beta，且不消除上文Quack/CUTLASS组合的独立依赖风险。建议相关packed-varlen/缓存服务先固定tag回归NaN尾部、分段边界、梯度和2CTA调度，再考虑采用。


### Triton：MoE matmul 的 −1 gather 必须连 scales 一起屏蔽

**triton-lang/triton；本轮增量。** 输入gather的−1表示无活跃行，原先仅对gather数组位置取模，并不能使被索引行有效。普通和persistent matmul现在分别mask values、tensor/MX scales，inactive贡献为0，bias/epilogue仍应用；torch参考也不再把−1解释为最后一行。Hopper E8M0 padding使用中性127，其他路径保留适配padding以免额外spill。

作者H200 sentinel probe此前读到相邻行448而非0，GB300 scales读也失败；修后相应用例通过。证据证明非法读取和潜在IMA风险，**未观察到该probe发生IMA崩溃，也未确认生产影响**。建议考虑采用并回归expert空路由、值/scale不同布局，不能只屏蔽输出；这是3.9发布后的main增量。

来源：[12538 / dcbd6631](https://github.com/triton-lang/triton/pull/12538)（2026-10-28T22:35:31Z）。

### Triton：CTA 共享副本落盘与 AMD strict scaled-upcast

**triton-lang/triton；本轮增量。** `local_store`为源layout未覆盖的shared-memory CTA replica补寄存器basis并broadcast，同一store layout同时用于barrier插入和ConSan；padded shared layout的CTA复制要求block basis为零。另一改动解除gfx1250-strict的硬件block32 scaled-upcast gate，并按实际pk8/block16选择调整assert。

建议编译器/算子开发者深入阅读layout与同步回归，硬件路径还受LLVM与目标能力约束；两项是能力/正确性落地，没有可引用的通用LLM速度或吞吐结果，不等同所有AMD低比特kernel已受益。

来源：[12535 / 51a27535](https://github.com/triton-lang/triton/pull/12535)（2026-10-28T17:38:09Z）；[12499 / 8afd743d](https://github.com/triton-lang/triton/pull/12499)（2026-10-28T15:45:56Z）。

### FlashInfer KDA KCP：工作区复用与关闭默认 debug probe，压低首步峰值

**flashinfer-ai/flashinfer；本轮增量。** 前反向send pool按互不重叠生命周期共用storage，ring pool首次ring调用才分配；字节查询统一为真实持有量，纠正旧估计最多低约8%。更关键的是 `spec3_probe`由默认record改off，避免首次checkpoint backward保留五张token平面；graph replay升级arena先清除pin再释放旧池，分配失败时不得沿用已失效的fast-path状态。

作者GB300、PyTorch2.14.1+cu130、249856tokens/4文档/H64、CP2/4、causal-conv+KDA一次非重入checkpoint训练步，CP2 contiguous首步61175.9→50218.6MiB，CP4 per-doc zigzag31110.1→25555.5；测量从inputs构建后起算，**包含保留workspace但不是整模训练显存**。稳态workspace和per-call峰值在第二项不变，首步已接近两者之和；GB200字节亦核。建议考虑复现冷启动、graph升级与失败恢复，调试时显式record且各rank一致。

来源：[7714 / e4c7e4a6](https://github.com/flashinfer-ai/flashinfer/pull/7714)（2026-10-28T14:40:22Z）；[7762 / 24527775](https://github.com/flashinfer-ai/flashinfer/pull/7762)（2026-10-28T22:29:54Z）。

### FlashInfer：HD256 packed GQA 增加实验训练接口

**flashinfer-ai/flashinfer；本轮增量。** `flashinfer.gqa_train`交付cu_seqlens打包、每segment右下对齐causal、Q/K长度可不同的前向及一阶反向；仅BF16输入输出/FP32累积，dK/dV可FP32窗口、按int32 row-map加到caller-owned FP32目标或BF16。最终guard还要求Hq/Hkv为偶数、Q/K/V最后维stride1、base和token/head stride满足16B对齐；MHA ratio1或GQA ratio3会拒绝。预编译SM100/103/107，不能外推其他head/dtype或高阶梯度。

作者GB200/CUDA13.4、Hq24/Hkv4/D256、activation checkpoint的单层2forward+backward（含初始化），9种packed/CP布局相对FA4特定开发分支+FP32 index_add workaround，几何均值1.029×、总时间1.020×，不是大幅全训提速。建议深入阅读并复现CP梯度累积与相同baseline，明确实验API兼容成本。

来源：[7774 / 509d751b](https://github.com/flashinfer-ai/flashinfer/pull/7774)（2026-10-28T23:20:46Z）。

### FlashInfer mHC：跨 token 数复用 workspace 的 ticket 固定在偏移0

**flashinfer-ai/flashinfer；本轮增量。** ticket原来排在T相关regions后；大T调用留下的FP32/BF16内容会在小T布局中被当调度计数，负数可触发sticky illegal instruction，小正数可静默跳工作。现在前反向ticket都放第一个region、偏移恒0，bytes envelope和kernel不改；caller首次须零初始化，之后kernel保持ticket为0。

作者B200/GB200/GB300真实复现同bucket大→小及graph warmup崩溃，新增#7784专门覆盖7832→7435默认workspace，修后bitwise对fresh workspace。某些GLM尺寸/SM107等路径不受影响，不能声称全模型曾错。建议优先采用相关修复并回归跨长度工作区、capture/replay；单次固定T通过不足以排除该bug。

来源：[7777 / e371b403](https://github.com/flashinfer-ai/flashinfer/pull/7777)（2026-10-28T23:23:25Z）；[7784 / 9288061a](https://github.com/flashinfer-ai/flashinfer/pull/7784)（2026-10-29T00:51:20Z）。

### FlashInfer PDL：设备生成的 scale/temperature 必须在等待后读取

**flashinfer-ai/flashinfer；本轮增量。** 量化、RMSNorm融合量化及softmax的scalar load移到 `griddepcontrol.wait` 后，并给wait加memory clobber防编译器上提；索引计算仍可与producer重叠。另对两款small-M BF16×FP4 Cake程序把alpha改成等待后的coherent load，避免非一致read-only load提前取旧值。

作者late-store负控在RTX5080旧8fail/新8pass、B200旧9fail/新10pass；对SM120 SASS也检查global load顺序。性能并非零代价：例如tiny softmax1×4096为1.56→1.69µs；两款alpha程序约1.007–1.018×变化是微测。建议优先回归动态scale链，明确PDL的“允许launch”与“数据已就绪”是不同信号；本轮仅源码/测试审阅。

来源：[6322 / 87071228](https://github.com/flashinfer-ai/flashinfer/pull/6322)（2026-10-28T10:33:54Z）；[7738 / 464ca3c8](https://github.com/flashinfer-ai/flashinfer/pull/7738)（2026-10-28T15:06:09Z）。

### FlashInfer attention：页表 ring 与复用 workspace 的计数器都要跨阶段闭环

**flashinfer-ai/flashinfer；本轮增量。** CuTe GQA paged decode的K prefetch领先V，活跃页表槽需 `prefetch_iters+3`，默认由4增为5；原来多token预测/长循环能错读s+4的V。split-free reduction另修无LSE时sink归一化与非2幂head-group的越界LSE写。作者B200压力90配置中旧30错、修后3600launch无错，单kernel时间比0.996–1.004。

Hopper FMHAv2在末个terminal ticket内复位共享调度counter，含FP8 ragged提早结束路径，防后续XQA读到非零；保持stream依赖，不加单独post-clear。建议相关服务考虑采用并回归prefill→decode、同buffer复用和图重放，不能从单attention输出正确推断整个服务协议无竞态。

来源：[6299 / 3c7e0755](https://github.com/flashinfer-ai/flashinfer/pull/6299)（2026-10-28T08:50:41Z）；[6203 / 32a8e5a7](https://github.com/flashinfer-ai/flashinfer/pull/6203)（2026-10-28T09:11:43Z）。

### FlashInfer MoE EP：发送前量化与小步 Lamport transport

**flashinfer-ai/flashinfer；本轮增量。** split layer新增quantize/dispatch/compute/combine四阶段，NVFP4/MXFP8及DeepSeek块FP8可在源rank量化连scale一起发；receiver复用scale而非重算。Prims-TS可做compute backend，但FP8 per-tensor缺校准global scale仍排除，DeepSeek量化目前是PyTorch实现，不能只按payload缩小推测整层加速。stage_hook可在graph中插event，分段event会阻止PDL相邻阶段重叠，分段和不能冒充完整span。

`NVLinkOneSidedAlltoAll`对SM100/103、≤64rank、BF16 combine的小decode步增加Lamport数据到达即poll/reduce路径。默认auto在非强制CFT时启用，按所有rank一致的runtime token上限选路；topk_weights全rank传或全不传是**未动态检测的caller前提**。GB300、DeepSeekV3/BF16、EP8/T1 dispatch+combine GPU span43.6→15.7µs，EP64/T12884.3→70.3，另增52.4–418.9MiB Lamport区，64MiB是选路速度阈值而非总显存预算。建议深入阅读并复现一致选路、空rank/graph及持有workspace成本，未证明整模端到端收益。

来源：[7426 / b1cf7660](https://github.com/flashinfer-ai/flashinfer/pull/7426)（2026-10-28T11:03:23Z）；[7435 / 9eb51e64](https://github.com/flashinfer-ai/flashinfer/pull/7435)（2026-10-28T18:29:41Z）；[7498 / 1f306300](https://github.com/flashinfer-ai/flashinfer/pull/7498)（2026-10-28T09:31:14Z）。

### FlashInfer b12x W4A16：避免有限但错误的下溢，以及未路由 scratch 参与求和

**flashinfer-ai/flashinfer；本轮增量。** BF16 E8M0 scale重偏置7→71，MMA内部缩放从2^-119变2^-55，补偿相应调整；可正常表达的scale-byte窗改为57–183，高值钳至183。旧路径部分真实checkpoint约111–117会下溢而aggregate cosine仍很高；**FP16 activation的同类范围问题仍留后续**，NVFP4/W4A8不走这个函数。

另map-free top-k sum跳过expert_id<0，防FC2未写的scratch被加进输出；不一概拒绝正越界ID，因为有expert map时可能是global ID。RTX5080新8例旧全错/新全过，但同环境147项既有失败仍在，两条其他动态caller是否被测未确认。建议考虑采用并回归真实scale分布、NaN-scratch与远端专家−1路由，勿只用cosine阈值验低精度正确性。

来源：[6317 / 30cd27ea](https://github.com/flashinfer-ai/flashinfer/pull/6317)（2026-10-28T09:00:16Z）；[6294 / f083cef0](https://github.com/flashinfer-ai/flashinfer/pull/6294)（2026-10-28T08:59:37Z）。

### FlashInfer：EP autotune 缓存一致性与小 top-k 的 DSA 工作区

**flashinfer-ai/flashinfer；本轮增量。** CuTe DSL MoE持久cache key去掉rank相关 `local_expert_offset`，避免一rank命中跳过timing all-reduce、其他rank仍等待而暖启动挂起；in-memory hash不改，旧key首次会miss/re-tune，num_experts等形状约束仍区分。另SM100a DSA indexer在top_k≤512把candidate floor恢复8192（≥1024仍16384），GB200/152SM pooled top512工作区114→57MiB；不改kernel/结果，延迟差在A/A噪声内。

建议分别回归多rank冷/暖cache、不同offset与DSA top-k边界；前者是协议正确性，后者是workspace减半，不等于KV容量或请求时延减半。

来源：[6139 / 9ef2e888](https://github.com/flashinfer-ai/flashinfer/pull/6139)（2026-10-28T08:56:06Z）；[7767 / ee385388](https://github.com/flashinfer-ai/flashinfer/pull/7767)（2026-10-28T22:35:10Z）。

### AITER：真实 token 数、−1 expert 与向量尾部的组合边界

**ROCm/aiter；本轮增量。** MXFP4 inline-sort接入device `num_local_tokens`并忽略无效expert，保留调优行；多阶段sort尾vector越界时逐元素载入、补−1，量化sort的padding slot不能零写到out之后。gfx1250 grouped TP MoE对−1在counter与多目的量化两处都须跳过；另per-tensor量化向量尾读写受边界约束。这些是不同kernel的修复，不能拿一项测试覆盖全体。

作者MI355X/DeepSeekV4.1Flash/ATOM固定overlay、TP2 AgentX约36000请求未再出现相关IMA，仍有0.02–0.09%请求错误；该实机树早于最终rebase。KimiK3 tuned MXFP4的M1传count单call36.6→21.2µs，尚无修后全服务速度结果。建议相关ROCm服务优先回归padding/count/graph重放，记录caller与AITER配对版本；绕过传count可能换来−1路由越界，不能当等价修复。

来源：[7020 / f2d9434b](https://github.com/ROCm/aiter/pull/7020)（2026-10-28T13:58:27Z）；[7085 / c90842ad](https://github.com/ROCm/aiter/pull/7085)（2026-10-28T13:27:39Z）；[7004 / 58c6c346](https://github.com/ROCm/aiter/pull/7004)（2026-10-28T02:56:45Z）；[6677 / bbef5a6f](https://github.com/ROCm/aiter/pull/6677)（2026-10-28T07:26:17Z）。

### AITER：按固定 rank 顺序累加，最终范围比 PR 说明更广

**ROCm/aiter；本轮增量。** 改变rank轮转的FP32加法顺序可令相同BF16行因batch offset/owner不同落到相邻ULP；最终diff统一多个custom all-reduce/LL/LL128、reduce-scatter及量化/归一化子路径的rank0→N−1顺序。**PR正文仍把reduce-scatter、QK norm、FP8与gfx1250列为未修，但最终patch实际修改这些中的多条路径，不能照抄旧清单。**

已读两份header与新增四rank回归源码；未发现可确认的实机通过记录或带宽结果，不能说所有硬件已验证。RCCL fallback等未审阅依赖不继承bitwise承诺。建议深入阅读并复现不同row offset、message size、所有rank输出、graph及性能，再考虑采用确定性要求的训练/服务。

来源：[6954 / b6a9d0ae](https://github.com/ROCm/aiter/pull/6954)（2026-10-28T04:37:44Z）。

### AITER gfx1250：四 rank Gluon all-reduce 接入，默认仍需 opt-in

**ROCm/aiter；本轮增量。** 新Lamport inbox/tag数据路径支持四rankBF16规整消息，communicator用 `AITER_GLUON_FUSED_AR=1`选择，普通AR≤4MiB、AR+RMSNorm≤8MiB且有shape/dtype/alignment等guard，其他回旧路径。inbox与counter按进程寿命持有，不能忽略资源成本。

作者gfx1250四rank、hidden2048微测，8token32KB为4.80µs vs RCCL7.34；8MB则38.00 vs27.30µs反而更慢，普通路径门限因此更低。建议考虑复现真实消息分布与竞争负载，性能/模型支持不能从目标架构名外推全AMD设备。

来源：[6923 / c9e5826a](https://github.com/ROCm/aiter/pull/6923)（2026-10-28T12:52:10Z）；[6974 / de06bd30](https://github.com/ROCm/aiter/pull/6974)（2026-10-28T21:42:59Z）。

### Dynamo main：Velo TCP/UCX response transport 成为可选运行时路径

**ai-dynamo/dynamo；本轮增量。** 新响应平面用每runtime共享listener与可复用peer连接，含stream握手、取消、心跳、版本/transport不匹配处理；UCX另需feature与环境选择，跨transport错误经公共TCP通道返回。已读生产patch与生命周期测试，不等于默认请求/响应所有平面都改用RDMA。

作者同allocation GB200饱和mock负载2048worker/6144并发，follow-up VeloTCP 5474 vs同配置control5501req/s、两者0error，该数字是修复无回退的控制，非对旧transport的统一提速；UCX硬件RDMA测试早于rebase且只覆盖通信。建议深入阅读并复现worker移除、runtime关闭、首响应取消、TLS与端口配置；服务框架变化应与模型吞吐分开验收。

来源：[15231 / 479d12dd](https://github.com/ai-dynamo/dynamo/pull/15231)（2026-10-28T22:33:39Z）。

### Dynamo：KVCR 提示接入与 Global Router 多副本视图骨架

**ai-dynamo/dynamo；本轮增量。** in-process SGLang读取backend启动metadata中的完整per-DP源坐标，private router kv_hint转成engine kv_hints，剥除caller自给hint；缺metadata不宣告、坏map报错。需要SGLang#45647，当前限single-node/PP1 HiCache；坐标是配置而非socket就绪或KV驻留，hint仍advisory。同host H100/Qwen3-0.6B检查可验证copy/output，但不是跨hostRDMA、hybrid或P/D性能qualification。

独立PoolViews库按relay connection epoch隔离每pool视图，重连屏蔽旧stream迟到更新，要求完整snapshot再update；不合并负载、不连socket、不选目的地，adapter在EOF/cancel必须显式disconnect。建议前者按受支持边界复现缓存命中，后者继续观察生产adapter/Global Router集成；不写“全局路由已完成”。

来源：[16117 / f38fb532](https://github.com/ai-dynamo/dynamo/pull/16117)（2026-10-28T06:14:36Z）；[17196 / 38042822](https://github.com/ai-dynamo/dynamo/pull/17196)（2026-10-28T22:00:24Z）。

### Dynamo：准入总预算、会话 fallback 与本地 stop 的资源闭环

**ai-dynamo/dynamo；本轮增量。** backend默认排队由每rank2×max_num_seqs改为40000/DP-rank，显式override（含0禁排队）保留；拓扑确认后调整上限仍保留已排请求。soft affinity fallback到另一worker时清掉旧DP rank，discovery移除时重绑新worker，避免DP2/rank1落到DP1报错；临时过载仍保留恢复偏好。另一修复在local stop全choice完成后释放routed stream并给terminal usage，保留晚到prompt logprob，不取消尚活跃siblings。

in-process SGLang也开始实际转发logit_bias，避免200但忽略参数。建议考虑采用并回归异构DP、client stop/usage、默认大队列的尾延迟和CPU内存；来源测试含CPU/mocker与早期GPU检查，未证明当前完整服务SLO提升。这些main提交不能自动算入旧v1.6.1。

来源：[17170 / cd3b977e](https://github.com/ai-dynamo/dynamo/pull/17170)（2026-10-28T05:13:52Z）；[14325 / 78ac1bc9](https://github.com/ai-dynamo/dynamo/pull/14325)（2026-10-28T17:14:36Z）；[16721 / 372c0ca2](https://github.com/ai-dynamo/dynamo/pull/16721)（2026-10-28T16:14:21Z）；[17511 / 843dc327](https://github.com/ai-dynamo/dynamo/pull/17511)（2026-10-28T17:32:15Z）。

### vLLM-Ascend RecoverSSM：当前只交付算子，尚未打开服务路径

**vllm-project/vllm-ascend；本轮增量。** verify记录correction/key/gate，采样后commit重建被接受前缀的KDA recurrent state并compact卷积历史，以每页一个checkpoint代替逐draft checkpoint。本PR为1/N，model/worker/config/P-D集成明确留后续；不能从算子入口推断服务已启用，也不保证tokenwise等价普通MTP。

当前extract源码有A3 144例/1280比较结果，A5完整current-source套件在commit微测设备OOM后未完成；正文继承旧集成实验不能当本次完整tree验证。部分算子前后性能有回退，减checkpoint不等于总KV显存同比减少。建议深入阅读状态重建并继续观察框架接线，再考虑服务复现；A2/950PR未由这些测试验证。

来源：[20128 / c9c4bc0e](https://github.com/vllm-project/vllm-ascend/pull/20128)（2026-10-28T10:07:47Z）。

### vLLM-Ascend：KDA state 精度与 MiniMax 混合 KV/PP 的统一几何

**vllm-project/vllm-ascend；本轮增量。** GLM KDA层分配与model cache估算统一传 `mamba_ssm_cache_dtype`，显式BF16终于生效，auto/FP32仍FP32、FP16拒绝；CPU接口测试不能证明全模BF16质量。M3则把dense/sparse/indexer对齐声明集中在model，受硬件/FP8/skip-layer guard控制地宣告64/128，令不同PP首层类型最终都选128，修PP0=128/PP1=256导致的OOM；**64是对齐最小值，不是执行block已降到64**。

作者A5八卡TP4/PP2、32/28层、MXFP8+FP8KV、EAGLE3旧接口overlay的GPQA198题180正确，但经过一次恢复；还有renderer替换LaTeX、length事件与旧依赖限定，不能当当前rebase tree无间断稳定性或统一质量证明。建议优先核dtype/spec一致、PP全rank布局，再按精确依赖版本复现精度与长跑。

来源：[18857 / 31f9175f](https://github.com/vllm-project/vllm-ascend/pull/18857)（2026-10-28T07:08:32Z）；[19179 / 3ed1f32d](https://github.com/vllm-project/vllm-ascend/pull/19179)（2026-10-28T03:18:33Z）。

### vLLM-Ascend：A2 融合能力、310P 显存记账与 A5 MoE 尺寸边界

**vllm-project/vllm-ascend；本轮增量。** A2硬件profile开启已存在的RMSNorm+FP32 cast kernel，DSV4.1 router可少一次HBM往返；gamma须同x dtype。910B4/CANN9.1、BF16/H5120微测T256 graph28.1 vs32.4µs，T1约持平，后续rebase未重跑NPU，不是全模倍速。310P non-RC KV预算则不再把其他进程/driver显存二次扣除，作者双进程同NPU可启动；不保证任意超卖容量。

A5 MegaMoE由白名单改为实际post-activation宽256–4096、128对齐，等价GMM1宽512–8192、256对齐；不适用于A3不同约束。建议对应设备用户考虑采用并回归真实模型路径、共享卡预算与边界尺寸，不能将纯gate变化当新kernel。

来源：[19304 / e8a6393d](https://github.com/vllm-project/vllm-ascend/pull/19304)（2026-10-28T14:14:47Z）；[19984 / 091834fd](https://github.com/vllm-project/vllm-ascend/pull/19984)（2026-10-28T13:42:32Z）；[19473 / 48906611](https://github.com/vllm-project/vllm-ascend/pull/19473)（2026-10-28T09:56:30Z）。

### vLLM-Ascend：Mooncake SSD exist 查询可触发 DRAM 预取

**vllm-project/vllm-ascend；本轮增量。** 新增默认关闭的 `enable_ssd_prefetch`，仅兼容Mooncake暴露ExistOptions时传 `prefetch_to_memory=True`；缺能力警告并关闭。cooldown/dedup仅显式设置时转发，省略保留Mooncake5s/30s默认；走config-dict setup时store自建TransferEngine而非共享global对象，所有权随接口改变。

本轮证据为connector代码与148单测声明，A2端到端TTFT材料指向上游Mooncake PR，未在这里重核，故不引用性能幅度。建议深入阅读并复现SSD→DRAM、压力backoff、TP重复probe与旧build降级，不把exist命中等同数据已经可消费。

来源：[19472 / 69b8a7a2](https://github.com/vllm-project/vllm-ascend/pull/19472)（2026-10-28T10:33:58Z）。

### TorchTPU：GQA 无需复制 KV，packed masking 与动态 collective shape 补齐

**google-pytorch/torch_tpu；本轮增量。** splash_sdpa直接把真实KV-head交Pallas，去掉repeat_interleave的HBM副本/反向保存，校验Q-head整除KV-head；默认block选≤512且能整除序长的128倍数，compute block跟随outer block，避免768长度或自定义小outer block失败；segment_ids连接packed文档隔离的前反向。

另experimental collective Meta用sym_sizes/empty_symint，reduce-scatter仅加可整除guard，避免每新长度重编；packed FP4转置view识别真实stride1维并修out=buffer绑定。建议考虑采用并回归GQA梯度、文档mask、动态长度compile及FP4 view；无本轮模型级速度/显存比例数据，JAX/Pallas依赖内部行为只按其公开契约描述。

来源：[5898 / 243c6a47](https://github.com/google-pytorch/torch_tpu/pull/5898)（2026-10-28T17:34:28Z）；[5946 / 8a3bd617](https://github.com/google-pytorch/torch_tpu/pull/5946)（2026-10-28T18:30:46Z）；[5811 / 704d465e](https://github.com/google-pytorch/torch_tpu/pull/5811)（2026-10-28T15:59:11Z）。

### TorchTPU：异步 host buffer 寿命与单 slice PJRT 初始化

**google-pytorch/torch_tpu；本轮增量。** nonblocking H2D总用pinned buffer，按Future持有源/目的pin指针直到传输结束，避免Python tensor提前释放；D2H也将pin寿命接到完成Future，非pin目的仍由caller负责。另world可容于单slice时设置skip MegaScale PJRT，即使JobSet注入多slice环境变量也不进入无必要的slice discovery；真正multi-slice不变。

建议考虑采用并回归丢弃源tensor、跨stream、异步D2H及JobSet拆分运行。这是内存/初始化正确性修复，不保证所有nonblocking调用都异步完成或全多slice拓扑适用。

来源：[5954 / ab218d63](https://github.com/google-pytorch/torch_tpu/pull/5954)（2026-10-28T02:11:55Z）；[5126 / 2813679a](https://github.com/google-pytorch/torch_tpu/pull/5126)（2026-10-28T07:06:24Z）。

### TorchTPU Kiln：输出 DMA 流水与嵌入 StableHLO 进入基础层

**google-pytorch/torch_tpu；本轮增量。** EmitPipeline由只读HBM→VMEM扩到body写VMEM并异步copy回HBM，仅在slot重用及尾部等待；目前双slot，输出累积、自动edge handling等仍后续。SpliceStablehlo把模块main克隆进Kiln body，经vector/tensor桥接并降f64为f32，用于未来FlexAttention mod lowering。

建议编译器开发者深入阅读、继续观察完整kernel与backward集成；这是可组合原语，不能写成任意动态matmul或完整高性能FlexAttention已交付，也不应忽略f64降精度。

来源：[5856 / 27df1d34](https://github.com/google-pytorch/torch_tpu/pull/5856)（2026-10-28T09:15:37Z）；[5780 / ee59ac68](https://github.com/google-pytorch/torch_tpu/pull/5780)（2026-10-28T11:57:14Z）。

### XLA：默认低精度 dot 选择与通信内存空间更明确

**openxla/xla；本轮增量。** default precision下，≤8bit整数×F32、输出F32/BF16且目标支持快速BF16/TF32并允许TF32时，将两操作数转BF16；整数可精确表示，**F32操作数却会降精度**。HIGH/HIGHEST、显式DotAlgorithm或禁TF32继续F32，不能称无损优化。NCCL2.32可对非RMA buffer禁用RMA kernel/copy engine，避免另一module缓存window影响本module；这是版本门槛依赖。

MORI新增单node/LSA team内连续PE子communicator（如3,4,5），任意0,2,7仍拒绝，并pin匹配facade。建议深入阅读，按显式精度需求、对称空间和真实子组复现；没有可外推的LLM总速度结果。

来源：[51539 / 78d7b386](https://github.com/openxla/xla/pull/51539)（2026-10-28T14:04:03Z）；[51587 / cfe605fe](https://github.com/openxla/xla/pull/51587)（2026-10-28T08:02:41Z）；[51776 / 1a32f46d](https://github.com/openxla/xla/pull/51776)（2026-10-28T12:37:35Z）。

### XLA：失败前分配的 VA temp 可重试释放，scaled-dot 与 warp 估算修正

**openxla/xla；本轮增量。** VA-remap scope记录reservation上的temps，TearDown释放成功者、析构重试未释放者，修setup早退后后续每次都ALREADY_EXISTS；其他非reservation transient仍未全清理。host mock负控覆盖失败路径，Mixtral8×MI355X/B300结果含后续PR2，不能归因单补丁。

scaled-dot给operand/scale约束K-minor并保留layout；warp估算对中间tile按byte width计，忽略不独立物化的broadcast/constant，runtime模型计多block驻留。建议考虑复现长上下文/低比特融合的峰值、失败重试与autotune选择，后两项没有本轮端到端模型速度保证。

来源：[51787 / 2280bc7a](https://github.com/openxla/xla/pull/51787)（2026-10-28T11:52:37Z）；[51789 / 63ca9856](https://github.com/openxla/xla/pull/51789)（2026-10-28T13:30:55Z）；[51571 / 6b152721](https://github.com/openxla/xla/pull/51571)（2026-10-28T16:13:44Z）。

### JAX：masked softmax 的导数不能用零乘 NaN 代替真正屏蔽

**jax-ml/jax；本轮增量。** 在softmax_custom_jvp路径对输入tangent与输出cotangent分别where置0，防masked位置的NaN/Inf经0×NaN污染整行，包含entropy/KL的p log p场景。新增JVP/VJP回归对照旧实现。建议相关JAX RL/蒸馏训练考虑采用并复测masked token梯度；仅该自定义JVP/where路径修复，不能推广所有NaN处理。

来源：[42093 / b9ee546e](https://github.com/jax-ml/jax/pull/42093)（2026-10-28T23:17:34Z）。

### torch_npu：通信域确定性、内存历史与 graph 输入契约

**Ascend/pytorch；本轮增量。** HCCL process-group options接受按通信域的hcclDeterministic配置；具体collective保证仍受HCCL实现与版本约束。memory recording新增clear_history控制，继续record可保留历史，但disabled仍清空；pluggable allocator仍不支持该记录路径。

Inductor继承上游codegen_upcast_to_fp32开关，避免triton_experimental半生效；layout约束前后同buffer alias跳过copy-back，防dynamic empty调度自依赖，真正clone仍回写。graph tree将DeviceMesh/ProcessGroup等opaque输入按版本接口注册并排除tensor/data_ptr复制，overlap view有bytewise复制路径。建议考虑采用并回归数值开关、动态图输入/alias与memory snapshot连续性；本轮以master commit核查，未确认CANN发行配套或实机运行。

来源：[e42557ada / e42557ad](https://github.com/Ascend/pytorch/commit/e42557ada313d171a64a2fb09dfe8b0c51887712)（2026-10-28T10:36:25Z）；[e54b5b518 / e54b5b51](https://github.com/Ascend/pytorch/commit/e54b5b51813dab9efc2acbdfec9a8030e2c8c390)（2026-10-28T09:32:44Z）；[03f8e384c / 03f8e384](https://github.com/Ascend/pytorch/commit/03f8e384c9efda4d17d69632cf8a6c712fefcd11)（2026-10-28T08:56:37Z）；[5559bf6c8 / 5559bf6c](https://github.com/Ascend/pytorch/commit/5559bf6c879b752fc18bca8c5d87823c0c491708)（2026-10-28T12:31:16Z）；[19ed3c300 / 19ed3c30](https://github.com/Ascend/pytorch/commit/19ed3c300b73be881294422ed6f67a2ffd5aee54)（2026-10-28T01:03:34Z）。

### MLX-LM DeepSeekV3.2：分块 top-k 与 latent-space sparse prefill

**ml-explore/mlx-lm；本轮增量。** 单batch、L>1且KV≥16384时默认走sparse_prefill：indexer以4096-key块维护running top-k，不物化B×H×L×N完整score；attention gather所选latent+rope，在absorbed latent空间计算。decode不改，无法选择topk时回退；query也切小块控制gather峰值。

作者M3Ultra512GB/MLX0.32.3、DeepSeekV3.2-4bit，一warmup+3timed，128K cache/pp512单chunk11.1→40.9tok/s，峰值425.5→395.7GB；完整32K/64K prompt benchmark仅1.35/1.84×。21988-token prompt KLD均值0.0014、top1 agreement99.06%，**不是输出严格等价**，128K/pp2048旧baseline因预计超内存未跑。建议考虑复现单机长prefill质量与实际chunk设置，不外推batched服务或decode吞吐。

来源：[2013 / 1f7057ca](https://github.com/ml-explore/mlx-lm/pull/2013)（2026-10-28T08:04:33Z）。

### MLX：低精度异常值、全负量化组与 GGUF 大尺寸加载

**ml-explore/mlx；本轮增量。** CPU/Metal FP8将NaN保留为E4M3 NaN而非饱和448，E8M0 scale0xFF正确解为NaN；affine量化max初值改负无穷，避免全负组错误扩大到0，CUDA改动作者未本地编译。GGUF转F16分配前补尺寸溢出检查，防极大tensor元素数乘2回绕为小allocation再越界写。

建议相关端侧量化/加载用户考虑采用，分别做NaN/Inf、signed group、畸形尺寸回归；不能从数值/边界修复推出模型质量或速度提升。本轮没有执行恶意文件或本地模型。

来源：[4756 / 7d2d94a5](https://github.com/ml-explore/mlx/pull/4756)（2026-10-28T10:06:59Z）；[4775 / 32a183e1](https://github.com/ml-explore/mlx/pull/4775)（2026-10-28T08:14:45Z）；[4764 / 93007e9a](https://github.com/ml-explore/mlx/pull/4764)（2026-10-28T23:27:24Z）。

### PyTorch COW：重落地后的事件生命周期方案再次撤回

**pytorch/pytorch；本轮增量。** 本窗main撤回e506c030重落地的COW completion-event方案，原因仅公开为内部revert；10-23撤回的是此前另一个commit。原PR正向设计与open状态都不能证明截止main仍启用。建议继续观察后续落地及明确失败证据，不凭撤回理由推断性能或正确性根因。

来源：[21756fd7f / 21756fd7](https://github.com/pytorch/pytorch/commit/21756fd7f339445d57f71b972d442abfbf31ee82)（2026-10-28T03:25:58Z）。

### Transformers main：以真实 tokenizer 格式判型，修 RoPE 与多模态输入边界

**huggingface/transformers；本轮增量。** AutoTokenizer面对Hub声明LlamaTokenizer但tokenizer.json实为ByteLevel时，改选TokenizersBackend并保留commit hash；remote auto_map和Mistral tekken优先路径例外，避免Metaspace实现误处理byte-level模型。RoPE setter在config初始化完成后才standardize，允许remote code提前赋rope_scaling；PermuteForRope支持单tensor与重命名映射，不再依赖可变self.config。Janus单字符串先包装单样本列表，防逐字符当多prompt。建议考虑采用并回归固定revision、特殊token/空格、load-save与单/多样本等价；这些是5.21之后的main修复，无质量或速度提升承诺。

来源：[48317 / 70b9938d](https://github.com/huggingface/transformers/pull/48317)（2026-10-28T10:26:19Z）；[50080 / 5e83c288](https://github.com/huggingface/transformers/pull/50080)（2026-10-28T15:02:34Z）；[50092 / 144ecf00](https://github.com/huggingface/transformers/pull/50092)（2026-10-28T23:25:49Z）；[50044 / c2779595](https://github.com/huggingface/transformers/pull/50044)（2026-10-28T16:14:28Z）。

### AITER MHA v4：sorted-sparse LSE 与 gfx950 kernel race 修复

**ROCm/aiter；本轮增量。** 主要面向diffusion推理，仍不支持causal、反向、Q端varlen；量化score的LSE只能用于相同recipe的ring chunks归并，不能当跨recipe绝对值。新增sorted-sparse路径LSE并更新gfx950多种dtype的预编译kernel对象，针对ragged tail、V-scale边界与压力下重复执行做回归。作者MI350/gfx950两文件361pass/1skip，sparse LSE相对torch.logsumexp误差按BF16/BF16FP8/FP8分别约0.0009/0.03/0.05nats；长序列特定形状还有不同幅度微测收益，不能合成全模型倍速。本轮核Python/调用契约与提交中二进制更新，未反汇编审计每个code object或独立验证race消失。建议相关稀疏注意力用户深入核构建artifact与source对应、全masked行/不连续LUT及交错压力，再考虑采用。

来源：[6841 / 8df0aa1f](https://github.com/ROCm/aiter/pull/6841)（2026-10-28T14:09:54Z）。

<a id="hardware"></a>

## 硬件与系统实践

### AMD在24×MI308X跑通DeepSeek-V4 Flash的75步GRPO：训练、FP8 rollout与在线换权重分别核验

**AMD；10-28新官方教程，官网只标日期、精确首发时刻未核；所报实验实际为8月17–21日，不能记作本日完成的训练。** [原文及固定版本/容器说明](https://rocm.blogs.amd.com/ecosystems-and-partners/deepseek-v4-mi308-aiter-backed-rl/README.html)。三节点中16卡BF16 Megatron actor用TP4/PP4/EP4、CPU optimizer offload、全重计算，另8卡vLLM TP8提供FP8权重/FP8 E4M3 KV rollout；按记录的路由做R3 replay、token importance ratio上限2，并等待RCCL RDMA换权与量化完成才恢复生成。作者给模型/数据SHA及ROCm7.2、AITER0.1、vLLM`752a3a504`、ROCm Megatron`fb455244`等版本，容器tag仍需部署时固定digest。

16K总序列/4K回答、32prompt×8回答、learning rate1e-6的前75步，30题AIME2024每题8次采样准确率30.0→49.2；这是单次轨迹的Step1/75比较，末值也是窗内最佳，不能当多seed无偏提升。文中**311.2 generated tokens/s**按整个75步聚合口径给出，缺actor/换权/idle分解与其他硬件基线，不是每GPU纯decode吞吐。最低log-PPL差只是分布一端，不能证明训练/rollout已完全一致；Step75 KL0.4706、ESS79.1%也显示仍需校准。未独立运行容器/加载权重。建议复现三步含换权、eval、checkpoint的闭环，再测漂移、尾延迟和恢复；本轮没有据教程推断这些分支均已合入上游。

稳定ID：`amd:deepseek-v4-flash-mi308x-rl:article:2026-10-28`。

<a id="evaluation"></a>

## 论文：评估、可靠性与训练诊断

### Heads of Security：按信任通道训练 attention gates，有限攻击下更稳

**2026-10-27 v1；回溯首次收录；`arxiv:2610.37657v1`；机构：Imperial／Tübingen AI Center／ELLIS／MPI-IS／Fraunhofer HHI。** [原始论文全文](https://arxiv.org/html/2610.37657v1)。把token标成trusted／untrusted／template，在每层每head的attention logits上学习通道对bias；冻结base，DPO只训GPT-OSS-20B的18,432或Llama3.3-70B的46,080个gate参数（§3、AppendixJ）。

相同评估预算下，表中各RL-Hammer attacker对GPT-OSS最高ASR为 **97% base／85% SecAlign／10% HoS**，Llama为99%／82%／1%（Table1）。但强化白盒攻击仍可达到50%ASR，只是约需2.6–3.6×计算；静态InjecAgent上的0%不能当安全证明（§4.1）。信任标签由系统给定，不能替代来源授权，纯linear attention也不直接支持此pairwise gate；应用需读取外部指令时还要额外授权逻辑（§6）。

**实践与建议：复现。** 同时测自适应攻击预算、正常tool-use utility、few-shot行为和信任标签误分，不把小参数防护写成端到端注入免疫。

### SafeTulu：梯度相关性过滤可改善安全，删得更多仍会反转

**2026-10-27 v1；回溯首次收录；`arxiv:2610.37481v1`；机构：DatologyAI／UIUC。** [原始论文全文](https://arxiv.org/html/2610.37481v1)。SimpleLoRIF用低秩压缩梯度表示和unsafe target clusters给训练样本排序，在Tulu3中删除最相似的20%，区别于只看文本是否显式有害（§3／5）。Qwen3-0.6B为939,343样本嵌入耗时约8H100×1小时，安全目标集为43,469条SafeRLHF unsafe回答。

三个训练seed的Qwen3-1.7B，GCG ASR **47.67→37.00%**，ActorBreaker **43.0→36.3%**；Llama3.2-1B GCG **38.0→30.33%**（Table1）。但Llama删40%时GCG反弹39%，超过不删38%；OLMo部分能力亦下降（Table6及Table1）。ActorBreaker为作者重实现、使用较小攻击生成模型，不能跨论文直接排安全榜。emergent-misalignment梯度信号跨模型迁移也可能低于随机，且未覆盖所有安全行为（§9／15）。

**实践与建议：复现。** 用独立unsafe目标和多个删除比例验收，追踪保留的能力及拒答，避免把梯度相关分数当样本因果责任。

### PIT／TVS：预训练中增长较多的题，可能更依赖时点知识

**2026-10-27 v1；回溯首次收录；`arxiv:2610.37631v1`；机构：UCL／University of Pennsylvania。** [原始论文全文](https://arxiv.org/html/2610.37631v1)。用2017–2024时点PIT-4B模型的likelihood变化构造Temporal Variability Score，在长度匹配后观察六个公开模型预训练checkpoint上哪些多选题随训练改善更多（§2–3）。

16,239题、四模型家族，高TVS组在最终checkpoint的36个模型任务组合中 **34个**占优，chance-normalized平均差 **7.3点**；控制平均NLL后差仍 **3.7点**，常识子集却从8.5降至0.5点（Table2），说明难度混杂不能忽略。每条checkpoint轨迹仅单次训练，PIT只有单一4B规模；污染检查覆盖PIT语料，并未覆盖所有被评估模型语料（Limitations）。

**实践与建议：阅读。** 适合拆分“更多训练提升的是哪类题”，暂不能据此断言预训练只学事实或TVS是纯知识因果测量。

### DPO稀疏性：支持样本与偏好梯度抵消提供一种局部解释

**2026-10-27 v1；回溯首次收录；`arxiv:2610.37085v1`；机构：东京大学。** [原始论文全文](https://arxiv.org/html/2610.37085v1)。在NTK线性化下把DPO更新与maximum-margin支持样本联系起来，再以preferred／dispreferred响应共享token导致的梯度抵消解释参数稀疏；后者是实验支持的假设，并非从Transformer结构推出来的定理（§3／6）。

Llama3-8B全参DPO、UltraFeedback1000pairs、三epoch，top10% pair的累计系数占比从 **11.75%→52.48%**；1000pairs上DPO梯度近零比例92.51%，SFT16.44%（Tables1–2）。但11–35×的cross-task interference下降及模型合并验证主要来自GPT2的小样本实验，不是8B合并质量提升；长训练NTK近似及完全不重叠回答均是边界（§6、AppendixB／D）。

**实践与建议：阅读。** 可用于设计DPO数据归因诊断，暂不能直接据此永久删去低权重样本或断言大模型任务向量不干扰。

### 自奖励RL的跨任务耦合：更多题也可能加速统一答案塌缩

**2026-10-27 v1；回溯首次收录；`arxiv:2610.37553v1`；机构：Columbia University。** [原始论文全文](https://arxiv.org/html/2610.37553v1)。对以自身多数票为reward的label-free RL建立可解模型：每题置信收紧与共享参数扰动竞争，更多题会减慢单题收紧，却不同比例减慢共享偏移，因而可能跨过集体塌缩阈值（§3–4）。

证据是可解模型16–24seed模拟，以及embedding64、两层Transformer的合成二元任务；使用单Apple M4 Pro CPU完成924个Transformer run，没有对前沿LLM做验证（§6、AppendixE）。作者推导的阈值依赖耦合、初始置信与噪声，不是“数据越多RL越差”的一般定律；对多数票reward的结论也不能直接套给外部verifier RLVR。

**实践与建议：阅读／观察。** 在自奖励训练监控跨题答案边际熵及真实held-out accuracy，不能因自评reward升到1就认定能力提升。

<a id="revisions"></a>

## 重要修订：按实际版本差分收录

### ZeRO-Prefill v3：补出“零开销”的通信／HBM账本与工作区间

**首发2026-05-03；v2 05-15→v3 10-27；回溯重要修订；`arxiv:2605.02960v3`；机构：Snowflake AI Research／University of Virginia／Harvard。** [v2全文](https://arxiv.org/html/2605.02960v2)、[v3全文](https://arxiv.org/html/2605.02960v3)。本轮逐版本对比确认：原MoE-Prefill更名ZeRO-Prefill，v3新增AppendixB–D的overlap knee、总物理通信、slot HBM与第二模型GLM-4.5；Table1改成on-path traffic口径并修正collective计数。核心AsyncEP仍是在大batch prefill期间异步拉专家权重，替代同步activation routing，不能把旧机制重新宣称今日发明。

新增实测8H200上每GPU每层AsyncEP收到 **1.97GiB FP8／3.94GiB BF16**，不是零通信；D2D-only overlap knee约8K／16K token，hybrid约16K。固定S=5 slots本身占11.25／22.5GiB；低载荷暴露transfer，某些BF16配置还挤占KV（AppendixB／C、Tables3–4）。GLM-4.5-FP8八H200的新增合成负载收益 **1.18–1.57×**（AppendixD）；原主结果仍是输出长度固定1的prefill-only、Qwen3-235B，不能外推逐token decode。

**实践与建议：阅读并复现修订实验。** 以总字节、关键路径等待、HBM slots、实际负载阈值验收；名称里的Zero指移出关键路径而非省掉网络。

### LiteTopK v4：新增三框架整模、CP与exactness验证

**首发2026-07-13；v3 07-26→v4 10-27；回溯重要修订；`arxiv:2607.11976v4`；机构：NTU／香港中文大学／ETH Zurich。** [v3全文](https://arxiv.org/html/2607.11976v3)、[v4全文](https://arxiv.org/html/2607.11976v4)。本轮对比确认：v3主要为vLLM中的GLM5.2及半层LongCat2.0延迟／内存图，v4将整模评测换为GLM5.2、DeepSeek-V4-Flash、Hy-4-Preview，新增vLLM／SGLang／Megatron-LM表、单独CP表、two-needle与Top-k recall检验、decode和模型特定并行配置。核心仍为sample-filter-select，保守gate避免漏掉真Top-k，LiteDSA再共享邻近query候选、保留逐query membership mask（§3–4）。

8×B200、FP8、batch1、1M context：SGLang中GLM5.2 **148.87→108.56秒（1.37×）**；Hy-4-Preview在TP8／CP8为 **41.71→28.02秒（1.49×）**；256K不少组合仅1.01–1.07×。测试配置Top-k recall为100%，two-needle准确率近似一致；但最大长度bucket只有5个实例，不能由此证明所有长文任务无损。GLM5.2 decode仅1.03–1.04×（Tables1–4），且Hy-4在Megatron尚无可用结果。

**实践与建议：复现。** 先核同checkpoint、并行方式与score精度，再跑tie／极端score和真实长上下文负载；不要拿kernel倍数替代端到端收益。

**仅入口变化，未重报成果：** QuantForge `2609.34680v1→v2`（v2 10-27；首发09-28）本轮打开两版全文，顶层正文及所有表格文本相同，新增开源search framework链接；不将其包装为新量化算法或新实验。[v2原文](https://arxiv.org/html/2609.34680v2)。

<a id="coverage"></a>

## 来源覆盖、补录与未完成区间

核心六仓的本轮表面库存含705个回溯合入PR、691个默认分支commit、255个release对象，已分页到边界；重要未审候选344个仍留本地队列。扩展24仓含1570个回溯合入PR（主窗536）、1805个默认分支commit与1646个release对象；未选择的1490个候选不视为逐项审完。两个集合与历史队列相交，不能相加当独立成果数。

PyTorch的merged搜索只返回少量条目，无法覆盖实际默认分支导出的ghstack提交；本轮用有界commit清单交叉核查，并保留“PR页面仍open、代码已导出”的区别。FlashAttention/NVSHMEM的100项release接口Transport closed，latest5恢复；前者已读beta37全文及完整compare，但不能称全历史release分页成功。Quack181正文取回失败、files可读，本轮只以TE实际guard说明依赖边界。PyTorch博客web.open失败，已用认证完整release notes与tag源码恢复关键证据。

本轮修正优先以最终代码/表注为准：TRL7609的reward统计与全过滤停机边界、AITER6954最终修改范围、NLD任务视图参数与整包存储计数、TorchTitan0.4.0和main的新API均已分别交代。旧COW回退后又有重落地，本窗再撤回是新的commit事件，不能沿用10-23状态推断当前代码。未经运行的负面路径推断均作标注。

主窗口继续到2026-10-29T01:02:28Z；下轮从此续扫并回溯72小时，同时处理全部旧队列。**全局完整语义覆盖水位仍为空**：没有把来源失败或未审候选当成功，没有声称任何厂商“没有发布”，也没有以固定条数截断有价值内容。详细查询、候选、稳定ID/版本、失败和待同步记录均在本地`.monitor/research-2026-10-29/`，本地保存与远端回读状态分别记录。

### 模型、硬件入口与工件

- HF主清单43组织×models/datasets×createdAt/lastModified共172查询分页至回溯边界，263去重候选；发现小米音频组织`mispeech`后另补4查询、3旧模型修改候选，总44组织/176查询/266候选。主清单134个对象的lastModified与昨日一致，保留昨日固定证据与未完语义，不重复抓取；其余129对象取得当前metadata/commit与选择性card/config，11个commit入口401、14个config等404逐请求留档。列表完成不是语义、张量或全量数据覆盖完成。
- 62官网入口53成功/9失败；正文/报告与工件定点补抓另计。OpenAI四入口403、Microsoft Research403、xAI news403、Kimi blog308、Dory报告403、RecursiveSI TLS失败，Jais报告重试仍403。Qwen/腾讯/百度/阶跃/Ascend动态正文与历史页仍有缺口；中英web检索补发现但不以没有命中断言无发布。中国厂商已实际检查DeepSeek/Qwen/Seed/Kimi/GLM/MiniMax/腾讯/百度/阶跃/小米/美团入口及对应HF组织。
- MiniMax英文news API补完9页、53个唯一newsId，`hasMore=false`，**只关闭该API的分页缺口**；最新记录仍停5月，和research/IR入口不一致，不能将其视为全厂商最新新闻清单。硬件已读NVIDIA/AMD官网、Gaudi、TPU、Neuron、Ascend入口；CANN/HCCL/libtpu与旧区间组件历史没有因此补齐。
- 旧backlog推进：WeVisOne2B存储精度冲突由两版tensor header关闭，质量/全张量数值未审；Open-SWE-Traces本窗只改EMNLP badge/引文、数据树未改，原effort覆盖/全行验真继续待查；K2-Horizon固定代码已复读`_can_record_outputs`与`forward`，`capture_outputs`运行语义还需匹配Transformers版本测试。Ling3.1的1M配方、ChartNet新行、Dolci清洗独立复核、Kimi承诺权重、旧模型运行时和报告缺口均继承。
- NLD-Uni工件与论文由论文分支统一展开，避免重复条目；已向该分支交付固定许可证、manifest与约18.27B整包存储参数信息。LFM3本窗视频帧率/框架帧数文档、Dory generation config、Open-SWE引文等不当新权重成果；尚未逐项完成语义的修改仍在ledger候选中，没有按数量截断后删除。

本分支所有性能为来源报告，独立训练/部署复现0；全局完整语义覆盖水位应继续为空。

### 论文与正式学术来源

已保存七类arXiv new列表、HF10-27／10-28各50条、26篇候选v1的abs与HTML、三对历史修订的两版HTML。PipeLink使用div式section／table，已修正解析器并实际阅读正文；Sharpness-Aware Muon与GPT2 pruning保留为已筛未选候选。EmbeddingGemma2报告由模型分支单独核查，避免重复。旧候选INC-MoE／DuplexTune／AutoProfiler／MPO已推进，其余旧队列仍保留。

主窗与72h回溯的arXiv API三类请求先429；改max_results=1000后，主提交／更新超时、回溯429。高级搜索的有效“most recent submission date”加large-language关键词无结果，只能说明该关键词查询未命中；误用updated_date字段返回表单错误，已单独记为失败，不能当更新枚举。中英文web搜索、HF、arXiv、OpenReview、ACL与PMLR入口均检查过；OpenReview API仍403，ACL／PMLR首页可读但未逐卷完成精确日期／版本事件审计。**主窗精确提交和真实updated事件，以及历史replacement语义队列仍未全覆盖；本节不声称无新作、无修订或已全量审完。**

本文归档：[GitHub日报](https://github.com/suhaibo666/tracker/blob/main/llm-training-daily/llm-training-daily-2026-10-29.md)。远端同步结论以提交及完整内容回读为依据，回执另存本地运行记录。
