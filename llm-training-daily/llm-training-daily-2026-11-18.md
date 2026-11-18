# LLM 技术动态日报 · 2026-11-18

> 时区 Asia/Shanghai（UTC+8）。增量扫描窗口 **(2026-11-17 09:01:04, 2026-11-18 09:00:05]**，向前回溯 72 小时至 **2026-11-14 09:01:04**。实际生成 **2026-11-18T09:21:47+08:00**。来源仅有日期时保留原日期，提交、发布、会程与发现时间按各自证据记载。

## 本次重要变化

- **模型。** GPT‑6.2 Astra/Sol API 上线，代码和 Agent 测试的收益随档位与工具配置变化。可用于本地研究的新增工件包括 SenseNova-Loop-9B 正式权重与 Sage S2.1 Flash BF16/FP8。腾讯 Hy4 Flash 披露 224B 总参数、12B 激活及自蒸馏后继续用 RL 压缩输出的做法。 [GPT‑6.2](https://openai.com/index/introducing-gpt-6-2-astra-sol/) · [SenseNova-Loop-9B](https://huggingface.co/sensenova/SenseNova-Loop-9B) · [Sage 报告](https://arxiv.org/html/2611.21072v1) · [Hy4 Flash](https://www.tencent.com/zh-cn/tencent-launches-hy4-flash-to-power-high-speed-cost-efficient-ai-deployment-for-everyday-workflows/)
- **训练。** Tunix 在 64 颗 TPU v5p 上用逐字节比对检查采样、prefill 和训练的 logprob。MLPerf v6.1 固定日志可核对达标 checkpoint 的计时，并对照厂商如何分配 rollout、训练和工具执行资源。 [Tunix Zero-TIM](https://developers.googleblog.com/en/unlocking-stable-on-policy-rl-on-tpu-with-bitwise-zero-traininference-mismatch/) · [MLPerf 配置解读](https://mlcommons.org/2026/11/mlperf-training-v6-1-insights/)
- **开源升级。** TorchTitan 默认权重同步切到 NIXL 后端，verl 改为一个模型并行组只在源 rank 收到初始数据分片。Megatron 修复 NCCL reshard 空闲 rank 和非完整目标 mesh 的边界。 [TorchTitan #6125](https://github.com/pytorch/torchtitan/pull/6125) · [verl #8708](https://github.com/verl-project/verl/pull/8708) · [Megatron #9485](https://github.com/NVIDIA/Megatron-LM/pull/9485) · [#9465](https://github.com/NVIDIA/Megatron-LM/pull/9465)
- **推理。** TensorRT-LLM 将逻辑取消与 KV 物理回收解耦。vLLM Gaudi 修复同一步前缀块读写顺序与四卡 Nemotron-H 的长提示丢失。昇腾 ChunkLoss 的设备端流式交叉熵和权重梯度路径可开展对照测试。 [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/commit/c2281e5b44018b8a032da2bb0408eb8e6cc2dc26) · [Gaudi 前缀顺序](https://github.com/vllm-project/vllm-gaudi/commit/bd8e1f65f017cedb2d2bb5135346e3b6c5d9256a) · [ChunkLoss](https://gitcode.com/cann/ops-nn/merge_requests/14705)

[模型与技术报告](#topic-1) · [训练架构、数据与后训练论文](#topic-2) · [训练系统与数值一致性](#topic-3) · [推理与部署](#topic-4) · [arXiv 版本变化](#topic-5) · [开源工程](#topic-6) · [硬件与软件协同](#topic-7) · [评估与可靠性](#topic-8) · [覆盖与续查](#coverage)

<a id="topic-1"></a>
## 模型与技术报告

### OpenAI 发布 GPT‑6.2 Astra 与 Sol，成本收益随任务和推理档位变化

11 月 17 日 17:45 UTC，OpenAI RSS 发布公告，18:00 UTC 的开发者公告确认 `gpt-6.2-astra` 和 `gpt-6.2-sol` 上线。两款模型均支持 Responses 和 Chat Completions，工具调用通过 Responses。文档分别列出 1,050,000 token 上下文，标准 API 的每百万 token 输入、缓存输入和输出报价为 Astra $8 / $0.40 / $40，Sol $2 / $0.10 / $10。Responses API 同日对这两个模型以及 6.1 Terra、6 Luna 开放 2 小时与 8 小时后台完成窗口的公测，模型 token 费比分别低 30% 和 40%。**考虑验证迁移。** 用已有工具调用样本检查 Responses 和原有前缀缓存的行为，再比较标准及后台档位成本。 [发布公告](https://openai.com/index/introducing-gpt-6-2-astra-sol/) · [API 变更记录](https://developers.openai.com/api/docs/changelog) · [Astra API 规格](https://developers.openai.com/api/docs/models/gpt-6.2-astra) · [Sol API 规格](https://developers.openai.com/api/docs/models/gpt-6.2-sol)

OpenAI 报告 Astra 在 Agents’ Last Exam 比 GPT‑6 Astra 高 5.2 个百分点，单任务成本低于三分之一。该图对 Fable 5.2 的统计排除了九个使用回退模型的任务。官方 FrontierCode 2.0 对照中，Sol 在双方都开 max 时距 Astra 0.5 个百分点，成本约为五分之一。Artificial Analysis 的独立 Terminal-Bench-Science 0.1 使用 mini-swe-agent，逐任务跑三次取 pass@1 均值，Astra xhigh 为 78.6%，max 为 78.1%。Epoch 列出的 Astra ECI 为 174，GPT‑6 Astra 为 169，Fable 5.2 同为 174。以上是各自的模型、工具和档位组合；厂商没有公开可复训的训练配方或权重。**深入阅读。** 先核对同一评测的推理档、工具与成本口径，再决定 Astra 或 Sol 的路由比例。 [OpenAI 结果与脚注](https://openai.com/index/introducing-gpt-6-2-astra-sol/) · [系统卡](https://deploymentsafety.openai.com/gpt-6-2-astra-sol) · [Artificial Analysis 实测](https://artificialanalysis.ai/evaluations/terminal-bench-science) · [Epoch AI 模型结果](https://epoch.ai/models/gpt-6-2-astra)

### 商汤发布 SenseNova-Loop-9B 正式权重与代码

11 月 17 日的新仓版本延续 Preview 的共享层思路。32 个物理解码层中的索引 4 至 27 层连续执行两遍，共 56 次层计算；循环块内部采用双残差流 mHC，混合线性与全注意力。约 9B 独立参数对应模型卡所称约 14.6B 推理等效规模。模型支持图文输入和最高 512K 上下文。正式版对比 Preview 的厂商结果中，AutomationBench 由 14.33 升至 36.67、Terminal-Bench 2.1 由 40.45 升至 59.55、GAIA 由 37.86 升至 69.90。团队称另有首日 AWQ 包，并估算 4-bit 权重搭配 8-bit KV 在 256K 时需约 15 GiB。模型元数据与官方代码说明可读，主仓的直接历史入口和 AWQ 对应工件仍待取回核验，部署示例部分沿用 Preview。**考虑复现。** 在连续批处理下分别记录加载显存、prefill 和逐 token decode，量化显存收益和循环计算开销。 [模型卡](https://huggingface.co/sensenova/SenseNova-Loop-9B) · [官方代码和中文说明](https://github.com/OpenSenseNova/SenseNova-Loop) · [固定模型仓元数据](https://huggingface.co/api/models/sensenova/SenseNova-Loop-9B?expand=sha&expand=lastModified&expand=createdAt)

### 补录｜腾讯 Hy4 flash 用混合注意力与自蒸馏换取更低任务成本

腾讯 11 月 16 日发布 Hy4 flash，224B 总参数包含 64B Engram，每 token 激活 12B。预训练采用混合注意力、LatentMoE 和 Engram，厂商称 KV cache 大小为 Hy3 的 1/24、Hy4 的 1/6。中文版披露先从 Hy4 自蒸馏，再用强化学习使端到端任务平均输出 token 数减少 50%。解码超过 150 tok/s 来自 megakernel、4-bit 量化和投机优化的综合配置。成本对照来自内部 WorkIndex 的办公任务与内部盲测，比较还受端点价格影响。TokenHub 文档列出 `hy4-flash` 的 1M 上下文、960K 输入上限及 64K 输出上限。官方称后续会公开权重，本次核验到的是 API 文档和技术披露。**深入阅读。** 重点核对端到端输出长度和内部办公评测的任务口径，继续跟踪权重。 [腾讯中文发布](https://www.tencent.com/zh-cn/tencent-launches-hy4-flash-to-power-high-speed-cost-efficient-ai-deployment-for-everyday-workflows/) · [英文发布](https://www.tencent.com/tencent-launches-hy4-flash-to-power-high-speed-cost-efficient-ai-deployment-for-everyday-workflows/) · [TokenHub 模型表](https://cloud.tencent.com/document/product/1823/130051)

### 上海 AI Lab Sage S2.1 Flash 技术报告公开，补录 BF16 与 FP8 工件

11 月 17 日 arXiv 公告收录技术报告，11 月 16 日已可定位主仓与 FP8 仓的固定提交。论文正文与官方模型页均写 Sage S2.1 Flash，arXiv 摘要元数据出现 S1.5 Flash，按固定正文和模型仓拼写。主干为 320B MoE，结合稀疏与线性注意力。Sage Sci-Perception 通过模态专用编码器和投影层，把异构科学观测映射至同一主干输入空间。工具、搜索、编程训练使用可验证环境，完成条件来自执行结果、环境状态变化或交付物检查。作者报告 33 项评测，其中 OfficeQA Pro 72.2、MLE-bench Lite 90.3 HumanRank；保留各自指标。部署文档给出 vLLM 0.33.0 补丁与自定义加载约束，固定仓有 BF16、FP8 工件；完整训练数据及独立部署验证仍待核验。**深入阅读。** 优先读报告的可验证环境与公开部署说明，关注 FP8 与 BF16 的兼容边界。 [arXiv v1 正文](https://arxiv.org/html/2611.21072v1) · [官方项目](https://internlm.github.io/Sage-S2.1-Flash/) · [BF16 固定仓](https://huggingface.co/internlm/Sage-S2.1-Flash/tree/76f4513db7464485605a002a06a6190de0460574) · [FP8 固定仓](https://huggingface.co/internlm/Sage-S2.1-Flash-FP8/tree/716e6436905650b7be93e1b7dedfb0bb95afd3a3)

### IFM K2-Horizon 完整报告对齐训练和推理工件

新证据是 11 月 17 日 arXiv 公告中的完整技术报告，六个旧模型不重复计为新权重。报告记录四个模型共用 22T token 预训练语料与配方、数据流水线合计生产约 10T 合成 token。36B-A4B 将专家混合放到注意力的 value 投影，结合 softplus 门控，在同训练数据下接近 32B dense，活跃参数约为后者的七分之一。Uno 使用无损扩散适配器，报告的解码最高提速 2.2× 对应论文中的模型与解码设置，无需另一个 draft 模型。报告同时给出 xLLM、RL360、分阶段权重、日志和数据流水线的入口，逐一下载复训与授权核验仍有缺口。**深入阅读。** 先读同语料缩放、MoVA 和 Uno 三组对照，再核对能否从公开工件还原训练阶段。 [技术报告 v1](https://arxiv.org/html/2611.21003v1) · [IFM 工件入口](https://ifm.ai/k2/) · [初始模型发布](https://ifm.ai/blog/k2/)

### NVIDIA Bolt E4B 报告解释为何保留纯 RL 配方

11 月 17 日 arXiv 公告中的报告补足 10 月已发布模型的训练细节。团队从 Gemma 4 E4B-it 出发，用 NeMo-Gym/NeMo-RL 进行多环境 RLVR 与 RLHF。训练批量 256、每题 8 条 rollout，按 2:1 混合非推理与推理样本，生成上限分别为 1,024 与 4,096 tokens。加 SFT 可更快提高多轮工具调用并抬高上限，作者在该数据混配下观察到对话质量与冗长度退化，最终发布采用纯 RL。报告覆盖 21 项评测，也披露 importance-matrix 校准的 Q4_K_M 与 DFlash drafter 工件。报告对话评测、量化和 RTX 5090 本地延迟均有各自条件。**深入阅读。** 第 3 节的 SFT 与纯 RL 对照适合检查多轮工具调用、冗长度和对话质量的取舍。 [报告 v1，第 3–5 节](https://arxiv.org/html/2611.20771v1) · [原模型页](https://research.nvidia.com/labs/nemotron/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB/)

### Microsoft MAI Code 2 与 Code 2 Flash 进入 GitHub Copilot

11 月 17 日发布的 Code 2 对准跨文件实现、调试和前端，Code 2 Flash 针对解释、局部修复和高频迭代。微软在 max reasoning 下平均四项基准，称 Code 2 为 GPT‑6.1 Sol 得分的 98%、成本的 40%；其中三项用内部 MSBench 和不启用子代理的 Copilot CLI，FrontierCode v1.1 采用 Artificial Analysis 的结果，MAI 成本以五折促销价计算。GitHub 确认可在支持的 Copilot 计划使用；MAI 正文将 Code 2 Flash 在 Foundry 标为有限预览。同期 MAI Licensed Weights 是通过 Frontier Company 销售与客户 Azure 部署的计划，页面写明即将开放。**继续观察。** 实际比较时固定 Copilot 工具环境，记录非促销价和任务完成质量。 [Microsoft AI 模型公告](https://microsoft.ai/news/introducing-mai-code-2-and-mai-code-2-flash/) · [GitHub 产品变更](https://github.blog/changelog/2026-11-17-mai-code-2-and-mai-code-2-flash-in-github-copilot/) · [受限权重计划](https://microsoft.ai/news/introducing-mai-licensed-weights/)

### Microsoft-Decision-1.1 加入图像输入与校准式决策

微软 11 月 17 日发布 GA。模型从 Gemma 4 26B-A4B 做 SFT 与强化学习，采用为结构化决策设计的推理引擎，返回选项的校准置信度，供系统路由、拒答或升级。Artificial Analysis Decision Index 由旧版 62 到 68，聚合每题延迟由 274 ms 到 123 ms。公开测量页列出文本入口，微软和 Foundry 文档另列出图像支持，记为平台能力与评测入口的区别。**考虑复现。** 对选项路由使用现有标注集测校准、拒答和 P95 延迟，图像输入单独分组。 [发布及训练说明](https://commandline.microsoft.com/microsoft-decision-model-ga/) · [Foundry 协议](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-microsoft-decision) · [AA 1.1 版](https://artificialanalysis.ai/decision-models/microsoft-decision-1-1) · [AA 1.0 版](https://artificialanalysis.ai/decision-models/microsoft-decision-1)

### Microsoft 发布 MAI Image 2.7 系列与语音模型 Realtime 1

MAI Image 2.7 和 Flash 进入 Foundry。2.7 模型卡报告 20B 非嵌入参数、flow matching、32K 文本上下文和每张最多 4,194,304 像素，支持文生图及输入图像的编辑。MAI Realtime 1 是语音到语音模型，官方对照覆盖 Big Bench Audio、Tau-Voice 和 Dynamics 365；可在 MAI Labs 试用，Foundry 为有限预览登记。微软主旨提到采用 MAI-Transcribe-2 的 Dragon Copilot Transcribe 已生产上线，单独的公开权重或 API 交付未核实。**继续观察。** 在 GA 和预览状态明确后分别核对图像编辑与语音延迟的真实端点。 [图像发布](https://microsoft.ai/news/introducing-mai-image-2-7/) · [图像模型卡 PDF](https://microsoft.ai/pdf/MAI-Image-2.7-Model-Card.pdf) · [语音发布](https://microsoft.ai/news/introducing-mai-realtime-1/) · [可用性总表](https://microsoft.ai/news/microsoft-ignite-2026/)

### LemonDecider 更新单次前向的结构化判断模型

inclusionAI 在 11 月 17 日 04:33 UTC 将主仓整体更新至 v3.1，基于 Qwen3.5-4B 的混合主干用 marker 位置读出最多 255 个选项的概率。正式包把约 1.20M 初始 SFT 与 35,467 条针对模型薄弱项或新增难度的增量决策训练融合成 BF16 权重，公开了两片体积与校验值及标量温度 0.835655。模型卡的 Decision Index 0.3 得分 63.78 来自 11 月 16 日、43 个基准、140,178 个可评分请求的开发记录，其中 79 个请求不受支持；尚无生产服务复测或榜单正式提交。**考虑复现。** 用发布包复测固定的少量请求，并核对 unsupported 的统计口径与概率校准。 [固定模型卡](https://huggingface.co/inclusionAI/LemonDecider/blob/8eca78d5be8b6b0c02346051d8c6a99370d86f5e/README.md) · [版本历史](https://huggingface.co/inclusionAI/LemonDecider/commits/main)

### Sarvam-105B-1711 更新后训练并发布 BF16、NVFP4

11 月 17 日 14:09 UTC 的两个 initial release 仓基于 105B 总参数、10.3B 激活的同一基座再后训练，面向工具调用、Agent、指令遵循和印度语言。作者同表的 IFEval 为 BF16 96.5 / NVFP4 96.4，IFBench 为 82.5 / 81.8，Arena-Hard v2 为 76.6 / 75.3，保留其他任务可能退化的口径。发布页有量化工件和部署约束，训练混合比例、成本及端点更新无第一方时间证据。**考虑复现。** 在匹配模板和生成设置下，用同一批 Indic 与工具调用样本多次比较两种精度。 [BF16 固定模型卡](https://huggingface.co/sarvamai/sarvam-105b-1711-bf16/blob/f91d460d23492ece0f98e6212bca2c26c4ad7cbc/README.md) · [NVFP4 固定模型卡](https://huggingface.co/sarvamai/sarvam-105b-1711/blob/3fe06304ae6861ad806b690caa4def7522c450cc/README.md)

<a id="topic-2"></a>
## 训练架构、数据与后训练论文

本节按 **arXiv 11 月 17 日公开公告批次**归入主窗。该批名义发布时间为 11-17 01:00 UTC，上一轮 01:18 UTC 抓取仍显示 11-16，本轮在官方清单和 RSS 确认已公开，首次可见秒级时间未证。各项单列论文版本提交时间。所有结果均为作者报告，本轮未独立复训或压测。 [arXiv 公告规则](https://info.arxiv.org/help/availability.html#announcement-schedule) · [cs.LG 公告](https://arxiv.org/list/cs.LG/new) · [cs.LG RSS](https://rss.arxiv.org/rss/cs.LG)

### Olmo-core 给出行级专家并行与万亿参数 MoE 的测量账本

**Ai2，arXiv 2611.18956 v1，11-13 21:20:46 UTC 投稿。** 新论文把现有 Olmo-core 3 的改动拆到通信、重算和拓扑。驻留的计算权重由 DDP 在全局 step 同步，分片优化器、EP 和 PP 承担容量压力。行级 EP 用 NVSHMEM 直接向目标专家缓冲区 PUT 或从其 GET，避开按 peer 拼连续块。可选 DeepEP v2 承担部分跨节点配置。§13.4 固定四张 B300、均匀路由和 batch，对每个完整 block 重算令峰值 active memory 从 103.03 降至 76.90 GiB，吞吐从 54.68K 降至 43.22K tokens/s/GPU，7 个 MoE blocks 的 EP 传输也会重跑。§14 在 512 张 B300、EP8/PP8、8K、MXFP8 和逐层重算时，1.20T/58.36B active 达 858 useful TFLOP/s/GPU，折合 38.1% BF16 参考 MFU。2.38T 的 689 TFLOP/s/GPU 是 EP32/PP8、DeepEP v2、九步中的最高点。两组均用随机路由和各配置代表性高点。公开实现固定到 Olmo-core 704625ad，olmo_symm_mem_rowwise_kernels.cuh::dispatchRowsPut 和 gatherRowsGet 分别实现按行远端写和读。复现可先按 §13.4 测三个独立进程并保留 EP 通信范围，再对学习路由记录专家偏斜、drops 和稳态 step 分位数。 [论文 §5.4–5.9](https://arxiv.org/html/2611.18956v1#S5.SS4) · [§13.4、表 29、图 62](https://arxiv.org/html/2611.18956v1#S13.SS4) · [§14.1、表 30](https://arxiv.org/html/2611.18956v1#S14.SS1) · [固定实现](https://github.com/allenai/Olmo-core/tree/704625ad94a7c8bc93f9bea7a0b865b9a0c3d44e) · [版本记录](https://arxiv.org/abs/2611.18956v1)

### 推理预训练的配比实验把事实覆盖和组合监督分开

**arXiv 2611.19042 v1，11-14 00:34:46 UTC 投稿。** 合成任务让全部原子事实都出现，刻意将部分组合从推理轨迹剔除，用 OOD 多跳问题测模型能否重组这些事实。真实文本实验以 Olmo-3 150M、10B tokens 固定预算，扫 FineMath 与 MathLake 的 OpenMathInstruct-2 子集。20% MathLake 时，GSM8K 解答序列的交叉熵为 1.608，纯 FineMath 为 1.785，纯 MathLake 为 7.513。后者 277M-token 池最多重复约 36 轮。另一组在总推理占比固定 20% 时从第 0/6000/8000 步引入推理数据，最终 CE 为 1.744/1.853/2.411。这个时序干预同时提高了后段推理密度，指标是解答序列 CE。复现可固定总 token，交叉扫混合率、唯一样本覆盖和重复次数，另记生成准确率。 [论文 §3、图 2](https://arxiv.org/html/2611.19042v1#S3) · [§4.2、表 1/2 和限制](https://arxiv.org/html/2611.19042v1#S4.SS2) · [版本记录](https://arxiv.org/abs/2611.19042v1)

### FlashREINFORCE 让每个提示的一条轨迹直接参加异步更新

**NVIDIA 等，arXiv 2611.18627 v1，09-19 03:17:42 UTC 投稿，11-17 公告新稿。** 一次 rollout 的奖励用批次均值居中，序列信任域用实际 rollout 策略日志概率矫正，从样本层面控制陈旧轨迹，逐样本平均缓解长失败轨迹放大负更新。Qwen3-30B-A3B Python 工具评测在同为 102.4K rollouts、第 800 步时，三个任务均分 67.1 对 GRPO 60.3，工具调用每轨迹 1.94 对 1.19，policy lag 约 8 对 1，两组策略新鲜度有差异。Qwen2.5-Math-1.5B 的 38.0 对 36.3 是跨论文峰值检查点比较，256K 对 512K rollouts、2000 对 1000 步。作者损失参考实现 flashreinforce/loss.py::flashreinforce_loss 在完整 learner batch 做奖励居中，按样本动作的 Bernoulli KL 拒收序列，以原轨迹长度归一化。复现应锁定 wall clock、rollouts 与提示覆盖，扫 lag 并报告长度、截断率、调用频率及采样动作之外的分布诊断。 [论文 §3、机制](https://arxiv.org/html/2611.18627v1#S3) · [§4.2、表 1–4](https://arxiv.org/html/2611.18627v1#S4.SS2) · [固定实现](https://github.com/yifanzhang-pro/FlashREINFORCE/tree/a1ebe22c6951e40b7be83395f9f15e9b41ab36dc) · [版本记录](https://arxiv.org/abs/2611.18627v1)

### Agentic OPD 实验测到工具交互模式跨域迁移

**arXiv 2611.19244 v1，11-14 08:18:45 UTC 投稿。** 固定 student 和目标任务，单独蒸馏 ALFWorld 教师后，τ² 成功率由 41.5% 降至 6.6%，语义判定的前置信息/授权依赖违规由 0.50% 升至 55.8%。加入 τ² 教师后，表 3 的组合达 68.9%，该样本中检出违规为 0。论文还比较同一教师领域 RL 前后检查点，对 student 轨迹 token 的 logp 差筛选监督，以减少带来负迁移的 token。语义违规率依赖 LLM 复核，数据覆盖和教师构成会改变结果。复现建议在目标 Agent 的同一回合日志标出请求、API 结果和前置授权，分别统计错误调用、等待及任务完成。 [论文 §3–4、表 3/4](https://arxiv.org/html/2611.19244v1#S4.SS2) · [§5、选择性监督](https://arxiv.org/html/2611.19244v1#S5.SS1) · [版本记录](https://arxiv.org/abs/2611.19244v1)

### RISE 把全图检索能力和 RL 问题采样拆成两项实验

**arXiv 2611.19871 v1，11-15 08:20:34 UTC 投稿。** 训练暴露全 Freebase 图，用类型化意图适配器、调用预算及可验证环境回执驱动搜索。按 gold SPARQL 拓扑分四档，并把训练采样推向组内仍有奖励方差的档位。Qwen2.5-7B 上 8×A100、GRPO G=8、64 问题/步、约 440 步，六项结构化任务均值 73.0%，同 backbone 的子图 RL KG-R1 为 62.8%。六任务配方子集上 uniform 66.4%、从易到难 70.7%、在线调度 71.4%，冷启动和 RL 均有独立消融。方法的可用性依赖图接口、金标准结构和可验回执。迁移到普通检索工具前可先复跑全图与子图配对，记录零方差组占比、工具调用预算和各难度覆盖。 [论文 §2、环境与课程](https://arxiv.org/html/2611.19871v1#S2) · [§3.1–3.2、表 1–4](https://arxiv.org/html/2611.19871v1#S3.SS2) · [版本记录](https://arxiv.org/abs/2611.19871v1)

### KL 蒸馏实验给 AdamW 和矩阵优化器同一个训练步数

**NVIDIA，arXiv 2611.19419 v1，11-14 14:04:42 UTC 投稿。** 四对 dense Qwen student/teacher 做在线 teacher KL，最长 1683 步。AdamW 在 4B←8B 与 14B←32B 每个有效 epoch 边界的验证 KL 都最低，Shampoo 的相对排序随配方和模型对变化。SOAP 在 Qwen3 各对约花 AdamW 的 2–3 倍 trainer time，32B←72B 的 561 步单次为 4.31×，未延长。耗时包含教师前向、验证和保存，指标检验教师匹配。结论限于一套 CoT 语料、四个模型对和文中 All-2D Muon、unblocked SOAP 等实现。复现实验可分别锁定样本数、训练时间与优化器调参预算，并另记下游准确率。 [论文 §3–5、图 1–3](https://arxiv.org/html/2611.19419v1#S4) · [版本记录](https://arxiv.org/abs/2611.19419v1)

### Latent Lattice 为每层增加沿 token 时间轴传递的状态

**MIT 等，arXiv 2611.21608 v1，11-16 18:59:38 UTC 投稿。** 每层保留一个跨 token 的横向 latent，与当前 token 的竖向残差流交互。它不需要增加逐 token 的 KV 项。20B FineWeb-Edu、24 层的约 406M 匹配对照中，标准 LLLM 与 Transformer 的估计前向成本为 0.72 与 0.71 GFLOP/token，域内 PPL 14.54 对 15.41。三轮 Jacobi 的 407M 试验达到 14.03，但估计前向需 1.85 GFLOP/token，同一 checkpoint 精确逐 token rollout 为 0.72 GFLOP 和 14.73。结果覆盖多类 backbone，主要训练规模在数亿参数。复现应固定参数和实际推理成本，分别测精确 rollout 与迭代方案，另记状态常驻和串行关键路径。 [论文 §3–4、表 1/4、PDF pp.3、9–10](https://arxiv.org/pdf/2611.21608v1#page=9) · [版本记录](https://arxiv.org/abs/2611.21608v1)

### ParGLU 在等专家预算下把一部分门控宽度换为 ReLU²

**arXiv 2611.20663 v1，11-16 07:26:03 UTC 投稿。** 将 SwiGLU 和 ReLU² 分支组合成专家 FFN，保持活动与总参数匹配。约 1.071B SmolLM tokens、4K context、三 seed 下，75M 与 152M 的验证 NLL 分别从 3.0503 到 3.0303、从 2.9654 到 2.9405，30M 则从 3.2500 到 3.2536。宽度和 75/25 分支分配是单 seed 结果，更大模型和墙钟吞吐待验证。复现可固定通信和 grouped-GEMM layout，对预算匹配的 30M/75M 对照同时记 NLL、step time 与专家负载。 [论文 §3–5、表 1–5](https://arxiv.org/html/2611.20663v1#S4) · [版本记录](https://arxiv.org/abs/2611.20663v1)

<a id="topic-3"></a>
## 训练系统与数值一致性

### Tunix 在 64 颗 TPU 上公开 Zero-TIM 复现件

Google Developers · 11 月 16 日 · 补录技术博客与实验分支。采样 decode、服务 prefill 和训练 forward 对同一采样 token 逐字节比较 logprob。Pallas 固定分片内累加，TP 用 ppermute 传输后按源 rank 顺序求和，三条路径共用 log-softmax 及采样变换。训练通过自定义 VJP 复用服务内核的前向，沿分块 KV 状态反传。

作者使用 Qwen3-8B、64 颗 TPU v5p（DP8×TP8）、FrozenLake 2×2 至 9×9、最多五轮交互、GSPO-token 与 leave-one-out 优势，对照普通 trainer-old logprob 和 token 级截断重要性采样。前 200 步单 seed 中，Zero-TIM 每次记录的平均绝对差为 0，观察到更高解题率。Zero-TIM 同时启用了跨轮 token ID 保真，因而无法从三臂数据单独归因解题率变化。线性注意力和 MoE 仍在扩展中。**考虑复现**，先用同权重三路逐层定位，再单独开关 token ID 保真。

[一手来源](https://developers.googleblog.com/en/unlocking-stable-on-policy-rl-on-tpu-with-bitwise-zero-traininference-mismatch/) · [固定复现 README](https://github.com/google/tunix/blob/4804a30967a066fb7ae3ac657747ee569768ea1d/canon-zero-tim/blog_reprod/README.md)

### MLPerf v6.1 补出异步 RL 拓扑与可回读日志

MLCommons · 11 月 17 日技术解读。结果表已收录于上期，本次新增官方配置解读，并补取 11 月 16 日发布的固定日志与补充文档。Qwen3.5-397B GRPO 上，NVIDIA 64→1,024 卡方案的训练/生成池从 32/32 改到 256/768，生成 TP4→TP8，训练用 MXFP8；Google 用 128 颗 TPU7x 训练，另用 128 组四芯片 vLLM 生成池，允许至多两个策略版本滞后，并为工具和测试运行最多 4,096 个 CPU sandbox。

固定 Google 样例日志的目标为 0.69，最终评估 pass@4 为 0.7809（251 个实例各 4 次），run_stop 落在达标的第 6 步，日志中的后续评估用时另计。Firmus 报告 8×GB300 上 GPT-OSS-20B 的 NVFP4 优化配方达标时间从自家原始参考配方的 69.60 分钟到 52.90 分钟，KRAI 的 8×MI300X Llama 2 70B LoRA 达 22.83 分钟，上轮为 28.65 分钟，补充材料称用 FP8 保留 activation、压缩 dropout mask 取消重算并融合 LoRA GEMM。Firmus 与 KRAI 的用时采用官方汇总。固定的 result_0 样例分别代表一次运行，两组对照的任务也各异。**建议深入阅读**，用固定提交的日志区分达标计时、最终评估和工具执行资源。

[一手来源](https://mlcommons.org/2026/11/mlperf-training-v6-1-insights/) · [补充报告](https://mlcommons.org/wp-content/uploads/2026/11/MLPerf-Training-v6.1-Supplemental.pdf) · [固定结果库](https://github.com/mlcommons/training_results_v6.1/tree/d5458b9f8c3c4a767e6f2c894b008cb95200e6f7) · [Google 固定 RL 日志](https://github.com/mlcommons/training_results_v6.1/blob/d5458b9f8c3c4a767e6f2c894b008cb95200e6f7/Google/results/tpu7x-4x4x8-128x2x2x1/qwen35_397b_grpo/result_0.txt)

### Primus 给 Inkling 补齐全层多机 BF16/FP8 配置

**UTC 11-17 22:45 · 主线 f39b0416b4。** MI355X 上的 Small 42 层使用四节点 PP4×EP8、全局 batch 256，Large 66 层使用十六节点 PP16×EP8、batch 1,024。四套配置均随机初始化、mock 数据、序列 4,096、50 步。Small BF16 延后 LM head 权重梯度到流水线 drain 并启用 Turbo 算子，作者称与关闭 Turbo 同损失时快 3.7%。Large FP8 默认用 Transformer Engine GEMM，作者以 FlyDSL 首步编译和调参开销解释该选择。**考虑复现**，分别统计冷启动与稳态 step，训练收敛仍需真实数据验证。

[AMD-AGI/Primus f39b0416b4](https://github.com/AMD-AGI/Primus/commit/f39b0416b420bdabb96a9c425f830c7ee1c05a37)

### Primus 将梯度 ReduceScatter 放到 SDMA 通信器

**UTC 11-17 21:43 · 主线 e227d9ace5。** Llama 3.1 8B 配方让梯度 ReduceScatter 使用参数 AllGather 已用的专用 SDMA 通信器，并同时交付可覆盖的 RCCL CE 调参。作者使用重新构建的 RCCL，在 8×MI355X 上报告每步节省 6.4 ms（0.8%）。**考虑复现**，以相同配方和同一 RCCL revision 比较 CE 路径开关，在 trace 上区分 copy engine 重叠与 kernel 本身的收益。

[AMD-AGI/Primus e227d9ace5](https://github.com/AMD-AGI/Primus/commit/e227d9ace5874c5c0cc96bc1b6ecd34e6397d93e)

### 昇腾 ChunkLoss 流式计算词表交叉熵

**UTC 11-17 14:29 · 主线 e96dd88c65、6befb233b6。** 前向把 LM head 线性投影与稀疏交叉熵合在设备内流式处理，输出 loss、逐 token max/sum 和隐状态梯度，无需物化整体 logits。反向复用统计量重算词表得分，输出 head 权重梯度。FP16/BF16、跨块统一归一化系数、忽略标签由接口明示。固定差分的前向支持表列 A2、950PR/DT。A3 只出现在权重反向支持表中。**考虑复现**，用参考实现核对 loss、dH、dW 和忽略标签，先验证 A2 与 950 的双向路径及包版本。当前公开材料有代码、文档和样例，缺同形状峰值显存与端到端训练对照。

[cann/ops-nn !14705](https://gitcode.com/cann/ops-nn/merge_requests/14705) · [cann/ops-nn !14125](https://gitcode.com/cann/ops-nn/merge_requests/14125)

<a id="topic-4"></a>
## 推理与部署

### NVIDIA 为 Qwen3.8-Flash-Next 发布更激进的混合精度 QAD

新工件 11 月 17 日经合并分片、加入校验值并把对照切换为原始 BF16。NVIDIA 使用 ModelOpt 与 NeMo AutoModel FSDP2 做 100 次优化器更新，除了路由专家，也把共享专家量化为 NVFP4，部分 GDN/QSA/索引器投影改为 MXFP8。视觉、PLE 与 MTP 被保留，MTP 没参与该轮训练。约 129.7 GB 的权重在 GB300 以 vLLM 0.30.0、TP4 加 EP、BF16 KV 验证，依赖配套的投影 padding/crop 与索引器策略；导出一致性核验至 16K，部分任务服务器上限为 262,144 tokens，测试关闭 MTP。模型卡称任务精度收益未达到统计确立。**考虑复现。** 先按配套配置跑 BF16 KV、关闭 MTP 的等条件短上下文对照，再扩到长上下文。 [固定模型卡](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4-QAD/blob/eec0322cb03229cae4cd8e2fa45547542e472eac/README.md) · [通过校验的文件提交](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4-QAD/commit/a874339cdb171ace5f36c494b6d0d3e54fdb92be)

### EchoMoE 公开跨层参考差分的 MoE 权重与运行时

McGill-NLP 11 月 17 日发布变体索引、Qwen3-30B-A3B 的 19 个学生工件，以及 Gemma 4、Kanana 2、Moonlight、Nemotron 3.5 Lightning 的样例和教师权重。做法是每隔若干层刷新参考隐藏态，Expert Parallel 的 dispatch/combine 传压缩后的差分。Qwen 代表配置每四层刷新，512/768 维编码，跨节点字节减少 58%，18 项评测均分由教师 66.9 降至 63.1。吞吐以 vLLM 0.29 + DeepEP、4 节点 32 张 H100 测量，部分小批工作负载收益收窄甚至为负；精度在另一 vLLM 插件环境中评测。**考虑复现。** 固定专家分配，按请求数与网络条件分别测吞吐，再与 18 项均分的损失对照。 [发布索引](https://huggingface.co/McGill-NLP/EchoMoE/blob/716911e3caf811f88f272ed0ccf06ca99e06c277/README.md) · [代表权重](https://huggingface.co/McGill-NLP/EchoMoE-Qwen3-30B-A3B-s4-r512-c768) · [代码和结果](https://github.com/McGill-NLP/EchoMoE)

### Apple FastFrames 公开两步视频扩散工件

FastFrames-S 1.3B 和 FastFrames-L 14B 在 11 月 17 日 21:20 UTC 以 Public release 提交开放模型仓，许可标作 Apple AMLR。两个模型都是自回归、两步 480p 视频扩散，采用轻量 KV refinement 与像素解码器替代较重的缓存更新和 VAE 解码。S 在作者 1 GPU、480p、满 KV cache 的 profiling 中合计 35.4 ms，两次 DiT 合计 32.3 ms；项目图中的 L 交互档使用 8 GPU。官方页面当时仍将论文链接标为稍后提供。**考虑复现。** 把两步 DiT、KV refinement 和像素解码分开计时，以确认单卡最慢环节。 [项目页](https://apple-aiml-research.github.io/ml-fastframes/) · [S 固定工件](https://huggingface.co/apple/FastFrames-S/tree/9a4b9cdc4deeedac6c832112707910af000f274d) · [L 固定工件](https://huggingface.co/apple/FastFrames-L/tree/45e2e3798cb0f8e6a3da7a604613e7e860f35cb4)

### Red Hat AI 与 Inferact 发布更新的 Kimi K3.1 Pro DSpark 草稿器

11 月 17 日，Red Hat AI 的新仓从 Inferact 工件同步，卡片写明接替早期 Preview。BF16 草稿器有 2.5B 参数、五层，每层 2,048-token 滑窗，读目标模型五层隐状态，每轮提议七 token。作者给出的九项常规任务合并接受长度为 3.864（含 bonus token），SPEED-Bench 880 段会话为 3.2690；BEAM 1M 的 700 条提示长度为 826K–1.04M，平均接受长度 3.421。保留不同负载的接受长度口径，卡片给出的常规部署命令默认上限 65,536，不能据此复现 1M 测试。**考虑复现。** 固定采样和验证策略，用长提示的每位置接受率和实际吞吐一起评估。 [固定草稿权重卡](https://huggingface.co/RedHatAI/Kimi-K3.1-Pro-DSpark/blob/a23f4cf46647ebd00f5db8c03d2129c2d7030cc4/README.md) · [早期 Preview](https://huggingface.co/Inferact/Kimi-K3.1-Pro-DSpark-preview)

### MetaEncoder v1 补齐许可文件

11 月 17 日 01:57 UTC，MetaEncoder v1 模型仓提交 Apache License 2.0，关闭此前直接读取 LICENSE 返回 404 的文件缺口。这是许可工件增量；训练来源、整包权重与独立评测仍待核验。**继续观察。** 后续优先跟踪模型卡中的训练来源和跨模态检索复测。 [固定 LICENSE](https://huggingface.co/facebook/meta-encoder-v1/blob/7615b21aee6eee8838a4682a6346613f088257a1/LICENSE)

### NVFP4-MLA 让同一份 NoPE latent 供两个 FP4 BMM 使用

**NVIDIA，arXiv 2611.18972 v1，11-13 22:00:49 UTC 投稿。** NoPE 用 token×channel 的 16×16 共享 scale 同时匹配两个 BMM 的归约轴，RoPE 尾部额外存一份 FP4 量化残差，解码在 FP32 累加和维护 softmax 状态。Rubin 上 DeepSeek-R1/Kimi-K3 在九项任务的平均精度接近 BF16，存储 MLA cache 分别缩为 1/3.20 和 1/3.05。Kimi 最后两层 MLA 仍为 FP8，压缩数不含 KDA recurrent state。对同预算 FP8 MLA 的 AgentX 回放，R1 的 48 会话、每个 prefill GPU 43 GiB 时 prefix hit 77.0%→87.1%，p90 TTFT 108→30.9 s。39 组固定长度、CUDA Graph kernel 形状的几何均速约 1.87×/1.99×。复现把同显存会话保留、同命中 kernel 时间和同用户速率的端到端吞吐分开测，固定 MTP 及 split-KV。 [论文 §3.1–3.5](https://arxiv.org/html/2611.18972v1#S3) · [§4.1–4.4、表 1–3、图 5/6](https://arxiv.org/html/2611.18972v1#S4) · [版本记录](https://arxiv.org/abs/2611.18972v1)

### LoopSkim 在循环网络中途纠正候选，完整深度才提交

**arXiv 2611.19896 v1，11-15 09:21:32 UTC 投稿。** 多位置草稿经过浅层准入后，循环模型在若干深度检查并裁掉被早期错误污染的后缀。最终完整深度验证决定输出，采样实验对不同深度耦合同一随机扰动，维持目标采样分布。单张 H200、BF16、单请求下，无需训练的配置对各自引擎 AR 的 greedy 平均加速 7.66×、T=1 采样 4.87×。相对 LoopSpec 的 AR 归一化加速比 1.73×，将 LoopSpec 调度移入同一引擎时为 1.64–2.08×，按模型与模式分组。计入 prefill 后带草稿头的加速保留 71–90%。复现检查同一随机种子下逐 token 一致性、早退后缀废弃量和多请求批下验证吞吐。 [论文 §3、无损条件](https://arxiv.org/html/2611.19896v1#S3.SS2) · [§4、表 1、图 2/3](https://arxiv.org/html/2611.19896v1#S4) · [版本记录](https://arxiv.org/abs/2611.19896v1)

### SpecRelay 让边缘草稿与云端验证跨 RTT 流水化

**arXiv 2611.18963 v1，11-13 21:40:06 UTC 投稿。** 边缘异步发送多个草稿树，对不确定的续写起点保留替代分支，用云端验收树谱系回收或取消后续工作。RTX 4090/Qwen3-1.7B 至 A100/Qwen3-14B 的 gRPC 实验基础 RTT 约 12.5 ms，额外注入 0–500 ms。纯云验证路径在额外 RTT ≥25 ms 的三套任务均胜修正的 SpecEdge 同步基线，最大 1.45–1.50×。SpecBench 零额外 RTT 同步 hedging 为 1.10×，异步 hedging 为 1.02×。纯云路径使用 greedy 并保持目标模型输出至 FP16 近似并列边界，完整系统另允许设备独立回答，须比较相同任务精度下的总完成时间。复现应同时扫网络抖动、录 verifier GPU time 并用正文附录 A 的同步参考。 [论文 §4–6、表 1–3](https://arxiv.org/html/2611.18963v1#S6) · [版本记录](https://arxiv.org/abs/2611.18963v1)

### CECAR 把专家权重命中纳入端侧 MoE 路由决策

**arXiv 2611.20119 v1，11-15 16:47:29 UTC 投稿。** 缓存替换分数、命中偏置和可变 top-k 共同决定专家权重 I/O。在同解码设置的 Qwen3-30B-A3B 上，默认路由 2.8 tokens/s、MoCCE 8.6、CECAR 12.5；专家命中分别为 68.3%、92.3%、96.6%，CECAR 平均启用 7.13 个 routed experts。三模型四任务精度均值较精确路由低 0.5 点，具体任务波动可达数点。代码 Qwen3_30B_A3B.py::FiddlerQwen3Moe._select_cecar 调整有缓存专家的选择分数，utils_cache.py::CachePolicyWrapper.get_eviction_scores_batch 提供逐候选替换分数。训练样本、缓存容量、存储链路和逐轮上下文会影响端侧收益。复现可保持 top-k 或精度约束一致，分开测命中、I/O 等待和输出方差。 [论文 §4、方法](https://arxiv.org/html/2611.20119v1#S4) · [§5、表 1–3](https://arxiv.org/html/2611.20119v1#S5) · [固定实现](https://github.com/hoonki2002/CECAR/tree/28cd07cc56edd1e88492e0f4648c627a7d3d65c6) · [版本记录](https://arxiv.org/abs/2611.20119v1)

### Stint 为长输出请求的 KV 恢复保留实际生成机会

**arXiv 2611.20106 v1，11-15 16:32:44 UTC 投稿。** 用 decode hidden state 预测剩余长度选抢占对象，把快照按实际大小放进 CPU 连续区，用双缓冲 bulk transfer 恢复。等待超过阈值的请求获得恢复及生成新 token 的保护时段。8 张 H200、Qwen3-32B BF16、等 GPU KV 容量、IMO-AnswerBench 固定输出长度的三次配对回放，相对同 vLLM 重算路径吞吐 +11.99%，p99 TPOT −50.69%。代价是 pooled p99 ITL 从 47.9 到 78.0 ms，超过 10 秒的暂停更普遍，单次最大暂停 49.3 s。主测固定长度且部分评测题用于预测器训练，文中另有 299 个留出题和真实 EOS 小规模验证。复现需要按长短输出、真实 EOS、队列等待和每请求最长暂停分层记录。 [论文 §6.1–6.4、图 8–10、表 2/3](https://arxiv.org/html/2611.20106v1#S6) · [版本记录](https://arxiv.org/abs/2611.20106v1)

### JointQ 联合搜索 GPTQ 格式的整数码和组 scale

**Fujitsu，arXiv 2611.20384 v1，11-16 01:21:50 UTC 投稿。** 固定零点，将连续 scale 从层重建目标中消元后重搜整数，并用多候选热启动和层级正则稳定低位解。四种配置×四个模型比标准 GPTQ 的六任务平均零样本准确率 14/16 胜，最大收益在低位 per-channel，评测含 actorder=false、mse=false 的主基线及强化 GPTQ 附录。Qwen3-14B W3 per-channel 平均准确率 0.6089→0.7203，WikiText-2 PPL 却 13.84→18.92。B200 离线量化耗时 2.63–15.71×，部分正则超参借助所报任务选择。固定实现中 quantize.py::run_init_local_search 为候选启动局部搜索，select_best_solution 按行选最小重建误差。复现应同时锁定校准样本、组大小、GPTQ 增强开关和量化时间，在独立任务上验证。 [论文 §3–4、表 1、§6 限制](https://arxiv.org/html/2611.20384v1#S4) · [固定实现](https://github.com/FujitsuResearch/OneCompression/tree/1ae59fd38c7f5d24e15c634d5f38ffb53c0620be/onecomp/quantizer/jointq/core) · [版本记录](https://arxiv.org/abs/2611.20384v1)

### MiSTiQ 用联合 scale 搜索提高极低位 KV 的保真度

**arXiv 2611.21216 v1，11-16 15:07:41 UTC 投稿。** Trellis beam search 为每条候选编码维护足够统计量，边扩展边更新连续 scale 再舍入。UE5M3、1.25 effective bits 的 Qwen3-8B 相对 1.24 bit 的 NSNQuant，LongBench 32.16→33.42，GSM8K 29.04→52.46，BF16 分别为 41.07 与 88.63。2.25 bit UE5M3 时 Qwen GSM8K 75.97 低于 NSNQuant 81.50，收益会随格式与任务变化。正文主结果是量化保真度，附录 G 的 VecInfer 组合有单独比较。复现先核对量化元数据计入的有效 bit、prefill/decoding 的量化成本，再做完整 serving SLO 测量。 [论文 §5–6、表 1–3、附录 G](https://arxiv.org/html/2611.21216v1#S5.SS2) · [版本记录](https://arxiv.org/abs/2611.21216v1)

<a id="topic-5"></a>
## arXiv 版本变化

### BrownoutMoE v2 补上突发负载的逐秒时延与部署条件

**arXiv 2607.04164 v2，11-15 01:44:26 UTC 提交修订，11-17 replacement。** v2 扩写控制器及冻结专家分组的部署描述，新增 §5.5 持续升压的逐秒 prefill/decode 延迟和 §5.6 泛化讨论。v1 已有融合与未融合对照，v2 §5.3 明确融合内核下多种设置均约 1750–1800 tokens/s，未融合才见显著吞吐差。新增逐秒对比仍使用 unfused vLLM。复现应在实际 fused 内核、真实 burst trace 上同时验证质量约束和 TTFT/TPOT。 [v1 §4](https://arxiv.org/html/2607.04164v1#S4) · [v2 §5.3–5.6](https://arxiv.org/html/2607.04164v2#S5.SS5) · [版本记录](https://arxiv.org/abs/2607.04164v2)

<a id="topic-6"></a>
## 开源工程

扫描主窗为 11 月 17 日 01:01:04 至 11 月 18 日 01:00:05 UTC（左开右闭），回溯起点为 11 月 14 日 01:01:04 UTC，并续查上一轮从 11 月 12 日开始的缺口。版本条目按实际工件时间列出，其余为主分支合入。

### 版本交付

- **sgl-router 0.1.0 独立分发。** SGLang 新的 tag 工作流按路由器自己的版本构建。PyPI 于 11 月 18 日 00:53:26 UTC 起提供 Linux aarch64、x86_64 两个 wheel，镜像仓库的 `lmsysorg/sgl-router:v0.1.0` 双架构 manifest 于 00:52:53 UTC 更新。此版本号适用于路由器。 升级时可先独立验证路由器镜像架构和后端协议。 [发布工作流 #50076](https://github.com/sgl-project/sglang/pull/50076) · [PyPI](https://pypi.org/project/sgl-router/0.1.0/) · [镜像 manifest](https://hub.docker.com/v2/repositories/lmsysorg/sgl-router/tags/v0.1.0) [官方 tag](https://github.com/sgl-project/sglang/releases/tag/router-v0.1.0)

- **FlashInfer 0.7.5rc3 已上传。** PyPI 首个 `flashinfer-python` 工件于 11 月 17 日 09:45:11 UTC 上传。GitHub 提供 cubin 和按 CUDA 版本区分的 JIT cache wheel。RC2 到 RC3 的功能增量修复输入行数为 0 时的量化 kernel 启动，另外两笔是文档和版本号。下文 FlashInfer 主线修复另有各自的合入提交。 评估升级时优先回归空行量化和实际使用的 CUDA wheel 组合。 [RC2→RC3 差分](https://github.com/flashinfer-ai/flashinfer/compare/v0.7.5rc2...v0.7.5rc3) · [PyPI](https://pypi.org/project/flashinfer-python/0.7.5rc3/) [官方 tag](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.7.5rc3)

- **TRL 1.18.1 稳定版已上传。** PyPI 首传于 11 月 17 日 12:14:26 UTC。准确率奖励改为独立进程评分，以免单个病态答案卡住训练。该 tag 还修复 Python 3.15 CLI 和支持新版 SmolVLM2 模板，并跟进 Transformers 5.23 对激活检查点配置的改名。 依赖准确率奖励的 GRPO 作业可先用小批次确认评分 worker 的完成与回收。 [PyPI](https://pypi.org/project/trl/1.18.1/) [官方 tag](https://github.com/huggingface/trl/releases/tag/v1.18.1)

- **Molt RL 0.2.3 已上传。** PyPI 首传于 11 月 17 日 03:54:45 UTC，依赖切换到 vLLM 0.34.0 与 AutoModel `2cacea76`。发布说明还列出 rollout KL 的 log-prob 保留、SFT 全评估集损失及多轮 VLM 回滚的未提交状态修复。 使用多轮 VLM rollout 时可复核回滚前后的 token/log-prob 轨迹与依赖锁。 [PyPI](https://pypi.org/project/molt-rl/0.2.3/) [官方 tag](https://github.com/NVIDIA-NeMo/labs-molt/releases/tag/v0.2.3)

### 训练、检查点与后训练

- **Megatron Core 修复两处 resharding 边界。** NCCL CopyService 重用已初始化的 group，覆盖同一 communicator 上有 rank 从未提交 P2P 的情况。Planner 对不完整的目标 mesh 仍生成一致的收发计划，将 tensor reshard 所需的完整 mesh 错误留给对应调用方。回归测试涵盖空闲 rank、重复发布及仅部分目标 rank 持有分片。 热换权重前可用带空闲 rank 的连续两次发布验证传输与收发计划。 [#9485](https://github.com/NVIDIA/Megatron-LM/pull/9485) · [#9465](https://github.com/NVIDIA/Megatron-LM/pull/9465)

- **Megatron Core 的重计算、推理后端与 TP 边界。** MFSDPv2 识别只重算部分参数的 unit，在 backward 必要时 unshard 全部 planes，测试对照损失和梯度。批次不变模式移除 DeepGEMM 选项，保留默认 `te_native` 与 `triton`，文档明确 CP 和 attention dropout 边界。Shortcut MoE 另解除 TP>1 的 overlap 限制，MIMO vision projection 恢复 TP>1 构建，DSA-over-GQA 增加 inference 与 wide-residual layer spec。 迁移批次不变 recipe 时先确认 backend 配置，再对照 partial recompute 的梯度。 [#8839](https://github.com/NVIDIA/Megatron-LM/pull/8839) · [#8923](https://github.com/NVIDIA/Megatron-LM/pull/8923) · [#9491](https://github.com/NVIDIA/Megatron-LM/pull/9491) · [#9483](https://github.com/NVIDIA/Megatron-LM/pull/9483) · [#9429](https://github.com/NVIDIA/Megatron-LM/pull/9429)

- **verl 的单控制器 RPC 改变分片交付约定。** 采用 source-only dispatch 的 mesh 每个 DP shard 只发给模型并行组的 source rank，其他 worker 初始拿到 `None`，worker preprocessing 再完成组内分发。自定义 mesh/worker 调用方需按新的 source/collect 和预处理约定迁移。PPO V1 另只在 `algorithm.use_kl_in_reward` 开启时从 TransferQueue 读取逐 token 的 reward 字段，避免非 KL 路径请求不存在的列。 使用自定义 worker 的部署应先验证一个 DP shard 对应唯一 source、其余 rank 能组内收到输入。 [#8708](https://github.com/verl-project/verl/pull/8708) · [#8565](https://github.com/verl-project/verl/pull/8565)

- **TorchTitan 默认切到 RDMA 权重同步，上一轮跟踪的 packed staging 已合入。** Trainer 在 push stream 直接写入 packed staging，generator 预取后按原权重布局拷回，减少 trainer 侧再拼装。Controller 默认启用 `ModelStateDictTransport`，默认后端 NIXL，也可选 Mooncake。继续使用 TorchStore 可显式设 `model_state_dict_transport=None`。 迁移 TorchStore 作业时可比较 trainer staging 与 generator pull 各自的耗时和峰值内存。 [#6097](https://github.com/pytorch/torchtitan/pull/6097) · [#6125](https://github.com/pytorch/torchtitan/pull/6125)

- **TorchTitan 的 SGLang rollout 增加跨 generator 复用和 attention-DP。** 共享 KV 走 Mooncake Store，要求开启 HiCache、运行级 `extra_backend_tag`，并关闭权重加载后的 KV reset。Attention-DP 可与 EP 联用，测试覆盖 DP rank 请求数不同的 bitwise parity；复用旧 KV 时仍需让命名空间与 policy 权重保持一致。 使用共享缓存时可验证运行 tag 隔离，并对照权重版本切换前后的 log-prob。 [#6027](https://github.com/pytorch/torchtitan/pull/6027) · [#6143](https://github.com/pytorch/torchtitan/pull/6143)

- **TorchTitan 按层选择重计算，移除图捕获前的不安全回收。** `BlockRematOverride` 按 block 路径覆写保存/重算区域，重叠或无效配置提前报错；accelerator graph wrapper 删除未同步的 `empty_cache()`。实验性 graph trainer 合入从两级 kernel cache 查找匹配子图并替换调用的 pass。 显存调优可从最重的 block 小范围覆写，再检查梯度与 capture 后的常驻内存。 [#6104](https://github.com/pytorch/torchtitan/pull/6104) · [#6131](https://github.com/pytorch/torchtitan/pull/6131) · [#6166](https://github.com/pytorch/torchtitan/pull/6166)

- **Miles 给 DeepSeek-V4-Flash 增加 Blackwell MXFP4 routed-expert rollout 与 QAT。** QAT 在首个 microbatch 将参与 forward 的专家权重投影到 MXFP4 网格，训练仍用 MXFP8 GEMM，优化器持有 FP32 master weights；rollout 用 MXFP4 权重 × MXFP8 激活，覆盖 B200/B300/GB200/GB300 的实验路径。TorchTitan→ICI 权重路径另将 Qwen3.5-MoE 的 grouped experts 拆为逐 expert 权重，限定 trainer EP=1，测试检查不同布局与不可整除分片。 评估量化训推差时可固定 checkout、seed 和 rollout，分别看无 QAT 与有 QAT 的 log-prob 偏差。 [#4017](https://github.com/radixark/miles/pull/4017) · [#4899](https://github.com/radixark/miles/pull/4899)

- **Miles 将 rollout 路由 replay 的搬运缩到当前 rank 和 microbatch。** token 与路由共享 CPLayout，output store 只取本 CP rank 的序列行和本 PP stage 的 MoE 层，后台 worker 在线程中预取后续 microbatch。随附的 e2e 路径覆盖 CP、PP 和 indexer replay；这些改动重排读取及预取，原路由决策仍来自 rollout 数据。 长序列作业可比较 output-store 读取量、重叠效果及跨 CP/PP 的路由一致性。 [#4508](https://github.com/radixark/miles/pull/4508) · [#4533](https://github.com/radixark/miles/pull/4533) · [#4870](https://github.com/radixark/miles/pull/4870) · [#4573](https://github.com/radixark/miles/pull/4573) · [#4627](https://github.com/radixark/miles/pull/4627) · [#4643](https://github.com/radixark/miles/pull/4643) · [#4875](https://github.com/radixark/miles/pull/4875) · [#4890](https://github.com/radixark/miles/pull/4890)

- **NeMo RL 按统计所需样本限制 streaming 和 loss 内过滤。** 新检查防止 PPO 全批归一化随 chunk 划分变化；GRPO 在 forward 后才剔除样本时要求 `masked_reward_policy=include`，使 pre-forward 基线与所声明的统计口径一致。SingleController PPO 可把 value model 常驻独立整节点 GPU，要求 Megatron policy 且 generation 非 colocated。部分异步 recipe 转用 NCCL reshard，Megatron policy Ray worker 同时移除 1000 调用上限。 升级 streaming PPO/GRPO 配置时应先验证各 chunk 分法下的 advantage 统计和过滤语义。 [#5216](https://github.com/NVIDIA-NeMo/RL/pull/5216) · [#5037](https://github.com/NVIDIA-NeMo/RL/pull/5037) · [#4868](https://github.com/NVIDIA-NeMo/RL/pull/4868) · [#5120](https://github.com/NVIDIA-NeMo/RL/pull/5120)

- **DeepSpeed 修复 ZeRO-3 checkpoint 迁移及 NVMe dtype。** 从未切 TP 的 Universal Checkpoint 恢复 AutoTP 时先取目标 TP shard。AutoEP checkpoint 转换逐个读取 optimizer 文件，降低转换时的 CPU 峰值内存。NVMe 换出的参数分区现在用模型参数 dtype 创建，回归增加 FP32 参数与优化器同时 offload。 迁移 checkpoint 时可把目标 TP shard、NVMe dtype 和逐文件转换峰值纳入恢复检查。 [#9069](https://github.com/deepspeedai/DeepSpeed/pull/9069) · [#8844](https://github.com/deepspeedai/DeepSpeed/pull/8844) · [#9099](https://github.com/deepspeedai/DeepSpeed/pull/9099)

- **Megatron Bridge 修复 Gemma 4 TP 梯度，并调整多个训练 recipe。** 开启 SP 时，复制在各 TP rank 上的 per-layer embedding 投影按 sequence-parallel 参数跨 rank 求和，以汇总各 token 子集的贡献。Inkling 2 的 GB200 预训练用两个 PP stage 并固定 FlexAttention backward tile。DeepSeek V3 VRNVL72 recipe 预取整步 microbatch，Qwen3.5-VL 合并 RoPE 并将 Hybrid block 作为图捕获单元。 可用 TP>1、SP 开关对照 Gemma 4 投影梯度，再单独度量各 recipe 的吞吐。 [#6978](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6978) · [#6970](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6970) · [#6985](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6985) · [#6973](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6973)

### 推理：投机解码、缓存一致性与多卡通信

- **vLLM 的 KV 分配和运行时复制。** DFlash/DSpark 的滑窗 draft 层只给末尾必要区间实际分配 KV block，prefix 复用按 draft 所需窗口验证。CUDA 上 CoW 块复制直接写入目标 block，避免批量 gather 临时张量。Prefix replay 重算纳入 `prefill_schedule_interval` 节流。多 attention backend 同时支持默认 block size 时，须各方请求同一替代规格才调整默认值。 接近显存上限的长前缀负载可对照 draft 分配量与 CoW 峰值，并测试连接器 cache hit。 [#64109](https://github.com/vllm-project/vllm/pull/64109) · [#61051](https://github.com/vllm-project/vllm/pull/61051) · [#67879](https://github.com/vllm-project/vllm/pull/67879) · [#60892](https://github.com/vllm-project/vllm/pull/60892)

- **vLLM 修复 NIXL 断链恢复与多卡 PP 传输选择。** Pull worker 在 NIXL 报远端断链时清理断连 peer，后续可重新握手；当前在途失败需照常处理。旧名 `NixlConnector` 与 `NixlPullConnector` 统一应用 DCP 校验，DCP>1 仅支持 MLA。Copy-engine PP transport 扩展到 TP>1、PCP>1 和同一 NVLink 域的跨节点场景，某 rank 建立失败时使各 rank 统一报告错误或以 `auto` 回退 NCCL。 部署 NIXL 与 copy engine 时可验证 peer 恢复后的新请求和各 rank 的一致回退。 [#68047](https://github.com/vllm-project/vllm/pull/68047) · [#60255](https://github.com/vllm-project/vllm/pull/60255) · [#66677](https://github.com/vllm-project/vllm/pull/66677)

- **vLLM 拓展 Kimi 投机与稀疏 prefill 路径。** Kimi K3 的 causal DSpark draft blocks 可走 CAKE；测试比较逐 token 参考、FP32 参考及有无 split-KV 的误差，适用已有 specialized decode 条件。RecoverSSM 保留超出纯 decode token 限制的 mixed CUDA graphs。稀疏 MLA 的 masked MHA、dense MHA、sparse MQA 路由提到 metadata builder 决定，Triton 滑窗 decode 只切实际滑窗覆盖的 split-K 区间。 混合 prefill/decode 压测可单查图 capture 覆盖、滑窗边界以及 causal DSpark 数值误差。 [#67588](https://github.com/vllm-project/vllm/pull/67588) · [#67903](https://github.com/vllm-project/vllm/pull/67903) · [#67338](https://github.com/vllm-project/vllm/pull/67338) · [#61646](https://github.com/vllm-project/vllm/pull/61646)

- **vLLM 修复权重重载、量化和算子融合的前置条件。** Direct reload 为需清零的 padding 参数从构建时设置 `fill_on_reload`，替换参数后仍保留此标记；在线 NVFP4 MoE 与 CPU offload 接通，offloader 跳过 meta 权重，在线量化按原始专家权重得到量化结果。RMSNorm fusion 限定末维归一化；同一 LoRA ID 绑定不同 adapter 时拒绝复用；ROCm KV 缩容后清掉持久 block table 的残留条目。 滚动重载带 padding 的 MoE 权重时可对照同一 checkpoint 的冷启动结果。 [#67563](https://github.com/vllm-project/vllm/pull/67563) · [#67136](https://github.com/vllm-project/vllm/pull/67136) · [#67989](https://github.com/vllm-project/vllm/pull/67989) · [#67970](https://github.com/vllm-project/vllm/pull/67970) · [#67966](https://github.com/vllm-project/vllm/pull/67966)

- **vLLM 增加三条模型执行路径。** Nemotron Labs Diffusion 注册 3B/8B checkpoint 的 tri-mode 推理，默认 masked block diffusion；混合因果性限制所选后端。Base Labs mammoth-1 / elephant-1 用训练时 block-bidirectional mask 执行并接入结构化决策端点。Llama 可选择 Transformers modeling backend。 选新模型执行路径前可用小 checkpoint 验证 mask、backend 选择和实际 API 响应结构。 [#58387](https://github.com/vllm-project/vllm/pull/58387) · [#63114](https://github.com/vllm-project/vllm/pull/63114) · [#66547](https://github.com/vllm-project/vllm/pull/66547)

- **SGLang 增加两种投机开销控制，并接入 Kimi K3 prefill CP。** 全滑窗 MHA draft 可用逐请求 KV ring，当前 paged MHA 路径限定 CUDA 的 trtllm_mha / FA4，不含 DP attention 与 CP；FA4 还要求 top-k=1。另一条实验路径以置信度为每个请求选择 compact verify 长度，约束 target backend 和 attention-DP 组合。Kimi K3 的 prefill CP 要求 FA3 或符合维度和设备条件的 FA4。 开启 compact verify 前应确认目标 backend 与并行约束，再比较接受长度和目标 model 实际验证行数。 [#50740](https://github.com/sgl-project/sglang/pull/50740) · [#46190](https://github.com/sgl-project/sglang/pull/46190) · [#38344](https://github.com/sgl-project/sglang/pull/38344)

- **SGLang 补齐 cache hit 后所需状态。** 直接 external linker 恢复 Inkling 的 MTP draft FULL/SWA KV 页面，scaled draft KV 有明确拒绝。SWA 分叉点 checkpoint 同时保存 prompt 尾部窗口，覆盖对齐/非对齐和跨 chunk。Shortest-prefill-first 的前缀去重遵循由短到长顺序，并为 encoder replay 行预留空间。 多层缓存命中测试宜先覆盖长短 prompt、非整页尾部和 external tier 恢复。 [#47293](https://github.com/sgl-project/sglang/pull/47293) · [#50059](https://github.com/sgl-project/sglang/pull/50059) · [#46022](https://github.com/sgl-project/sglang/pull/46022) · [#46063](https://github.com/sgl-project/sglang/pull/46063)

- **SGLang 修复 P/D 请求和图捕获的生命周期。** Mooncake transfer worker 先快照本房间传输条目，aux 写失败将 session 标记为失败。Decode 侧在被撤回的请求等候时继续轮询在途 transfer；PD 首 token 与首个 decode 输出合并发送。重叠调度等已发起的 forward 后再写入新权重。CUDA graph 重捕获前释放旧 decode graphs。 P/D 与在线换权重部署可关注撤回过程中的队列进展、forward/写入顺序和重捕获峰值。 [#51077](https://github.com/sgl-project/sglang/pull/51077) · [#50712](https://github.com/sgl-project/sglang/pull/50712) · [#50717](https://github.com/sgl-project/sglang/pull/50717) · [#50895](https://github.com/sgl-project/sglang/pull/50895) · [#50982](https://github.com/sgl-project/sglang/pull/50982)

- **SGLang 合入 Kimi K3.1 day-0 剩余路径及服务能力。** Kimi K3.1 的 PD decode 默认可用不持有 recurrent state 的 radix cache，FlashMLA prefill token packing 按架构与 KV 规格启用，确定性模式有条件限制；另修复 idle attention-DP rank、KDA checkpoint dtype。Qwen3.8 Flash Next 可用 KVShard。Beam search 加入 trie head 与共享上下文 attention。 升级 Kimi 配置时可优先对照确定性设置下的 cache 行为和 idle DP rank。 [#48414](https://github.com/sgl-project/sglang/pull/48414) · [#50839](https://github.com/sgl-project/sglang/pull/50839) · [#44447](https://github.com/sgl-project/sglang/pull/44447) · [#44349](https://github.com/sgl-project/sglang/pull/44349)

- **LMCache 按 chunk 内存类型选择传输引擎，修复 CacheBlend 槽位所有权。** MP 批次切成同一引擎的连续 runs，主机内存用 GPU DMA，NVMe 可走文件侧 DMA；peer-owned NIC DMA 仍是会报错的占位实现，较晚 run 失败可能留下已经提交的较早 run。CacheBlend 可复用被淘汰的压缩 ID，仅在 slot 仍归被淘汰 chunk 时清理，同文本再次插入的 matcher 条目得以保留。 试用 DMA 时可区分主机/NVMe 的路径和块大小，确认部分提交失败时的上层处理。 [#6549](https://github.com/LMCache/LMCache/pull/6549) · [#5646](https://github.com/LMCache/LMCache/pull/5646) · [#6698](https://github.com/LMCache/LMCache/pull/6698)

- **Dynamo 修正 SGLang KV 重复上报并向路由策略提供 KVCR 占用。** Worker-domain SGLang store 按幂等事件处理，避免重复 store 后单次 remove 仍留索引；vLLM 等来源仍按计数约定处理。实验性的 worker-selection policy 可读 KVCR host cache 占用及容量，容量依赖本地可推导的 KVCR 配置。另接入 TokenSpeed headless engine 的 MessagePack wire。 策略评估可对比重复事件后的实际索引占用和 worker 选路。 [#18904](https://github.com/ai-dynamo/dynamo/pull/18904) · [#18466](https://github.com/ai-dynamo/dynamo/pull/18466) · [#18328](https://github.com/ai-dynamo/dynamo/pull/18328)

- **llm-d Router 接入 vLLM upstream gRPC 与 SGLang Generate 的 P/D。** Sidecar 与 coordinator 支持 vLLM gRPC P/D 和 encode disaggregation，coordinator 的默认 worker protocol 仍为 HTTP。SGLang `Generate` 由 `--kv-connector=sglang` 解析并带上 bootstrap 描述，其他 connector 维持原 decode-only 透传。Weighted-fair-share 同时修复流缺 usage 时漏计 prompt。 轮转 worker protocol 前可分别验 decode-only、P/D 与 encode 路由，核对流式用量。 [#4011](https://github.com/llm-d/llm-d-router/pull/4011) · [#4013](https://github.com/llm-d/llm-d-router/pull/4013) · [#3908](https://github.com/llm-d/llm-d-router/pull/3908) · [#4280](https://github.com/llm-d/llm-d-router/pull/4280)

### 多模态、低精度算子与编译运行时

- **vLLM-Omni 支持跨 runtime/node 的 duplex replicas，并延长 PersonaPlex 会话。** Stage 选择可覆盖远端 stage replica，同主机 shared memory 与跨节点 connector 须匹配部署。Stage 0 多副本使用 processor-only cache 路径，避免多模态 payload 依赖某一 receiver 的缓存。PersonaPlex 保持 3,000 帧滑窗，以绝对位置使用 RoPE，验证 46,080 的位置表对应约一小时；部署长度超过对应 RoPE 表会报错。 长双工会话压测可分别核对跨机负载传输、位置上限和每 stage 的 replica 归属。 [#10422](https://github.com/vllm-project/vllm-omni/pull/10422) · [#10539](https://github.com/vllm-project/vllm-omni/pull/10539) · [#8783](https://github.com/vllm-project/vllm-omni/pull/8783)

- **vLLM-Omni 扩展多模态输出执行路径。** Request-mode diffusion 可由模型 emitter 发送中间 `CHUNK_READY` 视频块，调用结束后关闭回调。MiniMax H3 可在未缓存请求用 Heun，固定 distilled schedule 仍走 Euler。JoyAI-VL-Interaction 接入完整 async chunk，MOSS-TTS-Realtime 接入 MRv2，并恢复 Qwen3-TTS、Qwen3-Omni Talker 的 FULL graph replay。 接入流式视频时可验证中间块数、末块判定及异常后的回调回收。 [#10049](https://github.com/vllm-project/vllm-omni/pull/10049) · [#10400](https://github.com/vllm-project/vllm-omni/pull/10400) · [#10281](https://github.com/vllm-project/vllm-omni/pull/10281) · [#9906](https://github.com/vllm-project/vllm-omni/pull/9906) · [#10503](https://github.com/vllm-project/vllm-omni/pull/10503) · [#10499](https://github.com/vllm-project/vllm-omni/pull/10499)

- **FlashInfer 主线多项修复涉及 GPU 资源寿命和 MoE 索引。** 已捕获的 CUDA graph 保留其 BF16×FP4/MoE cuDNN graph 对象，避免缓存淘汰后 replay 引用失效。PDL producer 的输入挪到依赖等待之后读取，warp split-K BF16 GEMM 调整释放与写回顺序；MoE 空 EP rank 加零行，Llama4 路由只按本地 experts 算 padding 和 CTAs，permute/finalize 索引改为 int64，并补足 FINALIZE autotuner workspace。 多图并存服务可把 cache 淘汰后的 replay 和空 EP rank 纳入回归。 [#8188](https://github.com/flashinfer-ai/flashinfer/pull/8188) · [#7288](https://github.com/flashinfer-ai/flashinfer/pull/7288) · [#8081](https://github.com/flashinfer-ai/flashinfer/pull/8081) · [#9514](https://github.com/flashinfer-ai/flashinfer/pull/9514) · [#9513](https://github.com/flashinfer-ai/flashinfer/pull/9513) · [#9512](https://github.com/flashinfer-ai/flashinfer/pull/9512) · [#9507](https://github.com/flashinfer-ai/flashinfer/pull/9507)

- **FlashInfer 主线扩展 DCP、混合精度和 batch-invariant 工具面。** DCP 交换支持 destination-major 输入与 token mask。Masked grouped GEMM epilogue 可生成 NVFP4；token-sparse attention 对重复访问的 NVFP4 KV 先按 key 转为 FP8。Batch-invariant 采样提供带温度及 log-prob 的训练前向接口。同窗另一笔移除了项目内原有 CAKE 训练 kernels。 依赖 CAKE 训练入口的调用方应先核对 release 与 main 的包内容和迁移目标。 [#6483](https://github.com/flashinfer-ai/flashinfer/pull/6483) · [#7851](https://github.com/flashinfer-ai/flashinfer/pull/7851) · [#9527](https://github.com/flashinfer-ai/flashinfer/pull/9527) · [#9535](https://github.com/flashinfer-ai/flashinfer/pull/9535) · [#9326](https://github.com/flashinfer-ai/flashinfer/pull/9326)

- **FlashAttention 修复 FA3 尾部 NaN 污染，FA4 拓展 SM100 MXFP8。** FA3 在 GEMM 读取前清零 `seqlen_k` 之后的 K/V 行，新增变长、分页和 backward 覆盖。FA4 在 SM100 系列引入 native MXFP8 PV forward，当前 blockscaled PV 约束为已有 blockscaled QK 的非因果、非 local 路径。 回归可优先覆盖分页末页的异形输入，以及 blockscaled PV 的 mask 选择。 [#3174](https://github.com/Dao-AILab/flash-attention/pull/3174) · [#3020](https://github.com/Dao-AILab/flash-attention/pull/3020)

- **llama.cpp 修复 Qwen4 实验模型 FP8 转换和图复用。** 转换区分 n-gram table scale 和 block-scaled FP8，推理避免 n-gram scale 重复触发 graph 重分配。序列编辑保持 k-pool layout 的前缀，server slot scheduler 修复 LoRA 与 recurrent 请求判定，Qwen3Next 补上 output projection weight scale。 已有 FP8 GGUF 转换产物可用同一 prompt 对照输出与图重建频率。 [#32476](https://github.com/ggml-org/llama.cpp/pull/32476) · [#32483](https://github.com/ggml-org/llama.cpp/pull/32483) · [#32137](https://github.com/ggml-org/llama.cpp/pull/32137) · [#32434](https://github.com/ggml-org/llama.cpp/pull/32434) · [#32393](https://github.com/ggml-org/llama.cpp/pull/32393)

- **PyTorch 提升分片检查点读入并修复分布式边界。** FsspecReader 用批量 `cat_ranges` 和异步预取读 checkpoint。FSDP2 对 fused chunk-cat 的 cast 路径向量化，size-1 shard mesh 省掉多余复制和 reduce-scatter 缓冲。NCCL2 `gather_single` 修复多维输入。 大型对象存储 checkpoint 可同时量读请求数、恢复耗时和实际内存峰值。 [#194881](https://github.com/pytorch/pytorch/pull/194881) · [#204752](https://github.com/pytorch/pytorch/pull/204752) · [#204750](https://github.com/pytorch/pytorch/pull/204750) · [#204942](https://github.com/pytorch/pytorch/pull/204942) · [#205316](https://github.com/pytorch/pytorch/pull/205316)

- **PyTorch 扩展编译器用户 kernel 并修复精度和线程边界。** CuTeDSL 用户 kernel 接入 host 签名与 eager launcher，常量参数 side table 按内容去重；Inductor 支持嵌套 compile region 内的 max-autotune GEMM，以及面向树外 backend 的 graph trees 接口。Triton 代码生成修正 float64 常数字面量误截断，Dynamo 访问 code object ExtraState 前加锁。 编译链可分别验证 eager launch、嵌套区域的 autotune 开关和 float64 常量表达式。 [#203537](https://github.com/pytorch/pytorch/pull/203537) · [#203536](https://github.com/pytorch/pytorch/pull/203536) · [#205099](https://github.com/pytorch/pytorch/pull/205099) · [#194598](https://github.com/pytorch/pytorch/pull/194598) · [#203235](https://github.com/pytorch/pytorch/pull/203235) · [#201685](https://github.com/pytorch/pytorch/pull/201685)

- **Transformers 修复缓存、MoE 精度和 SFT mask。** BigBird block-sparse decode 正确接续 KV cache；AFMoE 在 BF16 权重加载时保持 FP32 expert bias；chat template 在左截断后正确裁切 assistant token mask。Continuous batching top-k 防越界，原生 PP 分配也把层更均匀铺到各 stage；无 added token 候选时跳过 trie 扫描。 训练回归优先检查左截断 mask 和 AFMoE router bias 的加载 dtype。 [#49984](https://github.com/huggingface/transformers/pull/49984) · [#50687](https://github.com/huggingface/transformers/pull/50687) · [#50758](https://github.com/huggingface/transformers/pull/50758) · [#50282](https://github.com/huggingface/transformers/pull/50282) · [#50693](https://github.com/huggingface/transformers/pull/50693) · [#50781](https://github.com/huggingface/transformers/pull/50781)

### 扩展项目

- **SpecForge 的 draft 训练切换到统一 loss 与配置。** DFlash、DSpark 等 objective 共用数值 primitive 和统一执行层，`training.loss.terms` 逐项声明，fused 和 backward checkpointing 独立。Flat loss 配置会报错，旧运行完整恢复需按迁移规则处理。EAGLE 缓存 rollout 接入 sequence parallel，试验 recipe 和 canonical e2e 同步合入。 迁移旧 draft recipe 时可先核对 loss 默认项、权重迁移与完整 resume 是否走各自入口。 [#1175](https://github.com/sgl-project/SpecForge/pull/1175) · [#1181](https://github.com/sgl-project/SpecForge/pull/1181) · [#1182](https://github.com/sgl-project/SpecForge/pull/1182) · [#1183](https://github.com/sgl-project/SpecForge/pull/1183) · [#1184](https://github.com/sgl-project/SpecForge/pull/1184) · [#1185](https://github.com/sgl-project/SpecForge/pull/1185) · [#1186](https://github.com/sgl-project/SpecForge/pull/1186) · [#1193](https://github.com/sgl-project/SpecForge/pull/1193) · [#1194](https://github.com/sgl-project/SpecForge/pull/1194)

- **verl-SpeCo 整合 Ray draft 训练入口。** Standalone workflow 同时支持在线生成和落盘的 raw hidden-state 数据，生产端可记录并重试按条失败。CPU Ray producer 的 TransferQueue 路径不支持依赖可见 NPU 的 Ascend Mooncake 传输。 可复用隐藏状态数据做重复实验，先验证在线和离线路径的样本拼装一致。 [#142](https://github.com/verl-project/verl-SpeCo/pull/142) · [#151](https://github.com/verl-project/verl-SpeCo/pull/151)

- **VeOmni 让符合条件的同模态批次略过模块间 SP all-gather。** 文本或视觉的单模态 shard 微批可保持各 rank 的序列切片，loss 经 SP reduction 合成；跨模态常规流程仍由对应模块传递完整表征。NVIDIA CUDA 路径合并 MoE SwiGLU backward 并保持分阶段舍入。Qwen4-Exp 可单独冻结 PLE embedding table。 混合数据并行训练可分别验证文本、视觉与混合微批的 loss 和通信量。 [#1455](https://github.com/ByteDance-Seed/VeOmni/pull/1455) · [#1467](https://github.com/ByteDance-Seed/VeOmni/pull/1467) · [#1354](https://github.com/ByteDance-Seed/VeOmni/pull/1354)

- **NeMo AutoModel 试验低秩梯度、NCCL-EP 和 packed FA4。** LoGRA 将选定 dense linear 的梯度在 backward 投影到低秩 sketch，优化器状态由 DCP 管理，当前接入 LLM finetune recipe。实验 NCCL-EP 是 opt-in BF16 dispatcher，要求 NCCL≥2.30.4。GPT-OSS packed FA4 保留 learned attention sinks，并为 physical padding 检查依赖能力。带 reasoning 的 mask-history 另修复最终轮前缀。 优化器显存对照应同时记录 sketch rank、训练曲线和 checkpoint 重启行为。 [#4758](https://github.com/NVIDIA-NeMo/Automodel/pull/4758) · [#4384](https://github.com/NVIDIA-NeMo/Automodel/pull/4384) · [#4681](https://github.com/NVIDIA-NeMo/Automodel/pull/4681) · [#4811](https://github.com/NVIDIA-NeMo/Automodel/pull/4811)

- **Prime-RL 统一预训练、SFT 和 RL 的权重初始化约定。** `model.init.type` 支持 scratch 或 pretrained，trainer-fed inference 可先加载 dummy weights，LoRA 仍要求有基座权重。Checkpoint 保存模型配置和 tokenizer，resume 核验架构；新增 scratch 测试覆盖多个模型原生实现。 从零启动 RL 前可跑一次小型 trainer→inference 权重发布和续训。 [#4476](https://github.com/PrimeIntellect-ai/prime-rl/pull/4476) · [#4477](https://github.com/PrimeIntellect-ai/prime-rl/pull/4477) · [#4433](https://github.com/PrimeIntellect-ai/prime-rl/pull/4433) · [#4518](https://github.com/PrimeIntellect-ai/prime-rl/pull/4518)

- **Prime-RL 改善 DeepSeek-V4 长上下文训练路径。** CP 只交换 compressor/indexer 所需 halo，梯度返还发送 rank；受支持的 SM100+ 路径用 cuDNN fused CSA/HCA pooling。Q norm+RoPE 与 indexer RoPE+quant 复用 vLLM kernel，mHC 改写 backward。Selective AC 可指定模块路径。 长上下文调参可同时观察 halo 交换字节、kernel 选路与梯度误差。 [#4354](https://github.com/PrimeIntellect-ai/prime-rl/pull/4354) · [#4355](https://github.com/PrimeIntellect-ai/prime-rl/pull/4355) · [#4356](https://github.com/PrimeIntellect-ai/prime-rl/pull/4356) · [#4357](https://github.com/PrimeIntellect-ai/prime-rl/pull/4357) · [#4358](https://github.com/PrimeIntellect-ai/prime-rl/pull/4358)

- **TRL 主线增加 vLLM 0.34.x 兼容与 score-centering 配置校验。** 在 top-k/top-p 截断条件下按实际 sampler 选项拒绝无效组合；兼容改动来自主线提交，1.18.1 的发布说明单独核查。 升级 TRL 与 vLLM 时应核对已安装 tag 支持范围，以及 rollout sampler 参数。 [#8414](https://github.com/huggingface/trl/pull/8414) · [#8418](https://github.com/huggingface/trl/pull/8418)

- **PEFT 修复临时禁用 adapter 与 CPT 输入语义。** MixedModel 退出 `disable_adapter` 恢复进入前的 adapter 状态；CPT 保留输入 token 角色。 嵌套 adapter 上下文和角色敏感输入可分别做最小回归。 [#3910](https://github.com/huggingface/peft/pull/3910) · [#3953](https://github.com/huggingface/peft/pull/3953)

- **Attention Gym 增加 paged sparse decode 参考与 Triton 路径。** Decode split 的结果只在合并阶段归一化，减少重复工作；Qwen WarpDrive megakernel 改为每层一次 launch，新 token 合到最后一个非空 KV split。 比较 paged sparse decode 路径时可固定索引密度，单测合并后的数值。 [#791](https://github.com/meta-pytorch/attention-gym/pull/791) · [#792](https://github.com/meta-pytorch/attention-gym/pull/792) · [#793](https://github.com/meta-pytorch/attention-gym/pull/793) · [#794](https://github.com/meta-pytorch/attention-gym/pull/794) · [#946](https://github.com/meta-pytorch/attention-gym/pull/946) · [#948](https://github.com/meta-pytorch/attention-gym/pull/948) · [#950](https://github.com/meta-pytorch/attention-gym/pull/950)

- **Helion 增加可版本化的树外 backend 接口。** 公开 `helion.backend_api`，覆盖 device 源码完成钩子与 FlyDSL reduction helper；Triton Gluon backend 增加 ring megakernel lowering，另提供 paged-attention kernel 示例。 树外 backend 可对照公开版本接口检查编译钩子与 reduction lowering。 [#4271](https://github.com/pytorch/helion/pull/4271) · [#4272](https://github.com/pytorch/helion/pull/4272) · [#4406](https://github.com/pytorch/helion/pull/4406) · [#4487](https://github.com/pytorch/helion/pull/4487)

- **Xinference 在 consumer 关闭时停止串行 ModelActor 流。** 清理 generator 避免中断连接后仍持有串行执行资源。 已有流式服务可检查客户端取消后 actor slot 是否及时释放。 [#6130](https://github.com/xorbitsai/inference/pull/6130)

### 回溯补录

- **11 月 16 日补录 FlashInfer 0.7.5rc2。** PyPI 首个工件 09:03:19 UTC。RC1→RC2 差分修复 GDN prefill 的元素对齐输入、strided paged metadata、显式长度的 SM120 cuDNN prefill 路由，以及 SM107 批量 NVFP4 quant dispatch。已知的 0.7.3 在此前已正式发布，本次 Atom 时间变化未产生新版本。 仍锁定 RC1 的部署可先核对 paged metadata 与目标卡上的 FP4 dispatch。 [RC1→RC2 差分](https://github.com/flashinfer-ai/flashinfer/compare/v0.7.5rc1...v0.7.5rc2) · [PyPI](https://pypi.org/project/flashinfer-python/0.7.5rc2/) [官方 tag](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.7.5rc2)

- **11 月 12–13 日补录 Prime-RL 训练快照和 draft 切换。** 异步 DCP 把 DTensor local shard 暂存到 pinned host memory，下一次 optimizer step 前等待 staging，与后台写盘分开。Resume 跳过没有 DCP `.metadata` 的不完整步。投机解码支持控制 API 热换 draft 与采样回放，依赖 Prime-RL 自有 engine admin 路由和兼容的 weight broadcast 后端。 长跑可验证切断写盘后的恢复点，以及 draft 更新前后的采样回放。 [#4428](https://github.com/PrimeIntellect-ai/prime-rl/pull/4428) · [#4466](https://github.com/PrimeIntellect-ai/prime-rl/pull/4466)

- **11 月 12–13 日补录 vLLM-Omni 的权重更新与状态正确性。** HunyuanImage3 接入 full-parameter RL rollout 的权重更新链路；AR→DiT 边界传递模型实际采样 token ID，避免 decode→re-encode 改写 BPE 后破坏 KV 命中。Qwen-Image-2.1 修复 decode graph pool 生命周期和 stream 恢复；MiMo-Audio 修复 eager/graph 的 seeded sampling。 多阶段后训练可把一个权重发布周期和 AR token ID 直通纳入端到端检查。 [#9535](https://github.com/vllm-project/vllm-omni/pull/9535) · [#9323](https://github.com/vllm-project/vllm-omni/pull/9323) · [#10268](https://github.com/vllm-project/vllm-omni/pull/10268) · [#10241](https://github.com/vllm-project/vllm-omni/pull/10241)

- **11 月 12–13 日补录 Xinference 的共享 KV 和取消路径。** Tensor-parallel/量化 SGLang 与 DeepSeek-V4/V4.1 可共享同引擎 HiCache host pools。TensorRT-LLM 1.4 的 Xavier 原生缓存可按 TP rank 保存 KV，完整命中要求各 rank 均可读，P/D 仍要求 TP=1。Streaming 请求维持注册直到结束，让 abort 找到实际运行的流。 共享 KV 试验应分别确认 TP 全体 rank 命中和客户端取消后的流释放。 [#6085](https://github.com/xorbitsai/inference/pull/6085) · [#6088](https://github.com/xorbitsai/inference/pull/6088) · [#6078](https://github.com/xorbitsai/inference/pull/6078)

<a id="topic-7"></a>
## 硬件与软件协同

### TensorRT-LLM 把逻辑取消与 KV 物理回收解耦

**UTC 11-18 00:25 · 主线 c2281e5b44。** 具独立物理所有权的 context 传输可先接受逻辑取消，执行器完成逻辑结果，已经公开的 send 继续 drain。KV 释放仍等待安全回收与 TP 组同意，超时无法绕过该栅栏。代码新增 disaggregated 测试。**考虑复现**，覆盖 decode 层早断、多个 TP rank 中一侧延迟及发送刚公开时取消，检查同一块不会重用或重复回调。

[NVIDIA/TensorRT-LLM c2281e5b44](https://github.com/NVIDIA/TensorRT-LLM/commit/c2281e5b44018b8a032da2bb0408eb8e6cc2dc26)

### vLLM Gaudi 修复同一步前缀读取与 Nemotron-H 长提示

**UTC 11-17 15:05、11-17 11:00 · 主线 bd8e1f65f0、f724715ea7。** #1918 在 HPU runner 分开执行 prefill forward 的情况下按前缀块数据依赖调序，防止本轮后写的共享块被另一请求提前读。#1990 修复 128 token bucket 换算至 528 token attention block 时整数比例为零，导致四卡 Nemotron-H 的后续 chunk 或 cache hit 丢失既有上下文，八卡的 128→128 路径保持原状。单测覆盖调序及非整比例边界。**应优先复验**，固定四卡张量布局，交替安排同前缀请求、换行位和长上下文召回。

[vllm-project/vllm-gaudi bd8e1f65f0](https://github.com/vllm-project/vllm-gaudi/commit/bd8e1f65f017cedb2d2bb5135346e3b6c5d9256a) · [vllm-project/vllm-gaudi f724715ea7](https://github.com/vllm-project/vllm-gaudi/commit/f724715ea72da0e6395d1c5b46473957dfcd74b0)

### vLLM Ascend 给共享 KV 和 P/D 增加明确的持有层

**UTC 11-17 13:53、11-17 11:04、11-17 11:23 · 主线 4bf5d364db、5d90273879、8d6180e28e。** DeepSeek-V4.1 的 GVA layerwise pooling 只传持久 KV 的源层，复用原 Memcache 配置和队列，可在受支持配置下与 Mooncake V2 Push 经 MultiConnector 联用。此 DS4.1 路径要求 Model Runner V2 和 eager，支持 TP/PCP，超出范围的布局包括 PP、DCP、KVPP 和 graph replay。另一个合入面向受支持的非混合 MLA/SFA，Mooncake Push 按生产者注册时记录的 KVPP 层所有权发送，避免由目标 rank 推算的源照样发 DONE 却缺 layer。该独立路径要求 prefill DCP=1，decode 无 KVPP。CPU tier 的共享 mmap 可选择 pin，复制布局仍按槽注册，复制式 MLA 注册整映射，初始化按逻辑 worker 分摊。**考虑复现**，逐层核对 KV 完整性、MTP 副本、首次 fault 与 steady-state 搬运，量化锁页和多进程部署开销。当前为主线能力。

[vllm-project/vllm-ascend 4bf5d364db](https://github.com/vllm-project/vllm-ascend/commit/4bf5d364db834417a4b35e419bb632e91fcb964e) · [vllm-project/vllm-ascend 5d90273879](https://github.com/vllm-project/vllm-ascend/commit/5d90273879b19c30ab770fe41e2ee4e626a58eab) · [vllm-project/vllm-ascend 8d6180e28e](https://github.com/vllm-project/vllm-ascend/commit/8d6180e28e0941fcfc85999bf2aafe194e98f8f6)

### vLLM Ascend 在不适配的 MXFP 权重和工作区上提前报错

**UTC 11-17 13:55、11-17 13:06 · 主线 b9c32b689a、50c4dab079。** MXFP 权重量化在方法构造时检查设备 MX MatMul 能力，覆盖多种 MXFP4/MXFP8 Linear 与 MoE。MXFP8 KV cache 仍由 attention 后端自己的能力门控。Batch-invariant 模式同步切换 fused infer attention 的 workspace query，旧扩展缺符号时在图捕获调用处报错，避免使用普通 kernel 的较小工作区。**考虑复现**，用 A3 的 CANN 9.2.0 与目标机型分别跑模型加载和 full graph capture。差分包含 mock 回归，缺多模型端到端性能。

[vllm-project/vllm-ascend b9c32b689a](https://github.com/vllm-project/vllm-ascend/commit/b9c32b689afbe43b8d49e956990cdfdd86d63c1d) · [vllm-project/vllm-ascend 50c4dab079](https://github.com/vllm-project/vllm-ascend/commit/50c4dab079304fb18da96157d87188c36826e1eb)

### SGLang TorchTPU 将 DSA 池化索引的原始行放进请求级环

**UTC 11-17 07:39、11-17 19:20、11-17 06:36 · 主线 cc781eee6c、13be33dbbc、50401445c4。** GLM-5.3-Flash 的 11 层 DSA 以前为每 token 每 rank 保存 5,632 B 原始 BF16 key/gate。可选环只保留仍开放池需要的行，也接入 MTP，作者给的 200 万 token 池目标缓存为每 rank 26.13→15.26 GiB。启用时拒绝 P/D、prefill CP 和无 chunk prefill。另两个合入分别给 SparseCore 测试布局实现 64 B/token 的 RoPE 打包（原 128 B，latent 合计 640→576 B/token，要求 head_dim≤64 且页布局避免 XLA tile padding），以及在 DSA 长上下文 prefill 按块实际最大页缩短 gather 循环。RoPE 差分明确生产 sparse MLA 仍走 TensorCore 布局。**考虑复现**，分开测请求级环的 slot 周转、256K 以上 gather 和生产缓存布局。

[sgl-project/sglang-torchtpu cc781eee6c](https://github.com/sgl-project/sglang-torchtpu/commit/cc781eee6c0e78164d3f47444c0ac431dc098a65) · [sgl-project/sglang-torchtpu 13be33dbbc](https://github.com/sgl-project/sglang-torchtpu/commit/13be33dbbcca34aaf2aff0a0039d54a6906ecd92) · [sgl-project/sglang-torchtpu 50401445c4](https://github.com/sgl-project/sglang-torchtpu/commit/50401445c47780ca4bc0b36cac3dfbfc17120624)

### vLLM TorchTPU 对齐各 DP rank 的 prompt logprob collective

**UTC 11-17 19:08 · 主线 74fe441785。** 各 rank 按当步请求中要打分的最大 prompt 行数算调用上界，行少或为零的 rank 用 dummy logits 参与余下 collective，且仅最后一个 PP stage 打分，避免锁步 TP/DP 组合里 collective 次数不同。**考虑复现**，覆盖仅一 rank 请求 prompt logprobs、空步与多 chunk，同步观测尾延迟及返回 token 序列。

[vllm-project/vllm-torchtpu 74fe441785](https://github.com/vllm-project/vllm-torchtpu/commit/74fe44178512ee0447dc44445a911a267dc65f6e)

### TorchTPU 增加 INT8 MatMul 与不等长 Splash Attention

**UTC 11-17 21:58、11-17 19:20 · 主线 63fd57aa9e、6d7003cb8d。** INT8 矩阵乘增加 INT32 累加语义的实现与测试。Splash SDPA 面向 cross-attention 支持 query 与 KV 轴不同长度，并增加 Pallas 形状与输出对照用例。**考虑复现**，给量化 linear 校验边界累加，给 cross-attention 校验非方形 mask 与块边界。证据为主线差分和新增用例。

[google-pytorch/torch_tpu 63fd57aa9e](https://github.com/google-pytorch/torch_tpu/commit/63fd57aa9e4b1b7eee7f55dff4fb292e489653e8) · [google-pytorch/torch_tpu 6d7003cb8d](https://github.com/google-pytorch/torch_tpu/commit/6d7003cb8d1b9303c9d457d7589d4ff9e0941e1c)

### hipBLASLt 在 gfx950 支持混合 MXFP4×MXFP8 Grouped GEMM

**UTC 11-17 23:50、11-17 20:43 · 主线 824addb774、0dfc9fd835。** Grouped GEMM 允许 A 与 B 类型不同，每个 operand 在所有 group 仍各自同型。TensileLite 分别修正两个 operand 的 K 尾部掩码、读宽和地址上界，以 BF16 输出及预 swizzle 的 UE8M0 scale 提供 gfx950 解法。DeepSeek V4 Flash TP4 作者调参形状覆盖 gate_up 和 K=576 的 down GEMM，去除 MXFP4 权重转成 MXFP8 后约 1.94 倍权重字节读入。另一个合入让本地自定义 kernel 注册可通过指定目录跨进程复用。**考虑复现**，看专家 group 大小、尾部边界及服务端复用调优工件，整模型收益仍待测。

[ROCm/rocm-libraries 824addb774](https://github.com/ROCm/rocm-libraries/commit/824addb7749c947f753a78f8d6138ba86644a5cf) · [ROCm/rocm-libraries 0dfc9fd835](https://github.com/ROCm/rocm-libraries/commit/0dfc9fd835c1abb7389dfe34320deb546001cd40)

### MORI-IO 增加 peer 存活检测与 Fabric 批量复制

**UTC 11-17 14:29、11-17 07:19 · 主线 c5bdc39136、eba92ef60c。** 注册声明 RAS endpoint 的远端 IOEngine 后，启用 RAS 的运行时维护 TCP peer 链路，用显式离开、链路关闭与 20 秒心跳超时区分失败。Fabric 路径增加批量提交并报告选路被拒原因。**考虑复现**，覆盖主动退出、静默 peer 与链路恢复，分别测批量复制的 enqueue 开销及回退。所查代码为机制合入，缺与实际 KV serving 配置对应的吞吐和恢复时间。

[ROCm/mori c5bdc39136](https://github.com/ROCm/mori/commit/c5bdc3913660f6b663ae09ba7c7176291d6f5343) · [ROCm/mori eba92ef60c](https://github.com/ROCm/mori/commit/eba92ef60cd52b058af3197d3e76661fa8efe8bf)

### Ascend CANN 扩展 A5 路由容量，补充 FP4 与小 batch 算子

**UTC 11-17 06:22、11-17 06:34、11-17 07:45 · 主线 20abf858b7、c3042b4fe3、f3e070d052。** 普通 A5 fullmesh_v2 Dispatch 与 MTE Combine 放开到 TopK32，分块预算含 Flag 与 Sum 临时区，目标限定非分层，Dispatch 排除 RoCE，Combine 排除 AddRmsNorm 融合。作者给出 Ascend 950、CANN 9.3.0、8 卡、49 条性能及 19 条功能用例。DynamicMxQuant 的 scale_alg=3 支持 950 FP16/BF16 偶数尾轴的 NVFP4 block16（E2M1 数据、E4M3FN scale），格式调整后的完整包通过 3,560 原生和 120 连续执行精度用例，85 条 TTK 有中断，性能仅来自前一实现阶段。mHC tiny 在 910B3 以 8 AIC+16 AIV 代替 20+40。作者用 msprof 比较无梯度的两组小 shape，(1,6,4,5120) 31.10→20.69 μs，(1,3,4,3584) 30.04→19.04 μs，默认门控受 shape 与片上容量约束。**考虑复现**，按路由、格式、边界 shape 分别验证，算子指标无法外推至系统收益。

[cann/ops-transformer !15967](https://gitcode.com/cann/ops-transformer/merge_requests/15967) · [cann/ops-nn !13691](https://gitcode.com/cann/ops-nn/merge_requests/13691) · [cann/ops-transformer !16117](https://gitcode.com/cann/ops-transformer/merge_requests/16117)

### CANN 试验性 FA4 入口落地，另有两处回退

**UTC 11-17 01:11、11-17 01:21、11-17 06:06、11-17 04:08 · 主线 2d37a19ae3、3b4fe8226a、183cf4d706、21c7c63346。** FA4 兼容入口把受支持的密集无 mask 前向投影到 flash_attn_exp，拒绝不支持参数与反向。作者在 Ascend 950、CANN 9.3 验证 20 组，并在六组合规输入中以相同缩放参数与现有算子对照。A2/A3 INT8 prefill 的 !16015 同日被 !16123 撤回。HCCL !4105 撤回上期记载的 !3875 所增 AIV/CCU/AICPU 卡死计数插桩，维护者给出的原因是 hcomm 侧接口目前无法用于端到端验证。**继续观察**，入口升级先核对默认 scale，卡死诊断待 hcomm 能力与后续主线版本可核。

[cann/ops-transformer !16019](https://gitcode.com/cann/ops-transformer/merge_requests/16019) · [cann/ops-transformer !16015](https://gitcode.com/cann/ops-transformer/merge_requests/16015) · [cann/ops-transformer !16123](https://gitcode.com/cann/ops-transformer/merge_requests/16123) · [cann/hccl !4105](https://gitcode.com/cann/hccl/merge_requests/4105)

### HCCL 合入实验性 Python 通信绑定与二级 AllGather

**UTC 11-17 11:39、11-17 06:21 · 主线 df8ce97148、c6759fa80b。** hccl4py 在实验目录用 ctypes 封装 14 个通信算子，通信域管理另有 hcomm 依赖。作者用 950PR、CANN 9.2、单机两卡做功能与边界验收，报告相对 C 接口开销 ≤11 μs，详细验收日志未公开。另一个合入以 AscendC AIV 直驱 URMA 实现二级 sequence AllGather，作者在 2×8 组网测试 mesh×mesh、GE 和 ACL superkernel。**考虑复现**，分别核对载入的 hcomm 版本、进程内调用开销与跨节点边界数据量。

[cann/hccl !3798](https://gitcode.com/cann/hccl/merge_requests/3798) · [cann/hccl !3722](https://gitcode.com/cann/hccl/merge_requests/3722)

### Ascend/PyTorch 修复 NPU Triton int64 除法并打开稀疏外层调度

**UTC 11-17 12:59、11-17 13:46 · 主线 fe7dfe2e38、da5475419d。** 整数除法按 FX 输出 dtype 选择 int64 实现，覆盖大于 32 位的正负值、不同舍入和融合图用例。FlexAttention 前向模板允许稀疏路径启用外层调度。**考虑复现**，升级后检查索引类大整数路径，并用相同 block mask 测调度收益。主线差分暂无整模型时延数据。

[Ascend/pytorch fe7dfe2e38](https://github.com/Ascend/pytorch/commit/fe7dfe2e3827fadb3478e73259f4599dbfb996e7) · [Ascend/pytorch da5475419d](https://github.com/Ascend/pytorch/commit/da5475419dacfba92868249b5b14bb1a4644be3e)

### Intel D270 公布预生产单卡推理条件，补上异构 P/D 设计

Intel · 11 月 16 日发布，11 月 17 日有页面更正。160 GB LPDDR5X、1.5 TB/s 的 D270 与 SN50 分别负责 prefill 和 decode 的 SC26 展示。单卡对 RTX PRO 6000 Blackwell Server Edition，Intel 报告 Qwen3-30B-A3B FP8 吞吐 1.12×，TP1、输入/输出各 1,024、并发 1/8/16/32、FP8 KV、前缀缓存关闭。Drex 2.0 FP8 在 Jev latency-v1、750 行、并发 1、图预热后报告最多低 29% 时延。附录中端到端主机、vLLM/SGLang 版本、attention 内核与编译配置不同，功耗是上限设置，异构 P/D 无同条件吞吐和 KV 传输数据。**继续观察**，优先等待公开的运行配置及同负载 P/D 对照。

[一手来源](https://www.intel.com/content/www/us/en/newsroom/news/data-center/why-intel-gpu-d270-belongs-in-the-hpc-conversation.html) · [SambaNova 分离方案](https://sambanova.ai/blog/the-air-cooled-token-factory-intel-gpu-d-series-sambarack-sn50)

### 四台 DGX Spark 以直连环组网

NVIDIA · 11 月 16 日 · 补录。四个 GB10 节点经 ConnectX-7 与邻居直连成环，合计 512 GB 内存、16 TB NVMe，套件由 OEM 和 Marketplace 销售。厂商列出 NVFP4 模型容量可达 700B，示例 DeepSeek-V4.1。HPC SDK 安装手册仍待发布，本轮未找到该四节点 LLM 例子的请求长度、并发、吞吐或跨网通信实测。**继续观察**，待获得固定部署配置后用同一模型分开量测单请求 TPOT、批吞吐和 KV 余量。

[一手来源](https://blogs.nvidia.com/blog/dgx-spark-hpc-developer-kit-sc26/)

### Lux 开放 4,032 张 MI355X 的科学计算集群

ORNL/OLCF · 11 月 16 日 · 补录。Lux 面向科学用户开放，504 节点分为 Slurm 与 Kubernetes 池，每节点 8×MI355X、每卡 288 GB HBM3E，采用 Infinity Fabric、Pensando 网络和 ROCm。公开用户手册给出 NUMA/GPU 对应与部署环境。220.79 PFLOPS 为 TOP500 HPL 结果，LLM 吞吐、低精度训练扩展或实际开放配额未由该指标给出。**继续观察**，读用户指南核对拓扑，等公开多机训推日志再评估迁移。

[一手来源](https://www.olcf.ornl.gov/2026/11/16/lux-genesis-mission/) · [用户指南](https://docs.olcf.ornl.gov/systems/lux_user_guide.html) · [AMD 公告](https://newsroom.amd.com/news/sc26-amd-expands-frontier-scientific-computing/)

### SGLang TorchTPU 缩短 DeepSeek-V4.1 Flash 冷启动

**UTC 11-17 02:20、11-17 08:51 · 主线 a9ebad95d0、361a36354c。** direct tracer 明确暴露 KV、压缩状态和 metadata，允许 SPMD compile，裁掉巨大的 host Engram 输入。作者在 TPU7x-8、attention DP8+EP8、冷缓存对比 Dynamo，7 图、8 rank 的验证输出一致，decoder warmup 1,788→313 s，服务就绪 46.7→22.2 分钟。后续 patch 把 MXFP4 expert 转换留在 device（post-load 357→25 s，作者报告逐字节相同），支持 GCS 对象范围读取 Engram 并重新绑定编译后的输入。在 GCS 加载、设备 repack 的不同配置，报告 9.3 分钟就绪。**考虑复现**，用固定 checkpoint 和对象存储逐阶段测启动，热态吞吐另计。

[sgl-project/sglang-torchtpu a9ebad95d0](https://github.com/sgl-project/sglang-torchtpu/commit/a9ebad95d0c8608343562027ca4d98cb0a8c5439) · [sgl-project/sglang-torchtpu 361a36354c](https://github.com/sgl-project/sglang-torchtpu/commit/361a36354cbb9bd70bd7c20b3cf9a23b1545cbcb)

### TensorRT-LLM 扩展 Kimi K3 NVFP4 Rubin 布局并简化词表贪心验证

**UTC 11-18 00:44、11-18 00:25、11-17 17:49 · 主线 134e90552f、390891c050、2a8d7346d8。** SM107 的 modeling_v2 加入分离式 TP1 context 和三条 generation 布局，分别为 attention DP+MoE EP16，以及 Helix CP8 搭配 MoE EP8 或 TP2×EP4。单模型 spec worker 在合规路径可按词表分片找最大值，再完成跨 rank 归并，跳过全词表 logits all-gather。另一个合入在 DeepSeek-V4.1 CSA2 中按 32→8→1 位置分层筛选，按索引处理相同分数，支持的 TopK 与 fallback 在代码中有门控。**考虑复现**，对齐 layout 与 sampling 设置后验证等分数及长上下文，再对比通信字节和延迟。

[NVIDIA/TensorRT-LLM 134e90552f](https://github.com/NVIDIA/TensorRT-LLM/commit/134e90552fbb5f565149f3be690b67c70a2c15cc) · [NVIDIA/TensorRT-LLM 390891c050](https://github.com/NVIDIA/TensorRT-LLM/commit/390891c0504de67ab65ce0c3c27395b4d409646f) · [NVIDIA/TensorRT-LLM 2a8d7346d8](https://github.com/NVIDIA/TensorRT-LLM/commit/2a8d7346d8fb0f7e786621dc1bdfb068b2283ae5)

### AMD AITER 在 gfx1250 合并 DCP 交换与 LSE

**UTC 11-17 07:51、11-17 17:59、11-17 16:16 · 主线 03f9f3757f、d6ea3d1a26、ecd79bcb5d。** DCP 用低延迟 scratch 在一次 launch 中交换各 rank 的局部注意力结果并按 LSE 归并。限 BF16 输出特征维 128 与 FP32 LSE，超出容量或不适配布局仍回退。作者的原路径数据为 Kimi-K3 TP8+DCP8、54 token 时每全注意力层 40 μs，公开差分补了多卡 oracle 与 graph replay，当前缺同口径整模型新旧对比。另两个算子把 sparse paged decode 合规尾处理的 FP8 量化留在寄存器，以及在 gfx950 路径把 RoPE、FP8 KV 写入与 BF16 MLA BMM 融合。**考虑复现**，覆盖全部 mask、NaN LSE、padding 与小批路径的等价性，再测实际服务。

[ROCm/aiter 03f9f3757f](https://github.com/ROCm/aiter/commit/03f9f3757fb5ff10916411ca08b6f5a80296f016) · [ROCm/aiter d6ea3d1a26](https://github.com/ROCm/aiter/commit/d6ea3d1a26dbe055f3a1b28e9de8275902465f46) · [ROCm/aiter ecd79bcb5d](https://github.com/ROCm/aiter/commit/ecd79bcb5d8a3fb5555679c7efe316bcf8fafc8b)

### TensileLite 用 wave 扫描替代逐组依赖读

**UTC 11-17 00:12 · 主线 54ead5d8d7。** 补录。gfx942/gfx950 满足 tile 和寄存器条件的 grouped GEMM 让 64 lanes 并行读取 group header，并用前缀和定位 workgroup 的输出 tile。单 group、其他架构、不能保持占用率的配置走原路径。作者的 MI355X 原路径在 384 组和一个 token 的 DeepSeek V4 down projection 为 2,313 μs，去掉空组时为 15 μs，两数用来展示空组开销。**考虑复现**，对照 group 非空率、稀疏 decode、host heuristic 和不同逻辑库裁剪，避免用该极端形状估整模提速。

[ROCm/rocm-libraries 54ead5d8d7](https://github.com/ROCm/rocm-libraries/commit/54ead5d8d74a792689a56b6a2082e5836c22d6d4)

### Transformer Engine 补齐门控激活与分组 MXFP8 路径

**UTC 11-17 21:58、11-17 16:47 · 主线 8e89d3ad8b、ef7f51595c。** CuTeDSL 门控 MXFP8 覆盖 GeGLU、QGeGLU、SwiGLU 与 clamped SwiGLU，并加前后向精度测试。分组再量化针对首维不等的组增加 warp tile 方案，测试要求该路径不输出 BF16 中间值，并对特殊值、scale 和布局做对照。**考虑复现**，在不均匀专家负载及非 128 对齐 gate 宽度校验跨后端字节，仓库已合入主线，未见同批的整模训练报告。

[NVIDIA/TransformerEngine 8e89d3ad8b](https://github.com/NVIDIA/TransformerEngine/commit/8e89d3ad8b2ce7adb9a8ad4bd5798197db16de8b) · [NVIDIA/TransformerEngine ef7f51595c](https://github.com/NVIDIA/TransformerEngine/commit/ef7f51595c541597a2e13c22a38eb44819bda782)

### vLLM-MUSA 确认 sparse indexer 与新版镜像契约

**UTC 11-17 08:54、11-17 05:39 · 主线 a99d2d8a6f、4955fe344c。** DeepSeek-V4.1 sparse indexer 调整候选过滤及 decode radix TopK，测试覆盖 TopK 512/2048、短行、不同索引 dtype，并和共享 kernel/参考选择集合比对。构建迁至 Python 3.12 与 torch 2.13.0.post1，合入说明记录旧 DeepEP wheel 缺 c10 符号需重建。**考虑复现**，核对 MUSA device 上的候选集合和端到端长上下文，并把 torch ABI 固定到镜像验证。本轮仍未取得 MTT S5000 的 Kimi K3.1 Pro 训练一手原文。

[MooreThreads/vllm-musa a99d2d8a6f](https://github.com/MooreThreads/vllm-musa/commit/a99d2d8a6fd4a160daeb95b4de8022162123e84d) · [MooreThreads/vllm-musa 4955fe344c](https://github.com/MooreThreads/vllm-musa/commit/4955fe344ce8e9c89283350dd613a53fc055c2f7)

<a id="topic-8"></a>
## 评估与可靠性

### Astra for Law 6.2 公布法律配置和多口径结果

11 月 17 日发布的版本是以 GPT‑6.2 Astra 为底座，组合任务专门指令、设置和法律检索工具的配置。Artificial Analysis 的 Harvey LAB-AA 共 120 项任务，要求所有条件满足且无重大幻觉才计全分。Astra for Law 的全通过率为 20.8%，通用 GPT‑6.2 Astra 为 15.0%，Muse Wonder Full 为 13.1%。Vals 11 月 16 日测试的另一套 208 道法律检索问题中，Astra for Law 全分 112 道，计部分分则为 90.9%。OpenAI 公告面向 ChatGPT Enterprise 开放受信任访问申请。**深入阅读。** 区分检索工具、模型配置与全通过规则，再看增益对应哪个环节。 [配置发布与评测定义](https://openai.com/index/astra-law-6-2/) · [Vals 原评测页](https://www.vals.ai/benchmarks/legal_research)

### Gemini 4 Hydrogen 补齐完整安全与安保报告

Google 模型卡现已链接 78 页报告，官方 PDF 对象更新于 11 月 17 日 15:41 UTC。报告提供模型单独运行与 Antigravity 运行时的评测结果。新版 Agent Behavior 测试中，Hydrogen 单模型的 under-service / overreach loss 为 33 / 60、Carbon 为 61 / 54；在 Antigravity 内为 12 / 17 与 13 / 24。三个模型都在本轮新版任务上重测，沿用旧报告的数字会混入测试集更新。报告也记录较多对无害任务的拒答；模型权重与完整适配训练配方仍未披露。**深入阅读。** 对比第 5.3 节的同批重测，区分模型行为与 Antigravity 的运行时影响。 [官方模型卡](https://deepmind.google/models/model-cards/gemini-4-hydrogen/) · [安全报告 PDF，第 5.3 节](https://storage.googleapis.com/deepmind-media/gemini/gemini_4_hydrogen_safety_report.pdf)

### RepoFactory 修订三个可执行环境归档

上一轮已收录 RepoFactory 的 200 个环境。11 月 17 日两次提交更新了实例 000165、000061、000080 的镜像归档和配套实例压缩文件，沿用 200 环境版本。对比或复跑时应保留实际数据提交；本轮确认了文件级替换，三个新环境未执行。**考虑复现。** 固定最新数据提交，先对三个更新环境做启动和验证器冒烟测试。 [首次修订](https://huggingface.co/datasets/IQuestLab/RepoFactory/commit/15fb80d60f53312ee7ca46db3df97d73a500f95d) · [最新修订](https://huggingface.co/datasets/IQuestLab/RepoFactory/commit/efdcaa8f6ea1bfa55eed2263e39186b928c7da88)

### 跨 Agent 接力实验定位逐轮摘要遗失的观察证据

**Yibo Liu，arXiv 2611.18642 v1，10-14 17:21:25 UTC 投稿。** 56 个可执行、规则隐藏的交互世界用同一顺序比较完整历史和 300 字符手递便笺。对 GPT-6.1 Sol，后四条 life 平均目标完成率 74%→57%，五分制下降 0.64，95% 重采样区间为 0.31–0.96。相同长度下，每次从完整记录重建便笺，在 56 世界追回 0.39 分。最高损失 10 世界的 70%/82% 恢复比例来自两种较强模型的子集。DeepSeek-V4.1-Flash 只评 31 个世界。复现适合用冻结工具轨迹配对只传摘要、摘要加原始证据回读和完整历史三组，按世界计不确定性。 [论文 §4–5、表 3/4 和图 4/5](https://arxiv.org/html/2611.18642v1#S5) · [版本记录](https://arxiv.org/abs/2611.18642v1)

<a id="coverage"></a>
## 实际覆盖、补录与续查

本轮收录 **275 个带稳定标识的版本事件**。正文标注回溯补录和更正。数值均来自所列来源，本轮未执行独立复训、设备压测或付费 API 实测。

- **厂商与模型工件。** 156 个厂商与评测入口中 121 个直连 HTTP 200。Hugging Face 181 个组织的 362 个目录页可读，新旧两次全站创建索引相接，主窗识别 3,660 个模型和 2,088 个数据集 ID。定向查询 142 个仓库，122 个固定卡可读。动态正文、被门控工件、长尾内容和既有仓修订仍为定向覆盖，具体失败留在台账。

- **论文与会议。** 核对 arXiv 十分类 11-17 公告、近期列表和 RSS，官方 OAI 列表全部关联版本。11-18 公告的最后五秒仍待复查。HF 11-17 已读 41 条，EMNLP 主会卷传输补齐，OpenReview API 403，SC26 会议 PDF 与精确发布时间仍有缺口。

- **开源工程**　主窗 `(2026-11-17T01:01:04Z, 2026-11-18T01:00:05Z]`，回溯至 `2026-11-14T01:01:04Z`。核查 85 个 GitHub 仓库地址（包含别名）。72 个默认分支历史完成分页，主窗 819 条提交；71 个合入仪表板与其他证据交叉核验后完整；70 个发布列表覆盖所查窗口，PyPI 可读 26/29。取得 428 个候选差分，含上一轮未完整覆盖的 11 月 12–13 日分段。与 41 份历史日报去重，登记主窗 174 个工程事件，回溯补录 10 个。
- **工程续查**　vLLM-Omni 的公开合入与默认分支列表已完成本轮分页，并补查了上一轮更早的窗口。PyTorch 合入筛选仍显示空结果，主分支记录和固定差分可核项目所选条目。slime 和 OpenRLHF 的所查入口本轮没有新主分支提交。保留 404 项和未逐条核对的候选、维护分支及版本改动历史。NeMo RL #5139 仍为开放；TorchTitan #6142、#6147 的 CI tag 未改变其合入状态。没有复跑设备基准。

- **硬件与软件协同。** 本轮记录 376 次直连请求，其中 330 次返回完整 HTTP 200 响应（含定向差分）；30 个 GitHub REST 窗内查询 403，通过 Git 历史恢复 21 个主仓的默认分支回溯。另查 NVIDIA、AMD、Intel、Cloud TPU、AWS Neuron、昇腾 CANN 与国产适配层，查阅 MLCommons 固定提交的四组日志。保留 CANN 正式包映射、libtpu 小版本行为、MUSA Kimi 训练原文、SC26 后续及其余 MLPerf 日志等缺口。

- 全局完整覆盖水位仍为空。下一轮以本次发现截止续扫并回溯 72 小时，各来源继承尚未完成的区间。本地 `.monitor` 保存访问回执、候选及已收录版本。
