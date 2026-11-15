# LLM 技术动态日报 · 2026-11-15

> 时区 Asia/Shanghai（UTC+8）。增量扫描窗口 **(2026-11-14 09:00:15, 2026-11-15 08:57:48]**，另回溯 72 小时至 **2026-11-11 09:00:15**。实际生成 **2026-11-15T09:37:07+08:00**。正文中的 UTC 时刻按源记录给出；只有发布日期的材料保留原日期。

> 已补传 11 月 13 日与 14 日日报，逐字回读归档基线 `374161c33876062c7ca0fe6de9ce231406bf8dd4` 上的 39 篇日报。本轮未独立复训、压测设备或实测付费 API。

## 本次重要变化

- **训练与训推同步。** Lara Mini Base 披露 6T 多语预训练和额外 100B token 的 128K 扩窗。SS-HSDP 的短程实验显示，分离前后向和优化器布局可提高 MFU，但 ZeroSM 在小负载下反而慢。TorchTitan 将 RDMA 同步接入 RL 循环，单次权重拉取与端到端 tokens/s 的后端排序不同。
- **推理约束更具体。** vLLM 将稀疏 NCCL 权重更新分块，H20 实验降低峰值显存压力。Edge0 合入 Apple Silicon Q3_K 路径，显存限额、主存上限和完整驻留有不同测量条件。Hydrogen INT8 首次给出 BF16 共享专家与 INT8 路由专家混用的启动边界。
- **评估更新可直接复核。** Decider `locate` 用独立 LoRA 返回原文跨度。RedactionBench 发布 200 行字符级遮盖评估。Artificial Analysis 更新 Coding Agent Index 的判分口径，跨版本趋势需要按同一子榜复查。
- **硬件路径出现新的容量与显存数据。** 昇腾 HiSparse 作者实验中可容纳更多并发，单 token 时延略升。cuDNN KDA 的同平台测试降低了 1M token 打包形状的峰值工作区。两项都适合用实际长短请求分布复验。

[模型与技术报告](#models) · [后训练](#posttraining) · [训练系统](#training) · [推理与部署](#inference) · [开源工程](#engineering) · [硬件与软件协同](#hardware) · [评估与可靠性](#evaluation) · [覆盖与续查](#coverage)

<a id="models"></a>
## 模型与技术报告

### 补录｜Lara Mini Base 把多语数据和 128K 扩窗配方写进同一份报告

**Translated · arXiv 2611.16956 v1，11 月 12 日 15:28 UTC 提交。** 面向多语应用的小激活预算，7.2B 总参数、0.6B 激活，34 层中 7 层使用全局注意力，其余交替使用 512-token 滑窗与 Gated DeltaNet。每 token 选 128 个路由专家中的 4 个，并经过 1 个共享专家。来自公开数据集的去重过滤池为 13.8T token，前三阶段读 6T，英语占 41%，其中 97% 的 token 属于文档首遍读取。后续用 40B 的 32K 序列及 60B 的 128K 序列扩窗。矩阵用 Muon，其余用 AdamW，采用 DP 和 4 路 EP，内部训练代码基于 Megatron-Bridge 0.5.0 与 Megatron-Core 0.18.0。作者披露各阶段采样与小尺度消融，完整预处理实现、训练硬件总预算和可直接重放的混合语料仍不齐备。

表 8 的同一模型扩窗前后，128K RULER 从 10.9 到 63.7，26 种语言的 ONERULER 从 0 到 80.9，同协议 Gemma-4-E2B 为 84.7。附录表 23 在支持的 44 种非英语里有 41 种领先同组基座，各语言任务覆盖不同，简中低于 Qwen3-1.7B 2.7 分。推理说明采用单 A100、vLLM 0.28.0。基础权重为人工审批的 CC BY-NC-SA 4.0；表 8 对应基座检查点。**建议深入阅读** §2.3、§3 和附录 C，先复验扩窗的 32K/128K 召回及每序列状态成本。 [固定报告](https://arxiv.org/html/2611.16956v1) · [版本时间](https://arxiv.org/abs/2611.16956v1) · [权重申请与许可](https://huggingface.co/translated/Lara-4-Mini-Base)

### 补录｜Perplexity Decider v1.3 分开部署判别与原文抽取头

**Perplexity · 11 月 12 日发公告，两个 v1.3 权重仓的固定提交均为 11 月 10 日。** 标准版基于 Qwen3.8-27B，Flash 基于 Qwen3.5-4B。标准版做通用文本训练、专门文本 LoRA 和视觉 LoRA 并加权合并，Flash 以标准版的输出分布做 KL 蒸馏。新增 `locate` 冻结原骨干及判别头，学习专用 token、跨度头及 LoRA，为每个字段打分并预测阈值，返回原文字符范围。作者合并该 LoRA 时观察到判别回归，部署按请求加载；混合判别和抽取会增加前向计算，卡片还说明单实例调用由锁串行化。

项目方自测中，抽取套件混合 exact-span F1、字符级 F1 和字段准确率，按任务主指标取均分，标准版 69.0、Flash 63.6、Liquid d1 43.3。生产样本的 93.98% 来自与 LLM 裁判的一致率，旧版为 93.68%，样本没有人工真值。两套权重 Apache-2.0。**建议考虑复现**，用自有标注分开看判别校准、精确跨度、无匹配样本和混合请求尾延迟。 [厂商训练与评估说明](https://www.perplexity.ai/en-GB/hub/blog/perplexity-decider) · [标准版固定卡](https://huggingface.co/perplexity-ai/pplx-decider-v1.3/blob/9c2ebb27470d3dec2ad7ab727727e40d142c7213/README.md) · [Flash 固定卡](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.3/blob/d06fb3f2d97e0a52325a0d463c450cf0338f44fe/README.md)

### SenseNova-Loop-9B-Preview 替换为 BF16 骨干工件

**商汤 SenseNova · 11 月 14 日 04:40 UTC，HF `89d79709`。** 该提交替换八个 `safetensors` 分片并更新 index，标注为骨干 BF16 转换，索引记录 19,318,793,184 字节。原模型用循环骨干作自投机草稿来源。新版本没有随包提供转换前后的质量、显存或接受率对照。**建议继续观察**，已部署者先固定 checkpoint，确认精度兼容并对同一批提示比较接受长度。 [固定差分](https://huggingface.co/sensenova/SenseNova-Loop-9B-Preview/commit/89d7970926e215f8da61726700cd026ccffbb6cb) · [固定配置](https://huggingface.co/sensenova/SenseNova-Loop-9B-Preview/blob/89d7970926e215f8da61726700cd026ccffbb6cb/config.json)

<a id="posttraining"></a>
## 后训练

### 补录｜DeepMind 证明两种把解码写进 SFT 目标的损失无下界

**Google DeepMind，arXiv 2611.16736 v1，11-12 12:51:08 UTC 提交，见 11-13 公告批。** 训练时用真实前缀，推理时用模型自己的前缀，作者检验了两种直接把解码放进目标的做法。论文 §3–4 证明贪心版本不连续且无下界，采样版本虽平滑也无下界。采样版本等于对加独立 Gumbel 扰动的贪心损失取期望。对采样路径做 stop-gradient 得到矩匹配更新，负方向在函数空间能降低自回归 NLL，但 §5.5 给出这一结论无法直接推到模型参数空间的理由。采样前缀跨时刻相关，Monte Carlo 估计的方差随序列增长。作者以固定输出长度建立记号，论文提供理论分析，未报告大模型训练对照。**建议**设计解码感知的 SFT 目标时先检验损失下界，并用 token 交叉熵控制更新方差与成本。 [固定论文 §3–5](https://arxiv.org/html/2611.16736v1#S3) · [版本记录](https://arxiv.org/abs/2611.16736v1)

<a id="training"></a>
## 训练系统

### 补录｜SS-HSDP 让 GEMM 参数布局和 Muon 更新单元独立组织

**清华大学、生数科技，arXiv 2611.16467 v1，11-12 08:22:25 UTC 提交，见 11-13 公告批。** 例如前后向使用拼接的 QKV，Muon 可按头更新 FP32 矩阵。SS-HSDP 通过共享缓冲区视图、分层分片与批量格式变换连接运行和优化两套布局，低精度分片在 microbatch 间复用。作者在固定单卡合成输入、AdamW、MXFP8 设置下对照 FSDP2、自行复现的 veScale-FSDP 和 Megatron-FSDP。128 GPU 上，12 层 73.5B MiMo 的 MFU 比最快基线高 21.7%（相对值），125.0 GiB/卡，比最低显存基线少 6.2%。另在同机 64×B200 的 BF16 MiMo 单次负载扫描里，ZeroSM 通信相对基于 SM 的 SS-HSDP 吞吐从最小负载低 33.5% 转到最大负载高 8.8%。性能实验取短程步时，作者表示后续开放实现。**建议**在相同模型分片下分开测零拷贝与 ZeroSM，覆盖可通信受限的短 microbatch。 [固定论文 §3–6、图 6/8、附录 B](https://arxiv.org/html/2611.16467v1#S5) · [版本记录](https://arxiv.org/abs/2611.16467v1)

<a id="inference"></a>
## 推理与部署

### Edge0 补录 145B/120B 权重，主窗合入 Apple Silicon 路径

**Edge0 / AutoArk · 11 月 11 日上传两套尺寸的 Q3_K/NVFP4 权重，11 月 12 日公告；11 月 14 日 07:33 UTC 合入 #214，10:05 UTC 更新卡片。** 从 Qwen3.8-Flash-Next 派生，145B/120B 每 token 分别激活 10.12B/8.64B，专家输入从 2,560 维投影到 1,792/1,280。`edge0-infer` 将服务转给 `edge0-strata`，靠专家缓存及 CPU/GPU 数据搬运降低设备常驻需求。合入的 Metal 路径仅接 Q3_K，模板 32K，首次启动会在可写权重目录生成 `mtp-q2_0.gguf`。项目文档在 M2 Ultra、145B Q3_K 报告 60.9 tok/s 聊天解码和 57.0 tok/s agent 解码；同表 RTX 5090 D 24 GB 的 36.8/44.9 tok/s 是把同卡限制到约 16 GiB 可用显存并给 20 GiB 主存的 Q3_K 实验。HF 差分说明通过引擎显存预留与 cgroup 主存上限分别实现。不限容且启用驻留下的 145B Q3_K 记录为 23.14 GiB VRAM 加 31.28 GiB RAM，权重约 40.81 GiB。公告的 29 GB 是另一条内部峰值口径，对手数值为估算，完整条件将随报告披露。

**建议考虑复现**，用固定模型包、同一输入长度和 MTP 设置分别测冷加载、主存/显存驻留及 agent 解码。公开权重所含 Edge0 贡献按 Community Model License 的发行范围声明，上游 Qwen 许可另行适用，特定商业模型服务需要单独授权。 [首发公告](https://edge0.ai/blog/introducing-edge0-145b-frontier-ai-capabilities-on-your-own-computer) · [Metal 合入 #214](https://github.com/Edge0-AI/Edge0/pull/214) · [固定性能与架构说明](https://github.com/Edge0-AI/Edge0/blob/f833336e487f61848ab5d49b2e7d06678812ea8a/docs/models/edge0-145b.md) · [显存测量更正](https://huggingface.co/Edge0/Edge0-145B-q3k/commit/1b1dbf70768126d5c957335634d39af643f6d079) · [固定权重与许可](https://huggingface.co/Edge0/Edge0-145B-q3k/tree/2d8011637b46d532a65ed4f9d78eb5b466e486d5)

### Hydrogen-1.0 上架 W8A16，暴露共享专家融合的混合精度边界

**Deep Principle · 11 月 14 日 01:35 UTC，HF `f0df36c8`。** 在此前 BF16/NVFP4/MXFP4 工件之外，发布 MIT 权重的 仅量化权重的 INT8 包。用无标定 `model_free_ptq`，稠密 MLP 和路由专家按每 128 个权重共享对称 INT8 scale，共享专家、注意力、路由器、embedding 与 MTP 留 BF16。卡片同日把 checkpoint 体积更正为约 793 GB（739 GiB）；厂商在 8×B300、SGLang 0.5.20 测试启动，指定关闭共享专家融合以避免 BF16 与 INT8 专家被并进同一个张量。其他运行时与硬件、同任务质量和时延仍缺对照。**建议考虑复现**，若需更通用的权重格式，先在目标内核验证载入和 MoE 路径，再和 BF16 做同预算评测。 [固定模型卡](https://huggingface.co/deep-principle/Hydrogen-1.0-INT8/blob/f0df36c8a7952bd0b9d4a8048cf3b2a334cc6017/README.md) · [体积更正](https://huggingface.co/deep-principle/Hydrogen-1.0-INT8/commit/f0df36c8a7952bd0b9d4a8048cf3b2a334cc6017)

### KLM 1.6 为小型混合骨干公开固定槽位批解码

**MinimaLabs · 11 月 14 日 20:02 UTC，HF `8fabc98a`。** 前次记录 99M 基座的训练配方，本次新增 `fast_generate`。提示先走模型的 batched prefill，解码对固定槽位每次重放 CUDA Graph 推进 8 token，在 GPU 采样，输出结束即补槽。同输入投影合为 GEMM，Triton 执行每行注意力，Gated DeltaNet-2 recurrence 和归一化按实现用 FP32 运算；安装 Flash Linear Attention 后可切换长提示的分块 prefill。厂商未随补丁给出按卡、批量和长度的吞吐或误差表。**建议考虑复现**，用同权重分测 eager/graph 的首轮编译、长短混合批和采样一致性。 [固定差分](https://huggingface.co/MinimaLabs/KLM-1.6/commit/8fabc98abe194f7450cf19ed743caa7e85a4e5b0)

### 两组模型卡修正加载路径

- **Liquid AI d1-2B / d1-Omni-2B · 11 月 14 日 08:49 UTC。** 原 fallback 在 Transformers 5.23 的同名 LFM3 类可导入后会自动切换实现。固定提交改为直接注册权重仓中的文本、视觉及 Omni 类，processor 也显式按配置映射取仓内版本。作者指出内置与仓内卷积和图像处理实现存在差异，迁移需验证权重和预处理等价性。**建议考虑复验**，固定图像/音频输入和依赖版本，保存 logits 与 processor 输出再升级。 [d1 固定差分](https://huggingface.co/LiquidAI/d1-2B/commit/a6bce774e6dde338dcc3036ceb5d91f78bf4c19e) · [Omni 固定差分](https://huggingface.co/LiquidAI/d1-Omni-2B/commit/51918b53052eacfd24062f99f13556c3c56fcf63)
- **腾讯 DSpine · 11 月 14 日 13:34/13:47 UTC。** 27B 和 8B 的 b16 草稿工件分别把 `config.json:architectures` 从 `DFlashDraftModel` 改为 `DSpineDraftModel`，差分仅改声明。模型选择由服务端注册表决定。**建议继续观察**，部署时核对目标运行时是否注册该架构并跑接受率回归。 [27B 固定差分](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16/commit/260a6c23070ca873ff1a5f33b7119ac3f6a245ee) · [8B 固定差分](https://huggingface.co/tencent/Qwen3-8B-DSpine-b16/commit/3b1e69d0354f9281fd6d9e3ca0cbf39c8185db63)

### 补录｜DeepSeek 将托管工具和上下文拼装接到既有推理接口

**DeepSeek · 11 月 13 日官方变更日志，仅给日期。** `deepseek-flash` 在 Messages、Responses、Chat Completions 接口开放托管搜索、文件解析和 URL 读取；服务端把检索或解析结果组进会话上下文。官方文档区分关闭 thinking 的搜索能力和开启后的多轮搜索能力，OCR/ASR/TTS 也有独立入口。变更涉及接入契约，日志未附模型权重或同工具对照，也没有尾延迟、引用正确率或失败重试统计。**建议继续观察**，若采用托管路径，测同一文档的解析质量、调用次数与时延。 [官方变更日志](https://api-docs.deepseek.com/updates/) · [官方集成指南](https://api-docs.deepseek.com/guides/chatbot)

### 补录｜Cairn 跨机复用编译和探测产物，再用通用预热容器启动指定模型

**KAIST、SK hynix，arXiv 2611.16180 v1，11-12 02:11:21 UTC 提交，见 11-13 公告批。** 作者在本地 NVMe 的 Qwen2.5-7B 冷启动中分解到权重加载 6.9 秒、编译 24.4 秒、引擎初始化 17.2 秒、KV cache 探测 12.6 秒。Cairn 用共享池存权重、编译缓存及探测产物，同时在模型未知时预热通用进程，通过 zygote 快速补货。两台各有一张 RTX A6000 的 VM 用同一宿主的远端 NUMA 模拟 CXL，覆盖 1B–14B 六模型。与移植到 vLLM v0.20 的同套 ServerlessLLM 对照，首次冷启动从 57–85 秒降至 11–21 秒，均要求共享池已有另一台服务器发布的产物。403 请求、五模型、keep-alive 为零的单小时请求流中，TTFT 中位数降 64% 至 11.8 秒。生产规模 64 GPU 数值来自模拟，空闲预热容器和 zygote 合计需要 2.9 GB 主机内存和 264 MiB CUDA 上下文。源代码计划在发表时开放。**建议**先做冷启动阶段剖析，再独立验证缓存跨主机命中、预热池占用和真实 CXL 表现。 [固定论文 §3–6、图 6/9](https://arxiv.org/html/2611.16180v1#S6) · [版本记录](https://arxiv.org/abs/2611.16180v1)

### 补录｜LazyWeights 把低比特整数再做无损压缩，分开验证文件、解码器和整机收益

**UT Austin、Case Western Reserve University，arXiv 2611.17108 v1，11-12 17:03:40 UTC 提交，见 11-13 公告批。** 将 GPTQ 量化、现行 GGUF 兼容整数与分 lane Huffman 结合。固定实现 `65c88e18` 的 `lazyweights/lanecodecs.py::encode_layer` 生成规范码、lane 码流与索引，`hw/huff/rtl/huff_engine.sv::huff_engine` 用 32 条 lane 和共享端口仲裁解码。Phi-3-mini、Mistral-7B、SmolLM-1.7B 在 1%/3.5% 相对 FP16 困惑度预算内，比各自最佳测试 llama.cpp 格式小 5.7–31.4%，其中 Mistral-7B 紧预算为 5.668→3.886 bit/权重，GSM8K 0.425→0.402。ZCU102 原型在 112.5 MHz 验证 29 矩阵 bit-exact。Orin Nano Super 的最多 36% 延迟、38% 能耗下降来自实测替代路径加布局前合成估算。Jetson Nano 用 USB 流式跑 7B 的 6.6 倍是紧预算、定驻留内存下的软件实测，宽松预算时 CPU llama.cpp 贪心略快。**建议**先复跑 GGUF 的质量与字节数，再依具体权重带宽、KV 开销与硬件解码位置测试延迟。 [固定论文表 III–V、§VIII](https://arxiv.org/html/2611.17108v1#S6) · [版本记录](https://arxiv.org/abs/2611.17108v1) · [固定实现](https://github.com/arnabsanyal/LazyWeights/tree/65c88e18ff8ae1e99b5414c146bce85f510eb0b5)

### 补录｜华为诺亚 HDM 用同一模型的一次前向判断显式推理是否值其成本

**华为诺亚方舟实验室、利物浦大学、伦敦大学学院，arXiv 2611.16961 v1，11-12 15:32:47 UTC 提交，见 11-13 公告批。** 冻结底模的快路径直接读取选项。一个小头用该次前向已有的隐状态和预测分布估计慢路径可带来的正确率变化，由请求侧阈值决定是否续跑自回归推理。JevBench 231 个公开题上，Qwen3.8-27B 达到 always-thinking 准确率的 90% 所需思考 token 为 103/题，重实现 RouteLLM 为 112、置信阈值为 138、always-thinking 为 591。慢路固定 4096 token，27B 有 14.9% 截断。比较基于有明确选项或结构化输出的任务，外部报告的耗时来自不同后端，论文的代码链接仍是占位符。**建议**固定模型、harness、答题准确率目标与同后端做 token/时延前沿，另测开放输出时能否复用快速读取方式。 [固定论文 §2–3、表 6](https://arxiv.org/html/2611.16961v1#S3.T6) · [版本记录](https://arxiv.org/abs/2611.16961v1)

### 补录｜AWS CURATOR 在同一个候选池内训练可联合判定的 span 证据选择器

**AWS AI Labs，arXiv 2611.16366 v1，11-12 06:29:40 UTC 提交，见 11-13 公告批。** 对 `pplx-decider-flash-v1.2` 用可变候选数和类别均衡的联合监督做微调，推理时共享上下文，一次批量给 span 打分再按 token 预算挑证据。在 QAMPARI 的 855 个开发问题、1024-token 预算下，证据召回 0.5466、评分 0.925 秒/题。原联合决策模型是 0.4279 和 0.909 秒，逐段重复完整上下文得 0.4844 和 20.065 秒，BGE span 重排为 0.3757 和 0.301 秒。开发集与 checkpoint 选择数据有重合。未用于证据训练的 FEVEROUS 上，含三个真证据的受控池扩大到 128 项时，作者报告 AUROC 比原联合决策高 0.189，各池是独立采样。论文使用固定检索池，尚未验证多步 Agent。**建议**先在独立问题和真实检索分布重复召回、完整证据覆盖、端到端回答质量与延迟比较。 [固定论文 §3–4、表 2 和图 3](https://arxiv.org/html/2611.16366v1#S4.T2) · [版本记录](https://arxiv.org/abs/2611.16366v1)

<a id="engineering"></a>
## 开源工程

### TorchTitan｜RDMA 权重同步接入 RL 循环

- **11 月 14 日 10:07（UTC+8），合入 main `631d544b1f`。** 昨日追踪的 RDMA 原语现在可由 `Controller.Config.model_state_dict_transport` 按任务启用，前置条件为 `cpu_prefetch`，默认继续用 TorchStore。Monarch 检查各生成器布局一致后注册训练端缓冲区，再为生成端规划读取。训练 actor 在第一次 `await` 前从 push stream 发起 staging，以保证其 collective 排在下一轮训练 collective 前。控制器等 pull N 完成才允许 push N+1 覆盖缓冲区。单机 8×B200 的 Qwen3-0.6B 确定性对照 10 步 loss 逐步相同。作者在两台 GB300 主机的 27B DAPO、4 个 TP1 生成器、6 步实验中，最慢生成器的稳态 pull 从 TorchStore/NIXL 的 15.6–16.8 s 降到 Mooncake 的 2.2–2.8 s 或 NIXL 的 0.96–1.08 s，整步吞吐均值分别是 13.0k、16.9k 和 15.7k tok/s，受 rollout 长度波动影响。Mooncake 结果用到了 PyTorch 的轮询和 NUMA 注册修复，PR 记录的默认环境固定版本仍可能读超时，CI 目前覆盖 CPU 单测。 [运行时接线 #5997](https://github.com/pytorch/torchtitan/pull/5997)

### Megatron-LM｜FSDP 的梯度完成点与低精度续训

- **11 月 15 日 02:33，合入 main `1c3539e6e9`。** 实验性 MFSDP v2 用 `GradientFinalizationMode` 表达外层规约何时提交和等待。中间 microbatch 的 `no_sync()` 选择 `SKIP_OUTER`，调度主体用 `REDUCE_OUTER` 发起规约，全部 backward（含延迟 wgrad）结束后显式等待。直接使用 context 的默认是 `REDUCE_OUTER_AND_WAIT`，三模式都保留内层规约或本地累积。嵌套作用域在异常退出时也恢复此前模式。作者两卡跑了 context 和累积套件，每 rank 74 项通过，3 项跳过，完整功能训练没有重跑。 [模式与单测 #9122](https://github.com/NVIDIA/Megatron-LM/pull/9122)
- **11 月 14 日 18:27，合入 main `05c4a8ba04`。** MFSDP v1 的 `start_param_sync(force_sync=True)` 被 FP8/FP4 参数 gather 的 checkpoint 恢复路径调用。这里现在只同步权重已复制的 bucket，分片 bucket 交给 forward pre-hook 按 FSDP unit 拉取，并在需要时覆盖 DP-Outer。旧路径可耗尽两槽 buffer，也可让 HFSDP 读到旧权重切片。作者用 4 rank CPU/gloo、CUDA 事件和流替身做回归。原路径无双缓冲时第二步相对 loss 偏差 0.1–0.4%，有双缓冲时触发 allocator 断言，修复后两种对照均通过。真实 GPU FP8/FP4 恢复仍待实测。 [强制同步修复 #9368](https://github.com/NVIDIA/Megatron-LM/pull/9368)
- **11 月 14 日 10:31，合入 main `4b11b8168e`。** TE CUDA Graph 捕获优先保留 layer 提供的 `rotary_pos_emb`，缺省时才用 chunk 生成静态 RoPE。Bridge Qwen3-VL 一类 layer 可以保住每次 replay 填充的 batch 形 mRoPE buffer，也避开 HybridModel 路径对缺少每 batch position IDs 的 mRoPE 调用。差分仅变更图输入构造，未附新增回归。 [图静态输入 #9333](https://github.com/NVIDIA/Megatron-LM/pull/9333)
- **11 月 15 日 00:23，合入 main `154e7bd2f5` 的工程指南。** Combined 1F1B 指南以五个调度节点和消费 stream 描述 EP All-to-All 与前反向计算怎样交叠，含 PP=1、交错 PP/VPP、延迟 wgrad 和两版 Megatron-FSDP。整层重计算限 eager，不能与 CUDA Graph、细粒度 activation offload 或 paged stash 组合。作者核了 31 个选项与 Sphinx 构建，性能数字引用技术报告。 [MoE A2A 指南 #9344](https://github.com/NVIDIA/Megatron-LM/pull/9344)

### DeepSpeed｜长序列 loss 和专家通信

- **11 月 15 日 01:00，合入 master `892ed83e41`。** FPDT 的 `FPDT_LogitsLoss.forward` 改用 `log_softmax` 直接向 NLL 提供对数概率。目标 logit −1000、另一个 +1000 的 FP32 例子，先求概率会下溢为 0，产生无穷 loss，乘零权重还会得到 NaN。新增用例以单 rank 通信替身测试普通与零权重，逐项比对 reference cross entropy、激活和权重梯度，目标 loss 为有限的 2000。 [FPDT loss 稳定性 #9068](https://github.com/deepspeedai/DeepSpeed/pull/9068)
- **11 月 15 日 07:32，合入 master `8702b42e9d`。** AutoEP 的 `skip_recompute_combine` 在符合条件的 Hugging Face decoder layer 重入式 checkpoint 重算中跳过 DeepEP combine 和专家后行权重计算，保留 dispatch、专家计算和反向必须的通信。仅支持白名单 preset、DeepEP 多 rank EP、BF16 dispatch，要求活跃的训练态重入 checkpoint，并拒绝 `moe_layer_pattern` 覆盖。重算用 NaN 占位，要求 MoE 输出的数值在该段重算中只经残差相加且被 checkpoint 丢弃。作者最近在 CPU 跑 145 通过、1 个 CUDA 用例跳过，并列出较早版本的 4×H100 parity。多步回归覆盖路由变化和输入、router、expert 梯度。 [AutoEP 重算 #8745](https://github.com/deepspeedai/DeepSpeed/pull/8745)
- **11 月 15 日 08:11，合入 master `7caa7a2267`。** Fused RMSNorm 对不超过 128 宽的 Q/K norm 启用多行 tile，一次反向读取同时产出 dx 和 gamma 梯度分块，宽层继续原 kernel。作者单 H100、Qwen3-30B-A3B attention layer、序列 8192 的历史对照，Q/K norm 0.73→0.33 ms，整层 7.80→7.52 ms。16×H100 训练的 −3.5% 还包含 fused RoPE，最终 rebase 后未重测。新增窄形状、宽度 1 的 FP64 参考和重复运行测试。FP32 规约顺序可能改变，eager 比较使用容差。 [窄行 RMSNorm #8664](https://github.com/deepspeedai/DeepSpeed/pull/8664)

### NeMo RL｜Token capture 与 GRPO 动态采样

- **11 月 14 日 20:25，合入 main `f79dd15461`。** SingleController 允许 GRPO 动态采样与 token capture 同时运行。finalizer 核对组内样本后，在 `commit_finalized` 前先检查有效样本比例，再用 shaping 和 penalty 前的原始环境 reward 判零方差，拒绝组与替补预留在同一提交边界完成。PPO 仍不支持此组合。作者单节点 H100、Qwen2.5-1.5B 的 3 组 100 步 smoke 均无报错，capture+采样与常规+采样的平均 reward 0.663 对 0.659。该对照没有实际触发跨步借组，也没有检验收敛。更大的 nightly 配方入库时尚未运行。 [动态采样与采集路径 #5202](https://github.com/NVIDIA-NeMo/RL/pull/5202)
- **11 月 14 日 17:54，合入 main `effdd9b112`。** `http_api_server_count` 可让多个异步 HTTP 前端共享一个 vLLM engine 与监听 socket，并向各前端同步 token capture 配置、rollout 权重版本，汇总多进程 Prometheus。默认为单前端。作者的 Qwen3-0.6B 单机 8×H100、TP2 的 Gym nightly 用于验证功能，在该规模未见提速，逐步生成量指标验证另带 #5155 修复。 [共享引擎 #5074](https://github.com/NVIDIA-NeMo/RL/pull/5074)

### vLLM｜候选版增量

- **11 月 14 日 18:15，候选标签 `v0.34.0rc4`（`62d4bc3c7e`）。** 与昨日报导的 rc3 直接比较只改了两个测试文件。依赖 LMCache 的 connector 测试在包缺失时跳过（例如 XPU 环境），mixed warmup 用例给 mock runner 补了 `num_speculative_steps=0`。这是候选标签，PyPI 文件目录在主窗无对应上传。 [rc4 标签](https://github.com/vllm-project/vllm/releases/tag/v0.34.0rc4) · [rc3→rc4 差分](https://github.com/vllm-project/vllm/compare/v0.34.0rc3...v0.34.0rc4)

### vLLM｜主分支的批重叠、稀疏更新和 KV 迁移

- **11 月 15 日 03:22，合入 main `5e51c0ca10`。** Model Runner V2 的 DBO 支持纯 TP（TP>1、DP=1、无 EP）。达到 token 阈值后拆两个微批，两个微批各用独立 PyNccl communicator，交替将 all-reduce 放到通信 stream，另一微批在计算 stream 继续。拆分步骤走 eager，未拆可继续用 CUDA Graph。作者在 MI355X 跑 85 项相关测试。在另一个含该实现的 M3 服务分支上，TP4、MXFP4 专家、单次 3600 s 轨迹、并发 64 时 TTFT P90 5.23→3.21 s、总吞吐 53,692→59,281 tok/s/GPU，并发 32 下总吞吐 44,573→44,009，收益依负载而变。 [纯 TP DBO #63483](https://github.com/vllm-project/vllm/pull/63483)
- **11 月 14 日 20:46，合入 main `01548eecde`。** `sparse_nccl` 从累积整次稀疏更新改为逐批接收、校验和写入。两端按统一元数据切成至多 512 MiB 的批，超过该大小的单 patch 单独发送，索引和值按类型拼接成连续数据。批失败会继续接收剩余广播以保持进程组可用，之前已写入的批保留。作者用 Qwen3-30B-A3B、4×H20 TP4 加 1 张 H20 trainer，无客户端负载时旧路径在 1.25% 元素变化 OOM，新路径测到 50%。3% 更新的单引擎交替计时为 4.29→2.14 s，基准用的分批调用方式与最终单调用实现有区别。另有双卡真实 NCCL 验证混合类型、非连续输入及失败后的下一次更新。 [稀疏 NCCL 权重更新 #66375](https://github.com/vllm-project/vllm/pull/66375)
- **11 月 14 日 10:17，合入 main `ea0e80845d`。** NIXL pull 支持 PP prefill → PP1 decode，覆盖布局兼容的 dense 与 attention-only hybrid KV。decoder 按远端 stage 的实际 layer 集合和 descriptor 发读，各 stage 的层要求恰好覆盖一次。无块要读的 stage 仍通知完成，prefill 端汇总所有 TP×PP worker 之后才能释放页。READ 失败后排空已提交句柄再释放目标块，PP pull 仍禁止 DCP、Mamba/SSM 和特定打包布局。作者用 8×H20 做了 874 项单测和 GSM8K 拓扑矩阵，后者保留不同 TP 的精度波动。 [NIXL 多 stage pull #65336](https://github.com/vllm-project/vllm/pull/65336)

### SGLang｜HiCache 的存储兼容与等待

- **11 月 14 日 17:44，合入 main `50106ca2d7`。** Direct external linker 支持 MHA MXFP8 KV，在对齐的整页对象里先放 FP8 K/V，再带上 K/V 的 UE8M0 scale。交错的 page size 128 和平铺布局都可按 payload slot 对齐读取 scale，FP4 scaled pool 仍拒绝。作者以 Inkling、1×B200、FA4、page 128、Mooncake RDMA 测三样本，第二轮前缀整段从 L3 reload 的 decode logprob 对 cold replay 平均 KL 为 0。刻意漏传 scale 的负对照 KL 0.15。另有页携带 scale 的单测。 [MXFP8 KV 外部存储 #50324](https://github.com/sgl-project/sglang/pull/50324)
- **11 月 14 日 13:22，合入 main `a05c729c4e`。** HiCache 在共享模型存储命名空间加入已解析的逻辑 KV dtype，E4M3 和 E5M2 即使用同一 `uint8` 物理类型也会落到不同 key。Mooncake、File、NIXL 等沿用该隔离。已有不带 dtype 的对象会冷启动重建，未知 dtype 或字面量 `auto` 报错。新增测试包含异 dtype miss、同 dtype 回读和 layout tag 保留。 [逻辑 dtype 命名空间 #38362](https://github.com/sgl-project/sglang/pull/38362)
- **11 月 14 日 18:21，合入 main `19af0c78a2`。** HiCache PP 同步从逐轮等 Gloo send 完成，改为保留 `pp_size + pp_async_batch_depth` 轮，窗口满再等最老一轮。接收 FIFO 和在途上限均保留。作者在 2×B200、Qwen3-1.7B、PP2×TP1、write-through 的 170 s 负载里，depth=0 的 PP0 等待占比从 12.1–12.6% 降到 0.2%，32 个固定 prompt 输出相同，吞吐差异小于重复启动的波动，未复现提报者的 PP8 大规模吞吐场景。 [HiCache PP 窗口 #46363](https://github.com/sgl-project/sglang/pull/46363)
- **11 月 14 日 09:50–09:51，连续合入 main `25363debb8` 与 `e5caad9a07`。** Shortest-prefill-first 在 `wait_complete` 下先批量查询 L3 完成状态，再把 buffer 中已 staged 的 prefix 算作命中，PP 的多 ticket readiness 归并为一次 all-reduce。可选 aging 让排队请求的优先级随全局处理的 prefill token 数提高，因子默认 0，decode 不再重计之前的 prefill。两份差分覆盖已缓存长 prompt 提前、等待长请求抢占新短请求，以及 PP 集体通信次数，未给设备吞吐。 [缓存感知排序 #49649](https://github.com/sgl-project/sglang/pull/49649) · [等待老化 #49658](https://github.com/sgl-project/sglang/pull/49658)
- **11 月 14 日 22:47，合入 main `0f0d687b28`。** Diffusion Ulysses>2 的多 rank IPC all-to-all 改为每个 peer 独立 CUDA IPC handle，换 plan 或清 staging LRU 时按同序同步、barrier 并收集已释放块。作者在 8×B300、Ulysses 8、一次一个可变几何视频请求的旧路径看到约 2 GB/请求的 live 增长，独立 8 进程替身验证单 handle 泄漏和每 peer handle 回收，两个设备 workload 的 mp4 MD5 分别 40/40 与 20/20 相同。 [扩散 IPC 生命周期 #50321](https://github.com/sgl-project/sglang/pull/50321)

### 兼容层、精度配置与补录

- **11 月 14 日 12:00，PyTorch 主分支提交 `f407abb11f`。** Dynamo 为可观察顺序的字典操作补上 key order guard，字典长度 guard 以实际类型和内建存储长度构建，绕开子类的 Python `__len__` 实现。kwargs 顺序可被观察时保护顺序，固定签名看不到顺序时可复用图。差分覆盖空字典、真实类型、内部长度与 kwargs，PR 的 612 项 Python 3.15 CPU 结果属于整栈。GitHub 的 stacked PR 页标 CLOSED，主分支提交保留了对应代码。 [主分支提交](https://github.com/pytorch/pytorch/commit/f407abb11f88c3b04bffba64a8f8d61e7c0207e2) · [整栈测试说明 #200601](https://github.com/pytorch/pytorch/pull/200601)
- **11 月 14 日 12:25，合入 LMCache 默认分支 `dev` 的 `bcafb03140`。** MP connector 同时识别新 vLLM `KVCacheGroup` schema 和旧结构。新分支从 manager config 取窗口和 Mamba mode，依据 `prefix_cacheable` 区分 scratch，只有 `dcp_sharded` 才按 DCP size 放大每块 token 数，也处理某一 PP stage 没有该组 layer 的情况。作者在旧版 CPU nightly 跑 1175 通过、145 跳过，并以新 vLLM `153dc046` 做实际 group 对象冒烟，GPU 端到端待核。 [LMCache KVCacheGroup #6608](https://github.com/LMCache/LMCache/pull/6608)
- **11 月 14 日 13:53，Megatron-LM 主分支合入 `cd3d901b71` 和 `63cc3591bd`。** FP4 训练可分别设置 MoE dispatch 前向与 combine 反向的 `nvfp4` 线路类型，需 NCCLEP flex、NVFP4 配方、op fuser、grouped GEMM 及支持该 payload 的 Transformer Engine。用全局的 wire 配方时，配置校验会拒绝保留首尾 BF16 层。FP4 模型的 `fp8_output_proj` 则让 LM head 进入独立 MXFP8 autocast，MLA K/V RoPE 融合量化还要求实际 DPA 的 MXFP8 前向格式与 FP8 backward。两项差分有配方和 guard 回归，未提供本轮大规模精度或吞吐数据。 [NVFP4 EP wire #8580](https://github.com/NVIDIA/Megatron-LM/pull/8580) · [MXFP8 LM head/MLA #8582](https://github.com/NVIDIA/Megatron-LM/pull/8582)
- **补录，11 月 14 日 08:26，Megatron-LM 合入 main `4b19c1f5f7` 和 `b4aa8f3f9d`。** 前者让 MFSDP v1 构造分布式参数时保留 `sum_gradients_across_tp_domain`，供梯度 finalizer 将无 sequence parallel 的 TP GDN/KDA `out_norm` 梯度跨 rank 求和。后者在动态推理 context 按 TP 检查 token capacity，并将 eager prefill padding 限制在真实输入 buffer 行数。作者用 4 rank GPU 跑新增的 7 项 context 测试，未跑完整 TP8 模型。 [TP 梯度标记 #9049](https://github.com/NVIDIA/Megatron-LM/pull/9049) · [eager prefill 容量 #9394](https://github.com/NVIDIA/Megatron-LM/pull/9394)

### Triton、FlashInfer 与 FlashAttention｜内核选择和正确性

- **11 月 15 日 00:20（UTC+8），Triton 合入 main `1c53e566ef`。** AMD 后端降低 GPU 或系统作用域的 release/acq_rel 原子操作时，在多 wave 的前置 barrier 前等待各 wave 自己的全局写入。CDNA2–4 等待 VM 写入，gfx1250 等待 store counter，后续由发出原子操作的 wave 完成 L2 回写。CTA 作用域与单 wave 保持原路径。新增 FileCheck 覆盖 CDNA/RDNA 的等待和 barrier 次序。做 split-KV 等先发布分片结果再更新计数器的内核时，检查编译器版本和对应 ISA。 [AMD 原子 release 顺序 #12657](https://github.com/triton-lang/triton/pull/12657)
- **11 月 14 日 09:35，FlashInfer 合入 main `5ced14a7a1`。** MegaMoE 的可选 batch-invariant 路径扩展到选定的 NVFP4、MXFP4、MXFP4×MXFP8、MXFP8、W4A16 和 W8A16 配置。外层 AutoConfig 只在同一算术指纹中选计划，缓存计划与当前策略和指纹校验。精确一致性的范围固定在架构和数值配方；切换架构或精度方案需重新对照。作者在 SM100/SM103/SM107 验证了覆盖矩阵，含 EP2/EP4，提交前简化测试和解决 CI 冲突后未重跑 GPU。 [低精度 MegaMoE #9454](https://github.com/flashinfer-ai/flashinfer/pull/9454)
- **11 月 14 日 10:15，FlashInfer 合入 main `9c50996794`。** H100 的 BF16/FP16、head dimension 256、HND page16 的非因果 paged prefill 可由 auto 选 cuDNN，用实际请求 Q/KV 长度计算工作量，保留 `2**24` 门槛、原生图重放和短 query 的原路径，Blackwell auto 仍限 causal。作者在 H100 80GB、cuDNN 9.24.0.43 的公开微基准，BF16、B/Q/KV=1/128/8192 时热图 replay 271.38→206.94 μs，host enqueue 21.21→36.79 μs。生命周期测试在注意力代码相同的较早提交上运行，未覆盖最终合并树的其他 GEMM 变化。 [H100 paged prefill #9473](https://github.com/flashinfer-ai/flashinfer/pull/9473)
- **11 月 15 日 03:24 与 03:50，FlashAttention 合入 main `fc6bed9785` 和 `aae6770da7`。** FA3 的两个 C++ API 入口均加入 KV batch 索引的长度校验，paged KV 为空时走已有空输入返回，输出为零、LSE 为正无穷。作者分别用 H200 和 H100 的精简 BF16 forward 构建做了回归，空页表变更覆盖两个 PyTorch 构建入口及数值边界。 [KV batch 长度校验 #3153](https://github.com/Dao-AILab/flash-attention/pull/3153) · [FA3 空 KV 修复 #3161](https://github.com/Dao-AILab/flash-attention/pull/3161)

### EvolvingInfra 重建 amd64 镜像，统一推理路由实验基线

**MIT Han Lab EvolvingInfra · 11 月 14 日 16:36 UTC，合入 #18，镜像标签 `v0.1.0`。** 文档把 amd64 构建从 `67b4761` 更新为与 arm64 相同的 `486dcd1`，因此两个镜像可按相同的路由标志配置实验。本轮回读 Registry 索引 `sha256:58dee280…`，确认其中的 amd64 `sha256:36d47c0a…` 与既有 arm64 `sha256:ef0ecc7b…` 两份子镜像。源版本映射来自项目文档；校验只覆盖 Registry 清单，未执行容器或进行跨机器性能比较。**建议考虑复现**，按硬件固定镜像 digest 及路由参数，再测同负载的跨实例调度。 [合入 #18](https://github.com/mit-han-lab/EvolvingInfra/pull/18) · [固定构建与镜像说明](https://github.com/mit-han-lab/EvolvingInfra/blob/0999e3f9ab6ca0390049606b5e10ca7cc333cc5d/docker/README.md)

<a id="hardware"></a>
## 硬件与软件协同

### 昇腾 HiSparse 首次接入动态 KV offload，A5 复验应把容量与单 token 时延分开

**vLLM Ascend，11 月 14 日 22:51 UTC，主线 `4b3bd7ebe510`。** 实验路径复用 vLLM Core 的 HMA 分配、请求生命周期和协调器，以 AscendC SIMT 解析 MTP 多查询的驻留与回填，配合映射 Host 内存和 Triton gather/scatter，复用原 SFA 算术。当前资格范围为 Ascend 950、支持 SIMT 的 CANN 9.2、Model Runner V2，以及 BF16 或 FP8 E4M3FN C8 KV。作者在较早生产基线 `1f3b0d42`、Core `39dbf5183` 上，以 A5 的 GLM-5.2 W4A8C8、TP8/MTP3、32K 输入/8K 输出对照，打开后平均 TPOT 在并发 4/8 增加 2.19%/6.13%，输出 token ID 与 MTP 计数匹配。单轮容量检查无抢占并发由 9 到 25，另一次只有开启组完成约 104 万输入 token。**建议优先复验**合入版本的 miss 路径与容量曲线，记录 Host 写回和回填带来的 TPOT 代价。PR 描述的旧版本数据无法证明合入头的质量或最大容量。 [固定提交及测试说明](https://github.com/vllm-project/vllm-ascend/commit/4b3bd7ebe51094f3afa5288bb7a3316d12d3154e) · [固定使用条件](https://github.com/vllm-project/vllm-ascend/blob/4b3bd7ebe51094f3afa5288bb7a3316d12d3154e/docs/source/user_guide/feature_guide/hisparse.md)

- **昇腾稀疏索引和 310P，11 月 14 日，主线。** BF16/FP32 共用候选 radix，保持分数降序、同分 token ID 升序和原 FP32 payload。融合 K=2048 且要求合法本地 ID 小于 2²⁰，无运行时范围回退。作者在同卡固定基线的算子计时中，A3 FP32 提升 4.0–4.3×，A5 BF16 为 1.03–1.04×，证据限于算子。另一补丁使 310P Model Runner V2 接收结构化输出，将调度器级语法掩码按请求和 draft 行重排再应用到 logits，并由拒绝采样截断首个不合语法的 draft。**建议复验**分页 ID 边界及混合 grammar/MTP 请求。 [候选 radix 固定补丁](https://github.com/vllm-project/vllm-ascend/commit/76ea9d24f155afe032ad9ac6c98717f1f7334f36) · [310P 结构化输出固定补丁](https://github.com/vllm-project/vllm-ascend/commit/d0ec914c8ec7c321a981bd6b89f8cb66ca3bd587)

### ROCm 将 All-to-All 输入边并入 GEMM，确定性 MoE 另有明确范围

- **hipBLASLt / TensileLite，11 月 14 日，develop。** `FusedA2A: A2AGemm` 先算本地 K 分片，将后续 peer 的 B 分片通过 xGMI SDMA 搬进 `a2aBuf`，让第 i 轮计算重叠第 i+1 轮传输，并与已有 `GemmA2A` 生产端使用同布局。在 gfx950、BF16 B、strided batch、无 K 尾段的条件下，`A2ASubRounds` 支持 1/2/4，K 必须是 `world × DepthU × A2ASubRounds` 的整数倍。作者称在生产 shape、world 2/4/8，相比独立 torch A2A + matmul 为 1.4–1.9×。旧设备库的序列化谓词变更后无法加载，应随升级重建。**建议考虑复现**，先验证设备库并跑相同 shape、不同 peer 数与缓冲槽。 [固定补丁、接口和作者测试](https://github.com/ROCm/rocm-libraries/commit/5cb2a154b171da494090e342e1454d2771f8190b)
- **AITER，11 月 14 日，主线。** `fused_moe(deterministic=True)` 在 gfx950 显式固定 MXFP8 激活，配 gate/up 交错预排布 MXFP4 权重、BF16 输出及 SwiGLU/SiLU。禁用 split-K，第二阶段先写逐 route 局部值，再以固定顺序归约 top-k，避免 atomic add 的调度顺序改变结果。代码拒绝 EP、共享专家和预量化激活等组合。PR 中间版本曾使用 BF16 激活，最终实现已经换成 MXFP8。**建议考虑复现**同 token 跨 batch 的逐位结果和性能。 [固定实现与测试](https://github.com/ROCm/aiter/commit/c47295b54e36b36b2c201817586892deaa148317)
- **ROCm ATOM，11 月 14 日，主线。** MiniMax-M3.1 的 MXFP4 索引缓存可让 TP 各 rank 对自己的上下文块调用 FlyDSL 评分并归并候选，兼容页 16 与页 128 的 KV 几何。作者在一个聚合 TP4 服务上用随机定长请求、DSpark7 固定接受长度复验，启用后六组 tok/s/GPU 均降 0.4–2.9%，但 128K 输入三组 median ITL 降 0.8–4.7%。P/D 分离的正向吞吐来自实验分支。**建议以目标部署拓扑和上下文长度做开关 A/B**。 [固定补丁、配方和对照](https://github.com/ROCm/ATOM/commit/004f342ab22dedbef6d476713eeabfe9dbdadcbb)

### NVIDIA 修改 MoE 通信完成检测，并减少 KDA 工作区与 NVFP4 启动次数

- **TensorRT-LLM，11 月 14 日，主线。** NVLink one-sided MoE All-to-All 的小步默认尝试 sentinel 协议，在接收缓冲预填标志字，接收端轮询落地数据，符合条件时省去独立完成标志。限定 SM100/103/107、最多 64 EP ranks、格式和每 rank token 上限。默认 `auto` 在不满足条件时使用 fence。hidden 7168 / top-8 BF16 示例在 EP8 到 EP64 额外分配约 52–419 MiB/rank。旧强制 CFT 参数迁移见文档，未见可核的端到端对照。**建议对小步、阈值附近和大步同时测延迟和显存**。 [固定协议、限制和回归](https://github.com/NVIDIA/TensorRT-LLM/commit/8ba55ac4bd818eadbbf404514c6aa583cf2116f2)
- **cuDNN Frontend，11 月 14 日，主线。** KDA 将长度规则的分片元数据按总打包 token 数设上限，避免每个文档都按最大分片数预留。作者在 B300 CP8、五种 1M-token THD 形状中与固定旧版比较，最大一组峰值分配 142.284→72.812 GiB（−48.8%），最大单轮 KDA 回退 0.54%，纳入对照的输出/梯度字节相同。**建议优先按实际短长文档配比复验峰值**。 [固定实现与对照矩阵](https://github.com/NVIDIA/cudnn-frontend/commit/3fa2cfb38f6b278e1ab2de2eb26c40f2bfa3f560)
- **Transformer Engine，11 月 15 日 00:33 UTC，主线。** PyTorch grouped MLP 把两侧 per-group NVFP4 global scale 及 GEMM alpha（dGLU 路径取平方根）收进一次 Triton launch，不支持的输入保留 eager 路径。配套单测检查逐位相等，未披露端到端提升。**建议在多碎组训练下观察小 kernel 启动和回退占比**。 [固定提交与测试](https://github.com/NVIDIA/TransformerEngine/commit/4f5c08e48b3e190d27c731c1f9515f021dec255a)

### TorchTPU 将单槽 FP4 MoE 权重按列搬运，Google Batch 开放单主机 TPU

- **vLLM TorchTPU，11 月 14 日，主线。** Fused MoE v2 在专家预计填得进一个 tile、权重只有一个 VMEM slot 时，把 gate/up 和 down 按输出列拆成 DMA 块。当前专家读完一块后即可搬下一专家对应块，首读再等待，VMEM 允许时增加 scale 双槽。多 tile 会重复读，仍沿用旧路径。提交有调度等值测试，未给可迁移的端到端吞吐。**建议测单槽和多槽、短专家与长尾专家的实际重叠**。 [固定 Pallas 实现和测试](https://github.com/vllm-project/vllm-torchtpu/commit/fca720e1a935bfed1e12d66ca3e4ba361b34684f)
- **补录｜Google Cloud Batch，11 月 12 日，GA。** 批作业支持 TPU7x 4 芯、v6e 1/4/8 芯和 v5p 4 芯的独立 TPU VM。通过 `machineType` 请求并限制在单 zone；多任务可铺到多台独立 VM，无跨 VM ICI。适合批量推理、评测和单主机训练。 [发布日期与 GA](https://docs.cloud.google.com/batch/docs/release-notes#November_12_2026) · [已支持的规格和调度限制](https://docs.cloud.google.com/batch/docs/create-run-job-tpus#limitations)

### CANN 在 A2/A3 扩展 INT8 稀疏 KV，另有稀疏元数据算子和显式回退

- **CANN `ops-transformer`，11 月 14 日，master。** `MixedQuantSparseFlashMla` 的 CSA 在 A2/A3 增加 A16C8 INT8 KV。每 token 打包 64 个 BF16 RoPE、448 个 signed INT8 nope、7 个 BF16 scale 和 padding，共 608 B。Vector 解包/反量化当前块，Cube 执行 QK/PV，沿用 512-token 分块和轮换 KV 缓冲。作者给出 A2 定向测试，A3 真机、代表性 DSV4 形状精度和同条件性能基线继续待验。独立的 `UnionSparseGen` 在 A2 SIMD/A5 SIMT 合并同一请求的相邻 query 所选 KV 块，输出去重索引、有效个数和逐查询 `exactMask`，当前要求 `[1,128]` block、group 8 和有序输入。**建议把接口精度、边界与端到端测量分开**。 [INT8 CSA !15357](https://gitcode.com/cann/ops-transformer/merge_requests/15357) · [UnionSparseGen !14010](https://gitcode.com/cann/ops-nn/merge_requests/14010)
- **CANN MC2，11 月 14 日，master。** `alltoallvquantgmm` 的 MXFP4 在 AIC-only APACE 模板性能退化后暂回 legacy。另一变更将 A6 的非量化 FP16/BF16 `AlltoAllMatmul` 分流到 APACE，校验通信域 rank 数与 tiling。两个范围不同，未有可对比的通用吞吐结论。 [MXFP4 回退 !15656](https://gitcode.com/cann/ops-transformer/merge_requests/15656) · [A6 拆分 !14602](https://gitcode.com/cann/ops-transformer/merge_requests/14602)

### 补录｜AMD 为 Helios 增加 288 GB HBM4 的 MI450X，并披露 MI455X 离线吞吐

**AMD，11 月 12 日 14:00 UTC，厂商公布。** MI450X 提供每卡 288 GB HBM4（MI455X 432 GB），共用 Helios 机架架构与 ROCm，AMD 标称两者核心计算、带宽和互联一致。AMD Performance Labs 10 月的 MI455X DeepSeek-R1 MLPerf Offline 配置测到 21,333 tok/s/GPU，同源说明引用 NVIDIA Vera Rubin NVL72 Closed Preview。MLCommons 固定结果表报 72 卡合计 1,183,326.85 tok/s，折合约 16,435 tok/s/GPU，两数相差约 29.8%。AMD 侧未见可复验的正式提交、完整栈与同场对照，数据只支持观察 Offline 吞吐。**建议关注**正式提交及 Server/Interactive SLO。 [AMD 公告及脚注](https://newsroom.amd.com/news/amd-advances-instinct-mi400-series-ai-scientific-computing/) · [AMD MI450X 规格](https://www.amd.com/en/products/accelerators/instinct/mi400/mi450x.html) · [AMD MI455X 规格](https://www.amd.com/en/products/accelerators/instinct/mi400/mi455x.html) · [MLCommons 固定结果表](https://github.com/mlcommons/inference_results_v6.1/blob/4bb63cd28eb136be221bc801a9e38c66c8a7553f/summary.csv)

### 补录｜Loom 在 Strix Halo 公开模型专用程序与同机 llama.cpp 对照

**Loom-Fun，11 月 13 日，补齐固定模型卡与对照。** 工程把 GGUF 原权重与模型专用 `.loom` 程序分开，发布模型描述、原始文件锁、生成的 kernel 和性能记录。Qwen3.8-27B 的同一 Strix Halo / GGUF / prompt 走 `hrx-loom 0.1.0` 与 llama.cpp ROCm b1374、Vulkan b12211，32K prefill 分别 465.0 / 293.6 / 236.6 tok/s，32K 后单流 decode 21.0 / 19.2 / 18.0。单流两边均开 MTP，draft 深度各用自身配置，Loom 计时在客户端、llama.cpp 使用服务端计时。Qwen3.8-Flash-Next 的额外工件需要 `hrx-loom 0.2.0.dev0` 的认证 CI wheel，模型卡注明 11 月 22 日到期。**建议先考虑复现 27B**，统一时钟边界、采样参数和 draft 深度再推广排序。 [固定 27B 卡](https://huggingface.co/loom-fun/bartowski-Qwen3.8-27B-loom/blob/f3966d7cd81a8b9cf8e979e0ed5fc70ec42c98f3/README.md) · [固定 Flash-Next 卡与版本限制](https://huggingface.co/loom-fun/unsloth-Qwen3.8-Flash-Next-loom/blob/0afc7fb9339cbacdbfa6e56245ef90af64a58b5c/README.md) · [作者总览与实验口径](https://loom-fun.github.io/loom-skills/)

<a id="evaluation"></a>
## 评估与可靠性

### ZONOS3 的公开语音评估子集和浏览器按语言对齐参数

**Zyphra · 11 月 14 日 05:48 UTC 更新模型工件，20:41 UTC 提交评估数据卡。** 继上一期技术报告，本次 `web/laci` 新增服务端范围的 44 语言对齐头及索引，尚未看到相对旧浏览器头的逐语言对照。`ztts2_eval_mini` 则把已有小样本整理成可下载的 Hub 分集。clean 为 35 个语言分集、3,500 条，CC-BY-4.0；in-the-wild 为 20 个分集、1,919 条，CC-BY-NC-SA-4.0，未包含留存的目标录音。每个分集权重和为 1，按样本加权聚合裁到 100% 的 WER，四种语言用 CER。**建议考虑复现**，先复算分语言指标并保留权重、识别器和是否含野外分集的配置，完整听评仍需独立样本。 [浏览器工件差分](https://huggingface.co/Zyphra/ZONOS3-Mini/commit/d2d142a86b1e6aeb4252679673c86ecd86177bf8) · [固定评估数据卡](https://huggingface.co/datasets/Zyphra/ztts2_eval_mini/blob/f5e879de0ff2dcb870b9b09835145e8d2bfb2b0b/README.md)

### RedactionBench 把 200 份文档的字符级遮盖评估发布为数据集

**RedactionBench 作者组 · 论文 6 月 17 日提交，11 月 14 日 02:55 UTC 上传数据，03:08 UTC 恢复榜单与说明。** 基准按上下文判断哪些内容该遮盖，覆盖 11 类文档，包含 101 份公开来源并转录增强的文档与 99 份人工合成文档，发布 CC BY 4.0 的 200 行 test split。预测使用字符位置，R-Score 用敏感文本召回与保留效用考察少遮和过度遮盖。卡片记录 44 个评估条目（含两个同权重镜像），最高的 Claude Opus 4.6 为 0.714；可调置信阈值的模型阈值直接在全测试集上选择，作者说明一名主标注者负责全量标注。**建议考虑复现**，在独立开发集调阈值并按文档类别复核，使用时检查各原始文件的来源和许可。 [固定数据卡](https://huggingface.co/datasets/RedactionBench/RedactionBench/blob/77684d7ea74207e6086ae3a05f46ddc69ca753d2/README.md) · [论文 v1](https://arxiv.org/abs/2606.18782v1)

### Nemotron 3.5 Parse 回填 ParseBench-2 分项结果

**NVIDIA · 11 月 14 日 04:33 UTC，HF `ea2e1fe4`；结果标记 11 月 6 日。** 模型仓新增 LlamaIndex 提交的 `nemotron_parse_3_5_sdk` 评估记录，总分 73.44，文本 81.16、格式 82.04、表格 69.50、图表 60.12、版面 76.40、表单 71.44。这组新基准结果属于 SDK 管线；11 月 5/7 日日报的 2,078 页实验比较的是旧 ParseBench 上的 DFlash2 解码，不构成质量前后对照。**建议深入阅读**，用分项表挑文档类型，补足准确率、端到端时延和 pipeline 参数。 [固定评估记录](https://huggingface.co/nvidia/Nemotron-3.5-Parse/commit/ea2e1fe486bea3e7d0d505df9bafa4ad562407a0) · [LlamaIndex 固定结果路径](https://huggingface.co/datasets/llamaindex/ParseBench-2-results/tree/23bfeda85210ab620b6302562aaa83c4927bff9e/results/nemotron_parse_3_5_sdk)

### 补录｜MAPE 演化对话记忆程序，验证原始证据和按意图处理的作用

**明尼苏达大学、Stevens Institute of Technology，arXiv 2611.17068 v1，11-12 16:42:04 UTC 提交，见 11-13 公告批。** 通过固定构建、检索、组合、回答接口演化整套记忆程序，诊断代理用错误轨迹指导下一轮候选。在 LongMemEval-S 的 400 个评估问题、相同 Qwen3-30B-A3B 服务底模下，普通 RAG 达 66.0%（每千问 $2.62），准确率目标程序达 77.0%（$52.84），准确率与成本联合目标达 72.25%（$5.76）。对准确率目标程序，拿掉原对话片段或拿掉按意图处理分别降至 70.50% 和 70.75%。这些评估问题未进入演化，却用于最终选程序并报告分数，费用也仅含服务 token 费用。**建议**保存可检索原话，先用未参与选型的问题验证增益，再比较建库开销及摊销。 [固定论文 §2.3–2.5、§3–4](https://arxiv.org/html/2611.17068v1#S4) · [版本记录](https://arxiv.org/abs/2611.17068v1)

### 补录｜Artificial Analysis Coding Agent Index v2.1 更新任务版本和判分方法

**Artificial Analysis，v2.1，官方发布帖为 11-14 00:04:35 UTC，早于本轮主窗。** 322 道任务分三项等权，分别是 FrontierCode-AA 2.0 的 132 道、Terminal-Bench 4.1 的 66 道、SWE-Atlas-QnA-AA 的 124 道，每题跑 3 次。Terminal-Bench 4.1 复用 4.0 的题目、指令和运行环境，更新验证器后重裁保存的结果，无法复判或属验证抖动时保留 4.0 分数。SWE-Atlas-QnA-AA 重写 1,306 项判据中的 127 项，每项由 GPT-6 Luna xhigh 投 5 票。FrontierCode-AA 2.0 任务未公开，排行榜比较的是模型和 harness 的组合，主动运行时间未含环境启动与裁判。**建议**历史看板固定指数、子榜和 harness 版本，量化迁移时在相同子题口径重算。 [官方发布帖](https://x.com/ArtificialAnlys/status/2121393821955334328) · [评分方法](https://artificialanalysis.ai/methodology/coding-agents-benchmarking) · [SWE-Atlas-QnA-AA 数据卡](https://huggingface.co/datasets/ArtificialAnalysis/SWE-Atlas-QnA-AA)

### 补录｜Deepnote 将主动介入和保持沉默都纳入 Agent 评估

**Deepnote，11-13 14:58:55 UTC 的作者发布帖。** 用模拟公司同一固定时点的 Slack 和 11 种服务记录评估 12 个模型，共 50 案例（40 个应介入，10 个应沉默）。每次至多 180 秒模型响应时间，两名不知被测模型身份的 LLM 裁判按是否介入、证据、表达和受众给分。GPT-6 Astra 平均 8.97/10、$0.693/题，GPT-6.1 Sol 8.65/10、$0.112/题，均在 8/10 个应沉默场景保持沉默，两者的任务 bootstrap 区间重叠。模拟是固定快照、统一 high effort，未评估真实同事对介入的后续响应。**建议**为主动 Agent 同时维护需要介入、应保持沉默和找错受众的独立样本，结合延迟与成本看各类失误。 [作者发布帖](https://x.com/jakubjurovych/status/2121256499552915586) · [官方方法、结果和限制](https://deepnote.com/research/benchmark) · [模拟代码](https://github.com/deepnote/agent-initiative-benchmark) · [合成公司数据](https://huggingface.co/datasets/deepnote/pagemantle-company-dataset)

<a id="coverage"></a>
## 实际覆盖、补录与续查

本轮收录 **74 个可去重的版本事件**。回溯条目注明原始日期，主窗内新合入、工件更新和候选版按各自状态记录。索引枚举和定向正文审读仍有缺口，全局完整覆盖水位保持原值。

- **模型与评估工件**　129 个厂商、研究与评估入口中，87 个直连 HTTP 200，另用网页索引恢复部分 OpenAI、xAI 与 Mistral 首屏。156 个 HF 组织查 312 个目录，308 个返回 HTTP 200，307 个可解析，取到 230 条窗口候选。全站创建目录为模型 29 页、14,362 个候选，数据 31 页、8,498 个候选。数据集一次响应截断后以 200 条分页接续，两类索引均跨过回溯起点。按机构和主题抽筛候选。四个组织目录失败、一个响应无法解析，部分厂商动态入口、老仓全站修订和长尾候选仍待查。Lara 权重需审批，FuseAlign 和 GlotCrawl 原卡直连 401。Perplexity 英版公告经网页索引打开，直连为 403。有日期但无时刻的发布保留原日期，11 月 13 日 DeepSeek 入口按回溯补录。

- **arXiv 公告与时间窗。** cs.AI、cs.CL、cs.CV、cs.DC、cs.LG、cs.NE、stat.ML、cs.AR、cs.PF、cs.IR 十分类 new/recent 可读。十个 new 页面均为 11-13 公告批，1,376 个唯一 ID 全部关联 arXivRaw 版本历史。按官方排程这是 11-13 01:00 UTC（北京时间 09:00）的名义公告。主窗 11-14 01:00:15–11-15 00:57:48 UTC 没有常规公告。主窗十分类可见 Atom 投稿数为 0，OAI 显示 CS/stat 11-14/15 数据桶无记录。已投稿但待发布的条目需要下次获得新批再查。 [公告规则](https://info.arxiv.org/help/availability.html#announcement-schedule) · [cs.CL 公告](https://arxiv.org/list/cs.CL/new) · [cs.IR 公告](https://arxiv.org/list/cs.IR/new)
- **版本与缺口。** OAI 回溯日期桶 11-11 至 11-13 的 CS 3,529 原始行和 stat 389 原始行翻到空游标。按 UTC 投稿或修订时间过滤，十分类回溯窗口可见 718 个 v1、495 个修订版本，最新投稿为 11-12 18:59:55 UTC。OAI 的日期桶代表元数据修改日期。十分类 Atom 回溯返回总数 718，但第一页为 HTTP 503，第二、三页共返回 468 项。名为 lastUpdatedDate 的 Atom 请求实际规范化成 submittedDate，独立更新查询仍待核。
- **学术、评估与发现入口。** HF 11-13/14 日榜分别返回 34/0 条，11-15 HTTP 400。OpenReview 组页可读，TMLR、NeurIPS 2026、ICLR 2027 的 API 查询为 HTTP 403，逐篇评审及修订时间线未核齐。检查了 AACL-IJCNLP、NeurIPS、TMLR、PMLR、MLSys、OSDI、SOSP、ISCSLP、CCS 官方入口，ACL Anthology 的 EMNLP 聚合页超时，主卷页可读。中英文主题发现回到论文或发布方方法和版本取证。两条评估补录有官方公布的 UTC 时间，七篇论文补录均为固定 v1。
- **本轮取证边界。** 39 份历史日报和状态去重。保存 35 个固定 arXiv HTML 全文版本。215 次 HTTP 获取中 38 次失败或部分返回，重试后仍有 23 个 URL 缺口。标题或分类候选中 586 项待细审，本轮未独立复跑训练或硬件压测。具体 URL、时窗和候选状态见论文线台账。

- **开源工程**　主窗 `(2026-11-14T01:00:15Z, 2026-11-15T00:57:48Z]`，回溯至 `2026-11-11T01:00:15Z`。38 仓中，37 仓默认分支记录完成分页，主窗 209 条提交，整个回溯范围 1,716 条。36 仓合并列表可用，主窗 181 条，整个回溯范围 1,340 条，两条入口有交集。37 仓的发布目录可读并覆盖所查窗口，四个被 CI 标签或高频分发挤满的 Atom 用官方发布分页补查。PyPI 18/18。打开 36 个 PR 原页和差分及 4 个固定提交差分，主窗收 24 个事件、回溯补 2 个。用 39 篇旧日报、历史状态和各版本台账去重，TorchTitan #5997 的 OPEN→MERGED 是新状态。PyTorch 合并筛选和默认分支历史不一致，按固定提交核验入选条目。 `zml/llmd` 返回 404，#6097 仍 OPEN，未审候选和未枚举的维护分支留待续查。

- **开源补扫**　另查 21 个仓库入口（含重定向和可能别名），默认分支与发布索引各完成 17 个。FlashInfer、FlashAttention、Triton 共打开五项已合入改动及差分。DeepEP 和两处 DeepGEMM 完成默认分支与发布索引检查；该结果仅覆盖所查分支和索引。四个旧入口 404，候选版、未审差分与维护分支留待补查。EvolvingInfra 另以固定合入和 Registry 清单核对镜像，未运行容器。

- **硬件与软件协同**　检查 NVIDIA、AMD、昇腾、Google TPU、AWS Neuron、Intel/Gaudi、MUSA、MLU、沐曦及相关编译、通信、推理仓。335 次 HTTP 请求中 307 次成功传输，索引发现主窗 213 条、重叠窗 1043 条候选（含已报和低相关）。审读 35 份固定 Git 补丁和 5 份 GitCode MR 文件差分，收录 17 个版本事件。MLCommons 历史基线通过固定 CSV 恢复，实时榜仍未完成。部分 GitHub REST 限流，Atom 备查存在条数上限。TorchTPU wheel 到源提交的对应关系、libtpu nightly、CANN 合入到正式发行包映射和候选尾部继续待查。

- 本地 `.monitor` 记录各来源实际访问、版本去重、未查完区间和归档回执。下一轮从本次发现截止续扫并回溯 72 小时，同时保留各来源独立的未完成区间。
