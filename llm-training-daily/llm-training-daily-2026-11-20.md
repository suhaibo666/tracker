# LLM 技术动态日报 · 2026-11-20

> 时区 Asia/Shanghai（UTC+8）。增量扫描窗口 **(2026-11-19 09:04:03, 2026-11-20 09:02:12]**，向前回溯 72 小时至 **2026-11-16 09:04:03**。实际生成 **2026-11-20T09:15:13+08:00**。来源仅有日期时保留原日期，提交、发布、会程与发现时间按各自证据记载。

## 本次重要变化

- **值得深入读的开放模型。** Olala mini 把Mamba-3、DiffTPA与latent MoE配在约6.8B总参/1B激活中，已交付权重、代码与70页报告；但交付卡承诺32K，训练扩窗、config字段和实际支持范围存在差异，长生成还要求FP32递归状态。[固定模型卡](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B/blob/b40f8165ee2254c18601220e19842600c6d91794/README.md)
- **先修复会改变训练结论的正确性。** verl拒绝detached k3奖励与policy gradient的错误组合；TensorRT-LLM纠正raw logprob被采样惩罚改写；MORI披露特定AINIC固件可成功返回但数据缺失，单READ规避仍有残余风险。[verl](https://github.com/verl-project/verl/pull/8458) · [raw logprob](https://github.com/NVIDIA/TensorRT-LLM/commit/ffed918b08313d69c1fff9dbb18b1d14531daa96) · [AINIC](https://github.com/ROCm/mori/commit/1eb89cc391b06622953f9b1ebe6e82987be68744)
- **训练系统有可执行的升级候选。** Megatron Core0.20.1正式回补MoK与MFSDP v2 EP恢复，TorchTitan激活卸载进入主线；TorchTPU交叉熵绕开全量log-prob物化，改善大词表显存及特定reshape编译瓶颈。正式tag与同日main必须分开评估。[MCore release](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.20.1) · [TorchTitan](https://github.com/pytorch/torchtitan/pull/6137) · [TorchTPU](https://github.com/google-pytorch/torch_tpu/commit/b033ad1468dc39e5c798502cc66683d773f253a0)
- **推理优化要同时看代价与部署约束。** FACT展示仅改host线程便能把投机加速变成减速；MLXcel INT8 MLA省缓存却在部分长窗口显著更慢；FlashInfer batch-invariant采样也存在明显性能退化区。适合按实际负载复现，而非只摘最好倍数。[FACT正文](https://arxiv.org/html/2611.23463v1) · [INT8 MLA证据](https://github.com/lablup/mlxcel/commit/7a85fe51310da33cf045bbb17357db5b5bfb3503) · [FlashInfer](https://github.com/flashinfer-ai/flashinfer/pull/9619)
- **评估纠错与组合消融优先。** harness-hub把每次调用输出上限从16K改32K后，thinking基座/适配器SWE hard51均74.5%，修正此前比较；小模型因子研究把MoE、Muon等组件及交互作用放在同一重复实验中，仍不能外推大规模。[预算更正](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted/blob/80d05c9051dfd2ae464e1daf4329cb15524caf60/README.md) · [因子实验](https://arxiv.org/html/2611.24540v1)

[模型与技术报告](#topic-1) · [预训练与架构](#topic-2) · [后训练](#topic-3) · [推理与部署](#topic-4) · [开源工程](#topic-5) · [硬件与软件协同](#topic-6) · [评估与可靠性](#topic-7) · [覆盖与续查](#coverage)

<a id="topic-1"></a>
## 模型与技术报告

### OVHai LLM／OVHcloud：Olala 从预告进入可下载权重与70页报告

- **时间与版本**：11月17日预告已见昨日；本次新增是11月19日公开权重与报告。post-trained 固定 `b40f8165ee2254c18601220e19842600c6d91794`（10:13:13Z），同一发布还提供 base、SFT、summarise、translate、MLX 和 GGUF。模型仓早于发布日期创建，不能把10月30日创建时间写成首发。Apache License 2.0 的 LICENSE／NOTICE 已在固定快照打开。
- **技术变化**：模型卡为6.8B总参、约1B激活；config 是36层、hidden1536、29个Mamba-3层+7个DiffTPA注意力层、256路由专家取6加1共享专家。报告 §7／Table23 把低KV开销与潜在空间MoE结合；§3/§6另给出组件消融。固定实现 `modeling_olala.py::OlalaMonoBlock.__init__` 将 M/V 分派至 Mamba3／DiffTPA，`OlalaMoE.forward` 用sigmoid加expert_bias选top6，再在384维latent执行专家。这两处与报告结构对应，未运行验证。
- **实物与边界**：树中BF16单文件 **13,579,677,592字节**，不能把“约1B激活”写成只需1B参数内存。模型卡要求其维护的推理fork；标准Ollama／LM Studio／llama.cpp／vLLM／mlx-lm尚不能直接加载。vLLM必须 `--mamba-ssm-cache-dtype float32`，文档称默认BF16递归状态会在长生成发生重复；CUDA13镜像要求driver580+。报告§12及附录C披露少数Mamba heads出现极端幅值、平方ReLU激活会让带FP16 block scales的量化路径溢出；不能据短prompt无异常推断通用量化可用。
- **冲突保留**：报告§7.4描述训练中的256K扩窗，config `max_position_embeddings=100000`，而**交付模型卡仍承诺32K输入+输出，128K写为将来版本**。日报只采用32K可支持范围，不把训练阶段或配置字段直接升级为交付保证。
- **意义／建议（推断）**：这是研究小激活参数混合架构、公开训练配方及数值退化的完整工件；复现优先验证长生成状态精度和量化，而非直接采用市场稿的“8GB可运行”概括。[固定模型卡](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B/blob/b40f8165ee2254c18601220e19842600c6d91794/README.md)；[固定技术报告](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B/blob/b40f8165ee2254c18601220e19842600c6d91794/Olala-v1.0-mini.pdf)。

### Google：Genie 4 GA与Interactions按需加载工具schema

- **时间版本**：官方changelog标11月19日；按日期发布记录收录主窗，未核精确上线秒。`models/genie-4`为服务别名，无公开权重／训练报告版本。
- **Genie变化**：通过Interactions `v1beta`，从文本或16:9参考图建立世界，实时键盘／文字操控720p、24fps视频，配48kHz立体声音频，支持WebRTC/WebSocket。上限120秒，但文档明确物体持久性和视觉一致性主要在前45—90秒。服务区域美国、西欧、日本；区域外网络时延会减弱操控响应和帧率。结束要显式complete以停止服务生成。属于世界生成接口交付，不能据此推断物理正确性或机器人可迁移能力。[模型正文](https://ai.google.dev/gemini-api/docs/models/genie-4)；[发布记录](https://ai.google.dev/gemini-api/docs/changelog#11-19-2026)。
- **工具加载变化**：同日Preview允许 `defer_loading:true` 配 `tool_search`，初始上下文只给deferred工具名／描述，调用时才加载schema；服务端可检索预声明function与MCP，客户端检索可接私有注册表、会多一次应用往返。仅Interactions、Gemini3.6+；generateContent不支持。本项没有公开延迟／准确率对照结果，不能把“少schema”直接量化为吞吐提升。[工具检索正文](https://ai.google.dev/gemini-api/docs/tool-retrieval)。

### OpenAI：GPT-Live-1.1语音迭代

官方11月19日changelog和模型页新增 `gpt-live-1.1`，宣称相较GPT-Live-1改善多语种转录、发音与口音／声音一致性，并保留边听边说的full-duplex模式；推理与工具任务可按自然语言描述交给后台。使用 `v1/live/sessions`；模型页只给此alias，没有固定日期snapshot、训练细节或量化基准，改善幅度不可比较。新同页提示工程文章说明声音选择与语言／地区指令，但不能替代新模型评测。建议用相同噪声／口音集A/B复测再迁移。[官方发布记录](https://developers.openai.com/api/docs/changelog)；[模型页](https://developers.openai.com/api/docs/models/gpt-live-1.1)。

### Vision-Decision：Glance-X-4B三方向视觉定位适配器

11月19日03:50:50Z固定 `fa2a2c36fc641814dd6252e9cb15ca3f64ea8594`，基于Qwen3.5-4B-Base的LoRA＋presence/box/IoU/multi-box heads；模型卡声明CC BY-NC-SA4.0、非商业用途，代码Apache2.0不可替代权重许可。作者让同一权重按image-first缓存图像、text-first缓存查询或joint双向融合，面向一图多问／一问多图的复用。RefCOCO八split平均Acc@0.5分别90.7/89.4/90.9%，原image-only preview90.8%；没有全面优于旧模型。300图自建eval是pipeline标注且同分布，不能当独立人工基准。基座需另下载，图像域COCO自然照片，未训练屏幕／文档；本轮只核卡、未核GitHub机制实现，吞吐收益不作定量结论。[固定卡](https://huggingface.co/Vision-Decision/Glance-X-4B/blob/fa2a2c36fc641814dd6252e9cb15ca3f64ea8594/README.md)。

<a id="topic-2"></a>
## 预训练与架构

**Do Frontier Model Ingredients Add Up?（2611.24540v1，提交 11-18）：组件增益大致叠加，但不能用单组件实验代替组合消融。**

第一作者：Sengupta, Biswa及其团队；原始提交：2026-11-18；19 November 公告恢复（首发秒界未确定）。

 作者在同一 8 层、宽度 512、约 51.5M 激活参数模型上，以 FineWeb 300M tokens、1,024 序列长度做 16 组分数因子配置×两 seed。Table 2 中 MoE、Muon、Gated DeltaNet 混合、额外归一化的平均损失 contrast 为 −0.0983、−0.0506、−0.0365、−0.0194 nats；参数预算匹配的 attention gate 为 +0.0018，落在 ±0.0043 的参考界内。决定性增量是重复运行与交互项分离，而非新组件。**限制：** MoE 总参约 161M，优化器学习率也随配方变化，分数设计还混叠高阶效应；这不是等总参、等墙钟或大模型结论。建议复现其因子实验和真实模型宽度下的因果性／极端 router 梯度检查，优先于直接复制最优配方。[正文 §3–6 / Tables 1–2](https://arxiv.org/html/2611.24540v1)

**General progress substitutes for specific training data（2611.24844v1，提交 11-18）：去掉技能监督会延迟技能，但背景能力进步也会降低补回数据的成本。**

第一作者：Marosi, Mark及其团队；原始提交：2026-11-18；19 November 公告恢复（首发秒界未确定）。

 方法在棋类、SMILES、括号语言和网页文本中，按背景 loss 对齐模型，把指定 next-token target 从 loss 中稀释，再恢复监督；用等难度的标签置换规则控制“只是学得更快”。§2.3 中棋类最后检查点补回技能仅需从头训练数据量的 0.10%–0.78%，但计入 50 步优化器状态恢复后为 3.1%–7.5%；置换规则需要 2.4–16 倍更多 tokens。7B 的 Rust 文档过滤实验提供相邻证据。**限制：** target thinning 保留输入 token，不等同完整数据删除；Rust 只过滤至约 3%，7B 部分仅单 seed，不能据此宣布所有过滤或遗忘无效。建议深入读 Fig. 3、Appendix E/F，并把“监督删除”“文档删除”“遗忘”分开复现。[正文 §2.3 / Fig. 3](https://arxiv.org/html/2611.24844v1)

**Chorus（2611.23386v1，提交 11-17）：共享 n-gram 表，再用层内适配复用，而不是每层各建记忆。**

第一作者：Chung, Stephen及其团队；原始提交：2026-11-17；19 November 公告恢复（首发秒界未确定）。

 同一查表特征在输入和偶数层 attention values 读取，用层特定投影／门适配；Ember 将表优化器状态从两个完整矩阵降为行列统计。Table 1：133M backbone＋60M 表、FineWeb 5.24B tokens、三 seed 下 BPB 为 1.0568±0.0001，TN-gram 1.0648±0.0006、Engram 1.0665±0.0001；22 项下游平均仅领先最强基线约 0.0030。Table 3 用同 optimizer 对照，避免把全部收益归给 Ember。**限制：** 表容量曲线与组件消融只有单 seed；99.2% 节省针对 optimizer state，不是完整训练显存；Ember 对 FactorEngram 反而有害。建议先复现共享／分表、投影共享与优化器正交对照。[正文 §2–3 / Tables 1–3](https://arxiv.org/html/2611.23386v1)

**Multi-Branch Transformer（2611.19800v1，提交 11-15；历史候选补录）：把串行层的一部分改为中部并行分支。**

第一作者：Li, Gaotang及其团队；原始提交：2026-11-15；历史候选补录（原公告日期未确定）。

 shared prelude→独立参数的分支→融合→shared coda；Joint-KV 再减少分支缓存冗余。Table 1 的四个 MoE 规模约 0.8B–2.1B，2 分支在匹配物理 block、宽度、训练 tokens 下 loss 分别降低约 0.025–0.031；W3 中 Balanced 2 分支为 2.436，对照串行 2.471（Table 2，body compute 1.5301 vs 1.5032 PFLOP-days）。**限制：** 1.26× 属拟合的训练 body-compute 等效关系，不能直接当完整训练加速；分支并发、融合和缓存共享还需服务实现配合。建议先读宽深配比与中部放置消融，再校验 Joint-KV 的质量／运行时，不跨设置比较损失。[正文 §3–4 / Tables 1–2, 6](https://arxiv.org/html/2611.19800v1)

<a id="topic-3"></a>
## 后训练

### Sionic：3B韩语高考数学GRPO小规模训练披露

11月19日05:06:50Z建仓、19:46:09Z快照 `4066e88b96590f7af3b5826509fcee64b4ab912d`。基于Nanbeige4.2-3B，第一阶段683题、40step，第二阶段1078独立题／1145行、10step，并披露难度过滤、动态采样与截断响应打零的训练选择。其2027学年考试46独立题每题16次采样、BF16、temperature0.6/top-p0.95/top-k20、thinking on、65536生成上限：**满分是第一阶段checkpoint的逐题多数票**；当前第二阶段仓的三选修均96分多数票、平均95.2—95.5。不能写成新仓一次采样满分，更不能外推其他学科／真实图像，图题用文字描述。作者称训练用8×B300，第一阶段约8.7h、第二阶段约6h；原版Nanbeige代码在Transformers5.x RoPE初始化有NaN问题，卡建议4.x／重算inv_freq，移动端另需tokenizer patch。Apache2.0仅卡声明，独立LICENSE404，未运行其补丁。建议作为小样本RL预算与checkpoint选择反例阅读。[固定卡](https://huggingface.co/sionic-ai/sionic-K-SAT-MATH-3B/blob/4066e88b96590f7af3b5826509fcee64b4ab912d/README.md)。

**MRPD: Replay, Don't Resample（2611.24059v1，提交 11-18）：专家训练日志本身就是可重放的课程。**

第一作者：Yue, Zhenrui及其团队；原始提交：2026-11-18；19 November 公告恢复（首发秒界未确定）。

 从共同 SFT 初始化出发，按专家原训练顺序重放带行为概率／奖励的轨迹，重算 policy gradient，并在记录的 prefix 上向最终专家蒸馏，省去学生重新 rollout 与环境调用。4B dense、三领域、300 步的 Table 1 中，三领域均值 MRPD 58.6，MOPD 58.1、MOPD＋GRPO 58.2、online GRPO 52.5；但 MOPD reasoning 63.8 略高于 MRPD 63.7，TIES search 42.6 也高于 MRPD 42.0。**限制：** 专家和学生必须同初始化，异初始化未测；大部分配置单 run，不能把约三分之一 MOPD 每步成本推广到所有尺度。建议复现原序／打乱序、同日志 PG×KL 消融，保留 logprob、奖励、顺序及 reset 元数据。[正文 §3–4 / Tables 1–4; Appendix G](https://arxiv.org/html/2611.24059v1)

**ECA: Evidence-Based Credit Assignment（2611.23824v1，提交 11-18）：给失败的搜索轨迹按“取得的证据是否有用”分配信用。**

第一作者：Wang, Wen及其团队；原始提交：2026-11-18；19 November 公告恢复（首发秒界未确定）。

 冻结当前策略快照，只给问题＋去重文档，计算参考答案 likelihood 增益；Horizon 保留跨步 telescoping 增量，并以 novelty gating、cross-fold 归一化和 search-local mask 分配辅助优势。Table 1：Qwen2.5-3B/7B 七 QA 宏 EM，Horizon 38.22/42.62，GRPO 37.19/37.82。3B 中 query-only 35.27、全轨迹广播 36.42 均低于 GRPO（Table 2），说明有非零奖励还不够，放对 token 更重要。**限制：** 教师前向增加训练成本；单个标准答案 alias 是不完美效用代理，文档 ID 去重也不能消除近重复。适合搜索 RL 信用分配复现，不直接推广到代码／任意 tool agent。[正文 §3–5 / Tables 1–2](https://arxiv.org/html/2611.23824v1)

**Depth Before Breadth（2611.24697v1，提交 11-18）：先由成功轨迹尾部向前扩展可解区域，再进入离轨探索。**

第一作者：Du, Siyuan及其团队；原始提交：2026-11-18；19 November 公告恢复（首发秒界未确定）。

 paired advantages 分别奖励“到达更可解状态”与“从该状态完成任务”。Table 2 的 Qwen2.5-1.5B WebShop 消融：reverse-only 68.8%，加 forward 74.7%，再加反向／正向优势为 76.6%／80.2%。Table 1 中 1.5B/7B 的 WebShop 成功率 80.2%/82.6%，GiGPO 引用值为 65.0%/72.8%。**限制：** 基线取自此前论文，作者标准差是同 checkpoint 的三次评估 seed，并非三次训练；需要成功参考与中间状态可靠 reset，suffix 全失败时优势仍为零。建议先验证环境 step 和墙钟是否一致记账，再复现课程×优势消融。[正文 §3–4 / Tables 1–2](https://arxiv.org/html/2611.24697v1)

<a id="topic-4"></a>
## 推理与部署

### mlx-community：Clairic结构化抽取的Apple Silicon移植

11月19日11:21:49Z `1a48d6b31d98575980883ace04cfb20d7814fc6e` 和Flash版 `ae97c5d2864c8982cf79eaf7cf2c2d27e46885f4`，为Cloudflare既有模型的**社区4bit移植**，不是Cloudflare新模型首发。固定LICENSE为Apache2.0。卡片保留Cloudflare逐字段grammar解码／typed head管线，用MLX引擎替换CUDA；只加载骨干无法重现结构化结果。M5 Max128GB/macOS27.2/mlx0.32.4热模型下，4K文档一字段7.2s、8字段70.7s，字段重读文档造成近似字段数×文长成本；这不是普通聊天tokens/s。作者说明量化会改少数答案、confidence均差0.024且最大0.68，应重新校准阈值；长近65K输入未测。本轮未审计移植代码或复跑一致性，对其准确率只作归属作者的记录。[固定移植卡](https://huggingface.co/mlx-community/clairic-4bit/blob/1a48d6b31d98575980883ace04cfb20d7814fc6e/README.md)。

- **Edge0部署文档修订**：11月19日16:44:40Z `17bb271c83b083de3ac2bc1922911ba1462bc788` 增加145B Q3_K“12GiB engine VRAM预算+20GiB cgroup RAM、SSD流式专家”行；实际仍同一RTX5090D24GB受人为限制，**不是实测12GB显卡最低配置**。同时新增NVFP4 Metal支持说明；未运行验证。建议关注内存分层而非误读为145B全装12GB。[固定卡](https://huggingface.co/Edge0/Edge0-145B-NVFP4/blob/17bb271c83b083de3ac2bc1922911ba1462bc788/README.md)。

**Late Entry（2611.24576v1，提交 11-18）：保留深层历史缓存，让新 token 跳过前几轮 recurrence。**

第一作者：Jin, Ziqi及其团队；原始提交：2026-11-18；19 November 公告恢复（首发秒界未确定）。

 轻量 SFT 先修复 embedding 与后期 hidden-state 的幅度错位；后台 piggyback 完成已提交 token 的全深度状态，Wake verification 对照完整 greedy 预测，必要时 rollback。Ouro-2.6B、T=4、B=8、单 H100：§4.5 的 K=2/τ=0 在 GSM8K 达 5.20× greedy-exact decode 加速，相对 K=4 block 增量约 26%；τ=∞ 的更快路线有 1.4–4.0 点质量损失，必须分开。**限制：** exact 限 greedy，采样尚未实现；浅 recurrence 的 rollback 可使 exact 路线慢于全层 block；Stage-2 约 1.9B tokens、8×H100，并非零训练成本。建议深入读 §4.5/Limitations，复现 rollback、bf16 容差和并发下动态 block 大小。[正文 §2–4 / Tables 3–5](https://arxiv.org/html/2611.24576v1)

**LoopCache（2611.24301v1，提交 11-18）：减少 looped decoding 的缓存读流量，比只提高同时推进 token 数更关键。**

第一作者：Fu, Zizhuo及其团队；原始提交：2026-11-18；19 November 公告恢复（首发秒界未确定）。

 每步循环只读一个完整 loop KV，按最近匹配循环做准确验证，随中间层 confidence 在线修订草稿，并只给少量浅层保留独立 KV。六个 looped 模型、4K–128K、LongBench/RULER/LongReason、8×A100（长上下文用 context parallel）下，§5.2 平均相对 AR 为 5.12×、相对 WFD/LoopSpec 为 10.25×/14.56×。**限制：** 后两个倍数主要反映这些基线在长上下文的流量放大，不能与单 H100 短上下文论文横比；YaRN 延展区间与原生窗口也要分开。建议用作者 §5.2 的 KV bytes／forward 与 committed tokens／forward 联合解释收益，先复现4K再扩长。[正文 §3–5 / Table 1, Figs. 8–12](https://arxiv.org/html/2611.24301v1)

**LoopFill（2611.19649v1，提交 11-14；历史候选补录）：把 suffix 草稿和浅循环预测合成一棵树，填补流水线气泡。**

第一作者：Zhu, Siqi及其团队；原始提交：2026-11-14；历史候选补录（原公告日期未确定）。

 两种 drafter 共享节点、计算和验证，以路径概率×suffix 频率排序，只让预算内节点参与 loop 执行。Table 2 的 concurrency=1、temperature=0、七任务算术平均：Huginn-3.5B 5.13×，LoopAhead 4.12×；Ouro-2.6B 5.85×，LoopAhead 4.19×，均相对各自 AR。**限制：** 温度1的 Huginn 平均仅2.60×；Huginn 节点预算16→64虽令B=1快19.5%，却令B=16吞吐低43%。建议复现请求并发×节点预算，勿把decode latency或acceptance当服务吞吐。[正文 §3–4 / Table 2, Fig. 5](https://arxiv.org/html/2611.19649v1)

**NarrowSpec（2611.23835v1，提交 11-18）：feature drafter 的 residual width 无需继承 target 宽度。**

第一作者：Valliappan, Alagappan及其团队；原始提交：2026-11-18；19 November 公告恢复（首发秒界未确定）。

 增加输入宽度适配；DFlash 的输出适配器与 target LM head 在加载时合并为一个线性矩阵，避免重新拉回全宽。Table 2–3 使用 Qwen3-8B/NVFP4、GPT-OSS-20B/MXFP4，BF16 head，880 SPEED-Bench prompts、batch1、32K context；半／四分之一宽在 B200 端到端 decode 提速 1.06–1.12×，同协议 GB10 为1.13–1.44×（Table25）。**限制：** 同数据预算而非同收敛程度、每配置单训练seed；KV attention未按宽度缩小，长context预期收益减小；已量化 target head 的folding未测，不能声称逐比特等价。建议按硬件复现宽度×acceptance×iteration-time，并把首decode/prefill单列。[正文 §2–4 / Tables 2–3, 25](https://arxiv.org/html/2611.23835v1)

**FACT（2611.23463v1，提交 11-17）：草稿训练、候选分配和执行条件的收益不能相乘。**

第一作者：Fares, Hadi及其团队；原始提交：2026-11-17；19 November 公告恢复（首发秒界未确定）。

 训练时让梯度穿过多步 feature rollout 并提高早期权重；推理时在固定256节点和逐深度相同宽度下用confidence重新分配候选。Table1：Llama-3-8B-Instruct、greedy、H100、每任务50 prompts、128输出上限，HumanEval静态1.46×→动态2.02×，MT-Bench1.31×→1.75×，latency含prefill。§4.4 只把host线程8→208，GSM8K speedup1.91×→0.92×而acceptance不变。**限制：** 不是与原生EAGLE-2整体系统比较；两seed训练干预同时改变weighting与gradient flow；短输出小样本不能证明采样／长输出／批服务收益。建议作为投机复现实验记账模板深入读。[正文 §4–5 / Table 1](https://arxiv.org/html/2611.23463v1)

**ShelfKV（2611.23769v1，提交 11-18）：图像从文本cache预算中受保护，但只在需要时进入attention。**

第一作者：Wang, Ziyi及其团队；原始提交：2026-11-18；19 November 公告恢复（首发秒界未确定）。

 CPU保存全部视觉KV，GPU放少量4-bit候选key；每步把当前query最相关的图像条目换入低分文本slot。Qwen3-VL-8B/4B-Thinking、R-KV/DBTrimKV，Table1/§5.4在匹配包含辅助数据的GPU缓存下，CharXiv后续问题相对DBTrimKV最高恢复20.3点；同384条目预算，相对最佳固定分区约多3点。**限制：** 每步更慢、CPU总内存未减少、prefill峰不变；只保护图像，文本丢失仍永久；实现只支持prefill就提供的图像。建议多轮视觉推理先诊断“保护不足”还是“容量不足”，将CPU带宽/解码开销与显存一同报告。[正文 §4–6 / Tables 1–3](https://arxiv.org/html/2611.23769v1)

**The Hidden Cost of MoE Diffusion Decoding（2611.21852v1，提交 11-16；旧候选补查）：专家量化后，采样与停止流水线会变成主瓶颈。**

第一作者：Samrat, Harshit及其团队；原始提交：2026-11-16；旧候选补查（本轮发现，不作当天首发）。

 canvas diffusion每次对全canvas×vocab处理，单输出token采样logits工作约为AR的S倍。26B diffusion MoE、单GB10、canvas256、单用户的Table1：sampling/host占34.4%–39.8%，专家GEMM约25%。Table2相对bf16，W4A16专家先到1.58–1.68×；加bit-exact pipeline到1.78–1.99×；turbo到2.42–2.67×。**限制：** bit-exact仅保证与W4A16参考逐token/step相同，turbo是accuracy-parity并非逐比特；accuracy只100题/任务，未证明等价；eager baseline，compile失败未报compiled对照，未测datacenter与多用户。建议优先复现profile分桶，拆开量化、采样融合和norm收益。[正文 §2–5 / Tables 1–3](https://arxiv.org/html/2611.21852v1)

**QDA（2611.23388v1，提交 11-17）：把量化配方搜索变成短代理评估＋长生成复核的闭环。**

第一作者：Li, Xingyang及其团队；原始提交：2026-11-17；19 November 公告恢复（首发秒界未确定）。

 agent组合兼容的PTQ组件，用不向agent暴露的推理轨迹KL评分，长生成抓循环等代理遗漏；Table2每搜索64 GPU-hours，四个推理模型相对最强公开配方平均提升W4A16/W4A8/W3A16准确率0.9/1.4/2.2点。Table1前三种格式在GH200/vLLM0.27.1，NVFP4在GB300/vLLM0.31，不能混成同硬件对照。**限制：** 多数搜索单run，Qwen3.6-35B-A3B为三run；W3部分搜索由已有GSQ checkpoint继续；库/evaluator/harness写为“将发布”，本轮未核验可用实现。建议等artifact后复现固定预算、独立heldout、代理KL与长生成失败检测。[正文 §3–4 / Tables 1–2](https://arxiv.org/html/2611.23388v1)

**Cap-and-Track（2611.24891v1，提交 11-18）：用后端拒绝信号跟踪容量，把长队列留在执行引擎之外。**

第一作者：Novotny, Benjamin及其团队；原始提交：2026-11-18；19 November 公告恢复（首发秒界未确定）。

 Cap在饱和资源处快速拒绝，durable priority queue承接与重试，Track只根据capacity refusal上探/回退，不猜输出长度。Meta异步分类同流量并行生产A/B，报告43%更少H100；稳定容量real-engine replay的goodput距调高的固定rate不足1%，refusal份额约其四分之一。**限制：** 43%是Cap＋Track整体相对旧queued/fixed-rate系统，不是单控制器消融；低优先级backfill峰等待更久，其他生产结果多属观察；未用于交互或agent-session流量。建议读队列位置、拒绝分类和低优先级SLO，复现先对齐goodput及尾等待。[正文 §3–6 / Limitations](https://arxiv.org/html/2611.24891v1)

<a id="topic-5"></a>
## 开源工程

主窗：(2026-11-19T01:04:03Z, 2026-11-20T01:02:12Z]。下面区分默认分支合入与正式release，所有性能/精度数据为作者在所述设置报告；本轮未使用GPU/NPU独立压测或复训。

### verl：拒绝会误导逆 KL 梯度的蒸馏配置

蒸馏的估计器与更新方式不能任意组合：把 detached k3 当奖励做 policy gradient，即便完全 on-policy 且不裁剪，也不保证是 reverse-KL 梯度。`DistillationLossConfig` 现拒绝 k3/low_var_kl 与 `use_policy_gradient=True`，PG 使用者须迁到 k1/kl；直接反传仍可用 k3，k1/kl 的直接反传组合继续拒绝。Python dataclass 的 PG 默认改为 False，与 YAML 对齐。

这是**breaking 配置校验**，没有重写 PPO 或 loss kernel；作者用精确枚举梯度、有限差分及 CPU 测试佐证，不是大模型训练质量实验。**建议优先审查已有蒸馏配置**，不要仅按估计器的数值低方差判断优化目标。

一手来源：[#8458](https://github.com/verl-project/verl/pull/8458)（`3108bebc`，2026-11-19T05:18:25Z）。

### verl 与 TRL：空样本不能污染指标，也不能让训练无限等批

verl 的 IS/RS 序列指标现统一只统计 `response_mask` 有有效 token 的行，避免平衡批次或固定形状 padding 把均值、极值、比例拉偏；loss 和梯度未改。TRL AsyncGRPO 则在连续丢弃的已生成 group 达到数据集 prompt 数时抛错，防止同奖励 group 全被过滤后 learner 永远等不到 batch。`filter_prompt` 未生成的行不计入该连续计数；未知长度的流式数据没有此上限。**建议回归 padding 不变性与全零方差奖励场景**，两项分别是指标和活性保证，不能据此宣称策略质量提升。

一手来源：[#8628](https://github.com/verl-project/verl/pull/8628)（`a4e0dd21`，2026-11-19T06:20:49Z）；[#8425](https://github.com/huggingface/trl/pull/8425)（`d9930ad7`，2026-11-19T11:27:16Z）。

### TorchTitan：把 remat 保留的激活卸到 CPU，并协调 FSDP 参数回传

新增 `ActivationOffloadConfig`，与 `ActivationRematConfig.offload_block_input/offload_regions` 配合：remat 决定重算、留 GPU 或卸载，saved-tensor hooks 把所选 storage 异步拷到 pinned host pool，在相邻 block backward 前回载。按容量复用 host buffer，以免 EP 路由大小变化让 pinned memory 不断增长；小于 1MiB 的 storage 留在 GPU，保存版本在 unpack 时检查。

#6137 进一步令 activation reload 等待 FSDP2 的 `get_all_gather_copy_in_stream()`，先完成参数 H2D，再让参数 all-gather 与激活 H2D 重叠。依赖昨日落地的 PyTorch API；单 rank/current-stream 分支没有同样收益，作者未跑该组合的端到端吞吐。默认不启用，PP 每 stage 的末 block 默认留 GPU，encoder 不卸载，GraphTrainer 拒绝该配置。**建议深入阅读并按 host 带宽/显存预算复现**，不能把可重叠的时间上界当实测加速。

一手来源：[#6020](https://github.com/pytorch/torchtitan/pull/6020)（`0e4284ad`，2026-11-19T22:58:27Z）；[#6137](https://github.com/pytorch/torchtitan/pull/6137)（`2377d43e`，2026-11-19T22:58:29Z）。

### TorchTitan：KDA/GDN 的 CP 元数据可进入整步 CUDA graph

此前每批新建的 linear-attention 元数据是 opaque leaf，且 CP 本地文档段数随 packing 改变，graph replay 既可能非张量值不等，也可能 shape 改变。现注册 KDA/GDN、Ulysses、CP 元数据与 routing 的 pytree，并按文档槽/分片范围给 routing 固定容量，以空段补齐。**建议在变文档边界的多批 replay 中回归**；这消除 metadata 障碍，不代表 Kimi-K3/Dist-MoE 整个训练栈或所有 CP load balancer 已完成。

一手来源：[#5680](https://github.com/pytorch/torchtitan/pull/5680)（`8592e69c`，2026-11-19T08:55:19Z）。

### DeepSpeed：ZeRO-1/2 offload 释放旧 optimizer view，再恢复片段映射

optimizer-state fragments 原是 device state 的 narrow view。offload 替换 state tensor 却保留 view，使旧 GPU 内存存活；reload 又建一份，safe_get/safe_set 还可能读旧值或因重绑遗漏而报错。现在卸载前去掉片段引用，reload 后重新链接状态；首次 step 前尚无状态的情形保持延迟初始化。作者在双 GH200、33.6M 参数、ZeRO1/2 的指定例子观察到每 rank 128MiB state 真正释放；**建议对实际 offload/reload 周期检查显存与 safe API 数值**，不外推成任意规模的固定节省量。

一手来源：[#9016](https://github.com/deepspeedai/DeepSpeed/pull/9016)（`40a9f40b`，2026-11-19T20:39:00Z）。

### DeepSpeed：Muon 的形状与 per-head 标签都必须贯穿更新路径

卷积梯度先展平做正交化后，现按实际矩阵维度计算 aspect-ratio scale，再恢复原形状进入溢出 guard；原来四维 shape 混用会失败。另在 ZeRO1/2 padding 分支、MuonWithAuxAdam 和 BF16 optimizer 补传 `muon_num_heads`，避免开启参数对齐等无关选项后悄悄退回整矩阵正交化。**建议回归更新张量本身**，普通 loss 有限不能发现 per-head 被忽略；作者这组验证是单 rank CPU，未给多卡训练收益。

一手来源：[#8552](https://github.com/deepspeedai/DeepSpeed/pull/8552)（`64616608`，2026-11-19T05:01:37Z）；[#8612](https://github.com/deepspeedai/DeepSpeed/pull/8612)（`b5e0ba67`，2026-11-19T05:03:30Z）。

### DeepSpeed：hybrid rollout 在无通信片段注入 kernel，训练仍共享原权重

segment-KI 通过 GLU/GDN 投影结构识别可融合片段，带 TP collective 的 projection 排除；GLU 还检查 SiLU，不能从旧描述中的模型家族名单推断所有 gated MLP 都支持。替换读取原 Parameter，经 autograd.Function 保留反向，避免 rollout/train 间复制或 inject/eject 权重；greedy decode 可走 CUDA graph/C++ loop，并保留逐级回退。EOS 检查的 D2H 同步被摊到每16步。**建议考虑在小批共置 RL 中复现**；作者的 Qwen3.5-4B/单4080 SUPER greedy batch1结果不代表大批、采样或多GPU AutoTP，后者明确未验证。

一手来源：[#8602](https://github.com/deepspeedai/DeepSpeed/pull/8602)（`97aa3c10`，2026-11-19T04:09:40Z）。

### vLLM：NCCL/IPC 换权重采用同一种有界字节流

packed 权重传输改为每个 tensor 起点16字节对齐的统一字节流，严格按 buffer 容量切 chunk，单个大 tensor 也可跨 chunk；IPC 首块携容量/handle，末块显式 `last`，单个 start/finish 生命周期只准备/完成一次 reload。它修正“整 tensor 跨过阈值才发出”无法约束峰值的问题。

容量不是总峰值保证：扩容时旧/新 buffer 短暂共存，allocator 可能保留旧块；发送跨块 non-contiguous tensor 仍需整 tensor contiguous 副本，CPU source 同步复制。**建议检查 trainer/engine wire schema 和重试边界后采用**，不能仅把 packed 开关等价为固定1GiB总额外显存。

一手来源：[#68043](https://github.com/vllm-project/vllm/pull/68043)（`97279420`，2026-11-19T05:42:11Z）。

### vLLM：KV 传输失败必须收敛，源 block 必须等 WRITE 真正结束

NIXL push 将默认 UCX worker threads 设为0以避免大 WRITE 提交卡在内部线程池，同时为 agent 并发访问启用同步模式；发送中的 block 持续保留租约，不能在提交尚未结束时释放给新请求。Mooncake store 则校验返回数量/状态，失败 load/put 走完成与失败记账，避免把部分返回当完整传输或让请求一直悬挂。**建议优先回放大KV、失败返回、超时与取消并发**；这里核实生命周期与负面测试，未测 RDMA 吞吐，传输提交不等于数据安全可复用。

一手来源：[#66795](https://github.com/vllm-project/vllm/pull/66795)（`cefcae8b`，2026-11-19T04:06:48Z）；[#67939](https://github.com/vllm-project/vllm/pull/67939)（`a38ff116`，2026-11-19T09:57:43Z）。

### vLLM：在线重重量化逐层回收源数据，并接受更多 FP8 MoE 工件

在线转换可读 `fp8`、compressed-tensors、Quark 的动态激活量化 FP8 MoE，覆盖 per-tensor/per-channel/block 权重scale，再转换到目标在线 MoE 格式（包括 block-FP8→MXFP4）。FP8 expert 反量化采用有界 FP32 staging，逐层完成后释放序列化源参数，避免整份源/目标同时驻留。目标 kernel 的硬件要求仍适用；源FP8布局不应由目标backend错误选择。**建议对峰值、源scale布局和 reload 数值做实测**，代码的分块/回收设计不足以证明所有模型都按同一比例省显存。

一手来源：[#68171](https://github.com/vllm-project/vllm/pull/68171)（`fd2b7ac4`，2026-11-19T12:15:02Z）。

### vLLM：MRv2 可增加异步调度深度

新增 `async_scheduling_depth>=2`，默认2表示GPU当前步加一层预排 host 工作；更深队列旨在 host 往返长于GPU一步时减少空隙，例如跨节点小批 speculative decode。只有启用 async scheduling 才有意义，相关 scheduler/request 状态与 adaptive verification 按在途步数调整。**建议对低batch、多节点负载复现 host bubble 与取消响应**；排得更远增加在途状态，不能只凭参数上调就推断吞吐或时延改善。

一手来源：[#64145](https://github.com/vllm-project/vllm/pull/64145)（`949de058`，2026-11-19T06:51:34Z）。

### vLLM/SGLang：长偏移、warp 同步和 SwiGLU clamp 的正确性回归

vLLM fused QK-norm/RoPE 的 token 行偏移改64位，并使 NeoX partial-rotary 的 warp collective 由全warp参与；fused KDA block reduction 在 lane0重用scratch前加 `__syncwarp()`；batched DeepGEMM MoE补传 SwiGLU clamp。SGLang KDA prefill 则把 program/token/sequence offset 运算提升到64位，修复超过2^31偏移的溢出。**对应长上下文与模型路径应优先做数值/竞态回归**；修复内核地址或同步边界，不代表这些长度已通过整模型容量或质量验证。

一手来源：[#68386](https://github.com/vllm-project/vllm/pull/68386)（`6bc4849a`，2026-11-19T14:46:32Z）；[#65241](https://github.com/vllm-project/vllm/pull/65241)（`0ae7032d`，2026-11-19T18:24:09Z）；[#68084](https://github.com/vllm-project/vllm/pull/68084)（`9e17d4a9`，2026-11-19T20:26:37Z）；[#50775](https://github.com/sgl-project/sglang/pull/50775)（`2e80690d`，2026-11-19T04:34:38Z）。

### SGLang：UniBoost 用有限提权照顾短 prefill，并保持多 rank 排序一致

`--schedule-policy uniboost` 对未命中缓存的 prefill 工作量给予有限boost，在偏向短请求时保留长请求的等待进度；gamma 大趋近FCFS、小更偏SJF，可选gamma-Ada按p95/p99尾部调节。TP/PP 用已处理prefill token/step时钟替代各rank墙钟，避免选出不同batch。PP>1的PD-prefill拒绝自动gamma，防止stage transfer polling差异使控制器分歧。**建议在长短混合RL rollout负载复现尾延迟、公平性和吞吐**；这是新可选策略，未有本轮独立服务压测。

一手来源：[#35637](https://github.com/sgl-project/sglang/pull/35637)（`99e4233f`，2026-11-19T22:46:40Z）。

### SGLang router：decode rank 可独立按预期 KV 占用选择

`--decode-dp-rank-policy` 可与prefill独立设置；`least_tokens` 从两随机rank比较当前KV改为选择预期持有KV最少的rank，结合engine load feed和在途请求。routing key/已缓存prefix先决定的rank仍有自己的约束。**建议检查DP负载上报、待到达KV和请求长度估计**，该策略改善决策输入，但没有同一负载下端到端优于旧策略的独立证据。

一手来源：[#51415](https://github.com/sgl-project/sglang/pull/51415)（`be322a83`，2026-11-19T20:29:39Z）。

### SGLang：Inkling-2-Small 接通混合注意力、MTP 和缓存数据面

Inkling2 主模型支持与几项独立基础改动一起落地：每层FA覆盖与block-anchored窗口；MXFP8 KV的四个UE8M0/head scale放入unified token envelope；DSpark draft在共享target FULL id空间且无需write translation时可把verify commit折入静态图；hybrid-SWA sidecar draft pools随target KV/SWA page一同PD传输或host restore。

不支持的pool类型保留原行为，draft状态索引与target页的对应关系仍须成立，模型支持不等价于任意后端/硬件组合都已验证。**建议深入阅读整套缓存契约并考虑模型级复现**；本轮未重跑模型精度、acceptance或吞吐，不能把拆出的基础PR分别算作完整模型性能成果。

一手来源：[#50186](https://github.com/sgl-project/sglang/pull/50186)（`6515ac29`，2026-11-19T15:35:21Z）；[#51297](https://github.com/sgl-project/sglang/pull/51297)（`91c88c66`，2026-11-19T07:16:08Z）；[#51298](https://github.com/sgl-project/sglang/pull/51298)（`21be2189`，2026-11-19T07:15:38Z）；[#51300](https://github.com/sgl-project/sglang/pull/51300)（`ecc5ab3a`，2026-11-19T07:14:48Z）；[#51296](https://github.com/sgl-project/sglang/pull/51296)（`87618387`，2026-11-19T07:17:12Z）。

### SGLang：DCP draft 随 target 打包；多 tokenizer 排空后才停 scheduler

DCP1→DCPN transfer 在draft已collapse且有pack buffer时，把draft行放进target的strided row打包路径，减少单独发送；仍检查目标token宽度，不能任意混用MLA与非MLA布局。shutdown路径则等所有tokenizer worker完成drain才停止scheduler，避免先停的一端使其他worker仍有请求却无执行者。**建议回放异构DCP、pending request与优雅退出场景**，两项是数据搬运与活性边界，未给统一加速数字。

一手来源：[#51455](https://github.com/sgl-project/sglang/pull/51455)（`03fde9cc`，2026-11-19T23:54:18Z）；[#50518](https://github.com/sgl-project/sglang/pull/50518)（`eae4bcb7`，2026-11-19T04:33:00Z）。

### FlashInfer：batch-invariant GDN 增训练式 stateless forward 和分块边界状态

不传 `initial_state` 的 `chunk_gated_delta_rule(batch_invariant=True)` 现表示所有packed整序列从零态出发，行结果与同类prefill/decode一致；可选 `train_states` 在同一launch写每64token块的起始FP32状态，供backward重算块内中间量。作者核对其buffer能交给cuDNN GDN backward，但不表示FlashInfer自身提供了完整autograd训练栈。**建议考虑复现训推行级一致性**；decode误漏state pool会被解释为每token重新开始，调用方须维护state契约，训练状态也增加显存。

一手来源：[#9677](https://github.com/flashinfer-ai/flashinfer/pull/9677)（`f6290475`，2026-11-20T00:16:22Z）。

### FlashInfer：sampling 的 batch invariant 依赖每行随机键，速度存在明显退化区

top-k/top-p/min-p及组合sampling可显式 `batch_invariant=True`，按行值、过滤参数、seed/offset决定输出，不再把batch slot混进Philox流；配套 `log_probs` 可计算同一kept-set分布下训练token的概率。默认 deterministic 仅保证同布局run-to-run，不保证换行/换batch不变。

作者在SM100/107冷L2、CUPTI、不用CUDA graph的分词表/批量微基准中，同时报告decode改善与大batch top-p明显退化（SM107 FP32 prefill geomean仅默认速度的0.17倍）。**建议为可复现RL选择性采用并量真实批量**，不能引用所有条目中最好的倍数代表采样整体。

一手来源：[#9619](https://github.com/flashinfer-ai/flashinfer/pull/9619)（`855eece4`，2026-11-19T04:24:24Z）。

### FlashInfer：GDP 的 FP16 state pool 省一半状态字节，随机舍入不一定更快

SM100/103的Cake Gated DeltaProduct decode增加FP16 pool：state载入后仍在FP32递推，每次store转FP16；默认最近偶数，传int64 CUDA `rand_seed`后用Philox和`cvt.rs`随机舍入。覆盖每token1–4次Householder更新、K128/V64，FP32路径和SM90 prefill不变。

pool容量及读写字节减半不等于吞吐翻倍。作者B200/GB300冷L2测量中，10轮随机舍入final-state geomean分别0.954/0.961倍，略慢于FP32；snapshot路径才有不同收益。**考虑在状态容量受限场景复现长序列误差与性能**，不将RNE和随机舍入的最好结果混用。

一手来源：[#9720](https://github.com/flashinfer-ai/flashinfer/pull/9720)（`a75cd12e`，2026-11-19T14:07:32Z）。

### FlashInfer：专家迁移暴露的是 prepared weight 别名，发布顺序仍由 engine 负责

`MoEEpMegaLayer.get_transformed_weights()` 返回backend运行布局的可写alias，避免engine只迁原始权重却漏掉预处理后的weight/scale。in-place更新可保持CUDA graph指针，销毁后访问会拒绝；调用方必须一起移动所有per-expert张量/normalization state，先完成旧读与传输，再发布新物理route，地址、slot容量与EP成员保持不变。**建议深入阅读迁移事务边界**；EP2/4/8资格验证早于最后accessor整理，完整EP16/32矩阵未在最终head重跑，不能宣称宽EP全覆盖。

一手来源：[#9621](https://github.com/flashinfer-ai/flashinfer/pull/9621)（`61a1f501`，2026-11-19T11:07:07Z）。

### FlashInfer：VC-Attention 修正预处理边界，并接 B200 INT8 Q/K 配方

VC prepass修复尾tile越界V gather，把k-means随机源与全局generator分开，cold start重置且batch/head共享初始行；还可只对指定token区间分组，避免joint text/audio/video被错误混入聚类。另SM100接受Hadamard后Q/K的INT8，按128token块缩放，V仍默认E4M3，输出BF16；B300保留E4M3配方。**建议对joint模态、奇数尾长及batch重排做数值回归**；作者B200测试与B300 sanitizer结果只覆盖所列路径，未给本轮独立视频质量/端到端结论。

一手来源：[#9549](https://github.com/flashinfer-ai/flashinfer/pull/9549)（`88fb283c`，2026-11-20T00:38:42Z）；[#8551](https://github.com/flashinfer-ai/flashinfer/pull/8551)（`6c41a72b`，2026-11-19T22:50:42Z）。

### FlashAttention：SM100 hdim64 backward 多用16KiB共享内存换访存布局

direct dK/dV store路径现把sdO分配到FP32 dV staging同样大小，令LSE/dPsum/dQaccum落到B200共享内存128KiB线以上，避开UMMA operand读的一半。作者在指定B200固定频率的hdim64 dense/varlen/deterministic形状报告约9%–14%局部提升；代价是16KiB更多SMEM。hdim96和2-CTA路径不在同一改动范围。**建议在匹配shape复现**，这是SMEM placement优化，不是算术量下降，也不能当所有FA4 backward提升。

一手来源：[#3065](https://github.com/Dao-AILab/flash-attention/pull/3065)（`dbabcc78`，2026-11-20T00:43:49Z）。

### Triton：统一在 CFG 上求 barrier、TMEM wait、async fence 和 cluster barrier

Membar把SMEM/TMEM依赖建成控制流图上的排序约束，用带cost的directed-multicut启发式找可行插入点；loop深度提高代价，以减少热循环中的同步。async proxy fence可跨warp-specialized roles与reachable noinline callee覆盖多个consumer，但必须在异步copy完成后才满足TMA读；cluster barrier也复用同一物理footprint、CTA routing和solver，分别处理scratch重用、mbarrier初始化与kernel退出。

这是编译器同步放置架构变化，启发式不保证全局最优；减少静态barrier数也不自动意味着kernel更快。**建议深入阅读并对自定义异步kernel做sanitizer和数值回归**，尤其跨CTA子视图、noinline与warp specialization组合。

一手来源：[#12752](https://github.com/triton-lang/triton/pull/12752)（`c421e6db`，2026-11-19T18:25:30Z）；[#12811](https://github.com/triton-lang/triton/pull/12811)（`b13d89fa`，2026-11-19T18:25:33Z）；[#12827](https://github.com/triton-lang/triton/pull/12827)（`7a4b3e27`，2026-11-19T18:25:35Z）。

### Triton：two-CTA 完成槽复用与循环后 accumulator reset 增守卫

无可pipeline的TMA operand load时，two-CTA MMA等待结束后增加cluster barrier，避免leader重用完成槽而peer还未观察旧完成，导致parity wait挂起；同步lowering同样补齐。accumulator-init优化则保留循环后仍被使用的条件zero reset，防止最后store误写前一次MMA结果。**建议升级前回归这两类最小kernel**，一项修挂起、一项修静默错误，不能靠一般matmul吞吐测试替代。

一手来源：[#12760](https://github.com/triton-lang/triton/pull/12760)（`cd16072f`，2026-11-19T03:44:57Z）；[#12798](https://github.com/triton-lang/triton/pull/12798)（`22459e3d`，2026-11-19T04:18:17Z）。

### Transformers：assisted decoding 可回滚 static cache，但滑窗不能恢复已丢状态

StaticLayer的crop通过原位回退write offset实现，保留编译依赖的tensor地址；辅助解码先prefill再检查 `is_croppable`，静态indexer层同时回退第二个offset。verification走可编译call，并用decode优化路径选择合适MoE算子，assistant draft仍保持自己的变长cache/执行。静态滑窗仅在能容纳整段generation时可crop；linear-attention past recording与graph的别名约束仍适用。**建议对拒绝draft、滑窗容量和混合模型回归**，支持static cache不等于所有assistant或所有生成长度都可CUDA-graph。

一手来源：[#49415](https://github.com/huggingface/transformers/pull/49415)（`c722b34a`，2026-11-19T13:44:13Z）。

### Transformers：packed isolation 在 blockwise overlay 后最后施加

blockwise mask的OR overlay可能重新放开不同packed sample之间的attention。`masking_utils.py`现把sample隔离AND放到blockwise组合之后，测试覆盖不同样本重复block id与offset的情形。**训练packed序列应优先做跨样本泄漏回归**，尤其采用block mask overlay时；这是mask正确性修复，未给训练质量/性能提升测量。

一手来源：[#50415](https://github.com/huggingface/transformers/pull/50415)（`de05253c`，2026-11-19T09:57:00Z）。

### Transformers：Dory 原生循环缓存；EXAONE/SarvamMLA 复用现有架构类

Dory支持将released block布局映射为decoder layer，循环cache按loop保存历史并支持last-loop只读路径，避免把共享层所有循环混成一次cache。EXAONE3.0/3.5/Deep仅增config/字段和权重名转换，Auto类直接复用Llama；SarvamMLA最终也复用DeepSeekV3模型类并规范Hub rope/config，不是新增一套独立MLA算子。**建议按工件做加载—保存—分块decode回归**；这是框架接入，不按当天新模型首发报道，也不把早期PR描述中后来删除的实现/测试数当最终内容。

一手来源：[#50099](https://github.com/huggingface/transformers/pull/50099)（`30e208b1`，2026-11-19T16:04:20Z）；[#50966](https://github.com/huggingface/transformers/pull/50966)（`c52179d2`，2026-11-19T14:44:52Z）；[#44569](https://github.com/huggingface/transformers/pull/44569)（`166948d9`，2026-11-19T14:28:29Z）。

### TRL：LFM2 tree packing 需要 Transformers5.22 的卷积 kwargs 透传

AsyncGRPO将tree recurrent patch从仅GDN扩为短卷积，LFM2及MoE可进入tree-packed训练；卷积patch复用causal_conv1d接口，而GDN专用更新只在存在时接入。启动时要求Transformers>=5.22，因为旧版不向被patch卷积路径传kwargs；测试增加packed梯度对照及旧版拒绝。**考虑在共享prefix多采样训练中复现**，没有同一批量下的本轮速度测量，也不代表所有含recurrent层模型都支持。

一手来源：[#7966](https://github.com/huggingface/trl/pull/7966)（`9f8725f7`，2026-11-19T10:59:01Z）。

### Miles：shared-outer expert LoRA 扩到 Qwen MoE/GLM，恢复时继承权重版本

native raw-mode LoRA为Qwen3.5/3.6 MoE与GLM-5.3-Flash routed experts注册shared-outer布局，与dense projection/lm_head多LoRA范围分开；它不是“所有expert adapter任意multi-LoRA”。checkpoint另保存并恢复LoRA weight version，使resume后继续该版本序列，避免训推同步版本回到初始值。**建议对导出、共享因子和resume后第一轮refit做集成回归**；adapter参数化改变不保证同质量，主线扩覆盖也不等于所有后端同样支持。

一手来源：[#4824](https://github.com/radixark/miles/pull/4824)（`ceae9c5f`，2026-11-19T23:51:01Z）；[#4826](https://github.com/radixark/miles/pull/4826)（`f205db8c`，2026-11-20T00:08:29Z）；[#4810](https://github.com/radixark/miles/pull/4810)（`1cc3d62e`，2026-11-19T23:50:06Z）。

### Miles：FP32 master→BF16 随机舍入可让小更新进入实际训练/rollout权重

RL小学习率更新可能小于半个BF16间隔；显式环境开关使每optimizer step将FP32 master随机舍入到BF16，训练和rollout使用同一份舍入权重。Philox键依赖seed、step和parameter shard，支持rematerialize与resume重建相同权重；仅BF16+distributed optimizer，precision-aware optimizer若不CPU offload会被拒绝，以免TE自己掌管master转换。**建议考虑复现小更新存活率与训推一致性**，未核独立长期reward/收敛收益，换sharding也不能自动继承同随机轨迹。

一手来源：[#4879](https://github.com/radixark/miles/pull/4879)（`47d67e2d`，2026-11-19T23:00:34Z）。

### Dynamo：在线预测输出长度；session gate 避免一个忙会话堵住其他会话

output-length插件按在线反馈预测reply长度，并把短reply优先级传给backend；它是显式router-policy选择，预测误差与实际backend优先级语义仍需验证。dispatch gate/session lanes则让相同session在自己的lane等待；忙rank只hold该session后续turn，其他共享heap/lane仍可仲裁。**建议对多Agent长短回合、取消与会话亲和复现**，不要把优先级设置等同于端到端公平性或吞吐保证；本轮未重跑博客的服务benchmark。

一手来源：[#18940](https://github.com/ai-dynamo/dynamo/pull/18940)（`672f84f9`，2026-11-19T01:26:50Z）；[#18443](https://github.com/ai-dynamo/dynamo/pull/18443)（`51ae4a04`，2026-11-19T14:27:48Z）。

### Dynamo：cache 命中信用按真正可装入的连续页计算

SGLang HiCache共享命中只计本地GPU/host/disk prefix之后、直到首个缺页的连续run，低于prefetch_threshold不计，LoRA页不享共享credit；旧worker未发布threshold时按256token默认。这样router不会为engine实际上不加载的离散页减少prefill估计。prefill tracker另修复乱序完成时anchor移动吞掉已预约工作量。**建议对缺页、阈值、LoRA与乱序完成回放**；都是负载估计正确性，未证明任意缓存命中率会增加。

一手来源：[#18176](https://github.com/ai-dynamo/dynamo/pull/18176)（`d364ed9d`，2026-11-19T23:36:46Z）；[#19113](https://github.com/ai-dynamo/dynamo/pull/19113)（`4dd50cbe`，2026-11-19T19:09:56Z）。

### LMCache：P2P 写策略可把后续 KV 放 peer，恢复进度须各 rank 一起重置

P2P store policy与write开关给L1压力下向headless peer写入KV的部署路径，并添加“早期留本地、后期到peer、replay仍命中”的验证工作负载。SGLang recovery callback则只标记reset，再经 `ready_prefix_count()` 现有MIN collective让所有TP/PP rank一起清store进度，避免单rank恢复后其余rank跳过本应重试的range。**建议按peer故障/恢复做一致性验证**；多级存储可配置不等价于自动多副本耐久保证，未做独立网络压测。

一手来源：[#6624](https://github.com/LMCache/LMCache/pull/6624)（`6ed33c9b`，2026-11-19T02:42:22Z）；[#6659](https://github.com/LMCache/LMCache/pull/6659)（`07f1c6bb`，2026-11-19T04:20:25Z）。

### NeMo RL：refit 可拆 Actor 并与初始化重叠；误推的 SSD 默认功能已回滚

single-controller可选将weight synchronizer置于独立Ray Actor，refit初始化与actor pools建立并行；setup用daemon thread承载可能阻塞的rendezvous，失败后不让executor join阻止进程退出。**建议在恢复/初始化失败场景复现**，没有本轮独立启动时长或训练吞吐结果。

需保留反向证据：同日main短暂误推的21项data-plane提交已由#5302整体撤回，包含默认SSD层、NVMe目录、若干I/O与recovery改动；**不能把中间commit列表当最终可用能力**。此前与SSD有关但未被本轮逐项审过的实现仍需按固定head确认。

一手来源：[#5182](https://github.com/NVIDIA-NeMo/RL/pull/5182)（`bf50129f`，2026-11-19T18:26:26Z）；[#5154](https://github.com/NVIDIA-NeMo/RL/pull/5154)（`492e1b7c`，2026-11-19T20:18:54Z）；[#5302](https://github.com/NVIDIA-NeMo/RL/pull/5302)（`047d885a`，2026-11-19T06:02:45Z）。

### Megatron-Bridge：online refit 流带上校准后的 KV scales

标准QKV映射层的ModelOpt streaming export支持对称FP8或NVFP4 K/V quantizer，每组输出HF attention prefix下当前q/k/v scale。FP8要求静态per-tensor E4M3；NVFP4是block16 E2M1、动态E4M3 block scale与静态global scale。Q scale跨TP取max，K/V必须符合相应契约；GPT-OSS原生MXFP4容不下KV metadata，组合被拒绝，rotation/pre-quant scaling也不支持。**建议对refit后scale与权重一起更新做数值回归**，不可把输出metadata当下游任意engine已验证。

一手来源：[#6726](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6726)（`769652a0`，2026-11-19T17:50:35Z）。

### PyTorch：pipeline 完成一个 microbatch 就释放其 loss 图引用

`_maybe_get_loss`将内部报表列表中的loss换成detach值，同时把原始带图loss交给当前backward，避免已消费microbatch图继续持有大的stage input。作者64microbatch Kimi-K3 rank-local复现峰值147.95→120.38GiB，另有单GPU双本地stage最小复现展示随microbatch增加的线性滞留消失。**建议在大microbatch流水训练优先回归峰值和梯度**；这些是作者限定复现，不是256GPU全系统统一节省量。

一手来源：[#199574](https://github.com/pytorch/pytorch/pull/199574)（`a1e7570a`，2026-11-19T06:05:54Z）。

### PyTorch：DCP 后台写失败必须阻止成功 metadata 发布

FileSystemWriter原来在后台thread抛I/O异常后，join仍正常返回，主线程可能把残缺WriteResult当成功并调用finish发布metadata，直到restore才发现缺项。现在后台异常传播到保存调用方，并有注入写失败测试。**检查多线程DCP checkpoint作业应优先回归**；保存future成功必须代表全部计划写完成，不能以文件目录或metadata存在认定可恢复。

一手来源：[#191899](https://github.com/pytorch/pytorch/pull/191899)（`5c82225c`，2026-11-19T19:25:31Z）。

### PyTorch：pinned allocator 先摘块后解锁释放，避免跨 GPU host copy 串堵

cudaFreeHost可能等待当前GPU队列，原empty_cache持allocator锁释放会让同进程另一GPU的non_blocking pinned copy也等同一长kernel。现锁内仅从表摘除可释放block，锁外free_block，保留原可释放性判定。**建议在单进程多GPU和host offload负载回归**；消除锁放大不消除cudaFreeHost自身同步，也没有任意训练吞吐必增的保证。

一手来源：[#200733](https://github.com/pytorch/pytorch/pull/200733)（`aba73527`，2026-11-19T23:22:58Z）。

### Megatron Core 0.20.1：回补 MoK、MFSDP v2 EP checkpoint 和 pipeline 修复

正式release在core_r0.20.0维护线上发布，固定tag commit。发布说明列出Mixture-of-Kittens backend、TransformerEngine2.20→2.21，以及MFSDP v2 EP恢复、schedule-node activation ownership、pipeline input storage、embedding main_grad和MLA RoPE量化等回补。**建议维护线用户按EP恢复/pipeline路径回归后考虑升级**；这里是发行聚合，不把已有主线PR重新算新算法，也不假设本日main新增内容均进该tag。tag尖端patch只更新文档版本/指针，功能范围来自官方release回补清单，未逐一复跑所有backport。

一手来源：[官方release](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.20.1)（2026-11-19T21:26:28Z，tag @ `f7aa73b2d9cdda75421be7067d221a5cb90e0f43`）。

### Megatron-Bridge 0.7.1：NCCL-EP 配方与 MFSDP v2 MoE 恢复进入维护发行

官方发行聚合B200/B300/GB200/VR相关MoE NCCL-EP配方、grouped tensors、Qwen3-VL padding/graph约束与MFSDP v2 MoE checkpoint resume等回补，并固定所需框架依赖。**建议使用对应维护镜像/recipe的团队评估升级**，核对容器与TE补丁组合；配方更新不是所有模型端到端加速证据。tag尖端patch为版本文档指针，本文功能范围按release清单，未独立测量所链接性能表。

一手来源：[官方release](https://github.com/NVIDIA-NeMo/Megatron-Bridge/releases/tag/v0.7.1)（2026-11-20T00:46:37Z，tag @ `e5e4889a28b996f731f18128080a5f67bb4ba21e`）。

### PEFT 0.22.0：PaCA/LoRA 初始化与 TP/FSDP2 集成一起发布

新增只训练选中输入列的PaCA（仅nn.Linear）、Astra激活协方差尾部特征方向初始化、NoRA归一化初始化，并改进TP replicated adapter初值同步、embedding_colwise、FSDP2/HSDP导出与merge/unmerge。并行改进要求至少Transformers5.17；sharded adapter导出需所有参与rank调用save_pretrained，只有写文件rank设is_main_process=True，不能仅主rank进入collective。

**建议深入阅读并优先回归已有并行LoRA流程**；release引用PaCA论文的时间/显存数字不是本版本所有工作负载实测，本轮不沿用其最好数字。tag固定patch同时含LoRA-FA参数兼容变更和旧Transformers弃用提示，升级须核调用方式。

一手来源：[官方release](https://github.com/huggingface/peft/releases/tag/v0.22.0)（2026-11-19T15:30:52Z，tag @ `0ea6d9656692aacf3f2e996cb156bcdfeaa63963`）。

### TRL 1.18.2：发行仅含冻结参数 BEMA 与 Muse 模板修复

本次正式patch release清单只有冻结部分parameter时BEMACallback的修复与Muse Glimmer JSON response-format模板识别。**相应用户可考虑升级并回归callback/模板**；上述AsyncGRPO活性与LFM tree packing按main合入独立报道，不能因为同日release存在便推定都在1.18.2。

一手来源：[官方release](https://github.com/huggingface/trl/releases/tag/v1.18.2)（2026-11-19T17:05:22Z，tag @ `c7caa02b6e162034d5736939cab49ebda4b4a080`）。

<a id="topic-6"></a>
## 硬件与软件协同

### TorchTPU：交叉熵改写同时消除 log-prob 临时量与特定 reshape 编译瓶颈

Google 11月19日把 class-index cross_entropy 的目标项选择从 gather 改为 masked sum，令 log_softmax 可以融合进归约，而无需把整张 log-prob 写入HBM。作者在TPU v7、PyTorch2.13.0+cpu、libtpu0.0.50、BF16 (24576,151936)上报告：调用内存增量7.47GB→0.4MB，缓存no-grad时间14.29→11.94ms，前反向31.48→28.98ms；切片展平、B2/V151936/S2048的首次调用87→1.9秒。比较基线与测试配置见提交正文。每样本loss保持一致，部分FP32总归约相差1–2ULP。**建议深入阅读并复现**大词表loss；这是lowering绕开物化的改进，未修libtpu通用reshape编译问题，nll_loss及其他需要物化的路径仍可能慢。

**日期/版本：** UTC提交日 2026-11-19；固定commit均进入默认分支，未据此确认正式包已包含。 [google-pytorch/torch_tpu @b033ad1468dc](https://github.com/google-pytorch/torch_tpu/commit/b033ad1468dc39e5c798502cc66683d773f253a0)

### TorchTPU：fused barrier 必须真正等待其他 rank

11月19日提交为barrier的dummy collective引入fence operand，并通过MakeCollectiveWork追踪完成，避免XLA把常量输入的barrier all-reduce折叠而提前结束。补丁新增一个rank延迟入场、其他rank必须等待的同步/异步及融合用例。**考虑采用并做故障边界回归**；本轮只核差分和测试，未在TPU多机独立复验，不能把这项同步保证等同于任意collective都自动成为完整设备栅栏。

**日期/版本：** UTC提交日 2026-11-19；固定commit均进入默认分支，未据此确认正式包已包含。 [google-pytorch/torch_tpu @93558d359295](https://github.com/google-pytorch/torch_tpu/commit/93558d3592958da70a24ad34adeedd92e69bdb3d)

### TPU inference：共享专家不能被重复 all-reduce，投机需匹配查询宽度

11月19日RowParallelLinear把defer_all_reduce与可在构造后修改的reduce_results保持同步：Transformers backend将共享专家down_proj设为不归约时，旧值可能让本层和MoE runner各归约一次。含bias但不skip_bias_add的延后归约会拒绝。另将batched RPA的decode_query_size设为投机宽度（例如3草稿对应4查询），并把bonus_logits转FP32，配套编译和runner用例。**考虑采用**，尤其检查共享/路由专家合并与bonus采样数值，未提供通用吞吐数据。

**日期/版本：** UTC提交日 2026-11-19；固定commit均进入默认分支，未据此确认正式包已包含。 [vllm-project/tpu-inference @80cfbb7697b5](https://github.com/vllm-project/tpu-inference/commit/80cfbb7697b5cb4484ec1e7681d8ee9bc38db917) · [vllm-project/tpu-inference @28147d54f2bc](https://github.com/vllm-project/tpu-inference/commit/28147d54f2bcee2dfa9736a5a01de8e04466f362)

### SGLang TorchTPU：DSpark 与有界重放共同改变 SWA 存储

11月19日系列提交把DeepSeek-V4.1 DSpark接入CSA2 TARGET_VERIFY及非因果草稿block，草稿复用目标embedding/lm_head；CED把前缀命中最后至多128个token折入extend，重放行只写SWA、不重新发布main KV/index或压缩pair状态。目标与草稿分别使用计价的request-window/ring，草稿每请求4×64-token页，限制valid_from防止读到槽前任内容；verify在设备侧选择提交行，处理host状态落后一个commit的问题。Host Engram快速路径随之支持部分折叠重放/非混合投机批次，不支持的边界回退，verify要求统一宽度。**深入阅读后考虑复现**：必须同时检查重放位置、草稿acceptance、页容量与释放；不能把四项当互不相关的速度开关，也未独立验证长上下文端到端性能。

**日期/版本：** UTC提交日 2026-11-19, 2026-11-20；固定commit均进入默认分支，未据此确认正式包已包含。 [sgl-project/sglang-torchtpu @3f6cb3b7601c](https://github.com/sgl-project/sglang-torchtpu/commit/3f6cb3b7601cd1938857316eee5c465071dda6d5) · [sgl-project/sglang-torchtpu @128629c5c4b6](https://github.com/sgl-project/sglang-torchtpu/commit/128629c5c4b66631173790cc8cfcba7227317915) · [sgl-project/sglang-torchtpu @2958de83c7e2](https://github.com/sgl-project/sglang-torchtpu/commit/2958de83c7e2dd1e33c3504d55b9cef2fb774428) · [sgl-project/sglang-torchtpu @78357c79e5cc](https://github.com/sgl-project/sglang-torchtpu/commit/78357c79e5cc2c4bb87af2cf8f5443fc0e4d239d)

### SGLang TorchTPU：KDA verify 的 VMEM 预算与小通信选择

GLM-5.3-Flash attention-DP8、每rank64 KDA heads、4-token verify的最小tile旧估约61MiB、超57.6MiB VMEM；新路径在双缓冲不适用时单缓冲dense状态输出并立即等待copy-out，已有可编译配置保留原tile/双缓冲。另一提交在单host、TP等于全world、满足BF16形状/容量门槛时选择two-shot Pallas all-reduce；作者tpu7x 2×2×1 TP8、关闭SparseCore offload的64–512KiB实验17–25µs对XLA19–28µs，约768KiB持平、1MiB起XLA更快。**考虑复现**：前者解决可运行性，后者仅特定小消息，不据内核数字推断全服务加速。

**日期/版本：** UTC提交日 2026-11-19；固定commit均进入默认分支，未据此确认正式包已包含。 [sgl-project/sglang-torchtpu @9c108f33e3d9](https://github.com/sgl-project/sglang-torchtpu/commit/9c108f33e3d9a69c8acc64581bf9828148dcc937) · [sgl-project/sglang-torchtpu @634c8a589cea](https://github.com/sgl-project/sglang-torchtpu/commit/634c8a589cea10a6cd1bee6a26f349df8cbcd856)

### SGLang TorchTPU：HiCache 页释放需覆盖流水并行，回收策略以最终补丁为准

释放fence把PP中仍可能在途的forward纳入判断，并令HiCache备份/页回收遵循该顺序；统一PP+投机解码会明确要求禁用HiCache；P/D prefill的投机写可由launch event覆盖，PP的P/D decode循环尚未驱动同样fence hooks。共享Full/SWA host arena新增SWA份额上限和Full优先回收；同一PR正文曾提出结构化树回收，但最终提交已撤回这部分，只保留share cap，不能按中途设计宣称已支持。**建议深入阅读并回归**多stage回收和混合host压力；无本轮设备故障注入。

**日期/版本：** UTC提交日 2026-11-19；固定commit均进入默认分支，未据此确认正式包已包含。 [sgl-project/sglang-torchtpu @81bed75bc282](https://github.com/sgl-project/sglang-torchtpu/commit/81bed75bc28220532cd3b496fee048b4c53ba9ca) · [sgl-project/sglang-torchtpu @ff104c9a159c](https://github.com/sgl-project/sglang-torchtpu/commit/ff104c9a159c4a757bf5e40e814138e30eeb489b)

### TPU Sync：每 worker 的 KV 几何减少 PP 填充，失败传输释放有显式状态

11月19日KVTransferSpec可为各worker登记不同block arrays；host store按真实数组求和计价，作者4-stage×8-worker prefill带草稿配置称约多30%block容量。num_kv_shards和dense node IDs仍共用；旧registry/store遇新spec会拒绝，部署应先升级registry/store，再让serving host停止padding。失败StartTransfer回滚另默认关闭（TPU_RAIDEN_ROLLBACK_FAILED_TRANSFERS=1启用），取消已准备receiver，只有没有push/copy仍在途时报告释放、由torch端抛TransferReleasedError供上层安全回收。**考虑复现**；报错不自动代表目标页已释放，更不能把所有失败都当可立即重用。

**日期/版本：** UTC提交日 2026-11-19；固定commit均进入默认分支，未据此确认正式包已包含。 [google/tpu-sync @97d0fd925b4a](https://github.com/google/tpu-sync/commit/97d0fd925b4ada4603d542c67840bd4411d53c1b) · [google/tpu-sync @0768591d9658](https://github.com/google/tpu-sync/commit/0768591d9658b6fd04272de2d7461781207f2e0e) · [google/tpu-sync @992c26024576](https://github.com/google/tpu-sync/commit/992c26024576acaf9ec7074b0dd4a930639497f9)

### TensorRT-LLM：raw logprobs 的口径恢复到采样惩罚之前

11月19日修复只为请求raw且会原地改写logits的行保存/恢复原值。单beam RAW排除embedding bias、repetition/presence/frequency penalties、禁词/no-repeat/min_length、temperature和top-k/p；guided decoding与logits processors仍会作用，beam>1仍来自beam history，不能称绝对未经处理的模型概率。另修正DSpark图pad_batch未传sampling_mode，增加置信度调度GSM8K集成用例。**优先考虑采用并回归**RL rollout的训练logprob、混合batch和投机采样；本轮没有运行这些GPU用例。

**日期/版本：** UTC提交日 2026-11-19；固定commit均进入默认分支，未据此确认正式包已包含。 [NVIDIA/TensorRT-LLM @ffed918b0831](https://github.com/NVIDIA/TensorRT-LLM/commit/ffed918b08313d69c1fff9dbb18b1d14531daa96) · [NVIDIA/TensorRT-LLM @8fe34e2336d7](https://github.com/NVIDIA/TensorRT-LLM/commit/8fe34e2336d70463c5ce37f2412eedf0640ac984)

### TensorRT-LLM：GMS 保留共享 storage 视图，UCX 回补虚假超时修复

GMS迁移改为每storage只复制一次，并按各parameter的dtype/shape/stride/storage_offset建立视图，保留重叠、tied和不同dtype视图；零字节storage、lazy conjugate/negative和越界映射明确拒绝，CPU storage测试只能验证迁移记账，不能代替真实GPU daemon。构建脚本另在固定UCX1.23.x回补timer-wheel补丁62182080，先推进时钟再执行callbacks以避免虚假UD peer timeout。**考虑采用**于共享权重/远程通信部署；需核实实际镜像含补丁，main提交不等于已有release全部包含。

**日期/版本：** UTC提交日 2026-11-19；固定commit均进入默认分支，未据此确认正式包已包含。 [NVIDIA/TensorRT-LLM @b963a7ab7565](https://github.com/NVIDIA/TensorRT-LLM/commit/b963a7ab75650e961b7eb4e7abb6a968e298117c) · [NVIDIA/TensorRT-LLM @ed5f9c0dd6c9](https://github.com/NVIDIA/TensorRT-LLM/commit/ed5f9c0dd6c900c4fcb349095b144dfa5d04d916)

### TensorRT-LLM：SM107 的 DeepSeek-V4.1 低精度路径按形状准入

11月19日增加SM107 CuTe DSL MXFP8 prefill GEMM、CSA2 projection和self-sampling GVR decode选择。以CSA2输出投影为例，仅SM107、TP1、hidden5120、输出8192等门槛才扩展backend候选，其余保留默认；同时新增linear/indexer/top-k测试。**继续观察并考虑聚焦复现**：补丁证明这些算子路径和选择逻辑已落地，未提供本轮核验到的等负载整网加速，不能据新硬件分支推断所有DSV4.1部署收益。

**日期/版本：** UTC提交日 2026-11-19；固定commit均进入默认分支，未据此确认正式包已包含。 [NVIDIA/TensorRT-LLM @d1819e9b2597](https://github.com/NVIDIA/TensorRT-LLM/commit/d1819e9b2597f15437a1ec0dcadeeba0d17acde1)

### MORI：AINIC RDMA READ 可静默缺数据，官方给出固件边界和规避

11月19日公开已知问题：Pollara AINIC 1.117.5-a-45之后的部分固件，多个READ在途且发生重传时，可能提前报告完成、数据晚到/缺失，或QP错误而没有failed work request；提交报告在a-77/a-196观察到、a-45未观察到。新增每进程首次READ告警与mori check提示，建议MORI_IO_QP_MAX_RD_ATOMIC=1降低每QP在途READ。该规避仅缓解风险：a-196测试未再见症状，a-77在高接收背压、多QP下仍可缺数据，作者无其他负载的吞吐约下降4%。**优先核对线上固件并考虑采用规避**；告警不是修好固件，也不保证任意后续固件都同样受影响，本轮未做网络故障注入。

**日期/版本：** UTC提交日 2026-11-19；固定commit均进入默认分支，未据此确认正式包已包含。 [ROCm/mori @1eb89cc391b0](https://github.com/ROCm/mori/commit/1eb89cc391b06622953f9b1ebe6e82987be68744)

### MORI：HBM 放置、SSD 读租约与 JIT arch 状态隔离

UMBP同优先级HBM池先选数据源GPU，跨GPU在peer access可用时用gather、否则hipMemcpy，减少按hash放置的无谓跨卡复制。远端读者完成报告可按代际释放SSD staging，不必只等lease计时；旧peer及失败RPC仍需按既有验证边界处理。JIT v2的显式arch参数变为调用作用域，不污染后续plan，并检查目标与设备兼容。**考虑采用/复现**缓存密集负载和跨架构预编译；这些机制没有可叠加的端到端收益数字。

**日期/版本：** UTC提交日 2026-11-19；固定commit均进入默认分支，未据此确认正式包已包含。 [ROCm/mori @ad78d79cc89a](https://github.com/ROCm/mori/commit/ad78d79cc89a79eee479cb07cec4557de3e7fc3d) · [ROCm/mori @11cb36a3c284](https://github.com/ROCm/mori/commit/11cb36a3c284703f5641e73c81d1d45e28ac4beb) · [ROCm/mori @099d4c708cf9](https://github.com/ROCm/mori/commit/099d4c708cf9a1af10801b85a44596b1e3fd56a6)

### AITER：all-reduce、RMSNorm 和 per-token FP8 融合，不能忽略销毁协议

11月19日gfx1250 Lamport/Gluon路径将AR+residual/RMSNorm+per-token FP8合到一次launch；按TP、hidden、payload、是否emit_bf16和128行分界选择。小hidden32/64/128另对齐HIP的amax floor和除法，零行尺度不能省略。两阶段AR移除末尾barrier、5→4 launches，改用epoch差值处理uint32回绕；因此本rank流排空后仍可能被peer读取，显式close先设备同步再有界host barrier，GC路径不发collective，peer中途死亡仍可能卡在设备同步。Ulysses intra-node A2A增加PCIe fine-grained IPC选择和全rank分配/关闭协议，默认仍uncached。**建议深入阅读并复现**数值、回绕、跨rank释放；“少一个barrier”不等于可以随时释放通信buffer。

**日期/版本：** UTC提交日 2026-11-19；固定commit均进入默认分支，未据此确认正式包已包含。 [ROCm/aiter @f7c728ab759a](https://github.com/ROCm/aiter/commit/f7c728ab759a8f6472a65e3fb2ef50862e9d6b56) · [ROCm/aiter @0ece88698e78](https://github.com/ROCm/aiter/commit/0ece88698e78263b38a707a8b2cfdf9bf3db83fa) · [ROCm/aiter @8a3bdd6ca74b](https://github.com/ROCm/aiter/commit/8a3bdd6ca74b69452a6fbf34dc58a73d9130de5f)

### ATOM：ReplaySSM checkpoint 必须是 verify 边界状态，避免多轮重复保存

11月19日decode checkpoint登记verify step，GDN/KDA ReplaySSM先在scratch中物化该step的SSM与conv图像再保存，不能直接复制已经前进的live slot；结束step不再保存失去意义的边界。LMCache prompt-end anchor另仅在PP保留chunk向下取整，处理其给下一轮留下的重复demand；单GPUQwen3.8-Flash-Next MXFP4、chunk256/block64、AgentX3600s/c48的作者对照报告吞吐约+4.8%，PP8 P/D未跑。**考虑采用并复现**多轮前缀重用的状态一致性，不能把这项单GPU收益外推到PP。

**日期/版本：** UTC提交日 2026-11-19；固定commit均进入默认分支，未据此确认正式包已包含。 [ROCm/ATOM @52a0cf79058a](https://github.com/ROCm/ATOM/commit/52a0cf79058a2c1f2f28b88cea9f0a3e5869f46e) · [ROCm/ATOM @5bb3fc2d3b2d](https://github.com/ROCm/ATOM/commit/5bb3fc2d3b2d75ec36e10f506c2727cb1fea3721)

### ATOM：草稿独立 KV 精度和 MI308X/低精度 indexer 支持有组合约束

独立KV pool的MHA草稿可选BF16/FP8，复用目标pool的MTP/MLA草稿必须同精度；与目标不同精度的KV connector/offload/P-D组合未经验证并拒绝。DSV4.1-Flash在gfx942增加FP4PTPC配方：加载时将路由专家转per-channel FP8，部分group32线性层仍BF16，不是原生FP4全路径。MiniMax-M3 NVFP4 indexer CP把本rank round-robin分片交给配套AITER scorer，default off，并依赖新版cp_world/cp_rank接口；不能只升级ATOM忽略AITER。**考虑聚焦复现**可装入性、数值和数据传输兼容，不把主分支支持等同完整产品矩阵。

**日期/版本：** UTC提交日 2026-11-19；固定commit均进入默认分支，未据此确认正式包已包含。 [ROCm/ATOM @0e6f1ac04aaa](https://github.com/ROCm/ATOM/commit/0e6f1ac04aaaa5dfaff101c17478094cf0adb5e3) · [ROCm/ATOM @765d5c45a707](https://github.com/ROCm/ATOM/commit/765d5c45a707cd9470a47874b187630fa88e3e60) · [ROCm/ATOM @f533f3feade1](https://github.com/ROCm/ATOM/commit/f533f3feade120801af7e77a5975a3280c8fcf73)

### FlyDSL：修复 traced return 被静默丢弃，并补齐 AOT 对象加载

11月19日AST重写修复if分支return只退出嵌套函数、导致外层继续返回错误值的问题。trace时已确定分支遵循Python；动态scf.if/for/while中的return改明确NotImplementedError并给位置。作者指出Kimi-K3 AttnRes的has_delta=True此前会忽略delta。flyc.load_module另可用LLVM ORC/JITLink进程内加载export_to_c对象、无需磁盘共享库/系统linker，但拒绝TLS、静态构造析构等平台runtime需求，HIP首个错误停止后续launch。**优先考虑正确性回归，再评估AOT采用**；并非支持任意Python控制流或任意ELF对象。

**日期/版本：** UTC提交日 2026-11-19；固定commit均进入默认分支，未据此确认正式包已包含。 [ROCm/FlyDSL @d2e7143cd712](https://github.com/ROCm/FlyDSL/commit/d2e7143cd71214a2937cd924f4dfc30bc56dfbb8) · [ROCm/FlyDSL @eb37caa92a23](https://github.com/ROCm/FlyDSL/commit/eb37caa92a232ba9b75da5c06431da3764687e82)

### Ascend：NPUGraph 的 while_loop 与编译缓存新增接入

11月19日NPUGraph dispatch捕获torch.while_loop：父图写首次predicate，body更新状态和条件，重放可以按当前输入执行0次或多次；与IF共用子图/stream/allocator管理，含3/0/5轮及嵌套IF测试。另一提交新增DVM compile cache：把producer身份和编译选项加入cache key，恢复graph-owned常量与别名，后端切换时重新注册codegen；身份不可持久化则绕过缓存，FX资源丢失则报兼容错误要求重建。**深入阅读并考虑复现**设备侧迭代/重复启动成本；本轮未测NPU硬件最低版本和复杂动态形状覆盖，不把基础接入扩成所有while/梯度变体均受支持。

**日期/版本：** UTC提交日 2026-11-19；固定commit均进入默认分支，未据此确认正式包已包含。 [Ascend/pytorch @6efa6d096cae](https://github.com/Ascend/pytorch/commit/6efa6d096cae9f160a535f5b6290a5e59d5f62df) · [Ascend/pytorch @f5688a84932d](https://github.com/Ascend/pytorch/commit/f5688a84932dbadc003e7c914293e92bc8ad5976)

### Ascend：Mooncake SWA 在 N−1 交接，idle DP 仍需加入 embedding TP

Mooncake P/D由decode重放最后prompt token，SWA若prefill计算到N可能提前回收decode仍需的页。新路径复用N−1 handoff并统一published length/remote-hit记账：prompt1023/window512/block128的测试强制复用空页，检查decode位置1022所需511..1021历史。embedding TP另修idle DP rank没有参与交换而其他rank等待的问题，保留上游persistent输入buffer用于图重放。**优先考虑采用并复现**长前缀P/D与负载不均场景；数据传输成功本身不足以证明窗口内容仍有效。

**日期/版本：** UTC提交日 2026-11-19；固定commit均进入默认分支，未据此确认正式包已包含。 [vllm-project/vllm-ascend @47b32942d6e7](https://github.com/vllm-project/vllm-ascend/commit/47b32942d6e74b757fcae48cd40c181e53c7e588) · [vllm-project/vllm-ascend @9408a39a017a](https://github.com/vllm-project/vllm-ascend/commit/9408a39a017a879e92c9d97fc466190caf498970)

### Ascend：FULL 图草稿元数据、A5 MLA 可见长度与 INT8 KPool

DSpark/DFlash（含DFlash2）FULL图padding后同步裁剪Q分段、KV lengths和block-table行，避免服务启动成功却在不足满batch时FIA收到N与N+1不一致。A5 packed MLA按当前请求真实可见KV、query end和replay floor准备每压缩率plan；共享仅限坐标/长度/floor一致，物理slot仍group-local。A3 GLM-5.3-Flash可选INT8 KPool，key/query配FP16 per-vector scales，key+scale共用受管理页以随P/D传输，独立于普通KV dtype且默认BF16；需要支持空pool请求的新ops-transformer。A5 MiniMax-M3 fused MsaIndexScore TopK则限topk为8倍数且≤128、需return_choice新接口，否则回退。**考虑分项复现**正确性及量化误差，未验证正式包最低版本及整网速度。

**日期/版本：** UTC提交日 2026-11-19；固定commit均进入默认分支，未据此确认正式包已包含。 [vllm-project/vllm-ascend @2f41be5d8f57](https://github.com/vllm-project/vllm-ascend/commit/2f41be5d8f579cab1cf6c1b09b1b1c55ccea8eb1) · [vllm-project/vllm-ascend @b69ebe54a5ce](https://github.com/vllm-project/vllm-ascend/commit/b69ebe54a5ce7f3b34da08633d6ce47bba2ae042) · [vllm-project/vllm-ascend @536f084ccb4b](https://github.com/vllm-project/vllm-ascend/commit/536f084ccb4b336bef7f4c0b8b2ab16d55b6af43) · [vllm-project/vllm-ascend @7110e02dbd3e](https://github.com/vllm-project/vllm-ascend/commit/7110e02dbd3eb66f0694c3948dc19924de94b731)

### CANN：MegaMhc 融合 Sinkhorn、Attention Residuals 扩展保留舍入边界

11月19日ops-transformer !16270把MegaMhc的Sinkhorn从预留变成实际支持，num_sinkhorn_iters仅0或20，20次固定迭代、无动态收敛提前退出；默认0仍返回logits。!15234为BlockAttnResUpdateV2增加可选输出RMSNorm及独立eps，保留混合输出先BF16舍入再RMSNorm、partial_block FP32原地累加语义；Prepare的block_res扩至BF16，query和在线softmax统计仍FP32，产品文档限定950PR/950DT。**深入阅读后考虑采用**，不能把融合写成改变了算法精度语义，也无本轮设备实测。

**日期/版本：** UTC提交日 2026-11-19；固定commit均进入默认分支，未据此确认正式包已包含。 [cann/ops-transformer @53ca24689039](https://gitcode.com/cann/ops-transformer/commits/detail/53ca24689039372ed471c3c98bdee19d957f8478) · [cann/ops-transformer @29e455827cb9](https://gitcode.com/cann/ops-transformer/commits/detail/29e455827cb9ff9d65dd226be7c6ab61061d4519)

### CANN/HCCL：E8M0 非连续 scale、接收 drain 与对称零拷贝边界

ops-nn !14562以UINT8独立视图搬运E8M0 scale字节后恢复dtype，避免通用contiguous下发不支持的kernel；原地累加yRef非连续则在workspace查询阶段拒绝，不能悄悄复制改变原地语义。HCCL !4111在batchsendrecv接收分支增加channel drain以避免持续NotifyRecord耗尽WQE，作者用跨超节点32P、1–16G数据量检查。!4005将AGV/Reduce接入对称内存零拷贝，AGV允许displs空洞/乱序但非空区间不得重叠；Reduce只探测输出，并限制确定性算法、dtype和PROD等组合。**考虑采用并做实际拓扑回归**；master API/依赖HCOMM需匹配，不能声称所有collective无条件零拷贝。

**日期/版本：** UTC提交日 2026-11-19；固定commit均进入默认分支，未据此确认正式包已包含。 [cann/ops-nn @d66efcc186c0](https://gitcode.com/cann/ops-nn/commits/detail/d66efcc186c044fdef7c7571e469723bf0f07ea0) · [cann/hccl @64713d433b73](https://gitcode.com/cann/hccl/commits/detail/64713d433b73f3c8abfc7f711eea7fe8c225084f) · [cann/hccl @a6370823eb1a](https://gitcode.com/cann/hccl/commits/detail/a6370823eb1a4a40e53caaa8e86d8d098dc26d87)

### MLXcel：INT8 MLA latent 真正省空间，但长窗口仍可能更慢

Lablup 11月19日将absorbed MLA缓存改为group64 affine INT8 latent加FP16 RoPE，典型512维latent每token/层672字节，对FP16的1152字节；短单query有Metal融合路径，较长或多query需组合/反量化。支持逐步扩到DeepSeekV3Attention系、Kimi Linear/K3和LongCat；KDA层不按MLA计价，GLM-4.7-Flash因质量门槛未过仍把INT8请求解析为FP16。作者M3 Ultra的LongCat-Flash-Lite在16K缓存294对504MiB，但decode较FP16在512长度+8%、2048−15.5%、16384−49.2%；Kimi K3未实测、560B LongCat及CUDA/ROCm未测。**考虑复现显存/速度折衷**，0个“decided disagreement”仅是在--decided 2及1024/4096前缀的短teacher-forced trace门槛下成立；仍有超过独立0.11-logit near-tie bar的top1 flip，不等于无损或任意任务质量相同。

**日期/版本：** UTC提交日 2026-11-19；固定commit均进入默认分支，未据此确认正式包已包含。 [lablup/mlxcel @43f1c8d46e54](https://github.com/lablup/mlxcel/commit/43f1c8d46e5499e7af2cc57b144234bd873b6114) · [lablup/mlxcel @a00629a623c5](https://github.com/lablup/mlxcel/commit/a00629a623c55a0e82fe6866e891d9bbfb833864) · [lablup/mlxcel @3f5679a407b0](https://github.com/lablup/mlxcel/commit/3f5679a407b001ebaa14f57a11aba85d2c9d49eb) · [lablup/mlxcel @c540d7994399](https://github.com/lablup/mlxcel/commit/c540d7994399e667be09c4916d5c68829be8cad0) · [lablup/mlxcel @7a85fe51310d](https://github.com/lablup/mlxcel/commit/7a85fe51310da33cf045bbb17357db5b5bfb3503)


### TorchTPU 与 TransformerEngine：运行边界和 backward 基础件

TPU host premapped buffer默认值按主机/容器有效内存与worker数计算：每worker<16GiB取0，其余min(4GiB,份额/16)再向下取2次幂；显式环境设置优先，适合检查多进程容器的host预算。flex_attention新增dK/dV backward内核及CPU参考、GQA、全mask行等测试；该提交本身不能证明完整dQ与高阶梯度均已接通。TransformerEngine cuBLASLt grouped GEMM在host检查输入/输出16字节对齐，使未padding梯度buffer中的错位view明确报错，而不拖成异步device fault。**考虑采用并回归**实际训练形状，不把新kernel和校验当未经测量的加速。

**日期/版本：** 2026-11-19 UTC提交，已入默认分支、未确认正式包包含。 [google-pytorch/torch_tpu @be151d1346d1](https://github.com/google-pytorch/torch_tpu/commit/be151d1346d1874671a10d5b05d1b9451555eff2) · [google-pytorch/torch_tpu @4ee4ba23f60c](https://github.com/google-pytorch/torch_tpu/commit/4ee4ba23f60cfd67ff07f9d7cb6caf5aeda6dd85) · [NVIDIA/TransformerEngine @db1aff638b07](https://github.com/NVIDIA/TransformerEngine/commit/db1aff638b070cb9d08a0dc91629ff1457ead413)

### MLXcel：compressed-tensors NVFP4 工件接入独立于缓存压缩

将nvfp4-pack-quantized文本checkpoint的weight_packed/weight_scale/weight_global_scale转换成既有ModelOpt三元组，随后复用重打包；冲突命名、错误全局scale等会拒绝，activation scale不作为运行时路径消费，Laguna保留自有expert sanitizer。含与ModelOpt fixture逐项对照的loader测试。**考虑采用并验证**具体导出工件；读得进权重不自动证明等同NVFP4激活计算，亦与INT8 MLA KV的精度/性能是两个问题。

**日期/版本：** 2026-11-19 UTC提交，已入默认分支、未确认正式包包含。 [lablup/mlxcel @a825a009c1be](https://github.com/lablup/mlxcel/commit/a825a009c1bea50ead7ce5b2bc8ce1c91262a0d1)

### Dynamo RL：把样本接受、权重发布与缓存失效留给训练框架

NVIDIA 11月19日发布RL基础设施技术说明，将服务划分为rollout请求、worker发现和定向管理三条平面。KV-aware routing平衡前缀命中与排队；ModelExpress复用集群GPU权重、增量传输及checkpoint+delta追赶以缩短更新/恢复，训练框架仍决定权重何时发布、KV如何失效和样本是否可接受。配套integration reference强调token IDs是训练权威，不能由输出文本重新tokenize，并要求逐worker检查路由/版本能力；发现接口不是全fleet事务。**建议深入阅读**作为verl/slime等接入设计参考。文中CoreWeave“最高15倍权重更新”、H Company“step最高−20%/GPU-hours−5%”未在该文给齐模型/集群/负载，故不据此排名，也不能将今日新文称为所有机制今日首次发布。

**机构/日期：** NVIDIA，2026-11-19。 [官方原文](https://developer.nvidia.com/blog/reducing-rollout-bottlenecks-in-reinforcement-learning-with-nvidia-dynamo/)

### Dynamo 工具解析：跨模型 GOLDEN 语料与逐切分流式验证

NVIDIA 11月19日方法文章介绍frontend-crates以有序reasoning/text/tool-call事件作为batch与stream共用边界；UNIFIED YAML语料覆盖截断、畸形、多调用、交错与EOS，先按模型语法独立审查GOLDEN再实现parser。未知工具名保持原生成、由serving层校验；finish用于截断/取消/传输结束，finish_at_eos只用于后端确认的grammar EOS。当前v2默认在引号文字内也识别完整结构marker，故引用工具格式示例可能被解析，必须按策略测试。文章示例固定Rust1.96.1/dynamo-parsers-v2=0.10.3；仓库当前读取HEAD为79179a03，0.x API和注册不自动代表后端已支持。**考虑采用语料方法并做真实模型端到端回归**；本轮未编译运行parser，纯fixture通过不证明服务完备，也无性能比较结论。

**机构/日期：** NVIDIA，2026-11-19。 [官方原文](https://developer.nvidia.com/blog/?p=125544)

<a id="topic-7"></a>
## 评估与可靠性

### Hugging Face harness-hub：pi／Hermes适配器更新评测，纠正“少思考同精度”的边界

- **时间版本**：pi卡11月19日08:01:45Z `80d05c9051dfd2ae464e1daf4329cb15524caf60`，与昨日末快照 `78eacecca691deb59f703fc270120e2e5b2e0194` 对比；Hermes为07:55:58Z `09fb7412411abeeb38230bca646f78ba5993b25b`。本项为评测及卡片修订，不能写成Qwen3.8基座模型首发。两张卡声明Apache-2.0；独立LICENSE文件404，不能据此声称已核完整许可文件。
- **变化与原因**：pi旧卡把SWE-bench Pro hard51的thinking基座写64.7%、适配器66.7%；新卡说明当时每次模型调用16K输出上限截断长思考，改32K重跑后**两者均74.5%**。no-thinking仍为基座39.2%、适配器64.7%。因此“不开thinking匹配最高强度thinking”的说法只在其Terminal-Bench设置附近成立，不能覆盖SWE-bench Pro。Hermes同日重新披露hard51无thinking由29.4%到43.1%，提升13.7点、p=0.092，不能称统计显著；其thinking-on很多session撞131072上下文上限。
- **条件**：文本模式、关闭视觉塔、每题一次、100-step预算、temperature0.8/top-p1.0；SWE仅hard51、2h，TB2.1为共同成功评分任务子集、1h。checkpoint用SWE及18题TB子集选取，属于选择偏差来源；pi卡提供去除选模题的68题TB分项。token节省来自作者轨迹，部分thinking服务计数缺失，卡用轨迹回数，不是独立复现。
- **意义／建议（推断）**：这次最值得保存的是输出上限与上下文预算改变结论的证据。后续比较agent RL适配器应锁定harness、预算及同一任务交集，逐次保存失败原因和轨迹。[pi固定卡](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted/blob/80d05c9051dfd2ae464e1daf4329cb15524caf60/README.md)；[Hermes固定卡](https://huggingface.co/harness-hub/Qwen3.8-27B-hermes-adapted/blob/09fb7412411abeeb38230bca646f78ba5993b25b/README.md)。

### inclusionAI：SingProbe 新增DeepSeek／Ling内在流式防护权重

- **时间版本**：11月19日DeepSeek adapter `9c7c433c66114fa18a29bb1d1a15756861eb0879`（11:33:56Z），Ling adapter `2eaba21b0dd5081a32d29e0b7b5ae52811a94598`（11:32:49Z）。这是新适配工件；SingProbe方法报告不是今日首次提出。卡片MIT声明已固定，独立LICENSE404。
- **机制／实物**：分别为10.10M／8.13M参数探针，读取基座的 `[12,25,39]`／`[23,47,71]` 零起层hidden states，8-head×64、2048-token窗口，为每token提供8类query intent、response unsafety及hallucination风险。树中BF16 safetensors分别20,196,980／16,264,820字节；它们不包含DeepSeek／Ling基座。
- **结果限制**：DeepSeek探针流式response/token AUC为0.9907/0.9522，对照Qwen3Guard-Stream-8B-strict为0.9640/0.8893；但hallucination AUC0.7951低于DRIFT0.8000，不能写成所有任务领先。每个benchmark等权，F1阈值在suite上最大化均值；“良性误报0.009%”使用校准前阈值0.987179，发行阈值是0.5，**不能照搬成部署误报率**。示例依赖code.alipay上的SGLang261029分支和8×H20，本轮未能证明匿名公共用户可取得该runtime，亦未审计方法实现。
- **建议（推断）**：适合跟进“模型内部探针替代外部guard模型”的部署成本；优先核公共推理代码和独立校准集。[DeepSeek固定卡](https://huggingface.co/inclusionAI/DeepSeek-V4.1-Flash-1017-singprobe/blob/9c7c433c66114fa18a29bb1d1a15756861eb0879/README.md)；[Ling固定卡](https://huggingface.co/inclusionAI/Ling-3.1-flash-1019-singprobe/blob/2eaba21b0dd5081a32d29e0b7b5ae52811a94598/README.md)。

- **Meta R3D-Bench v2**：11月19日22:45:37Z `3530576dd8a1f167d74d1459e3c7d9f37cf60670`。卡及parquet列表可读，问题数3033→2950，仍57个ADT序列、eval-only；新卡披露对象可见性／静态条件、问题平衡／cap／抽样规则与pointing disambiguation。MC按accuracy、距离体积按MRA，旧新题集分数不可直接比。本轮未下载parquet独立数行。[固定卡](https://huggingface.co/datasets/facebook/r3d-bench/blob/3530576dd8a1f167d74d1459e3c7d9f37cf60670/README.md)。

- **Meta omnilingual-gaia2**：11月19日11:07:10Z `348deb1cacc031ea30616d1d9dc549a001603e6b`，diff只改dataset card `cc-by-nc-4.0`→`cc-by-4.0`。没有独立LICENSE文件，本轮只确认metadata声明更改，不能替代全部组成资源的授权核查。[固定卡](https://huggingface.co/datasets/facebook/omnilingual-gaia2/blob/348deb1cacc031ea30616d1d9dc549a001603e6b/README.md)。

**Introspective Oversight（2611.23365v1，提交 11-17）：单独训练第三人称observer角色回答可验证的自身行为问题。**

第一作者：Wang, Alex Lin及其团队；原始提交：2026-11-17；19 November 公告恢复（首发秒界未确定）。

 行为预测和hidden-trait监督训练observer，保留与assistant共享权重的特权信息。§4.1 Qwen3.6-27B、非reasoning transcript、按预填答案token logprob的五类OOD平均AUROC0.907；第一人称assistant0.877，honesty加入persona后0.884。**限制：** reward-hacking仅40例；开启reasoning后observer只匹配外部transcript monitor，未测更强模型隐藏目标、CoT监控或自由文本正确性。建议深入读训练角色分离与同权重／外部模型控制，复现时保持transcript来源、score读取与reasoning设置一致。[正文 §2–4 / Table1, Fig.3, Limitations](https://arxiv.org/html/2611.23365v1)

**ExecJudges（2611.24849v1，提交 11-18）：每个topic先写可执行评分程序，再在CPU给文档批量打分。**

第一作者：Agichtein, Eugene及其团队；原始提交：2026-11-18；19 November 公告恢复（首发秒界未确定）。

 三个独立程序以词法/NLP库/池统计返回四个判断并合并grade，避免逐文档LLM调用。Table1相同每topic16人工label，DL2022 ordinal α：ExecJudges-L .702，classifier .573，listwise Sonnet .591；但Robust04/TREC-COVID低于同label Sonnet约.05。Table2在匹配pointwise文档上DL2022 .628 vs .577。**限制：** label按真grade分层抽取，现实不知道grade时未证明同收益；最强top10系统排序较不稳定，规则可被gaming且未测；同一模型家族。建议复现主动选label及top-ranker排名，别把“CPU评分无额外模型调用”写成无计算成本。[正文 §3–6 / Tables1–3](https://arxiv.org/html/2611.24849v1)

**STEMBench（2611.24321v1，提交 11-18）：科研agent按领域figure-of-merit评分，同时公开无领域知识baseline与可运行专家anchor。**

第一作者：Avanesyan, Alina及其团队；原始提交：2026-11-18；19 November 公告恢复（首发秒界未确定）。

 十二个HEP/天体/神经科学任务分别给原始分数、baseline差距及独立submission有效性，避免只看“代码跑通”。Appendix C/Table4：离线单T4、1小时、累计1M输入＋输出token、200messages；Table5八模型每格三次保留运行均值，含零，288run。高effort至少一模型的均值在7/12任务越过expert anchor，但Higgs uncertainty、latent/causal推断等仍难。**限制：** 十二任务不是科研覆盖面；expert是具体公开方法而非不可超越上界；“数值更高”与“提交可复现”不能合并。建议优先使用原生metric和baseline/expert双anchor设计，复现时检查retained-run规则与失败记录。[正文 §3–5; Appendix C/D Tables3–6](https://arxiv.org/html/2611.24321v1)

**FutureBench（2611.23601v1，提交 11-17）：要求“核验”常同时改变证据优先级，不能把prompt差异直接解释为推理能力。**

第一作者：Lai, Yi-Cheng及其团队；原始提交：2026-11-17；19 November 公告恢复（首发秒界未确定）。

 3,629时间戳问题、真假/冲突passage三元组和无context probe；Table2 pre-cutoff o4-mini错误答案采纳33.9%→87.0%，GPT-5.1 42.7%→74.8%，但post-cutoff多数模型Verify反而降低采纳。902条四prompt比较进一步拆分checking和passage priority。**限制：** 模板措辞也变化，未纯隔离因子；生成与评判均依赖Gemini，无人类裁决；未保留immutable provider snapshot及原访问日期，精确重跑受限。建议复现时把“信哪份来源”与“是否检查”做正交设计，保留路由snapshot；适合RAG评估诊断，不作为新模型榜单。[正文 §4–6 / Table2, Limitations](https://arxiv.org/html/2611.23601v1)

**Language Models Make New Research Sound Old（2611.23546v1，提交 11-17）：模型的“是否已研究”判断与自然语言报告之间有大缺口。**

第一作者：Zyskind, Guy及其团队；原始提交：2026-11-17；19 November 公告恢复（首发秒界未确定）。

 40个新接收论文问题中，Astra强制独立判断98%未研究，普通回答仅8%明确说open；六模型差距30–90点；给同论文后单独检查全文，多数会否定此前过度近似。§5.1同80个matched studied/open问题、相同相关文献，六模型mean AUC：直接判断84%，novelty grade约52%。**限制：** 多为ML问题，open标签和paper覆盖由搜索/AI审查决定，不能证明绝无先例；RINoBench可能有记忆，模拟读者未证实真实研究者决策效应。建议将该结果用于文献调研流程：分开精确定义问题、独立status判断、逐篇全文覆盖检查，并明确剩余未知。[正文 §3–5; Appendix A–C](https://arxiv.org/html/2611.23546v1)

<a id="coverage"></a>
## 实际覆盖、补录与续查

本轮收录 **158 个带稳定标识的版本事件**。正文标注回溯补录与评测修订；独立抽查后补齐了AINIC规避仍可缺数据、统一PP投机HiCache门禁及MLXcel质量阈值的边界。数字均为作者报告，本轮未执行独立复训、设备压测或付费API实测。

### 厂商与模型工件

- AutoClaw11月19日GLM5.3Flash文章正文已打开，披露320B/18B、45层、30T语料、IndexPool四合一与国产加速器serving；该型号在更早来源已存在，正文没有训练／权重新版本号，不能据博客日期报“今日首发”。其3×serving对照缺硬件／批量完整条件，暂不作新指标。

- MiniMax M2/M3、NVIDIA RivaTranslate1.1的当日改动仅 `new_version` 导流指针；Xing4.0只是生态链接／引用修订，均不报新权重。

- LingDT-Decision-Fast-1.0当日卡正文可读，但自称staged profile，许可证／模型权重／API上线未承诺，吞吐表缺硬件和并发，列待查，**不报开源模型发布**。MathTok只有MIT frontmatter，无方法内容，筛除。其余全局长尾未逐项语义审计。

- 157站点含128个HTTP200、14个403、6个连接失败、9个404；200可能只是动态骨架。HF181个组织名（大小写归一后176个）及models/datasets目录可读，主窗全局创建目录枚举4339模型／1707数据集；目录枚举不代表语义审计完整。官方动态渠道、任意仓lastModified、历史长尾与原待查区间均保留，未搜索到不等于没有发布。

### 论文与会议：19 November 公告恢复及补查

本轮主窗为 `(2026-11-19T01:04:03Z, 2026-11-20T01:02:12Z]`，72 小时回溯自 `2026-11-16T01:04:03Z`。十个 arXiv 分类均已显示 **19 November** 公告，尚未看到 20 November 公告；以下新批次按 **overlap／公告边界不确定的恢复项** 记录，不能断言全部首发于主窗。各项固定 `v1`；提交时间见 ledger，发现日期为 2026-11-20；提交时间不等于公开时间。所有数字均为作者报告，本轮未独立复现，也未完成专用实现的代码审计。HTML 正文、摘要页和列表快照保存在同目录 evidence。

本轮未提升为日报项：MoE-Context（2611.24379）主要是Qwen3-1.7B合成检索，关键词检索已完美、probe不迁移；Reliability is a user-specific property（2611.23732）是路线图而非新量化结果；CurvDelta、Levenshtein diffusion正文已保存但实验审读未完成。CPTC（2611.22019）HTML404，继续等待PDF及Trainium4实现证据。前轮替换版本与formal venue缺口继续开放。

取证统计：21项入选（19 overlap恢复、2 backfill），25份正文下载、23份语义审读、93次来源请求；962个19Nov分类去重ID，合并HF发现1056候选。当前候选1020、继承276、替换版本358。OAI合并6833 unique ID，但请求日期被服务端改写，不能视作精确窗口完成。OpenReview三接口403；ACL/PMLR仅索引与定向检索，SC26 tech_paper404，均不宣称完整增量覆盖。全部数字为作者报告，未独立复现，未完成专用实现代码审计。

### 工程源覆盖与续查边界

核心六项 Megatron-LM、vLLM、verl、SGLang、slime、TorchTitan 的默认分支履历、merged PR 目录和 release 入口均已检查。**slime 未跳过**：[THUDM/slime 官方 main](https://github.com/THUDM/slime/commits/main/) 当前头仍为 `79aa41a884ab0bf65222c94418f6357d3c31192b`（2026-11-05T23:32:32Z），本轮72小时默认分支/merged目录为空；最新release仍为10月9日v0.4.0。`zai-org/slime` 入口404另记失败，不用别名失败替代canonical成功。Miles单独按独立仓库报道，不作为slime无变化的替代证据。

扩展工程总计86个入口，73个默认分支、merged分页和release回溯目录完整；13个历史/别名入口仍404。列表遍历只证明发现范围，不代表语义全覆盖：本轮选择性核实67个新事件，3,045个未报道库存行（含维护、重叠历史和待查变化）保存在 `unreviewed-candidates.json`，旧386项pending继续保留；未逐一遍历所有维护分支历史。TorchTitan #6142/#6147和Transformers #50717仍是draft/null merge SHA，不算已发布能力。硬件栈、TensorRT-LLM、llama.cpp和Dynamo博客由硬件线单列。

官方REST匿名403和部分固定commit `.patch` 503已精确记录；四份release与两项PyTorch patch经GitHub连接器恢复，失败日志仍留存，不将暂时访问失败解释为没有发布。没有GPU/NPU压测、复训或完整上游测试套件结果。

- **硬件与软件协同。** 337次直连请求，262次完整HTTP200；21仓Git补查中8仓深度跨回溯边界，其余分页/超时保留。检查NVIDIA/CUDA/NCCL/TensorRT、AMD/ROCm/AITER/MORI、昇腾CANN/HCCL/torch-npu、TPU/Google、AWS Neuron、Intel/Gaudi、Apple MLX及国产厂商入口；66个固定提交候选差分取证，正文选择性分析。GitHub部分patch503由diff200恢复；REST403与Git失败不当扫描成功，初始沿用旧until的REST查询已另发当前窗口查询，旧记录不作主窗覆盖证明。llama.cpp、XLA/ROCm-libraries等目录/分页及语义未闭合，CANN正式包映射与libtpu内部行为继续查；不声称上述所有供应商无新增。

全局完整覆盖水位仍为空。下一轮从本轮截止 2026-11-20T01:02:12Z 续扫并回溯72小时，同时继承各来源尚未完成的区间。本地.monitor保存来源清单、检查回执、候选、已收录版本与同步状态；来源访问失败和目录语义未审均未冒充完成。
