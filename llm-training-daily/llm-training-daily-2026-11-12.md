# LLM 技术动态日报 · 2026-11-12

> 时区 Asia/Shanghai（UTC+8）。增量扫描窗口 **2026-11-11 09:00:49 → 2026-11-12 09:05:00**，向前回溯 72 小时至 **2026-11-08 09:00:49**。实际生成时间 **2026-11-12T09:14:17+08:00**。日期未披露到时分时保留原日期。首次发布、工件或代码更新、会议报告与本轮发现分别标注。
>
> 生成前核对本地 36 篇历史日报及索引，与 GitHub `main` 的 `65615a98905725d9bacbc708fd210dcbc7d44192` 逐字一致。前次待上传队列为空。
>
> 本轮核验公开的一手报告、工件、发布记录及部分代码补丁。性能与能力指标为原作者或项目方的口径。

## 本次重要变化

- **模型与报告。** Qwen4-Max Preview 披露百万 token 上下文及思考预算，文心 5.5 Flash Preview 是 Flash Base 的首个后训练版本。Runway 开放 Praxis-1 权重与训练入口，附第三方仿真榜单可复核。
- **训练与后训练。** 数学中训练论文对照了下一 token 续训与 on-policy 蒸馏。verl FSDP actor 接入 batch-invariant 算子和 FP32 log-softmax，Megatron 修复 GTP checkpoint 的 SwiGLU 分片，TorchTitan 恢复训练 MoE router gate。
- **长上下文与投机解码。** ForeSpark 按实测验证成本动态分配草稿预算。TKNP 在 32 张 H100 上把请求 KV 留在归属节点，收益集中在 KV 放不下的高并发长上下文。TensorRT-LLM 合入稀疏 decode KV offload，昇腾 GDN 检查点可在单次 prefill 生成。
- **硬件与系统。** Neuron 2.33.1 修复 DMA 并发挂起。cuDNN Frontend 合入 SM120 AOT 注意力和可选的确定性 GDN/KDA 训练路径。SGLang TPU 修复在线 FP8 下 TP 复制 KV 头不一致，ATOM 加入 rollout 暂停和权重更新后恢复。

[模型与技术报告](#models) · [研究论文](#papers) · [开源工程](#engineering) · [硬件与软件协同](#hardware) · [覆盖与续查](#coverage)

<a id="models"></a>
## 模型与技术报告

### Qwen4-Max 开放限时预览，思考与最终回答共用输出预算

**11 月 11 日 10:00（北京时间），API 预览。** `qwen4-max-preview` 接收文本、图片和视频，输出文本，支持工具调用、联网搜索与结构化输出。QwenCloud 文档列出 1M 上下文和最多 393,216 个输出 token，思考与最终回答共用这笔额度。Chat Completions 的 `max_tokens` 与 `max_completion_tokens` 都约束两者总量，多轮默认要求在助手消息的 `reasoning_content` 中回传历史思考。按旧版可见答案长度设置预算，可能让新模型在给出完整答案前耗尽额度。预览只向 Token Plan 开放，文档承诺期间版本固定，11 月 25 日 24:00（UTC+8）预览结束后下线。已核到的官方材料披露 API 契约和示例，训练配方、下载权重与可复核的独立分数仍待查。 [QwenCloud 预览规格](https://docs.qwencloud.com/developer-guides/getting-started/latest-model)

### 文心 5.5 Flash Preview 发布首个后训练版本，长程案例依赖外部工具

**11 月 11 日（北京时间），托管 API。** 百度把 11 月 5 日已发布的 Flash Base 后训练为 280B 总参、每 token 激活 14B、1M 上下文的文本模型，开放千帆与 OpenRouter API，预告 11 月 25 日前开放权重。千帆披露使用 FP4 专家权重与草稿头，标称解码超过 200 token/s，默认输出 64K，页面未给出这项速度的硬件、并发和上下文口径。厂商称训练用了超过 1,400 万个可执行代码任务和多个 Agent 框架的执行轨迹。公开的虚幻引擎案例运行约 11 小时，视频案例另有代码渲染与外部生图、视频工具参与。它们展示了模型在具体工具链中的长程执行，评测附表与案例均由厂商发布。 [文心发布页及评测附表](https://ernie.baidu.com/zh/release/ernie-5-5-flash-preview/) · [千帆规格](https://cloud.baidu.com/product/qianfan_home/model_3.html#ernie-flash)

**同期产品页**还披露长上下文的 ERNIE-Light-DT-1.0，使用 BSA 稀疏注意力与反馈式历史压缩，支持按请求开关思考和最高 128K 输出。千帆页面未给出与同尺寸稠密注意力对照的 KV 与质量曲线。 [千帆同期模型规格](https://cloud.baidu.com/product/qianfan_home/model_3.html)

### Runway 开放 Praxis-1 权重，视频隐状态驱动 1.6 秒动作块

**11 月 11 日，公开权重与代码。** Runway 将 9 月预告的 27B MoE 扩散模型公开无动作头的预训练基座、单臂 DROID 与双臂 YAM 微调版，采用 Apache-2.0。输入指令、最新相机帧与机器人姿态，视频骨干预测未来画面，动作专家读取视频隐状态，按 15 Hz 输出 24 行动作。机器人执行这 1.6 秒动作后把新观测送回策略。每种机体用独立输入输出投影，共享骨干和动作专家。NVIDIA RoboLab-120 榜单列出 DROID 微调版完成 555/1,200 次仿真试验，成功率 46.2%，原始成功率排名第 5。该成绩绑定 DROID 数据和仿真协议。代码库带推理、全量微调和 LoRA 路径。 [开放公告](https://runway.com/news/company-news/praxis-1-is-now-available) · [技术报告 §2、§4](https://runway.com/research/praxis-1-video-pretraining-to-robot-control) · [固定基座](https://huggingface.co/runwayml/praxis-1-base/tree/2d61610bda3d4eb620c6b9fb7a83281303a0a6f5) · [固定 DROID](https://huggingface.co/runwayml/praxis-1-droid/tree/6bb70750663e53592cc38f46e531a5a01a12d291) · [固定 YAM](https://huggingface.co/runwayml/praxis-1-yam/tree/b72c4bbc0430d40ef9bc26c49ae6d9773298ee07) · [RoboLab-120](https://research.nvidia.com/labs/srl/projects/robolab/leaderboard.html?family=rfms&open=1)

### 补录｜PAN 2 用带时间戳的文本状态连接语言推理与视频扩散

**IFM / MBZUAI，11 月 10 日（公告当地日期），公开视频模型权重及训练代码。** PAN 2 把场景和动作先交给外部多模态语言模型，生成包含事件、动作与镜头运动的带时间戳文本。29B MMDiT 对文本与可选起始图像潜变量做条件扩散，再用 VAE 生成最长 20 秒的视频。Apache-2.0 发布包提供视频模型权重及训练、推理代码，可固定到 `a7936f7b`。厂商报告 75.1% 单步完成、34.0% 整体计划完成，Physics-IQ Verified 的报告配置使用 Claude Opus 5.5，RBench 用 GPT-6 Astra。复现这些结果需匹配外部推理器和提示设定，公开视频权重本身不能确定整套系统的分数。 [IFM 发布公告](https://ifm.ai/pan2/press-release/) · [技术报告及评测配置](https://ifm.ai/blog/pan2/) · [固定权重目录](https://huggingface.co/IFM/PAN2/tree/a7936f7bea20ec3b9de809583712b40693071819) · [训练和推理代码](https://github.com/ifm-ai/PAN2)

### BotPsy 把小模型的工具调用拆成 SFT、在线蒸馏和两轮 RL

**Tether AI Research，11 月 11 日，Apache-2.0 权重。** BotPsy 基于 Qwen3.5-0.8B，先做 SFT，再做多教师 on-policy 蒸馏和两轮强化学习。发布 BF16 safetensors 与 BF16、Q8_0、Q4_K_M GGUF。统一开启思考、以 vLLM 跑 BF16 的厂商评估中，六项 Agent 基准平均 45.0%，高于同协议的原始 0.8B（24.7%）和 Qwen3.5-2B（38.8%），但 Claw-Eval 落后 2B（46.5% 对 48.9%）。发布模板默认关闭思考，手机上的量化效果、延迟和语言迁移仍需在实际运行时验证。 [技术发布](https://qvac.tether.io/blog/on-device-ai-agent-botpsy) · [固定模型卡](https://huggingface.co/qvac/BotPsy-0.8B/blob/24561e9681643eaf807fc28b0875210cb17f89c1/README.md)

### Nex-N2.6 固定 FP8 工件，评测回放检查重复推理和工具选择

**11 月 11 日 23:03（北京时间），固定模型卡与运行时元数据。** Nex-AGI 基于 GLM-5.3-Flash 发布约 321B 的 FP8 权重。内部回放固定历史对话与工具返回，只重测下一次响应，推理重复率由 Nex-N2.5-Pro 的 16.0% 和底座的 6.0% 降到 1.6%。团队还报告调用工具的响应中工具名都来自当次请求的列表，这项指标只检验选择范围。厂商报告 OSWorld 83.0、Terminal-Bench 2.1 84.1、AutomationBench 45.7，明确属于模型级结果，未对这份 FP8 导出单独评估。 [固定模型卡](https://huggingface.co/nex-agi/Nex-N2.6/blob/987472298e7867721576ac49a253a6f5b526013e/README.md) · [提交历史](https://huggingface.co/nex-agi/Nex-N2.6/commits/main)

### Raon-1 nano 发布 MTP 权重，混合注意力仅六层保留 KV

**KRAFTON，11 月 11 日 14:21（北京时间），新增 MTP 工件。** Raon-1 nano 从 30B-A4B mini preview 剪枝、蒸馏为 dense，再续训 3T token，经 SFT、on-policy 蒸馏和 RL。主模型 3.1B，其中逐层 embedding 表约 1.5B。24 层由 18 层定长状态 Raon Delta Attention 与 6 层 Gated MLA 组成，后者的 BF16 KV 约 7 KiB/token，原生上下文标称 262,144。当天新增单层 55M MTP 权重与参考 forward，模型另提供 GGUF、MLX 和 LiteRT-LM。部署端能否用 MTP、实际内存和长上下文速度取决于运行时与格式。 [固定模型卡](https://huggingface.co/KRAFTON/Raon-1-nano/blob/fcf2d1d55e84e2f9fbe7b5d724dcfc4d861cfcee/README.md) · [MTP 提交](https://huggingface.co/KRAFTON/Raon-1-nano/commit/6d045a9e817fef24a8672b2a64c3f5423f1e3c2a)

### ZoomGain 从全图和局部裁剪的感知差中构造 RL 数据

**inclusionAI，11 月 11 日，4B/9B 权重、训练数据和评测集。** 多个定位模型先在全图对目标达成一致，再对比全图与裁剪的描述，提取放大后才能辨认的细节，用局部图验证答案并过滤可脱离图像回答的题。训练与测试仍要求模型依据全图作答。作者在中英属性、计数和 OCR 的合并基准上报告 28 个对照模型最高 61.08，Qwen3.5-9B 为 32.13。结果受该基准的图像分辨率、题型和评分协议约束。 [论文 v1](https://arxiv.org/html/2611.14076v1) · [固定 4B](https://huggingface.co/inclusionAI/ZoomGain-4B/tree/2ac9e28f1de993b4d7b6ebeda1368ff2e6982a71) · [固定 9B](https://huggingface.co/inclusionAI/ZoomGain-9B/tree/b4550ef2e5a30b33c5133278036a682197489145) · [固定 RL 数据](https://huggingface.co/datasets/inclusionAI/ZoomGain-RL-Data/tree/1c37933a47b417f05a95c8fe2c8a1ea02d71f3b3) · [固定评测集](https://huggingface.co/datasets/inclusionAI/ZoomGainBench/tree/cf77d2f028920a3cac51a7931e88688b1df52bf6)

### inclusionAI 发布直接输出选项概率的 Sing VL S1，LemonDecider 补齐评测明细

**11 月 11 日 13:38（北京时间），Sing VL S1 固定卡片。** 在 Qwen3.5 上做多模态类型化决策，可接收文本或 JSON 及最多四张图片，对每道具名问题用一次 forward 给出选择题选项概率、是非概率或有序评分分布。作者报告 Decision Index Public 0.3 文本 69.76、视觉 73.94。该表是模型卡披露的公开集结果，验收记录和私有集表现仍待查。 [固定模型卡](https://huggingface.co/inclusionAI/SingVL_S1_4B/blob/93f15cd61c1cf417d9cb2631f6be2df13510829e/README.md)

**同日，LemonDecider step1241 工件。** 公开 140,620 条旧评测记录，其中 140,178 条计分，442 条按评测规则排除。两份权重分片 SHA-256 与评测包的模型身份记录吻合。数据卡和目录称 Decision Index 0.3，模型卡称 0.4，两处都报 61.67。发布方注明这是 11 月 4 日的公开集自测，尚无外部榜单验收记录。 [权重替换提交](https://huggingface.co/inclusionAI/LemonDecider/commit/ef27e061ea67b7df6109b189aaf6b1231fc9bfd9) · [固定模型卡](https://huggingface.co/inclusionAI/LemonDecider/blob/05ec18fa0994b7369cb25ac76c19ed0865c776fd/README.md) · [固定结果与来源记录](https://huggingface.co/datasets/inclusionAI/lemondecider-decision-index-results/tree/1d665f244fb08c6f3394dfb28aa59658eb22c936)

### 补录｜ERNIE Evolve 用独立评测筛选训练与推理代码改动

**百度，11 月 9 日（官方日期），技术报告与优化工件。** 系统外层选择研究方向，内层沿候选树试改代码，在隔离环境运行实验。独立评测程序和审查决定是否保留改动。百度报告 NanoGPT Speedrun Track 1 从 39.9 秒降到 38.9 秒，Track 3 从 2,690 步降到 2,590 步，SOL-ExecBench 平均 0.819，仍低于公开对照的 0.827。在线上研发案例中，百度报告 ERNIE 5.5 Flash 的确定性推理吞吐提高 33.7%，限定同一服务和确定性约束。开放仓包含公开任务的优化工件，生产配置和独立复跑证据仍待查。 [文心报告及表 1](https://ernie.baidu.com/zh/research/posts/ernie-evolve/) · [公开结果仓](https://github.com/ernie-research/ERNIE-Evolve-Results)

### 已有模型的量化、权重和许可变化

- **NVIDIA Kimi K3.1 Pro NVFP4 R3，11 月 11 日 12:13（北京时间）。** 路由专家从 MXFP4 转为 NVFP4，共享专家和 KDA 用 MXFP8，MLA 保留 BF16。NVIDIA 的 GB300 评测中，Terminal-Bench 2.1 从原 MXFP4 的 88.20 降至 87.78，AA-LCR 从 81.69 到 82.06。该评测中主 KV 与索引缓存用 FP8、SSM 状态用 FP32。数字依赖这组运行时精度和采样口径。 [R3 固定差分](https://huggingface.co/nvidia/Kimi-K3.1-Pro-NVFP4/commit/546b174b3df6aa25031da86e1551af5addd19bbb) · [固定模型卡](https://huggingface.co/nvidia/Kimi-K3.1-Pro-NVFP4/blob/a0b0283f0f4f91f09b37a6e5fe45a62de269cbd0/README.md)
- **Jais-3 MoVA 36B-A4B，11 月 11 日 23:18（北京时间）。** Inception42 为这款 512K 稀疏模型发布三片 BF16 权重，合计约 74.9 GB。官方卡另列 FP8、GGUF 和 MLX 工件，BF16 仓可作为同底模量化质量对照。 [固定权重](https://huggingface.co/inception42/Jais-3-MoVA-36B-A4B/tree/438040d486505afb0ba9767e3566bca9fcb4602e) · [BF16 提交](https://huggingface.co/inception42/Jais-3-MoVA-36B-A4B/commit/438040d486505afb0ba9767e3566bca9fcb4602e)
- **Nota AI Nemotron 3.5 Super VL，11 月 11 日 20:34（北京时间）。** 40% 全局剪枝版从原 NVFP4 checkpoint 的 20,480 个路由专家中保留 12,288 个，各层保留数写入 `num_experts_per_layer`，需要定向 vLLM 补丁。权重从 70 GiB 减至 46 GiB。发布方报告韩语回答一致性由 76.2% 升至 90.1%，同期韩语题准确率从 71.4 降至 67.6，英语题从 84.0 降至 78.4。 [固定模型卡](https://huggingface.co/nota-ai/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-Global-Pruned-40/blob/bbd3b629b0391deb093e876fec1655931ff4d16b/README.md)
- **Ripple-65B Preview NVFP4，11 月 11 日 09:58（北京时间）。** 维护者更新 DGX Spark 推理包并打开前缀缓存。提交没有发布同配置的缓存命中与冷启动对照。 [固定差分](https://huggingface.co/ricursiveAI/Ripple-65B-Preview-NVFP4/commit/2789217ba87d93e8d032fad25217c4c63a0e1d5b)
- **RoboBeats，11 月 11 日 19:04（北京时间）。** 0.8B、2B 和 4B LoRA 工件的当前卡片许可改为 CC BY-NC 4.0。11 月 5 日日报记录的是当时仓库标示的 Apache-2.0。 [0.8B 固定卡](https://huggingface.co/superbigsaw/RoboBeats-0.8B/blob/0cb1bfe6f47ede15fc88ccda4599a9cf3183714a/README.md) · [2B 固定卡](https://huggingface.co/superbigsaw/RoboBeats-2B/blob/b176d3774ec0593b21c3927256c6661b1a567800/README.md) · [4B 固定卡](https://huggingface.co/superbigsaw/RoboBeats-4B/blob/2489aeb1acc7dd47571450f13d92744fb9113ba9/README.md)

### USDCraft-10K 扩至 400 类可仿真的 OpenUSD 对象

**NVIDIA，11 月 11 日 14:55（北京时间），数据集快照。** 固定卡记录 3,816 个对象，其中 3,116 个有关节，700 个为刚性静态对象。生成过程由 DeepSeek-V4.1-Flash 写资产程序，再编译为带部件、关节范围、碰撞几何和质量的 PhysX USD。每个对象附提示、文件索引和哈希，可按类别与关节类型切分。10K 为规划规模，当前只核到这批资产，卡片所列仿真格式没有给出真实机器人迁移成绩。 [固定数据集卡](https://huggingface.co/datasets/nvidia/USDCraft-10K/blob/64d4b9df1fc8f6bba862c4701bc70dd750a04db7/README.md) · [固定扩容提交](https://huggingface.co/datasets/nvidia/USDCraft-10K/commit/64d4b9df1fc8f6bba862c4701bc70dd750a04db7)

### GTAVBench 给 150 个评测任务补齐可交互入口

**OpenMOSS，11 月 11 日 11:02–11:19（北京时间），评测工件修订。** 固定版为 150 条测试任务加入逐行 `play_url`，将英文指令、1280×720 初始画面与浏览器中的任务入口关联。卡片区分两种协议，一种给模型文字指令并接收 JSON 键鼠动作，另一种在首张截图中给指令并用 Codex harness 执行动作。Parquet 可复核题目和初始状态，执行评测依赖单独的游戏环境和 runner。 [浏览入口固定差分](https://huggingface.co/datasets/OpenMOSS-Team/GTAVBench/commit/9228b80414279f0a41fabb90e98b61bfc37ada1a) · [固定数据卡](https://huggingface.co/datasets/OpenMOSS-Team/GTAVBench/blob/ffef6928ffe73d9671e9c905146483261cdee142/README.md) · [论文 v1](https://arxiv.org/abs/2611.13028v1)

<a id="papers"></a>
## 研究论文｜预训练与架构

以下 arXiv 条目来自 11 月 11 日分类公告，按官方常规排程对应 11 月 11 日 01:00 UTC，早于本期主窗起点 49 秒，作为 72 小时重叠补扫收录。前一期在抓取时仍得到旧批次，逐篇公开秒无法确认，各条另列 v1 提交时间。美东 11 月 11 日是假期，常规应在 11 月 12 日 01:00 UTC 发布的一批顺延。 [公告与假期规则](https://info.arxiv.org/help/availability.html#announcement-schedule) · [cs.CL 公告](https://arxiv.org/list/cs.CL/new?skip=0&show=2000) · [cs.LG 公告](https://arxiv.org/list/cs.LG/new?skip=0&show=2000)

### 有限预算下，训练损失面能改善模型尺寸与数据量外推

**11-10 03:56:41 UTC 提交｜arXiv:2611.13534v1。** 作者在数字化的 245 次 Chinchilla 训练中，使用 ≤4B 的 221 次试验拟合，在 >4B 的 24 次试验留出验证。对低损失留出子集，原始幂律的 MAPE 为 1.92%，剔除高损失训练试验后为 1.34%，在相同筛选样本上拟合对数二次损失面后为 0.79%。全 24 次留出试验上，后两者分别为 2.34% 和 1.86%。原始数据缺少每次学习率日程，损失阈值只能作为训练收敛程度的代理。做预训练试跑时，可记录每个预算点的优化器设置，并用较大模型的留出点选择外推形式。 [论文 §6、表 2、图 7](https://arxiv.org/html/2611.13534v1#S6.T2)

### 数学中训练让学生从语料前缀生成，再用教师重评分

**11-10 10:15:20 UTC 提交｜arXiv:2611.13855v1。** OPMD 从数学语料选取前缀锚点，在学生自己的续写上用冻结教师提供逐 token 信号。在同一数学语料的对照中，Qwen3-1.7B-Base 的 MATH-500 avg@16 从 18.5% 降至下一 token 续训的 1.5%，使用 4B 教师的 OPMD 后为 30.5%，换 14B 教师为 29.9%。Llama-3.2-3B 的续训先导实验方向相反，评测限于 MATH-500 与 Minerva。适合在候选基座上用同源语料做 NTP/OPMD 小规模对照，分别记录平均解题率和至少一次成功率。 [论文 §3–5、表 1、附录表 2–3](https://arxiv.org/html/2611.13855v1#S4)

### 27B 模型迁移到字节词表保留多项能力，中文长输入代价突出

**11-10 05:33:41 UTC 提交｜arXiv:2611.13607v1。** 作者删除 Qwen3.8-27B 的 BPE 合并规则，保留单字节与特殊符号对应的词嵌入，再用自生成数据和通用数据混合续训。相同 RTX 6000 Ada 上，思考模式 MATH-500 为 95.4% 对原版 94.0%，题目级 bootstrap 差值区间跨零。C-Eval 两种模式各降 4.5 和 4.7 个百分点，中文生成字符速率约为原版四分之一，长输入 prefill 慢 5–6.5 倍。实验为单次训练，评测重采样只覆盖题目抽样。评测仓的 `e10_bootstrap/e10_boot.py::pboot` 按题做配对重采样，`e6_efficiency/e6_bench.py::call` 分别记录字节、字符和 token。仓库包含评测侧资料，权重和字节解码服务组件尚未随仓发布。比较分词器时宜按相同内容量测吞吐、首 token 延迟与输出上限命中率。 [论文 §3、§5、§7、表 1](https://arxiv.org/html/2611.13607v1#S5.T1) · [固定评测快照](https://huggingface.co/datasets/leiwhere/ld3-byte-eval/tree/6541069f77e5a6207036d11b37b612230bb00c8b)

### 循环深度可共享末轮 KV，循环位置决定 prefill 成本

**11-09 23:55:02 UTC 提交｜arXiv:2611.13378v1。** 因果 encoder–decoder（CED）把循环放在 encoder 时每轮更新全局 KV 表征，并以随机循环深度训练促使各轮收敛，推理只保留每 token 的末轮 KV。150M 模型在 FineWeb-Edu 的 2,048 token 验证中，encoder 循环从 0.8716 降至 0.8468 BPB，常驻全局 KV 同为 8 KB/token。八轮的 12 层配置中，encoder 循环把每 prompt token 的层执行次数从 6 提至 34，decoder 循环保留 6 次但 BPB 为 0.8648。大模型实验测的是未按循环训练的 DeepSeek-V4.1-Flash 服务成本。评估时应同时扫 prefill/decode 比例和缓存命中率，并单测模型质量。 [论文 §3、§5–6、§8、表 1、表 3](https://arxiv.org/html/2611.13378v1#S5)

### MoE 领域专长可在专家内的神经元粒度检验

**11-10 12:14:57 UTC 提交｜arXiv:2611.13975v1。** 作者先用路由权重和神经元输出估计重要性，再在独立数据验证专家路由与领域富集。在 6.25% 的选择预算下，Qwen3.6 约 13.39% 的专家富集某领域神经元，容纳该领域 83.40% 的独占神经元。逐专家裁神经元保留 75% 时，Qwen3.6 两种打分均保留 97.63% 的平均能力，OLMoE 分别为 95.30% 和 89.91%。领域划分、校准数据和评分方法都会影响压缩结果。选剪枝粒度时可在目标模型上同时跑领域留出及同预算整专家剪枝基线。 [论文 §3、§5、表 1–3](https://arxiv.org/html/2611.13975v1#S5)

## 研究论文｜后训练

### 策略优化理论把覆盖坍塌与参考策略更新联系起来

**11-10 18:31:36 UTC 提交｜arXiv:2611.14467v1。** 作者构造基座已有良好答案覆盖的实例，证明固定参考的正则化优化仍可让学习过程难以采到关键答案。近端内循环优化后周期性更新参考策略的 APMD，在论文的 softmax 线性策略假设下可保持覆盖，并为若干简化的 PPO、GRPO 等变体给出收敛结果。图 2 用精确梯度展示反例和参考重置后的改进，结论适用于把完整回答当作一个动作的单轮设定。实际训练可把重置周期和内循环步数作为独立消融，跟踪稀有成功轨迹的覆盖。 [论文 §2–5、图 1–2](https://arxiv.org/html/2611.14467v1#S2)

### AutoScaffold 用冻结教师补足困难任务的训练信号

**11-10 18:58:46 UTC 提交｜arXiv:2611.14515v1。** 教师根据类别统计、失败轨迹和分组结果更新提示脚手架，让全失败的 rollout 组有机会产生带奖励差异的轨迹，评测时移除提示。Qwen2.5-1.5B-Instruct 在 ALFWorld 达到 93.9% 成功率，比论文引用的 GiGPO 基线高 7.2 个百分点。Qwen3-4B-Instruct 数学平均为 70.1%，较 GRPO 高 1.4 个百分点。教师每十步更新一次，附录显示额外验证 rollout 主导更新耗时。复现可用本地同底座、同训练时长基线，记录成功组比例和额外 rollout 成本。 [论文 §3–4、表 3、表 5、附录表 11](https://arxiv.org/html/2611.14515v1#S4)

### 多轮 OPD 用最长公共子序列选择与失败轨迹匹配的示范

**11-09 19:18:25 UTC 提交｜arXiv:2611.13135v1。** 方法在经过奖励验证的成功轨迹池中，选出与学生失败轨迹共享最多有序动作的示范。最长公共子序列允许中间出现绕路，选中的示范进入教师上下文用于蒸馏。相较普通 OPD，ALFWorld 未见布局的 Qwen2.5-3B/7B 成功率从 78.4%/83.6% 升至 85.8%/94.0%。对 3B 把成功轨迹也纳入参考修正后，未见布局得分降到 72.4%。序列对齐是隐状态一致性的代理，测试依赖可判定成功的环境。可将只修正失败作为消融，记录已有成功率是否被更新破坏。 [论文 §3–4、表 1–3](https://arxiv.org/html/2611.13135v1#S4)

### 先蒸馏出有能力的教师，再从同一个学生基座重新训练

**11-10 07:30:34 UTC 提交｜arXiv:2611.13680v1。** BOPD 第一步让参考解可见的自蒸馏产出教师，第二步回到原始学生初始化，让已经学过参考解的教师在无参考条件下给学生 rollout 提供逐 token 概率。三个数学集的 4B 平均 pass@12 为 54.44%，相比第一步的 46.67% 和原始模型的 42.22%。同尺寸 HMMT25 的 avg@12 从第一步 13.06% 降至 11.39%，8B 代码均分相对原始模型仅增 0.09 个百分点。适合在参考答案覆盖不均时比较两阶段开销，按任务分别报告通过率与平均正确率。作者给出的实现链接目前是空仓。 [论文 §4–5、表 1–2](https://arxiv.org/html/2611.13680v1#S5) · [作者仓库](https://github.com/YueWu0301/BOPD)

## 研究论文｜推理与训练系统

### ForeSpark 在生成草稿前联合选择草稿长度与验证预算

**11-10 01:37:38 UTC 提交｜arXiv:2611.13442v1。** 预估器读目标模型当前特征，预测不同草稿长度的接受收益。调度器将收益与草稿、验证的实测时延表结合，再按请求长度打包执行。ShareGPT 端到端、用户交互速率匹配时，Qwen3-14B、GLM5.3-Flash、GLM5.3 的吞吐最高为最强基线的 1.34/1.21/1.18 倍。90% SLO 达成率下的请求率倍率为 1.20/1.13/1.17。增益取决于 CUDA Graph 分桶、硬件时延画像和负载。全局排序也可能让单个请求在一轮里得不到草稿。评估应按实际并发重测时延表，并检查单请求尾延迟。 [论文 §4–6、图 8–10、Limitations](https://arxiv.org/html/2611.13442v1#S6)

### TKNP 以按请求归属的 KV 缓存换取长上下文并发

**11-09 22:01:46 UTC 提交｜arXiv:2611.13293v1。** 在解码池中，dense 权重分布在节点之间，注意力及请求自身的 KV 保留在其归属节点，每层跨节点交换激活。32 张 H100、Llama 3.1 405B、固定 1,024 输出 token 的大批量长上下文负载中，含排队、分批放行及收尾的完成吞吐最高达 DP 的 2.06 倍。能一次放下所有 KV 的低并发情形，DP 因省去每层通信而领先。实验使用 InfiniBand、FP8 权重及 BF16 KV，关闭共享前缀。复测宜同时记录入场等待和已入场请求的 ITL。 [论文 §4–6、图 8–12](https://arxiv.org/html/2611.13293v1#S5)

### ADIndex 让每层按上下文特性选择共享稀疏索引预算

**11-07 06:35:09 UTC 提交｜arXiv:2611.12956v1。** 先由蒸馏后的组级索引器为一组查询建立共享 token 候选池，再在池内逐查询细排。DeepSeek-V3.2 上相对原始 DSA 的端到端速度在 16K 为 0.86 倍、128K 为 1.54 倍，128K RULER 得分由 93.69% 变为 93.34%。同长度下去掉组索引器训练，得分再降至 91.66%。该收益来自长上下文省掉重复全量扫描。可依据流量的长度分布测收益交叉点，给短上下文保留基线路径。 [论文 §4–5、图 3、表 1–3](https://arxiv.org/html/2611.12956v1#S5)

### 多分支草稿提升小批量树验证的接受长度

**11-09 05:15:08 UTC 提交｜arXiv:2611.13017v1。** MFD 一次生成多个候选续写模式，合并到受预算限制的树里由目标模型验证。作者在 SGLang 0.5.22 改造版、单 RTX PRO 6000、batch=1、Qwen3-8B、温度 1 下测得相对同栈自回归解码 5.26 倍加速，共享相同优化、训练数据与树预算的 D-PACE+DDTree 为 4.82 倍，均不含 prefill。论文将优势限定在小批量，单一路径直接选模式的消融反而弱于单模式草稿。复现应固定验证节点预算，同时计入构树时间及不同并发下的核开销。 [论文 §3–6、表 2–3、附录 I](https://arxiv.org/html/2611.13017v1#S5.T2)

### B3 在当前扩散块收尾时提前启动下一块

**11-10 17:40:24 UTC 提交｜arXiv:2611.14387v1。** B3 用生成过程中观察到的熵水平确定重叠时机，当前块与下一块共用前向，注意力掩码阻止当前块读取下一块。三个自编辑扩散模型、十项英文纯文本任务上，生成吞吐升 11.5%–14.5%，各模型平均准确率变化在 −0.40 到 +0.76 个百分点之间。该方法依赖自编辑纠正未定稿上下文的影响，输出没有与原版一致的保证。上线前宜在目标批量下重新测吞吐，并保留质量回归。 [论文 §3–5、表 1–2、Limitations](https://arxiv.org/html/2611.14387v1#S5)

### LM-SimBench 用跨平台实测轨迹验证训练配置筛选

**11-10 16:26:19 UTC 提交｜arXiv:2611.14285v1。** 数据采集覆盖 10 个模型家族的 23 种模型与长度组合、348 种可运行并行配置。对两个小型 Qwen3，8 卡 Ascend 910B3 的实测 Top-3 同时包含 8 卡 V100 的实测最优。直接选 NPU Top-1 时，1.7B 的 GPU 迭代时间损失为 6.9%。V100 使用 FP16/Megatron-LM，910B3 使用 BF16/MindSpeed-LLM，实测交集仅 15 与 11 组。固定实现的 `src/lm_simbench/ranking.py::top_k_hit_rate` 判定是否覆盖实测最优，`top_k_relative_regret` 按全局最优实测耗时归一化候选损失。可用跨平台数据缩小候选列表，再在目标硬件上实测定案。 [论文 §4、§6.3–6.4、图 3、表 7](https://arxiv.org/html/2611.14285v1#S6.SS4) · [固定评估实现](https://github.com/HongriJiujiu/LM-SimBench/blob/ce2d9681a951f10a97f1f8467504c789d0ba9fa7/src/lm_simbench/ranking.py)

## 研究论文｜评估

### 让 LLM 过滤语法最小对，可能删掉母语者认可的难题

**11-10 09:22:38 UTC 提交｜arXiv:2611.13788v1。** 用留出的母语评判者判定题目标签是否有支持。在预注册英语 BLiMP 中，三模型投票的识别 F1 为 0.160，三人小组为 0.331。三模型投票误标的母语支持题上，十个开放模型的准确率平均比未标记的母语支持题低 32.1 个百分点。英语样本含 348 个句对，只有 19 个标签不受支持，法语和荷兰语提供补充证据。构建测评时可保留过滤前后难度分布，并对 LLM 建议删题的集合做人类抽检。 [论文 §3–5、表 1–2、Limitations](https://arxiv.org/html/2611.13788v1#S4)

### 删除跨轮推理历史的效果随模型显著变化

**11-09 20:51:21 UTC 提交｜arXiv:2611.13223v1。** 七个开放推理模型在修改后的 DeepSeek Harness 下比较保留、跨轮删除和会话中途删除。跨轮删除让 Baichuan-M2-32B 成功率从 51.8% 降到 0.5%，Qwen3-32B 从 40.3% 变为 40.5%，同为 32B。R1-Distill-14B 中途删除时重复工具调用会话占比由 18.1% 升至 68.7%。各模型任务子集及有效会话数有差异，结果来自同类多轮工具任务。模型切换和历史压缩回归可分别记录成功率、有效运行分母及重复调用。 [论文 §3–5、表 3–6](https://arxiv.org/html/2611.13223v1#S4)

### ARE-bench 在真实研究工作区检查延续任务是否完整交付

**11-10 17:21:31 UTC 提交｜arXiv:2611.14364v1。** 基准从已发生的研究会话恢复中途工作区、失败尝试与待办，然后按任务的逐项清单评分。75 道任务里有 8 道 GPU 任务，三次运行的 GPT-6.1-Sol 清单项通过率为 92.72%，全部适用项通过的任务率为 66.22%。模型使用不同 harness，裁判由 GLM-5.3 担任，排行榜混合了模型、工具封装和评审影响。复用该协议时，可把训练能恢复运行和论文中的坐标、标签、目标函数是否保真分成验收项。 [论文 §3–4、表 3–5、表 13、Limitations](https://arxiv.org/html/2611.14364v1#S4)

<a id="engineering"></a>
## 开源工程

### Megatron 修复 GTP 下 SwiGLU checkpoint 的行归属

**11-11 19:00，`main` 默认分支。** 普通 TP 的保存逻辑假定本地 FC1 总含 gate 与 up 两半。GTP 会在 TP 内连续切片，使某些 rank 只持有其中一半。checkpoint factory 现在用 TP/GTP 坐标映射全局 gate/up 偏移，运行时权重布局保持原样，当前覆盖无 padding 且本地行数为偶数的分片。迁移前用带不同 gate/up 哨兵值的 checkpoint 验证 GTP2 到 TP-only 的读回。 [Megatron-LM #9265 `75e0865aed`](https://github.com/NVIDIA/Megatron-LM/pull/9265)

### Megatron 为每次重算隔离 DSA top-k，避免在单点 DP 轴重复分配权重

**11-11 19:00–11-12 01:25，`main` 默认分支。** 非 packed 的 DSA 索引共享原先把 holder 放在可能跨 forward 复用的 mask 或 config 上，后一批可能覆盖早一批重算所需的索引。新逻辑把共享状态与 recompute 上下文绑定。MFSDP v2 则在 DP 轴大小为 1 时让 unsharded view 直接别名底层权重，通过存储别名检查保留持久权重及 MXFP8 scales。回归分别覆盖交错 forward/backward 以及 singleton 与非 singleton 轴混合的参数释放。 [Megatron-LM #6793 `ae20eba3a7`](https://github.com/NVIDIA/Megatron-LM/pull/6793) · [Megatron-LM #8828 `268cabb32e`](https://github.com/NVIDIA/Megatron-LM/pull/8828)

### Megatron 将大池地址乘积升为 int64，并把并行 wrapper 配置收归 cfg.ddp

**11-11 13:24–11-11 19:00，`main` 默认分支。** paged KV append 与 Mamba state/conv 更新用 64 位计算 slot、row 或 block 的偏移，修复池超过 2^31 元素时的静默回绕。DDP/MFSDP wrapper 的构造、bucket、通信 hook 和 checkpoint backend 读取统一的 cfg.ddp，Torch FSDP2 保留自身 schema，optimizer 仍管其拥有的 overlap。大池回归应比对 log-prob 和 KV 内容，外部 Python launcher 则应查是否直接读写过旧训练字段。 [Megatron-LM #9087 `7a01f007fb`](https://github.com/NVIDIA/Megatron-LM/pull/9087) · [Megatron-LM #8205 `5bce4b3f7c`](https://github.com/NVIDIA/Megatron-LM/pull/8205)

### verl 为 FSDP actor 加入与 vLLM 对齐的 batch-invariant 路径

**11-11 11:14，`main` 默认分支。** `batch_invariant` 默认为关，启用后训练进程安装 vLLM 的 matmul、softmax、mean、RMSNorm 等确定性覆盖，并复用采样端公式，先对全词表做 FP32 log_softmax 再 gather。作者在同一组 4096×152064 BF16 logits 上对照了原有公式的差异。先固定采样设置与权重，区分公式自身的逐位相等和完整 rollout 对 actor 的逐位相等。 [verl #8026 `2b4ed6826a`](https://github.com/verl-project/verl/pull/8026)

### verl 的 Ascend GRPO 配方保持 reference 权重常驻

**11-11 11:05，`main` 默认分支。** Qwen3.5-35B-A3B 配方把 `ref.megatron.param_offload` 关掉，并显式列出 rollout/ref 继承的动态 batch 选项。PR 在 8 张 A5 上报告 reference log-prob 从每步 23.5 s 降至 9.5 s，并注明 16 卡 A3 配方也已验证。复测应同时记录显存余量、offload 开关和 reference 这一段的耗时。 [verl #8618 `14535ee1d5`](https://github.com/verl-project/verl/pull/8618)

### DeepSpeed 修复 ZeRO-2 累加器释放和 ZeRO-3 提前停机

**11-11 09:25–11-11 11:53，`master` 默认分支。** ZeRO-2 CPU offload 在 request 边界先发布 compact partition 梯度，只有 host dtype 能逐位恢复运行和时才释放设备累加器，下一次 backward 重新装入。ZeRO-3 eager rollout 在 optimizer 的 DP 参数收集组同步 EOS，等最晚 rank 结束并保留各序列首个 EOS。graph rollout 走独立路径。用多次 backward、不同累加 dtype 和各 rank 不同 EOS 的例子验证。 [DeepSpeed #9044 `f9fe4df347`](https://github.com/deepspeedai/DeepSpeed/pull/9044) · [DeepSpeed #8390 `8a6b53fea9`](https://github.com/deepspeedai/DeepSpeed/pull/8390)

### DeepSpeed 按每 rank 的 prefill KV 形状建立 AutoTP rollout cache

**11-11 17:20，`master` 默认分支。** AutoTP rank 只持有本地 KV heads。静态图缓存以前从全局 config 提前分配，复制本地 head 时会报 shape 错，单本地 head 时还可能静默广播。现在图和缓存路径按 rank 自己的 prefill KV 建立。复测应覆盖 TP>1 且每 rank 只有一个 KV head 的 GQA，逐 token 对照非 graph 路径。 [DeepSpeed #9008 `2dabbbe303`](https://github.com/deepspeedai/DeepSpeed/pull/9008)

### TRL 修复 VESPO 的 token_mask 与 DPO 的 WPO×SFT

**11-11 20:10–11-12 08:13，`main` 默认分支。** VESPO 将重要性 ratio 为零的 token 同时从序列权重和 loss 剔除。此前 log(0) 的 clamp 项会几乎否决整条序列，被拒 token 本身仍带梯度。DPO 的 WPO 则改用非原位乘法，避免 `sft` loss 从 scalar expand 成共享存储后原位写报错。回归可用单个极端 ratio 与 `["sigmoid","sft"]` 混合配置检查有效 token 的梯度和首步 loss。 [trl #8108 `861442d102`](https://github.com/huggingface/trl/pull/8108) · [trl #7651 `acf8506423`](https://github.com/huggingface/trl/pull/7651)

### TRL 在 optimizer step 结束发布 rollout token 额度

**11-11 19:53–11-12 01:25，`main` 默认分支。** 异步 GRPO/蒸馏在每步结束立即公布按实际消费重算的 token allowance，消除权重同步、日志和 checkpoint 期间沿用上一额度的窗口。AsyncGRPO 将 fused LM head 挂到传入 PeftModel 的实际 base model，并在初始化拒绝无法经 vLLM 权重同步携带的 prompt-learning adapter。DeepSpeed ≥0.19.3 的 PEFT+ZeRO-3 组合恢复尊重 `use_reentrant`，按版本和 adapter 类型做首步验证。 [trl #8243 `e1f5615e50`](https://github.com/huggingface/trl/pull/8243) · [trl #8207 `3ca0435c92`](https://github.com/huggingface/trl/pull/8207) · [trl #8120 `bb955a6e4a`](https://github.com/huggingface/trl/pull/8120)

### TorchTitan 在 RL 中恢复训练 router gate

**11-12 04:44，`main` 默认分支。** RL recipe 只冻结 moving expert bias、关闭路由均衡辅助损失，router gate 恢复梯度。作者的四层 Qwen3.5-35B-A3B 缩小版跑 60 步，trainer/generator log-prob 差变化约 −1%，两次冻结 gate 的对照约 −8% 和 −15%。检查 gate 梯度、专家负载分布和训推 log-prob，再放大到完整模型。 [torchtitan #5730 `70859a06ba`](https://github.com/pytorch/torchtitan/pull/5730)

### TorchTitan 降低 Muse Spark Open LoRA SFT 的权重与确定性成本

**11-11 10:16–11-12 01:07，`main` 默认分支。** SLAKE 配方可只挂 decoder LoRA，也可同时挂视觉塔与 decoder。图像进 user turn，只有回答 token 计入 SFT loss。冻结参数按 param_dtype 存储，Dist-MoE expert LoRA 物化有效权重时去掉一次完整权重副本。确定性路径给 FlexAttention block mask 补 dQ 写序，并处理 NaViT 位置插值反向。用同一图文批次核对 loss mask、显存和重复反传结果。 [torchtitan #5885 `717fc32608`](https://github.com/pytorch/torchtitan/pull/5885) · [torchtitan #5893 `0c9b11c2a3`](https://github.com/pytorch/torchtitan/pull/5893) · [torchtitan #5894 `847fabbca0`](https://github.com/pytorch/torchtitan/pull/5894) · [torchtitan #5895 `8958ba23a3`](https://github.com/pytorch/torchtitan/pull/5895) · [torchtitan #5891 `fd2727957c`](https://github.com/pytorch/torchtitan/pull/5891) · [torchtitan #5892 `8a2c7dc951`](https://github.com/pytorch/torchtitan/pull/5892)

### TorchTitan 修复 SGLang rollout 接口漂移与 MoE SwiGLU 重编译

**11-12 02:53–11-12 06:24，`main` 默认分支。** SGLang engine 接收 generator 传入的 `vision_processor` 和新的 `Prompt` 结构，恢复文本 rollout 的构建与入队。视觉请求仍须按 backend 实际支持验证。AllToAll 专家 token 数每步变化，fused SwiGLU 把 `NUM_ROWS` 从编译期常量改成运行时值。PR 的 8× TPS 是一个重编译密集配置的观测，宜先看 warmup 后编译计数和稳态 TPS。 [torchtitan #6024 `31ceb63831`](https://github.com/pytorch/torchtitan/pull/6024) · [torchtitan #6030 `b8cebbfb74`](https://github.com/pytorch/torchtitan/pull/6030)

### NeMo RL 将原生 vLLM rollout worker 接入 Dynamo

**11-11 15:54，`main` 默认分支。** SingleController 的 `policy.generation.dynamo: true` 让 NeMo RL worker 通过 Dynamo 通用 backend SDK 注册自己的 AsyncLLM，由 managed frontend 路由。refit、暂停、量化和 speculative decoding 继续用同一 worker 路径。迁移应实际跑权重更新、暂停/恢复与故障重启，并比较请求是否始终带着预期权重版本。 [RL #4905 `f488a0c6eb`](https://github.com/NVIDIA-NeMo/RL/pull/4905)

### NeMo RL 补 DSpark 的 tiled loss 与 SingleController PPO 序列过滤

**11-11 22:33–11-12 06:18，`main` 默认分支。** DSpark loss 增加分块 hard-CE、total variation 与 confidence 目标，处理 split vocabulary 并保留 target logits 和输出头的 stop-gradient。此合入覆盖目标、head/provider 和分布式测试，训练循环配置仍在后续。SingleController PPO 可在 loss 内按 log-prob 误差筛序列，限定单个 PPO epoch 且继续拒绝 KL-in-reward。SC 的 DAPO overlong 和截断惩罚覆盖普通 token capture。先分别测 TP 数值、PPO 过滤比例与 reward shaping 前后分布。 [RL #3726 `e6aae2a79b`](https://github.com/NVIDIA-NeMo/RL/pull/3726) · [RL #5130 `27fac5f3d6`](https://github.com/NVIDIA-NeMo/RL/pull/5130) · [RL #4882 `d51b2842e8`](https://github.com/NVIDIA-NeMo/RL/pull/4882)

### Transformers 修复 Ministral 3 / Mistral 4 packed FlashAttention 边界

**11-11 11:24–11-12 02:27，`main` 默认分支。** 模型将为 query scaling 单独取出的 `position_ids` 传回 attention interface，使 FlashAttention 能识别 padding-free packing 的文档边界。Step3.7 的 vision RoPE 同时纠正 W/H 频带对应，避免非方形图像沿错误空间通道旋转。对 packed attention 比较每条文档单跑输出，对视觉用非方图和上游参考输出做 parity。 [transformers #50450 `a065282cd6`](https://github.com/huggingface/transformers/pull/50450) · [transformers #50547 `335dd96bdd`](https://github.com/huggingface/transformers/pull/50547)

### Transformers 统一导出产物对象与 Hub 工件

**11-11 23:04，`main` 默认分支。** 导出器返回 `ExportedModel`，可用 `.program` 取得 backend graph，也可保存产物、`export.json` 元数据和 config，再从目录或 Hub 载入。依赖直接 program 返回值的调用需要迁移。选一个动态 shape ONNX 模型做本地与 Hub round-trip，并在运行时核对导出前后的输出。 [transformers #50029 `28ec08c469`](https://github.com/huggingface/transformers/pull/50029)

### vLLM 用完整调度 token 预算测 KV reserve，并在 graph capture 前释放 warmup

**11-12 07:24–11-12 08:55，`main` 默认分支。** Extensible KV cache 的 warmup 覆盖多请求填满 `max_num_batched_tokens` 的情形，即使单请求的 `max_model_len` 更小，并释放 graph capture 前的临时块。配置了 `gpu_memory_utilization` 时，统一视为引擎占用的硬上限并增加剩余显存提示。用大视觉 encoder graph、单请求配置和长短混合的满预算 step 测启动峰值与稳态余量。 [vllm #66731 `2199d6f488`](https://github.com/vllm-project/vllm/pull/66731) · [vllm #66523 `3319c629a1`](https://github.com/vllm-project/vllm/pull/66523) · [vllm #66704 `63980fbf1e`](https://github.com/vllm-project/vllm/pull/66704)

### vLLM 修复 pipeline stage 间 graph padding、计划 buffer 与残差 dtype

**11-11 14:43–11-12 06:59，`main` 默认分支。** MRV2 的 NCCL PP 接收端先拷贝发送端实际行数，再按本地 graph 补齐。FlashInfer metadata builder 在复写 pinned plan 前等前一次复制完成。Step5 的 NVIDIA 实现在开启 FP32 residual 后用 FP32 中间接收缓冲，作者在一组 PP1/PP2 配置中对齐了 418 个 prompt log-prob。建议用相邻 stage 的 graph mode 不同且 interleave 打开的场景对照。 [vllm #66465 `3f4fbd6fc9`](https://github.com/vllm-project/vllm/pull/66465) · [vllm #64165 `73a1377b6b`](https://github.com/vllm-project/vllm/pull/64165) · [vllm #65755 `078f855f9c`](https://github.com/vllm-project/vllm/pull/65755)

### vLLM 的 Transformers backend 补齐 CPU offload、M-RoPE 与 MoE

**11-11 16:44–11-12 03:02，`main` 默认分支。** decoder 层逐个交给 offloader，使 `--cpu-offload-gb` 对 Transformers 模型和其中的专家生效。M-RoPE 的多轴 positions 对齐 graph 输入。Qwen2-MoE 转用 Transformers 建模后端。直接馈入 down projection 的 GLU 可与其输入量化融合，覆盖 NVFP4 路径。迁移时同时对照原生后端、eager 与 graph，并记录逐层加载峰值。 [vllm #66419 `b32a87094a`](https://github.com/vllm-project/vllm/pull/66419) · [vllm #66833 `8d902b4137`](https://github.com/vllm-project/vllm/pull/66833) · [vllm #66827 `8d5b613174`](https://github.com/vllm-project/vllm/pull/66827) · [vllm #66809 `a9185fc5c5`](https://github.com/vllm-project/vllm/pull/66809)

### vLLM 用 TP 同步 DSpark 验证，并在 DCP 复制 sliding-window draft KV

**11-11 09:58–11-12 06:46，`main` 默认分支。** Adaptive DSpark 将验证预算决策在 TP rank 间对齐，防止 rank-local stale confidence 使形状或执行分支不一致。窗口 MLA draft 在 DCP 各 rank 保持完整副本，NIXL 从一个 prefill rank 拉 draft cache，target KV 仍分片。DFlash 直接按 packed block stride 写入 slot。GLM-5.3-Flash DFlash 继承 FP8 KV，在 TP4、48 个 2048-token draft window 的示例每卡少约 790 MiB。回归应联合压测 DCP/P-D、TP>1 和 KV dtype。 [vllm #62769 `80ac42b811`](https://github.com/vllm-project/vllm/pull/62769) · [vllm #63796 `665357362d`](https://github.com/vllm-project/vllm/pull/63796) · [vllm #66604 `4ad27d8a73`](https://github.com/vllm-project/vllm/pull/66604) · [vllm #66292 `bc11b62ca8`](https://github.com/vllm-project/vllm/pull/66292)

### vLLM 修复 Muse Spark DSpark 的 norm，并融合 SM100 draft 尾部

**11-11 11:23–11-11 23:18，`main` 默认分支。** Muse Spark 的 gated residual RMSNorm 切到 FlashInfer CuTe DSL，绕开 Cake 小批次读取先于 PDL wait 的竞态。SM100 非 specialized decode 路径融合 draft norms 与 Markov sampling 尾部，作者在 B300、TEP4、C48、140K 输入测得 TPOT 下降 2.3%–4.4%，依 adaptive verification 开关而变。先核对 TP1+EP 的精度，再分开测 specialized/普通路径。 [vllm #66986 `1c76375042`](https://github.com/vllm-project/vllm/pull/66986) · [vllm #66764 `595b1514e8`](https://github.com/vllm-project/vllm/pull/66764)

### vLLM 可让 Mooncake PD 与 store 共用 TransferEngine

**11-11 09:24–11-12 04:18，`main` 默认分支。** Mooncake store 通过 opt-in 配置复用 worker 中 PD connector 的 TransferEngine，减少第二次 RDMA 初始化。可选 lazy offload 从 GPU free queue 选复用前缀，在完整 attention 链闭合后发出并固定源块直至传完。LMCacheConnectorV1 转发 shutdown，避免后台线程拖到强杀。应测两条传输路径同时运行以及 SIGTERM、重启和未完成 transfer 的资源回收。 [vllm #65943 `6e3367c33a`](https://github.com/vllm-project/vllm/pull/65943) · [vllm #61797 `7f079220d1`](https://github.com/vllm-project/vllm/pull/61797) · [vllm #64324 `f7be53ae74`](https://github.com/vllm-project/vllm/pull/64324)

### vLLM 修复 FlashInfer query scale 和多模态 PrefixLM

**11-12 00:05–11-12 02:00，`main` 默认分支。** 只有 FP8 query 才乘 checkpoint 的 q_scale，避免给 BF16/FP16 query 的 softmax scale 误加倍率。多模态 batch prefill 可走 FlashInfer 原生 mm-prefix wrapper，每 query 传前缀范围，在 tile 内求因果滑窗与多模态前缀的并集，避免显式 Q×KV mask。用非 1 q_scale 和图文混合的滑窗边界做后端对照。 [vllm #64329 `fe3ed684ce`](https://github.com/vllm-project/vllm/pull/64329) · [vllm #46558 `c3d1ce62b0`](https://github.com/vllm-project/vllm/pull/46558)

### vLLM 扩展 DeepSeek/GLM 的 token sharding，并给 Engram 改 lookup tile

**11-11 19:43–11-11 19:50，`main` 默认分支。** GLM-5.3 / DeepSeek-V3.2 的 ShardQ 将残差、norm 和复制的 A projection 延伸到 token 分片，在后续 TP attention 前才聚合较小的投影。DeepSeek-V4.1 Engram 大查表用 128-row tile 并加宽 packed load，还支持跨引擎拆表。用真实并发分布分别核对通信量、查表 miss/延迟和低并发退化。 [vllm #66549 `1c4211fb29`](https://github.com/vllm-project/vllm/pull/66549) · [vllm #66552 `2acfc19eeb`](https://github.com/vllm-project/vllm/pull/66552)

### vLLM 校正 batch-invariant pooling 与 Blackwell MoE 调参

**11-11 23:08–11-12 07:26，`main` 默认分支。** 开启 batch-invariant 时，MEAN pooling 按序列切片求和，替代非确定性的 GPU `index_add_`。Blackwell MoE 为指定专家形状选固定调优配置，仍维持不随 batch 改变的选择。DP MoE 则跳过未暂停 rank 的重复 resume collective，暂停和新加入 EP rank 仍同步。先在单独、拼批与重复 resume 三种输入下查输出和进度。 [vllm #60035 `3bc0f43088`](https://github.com/vllm-project/vllm/pull/60035) · [vllm #63627 `d74f39c5a6`](https://github.com/vllm-project/vllm/pull/63627) · [vllm #55229 `5be8acb7f2`](https://github.com/vllm-project/vllm/pull/55229)

### SGLang 将 Kimi K3.1 DFlash2 和 per-M GEMM 从日零分支带回主线

**11-11 12:53–11-12 08:54，`main` 默认分支。** K3.1 的 sliding-window MLA DFlash2 draft 接通 DCP 的 CuTeDSL 路径。decode 和 verify 的 dense BF16 GEMM 使用随 M 区间选择的 TGV tactic，部分 attention GEMM 走侧流。KDA 加入 ModelOpt MXFP8 的 b_proj/f_a_proj 加载，视频接通逐帧采样和 MoonViT 时序处理。分别对短窗口越界、TP>1 scale shard、视频采样上限和小 M decode 做对照。 [sglang #49621 `426cca6aa7`](https://github.com/sgl-project/sglang/pull/49621) · [sglang #49478 `6685123903`](https://github.com/sgl-project/sglang/pull/49478) · [sglang #49851 `7352f1aa85`](https://github.com/sgl-project/sglang/pull/49851) · [sglang #49458 `affb3eb502`](https://github.com/sgl-project/sglang/pull/49458)

### SGLang 在 ReplaySSM 接入 DFlash/DSpark，并在线估计草稿接受率

**11-11 09:15–11-12 05:30，`main` 默认分支。** GDN verify 接通 ReplaySSM ring，草稿验证概率的 top-k 改用 radix，DSpark 可用服务中拟合的接受率替代训练好的 confidence head。EAGLE 的恒定 query 长度显式传给 FlashInfer graph planner，Triton split-KV verify 补 attention sinks。各项有独立适用条件，组件 PR 的旧分支 GPU 结果需在当前 main 重测。比较 acceptance、校准误差、状态回放一致性及 TPOT。 [sglang #46185 `1017f57660`](https://github.com/sgl-project/sglang/pull/46185) · [sglang #46186 `884d29b207`](https://github.com/sgl-project/sglang/pull/46186) · [sglang #44588 `e11c9583e6`](https://github.com/sgl-project/sglang/pull/44588) · [sglang #44115 `c8c0941f54`](https://github.com/sgl-project/sglang/pull/44115) · [sglang #47513 `0f0ed7f24f`](https://github.com/sgl-project/sglang/pull/47513)

### SGLang 稀疏索引减少 host 同步、合并写放大和长上下文中间量

**11-11 18:22–11-12 08:14，`main` 默认分支。** DCP top-k 将候选在 shared memory 重排后做合并写，小 rank 数用寄存器 fallback。eager metadata 减少 rank 分歧路径的 host sync。SM100 的可选 LiteTopK 针对 DSA FP8 和 DeepSeek-V4.1 FP4 prefill。另按 batch 大小选 expand_prefill token block，并改用动态 shared memory 为 cluster top-k 分配缓冲。用多个 DCP rank 与长短混合 prefill 分开测通信、索引开销及 top-k 一致性。 [sglang #49922 `2c597c9225`](https://github.com/sgl-project/sglang/pull/49922) · [sglang #46830 `75e8d543dc`](https://github.com/sgl-project/sglang/pull/46830) · [sglang #47469 `4a99de617f`](https://github.com/sgl-project/sglang/pull/47469) · [sglang #49499 `fc6d3b6f91`](https://github.com/sgl-project/sglang/pull/49499) · [sglang #49476 `56e38dfe0f`](https://github.com/sgl-project/sglang/pull/49476)

### SGLang 修复 PP P/D abort 与 beam scheduler 容量

**11-11 13:58–11-11 22:10，`main` 默认分支。** P/D prefill 的 PP event loop 主动处理已在途的 chunked-prefill 取消，启动时也拒绝会令 PP 各 stage 收包频率错位的 recv interval。beam 请求的 leader 准入后立即保留其成员所需的 request rows。KV metrics backpressure 与 prealloc token 统计各补回归。用中途取消、请求池将满、metrics consumer 慢的组合测进度和释放。 [sglang #49722 `5ff617c12b`](https://github.com/sgl-project/sglang/pull/49722) · [sglang #49756 `e0e4d2bffc`](https://github.com/sgl-project/sglang/pull/49756) · [sglang #46855 `6047c17aeb`](https://github.com/sgl-project/sglang/pull/46855) · [sglang #46820 `f39ac6b55b`](https://github.com/sgl-project/sglang/pull/46820) · [sglang #44236 `bf8c994ca8`](https://github.com/sgl-project/sglang/pull/44236)

### SGLang 将多模态路由身份与实际 KV 复用身份分开

**11-11 10:54–11-11 15:54，`main` 默认分支。** KV events 可按 block span 发出调用方提供的多模态路由标签，实际复用仍沿用引擎内部的图像标识。嵌入 lookup 使用合法词表索引，同时保留 forward batch 供模型后续使用的多模态位置标记。llguidance 拒绝 token 前先检查，避免污染之后的回退与 mask。Rust router 补健康探针及 worker info 兼容。回归包括相同标签不同图像、回放和无健康 worker。 [sglang #45737 `3ff87c57f5`](https://github.com/sgl-project/sglang/pull/45737) · [sglang #48385 `936d12c3cf`](https://github.com/sgl-project/sglang/pull/48385) · [sglang #49655 `10163c2811`](https://github.com/sgl-project/sglang/pull/49655) · [sglang #49727 `b2568f706b`](https://github.com/sgl-project/sglang/pull/49727)

### SGLang 替换 MXFP4 的 wheel 内 vendored triton_kernels

**11-11 09:14–11-12 02:33，`main` 默认分支。** 去掉 `triton_kernel` MoE runner 与随 wheel 带的 `triton_kernels`，保留的 Triton runner 改用 JIT kernel 做 MXFP4 专家反量化。GPTQ/AWQ kernel 按逻辑 K 处理尾块并验证 scale 网格。FlashInfer 启动调优改用有变化的 token，避免全零 dummy 输入把负载集中到同一批专家。检查旧启动 flag、非整 K 分片及实际路由分布。 [sglang #49660 `df4ec2e074`](https://github.com/sgl-project/sglang/pull/49660) · [sglang #49661 `ed15b28dcc`](https://github.com/sgl-project/sglang/pull/49661) · [sglang #49662 `62eab14422`](https://github.com/sgl-project/sglang/pull/49662) · [sglang #49337 `2dcc93b624`](https://github.com/sgl-project/sglang/pull/49337) · [sglang #49497 `07a6c95548`](https://github.com/sgl-project/sglang/pull/49497)

### SGLang 修复 MiniMax-H3 公共 checkpoint 和图像编辑预处理

**11-11 16:54–11-11 17:02，`main` 默认分支。** MiniMax-H3-Image 适配公共快照中本地 text_encoder 路径并保留 H3 视频 conditioning。Qwen-Image-Edit 与 FLUX.2 的 VAE encode 避免仅因预处理后的图像全为正而再做归一化。Anima 在 serial CFG 的两支间复用每 step 相同的 85 个 timestep AdaLN 投影。用公共快照做首个生成，另用全亮图像和多步 CFG 检查输出与 kernel 次数。 [sglang #49574 `6be0d75ae0`](https://github.com/sgl-project/sglang/pull/49574) · [sglang #49570 `23f12b7cb6`](https://github.com/sgl-project/sglang/pull/49570) · [sglang #49504 `68339b5511`](https://github.com/sgl-project/sglang/pull/49504)

### Miles 接入 Muse Spark Open 的训练、转换与 rollout 刷权重

**11-11 12:04–11-12 06:40，`main` 默认分支、`sglang-miles` 特性分支。** Miles 增加 Muse Spark Open 的模型插件、Megatron/HF 转换、模板和 rollout 配方。SGLang 的 `sglang-miles` 分支在权重同步后刷新 Muse Spark 派生权重，修复同机 rollout 首次 refit 后 sampled-token log-prob 接近均匀分布的问题，并阻止不匹配 prefix 行的 `kv_b_proj` LoRA。Miles 默认把 GLM-5 的该投影排除出 LoRA。需要固定匹配的两个分支，验证首轮生成和第一次刷权重后的 log-prob。 [miles `706c6106ca`](https://github.com/radixark/miles/commit/706c6106cafe24bad5eba7cceb09ee4f24c2d1b3) · [miles `0d8e220a0e`](https://github.com/radixark/miles/commit/0d8e220a0e16a007433db94f4adeda576836aac8) · [sglang #49688 `94c5c716ee`](https://github.com/sgl-project/sglang/pull/49688) · [sglang #49848 `fd017154ad`](https://github.com/sgl-project/sglang/pull/49848)

### Miles Diffusion 增加 MiniMax H3 四步 DMD2 与 Flow-GRPO score centering

**11-11 21:32–11-11 23:45，`main` 默认分支、`sglang-miles` 特性分支。** DMD2 随机截断 rollout 的去噪步，把学生样本送 fake-score critic，学生用 real/fake score 差构造分布匹配梯度。H3 四步配方带评测媒体和 DMD2 指标，训练无需媒体 reward 时可跳过 VAE decode。实验性的异步 Flow-GRPO score centering 以 Gaussian 下的截断 ratio 期望修正梯度，保持 loss 值与 clip 决策。应分别检查随机停步、critic 更新频率和实际 rollout/actor mismatch，SGLang controller 接口取自特性分支。 [miles_diffusion `68d3346936`](https://github.com/radixark/miles_diffusion/commit/68d3346936dec4a3b4307d339004d18e380d6e95) · [miles_diffusion `23a795a949`](https://github.com/radixark/miles_diffusion/commit/23a795a94973fca489afb6089bf4e038f8f5508f) · [miles_diffusion `4f9ceaec6c`](https://github.com/radixark/miles_diffusion/commit/4f9ceaec6cfcf4112d35614fee81d0eae54f6773) · [miles_diffusion `b1e4e6e4d0`](https://github.com/radixark/miles_diffusion/commit/b1e4e6e4d034e7c7473f1007a1087b0309a4018e) · [miles_diffusion `5823c16d02`](https://github.com/radixark/miles_diffusion/commit/5823c16d02ad64133bf857e9b619b58992c0d593) · [sglang #49440 `5fb7b3c442`](https://github.com/sgl-project/sglang/pull/49440)

### Miles Diffusion 从 output store 按需取回轨迹和生成结果

**11-11 14:02–11-12 03:58，`main` 默认分支、`sglang-miles` 特性分支。** 启用 Mooncake output store 时，rollout 传逐样本 slice 引用，训练 shard 在消费前取回并在训练后释放。启用输出仓的 SGLang 特性分支将训练生成结果以引用返回，评估仍 inline。记录所有 slice 的生命周期和中止/重试后的 store 容量，避免仅看 trainer 的 host 内存。 [miles_diffusion `01869a40b9`](https://github.com/radixark/miles_diffusion/commit/01869a40b945890c19eb8f30c66a571fb9289d2c) · [miles_diffusion `63ed7aa950`](https://github.com/radixark/miles_diffusion/commit/63ed7aa950314c97030ea7a91788fd18e148c501) · [miles_diffusion `f09118a0eb`](https://github.com/radixark/miles_diffusion/commit/f09118a0ebd249d28d5fbe5b170bd2c93570b69e) · [miles_diffusion `f853234976`](https://github.com/radixark/miles_diffusion/commit/f853234976976bb10714beda51196c552c80c6e0) · [sglang #49877 `18a44326f6`](https://github.com/sgl-project/sglang/pull/49877)

### TensorRT-LLM 在 KVCM2 做稀疏 decode KV offload

**11-11 14:49–11-11 21:34，`main` 默认分支。** decode-only worker 将 GLM-5.2 DSA 的 MLA latent 和 DeepSeek-V4 Flash 的 C4 KV 历史放在 pinned host memory，每步按索引取回 GPU 工作集，索引和滑窗 cache 常驻 GPU。P/D coordinator 另修复空闲时超时的旧式接收不退休与 transfer 失败后 KV 持有。对照 resident 与 offload 的显存、PCIe 流量和 TPOT，另注入 idle decode、失败或超时检查资源回收。 [TensorRT-LLM #21350 `15e973a7cf`](https://github.com/NVIDIA/TensorRT-LLM/pull/21350) · [TensorRT-LLM #21909 `bf13dc5ceb`](https://github.com/NVIDIA/TensorRT-LLM/pull/21909)

### TensorRT-LLM 为 DSA prefill 提供不落 logits 矩阵的精确 top-k

**11-11 09:49–11-11 18:27，`main` 默认分支。** SM100/SM103 单请求 prefill chunk 可开启 `TRTLLM_THROWAWAY_TOPK=1`，CuTe DSL 先估 score 下界再过滤候选、完成精确 top-k，绕开完整 Q×K logits 写出。Qwen4 split-K mailbox 与 HC decode 的 cluster 等待也加入 acquire fence，现有补丁证据含 CPU 源码检查。先对照完整 logits 路径验证相同 top-k，再用多请求确认 fallback 和硬件端同步。 [TensorRT-LLM #21892 `5626ccfa7d`](https://github.com/NVIDIA/TensorRT-LLM/pull/21892) · [TensorRT-LLM #21853 `c523c03210`](https://github.com/NVIDIA/TensorRT-LLM/pull/21853)

### TensorRT-LLM 避免 spec-decode 按 batch 形状重编译并修复 n-gram 候选

**11-11 12:25–11-11 23:26，`main` 默认分支。** spec-decode 对请求数、行偏移及对齐随 batch 改变的实参取消相应 Triton specialization，减少 warmup 后全 replica 阻塞编译。NGramPoolManager 在候选快照上删除短匹配，single-match 模式只保留一个。PrimTS full-context decode 路径更新到新的 compile/runtime ABI。对比 Kimi K3/DSpark 请求进出时的编译次数与 TPOT 峰值，再核对 n-gram 行为和 TP2/4/8 完整上下文。 [TensorRT-LLM #21666 `4045bd8cca`](https://github.com/NVIDIA/TensorRT-LLM/pull/21666) · [TensorRT-LLM #20487 `27b44ebba9`](https://github.com/NVIDIA/TensorRT-LLM/pull/20487) · [TensorRT-LLM #22050 `1955082fbb`](https://github.com/NVIDIA/TensorRT-LLM/pull/22050)

### TensorRT-LLM 扩展 Muse Spark 服务及 MiniMax-M3 混合量化加载

**11-11 11:56–11-12 06:03，`main` 默认分支。** 服务层根据 muse_spark 类型使用对应推理和工具解析，`reasoning_effort` 接受新的两档，并为 Muse Spark 原生工具接通解析。MiniMax-M3 loader 将 compressed-tensors 包内的 MXFP8 base 与 NVFP4 routed experts 展开为逐层规则，对冲突的 fused component 和不支持的 LM head 量化显式报错。先核对流式工具输出，再在真实混合 checkpoint 检查模块量化表。 [TensorRT-LLM #21049 `2e652e9280`](https://github.com/NVIDIA/TensorRT-LLM/pull/21049) · [TensorRT-LLM #20847 `88e0dcb1e4`](https://github.com/NVIDIA/TensorRT-LLM/pull/20847)

### TensorRT-LLM 增加可与推测解码搭配的 Gumbel-max 水印

**11-11 15:55，`main` 默认分支。** PyTorch backend 用密钥和最近 token 派生抽样随机性，检测端持同一密钥寻找统计信号。支持 exact-match 验证及拒绝采样的独立密钥或 target-only 模式。评估应分开量化生成质量、短输出检出率、误报及不同 speculative 算法的开销。 [TensorRT-LLM #21047 `e8e94448bb`](https://github.com/NVIDIA/TensorRT-LLM/pull/21047)

### PyTorch 增加 functional MXFP8 量化与 cuBLASLt grouped scaling

**11-11 14:26–11-12 06:50，`main` 默认分支。** `F.quantize_tensor` 与 `F.quantize_tensor_dual` 选择单轴和双轴量化，当前初版支持 RTNE MXFP8，CuTe kernel 的块尺寸更通用。cuBLASLt grouped GEMM 接入 tensorwise/groupwise FP8、Hopper 128 元素 block scale。Blackwell MNxK4 则打包四个 UE8M0 scale，支持 1×32 和 1×128。用非均匀分组及相同数值的不同 scale 布局验证 dispatcher 选路和误差。 [pytorch `8d74659063`](https://github.com/pytorch/pytorch/commit/8d746590637f1c440c3cd0169e92923ea9b3e148) · [pytorch `5a6baa6e65`](https://github.com/pytorch/pytorch/commit/5a6baa6e65c587044f8f1d93b362d6f0c16734c4) · [pytorch `de5c43b34e`](https://github.com/pytorch/pytorch/commit/de5c43b34e158e268238b7303fd0bc4aa9702af7) · [pytorch `ca91b958db`](https://github.com/pytorch/pytorch/commit/ca91b958db0edfa09f481eaefc4b97a5ddb426a1) · [pytorch `f864236ccf`](https://github.com/pytorch/pytorch/commit/f864236ccf8aa76fa1ef8573da7ada1ae73e0ec3)

### PyTorch 给 FSDP 梯度归约增加替换接口

**11-12 02:48，`main` 默认分支。** `FSDPModule.set_custom_all_reduce` 能替换 HSDP 或 replicate_with_fsdp 的梯度复制归约，FSDP 仍管理梯度缩放、累加、stream 和 buffer 生命周期。TorchRec 的 AMD sharded relay 为 SUM/AVG 同步调用提供可选转发，异步与其他 op 仍走原 process group。用实际分片尺寸分别测试 reduce-scatter 和 replica all-reduce，并核对 optimizer parity 与 graph replay。 [pytorch `e188a78898`](https://github.com/pytorch/pytorch/commit/e188a78898276767468cb791d3a55126457e1db5)

### PyTorch 修正嵌套 graph break 的上下文，并启用测试覆盖

**11-12 02:30–11-12 08:40，`main` 默认分支。** Python 3.13/3.14 comprehension 的条件跳转可能缺异常表条目。Dynamo 只在邻居 handler 匹配时补条目，恢复 context 和保存的 loop binding，无法安全重建时走 eager。其他补丁保留 LOAD_METHOD 的 NULL sentinel，向 caller 传播 checkpoint loop 的 break，并在测试套件开启 nested graph break。用带 `with`、try 和数据依赖 filter 的 `torch.compile` 用例查实际 graph 与 context 进入/退出次数。 [pytorch `de66a38847`](https://github.com/pytorch/pytorch/commit/de66a38847cfd3657dc50331af4d16a41f422e91) · [pytorch `28882a2fc3`](https://github.com/pytorch/pytorch/commit/28882a2fc3cb868bd896169c12e8ed0dde113a1f) · [pytorch `a0b613b1db`](https://github.com/pytorch/pytorch/commit/a0b613b1db26294d46442d7695778422f81d2b23) · [pytorch `cfa0c687b4`](https://github.com/pytorch/pytorch/commit/cfa0c687b4ef1d7b9088de7281261dd819f23691)

### PyTorch TCPStore 统一使用 libuv，Inductor 修正融合与输出去重边界

**11-11 10:21–11-12 02:23，`main` 默认分支。** 移除旧 TCPStore backend，`use_libuv=False` 成为 no-op，Windows 和无 TensorPipe 构建也使用 libuv。Inductor 处理 unfused occupancy 为零的 epilogue 评分，保持 no-op 消除前后的两输出身份，且配置 round-trip 保留 `ctas_per_cga`。升级前检查依赖旧 TCPStore 语义的启动器，编译回归保留双输出 alias 与 cluster 参数。 [pytorch `392f184f10`](https://github.com/pytorch/pytorch/commit/392f184f108476e4da04c5de90dc7b53484dd153) · [pytorch `a57cc767aa`](https://github.com/pytorch/pytorch/commit/a57cc767aa83ee7cf2c110f03e6b9da71ac510ff) · [pytorch `2782ed0d62`](https://github.com/pytorch/pytorch/commit/2782ed0d62c209b3fbf98e3593b4abb23792806c) · [pytorch `56eb8da70b`](https://github.com/pytorch/pytorch/commit/56eb8da70b259e4357936ee7bf4745ccf484d0d2) · [pytorch `9942f80619`](https://github.com/pytorch/pytorch/commit/9942f806193d57b7659de920e28c5d7327952511)

### Triton 修正 AMD async wait 范围与解释器 atomic 顺序

**11-11 11:40–11-12 07:13，`main` 默认分支。** AMD TDM 根据控制流中的 pending store/scatter 推断 wait 是否只完成 shared 作用域，减少纯 load 阶段多余 CTA barrier。解释器将 C++ load 或 compare-exchange failure 不接受的 release/acq_rel 规范化，避免启用标准库断言时进程退出。NVIDIA 同窗恢复 `--regAllocOptLevel=2` ptxas workaround。复测时固定 Triton 与 ptxas 版本并检查有写和只读的 wait 路径。 [triton #12648 `58c0b9ae2f`](https://github.com/triton-lang/triton/pull/12648) · [triton #12710 `add426adc3`](https://github.com/triton-lang/triton/pull/12710) · [triton #12717 `4a7b73ca19`](https://github.com/triton-lang/triton/pull/12717)

### FlashAttention 4 beta39 带入 SM100 通用 hd256 backward 与 MXFP8 前向

**11-11 16:12，预发布。** SM100 的 hd256 backward 合并到通用 kernel，接通 PackGQA 并保留 2CTA 不适用时的路径。MXFP8 前向先覆盖 QK，PV 仍 BF16。动态 varlen forward 延迟绑定 tile 并可自重置，padded varlen 也可进入 persistent scheduler。版本为 beta，优先做 varlen Q/K、Q/KV heads 不同和 hd256 反向对照。 [flash-attention fa4-v4.0.0.beta39 `dceb374126`](https://github.com/Dao-AILab/flash-attention/releases/tag/fa4-v4.0.0.beta39)

### FlashAttention 的 aux_tensors 末 tile 改用 clamp

**11-12 07:13，`main` 默认分支。** 自定义 score_mod/mask_mod 可能在尾 tile 读辅助张量。越界索引现在钳位到最后合法下标，替代每元素 fastdivmod。该 tile 的越界分数继续被 mask。用非整 tile 的前后向与多种 score_mod 对照，注意真实边界内索引保持一致。 [flash-attention #3150 `f63c472c4d`](https://github.com/Dao-AILab/flash-attention/pull/3150)

### FlashInfer 接入确定性 BF16 MoE、SM107 MXFP8 输出和 SM12x NVFP4

**11-11 10:05–11-12 07:22，`main` 默认分支。** 统一 MoELayer 提供显式选择的 `BatchInvariantBf16Config`。SM107 CuTe DSL BMM 可接受 strided A/out 并写 MXFP8 数据及 128×4 swizzled UE8M0 scale，供下一 GEMM 直接消费。SM12x NVFP4 runner 接新 kernel 并复用既有 prepared-weight ABI。BF16 2-CTA TGV kernel 加退出 barrier。逐 backend 测单请求与混批的输出、stride/scale 及 CUDA graph+PDL。 [flashinfer #9283 `3db2f474e4`](https://github.com/flashinfer-ai/flashinfer/pull/9283) · [flashinfer #9003 `a2936316aa`](https://github.com/flashinfer-ai/flashinfer/pull/9003) · [flashinfer #6510 `3dfe81de82`](https://github.com/flashinfer-ai/flashinfer/pull/6510) · [flashinfer #9146 `24630ce240`](https://github.com/flashinfer-ai/flashinfer/pull/9146)

### FlashInfer 扩展 SDSA key 容量并减少重复计划和编译

**11-11 09:46–11-12 05:42，`main` 默认分支。** SDSA `group_union_masks` 超过 2^20 keys 后以固定步长扩位图，并暴露随设备、group、top-k 变化的上限。cuDNN 9.24 的特定 paged prefill 组合可复用已准备的计划，FP8 等仍按更高版本 gate。delta-rule DSL 走跨进程磁盘编译缓存。NVLink 单侧同步收敛为 fence/sentinel，移除 counted-writes 路径。用边界形状、冷启动和滚动升级配置分别回归。 [flashinfer #9323 `c5c39b0e87`](https://github.com/flashinfer-ai/flashinfer/pull/9323) · [flashinfer #9351 `cfa0318a13`](https://github.com/flashinfer-ai/flashinfer/pull/9351) · [flashinfer #9357 `36186cb4d2`](https://github.com/flashinfer-ai/flashinfer/pull/9357) · [flashinfer #8770 `a4cd87cd73`](https://github.com/flashinfer-ai/flashinfer/pull/8770)

### LMCache 修复取消后释放、gRPC 识别和 KV events 路由

**11-11 09:39–11-12 02:23，`dev` 默认分支。** MP connector 仅对实际已调度 store 的请求延迟释放，避免在途 load 被取消后仍报告发送完成。ZMQ 客户端发现目标讲 gRPC 时快速报出可行动配置。KV event 由 scheduler 直接订阅和 drain，不再等 worker 输出。用取消、同端口协议切换及 MultiConnector shutdown 验证资源与事件。 [LMCache #5938 `752e01fd71`](https://github.com/LMCache/LMCache/pull/5938) · [LMCache #6544 `81382fbffd`](https://github.com/LMCache/LMCache/pull/6544) · [LMCache #6122 `b134e70270`](https://github.com/LMCache/LMCache/pull/6122)

### LMCache 将预取前缀留在 L1，给 CacheBlend 增加 tags 分区

**11-11 11:06–11-11 20:15，`dev` 默认分支。** 默认 prefetch 把从 L2 读回的热 key 留到 L1 淘汰，避免连续请求反复读取。CacheBlend 以 canonical tags 分隔候选，同 token 不同 tags 走不同命名空间。P2P channel 可在本地注册额外目标区直接读取 peer L1。对比热门前缀复用、标签匹配和 GPU/host 目标内存路径。 [LMCache #6428 `cefba684ef`](https://github.com/LMCache/LMCache/pull/6428) · [LMCache #6201 `c619aa4f4d`](https://github.com/LMCache/LMCache/pull/6201) · [LMCache #6490 `15ea78e172`](https://github.com/LMCache/LMCache/pull/6490)

### Dynamo 拆出 KV router 组件并读取编译器生成的 LPX placement

**11-11 10:07–11-11 13:02，`main` 默认分支。** KV router 拆分为事件、hash、索引、affinity、worker selection、admission 等 crate，原组合入口保留服务和插件注册。Operator 读取编译器产出的 `lpuUnits`，把保留量和 connector 投影给运行时。拒绝同 workload 混用新旧格式。Kimi K3 解析器升级后重复模型索引仍生成独立工具调用 ID。验证启用的 feature、placement revision 和流式多工具路由。 [dynamo #18055 `79692cf58e`](https://github.com/ai-dynamo/dynamo/pull/18055) · [dynamo #18362 `57d7a1dced`](https://github.com/ai-dynamo/dynamo/pull/18362) · [dynamo #18485 `909d4efa64`](https://github.com/ai-dynamo/dynamo/pull/18485)

### llama.cpp 修复 CUDA graph cluster 更新与 CPU attention 累加

**11-11 16:44–11-11 20:00，`master` 默认分支。** CUDA graph 检查 kernel cluster dimensions 变化并重新实例化，避免更新返回成功但继续用旧 cluster。小 batch CPU flash attention 的 V 累加升为 FP32。CUDA Tensor Core 路径补 head dim 40/72，CUDA 还接入 OCP FP8 的初步支持，MoE 与原生 tile-prefill 留作后续。用 Hopper 动态 M、长上下文 CPU 和视觉塔 head dim 分开回归。 [llama.cpp #31284 `2e325dcd57`](https://github.com/ggml-org/llama.cpp/pull/31284) · [llama.cpp #29810 `7f24aacde3`](https://github.com/ggml-org/llama.cpp/pull/29810) · [llama.cpp #30285 `9d7b2d7397`](https://github.com/ggml-org/llama.cpp/pull/30285) · [llama.cpp #28413 `9e9619d030`](https://github.com/ggml-org/llama.cpp/pull/28413)

### KAI Scheduler v0.19.0 把 NUMA 内存组与原地扩容纳入调度

**11-11 20:22，正式发布。** NUMA Memory Manager group 成为调度约束并导出 Pod 实际内存归属。可选原地扩容回收机制为 kubelet 延迟的 resize 腾出节点资源。NvFraction 支持 memory/fraction limits，Pod 级资源按 kubelet 准入口径计费。升级前测试已有 CRD、同节点 resize、NUMA 拓扑约束与抢占条件。 [KAI-Scheduler v0.19.0 `2305acfc34`](https://github.com/kai-scheduler/KAI-Scheduler/releases/tag/v0.19.0)

### FlashInfer v0.7.4 发布可选 batch-invariant 低精度算子

**11-11 10:18，正式发布。** 正式版给 mm_bf16、bmm_fp8、mm_mxfp8、mm_fp4 与 MXFP8/MXFP4/NVFP4 量化提供 `batch_invariant=True`，需受支持的 Blackwell/Rubin 输入并选择自动 backend。Rubin 同时接入 DeepSeek-V4 sparse MLA decode 与 MXFP4 EP MoE，Prims-TS 在 Blackwell 覆盖 Gemma 4 的 512 维 FP8 Q/NVFP4 KV。迁移先逐算子测相同样本在不同 M 下的位模式，再核对完整 trainer/rollout 及设备、形状的支持矩阵。 [flashinfer v0.7.4 `77039051b0`](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.7.4)

### vLLM v0.34.0rc1 纳入 speculative greedy fallback 的 NaN 处理

**11-12 03:37，预发布。** 候选版 tag 指向明确提交。其顶端 cherry-pick 让 torch fallback 把 NaN 当作 −inf，并以小词表用例覆盖 fallback。该候选版用来验证升级兼容性时，应固定 tag 或镜像 digest，优先复测有 NaN 和跨 tile 并列值的 spec-decode logit。 [vllm v0.34.0rc1 `52aeb58065`](https://github.com/vllm-project/vllm/releases/tag/v0.34.0rc1)

<a id="hardware"></a>
## 硬件与软件协同

代码条目标注所核对的合入提交。性能数据沿用各项目的测试条件。

### NVIDIA 与 CUDA

- **cuDNN Frontend 为 SM120 提供 AOT 注意力。** 可选构建将 FROST FP16/BF16 prefill、split-KV 合并及量化 KV decode 的 cubin 通过 C ABI 导出，以 C++ 头文件接口加载。批量、长度与支持的 stride 可在运行时传入，PackGQA 组宽仍由编译期模板决定。项目附图捕获用例。适合分开测首次加载、预热后 TPOT 与 workspace 峰值。 [NVIDIA/cudnn-frontend #1782 · a1b4b8a7c · 11-12 07:26:31](https://github.com/NVIDIA/cudnn-frontend/commit/a1b4b8a7ca3d27601e7e6cb25007ac71ea8db490)

- **Hopper 的 FROST 前向接受分页 KV。** SM90 原生 FP16/BF16 注意力按页表容量确定 launch 上限，以设备上的长度限定有效序列，覆盖文档所列 D128、D256、D512 组合。H100 NVL 有分页确定性用例，分页 D512 仍保持 cuDNN backend 优先。可核查长尾、split-KV workspace 与 fallback。 [NVIDIA/cudnn-frontend #2431 · 8e1910fed · 11-12 04:05:11](https://github.com/NVIDIA/cudnn-frontend/commit/8e1910fed0702b034079bf8e3a71aa7927715ebe)

- **Cake 增加确定性 GDN、KDA 和低精度 MLP 训练原语。** cuDNN Frontend 合入 manifest 运行时和可选 chunk-64 GDN/KDA 前后向，随后接入 grouped GEMM、FP32 router、SwiGLU/MXFP8、causal convolution 与 RMSNorm。初始 GDN 入口要求 packed 文档从零状态起步，暂不接收状态池、checkpoint 或 CP。适合先复查固定种子前后向与混排文档。待整模型吞吐与规模结果。 [NVIDIA/cudnn-frontend #2432 · 7d49bde1a · 11-11 13:54:38](https://github.com/NVIDIA/cudnn-frontend/commit/7d49bde1a3fe8fd7ef55eabcf17ff7ba0c1afbd5) · [NVIDIA/cudnn-frontend #2437 · 1328570b4 · 11-11 13:55:52](https://github.com/NVIDIA/cudnn-frontend/commit/1328570b43bd3f14af3927a40c9fb4473c7b03db) · [NVIDIA/cudnn-frontend #2439 · 17e10b09e · 11-11 14:05:58](https://github.com/NVIDIA/cudnn-frontend/commit/17e10b09eb35160609a8bde298b6931b098aa9d3)

- **SM120 FP8 注意力修订共享内存同步。** FROST 调整 V 尾块及相关共享内存复用，协调生产与消费的 stage 生命周期。项目补充 SM120 D128/D512 的 E4M3/E5M2、BF16 输出重复性与邻居替换检查。可将数值重复性和 racecheck 分开纳入升级回归。 [NVIDIA/cudnn-frontend #2452 · 683f294b4 · 11-11 19:08:06](https://github.com/NVIDIA/cudnn-frontend/commit/683f294b41825e2cb1969c2007bdf118eac191e4) · [NVIDIA/cudnn-frontend #2469 · 2ea6ff880 · 11-11 17:53:28](https://github.com/NVIDIA/cudnn-frontend/commit/2ea6ff88037ddcfece9e5a70034d670958cfc6ef)

- **Transformer Engine 主干更新 NVFP4 布局、累加接口和框架下限。** 融合 grouped MLP 按具体 cuDNN wrapper 能力传 canonical NVFP4 布局，接受 swizzled scale 时可传平坦激活与概率视图。cuBLAS GEMM v2 支持独立的 C 累加张量与 D 输出。主干最低 PyTorch 2.8、JAX 0.5.3，JAX 依赖 cuDNN Frontend 1.31.0 起。升级镜像宜先验证编译、别名与 scale 布局。 [NVIDIA/TransformerEngine #3820 · 6f3b0e82c · 11-11 18:40:57](https://github.com/NVIDIA/TransformerEngine/commit/6f3b0e82c2c2adafb7458e598e2ad9bb4a655b76) · [NVIDIA/TransformerEngine #3811 · d08bc44de · 11-12 04:49:49](https://github.com/NVIDIA/TransformerEngine/commit/d08bc44de042ba2a79e03e600dcf9e3335bac536) · [NVIDIA/TransformerEngine #3788 · 036aca921 · 11-12 04:49:39](https://github.com/NVIDIA/TransformerEngine/commit/036aca9214da3a90aee2e79daa4f99d81ee2f27d)


### AMD ROCm

- **FlyDSL 0.5.0 补齐 GitHub 正式 Release。** 11 月 9 日日报已记 PyPI wheel。本轮增量是 GitHub 版本正文发布，明确 Windows x64 原生构建、JIT 和 AOT 导出，汇总已归档的 FP4/FP6 寄存器布局及磁盘缓存隔离。可用正式 changelog 对照 Windows 构建和缓存冷启动。 [ROCm/FlyDSL v0.5.0 · 正式版 · 11-11 11:45:23](https://github.com/ROCm/FlyDSL/releases/tag/v0.5.0)

- **ATOM 对 Engram 的 UVA 注册跨 DP rank 串行化。** 项目报告四个 DP-attention rank 并发注册约 92 GiB 的只读映射会卡在 KFD。注册和解除注册共用跨进程锁，空洞默认按 1 GiB 分块，可选先锁驻留页。仍需测不同 NUMA 分布下的冷启动、注册失败后的 copy 路径及并发恢复。 [ROCm/ATOM #2932 · 47e0c7075 · 11-11 23:21:10](https://github.com/ROCm/ATOM/commit/47e0c707571b0902d885ee19984401363492466f)

- **ATOM rollout 暂停后可跨权重更新恢复。** keep 模式停止新调度，权重提交或清空 KV 前等待在途传输并对齐 PP 头部，然后抢占序列。恢复时用新权重重算上下文，保留已发 token 与 logprob。无法清除外部缓存或 P/D 在途请求无法安全重算时拒绝。适合与 verl partial rollout 对照停顿、拒绝语义和重算成本。 [ROCm/ATOM #3059 · 28e416cb6 · 11-11 10:29:43](https://github.com/ROCm/ATOM/commit/28e416cb6977df4be89cfd080653b3dedd72a30f)

- **ATOM 合入 prefix-spread DP 路由和 Mooncake Store 恢复。** 按前缀定位可复用的 DP rank，结合负载分流并接入原生 Store 恢复。未完成的 HIP IPC event 留待下一次 DP rendezvous，避免销毁时阻塞对端需要的 collective。可测命中率、P99 排队与 Store 失败回退。 [ROCm/ATOM #2755 · 399858c32 · 11-11 14:52:47](https://github.com/ROCm/ATOM/commit/399858c328fdf60ae0049ea210c2a5ce4f7dfdf3)

- **ATOM 的 vLLM 插件处理 Kimi 与 GLM 混合状态。** Kimi-K3 将对齐边界的 KDA 状态附在 LMCache MP PAGE 对象上，依赖具体 KV 物理布局以一并提交和恢复。GLM-5.3-Flash 在 vLLM 0.31 插件接入 MLA、KDA 和 DSA indexer 三类状态及前缀缓存/MTP。可按两条路径的启动约束做跨进程前缀命中与重算的 logit 对照。 [ROCm/ATOM #3117 · eec6c47c2 · 11-11 11:17:04](https://github.com/ROCm/ATOM/commit/eec6c47c21d25a336fff93a571424c8aeb202bb0) · [ROCm/ATOM #3137 · e91df9573 · 11-11 11:16:42](https://github.com/ROCm/ATOM/commit/e91df9573ffe3ef6dabc0d2016ef3f9f59a8f6cc)

- **hipDNN 扩充 FlyDSL SDPA 前向和图捕获能力标记。** FlyDSL 前向在 gfx11、gfx12、gfx90a、gfx942、gfx950 增加变长、ragged 和 vLLM 布局，项目用 AOTriton 前向网格对照。部分 capture-safe engine 暴露新能力标记，MIOpen 未声明。可验证真实设备图内外数值、分页与 dropout。 [ROCm/rocm-libraries #15066 · 4c43643b1 · 11-11 16:21:28](https://github.com/ROCm/rocm-libraries/commit/4c43643b117619b48ae31abff1d8e6a809c8a34c) · [ROCm/rocm-libraries #15319 · cf00c41e9 · 11-11 09:01:42](https://github.com/ROCm/rocm-libraries/commit/cf00c41e9126c4e3fb5320514ec20aca05bb94ad)

- **ROCm 扩展 FP8 GEMM 布局、调度和选核。** hipBLASLt 为 gfx1250 F8SS/F8BS/F8HS NN 增加 Origami 预测库及行主序 A/B 处理，CK Tile 为 gfx1250 comp_tdm 和 comp_tdm_v2 接入 Stream-K。可按实际 LLM GEMM 形状、转置和精度做选核与吞吐对照。 [ROCm/rocm-libraries #15559 · 3fd360629 · 11-12 07:19:46](https://github.com/ROCm/rocm-libraries/commit/3fd3606294b368adbf5e934bda66a9e24d206b88) · [ROCm/rocm-libraries #14769 · 2350b4114 · 11-11 23:45:08](https://github.com/ROCm/rocm-libraries/commit/2350b41148b4f57382868d5e2bc0146f97204a60) · [ROCm/rocm-libraries #15083 · d19ff95d4 · 11-11 15:39:36](https://github.com/ROCm/rocm-libraries/commit/d19ff95d472ec8d77c60bdae8ec9ed9b3ac7aca9)

- **AITER 增加分页 prefill、MXFP8 输出量化与量化归约。** FlyDSL FMHA 增加 gfx1250 paged-KV MHA 前向 prefill，batched MXFP8 GEMM 支持可选融合量化 epilogue。QuickAllReduce 增加 INT5/INT6 并保留 vLLM 兼容别名，有 codec 和多卡测试。可分别测页表形状、写回带宽及量化通信误差。 [ROCm/aiter #7553 · 3b15efe5b · 11-11 15:00:12](https://github.com/ROCm/aiter/commit/3b15efe5b04c0b3206490fd14bae9635b7c17ace) · [ROCm/aiter #7178 · ef7ab2b84 · 11-11 23:40:26](https://github.com/ROCm/aiter/commit/ef7ab2b841063536d27da3a53026dfb35c244c92) · [ROCm/aiter #5555 · cb4ac3423 · 11-11 11:05:27](https://github.com/ROCm/aiter/commit/cb4ac3423f56ef03ddf9d1c329d3ee32ca039270)


- **ROCm attention 修复 gfx1250 decode 写回与 A2A scale 对齐。** Gluon unified attention 对 GPT-OSS 特定 decode 形状由寄存器直接全局写回，绕开 LDS 读写可见性问题。项目对 gpt-oss-120b 在 8k/1k 和 1k/1k、并发 16/64 的 serving 运行报告 TP1/TP2 的坏行在两轮降为 0，extend 仍走旧路径。另为融合 A2A 的 MX Q scale 补足 MHA 消费端整 tile backing，逻辑形状不变。优先用线上 attention 对照和非整 tile 测试复核。 [ROCm/aiter #7203 · 61b4e7bab · 11-12 07:28:10](https://github.com/ROCm/aiter/commit/61b4e7babe0a780c4bac578a0d5a4022fd301a03) · [ROCm/aiter #7804 · f6ac01df0 · 11-11 22:09:01](https://github.com/ROCm/aiter/commit/f6ac01df04d10cd1a5171bb3c7ee3c027fa10c36)

- **AITER 开放 DeepSeek V4.1 算子路由，并融合 Qwen3-Next attention 前置计算。** DeepSeek V4.1 的多条稀疏 attention 路径接入 provider 合约，项目明确首次调用与图捕获的约束。gfx950 的 Qwen3-Next full-attention 路径将 Q/gate 拆分、Q/K RMSNorm 和部分 RoPE 合为一次 launch，KV 写入仍由 attention backend 负责。大批量 Gluon 路径要求 Triton 3.8 起。适合复查模型形状和图内 provider fallback。 [ROCm/aiter #7728 · 66ec1adad · 11-11 19:17:50](https://github.com/ROCm/aiter/commit/66ec1adad26423408d0fd776e061efcf6fbc9a39) · [ROCm/aiter #6835 · 6efb333ad · 11-11 16:21:24](https://github.com/ROCm/aiter/commit/6efb333ad663d3fec3cbf9ce0084c9c1887b6010)

- **hipBLASLt 收紧低精度 GEMM 的布局和 scale 匹配。** gfx950 的 FP4→BF16 subtile 路径对 post-loop store 按实际转置布局检查。Tensile 问题定义拆开 packed 与 pre-swizzled UE8M0 scale 形态，避免 packed 参数被交给要求预重排的 kernel。可用每种转置、scale 布局与非默认选核配置对照数值。 [ROCm/rocm-libraries #15503 · cd9cf22d7 · 11-12 01:33:07](https://github.com/ROCm/rocm-libraries/commit/cd9cf22d73bd1a145fc4c394eaf47c69c32c462a) · [ROCm/rocm-libraries #14625 · 0eb242695 · 11-11 17:49:00](https://github.com/ROCm/rocm-libraries/commit/0eb2426958d13441e7eaac3bd5609846106704d1)


### Google TPU 与 AWS Neuron

- **Neuron SDK 2.33.1 修复 DMA 与 collective 并发挂起。** Runtime 调整 HWDGE 中非 GPSIMD 的 DMA 触发顺序，处理与其他 transpose DMA（包括 collective）的竞争，并让 GPSIMD 对无效索引分组给出错误。runtime-lib 更新到 2.35.33.0，框架包随之重建。可按发布清单在同一平台复测通信和转置重叠的长稳负载。 [aws-neuron/aws-neuron-sdk v2.33.1 · 正式版 · 11-12 06:16:21](https://github.com/aws-neuron/aws-neuron-sdk/releases/tag/v2.33.1) · [官方 Runtime 变更](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/2.33.1.html#neuron-runtime)

- **SGLang TPU 修复 TP 复制 KV 头的在线 FP8 失配。** 原 fused QKV shard 的 per-tensor scale 随同卡 Q 头变化，TP 大于 KV 头数时，同一复制头的量化字节可能不同。改成每个输出 row 一个 scale，TPU dense FP8 已支持该形状。可测试 EAGLE3 draft 继承目标配置和 Raiden push-reshard 一致性。 [sgl-project/sglang-torchtpu #1767 · c5ef70856 · 11-12 06:03:14](https://github.com/sgl-project/sglang-torchtpu/commit/c5ef70856c81f21e151c1e1382d9e60f36b8700b)

- **SGLang TPU 改进混合注意力缓存与 DSA indexer。** GPT-OSS 一类模型的 P/D KV offload 同时传 full-attention 页和分支末端 SWA 窗口，双方只在窗口完整时接受恢复。要求开关启用且 page-size 大于 1。另移植 SparseCore cooperative top-k、线性复杂度的 schedule 构造与 blocked prefill。可分别查分支缓存命中、Top-k 误差和 DCP 合并开销。 [sgl-project/sglang-torchtpu #2086 · 5c136df1b · 11-11 21:19:05](https://github.com/sgl-project/sglang-torchtpu/commit/5c136df1b4e2973f9f65079098c7110b949264b7) · [sgl-project/sglang-torchtpu #2093 · 50e738fdd · 11-12 01:50:46](https://github.com/sgl-project/sglang-torchtpu/commit/50e738fdd60bfe66cef0df04de625dc379c5d425)

- **TPU MoE decode 增加 TP 内核、预取和跨 rank 统一回退。** SGLang 的 fused MoE decode 接入 TP experts 与归约，测试包含 GLM-5.3 TP8 桶。vLLM TorchTPU 在当前专家 tile 计算期间读下一专家 token 行，megakernel 由 TP 组达成一致后再继续或回退，并检查 chip、并行和功能约束。可比较小批量 TPOT 与不均匀路由下的 fallback。 [sgl-project/sglang-torchtpu #2009 · 397c9871e · 11-11 11:30:28](https://github.com/sgl-project/sglang-torchtpu/commit/397c9871e131a4afd7fb7bf98e4cf1bfbf20ce99) · [vllm-project/vllm-torchtpu e9cf14895 · 11-12 05:29:07](https://github.com/vllm-project/vllm-torchtpu/commit/e9cf1489562d91f6718b1d0bda712816602c9b13) · [vllm-project/vllm-torchtpu 2ca080d79 · 11-11 12:09:55](https://github.com/vllm-project/vllm-torchtpu/commit/2ca080d798328769703a157df0c6dd8d2539eec2)

- **TPU 推理接入 GDN 投机状态，训练注意力修正 dropout 索引。** tpu-inference 给同步调度传递 GDN 投机元数据与 Mamba read offset。TorchTPU 将 flash attention dropout hash 的 KV 长度改为未补齐长度。可分别用接受长度变化核验状态演进，用同一随机种子核验不同 padding 下的 parity。 [vllm-project/tpu-inference #3971 · fd9391900 · 11-12 02:13:18](https://github.com/vllm-project/tpu-inference/commit/fd939190003a32177bd85ad0978f4091f5526700) · [google-pytorch/torch_tpu 43b49e765 · 11-11 09:28:06](https://github.com/google-pytorch/torch_tpu/commit/43b49e765fcdc7ee9f036254b17cab10984f0d0e)

- **TorchTPU SparseCore 扩展到 64 位 embedding ID。** 预处理与统计接受 int64 索引及偏移，保留 int32 快速路径，分区取模和偏移避免提前窄化，更新 JAX embedding 依赖。补丁包含跨越 2³¹ 和 2³² 的对照测试。适合验证大词表或特征字典导入及旧格式兼容。 [google-pytorch/torch_tpu 81e7c4456 · 11-11 13:34:24](https://github.com/google-pytorch/torch_tpu/commit/81e7c4456b57c6f7019ba61d8801a9d5ef17eb28)

- **TPU Sync 加入版本化权重控制面与 Mooncake buffer provider。** 权重传输新增版本化控制面 proto，属于分阶段接口增量。Mooncake TENT provider 为 JAX TPU 数组提供适配。可跟踪控制面与设备传输落地，复核权重切换可见性及资源释放。 [google/tpu-sync d2d9d37b6 · 11-11 16:44:07](https://github.com/google/tpu-sync/commit/d2d9d37b6b05a33fba2a3e498e5d9fdc46d4dfa6) · [google/tpu-sync ab69861b1 · 11-11 09:48:12](https://github.com/google/tpu-sync/commit/ab69861b1ec5742799bf246c27fb0d711830ebce)


### 昇腾、沐曦与摩尔线程

- **vLLM-Ascend 实验性 GDN 单次 prefill 导出检查点。** 同一次前向导出卷积历史和 FP32 recurrent checkpoint，减少对齐边界的重复 prefill。默认关闭，要求 Runner V2 与 align prefix cache，不支持时回退。项目用 Qwen3.5-9B BF16、TP1、随机 6000 入／512 出测试，并发 1/4/16 的单轮 TTFT 分别降 27.6%/10.28%/5.53%。PR 未披露芯片型号，证据限单组负载。优先核对 prefix 命中后的 logit。 [vllm-project/vllm-ascend #21040 · 8f191f1a4 · 11-12 08:38:19](https://github.com/vllm-project/vllm-ascend/commit/8f191f1a4925011caf1febd29aaa7bfbb0281c9c)

- **GLM-5.3-Flash 的 PCP 前缀恢复与通信重叠。** AscendStore 对各 PCP KDA shard 独立存键，每个合法边界具备全部 shard 才命中，带 sharded Mamba 时暂拒绝 TP>1 与 PCP>1 同开。prefill 提前发 MLA main-KV 和 KPool AllGather，纯 prefill 的 KDA 改 ReduceScatter。项目用 200 并发、131,072 入／1 出、其中 117,504 入已缓存，报告平均 TTFT 48.37→46.38 秒，测于最终重构前。可复核未命中和混合 decode 的收益。 [vllm-project/vllm-ascend #20835 · 441c6a556 · 11-11 17:19:29](https://github.com/vllm-project/vllm-ascend/commit/441c6a55661befe8141a3895df724013537c6791) · [vllm-project/vllm-ascend #20454 · 77e1f2d79 · 11-11 14:12:54](https://github.com/vllm-project/vllm-ascend/commit/77e1f2d79835c06567fb778959f351a7e3d4be84)

- **Ascend 逐层推送 KV，并修复多 chunk 权重 IPC 引用寿命。** 实验 MooncakePushConnector 由 D 分配目标 KV，P 写各层后发完成信号。当前限 DCP=1，P 抢占尚未支持，真机顺序待核验。NPU IPC 导入器跨 chunk 复用同一 export 映射，在 finish/shutdown 释放，修复 staging buffer 每轮无法回收。可分别验证取消和多轮权重更新后的显存。 [vllm-project/vllm-ascend #20940 · 1af7c5fc8 · 11-11 16:11:34](https://github.com/vllm-project/vllm-ascend/commit/1af7c5fc8aa2c0007ea53a61d0805f53c7983019) · [vllm-project/vllm-ascend #20740 · 9b3e01281 · 11-11 18:53:31](https://github.com/vllm-project/vllm-ascend/commit/9b3e01281c28642c68af88997ab93fd0125ff67f)

- **TorchNPU 扩展异步通信语义、functional AllReduce 与 E8M0 dtype。** WorkHCCL 的 blockCurrentStream 把等待挂到当前流，functional AllReduce 增加 CCU_MS 入口及 A5 条件测试。E8M0 dtype 映射配套 12 项 CTest，仍需验证目标 HCCL 版本的 collective 支持。可核查异步依赖与 scale 通信。 [Ascend/pytorch 7e73e9a3f · 11-11 16:47:32](https://github.com/Ascend/pytorch/commit/7e73e9a3f5aac362b0ad74f4a2288aa1a864d299) · [Ascend/pytorch bcb6d14a0 · 11-11 15:46:16](https://github.com/Ascend/pytorch/commit/bcb6d14a0a9f51c40aa68e4bc3e1313fb6e67465) · [Ascend/pytorch 9393794d1 · 11-11 14:45:44](https://github.com/Ascend/pytorch/commit/9393794d10510a11250d755b602f79058da2c180)

- **TorchNPU Inductor 修复 dropout CSE，调整常量索引传递。** fallback_random 下的 AOTAutograd 路径将 npu._npu_dropout 纳入随机算子集合，防止合并独立 dropout 和扰乱后续 RNG。indexed slice-cat 的索引列表改在 pre-grad 注册为 buffer。训练升级可核查多分支随机性和图缓存索引生命周期。 [Ascend/pytorch c491b01e5 · 11-11 15:20:39](https://github.com/Ascend/pytorch/commit/c491b01e5f8e500d05ef6bff055ceafb0186862b) · [Ascend/pytorch 4afb6ccf5 · 11-11 23:02:56](https://github.com/Ascend/pytorch/commit/4afb6ccf55263db096f1d4aead3a1707d5b4b8c3)

- **HCCL 调整 AllToAllV 拓扑选路并暂停任务缓存。** Pairwise 在满足对称、8 卡模块和 16 至 128 rank 等条件时获优先级，保留 Mesh 处理其他组网。另暂停 V/VC 的 AICPU task cache 与 warmup，DeepSeek 挂起原因尚待确认。UBX 加入机内 OmniPipe AllGatherV，合入记录报告 RDV 和拓扑测试。可核对实际拓扑的算法选择和 hang 率。 [cann/hccl !3754 · 77730b79e · 11-11 20:19:26](https://gitcode.com/cann/hccl/commits/detail/77730b79ee499a97cd9b8583b672f9266d89aa8e) · [cann/hccl !3963 · ac99a53d1 · 11-11 11:30:41](https://gitcode.com/cann/hccl/commits/detail/ac99a53d15c15b3b84526184bf3db4537399725a) · [cann/hccl !3951 · 502e8c477 · 11-11 16:07:49](https://gitcode.com/cann/hccl/commits/detail/502e8c477fde929387826ddb9cbc38a675eea331)

- **CANN 增加量化稀疏 indexer 并修正 MLA 非标准头组。** ops-transformer 新增 QuantSparseLightningIndexer 与 metadata 路径，DeepSeek V4.1 接入相应变体，LightningIndexerV2 扩充 fast 模板维度和流式 TopK。FIA MLA 按 gSize 分解转置写回偏移，覆盖 51/72 头等组合。修复合入记录只给静态检查。可测不同可见 KV 长度与头组下的数值。 [cann/ops-transformer !15724 · 83de03539 · 11-11 17:30:32](https://gitcode.com/cann/ops-transformer/commits/detail/83de0353964d932b91186cc00bc4ffe166cc1d02) · [cann/ops-transformer !15751 · a6edf1292 · 11-11 17:36:48](https://gitcode.com/cann/ops-transformer/commits/detail/a6edf129237657c54c539be7602c81fc0f7b3a0d) · [cann/ops-transformer !15479 · 906975ec0 · 11-11 15:04:52](https://gitcode.com/cann/ops-transformer/commits/detail/906975ec0a9965cfe1d69ae4d12c8554758f8084) · [cann/ops-transformer !15750 · 1d1ae6fc5 · 11-11 19:20:35](https://gitcode.com/cann/ops-transformer/commits/detail/1d1ae6fc55c125dc10eada34765b39de8465b815)

- **CANN 修复 fused CE、权重量化 matmul 和 LayerNorm 融合边界。** fused linear CE 梯度修正空输出与索引处理，arch35 的 WeightQuantBatchMatmulV2 在无尾块时保留原 baseN，LayerNorm fusion 从匹配子图取 epsilon。权重量化提交给出 Ascend950、CANN 9.3.0 的 ND/NZ 上板对照及 7 条 host UT。可将空输出、非默认 epsilon 和整除形状纳入升级回归。 [cann/ops-nn !13730 · b6bf19315 · 11-11 14:57:18](https://gitcode.com/cann/ops-nn/commits/detail/b6bf19315445bdaada4f68b38fa38dddd4fcff8e) · [cann/ops-nn !13520 · 6671fd164 · 11-11 11:51:46](https://gitcode.com/cann/ops-nn/commits/detail/6671fd164c319ca247e6014dc4e505192d5c3876) · [cann/ops-nn !14106 · 16166c18f · 11-11 17:30:20](https://gitcode.com/cann/ops-nn/commits/detail/16166c18f896b646accf0e04d2378c8e74ad2aae)

- **昇腾社区公开两例 Atlas 900 A3 排查。** 四机 CANN 8.1.RC1 案例从通信域不对称追到同卡双 Die 的 RoCE 路由，文章把 CANN 8.1.RC2 的 NHR-HCF 列为升级路径。另一案例设备计算时间接近，容器 Host FREE 与 MEMCPY 延迟受 CPU 争用影响，调整权重后接近裸机。均为单案例，可按拓扑、调度日志和 Host timeline 核对。 [昇腾社区案例 20261111-1 · 2026-11-11（仅披露日期）](https://www.hiascend.com/ascendgateway/ascendservice/tech/details?techColumnCode=20261111-1) · [昇腾社区案例 20261111-2 · 2026-11-11（仅披露日期）](https://www.hiascend.com/ascendgateway/ascendservice/tech/details?techColumnCode=20261111-2)

- **vLLM-MetaX 主干迁移 vLLM 0.33 的注意力、MoE 与 DeepSeek V4 路径。** 同批提交更新 MACA attention、modular MoE kernel 和 DeepSeek V4/V4.1 接入，适配上游接口。证据为已合入源码，未找到等价稳定版的部署指标。迁移可用同一 checkpoint 核验量化加载、数值与 fallback。 [MetaX-MACA/vLLM-metax 31f953707 · 11-11 18:07:13](https://github.com/MetaX-MACA/vLLM-metax/commit/31f95370752da4e9d4db9b3a3873fcc119a87393) · [MetaX-MACA/vLLM-metax 1d564ef1f · 11-11 18:07:13](https://github.com/MetaX-MACA/vLLM-metax/commit/1d564ef1fb853a13cb7cdda91ca191dec5bd6744) · [MetaX-MACA/vLLM-metax b1e1345b0 · 11-11 18:07:13](https://github.com/MetaX-MACA/vLLM-metax/commit/b1e1345b01e10fe6c418cac70c63af0f96e5a3ca)

- **vLLM-MUSA 合入 DeepSeek V4.1 W4AFP8。** 新增 packed W4AFP8 expert 路径和 shared expert overlap，并加量化 MoE 与路由映射回归。当前为默认分支合入，实际硬件覆盖和吞吐待固定版本测量。可用相同专家分布对照 BF16 与 W4AFP8 的误差、吞吐及显存。 [MooreThreads/vllm-musa #338 · 45341a3a3 · 11-11 11:24:21](https://github.com/MooreThreads/vllm-musa/commit/45341a3a3499e51a39dcd47da3d54adb2b43f958)

- **CANN 调整 DeepSeek V4 单卡部署脚本的版本矩阵。** 脚本将 CANN 9.0.0–9.2.0、Python 3.11/3.12 和 kt-kernel 构建产物检查归到同一交付路径。GPQA 解析按报告主指标提取，功能验收改用 Chat Completions。项目完成部分容器和离线检查，Python 3.12 路径经过模拟验证，完整模型编译和 NPU 推理仍待验证。 [cann/cann-recipes-infer !1226 · 7e42a0954 · 11-11 15:47:54](https://gitcode.com/cann/cann-recipes-infer/commits/detail/7e42a0954c952201f7bfbc03d620a89b6c0efa1a)

<a id="coverage"></a>
## 覆盖与续查

### 模型、训练数据与评测覆盖

- 主窗为 (2026-11-11 01:00:49Z, 2026-11-12 01:05:00Z]，重叠从 2026-11-08 01:00:49Z 起。按官方明确时刻或固定工件的有效变更时间收录，只有日期的公告保留当地日历日期。已按前期日报排除文心 Base、Arete 等已报首发。AWS 对 Kimi K3.1 Pro Highspeed 的首发记录为 11 月 6 日，11 月 10 日技术博客属于后续使用说明。
- 直接检查 121 个新闻、API、模型与榜单等入口，86 个返回 HTTP 200，21 个超时或无响应，6 个 403、6 个 404、1 个 429、1 个 302。IFM 的 PAN 2 两份官方页面经浏览器补读。用中英搜索、ModelScope、EvalMap、Jiufeng 等寻找候选，一手发布、提交和工件用作事实证据。动态历史和被拒绝的入口仍有缺口。
- HF 全站按创建时间翻到重叠起点前。模型 14 页共 14,000 条，重叠窗内 13,532 条，截止后再取一页补齐末段并增加 11 条。数据集首轮返回截断 JSON，改每页 500 后读取 15 页、7,500 条，重叠窗内 7,360 条。154 个定向组织的 308 次模型／数据集按修改时间查询均返回 200，得到 340 条候选。组织结果为首批页面，全站 lastModified 和候选全量语义核对待查。
- 定向首轮接口共 62 个请求，12 个成功、49 个无 HTTP 返回、1 个 401。对一手公告、固定工件和提交作了补充读取，XYBench 与 AgentPerf v2 仍返回 401。保留历史缺口及本轮待查，包含文心 Preview 权重、Qwen4 后续开放、Decision Index 版本口径与未核工件。发布状态、版本元数据、访问失败与工件已核验分别记录在同目录 ledger.json。

### 论文与会议

九个 arXiv 分类的 11 月 11 日公告合并为 895 个唯一 ID，其中 573 个至少有一处新稿或跨列记录，322 个只出现在修订区，全部与 OAI 版本历史关联。该批按排程落在重叠窗。OAI 抓取的公开记录中，主窗内的提交/修订时间戳为 0，最晚可见提交为 11 月 10 日 18:59:54 UTC。Atom 提交/更新时间检索连接失败，延迟公开与逐篇修订语义继续保留待查。 [arXiv 公告规则](https://info.arxiv.org/help/availability.html#announcement-schedule) · [cs OAI 元数据](https://oaipmh.arxiv.org/oai?verb=ListRecords&metadataPrefix=arXivRaw&set=cs&from=2026-11-11&until=2026-11-12)

AACL-IJCNLP 官方议程截至 11 月 10 日，当天工作坊已见前期日报。NeurIPS 正式活动在 12 月，SC26 排程自 11 月 15 日开始，EMNLP 2026 已在 10 月结束。TMLR 目录只标月份，OpenReview API 与若干论坛详情遇到验证页，本期尚未核定正式发表日或审稿状态变更。 [AACL Anthology](https://aclanthology.org/events/aacl-2026/) · [EMNLP](https://2026.emnlp.org/) · [NeurIPS](https://neurips.cc/Conferences/2026) · [SC26](https://sc26.supercomputing.org/program/) · [TMLR 目录](https://jmlr.org/tmlr/papers/) · [OpenReview](https://openreview.net/group?id=TMLR)

### 开源工程覆盖

工程检查窗口（UTC）为 2026-11-11 01:00:49 至 2026-11-12 01:05:00，回溯至 11-08 01:00:49。记录 56 组、157 个版本事件，127 项有 PR 补丁、26 项有固定提交补丁，4 项为发布版本。历史日报和状态按主链接、提交及 PR 别名查重均为零。未独立复训或压测。

- GitHub 检查 75 仓。默认分支、Release Atom 和主窗 merged PR 分页各完成 70 仓。15 个核心及重点仓补查回溯期的全部 merged PR 分页。对 Atom 尾部落在回溯窗内的 llama.cpp、TorchTitan、PyTorch、Run:ai 补查 GitHub 发布页，四仓均查到更早日期。slime 与 OpenRLHF 的默认分支、合并列表及发布工件均未见窗口内增量。
- PyPI 查询 34 个包名，29 个可读。attention-gym、miles-diffusion、neptune-model、radixark-miles、xla-serving 的包名入口返回 404。GitHub 中另行核对 Miles 和 Miles Diffusion 的代码。
- 5 个 GitHub 仓入口返回 404，涉及 SqueezeAILab/SeerAttention、attention-gym/attn-gym、sgl-project/sgl-flash-attention、sgl-project/sgl-kernel、zml/llmd。工程台账保留原路径，另有已核对的 SeerAttention、attention-gym、SGLang 主仓入口。部分 release REST 取证返回 403 或连接失败，已收录的 4 个发布版本均从发布 HTML 核对时间和提交。PR 补丁的 429 已用固定合并提交补丁恢复。
- 扩展仓的回溯主要依赖默认分支和发布源，15 个定向仓以外的历史特性分支合并未逐仓穷尽。候选按机制和训练、推理影响抽筛，逐源进度、失败和待查范围见本地 `ledger.json`。

### 硬件与软件栈覆盖

- **硬件与软件协同。** 核查 NVIDIA CUDA/cuDNN/NCCL、AMD ROCm/AITER/ATOM/FlyDSL、Intel Gaudi、昇腾 CANN/TorchNPU、TPU、AWS Neuron 及国产适配栈的官网、版本、默认分支和定向补丁。台账共 508 次请求，361 次成功读取响应。GitHub REST 末端复查遇到 403，随后用 Atom 补查主要分支。OneCCL、MLCommons 及部分社区入口仍有缺口。Ascend 首屏与两条案例正文已读取，CANN 组件映射、libtpu 工件差分及 A5X 规格冲突继续留查。

- **续扫边界。** 本轮发现截止为 2026-11-12 09:05:00。下轮从该位置续扫并回溯 72 小时，同时继续处理分源待查记录。全局完整语义水位仍未闭合。本轮未做独立复训、设备压测或付费 API 实测。
- **补录与版本边界。** 上轮未确认的 arXiv 11 月 11 日公告已核到分类清单和版本史，首次核验的论文按重叠窗补录。FlyDSL 0.5.0 本轮记 GitHub 正式 Release，11 月 9 日日报记 PyPI wheel。
