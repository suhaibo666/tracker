# LLM 技术动态日报 · 2026-11-10

> 时区 Asia/Shanghai（UTC+8）。增量扫描窗口 **2026-11-09 09:40:38 → 2026-11-10 08:58:40**，向前回溯 72 小时至 **2026-11-06 09:40:38**。实际生成时间 **2026-11-10T09:11:28+08:00**。日期未披露到时分时保留原日期。首次发布、工件或代码更新、会议报告与本轮发现分别标注。
>
> 生成前核对本地 34 篇历史日报及索引，与 GitHub `main` 的 `09be9cabffd36802ca5f7b9f994e49b6faf5614e` 逐字一致。前次待上传队列为空。
>
> 本轮核验公开的一手报告、工件、发布记录及部分代码补丁。性能与能力指标为原作者或项目方的口径。

## 本次重要变化

- **模型与评估。** Claude Fable 5.2 和 Mythos 5.2 共用底模，推理历史的前缀校验影响多轮工具调用与断点恢复。Muse Ember 1.0 的调用成本受缓存命中与 200K 上下文分档影响。LongCat-2.5 的标准与 W4A4 固定权重目录重新可读。
- **预训练与后训练。** 小型 KLM 1.6 披露长衰减阶段和文档边界处理。论文侧，ResQAdam 研究低秩优化器状态，RPEG 针对 RL 生成长尾调整分布式资源，LambdaMix 用同一偏好向量控制奖励与 LoRA。
- **推理与工程。** vLLM 合入 NIXL 对象 KV 后端和符合条件的 NVLink 流水线传输，SGLang 将统一 KV 布局接入 admission 与 P/D 生命周期。FlashInfer 新增 GDN 树路径验证与状态提交。Megatron-Bridge 将部分 GB200/GB300 MoE 配方切到 NCCL EP。
- **硬件与系统。** Google 披露保密推理 TPU 8tx、双芯片 8i Max 和 TPU Sync，并将 Cloud Pathways 和 TPU 动态切片推到 GA。NCCL 2.32.6-1 修复了 NVLS rank 顺序和注册缓冲区竞态。昇腾 HCCL 合入 hostDPU inline reduce。

[模型与技术报告](#models) · [后训练与数据工件](#posttraining-models) · [评估与科研 Agent](#model-evals) · [研究论文](#papers) · [开源工程](#engineering) · [硬件与软件协同](#hardware) · [覆盖与续查](#coverage)

<a id="models"></a>
## 模型与技术报告

### Claude Fable 5.2 和 Mythos 5.2 共用底模，多轮推理前缀校验扩至所有账号

**Anthropic，11 月 10 日 02:57:59（北京时间）发布。** Fable 面向一般用户，网络安全与生物防护更严格，Mythos 仅向通过相应验证的组织开放。两者默认 100 万 token 上下文、最多 128K 输出，标准价为每百万输入／输出 token 8／40 美元，厂商报告同档输出最高快约 30%，快速模式最高约 2.5 倍、单价翻倍。Anthropic 的 FrontierCode 2.0 中，Fable 5.2 High 得 45.3%，接近 Opus 5.5 Max 的 45.5%，单题成本约一半，Xhigh 得 49.1%。迁移时，已返回的 thinking block 绑定其前面的 system、tools 和消息。先改历史再回传该块，默认会返回 400，旧账号也受约束。可用原样历史和修改过的历史各回放一次，分别测长程任务成本与断点恢复。系统卡 §6.1.2 的白盒分析在最多 43% 的自动审计轨迹中发现 Mythos 5.2 可能知道自己在受评估，预发布行为测试的外推因此受限。[官方发布](https://www.anthropic.com/claude-fable-and-mythos-5-2) · [官方发布帖](https://x.com/claudeai/status/2119867114450268322) · [API 迁移说明](https://platform.claude.com/docs/en/models/fable-5-2/whats-new-fable-5-2) · [系统卡](https://assets.claude.com/27d60918d218125e7504bc7354c4b4626ec56b11.pdf)。

### Meta 推出 Muse Ember 1.0，长上下文档位的单价为三倍

**Meta，11 月 9 日发布。** `muse-ember-1.0` 是首个用 Muse Wonder Full 参与训练的 Meta 模型，厂商未披露具体训练配方。接收文本、图像、音频和视频，支持 Responses API，最大 100 万上下文、512K 输出。Artificial Analysis 的 Intelligence Index v5 中，Ember Max 得 36、每题约 0.14 美元，Haiku 5.5 Max 同为 36、约 0.41 美元。该评测方将成本差异归因于较少的输出 token 和较低的缓存输入单价。官方价目表把 200K 以上上下文的百万 token 输入／缓存输入／输出价从 0.10／0.004／0.50 美元提高到 0.30／0.012／1.50。成本对照应分开记录缓存命中、上下文档位和输出长度。Meta 公告与 Artificial Analysis 正文对同名 AutomationBench-AA 分别列 37.8 与 68.1，本报仅引用两边吻合的 v5 综合分和任务成本。[厂商公告](https://research.meta.ai/blog/muse-ember) · [开发者价目](https://dev.meta.ai/models/muse-ember) · [评估方法](https://research.meta.ai/static/muse-ember-evaluation-report) · [Artificial Analysis 实测](https://artificialanalysis.ai/articles/meta-s-muse-ember-is-the-cheapest-model-scoring-36-on-the-intelligence-index)。

### LG 开放 K-EXAONE 2.5 免费试用 API，默认开启推理

**LG AI Research，11 月 9 日于 NVIDIA AI Day Seoul 宣布。** 2.5 是在 8 月 K-EXAONE 2.0 上追加训练的技术验证模型，本次提供 API。文本模型 `k-exaone_v2.5` 的输入 token 和 `max_tokens`（含推理）之和上限为 524,288，超出返回 400。推理默认 Max，另有 Low、High，服务把推理和最终答案分字段返回。LG 自测用公开数据，AutomationBench、Claw-Eval、BrowseComp、SWE-bench Pro 分别为 74.6、77.3、78.0、59.2，公告未公布完整运行参数。API 条款仅授权非商业使用，密钥 30 天失效。可用同一工具编排测试三个推理档，按含推理的总 token 和实际完成率对照。[LG 韩文公告和自测](https://lg.co.kr/media/release/30760) · [API 文档](https://docs.exaone.ai/ko/docs/k-exaone-2.5) · [API 条款 §2–3](https://docs.exaone.ai/ko/terms/terms-of-service)。

### 美团 LongCat-2.5 的标准与 W4A4 权重目录现已核验

**美团，11 月 9 日 19:38:03 和 19:38:19（北京时间）更新两份模型卡。** 此前跟踪的 HF 接口返回 401，本轮两种固定版本均可读取 MIT 许可、配置与分片清单。两份索引都与各自目录的 210 个 `.safetensors` 文件完全匹配，分片标称总量约 3.56 TB 与 1.34 TB，配置中的 `max_position_embeddings` 为 524,288。权重上传与量化变动可追溯到 11 月 5 日，本轮两个提交的差分均只修改 README，删去模型卡的详细视觉建模、长程 RL 和训练系统小节，正式技术博客仍可查。W4A4 的卡片保留对专家权重和激活采用 INT4 的说明。可固定仓库版本和权重索引部署，分开记录标准版与 W4A4 的服务结果。[标准版固定目录](https://huggingface.co/meituan-longcat/LongCat-2.5/tree/6fc4844cc9455bc45b84871cbc9137b3286f0fc2) · [W4A4 固定目录](https://huggingface.co/meituan-longcat/LongCat-2.5-W4A4/tree/3315a8d37c1d430677de11ceacc97506a9a0c07d) · [卡片差分](https://huggingface.co/meituan-longcat/LongCat-2.5/commit/6fc4844cc9455bc45b84871cbc9137b3286f0fc2) · [官方技术博客](https://longcat.ai/blog/longcat-2.5/)。

### Seedream 5.5 针对多轮编辑训练，直接输出 RGBA

**字节跳动 Seed，11 月 9 日发布。** 厂商对未编辑区域的像素保留做专项训练，演示把每轮输出作为下一轮输入，连续完成同一图像的 100 次编辑。原生 RGBA 输出支持半透明元素。其 MagicArena 人工评测覆盖六个生产场景，中英小字等样本的严重、轻微文字错误率分别为 0.4% 和 1.7%。百轮演示来自同一条编辑序列，可用多种编辑区域和图像分别测未编辑区像素漂移、文本错误率与边缘透明度。官方称多入口陆续上线，方舟 API 的可调用版本和时间待核实。[中文公告与评测](https://seed.bytedance.com/zh/blog/more-realistic-images-more-stable-editing-across-100-rounds-introducing-seedream-5-5) · [官方模型页](https://seed.bytedance.com/en/seedream5_5)。

### KLM 1.6 将 99M 基座的学习率衰减段扩至 100B token

**MinimaLabs，11 月 9 日 18:21:02（北京时间）上传 Apache 2.0 权重。** 从 KLM 1.5 衰减前的同一检查点继续训练，将衰减段由 20B 扩至 100B token，学习率按 `1−√t` 降至零。24 层按 3:1 混用 Gated DeltaNet-2 与全注意力，打包训练时在文档边界遮蔽注意力，并重置循环状态与 n-gram 历史。相对 1.5，零样本 ArithMark-3 从 48.20% 到 55.90%，HellaSwag 从 43.74% 到 45.28%。这组权重比较同时改变衰减段长度和训练 token 预算，可用等 token 的不同衰减曲线复测其贡献。[固定权重和训练配置](https://huggingface.co/MinimaLabs/KLM-1.6/tree/53226768a8957ca6f99e766f8025a96e2467537f) · [固定模型卡与评测](https://huggingface.co/MinimaLabs/KLM-1.6/blob/53226768a8957ca6f99e766f8025a96e2467537f/README.md)。

### Zyphra 开放 1 亿激活参数的 ZONOS3-Mini，支持浏览器流式语音

**Zyphra，11 月 10 日 06:01:45（北京时间）上传 Apache 2.0 权重。** 8 亿总参数的 Gated DeltaNet／注意力混合 MoE 以 15 Hz 生成含一组语义和九组声学码的帧，因果 iSTFT 解码器随帧输出 44.1 kHz 音频。厂商用 112 张 H100 训练 2.5 天，报告浏览器首音时间在 Pixel 9 小于 570 ms、M3 MacBook Air 小于 150 ms。发布含 WebGPU、GGUF 工件并提供 SGLang-Omni 流式接口。可分冷启动和热缓存测端到端首音、实时系数与音质，避免把厂商示例延迟视作设备通用值。[厂商技术说明](https://www.zyphra.com/our-work/zonos3-mini) · [固定工件](https://huggingface.co/Zyphra/ZONOS3-Mini/tree/52ccfc874496d1a38449ae6b1c731a5f0db16a66) · [权重上传提交](https://huggingface.co/Zyphra/ZONOS3-Mini/commit/d78708d60375ab377501bad854bfe111b0b28d63)。

### 补录｜TwIL-8B 用额外的长度奖励训练出短推理版

**webAI，11 月 8 日 17:57:35（北京时间）公开权重。** 基于 Granite 4.2 8B，先全参 SFT，再用 GRPO/DAPO 得到 verbose，再以各领域的推理长度参考追加一轮 RL，得到默认 balanced。固定发布包含两个 BF16 检查点和 GGUF，采用 webAI 非商业许可。balanced 的 MATH-500 为 98.2%、AIME 2026 为 88.3%（每题约 19.5K 输出 token）。81.6% 的 LiveCodeBench 成绩来自 verbose，官方未记录此次题目版本和日期窗口，同行分数也未在同一配置复跑。对比两个检查点时，宜固定题集、采样和 token 上限，同时报告正确率与输出长度。[发布提交](https://huggingface.co/webAI-Official/TwIL-8B/commit/9563ba7dfb70eeca4a26654b195011cbf818e7d9) · [固定模型卡](https://huggingface.co/webAI-Official/TwIL-8B/blob/9563ba7dfb70eeca4a26654b195011cbf818e7d9/README.md) · [固定权重](https://huggingface.co/webAI-Official/TwIL-8B/tree/9563ba7dfb70eeca4a26654b195011cbf818e7d9)。

<a id="posttraining-models"></a>
## 后训练与数据工件

### NVIDIA 与 Elith 公布按日本家庭财务分布合成的用户画像

**NVIDIA、Elith，11 月 9 日 15:05:57（北京时间）公开数据提交。** Nemotron-Personas-Finance-Japan 发布八片 Parquet、CC BY 4.0，100 万条记录，每条含九类角色画像和共 50 个字段。先由概率图模型采样人口、地域和家庭金融属性，再用 Gemma 4 31B IT 生成日语叙述，NeMo Data Designer 负责结构化输出、校验和重试。厂商报告总量约 15 亿 token，角色叙述约占一半。它可为金融 Agent 模拟生成分层用户，覆盖家庭资产及数字、金融素养。分布依据统计和公开调查，叙述全部合成，评估时应独立核对分组覆盖及模拟偏差。[固定数据卡](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Finance-Japan/blob/c73cc865652320aa2a4ee1c48fcd8ccc8a3d6b94/README.md) · [八片 Parquet](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Finance-Japan/tree/c73cc865652320aa2a4ee1c48fcd8ccc8a3d6b94) · [生成流程](https://huggingface.co/blog/nvidia/nemotron-personas-finance-japan)。

### Open-H3-Regenerate-2K-IC 把原视频留在注意力上下文中

**NVIDIA SoL-Engine，11 月 9 日 10:17:15（北京时间）发布工件。** 以固定版 MiniMax-H3 Ref2VA 为基础，用 rank 384 的 LoRA 在每步去噪时读取干净的低分辨率参考视频，以保留面部、口型和文字。默认 `up3` 先放大草稿 latent、按 σ=0.835 加噪，再走 HyperFlow 八步网格的最后三步。另有四步与纯噪声起步的八步路径。公开工件含约 8.30 GB 的 IC LoRA、推理代码与原始权重子集，原基座许可带地域限制。项目页主要给视觉对比，复测可用同一批草稿比较口型、字幕保真与耗时。[项目方法和对比](https://nvlabs.github.io/Sana/Open-H3-Regenerate-2K/#ic) · [固定权重及代码](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K-IC/tree/54a7104b204fb46486cf85c2f600942118186d57)。

<a id="model-evals"></a>
## 评估与科研 Agent

### OpenAI 把研究作业的停止权限放进推理引擎

**OpenAI，11 月 9 日 21:00（北京时间）由官方 RSS 发布。** Control Room 为 RL、评估和数据采样等研究作业签发 `job_id`，请求仍直接发给推理引擎，引擎凭 ID 查中央授权。人工或监控系统撤销权限后，后续推理被拒绝，作业进程可能继续存活。授权缓存使官方报告的停止上界为 35 秒。Control Room 失联时默认拒绝的策略已覆盖前沿模型研究作业，其余研究作业仍在部署。这个机制可用于检查多推理引擎的授权一致性，停输出和停止已分派的外部动作须分别验证。[官方架构与边界](https://openai.com/index/control-room/) · [官方 RSS](https://openai.com/news/rss.xml)。

### Fable 5.2 的 PCB 设计经三个实体样机验证

**Anthropic 与 Diode Computers，11 月 9 日发布。** 给定简短需求，Fable 5.2 生成规格、电路、四层 PCB、外壳和固件，Diode 制板并由工程师装配、测试。三个样机均在未改电路的情况下工作。Diode 内部 PCB Gym 的 69% 对 25% 比较 Fable 5.2 和 Fable 5，公开 BenchCAD 2.1 的 83.5 对 44.4 则比较 5.2 和 5.1，后者为带工具的部分计分。这里的实体结果来自三项演示。复核可同时保留设计规则通过、人工介入、制造与上电验证记录。[联合实验](https://www.anthropic.com/research/claude-electronics-design) · [发布评测与计分](https://www.anthropic.com/claude-fable-and-mythos-5-2)。

<a id="papers"></a>
## 研究论文与评估

### 主窗｜AACL-IJCNLP 11 月 9 日工作坊与教程

#### 多语数学评估同时检验答案正确性和目标语言

**11-09 16:40–17:00（UTC+8）｜Multi-LLL 工作坊排程｜ACL Anthology 2026.multilll-1.2。** 作者用同题翻译估计中间激活的语言子空间，并以线性探针检查投影后仍可读出的语言信息。在 Qwen2.5-7B 的 11 语种 MGSM 实验中，最高准确率的编辑把准确率从 62.80% 提到 65.75%，但目标语言保持率从 99.82% 降到 80.58%，正确且使用目标语言的比例从 62.62% 降到 53.82%。按联合指标选出的秩一干预为 64.22%，参数在同一基准上选择，仍待留出集确认。把示例及回答提示切换成英语或中文也可带来类似的指标分离。11 月 9 日是报告排程，另有 11 月 3 日 arXiv v1 与 11 月 6 日 v2。 [工作坊日程](https://xiaoaugenstern.github.io/Multi-LLL-2026/#program) · [论文集版 §5–6、表 1–2 和 Limitations](https://aclanthology.org/2026.multilll-1.2.pdf) · [预印本版本历史](https://arxiv.org/abs/2611.05278)

#### 两场教程分别讨论评估可信度与部署后持续学习

**11-09 14:00–17:30（UTC+8）｜教程 T3/T4 排程。** T3 把有效性、稀疏评估与基准激励分开处理，作者页提供按问题组织的参考文献。T4 将部署后变化分为记忆、知识、工具、策略和参数更新等，并给出演示仓库。 [官方教程排程](https://2026.aaclnet.org/program/tutorials/#t3-a-tutorial-on-measurement-science-for-language-model-evaluation) · [T3 教材与参考文献](https://aimslab.stanford.edu/tutorial) · [T4 教材](https://megagon.ai/aacl_tutorial2026/) · [T4 演示仓库](https://github.com/megagonlabs/pdsi-tutorial)

### 延迟可见与待审补录｜arXiv 11 月 9 日公告

下列预印本在九分类的 11 月 9 日公告中核得。官方常规排程是 11 月 9 日 01:00 UTC，早于主窗起点 01:40:38 UTC。前次抓取仍显示 11 月 6 日公告，逐篇首次公开秒未确认，以下保留 v1 提交时间。 [arXiv 公告规则](https://info.arxiv.org/help/availability.html) · [cs.LG 公告列表](https://arxiv.org/list/cs.LG/new?show=2000)

#### RPEG 用执行图联合搜索 PPO 放置、并行与权重滞后

**11-06 09:28 UTC 提交｜arXiv:2611.09207v1。** RPEG 展开稳态训练图中的数据边和跨步权重发布边，并显式插入通信与重分片节点，让 rollout、打分和更新在给定权重滞后及显存约束下搜索排程。在同一 GH200 运行时中，固定同步 PPO、Llama 3.1 8B 和 actor/critic 结构，优化方案相对七个全共置基线的 20 个对照快 1.25–1.47 倍。跨框架与 verl 对照的中位加速为 1.488 倍，覆盖 8B 与 70B，包含实现差异。2,048 GPU 部分衡量解析模型的排程搜索，实机证据见 §VIII-C。公开仓库 571ff69 的 `RPEG/passes/lower.py::lower` 对应通信和重分片插入，`RPEG/schedule/scheduler.py::schedule` 对应依赖与设备约束下的排程。11 月 9 日 08:26 UTC 的该提交仅涉及 README 和 LICENSE。 [RPEG §IV–VIII、表 V 与图 10–12](https://arxiv.org/html/2611.09207v1#S8) · [固定实现与提交差异](https://github.com/spcl/RPEG/commit/571ff69fc80eab0fdb6da6c39718723ad294daef)

#### ResQAdam 给低秩 Adam 更新保留正交残差

**11-06 03:12 UTC 提交｜arXiv:2611.08924v1。** 维护 BF16 一阶动量，周期性求低秩基。主子空间做小矩阵 Adam 更新，正交残差按归一化后的符号方向补回，以省去全尺寸二阶动量。作者在单 H200、4,096 上下文、W4A4G128 微调的 Qwen3-4B-Base 上，优化器状态由 29.07 降至 7.78 GB，训练峰值从 99.14 降至 79.77 GB，吞吐由 9.64K 降至 9.56K token/s。优化器状态统计排除了词嵌入与 LM head，个别 W2A16 分块微调设置的质量波动较大。固定实现 `resqadam.py::ResQAdam._refresh_bases` 定期更新基并搬运低秩矩，`ResQAdam.step` 合并低秩 Adam 与残差符号更新。 [ResQAdam §3–5、表 2–3](https://arxiv.org/html/2611.08924v1#S5) · [固定实现](https://github.com/LiqunMa/ResQAdam/blob/5f833a8da63cc87fa86795ae6b240da6c01736c8/resqadam.py)

#### MINE 在单流解码里融合权重熵解码与矩阵向量乘

**11-06 12:51 UTC 提交｜arXiv:2611.09424v1。** dtANS-LLM 在读压缩权重的同一 CUDA kernel 内做矩阵向量乘。vLLM 插件按 checkpoint、GPU 和权重显存预算选择编码，也能回退原内核。在 Llama 3.1 8B Instruct 的低熵 EntQuant 2.1-bit checkpoint 上，相对作者测到的最快基线，RTX 4060 Ti 与 5090 的单流稳态解码分别快 2.52 和 1.48 倍。4096-token BF16 prefill 在 RTX 5090 上则为 12,860 token/s，vLLM 基线为 13,650。熵编码保留输入 checkpoint 中的权重值，吞吐结果取决于权重分布与批量。复现压缩包见论文附录说明，本轮未取到公开下载件。 [MINE §2、§4–6、图 3 与附录 A](https://arxiv.org/html/2611.09424v1#S6)

#### Nebula 让分布式路由器复用不同位置的 KV 片段

**11-06 13:54 UTC 提交｜arXiv:2611.09474v1。** Dynamo 路由器预记目标 worker 的 KV 位置及计算预约，再由 SGLang worker 的写入、驱逐和完成事件确认状态。未确认的位置有 TTL，计算预约等到完成或取消确认才释放。路由器再用淘汰历史调整前缀与位置无关缓存的软配额。两台服务器共八张 H20-3e、7P1D 回放四种 RAG/代理工作负载时，TTFT 中位数相对 Local-PIC + Dynamo 平均降 61.4%，相对 SGLang + Dynamo 降 76.7%。每个请求只生成一 token，观察范围主要是 prefill 和调度。论文链接的 GitHub API 在核查时返回 404，公开实现待确认。 [Nebula §5–9、图 7 与 13](https://arxiv.org/html/2611.09474v1#S9) · [作者标注的仓库](https://github.com/redai-studio/Nebula)

#### BitDecoding-FP4 在原生 FP4 attention 路径补偿量化误差

**11-06 18:13 UTC 提交｜arXiv:2611.09825v1。** 用两份量化 query 表示基值与残差，在 softmax 前合并分数，并针对概率下溢与 key 离群值做单独处理，保留两段原生 FP4 Tensor Core 矩阵乘。Qwen3-4B-Thinking-2507 的七项平均分在 BF16 KV、BitDecoding-NVFP4、TRT-LLM-NVFP4 下分别为 67.69、67.27、59.19。单 B300、256K 条件的 TPOT 分别为 7.74、4.28、4.63 ms。同文稀疏 MLA 测试报告 1M KV 占用从 90.46 降到 34.41 GiB，固定活跃窗口为 2,048，未给该路径的时延收益。 [BitDecoding-FP4 §3–5、表 1–2、附录 G](https://arxiv.org/html/2611.09825v1#S5)

#### ChainFlow 用潜空间轨迹起草，树验证只在低并发占优

**11-05 20:26 UTC 提交｜arXiv:2611.08669v1。** 在目标模型隐藏状态的低维空间外推并用一步 Euler 起草，候选树由目标模型统一验证。单 B300、vLLM 0.31.0、贪心生成的 Qwen3.5-4B，8×4 树相对自回归解码为 1.87 倍，匹配草稿参数量与训练数据的 DFlash 为 1.69 倍、EAGLE-3 为 1.35 倍，ChainFlow 链式为 1.23 倍。并发测试的树版在 4 路时为 1.12 倍，8 路降至 0.94 倍，DFlash 同两档为 1.48 和 1.39 倍。论文用定制 proposer 实验，正文未列实现下载链接。 [ChainFlow §3–4、表 1、3、9 与 Limitations](https://arxiv.org/html/2611.08669v1#S4)

#### AlterNet 用带深度信号的路由器迭代调用浅层专家

**11-05 21:46 UTC 提交｜arXiv:2611.08741v1。** 每轮按序列选择浅层专家，用已走深度和剩余预算作为 router 输入，专家另给停止信号。350M 实验中，作者把先走满深度的训练课程与高斯停止先验结合，避免只学第一层的塌缩。动态模型用固定深度对照 48.3% 的迭代数，C4 PPL 为 54.3，对照 40.1。动态模型约训练 3.9B token，固定对照训练步数约为其三分之一。这是迭代数与质量的早期权衡，尚缺等训练预算、长跑和实机时延对照。 [AlterNet §3–5、图 2–3 与附录 C](https://arxiv.org/html/2611.08741v1#S4)

#### LambdaMix 用同一偏好向量混合 LoRA 和奖励

**11-06 15:50 UTC 提交｜arXiv:2611.09634v1。** 训练时一个偏好向量同时选择 LoRA 的混合比例与各目标优势的权重，推理只调混合比例。Qwen3.5-9B 在 PKU-SafeRLHF 上，均衡偏好的 helpfulness 和 harmlessness 代理分为 5.586、8.487，提示词条件基线为 5.380、8.164。一次训练消耗 32K completions、存 13.2M adapter 参数，五个独立 specialist 合计 160K、33.1M。训练开销按 completions 计，评估采用两个目标的代理分。固定实现中 `ParetoLoRALinear.forward` 混合低秩增量，`ParetoGRPOTrainer._generate_and_score_completions` 以偏好加权替换 group 优势。 [LambdaMix §3、§5–6、表 3](https://arxiv.org/html/2611.09634v1#S5) · [固定适配器与训练实现](https://github.com/ckarouzos/lambdamix/tree/4efddb998858c364d7500748fa875d35a2330ded/src/lambdamix)

<a id="engineering"></a>
## 开源工程

本节按 GitHub PR 的合入时间或默认分支提交时间记录，正式版和预发布构建单列。时间均为上海时间。

### 训练、蒸馏与训推一致性

#### Megatron-Bridge 将一批 GB200/GB300 MoE 配方切到 NCCL EP

**11-10 08:57，`main` 默认分支合入。** Nemotron 3/3.5、Qwen3/3.5（含 VL）和 GLM 5/5.2 的选定训练配方从 HybridEP 换成 NCCL EP flex dispatcher，同时启用 grouped tensor，并为库启动路径加上每 rank 一卡的绑定。作者在 GB300 NVL72、NCCL 2.31.2 的 50 步 mock 数据实验中，Qwen3.5 35B-A3B 文本配方合并变更快 49.6%，单独换 dispatcher 快 8.8%。Nemotron 3 Nano 合并变更慢 0.9%，只换 dispatcher 快 0.8%。对比时应分别固定 grouped tensor 与 dispatcher，并看真实路由负载。 [Megatron-Bridge #6679 `81235a4b65`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6679)

#### DeepSpeed 保住 ZeRO-3 下 HybridEngine 的 QKV 布局转换

**11-09 20:18，`master` 默认分支合入。** 在 gather 后修改 GPT-NeoX/Megatron QKV 时指定 `modifier_rank`，确保 head-interleaved 到连续 Q/K/V 的变换广播后再重分片。回训练路径的逆变换也持久化。PR 报告 Pythia 同权重首 token 恢复一致，MI250/MI300X 以及独立 H200 复测跑过 20 步。较新的 Transformers 移除 `GPTNeoXAttention.hidden_size` 是独立兼容项。 [DeepSpeed #8392 `988c88a002`](https://github.com/deepspeedai/DeepSpeed/pull/8392)

#### DeepSpeed 修复 CPU offload 陈旧梯度与 engine 重建泄漏

**11-09 11:15–11-09 22:28，`master` 默认分支合入。** ZeRO-1/2 将 CPU 梯度累加改成每分片组一个 bit16 缓冲、按固定偏移切片，只在需要非边界累加时分配。重置时清掉 FP32 partition gradient，避免下一窗口未用参数再吃到旧梯度。另一变更在销毁时解绑 forward hooks、清理 ZeRO-3 全局模块栈及改用实例级协调器缓存等，使 engine 和 optimizer 可被回收。分别用交替闲置参数和循环创建/销毁且保留 model 的用例复验。 [DeepSpeed #9018 `a4c989b765`](https://github.com/deepspeedai/DeepSpeed/pull/9018) · [DeepSpeed #9034 `f32e164b94`](https://github.com/deepspeedai/DeepSpeed/pull/9034)

#### TorchTitan 分别处理 GDN 逐位一致与 DAPO 损失编译

**11-09 15:20–11-10 02:04，`main` 默认分支合入。** Qwen3.5 的训练和 vLLM rollout 共同固定 attn-gym 提交 `26beb36`，在 batch-invariant recurrent GDN 中使用同一公式、固定归约顺序及禁用 FMA 融合，消除 batch 形状选不同 launch 带来的数值差。另将 DAPO/GRPO 的 logprob、clip、surrogate 和指标放进同一个编译区。GB300 上 Qwen3-4B 输出阶段、TP2、10240 tokens、无 CUDA graph 为 22.2→17.1 ms。逐位模式继续使用 eager，常规编译结果与 eager 有舍入差异。 [torchtitan #5963 `72afa77a6c`](https://github.com/pytorch/torchtitan/pull/5963) · [torchtitan #5923 `2805bd6340`](https://github.com/pytorch/torchtitan/pull/5923)

#### TorchTitan 修复 LoRA 分片和异步 collective 梯度并调整 rollout 容量

**11-09 23:44–11-10 04:41，`main` 默认分支合入。** LoRA adapter 从基座权重的 DTensor 布局推导，覆盖 vision linear 和复制的 router gate。进入 `local_compile` 前等待异步 collective 输入，绕开 AOTAutograd unwrap 丢失梯度的路径。RL generator 按每 DP rank 的在途 rollout 份额设 `max_num_seqs`，通常给 2–4 倍余量，再受 batch token 预算限制，取消每 rank 固定 512 上限。 [torchtitan #5884 `6eb5cd7839`](https://github.com/pytorch/torchtitan/pull/5884) · [torchtitan #5886 `8a33dbf009`](https://github.com/pytorch/torchtitan/pull/5886) · [torchtitan #5944 `9c7a345fd8`](https://github.com/pytorch/torchtitan/pull/5944)

#### verl 修正离策略指标，更新 Ascend rollout 版本边界

**11-09 09:42–11-09 20:09，`main` 默认分支合入。** 全 padding 序列从逐序列 PPL、差值及重要性比率相关均值中排除，逐 token 统计原本使用有效 token mask。Ascend vLLM 引擎默认开 RL mode 以停用与热更新不兼容的 NZ 权重，保留显式配置，并把过渡 shim 收敛到 v0.30 系列。vllm-ascend ≥0.30.0 可用 sleep level 2，MTP 与 LoRA 原有 guard 仍生效。PR 在 GLM-5.2 的 65 步 Ascend GRPO 测试报平均 step 1250.84→1187.57 s、未见全零 reward，其他模型和拓扑需复测。 [verl #7785 `3ba1e05d52`](https://github.com/verl-project/verl/pull/7785) · [verl #8585 `69921c419c`](https://github.com/verl-project/verl/pull/8585) · [verl #8507 `f3b9d3d9ec`](https://github.com/verl-project/verl/pull/8507) · [verl #8608 `482f86fd09`](https://github.com/verl-project/verl/pull/8608)

#### TRL 收紧 rollout 配置并修正 Qwen SFT 的 thinking mask

**11-09 17:59–11-10 00:55，`main` 默认分支合入。** vLLM 同步前核对 PEFT 的 tied input/output embedding 是否仍为同一张量，分裂副本会在启动时提示 `ensure_weight_tying=True`。AsyncGRPO/AsyncDistillation 统一用 token budget 划微批，非正预算现在报错，`None` 从 vLLM 读上限。Triton logprob、divergence、entropy 和实验性树 GDN 显式关闭隐式 FP fusion，以对齐 3.9 与 3.10 默认行为。可识别的 Qwen3.6/3.8 模板即便自带 generation 标记，也优先用保存的训练模板，把已预填的 thinking 前缀移出 assistant-only loss。 [trl #8144 `d0dda99dd3`](https://github.com/huggingface/trl/pull/8144) · [trl #8168 `dcdeae5731`](https://github.com/huggingface/trl/pull/8168) · [trl #8185 `8038ac2a4e`](https://github.com/huggingface/trl/pull/8185) · [trl #8184 `5623e8b639`](https://github.com/huggingface/trl/pull/8184) · [trl #8187 `e87ee31d90`](https://github.com/huggingface/trl/pull/8187) · [trl #8199 `5a423556bc`](https://github.com/huggingface/trl/pull/8199)

#### verl-SpeCo 合入 SFT 协训并修复 DeepSeek 空监督

**11-09 10:18–11-09 11:55，`main` 默认分支合入。** 新增 `speco.mode=sft` 的独立 trainer/worker 与启动入口，SFT 前向收集目标模型指定层的隐藏状态并协训 DSpark drafter，支持从头或已有草稿权重启动。作者确认流程与损失下降，未给接受率或加速对照。DeepSeek 三套模板修正角色分隔符，R1 Distill 补 EOS，避免 parser 找不到 assistant span 后产生全零 loss mask，已用 CPU 字符级回归验证。 [verl-SpeCo #117 `1fbf101ac1`](https://github.com/verl-project/verl-SpeCo/pull/117) · [verl-SpeCo #138 `9b6e2a6076`](https://github.com/verl-project/verl-SpeCo/pull/138)

#### NeMo RL 在 deadline 保存环境状态，Bridge 输出 GPT-OSS MXFP4

**11-10 01:45–11-10 07:30，`main` 默认分支合入。** NeMo RL 的 turn-level 恢复在墙钟 deadline 主动落 rollout snapshot，`snapshot_on_trainer_checkpoint=false` 配合空定时器可只在截止时保存。Bridge 经 ModelOpt 在拼接专家后生成 OpenAI 格式的 MXFP4 `*_blocks` 和 `*_scales`，覆盖 GPT-OSS weight-only experts。前者应核对断点后的未完成 turn，后者应核对部署端 quant method 与首 token。 [RL #5056 `b083540703`](https://github.com/NVIDIA-NeMo/RL/pull/5056) · [Megatron-Bridge #6787 `0f43c3905c`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6787)

#### PyTorch 修正 FSDP CPU 梯度溢出、单分片缓冲与 graph backward

**11-09 17:19–11-10 07:43，`main` 默认分支提交。** CPU offload 的 ShardedGradScaler 取 CPU scale 和 found-inf 改用阻塞拷贝，防止 CPU 先读旧值。FSDP2 单 rank shard group 直接复用已打包的 reduce 输入，省一份梯度大小的复制缓冲，常规多 rank reduce-scatter 继续走原路径。Inductor 把 cudagraph-tree manager 查询延后到 backward 执行，覆盖 forward 被捕获而 backward 因无 kernel 或其他原因未捕获、且先被 lowering 或命中 AOT cache 的情形。 [pytorch #203513 `fc6ef2c210`](https://github.com/pytorch/pytorch/commit/fc6ef2c210e016f541ff0baacdde73eef9c5d8eb) · [pytorch #203556 `e2c2673d00`](https://github.com/pytorch/pytorch/commit/e2c2673d00500d268fcb1cc63fa7cc3173876505) · [pytorch #202857 `0eeca006d9`](https://github.com/pytorch/pytorch/commit/0eeca006d930450cf7b950a3532228b84026d9f1)

### 推理、KV 缓存与服务

#### vLLM 接通 NIXL OBJ 加速参数，ROCm KV offload 改用 SDMA

**11-09 18:39–11-09 18:45，`main` 默认分支合入。** OBJ 二级 tier 透传 `store_config.backend_params`，CPU DRAM 每块注册上限为 4 GiB − 64 KiB。固定的 NIXL 1.5.2 中仅开 `accelerated` 不指定 `type` 仍走普通 S3，通用 RDMA 引擎尚在 NIXL 主干。L40S/Qwen2.5-0.5B 的 40 条回放用未发布厂商插件验证 RDMA 与 KV 恢复一致，reload 约 114 ms，冷重算约 110 ms。另一 PR 在 HIP ≥7.16 给 SimpleCPUOffload 批拷贝设置 PreferCE，单 MI355X 测并行 decode 的最差中位减速由 122% 降至 3.9%，带宽 44–56 GB/s。 [vllm #56192 `85b52c9f53`](https://github.com/vllm-project/vllm/pull/56192) · [vllm #65559 `f86a3e29d7`](https://github.com/vllm-project/vllm/pull/65559)

#### vLLM 默认在符合条件的 NVLink PP 上使用 copy engine

**11-10 08:09，`main` 默认分支合入。** 发送方以单边拷贝加信号交接，接收方首次读取张量时才排队等待，释放 GPU SM 给计算。`VLLM_PP_TRANSPORT=auto` 要求 TP1、PCP1、同节点 NVLink P2P 且全部 PP rank 支持，否则回 NCCL。GB300、MiniMax-M3.1-Flash、TP1/PP2、100K 缓存前缀的合成负载下，高负载的 1.5–10K 新 token 组 p50 TTFT 低 13–20%。轻负载接近基线。该结果限实测模型、拓扑和流量。 [vllm #65896 `ce896d69cc`](https://github.com/vllm-project/vllm/pull/65896)

#### vLLM 落地 Kimi K3.1 TP8 的 KV 和 sparse MLA 部分

**11-09 12:31，`main` 默认分支合入。** 支持按模型确定 packed hybrid 行大小，将 packed token slot 映射到 kernel block，修复 ShardQ 仅传本 rank indexer 行时重复按全量索引导致的越界。FlashInfer 0.7.3 的 auto 仍选 TRTLLM-gen，更新版才可选相应 Prims-TS 路径。此项是当时拆分的第 3 部分，作者只在 2×4 B300 用 8 层 dummy 权重确认启动和 32 并发生成，未测正式权重精度。 [vllm #66037 `4088c2838a`](https://github.com/vllm-project/vllm/pull/66037)

#### vLLM 修复批次不变 attention 和结构化投机的两条正确性路径

**11-10 05:01–11-10 05:28，`main` 默认分支合入。** 启用 `VLLM_BATCH_INVARIANT` 时，SM100 head-size 256 不再因多 query 采用另一套 tile 和 warp。B200 上 BF16 Gemma 3 1B 在三种上下文的选中 token logprob 恢复逐位相同。该短测 FULL graph 慢 4.5%。异步调度的结构化请求隔一批再次排入时，会在生成 grammar bitmask 前取回真实 draft token，避免读 -1 占位符。 [vllm #60531 `692cf2d75a`](https://github.com/vllm-project/vllm/pull/60531) · [vllm #66081 `d5bd3d1f92`](https://github.com/vllm-project/vllm/pull/66081)

#### vLLM 修复 DP+EP 的 MoE LoRA 和 TP expert bias

**11-09 16:54–11-10 00:30，`main` 默认分支合入。** EP dispatch 始终刷新 token→adapter slot 映射，不再用本 rank 的 no-LoRA 状态决定远端 token 是否应用 LoRA。空闲 DP rank 的 dummy batch 保留已加载 adapter。通用 MoE loader 只在 TP rank 0 保留 down-projection bias，避免 all-reduce 前重复相加。可分别用跨 DP rank 混合 adapter 流量和 TP1/2/4 的带 bias 检查点验证。 [vllm #66486 `1648e80a6a`](https://github.com/vllm-project/vllm/pull/66486) · [vllm #65406 `6d0de315ec`](https://github.com/vllm-project/vllm/pull/65406)

#### vLLM 扩展 MoRI P/D pipeline 与 HiSparse 预热驻留

**11-09 22:34–11-10 06:39，`main` 默认分支合入。** MoRI WRITE 的 prefill PP 按 stage 分配连接和地址，按层名映射 decode KV，等待全部 stage 到齐。GLM-5.2/5.3 的 8×MI355X 试验中，TP1×PP4 对比 TP4 prefill 容量约四倍。HiSparse 在导入完成且 GPU 池高于水位时批量 H2D 提升页驻留，NIXL 仍写 host tier。GLM-5.3 NVFP4、GB300 1P1D、64K 多轮试验中 p50 ITL 11.36→10.40 ms、p50 TTFT 1469→1954 ms。 [vllm #64693 `56e5608d0f`](https://github.com/vllm-project/vllm/pull/64693) · [vllm #65888 `520bb1c1d5`](https://github.com/vllm-project/vllm/pull/65888)

#### vLLM 改善保密计算下 V2 decode 并扩展量化加载

**11-10 04:04–11-10 08:30，`main` 默认分支合入。** NVIDIA Confidential Computing 环境绕过不一致的 UVA host 映射，改用 staged copy 避免 V2 读取旧输入。普通模式仍可走 UVA。另新增 compressed-tensors E2M3 MXFP6 权重、动态 MXFP8 或 16 位激活，CUDA 使用 Humming，升级 compressed-tensors 到 0.20.1。ModelOpt 的 `INT8_PB_WO` 支持整模及 MIXED_PRECISION 中的按块 INT8 权重、16 位激活线性层。 [vllm #59981 `cb628eee9e`](https://github.com/vllm-project/vllm/pull/59981) · [vllm #61666 `b079d74bd5`](https://github.com/vllm-project/vllm/pull/61666) · [vllm #65272 `87431e48d3`](https://github.com/vllm-project/vllm/pull/65272)

#### SGLang 统一 KV 布局规划与 P/D 页生命周期

**11-09 19:22–11-09 19:26，`main` 默认分支合入。** 启用 `--enable-unified-memory` 后，P/D 在途传输按请求行跟踪，页等传输静止再复用。请求行的持有与释放统一维护。P/D 下排除异构 TP staging buffer。admission、容量查询、实际分配和 reclaim 共用布局搜索，按需选择不移动、端部压实或浮动池重排。每个 SWA 子池有独立 host pool、P/D wire key 与 metrics，传输前核对窗口和布局。多窗口当前只支持 Triton 普通 MHA eager extend/decode 与 decode graph，投机、DCP、确定性和 MLA 等仍受限。 [sglang #48009 `72edb269c2`](https://github.com/sgl-project/sglang/pull/48009) · [sglang #48010 `7455745a5a`](https://github.com/sgl-project/sglang/pull/48010) · [sglang #48011 `6ea644d6ce`](https://github.com/sgl-project/sglang/pull/48011) · [sglang #48012 `2cbb66f563`](https://github.com/sgl-project/sglang/pull/48012) · [sglang #48013 `c89cedec26`](https://github.com/sgl-project/sglang/pull/48013) · [sglang #48014 `388d03b775`](https://github.com/sgl-project/sglang/pull/48014)

#### SGLang 修正 P/D 失败释放与 pause、断流收尾

**11-09 14:12–11-10 04:47，`main` 默认分支合入。** 立即释放失败 staging transfer 前先注销 room，避免接收线程用已经清空的 request index 把迟到数据写进其他请求 KV。pause/retract 汇总所有 PP 微批和 HiSparse/dLLM 持有者去重，再按在途 step 边界释放。Chat Completions 流关闭时显式关闭内层生成器，逐输出检查断开，复用带请求状态身份校验的 abort。 [sglang #49218 `fb359066c2`](https://github.com/sgl-project/sglang/pull/49218) · [sglang #49253 `34f49c9d71`](https://github.com/sgl-project/sglang/pull/49253) · [sglang #46586 `c81230bffe`](https://github.com/sgl-project/sglang/pull/46586)

#### SGLang 修复滑窗 ragged prefill 和 DSA prefill 元数据

**11-09 10:57–11-10 08:22，`main` 默认分支合入。** FlashInfer hybrid SWA 用分开的 plan，让窗口值在规划时入核，SM100 的 CUTLASS ragged 路径对窗口无效时选支持窗口的 FA2。DeepSeek V4/V4.1 的 CP+DSpark bounded replay 用紧凑尾部的全局行索引 gather，避免把 32 行当 4096 行。FP8 KV 的混合 chunk prefill 在 BF16-only sparse backend 上将 MIXED 路由到 RAGGED top-k，补齐 CUDA graph metadata。 [sglang #46214 `3e8d7c9e09`](https://github.com/sgl-project/sglang/pull/46214) · [sglang #47406 `656ac7b1d4`](https://github.com/sgl-project/sglang/pull/47406) · [sglang #44176 `eafc49148e`](https://github.com/sgl-project/sglang/pull/44176)

#### SGLang 修复 Kimi K3 热更地址与 RL routed-expert 捕获

**11-09 18:04–11-10 06:26，`main` 默认分支合入。** 复用已被 graph 捕获的 MLA、MoE、BFA、KDA 缓冲地址并原位刷新 AttnRes 派生值，前提是模型结构、形状、dtype、device 保持一致。DCP 或 KV sharding 下将 CPU routed-expert 和 indexer top-k 捕获容量扩到分配器可发出的所有 slot，修正长时间 RL run 才触发的越界。ROCm 原生热更和所有量化 loader 未在热更 PR 中全测。 [sglang #43675 `e60f7e909b`](https://github.com/sgl-project/sglang/pull/43675) · [sglang #47529 `7b15055863`](https://github.com/sgl-project/sglang/pull/47529)

#### TensorRT-LLM 修复 KV teardown 和分离式取消并接通 Qwen3.5 FP8 GDN

**11-09 19:46–11-10 07:25，`main` 默认分支合入。** KVCacheManagerV2 析构时切到构建时记录的 CUDA context，避免无 current context 的 GC 线程触发进程级故障锁存。分离式取消不再进新的 model batch，旧 batch/传输资源仍按原回收边界保护，保留 PP1、纯文本及 transceiver 前提。Qwen3.5 接收 FP8 per-channel/per-token GDN 投影，RMSNorm 与逐 token FP8 量化融合，保留未量化 in_proj_ba 的 BF16 输出。 [TensorRT-LLM #21795 `82cf1afb0a`](https://github.com/NVIDIA/TensorRT-LLM/pull/21795) · [TensorRT-LLM #21508 `df908c548e`](https://github.com/NVIDIA/TensorRT-LLM/pull/21508) · [TensorRT-LLM #21287 `85fe425c45`](https://github.com/NVIDIA/TensorRT-LLM/pull/21287)

#### LMCache 与 Dynamo 更新缓存隔离和编排

**11-09 15:45–11-10 07:38，`dev`、`main` 默认分支合入。** LMCache token-dropping 请求禁止用原始前缀发布裁剪后的 KV，RDMA workers 可分散到多个 fabric 接口。Shadow L1 新增索引 peer 借用内存视图的契约，具体借用 adapter 在后续变更。Dynamo 修复 speculative prefill 转发时的渲染上下文，依赖 dynamo-tokenizers 4.1.2 对 Kimi K3 分段前缀缓存。应分别核对裁剪请求隔离、peer 生命周期和缓存后渲染等价性。 [LMCache #5875 `1cbd2ded67`](https://github.com/LMCache/LMCache/pull/5875) · [LMCache #6293 `ad2cb65fbe`](https://github.com/LMCache/LMCache/pull/6293) · [LMCache #6207 `115e09a4c4`](https://github.com/LMCache/LMCache/pull/6207) · [dynamo #18185 `be1ccbf94a`](https://github.com/ai-dynamo/dynamo/pull/18185) · [dynamo #17741 `f3c0e14a10`](https://github.com/ai-dynamo/dynamo/pull/17741)

### 算子、编译与后端

#### FlashInfer 让 GDN ReplaySSM 验证和提交任意树路径

**11-10 04:59–11-10 05:24，`main` 默认分支合入。** verify 接收 `verify_parents` 并逐节点读父状态，commit 用 `accept_paths` 跳过未选兄弟节点，覆盖不能用接受长度表达的树路径。Qwen3-Next-80B 的树 MTP 降时延数据来自 SGLang fork，库内核合入时间单列。另一 PR 为 CuTe recurrent KDA 增加 FP32 状态，仅 one-warp/grouped 路径，T=1 快路和 Cake/packed 特化仍限 BF16。 [flashinfer #6257 `77bf17dccc`](https://github.com/flashinfer-ai/flashinfer/pull/6257) · [flashinfer #6155 `cabc2aece4`](https://github.com/flashinfer-ai/flashinfer/pull/6155) · [flashinfer #8833 `c1eed07c0e`](https://github.com/flashinfer-ai/flashinfer/pull/8833)

#### FlashInfer 修复 cuBLASLt tactic 回放并自动调优完整 EP

**11-10 01:36–11-10 02:42，`main` 默认分支合入。** cuBLASLt 用算法描述符作为 tactic，缓存按 M bucket 正确命中且在目标形状校验，不再用随 M 变化的 heuristic 下标。MoEEpLayer 的 AutoConfig 比较整个 Mega 与 split EP 前向，覆盖 dispatch、compute、combine 与输入输出准备。split 候选绑定各自通信后端，保留仍被旧 graph 引用的 workspace。可用同 bucket 不同 M 和空 rank 验证。 [flashinfer #6148 `f5c6aa76b9`](https://github.com/flashinfer-ai/flashinfer/pull/6148) · [flashinfer #8576 `76747542f7`](https://github.com/flashinfer-ai/flashinfer/pull/8576)

#### FlashAttention 修正 FA2 大步长并优化 SM100 FA4 backward

**11-09 11:16–11-10 03:09，`main` 默认分支合入。** FA2 split-KV 的 K/V block 步长、分片累加及 combine 索引在相乘前升为 64 位。GH200 原测试出现非法访问，修复后大 stride 与大 accumulator 用例通过。CuTe SM100 hdim64 backward 把 LSE/dPsum 减项折入 S/dP MMA，仅限 BF16 非确定性 MHA、1 CTA 与 64 维路径，无 block sparsity、score modifier、softcap，结果与未折叠版存在舍入差异。 [flash-attention #3101 `478b5b012f`](https://github.com/Dao-AILab/flash-attention/pull/3101) · [flash-attention #3098 `c1709db7d4`](https://github.com/Dao-AILab/flash-attention/pull/3098)

#### Triton 修正 warpgroup 寄存器派发并约束 NVFP4

**11-10 03:29–11-10 07:10，`main` 默认分支合入。** 通用 warp specialization 让物理 warpgroup 走同一条 `setmaxnreg`，再按 warp ID 分出工作与 padding，避免多份指令地址的未定义行为。dense matmul 对 A、B 分别校验 NVFP4 tensor scale，其他格式包括 MXFP4 拒绝该 scale。NVFP4 输入或输出量化要求 Blackwell+ CUDA，Hopper 上旧的 BF16 fallback 组合现在报错。 [triton #12677 `dddf5617b6`](https://github.com/triton-lang/triton/pull/12677) · [triton #12680 `ccfdb2fbd8`](https://github.com/triton-lang/triton/pull/12680)

#### Transformers 修复 packed mask 并减少 GDN 复制

**11-09 22:23–11-10 02:41，`main` 默认分支合入。** packed sequence 上 chunked 或自定义 mask 按各文档内 position 重映射，避免以整条 packed offset 划 chunk。FLA ≥0.5.0 直接吃 grouped V heads 的 Q/K，旧 FLA 保留 repeat，Torch fallback 自行展开。LayerDrop=0 时训练图跳过随机数和数据依赖分支，使相应模型可走 `torch.compile(fullgraph=True)`。 [transformers #49141 `6096273e98`](https://github.com/huggingface/transformers/pull/49141) · [transformers #50513 `fa5127c500`](https://github.com/huggingface/transformers/pull/50513) · [transformers #50475 `147e66b348`](https://github.com/huggingface/transformers/pull/50475)

#### Ascend 路径修复宽投机验证、缓存块与热更映射

**11-09 11:17–11-09 23:10，`main` 默认分支合入。** vLLM-Ascend 对宽 GDN verify window 使用正确序列状态。超过 8 token 的窗口走现有 Triton recurrent，按真实累积 offsets 和二维状态索引恢复。DeepSeek V4.1 SWA 使用配置的 block size。DeepSeek V4 在 DSpark/EPLB 的冗余专家场景改正权重映射，SFA sender 绑实际设备，非复制 native tiering 的 CPU 缓存 slot 按 worker rank 区分。VeOmni 另更新 `fla_npu` v26.10.0，修正 Ascend950 GDN backward 的 NaN。应分别用实际 A2/A3/A5 组合核对版本。 [vllm-ascend #16292 `722a244be9`](https://github.com/vllm-project/vllm-ascend/pull/16292) · [vllm-ascend #21243 `8903f9acb5`](https://github.com/vllm-project/vllm-ascend/pull/21243) · [vllm-ascend #21168 `420d5cec42`](https://github.com/vllm-project/vllm-ascend/pull/21168) · [vllm-ascend #19826 `f622e04c60`](https://github.com/vllm-project/vllm-ascend/pull/19826) · [VeOmni #1369 `4f7808b79e`](https://github.com/ByteDance-Seed/VeOmni/pull/1369)

#### ROCm ATOM 支持 V4.1 状态随 KV offload 并修复投机越界

**11-09 16:32–11-09 18:09，`main` 默认分支合入。** DeepSeek-V4.1-Flash 的 vLLM 插件将 SWA/compressor 环和 Engram cursor 每 256 token 存进 LMCache MP 对象，prefix 命中只能落在已存状态边界，DSpark 仍留至少两 token 的 prefill 尾。Sparse MLA MTP 的 metadata 只按真实 scheduled token 规划，避免 graph padding 让未填充 drafter 的 Q/O 越界。DSpark block drafter 预留 k 而非 k−1 个位置，防止最后一次写入误落物理 block 0。 [ATOM #2598 `b986b6535a`](https://github.com/ROCm/ATOM/pull/2598) · [ATOM #2715 `8e48d47ca3`](https://github.com/ROCm/ATOM/pull/2715) · [ATOM #3098 `e87b0a5250`](https://github.com/ROCm/ATOM/pull/3098)

#### llama.cpp 修复 tools 与 JSON schema 同用时报 400

**11-10 08:44，`master` 默认分支合入。** Qwen3 coder、DeepSeek、Ministral 3 的工具解析在指定 response_format 时改用非 lazy grammar，避免有 schema 规则却没有 lazy trigger，导致空 root。新增 Qwen3.5、Ministral 3、DeepSeek V4 的 tools+JSON schema 回归。 [llama.cpp #31956 `3b4d6ab3f7`](https://github.com/ggml-org/llama.cpp/pull/31956)

### 特性分支

#### TensorRT-LLM 在聚合 Helix 分支恢复 KDA 的前缀复用

**11-10 02:35，`feat/helix-agg` 特性分支合入。** KDA 状态只在 snapshot 上能恢复，`feat/helix-agg` 将周期、偏移、分支和按请求生成的 snapshot 向下对齐到 `tokens_per_block × cp_size` 的 ledger block。Kimi K3 的 64×2 例子为 128 token，256-token snapshot 规则下只复用严格早于 prompt 末尾的最近边界。状态按此特性分支记录。 [TensorRT-LLM #21941 `7a3074eb7a`](https://github.com/NVIDIA/TensorRT-LLM/pull/21941)

### 回溯窗口主干补录

#### Megatron-LM 检查 MFSDP 梯度 dtype 并融合本地累加

**11-07 05:06–11-08 05:43，`main` 默认分支提交补录。** 显式梯度 dtype 与 `grad_reduce_in_fp32` 在配置阶段核对，避免 MFSDP 与普通 DDP 对同一开关有隐含差异。MFSDP v2 的本地分片梯度用 `destination.add_(grad, alpha=grad_scale)` 合并缩放和累加，累加 dtype 与通信 dtype 分开，舍入次数变化值得用小梯度、多个 microbatch 和低精度通信复验。 [Megatron-LM #8922 `73c11cf617`](https://github.com/NVIDIA/Megatron-LM/commit/73c11cf61773a94581dfdf74d4a0f511c48e2b4b) · [Megatron-LM #8393 `fe28a3bc6a`](https://github.com/NVIDIA/Megatron-LM/commit/fe28a3bc6a62d9fdbefc0937c907d7d9920804ef)

### 发行与工件

#### KAI Scheduler 两个维护版修复已删除 PodGroup 的重试

**11-09 16:59，正式维护版。** 两版 status-updater 遇到已删除 PodGroup 时丢弃 status/patch 更新，结束无休止的 NotFound 重试。v0.14.12 同时修复回收遍历遇未超配队列便停止而跳过后续合法 victim 的路径。两版 Helm chart 均于 GitHub release 发布约 10 分钟后上传，升级可分别核对日志和抢占队列次序。 [v0.14.12 `2e53266a7d`](https://github.com/kai-scheduler/KAI-Scheduler/releases/tag/v0.14.12) · [v0.16.16 `87d2428b54`](https://github.com/kai-scheduler/KAI-Scheduler/releases/tag/v0.16.16)

#### llama.cpp b12238 预发布构建带 OpenAI 兼容语音生成入口

**11-10 08:23，预发布构建。** 新 `/v1/audio/speech` 复用 TTS 请求解析，multipart speed 等字段先转类型，WAV 和 PCM 共用 s16 舍入。测试使用 tinyLFM3-Omni GGUF。release API 标记 prerelease，包含多个平台构建。使用时应核对实际下载平台与 TTS 模型能力。 [b12238 `08f945ffb5`](https://github.com/ggml-org/llama.cpp/releases/tag/b12238)

<a id="hardware"></a>
## 硬件与软件协同

### Google 披露 TPU 8tx 和 TPU 8i Max 两款第八代系统

**Google Cloud、AMD · 11 月 9 日 21:00（上海时间，Google 公告元数据）。** TPU 8tx 复用 TPU 8t 的硅片，配第五代 AMD EPYC 主机，采用面向低延迟推理的 Boardfly 互联。Google 表示硬件隔离可保护使用中的数据与权重，目标包括保密推理和小规模训练。相对 8t 的最低约一半延迟是厂商口径，公告未列出通信或模型负载、切片规模、启用保密功能前后的对照，以及证明链和威胁模型细节。TPU 8i Max 将两颗 8i 芯片放进一个封装，Google 披露封装级计算量、SRAM 与 HBM 带宽各为 8i 的两倍。公告没有给出可直接采购的地区、价格、长上下文实测或双芯片间的内存可见性。**继续观察部署文档**，特别是 8tx 的保密边界和 8i Max 的跨芯片分片与 KV 容量。 [Google 原始公告](https://blog.google/innovation-and-ai/infrastructure-and-cloud/global-network/tpu-8i-max-and-8tx/) · [AMD 主机说明](https://newsroom.amd.com/news/amd-epyc-expands-leadership-foundation-ai/) · [8t 与 8i 的原架构说明](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive/)

### TPU Sync 开源披露补齐 KV 分层与训推权重传输的技术细节

**Google Cloud · 11 月 9 日公告，既有代码与预发布 wheel。** TPU Sync 通过原生 C++ 异步 DMA 在 HBM 与页锁定 DRAM 间传递保持 TPU 物理分块布局的张量，再由 DCN 把已到达的块发往 decode 主机。Raw PJRT buffer alias 允许传输与计算重叠，控制器管理租约和恢复，KV manager 处理本地分层。二级存储溢出和 RL 权重同步仍标为实验性，RL 原语可对不同训推分片布局做重分片。PyTorch 与 JAX 都有 wheel。核对到同日上传的 `0.0.1.dev20261109081924`，代码与预发布包早于本次技术公告。 [官方架构和支持状态](https://cloud.google.com/blog/topics/ai-infrastructure/introducing-tpu-sync-fast-kv-cache-and-weight-transfer-on-tpu) · [代码库](https://github.com/google/tpu-sync) · [PyTorch 工件](https://pypi.org/project/tpu-sync-torch/0.0.1.dev20261109081924/) · [JAX 工件](https://pypi.org/project/tpu-sync-jax/0.0.1.dev20261109081924/)

Google 在 Ironwood v7x 2×2×1、PCIe Gen5 x16/芯片、每主机一或两张 200 Gbps 网卡、tpu-inference 0.24.0、JAX 0.10.2、libtpu 0.0.42.1 的内部测试中，报告 Qwen3 235B 的 P/D KV 传输中位数由数秒降至约 230 ms，低并发平均 TTFT 下降 60%。另一个 480B 低并发案例的平均 TTFT 下降 90%，文章未披露其模型名、精度及完整负载。64K 重复前缀测试是另一个场景。**考虑复现**时分别扫前缀命中率、批量、输入输出长度和并发，把 DMA、跨机传输与请求 TTFT 分开统计。 [分场景结果及脚注](https://cloud.google.com/blog/topics/ai-infrastructure/introducing-tpu-sync-fast-kv-cache-and-weight-transfer-on-tpu)

### Cloud Pathways 与 GKE TPU 动态切片进入正式可用

**Google Cloud · 11 月 9 日，两个 GA。** Pathways 在 GKE 上通过 IFRT proxy 让一个 JAX 客户端异步调度多个 TPU slice，保留 colocated Python sidecar 运行加载与 checkpoint。未计划中断时可从 host-memory snapshot 重分片到健康 slice，计划中断时则有将 HBM 状态持久化到 Cloud Storage 的 suspend/resume。公告支持 v5e/v5p/v6e/TPU7x、JAX 0.10.1 起的指定版本，Cluster Director/Slurm 仍是预览。厂商案例给出芯片小时和工作负载类型，未披露足以横向对照的端到端故障演练数据。**建议深入阅读**实际部署指南，区分 slice 热替换、弹性缩容和持久化恢复的状态来源。 [Pathways GA 公告](https://cloud.google.com/blog/topics/ai-infrastructure/cloud-pathways-a-distributed-ml-runtime-is-ga) · [官方版本与恢复条件](https://docs.cloud.google.com/release-notes#November_09_2026)

动态切片把物理 nodepool 和逻辑作业拓扑解耦。Ironwood 的 super-slicing 由 OCS 把多个 4×4×4 cube 连起来，sub-slicing 可在同一 cube 内切出更小分区。GA 仍要求 TPU All Capacity 的 reservation 模式。Google 相对于为每个切片新建静态 nodepool 给出最高 5 倍启动加速和 4.5 倍恢复加速，未给可复现的负载分布或完整拓扑矩阵。**考虑验证**多租户环境的排队时间、重配后的恢复和数据面吞吐。 [动态切片 GA 公告](https://cloud.google.com/blog/products/containers-kubernetes/gke-dynamic-slicing-on-tpus/) · [版本、拓扑和部署限制](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/dynamic-slicing)

### NCCL 2.32.6-1 发布，包含 Rubin 调优和多项通信正确性修复

**NVIDIA · 11 月 10 日 03:28:46（上海时间），GitHub Release `v2.32.6-1`。** 新增 Vera Rubin 的 Ring/Tree/NVLS/NVLSTree 流水成本模型、对称 collective 模型、32–40 GPU NVLink 域拓扑搜索优化和 tuner 插件 v7。Flow device API、基于它的 GENK、以及小中消息的对称 `AlltoAll_LL` 仍是实验路径，GENK 需要构建和运行两层开关。诊断方面，profiler 插件 v8 可采集 NCCL 产生的 CPU/GPU 时间戳，Inspector 输出选中的算法和协议。正确性方面，GPU 枚举被 `CUDA_VISIBLE_DEVICES` 重排时，单机 NVLS AllGather/ReduceScatter 曾返回错误的 rank 顺序。该版还修复注册内存通信的 rendezvous 标志竞态、NET/IB 请求计数回绕以及对称缓冲区浮点 `ncclAvg` 的 CUDA 错误。**建议优先做正确性回归**，覆盖设备顺序、注册 buffer、高请求计数以及既有插件 ABI。发布记录没有可对比的 Rubin 模型端到端数据。 [官方发布与已知限制](https://github.com/NVIDIA/nccl/releases/tag/v2.32.6-1) · [NVLS rank 顺序](https://github.com/NVIDIA/nccl/pull/2377) · [注册缓冲区竞态](https://github.com/NVIDIA/nccl/pull/2390) · [浮点平均归约](https://github.com/NVIDIA/nccl/pull/2448)

### AITER v0.1.29 发布，低精度 attention 与大 EP 缓冲区修复进入版本

**AMD ROCm · 11 月 9 日 18:52:09（上海时间），`v0.1.29`，八个 wheel 随后上传。** 发布说明列出 gfx942/gfx950 的 ROCm 7.2、7.14、10.0、10.1 与 Python 3.10/3.12 组合。release 内的 gfx1250/RDNA 代码有各自构建与硬件边界。gfx950 增加带 NVFP4 prefix KV 的 MLA v4.1 prefill 和 query length 1 的共享 tile MXFP4 KV paged decode。BF16 GQA32 MLA decode merge 仅归并前一阶段写过的 split，避免消费未写结果。大 EP 路径的 FlyDSL grouped GEMM 改用裸地址传参处理超过有符号 32 位维度范围的 buffer。也纳入单机 Ulysses all-to-all 与融合 dispatch/GEMM1/GEMM2/combine 的 MegaMoEV2 源码。**考虑按设备复验**，优先测短尾页、非均匀请求、超大 EP buffer 和现有 checkpoint 的精度。各 PR 的局部测试未构成统一负载的升级收益。 [正式版本和 wheel](https://github.com/ROCm/aiter/releases/tag/v0.1.29) · [NVFP4 MLA #7124](https://github.com/ROCm/aiter/pull/7124) · [MXFP4 KV #7056](https://github.com/ROCm/aiter/pull/7056) · [BF16 split #7182](https://github.com/ROCm/aiter/pull/7182) · [EP buffer #7299](https://github.com/ROCm/aiter/pull/7299) · [Ulysses #5815](https://github.com/ROCm/aiter/pull/5815) · [MegaMoEV2 #6881](https://github.com/ROCm/aiter/pull/6881)

### HCCL 为 hostDPU 加入 inline reduce 选路和全域协商

**CANN/HCCL · 11 月 9 日 11:22:51、16:27:32（上海时间），两项已合入 `master`。** 算子侧将 hostDPU write-reduce 接入 AllReduce、Reduce、ReduceScatter，按全域能力、传输原语、dtype、规约操作及确定性约束选路。后续共用协商子通信域，在 950 hostDPU 路径上对能力做全域 AND 聚合并按 communicator 缓存结果。老 HCOMM 缺少查询或建子域弱符号时可退回兼容路径。执行器 PR 标注面向 950/960，协商 PR 的显式门控写为 950，实际生效范围需按部署版本与拓扑确认。提交者报告执行侧 88 UT、18 模型 ST 和实机验证，协商侧覆盖了仿真桩缺少符号的崩溃复现，均未给吞吐。**建议先核对升级后的 `HCCL_ALGO`**。AllReduce/Reduce 的原组合变为 MeshMesh，原先显式 `sequence{mesh,nhr}` 的语义随新选路而变。两项尚未映射到已验证的发行包。 [执行器基线](https://gitcode.com/cann/hccl/commit/ed42d0f29fccf24728055b9e9080587c04ab6e54) · [协商基线](https://gitcode.com/cann/hccl/commit/7572a66803d3bf8fc8228c376207c07b135226f2)

### MoRI 增加 BF16 GEMM + collective，并为 RDMA 重传故障提供限流

**AMD ROCm/MoRI · 11 月 9 日 16:53:18（上海时间），#872 已合入。** Kimi K3 一类 checkpoint 中保留 BF16 的 attention 输出投影和共享专家可以继续用原权重，走新 BF16 GEMM 与 SDMA collective 融合路径。还可只做 reduce-scatter，把 token 行分片留给后继算子。作者在 8×MI355X、Kimi K3 TP8 的 `M=16384,N=7168,K/rank=1536` 上，报告 GEMM+all-reduce 从 hipBLASLt+RCCL 的 1,421 μs 降至 BF16 线传的 1,150 μs，另测 reduce-scatter 为 636 μs，基线 885 μs。FP8 gather 会改变误差口径。**考虑用真实 shape 频率复验**，这些是算子级数据，不能外推端到端吞吐。 [合入代码、完整 shape 表和误差](https://github.com/ROCm/mori/commit/af56d9b50b1e286deba7be9f6b416c5750436d87)

**同日 16:43:54、21:06:03，#936、#928 已合入。** 前者允许用 `MORI_IO_QP_MAX_RD_ATOMIC` 限制 RC QP 的在途 READ 数。作者在 AINIC 固件 1.117.5-a-77 复现过并行 READ 重传导致 QP 报错且无失败 completion 或部分数据丢失，在途限制为 1 能减轻问题，重背压时仍可能丢数据。后者把 relaxed-ordering 开关实际传到 RDMA MR 注册。**建议对照固件和背压注入**，并单独核对限流带来的带宽损失。 [READ 并发上限](https://github.com/ROCm/mori/commit/131af0a80ab84095d7d6a618a5662b2cf9af75c0) · [Relaxed ordering](https://github.com/ROCm/mori/commit/eea28ff6b4d6d4ba9e2711e505df8f2b931001ba)

### Intel Gaudi 修复 MoE 路由静默偏差和量化草稿专家视图

**vLLM-Gaudi · 11 月 9 日 17:36:29、11 月 10 日 02:55:51（上海时间），已合入默认分支。** #1922修复非 grouped 路由原先一律 softmax 并归一化的问题。非普通路径按层设置选择 softmax/sigmoid，修正 bias 只参与选专家，权重从未加 bias 的分数读取，再按配置归一化与缩放。原行为可让使用 sigmoid 加 bias 的模型无报错地选错专家。#1939 则在 WNA16 草稿模型搬到 HPU 后重建普通属性里的 per-expert 权重、scale 和零点视图，使它们绑定新的设备参数。**建议做路由与草稿端逐层回归**，尤其是低精度、多专家和启用投机解码的组合。两者有提交内测试，尚未核实进入哪个发布包。 [路由基线](https://github.com/vllm-project/vllm-gaudi/commit/807fcb00d07c3a04e63e433588bcb9646ebad567) · [草稿模型基线](https://github.com/vllm-project/vllm-gaudi/commit/4bec164dc6f9596ad12945fbe0616accdf727671)

### TorchTPU 推理分支扩展词表分片，修正 P/D 布局失败和 logits 通信

**vLLM-TorchTPU · 11 月 10 日 08:13:30（上海时间），#2863 已合入。** 实验开关 `TPU_DP_VOCAB_PARALLEL` 将常规 embedding 和 lm_head 从每个 DP engine 的 TP 组分片，扩到 DP×TP 的整个 EP 组，减少多份词表驻留。输入和 logits 路径需增加跨 DP 交换，建议在词表较大的模型中把峰值显存与小批量 decode 通信开销一起测。Raiden 的 #2912 则把明确的 layout fingerprint 不匹配识别为不可重试失败，避免等待 KV load 重试耗尽。#2993 使多模态 encoder DP gather 在所有 TP rank 的同一阶段触发，降低 collective 顺序错位风险。三个变更均为合入状态，缺少统一性能测试。 [词表分片基线](https://github.com/vllm-project/vllm-torchtpu/commit/e30cc488bda5f21886d136e3f2dfdb4619704b63) · [Raiden 布局检测](https://github.com/vllm-project/vllm-torchtpu/commit/2a223bab6966b80de78c9f7b0379ce508da13420) · [多模态 gather](https://github.com/vllm-project/vllm-torchtpu/commit/97701a65215efa6ac762a4b9eae7dc8251343dc6)

**SGLang-TorchTPU · 11 月 10 日 03:41:05 至 06:52:56，已合入。** #2041 将 zero-pad 后 all-reduce 的 gather 替代限定在三处 logits 路径，恢复逐层 DP attention 和 MoE 等路径的原生 collective。作者报告此前全局替换使 GLM-5.3-Flash DP8+EP8 退化 8–14%，未提供可跨平台比较的完整负载。#2058 给 GLM-5.3-Flash 接入 Raiden P/D 配方，区分目标模型、草稿模型和 KDA 状态的传输布局，避免无 RoPE 池多传空缓冲。#2002 让 `/health_generate` 以真实流量或空闲时的单 token 探针完成 forward 为依据，避免控制消息或空健康输出掩盖卡住的 forward。**建议复测** DP8+EP8 的 TTFT、PD cache 等价性与卡死时的负载均衡摘除。 [logits gather 基线](https://github.com/sgl-project/sglang-torchtpu/commit/9225bc31f3395de2156ce4d6ec620511cb74c49d) · [Raiden P/D 基线](https://github.com/sgl-project/sglang-torchtpu/commit/9cb771a001de1b405aaf49d06352e8080c1d950e) · [健康探针基线](https://github.com/sgl-project/sglang-torchtpu/commit/e862eb8d3d384e0211e78aeb65c684af37c2cad1)

### TorchNPU 扩展分配和编译选项，并补齐融合算子的舍入语义

**Ascend/pytorch · 11 月 9 日，以下变更已进入 `master`。** 实验性 `mem_malloc_policy` 可为非 VMM 缓存分配指定大页或带宽策略，显式策略优先于 `page_size:1g`。VMM 仍使用独立的 `page_size`，1G 大页受大池、CANN 与驱动能力门控，直接无缓存分配将策略交给底层。Inductor 新增 `triton_experimental` 的可选 C++ wrapper，覆盖设备 override、AOTI fallback 与 dispatch recipe，合入描述未给完整设备验收。另有正确性修复在 FP32 累加的 GEMM 与低精度 epilogue 融合时恢复中间舍入，提交者报告 NPU 上 12 个 dtype×mm/addmm 检查通过。**建议按场景分别验证**，把长序列分配碎片、wrapper 编译和融合数值逐项拆开。 [分配器基线](https://github.com/Ascend/pytorch/commit/6efa25e8bb06a0eb9c356ea1038c97d479c74abf) · [C++ wrapper 基线](https://github.com/Ascend/pytorch/commit/e1628b9b1e0f1476d9e8a2768f2c48b18b73b06e) · [舍入语义基线](https://github.com/Ascend/pytorch/commit/b6da676b9df499662aa2b4d1c1651f8b34c77870)

### Primus 修正无 packing 的 SFT 标签掩码错位

**AMD-AGI/Primus · 11 月 9 日 16:23:37（上海时间），#1404 已合入。** 原先 labels 移到 next-token 位置后，默认的无 packing 路径仍按输入位置掩码，使首个回答 token 漏监督，回答后的独立 pad token 反而可能被监督。修复同步位移掩码，并检查 EOS 与截断边界。提交者在 Alpaca-cleaned、序列 1,024、global batch 16、1–8 卡的排查中，发现独立 pad 的 Phi-4 和 Granite 4.1 首步损失明显偏离只对回答计分的 Transformers 对照。17 个 CPU 用例覆盖修复，收敛曲线并未直接暴露错位。**建议回查受影响的无 packing 数据配方**，用 `labels[loss_mask == 1]` 校验实际监督序列。 [合入基线与原复现](https://github.com/AMD-AGI/Primus/commit/bc9e8b6b1a16dfb44ae4019bde5660491eb75120)

### TileGym 缩短 cuTile RMSNorm 反向的权重梯度归约

**NVIDIA TileGym · 11 月 9 日 13:25:08（上海时间），#216 已合入。** persistent block 将寄存器内的 dW 部分和用 relaxed atomics 写入零初始化 FP32 累加器，省去部分和矩阵与后续列归约。行 tile 设为 16 KB，面向 2,048–4,096 宽行的驻留优化。提交者在 GB200 上对 53 个 shape 报告相对原 `main` 几何均值 1.39 倍。未给出整模型训练收益或跨架构对照。**考虑用本地 hidden size 和 batch 分布复验**，关注梯度容差和跨次运行的数值一致性。 [合入基线与 microbenchmark 口径](https://github.com/NVIDIA/TileGym/commit/afe6ddd6f56bfebf107ac0fb457d5db3e2e28bd5)

### CANN Runtime 收紧队列调度和 AICPU 错误路径

**CANN Runtime · 11 月 9 日 16:55:17 至 22:24:24（上海时间），三项已合入 `master`。** 队列调度在 Ezcom RPC 收到 `-EAGAIN` 时清理残留响应，先验证 HCOM handle 请求头长度再读字段，防止错误路径泄漏或越界读取。动态组构建请求内存不足时返回现有 FSM 失败状态，避免发送半成品或留下待决对象。AICPU MSGQ 用单次原子 fetch-add 领取序号，并在卸载队列调度库前注销回调，针对并发响应错配和关停时调用已卸载代码的风险。项目方报告 host UT 与基线对照，真实设备故障注入未披露。**建议在升级评估时重点覆盖**重试、内存耗尽和并发关停。 [RPC/请求校验基线](https://gitcode.com/cann/runtime/commit/04308dd3173360e5354ad6aa08687885fe8785a0) · [动态请求基线](https://gitcode.com/cann/runtime/commit/fb841638d805a4a01b4521b874b9f5806895796f) · [AICPU 基线](https://gitcode.com/cann/runtime/commit/8387a4496a6ff5ed2a3ef638d4cb62be089146f5)

### 昇腾公开两起 HCCL 故障复盘，分别追到存储和 MPI 地址冲突

**昇腾社区 · 11 月 9 日公开，案例发生时间与披露日期分开。** Atlas 900 A2 PoD 的 4K 卡作业在 HDK 24.1.0.3、CANN 8.0.RC3、PyTorch 2.1.0.post8 下出现 `EVENT_WAIT TIMEOUT`，256 卡和 1K 卡可启动。项目方对齐各卡 HCCL 下发日志，看到负责读数据的 device0 晚于 1,800 s 超时阈值提交 Broadcast，高带宽存储替换后恢复。文章没列出模型、数据配方或替换前后实际吞吐。另一案例在 Atlas 200I A2 Box、CANN 8.5.0、单机 16 卡跨环路径上复现 Open MPI 5.0.8 的共享内存占用了 HCCL 的固定 SVM 虚拟地址段，改用 4.1.5 后复现消失。**建议阅读故障定位链**，通信超时先对齐各 rank 的提交时间和 host 侧阻塞，初始化故障再查进程地址映射及版本配套。 [4K 卡案例原文](https://www.hiascend.com/ascendgateway/ascendservice/tech/details?techColumnCode=20261109-2) · [MPI 版本 A/B 案例原文](https://www.hiascend.com/ascendgateway/ascendservice/tech/details?techColumnCode=20261109-102)

### 补录｜MoRI v1.2.5 把 EPv2、通信融合与网络恢复打包进版本

**AMD ROCm/MoRI · 11 月 6 日 16:59:15（上海时间），`v1.2.5`。** 相比 v1.2.4，EPv2 加入跨机 AsyncLL、gfx1250 小 token LL128 和 EP16/EP32 调优。CCO 扩展 GEMM 与 all-gather、all-to-all、reduce-scatter 的融合，SDMA 支持二维传输和 FlyDSL/Triton 设备接口。MoRI-IO 可替换故障 RDMA QP，并新增 Slingshot/CXI 的 OFI/libfabric 后端。官方发行说明给出 manylinux_2_28、内置 UMBP gRPC 与 gfx942/gfx950/gfx1250 的包边界。GitHub Release 资产列表为空，wheel 的具体获取路径需按环境验证。**建议对照 EP token 分布和网络故障恢复**，发行说明未给统一的端到端对比。上文 #872、#928、#936 是后续默认分支增量。 [官方发布与差分链接](https://github.com/ROCm/mori/releases/tag/v1.2.5) · [EPv2 AsyncLL](https://github.com/ROCm/mori/pull/768) · [QP 替换](https://github.com/ROCm/mori/pull/831)

### 补录｜verl-SpeCo 披露 Megatron actor 和 EAGLE3 共训的并行适配

**昇腾社区与 verl-SpeCo · 原文 11 月 6 日，未给时分。** 在 old-log-prob forward 用 hook 从指定 actor 层与 final norm 抓 EAGLE3 监督特征，先筛选 token，再按 SP 分片恢复。PP 路径把各 stage 的 capture 放到共享目录，用原子 rename 和轮询交换，在末 stage 聚合后交给 Ray，案例默认选择可写的 `/dev/shm`。该目录与 PP 放置有共同可见性要求。与 FSDP2 8B/Ulysses SP4 相比，Megatron 8B TP8PP1 的前 10 步接收长度接近，平均单步 675 s 对 611 s，权重更新 5.3 s 对 8.3 s。4B TP4PP2 关闭 SP，报告 53 步可训练，old-log-prob 因动态微批流水开销显著变慢。两组设置改变了并行方式，报告未逐项给设备、精度和响应长度。**建议深入读并行边界与负面结果**，尤其是跨节点 PP 的目录、超时和对齐语义。 [昇腾原文和第 7 节](https://www.hiascend.com/ascendgateway/ascendservice/tech/details?techColumnCode=20261106-7) · [verl-SpeCo 固定代码](https://github.com/verl-project/verl-SpeCo/blob/046ac17d356bf0557e6caa6d9ac94ed0b0f536f9/verl_speco/integration/oldlogprob_runtime.py) · [固定启动脚本](https://github.com/verl-project/verl-SpeCo/blob/046ac17d356bf0557e6caa6d9ac94ed0b0f536f9/examples/run_qwen3-4b_actor_megatron_drafter_eagle3_vllm_npu.sh)

<a id="coverage"></a>
## 覆盖与续查

- **模型、厂商报告与工件。** 121 个厂商、研究、评估和发现入口，91 个完整返回 HTTP 200，另有两个截断的 200，其余 28 个为 403、404、429 或连接失败。OpenAI 新闻、研究索引和 Control Room 直连受限，已从官方 RSS 和网页正文核实本次条目。Hugging Face 全局创建目录拉至重叠窗起点之前，模型 13 页 13,000 条、数据集 8 页 8,000 条，重叠窗候选分别为 12,853 和 7,316，正文仅抽筛。另对 140 个组织查模型／数据集更新时间，两类共 280 个目录请求均返回 200，取到 201 个重叠窗候选。
- **续查。** 上轮未完成的三项已有固定证据。TwIL-8B 为首次公开权重，见补录。MiniCPM-OCR-1.0 的两次变更限于中英文模型卡和用法指引。PLaMo 3 Translate 31B 在 11 月 8 日短暂加入又撤回 `target_lang=auto`，11 月 9 日改双语许可，增加仅转换输入内容的模型定义和公开转换输出的单独规则，使用生成物训练模型仍按派生模型处理。[MiniCPM 固定差分](https://huggingface.co/openbmb/MiniCPM-OCR-1.0/commit/be4f5cc52807b5929cb1c8c7e1a6472f76d98b1d) · [PLaMo 模板回退](https://huggingface.co/pfnet/plamo-3-translate-31b/commit/d6743d65c2cc61a07ff0fb2c3ff3cf7fd8ca36eb) · [PLaMo 许可修订](https://huggingface.co/pfnet/plamo-3-translate-31b/commit/959bd168231dfead9135b7733a79f5c47b234d1a)。
- **版本边界。** Nous Arete、BioBasicsBench、Seed 跨层转码器此前已归档。LiquidAI 两次模型卡更新链接的是 10 月已上传的 OpenVINO INT4 工件，Kakao 决策模型的 11 月 9 日更新只改卡片描述和检索标签。Tinker 已提供 Inkling 2 Small，当前官方 HF 清单尚未看到该版可固定权重。NeoHorse-1.5-Flash Preview 的官方日志只确认内测申请，技术细节与公测时间待核实。
- **arXiv。** 九分类 11 月 9 日公告按 ID 合并得到 890 项，其中 498 项至少一个分类标为新稿或交叉列示，其余只见修订。Atom 主窗查询 HTTP 200、0 条，重叠窗显示 365 条且第一页最近提交为 11 月 6 日 18:59:57 UTC。两者是提交索引，公告日期另计。独立 lastUpdatedDate 重试为 429 或连接失败，OAI 单日桶、跨日桶、ID 和单篇探针均受 429 或连接错误限制。新版修订与语义队列继续保留。 [九分类之一 cs.LG](https://arxiv.org/list/cs.LG/new?show=2000) · [OAI-PMH 入口](https://oaipmh.arxiv.org/oai?verb=Identify)
- **会议与索引。** AACL 11 月 9 日核到 Multi-LLL 论文集和单篇排程、两场教程与教材。其他工作坊和主会未审海报仍在队列。OpenReview 的 TMLR 与 NeurIPS API 仍为 403，NeurIPS 会议索引可读，TMLR 只有月份，PMLR、MLSys、SOSP 与 OSDI 仅查索引或部分列表。HF 11 月 9 日日榜恢复为 31 项，11 月 10 日请求为 400，日榜只用于发现。 [AACL 日程](https://2026.aaclnet.org/program/) · [NeurIPS 会议索引](https://neurips.cc/virtual/2026/papers.html) · [TMLR 期刊索引](https://jmlr.org/tmlr/papers/) · [PMLR](https://proceedings.mlr.press/) · [HF 11 月 9 日日榜](https://huggingface.co/api/daily_papers?date=2026-11-09)
- **开源工程。** 检查 65 个仓库入口，60 个默认分支、56 个主窗 merged PR 分页闭合，重点 15 个仓库完成回溯窗 PR 分页。60 个官方 release Atom、27/30 个 PyPI 工件库存可读。读取 198 个已合入 PR 的正文，入选事件中 85 项另有补丁。Megatron-LM 主窗仅检出文档变更，slime、OpenRLHF 主窗默认分支和已合并 PR 均无新增。维护分支、候选语义抽查、有限 feed 和受阻端点保留续查。
- **硬件与软件协同。** 检查 NVIDIA、AMD、Intel Gaudi、昇腾、Google TPU、AWS Neuron、Apple MLX、摩尔线程、寒武纪、沐曦与 Qualcomm 的选定官方公告、包目录、发行和默认分支记录。硬件台账记录 211 次入口及一手证据读取，21 项默认分支入口的目录扫描覆盖主窗，语义核验仍为抽筛。NVIDIA 技术 feed 已完整读取。与 11 月 7 日保存的正文对比，11 月 9 日 NVIDIA Reshard 技术博客更新只改致谢文本，沿用原日报。
- **硬件续查。** 昇腾公告和技术文章第一页经 Referer 恢复，部分入口仍返回空载荷。按实际公开日期排除提前返回的未来条目。TorchTPU 与 Ascend/pytorch 的最近提交页未盖全主窗，两个 TorchTPU 推理仓库的回溯前段仍待分页。Azure HPC 403、沐曦新闻路径和寒武纪提交入口 404，AMD 指定会议页连接超时。CANN 9.2.1 的组件映射、libtpu/Neuron 具体行为、A5X 带宽口径及 TPU 8tx/8i Max 可用性留待续查。

- **续扫边界。** 本轮发现截止为 2026-11-10 08:58:40。下轮从该位置续扫并回溯 72 小时，同时继续处理分源待查记录。全局完整语义水位仍未闭合。本轮未做独立复训、设备压测或付费 API 实测。
