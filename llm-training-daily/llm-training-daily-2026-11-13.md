# LLM 技术动态日报 · 2026-11-13

> 时区 Asia/Shanghai（UTC+8）。增量扫描窗口 **2026-11-12 09:05:00 → 2026-11-13 09:03:00**，向前回溯 72 小时至 **2026-11-09 09:05:00**。实际生成时间 **2026-11-13T09:27:50+08:00**。日期未披露到时分时保留原日期。首次发布、工件或代码更新、会议报告与本轮发现分别标注。
>
> 生成前核对本地 37 篇历史日报及索引，与 GitHub `main` 的 `ebda0ad39dc79d5e5476a78cf487092353151e19` 逐字一致。前次待上传队列为空。
>
> 本轮核验公开的一手报告、工件、发布记录及部分代码补丁。性能与能力指标为原作者或项目方的口径。

## 本次重要变化

- **可复现工件。** Inkling 2 Small兑现原生NVFP4/MXFP8开放权重，LongCat-Next补齐四档dNaViT；LiquidCodec把独立音频tokenizer的训练、重建与可学习性条件一并披露。Hydrogen正式API的共享token预算与参数迁移需单独验收。
- **训练数值与结论边界。** FP4长尾事实研究提示平均loss可能掩盖学习效率差距，但实验规模仍小且为仿真。BF16随机舍入、过程奖励的匹配计算负面结果及Plumbline训练正确性检查，分别提供可复现的诊断方向。
- **开源训推机制。** Megatron、TorchTitan与SGLang的参数身份、共享梯度、adapter同步和PP生命周期修复值得优先核对。版本预览、同日撤回与后端局部支持均保留状态，不把合入当成正式发布。
- **KV与硬件路径。** 稀疏索引键需随缓存一起换出；SWA槽位复用、可续租P/D缓存、变长注意力越界和低精度scale布局是本轮工程重点。Rubin报告的总吞吐包含缓存输入，部分优化来自单节点模拟，不能照搬为生成token加速。
- **评估与可靠性。** 自动红队的重复成功门槛改善训练数据筛选，仍不能消除评分偏差；AlphaProof Code的可检查证明保留明确可信边界。MACaQ的校准资源与推理内存需区分，其昇腾算子PR仍未合入。

[模型与技术报告](#models) · [研究论文](#papers) · [开源工程](#engineering) · [硬件与软件协同](#hardware) · [覆盖与续查](#coverage)

<a id="models"></a>
## 模型与技术报告

### Inkling 2 Small 兑现开放权重，原生 NVFP4 训练可开始本地复现

**Thinking Machines Lab，11 月 12 日 19:53 UTC 上传权重，Apache-2.0。** 本次增量是把 11 月 5 日预告兑现为可下载的 NVFP4 与 MXFP8 工件。固定配置为 48 层、384 专家选 8，40 层 1,024 token 局部注意力与 8 层全局注意力，594B 总参、17B 激活，支持图文输入和 1M 上下文。卡片明确预训练、中训练、后训练均使用 NVFP4。不能因 `torch_dtype=bfloat16` 就把这个仓库当 BF16 检查点。

官方给 NVFP4 的部署门槛为聚合显存至少 500GB（例：2×B300 或 4×H200），MXFP8 为至少 800GB（例：4×B300 或 8×H200）。这是部署建议，不保证任意并发和上下文下足够。训练数据配比、总 token 与计算预算仍未完整公开。**建议训练与部署团队深入阅读并考虑复现**：现在可在固定权重上检查低精度训练与实际运行时的质量、缓存和显存边界。11 月 5 日已披露的分数不作为本轮新增成绩。 [固定 NVFP4 卡片](https://huggingface.co/thinkingmachines/Inkling-2-Small/blob/e242a97e271095c11bc33ddb3850d37e22917343/README.md) · [固定配置](https://huggingface.co/thinkingmachines/Inkling-2-Small/blob/e242a97e271095c11bc33ddb3850d37e22917343/config.json) · [固定 MXFP8 工件](https://huggingface.co/thinkingmachines/Inkling-2-Small-MXFP8/tree/3febc62259ee92ea36e2fb6e8303371c4eb3922c)

### Gemini 4 Hydrogen 正式提供 API，百万上下文需要按共享预算迁移

**Google DeepMind，11 月 12 日，GA 托管模型。** Hydrogen 从 Carbon 适配，支持图像、音频、视频与文本输入，新增 `xhigh`，默认 `high`。输入、思考与输出共享 1,048,576 token。同步每跳上限 65,536，超过此长度需 background 或客户端 continuation。文档列每百万输入/输出 $1.50/$7.50，整个上下文窗口同价。移植旧调用时，`temperature/top_p/top_k` 自定义值会被忽略，`max_output_tokens` 还包括思考。客户端须完整保留 thought signatures。

模型卡的 Terminal-Bench 4.1 为 57.0%，低于表内 Sonnet 5.5 的 63.9%。方法附录说明 Hydrogen 是 high、mini-swe-agent vision、每任务 5 次取平均 reward，竞争项取公开榜，均作弊校正。这些数据不构成相同 harness 的独立对照。卡片把架构、数据和系统说明指回 Carbon，未给 Hydrogen 的参数量和独立训练配方，完整安全报告尚待发布。**建议有 agent 工作负载的团队考虑小规模试用**，按思考档位、完成率、总 token、延迟和迁移兼容性验收。 [API 迁移契约](https://aistudio.google.com/docs/whats-new-gemini-4-hydrogen) · [模型卡](https://deepmind.google/models/model-cards/gemini-4-hydrogen/) · [评测方法第 2 页](https://deepmind.google/models/evals-methodology/gemini-4-hydrogen/)

### LiquidCodec 开放独立流式音频 tokenizer，分组量化消除码本间残差依赖

**Liquid AI，11 月 12 日技术报告与独立 BF16 工件，LFM Open License v1.0。** 这次补齐 LFM3-Audio/Omni 背后的约 74M 参数 codec：24kHz 单声道、12.5Hz 帧率、每帧 12 码本、约 1.65kbps。对称且因果的 LFM3 编解码器直接预测频谱系数，GFSQ 各组独立量化，允许下游模型并行预测码本。这不等于全部 TTS 路径没有自回归步骤。

报告披露先用 107 语言共 264k 小时、不含 GAN loss 预训练，再在 55k 小时原生至少 24kHz 的较干净语音上加入对抗评判器，另做 decoder-only QAT。M4 Max 单 CPU 线程 FP32 中编码/解码 RTF 为 0.0028/0.0115。同表 Mimi（截为 12 码本）为 0.0361/0.0329。八套重建评测的 ViSQOL/mel 距离领先，CER、说话者相似度与 TTS 错误率并非各项都最好。仅评语音，音乐和环境音未评。**建议语音多模态训练团队深入阅读并考虑复现**：它同时比较重建质量与固定数据、骨干和计算量下的 TTS 可学习性。 [训练与架构报告](https://www.liquid.ai/blog/liquidcodec) · [固定卡片及完整条件](https://huggingface.co/LiquidAI/Liquid-Codec/blob/b5abfac46ef92275f191a52c41013315f237301d/README.md)

### LongCat-Next 补齐四档 dNaViT 权重，视觉离散表示从占位变成可运行工件

**美团 LongCat，11 月 12 日 09:13 UTC，MIT 仓库新提交。** 11 月 11 日日报记录的 dNaViT 仍是说明与占位目录。本次新增 G16、G16R1、G16R52、G64 四份 tokenizer 权重及独立配置、图像处理器和推理实现。G16 配置为 16 组、每组 16,384 项，G64 为 64 组、每组 4,096 项，二者都使用 27 层视觉编码器和 24 层像素解码器。

固定实现 `dNaViTTokenizer.encode` 输出离散索引与合并后的网格，`decode` 根据索引和网格恢复像素。`_round_input` 将图像对齐到 patch×merge 的 32 像素倍数。该代码注明只实现 inference。**建议统一视觉理解/生成与 tokenizer 团队考虑复现**：已有独立工件可做重建和语义保持实验，但此次提交没有新增同配置吞吐、压缩率或下游质量对照，不能把旧论文成绩归给这四档新包。 [固定增量](https://huggingface.co/meituan-longcat/LongCat-Next/commit/926e550cb64ac1a4469117dafc7f1a14303cd23a) · [G16 配置](https://huggingface.co/meituan-longcat/LongCat-Next/blob/926e550cb64ac1a4469117dafc7f1a14303cd23a/dNaViT/dNaViT-G16/config.json) · [固定推理实现](https://huggingface.co/meituan-longcat/LongCat-Next/blob/926e550cb64ac1a4469117dafc7f1a14303cd23a/dNaViT/dNaViT-G16/modeling_dnavit.py)

### MetaEncoder v1 把多模态检索复用于封闭决策，置信度需要单独校准

**Meta 的 facebook 官方组织，11 月 12 日新建并上传工件。** 公开约 28.43B BF16 参数与 6,656 维归一化嵌入，提供图文、视频帧和文档图检索，也以 `decide` 输出是非、选项与有序评分。固定 `MetaEncoder._encode_batch` 池化最后一个 token 并投影。`decide_batch` 将每道题与其候选分别编码，再做余弦相似度比较。它并非把所有题共享一次 forward 的生成式 API。

`answer` 以默认温度 0.02 对相似度做 softmax，`confidence` 是 1−归一化熵。`calibrate/fit_temperature` 可按带标签样本最小化 NLL。不能把未经校准的这一分布直接当真实准确率。厂商图片表中 JevBench-hard 0.685 低于旧版 0.739，MMEB V3 文档 0.827 低于 WeMM-Embedding 9B 的 0.833。卡片给单张 80GB GPU、训练序列最长 16,384、视频最多均匀取 8 帧。本次未找到训练报告、许可字段或 LICENSE 文件。**建议继续观察，研究端可先评估复现路径**，商业采用前应先补齐许可与训练来源。 [固定卡片](https://huggingface.co/facebook/meta-encoder-v1/blob/ed04ebd4defb8e837b5eaea164143177272696ac/README.md) · [固定实现](https://huggingface.co/facebook/meta-encoder-v1/blob/ed04ebd4defb8e837b5eaea164143177272696ac/modeling_metaencoder.py) · [厂商原始评测表](https://huggingface.co/facebook/meta-encoder-v1/blob/ed04ebd4defb8e837b5eaea164143177272696ac/benchmarks.png) · [固定完整工件](https://huggingface.co/facebook/meta-encoder-v1/tree/ed04ebd4defb8e837b5eaea164143177272696ac)

### vLLM Semantic Router 发布 2B 多模态决策工件

**vLLM Semantic Router，11 月 12 日，Apache-2.0。** d3.1-nano 以 Qwen3.5-2B 为底座，共 2.21B 参数（视觉编码器 0.33B），在一次调用中返回多题的选项、是非与评分概率，支持文本/JSON、图片和视频。发布包附 `decision_index` 接入。引擎默认对 choice 题用两种选项顺序评分，可关闭 permutation averaging。

模型卡限定图片最多约 1.6MP，视频取 2FPS、最多 32 帧、每帧约 0.2MP，示例固定 Transformers 5.17.0。此次尚未发布可核的 Decision Index 成绩、训练流程或硬件延迟，不能据体积推断优于托管决策 API。**建议路由团队考虑复现**，先测选项顺序敏感性、校准、共享状态重复计算与实际负载成本。 [固定模型卡](https://huggingface.co/vllm-sr/d3.1-nano/blob/f3e3b4006ea8ee80cb95471ac27094908d4f630c/README.md) · [固定发布目录](https://huggingface.co/vllm-sr/d3.1-nano/tree/f3e3b4006ea8ee80cb95471ac27094908d4f630c)

### 补录｜Gemini Decisions API 用单次前向评分代替封闭任务的逐 token 生成

**Google，原始发布日期 11 月 11 日。72 小时回溯补录。** `gemini-decision-1` 面向路由、工具选择和上下文判断：同一份至多 256K token 的图文音视频/PDF 输入，可提最多 1,000 道独立题，混合二元、2–255 选项与有序 rubric。返回概率与置信度而不生成自然语言，按输入计费，题目指令和选项仍计入输入。

官方在至多 16K 上下文、1–100 题的条件下报告模型计算中位数 50–100ms、端到端 150–250ms。没有同一证据支持把该延迟外推到 256K/1,000 题。架构与训练配方、权重未公开。**建议路由和分类服务团队考虑小规模复现**，用自己的校准集、类别不均衡和尾延迟验收。各题独立并不自动保证跨题逻辑一致。 [官方契约](https://aistudio.google.com/docs/decisions-api) · [公告及延迟条件](https://blog.google/innovation-and-ai/technology/developers-tools/gemini-decisions-api/)

### 补录｜AlphaProof Code 让不可信证明 agent 对照冻结规范交付可检查证明

**Google DeepMind，原始公告 11 月 10 日。代码基线 b0f729ac（11 月 11 日），回溯补录。** AlphaProof Code 将人工审核的规范与 prover 工作区隔离，gatekeeper 只接受目标文件/定理名，并让 Lean 在干净环境重放。规范修订必须经独立审查，不能让 prover 自行削弱安全目标。论文 §7 披露五人四周、约 6.5M 行 proof code，并以重新引入两个历史缺陷来检查系统能给出不安全见证。

验证对象是 `libsignal` v0.105.4 的 AArch64 `libsignal-cli` 构建及一对一 PQXDH/Triple Ratchet。§5.2–5.3 保留密码学假设、hooked routine 契约、AArch64/Linux 环境语义等可信边界。握手 HKDF 的随机预言机步骤是理想化。固定提交进一步把 trusted hooks 从 37 减到 27，仍不能据此宣称 Signal 应用整体无漏洞。公开复核需固定 Lean 4.34.0、Rust 1.98.1 和构建工具，官方估计默认检查数小时、峰值内存约 15–34GiB。**建议 agent 评估和可靠性团队深入阅读**：价值在冻结目标、规范修订与独立检查的闭环，本轮未独立重放全部证明。 [官方公告](https://deepmind.google/blog/introducing-alphaproof-code/) · [论文 §2、§5、§7](https://deepmind.google/documents/36/Formal_Verification_of_Signal_4.pdf) · [固定复核入口](https://github.com/google-deepmind/code-verification/blob/b0f729accbc21a0413e7ca4e1da43fa45b5f2e1b/README.md) · [缩小可信边界的提交](https://github.com/google-deepmind/code-verification/commit/b0f729accbc21a0413e7ca4e1da43fa45b5f2e1b)

### Sarvam 30B 修复 HF/vLLM 的 MoE 舍入顺序，并恢复 Transformers 5 加载边界

**Sarvam AI，11 月 12 日 12:55 UTC，模型仓库已提交。** `SarvamMoEGate` 默认 `router_numerics=vllm`，保留 router FP32 权重，同时匹配 vLLM 的激活精度舍入与稳定 tie-break。专家 down projection 先保留 FP32 accumulator，乘路由权重后才舍入，专家求和再乘 routed scale，共享专家保持未缩放。补丁同时恢复 meta-device 加载后的 RoPE buffers，精确指定 FP32 gate 参数，避免 Transformers 5 的子串匹配把 `gate_proj` 也错误提升精度。

新增回归覆盖 train/eval、梯度、存取后 RoPE 与稳定路由。CUDA 参考测试明确固定 vLLM **0.15.0**，不是对当前所有后端的逐 bit 保证。本轮读了补丁及测试，未在 GPU 上执行。**建议曾用 HF 训练、vLLM 采样的 Sarvam 实验优先检查**：推理分支、dtype 和缩放次序能改变离散专家选择，模型分数变化应先排除运行时差异。 [固定补丁](https://huggingface.co/sarvamai/sarvam-30b/commit/2784c524a2f2a08f597adc5530d13ec66079e347)

### SmolVLM2 修复字符串消息被 chat template 静默丢弃

**Hugging Face，11 月 12 日 02:53 UTC，模板修复。** `chat_template.json` 与 `tokenizer_config.json` 增加 `content is string` 分支，原先只会遍历带 `type=text/image` 的列表。对字符串输入，旧模板保留角色却丢掉正文，服务仍可能正常生成。

本轮用 Jinja 做最小复核：相同字符串问句在旧模板中变成空 `User:`，新模板保留问句。原有列表文本输入前后相同。未下载模型权重或重新跑能力基准。**建议使用 OpenAI 风格字符串 `content` 的现有部署优先采用模板修复**，并核对打包后生效的是哪一份模板。 [固定差分](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct/commit/be92141c8d0ceb98596a923c308ac2bbb4a7c466)

### Red Hat 发布 Bolt E4B NVFP4，未量化 embedding 决定最终体积

**RedHatAI，11 月 12 日，OpenMDW-1.1 工件。** 对 NVIDIA Bolt E4B 的语言层 linear 做 NVFP4 权重/激活 GPTQ，保持 embedding、逐层输入 embedding、输出头与音视频塔 BF16，检查点从 16.0GB 降到 11.6GB。4bit 层不意味着整个模型缩到四分之一。

固定卡片披露单 B200、vLLM、thinking on、并发 64、最多生成 32,768 token 的 Inspect 评测：GSM8K 93.2%、GPQA Diamond 47.0%、MMLU-Pro 67.7%。原基座未给这组同协议分数，且 GSM8K 在其后训练数据列表中，不能据此认定量化无损或证明该题集泛化。**建议部署团队考虑复现**，优先对照相同 prompt/采样/解析与 BF16 权重，并单测多模态路径。 [固定卡片与配方](https://huggingface.co/RedHatAI/NVIDIA-Nemotron-3.5-Bolt-E4B-NVFP4/blob/b24831981e7809842fbc4f3f7a69a70da50f2403/README.md)

### 已有工件的许可与模板增量

- **Microsoft FrogNano，11 月 12 日。** 仓库新增实际 Apache-2.0 LICENSE，模型卡许可从 MIT 改为 Apache-2.0。模型原始发布日期仍为 9 月 22 日，权重未重发。**建议已有使用者核对锁定版本中的许可与来源记录**。 [固定卡片差分](https://huggingface.co/microsoft/FrogNano-4B-2609/commit/1844d06083df631f634e417938b3b8feda5f9f48) · [实际许可](https://huggingface.co/microsoft/FrogNano-4B-2609/blob/1844d06083df631f634e417938b3b8feda5f9f48/LICENSE)
- **KRAFTON Raon-1 mini preview，11 月 12 日。** `tokenizer_config.json` 内嵌完整文本 ChatML 模板，包含工具调用与历史 thinking 处理。该字段原先不存在。**建议依赖 tokenizer 配置分发模板的部署考虑采用**，此次没有新权重或质量成绩。 [固定差分](https://huggingface.co/KRAFTON/Raon-1-mini-preview/commit/1b4f494d71dff50318288309f64c72aa8c11340e)

<a id="papers"></a>
## 研究论文｜预训练与训练数值

本轮 arXiv 九分类入口仍返回 **11 月 11 日公告**，未取得可确认的 11 月 13 日常规批次。同时，OAI分页恢复出11-13元数据记录，包括KernelMosaic等新发现候选。OAI日期不等于首发秒。下列分别标注原始v1提交、旧工作补录或本轮新增取证。对已收录论文按 ID/版本排重，本节没有把热榜登榜日期当作首发。OpenAI 11 月 12 日的报告是本轮新公开研究。

### 补录｜FP4 的近似损失曲线可能掩盖长尾事实学习变慢

**Chang Lu、Mengjiao Zhang，独立研究者｜11-09 18:30:41 UTC 提交｜arXiv:2611.12832v1。** 以可控出现次数的合成人物事实测量预训练的“曝光成本”，而非只比较 token 平均损失。40M 架构上，仿真 NVFP4 相对 BF16 在纯合成语料和混入 FineWeb-Edu 时，达到相同召回需 1.32×、1.52× 曝光，六个种子的区间分别为 [1.16,1.50]、[1.35,1.71]。对应损失只差约 0.5%、2%。把两模型都切成 BF16 GEMM 读出后仍需 1.26×、1.30×，显示部分差距在学习阶段形成。作者用拆分 Adam 两个矩估计的噪声实验区分步长与更新信噪比。仅提高学习率未消除差距。**建议深入阅读并考虑小规模复现**：可在低精度配方验收中增加按曝光频次分桶的事实探针。模型最多 96.6M、1B token，单 RTX 4090 上以 FP32 乘法仿真量化操作数。没有原生 Blackwell FP4 吞吐或位级验证，探针沿用训练模板，不能直接外推生产模型。论文提供方法与实验配置，正文未找到公开实现链接。 [固定正文 §3–7、Limitations](https://arxiv.org/html/2611.12832v1#S3)

### 补录｜NVIDIA 用随机舍入缓解 RLVR 中 BF16 权重更新停滞

**NVIDIA｜11-10 04:07:19 UTC 提交｜arXiv:2611.13545v1。** 小学习率下，FP32 主权重已更新，最近舍入后的 BF16 权重却可能连续多步不动。这与训练和 rollout 引擎之间的数值失配是两个问题。DeepSeek-R1-Distill-Qwen-1.5B、学习率 1e-6 的首轮四个优化步骤后，BF16 仍有 98.84% 权重等于初值，FP16 为 90.79%。作者在 64 张 H100 的位级对齐对照和 64 张 B200 的其他试验中，测试 1.5B、14B、30B-A3B 与两组数学数据：每次更新后对主权重随机舍入一次，并把同一份 BF16 权重用于训练 forward 和 rollout，可追平 FP16 的奖励收敛轨迹。把 BF16 学习率翻倍也能加快，但相同奖励下参考 KL 更大。**考虑复现**，先监测有效权重变化率，再做同预算对照。论文的“收敛速度”指 RL 迭代数，没有证明相同墙钟收益。其 vLLM TorchInductor 奖励崩溃观察限 0.12–0.17，不代表当前版本。异步大模型与 MXFP8 仅初步结果。未找到配套公开实现。 [正文 §2–4、表 1、图 3–7](https://arxiv.org/html/2611.13545v1#S3)

## 研究论文｜后训练与 Agent 学习

### 自动红队的重复成功门槛改善安全训练数据，但不能消除系统性评分偏差

**OpenAI｜11-12 官方发布｜《Improving Fuzzy Robustness in LLM Safety with Automated Red Teaming》。** ART 让攻击模型根据防守模型和判分反馈持续修订候选，只有跨多次独立回答和评分均成功才接受，即 `worst@n`。相较一次偶然成功，重复门槛提高独立复测可靠性与对留出防守模型的迁移。受控试验从同一内部 GPT-5.6 Terra 检查点出发，将增强任务约按 3:1 混入原任务并下调其他数据，保持总 RL 预算固定。报告固定攻击安全率 45%→68%，12 类自适应攻击下 71%→92%。**建议深入阅读**数据筛选与留出攻击协议。报告也显示，更强的、专门针对最终防守模型训练的 RL 攻击者最终仍能突破。重复抽样缓解评分方差，不能解决评分偏差，因此作者选择人工核验的离线批次。控制试验中 cyber 提问更保守。GPT-6 整体安全/有用性结果含其他训练变化，不能归因于这一项。训练数据、内部模型和完整配方未开放，无独立复训。 [官方文章](https://alignment.openai.com/fuzzy-robustness/) · [报告 §2.2–3.3、§4，页 5–11](https://cdn.openai.com/pdf/OAI_Improving_Fuzzy_Robustness_in_LLM_Safety_with_Automated_Red_Teaming.pdf)

### 新增取证｜KernelMosaic 用开发轨迹训练 CUDA 生成，收益主要在单算子和算子链

**香港城市大学｜v1提交11-10 17:45:06 UTC。本轮在11-13 OAI元数据发现，精确首次公开时点未确认｜arXiv:2611.14604v1。** 从41类来源整理14,566个任务，保存实现、失败修复、性能优化和执行反馈。相同Qwen3.5-9B、最多262,144生成token/3,600活跃秒的Agent协议下，完整轨迹SFT把KernelBench v0.1的L1/L2/L3正确率从4/0/0提升至88/82/16%，比PyTorch eager更快且正确的任务覆盖为64/62/12%。推理在B200，候选执行在H200。**建议深入阅读并继续观察工件**：增益并非单次代码生成，长预算和PyTorch eager分母须保留，外部榜单协议不匹配。1k轨迹对1k直接回答的消融还同时改变正确性筛选、独立抽样及监督token量，不能把全部收益只归给推理轨迹。作者做标准化代码去重排除KernelBench，不能据此保证语义无污染。论文称开放数据/框架，本轮原文和检索未定位可下载主工件，暂不认定已具备完整复现条件。 [正文 §3–6、表 2、附录 A/B/D.5](https://arxiv.org/html/2611.14604v1#S6)


### 补录｜HiLo 传压缩梯度，让 rollout worker 在本地重放优化器

**Uppsala University / RISE｜11-09 22:18:34 UTC 提交｜arXiv:2611.13306v1。** 保持 rank-64 LoRA 的训练容量，改为同步压缩后的梯度，而非截断经过 AdamW 等优化器后的参数更新。学习器与 worker 都消费同一解码梯度，按版本顺序重放。Qwen3-1.7B 的 Knights and Knaves OPD 在三节点 GH200、1 个 learner 和 8 个 worker、每 worker 62.5 Mbps 限速时，每步从 33.1 秒降至 16.1 秒。不限制带宽时反而从 15.3 升至 15.9 秒。Countdown-4 的 0.6B 学生在约 1.6% 同步字节下 pass@4 为 63.7%，完整同步 64.0%。**建议深入阅读/考虑复现带宽受限场景**。同版本一致性依赖相同优化器状态、数值栈及消息恰好一次有序消费，不能去掉异步策略陈旧。rank-64 1.7B 配置每 worker 仅 FP32 AdamW moments 就约 558 MB。异构设备、丢包和重启未验证，限速集群不等于真实 WAN。质量只测 OPD/RLAD。正文未找到公开代码入口。 [正文 §3、§4 表 1–3、Limitations](https://arxiv.org/html/2611.13306v1#S4)

### 补录｜SteerDistill 从量化学生的实际失误处生成可验证修正

**Genming Cui 等 / SteerDistill｜11-10 00:45:58 UTC 提交｜arXiv:2611.13410v1。代码固定 `937744a4`。** 先按环境状态转移而非工具字符串定位学生与成功教师轨迹的首次行为分歧，复位到错误前，让教师指导学生自己重试。修正只要通过局部执行核验，即使任务最终失败也保留。蒸馏时，修正 turn 学带提示的冻结学生分布，其余 turn 学 BF16 教师，推理只运行学生。Qwen3-8B W4A16 五项 Agent 指标均值 53.18，对任务专用 BF16 起点 53.54、作者移植的 QeRL 51.00。**考虑复现**可执行工具环境。对照从任务专用 RL 后权重开始，QAD/QeRL 已移植为 A100 上的分组整数量化，不能拿来宣称胜过其原始 NVFP4 系统。发布仓仍把 checkpoint 下载列为 TODO。W4A8/W2A16 参考路径不等于原生低比特 GEMM 加速。代码对应：`transition_verifier.py::compare_transition` 核验前后状态/读取结果，`training/mytrainer.py::SkillKDTrainer.compute_loss` 路由 FKL/RKL，`materialize.py::rows_from_run` 分离有/无指导前缀。 [论文 §3–4、表 1](https://arxiv.org/html/2611.13410v1#S4.T1) · [固定实现](https://github.com/air-embodied-brain/SteerDistill/tree/937744a4003eb96078ebe852f023d1249c2466b4)

### 补录｜CAMP-II 把已有文件操作先验用于记忆蒸馏

**CAMP-II 作者团队｜11-10 12:49:24 UTC 提交｜arXiv:2611.14009v1。** 把持久记忆写入普通文件，跨上下文替换仅保留文件路径，让学生自行读取。异步 OPD 直接监督学生采样的记忆写入 token。相同 Qwen3.8-27B 教师、4B 学生、100 个 learner update、6,400 episodes 下，640 个五领域测试任务均值为 64.6%，对文件记忆但用任务奖励的 CAMP-I 为 54.2%，冻结 4B 为 45.3%。屏蔽记忆 token 的损失会丢失大部分收益，等量随机屏蔽仍为 63.3%。**建议深入阅读**接口熟悉度与训练信号的交互。它给“先更换记忆工具还是先训练记忆内容”提供可控实验。外部测试限无网络子集。一名教师、学生最多 9B，协议和任务构造不能替代真实跨领域迁移验证。正文未找到公开实现链接，结论依论文及其消融，未独立复训。 [正文 §4–6、表 1–2、图 6–7](https://arxiv.org/html/2611.14009v1#S6)

### 补录｜只学习删掉少量权重，也能得到接近全量微调的数学结果

**Deletion Tuning 作者团队｜11-07 17:33:20 UTC 提交｜arXiv:2611.10866v1。** 冻结底座，在 MLP gate projection 每个权重上学习一个退火二值掩码。部署只把选中的原权重置零，不生成新非零值。H200 上按相同训练步数比较，Qwen3-30B-A3B 在 SimpleRL-Zoo 六基准的最佳验证检查点均值为 58.54，对全量微调 58.43。压缩掩码为 29.6 MB，前者仍需保留原模型。**建议深入阅读/继续观察**小任务切换和能力保留。GRPO 实验删掉不到总权重 0.05%，不能期待稀疏矩阵计算加速。训练仍有逐权重连续 gate 和优化状态，磁盘掩码小不等于训练显存小。结果使用最佳验证检查点，较小底座有落后，尚缺公开代码和本轮独立复现。 [正文 §4、§5.1–5.4、表 1–3](https://arxiv.org/html/2611.10866v1#S5)

### 补录｜密集过程奖励的优势会在计入全部计算后反转

**Case Western Reserve University｜11-10 04:25:22 UTC 提交｜arXiv:2611.13560v1。** 研究同时匹配生成 token 与训练 GPU 小时，把中间状态续采、丢弃样本、打分和同步都计入预算。Qwen2.5-1.5B 的 Monte-Carlo 过程奖励虽在相同步数领先，却因 91% 生成 token 用于额外续采，在相同 token/训练时长下分别比 outcome-only GRPO 低 14.3/7.8 个百分点。六个小模型、1,668 次运行表明，按预测成功率接近 1/2 选 prompt 有小幅收益，但把基线初始学习率翻倍已能接近至半个百分点内。**建议深入阅读**预算、随机选择和调参控制。任务限数值答案 MATH、小模型及短输出（380/768），主配方不含 ratio clipping/KL，不能推广为所有过程奖励无用。作者声明发布训练器和运行记录，当前固定仓仅有“CODE WILL BE AVAILABLE SOON”，复现包尚未兑现。 [正文 §2–4、图 1–3](https://arxiv.org/html/2611.13560v1#S3) · [固定仓状态](https://github.com/vicky157/grpo-validation/tree/9b8a13dd2bb4f5a0fbfdb804df6eb9becbeee1f1)

### 补录｜奖励投机取决于模型是否跨 prompt 共享同一种错误

**RLVR-Reward-Hacking 作者团队｜11-09 20:39:36 UTC 提交｜arXiv:2611.13218v1。实现 `1cdddd43`。** 把 verifier 的静态误报率与“同一错答案在多少题被接受”分开控制。在 64 道乘法、100 种两位数输出和 101,706 参数模型上，跨 prompt 参数共享既可抑制特有漏洞，也可放大通用漏洞。给每题独立复制相同 Transformer 后，这一普遍性斜率消失。采样 GRPO 下投机比例相对底座的变化从 −0.103 到 +0.663，十种种子。**建议深入阅读并复现诊断方法**：verifier 硬化不宜只按误报总数排序，还应估计模型在这些漏洞上的概率质量和共享梯度。它是可穷举的玩具任务，漏洞来自 SFT 注入，不能作为大模型因果定论。Adam 下放大符号依赖步长、种子。代码 `env.py::LeakyVerifier` 分离两个控制量，`exp10_copies.py::train_copies` 执行去共享对照。 [论文 §5–7](https://arxiv.org/html/2611.13218v1#S6) · [固定实现与结果](https://github.com/Rohan-Pandey1729/RLVR-Reward-Hacking/tree/1cdddd43cf6a37d6f7d6d32ec9fa8a3348b93e7b)

## 研究论文｜推理优化与服务系统

### 补录｜MACaQ-R 精修 GGUF 码值，校准峰值内存不同于推理占用

**Huawei CSL / MACaQ｜论文 v1 11-07提交，已在11-11日报收录。MACaQ-R workflow于11-11 22:37:59 UTC入库，11-12 09:57:25 UTC补充introduction，固定 `3810efca`。本轮回溯补录实现路径。** MACaQ 给每个 Linear 独立做通道缩放、32 元素块 Hadamard 和 Hessian 引导的 FP4 相邻码值翻转。激活变换留在运行时，避免把变换跨层传播，因而可以逐层加载校准。Qwen3.5-122B-A10B 在单张 96GB GPU、32 flips、512K 校准 token 下，64 分钟完成量化，峰值设备内存 32.5GB。结果模型推理占 74GB，WikiText-2 PPL 为 5.14 对 FP16 5.06。MoE 只量化专家。论文的 **11×** 是同硬件 Llama-3-8B 上达到 DuQuant++/MR-GPTQ 最佳 PPL 的量化时间对比。不是推理加速。Qwen3-32B 的在线变换额外开销 2.3% 限 batch 1、context 512，论文只给设备 504 TFLOPS FP16 / 96GB / 1.79TB/s，未命名型号。

上述量化数据补全旧论文的实验条件，本轮补录的 **MACaQ-R** 实现 保持原 GGUF 格式及 block scale，只精修权重码值，可用标准 llama.cpp。完整 MACaQ 的激活变换需要随仓 runtime。vLLM、Diffusers、PyTorch 是各自的适配与依赖环境，不能把“model agnostic”理解为任意模型开箱即用。CANN ops-nn !12568、ops-tilelang !44 本轮均仍 **open、未合入**，全模型昇腾执行还需运行时集成。**考虑复现**小规模质量/校准成本对照，再评估目标 backend。目前未独立量化或压测。固定源码 `quantize/macaq_mxfp4.py::mxfp4_realquant_h` 按缩放、旋转、选指数、refine 顺序产出工件，`_flip_sweep_fast` 每行只接受损失下降最大的相邻码值变更并增量更新。`OCPDenseW4A4Linear.forward` 把变换交给所选运行核。 [论文 Method、Experiments、图 2–3](https://arxiv.org/html/2611.10649v1#Sx3) · [固定代码与后端边界](https://github.com/huawei-csl/MACaQ/tree/3810efca0b9e819c8eb51addd93c3c87225b7476) · [说明增量提交](https://github.com/huawei-csl/MACaQ/commit/3810efca0b9e819c8eb51addd93c3c87225b7476) · [未合入 ops-nn](https://gitcode.com/cann/ops-nn/pull/12568) · [未合入 ops-tilelang](https://gitcode.com/cann/ops-tilelang/pull/44)

### 补录及新版本入口｜Together Compiler 以常驻引擎提高优化试验吞吐

**Together AI｜官方文章及原报告11-04。arXiv v1提交11-10 19:37:09 UTC、本轮发现｜arXiv:2611.14681v1。** 热加载候选forward而非反复重启服务。少层快tier筛掉错误/慢方案，多层慢tier用真实数据分布确认收益，成本模型选择TP/EP等初始布局。论文受控对照中，两tier评估1,354个候选，对单慢tier168个，找到的结果再快7.2%。对截至10-27的AgentX最佳镜像，以同并行策略、每配置24小时/两倍副本优化资源，报告最高2.6× TPS。另一个MiniMax-M3.1-Flash低延迟试验把TP4改TP8后得1,111 TPS，对公开633，因此这一点包含并行度变化。**建议深入阅读**测量保真度和长驻试验平台。Trainium2用70K输入的合成多轮而非AgentX。改真实输入分布后有优化退步，不能只用随机数据核时间验收。原报告38页，本轮arXiv文字可定位。未核到公开TKCC源码，厂商生产应用声明不等于独立复现或开源发布。 [官方11-04文章](https://www.together.ai/blog/together-compiler-continual-compilation-for-ai-inference) · [原报告封面与 §4–5](https://www.together.ai/together-compiler-paper) · [arXiv版本 §4.2、§5.2、附录 A.3](https://arxiv.org/html/2611.14681v1#S5.SS2)


### 补录｜Evolving Infra 让 P/D 角色切换与 KV 生命周期分别管理

**Georgia Tech、UIUC、UC Berkeley、NVIDIA、OpenAI、MIT 作者团队｜11-09 12:26:02 UTC 提交｜arXiv:2611.12304v1。** 常驻对称引擎可以原地切换 P/D 角色，跨节点共享 KV 池保留缓存。目录按 Agent 活跃分支计引用，优先驱逐 compaction 后或已结束分支的缓存，保留等工具返回的旧前缀。在线 planner 用 cached/uncached token、输出长度及投机接受量拟合阶段工作量。36 张 GB300、GLM-5.2/Kimi K2.6、NVFP4 权重/FP8 KV/MTP 条件下，对 Dynamo 1.5.1 的静态负载试验，平均 E2E 延迟降低 1.15–4.12×，角色切换平均 6.2 秒，对需重启的基线 754 秒。**建议深入阅读**Agent 元数据如何进入缓存策略。最高 13.92× P95 TTFT 并非所有负载：decode-heavy Putnam 为吞吐减少 P 节点后，两档并发的 P95 TTFT 更差。Claude Code 是固定输出长度/工具耗时的轨迹回放。正文未找到系统公开实现，3,072 GPU 是模拟，不能与实测 36/256 GPU 混写。 [正文 §4–6、图 6–8、附录 D](https://arxiv.org/html/2611.12304v1#S6)

### 补录｜NVLinkOneSided 用来源 rank 专属槽位取消先交换元数据

**NVIDIA｜11-09 20:06:38 UTC 提交｜arXiv:2611.13188v1。** 每个 source 只写目标 GPU 上自己的区域，同一 token 命中该 rank 多个专家只传一次。grouped GEMM 通过槽位索引读取，免去完整 expert-major 重排。Fence 以显式标志同步，Sentinel 借数据到达判断就绪。在 8 张 B300、DeepSeek-V4.1-Flash、每 rank 1 token 时，Sentinel dispatch+combine 为 17.4µs，对 DeepEP 47.9µs。整层为 36.7µs，对 MegaMoE 49.8µs。**建议深入阅读/考虑微基准复现**小批 decode。到每 rank 256 token 及以上，能重叠通信与计算的融合 MegaMoE 领先，不能把小批优势外推 prefill。限一个 NVLink domain，工作区随 EP 规模增长，不是跨节点 RDMA 方案。已核 FlashInfer `60bd3e6e`：`include/flashinfer/comm/moe_alltoall_sentinel.cuh::moeA2ASentinelDispatchKernel` 推送并接收payload，`moeA2ASentinelCombineKernel` 等专家输出、收回部分结果并重置槽位。公开源码支持该机制，但未本轮压测。 [固定实现](https://github.com/flashinfer-ai/flashinfer/blob/60bd3e6e0cbb4c1363dd0122627678f62ecc04a0/include/flashinfer/comm/moe_alltoall_sentinel.cuh) · [论文 §3–6、表 4–6](https://arxiv.org/html/2611.13188v1#S5)

### 补录｜FOSCAR 把低比特 KV 的 scale/offset 移出逐元素重建

**FutureMLS-Lab / FOSCAR｜11-06 17:00:25 UTC 提交｜arXiv:2611.09722v1。公开实现 `9089e1ff`。** 依据量化分组方向，把仿射 scale/offset 折到另一个矩阵操作数或乘法结果，减少低比特 KV 反量化的发射指令。折叠在精确算术下保持同一解量化结果，但不能消除 KV 量化本身误差。H100/B200、32 Q 头/8 KV 头、head dim128、page64、group128 的 decode sweep 中，INT2 相对 FlashInfer BF16 的几何平均核加速 3.36×/2.67×。对更强 TensorRT-LLM FP8 为 2.03×/1.48×，B200 对 NVFP4 仅 1.16×。单卡 Qwen3-8B、100K 输入/1K 输出、99% prefix hit，替换 OSCAR 的核并保留旋转/BF16 窗口后，H100 峰值总吞吐 487 对 269 token/s。**考虑复现**目标上下文与缓存命中率。质量对照主要是短 prompt 的 GSM8K/GPQA，不能证明 100K 长文质量等价。吞吐还包含容量允许的并发差异。固定实现 `IntDecodeSm100` 对应折叠校正，`quant_layout.py::validate_quant_flex` 限定分组/layout。channel-wise K 的额外舍入与误差在源码明确披露。 [论文 §4、§6](https://arxiv.org/html/2611.09722v1#S6) · [固定实现](https://github.com/FutureMLS-Lab/FOSCAR/tree/9089e1ff5a9eb5fdef70a6a576fe2d194277ccc6)

### 补录｜Transformer-FSQ 为每 token 存离散潜码，极限压缩仍有质量代价

**Transformer-FSQ 作者团队｜11-06 15:27:24 UTC 提交｜arXiv:2611.09598v1。** 在冻结底座上训练 KV 投影的有限标量量化瓶颈，存 bit-packed 整数潜码，decode 在融合 Triton 核内重建。对 Qwen3-0.6B/1.7B/8B，用每原始参数 0.25 个训练 token 适配后，在 8,192 长度 RULER 上，12–97.5× 缓存压缩的准确率高于作者测的五类基线。4 张 L40S 分别运行单进程计时，Qwen3-8B 在 32× 压缩时全模型 decode 吞吐为未压缩 native SDPA 的 1.01×（区间0.98–1.05），97.5× 时1.54×。**建议深入阅读并继续观察**紧内存部署：它不是几乎无损压缩，8B 的32× RULER由未压缩0.922降至约0.493。“百万 token 容量”含并发序列的投影容量，质量只测8K RULER和≤32K的三项LongBench，未证明百万长文推理可用。原始吞吐基线是 SDPA，不能当作胜过最新服务引擎。正文未找到公开代码地址。 [正文 §4–6、图 2/6、表 2](https://arxiv.org/html/2611.09598v1#S5)

### 补录｜DARIC 以分布感知重算连接位置无关 KV 片段

**DARIC 作者团队｜11-06 07:41:55 UTC 提交｜arXiv:2611.09109v1。** 针对拼接独立文档 KV 后缺失跨文档依赖的问题，按因果注意力结构分配片段重算预算，配套 Redis radix 索引、host/disk cache 和 vLLM KV Connector。四张 H20 96GB、256GB host cache、FP16 配置中，MiniMax-M2 的前500条 SWE-bench 测试使用 Claude Code harness：成功率68.4% 对完整重算68.6%，TTFT1,417 对2,923ms。**建议深入阅读**可复用片段的质量–延迟预算。运行基线是vLLM0.11.0/0.11.1，EPIC由作者移植且预算略不同，当前版本行为需重测。Mistral-7B 的复杂多跳任务仍有质量差距，短函数调用改用更小预算。正文未找到公开代码链接，不能把研究集成当作当前vLLM已发布能力。 [正文 §3–5、表 1–3](https://arxiv.org/html/2611.09109v1#S5)

### 补录｜同架构模型更新后直接复用旧 KV，会改变答案与停止行为

**DeepSeek-V4-Flash cache study 作者团队｜11-10 13:17:04 UTC 提交｜arXiv:2611.14051v1。** 研究固定 token 输入，把 Flash 的缓存交给更新后的 Flash-0731，分别测恢复自身缓存、直接跨版复用及校准映射。在8张H100、SGLang固定`1fdd6c89`、TP8、256-token页/4,096-token prefill 的585样本中，2WikiMQA F1从61.7降至54.7，prepared-cache TTFT中位逐样本加速2.097×。短BABILong则0.855×，反而更慢。**建议深入阅读**跨版本缓存有效性验收。prepared-cache不含先前生成/持久化旧缓存成本。部分同版恢复也与原生生成不同，七条存储条件异常另做post-hoc重测，作者保留主表。第一token接近不保证整段输出相同，F1损失也可能来自EOS变化。未找到公开完整复现包，本轮未运行服务。 [正文 §3–4、附录 A–D，表 8–14](https://arxiv.org/html/2611.14051v1#S3)

### 补录｜REAL Loss 修正投机接受长度目标的分支加权

**Yu Tian / REAL Loss｜11-10 17:45:12 UTC 提交｜arXiv:2611.14399v1。作者分支 `3dff4579`。** 连续接受的前一个 token 同时决定下一次验证前缀。把每步平均接受率直接相乘会丢失这一分支依赖。REAL 使用目标采样路径权重，并对末步词表求和降方差。Qwen3.6-35B-A3B target、Domino草稿、7个预测token、3 epoch下，九任务/五seed的概率采样接受长度4.792，对static LK4.740、WTV4.715。**建议深入阅读/考虑损失消融**，收益是受控小幅增量。greedy服务与温度0.5时static LK更好，VSD/DBLast是loss级适配，非完整系统复现。源码`models/dflash/metrics.py::compute_eal_terms`构建路径权重与overlap，`compute_metrics`消费对应block目标。需要共享词表和目标采样数据，不能当作vLLM主线已集成。 [论文 §3–4、表 1/5/11](https://arxiv.org/html/2611.14399v1#S4) · [固定作者实现](https://github.com/King-ty/speculators/tree/3dff4579c5f93f692108df8b5178c9e9200b8536)

### 补录｜POEM 根据删除 block 对输出的影响选择稀疏 prefill

**POEM 作者团队｜11-10 04:52:58 UTC 提交｜arXiv:2611.13578v1。** 先按注意力质量选候选block，再结合其value与估计输出的偏差重排，并离线按KV共享组调阈值。单A100、Qwen3-8B、128K下，含selector的注意力时间143.9ms，对同后端dense FlashAttention-2为928.0ms。这是注意力局部6.5×，未报告等比例端到端服务收益。Qwen3-14B LongBench38.68接近dense38.77，RULER均值89.68仍低于dense91.72。**建议深入阅读/继续观察**输出误差代理。对照限定FA2后端block选择，未纳入带在线PV跳过的SpargeAttention。Qwen需YaRN扩窗，Llama3.1-8B部分分项掉点。公开实现承诺在匿名补充材料，正文未给可定位仓库。 [正文 §4–5、表 3–5、图 5](https://arxiv.org/html/2611.13578v1#S5)

### 补录｜粒子解码的熵读出可用于分配后续采样预算

**SRE 作者团队｜11-10 00:49:36 UTC 提交｜arXiv:2611.13414v1。** 从Power-SMC已有粒子权重累计序列Rényi熵，先用8粒子探测，再只对高不确定题重跑64粒子。Qwen2.5-Math-7B、MATH500的三个seed中，统一对60%题升级可达到均匀64粒子的平均准确率76.6%，粒子预算减少27.5%。逐seed选择升级率的平均节省34.2%包含更大的选择自由度。**建议深入阅读**预算控制，同时测实际GPU时间。零边际成本仅相对已经运行的SMC，不能称普通单次生成免费获得同等信号。理论保证针对标准SMC估计，实验stepwise均匀累计是近似。任务限数学，完整读出patch与逐题日志仍待发布。 [正文 §4、§5.3–5.6、附录 T](https://arxiv.org/html/2611.13414v1#S5.SS6)

### 补录｜ReTrace 把成功 Agent 轨迹编译成无需运行时规划的流程

**ReTrace 作者团队｜11-10 18:49:47 UTC 提交｜arXiv:2611.14492v1。** 从SOP与含推理的轨迹抽取条件和动作，跨轨迹合并再生成代码，并在训练输入上修补流程。τ2-bench telecom solo模式、40条测试输入上，ReTrace任务成功75.8%，原Agent80.6%，每任务时间1.3秒对170秒。加入修补后81.2%。**继续观察**高频、边界稳定的工具流程。它减少运行时规划而非免除所有模型/外部工具，构建成本只给下界。对同一SOP的新输入，不是新流程迁移。企业案例平均时间163秒反高于Agent153秒，部分原因是失败的网页工具调用更快，平台“完成”也未必是真成功，须人工结果审计。正文未找到公开代码入口，不能直接采用为受证实可靠的自动编译器。 [正文 §3–5、表 1–2、§7](https://arxiv.org/html/2611.14492v1#S5.T1)

## 研究论文｜评估与训练可靠性

### 补录｜Plumbline 以独立参考逐段验证训练，而非用收敛代替正确性

**Amazon｜11-10 00:13:56 UTC 提交｜arXiv:2611.13393v1。** 把数据模板/标签、模型加载、保存恢复、导出服务、优化器更新与架构单步梯度分开核对。典型做法包括加载前以NaN填满参数、学习率0导出应逐位保留原权重、恢复后逐rank比较保存状态，以及与CPU FP32参考对照首几层真实权重的梯度。三框架/五模型的13组组合发现61个问题，其中32个静默。生命周期/单步检查首先发现29个静默问题，至少16个对应运行通过了收敛检查。**建议深入阅读并采用验证思路**，特别适合新模型接入。梯度阈值为测得噪声底的4倍，不能任意套用。若训练器与参考共享同一错误代码便可能漏检，没有可信参考时应返回未判定。论文称生产使用并计划开源，未给公开实现，问题发现数来自作者内部审计，无独立复验。 [正文 §3、表 1、§4.2–4.4、§5](https://arxiv.org/html/2611.13393v1#S3)

### 补录｜SELD 检查压缩后“答案未变、证据已丢”的情况

**Wentao Lei、Li Liu｜v1记录10-14 15:58:07 UTC提交，11-11公告列入｜arXiv:2611.12903v1。** 比较原文与删改文本第一回答位置的全词表logit模式，去掉整体平移和缩放后看角度变化，检测只盯答案无法发现的证据损失。2,414题HotpotQA/2Wiki构造等token长度的证据删除/非证据对照，在答案保持不变的配对中，SELD正确排序82.7%，JS散度70.3%、原答案似然下降53.9%。LLMLingua-2真实输出中的等长交换、Provence与第五个Granite读者也支持该方向。**建议深入阅读/考虑加入压缩验收**。这是同题两个删改的相对排序，不是已经校准的跨请求在线告警。需要原上下文及全词表输出，自然压缩中的全局阈值和下游模型迁移仍有边界。代码/SEL-Pairs重建脚本为待发布，未独立运行。 [正文 §3–5、表 1–2、§6](https://arxiv.org/html/2611.12903v1#S5)



### 新增取证｜tau2-bench 的稳定 LLM 判分与真实写操作正确性并不等价

**Sungbin Im / aSSIST、AIM Intelligence｜v1提交11-10 05:34:47 UTC。本轮发现｜arXiv:2611.14581v1。** 在固定tau2-bench `363133ad` 的90项airline/retail任务、两模型、四种gate、三次重复，共2,160条episode中，gpt-4.1-mini judge两次自一致98.2%，但与原生终态评分仅一致56.2%。两人标注120条后确认11次不应发生的写入，原judge及带policy的judge均放过11次。不截断输入后仍放过10/11次。**建议深入阅读/采用多轴验收**，分别测任务完成、policy和commit正确性，并记录实际数据库写入。无需模型的边界检查捕获全部11次，但借助“任务是否预期写入”的gold信息。它会漏掉内容/时机错误，不能宣称部署中无gold就能替代完整评测。更强judge结果随policy提示显著变化。actor和judge主要来自单厂商、两领域，gate未显著改善也不能外推所有验证器。作者称代码/轨迹在补充材料，本轮未定位独立公开下载链接。 [正文 §4–6、表 3–6、§8](https://arxiv.org/html/2611.14581v1#S5)

### 新增取证｜清理 Agent 历史前，先核对错误是否仍留在外部状态

**Salesforce AI Research｜v1提交11-10 19:06:36 UTC。本轮发现｜arXiv:2611.14654v1。** 冻结任务中途的历史与世界快照，分别改文本记录和实际状态。AgentCE-Bench 319状态、两接手模型中，保留错误并附真实更正说明使后续目标恢复率50.57%→65.78%。把错误改写成“已完成正确动作”，外界仍错时则从未编辑的50.13%降至43.70%。AppWorld 65任务族的交叉试验也显示：动作实际未生效时成功回执比失败回执差16.92个百分点，已生效时则好10.38。**建议深入阅读**compaction/接手协议的状态一致性。它不是已验证的自动压缩策略。前者同时改变真伪、角色、位置和保留信息，后者回执内容也不完全匹配。AppWorld预注册脚本因封存门槛未运行，结果由两份独立重实现交叉核对，偏差明确保留。正文未给公开完整工件链接，无本轮复现。 [正文 §4–6、表 1、附录 G/H](https://arxiv.org/html/2611.14654v1#S5)

### 新增取证｜高弃答率并不能证明模型按置信阈值行动

**Dongsun Moon，独立研究者｜v1提交11-10 19:13:43 UTC。本轮发现｜arXiv:2611.14661v1。** 在四组3–8B开放训练配方的SFT/偏好/最终12个检查点上，将“仅回答答案或I abstain”的格式行与得分阈值交叉。1,000项TriviaQA/PopQA的配对设计中，t=0.75加格式行使平均弃答20%→91%，但同格式内把阈值0.5升到0.9，弃答反少0.8个百分点。**建议深入阅读**阈值遵从的测量：同时报告覆盖率、阈值变化方向及被回答题目的正确性。外部校准P(True)可构造单调覆盖，但需标注校准集、额外查询和跨分布失败。单一OLMo3 LoRA干预没学好未见阈值，不证明训练普遍无效。限短事实题、greedy与指定措辞，未查预训练污染。论文公开模型固定版本信息，但本轮未定位完整实验代码。 [正文 §3–6、表 2/4](https://arxiv.org/html/2611.14661v1#S4.SS2)

<a id="engineering"></a>
## 开源工程

### Megatron 修复原生 FP8 GTP 优化器 checkpoint 的参数身份

**11-12 23:24，`main`默认分支。** GTP 在 `--fp8-param-gather` 下把 FP8 参数反量化为 BF16 保存时，副本与原参数的对象身份不同，参数型 `fully_reshardable` 优化器映射会找不到对应项。新增 `set_checkpoint_parameter_id` / `get_checkpoint_parameter_id`，让保存副本显式关联原参数。默认 `dp_reshardable`、模型单独保存不受这条缺陷影响。作者用 4 张 GB300 验证 Adam/Muon 映射和 BF16 权重/FP32 moment 往返，明确尚未验证完整训练 save/resume 等价。建议考虑复现：与昨日 SwiGLU 分片顺序修复一起，重点跑目标 GTP 格式的跨并行度恢复。 [Megatron-LM #9343 `83295310e3`](https://github.com/NVIDIA/Megatron-LM/pull/9343)

### Megatron 的两项 MFSDP 修复分别收紧参数收集与梯度归约边界

**11-12 16:13–11-12 16:28，`main`默认分支。** MFSDP v2 将 mHC 包装层设为 FSDP unit。包装层直接调内部 helper 会绕过 `__call__` hooks，重计算中的内部 layernorm 另设 unit，保证参数 gather 与梯度 reduce。MFSDP v1 则把 HSDP/HFSDP 的 DP-Outer 归约延到 DP-Shard 完成后，每 bucket 每步只执行一次，避免 MTP 与主模型分别反传共享 embedding 时重复累加。代价是失去末 microbatch 的这部分通信重叠。作者在 EP2、embedding 独立 bucket、`deterministic_mode` 下的 GPT+MTP 对照，由首步损失差 7–12% 恢复到 5e-6 内。PR 声明叠在 #9114 的梯度 bucket stream race 修复上。建议采用前先复跑真实 overlap schedule 与共享参数配置。 [Megatron-LM #8554 `4a9e32695e`](https://github.com/NVIDIA/Megatron-LM/pull/8554) · [Megatron-LM #8941 `26a2da6239`](https://github.com/NVIDIA/Megatron-LM/pull/8941)

### Megatron HybridModel 允许下游用 Python 配置接入自定义层

**11-12 21:41，`main`默认分支。** 下游 `TransformerConfig` 子类可通过 `layer_type` 指定 `MegatronModule`，在 `hybrid_layer_config_list` 中与内置层组合，无需向 Core 注册或另写 module spec。旧 pattern 字符串仍只接受内置层。补丁同时覆盖层分配、recompute 和 CP layout 适配，不能据此推断任意自定义算子已兼容全部并行策略。建议深入阅读：适合把模型专属层从 Core 移到下游，但应按自定义层实际通信与重计算测试验收。 [Megatron-LM #9116 `fccce52694`](https://github.com/NVIDIA/Megatron-LM/pull/9116)

### Megatron checkpoint 规划按分片布局避免二次复杂度验证

**11-12 23:51，`main`默认分支。** 全局保存计划原先沿最大 tensor 维度扫描重叠，MoE 的专家轴与 GTP 特征轴分片使大量 chunk 同时活跃，协调 rank 的验证退化到逐对比较。新增 `global_plan_validation.py`：规则网格用重复检索，单个不规则维度在每网格单元内做区间扫描，其余布局及零尺寸 chunk 回退逐对检查。保留边界、覆盖体积与重叠三个判据。代码给出的复杂度分别为线性、n log n 和二次。建议考虑复现大专家 checkpoint 的协调端 CPU 时间。补丁没有提供通用训练吞吐收益。 [Megatron-LM #8591 `97b50bdd93`](https://github.com/NVIDIA/Megatron-LM/pull/8591)

### TorchTitan 的 LoRA RL 同步只发送可训练参数，修复 lm_head 加载时序

**11-12 11:42–11-13 05:50，`main`默认分支。** 生成器先独立加载冻结基座，`Trainer.push_model_state_dict` 只发可训练参数与 buffers。`LoRATransform.trainable_modules` 可保留完整视觉塔训练，且校验拼写、LoRA 目标重叠和 tied weights。加载前先完成独立 `lm_head` 所属 FSDP group 的 pending forward，再 reshard，避免 adapter-only 的设备拷贝与预取 all-gather 竞跑。作者在 4×GB300、Qwen3.5-0.8B Geometry3K rank32 LoRA 中把每次同步从约 2.2GB 降到 38MB，后续步均时 10.4s→8.6s。全参训练与 LoRA 的奖励不可视为同一优化目标的性能对照。冻结量化参数暂不支持此加载路径，每个生成器仍各自读基座存储。建议考虑复现，尤其验证恢复后第3次及更晚权重同步。 [torchtitan #5844 `204b211832`](https://github.com/pytorch/torchtitan/pull/5844) · [torchtitan #5982 `8e63247cd5`](https://github.com/pytorch/torchtitan/pull/5982) · [torchtitan #6003 `f83c46dc0c`](https://github.com/pytorch/torchtitan/pull/6003)

### TorchTitan 为 Muse Spark 接通可 CUDA Graph 的 HybridEP

**11-12 15:55，`main`默认分支。** Hopper 上 Dist-MoE 不可用，原 AllToAll EP 每层主机同步。HybridEP 的非阻塞容量行可能未写入，若直接对所有行做专家后 RMSNorm，会让脏行参与参数梯度。补丁在 `output_postprocess` 前后 mask 有效行，并用 `local_compile` 融合 mask，允许非阻塞 HybridEP 与 CUDA Graph。作者报告相应配置较 AllToAll 约高60% TPS，但这是特定 Muse Spark 路径，不能外推到全部 MoE。建议深入阅读容量行的正反向 mask，并以目标模型的梯度和图回放验证为采用前提。 [torchtitan #6032 `09d76a5177`](https://github.com/pytorch/torchtitan/pull/6032)

### TorchTitan 显式声明视觉编码器为非序列并行区域

**11-13 02:28，`main`默认分支。** `disable_dense_sp()` 在视觉 encoder 的 forward 内暂时关闭 decoder 的 SP 标志，Qwen3.5、MoonViT 和 NaViT 的视觉塔及 projector 因此使用统一的 RowParallelLinear 但归约为 TP 不变张量。此前依靠独立 Invariant 类逐处绕过 decoder 的 reduce-scatter。上下文退出后恢复设置，补丁不是让视觉塔新增 SP。建议深入阅读：在多模态模型重构 TP/SP collectives 时，这个区域边界能避免把完整 patch 序列误当分片激活。 [torchtitan #6043 `c6f7828698`](https://github.com/pytorch/torchtitan/pull/6043)

### 恢复训练修复：远端 checkpoint 保留策略、分组学习率与 fused optimizer step

**11-12 22:26–11-13 06:53，`main`默认分支、`master`默认分支。** TorchTitan 对 fsspec 关闭 listing cache，使异步保存子进程写出的 checkpoint 能被 `keep_latest_k` 看到并清理，代价是更多对象存储请求。Megatron-Bridge 恢复 scheduler override 时保留自定义参数组的 LR 上下界，避免 LSQ scale 组随全局 lr=0 停训。布局变化则警告并回退旧行为。DeepSpeed universal loader 为 fused/capturable 组把 `step` 放为参数设备上的 FP32 tensor，覆盖 ZeRO1/2/3。本次作者回归只在 CPU 上运行，GPU恢复仍应验收。建议将这三类条件加入现有恢复演练，检查参数/优化器状态之外的保留策略与分组超参。 [torchtitan #6061 `240d6bf853`](https://github.com/pytorch/torchtitan/pull/6061) · [Megatron-Bridge #6698 `5ae0dc4282`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6698) · [DeepSpeed #9056 `988b2e3196`](https://github.com/deepspeedai/DeepSpeed/pull/9056)

### Megatron-Bridge 修复 Qwen3.5-VL Hybrid attention 的 TE 图捕获

**11-12 22:51，`main`默认分支。** HybridModel 的静态图 helper 在真实 forward 前索取 RoPE，但多模态位置依赖当前 batch 的 `position_ids`，导致预热后各 rank 失败。仅在 TE attention CUDA Graph 捕获的该特定外部调用中返回 None，保留层提供的 batch-shaped rotary buffer，回放时填入当前 mRoPE。其余缺少位置的调用仍报错。补丁是等待 Core #9333 的兼容处理。建议受影响配方考虑采用，并核验图回放时不同图片/视频位置而非只看启动成功。 [Megatron-Bridge #6868 `a3fe37431f`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6868)

### DeepSpeed ZeRO-3 为模型指定的敏感参数保留 FP32

**11-13 05:54，`master`默认分支。** 此前仅敏感 buffers 保留 FP32，ZeRO-3 的单 dtype flat subgroup 仍会把 attention sink、norm 或 hyper-connection 参数变为 BF16/FP16。补丁按 dtype 分开参数 subgroup、梯度缓冲与归约，使模型 `_keep_in_fp32_modules*` 指定的参数能够以 FP32 保存、gather、更新和checkpoint。范围明确限于无 parameter offload 的 ZeRO-3。其他 ZeRO级别或参数offload会警告并保留较窄支持。建议考虑复现低精度训练中这些小参数的数值与恢复精度，不把局部 FP32等同全模型 FP32成本。 [DeepSpeed #8843 `3fe2d95081`](https://github.com/deepspeedai/DeepSpeed/pull/8843)

### DeepSpeed 为 GLM-5.2 增加可反传的 TileLang 稀疏注意力

**11-13 05:44，`master`默认分支。** 新增 `deepspeed.models.glm_moe_dsa.replace_attention(model)`，在初始化 DeepSpeed 前替换 stock Transformers 的稠密 seq×seq indexer/attention 中间量，只处理每 query 选出的2048个键并实现 backward。作者报告完整78层743.4B模型在64张H200上训练到每GPU64K上下文。8层切片16K、10步对照损失一致且快2.7倍。短训练切片不能证明长训练质量等价，也不能把64K容量结果与切片加速比合成同一测试。建议考虑复现长上下文梯度和峰值显存。 [DeepSpeed #8659 `18e005aa3b`](https://github.com/deepspeedai/DeepSpeed/pull/8659)

### NeMo RL 直接把路由专家分片重排到 TRT-LLM 本地 EP 槽

**11-12 16:43，`main`默认分支。** `refit_transport: nccl_reshard` 使用 xferdtensor 从 Megatron 训练布局搬运到生成器专家布局，避免先聚合完整 routed experts。attention、norm、embedding、router、shared expert 与MTP仍走原misc广播。只支持非共置、纯EP、无expert TP/PP/EPLB且符合Qwen专家命名的MoE。dense MLP启动时拒绝。P/D双角色已接配置与单测，但没有真实分离MoE运行记录。建议继续观察并先在支持的共识布局小规模复现，不能将分片接口视为任意M×N权重更新已验收。 [RL #4861 `defdaef486`](https://github.com/NVIDIA-NeMo/RL/pull/4861)

### VeOmni 把 QAT 从全局开关改为逐算子族选择

**11-12 16:13，`main`默认分支。** 删除 `qat_implementation` 的全局传递，为linear、SwiGLU、routed experts和KV rounding分别建立 `VeomniOp` 实现行。不同族可按部署GEMM与KV格式独立选择量化宽度。DeepSeek-V4是本次实证消费者，Qwen3接线曾尝试后撤回，不能称全模型族QAT均已验证。建议深入阅读：训推一致性配方应逐站点对齐，而不是用一个FP8开关替代不同量化粒度。 [VeOmni #1414 `f71831aa25`](https://github.com/ByteDance-Seed/VeOmni/pull/1414)

### VeOmni 接入 BAGEL 原始工件并撤下旧 checkpoint 恢复兼容层

**11-12 16:21–11-12 16:58，`main`默认分支。** BAGEL HF layout 直接读取上游 `ema.safetensors` 与 `ae.safetensors`，把视觉、VAE、文本和flow模块映射到运行时，导出时反解融合QKV回到上游键。不再要求先离线拆目录。另一个独立改动删除0.1.12/0.2.x遗留恢复转换，旧布局和未完成目录会明确报错。建议考虑采用直接工件路径，升级前单独确认存量checkpoint仍属当前按状态owner拆分的布局。 [VeOmni #1456 `cb4558f421`](https://github.com/ByteDance-Seed/VeOmni/pull/1456) · [VeOmni #1437 `e8978f59f9`](https://github.com/ByteDance-Seed/VeOmni/pull/1437)

### Molt 修复 rollout KL 信号丢失与截断回复重试

**11-12 09:16–11-13 08:50，`main`默认分支。** 以rollout policy为KL参考时，log-prob先被归为heavy字段offload，控制器读到None。Env agent在未启用IS/score centering时还可能根本不记录它。补丁让这项参考信号留在轻量记录并总是按需求请求。Chat训练遇到max_tokens截断即结束该rollout。评估可重采样同一轮，也可由agent基于截断内容继续，重试次数用独立限制。建议采用时核对评估harness与训练样本的截断定义，并用真实offload路径验收，未offload的单测不足以覆盖原故障。 [labs-molt #258 `29ebe582ef`](https://github.com/NVIDIA-NeMo/labs-molt/pull/258) · [labs-molt #265 `094f1e50ee`](https://github.com/NVIDIA-NeMo/labs-molt/pull/265) · [labs-molt #262 `f2a12e9a2c`](https://github.com/NVIDIA-NeMo/labs-molt/pull/262) · [labs-molt #263 `f320beec12`](https://github.com/NVIDIA-NeMo/labs-molt/pull/263)

### TRL 收窄 assistant-only loss 边界并补充 AsyncGRPO 信号诊断

**11-12 18:13–11-12 23:42，`main`默认分支。** Nemotron3系列的预填think块、gpt-oss与Muse Glimmer每轮开头预填的assistant标记移到generation mask外，渲染文本相同，后续模型实际生成的多channel标记仍计损失。AsyncGRPO新增丢弃零方差组的平均奖励和组内逐token重复率，用来区分题目过易/过难与采样失效。每reward函数均值改为每个实际评分rollout计一次，包含被丢弃组，避免fork样本重复加权。这些指标修正不改变优势或训练loss。建议考虑采用并重建mask黄金样本及监控口径，避免与旧曲线误比。 [trl #8217 `df64ea582e`](https://github.com/huggingface/trl/pull/8217) · [trl #8219 `338d2f432b`](https://github.com/huggingface/trl/pull/8219) · [trl #8221 `ae12802450`](https://github.com/huggingface/trl/pull/8221) · [trl #7901 `49cc76763c`](https://github.com/huggingface/trl/pull/7901) · [trl #8250 `feaa77e15a`](https://github.com/huggingface/trl/pull/8250)

### Transformers 统一 EP dispatch，训练用 all-to-all、共享批推理用 all-reduce

**11-12 18:33–11-13 01:11，`main`默认分支。** `DistributedConfig.ep_implementation` 选择experts的dispatch/combine：auto在共享批、ep_size等于tp_size的推理中用all-reduce避免每层host同步，其余走token all-to-all。DeepGEMM MegaMoE另有专用实现接收全局expert id与EP group。Gemma4、Inkling等迁移至 `ep_dispatch_experts`，GPT-OSS MXFP4同时修复blocks/scales分片维度和forward契约。旧 `ep_router`、megamoe_router/experts计划删除，旧配置会给出替代方案而非静默忽略。建议深入阅读并在升级时逐模型跑训练梯度、加载和CUDA Graph，不能把所有EP规模都套用共享批all-reduce。 [transformers #49429 `b7394bc753`](https://github.com/huggingface/transformers/pull/49429) · [transformers #49606 `f5d8e7df0c`](https://github.com/huggingface/transformers/pull/49606) · [transformers #49607 `290c247f7c`](https://github.com/huggingface/transformers/pull/49607) · [transformers #49613 `95e9b5f694`](https://github.com/huggingface/transformers/pull/49613) · [transformers #49615 `506ac61281`](https://github.com/huggingface/transformers/pull/49615)

### Transformers 修复 FSDP2 checkpoint hooks、Inkling 路由精度和多adapter恢复

**11-12 15:34–11-13 01:35，`main`默认分支。** 激活checkpoint从包装整层 `__call__` 改为包装 `forward`，使FSDP2 pre/post hooks留在重计算外。Inkling在autocast下也强制projection/sigmoid/top-k/归一化FP32，避免BF16分数先舍入为tie而选错expert，并保留校正bias/scale精度。Trainer恢复含多个PEFT adapter时先加载根目录default adapter，补足原先漏载。建议将这些改动视为正确性修复，分别验证分片重计算、专家id和LoRA参数往返。 [transformers #50490 `c527bbf6cf`](https://github.com/huggingface/transformers/pull/50490) · [transformers #50621 `52030880e9`](https://github.com/huggingface/transformers/pull/50621) · [transformers #50368 `acf8004ec9`](https://github.com/huggingface/transformers/pull/50368)

### vLLM 为流水线投机草稿隔离单stage执行

**11-13 05:39–11-13 05:40，`main`默认分支。** PP目标模型的草稿模型在最后一个PP rank内构建并运行于独立单stage group，避免继承目标stage分配而启动失败。草稿层号从目标层之后编号，部分分布式加载器各自加载草稿。FlashInfer图支持判断改用实际KV spec，不再把目标head数误套给不同head数的drafter。PP>1的OPT eager路径已有验证，相关CUDA Graph仍受#66465等条件约束。建议考虑复现目标PP与草稿head数不同的组合，并检查实际依赖版本。 [vllm #66063 `663daccfb9`](https://github.com/vllm-project/vllm/pull/66063) · [vllm #62964 `c907905a87`](https://github.com/vllm-project/vllm/pull/62964)

### vLLM 的结构化投机输出移除异步调度交接气泡

**11-13 08:31，`main`默认分支。** 草稿token的pinned D2H副本与对应CUDA event绑在一起，在调度器真正取token时等待一次。不需要token id的路径只传草稿宽度。作者用B200、DeepSeek-V4-Flash-0731、DSpark7、异步调度、PP2/TP1验证，紧约束结构化输出并发64吞吐约+2.68%。TP2/PP1与同步调度没有显著收益。建议仅在相近PP与结构化工作负载下复现，不把局部调度收益推广为常规decode加速。 [vllm #64531 `c07d58dc82`](https://github.com/vllm-project/vllm/pull/64531)

### vLLM 扩展 GLM-5.3 的低延迟与 BF16 MegaMoE 路径

**11-12 11:11–11-13 00:45，`main`默认分支。** GB300 TP4支持配置默认尝试 `latency_specialized`，小batch使用专门decode。MTP在attention/KV之后即挑出最后token行，再运行norm/MoE，减少无用行。BF16未量化checkpoint亦可显式选DeepGEMM MegaMoE，作者8×B200 DEP8、无MTP/无prefix cache、FP8 KV的一次对照中8K/1 prefill总token/s+40.3%、1K/1K输出token/s+13.6%。mHC另按架构/token阈值选择FlashInfer融合实现。PyPI wheel缺experimental源码时保持TileLang。建议考虑复现，先核实runtime、源码包和后端实际选择。各PR的收益不能相加。 [vllm #67104 `778f0fe600`](https://github.com/vllm-project/vllm/pull/67104) · [vllm #67099 `02981a8f53`](https://github.com/vllm-project/vllm/pull/67099) · [vllm #67040 `20ad1db3fe`](https://github.com/vllm-project/vllm/pull/67040) · [vllm #65521 `3de0b48c85`](https://github.com/vllm-project/vllm/pull/65521)

### vLLM 在 Hopper 运行融合 dispatch/GEMM/combine 的 FP8 MegaMoE

**11-12 17:07，`main`默认分支。** `flashinfer_moe_ep_cutedsl` 在Hopper接入FlashInfer0.7.2已有的SM90 block-FP8 MegaMoE，限定128×128 block权重、EP与NVSHMEM，并拒绝不等价的router-weight-on-input用法。作者GLM-5.3-Flash、8×H200、DP8/EP8、128K/512、MTP3、并发8–192报告输出吞吐+11–18%。4×H200的2K/1 prefill是另一配置。默认后端不变。建议考虑复现长context与忙闲rank，结合模型质量验收。 [vllm #62603 `94febce756`](https://github.com/vllm-project/vllm/pull/62603)

### vLLM 在 TP ranks 间共享 DeepSeek-V4.1 的 CPU KV

**11-12 18:59，`main`默认分支。** 主KV、indexer K和compressor state标记为TP复制后，复用既有host共享offload路径，不再每rank重复占同一CPU预算。作者8×H100、TP8、每rank20GiB GPU KV、总CPU KV固定64GiB，在407个LongBenchV2 context的3072请求回放、并发8、最多128输出中，requests/s为0.961→5.785，输出token/s25.60→146.09。收益与共享容量避免重算相关，不能当纯kernel倍率。建议复现具有大量相同TP复制缓存的offload负载。 [vllm #65373 `9a77b4201b`](https://github.com/vllm-project/vllm/pull/65373)

### vLLM KV offload 支持 DCP 混合模型的 partial tail，并修复不可命中缓存

**11-12 19:56–11-12 20:01，`main`默认分支。** partial-tail准入改按实际cache-hit alignment判断，并逐group计算物理块地址，允许DCP下attention与recurrent block大小不同。EAGLE某些证明位置尚未验证。ARC像LRU一样从运行请求的尾部chunk开始驱逐，保住可连续命中的前缀。任一必需group MISS时清除该tail边界的pending标志，避免已无可用边界仍反复延后请求。建议考虑采用并测多group缺失、容量压力与重算次数，而非只看名义缓存容量。 [vllm #58619 `8a36f080e2`](https://github.com/vllm-project/vllm/pull/58619) · [vllm #67098 `19812fa0be`](https://github.com/vllm-project/vllm/pull/67098) · [vllm #67152 `82915781df`](https://github.com/vllm-project/vllm/pull/67152)

### vLLM Mooncake 为抢占后的新尝试保留状态，并把存在性检查失败当作错误

**11-12 18:54–11-13 00:04，`main`默认分支。** prefix-cache reset、pause/sleep可在下一步同时报告旧attempt抢占与重新准入。scheduler和worker现在区分旧尝试清理与新尝试任务，避免新load丢失、断言或首次save被跳过。存在性SDK负值/异常则跳过可选PUT并记录失败，同时完成job以释放GPU引用。在途native transfer仍持有buffer直到返回。建议考虑复现带pending load的pause/resume和存储错误，关注完成回执与buffer寿命。 [vllm #66384 `3f1ebac297`](https://github.com/vllm-project/vllm/pull/66384) · [vllm #67080 `f55b08ec0f`](https://github.com/vllm-project/vllm/pull/67080)

### vLLM 为 KV store 标明触发原因，批量化 HiSparse 驻留簿记

**11-13 07:11–11-13 07:13，`main`默认分支。** offload manager的 `prepare_store/complete_store` 新增必需keyword trigger，区分计算完成与state block即将被复用的handoff。各trigger handler隔离，scheduler统一拥有job完成、fence与flush。这里是lazy store RFC第一阶段，ON_OVERWRITE与lazy模式尚属后续工作。HiSparse把一请求约1300页×20groups的逐块调用改为每操作一次block-pool批处理，保留free顺序。建议深入阅读，并同步适配外部manager签名。不要把重构写成lazy offload已发布。 [vllm #64286 `3bc1d31306`](https://github.com/vllm-project/vllm/pull/64286) · [vllm #65607 `508e5784b1`](https://github.com/vllm-project/vllm/pull/65607)

### vLLM 改善大CPU层启动与量化模型加载时的显存复用

**11-12 10:19–11-12 16:27，`main`默认分支。** 共享CPU tier拆成多个shm段，scheduler跳过其他worker已做的prefault。作者8worker H100节点的1250GiB层设置从397–460s降到135–138s。load/profile的allocator split阈值20→21MiB使释放的1–10MiB权重临时量可复用，8×B200 Kimi-K3-REAP分布式RunAI加载由690–703s降到96.8s。FlashInfer一次性autotune峰值另从长期KV reserve剔除，但保留前后真实warmup transient。三者分别解决启动、加载与缓存预算，建议按实际路径复现，不能将耗时比合成在线吞吐提升。 [vllm #66391 `9e6815acab`](https://github.com/vllm-project/vllm/pull/66391) · [vllm #66561 `6d3eba0272`](https://github.com/vllm-project/vllm/pull/66561) · [vllm #66197 `318fca3fdd`](https://github.com/vllm-project/vllm/pull/66197)

### vLLM 修复并发截断与进程内引擎清理

**11-12 12:37–11-13 07:32，`main`默认分支。** HF fast tokenizer会在backend修改每次请求的truncation配置，共享renderer线程池可静默用错长度。BaseRenderer现复用逐worker tokenizer池，覆盖DeepSeek、Kimi、Inkling等原先未池化的renderer。进程内shutdown另移除Dynamo bytecode hooks、重置global graph-pool句柄并清掉图池中cuBLAS workspace，避免下一引擎留权重、少显存或触发use_count断言。建议考虑采用并测不同截断长度并发和连续多次构建/销毁引擎。 [vllm #65932 `cfe069c9e2`](https://github.com/vllm-project/vllm/pull/65932) · [vllm #67081 `ee548e7f60`](https://github.com/vllm-project/vllm/pull/67081) · [vllm #67201 `701c2ec956`](https://github.com/vllm-project/vllm/pull/67201) · [vllm #67279 `f48bfde4dd`](https://github.com/vllm-project/vllm/pull/67279)

### vLLM 扩展 NVFP4 融合输入与 Mamba ReplaySSM

**11-12 10:05–11-13 04:08，`main`默认分支。** 更多NVFP4 linear backend可直接消费 `QuantizedActivation`，减少激活后重复量化。Nemotron共享expert的BF16 ReLU2与NVFP4量化融合，收益主要在较大token块，端到端不能用kernel倍率代替。Mamba2 ReplaySSM投机验证新增Triton verify/materialize，沿用runner所有的ring tracker，使该机制可脱离FlashInfer CUDA专属实现。不支持stochastic rounding，并保留PP/microbatch等限制。建议按实际后端分别复现字节一致性和accepted-token状态恢复。 [vllm #65149 `1c4f8c41a3`](https://github.com/vllm-project/vllm/pull/65149) · [vllm #59384 `bfc1e38113`](https://github.com/vllm-project/vllm/pull/59384) · [vllm #66318 `e509ffb943`](https://github.com/vllm-project/vllm/pull/66318)

### vLLM 的 MiniMax 稀疏注意力覆盖 Hopper 与 SM120/121，LMCache 可导入可扩展KV

**11-12 10:16–11-12 12:06，`main`默认分支。** MiniMax-M3的FlashInfer MSA在SM90/SM120/121接管原Triton fallback，旧FlashInfer或不支持的配置仍回退。作者8×H100/H200 M3.1-Flash TP8+MTP并发1–10的ITL p50降低9–22%。另一改动按connector声明的 `EXPORTS_FD` 分配POSIX-fd可导出extensible KV，供LMCache MP VMM逐层导入。不支持VMM或显式禁用extensible cache会启动时报错。建议依目标GPU和connector组合复现，并核实0.7.3以上依赖探测结果。 [vllm #66018 `d0eca76eb4`](https://github.com/vllm-project/vllm/pull/66018) · [vllm #66674 `6201c7d3cd`](https://github.com/vllm-project/vllm/pull/66674)

### SGLang 为 PP 路由回放补齐后续stage，并等所有stage释放显存

**11-12 19:48–11-12 21:34，`main`默认分支。** `return-routed-experts` 的int32 `[tokens,layers,topk]`随microbatch逐stage合并，最后stage回送后由首stage写host cache，修复原先PP后半层专家id全零。PP投机与该开关组合仍明确拒绝。`release_memory_occupation` 则先TP一致判断idle，再逐PP stage归并释放结果，首stage只有收到最终确认才答复，防训练器过早分配GPU内存及部分rank独自返回导致barrier挂住。建议考虑采用并以PP2以上真实训推切换验收。 [sglang #48412 `caae1311f9`](https://github.com/sgl-project/sglang/pull/48412) · [sglang #49798 `30593f940b`](https://github.com/sgl-project/sglang/pull/49798)

### SGLang 同日撤回 GDS recurrent-state 默认淘汰策略

**11-12 19:49–11-12 21:42，`main`默认分支。** GreedyDual-Size最初按“从更浅checkpoint重建该state需多少prefill”给recurrent state定价，并把混合模型默认从LRU改为GDS。数小时后整个改动被撤回：维护者指出默认策略尚未达成一致，证据只来自一个模型和一条trace。截止本轮，默认仍为LRU，GDS选项也随撤回移除。建议继续观察。这是合并与撤回的状态记录，不能作为已发布能力推荐。 [sglang #49430 `96eb498eaa`](https://github.com/sgl-project/sglang/pull/49430) · [sglang #50118 `7b790e1c5f`](https://github.com/sgl-project/sglang/pull/50118)

### SGLang 修复 HiCache 跨rank失败后的键分歧及 Mamba 陈旧COW来源

**11-12 16:58–11-13 05:10，`main`默认分支。** trailing-window sidecar的真实storage key在scheduler一致命中时就同步，不再等各rank IO线程读成功才替换placeholder。这样某rank KV短读时，各rank会失效同一批sidecar键，后续backup ack计数不再永久分歧。未准入请求再次match且节点device state已被驱逐时，清空旧 `mamba_cow_src_index`，避免从复用槽复制其他请求state。建议考虑采用并注入单rank短读和队列等待期间驱逐。 [sglang #50056 `b6f7e75204`](https://github.com/sgl-project/sglang/pull/50056) · [sglang #50009 `cabb43afd5`](https://github.com/sgl-project/sglang/pull/50009)

### SGLang 的 DFlash draft KV 改为每请求ring并接通P/D传输

**11-12 11:09–11-12 18:20，`main`默认分支。** 全sliding MLA草稿只需要最后窗口，Kimi-K3.1 DFlash2把每目标slot草稿KV改为每请求window+block环，作者共置配置可增加约15%目标KV预算。后续提交把ring通过Mooncake从prefill传到decode。统一内存、非全sliding等配置仍受准入条件约束。candidate top-k另依据目标lm_head自身的vocab分片group gather，避免复制head重复候选或DP attention仅聚到局部词表。建议复现ring wrap、P/D迁移与非标准lm_head布局。 [sglang #49664 `cb79602a7e`](https://github.com/sgl-project/sglang/pull/49664) · [sglang #49665 `406b7275bb`](https://github.com/sgl-project/sglang/pull/49665) · [sglang #47267 `353cd07596`](https://github.com/sgl-project/sglang/pull/47267)

### SGLang 的 Kimi-K3.1 HiSparse与低开销DCP路径回归main

**11-12 10:46–11-12 16:57，`main`默认分支。** CUDA PD decode可为K3.1的DSA层开启HiSparse并保留KDA state在GPU，DFLASH block最多8。限定flashmla_kv，DCP、decode radix cache及ROCm组合仍拒绝。HiSparse对未验证投机算法启动时报错，随后加入已验证DFLASH。独立DCP路径支持FlashMLA原生head数和预计算schedule，breakable prefill graph可回放prefix hit。FlashInfer Prims-TS D576 FP8 sparse MLA是另一个opt-in后端，依赖0.7.4且默认不变。这些路径的限制不能互相拼接成全组合支持。建议深入阅读各准入判据，再按具体组合复现。 [sglang #50005 `56005a53a5`](https://github.com/sgl-project/sglang/pull/50005) · [sglang #48846 `4464374072`](https://github.com/sgl-project/sglang/pull/48846) · [sglang #49980 `175ab80201`](https://github.com/sgl-project/sglang/pull/49980) · [sglang #49943 `74cb973d78`](https://github.com/sgl-project/sglang/pull/49943) · [sglang #48602 `c7f3e2bdfd`](https://github.com/sgl-project/sglang/pull/48602)

### SGLang 按工作负载调整 GLM state/KV 配额及短prefill SP

**11-12 17:18–11-12 19:06，`main`默认分支。** GLM-5.3-Flash的默认Mamba/KDA份额0.9→0.25，并在支持组合中选lazy tracking，为多轮agent保留更大KV工作集。高吞吐短请求cookbook仍显式固定0.9，较小state池会限制并发。`--layernorm-sp-min-tokens` 允许小extend走复制路径，阈值默认0维持原行为，交叉点与部署通信有关。P/D bootstrap另按page-rounding、handoff与alignment预留严格限制prompt，防接近pool容量的请求永远NO_TOKEN堵队首。建议分长会话和短并发两种流量复现配额，避免直接照搬默认。 [sglang #50018 `36027fd152`](https://github.com/sgl-project/sglang/pull/50018) · [sglang #48947 `a8ff994853`](https://github.com/sgl-project/sglang/pull/48947) · [sglang #49669 `fcb92bfda3`](https://github.com/sgl-project/sglang/pull/49669)

### SGLang 新增磁盘专家层，容量收益以同步IO等待为代价

**11-12 10:27，`main`默认分支。** `--paged-experts-enable-disk` 在GPU expert slots和pinned host rows后增加磁盘层，整层专家留在cache文件，router选到磁盘专家时读入host再copy至GPU，captured decode也会等待读完成。原本host RAM放不下的模型因此可启动。这是容量路径，原配置不能启动就不存在公平的“此前吞吐”基线。建议继续观察并复现尾延迟、磁盘IO和专家热点。不能把能运行等同适合高吞吐服务。 [sglang #49195 `88a44c1adb`](https://github.com/sgl-project/sglang/pull/49195)

### SGLang 分块门控后台 all-gather，并暴露外部linker对象布局

**11-12 10:58–11-12 15:03，`main`默认分支。** 后台multicast all-gather分成有界copy chunks，前台通信开始时暂停尚未发出的块，已发块完成后再恢复，保留整块输出分配。每rank直接写最终rank-major位置，只用一对entry/exit barrier。该设计调度NVLink/copy-engine竞争，不能宣称降低最终output显存。direct Mooncake linker现可通过HiCache L3 layout接口给外部router描述对象键，支持其pin会话页等决策。建议深入阅读完成与恢复边界，并用前台collective尾延迟验证。 [sglang #45947 `725b98b114`](https://github.com/sgl-project/sglang/pull/45947) · [sglang #49434 `2d7874b42f`](https://github.com/sgl-project/sglang/pull/49434)

### SGLang 优化投机采样、路由元数据与 NVFP4 草稿

**11-12 09:35–11-12 15:36，`main`默认分支。** Gumbel-max proposal把clamp/divide/argmax合为一次遍历，保留NaN等语义。小batch大词表的最终rejection sample按vocabulary tiles并行，恢复全局CDF并取最小匹配index。Kimi-K3路由元数据从最后一个小routing CTA搬到独立1024线程PDL kernel。显式NVFP4 MTP草稿请求现可从ModelOpt混合精度的block-FP8专家在线转换，dense SiLU×mul量化融合扩到Llama及compressed-tensors W4A4。建议按目标词表、batch和导出格式复现，区别kernel减少与端到端提升。 [sglang #48367 `d46d9adff7`](https://github.com/sgl-project/sglang/pull/48367) · [sglang #48949 `93dbe651f0`](https://github.com/sgl-project/sglang/pull/48949) · [sglang #49923 `bad3dd5100`](https://github.com/sgl-project/sglang/pull/49923) · [sglang #48295 `46aa131b7b`](https://github.com/sgl-project/sglang/pull/48295) · [sglang #48263 `ed037608a6`](https://github.com/sgl-project/sglang/pull/48263)

### SGLang 的 XPU GDN 投机与 H3 近似cache有明确适用边界

**11-12 10:06–11-12 13:52，`main`默认分支。** XPU用共享GDN backend和独立SYCL conv支持NEXTN/NGRAM，作者单Arc B60、Qwen3.5-9B、1024输入/512输出、batch1的NEXTN top1 TPOT为40.97→33.09ms，draft额外权重会降低可容纳输入。MiniMax-H3的quality=high改为固定Cache-DiT schedule，50步中24步真跑DiT，前15步全算。只在4×H200 T2VA、1344×768、124帧等限定配置用3prompt单seed比较。建议继续观察质量稳定性，不能把缓存重用称无损或混用H3-Image6步结果。 [sglang #44017 `46dc69747c`](https://github.com/sgl-project/sglang/pull/44017) · [sglang #49762 `f92ff40b13`](https://github.com/sgl-project/sglang/pull/49762)

### FlashInfer 为 BF16 MegaMoE 增加显式 batch-invariant 算术契约

**11-13 07:45，`main`默认分支。** SM100/103/107的opt-in模式固定K16累加、SwiGLU及slot顺序finalize，同时对显式knobs、autotune与cached plans校验同一算术契约。routing weight提前/延后应用仍属不同数值类。新增架构算术fingerprint与有序trace的Lean模型，不能从形式模型直接推断整个分布式程序已形式验证。建议深入阅读并考虑复现：需要训推bitwise对齐时，验证batch、row位置、plan选择与routing-weight位置，而不是只打开deterministic开关。 [flashinfer #9417 `60bd3e6e0c`](https://github.com/flashinfer-ai/flashinfer/pull/9417)

### FlashInfer 修复三类容易静默影响结果的低精度边界

**11-13 06:53–11-13 06:55，`main`默认分支。** packed KDA prefill的尾chunk会TMA读到下一序列V，NaN/inf乘零仍可污染前一序列。补丁只清尾部无效V。grouped MXFP8对E8M0 scale byte=0的非零极小BF16块改用正确倒数，避免量化为全零。DeepSeek路由sigmoid从tanh形式改为exp形式，修复约小于-18的FP32 logits全部舍入成0而产生零routing weights。作者将前两项定义为正确性修复且基本速度持平。建议考虑采用并增加packed非有限值、极小值与负logit回归。 [flashinfer #9407 `ab3e738bd4`](https://github.com/flashinfer-ai/flashinfer/pull/9407) · [flashinfer #9416 `ee59cd5d98`](https://github.com/flashinfer-ai/flashinfer/pull/9416) · [flashinfer #9393 `e4ad8d4a19`](https://github.com/flashinfer-ai/flashinfer/pull/9393)

### FlashInfer 收紧 PDL 发布、TMEM 读取与释放时序

**11-12 09:30–11-13 06:56，`main`默认分支。** TRT-LLM-gen routed-MoE在trigger前完成生产者写入，consumer先grid dependency wait再读。DeepSeek路由的wait补上compiler memory clobber，Muse Spark小行数norm在等待后再读分支。MQA logits在所有math warp完成TMEM load后才由warp0释放，BF16 rank-major MegaMoE另修每warp自身TMEM lane quarter寻址。PDL关闭并不能代替所有独立TMEM修复。建议深入阅读并考虑复现反复CUDA Graph replay、毒化workspace和多warp压力。部分正确性路径会略慢，不能只按benchmark回退。 [flashinfer #8610 `05e76524d0`](https://github.com/flashinfer-ai/flashinfer/pull/8610) · [flashinfer #9341 `60794ce3b2`](https://github.com/flashinfer-ai/flashinfer/pull/9341) · [flashinfer #9359 `0f2d52c265`](https://github.com/flashinfer-ai/flashinfer/pull/9359) · [flashinfer #8855 `d4a76c440c`](https://github.com/flashinfer-ai/flashinfer/pull/8855) · [flashinfer #6890 `b2ba83286a`](https://github.com/flashinfer-ai/flashinfer/pull/6890)

### FlashInfer 的路由与 paged-attention 接口不再静默接受错误条件

**11-12 09:44–11-13 05:25，`main`默认分支。** 单group多expert路由给未写的intermediate top-k槽填sentinel，防读到旧shared memory。零token直接no-op，非空launch检查CUDA返回值，避免错误延迟到无关算子才出现。标准FA2/FA3 paged adapter原先接收但忽略sinks，现在启动前拒绝。支持sink的TRT/custom JIT路径不受此限制。建议考虑采用，并检查集成端是否依赖曾被静默忽略的参数。 [flashinfer #9344 `96236732c8`](https://github.com/flashinfer-ai/flashinfer/pull/9344) · [flashinfer #9345 `c2f207129b`](https://github.com/flashinfer-ai/flashinfer/pull/9345) · [flashinfer #9410 `236a0278bb`](https://github.com/flashinfer-ai/flashinfer/pull/9410)

### FlashInfer 扩展 NVFP4 sparse-MLA prefill并显式约束 FP8宽度

**11-13 06:55–11-13 06:56，`main`默认分支。** SM100/103 Cake sparse-MLA prefill新增预打包NVFP4 query路径，调用方须提供uint8 packed query或先用专用packer，不能把BF16输入自动视作该prefill。FP8 route对每item最多1152 sparse slots加统一限制，避免C128层长context超过shared buffer容量。某些旧拒绝head/row形状新增M64调度。MSA BF16-Q/E4M3-KV prefill只访问被选KV blocks，作者GB300的8-token chunk+128K prefix kernel约快10倍，较大query收益远小且非服务吞吐。建议考虑复现精度、索引宽度与长prefix小chunk情形。 [flashinfer #6851 `b86beb000b`](https://github.com/flashinfer-ai/flashinfer/pull/6851) · [flashinfer #9049 `ca4b8578cc`](https://github.com/flashinfer-ai/flashinfer/pull/9049) · [flashinfer #9394 `976e7daad5`](https://github.com/flashinfer-ai/flashinfer/pull/9394)

### FlashInfer 为逐token逐head FP8 KV增加原生写侧算子

**11-13 01:18，`main`默认分支。** `fp8_token_head_quantize_append_paged_kv_cache_with_slot_mapping` 对每token/head在device算scale并把量化值与FP32 scale写入同一对齐slot，补齐FA2对应读侧。支持FP16/BF16输入、E4M3/E5M2、NHD/HND和64/128/256/512维head，负值/越界slot忽略。每slot是head_dim个FP8字节加16字节scale/padding，而非零元数据开销的8bit缓存。建议深入阅读写读契约并复现图捕获、padding和slot mapping。 [flashinfer #6863 `bbeef7b8e0`](https://github.com/flashinfer-ai/flashinfer/pull/6863)

### FlashInfer 的 SM120 MegaGate仍为实验入口

**11-13 05:13，`main`默认分支。** CuTe DSL实现把BF16 router GEMM、sqrt-softplus、带bias top-k、无bias归一化与logical→physical expert映射合为一个kernel，目标为DeepSeek-V4/V4.1 Flash top6形状。SM12x分派到新实现，SM10x继续用DeepGEMM。代码使用persistent CTA组、FP32 split-K scratch及可自复位counter支持graph replay，不能视作SM120支持全部MegaMoE。建议继续观察并按目标router shape复现。 [flashinfer #8479 `0df46f4a66`](https://github.com/flashinfer-ai/flashinfer/pull/8479)

### TensorRT-LLM 把投机 replay state 纳入按请求分配的KV区域

**11-12 21:40，`main`默认分支。** 原先Mamba2/GDN的replay输入与计数是按SSM槽索引的旁路tensor，V2 suspend/resume搬页后状态丢失。KDA又把replay bytes跟每份prefix snapshot复制。新region按request持有replay，随请求迁移并避免在每快照携带，分开运行请求数与snapshot槽数。建议深入阅读并考虑复现包含未提交draft的暂停/恢复、prefix checkpoint和slot复用，这是状态正确性与内存所有权变化。 [TensorRT-LLM #21324 `e0b8a46aba`](https://github.com/NVIDIA/TensorRT-LLM/pull/21324)

### TensorRT-LLM 冷页压缩可保留首尾高精度KV

**11-12 16:34，`main`默认分支。** `quantization_for_cold_page` 新增 `mixed_precision_first_n/last_n`：中间历史在host/disk存NVFP4，首N与最近N token保留runtime KV精度。高精度页通过V2 high layers保留并直接由block offsets寻址，非所有token都先解压成同精度再attention。建议考虑复现实际长context质量与分页内存。未给出完整任务/硬件同口径质量曲线时不能称压缩无损。 [TensorRT-LLM #20197 `287f69ab6d`](https://github.com/NVIDIA/TensorRT-LLM/pull/20197)

### TensorRT-LLM 修复 MXFP4输入scale和两个workspace容量错误

**11-12 17:54–11-13 00:48，`main`默认分支。** W4A8 MXFP4/FP8 MoE用原始FC1 input scale量化输入，区别于GEMM1输出所需FC1/FC2 scale比，避免DEP/DTP的错误幅度。MNNVL fused RMSNorm shared memory改由kernel真实需要推导，并对线程/驻留形状加guard。GQA segmented-decode workspace纳入FlashInfer请求的LSE softmax stats，修复Qwen3.5-397B TP4高batch超过固定32MiB。建议受影响量化并行配置考虑采用，分别核对输出与高并发warmup。 [TensorRT-LLM #22106 `e275e3ad2f`](https://github.com/NVIDIA/TensorRT-LLM/pull/22106) · [TensorRT-LLM #21785 `b38f9015d4`](https://github.com/NVIDIA/TensorRT-LLM/pull/21785) · [TensorRT-LLM #21863 `8151a93a19`](https://github.com/NVIDIA/TensorRT-LLM/pull/21863)

### TensorRT-LLM 清理MoE与NIXL资源时区分停止访问、释放池和注销

**11-12 17:30–11-13 02:45，`main`默认分支。** executor shutdown主动释放进程级cached MoE workspace，避免下一模型估算KV前仍留前模型约28GiB/rank临时区。transceiver先prepare/drain，资源manager释放pool，再完成NIXL注销。失败保留可重试状态。transfer status显式release释放GIL，但review指出失败后的C++析构重试仍缺锁竞争验证。建议考虑复现多模型连续启动和失败清理。不能仅凭CPU mock断言GPU/NIXL生命周期已全面解决。 [TensorRT-LLM #20840 `134d9a2b4c`](https://github.com/NVIDIA/TensorRT-LLM/pull/20840) · [TensorRT-LLM #21784 `fe3d346681`](https://github.com/NVIDIA/TensorRT-LLM/pull/21784) · [TensorRT-LLM #21612 `fd466b3c09`](https://github.com/NVIDIA/TensorRT-LLM/pull/21612)

### TensorRT-LLM 处理超时后的迟到P/D消息并释放FlashMLA调用的GIL

**11-12 10:21–11-12 21:16，`main`默认分支。** main回移release/1.3的Python V2 transceiver修复：清理过的TxSession收到迟到REQUEST_DATA应答FAILED，失败receive等全部writer回报或有限超时后才回收页，避免generation瓶颈下永久等待及早复用。独立runtime补丁给FlashMLA pybind调用释放GIL，防同stream host callback拿不到GIL而互等，并为attention-DP dummy分配配对draft KV。建议考虑复现拥塞、context端先超时与取消交错。新模型bring-up发现的修复不等于该模型runtime已注册。 [TensorRT-LLM #20943 `fd57da4e39`](https://github.com/NVIDIA/TensorRT-LLM/pull/20943) · [TensorRT-LLM #21788 `e340e88349`](https://github.com/NVIDIA/TensorRT-LLM/pull/21788)

### TensorRT-LLM 减少长多轮prompt重复分词，并跨rank共享JIT构建

**11-12 17:02–11-12 18:16，`main`默认分支。** prefix tokenization cache每256token保存checkpoint，允许历史尾部改写时从共同prefix内部拼接，且验证重分词区间重现缓存ids。`TLLM_JIT_SHARE_DIR` 则让DeepGEMM/CuTe warmup kernel由一个进程claim编译并发布，其他rank复用。键包含source、编译参数/工具与headers内容，目录必须全launch可见。建议考虑复现100K级多轮前端CPU成本与多节点冷启动，分别看token id一致性和编译失败传播，收益不能外推为decode kernel加速。 [TensorRT-LLM #21308 `7473ce2de7`](https://github.com/NVIDIA/TensorRT-LLM/pull/21308) · [TensorRT-LLM #21791 `737bf56106`](https://github.com/NVIDIA/TensorRT-LLM/pull/21791)

### TensorRT-LLM 的 Beam新提交仅落在组件边界

**11-13 04:36，`main`默认分支。** 新增StairCase/ModelingV2算子、dense/MoE块、6层BF16与直接构造的52层NVFP4组件。后者为4×GB300、attention TP4、MoE TP4/EP1。PR明确不注册Beam runtime，不提供可运行的 `LLM()` 或 `trtllm-serve`。route、engine/checkpoint adapter与整模型accuracy gate在后续#21473。建议继续观察，当前可研究组件实现，不能发布“TRT-LLM已支持Beam服务”的工程结论。 [TensorRT-LLM #21471 `08b8c22500`](https://github.com/NVIDIA/TensorRT-LLM/pull/21471)

### TensorRT-LLM 扩展 GLM-5.2 特化路径并修复 attention-DP embedding

**11-12 09:44–11-12 21:24，`main`默认分支。** GLM-5.2 NVFP4的SM103、4×B300/GB300 attention-TP4/expert-EP4特化路径由MTP1扩到MTP3/5，并为64K–160K低batch索引与projection提供专门kernel，不能外推所有模型/GPU。Qwen3 attention-DP且lm_head不TP时embedding改为同样的完整副本，满足tied权重布局。Qwen3.8-Flash-Next GPU FP8 PLE通过uint8 view清零非本rank行，避免不存在的FP8 masked_fill kernel。建议按实际特化target和ADP配置考虑复现。 [TensorRT-LLM #21695 `9da7960602`](https://github.com/NVIDIA/TensorRT-LLM/pull/21695) · [TensorRT-LLM #20089 `bde3ed9bf7`](https://github.com/NVIDIA/TensorRT-LLM/pull/20089) · [TensorRT-LLM #21346 `ad43776e8f`](https://github.com/NVIDIA/TensorRT-LLM/pull/21346)

### Attention Gym 0.0.20固定deterministic调优配置并改变KDA黄金数值

**11-12 12:21，正式发布。** 新版将多个linear/sparse入口的 `autotune=` 迁到 `kernel_options={"autotune":...}`。deterministic algorithms或显式autotune=False时用固定配置，升级后的旧tuned winner会重调一次。KDA BF16 forward的rebase/FTZ及陡gate backward修复会改变输出和梯度bits，发布页要求更新黄金数值。建议考虑采用前先迁移配置并重跑数值对照，不能把固定调度等同升级前bitwise不变。 [attention-gym v0.0.20 `7adfbadb06`](https://github.com/meta-pytorch/attention-gym/releases/tag/v0.0.20)

### Attention Gym用warpdrive重写frost前反向并替换vendored cuDNN包

**11-13 07:15–11-13 07:15，`main`默认分支。** GDN/KDA forward、反向与recompute迁到warpdrive roles/channels，最后switch-over才让 `cudnn` backend实际改用frost。移除vendored cudnn_fe并沿用backend/driver API，CuTeDSL最低要求降到4.6.2的linear依赖。随后优化GDN链式recompute的单向TMEM handoff与KDA per-chunk关键路径，避免新抽象的同步簿记退步。建议深入阅读迁移后的所有权与barrier模型并复现长链形状。几个先导PR合入feature branch不能单独代表最终backend切换。 [attention-gym #840 `fec93f4e49`](https://github.com/meta-pytorch/attention-gym/pull/840) · [attention-gym #841 `6ebe162912`](https://github.com/meta-pytorch/attention-gym/pull/841) · [attention-gym #842 `52acf1594a`](https://github.com/meta-pytorch/attention-gym/pull/842) · [attention-gym #843 `bef316ec7c`](https://github.com/meta-pytorch/attention-gym/pull/843) · [attention-gym #844 `3a05f13f1d`](https://github.com/meta-pytorch/attention-gym/pull/844) · [attention-gym #851 `19f9b20034`](https://github.com/meta-pytorch/attention-gym/pull/851) · [attention-gym #852 `e18ad94277`](https://github.com/meta-pytorch/attention-gym/pull/852)

### Attention Gym修复packed metadata对齐、跨device调用并披露gate误差边界

**11-13 06:47–11-13 09:01，`main`默认分支。** KDA backward的int32 metadata标注实际4字节而非128字节对齐，使合法连续slice无需额外copy。lazy CUDA调用按operand device设guard，不再误用线程current device。文档明确FP32 chunk gate前缀和的分布敏感性：普通gate中孤立约1e5或更大负值可吞掉后续增量，代码不clamp、不主机校验。`ReplayState.slot_layout` 另公开六份replay buffer的shape/dtype供服务池预分配，fused kernel仍要求K=V=128。建议考虑采用修复并按真实gate分布测输出、state和梯度。 [attention-gym #865 `0c784ae11c`](https://github.com/meta-pytorch/attention-gym/pull/865) · [attention-gym #866 `4214fc89d5`](https://github.com/meta-pytorch/attention-gym/pull/866) · [attention-gym #867 `5261a03e26`](https://github.com/meta-pytorch/attention-gym/pull/867) · [attention-gym #862 `789314aba6`](https://github.com/meta-pytorch/attention-gym/pull/862)

### PyTorch默认构建NCCL EP，并提供显式cuDNN Python SDPA入口

**11-13 04:33–11-13 08:24，`main`默认分支。** CUDA与NCCL构建满足条件时 `USE_NCCL_EP` 默认ON，可显式OFF。这改变可用构件，不能保证所有wheel和平台都已发布TokenSwitchNCCL。`torch.nn.attention.activate_flash_attention_impl("CUDNN")` 可手动注册nvidia-cudnn-frontend的Python SDPA，仍为beta且需另装包。现有C++ cuDNN默认不会因此自动切换。建议继续观察wheel/版本落地，并在研究Python开放kernel时显式测选择与fallback。 [pytorch #204363 `21e85424b9`](https://github.com/pytorch/pytorch/commit/21e85424b9aaa3854ab1b609118816fac4af19a9) · [pytorch #195981 `7cd1066602`](https://github.com/pytorch/pytorch/commit/7cd1066602ad8fb7c334b37a0a3751543459b235)

### PyTorch 为低精度分片norm升精度，并融合异dtype foreach更新

**11-12 12:43–11-12 13:06，`main`默认分支。** DTensor `_NormPartial` 在all-reduce前用FP32做FP16/BF16 norm的幂变换、最后转回原dtype，避免FP16 shard norm=256平方溢出后clip_grad_norm_把全部梯度置零。CUDA in-place `_foreach_add_/_sub_` 对每list内部同dtype、两list分别为FP32/FP16/BF16的组合新增multi-tensor融合路径，减少优化器一tensor一launch。建议考虑复现目标optimizer与分片clip。norm作者本轮只跑LocalTensor多项测试，不能写成多进程完整验收。 [pytorch #203915 `e691a5ea7e`](https://github.com/pytorch/pytorch/commit/e691a5ea7ece349b3dc5eb80e7f4ef121e307096) · [pytorch #203910 `1e562eecd7`](https://github.com/pytorch/pytorch/commit/1e562eecd7cc4b320e029dc11df99981233c7aca)

### PyTorch 修复编译缓存的alias与custom-JVP正确性

**11-12 09:41–11-13 01:00，`main`默认分支。** AOTAutograd在首次不共享storage但后续变成重叠view的可变输入间安装guard，避免复用缺synthetic-base处理的旧图。Dynamo遇到forward AD的custom jvp改为graph break并加dual-level guard，避免静默丢掉自定义切线。CUDA graph-tree wrapper的mutation判断使用创建时确定的tree规则，不再被之后配置翻转误判。建议考虑采用并复现静态shape alias变化、进入dual_level及配置切换的缓存复用路径。 [pytorch #185891 `d880a0f53a`](https://github.com/pytorch/pytorch/commit/d880a0f53a44f4cda5e15926cf542fe164125437) · [pytorch #193277 `1e3e90ec55`](https://github.com/pytorch/pytorch/commit/1e3e90ec552070dba85c79e5929834bc120f096d) · [pytorch #202858 `8a304b0f1f`](https://github.com/pytorch/pytorch/commit/8a304b0f1f9ceca0c76c28c15b3455a76c7ec97d)

### PyTorch的profiler按session启用GPU trace，动态TMA可在host构描述符

**11-12 10:51–11-13 03:10，`main`默认分支。** Kineto区分设备初始化与当前session请求，修复第一轮CPU-only或GPU trace设置对后续session永久粘住。Inductor host-side TMA descriptor可从launch的动态size/stride参数构建，并在descriptor位置缓存少量形状，避免每CTA初始化descriptor。仅能在host求值的表达式走此路。CUDA Event析构也避免无谓创建原device primary context。建议考虑采用，并用CPU→GPU→CPU profiler序列与动态batch小GEMM检查trace完整性及冷启动副作用。 [pytorch #202625 `de2767670e`](https://github.com/pytorch/pytorch/commit/de2767670e4e4ccd03fbfc88ef8c43f9e172e36a) · [pytorch #202612 `18f2eae1b3`](https://github.com/pytorch/pytorch/commit/18f2eae1b30a6057d56ad2585a89fb3b965c3f43) · [pytorch #204542 `4b71f179c9`](https://github.com/pytorch/pytorch/commit/4b71f179c99eb7555e48ea95a1a7545ebe5c302a)

### Triton FPSan新增表达式barrier，指针对齐分析保留scalar cast信息

**11-12 21:41–11-13 05:23，`main`默认分支。** `tl.fpsan.barrier` 与 `homomorphic_cast` 允许在选定中间量检测重结合、FMA或过早窄化。无FPSan时barrier不引入实际计算，不能当数值误差上界。scalar int↔pointer cast保留字节对齐，修复若干SGLang gather被降成窄load的机会。另撤回前一天warp-specialized worker dispatch改动，维护者认为原行为不构成所担心的UB且新方案让部分kernel变慢。建议深入阅读并以固定编译器baseline重测，不能把已撤回的调度当新默认。 [triton #12725 `69524224de`](https://github.com/triton-lang/triton/pull/12725) · [triton #12711 `8a439de83a`](https://github.com/triton-lang/triton/pull/12711) · [triton #12720 `5b4f371434`](https://github.com/triton-lang/triton/pull/12720)

### Dynamo把worker选择、KV hint与global pool选择分成插件阶段

**11-12 11:51–11-13 06:46，`main`默认分支。** router新增hint-policy角色，把worker选择决定的目标与copy source传给engine hint。KVCR选择器与 `kv_fetch` 分别注册，依赖KV fetch的部署应显式选kvcr或配置hints。Global Router可opt-in把pool视为router-core worker用round-robin插件。默认选择路径不变。另有opt-in resident block计数输入描述缓存占用而非只看active load，尚未提供session liveness。建议深入阅读插件配置和commitments边界，再用真实KV复制成本验证策略。 [dynamo #17359 `e629e8d379`](https://github.com/ai-dynamo/dynamo/pull/17359) · [dynamo #17647 `53eec1d599`](https://github.com/ai-dynamo/dynamo/pull/17647) · [dynamo #17360 `087f8d0fdb`](https://github.com/ai-dynamo/dynamo/pull/17360) · [dynamo #17597 `a2463598e4`](https://github.com/ai-dynamo/dynamo/pull/17597) · [dynamo #16078 `a654e3bb1d`](https://github.com/ai-dynamo/dynamo/pull/16078)

### Dynamo修复worker删除事件顺序，并启用Pareto部署搜索

**11-12 12:22–11-13 02:20，`main`默认分支。** KvIndexer的worker/rank删除进入同一FIFO mutation队列，避免较早Stored事件在删除后“复活”旧缓存并虚增prefix credit。聚合prompt_logprobs请求保守不给cache credit，因为评分会重算。GLM-5.3-Flash图片planner读取checkpoint token几何、布局不支持时回退文字prefix路由。DGDR v1beta2的Pareto模式接通CRD、lowering、搜索与非支配front发布，区别于单目标Optimize。建议考虑复现缩容、评分请求与多模态路由，Pareto搜索结果仍依赖模拟器和负载模型。 [dynamo #16990 `164358ac55`](https://github.com/ai-dynamo/dynamo/pull/16990) · [dynamo #17441 `bffd7b393e`](https://github.com/ai-dynamo/dynamo/pull/17441) · [dynamo #17121 `02be33b210`](https://github.com/ai-dynamo/dynamo/pull/17121) · [dynamo #17611 `62c825d89d`](https://github.com/ai-dynamo/dynamo/pull/17611)

### LMCache区分content fingerprint去重与真正跳过KV写回

**11-12 12:00–11-12 19:49，`dev`默认分支。** CacheBlend非prefix复用的document若换位置，rolling-prefix key会变，原fingerprint去重仍写KV。新 `--enable-dedup-content-kv` 只跳过完全落在已加载content范围内的chunk写回，部分覆盖和miss照常保存。SGLang chunked prefill的早期失败store会保留为下次callback重试范围，不被更晚pending store推进cursor掩盖，未新增自主retry scheduler。隔离LRU另不再复制整tenant key表才选少量victim。建议考虑复现带宽受限L2的读写竞争、失败callback和租户隔离。 [LMCache #6521 `82cd869063`](https://github.com/LMCache/LMCache/pull/6521) · [LMCache #6404 `2c5c2917ff`](https://github.com/LMCache/LMCache/pull/6404) · [LMCache #6393 `3aefae1d57`](https://github.com/LMCache/LMCache/pull/6393)

### NVSHMEM默认分支公开CUDA13.5 CFT host RMA与显式CFT/TMA路由

**11-13 00:38–11-13 00:42，`devel`默认分支。** 本轮11月12日一次默认分支提交批的作者日期多在10月。这里记录当前可定位代码增量，未核到新正式release。host fabric-copy支持RMA、scalar staging及signal SET/ADD，deferred fence用CUDA events保留逐PE顺序。需要CFT build与CUDA13.5条件。device显式CFT/TMA route只支持block scope，要求16字节对齐与handle transaction粒度，缺capability返回NOT_SUPPORTED而非偷偷换路。NBI get/put-signal仍主动完成以保证payload先于signal。建议深入阅读完成语义并继续观察发布/实测，不宣称现有NVSHMEM3.8包已包含。 [NVSHMEM ed02fd097f `ed02fd097f`](https://github.com/NVIDIA/nvshmem/commit/ed02fd097f3d5dccd9f6ac251bc7ae9527e7c4de) · [NVSHMEM 368df40859 `368df40859`](https://github.com/NVIDIA/nvshmem/commit/368df40859ecf4fc52fae6b131b658986af52a11) · [NVSHMEM 9e78c5c3db `9e78c5c3db`](https://github.com/NVIDIA/nvshmem/commit/9e78c5c3dbd389a6370a3a935e7bc82bcdbb603e) · [NVSHMEM 72ade035b3 `72ade035b3`](https://github.com/NVIDIA/nvshmem/commit/72ade035b3c9fdc145116cb932844385cb4bcec5)

### NVSHMEM修复CFT handle完成计数、复用与quiet覆盖

**11-13 00:39–11-13 00:46，`devel`默认分支。** fabric completion的expect计数在issue前建立，避免complete_tx先于expect_tx。NBI put在返回并复用shared barriers前等待consumer发布pending handle。quiet/release覆盖参与warp及已拥有slot，release_smem先等待issuer，collective的CFT dispatch按构建与硬件capability守卫。建议考虑复现反复NBI、不同threadgroup scope和slot复用，源代码本轮可核，未做设备压测。 [NVSHMEM c5a5610c1a `c5a5610c1a`](https://github.com/NVIDIA/nvshmem/commit/c5a5610c1aabffa5f310c883ba51b5483b764f3a) · [NVSHMEM adfd6a3305 `adfd6a3305`](https://github.com/NVIDIA/nvshmem/commit/adfd6a3305d87038b3fd2de55c380bb79f4b54cb) · [NVSHMEM 9e5543f5a7 `9e5543f5a7`](https://github.com/NVIDIA/nvshmem/commit/9e5543f5a7267393c466947e0b7f0eb2a9688585) · [NVSHMEM 06504efa02 `06504efa02`](https://github.com/NVIDIA/nvshmem/commit/06504efa025841a0dcf1848e7f12d0d79ad712dd)

### NVSHMEM支持不均匀节点PE拓扑，并修复多NIC原子与退出路径

**11-13 00:39–11-13 00:47，`devel`默认分支。** 初始化移除每node PE数相同的限制，保留任意PE map的共享clique与已知membership的team split。完整非均匀拓扑仍应结合整批相关提交测试。IBGDA/GPUNetIO的self-target AMO按稳定地址选NIC并drain所有loopback QP。另初始化QP-sharing标志、避免重复NIC映射double free，UCX packed-rkey增加上限与边界检查，proxy失败退出先停progress线程避免CUDA隐式等待。LL128也修正两处load地址。建议通信栈维护者深入阅读并定向复现，不将单点修复概括为任意混合集群支持。 [NVSHMEM a8aa742a3a `a8aa742a3a`](https://github.com/NVIDIA/nvshmem/commit/a8aa742a3a59b807f52c248a209b34d2550d9098) · [NVSHMEM 53d4ec1d82 `53d4ec1d82`](https://github.com/NVIDIA/nvshmem/commit/53d4ec1d82399b9e293249a81b15435926bca57e) · [NVSHMEM c141100ab7 `c141100ab7`](https://github.com/NVIDIA/nvshmem/commit/c141100ab7e3122935c01c088ad9a9f6f5343b49) · [NVSHMEM 4279506044 `4279506044`](https://github.com/NVIDIA/nvshmem/commit/4279506044118fb23d007403c6758c1f8dcb55d5) · [NVSHMEM 3981ad94f6 `3981ad94f6`](https://github.com/NVIDIA/nvshmem/commit/3981ad94f6b54b7976e3f35173a7edb1a7beb462) · [NVSHMEM 89743ea22c `89743ea22c`](https://github.com/NVIDIA/nvshmem/commit/89743ea22c58d6c88d93121f3413dfd45f06daf0) · [NVSHMEM 7f453ed5df `7f453ed5df`](https://github.com/NVIDIA/nvshmem/commit/7f453ed5dfaf1acf2af00b410604ad926bf7c092) · [NVSHMEM 7d3b735ac1 `7d3b735ac1`](https://github.com/NVIDIA/nvshmem/commit/7d3b735ac16e8b8add7aa7a72a6399ed8b311b49) · [NVSHMEM 5952535f32 `5952535f32`](https://github.com/NVIDIA/nvshmem/commit/5952535f32a624c5e6dd38fc31cb7726fb976045) · [NVSHMEM 444ab2a844 `444ab2a844`](https://github.com/NVIDIA/nvshmem/commit/444ab2a84424d53a61a730b1ba8f1f7b5cd54d85)

### TileLang修复昇腾全核同步调度与PTO内存证明

**11-12 11:27–11-12 16:33，`main`默认分支。** mode0全核cross-core sync令生成函数使用batch scheduling，避免多stream各占部分core后互等。local模式1/2/4不启用。PTO VMI以bits验证dtype-changing view容量并保留实际单元素footprint，integer-cast事实先验证可表示性且守住词法scope，避免把未执行分支的约束用于后续越界证明。显式SIMD Parallel保留原loop，整数atomic适配PTO signless类型。建议考虑复现自定义kernel的同步、多view与cast边界。验证绑定的PTOAS/CANN版本而非仅Python单测。 [tilelang #3614 `d2dbf20cb2`](https://github.com/tile-ai/tilelang/pull/3614) · [tilelang #3894 `558d53b491`](https://github.com/tile-ai/tilelang/pull/3894) · [tilelang #3793 `eb86343c6a`](https://github.com/tile-ai/tilelang/pull/3793) · [tilelang #3796 `ed72fdbdea`](https://github.com/tile-ai/tilelang/pull/3796) · [tilelang #3792 `aee965e5ed`](https://github.com/tile-ai/tilelang/pull/3792)

### ZML的Furiosa TCL路径接入block-128 FP8 SwiGLU专家

**11-12 20:47，`master`默认分支。** 仅plain E4M3FN block128权重与对应BF16/FP32 scales可选TCL，activations逐block量化、projection应用checkpoint scales，保留BF16 materialization与FP32 down/routing accumulation。EP按本地专家与all-reduce复用既有逻辑，小token走expert-stationary，大token保留blockwise。其他FP4/MXFP8等量化仍明确拒绝。建议继续观察目标硬件数值和性能证据，不能称Furiosa支持所有低精度MoE格式。 [zml #1053 `c40fa52d2d`](https://github.com/zml/zml/pull/1053)

### Miles 0.2.0正式打包多模态token-space RL，迁移边界披露充分

**11-12 17:37，正式发布。** 本次版本将多模态rollout/trainer统一media processor与token-space输入、native multi-LoRA及output-store等阶段性能力打包发布，并列出三类训练backend与NVIDIA/AMD/TPU路径。Trainium仍写landing soon。Muse Spark主镜像证据只是last4层4×H200 CI，完整模型旧成绩来自独立镜像。MiniMax-M3.1-Flash也仅4层验证。升级有Python3.11+、不可变AllConfig、默认prompt shuffle和torchtitan0.3 checkpoint不兼容等变化，不能照抄旧launcher。建议深入阅读release的逐模型“实际跑过什么”，按所需组合再考虑复现。 [miles v0.2.0 `8e419cc739`](https://github.com/radixark/miles/releases/tag/v0.2.0)

### vLLM Rubin技术报告：共享缓存与拓扑专用kernel共同影响吞吐口径

**2026-11-12，官方报告（日期未披露时分）。** Inferact/vLLM报告Vera Rubin NVL72的AgentX总吞吐，相比GB300的58倍统计包含cached input、uncached input和output，387 token/s是对应每用户速率，不能当58倍decode。prefill rank共享Mooncake DRAM/闲置HBM池及Engram HBM，HBM pool仍实验。28%是32K-token/rank DEP2 prefill-step改善，25% target-forward则在单4GPU节点模拟DEP32，非整架端到端对照。建议深入阅读负载、prefix hit与内存复用，复现时分别报告cache命中输入、未命中输入和输出吞吐。 [vLLM 技术报告 `2026-11-12`](https://vllm.ai/blog/2026-11-12-deepseek-v41-flash-vera-rubin)

### llama.cpp 为2–24列量化GEMM新增小batchCUDA kernel

**11-12 17:28–11-12 21:12，`master`默认分支。** 新 `mul_mat_q_few` 面向小batch与投机verify，读取权重后只解码一次并用于所有列的int8 tensor-core MMA，避免MMVQ逐列重复和大MMQ的低tile数/stream-k开销。支持若干Q/K/IQ格式，范围为Ampere到DGX Spark，不能泛指所有GGUF格式。NVFP4另把外部weight scale融进batched MMQ epilogue及stream-k fixup，省去中间量和MoE scale gather。建议考虑复现实际draft宽度与量化格式，并保留perplexity/输出验收。 [llama.cpp #31533 `0fb8f15770`](https://github.com/ggml-org/llama.cpp/pull/31533) · [llama.cpp #26048 `0d69255c52`](https://github.com/ggml-org/llama.cpp/pull/26048)

### llama.cpp 修复跨backend专家预取、空tile越界和SYCL大模型输出

**11-12 11:45–11-12 17:24，`master`默认分支。** 预取只读先前split的compute结果，避免当前copy callback尚未上传的新expert bank被下一CPU split提前读成旧值。不支持cached matmul的backend改走uncached。CUDA MMQ fixup在空MoE tile一致提前返回，防尾部ids只读越界。SYCL oneDNN FA cache改为context内持有并处理线程/失败复用，AOT补buffer size设置修复超过4GB模型生成乱码。建议相关混合backend与Intel部署考虑采用，不能仅凭启动成功判定修复已覆盖。 [llama.cpp #32100 `3bce8bd50c`](https://github.com/ggml-org/llama.cpp/pull/32100) · [llama.cpp #32067 `5d4c8eeb24`](https://github.com/ggml-org/llama.cpp/pull/32067) · [llama.cpp #31138 `09338b85e8`](https://github.com/ggml-org/llama.cpp/pull/31138) · [llama.cpp #31873 `dcbff21661`](https://github.com/ggml-org/llama.cpp/pull/31873)

### llama.cpp 扩展Intel Vulkan prefill与SpacemiT RVV GDN，取消工具调用可通知MCP

**11-12 18:19–11-13 00:19，`master`默认分支。** Intel专用FA prefill shader按query激活规模选择，启用范围为XE1 ARL-H、XE2/XE3。A770因coopmat数据不稳定仍关闭。SpacemiT F32 GatedDeltaNet由原scalar路径接RVV dot/outer/decay，expf仍scalar，作者A100/X100平台kernel验证不应误读为NVIDIA A100数据。服务端POST /tools把client断开/timeout传给transport，并发送MCP cancelled notification。远端是否及时停止仍由MCP server实现决定。建议按对应设备复现推理，并检查实际工具取消回执。 [llama.cpp #29357 `a963f65957`](https://github.com/ggml-org/llama.cpp/pull/29357) · [llama.cpp #31897 `a8575a3ef4`](https://github.com/ggml-org/llama.cpp/pull/31897) · [llama.cpp #32025 `d8c8cc4d9b`](https://github.com/ggml-org/llama.cpp/pull/32025)

<a id="hardware"></a>
## 硬件与软件协同

以下工程条目固定到所列提交。均为默认分支代码增量，不能等同于正式版已发布。测试数字来自维护者，本轮未在相应加速卡复跑。

### Transformer Engine 在重算路径复用 NVFP4 amax，同时修正 RHT 符号契约

融合 GroupedMLP 在反向会重建 FC2 输入，原先前向仍生成随后丢弃的列向量量化副本。新路径先判断重算，只保留所需方向，再复用前向 amax，省掉反向 Hadamard-amax 扫描。更关键的配套修复是把随机 Hadamard 的 sign mask 交给支持它的 cuDNN wrapper。旧 wrapper 不支持且 mask 非零时回退，避免把不同变换的 amax 用作量化尺度。维护者在 GB300、8 专家 NVFP4+RHT 配合匹配 cuDNN Frontend 时报告输出与多项梯度逐位匹配。已发布的 Frontend 1.31.0/1.32.3 缺少相应 wrapper 特征，重算测试仍可走 RMS 容差，不能据此宣称所有版本逐位一致。130B hybrid MoE 代理中 FC2 量化约 570→330 µs。这是融合调用的局部时间。建议低精度训练团队深入阅读并复现重算开关对照，匹配 cuDNN Frontend 版本。

[NVIDIA/TransformerEngine #3837 · 11-12 13:16 · d325253ead](https://github.com/NVIDIA/TransformerEngine/commit/d325253ead6908c716998b1478039f7952b7ba5a)

### Transformer Engine 统一 GEMM 就绪量化布局与 UE5M3 GEMM 入口

量化输出现在尊重 `optimize_for_gemm`，需要时先写紧凑 scale scratch 再批量 swizzle，复用输出时保持目标缓冲及图重放地址。NVFP4-UE5M3 GEMM 则接入 cuBLAS，删除原 Python 特殊分支。两项都改变量化器与 GEMM 的交接契约。不能仅凭精度名字相同假设 scale 排布相同。建议已有 NVFP4 自定义调用者核对量化输出布局与捕获重放，暂不把代码减少解释为端到端提速。

[NVIDIA/TransformerEngine #3817 · 11-12 22:45 · bb3be5235f](https://github.com/NVIDIA/TransformerEngine/commit/bb3be5235f8a1c8eb2b2a45e779b7953d1adec79) · [NVIDIA/TransformerEngine #3827 · 11-12 09:23 · 42ec40e951](https://github.com/NVIDIA/TransformerEngine/commit/42ec40e9512d0f39b13145c1d2d53976db6d504a)

### cuDNN Frontend 修复非紧凑 THD 物理起点，扩展 SM100 DSA 训练头数

D512 backward 旧路径按逻辑长度前缀定位，带间隙 THD 会读入 padding 并写坏间隙。新实现为九个输入输出端口分别物化物理起点，覆盖 delta、GQA 梯度折叠与空序列保护。维护者报告 B300 的 BF16 紧凑路径约有 1.3% 额外成本。另将 H64 kernel 分头窗口复用于 H96/H128，补齐 D512/D576、top-k 2048 等 SM100 前向及训练组合。小 top-k H128 仍走专用 kernel，不同路径舍入可不同。建议变长训练先验证有间隙和空尾序列，再评估新增配置。

[NVIDIA/cudnn-frontend #2525 · 11-13 08:18 · fed09b0fa6](https://github.com/NVIDIA/cudnn-frontend/commit/fed09b0fa691273435e993676f4bfb28c28a0461) · [NVIDIA/cudnn-frontend #2478 · 11-12 12:12 · 7c6869f6b5](https://github.com/NVIDIA/cudnn-frontend/commit/7c6869f6b543605b62c13538bc32a41791e5e039) · [NVIDIA/cudnn-frontend #2474 · 11-12 10:12 · a217f27659](https://github.com/NVIDIA/cudnn-frontend/commit/a217f276597cd67564f596cbc689f5038fe6f94b)

### Cake 补齐确定性 LM-head、MoK 与 mHC 训练接口

新增准备式 LM-head log-probability/梯度计划和 MoK、mHC 训练 API，并提供 JAX 接口。LM-head 初始契约限定 SM100、CUDA≥13.4、TVM-FFI、连续 BF16 输入，1–4096 token、hidden 为 512 倍数、词表为 256 倍数。合法 label 或 -100 由调用方保证。接口约束与计划复用值得训推一致性研究者阅读，但这批 API 的确定性不能直接扩展为整模型、跨硬件或跨并行规模的逐位保证。建议先在支持的形状与梯度口径内复现。

[NVIDIA/cudnn-frontend #2445 · 11-12 13:59 · 38f7316689](https://github.com/NVIDIA/cudnn-frontend/commit/38f7316689a02ca57e96d5786557d47c71898c16)

### vLLM-Ascend 为 Mooncake V2 引入可续租 KV 生命周期

固定 480 秒回收改为生产端单调时钟上的心跳租约：默认初租 120 秒、20 秒心跳、80 秒续租窗口。完成或取消后终止续租，过期心跳不能复活已释放 KV。P/D 时长必须一致，缺少或不匹配的请求元数据会抛错，可能终止 D 的 EngineCore。文档要求先升级 P。建议有长排队 P/D 服务的团队优先审阅，并验证取消、过期及升级顺序。尚未被 scheduler 接纳的输入队列请求不因此自动受到保护。

[vllm-project/vllm-ascend #21425 · 11-12 09:11 · 451b02ce13](https://github.com/vllm-project/vllm-ascend/commit/451b02ce1372d32aa03633303f694ad17b13ecc2)

### vLLM-Ascend 阻止 SWA 回收槽位被旧缓存键重新保存

chunked prefill 释放 SWA block 后，原 tracker 只追加新 ID，可能把已复用物理槽位写到旧远端键。新路径每次保存前从当前 block table 刷新 ID，以 0 标记释放位置。非逐层保存还等待队列排空，才让下一 chunk 复用来源槽位。代价可能是 prefill 延迟增加，PR 明确没有量测，NPU 服务验证也仍待做。建议启用 KV Pool 的混合注意力服务优先回归“释放→重用→远端 miss”链路。

[vllm-project/vllm-ascend #20853 · 11-12 21:48 · e46aedbc67](https://github.com/vllm-project/vllm-ascend/commit/e46aedbc67395a6423576838321b11af6a78c263)

### vLLM-Ascend 修正图重放 padding 与 RFork 草稿权重归属

MRV2 捕获时全 True 的持久 padding buffer 以前可原样进入真实 batch，影响 EPLB 统计及有界 decoder replay。现每批原址刷新有效行/填充行，不再依赖跳 padding 开关。RFork 另从 checkpoint 权重名称恢复 DSpark embedding/LM-head 的所有权标志，避免跳过 `load_weights` 后错误换用 target 参数。前者有维护者报告的 CPU 与单卡 NPU 重放验证，后者称 NPU 接受率回到默认加载器水平但未披露完整数表。建议分别比较冷捕获首批、变 batch 和两种加载器。

[vllm-project/vllm-ascend #21475 · 11-13 00:01 · e8ae04b5c2](https://github.com/vllm-project/vllm-ascend/commit/e8ae04b5c29f0d69b9dd729cbe838215e451045e) · [vllm-project/vllm-ascend #21333 · 11-12 14:48 · b27049dc11](https://github.com/vllm-project/vllm-ascend/commit/b27049dc118ccc07b55cd048fdc70f4690b932c8)

### vLLM-Ascend 的 Gloo 专家迁移保留具体 NZ cast 参数

FRACTAL_NZ 不是单一布局，W4A8MXFP 权重带 `customize_dtype`/`input_dtype` 时可能是格式 50，硬编码格式 29 会跳过 NPU staging 或产生不同存储大小。新 `NzCastSpec` 随权重 view 及接收缓冲传递，用目标张量的实际 cast 配方重建数据，缺契约则显式报错。维护者在 A5 验证简单替代 cast 都不足。建议启用量化 EPLB 的团队核对跨 rank 的格式、存储宽度及 cast 参数，而非只比较逻辑 shape。

[vllm-project/vllm-ascend #21316 · 11-12 16:26 · 4525b3ef93](https://github.com/vllm-project/vllm-ascend/commit/4525b3ef933bf32fc7ed6908937c85c67ae939d4)

### A3 Engram 可跨 P/D 共享 owner 表，PCP 压缩改为本地计算

共享表实现使用 MemFabric BM GVA：owner 只加载一次不可变 INT8 code/FP32 scale，worker 连接时解析地址，按需 gather/dequantize 到本地 HBM，不再各自预留完整表容量。当前endpoint版只有44项CPU测试，尚无A3硬件/3P1D验证。此前共享目录版MemFabric在64张A3、3P1D原型中报告表backing从约824降至208GiB。旧测试还使用修复重复native库加载的实验wheel与leaf-module覆盖，打包修复不随本PR交付，不能当作新endpoint版或完整框架包的稳定吞吐保证。owner故障不是透明切换。另一改动把 PCP 的 C1/C2 压缩移到本地后再 gather。作者在 16 A3、DeepSeek-V4.1-Flash-W4A8、并发16、32K/128K输入/输出1、单轮 A/B 下报输入吞吐增加约3.1%–12.7%。A5是另一量化格式，且预检出现不同首 token。A3的100题GPQA子集为72/100→71/100，有界replay组为67/100→67/100，不能宣称质量等价。上述性能A/B绑定PR记录的89eb8a5e7→709ae5c25修订。建议共享表先验证所有权与故障清理，PCP 改动继续观察长输入质量。

[vllm-project/vllm-ascend #19528 · 11-12 19:36 · 89facb13f0](https://github.com/vllm-project/vllm-ascend/commit/89facb13f01de75b6482d0658d87403e925f1493) · [vllm-project/vllm-ascend #20374 · 11-12 20:39 · 3f1de8b04d](https://github.com/vllm-project/vllm-ascend/commit/3f1de8b04de8feab92587439b24e2cbf67133720)

### TorchNPU 补齐编译训练的 graphsafe RNG 与侧流图内存归属

Inductor/AOTAutograd/checkpoint 的 `preserve_rng_state=True` 路径补入 NPU 注册、seed/offset 传递和 GeneratorState 适配。fake 编译路径不触碰真实 generator。另修正侧流 WAIT 尚未加入捕获时的循环判定，改以设备活跃捕获数决定队列行为，避免后续 allocation 落入默认 pool。建议在包含 dropout 的编译训练中比较重算开关与梯度，再用侧流依赖做捕获/重放与失败清理验证。这里是公共编译链路补齐，不能据此保证所有随机算子已覆盖。

[Ascend/pytorch · 11-12 21:51 · 4455514171](https://github.com/Ascend/pytorch/commit/44555141711c0c825c5a201e303dcadad1f5fcb4) · [Ascend/pytorch · 11-12 09:30 · 0b483ce8df](https://github.com/Ascend/pytorch/commit/0b483ce8dfd62a8014013be6f2ec7bd61031dc18)

### TorchNPU 新增原生 external Event 与默认关闭的通信飞行记录器

`npu.Event(external=True)` 新路径以 CANN≥9.3.0 为门槛，并探测动态符号，用于图内 external record/wait 和跨图依赖。旧 `ExternalEvent` 保持 legacy 路径。HCCL Flight Recorder 接入共享 recorder，在提交/完成时 record/retire，watchdog 的额外 dump 默认关闭，完整使用文档仍分开交付。显存 history 同期可选全局 annotation 回调并导出阶段标记。建议通信超时、图重放和显存排查团队深入阅读这些入口，先小范围验证版本与采集开销。

[Ascend/pytorch · 11-12 19:43 · 742452710e](https://github.com/Ascend/pytorch/commit/742452710e4d99c76c58ddef60d51247ec2ab8c8) · [Ascend/pytorch · 11-12 10:34 · 9a74bb83d6](https://github.com/Ascend/pytorch/commit/9a74bb83d6efeaab71749e13fc7f270a7f2aa5dd) · [Ascend/pytorch · 11-12 10:26 · d238684ea2](https://github.com/Ascend/pytorch/commit/d238684ea2ad9fae98c8b42fa26ff1b1ce294ed7)

### TPU 修复 VMEM 槽位复用竞争，并停止关闭 DP 启动屏障

SGLang TPU 的非量化 RoPE 曾把输入/输出放在同一双缓冲 VMEM window。JAX 0.11.2 的流水线可在旧输出 DMA 完成前预取覆盖槽位。新代码分开 input(2)/output(2)，仍在 HBM 别名输出。另一提交停止自动注入关闭 libtpu 程序启动全局屏障的参数，因为不同 DP 组的 collective 交叠可能让 SparseCore 挂起或返回错值。代码保留现有 LIBTPU_INIT_ARGS，不会覆盖用户手动设置的 false，部署需检查旧配置。建议使用相关版本的团队优先复现多 tile RoPE 与跨组并发 collective，勿单用数值容差掩盖调度竞争。

[sgl-project/sglang-torchtpu #2103 · 11-12 11:27 · ac3e8f53f8](https://github.com/sgl-project/sglang-torchtpu/commit/ac3e8f53f8854472889c552050aca8f79b03f644) · [sgl-project/sglang-torchtpu #2148 · 11-13 02:43 · a53588026b](https://github.com/sgl-project/sglang-torchtpu/commit/a53588026b5c7fc07cf67986c7c9c6287563ab6b)

### Muse TPU megakernel 扩展专家布局，同时显式触发数值配置重追踪

新代码把 TP block-256 FP4、整专家 EP decode 及 PP8 prefill stage 收到统一 MoE mode。EP 大步可走 all-to-all，小步保留 gather。后续提交使配置改变真正重追踪，避免仅修改 kernel 数值选项却命中旧 executable，并在加载时一次性把现有 block-16 专家重定量化到 block-256，分块限制临时内存。重定量化改变工件数值，不能据原权重质量推导无损。建议考虑复现布局、启动成本、大小步切换和质量四项对照。

[sgl-project/sglang-torchtpu #2099 · 11-12 12:07 · 15e6693c1d](https://github.com/sgl-project/sglang-torchtpu/commit/15e6693c1d9c65524ee62ce8f2e9f8490df8bc16) · [sgl-project/sglang-torchtpu #2122 · 11-12 13:35 · 200eae95be](https://github.com/sgl-project/sglang-torchtpu/commit/200eae95beb37c151be6e6b1935c7bd25129d673) · [sgl-project/sglang-torchtpu #2131 · 11-13 00:59 · e185863cf2](https://github.com/sgl-project/sglang-torchtpu/commit/e185863cf24e794b59c4267b3d5eb133a48a5b8c)

### TPU Inference 保留异构头维的 SWA，并让 MoE 残差维持 token 分片

混合 head dimension 的 KV 规划不再一律把滑窗退化为全缓存，而是按组规划与页填充，配置需传播到 worker。ROUTING_MOE 的 `enable_sp` 路径另保留 residual token sharding，调整输入输出分片交接。测试覆盖四设备与不同 mesh。调度进程改为 spawn，避免 fork 已初始化的运行时。建议混合注意力/序列并行部署复核每组 KV 容量、分片输出与进程启动方式。这些合入尚无统一生产负载的显存或吞吐对照。

[vllm-project/tpu-inference #3957 · 11-13 03:28 · ba4e34f164](https://github.com/vllm-project/tpu-inference/commit/ba4e34f164f42c38ea0061fccd50383f44ba6bd5) · [vllm-project/tpu-inference #3961 · 11-13 05:31 · 3b9743e890](https://github.com/vllm-project/tpu-inference/commit/3b9743e890cb545f9a362bb1fbac11a19512124b) · [vllm-project/tpu-inference #3956 · 11-13 03:34 · c5265ef957](https://github.com/vllm-project/tpu-inference/commit/c5265ef957a30bbc98229b1a0832a306b359f245)

### AITER 修复注意力越界、长张量地址溢出和非均匀 MLA 元数据

非分页 split-K decode 原 mask 只约束 tile 内偏移，softmax 虽丢掉无效列，K/V load 仍会越界。现使用绝对位置先约束读取，分页路径不受此缺陷影响。但 cache_seqlens 超过物理 seqlen_k 的调用仍会越界，本次不修复这一契约违规。prefill 把 stride 运算提升到 int64，同时保留外部签名以保留向量化特化。MLA 则按原请求定位 query offset，将 query 长度显式传入 tile 计数函数，并写入 LDS 供后续阶段使用。空图填充请求最少一个 tile，避免折头后越界、零工作或挂起。建议长上下文/混合长度压测把地址与分配器边界纳入，不能只检验答案相同。

[ROCm/aiter #5793 · 11-13 03:53 · e9f7b554d3](https://github.com/ROCm/aiter/commit/e9f7b554d346f2f20366dc2bc73e736a7779cd61) · [ROCm/aiter #7278 · 11-12 23:56 · ec6494bd89](https://github.com/ROCm/aiter/commit/ec6494bd89c9e7916d35fbaac6b1bd988360c6a8) · [ROCm/aiter #6091 · 11-12 15:13 · 20c9ddbc4e](https://github.com/ROCm/aiter/commit/20c9ddbc4e2b8b4674cfcc050b4be6a3b3c7fb01)

### AITER 修正 MXFP4 专家 split-K 与尾 scale，新增 NVFP4 index-key 缓存

CK-Tile stage1 原切分可跨越真实 K 或在每段丢掉 padding 列。新路径只取能整分256列 tile 的 split，带 padded K 时回退1，并检查原始 gate/up 输出宽度。MoE sort 对行尾跨 load 的 scale 改逐字节读取，避免最后路由的尺度被抹零。另在 gfx950 为 MiniMax-M3 indexer 加入 NVFP4 key：128 token×128维每页9216字节，调用方必须写入 code+scale，AITER只消费。建议先复现非对齐 K/行尾回归，再评估 index 压缩的选择质量。

[ROCm/aiter #7136 · 11-12 15:23 · f1a4328e1a](https://github.com/ROCm/aiter/commit/f1a4328e1abef37541e0368f1dffba0a4da2633d) · [ROCm/aiter #7653 · 11-12 12:56 · 1dfc7497cd](https://github.com/ROCm/aiter/commit/1dfc7497cd628cd2b6fc3277264656117a58d9cd) · [ROCm/aiter #6958 · 11-12 19:33 · f278809b1a](https://github.com/ROCm/aiter/commit/f278809b1ad0b396c290718a3796062a6496c23b)

### AITER 的 provider 路由可注册为不透明图节点，Sol-Attn 先落前处理

`route(..., fake=...)` 可将 provider 选择从 Dynamo 跟踪的 Python 内部移到一个注册 op，维护者计数后端实验从2个图变为1个。但本提交时树内 op 尚未传入 `fake=`，因此只是可用机制，不能称现有模型默认消除了 graph break。Sol-Attn 同期加入 block pooling、选块 LUT/bitmap 和低秩 tilt 前处理，消费这些操作数的 ASM 改动仍在后续 PR。建议编译/稀疏注意力开发者深入阅读接口和前置测试，等端到端路径齐备后再评估采用。

[ROCm/aiter #7850 · 11-12 18:43 · ef839bb864](https://github.com/ROCm/aiter/commit/ef839bb864c8b39ab6b926d0ea247ef996bdf727) · [ROCm/aiter #7729 · 11-12 17:46 · 07d1b2af27](https://github.com/ROCm/aiter/commit/07d1b2af27c7baa642c3df32d23cc031efb9fb60)

### ATOM 把 DSA 索引键纳入换出，并扩展 MiniMax PAGE 与 PP→TP 传输

Kimi-K3.1 从 offload 恢复前缀时，旧代码只恢复 MLA 数据、留下物理页原有 index key。新代码一起发布索引 slice，避免 sparse top-k 排到旧请求。Mooncake Store 同期支持 M3 PAGE codec，并按 namespace 隔离版本。PP4 prefill→TP4 decode 增量则用全局层/索引层序号与每个 KV head 的 page run 描述布局，按 consumer head 切分搬运。这是标为1/2的传输部分，不能称整套异构并行服务已经验证。建议换出服务优先回归索引一致性，PP→TP继续观察完整集成。

[ROCm/ATOM #3063 · 11-12 10:44 · a11bb34c80](https://github.com/ROCm/ATOM/commit/a11bb34c8004f4c2497597cc907b70ae5fa63674) · [ROCm/ATOM #2752 · 11-12 16:46 · 88298b9fd8](https://github.com/ROCm/ATOM/commit/88298b9fd87b220f37001832b7c024fe08dde157) · [ROCm/ATOM #2721 · 11-12 13:36 · f9698ba2c0](https://github.com/ROCm/ATOM/commit/f9698ba2c0149c97ba9bdad8f8e80a3630054fed)

### ATOM 调度恢复事件驱动重排，原生 MX 共享专家修正量化尺度

waiting queue 原来每 tick 全量重建，在16,384等待者的作者微测约2.6ms。现在仅 load完成/失败事件触发，并把抢占/停放请求放到持有缓存块的已就绪 resume 后，避免饥饿。DeepSeek-V4.1 fused shared-expert 路径另按 native MX 的1×32 E8M0尺度量化，rank中间宽度不满足128倍数（如TP4为576）时记录并回退。建议复现多tick抢占与换出恢复，再检查不同 TP 的融合选择。队列微测不是整服务延迟承诺。

[ROCm/ATOM #3217 · 11-12 18:11 · 4f83e17bff](https://github.com/ROCm/ATOM/commit/4f83e17bffcd45aa3d3416a3a6b3b62b39162a87) · [ROCm/ATOM #2933 · 11-12 16:17 · b1392d302a](https://github.com/ROCm/ATOM/commit/b1392d302abab4c4f1ed1599b30cdcb9c3dcc6fc)

### ROCm CK 的 BF16 舍入修复影响 PyTorch 生成式 SDPA 构建

CK CMake 构建已定义 RNE，但 PyTorch 直接生成源码并自行编译时未带该宏，可能落入 TRUNCATE，给 P/dS/O/梯度引入偏差。生成实例现自行设置默认：backward用mode3，gfx11/12 forward用mode5，外部显式宏仍优先。gfx9 forward不改。维护者报告此前多项 BF16梯度用例超PyTorch容差，FP16不受影响。建议 AMD BF16训练团队核对实际生成源码的构建方式，并重跑带mask的梯度对照，不能只依据系统ROCm版本判断是否覆盖。

[ROCm/rocm-libraries #15648 · 11-13 05:51 · f4546a1391](https://github.com/ROCm/rocm-libraries/commit/f4546a139191a416c54ece2918fed4bf7041e0ae)

### ROCm 修复 LDS 操作数竞争及变长注意力边界，gfx1250 新增 TDM split-KV

gfx942自定义 grouped GEMM 的八个汇编 kernel 收紧等待计数，确保九次 LDS操作数读取完成后再 MFMA。作者的仿真 race detector 能发现常规数值检查漏掉的竞争。gfx950 varlen注意力把K/V资源界限收紧到每条序列末端，但此路径仅由手工spec触达，已发布catalog未提供varlen变体，错误的cu_seqlens仍属调用方责任。gfx942无LSE的无key滑窗行改写0，直接Python/API可用，hipDNN图仍保留拒绝策略。gfx950 varlen无LSE的无key组合也仍由host拒绝。另给gfx1250的TDM前向加入split/paged-KV，针对小decode网格填满算力。建议分别跑竞争/短尾/无key行与真实batch负载，不把新pipeline等同普遍加速。

[ROCm/rocm-libraries #14933 · 11-13 05:52 · 1866d8c9ac](https://github.com/ROCm/rocm-libraries/commit/1866d8c9ac3cf3d9ddf71d8c0a1416a16dc66a34) · [ROCm/rocm-libraries #14981 · 11-13 03:18 · 06d34f5b8b](https://github.com/ROCm/rocm-libraries/commit/06d34f5b8bbae7f39bf8153be0df1bc50baca1af) · [ROCm/rocm-libraries #15341 · 11-12 12:06 · 98f8b82b56](https://github.com/ROCm/rocm-libraries/commit/98f8b82b561a536ac61d3999f1b730e34b70743d) · [ROCm/rocm-libraries #14371 · 11-12 22:19 · aca3d5cef6](https://github.com/ROCm/rocm-libraries/commit/aca3d5cef647336e7eac6e63a51a606d2e14deb2)

### CANN 修复宽 K 分块 KDA 的缓冲与同步边界

A2/A3 的 `chunk_kda_fwd` 在 K>128 时使用独立的两份 L1 B slot，避免下块预取等待尚未消费的当前块。Post-WU/FwdH 沿K分块，并把整块快速更新约束在32KiB FP32临时缓冲内。尾块回写还补入屏障和AIV/AIC握手，防止覆盖未读seed。逐文件差分可定位到 prepare、post_wu 和 arch22 fwd_h。建议 KDA 训练团队优先回归宽 K、短尾、连续 chunk 的状态，避免只用常见128维整块验证。

[cann/ops-transformer !15605 · 11-12 22:12 · 4197a29c07](https://gitcode.com/cann/ops-transformer/commits/detail/4197a29c0779fe52db45712f524c7a0a32bf8abe)

### CANN 修复专家 scale 归属、对称 AllGather 落位和 A5 CCL 容量单位

MoeInitRoutingV3/V4 在quant_mode=1、按专家smooth scale的大H分块路径补上 `expertIdx*H`，不再让不同专家共用scale[0]。quant_mode=13的INT4推导另允许未知rank延后校验。维护者在950DT/CANN9.3.0报告199项kernel/eager/ACLNN回归，图模式未全部重跑。HCCL三级对称AllGather的L2输出偏移补齐R0因子和本地rank项，已有结果限hccl-vm模拟拓扑。MC2未显式给容量时重新沿用通信域窗口。A5非hierarchy正数容量q按HCCL配置MB解释，传给底层的实际字节数为q×2×1024×1024。零值仍查询默认窗口，A3不转换，原手传字节的调用需调整。该项仅有本地桩验证，未完整编译扩展或做NPU多rank实测。建议按三种具体触发条件分别核对。

[cann/ops-transformer !15813 · 11-12 17:57 · db0c617857](https://gitcode.com/cann/ops-transformer/commits/detail/db0c61785779147dcf87984fa47c910135a8dfe9) · [cann/hccl !3999 · 11-12 20:07 · cd27f581a6](https://gitcode.com/cann/hccl/commits/detail/cd27f581a68329fadc428385af8880f33824d29d) · [cann/ops-transformer !15445 · 11-12 10:35 · ce7c1e2732](https://gitcode.com/cann/ops-transformer/commits/detail/ce7c1e2732a3d6ef90bede26207caa0896b8647b)

### CANN 提升索引地址与属性宽度，并撤回同日 GMM 回归

HiFP4 paged indexer 在乘 block stride 前先把 block ID 转为int64，防止先发生32位乘法溢出。dense/sparse lightning-indexer KL梯度按int64读取 sparse_mode/cmp_ratio，校验后才赋值，避免属性截断。同日GMM从GE平台查询切换运行时查询的改动随后撤回，原因是A6加载库缺少新增符号，当前不能列成可用改进。A6冒烟仍待提交者。建议长缓存地址、异常大属性和A6加载分别建回归，持续关注被撤回问题的替代修复。

[cann/ops-transformer !15606 · 11-12 11:16 · 89b34e11f7](https://gitcode.com/cann/ops-transformer/commits/detail/89b34e11f7aa0aa14983774719729094b899bb98) · [cann/ops-transformer !15741 · 11-12 15:53 · a65e12e757](https://gitcode.com/cann/ops-transformer/commits/detail/a65e12e7574b558248b8b1992ec7d42673cc0e66) · [cann/ops-transformer !15737 · 11-12 15:53 · 370cad83ac](https://gitcode.com/cann/ops-transformer/commits/detail/370cad83aceab8b53f3b097fe6d49f92ac39de5b) · [cann/ops-transformer !15867 · 11-12 21:49 · 51c37d34ed](https://gitcode.com/cann/ops-transformer/commits/detail/51c37d34edb2bfb271d95f584d57575f14323d61)

### CANN 实验区加入量化 MatMul+ReduceScatter，recurrent KDA 保留输入步长

实验MC2算子把量化矩阵乘和reduce-scatter交接到共享通信窗口，提供torch extension及同步/布局约束，仍在experimental目录，不应直接视作稳定训练primitive。recurrent KDA另显式传递Q/K/V token/head stride，并校验正步长、末维连续和可表示范围，A2/A3以8-token窗口处理更长逻辑序列，A5保留自身长度约束。建议分别作为通信融合与非连续view处理的源码阅读材料，采用前补齐目标shape、多rank与长序列状态验证。

[cann/ops-transformer !15371 · 11-12 19:43 · 3b7a00f5af](https://gitcode.com/cann/ops-transformer/commits/detail/3b7a00f5af3d078dbad877ff144814c22f9ed7c5) · [cann/ops-transformer !15324 · 11-12 09:46 · 155c93cc90](https://gitcode.com/cann/ops-transformer/commits/detail/155c93cc90541443d210e0eb51ff681633670595)

### Gaudi 与 MUSA 接入新的模型路径，支持范围仍随运行时变化

vLLM-Gaudi合入Kimi-K3模型支持，并修正Gemma4图像归一化的设备路径，不能从“支持”推导吞吐等价。vLLM-MUSA则加入v0.32上的Qwen3.8 QSA及TileLang sparse attention路径，适用条件需与torchada/MUSA架构匹配。建议使用对应设备的团队考虑模型加载、稀疏算子与图像质量回归，公开提交尚不足以给出跨厂商性能比较。

[vllm-project/vllm-gaudi #1940 · 11-12 17:53 · 7c94381bf6](https://github.com/vllm-project/vllm-gaudi/commit/7c94381bf6fc197e2d11fb98c81380723ba28426) · [vllm-project/vllm-gaudi #1969 · 11-12 22:21 · f8dcede8da](https://github.com/vllm-project/vllm-gaudi/commit/f8dcede8da7b8982e1ba2312e966216b2c2e594c) · [MooreThreads/vllm-musa #335 · 11-12 09:40 · cf714d1658](https://github.com/MooreThreads/vllm-musa/commit/cf714d16589846ff42ce2f328e3096908b3b28f2)

### vLLM-Ascend 打通 GLM DFlash2 与 Kimi 目标 DCP 的缓存契约

GLM-5.3-Flash补齐Eagle3辅助状态接口、mHC延迟残差完成和独立draft SWA物理缓存，解决启动失败、草稿attention为零及无RoPE融合误入的问题。Kimi GQA DSpark则让target DCP分片、draft复制缓存共存，把布局信息贯穿planner、slot mapping、prefix lookup和Mooncake传输。后者作者在32张A5、K3.1 Pro量化target/BF16 draft、TP8/DP4、DCP1对DCP8做过有限题集验证，但结果绑定PR内记录的修订，padding FULL-graph元数据还指向独立修复。建议考虑复现，先检查target/draft的分组和图模式，不能只看接受率非零。

[vllm-project/vllm-ascend #21439 · 11-12 16:21 · 0bffd67a63](https://github.com/vllm-project/vllm-ascend/commit/0bffd67a63d391b9d329768d5832a244dbbfb6a1) · [vllm-project/vllm-ascend #15851 · 11-12 16:50 · e9717b0d77](https://github.com/vllm-project/vllm-ascend/commit/e9717b0d77a77a4870018a040aaf734baabdd745)

### cuDNN 分页注意力按原生硬件重新选择路径，版本号不足以确定 nightly 行为

B200/B300新增长KV前缀的tail split及混合长度full-prefill选择，但B200所得D128阈值直接搬到B300失利，最终用B300原生训练/留出shape把多tile break-even从74调到148。其混合长度声明固定每请求Q=KV，不能外推为prefix或decode结果。另解除单query MXFP8后端guard时，作者说明9.30.0.5 nightly通过，而同版本号的.2–.4仍崩溃。SM100 FP8分页路径同期分开K/V producer，SM120 decode异步发布拷贝。建议内核开发者阅读真实shape选择与构建号，服务侧按实际长短分布复测。

[NVIDIA/cudnn-frontend #2520 · 11-12 22:08 · 7c6ea67c63](https://github.com/NVIDIA/cudnn-frontend/commit/7c6ea67c633a8509ba976cbf3b0bb0c32473b606) · [NVIDIA/cudnn-frontend #2530 · 11-13 01:32 · e2fbd4f69d](https://github.com/NVIDIA/cudnn-frontend/commit/e2fbd4f69d1668e545eee6db725a5a8219a367ef) · [NVIDIA/cudnn-frontend #2521 · 11-12 23:30 · 88ab4f7676](https://github.com/NVIDIA/cudnn-frontend/commit/88ab4f76762e483b07c709a7c0f257ba4672caf1)

### TorchTPU 增加子图 C++ 调度，MXFP8 反量化补入 CuTeDSL 后端

TorchTPU的`invoke_subgraph`把子图main克隆为私有函数并以`func.call`降低。相同指纹、设备分配与标识复用函数，允许有界动态输入，但不支持动态结果类型。Transformer Engine另新增CuTeDSL MXFP8反量化并修正rowwise共享内存线程排布的bank conflict。两项分别值得编译器和低精度算子开发者深入阅读。它们是可定位的实现能力，本文未核到统一LLM负载下的收益，部署采用仍需实际模型验证。

[google-pytorch/torch_tpu · 11-13 04:08 · e95522a51b](https://github.com/google-pytorch/torch_tpu/commit/e95522a51bb9af52370b08d14021ff64059bad5d) · [NVIDIA/TransformerEngine #3842 · 11-13 00:59 · c60ad6c9a2](https://github.com/NVIDIA/TransformerEngine/commit/c60ad6c9a2c166de4404a0a06bdfe30dff4bc341)

<a id="coverage"></a>
## 覆盖与续查

模型线检查 129 个厂商、研究、API、注册表与发现入口。HTTP 状态分布 {403: 9, 200: 108, 0: 4, 429: 1, 404: 7}。定向 HF 154 个组织、308 个模型/数据集 lastModified 首屏查询（每屏 50），状态 {200: 308}。364 条重叠窗候选。全站 createdAt：模型 29 页、数据集 17 页，均翻过回溯起点。这只覆盖创建索引，不能代替全站修订或全部候选语义审阅。

定向检查 57 个仓库的 info/commit API，其中 5 次 401。固定工件、配置、diff、官网报告与访问回执均保存在 models/evidence。中英文主题检索及全站候选发现补到 facebook/MetaEncoder v1。该仓不在原 154 组织清单，后续需扩充。候选合并按稳定仓库+commit、模型版本与官方发布日期进行，旧论文/旧分数不会因卡片更新时间变成当天首发。

收录 13 个稳定事件版本（正文 12 个标题）。AlphaProof Code 11-10 与 Gemini Decisions API 11-11 为72小时回溯补录。LongCat-Next dNaViT 权重缺口已在固定提交下解决。LLMON方法在10-30已收录，本次11-12官方博客与卡片无足以改写旧结论的新增实验，因此只记检查，不重复正文。H3-Image新增部署链接、Cosmos3链接修复、PAN2下载统计字段、Jais引用与Granite签名等均过滤。

主要续查：HF历史401（XYBench、PLE、部分NVIDIA工件），MetaEncoder许可/训练报告，Hydrogen完整安全报告，ZONOS3新增文本行到runtime对应，Cactus受限量化卡，以及全站lastModified与剩余长尾语义候选。Claude自托管Managed Agents只核到11-12公告，深页HTTP200但仅壳，未以此宣称机制已核。原相关历史待查638行在 inherited-backlog.json，不将它们自动结案。本轮没有独立复训或设备压测。仅本地重放了SmolVLM2修复前后Jinja字符串/列表模板。

- **论文与学术来源实际覆盖。** 主窗 (2026-11-12 09:05, 2026-11-13 09:03]，回溯至11-09 09:05（Asia/Shanghai）。抓取arXiv cs.AI/cs.CL/cs.CV/cs.DC/cs.LG/cs.NE/stat.ML/cs.AR/cs.PF 的new/recent。九分类new反复返回11-11公告，895个去重ID全部与OAI版本史关联。cs.CL/cs.LG末次刷新仍旧批次，不能声称11-13常规公告完整。
- **OAI恢复与日期边界。** cs的11-09至11-13元数据切片遍历5页至空resumptionToken（5,729原始rows），stat同窗610 rows完整。另完成cs 11-10至11-12旧切片。合并目录4,086个不同ID，回溯提交时间内930个v1与661个修订版本（包含本轮重新取得的历史版本）。OAI datestamp11-13显现14520–14681中的新增/迟发记录，包括KernelMosaic、TogetherCompiler、tau2判分、历史与世界状态、弃答阈值等，本轮分别读正文。不能用OAI日期推断第一次公开的秒。公开原始v1提交最晚仅到11-10 19:37，未公告提交和独立lastUpdatedDate区间继续留查。
- **发现入口。** HF11-11、11-12日日榜分别39/21条，只用于线索与去重。11-13为HTTP400。读取AACL程序/工作坊/教程/海报与ACL Anthology AACL。EMNLP页面HTTP200但正文下载超时、不完整。打开NeurIPS、TMLR、PMLR、MLSys、OSDI、SOSP入口，但未逐篇完成公开/接受/修订时间线。OpenReview TMLR/NeurIPS/ICLR API均403。ARRSI组页可读，不等于完整评审API恢复。中英文主题检索补充固定名单，关键结论均回到原论文/官方来源。
- **证据深度。** 本轮38个abs元数据、36份可取HTML全文、OpenAI20页原报告与Together38页原报告。27个正文条目对应25个此前未收arXiv版本、1份OpenAI11-12报告、1个MACaQ-R新commit。MACaQ论文v1已在11-11日报，本轮仅补条件作为代码增量背景，不重复计事件。读取固定实现六组并保留源码对应，检查另一个发布仓发现仅coming-soon。没有独立训练、量化或设备压测。候选队列672项是保留待审范围，不代表全部值得收录或已完成语义审阅。
- **继续追踪。** 精确submitted/updated Atom429、当前公告可见性、早于11-09的旧缺口、其余候选及修订、正式会议逐篇时间线、论文声明开放但尚未定位的主工件继续保留。KernelMosaic/Plumbline/HiLo/CAMP-II/SELD/FSQ/POEM/DARIC/SRE/ReTrace等不把论文声明的开源当作已核实完整包。DenseGRPO仓当前README仅“CODE WILL BE AVAILABLE SOON”。本线175次HTTP检查有14次失败或部分返回，详见ledger与fetch logs。全局完整语义水位不应推进。

### 开源工程覆盖

工程主窗（UTC）为 2026-11-12 01:05:00 至 2026-11-13 01:03:00，回溯至11-09 01:05:00。本线整理197个稳定版本事件、76组正文，所有关键事件有实际打开的原始PR、固定提交补丁、正式release或官方技术报告。主链接、稳定ID/version与既有状态查重，无重复。无独立复训或设备压测。

- GitHub检查75个仓路径入口（包含更名/重定向别名与分给硬件线的目录），默认分支履历、release Atom与主窗merged分页各70个入口成功。15个核心及定向仓补查全部回溯merged分页。五个旧路径返回404：SqueezeAILab/SeerAttention, attention-gym/attn-gym, sgl-project/sgl-flash-attention, sgl-project/sgl-kernel, zml/llmd。slime、OpenRLHF默认分支与发布工件本轮目录未见主窗增量，这只是已查路径的结果。
- 34个PyPI包名查询29个成功，404入口为attention-gym, radixark-miles, neptune-model, miles-diffusion, xla-serving。用途为发现版本/工件，不因上传新wheel或版本号本身收录。llama.cpp发布页补查7页跨过回溯起点。TorchTitan/PyTorch/Run:ai也检查正式发布页，过滤ciflow/trunk/nightly等非实质版本。
- 多项PR .diff访问遇429，一次vLLM PR HTML遇502。均已对入选条目用精确merge commit的.diff/.patch或重取HTML恢复，见patch-recovery.json、patch-fetch.json及原始pr/patch文件。未恢复来源与缺口仍在ledger里，不能以成功目录替代语义全覆盖。
- 时间含义：普通默认分支PR用GitHub mergedAt。PyTorch ghstack关闭PR及已进入默认分支的stack commit用实际default-branch committedDate并记录alias，PR stack base并非最终运行分支。NVSHMEM作者日期多为10月、默认分支本轮11-12公开批，未称新正式release。日期未披露时分的Rubin官方报告保留日期口径。
- 回溯期非15个定向仓的feature-branch merges没有逐仓穷尽。其余候选语义审阅、11-11 EPD技术写作联合PR复核与继承旧缺口继续待查。具体窗口与原始inventory见ledger.json、decisions.json、inherited-backlog.json。TPU/Ascend/ROCm等专属仓的正文由硬件线承担，本线仅提供目录，NVSHMEM/TileLang由本线展开。

### 硬件与软件栈覆盖

核查NVIDIA CUDA/cuDNN/NCCL/Transformer Engine、AMD ROCm/AITER/ATOM、Intel Gaudi、昇腾CANN/TorchNPU、TPU/JAX、AWS Neuron及MUSA/MLU/沐曦/Apple等公开入口、版本和代码。记录515次HTTP请求（包含补丁/逐文件差分与重试），507次传输成功。传输成功不等于完成语义审阅。主窗候选台账枚举564个提交，筛选正文65个稳定版本事件。代码与PR差分固定版本，GitCode逐文件差分另保存在`*-files.raw`。

主窗之外仍保留72小时内的目录与候选复审缺口，未选集合含常规/范围外与未完成审阅两类，不能当作逐条已否决清单。MLCommons入口403，沐曦旧news入口及部分旧仓Atom404。可替代入口的成功不抹去原访问失败。昇腾11-12两篇技术文章目录可见，正文接口404，故未据摘要写结论。CANN组件映射、libtpu历史工件差分、A5X规格冲突与长尾语义候选继续留查。TorchNPU #5953目前仍只有bot分类证据，driver #225保持open，未声称故障已解决。

MACaQ的两个昇腾算子PR（ops-nn !12568、ops-tilelang !44）本轮明确核为open。论文与现有后端在论文段合并展开，未重复记作硬件已发布能力。未发现主窗stable libtpu/FlyDSL版本工件增量，不因版本目录可读而重收旧版本。

- **续扫边界。** 本轮发现截止为 2026-11-13 09:03:00。下轮从该位置续扫并回溯 72 小时，同时继续处理分源待查记录。全局完整语义水位仍未闭合。本轮未做独立复训、设备压测或付费 API 实测。
- **归档状态。** 本轮新日报与日期索引保存在本地，GitHub写入暂未完成：Git HTTPS无可用认证，SSH校验失败，网页未登录，当前会话未安装可写GitHub连接。远端最后核验提交仍为 `ebda0ad39dc79d5e5476a78cf487092353151e19`。两个待同步文件和哈希独立记录，下轮优先补传，不因上传失败重复生成正文。
- **补录与更正。** 本文按各条原始日期标注回溯或旧缺口补录；没有重写历史日报。SGLang GDS默认淘汰、CANN GMM同日撤回按最终状态记录。没有证据充分的厂商不能由搜索空缺推断“没有发布”。
