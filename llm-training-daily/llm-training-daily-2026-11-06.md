# LLM 技术动态日报 · 2026-11-06

> 时区 Asia/Shanghai（UTC+8）。增量扫描窗口 **2026-11-05 09:05:08 → 2026-11-06 08:58:16**，回溯自 **2026-11-02 09:05:08**。实际生成时间 **2026-11-06T09:14:27+08:00**。正文区分发布、提交、公告与本轮发现，来源仅披露日期时保留原日期。
>
> 本轮生成前核对本地 29 篇历史日报和索引，与 GitHub `main` 的 `dd941277520a4f204946c218df432a19602d71ad` 同内容。上期无待同步文件。历史归档用于去重和确认变化，本轮技术结论来自重新打开的来源。
>
> 评测和性能数据标注作者、厂商或独立机构口径。本轮做了资料和代码核验，未独立复训、测量设备性能或调用付费模型 API。

## 本次重要变化

- **两份报告值得并读。** 百度开放 ERNIE 5.5 Flash Base 的 FP8 和 BF16 基座权重，报告披露 33.5T token 和两类压缩注意力。Thinking Machines 披露 Inkling 2 Small 的原生 NVFP4 预训练和统一异步 RL，权重仍待发布。
- **TPU 线性注意力有直接的大规模训练数据。** Systolic Linear Attention 在 512 芯片 TPU v7x 的单种子对照中，从 15,135 提高到 17,445 token/s/芯片。要复现收益，需拆分状态精度、归约结构与内核实现。
- **优先复查缓存与数值正确性。** LMCache 主干按 LoRA 身份隔离 MP cache key，FlashAttention 2 对左填充 KV 走带尾块掩码的路径，vLLM Ascend 修正 DeepSeek V4.1 缓存块大小与 FP8 跳过层匹配。
- **补录 vLLM 0.33.0 正式包。** PyPI 工件在 11 月 2 日发布。GitHub release 与 tag、PyPI 的发布状态需分别核实。

[模型、报告与评估](#models) · [研究论文](#papers) · [核心开源工程](#core) · [扩展开源工程](#extended) · [硬件软件协同](#hardware) · [覆盖与续查](#coverage)

<a id="models"></a>
## 模型与技术报告

### 文心 ERNIE 5.5 Flash Base 开放预训练终点权重

**11-05，权重与技术报告。** 百度发布 280B 总参数、每 token 激活 14B 的稀疏 MoE 基座。报告披露用 Muon 与 FP8 混合精度从头训练 33.5T token，预训练最长 128K，微调和推理支持 1M。43 层主干中，37 层 HCA 把远端历史按 128:1 压缩，6 层 MLA 保留逐 token 访问并用 DSA 索引选 top-k。路由由多个独立 sigmoid 路由器共同打分，每 token 选 512 个路由专家中的 10 个，另加 1 个共享专家。ECHO 复用主干最后一层预测未来 token。固定版 FP8 和 BF16 各有 128 个 safetensors 分片，配置的 DSA top-k 为 2048。许可 v1.0 允许非商业使用，商业用途按相关产品或服务年收入设通知和授权条件。[百度技术报告 §2–3、§5](https://ernie.baidu.com/zh/release/ernie-5-5-flash-base/) · [FP8 权重](https://huggingface.co/baidu/ERNIE-5.5-Flash-Base/tree/028544b2ffceea189b2d8768ed2b9d09b06f2343) · [BF16 权重](https://huggingface.co/baidu/ERNIE-5.5-Flash-Base-BF16/tree/dfab899dadcff8f4d08af24b75a26c6d4d2e0b09) · [许可 §4](https://huggingface.co/baidu/ERNIE-5.5-Flash-Base/blob/028544b2ffceea189b2d8768ed2b9d09b06f2343/LICENSE)。

百度以相同评测框架、提示和 few-shot 样例对比四个已发布的 Base checkpoint。1-shot LiveCodeBench pass@1 为 40.1，DeepSeek-V4-Flash-Base 为 31.9。0-shot LongBench-v2 EM 为 48.7 对 44.5，InfiniteBench 均值则为 46.5 对 48.0。这批权重可作为领域续训和后训练对照。长上下文复现时宜把 128K 训练长度、1M 支持上限和各长度段效果分别记录。[技术报告 §4](https://ernie.baidu.com/zh/release/ernie-5-5-flash-base/)。

### Inkling 2 Small 披露原生 NVFP4 预训练与统一异步 RL

**11-05（来源仅给日期），Tinker 开放使用，权重待发布。** Thinking Machines Lab 公布 594B/A17B、支持图像输入及可变推理强度的 MoE。其 H1 在 NVIDIA GB300 NVL72 上以原生 NVFP4 训练 8.4T token，耗时 68 小时。H2 在 512K 上下文训练 3.8T token，H3 引入 1M 上下文的推理与 agent 轨迹，H4 收窄到约 34B 高质量 agent 轨迹 token。随后把推理和 agent 任务放入同一次异步 RL，9 天完成 1,075 步、逾 1,700 万条 rollout。其 StableReinforce 使用分数中心化，策略陈旧度中位数 17 次更新，rollout 时长 p50 为 8 分钟、p99 为 4 小时。完整 NVFP4 配方待后续披露，Apache 2.0 权重是完成外部安全测试后的计划。[官方发布与训练说明](https://thinkingmachines.ai/news/inkling-2-small/)。

Artificial Analysis v5.0.1 指数得分为 31，前代为 13，但平均每任务输出从约 37K 升到约 150K token。其 FrontierCode 2.0 得分从 5% 到 25%，在同一第三方报告的 Terminal-Bench 4.0 上得 13%，低于 Claude Sonnet 5.5 (medium) 的 30%。评估长程 RL 收益时可对齐轨迹上限和输出预算。厂商自己的评测把最大轨迹从前代 256K 提到 1M。[独立评测与输出预算](https://artificialanalysis.ai/articles/thinking-machines-releases-inkling-2-small-highest-scoring-open-weights-model-outside-of-chinia-upon-weights-release-it) · [厂商评测设置](https://thinkingmachines.ai/news/inkling-2-small/)。

### GPT‑6 Cyber 用专门 RL 训练利用能力与授权范围遵从

**11-05 17:00（北京时间），Daybreak Red 限制开放。** GPT‑6 Cyber 从 GPT‑6 Astra 派生，后训练包括复现崩溃、验证利用、扫描代码库，也奖励遵守授权范围、准确汇报及减少对正当任务的拒绝。OpenAI 的指定漏洞利用评测 ExploitGym 从 GPT‑5.6 Cyber 的 32.1% 提高到 48.5%。在另设的 Honeypot 测试中，未加部署指令时，越界利用辅助服务的比例从 5.51% 降到 0.07%，加指令后该样本中观察到 0%。[官方发布](https://openai.com/index/introducing-gpt-6-cyber/) · [系统卡 §2、§4.1](https://deploymentsafety.openai.com/gpt-6-cyber) · [官方 RSS 时间](https://openai.com/news/rss.xml)。

Irregular 在更新后的 CyScenarioBench 中记录 Cyber 至少一次解出 11/11 道题，Astra 为 8/11。在更新后的 FrontierCyber 上，Cyber 为 90/288，Astra 为 96/288，均未攻破完全加固的目标。值得并读系统卡里的任务完成、范围遵从和独立测试，三种口径对应不同的部署问题。[系统卡 §5、§8.1](https://deploymentsafety.openai.com/gpt-6-cyber)。

### ByteDance Seed 发布两组 Qwen3 机制解释工件

**11-05，Apache 2.0 权重与示例。** 面向 Qwen3-8B-Base 的跨层 transcoder 在 36 层各放 32,768 个 latent，每 token 按激活乘 decoder 范数跨层合选 360 个，某层 latent 可重构本层与后续层的 MLP 输出。完整 FP32 工件约 377 GB，卡片提供跳过 decoder 的仅编码模式。面向 Qwen3-30B-A3B 的三组独立 SAE 则读取第 4、30、46 层 MoE 块输出，各从约 105 万 latent 取 top-100。卡片报告三层未解释方差分别为 12.6%、25.1%、7.6%。适合先在固定基座和 hook 上复现重构，随后单独验证特征对具体行为的因果贡献。[跨层 transcoder](https://huggingface.co/ByteDance-Seed/Cross-Layer-TopK-CLT/tree/7675a16683788bc4d119d6771e549b35112f0255) · [MoE SAE 与评测口径](https://huggingface.co/ByteDance-Seed/Qwen3-30B-A3B-MoE-SAE/tree/6a6cc923f40b26e1f54c833b075c15eb4e4aa7b0)。

### Kimi K3.1 Pro 的参考编码器移除 medium 档

**11-05，既有发布的兼容性变更。** `encoding_k3.py` 将本地 `thinking_effort` 校验收窄到 `low`、`high`、`max`。用同一条用户消息调用新旧版 `build_chat_segments`，在 `thinking=True` 时，旧版接受四档，新版仅 `medium` 触发 `AssertionError`。生成给模型的说明字符串仍列出 `medium`。使用参考编码器的调用方应按新校验值接入，托管 API 的档位需按其自身版本验证。[Moonshot 固定提交与代码差异](https://huggingface.co/moonshotai/Kimi-K3.1-Pro/commit/d9e99b0792a5e93b78ea9bea62a68167e36ddd9e)。

### NVIDIA 补齐 Cosmos 3 Nano 交互式仿真复现路线

**11-05（来源仅给日期），官方配方发布。** NVIDIA 发布四个既有 16B 可训练参数专用 checkpoint 的联合说明。Transfer-Auto 用于多视角驾驶数据，Sim-Auto 流式生成同步相机与 LiDAR，Sim-Transfer 以仿真器控制生成视频，Sim-Bimanual 以动作预测视频。流式生成保留历史的 KV，分块去噪后把生成片段接入下一轮。官方配方从双向领域适配过渡到 teacher forcing、四步一致性蒸馏，再以 self-forcing 在学生自己生成的历史上训练。四套 HF 目录的截止前提交均为 10-30，本轮新增的是使用和训练说明。续训需完整训练 checkpoint，推理导出的权重缺优化器状态。[NVIDIA 联合说明与工件入口](https://huggingface.co/blog/nvidia/cosmos-interactive-world-simulation)。

### Gemini Omni 1.2 Flash 扩展视频生成与逐轮编辑接口

**11-06 01:00（北京时间），API 正式上线。** `gemini-omni-1.2-flash` 支持原生生成 15 秒视频、最多 6 张参考图，并能逐轮延长至总计 30 秒。旧版 `1.1` 的自动路由计划于 12-03 生效，调用方复测时需固定模型 ID 和日期。模型卡确认文本、图像、音频和视频输入，架构和训练细节仍沿用 Omni Flash 卡片的披露范围。[Google 官方发布](https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemini-omni-1-2-flash/) · [Gemini API 更新记录](https://ai.google.dev/gemini-api/docs/changelog#11-05-2026) · [DeepMind 模型卡](https://deepmind.google/models/model-cards/gemini-omni-1-2-flash/)。

### Ai2 ToyBlocks 分开发布视觉规划与低层动作权重

**11-06 凌晨（北京时间），HF 工件。** Molmo2-4B-tb 用目标和前视图预测下一步以及图像坐标。322.5M 参数的 ManiFlow-tb 用图像、机器人状态、任务名和目标热图执行动作，权重是第 200,000 步的非 EMA 版本。卡片对同一上层模型切换低层执行器，困难组未见积木时，仿真闭环成功率为运动规划器 14.5%、ManiFlow 10.0%。真实机器用 ManiFlow 的结果是 10 次中成功 6 次。三种仿真难度各 200 条 episode，适合复查视觉规划错误和执行失败分别占多少。[Molmo2 固定工件与评测](https://huggingface.co/allenai/Molmo2-4B-tb/tree/a154d34f1d7bf32328eeb10401b89548898c0390) · [ManiFlow 固定工件](https://huggingface.co/allenai/maniflow-tb/tree/fc869329df7fe3ea576e5c8aa1ce86cfb46aaf77) · [仿真训练数据](https://huggingface.co/datasets/allenai/toyblocks-data/tree/90ca4125f69d081e04a26f11c8c78c1804f211af)。

## 评估与数据

### METR 审查 Anthropic 的 AI 研发加速证据

**11-05（来源仅给日期），独立审查。** METR 在获准访问 Anthropic 内部预训练、RL、算力和评测数据后，估算 2025 年 10 月以来内部 AECI 增长斜率为此前的 1.9 倍，95% 置信区间 1.5–2.8 倍，替换评测纳入、模型日期和饱和处理等口径后得到 1.1–2.9 倍。它判断 AI 协作可能解释当前加速的大部分，但反事实归因高度不确定，多项内部证据经删节。配套审查同意 7 月 15 日模型直接自动化关键研发的风险较低，认为提前预警阈值是否越过仍有争议。阅读时宜分开看实测斜率、AI 贡献的主观估计及完整替代研究员的证据。[独立评估](https://metr.org/blog/2026-11-05-anthropic-ai-rd-acceleration-assessment/) · [八月风险报告复核](https://metr.org/blog/2026-11-05-rd-section-anthropic-risk-report-august-2026-review/)。

### LifeVal 补全 169 个个性化 agent 任务的评分方法

**11-04 提交，重叠窗口补录。** OpenAI 的 arXiv v1 补上 11-02 公告未给出的实验口径。169 项任务覆盖 GPT‑5.2 到 GPT‑6 Astra，每条 rollout 上限 8 小时。GPT‑5.6 主评分器逐项查看 rubric、输出工件与模拟网站后端状态差，另一评分器把访问环境内部 API 的轨迹计零。结果图用 bootstrap 给出 95% 置信区间。美国英语支付意愿调查影响任务分布，未冻结的网页缓存使复测仍有变异。复用其方法时，可先把实际副作用的状态差和口头完成声明分开评分。[LifeVal v1 §3.5、§4–5](https://arxiv.org/html/2611.06687v1) · [提交记录](https://arxiv.org/abs/2611.06687v1)。

### LemonDecider 数据集把逐行许可与软标签一同发布

**11-05，训练数据工件。** inclusionAI 发布 1,199,692 条由 287 个上游数据集重构的类型化决策，统一为选项、二元或评分分布，其中 61,057 条保留投票或分数形成的非独热软目标。固定版目录含训练 JSONL、按行对齐的上游许可文件、汇总和验证报告。卡片将 66.2% 标为可商业用、31.0% 标为未指定、2.8% 标为非商业，归类属于发布方尽力整理。重用时应把上游许可逐行过滤和 soft-label 子集的校准结果分别验证。[固定数据卡、许可清单与验证工件](https://huggingface.co/datasets/inclusionAI/LemonDecider-Decision-Mix/tree/cf1f10d526d52ec32bbd082ba15ff94c320f44b2)。

### Nanbeige4.3-33B 新增第三方预算与能力比较

**11-05（来源仅给日期），既有权重的新评测。** Artificial Analysis 报告 Nanbeige4.3-33B 与 Qwen3.8-27B (xhigh) 在 Intelligence Index v5 同得 20。Nanbeige 在 Terminal-Bench 4.0 为 30% 对 6%，HLE-Diamond 为 7% 对 15%，平均每项输出 158K 对 110K token。AA-Omniscience v1.1 的较低幻觉率与较低尝试率并存，答对总比例也较低。适合按目标任务与输出预算选对照，HF 版本在 10-28 已存在。[第三方评测](https://artificialanalysis.ai/articles/nanbeige-has-released-nanbeige4-3-33b-scoring-20-on-the-intelligence-index) · [已发布权重](https://huggingface.co/Nanbeige/Nanbeige4.3-33B/tree/df919e1e2cf077a22204c78b6c826bf3db18bc08)。

### Anthropic 用分层分类观察科研场景中的 Claude 使用

**11-05（来源仅给日期），观察研究。** 报告以两级模型分类器识别消费计划科研会话，并映射到 SciNet 的领域与任务。2026-08-02 至 09-30 主样本为 207,049 条，数据分析占 Claude 会话 33.7%、Claude Code 会话 48.1%，数据采集为 11.5% 和 11.9%。另一组 528 人调查偏向已采用 AI 的生命与健康科学研究人员。会话份额来自分类估计，报告的产出加速倍数来自情景模型。适合用来构造采用场景的分层评测，实际节省时间需另测。[Anthropic 方法、图 3 与限制](https://www.anthropic.com/research/how-researchers-use-claude)。

<a id="papers"></a>
## 研究论文

本节按 arXiv 版本提交时间（UTC+8）归类。以下首录主要来自 arXiv 标为 11 月 5 日的公告列表，提交时间均在回溯窗口。公告页标签和逐篇提交时间分别记录，未据此推断逐篇首次公开时刻。实验数字来自作者的固定版本。 [arXiv 公告规则](https://info.arxiv.org/help/availability.html) · [11 月 5 日计算语言学公告列表](https://arxiv.org/list/cs.CL/new?show=2000)

### 预训练与架构

#### Systolic Linear Attention 按 TPU 矩阵单元重写门控 Delta 更新

**11-05 01:42:58｜v1 首录｜§2–3，Tables 6、12，Appendix C**

TPU 的 256×256 矩阵单元与窄键、逐通道衰减、块内顺序求解配合不佳。SLA 让多组 value head 共享 256 维 Q/K，把遗忘门按 tile 分组，将衰减提到矩阵乘法外，并用分块倍增做块内求逆。中间量与反向留在融合的 VMEM kernel。512 芯片 TPU v7x 上，63.1B 总参数、3.74B 激活参数的 8K NoPE MoE 训练至 120.8B token 时，R4 对 MaxText KDA 从 15,135 提升至 17,445 token/s/芯片，11 集 PPL 12.596→12.454。这个配置单种子，收益同时包含 BF16 状态、结构和 kernel 变化，长上下文结果也随是否启用 SSMax 变化。**建议深入阅读**。将 KDA 精度、状态大小和求逆方式拆成对照，尤其留意附录中近似求逆的数值退化。 [固定全文](https://arxiv.org/html/2611.06828v1)

#### BabyLM 受控实验区分循环深度和数据组成

**11-04 17:21:38｜v1 首录｜§3–6，Tables 1–2、4，Appendix L**

同一 decoder-only 实现，12 层共享四轮与 12 层单次、48 层独立参数对照。100M 唯一词、1B 累积词曝光，主条件各三种子。纯文本 BLiMP 为 76.16、75.21、70.80，循环对同参数单次仅 +0.95 点（Welch p≈0.12），相对推理计算匹配的 48 层则高 5.36 点。仅初始视觉前缀的严格对照中，循环与标准的多模态损失差异约 −0.01 点。多模态把半数文本换成 caption，同时改变保留语料的构成与 batch 组织，论文只有循环侧的无图匹配对照。**建议深入阅读**。有限数据实验可参照作者的独立层深度和输入组成控制，避免把 BLiMP 的变化外推到大规模多模态训练。 [固定全文](https://arxiv.org/html/2611.06206v1) · [作者数据集](https://huggingface.co/datasets/daichi812/babyloop-mm-corpus)

#### LooM 只让解码层反复读取，限制循环对持久 KV 的影响

**11-05 01:52:11｜v1 首录｜§3–5，Tables 2–4**

编码器与首个解码层建立共享全局 KV。后续层可以反复读和重选索引，每层局部滑窗只在首次访问时发布该 token 的 KV。204M 参数、同数据与更新步数下，Random-Depth 从 11 次调用开始超过六次调用的单遍基线，21 次把交叉熵从 3.0506 降到 3.0307 nats/token，128-token 提示后的缓存同为 364.1 KiB。单 H800、batch 1 的 Python 参考实现中，表列六次与 21 次路径耗时 56.52 和 112.54 ms/token。**建议跟踪**。等有优化实现和更大规模数据，再看延迟预算下的收益。现有结果可用来设计固定 KV 的深度消融。 [固定全文](https://arxiv.org/html/2611.06835v1)

#### AMD 把稀疏 KV 搬运交给 TDM，把部分集合通信搬离计算单元

**11-04 07:44:04｜v1 首录｜§3–4，Tables 8–9、13–14**

MI455X 的 TDM 将稀疏行直接 gather 到 LDS 环形缓冲。对同卡上最好的显式加载实现，DeepSeek V4 CSA 的 Flash/Pro、4K–16K 前后向 kernel 用时几何平均改善 1.88×，最高 2.74×，来自 bring-up 平台。MI355 上将集合通信搬运交给 SDMA，仍由本地 persistent kernel 做 ReduceScatter 的归约。与独立 GEMM 比，AllGather 并行时的几何平均耗时增幅从 28% 降到 8%，八卡 Llama 3.1 8B 的完整训练吞吐提升 2.1%。**建议深入阅读**。Kernel 与整步要分别验收，并记录 wave、寄存器占用和可重叠区间。 [固定全文](https://arxiv.org/html/2611.05751v1) · [Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo)

### 后训练

#### 多教师同策略蒸馏的域内成绩掩盖了组合和外推缺口

**11-04 04:53:26｜v1 首录｜§5–6、9.1，Figures 2、5–7**

Qwen3-8B 学生在自己的 rollout 上分别匹配专用教师。单教师 OPD 的数学域内均分达到教师的 97.10%，作者划为域外的 AMC23、AIME24/25 均分比基座低 8.56 个百分点。代码与工具组合中，MOPD 的 SWE-bench Verified 为 8.05%，基座 6.63%，同配比 mixed RL 7.79%，共享教师 9.84%。域外标签对应所用后训练配方及教师得分，预训练暴露情况未知。跨组合结果来自有限模型家族，mixed RL 与 MOPD 的配比与课程调参也有限。**建议考虑复现**。能力合并验收应同时保留基座、教师和组合任务成绩，并另设按难度外推的 holdout。 [固定 PDF](https://arxiv.org/pdf/2611.05609v1) · [作者项目页](https://yichenzw.com/opd-generalization.github.io/)

#### 多轮 RL 的优势取决于剩余交互轮数

**11-04 21:24:54｜v1 首录｜§2–5，Figures 3–4，Table 1**

在同一任务上分别扫描 rollout 数与轮数上限，并在相同中间状态比较每个策略的 16 次下一步动作。FrozenLake 的动态规划价值与 DeepCoder-MT 的单元测试均显示，完整预算训练带来的轨迹收益没有稳定延伸到逐步动作。BudgetMix 将多种交互轮数上限放进训练。Qwen3-4B/8B 在 FrozenLake、DeepCoder-MT、BFCL-v3 和 InterCode-SQL 上，固定 Pass@8，25%/50%/75% 轮数预算相对全预算 RL 平均 +7.5/+5.3/+4.0 点，满预算相当。可靠的逐步质量信号只覆盖两个环境。**建议考虑复现**。在实际工具时限和重试配额下固定采样量做预算曲线，分别监控整条轨迹与局部动作。 [固定全文](https://arxiv.org/html/2611.06483v1)

#### 固定每题奖励总量，改变不同成功率题目的更新权重

**11-04 12:32:08｜v1 首录｜§4.1、5，Algorithm 1、Table 2**

每道题的 rollout 奖励先减去组内最小值，再把剩余差额归一到单位总量，最后中心化成优势。全同分组跳过，因此二元奖励下，组内成功样本越少，每个成功样本分到的正奖励越多。Qwen3-8B、八个 rollout 一组、保留参考策略 KL，GRPO 的五项可验证任务平均 59.87%，加上此方法提高 6.66 点。主实验单训练种子，未扫描组大小，主要为二元可验奖励，且正文未给完整停训阈值与每次训练长度。**建议考虑复现**。固定采样、墙钟和 checkpoint 选择，同时记录零更新组与各难度的优势权重。 [固定全文](https://arxiv.org/html/2611.05942v1)

### 推理与服务

#### D3D 用独立扩散草稿覆盖下一轮可能的验收结果

**11-04 20:27:58｜v1 首录｜§3–4，Tables 1–2**

目标模型验证当前块时，独立 drafter 在另一张卡上打包预测若干拒绝位置及全接收时的续写。已验证前缀（含目标模型新 token）匹配预算分支才复用，未命中走同步补 draft。算法写有拒绝采样，实测为 Transformers greedy、non-thinking，异步方法占两张 H800，同步基线占一张。七项平均加速相对 AR 为 Qwen3-8B 5.00×、32B 5.71×，同为双卡异步的 PARD-2 分别为 4.08×、3.88×。32B 换 1.7B 草稿、并发 16 时，MATH-500 为 2.7×、MT-Bench 1.5×。**建议考虑复现**。官方有代码和草稿入口，复测时同时计 GPU 成本、非 greedy 正确性、分支命中率与生产后端。 [固定全文](https://arxiv.org/html/2611.06425v1) · [作者实现和工件入口](https://github.com/fudan-generative-vision/D3D)

#### Split-KV 让训练扩散草稿的上下文并行只传短侧

**11-04 12:16:46｜v1 首录｜§4–5，Algorithm 1、Figures 6–7**

长上下文的 K/V 分片留在本地，各 rank 收集较少的 draft query，计算局部输出与 log-sum-exp，再做全局归一和输出 reduce-scatter。反向沿同一分界交换梯度，变长分片按成本做 LPT 负载分配。DFlash 两种配置在 64 卡、每样本 1M context、8K query 的弱扩展端点，相对当档最快可用基线快 2.81×、2.19×，从八卡到 64 卡弱扩展效率 91%、85%。只测固定长度合成输入、DP=1、离线 target hidden state，剔除特征提取，Ulysses 在 64 卡受头数限制。**建议考虑复现**。把实际样本的 query/context 比例和在线特征生成纳入整步验证。 [固定全文](https://arxiv.org/html/2611.05926v1)

#### flash-comm 把数据就绪与归约求值顺序分开

**11-04 23:16:17｜v1 首录｜§3–6，Tables 5–7**

数据块用哨兵或序列号携带就绪信息，接收端按块轮询和重新布防，减少跨卡 barrier。浮点归约按 rank 固定顺序，或由唯一 owner 计算后分发，以维持参与卡之间的一致性。八张 B300，DeepSeek-V4-Flash-0731 的单节点 TP 解码，CUDA Graph 下相对 SGLang custom AR v2 的 TPOT 在并发 16–512 降 1.7–6.8%，并发 32 时 4.38→4.08 ms。八张 B300 的微基准 AllReduce 中位降幅 28.8%。优化限 Hopper/Blackwell、NVLink/NVSwitch、对齐且 peer-accessible 的 BF16/FP32。哨兵冲突的输入位模式会被规范化为 NaN。**建议深入阅读**。比较 CUDA Graph buffer 重用、非有限值和端到端 TPOT，而非只测微基准峰值。 [固定全文](https://arxiv.org/html/2611.06634v1)

#### vExpert 以主存冗余专家应对 MoE decode 热点漂移

**11-04 15:02:52｜v1 首录｜§3–6，Figures 4、6–7**

保留 GPU 上的常规专家，Host DRAM 提供全覆盖的冗余池。当前层执行时异步把下一层选中的冗余专家拷入 HBM slot，只预留两层的 slot。分配器权衡分流收益与 H2D 代价。DeepSeek-V3.1 671B 在 8×8 张 H100、混合公开数据模拟负载漂移时，batch 256 的平衡比 0.76，对比一份 GPU 冗余专家的 0.49；相对无均衡 SGLang 吞吐 +28.4%，batch 128 为 +12.4%。简单单专家逐层 offload 反而约 −10%。**建议考虑复现**。先验证真实热点漂移和 H2D/计算能否重叠。本文结论集中于 decode 与该 PCIe/NVLink 拓扑。 [固定全文](https://arxiv.org/html/2611.06057v1) · [作者实现](https://github.com/Flying-Cloud/vExpert)

#### 非前缀 KV 修复的小预算受检索与上下文兼容性限制

**11-04 20:02:58｜v1 首录｜§2–5，Table 1，Appendix B**

块独立编码后，作者把修复预算从全局 token 选择器转给含答案的块，或 BGE-M3 检索首块，再用当时的复用缓存重算。NQ-Open/Qwen3-8B、5% 预算，token 选择器的恢复比例 0.16，已知答案块 +0.35，检索首块 +0.22。这个指标以确实依赖检索信息的问题和原始差距计算。LoCoMo 的会话块常放不进 5% 预算，检索错块时移动和重算可劣于 token 选择器。**建议深入阅读**。复用系统的回归集可按首块命中和干扰块相似度分层，同时记录实际 TTFT。 [固定全文](https://arxiv.org/html/2611.06397v1)

### 评估与可靠性

#### DSSE-1 将后训练数据的意外行为副作用做成可测样本集

**11-04 18:57:28｜v1 首录｜§4–7，Figure 1、Table 5**

从预先定义的八类目标行为出发，搜索外观接近日常 SFT、却能诱发目标行为的合成数据。以 Qwen3-1.7B 做 LoRA，第一阶段 9,119 份候选由未参与搜索的评审模型重评，再结合行为分数和有效样本要求，留下 6,575 份。论文的 64 份数据多选实验中，ML 研究人员命中 63%，随机为 50%，GPT-5.6-sol xhigh 为 86%。这是指定候选行为的预测题，集合分数依赖评审模型，部分数据重训时未复现副作用。**建议深入阅读**。可用于比较数据审计器，但生产 SFT 还需测真实多轮场景和未预设的行为。 [固定 PDF](https://arxiv.org/pdf/2611.06324v1)

#### 上下文编码压缩改变拒答率，安全性要随压缩路径一起回归

**11-04 19:42:42｜v1 首录｜§3–5，Tables 1、3**

以同一 decoder 比较原始输入、定长 ICAE 或可变长度 ARC-Encoder。在作者生成的 held-out 集中，Llama-3.1-8B-Instruct 的有害提示拒答率经 ARC 从 96.15% 降至 36.54%。作者在定长 ICAE 记忆向量上按拒答方向做门控偏移。AdvBench 的 Mistral-7B 定长路径从 20.38% 提至 67.31%（偏移系数 1），与前述样本集分开计。缓解证据仅限定长压缩，拒答是所测的对齐行为。**建议考虑复现**。上线前让安全与无害成对样本都经过实际压缩路径，独立测漏拒与过拒。 [固定全文](https://arxiv.org/html/2611.06380v1)

#### Dafny 验证通过后，还要审证书是否约束了实现

**11-04 04:20:57｜v1 首录｜§2–6、9，Tables 3–5**

把代码生成分成 verifier 接受、外部规约语法绑定、规约或实现语义忠实三个层次。Claude Opus 4.7 复现 Li 等流水线，在 HumanEval 的 1,640 次尝试中，1,625 份经 Dafny 验证通过（99.1%）。通过样本里，只有 1.9% 通过确保条款绑定检查。在作者强制绑定的流水线上，VeriContest LC 仍有 14.82% 的下游语义缺口。绑定检查只识别特定语法结构，能漏掉恒真规约，主实验也只用一族模型。**建议深入阅读**。生成证书的回归可并列报告接受、绑定及隐藏测试或规约执行分数。 [固定全文](https://arxiv.org/html/2611.05590v1)

#### FrontierPhysics 给端到端物理研究任务加上专家复审

**11-05 02:56:22｜v1 首录｜§2–3，Table 2，Appendices E–G**

56 个任务来自九个物理方向，要求代理产出实验文件、结构化答案与论文式结果。整体通过需要确定性 verifier 和 rubric 的阻断项全部满足。v0.1 每题三条、每条 10 小时上限，各模型沿不同 CLI/harness 运行。最高配置 Opus 5.5 + Claude Code 为 46/168（27.4%）。对 GPT-6 Astra 的 168 条抽查中，专家找到九条原本错判失败。**建议深入阅读**。这些是模型加 harness 的组合成绩。若以类似题目做验收，应抽查 rubric 的假阴性，并分列工具、时间和成本。 [固定全文](https://arxiv.org/html/2611.06910v1)

### 版本与撤回

#### HyScale v3 增加 CSA 与 QSA 的百万 token 层级对照

**11-05 02:23:08｜v3 实质修订｜§3.3、4.4–4.5，Figure 10**

10 月 27 日日报已收录 v1 的 FlashTopK、选中索引重放与 GLM-5.1 训练。v3 将对照扩到块级稀疏注意力。按两份模型论文在 Megatron-LM 实现层，32 张 H800、CP32、1M context 下，DeepSeek-V4-Flash CSA 层为 2.01×，Qwen3.8-Flash-Next QSA 层为 1.57×。FlashTopK 的分项增益比 token 级 DSA 小，块级索引器本来就只给每个 query 约四分之一的 key 打分。**建议深入阅读**。这批新增数据是注意力层级消融，跨模型比较应分别核对选择、通信和稀疏 kernel 的分项收益。 [前版全文](https://arxiv.org/html/2610.32057v2) · [固定 v3 §4.5](https://arxiv.org/html/2610.32057v3#S4.SS5)

#### Cursor 的定向反馈论文补充与 Composer 2.5 的关系

**11-05 01:41:20｜v2 归因修订｜§1 脚注 1**

昨日收录的论文 v2 补充说明，公开论文实验以 Qwen3 为基座，Composer 2.5 使用过同类技术的一个版本，产品基座不涉及 Qwen 系列。**建议跟踪**。比较开源基座实验与产品发布时，应把代码通过率和反馈指标归到论文实验，把方法采用归到作者披露。 [前版 PDF](https://arxiv.org/pdf/2611.05266v1) · [固定 v2 §1](https://arxiv.org/html/2611.05266v2#S1)

#### 两篇旧稿标记为 arXiv 管理员撤回

**11-04 04:53:09 与 06:38:53｜v2 撤回状态**

LLM-as-a-Judge 的《Reliability without Validity》和权重合并的《Darwin Family》均由 arXiv 管理员撤回。登记页给出同一项 arXiv 内容质量标准理由。两篇旧稿此前未在日报收录，本次只登记状态。**建议跟踪**。引用时先核对撤回版本和后续是否有重新提交。 [Judge 论文状态](https://arxiv.org/abs/2606.19544v2) · [Darwin Family 状态](https://arxiv.org/abs/2605.14386v2)

### 论文来源覆盖

七个 arXiv 分类的 11 月 5 日新稿、交叉刊登和替换栏目显示数均已核对，共 905 个去重条目。回溯窗口的提交查询按这七个分类取齐 1,548 条元数据，最新可见提交时间为 11 月 5 日 02:59:58（UTC+8）。主窗口提交查询返回空集，更新时间查询被上游改写成提交时间查询。两项窗口的实际新提交和修订覆盖仍待补查。HF Daily 11 月 2–5 日可用，6 日 API 尚未开放。OpenReview TMLR API 返回 403，ACL、PMLR、NeurIPS 已查看目录或候选入口，精确发布日期和修订历史仍有缺口。 [arXiv 列表](https://arxiv.org/list/cs.LG/new?show=2000) · [HF Daily](https://huggingface.co/papers/date/2026-11-05) · [TMLR](https://openreview.net/group?id=TMLR) · [ACL Anthology](https://aclanthology.org/volumes/2026.emnlp-main/) · [PMLR](https://proceedings.mlr.press/) · [NeurIPS 2026](https://neurips.cc/Conferences/2026/Schedule)

<a id="core"></a>
## 开源工程 · 训推权重发布、提交和可见性

### verl 把权重快照发布和 rollout 提交分开，验证及退出等待最终提交

11 月 5 日合入 `main`。支持 `support_publish_weights` 的 backend 先发布不可变权重快照，trainer 在 `WeightUpdateTicket` 到达 published 后即可继续，rollout 的中止、释放 KV、接收、安装及恢复在后台完成。下一版本的发布等待上一版 committed，`separate_async` 的验证和训练结束也显式等待最后一版 committed，避免评估旧权重或退出时漏装最后一版。默认 NCCL/NIXL 路径继续在提交后返回。作者在外部 `tpu-sync` backend 上用 Qwen3-8B、两组 v6e-8、trainer FSDP8 / rollout TP8 跑 10 步，测得 trainer 阻塞时间 2.06 → 0.66 秒，剩余同步移到后台，每步范数一致。该数字衡量前台等待，生产 rollout 主动拉取的一侧读传输和独立调度仍在开放 PR #8526。接入新 transport 时，分别记录 published、committed 和权重版本，可验证训练、采样及评估消费的是哪一版。 [verl #8542](https://github.com/verl-project/verl/pull/8542) · [verl #8530](https://github.com/verl-project/verl/pull/8530) · [开放 PR #8526](https://github.com/verl-project/verl/pull/8526)

### slime 按 bucket 广播完整权重，并适配 Megatron Core 0.20

11 月 5 日合入 `main`。跨节点完整权重更新在 SGLang 的 `FlattenedTensorBucket` 支持混合 dtype、bucket 非空且调用未指定自定义 `load_format` 时，把参数打包为一个字节张量，每 bucket 一次 collective，接收端按原 dtype 和 shape 重建。其他格式走逐参数广播。作者用 Moore Threads S5000 和 Qwen3-30B-A3B 对比两种广播策略，训练与推理为 1+1 节点时从 7.186 降至 3.930 秒，2+3 节点从 15.437 降至 4.686 秒。自动选择的最终补丁有 90 项相关测试，PR 仍将其多卡部署验证列为待做。复测时宜分开计打包、传输及 rollout 安装开销。 [slime #2516](https://github.com/THUDM/slime/pull/2516)

同日修复 Core 0.20 对接中的两个首跑问题。slime 按新字段 `use_gloo_process_groups` 读取禁用 Gloo 的最终状态，对旧版本保留旧字段回退。Qwen3.5 mRoPE 接受 MCore 新传入的 `return_raw_freqs`，并对实际要求原始逐轴频率的 fused 路径显式拒绝，现有模型配置关闭该融合。以容器里固定的 Megatron `ae76c60e6` 为升级基线时，值得同时回归 Gloo 开关和多模态首个训练步。 [slime #2518](https://github.com/THUDM/slime/pull/2518) · [Megatron 依赖基线 #2514](https://github.com/THUDM/slime/pull/2514)

### vLLM 原地更新后重建 Kimi-K3 ROCm decode 的派生权重

11 月 5 日合入 `main`。KDA 的卷积及归一化副本、MLA 折叠权重位于 `state_dict` 之外，主权重被覆盖后可能继续被 decode 或已捕获 CUDA graph 读取。新 `refresh_derived_weights()` hook 接入 kernel-format reload、sparse NCCL 完成和 elastic-EP 新 worker 初始化，重建以原地写入保持地址。作者的直接回归覆盖 KDA kernel-format reload 的数值与指针不变，sparse NCCL 与 MLA 情形依赖代码路径，elastic-EP 接点为防御性改动，Kimi-K3 当前无法完成该扩容路径。热更新验收可对照同一权重版本的 prefill、eager decode 和 graph replay。 [vLLM #64886](https://github.com/vllm-project/vllm/pull/64886)

## 开源工程 · 训练并行和可复现实验

### Megatron-LM 动态 CP 进入混合模型，MTP token 归一化修复也覆盖静态 CP

11 月 5 日合入 `main`，合并提交 `8817e0fd` 是本轮截止点默认分支提交的祖先。DCP V2 的最后一部分让 `PackedSeqParams` 携带每个 packed microbatch 的运行时 CP group，供 HybridStack、Mamba、GDP、GDN、MLA/DSA 和 MTP 消费，CP1 构建的模型可处理依次到来的 CP1/2/4 microbatch。MTP 在有效 CP 组上先 SUM 聚合主损失和 MTP 的有效 token 数，再计算比例，修复不同 rank 的掩码数不一致时的梯度偏差，静态 CP>1 同样受益。每个 MTP 深度每 microbatch 多一次双元素 all-reduce，日志在 per-token 模式改为全局 token 加权。动态 CP 要求 per-token loss 与变长 packing，CUDA graph、确定性模式、router z-loss、CSA、DSA-GQA 和 DeepSeek-V4 在支持范围之外，Mamba 另有限制。作者报告部分回归，完整 GPU 矩阵仍需核实。比较旧曲线时应先区分梯度修复和日志聚合定义的变化。 [Megatron-LM #7048](https://github.com/NVIDIA/Megatron-LM/pull/7048) · [默认分支祖先关系](https://github.com/NVIDIA/Megatron-LM/compare/8817e0fd0a818b657bc80e8499caff408c3495ec...70f7a269ddf45b6f5b237aff6668fd1ddd79ef2d)

### Megatron-LM 修复分块 MoE 的 EP 调用数和 MFSDP 延迟梯度时序

11 月 6 日合入 `main`。序列分块时 `torch.chunk` 可能少于请求的块数，EP rank 遇到短 padding microbatch 会少发 all-to-all。MoE 改用固定次数的 `tensor_split`，短序列用 masked dummy 补齐，且 dummy 排除在路由统计之外。RouterReplay 按 microbatch 分片传入每个 chunk，最后拼回 top-k，选择性 checkpoint 重算也能拿到该 chunk 的索引。另一处修复为 meta 参数物化或分片时保留 TE 的 `skip_backward_post_hook`，让 MFSDP v2 等到 `backward_dw()` 后再规约。分块改动的作者报告 28 个 CPU 测试通过，GPU suite 待跑。延迟梯度的作者测试用 meta `te.Linear` 对照立即 wgrad，要求 5 步、每步 3 个累积 microbatch 的 loss 逐位相等。#7902 的更大改动另行拆分权重释放与梯度规约，截止检查仍开放。部署回归可同时覆盖极短填充批次、路由重放及 meta 初始化的延迟 wgrad。 [Megatron-LM #8463](https://github.com/NVIDIA/Megatron-LM/pull/8463) · [#8637](https://github.com/NVIDIA/Megatron-LM/pull/8637) · [开放 PR #7902](https://github.com/NVIDIA/Megatron-LM/pull/7902)

### Megatron-LM 为 GTP 数据 lane 分开 HSM 随机流，routed-only MoK 先进入 dev

11 月 6 日合入 `main` 的 #8996 让 MTP hidden-state mixing 按 `dp_rank * gtp_remat_size + gtp_remat_rank` 初始化 RNG。GTP 重算组内各 lane 消费不同 microbatch，旧的纯 DP rank seed 令它们抽到相同 mixing randomness，TP/CP 仍同步使用同一流。作者的新 per-rank 测试通过，4 卡套件有环境不匹配失败，建议将 GTP 有无分别做重放对照。当天 #8497 合入 `dev`，让 MoK 仅负责 routed experts，MCore 保留 shared experts、latent projections 和原生精度处理，另接 SiTU-GLU 与 TP+SP、ETP=1。作者报告 64×GB200、TP4/EP64/CP16 的 9 层 Kimi-K3 MXFP8 proxy 验证，仍需配套 MoK 改动并跟踪主线集成。 [Megatron-LM #8996](https://github.com/NVIDIA/Megatron-LM/pull/8996) · [dev PR #8497](https://github.com/NVIDIA/Megatron-LM/pull/8497)

### TorchTitan 修复共享专家的批次不变性，DistMuon 在更新后释放 GPU scratch

11 月 5–6 日合入 `main`。当 TP 上叠加 EP、无序列并行且启用 batch-invariant 模式，若先在本地把 shared 和 routed 输出相加再规约，同一个 token 随 batch 改变 owner 后会改变浮点求和顺序。#5829 先分别规约 shared partial 和 zero-fill 的 routed 输出，再相加，代价是一次额外 TP all-reduce。作者在 2×GB300 的 TP2/EP2 debug 模型上，DeepSeek V3 和 Kimi K3 的 trainer logprob 最大差异分别从 3.8e-3、4.6e-2 降为 0。#5677 则在 eager optimizer update 后将 accelerator scratch 的 storage 缩为零，保留 tensor 元数据、CPU scratch 和流/事件，供后续常规 allocator 重用。单卡 GB300 合成测试中两个 eager optimizer 的 peak allocated 为 984 → 696 MiB，独立 CUDA graph 池没有同样的复用保证，最终版尚缺新鲜的多 rank NCCL 回归。前者适合定位 rollout 数值差异，后者可按 eager、VPP 和 graph pool 分别量显存。 [TorchTitan #5829](https://github.com/pytorch/torchtitan/pull/5829) · [#5677](https://github.com/pytorch/torchtitan/pull/5677)

### verl 补齐 397B 和 1.6T 两种全参 GRPO 运行配方

- 11 月 5 日合入 `main` 的 Qwen3.5-397B-A17B 配方使用 64 颗 TPU v7x，其中 32 颗跑 TorchTitan trainer（eFSDP8×EP8、BF16 参数、FP32 主权重、AdamW host offload），另 32 颗分成四个 vLLM rollout 副本。8K 上下文配置为 512+7680，GBS64、每样本 16 个 rollout。作者第 2–18 步平均 130.6 秒，trainer MFU 约 3.8%，trainer HBM 峰值 94.7/101.7 GB 每设备。配方固定外部 plugin `0bc8bbac` 和 TorchTitan TPU fork 的历史提交 `4e16e67ae`，Ring-of-Experts PR #142 本轮仍开放，CI 的 v6e 小模型测试与 397B 实验是两组证据。复跑先固定镜像、静态 shape 和 EP 提交。 [verl #8469](https://github.com/verl-project/verl/pull/8469) · [TPU TorchTitan #142](https://github.com/google-pytorch/torchtitan/pull/142)
- 同日合入的 DeepSeek-V4-Pro 例程为 61 层、约 1.6T 参数提供 64×8 H100（PP32×EP8）与 64×4 GB200（PP16×EP16）两套拓扑。CPU optimizer offload、BF16 梯度缓冲和分布式 rollout 是显存调节项。作者提供三步级全模型结果，H100 约 672–700 秒稳态步时，GB200 约 960–972 秒，拓扑、内核和基线不同，宜按各 profile 独立预算。Megatron-Bridge 0.6.2 基线上还需量化参数 detach 及 mixed-dtype norm/clip 等 Megatron 补丁，两个对应主线 PR 仍开放。复现应固定补丁分支，并分别核对 stage 0 的主机内存和更新耗时。 [verl #8383](https://github.com/verl-project/verl/pull/8383) · [Megatron #8245](https://github.com/NVIDIA/Megatron-LM/pull/8245) · [#7279](https://github.com/NVIDIA/Megatron-LM/pull/7279)

### slime 加入 Nemotron 3.5 Super VL 的多模态 GRPO 路径

11 月 6 日合入 `main`。88 层 Mamba2/MoE/attention 混合解码器和 C-RADIO 视觉塔经 MCore HybridModel 接入，提供 HF↔Megatron 权重转换及 colocated SGLang rollout，同步涵盖 MTP、hybrid routing replay 和 packed-sequence Mamba state 修复。作者在 4×8 H100、TP2/EP32 的 GEO3K 配方报告 step 0 的 train/rollout logprob 绝对差 0.0143，评估从 0.42 到 0.71。该结果来自仓库所带的 Megatron/SGLang 补丁与单一任务配方，可从转换单测和配方 README 逐段验证模型结构与权重同步。 [slime #2471](https://github.com/THUDM/slime/pull/2471)

## 开源工程 · 推理缓存、稀疏注意力与图回放

### vLLM 的 KV 页默认 16→64，启动映射和 P2P reset 分别加固

- 11 月 5 日合入 `main` 的 #63231 将通用 KV block 默认改为 64 token，显式参数和 backend 约束仍参与选择，ROCm 原生 paged attention 默认保留 16。作者报告 171 个 CI job 通过，初次 CI 暴露 offload chunk 对齐及 MI250 回退路径问题，未提供 16/64 的实盘服务对比。上线时可用相同输入长度分布对照 prefix 命中、尾块浪费与吞吐。 [vLLM #63231](https://github.com/vllm-project/vllm/pull/63231)
- 11 月 6 日合入 `main` 的 #65554 为无 RDMA connector 的可扩展 KV 也做 segment 对齐，并给 block-outermost 布局在分配时提交第二块。FlashInfer autotune 在 cache 扩展前调用 sparse FlashMLA，padding gather 的越界预读原可触到未映射显存。作者在 8×B300、DeepSeek-V3.2 FP8、EP+TP2+DP2+PP2、空 autotune 缓存的启动测试由 3/3 崩溃转为通过，另报告 70 项回归通过。重现需保留空缓存和同类布局。 [vLLM #65554](https://github.com/vllm-project/vllm/pull/65554)
- 11 月 5 日合入 `main` 的 #65124 在 reset drain 时主动发送 AbortFetch 并立即报告作业失败，slot 等 AbortAck 或会话关闭再释放，避免静默 prefiller 把清缓存、sleep 或更新权重的引擎线程拖到完整 load timeout。默认 ACK 等待上限 5 秒，作者报告 463 项相关测试通过。旧的 consumer slot quarantine PR #61667 仍为关闭未合入，关联 issue #61532 保持开放，故此处证据限定为 reset 的主动中止和等待路径。 [vLLM #65124](https://github.com/vllm-project/vllm/pull/65124) · [#61667](https://github.com/vllm-project/vllm/pull/61667) · [issue #61532](https://github.com/vllm-project/vllm/issues/61532)

- 11 月 6 日合入 `main` 的 #65539 按 PP×PCP×TP 的真实 worker 布局，以 `rank // (tp_size * pcp_size) % pp_size` 推导 PP rank，修复 MooncakeStore 键、stage 层范围和 Ray actor 命名。作者在 4×B300、DeepSeek-V2-Lite-Chat、PP2×PCP2×TP1、14/13 非均匀分层、关闭本地 prefix cache 的两次 GSM8K 5-shot 跑测中，第二轮外部命中率 0.0% → 97.7%，分数仍在观测波动内。回归组合并行可先在相同进程拓扑上核对 connector 键及层范围。 [vLLM #65539](https://github.com/vllm-project/vllm/pull/65539)

### vLLM 稀疏 MLA 跨层共享与 ShardQ 切到正确的数据所有者

11 月 5 日合入 `main`。#64095 让 `AttentionLayerBase` 声明共享目标，V2 runner 让消费层引用来源层的 KV spec 并跳过自写。共享目前限 sparse MLA，并强制 MQA prefill，因为 sparse MHA prefill 会取本层 latent，dense MLA 也会拒绝；PCP+DCP 的 decode LSE merge 则移到可由模型 opt-in 的 MLA 基类。作者的端到端检查使用 8 层随机权重 DSA 模型和 TP2，应按具体模型回归跨层一致性。#63956 把 DSA prefill 从按 head 切换为按 token 在 TP rank 间做 zigzag 分片，混批的 decode token 仍按 head 切分，最终合入的默认门槛为 prefill 1024 token。作者对 GLM-5.3 NVFP4、TP4、单并发的 B300 测试报告 1M 输入 TTFT 162.5 → 62.0 秒，PR 同时注明性能/准确率数据里有结构重写前、使用 8192 门槛的批次。采用时宜在最终提交上重跑不同 prefill budget 与混批，区分 TTFT 与 decode 指标。 [vLLM #64095](https://github.com/vllm-project/vllm/pull/64095) · [#63956](https://github.com/vllm-project/vllm/pull/63956)

### vLLM DeepEP V2 显式标记 padding，Rubin 新增原生 LUT-B MoE

- 11 月 6 日合入 `main` 的 #54109 以 `PaddedStandard` 区分 DeepEP V2 token buffer，decode CUDA graph 的接收容量按最坏情形分配，未写尾部可能含任意数据，prefill 又存在 expert-alignment 间隙。显式格式要求 expert kernel 识别 `topk_ids=-1` 或设备侧真实范围，仅选择支持 padding 的实现。复核自定义 experts 时，应给尾部填充非零垃圾，并验证真实 token 与紧凑输入一致。 [vLLM #54109](https://github.com/vllm-project/vllm/pull/54109)
- 同日合入的 #62759 以 3-bit LUT 索引和每 8×64 tile 的指数压缩 routed experts，激活用 MXFP8。Rubin SM107 走原生 grouped GEMM，其他设备走预解码权重并模拟激活量化的 Triton 路径。作者单 Rubin、CUDA 13.4 测得 GLM-4.7-Flash 加载显存 56.3 → 15.9 GiB，GSM8K 5-shot BF16 0.857、模拟 0.843、原生 0.839（各约 ±0.010）。模拟可验数值，原生存储与算力收益需在 Rubin 上单独量。 [vLLM #62759](https://github.com/vllm-project/vllm/pull/62759)

### SGLang 修复分离式推理的 KV 发布、混合缓存边界和 compact verify

- 11 月 5–6 日合入 `main`。#48452 让先于 decode bootstrap 完成又让出资源的 optimistic prefill，在释放 request-owned KV 前写入 radix tree，write-through 时也备份，重排请求可沿 L3 baseline 恢复已经算过的前缀。#48421 则要求包含 SWA/Mamba 组件的 L3 hit 必须完整分配，部分 host 容量无法保证缩短位置有对应状态，容量不足走 revoke/retry。前者有强制 write-through 的单测，后者的作者证据为 CPU 隔离回归，GPU 准确率待验。对应两个不同测试应分别覆盖 peer bootstrap 延迟与 host 池压力。 [SGLang #48452](https://github.com/sgl-project/sglang/pull/48452) · [#48421](https://github.com/sgl-project/sglang/pull/48421)
- 11 月 6 日合入的 #42718 用 packed layout 的 `qo_indptr` 固定长度查找 token 属于哪条请求，统一提供给 DeepSeek-V4.1 的 Engram 哈希和请求池索引。原代码假定每请求相同长度，遇到 42 token/8 槽等 compact DSpark verify 档位会在捕获阶段报错或映射到错误请求。作者报告 CPU 及 RTX PRO 6000 Blackwell replay 回归通过，覆盖空行、尾部 padding 与 dense 对照。使用 compact graph 时应验证 replay 更新的是同一份 staged layout。 [SGLang #42718](https://github.com/sgl-project/sglang/pull/42718)

### SGLang top-k 精确性成默认，Kimi DCP 精确合并仍处于逐步移植

11 月 5 日合入 `main`。#47960 让 DeepSeek-V4 top-k v2 默认采用 exact 模式，cluster 阈值 bin 溢出时通过细直方图再筛选，修复截掉候选导致的错误 top-k 值。deterministic 模式进一步按最小索引处理平分，fast 保留可选，NaN 只保证合法输出。作者在 B300、CUDA graph、108 个真实 indexer shape 上，exact 相对主线整体中位数快 3.5%，Streaming 类中位数慢 0.8%，不能从 kernel 推到整机收益。#48588 另为 Kimi-K3.1 DCP 合入 NVLink multicast 的跨 rank top-k 候选合并，先发 384 候选，证书不足再读取完整本地 top-k。8×B300 测试有 5 项通过，16-rank 路径未测，合入时没有运行时 caller，#48721 仍开放。验证集可优先放入真实分数溢出、平分及单 rank 占据大部分 top-k 的样例。 [SGLang #47960](https://github.com/sgl-project/sglang/pull/47960) · [#48588](https://github.com/sgl-project/sglang/pull/48588) · [开放 PR #48721](https://github.com/sgl-project/sglang/pull/48721)

### SGLang 修复 ROCm 页表静默错误，昇腾接入跨请求 ViT 批处理

11 月 5 日合入 `main`。#46928 修复 ROCm AITER unified attention 在 dense 模型、page size 1、CUDA graph 下未拷贝 page table 的问题，decode 原会读全零页表。作者在单 MI355X、Llama-3.1-8B-Instruct、200 道 5-shot GSM8K 上三轮由 0.00 恢复到 0.755–0.765，page size 16 参考为 0.765，MI300X CI 也通过该用例。#44435 让符合条件的 NPU 请求把图像 embedding cache misses 跨请求合批，给图捕获输入和返回 embedding 必要的独立存储，防止后续 replay 覆盖缓存；作者有 Ascend 910 的图像顺序、缓存复用与 eager/graph 对照，音频及其他 VLM 仍需单独验证。迁移时关注图模式、缓存 miss 以及小 page 这些容易从常规部署样例中遗漏的条件。 [SGLang #46928](https://github.com/sgl-project/sglang/pull/46928) · [#44435](https://github.com/sgl-project/sglang/pull/44435)

### TorchTitan 为 decode 单独捕获 FULL graph，并修复 rollout 停机顺序

- 11 月 6 日合入 `main` 的 #5704 为 uniform decode 使用独立 graph key，将 FA4 的 `max_seqlen_q` 从混批图的 token 档位降到 1，并使 DP rank 在 peer 有 prefill 时共同回到混批图。#5490 让 GDN/KDA 在符合条件的 DP>1 下保留 FULL prefill/mixed graph，batch-invariant 分支仍约束图覆盖。作者测得 Qwen3-8B TP2、2K context、batch 128–1024 decode step 低 22–30%，并报告 Qwen3.5-35B-A3B DP2×EP2 的 prefill 改善。当前 hybrid DP>1 示例的 MoE dispatcher 仍走 eager，实际采用应按 dispatcher 和图模式分开测。 [TorchTitan #5704](https://github.com/pytorch/torchtitan/pull/5704) · [#5490](https://github.com/pytorch/torchtitan/pull/5490)
- 11 月 5 日合入的 #5292 在控制器关闭时先停 generator、完成或失败化其请求，再等 worker 上已提交的 rollout group，最多 30 秒后停止 worker mesh。旧顺序会让 Monarch 0.7 向已停止 worker 回送回复而打断整个 cleanup。作者在 8×H100 的 Qwen3-4B DAPO-Math 3 步测试，失败退出由 2/3 轮变为 0/3，且该改动覆盖 worker 执行路径。11 月 6 日合入的 #5237 另修正 Terminal-Bench 的评分时序，把 tmux 清理移到评分之后，避免 agent 用 `&` 启动的服务在 verifier 检查前被杀。评估稳定性应把控制器退出和 sandbox teardown 分开回归。 [TorchTitan #5292](https://github.com/pytorch/torchtitan/pull/5292) · [#5237](https://github.com/pytorch/torchtitan/pull/5237)

## 开源工程 · 新模型接入和发布工件补录

- **vLLM 接入 Ripple-65B Preview**。11 月 6 日合入 `main`，提供 BF16/NVFP4 的原生模型类和 reasoning/tool parser，以及可选 DFlash2。draft 通过 `no_rope_layers` 让 NoPE 层使用未旋转的 context key，作者在单 B300、NVFP4 TP1、GSM8K 5-shot、15 个 speculative token 下测得平均接收长度 4.98 → 7.34，使用 draft 前后该项得分均为 0.916。它衡量该任务的接收而非真实负载提速，适合先复核目标 checkpoint、draft 和 parser 组合。 [vLLM #65745](https://github.com/vllm-project/vllm/pull/65745)
- **补录 11 月 2 日的 vLLM 0.33.0 正式包**。11 月 3 日日报已记录 GitHub Release 对象使用 `v0.33.0rc3` 标签并留待确认正式工件。本轮打开独立的 annotated `v0.33.0` tag，确认它和 rc3 都指向 `4ba787ec9959e566d3b587a796dff07673324da5`。PyPI 正式版本 `0.33.0` 于 11 月 2 日 18:12–18:13（上海）上传 sdist 和 Linux x86_64/aarch64 wheel，三件均未 yanked，元数据固定 `torch==2.14.0`。GitHub 正式 tag 的 Release API 返回 404，容器镜像工件仍待核验。部署可按正式 tag 和 PyPI 文件锁定版本。 [GitHub rc3 Release](https://github.com/vllm-project/vllm/releases/tag/v0.33.0rc3) · [v0.33.0 tag](https://github.com/vllm-project/vllm/tree/v0.33.0) · [PyPI 0.33.0 文件](https://pypi.org/project/vllm/0.33.0/#files)

verl 维护分支另有一处可定位的修复。11 月 5 日 #8509 合入 `release/v0.9.1`，仅在启用 KL-in-reward 时取 `token_level_rewards`，避免未生成该字段的 PPO 指标路径报错。该分支提交需与已有 v0.9.1 tag 分开追踪。 [verl #8509](https://github.com/verl-project/verl/pull/8509)

工程覆盖按上海时间标注事件日期。GitHub 主窗合入清单与默认分支提交分别交叉核对，条目中的测试和测速均为上游记录。已合入的代码按对应 PR 或 commit 定位。六个项目的近期公开 Release 列表在主窗没有新发行对象，SGLang v0.5.23 已在上期窗口覆盖，其他 tag、镜像和分发渠道按条目核验。高价值候选核查了原始补丁，完整语义复核和历史积压仍有待续查。

<a id="extended"></a>
## 扩展开源工程

### LMCache v0.5.7 将多进程传输和 KV 分组能力带入正式版

**2026-11-05T23:41:01Z · LMCache/LMCache v0.5.7**

在正式版中可选用 `grpc.aio` 或原生 gRPC 传输，并支持 Unix socket。engine-driven 路径可以传多组 KV，包括 DeepSeek V4 压缩组。版本还纳入 Device-DAX/CXL L1 偏移、MP 路径的 Trainium/Neuron 适配，以及对 MP 和前端 HTTP 端点扩大的 API key 覆盖。官方镜像改用非 root 运行。发布说明以功能和修复为主，未提供统一的跨后端性能对比。采用前应验证持久卷权限、鉴权和实际模型的 KV 布局。 [LMCache/LMCache v0.5.7](https://github.com/LMCache/LMCache/releases/tag/v0.5.7) · [压缩 KV 组](https://github.com/LMCache/LMCache/pull/6124) · [原生传输](https://github.com/LMCache/LMCache/pull/6031)

### LMCache 将 MP 缓存键按 LoRA 身份隔离

**2026-11-05T19:41:34Z · LMCache/LMCache #5713**

启用 LoRA 时，相同 token 和 `cache_salt` 可让 base 与不同适配器误命中彼此的 KV。补丁使用适配器名称和路径的 128 位 BLAKE2b 摘要构造内部 `lmcache.tag.lora`，覆盖客户端自带的同名 tag，同时保留用户的 salt。lazy offload 合并 STORE 元数据时也保留 request config，使读、写和释放锁使用同一身份。值得用同 prompt 交替请求 base 与多个 LoRA 做回归。这个变更属于默认分支提交。 [LMCache/LMCache #5713](https://github.com/LMCache/LMCache/commit/ea6ecd0a3e81768dd40be159422d095720138b21)

### LMCache 为 vLLM VMM sleep 和唤醒补上注册生命周期

**2026-11-06T00:40:08Z · LMCache/LMCache #6358**

MP server 导入物理 KV 页会延长其寿命。worker 现在在释放前等待已跟踪传输、注销注册并关闭导入，唤醒后重建注册，使相同虚拟地址的新物理页可见。设计文档明确写出超过 `mq_timeout` 后仍会继续注销，恢复时 server 不健康则由后续心跳补注册。适合复现 sleep、在途拷贝和 server 重启的组合，确认释放量及错误行为。 [LMCache/LMCache #6358](https://github.com/LMCache/LMCache/commit/012965f888cccaa5102cec7a745990f60e8ca721)

### Dynamo v1.6.2 修正 TRT-LLM 超长请求的错误分类

**2026-11-05T22:11:38Z · ai-dynamo/dynamo v1.6.2**

对超过引擎 `max_num_tokens` 的非流式 `/v1/messages` 请求，TensorRT-LLM 现返回 HTTP 400 并保持 worker 运行，之前返回 500。SGLang decode 与聚合 worker 可选择启动后 freeze GC，默认关闭。维护者建议在启用前比较长跑内存。发行说明未给出通用流式停顿收益，升级时应同时验证客户端重试策略与 worker RSS。 [ai-dynamo/dynamo v1.6.2](https://github.com/ai-dynamo/dynamo/releases/tag/v1.6.2) · [超长请求修复](https://github.com/ai-dynamo/dynamo/pull/16193)

### FlashAttention 2 阻止左填充 KV 走无尾块掩码内核

**2026-11-05T17:54:21Z · Dao-AILab/flash-attention #3092**

此前在名义 Q/K 长度按块对齐且缺少显式 cache lengths 时，`leftpad_k` 可令实际 K 长度不对齐，`Is_even_MN` 路径仍跳过最后一个 K 块的掩码，软最大值可能读入下一序列的缓存行。前向及 split-KV dispatch 对 `leftpad_k` 或 `seqused_k` 使用尾块安全路径，并增加左填充回归。使用 `flash_attn_with_kvcache` 的 serving 路径应优先查这组输入。 [Dao-AILab/flash-attention #3092](https://github.com/Dao-AILab/flash-attention/commit/34a06b99d60ea26bfce91432c9fbb6de1a1b94ec)

### FlashInfer FA2 修正相等大最大值时的历史重缩放

**2026-11-05T23:41:18Z · flashinfer-ai/flashinfer #6908**

FP32 与 VO-split prefill 先计算历史最大值与新最大值之差，再乘 softmax scale。两者相等时历史权重应为 1，空历史仍单独置零。维护者构造的极值输入原来得到 128 个 NaN，修复后得到预期单位输出。适合加进有限极值回归，覆盖范围限于补丁涉及的 FA2 路径。 [flashinfer-ai/flashinfer #6908](https://github.com/flashinfer-ai/flashinfer/commit/e5eda939f93dbc527bb22bd7d9e8658f1ebd6f5f)

### FlashInfer 主干新增 batch-invariant FP8 BMM

**2026-11-05T19:16:07Z · flashinfer-ai/flashinfer #8453**

新增 `bmm_fp8(batch_invariant=True, backend="auto")`，固定与 batch/M 无关的归约类别，调度器只在保持同一结果语义的程序之间排序。支持 SM100/103/107、两输入同为 E4M3 或同为 E5M2、标量 FP32 scale，K/N 分别对齐 16/8，输出 BF16 或 FP16。值得在 rollout 中比较单条与混批的逐位一致性及吞吐，结论以该算子和输入契约为界。 [flashinfer-ai/flashinfer #8453](https://github.com/flashinfer-ai/flashinfer/commit/6ff41626ee40fc1021504f163d7e9d8c6880e4ee)

### FlashInfer 实验 NVFP4 MLA wide route 保留小 V scale

**2026-11-05T21:05:23Z · flashinfer-ai/flashinfer #8837**

旧 wide kernel 固定把 V scale 除以 8，低值可下溢为零，另有 scale 字节解释为 NaN。新路径按 token 选缩放幂并在 P 上补偿。维护者对 GB300/GB200 的 wide 路径微基准比较新旧完整调用，中位延迟代价约 0.7%–7.1%，最小差异处于约 1% 的重建波动内。仅涉及 packed token/head 行数大于 48 的实验性路径，可复现低值动态范围与延迟的权衡。 [flashinfer-ai/flashinfer #8837](https://github.com/flashinfer-ai/flashinfer/commit/5a94f96ec0d0d4a909fcfefaf23ada0574ef6551)

### PyTorch ROCm 限制 BLAS handle 的跨 stream 复用

**2026-11-05T22:08:12Z · pytorch/pytorch #202678**

旧线程退出时会立即回收仍有排队工作的 handle。不同 stream 的 hipBLASLt 单内核 split-K 可能共享其同步标志，引发死锁或错误结果。补丁把用过的 handle 放回原设备与 stream 的池。PR 提供 MI300X、ROCm 7.0、hipBLASLt 1.2.70002 的生产卡死和 GSU=2 FP16 最小复现。适合为多线程推理及图捕获检查线程退出、外部 stream 和分流复用。 [pytorch/pytorch #202678](https://github.com/pytorch/pytorch/commit/cbd69365b9516c0f819385223c6c5f72626e93b1)

### PyTorch GDS 增加 stream-ordered I/O 并收紧同步 API

**2026-11-05T20:42:39Z · pytorch/pytorch #197571**

`load_storage` 和 `save_storage` 在调用 cuFile/hipFile 同步 I/O 前等待当前 stream，并拒绝 graph capture。新异步 API 把读写提交到 stream，补丁也处理 ROCm 的接近 2 GiB 单请求上限导致的截断。回归覆盖 API 顺序和兼容模式，兼容模式经 POSIX 与 bounce buffer，吞吐需单独测。使用 GDS 写 checkpoint 的代码应验证读写顺序、错误传播和大存储分块。 [pytorch/pytorch #197571](https://github.com/pytorch/pytorch/commit/9bb2836683cf1294e6077f5fb3e5a419dec665c2)

### Transformers 修正单副本 expert FSDP 的梯度缩放

**2026-11-05T03:09:22Z · huggingface/transformers #50292**

使用 token dispatch、`efsdp_size == 1` 时 FSDP2 不执行 reduce-scatter，旧路径在复制梯度时除以 FSDP size，强制 sum 逻辑又重复除一次。现在只在实际有 reduce-scatter 时强制 sum。新增 BF16 expert 梯度回归，适合对照 FP32 或单卡参考验证 EP/FSDP 布局。 [huggingface/transformers #50292](https://github.com/huggingface/transformers/commit/b3187c7faec959b0e5facb40424e68441095adca)

### DeepSpeed 打开 AutoEP × AutoTP folding 的 ZeRO-3 CPU optimizer offload

**2026-11-05T23:27:14Z · deepspeedai/DeepSpeed #8829**

在专家和非专家参数按各自副本组切分的前提下，新增经过覆盖的 ZeRO-3 CPU 优化器卸载组合。配置允许显式 `device: "none"`。参数卸载、NVMe optimizer offload、ZeRO-1/2 optimizer offload 和 Muon offload 仍被拒绝。适合在原先需要常驻 GPU 优化器状态的拓扑上复现数值一致性与 step time。 [deepspeedai/DeepSpeed #8829](https://github.com/deepspeedai/DeepSpeed/commit/f47dc3e3d48e748d699720ea65eca1f45c136255)

### DeepSpeed ZeRO-2 清除 offload scratch 中残留的溢出

**2026-11-05T02:53:04Z · deepspeedai/DeepSpeed #8728**

当低精度 master 只恢复本 rank 的梯度片段时，整段溢出检测可能读到其他参数组残留的 NaN，导致后一有限步也被跳过。补丁在部分复制前清零目标 view。测试使用 DP=2 和两个参数组，要求一次真实 NaN 后连续有限步正常更新。值得将其纳入 CPU offload 恢复训练的回归。 [deepspeedai/DeepSpeed #8728](https://github.com/deepspeedai/DeepSpeed/commit/0d0aa30b7b9c5d40fafa470f5d8be370bc9ab9c2)

### TRL 为实验异步 GRPO 公开 rollout 扩展点

**2026-11-05T15:44:52Z · huggingface/trl #8096**

`AsyncRolloutLoop` 允许在独立 rollout 进程中定制生成、奖励、优势和过滤，通过 Trainer 子类的 `loop_cls` 选择。类要能在 spawn 中导入，异步资源在 `setup` 中建立。全组同奖励仍按奖励丢弃，过滤也计入 epoch。适合复现自定义环境或 curriculum 时审计训练样本的实际分布与 worker 生命周期。 [huggingface/trl #8096](https://github.com/huggingface/trl/commit/3a68e829d6ae7d8ec69cd09a7be3eb38c60e37d0)

### TensorRT-LLM 为 Helix prefill 收集 MLA 历史 KV

**2026-11-05T18:20:13Z · NVIDIA/TensorRT-LLM #21527**

原路径在 Helix context 布局遇到 cached tokens 时受限。补丁按 chunk 计算各 CP rank 的局部长度和偏移，对局部缓存作变长 gather，再由 reorder kernel 恢复全局 token 顺序。新增重排和多 GPU 用例。可用混合命中长度验证结果，再量化通信和暂存开销。 [NVIDIA/TensorRT-LLM #21527](https://github.com/NVIDIA/TensorRT-LLM/commit/57968941f191eb6015e0e0fc586ff5d06313d42d)

### vLLM Ascend 修正 GLM-5.3-Flash FP8 跳过层失配

**2026-11-05T08:59:03Z · vllm-project/vllm-ascend #20861**

checkpoint 的平面层名遇到多模态 wrapper 增加的 `language_model.` 前缀后无法精确匹配，部分应保留精度的 sparse indexer 投影被量化，PR 报告其查询向量为零。修复在配置匹配失败后补做后缀检查并发出一次日志。值得复现同一权重、硬件和评测配置下的 indexer 输出及模型质量。问题报告的端到端分数属于提交者结果。 [vllm-project/vllm-ascend #20861](https://github.com/vllm-project/vllm-ascend/commit/6adbb114ccd132a959898e6201503c5457f3e5cc) · [问题与复现](https://github.com/vllm-project/vllm-ascend/issues/20860)

### vLLM Ascend 保持 DeepSeek V4.1 KV 分配与运行时元数据一致

**2026-11-05T06:27:55Z · vllm-project/vllm-ascend #20826**

构建模型时 prefix cache 或 chunked prefill 可将块设为 128，旧 backend 钩子在未显式指定 `--block-size` 时再写入 16，后续 MLA 元数据据此计算错误稀疏坐标。修复保留模型建立时使用的块大小。优先复现未指定参数和显式指定参数两组输出。 [vllm-project/vllm-ascend #20826](https://github.com/vllm-project/vllm-ascend/commit/0f9d627996b3be5c67bde3d74a098c836364b3d9)

### vLLM Ascend 在 DFlash 缓存写入前旋转 context K

**2026-11-05T08:26:21Z · vllm-project/vllm-ascend #20002**

Ascend RoPE 返回新 tensor，旧适配按上游 in-place 假设丢弃返回值。修复令 draft cache 使用返回的已旋转 K，同时保留 310P 需要的兼容实参。维护者在 Qwen3.8-27B + DSpark 观察到接受长度下降，补丁给出旋转相关测试，未给出可推广的加速数值。适合对比 draft 接受长度及最终一致性。 [vllm-project/vllm-ascend #20002](https://github.com/vllm-project/vllm-ascend/commit/3fa30d2869b62a5354e92aeae921de01c0c78a56)

### vLLM Ascend 对齐 native KV offload 的调度与注册组

**2026-11-05T08:38:01Z · vllm-project/vllm-ascend #14421**

hybrid cache 的部分组不允许 offload。按所有组注册会与 scheduler 的 group sizes 不一致。补丁用一次有序选择驱动布局适配、tensor 注册与 canonical refs。PR 引述 DeepSeek-V4.1-Flash-w4a8、4 物理 NPU/8 die、TP8、vLLM 0.32.0 的启动验证，迁移时还需覆盖实际换入换出和混合模型。 [vllm-project/vllm-ascend #14421](https://github.com/vllm-project/vllm-ascend/commit/f1a99b9617e0528e0471c9e29a5ce6fa23672643)

### vLLM Ascend 声明 MLA 与稀疏索引 KV 在 TP 间复制

**2026-11-05T01:46:09Z · vllm-project/vllm-ascend #20766**

在 MLA 与 sparse indexer cache spec 中写入 `max_tp_shards=1`，让现有共享 CPU tiering 识别完全复制的块。在 TP8 上，旧布局对相同块可占八份 CPU 空间。适用条件是单机多进程、共享区和真正复制的缓存组。建议核对每种缓存组的 host 分配，来源给出合约测试而无端到端吞吐。 [vllm-project/vllm-ascend #20766](https://github.com/vllm-project/vllm-ascend/commit/e04a09812d3cb8b0487b0e82dd51ba2eea6d7335)

### AITER MegaMoEV2 在 gfx950 合并 EP8 dispatch 到 combine

**2026-11-05T08:23:52Z · ROCm/aiter #6881**

DeepSeek-V4 A8W4 的 opt-in small-M 内核还融合量化，large-M 采用持久内核并单独量化。两路限制为 SiLU，small-M 最多 512 token/rank，修复覆盖无 token rank 和无效 top-k 槽。可在相同并行度下分均匀和偏斜路由复现，单内核落地不等于所有 shape 的整机吞吐提升。 [ROCm/aiter #6881](https://github.com/ROCm/aiter/commit/a060781a9efa8ec5d18925e25113d3056099504b)

### AITER 修正 Triton split-KV padding 行的并发写

**2026-11-05T14:48:29Z · ROCm/aiter #7472**

当每 KV head 有 3 个 query heads 等场景出现未用行，尾行会映射到下一 q block 的首 token，两个程序可能覆盖其 segment max/exp sum，使输出跨运行变化。补丁屏蔽这些存储，测试重复 prefill 的逐位一致。值得在非整除 GQA 和因果尾块的长序列路径复验。 [ROCm/aiter #7472](https://github.com/ROCm/aiter/commit/656e98e68b499dca5748cd87343d9f3e797bc049)

### ATOM 合并 Qwen3.5 GDN 的窄 FP8 投影

**2026-11-05T07:16:03Z · ROCm/ATOM #2738**

同一 per-token FP8 配方的 `in_proj_ba` 与 `in_proj_qkvz` 可合并，复用激活量化并避免 32 列小 GEMM 的 tile padding。PR 以 TP4、batch 1、FP4 路径历史 trace 估算约 11.8 µs/层、45 层约 0.53 ms/迭代，属于开销拆分估算。block-scale checkpoint 不满足这个对齐条件。应在同样的模型、量化和 TP 配置复验。 [ROCm/ATOM #2738](https://github.com/ROCm/ATOM/commit/30d2ad88f93bc2808f37f5f1c3f98ee0370a1fd8)

### ATOM 补齐稀疏 decode 的 CUDA Graph padding 元数据

**2026-11-05T02:49:12Z · ROCm/ATOM #2992**

TR_HYBRID 用上一档 graph batch 回放，旧代码只为实际调度 batch 构建 sparse block table 和 seq_lens。修复覆盖 padded batch，避免例如 65–127 个请求回放 128 档位时读取未初始化页表。适合用阶梯间请求数做压力测试，尤其是增减并发时。 [ROCm/ATOM #2992](https://github.com/ROCm/ATOM/commit/ab4ada40a2d9ffe1fb1cc50f95d85ce63ce92619)

### TorchTPU 修正 3D 激活 checkpoint 的边界与峰值显存

**2026-11-05T20:17:17Z · google-pytorch/torch_tpu 594e433d43**

旧分区可能漏标 3D Linear 尾部 view 的 producer，反向开头重算全部区块，且优化 barrier 保留 3D dot 布局导致层间 activation 同时驻留。补丁传播 `MUST_SAVE` 并以二维 view 过 barrier。适合在 TPU 上按 PyTorch 版本复验 [B,S,D] MLP 的实际峰值，提交未给普适节省百分比。 [google-pytorch/torch_tpu 594e433d43](https://github.com/google-pytorch/torch_tpu/commit/594e433d439bdea42fa3ed0e3d9d6a6261f13d05)

### TorchTPU Flash Attention 前向直接支持非对齐序列

**2026-11-05T22:42:47Z · google-pytorch/torch_tpu 9997c78002**

前向 custom call 移除外部 PadSequenceDim/SliceSequenceDim，在 kernel 内丢弃越界 query 输出并将越界 V 清零，避免 NaN 进入累加。范围是前向，不应外推到反向。适合测奇数长度、越界 NaN 与 padding 前后的差值。 [google-pytorch/torch_tpu 9997c78002](https://github.com/google-pytorch/torch_tpu/commit/9997c7800239d382a662beaf4043068fc1e33114)

### llama.cpp 修复量化 KV 的设备端状态复制范围

**2026-11-05T02:31:02Z · ggml-org/llama.cpp #30541**

量化类型的每元素字节数是按 block 表示的，旧 device state 路径把 block 数直接当元素数，导致复制不足或异常。保存、恢复及异构分块间拷贝都用 block byte size 和 block element count 换算，验证整块对齐。使用 q8_0/q4_0 做 slot state 往返比较有直接价值。 [ggml-org/llama.cpp #30541](https://github.com/ggml-org/llama.cpp/commit/e56f006ed6b416ae8b7ccb5243b204743f09df1e)

### llama.cpp 让 server checkpoint 通过引用计数共享

**2026-11-05T10:02:36Z · ggml-org/llama.cpp #30137**

slot 复制状态时复用 checkpoint 的共享所有权，避免重复深拷贝。补丁同步调整 slot save/restore appendix 及 common state buffer。值得量化相同前缀多 slot 的内存与序列化成本，主干提交未给通用性能量。 [ggml-org/llama.cpp #30137](https://github.com/ggml-org/llama.cpp/commit/e152332a15986f4b15f79396df1980276b984001)

### MLX-LM 为 DSA sparse prefill 增加 Metal 融合内核

**2026-11-05T20:48:54Z · ml-explore/mlx-lm #2027**

DeepSeek V3.2 和 GLM DSA 路径可让每个 query 按 top-k index 直接读 latent/RoPE cache 行，避免物化整块 `[L,topk,r+rope]`。实现检查设备、形状、dtype 并保留 fallback。适合在 Apple GPU 上测长前缀峰值显存及 dense/sparse 的交叉点。 [ml-explore/mlx-lm #2027](https://github.com/ml-explore/mlx-lm/commit/56230fa02b8ff8330daf26dc95c28480423ed5fe)

### OpenXLA 回收失败编译前先等待在途 GPU kernel future

**2026-11-05T10:02:37Z · openxla/xla 90af967960**

早退释放 IR 或 buffer assignment 后，线程池任务仍可能持有这些裸指针。补丁在 LLVM IR 路径等待两组 future，conditional 先取 predicate allocation，bitonic sort 统一收集并 join 已启动阶段。主要收益是失败路径可靠性，适合在编译错误注入时验证稳定退出。 [openxla/xla 90af967960](https://github.com/openxla/xla/commit/90af967960005ac01019e9f2348afcd57565b020)

### Triton 为 Blackwell 增加 FP6 转换与 TMA 紧凑写

**2026-11-05T22:29:40Z · triton-lang/triton #12541**

`to_fp6` 将 FP32/FP16/BF16 按最近偶数舍入和有限值饱和映射到 E2M3 或 E3M2，先放在 uint8 的低 6 位。SM100/103 的 TMA 用固定 128 值宽度从 padded 共享内存写出每行 96 字节，免去显式 global 端 bit packing。适合复现布局与量化误差，再实测传输收益。 [triton-lang/triton #12541](https://github.com/triton-lang/triton/commit/7452346c7251d033165682662a2602251fa68166)

### Triton 限制不能安全切换 accumulator 的 TMEM 多缓冲

**2026-11-05T16:41:02Z · triton-lang/triton #12562**

只有循环中的 TMEM 读取与下一步不复用 accumulator 的条件一致时，才允许切换到新 accumulator。其余情况禁用 TMEM multibuffering，避免读取错误的累加结果。适合用循环内条件读和 accumulator 重用的反例回归。 [triton-lang/triton #12562](https://github.com/triton-lang/triton/commit/13b7237e949f3fc975486c6dbd45763acf4752bd)

### Triton ConSan 在启动屏障前完成检测状态初始化

**2026-11-05T16:19:28Z · triton-lang/triton #12333**

旧路径在 CTA 启动屏障后清零 commit tables，其他 CTA 可能先读取未初始化数据，或其更新随后被覆盖，导致误报或漏报。补丁在全状态初始化后才执行 startup sync。适合复核多 CTA race 样例及已知无 race 的对照。 [triton-lang/triton #12333](https://github.com/triton-lang/triton/commit/547e5806242933c6ddf2bb39b1e40c495da68edb)

### Transformer Engine 恢复 eager unfused attention 的 softmax 梯度复用

**2026-11-05T15:04:10Z · NVIDIA/TransformerEngine #3815**

eager unfused attention 只对独占中间梯度 opt-in 覆盖。观测 hook、编译、tensor subclass 和需要反向求导的路径保留独立输出，显式覆盖递增 version counter。适合在训练中分别测 eager 与编译路径峰值显存和二阶梯度行为。 [NVIDIA/TransformerEngine #3815](https://github.com/NVIDIA/TransformerEngine/commit/eb10c80d9f3b5c06ed8d792af7a0e99947ab9af8)

### Transformers 避免编译态单 token decode 的 GQA packing 分支

**2026-11-05T14:00:36Z · huggingface/transformers #50388**

SDPA 在不能使用原生 GQA、query 长度为 1 时原本可能尝试额外打包。补丁在 Dynamo 编译时跳过该分支，继续走兼容路径。适合对照 eager 与 compile 下 prefill 后的单 token logits 和吞吐。 [huggingface/transformers #50388](https://github.com/huggingface/transformers/commit/2aab563bd5c98d748e1db3e055d1527468631b8d)

<a id="hardware"></a>
## 硬件软件协同

### AMD 用 DeepSeek-V4-Flash 代理任务拆分训练瓶颈

**2026-11-05 · ROCm 技术博客**

AMD 使用单机 8×MI350X、BF16、TP1/PP1/EP8、序列 4096、微批 1、全局批 8 的八层代理，逐项替换融合、attention backend 与专家路径。优化后无 profiler 的稳定步为 239.8 ms、17,079 tok/s/GPU。采集本身会改变步时，rank 0 PyTorch trace、8 rank trace 和 rocprofv3 分别约为基线的 1.09×、1.20×、5.06×。因此作者用无采集的 step 比速度，用 trace 查算子与通信。将全局批增到 32 时为 22,122 tok/s/GPU，单步 740.6 ms。各档只跑一遍，未估计噪声，结果只覆盖该代理。值得复现其单变量流程和 profiler 开销测量。 [ROCm 技术博客](https://rocm.blogs.amd.com/software-tools-optimization/madengine-primus-profiling/README.html)

### AMD Quark 把逐层 RTN 量化放进推理引擎的加载流程

**2026-11-05 · ROCm 技术博客**

vLLM 插件、ATOM 和 SGLang 的路径均在权重加载时做逐层转换，允许一个 checkpoint 做混合精度探索或用于频繁变化的 rollout 权重。ATOM 可把部分已量化格式逐层反量化再重打包。加载仍需读取源权重，RTN 路径没有离线校准和精度恢复。作者表 1 在 GSM8K 上给出 TP8 DeepSeek-R1 的 MXFP4 离线 0.9568、在线 0.9573，未量化为 0.9644。表 2 的 ATOM Qwen3-235B MXFP4 为离线/在线 TTFT 450.51/445.52 ms、TPOT 34.16/34.03 ms，在线转换耗时 1.398 s。表中未列齐设备、请求长度和并发，适合复现启动峰值与权重切换周期，不宜直接换算生产吞吐。 [ROCm 技术博客](https://rocm.blogs.amd.com/artificial-intelligence/online_quantization/README.html)

### NVIDIA DSX OS 用 NCCL 测试门控故障节点回池

**2026-11-05T17:00:00Z · NVIDIA 技术博客**

NVCRE 支持 NCCL 集合通信和实际训练 workload，失败时二分节点组复测。启用生命周期控制后，NVSentinel 可在新节点或修复节点加入时施加调度门禁，测试通过后撤销。默认 NVSentinel 只监测，NVCRE 未配置性能阈值时也可让慢节点通过。文章中的 300 GB/s 为 YAML 示范阈值。值得参考其验收和再入池流程，用现场拓扑与负载校准阈值。 [NVIDIA 技术博客](https://developer.nvidia.com/blog/configure-certify-and-operate-kubernetes-gpu-clusters-with-nvidia-dsx-os/)

### 补录｜NVIDIA Attribution 将故障诊断绑定到原始证据

**2026-11-02T23:39:34Z · NVIDIA 技术博客**

Attribution Blueprint 收集任务上下文与日志并调用服务，规则引擎归并症状，诊断保留证据链。公开 Docker quickstart 可处理仓库内样例，规则带同处定义的测试。文中的示例证明工作流可跟踪到具体日志，未提供故障归因准确率或跨集群泛化数据。适合用已知根因的多 rank 故障检验二次 NCCL 症状能否正确归并。 [NVIDIA 技术博客](https://developer.nvidia.com/blog/diagnose-distributed-ai-workload-failures-with-nvidia-attribution/)

<a id="coverage"></a>
## 覆盖与续查

- **模型与机构。** 核对 82 个中英文官方入口，其中 75 个直连返回 HTTP 200。检查 64 个 Hugging Face 组织及 26 项固定模型或数据工件。HF 的全站 lastModified 枚举、限制访问的工件与未抽中的条目留待续查。
- **论文与评估。** 检查 arXiv 公告、分类与版本、Hugging Face Papers、OpenReview 等入口，源别计数和受阻区间见论文覆盖段与本地台账。arXiv 时间过滤对修订的覆盖有限，公告页与提交时间分开保存。
- **开源工程。** 核心六仓分别核对主干、合入记录与发布面。扩展 37 仓列出 3042 条回溯窗口默认分支提交（主窗 735），37 个近期 release feed。37 个仓库的独立合入查询完整返回，含 2997 条合入记录。正文只展开核过机制和限制的条目。短 Atom feed、默认分支以外的 PR、未逐条审核的候选保留各自覆盖边界。
- **硬件与软件栈。** 检查 NVIDIA、AMD、Intel Gaudi、Ascend、Cloud TPU、AWS Neuron 的公告、文档或组件包。Ascend bulletin 替代路径返回 404，CANN/HCCL 动态页仅取得页面框架。libtpu 0.0.52 的 11 月 5 日 wheel 已核，版本级行为说明待查。CUDA 13.5 文档仅观察到更新时间文字变化。
- **版本与补录。** vLLM 0.33.0 PyPI 工件、回溯窗口的论文与发布后证据在对应条目标明原日期和状态。FlashInfer `v0.7.4rc1` 的页面在窗口内显示新 tag，当前页只给 tag 说明，主干变更逐项列明，不据此归入候选包。
- **尚未完成的区间。** 各条线的检索失败、有限分页、历史欠账与未审核候选留在本地 `.monitor/research-2026-11-06/` 和 `.monitor/state.json`。下次以本轮发现截止续扫并回溯 72 小时，按源继续补查。本轮全局语义覆盖仍为未完成。
