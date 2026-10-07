# LLM 技术动态日报 · 2026-10-07

> 窗口：2026-10-06 08:00 → 2026-10-07 08:00（北京时间，24h；即 UTC 10-06 00:00 → 10-07 00:00） · 生成时间：2026-10-07 10:40

## 今日要点

1. **Mistral Large 4 发布（公测 API，权重月底放出）**：**1T 总参 / 49B 激活**的多模态 MoE，自有欧洲机房 **3,800 张 Grace Blackwell** 从零训练；RL 阶段在 **约 3k GPU** 上单 run **每天产出约 330 亿 token、其中约 160 亿为可训练 completion token**，actor fleet 弹性伸缩 + 异步训练。除规模数字外，架构、精度、并行配置**全部未披露**。
2. **torchtitan #4923 `HiMidLoLinear`：fp32 LM head 不再需要 fp32 GEMM**。forward 用 bf16 GEMM + fp32 累加输出，backward 把 fp32 的 grad_output **精确拆成 2–3 个 bf16 分片**各跑一次 bf16 GEMM（9 个 GEMM 里只跑 2 个）。Qwen3-30B-A3B、8×GB300 上 fp32 head 配置 **MFU 15.3% → 18.8%、tps/GPU +22.9%**；LM head 推理单次 1850 → 362 µs。
3. **torchtitan #5081：batch invariance 真正需要的 NCCL 变量只有两个**（`NCCL_ALGO=allreduce:tree` + `NCCL_MAX_NCHANNELS=1`），vLLM 那套 10 个变量里有 1 个不是 NCCL 变量、1 个取值非法、还顺手把 FSDP 的 NVLS 关了。附带一条必须记住的结论：**TP=2 永远 bitwise 不变（两数相加无顺序问题），所以用 TP=2 的 parity 测试抓不到任何跨 rank 归约顺序 bug**——默认 NCCL 在 TP=4/7 即 variant。
4. **vllm-ascend 一天 16 个合入，量化 KV cache 双线推进**：DeepSeek V4 **TurboQuant INT4 KV cache**（A2/A3，#17676，512 元素 C4 状态压到 258 字节）与 **C8-MXFP（FP8 KV + E8M0 scale，A5，#17180）**；同时修掉一个**把 QwQ-32B decode 吞吐打到 44% 的非连续 KV 布局回归**（#17924：456.6 → 1029.6 tok/s）和一个 **`rankSize` 未初始化导致申请 419 GiB workspace 的 EP 算子 bug**（#17914）。
5. **"静默算错/只在恢复时才暴露"类修复今天异常密集**，建议逐条对照自查：torchtitan **MXFP8 run 的 DCP checkpoint 存得进、读不出**（#5094）；**MTP 分深度 loss 归一化分母用错**（#5055）；SGLang **PD 分离下 Mooncake 提前传 KV，decode 侧收到 31.7M–34.2M / 38.3M 行全零 KV 且每次传输都报成功**（#41395）；DeepSpeed **inf/NaN 梯度范数哨兵 `-1` 被平方成 1.0，坏梯度被当作健康步更新**（#8638）；TRL v1.14.2 **GRPO 在 eos 之后继续生成并拿去训练**（#7505）；PyTorch **memory-efficient attention backward 在 head_dim=128 就有 shared memory 竞争**（#199775）。
6. **论文侧两篇直接对口**：**ThunderSyncRL**（Stanford）把 GRPO 组梯度精确分解成两个可流式累加的统计量，零 staleness 下把同步 RL 步时 1313.6 → 687.7 s（1.91×）；**CIPHER-MoE** 在 Megatron 上给出 **DeepSeek-V4-Pro 1.6T（TP2/PP8/EP64）** 的领域 SFT 实测——vanilla 首个 iteration 前即 OOM，expert 侧容量准入后可训。

---

## 1. 模型发布与 Tech Report

**Mistral Large 4（ML4，Public Preview）**｜Mistral AI｜2026-10-06｜https://mistral.ai/news/mistral-large-4 ｜ https://docs.mistral.ai/getting-started/changelog/

- **规格**：**1 trillion 总参 / 49 billion 激活**，原生多模态，"hybrid instruct-and-reasoning MoE"。changelog 写 **1M context**（博客正文未写）。API id `mistral-large-4`，$1.36 / $4.18 每百万 token（输入/输出；changelog 称上线两周五折，该价是否已含折扣未确认）。**权重本月底放出**，HF `mistralai` 组织目前尚无对应仓库（最新仍是 07-16 的 Shieldstral-1.0-3B）。
- **预训练 infra**：原文 "trained from scratch on **3,800 NVIDIA Grace Blackwell GPUs** in Mistral's own datacenters in Europe"。训练数据覆盖 160+ 语言。
- **RL infra**：原文 "At our current scale (**3k GPUs**), a single training run produces roughly **33 billion tokens per day**, of which around **16 billion trainable completion tokens** after filtering and masking"。架构为**弹性伸缩的 actor fleet 并行生成数万条 rollout，训练异步进行**；称有"新方法与优化"压低 off-policy drift 并保持低 staleness，但**无算法形式、无 staleness 数字**。verifier 由 reward model、单元测试、LLM judge、静态检查按需组合。
- **benchmark（官方自报，节选）**：DeepSWE v1.1 **61.7%**、Terminal-Bench 4.0 28.3%、SWE-Atlas-QnA 59.4%、Cybench 93%、AutomationBench 59.9%、AA-Briefcase Elo 1,393、B3 攻击抵抗 93.3%。Surge AI 盲评代码质量 3.74/5（低于 Claude Opus 5 的 4.22，高于 Kimi K3 3.59、GLM-5.3 3.60）。
- **简析**：
  - **核心信息量在两个可折算的数**。3k GPU / 33B token·天 ≈ **每卡每天 1100 万 token（含 rollout 与训练两侧摊销）**；可训练占比 **≈48%**（16/33），即过滤与 mask 后一半 token 不进 loss。这两个比值可直接拿来对标我们 RL 集群的"每卡日产 token"和"有效 token 占比"。
  - **与 Beam 的对照**（上期收录）：Beam 是 501B/23B、预训练 6144 GB300、RL 10.5K GB300；ML4 是 **1T/49B 却只用 3,800 卡预训练、3k 卡 RL**。总参翻倍而卡数减少约 40%，要么训练时长显著更长，要么 token 预算更小——**预训练 token 数、训练时长、MFU 均未披露**，无法判断。
  - **相对前作**：Mistral 首次进入 1T 档并把 RL 吞吐当作一等指标公开；"hybrid instruct-and-reasoning" 暗示单权重同时覆盖非推理与推理模式。
  - **未披露/可疑**：层数、expert 数、top-k、注意力结构；数值精度（BF16/FP8/MXFP8）；并行配置；预训练 token 数与 goodput；RL 算法；license；推理吞吐。"Grace Blackwell" 未区分 GB200 / GB300。benchmark 多为选择性对比（各表对手不同），且 AA Cyber Index 一项称 Opus 5.5 / GPT-6 Astra "接近零分"——那是拒答而非能力差，不可当能力对比读。
- **对我的意义**：在技术报告出来之前只有两点可用。(a) **"每卡日产 token"与"可训练 token 占比"** 应成为我们 RL 平台的标准看板指标，ML4 给了一个 3k 卡量级的公开参考点（约 1100 万 token/卡·天、48%）。(b) 月底权重 + 报告放出时重点看它的 **1T MoE 在 3,800 卡上的并行切分与 EP 通信方案**——这是比 Beam（6144 卡训 501B）更"卡少参数多"的配置，显存与 expert 参数分片压力更接近我们的现实约束。

**`openai/math`：722 篇模型生成的数学手稿公开**｜OpenAI｜2026-10-06（仓库创建 21:47 UTC）｜https://openai.com/index/sharing-ai-progress-in-mathematics/ ｜ https://github.com/openai/math

- 由"未发布的内部模型"产出 **722 篇手稿、归为 372 个 family**；共提出约 4,000 个问题，平均每个结果耗用"3 小时 ChatGPT Pro thinking 算力"。部分有 Lean 形式化，README 明示**未形式化的结果可能有问题**。无模型规模与训练 infra 数据。
- **对我的意义**：与 infra 无直接关系，属能力/评测信号。唯一可借鉴处是"**长时推理结果 + 形式化验证器**"这条 verifier 路线——对 RLVR 的奖励可信度是参考范式。

**`allenai/miles-olmo-core` 首次源码发布**｜AI2｜2026-10-06 04:17 UTC｜https://github.com/allenai/miles-olmo-core

- radixark/miles（SGLang rollout + Megatron-LM 训练的 RL 后训练框架）的 AI2 fork，对接 OLMo-core trainer 与 open-instruct。README 自述 "exploratory fork… does not support any publicly released models at this time"，无 fork 专属性能数据。
- **对我的意义**：信号意义大于实用价值——**AI2 选择把 RL 后训练接到自家 Olmo-core trainer 上而不是沿用 Megatron**，与上周 Olmo-core 3（面向大 MoE 的训练 infra）是同一条线。可作为"RL 框架 trainer 后端可插拔"的参考实现跟踪。

> 其他厂商：已逐一核对 20 个 HF 组织的 createdAt，**窗口内无任何新建模型**（deepseek-ai 最新 09-10、Qwen 09-20、zai-org 08-25、moonshotai 06-13、XiaomiMiMo 09-27、nvidia 10-01、Aleph-Alpha 10-02）。**Reflection AI Beam 的权重与技术报告仍未放出**（HF 两个候选组织名均返回空）。GitHub 侧另有 `google-deepmind/gemma` 一条值得知道的小修正：`Embedder.encode_audio` 的顺序由 Projection → Norm 改为 **Norm → Projection** 以匹配上游训练配置（用该库跑音频输入的结果会变）。

## 2. 论文精选（训练 / 后训练 / 推理）

> 日期口径说明：下列除 FC-SWE 外 v1 均为 10-05 提交，按 arXiv 流程于 10-06 00:00 UTC（窗口起点）公告，与上期 HLA（2610.05842）属同一批，本期按"公告落在窗口内"收录。细节由子 agent 读全文/结构化报告得出，**未经本会话逐条对照原文**；各条已标注阅读深度。

**ThunderSyncRL: Lossless Acceleration of Agentic Reinforcement Learning**｜Stanford / Yonsei / Korea Univ. / Bespoke Labs｜arXiv 2610.05935（v1 2026-10-05）｜https://arxiv.org/abs/2610.05935 ｜阅读深度：结构化报告 + 正文 §3.4/§4.1 与附录 D/G/I

- **问题**：同步 RL 下 learner GPU **49.5% 的步时间在空等 rollout**；异步能填满但引入 staleness。
- **机制**：把"何时算梯度"与"何时更新策略"解耦——执行异步、更新同步、零 staleness。GRPO 组梯度被**精确分解**为两个可流式累加的统计量：`g_G = (1/m)·(G1 − μ_G·G2)/(σ_G + ε)`，其中 `G1 = Σ r_i H_i`、`G2 = Σ H_i`、`H_i = ∇_θ Σ_t w_{i,t} log π_θ(a_{i,t}|h_{i,t})`。**每条轨迹 reward 一到就做 backward 累加 G1/G2，组关闭后只做一次标量组合**。OPD（on-policy distillation）同理按 turn 流式 backward。batch barrier 处做一次 clip + 一次 optimizer step，发布 θ_{k+1} 并等所有 rollout replica 确认、清 prefix cache。
- **配置**：单节点 8×B200；Qwen-3.8-27B 全量微调与 GLM-4.7-Flash-31B（MoE，64 experts top-4）。GRPO 为 4 卡 learner + 4 卡 rollout，32 prompt × 8 轨迹；轨迹上限 300 turn、总上下文 131,072。baseline 为 **verl 官方 pipeline**（GRPO）与 SkyRL（OPD），rollout 均为 vLLM。
- **数字**：GRPO 步时 **1313.6 → 687.7 s（1.91×）**，OPD 1330.3 → 685.7 s（1.94×）；Async / Fully Async 步时几乎相同（695.0 / 690.0 s），**ThunderSync 的优势是零 staleness 而非更快**。SWE-bench Verified 达到 74.01% pass@3 用 121.3 GPU-h（Sync 为 233.5）。FP32 下梯度相对误差 GRPO 5.1e-7；**BF16 下与 FP32 Sync 的相对误差约 0.008–0.04，不是 bitwise 等价**。
- **代价与未测**：梯度累加器是 **FP32、放在 host 内存**，每 rank 上界 `4(2k+1)P` 字节——四个 learner rank 合计 **Qwen 1.30 TB、GLM 1.49 TB**；每个 learner GPU 放一份完整 BF16 student，只有 AdamW state 分片。理论上限 2×，要求 rollout 与 learner 耗时平衡。**未测多节点、未测 TP/PP 切分的大 learner**；设定是 0 个额外 PPO epoch、无 KL——带 importance ratio clip 时式中的线性分解不再成立（子 agent 的推断，论文是否讨论未读到）。
- **对我的意义**：思路可直接移植到 verl 式 GRPO——**把 backward 拆成"reward 到即算 H_i、组关闭后用 G1/G2 标量重组"**，零 staleness 吃掉 learner 约一半空闲。但移植的真正工作量在存储：TB 级 FP32 累加器在需要 TP/PP 的大模型上必须分布式存放，且昇腾 host 内存与 NPU 间的搬运带宽会成为新瓶颈。建议先在不带 ratio clip 的纯 PG 配置上验证等价性。

**CIPHER-MoE: Balancing Efficiency and Routing Fidelity in Trillion-Scale MoE Training**｜同济 / Cornell / 哈工大(深圳) / Shenzhen Loop Area Institute｜arXiv 2610.05744（v1 2026-10-05）｜https://arxiv.org/abs/2610.05744 ｜阅读深度：正文与附录 A.5 全文；证明未细读

- **问题**：领域数据上路由极度倾斜——DeepSeek-V4-Flash 上 **10% 的 expert 吃掉 88% 以上的路由负载**，EP all-to-all 的峰值缓冲无界。
- **机制**：token 侧 Top-K 不变，**expert 侧加一层准入**。依据 cosine 亲和度 `c_{i,e} = w_e^T x_i / (‖w_e‖‖x_i‖)` 先按阈值 τ 过滤，再保留亲和度最高的 `min(C_e, |P_e^τ|)` 个 token。被拒的 (token, expert) 对两种处理：**strict**（gate 置零，其余 expert 路径保留）或 **reroute**（改派给有余量的 expert）。另加 LocMoE 的 locality loss（λ=0.01）。capacity factor = 8。
- **配置（Megatron，四个模型的完整并行切分是本文最有价值的部分）**：
  | 模型 | 规模 | 节点 | 并行 | seq | MoE |
  |---|---|---|---|---|---|
  | DeepSeek-V4-Pro | 1.6T | 32 | TP2/PP8/EP64/CP1/ETP1/VPP4 + SP | 4K | 384 experts，Top-6 |
  | DeepSeek-V4-Flash | 284B | 8 | TP2/PP4/EP16/CP1/ETP1/DP16 + SP | 8K | 256 experts，Top-6 |
  | DeepSeek-V3 | 671B | 16 | TP4/PP16/EP4/CP1/DP4 + SP | 8K | 256 experts，Top-8 |
  | GLM-5 | 744B | 16 | TP2/PP8/EP32/CP1/ETP1/VPP2/DP16/EDP1 | 8K | 256 experts，Top-8 |
- **数字**：**V4-Pro vanilla 在第一个 iteration 前就 OOM**，CIPHER 可训（GBS 256 时 45.16 s/iter）。strict 加速：V4-Flash 1.42–1.58×、V3 1.78–1.94×、GLM-5 1.13–1.14×；额外开销 strict 占总训练时间 1.57%、reroute 3.85%。**strict 的质量普遍不低于 reroute**（作者解释为 reroute 引入梯度干扰）。
- **局限**：任务是**领域 SFT 而非预训练**；质量有升有降（V4-Flash 的 OR 加权平均 68.59 vs baseline 69.09）；单次运行无多 seed。**未披露加速器型号、每节点卡数、互联、精度、MFU**；加速里 locality loss 与 capacity cap 各占多少未拆分；代码 "soon"。
- **对我的意义**：两点。(a) **"expert 侧 top-C 准入 + strict drop"把 EP all-to-all 的峰值缓冲变成有界量**，这对昇腾上静态 shape、定长通信 buffer 的 EP 实现是天然契合的约束，也是领域 SFT / RL 阶段路由倾斜导致 OOM 的低成本兜底。(b) 表中 **V4-Pro 1.6T 的 TP2/PP8/EP64/VPP4** 是目前公开可见的、最大规模 MoE 在 Megatron 上的可运行切分之一，可作为我们规划同档模型时的起点参照。

**FC-SWE: Failure-Conditioned RL for Long-Horizon Software Engineering Agents**｜NVIDIA / 清华｜arXiv 2610.07898（v1 2026-10-06）｜https://arxiv.org/abs/2610.07898 ｜阅读深度：正文 p1–8、14–15

- **机制**：每个 issue 采 G=8 条初始轨迹，失败后重置仓库、把**失败 patch 与 verifier 输出作为上下文**发起恢复轨迹（每链最多 R=2 次）。两个改动：(1) **trajectory-local reward**，未执行位记为 ⊥ 而非 0，`[0,1]` 链不被广播成 `[1,1]`；(2) **active-set advantage**，同一 issue 下所有实际执行的轨迹合成比较组，leave-one-out baseline + std 归一，padding 行不进统计也不进 loss。作者承认比较集依赖 outcome，不再继承无偏性保证。
- **配置**：Qwen3.5-4B，R2E-Gym 4,518 任务，上下文 65,536，每步 16 任务 × 8 chain。**训练并行 TP4/PP2/CP2、rollout TP1、权重异步同步**；transport tensor 固定 256 行。clip 下界 0.2、上界 0.28；**非法 tool call 或 malformed thinking 的 token advantage 直接覆写为 −5**。
- **数字**（SWE-bench Verified，verifier-assisted）：Resolved@1 / @2 为 Base 27.9 / 36.3 → GRPO 38.9 / 48.5 → **FC-SWE 41.7 / 52.8**。**去掉 −5 惩罚的奖励广播变体，在线成功率从 step 28 的 76.6% 崩到 step 31 的 2.3%。**
- **局限**：每步轨迹数最多翻倍，**未见等 rollout 预算的 GRPO 对照**（如 G=16）；checkpoint 选择用了 SWE-bench Verified 的 100 题子集；GPU 型号/数量/步时未披露；只在 4B 上训练。
- **对我的意义**：工程上最值得抄的是**"静态 G×R 行 + activity mask"**——组大小随 outcome 变化时仍保持分布式 batch 形状固定，对昇腾静态图友好。**格式惩罚 −5 是稳定器而非锦上添花**，那条 76.6% → 2.3% 的崩溃曲线值得记。

**LoGRA: Scaling LLM Reinforcement Learning with Low-Rank Gradient Sketches**｜NVIDIA｜arXiv 2610.06647（v1 2026-10-05 16:28 UTC）｜https://arxiv.org/abs/2610.06647 ｜阅读深度：**仅 alphaXiv 结构化报告 + 摘要，公式未对照原文**

- **机制**：对权重 `W ∈ R^{d×k}` 每步采样随机符号投影 `A ∈ R^{r×k}`，在 backward 前先投影输入、直接累加 sketch `S = G A^T ∈ R^{d×r}`，**不构造完整梯度**；更新 `W ← W − αη U A` 仍作用于全量权重。优化器 RowAdam（每 sketch 行一个二阶矩、无一阶动量）。**策略同步只下发低秩因子 + 随机种子**。rank 256。
- **数字**（单节点 8×H100，4 卡 FSDP 训练 + 4 卡生成）：7B 平均显存 31.82 → 17.29 GiB/卡（−45.7%），Pass@1 72.48 → 72.33；**27B 下 dense Adam OOM，LoGRA 51.54 GiB 可训**。
- **局限**：是单 response 的 PPO 而非 GRPO 多样本组；只更新 attention 与 MLP 投影矩阵；dense 与 LoGRA 优化器不同、对比不纯；"无 KL 控制"对照同样稳定。**权重同步的带宽/延迟无量化数字**。
- **对我的意义**：可借鉴的是**"因子 + 种子"的增量权重下发**——把 trainer → rollout 集群的同步量从全量权重压到 `d×r`。与 Beam / Kolibri 的"全量权重快推"是正交解法，在 HCCL 跨域带宽紧张时更有吸引力。算法本身尚不具备替换全量更新的证据。

**ORCA: The Annealed Spectral Conditioning Optimizer**｜北京大学 / Kling Team｜arXiv 2610.06116（v1 2026-10-05）｜https://arxiv.org/abs/2610.06116 ｜阅读深度：正文 p1–2、5–9、12–14；理论节未读

- **机制**：在 Muon 上加一个**早期软正交正则、到点硬切除**：`R_orth(W) = ¼‖WW^T − I‖_F²`，前 `⌊τT⌋` 步系数为 γ、之后为 0（默认 γ=1、τ=0.25），释放后严格退化为 Muon。**MoE 必须逐层释放**（首层 20%、末层 60%，线性插值），否则 router 受扰。
- **数字**：LLaMA 1.3B 上约 60k 步达到 Muon 100k 步的 loss（1.66× token 效率，但系从 51k checkpoint 续跑、未重新调参）；8B MoE（1B 激活）上 optimizer update **237.4 vs 227.2 ms，约占 1.3 s 步时的 0.8%**，峰值内存 101.66 vs 101.41 GB。
- **局限**：最大只到 8B / 约 105B token；正则生效期间 val loss **高于** Muon，中途看曲线会误判；τ 需预知总步数；硬件未披露；无低精度下的行为；MoE 逐层释放比例无消融。
- **对我的意义**：实现成本极低（每矩阵多 2 次 matmul，不改通信复杂度），已有分布式 Muon 实现的团队可直接 A/B。结合上期 Kolibri（Muon + 分组优化器）来看，**Muon 系优化器在 MoE 上正在成为默认选项**，昇腾侧的分布式 Newton–Schulz 实现值得提前准备（TE v2.20.2 本期也修了分布式 Newton–Schulz 的数值正确性，见第 5 节）。

**【更新】Request Order Matters: Cache-History Sensitivity in Selective KV-Cache Reuse for Rolling Agents**｜CMU / Capital One / UChicago / UC Berkeley / USC / UW｜arXiv 2610.05833｜https://arxiv.org/abs/2610.05833 ｜上期仅列标题，**本期补全文精读**

- **发现**：CacheBlend / LMCache 式的"选择层按 KV 偏差取 top-k token 重算"在 cache 跨请求持久化后，**同一 prompt 的答案取决于此前的请求顺序**。跨 5 种顺序的答案变异率：token top-k **69.0%** vs 按整段文档重算（document-aligned，重算 token 数相同）**26.1%**。
- **关键对照**：每次请求前清 cache 时两者保真度 94.0% vs 92.6%，无显著差异——**问题只在有历史累积时出现**。消融显示起作用的是**重算区间的连续性**：top-k 把预算散到中位 180.5 段、捕获 99.7% 的偏差，保真度却只有 38.6%；四种连续 span 策略为 89.4–92.8%，其中随机选文档的那种只捕获 4.8% 的偏差。
- **配置**：Qwen3-8B，greedy；LMCache 0.5.1 + vLLM 0.23.0；单张 L4；7,735 次 rolling update，prompt 约 7.2K–7.7K，重算预算 5%。TTFT 中位数 full prefill 2,408 ms → 约 420 ms（5.7×）。
- **局限**：单模型、单卡、单一工作负载、串行请求；未解释机理。
- **对我的意义**：**非前缀 KV 复用会让同一 prompt 的输出依赖调度顺序**，在共享 cache 的 agentic RL rollout 中直接破坏可复现性与训推一致性。上线任何"选择性重算"类 KV 复用前，评测必须用**持久化 cache + 乱序请求回放**，clean-cache 单请求测不出来。

> 未精读但相关（仅标题/摘要片段）：**2610.05748 MOLT**（首尔大学，serving replica 闲置显存做机会式微调）；**2610.06597 HEAR Protocol**（哈工大深圳，agent harness 与推理引擎的跨层协议）；**2610.05974**（中科院软件所/Microsoft，RL 提升后的 teacher 经 OPD 可传递哪些增益）；**2610.06671**（EPFL/ETH/AI2，蒸馏时按熵混合分布）；**2610.06286 DeferKV**、**2610.06479**、**2610.06686 OVAL**（三篇 KV cache 压缩/检索）；**2610.06251**（批内共享的停止决策使 KV 量化后答案受无关请求影响——与上面【更新】同属"serving 非确定性"话题）；**2610.07822 Nucleus Speculative Decoding**（10-06，有损推测解码）。
> 窗口外线索（cs.DC 10-05 公告批次，仅标题）：**2610.03415 RailWave**（EP 通信的时空调度）、**2610.03286 VenusRL**（全解耦 agentic RL 系统）、**2610.03457**（跨超算中心 LLM 预训练）、**2610.02658**（CUDA 程序向 Tenstorrent 迁移）。前两篇与本期主题高度相关，建议按需补读。

## 3. AI Infra 技术文章

> 两条均来自 NVIDIA Technical Blog，页面只标日期（10-06）无时分；正文经抓取摘要转述，数字入稿前未逐字对照原页。

**AICR v1.0: Open, stable, and verifiable GPU cluster configuration**｜NVIDIA｜2026-10-06｜https://developer.nvidia.com/blog/aicr-v1-0-open-stable-and-verifiable-gpu-cluster-configuration/

- NVIDIA AI Cluster Runtime 为 GPU K8s 集群提供 **version-locked 的已验证 recipe**，把 host kernel、GPU driver、container runtime、网络、存储、operator、workload framework 的兼容组合钉死。四个能力：Snapshot（采集实况）、Recipe（期望配置 + 约束）、Bundle（渲染成 Helm / Argo CD / Flux 产物）、Validation（对比并跑 deployment / conformance / performance 检查）。
- v1.0 的实质是**稳定性契约**：CLI、`aicrd` REST API、Go SDK、bundle 布局与 artifact schema 纳入 semver。覆盖 10 种 accelerator（Rubin 到 Ampere）、11 个 K8s 服务；训练侧 Kubeflow 与 Slurm，推理侧 Dynamo 与 NIM。**每个 recipe 附带在实测硬件上生成的 signed validation evidence**（`aicr evidence verify`）。
- **对我的意义**：这正是我们"驱动—固件—CANN—torch_npu—HCCL—框架"版本矩阵问题的同构解法。可借鉴的不是工具本身而是两个设计：**把"已验证组合"做成带签名证据的一等制品**，以及 **Snapshot → Validation 的漂移检测闭环**。万卡集群里"某批节点固件/驱动与其余不一致"是 straggler 与静默错误的常见来源，这类闭环比事后排查便宜得多。

**Control How Your GPU Shares Work with Green Contexts**｜NVIDIA｜2026-10-06｜https://developer.nvidia.com/blog/control-how-your-gpu-shares-work-with-green-contexts/

- Green context 允许**单进程内显式划分 SM 与 workqueue**（Driver API 自 CUDA 12.4、Runtime API 自 CUDA 13.1）。除 SM 分区外可单独配置 workqueue，避免互不相关的 stream-ordered 负载被映射到同一底层 workqueue 而意外串行化。
- 基准（148 SM 的 Blackwell，bulk 负载打满时测关键 kernel 延迟）：green context 分区（8 SM 关键 / 140 SM bulk）**0.007 ms**；默认 context + 高优先级 stream 0.140 ms；默认 context 等优先级 3.727 ms。无与 MPS / MIG 的直接对比。
- **对我的意义**：对训练意义有限，对 **RL 推理侧"rollout 生成 + 权重热更新/量化"同卡并发**有直接价值——把一小撮 SM 留给权重更新与控制面 kernel，避免被 decode 的 bulk 负载饿死。昇腾侧是否有等价的算力切分原语值得确认。

> 逐来源核对，窗口内无新增 infra 长文：PyTorch Blog（最新 10-05）、vLLM Blog（09-29）、Meta Engineering（09-23）、Microsoft Research（09-30）、AWS ML Blog（10-05）、Together AI（10-05）、Fireworks（09-30）、SemiAnalysis（10-05 20:54 UTC，差约 3 小时入窗且非 infra）、Cerebras（10-01）、ROCm Blog（09-30）。抓取失败见运行备注。

## 4. 硬件与互联

**无新增。** NVIDIA、AMD、Intel Gaudi、Google TPU、AWS Trainium、华为昇腾、寒武纪、海光、Cerebras、Groq 窗口内均无新芯片、超节点/互联或 NCCL/HCCL/RCCL/CUDA/ROCm/CANN 主版本发布。NVIDIA/nccl 最新仍为 v2.32.3-1（09-17），ROCm/rccl 最新 tag rocm-7.2.4（05-28）。

两条侧面信号：
- **RCCL 的 symmetric memory 支持进入 PyTorch 主干**（pytorch #192524，见第 5 节）。这是**非 CUDA 后端接入 symmetric memory 的第一份完整参考样本**，commit message 逐条列出了不能复用 CUDA 路径的原因，对 HCCL 侧做同类能力有直接参考价值。
- **预告**：OCP Global Summit 2026 预计 10-12 至 10-16（San Jose；日期来自二手页面，**未能一手确认**），AMD 已预告 Lisa Su keynote，下周大概率有 Helios / UALink / CPO 类披露。

## 5. 开源框架与社区

### 正式 Release

**huggingface/trl v1.14.2**｜2026-10-06 20:30 UTC｜https://github.com/huggingface/trl/releases/tag/v1.14.2

- 补丁版，修两处**静默训练错误**与三处崩溃。重点是 **#7505**：GRPO / RLOO / Distillation 在默认生成路径（`use_vllm=False`）下，对于"结束 turn 的 id 只在 generation config 里声明"的模型（Gemma 3/4、Phi-3.5 等），**生成不会在 turn 结束处停下，其后直到 `max_completion_length` 的所有内容都被拿去训练**。唯一可见迹象是 `completions/clipped_ratio` 接近 1。走 vLLM 时生成能正确停止，但指标仍错，且 `mask_truncated_completions=True` 会**丢掉每一条已完成的 completion**。
- 另：**#7463** completion 起点改为"tokenized prompt 与 prompt+completion 首次分叉处"；#7507 修无 reference model 时 FSDP 下的 `precompute_ref_log_probs`；#7471 无可训练 token 的 batch 上 DFT loss 保持有限。
- **对我的意义**：与上期 torchtitan #5049（采样温度没进 logprob）同属一类——**"生成侧与训练侧对同一条序列的边界/分布理解不一致"**。自查清单再加两项：(1) trainer 认定的 eos 集合是否等于生成侧实际使用的全部停止 id；(2) completion mask 的起点是按字符串拼接算的还是按 token 分叉算的（两者在 tokenizer 合并边界 token 时不等）。

**NVIDIA/Megatron-LM `26.09-alpha.rc1` / `26.09-beta.rc1`**｜2026-10-06 09:08 / 09:24 UTC｜https://github.com/NVIDIA/Megatron-LM/releases/tag/26.09-beta.rc1

- 两个 tag 的 release body **均为空**，无 changelog。配套信号是同日 **#7894 把 Transformer Engine 升到 2.20 分支 tip**。按 rc 处理，等正式版说明。

**【补录】NVIDIA/TransformerEngine v2.20.2**｜2026-10-05 21:14 UTC（北京 10-06 05:14，**属上期窗口、上期漏收**）｜https://github.com/NVIDIA/TransformerEngine/releases/tag/v2.20.2

- v2.20.0 / v2.20.1 因发布流程问题被跳过。对训练 infra 有意义的条目：
- **Known Issue（升级前必读）**：**cuBLAS 13.7.0–13.8.0 在 Blackwell 与 Rubin 上，当 NCCL kernel 并发运行时，可能静默地留下未写入的 grouped-GEMM 输出 tile，产生错误梯度**；TE 侧无规避手段，须升级到 cuBLAS 13.8.1+（#3578）。
- **低精度**：实验性 **CuTeDSL 后端**做 MXFP8 量化（`NVTE_WITH_CUTEDSL=1` 编译、`NVTE_ENABLE_CUTEDSL_BACKEND=1` 运行，自动回落 CUDA C++）；grouped MXFP8 的 weighted SwiGLU / clamped-SwiGLU 融合路径与原地 requantization；NVFP4 随机舍入加宽指令、MXFP8 cast kernel 寄存器常驻；`Linear` 在 FP8 / MXFP8 / NVFP4 recipe 下支持 `torch.compile(fullgraph=True)`。
- **MoE / EP**：融合 sigmoid router 新增可选的 **Quantile Balancing 直方图累加**（与上期 Kolibri 的 EQB 同一思路进入 TE）；**CUDA graph 可捕获的融合 EP prepare-and-dispatch 路径**；NCCL-EP 的同步、超时、共享 workspace 故障修复，并通过 `NCCL_DEV_API_JIT` 支持"用 NCCL 2.30.5–2.30.x 编译、配更新的 NCCL 运行时"。
- **修复中值得记的**：L40 上 BF16 GEMM 精度损失（排除了以 BF16 存部分和的 cuBLASLt split-K 算法，#3440）；Triton permutation kernel 中**清零与 routed 梯度写入之间的竞争**（#3581）；NVFP4 随机量化因 RNG tensor 生命周期导致的数据损坏（#3533）；**分布式 Newton–Schulz 的数值正确性与 workspace 生命周期**（#3541）；delayed-scaling FP8 在 eval 模式/模式切换/嵌套 checkpoint 下的 activation-recompute 元数据配对（#3394）。
- **Breaking**：cuDNN 最低版本 9.3 → **9.12.0**；cuBLASMp 最低 0.8.0 → 0.8.1（通算重叠 GEMM 新增 MXFP8）；**P2P context-parallel attention 前向临时存储由 O(CP) 降到 O(1)**，`nvte_cp_thd_out_correction` 需传新参数 `old_lse`；移除 `NVTE_FUSED_ATTN_BACKEND`（改自动选择）。
- **对我的意义**：三条。(a) **"cuBLAS grouped GEMM 与 NCCL kernel 并发时静默丢 tile"** 是典型的"通算重叠才触发"的 SDC 类问题——昇腾侧 GMM 算子与 HCCL 并发执行时是否有同类输出完整性校验，值得专项确认。(b) **P2P CP attention 的 O(CP) → O(1) 临时存储**是长上下文训练的直接显存收益，对应实现思路（携带 `old_lse` 做增量修正）可对照我们 CP 实现。(c) Quantile Balancing 进入 TE 融合 router，说明**全局分位数均衡正在从论文方案变成主干算子**。

> 窗口内无 Release：torchtitan（v0.3.0，09-03）、PyTorch（v2.14.1，09-30）、DeepSpeed（v0.19.7）、verl（v0.9.1）、OpenRLHF（v0.11.2）、vLLM（v0.31.0，10-05，上期已收）、SGLang（v0.5.21，10-02）、TensorRT-LLM（v1.3.0rc29）、**vllm-ascend（v0.27.1rc1 仍为 09-29）**、NCCL、RCCL。

### 重要合入

**pytorch/torchtitan**（窗口内 26 个 commit）

- **#4923 新增 `HiMidLoLinear`，删除 `CastLinear` 与 `RouterGateLinear`**｜https://github.com/pytorch/torchtitan/commit/36adc3a30df688ba1fc47d30827f2d4954bf52a2
  - **问题**：LM head 与 router gate 需要 fp32 输出。原 `CastLinear` 每次 forward 把权重 `.to(fp32)`（Qwen3.5 的 vocab 是 248k）再跑慢速 fp32 GEMM；BF16x9（用 9 个 bf16 GEMM 模拟 fp32 matmul）更快但仍慢；TF32 快但不准。
  - **做法**：forward 用 `torch.mm(out_dtype=float32)`——**GEMM 在 bf16 上算、累加在 fp32**。backward 把 fp32 的 grad_output **拆成 bf16 分片**（hi + mid 两片约 16 bit；hi + mid + lo 三片为精确），每片跑一次 bf16 GEMM，即 **9 个 GEMM 里只跑 2 个**。拆分用整数移位上转，编译器无从消除 `.to(bf16).float()` 往返。LM head 默认 `hi_mid`，router gate 用 `hi_mid_lo`。
  - **为什么两处片数不同**（很有启发）：router 的 grad_input 只对其 experts 求和，**误差由拆分精度决定**，第三片把误差再压约 7×；LM head 对整个 vocab 求和，**误差由 tensor core 的累加决定**，第三片无收益却要多付约 1.5× backward。
  - **受控数据**（1×GB300，真实 Qwen3.5-27B lm_head，vocab 248320 × hidden 5120，2048 token）：
    | 方案 | 推理单次 (µs) | 训练单 chunk fwd+CE+bwd (ms) | logprob 误差 | dW 误差 |
    |---|---:|---:|---:|---:|
    | upcast + IEEE GEMM | 1850 | 229.5 | 8.9e-06 | 1.65e-03 |
    | upcast + BF16x9（trainer 原方案） | 1850 | 77.5 | 2.2e-06 | 1.65e-03 |
    | upcast + TF32 | 1868 | 20.3 | 1.9e-05 | 1.66e-03 |
    | bf16 GEMM + fp32 out + bf16 bwd（Megatron 做法） | — | 12.0 | 1.0e-05 | 2.13e-03 |
    | **bf16 GEMM + fp32 out + hi+mid bwd（本 PR）** | **362** | **18.9** | 1.0e-05 | 1.65e-03 |
    | 同上，grad 保持 fp32 | — | 18.5 | 1.0e-05 | **2.78e-05** |
    | 纯 bf16 head | 360 | 12.3 | **1.7e-02** | 1.24e-02 |
  - **端到端**（Qwen3-30B-A3B，2 节点 × 4 GB300，PP2 × DP4、EP4，HybridEP + CUDA graph，1M token/步，两个 seed）：fp32 head 配置 **MFU 15.3 / 15.2% → 18.8 / 18.6%，tps/GPU 13,686 → 16,822（+22.9%）**；只换 router gate backward 则 +0.6%。
  - **行为变更**：`grad_weight` 现在以 fp32 返回（FSDP2 在 10-02 之后的 torch nightly 保留它），**RL loss 参考值从 step 3 起与此前分叉**；batch-invariant 模式下 router 改走 `batch_invariant_ops`（cuBLAS 的 `out_dtype` GEMM **不是** batch-invariant）；移除全局开关 `enable_fp32_matmul_emulation_with_bf16x9`。作者注明端到端验证早于 review 3 的若干改动（当时为截断而非舍入）。
  - **对我的意义**：**这是本期最可直接移植的一条**。"fp32 logits 是 RL 训推一致性的刚需"（上期 Kolibri 也是两侧 LM head 保 FP32），而代价一直是 LM head 的 fp32 GEMM。"bf16 GEMM + fp32 累加输出 + 梯度分片 backward" 把这个代价压到纯 bf16 的约 1.5×，且 logprob 误差与 IEEE fp32 同量级（1.0e-05 vs 8.9e-06）、比纯 bf16 好三个数量级。昇腾侧需要确认的只有一点：**MatMul 算子是否支持 bf16 输入、fp32 累加输出**；若支持，分片 backward 是纯框架层改动。
- **#5081 batch invariance 只保留两个 NCCL 环境变量**｜https://github.com/pytorch/torchtitan/commit/1e3c22236469c91abbfd2e2ba2a04f71a8509a14
  - `set_batch_invariance` 原样照抄了 vLLM `override_envs_for_invariance()` 的 10 个变量。逐个核对 NCCL 2.30 源码后只留两个：**`NCCL_ALGO=allreduce:tree`**（tree 对任意消息大小都沿一棵固定的树求和；ring 的顺序取决于消息如何分块）与 **`NCCL_MAX_NCHANNELS=1`**（NCCL 建两棵跨节点树并在 channel 间交替，多 channel 时某元素由哪棵树归约取决于消息大小）。
  - 被删的 8 个里：`NCCL_P2P_NET_DISABLE` **不是 NCCL 变量**；`NCCL_NTHREADS=1` 不是 32 的倍数，被 `getNthreads` 判为非法而改用最大线程数；`NCCL_PROTO=Simple` 无必要（Simple / LL / LL128 求和顺序相同）；**`NCCL_NVLS_ENABLE=0` 冗余且副作用是把所有 batch-invariant 训练里 FSDP all-gather / reduce-scatter 的 NVLS 也关掉了**。
  - **探针结果**（单节点 7×H100，bf16 `[num_tokens, 4096]`，检查同一行在不同 `num_tokens` 下是否 bitwise 相同）：NCCL 默认——TP=2 invariant、**TP=4 / TP=7 variant**；仅 `ALGO`——三者皆 invariant；`MAX_NCHANNELS + PROTO` 而无 `ALGO`——TP=4 variant。端到端 TP=4 parity 测试 3/3 通过，去掉 `NCCL_ALGO` 的负对照三项全挂（max_delta 2.4e-2 / 7.4e-2 / 7.0e-3）。
  - **未测多节点**——保留 `MAX_NCHANNELS=1` 与删除 `PROTO` 都是读源码的结论。
  - **对我的意义**：两条。(a) **"TP=2 的 parity 测试抓不到跨 rank 归约顺序 bug"** 是很容易踩的验证盲区，我们的训推一致性回归应至少覆盖 TP≥4（最好含奇数）。(b) **HCCL all-reduce 的求和顺序是否随消息大小/分块变化**需要有明确答案——这是昇腾上实现 batch invariance 的前置条件，建议直接照搬文中那段约 20 行的探针脚本在 HCCL 上跑一遍。
- **#5067 按父 scope 给 routed expert 的 FSDP 集合通信分桶**｜https://github.com/pytorch/torchtitan/commit/17bf1f6cab
  - traced 的集合通信节点标注的是父 scope `layers.N.moe.routed_experts`，而 bucket plan 写的是 `w13` / `w2` 子 scope，于是**匹配不上，每个 MoE 层留下两个 routed-expert bucket**。改用父 scope 后合并为一个。
  - **256×GB300、DeepSeek V3 671B DistMoE**：GraphTrainer **5,670.5 → 5,830.5 tokens/s/GPU（+2.82%）**；同集群 eager 为 5,822.0；**峰值显存 249.38 vs eager 265.08 GiB**。all-gather 与 reduce-scatter 各由 179 降到 121（58 个 MoE 层每层恰好少一对）。配套 #5058（把 split-pad-cat 移出每个 microbatch、并入最终 reduce_grad）与 #5059（WGrad 累加穿透 view 融合成 `addmm_`）。
  - **对我的意义**：给了一个**256 卡 GB300 上 DeepSeek V3 671B 的公开吞吐基线（约 5.8k tokens/s/GPU）**；另一层意思是"**按 scope 名字匹配的优化规则静默失配**"——没有报错、只是少了 2.8%。我们任何基于模块名/正则的分桶、重计算、量化白名单都该加"规则命中数"断言。
- **#5094 修复 MXFP8 的 DCP checkpoint 序列化**｜https://github.com/pytorch/torchtitan/commit/4b56d82cf4
  - `MXFP8Linear` 把 `_LinearShardedTensorWithMXFP8Compute` 装成 `nn.Parameter`；DCP 写入时经 `__get_tensor_shard__` 取数据，而 `_ShardedFSDPTensor` 没实现该 hook，于是 **DCP 把 wrapper 本身 pickle 进了 checkpoint**。读取走 `weights_only=True`，报 `Unsupported global`。**保存成功，只在 resume 时才暴露；任何从 DCP checkpoint 恢复的 MXFP8 run 都受影响。**已在 671B / 16 节点上验证修复。
  - **对我的意义**：与上期 #5017（dataloader 状态让 DCP 在 load planning 阶段失败）是**同一失效模式的第二例：存得进、读不出**。结论很明确——**checkpoint 的 CI 必须包含"保存后用全新进程、清空 safe globals 再加载"这一步**，`state_dict()` 往返测试不够。自定义 tensor 子类（量化权重、分片权重）是高危区。
- **#5055 修复分深度 MTP loss 归一化**｜https://github.com/pytorch/torchtitan/commit/ed35bdaede
  - 在数据管线中为每个 MTP 深度计算各自的 loss-token 数与 routing-token 数；每个 MTP 目标和 MoE 辅助 loss 用**与之匹配的分母**归一，跨 packed sequence 与 padding 边界保持正确。RL batcher 提供全局计数，标准 trainer 只做累加/归约。
  - **对我的意义**：MTP 第 d 个头的有效 token 数随深度递减（每条序列尾部少 d 个），packed sequence 下每个子序列都要少，**用主 loss 的分母会系统性低估深层 MTP loss**。我们若用 MTP 头训练 draft 能力，值得核对分母。
- **#4958 / #5070 EP dispatch/combine 成为单一重计算区域；尾部加法设为"保存"以跳过重放**｜https://github.com/pytorch/torchtitan/commit/27eb871155 ｜ https://github.com/pytorch/torchtitan/commit/304b57c1ff
  - #4958：EP>1 时 all-to-all dispatcher 声明 `dispatch` 与 `combine` 两个 remat region，覆盖集合通信及其前后处理。此前 **split sync 是裸算子，即使 `ep_communication` 被保存，也会在每次 backward 重放（每个 MoE 层一次 host 同步）**。另：combine 改为直接用 bf16 expert 输出乘 fp32 router score（乘积仍在 fp32 计算，但为 backward 保存的是 bf16 输入而非两倍大的 fp32 副本）。qwen3 MoE（30B-A3B 宽度，4 层，TP2 EP2）无 AC 时激活显存 **6.5 → 5.5 GiB（每层 256 MiB）**，loss bitwise 相同。
  - #5070：`torch.utils.checkpoint` 在所有被保存的 tensor 重算完后即停止重放，而 `remat.checkpoint` 重放整个 block，导致 block 尾部的残差加与 shared-expert 加被重放、其输入（MoE combine 输出等）从 forward 一直存活到重放。把这些加法设为固定 `recompute=False` 的 region 后激活显存 8.18 → 7.93 GiB（qwen3 MoE）、9.11 → 8.67 GiB（DeepSeek V3 16B），步时不变。
  - **对我的意义**：两条都是"重计算粒度"的细账。**"混合 dtype 乘法时让 autograd 保存较窄的那个操作数"** 是零成本的显存优化，可在我们 MoE combine 处直接检查；"重放不该覆盖不保存任何东西的尾部算子"则是自定义 checkpoint 实现相对 `torch.utils.checkpoint` 容易丢掉的 early-stop 语义。
- **#4639 Kimi K3 KDA 的 context parallelism**｜https://github.com/pytorch/torchtitan/commit/3f087cf151
  - 为 Kimi K3 的 KDA/MLA 混合 decoder 启用 CP。**KDA 层**用 Attention Gym 的原生 CP 实现：各 rank 保留本地序列片段，**交换短卷积的历史（halo exchange），并从前序片段合成递归状态**；**MLA 层**可用 all-gather K/V（contiguous 或 headtail 切分）或 Ulysses（仅 contiguous）。两者复用同一 CP mesh。
  - CP=2 相对 CP=1 的 loss 偏差：step 1 为 0.001%，step 100 为 0.966%（all-gather）/ 1.228%（Ulysses）。局限：不支持 PTRR；要求等长片段（`T % C == 0`，headtail 为 `T % 2C == 0`）；**不支持 sample packing**（路由把整个 batch 当一条全局序列）。
  - **对我的意义**：**线性注意力（delta-rule 类）的 CP 与 softmax 注意力的 CP 是两套机制**——前者要传的是"卷积 halo + 递归状态摘要"而不是 K/V。上期 HLA 条目里提到的"线性注意力在序列并行下的可行性"在这里有了一个工程答案，可作为我们将来支持 KDA/GDN 类层 CP 的参考。step 100 约 1% 的 loss 偏差是否纯属数值噪声，PR 未论证。
- **#5029 [RL] vLLM 引擎跑在独立线程**｜https://github.com/pytorch/torchtitan/commit/e06ff04857
  - 此前 `VLLMGenerator` 的一切都在一个 event loop 上，`LLMEngine.step()` 可阻塞约 30 ms，期间 Monarch actor 收不进请求。改为引擎循环独占线程，endpoint 经线程安全 inbox 与 `concurrent.futures.Future` 交互。
  - 8×H100、2 generator × TP4、每波 512 请求：warm wave 均值 `max_tokens=48` 时 **2.76 → 1.46 s（−47%）**，256 时 −13%，1024 时 −4%。每个 generator 的 rank 0 发起全部 256 个 `generate` 的耗时由 2.26–2.55 s 降到 83–117 ms。
  - **对我的意义**：短生成（工具调用密集的 agentic rollout 正是这种形态）下，**请求准入被引擎 step 阻塞**可吃掉近一半墙钟。我们的 rollout 服务若是"actor event loop 内同步 step"，值得做同样的拆分。
- 其余择要：**#5061** 权重同步使用本地缓存的预计算路由计划（此前每次拉权重都重建路由；也为 PP 下合并多 trainer 的 state_dict 与 torchstore 内 dtype 转换解锁）；**#5057** 训练步数超过 `lr_scheduler.total_steps` 时 **LR 会外推——linear/sqrt 变成负数、cosine 回升到峰值**，现改为钳位到末值；**#5082** `ChunkedLossWrapper` 在 backward 前 `del logits`，Qwen3-8B 峰值显存降 0.58–1.16 GiB、结果 bitwise 相同；**#5045** 新增 `debug.save_parallelism_file`，rank 0 落盘实际 mesh 布局 JSON（主要用途是**核实 EP 是否真的落在节点内**）；**#5079** Qwen3.5 shared expert 的 sigmoid gate 编译成 region（保持 batch-invariant）；**#5074** EMA 支持多份副本；**#5087** HF checkpoint 并行加载；**#4897** attention metadata 归属下放到各 attention backend；**#5091** GraphTrainer 单 stage 支持 MTP。

**vllm-project/vllm-ascend**（窗口内 16 个 commit）

- **#17676 DeepSeek V4 的 TurboQuant INT4 KV cache（A2/A3）**｜https://github.com/vllm-project/vllm-ascend/commit/621e741bad97fbb2299e5bebae2ead3e598e9cff
  - `--kv-cache-dtype turboquant_4bit_nc` + `VLLM_USE_V2_MODEL_RUNNER=1`。**只量化占显存主导的 C4 压缩 KV 状态：每个 512 元素的 C4 状态用 258 字节的 packed INT4 slot（含量化元数据）**；SWA cache、C128 状态等保持原 dtype。算子不入仓，懒加载已安装 CANN 包里的 `cann_ops_transformer.mixed_quant_sparse_flash_mla[_metadata]` 与 `cann_ops_nn.turbo_quant`。框架侧新增 strided cache view（暴露 packed C4 页而不为每次 attention 物化连续 BF16 副本）、DSA query/KV 旋转与输出逆变换。
  - 限制：仅 DeepSeek V4（非 V4.1）、A2/A3、BF16 模型、`head_dim=512`、`index_topk` 为 512 或 1024；**不支持 context parallelism、KV layer parallelism、KV transfer**。不支持的配置在校验阶段失败而非静默回落。
  - **测试只有单元覆盖，commit message 无任何精度或吞吐数据。**
  - **对我的意义**：C4 状态 512 × 2 字节（BF16）= 1024 字节 → 258 字节，**约 4× 密度**，对长上下文并发是实打实的容量提升。但**不兼容 PCP 与 KV transfer**——即与上期 #17598（PCP 切分 O-projection）和 PD 分离都互斥，部署前要先定取舍。无精度数据，建议等上游给出 GPQA 类对照再评估（同日 vLLM 上游的 UltraQuant 4-bit 有完整数据可参照，见下）。
- **#17180 C8-MXFP：FP8 KV cache + E8M0 scale（A5）**｜https://github.com/vllm-project/vllm-ascend/commit/a7f26e5c4e4d9dd7f20fa114f351860bd6552731
  - 基于 QFA（Quant Flash Attention）双算子接口新增 `AscendC8MXFPAttentionBackend`，支持 `K_DYNAMIC_V_STATIC_MXFP8_PER_CHANNEL` 类型——**K 动态 scale、V 静态 scale**；处理 PA_NZ 布局 scatter、静态 V scale cache 填充、ModelSlim 配置映射。
  - **commit message 自述一个未决问题**：`model_runner_v1.py` 中 `k_shape` 构造时漏了第一维 `num_blocks`，会在 `_split_hybrid_c8_mxfp_cache_buffer` 解包时 `ValueError`、随后 `IndexError`。**message 未说明合入版本是否已修**。
  - **对我的意义**：MXFP8（power-of-two scale）KV cache 落地 A5，方向上与 NVIDIA 侧 MXFP8 对齐。但 hybrid 模型路径带着一个自述的 shape bug 合入，**MRV1 + hybrid + C8-MXFP 组合先别用**，等后续修复 commit。
- **#17924 PA 层的 KV cache 保持连续——修复一个 2.25× 的 decode 回归**｜https://github.com/vllm-project/vllm-ascend/commit/ba58e08a83566019add8bb36444669a53ae22895
  - #14340 引入的非连续 block-major 分配使普通 FullAttention 层走 paged attention 时严重降速。改为在 KV 分配时检查 PA 资格并复用 split K/V 分配。
  - **A3、CANN 9.2、QwQ-32B、TP4、`FULL_DECODE_ONLY`、340 请求、并发 60、约 3,500 入 / 1,500 出**：输出吞吐 **456.6 → 1029.6 tokens/s**，mean TPOT **118.8 → 49.7 ms**。
  - **对我的意义**：PA 算子对 K/V 连续性的敏感度是 2× 量级——**"为了某个新特性改 KV 布局"必须带普通稠密模型的性能门禁**。若近期升级过 vllm-ascend 主线且跑的是非 MLA 的普通 PA 模型，这个回归很可能已经在线上。同日配套 #17925（EAGLE3 走非 PA FIA 路径时 K/V 需连续，否则报 `In non-PA scenarios, value tensors must be contiguous`；并把 graph 的 PA/非 PA 模式固定在 capture 时）。
- **#17914 给 `dispatch_ffn_combine` 系列算子显式传 `worldSize`**｜https://github.com/vllm-project/vllm-ascend/commit/cd96b9d33f4b5f98abcf328ce25c37512292e3a5
  - 三个 EP 融合算子的 tiling 通过 `ge::HcomTopoInfo::GetGroupRankSize` 查 HCCL group 的 rank 数。**在 ACLNN 路径上 GE 图执行环境可能未初始化、HCCL group 未注册进 `HcomTopoInfo`，查询失败后 `rankSize` 保持未初始化**，于是用一个垃圾 world size 申请了**约 419 GiB 的 workspace 而 OOM**。
  - 改为调用方显式传 EP world size（与官方 `moe_distribute_dispatch` 一致），tiling 校验 `0 < worldSize <= 768`。2×910_93 上验证：workspace 由 419 GiB 降到约 32 MiB（`max_output_size=512`）/ 约 208 MiB（65536）。**自定义 OPP 包与 C++ 扩展必须同时重建发布**（ACLNN 接口与算子属性列表都变了）。
  - **对我的意义**：**"查询失败但不检查返回值、让输出参数保持未初始化"** 是 C++ tiling 代码里的经典坑，且表现为与拓扑相关的偶发 OOM，极难归因。更一般的教训是：**算子不应从 GE 的全局拓扑表隐式取并行信息**——与上期 Megatron #6258（CI 棘轮禁止从全局读 process group）是同一原则在 CANN 算子层的体现。我们自研的涉及 HCCL 的 ACLNN 算子可按此口径排查。
- **#15079 MRV2 下 spec-decode 的 draft LM head 与 lmhead TP 对齐**｜https://github.com/vllm-project/vllm-ascend/commit/178e4d2fdb4e0302279db86b190b73808293eb97
  - lmhead TP 把 LM head 按 vocab 切在 DP 组上，**组内每个 rank 每步必须给 head 的集合通信喂相同行数**，空闲 DP rank 也不例外。draft 侧在三处与此不符，**任意一处都会导致 HCCL 死锁或启动中止**：形状（忙 rank 采样真实 `num_reqs` 行、空闲 rank 的 dummy propose 采样 dummy-batch 行）、顺序（空闲路径先发 draft 的集合通信再发 target 的）、构造（draft 的稠密 EAGLE/DFlash head 撞上 target 侧仅限 MoE 的细粒度 TP 校验）。
  - 修法：两侧统一 zero-pad 到 `max_num_reqs * (num_speculative_steps + 1)` 再裁回；空闲 rank 在 `execute_model` 尾部先加入 target head，匹配忙 rank 的 target-then-draft 顺序。**无法对齐的组合在构造期显式报错而非挂起**：概率式 draft 采样、DFlash2、`enable_adaptive_verification`、`use_local_argmax_reduction`、prompt logprobs。DSpark 的硬件验证仍待补。
  - **对我的意义**：**"把一个按请求数变化的张量喂给跨 DP 组的集合通信"** 是一整类死锁的根源；解法永远是**定长 pad + 全组同序**。"对不齐的组合在构造期报错"比运行时挂起好一个量级，值得作为我们细粒度 TP 特性的准入原则。
- **#17911 DSpark 的 SAS metadata 与共享可见 KV 对齐（修 hang）**｜https://github.com/vllm-project/vllm-ascend/commit/66a2b645927d65788b40a7017a01c1428dca2141
  - DeepSeek V4 DSpark 在 FP8 SparseAttnSharedkv 下，**draft block 跨越 128-token KV tile 边界时会挂起**。共享 metadata builder 用的是 127 的左窗口，而执行时带 5 个 draft token 用的是 132；causal metadata 还给实际共享同一组显式 non-causal SWA 索引的 query 规划了不同的可见 KV 长度。split-G 执行下 S2 迭代次数因此不一致，**vector core 卡在 barrier 上**。
  - 验证：A5 单算子回放中原始 128-head / 133-visible-KV 输入 25 秒超时；2×A5、16 NPU、DeepSeek-V4-Pro-0813、PCP=8、PP=2、DSpark=5、FP8 KV/indexer：**51/51 流式请求通过**（1–16,384 输入 token，并发至 16）。
  - **对我的意义**：又一例"**metadata 规划与实际执行用了不同的窗口长度 → 多核迭代次数不一致 → barrier 死等**"。这类 bug 只在序列长度落在 tile 边界附近（127/128/129/133）时触发，回归用例应显式覆盖边界值。
- **#17921 V4 IndexCache 在每个 PP stage 的第一个 C4 层初始化**｜https://github.com/vllm-project/vllm-ascend/commit/58421ca59b6efd33ee548751ef62c26846a413af
  - V4 只在 C4 层挂 Indexer，复用调度按 **C4 Indexer 序号**计，而原 PP 校验器用的是 **transformer 层号**——会误拒对齐的切分、也会漏掉跨 stage 依赖。改为：仅当某 PP stage 的第一个 C4 会复用上一 stage 产生的索引时，让它本地重算 Top-K。例（43 主层，`index_topk_freq=4`）：切分 `20,23` 原被误拒、现接受且无额外 Top-K 计算；`21,22` 原被错误接受（层 22 存在跨 stage 依赖）、现由层 22 重算。
  - **对我的意义**：PP 切分点不再必须对齐全局 Top-K 重算层，**层数均衡的自由度回来了**；本方案是"本地重算"而非"跨 stage 传索引"（#15614 的另一路线），省通信但多一次 Top-K。
- **#17828 GLM packed causal convolution 的首个有效行被当成 padding**（上期备注中"差几小时落在窗口外"的那条，本期入窗）｜https://github.com/vllm-project/vllm-ascend/commit/7a058273d7728ffac2021f5559d8f576ea73afc2
  - `copy_conv_state` 把有效请求映射到 packed 行 `0, 1, …`、无效请求映射到 `PAD_SLOT_ID`，而 wrapper 给 FLA 的 prefill 与 decode 都传了 `null_block_id=0`——**第一个有效 packed 行被视为 padding，kernel 跳过计算且不更新状态，返回值可能为零或未初始化（含 NaN）**。改用 `PAD_SLOT_ID`。
  - 真实 910B3（CANN 9.1、torch_npu 2.10.0.post4）：修前 12 个 packed 布局用例全挂，修后 24/24 通过。
  - **对我的意义**："**0 既是合法索引又被当作空哨兵**"的经典错误。跑 GLM-5.x 且用非连续卷积状态布局的服务，第一个请求的输出一直是错的。
- 其余：**#17922** GLM-5.3-Flash + C8 KV 启动失败（Ascend 首轮配置含 packed C8 scale 字节算出 block size 8704，后续通用 MLA 对齐不含这些字节改成 8960，两处不一致；触发源是 #14340 给 `update_block_size_for_backend` 加了 `super()` 调用）；**#16655** A3 上 W4A8 的 **GMM + dequant + SiTU + per-token INT8 量化四合一融合算子**（`torch.ops._C_ascend.gmm_dequant_situ_quant`，自动启用、无开关；message 无性能数据）；**#17698** GLM-5.3-Flash layerwise memcache DRAM/SSD pooling（**128K cold/hot 分歧仍未解决**，作者自述）；**#17913** KDA 直接调 CompiledKernel 时漏传 3 个 constexpr 参数（Triton 3.6 on Ascend 报 `launch expects 21 arguments, got 18`）；**#17209 / #17937** 自定义 QLI 与 KvQuantSparseFlashAttention 算子改名加 `Vllm` 后缀，**避免与 CANN 9.2 / A5 内置同名算子的注册冲突**（直接调用旧自定义名的代码需更新；#17937 作者自述未做 NPU 验证）；**#17919** 自定义推测模型下 GLM C8 的 `enable_fa_quant` 判定修正。

**NVIDIA/Megatron-LM**（窗口内 10 个 commit）

- **#7885 修复 CUDA graph：在全迭代捕获开始时预置 RNG 捕获状态**｜https://github.com/NVIDIA/Megatron-LM/commit/d6316eb45bf1bfbb6f1f848c0b56e55166564d80（message 仅一行）
- **#7701 把量化 `RecipeConfig` 存进 checkpoint yaml**｜https://github.com/NVIDIA/Megatron-LM/commit/e4294782fb94808dff2b2ed6818f27960c9c6b60（message 仅一行）
- **#7894 TE 升到 2.20 分支 tip**；#7842 GDP decode 输出用 `reshape` 而非 `view` 展平；#7855 / #7839 MIMO 多模态相关；#7587 `LoggerConfig` 迁移；#7741 / #7887 / #7916 为配置与测试杂项。
  - **对我的意义**：信息量不大。唯一值得留意的是 #7701 的方向——**量化 recipe 随 checkpoint 持久化**，避免恢复时用错精度配置；与 torchtitan #5094 是同一关切（低精度训练的 checkpoint 一致性）的两个面。#7885 表明"**整个迭代放进 CUDA graph 时 RNG 状态的捕获时机**"仍在出问题，dropout / 随机舍入（NVFP4）路径受影响，但 message 未给症状。

**NVIDIA/TransformerEngine**（窗口内 4 个 commit）

- **#3622 多张量 swizzle kernel 补 `__launch_bounds__`**（上期备注中的另一条，本期入窗）｜https://github.com/NVIDIA/TransformerEngine/commit/b47a356561ddfa8740cb86cd4dc176606bd93e31
  - 这四个 kernel 以 `TB_DIM × TB_DIM`（1024）线程发射，却不像其它 swizzle kernel 那样带 `__launch_bounds__`，ptxas 可能为每线程用超过 64 个寄存器（int4 变体可达 99），发射时报 **"too many resources requested for launch"**；哪些变体失败取决于目标架构与 CUDA 版本。
- **#3595 SM12x 上确定性 backward 跳过 FA4**；**#3592 修正 MXFP8 的 ONNX block 量化属性**；#3620 移除 CuTeDSL 首次发射保护。
  - **对我的意义**：#3622 属于"换一版编译器就挂"的典型——**寄存器预算是隐式契约**。昇腾侧无直接对应，但"kernel 的资源上界应显式声明而非依赖编译器当前行为"是通用原则。

**deepspeedai/DeepSpeed**（窗口内 2 个 commit）

- **#8638 无效 group-norm 哨兵不再变成有限的 1.0 步**｜https://github.com/deepspeedai/DeepSpeed/commit/991ccf0f2e343b981a04289b2f1786620bf29869
  - DeepSpeed 用 **`-1` 编码 inf/NaN 的 group norm**。用 `vector_norm` 合并各组时哨兵被**平方成 `1.0`**，于是**梯度裁剪与 Adam 把一个失败的 group 当作健康的，施加了一次未裁剪的更新**。改为把非有限或负的 group norm 视为溢出（`inf`）、跳过 optimizer step、每步重置 overflow、loss scaler 只前进一次；覆盖 ZeRO-3 / SuperOffload / ZenFlow / FP16 fused+unfused / BF16。
  - **对我的意义**：**"用带内哨兵值表示异常，再被后续数学运算洗白"**——NaN 梯度没有被跳过而是变成一步范数为 1 的正常更新，loss 曲线上几乎看不出来。我们的梯度范数/溢出检查路径应确认异常是用**带外标志**传递的，且在跨参数组、跨 rank 归约之后仍然保真。

**vllm-project/vllm**（窗口内 69 个 commit；细节来自 PR 描述，经子 agent 提取）

- **#57635 FlashInfer autotune cache 按 rank 持久化——修"只有 rank 0 命中 cache"的死锁**｜https://github.com/vllm-project/vllm/commit/3403e0f176efb90b0d92cc0ff1d350f092ed9e9f
  - autotune cache 只由 rank 0 落盘、下次启动广播给所有 rank；但 MoE 条目按 `tp_rank` / `ep_rank` 做 key，**只有 rank 0 能命中**。命中会跳过同步 tuning 的 per-tactic reduce，于是 rank 1..N−1 卡在 reduce 里、rank 0 在尾部 barrier 空转，**直到 30 分钟超时**。任何能看到该文件的**第二次**引擎启动都会触发。现场：8×H100、TP8+EP8 FP8 MoE，kubelet 8.5 小时内原地重启 18 次全部失败；挪走 cache 目录后 20 秒 READY。
  - **对我的意义**：**"首次启动正常、重启必挂"** 且症状是 GPU 0 满载、其余 0%。凡是"rank 0 落盘、全员广播"的缓存（autotune、编译缓存、profile 结果），只要 key 里含 rank 就是这个坑。我们的 torch.compile / ACL graph 缓存复用逻辑可对照。
- **#57057 UltraQuant 4-bit KV cache 后端**｜https://github.com/vllm-project/vllm/commit/385d86a6cf6f05051899b53d83ec55c63cccb379
  - `--kv-cache-dtype ultraquant_4bit`：**FP4 E2M1 码 + 每 32 元素一组的 UE8M0（2 的幂）scale + Hadamard 旋转的 key**，相对 fp8 约 2× KV 密度。GPQA-Diamond（多 seed 均值）UQ 4-bit / fp8：Qwen3.8 93.27% / 92.42%，Qwen3.6-27B 86.87% / 85.61%。
  - 262K token agentic replay（MI355X，Qwen3.8，TP=8）：低并发下慢 2.7%，**并发 38 时 414.6 vs 225.4 tok/s（+84%）、并发 42 时 430.7 vs 114.7（+275%）**——收益完全来自 fp8 在高并发下 KV 用尽（peak 100%）而 4-bit 只用到 44–45%。需 FlyDSL-capable ROCm build；各基准测自不同软件基线，绝对值仅供参考。
  - **对我的意义**：给上面 vllm-ascend TurboQuant INT4 提供了一个**同日的、带精度与并发曲线的参照**：4-bit KV 的价值不在单请求速度（略负），而在**把 KV 耗尽的拐点推后**。评估昇腾 TurboQuant 时应直接测"并发—吞吐"曲线的拐点位置而不是单点 TPOT。
- **#49194 去掉 NCCL symmetric reduce-scatter 的 staging 拷贝**｜https://github.com/vllm-project/vllm/commit/4a99ca40d647a0b5618315fe44fcb85554ddcb53
  - 原路径"expert 输出 → 拷到已注册输入 → ReduceScatter → 拷到最终输出"，改为 expert kernel 直接写已注册输入、ReduceScatter 直接写最终输出。前提 `VLLM_USE_NCCL_SYMM_MEM=1`、token 数均匀、NCCL ≥ 2.29.2。4×GB200 上算子延迟降 9.6–16.0%；E2E（Nemotron Ultra 550B A55B NVFP4，TP1/DP4/EP4）50,000/2,048 负载吞吐 +8.91%。作者自述并发 512 时 TTFT 六次全部变差、rebase 后的 head 未重测。
  - **对我的意义**：symmetric memory 的收益要靠**"计算 kernel 直接写进已注册窗口"**才能兑现，否则两次拷贝会吃掉大半。HCCL 侧若做同类零拷贝集合通信，算子输出缓冲的注册方式要一并设计。
- **#55804 EPLB 的 torch P2P 传输改用组内 rank**｜https://github.com/vllm-project/vllm/commit/15c1cc12ac50c1fb36fd92978c14ad2e2f865055
  - 开 PP 时 EPLB 把 EP 组内的 peer rank 传给 communicator，而 torch NCCL/Gloo 后端把它当**全局 rank**——EP 组含全局 rank `[2, 3]` 时，本地 peer `1` 被解释成组外的全局 rank 1，expert 权重迁移失败。改为显式传 `group=` 与 `group_peer=`。本地验证走的是 Gloo，非 NCCL 设备传输验证。
  - **对我的意义**：**组内 rank 与全局 rank 混用**在 PP×EP 组合下必现；我们 HCCL 侧 EP 负载均衡若做 expert 权重迁移，可照此口径检查。
- **#59128 block-FP8 DeepGEMM experts 跳过 padding 的无效工作**｜https://github.com/vllm-project/vllm/commit/5ab4e4445bcbe2bf07351b82d1af160234c32273
  - 两个 grouped GEMM 改用 DeepGEMM 的 prefix-sum scheduler，融合激活量化使用同一组 live expert 端点，不再访问未用的 padded tile。512 容量 / 16 live token 时 B200 472.2 → 347.5 µs（−26.5%）、H200 694.4 → 438.1 µs（约 −37%）；满载时仅 4–8%。输出 bitwise 一致。SM103 / 分布式 DeepEP V2 未测。
- **#59103 session 截断 token 时丢弃过期的 block hash**｜https://github.com/vllm-project/vllm/commit/2b32d9ac6a910faf786a2904b87d5abed044c228
  - streaming-input session 续接时截断 token 序列，而 block hash 只追加不回收；若被丢的 token 恰好补满一个 hash block，**旧 hash 仍标识被替换的序列，使前缀匹配旧 token 的请求命中由新 token 算出的 KV**（prefix cache 串数据）。
  - **对我的意义**：prefix cache 的 hash 与 token 序列必须**同生共死**；任何"截断/回滚 token"的路径（推测解码回滚、session 续接、RL 轨迹截断）都要同步回收 hash。
- 其余：**#60140** GLM-5.3 默认切到 fp8 KV cache（KV 40.13 → 20.69 GiB/rank，吞吐 +2–5%，**TTFT p50 回退 7–11%**）；**#55684** 加载时把 ModelOpt MXFP8 linear 重量化为 FP8 PTPC（端到端评测未做）；#56881 Kimi-K3 decode-context-parallel prefill 用融合 K/V pack kernel；#59532 DSv4.1 decoder replay 层进 CUDA graph。

**sgl-project/sglang**（窗口内 64 个 commit；细节来自 PR 描述，经子 agent 提取）

- **#41395 [PD] Mooncake 提前传 KV 前等待 prefill 完成**｜https://github.com/sgl-project/sglang/commit/c683190b40f137f91f8d82fdf7df64ea028f888b
  - overlap scheduling + chunked prefill 下，Mooncake sender 可能在某 chunk 的 forward **写完 KV 之前**就把它交给 transfer worker——等待用的 event 被记录了但没被传递、也没人等。结果 "The decode then receives zero or stale KV for most of a long prompt, and **every transfer still reports success**."
  - MI355X、GLM-5.2、PD prefill TP4 → decode TP4、Mooncake RDMA、491K token cold prompt：修前 **30 个 chunk 中 25–29 个在 forward 完成前被读取**，decode 侧首个 cold 请求收到 **31.7M–34.2M / 38.3M 行全零 KV**（4/4 新起的 server 复现）；长上下文 needle prompt **0/6 正确 → 6/6 正确**。稳态吞吐不变。
  - **对我的意义**：**PD 分离下最危险的失效模式——传输层全绿、内容全错**。任何"边算边传"的 KV 早传优化，都需要一个**内容级校验**（至少对首个 cold 请求抽查非零率/校验和），不能只看传输成功状态。我们昇腾 PD 部署的 KV 传输链路值得加同样的探针。
- **#42625 prefill-delayer 的队列超时判定跨 rank 汇总**｜https://github.com/sgl-project/sglang/commit/90aa1cb0fbc0b99ba74a18a0df44151da40a59d5
  - 其它决策输入都经 all-gather 共享，唯独 wall-clock 上限用各 rank 自己的 `perf_counter()` 判——**落在 deadline 两侧的 rank 走不同分支 → 同一 TP 组组出不同的 batch → 集合通信不匹配 → 挂起**，"shows up as an NCCL hang rather than an obvious scheduler bug"。修法是在已有 all-gather 的信息里多带一个字段，按 TP0 的标志统一决策。
  - **对我的意义**：**任何基于本地时钟的调度分支都必须先达成全组一致**。这是多 rank 调度器的硬规则，与上面 #15079 的"全组同形同序"是同一原则的时间维度版本。
- **#32779 面向 DSA 的 Triton sparse-MLA prefill 后端（SM90/SM120）**｜https://github.com/sgl-project/sglang/commit/e446698cdd09ffa9c6b464228d6d98db2a311be6
  - 动机：`flash_mla_sparse_fwd` 要求 `num_heads % 64`（SM90）/ `% 128`（SM100+），GLM-5.x 在 TP8 下每 rank 只有 8 个真实 head，**tensor core 的 7/8（SM100+ 为 15/16）工作量花在 padding 上**。新后端把 head 维只 pad 到 16 行 MMA tile；`--dsa-triton-union {2,4}` 让相邻 token 共享一次 gather 的索引并集、用 per-row ownership bitmask 恢复各自的 softmax 支撑（作者称是代数改写而非近似）。
  - SM90 E2E（GLM-5.2，78 层，TP8）4096 token prefill 0.888 s → 0.516 s（union2，1.72×）。**训推一致性相关**：union 在 `--enable-deterministic-inference` 下被强制关闭（token 输出会依赖与谁同组）；SM90 FP8 栈上 4096 token 起 reference 自身重跑的 rel L2 就漂 3–7%，"A greedy-token match is therefore not a usable metric on this stack"。
  - **对我的意义**：**"TP 把 head 数切到远小于算子的 tile 对齐要求"** 是 MLA/DSA 类模型在高 TP 下的普遍浪费，昇腾 FIA/SFA 算子在 TP8 下是否有同类 padding 开销值得量一下。另一条——**FP8 栈上 greedy token 匹配已不能当正确性指标**——对我们做量化推理回归的方法论有直接影响。
- **#42750 / #42489**：#42489（GLM-5.3-Flash DFlash decode 在 8×B300 上 1091.9 → 1199.7 tok/s，+9.88%）因未 rebase 合入，**引入页表 id 放大 `pool_size` 倍（GLM-5.3-Flash 为 4×）的回归**，GSM8K 0.930 → 0.892；约 7 小时后由 #42750 修复。**对我的意义**："两个各自正确的 PR 交错合入产生错误结果"——合并队列若不强制 rebase + 重跑精度门禁，这类问题无法靠 review 发现。
- **#42467**：带 `skip_radix_cache_insert` 的请求在 chunked prefill 下 `prefix_indices` 不前进，导致 KV 池泄漏或 livelock；触发字段是普通请求字段，**任意客户端可用一条长 prompt 打挂非 PD server**。**#42659**：MLA KV/Q 的 BF16 → FP8 转换改用饱和语义——**有限溢出与 ±inf 现在钳到 ±448 而不是变成 NaN**（1024 行 9.53 → 3.10 µs）。

**pytorch/pytorch**（窗口内 63 个 commit，按分布式/编译关键词筛选）

- **#192524 ROCm 上启用 NCCLSymmetricMemory（RCCL）**｜https://github.com/pytorch/pytorch/commit/e64cfa6fe9f4b98e1a99c5eee573c526f4c29efa
  - 逐条列出不能复用 CUDA 路径的原因：RCCL `ncclMemAlloc` 无 CUMEM 时回落普通分配而被 capture 拒绝；graph capture 期间的同步 memset/memcpy 会杀掉 capture；录到 capturing stream 上的异步 memset 成为 graph node，"can wipe a peer signal pad on every replay"；fused 多目的 reduce-scatter 在特定架构上触发内存越界；PG 销毁后保留的 handle 经已死的 communicator 发 barrier kernel。方案是 per-device 管理器在 communicator 注册时快照 device-API 支持、新窗口用非 capturing 的 setup stream 异步清零、reduce-scatter 每个目的地发一个 kernel、stale handle 直接报错而非重建。**有 5 处改动未加 ROCm 宏保护、会改变 CUDA 行为**（具体清单未读出）。无性能数字。
  - **对我的意义**：这是**"第二家硬件接入 symmetric memory"的完整踩坑清单**，尤其是"**graph capture 期间不得有同步内存操作**"和"**异步清零若被录进 graph 会在每次 replay 抹掉对端信号**"两条，HCCL + ACL graph 的组合大概率会遇到同构问题。
- **#177961 ROCm 启用原生 AsyncTP**｜https://github.com/pytorch/pytorch/commit/a476c633b36df133775c886b3c10b12ad0087d9e
  - `fused_all_gather_matmul` 把 activation shard 的 all-gather 与 `A @ B` 重叠：persistent GEMM 在读每个 gathered chunk 前等待 per-chunk signal。**舍入一致性**：ck_tile 在 gfx942 上默认把 fp32 → bf16 做截断，"about half of the 4.19M outputs were one bf16 step from the correctly rounded value"，现改为 round-to-nearest-even。**死锁类发现**：persistent grid 必须在每个 shader engine 留一个空闲 CU，按总数预留会让大 GEMM 必然死锁；沿用 CUDA 的发射顺序在 2 rank 下 3/3 挂起，故改为 peer copy 先于 GEMM 发出。选择器只查 `local_M % 256 == 0`，M=4096 / 2 rank 时对很多 `K ≥ 1024` 的形状反而更慢。
  - **对我的意义**：通算重叠（AsyncTP）移植到非 NVIDIA 硬件的两个通用教训：**(1) 发射顺序与空闲计算单元预留是死锁的主要来源，不能照搬 CUDA 的顺序；(2) fp32 → bf16 的默认舍入模式在不同后端可能不同**——后者是跨硬件训推一致性里很隐蔽的一项，昇腾侧 Cast 算子的舍入模式应明确核对。
- **#199775 修复 memory-efficient attention backward kernel 缺失的 `cp.async` drain**｜https://github.com/pytorch/pytorch/commit/b50a07471edbdf47bdd042a13b774d3a7a1664d2
  - CUTLASS mem-efficient attention 的 backward kernel 里，每个 sm80+ GEMM 的 mainloop 结束时最后一组 `cp.async` 仍在飞行，后续代码在无人等待的情况下写入与其目标别名的 shared memory。"This one also reaches ordinary shapes: **head_dim 128 is affected**, whereas the forward race needs head_dim >= 256." H100 上 racecheck：f16 m=129 k=512 由 10 个 error 降到 0。
  - **对我的意义**：**走 `scaled_dot_product_attention` 的 mem-efficient 后端（非 FlashAttention）做训练的代码，head_dim=128 的梯度一直存在竞争**。我们在 GPU 上跑的对照基线若走此路径，其"正确性"需打折扣。
- **#199618 symm_mem：multimem barrier 的计数器移出 signal pad**：barrier 的到达计数与 rank 0 的 signal 标志共用一个 word，barrier 到达落在 pending signal 上会使该 word > 1，`wait_signal` 永不返回。回归测试在父提交上 **20/20 死锁**。
- **#198104 [inductor] alias buffer 被 mutate 前先 realize 其消费者**：`torch.compile` 下 mutation 节点只依赖原 buffer 的 producer，fill kernel 先跑，alias 的消费者用到了 mutate 之后的值——**静默错误结果**，2.14.0 上可复现。**#199811** `torch.add(inp, a @ b, alpha=0.5)` 融合成 addmm 时**丢掉 alpha**（自 addmm 融合引入起就存在）。**#198737** LayerNorm 常数输入返回非零的修复**会改变所有 GPU 后端非常数输入的 bitwise 前向输出**（CUDA 侧仅经算术推理、未实测）。
  - **对我的意义**：三条都影响"升级 torch 后数值变了"的排查。尤其 #198737——**升到含此修复的 nightly 后 LayerNorm 输出 bitwise 变化是预期行为**，建议与上期 #198668（FSDP2 梯度累积变更）一起记入版本矩阵。

**NVIDIA/TensorRT-LLM**（窗口内 24 个 commit，仅浏览标题）：#19731 MiniMax-M3 的 NVFP4 hybrid cache 原生 P128 draft KV view；#17306 C++ KVCacheManagerV2 支持 beam search；#19830 `kdaDecode` 在 lane-0 block-reduce 写入前补 `__syncwarp`（又一例 kernel 内同步缺失）。均未读正文。

**volcengine/verl、OpenRLHF/OpenRLHF、THUDM/slime、Ascend/pytorch（torch_npu）**：窗口内 **0 commit**。

**MindSpeed / MindSpeed-LLM / MindSpeed-RL**：**未确认**（Gitee 抓取超时、GitCode 返回 418；release notes 页可读到的最新仍为 MindSpeed-LLM 26.0.0，2026 年 4 月，配套 PyTorch 2.7.1 / TorchNPU 26.0.0 / CANN 9.0.0）。"未确认"不等于"没有"。

---

## Sources

- Mistral Large 4：https://mistral.ai/news/mistral-large-4 ｜ https://docs.mistral.ai/getting-started/changelog/
- OpenAI math：https://openai.com/index/sharing-ai-progress-in-mathematics/ ｜ https://github.com/openai/math
- allenai/miles-olmo-core：https://github.com/allenai/miles-olmo-core
- ThunderSyncRL：https://arxiv.org/abs/2610.05935
- CIPHER-MoE：https://arxiv.org/abs/2610.05744
- FC-SWE：https://arxiv.org/abs/2610.07898
- LoGRA：https://arxiv.org/abs/2610.06647
- ORCA：https://arxiv.org/abs/2610.06116
- Request Order Matters：https://arxiv.org/abs/2610.05833
- NVIDIA AICR v1.0：https://developer.nvidia.com/blog/aicr-v1-0-open-stable-and-verifiable-gpu-cluster-configuration/
- NVIDIA Green Contexts：https://developer.nvidia.com/blog/control-how-your-gpu-shares-work-with-green-contexts/
- trl v1.14.2：https://github.com/huggingface/trl/releases/tag/v1.14.2
- Megatron-LM 26.09 rc：https://github.com/NVIDIA/Megatron-LM/releases/tag/26.09-beta.rc1
- TransformerEngine v2.20.2：https://github.com/NVIDIA/TransformerEngine/releases/tag/v2.20.2
- torchtitan：#4923 https://github.com/pytorch/torchtitan/commit/36adc3a30df688ba1fc47d30827f2d4954bf52a2 ｜ #5081 https://github.com/pytorch/torchtitan/commit/1e3c22236469c91abbfd2e2ba2a04f71a8509a14 ｜ #5067 https://github.com/pytorch/torchtitan/commit/17bf1f6cab ｜ #5094 https://github.com/pytorch/torchtitan/commit/4b56d82cf4 ｜ #5055 https://github.com/pytorch/torchtitan/commit/ed35bdaede ｜ #4958 https://github.com/pytorch/torchtitan/commit/27eb871155 ｜ #5070 https://github.com/pytorch/torchtitan/commit/304b57c1ff ｜ #4639 https://github.com/pytorch/torchtitan/commit/3f087cf151 ｜ #5029 https://github.com/pytorch/torchtitan/commit/e06ff04857
- vllm-ascend：#17676 https://github.com/vllm-project/vllm-ascend/commit/621e741bad97fbb2299e5bebae2ead3e598e9cff ｜ #17180 https://github.com/vllm-project/vllm-ascend/commit/a7f26e5c4e4d9dd7f20fa114f351860bd6552731 ｜ #17924 https://github.com/vllm-project/vllm-ascend/commit/ba58e08a83566019add8bb36444669a53ae22895 ｜ #17914 https://github.com/vllm-project/vllm-ascend/commit/cd96b9d33f4b5f98abcf328ce25c37512292e3a5 ｜ #15079 https://github.com/vllm-project/vllm-ascend/commit/178e4d2fdb4e0302279db86b190b73808293eb97 ｜ #17911 https://github.com/vllm-project/vllm-ascend/commit/66a2b645927d65788b40a7017a01c1428dca2141 ｜ #17921 https://github.com/vllm-project/vllm-ascend/commit/58421ca59b6efd33ee548751ef62c26846a413af ｜ #17828 https://github.com/vllm-project/vllm-ascend/commit/7a058273d7728ffac2021f5559d8f576ea73afc2
- Megatron-LM：#7885 https://github.com/NVIDIA/Megatron-LM/commit/d6316eb45bf1bfbb6f1f848c0b56e55166564d80 ｜ #7701 https://github.com/NVIDIA/Megatron-LM/commit/e4294782fb94808dff2b2ed6818f27960c9c6b60
- TransformerEngine #3622：https://github.com/NVIDIA/TransformerEngine/commit/b47a356561ddfa8740cb86cd4dc176606bd93e31
- DeepSpeed #8638：https://github.com/deepspeedai/DeepSpeed/commit/991ccf0f2e343b981a04289b2f1786620bf29869
- vllm：#57635 https://github.com/vllm-project/vllm/commit/3403e0f176efb90b0d92cc0ff1d350f092ed9e9f ｜ #57057 https://github.com/vllm-project/vllm/commit/385d86a6cf6f05051899b53d83ec55c63cccb379 ｜ #49194 https://github.com/vllm-project/vllm/commit/4a99ca40d647a0b5618315fe44fcb85554ddcb53 ｜ #55804 https://github.com/vllm-project/vllm/commit/15c1cc12ac50c1fb36fd92978c14ad2e2f865055 ｜ #59128 https://github.com/vllm-project/vllm/commit/5ab4e4445bcbe2bf07351b82d1af160234c32273 ｜ #59103 https://github.com/vllm-project/vllm/commit/2b32d9ac6a910faf786a2904b87d5abed044c228
- sglang：#41395 https://github.com/sgl-project/sglang/commit/c683190b40f137f91f8d82fdf7df64ea028f888b ｜ #42625 https://github.com/sgl-project/sglang/commit/90aa1cb0fbc0b99ba74a18a0df44151da40a59d5 ｜ #32779 https://github.com/sgl-project/sglang/commit/e446698cdd09ffa9c6b464228d6d98db2a311be6 ｜ #42750 https://github.com/sgl-project/sglang/commit/45db4305303a106988a65a1855a5183054c3837d
- pytorch：#192524 https://github.com/pytorch/pytorch/commit/e64cfa6fe9f4b98e1a99c5eee573c526f4c29efa ｜ #177961 https://github.com/pytorch/pytorch/commit/a476c633b36df133775c886b3c10b12ad0087d9e ｜ #199775 https://github.com/pytorch/pytorch/commit/b50a07471edbdf47bdd042a13b774d3a7a1664d2 ｜ #199618 https://github.com/pytorch/pytorch/commit/e865d22ba219a00a04d4adff6855c9d612cbcda5 ｜ #198104 https://github.com/pytorch/pytorch/commit/dae455f09ffe2b2d70c5079eff3305940f1fe5ee ｜ #198737 https://github.com/pytorch/pytorch/commit/dca0b4b608bd6358abea8b78f0eb269f615254bc
- 窗口外线索：https://pytorch.org/blog/pytorch-hardware-enablement-updates-from-the-acceleration-integration-working-group/

## 运行备注

- **窗口**：严格取北京时间 10-06 08:00 → 10-07 08:00。本次实际运行于 10:17（计划 08:08），08:00 之后的内容未收。
- **去重**：已对照 10-06 期。Beam、Kolibri、HLA、vLLM v0.31.0、torchtitan #5049 等均未重复。标注「更新」的一条：*Request Order Matters*（上期仅列标题，本期补全文精读）。上期备注里"差几小时落在窗口外"的 vllm-ascend #17828、TransformerEngine #3622、torchtitan #5007 / #4850 本期已入窗并收录（后两条为配置整理与 CI，未展开）。
- **破例补录一条窗口外内容**：TransformerEngine **v2.20.2**（10-05 21:14 UTC）落在**上期**窗口内但上期漏收（上期只扫了 TE 的 commit、把 Release 列表写成"窗口内无"）。因其 Known Issue 涉及**静默错误梯度**，本期以【补录】标注收录，未计入"今日要点"。
- **论文日期口径**：ThunderSyncRL / CIPHER-MoE / LoGRA / ORCA 的 v1 为 10-05 提交，按 arXiv 流程于 10-06 00:00 UTC 公告，本期按"公告落在窗口内"收录；其中仅 LoGRA 的提交时间经 arXiv abs 页确认到时分，其余三篇只有 PDF 首页日期戳，**公告批次未能一手核实**（arXiv 列表页滞后）。若坚持按 v1 提交时间，则窗口内只有 FC-SWE 一篇。
- **验证强度（请按此折扣阅读）**：
  - 本会话**直接读到原文**的：torchtitan、vllm-ascend、Megatron-LM、TransformerEngine、DeepSpeed、trl、TensorRT-LLM 的 commit message 与 Release notes（经 GitHub 工具）；Mistral Large 4 博客（经 WebFetch 抽取，带引号的句子为要求逐字引用后的返回，benchmark 表为摘要式抽取）。
  - **经子 agent 读取、本会话未逐条复核**的：全部 6 篇论文；vllm / sglang / pytorch 三个仓库的合入细节（vllm 与 sglang 的 commit message 只有标题，细节来自 PR 描述）；NVIDIA 两篇博客；OpenAI math 与 miles-olmo-core。
  - LoGRA 仅基于 alphaXiv 的 AI 生成结构化报告，公式未对照原文。CIPHER-MoE 摘要中"Top-1 expert 负载下降 64.9 个百分点"一句未在已读正文中定位到出处，**未采用**。
  - Mistral Large 4 的"每卡每天约 1100 万 token""可训练占比约 48%"为**本日报据官方两个数字推算**，原文未给。
  - Megatron #7885 / #7701 的 commit message 仅一行；两个 26.09 rc tag 的 release body 为空。vllm-ascend #17676 无精度/性能数据；#17180 带着作者自述的 shape bug 合入，是否已修未说明；#17937 作者自述未做 NPU 验证。
- **检索失败或受限**：
  - arXiv abs 页（2610.05744 / 05935 / 07898 / 06116）代理 **429**，未重试；`arxiv.org/list/cs.DC/new` 可抓但只到 10-05 公告批次；cs.LG / cs.CL / cs.AR 列表在出现 429 后未再尝试。
  - `huggingface.co/api/daily_papers?date=` 返回 400，改抓 `huggingface.co/papers?date=2026-10-06` 成功（当日榜单以视频/VLA/agent 评测为主，无训练 infra 高票论文）。HF 组织 API 的 curl 被代理 403，改用 WebFetch 全部成功。
  - alphaXiv 仅执行 2 次主题检索（工具限额），FP8/MXFP8 低精度训练、SDC/容错、checkpoint、集合通信、编译器、PD 分离等主题**未命中窗口内论文——可能确实没有，也可能是检索不足**。hot feed 只返回 id 未逐个解析。
  - 博客抓取失败：LMSYS/SGLang Blog、Modal、Anyscale（均只返回导航骨架，疑为 JS 渲染）、华为计算新闻页；昇腾社区首页无带日期的新文。机器之心/量子位未单独抓取。
  - 中英文 WebSearch 对"10 月 6 日厂商发布"返回的多为旧闻或二手聚合页，未采信。**国内厂商（DeepSeek、阿里、Moonshot、智谱、MiniMax、字节、腾讯、百度、阶跃、小米、美团）的官方博客/公众号未直接核查**，仅有 HF 与 GitHub 侧的否定证据——若某家只在公众号或官网发布，本期会漏。
  - 厂商 GitHub org 扫描本期**已执行**（15 个 org 的 `pushed:>=` 与 `created:>=`），但多 org 合并查询下按 org 归属的"无"系推断；NVIDIA、openai、MiniMax-AI 等 org 的例行更新未逐一看。
  - 硬件厂商：Intel Gaudi、Google TPU、AWS Trainium、Groq、寒武纪、海光的官方 newsroom 未直接抓取，仅靠搜索（US-only，中文覆盖有限）。OCP Summit 日期仅有二手来源。
  - MindSpeed 系列、CANN：Gitee 超时、GitCode 418，未能在窗口粒度核对。
- **窗口外但值得补读的线索**：
  - **PyTorch Blog 10-05《PyTorch Hardware Enablement: Updates from the Accelerator Integration Working Group》**——与 torch_npu 高度相关：Cross-Repository CI Relay（OIDC 校验，L1–L4 分级）、60 万+ 测试用例的 device-generic 重构、OpenReg（PrivateUse1 参考后端）、`REGISTER_PRIVATEUSE1_PROFILER`、c10d 自定义后端 OCCL（RFC #176877）、Dynamo/Inductor 的 out-of-tree 后端支持（RFC #181093）。文中未提 Huawei/Ascend。属上期窗口，上期未收。
  - Fireworks 09-30《Reinforcement learning: Why alignment of numerics and MoE routing matter》；Cerebras 10-01《Disaggregated Inference From The Ground Up》；arXiv 2610.03415 RailWave、2610.03286 VenusRL。
- **保存结果**：
  - GitHub：随本次提交写入 `suhaibo666/tracker` main 分支 `llm-training-daily/llm-training-daily-2026-10-07.md`（经 GitHub MCP 工具，新建文件）。
  - 本地：`device_commit_files` 首次写入**失败**，原因与上期相同且与"电脑离线"无关——设备桥接在线，但本会话**没有任何已连接文件夹**，报 "Files can't be written here without a grant"。未重试写入；GitHub 保存完成后会发起**一次** `device_request_folder_access`（目标 `/Users/suhaibo/workspace/90-knowledge/llm-monitor`），若有人批准则补写本地副本，结果以最终回复与通知为准。
  - **一次性修复建议（同上期）**：在该电脑的 Claude 桌面端用「Add folder」把 `~/workspace/90-knowledge/llm-monitor`（或上层 `~/workspace`）加为这个定时任务的连接文件夹，之后本地备份即可自动完成。在此之前本地副本请从 GitHub 拉取。
