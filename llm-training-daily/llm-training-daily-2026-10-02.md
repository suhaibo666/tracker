# LLM 技术动态日报 · 2026-10-02

> 窗口：2026-09-30 08:00 → 2026-10-02 02:35（北京时间，约 42.5h；因上一期为 09-24，09-25～09-29 的间隙不在本期范围） · 生成时间：2026-10-02 02:35

## 今日要点

1. **DeepSeek 开源面向 Ascend 950 的 DeepGEMM-Ascend 与 DeepEP-Ascend**（09-30）：AscendC + CANN 9.2.0 原生实现，非 GPU 代码移植。Dense GEMM 利用率 BF16 99.8% / FP8 99.5% / FP4 98.3%；MegaMoE（EP384）峰值 846 TFLOPS；DeepEP-Ascend EP8 dispatch 373 GB/s、EP128 降到 313 GB/s。这是昇腾 950 上 GEMM 与 EP 通信第一份公开可复现基线。
2. **verl 合入 score centering（#8010）**：针对训推不一致的加性、无超参修正，on-policy 时为 no-op。fp8 rollout + bf16 trainer 的 Qwen3-0.6B Countdown 复现中，`bypass_pg` 第 72 步崩塌（acc 0.00），`bypass_pg_sc` 达 0.67。目前仅支持 vLLM + FSDP + v1 trainer，**不支持 Megatron**，Ascend 侧需自行移植（slime 的 Megatron+SGLang 实现是更近参考）。
3. **FP8 attention 训练的根因级发现（Delta-Matching, arXiv 2609.37852）**：FlashAttention 反向用 saved output 算 row correction，量化后 delta 陈旧、破坏 softmax 梯度行和为零的不变量，误差逐步累积。1.67B/30B token 下 train CE：BF16 1.43、Delta-Matching 1.43、TE FP8 1.57、朴素 FP8 1.89；569M 规模几乎看不出来，约 19k step 后 gap 才显现。
4. **vllm-ascend：分页粒度必须对齐 packed page 宽度**（#17115）。GLM-5.3-Flash C8 的 512-token block 只打包 264 KiB，留 16 KiB 空洞，非连续 view 触发 CANN AutoContiguous 每层每 step 拷贝整个 ~2 GiB KV 池（占 87% device time，decode 塌到 ~5 tok/s）。改为 640 token block 后 KV 容量 2.27M → 4.04M token（+78%），decode 34.89 tok/s 与 C16 持平。
5. **torchtitan 定位出 DeepEP handle 泄漏（#4918）**：Full AC 的 early-stop recompute 在 combine 前停下，dispatch 的 routing layout handle 永不释放；8×H100 Qwen3-30B-A3B 下 peak active 从 64.3 GiB 涨到 77.2 GiB、14 次 allocator retry。这类"重计算提前退出导致通信库 handle 泄漏"在自研 dispatch/combine 上同构可复现。
6. **Megatron-LM 为 wide residual 加 selective replay（#6805/#6855）**：残差边界激活从 `K*X` 摊薄到 `K*X/M`；并明确"宽残差只属主干、MTP 头保持普通宽度 D"的契约，解除了 wide_residual 与 MTP 的互斥。

---

## 1. 模型发布与 Tech Report

**DeepGEMM-Ascend（首次开源）**｜DeepSeek｜2026-09-30（README changelog：「2026.09.30: Initial release ... with support for Ascend 950 devices」）｜https://github.com/deepseek-ai/DeepGEMM-Ascend

- **目标硬件**：Ascend 950 系列，性能数据标注 950DT；未提 910C/910B 支持。
- **精度**：FP4 / FP8 / BF16。
- **Dense GEMM（M=4096, N=7168, K=16384）**：BF16 431 TFLOPS（上限 432，99.8%）；FP8 861（865，99.5%）；FP4 1701（1730，98.3%）。
- **Grouped GEMM（MoE）**：m-grouped 818–862 TFLOPS（8-expert 配置 831）。
- **MegaMoE 融合算子**：dispatch + grouped GEMM + SwiGLU + combine 融合，384-expert 峰值 846.3 TFLOPS。
- **其他 kernel**：MQA Logits（FIX-pipe bound，scoring 99% 利用率，565–743 TFLOPS）；HC Prenorm GEMM 接近打满 HBM 写带宽（3463 GB/s）。
- **实现栈**：AscendC primitives + CANN 9.20（依赖 `bin/bisheng`、`bin/ld.lld`），JIT 走 DeepJIT（`DG_JIT_CACHE_DIR` 控缓存），mHC kernel 用 Tilelang。依赖 Python 3.10+、torch_npu、C++20 `<format>`。MIT license。
- **简析**：核心创新不在算法而在"第三方把昇腾算子效率做到接近硬件上限并公开数字"。相对 GPU 版 DeepGEMM 的变化是完全重写到 AscendC，接口保持 API 兼容。未披露：FP4 的具体格式（MXFP4 还是其他）、910C 支持、与原生 CANN/CATLASS 算子的对比、多卡与集群级数据、是否已用于 DeepSeek 自家训练、roadmap。
- **对我的意义**：MoE 训练的算子天花板有了外部参考值——若自研 grouped GEMM 与 MegaMoE 类融合算子达不到 800+ TFLOPS 量级，差距就是可量化的工程债。MegaMoE 的融合边界（dispatch+GEMM+SwiGLU+combine 一体）也可直接作为我们融合算子的设计目标。

**DeepEP-Ascend（首次开源）**｜DeepSeek｜仓库 created 2026-09-30T00:44:59Z（首个 commit 时间戳 09-29，公开可见在窗口内）｜https://github.com/deepseek-ai/DeepEP-Ascend

- **目标硬件与前提**：Ascend 950（A5），要求 rank 间具备 **UBMEM** 连接与 **UBC_CTP/URMA** channel，要求偶数 rank，部署于 supernode（netlayer 1 外部 Clos 网络）。
- **通信原语**：EP all-to-all（MoE dispatch/combine）；PPBuffer（相邻 pipeline rank 间 send/recv）；BucketBuffer（batched all-gather、registered storage、多 group/session）；远端内存访问经 **Engram** 接口。
- **精度**：dispatch 支持 BF16 / FP8（row-major scales），combine 为 BF16；支持 deferred epilogue 与 cached handle。
- **带宽实测（dispatch / combine，GB/s）**：EP8 373–375 / 345–347；EP16 348–352 / 338–341；EP32 335–340 / 320–324；EP64 323–327 / 294–298；EP128 313–320 / 272–278。README 称 EP≤32 时 dispatch 达物理 payload 带宽上限的 90–95%。
- **软件栈**：CANN 9.2.0（已验证）、AscendC、Bisheng、HCCL/HCOMM headers、Python 3.10+、torch_npu、C++20。固件需 Atlas 850E 的 Q3 商用 HDK，README 称约 2026 年 10 月中公开可用。
- **限制（对万卡最关键）**：**专家负载均衡（EPLB）kernel 未实现**；不支持 hybrid 通信、CPU-backed Engram、**graph capture**；bucket reduce-scatter / all-reduce 仍在开发；PP 与 Engram 为实验性。
- **简析**：EP 从 8 扩到 128，dispatch 带宽衰减约 15%、combine 约 20%，这个衰减曲线是超节点内 all-to-all 的实测标尺。未披露：license、与 GPU 版 DeepEP 的 API 兼容程度、910C 支持、低时延模式的绝对时延、EP256+ 与跨 supernode 数据。
- **对我的意义**：可以直接拿 EP8/EP32/EP128 三档带宽对标我们的 HCCL EP 实现，定位是算子问题还是链路问题。但 EPLB 缺失 + 不支持 graph capture，意味着现阶段还不能直接用于万卡训练主干；若要评估，先把这两项的补齐成本算清楚。

**Gemini 4 Argon**｜Google DeepMind｜2026-09-30（官方 X 帖时间戳 20:03:30Z；TechCrunch 23:43Z）｜https://deepmind.google/models/gemini/

- 单次输出上限从 64K 提升到 1M token，context window 1M；文本/图像/视频/语音输入，文本输出。
- 定价：发布促销 $2/$10 per 1M in/out，后续 $4/$20；cached input 95% 折扣。
- 第三方评测（Artificial Analysis）：Intelligence Index 53，与 GPT-6 Astra (max) 持平；每任务平均输出 62k token。DeepSWE v1.1 77.9%（GPT-6 Astra 74.1%、Claude Opus 5.5 74.2%）；Terminal Bench 4 为 57%，FrontierSWE 相对偏弱；AA-Omniscience 准确率 50%、幻觉率 15%。
- 可用性：先通过 Fairwind Program 面向网络安全伙伴与美国政府实体，后续扩展到付费 API 与 Google AI Ultra。
- **未披露**：参数量、MoE 结构、激活参数、层数/hidden、注意力类型、训练 token、TPU 代次与规模、精度格式。无 model card 技术细节。
- **对我的意义**：infra 层面几乎无可借鉴信息。唯一有参考价值的是"单次输出 1M token"这个量级——如果成为通用预期，RL rollout 与 serving 的 KV 生命周期管理压力会再上一个台阶。

> 其他厂商：窗口内 OpenAI、Anthropic、Meta、xAI、Mistral、Qwen、Moonshot、智谱、MiniMax、字节 Seed、腾讯、百度、阶跃、小米、美团、NVIDIA Nemotron、Microsoft Phi、AI2 OLMo 均未发布新的 LLM 权重或 tech report。10-01 为国庆假期，国内厂商静默与此一致。窗口外条目见运行备注。

## 2. 论文精选（训练 / 后训练 / 推理）

### 训练系统

**Mixture-of-Kittens (MoK): MoE Megakernel for NVL72s**｜Stanford + Cursor Research｜arXiv 2609.36070（v1 09-28，09-30 listing）｜https://arxiv.org/abs/2609.36070

- **出发点**：在 NVL72 这类 scale-up 域里，为 scale-out 网络设计的 MoE 系统在近一半测试配置下不如朴素 PyTorch+NCCL。
- **三个机制**：按算子选 push 或 pull（pull dispatch + push combine，信号开销从 8–18% 降到 1% 以下）；minibatch 粒度做成可调参数（最优值在 512–32K token，最好与最差粒度吞吐差 1.38–3.53×）；彻底消除 CPU-GPU 同步（GB300 的 Grace 主机执行常见 PyTorch op 比 DGX 慢 1.46–2.97×）。最终做成确定性 megakernel，融合 dispatch / shared+routed expert FFN / combine，支持 MXFP8。
- **数据**：Kimi K2.7、GLM-5.2、Qwen3.5-397B、DeepSeek-V4-Pro 的层形状，EP=64、每卡 2048 token。对最强基线（DeepEP / HybridEP+Megatron / NCCL）：MXFP8 前向 2.37×、反向 1.78×；BF16 前向 1.92×、反向 1.58×。Cursor 生产环境 512 卡跨多个 GB300 NVL72 机柜，端到端 tokens/s/GPU 比 DeepEP 版本高 1.41×。已开源。
- **局限**：只针对单跳 scale-up fabric；端到端只在 512 卡验证。
- **对我的意义**：昇腾超节点同样是大 scale-up 域。三条机制可逐一对照自查：host 侧同步开销（鲲鹏 host 同样可能偏慢）、dispatch/combine 的 push-pull 选择、以及 minibatch 粒度是否被当成可调参数。确定性 kernel 对 RL 训推一致性也有直接价值。

**Delta-Matching: Closing the Final Gap of Native 8-bit Training for LLMs**｜CMU + NVIDIA + MIT（Song Han）｜arXiv 2609.37852（v1 09-29，09-30 listing）｜https://arxiv.org/abs/2609.37852

- **根因**：FlashAttention 反向用 saved output 算 row correction（delta = rowsum(dO∘O)）。前向与反向操作数分别量化后 delta 陈旧，破坏 softmax 梯度行和为零的不变量，误差在训练中逐步累积。
- **做法**：改用未量化概率 × FP8 反向操作数的 FP32 乘积算 delta，恢复不变量；attention core 的 7 个 matmul 得以全部走 block-scaled FP8。
- **数据**：1.67B GDN/GQA hybrid、30B token，最终 train CE：BF16 1.43、Delta-Matching 1.43、cuDNN/TE FP8 1.57、朴素 FP8 1.89。5.29B 上朴素 FP8 CE 1.93（BF16 1.33），RULER-8K 从 53.5% 掉到 16.3%。569M 规模 gap 很小；stale-delta 的 loss gap 要到约 19k step（20B token）才超过 0.01。8K→64K 上下文扩展时，两个 cuDNN/TE FP8 种子有一个发散。
- **局限**：最大 5.29B / 30B token；只测 attention kernel 吞吐，无端到端与分布式数据；head dim 128 时比 cuDNN BF16 还慢。
- **对我的意义**：这是 FP8/MXFP8 全链路 attention 的必读数值陷阱清单。两条结论要写进我们的验证规范：小规模（<1B）短训练通过**不能**作为万卡长训的安全证据；反向 row correction 的计算方式必须单独审查。

**ID Balancing: Stable Training of Extremely Sparse MoE via PID-Based Load Control**｜Alibaba Qwen Team｜arXiv 2609.39137（v1 09-30，10-01 listing）｜https://arxiv.org/abs/2609.39137

- **框架**：把 DeepSeek 的 aux-loss-free 视为固定步长积分（I）控制器，把 Kimi K3 的 Quantile Balancing 视为广义比例（P）控制器，提出 Integral–Derivative 控制器：积分项随负载误差缩放，微分项只在不均衡恶化时启用。
- **数据**：768 专家、Top-10/5/3（稀疏度 1.3%/0.65%/0.39%）、18.9B 总参、120B token。Top-3 下 worst-case backbone MaxVio 比最佳基线降低 >50%、MinVio 降 12%；从 18.9B 扩到 69.9B（激活 1.03B→3.2B）时 MaxVio 基本不变，比 aux-loss 基线低约 89.6%；另做了 2.3× 峰值学习率压力测试。语言建模与下游指标持平。
- **局限**：只报负载指标，没有 MFU 或 EP 通信时间收益；规模最大约 70B。
- **对我的意义**：改动只在 router 的 bias 更新规则，适合在 Megatron/MindSpeed 路由器里直接 A/B。超稀疏 MoE 的负载不均直接决定 EP straggler 与 all-to-all 尾延迟，这是低成本的稳定性杠杆。

**HAPMoE: Heterogeneity-Aware Automatic Parallelism Planning for MoE Training**｜北大 + 无问芯穹 + 清华｜arXiv 2609.39350（v1 09-30，10-01 listing）｜https://arxiv.org/abs/2609.39350

- warm-up profiling + MoE-aware 代价模型，在 DP/PP/TP/CP/EP/TPE 六维空间做剪枝 DP 搜索，支持非均匀 PP/DP 层划分与逐 stage 重计算策略，输出可直接用于 Megatron-LM 启动脚本。
- **实验含昇腾**：H800、**Ascend 910B**、MI300X 三种卡的同构与异构组合，16–32 卡；Mixtral-S/L 与 LLaMA-2 7B/13B。异构 MoE 下几何平均吞吐为 Megatron-Infinigence 的 1.67×/1.78×，最高 3.2×；非均匀划分额外带来 4–78%，关掉后含 910B 的异构配置延迟膨胀 2.3–3.4×。搜索 <60s；4 节点 profile 预测 16 节点延迟误差约 10%。
- **局限**：最多 32 卡；不支持在线重配置；不考虑故障。
- **对我的意义**：直接覆盖 910B 与 GPU 混合场景。万卡本身参考有限，但"非均匀 PP 划分 + EP 组尽量放在同构子集群内"这个结论可用于新老昇腾代际混部的切分策略。

### RL 与后训练

**PACE: Pool-Aware Effective Staleness Control for Async RL**｜ByteDance + TAMU + UVA｜arXiv 2609.36830（v1 09-29，09-30 listing）｜https://arxiv.org/abs/2609.36830

- **分解**：token 平均 staleness = Generation Staleness（rollout 生成中跨越的策略版本数）+ Waiting Staleness（生成完成后在样本池排队）。用池占用率相对目标的超出比例、经 EMA 平滑作为拒绝率（依据 Little's law）。排序按"有效 staleness"：Waiting 全额计入，Generation 按 prefix-forgetting score（新策略重打前缀的 logprob 漂移）做百分位锐化加权（γ=4），避免只按"年龄"过滤时系统性丢掉长而难的样本。
- **数据**：Qwen3-8B / DAPO-Math，6 benchmark 平均：同步 41.07、非过滤异步 36.34、M2PO 38.82、PACE 41.29。多轮工具调用（Qwen3-4B，ReTool 风格）：异步 29.24、同步 45.06、PACE 47.91。每步耗时：异步 26.19s、PACE 28.22s、同步 97.01s。另在 Qwen3-30B-A3B 与 REINFORCE++ 上验证。
- **局限**：任务集中在数学 + 一个工具环境；需记录 token 级行为 logprob 与版本号，还要做前缀重算分；比较的是固定步数内最佳 checkpoint。
- **对我的意义**：可直接移植成 verl 异步模式下样本池的准入控制，不改 loss。对 agentic 长尾 rollout 收益最大——这正是我们异步 RL 里最难调的一环。

**CadenceRL: Reshaping Rollout Workloads for Async RL on Heterogeneous Accelerators**｜北邮 + 阶跃星辰｜arXiv 2609.36899（v1 09-29，09-30 listing）｜https://arxiv.org/abs/2609.36899

- **洞察**：decode 受内存带宽限制，大 batch 换吞吐、小 batch 换单条轨迹完成时间，因此让不同 worker 分别专做吞吐或专做尾部。三个机制：Pacing（换出长上下文轨迹、换入短轨迹，让 worker 留在高吞吐区）；Concentration（staleness 预算耗尽时把长尾集中到高带宽卡）；Late-bound KV preparation（先把 KV 前缀快照到 host 或 relay 节点走 RDMA，之后再决定目标 worker，避免目标端重新 prefill）。
- **实验池含昇腾**：Huawei **Ascend A910X** + 天数智芯 BI-V150 + H800，每种卡一个集群；集群内 8×200G RoCE，集群间仅 20 Gbps。训练端固定 H800。基于 vLLM 与 Steptron，Qwen3-8B/14B/32B，PPO，global batch 8192。
- **数据**：异构池上 decode 吞吐最高 +48%，P95 轨迹延迟最高 -64%（对比 Laminar、AReaL）；同构 H800 池上退化为明确的吞吐-延迟权衡。reward 曲线与基线重合。
- **局限**：跨集群跨卡型迁移退化成 token 重传 + 重新 prefill；32B 上 KV 迁移成本上升；训练端刻意留了余量、不是瓶颈。
- **对我的意义**：少有的在真实昇腾 + GPU 混合 rollout 池上做的实验。"高带宽卡跑尾部、性价比卡跑大 batch"可直接用于我们 RL 集群新老 NPU 混部的 rollout 调度。

**EasyPPO: Stabilizing the Critic Is Key**｜UC Berkeley + Princeton｜arXiv 2609.36802（v1 09-29，09-30 listing）｜https://arxiv.org/abs/2609.36802

- **两个崩溃来源都在 critic**：(a) actor 与 critic 都过滤 overlong 样本时，策略目标变成"在完成条件下的 reward"，截断率不降反升——改为只对 actor 过滤，critic 用全部样本；(b) 不同 prompt 回报噪声差异大，高方差 prompt 主导 critic 梯度——按每 prompt 回报标准差的倒数加权 critic MSE，并把 critic mini-batch 适度减小（每个 rollout batch 切 4 份）。
- **数据**：verl 实现，Qwen3.5-9B，rollout batch 512/1024。最佳验证分 FrontierCS 14.82（PPO 12.90）、AIME24 65.52（64.06）、Search-R1 43.22（39.48）。所有基线至少在一个任务上崩溃过，只有 EasyPPO 三任务、三种子均未崩。
- **局限**：只在 9B、200–300 步、严格 on-policy 同步设置下验证。
- **对我的意义**：三处改动都是几行配置，若回到带 critic 的 PPO 路线（见上期 PACT）可直接叠加。

### 推理

**AVSG: Accelerated Vectorized Sparse Gather for KV Offload in Sparse-Attention Serving**｜华为 2012 实验室理论部｜arXiv 2609.37538（v1 09-29，09-30 listing）｜https://arxiv.org/abs/2609.37538

- **机制**：针对 DeepSeek Sparse Attention（DSA）。完整 KV offload 到 host DRAM，HBM 保留按 slot 组织的缓冲区；向量化 open-addressing 哈希匹配 top-K index；只传 miss 的 KV 并把 H2D 在多个 AICore 之间均分；用 lifetime 衰减 + 命中刷新实现近似 LRU 驻留；MTP 多 token 共享同一缓冲区，推测 token 的条目在每步末失效。
- **数据**：单 NPU 上哈希匹配比标量双指针快 2.80×，有效 H2D 带宽是按请求分配的 34.44×，8K 缓冲区真实请求命中率 94.83%，MTP-3 冷启动复用率 61.87%。端到端：**12 节点 910B，5P1D（prefill TP16/EP16，decode DP16/EP16）**，GLM-5.3-w4a8c16 + 3-token 推测解码，2000 条生产 agent 请求、平均 prompt 48K：TPOT 91ms → 55ms（-39%），output TPS 451 → 571（1.27×），端到端时延 120s → 76s；代价是多用约 4GB HBM、TTFT 升高。代码在 hicann/cann-recipes-infer。
- **局限**：只适用于 DSA 这类 exact top-K 选择；只测冷启动、单一负载。
- **对我的意义**：这是直接基于 CANN 的 DSA 长上下文 offload 算子，部署 DeepSeek-V4 / GLM-5.x 可直接复用；"跨步 KV 复用 + 推测 token 条目失效"的设计同样适用于 RL rollout 的长上下文 decode。

### 互联与编译

**Purlin: Separating Orchestration from the Datapath of Collectives**｜Stanford + NVIDIA（Kozyrakis）｜arXiv 2609.36954（v1 09-29，09-30 listing）｜https://arxiv.org/abs/2609.36954

- **三层解耦**：语义层（packed/scattered/transposed 三种 layout + copy 或 reduce 声明一个集合通信）；编排层（统一的 SNAC 协议 Stage-Notify-And-Consume，从语义自动推导调度）；数据通路层（硬件相关的 Atom copy/reduce，按代际特化）。新硬件机制只需改 Atom 层，用模板元编程在编译期派发。
- **数据**：A100/H200/B200 上 7 种集合通信，延迟最高快 5.14×、带宽最高 4.50×。B200 上 `Atom::copy` 用 **16 个 SM** 就到约 760 GB/s，NCCL device API 需要 64 个 SM。SGLang 离线吞吐平均 1.13×/最高 1.37×；在线（DeepSeek-V4-Flash FP8，8×H200，Mooncake trace）交互性平均 1.26×/最高 2.85×，最大收益出现在过载时。AIME26 精度持平。
- **局限**：只覆盖 scale-up 域（NVLink），只做推理；高并发时 MSCCL++ 偶有反超。
- **对我的意义**："用更少的核跑满链路带宽"是通算融合的关键指标（16 SM vs 64 SM）。三层解耦的架构可对照 HCCL 在超节点内的演进路线，尤其是把硬件特化收敛到 Atom 层这一点。

**LampAttention: Look-Ahead Mixed-Precision FlashAttention for Dedicated Accelerators**｜维也纳大学 + 华为海森堡研究中心 + 华为｜arXiv 2609.39361（v1 09-30，10-01 listing）｜https://arxiv.org/abs/2609.39361

- attention 内部 QK 累加与 exp 默认 8-bit（e4m3 累加，ue5m3 + LUT 单周期算 exp），再用 running max 加解析误差界两阶段识别"危险"子块，用 16-bit（e4m11/ue5m11）重算。需要 QK-Norm 保证 logit 不超过 448。论文同时给出一份假想专用加速器规格。
- **数据**：纯软件模拟、无实测时延。Qwen3-32B 的 C4 perplexity：纯 8-bit 28.18、LAMP 23.74、32-bit 23.80；46–70% 的 tile 完全不需要 16-bit 重算，需重算 ≥2 个子块的 tile 占 16–33%。
- **对我的意义**：华为参与的"attention 内部精度"软硬协同方向，可能预示后续昇腾 Cube/Vector 单元在 8-bit 累加与 LUT exp 上的设计，值得作为前瞻信号跟踪。

> 其他窗口内相关、未精读：
> - 2609.36959 Cobalt（浙大/北大/上交）：利用专家共激活关系优化分布式 MoE 训练。
> - 2609.36662 AutoLoCo（清华）：跨数据中心预训练的自适应同步（LocalSGD/DiLoCo 一类）。
> - 2609.39223 QATFactory（Together AI 等）：与部署格式对齐的 QAT + 蒸馏框架。
> - 2609.37693「FP64 Is All You Want, INT8 Is All You Need, FP4/6/8 Is All You Have」：低精度格式的分析/立场文章。
> - 2609.36654 Replay the Curvature：NVFP4 推理量化。
> - 2609.39138 MoSE（南科大等）：训推共用 fabric 时 PD 分离 KV 传输的 mode-switching 扩展网络。
> - 2609.37626 SPLASH：attention 并行布局的无缝切换。
> - 2609.38981 Vosti：确定性 LLM 推理的规约、实现与验证（与 RL 训推一致性相关）。
> - 2609.37828（浪潮）：GPU 服务器拓扑、并行方式与拥塞控制对 MoE 推理的联合影响（仿真）。
> - 2609.37916 RLX：Rust 实现的多后端张量编译器与分布式运行时。
> - 2609.36601 SAKI / 2609.38360 SCOUT：on-policy 蒸馏的 teacher 适配。
> - 2609.39247、2609.37825：RLVR 中 critic 回潮的两篇。
> - 2609.39114 / 2609.39767：Muon 行归一化分析；预训练中局部损失景观几何的演化。
> - 推测解码与 KV：2609.37532 DScale、2609.37029 LongSpark、2609.39972 UBTree、2609.38121 WUSH-KV、2609.39131（High Bandwidth Flash 特性分析）。

## 3. AI Infra 技术文章

**From Upstream Changes to Downstream Confidence: Torch Spyre 与 PyTorch CRCR 的集成**｜PyTorch Blog（IBM/Red Hat）｜2026-09-30｜https://pytorch.org/blog/from-upstream-changes-to-downstream-confidence-inside-torch-spyres-integration-with-pytorch-crcr/

- CRCR（Cross-Repository CI Relay）通过 `repository_dispatch` 把 upstream PR 推给树外 PrivateUse1 后端 CI，结果汇总到 hud.pytorch.org/crcr，接入分 L1–L4 四级。
- Torch Spyre 已达 L2：用 agent 做测试选择（2.13 → 2.14 只需评估约 4000 个变更测试，不用重跑数万个）；用声明式 YAML 适配测试，分 mandatory_success / xfail / xfail_strict / skip 四档、可按 dtype 粒度排除，不需 fork 测试代码；每个 job 控制在约 30 分钟内，wheel 只构建一次多处复用。
- **对我的意义**：torch_npu 正是 PrivateUse1 后端，而本窗口的 torch_npu 提交里就有两条是"跟随 2.14 上游特性"。接入 CRCR + 借鉴这套 YAML 测试分档，可以在 PyTorch 版本升级时提前拦截上游回归，而不是等到合入后再排查。

**Automating Performance Bottleneck Identification with the TraceLens Agent**｜AMD ROCm Blog｜2026-09-30｜https://rocm.blogs.amd.com/software-tools-optimization/tracelens-analysis-agent/README.html

- 读 PyTorch / JAX / RocProf trace，构建 Python op → kernel 调用树，再用 roofline 分析、NcclAnalyser 与 TraceDiff 做诊断；compute 与 system 两层子 agent 并行分析，输出可交给 Hyperloom 自动优化。
- 示例：18 个 BF16 GEMM 占 45.7s（约为计算时间的 80.6%），效率为 708 TFLOPS 峰值的 68.7%–86%。
- **对我的意义**："trace 进、带证据的优化报告出"这个形态可以照搬到 msprof 与万卡 profiling 上。注意它的价值在于把 roofline 判据和通信分析固化成自动化规则，而非 agent 本身。

**Expanding AI Storage Access with NVIDIA cuObject and the SCADA Server SDK**｜NVIDIA Technical Blog｜2026-09-30 19:13 UTC｜https://developer.nvidia.com/blog/expanding-ai-storage-access-with-nvidia-cuobject-and-the-nvidia-scada-server-sdk/

- cuObject client/server GA（cuObject Server 2.0.0，配合 CUDA 13.4）：控制面走 HTTPS/TCP，数据面走 RDMA，绕过存储服务器 CPU。新的 SCADA Server SDK 让存储厂商响应 GPU 发起的 IO 请求，IBM Storage Scale 已有原型。文中未给带宽与延迟数字。
- **对我的意义**：GPU 直连对象存储正在标准化，可作为万卡 checkpoint 与数据加载走 RDMA 直通的参考方向；昇腾侧对应能力需要对齐，否则 checkpoint IO 会成为相对短板。

**What's new in AI infrastructure and orchestration in September**｜Google Cloud Blog｜2026-09-30｜https://cloud.google.com/blog/topics/ai-infrastructure/whats-new-in-ai-infrastructure-this-month

- 月度汇总（部分内容此前已单独发布）。GKE Agent Substrate：agent 负载密度提升 10 倍，resume <500ms，每秒 500 多次 suspend/resume。GKE Pod Snapshots（含 GPU 显存状态）：推理启动时间减少 89%。Dataplane V2 扩展到 15,000 节点。Agent Sandbox 的 RL 优化功能 GA，附 RL 编排 SDK。
- **对我的意义**：RL 环境沙箱的规模化与快照恢复，是 agentic RL rollout 基础设施的参考方向——我们当前 rollout 环境的启停开销还没有被当作一等指标来优化。

> 次要：AMD AIM 统一部署体验（按硬件自动选 TP 与精度 profile，09-30）；NVIDIA 两篇应用向博客（HSTU 生成式推荐用 Dynamo-Triton + FlexKV，batch 8 全命中时 3 层 0.423 ms/请求、8 层 0.678 ms/请求，比无缓存 AOTI 快 4.47× 和 5.93×；NeMo Relay 的 agent harness 行为追踪，108 次运行，Qwen Coder 30B 成功率 70%→81% 但 LLM 调用 3.8→4.9 次、耗时 27s→42s）。

## 4. 硬件与互联

本节窗口内唯一的硬核数据来自 DeepSeek 开源的两个昇腾 950 组件（见第 1 节）：CANN 9.2.0 为 950DT 的验证版本，DeepEP-Ascend 要求 UBMEM/URMA 与 supernode 拓扑，固件需 Atlas 850E 的 Q3 商用 HDK（README 称约 10 月中公开）。这是昇腾 950 超节点（SuperPoD Flex / UBL128）首次有第三方公开的软件侧实测。

**SemiAnalysis：GPT-6.1 Sol Ultrafast 跑在 NVIDIA GPU 上而非 Cerebras**｜SemiAnalysis｜2026-09-30｜https://x.com/SemiAnalysis_/status/2105055045959995719

- Ultrafast 约 300 tok/s，比标准部署快约 8 倍；SemiAnalysis 判断是在 NVIDIA GPU 上以极低 batch 运行。Cerebras 股价当日跌约 6–7%。
- **对我的意义**：低 batch 极速推理靠"通用 GPU + 牺牲吞吐"而非晶圆级专用芯片。评估昇腾超节点要不要做低延迟推理档位时，这是一个有用的参照：关键变量是单卡带宽与调度粒度，不是新硬件。

> 窗口内没有新芯片发布，也没有 NCCL / RCCL / CUDA / ROCm / CANN 的主版本 release（NCCL 仓库窗口内 0 commit）。Intel Gaudi、AWS Trainium、寒武纪、海光、Groq、Cerebras（技术面）、UALink、UEC 均无新发布。

## 5. 开源框架与社区

### 正式 Release

**pytorch/pytorch v2.14.1**｜2026-09-30 19:14 UTC｜https://github.com/pytorch/pytorch/releases/tag/v2.14.1

- 补丁版本。对训练侧最关键一条：**CUDA 13.2 Linux 二进制升级到 CUDA 13.2.2（#196351）**，该 NVIDIA 更新修了两个会产出错误结果的问题——cuBLAS `cublasLtMatmul()` 在 **NVFP4 matmul 时可能忽略 tensor-wide scaling**（CUDA 13.2 Update 1 引入），以及编译器线程重汇聚失败导致嵌套分支 kernel 中寄存器残留/损坏（自 CUDA 12.8 起存在）。其余为 MPS 上 linalg 的修复。
- **对我的意义**：有 NVFP4 实验线的必须升级。对 Ascend 无直接影响，但 torch_npu master 已在跟随 2.14。

> 窗口内无 Release：Megatron-LM（core_v0.19.2）、torchtitan（v0.3.0）、DeepSpeed（v0.19.7）、verl（v0.9.1）、OpenRLHF（v0.11.2，窗口内 0 commit）、slime（v0.3.2）、vLLM（v0.30.0）、SGLang（v0.5.20）、TE（v2.19）、NCCL（v2.32.3-1）。**差几小时落在窗口外**：vllm-ascend v0.27.1rc1、TensorRT-LLM v1.3.0rc29、trl v1.14.1（均 09-29）。

### 重要合入

**vllm-project/vllm-ascend**（窗口内 47 个 commit，择要）

- **#17115 为 GLM-5.3-Flash 启用 sparse SFA C8（INT8 packed KV cache）**｜https://github.com/vllm-project/vllm-ascend/commit/88aebfb2dad70efb5721c1691a91b12f30299b25
  - 第一部分是把 C8 路径打通到 kpool-indexer 模型（原 `use_sparse` gate 只覆盖 lightning-indexer 模型，导致该模型永远建不出 C8 KV spec）。
  - **真正值得记的是分页工程细节**：C8 packed page 必须覆盖决定共享池 slot 大小的 FP32 KDA state page（~280 KiB）。512-token block 只打包 264 KiB，留下每 block 16 KiB 空洞，非连续 view 随即让 CANN 算子的 AutoContiguous **每层每 decode step 拷贝整个 ~2 GiB KV 池——实测占 87% device time，单流 decode 塌到约 5 tok/s**。因此 block 由 packed page 宽度反推：`ceil(280KiB/528B)=544`，再向上对齐到 hybrid-split 对齐值 = **640 token**（640×528=330 KiB 完全覆盖 slot），kernel block 经 hybrid split 仍为 128。
  - 实测（A3、TP16、gpu-memory-utilization 0.85、含 #16706）：**KV cache 容量 2.27M（C16）→ 4.04M token（+78%）**；单流 decode（1k-in/1k-out ×32，ais_bench）34.89 tok/s，与健康的 C16 baseline（约 34.4）持平；gsm8k（128 并发）97.19%。
  - **部署警告**：cherry-pick #16706 后必须从源码重建自定义 CANN 算子包（`vllm_ascend/_cann_ops_custom`），用旧算子二进制跑新 host 代码会**静默算错**（修前 kernel 读错 dequant scale 位置）。
  - **对我的意义**：这条"分页粒度不对齐 packed page 宽度 → AutoContiguous 全池拷贝"的陷阱，任何自定义 KV 布局都该照着自查，而且它的表现形式（87% device time 凭空消失）极易被误判为算子慢。
- **#14340 混合模型（Qwen3.5）与 GQA 支持非连续 KV cache**｜https://github.com/vllm-project/vllm-ascend/commit/5fa57c55bb37861d3da0274fbf7a28d15d745b87
  - 支持 Attention / Mamba 独立 page size，混合布局改用基于 stride 的 cache view 而不物化带 padding 的区域；新增 `gather_ssm_states` / `scatter_ssm_states_` Triton kernel。
  - **明确的布局契约**：只有第一个 state stride 可含物理页 padding；内层 state 维度必须连续；state 索引选物理行、可乱序或不连续；`has_initial_state=False` 产生零初始态而不读 cache。
  - 验证：gather/scatter 18 例、MRV2 cache 布局 49 例通过；端到端 Qwen3.5-35B-A3B W8A8（V1 engine、ModelRunner V2、FULL_DECODE_ONLY）与 Qwen2.5-7B GQA 的短序列/长上下文/32 路并发全部零失败。
  - **对我的意义**：混合注意力（Attention + Mamba/KDA）在昇腾上的 KV 显存效率基础设施；那条布局契约是写自定义 state 算子时的硬约束。
- **#17270 Kimi-K3 的 O-projection 与 MM ReduceScatter 融合**：A5 纯 prefill batch 上改用 `torch_npu.npu_quant_mm_reduce_scatter`，把投影与 TP 归约合一。早期 A5 profiling（TP8/BF16，M=16256 K=1536 N=7168）：KDA O-projection pattern 1.625 → 1.489 ms（-8.36%），含 TensorMove 的 MLA pattern 2.320 → 1.558 ms（-32.82%）；目标 pattern 时间每 chunk 减少 27.64 ms（-16.47%）。作者明确声明这些是同 shape 采样、**非端到端或 TTFT 数据**。
- **#17567 A5 上融合 Kimi K3 latent RMSNorm 与 MXFP8 量化**：后续 up-projection 用 W8A8 MXFP8 时改走 `npu_rms_norm_dynamic_mx_quant`，把 `(quantized, scale)` 元组直接喂给既有 prequantized linear 输入路径。A5 单卡 event interval 中位数 **37.3–38.1 µs（融合）vs 67.6–69.8 µs（分离 norm + quant）**，近 2× 的 kernel 级收益。非整模延迟。
- **#17364 grouped matmul SiTU 量化融合**：引入 Ascend 950 原生 `GroupedMatmul(MXFP8 × MXFP4) + SiTU + dynamic MX quant` 融合算子，经 `DeviceOperator` 接入 W4A8 MXFP MoE 路径。其中保留了 **RL 权重生命周期逻辑：SiTU W13 为融合 kernel 保留原生 FP4 dtype 元数据，并在 RL 权重重载前恢复为 `uint8`**。仅用于 group size 32、group-list type 0/1、`linear_beta` 为正，其余回退拆分路径。注意作者声明本地未跑 CI / CANN 构建 / NPU E2E。
- **#16706 A2/A3 支持 INT8 C8 RoPE0，A5 支持 INT8 与 FP8_E4M3FN**：C8 `KvQuantSparseFlashAttention` 支持 `rope_head_dim=0`，调用方可提供 512 元素 query 与 528 字节 packed KV 行（512 个单字节 NoPE 值 + 偏移 512 处 4 个 FP32 scale），不再需要外部 RoPE payload。契约：`key/value_quant_mode=2`、`quant_scale_repo_mode=1`、`tile_size=128`；A5 用 `sparse_block_size=1`、`return_softmax_lse=False`；HiFloat8 仍限 RoPE64。**PR 明确声明无性能提升主张**。
- **#17542 GLM-5.3-Flash 的 Triton KPool 与 gated norm 优化**：把 KPool score sanitization 与 index expansion 融进最终 SFA buffer，在 paged read 上保持 FULL graph。W8A8、TP4/DP2/EP8、3500 input/128 output 下 **TPOT 在 C1 改善 7.81%、C8 改善 7.08%**；GSM8K 97.04%；13 个 indexer shape 与 mainline 逐位一致。测量早于 FULL-graph dispatch 修复。
- 其余：#17565 避免整池 recurrent-state 拷贝（1M token prefill 的 OOM 来源，改用 stride-aware Triton gather）；#17614 KDA prefill state-copy 计划启动时预编译，serving 期间不再走 Triton JIT；#15182 MLA preprocess 迁到 ACLNN 两段式接口以支持 NPU Graph 捕获（原同步 `aclrtMemcpy` host tiling 拷贝在 capture 下报 107030）；#16392 Ascend950 支持 `store_kv_block` 自定义算子；#17427 Engram 多节点 DP 支持 node-local head sharding。CI 已升到 torch=26.2.0。

**Ascend/pytorch（torch_npu，窗口内 18 个 commit）**

- **NPU Static Triton Launcher（MR !45989）**｜https://github.com/Ascend/pytorch/commit/d2000432f8a77f89b25550cacdea1a43b6892b79
  - `NPUStaticArtifactAdapter` 从 Triton-Ascend 编译结果提取 NPU Bin、kernel signature、constexpr、参数 ABI 与 FFTS/SIMT 启动元数据，生成可序列化的 `NPUStaticallyLaunchedTritonKernel`；新增 C++ Static Launcher Runtime，用 `aclrtBinaryLoadFromData` / `aclrtBinaryGetFunction` / `aclrtLaunchKernelWithHostArgs` 完成加载、取函数、参数打包与下发，并管理 autotune loser 释放与 winner 持有。
  - 接入 NPU Autotuner、TritonBundler 与 FX Graph Cache：**缓存命中后直接恢复 NPU Bin 与静态 kernel，省掉重新生成/编译/加载每 kernel 的动态 C++ launcher**。
  - 支持动态 shape、非默认 stream、NPU Graph capture/replay；C++ Wrapper、Tensor Descriptor、Workspace、Device Print 等不支持场景**默认安全回退**普通 launcher，严格模式下抛具体原因。配置：`use_static_triton_launcher`（env `TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER`，OSS **默认 True**）、`strict_static_triton_launcher`、`static_launch_user_defined_triton_kernels`、`keep_static_cubin_raw`。
  - **对我的意义**：直接降低 NPU 上 `torch.compile` 的冷启与每 kernel launcher 生成成本；默认开启，升级时需知其回退条件。
- **2.14 版本特性跟随（MR !46747）**：(a) 上游 #188548 为 `PyProcessGroup` 补 `all_gather_single` / `reduce_scatter_single` 的 C++ override，torch_npu 结论为**零适配**（HCCL ProcessGroup 是 C++ 原生实现、不走 trampoline），并新增版本门控回归测试固化该结论；(b) object collectives 新增 `weights_only` 参数，torch_npu 同步适配 `_gather_object`，**torch < 2.14 下显式传 `weights_only=True` 直接 raise，绝不静默降级到不安全的 pickle 路径**；同时补上自 torch 2.6 起就有、PTA 长期未同步的 `group_dst`（`dst` 为全局视角、`group_dst` 为 group 内视角，互斥；`dst` 默认值由 0 改为 None）。
  - **对我的意义**：`gather_object` 的 `dst` 默认值变更与新增 `group_dst` 属**接口语义变更**，万卡脚本里用到 object collectives 的地方需复核。
- **停止声明 HCCL tensor-alloc 支持（MR !47428）**：`ProcessGroupHCCL::supportsTensorAlloc()` 原先返回 `IsGteCANNVersion("9.2.0-beta")`——**拿 CANN 版本冒充 c10d 能力声明**，而 `allocateTensor()` 在 torch_npu 中从未实现，于是信任该标志的调用方会拿到 `RuntimeError: Backend hccl does not support allocateTensor`。触发条件是运行时 **CANN ≥ 9.2.0-beta**，CANN 9.1 环境下该标志为 false 故此前未暴露。修法是删除 override、把 CANN 版本闸门移进其唯一真正使用者 `getMemAllocator()`。
  - **对我的意义**：正在做 CANN 9.2 迁移的集群值得提前知道这条。
- **DVM 升级到 r2.11（MR !47318）**：新增 DVM cat 融合、完整 View 融合、VF 融合；新增分解启用/禁用列表与矩阵乘融合否决回调。新配置（须在加载 DVM backend 前设置）：`enable_decomp_list` / `disable_decomp_list`（禁用优先）；`matmul_fusion_rule = None`，可设 `(n, m, k)` 回调，**仅显式返回 `False` 时否决 MM/BMM/ADDMM/BADDBMM 融合**。注意移植分支尚未构建 wheel 或跑 NPU 测试，CI 待确认。
  - **对我的意义**：`matmul_fusion_rule` 这种"按 shape 否决融合"的回调，对万卡调优中个别形状融合反而变慢的场景非常实用。

**volcengine/verl**

- **#8010 bypass-mode REINFORCE 的 score centering**｜https://github.com/verl-project/verl/commit/8718ca30a3f002f93b7c4fd99b9b2506718681bc
  - **机制**：train-infer mismatch 下策略梯度带漂移项 `E_q[R] * E_q[∇log p]`，会把 trainer 蒸馏向（量化过的/过期的）sampler。score centering 从每个 token 的 score 里减去 sampler 下的期望 score——**加性、无超参、on-policy 时恒为 no-op，可与 token-level TIS / IcePop 组合**。工程上每个生成 token 存 sampler 的 top-k head（k=128，k=32 也能对上），尾部用 trainer 尾部按 sampler 尾部质量重标定，trainer 只需评估 k 个 head log-prob。
  - **复现数据**（Qwen3-0.6B、Countdown、2×RTX PRO 6000、fp8 W/A + fp8 KV 的 vLLM sampler vs bf16 FSDP trainer、REINFORCE、SGD lr 1e-2、无裁剪、64 prompt×8、400 步）：`bypass_pg` 训练准确率 **0.00**（第 72 步崩塌，`ppo_kl` max 1.13）；`bypass_pg_token_tis` 0.28；**`bypass_pg_sc` 0.67**（held-out 0.63/0.56，`ppo_kl` max 0.08，稳定）；`bypass_pg_token_tis_sc` 0.65。代价：该配置下每步 25s（sc）/21s（tis_sc）vs 14–17s（pg）；切到 flat logprobs 后降到 14–16s。
  - **新配置**：`rollout.topk_log_probs: 128`、`rollout_correction.score_centering: true`（driver 与 actor 两侧必须一致）；指标 `actor/sc_correction`、`actor/sc_sampler_head_mass`。
  - **作用域限制（关键）**：仅 vLLM rollout + FSDP actor + v1（TransferQueue）trainer，`top_p=1`、`top_k=-1`、`logprobs_mode=processed_logprobs`，**不支持 fused kernels**；SGLang / TRT-LLM、**Megatron / veomni / torchtitan、legacy trainer 均不在范围**。核心算子在 `verl/trainer/ppo/score_centering.py`（分块、vocab-parallel-aware autograd function，backward 重算，**移植自 slime**）。前序：slime 已于 09-23 合入同一方法（#2405/#2406，Megatron + SGLang）。
  - **对我的意义**：这是当前 RL infra 最值得跟的一条——**训推不一致（尤其 fp8/量化 rollout）下的无超参修正**，对"昇腾上 rollout 与 trainer 精度/算子不同源"几乎是对症药。但它不支持 Megatron，MindSpeed-RL / verl+Megatron 要用需自行移植，**slime 的 Megatron+SGLang 实现是更近的参考**。
- **#8064 fp8 weight sync 丢弃 tied-embedding 别名**：`tie_word_embeddings=True` 的模型 state dict 同时带 `model.embed_tokens.weight` 与别名 `lm_head.weight`，可能落在不同 bucket；vLLM 0.29 按每次 `load_weights` 调用检查被跳过的别名，于是只带别名的 bucket 校验失败、**首次 weight sync 直接中止**。非量化分支早已用 `drop_tied_alias_updates` 过滤（#7973），fp8 分支漏了。Qwen3-0.6B + `rollout.quantization=fp8` + vLLM 0.29.0 复现，修后 3 步 GRPO 跑通。#8010 的 fp8 实验依赖本 PR。
- **#8065 VeOmni API 升级 + GPU CI 迁到 uv**：VeOmni engine 改为直接从 `AcceleratorConfig` 初始化 parallel state；首步数值对齐（单卡 FSDP baseline loss 2.7822391987 vs 八卡 VeOmni FSDP=4/Ulysses=2 同值）。**与昇腾直接相关**：Ascend workflow 的安装统一到 Transformers 5.16.1，NPU 路径**保留环境内既有的 Ascend 环境**（共享 lock 解析的是 CUDA 依赖），launcher 里保留 NPU 自动探测——在昇腾上跑 verl 的人需知 NPU 是"例外分支"，依赖解析走 CUDA、NPU 靠 ambient 环境兜。

**pytorch/torchtitan**

- **#4918 释放被 FullAC early-stopped recompute 重放的 dispatch handle**｜https://github.com/pytorch/torchtitan/pull/4918
  - DeepEP 每次 dispatch 把 routing layout handle 存进全局 dict（键为 CPU int tensor `handle_id`），只有 combine 会 pop。Full AC 的 early-stop recompute（#4836 起默认开）在 backward 重放 dispatch 后**在 combine 之前停下**，handle 永不移除——每 MoE 层每 step 泄漏一个（该 shape 下约 2.8 MB/个，48 层 × 100 step ≈ 13 GiB）。
  - 修法：给每个 `handle_id` 挂 `weakref.finalize`，并给 dispatch kernel 加 `@torch.compiler.disable`（Dynamo 对 `_registry` 大小建 guard 而调用本身会改变它）。
  - 实测（8×H100、Qwen3-30B-A3B、dp_shard=4 tp=2 ep=2、DeepEP v2、FullAC、32k tokens/DP rank）：main 的 peak active 从 step2 的 64.3 GiB 涨到 step100 的 **77.2 GiB、14 次 allocator retry**；本 PR 全程平稳在约 64 GiB、**0 次 retry**，median tps 5252/5294 → 5380。Selective AC 与 HybridEP 不受影响。
  - **对我的意义**：MoE + EP + 全量重计算是万卡 MoE 标配组合，这类"重计算提前退出导致通信库 handle 泄漏"在自研 dispatch/combine（MegaMoE、HybridEP 类）上同构可复现，排查思路可直接搬。
- **#4927 PP=1 的图内梯度累积与 FSDP overlap**：调度变成 `FWD_BWD_WITH_UNSHARD → FWD_BWD × (n−2) → FWD_BWD_WITH_REDUCE_GRAD`，FSDP 集合通信的拆分放在 bucketing 之前。GA 机制：第一个 microbatch 把 unsharded 梯度作为输出产出成为累加器，中间 microbatch 用 `add_` 或 wgrad-fused `add_` 就地累加，最后一个在 reduce_grads 前完成累加再做 overlapped reduce；全部用 graph pass 实现以避免重 trace。实测 **GB300×3、EP4 FSDP4、DSv3 16B GA：MFU 29 → 31**，但该数字来自"UNSHARD/REDUCE_GRAD 分离 + 全部 bucketed"而非 overlap 路径；作者保留两条路径，预期 FSDP 跨出 NVLink 域后 overlap 才占优。
  - **对我的意义**：给出了"域内 bucketing 优先、跨域 overlap 优先"的实测分界，这正是昇腾超节点内外 FSDP 策略选择的核心问题。
- **#4660 CUDA graph 分别捕获优化器更新**：forward-backward 与 optimizer update 放在两个独立 graph、共享 graph 持有的梯度；**梯度裁剪、finite 检查与 fused Adam update 都被 capture，checkpoint wait 与 LR scheduling 保持 eager**。引入 `Optimization` 抽象（类比 Megatron-Core 的 `MegatronOptimizer`）。无 benchmark。
  - **对我的意义**：ACL Graph / NPU Graph 捕获优化器步的工程形态对标，那条切分线可直接借用。
- **#4976 RL generator 的 token-in/token-out 采样**：vLLM 默认对每个生成 token 做 detokenize 并为每 token 每请求构造 `{token_id: Logprob}` dict，而 generator 从不读文本（stop 条件是 token id）。改为 `SamplingParams(detokenize=False, flat_logprobs=True)`。去掉约 75% 的 vLLM 每步 CPU 工作和几乎全部 GC 暂停，但因 engine step 比这部分 CPU 工作长 4–13 倍，H100 上端到端只 0–2%（4B bs128/512in/2048out 4.67k → 4.72k tok/s；GB300 Qwen3.5-4B bs64/1024in/512out 9.88k → 10.14k）。分解：4B bs256 每步 CPU 输出处理 4.41 → 1.15 ms，每 batch GC 暂停 108 → 5 ms；**2048out 场景 GC 从 799 ms → 3 ms**。
  - **对我的意义**：长输出（≥2k token）场景 GC 是真瓶颈，这是同进程 colocate rollout 可直接复用的一刀。
- **#4986 inter-generator router 默认改为 sticky session**：原默认 least-loaded 导致同一 multi-turn rollout 各轮落到不同 generator、吃不到前缀 KV——**Qwen3.5-27B agentic run、12 generators 实测前缀缓存命中率仅约 35%，而 KV 占用只有 12–17%**。
  - **对我的意义**：多轮 agentic RL 下 router 策略直接决定 prefix cache 命中率，是 rollout 吞吐的一阶因素，且几乎零成本。
- **Breaking**：#4908（Tyro CLI → Python config loading）、#4941（移除 torchcomms 依赖）。

**NVIDIA/Megatron-LM**

- **#6805 wide-residual 流的选择性 replay**｜https://github.com/NVIDIA/Megatron-LM/pull/6805
  - 为 #6716 的 streamwise wide-residual（主 decoder 携带 `K * hidden_size` 残差流）加有序选择性 replay：把本地层切成 replay block，块内只把"残差读 / 分支 norm / 非末端残差写"三类廉价算子交给 `CheckpointWithoutOutputManager`，分支主体（attention / Mamba / MLP / MoE）仍留在普通 autograd 图里不重算。
  - 显存模型：单层普通残差激活成本 `X`，K 流残差为约 `K*X` 每个保留边界，块大小 M 层后摊薄到 **`K*X/M` 每层**（只覆盖残差边界激活，分支/优化器/通信内存不变）。`fp32_residual_connection=True` 时残差流保持 FP32、分支按 `params_dtype` 执行。
  - 开关：`--recompute-granularity selective --recompute-modules residual_stream --residual-stream-recompute-num-layers 4`。校验会拒绝无 wide residual、非 selective、非正块大小、CUDA graph capture 等组合。
  - **对我的意义**：一个新的重计算粒度——用极低重算代价换 K 倍残差显存，与昇腾侧自研 `--recompute-modules` 语义可直接对标。
- **#6855 wide residual 支持 MTP 与 MIMO**：`HybridStack` 新增 `is_mtp_layer` 区分主 decoder 与 MTP 辅助栈。主 decoder 做一次 `[S,B,D] → [S,B,K,D]` 展开、可选残差 replay、末端读回到 `D`；**MTP 栈不展开、不建宽读出、不参与 replay，shifted-token embedding 与辅助层全部保持普通宽度 D**，因此 `wide_residual + MTP` 的互斥拒绝被移除。MIMO 侧：文本 embedding、projector 输出与合成张量都在 `D` 上组合，合成后在主 decoder 边界只展开一次。
  - **对我的意义**：MTP 是 DeepSeek 系必配，这条明确了"宽残差只属于主干、MTP 头不跟着变宽"的工程契约，移植到 MindSpeed / MindFormers 时是关键边界。
- **#7359 用 optimizer 自己的 GTP group 对 GTP-replicated 参数去重**｜https://github.com/NVIDIA/Megatron-LM/pull/7359
  - `param_is_not_gtp_duplicate` 原先从 MPU 全局量读 GTP rematerialization 轴的 rank，只在调用方的 GTP 轴恰是全局轴时才正确。自建 grid 的模块（`skip_model_parallel_init=True`）会让**每个 rank 都读成 rank 0，去重静默失效**：`get_grad_norm_fp32` 在跨全世界的 group 上求和，每个复制参数被贡献 `gtp_remat_size` 次，**报出的 grad norm 随 GTP size 线性放大**，梯度裁剪行为随之改变而其它日志量看起来完全正常；`count_zeros_fp32` 同样多计。
  - 修法：把 optimizer 自己 `ProcessGroupCollection` 里的 `gtp_remat` / `expt_gtp_remat` group 透传给该 filter，MPU 读取仅作回退。新测试固化契约：各 `gtp_remat_size` 的归约 norm 必须等于单进程 norm。注意作者声明单测需多卡、**未本地运行**。
  - **对我的意义**：典型的静默数值错误——混合并行轴 + 模块自建 process group 时 grad norm/裁剪被放大。万卡上 GTP/EP/CP 叠加的场景值得照此自查一遍。
- **#7430 训练 config 新增独立 `RLConfig` 段落**（479+/258−，9 文件；PR 描述模板未填，无机制细节）。配合同窗口的 `Serialize callable model-config fields`（#7571），说明 Megatron 主干正在把 RL 配置收敛成一等公民——对以 Megatron 为训练后端的 verl / MindSpeed-RL 意味着配置面将要对齐。

**NVIDIA/TransformerEngine**

- **#3454 NVFP4 row-scaled 路径把 row/col amax 融进单个 TMA-tiled kernel**：原先 columnwise amax kernel 按列主序读全局内存（非合并访问）并主导运行时；改为单 kernel 经 TMA 把 128×128 chunk 流过 shared memory、从 SMEM 做列归约。由 `fused_amax_supported`（BF16、维度 128 对齐）gate。
  - **后续修了一个隐蔽 bug**：fused wrapper 原用无条件 memset 清零两个 amax buffer，而 kernel 在 `noop[0]==1` 时提前返回，于是**被跳过的（graph-replay）调用会清掉此前已发布的 amax**；改为 noop-aware 的清零 kernel。
  - **对我的意义**：这类"提前返回的 kernel + 无条件 memset 的 wrapper → 丢失已发布的 scale"在 ACL Graph + 动态量化路径上完全同构，值得专门检查一遍。
- **#3467 grouped 输入的单次 launch fused amax**：在打包的 `(sum_M, K)` 输入上单次 launch 算出每 expert 的 row/col amax，取代每 expert 一次 launch（`nvte_group_nvfp4_compute_amax`）。后续修了两处：公共入口直接调 launcher 会绕过 eligibility 检查，导致**未对齐的 K 静默丢弃尾列**、异构 per-expert amax buffer 可能解引用空指针。
  - **对我的意义**："每专家一次 launch"是 MoE + 低精度的典型小 kernel 泛滥问题；"未对齐 K 静默丢列"值得在自家 grouped 量化算子上自查。
- **#3565 非 FP8/FP4 GEMM 拒绝 bias/out dtype 不匹配**：cuBLASLt 在 `BIAS_DATA_TYPE` 未设时假定 bias dtype 等于 D，而 TE 只在 FP8/FP4 路径设该属性，因此 **BF16 bias + FP32 out 会被重新解释并越界读取 bias 分配**。
- **#3313 Blackwell（SM100/103）上 THD + dropout 训练优先选 FlashAttn 2 而非 Fused Attn**：cuDNN 的 THD dropout kernel 在 SM100/103 上远慢于 FA2，且 cuDNN 团队确认不在修复路线图；原先的 Hopper+ 通用规则让该组合静默走了慢路径且无任何提示。
- **#3503 以 Dispatch/Combine 为基础算子的 MoE Sequential Block**（commit message 无机制描述）：把 dispatch/combine 提升为一等 op 是 EP 可组合性（与 AC/compile/graph 交互）的前提，与 torchtitan #4918 暴露的 handle 生命周期问题是同一方向的架构应答。
- **#3586 在 C++ 层借用 NCCL comm 以支持 `ProcessGroupNCCL2` 后端**。

**deepspeedai/DeepSpeed**

- **#8655 fp16 loss scaling 下保持 Muon 的 update 原尺寸**｜https://github.com/deepspeedai/DeepSpeed/commit/3509874ec2502cceab63e842c9c91e7440f0fc08
  - fp16 下 Muon 参数在 ZeRO-1/2/3 几乎不训练：Newton–Schulz 在任何 loss scale 下返回同样的 update，但 step 把它当成被 scale 过的梯度——`unscale_and_clip_grads` 除以 loss scale，于是 Muon 参数实际只移动 `1/loss_scale`，**默认动态 loss scaler 下是 1/65536**。
  - 实测 2×H20 单步 update norm（static scale 1 vs 1024、无裁剪）：ZeRO-1/2/3 main 为 `4.47e-2` vs `4.37e-5`，本 PR 为 `4.47e-2` vs `4.47e-2`（CPU offload 路径本来就正确）。60 步小回归：bf16 `0.403 → 0.295`，fp16 main `0.403 → 0.385`，本 PR `0.403 → 0.295`。
  - 实现上在 cast 到 fp32 之后、`unscale_and_clip_grads` 之前把 update 放大回去（留在 fp16 梯度缓冲里放大会溢出）。未覆盖 ZenFlow 与 SuperOffload。
  - **对我的意义**：这是连续第三个 Muon + ZeRO 的静默错误（前两个是 #8600、#8464）。通病是"优化器 update 不是梯度、不能走 unscale 路径"。bf16/fp32 下 loss scale 为 1 故不触发，但在昇腾上若用 fp16 + ZeRO + Muon，这个 bug 几乎必然存在。
- 生态信号：#8718 **新增华为 Fengchun Hua / hipudding 进入 TSC committer 名单**。

**sgl-project/sglang**

- **#37787 为 DSA 类模型在 NPU 上加 decode context parallel**｜https://github.com/sgl-project/sglang/pull/37787
  - 21 文件 / +1570 −55，由 `sglang-npu-bot` 合入。速度测试配置为 TP16 EP16、input 64k / output 1k、90% prefix cache，对比 base 与 `tp16 ep16 dcp2`。**但精度与速度结果只以截图给出，正文无可引用数值**。
  - **对我的意义**：昇腾上长上下文 decode 阶段的 context parallel 能力落地（64k 输入、DCP=2），是 NPU 侧超长序列推理并行的关键增量——但数值待核，建议直接看 PR 截图或自行复测。
- **#41717 Kimi-K3 的 NPU PD 分离部署 recipe**（文档，但对昇腾部署有直接价值）。
- **#41970 gfx950 上 DeepSeek-V4.1 的 fp8 dense GEMM 切到 aiter 的 MXFP8 GEMM**：kernel 级 aiter/native 比值 M=6 **1.05×（更慢）**、M=12 1.02×、M=24 **0.65×**、M=48 0.43×、M=96 0.48×、M=512（prefill）0.71×——即**小 M（低并发 decode）无收益甚至微负，M≥24 后收益显著**。端到端 8k/1k 并发 16：TTFT 1096 → 943 ms（-14.0%），TPOT -4.2%，输出吞吐 +4.8%。
  - **对我的意义**：虽是 AMD，但"MXFP8 GEMM 在小 M 不划算、大 M 才赢"的规律对昇腾的 MXFP8 算子选择同样成立，可与 vllm-ascend #17567/#17364 横向对照。
- 其余：#41500 把 ascend 一致性 GT 对到 **CANN 9.1.0** baseline；#39614 Qwen4-Exp 压缩 QSA indexer cache 可选 fp8 存储。

**vllm-project/vllm**

- **#50030 per-token NVFP4 CuTe-DSL MoE 后端**：主要动机是**非 gated 模型**（此前这类模型的 online per-token NVFP4 会回退到 Marlin W4A16）。对齐规则值得记：runtime 中间维度 gated 对齐 64 / 非 gated 对齐 128；**runtime hidden 维 K 对齐 256**——GEMM1 无 K-tail predicate 地抓取完整 K tile，Nano 的 `K=2688` 会越界读激活/scale，故 pad 到 2816。
  - GSM8K（1319 题、5-shot、greedy、eager）：Qwen3-30B-A3B TP4 BF16 94.09% / CuTe 93.56% / TRTLLM 93.71%；**Nemotron-3-Nano-30B-A3B TP4+EP4 BF16 23.20% → CuTe 17.59%（仍差 5.61 个百分点，作者明确未达精度对等）**；Nemotron-3-Super-120B-A12B TP8+EP8 BF16 93.78% → CuTe 94.24%。
  - **对我的意义**：W4A4 在非 gated MoE 上的精度代价有了一手数据（小模型掉 5.6 点是警示）。对昇腾是精度方案对标，不是可移植实现。
- 其余以 CPU/aarch64 kernel、ROCm（#58819 DeepSeek V4.1 的 a4w4 FP4 激活 MoE on AITER）、KV connector（#54483 NIXL 跨 cache group 合并 host-buffer KV 拷贝）为主。

**THUDM/slime**：窗口内仅 1 个清理类 commit（#2432 移除过时的 Straw queue limits）。注：slime 的 score centering（#2405/#2406，Megatron + SGLang）在 09-23 合入、**早于本窗口**，但正是 verl #8010 的前序参考，对昇腾侧移植最有价值。

**OpenRLHF / NCCL**：窗口内 0 commit。

**MindSpeed / MindSpeed-LLM / MindSpeed-RL**：**未确认**。WebSearch 只返回 gitee/gitcode 的仓库首页与 release 列表页，未返回任何带窗口内时间戳的发布信息，无法确认是否发版（"未确认"不等于"没有"）。

---

## Sources

- DeepGEMM-Ascend：https://github.com/deepseek-ai/DeepGEMM-Ascend
- DeepEP-Ascend：https://github.com/deepseek-ai/DeepEP-Ascend
- Gemini 4 Argon：https://deepmind.google/models/gemini/
- 论文：https://arxiv.org/abs/2609.36070 ，https://arxiv.org/abs/2609.37852 ，https://arxiv.org/abs/2609.39137 ，https://arxiv.org/abs/2609.39350 ，https://arxiv.org/abs/2609.36830 ，https://arxiv.org/abs/2609.36899 ，https://arxiv.org/abs/2609.36802 ，https://arxiv.org/abs/2609.37538 ，https://arxiv.org/abs/2609.36954 ，https://arxiv.org/abs/2609.39361
- PyTorch CRCR：https://pytorch.org/blog/from-upstream-changes-to-downstream-confidence-inside-torch-spyres-integration-with-pytorch-crcr/
- AMD TraceLens：https://rocm.blogs.amd.com/software-tools-optimization/tracelens-analysis-agent/README.html
- NVIDIA cuObject/SCADA：https://developer.nvidia.com/blog/expanding-ai-storage-access-with-nvidia-cuobject-and-the-nvidia-scada-server-sdk/
- Google Cloud 月报：https://cloud.google.com/blog/topics/ai-infrastructure/whats-new-in-ai-infrastructure-this-month
- PyTorch v2.14.1：https://github.com/pytorch/pytorch/releases/tag/v2.14.1
- 各 PR / commit 链接见正文。

## 运行备注

- **窗口说明**：本次定时任务触发于北京时间 10-02 02:35（UTC 10-01 18:34），非配置的 08:00。上一期日报为 **09-24**，中间 09-25～09-29 没有日报；按"只收窗口内"的规则，本期窗口取 09-30 08:00 → 10-02 02:35（约 42.5h），**09-25～09-29 的间隙未补收**。标题日期按生成当日的北京时间取 2026-10-02。
- **因间隙而未计入的 Release**（均在 09-29，落在窗口外，若需补收请单独指示）：TensorRT-LLM v1.3.0rc29、trl v1.14.1、vllm-ascend v0.27.1rc1。
- **去重**：已对照 09-24 那期排除 Megatron #6486/#7573/#7579、torchtitan #4704/#4837/#4596、TRT-LLM v1.3.0rc28、TE #3282、verl #7994/#7679、sglang #40779、vllm-ascend #17361/#17355/#17066、NCCL4Py v0.6.0、Qwen3.8-Omni、MiMo-V2.6、Fireworks Ember-1、NVIDIA SWE-Serve。
- **验证强度偏弱、已就地标注的条目**：Megatron #7359（多卡单测未本地运行）、vllm-ascend #17364（未跑 CI/CANN 构建/NPU E2E）、torch_npu DVM r2.11（未建 wheel、CI 待确认）、verl #8065（自述人工逐行复核与完整 CI 未完成）、sglang #37787（数值仅截图）。vllm-ascend #17270/#17567/#16706 的数字均为作者声明的"非端到端/无性能主张"，已原样标注。
- **窗口外、未收录（供参考）**：
  - 09-29 vLLM Blog《Taking vLLM Apart: A Practical Guide to Disaggregated Serving》：https://vllm.ai/blog/2026-09-29-disaggregated-serving-guide
  - 09-28 华为开源 openPangu-2.0 的预训练/SFT/RL 训练代码（Pro 505B 总参 / 18B 激活，512K 上下文，约 34T token；称昇腾原生训练效率提升 30%，在 GitCode 的 ascend-tribe 组织下）——**这条与读者相关度很高，建议补读**。
  - 09-28 ROCm《Verl 0.9.0 on ROCm 10.0》；09-28 SemiAnalysis《How GLM5.3 Sparse Attention Affects HBM Memory Usage》；09-23 ROCm《Debugging Logprob Mismatches in LLM RL》。
  - 09-29 OpenAI GPT-6.1 Sol（官方 X 17:26Z，DevDay 09-29）：1.05M context、最多 128K 输出、$2/$10；架构未披露。09-28 Anthropic Claude Sonnet 5.5。
  - 09-17 昇腾 960 超节点（4096 卡、1PB HBM、NPO 光引擎，2027 Q2/Q3 上市）。
- **窗口内、未采信或低相关**：
  - 媒体转述的 DeepSeek 昇腾组件推理数字（DeepSeek-V4.1-Flash 在 EP32/128K/投机接受率 0.85 下 TPOT=5ms 时 2469 tok/s、TPOT=10ms 时 5102 tok/s；Scale-up 128 卡 3.2Tbps 单层交换、Scale-out 256K 卡两层交换）来自量子位与机器之心转述，**未在一手仓库或官方页面核实，故未写入正文**。DeepSeek 官方原文（推测为公众号）未获取到。
  - 「六件套」中 TileLang 昇腾版、FlashMLA 昇腾版、TileKernels 昇腾版未能在仓库层面核实（FlashMLA README 无昇腾内容；DeepSelect v1.0.0 为 09-10，可能是 GPU 版）。
  - Google Project Suncatcher 4 颗 TPU 入轨（媒体称发射日 10-01）：未找到 Google 原文，未确认是否已发射。
- **检索失败或受限**：
  - arXiv：`catchup/cs.LG/2026-10-01` 被代理 429 限流，**cs.LG 与 cs.CL 在 10-01 的完整 listing 未能逐条浏览**，该部分用 alphaXiv 检索补齐，可能有遗漏。`huggingface.co/papers?date=2026-10-01` 返回 400（09-30 正常）。arXiv 直连 curl 被代理 403。cs.DC 的 pastweek 页面返回过期内容（停在 09-11），改用 catchup 页面。
  - GitHub：`gh` CLI 对 deepseek-ai 全部仓库返回 403（本会话无该仓库权限），改用 MCP 工具但一次只能查一个仓库，**14 个厂商 org 的 `pushed:>=2026-09-30` 系统扫描只完成了 deepseek-ai 一个**，这是本期最大检索缺口。`list_commits` 对 vllm-project/vllm 返回 100 条分页上限且仅含 sha，改用 PR 标题检索，**vllm 一节不保证窗口内穷尽**。
  - 未获取任何新模型的 config.json 原文（窗口内本无新开放权重 LLM，但方法上未执行到位）。
  - lmsys.org/blog、Modal、Anyscale 需 JS 渲染无法抓取；lm-sys.github.io 被 robots 禁止；rocm.blogs.amd.com/blog-index.html 404；Together AI 部分文章无日期；Google Cloud 博客列表页无日期；weibo 与 X 被 robots 禁止；digitimes 403。
  - Gitee/GitCode 上 MindSpeed 系列的 Release 日期无法确认。
  - curl 直连被代理拒绝（CONNECT tunnel failed / 403），所有 HF API 查询改用 WebFetch 完成；WebFetch 经小模型摘要，存在漏项与转述风险。
- **保存结果**：
  - GitHub：已写入 `suhaibo666/tracker` main 分支 `llm-training-daily/llm-training-daily-2026-10-02.md`（commit `b85658c`）。注：`gh` CLI 与 curl 的写操作被代理拒绝（403），改用 GitHub MCP 工具完成。
  - 本地：**未写入**。`device_commit_files` 所需的设备桥接在本次运行中掉线，`get_device_info` 两次均返回「设备未连接」，按要求未反复重试。目标路径 `/Users/suhaibo/workspace/90-knowledge/llm-monitor/llm-training-daily-2026-10-02.md` 需手动从 GitHub 拉取，或在电脑在线时重跑本次任务。
