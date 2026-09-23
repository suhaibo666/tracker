# LLM 技术动态日报 · 2026-09-23

> 窗口：2026-09-22 08:00 → 2026-09-23 13:27（北京时间，手动试运行，见运行备注） · 生成时间：2026-09-23 13:27

## 今日要点

1. **Megatron-LM 把 MXFP8 权重接进 Megatron-FSDP（#7114）**：FSDP 分片、all-gather 的都是 MXFP8 数据平面加 scale 平面，结果与 TE 单卡参考逐位一致。这是低精度参数分片训练的重要一步。
2. **torchtitan 连续做了三个结构性合入**：Float8/NVFP4 的 autograd 从 TorchAO 收回 torchtitan 自己维护（#4820）；DistMuon 支持在单个 attention head 内部切分（#4840）；删除 `ModelSpec` 与每个模型的 `parallelize.py`（#4810，breaking）。fork 或适配 torchtitan 的 NPU 分支需要尽快 rebase。
3. **ByteDance Seed 发布 NSP（嵌套序列并行）**：让不同大小的 SP 组嵌套在同一批 GPU 上，用来处理变长长序列训练。在 Qwen3-MoE、384K 上下文下，吞吐最高是静态 SP 的 1.48x、FlexSP 的 1.16x。
4. **DeepSeek 公开 Agentic RL 沙箱基础设施 DSec**：单个规模单元约 160 节点，每天约 300 万个沙箱，线上并发超过 38 万。它和 RL 框架协同设计：rollout 状态与可抢占的训练资源解耦。
5. **vLLM v0.30.0 发布**：新增 Fast Start（常驻的 IPC 权重缓存守护进程）和 HiSparse（sparse-MLA 的 host 侧 KV 层）；Model Runner V2 支持 DBO 与 PP 下的投机解码。vllm-ascend 同期发布 v0.23.0.post1，并合入了 PCP 下 embedding/LM head 分片等特性。模型侧，Anthropic 发布 Claude Opus 5.5，OpenAI 发布 GPT-6 Sol/Luna，两者都没有披露架构。

---

## 1. 模型发布与 Tech Report

**Claude Opus 5.5**｜Anthropic｜2026-09-22（约 16:30 UTC）｜https://www.anthropic.com/claude-opus-5-5
- Claude 5.5 系列的第一款模型；Sonnet 5.5 和 Haiku 5.5 官方说「未来几周」发布。
- 规格：1M 上下文，最大输出 128K，adaptive thinking 常开，默认 effort 为 medium。
- 定价（每 1M tokens）：输入 $4、输出 $20，cache read $0.20；fast mode 为 $8 / $40。
- 官方称相比 Opus 5：典型负载成本约低 40%，输出速度快 30% 以上，同一任务消耗的 token 更少。
- 部分 benchmark：Terminal-Bench 4.0 66.4%（Opus 5 为 52.3%），CursorBench 4.0 57.8%，OSWorld 2.0 81.8%（部分）。
- 架构、训练规模、硬件均未披露。
- **对我的意义**：对训练 infra 没有直接信息量。「serve 所需算力下降 + 输出提速 30%」说明闭源头部厂商仍在持续压 serving 成本。

**GPT-6 Sol / GPT-6 Luna**｜OpenAI｜2026-09-22（约 18:00 UTC）｜https://openai.com/index/introducing-gpt-6-sol-and-luna/
- 官方称用与 GPT-6 Astra「相似的方法」训练的较小模型：Sol 面向复杂编码和 Agent 任务，Luna 面向高吞吐、任务单一的场景。
- 规格：1.05M 上下文，最大输出 128K；reasoning effort 分 none 到 max 六档。
- 定价（每 1M tokens，输入/输出）：Sol $2 / $10，Luna $0.10 / $0.50。
- Sol 成绩：DeepSWE v1.1 68.8%（max effort），OSWorld 2.0 60.5%。
- 架构、精度、硬件未披露。
- **对我的意义**：大模型蒸馏或同配方训练出的小模型成了主流的价格档位；训练侧的大规模蒸馏和后训练流水线吞吐更值得投入。

**Hy Image3.5 preview**｜腾讯混元｜2026-09-22 10:47（北京时间）｜https://www.ithome.com/1/005/589.htm
- 闭源图像生成模型，仅提供 API：支持 2K 分辨率、最多 5 张参考图、多轮编辑。
- 与 LLM 训练 infra 关系弱，这里只记一笔。

> 窗口内 DeepSeek、Qwen、Moonshot、智谱、MiniMax、Meta、Google、Mistral、NVIDIA、AI2、字节 Seed、百度、阶跃、美团、Microsoft 均无新模型权重或 tech report。刚好落在窗口外的 Xiaomi MiMo-V2.6、Grok 4.7 见运行备注。

## 2. 论文精选（训练 / 后训练 / 推理）

### 训练系统

**NSP: Accelerating Variable-Length LLM Training via Nested Sequence Parallelism**｜ByteDance Seed｜arXiv 2609.22755（9-19 提交，9-22 上 listing）｜https://arxiv.org/abs/2609.22755
- **与 FlexSP 的区别**：FlexSP 把 GPU 切成互不相交的动态 SP 组。NSP 则在同一次 fwd/bwd 里，把不同大小的 SP 组嵌套在同一批 GPU 上：长序列走大组，短序列留在小组，按 GPU 而不是按组做负载均衡。
- **Planner**：用 profiling 得到的代价模型，在显存约束下做 beam search，生成一棵 SP 树。
- **Executor**：
  - 层间 phase streaming，让不同树层级之间的计算与通信重叠；
  - 以整棵树为粒度做重计算。
- **实现**：基于内部的 FSDP 风格 PyTorch 框架，兼容常见 SP 后端，不需要改模型。
- **实验**：64 卡生产集群，Qwen3-MoE，最长 384K。端到端吞吐最高是静态 SP 的 1.48x、FlexSP 的 1.16x。
- **对我的意义**：长上下文、变长数据混训是昇腾万卡集群的真实痛点。NSP 的核心只是 CP/SP 组的动态嵌套与调度，和硬件无关，可以评估能否移植到 MindSpeed 的 CP（Ring/Ulysses）实现上。

**QEffect: Explicit State and Resource Contracts for Low-Precision Pipeline Parallel Training under Captured Graphs**｜宁波工程学院 / 浙大｜arXiv 2609.23536（9-20 提交，9-22 上 listing）｜https://arxiv.org/abs/2609.23536
- 针对的问题：CUDA Graph replay 与 FP8 delayed scaling、Zero-Bubble（dI/dW 拆分）PP 同时使用时，会出现静默竞争和数值损坏。
- 论文把正确性形式化为四条不变量：
  - 量化状态的更新必须串行；
  - 保留下来的 backward 工作按「代」管理归属；
  - 缓存权重带有效性版本号；
  - stream 与 graph 之间做双向完成同步。
- 实现与结果：基于 TorchTitan + Transformer Engine。跨 scaling 翻转仍保持与 full-backward 基线逐位一致，并支持冷重启。在 H800 上，graph capture 让单层提速 1.82–2.79x；梯度直接落位（每 rank 每步少 96 次冗余拷贝）再带来 1.132x。
- **对我的意义**：昇腾上 ACL Graph（aclgraph）+ FP8/MXFP8 + ZB-PP 会遇到完全同构的风险。这四条不变量可以直接变成 MindSpeed 的 code review checklist 和回归测试项。

**DeepSeek Elastic Compute (DSec): A Sandbox Infrastructure for Effective Agentic Training at Scale**｜DeepSeek-AI & 清华｜arXiv 2609.22978（9-19 提交，9-22 上 listing）｜https://arxiv.org/abs/2609.22978
- 统一 SDK，支持 FnCall / 容器 / microVM / 完整 VM 四种后端。
- 与 RL 框架协同设计：有状态的 rollout 执行与可抢占的 GPU 训练解耦；沙箱生命周期随训练协调，空闲资源可以回收，但 rollout 状态保留。系统也包含防 reward hacking 的机制。
- 镜像基于 3FS 按需加载，采用分层、带版本的环境：
  - 如果改成 eager 拉取镜像，完成时间会变为 1.7x；
  - 按需加载让累计磁盘写入减少 57%。
- 规模：单个规模单元约 160 节点、每天约 300 万沙箱；线上并发超过 38 万，每秒创建超过 5000 个。
- **对我的意义**：Agentic RL 的瓶颈正在从 rollout 引擎转向环境与沙箱层。可以对照 verl 的 agent loop，评估自家的沙箱能力（冷启动、镜像分发、训练抢占后状态能否保留）。

### 后训练

**Conduit: An Experience Data Plane for Distributed Reinforcement Learning**｜Aalto / 浙大等｜arXiv 2609.24456（9-21）｜https://arxiv.org/abs/2609.24456
- 在 RLlib（8×A100）上，experience 数据的搬运和处理占迭代时间：DQN 50.2%，PPO 26.2%。
- Conduit 把这条数据路径做成一个显式的数据平面，在 CPU/GPU 分层之间调度，暴露延迟最多降低 97%，端到端最多降低 38%，可扩展到 1024 GPU。
- 注意：实验对象是经典 RL，不是 LLM RL。
- **对我的意义**：思路可以借鉴到 LLM RL 的 trajectory/payload 平面（MiMo-V2.6 也单独做了一个 payload plane），优先级中低。

**1% of Tokens Can Be Enough: On Gradient Estimation in On-Policy Distillation**｜MBZUAI & 蚂蚁｜arXiv 2609.24432（9-21，HF Daily 9-22）｜https://arxiv.org/abs/2609.24432
- 用「信息效率比」（一个信噪比指标）选出需要 teacher 监督的 token。
- 只给 0.1–1% 的 token 提供 teacher logits，结果可以持平甚至超过全量 on-policy distillation。
- **对我的意义**：teacher 前向是 OPD 的主要算力开销。token 稀疏化可以直接降低 teacher 推理的集群开销。

### 推理

**HySparse2: Hybrid Sparse Attention with Two-Level KV Sharing**｜Xiaomi LLM-Core｜arXiv 2609.26368（9-22）｜https://arxiv.org/abs/2609.26368
- 架构：YOCO 式的 self-decoder（SWA + full attention）加 cross-decoder（token 级稀疏注意力）。
- KV Bridging：由 self-decoder 的 hidden state 构造 cross-decoder 的全注意力 KV，稀疏层复用这份 KV 和 selection 索引。因此 prefill 只需要跑 self-decoder（49 层里的 25 层）；在 PD 分离部署下，prefill 节点只需约一半权重。
- 验证：80B-A3B MoE（500B 预训练 + 100B 后训练 token，256K 上下文）上，RULER-v2 比 HySparse 高 19.81 分，MRCR-v2 高 11.30 分。
- 1M 上下文下，prefill FLOPs 比 HySparse 少 2.92x，比 Hybrid SWA 少 5.02x；FP8 KV cache 只有 2.69 GB。另在 290B-A8B / 1.8T token 规模上验证了 KV Bridging。
- **对我的意义**：这是训练期就决定的架构变化（跨层 KV 共享 + 可提前退出的 prefill）。它会影响 PP 切分和 PD 部署的形态，值得在架构选型评审时参考。

**Fast Recovery for LLM Serving via Decoupled Device Memory Lifetime in Dynamo**｜NVIDIA｜arXiv 2609.25451（9-21）｜https://arxiv.org/abs/2609.25451
- 18 周的线上故障数据显示：大多数故障是「设备保留型」的，即引擎进程挂了，但 GPU 和上面的显存分配还在。
- 做法：GPU Memory Service 把显存所有权从引擎进程里剥离，权重常驻，并只读共享给替换引擎和预热好的 Shadow 引擎。
- 效果：在 vLLM/SGLang 上，副本恢复时间小于 7 s，比 warm restart 快 13–29x；代价是每卡固定 4–8 GiB。按生产 trace 回放，可挽回 79% 因恢复而损失的 GPU 小时。
- **对我的意义**：这与 vLLM v0.30 的 Fast Start 是同一方向。训练侧的故障恢复（checkpoint 常驻 HBM 或 host、进程级热替换）可以用同一套思路。

**Disaggregated Quantization: Specializing LLM Prefill and Decode**｜NVIDIA & ISTA｜arXiv 2609.26333（9-22）｜https://arxiv.org/abs/2609.26333
- 按阶段用不同的量化格式：compute-bound 的 prefill 用 NVFP4 计算原生权重，memory-bound 的 decode 用 1–3.5 bit 的 LUT/GGUF 仅权重量化。
- 在 Qwen3.8-27B 上，训练得到的 NVFP4 prefiller 让 IQ1_S 的 MMLU-Pro 提高 32.5 分。
- SSD 流式 prefill 让 8K 输入的 TTFT 提速 1.78x（llama.cpp）；共享权重版本已在最大 2.8T 参数的模型上验证 PTQ。
- **对我的意义**：PD 分离后，两侧可以用异构精度甚至异构硬件。这对昇腾「训推一体 + 不同代卡混部」的部署规划有参考价值。

**H-Spec: Parallel Speculative Decoding Without a Drafter-Side KV Cache**｜Harvard / Red Hat / ISTA / NVIDIA｜arXiv 2609.24197（9-21）｜https://arxiv.org/abs/2609.24197
- Mamba-attention 混合结构的并行 drafter：attention 部分原地复用 target 模型的 KV，Mamba 部分用 target 最后一个 token 的 hidden state 初始化，因此 drafter 不需要自己的 KV cache。
- 平均接受长度提高 5.0–13.3%，batch=1 的 ITL 提速 5.3–12.6%；在所有测试的并发度下吞吐都更高。
- **对我的意义**：解决了高并发下 drafter KV 带来的显存压力，适合显存紧张的推理部署。

**Adapting Tree-Structured Speculative Decoding to DeepSeek-V4**｜百度百舸 & 复旦｜arXiv 2609.24698（9-21）｜https://arxiv.org/abs/2609.24698
- 问题：DeepSeek-V4 的在线压缩注意力（CSA/HCA）会让树的不同分支压缩出不同的状态。
- 做法：在 SGLang 中实现分支感知的因果校验、用 scratch pad 隔离分支状态，并对 C4 和 C128 两种压缩器采用不同的刷新节奏。
- 结果：8 卡 DeepSeek-V4-Flash，D=8 时接受长度为 2.83–3.41（线性投机为 2.39–2.84）；吞吐峰值 +18.5%，D≥7 后稳定在约 +9–10%。
- **对我的意义**：vllm-ascend / MindIE 适配 DSV4 系列的投机解码时，会碰到同样的分支一致性问题。

> 其他窗口内相关论文，未精读：2609.24639（PD 分离推理的功耗感知资源供给）、2609.26763（SARA，Agentic 服务的 SLO 感知分配）、2609.25782（HBM/HBF 冷热 KV 分层）、2609.25463（华为等，RL rollout 效率综述）、2609.24205（按阶段调 GPU 频率的训练节能）、2609.22087（可容忍可预测算力中断的训练）。

## 3. AI Infra 技术文章

**Hardware-Agnostic Models in vLLM**｜PyTorch Blog（IBM / Meta / HF）｜2026-09-22｜https://pytorch.org/blog/hardware-agnostic-models-in-vllm/
- 背景：vLLM 的模型定义正在为 Blackwell / CDNA4 做 flat 化特化，这会破坏 `torch.compile` 以及树外的硬件插件。
- 做法：新增 `model_executor/hw_agnostic` 层集，只用原生 PyTorch、Triton 和 Helion，可以 full-graph 编译，支持 CustomOp/PluggableLayer。
- 效果：在 H100 上比原生实现只慢 3.4%（3 个模型的几何平均）。已合入，通过 `USE_HW_AGNOSTIC=1` 开启。
- **对我的意义**：vllm-ascend 这类树外插件的长期维护成本和这条路线直接相关。建议跟进，评估 NPU 能否以 hw_agnostic 路径为 fallback。

**How Shopify built a continual learning loop with PyTorch and vLLM**｜PyTorch Blog｜2026-09-22｜https://pytorch.org/blog/how-shopify-built-a-continual-learning-loop-with-pytorch-and-vllm/
- 流程：线上失败样本每天回流训练，先 SFT 蒸馏再做 GRPO；全参训练用 TP + CP + DP，serving 用 vLLM。
- 6000 token 的 system prompt 被压成约 1500 个可学习的 gist token，没有测到质量损失。
- 收益：成本从每年约 $27M 降到约 $1M，TTFT 降低 19%。
- **对我的意义**：这是「持续后训练 + 推理闭环」落地的一手数据。训推之间的权重同步频率和流水线自动化是基础设施的关键。

**Topology-Aware Workload Scheduling with NVIDIA Topograph**｜NVIDIA Technical Blog｜2026-09-22｜https://developer.nvidia.com/blog/topology-aware-workload-scheduling-with-nvidia-topograph/
- 开源工具：自动发现云厂商 API、IB、Spectrum-X 以及多节点 NVLink 域的拓扑，统一建模。
- 输出给调度器：K8s（labels / KAI / Kueue TAS）和 Slurm（tree/block topology，25.05+）。
- **对我的意义**：万卡作业的放置质量直接决定跨层级集合通信的效率。昇腾超节点（UB 域）加 RoCE scale-out 同样需要类似的拓扑抽象，可以参考它的数据模型。

**Private High-Performance AI Inference with NVIDIA Confidential Computing**｜NVIDIA Technical Blog｜2026-09-22｜https://developer.nvidia.com/blog/enabling-private-high-performance-production-ai-inference-with-nvidia-confidential-computing/
- 测试配置：DGX B200（TP8），DeepSeek-R1，TRT-LLM，32K 输入 / 1K 输出。
- 开启机密计算后，吞吐保持在基线的 96.1–98.2%，TPOT 增加 1.2–4.3%。
- 适配改动：pageable H2D 拷贝；autotune 计时改用 GPU timer；集合通信要能在没有 NVLS multicast 的情况下工作。
- **对我的意义**：政企场景的机密推理需求在增长。「不能使用 NVLS multicast」这一约束对通信库选算法有启发。

## 4. 硬件与互联

无新增。窗口内没有能确认发布时间的硬件或互联重大发布。

以下均早于窗口：

- Huawei Connect 2026（9-17 至 9-18）发布 Ascend 960 SuperPoD（NPO 超节点，2027 Q3 出货）和 UnifiedBus 相关新闻稿；
- NCCL 最新版为 v2.32.3-1（9-17）；
- NVIDIA DSX Ready（9-21）。

## 5. 开源框架与社区

### 正式 Release

**vLLM v0.30.0**｜2026-09-22 05:20 UTC｜https://github.com/vllm-project/vllm/releases/tag/v0.30.0
- 规模：762 个 commit，315 名贡献者。
- **Fast Start**：每张 GPU 一个常驻守护进程，保存量化后、按 TP 切好的权重；引擎重启后通过 CUDA IPC 直接映射（`--load-format ipc_cache`）。支持 FP4 checkpoint 和多机 TP；DP 支持已在 #57386 合入。
- **HiSparse**：sparse-MLA decode 的 host 侧 KV 层，top-k 命中不了的页从 pinned host memory 取回。
- **Model Runner V2**：
  - dual-batch overlap；
  - PP 下支持 MTP/EAGLE3 投机解码；
  - H200 上 graph capture 从 12s 降到 2s，引擎初始化从 28.9s 降到 8.2s。
- **大规模 serving**：
  - sparse-MLA 支持 PCP+DCP；
  - Elastic EP 在重配置之间复用 CUDA graph；
  - 无 NVLink 机型可用 FlashInfer PCIe IPC all-reduce；
  - DeepEP v2 async finalize。
- **Breaking changes**：
  - scale-out 端点改为 `--enable-scale-out` 显式开启；
  - 移除 GPTQ 的 `g_idx` 激活重排；
  - YaRN 与 Transformers 对齐；
  - 默认 wheel 改为 CUDA 13.0。
- **对我的意义**：Fast Start 对 RL 训推切换、推理故障恢复都有价值。需要关注 vllm-ascend 何时能把 IPC 权重缓存迁移到 NPU（依赖 ACL IPC 能力）。

**vllm-ascend v0.23.0.post1**｜2026-09-22 02:47 UTC｜https://github.com/vllm-project/vllm-ascend/releases/tag/v0.23.0.post1
- 修复 DP 对齐 dummy run 时写入过期 KV 的问题（#15362）。
- 修复 ACL Graph 模式下，RL 重载权重后 W8A8 MXFP8 变换 buffer 不稳定的问题（#13905）。
- **对我的意义**：用 verl + vllm-ascend 做 MXFP8 rollout 的，建议升级到这个版本。

### 重要合入

**NVIDIA/Megatron-LM**
- **MXFP8 权重接入 MFSDP（#7114，9-22）**｜https://github.com/NVIDIA/Megatron-LM/commit/85cda9247bc5ea0afe2bc6e5207839115d10e2f9
  - 通过 `QuantizedDBuffer` 实现：优化器更新高精度 main weight，MFSDP 直接重量化到未 swizzle 的紧凑视图，再分发数据和 scale 两个平面。
  - 按存储 dtype 分组参数；MXFP8 组的 `Shard(0)` 使用 `BlockAtomic(32)`。
  - 在 2 张 GB200 上跑 3 步 FusedAdam，与 TE 单卡参考逐位一致。
  - **对我的意义**：FSDP 通信量直接按 FP8 计。昇腾 MXFP8（A5 代）上 MindSpeed FSDP2 做同类改造时，可以参考 32 元素对齐的原子分片约束。
- **HybridModel hash MoE routing（#6403）**：新增基于哈希的 MoE 路由选项。
- **Energon 多模态 SFT 数据管线（#7254 / #7255）**。

**pytorch/torchtitan**
- **Float8/NVFP4 的 autograd 由 torchtitan 自己维护（#4820）**：dense 和 grouped 的 Float8/NVFP4 autograd，以及 FSDP 管理的权重操作数，都收回 torchtitan；TorchAO 只提供底层量化和 GEMM kernel。
- **DistMuon 变长 BlockShard（#4840）**：支持在单个 q/kv head 内部切分（head 内还有 no-rope/rope 两段）。
- **NIXL 权重同步（#4549）**：通过 RDMA 更新的权重从 CuMem allocator 分配，保证地址固定。这是 RL 权重同步路径。
- **BREAKING（#4810）**：删除 `ModelSpec` 和每个模型的 `parallelize.py`，并行化逻辑移入 `Model`/`Decoder`，新增 `Model._apply_fsdp()`。不保证向后兼容。
- **EMA 伪优化器（#3985）**：可追踪 MoE 的 `expert_bias`；支持 CPU offload，在 side stream 上更新。
- **RL checkpoint 转换后恢复 DTensor placement（#4787）**。
- **对我的意义**：#4810 会直接打断基于 ModelSpec 的 NPU 适配层。#4820 表明低精度训练的主导权正在从 TorchAO 移到训练框架。

**pytorch/pytorch**（9-22）
- Pipelining 一组 PR：
  - FSDP unshard 分散到各个 schedule 计算段里（#196706）；
  - 根据 stage 放置推导有向 P2P 进程组；
  - 基于 schedule 做激活存活期分析；
  - 接收 buffer 延迟到需要时再分配。
- `DeviceMesh.abort()` API（#189655）。
- Inductor 在 Blackwell max-autotune GEMM 上启用 2-CTA。
- 注意：outer gradient accumulation（#196641）当天合入后又被 revert。

**deepspeedai/DeepSpeed**
- **ZeRO-3 下 Muon 改为每个优化器步只跑一次（#8600）**：之前每个 micro-batch 都会推进 momentum、并对部分梯度做 NS 正交化。修复后，gas=1 与 gas=4 的权重偏差从 1.3e-1 降到 3.6e-4。
- **AutoEP 在各层之间共享同一个 DeepEP buffer（#8557）**：修复 EP32 在第 28 个 buffer 时 `ncclDevCommCreate` 失败的问题。
- **AutoEP 区域编译（#8380）**：Qwen3-30B-A3B EP16 上，每步吞吐 +5.7–6.6%。
- **NPU accelerator 实现 graph 操作（#8605）**：基于 `torch.npu.NPUGraph`。之前是空桩，`graph_harvesting` 缓存的是空图，梯度范数和 clip 会静默冻结。910B4 上 replay 提速 4.1–22.7x，5 步训练与 eager 逐位一致。
- **对我的意义**：#8605 修的是昇腾上一个静默错误。用 DeepSpeed + NPU 并开启 graph_harvesting 的训练需要复查历史结果。

**volcengine/verl**
- **FP8 MoE refit 修复（#7986）**：kernel repack 后 shape 和 dtype 都不变的 MoE 层（FlashInfer CUTLASS block-FP8 的 w13→w31、TRT-LLM MXFP8 的 interleave）之前不会被重新 staging。在 1×B200 上，修复后 rollout_corr/kl 从 0.047 降到 0.0022（bf16 基线为 0.0019）。影响 main 分支上 Hopper EP>1 的 block-FP8 配置。
- **NPU 上 Qwen3.5-122B 性能优化（#7980）**：作者称吞吐翻倍，但 PR 描述没有细节。
- **rollout 相关**：新增 `ray_actor_max_concurrency`，避免 wake_up、权重同步等控制调用被大量 generate 请求饿死；`init_standalone` 可接受外部传入的 `RayResourcePool`。
- **对我的意义**：#7986 是 FP8 rollout 训推不一致的一个隐蔽根因。做昇腾 MXFP8 rollout 时要检查同类「同 shape 重排」问题。

**THUDM/slime**：支持 score centering（#2405，+1966/−40），PR 无说明。

**NVIDIA/TransformerEngine**
- **CP P2P 前向峰值显存从 O(C) 降到 O(1)（#2916）**：KV 双缓冲，并在累加器中做 running merge。
- **Linear/LayerNormLinear/LayerNormMLP 不再对输入做 view（#3523）**：归一化、transpose、comm-GEMM overlap 等路径支持任意输入 shape。
- **对我的意义**：#2916 是一个可移植的 ring-attention 显存优化，CP 规模越大收益越明显。

**NVIDIA/TensorRT-LLM**
- **Blackwell 上的 MXFP8 fused FC12 MoE kernel（#19548，合入 `feat/rl_rc` 分支）**：GB300 DEP4、16k token 时从 2514 降到 1913 us。
- **Helix CP 下的投机校验组（#19273）**：支持 fp8/fp4 MLA 和 DSpark，但 PR 描述说明只做过静态验证，合入时未编译运行。
- **其他**：DeepSeek-V4 的 NVFP4 MLA residual 开关；Q 的 FP8 量化融合进 absorb-bmm 的 epilogue；升级到 PyTorch 2.14、Triton 3.8 和 C++20。

**vllm-project/vllm**（Release 之外）
- DeepSeek-V4.1 的 MXFP8 `wo_b` GEMM 与 SP reduce-scatter 融合（#57428）：TP4 算子级最多快 37%，serving +2.9%。
- ROCm 支持 BF16 AsyncTP 融合；可选在加载时把 MXFP4 反量化。

**sgl-project/sglang**
- Qwen4-Exp 支持 PP serving 以及 PD-prefill 侧的 MTP（#40501）。在 GB300 上，仅 prefill、ISL 8192 时，TP1×PP4 为 38,310 tok/s/GPU，DEP4 为 30,162，约 1.27x。
- DSv4.1 的 FP4 indexer 跳过不可见 tile。
- 修复 NIXL 传输 MXFP8 KV block scale 的问题。
- BREAKING：移除已废弃的端点、环境变量和别名（#40795）；并移除 C++ radix tree 和 HiRadixCache。

**vllm-project/vllm-ascend**（Release 之外）
- **PCP 下 embedding/LM head 分片（#17159）**：PCP>1 时默认开启。A3×8、TP4×PCP2、128K prefill 下，每个 rank 省 0.24 GiB，吞吐 −0.34%。
- **KVPP 支持 PCP pooling 和 PIECEWISE 执行（#17112）**。
- **DP=1 时也可开启 MoE SP（#17061）**。
- **MRV2 在 graph 模式下支持 o_proj TP（#16967）**。
- **Mooncake 逐层 KV pool 支持 PP（#17183）**。
- **DCP decode 的 stream 与 V-up 投影优化（#16977）**。

**Ascend/pytorch（torch_npu）**
- **fxrt 成为 torch_npu 树内子包（`torch_npu.fxrt`）**：需要显式调用 `register_fx_wrapper()` 才会启用，`import torch_npu` 时不加载；可用 `--disable_fxrt` 去掉；新增构建依赖 nanobind。
- **PyTorch ≥2.14 时移除 Dynamo Stream/Event 的 in-graph patch**：改为依赖上游 #194343。
- **Inductor FlexAttention**：合并有界 workspace 与运行时 DispatchPlan（fwd / dQ / dK-dV 统一分发）。
- **对我的意义**：升级到 PyTorch 2.14 时要注意 Dynamo patch 的行为差异；fxrt 进入主包后，FX 图运行时成了官方路径。

**MindSpeed-LLM**
- 没有能确认的 Release。
- GitHub 镜像有一个窗口内提交：`fix(fsdp2): stabilize Step3.5 EP training`。它把专家权重统一为 `[expert, output, input]` 布局，供 FSDPTurbo 的 grouped GEMM 使用；只对混合 Tensor 和 DTensor 的参数组关闭 AdamW 的 `foreach`。
- 链接：https://github.com/Ascend/MindSpeed-LLM/commit/53348315c8155c6c370d2c279c98433c93a263c3

> 窗口内无 Release：Megatron-LM（core_v0.19.2，9-18）、torchtitan（v0.3.0）、PyTorch（2.14.0，9-02）、DeepSpeed（v0.19.7）、verl（v0.9.1，9-20）、OpenRLHF（窗口内也无提交）、slime、SGLang（v0.5.20，9-18）、TRT-LLM（v1.3.0rc27，9-18）、TE（v2.19）、TRL（v1.13.0）。TRL 合入了 GRPO/RLOO/DPO 等使用的 fused logprob+entropy kernel（#7253）。

---

## Sources

- Anthropic Claude Opus 5.5：https://www.anthropic.com/claude-opus-5-5 ；TechCrunch：https://techcrunch.com/2026/09/22/anthropic-releases-opus-5-5-with-lower-prices-and-fable-level-performance/
- OpenAI GPT-6 Sol/Luna：https://openai.com/index/introducing-gpt-6-sol-and-luna/ ；https://community.openai.com/t/announcing-gpt-6-sol-and-gpt-6-luna-in-the-api-codex-and-chatgpt/1399925
- 腾讯 Hy Image3.5：https://www.ithome.com/1/005/589.htm
- 论文：https://arxiv.org/abs/2609.22755 ，https://arxiv.org/abs/2609.23536 ，https://arxiv.org/abs/2609.22978 ，https://arxiv.org/abs/2609.24456 ，https://arxiv.org/abs/2609.24432 ，https://arxiv.org/abs/2609.26368 ，https://arxiv.org/abs/2609.25451 ，https://arxiv.org/abs/2609.26333 ，https://arxiv.org/abs/2609.24197 ，https://arxiv.org/abs/2609.24698
- PyTorch Blog：https://pytorch.org/blog/hardware-agnostic-models-in-vllm/ ，https://pytorch.org/blog/how-shopify-built-a-continual-learning-loop-with-pytorch-and-vllm/
- NVIDIA Technical Blog：https://developer.nvidia.com/blog/topology-aware-workload-scheduling-with-nvidia-topograph/ ，https://developer.nvidia.com/blog/enabling-private-high-performance-production-ai-inference-with-nvidia-confidential-computing/
- Release：https://github.com/vllm-project/vllm/releases/tag/v0.30.0 ，https://github.com/vllm-project/vllm-ascend/releases/tag/v0.23.0.post1
- 各 PR/commit 链接见正文。

## 运行备注

- **手动试运行**（非 08:00 定时触发）。
  - 触发附言要求把窗口改为「过去 24 小时，以当前时间为终点」。
  - 为兼顾默认规则，实际窗口取两者并集：2026-09-22 08:00 → 2026-09-23 13:27（北京时间）。
- **去重基线**：`suhaibo666/tracker/llm-training-daily/` 最近一份是 2026-09-02，与本窗口相隔约三周，没有逐条读取全文。本期条目均在窗口内，不存在与既有日报重复的问题。
- **窗口外未收录（仅供参考）**：
  - **Xiaomi MiMo-V2.6（Pro 1.02T/42B-A、Flash 约 309B/15B-A）及其 tech report**：X 发布于 9-21 20:51 UTC（北京时间 9-22 04:51），HF repo 创建于 9-21，早于窗口。报告披露了 Muown 优化器、MXFP4 QAT、异步 GRPO + partial rollout、rollout routing replay；另有一个结论：RL 期间冻结 router 可以避免专家坍缩（负载 CV 从 0.78 恶化到 2.0，22% 专家变冷）。建议另找时间精读。
  - **xAI Grok 4.7**（9-21 21:16 UTC）；**StepFun Step 5 Preview**（9-20）；**Fireworks 博客《The frontier isn't a model. It's a router.》**（9-21）。
  - **9-21 的 vLLM 博客「Qwen3.8-2.4T PD Serving」** 和 **SemiAnalysis「Computation and Data Movement for Inference」**：处在边界，未收录。
- **检索失败或受限**：
  - huggingface.co 与 arxiv.org 的 curl 被代理拦截（403），改用 WebFetch。
  - HF org 按 createdAt 排序的列表可能不完整（XiaomiMiMo 的新 repo 没有出现在列表里），「HF 无新增」不是定论。
  - `huggingface.co/papers?date=2026-09-23` 返回 400；arXiv 9-23 的 listing 没有直接拿到，由 alphaXiv 补覆盖。
  - lmsys.org/blog、Modal、Anyscale、华为新闻中心的列表没能解析；Intel newsroom 的索引已过期。
  - 知乎、量子位没有直接访问到。
  - Gitee、GitCode（MindSpeed）无法访问（robots 超时 / HTTP 418）。
  - `search_pull_requests` 对 pytorch/pytorch 返回 0 条，改用 `list_commits` 扫 main 分支，只合入 release 分支的 PR 可能漏掉。
- **保存结果**：
  - GitHub：写入 `suhaibo666/tracker` main 分支 `llm-training-daily/llm-training-daily-2026-09-23.md`。
  - 本地：**未写入**。本次运行在云端执行，没有连接用户电脑，`device_commit_files` 不可用，所以没有保存到 `/Users/suhaibo/workspace/90-knowledge/llm-monitor/`。这与桌面 App 是否打开无关，而是任务未绑定该电脑，或本次运行未获得电脑访问权限。可以在该电脑上从 GitHub 拉取。
  - 会话内副本：`outputs/llm-training-daily-2026-09-23.md`。
