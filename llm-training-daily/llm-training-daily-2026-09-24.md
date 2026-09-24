# LLM 技术动态日报 · 2026-09-24

> 窗口：2026-09-23 08:00 → 2026-09-24 08:00（北京时间） · 生成时间：2026-09-24 09:52

## 今日要点

1. **Megatron-LM 让 M-FSDP v2 支持 PP/VPP（#6486）**：所有 VPP chunk 共用同一个 `fully_shard_context`（共享 comm stream、prefetch 顺序和 microbatch 状态）。在 DeepSeek-V3 proxy 上与 DistOpt 的 val-loss 相差约 0.3%。这是 MFSDP 进入「FSDP + PP」万卡标准配置的最后一个阻塞点。
2. **RL 基础设施：一篇权重同步论文、一个死锁修复和一个 rollout 控制原语**：
   - Meta FAIR 的 **WeightBridge**（arXiv 2609.25442）：GPU 停顿比 NCCL broadcast 低 16.6–29.4x，已在 Kimi-K2 1T 上验证。
   - **sglang #40779**：修复 `pause_generation` 与权重更新之间的死锁，在 512 个在飞请求时复现过。
   - **verl #7994**：rollout 可以在不暂停的情况下中止在飞请求，并暴露 KV 压力快照。
3. **Qwen3.8-Omni tech report 发布**：骨干是 Gated DeltaNet + MoE + QSA 稀疏注意力；256K 原生长度预训练约 2.5T token；采用「dense 注意力 warmup → indexer 蒸馏 → 联合稀疏训练」的四阶段配方。参数量和训练硬件均未披露。
4. **TensorRT-LLM v1.3.0rc28 发布**：新增 **Router Replay (R3)**，把 MoE 逐 token 的路由结果回传给训练引擎；集成 NCCL-EP 0.2；NVLink one-sided MoE A2A 上限提高到 256 rank；KVCacheManagerV2 默认开启。另外，TE #3282 让 EP dispatch 可以被 CUDA Graph 捕获，并省掉一次 AllGather。
5. **vllm-ascend 当天合入 44 个 commit**：细粒度 TP 扩展到 MRV2 的 MLP 和 embedding（A3 PD 实测 TPOT 几乎不变，MLP=2 时每 rank 省 0.55 GiB）；MooncakeConnector V2 支持 PCP；block table 只提交变更区间（H2D 从约 2.72 ms/step 降到约 0.14 ms/step）；新增 W4A4C8 量化方案文档。

---

## 1. 模型发布与 Tech Report

**Qwen3.8-Omni: Towards Native Omni-Modal Agents（tech report）**｜Qwen / 阿里｜arXiv 2609.25611（9-22 提交，9-23 进入 cs.CL listing）｜https://arxiv.org/abs/2609.25611

- **架构**：Thinker-Talker。Thinker 复用 Qwen3.8-Next 的混合骨干：sparse MoE，token mixing 由 Gated DeltaNet 与 attention 层交替组成；attention 层改用 **QSA（Qwen Sparse Attention）**，即轻量 indexer 给压缩后的 micro-block 打分，再对选中块内的原始 token 做全注意力。
- **音频编码器**：AuT 下采样 16x，得到 6.25 Hz 的 token；新增 Spatial AuT，输入 FOA 多声道的复数 STFT，保留相位和声道间幅度。
- **词表**：约 250K 的 byte-level BPE。
- **预训练（四阶段，全程 256K 原生长度）**：
  - S1：冻结 LLM，对齐编码器；
  - S2：全量训练约 2.5T token（文本 1.1T、音频 0.7T、图像 0.35T、视频 0.15T、音视频 0.3T）；
  - S3：冻结骨干，从 dense 注意力分布蒸馏，warmup QSA indexer；
  - S4：打开稀疏注意力，骨干与 indexer 联合训练。
  - 后训练结束后把上下文扩展到 1M。
- **后训练**：先从各领域专家教师（各自做过 SFT+RL）多教师蒸馏，再做统一的 outcome-reward RL。Talker 训练用到多教师 on-policy 蒸馏（MOPD）和 GSPO，Talker 带 RVQ codec 与 MTP head。
- **结果**：
  - 文本：SWE-bench Pro 63.3，GPQA-D 91.0，HLE 36.5，LiveCodeBench v6 92.6；
  - 多模态：29 项音频 / 音视频评测平均比 Qwen3.5-Omni-Plus 高 25% 以上；
  - 实时交互：音频 TTFT 约 0.59 s。
- **简析**：
  - 核心增量是把 DSA/NSA 式的「dense → indexer 蒸馏 → 稀疏」迁移配方，做到线性注意力 + 稀疏注意力的混合骨干上，并且全程 256K 训练。
  - 未披露：总参数和激活参数、专家数与路由方式、训练硬件、并行配置、精度。没有 HF 权重也没有 license，看起来只提供 API。
  - 模型本身 9-18 已经上线，这里收录的是 tech report。
- **对我的意义**：长上下文持续预训练中「先训 indexer 再联合稀疏」的阶段划分，可以直接作为昇腾上 DSA 类稀疏注意力迁移训练的参照。同时要注意 GDN 与稀疏注意力混排后 CP 切分的非对称性：线性层的 state 需要跨 rank 传递，而稀疏注意力需要 gather 被选中的块。

> 其他厂商：窗口内 OpenAI、Anthropic、Google、Meta、xAI、Mistral、DeepSeek、Moonshot、智谱、MiniMax、字节 Seed、腾讯、百度、阶跃、小米、美团、Microsoft、AI2 均未发布新的 LLM 权重或 tech report。非 LLM 发布（Nemotron 3 Diarization 等）见运行备注。

## 2. 论文精选（训练 / 后训练 / 推理）

### RL 基础设施 / 后训练

**WeightBridge: An Efficient Weight Transfer Library for Reinforcement Learning**｜Meta FAIR / Harvard｜arXiv 2609.25442（9-22 提交，9-23 进入 listing）｜https://arxiv.org/abs/2609.25442

- **自动推断布局**：把各推理引擎的 `load_weights` 当黑盒，喂入「元素 ID」而不是数值，自动推断 trainer 与 rollout 之间的布局映射，不需要为每个模型手写 reshard 规则。ID 映射用二维二阶差分编码压缩，大小随切片操作数量增长，与模型规模无关。
- **全局规划**：每个逻辑元素只从 trainer 发出一次，每个 rollout 节点只接收一份去重后的副本，节点内再经 NVLink 扇出，达到带宽下界。传输是 receive、RDMA、NVLink 三段流水，pack/assemble kernel 耗时占比不到 3%。
- **实验**：H100，Megatron trainer + SGLang rollout（Miles 框架），模型为 Moonlight-16B-A3B、Qwen3-30B-A3B、Qwen3-235B-A22B、Kimi-K2 1.03T，规模 2–64 节点。
  - GPU 停顿比 NCCL broadcast 低 16.6–29.4x，比 P2P 低 5.4–42x；
  - 端到端耗时是理论下界的 1.18–2.12x；
  - 在 Meta 内部 RL 框架中提速 6.3–11x。
- **反直觉发现**：全同步传输（停顿 137 ms）比经 host 暂存的异步传输（609 ms）更快，瓶颈在 PCIe。
- **局限**：trainer 与 rollout 的精度必须一致（不支持量化 rollout），也没有弹性和容错。
- **对我的意义**：「每节点一次入、一次出，节点内扇出」正好对应昇腾「HCCS 节点内 + RoCE 跨节点」的拓扑，可以对照 verl 在 Megatron/MindSpeed → vllm-ascend 这条路径上的 reshard 实现。不支持 MXFP8/W8A8 rollout 是移植前必须补的缺口。

**PACT: From Credit Assignment to Critic Alignment**｜AllSpark Research｜arXiv 2609.26355（9-22）｜https://arxiv.org/abs/2609.26355

- **理论**：用 Completeness、Prefix Consistency、Neutrality 三条公理唯一确定 token 级 credit。据此推出：理想的 OPD teacher 等价于隐式 critic；RLOO 的响应级信号在期望上等于 token credit；credit 近似稀疏，因此 GAE 下 critic 的误差可能与 credit 本身同量级。
- **方法**：先更新 actor，再用 IS 校正后的目标训练 critic，让 critic 跟上已经更新的策略。
- **结果**：
  - Qwen3.5-4B + OpenCode 做 Agentic 数学：Avg@16 为 72.87%，比 GRPO（Clip-Higher）高 8.80 pp，比 PPO（λ=1）高 13.16 pp；PPO λ=0.95 崩溃。
  - Qwen3.6-35B-A3B 做 SWE-bench Verified：67.4%，GRPO 为 65.4%。
- **对我的意义**：改动只涉及更新顺序和 critic 目标，接入 verl 的 PPO 路径成本很低。长程 Agentic RL 如果要回到带 critic 的方法，这篇可以作为依据。

**Greedy Decoding Is Not Precision-Invariant**（TMLR）｜UTK / UChicago / Amazon｜arXiv 2609.26621（9-22）｜https://arxiv.org/abs/2609.26621

- **现象**：同一硬件上，1.1B–7B 的 6 个模型用 BF16 和 FP16 做 greedy 解码，49–100% 的 prompt 输出不同。
- **原因**：是否翻转主要由 lm_head 处 top-2 logit 的 margin 决定，而不是模型主体累积的误差；把更多计算改成 FP32 反而让一致率下降。
- **缓解**：只在 margin 较小时用 FP32 重算 lm_head，精确一致率提高 12–36 pp，开销不到 4%。但 batch≥8 或端到端 FP8 时效果消失。
- **对我的意义**：这解释了 RL 训推 logprob 不一致的一部分来源（昇腾训练侧 vs vllm-ascend rollout）。「按 margin 门控的 FP32 lm_head」是一个便宜的诊断和缓解手段。

**Train Where the Quantized Model Goes: On-Policy Distillation for Low-Bit Reasoning**｜中科院自动化所等｜arXiv 2609.26708（9-22）｜https://arxiv.org/abs/2609.26708

- **问题**：低于 3 bit 的 QAD 在短答案 QA 上能保留约 82% 的 BF16 性能，但在 MATH-500 上只剩 35%，长生成会陷入重复循环。作者归因为量化放大了 exposure bias。
- **方法**：QAD 之后再加一段 OPD，让学生走部署时的量化前向路径做 rollout，由 BF16 teacher 提供 token 级反馈，同时加 verifier 奖励。
- **结果**：在 Qwen3 0.6B–4B 上，MATH-500 的保留率从 35% 升到 70%，HumanEval 从 66% 升到 91%。
- **对我的意义**：低 bit rollout 或 serving 模型的「量化感知 OPD」配方。目前只在小模型上验证过。

### 训练系统 / 互联

**Co-Fabric: Breaking Host-Domain Boundaries for Unified xPU Interconnection**｜浪潮信息（IEIT）｜arXiv 2609.25560（9-23 进入 listing）｜https://arxiv.org/abs/2609.25560

- **设计**：总线式 scale-up fabric。四层协议栈，处理延迟在 ns 级，可靠性在协议内实现；按端口 ID 跨 host 域路由，并提供统一地址空间。
- **实验**：64 个 xPU 组成 3D mesh，对比 8×400G RoCE。
  - AllReduce：延迟降到 RoCE 的 10–50%，带宽提高 2–15x；
  - DeepSeek-R1 推理快 30–80%（128 并发时 1.8x）；
  - 从 8 卡扩到 64 卡，DeepSeek-V3 LoRA 保持 93% 扩展效率，QwQ-32B 全参保持 98%；
  - 互联成本低约 80%。
- **对我的意义**：国产统一总线超节点（与灵衢 / UB 同一路线）的又一个实现。它的 3D mesh 拓扑和跨 host 地址空间设计，可以用来和 CloudMatrix/UB 做横向对比。

**Accelerating the Mitigation of LLM Inference Nondeterminism Across GPU Architectures**｜Georgia Tech / UC Merced｜arXiv 2609.25624（9-23 进入 listing）｜https://arxiv.org/abs/2609.25624

- **方法**：GEMM 采用固定配置，16-bit 权重在寄存器里升到 FP32，按 IEEE-754 顺序累加。规约顺序只由问题 shape 决定，与 SM 数量和调度无关。
- **结果**：Ampere、Ada、Hopper 三代架构上线性层输出逐位一致；端到端比现有做法（把 BF16 权重整体拷一份 FP32）快 1.17–3.1x，权重访存减半。
- **对我的意义**：这是「shape 决定规约顺序」的 batch-invariant GEMM 设计，可以迁移到 AscendC GEMM，用于 RL 可复现 rollout 和训推混合硬件场景。

### 推理

**PatchKV: Efficient KV Cache Recovery for Dynamically Edited LLM Contexts**｜天津大学｜arXiv 2609.26219｜https://arxiv.org/abs/2609.26219

- **场景**：Agent 上下文中间一段被编辑、后面的长后缀保持不变。
- **方法**：用离线 drift 模型预测会失效的邻近区域，再按存下来的注意力补充远处的稀疏块，只重算这部分；其余后缀从 CPU 恢复，按块带精度标签，反量化、RoPE 校正、KV 页放置在一个 fused 步骤里完成。
- **结果**：恢复后的 TTFT 比全量重算快 2.51–3.85x，比 CacheBlend 快 1.26–2.06x。
- **对我的意义**：多轮 Agentic RL rollout 中工具输出被改写时，可以借此复用 KV。

**You Only Need 2/3 of the Chosen Experts**｜中科院自动化所 / USTC｜arXiv 2609.25809（仅读摘要）｜https://arxiv.org/abs/2609.25809

- **实验范围**：9 个架构族、12 个 MoE 模型。
- **结论**：统一只保留 top-k 中约 2/3 的专家，性能保持 98.8%，提速 1.2–1.7x；按 token 精细分配专家只在激进剪枝时才有额外收益（最多 +3%）。
- **对我的意义**：推理和 rollout 阶段可以直接降低 EP all-to-all 的通信量，成本极低。

**CompKV: Compensation-Aware KV Selection**｜清华等｜arXiv 2609.26300（仅读摘要）｜https://arxiv.org/abs/2609.26300

- 选块的依据是「后续补偿步骤的残差」，而不只是注意力质量；残差由块内注意力质量和 logit 波动共同决定。
- 结果：自注意力最多提速 6.85x，RULER 和 LongBench-Pro 上质量不降。
- **对我的意义**：可作为 DSA/NSA 类块稀疏 decode kernel 的选块准则。

> 其他窗口内相关论文，未精读：
> - 2609.25949 Flux：面向训练依赖图的光电路交换 MILP 调度，仅做了仿真。
> - 2609.25869 Tessera：动态块稀疏注意力运行时。
> - 2609.25482：Terminal Shrinkage Averaging，末期迭代与最近 K 个 checkpoint 平均的混合。
> - 2609.25048：OPD 中 prompt 数量与 rollout 刷新频率的相互作用。
> - 2609.27421：反事实约束 OPD（阿里）。
> - 2609.27981：风险可控的 KV 驱逐。
> - 2609.26796 Flash-dLLM：扩散 LLM 的 IO 感知 KV cache。
> - 2609.24797：Complex KDA。

## 3. AI Infra 技术文章

**How SWE-Serve Exposes the Gap Between Local Tests and Live Serving**｜NVIDIA Technical Blog｜2026-09-23｜https://developer.nvidia.com/blog/how-swe-serve-exposes-the-gap-between-local-tests-and-live-serving/

- **基准构成**：从 83 个已合入的 SGLang PR 构建 53 个仓库级任务，其中投机解码 14 个、kernel/量化/性能 8 个；参考修复的中位数是改 7 个文件、553 行。19 个任务需要用真实模型拉起服务。
- **关键结论**：Agent 生成的补丁在不做 live serving 校验时通过率 69.4%，做完整端到端校验后只有 45.9%，约三分之一「通过」的补丁会弄坏真实服务。
- **模型对比**：11 个模型的 pass@1 在 34.6–75.5% 之间；同样 64% 通过率的模型，每个任务成本在 $0.95 到 $7.24 之间。
- **对我的意义**：用 coding agent 改 MindSpeed 或 vllm-ascend 时，必须用端到端拉起服务或训练来把关，只看单测会把正确率高估约 1.5x。

**Introducing Ember-1**｜Fireworks AI｜2026-09-23｜https://fireworks.ai/blog/ember-1

- 对 Kimi K3 做后训练：输出 token 减少 35–40%，质量不降；推理 token 占输出的比例从约 90% 降到 71.3%。
- 过程中做了 50 多组训练实验、200 多次评测，没有披露硬件和规模。
- **对我的意义**：压缩推理长度是提高单卡 serving 吞吐的低成本手段，本质是一个后训练信号。

## 4. 硬件与互联

**NCCL4Py v0.6.0**｜NVIDIA｜2026-09-23 17:51 UTC｜https://github.com/NVIDIA/nccl/releases/tag/nccl4py-v0.6.0

- **Host API**：基于 NCCL 2.32.3 头文件重新生成。新增每个集合通信的 launch-completion CUDA event（需要 CUDA 12.3+）、NVLS host-mode 配置、CFT（Compute Fabric Transport：multicast / counted-write）能力查询，以及 window 标志 `GIN_ONLY` / `CFT_COUNTED`。
- **实验性 CuTe DSL device API**：可以在 kernel 内对对称内存（LSA）window 和 NVLS multimem 做 reduce/copy（`lsa_/multimem_reduce_sum`、`_copy`、`_reduce_sum_copy`）。
- **Breaking**：`ThreadScope.THREAD` 的值从 3 改为 10；barrier session 必须恰好调用一次 `destroy()`。
- **对我的意义**：NVIDIA 正在把 kernel 内通信原语（GIN、CFT、symmetric memory）开放给 Python/CuTe 的 kernel 作者。HCCL 和灵衢的 device 侧通信 API 需要对齐这个方向，才能写 MoE 的计算通信融合 kernel。

> 其他硬件厂商：窗口内没有新芯片、互联、CUDA/ROCm/CANN 大版本，也没有 NCCL 或 RCCL 主版本发布。阿里 9-22 发布的平头哥真武 V900 和磐久 AL64 光互联超节点早于窗口，见运行备注。

## 5. 开源框架与社区

### 正式 Release

**TensorRT-LLM v1.3.0rc28**（prerelease）｜2026-09-23 06:34 UTC｜https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.3.0rc28

- **RL**：**Router Replay (R3)**，把逐 token 的 MoE 路由结果回传给训练引擎，用来解决 MoE RL 训推路由不一致（#18397）。
- **通信与 EP**：
  - 集成 NCCL-EP 0.2 低延迟 EP（#18689）；
  - NVLink one-sided MoE A2A 的 rank 上限提高到 256（#18800）；
  - NIXL bounce-buffer cache transceiver；
  - 统一 KV 传输后端接口。
- **KV cache**：KVCacheManagerV2 对 Llama/Llama4/Nemotron 多模态默认开启；DSV4 支持 NVFP4 KV；PrimTS MLA decode 支持 FP8 KV。
- **Rubin（SM107）**：DSV4/DSA kernel；2:4 激活稀疏 FMHA；DeepGEMM FP8 block scale；CuteDSL fused FC1+FC2 NVFP4 MoE。
- **Breaking**：移除旧的 TensorRT serve/eval/bench 路径；删除 Eagle 的 `eagle_choices` 字段；Triton MoE 后端被标为弃用。
- 配套版本：PyTorch 2.13.0 / Triton 3.7.1。main 分支当天已升级到 PyTorch 2.14 / Triton 3.8（#19477）。
- **对我的意义**：R3 进入推理引擎主干，说明「rollout 路由回放」正在成为 MoE RL 的标配接口，vllm-ascend 和 verl 的 NPU 路径需要同类能力。

**NCCL4Py v0.6.0**：见第 4 节。

### 重要合入

**NVIDIA/Megatron-LM**

- **#6486 让 M-FSDP v2 支持 PP**｜https://github.com/NVIDIA/Megatron-LM/commit/2539bc339f3d19503c4f6d6943f4dff69d4249bf
  - 原问题：VPP 下每个 chunk 各自开一个 `fully_shard_context`，各有独立的 comm stream、microbatch 状态和 prefetch 顺序，1F1B 调度没法协调 AG/RS。
  - 改法：新增 `reuse_existing`，所有 VPP chunk 共享一个 context，finalize 推迟到外层执行。
  - 验证：DeepSeek-V3 proxy，PP2/VPP2/DP4，1000 步，val-loss 6.023，DistOpt 为 6.019。
  - **对我的意义**：MindSpeed 基于 FSDP2 的 PP 组合可以参照这里的「共享 context + 推迟 finalize」。
- **#7573 GTP + MTP 下 output_layer 的 wgrad reduce-scatter 每步只做一次**｜https://github.com/NVIDIA/Megatron-LM/commit/bf3ffc38f1602eae9481be4fc2eacc7fea80329b
  - 做法：利用 TE `general_gemm` epilogue 的 `accumulate=True` 累加。第一次 backward 直接覆写，兼作清零，不需要第二块 buffer；最后一次消费时再发出唯一一次 RS，与 backward 计算重叠。
  - 结果：4×GB200、GTP=4 时，reduce-scatter 从每步 3 次降到 1 次，梯度与原来逐位一致，峰值显存不变。
  - **对我的意义**：MTP 训练中 vocab 大小的集合通信可以被合并，这在 HCCL 上同样是热点。
- **#7579 residual-connection 与 streamwise 算子原语**（+2063 行）
  - 内容：`ResidualConnection` 协议，以及 BF16/FP16/FP32 的 Triton fused fwd/bwd。
  - 定位：宽残差架构（#6716）的前置工作，从 nv-mistralai-megatron 移植过来，说明 Transformer block 结构即将改动。
- **#7539 FSDP placement 中的 `Flat` 改名为 `RowAtomic`**：显式写了该 placement 的配置属于 breaking。
- **#7344 checkpoint 目录内保存 `run_config.yaml`**。

**pytorch/torchtitan**

- **#4704 TP 投影后端重构**｜https://github.com/pytorch/torchtitan/commit/9e159aed7267f77fbfa746ee051525a4c5594e5e
  - 新增 `ColumnParallelLinear` 和 `RowParallelLinear`：前者负责输入 all-gather，后者负责计算后的 RS/AR；两者都是 `Linear` 的子类，FQN 不变。
  - `AsyncTensorParallelTransform` 改为通用实现；LoRA 和量化投影会被显式拒绝。
  - **对我的意义**：NPU 后端可以在继承通信边界的同时替换本地计算（`_linear()`）。已 fork 的 NPU 分支需要注意这次改动。
- **#4837 RegionAC 管理 AllToAll 通信区域**
  - `TokenDispatcher` 改为 Module，新增三个 remat 区域，由同一个开关控制；DeepEP 的 dispatch/combine 始终保存。
  - **对我的意义**：可以对 MoE all-to-all 的激活做选择性重计算。
- **#4596 Kimi K3 recipe 改用按 head 切分的 DistMuon**
  - KDA 层（占全部层的 3/4）的 q/k/v、forget_b、output_gate 按 head 分块；`beta`、`A_log`、norm、embedding、LM head 等仍用 AdamW。
  - 验证：FSDP4 与 FSDP4+EP4 的 loss 曲线一致。
  - **对我的意义**：给出了 Muon + EP 大规模训练中哪些参数该按 head 分块、哪些保留 AdamW 的现成规则。

**pytorch/pytorch**

- **#196641 Pipeline 支持 outer gradient accumulation**：昨天被 revert，今天重新合入。
- **分布式基础**：
  - #198306 修复 symm_mem 在各后端 CUDA 清理时的同步问题；
  - #198251 PyProcessGroup 的 reconfigure 可以透传到 Python 层 override，这是自定义后端（包括 NPU）做弹性容错的前置条件；
  - #197616 拆出 TorchComms 后端的创建逻辑。
- **显存**：#198129 导入 expandable segment 时只保留共享部分的地址空间，减少 IPC 共享（RL 权重传输、KV offload）时的 VA 占用。
- **其他**：#198250 CP mask 使用实际的 BlockMask 块大小；#198168 DTensor 允许剩余 rank 上为空分片。
- 另有一组 dynamo perf-guard PR 当天合入又整体 revert，可以忽略。

**deepspeedai/DeepSpeed**

- **#8464 修复 ZeRO CPU offload 下的 Muon**｜https://github.com/deepspeedai/DeepSpeed/commit/d099bc61bb95bb71abaac3d3c1ba6911f15b2b95
  - 原问题：参数被展平成 1D 分区后丢失矩阵形状，Newton–Schulz 被**静默跳过**，退化成一阶动量；ZeRO-3 offload 下 momentum 在 CPU、grad 在设备上，导致重复更新。
  - 修复：做 NS 之前先还原成 2D/3D；all-gather 临时 buffer 改为 LRU 缓存，上限 256 MB。
  - **对我的意义**：又一个 Muon 在 ZeRO 下的静默错误（昨天是 #8600）。在昇腾上用 DeepSpeed + Muon 的训练需要复查。

**volcengine/verl**

- **#7994 在不暂停的情况下中止在飞请求，并暴露 KV snapshot**｜https://github.com/verl-project/verl/commit/178b0eeb983157e2d3d9c66b416c5d8e6e708e8c
  - `abort_only=True` 直接取消 vLLM 的在飞请求，不关闭准入、不清空 prefix cache。
  - `snapshot()` 读取 `kv_cache_usage_perc` 和 waiting/running 请求数；PD 分离下不支持。
  - **对我的意义**：rollout 调度器可以根据真实的 KV 压力做准入和中止，是 partial rollout 和异步 RL 的基础原语。
- **#7679 rollout 默认 `detokenize=False`**：rollout 是 token-in / token-out，去掉了热路径上每个 token 都要做的 CPU 解码；有字符串 stop 条件时自动回退。

**THUDM/slime**

- #2406：score centering 支持 top-p replay。
- #2407：移除过时的 rollout 全局数据集开关，属于小的 breaking。

**NVIDIA/TransformerEngine**

- **#3282 非 eager 模式下 EP 的 prepare+dispatch 融合**｜https://github.com/NVIDIA/TransformerEngine/commit/f7273d7d806e1905afc08e19c5d5a3de07a25eb3
  - 每个专家、每个 rank 的接收数量直接从 dispatch 自身的 count scan 推导，省掉单独的一次 AllGather，整个 dispatch 可以被 CUDA Graph 捕获。
  - 由 `NVTE_EP_FUSED_PREPARE_DISPATCH` 开启，调用方需要提供静态的 recv buffer；计划进入 2.20。
  - **对我的意义**：MoE dispatch 关键路径上少一次集合通信，并且可以整图捕获。HCCL 的 dispatcher 和 aclgraph 可以照这个模式改。
- #3195（文档）：NVFP4 默认 recipe 的随机舍入只在 CC 10.0/10.3 上可用，CC 12.0 必须用 round-to-nearest。

**vllm-project/vllm**

- **#57586**：开启 `VLLM_BATCH_INVARIANT` 时，默认使用可中断的 CUDA graph（不走 torch.compile），让调好的 matmul 配置看到真实的 M。RL 训推一致性模式的开销明显下降。
- **#58169**：DCP 计算序列长度时去掉一次 CPU-GPU 同步。
- **#58051**：TritonExperts 跳过路由到非本地专家的 top-k 槽位。
- #57176 / #51800：per-token NVFP4 MoE 后端必须显式选择，并移除 Quark 的静默在线量化，都属于「不再悄悄改变数值」一类的改动。
- #56013：XPU 后端升级到 PyTorch 2.14。

**sgl-project/sglang**

- **#40779 [RL] 修复 `pause_generation` 与权重更新互相死锁**｜https://github.com/sgl-project/sglang/commit/6fe4b66f5c
  - 问题一：暂停期间 idle sleeper 仍在周期性调用 `empty_cache()`，与权重更新争用 allocator。
  - 问题二：retract 之后，waiting_queue 里的请求并不占 KV，但 `is_fully_idle()` 仍把它们算进去，导致 flush 空转，权重更新一直重试到超时。这在 512 个在飞请求时复现过。
  - **对我的意义**：标准 RL 权重同步循环中的硬挂死，用 SGLang 做 rollout 的应尽快合入。
- **#37284**：权重校验通过后释放参考快照，去掉权重同步路径上一份常驻的全模型副本。
- **#33723**：elastic EP 扩容后重新捕获 decode CUDA graph。
- **NPU 相关**：
  - #28417（NPU piecewise graph）当天合入后又被 revert（#40895），说明 SGLang 在 NPU 上的分段图捕获还不稳定；
  - #40445 把 FIA 的 KV 写入融合成一次 `npu_scatter_pa_kv_cache`；
  - #40438 对带 clamp 的 SwiGLU checkpoint 不走 fused gmm1+swiglu；
  - #35958 优化 Qwen3.5/3.6 的 NPU decode。

**vllm-project/vllm-ascend**（44 个 commit）

- **#17361 MRV2 细粒度 TP 扩展到 MLP 和 embedding**
  - 两类约束：MLP TP 与 o_proj 一样要求组内 token 数一致，因此继承 graph 模式、PCP、PD 等门控；embedding TP 在算子内部 pad 到固定容量，与 step 的 shape 无关。
  - 测试环境：A3 单机 16 芯片，PD 分离，DSV3.1 架构 W8A8 截断到 6 层，D 侧 dp8×tp1。
  - TPOT：基线 7.12 ms，三个开关全开 7.14 ms；embedding=2 时 5 条 prompt 输出逐位一致。
  - 显存：每 rank 固定多出约 0.42 GB 的 HCCL buffer；MLP=2 时每 rank 省 0.55 GiB，o_proj 每层每 rank 省 0.109 GiB。
  - **对我的意义**：这是 A3 上大 DP decode 的显存杠杆，而且给出了每个开关的收益和代价。
- **#17355 MooncakeConnector V2 支持 PCP**（+1077/−263）：去掉 `pcp_size == 1` 的限制，PCP 与 PD 分离可以组合使用，这是昇腾长上下文 serving 需要的形态。
- **#17066 MRv1 block table 只提交变更区间**：用 Triton scatter 写入并做双缓冲，默认开启。作者早期在 DSV4 DP16 decode 上的 profile 显示 H2D 从约 2.72 ms/step 降到约 0.14 ms/step；这个数字来自适配 main 之前，合入前没有在 NPU 上实跑。
- **#17336**：MLA 在 DCP 下的投机解码元数据统一成一条路径。
- **#17119**：DSpark 复用 MTP 的 cache 排除逻辑。
- **#16973**：AscendStore 为 DSV4.1 做准备，并支持投机 cache 复用。
- **#17226（文档）**：W4A4C8 量化方案，以及 Ascend 950 上 DSV4-Flash/Pro 和 GLM-5.2 的精度。昇腾推理栈多了一个新的精度档位。
- **#17154**：退出时销毁原生 HCCL 通信域，修复资源泄漏。

**NVIDIA/TensorRT-LLM**（Release 之外）

- #19477：升级到 PyTorch 2.14 / Triton 3.8。
- #18231：Ray 编排下支持 MNNVL all-reduce，这正是 RL 框架嵌入推理引擎时走的路径。
- #19361：修正精度支持矩阵，W4A8-NVFP4-FP8 MoE 不支持 SM107。

**huggingface/trl**

- #7243：DPO 改为分块计算 log-prob。
- #7248：MoE aux loss 系数改为从模型 config 读取，修复一个静默错误。
- #7331（BREAKING）：移除实验性的 GSPO-token trainer。

**Ascend/pytorch（torch_npu）**

- **!46884**：删除 `simt_default_warp_stacksize=8192`，以及自动注入的 `simt_stack_limit`，改用 Triton 后端默认的栈策略。SIMT kernel 的 occupancy 可能因此变化，升级后需要回归性能。

**MindSpeed / MindSpeed-LLM**：窗口内无新 Release，也没有新的重要合入（MindSpeed-LLM 唯一一个 commit 昨天已报，MindSpeed 只有文档和 CI 改动）。

> 窗口内无 Release：Megatron-LM（core_v0.19.2）、torchtitan（v0.3.0）、PyTorch（2.14.0）、DeepSpeed（v0.19.7）、verl（v0.9.1）、OpenRLHF（窗口内也无提交）、slime（v0.3.2）、vLLM（v0.30.0）、vllm-ascend（v0.23.0.post1）、SGLang（v0.5.20）、TE（v2.19）、TRL（v1.13.0）。

---

## Sources

- Qwen3.8-Omni：https://arxiv.org/abs/2609.25611
- 论文：https://arxiv.org/abs/2609.25442 ，https://arxiv.org/abs/2609.26355 ，https://arxiv.org/abs/2609.26621 ，https://arxiv.org/abs/2609.26708 ，https://arxiv.org/abs/2609.25560 ，https://arxiv.org/abs/2609.25624 ，https://arxiv.org/abs/2609.26219 ，https://arxiv.org/abs/2609.25809 ，https://arxiv.org/abs/2609.26300
- NVIDIA SWE-Serve：https://developer.nvidia.com/blog/how-swe-serve-exposes-the-gap-between-local-tests-and-live-serving/
- Fireworks Ember-1：https://fireworks.ai/blog/ember-1
- NCCL4Py v0.6.0：https://github.com/NVIDIA/nccl/releases/tag/nccl4py-v0.6.0
- TensorRT-LLM v1.3.0rc28：https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.3.0rc28
- 各 PR/commit 链接见正文。

## 运行备注

- **窗口与去重**：
  - 标准 24h 窗口（周四）。
  - 昨天的日报（手动试运行）覆盖到 9-23 13:27（北京时间），与本窗口有约 5.5 小时重叠。重叠部分的条目已按昨天的日报去重（例如 verl #7999、torch_npu !45086 的 Dynamo patch 移除昨天已经报过，今天不再重复）。
  - arXiv 9-23 的 listing（北京时间 9-23 08:00 公布）昨天只收了部分，今天补上了未收录的部分。9-24 的 listing 生成时还没有出来。
- **窗口外、未收录（供参考）**：
  - **阿里云栖 9-22（北京时间）**：平头哥真武 V900（算力是 M890 的 3 倍，216GB 显存，片间 1200 GB/s，支持 FP8/FP4，磐久超节点 2027 Q1 出货）；磐久 AL64 SNPO 光互联超节点（单机 64 卡，可无收敛扩到 1024 卡，51.2T 交换，近封装光）。这两项是昇腾 960 SuperPoD 和灵衢的直接竞品，昨天的日报漏收，建议补读：https://www.ithome.com/1/005/602.htm 。
  - AMD ROCm 博客（9-21/22）：Kimi-K3 on MI350X；GLM-5.2 MXFP4 在 MI355X 上的 prefill CP。
  - vLLM 博客 vllm-metal（9-22）。
- **窗口内、低相关，仅记录**：
  - NVIDIA Nemotron 3 Diarization（100M 说话人分离模型，OpenMDW 许可）；
  - Gemini 3.8 Flash TTS（博客日期 9-23，但 API changelog 记在 9-22，时间有歧义）；
  - Together《How to train your own Jev》（4B 分类模型的 SFT 教程）；
  - Anthropic 生物实验（约 950 个 agent，没有论文）。
- **时间无法确认、未收录**：
  - 量子位转载的商汤《异构混推》（9-23，具体时刻不详，可能早于 08:00）：日 token 量从 2 月的 0.46T 增长到 8 月的 4.5T，P/D 角色按负载动态切换，下一步计划按 encoder、attention、FFN、vision 拆分弹性池。
  - CUDA 13.4 Update 1 的 release notes（页面没有日期）。
- **检索失败或受限**：
  - arXiv abs 页面多次返回 429，改用 alphaXiv 读全文；arXiv export API 被 robots 拦截。
  - WebFetch 截断了 cs.LG/cs.CL 的 listing（分别只看到约 46/118 和 45/80 条），后面的部分用 alphaXiv 检索补上。
  - `huggingface.co/papers?date=2026-09-24` 返回 400。
  - HF org 按 createdAt 排序的列表会漏掉「先私有、后公开」的 repo。
  - lmsys.org、modal.com、anyscale.com 需要 JS 渲染，改查 lm-sys.github.io 的仓库；Google Cloud blog 的文章没有日期；华为新闻、昇腾社区、机器之心、知乎没有拿到带日期的列表。
  - Gitee/GitCode 上 MindSpeed 的 Release 无法确认日期，因此是「未确认」，不等于「没有」。
  - CompKV、2/3 Experts 两篇只读了摘要。
- **保存结果**：
  - GitHub：写入 `suhaibo666/tracker` main 分支 `llm-training-daily/llm-training-daily-2026-09-24.md`。
  - 本地：**失败**。`device_commit_files` 两次都返回「设备未连接到 bridge」（电脑离线），按约定不再重试，没有写入 `/Users/suhaibo/workspace/90-knowledge/llm-monitor/`。可以在电脑上从 GitHub 拉取。
