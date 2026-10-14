# LLM 技术动态日报 · 2026-10-14

> 主扫描窗口：2026-10-08 08:00 → 2026-10-14 10:34:06（北京时间）；回溯查漏从 2026-10-05 08:00 开始。生成时间：2026-10-14 10:54（北京时间）。本期为手动首跑，包含停跑期间的增量，不是单个24小时的新闻量。
>
> 历史基线：tracker/main `b8b5d36b8bd64fa4c715354e2335c4ce94ddacbc`，最新旧日报截至10-08 08:00。来源日期保留原文口径；没有时区/小时的公告不补造精确发布时间。所有性能数据均为作者或厂商报告，本次做来源核验与条件审查，未运行模型或训练性能复现。

## 今日要点

1. **先检查静默正确性问题。** Megatron 的 HybridEP 自定义 all-gather 超时可能继续读旧 routing map；混合优化器可能复用未激活参数的旧梯度；SGLang 的状态布局/HiCache 修复和 vLLM 的重载问题，都说明 loss 或 cache hit 看起来正常并不够。
2. **RL 权重更新要检查“写入了什么”和“何时可见”。** vLLM 修正最终 kernel storage 与 CUDA graph 地址，新增 token 区间权重版本信息；torchtitan 按模块同步以减少临时内存。SGLang 的更新失败返回可能发生在新权重已经加载之后，不能当作原子回滚。
3. **torchtitan 的 RL 默认冻结 MoE routing 是算法行为变化。** bias-freeze 的零学习率实验支持训推漂移诊断，但不能据此声称冻结 gate 已证明普遍改善长期 RL；恢复旧训练配置也有 optimizer-state 限制。
4. **模型配方值得看细节。** LFM2.5披露预训练/mid-training/Agent后训练链路；Youtu把结构与内容token分开蒸馏；LightOnOCR-3披露有限滞后的异步GRPO和多seed参数平均。它们都不应被简化成一个榜单分数。
5. **本期蒸馏研究不是同一个结论。** Semi-OPD研究离线学生轨迹，BranchOPD研究信息量大的分叉，RTI用奖励侧信息修正失败轨迹；要分别核对采样预算、协议修复、特权信息和成本口径。
6. **预训练与推理各有可复现切口。** ShellElastic关注一套共享权重的多宽度预训练，MSAR改变深度残差检索；DCOCO与TokenRouter分别面向投机解码和逐token跨模型路由，不能把论文倍率外推为通用serving收益。
7. **硬件数字要带负载条件。** MI455X对MI355X的“约34倍”依赖90 tok/s/user约束；Kimi-K3短上下文部署在固定KV池下改序列槽就改变吞吐；ReplaySSM的高batch收益不适用于batch1。

## 阅读路径

- **训练/RL正确性**：先读工业开源项目中的 Megatron、vLLM、SGLang、torchtitan。
- **后训练算法与数据**：读 Semi-OPD、BranchOPD、RTI、TerminalHorizon，再看 LightOnOCR/Youtu 的应用配方。
- **预训练与模型设计**：读 LFM2.5、ShellElastic、MSAR、优化器 batch-size 补录。
- **推理与硬件优化**：读 DCOCO、TokenRouter、ReplaySSM、AMD部署案例与性能口径。
- **覆盖边界**：文末列出实际检查范围与待补查项；没有把未检索到的内容判定为“没有进展”。

## 工业开源项目

**状态口径**：以下 PR 已核对官方正文、`merged_at` 与合入 SHA。除 slime v0.4.0 外，所述变更为 **main 已合入，未确认进入正式 release**。日期按 GitHub 的 UTC 合入时间；不能以 committer 时间替代合入时间。测试和性能均为作者报告，本轮未运行这些训练任务或完成完整代码审计。

### Megatron-LM

#### HybridEP 自定义 all-gather 超时后静默错路由

- 日期/状态：2026-10-09T01:42:41Z，main 已合入；8a9923b694c5c72cbdb5bd9c67e6f94c2f117092。[PR #7771](https://github.com/NVIDIA/Megatron-LM/pull/7771)。
- 机制：单 NVLink 域使用的 custom all-gather 约 20 秒超时后打印 warning 并 break，继续读取仍含上一 MoE 层内容的 routing buffer；编译、capture 或慢卡均可触发。MCore 在 DeepEP 支持参数时设置 enable_custom_allgather=False，仅 routing-map gather 改走 NCCL，token dispatch/combine 不变。
- 证据/边界：256×GB300 DeepSeek-V3 MXFP8 观察到 warning 与数值分叉重合；DeepEP 1.2.1+17cfb81、4 GPU 人为单 rank 停顿 30 秒时，custom 路径 3 个 rank 损坏，NCCL 全部正确；5 秒不触发。50-step 作业最快稳态 step 接近 10.87 秒，但不是干净 A/B，不能宣称零性能代价。
- 实践建议：搜索 HYBRID-EP ALLGATHER TIMEOUT，不能仅按警告忽略；补丁为 MCore 规避，DeepEP 根因仍需 fail loudly 的上游修正。

#### delayed wgrad 的归约尚未完成，optimizer 已读取

- 日期/状态：2026-10-08T21:04:57Z，main 已合入；441a987409b8fdd2b90b463f7999bac970109a43。[PR #7854](https://github.com/NVIDIA/Megatron-LM/pull/7854)。
- 机制：TE backward_dw 可在 autograd 最终 callback 等过 reduction stream 后再启动 MFSDP v2 归约。新增 caller_managed_grad_sync 策略，MCore 在所有 backward（包括 delayed wgrad）结束后明确调用 finish_grad_sync；默认自动模式在无 pending completion callback 时拒绝新 backward 工作。
- 证据/边界：2×RTX A6000 上最新 adapter suite 每 rank 17 passed / 5 skipped；缺失 wait 的 CUDA capture 负对照得到 unjoined work。未运行完整功能训练，Blackwell/4+ GPU 用例跳过。
- 实践建议：延迟梯度与异步通信组合要按最终消费者定义完成边界，不能将 autograd callback 结束等同于全部 gradient ready。

#### CPU/GPU 混合优化器复用旧梯度，更新本步未激活参数

- 日期/状态：2026-10-08T20:36:51Z，main 已合入；29515183a63c4637f103523a1102414ea24fe193。[PR #6871](https://github.com/NVIDIA/Megatron-LM/pull/6871)。
- 问题/修复：参数上一 step 有梯度、下一 step 无梯度时，只设置 requires_grad=False 不会让 PyTorch optimizer 跳过持久 inner parameter 的旧 grad。补丁清空 FP32 master 与 CPU-offloaded copy 的 grad，并在重新激活时重新挂回缓存 grad buffer。
- 证据：active/inactive/active 与 torch.optim.SGD 对照两个用例先失败后通过；LLaVA image→text-only→image 的 text-only step，vision class_token 原版仍变化约 0.001，修后为 0。没有新性能加速结论。
- 实践建议：多模态条件分支与稀疏激活模型使用混合 optimizer 时优先核查；零梯度张量与 grad=None 的优化器语义不同。

#### TP>1 的 MTP loss 错用全词表 logits

- 日期/状态：2026-10-10T06:15:09Z，main 已合入；b86e75c184483a54689d17b1a9eb6c5bcc999425。[PR #7682](https://github.com/NVIDIA/Megatron-LM/pull/7682)。
- 机制：调用方 runtime_gather_output=True 会让 MTP head 在每 rank 返回完整词表，但 vocab-parallel CE 按张量宽度推断本地 shard，导致 loss/grad 错误。MTP logits 不对外返回，去掉对调用方该 flag 的继承即可保持词表切分。
- 边界：PR 未给数值验证结果、测试清单未勾选，不应写“已验证收敛恢复”。
- 实践建议：RL/SFT 的 MTP 训练若为了返回 logits 开 gather，要检查辅助 loss 是否仍处于 vocab-parallel 语义。

#### MFSDP v2 optimizer 可用 DCP 保存与恢复

- 日期/状态：2026-10-11T08:48:55Z，main 已合入；a8a5977f7611e9a8aac205eabd2f9b43f786bc8f。[PR #6215](https://github.com/NVIDIA/Megatron-LM/pull/6215)。
- 增量：补全 FullyShardedOptimizer 的 sharded_state_dict/load_state_dict，使用 fsdp_dtensor，v1/v2 同一磁盘格式。空本地 shard 也发出全局 key 对应的空 placeholder，避免每 rank 的 DTensor key 集不同导致逐 key all_gather_object 死锁；step 存于 param_to_group_meta。
- 证据/边界：2×RTX 6000 Ada 的单/双 rank checkpoint 往返，模型、optimizer state、step、DTensor metadata 零容差一致。端到端训练 ckpt-resume 功能测试留待后续，不能推广到所有并行拓扑。
- 实践建议：评估 MFSDP v2 可恢复性时要测试空 shard 和改变并行布局后的元数据，而非仅比本地权重。

#### GTP_remat 的权重分片与序列/批次分区解耦

- 日期/状态：2026-10-13T10:53:17Z，main 已合入；6ae07117c01941cbd46c8e083b02462e6d4869ce。[PR #7748](https://github.com/NVIDIA/Megatron-LM/pull/7748)。
- 增量：tensor-parallel-num-sequence-shards 控制 GTP_remat 内 sequence partition，weight shards 仍独立指定。可将 weight all-gather / grad reduce-scatter 限于单域，同时 CP 跨域。例如 TP1、weight-shards64、sequence-shards64、configured CP2 在 128 GPU 上将一个序列分到 128 rank，权重仍只在64-rank组分片。
- 边界：PR 的布局示例假设每 NVLink 域64 GPU；TP>1时该 sequence partition 要求 SP、整除约束和 tp-cp-ep-dp-pp rank 顺序。非 GTP 分片的 norm/bias optimizer state 在 GTP组内仍复制。PR列出测试命令，未给实际吞吐结果。
- 实践建议：长上下文布局可按计算/通信拓扑分别设计，但先核对 sample accounting、scheduler 与 checkpoint reshard 的行为。

#### 确定性模式固定 Triton autotune 配置

- 日期/状态：2026-10-10T22:12:10Z，main 已合入；ebb93b2eda9f820f5c32f36152e84b21ae728a56。[PR #6076](https://github.com/NVIDIA/Megatron-LM/pull/6076)。
- 机制：计时选优在各 rank/各 run 可选不同 tile/reduction order，缓存未命中仍会各自计时。deterministic-mode 下按随包表→指定 block_sizes→最便宜候选确定性选型，并可跨 rank 周期校验；普通模式保留 timed autotuning。TE MoE permutation 数值不依赖选型，保留计时。
- 边界：GB300 测量中默认最便宜候选可使 MLA RoPE/mHC kernel 慢12.5×/16×；带表的测量形状总 Triton时间在1%内，未测head count在6%内。SM100 表复用 SM103、未独立实测；不能将此结论推广整步或其他GPU。
- 实践建议：严格重现实验应把 autotune policy/table 纳入环境基线。

#### DSv4 packed THD 静态 CP：功能合入但 GPU 验证待补

- 日期/状态：2026-10-09T16:18:53Z，main 已合入；6605602ea9e4fc16ecc3912dfc35fb528a23a381。[PR #7469](https://github.com/NVIDIA/Megatron-LM/pull/7469)。
- 增量：TP1、contiguous CP 下覆盖 window attention、CSA ratio4、HCA ratio128 和 MTP；处理跨rank压缩组、文档相对RoPE、边界hidden-state及梯度归还，压缩KV/indexer K的异步gather与梯度RS重叠。packed输入要求 cudnn backend。
- 边界：96组CPU custom-autograd比较与4组纯attention比较通过，外部 CUDA kernels/collectives 用替代物；作者明确未跑真实GPU单测、全训练、峰值显存与吞吐。dynamic CP与compact BF16/MXFP8 indexer为后续。
- 实践建议：可作为实现路线跟踪，不宜写成生产可用的性能进展。

### vLLM

#### MegaMoE 直接加载到 kernel layout，修复热重载失效

- 日期/状态：2026-10-13T18:52:09Z，main；c5bda658d20b8fd187cfc8049e964ab39fe2f7f3。[PR #60096](https://github.com/vllm-project/vllm/pull/60096)。
- 机制：旧loader-side参数经finalize转换后被丢弃，后续load无有效写入目标；新backend在构建时注册最终DeepGEMM布局，并逐shard原地加载，shared-expert融合也在构建时决定。
- 证据：GB200、EP、deep_gemm_mega_moe、完整GSM8K1319题/5shot/greedy；DSV4-Flash TP2+MTP2原版reload后0.948→0.008，修后0.951→0.948；DSV4.1-Flash TP4原版0.898→0.010，修后0.904→0.901。identity reload的1588个per-rank参数checksum/存储地址均不变。
- 边界：main对照来自#58053前实现；作者说明后续抽象仍保留同一问题。EPLB rebalance后的reload仍未修；下游backend接口需迁移。勿写“所有MegaMoE热更新已解决”。
- 实践建议：RL refit门禁应包含同权重重载后的逐参数checksum、地址稳定性和输出验证。

#### MLA 派生权重重载必须原地保持 CUDA graph 地址

- 日期/状态：2026-10-08T01:38:07Z，main；9dba339d07eaf16c9aa874eb042838f20c3ee0c8。[PR #60389](https://github.com/vllm-project/vllm/pull/60389)。
- 增量：AITER W_K/W_V及其scale、DCP query-replication派生权重改为注册参数与prefer_copy原地写。旧rebind让captured graph继续读取旧storage，sleep2+reload后输出乱码/静默退化。
- 证据：DeepSeek-V2-Lite-Chat、MI300X FP8 / MI355X MXFP4、TP1/2与DCP2，修后输出及地址稳定；eager原本不受影响。原有perplexity检查可能漏检，因为debug checkpoint未改变attention weights。
- 去重：10-08日报只列作下一期线索，本文补一手正文与触发边界。

#### 异步 RL 输出带 token 范围对应的权重版本

- 日期/状态：2026-10-11T03:33:30Z，main；b89852aad0854cc8df9138bc412b9a75c1fa8120。[PR #59582](https://github.com/vllm-project/vllm/pull/59582)。
- 增量：pause(keep)跨权重更新的一个completion可以返回多个半开token区间版本标签，连续同标签合并，A→B→A保留三段；仅final output/流最后chunk返回，含abort输出。Python CompletionOutput与/inference/v1/generate可用；该PR不改变OpenAI-compatible endpoints。
- 证据：H20、Qwen3-1.7B、async scheduling开/关各135检查，用清零lm_head强制输出token0校准版本切点。只添加metadata，不改变sampling；调用方需在无step in-flight的暂停期正确更新标签。正文仍留draft措辞，但API明确已合入。
- 实践建议：可为staleness分析、按段mask/replay提供可观测性，标签本身不证明策略校正有效。

#### NCCL 权重连接可拆除，保留健康推理 workers

- 日期/状态：2026-10-11T13:15:15Z，main；bd5bbce6e45e09dbd07ac3e8c394626c059078f7。[PR #61060](https://github.com/vllm-project/vllm/pull/61060)。
- 机制：仅丢PyNcclCommunicator引用不会销毁native communicator；新增shutdown_weight_transfer_engine，dense/sparse/M2N显式destroy，active update拒绝shutdown，重复inactive shutdown幂等，weights/KV不卸载。
- 证据边界：8次H200 NCCL connect/broadcast/shutdown原版增长2.84GiB/GPU、候选零增长与约486MiB/次回收，全部测于fork #119的f49e6dff710；新main-based分支只完成静态检查，fresh runtime验证待做。不可把fork性能直接说成当前合入head实测。
- 实践建议：trainer独立重启场景同时检查communicator释放与worker存活；适合与slime恢复路线并读。

#### batch-invariant 模式补齐 router 与量化线性路径

- 日期/状态：2026-10-13T19:43Z，main；[#59985](https://github.com/vllm-project/vllm/pull/59985) 68fc395554d2ff16f91457bc031334006339f77c；[#60524](https://github.com/vllm-project/vllm/pull/60524) b33cb8123707977aed605937a772bf0bc5d2a92d。
- 机制：GateLinear专用tier按token数选择GEMM与reduction顺序，绕过batch-invariant路径，微小router差异可改变expert选择；开启flag时禁用专用tier。CompressedTensorsWNA16的W4A16/W8A16绕过Marlin动态tiling，改解包+batch-invariant matmul。
- 证据/边界：GB300上GLM-4.5-Air、MiniMax-M2、ERNIE-4.5-21B-A3B各22/22 invariance tests通过；GateLinear单call成本在作者形状中1.1–2.7×，非整模型倍率。#60524正文报告A10/B300模型suite通过；本文未逐项核读其长吞吐日志，不引用具体损失。flag关闭行为不变。
- 实践建议：训推一致性评测不要将设置环境变量等同于全kernel已具备此保证，升级后应重跑真实模型与量化路径。

#### Mamba2 投机解码的 prefix checkpoint 保存了过旧 conv 输入

- 日期/状态：2026-10-13T10:09:18Z，main；c7a305ec10f3ec06c2784f1258162483c5fc8d73。[PR #60797](https://github.com/vllm-project/vllm/pull/60797)。
- 机制：物理conv row包含kernel_size−1+num_spec slots，旧checkpoint把整个宽度当有效历史，restore后前几个读取槽比应有输入旧num_spec token。改只保存conv_kernel_size−1有效输入；layout/KV容量不变。
- 证据：Mamba2 hybrid+MTP、prefix cache align、shared-prefix-checkpoint、FP8 KV，长记录复制任务有cache hit时错误，首发及关闭spec/prefix-cache均正确；含NIXL PD在内的五种配置修后0次错误。具体模型和GPU未在正文给出，不能概括为通用准确率提高。只覆盖Mamba2，FlashKDA fallback仍可能用完整conv宽度。
- 实践建议：混合模型cache回归要测试恢复点附近token语义与投机槽位，不只验证传输byte数。

### verl

#### MI355X DeepSeek-V4-Flash 的三模式 GRPO 配方

- 日期/状态：2026-10-09T03:43:27Z，main 已合入；616f063ed465ed7595aa2cd98f82299bb8cf8a5c。[PR #8146](https://github.com/verl-project/verl/pull/8146)。
- 增量：TRAIN_MODE=qat/fp8/bf16，QAT 为 BF16训练+FP8 fake quant+STE，FP8模式为 TE blockwise；对齐 rollout 的float activation scale与KV/indexer舍入，避免照搬NVIDIA上的power-of-two激活量化策略。
- 验证：4节点×8 MI355X、ROCm7.2.3、DAPO-Math GRPO/GSPO、300 step；QAT train/rollout KL约0.0022→0.0029，FP8约0.0022→0.0039，无文中所述NaN/hang；AIME2024 QAT/FP8 0.60→0.76，BF16 0.59→峰值0.79。step约490/420/330秒，weight sync约85秒，因此低精度训练没有更快的端到端结论。
- 关键边界：Megatron与Bridge补丁在作者fork、TE wheel与libhsa-runtime为作者发布asset；配方还带vLLM0.26的构建期补丁。是合入verl的可复现集成配方，不是这些依赖均已原生上游支持。
- 实践建议：低精度训推一致性要逐后端对齐量化网格；评估成本必须含refit。

#### TPU SFT / GRPO 接入采用硬件插件边界

- 日期/状态：[SFT #8098](https://github.com/verl-project/verl/pull/8098) 2026-10-09T09:25:46Z，e78c6bcde5adf18901d1d1d1d3a35a7e03ba33da；[GRPO #8155](https://github.com/verl-project/verl/pull/8155) 2026-10-13T03:33:33Z，f95be0226a65d3f335901ee6703f974fe82f840d；均main。
- 增量：TorchTitan训练、vLLM rollout、verl-hardware-plugin。TPU runtime workaround留在插件，verl核心保留device-agnostic聚合；SFT支持Ray和torchrun，用256桶padding。GRPO脚本默认Qwen3-4B-Base/GSM8K、128 prompts×16 responses、2048 output tokens、250 steps。
- 边界：以上是配方配置，PR正文未给完整250-step训练曲线/吞吐，不可写成TPU性能验证。需既有Ray TPU集群、插件、模型和数据。
- 实践建议：对其他非CUDA后端，插件承载设备特殊逻辑的集成方式可借鉴。

#### chunked vocab-parallel entropy 的反向原地修改失效

- 日期/状态：2026-10-12T01:48:04Z，main；3f3594e95021399320781c4148376764ca3b179e。[PR #8195](https://github.com/verl-project/verl/pull/8195)。
- 机制：logits各chunk是同一tensor的view，原地entropy backward改共享version counter，让后续chunk保存的logits失效。chunk路径改out-of-place backward，只增加一个chunk大小临时量；未chunk路径保持原内存行为。
- 证据：Nemotron-H30B-A3B、256K context、bypass+calculate_entropy=True首个update触发；4 GPU/TP2测试原版所有rank失败，补丁通过chunked/unchunked梯度对照。entropy系数为0也不保证不backprop，因为0×entropy仍在图里。
- 实践建议：长上下文entropy分块不是只需验证forward；应检查view与in-place backward交互。

#### 多轮会话固定到 vLLM replica 内同一 DP rank

- 日期/状态：2026-10-10T03:39:25Z，main；9058412fb308e9f2e31a794ad1d18d97d4286485。[PR #8179](https://github.com/verl-project/verl/pull/8179)。
- 机制：原每turn用新uuid，单个DP>1 replica内仍被负载均衡到不同engine，丢prefix locality；现在传session_id并按CRC32%本server DP size固定，PD两腿各按自己的DP size映射。
- 证据：Nemotron-H30B-A3B、GB200、每节点DP4/EP4 vLLM、SWE-bench Pro多轮replay，hit rate0.81→0.86、长session快10–15%；完整RL运行hit0.93是另一口径，不能当同组A/B。hash pinning忽略即时负载。
- 实践建议：agent RL应区分replica-level与replica内部DP-level两层stickiness，观察热点倾斜。

#### HF/MCore checkpoint 转换迁移 Megatron-Bridge

- 日期/状态：[#8094](https://github.com/verl-project/verl/pull/8094) 2026-10-09T06:52:27Z，05093df562b90f659385ec0a313d03fc22da2ba5；[#8166](https://github.com/verl-project/verl/pull/8166) 2026-10-09T10:03:57Z，711b34924b2b33ef7db466387244e0981e45c810；均main，前者BREAKING。
- 增量：用Bridge provider/weight mapping完成HF↔MCore与export，保留HF artifacts、支持v2训练checkpoint布局；补TP/ETP转换。
- 证据/边界：四个Qwen文本/VL/3.5模型的TP1/PP4→TP2/PP2转换、哈希和forward一致；#8166另在8GPU上Qwen3-30B-A3B TP2/PP2/ETP2/EP2→TP4/PP1/ETP2/EP4验证。仅model weights和HF forward，不覆盖optimizer、backward、完整训练，VPP关闭。
- 实践建议：自定义转换脚本/旧MCore API依赖升级前需适配，不能只验证“文件写出来”。

### SGLang

#### GDN CuTe DSL decode 的状态布局转置错误

- 日期/状态：2026-10-11T04:43:17Z，main；65e6bb6be7cc76c7fb0cc9b9b86c4bf6231f2e6f。[PR #43332](https://github.com/sgl-project/sglang/pull/43332)。
- 机制：state pool已改[N,HV,V,K]，CuTe decode仍按[N,HV,K,V]读；修为转置view与K连续shared tile，计算循环保持。恢复此前跳过的precision测试。
- 证据：单H200、Qwen3.5-4B BF16、TP1、无MTP、GSM8K200题5shot，CuTe 0.010→0.875，Triton对照0.870；只应表述为该配置正确性修复。
- 实践建议：state layout版本变更应覆盖每个可选backend的读写，不能只测默认Triton。

#### HiCache/Mamba 三类静默状态破坏

- 日期/状态：[#38217](https://github.com/sgl-project/sglang/pull/38217) 10-11T01:50:13Z，156d54e56cf29d6871912ee1efeec6db2d5255c0；[#43567](https://github.com/sgl-project/sglang/pull/43567) 10-11T13:12:52Z，0923bcd258b41aa9a5a6b4ee10cfa5229f71cfc0；[#43594](https://github.com/sgl-project/sglang/pull/43594) 10-12T07:41:00Z，a70a83bcd237b7de6136574f47515a58bf9efd83；均main，年份2026。
- #38217：HybridLinearKVPool接管同步后读取MLA KV漏掉outer wait；B300两层FP8 MLA、16K token异步load-back立即读取20/20错误→0/20。
- #43567：lazy checkpoint无pending slot时原地覆盖keep slot，却仍保留旧prefix长度，overlap prefill可把新内容以旧key发布。GB300/Qwen3.5-0.8B、1请求/4Mamba slots、overlap+decode graph的full-vocab KL 0.6494→0；spec verify修复路径未由E2E覆盖。
- #43594：MLA KV可共享TP key，但Mamba是分片state；加入TP rank/size后缀，旧无rank对象不复用。GLM-5.3-Flash restart后四rank、两个4096-token L3 restore保住136页；仅验证buffered POSIX I/O。
- 实践建议：缓存命中统计不能代替状态一致性；审计key ownership、异步完成与失效/重新发布顺序。

#### HiCache 还原所有 MTP draft 的 KV 与 recurrent state

- 日期/状态：2026-10-12T07:38:00Z，main；5ca0695aa69f891bf98a57e7dd83cf75f1bc24c6。[PR #40223](https://github.com/sgl-project/sglang/pull/40223)。
- 增量：Inkling多MTP head的独立draft convolution/temporal state全部进入恢复；full KV、共享target SWA索引的draft、按full-token ID索引的draft分别进入正确pool。恢复需在target最后一层完成event之前结束，并计入固定host预算。
- 边界：GB300 64个byte-exact GPU transfer cases、4个synthetic Inkling E2E；TP1、BF16、FA4、CUDA graph关，未验证trained-model质量、NPU实机或外部L3。没有吞吐结论。
- 实践建议：投机head增加后，“target输出看起来正常”不足以证明draft缓存恢复正确，应逐head校验state和接受率。

#### 权重更新后的 flush 被拒绝时返回失败，且权重已经加载

- 日期/状态：2026-10-12T04:57:58Z，main；2240049194a0fd803e1fab67f28eaa01ad7b588e。[PR #43758](https://github.com/sgl-project/sglang/pull/43758)。
- 机制：pause_generation(in_place)仍有运行/排队请求，flush正常被拒，旧assert杀scheduler/server。现在返回success=False/HTTP400，提示先retract或完成请求再flush。
- 关键边界：这是部分成功：新权重仍已加载且version已记录；不是原子回滚。Qwen3-0.6B单在途请求复现修后continue_generation和health存活。
- 实践建议：RL编排收到失败不能直接假定旧policy仍生效；必须区分“weight写入完成”和“cache一致性完成”。

#### GLM-5.2/5.3 DSA 的 DCP + MTP

- 日期/状态：2026-10-13T18:35:36Z，main；8bf145d119033dfd37c54c19b794b5e60b391fe3。[PR #43735](https://github.com/sgl-project/sglang/pull/43735)。
- 机制：target KV按位置条带分片，index-K和MTP draft pool按widened slot复制；全局top-k后按rank过滤/压紧，sparse attention返回LSE再合并。内存核算显式承担index/draft复制成本。
- 边界：launch拒绝prefill CP、HiSparse、HiCache、PD、tree drafting组合；只支持当前TRTLLM路径等约束。正文还保留FlashInfer0.7.0.post1在B300/64 heads、短KV、≥1024 rows/call的kernel问题，不应写成所有DCP8场景可用。
- 证据：8×B300、GLM-5.3 FP8，GSM8K1319题TP8 0.912、DCP2+MTP0.919、DCP8+MTP0.919；AIME26长推理有采样/截断因素。建议主要报道新增能力和限制，不用跨配置数字宣传DCP更快。

#### hybrid SWA 的 P/D 可用不同 attention TP

- 日期/状态：2026-10-13T06:21:55Z，main；3eebf987ed622395f8fae3ed99ec6a1ed1ffb83e。[PR #40909](https://github.com/sgl-project/sglang/pull/40909)。
- 增量：SWA state按token与KV-head ownership切片传输，NIXL/Mooncake保留各自完成语义；只支持PP1和分离的逐层K/V buffers，非MLA不等TP+MXFP8在首次handshake明确拒绝。
- 证据：GPT-OSS-120B、两节点各4GB300、TP1↔TP4、两backend，cold/warm/window边界/并发请求功能验证；Mooncake使用staging、TCP。冷请求比较精确；warm/concurrent存在prefill logits已不同的对照，不能宣称普遍跨TP输出等价。无RDMA性能测量。
- 实践建议：可让prefill/decode按负载分别选TP，但须先核对cache layout与head切分可表达性。

#### ROCm HiCache opt-in GTT 分配避免主存迁移停顿

- 日期/状态：2026-10-12T21:05:52Z，main；6966c5d027163f1d2005e5fec8e0ed40cf95213e。[PR #42553](https://github.com/sgl-project/sglang/pull/42553)。
- 机制：HSA_USERPTR_FOR_PAGED_MEM=0配合默认pinned allocator时，改用精确字节数hipHostMalloc，不再仅对mmap页hipHostRegister；最后view释放才free，分配失败不默默退回USERPTR。
- 证据：MI355X人工页迁移实验legacy每次最大约64秒stall、GTT约0.8ms/无eviction，是诱导GPU workload实验，不是token延迟。真实GLM-5.2 MXFP4 TP4/EP4、HiCache180GB/rank、AgentX同机A/B总吞吐+5.11%、P90 interactivity+3.69%，但output吞吐−4.98%；不能只挑正数字。
- 边界：opt-in、受driver GTT容量限制；custom/unpinned allocator不变。

### slime

#### v0.4.0 正式发布：持久化分布式 rollout 与训练恢复

- 日期/状态：2026-10-09T03:04:12Z，正式 release。[官方 release](https://github.com/THUDM/slime/releases/tag/v0.4.0)。版本号更新提交 0b0c277d5b4cc66efc3b6db269e2e5184e8f3b1e；未另外解析 release tag 的对象 SHA。
- 增量：同步与 fully asynchronous 统一 train.py 生命周期，移除旧 train_async.py。启用 straw 后每个有 CPU 资源的 Ray 节点一个 rollout worker；大数据通过共享文件系统引用传递，Ray 继续承载控制消息。prompt、partial rollout、completed sample、转换后的训练 batch 持久化，并与 checkpoint 协调恢复/回滚。Megatron 失败后可保留健康 SGLang 引擎与 router，重提 trainer 并复用已接受样本。
- 边界：当前支持的共享文件系统目标是 JuiceFS，其他网络文件系统需部署验证；PipelineRL 的 flush-cache-interval 要求训练/rollout GPU 分离。release 没有给出规模吞吐验证，不应把开头引用的 OpenAI 1 万 agent 规模说成 slime 已验证规模。
- 去重：score centering #2405/#2406、straw、#2443 CP 梯度修复、#2444 trainer 重启此前已报道或早已合入；本次新信息是正式 v0.4.0 发布及打包边界，不应将所有功能称为本周新发明。
- 实践建议（推断）：长周期异步 agent RL 可以先验证 checkpoint 与持久队列的共同提交边界，再考虑扩大并发。

#### torch 2.13 + offload-train 首次 optimizer step 段错误

- 日期/状态：2026-10-10T07:54:02Z，main 已合入；642dcfd1e4d7ef541a756fd58d056f50c8c5d5e0。[PR #2453](https://github.com/THUDM/slime/pull/2453)。
- 问题/机制：Megatron 模块 import 时保存了原始 collective 函数，后续 monkey_patch_torch_dist 只替换 torch.distributed，旧绑定仍把 ReloadableProcessGroup 交给 PyProcessGroup。torch 2.13 更名的 collective 不在 wrapper override 中，首次 optimizer step 在 wait 中 segfault。补丁重新绑定已导入 megatron 模块中指向原函数的属性。
- 证据：1×H200 trainer + 1 rollout engine、Qwen2.5-0.5B、critic、offload-train、disk weight transport、torch 2.13.0+cu130；原版 3/3 次首步失败，补丁后完成 3 step 与 4 次权重更新。TP>1 未由该运行覆盖。
- 实践建议：更新 PyTorch 并启用训练卸载的 slime 环境应优先检查此修复；“运行时 monkey patch”不能覆盖 import-time 别名是可迁移的排障结论。

#### top-p score-centering 对齐并加入真实 agent E2E

- 日期/状态：2026-10-10T14:32:09Z，main 已合入，晚于 v0.4.0；58b0878679de339ec7844b45d884a87955a134be。[PR #2457](https://github.com/THUDM/slime/pull/2457)。
- 增量：跨 agent turn 保留原 token 与 sampler replay metadata；reward 按唯一 rollout 而非 fork 数量归一；top-p SC correction 的中心化与已有非 top-p 约定对齐，weighted score gradient 不变。
- 验证：134 SC、176 agent/group/replay/transport 测试；真实 Qwen GDN CP4 梯度与 CP1 在 BF16 容差内。单机 8×H100、SC、top_p=0.95、top_k=-1 的 E2E 457 秒完成、token/replay 审计精确、8 rank 均实际更新。仅一次配置；dense model 未启用 R3。
- 实践建议：agent fork/replay 数据一致性与奖励归一应该同时审计，不能仅看最终 reward。

### torchtitan

#### DeepEP + activation checkpointing 修复的后续：显式确定性或禁止危险配置

- 日期/状态：2026-10-08T22:54:45Z，main；70ad4c997248aba994d4089d9fb19cdd36ea749d。[PR #5115](https://github.com/pytorch/torchtitan/pull/5115)。
- 更新关系：10-08日报已报道#5123同类错误；本次应标【更新】，补充#5115的新配置检查与取舍，不再当首次发现。
- 机制：recompute dispatch的receive row顺序B可能不同于forward A，但原autograd ctx保留A布局；loss正常而grad错。新增dispatcher deterministic选项和启动检查：FullAC、自定义SAC没保存dispatch+combine、RegionAC没保存ep_communication且非deterministic时拒绝。
- 证据：GH200单节点EP4、D6144/F1792、96 experts/rank、top8，对无checkpoint基线，危险SAC输入grad相对L2约0.49、router/expert约1.4；deterministic排序让dispatch10.3→14.3ms、block fwd+bwd约+3%；保存ops替代约保留3GiB/layer。新增GPU回归当时只在EP1执行，PR明确EP≥2负对照仍待跑；不要混淆诊断实验与新测试覆盖。
- 实践建议：选择确定性sort的时间成本或保存communication的内存成本，不能用旧loss曲线证明错误配置有效。

#### RL 默认冻结 MoE routing：重要算法行为改变

- 日期/状态：2026-10-13T03:10:35Z，main；4a4176e9720513c7ebfb9b9bcccbeeedfcee217b。[PR #5271](https://github.com/pytorch/torchtitan/pull/5271)。
- 增量：build_rl_model_configs默认FP32 lm_head、freeze_expert_bias、freeze router gate、aux_loss=None；两flag可opt-out，预训练/SFT默认不变。16 MoE recipe冻结routing，其中13去aux loss。
- 证据：早期main、H100、真实Qwen3-30B-A3B前8层，lr0、30轮generate/score/update/sync，bias更新时mean abs logprob差0.053→0.157；冻结时0.054→0.053。说明非optimizer的bias更新本身能放大训推差异，不是完整RL收益实验。
- 边界：只有随机debug模型2×H100三步smoke，未跑真实checkpoint长RL和FSDP+TP+EP；冻结gate未有ablation。冻结保存的checkpoint不能以freeze=False恢复，因为缺gate optimizer state。正文测bias-frozen用更早revision。
- 实践建议：升级属于策略/可训练参数集合变更，需与原训练目标一起回归，不能宣传为无代价通用稳定方案。

#### FSDP generator 按模块同步，避免整个模型双份驻留

- 日期/状态：2026-10-11T04:25:15Z，main；39513f1cb6a9035f14950e710ac01fef193c9cca。[PR #5268](https://github.com/pytorch/torchtitan/pull/5268)。
- 机制：generator forward/CUDA graph保留unsharded storage，load前旧路径一次性恢复全部sharded buffer，FSDP degree1等于双份模型。新路径按共同fully_shard的module group分配→load→unshard→free；tied embedding组需要跨load保留。
- 证据：H10095GiB、BF16、gpu_memory_limit0.9，Qwen3.5-35B-A3B TP1额外sync内存64.6GiB（旧state dict体积口径）→实测1.57GiB；9B TP1旧路径观察OOM，新峰值1.89GiB，两轮sync约0.35–0.37秒；greedy同权重一致、扰动norm变化、撤销恢复。
- 边界：64.6GiB旧值是state dict size，不是旧路径成功运行所得峰值；109-step DAPO运行最终有不相关host OOM。KV容量无需为整模型临时副本缩减。
- 实践建议：跨train/generator的dtype转换、staging与load生命周期应共同计峰值；另#5107把trainer cast移入完整pinned CPU mirror，转移了内存成本并非消失。

#### 多轮 rollout 用 delta prompt + NumPy 降低主控CPU开销

- 日期/状态：[#5273](https://github.com/pytorch/torchtitan/pull/5273) 2026-10-12T03:37:22Z，c918c0898f7ee39c069bc927c3e6292af164f294；[#5274](https://github.com/pytorch/torchtitan/pull/5274) 03:38:24Z，c6377370b217b22cc38c3720d806f0ea950b683d；main。
- 机制：turn prompt以相对上turn完整token的最长公共前缀+delta存储，避免完整对话反复复制；per-token Python list改NumPy arrays，拼接后torch.from_numpy，减少pickle/GC/pack开销。render历史改写仍由最长前缀处理。
- 证据：合成16-rollout、120-ply chess group token字段pickle236→16.9MB、unpickle1.83→0.11秒；后续NumPy层在合成16组60-ply pack6.88→0.28秒。78 batch×3 packing、658 branch samples逐bit一致，另618 rollout/4000随机message tree验证语义。
- 边界：这些为CPU合成负载，不是生产RL端到端提速；custom RolloutTurn构造须适配delta与NumPy字段，记录格式改变。
- 实践建议：多轮agent规模上升时，主控event loop、GC、序列化可以先于GPU成为瓶颈。

#### GraphTrainer 的 SimpleFSDP 去掉 TP 梯度中途 BF16 舍入

- 日期/状态：2026-10-13T04:54:23Z，main；c29918394cc4140e12e542ea2d2b58f92dc9a3c1。[PR #5171](https://github.com/pytorch/torchtitan/pull/5171)。
- 机制：PyTorch nightly改变FSDP2 unsharded grad默认dtype后，eager在reduce_dtype=FP32下TP all-reduce→RS；SimpleFSDP因autograd intermediate BF16把TP和router FP32 wgrad中途舍入。用一个custom autograd node声明output grad dtype，并在同节点完成TP及DP reduction，all-gather仍BF16。
- 证据：8×H100、torch2.16.0a0+gitbe3a22f、20步，Llama3/Qwen3 dense与Qwen3 MoE等5种numerics测试恢复bitwise，移除xfail。
- 边界：Dynamo尚不支持set_output_grad_dtype，torch.compile路径跳过该调用；GraphTrainer用make_fx不受影响。不能泛化所有compile mixed-precision路径已修。
- 实践建议：追nightly时应把梯度dtype默认值变化列为数值baseline变更。

#### DistMuon 通信改 BF16，同时保持更新逐位一致

- 日期/状态：2026-10-13T17:43:27Z，main；210fc60cb2b88bf21f41ebc65a75a05f254ba99f。[PR #5279](https://github.com/pytorch/torchtitan/pull/5279)。
- 机制：Newton–Schulz开始就转BF16，因此重分布input/direction可用BF16传；BlockShard不同LR ratio缩放延到接收端storage dtype做，避免新增一次BF16舍入。exchange buffers减半。
- 证据：Kimi K2.5 debug、Moonlight27层、KimiK3 debug的40步parameter/momentum digest及loss/gradnorm与基线相同；8×A100 K2.5三层EP8最慢rank251.8→245.9ms，8×3090 PCIe例2689→1195ms。
- 边界：性能为小模型/指定互联诊断，不能推广完整大模型；价值是传输dtype优化的数值等价条件。

## 模型发布、技术报告与多模态工件

### LFM2.5-24B-A2B：扩充预训练，并在真实 Agent harness 内做后训练

**日期/版本**：Liquid AI，2026-10-13；模型卡基线 `1769516ac588c4c558089c94f27d96b0dc67b183`。来源：[官方博客](https://www.liquid.ai/blog/lfm2-5-24b-a2b)、[固定模型卡](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/blob/1769516ac588c4c558089c94f27d96b0dc67b183/README.md)。

- **增量**：保留短卷积/GQA 混合骨干和 top-4 MoE，预训练从 17T 扩到 33T token；mid-training 为 2T@32K 加 300B@131K。后训练串联 SFT、教师专化、非 Agent/Agent MOPD 与 Agentic RL，最后进入 OpenClaw、Hermes、Pi、OpenCode 等 harness。
- **证据与边界**：名称中的 A2B 是近似命名，卡中激活参数为 2.45B。LiveClaw 自报 48.09，对照 Qwen3.6-35B-A3B 为 38.19；但 SWE-bench Verified/OpenCode 为 51.0 对 66.2，不能称编码全面更强。21K output tok/s 是单 B300、SGLang 0.5.22、BF16、1024 输入/最多256输出、高并发总吞吐。
- **建议：深入阅读训练配方并试本地 Agent**。RL 算法与阶段消融未充分披露；reasoning-only、专用 `lfm1.0` 许可证。作者报告过通过 git 历史/外部平台找答案的 reward hacking，采用时应同时复核任务隔离和实际成功率。

### Youtu-Parsing-Omni：将多模态解析统一到 schema，并区分结构与内容 token 蒸馏

**日期/版本**：腾讯，10-09 权重工件提交；核读版本 `05d44d70575744ce1009897bcba42c5986c3b7c3`。来源：[模型卡 Method/Results/Serving](https://huggingface.co/tencent/Youtu-Parsing-Omni/blob/05d44d70575744ce1009897bcba42c5986c3b7c3/README.md)。

- **增量**：5B 模型用共享编码器处理文档、图像、音视频，输出 OmniSchema JSON。SFT、schema-routed RLVR 后进行 OSAD：学生采样先过格式 gate，冻结副本额外看 reference parse；内容 token 用 JS、结构 token 用高温 forward KL，另加全 token JS；轮末合并 LoRA 并更新教师。
- **证据与边界**：官方卡中 OmniDocBench v1.6 为 96.96，TeleOCR 为 96.91；OmniParsingBench 平均 75.08，低于 Gemini-3-Pro 77.44。发布配置为 BF16、greedy、指定模板，服务示例 `max-model-len=65536`、`max-num-seqs=1`，不能将 config 最大位置数当作已验证的 1M 高并发服务能力。
- **建议：考虑结构化数据生产复现**。本轮依据模型卡，未解析 PDF 或复核实现；先测 schema 合法率与内容准确率，再测并发。许可证为 `youtu-parsing`。

### LightOnOCR-3：OCR 质量、页吞吐与异步 GRPO 配方一起公开

**日期/版本**：LightOn，2026-10-08；发布 0.8B/1B/4B 命名的三个规格。来源：[作者完整技术博客 §2–3](https://huggingface.co/blog/lightonai/lightonocr-3)、[4B 固定模型卡](https://huggingface.co/lightonai/LightOnOCR-3-4B/blob/b71010f095dc3735de7ee6b9969bff8f8f50b39a/README.md)。

- **增量**：SFT 使用 Muon、packing 和图像增强；GRPO group=8，verl 异步 one-step-off-policy 将 rollout/learner 放在不同 GPU 池，每批生成前同步。每 seed 抽 600 页、不同 seed 重叠约14–15%，最后做参数平均；框定位奖励同时惩罚多框和漏框。
- **证据与边界**：单 H100、vLLM 0.30.0、相同512页测试，同分辨率下4B比同骨干 Chandra-OCR-2 多服务14%页/s，主要来自输出工作量减少，非单 token 计算更快。最佳质量的400 DPI/5MP配置与1540px高吞吐配置不能混用；旧 LightOnOCR-2 分数的页眉页脚口径也不同。
- **建议：参考完整实验流程**。分别记录页级准确率、图像/输出 token 与页吞吐；模型只支持空 prompt 或 `grounding`，任意问答并非其训练范围。尚无完整消融证明收益分别来自 Muon、异步训练或权重平均。

### Gemini 4 Argon：公开可用性及十月模型卡更新

**日期/状态**：Google，2026-10-13 公告；9-30 初次发布已在旧日报收录，本次只记录可用性和新披露。来源：[公告](https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-gemini-4-argon/)、[October 2026 模型卡](https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-4-Argon-Model-Card.pdf)。

- 模型卡描述 sparse MoE、原生文本/视觉/音频，训练使用 TPU、JAX 与 ML Pathways；**输入与输出共享最多1M token预算**。参数量、训练 FLOPs、优化器和精度未披露。
- **建议：继续观察并按实际任务复测 API**。这是公开可用性更新，不是新的模型代际首发；卡中仍提示幻觉、慢响应和 timeout。本轮未核 continuation API 细则，不推测协议行为。

### Qwen-Image-2.1-Turbo：低步数 checkpoint 与采样配置必须一起使用

**日期/版本**：Qwen，10-09 创建工件；核读 `fc13ba1986fcb905854dc8de2f52eee395f49b5b`。来源：[固定模型卡](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/fc13ba1986fcb905854dc8de2f52eee395f49b5b/README.md)。

- 7B 图像生成/编辑骨干，默认 CFG=1、8步，并复用文本与参考图像 prefix KV。采样 sigmas 随 checkpoint 保存，仅修改 `num_inference_steps` 不会覆盖它，需要显式传入 sigmas；依赖支持该配置的 Diffusers PR #14950，其他 schedule 未评测。
- **建议：小规模质量/时延对照**。官方卡未披露训练方法或可比延迟数据，因此只称加速 checkpoint，不推断某种蒸馏算法，也不把8步直接换算成速度倍数。许可证为 `qwen-research`。

### Mistral Arabic 工件：流式 ASR 与方言数据质控

**日期/版本**：10-08 创建工件，窗口内更新；本轮未确认精确公告日。来源：[Voxtral 固定模型卡](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-Arabic/blob/7c60cf2acb1ceb823efe937ddbc88e5e7b6e7dc2/README.md)、[LIDstral 固定模型卡](https://huggingface.co/mistralai/LIDstral-Arabic/blob/077451f0a81211d2d9fe5306f84eb1c1ac7d3287/README.md)。

- Voxtral 是已有实时模型的阿拉伯方言/code-switching 微调，官方七组阿拉伯评测平均 CER 8.82%，使用 temperature=0、MTNRA normalization 和480ms**配置转录延迟**，后者不是实测端到端延迟。
- LIDstral 是 fastText→小 MLP→分类校准的51类语言/方言分类器，可用于预训练数据分桶；它始终从已知类别中输出，未知语言可能误分。两者为 Apache 2.0。
- **建议：按数据与语音场景观察**。先统一 normalization 和流式延迟再比 ASR；分类器需另测 OOD/混合语言，不能当可靠的自动拒识器。

### 【补录】EmbeddingGemma 2：多模态检索的维度与精度约束

**日期/版本**：Google，2026-10-06；旧日报仅列待回溯线索。来源：[官方模型卡](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2)、[固定 HF 版本](https://huggingface.co/google/embeddinggemma-2/blob/914f7f89142e33e77833254d9c9b90c3cef7303b/README.md)。

- 740M，文本/视觉/音频共享8,192-token上下文，输出768维，可截断至512/256/128维并重新归一化。MMEB v2从768维59.01降至256维56.24、128维45.65；最大压缩不是无损收益。
- 官方明确要求 BF16 或 FP32，**不能使用 FP16**，否则可能出现 NaN 或静默退化。query/document必须使用相同维度与推荐 task prefix。
- **建议：考虑多模态 RAG 对照**。把256维作为待测候选，按自己的检索分布确定存储/质量折中，不直接套用文本检索经验。

### 【补录】Liquid open d1：单次前向输出决策分布

**日期/版本**：2026-10-07；`d1-3B@051bcc464b01b9f92942b364d9586b0ef5912432`。来源：[作者发布文](https://huggingface.co/blog/LiquidAI/open-d1)、[3B模型卡](https://huggingface.co/LiquidAI/d1-3B/blob/051bcc464b01b9f92942b364d9586b0ef5912432/README.md)、[600M模型卡](https://huggingface.co/LiquidAI/d1-omni-600M/blob/02b55d7076f15129e59ab3f94783f32c4b088674/README.md)。

- 输入状态和具名问题，一次前向返回概率/类别或等级期望，不生成文本再解析。3B版官方“8ms”来自 RTX4090、BF16、warm、CUDA graph、单问题20次中位数；3.4K-token状态为102ms，不能概括成所有请求8ms。
- **建议：作为 router/judge 候选复测**。600M为experimental，图像与音频不能同请求，有图时state/question文本限896 token；音频≤30秒，训练集中于英语speaker-assistant交互。其推荐FP16/FP32，与EmbeddingGemma 2不同，dtype应按模型选择。作者跑分不等于正式榜单提交，也不能替代自有数据校准。

## 预训练与模型架构

### ShellElastic：一次预训练同时产出多个宽度的模型

**日期/状态**：2026-10-10，arXiv **2610.15124v1**，研究原型。来源：[论文 §3、§4.1–4.2、Table 2](https://arxiv.org/html/2610.15124v1)。

- **增量**：把隐藏维度分为 dense core 和沿宽度依次追加的 shells，配合块下三角矩阵、组内归一化与对齐的注意力头边界，使窄模型的隐藏状态精确包含在宽模型前缀里；一次完整主干前向即可计算多个子模型的损失。配套 μ-Elastic 处理共享权重的更新尺度与超参数迁移。
- **证据**：32 层模型，在 119M–285M、375M–906M、675M–1.69B 三组规模比较。最大组五档子模型的平均 5-shot 准确率为 44.0%，MatFormer 为 43.4%，分别训练五个模型为 44.8%。最后一组只做一次运行；“五分之一优化步数”不能直接理解为五倍墙钟加速。
- **建议：深入阅读**。适合评估“一套权重服务多个算力预算”的预训练路线；尚不能外推到数百 B MoE。工程采用前应测量块结构算子的实际效率及多 readout 开销，本轮未执行代码复现。

### MSAR：沿网络深度做多路注意力检索

**日期/状态**：2026-10-09，arXiv **2610.13630v1**。来源：[论文 §3、§4、Table 1 与附录 A](https://arxiv.org/html/2610.13630v1)。

- **增量**：在 AttnRes 基础上，用不同尺度的因果卷积形成多路上下文视角，检索历史层表示后再轻量混合；Block MSAR 保存单路块摘要，按需展开多路，减少持久化状态和通信负担。
- **证据**：18B/28B 总参数、约 1.7B/2.7B 激活参数的 MoE，序列长度 8,192。28B 的 12 项评测均分从 AttnRes 的 49.4 提升至 52.4；相对 Vanilla 的额外激活参数和训练 FLOPs 约 0.3% 和 0.5%。论文的 1.58× 是四个 scaling 尺度平均、相对 Vanilla 的**同等 loss 计算量**收益，不是相对 AttnRes 或实测集群吞吐加速。
- **建议：考虑小规模复现**。先比较收敛与实际通信/显存；本轮未核得足够的训练 token、硬件与墙钟细节，不能据 FLOPs 直接估算生产收益。

### 【补录】优化器排名会随 batch size 改变

**日期/状态**：2026-10-06，arXiv **2610.08975v1**；处于本次回溯窗口，既有归档未收录。来源：[论文](https://arxiv.org/abs/2610.08975v1)、[作者项目页的实验协议与排名比较](https://batch-size-scaling.com/)。

- **增量**：研究把“选择优化器”和“扩大 batch 后调学习率”放到同一个比较框架中；即使针对每个 batch 重新调参，优化器的相对排名仍可能反转。
- **证据边界**：作者在固定 token 预算下比较 Adam、Lion、Muon、SOAP、Shampoo，batch 从 128K 到 2M token/update；HyperBall 约束下，Muon 在 128K 优于 Shampoo，而 2M 时排序反转。该结论依赖其训练与正则化协议，不能解读成大 batch 普遍应换某个优化器。
- **建议：用于实验设计**。扩卡或改变全局 batch 时，重新比较优化器及 loss，而不是只把单一 batch 上的排名照搬到新规模。

## 后训练与 Agent 数据

### Semi-OPD：固定初始学生轨迹可能比持续在线采样更稳

**日期/状态**：2026-10-08，arXiv **2610.11291v1**。来源：[论文 §2–4、Table 1、附录 A/C](https://arxiv.org/html/2610.11291v1)。

- **增量**：初始学生只采样一次，教师 log-prob 也离线计算；训练时仍用当前学生参数重算 log-prob 和优势。这样把采样、教师评分、训练拆成独立阶段，并用初始 teacher–student top-64 token overlap 作为选择 OPD 或 Semi-OPD 的诊断信号。
- **证据**：17 组师生模型中 14 组更优。Table 1 的 Qwen3-1.7B non-thinking ← Qwen3-4B-Instruct-2507，数学 avg@32 为标准 OPD 26.2、Semi-OPD 39.8，取 steps 60/80/100 平均；其中 OPD 行引用既有文献，训练上限16K，Semi为4K。实验使用 NeMo-RL/vLLM、BF16、1–8个8×H100节点。
- **建议：优先做基线对照**。适合已有指令模型且师生分布差异较大的蒸馏。最高11.4×是100步GPU-hours之比（Semi包含离线准备），不是等长度墙钟加速；应另做匹配rollout上限与总预算的对照，不能外推到所有模型与Agent环境。

### BranchOPD：在有信息量的决策点分叉，而不是重复整条采样

**日期/状态**：2026-10-09，arXiv **2610.13481v1**。来源：[论文 §4–5、Tables 1/3/11/12](https://arxiv.org/html/2610.13481v1)。

- **增量**：结合学生不确定性和师生分歧挑选分叉位置，复用前缀后从学生策略采样多条后续轨迹；另有 protocol-boundary repair（PBR），针对结束符和协议边界抑制重复生成。
- **证据**：Qwen3-8B 教师、Qwen3-4B-Base 学生，数学 avg@16：Vanilla OPD 25.82、加 PBR 后 38.94、BranchOPD 44.37。总提升中的很大一部分来自 PBR，不能全部归因于分叉。相比七条独立采样，ALFWorld/WebShop 学生生成 token 分别节省 19.1%/31.2%，**不包含**教师评分、反传和环境重放。
- **建议：考虑复现**。对可重放状态的 Agent 任务有价值；应同时计量环境 replay 成本，并保留“仅 PBR”“随机分叉”的消融基线。

### RTI：用训练时奖励侧信息救活全失败的 rollout 组

**日期/状态**：2026-10-09 首发、10-12 修订，arXiv **2610.13438v2**。来源：[论文 §4–5、Tables 1/2](https://arxiv.org/html/2610.13438v2)。

- **增量**：对失败轨迹做 validator→corrector→verifier 干预，用 rubric 定位并最小化修正关键错误；随后移除干预信息，以原策略重评分，通过 perplexity 门控剔除偏离太远的轨迹，再对关键修正 token 加权。
- **证据**：Qwen3-8B 的 Navigation Pass@1，GRPO 35.33%、RTI 57.33%；τ²-Telecom 为 36.84%→61.40%。这是指定五组任务、两个骨干的作者实验，不是通用能力提升幅度。论文明确用验证器防止特权信息泄漏，但该检查不等于已证明不存在泄漏。
- **建议：深入阅读**。针对稀疏奖励、初始策略几乎无法完成的任务值得试验；需要可用且可信的 rubric，核算干预模型的额外调用成本，并独立做 train/eval 隔离。本轮未复现实现。

### TerminalHorizon：长轨迹的数据价值来自任务里保留了真实工作

**日期/状态**：2026-10-09，arXiv **2610.13166v1**。来源：[论文 §2–4、Table 1、附录 A/F/G](https://arxiv.org/html/2610.13166v1)。

- **增量**：先合成专业任务环境，再让独立执行发现“构建环境时已替 Agent 做完”的工作，据此重建环境；围绕同一目标逐步增加后续要求，保留探索、修正和验收环节。
- **证据**：1,535 个环境，过滤后 2,852 条轨迹，平均 52.8 轮。Qwen3.5-27B 经 SFT，Terminus2 下 Terminal-Bench 2.0 为 55.8%，表中 base 为 41.6%（该基线标注来自模型卡）。训练为 BF16、ZeRO-3、262,144 上下文、3 epochs，默认 18 节点共 144×H20-141GB。
- **建议：参考数据生产流程**。不能把“小数据量”理解为低采样或低训练成本，也不能把 12 小时研究案例当作一般性递归自我改进证明；采用前应复核去污染与评测 harness。

## 推理算法与服务系统论文

### DCOCO：补回并行 draft 内的上下文依赖，同时保留并行计算

**日期/状态**：2026-10-08，arXiv **2610.12570v1**。来源：[论文 §4–5、Table 19](https://arxiv.org/html/2610.12570v1)。

- **增量**：DFlash 的多个位置并行预测时看不到前一个真实 draft token；DCOCO 为位置保留多个候选，并行打分相邻位置的兼容性，再选择路径，减少串行 corrector 的额外代价。
- **证据**：单 B200、Qwen3-8B、thinking off、10 个评测集合平均，32 并发、greedy：DFlash 9,655.0 output tok/s，DCOCO 11,512.4（+19.2%）；温度 1、三种子平均为 7,857.9→8,836.1（+12.4%）。该比较比“相对 vanilla 最高 6.6×”更贴近已有 speculative serving 的增量。
- **建议：考虑推理栈复现**。应在自己的输入/输出长度、并发、硬件及质量条件下重测；更长 accepted length 不必然带来更高吞吐，校正器还需要训练。

### TokenRouter：token 级跨模型切换需要保留请求状态

**日期/状态**：2026-10-08，arXiv **2610.12242v1**；论文页标注 NeurIPS 2026 接收。来源：[论文 §3–5、Table 3、附录 A.4](https://arxiv.org/html/2610.12242v1)。

- **增量**：请求侧描述路由逻辑，运行时按模型组织 subserver；跨模型交接时保留请求/KV 状态并恢复执行，结合异步调度和 delayed batching，避免把每个返回 token 当作全新的请求处理。
- **证据边界**：8×A100-80GB 服务器上研究五种 token routing 算法，Qwen3 小/大模型配置含 TP1/TP2、CUDA MPS。Table 3 相对官方 R2R，不同模型对吞吐提升 1.99–3.21×；更大的 2.01–64.15× 范围来自不同算法/负载及特定实现基线，**不是**通用 vLLM/SGLang 单模型 serving 加速。
- **建议：按场景采用**。适合逐 token 在大小模型间切换的研究与服务；对 query 级路由或普通单模型推理，不能直接套用收益。

## 硬件、软件栈与系统工程

### ReplaySSM：重放短窗口输入，减少递归状态的逐 token 读写

**日期/状态**：NVIDIA，2026-10-12；新工程实验披露，非算法首次发布。来源：[官方技术文的状态表示、评测设置与 Tables 1–2](https://developer.nvidia.com/blog/replayssm-rethinking-state-storage-for-faster-hybrid-model-decoding/)。

- **增量**：存 checkpoint 与最近输入，普通步只计算输出，窗口满再更新状态；高精度 checkpoint 保证递推，低精度副本仅用于输出，避免量化误差反复反馈。投机拒绝可回退输入指针。
- **证据**：4B模型 BF16/单H100，大MoE NVFP4/单B300，无TP、FP32状态、窗口16。batch256包含flush时，kernel为1.46–1.80×，decode step为1.20–1.47×；batch1无收益。博客未固定vLLM commit和完整输入输出长度。
- **建议：考虑高并发混合模型复现**。低精度状态实验与上述FP32性能实验不同：Nano-4B在GPQA-Diamond、NVFP4读取副本、FP16/FP8加随机舍入的递推配置下，相对FP32下降5.6–6.7个百分点，不能说量化零损。

### AMD Kimi-K3 2P2D：短上下文吞吐先被序列槽位限制

**日期/基线**：2026-10-08；厂商 fork `raviguptaamd/vllm@206fffe`，MoRI `624002c897a3`、AITER0.1.19、ROCm7.2。来源：[官方部署与干预实验](https://rocm.blogs.amd.com/software-tools-optimization/di-kimi-k3-mxfp4/README.html)。

- **增量**：prefill/decode各用2节点×8卡，专家EP16、非专家TP2×DP8；gfx942没有原生MXFP4，转packed INT4路径。RDMA接收可见性与KDA状态块回收未清零是两个不同正确性问题。
- **证据**：MI300X、同13GB KV池、1024/256输入输出，将 `MAX_NUM_SEQS` 从8改32，并发128的输出吞吐456→586 token/s，支持“该短上下文拐点主要是槽位限制”的诊断；50K上下文下，两平台各自在并发1/8/16/32的distinct-needle测试中合计57/57通过。
- **建议：借鉴诊断方法**。先控制硬件/KV池不变再改槽位；MI300X/MI325X网卡、graph和KV配置不同，不能据此做纯芯片排名。这是厂商fork配方，非所有改动已进入upstream vLLM的证明。

### MI455X/MI355X：把吞吐倍率放回用户速率和互联拓扑条件中

**日期/状态**：AMD，2026-10-08技术解释；底层测量来自7月。来源：[官方方法与脚注](https://rocm.blogs.amd.com/software-tools-optimization/mi455x-vs-mi355x/README.html)。

- 同为单GPU、ATOM、DeepSeek-V4-Flash、专家FP4、其余大部FP8、1024/1024输入输出；要求90 tok/s/user时吞吐比33.98×，改成20 tok/s/user时为2.47×。因此“34倍”是特定服务约束下的比值。
- 互联16GB消息实验中，4×MI455X对8×MI355X的单向单对带宽最高27×，每卡单向聚合仅3.8×；不能将单对数字当作MoE all-to-all整体增益。MXFP4 GEMM正文/脚注基线不一致，本期不引用其3.3×作为同条件实测结论。
- **建议：用于审查选型口径**。要求完整交互速率—吞吐曲线及聚合通信数据；预发布平台和未固定的软件栈限制复现。

### Primus：混合架构从随机初始化预训练的可运行配方

**日期/基线**：AMD，2026-10-08；`rocm/primus:v26.7`，Primus main未固定SHA。来源：[官方实验设置与结果](https://rocm.blogs.amd.com/artificial-intelligence/primus-hybrid-llm/README.html)。

- 配置化组合MLA、Mamba2/GDN/KDA与MLP，用分块fused CE控制logits显存。环境变量可覆盖YAML；`expandable_segments:True`改善碎片但作者报告约4%吞吐代价。
- 单节点8×MI355X、BF16、seq2048、FineWeb-Edu、seed42，验证300M/1B模型；1B GDN为100B token、microbatch64、global512、ZeRO-1，最终loss2.487且报告无NaN/跳步。
- **建议：作为小规模预训练复现起点**。归档容器、环境变量和有效配置；300M默认50步测试不代表文章长跑，复现实测还需 `fused_ce_mode=2` 和 `PRIMUS_FLA_MLA_ATTN=0`。单节点结果不能证明多节点/长上下文扩展。

### Run:ai 开源组件：分片、配额与调度有不同职责

**日期/状态**：NVIDIA，2026-10-13；官方功能与发布说明。来源：[组件划分与限制](https://developer.nvidia.com/blog/nvidia-runai-is-now-open-source-as-part-of-nvidia-dsx-ai-factory-platform/)。

- KAI负责拓扑/gang/公平调度，Fractioning Operator接入CUDA13.4原生显存硬限制，Resource Management表达组织配额；Workload Map描述多角色任务，**本身不执行调度或扩缩容**。
- **建议：按组件评估现有平台集成**。本轮未部署，文章无本次变更的可比利用率实验；显存硬限制不等于完整安全隔离或性能隔离。

### AWS MoE kernel 工程：小 M、后端路由和 JIT 分开诊断

**日期/状态**：AWS，2026-10-13工程总结；非这些优化本周首次合入。来源：[官方三个优化案例](https://aws.amazon.com/blogs/machine-learning/mixture-of-experts-kernel-engineering-on-amazon-sagemaker-ai/)。

- 针对H200 decode小M，Swap-AB改善矩阵tile利用；FP8 MoE选择不同后端；对可变token参数禁用过度specialization以减少重复JIT。作者分别报告吞吐+2–4%、+10%、TP8下TTFT改善9%，指标和改动不同，不能相加。
- **建议：作为profiling线索**。具体模型、序列长度、batch、重复次数和完整版本未给全，不能承诺生产收益；先用编译事件和kernel形状定位自己的瓶颈。

### HyperPod共享集群：核对GUI与CLI的同一条授权路径

**日期/状态**：AWS，2026-10-08参考架构。来源：[官方架构与作业配置](https://aws.amazon.com/blogs/machine-learning/share-gpu-clusters-across-teams-with-isolation-and-fairness-using-amazon-sagemaker-hyperpod/)。

- Studio execution role与CLI Identity Center role都要配置EKS access entry；namespace、存储目录、Kueue队列/priority-class共同表达团队的权限与资源份额。
- **建议：用于共享集群设计核对**。检查两种入口实际落到的namespace和队列；本轮未部署，原文无公平性/等待时间实验，namespace/RBAC也不证明全栈设备、网络和存储隔离。

### 【补录】ROCm10.1：存储直达、NUMA放置与kernel replay

**日期/版本**：2026-10-05，ROCm10.1。来源：[官方发布说明](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-10.1-blog/README.html)。

- hipFILE增加HIP stream异步路径和批量I/O；HIP虚拟内存接口可指定CPU NUMA节点。profiler的kernel replay beta在一次应用运行里重放dispatch、各轮间恢复被追踪设备内存以采多组counter；编译器升LLVM24并缓存LTO partition。
- **建议：按I/O与可诊断性需求评估升级**。原文无可归因的端到端训练加速实验；replay为beta，NUMA接口提供能力也不意味着所有通信路径自动采用。10-05补录不构成对旧10-08日报24小时窗口判断的反证。

### 【补录】AMD TAF：用固定GEMM形状集补充峰值FLOPs

**日期/状态**：2026-10-06方法介绍，测量来自9月。来源：[官方TAF方法、测试条件与脚注](https://rocm.blogs.amd.com/artificial-intelligence/taf-blog/README.html)。

- 预先固定来自DeepSeek-V3的10个GEMM形状，目标dtype正态分布初始化，每shape用mblas-bench测30秒，以总FLOPs和等效总执行时间聚合。单MI325X、1000W、ROCm7.2.1下，BF16 TAF为616.42 TFLOP/s；这是指定形状集的算子结果，不是MFU或token/s。
- **建议：借鉴基准协议**。实际对照应使用自己模型的shape分布，统一dtype、输入初始化和功耗；TAF不能替代通信、内存与调度的端到端测量。

## 去重、补录与待跟进

- **已去重**：GPT-6 Sol/Luna十月版、Claude Haiku5.5、Mistral Large4预览、MiMo-V2.6既有训练报告，以及旧稿中的vLLM0.31.0、score-centering/straw等。MiMo报告新上arXiv不自动算作新的训练成果；slime本次收的是正式版本及后续修复。
- **只记状态变化**：Reflection Beam官方页10-08更新评测，仍未在本轮核得权重/技术报告工件；不重复旧宣传数字。GLM HF中的榜单元数据修改，不作新模型发布。
- **代码待核**：DeepSeek-V4.1-Flash的10-12 indexer K-cache修复、MiMo预处理配置修改，已记录线索，但本轮未完成具体实现复核，不在正文宣称效果。
- **论文待精读**：DoSA（2610.13990，平台日期口径需复核）、SoftwareGym（2610.12611）、ProgramGym（2610.13395）、Thinking Inertia（2610.11765）等已进入候选清单；没有仅凭热度/摘要塞入技术结论。Ai2 Bolmo/Nature正文访问失败，也留待补查。
- **部署待核**：AWS Ray LLM DLC多节点PD文章的实例/EFA链路组合未完成交叉确认，未把示例写成已验证的加速配方。

## 实际来源覆盖与缺口

| 来源方向 | 本轮实际检查 | 已知边界与后续任务 |
|---|---|---|
| 六个重点开源项目 | 默认分支commits API按主窗口逐页获取：Megatron69、vLLM384、verl39、SGLang466、slime11、torchtitan94条；查询release，并打开入选PR正文核对合入时间/SHA | “完整”只指该日期过滤下API返回清单取完，非每个commit代码审计。committer时间不等于merge时间；已对入选项另核。10-05至10-08的72小时全量commit回溯尚未完成，已入待补查记录 |
| 模型厂商 | OpenAI、Anthropic、Google、Liquid、Qwen、腾讯、Mistral、Ai2、xAI、Reflection等官方页面；16个官方HF组织分别查询最新7个创建工件，并对候选核固定版本卡/config/提交 | HF创建/修改时间不是公告时间；最新7个工件也不覆盖所有旧仓库更新。ByteDance和MiniMax动态网页提取不足，中文闭源API发布覆盖较弱 |
| 论文/技术平台 | HF Daily Papers按10-08至10-14逐日查询，共183个唯一候选；主题搜索补充arXiv，正文精读9篇，均固定版本；OpenReview、ACL Anthology、PMLR做定向搜索 | HF为精选入口，非全量arXiv；10-14列表在运行时尚会变化。会议平台未做全量增量枚举，未获得可确认的本窗口新条目不能解读为无更新；未精读候选单独保留 |
| NVIDIA | Technical Blog、候选正文、NCCL release页面 | 未穷举CUDA全部组件、Dynamo/TRT-LLM所有提交；博客公开评测不等于独立复现 |
| AMD | ROCm posts列表及具体文章、新闻入口 | 初始首页缓存过旧，改从posts与原文核日期；个别Web提取失败已用官方HTTP正文补读。没有遍历所有ROCm组件仓库 |
| Intel | Gaudi软件更新入口、官方域定向搜索、Habana release notes尝试 | release notes不可读，旧更新页不足以判断当前版本；oneAPI/OpenVINO/XPU仓库未扫描，列为覆盖缺口 |
| 华为昇腾 | 官网、产品公告、CANN文档目录与版本链接、中文主题检索 | 多处仅返回导航，未核实窗口内发布日期；Gitee/GitCode与vllm-ascend本轮未做完整增量扫描，列为待补查 |
| Google TPU | Cloud Blog定向检索、TPU release notes | 未遍历JAX/XLA/libtpu提交；不能据页面未见新条目判断软件栈无更新 |
| AWS | ML技术博客、Neuron SDK release index与分类入口 | H200 MoE与HyperPod原文已读；未穷举Trainium/Inferentia组件提交和全部工程渠道 |

公共GitHub未认证API曾触发403，后切换已认证连接器完成六项目查询；verl入口从旧组织重定向到 `verl-project/verl`。本轮没有把访问失败、缓存旧或搜索无结果混同为“没有发布”。PyTorch本体、DeepSpeed、TRL、OpenRLHF、FlashAttention、Triton、TensorRT-LLM等扩展项目尚未逐项增量扫描，后续需补齐或轮转覆盖。

## 本期优先实践建议

1. **先做正确性门禁**：权重identity reload、参数地址/校验和、跨cache恢复的输出与梯度，以及异步更新的部分失败处理。
2. **再选实验切口**：有蒸馏任务先测师生overlap和匹配预算的Semi-OPD；Agent任务分开评估PBR、分叉和RTI，不把组合收益全部归因于一个组件。
3. **最后比较性能**：记录硬件、精度、batch/并发、输入输出长度、质量指标、版本和总成本；用同一负载核对kernel、step和端到端收益。

以上是根据本轮证据给出的工程判断，不代表这些方法已经在你的训练环境验证。
