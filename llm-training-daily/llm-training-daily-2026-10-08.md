# LLM 技术动态日报 · 2026-10-08

> 窗口：2026-10-07 08:00 → 2026-10-08 08:00（北京时间，24h；即 UTC 10-07 00:00 → 10-08 00:00） · 生成时间：2026-10-08 10:05

## 今日要点

1. **两个"loss 照降、梯度是错的"静默 bug 同日修复，都值得立刻自查。** torchtitan #5123：DeepEP 在 FullAC（或 RegionAC 未保存 `ep_communication`）下重放 dispatch，而 DeepEP 用 atomics 分配 receive slot，重放顺序与 forward 不同——DeepSeek-V3 671B 形状下 routed-expert 权重梯度相对 L2 偏 **1.41**、与正确值 cosine 仅 0.01–0.03，第 2 步 loss 却只差 6.1263 vs 6.1208。slime #2443：CP>1 时 linear-attention（Qwen3.5 / Qwen3-Next 的 GDN，占 3/4 层）的 all-gather backward 只取本地切片，真实 GDN 层 CP=4 输入梯度偏 **9.4%**，而本层参数梯度看起来是对的。
2. **RL 权重 refit 成为本期论文主线**：NVIDIA 的 **NCCL M2N**（布局描述符 → 确定性重叠区间 → 拓扑感知路由，DeepSeek-V3 @256 GPU 的 weight sync 5.78 → 2.77 s）与 **NeMo-DCR**（BF16 下每步只有 0.6–1.2% 元素变值，XOR delta 跨 region refit 快 12–40×）分别覆盖同集群高带宽与跨集群低带宽两种场景；两篇 NVFP4 RL 论文（TRACE、TRIAGE）则都把端到端瓶颈指回 refit。
3. **vllm-ascend 单日 9 个合入，是近几期昇腾侧最密的一天**：EPLB 默认改走 **CANN HIXL** NPU-to-NPU 直读（前台每 cycle 仅等 1.8–2.6 ms）；**TurboQuant 4-bit latent KV**（386 B/slot，KV 容量 2.68× bf16，但并发 1 时吞吐 −24.8%）；两个真 bug——PCP 短 prefill 下各 rank 对"是否 gather O-proj"判断不一致导致 **hang**（#17947），hybrid 模型页复用时 GDN state 覆写 C8 V-scale 导致 **NaN**（#17956）。
4. **trl #7549：data-parallel vLLM 上不带 seed 的 GRPO 组内大量逐字重复**。各 DP engine 的 engine seed 相同且无 rank 偏移，第 k 个请求拿到相同 seed；8 个真实训练运行中短 completion 的重复率 **65–70%**，"有 advantage 信号的 group"从期望 92% 掉到 66%，而平均准确率完全正常。vLLM main 上该行为仍在。
5. **Megatron-LM #7287：异步 checkpoint 的同步前缀 17.75 → 5.22 s**。48 层 × 128 experts、EP=8 时 prepended layer/expert 轴让每 rank 枚举 470 万个全局网格条目，`mcore_to_pyt_state_dict` 11.71 → 0.07 s（165×），metadata 逐字节不变。同日 #7562 修掉 hybrid Mamba + squared-ReLU MoE 在 batch-invariant 模式下训推 logprob 对不齐（多舍入一次 BF16）。
6. **更正上期**：上期收录的 pytorch #192524（ROCm 上启用 NCCLSymmetricMemory / RCCL）已于 10-07 22:04 UTC **被回滚**，理由是内部 RCCL 版本失败，待 reland。

---

## 1. 模型发布与 Tech Report

窗口内**没有任何开放权重、训练代码或带架构/集群细节的 tech report**。仅两条闭源产品发布，infra 信息量很低，简记：

**Claude Haiku 5.5**｜Anthropic｜2026-10-07（页面只给日期）｜https://www.anthropic.com/claude-haiku-5-5

- 首个带可调 effort 的 Haiku 级模型，定位为 Opus 5.5 / Sonnet 5.5 的 subagent 与 compaction/分类类负载。定价按 prompt ≤100k / >100k 分档：input $0.10 / $0.50，output $0.50 / $2.50，cache read $0.01 / $0.05（每百万 token）；官方称平均比 Haiku 4.5 便宜约 75%。
- 官方自报：OSWorld 2.1（offline subset）72.4%、Terminal-Bench 4.0 39.2%、HLE 无工具 45.9%。System card 给的 knowledge cutoff 为 2026 年 6 月。
- **未披露**：参数量、架构、上下文窗口、训练算力、tokens/s（"最快"无数字）。
- **对我的意义**：无 infra 可借鉴内容。唯一信号是小模型价格再降一个量级，agentic RL 里用它做 judge / verifier 的成本模型需要重算。

**GPT-6 Sol / Luna 10 月版全量上线 ChatGPT**｜OpenAI｜2026-10-07｜https://openai.com/index/gpt-6-for-everyone/ ｜ https://deploymentsafety.openai.com/gpt-6-october

- 不是新一代模型：GPT-6 首发在 9 月，本次是 10 月 checkpoint 在 ChatGPT 取代 GPT-5.6；Codex 仍用 9 月版。无架构、参数量、训练规模信息，无 API 变更。
- **对我的意义**：无。

> 去重条目状态：**Reflection AI Beam 的权重与技术报告仍未放出**（官方博客仍写"本月稍后"，HF 上查不到该 org）；Mistral Large 4、Kolibri、`openai/math`、`allenai/miles-olmo-core` 窗口内无新进展。
>
> 已核对无新增：HF 组织最新模型 createdAt——deepseek-ai 09-10、Qwen 09-20、zai-org 08-25、moonshotai 06-13、MiniMaxAI 08-07、XiaomiMiMo 09-27、nvidia 10-01、Aleph-Alpha 10-02，其余更早。GitHub 15 个 org 窗口内唯一新建仓库是 `google-deepmind/motion-forecasting`（与 LLM 无关）；deepseek-ai、MoonshotAI、zai-org、meta-llama、ByteDance-Seed、XiaomiMiMo、stepfun-ai 窗口内无任何 push。

## 2. 论文精选（训练 / 后训练 / 推理）

**NCCL M2N: A Layout- and Topology-Aware Collective for Distributed Tensor Resharding**｜NVIDIA｜arXiv 2610.07516（v1 戳 10-05，10-07 批次公告）｜https://arxiv.org/abs/2610.07516

- **问题**：RLVR 中 trainer 与 generator 并行布局不同，每步要把权重从 M 个源 rank 重分片到 N 个目的 rank。flat direct send 对每个目的副本重复发送；all-gather + broadcast 把流量压在单 root。基线下 DeepSeek-V3 @256 GPU 的 weight sync 占 RL step 的 **29.4%**。
- **机制**：布局 L=(S, M, p)，每个 mesh 轴标为 SHARD(d) 或 REPLICATE。调度由两端布局确定性推导——源/目的区间在各维的交集非空才传，所有 rank 独立算出同一调度，无需协调。层次路由：每份源贡献只进每个目的 NVLink 域一次，域间 pipelined ring 转发，域内 NVLink 扇出副本。给出数据移动下界 `T_SOL = max(B / min(N_s·β_N, K·ρ·β_N), (ρ−1)B / (K·ρ·β_V))`。
- **配置**：GB200 NVL72 + NDR IB，NCCL 2.31。端到端为 trainer TP1/PP8/EP16（128 GPU）→ generator TP16/DP8（128 GPU），NeMo-RL 异步。
- **数字**：单层 7168 MB FFN-MoE 张量，目的副本数 r_d=2/4/8 时 hierarchical 为 10.25 / 13.83 / 9.78 ms（SOL 8.96 ms），对 flat direct send 快 2.0× / 2.8× / 7.9×；AG+bcast 为 222–240 ms，root 有效注入带宽仅约 30 GB/s。端到端 **weight sync 5.78 → 2.77 s（2.09×），step 19.68 → 17.18 s（−12.7%）**。
- **局限**：每个布局最多一个 sharded mesh 轴，TP×EP×DP 三向组合不能原生表达；源与目的 mesh 必须不相交；出错即 communicator 级 fail-stop；只有 256 GPU 一个端到端点，无 RoCE。
- **对我的意义**：三段式抽象与硬件无关，HCCL 上可做同构原语（域内 HCCS、域间 RoCE）。SOL 公式可以直接拿来估我们 refit 距带宽极限多远；单 root 的 AG+bcast 是最该先淘汰的路径。与上期 vLLM v0.31.0 的 sharding-aware M2N 后端（#51520）是同一条线的两端。

**NeMo-DCR: Bit-Exact Delta-Compressed Refit for Scalable Agentic RL at Trillion-Parameter Scale**｜Aalto / NVIDIA｜arXiv 2610.08430（v1 戳 10-06；**公告压在窗口右边界**，见备注）｜https://arxiv.org/abs/2610.08430

- **观察**：BF16 下每步 GRPO（lr 1e-6）只有 **0.6–1.2%** 的元素改变存储值（6 个模型前 5 步）。
- **机制**：(1) canonical 坐标——shard owner 用固定仿射索引映射把变更直接投影到 HF checkpoint 的张量名 + flat index，覆盖 96–97% 权重字节；其余走"转换后与残差基线逐 bit 比较"。(2) 路径保 bit 且无重叠写时发 **XOR mask**（高位多为零、易压缩），否则发绝对覆盖；游程 + gap 编码位置后 zstd level-1。不用算术重构，避免舍入累积。(3) 接收端 hook 住 vLLM 原生 loader 的最终 storage copy 原地应用，不留基线副本；**重试一律改为覆盖**（XOR 重复应用会还原旧值）；全部 ACK 后 compare-and-set joint commit。
- **配置**：训练 32×GB300，rollout 64×H100 在另一 AWS region，每节点跨集群实测至多 5 Gbps。
- **数字**：120B（247.2 GB）全量传输 750 s；delta 在 3% / 5% 变更率下 22.6–24.5 s / 41.6–49.7 s（**15–33×**）。1T @3% 为 150 s，对比全量 87.5 min。传输下界占 refit 时延 77–94%。源基线占 host 内存约 1.03× checkpoint。
- **局限**：3% / 5% 是**均匀随机合成注入**，真实变更的分布与可压缩性可能不同；只验证 BF16 接收端，rollout 侧量化（FP8/FP4）时变更要走 residual 路径、收益未测；未与同集群 NCCL refit 对比。
- **对我的意义**："每步约 1% 元素变值"是可以在我们后训练任务上花半小时验证的事实，成立的话跨集群 refit 就有一个数量级的空间。整套方案是 Python + zstd、不依赖 NVIDIA 硬件。与 M2N 互补：同集群用拓扑感知 collective，跨集群用 delta。

**Expert Coupling in MoE Pretraining: Reducing All-to-All Overhead with Correlated Placement and Token Shuffling**｜Zyphra｜arXiv 2610.09372（v1 戳 10-07）｜https://arxiv.org/abs/2610.09372

- **问题**：8×MI300X/节点、节点间每 GPU 一块 100 Gb/s RoCE 的集群上，EP all-to-all 占 step 的比例：top-2 为 13.5%（EP8）/ 44.8%（EP32），top-6 为 24.0% / **60.4%**。
- **观察**：路由早期就出现稳定相关——top-2 下 0.8% 的 expert 对承载 42% 的 token（独立假设下 1.6%）；128 个 layer-8 expert 里有 112 个把 ≥30% 的 token 送往同一个 layer-9 expert。
- **机制**：(1) 去重 dispatcher，每个 (token, 目的 rank) 只发一行；(2) correlated placement，按共选图用 Kernighan–Lin 式局部交换把常被同时选中的 expert 放到同一 GPU / 节点，每 GPU expert 数不变、无副本；(3) token shuffling（仅 TP=EP 且开 SP），用前两层的 expert 对预测下一层 owner rank，把置换融合进已有的 reduce-scatter，不新增 collective。**都不改路由决策和 loss。**
- **数字**（Megatron-LM，12 层、128 experts）：placement + 去重的 step 加速，top-6 在 EP8/16/32/64 为 1.14× / **1.41×** / 1.25× / 1.23×；加 shuffling 后 TP16 EP16 top-6 的 A2A 加速 1.95× → 2.63×。相关表拟合极便宜：4096 个 token 即可，step 199 拟合的表已拿到约一半收益。
- **局限**：只有一个 12 层小模型、2.1B token、最大 EP64；未讨论 placement 换位时 expert 权重与 optimizer state 的迁移成本；未见 PP/CP 组合。
- **对我的意义**：这是节点间带宽紧（RoCE）时的纯软件收益，正对昇腾 HCCS + RoCE 的拓扑。接入成本低：采一个 microbatch 的路由 trace → 划分 → 给出每层 expert 置换。上规模前要先验证 256+ expert、top-8 下相关性是否还在。

**Memory-Efficient Expert Routing for Distributed MoE Training（RelayMoE）**｜William & Mary / BSC｜arXiv 2610.07333（v1 戳 10-05，10-07 批次）｜https://arxiv.org/abs/2610.07333

- **诊断**：开了 CP/SP 之后，长上下文 MoE 的显存峰值来自 dispatch 的 top-k 扩展 buffer 而非 attention。实测（top-8、seq 16384、8×H100、TP=SP=2/CP=4/EP=8）：输入激活 336 MiB，每个扩展 buffer 约 2.6 GiB，峰值 51 GB 而谷值仅约 14 GB。
- **机制**：把 A2A 换成沿 EP 组的 P2P 环形流转——expert 权重或 token 激活二选一，按每跳通信量 `(E/EP)·(W/ETP)` 对 `(B·S/(SP·CP))·(H+M)` 取小者；反向逐跳"重算 → 用完 → 释放"；计算当前分区时异步预取下一分区。与 A2A 数学等价，无 capacity factor、不丢 token。
- **数字**（Megatron-LM，H100）：MoE 执行峰值显存 EP=8 平均降 3.8×、EP=64 平均降 7.6×；全模型峰值最多降 35.7%；最大可训序列 Qwen3-30B-A3B 由 393K 到 786K。
- **局限**：expert 权重体积大（E384、H2048）且序列短时慢于 A2A；最大 32 GPU；与 DeepEP 只在单节点比过，吞吐"相当"。
- **对我的意义**：结论本身值得在我们的配置上复核一次——如果 OOM 边界确实由 dispatch 扩展决定，单卡 HBM 较小的平台上这是比继续加 CP 更直接的手段。

**TRACE（阿里 / OSU，2610.07767）与 TRIAGE（InfiX.ai / NVIDIA，2610.07043）：两条稳定 NVFP4 RL 的路线**｜https://arxiv.org/abs/2610.07767 ｜ https://arxiv.org/abs/2610.07043

- **TRACE**：训练侧 fake-quant 不做 round-to-nearest，而是在两个相邻 E2M1 码字里选离 **rollout 实际码字**更近的那个；因为 >99% 的失配只差一个相邻码字，只需回传后半层的 1 bit mantissa（7.5 KB/token，全量记录约 50 KB/token）。栈为 verl + Megatron + SGLang。Qwen3.5-35B-A3B 四基准均分：BF16 74.9 / QAT 59.6 / QUADS 68.8 / **TRACE 75.3**；128K 输出下解码吞吐最高为 BF16 的 5.4×；step time +7.4%。局限：大模型（122B、2.4T）只训 50 步；训练侧是 fake-quant 而非原生 FP4。
- **TRIAGE**：learner 与 sampler 都跑原生 NVFP4 时失配会自放大。定义 δ = log π_l − log π_s，指出只有 δ·g > 0 的 token 放大失配，而 TIS / clip 只看幅度不看方向。**崩溃前先出现两个放大区的不对称**（ρ_asym 在 30B 上 1.34 → 1.85），此时 |δ|<0.05 的 token 仍占 90% 以上；尾部集中在约 5% 的 64-token 段里，响应级均值看不出来。方法是段级门控 + 有界修复项，纯 loss 侧改动。Qwen3-30B-A3B 五基准：BF16 72.41% / NVFP4 在 step 700 崩溃 / +TIS 63.18% / **+TRIAGE 70.96%**。rollout 吞吐 2.3×，但**端到端只有 1.21–1.30×**，作者明言瓶颈转到参数同步与 Amax 等量化 reduction。只测了同步 RL。
- **对我的意义**：两点可直接落地。(a) 训推一致性的目标应是**两条路径落到同一码字**，而不是各自贴近高精度值——对昇腾上任何低比特 rollout 都成立，没有原生 FP4 可以先在 FP8 / INT 量化上验证。(b) **ρ_asym 与 64-token 段级尾部集中度**是比 mean|δ| 更早的崩溃预警，与精度格式无关，建议加进 RL 的训推一致性看板。

> 未精读但相关（窗口内，仅列线索）：**2610.07593 TRANSIT**（UIUC / IBM，`LD_PRELOAD` 把 GPU 分配重定向到 UVM 做透明 scale-in，64×H100 上 GPU 数减半仍保 90%+ per-GPU 吞吐；强依赖 CUDA UVM）；**2610.09424 CoMoE**（清华 / 阿里云，无 P2P 的消费级 GPU 上用 host 内存做 EP 路由中枢）；2610.07688 FailBench（跨并行策略评测容错）；2610.08378 Lachesis（agent 负载 KV 在 HBM 与高带宽闪存间按生命周期放置）；2610.09307 vLLM-Omni Technical Report；2610.07332（agentic RL 下的 MoE expert 选择）；2610.08448（跨 tokenizer 的 on-policy 蒸馏，HF 153 票，v1 日期未确认）。

## 3. AI Infra 技术文章

**DeepSeek-V4.1-Flash on vLLM: 5x Agentic Throughput Since Day 0**｜vLLM Blog（Inferact 与 vLLM Team）｜2026-10-07｜https://vllm.ai/blog/2026-10-07-deepseek-v41-flash

- **总体**：SemiAnalysis AgentX benchmark 上相对 day-0，低延迟配置约 1.9×、高吞吐配置约 5×（TL;DR 写 5.3×，正文写"about 5x"，口径不一）。
- **SWA bounded replay（核心）**：prefix cache 里 SWA KV 的存储是 global KV 的 10 倍以上，而 decode 只读每层最后 128 个位置。做法是**只缓存 global KV**，命中长度 H 时重跑 [H−128, H) 重建 SWA KV；prefill 侧 layer 20 跑全部 token，layers 21–39 只跑每个请求最后 128 个 token，长 prompt 下近半个模型被跳过。结果非 bit-exact（GSM8K / GPQA 差异在约 1.5 个标准误内）。prefill 计算时间降 30–40%——**但不配 CUDA graph 时，1K prompt 下反而比 baseline 慢最多 12%**（短 prompt 由 kernel launch 开销主导），所以 layers 0–20 与 21–39 按各自 batch 形状分段捕获。
- **kernel**：Sparse MQA logits（后续 indexer 层只对 16K 候选位置打分）在 512K 上单层 14–23×、1M prefill 2×；MegaAttention + NVFP4 KV 比原 FP8 cache 小 45%；fused WO-A（CuTe-DSL，融合 inverse RoPE、FP8 量化、batch GEMM、MXFP8 重量化）该路径最多约 2.1×；小 batch 下在旁路 CUDA stream 预算下一个 mHC block 的系数，TP4 延迟降约 4%；Engram 层的 CPU-offload lookup 开 THP 后最多快 10×。
- **部署**：高吞吐用 DEP2（DP attention + experts 跨 GPU 切分），理由是 DP 避免了 TP 会复制的共享 KV latent。
- **对我的意义**：(a) "裁剪计算量的优化必须和图捕获配套，否则短序列劣化"在 CANN 图下沉 / torch_npu 的 host-bound 小算子场景同样成立，12% 这个反例数字可以直接引用。(b) THP 对 CPU 侧大表 lookup 的 10× 与加速器无关，鲲鹏主机侧可直接验证。(c) 注意同日 vLLM #59197 正是在修 bounded replay 与 PD 分离叠加时的块错位（见第 5 节）。

> 其余来源窗口内无 infra 长文。HF Blog 当日有 NVIDIA《Fine-Tuning Nemotron for IOI and IMO》（10-07），未披露硬件、GPU 数与训练框架，未收。PyTorch Blog 最新 10-06、NVIDIA Technical Blog 最新 10-06、AWS ML 10-06、Together AI 10-06、Meta Engineering 10-06（非 AI）、Microsoft Research 09-30。

## 4. 硬件与互联

**无新增。** NVIDIA、AMD、Intel Gaudi、华为昇腾、Cerebras、Groq 窗口内无新芯片、超节点/互联或 NCCL / HCCL / RCCL / CUDA / ROCm / CANN 主版本发布（Google TPU、AWS Trainium、寒武纪、海光仅靠搜索判断，否定强度较弱）。NCCL 最新仍为 v2.32.3-1（09-17），RCCL 为 rocm-7.2.4。

两条相关信号：

- **OCP Global Summit 2026 日期一手确认为 10-12 至 10-15（San Jose）**，上期写的"至 10-16"来自二手页面，以此为准。AMD 已预告 Lisa Su keynote。
- 软件侧的昇腾互联进展见第 5 节 vllm-ascend #17578（HIXL 进入 EPLB 默认路径）。

## 5. 开源框架与社区

窗口内**所有被跟踪仓库均无正式 Release**。verl、OpenRLHF、NVIDIA/nccl 窗口内 0 commit。

### pytorch/torchtitan（15 个 commit）

- **#5123 DeepEP 在 FullAC 下梯度错误：保存 dispatch / combine 而不是重放**｜https://github.com/pytorch/torchtitan/commit/0167526a9fdd67ad6b0c3df4c739d6a148c582e6
  - **根因链**：dispatch 返回收到的行和一个记录来源的 handle，backward 靠 handle 路由梯度 → FullAC 在 backward 重放 forward，dispatch 跑第二次 → **DeepEP 用 atomics 分配 receive slot，第二次的行顺序可能不同** → grouped GEMM 的激活来自第二次 dispatch，梯度却按第一次的 handle 路由，每个 token 的梯度遇上了另一个 token 的激活。
  - **数字**（DeepSeek-V3 671B 形状，D=7168、256 experts、top-8，EP=8，8×H100，bf16，4k tokens/rank，相对 L2）：main 上 routed experts **1.41**、router gate 1.20、MoE input 0.93；修后 2.7e-5 / bitwise / 6.3e-4，与"无 AC 对无 AC"的噪声基线相同。DSv3 debugmodel 第 2 步 loss：无 AC 6.12075，FullAC main 6.12634，修后 6.12075。
  - **修法**：把 `deepep::dispatch` / `combine` 注册为 ordered effects，selective checkpointing 默认保存这类 op。副作用：FullAC 在 DeepEP MoE 层的省显存效果基本退化到 SelectiveAC 水平（2 层 4k 时保存激活 114 → 677 MiB），速度反而快约 5%。
  - **作者自述**：目前没有 recipe 把 DeepEP 配 FullAC / RegionAC，但也没有任何机制拒绝这个组合。**PP 的 split backward（ZBV、DualPipeV）下现在会报错而不是静默算错——即在 split backward 支持 saved effectful op 之前，DeepEP 配这些调度没有可用的 AC。** 顺带发现 DeepEP 的 CI 安装自 09-30 起每晚失败，这一周没有任何 DeepEP 测试实际跑过。
  - **对我的意义**：机制可直接类比到任何"receive 顺序非确定的 EP all-to-all + 重算式 AC"。三条可操作：输出顺序非确定的通信算子，backward 绝不能靠重放 forward 获得；回归必须做"与无 AC 的逐张量梯度比对 + cosine"，看 loss 没用；统计 dispatch 调用次数（3 次而非 2 次）是个便宜的断言。
- **#5086 gated activation 的 unbind 移入编译区（breaking）**｜https://github.com/pytorch/torchtitan/commit/948d65c868c5fa8f0290bcf9e54b69f004721f54
  - 融合权重 `w13` 输出 `[T, 2, F]`，在 eager 里 unbind 后 autograd 会在 backward 自动插一次 `torch.stack`。Qwen3.5-4B FFN（16k tokens，H100）上这次拷贝占非 matmul kernel 时间 1642 us 里的 **575 us**；移入编译区后 1642 → 1066 us，输出与梯度 bitwise 不变。另一组数：hidden size 被编成 dynamic 时 fwd+bwd 为 2087 us，static 为 1092 us。
  - Breaking：`activation_fn(gate_up, offsets=...)`；`local_compile_regions` 里 `swiglu` / `situglu` 改名 `fused_binary_activation`，旧名启动即报错。
  - **对我的意义**：fused gate_up 布局 + 自研融合激活算子时，unbind 应在算子内部、backward 直接写融合梯度；这笔隐式拷贝在 profile 里不显眼但占到三分之一。
- **#5109 GraphPP：dI/dW 拆分后给 dW 图绑定 EP receive count 符号**｜https://github.com/pytorch/torchtitan/commit/1472b0714f2a2500a4bc47728b1c30b350100de7
  - DeepSeek-V3 配 Interleaved1F1B / ZBV / DualPipeV 的 full Inductor 编译失败。起因是 #4927 之后 expert weight-grad 的 grouped GEMM 被挪进 dW 图（这正是 zero-bubble 想要的），dW 图第一次依赖形状 `(u6 + u7, D)` 的 per-token 张量，而 u6 / u7 是 EP all-to-all 的 receive count，没有任何 dW op 直接读它们。修法是转发符号绑定节点，并给无 user 的 SymInt 加一个 side-effectful 的 `sym_constrain_range_for_size` 防止被 DCE。10 步、8 rank loss 与 grad_norm 与 main bitwise 相同。
  - **对我的意义**：EP + zero-bubble PP + 图编译三者叠加时的必经之坑，图模式下做 dI/dW 拆分 + 动态 EP 形状会遇到同形问题。
- **#5097 [rl] vLLM `prefix_cache_retention_interval` 设为 `None`**｜https://github.com/pytorch/torchtitan/commit/98e7b95010467eee5a82da8f783b5951f1e67b85
  - vLLM 自 v0.29.0 起默认 `0`：sliding-window 与 Mamba / GDN 的 state block 只 hash replay boundary，**decode 期间跨越的 block 未 hash 就释放**。多轮 rollout 的下一轮 prompt 逐字包含上一轮 completion，所以永远命中不到上一轮 prompt 之后。算例（1024-token block，轮 1 prompt 5500 + 生成 1800 + tool 返回 700）：轮 2 命中 5120 → 7168，需重算 2880 → 832 token。hybrid 模型的命中上限被最保守的 cache group 卡住。
  - 实测命中率与 step time 对比在 PR 里**留空未填**。
  - **对我的意义**：多轮 agentic RL 跑 hybrid 模型时，vLLM 的这个默认值是反向优化，vllm-ascend 上应检查同名配置。
- **#5103 [rl] prefetch staging buffer 绑定到 GPU-local NUMA 节点**：未绑定时 UCX 可能选到跨 socket 的 NIC。Qwen-27B 上稳态 GET 均值 2.77 → 2.68 s，**最差 3.89 → 2.87 s**——收益主要在长尾。
- **#5105 [rl] 修启动期间歇性 EADDRINUSE**：Monarch 选 `MASTER_PORT` 的方式是 bind `127.0.0.1:0` 再关闭，之后无人持有；09-29 以来 **424 次 RL CI 运行里 53 次（12.5%）** 因此失败。更值得记的是：在"actor 失败时返回非零退出码"那个修复之前，这个问题已经带着相同比例被 CI 判绿至少一周。修法是由 rank 0 直接 bind port 0 并持有到进程退出。
- **#4821 `NVFP4GroupedLinearConverter`**：DeepSeek-V3 routed experts 支持 NVFP4，671B recipe 为 **FFN 与全部 61 层 routed experts 用 NVFP4、attention 路径用 MXFP8**，在 64×GB300 上验证；PR 未给吞吐或精度数字。
- 其余：#5088 每个梯度累积 group 重放同一张 CUDA graph（无数字；拒绝 deferred gradient reduction 与 SDC replay）；#4893 SelectiveAC 迁到 torch_remat，固定为"保存除 routed-expert grouped matmul 外的所有 region"；#5090 RegionAC 暴露 per-block saved-tensor hook（为激活 offload 铺路）；#5095 LoRA 按 FQN 指定目标；**#5110 dense Llama3 / Qwen3 的 AOT FX trace 与 eager 数值发散，被标成 strict xfail 而非修复**。

### THUDM/slime（3 个 commit，三条都重要）

- **#2443 CP>1 时 linear-attention 梯度错误**｜https://github.com/THUDM/slime/commit/f0f6e5545fcc534b13bfab8568e518c9dd450f2a
  - CP 下 `HuggingfaceAttention` all-gather 全序列，每个 rank 在完整序列上算，只保留自己两个 zigzag chunk 的输出。自 #1748 起 gather 用的 autograd Function 在 backward **只返回本地切片**——这只在各 rank 梯度相同时正确（SP 成立）。CP 下每个 rank 只收到自己输出 chunk 的梯度，一个 chunk 的真实梯度是所有 rank 的求和，取切片等于丢掉了"通过影响其他 rank 上后续 chunk"的全部路径。
  - **数字**：真实 `Qwen3_5GatedDeltaNet`、bf16、CP=4，输入梯度偏 **9.4%**（修后 0.06%，bf16 噪声）；**本层参数梯度修前修后都是 0.3%**——所以单看这一层的 weight grad 发现不了，错的是交给下层的梯度。toy causal 层 CP=2/4/8 偏 74% / 129% / 112%。
  - 修法：CP 的 gather 改回 `torch.distributed.nn.functional.all_gather`（backward 是 reduce-scatter）。PR 附纯 CPU + gloo + float64 的复现脚本。
  - **对我的意义**：凡是对无法按序列维切分的层（linear attention / Mamba / GDN）采用"all-gather 全序列 + 各 rank 算全量 + 只留本地输出"的实现，都要确认 backward 是 reduce-scatter 而不是 slice。复现脚本可以直接搬到昇腾环境做回归。
- **#2444 训练侧重启、保留 serving 与已接受的 rollout**｜https://github.com/THUDM/slime/commit/4a7a50909ac310b7b23caf851ef05a2af26baa74
  - 失败的 Megatron 训练任务在同一 Ray 集群上重启，保留健康的 SGLang engine、router 和已接受的 rollout 数据；新 trainer **可以换并行布局**，重放最后一次已提交 checkpoint 之后的 batch，不重新生成样本。serving 做成具名、detached 的 `ServingCluster`，trainer 是可替换的借用方。
  - 几个容易做错的点：**"运行时训练完成"与"持久化 checkpoint 完成"分离**，只有已提交的 checkpoint 才释放重放 pin；**weight-update-group reset 要先发给所有 serving group 再等待**——prefill 与 decode 可能共享一个 NCCL group，先等一侧会让健康的 PD engine 死锁；清理失败不覆盖原始训练异常。
  - **验证**：4 主机 × 8 H100，真实训练 OOM 触发；Qwen3-30B-A3B + R3 下 TP 1→2 / DP 4→2 恢复，serving 保留、样本与路由数据重放。全局 batch 只转换发布一次、每个 DP rank 只收"对共享 tensor record 的选择"后，DP 分区阶段 3.097 → 0.096 s（作者声明不是端到端加速）。
  - **局限**：不引入自动重试；丢失 serving owner 或整个 Ray 集群仍需冷启动；不承诺重启后逐位相同。
  - **对我的意义**："只重启训练侧"是 RL 容错性价比最高的形态，且"因 OOM 失败 → 换更大 TP 重来"正是真实运维路径。PD 共享通信组的 reset 死锁在 HCCL 上同构。
- **#2442 disk-delta sync 遵循 checkpoint 的 shard index**｜https://github.com/THUDM/slime/commit/2f2318653f6f794dddd321eff7c9d4b7b174643f
  - checkpoint 目录里有未被引用、张量名重复的 safetensors 文件时，trainer 的 baseline reader 和 engine 的 writer 都扫描**每一个** safetensors 文件，双方对同一个陈旧文件达成一致、**checksum 校验通过**，而真正用于 reload 的 indexed 权重没变。修法是以 `model.safetensors.index.json` 为准。13 个 CPU 回归用例在 baseline 上 10 失败。
  - **对我的意义**："同步成功但权重未更新"是 RL 里最危险的一类故障，且校验和发现不了。端到端验证必须比对 reload 后的权重与前向输出。

### huggingface/trl（10 个 commit）

- **#7549 AsyncGRPO 在 data-parallel vLLM 上组内 completion 重复**｜https://github.com/huggingface/trl/commit/9bb31ab972fb21f218f9ebf10c2816037270cd82
  - **机制**（vLLM Model Runner V2）：不带 `seed` 的请求在 admit 时从 numpy **全局** RNG 取 seed；每个 engine 用相同的 engine seed（默认 0）播种，**没有 DP rank 偏移**；Gumbel 噪声是 request seed 与 token position 的无状态函数、与 batch 无关。于是各 engine 上第 k 个被 admit 的请求 seed 相同，同 prompt 解码出相同文本。
  - **数字**（8 个内部 async GRPO 运行，vLLM 0.25.1，DP=8，每 prompt 16 个 generation）：平均 completion 约 400 token 的运行重复率 7–14%；约 125–175 token 的三个运行为 **65.7–70.0%**，带 seed 的对照组 ≤0.13%。其中一个运行每组平均只有 5.5 / 16 个不同 completion；全对 group 27.8%（独立采样期望 6.7%），**有 advantage 信号的 group 66%（期望 92%）**。按 pass-rate 分箱的平均准确率与对照相差 0.01 以内。
  - 修法：每个请求带 `"seed": random.getrandbits(63)`。**修后的训练效果与吞吐代价均未测**；V1 runner 上带 seed 的行是逐行采样，大 batch 可能变慢。PR 自述为 AI 生成。
  - **对我的意义**：GRPO 类训练必须客户端显式给每个 generation 不同 seed。诊断指标是"组内 distinct completion 数"和"全对 / 全错 group 比例对独立采样期望"，不是 loss。值得注意的是 batch-invariant 的无状态采样让这类 seed 碰撞更持久——batch 扰动不再能把同种子的轨迹分开。
- **#7390 DPO / KTO 用 fused LM head 打分（breaking）**：不再物化 `[batch, seq, vocab]` logits。1×H100、bf16 下峰值显存 Qwen3-0.6B 8k 为 28.7 → 7.0 GiB（−76%），Qwen3-8B LoRA 4k 为 29.8 → 20.6 GiB；**head 冻结（LoRA）的 4B / 8B 上 step 慢 1–5%**，因为 kernel 在 backward 重算 logits tile。配套 #7406 把 entropy 等可选输出做成 Triton 编译期开关，reference model 不再为 policy 独有的统计量付费。
- 其余：#7567 / #7568 支持 vLLM 0.31.0、放弃 0.20.2。

### vllm-project/vllm-ascend（9 个 commit）

- **#17578 EPLB 的 expert 权重迁移改走 HIXL（默认）**｜https://github.com/vllm-project/vllm-ascend/commit/0e181fc85fff24db10e5f92f10319c1913e6e9c5
  - 用官方 CANN HIXL Python binding：**一次性注册** expert 权重内存，按 peer 批量发起 receiver-initiated 的 NPU-to-NPU 读取；不新增 native bridge。保留 `communicator="torch_gloo"` 回退。STAIR 的迁移上限默认改为 `-1`（无限制），配确定性线性源选择器。
  - **前台绝不循环等待传输**：未完成的迁移推迟到后续 model step，每 step 只做一次 readiness 检查。
  - **验证**：双主机 DeepSeek-V4-Flash W8A8 MTP，**DP4 × TP8 × EP32**，三次 256 请求压力共 768/768 完成，mean TPOT 37.67 ms；前台 readiness 等待约每 cycle **1.8–2.6 ms**，迁移本身跨数百个 step。没有与 Gloo 的直接对比数字。
  - **对我的意义**：HIXL 第一次进入一个上游项目的默认数据路径，"一次注册 + receiver-initiated 批量读"的用法是可参考的样板；训练侧的动态 expert 重排、RL 权重推送都可以评估同一通道。默认路径依赖 CANN 发行版里的 HIXL Python 包，升级前确认。
- **#17726 TurboQuant 4-bit latent KV cache**｜https://github.com/vllm-project/vllm-ascend/commit/72f6b4760c79b58445cc3770f45f928881d1961a
  - `--kv-cache-dtype turboquant_4bit_nc`。每 slot 386 B = int4 MLA latent（512 维，256 B）+ bf16 rope（128 B）+ scale（2 B）；C8 为 656 B，bf16 为 1152 B。
  - GLM-5.2 W4A8C8、8×910B4、TP8 + EP、KV 预留 6.87 GiB：KV 池 76,288 → **204,160 token（2.68× bf16，1.59× C8）**。128K 输入 / 2K 输出：并发 1 时输出吞吐 46.03 → 34.63 tok/s（**−24.8%**），并发 2 时 46.78 → 57.56（+23.0%）。GPQA-Diamond 90.40%（无同配置 bf16 对照）。
  - **对我的意义**：收益完全来自容量换并发，不是单请求加速——长上下文 RL rollout 这类 KV 为瓶颈的场景才值得开。上期 #17676 是 DeepSeek V4 的 INT4 KV，这条把 4-bit 扩到 MLA latent。
- **#17947 PCP：短 prefill 下 SFA O-projection gather 不一致导致 hang**｜https://github.com/vllm-project/vllm-ascend/commit/eee24e2e634bc51f2d3b08f3989f2b3f006a6c51
  - PCP 可能把短 prefill 切成单 token 的本地分段，各 rank 的本地 attention state 不同（9-token 请求下 PCP0 是 `PrefillNoCache`、PCP1–7 是 `DecodeOnly`）。SFA 用**本地** state 决定是否 gather 被切分的 O-proj 权重，部分 rank gather、部分跳过，集合通信序列错位。修法是把全局量 `pcp_has_global_prefill` 纳入判断。
  - 验证：8×A5、GLM-5.1 W4A4、TP1 × PCP8，MTP 17/17、无 MTP 5/5 通过，输入长度刻意覆盖 8/9/10/15/16/17/32/256。CANN 9.2.0、torch_npu 2.10.0.post4。
  - **对我的意义**：这是上期 #17598（按 PCP 切分 O-proj 权重，默认开启）引入的回归面。**如果已跟进 #17598，这条必须一起取。** 通用规则：任何"是否参与集合通信"的分支都必须基于全局一致的量。
- **#17956 hybrid cache 页重分配后恢复 C8 V-scale**｜https://github.com/vllm-project/vllm-ascend/commit/9d56044a9bc1c7629fc52354d799ff534bf38008
  - Qwen3.6-27B W8A8+C8：启动后第一个请求正常，之后的请求输出 128 个 `!`。full-attention 与 GDN / Mamba cache group 复用同一批物理页，C8 MXFP8 的 V-scale 在 cache setup 时从 checkpoint 初始化，**GDN state 写入会覆写它的 backing bytes**（地址追踪显示 4 个 C8 packet 里 2 个重叠），页再分给 full attention 时 scale 已损坏，有限输入产出 NaN。修法是在页清零路径之后恢复 checkpoint scale。无端到端修后数字。
  - **对我的意义**：低比特 KV + hybrid 架构的组合必查项——静态量化参数与运行时状态共页时，页重分配要重新恢复前者。
- **#17680 MRV2 的 UVA buffer 映射为 NPU typed view**：用 `aclrtHostGetDevicePointer` 把已注册的 pinned CPU 张量直接映射给 NPU，省掉 metadata / state buffer 的 H2D 拷贝。Qwen3.6-35B-A3B、TP2/PP2 上**每 chip 进程内存少 3,940 MB**，KV 容量不变，吞吐无实质变化。作者自己定级为 implementation candidate：完整 ACL Graph replay 与并发压力未过；双 NPU 探测在同进程先于 Triton 读取运行时重现过 ACL 507035 故障。
- 其余：#17940 MLA 的 PCP 与 DCP 叠加（测试一节为空）；#15614 IndexShare / IndexCache 的 Top-K indices 跨 PP stage 传输（此前只传 hidden states，下游 stage 会用到陈旧索引；测试一节为空）；#17934 MRV1 GQA cache 改零拷贝 `(K, V)` tuple，Mooncake V1 对 strided view 改按物理 span 注册；#17773 GLM-5.3-Flash 的 1P1D PD 与 KV cache pool 部署文档（作者注明未做 NPU 实测）。

### NVIDIA/Megatron-LM（10 个 commit）

- **#7287 prepended-axis shard 走 checkpointable 路径**｜https://github.com/NVIDIA/Megatron-LM/commit/e4b048e8c87860677d0bb361d0833e6bb1988572
  - `mcore_to_pyt_state_dict` 对所有带 prepended axis 的 `ShardedTensor` 走 legacy 转换，而后者对每个 key 枚举**整个全局 shard 网格**。`TransformerBlock` 加一个 layer 轴、`SequentialMLP` 再叠一个 global-expert 轴，48 层 × 128 experts、EP=8 时每 rank 合计 4,734,725 个网格条目，且全部**同步发生在 async write 返回之前**。
  - 8×H200、`torch_dist`：`mcore_to_pyt_state_dict` **11.71 → 0.07 s**，`async_save()` 同步前缀 **17.75 → 5.22 s**。DCP metadata 与 legacy 完全相同，新旧 checkpoint 双向兼容。
  - **对我的意义**：万卡 MoE 可以直接用。自查方法是量 save 的同步前缀耗时——这部分异步 checkpoint 藏不住，成本随"层数 × expert 数"涨，与本地 shard 数无关。
- **#7562 batch-invariant hybrid MoE 的训推 bitwise 对齐**｜https://github.com/NVIDIA/Megatron-LM/commit/1abb04855531f33df5d4843489581ba0be0bab7a
  - Nemotron Nano v3 30B-A3B GRPO 三步的 Generation KL 为 0.0025 / 0.0041 / 0.0025——**平坦而非增长**，作者据此判断是 per-token 算术偏置而非累积误差。两处根因：推理侧 squared-ReLU kernel 在乘 router 概率前多舍入一次 BF16（训练侧融合实现的中间量保持 FP32）；Mamba 训练走融合 kernel 并在 scan 之后施加 gate，推理把 gate 送进 scan，代数等价、算术不等价。修后三步 KL 全为 0。
  - 代价：该模式下 hybrid 模型不能用 packed sequence，也不能用融合的 Mamba 训练 kernel。另一个陷阱：`MAMBA_DETERMINISTIC` 在 import 时就被 `@triton.autotune` 固定，事后设环境变量不生效但看起来生效。
  - **对我的意义**：推理侧重写的 kernel 要复刻训练侧的**舍入序列**，不只是数学表达式。"KL 平坦对增长"是区分两类偏置的实用判据。
- **#6890 融合 MLA RoPE 的 per-head bounds mask**｜https://github.com/NVIDIA/Megatron-LM/commit/442d0183d608092819844c6b0897ffd8b3adfb2e：基指针已前移到本 head block，mask 却还在比较块内局部索引，恒为真；`BLOCK_H` 不整除 head 数时最后一个 program 会旋转下一个 token 的前几个 head 或越界写。**潜在 bug**：已发布 MLA 模型的 head 数都是 2 的幂，触发需要 96 / TP=8 这类配置。验证只在单张 L4 上做，作者明确不做性能主张。"同输入换 block size 比 bitwise"是自研融合 kernel 很便宜的回归手段。
- **#7802 MCore 支持 scale learning（NVFP4 LSQ）**：`--lsq-scale-lr` 给可学习的 NVFP4 scale 单独学习率。附带修了一个 EP bug——scale 参数没有 `allreduce` 属性，**expert scale 的梯度在 data-parallel group 而不是 expert-data-parallel group 上归约**。新增参数不自动继承并行语义，低比特 QAT 引入新参数时要显式检查通信域。
- **#7927 GTP wgrad scratch 跨 stream 复用加 fence**：`Work.wait()` 只对 RS stream 排序，scratch 还回 pool 后可能被另一条 stream 在读取完成前覆写；改为 buffer 带 completion event、checkout 时等待。PR 无数字、未做 functional test。
- 其余：#7800 运行时检测 CUDA graph generator 的懒注册（无细节）；#7933 推理 API 参数。

### vllm-project/vllm（58 个 commit；细节来自 PR 描述，经子 agent 提取）

- **#59164 MoRIIO：同步 READ 的目的块不再被清零**｜https://github.com/vllm-project/vllm/commit/df417f780a19b68cebc4379942de6ea0e6f6aceb
  - hybrid 模型 PD 分离下**静默精度崩塌**：GSM8K 全量 main 为 265/1319（20.1%），修后 1268/1319（96.1%）。根因是同步 RDMA READ 与"回收 attention page 的 GPU 清零"竞争，清零可能覆盖刚读入的 KV。配置为 Kimi-K3 + DSpark3，P8/D8。作者自述模型评测跑的是重构前的实现，不是最终 revision。
  - **对我的意义**：块回收清零与外部 KV 写入之间必须有明确的所有权和时序契约，昇腾 PD 分离（Mooncake / HIXL）同样适用。不 crash，只掉精度。
- **#59197 DSv4.1：P/D 已传过来的 KV 不再做 SWA bounded replay**｜https://github.com/vllm-project/vllm/commit/ef63c23d35acccbba8e014fc88da19d541da38a9：replay 的整块取整让 decode 分配小于 prefill 导出，N % 128 ≠ 0 时 NIXL 把 prefill 块 k+1 放进 decode 块 k。修法包括让 D 侧自算最后一个 prompt token。精度差异在噪声量级，主要收益是消除错位与省掉 128 token 重算。
- **#58124 `reload_weights` 支持 runai_streamer**：对象存储 URI 会把 load format 强制解析为 runai_streamer，此前 sleep level 2 后热重载直接 `NotImplementedError`。**另一个静默 bug**：`reload_weights(weights_path=...)` 改了 `model` 却没清 `model_weights`，继续从旧 bucket 加载原权重。Qwen2.5-14B 冷重载：本地 XFS 上 streamer 23.9 s 对 default 112.7 s；**NFSv4.1 上 streamer 默认配置反而慢于 default**（113.9 s 对 89.0 s）。
- **#60210 MRV2 + PP + MTP 下 Mamba align 状态拷贝读到别的请求的 block**：延迟执行的 postprocess 引用了每 step 按 batch 顺序重写的缓冲，表现为全 NaN logits。Qwen3.6-27B、PP=2：平均接受长度 2.95 → 4.13，输出 1177 → 1506 tok/s。通用规则：延迟执行的后处理按稳定的 request slot 索引。
- **#58373 `ep_gather` 输出偏移加宽到 int64**：`cur_token * stride` 用 int32，输出超过 2^31 / hidden_size 行即回绕越界写（K=6144 时为 349,525 行）。PR 的测试一节为空。WideEP 大 batch 下自研 EP gather / scatter 的行偏移值得自查。
- **#46963 NVFP4 KV cache 扩到 SM8x / SM12x**：KV 容量 3.32×，但 GSM8K 上 Qwen3-0.6B 由 0.384 掉到 0.174，Qwen3-4B 由约 0.92 到 0.88。4-bit KV 的精度损失强烈依赖模型规模。
- 另：#60192 去掉 CI 里强制的 `VLLM_BATCH_INVARIANT` 并把 ragged prefill 打分容差放宽到 0.2（默认配置下 logprob 不是 batch-invariant 的又一个佐证）；#55053 PD + MTP 时 prefill 侧为投机 token 预留的 lookahead 块导致 KV 写入长度不匹配与 hang。

### sgl-project/sglang（56 个 commit；细节来自 PR 描述，经子 agent 提取）

- **#42585 重载 FP8 权重时先 shard 再量化**｜https://github.com/sgl-project/sglang/commit/b83827ea02618332f4bc8b60dec92baec80a3c37
  - 重载**未改动**的 checkpoint 也会改变 FP8 权重和 scale：量化发生在原生 TP sharding 之前，scale 更新用了错误的 partition。Qwen3-0.6B、TP4 / attention-DP2，重载后 GSM8K 子集 **45% → 1.5%**。修后重载与冷启动的 21,082 个生成 token logprob 完全一致；额外峰值内存 350.71 → 16.63 MiB。作者自述 rebase 后未重跑 GPU 测试，且只在小模型上验证。
  - **对我的意义**：RL 在线权重热更新 + 在线量化 + TP 的顺序必须是"先 shard 再量化，scale 按本 rank 分区"。"重载后与冷启动逐 logprob 比对"是很好的回归门禁。
- **#39923 PCP 与 FP8 unified KV 组合（AMD gfx950）**：每个 rank 的本地 K 打包成 FP8 + 内联 E8M0 scale 的 raw `uint8` 行跨 CP 组 gather。8×MI355X、DeepSeek-V4-Pro + EAGLE：PCP8 + decode-attn-TP 相对 TP8 在并发 16/32/48 下吞吐 +15.8% / +13.8% / +13.9%，C16 TTFT 1852 → 1196 ms；并发 1 时仍低 4%。
- **#42611 融合 KDA 的 beta sigmoid 与 `torch.sigmoid` 逐位一致**：`tl.sigmoid` 在全部 2^32 个 fp32 输入里有 9570 万个与 `torch.sigmoid` 不同，而 KDA 状态自反馈、误差沿序列放大——GLM-5.3-Flash-NVFP4 teacher forcing 下 mean |Δlogp| 0.240 → 0.000。融合的 KDA projection 与分开的 projection 仍不一致（0.255），未修。递归 / 线性注意力里，融合 kernel 换超越函数的近似实现要按逐位一致来验。
- 另：#42426 等 9 个 PR 的栈把 TP / attn-TP / replicated 的分区选择下沉到线性层（`parallel_group=`，对下游 fork 是接口变化）；#42822–#42825 删除 `Req.prefix_indices`（对 fork 调度器 breaking）；#42399 HiCache 新增 SeaweedFS L3 后端——Qwen3-4B 上约 25k token 才打平重算，瓶颈是 Python S3 客户端。

### pytorch/pytorch（77 个 commit，按分布式 / 编译关键词筛选）

- **【更新】Revert #192524（ROCm 上的 NCCLSymmetricMemory / RCCL）**｜https://github.com/pytorch/pytorch/commit/2eb73612d372439ded791f310bc7ccf085a4ea13：上期收录的这条已回滚，理由原文 "it has some internal failure w.r.t RCCL version"。上期"非 CUDA 后端接入 symmetric memory 进入主干"的判断需改为"待 reland"。
- **#199729 [FSDP2] resharding 时丢弃 pending 的 post-forward all-gather**：`reshard_after_forward` 为 int、某 group 前向用到而反向没用到、且处于梯度累积的非最后一次 backward 时，下一次 forward 会把较小 mesh 上的 all-gather buffer 当成 full mesh 拷出，报 `start (0) + length (512) exceeds dimension size (256)`。4×B200 上 8 个测试通过。
- **#197204 [FSDP2] 新增原生 collective copy op**：`_all_gather_copy_out_` 与 `_reduce_scatter_copy_in_` 直接在 collective buffer 与参数 / 梯度布局间拷贝，省掉 split+cat 中间缓冲，copy-in 可把 bf16 cast 进 fp32 buffer。FSDP2 目前还不调用（opt-in 在 #197701），无性能数字。**只有 CUDA kernel，其他后端走 composite 回退**——torch_npu 想在启用后拿到收益需要自己实现。
- **#199637 自定义 MemPool allocator 回调不再在 caching allocator 锁内调用**：线程 A 持 GIL 等 allocator mutex、线程 B 持 mutex 调 Python 回调等 GIL 的死锁。用 MemPool 挂自定义分配器（symmetric memory、RDMA 注册内存）的直接相关。
- **#199924 [c10d] 修 Python 子类化 `Backend` 时的属性无限递归**：未定义 `supports_splitting` 等能力属性时读取即递归到栈溢出，`dist.split_group` 不带 `pg_options` 也会踩到。用 Python 包一层 HCCL 做自定义 c10d backend 的直接相关。
- 另：#198845 把 gfx950 的 MXFP8 grouped GEMM（FlyDSL）接入 Inductor autotune，是非 NVIDIA 后端以 template 形式接入厂商 DSL kernel 的范例；两个 Inductor 的 perf PR（#198809、#199530）当日被回滚。

### NVIDIA/TransformerEngine（3 个 commit）、TensorRT-LLM（18 个）、其它

- **TE #3637 带 expert bias 的融合路由保留极小 sigmoid score**｜https://github.com/NVIDIA/TransformerEngine/commit/5e994640625626263aa62250c709e35601563566：原实现在 top-k 后用 `(score + bias) - bias` 恢复分数，加 bias 会把极小 score 舍掉、减法恢复不了。改为从未加偏置的中间量 gather。PR 无数字。DeepSeek-V3 式 aux-loss-free 的 expert bias 方案都有这个面。
- TE #3546 `LayerNormLinear` 的 Userbuffers all-gather overlap 在 no_grad 下失效（无细节）——RL 的 rollout 与 reference 前向全在 no_grad 下，等价的通算重叠机制值得查一眼这条路径。
- TensorRT-LLM #19899：multi-stream 调度器的依赖图只要求 in-place 写排在所有 writer 之后，没有排在该张量所有 **reader** 之后，eagle3 的 hidden states 被覆写、接受长度下降；修法是按 storage root 跟踪 reader。
- DeepSpeed 1 个 commit（纯 CI）。**Ascend/pytorch（torch_npu 镜像）2 个 commit，均为 API 测试用例补充**，其中一条覆盖了 `checkpoint(use_reentrant=True)` 含 Dropout 时的 NPU RNG 状态保存 / 恢复。
- **MindSpeed / MindSpeed-LLM / MindSpeed-RL：未确认**（Gitee 返回 robots 限制、GitCode 返回 418）。

---

## Sources

- Claude Haiku 5.5：https://www.anthropic.com/claude-haiku-5-5
- GPT-6 10 月版：https://openai.com/index/gpt-6-for-everyone/ ｜ https://deploymentsafety.openai.com/gpt-6-october
- NCCL M2N：https://arxiv.org/abs/2610.07516
- NeMo-DCR：https://arxiv.org/abs/2610.08430
- Expert Coupling（Zyphra）：https://arxiv.org/abs/2610.09372
- RelayMoE：https://arxiv.org/abs/2610.07333
- TRACE：https://arxiv.org/abs/2610.07767 ｜ TRIAGE：https://arxiv.org/abs/2610.07043
- TRANSIT：https://arxiv.org/abs/2610.07593 ｜ CoMoE：https://arxiv.org/abs/2610.09424
- vLLM Blog（DeepSeek-V4.1-Flash）：https://vllm.ai/blog/2026-10-07-deepseek-v41-flash
- OCP Global Summit 2026：https://www.opencompute.org/events/ocp-summit/2026-ocp-global-summit/
- torchtitan：[#5123](https://github.com/pytorch/torchtitan/commit/0167526a9fdd67ad6b0c3df4c739d6a148c582e6) ｜ [#5086](https://github.com/pytorch/torchtitan/commit/948d65c868c5fa8f0290bcf9e54b69f004721f54) ｜ [#5109](https://github.com/pytorch/torchtitan/commit/1472b0714f2a2500a4bc47728b1c30b350100de7) ｜ [#5097](https://github.com/pytorch/torchtitan/commit/98e7b95010467eee5a82da8f783b5951f1e67b85) ｜ [#5103](https://github.com/pytorch/torchtitan/commit/ee1c2eaaee3f12cab780ba7d7626c2d97c437367) ｜ [#5105](https://github.com/pytorch/torchtitan/commit/6113f19ea656900eaf50e0404fbfc68cd86914f6) ｜ [#4821](https://github.com/pytorch/torchtitan/commit/4b1f9cea10bc2157d76cc9c4443cd78ba87f3a79)
- slime：[#2443](https://github.com/THUDM/slime/commit/f0f6e5545fcc534b13bfab8568e518c9dd450f2a) ｜ [#2444](https://github.com/THUDM/slime/commit/4a7a50909ac310b7b23caf851ef05a2af26baa74) ｜ [#2442](https://github.com/THUDM/slime/commit/2f2318653f6f794dddd321eff7c9d4b7b174643f)
- trl：[#7549](https://github.com/huggingface/trl/commit/9bb31ab972fb21f218f9ebf10c2816037270cd82) ｜ [#7390](https://github.com/huggingface/trl/commit/ba406309c7558fe0b7db05cd5ec2a9790a1d1b42)
- vllm-ascend：[#17578](https://github.com/vllm-project/vllm-ascend/commit/0e181fc85fff24db10e5f92f10319c1913e6e9c5) ｜ [#17726](https://github.com/vllm-project/vllm-ascend/commit/72f6b4760c79b58445cc3770f45f928881d1961a) ｜ [#17947](https://github.com/vllm-project/vllm-ascend/commit/eee24e2e634bc51f2d3b08f3989f2b3f006a6c51) ｜ [#17956](https://github.com/vllm-project/vllm-ascend/commit/9d56044a9bc1c7629fc52354d799ff534bf38008) ｜ [#17680](https://github.com/vllm-project/vllm-ascend/commit/ec570818bc428da1e3dc1820434cbadf2d5cf300) ｜ [#17940](https://github.com/vllm-project/vllm-ascend/commit/3800d9ec71bf635439e014668697891092d5db19) ｜ [#15614](https://github.com/vllm-project/vllm-ascend/commit/fe85f2bc4a652a0323b53c7cc4a22376f5c9d65e)
- Megatron-LM：[#7287](https://github.com/NVIDIA/Megatron-LM/commit/e4b048e8c87860677d0bb361d0833e6bb1988572) ｜ [#7562](https://github.com/NVIDIA/Megatron-LM/commit/1abb04855531f33df5d4843489581ba0be0bab7a) ｜ [#6890](https://github.com/NVIDIA/Megatron-LM/commit/442d0183d608092819844c6b0897ffd8b3adfb2e) ｜ [#7802](https://github.com/NVIDIA/Megatron-LM/commit/965e9068c77be47ed5a88424652f7b07be14d363) ｜ [#7927](https://github.com/NVIDIA/Megatron-LM/commit/9de9bde025fea4b5d72fa78a0b94d4d0f3c85c48)
- vllm：[#59164](https://github.com/vllm-project/vllm/commit/df417f780a19b68cebc4379942de6ea0e6f6aceb) ｜ [#59197](https://github.com/vllm-project/vllm/commit/ef63c23d35acccbba8e014fc88da19d541da38a9) ｜ [#58124](https://github.com/vllm-project/vllm/commit/c83935b3008553c72b1bad09f842b4c99b7fa47f) ｜ [#60210](https://github.com/vllm-project/vllm/commit/8a26869bcad1ae4ab3069a7d2aece86c17c1c695) ｜ [#58373](https://github.com/vllm-project/vllm/commit/283d76d56a45d9bbead5ba650402fb0957abc30c) ｜ [#46963](https://github.com/vllm-project/vllm/commit/554340f3d3259e321be4c07282be7a02a5aeef83)
- sglang：[#42585](https://github.com/sgl-project/sglang/commit/b83827ea02618332f4bc8b60dec92baec80a3c37) ｜ [#39923](https://github.com/sgl-project/sglang/commit/9ddbba50dd8c8f1a05b2970a997350d9c5844e91) ｜ [#42611](https://github.com/sgl-project/sglang/commit/012cd3858b5f1a57e32cfc78ba3f787ee7de8786)
- pytorch：[Revert #192524](https://github.com/pytorch/pytorch/commit/2eb73612d372439ded791f310bc7ccf085a4ea13) ｜ [#199729](https://github.com/pytorch/pytorch/commit/75bf453412) ｜ [#197204](https://github.com/pytorch/pytorch/commit/90ef23c2ba) ｜ [#199637](https://github.com/pytorch/pytorch/commit/31a78370bb) ｜ [#199924](https://github.com/pytorch/pytorch/commit/ab4b2a20dc)
- TransformerEngine：[#3637](https://github.com/NVIDIA/TransformerEngine/commit/5e994640625626263aa62250c709e35601563566) ｜ TensorRT-LLM：[#19899](https://github.com/NVIDIA/TensorRT-LLM/commit/b57220da7b7a0757359ad769a304a0abcac85836)

## 运行备注

- **窗口**：严格取北京时间 10-07 08:00 → 10-08 08:00。本次实际运行于 09:47（计划 08:08），08:00 之后的内容未收。
- **去重**：已对照 10-06、10-07 两期，无重复条目。标【更新】的一条：pytorch #192524 被回滚。更正一处：OCP Summit 止于 10-15 而非上期写的 10-16。
- **论文日期口径**：2610.07516 / 07333 / 07043（v1 戳 10-05）与 07767（10-06）属 10-07 00:00 UTC 公告批次，其中仅 07043 经 arXiv cs.LG 公告页直接确认，其余由编号段推断。**NeMo-DCR（2610.08430）推断在 10-08 00:00 UTC 公告，正好压在窗口右边界**，HF Daily Papers 列在 Oct 7 版面，本期按"偏向收录"处理。Zyphra（2610.09372）v1 戳 10-07，按提交时间收录。
- **验证强度（请按此折扣阅读）**：
  - 本会话**没有直接读任何一手原文**，全部内容来自 5 个子 agent 的提取报告，本会话只做了去重、交叉比对和筛选，**未逐条复核原文**。
  - 6 篇精读论文子 agent 读的是全文正文（附录未读）；TRANSIT、CoMoE 只读了引言；"未精读"清单多数只有标题或摘要首句。
  - Megatron、torchtitan、slime、trl、vllm-ascend、TE 的细节来自 commit message 或 PR 描述原文；vllm、sglang、pytorch、TensorRT-LLM 的 commit message 基本只有标题，细节来自 PR 描述。
  - vLLM 博客与两条模型发布经 WebFetch 的小模型摘要提取，数字未逐字核对。Haiku 5.5 的 benchmark 表里有两项数值相同（疑似摘要串行），本日报未引用那两项。
  - 无实测数字、已就地标注的：torchtitan #5097（命中率对比留空）、#5088、#4821；Megatron #7927、#7800；vllm #58373；vllm-ascend #17940、#15614（测试一节为空）、#17956（无端到端修后数字）；TE #3637、#3546。
  - 作者自述验证与最终代码不一致的：vllm #59164（评测跑的是重构前实现）、sglang #42585 与 #42426 栈（rebase 后未重跑）。
- **检索失败或受限**：
  - HF Daily Papers 的 `?date=` 参数返回 400；主页成功，实际路径形如 `/papers/date/YYYY-MM-DD`。10-08 版面未获取。
  - `arxiv.org/list/cs.DC/new` 与 `cs.AR/new` 返回的是滞后一到两天的旧页面，**窗口内 cs.DC / cs.AR 新提交未能直接核对**，本期系统类论文全部来自 alphaXiv。cs.LG / cs.CL 公告页各只读了前约 44 条。
  - alphaXiv 只做了 2 次主题检索（工具限额），结果被 MoE 与 RL 占满。**优化器、FP8/MXFP8 预训练、SDC / checkpoint、训练稳定性、数据、奖励模型、异步 RL staleness、PD 分离与 serving 调度这些主题检索不足，不能断言"无"。**
  - 博客列表页只返回导航壳：LMSYS/SGLang Blog、Modal、Anyscale、昇腾社区（hiascend.com）、机器之心。华为计算官网未抓。Google Cloud Blog 列表无日期。
  - 硬件侧 Google TPU、AWS Trainium、寒武纪、海光没有抓到厂商 newsroom，仅靠搜索；Intel Newsroom 页面疑似陈旧，Gaudi 侧结论不可靠。
  - 国内厂商（Moonshot、MiniMax、字节、腾讯、百度、阶跃、小米、美团）官网 / 公众号未直接核查，只有 HF 与 GitHub 的否定证据加两轮中文搜索——只在公众号发布的内容会漏。DeepSeek 的 API news 页无带日期条目。
  - NVIDIA GitHub org 的 pushed 结果只看了前 60 条。
  - MindSpeed 系列：Gitee `ROBOTS_DISALLOWED`、GitCode HTTP 418，未确认。torch_npu 的正式版本信息在 Gitee / GitCode 侧，GitHub 镜像不发 Release。
- **窗口外但值得补读的线索**：
  - **NVIDIA Technical Blog 10-06《How DOCA GPUNetIO Unifies GPU-Initiated Networking Across the NVIDIA Software Stack》**——上期漏收，与集合通信直接相关：CUDA kernel 内直接 post WQE、敲 doorbell、poll CQE，CPU 只做控制面；NCCL 自 2.27 起 GIN 用它做后端，小消息 reduce-scatter 收益最大、约 2 MB 起下降。可对照 HCCL 经 host 代理下发 RoCE 的路径。https://developer.nvidia.com/blog/doca-gpunetio-gda-ki-unified-gpu-networking/
  - PyTorch Blog 10-06《Modernizing Table Batched Embeddings with FBTriton》（推荐系统，相关度中等）。
  - Ant Ling-3.1-flash（约 10-02，二手：约 560B 总参 / 25B 激活、"7 层 KDA + 1 层 Gated MLA"，未回溯一手）；EmbeddingGemma 2（Google，10-06）。
- **下期线索（10-08 00:00 UTC 之后已合入）**：
  - **vllm-ascend #17961**：A5 MXFP8 O-proj TP 与 MRV2 DSACP serving 的四项修复，Ascend 950 上 64/64 用例最大绝对差 0；GSM8K 96.51%，但有 2 个样本跑到 32768 token 输出为空。A5 + DSA 部署必读。
  - torchtitan #5111（FFN depth-scaled 初始化只作用于 `w2`，**有意改变从零训练的初始化数值**，收敛验证待做）；"Paged Stash"（可 CUDA graph 化的 MoE + 分页激活暂存，无细节）；#5135（MXFP8 FSDP 单测此前在 CI 里从未执行过）。
  - vllm #60389（权重重载时 MLA 派生权重保持原地）；TensorRT-LLM #19643（PP sample-state relay 线程上保持 MPI progress，hang 类）；pytorch #199128（`torch.cuda.execute_on_streams`）。
- **保存结果**：
  - GitHub：随本次提交写入 `suhaibo666/tracker` main 分支 `llm-training-daily/llm-training-daily-2026-10-08.md`（新建文件）。
  - 本地：**未写入**。`device_commit_files` 报 "Files can't be written here without a grant"——与前两期原因相同，设备桥接在线，但这个定时任务的会话**没有任何已连接文件夹**。按要求未重试，也未发起文件夹授权请求（无人值守时没人应答，且授权只对单次会话有效）。这已是连续第三期本地备份失败。
  - **一次性修复**：在该电脑的 Claude 桌面端把 `~/workspace/90-knowledge/llm-monitor`（或上层 `~/workspace`）加为这个定时任务的连接文件夹。在此之前本地副本请从 GitHub 拉取。
