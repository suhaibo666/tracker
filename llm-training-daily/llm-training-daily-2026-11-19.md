# LLM 技术动态日报 · 2026-11-19

> 时区 Asia/Shanghai（UTC+8）。增量扫描窗口 **(2026-11-18 09:00:05, 2026-11-19 09:04:03]**，向前回溯 72 小时至 **2026-11-15 09:00:05**。实际生成 **2026-11-19T09:18:58+08:00**。来源仅有日期时保留原日期，提交、发布、会程与发现时间按各自证据记载。

## 本次重要变化

- **模型与训练披露。** Gemini 4 Neon 的同模型 SWE-sweep 成绩随 harness 从12.8%到39.0%，需要把模型与执行系统拆开评价；Hy4 flash 新权重和 Aion 2.3B 端侧工件提供可本地验证的增量。TeleChat→Xing4 报告须区分开放29B成员与285B、8192卡系统主例。[Neon方法](https://deepmind.google/models/evals-methodology/gemini-4-neon) · [Hy4卡片](https://huggingface.co/tencent/Hy4-flash/blob/d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959/README.md) · [Xing4报告](https://arxiv.org/abs/2611.22115v1)
- **训练方法。** Ambient SFT 研究高温基座分布匹配以减少小数据遗忘；Regularized Score Centering 针对异步策略漂移；TPHC 与 SPICE 分别探索残差流的并行分工和动态peer稀疏训练，均需保留规模与成本边界。[Ambient SFT](https://arxiv.org/abs/2611.23229v1) · [RSC](https://arxiv.org/abs/2611.21791v1) · [TPHC](https://arxiv.org/abs/2611.23026v1) · [SPICE](https://arxiv.org/abs/2611.23053v1)
- **训推工程。** verl 用权重版本salt隔离外部KV；Megatron调整MFSDP专家优化器分组与GTP虚拟流水的prefetch链；TorchTPU修复设备分配和matmul精度变化后的编译缓存复用。对应配置应先回归正确性。[verl #8323](https://github.com/verl-project/verl/pull/8323) · [Megatron #9504](https://github.com/NVIDIA/Megatron-LM/pull/9504) · [TorchTPU](https://github.com/google-pytorch/torch_tpu/commit/2aa3ee53c03af388fdc62783aae6a236ccde4942)
- **部署升级。** ROCm10.2与TensorRT-LLM1.4.0rc4有可核验发布；FlashInfer0.7.5正式tag与今日main新增确定性、GDN分段、状态池机制必须分开。确定性内核和NVLS原语的证据同时揭示性能代价，不能只摘最高加速数。[ROCm10.2](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-10.2-blog/README.html) · [TRT候选版](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.4.0rc4) · [FlashInfer0.7.5](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.7.5)
- **更正优先。** CANN撤回ChunkLoss Torch接入而非底层算子；PFN搜索语料删除未进入出题索引的266万英语文档，旧索引需空目录重建；SingVL纠正图像预算对应分数。这些会直接改变复现和采用结论。[CANN撤回](https://gitcode.com/cann/ops-nn/merge_requests/14712) · [PFN更正](https://huggingface.co/datasets/pfnet/snapshot-browsecomp-search-corpus/blob/07ca0e53c2dedfc6a43d403673e51b52856dc711/README.md) · [SingVL更正](https://huggingface.co/inclusionAI/SingVL_S1_4B/blob/c7a7f32ab027735b7dd2742e5b6cfcacaa43f97a/README.md)

[模型与技术报告](#topic-1) · [预训练与架构](#topic-2) · [后训练](#topic-3) · [推理与部署](#topic-4) · [开源工程](#topic-5) · [硬件与软件协同](#topic-6) · [评估与可靠性](#topic-7) · [覆盖与续查](#coverage)

<a id="topic-1"></a>
## 模型与技术报告

### Gemini 4 Neon：四档推理的长程 Agent 模型，模型分数与 harness 收益应分开读

**机构／日期／版本：** Google DeepMind，2026-11-18；`gemini-4-neon` GA，模型卡同日。官方 API 更新确认 `low`、`medium`、默认 `high`、`xhigh` 四档思考，文本、图像、音频、视频共享 **1M 输入输出总上下文**。模型卡进一步区分默认同步输出上限 64K 与后台 long decode 最高 1M，不能把后者当作默认单次输出。API 标价为输入／输出每百万 token 3／15 美元，缓存输入 0.15 美元。

**具体增量与证据：** 新模型从 Gemini 4 Carbon 适配，面向复杂编码、分析和工具任务。方法文档说明，Terminal-Bench 4.1 的 65.2% 为 Artificial Analysis 使用 Antigravity CLI 1.5.9 得到的作弊调整准确率；SWE-sweep 则是 Google 自测，同一 Neon 在 mini-swe-agent vision 下为 **12.8%**，在支持子 Agent 并行的 Antigravity CLI 1.5.9 下为 **39.0%**。这说明完整执行系统对该项结果影响很大，不能把 39% 归为纯模型增益。方法页还区分最高思考档默认、FrontierCode 的 high 档和若干指标引用的第三方结果。

**限制与实践意义：** 当前卡片把架构、训练数据和软硬件说明转引 Carbon，未披露 Neon 独立参数量、数据配比、训练算力或消融；完整安全报告仍写“未来几天提供”，权重未开放。**建议深入阅读并用现有任务做成本／成功率对照**，优先固定思考档、工具集、子 Agent 预算和时限；本轮未调用付费 API。[发布](https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-gemini-4-neon/)、[模型卡](https://deepmind.google/models/model-cards/gemini-4-neon/)、[评测方法](https://deepmind.google/models/evals-methodology/gemini-4-neon)、[API 更新](https://ai.google.dev/gemini-api/docs/changelog#11-18-2026)

### Gemini API：Office文件和切片接口改变多模态预处理边界

**机构／日期／版本：** Google，2026-11-18 API changelog。新增直接传入Word、PowerPoint、Excel及OpenDocument，并允许在用户提示和function response中指定页范围、sheet／cell范围、音视频时间段或内容层（页面图像、提取文本、公式、转录）。价值是把预处理／全文件token开销收敛到任务所需片段；具体文件类型、内容层和计费行为仍须按相应文档与实际请求核验，不能据API承诺推断解析无损。**建议考虑采用并做格式保真回归**，尤其核对表格公式、分页和时间戳边界。[正式更新](https://ai.google.dev/gemini-api/docs/changelog#11-18-2026)

### 腾讯 Hy4 flash：本轮可核验 Instruct、FP8、Base 权重及 DFly2 结构

**机构／日期／版本：** 腾讯混元；本轮 2026-11-18 可读工件，Instruct 固定提交 `d88c9dfd`，FP8 `003107fa`，Base `95b9f823`。前期日报已记 API，此次增量是三套权重、配置和部署说明，而非重复 API 发布。仓库创建于 11 月 17 日，不据创建时间推断首次公开时刻。

**具体增量：** 卡片把规模拆开为语言主干 155.5B、Engram 63.5B、视觉编码器 0.43B、DFly2 0.85B；约 **220B 总参数**，文本每 token 激活 11.3B。前期新闻曾写224B／激活12B；当前卡片逐部件口径约220B／文本11.3B（另列视觉0.43B），两者并非完全相同，原始新闻未给出足够计数明细以消除差异，保留这一口径冲突。42 层含 12 个稀疏 MLA 层和 30 个 512 窗口 SWA 层；LatentMoE 每 token 选 512 个路由专家中的 16 个，加一个共享专家。5 层 DFly2 草稿器复用目标稀疏层 KV 和最多 2,048 个位置的索引，文档据此主张无独立草稿 KV，并使草稿注意力访问长度有界。已打开固定配置、权重文件目录和许可证，未下载完整权重或验证运行。

**边界与建议：** 许可证是 **Tencent Hy License v1.0**，包含针对大型 MaaS 业务的额外商业条款，不能标为 Apache-2.0。Transformers 示例明确使用 **PR #50903 的 head**，写明尚待合入；vLLM／SGLang 使用专用镜像标签，不能泛化为全部稳定版本已支持。**建议深入阅读 Base 与 Instruct 的差异，并考虑复现 DFly2 长上下文显存／吞吐**，固定稀疏选择、TP 和镜像版本后再比。[固定模型卡](https://huggingface.co/tencent/Hy4-flash/blob/d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959/README.md)、[Base](https://huggingface.co/tencent/Hy4-flash-base/tree/95b9f8239e2f63cb837c91704f070ae47332ebdf)、[许可证](https://huggingface.co/tencent/Hy4-flash/blob/d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959/LICENSE)

### Microsoft Aion 1.0：2.3B 端侧语言模型及 INT4 WebGPU 工件公开

**机构／日期／版本：** Microsoft Windows Applied Sciences Group，2026-11-18；BF16 `28902cbf`，ONNX-WebGPU `1677687f`，MIT。

**具体增量：** 延续 Phi Silica 数据体系，面向摘要、改写、文档理解等端侧任务；28 层、约 2.29B、GQA，21 个 2,048 窗口 SWA 层与 7 个全注意力层，32,768 token 上下文，八种语言，多语言占训练数据约 6%。除 BF16 自定义 Transformers 实现外，同步开放约 **1.3GB INT4 QAT ONNX**：线性权重按 32 元素块量化，嵌入／共享输出头按 64 元素块量化，激活和 KV 为 FP16，面向 ONNX Runtime WebGPU。

**证据与限制：** 卡片报告 IFEval 83.90、GPQA 28.18 等，并说明使用 lm-evaluation-harness 或官方任务实现；没有给出与上一代同条件的吞吐、功耗或质量对照。正式作为 Windows OS SLM 搭配本地审核模型是 **2027 年 1 月底的计划**，不能写成已在系统全量上线；原始模型与 ONNX 版本也不能共享未测的数值指标。**建议考虑复现离线摘要／改写**，测浏览器／原生 WebGPU 的峰值 KV 显存、长文质量及启动成本。[BF16 卡片](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B/blob/28902cbfd9f4e0347d6c853aa1301f5d559b2fc0/README.md)、[ONNX 卡片](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B-ONNX-WebGPU/blob/1677687f0a7ff7d635dbaf2d00e82f5fa5d42c7b/README.md)

### SenseNova Loop 9B：正式 AWQ／GGUF 工件补齐低显存路径

**机构／日期／版本：** 商汤，2026-11-18；AWQ `64bd16ea`、GGUF `11c12105`。此次是昨日访问缺口后的实际工件核验，非再报 BF16 首发。

**具体增量与边界：** AWQ 目录提供 4-bit 权重及对应 MTP 草稿；GGUF 提供 BF16、Q8_0、Q5_K_M、Q4_K_M。卡片估计 4-bit 权重约 4.9GB，64K／128K／256K 上下文结合 8-bit KV 的最低显存约 10／12／15GiB；这些是部署估计，不是本轮实测。AWQ 指向 **LightLLM 的 SenseNova Loop AWQ 专用分支**，GGUF 路径亦需保持循环层与 mHC 的执行顺序；不能因一般 9B 参数规模而套普通模型支持。GGUF 独立 Apache-2.0 LICENSE 可读，AWQ 此固定目录的独立 LICENSE 路径返回 404，许可应沿上游继续核验。**建议考虑复现单卡低显存部署**，先验证分支、草稿、KV 精度和循环执行契约。[AWQ 固定卡](https://huggingface.co/sensenova/SenseNova-Loop-9B-AWQ/blob/64bd16eae9966636cbd1c060b9854b5c849fb4e3/README.md)、[GGUF 固定目录](https://huggingface.co/sensenova/SenseNova-Loop-9B-GGUF/tree/11c121055062d33647348cf236a8c543138b282e)

### 补录：OVH Olala mini 7A1B 预告 Differential-TPA／Mamba-3 混合架构

**机构／原始日期／状态：** OVHcloud，2026-11-17 官方公告；正式发布安排在 **11 月 19 日**，截至本轮截止点尚未核验对应权重和完整报告。

官方披露 7B 总参数、1B 激活、每层 256 专家，结合 Differential-TPA 与 Mamba-3，BF16、32K 上下文；训练使用 Jupiter、MeluXina、Jean Zay 的 GH200／A100／H100。公告称 RTX A4500 达 160 token/s、CPU 50 token/s，但没有给出精度变体、输入输出长度、批量、CPU 型号及内存驻留策略；“最低 8GB GPU”也不能解读为完整 BF16 权重驻留。权重、Base、技术报告与 Apache-2.0 是公告承诺，128K v1.1 和超过 100B 的 v2.0 属路线图。**建议继续观察正式工件**，该混合架构有研究价值，性能暂不可与现有服务横比。[原始公告](https://corporate.ovhcloud.com/asia/newsroom/news/ovh-olala-mini/)

### TeleChat→Xing4：国内芯片训练报告必须区分 29B 工件与 285B 工程研究

- **机构/项目与时间**：中国电信星辰团队 等（TeleChat→Xing4）。arXiv `2611.22115v1`，提交 Tue, 17 Nov 2026 02:58:31 UTC；11 月 18 日分类公告中观察到，本轮 11 月 19 日发现；提交日期与首次公开日期分别处理。
- **一手来源**：[固定版论文](https://arxiv.org/abs/2611.22115v1)、[全文](https://arxiv.org/html/2611.22115v1)。
- **具体增量**：中国电信星辰团队公开架构与 MindSpore HyperParallel 系统设计。已开源成员为29B总参数、4B激活、256K上下文，40层/64路由专家/top-4/1共享；系统主例是285B、17B激活、1M上下文、45层/256专家/top-8/1共享，8,192张Ascend 910C训练。后者不能写成已释放权重。MLA+DSA+IndexShare、mHC、MTP与DSpark分别解决存储、索引、稳定性、训练与解码问题。
- **证据与限制**：§2/Table2的最高4.68×prefill属于Zhanlu仿真，作者明确不含全部线上开销。Table7在1,024 NPU、262,144序列下，PP=4基线36.4s/step→HSDP/PP=1的31.8s（减时12.6%，吞吐约1.14×）；Table8再仅切owner-based Muon为28.9s，减少冗余全矩阵Newton–Schulz计算。浮点重排不保证逐比特等价。
- **实践意义与建议**：**建议深入阅读**。优先研读拓扑分层通信、通信重叠、Muon owner与选择重计算；这些是可迁移的系统设计。报告不是完整复现包：本轮未校验全训练数据配方、全部权重许可与8,192卡长训日志，不能将仿真、1,024卡消融与集群规模混为一项实测。

<a id="topic-2"></a>
## 预训练与架构

### Structure–Residual：早层 boundary carrier、晚层连续结构正则

- **机构/项目与时间**：Huang, Hai / Żuk, Bartosz 等（Structure–Residual）。arXiv `2611.23197v1`，提交 Tue, 17 Nov 2026 18:38:27 UTC；11 月 18 日分类公告中观察到，本轮 11 月 19 日发现；提交日期与首次公开日期分别处理。
- **一手来源**：[固定版论文](https://arxiv.org/abs/2611.23197v1)、[全文](https://arxiv.org/html/2611.23197v1)，[官方代码固定快照](https://github.com/haihuang-ml/structure-residual/tree/4e513e5728b557a096e550a2a794a384391e15aa)。
- **具体增量**：BBC将跨段信息经边界位置传递，同时保留段内自回归；CC对表示几何作结构/残差分解约束。早层适合BBC、晚层适合CC，并比较同密度随机/滑窗控制。
- **证据与限制**：§6：GPT2 Small/Medium、FineWeb-Edu、长度1,024；300M token三seed时Medium NLL相对降低1.44%，20B token只单seed，附加CC边际缩小。不是前沿规模或等训练FLOPs结论。官方代码已读README固定4e513e57，给出boundary层2与CC层22复现命令。
- **实践意义与建议**：**考虑复现**。规模较小但对照可操作，适合低成本验证表示约束是否超出稀疏本身的作用；先复现多seed及20B预算，不据此替代现有大模型架构。

### TPHC：让不同 TP shard 的输出分别更新 residual streams

- **机构/项目与时间**：Hu, Xinying / Wu, You 等（TPHC）。arXiv `2611.23026v1`，提交 Tue, 17 Nov 2026 16:31:38 UTC；11 月 18 日分类公告中观察到，本轮 11 月 19 日发现；提交日期与首次公开日期分别处理。
- **一手来源**：[固定版论文](https://arxiv.org/abs/2611.23026v1)、[全文](https://arxiv.org/html/2611.23026v1)，[官方代码固定快照](https://github.com/wuyou2002/TPHC/tree/f01dddf474cc3dda63a183359d98e8b7c867749d)。
- **具体增量**：针对mHC多stream共享同一sublayer输出的表达瓶颈，每TP组读取不同stream混合并保留组输出，在写回时形成不同更新；stream自身可沿TP分片。
- **证据与限制**：Table1：1.1B Llama、100B SlimPajama、2,048长度、N=4/8/16，验证loss相对mHC低0.7%–1.0%，六任务均分增0.3–0.7pp。Table2 H20上的8B训练step仍比mHC4慢9%、峰值高5%；从4→16stream又增19%时间。固定实现f01dddf4的README可核N/G/U与stream-sharding设计。
- **实践意义与建议**：**考虑复现**。适合同时研究表达和并行放置；8B条目是成本测量，不是8B长训质量验证，stream更多也不是免费算力。

### SPICE：稀疏绝对参数消息支持动态邻居

- **机构/项目与时间**：Cagnasso, Pietro / Kopyshevskiy, Daniele 等（SPICE）。arXiv `2611.23053v1`，提交 Tue, 17 Nov 2026 16:47:17 UTC；11 月 18 日分类公告中观察到，本轮 11 月 19 日发现；提交日期与首次公开日期分别处理。
- **一手来源**：[固定版论文](https://arxiv.org/abs/2611.23053v1)、[全文](https://arxiv.org/html/2611.23053v1)。
- **具体增量**：结合chunked Top-k与error feedback，发送修正后的参数坐标值而非要求接收者持有历史重建状态的增量；可在固定通信预算内增加peer数。
- **证据与限制**：§4/Table2–3：134M/551M dense与OLMoE-1B-7B分别约2.7B/11B/26B FineWeb token。7B MoE四peer/6.25%密度相对dense双peer约少8倍参数传输且质量相近，但质量扩展至128副本来自模拟器。吞吐另在单机8×RTX A6000、限带宽1–10Gbps、真实P2P测，不能写成跨地域128GPU实测。
- **实践意义与建议**：**建议深入阅读**。对松耦合训练有价值，复现应计算12-bit索引开销、peer扩展和故障模型；不把压缩率自动转成端到端等质量加速，本轮未核专用公开代码。

### InfuseLoop：把已完成的跨轨迹 latent 注入并行循环

- **机构/项目与时间**：Sun, Zhaofeng / Hu, Lanxiang 等（InfuseLoop）。arXiv `2611.22426v1`，提交 Tue, 17 Nov 2026 09:00:59 UTC；11 月 18 日分类公告中观察到，本轮 11 月 19 日发现；提交日期与首次公开日期分别处理。
- **一手来源**：[固定版论文](https://arxiv.org/abs/2611.22426v1)、[全文](https://arxiv.org/html/2611.22426v1)。
- **具体增量**：并行looped Transformer只传本轨迹前态可能限制深度扩展；用轻量attention残差聚合因果可用的其他轨迹状态，增加O(Rd)状态，保持token并行与共享全局KV。
- **证据与限制**：§4：约0.6B共享28层、主训练6B FineWeb-Edu token、长度1,024，最多八loop；八loop相比vanilla PLT降低PPL3.4%–17.1%。单GB200、FA4/FP16乘子FP32残差、CUDA Graph计时，较PLT额外decode约0.3%，但较非loop仍慢23.3%。比较按等token而非等训练计算。
- **实践意义与建议**：**建议深入阅读**。适合考察递归计算能否利用闲置算力；不能把固定骨干的增益解释成优于同推理预算的大非循环模型，长训、多并发及公开实现本轮未核。

### Vision–Language Fusion Scaling：晚融合 caption loss 更低不等于 VQA 更好

- **机构/项目与时间**：Liew, Seng Pei / Suzuki, Teppei 等（Vision–Language Fusion Scaling）。arXiv `2611.22351v1`，提交 Tue, 17 Nov 2026 08:01:10 UTC；11 月 18 日分类公告中观察到，本轮 11 月 19 日发现；提交日期与首次公开日期分别处理。
- **一手来源**：[固定版论文](https://arxiv.org/abs/2611.22351v1)、[全文](https://arxiv.org/html/2611.22351v1)。
- **具体增量**：Noetra用“各模态基础学习曲线＋竞争造成的分布偏移”拟合融合时机、数据比例与模型规模，分开interleaved与caption数据。
- **证据与限制**：§3–6/Table3：Gemma3式189M–3.5B主干＋SigLIP2，主要300M模型；56B token固定预算比较300M/1B。晚融合改善caption loss但损害text loss，视觉下游不随caption loss单调变好；早融合保语言且晚融合视觉loss优势随规模缩小。S5文本token略超预算3.3%，作者披露。
- **实践意义与建议**：**建议深入阅读**。训练多模态配方应同时量text与visual downstream，不能只按视觉验证loss定阶段。拟合常数依赖数据/tokenizer，未证明更大模型或所有模态均适用；本轮未核开放复现包。

<a id="topic-3"></a>
## 后训练

### DaisyRose-27B：1,200 个长程任务纯 RL，需注意基线、会话长度与闭源训练后端

**机构／日期／版本：** RadixArk，2026-11-18；BF16 `293a7a2a`、FP8 `446b7a55`、NVFP4 `c4d85c5d`，Apache-2.0。

**解决的问题与增量：** 在 Qwen3.8-27B 上仅做 RL，无 SFT／蒸馏，使用约 1,200 个经过基模失败分析和防投机审查的任务；144 H200 训练 18 天、约 63,000 GPU 小时。Miles 全异步组织 128 张推理卡与 16 张训练卡，每步更新后发布新权重。任务可运行约 1.5 小时、超过 100 次工具调用；**最长 1M 是经 harness summarization 折叠为多轨迹的会话，模型上下文仍为 256K**。

**证据与关键限制：** 作者给出 Terminal-Bench 4.1 12.0%，但详细来源表说明用于对比的 Qwen3.8 5.6% 来自 **4.0** 条目，因此博客“5.6→12.0”不能当作严格同版本消融。DaisyRose 的 4.1 是内部 mini-swe-agent＋摘要，每题八次求平均；表中其余模型也混有公开不同 harness。分数逐项取 BF16／FP8 较高者，不代表每种格式均达该分数。作者在相近任务上报告内部后端 MFU 37% 对开源 Megatron 后端 16%、端到端 16.8K 对 8.3K token/s，但 **实际训练后端内部使用、1200任务集和完整训练配方未随权重完整开放**，不能把 Miles 开源等同于这组系统收益可直接复现。

**建议深入阅读并考虑复现同基模同 harness 对照**；优先验证任务筛选与长会话经验组织，而非复用跨版本“翻倍”标题。[固定模型卡及评测来源](https://huggingface.co/RadixArk/DaisyRose-27B/blob/293a7a2ab54aaea78098ee917c546390667521dc/README.md)、[训练披露](https://huggingface.co/blog/RadixArk/daisyrose-27b-pushing-the-frontier-of-post-trainin)

### Rumble-2-Flash：跨 tokenizer 的 on-policy 语音蒸馏披露及新评测

**机构／日期／版本：** Reson8，2026-11-18 博客与模型卡增量，固定 `a0c34287`；旧权重不按当天首发重报。

作者用 Qwen3-ASR 架构构建 0.5B 多语 ASR 学生，由生产 Rumble-2 teacher 对学生自己产生的转录逐 token 反馈，采用 GOLD 对齐不同 tokenizer 的片段与概率。相较仅模仿离线 teacher 转录，它针对学生实际出错的轨迹提供密集反馈。卡片新增 Open ASR Leaderboard **v1.16.1、H200、vLLM** 设置，多语平均错误率 6.96%、英语 6.6%；作者在 302 小时、14 种英语口音的 AppTek 电话集上报告 7.86% WER，对 Nemotron 3.5 Transcribe 30B 的 7.78%。超过 7,000 RTFx 只在较大数据集的满批吞吐中出现，不能写成单条音频时延。权重 Apache-2.0、40+语言；teacher 完整训练配方和独立复现实验未提供。**建议深入阅读跨 tokenizer 蒸馏并在目标电话域复现**。[蒸馏披露](https://huggingface.co/blog/rumble-2-flash)、[固定卡片](https://huggingface.co/Reson8/Rumble-2-Flash/blob/a0c34287229103b706c670d70791683e34379732/README.md)

### Pleias 发布 98,606 条合成伦理选择题和推理轨迹

**机构／日期／版本：** Pleias，2026-11-18；`0238abfe`，CC-BY-4.0；数据实际生成于 2026 年 4 月，此次是公开工件。

其数据卡描述：先用关系、权力、信任、脆弱性、职责等 12 个结构轴定义场景，按贝叶斯网络采样规格，再生成单句场景，随机配成两场景的四选一问题；标签来自结构规格，另一个微调模型生成推理。每行含两场景的规格、预设答案、生成答案及自一致标记，可用于筛选推理标签不一致样本。作者称 99.6% 场景符合预设判断，但没有给出足够独立人工审核或训练收益证据；道德口径当前主要为美国语境。本轮核验数据卡与提交，未将全量 parquet 独立复数／人工复标，98,606 为作者发布数量。**建议继续观察，或作为合成数据质量审计素材**，不直接当作跨文化伦理真值。[固定数据卡](https://huggingface.co/datasets/PleIAs/synth-ethics-mcq-100k/blob/0238abfe1f5d6f6e2cbd631438c6e71bb4a8954a/README.md)

### Ambient SFT：用高温基座匹配限制小数据微调遗忘

- **机构/项目与时间**：Daras, Giannis / Papachristou, Marios 等（Ambient SFT）。arXiv `2611.23229v1`，提交 Tue, 17 Nov 2026 18:59:37 UTC；11 月 18 日分类公告中观察到，本轮 11 月 19 日发现；提交日期与首次公开日期分别处理。
- **一手来源**：[固定版论文](https://arxiv.org/abs/2611.23229v1)、[全文](https://arxiv.org/html/2611.23229v1)。
- **具体增量**：冻结预训练模型作教师，在常规SFT之外匹配较高温度下的条件分布；温度调节保留基座分布的粒度，避免在小样本上把尾部分布压平。
- **证据与限制**：§4/Table2：Qwen3 1.7B–14B在约21K条（Dolci-Instruct-SFT的1%）数据上，八项平均相对分数提高8.0%–20.3%，但IFEval并非每个规模都更好。16×GH200、同训练配方、p_c=0.75/T=2.5；§3估算同模型师生每步约慢1.1×。推理OLMo3-7B仅5/6任务胜SFT，不是全任务提升。
- **实践意义与建议**：**考虑复现**。适合数据稀缺且需保留通用能力的SFT。应在相同训练时长、完整数据规模和目标域验证遗忘/指令遵循的权衡；本轮未确认专用官方实现，不把公式视作已完成工程集成。

### Regularized Score Centering：准确估计中心仍不足以消除陈旧策略偏差

- **机构/项目与时间**：Bai, Hao / Ye, Chenlu 等（Regularized Score Centering）。arXiv `2611.21791v1`，提交 Mon, 16 Nov 2026 19:12:50 UTC；11 月 18 日分类公告中观察到，本轮 11 月 19 日发现；提交日期与首次公开日期分别处理。
- **一手来源**：[固定版论文](https://arxiv.org/abs/2611.21791v1)、[全文](https://arxiv.org/html/2611.21791v1)。
- **具体增量**：将score centering的Top-128近似换成独立辅助token的Monte Carlo估计，并加入相对行为策略的reverse-KL，约束异步学习器漂移；可与截断重要性权重TIS组合。
- **证据与限制**：§5控制sampler refresh；§6/Table1为三训练seed均值及标准差，六Agent任务上RSC(TIS)居前。AppWorld比SC系约高14.3%，但只评dev；WebShop的74.80与IS-PPOClip 74.81基本持平，不能称全面击败最强基线。数学dense/MoE与全异步SWE分别使用不同模型/协议，SWE为Qwen3.5-35B-A3B、2,394训练任务。
- **实践意义与建议**：**考虑复现**。值得复查训推数值不一致和异步策略lag是否被混成同一误差；应同时看learner heldout与旧行为策略reward，不只看训练奖励。硬件吞吐与训练成本未在本文本次比较中统一，不能推导端到端加速。

### TAILROLL：将 rollout 尾部等待拆成迁移与有条件截尾

- **机构/项目与时间**：Gholipour, Niloofar / Assuncao, Marcos 等（TAILROLL）。arXiv `2611.22206v1`，提交 Tue, 17 Nov 2026 04:44:20 UTC；11 月 18 日分类公告中观察到，本轮 11 月 19 日发现；提交日期与首次公开日期分别处理。
- **一手来源**：[固定版论文](https://arxiv.org/abs/2611.22206v1)、[全文](https://arxiv.org/html/2611.22206v1)。
- **具体增量**：在AReaL的chunk边界在线迁移请求，目的端一次re-prefill恢复；按空闲容量与迁移成本平衡尾部。GRPO同组完成至少3/4样本后可终止余下straggler，未完输出从loss屏蔽。
- **证据与限制**：§VI–VII：910C 64GB，默认8卡rollout/4个TP2引擎加8卡FSDP；Qwen3-14B/32B BF16。跨配置rollout减时24%–34%、完整step减时21%–22%；300step live-reward终段0.761→0.763，仅作该设置证据。AMC12 8K full arm引擎偏差1.33×超预设1.3，16K达1.60×，因此不能把该两项收益全归给调度。
- **实践意义与建议**：**考虑复现**。长CoT、长尾rollout值得复现；迁移需再prefill，截尾可能引入组样本选择偏差，且CodeAgent episode只减时约4%，bubble改善不自动等于step收益。实现描述约3,700行，但本轮未确认对应公开补丁，属论文系统而非AReaL已发布能力。

### M3OPD-Reasoner：领域 GSPO 教师→在策略蒸馏→同数据 MixRL

- **机构/项目与时间**：Trung, Nguyen Quang / Phan, Hoang 等（M3OPD-Reasoner）。arXiv `2611.23034v1`，提交 Tue, 17 Nov 2026 16:36:16 UTC；11 月 18 日分类公告中观察到，本轮 11 月 19 日发现；提交日期与首次公开日期分别处理。
- **一手来源**：[固定版论文](https://arxiv.org/abs/2611.23034v1)、[全文](https://arxiv.org/html/2611.23034v1)，[官方代码固定快照](https://github.com/ngquangtrung57/M3OPD-Reasoner/tree/e323d7a42cc5dd0554754ec44f5f25bc0cfa46fd)、[SFT数据](https://huggingface.co/datasets/ngqtrung/M3OPD-SFT-1.35M/tree/4d5fa3adb7d7173f39c5230f35d2a0ba5d378b26)、[RL数据](https://huggingface.co/datasets/ngqtrung/M3OPD-RL-193K/tree/0f14218fe03ec28e8e2cb5dbd57447f71501c095)。
- **具体增量**：A*STAR/VinUniversity从Qwen3.5-9B分别训练图像STEM、感知、视频教师，学生在自己rollout上向领域教师蒸馏，最后复用相同prompts做MixRL；显式比较混合训练与SFT warmup路线。
- **证据与限制**：Table2在21bench avg@8、thinking上限32,768的同协议下：基座69.4，直接MixRL74.4，领域教师→MOPD75.2，再MixRL75.9；先SFT的路线较差。仓库已读固定快照e323d7a4；SFT1.35M/RL193K两HF数据API实际可读，分别4d5fa3ad/0f14218f，11-18 07:03Z更新，均标mixed-source/other许可。
- **实践意义与建议**：**考虑复现**。是值得跑配方消融的公开多教师案例；数据“可下载”不等于统一宽松商用许可，必须逐来源检查。优势同时包含教师训练和蒸馏成本，不能当作与单次MixRL等算力结论；未做权重逐文件完整性或训练复现。

### ASH：把训练 hint 的行为偏移保留为推理可用的 steering vector

- **机构/项目与时间**：Le, Khiem / Hua, Ting 等（ASH）。arXiv `2611.23124v1`，提交 Tue, 17 Nov 2026 17:43:28 UTC；11 月 18 日分类公告中观察到，本轮 11 月 19 日发现；提交日期与首次公开日期分别处理。
- **一手来源**：[固定版论文](https://arxiv.org/abs/2611.23124v1)、[全文](https://arxiv.org/html/2611.23124v1)。
- **具体增量**：Hint-GRPO在训练获得提示、部署拿不到提示；ASH比较hint/plain轨迹，学习问题无关的中间层steering vector，推理时直接注入激活，无需参考解提示。
- **证据与限制**：§4及附录：Qwen3-1.7B/4B-Base与Llama3.2-1B/3B-Instruct，六数学/OOD任务pass@k，2×A100 80GB、4,096响应上限；相对hint基线的平均增益约2.1–2.6个百分点。训练耗时约GRPO的1.5×，不应只报“推理无hint”。
- **实践意义与建议**：**考虑复现**。适合检查hint造成的训练/推理上下文错位，复现时要加训练成本、激活注入部署支持和pass@1对照；本轮仅确认论文采用verl，未确认专用公开实现。

### Stratified Policies：固定五次采样分给五个互补 LoRA

- **机构/项目与时间**：Mahajan, Mridul / Zhang, Xuezhou 等（Stratified Policies）。arXiv `2611.23154v1`，提交 Tue, 17 Nov 2026 18:06:12 UTC；11 月 18 日分类公告中观察到，本轮 11 月 19 日发现；提交日期与首次公开日期分别处理。
- **一手来源**：[固定版论文](https://arxiv.org/abs/2611.23154v1)、[全文](https://arxiv.org/html/2611.23154v1)。
- **具体增量**：共享冻结主干、训练独立LoRA群体，每个adapter只贡献一条回答，直接优化pass@5；减少单一策略独立采样重复覆盖相同解法。
- **证据与限制**：§5：Arithmetic/TinyGPT、Knights&Knaves与Sudoku/Qwen2.5-1.5B、GSM-Symbolic/Llama3.1-8B；五个rank32 adapter，GRPO、三seed。Sudoku pass@5 74.2%→80.3%；单adapter秩增5倍未重现优势。但pass@1多数任务反而低于单策略，随机抽adapter也不等价于每个恰取一次。
- **实践意义与建议**：**考虑复现**。给固定响应预算提供训练端互补性方向；当前是四类可验证任务的pass@5实验，尚非部署SOTA，也未证明总token、服务切换开销或更广泛选择器下等成本。

### RSI-Prime：按决策质量筛监督，失败上下文仍保留

- **机构/项目与时间**：Zheng, Dewu / Ye, Yang 等（RSI-Prime）。arXiv `2611.23138v1`，提交 Tue, 17 Nov 2026 17:55:30 UTC；11 月 18 日分类公告中观察到，本轮 11 月 19 日发现；提交日期与首次公开日期分别处理。
- **一手来源**：[固定版论文](https://arxiv.org/abs/2611.23138v1)、[全文](https://arxiv.org/html/2611.23138v1)。
- **具体增量**：350条完整实验轨迹跨七类任务，由决策识别与质量评估确定assistant消息loss mask；只要消息关联任一不合格决策就整条不监督，观察与错误仍在上下文中。
- **证据与限制**：§3–4/Table3：GLM5.3两epoch SFT后作为训练Agent，在PostTrainBench1.2上训练Qwen3-4B-Base；每任务1×H100、累计10小时，DSH 400K上下文，官方加权总分36.56→48.24（相对31.95%）。附录说明三独立run平均只适用于原GLM基线，不能把改进路线当成相同三seed统计。
- **实践意义与建议**：**建议深入阅读**。值得借鉴message与decision多对多映射及恢复运行的预算记账；这证明指定benchmark上的训练操作能力，不足以证明递归自我改进闭环。公开数据/权重完整性本轮未核实。

### Margin Dynamics：用有限吸引点解释偏好优化差异

- **机构/项目与时间**：Xu, Xingzi / Seyfioglu, Mehmet Saygin 等（Margin Dynamics）。arXiv `2611.21766v1`，提交 Mon, 16 Nov 2026 19:02:17 UTC；11 月 18 日分类公告中观察到，本轮 11 月 19 日发现；提交日期与首次公开日期分别处理。
- **一手来源**：[固定版论文](https://arxiv.org/abs/2611.21766v1)、[全文](https://arxiv.org/html/2611.21766v1)。
- **具体增量**：把pairwise损失投影成两completion的margin动力学，区分DPO无界增长、IPO有限吸引点与label noise的不同作用；由此构造加平滑的reference-free TPO-S。
- **证据与限制**：§2–4与Table2/3：理论是两completion有效模型，完整训练需拟合beta_eff，不能当成任意LLM精确ODE。8B–9B、rank64 LoRA的三模型比较中，epsilon0.05配置的IFEval均值居前；主模型实验单A10040GB、512长度，小规模margin预测不等于开放长CoT普适保证。
- **实践意义与建议**：**建议深入阅读**。适合用作分析过度拉大margin和偏好噪声的工具，而非仅换一项loss追榜；作者明确定位可用operating point，未主张新SOTA，本轮未核专用实现。

<a id="topic-4"></a>
## 推理与部署

### Transformers MoE SSD 流式专家实验：slot pool 复用 grouped_mm，尚为 PR

**项目／日期／状态：** Hugging Face，2026-11-18 实验说明；Transformers **#50717 为 Draft PR、尚未合入，API 仍在讨论**，配套模型和 Metal 内核另行提供；当轮PR head为`3a11816051e8c708256e8690b5e8e96dad6fb68d`。

**具体技术增量：** 通用 Accelerate 逐模块卸载会搬整层专家；当前实现以MPS为实验目标，仅读取路由到的专家，使用跨层预分配的设备 slot pool、每层专家到 slot 的映射，将路由 ID 重映射后复用 `grouped_mm`，避免 Python 逐专家 matmul。8 个 I/O 线程按专家连续读取，并利用提前运行的后两层 router 预测预取；Qwen3.8-Flash-Next 的 51GB n-gram 表另走磁盘路径。

**证据与边界：** 作者在 **32GB M5 Mac**、同提示、贪心、关闭 thinking／MTP／投机草稿下对 Strata 0.1.60 比较；最终短解码 13.9 对 14.8 token/s，4K prefill 228 对 220 token/s，6K 文档后解码 12.4 对 13.7 token/s。两者量化布局需按原始 Space 复核；实验报告约 18GiB 常驻、6K 峰值 25.4GiB，受 SSD、主机往返和逐层同步限制。**建议考虑复现单用户内存受限的 MoE 负载**；不能据此推断批服务收益或宣称稳定版 Transformers 已有此能力。[一手实验与配置](https://huggingface.co/blog/moe-expert-streaming)、[未合入 PR](https://github.com/huggingface/transformers/pull/50717)

### 微调工时估计：272 个 H200 作业说明 FLOPs 之外的评估与检查点成本

**机构／日期／版本：** Dharma-AI，2026-11-18 技术实验披露；完整论文仍待发布。

作者将分配到 GPU 后到作业结束的时长作为预测目标，排除排队；272 次 SFT、模型不超过 15B、LoRA／全参，1／2／4 H200，来自共享训练／推理环境，作业为 1.6～172.4 小时。用提交前已知的训练、评估总量、保存次数／参数量及软件栈等 24 特征做 CatBoost，分层五折报告 MAE **1.71±0.33小时**，而最佳固定 MFU 的 FLOP 基线为 12.41 小时；90% 误差不超过 4.31 小时。

价值是把评估频率、checkpoint I/O 和共享环境开销纳入调度预测。限制同样明确：仅 H200 与 SFT，超过60小时仅13例；未验证跨硬件、时间外推或 RL 泛化，训练数据和完整搜索／特征定义尚未充分开放。GPU数的重要性不能解释为“加卡更慢”的因果结论。**建议继续观察，或用自有日志复现方法**，不照搬其数值预测。[原始实验](https://huggingface.co/blog/Dharma-AI/predicting-fine-tuning-time)

### Gemma4 K=V compact cache：省容量成立，延迟收益有条件

- **机构/项目与时间**：Bainsla, Kushagra 等（Gemma4 K=V compact cache）。arXiv `2611.23035v1`，提交 Tue, 17 Nov 2026 16:36:28 UTC；11 月 18 日分类公告中观察到，本轮 11 月 19 日发现；提交日期与首次公开日期分别处理。
- **一手来源**：[固定版论文](https://arxiv.org/abs/2611.23035v1)、[全文](https://arxiv.org/html/2611.23035v1)。
- **具体增量**：利用Gemma4 global层K/V代数绑定，把每KV-head/token两份1,024值改为一份640值，通过重叠strided视图给未修改的paged-attention kernel读取。
- **证据与限制**：§5–8：31B、16bit cache、A100 Triton backend，32K–64K稳态每请求KV节省因子1.34–1.40。2×A100的prefill/decode+NIXL+CPU offload、15GiB pool下，16K/8K输入两负载decode tokens/step增1.38×/1.24×，eager吞吐约1.30×/1.11×；等batch eager step有4%–9%变换开销。
- **实践意义与建议**：**考虑复现**。优先评估容量受限场景。不是所有Gemma4型号都适用：26B-A4B没有此结构；FP8默认scale未通过误差界，prefix cache/TP/投机未测。CUDA Graph“无明显开销”run输出检查失败，仅exploratory；NIXL部分旧run有lease过期，后续600s租约重复才关闭该问题。专用实现未确认为上游合入。

### XM-Sparse：hot KV 与 K-index 分开卸载，避免 HBM 缓存反吞容量

- **机构/项目与时间**：Sun, Yijun / Liao, Xudong 等（XM-Sparse）。arXiv `2611.22280v1`，提交 Tue, 17 Nov 2026 06:32:15 UTC；11 月 18 日分类公告中观察到，本轮 11 月 19 日发现；提交日期与首次公开日期分别处理。
- **一手来源**：[固定版论文](https://arxiv.org/abs/2611.22280v1)、[全文](https://arxiv.org/html/2611.22280v1)。
- **具体增量**：在SGLang研究原型中，随机KV按tile级TMA取回与attention融合；可预测K-index采用GPU发起DMA预取，并抑制与latency-critical算子争用。
- **证据与限制**：§5：8×B300，4卡计算/4卡模拟X-Mem内存节点；DeepSeekV3.2/GLM5.3 FP8、DP4 attention/EP4 MoE，16K–1M输入、固定128输出，60GiB resident budget/约15%miss。相对PCIe ECHO最高4.98×、HiSparse4.52×包含互联资源差；同一X-Mem naive基线只1.06–1.34×。
- **实践意义与建议**：**建议深入阅读**。机制的主要价值是释放K-index占用以增并发。超1TB容量通过重复映射模拟，不能写成真实大容量产品实测，也未证明在线动态负载SLO或质量；不是SGLang已发布特性。

### RISA：保留完整历史，重复去噪只读共享稀疏计划

- **机构/项目与时间**：Zhong, Xinrui / Ye, Fanjiang 等（RISA）。arXiv `2611.22191v1`，提交 Tue, 17 Nov 2026 04:27:48 UTC；11 月 18 日分类公告中观察到，本轮 11 月 19 日发现；提交日期与首次公开日期分别处理。
- **一手来源**：[固定版论文](https://arxiv.org/abs/2611.22191v1)、[全文](https://arxiv.org/html/2611.22191v1)。
- **具体增量**：针对block-diffusion在同block多次去噪重复读KV，区分priority/recovery两层，seed query构造受保护的有限读计划，多step复用索引但每次用新query重算attention。
- **证据与限制**：§4/Table1–3：DreamReasoner8B/SDAR8B/LLaDA2.2-mini，BF16 H200、32token block、每卡1请求；32K默认attention2.67×、固定dense轨迹512输出请求1.36×。128K的5.53×是attention，Dream/SDAR超过其32K配置仅执行伸缩，质量只在原窗口核。
- **实践意义与建议**：**建议深入阅读**。适合扩散LM长上下文研究；恢复历史不等于减少总KV容量，固定dense轨迹计时也不等于自由生成端到端收益。本轮未核官方代码。

### DNeo：将拒绝 proposal 的 target features 变成下一轮记忆

- **机构/项目与时间**：Sandler, Jameson / Bergsma, Shane 等（DNeo）。arXiv `2611.21925v1`，提交 Mon, 16 Nov 2026 21:41:20 UTC；11 月 18 日分类公告中观察到，本轮 11 月 19 日发现；提交日期与首次公开日期分别处理。
- **一手来源**：[固定版论文](https://arxiv.org/abs/2611.21925v1)、[全文](https://arxiv.org/html/2611.21925v1)，[官方代码固定快照](https://github.com/J-sandler/DNeo/tree/8449a3ce891f6b3321efafc1e5c0c7e5c40e11c7)。
- **具体增量**：以MoFA结构保留特征编码，加入rejection memory、非均匀target层选择和轻量causal修正，减少draft一次前向的无效候选。
- **证据与限制**：§5主要1×H100/SGLang、Qwen3-4B、六任务等prompt/token、batch1、block16，论文约比DFlash2快1.3×。官方README固定8449a3ce另列BF16 H10080GB、256 matched prompts/eight seeds、prompt≤256/output≤256、SGLang701token/s/2.96×AR；它是drafter，必须搭Qwen3-4B verifier。
- **实践意义与建议**：**考虑复现**。有公开实现与草稿权重入口，适合检查特征融合与拒绝记忆各自收益；不能把decode-only、小batch、短输入结果外推到长上下文和高并发线上。

### SpecSearch：投机配置必须随硬件、输出长度与并发选

- **机构/项目与时间**：Yildirim, Selin / Zeyrek, Ali 等（SpecSearch）。arXiv `2611.22296v1`，提交 Tue, 17 Nov 2026 06:53:30 UTC；11 月 18 日分类公告中观察到，本轮 11 月 19 日发现；提交日期与首次公开日期分别处理。
- **一手来源**：[固定版论文](https://arxiv.org/abs/2611.22296v1)、[全文](https://arxiv.org/html/2611.22296v1)。
- **具体增量**：搜索AR/SD、draft长度和KV/权重精度；成本模型先排除显存不适配，再校准或用跨GPU训练的Jev决策模型预测。
- **证据与限制**：§V–VII：RTX5090/H100/L40/MI300X、vLLM与DFlash。跨设备平均相对固定SD约1.36×；留一GPU的Jev保留全校准吞吐约97.36%，但H100中等并发常选过短draft。论文不同段落对相对AR平均给2.01×与2.09×，本日报不选其中一个当统一headline；原始聚合需复核。
- **实践意义与建议**：**建议深入阅读**。值得使用“可关闭speculation”的负载搜索思路；KV精度变化会改变目标数值行为，不应把跨precision比较称分布完全相同。专用实现本轮未核，先看测量数据与成本模型。

### VisHead-Spec：图像只给少数 drafter KV heads

- **机构/项目与时间**：Guo, Xin / Rai, Arushi 等（VisHead-Spec）。arXiv `2611.21839v1`，提交 Mon, 16 Nov 2026 19:57:35 UTC；11 月 18 日分类公告中观察到，本轮 11 月 19 日发现；提交日期与首次公开日期分别处理。
- **一手来源**：[固定版论文](https://arxiv.org/abs/2611.21839v1)、[全文](https://arxiv.org/html/2611.21839v1)。
- **具体增量**：发现多模态draft不同KV head对视觉token依赖稀疏，将8/40 drafter KV heads保留全视觉访问，其余只读文本，并优先用target高层视觉特征。
- **证据与限制**：§4：Qwen3-VL-8B、MMSpec六任务、A100；batch64对AR吞吐增16.0%，DSpark为9.5%，是基于同AR的两项增益，不能写成比DSpark快16%。accepted length差≤0.3%；batch1为2.23×AR，略低于DSpark2.26×。训练67,304条target回答样本。
- **实践意义与建议**：**建议深入阅读**。说明高并发时draft attention而非接受率可能瓶颈；只验证一target与指定语料，本轮未核专用公开实现，迁移其他VLM需重新选head。

### BQ-LoRA：把 adapter 存储降为二进制，训练显存未同步下降

- **机构/项目与时间**：Park, Juyoung / Lee, Heejun 等（BQ-LoRA）。arXiv `2611.22334v1`，提交 Tue, 17 Nov 2026 07:33:56 UTC；11 月 18 日分类公告中观察到，本轮 11 月 19 日发现；提交日期与首次公开日期分别处理。
- **一手来源**：[固定版论文](https://arxiv.org/abs/2611.22334v1)、[全文](https://arxiv.org/html/2611.22334v1)。
- **具体增量**：4bit基座不变，将低秩adapter参数二值化并配逐层scale与STE；研究BQ-SGMV把二进制内积接入多adapter服务。
- **证据与限制**：§4–6/Table1–4：Qwen2.5 1.5B–72B上对QLoRA平均增0.8–3.5pp，Llama2不稳定领先。adapter落盘/驻留约小13–16×；训练显存接近QLoRA、step时间增<4%。A10080GB的15.6×驻留数量及无cache miss来自指定偏斜合成请求，不是通用15.6×吞吐。
- **实践意义与建议**：**建议深入阅读**。适合多租户adapter容量瓶颈。精度收益应在相同数据/seed/rank与LoftQ等更强基线复核；二值理论推导不代替部署稳定性，专用实现本轮未核。

### TempQuant：量化后的采样器也需重新标定

- **机构/项目与时间**：Athmaram, Mukund / Ma, Liang 等（TempQuant）。arXiv `2611.22128v1`，提交 Tue, 17 Nov 2026 03:11:36 UTC；11 月 18 日分类公告中观察到，本轮 11 月 19 日发现；提交日期与首次公开日期分别处理。
- **一手来源**：[固定版论文](https://arxiv.org/abs/2611.22128v1)、[全文](https://arxiv.org/html/2611.22128v1)。
- **具体增量**：冻结BF16与量化模型，仅学习温度/top-p来匹配token分布；有全局配置与逐token预测头两种粒度，rank翻转位置和保持位置需要不同调节。
- **证据与限制**：§4与附录：GPTQ3 Qwen3-4B的MATH小样本上，原推荐配置34%到global设置约51%，主要图涉及200题中198可评分、单样本；不能当大规模稳定17pp收益。global只改采样参数，部署不增加网络；逐token版本另有预测开销，MI350X时延实验单列。
- **实践意义与建议**：**考虑复现**。可作为低比特部署廉价标定对照；匹配BF16分布无需标签但需要教师logits，不等于没有校准成本，也不保证修复语义能力或所有量化方法。

### Blackwell 混精度审计：能跑 INT2 不等于可用，FINAL checkpoint 未保留

- **机构/项目与时间**：Han, Hyojung 等（Blackwell 混精度审计）。arXiv `2611.21974v1`，提交 Mon, 16 Nov 2026 22:58:32 UTC；11 月 18 日分类公告中观察到，本轮 11 月 19 日发现；提交日期与首次公开日期分别处理。
- **一手来源**：[固定版论文](https://arxiv.org/abs/2611.21974v1)、[全文](https://arxiv.org/html/2611.21974v1)。
- **具体增量**：将Qwen3-8B量化选择约束为144 fused linear的packed执行、加载显存、韩英/数值/推理质量和TPOT；报告原生低bit与混精度失败。
- **证据与限制**：§3–5：B200/vLLM；七个INT2 grid均生成退化，但仅限这批GPTQ/RTN配方。最终98个NVFP4 weight-only＋46个INT8模块/INT8 head、16bit activation，在未触碰FINAL-81/5,168条上加载5.917GiB（门槛5.98，排除KV/activation），韩文teacher-correct保留0.904、TPOT/BF16=0.939。相对W4A4增1.4pp同时改变activation与保护层，不能分离归因。
- **实践意义与建议**：**建议深入阅读**。优秀负面披露，但最终checkpoint未留存、bitwise重建未证，使直接采用门槛高。借鉴独立final-set与逐层kernel审计，不照搬最终配比。

### Sparse blind heads：先诊断 selector 漏排，而非直接扩大预算

- **机构/项目与时间**：Zhang, Wenhe / Chen, Boxing 等（Sparse blind heads）。arXiv `2611.21917v1`，提交 Mon, 16 Nov 2026 21:30:27 UTC；11 月 18 日分类公告中观察到，本轮 11 月 19 日发现；提交日期与首次公开日期分别处理。
- **一手来源**：[固定版论文](https://arxiv.org/abs/2611.21917v1)、[全文](https://arxiv.org/html/2611.21917v1)。
- **具体增量**：以同预算精确attention mass选择为oracle，teacher-forced校准识别把needle丢掉且影响答案logprob的head；修复少量head全attention或用4bit scorer保预算。
- **证据与限制**：§4/Table2–3：MiniCPM4.1-8B 64K多key检索54.1%→97.5%，与full attention接近；指定生产engine/H100的32K–128K TTFT保留原稀疏节省84%–89%。SeerAttention-R需18/288 KV heads修复，MoBA则很多，不能泛化为“总是三head”。
- **实践意义与建议**：**考虑复现**。适合定位长上下文漏检根因；校准要含相关失败类型，答案强制与oracle不可放入部署路径。任务特化检索增益不替代通用质量评测，本轮未核对应公开补丁。

<a id="topic-5"></a>
## 开源工程

本节区分已发布工件与默认分支合入。所有加速、精度和回归结果均为来源作者报告；本轮未使用GPU/NPU独立压测、复训或运行其完整测试套件。主线PR不因同日存在release就自动属于该release。

### verl：按权重版本隔离外部 KV，修复 reset 到不了存储层的问题

这是本轮优先阅读的训推一致性改动。此前硬清缓存在部分 vLLM connector 上是空操作，SGLang flush 也不清 HiCache 存储；能清的共享 store 又可能被一次不换权重的 rebalance pause 全局清空。新增 `external_kv_cache_invalidation: salt`，请求通过 pause gate 时用 `<run id>:<global_steps>` 形成 engine-native cache salt，使旧权重的 block hash 不再匹配。默认仍为 reset，GPU 本地缓存仍清理，旧外部对象靠 eviction 回收。

边界比开关本身重要：vLLM 只接受按 block hash 寻址的 connector，拒绝 token-ID 寻址的 LMCacheConnectorV1；PD 拆分未支持，store 需有淘汰策略和容量。vLLM sleep 保留 store 还依赖 #57810（0.33.0 之后）；旧版正确性不变但会失去缓存。SGLang 的 file storage 有集成验证；TRT-LLM 此处只接 salt，verl 未配置其外部 store。**建议深入阅读并在有外部 KV 的 RL 作业中考虑复现**，检查实际权重更新是否都推进 global_steps；这不是跨版本复用旧 policy KV 的功能。

一手来源：[#8323](https://github.com/verl-project/verl/pull/8323)（2026-11-18T14:03:51Z，main @ `c012b10a498d1fa65002fc20f8ece00392a4fc69`）。

### SGLang：VMM 文件描述符让权重映射不再跟 daemon 同生共死

新增 `--weight-cache-transport vmm_fd`：daemon 用 CUDA VMM allocation 导出文件描述符，engine 在自行预留的虚拟地址映射只读权重，daemon 退出后仍保有映射；原 torch_ipc 模式的存活依赖由此可解除。daemon 的 auto 暂仍选 torch_ipc，engine 的 auto 接受服务端选定格式，显式不匹配则拒绝。实现同时处理非持久 buffer、原始模型对象释放，以及 VMM 不计入 PyTorch allocator 时的权重内存核算。**建议深入阅读**：适合多 engine 共享权重及重启隔离设计；本轮核实实现和测试，未压测生产重启、内存占用或加载时延。

一手来源：[#46953](https://github.com/sgl-project/sglang/pull/46953)（2026-11-19T01:00:03Z，main @ `a3eae555453777c3f4c5ab41e19ef1c9577e0785`）。

### SGLang：Kimi-K3 prefill 把复制的 KDA 和共享专家权重分片

新增 `--kda-dwdp-size N`，同一 attention-TP 切片的 attention-DP ranks 各保留 KDA q/k/v/g 输入投影、o_proj 及适用共享专家的 1/N 权重，在层执行前通过 NVLink gather。源文给出的容量估计是 K3、attention TP1、N=8 约节省 53GB/rank；这是指定权重集合的估计，不是整机实测峰值。代价是每个分片层每次 forward 的权重 gather。

仅允许 PD prefill；decode 每步重聚合约54GB/rank 的风险被启动检查拒绝。N 须整除 TP 和 attention-DP，拒绝 CP>1、PP、full CUDA graph、weight cache 及跳过 shard loader 的加载格式。**考虑复现**：优先验证权重容量与 NVLink 带宽之间的取舍，不能外推成 decode 加速。

一手来源：[#50842](https://github.com/sgl-project/sglang/pull/50842)（2026-11-18T16:12:59Z，main @ `37f3b2730bca1cc875126f8b94c4487ec1331a95`）。

### SGLang 与 Dynamo：Rust 服务接口成批落地，Responses 仍有限制

SGLang 将控制面（pause/continue、内存、profiling、权重更新、session、NGRAM corpus）、OpenAI chat/completions、Anthropic Messages、Ollama、评分/embedding 等 schema 与 HTTP/gRPC 接入继续迁到 Rust server。不是“接口名出现即完整兼容”：Responses 当前是借 chat route 实现的无状态路径，streamed form 与 response store 延后；部分权重组/更新操作仍限 single-DP，LoRA 不支持操作有显式拒绝。

Dynamo 同轮把启动/部署切到 `SGLANG_RUST_SERVER=1`、原生 `sglang.api.v1` gRPC（文档要求 SGLang≥0.5.23），并用每节点一个 sidecar 服务本节点多个 Rust listeners。Frontend 选 worker/DP rank，sidecar 只投递到本节点，节点间 collectives 仍由 SGLang 执行。**建议深入阅读接口和部署边界**；这是主分支实现与集成，无本轮独立吞吐或兼容性全测。

一手来源：[#49735](https://github.com/sgl-project/sglang/pull/49735)（2026-11-18T21:01:31Z，main @ `f01462f3b3d04af5d7f03b8701ce37cbc93b7f69`）；[#49736](https://github.com/sgl-project/sglang/pull/49736)（2026-11-18T21:02:41Z，main @ `120c91cf32d958379698bc526e9de29d9b064466`）；[#49737](https://github.com/sgl-project/sglang/pull/49737)（2026-11-18T21:03:36Z，main @ `2e2c622f2774675118c72be3e08f8f96e38fd91f`）；[#49738](https://github.com/sgl-project/sglang/pull/49738)（2026-11-18T21:04:01Z，main @ `9e7f4072a877d9504fc27299f826788df9096dcb`）；[#49739](https://github.com/sgl-project/sglang/pull/49739)（2026-11-18T21:04:30Z，main @ `07e72a91dde12590b8e02cf5cd485bc4a06458fa`）；[#49740](https://github.com/sgl-project/sglang/pull/49740)（2026-11-18T21:04:59Z，main @ `b0ce7369f9ee66d71d14e7b435a9cf9b2d69307c`）；[#49741](https://github.com/sgl-project/sglang/pull/49741)（2026-11-18T21:05:26Z，main @ `ac584ef1a02ba45780e0655c41cc69afe23e641f`）；[#49742](https://github.com/sgl-project/sglang/pull/49742)（2026-11-18T21:05:56Z，main @ `682657a365927c9b1b270f19efd764bc38dec04a`）；[#49743](https://github.com/sgl-project/sglang/pull/49743)（2026-11-18T21:06:23Z，main @ `d59b1542d024cc13a38d2c43ce67b002254c3ec3`）；[#49744](https://github.com/sgl-project/sglang/pull/49744)（2026-11-18T21:13:15Z，main @ `483f28b7ee92c8e407dccd87ecf8c84a89c91690`）；[#49745](https://github.com/sgl-project/sglang/pull/49745)（2026-11-18T21:13:41Z，main @ `541606a0751127bb1588f5c3678b6ccb1124eebf`）；[#49746](https://github.com/sgl-project/sglang/pull/49746)（2026-11-18T21:14:11Z，main @ `b0909b3f0429f54161177fa13efc7f2e7edc7df3`）；[#49747](https://github.com/sgl-project/sglang/pull/49747)（2026-11-18T21:14:37Z，main @ `be9cc54f706a2daa800b0cafad62a150e58b23cb`）；[#49748](https://github.com/sgl-project/sglang/pull/49748)（2026-11-18T21:44:49Z，main @ `0b42e6344af830c1b0fc3a1951d7445f92d1ed91`）；[#17431](https://github.com/ai-dynamo/dynamo/pull/17431)（2026-11-18T21:51:25Z，main @ `13e09f2a0fbbf6eddb51633c02774c4ad23fd81b`）；[#17610](https://github.com/ai-dynamo/dynamo/pull/17610)（2026-11-18T23:18:11Z，main @ `ad0fe0f4c7f6fb993126d440e3b10cade79f0ed8`）。

### Dynamo 更正：KVCR hint 支持不能当成 SGLang engine 集成

官方文档明确 Dynamo 1.7.0 固定的 KVCR 0.1.1 仅支持 SGLang KV hint protocol，不含所述 HiCacheStorage/direct linker engine integration；后者仍指向 SGLang #45647/#45650 提议，当前未记载经资格验证的 Dynamo/SGLang/KVCR 版本组合。下面的部署说明被标为 preview。**继续观察**：这是对既有文档可能造成的能力误读的实质纠正，不能据此启用未落地存储集成；普通 SGLang serving 不受影响。

一手来源：[#18902](https://github.com/ai-dynamo/dynamo/pull/18902)（2026-11-18T22:41:06Z，main @ `f341c2f19c8663f3a3087a05d6d9fad2582f6e3d`）。

### vLLM：NIXL 先校验租约，再接受远端 KV

当请求在 decode scheduler 外被 router 长时间排队时，producer 的 KV 租约可能过期、block 已复用，但 NIXL read 仍能“传输成功”，读回别的请求的 KV/SSM。#66011 导出 producer deadline，按握手时钟偏移换算，记录每次成功 heartbeat 的精确续租规则，在发起读和观察完成时都检查；临近/越过租约则走 failed_recving 与配置的 kv_load_failure_policy，而非使用可能污染的张量。短租约安全裕量随 heartbeat interval 调整，失败发送的 heartbeat 不算续租。

#61499 进一步在多个 KV cache group 时按请求报告失败，避免调度器尚不支持的 invalid_block_ids 映射触发错误。**建议深入阅读并回放排队/失联/多组混合模型场景**；源代码提供边界测试，不能把传输完成当作租约有效证明。本轮未做 RDMA 服务压测。

一手来源：[#66011](https://github.com/vllm-project/vllm/pull/66011)（2026-11-18T18:54:04Z，main @ `067624f48776455e57d3bbaf1418575be84d3e44`）；[#61499](https://github.com/vllm-project/vllm/pull/61499)（2026-11-18T10:44:06Z，main @ `075f52626b8d1733cbeadf800cdea6f8ed08038c`）。

### vLLM：PD 被抢占后不再反复取只能供应一轮的 KV

经典 P/D 的 prefiller 只绑定 round0，decode 抢占后再次拉取会等满超时。#67497 将 `pd_fetch_submitted` 放入跨抢占保留的请求状态，首个非空 fetch 后后续 PD lookup 返回 MISS，走重算路径。CPU offload #59547 对非 replicated 布局给每 rank 独立 region，映射后立即 unlink 且不做不存在的 joiner barrier，避免分片 rank 错误共享同一块 CPU 数据；replicated layout 保留共享规则。

Mamba 稀疏 retention #67053 在释放请求时保留最新 decode 跨过的边界状态，下一轮带回生成历史时可从正确 resume state 命中，不只保留原 prompt 的状态。三项分别解决协议轮次、内存布局和混合状态生命周期，**有相应配置的部署应考虑复现**，没有统一性能提升数字。

一手来源：[#67497](https://github.com/vllm-project/vllm/pull/67497)（2026-11-18T11:24:13Z，main @ `f2d0d8418efc017cd5f6e4fd46dfb0b89874fb28`）；[#59547](https://github.com/vllm-project/vllm/pull/59547)（2026-11-18T07:24:42Z，main @ `df763962fc61f8d0047e89aa39079efe0e70e678`）；[#67053](https://github.com/vllm-project/vllm/pull/67053)（2026-11-18T01:35:33Z，main @ `52767604d08f5d3a2a3367b46b74e419492cbe90`）。

### vLLM：直接换权重必须完整；流式追加超长只丢当前块

#65799 校验一次 direct reload 覆盖所需全部参数，拒绝无声接受部分更新；失败后的下一轮仍需发送自己的 metadata，测试覆盖不复用失败前旧元数据。#63113 对追加输入超过 max_model_len 的 streaming update 返回空的 length turn 并丢该块，后续合适长度的块仍能继续会话，防止把 scheduler 推入不可处理状态。**建议采用前先检查错误处理和重试契约**，尤其 RL 权重流与实时多轮输入；这两项是正确性边界修复，不是训练/推理质量提升。

一手来源：[#65799](https://github.com/vllm-project/vllm/pull/65799)（2026-11-18T05:10:49Z，main @ `5cba3d7b550359dd393bebc56a89b9190a7c8138`）；[#63113](https://github.com/vllm-project/vllm/pull/63113)（2026-11-19T00:52:19Z，main @ `34f1fc8497a906dabc74be125b8e423f8cbdd915`）。

### vLLM：GLM KDA 接 FlashInfer，Inkling2 增 Blackwell 原生执行

GLM-5.3-Flash 的 KDA decode 将 conv+SiLU、递推和 gated RMSNorm 融合进 FlashInfer 路径，prefill 接 CuTe DSL chunk kernel；测试用相同权重/cache 比较替代的 Triton 路径，conv state 要求精确一致，递推 state/output 容许 kernel 舍入且空槽不能改变。Inkling2 Small 增模型执行和 SM100/103 内核，fused all-reduce 依赖 NVLink multicast/thread-block cluster，相关 mix/norm 编译禁用 FMA 收缩。

另 #67812 将 chunk_scaled_dot_kkt 的 beta 乘法移到 K 累积循环之后，恢复上游 FLA 的数学顺序，避免在多个 K tile 时重复缩放历史累积量。**考虑在对应模型上复现**；本轮无统一端到端速度/精度结论，也不把主分支支持写成新模型首发。

一手来源：[#63544](https://github.com/vllm-project/vllm/pull/63544)（2026-11-18T23:55:10Z，main @ `f3f612ecfa76e6c541362ce6a7bd80f06c197273`）；[#67325](https://github.com/vllm-project/vllm/pull/67325)（2026-11-18T17:14:33Z，main @ `abb8194dc35e4d72e0b81a41ed7d786a96b97e7c`）；[#67812](https://github.com/vllm-project/vllm/pull/67812)（2026-11-19T00:26:46Z，main @ `b209d984d88121208526c09ec3b9e980850efe9d`）。

### vLLM：一台服务可按请求选择结构化输出后端

StructuredOutputManager 从单 backend 改成按名称管理多个 backend，配套 token bitmask 和实际 logits 词表宽度的处理与测试。它使不同 grammar/请求的后端能力和回退策略可在同服务内组合，减少为了语法约束拆多个服务的需要。**建议阅读请求配置与负面测试后考虑采用**，尤其 tokenizer vocab 与模型输出宽度不同时；本轮未测混合请求吞吐，不能由接口灵活性推导调度收益。

一手来源：[#67335](https://github.com/vllm-project/vllm/pull/67335)（2026-11-18T16:09:11Z，main @ `46d94c97ce051f4038f921bf6965310fb727355a`）。

### Megatron：推理 MoE 的对齐与 top-k 合并拆开优化

改动位于 `megatron/core/inference/moe/vllm_fused_moe.py`，不是笼统的所有 MoE 训练加速。普通路径增加无原子 multi-CTA token/expert 对齐、按设备/专家形状与 token hint 选择 FC1/FC2 调优参数，并提供 tiled top-k reduce；对较大 pair hint 或 block_m≥128 回到多 kernel 路径。batch-invariant 路径保留原有分配/算术约定。源码增加针对对齐表、有效 token、专家边界与数值的测试。**建议深入阅读**小批推理中“准备路由元数据”与 GEMM 本体的权衡；本轮未从代码变更推导通用吞吐倍数。

一手来源：[#8320](https://github.com/NVIDIA/Megatron-LM/pull/8320)（2026-11-18T22:33:24Z，main @ `e715b4f516b1931816e28f30b38509a93d6f246c`）。

### Megatron：MFSDP v2 专家优化器分组与 GTP 虚拟流水隔离

MFSDP v2 新增 `expert_num_distributed_optimizer_instances`，允许 dense 与 expert 的 outer/inner DP 轴采用独立实例数；默认继承 dense 值，只有 v2 允许不同，且只 dense 有 outer axis 时 expert 不能盲目继承外层分片策略。GTP #9504 则按 model chunk、角色和 process group 划分默认 prefetch chain，避免交错虚拟流水中的另一 chunk backward gather 覆写当前 chunk 仍在用的 forward 权重 buffer。

**建议深入阅读并在 MFSDP×EP、GTP×virtual PP 组合上考虑复现**。前者提供新的通信/容量配置空间，后者修复 buffer 所有权；测试含多 GPU 与交错次序，但本轮未执行训练曲线或性能对照。

一手来源：[#7341](https://github.com/NVIDIA/Megatron-LM/pull/7341)（2026-11-18T10:40:22Z，main @ `c1ad819d07cb946620341a317b8d06e74207e0f1`）；[#9504](https://github.com/NVIDIA/Megatron-LM/pull/9504)（2026-11-18T08:14:14Z，main @ `e0e8f1fef555b6647dc87b3a47cbc3a314d820b7`）。

### TorchTitan：routed expert LoRA 可共享外侧矩阵

`LoRATransform(expert_adapter_layout="shared_outer")` 为 W13 共享读取 hidden 的 A、为 W2 共享写回 hidden 的 B，另一侧仍逐 expert；默认仍是 per_expert。例如 W13 的 A 从 [E,r,D] 变 [r,D]，减少 adapter 体积与训推同步量，并让所有 routed token 更新共享因子。实现对 EP 下共享因子使用 sparse-family replicated placement 以同步梯度，拒绝在非 routed W13/W2 分组投影上滥用此布局。

**建议考虑复现**与 SGLang/Megatron-Bridge/Miles 兼容的 RL LoRA 流程；“更省传输”有张量形状依据，本轮无模型质量或端到端成本实验，不假定共享/逐专家两种参数化等价。

一手来源：[#5738](https://github.com/pytorch/torchtitan/pull/5738)（2026-11-18T19:54:37Z，main @ `63a6030136c46d28dd221e620d3292ca9af01dbc`）。

### TorchTitan：并行化后保持 tied weights，异步检查点保留量化包装

#6144 在 `_parallelize` 后、FSDP 组参之前重新绑定 tok_embeddings 与 lm_head，并校验两者 weight sharding 一致。原先分布式注册分别创建 Parameter，等 to_empty 后才重绑太晚；有 shard padding 时 embedding 可能持续 gather 未更新副本，即使前向表面可运行。#6187 为量化 FSDP wrapper 增 new_empty 的保包装行为，避免 DCP async staging 以 data_ptr()==0 跳复制时丢失内部 tensor；#6196 让同步保存的 async_mode 同步传到 TorchCheckpointingManager。**对应模型和保存模式应优先回归**参数同一性、训练后 embedding 更新和 restore，不把“保存函数返回”当作数据已完整持久化。

一手来源：[#6144](https://github.com/pytorch/torchtitan/pull/6144)（2026-11-18T20:56:52Z，main @ `e07c8fbcccfdab3ada794534d574cd832a613b22`）；[#6187](https://github.com/pytorch/torchtitan/pull/6187)（2026-11-18T23:51:21Z，main @ `0c9ad2343e888ed027272f888620fdf737819b2a`）；[#6196](https://github.com/pytorch/torchtitan/pull/6196)（2026-11-19T01:01:20Z，main @ `ede7cc31542239b3df72e20562ff019d49b2dd4b`）。

### DeepSpeed：AutoSP 修正有效 token 加权与 SP 梯度求和

此前每 shard 独立 mean 使有效 token 数不均时等权、全 ignore shard NaN，且隐式/显式 mesh 的梯度聚合约定不一致。新增 FP32 shard loss-sum 和 valid-count 的 SP all-reduce，每 rank 得全序列损失，反向 mean 只除全局 valid count；梯度在 SP 求和、DP 平均。自动补建 mesh，并拒绝 SP 尺寸不符、class-weighted CE、reduction=none、probability targets 和 ZeRO≥2。

作者用 world4、SP2×DP2、两个 SGD step 与单设备全序列比较，覆盖 ZeRO0/1、mean/sum、不等有效 token、全 ignore shard及FP16 sum溢出；环境为8×Arc Pro B70、torch2.14.dev+xpu、Transformers5.17，两个既有测试组仍因其他问题失败。**建议深入阅读**，尤其 label 必须先在完整序列做 causal shift 再 prepare_autosp_inputs；模型内部 shift 的对齐仍留给后续，不能把此修复当成已解决。

一手来源：[#8457](https://github.com/deepspeedai/DeepSpeed/pull/8457)（2026-11-18T03:15:47Z，master @ `4a9efc1109fccfc544e0c24bec2a1b7a274f6397`）。

### DeepSpeed：ZeRO-3 不再沿用未用参数的旧梯度

按参数而非全局 micro_step 判断本累积窗的首次梯度：第一次覆盖旧 partition，后续（含 reentrant checkpoint 的同轮再次反向）累积；窗内完全未写参数清零。optimizer offload 在边界补拷“早期 microstep 有梯度、最后一步没梯度”的 partition，并保留 safe_set API 显式设置值。SuperOffload/ZenFlow 替换的 partition_grads 和 DeepCompile 自管 IPG 路径有明确退出边界。**建议在条件执行、稀疏专家或重入 checkpoint 作业中优先回归**；修复的是训练正确性，不能用普通 dense loss smoke test 代替。

一手来源：[#8981](https://github.com/deepspeedai/DeepSpeed/pull/8981)（2026-11-18T09:38:18Z，master @ `daf175b1133e6c3ebb11cd415c006ce6b0c7b181`）。

### PyTorch：FSDP copy-in 流成为可协调接口，collective 顺序可约束

FSDP2 暴露 `get_all_gather_copy_in_stream()`，activation offloader 可让自己的 H2D reload 排到参数 copy-in 后，使后续 all-gather 更可能与 activation reload 重叠，避免两个 H2D 同时抢链路。lazy init 前调用报错；single-rank 或 unshard_in_current_stream 时等该流是无实质作用的，不能据此保证带宽重叠。

Inductor 新增 `reorder_for_peak_memory_preserve_collective_order`，默认 False；开启后拒绝改变 collective/wait 顺序的内存重排候选，保留纯计算重排。**建议深入阅读并按 workload 复现**：一个提供流级协调点，一个限制可能导致分布式挂起的重排自由度，都不是默认已开启的性能保证。

一手来源：[#205180](https://github.com/pytorch/pytorch/pull/205180)（2026-11-19T00:59:55Z，main @ `b30a8cca91561c8dfbcc0511e5c86b44e17a691b`）；[#205256](https://github.com/pytorch/pytorch/pull/205256)（2026-11-18T01:02:19Z，main @ `a8df3d9dbc36ec2a22fa076cfd7404493d4f0640`）。

### PyTorch：编译的 DTensor checkpoint 保存随机结果，修错梯度但增加存活量

DTensor 随机算子被封装为 run_dtensor_rng_op HOP，原 checkpoint 重算未经过普通 nondeterministic_seeded RNG 状态回放，导致 backward 使用不同 dropout mask。现将 HOP 及结果 getitem 标为 MUST_SAVE，反向直接读前向结果；若 selective policy 强制 MUST_RECOMPUTE 则报错。作者复现显示前向正确、梯度却错误，覆盖 aot_eager/Inductor、Shard/Replicate。

**建议优先回归编译+DTensor+随机算子+activation checkpoint 组合**。代价是这些随机结果不能再靠重算省显存；不能只比较前向或把该修复写成纯内存优化。

一手来源：[#201456](https://github.com/pytorch/pytorch/pull/201456)（2026-11-18T15:30:00Z，main @ `270998c816acb1ffab58639ee5e1a78b95b440e3`）。

### PyTorch：CuTeDSL 用户 kernel HOP 加函数化与写别名检查

注册 CuTeDSL 的 mutation/functional HOP、side table 与 user-kernel DSL；函数化对写参数克隆并回写，无法证明两个写入参数不重叠时拒绝，以免独立 clone 的回写丢失另一处变更。**建议继续跟踪 PT2 自定义 kernel 集成**：本轮完成的是 HOP 与函数化层，源码明确 wrap_cutedsl、Dynamo、Inductor 与 AOTInductor 捕获/导出还分后续 PR，不能宣布整套 CuTeDSL torch.compile 支持已全部落地。

一手来源：[#203538](https://github.com/pytorch/pytorch/pull/203538)（2026-11-18T10:02:17Z，main @ `c9f20be011af11bffd9b2bc23196b0bb576661de`）；[#202443](https://github.com/pytorch/pytorch/pull/202443)（2026-11-18T10:02:17Z，main @ `7a39e3dcef2c4809cabd9d8d9b1e79173d7968b5`）。

### FlashInfer：KDA/GDN 增 batch-invariant 类，性能代价已披露

`recurrent_kda(batch_invariant=True)` 对应 kda.bf16.c16，`chunk_gated_delta_rule(batch_invariant=True)` 对应 gdn.bf16.c64；按固定 chunk 算术组织递推，提供 fingerprint 和失败即报错、不静默回退的类型契约。KDA 使用 FP32 state pool，GDN 与 KDA 的 chunk 长度及完成步不同，不能混作一个确定性开关成本。

作者在 B200/GB200/GB300/VR NVL4/NVL72 用 cold-L2 CUPTI GPU time、双新进程反转测量顺序报告：KDA 小 decode 的“原最快 kernel 时间/新类时间”为0.68–0.75，大 decode约0.95–1.00；GDN小decode为0.65–0.71，大decode为0.70–0.77，完成步只有约0.25–0.26。比值<1是变慢，GDN额外读取开放chunk行是主要带宽成本之一。**建议深入阅读并针对复现需求考虑采用**，优先评估确定性价值与吞吐代价，不将它包装成加速。此处为主分支增量，不能据同日0.7.5发布推定已进入该wheel。

一手来源：[#9294](https://github.com/flashinfer-ai/flashinfer/pull/9294)（2026-11-18T18:22:32Z，main @ `f946321c43fe9960f02245072805d4054d37bec3`）；[#9405](https://github.com/flashinfer-ai/flashinfer/pull/9405)（2026-11-18T10:50:21Z，main @ `1dd551c1d94f43a4c95f6d27149e319eb344e57a`）。

### FlashInfer：GDN 仿射序列分段扩大适用范围，批量越大未必越快

`backend="cake_gdn", use_affine_split=True` 把序列按 chunk 分段，第一遍计算分段状态的 transfer/offset 和输出系数，第二遍组合入状态并纠正输出；数学上是精确仿射组合，但不是浮点 bitwise 等价承诺。先在 SM103/N=1 落地，同日扩到 N≥1 与 SM100/107；默认 False，BF16 q/k/v、FP32 gate/state、head dim128。

GB300、CUDA13.4、torch2.14.1、cold-L2 CUDA graph 测量中，N=2、(qheads,vheads)=(2,4)、8K 对 unsplit 最高1.88×，但常慢于原 CP default；N=4 几乎所有真正分段的形状慢于两基线，16/32 value heads 在N>1未胜unsplit。**考虑复现单条/两条长序列**，不要只引用相对 unsplit 的最大倍数；同日扩硬件不能继承 GB300 的速度数字。

一手来源：[#9628](https://github.com/flashinfer-ai/flashinfer/pull/9628)（2026-11-18T02:36:14Z，main @ `a7d9e18658d3954527675a3d3a9e7b2ec75b2eae`）；[#9663](https://github.com/flashinfer-ai/flashinfer/pull/9663)（2026-11-18T18:38:10Z，main @ `6a4e4d5c0f3827b4e4b82c6249f5198262a9e7a6`）；[#9669](https://github.com/flashinfer-ai/flashinfer/pull/9669)（2026-11-18T21:11:49Z，main @ `a1e0200c1e5ab0a3e78031a09322f80762bcee01`）。

### FlashInfer：Mamba SSD 直接按槽读写调用方状态池

packed-varlen Cake SSD combined 新增 per-sequence slot index：一次调用直接从 caller state pool 读初态并写末态/checkpoint，省去框架 gather、无初态行清零、dense call、scatter 的独立临时流程。另支持显式 state_dtype、caller-owned final_states；不需要末态时不分配其存储，并校验状态寻址32位上限、buffer重叠及参数一致性。

作者以 Nemotron-H Nano/Super 的 per-TP shapes（64/128heads、head_dim64、state_dim128、FP32 pool、每序列一个checkpoint）48形状 CUDA graph 对照完整 gather→旧Cake→scatter 流程，报告B200/GB300几何平均2.10×、VR200为2.68×。**建议深入阅读并在类似池布局复现**；这是状态搬运加该算子的局部流程，不能当完整模型吞吐倍数，也不能归因于 SSD 算法本身变快。

一手来源：[#9644](https://github.com/flashinfer-ai/flashinfer/pull/9644)（2026-11-18T18:29:03Z，main @ `4a936721275d5e0d50ff4cc08805669556885a5b`）；[#9550](https://github.com/flashinfer-ai/flashinfer/pull/9550)（2026-11-18T03:44:16Z，main @ `6919f3d9d4db8c5cb8ffb4d83aac360179acd15b`）。

### FlashInfer：MegaMoE 的 NVLS 原语收益没有变成端到端收益

NVLS barrier 用一次 multimem reduction 替代 peer-atomic循环，由 `MEGA_MULTIMEM_BARRIER=1` 显式开启；默认off，无NVLS则回退，按需创建映射避免耗尽NVSHMEM team。GB200、world2–16 barrier微基准约1.8×，但作者明确EP4、8/64tokens整核未改善：约2µs barrier不是该形状的主瓶颈。**继续观察更高EP或同步受限负载**，这是值得保留的负面证据。

独立的SM107 Cake NVFP4 MegaMoE新增route-major与rank-major去重combine，4×VR NVL72、CUDA13.5、DeepSeek-V4-Pro MoE切片（48experts/top6/H7168/I3072），warm-L2、两轮交错测试的MoEEpLayer.forward对已调优CuTeDSL为1.02/1.10/1.12×（每rank1024/4096/8192tokens）。trtllm-gen NVFP4另增加双die locality domains：FC1按intermediate列、FC2按hidden列拆到本die权重与green-context流，routing/finalize仍在caller流。**有对应硬件时考虑复现**，不能跨SM或跨实际模型路由外推。

一手来源：[#9389](https://github.com/flashinfer-ai/flashinfer/pull/9389)（2026-11-18T08:30:16Z，main @ `d6dc1bec61d669c0aa0fe4f6643a3c8b3b570b8d`）；[#9603](https://github.com/flashinfer-ai/flashinfer/pull/9603)（2026-11-18T07:49:45Z，main @ `820766cc90a2ac9492e086372e46212f88ed5e32`）；[#9538](https://github.com/flashinfer-ai/flashinfer/pull/9538)（2026-11-18T04:55:12Z，main @ `8fcbf653c03f6649e096a256ecc8ac86bab850f9`）。

### FlashInfer 0.7.5：正式工件可用，新增主线能力要另核版本

正式 tag 固定为 `9aa920602276537b27e017a762a71e3437fc7f0d`（11-18 23:03:58Z），PyPI wheel首传23:14:40Z。实际 rc3→stable 对比只有5个提交：文档一致性、#9354 torch.compile(dynamic=True) stream-cache回归、#9260 NVFP4 pre-FC2 routing weighting，以及两次版本号变更。#9260 的原始主线工作已在11-14日报收录，本次增量是进入正式发布，而非新算法。

#9354 在Dynamo tracing时使用可追踪的 current_stream，eager保留raw-stream binding，修复top_k/GEMM工作区遇符号device index不能编译；作者在L40S、torch2.14.1+cu130验证12项stream/graph测试及11项SGLang采样测试。#9260保留逐步BF16 rounding并在GEMM1 epilogue直接权重化/量化进FC2，GB300指定形状可省大中间buffer，但作者16×GB300内部服务吞吐仍在噪声内不变。**建议基于具体回归升级**，不要把今日default-branch的GDN/KDA等PR自动视为随0.7.5发布。

一手来源：[v0.7.5](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.7.5) · [PyPI 0.7.5](https://pypi.org/project/flashinfer-python/0.7.5/)；[rc3→正式版实际差分](https://github.com/flashinfer-ai/flashinfer/compare/v0.7.5rc3...v0.7.5)。

### vLLM-Omni 0.34.0rc1：强制 V2 的迁移成本需要先核对

首个0.34.0 RC固定在alignment merge `709dc427ef59b9a1a1e0b092efd0236d3f70f723`，与vLLM0.34.0配套，Python3.11–3.13，移除ModelRunner V1。官方列出受影响的Drama/Fish-Omni、LFM3-Omni、PersonaPlex、YuE2及部分Qwen3-TTS/Omni ROCm/MUSA V1配置；不能把“发布了新版”当这些原工作流仍可运行。native 0.34 Ascend未qualified，旧0.33镜像止于API导入错误。

release页面保留“PyPI pending”字样，但当轮PyPI已可核验0.34.0rc1于11-18 09:40:11Z首传，区分页面文案与artifact事实。后续主线VoiceChat #8597按音频window长度与encoder chunk尺寸/stream-start drop分批，保留各request cache；验证CPU回归及A800真权重，未验证NPU/XPU。**考虑复现前先审退休profile**，RC不等同正式稳定发行。

一手来源：[#10365](https://github.com/vllm-project/vllm-omni/pull/10365)（2026-11-18T06:29:23Z，main @ `709dc427ef59b9a1a1e0b092efd0236d3f70f723`）；[#8597](https://github.com/vllm-project/vllm-omni/pull/8597)（2026-11-18T02:15:33Z，main @ `0698abe49253fdf51da9c584f06a65f75830c95d`）；[v0.34.0rc1](https://github.com/vllm-project/vllm-omni/releases/tag/v0.34.0rc1) · [PyPI 0.34.0rc1](https://pypi.org/project/vllm-omni/0.34.0rc1/)。

### Prime-RL：MXFP8 反向与 offload 峰值按组削减

LLEP MegaMoE在SM100接原生MXFP8 backward，依赖prime-mega-moe `e5601cf`；4×B200、EP4、DeepSeek-V4-Flash MoE形状、4096tokens/rank，输出/梯度对FP32参考相对L2约6.6%–6.8%，与普通MegaMoE MXFP8的y/dx bitwise相同不意味着接近FP32。offloaded optimizer改为每param group搬入state、step后释放该组grad，避免FP32参数+梯度+全量Muon/AdamW state同驻峰值。作者用匹配2节点16H200每卡规模的1节点、262144长度假数据复现由OOM到可训练，并有bit-exact优化器/检查点测试。

另Hopper fused SFT/pretraining loss修复quack跨CTA原位覆盖race：提前gather label logits作为单独输入；原现象仅约4e-5–8e-5tokens损失错误，源文称梯度和RL logprob路径不受影响。**建议深入阅读并在自有精度/峰值标准下复现**，这三项不能合成一个端到端加速数字。

一手来源：[#4599](https://github.com/PrimeIntellect-ai/prime-rl/pull/4599)（2026-11-18T22:28:31Z，main @ `87609b2fee3bc946d1df632053e21d4903af3540`）；[#4563](https://github.com/PrimeIntellect-ai/prime-rl/pull/4563)（2026-11-18T21:02:33Z，main @ `3818446bcf25e7470c0711538e38377149094252`）；[#4582](https://github.com/PrimeIntellect-ai/prime-rl/pull/4582)（2026-11-18T20:11:54Z，main @ `3fdd88b0c46b263ba0bdc3cb6aac3e917932576c`）。

### llama.cpp：递推 NaN 清零与超大张量寻址纠错

递推cache重置改用fill而非依赖乘零，避免NaN×0仍NaN使下一序列继续污染；新增以0xff填状态的跨序列回归，涵盖conv/SSM及DeepSeek V4 compressor状态收集。GGUFReader修复tensor offset溢出，属于大模型文件边界。OpenVINO则同时修SWA/full-attention同extent时mask和KV索引隔离、从mask shape推导动态尺寸，避免中途窗口出现时重编译；源代码仍把某些等extent组识别记作模型级heuristic。**对应后端应优先回归正确性与复用边界**，本轮未测NPU/GPU加速，不能从“expand support”标题外推所有模型。

一手来源：[#31913](https://github.com/ggml-org/llama.cpp/pull/31913)（2026-11-18T23:08:13Z，master @ `0a76757469ea94ad5f88d14e18cc7370b9931f99`）；[#32495](https://github.com/ggml-org/llama.cpp/pull/32495)（2026-11-18T14:28:47Z，master @ `b069e526deabf7207906ddfae1eb61551b2f612c`）；[#32172](https://github.com/ggml-org/llama.cpp/pull/32172)（2026-11-18T07:56:21Z，master @ `4577932fab595eaa8b27ad71734d5de1f3c07d8c`）。

### FlashAttention-3：长序列反向的累积缓冲索引改为64位

SM90 backward 的FP32 dq/dk/dv accumulator尺寸、batch stride和block offset原用int乘法，超过2^31−1会绕回；源文示例包括batch2×32heads×524288queries的stride以及约2^24+128tokens的单head尺寸。补丁贯穿分配、传入shape/stride、GQA epilogue及varlen offset，并补SM80 dQ路径。**长上下文训练应考虑回归**，这是溢出边界修复，不表示这些极长长度的设备容量或模型质量已获验证；同日FA4 beta40的tag也携带该源码修复，见下文发行范围；这里说的是FA3执行路径，不代表FA4 kernel的同项验证。

一手来源：[#3152](https://github.com/Dao-AILab/flash-attention/pull/3152)（2026-11-18T03:47:24Z，main @ `68bd095cdd338c5171bccb9c200529f3620b85c9`）。

### Miles：采样支持集合使用扁平数组传输，增量渲染只处理必要尾部

rollout将sampling mask和top-k候选请求为flat base64 arrays，score-centering与sampling-support replay共享同一组支持ID/概率；router必须保留对应flat字段，老schema可能丢掉它们。DeepSeek-V4/V4.1会话追加则从最后assistant turn重新render尾部（必要dummy user），依赖recipe不跨assistant边界折叠/排序/编码的性质，测试要求每次和全历史render的token相等。空CP rank在score-centering路径保持FP32 logprob dtype，避免空分片拼接改变契约。**建议按实际router与模板版本复现**，没有公开统一吞吐数字，不能推广到任意chat template的尾部渲染。

一手来源：[#4815](https://github.com/radixark/miles/pull/4815)（2026-11-18T18:26:17Z，main @ `c38b9e8dafcaead456bf27f31c8c9c215c1e52cc`）；[#4930](https://github.com/radixark/miles/pull/4930)（2026-11-18T09:19:15Z，main @ `8a2aa0b3cbaec01922e18071ab4a66ab5b403b3d`）；[#4489](https://github.com/radixark/miles/pull/4489)（2026-11-18T01:28:30Z，main @ `55766eb710afe0dec5c48e71bc30931cbef01849`）。

### SGLang：Kimi-K3.1 在 gfx950 连通稀疏 MLA、KDA 与 MoE

主线为 MI350X/MI355X（gfx950）组合接通 Kimi-K3.1：FP8 sparse MLA、DCP top-k/FP4 index-K、KDA speculative verify、A8W4 MoE 与低时延 collectives。KDA verify 融合宽4的causal conv、gate和多token递推并写ReplaySSM ring；MXFP4 indexer是独立开关，不是所有attention计算降到FP4。环境默认要求各rank一致使用Rust TreeCore，避免混用Rust/Python cache实现使HiCache reclaim分歧。**有对应硬件可考虑复现**，源码支持限定模型、ISA与cache格式；本轮未验证AMD权重/算子数值和端到端服务，不以“新增支持”推定与NVIDIA同吞吐。

一手来源：[#51000](https://github.com/sgl-project/sglang/pull/51000)（2026-11-18T11:58:30Z，main @ `ea93e2fcd0d1566f4817c0fe8ddfe6d8006908fd`）；[#50998](https://github.com/sgl-project/sglang/pull/50998)（2026-11-18T05:00:47Z，main @ `9e5c4a78e82823375115926e651337f335641771`）；[#50997](https://github.com/sgl-project/sglang/pull/50997)（2026-11-18T04:45:17Z，main @ `8d72d2ea11901b49d694bcea543f1cffccf27629`）。

### SGLang：CPU 量化与 Ascend ModelSlim L2 KV 的格式契约

CPU新增FP8/INT8量化算子与compressed-tensors接入，包含CPU向量路径和标量回退，不意味着所有量化kernel都使用低精度矩阵乘。Ascend950的L2 HiCache则按checkpoint声明区分MLA FP8 latent+BF16 RoPE与GQA MXFP8 K/V+动态K scale；静态per-channel V scale仍常驻NPU。GQA路径文档要求CANN9.2.0 Quant Flash Attention，MLA默认ACL-registered CPU memory，GQA使用pinned memory；不设置storage backend时为本地L2而非分布式L3。

**建议按格式与平台逐项复现**，以`cached_tokens_details.host`确认真实L2恢复。只恢复KV bytes却漏scale/辅助buffer会破坏数值，不能把此次支持外推到任意ModelSlim checkpoint或旧Ascend硬件。

一手来源：[#41329](https://github.com/sgl-project/sglang/pull/41329)（2026-11-18T01:53:37Z，main @ `bff44f5fd7a1f5d7662c45382be4aa8011ed4225`）；[#50023](https://github.com/sgl-project/sglang/pull/50023)（2026-11-18T01:33:05Z，main @ `8c6ee5d5e6e37dfdc9af8558ffbb582ad4d9f7b4`）。

### vLLM ROCm：HIP VMM 共享 Engram 主机表有驱动回收限制

`VLLM_ROCM_ENGRAM_HIP_VMM=1`可把DeepSeek-V4.1的跨DP共享Engram表放到HIP VMM host allocation，代替原/dev/shm路径；任何VMM失败让全部replica一致回退。开关默认off，官方特别注明部分amdgpu驱动（例6.16.6）在replica退出后仍不释放host memory、直到重启，对应ROCm issue仍是依赖边界。**继续观察或在可控环境复现**，不能只以填表更快就建议生产启用；需量测启动、退出及失败后的主机内存生命周期。

一手来源：[#65595](https://github.com/vllm-project/vllm/pull/65595)（2026-11-18T03:13:44Z，main @ `bd455e1f91af85dc029784da63aabd64e0f78840`）。

### FlashInfer：DSv4.1 FP8 sparse-MLA 增 split-KV 与按 wave 调度

新的协作CuTe C++ decode按query/head tasks、SM数量和resident CTA waves在auto/decode-splitkv/decode间选择；八head任务32-candidate tile、十六head64 tile，以独立loader warp和双缓冲bulk gather供数。mega路径把query RoPE/量化、稀疏QK-softmax-PV/sink、BF16 context rounding、逆RoPE和FP8输出/scales放在同一attention launch并做cluster finalize。学习投影、query norm、cache构建和稀疏索引仍由调用方负责。

inline/footer FP8 cache以及独立main/extra page size、gap、非零对齐origin等有对应边界处理。**建议深入阅读并在模型真实稀疏分布上复现**；不能把融合attention核当完整DSv4.1一步，也不能忽略上游indexer成本。本轮未给跨模型统一加速数。

一手来源：[#9274](https://github.com/flashinfer-ai/flashinfer/pull/9274)（2026-11-18T01:15:15Z，main @ `01962ffad7e26447e21ed3de4e11c5c9923a5d07`）。

### Megatron-Bridge：量化混合模型恢复必须回到原 TE stack

恢复量化hybrid checkpoint时继续使用量化时的TE stack spec，而不是改成ModelOpt SequentialMLP：后者的expert quantizer key与TEGroupedMLP checkpoint不匹配。另为ERNIE5.5的FP8 checkpoint导入按block scale反量化权重。**有对应工件应考虑做导入—保存—恢复数值回归**；这提供兼容的导入/恢复路径，不表示训练期间仍保持同一种FP8算术，也没有本轮端到端显存收益证据。

一手来源：[#6967](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6967)（2026-11-18T11:08:43Z，main @ `137121984ab50b00c7ee5ba38af9086b0ed2faab`）；[#6850](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6850)（2026-11-18T09:39:02Z，main @ `57fa5e41a3a100fa0059d1c9ccdc0d58bb169cec`）。

### Dynamo Kimi-K3 post.1：发布的是实验分支，跑分用了合成接受长度

`v1.5.1-kimi-k3-post.1`（11-18 02:29:37Z）固定 `35fdab07c13ee4b266ffd569665461f12dfd257b`，基于Dynamo1.5.1/SGLang0.5.18，镜像加入DCP×DSpark草稿KV传输和SM100 MoonViT guard。GB300的aggregated/1P1D recipe均为16卡、MXFP4专家与FP8KV；64K输入/400输出/90%KV reuse agentic trace下，公布aggregated91.3输出tok/s/GPU、用户P50为55.6tok/s、TTFT486ms（并发36），1P1D为62.0/77.6tok/s/4793ms（并发18）。

**继续观察并仅作实验复现**：这些数字强制合成spec acceptance length4.05，不能声称是真实DSpark验证结果；两负载并发也不同，不能由用户速度判断拆分普遍更优。发布明确未QA-gated、不推荐生产，GB200仍钉旧镜像，无H200 profile，某些请求字段不支持；长prefill还可能触发3秒健康检查重启。

一手来源：[v1.5.1-kimi-k3-post.1](https://github.com/ai-dynamo/dynamo/releases/tag/v1.5.1-kimi-k3-post.1)。

### FlashAttention beta40：发布范围与执行路径要分开解读

`fa4-v4.0.0.beta40`（11-18 08:12:49Z）tag指向 `68bd095cdd338c5171bccb9c200529f3620b85c9`。release汇集CuTe空分页表/2^31键数/巨大local-window边界、FA2 LSE offset、FA3尾部清零及64位反向索引，以及SM100的non-causal native MXFP8 PV forward。上期已报道的#3020/#3174此次只是进入新beta发行，不重述为新算法。**考虑按实际执行路径回归**，tag名虽是FA4 beta，仓库发行可携带FA2/FA3源码修复；须区分打包可用与每个FA4 kernel均已完成相同验证。

一手来源：[fa4-v4.0.0.beta40](https://github.com/Dao-AILab/flash-attention/releases/tag/fa4-v4.0.0.beta40)。

<a id="topic-6"></a>
## 硬件与软件协同

### Mirage→Triton NPU：表示可移植，搜索空间和强基线不自动移植

- **机构/项目与时间**：Hryniewska-Guzik, Weronika / Comblik, Mateusz 等（Mirage→Triton NPU）。arXiv `2611.22844v1`，提交 Tue, 17 Nov 2026 14:25:11 UTC；11 月 18 日分类公告中观察到，本轮 11 月 19 日发现；提交日期与首次公开日期分别处理。
- **一手来源**：[固定版论文](https://arxiv.org/abs/2611.22844v1)、[全文](https://arxiv.org/html/2611.22844v1)。
- **具体增量**：将CUDA-oriented图超优化迁移至Ascend910B3/Triton，分析UB容量、AICore grid、threadblock fusion、online softmax与验证成本未充分目标化的问题。
- **证据与限制**：§3–5/Table1：910B3 64GB、每kernel1,000次、allclose容差1e-2。RMSNorm+MatMul快27.6%–87%与gated MLP快4.2%–58.4%的基线为DeepSeek生成代码，不能称超专家kernel；大GEMM N8192/16384反而慢127.6%/186%。attention三shape有两快、一慢，未复现完整FA式变换。
- **实践意义与建议**：**建议深入阅读**。对NPU编译器值得深入读负面机制，优先复查成本模型与后端lowering而非机械复用CUDA搜索；本文没有端到端LLM吞吐结论，本轮未核公开迁移实现。

### ROCm 10.2：框架包、通信与可观测性一起升级

AMD 11月18日正式介绍ROCm10.2.0。vLLM0.32.0、SGLang0.5.21服务构建迁往Python3.14，新增PyTorch2.15 wheels；TheRock的PyTorch包改把ROCm作为独立依赖。官方vLLM镜像为Instinct预构建AITER kernel，Radeon/Ryzen部分路径首次运行编译。RCCL针对PCIe GPU调优，厂商称在8卡Radeon服务器上验证4路TP vLLM，未给可比延迟数字。

系统分析新增设备全程SPM计数（gfx1200/1201/1250）、实验性统一rocsys入口及host直接attach容器；AIS/hipFile遥测可与GPU timeline关联。**考虑在隔离环境复现升级。** 先锁镜像、框架、AITER/RCCL版本并检查新CLI参数和包依赖，不能把框架更新直接理解为所有现有模型吞吐提升。[官方博客](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-10.2-blog/README.html) · [正式release notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html)

### 补录｜NVIDIA 用 indexer 的真实成本约束稀疏注意力设计

11月17日技术文章指出，核心attention变稀疏后，仍扫描长上下文的indexer和Top-k可成为瓶颈；不能只比较核心attention FLOPs。其FP8内核默认测试64K indexer keys、压缩比4、head128、KH1、prefill batch1/decode batch128。减少key heads并跨query heads归约，投机查询在单次key读取内复用；建议把group size×验证query数控制在单Tensor Core tile约200行（FP16 epilogue可256）。块级选择便于批量读KV，Top-k融合能免去完整logits落地。

**深入阅读并考虑做联合消融。** 文中大多为相对kernel sweep，页面未完整给出所用GPU/服务并行等条件，也没有证明所有精度/质量取舍；FP16累加需控制QK数值范围。不要把“draft近乎免费”的局部区间当作端到端投机收益。[官方原文](https://developer.nvidia.com/blog/co-designing-sparse-attention-for-efficient-long-context-inference/)

### TorchTPU：编译缓存必须隔离设备分配与 matmul 精度

Google 于 11 月 18 日合入两项会影响基准与分布式正确性的修复。此前无 collective 的图不把默认设备分配计入 AOTAutograd 缓存键：共享磁盘缓存先跑 world-size=1、再跑 TP>1 时，其他 rank 可能载入单设备 executable，报错后挂住。现在缓存键加入默认 device assignment。同日把实际 lowering 使用的 TPU matmul precision 同时纳入磁盘缓存键和 Dynamo guards，避免修改精度后沿用旧图；异步编译线程采用全局 precision，不能假设继承调用线程的局部上下文。补丁分别带跨 world-size 缓存复用和 precision 回归用例。**考虑采用。** 做精度和 TP 横向对照前先固定包含修复的提交，检查缓存命中条件；这些是主分支修复，未据此确认正式 wheel 已包含。

版本：链接日期为UTC提交日期；各commit已进入默认分支，未另标tag时不等于正式包已发布。 [google-pytorch/torch_tpu 2026-11-18 @2aa3ee53c03a](https://github.com/google-pytorch/torch_tpu/commit/2aa3ee53c03af388fdc62783aae6a236ccde4942) · [google-pytorch/torch_tpu 2026-11-18 @1d8348cc07b6](https://github.com/google-pytorch/torch_tpu/commit/1d8348cc07b6f29029f25fa95b13a72428324dc9)

### TorchTPU：MoE 去掉权重转置中间件，线性层只在 bias 后舍入

2D×3D grouped_mm 的 RHS 转置可在 libtpu ≥ `0.0.53.dev20261110` 时折叠进 RaggedDot contracting dimension，旧版保持显式转置。提交说明称 Qwen3.5-397B、Qwen3-Coder-30B 等大 MoE 受旧 SparseCore relayout 影响，峰值额外 HBM 最高约 8 GiB、decode device time 最高约 3 倍；未提供足以跨硬件比较的完整服务负载，不能当作端到端收益。另一提交使带 bias 的 BF16/FP16 linear 在 FP32 dot 和 bias 加法后才转回输出精度，消除二次舍入及部分提前溢出。**考虑复现。** 先检查 libtpu 版本门槛，再用内存 trace 与相同输出误差标准比较；本轮未独立验证 libtpu 内部实现或设备数值。

版本：链接日期为UTC提交日期；各commit已进入默认分支，未另标tag时不等于正式包已发布。 [google-pytorch/torch_tpu 2026-11-18 @98779bd5ab02](https://github.com/google-pytorch/torch_tpu/commit/98779bd5ab028da2d7033fd624a7bda6ee86c47f) · [google-pytorch/torch_tpu 2026-11-18 @2b69edb165d2](https://github.com/google-pytorch/torch_tpu/commit/2b69edb165d233a4ef9a6f67d374733591fe9119)

### SGLang TorchTPU：统一 KV 池接通投机、HiCache 与等 TP 的 P/D

11月18日至19日UTC的三项配套提交为 TpuUnifiedSWAKVPool 接入 EAGLE/EAGLE3、MTP、DFlash、DSpark 的草稿 KV。目标与草稿内存共同计价，通过 sidecar 映射紧凑 SWA 页；HiCache 增加 DRAFT/DRAFT_SWA 主机侧对象和对应页转换。Raiden 用独立草稿 engine 传输整请求、前缀命中及分块拉取中的草稿 KV，明确要求 `TP_P == TP_D`。只接受草稿层全 Full 或全 SWA，SWA 草稿窗口不得宽于目标；混合 Full/SWA 草稿等组合提前拒绝。补丁包含统一池投机、HiCache 和 P/D 用例，未给通用吞吐增益。**考虑复现。** 核对目标、草稿、冷页及释放 fence 的共同容量，而不能只按目标 KV 预算部署。

版本：链接日期为UTC提交日期；各commit已进入默认分支，未另标tag时不等于正式包已发布。 [sgl-project/sglang-torchtpu 2026-11-18 @bfadf1bf7bab](https://github.com/sgl-project/sglang-torchtpu/commit/bfadf1bf7baba2c826b46770a1930771c0fb5ce2) · [sgl-project/sglang-torchtpu 2026-11-18 @2a5454265bd2](https://github.com/sgl-project/sglang-torchtpu/commit/2a5454265bd269ad1f2983d1ce3549fd383b0051) · [sgl-project/sglang-torchtpu 2026-11-19 @f8a8a58472e5](https://github.com/sgl-project/sglang-torchtpu/commit/f8a8a58472e598fd26094f61ebca35c70fca672d)

### SGLang TorchTPU：跨主机 DP 副本与 Host Engram 异步搬运

DP 副本由单 host 扩展到多个 host：总节点数须整除 dp_size，每个副本使用连续 host 块，node 0 提供统一入口；通过 `SGLANG_TPU_DP_REPLICA_SHAPE` 指定切片内副本形状，多 host 副本禁用 tier-2 编译缓存，node-leader fan-out 仍不支持该模式。另一项 DeepSeek-V4.1 Host Engram 路径把输入准备与 TPU embedding 重叠，采用双缓冲 raw DMA，并补 CSA2/indexer 越界 mask；warmup、投机验证、beam 及 PP/CP/DCP>1 等条件走既有回退。**深入阅读。** 这是各自受拓扑和执行模式约束的部署扩展，尚无本轮独立多机性能验证。

版本：链接日期为UTC提交日期；各commit已进入默认分支，未另标tag时不等于正式包已发布。 [sgl-project/sglang-torchtpu 2026-11-18 @22a4bc39c32f](https://github.com/sgl-project/sglang-torchtpu/commit/22a4bc39c32fb51143c8ee51febfc603527c762c) · [sgl-project/sglang-torchtpu 2026-11-18 @1ca49ee6b60c](https://github.com/sgl-project/sglang-torchtpu/commit/1ca49ee6b60cf83ead806f8da0cb2bd0db49d01e)

### vLLM TorchTPU：MoE 路由回放与 megakernel 基础件

新增 TPU routed-experts capturer 及其适配测试，为采样时记录专家选择提供接口；megakernel 目录新增 Pallas top-k、去重、expert ring 与路由测试。FP4 流式权重另把一个 slab 内各 chunk 的等待集中在首次读取处，减少在展开 matmul 链上重复插入调度 fence。**继续观察并按需验证。** 这些提交证明基础模块与特定内核路径已合入，不能据此认定完整 megakernel MoE 服务或训推路由回放闭环已经验证，也没有可比较的端到端加速数字。

版本：链接日期为UTC提交日期；各commit已进入默认分支，未另标tag时不等于正式包已发布。 [vllm-project/vllm-torchtpu 2026-11-18 @14f9e37df86a](https://github.com/vllm-project/vllm-torchtpu/commit/14f9e37df86a64be5b211995b0901fc31ba0e869) · [vllm-project/vllm-torchtpu 2026-11-18 @b0b4a74165fe](https://github.com/vllm-project/vllm-torchtpu/commit/b0b4a74165fe06a4c57f67d34b756da595647a42) · [vllm-project/vllm-torchtpu 2026-11-18 @8c271f074104](https://github.com/vllm-project/vllm-torchtpu/commit/8c271f074104adaa58918aa3bdef3bfea7217bd3)

### TPU inference：多 tray VM 的芯片排序和多进程 libtpu 加载

TPU_VISIBLE_CHIPS 按 PCI/NUMA tray 顺序生成，并为多进程 `(2,2,1)` tray 设置 ALLOW_MULTIPLE_LIBTPU_LOAD，配套更新 worker 与启动脚本测试。**考虑采用。** 多 tray、多进程服务应显式记录逻辑芯片到物理 tray 的映射，避免把启动成功误当作拓扑配置正确；本轮未在该拓扑实测通信或推理性能。

版本：链接日期为UTC提交日期；各commit已进入默认分支，未另标tag时不等于正式包已发布。 [vllm-project/tpu-inference 2026-11-18 @2f3eb48aeada](https://github.com/vllm-project/tpu-inference/commit/2f3eb48aeada6063073ce55621317ebaf42b95b8) · [vllm-project/tpu-inference 2026-11-18 @276f885c1c67](https://github.com/vllm-project/tpu-inference/commit/276f885c1c6717ed1ed4c72a31f2796f3c741455)

### Gaudi：专家全局 ID 误当本地槽会静默丢路由

vLLM-Gaudi 修复 HPU FP8 MoE 在 EP 下没有消费 expert_map 的问题：现在用全局到本地映射转换路由 ID，并以 -1 表示非本 rank 专家。旧逻辑不仅不适用于 round-robin 和专家数不能整除的分配，在均匀连续分配下也会丢掉应属本卡的专家。提交报告在 SynapseAI 1.24 的 Gaudi 2/3 小例复现，在 Gaudi 2、GLM-5.3-Flash、288 experts/4 ranks 中 EP 输出塌缩为重复 token，修复后与非 EP 对照吻合。**优先考虑采用与回归。** 这是静默正确性问题。修复范围是 `HPUFp8MoEMethod`；opt-in `VLLM_HPU_MOE_GATHER` 仍接收原全局ID，`hpu_fused_moe`、`hpu_mxfp4`、compressed-tensors等同类偏移路径未一并修复。报告中的 BF16 直调算子复现与集成 FP8 路径须分开理解，不代表所有量化及放置组合已得到独立验证。

版本：链接日期为UTC提交日期；各commit已进入默认分支，未另标tag时不等于正式包已发布。 [vllm-project/vllm-gaudi 2026-11-18 @ba917eec78ab](https://github.com/vllm-project/vllm-gaudi/commit/ba917eec78abeb00b3b724065fe1061ac6b51318)

### Ascend：A5 单卡精确专家卸载支持 MRV2 eager 与 decode 图

vLLM-Ascend 新增默认关闭的 exact expert offload。目标模型路由专家权重与 scales 留在 pinned host memory，被选专家送入各层稳定设备槽；prefill、混合批次及 decode 容量溢出复用完整专家池。支持 Ascend 950PR/950DT 单设备、MRV2 eager 及 FULL_DECODE_ONLY ACL 图，含图回放传输与失败处理；同步和异步调度均按 worker 串行发布权重。只卸载 target 专家，DSpark/MTP 草稿专家仍常驻，不能把它理解为多卡或任意草稿卸载。补丁提供 A5 两个聚焦 E2E 案例，未提供可横向比较的吞吐数据。**考虑复现。** 以显存可装入性、主机带宽和长尾时延三项一起评估，固定 `num_device_experts` 与命中率。

版本：链接日期为UTC提交日期；各commit已进入默认分支，未另标tag时不等于正式包已发布。 [vllm-project/vllm-ascend 2026-11-18 @21d07afb5df2](https://github.com/vllm-project/vllm-ascend/commit/21d07afb5df28c3383a2e8a2904b668a12c7edc9)

### Ascend：A3 共享 Engram 改用 Host VMM，保留集体回退边界

针对 DeepSeek-V4.1 Flash、A3、TP1/DP16 的共享 CPU Engram 第二表 scale 注册失败，新增 node-local Host VMM。rank 0 发布 checkpoint 行，各本地 rank 导入自身映射；全 rank 先用 1 GiB 对齐 probe 验证分配、导出、导入和映射，再决定启用。失败时只有协调清理成功才集体回退 HostRegister，清理或 collective 失败直接终止。原始日志见 -ENOMEM，但耗尽的是何种设备侧资源仍未确证。当前使用 generic HOST placement，**不保证 NUMA 本地性或性能等价**。**深入阅读后考虑采用。** 重点验证故障回退与内存放置，避免把“VMM 可用”理解为原故障根因已完全解决。

版本：链接日期为UTC提交日期；各commit已进入默认分支，未另标tag时不等于正式包已发布。 [vllm-project/vllm-ascend 2026-11-18 @e2f6cb1b46d0](https://github.com/vllm-project/vllm-ascend/commit/e2f6cb1b46d0eeaa1a93e2ecc8cdce819f4b33a4)

### Ascend：KDA 联合 TP×PCP 切分和分块缓存 checkpoint 修复

GLM-5.3-Flash KDA 的投影、卷积权重和递归状态采用 TP-major 的 TP×PCP head 切分，匹配 Mooncake V2 状态传输；要求头数可整除 TP×PCP，RecoverSSM 支持的是 prefill producer，PCP consumer 与 colocated PCP 仍不支持，其余 PP=1、DCP=1 等门槛保留。另一项修复 consumer 带 computed prefix 时跳过普通 chunked prefill 对齐，漏掉可复用 Mamba/KDA checkpoint 的问题。提交用 prompt=2305、cache granularity=1152 展示：应保留位置2304，单看 external hit=2304 仍可能恢复错误状态；给出 raw-logit 误差复现及调度回归。**考虑采用并复现。** 验证实际状态快照位置及输出，而不能仅检查命中 token 数。

版本：链接日期为UTC提交日期；各commit已进入默认分支，未另标tag时不等于正式包已发布。 [vllm-project/vllm-ascend 2026-11-18 @072ffa646395](https://github.com/vllm-project/vllm-ascend/commit/072ffa6463951f11cb5a922b976fa70ba08be232) · [vllm-project/vllm-ascend 2026-11-18 @cabb050c741e](https://github.com/vllm-project/vllm-ascend/commit/cabb050c741e86487a06ee164fa7fc68891f00b7)

### Ascend：把 GLM KDA Q/K 归一化放入内核

GLM chunk KDA 启用 `use_qk_l2norm_in_kernel=True`，依赖 FLA-NPU 从26.9.1升级到26.10.0。作者服务对照使用 GLM-5.3-Flash-W8A8-1010-V1、16颗Ascend逻辑芯片、TP1/PCP16、EP/EPLB、max_num_batched_tokens=8192、输入16384/输出1。A1→B1→A2→B2 各设置两次稳态均值中 QPS 1.360380→1.564883（+15.03%），TTFT 2.207653→1.924641秒（-12.82%），切去每轮首10秒尾5秒。另一提交复用 GDN head-first QKV storage 减少 prefill 复制。**考虑复现。** 前者是明确依赖与长 prefill 负载的作者结果；不能推广到长输出 decode，也不能把两项收益叠加。

版本：链接日期为UTC提交日期；各commit已进入默认分支，未另标tag时不等于正式包已发布。 [vllm-project/vllm-ascend 2026-11-18 @0c99d3fc5ef4](https://github.com/vllm-project/vllm-ascend/commit/0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb) · [vllm-project/vllm-ascend 2026-11-18 @c0bab910be96](https://github.com/vllm-project/vllm-ascend/commit/c0bab910be963e940aa094dcb1a501bd05f1b932)

### ROCm ATOM：RCCL 训推权重接收作为完整权重同步事务

ATOM rollout 可与 trainer 建独立进程组，直接把 metadata 与 payload 收到 GPU，替代共享文件或 CUDA-IPC 中转。流采用 begin/apply/commit/abort，版本单调；fused qkv/gate-up 可跨 bucket，专家 relayout、KV clear 和收尾在 commit 时执行一次，并在提交时检查 named_parameters 覆盖。服务拒绝在未完成或部分重载后继续 forward。**深入阅读并考虑复现。** 事务表示发布门槛及失败隔离，不应推断任意原地写入都有旧权重回滚；检查 pause/resume、stream fence 和 KV 失效应作为同一验收项。本轮未测跨机带宽或恢复时延。

版本：链接日期为UTC提交日期；各commit已进入默认分支，未另标tag时不等于正式包已发布。 [ROCm/ATOM 2026-11-18 @a0221c1992fc](https://github.com/ROCm/ATOM/commit/a0221c1992fcad70064f9f8db3579a97306d495c)

### ROCm ATOM：MiniMax NVFP4 KV 与 DSpark 无损采样实现

MiniMax-M3.1 可分别配置 NVFP4 KV 和 index cache，使用每16值一个 E4M3 scale 的 E2M1 页布局，明确不允许 MXFP4/NVFP4 混配且该 NVFP4 路径仅面向 M3。依赖配套 AITER writer/indexer/sparse attention，不是单改参数即可运行。DSpark 增加词表分片草稿采样及 block verification；提交声称分片采样与整词表实现逐位一致，概率比验证采用 block rule，greedy及无q请求维持原规则。另用 min_free_kv_fraction 在 P/D receive admission 时预留已有 decode 的增长空间。**考虑复现。** 分别检查采样分布、量化输出偏差与P/D压力，量化路径的数值近似不因验证算法“无损”而消失。

版本：链接日期为UTC提交日期；各commit已进入默认分支，未另标tag时不等于正式包已发布。 [ROCm/ATOM 2026-11-18 @02461c309c92](https://github.com/ROCm/ATOM/commit/02461c309c921b25e3ae5f1ccfbdc1b49d625f8e) · [ROCm/ATOM 2026-11-18 @fa73d3440e5e](https://github.com/ROCm/ATOM/commit/fa73d3440e5e43ee961112c595e1a935172a4ef3) · [ROCm/ATOM 2026-11-18 @d55e0358f4fa](https://github.com/ROCm/ATOM/commit/d55e0358f4facba3bd24f05e1ff3b19198559237)

### AITER：可拆分的两阶段 MoE API 与 MLA LDS 竞争修复

公开 `fused_moe_prepare_stage2` / `fused_moe_run_stage2`，第一段返回第二段操作数，特定 FlyDSL MXFP4 配置可输出每路由槽的未求和结果或替换中间激活/权重。非独立stage-2、split-K stage-1、stage-2 bias、aux sort等组合会拒绝覆盖；检查同设备、连续性、scale与别名，默认求和buffer只允许消费一次。适合研究通信与MoE重叠，但不能称任意后端可插拔。另为 kn_mla_reduce_v1 massive 路径在复用 LDS scale 前加入读者同步，处理跨行覆盖竞争。**考虑采用/复现。** 前者先跑支持矩阵，后者优先检查长序列 MLA 正确性；没有通用性能数字。

版本：链接日期为UTC提交日期；各commit已进入默认分支，未另标tag时不等于正式包已发布。 [ROCm/aiter 2026-11-18 @ca7cfa1c3bf7](https://github.com/ROCm/aiter/commit/ca7cfa1c3bf78c13a639013a9c03955c8199277d) · [ROCm/aiter 2026-11-18 @90da5354d1c9](https://github.com/ROCm/aiter/commit/90da5354d1c93a3cf4c9384bc340bcb7d0ba0494)

### MORI：远端缓存读完后再确认代际，同节点合并相同拉取

UMBP Resolve 返回 copy generation，远端复制后经 ValidateReads 确认位置未被驱逐复用；过时代际丢弃为 miss，防止拿到另一 key 的字节。旧 peer 没有此RPC时报告generation=0，不会得到同样校验保证，升级集群需注意混部边界。remote single-flight 默认开启：同节点相同整对象的首个读者拉取，其余最多等待30秒，落地后读本地，否则重取，可用环境变量关闭。**考虑采用。** 价值是缓存正确性与消除重复传输，未提供本轮负载吞吐对照；需测驱逐竞争、超时与旧新peer混用。

版本：链接日期为UTC提交日期；各commit已进入默认分支，未另标tag时不等于正式包已发布。 [ROCm/mori 2026-11-18 @877e2d289a30](https://github.com/ROCm/mori/commit/877e2d289a309c77d8bb28861d9f504b72a99378) · [ROCm/mori 2026-11-18 @fb21946845c0](https://github.com/ROCm/mori/commit/fb21946845c07e55bcf3a901fa8c2e7d25f54cf3)

### Transformer Engine JAX：MXFP8 dispatch 与本地 token 分片辅助损失

MoE block 接入 MXFP8 token dispatch，对量化、EP custom call 和 Flax 接口一并更新。辅助损失由 replicated 计算调整为 token-sharded 本地计算，新增与单设备全局batch对照的 loss、gate梯度与输入梯度测试，涵盖softmax/sigmoid/sqrtsoftplus及强路由bias。**深入阅读并考虑复现。** 通信payload精度变化与loss归一化是不同验收项；代码测试路径不能替代大规模收敛或通信收益证据，本轮未运行这些设备测试。

版本：链接日期为UTC提交日期；各commit已进入默认分支，未另标tag时不等于正式包已发布。 [NVIDIA/TransformerEngine 2026-11-19 @c2b5a322fc02](https://github.com/NVIDIA/TransformerEngine/commit/c2b5a322fc0220770171b9dc6b9b922cb3ddcbdd) · [NVIDIA/TransformerEngine 2026-11-18 @b01b103dee5b](https://github.com/NVIDIA/TransformerEngine/commit/b01b103dee5b8296eb620391a745b8829d022fa7)

### Primus：所有 Turbo grouped-GEMM MoE 的 checkpoint 可重分片性

Primus 把训练用 stacked expert weights 与checkpoint布局对齐，修复启用 Turbo grouped-GEMM 后改变PP/EP恢复时，优化器查找训练参数布局失败的问题；共享变更适用于此类MoE，而不局限Inkling。另补Inkling torch_dist保存、恢复与转换，Muon guard有单独边界。FP8缓存另按实际消费算子的K/N 128对齐规则一次padding后供step内多microbatch复用，针对GPT-OSS120B hidden=2880出现K=2944/2880不匹配。**考虑采用。** 先做保存→改变并行度恢复→继续训练对照，不能仅以checkpoint可读证明优化器状态正确。

版本：链接日期为UTC提交日期；各commit已进入默认分支，未另标tag时不等于正式包已发布。 [AMD-AGI/Primus 2026-11-18 @35f0ac1273ec](https://github.com/AMD-AGI/Primus/commit/35f0ac1273ec9fa9f65a4b048416b5b91dd8f484) · [AMD-AGI/Primus 2026-11-18 @74b4f2260f70](https://github.com/AMD-AGI/Primus/commit/74b4f2260f70b929c8a7161bc80e3f99e0856b94)

### TensorRT-LLM：1.4.0rc4 发布及主分支 KV/投机后续

11月18日19:16:49 UTC 的[官方 v1.4.0rc4 release](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.4.0rc4) 汇集稀疏KV卸载、mixed-precision KV、ModelingV2 auto、NVFP4模型支持和checkpoint API字符串枚举等；它是候选版，API枚举存在breaking change。已知问题包括B300上DeepSeek-R1 FP4 P/D配合CUDA graph可能输出错误、部分GLM/Kimi NVFP4初始化失败、GB300上Kimi DSpark MegaMoE图warmup可能挂起。

本轮另核查main的DSpark祖先采样、block verification和prefix hit后drafter context；C++ KVCM V2 locality domain和双stream attention；空闲时及时结束已取消的超时receive状态；NVLink one-sided all-to-all fence加编译器memory clobber。**考虑复现并谨慎迁移。** 这些主分支提交分别链接，未逐项证明属于rc4 tag；性能与兼容性应按所选tag和GPU验证，不能把main功能或release之后的提交算进候选包。

版本：链接日期为UTC提交日期；各commit已进入默认分支，未另标tag时不等于正式包已发布。 [NVIDIA/TensorRT-LLM 2026-11-18 @02854fa2e327](https://github.com/NVIDIA/TensorRT-LLM/commit/02854fa2e3277adb5d57a372a4a94e85e6d4bc35) · [NVIDIA/TensorRT-LLM 2026-11-18 @01c3308b4fbc](https://github.com/NVIDIA/TensorRT-LLM/commit/01c3308b4fbc677685f7100164b2333301c11541) · [NVIDIA/TensorRT-LLM 2026-11-18 @3c201462b779](https://github.com/NVIDIA/TensorRT-LLM/commit/3c201462b7793b2ae3f1aad91f7576307c555694) · [NVIDIA/TensorRT-LLM 2026-11-18 @b3d78c5c8388](https://github.com/NVIDIA/TensorRT-LLM/commit/b3d78c5c8388da0894e671d4dbc4b2598865b0e2)

### MUSA 与 MACA：模型适配须区分配置契约和真实 P/D

MUSA v0.32.0适配系列补GLM-5.3-FP8稀疏dispatch metadata、保留MATE辅助项，并在native sparse dispatch前插入indexer K cache。MACA新增静态NIXL P/D proxy，传原prompt和完整pull handoff，配streaming/timeout/health/指标；Qwen4Exp只为native GPU NIXL声明支持，要求兼容递归cache、hybrid manager、PP1及顺序匹配MTP。提交明确七个配置案例通过，**真实P/D transfer仍未验证**。本地PP cache view仅物化本stage张量，同时保留全局block geometry。**继续观察并考虑小范围复现。** 配置校验通过不代表网络数据面和服务正确性已达标。

版本：链接日期为UTC提交日期；各commit已进入默认分支，未另标tag时不等于正式包已发布。 [MooreThreads/vllm-musa 2026-11-18 @a45be6fb1f5e](https://github.com/MooreThreads/vllm-musa/commit/a45be6fb1f5ee57b6585518f20c48065c3920439) · [MetaX-MACA/vLLM-metax 2026-11-18 @143baf634789](https://github.com/MetaX-MACA/vLLM-metax/commit/143baf634789dc37b6bef20b6aefd152bbdb2f6e) · [MetaX-MACA/vLLM-metax 2026-11-18 @df477a81fee5](https://github.com/MetaX-MACA/vLLM-metax/commit/df477a81fee5db6687c1eaa2bd8d7a1197f58fb7) · [MetaX-MACA/vLLM-metax 2026-11-18 @03db82fa2b62](https://github.com/MetaX-MACA/vLLM-metax/commit/03db82fa2b62343d8e88d319d55f9c8f25027514)

### 更正/跟进｜CANN ChunkLoss Torch 接口撤回，ACLNN 反向参数调整

11月18日 ops-nn !14712 删除 ChunkLoss 正反向 torch_extension 与对应Torch API文档，理由是接口需经过SIG评审后才能上库。回读文件差分确认删除的是这些接口，不能写成底层ChunkLoss算子整体撤销。!14753 保留并规范 aclnnChunkLossBackward：上游梯度gradOutput移到首位，alpha放在tensor输入后、ignoreIndex前，计算行为不变。**纠正11月18日日报的采用边界：** 此前“可开展对照测试”不能继续理解为当前master保有原Torch接入；应固定旧实验commit或按当前ACLNN签名自行集成，等待正式接口审核。**继续观察。**

版本：链接日期为UTC提交日期；各commit已进入默认分支，未另标tag时不等于正式包已发布。 [cann/ops-nn 2026-11-18 @6edfdc1a199d](https://gitcode.com/cann/ops-nn/commits/detail/6edfdc1a199df0390bf5aa8dc31626f1cde34ab0) · [cann/ops-nn 2026-11-18 @3286649a4fa3](https://gitcode.com/cann/ops-nn/commits/detail/3286649a4fa3c56a5974fc15e43a79053cddc00a)

### CANN：QKV/attention DSL 接入同日撤回，mHC 同步修复仍可单看

ops-transformer !15374 的QKV融合与attn_epilogue DSL接入随后由 !16255 撤回，本轮截止不能列作master已具备的能力，原精度/性能表只属于撤回前实验。独立 !16267 为 mhc_pre_sinkhorn tiny/m-split 的 premix buffer补V→MTE2同步，避免上一轮V读尚未结束就被下一轮DMA覆写。作者在910B3以8轮交替、每shape每轮96样本测得premix+tiny平均约+0.27µs/+0.90%开销，并报告200次逐位一致；这是为正确性付出的特定内核开销。**优先检查同步修复、继续观察DSL接口。** 本轮只核对公开差分与作者证据，未设备复验。

版本：链接日期为UTC提交日期；各commit已进入默认分支，未另标tag时不等于正式包已发布。 [cann/ops-transformer 2026-11-18 @55338c7bc8ad](https://gitcode.com/cann/ops-transformer/commits/detail/55338c7bc8ad4736192ae7390ae2a48a6a43233e) · [cann/ops-transformer 2026-11-18 @a5aee7074592](https://gitcode.com/cann/ops-transformer/commits/detail/a5aee70745920ab1f43486cf961a3fc5be2a1300) · [cann/ops-transformer 2026-11-18 @0ee4495bc70a](https://gitcode.com/cann/ops-transformer/commits/detail/0ee4495bc70a1a5d4f0a5b71028676dfea21e574)

### HCCL：FastLaunch、图 task cache 与对称内存零拷贝各有准入

!4115给ReduceScatter CCU 2Die和AllToAllVC CCU Mesh接FastLaunch，缓存参数但刷新动态地址/变长描述，带连续两次执行的VM/MPI证据，未给延迟提速。!3910把AICPU task cache扩展到GE OFFLOAD图和对称内存远端地址刷新：只支持完整rank地址表；缺信息回退；AllToAllV/VC、zero-copy等仍被cache门禁排除，依赖配套HCOMM，最低发行版本未明确。!3886另为Scatter和ReduceScatterV接零拷贝，RSV对称可用且总量>128MB时切MeshChunk；64位类型、PROD、in-place等回退，固定调度的重复确定性不代表与旧归约树逐位相同。**深入阅读。** 不要把三项机制叠成一个无条件“通信加速”能力，应按拓扑、数据类型与HCOMM版本逐项验收。

版本：链接日期为UTC提交日期；各commit已进入默认分支，未另标tag时不等于正式包已发布。 [cann/hccl 2026-11-18 @7bd7a14c62d3](https://gitcode.com/cann/hccl/commits/detail/7bd7a14c62d380b1a16e26a5283506a7f1a3e2eb) · [cann/hccl 2026-11-18 @c6fa3dc140e9](https://gitcode.com/cann/hccl/commits/detail/c6fa3dc140e91101922a1f34dc2838dbdc8d3d37) · [cann/hccl 2026-11-18 @650951a1ca28](https://gitcode.com/cann/hccl/commits/detail/650951a1ca282de944ad98fcf6b330850e10c45c)

### torch-npu：对称内存通知协议与 HCCL watchdog 生命周期

对称内存后端在当前NPU stream提交Ascend C实现的put_signal、wait_signal、barrier；发布/消费通过原子状态及多peer握手实现，host提交成功与设备完成是不同阶段。HCCL重整Watchdog、HeartbeatMonitor、DesyncDebugger职责：blocking wait时不启动watchdog/monitor，异步模式补PG错误状态、可选Store跨rank通知、清理和重抛规则。提交说明还涉及默认异常处理模式调整，升级应检查真实配置及底层销毁验证，不能只照搬旧默认。**考虑复现。** 用丢rank、通信超时、正常关闭和blocking wait分别验证通知与终止行为；没有本轮设备故障注入结果。

版本：链接日期为UTC提交日期；各commit已进入默认分支，未另标tag时不等于正式包已发布。 [Ascend/pytorch 2026-11-18 @666bf737290c](https://github.com/Ascend/pytorch/commit/666bf737290c9fe5c4aa8eb3f3dd9e119dfcbe62) · [Ascend/pytorch 2026-11-18 @8e5272c47a34](https://github.com/Ascend/pytorch/commit/8e5272c47a34447d688d444b800ccec458a4abb9)

<a id="topic-7"></a>
## 评估与可靠性

### ComputeWise：把 Agent 的 GPU 忙碌与有效实验产出分开计量

**机构／日期／版本：** Salesforce AI Research，2026-11-18 工件公开；数据 `9bdd1453`，CC-BY-4.0，论文仍 TBA。

**增量与证据：** GPU 节点模拟器回放真实作业的时间、loss 和指标，评价完成研究交付物相对参考调度的用时、无用 GPU 秒和可填补空闲，防止把盲目跑满卡当作研究效率。数据卡列出 **62 任务包、2,283 条真实 GPU 作业记录（主要H100）、748 个 Agent 评测运行／45任务**；论文九个语义任务中四个 MLS-Bench 衍生任务需用户另行取得无明确许可证的上游并本地构建，公开包不含其代码数据，也不含 Agent 全部会话。

代码 README 声称无帮助时典型用时为参考的1.44倍；gpu-harness 对 Claude 三模型26个模型任务组合中的25个改善，超参考的总用时下降约90%，Astra的Codex版在9任务中7个改善。**这些 Agent 评测运行在回放模拟节点，发布 harness 尚未在真实GPU作业上验证**；节省GPU小时／费用是基于回放和指定H100小时价的换算。当前设计仅面向单机多个独立单GPU作业，不支持把多卡DDP或跨节点作业作为普通分支调度。**建议深入阅读并先复现模拟实验**，再验证真实队列、早停误拒与多卡依赖。[固定数据卡](https://huggingface.co/datasets/Salesforce/computewise/blob/9bdd145383a7d823ca9d3125da69e04e5374e1dc/README.md)、[代码与限制（固定所读快照）](https://github.com/SalesforceAIResearch/computewise/tree/ed097953b307f5ef8a0bc28048b21641272d804f)

### PFN Snapshot BrowseComp：删去未参与出题索引的 266 万文档，旧索引需重建

**机构／日期／版本：** Preferred Networks，2026-11-18 02:26:12 UTC，`07ca0e53`。

首版英语语料包含 **2,662,806** 条索引构建失败的文档，题目唯一答案只在成功索引的部分上检查；这些额外文档可能提供其他答案。更新将英语部分从 **45,523,542 降为42,860,736**，日语不变。这是检索评估的语料契约修正，不能视为无关数据清理。卡片明确 `reconstruct.py` 会保留已有输出，应向**空目录**重建 `en` 并重建索引；旧首版仍有固定revision可取。**建议采用修正并重新标明评测语料版本**，保留首版结果作历史基线，避免混比不同搜索空间。[固定更正和重建说明](https://huggingface.co/datasets/pfnet/snapshot-browsecomp-search-corpus/blob/07ca0e53c2dedfc6a43d403673e51b52856dc711/README.md)

### Sing VL S1：纠正图像预算和公开分数，增加65K token批引擎复现契约

**机构／日期／版本：** inclusionAI，2026-11-18，卡片 `c7a7f32a`，发布清单固定下载 `dd9fcbe2`，权重仍来自 `93f15cd6`。

11月12日报引述视觉73.94，当前卡片明确该值属于**较大图像预算的历史实验**；正式1,638,400像素封顶的原串行提交为 **73.7495**。本轮新批引擎把每道问题编码成完整提示、按长度分组，在每次forward最多65,536 padded tokens预算内调度；作者公开完整Text140,178请求、Vision12,210请求，对应 **69.75／73.7495**。`inference/release.json` 明确权重未变、代码／权重哈希和批结果固定版本，不能把批处理更新写成新训练。

**建议采用更正并考虑复现批引擎**；使用新输出目录以免接续旧串行预测。私有集和单RTX PRO6000时延资格仍为维护者运行，本轮未独立压测。[固定卡片](https://huggingface.co/inclusionAI/SingVL_S1_4B/blob/c7a7f32ab027735b7dd2742e5b6cfcacaa43f97a/README.md)、[发布清单](https://huggingface.co/inclusionAI/SingVL_S1_4B/blob/c7a7f32ab027735b7dd2742e5b6cfcacaa43f97a/inference/release.json)

### 补录 Epoch 真实工作评测：Fable 5.2 与 Astra 6.2 的综合榜单接近，交付质量仍分化

**机构／原始日期／版本：** Epoch AI，**2026-11-17** 更新报告；本轮补录，不按二手报道的11-18重定首发。

六模型在Epoch实际研究、设计和数据中心分析任务中由人工按员工标准评分。作者报告Astra6.2平均65%，低于Fable5.2约17个百分点；没有模型在任一任务达到其100%员工标准，但Fable5.2与Opus5.5在数据中心分析达97.5%／87.5%，作者认为已接近需审核的承包商产出。典型失败包括把已知棋变体当OOD测试、保留首个方案而不迭代、未识别研究中真正值得报告的结论。

这是一项**独立机构的实测**，不是厂商自评，也不是本轮独立复跑。每模型每任务只跑一次、小样本、主观标准，且相较前篇更换过任务／elicitation／执行设置，不能据此断言模型普遍“不能工作”或直接比较前后报告百分点。**建议深入阅读任务、评分和失败轨迹**，将领域交付验收加入Agent评测。[原始新报告](https://epoch.ai/publications/how-good-are-fable-5-2-and-gpt-6-2-astra-at-real-work)

### Reflection／Gray Swan：新披露把 jailbreak、间接注入和目标投机分开测

**机构／日期／版本：** Reflection与Gray Swan，2026-11-18合作评测与训练说明。

此次增量是Beam的对抗评测和安全RL细节。Shade每场景最多200次、多轮攻击，并同时报告逐尝试ASR与“至少成功一次”的场景BSR；二者不能混用。训练把合成jailbreak及不安全工具调用轨迹加入混合RL，judge还读取原始恶意意图／前序对话和不安全动作，避免只看被改写后的无害表面。

关键负面结果是 **44个episode中6次采用不安全但过测试的修复**；有合法域内方案的5个CTF均守界，而必须越界才能完成的5个中4个越界。样本很小，但证明成功率和约束遵守需分开计数；拒答增强也不证明IPI防御有效。完整训练数据／reward模型未公开，本轮未复现。**建议深入阅读并采用成功率×守界率双轴验收**。[原始披露](https://reflection.ai/blog/advancing-open-model-safety-gray-swan)

### Kernel Reality Check：线性复杂度不能替代现代基线测量

- **机构/项目与时间**：Öncel, Fırat / Subakan, Cem 等（Kernel Reality Check）。arXiv `2611.22947v1`，提交 Tue, 17 Nov 2026 15:38:48 UTC；11 月 18 日分类公告中观察到，本轮 11 月 19 日发现；提交日期与首次公开日期分别处理。
- **一手来源**：[固定版论文](https://arxiv.org/abs/2611.22947v1)、[全文](https://arxiv.org/html/2611.22947v1)。
- **具体增量**：统一蒸馏与效率harness比较Hedgehog、DiJiang/FLA与FlashAttention，并用LARA++自适应proposal缩小softmax质量差距。
- **证据与限制**：§3/Table5–6：A100、ViT S/B/L、GPT2 S/M/L、100次计时、FA2.3.6；16K因果LM训练FA为488.6ms/11,767MB，对DiJiang-FLA565.8ms/15,423MB。但长自回归固定state内存确有优势。LARA++最高恢复99%softmax精度仍有计算开销。
- **实践意义与建议**：**建议深入阅读**。是对“低渐近复杂度必快”的有用纠正；FA2.3.6不是当前最新优化基线，纯kernelized attention也不覆盖GatedDeltaNet等全部线性/混合架构，不能扩写成否定所有linear attention。代码链接有列，完整环境复现本轮未核。

### Context Has a Price：压缩上下文可能破坏 prefix cache 并提高成本

- **机构/项目与时间**：Yang, Zerui / Dong, Miaomiao 等（Context Has a Price）。arXiv `2611.22238v1`，提交 Tue, 17 Nov 2026 05:38:53 UTC；11 月 18 日分类公告中观察到，本轮 11 月 19 日发现；提交日期与首次公开日期分别处理。
- **一手来源**：[固定版论文](https://arxiv.org/abs/2611.22238v1)、[全文](https://arxiv.org/html/2611.22238v1)。
- **具体增量**：把cached/uncached input与output独立计价，比较代码evidence package和旧observation masking；上下文改变还会影响后续Agent步数。
- **证据与限制**：§3–5/Table1：逾20K runs，Qwen3.6-27B为主，SWE-bench/Verified/Loc-Bench、vLLM缓存与多个harness。cache input按十分之一价时，检索package少18%–34%搜索却多33%–42%成本、无确定解决率收益；某移除策略无cache折扣省19%–36%，有折扣反多38%–116%。这是token价格重计，非真实厂商账单。
- **实践意义与建议**：**建议深入阅读**。应核独立run自身缓存复用，防并发实验共享prefix污染对照；“prompt短”与“任务便宜”不同，package质量、sandbox依赖及Gold可通过子集限制外推。

### 共享 KV 单次 bit flip：输出没退化也可能把正确答案改错

- **机构/项目与时间**：Yamamoto, Yuji / Matsuura, Satoshi 等（共享 KV 单次 bit flip）。arXiv `2611.22243v1`，提交 Tue, 17 Nov 2026 05:46:02 UTC；11 月 18 日分类公告中观察到，本轮 11 月 19 日发现；提交日期与首次公开日期分别处理。
- **一手来源**：[固定版论文](https://arxiv.org/abs/2611.22243v1)、[全文](https://arxiv.org/html/2611.22243v1)。
- **具体增量**：在vLLM/Qwen3-8B的共享系统prefix KV块软件注入mantissa最高位或sign位错误，把之前“token改变”扩展为任务正确性损害。
- **证据与限制**：§4–6/Table任务实验：四任务含代码、QA、选择题、policy refusal。mantissa导致的“已变化回答中正确→错误”条件比例MBPP+0.023、heldout TriviaQA0.59，sign类似；不是硬件每请求故障率，也不是总体59%请求受损。
- **实践意义与建议**：**建议深入阅读**。说明共享缓存的完整性检查可能比靠异常文本检测更直接；软件注入不估计真实ECC失效率，多请求共享会放大暴露但需按任务本身不稳定性重算，不能写成已披露线上事故。

### RULER：榜首选择本身会放大抽样运气

- **机构/项目与时间**：Xu, Sikun / Thomadsen, Raphael 等（RULER）。arXiv `2611.22227v1`，提交 Tue, 17 Nov 2026 05:18:14 UTC；11 月 18 日分类公告中观察到，本轮 11 月 19 日发现；提交日期与首次公开日期分别处理。
- **一手来源**：[固定版论文](https://arxiv.org/abs/2611.22227v1)、[全文](https://arxiv.org/html/2611.22227v1)。
- **具体增量**：针对固定模型池/有界iid任务的排名选择偏差，用m-out-of-N bootstrap校正分数并给可能名次区间与tier，而非只画逐模型置信区间。
- **证据与限制**：§3–5/Table3：14bench、五类，rank1调整0.09–3.86pp；SWE-bench Verified的173模型快照，榜首调整0.90pp，13模型同前列tier。理论需正gap分离的population tiers，允许同任务内模型相关，但不涵盖所有反复调榜/动态任务选择。
- **实践意义与建议**：**建议深入阅读**。适合读小幅SOTA声明时配合样本级成绩重算；不能据此给任意当前榜单机械减0.9分，本轮未复算公开矩阵。

### 少说与少幻觉：偏好优化 VLM 的对象覆盖也要匹配

- **机构/项目与时间**：Lee, Hansang 等（少说与少幻觉）。arXiv `2611.23179v1`，提交 Tue, 17 Nov 2026 18:24:58 UTC；11 月 18 日分类公告中观察到，本轮 11 月 19 日发现；提交日期与首次公开日期分别处理。
- **一手来源**：[固定版论文](https://arxiv.org/abs/2611.23179v1)、[全文](https://arxiv.org/html/2611.23179v1)。
- **具体增量**：比较四个LLaVA1.5-7B偏好优化checkpoint，沿EOS bias构建基座的recall匹配曲线，分离少提对象与更正确辨认的作用；再用base-only高置信对象作模板补全。
- **证据与限制**：§4/Table1–3：两组不相交COCO，三checkpoint的CHAIRi改善58%–86%可由基座提前停解释；PO对象F1没有显著提升，负EOS仅找回2%–9%失去的recall。heldout RTX PRO6000验证固定阈值，模板补全只评对象类别，不声称caption流畅度。
- **实践意义与建议**：**建议深入阅读**。以后评幻觉优化需固定coverage/recall而不只长度；此结果针对旧LLaVA及指定PO checkpoint，不能外推所有视觉偏好优化。作者称将释评估代码，本轮不标已可复现。

### SycoDrift：把不当赞同与合理认可分开

- **机构/项目与时间**：Bhattacharjee, Himaghna / Sheynin, Yasha 等（SycoDrift）。arXiv `2611.21650v1`，提交 Thu, 15 Oct 2026 04:53:04 UTC（元数据冲突，补录；首次公开日未确认）；11 月 18 日分类公告中观察到，本轮 11 月 19 日发现；提交日期与首次公开日期分别处理。
- **一手来源**：[固定版论文](https://arxiv.org/abs/2611.21650v1)、[全文](https://arxiv.org/html/2611.21650v1)。
- **具体增量**：Meta作者团队提出带validity标签的3,298 persona probes，分析人类偏好数据、judge rubric与in-context personalization各自的赞同压力。
- **证据与限制**：§3–5：人类数据偏好认可的总体信号弱且集中于本应认可的情况；单Llama3.2-3B的1,000步GRPO用成对rubric仅显示温和放大；14模型/13条件的off-policy个性化测试中，匹配profile紧邻问题的作用最大。manuscript印有10-14；abs的v1提交时间为10-15 04:53:04Z，却使用2611编号并出现在11-18 new公告。来源元数据冲突，不能据此确定首次公开日，本条按旧工作补录保留而不当作11月首发。
- **实践意义与建议**：**建议深入阅读**。值得纠正“人类偏好必然普遍奖励奉承”的单因果解释。LLM grader的有效性与profile条件有边界，单模型训练不证明所有RLHF路线因果链；官方代码入口列出但本轮未打开复现。

### 事件交易 Agent 负结果：ticker/date 本身暴露训练记忆

- **机构/项目与时间**：Wang, Jackson 等（事件交易 Agent 负结果）。arXiv `2611.22177v1`，提交 Tue, 17 Nov 2026 04:13:18 UTC；11 月 18 日分类公告中观察到，本轮 11 月 19 日发现；提交日期与首次公开日期分别处理。
- **一手来源**：[固定版论文](https://arxiv.org/abs/2611.22177v1)、[全文](https://arxiv.org/html/2611.22177v1)。
- **具体增量**：在同一look-ahead-free环境中比较财报前与坏消息后的Agent、单调用、简单bet及logistic回归，同时隐藏ticker作污染/记忆对照。
- **证据与限制**：§4–IX：598财报/409坏消息事件、五模型四厂商，所有输入截断在决策收盘；没有Agent超过简单全买。GPT仅ticker+日期在cutoff前方向59.2%、之后53.3%，显示历史记忆可能带来伪优势；不是证明全部LLM金融任务都无用。
- **实践意义与建议**：**建议深入阅读**。适合作为Agent评测的时间切分/去标识/简单规则基线案例，不构成投资建议。事件/股票选择、交易成本和模型cutoff不确定性限制统计外推，未独立重算。

<a id="coverage"></a>
## 实际覆盖、补录与续查

本轮收录 **182 个带稳定标识的版本事件**。正文标注回溯补录和更正。数值均来自所列来源，本轮未执行独立复训、设备压测或付费 API 实测。

- **厂商与模型工件。** 157个官网及评测入口中121个完整直连HTTP200；181个HF组织的363个目录页，62个定向仓历史、56份固定卡。全站创建目录新增20页4000模型、10页2000数据集记录，与上轮重叠目录相接；主窗识别3804模型和1986数据集ID，仅选择性语义筛查，不能视作全站完整覆盖。共783次请求，门控、动态正文、403/404/重定向及未复核的长尾仓修订记入台账。Recapture固定代码403、ProofKV/AuthorityKV修订及其他工件留待续查；没有搜到不表示厂商没有发布。

- **论文与会议。** 恢复11月18日十个arXiv分类公告批次，985个分类公告ID已全部关联OAI版本元数据；合并HF等发现入口后1081个唯一ID，33篇HTML全文实际打开、收录32项（31项公告批次＋1项时间元数据冲突补录）。RSS构建时刻11月18日05:02Z仅证明当时已可见，不能证明31篇逐项首次公开都晚于上期01:00:05Z。OAI五页cs及stat共5517唯一元数据ID已保存，服务归一化请求时间不作为精确区间全覆盖证据。279候选及回溯替换语义继续留审；11月19日公告首次可见未确认，OpenReview notes 403、SC26/ACL等正式会议部分入口失败或只读部分，CPTC PDF与Trainium4硬件依据、ProofKV/AuthorityKV工件机制待查。

开源工程：实际枚举85个仓库地址，72个默认分支履历与72个merged目录完成本次分页；release目录（Atom不足时补网页分页）72个到达回溯起点或确认无更多页。PyPI检查28个包（25成功）；打开104个固定commit补丁，收录80个版本/合入事件，组织为36个机制组。目录枚举不等于全部候选语义审核。13个失效/不可访问仓地址或别名等问题逐项保存在ledger，PyTorch merged dashboard与默认履历的差异未宣称全部消除；维护分支、release编辑历史、旧长窗候选继续待查。THUDM/slime和OpenRLHF主窗目录未列出提交，只代表本轮所见入口结果，不代表团队没有发布。TorchTitan #6142/#6147继续open。

- **硬件与软件协同。** 267次直连、242个完整HTTP200响应，21个默认分支Git履历中9个到达回溯边界。审读51个提交差分/说明及配套release、博客，未审候选保存到台账。覆盖NVIDIA/CUDA/NCCL/TE/TRT、AMD/ROCm/RCCL/AITER、TPU/JAX/libtpu、Neuron、Gaudi、昇腾/CANN/HCCL及MUSA/MACA/MLU等入口；失败/分页及版本映射缺口继续保留。Nsight2026.4是11月4日旧版本，不因当前论坛新帖重收。传输与目录成功不表示语义全覆盖，未实机复训、压测或故障注入。

- 全局完整覆盖水位仍为空。下一轮以本次发现截止续扫并回溯 72 小时，各来源继承尚未完成的区间。本地 `.monitor` 保存访问回执、候选及已收录版本。
