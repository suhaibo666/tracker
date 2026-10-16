# LLM 技术动态日报 · 2026-10-16

> 时区：Asia/Shanghai；实际生成时间：2026-10-16T09:17:07+08:00。
>
> **增量事件窗口：2026-10-15 09:55:47 → 2026-10-16 09:00:40；72小时回溯起点：2026-10-12 09:55:47。** 同时续查此前保存的10-05以来待办。窗口来自持久化成功归档记录，不能用调度启动时刻或本地mtime替代。页面/代码核验持续至生成时间，动态页面是读取快照。
>
> 已先核本地状态与GitHub `tracker/main @ 9147aa9c1c52318a66ca77d98994dc66b7b3775d`，12个历史文件（含索引）Git blob一致，无待补传文件。**事件游标不等于全来源完整覆盖水位**；arXiv主窗API限流、正式会议与部分厂商/硬件渠道仍有缺口，详见文末。
>
> 正文中的精确GitHub/HF/arXiv时刻默认UTC；博客仅给日期时保留原日期。所有性能、训练与评测数字均来自原作者/厂商，**本轮未做独立GPU/NPU复现**。已合入main、dev、release支线、正式发版、未来计划与尚未合入的工作分别标记。

## 本次重要变化

1. **Meta开放新816B多模态MoE，工件与许可比榜单更需先核。** Muse Spark Open给出BF16/NVFP4权重、1M上下文与多框架入口；完整训练配方未披露，公开卡可读不等于门控raw与权重已审计。
2. **Nemotron新报告提供可读的持续训练与压缩细节。** 软件中训使1M长上下文能力大幅回落，后续长上下文恢复再拉回；Bolt组合结构压缩与低bit QAD，但24GB是排除视觉模块的文本预算。
3. **预训练方向独立纳入。** LatentLoop以第一轮token压缩换宽度，CAST按历史梯度尺度与谱阈值裁剪；原生多模态配对消融保留caption，避免把文本增量算成视觉收益。三者都需保留预算、seed与成本边界。
4. **后训练要用匹配墙钟和完整训练曲线判断收益。** 自蒸馏统一对照出现等时预算排名反转；CaLIF保住答案质量但约束满足率低于纯IF-RLVR。TRL树形复用带来工程机会，同时明确拒绝混合线性注意力等未支持路径。
5. **多项关键修复会改变数值或兼容性。** vLLM修多轮投机随机数复用偏差，并改变V4.1 KV字节格式；Megatron非默认RMSNorm与梯度通信dtype、torchtitan mHC归约轴、DeepSpeed CPU Lion均值得先审适用条件。
6. **服务性能需区分算子、调度和完整负载。** GLM indexer的4倍算子收益只带来受测服务约8%吞吐改善；KV空间等待可阻塞decode调度。DSA top-k还有未修CUDA路径，P2P consumer超时释放问题仍未闭环。
7. **短跑通过不等于稳定。** verl TPU GRPO从十步样例延长后暴露HBM碎片，改配置才完成100步；FOLD报告的BF16输出差异与缓存显存成本，也应和加速一起读。

## 阅读目录

- [模型与技术报告](#models)：Muse Spark Open、Nemotron Super VL/Lightning/Bolt、Granite Switch及Muse Wonder补录。
- [预训练与架构](#pretraining)：LatentLoop、CAST、视觉配对预训消融。
- [后训练与协作](#posttraining)：自蒸馏统一对照、CaLIF、VRSD、CoSIR。
- [推理与部署研究](#inference)：Jeva、FOLD、ReaDy。
- [评估与可靠性](#evaluation)：Base LM评测协议与排序稳定性。
- [工业开源：核心六项目](#core)：数值、投机、缓存、并行、异步RL与TPU稳定性。
- [扩展工程与硬件软件协同](#extended)：TRL/Transformers、TensorRT-LLM、Ascend、DeepSpeed、PyTorch、Triton、FlashAttention及Xeon部署。
- [覆盖、失败、补录与更正](#coverage)：实际来源检查、未完成区间与续查清单。

<a name="models"></a>

## 模型与技术报告


### Meta Muse Spark Open：816B/42B 原生多模态权重发布，许可与资源预算应随模型一起看

**日期/身份。** [Meta 10-15 官方公告](https://research.meta.ai/blog/muse-spark-open)明确当天开放权重。官方组织是 **`meta-models`**，不能继续只扫 `meta-llama`。本轮 API 可见 BF16 `0fb02478b02565b2ce7fc57e6051c91172774ac9`、NVFP4 `4239e4db19376904f2e82c167e78f70abb5455fe`；二仓创建于 9-15，但这不证明当时公开或可下载。公开渲染[模型卡](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B)已读；固定 raw README/config/commits 请求分别 401/403，故只把 SHA 当 API 版本记录，不声称已审计门控配置或实际权重。

**架构/训练披露。** MoE Transformer，816B 总/42B 激活，62 层、hidden8192，256 专家 top8；GQA 128Q/16KV、head64，带 attention output gate、output norm 和 QK norm。1.8B 感知编码器为 50 层、patch14，使用 absolute+2D RoPE；输入 text/image/video/audio，输出 text，推理上下文 1,048,576。开放版从 Muse Spark 衍生并额外做安全后训。训练数据仅披露 public/web/第三方/Meta 产品服务/合成数据，覆盖 100+ 语言；未披露训练 tokens、混合比例、预训 FLOPs、训练 GPU 小时或可复现的完整后训配方。

**评测与限制。** 卡用 xhigh、temp1/top_p1/top_k disabled；MCP-Atlas89.2、JobBench61.8、Terminal-Bench2.1（Terminus2）76.4、DeepSWE1.1（Muse Code）60。不能改写成所有任务领先。[8 页评测方法](https://research.meta.ai/static/muse-spark-open-evaluation-methodology)明确：综合采用第三方榜单、内部运行与其他厂商自报的最高可比主指标；DeepSWE 使用不同模型偏好的 harness，Mistral Large4/Beam部分仅自报，其他模型提示/工具未专门调优。AA Index采集日写 **10-07**，早于10-15发布，保留该历史采集日，不当实时评测。没有匹配 BF16/NVFP4 的独立质量消融，不能称量化无损。

**许可/部署。** Muse Model License，含署名和受限商业条款，不能标 Apache/MIT。公开许可要求产品显示 Built with Muse Spark Open；特定 Model-as-a-Service 业务在连续12个月集团总收入超过2000万美元时须另签协议，原文另列内部使用/非MaaS产品例外，具体适用以[许可原文](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B/blob/main/LICENSE.md)为准。本轮许可事实来自公开模型卡门控区的完整许可文本，独立blob访问403。[部署文档](https://dev.meta.ai/docs/muse-spark-open)给 BF16 权重约1.65TB、总预算约2TB VRAM；NVFP4 权重约537GB，仍需为KV/activation/runtime留余量。42B激活不代表只加载42B权重。公告的双Mac与kernel加速是另一部署场景，不据此推断通用端到端性能。

**判断/建议。** 深入阅读：这是本期最明确的新大规模开放权重事件；复现应同时固定checkpoint、reasoning effort、harness与多模态处理，先确认资源和许可条件。稳定工作 ID：`meta:muse-spark-open:2026-10-15`。

### NVIDIA Nemotron 3.5 Super VL：值得读的是蒸馏、软件中训与长上下文恢复之间的相互影响

**日期/工件。** [官方研究页](https://research.nvidia.com/labs/nemotron/Nemotron-3.5-Super-and-Lightning/)标 Published **10-14**；[10-15发布公告](https://blogs.nvidia.com/blog/nemotron-3-5-super-vl/)和卡 Release Date 为10-15。Super VL base/BF16/NVFP4及drafter可见；BF16固定卡 `97e009758ae21747ee3752d2e43b005fb1e23a61`、NVFP4 `8a115b9d966adce8c059a7bcf47f49a1e57957d8`。HF三仓10-12创建不当正式发布。Lightning30B-A3B已是旧权重，不能随新报告再说首次推出。

**模型/预训。** 名义120B/12B，Mamba2–attention–MoE hybrid（40/8/40层）、512 routed experts top22，接C-RADIO和VL connector；报告计入vision/MTP等实际分母124.453B。79页[官方技术报告](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Super-and-Lightning-Technical-Report.pdf)§2披露：先冻结骨干做100B token MTP warmup，再1T token从下一个规模的Nemotron3蒸馏。Lightning用Super3 top128 logits、sink余量、forwardKL/CE=.9/.1；Super用Ultra3、skew reverseKL α=.1。Super另做100B Indic CPT（10B选中Indic+90B旧混合），35B@1M长上下文、**65.4B@256K软件中训**、最后9B@1M恢复。模型卡写软件中训67B，和报告65.4B不完全一致，本条采用报告并保留差异，不能自行补解释。

**中训因果证据。** 软件blend为issue–PR41%、整仓20%、agent traces10%、commit3%、build/CI1%、预训replay20%、长context5%；trace只训练模型turn，mask工具返回。1M RULER从79.9因256K中训降到33.5，再9B恢复到81.0。中训消融是两个checkpoint接受**相同独立33B实验SFT**，不是最终后训pipeline的逐步消融；DeepSWE pass@1 6.4→10.4，普通SWE Verified/Multilingual/Pro大多在约1点运行波动内。价值在于验证“长上下文训练并非一次永久获得”，不能把全套最终agent成绩都归给中训。

**后训/MTP。** text→VL两阶段SFT、RLVR、专项agent/reasoning/vision/RLHF教师，之后两阶段MOPD整合；学生生成后用所属教师token log-prob减学生log-prob作dense信号，不采教师响应、不依赖verifier。异步一版本stale以importance权重修正且[.2,5]外mask，报告承认context偏移及mask仍引偏。stage2引入teacher/student冻结混合 α=.2限制负信号；128 GB200 nodes，30+40 steps。最终冻结骨干单训MTP25K步，7-token draft在SPEED-Bench的平均接受长度，temp1由3.787→4.612；不是端到端tok/s倍率。

**量化边界。** NVFP4不是全模型四比特：MoE/shared GEMM NVFP4、attention/Mamba linear FP8、vision/connector与latent投影BF16，SSM FP16+随机舍入；whole-model4.82 bits/element，不含cache/activation。SFT-QAD200 iterations后36bench相对BF16分数比中位数99.62%，两边评测都用FP8 KV；verbosity的 **Ratio of Totals +0.44%**（先将各benchmark的平均轨迹token长度求和，再算整体NVFP4/BF16变化），不是逐benchmark比值的算术平均。36基准是作者汇总口径，本轮未自行重建表中多指标行的精确纳入名单。MMLU-ProX76.91→73.76（-3.15pp）、AA-LCR68.62→67.69，不能写无损。卡OpenMDW1.1；本轮没有执行训练或推理。

**判断/建议。** 深入阅读并考虑复现“256K中训→1M恢复”与按域MOPD的消融；部署数字的条件另见硬件与软件协同章节。稳定 ID：`report:nvidia:nemotron-3.5-super-lightning:2026-10-14`；关联HF版本见上。

### NVIDIA Nemotron 3.5 Bolt：结构压缩后做2bit QAD，24GB目标明确排除视觉模块

**日期/来源。** 10-15 [研究页](https://research.nvidia.com/labs/nemotron/Nemotron-3.5-Bolt-24GB/)与19页[技术报告](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-Technical-Report.pdf)；量化卡 `c6cc2e8ffa353403213d6b9b18aaf8eb684d705a`，BF16 `a6a1e5030819a130a1ea08aadb1fe42f4cc86777`。BF16/24GB/GGUF/LiLiCorr工件可见；公告提BoltE4B是**10-23未来计划**，不得写已发布。

**压缩/数据。** 从SuperVL120B-A12B以Star Elastic压到74B-A9B，保持40 Mamba/40MoE/8attention与hidden4096，专家topK22→9；budget-conditioned router联合选择各层专家数和intermediate width，以预算loss+parent logit KD训练。70% parent SFT+30% Ultra-v3预训数据；8M token校准，8B@8K架构搜索+64B@8K恢复+48B@128K，共120B。作者报32 GB200 nodes/128GPU约64h，这是压缩/恢复阶段成本，不能当整模型全部训练成本。重新接回vision tower后只训练VL投影，再vision-heavy SFT、MCP/软件agent RL、RLHF和checkpoint merge恢复。

**低bit机制。** routed/shared expert FC层为IQ2_XXS **2.0625bit**；Mamba linear NVFP4、attention/latent projection/embedding/head Q8_0，router/conv F32。26B token QAD以冻结BF16教师，采用词表分布逐元素min/max的weighted Jaccard loss、temp1、不加CE；再以最终更省token的教师做10M token RLHF blend蒸馏。不是所有层2bit，也不是无损。报告Table7显示TerminalBench2.1 BF16 53.93→量化50.98、AA-LCR67.38→64.06，量化目标是固定体积的能力权衡。

**资源/性能限定。** §8.2 的24GB按batch1、context131072、Mamba prefill chunk8192估算：权重22.01GB、cache/activation0.87GB、drafter0.96GB，报告估算总计23.83GB（分项显示值经过舍入，不构成严格加法等式）；**不含vision encoder/projector1.68GB，仅文本部署预算**。所以不能写完整多模态agent在24GB全装下。报告Table8：同RTX Spark（Surface Laptop Ultra128GB）、llama.cpp、Windows11 ARM x64、SPEED-Bench-Coding，Bolt+LiLiCorr83.99TPS，Qwen3.6-35B Q4+MTP96.63、Gemma4-26B Q4+MTP100.56；AA-AgentPerf Local replay600.80/453.97/480.68秒。原文未给完整输入长度分布与固定llama.cpp SHA，不从高质量宣称同体积普遍最快。

**判断/建议。** 考虑复现结构搜索+低bit恢复与teacher切换；只按目标任务/内存/交互速度的Pareto评估。OpenMDW1.1。稳定 ID：`report:nvidia:nemotron-3.5-bolt:2026-10-15`。

### IBM Granite Switch 4.1：同一endpoint组合aLoRA决策、聊天与ASR，避免误写成联合多模态预训

**日期/增量。** 新8B/30B composed checkpoint于10-14晚上传，[官方博客](https://huggingface.co/blog/ibm-granite/granite-switch-4-1)10-15发布；不是5月preview仓README的普通刷新。8B卡固定 `0d1625d16df2d54cafb6bc5cb0650aec043f77d6`；已读[卡](https://huggingface.co/ibm-granite/granite-switch-4.1-8b/blob/0d1625d16df2d54cafb6bc5cb0650aec043f77d6/README.md)及[BUILD](https://huggingface.co/ibm-granite/granite-switch-4.1-8b/blob/0d1625d16df2d54cafb6bc5cb0650aec043f77d6/BUILD.md)。

**机制。** Granite4.1-8B/128K base+guardian-core rank16 aLoRA+decision rank64 aLoRA，以control token按需激活，能复用激活前的base前缀KV。decision对单token option label取logits并softmax，一步输出相对分布；不等同自由生成答案或经过外部实测的置信度。嵌入470M GraniteSpeech5TurboCTC先转录英文音频再拼入chat；卡明确encoder与LLM**未联合训练、无训练的跨模态projection**，转录小写无标点且无decode-time steering。

**证据/边界。** BUILD给base8.381B/composed8.776B（+4.72%）；公开DecisionIndex fast1068行v0.3、固定State+Decision提示、hosted分42.9，local/hosted约.6点内一致。这只能说明指定subset与serving路径，不能据一轮Doom演示或未披露环境的race宣称通用10倍提速。卡的benchmark固定system prompt与推荐的低级接口“无system共享prefix”是不同调用路径，不能混写为一个完全相同实验。Apache2.0；自定义adapter SDK仍为计划。

**判断/建议。** 深入阅读模块化endpoint与prefix重用契约；复现分类应固定label tokenization、softmax范围及adapter激活位置。稳定ID：`hf:ibm-granite/granite-switch-4.1-8b@0d1625d16df2d54cafb6bc5cb0650aec043f77d6`。


### 回溯模型补录


#### Meta Muse Wonder：10-12闭源新系列与模型卡补漏，训练细节尚少

[官方公告](https://research.meta.ai/blog/muse-wonder)10-12、[13页模型卡](https://research.meta.ai/static/muse-wonder-model-card)明确Wonder与WonderFull同日发布，1M context、text/image/video/audio→text；Wonder走API/MuseCode，Full面向MuseCode订阅。本轮从恢复的research.meta.ai入口发现，属于Meta旧动态渠道覆盖缺口的补录。公告称更大更强base及长任务训练带来OOD泛化，未披露参数、tokens、FLOPs或独立预训/后训消融，不能用“gigawatt Prometheus”推算模型训练算力。MemoryBench是64K–100K带冲突/更新事实的内部多轮评测；FocusBench是内部动态阻碍任务，不等同所有真实个人助理成功率。模型卡统一max effort，外部编程成绩保留各厂商harness且取最高可比结果，SWE-Marathon尤其不受控。适合短录新模型/评估边界，不作可复现训练配方重点。稳定ID：`meta:muse-wonder:2026-10-12`。


<a name="pretraining"></a>

## 预训练与架构

以下三篇均为72小时回溯窗首次详报；论文原始日期见每条，不当作10-15/16首发。

### LatentLoop：压缩第一轮循环，把算力还给宽度

- **机构/项目**：上海人工智能实验室；SJTU、USTC、CUHK、NTU（四组按原文缩写保留）（LatentLoop）；作者Zefeng He等。**身份/时间**：arXiv **2610.19975v1**，2026-10-14T10:57:55Z。代码基线 **lcqysl/LatentLoop@deef911a45f68a127b0ba5e52dd12ac1abc9783c**，10-15T07:47:34Z；其前一提交`4ac0c32`（07:45:01Z）增加checkpoint和原生推理loader，属于本次增量窗工件变化。没有据此证明模型权重首次公开时刻。
- **方法**：循环MoE两次访问中间层：第一遍相邻token均值压成一半长度，第二遍保持完整序列；节省计算扩大宽度。共享router/expert，attention各自独立；可学习逐通道/接收位置的写回系数。写回必须因果对齐，源`[2j,2j+1]`只能更新`2j+1,2j+2`（零基）。
- **正文证据**：§4、Tables2/4/5、附录A/B/E。FineWeb-Edu、TinyLlama词表，1.3B/3B/7B总参（约0.2B/0.5B/1B激活）、15个预算端点、4096上下文、batch256。参数/KV/step-FLOPs偏差均<1%，总理论FLOPs偏差≤0.00943%。3B/40B参考预算，NLL为Base **2.0426**、Loop **2.0293**、LatentLoop **2.0199**；7B/140B参考预算九任务均分，5-shot **62.19/62.55/63.40**，0-shot **60.18/61.40/61.57**，并非各任务均胜。NLL1.988处的**19.0%/7.9%省算力是拟合前沿预测**，不是实测训练时长。
- **消融/成本边界**：3B/40B将压缩移到第二遍，NLL **2.0386**；两遍都压缩 **2.0467**。单一全局写回scalar已恢复大部分增益（2.0210），逐通道系数2.0199，不能全归因于精细decoder。实际GPU-time相对Loop在3B/7B约省9%/13%，但运行日期/资源池有差异，1.3B基线8GPU、LatentLoop64GPU；完整7B阶梯（含分支及资格运行）约64×H200×35h。理论FLOPs排除通信/重计算/优化器等；15端点复用训练前缀，不是15独立seed。
- **实现核对/建议**：已读`model/latent.py::latent_correction`的完整pair构造与延迟写回，`ScaledDecoder.forward`的1+delta系数，以及`model/model.py`每次层访问独立KV和MoE共享路径。README明确只发布最终架构/九配置/预训练入口，内部消融和集群分析工具省略。建议先做整序列/缓存增量的一致性与等预算对照，再考虑移植。
- 来源：[版本历史](https://arxiv.org/abs/2610.19975)、[正文v1](https://arxiv.org/html/2610.19975v1)、[固定实现](https://github.com/lcqysl/LatentLoop/tree/deef911a45f68a127b0ba5e52dd12ac1abc9783c)。


### CAST：历史二阶矩定尺度，裁剪后谱范数定阈值

- **机构/项目**：美团（Meituan，北京）（CAST）；作者Jiacheng Li等。**身份/时间**：arXiv **2610.19761v1**，2026-10-14T08:15:43Z。
- **方法**：在AdamW前，用上一时刻二阶矩的行列摘要标准化矩阵梯度，做soft spectral clipping，再还原尺度进入AdamW；每矩阵的裁剪后谱范数EMA定下一步阈值。前40成功更新校准，状态只在优化器成功更新后提交。**并非所有MoE参数都走谱分支**：dense组二维矩阵（排除embedding/head）被选中，expert等走历史输出norm控制器。
- **正文证据**：Tables1–4、§5和附录B/C。19.08B总参、约647.6M目标激活backbone（不含embedding/head）的shortcut MoE，83.89B nominal token、5000步，seq8192、global batch2048、BF16、EP8；单一seed48。20子集平均验证loss **GlobalGC1.8962、AGC1.8856、SSC-constant1.8864、CAST1.8738**。去历史尺度为1.8747，固定/余弦阈值变体1.9215/1.9034，表明本设置阈值适应的贡献明显大于尺度项。相对梯度spike为CAST1次、SSCconstant22次（定义为超过前100步中位数2倍）。
- **性能及局限**：64设备，设备约为H100算力60%，正文**没有给出具体SKU，不能写64×H100**；尾200步CAST44.408s、SSCconstant43.533s，单步慢2%。达到最优baseline最终平滑训练loss少7.2%步数；结合上述timing得到的5.5%时间节省是估算（对照ClipByValue步时），不是完整多seed time-to-quality实测。语料/tokenizer仅称recorded configuration，本轮未核得公开专用代码/完整数据清单。建议先核all-gather完整逻辑矩阵的通信成本，再做多seed和下游对照。
- 来源：[历史](https://arxiv.org/abs/2610.19761)、[正文v1/Tables1–4](https://arxiv.org/html/2610.19761v1)。


### 原生多模态预训的视觉消融：保留关联文本，才能归因视觉增益

- **机构/项目**：香港中文大学；腾讯Foundation Model Department（More Than Seeing）；作者Haoyuan Wu等。**身份/时间**：More Than Seeing: Vision as Scalable Anchors in Native Multimodal Pre-Training，arXiv **2610.19443v1**，2026-10-14T02:33:34Z。
- **设计**：六档71M→3B**激活非embedding参数**的原生MoE，从头训练，无预训视觉encoder；250B text-source+75B多模态输入token、固定r0.3。只将image patch embedding换成content-free pad embedding，保持文本、顺序、位置、长度和text loss mask，避免把额外caption文本的收益冒充视觉收益。
- **证据**：§2.4/3–5、附录B/F和Tables9–12。325B nominal输入时，15个文本空间任务按六组等权，A3B image **44.25%** vs pad **39.87%**（+4.38点）；总体15项语言基准接近。A3B训练沿数据轴，定位/导航/符号规划的平均image-pad差为 **+2.88/+3.02/+5.19点**，另外三组均值差在±0.6点内；视觉增益不是所有任务/尺度单调。5–12块Blocksworld长计划image16.62%、pad12.99%，完整可执行且达目标才算成功。
- **不能略掉的负面**：A3B image虽减少“合法但未达目标”的计划（523 vs1121），却增加前置条件失败（882 vs427）；不能写视觉训练消除执行错误。纯文本loss最优前沿和N/D配比几乎不变；视觉对多模态loss的约6.2%差（1e20多模态FLOPs处）是拟合函数值。预算点共享/相关，拟合好不证明训练seed复现；固定配比、私有web混合语料、未知完整硬件/卡时，且本轮未核得公开专用实现/配对checkpoint。建议多模态CPT/预训对照保留“相同caption+pad”组和执行错误细分。
- 来源：[历史](https://arxiv.org/abs/2610.19443)、[正文v1](https://arxiv.org/html/2610.19443v1)。


<a name="posttraining"></a>

## 后训练与协作

### More Methods, Better Agents?：统一预算后，自蒸馏收益与失败都更清楚

- **机构/项目**：浙江大学CAD & CG国家重点实验室；蚂蚁集团（Self-Distillation-AgentRL）；作者Linhao Zhong等。**身份/时间**：arXiv **2610.19789v1**，2026-10-14T08:36:31Z。官方HTML404；已下载并逐节/表读**官方TeX source**。代码 **aim-uofa/Self-Distillation-AgentRL@35a44f5c5647b1785ad23ddf0ffbd196b695bf86**，10-15T03:16:59Z初始提交，在本增量窗。
- **协议**：10类自蒸馏对GRPO，Qwen3.5-2B/4B/9B×ALFWorld/ScienceWorld/Search-QA/WebShop共12设置、132次训练，每设置每方法100全局步，单节点8×H800、FSDP、全参、AdamW1e-6；总记录 **12,124.4 H800卡时**（含训练10,580.7、循环验证1515.3、保存28.4；不含启动/额外重评）。最终checkpoint用T0.4、**8个rollout评测seed**，不能写8个训练seed。
- **主结果/负面**：表`generated_full_performance_cost`：12设置等权混合任务指标均分 **GRPO49.68、PCSD54.26、AgentOPSD51.34**，另外8方法低于GRPO；各设置并无统一赢家，PCSD6胜5负1平。120组对GRPO比较中112更慢，其中58最终更差、7持平。PCSD/AgentOPSD平均配对净训练时间约 **1.167×/1.187×**。等GRPO净训练时间预算，ALFWorld三规模均分AgentOPSD step100 **68.10→预算内37.63**，低于GRPO58.10；WebShop PCSD **73.17→63.83**，低于GRPO68.20。这两组是**训练验证协议指标**，不要与8seed最终重评混称。
- **稳定性**：4B ALFWorld AgentOPSD step35验证60%，steps45–70六次≤4.3%，最终79.3%，说明好终点会掩盖严重中间崩溃；9次运行最终较峰值低≥20点。非有限梯度与失败不一一对应，不能把全部退化归因NaN。
- **边界/实现**：共同框架中仍保留方法特有超参，并含Skill-SD/RLSD适配；GRPO+OPSD沿用SDAR constant-gate实现，不等同所有OPSD数学变体。每配置只一训练运行、100步、四文本环境、512token/turn，评测复用固定验证任务；不证明所有长程Agent或这些方法的最佳调参均无效。已读固定README的launchers/论文配置和8seed评测说明，未逐函数重审432文件或跑132次训练。建议所有新OPSD方案保留GRPO、匹配墙钟、全曲线及训练seed重复。
- 来源：[abs/历史](https://arxiv.org/abs/2610.19789)、[官方TeX v1](https://arxiv.org/src/2610.19789v1)、[PDFv1](https://arxiv.org/pdf/2610.19789v1)、[固定代码](https://github.com/aim-uofa/Self-Distillation-AgentRL/tree/35a44f5c5647b1785ad23ddf0ffbd196b695bf86)。


### CaLIF：约束可验证，不等于答案有用

- **机构/项目**：延世大学；华盛顿大学；FuriosaAI（CaLIF）；作者Sangyeon Yoon等。**身份/时间**：Diagnosing and Correcting Reward Hacking in Verifiable Instruction Following，arXiv **2610.19525v1**，2026-10-14T03:42:59Z。
- **方法**：冻结reference分别在“请求+约束”和“仅请求”生成基准响应，二者在仅请求下的平均token loglikelihood形成区间；区间内、verifier reward>0才加bonus，低于下界只保留verifier，高于上界置零。复用KL reference，离线求界，每rollout至多额外一次teacher-forcing；不是新增judge训练。
- **证据**：§5/6、Figures5–8、Tables1/4、AppB。Qwen3-1.7B/4B双模式+Tülu3-8B-DPO五配置；95,373多约束prompt，verl GRPO、8×H100、8rollout、batch1024、4epoch（4B thinking3epoch），T1、KL0.001，响应2048/8192。IFBench五配置平均：base约束满足率 **26.98**/答案质量 **4.51**；纯IF-RLVR **44.93/2.96**；CaLIF **34.64/4.59**。质量1–10由LLM judge评分、约束0–100，不能相加。CaLIF在三rubric基准相对base40/45指标改善，纯RL14/45。
- **消融/权衡**：Qwen1.7B直接奖励likelihood，质量 **3.98→1.91**，长度485→1677、unique2gram0.713→0.177，出现重复刷似然；去下界/上界质量2.16/3.51，均低于完整CaLIF4.01。CaLIF保持质量但**约束分低于纯IF-RLVR**，是Pareto权衡，不能称所有指标领先。主质量judge GPT-5.6 Sol，frontier诊断另三judge；报告3次评测不等于3训练seed，未给完整训练卡时。区间依赖base能回答请求且offline抽样可靠，不能保证开放语义正确性。正文仅链接通用verl和IF数据，本轮未核得专用CaLIF公开实现。建议指令RL验收同时保留语义质量、约束成功及重复率。
- 来源：[历史](https://arxiv.org/abs/2610.19525)、[正文v1](https://arxiv.org/html/2610.19525v1)、[训练数据](https://huggingface.co/datasets/allenai/IF_multi_constraints_upto5)。


### 【昨日待办补齐】VRSD：教师提示改变特权解的使用方式

- **机构/项目**：King’s College London；LMU Munich；Munich Center for Machine Learning；腾讯优图实验室（VRSD）；作者Jiazheng Li等。**身份/时间**：Self-Distillation with Verbal Regularization，arXiv **2610.16354v1**，2026-10-12T08:11:35Z，在72h交叠窗。昨日待办的VRSD与Jeva ID在调度文字中颠倒，本项已按论文标题核正。
- **方法/证据**：保留相同自教师、完整验证解PI和蒸馏目标，只给教师加验证/反思/替代路径提示；multi将同题8rollout均分到aspect提示。Table1三模型×评分/化学/SQL/Python四任务，multi相对OPSD 11/12 Pass@1改善，四任务均分Qwen4B **71.2→78.4**、OLMo3-7B72.4→75.2、Qwen8B75.9→77.5；唯一回退Qwen8B Python65.6→65.4。4B SQL baseline71.3、OPSD69.8、multi81.0，学生token entropy0.03→0.54；不是所有entropy或Worst@16均提高。
- **协议/理论边界**：训练batch32、8rollout、无KL、earlystop连续4验证无改善；评分/化学/SQL2×H10080GB、2048响应，Python2×H200141GB、8192响应且不同LR/divergence配方；评测每题16样本T0.6/top-p0.95。平均3.9点是跨三模型/四任务，不是任意LLM通用提升；seed与总卡时无充分对照。熵结论依赖仿射PMI衰减及协方差条件，提示不保证普遍熵增加；earlystop与额外教师提示token也使等step不等于等成本。
- **实现核对**：**lijiazheng99/VRSD@dc203aba11f45c4fce156922ba5f65a293675b6e**（10-14T07:32:35Z）。已读`scripts/train.sh`方法分支与`verl/trainer/ppo/ray_trainer.py::_build_teacher_message`：aspect按rollout索引分桶，改造teacher reprompt，append学生responses作teacher-forcing，未修改学生原始prompt。公开数据prepare脚本可见，未跑训练。建议已有OPSD可作便宜prompt消融，同时保留prompt-only无PI控制及等总token/墙钟预算。
- 来源：[历史](https://arxiv.org/abs/2610.16354)、[正文v1/Table1/AppD](https://arxiv.org/html/2610.16354v1)、[固定实现](https://github.com/lijiazheng99/VRSD/tree/dc203aba11f45c4fce156922ba5f65a293675b6e)。


### 【昨日待办补齐】CoSIR：在可计算信息价值的玩具环境中训练协作

- **机构/项目**：亚利桑那州立大学；休斯敦大学；俄亥俄州立大学；约翰斯·霍普金斯大学（CoSIR）；作者Keru Chen等。**身份/时间**：Training LLMs for Multi-Agent Collaboration with Verifiable Information Rewards，arXiv **2610.17893v1**，2026-10-13T07:46:48Z。
- **方法**：分布式rating-table任务，Bayesian evaluator只看每agent初始证据及收到的报告（不用真实隐藏项）用动态规划计算“哪选项最优”的概率；团队对正确选项的surprisal下降作为分享奖励，请求按潜在下降奖励，另验陈述真实性/最终私有答案。自然语言报告由规则+**冻结初始Qwen3-8B parser**解析，可计算reward不等于无解析误差。
- **证据**：Table1、§4、AppF/H。三模型各3轮新游戏收集，每轮约5h含采样训练、合计约15h/运行；实验总体用2×H100和4×Gaudi2，但未明确每模型/阶段分配，**不可把15h直接换成固定总GPUh**。Qwen8B七项OOD得分SUM **352.60→384.44**，即七任务等权平均约50.37→54.92（+4.55点），不是+31.85点平均accuracy；Outcome-RL363.72，ToM354.17、TalkHier364.65。Qwen4B/Gemma SUM333.17→357.87、208.02→235.94。每任务4次评测，不等于4训练seed。
- **局限**：Qwen4B SplitQA57.50→57.86，变化小于std；有些单任务其他方法更强。MAST specification/system-design失败88.2%→88.3%，协作训练不改善所有失败模式；Einstein绝对分仍低。GSM8K/MMLU/IFEval变化[-0.6,+0.8]点。Toy环境有已知事实、可算后验，开放任务不能直接复制信息奖励；奖励还混有一般prompt imitation和request demonstrations，不把全部收益归为surprisal一项。本轮正文未给独立官方代码/数据链接可核，移植前仍需工件。建议先用于有已知证据分片的受控任务，保留Outcome-RL与相同通信预算。
- 来源：[历史](https://arxiv.org/abs/2610.17893)、[正文v1/Table1/AppF–H](https://arxiv.org/html/2610.17893v1)。


<a name="inference"></a>

## 推理与部署研究

### 【昨日待办补齐】Jeva：把候选编码缓存与当次集合决策分开

- **机构/项目**：新加坡国立大学（Jeva）；作者Qi Li等。**身份/时间**：Jeva: System One Models from First Principles，arXiv **2610.18773v1**，2026-10-13T17:31:09Z。代码 **LiQiiiii/Jeva@f231c9ac31ba72f2ca4ca8aac662b279f3438fdb**（10-14T13:09:38Z）。
- **架构**：Qwen3.5-4B分别编码请求/每候选；单个768维决策block只更新请求state，对不变候选memory读一次；共享打分支持可变候选。coverage估计将null与候选竞争分开，支持Choice/Noul/Score；候选语义向量及每层KV可跨新集合复用。LoRA64+投影/决策head13.37M训练参数，多阶段数据含replay，未给完整训练设备/成本。
- **证据**：§4/Figures6–9、AppC。600 Choice、最多五种顺序，**0 flip、TV3.17e-8**；仅证明测试顺序表现，精确数学置换等变仍有tie/浮点边界。BANKING77 K16的3080项（588无答案）null AUROC **98.66%**，5.02%FPR处检出93.88%。Typed Decisions accuracy78.30，略低于InternDecision4B78.70；Mind2Web只测16候选选择，不是完整网页Agent。
- **性能/约束**：单H200、不同模型保留原runtime。ToolRet K2048优化batch后p50 **12.040s**、accuracy69.88，recursive JevK5 **28s/67.47**；仅Jeva旧chunk2→新bounded batching就有94.459→12.040s大变化，不能全归架构。新集合完全复用MLDR候选，p50 **2102.0→75.1ms（27.98×）**，**priming单独计时并排除**，不适用于全冷候选；论文主文说50 matched queries，附录C.5 figure说46 novel-set requests/level，样本数口径不一致，本轮保留。cache组decision不变但maxprob差0.00736，不称bit-exact。
- **实现核对/建议**：已读`matrix_decision.py::prepare_option_cache/compile_option_memory/forward_from_cache`，投影KV只依赖候选，detach cache禁止训练模式，softmax条件候选概率与sigmoid coverage组合null。需weights/tokenizer/template/length/numeric配置不变才复用，更新参数后失效；near ties需要身份tie-break。适合工具路由/标签选择的结构消融，先测试冷/部分/全复用、额外存储、真实候选扩张干扰。
- 来源：[历史](https://arxiv.org/abs/2610.18773)、[正文v1](https://arxiv.org/html/2610.18773v1)、[固定代码](https://github.com/LiQiiiii/Jeva/tree/f231c9ac31ba72f2ca4ca8aac662b279f3438fdb)、[权重入口](https://huggingface.co/LIQIIIII/Jeva-4B)。


### FOLD：草稿保留局部位置，远历史单独归一化读取

- **机构/项目**：中山大学；Canva Research；北京大学（FOLD）；作者Shuxuan Li等。**身份/时间**：arXiv **2610.20392v1**，2026-10-14T15:55:10Z。实现 **NuyoahNasuS/FOLD@22beb43a7704beeabd6e5a242404aa2f44982a79**（10-14T14:41:10Z）。
- **方法/证据**：不训练、不改target verification；draft的局部window保留真位置，远历史映射到共享位置，window/memory分别归一化后混合，以alpha2减轻注意力稀释。§4/Table2/3、AppB/C；vLLM0.28、单H10080GB、BF16、batch1/greedy、CUDA graph、prefillchunk8192。七长上下文benchmark（约1K–256K）DFlash平均**decode**加速：Llama3.1-8B **2.11→3.40×**、Qwen3.5-9B2.98→3.39×、Gemma4-12B2.46→2.68×，均相对AR；speedup是逐请求TPOT比值均值、跨任务等权，**不含prefill**。Llama LCA122K的2.08→4.22×是特例，不是所有任务翻倍。
- **负面/一致性**：短于16K且原draft未退化时可略慢；Gemma部分LCA StreamingLLM更快，FOLD非处处胜。额外未旋转side KV随上下文线性增长，最大Qwen3.5约6GiB，高并发显存代价未实测。Qwen draft读取层数差异、baseline storage matching需保留，不能只说“一个位置技巧”。SGLang0.5.18复核1531请求，FOLD平均1.54→2.50×，但**83条与AR输出不同**；其他方法79–84条，均超作者原约2%差异上限，作者事后改为同输入测速、没有逐条核完。排除任一方法不同的94条后1437共同集排序不变、各均值变化<0.01。精确算术的分布保持不能替代BF16输出一致性检验。
- **代码/建议**：已读`fold/reference.py::fold_block`：远key共享虚位置、alpha-softmax和log-partition/alpha混合；README及`fold/attention.py`说明两次FlashAttention+side cache、vLLM plugin及DeepSpec patch；未跑kernel。建议先以相同输出集/同GPU配对测TPOT、TTFT、侧缓存和并发，再纳入线上性能承诺。
- 来源：[历史](https://arxiv.org/abs/2610.20392)、[正文v1/附录C.3](https://arxiv.org/html/2610.20392v1)、[固定实现](https://github.com/NuyoahNasuS/FOLD/tree/22beb43a7704beeabd6e5a242404aa2f44982a79)。


### ReaDy：从混合线性注意力的已提交state补充草稿上下文

- **机构/项目**：StartLux；中国科学院大学；香港大学（ReaDy）；作者Bohan Sun等。**身份/时间**：arXiv **2610.20503v1**，2026-10-14T17:08:52Z。代码 **StartLuxLabs/ReaDy@8ac61b51d529fad980566cfde7a3c41415ba93a5**（10-15T02:49:13Z增量窗提交，内容是添加论文链接，不能当新算法/新kernel首发）。
- **方法/证据**：DSpark backbone+sequential head，给每draft位置学习query，从target已提交递归state（层30/62）作rank8读取并在最后draft层加残差。20k reader-only+50k reader/LoRA两阶段适配，target冻结，CE+0.1L1。§4/Table1/2/3–6；Qwen3.8-27B、单B200、SGLang、BF16、concurrency1、thinking、greedy、max1024输出/16384上下文、block7。七任务平均TPS **AR95.86、DSpark280.42、ReaDy302.78**；报告decode约3.16×、相对DSpark+7.97%，不含prefill。无state的额外finetune已到286.50TPS，ReaDy相对它**+5.68%**，故7.97%不能全归state访问。
- **消融/局限**：rank128接受长度略高4.5404 vsrank8的4.5285，但throughput gain7.81%<7.97%；多层注入速度更慢而接受长度近同。论文给四GPU训练loss聚合，但硬件型号/时长、训练样本具体混合比例和seed无完整公开；只一target、一低并发解码场景，没证明采样、长输出、量化、多target通用收益。理论分解只动机说明state有额外方向，不声称精确恢复未来readout。
- **工件核对/建议**：已读`reader.py::StateReader.forward`和`adapter.py::install`：固定5120输入、2×48头×128×128state形状、rank8、层4hook、每次propose从30/62层temporal cache收集，**公开适配器batch≤4、draft7**，不是任意HLA通用插件。仓库提供推理实现与reader权重入口，没有完整两阶段训练入口；README明确图是论文结果，非新Triton后端完整复测。建议先保留DSpark+FT控制，并验commit/rollback生命周期与state同步，测新后端而不沿用论文TPS。
- 来源：[历史](https://arxiv.org/abs/2610.20503)、[正文v1](https://arxiv.org/html/2610.20503v1)、[固定代码](https://github.com/StartLuxLabs/ReaDy/tree/8ac61b51d529fad980566cfde7a3c41415ba93a5)、[发布权重入口](https://huggingface.co/startlux-models/ReaDy-Qwen3.8-27B)。

<a name="evaluation"></a>

## 评估与可靠性

### Base LM评测协议：同一批模型，结论能被评测装置翻转

- **机构/项目**：奥斯陆大学信息学系（Base LM Evaluation Setups）；作者Shenbin Qian等。**身份/时间**：How to Evaluate When You Don't Know How to Evaluate? Choosing Reliable Setups for Base Language Model Benchmarking，arXiv **2610.19185v1**，2026-10-13T20:50:28Z。
- **证据与方法**：26个不同语料的2B单语模型、4语言80任务、中间checkpoint，加三个≥7B模型；扫8个评测维度，以单调性、非随机性、后期排名一致性的等权percentile作为meta-measure。Figure1中相同6个挪威语模型，仅换题式/归一化/汇总/分数定义，最终排名Kendall **tau=-0.6**，同一语料可从第一变末位。
- **可借鉴结论**：多prompt、5-shot和字符归一化answer probability一般更有信号；cloze对2B较好，较大模型MC/accuracy更有竞争力。**多prompt最大值并非天然最好**：附录E 672成对setup，max在非随机性80%胜、mean在单调性58%胜；max相对单prompt chance有机械抬分，最早2B token checkpoint已多3.0点，非全是能力信号。建议预训选数据前先证明排序对合理协议稳健。
- **成本与边界**：LM Eval Harness+vLLM、单checkpoint单GH200120GB、固定fewshotseed；2B评测约11,000GPUh，大模型约7,000GPUh；这些是评测成本，非训练成本。四种欧洲语言，题式变体仅芬兰/挪威、生成任务仅挪威；大模型仅三种且训练预算/架构不同。meta-measure是代理，无“最好评测”的groundtruth，高单调/稳定也可能只测狭窄能力。正文给开放checkpoint集合但本轮未核专用代码完整工件。
- 来源：[历史](https://arxiv.org/abs/2610.19185)、[正文v1/附录E](https://arxiv.org/html/2610.19185v1)、[开放预训模型入口](https://huggingface.co/openeurollm/prelude)。


<a name="core"></a>

## 工业开源：核心六项目


### 非默认 batch-invariant RMSNorm 修复 forward 与输入梯度

稳定ID：`github:NVIDIA/Megatron-LM:pr:6859`；[PR #6859](https://github.com/NVIDIA/Megatron-LM/pull/6859)，main，2026-10-15T05:09:19Z，merge `cea0980bd1813ef62ed34d0b47a972431632977e`。

forward 原来算出但没用 zero_centered_gamma 的有效权重；backward 的方差项又多乘一次逐元素 weight。权重全 1 初始化会把第二个错误完全掩盖，默认 te_native 不安装此函数，实际影响显式 deepgemm/triton batch-invariant 路径。

作者在 8×L4 / torch2.12+cu130 与 CPU float64 独立解析/有限差分 harness 上验证：非单位权重的 fp32 输入梯度误差 0.9878→2.599e-5，forward 对独立公式一致；未装 TE，因此直接调用 autograd Function，未做 patched TE 层或端到端训练。**最终 diff 已增加随机非单位权重、两种 gamma 及三次 replay 对照测试**，PR 正文早期“未加测试”的勾选不是最终 diff 状态；不能将新增测试代码当作全部已执行。

建议：确定性测试同时核数学正确性，权重必须离开初始化值；不把本修复写成默认训练受影响或收敛已证实。

### MFSDPv2 允许 shared-layer MTP 重入 backward

稳定ID：`github:NVIDIA/Megatron-LM:pr:7757`；[PR #7757](https://github.com/NVIDIA/Megatron-LM/pull/7757)，main，2026-10-15T06:36:02Z，merge `9131593673d037d360d252904f9198d36bc0c246`。

同一 FSDP block 在两个 MTP depth 中复用，一次 autograd 先进入 depth2 再进入 depth1，但 parameter post-accumulate hook 只在两次贡献收齐后触发，第二次 pre_backward 会遇到 BACKWARD→BACKWARD 的非法状态。补丁在 backward 已活跃时跳过重复 setup，让既有 gradient-ready 回调执行归约/reshard。

作者两张 RTX A6000 最新记录 context/fully-shard suites 每 rank 45 passed/9 skipped；五个 SGD step 将两深度 shared residual MLP 与相同输入/权重的未分片 baseline loss 对照。正文另有旧 44/6 记录，不能累计。普通 autograd、无 checkpoint，四 rank cases 跳过；phase completion 与 gradient-ready 进一步拆分仍由 #7764 跟踪。

建议：共享层/多次调用在同一次 backward 内独立测状态机；不外推 checkpoint、长训练或所有重复调用拓扑。

### MFSDPv2 的 reduce-scatter 输入终于遵循 grad_comm_dtype

稳定ID：`github:NVIDIA/Megatron-LM:pr:8185`；[PR #8185](https://github.com/NVIDIA/Megatron-LM/pull/8185)，main，2026-10-15T22:38:57Z，merge `0adca6528c9fb100cd4ccc425f410795de9beafa`。

partial_grad 以前采用 compute .grad 的 dtype，BF16/MXFP8 模型即使配置 FP32 gradient communication 也可能仍用 BF16。现在取 MixedPrecisionPolicy.grad_comm_dtype，未设置回退 main-gradient dtype，并让最终 .grad 保持 main-grad dtype。实际 patch 覆盖普通与量化 parameter group。

作者说明四 GPU 通过；新增测试核默认、BF16 override、未设 main_grads_dtype，以及 MXFP8 compute dtype 回退和 baseline 训练对照。没有可引用训练吞吐/长程质量结果。

建议：低精度 A/B 先记录通信 buffer 的真实 dtype，否则 MFSDPv1/v2 比较可能没有对齐条件。

### GDN decode 直接按 slot 原地更新 recurrent state

稳定ID：`github:NVIDIA/Megatron-LM:pr:7896`；[PR #7896](https://github.com/NVIDIA/Megatron-LM/pull/7896)，main，2026-10-15T06:24:32Z，merge `9bd7ba4c68b91f7db1a0bb546dba73f05d8509c4`。

用 slot-indexed recurrent/conv kernels 替代 gather→FLA→masked scatter，padding slot=-1 在读输入/state 前退出并写零；仍不支持 GDN speculative state capture。

GB200、Qwen3-Next GDN 维度、CUDA graphs：每层 bucket/active=128/32，BF16 state 229.3→82.0µs（2.80×）；512/512 为944.3→485.2µs（1.95×），每步16→9 kernels。12 GDN+4 attention+16 MLP 的 hybrid DynamicInferenceEngine 中，100/128/512 requests 的 median decode step 提速1.31/1.36/1.62×。活跃行 output/conv/SSM cache 与旧实现逐bit一致；这些是作者特定负载，非本轮复现。

边界：SSM suite 与 main 同有一项既存 GDN2 failure；非 decode graph prefill 的 FLA metadata lifetime IMA 仍由 #8151 跟踪。建议按实际 bucket occupancy 和 state 大小测，不能称所有 graph 内存生命周期已修。

### dev：shared expert 与独立 wgrad overlap 的启动点前移

稳定ID：`github:NVIDIA/Megatron-LM:pr:7390`；[PR #7390](https://github.com/NVIDIA/Megatron-LM/pull/7390)，dev，2026-10-15T16:06:00Z，merge `19b2454da6613b5cdbaccb460d5963b744c1d047`。

Flex dispatch backward 的 post-hook 在 shared FC1 backward/输入梯度合并前提交 expert wgrad，原输入节点仍等完成并执行梯度 hooks；per-forward state 防重发。常规 all-to-all 原调度不变。

8×H200、EP8/TP1、DeepEPv2、每rank16384 tokens、H1536、expert/shared FFN512、128 experts/top8，四独立运行/7680 unprofiled samples：单层 fwd+bwd BF16 12.339→11.960ms（-3.07%，95%CI 2.90–3.24），FP8 blockwise -1.97%。较小4096tokens、H/FFN2048、16experts/top2反而慢0.48%；DeepEPv1/v2/常规a2a各25 cases在8rank通过。

**只合入 dev，非 main/release；单层非完整训练吞吐。**建议先判别 shared/dgrad/wgrad 的真实重叠窗口和形状，再用实训确认。

### 多轮 block verification 不再复用已被决策条件化的随机数

稳定ID：`github:vllm-project/vllm:pr:61161`；[PR #61161](https://github.com/vllm-project/vllm/pull/61161)，main，2026-10-15T21:46:38Z，merge `9edcc7fa7ec858e13d4dc10ed968b741e2c95d6a`。

未被接受的位置在下一轮会被重访；旧 (request_seed, position) 同时复用 acceptance uniform 与 probabilistic draft Gumbel，造成跨轮偏差。新增共享 spec_step_seed 把 draft step 混入 seed 高32位，覆盖 V2 block rejection、cached drafting、DFlash2/LiLiCorr walk、DSpark 及普通 watermark fallback，无新 host sync/kernel launch。

真实 V2、一层 context-independent Llama target/draft、TP1/FP32、2 speculative tokens、每配置10000请求：第三token P=0，确定性block预期0.5，原0.4171→0.4989；概率draft block预期0.3，0.3558→0.2898。四更新配置通过六输出边缘分布、seed repeat、request reorder、split batch；负对照分别撤 acceptance/draft/candidate-walk seed helper 会失败。作者 CUDA 350 tests/3已有skip，CPU175/76skip。

限制：受driver限制用Triton attention与bounded KV；未测AMD、full-runner graph、adaptive-policy模型eval或吞吐。**standard probabilistic 的 seeded token 序列也会变化，但更新后仍可复现**。建议升级先更新 golden，并独立核目标分布，不能仅核固定seed输出相同。

### DeepSeek V4.1 KV 改 interleaved 记录，跨版本缓存不互通

稳定ID：`github:vllm-project/vllm:pr:61371`；[PR #61371](https://github.com/vllm-project/vllm/pull/61371)，main，2026-10-15T20:58:26Z，merge `f8460ef23659afd69046a81529ec2295e4c3f686`。

FlashMLA pin→5dceadb，V4.1 writer/gather/decode统一528B MXFP8及288B NVFP4 interleaved records；TMA要求physical block stride含整数record。配合已有DeepGEMM 256B indexer对齐，MegaAttention容量损失从潜在24.6%压到9.5%，并非无成本升级。

GB200/SM100、DSv4.1-Flash、DEP4、MegaMoE、MXFP4 indexer、DSpark5、maxlen8192：MegaAttention在相等KV memory下token容量较main少9.5%；两FlashMLA后端GSM8K1319题/5shot/greedy/1024cap/c32均92.87%±0.71，所有请求完成。writer→FlashMLA→gather中页/padded block读回测试及分组suite已读；SM100 V4.1 CI缺口由作者本地实测补，非普遍CI覆盖。

**此前/此后构建之间传输或offload的V4.1 KV字节不兼容**，V3.2/V4.0/SM90 legacy路径不变；无decode速度主张。建议滚动升级时将KV format/version纳入兼容协商并重算cache token容量。

### DP 小 CPU 元数据使用低延迟 Gloo collective

稳定ID：`github:vllm-project/vllm:pr:61767`；[PR #61767](https://github.com/vllm-project/vllm/pull/61767)，main，2026-10-15T22:29:14Z，merge `de7d70676d9a39353aab4796a6987d345c597991`。

以power-of-two block递归倍增再并发交换block sums替代小CPU张量 all_reduce，支持任意DP size与stateless groups；自动用于V2 padding/cudagraph sync及engine-core DP状态sync，无env开关。求和顺序固定以保证各rank一致。

GB200节点Ethernet/Gloo、6×N int32、每方法5000次随机交错、跨rank p50中位数：4node/16ranks 5354→546µs（9.8×），2node/8rank1977→362µs；这是collective微基准，未给服务端到端吞吐。分布式测试DP2/3/4/6/8/9/12、exact/int及float各rank逐bit一致；DP4 multiprocessing async LLM通过，Ray未测。

建议先用自己的DP size/NIC/CPU负载复测小消息延迟，不将9.8×写成模型服务提速。

### P2P control plane 离开模型步边界，但内存安全待办仍在

稳定ID：`github:vllm-project/vllm:pr:58168`；[PR #58168](https://github.com/vllm-project/vllm/pull/58168)，main，2026-10-15T05:50:56Z，merge `c5f8adc0b1944b9142a31d85cfbc6a61973d0b3a`。

新增OffloadingManager公共 lock/unlock/locked 覆盖整棵manager状态树；caller跨多操作持锁。on_schedule_end移到真正step末，P2P tier默认1ms control-plane thread可在model执行时服务peer，0关闭恢复每step。shutdown先stop/join再tear down。

两H200 pods、每侧Llama3.2-1B/enforce-eager、96GiB CPU tier、每次新约27K prompt/约1720chunks；source用8并发27Kprefill压到约1.4s step，warm3reps：consumer lookup delay0.2196→0.0534s，TTFT0.310→0.144s；idle基本持平。作者901passed/12skip、互斥失效负对照检测86次overlap。无专门服务吞吐A/B；14K chunk成功不替代NIXL所有线程模式认证，num_workers=1和外层锁是条件。

同日 [#56836](https://github.com/vllm-project/vllm/pull/56836)（05:23:35Z）为成功握手后未答lookup加30s progress deadline，仅新probe进展续期，session-wide失效并MISS回本地prefill；229CPU tests，当前head无新GPU/NIXL验证，pre-handshake ConnectAck仍另案。另 [#59096](https://github.com/vllm-project/vllm/pull/59096)（07:57:47Z）只允许mark_miss覆写RESOLVED，保护较新的PENDING/IN_FLIGHT probe，3回归在base失败、CPU tiering436/1skip。

建议：把control progress、probe generation与远程WRITE终止分别建门禁。以上**不能宣称 #61532 consumer abort/close 释放slot的竞态已解决**。

### MegaMoE 汇合成格式矩阵，新增 NVFP4 kernel 行尚无模型 caller

稳定ID：`github:vllm-project/vllm:pr:61538`；[PR #61538](https://github.com/vllm-project/vllm/pull/61538)，main，2026-10-15T19:02:02Z，merge `4d58fa254a869a5d9d0aab250ab125503d3cef26`。

统一DSv4/V4.1、KimiK3和底层backend为DeepGemmMegaMoEExperts；构造期按routed format/activation/shared格式确定融合，新增NVFP4 dispatch与scale处理。已有DS/Kimi继续MXFP4，**MXFP8/NVFP4/BF16新矩阵行尚无模型caller**，不是现成模型已获得所有精度支持。

GB200/SM100现pin：138 tests；真实kernel对相同量化weights/activations的FP32参考，calc_diff<1e-3且cos>0.99，含shared fusion负对照和staging逐bit quantizer对照。只有单rank kernel numerics，DSv4.1和Kimi端到端GSM8K以及多rankEP数值仍未做。

NVFP4 L2 activation max作为host scalar被graph捕获；reload若改变w2 input scales需要recapture，eager会重算。建议权重同步系统将scale改变与graph recapture联动；不能把结构重构写成端到端提速。

### TPU GRPO 从十步性能样例转向百步 HBM 稳定

稳定ID：`github:verl-project/verl:pr:8239`；[PR #8239](https://github.com/verl-project/verl/pull/8239)，main，2026-10-15T02:25:13Z，merge `f724e7f2c9c539f3f0ab260bf0363a1f5eaebee4`。

昨日 #8226 的microbatch4仅测10步，完整100步在step37遭不可恢复HBM fragmentation。新样例microbatch2，并pin hardware-plugin e5ab532（每microbatch backward TPU同步），删除Ray<2.59下阻止设置TPU_VISIBLE_CHIPS的env。

Qwen3-4B、2×TPUv6e-4 slices（4trainer+4rollout chips）、32prompts×8=256trajectories、512/1024token caps：作者100/100steps完成，48m45s总时长；steps6–100（95steps）mean step25.75s/median24.82，update_actor mean14.98s，约6048tok/s是全8chips/allsteps口径。不是与昨日2–10step均值32s的受控等长A/B；生成长度、学习过程与overlap readiness均变，不能直接包装为新2倍加速。39.1→95.3 GSM8K val是该训练跑观测，非独立泛化结论。

建议：配置验收延长到真正运行长度，记录fragmentation retries/分阶段计时；用pin重现，别继续照搬microbatch4短跑参数。

### Megatron 可选 Liger TP fused CE，整步吞吐增益仍不确定

稳定ID：`github:verl-project/verl:pr:7945`；[PR #7945](https://github.com/verl-project/verl/pull/7945)，main，2026-10-15T06:31:02Z，merge `5b550b0c3dad8c7bb7aab834a3fe195f8cdfc517`。

use_fused_kernels && use_liger启用Liger公共TP scaled CE API，默认仍Triton；Liger负责chunk/padding/native eligibility和fallback，固定liger-kernel0.8.4，native LCK另装。构造共同capacity reservation并让所有PP/VPP ranks参与bootstrap，SP gather的backward按Liger已有梯度归约调整。native Liger不支持DeepEPv1 RDMA；后续context必须适配初始reservation，执行按进程序列化。

H200与B300：作者54 verl focused+89 upstream configuration contracts通过，8GPU合成4层Qwen2、TP2/PP2/DP2、SP/BF16的2optimizer-step SFT可保存checkpoint；这些native源代码/本地binary环境仍torch2.9.1，非新默认torch2.13 released-wheel环境。

历史长response2048、四seed、每backend100step（B300Qwen3-4B TP1/H20014B TP4/H2001.7B TP1）共2400steps，**三个actor token吞吐与whole-step gain的95%CI均含0**；不能把isolated output-head45–65%提升写成RL吞吐。发布的Python wheel曾遗漏experimental包，其安装/执行与新Bridge栈仍未核，PR合入不消除此证据缺口。

建议：选它首先为可切换output-head后端；上线前验证已发布两类wheel、实际native dispatch而非fallback、capacity/同进程actor-ref与checkpoint。

### Ascend fused CE 提供 CANN 路径与熵梯度混合 backward

稳定ID：`github:verl-project/verl:pr:8143`；[PR #8143](https://github.com/verl-project/verl/pull/8143)，main，2026-10-15T04:44:06Z，merge `843dc70fd4a1598616faf68a8fee8c76888d476d`。

NPU可调用CANN/torch_npu vocab-parallel fused linear CE，避免910B CUDA Triton路径UB溢出；dentropy=0用CANN CE，非零用chunked PyTorch CE+entropy backward，因此不因entropy_coeff非零强制禁fused。默认不materialize full logits；另把THD/BSHD序列切片contiguous，规避部分NPU jagged NestedTensor引用整packed storage导致unbind尺寸错。

已读实现与CPU/NPU测试文件，正文列910B数值/最后vocab id/hybrid梯度测试命令及训练修复，但**未列明确执行通过数量、完整端到端配置或吞吐/显存A/B**。不将勾选项/测试存在写成910B训练性能已证。

建议：实际检查CANN API availability与backend，覆盖非零entropy_coeff、vocab shard最后id及真实actor update；收益仍需本地测。

### FSDP checkpoint CPU offload 不再只看 CUDA

稳定ID：`github:verl-project/verl:pr:8232`；[PR #8232](https://github.com/verl-project/verl/pull/8232)，main，2026-10-15T04:46:12Z，merge `879ff5771c57c23d8d419fc2bdd599c0feb148bc`。

四处save/load sharded state-dict gather改查平台is_device_available，让非CUDA accelerator同样offload到CPU。

2×ArcProB60/XPU、torch2.11/xccl、8层4096wide模型一次optimizer后同样sharded gather：tensor由xpu变cpu，peak device memory2816.8→2560.8MiB（-256.1）。这是独立FSDP机制probe，**未跑verl checkpoint manager端到端save/load roundtrip**，不能声称所有后端checkpoint恢复已验证；会增加host memory压力。建议重测真实model+optimizer checkpoint峰值和恢复，评估host RAM余量。

### 128K coding-agent RL 支持活会话跨权重更新与分组重试

稳定ID：`github:THUDM/slime:pr:2474`；[PR #2474](https://github.com/THUDM/slime/pull/2474)，main，2026-10-15T15:02:43Z，merge `2f4280e3b0e6934b954db93153bd79d9c9312830`。

Qwen3.8-27B/MiMo code split的fully async样例：默认3training+3inference GPU nodes（各8GPU）+2sandbox nodes，训练TP4/PP3/CP2，12个TP2 inference engines，context131072/output8192；README称H10080GB已exercise该128K配置。

实际diff显示 admission pause保留inflight agent task；partial-rollout让活sandbox/CLI续model turn。已关闭的segmented session若ABORTED/PENDING，重试原始input而非仅resume token；完成的siblings保留，zip(strict)防错配。compaction在70%context触发，每trajectory最多两次；terminal sandbox outcomes以零reward训练。另含nativeTP/CP GDN、packed/loss memory及top-p跨CUDA stream metadata lifetime修复。

作者pre-commit、640CPU tests/1skip及launcher语法验证；没有完整200updates学习曲线、稳定吞吐、成本或fresh-run baseline。75files是大改，本轮只精读上述生命周期/README/top-p patch，不声称所有feature逐行审完。建议先验权重版本跨度、compaction masks与logprobs、terminal奖励分布，再扩并发。

### PP 输出交换按 rank 奇偶排序，消除 send-first 环等待

稳定ID：`github:sgl-project/sglang:pr:44453`；[PR #44453](https://github.com/sgl-project/sglang/pull/44453)，main，2026-10-15T17:46:38Z，merge `c5a5bf0d0bd3ff2d5d08bda41aa9273feea98584`。

CUDA isend只有CPU调用异步；同pair的NCCL P2P按同stream排队，在buffer满时send等peer recv。两个microbatch inflight的PP2 send-first顺序构成环等待；改even send→recv、odd recv→send，NPU PP2 batch路径不变。

作者2×B300/NCCL2.30.7隔离probe：强制NET/IB loopback时main K≥3消息hang，parity测至64无hang。upstream Qwen3-0.6B PP2在一节点IB loopback、并发错开logprob请求，main3/3fresh launches前3round内hang，修后72/72round完成；Qwen3-8B greedy logprobs逐bit同，性能差在run noise内。

原真实跨节点PD-prefill修复发生在fork；**上游main没有再跑真正多节点、PD event loops或PP>2**。建议PP验收覆盖staggered microbatch和小buffer网络，而不只单node NVLink与单请求。

### DSA top-k v2 溢出桶改 exact refinement，CUDA cluster 缺口仍在

稳定ID：`github:sgl-project/sglang:pr:37941`；[PR #37941](https://github.com/sgl-project/sglang/pull/37941)，main，2026-10-15T07:52:49Z，merge `86f4ede4fbc3543ea431bb7caccd7069dde7430b`。

coarse fp16 histogram threshold-bin超过2048候选后旧实现按先到者截断，可能选错top-k。新12/12/8 exact-key radix rounds与最终scan缩桶；signed-zero边界按FP32修正；ROCm long-row split也读uncapped count再全行refine。

MI355X/gfx950 GLM5.2 decode观测约0.04%rows进入该路径、最大bin4139；330cases在作者merged tree通过，旧10 overflowing非identical cases全失败。五paired runs的普通randn8192/32768行延迟约-1.7/-0.7%，但必须精确处理的cold two-values cases慢203–224%；这是修正错误答案的成本，非通用加速。

**CUDA-only TopKCluster仍截断**，作者无NVIDIA硬件，CUDA长于24576的测试跳过，不能写所有DSA top-k精度已修。同日 [#37625](https://github.com/sgl-project/sglang/pull/37625)（13:44:59Z）修legacy AOT与JITv1 stash overflow，SM120 v1当前15tests通过、旧9fail；完整AOT矩阵/ROCm未跑，无速度基准。建议按实际selector路径造overflow/ties/相邻FP32/±0负对照。

### HiCache 新 page_unified 与拓扑无关 L3 chunk key 尚是库能力

稳定ID：`github:sgl-project/sglang:pr:39727`；[PR #39727](https://github.com/sgl-project/sglang/pull/39727)，main，2026-10-15T07:46:14Z，merge `9ea5472833296e97724f12df716936576d34ddbb`。

MHA页按page/head_group/layer/KV/page_size/head/dim布局，让一个layer×head矩形含K/V、可用单offset+length表达；key移除writer TP/PP/CP rank后缀，改page_hash/digest/layer-range/head-group，以内容坐标命名chunk。fleet-wide head-group/layer-partition必须一致，缺省仅同拓扑共享，可能静默0hit。

23CPUcases核stride、恰好平铺、ragged61layers/30partition尾段、TP2=两个TP4对象并集/PP并集和identity字段digest；B200仅pool shape allocation smoke。**没有server CLI flag、transfer arm，用户当前不能直接开启**；无L3端到端共享命中或性能数据。建议作为跨拓扑缓存schema基础跟踪，等待集成后用真实混合TP/PP fleet验证。

### DSv4 mHC HcPost 修正分支归约轴，旧checkpoint语义有差异

稳定ID：`github:pytorch/torchtitan:pr:5417`；[PR #5417](https://github.com/pytorch/torchtitan/pull/5417)，main，2026-10-15T19:44:06Z，merge `d1991d1e48650ee0c92f0474b2ce0bd67c69463e`。

从参考[B,S,hc,hc,D]迁至[T,hc,hc,D]时仍用dim2，导致每支只按comb row-sum缩放而不混合其他residual。改dim1，实现comb转置乘residual。

两个CPU测试用非对称comb对原参考与闭式矩阵形式，新版通过、旧版每元素不符；无全模型训跑或准确率。参数shape不变，旧checkpoint仍能load，**但此前训练并未实现该混合**，load成功不能当语义等价。建议标注checkpoint训练revision，再决定续训/重训对照。

### TitanRL 四项数据通路改动需分开看传输、GIL 与默认行为

稳定ID：`github:pytorch/torchtitan:pr:5285`；[PR #5285](https://github.com/pytorch/torchtitan/pull/5285)，main，2026-10-15T05:58:43Z，merge `bdb253e827a79bb8fd02eeeaa33d4ea83239851b`。

[#5285](https://github.com/pytorch/torchtitan/pull/5285) 为async_loop.batcher.rdma_min_tensor_bytes加opt-in：大tensor保存在batcher，以MonarchRDMABuffer引用传输，trainer读自己DP列；默认None按值RPC。单host/TCP/4process、5.2GBstep/DP2×TP2/614Btoken旧benchmark 9.1–11.1→3.3–5.1s，主要来自只读自己的列，不全是RDMA网络收益；GPU H100/Qwen0.6B/5step仅证loss parity。

[#5286](https://github.com/pytorch/torchtitan/pull/5286)（06:05:12Z）read-ahead2个microbatch+默认false的lazy preprocessing；需同时显式启用，拒绝CUDA graphs/SDC replay/GraphTrainer。H100小payload下整体forward_backward差在噪声，lazy每microbatch约4ms成本；不能说开预取必加速。当前GPU数字来自引入opt-in之前head，后续stack DP2/5step TCP及ibverbs有parity。

[#5291](https://github.com/pytorch/torchtitan/pull/5291)（06:08:21Z）把build/pack搬独立BatcherActor进程。CPU合成event-loop100%busy旧baseline3.7–5.4→fast path0.61–0.70s；idle反而多约0.1s，baseline后来to_thread未重测，10node诊断负载非此PR现head规模复现。配置breaking：async_loop.training_sample_builder→async_loop.batcher.training_sample_builder；Monarch≥0.7是fast-path条件；最多3registered batches可高占host RAM，exclusive GPU0 context竞争仍有限制。

[#5309](https://github.com/pytorch/torchtitan/pull/5309)（06:10:13Z）默认按值路径也只向每DP submesh发其列，setup核rank布局；旧CPU2.6GB/DP4step5.3→2.1s/每rank7.6→1.9GB，不应再把同一column收益归给RDMA。CP×TP ranks仍各自重复读同列，多个host未正式验证。

建议：先升级配置与Monarch，按真实payload分别测pack、handle消息、read、训练等待和host峰值；不能用四个局部速度相乘估端到端。

### 首个 WGRAD 直接产生 grad_dtype，避免先 BF16 舍入

稳定ID：`github:pytorch/torchtitan:pr:5399`；[PR #5399](https://github.com/pytorch/torchtitan/pull/5399)，main，2026-10-15T14:40:08Z，merge `b1be9efcfa90193f6264878d79fdb1caac81ff78`。

FSDP2 gradient accumulation首microbatch、sync disabled且尚无running grad时，Linear曾先BF16 mm再autograd upcast FP32；现在weight.grad_dtype较宽时用torch.mm(out_dtype=grad_dtype)，去一次cast/全尺寸temporary，保留FP32 accumulator输出。无FSDP/每microbatch同步路径不变。

新GPU测试比较FP32输出与mm(out_dtype=FP32)逐bit，旧实现失败；未列吞吐/显存A/B，且这一特定FSDP FP32 reduce+accumulation数值会与旧版不同。建议更新golden前确认变化源自更高精度首wgrad，不要简单归为随机漂移。

短跟踪：[spmd_types #5398](https://github.com/pytorch/torchtitan/pull/5398)（21:43:45Z）module边界forward redistribute启async_op、首次consumer stream wait，backward仍wait，无实测吞吐；[GraphPP #5208](https://github.com/pytorch/torchtitan/pull/5208)（18:15:10Z）每stage/backward-path复用persistent accumulators，REDUCE_GRAD仅消费最终handles，作者90focused tests，无整训性能。

### 旧待办的证据补充（非本窗首次发布）

以下PR上轮只留清单/待精读，本轮重新打开正文与patch补证，日期仍归原始合入，不包装成10-16新发布。

- **vLLM在线MoE再量化API**：稳定ID `github:vllm-project/vllm:pr:60276`，[#60276](https://github.com/vllm-project/vllm/pull/60276)，main，10-14T07:49:38Z，c9bfa574ce690e45d1da5d506d17ed1586dcf9dd。为CT/ModelOpt/Quark NVFP4 MoE实现dequantize_weight→online quant method，共用Linear lifecycle，包含padding与reload。H100在259ffd head42passed/1skip、MI35045passed；本轮没有打开独立E2E结果评论，不能把body“见下文”当质量结论。仍先load原格式全部weights再再量化，可能高峰内存；源format还要求可运行backend，未实现dequantize_weight到loading时才失败，NVFP4 per-token目标未测。建议先核峰值/目标method/重载，不把“any online”当所有源目标组合已认证。
- **CT MXFP4 checkpoints兑现FP8激活声明**：稳定ID `github:vllm-project/vllm:pr:59815`，[#59815](https://github.com/vllm-project/vllm/pull/59815)，main，10-14T20:33:37Z，984c242ede0beaf1c3ba9eb4c01b98bed8973eac。新MoE method继承native MXFP4权重/backend/EPLB机制，将checkpoint activation key传Humming；dense也区分W4A8而非错误W4A4。依赖[#61556](https://github.com/vllm-project/vllm/pull/61556)已核10-14T20:32:37Z合入main，故不是仍待合入的依赖。H200上Qwen3.6真实weights及GLM5.3五层dummy/TP1/2/EP核layer选择；QwenGSM8K1319题W4A16 0.958/W4A8 0.961是单次，不是质量提升证明。**GLM大GEMM M≥256 microkernel W4A8快1.4–1.8×，小Qwen1×H200整服务却慢3–8%**，activation quant成本未摊销。BlackwellMoE仍忽略该key走MXFP8/BF16，非CUDA dense保持旧MXFP4 activation路径并warn。建议优先兑现量化语义，再按真实shape选择，不能写跨硬件普遍提速。
- **GPT-OSS batch-sharded sampling补hook**：稳定ID `github:vllm-project/vllm:pr:61257`，[#61257](https://github.com/vllm-project/vllm/pull/61257)，main，10-14T17:11:11Z，2c74fd1873e36a1ee76d6b9dc8ef3c5e5ef80092。实际production只加compute_logits_local/skip_gather，否则V2会静默回regular sampling。2×L20/TP2，tiny GPT-OSS本轮正文12tests、10独立进程稳定性以及Llama empty-rank控制通过；完整pre-commit与上游GPU CI仍未完。旧benchmark GPT-OSS20B/c256/256in128out/graphs/3fresh servers：输出tok/s+4.49%或+5.97%，top_p1/.95，review仅改test未重跑；不外推Llama。建议核rank-local vocab宽度与实际启用状态，shared LoRA另案。
- **SGLang DeepSelect长行与bf16 index logits**：稳定ID `github:sgl-project/sglang:pr:44051`，[#44051](https://github.com/sgl-project/sglang/pull/44051)，main，10-15T01:00:20Z（比本主窗早55分钟），9d68ecce2e811d908551d58a2a8792c75245d181。bf16/topk≤1024 cluster按行数/occupancy选，不按graph padded width，内部按row end；index flag `SGLANG_OPT_DSV41_BF16_INDEX_LOGITS`默认off。B200 cold-L2/width1M/topk1024，1row/end128K 19.0→6.1µs，32row/end1M 82.8→22.3µs；2×GB300/DSv4.1Flash index层128K topk51.1→27.6µs，非整个服务吞吐。GSM8K on0.893/off0.913或rerun0.897、有125K/209K passkey，不能据数题噪声说长文质量全等。配套稳定ID `github:sgl-project/sglang:pr:44210`，[#44210](https://github.com/sgl-project/sglang/pull/44210)，10-14T09:31:20Z，修SM90 score row需128B alignment导致启动退出；8×H200 cookbook recipe可启动/服务，未给吞吐。建议将dtype+row stride+padding storage一起核验，不仅看数学shape。

### 其他工程短讯与尚未启用的能力

- **vLLM Mamba align gather融合**：稳定ID `github:vllm-project/vllm:pr:58737`，[#58737](https://github.com/vllm-project/vllm/pull/58737)，main，10-15T16:02:54Z。MI355X隔离同process1request/7specblocks，7→1kernels，10.81→1.78µs（6.06×）；55tests含noncontiguous stride、zero-row、empty-batch、64bit row offset等。作者明确**无可测服务吞吐提升**，不能以标题“6×”写E2E。
- **SGLang graph-tail routing边界**：稳定ID `github:sgl-project/sglang:pr:44347`，[#44347](https://github.com/sgl-project/sglang/pull/44347)，main，10-15T04:14:29Z。persistent tail scalar把MoE GPU row count与实际SWA replay tail绑定并保持idleDP0，退出恢复refs。8graph checks来自优化分支相同方法AST，不是此PR独立完整E2E；没有独立吞吐收益。建议测tail/idleDP组合。
- **SGLang MegaAttention集成**：稳定ID `github:sgl-project/sglang:pr:43922`，[#43922](https://github.com/sgl-project/sglang/pull/43922)，main，10-16T00:21:44Z。已读patch新增backend/pack-query/cache path，并对flashmla_mega断言DSv4.1/SM100或SM103；正文只有“Integrate MegaAttention”和CI状态，不附准确率/速度条件。仅作代码可用性跟踪，不据Perf标题宣称提速。
- **vLLM UMBP第一片**：稳定ID `github:vllm-project/vllm:pr:61789`，[#61789](https://github.com/vllm-project/vllm/pull/61789)，main，10-15T15:12:40Z。本轮只读body：runtime-neutral对象/transfer plan/connector facade与ownership lifecycle；明确不注册MORI backend、contract tests在后PR。不能写完整新分布式缓存/传输后端上线。
- Megatron #7728/#7522/#7803/#7789、vLLM #61074/#61647/#61643/#59468/#58668、SGLang #43502/#41303/#44173/#44351、torchtitan #5420等已打开正文但未全部精读patch或提取适用边界；保留原文供后续，不据标题发布技术结论。Muse模型可能由model方向并入，核心方向不重复。

### vLLM两个指定跟踪与更正

1. **#61530仍未合入**：稳定ID `github:vllm-project/vllm:pr:61530`，[MXFP4 Engram PR](https://github.com/vllm-project/vllm/pull/61530)，state=open、merged=false、merged_at=null；head仍1800929afdd3728860f794bcf69ae7beda95649d，updated_at10-15T08:57:00Z。没有新增已发布能力，旧host-memory/quality数据不重复当新进展。
2. **更正 #61532资源类型**：稳定ID `github:vllm-project/vllm:issue:61532`，[consumer abort-ack timeout issue](https://github.com/vllm-project/vllm/issues/61532) 是issue，**不是PR**；此前日报若称“consumer PR #61532”须更正。当前open，updated_at仍10-14T10:08:46Z，2comments均10-14，无新合入状态。评论说明producer #59790移除timed drain fallback，但consumer 10s abort-ack expiry与close仍待解。
3. 本轮重新取截止前默认分支 `a956a8126e6bd184cb0a6b8f7318fe4717675c2f`（committer10-16T01:00:17Z）的[client.py](https://github.com/vllm-project/vllm/blob/a956a8126e6bd184cb0a6b8f7318fe4717675c2f/vllm/v1/kv_offload/tiering/p2p/session/client.py)及[server.py](https://github.com/vllm-project/vllm/blob/a956a8126e6bd184cb0a6b8f7318fe4717675c2f/vllm/v1/kv_offload/tiering/p2p/session/server.py)：consumer仍在aborted_at超过_ABORT_ACK_TIMEOUT_S后报failed；server撤供集合仍用set(timed_out)。上轮pending-job组合兼容性疑点仍仅静态未复现，不能宣布新增bug已证或P2P所有abort路径安全。新增lookup timeout/control thread不等于远程WRITE终止凭据。继续保留pin liveness、pending withdrawal composition、close/timeout与实际NIXL transport终止保证待办。


<a name="extended"></a>

## 训练与推理工程：扩展项目

### TRL 1.16：AsyncGRPO 的树形复用进入版本发布，收益取决于真实分叉与共享比例

**项目/日期/版本。** Hugging Face TRL `v1.16.0` 于 `10-15T18:00:33Z` 发布。树形打包 [#7328](https://github.com/huggingface/trl/pull/7328) 在10-13已合入 `ea4d31033e216377bdf66576d426e48187d3eb06`，属于回溯补录与本轮正式发版；AMD路径 [#7689](https://github.com/huggingface/trl/pull/7689) 于10-15合入。不是所有 trainer 同时换成树形训练。

**具体变化。** AsyncGRPO 将多轮、多生成的相同前缀组成树；无分叉链压成 segment，每段查询通过一次 varlen attention 访问自己的祖先K/V。非attention层只算唯一token，不构造完整稠密树mask。树节点以 `(token, next_token)` 为键，在分叉前复制一个节点，避免共享位置对应不同监督标签；`token_index` 将logprob映射回各原始rollout，保留独立 advantage、old logprob 与训练目标。行预算同时约束唯一前向token与loss项，防止前向去重后无限塞目标。已实际读 `packing.py`、trainer与配置patch。

**证据。** 作者在单H100、BF16、gradient checkpointing、四条各32K rollout、相同chunked LM head下，对比Qwen3-1.7B的FA3序列打包：共享比例3.37时前反向8179→2755ms、峰值35.2→13.6GiB；无共享时8677→8737ms，略慢约1%。旧Flex实现的SWE-Gym端到端数字不是最终FA3路径的端到端实测，不能用“FA3微测更快”推导其端到端下限。默认 `fork_threshold_tokens=1024` 的真实共享比例可能只有1.45，调低分叉门槛的案例才到4.21。MoE辅助均衡loss按唯一前向token统计，也不宜把“原始rollout目标保留”扩大成所有辅助项逐项完全相同。

**AMD增量。** #7689以AITER v2替代ROCm上不可用的FA3。作者2×MI300X、ROCm7.2、PyTorch2.13、vLLM0.30，Qwen2.5-1.5B、GSM8K、16prompt×8生成、512-token输出、200步：同16K预算下，训练token/步11.7K→16.2K，前反向token/s20.7K→54.2K，**墙钟token/s只增29%**（7.9K→10.1K），不是整任务2.6倍。该实验一卡rollout、一卡训练，步均处理量发生变化；不能当等优化步、等学习token的能力对照。

**限制与建议：考虑复现。** 树路径拒绝线性注意力/卷积（Qwen3.5等）与attention sink（GPT-OSS）；AMD还不支持softcap，Gemma2对应测试跳过。tiny VLM parity只覆盖text输入，不能外推图像树训练。用自有agent轨迹先测共享比例、有效训练token/秒与reward/token，再核原始rollout loss及梯度；MFU可能下降，因为它按被消掉的前向token计数。

来源：[release](https://github.com/huggingface/trl/releases/tag/v1.16.0)、[#7328正文与代码](https://github.com/huggingface/trl/pull/7328)、[#7689及实测](https://github.com/huggingface/trl/pull/7689)。数值均为作者测试，本轮未做GPU复现。

### Transformers 5.20：FSDP×TP检查点可换网格恢复，packed混合模型回退开始拒绝静默串样本

**日期/版本。** `v5.20.0` 于 `10-15T18:05:36Z` 发布，原生Muse Spark Open接入 [#49644](https://github.com/huggingface/transformers/pull/49644) 于17:35Z合入；本条重点是保存恢复和正确性。不能仅从模型注册推断高性能后端、全部多模态路径均已验证。

**模型与优化器检查点。** [#48802](https://github.com/huggingface/transformers/pull/48802) / [#49256](https://github.com/huggingface/transformers/pull/49256)，分别合入 `89d1c8802ecdca5443c619cc143c191f4dfd882a` / `e62ee803f601cdc112ecc64cb6b685d59cbbbf88`。模型可由所有rank调用 `save_pretrained(..., distributed_checkpoint=True)` 写rank-local safetensors/DCP，按不同TP/FSDP mesh重载；另有单进程离线consolidation。针对 `_StridedShard` 非连续区域的DCP元数据，先重分布成连续Shard再存取，加载后还原placement，因此有额外通信与临时显存。优化器按参数名flatten状态，DTensor与普通Tensor分组，以适配换网格和foreach。作者四进程TinyMixtral示例先FSDP2×TP2训练、再TP4恢复继续过拟合并校验greedy输出；这是具体可复现烟测，不能代表任意优化器、大模型与EP组合都认证。

**静默跨样本污染。** [#49474](https://github.com/huggingface/transformers/pull/49474)（10-15）指出：`causal_conv1d_fn`、`mamba_chunk_scan_combined`、`chunk_gated_delta_rule` 的torch参考实现不接受 `seq_idx` / `cu_seqlens`，旧fallback曾丢掉这些边界，导致packing后不同样本状态相连。新行为是检测到边界参数便抛错并指明需安装的包；**并没有实现一个正确支持packing的纯torch回退**。即使参数只标识单序列也拒绝，普通不带边界的padded batch仍可回退。适合优先核查混合架构训练环境是否真加载到原生kernel，不能用loss下降证明数据隔离正确。

**EP兼容性回归。** [#49633](https://github.com/huggingface/transformers/pull/49633)（10-15）恢复旧 `enable_expert_parallel=True` 的“单独应用EP计划”行为；5.19把它当作 `ep_size=tp_size`，同时套TP计划，KV头少于TP数的模型会被错误切attention而崩溃。显式新并行配置与旧flag语义需区分。建议升级时做一次不同mesh保存/恢复和packed/unpacked隔离回归；没有本轮性能倍率可报。

来源：[release](https://github.com/huggingface/transformers/releases/tag/v5.20.0)、上述五个PR正文和关键patch；实际读取 `distributed/checkpoint.py`、`hub_kernels.py` 与并行配置改动。

### Muse Spark Open 的后训起点：TRL提供64张H100 LoRA配方，5.20已包含保存修复

TRL [#7536](https://github.com/huggingface/trl/pull/7536) 于 `10-15T17:34:22Z` 合入，1.16包含tiny模型、VLM训练测试与LoRA SFT示例。作者报告8节点×8H100，`fsdp_size=64, ep_size=8`、token dispatch与packing，131K token/步、14.2秒/步、每GPU40.2GB。它提供816B模型低成本适配的具体入口，但不能把LoRA显存写成全参数训练预算；tiny测试覆盖也不等于对816B全部路径跑完整训练。

要求Transformers≥5.20.0。交叉核验确认适配器保存修复[#49500](https://github.com/huggingface/transformers/pull/49500)已于10-14合入，commit `f0705f991a71f05dad9569336b181119f5546f93`是5.20.0祖先；PR旧备注不能继续当未合入依赖。PEFT0.21.2是该修复验证环境，不据此称硬性最低版本。2×H100上的tiny Qwen/Llama保存回归不能替代816B整流程，建议按示例先做保存—重载—生成等价性验证。来源：[#7536正文及 `examples/sft_muse_spark_open` patch](https://github.com/huggingface/trl/pull/7536)、[1.16 release](https://github.com/huggingface/trl/releases/tag/v1.16.0)。

### TRL的VLM生成修复：默认屏蔽图像占位符，覆盖范围限Transformers采样

[#7695](https://github.com/huggingface/trl/pull/7695) 于10-15合入。GRPO/RLOO/Distillation在模型生成 `<image>` 等占位token时，后续拼接前向会多出没有视觉特征对应的图像位置而报错；现在从**模型config**取placeholder ID放入默认 `suppress_tokens`，避免某些processor的同名ID其实不是占位符。已读三个trainer的patch。

用户 `generation_kwargs` 可覆盖此默认；vLLM生成路径不在该PR覆盖范围，Distillation对vLLM的参数传递也保持原样。建议VLM在线训练同时检查rollout后端和配置覆盖，不把升级TRL写成所有后端自动解决。另[#7719](https://github.com/huggingface/trl/pull/7719)保存forward中计算的stride，修torch2.6在autograd backward里读取stride导致的fullgraph编译失败；它与昨日记录的静默loss/NaN错编译修复是两个问题。

### TensorRT-LLM：调度线程避免等待KV接收空间，长上下文索引修补32位乘法溢出

以下均为**main已合入**，未据此断言已经进入正式稳定release。

- **[#19940](https://github.com/NVIDIA/TensorRT-LLM/pull/19940)**，10-15，合入commit `097d2adf4aad3997cb83199b6fa377e49f6de422`。generation接收KV时，在executor线程申请bounce region曾每请求最多等0.2秒，空间不足后仍走分片fallback，导致整次decode及ADP其他rank一起停顿。接收端改零等待，发送worker线程保留0.2秒等待。作者GLM5.2-NVFP4、3 context DEP4+2 generation DEP4、576并发的多轮负载，p99 decode step约146–147→63–67ms，p25用户输出88.4→99.2token/s；GPU型号未在已读实验段给出，完整prefix接收占比51%→44%，不能据单组流量得出通用55%尾时延改善。已读实际timeout改动及回归测试。**建议考虑采用**：排查异步服务尾延迟时，追到admission线程上的内存等待点，而非只看传输kernel。
- **[#20190](https://github.com/NVIDIA/TensorRT-LLM/pull/20190)**，10-15，`e30418f3432bdf1269972f65234f84b429f83494`。Indexer K-cache gather/scatter虽把offset存为int64，`token_idx * head_dim`仍先做32位乘法；head_dim64、row≥2的25次方就溢出，可能写到buffer前方且不报错。现在乘法前cast int64，payload与scale两边均修。触发案例是同一步超过32条1M context请求；不是单条32M上下文限制。作者GB300验证gather至268,439,040 /134,219,520 rows（47项），因需要>2GiB cache未新增大规模单测，scatter未给同规模独立认证。**建议优先回归**自有大batch×长context与边界索引，不能因小规模用例通过排除风险。

### TensorRT-LLM GLM-5.3-Flash：只给可见pool打分，算子倍率没有等额传到服务吞吐

[#20251](https://github.com/NVIDIA/TensorRT-LLM/pull/20251) 于10-15合入 `172b96d051b62b39c1c80d27cbe189f72024a82f`。旧prefill按最大context容量启动pool scorer，13K实际输入/max131K时约90% program仅写最小值占位。改用host已有长度限制grid，BF16输入保留BF16 tensor-core乘法、FP32累加，16行复用key时用2warps；decode/MTP仍保持固定graph grid。已读backend、Triton kernel、model调用patch。未写的不可见列必须由top-k读取范围保证隔离。

作者4×B200、TP4/EP4/MTP3、BF16 KV、ISL13312/OSL33、chunked prefill、maxnumtokens16384，源基线`eaefddcecd`，NGC1.3.0rc25+本地main build：单层scorer13K时3.85→0.94ms（4.1倍），C1 TTFT431.5→393.9ms（-8.7%）；C8吞吐80.0→86.5token/s（+8.1%），TTFT1578.6→1324.7ms。**显存分配未减少**：仍保留原score buffer，decode step约+0.2%。BF16累加顺序在极接近top-k边界可换一个pool，不能称逐bit全模型等价。建议**考虑复现**长输入服务，按TTFT、并发和top-k质量验证，避免宣传“服务快4–7倍”。

### TensorRT-LLM DeepSeek-V4：删掉重复归约，并使MoE输出走MNNVL

[#20233](https://github.com/NVIDIA/TensorRT-LLM/pull/20233) 于 `10-15T23:52:13Z` 合入 `bf0e298f11ec7fff43b86fc559cf6f64181323c8`。GB300、DeepSeek-V4-Pro FP4、跨两节点TEP8、gen-only下，换到含NCCL2.31.2的PyTorch26.09后，device step14.68→17.39ms。作者定位57KB小消息低于symmetric注册门槛，NCCL选择从约14.8µs变为32.9µs。新代码去掉attention输出已完成TP归约后的重复pre-MoE allreduce，并为MoE输出AllReduce传入dtype，使AUTO可选MNNVL。

已读模型patch和原作者定位实验，不能把它简化成“升级NCCL必慢”或全模型禁用NCCL。变化略改RMSNorm eps相对尺度；作者另报V4 Flash TEP4+MTP GSM8K95.30/旧95.07、acceptance1.9584/1.9581，是特定精度烟测，不是V4-Pro所有任务认证。建议**优先阅读机制、按配置回归**；在通信trace中确认实际kernel与重复collective，比只锁库版本更有价值。

### TensorRT-LLM Rubin回合并修复：KV重叠调度与量化缓存仍有明确禁用路径

[#19959](https://github.com/NVIDIA/TensorRT-LLM/pull/19959)，10-15 main `ff7ea92dfb75fb383605dfcd78f68fbe1e06af4b`，在`823851e1af`基础上合并多个正确性修补。重要问题是overlap scheduler先分配N+1步，覆盖N步的draft width，N步回退时可能释放N+1仍写的页；改按每次allocation保存宽度。另补padding input-id清零、`compressor_postprocess_scatter`声明scale mutation、SM107派发guard等。这里根据PR列明的机制做筛选，未逐文件审计整个复合PR。

**不能扩张成Rubin全能力开放。** PR明确主线仍拒绝FP4 MLA disaggregated serving，Helix全链路Python/NIXL集成是后续；某些PP事务dummy保护也未启用。建议**深入阅读并追踪**动态draft length、CUDA graph padding与缓存页寿命的组合用例，本轮没有全链路稳定性验证。

### vLLM Ascend：修复量化共享专家gate与Mooncake页合并，bounded replay仍待NPU验证

**正确性优先。** 10-15 [#18894](https://github.com/vllm-project/vllm-ascend/pull/18894)，`631569ce6952432f3bbe8b4d5d08e66835127b47`：融合W4A8MXFP及W8A8/W4A8共享专家曾直接返回down projection，遗漏可选 `sigmoid(expert_gate(hidden_states))`；现在两路径共用gate后处理。已读 `shared_experts.py` patch；作者24组合等回归已编写，但本地repository pytest/NPU未执行，不能写成NPU认证。**建议优先检查**含shared-expert output gate的量化模型输出。

**KV布局契约。** 10-15 [#18747](https://github.com/vllm-project/vllm-ascend/pull/18747)，`c5abc52f0c5e68b297779c722dedd940b07ef2ee`：相同共享物理页的tensor view可能只占部分页，过去仅据storage容量判断可merge，P/D两端剩余容量不同会发布不同descriptor数量。现在要求每block内部连续、各byte range并集完整覆盖page stride，空洞/尾padding/部分页退回逐tensor传输；允许覆盖全页的重叠alias。已读实际range检查。作者已有两节点DSV4.1-W8A8服务（prefillDP4TP4、decodeDP8TP2、DSpark5、并发8、maxoutput2048、thinking off）装helper后GSM8K1319请求无失败、1278正确；这是既有部署上的helper patch，不是全新branch安装验收，旧bypass1279正确的单题差不能解释为能力变化。**建议考虑采用**布局检查，保留端到端传输回归。

**新机制需要观察。** 10-15 [#17496](https://github.com/vllm-project/vllm-ascend/pull/17496) main `a58da927ab5e7875acbf748f10ebb0d01e02cfb8`，DeepSeekV4.1 decoder bounded replay让KV-source最后一层后的eligible prefill只处理末尾SWA窗口；decode与prompt-logprob所需行保留，DP含idle rank协商计数。仅MRV2、`NONE`/`FULL_DECODE_ONLY`，只在eager裁剪，graph步保留全batch；MRV1/PP/PCP/DCP/microbatching/LoRA等禁用并警告。作者测试栏仍为“Waiting to be tested”，**已合入不等于验证通过**，不报加速倍率。

**算子后端切换。** [#17354](https://github.com/vllm-project/vllm-ascend/pull/17354) 10-15 main `7a50474d372e5aa04231cfeaab6a089daf179826`，DSV4 FP8/TurboQuant共用CANN `MixedQuantSparseFlashMla`；FP8采用PA_BBND、608B/token（64 BF16 rope、448 FP8 nope、7 BF16 scale与padding），并由host已有长度避免hot-path `.item()`同步。这是具体算子集成，未披露本PR独立benchmark/完整测试结果，不把布局变小直接换算为整模型显存/吞吐收益。

### DeepSpeed CPU Lion：SIMD符号掩码错误会让更新几乎归零

10-15 [#8813](https://github.com/deepspeedai/DeepSpeed/pull/8813) 合入**master** `ca913e430945d0e2325b0d832b947e721f8e8b6a`。CPU Adam/Adagrad/Lion转为纯CPU C++构建，消除虽然没有CUDA代码却链接CUDA库、检查toolkit的遗留依赖；另修CPU Lion AVX2/AVX512的sign mask从`-1.0f`改为`-0.0f`，旧向量元素几乎得不到±lr更新。已读 `cpu_lion.h` 及builder patch和参考测试。

这同时影响CPU optimizer offload的可用性与数值正确性；CPU Adam构建修复不等于Adam更新公式也有同一错误，标量路径也不能等同SIMD路径。**建议优先验证**使用CPU Lion的训练，比较各SIMD与标量/参考更新；没有整步性能实测，不报训练提速。

### Triton与FlashAttention：volatile读取及CUDA graph边界的具体修补

- **Triton [#12268](https://github.com/triton-lang/triton/pull/12268)**，10-15 main `e9cd89dc76451c526e3b25bf43e0ab00d72c2ee8`：layout优化对小load提前允许rematerialize，绕过write-effect检查，使volatile load在多个layout下变成多次读取、可见不同值；现在先检查副作用，必要时保留一次load并convert_layout。已读2个文件patch。作者L4、numwarps4的复现从6次volatile读取降到1次，不一致reader由19725/20000到0/20000；这是有同步/副作用语义的kernel正确性证据，不是任意LLM提速结论。**建议核查**自定义轮询/通信kernel。另[#12304](https://github.com/triton-lang/triton/pull/12304)将BF16扩展纳入FP32加法链成本估计，保留混合FMA类型约束，未给端到端性能倍率。
- **FlashAttention [#2984](https://github.com/Dao-AILab/flash-attention/pull/2984)**，10-15 main `73019affed8c7754c46ce6cbe22ad1d893f76c1d`：paged KV bounds check的`max().item()`会host同步，CUDA graph capture时改跳过；eager检查保留。作者B200/PyTorch2.14.1+cu132四项capture/replay测试通过。**边界转交给调用者**：必须保证每次replay的cache长度+append不超过block-table容量，warmup合法不能保证将来replay合法。已读C++guard、Python说明与测试patch，建议采用时同时审容量契约。
- **FlashAttention [#2965](https://github.com/Dao-AILab/flash-attention/pull/2965)** 10-15只是hd256 backward统一系列第2/5步；64-row KV tile当前藏在测试开关，默认CUBIN与parent相同，hd128测试反而慢。**继续观察**后续启用与完整benchmark，不能从合入标题写“FA4默认hd256加速已发布”。Triton[#12315](https://github.com/triton-lang/triton/pull/12315)作者也明确native FP4/e8m0主要为类型清理/语法糖，本期不作为新低比特计算能力宣传。

### 补录：PyTorch通信与FSDP、Triton混合低比特的旧缺口

**PyTorch NCCL2 symmetric mempool。** 原主线[#200771](https://github.com/pytorch/pytorch/pull/200771)实际commit `36f962c354af080ba16dc3a182f4b4e73386a829`（10-14T01:17:53Z），本轮补完昨日待核项；[#200896](https://github.com/pytorch/pytorch/pull/200896)是10-14的 **release/2.15 backport**，不能当第二个main特性。空pool注册`symm=True`后新增segment过去只做普通ncclCommRegister，现在按pool标志补symmetric window，恢复symk/copy-engine路径。作者2×GB200、NCCL2.30.7有正反例；window注册是collective，所有rank必须同序分配segment，普通all-to-all仍无对应symmetric kernel。已读主线实际commit及allocator hook patch；建议检查真实trace，不凭API参数断言通信走了期望路径。PR本身merged_at为空时，以实际默认分支commit为证据。

**PyTorch FSDP2。** 10-07 [#199729](https://github.com/pytorch/pytorch/pull/199729)，主线 `75bf453412a6e918f01817f653ab0108be840cce`。未被backward用到的预取allgather可能按post-forward子mesh产生结果，下一次forward全mesh复用会copy-out尺寸不符；转回sharded时等待并丢弃pending结果。作者4×B200八项测试；这不是取消仍在进行的通信，等待完成后才丢引用。已读 `_to_sharded` / `_wait_all_gather` patch，作为**补录**建议相关prefetch+不同post-forward mesh用例回归。

**Triton两条不能混称“FP4加速”。** 10-06 [#12135](https://github.com/triton-lang/triton/pull/12135)，`4daa27c97f93aaca1e74d38d51d23f2ac2cc5908`，在Blackwell matmul中把NVFP4 activation tile经FP32算术转成MXFP8，再乘MXFP4 weights，不写中间HBM；有额外舍入误差，作者明确显著慢于MXFP8×MXFP4，只是正确性支持。10-07 [#12132](https://github.com/triton-lang/triton/pull/12132)，`ac37c71f15e6f67cf4f3b2484a0f43ec513d1171`，针对Hopper非persistent MXFP4按寄存器/共享内存/grid资源设置register budget。H100受测256expert、K2048/N4096 grouped GEMM中，ptxas13.3/13.4比旧版本寄存器121→153导致驻留块减半；cap恢复122寄存器/两块。作者固定PTX与输入的assembler-only对照揭示机制，但32K-row大tile仍有未解决spill约2–3%慢的情况。已读相关代码patch与实验限定，**建议复现自己的shape**，不把合成encoder的恢复当所有MoE场景通过。

### 新披露的旧接口风险：PyTorch只补了文档，错误的collective backward仍存在

10-15主线[#200971](https://github.com/pytorch/pytorch/pull/200971)，commit `152bd17a149b132b186823c3b936d18ddeb31c23`，明确 `torch.distributed.nn.functional` 已弃用接口的反向限制：all_reduce仅SUM/AVG正确，PRODUCT/MAX/MIN直接复用op归约梯度是错的；reduce/reduce_scatter仅SUM，reduce还忽略非dst输出梯度。作者两rank gloo实例PRODUCT(x=2,3)得到(1,1)，按该测试loss应为(6,4)。**本次是警告披露，未改计算行为**；升级到含该commit也不会自动修正训练。

建议**优先审查**自定义分布式loss/归一化/路由中带autograd的非SUM collective，按具体新接口验证梯度。PR提到 `_functional_collectives` 的后续[#200880](https://github.com/pytorch/pytorch/pull/200880)，本轮重新核实该PR仍open、merged=false，base为开发stack分支；全部op也未审计，不能当已发布替换承诺。

## 硬件与软件协同

### Intel Inference Microservices 0.1.0：将Xeon模型与硬件配置打包，当前边界是单模型单主机BF16

**10-14原始公告，回溯补录。** 官方给`intel/inference-xeon-qwen-qwen3-4b:0.1.0`等model-specific镜像，启动时识别CPU代际、分配核数、NUMA及AMX/AVX能力，选择已验证profile并解析vLLM参数，提供OpenAI兼容接口。支持4/5/6代Xeon，当前vLLM BF16；不是新模型或新预训练配方。

model-specific镜像默认禁通用fallback，未匹配profile启动报错；任意HF模型的base镜像默认允许硬件通用profile。`detect-hardware`与`dry-run --format json`可查看实际配置。工具调用和structured output只是vLLM透传、尚非已验证能力；GPU、更多精度/引擎、Helm/Operator均为规划。没有可比较的吞吐/时延实验表，不能据“tuned”推断超越GPU或通用成本优势。建议**继续观察/小规模采用**于已有Xeon容量，锁镜像并保存dry-run配置做基线。

来源：[Intel官方全文](https://www.intel.com/content/www/us/en/newsroom/news/artificial-intelligence/intel-inference-microservices-tuned-llm-processors.html)（Web全文可读，直接HTTP403；两种访问结果分别记账）。

### Muse Spark Open的GB300宣传数字：先补负载协议，再用于容量规划

NVIDIA 10-15 [官方部署文章](https://developer.nvidia.com/blog/scale-agentic-coding-workflows-with-metas-muse-spark-open-on-nvidia-gb300-nvl72/)提供NVFP4、vLLM/SGLang/TRT-LLM与Dynamo部署入口。文中“每GPU最高50K token/s、每用户600 token/s”属于厂商报告；配图caption提4GPU、speculative decoding、acceptance length=6，却未给输入输出长度、并发、框架固定版本与完整复现命令，也未说明两个极值是否同一运行点。**继续观察，不以这些数字作验收或采购依据。**模型机制和权重单列于模型章节，不能把NVFP4名字当质量等价性证明。

<a name="coverage"></a>

## 覆盖、失败、补录与更正

### 本轮实际检查了什么

| 来源类别 | 实际范围 | 覆盖结论 |
|---|---|---|
| 模型与技术报告 | 29个HF组织modified前100/created前30两路共58请求，再增Meta新组织meta-models；官方机构站点、模型卡与4份模型/评测PDF；NVIDIA两份完整技术报告重点章节 | HF指定请求成功不代表所有工件、分支、闭源公告均穷尽；Meta公开卡与门控raw分开记录。新增meta-models纳入后续来源清单。 |
| 核心开源六项目 | Megatron/vLLM/verl/SGLang/slime/torchtitan：回溯+主窗750个默认分支commit，主窗216个所有分支merged PR，54个PR正文、37个PR文件patch列表；release前20或不足20 | 指定清单分页取尽；技术内容选择性审阅。dev与backport没有冒充main，源码存在不等于默认开启/正式发版。 |
| 扩展开源九项目 | PyTorch329、DeepSpeed10、Transformers80、TRL61、OpenRLHF0、Triton35、FlashAttention4、TRT-LLM138、vLLM-Ascend142个commit，共799；主窗merged搜索141项，release各前6 | 分页至不足100，搜索incomplete=false。PyTorch ghstack用真实默认分支commit补足PR状态，未称所有项目代码全部审计。零条只限这些查询。 |
| 旧工程缺口 | PyTorch/Transformers/Triton从10-05T00:00Z至10-12T01:55:47Z补取442/79/69项共590条；核心旧候选再读再量化、W4A8、sampling、DeepSelect等 | 清单已恢复，旧区间仍选择性审阅，不能关闭全区间缺口。10-11至12部分与上轮重叠，590不是590项新工作。 |
| 论文发现与正文 | HF10-12至15共176唯一候选；arXiv cs.CL/LG/AI new三列表1266次出现/928唯一ID、标题宽筛370；14候选全文、11篇详核；6个官方仓库固定SHA静态核 | 选入11篇首发都在72h回溯窗，均明确原始日期；三仓主窗提交不等于论文当日首发。未执行训练或GPU kernel。 |
| 硬件软件六方向 | NVIDIA Technical Blog；AMD ROCm posts；Intel新闻/IIM与Gaudi notes；Ascend公告与插件代码；TPU notes/Cloud Blog/JAX；AWS Neuron官方release与SDK/NxD/NxDI/NKI；共8个硬件相关仓库release检查 | 具体LLM适配/性能证据按正文保留；未扫描全部CUDA/ROCm/CANN/HCCL/XLA/libtpu/NKI组件commit。 |
| 正式会议与开放评测 | PMLR定向论文、TMLR列表/OpenReview两候选、ACL EMNLP事件入口；SemiAnalysis原始动态dashboard和TPU长文 | PMLR旧ICML论文未伪装新发；OpenReview/ACL未取得完整可核增量。InferenceX尚未锁本窗数据提交，不采用二手硬件倍率。 |

模型渠道的具体边界也需要保留：OpenAI/Anthropic当日官方全文已开，分别是Rosalind访问计划和独立评估访问安排，没有把它们扩写成新训练结果；Google/Microsoft/NVIDIA/Meta新页与报告核查较深。DeepSeek/Qwen/Seed/Kimi/Z.ai/MiniMax/Tencent/Baidu/StepFun/Xiaomi/Meituan/Mistral/xAI/Ai2等结合官方站点与HF，仍有动态空壳/分页/门控；IBM新卡与BUILD深入核查，Cohere/TII/Nous主要为HF。未搜到不代表厂商没有发布。

### 失败、恢复和未完成区间

- **arXiv主窗API429**：`lastUpdatedDate`的10-15T01:55→10-16T01:00联合分类请求未取完，三份new列表仅显示10-15。本轮不能声称主窗没有新论文/修订，持久化保留该区间，下一轮优先补查。OpenReview两forum/API403或验证页，ACL指定EMNLP入口404；正式会议日级增量仍不完整。
- **GitHub匿名API403限流已换入口补取**：扩展旧清单、若干PR files用认证connector恢复；论文6仓用只读clone固定截止前SHA。初次失败保留在日志，恢复只关闭相应访问失败，不等于全部技术审阅完成。
- **模型渠道**：Meta门控raw/config/commits仍401/403；MiniMax正确IR已读、研究API却停在旧月份，差异尚未解释；Qwen/Tencent/Baidu/Kimi等动态正文不完整。Seed两路article API200仍有分页边界，LongCat正确ChangeLog已恢复，不能把早先误到用户协议的URL称为changelog。
- **硬件资料**：Gaudi1.24.2具体notes路径404，wrapper release存在但未读实质变更；华为旧公告路径404后已找到正确`/productbulletins`，CANN/HCCL本体仍未扫全。AWS原猜测域路径失败后正确readthedocs首页200；Intel IIM直接HTTP403但Web全文成功。TPU/XLA/libtpu/TorchTPU与NKI/Neuron组件逐commit缺口保留。
- **旧工程区间**：核心10-05T00:00Z→10-08T00:00Z的剩余PR审阅；PyTorch/Transformers/Triton10-05→10-12的剩余审阅；本轮750/799清单未选中的有价值候选仍需按价值继续。TensorRT-LLM Mooncake loader、部分ROCm/CP大PR及更多Ascend算子未全部详核。
- **待续研究**：KV4-RL原10-10、Long-Horizon RL Needs Muon、Continuum Attention及其他保存候选，正式会议新版本、Ai2 Bolmo/Nature全文仍待补。Muon稿现有证据不足以支持标题中的普遍必要性，不从曲线猜指标。Nemotron DSpark fused-scale张量diff、UserRepGen正式论文/代码、GLM-Legal权重状态仍未关闭。
- **未解决的工件边界**：Step5 FP8 raw/历史本轮已可读，只关闭访问缺口；NVFP4独立质量/服务验证和输出上限冲突仍待查。NVIDIA三张Embed/Rerank卡当前可见上传却写未来10-20 ReleaseDate，Bolt E4B标未来10-23，均不当10-15正式模型。Super软件中训卡67B/报告65.4B差异保留。

### 补录与更正

1. 论文11篇均以原始v1日期记账；LatentLoop/Reality Check/ReaDy主窗有代码提交或工件变化，**不能把HF精选日/添加论文链接的commit当论文首发日**。Muse Wonder10-12、Intel IIM10-14，以及旧工程10-06/07/14各项标为补录。
2. **纠正昨日待办的论文ID对应**：VRSD是 `2610.16354`，Jeva是 `2610.18773`；按本轮实际论文标题核正，去重记录采用正确ID。
3. **纠正vLLM #61532资源类型**：[consumer timeout #61532](https://github.com/vllm-project/vllm/issues/61532)是issue，仍open；旧日报“consumer PR”的措辞有误。#61530仍未合入；本轮lookup deadline/control thread不等于远程WRITE寿命问题已修。
4. **修订TPU配置采用建议**：昨日verl #8226的10步性能只支持短跑；本轮#8239给出microbatch4在37步碎片失败、microbatch2+plugin pin完成100步的新证据。保留旧报告的历史观察，用本轮长跑限制修订实践建议，不把跨长度均值直接算成加速。
5. 交叉审阅已纠正草稿中的过期依赖：Muse保存修复TF#49500已包含5.20.0；PEFT0.21.2是验证版本，不能写成必需版本。PyTorch#200896是release/2.15 backport，main真实commit来自#200771，避免重复记特性。

本轮来源清单、请求结果、稳定ID/版本、失败及续查事项保存在本地 `.monitor/research-2026-10-16/`，运行进度与远端同步状态在 `.monitor/state.json` 分开维护。**全局完整覆盖水位继续保持未建立**；本报告是有边界的持续监测结果，不声称全面监控。仅维护指定的日报与索引，没有改写llm-knowledge的wiki。
