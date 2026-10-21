# LLM 技术动态日报 · 2026-10-21

> 时区：Asia/Shanghai（UTC+8）。增量扫描窗口：**2026-10-20 08:55:34 → 2026-10-21 08:56:38**；72小时回溯自 **2026-10-17 08:55:34**。从上次成功归档的扫描位置续查，早期未完成区间见文末；扫描游标不是所有来源均已深审的水位。
>
> 实际生成时间：**2026-10-21T09:13:32+08:00（Asia/Shanghai）**。下文精确事件时间除注明北京时间外均为UTC；首次提交、公告日、权重上传、修订、合入和正式发布分别记录。截止后读取只核实本窗事件与访问状态，不把截止后新事件并入本窗。
>
> 证据来自本轮实际打开的一手全文、模型卡/工件、官方技术文、release与具体PR/commit。所有训练、性能、质量及硬件测试均为作者/厂商披露，本轮未独立运行GPU/NPU/TPU实验。实现精读为选定关键路径，不是全部候选逐行审计。

## 本次重要变化

- **原生PyTorch上TPU有可用包与具体系统设计。** TorchTPU0.2.0公开源码/包，eager、DeferAndFuse、Dynamo编译与分层缓存都有证据；8192-chip弱扩展、默认矩阵精度和RL共置限制需要一起读。llm-d1.0把多租户流控、P/D成对升级与会话路由放入稳定v1控制面。
- **记忆模型值得复现，压缩率不能代替保真。** MemT-1.0权重/报告补录给固定会话状态与逐token更新，128K历史速度/存储有受限实验；文本记忆基线仍有更高任务分数，软件非商用许可与权重许可须分开。Muse Glimmer报告补录则披露预训logit蒸馏→midtraining→OPD/RL和低精度取舍。
- **分布式正确性优先于新kernel倍率。** vLLM Mooncake多peer失败接收、LoRA level2唤醒、SGLang EPLB/量化scale、Megatron冻结MFSDP/checkpoint、Ascend持久KV池都有可定位改动。旧P2P consumer late-WRITE待办仍未闭合；Ascend受影响持久cache需迁移命名空间/失效旧entry。
- **训练与后训练机制都有明确实验边界。** MFSDP分片Muon、TorchTitan RL CP和verl tail-aware KL进入main；论文C-OPSD、AsyncOPSD、AdamPM与targeted CPT各有可复现方向，也分别有verifier、冻结teacher、实际峰值收益较小或负迁移限制。
- **新推理优化必须按完整配置比较。** AMD MI355X megakernel提供专用分支和复现脚本，最快GPT-OSS比较混合drafter/精度/分区；TideKV以TTFT换容量，TokenGate需要任务训练，cuTile Mamba-2仅前向kernel测量。
- **评测和数据工件也有立即可用的纠偏。** FairStress实际Core48,005行、split路径不一致；POPE读出空格token可移动6.13pp F1；ReaDy v2删掉重叠风险披露但没有新实验证据。数据路径、token读出和历史版本都应固定再比较。

目录：[模型与报告](#models) · [预训练与架构](#pretrain) · [后训练](#posttrain) · [推理与训练系统](#inference) · [核心工程](#core) · [扩展工程](#engineering) · [硬件软件协同](#hardware) · [训练数据](#data) · [评估与可靠性](#evaluation) · [版本复核](#revisions) · [覆盖与缺口](#coverage)


<a name="models"></a>

## 模型与技术报告

### MemT-1.0：原生记忆权重与报告补录，固定状态换取历史细节损失

**MemTensor；论文[2610.26825v1](https://arxiv.org/abs/2610.26825v1)首次提交10-19T17:54:35Z，Nano/Flash权重10-19T22:06Z上传，本轮补漏；10-20T05:49Z仅更新报告链接。** [Nano固定工件](https://huggingface.co/MemTensor/MemT-1.0-Nano/tree/30fe0b22422c4cc4c87475f98db5c8baa5c1a902)、[Flash固定工件](https://huggingface.co/MemTensor/MemT-1.0-Flash/tree/a707cf84ebbaf2151a005001f9dfed9ee97f60c8)。发布卡称Nano为Qwen3.6-35B-A3B骨干、约36B总/3B激活，Flash为DeepSeek-V4-Flash骨干、约288B总/17B激活；这是发布方参数口径；HF元数据对Nano统计约35.11B个BF16张量元素，与卡称约36B有计数差异，本轮未逐张量核算可训练参数。Nano/Flash新增可训练记忆约451M/3.61B，动态记忆约10M/336M元素，不能把后者写成上下文token数。

问题是长会话每轮重放历史及保留KV的成本。模型在骨干attention旁并联记忆读出，段末以稳定GDN-2逐token写入固定形状矩阵；推理时变的是会话状态，骨干权重不在线更新。论文§3.2披露20B CPT+14B后训练数据储备，包含匿名内部交互；§3.4–3.6中Nano先用CE进行CPT，再用全上下文teacher的Top-64分布KL蒸馏，2K段/最多64K pack、只训练记忆参数。Flash先准备骨干，再做记忆蒸馏；不应把“34B数据”解释成两个checkpoint均训练34B token或原骨干预训练量。[固定Nano代码](https://huggingface.co/MemTensor/MemT-1.0-Nano/blob/30fe0b22422c4cc4c87475f98db5c8baa5c1a902/modeling_memt.py)的`MemTMemory.forward`在commit路径逐token更新FP32矩阵；`MemTAttention.forward`把门控记忆读出加回attention；`commit_memory`与`MemTSession._finish`区分读快照、完成回复后的写入和清理原文。这是静态对应，未执行其CUDA内核或训练系统。

§6.2/Table2的11任务平均Flash **50.60**、Nano **46.97**，高于所列参数化记忆基线Temp-LoRA38.08；但文本记忆MemOS在两骨干上约68.5，故不能写“全面超过RAG”。§6.4/Table4中Nano改为按turn更新降至40.98，移除后训练降至36.28，支持逐token存储与后训练的重要性。§6.5–6.6的Nano **128K历史1.86×端到端加速、66×每轨迹状态缩减**，条件为HF、独占2GPU、batch/concurrency1、35条BEAM历史、512-token共同query、强制128输出token；计入历史准备/状态恢复，排除加载/分词/scoring，GPU型号未在该测量段给出。256K下才约3.99×；66×不含共享权重、embedder和hyper-memory，不能写成模型总显存缩减。

固定容量会丢失精确旧证据，§8明确保留与覆盖冲突；独立会话数增多仍线性增加状态总量。实践上先验证长历史精确召回、更新冲突、未完成tool链和保存恢复。许可须分开：[模型卡](https://huggingface.co/MemTensor/MemT-1.0-Flash/blob/a707cf84ebbaf2151a005001f9dfed9ee97f60c8/README.md)标Flash MIT、Nano Apache2.0，而[固定GitHub仓](https://github.com/MemTensor/MemT-1.0/tree/e75c13d090a6a9d34b29146d901bd84323185f24)软件为PolyForm Noncommercial、报告CC BY-NC-SA4.0；不能把整个推理栈统称可商用Apache/MIT。

稳定ID：`arxiv:2610.26825v1`；`hf:MemTensor/MemT-1.0-Nano@30fe0b22422c4cc4c87475f98db5c8baa5c1a902`；`hf:MemTensor/MemT-1.0-Flash@a707cf84ebbaf2151a005001f9dfed9ee97f60c8`。

### Muse Glimmer报告：从Spark预训logit蒸馏到OPD/RL，量化质量与单卡容量要分开看

**Meta Muse Glimmer团队；2610.24628v1，2026-10-17T11:05:21Z；补录技术报告，非新权重。** [全文](https://arxiv.org/html/2610.24628v1)。§2/Table1披露约29.6B dense、52层、6656维、32Q/2KV、SwiGLU、每4层三local一global、local窗口2048，图像编码器约1.8B；这些是论文规格，实际发布工件由模型条目另核。§4.2的预训先用Muse Spark在线记录top256 logits，前95%仅KL、最后5%切CE纠正截断偏差；后续domain midtraining，再以CE扩32K→128K并加入agent数据。报告没有披露足以复算的总训练token/GPU预算与完整数据配比。

§5说明SFT轨迹取自多个Spark RL/RLHF run，清理reward/身份/工具格式，再用学生自身rollout做OPD，接RL；“OPD提高后续RL”缺完整等总算力消融，不能当普遍定理。Table2 high模式SWE-Verified76.0、Pro51.2、MCPAtlas75.5；同表Qwen3.6-27B SWE-V77.2、OSWorld75.6高于本模型65.9，不是所有agent项领先。AppA披露OSWorld为361任务、四次尝试、200步且不同模型action接口不同，SciCode分数又早于9月harness变化；应保留各harness条件。

§7/Table3的15项均分BF16 **75.7**，EvoPress75.6、Greedy-KL75.3、Q4KM74.8、NVFP4/MXFP874.4；低精度并非一律无损。目标是LM权重<20GB，在24/32GB设备留KV、视觉和drafter空间，不能由权重大小保证128K全栈单卡可用。DFlash为5层、block16、全local2048、target层{1,13,25,37,49}特征，离线用目标rollout训练。**建议深读训练配方、复现真实单卡任务**：固定量化变体、effort、完整上下文和draft/KV峰值，同时测任务成功率。稳定ID：`arxiv:2610.24628v1`。

### d3-turbo：回溯补录共享状态的多模态决策权重

**vLLM Semantic Router；[固定模型卡与工件](https://huggingface.co/vllm-sr/d3-turbo/tree/a0f3e9e59ebd1202fe1d4de1c4c0f9a986758cbb)。** v1.0整包10-19T13:58Z、v1.1为18:31Z，主窗10-20补Engine用法和榜单图日期说明；本轮补录旧新权重事件，不重报10-10的Decision3系列首次发布。tree见11个safetensors分片与独立readout，Apache2.0。

卡称26.09B（含0.46B视觉编码器），一次读state并回答多道typed questions，支持文本/图像/视频与choice/noul/score/box/track，输出选项概率而非生成长文本；默认32K、显式long_context后131,072/token每问题。作者在单MI325X、AutoModel下报告255题约2.12秒、相对d3-flash15.5倍，但单题d3-flash更快；共享输入收益依赖同一state的提问数。Jev70.2/vision72.9是作者估算，非官方board结果；完整训练数据/预算未在卡披露，不能把softmax输出天然视为业务校准概率。建议比较同一输入多题吞吐、题间干扰及拒答阈值；本轮仅核查卡/配置/工件存在，未完整审计其154KB runtime或运行benchmark。

稳定ID：`hf:vllm-sr/d3-turbo@689b3af198baaf6f11110be38a9daa7188dfd8ad:v1.1`；文档基线`a0f3e9e59ebd1202fe1d4de1c4c0f9a986758cbb`。

### Xing4.0-29B-Intent：补录schema驱动抽取工件，275万语料不是本日新上传

**中国电信星辰通用人工智能实验室；本轮名单外搜索发现。** [固定模型卡](https://huggingface.co/XingChen-AGI/Xing4.0-29B-Intent/blob/070c34b10a7cb8dab87c9f602e38cd67826ab48f/README.md)、[固定数据卡](https://huggingface.co/datasets/XingChen-AGI/Xing4.0-29B-Intent-oss/blob/cc6fe57ac81228f47dd66821ab54dcc85ddd7e78/README.md)。权重整包10-15T12:08Z、数据10-16T07:30Z已上传，10-17/19改卡、主窗10-20只补library metadata；因此是旧工件补漏，不把10-20媒体报道当新权重事件。

它把意图/槽位、NER、关系、事件抽取统一为schema约束的`category+args` JSON数组。卡称29B总/4B激活MoE；[实际配置](https://huggingface.co/XingChen-AGI/Xing4.0-29B-Intent/blob/070c34b10a7cb8dab87c9f602e38cd67826ab48f/config.json)为40层、64路由专家top4、1共享专家、KV低秩512、1个MTP层。数据卡称2,753,000样本，来自61公共数据集重构与电信设备场景内部标注，覆盖19域/52语言；本轮未下载逐行重算数量或审计各源去重、原许可与train/test交叉。

作者以48公共数据集抽取6,000测试样本，按每数据集字段值Micro-F1→任务宏平均→四任务等权平均，自报总58、意图81；不是业务准确率，类别正确不保证字段全对。比较模型的完整提示、推理预算和拆分未在卡中给足，不据此宣称全面胜过所列闭源模型。模型/配套数据标Apache2.0，数据仍保留各原始来源条款。建议保存schema/prompt相对路径、解析JSON后评估字段，并独立留出业务域测试。

稳定ID：`hf:XingChen-AGI/Xing4.0-29B-Intent@070c34b10a7cb8dab87c9f602e38cd67826ab48f`；`hf-dataset:XingChen-AGI/Xing4.0-29B-Intent-oss@cc6fe57ac81228f47dd66821ab54dcc85ddd7e78`。

### 微软Sentiment-Analysis-7B：小样本LLM裁判RL的新权重，评测分母仍缺失

**Microsoft；空仓初始10-17T00:51Z，实际整包10-20T19:07:36Z上传，模型卡19:28Z修订。** [固定模型卡](https://huggingface.co/microsoft/Sentiment-Analysis-7B/blob/c443f040bb22ad30d69dbdd4482c8aa0f46a4d62/README.md)、[上传commit](https://huggingface.co/microsoft/Sentiment-Analysis-7B/commit/02605a683b62681af76cb6548ff679ea17f007b2)。本轮tree确认新增15个safetensors分片、配置和tokenizer；创建时间不等于权重公开时间。

模型基于Mistral-7B-Instruct-v0.2，对“用户面向当前服务/产品/agent的情绪”做三分类，输出标签及解释。作者称用约300条内部样本、无预定义训练标签、以LLM裁判进行RL微调；卡未披露RL算法、裁判型号、奖励/采样配方、训练预算或训练数据。自报accuracy由62.0%至72.8%，但测试样本数、数据划分、置信区间均未给，不能据此推断通用情感理解提升。模型卡标英语、MIT。

它的标签规则具有产品约定：单纯故障陈述判负面，加礼貌求助可判中性；通用情绪数据集不一定采用同一标准。建议先冻结系统提示与业务标签定义，用独立人工标注集检验这类边界；不要将其直接当RL配方已可复现。

稳定ID：`hf:microsoft/Sentiment-Analysis-7B@c443f040bb22ad30d69dbdd4482c8aa0f46a4d62`。

<a name="pretrain"></a>

## 预训练与架构

### KronLSTM：把整向量覆盖改成联合key/value地址上的标量修正，长上下文收益仍有明显失败项

**NX-AI/KronLSTM项目；2610.26834v1，2026-10-19T17:56:52Z；回溯补录。** [全文](https://arxiv.org/html/2610.26834v1)、[官方代码](https://github.com/NX-AI/KronLSTM)。§2的Kronecker Delta Rule只沿value×key外积方向修正双线性标量误差，旧关联受扰同时取决于key与value重合；与DeltaNet重建整个value的目标不同，不能说在同一目标下严格支配。完整块另加key方向遗忘、分离erase/write、L2规范化与输出RMSNorm。

§3/Tables1–4比较1.5B、100B-token、8K预训的linear与SWA hybrid。linear六项均分 **KDA58.12→KronLSTM59.72**；现实召回均分GDN29.81→30.31，但NQ、DROP不领先。8K训后16K S-NIAH1仅11.6，GDN99.2，表明单needle并非全面改善。32K扩展从75B checkpoint再训25B、global1M token、32×GH200、BF16；不是额外25B不受控比较。扩展后GDN-2/KronLSTM linear六项58.75/60.19，但32K retrieval两者仍明显下降。

§3.5单H10080GB固定16384 token/step，KronLSTM36.8–44.5K tok/s、比GDN-2快11–18%；GDN仍更快，不能写最速linear attention。§2定理限定iid三元组、固定步长、正定协方差，不保证整套深层可训练网络任意语言序列收敛。**建议深读/小规模复现**：先按任务类型核key/value干扰与write-gate消融，再测硬件kernel、上下文扩展后的绝对召回；本轮未代码审计。稳定ID：`arxiv:2610.26834v1`。

### Looped Transformer截断BPTT：固定循环数时截断可能合算，循环数也可选时更少完整循环更好

**Looped Transformer scaling研究团队；2610.26030v1，2026-10-19T08:52:29Z；回溯补录。** [全文](https://arxiv.org/html/2610.26030v1)。§4–6联合扫物理参数、token、forward循环T与backward循环B：592 runs/347配置，230拟合、117留出，58–919M非embedding参数、FineWeb-Edu100B子集、seq2048、MuonH/AdamW。截断前T−B次forward的state梯度，而每轮输入注入仍能把梯度传到prelude。

Table7在同模型/17.2B token/**相同训练FLOPs**下，(8,2)→(4,4)的1B validation loss **2.1144→2.1057**，五个规模均改善。固定T=8时最佳B可为2；允许减少T后完整BPTT胜出，两结论不能混用。§6.2拟合允许模型大小自由时总体偏好无循环；§6.3两次完整循环以约27%更少非embedding参数、约8%更多训练算力达到略低loss，是参数/算力权衡，不能写循环在统一算力下一律优于普通Transformer。

只测validation loss、dense、T≤16，未测任务推理和MoE，超出区间的收益上限是拟合外推。**建议复现**：把“更多forward但截断backward”与“较少forward完整backward”列成同FLOPs对照，随后才验证数学/代码下游收益。实现未审计。稳定ID：`arxiv:2610.26030v1`。

### Where RL Breaks：用失败n-gram检索35M继续预训语料，收益应对照同预算随机检索与不做CPT

**Where RL Breaks项目；2610.25836v1，2026-10-19T05:48:15Z；回溯补录。** [全文](https://arxiv.org/html/2610.25836v1)、[官方代码](https://github.com/WANGXinyiLinda/where-rl-breaks)。§3–4从RL rollout里找低条件准确率短语/运算片段，到公开Olmo-mix/Dolmino检索、各query平衡凑35M token，CPT后重跑固定GRPO。对照也走相同检索pipeline和预算，仅换content-neutral query；无额外标签/教师不等于无需先跑RL发现失败。

Tables2/4：OLMo2-1B在midtraining前插入时GSM8K无CPT37.4、control36.2、targeted38.2；**比control+2.0pp，比不做CPT仅+0.8pp**。已发布base的multitask是52.6/49.5/51.5，targeted仍低于不做CPT。Qwen3-4B迁移OLMo检索语料multitask72.9/72.1/74.0。OLMo GSM8K三seed，其他多数单seed；一般推理有算力截断，仅在匹配RL步数比较。

AppA5/6更正了容易夸大的因果解释：CPT有时只使反思短语出现更频繁，条件成功率未升；“修复状态”与最终accuracy可独立变化。Qwen3-4B自检索从35M扩1B预RL八项均分40.5→18.8，更多同类CPT会破坏能力。**建议小规模复现**，同时保留不做CPT、随机、目标检索三臂及replay/一般能力检查，不据摘要10/11相对control胜出写成所有任务绝对改善。稳定ID：`arxiv:2610.25836v1`。

### Safety Mid-training：全token训练结构化知识/推理后，安全性更能承受general-only后训

**Safety Mid-training研究团队；2610.26809v1，2026-10-19T17:49:49Z；回溯补录。** [全文](https://arxiv.org/html/2610.26809v1)。§3/AppB用模型自生并过滤的安全与一般任务“结构化文档+推理轨迹”做causal LM，所有system/user/assistant token均算loss，packed3072；区别于仅assistant loss的SFT。Qwen3-4B参考约198M token、1 epoch、batch64、AdamW2e-5；reasoning-only约105M。

Table2 general-only SFT后，Qwen3-4B Base+SFT的ASR13.6/utility59.4，本法7.6/74.9；AP+SFT为14.5/73.3。Table3相同reasoning数据直接SFT为22.8/68.9，MT+SFT5.4/74.6，支持消费方式影响，但完整document+reasoning还引入额外数据量。主均分为排除共源数据后的7安全/7效用项，图1等另用9+9全套，不能混报。

实验≤8B，judge和合成过滤相关、语言与交互覆盖有限；不能把较低静态ASR解释成任意继续训练不再失效。**建议观察/复现**，用严格同token、同source配比对齐全token/assistant-only目标，并测general-only GRPO后安全与过拒绝。本轮未发现已核定专用实现。稳定ID：`arxiv:2610.26809v1`。

### t-Muon：先按attention结构分块再做tensor polar，较大收益不一定来自跨切片变换

**t-Muon研究团队；2610.24938v1，2026-10-18T01:20:16Z；回溯补录。** [全文](https://arxiv.org/html/2610.24938v1)。§3把Q/K/V梯度按layer、QKV、head或head-stack组成三阶张量，固定正交变换后各slice取polar；其余MLP/attention-output仍Muon，embedding等AdamW。identity Head-II即独立head正交化，作者明确不把既有head-wise思路当新贡献。

Tables1–2主实验50–780M、1B FineWeb token、LR sweep三seed；LLaMA350M Muon ppl19.386→Head-II19.055。Table5在FineWeb-Edu、1.34B模型/13.1B token单seed为9.45→9.34；**DFT/DCT/DST变体没有优于Muon**。GPT-2的Muon对fused-QKV整体正交，而t-Muon先拆块，未隔离拆分本身的收益。理论用精确polar/固定变换/单group，实用多项式近似与混合optimizer不完全覆盖。**建议小规模复现**独立QKV和head分块基线，暂不据此替换大规模训练optimizer；无下游结果，代码未核。稳定ID：`arxiv:2610.24938v1`。

### AdamPM：只pool二阶矩可省optimizer state，实际训练峰值省得远少于静态状态估算

**AdamPM项目；2610.24598v1，2026-10-17T10:04:48Z；回溯补录。** [全文](https://arxiv.org/html/2610.24598v1)、[官方代码](https://github.com/jixiaowei95/AdamPM)。§3保持逐元素一阶矩，把同一行/列相邻s个平方梯度先平均再做EMA，二阶状态降为1/s；全部optimizer状态由2P降P(1+1/s)。不是把梯度先平均平方，亦不pool一阶矩；后者Table2明显恶化。

Table1 C4、seq256、1.3B/13.1B token，Adam ppl14.83、AdamPM-4 **14.66**，BF16参数+state估算7.48G→5.61G；部分GaLore/APOLLO等数值取原论文，不是同机重跑。AppC1在4×A10040GB、batch32，实际每卡训练峰值38.11→35.96G，时间116.2→116.4h；不能写“显存省25%且训练加速”。s=∞的ppl15.54比Adam更差；7B只有静态内存估算，无完整训练。**建议复现s=4/8与非整除组、checkpoint恢复**，区分状态、activation、临时buffer与实际峰值；本轮未代码审计。稳定ID：`arxiv:2610.24598v1`。

<a name="posttrain"></a>

## 后训练

### AsyncOPSD：异步rollout可近翻倍样本吞吐，但实验冻结teacher而不是始终动态共享教师

**AsyncOPSD研究团队；2610.25880v1，2026-10-19T06:38:04Z；回溯补录。** [全文](https://arxiv.org/html/2610.25880v1)。§4将rollout actor和learner解耦，对过旧queue数据设cutoff，并按token entropy调蒸馏权重。**§5.1明确所有实验将teacher冻结为初始模型**，学生TM-off、teacher TM-on且多看reference；不能把摘要“teacher/student共享参数”理解成该实验每步更新同一个teacher。

Qwen3 instruct1.7/4/8B、OpenThoughts math、8×A100、LoRA64/BF16、global32，异步4GPU actor+4GPU learner，同步8GPU交替。Tables2–3：4B accepted sequences/s **2.40→5.39（2.25×）**、Avg@12三项63.60→64.70；8B1.70→3.40（1.99×）、64.80→65.67；1.7B1.96×。三次重复用于吞吐/staleness误差，不能自动当accuracy多训练seed区间。评估T1、max38K。

吞吐口径为实际训练消费的序列数/Trainer时间，整套batch与refresh策略共同变化，不是tokens/s或等wallclock质量保证。小规模/LoRA/三数学集限定外推。**建议复现**时固定teacher策略、显卡分配与接受样本定义，同时画质量—wallclock曲线；实现未审计。稳定ID：`arxiv:2610.25880v1`。

### C-OPSD：删掉多数蒸馏行后再packing，在单GB300省约40%训练时间

**C-OPSD研究团队；2610.25842v1，2026-10-19T05:55:07Z；回溯补录。** [全文](https://arxiv.org/html/2610.25842v1)。§4–5测得错误rollout梯度coherent share约正确者三倍，均值方向又大体重合，据此每题只生成一条，在microbatch n=4内保留前min(k,n−k)条错误响应（k为正确条数）。**不是保留所有错误，更不是停止生成正确响应**；选择后在teacher/student forward前删行，末尾packing保留归一化权重。

Tables6–9：Qwen3-1.7/4/8B，32题/step、rollout非thinking/max1024、100steps，eval thinking/max38912/Avg@48，五训练seed；每benchmark选最好checkpoint。三项平均full/C为42.26/42.06、62.63/62.86、64.96/65.05，保留26%；1.7B同数随机41.58、correct-only40.35。Table11实际执行row减约68%，Table13单GB300100-step时间14.2→8.4、19.8→11.8、22.6→13.7分钟，约40–41%。

FLOPs模型省55.5%与实测40%不同，rollout成本未省，结果还依赖判题器、microbatch和packing。**建议优先小规模复现**：保留随机/全错臂，验证空组和gradient accumulation权重；不要在无可靠verifier任务直接套用。专用代码未核。稳定ID：`arxiv:2610.25842v1`。

### BSD：外部教师只给文本，学生同token空间的privileged self-teacher负责细粒度纠错

**TUMLegalTech/BSD项目；2610.26397v1，2026-10-19T13:40:50Z；回溯补录。** [全文](https://arxiv.org/html/2610.26397v1)、[官方代码](https://github.com/TUMLegalTech/Bridged-Self-Distillation)。§3先离线每题取一条外部teacher trace，学生不看trace生成；定期同步的student副本看trace给同rollout打分。只在失败rollout负advantage上用logprob差重权惩罚，成功轨迹维持GRPO，因此仍依赖verifier，不是跨tokenizer直接KL。

Table1/AppA：0.6/1.7/4B及额外1K轨迹激活thinking的4B-Instruct，2400题/150更新、LoRA32、H100/H200；eval三套各30题、Avg@4、独立128题dev选checkpoint。4B共享Qwen3-32B teacher为OPD60.3→BSD63.9，跨tokenizer为xOPD59.7→63.6（与SFT63.6并列），0.6B跨tokenizer14.2→12.8反降，不能称每个pair都胜baseline。

闭源teacher仅返回最终文本，作者将其构成privileged trace，未取得隐藏CoT。Table6训练GPU-hour排除离线teacher生成、checkpoint/eval，API compute未计；BSD相较OPD并非总是更省（4B共享teacher29.4→31.4GPUh）。**建议复现跨tokenizer场景**，对齐rollout数量、response cap和所有teacher成本；本轮未代码审计。稳定ID：`arxiv:2610.26397v1`。

### OPDArena：教师能力来源比教师大小更能解释学习—遗忘取舍

**OPDArena项目；2610.25651v1，2026-10-19T01:51:24Z；回溯补录。** [全文](https://arxiv.org/html/2610.25651v1)、[官方框架](https://github.com/StellarCurtain/OPDArena)。五种Qwen2.5/Qwen3 Base、同起点/OpenR1-50K一epoch、Raw无chat模板，比较SFT/RLVR/多种teacher OPD；Self-OPD这里指独立SFT或RLVR endpoint冻结后教新Base学生，勿与privileged-context OPSD混用。数学5项和OOD7项分别等权，主结果两训练seed。

Table1 Qwen3-8B：SFT数学31.32→47.32但OOD51.37→48.54；RLVR45.39/52.12，Self-OPD(RLVR)45.48/52.12，Self-OPD(SFT)44.34/47.23。Qwen2.5-7B Self-OPD(RLVR) OOD仍−0.24pp，不能写绝不遗忘。§4发现OOD预测分布变化与遗忘关联更稳定，教师后训偏好让更新更集中域内；属于受控关联/机制诊断，不是任意teacher泛化定律。**建议深读并加入OOD retention面板**，固定prompt模板、teacher来源、KL估计与训练预算，不用数学分数单独挑teacher；框架未代码审计。稳定ID：`arxiv:2610.25651v1`。

### Fisher几何/RGBM：同源专家的局部二阶蒸馏近似可先解merge，再谨慎回写高残差分支

**RGBM研究团队；2610.26525v1，2026-10-19T14:54:47Z；回溯补录。** [全文](https://arxiv.org/html/2610.26525v1)。§2假设同base、共享basin的专才，MOPD目标二阶展开对应Fisher merge；为避免全参数Fisher，在task-vector子空间解最坏域残差，再只训残差高的隔离分支、shrinkage回写。**“merge最优”仅局部二阶近似，非任意异构权重替代蒸馏。**

§4/Table1在SmolLM3-3B三专家Open-MOPD，同协议重测600step MOPD宏均分30.93、Phase1 merge32.52、Phase2 **33.03**；但数学23.86→23.93几乎不变，+0.51主要来自code/IF的跨域变化。继续joint distill降30.91；不同方法复用相同branch再比较回写，不能把全部差异归新训练量。第二套为Qwen3-4B两专家；仅K=2–3，线性诊断只检查一阶reward叠加、不界定三阶余项。**建议观察/小规模复现**同源checkpoint的残差校准、branch隔离和回写幅度，记录求系数/训branch全部成本；本轮未核专用实现。稳定ID：`arxiv:2610.26525v1`。

### Nash-MOPD：按每单位域曝光的相对进步分配采样，不让快学域吞掉预算

**Nash-MOPD研究团队；2610.26124v1，2026-10-19T10:15:12Z；回溯补录。** [全文](https://arxiv.org/html/2610.26124v1)。§4复用reverse-KL和变化率，按已分配曝光归一化、解局部minimax，再施加上下界/平滑；无需额外teacher queries。模型是各域独立局部进展，未刻画跨域迁移/遗忘。

§5固定200steps、每20步eval、最终step200比较，DeepSeek1.5B三域平均math62.93→64.12、code48.80→49.84、IF23.08→25.89（Fixed→Nash）；Qwen3-8B两域math62.56→63.56、code56.71→58.06。avg@4/16与pass@k另列，收敛曲线又是pass@1，不能拼成一条百分比。没有已核同GPU时长的speedup，分配未计token长短/时间差；收益大小和rate估计不确定度还需更多seed。**建议复现scheduler**时同时报域tokens、teacher前向和wallclock，检查慢域是否只是更长输出。实现未审计。稳定ID：`arxiv:2610.26124v1`。

<a name="inference"></a>

## 推理与训练系统

### llm-d 1.0：把多租户公平性与分离部署升级纳入同一控制平面

**llm-d社区；v1.0.0于2026-10-20T02:30:10Z正式发布，tag指向378b42d18dabe0b9c4ced4e6f3670d93b52db08e。** [正式release](https://github.com/llm-d/llm-d/releases/tag/v1.0.0)、[官方技术说明](https://llm-d.ai/blog/llm-d-v1.0-state-of-the-art-inference-ready-for-day-2)、[该版本测试矩阵](https://github.com/llm-d/llm-d/blob/v1.0.0/release/README.md)。原有前缀感知路由、KV卸载、P/D分离和wide EP并非这次才出现；1.0的关键是把它们变成可操作的共享服务。

- `flowControl`统一多租户流控，按fairness ID和priority band形成队列，先保优先级、同档轮转，并用概率准入缓和过载；默认仍近似FCFS，需要运营方配置策略。
- router热备与`DisaggregatedSet`管理P/D成对滚动升级，以Pod快照和revision控制ready/路由，避免新旧P/D任意配对。配置API升至v1；session-state-producer、turn priority和Responses `previous_response_id`服务多轮上下文复用。
- 版本并非所有依赖都“1.0”：官方引擎镜像表仍分别为vLLM 0.30/0.31（依硬件）、SGLang0.5.22、TRT-LLM1.3.0rc29；ROCm PD暂钉v0.32.0-rc.2。旧自建ROCm镜像弃用；P/D指南至少要求LWS0.11.1并启用DisaggregatedSet，建议LWS1.0。

官方明确active-active router、in-flight请求failover/migration仍是后续工作，**热备与安全滚动升级不等于任意进行中请求无损迁移**。测试矩阵链接是具体指南/平台的证据，不能读作任意模型×硬件×引擎组合都验证。本轮不把厂商生产案例的高缓存命中率或费用倍数当统一基准。建议已有Kubernetes多租户平台**考虑小流量验证**：压测队列公平性、低优先级shed、热备切换和P/D混合版本；先固定镜像矩阵再迁移。Google Cloud关联稿搜索日期10-20而本轮原页显示10-21，未据此补入更晚性能结果。

稳定ID：`github:llm-d/llm-d:release:v1.0.0`。

### TokenGate：训练可微token gate后在早层剪prompt，须把任务适配成本和质量边界算入

**TokenGate项目；2610.26754v1，2026-10-19T17:15:53Z；回溯补录。** [全文](https://arxiv.org/html/2610.26754v1)、[官方代码](https://github.com/tokengate-models/TokenGate)。§4早层给token分数、top-K后上层只处理保留者；把分数写进attention logits形成gate梯度，再加head gain和随保留长度变化的temperature。不是无需训练的KV剪枝；主模型Qwen3-1.7B，五任务各自LoRA16适配，gate另Muon，4×A100，训练与eval数据隔离。

§5/Fig3同任务精度下作者报告TTFT **1.4–11.3×**、prefill peak1.5–3.4×改善；是不同任务/预算点范围，不是统一并发服务SLA。完整图在AppD2，主图去掉显著低质/慢点。Qwen3.5-2B NIAH32K的layer3门同accuracy TTFT3.8×，早至linear层可7.8×但accuracy低至差6%，不要合称无损7.8×。每训练step约慢1.3×，四/五任务更早收敛可抵消总时间。

**建议深读/复现目标语料**：拿同LoRA无gate、同层FastKV、随机选择比较，测跨任务与超训练长度的quality回归；prefill优化不能自动推导整个长生成任务同倍加速。代码未审计。稳定ID：`arxiv:2610.26754v1`。

### TideKV：逐层chunked prefill后再驱逐KV，降低构建峰值但会增加TTFT

**TideKV研究团队；2610.24499v1，2026-10-17T06:44:27Z；回溯补录。** [全文](https://arxiv.org/html/2610.24499v1)。§4改调度为layer-major、一层先处理所有prompt chunk，再按SnapKV/PyramidKV等原selector压缩，避免在编码prompt中途驱逐造成隐藏态偏差；decode前依然是有损压缩cache，不是所有输出与full KV严格等价。Table8在20条16K prompt上，相对Full-OneShot的TideKV logit top1一致.95，与BF16 chunking floor相同；只验证该样本与条件。

Tables2/9：Llama3.1-8B、Mistral7B/BF16、batch1/chunk2048、相同最终512MiB物理KV。A10040GB128K Snap→Tide peak **33.080→18.572GiB（−43.86%）**、TTFT42.037→43.791s（+4.17%）；64K TTFT+16.74%。RTX409024GB128K Snap OOM、Tide18.529GiB/47.110s；Orin64GB的64K也从OOM变可行。LongBench同selector差异小但有正有负。

Table5 SDPA64K虽然allocated下降，reserved/NVML反而增约23%，要把allocator/backend纳入。**建议复现内存受限长prompt**：分清final cache、allocated、reserved、NVML以及首token时间，固定软件栈与同selector；本轮未代码核验。稳定ID：`arxiv:2610.24499v1`。

### Tandem：移动NPU混合attention agent要在prompt分叉处存recurrent state，投机收益依赖rollback布局

**Tandem项目；2610.26660v1，2026-10-19T16:18:16Z；回溯补录。** [全文](https://arxiv.org/html/2610.26660v1)、[官方实现](https://github.com/flyingdoog/tandem)。§4在相邻GUI请求历史重写的分叉点checkpoint recurrent state，大部分prefill留矩阵单元、短尾生成rollback state，验证kernel对一块权重一次服务多draft token；idle时释放NPU power vote。是llama.cpp运行时及Hexagon kernel改动，不是新agent训练法。

§6/Table2 OnePlus13T/Snapdragon8Elite/16GB、Qwen3.5-4B Q4_K_M，同模型回放61text steps为2.01→1.44s（−28%），510 held-out text2.02→1.45s，截图6.32→5.35s（−15%）；模型请求时间**不含app执行**。正文主latency用15W制冷器，能耗另在电池无冷却下测10.1 vs15.1J；不能把两个protocol混成全天手机续航预测。Table4减少图像token也改变任务质量。

rollback额外201MiB、draft68.8MB、额外checkpoint约50MiB，prompt cache cap512MiB。优化单项在已测trace维持action，CPU/NPU视觉路径仍可能不同。Granite Mamba2 scan仍CPU、并非所有混合层全上NPU。**建议观察/复现**相同prompt结构和模型的短动作场景，另测热稳态、实际任务端到端时间；代码未审计。稳定ID：`arxiv:2610.26660v1`。

### PACT：NVFP4同时用于linear GEMM和保存activation，在16GB消费卡换容量与速度

**PACT研究团队；2610.25048v1，2026-10-18T06:14:14Z；回溯补录。** [全文](https://arxiv.org/html/2610.25048v1)。§3将冻结权重、linear计算和保存的activation纳入NVFP4数据流，非线性/LoRA/attention仍BF16。§4主实验**1B–3B** decoder（HTML数学排版重复数字不能读成11–33B）、单RTX5080 16GB、PyTorch2.11/CUDA13.2、LoRA16、三seed学习率扫选；非多GPU大模型预训结论。

Table2九分类项Llama3B BF16 LoRA .810、QLoRA .804、PACT .801，1B .729/.715/.703，存在质量损失。固定batch4/seq1024速度1.7–1.9×、peak减47–51%；Table5各方法最大可容batch不同，1.85–2.41×吞吐把容量和kernel效果混在一起。单GEMM6–7×不等于端到端同倍；训练时长还排除offline calibration/eval。

**建议小规模复现**需要长seq但显存紧的LoRA任务，同时与QLoRA、TE、checkpointing做同batch/质量对齐，保留attention和非线性精度边界；本轮未核定专用发布代码。稳定ID：`arxiv:2610.25048v1`。

### LatComm：冻结两端LLM、训练跨层读写bridge可复用压缩context，但尚未验证真实多agent通信

**LatComm项目；2610.26840v1，2026-10-19T17:58:41Z；回溯补录。** [全文](https://arxiv.org/html/2610.26840v1)、[官方代码](https://github.com/huawei-lin/LatComm)。§3把encoder多层context隐态压成固定slots，receiver对应层按slot/layer gate读取；packet无query、可多次复用，需每对encoder/receiver独立训练bridge和生成packet，不能跨任意模型或API直接传。

§4六model-pairs、receiver约1B、encoder≤9B，先2M英语passage再12种多选数据；默认每layer-pair32×256slots。主结果TaskAvg65.43/held-out49.76，高于最佳压缩baseline61.98/46.70。**原文raw-text未做任务适配，在held-out反达65.06，高于全部训练bridge的41.72–49.76**，不能写压缩模型全面胜原文；constant packet也很强，必须保留无信息控制。

Table7 H100 batch1、QuALITY115篇/2086题，每context18.1题，预编码.71s后每问131ms vs原文511ms，总3.09 vs9.26s；这是可复用多问摊销，且peak指loaded models之上单次call增加，不是全栈显存。只测英语多选单字母，未测自由生成/agent，除默认pair stage2三seed外多为单seed。**建议观察/复现长期复用context**，先保住held-out质量再算编码摊销。稳定ID：`arxiv:2610.26840v1`。

<a name="core"></a>

## 核心开源工程

本节事件主窗为 **2026-10-20T00:55:34Z→2026-10-21T00:56:38Z**，回溯至2026-10-17T00:55:34Z。以下未特别标明者均已合入main，以各PR merge commit冻结源码；作者测试结果未在本机重跑。release collection另查，本主窗未见新正式发布。建议优先回归Mooncake失败接收、LoRA深睡唤醒、EPLB路由/副本与graph地址寿命，再评估融合性能。

### NVIDIA/Megatron-LM

**MFSDP v2: add Muon with sharded momentum。** MFSDP v2 的 Muon 先把动量留在分片上，再由固定 owner 收齐矩阵做正交化、scatter 更新并刷新模型权重，避免每张卡常驻完整动量。新增独立 `experimental.muon.FsdpMuon`，当前各 buffer 在调用者 CUDA stream 串行完成；限一维 DP mesh、二维参数及非量化 main weight/gradient，不支持 TP，非二维参数需其他 optimizer。作者报告双 GPU 的16组数值对照逐位一致；EO 内部算子未重审。建议先用于受支持矩阵，并把 embedding、输出层及 checkpoint reload 纳入对照。

来源：[#8313](https://github.com/NVIDIA/Megatron-LM/pull/8313)（main，合入2026-10-20T21:22:05Z；`github:NVIDIA/Megatron-LM:pr:8313`）。源码基线 `91d774de79a6d732f28d7f44fab0abbaac1b57a6`。

**fix: Defer frozen MFSDP cleanup without input gradients。** 冻结 MFSDP 单元即使自身没有可训练权重，也可能包住可训练的分片 child；输入不需梯度时 full-backward hook 会过早触发并释放 child 反传仍要读取的冻结权重。现在把该类完成回调延到最终 autograd callback，每个 microbatch 清理权重，最后 microbatch 才自动等梯度；组合1F1B调度显式调用 `context.post_backward()`。建议冻结基座/局部微调核查 `test_frozen_weights_without_input_gradients`，不要把 `finish_grad_sync()` 的等待当成权重清理。

来源：[#8113](https://github.com/NVIDIA/Megatron-LM/pull/8113)（main，合入2026-10-20T08:33:51Z；`github:NVIDIA/Megatron-LM:pr:8113`）。源码基线 `d37b3a8fdb37af1b3a1d75720636af4c323bd522`。

**Sync tied embedding copies under meta-device init。** meta 初始化下 tied embedding 的跨 PP/MTP 副本曾跳过 all-reduce，MFSDP 因而可能从不同初值训练。`LanguageModule._materialize_shared_embedding_weight` 先分配并初始化共享副本，再执行 embedding-group all-reduce；参数若被初始化器替换，补回 requires_grad 与公开属性。涉及 `--use-megatron-fsdp --init-model-with-meta-device` 且 PP>1 或 MTP；建议从该组合重验 tied-copy 相等，而非只测单 stage。

来源：[#8010](https://github.com/NVIDIA/Megatron-LM/pull/8010)（main，合入2026-10-20T18:17:33Z；`github:NVIDIA/Megatron-LM:pr:8010`）。源码基线 `fa322ce62077fa34ac6927fdc66ab2c8b88fcc30`。

**refactor(checkpoint): re-architect CheckpointWithoutOutput internals。** 输出丢弃重算改用明确的 RECORDED→RELEASED→REBUILT→DONE 生命周期：输入由 saved-tensor hooks 保存，重算恢复 RNG/TE FP8 上下文，再在 storage 层让旧输出及其 views 看见重建数据，避免 autograd 引用环与重复解包。无 autograd 图时 `checkpoint()` 直接调用；同一图重复 backward 明确报错。调用接口不变，依赖的 TE/storage-sharing 实现未逐行重审；建议重点回归 activation offload、views、FP8 与 CUDA graph。

来源：[#8334](https://github.com/NVIDIA/Megatron-LM/pull/8334)（main，合入2026-10-20T13:43:33Z；`github:NVIDIA/Megatron-LM:pr:8334`）。源码基线 `d2c430ab485c6947701447e5d551d3aafaa39161`。

**Fix `config.softmax_scale` not being considered。** 部分直接 FlashAttention 路径忽略 `config.softmax_scale`。`Attention.get_softmax_scale` 统一按实例值→配置值→head-dim 默认值选取，接到 decode/prefill；MLA 在配置 scale 上继续应用 YaRN mscale²。该修复改变显式自定义 scale 模型的实际数值，默认路径保持原取值规则；建议以非默认 scale 覆盖 FA2/3/4 和 MLA，而非只跑默认模型。

来源：[#3698](https://github.com/NVIDIA/Megatron-LM/pull/3698)（main，合入2026-10-20T18:44:35Z；`github:NVIDIA/Megatron-LM:pr:3698`）。源码基线 `0156c213738eaf718f7fba2a77a3b79fa4f4ef97`。

**Refinements to Offline KD。** 离线 logits KD 把 tar 身份改为全局样本区间，teacher 汇聚完整 CP 序列，再由 student 按 DP/MBS/GBS 与本地 CP 重切，支持复用 teacher 缓存训练不同 student 拓扑。关键边界是同 seed/数据管线，MBS/GBS 整数倍关系、顺序消费与样本起点检查；teacher 较短序列未处理。尤其 packed SFT 并非“CP任意互换”：源码要求 load CP 整除 saved CP，向更小 CP 切换仍警告文档截断点可能不同，内容对齐不被检测或修复。建议精确蒸馏保持 packed CP 一致；异步写盘 cadence 随 checkpoint，并检查失败事件/`.stale` 隔离。

来源：[#5482](https://github.com/NVIDIA/Megatron-LM/pull/5482)（main，合入2026-10-20T21:41:40Z；`github:NVIDIA/Megatron-LM:pr:5482`）。源码基线 `fce1e2051f8ad37de5473ae103074748a8948ea5`。

**Stage request payloads off the dynamic engine loop。** 动态推理曾在完成请求的 bookkeeping 中同步写 payload，慢网络写会堵住其他请求 decode。现在先在 loop 上形成 host payload，再以 `asyncio.to_thread` 执行 stage；该请求仍等写完成才发 reply，ZMQ 只在 loop 线程使用。任务集合保活，reset 拒绝未 drain 的任务，正常 shutdown 先等 staging；写失败回退未 offload reply。作者 generation-bound NeMo-RL例子约减1.3秒/step，默认异步布局原本可隐藏开销；建议监控 in-flight/peak backlog，不能据此推成所有RL作业提速。

来源：[#8169](https://github.com/NVIDIA/Megatron-LM/pull/8169)（main，合入2026-10-20T12:14:12Z；`github:NVIDIA/Megatron-LM:pr:8169`）。源码基线 `894b5114e3cb1f3d8d5ead12634e8bc7881d30b3`。

**Fix strict checkpoint loading with idle rerun state。** 启用 rerun 验证保存的普通 checkpoint 含 idle sentinel，但 load 计划曾省略它，严格 `raise_all` 因缺 key 失败。现在在符合相同拓扑条件的 load 计划中请求 idle shard，却不触发 rerun；release、finetune、拓扑检查仍生效。作者给出双 GPU torch_dist 保存/恢复用例；建议严格恢复同时核验 iteration、FLOPs 与 rerun mode。

来源：[#7222](https://github.com/NVIDIA/Megatron-LM/pull/7222)（main，合入2026-10-20T11:58:36Z；`github:NVIDIA/Megatron-LM:pr:7222`）。源码基线 `e564873e9b72839b4bbd28385ce1cdb75a499eb2`。

### vllm-project/vllm

**[Bugfix][KV Connector][Mooncake] Wait for all peers before completing failed receives to avoid potential KV block write-after-free。** Mooncake 多 P→单 D 接收中，一个 P 报错曾立刻发布 receive-complete，其他 P 仍持目标地址，调度器归还 KV block 后可发生 WRITE-after-free。现在每个 peer 拥有 pending map，共享请求计数；成功、失败及 FINISH 遗漏项均只 settle 本 peer，失败只报一次，计数归零才完成接收。作者以Qwen3-32B/BF16、P TP2→D TP1、H20/200Gbps RoCE、1800秒回放791请求故障注入观察到 WRITE-after-free 21→0、重分配后 WRITE 20→0。仅证明该复现路径；本地 socket timeout 后远端是否静默仍不在范围。建议 PD 部署优先回归 delayed peer、部分回复及过期 FINISH。

来源：[#62282](https://github.com/vllm-project/vllm/pull/62282)（main，合入2026-10-20T04:13:30Z；`github:vllm-project/vllm:pr:62282`）。源码基线 `34ac4f96a399836393894207dae296760b413715`。

**[Bugfix][Distributed] Preserve symmetric collective outputs。** NCCL symmetric-memory ReduceScatter/单 tensor AllGather 曾直接返回可复用 scratch，下一次同形 collective 会覆盖仍在使用的旧结果。现在返回 clone，caller-owned ReduceScatter output 不复制；scratch/注册继续缓存。成本是每次受影响通用调用新增结果存储与一份输出大小的 D2D copy；回归覆盖保留两个结果及 CUDA graph 改输入重放。建议 TP/sequence-parallel 正确性测试显式检查旧结果不变，不能只看本次 collective 值。

来源：[#58154](https://github.com/vllm-project/vllm/pull/58154)（main，合入2026-10-20T18:21:38Z；`github:vllm-project/vllm:pr:58154`）。源码基线 `8122776fba2dcfae2973a0659db2da6ac688306b`。

**[Bugfix][LoRA] Restore LoRA state on wake-up from level-2 sleep。** level-2 sleep 会丢弃 weights pool，LoRA slot、punica metadata 与 TP logits 映射此前也在其中，未 reload 的 wake 可能静默退化成 base 输出，TP>1还影响非LoRA请求。LoRA manager 现建在可 offload/restore 的 runtime pool，覆盖 CUDA/ROCm 的 MRV1/MRV2；代价是 level-2 sleep 把 slot 状态复制到 host。XPU 不覆盖，reload/weight-transfer 仍有自身 reset 语义。建议 RL 原地权重更新及 sleep_preserve_parameter_names 流程做 wake前后 token/slot对照。

来源：[#62358](https://github.com/vllm-project/vllm/pull/62358)（main，合入2026-10-20T06:16:46Z；`github:vllm-project/vllm:pr:62358`）。源码基线 `86129933fb2ddc5f1e4d9ce32c4062277c7ac2af`。

**[Quantizaiton][Feature] Pluggable KV Cache Data Types。** KV cache dtype 从闭表扩展为平台注册协议 `KVCacheDTypeHandler`：激活平台时注册名称、storage torch dtype 与 quant mode，配置/CLI/attention selector 读取同一可变注册表。`KVQuantMode.CUSTOM` 表示量化但由后端自管 dispatch，不能当成已支持通用 FP8/per-token-head kernel。注册发生在分配前，重复注册会警告并覆盖；50文件涉及下游 consumers，本轮只核查注册、选择与测试契约。建议插件实现者同时验证存储布局、kernel选择和晚注册可见性，不把“接受字符串”当成kernel支持。

来源：[#52670](https://github.com/vllm-project/vllm/pull/52670)（main，合入2026-10-20T02:03:14Z；`github:vllm-project/vllm:pr:52670`）。源码基线 `9499c949be17a8a10ac4176a33843e82469d2ff5`。

**[Bugfix] Make cooperative TopK exact on candidate overflow。** Cooperative TopK 的阈值粗桶若超过候选容量，原先截断会漏选本应进入TopK的FP32分数。现在在 scatter 前检测溢出并走 exact radix recovery/rescan，正常桶沿原快路径；测试覆盖2048/2049边界、集中分布与特殊值。作者B300微基准显示恢复有额外成本，250K宽、64行、k=2048的受控边界约40→54µs。该正确性修复没有宣称提升真实模型准确率；建议稀疏attention以选值/唯一性对照torch.topk，并测溢出率。

来源：[#59645](https://github.com/vllm-project/vllm/pull/59645)（main，合入2026-10-20T22:06:56Z；`github:vllm-project/vllm:pr:59645`）。源码基线 `612e928016e2cfe28da95fc03ed894aceaf18299`。

**[Bugfix][DeepSeek V4] Load redundant EPLB expert slots。** NVIDIA DeepSeek V4/V4.1 的 expert mapping 未传 redundant expert 数，启用 EPLB 后副本槽可能保留未初始化内容。现把 `first_layer.ffn.n_redundant_experts` 送入 mapping，使 FusedMoE load_weights 填充副本；deep_gemm_mega_moe 原已加载，ROCm不在此修复范围。作者4×B300检查320副本槽从全部不符到全部相符，但250题GSM8K并未上升，提示短评测不敏感。建议直接验证副本checksum并强制路由/重平衡。

来源：[#59611](https://github.com/vllm-project/vllm/pull/59611)（main，合入2026-10-20T21:16:57Z；`github:vllm-project/vllm:pr:59611`）。源码基线 `9d4192aaa1071e64553412f874d2159d6a2f1522`。

**[Quantization] replace_parameter keeps the parameter object by default。** 权重后处理重新注册 Parameter 会丢失类和 loader 属性，影响 RL refit/reload。`replace_parameter` 现在默认保留对象；shape/stride/dtype/device相同则copy并保持地址，否则重绑.data；meta→实tensor与 requantization source 等例外仍新建。变换先作用于属性副本，保留 checkpoint 的维度标签。这只是三阶段改造第一步，不代表所有删除/重命名参数已解决；建议量化 reload同时核对对象身份、layout tags与 graph地址。

来源：[#62326](https://github.com/vllm-project/vllm/pull/62326)（main，合入2026-10-20T06:16:32Z；`github:vllm-project/vllm:pr:62326`）。源码基线 `b44446c1806f7ff57dbf21526074d6cd5cedaa2c`。

**[RL] Send NCCL weight metadata once and reuse it。** NCCL RL权重同步无需每轮重复传约上万项 names/dtypes/shapes。发送端比较本轮参数 metadata，未变时发送空 update-info；worker按model复用上次列表，重初始化engine会清空，target/draft分别缓存。数据张量仍逐轮同步，首轮空列表仍不更新；建议动态参数集合、target+draft与失败重试覆盖 metadata状态，不将此项当作减少权重传输字节。

来源：[#35911](https://github.com/vllm-project/vllm/pull/35911)（main，合入2026-10-20T06:21:58Z；`github:vllm-project/vllm:pr:35911`）。源码基线 `fce64bc5c2c939a4521bfb9b462c8c74c1c13893`。

**[Feature][Quantization] Gather MiMo fused-QKV FP8 rows exactly by default。** MiMo fused-QKV checkpoint TP 与运行TP不同，原路径常经反量化→重新量化。满足 checkpoint TP 可整除运行TP及block条件时，现在直接按原bytes/scale收集：整block做置换，跨block的K/V chunk保留同组并补零到block边界，forward按对应layout拆分。不满足条件保留requant fallback。该路径要消除重分片引入的额外量化误差；建议同时比较weight/scale字节及输出，不只检查shape。

来源：[#59053](https://github.com/vllm-project/vllm/pull/59053)（main，合入2026-10-20T03:10:55Z；`github:vllm-project/vllm:pr:59053`）。源码基线 `3b518982806cacf5346c04fdcab87e955437d017`。

**  [Bugfix][DCP] Support strided LSE in direct A2A reduction。** 显式 `VLLM_USE_DIRECT_DCP_A2A=1` 下，转置 MLA LSE 曾因要求 packed heads 被拒。direct A2A kernel现在同时读取 token/head stride，以真实stride寻址，避免 contiguous copy；reduction算术与默认backend未改。作者只在特定DeepSeek-V2-Lite长上下文负载观察收益，高并发结果不确定；建议 strided LSE 与空KV对照普通reduction后再开。

来源：[#60373](https://github.com/vllm-project/vllm/pull/60373)（main，合入2026-10-20T02:09:39Z；`github:vllm-project/vllm:pr:60373`）。源码基线 `e10ecb6bd65b94e7f281bd3f9d85fd440d7dd72b`。

**[Core] Don't apply spec decode async scheduling fallbacks with MRV2。** MRV1 的 speculative async-scheduling 限制错误套到 MRV2，会关闭 ngram/suffix 或 unpadded drafter 等组合。现在默认fallback和显式enable校验共用 V1-only incompatibility，MRV2不再继承该限制；executor、pooling及其他硬件限制仍独立生效。建议升级时检查实际生效配置和吞吐，而非假设命令行启用即被采用。

来源：[#62521](https://github.com/vllm-project/vllm/pull/62521)（main，合入2026-10-20T17:41:07Z；`github:vllm-project/vllm:pr:62521`）。源码基线 `f1dbfef79973d0f8d05be56e8614db0409e3d7d3`。

**[Bugfix] Reject mixed structured output backends during validation。** structured-output auto可逐请求解析到不同backend，但V1 engine只持一个实例，后续请求曾用错grammar backend并500。frontend记录首次选择并在常见路径入场400，engine-level manager再做权威一致性guard，覆盖多API-server/DP进程局部配置无法共享的情况。并未增加多backend并用能力；建议生产显式指定backend或限制schema集合。

来源：[#49625](https://github.com/vllm-project/vllm/pull/49625)（main，合入2026-10-20T18:13:14Z；`github:vllm-project/vllm:pr:49625`）。源码基线 `88077b746cb21e1ad5b9f25d9cae613814e892c3`。

**[Rust Frontend] Reject silently dropped request fields on chat, completions, and generate。** Rust frontend对未实现的非默认字段由静默忽略转为明确400，例如generate stop字符串/text输出、chat return_prompt_text/return_weight_versions；null、缺省及支持的Python默认值保留兼容。结合 #62642，prompt truncation现在按context长度预算，-1保留max_model_len−1，不再先为max_tokens预留空间。建议Python→Rust切换时做请求契约回归；这批改动不代表Rust已实现被拒功能。

来源：[#62434](https://github.com/vllm-project/vllm/pull/62434)（main，合入2026-10-20T14:18:12Z；`github:vllm-project/vllm:pr:62434`）。源码基线 `255e3a310c1b72a8498ffeb99367cd748d2a1688`。

### verl-project/verl

**[algo, fsdp, megatron, veomni] feat: add tail-aware top-k KL for on-policy distillation。** OPD 新增 `forward_kl_topk_tail`：把teacher top-k以外词项合成一个tail类别，在原top-k项上补tail KL，避免截断目标为负后clamp丢梯度。沿用teacher payload；Megatron TP采用本地vocab shard解析梯度，不收齐全词表。源码只对student tail的log用epsilon保护，teacher零tail按连续极限贡献0，并拒绝破坏分布的逐项logprob clamp。目标是K+1粗粒化KL/full-KL下界，接近零tail时epsilon改变理想目标；作者主张训练效率，非新损失发明或更高最终上限。建议对照 loss/梯度、tail_mass_eps与等step学习曲线。

来源：[#7855](https://github.com/verl-project/verl/pull/7855)（main，合入2026-10-20T01:53:15Z；`github:verl-project/verl:pr:7855`）。源码基线 `5a8dd519ab32bfd36d2509a137f830ba79e89c54`。

**[megatron] fix: leave HybridDeviceOptimizer state untouched in optimizer offload (#8237)。** Megatron HybridDeviceOptimizer本就按比例固定CPU/GPU状态，phase-switch helper再把GPU fused optimizer状态搬到CPU会崩溃。`offload_megatron_optimizer`现跳过HDO状态迁移，正常optimizer仍按原路径；master/copy参数helper照常调用，不能读成整个offload函数无操作。作者真实HDO复现与fraction0/0.5回归通过；建议分别覆盖offload与checkpoint resume（后者另有修复）。

来源：[#8286](https://github.com/verl-project/verl/pull/8286)（main，合入2026-10-20T08:06:50Z；`github:verl-project/verl:pr:8286`）。源码基线 `472dd2153a55f42a6683c017210bd6784da4690b`。

### sgl-project/sglang

**[Fix] eplb: Fix Qwen3 MoE output corruption after EPLB rebalance。** Qwen3 MoE `forward_normal` 开EPLB后未把expert-location dispatch送给TopK，重平衡后logical id和physical位置脱节，长输出尾部可损坏。现在按layer构造 `ExpertLocationDispatchInfo` 并传入TopK。PR提供4×A100/8×L20复现与输出截图，没有新增unit或系统化准确率结果；建议测试必须跨越 rebalance iteration并核查尾部生成。

来源：[#39580](https://github.com/sgl-project/sglang/pull/39580)（main，合入2026-10-20T10:00:40Z；`github:sgl-project/sglang:pr:39580`）。源码基线 `9db58ab547516435b544ae050dd6c2f91b52bcf3`。

**[Fix] moe: Map expert ids for layers whose runner needs local ids under EP。** global配置写FlashInfer并不代表每层实际runner相同：FP8 draft可能fallback到Triton，原dispatcher仍传global expert id，EP下越界或跑错expert。现在把实际runner交给dispatcher，只在需要时开启local-id映射；LoRA+EP遇该重映射组合明确拒绝。最终stack还有独立per-layer契约重构，本文只核查本PR的一向映射规则；建议混合NVFP4 target/FP8 draft的EP部署检查实际runner而非全局flag。

来源：[#44480](https://github.com/sgl-project/sglang/pull/44480)（main，合入2026-10-20T23:07:49Z；`github:sgl-project/sglang:pr:44480`）。源码基线 `e2e1a97fc3e057f00ba464d49b46e723403811e6`。

**[Fix] [NVIDIA] modelopt-nvfp4: Keep MoE global scales in gate/up order。** ModelOpt NVFP4 gate/up全局scale与物理weight顺序混淆的同日修复已被替换：最终方案在 `_load_w13` 重排weight和block scale，却保持全局scale为canonical [gate,up]，并保留SwiGLU beta的up-alpha校正。#44858方案已回退，不能并列当两项独立增量。作者CPU scale测试20过，完整Blackwell路径未全部验证；gate/up不同scale且共用clamp时仍有精度警告。建议对非相等scale checkpoint跨backend/TP/reload回归。

来源：[#45289](https://github.com/sgl-project/sglang/pull/45289)（main，合入2026-10-20T23:49:14Z；`github:sgl-project/sglang:pr:45289`）。源码基线 `5b1dbc2c80e59610e1d3189fbb7c6f1583e35ebb`。

**[Fix] [SM120] quantization: Requantize block-FP8 MoE weights to UE8M0 for DeepGEMM。** SM120曾对所有block-FP8跳过UE8M0 requant，DeepGEMM grouped-MoE却需要该scale布局，MTP draft可在首个graph capture报CUDA719。现在显式传 `grouped_moe=True`，只给MoE恢复requant，dense仍走CUTLASS/Triton FP32 scale。PR的端到端0.976 GSM8K同时使用另一TP修复，不能归功此项单独；建议SM120草稿专家同时验证scale、TP与runner。

来源：[#43556](https://github.com/sgl-project/sglang/pull/43556)（main，合入2026-10-20T02:06:22Z；`github:sgl-project/sglang:pr:43556`）。源码基线 `67d9ee156cd01d5895967198730b654f428fc414`。

**[Fix] deepseek-v4.1: Keep the padded query buffer inside decoder tail graphs。** DeepSeek-V4.1 FlashMLA对query补到64 heads时，按row bucket替换q_pad_buffer会释放已capture地址，较大bucket graph重放可写入复用内存/flush后MMU fault。现在metadata只保留只增不减buffer，各调用取slice；capture后禁止增长并assert。增加常驻最大buffer（作者示例2048行TP8约128MiB），换取地址稳定。TP1、trtllm和flashmla_mega无该padding路径；建议不同bucket+flush_cache连测。

来源：[#45301](https://github.com/sgl-project/sglang/pull/45301)（main，合入2026-10-20T05:41:54Z；`github:sgl-project/sglang:pr:45301`）。源码基线 `472f0d540bdbbd1c51eace905780cd31752119b5`。

**[Perf] prefill-cuda-graph: Make prefill shared-read completion explicit and default-on。** prefill shared-read-done默认开启，overlap下后续对req_to_token/full_to_swa的写可以只等metadata准备，不必等整段forward。有效前提是event后不再读live表，必要时backend先snapshot；CUDA的trtllm_mha及受限DSv4路径获益。DSv4.1低比率indexer、HiSparse、DSPARK、部分CP/request-window/graph、HiCache/LMCache等仍保整forward barrier；显式开flag也不能绕过安全guard。建议按backend和cache组合观察fastpath日志并跑输出对照。

来源：[#37958](https://github.com/sgl-project/sglang/pull/37958)（main，合入2026-10-20T23:12:56Z；`github:sgl-project/sglang:pr:37958`）。源码基线 `bf145750a65245802dd21f3b850a604635181532`。

**[Feat] KVShard: PD transfer and e2e test。** KVShard打通PD传输：prefill各rank仅发送自己拥有的logical pages，decode从整个shard group拉取；owner取logical_page % N，wire值已是owner-local page id，不依赖分配位置。MLA按attention-TP与GQA按prefill-CP分别有注册E2E，作者GSM8K 0.82/0.935；部分MLA要求P/D attention TP相等，decode DCP明确不接受sharded prefill。建议先验证rank覆盖/空stripe/部分tail page，再评价内存收益。

来源：[#38473](https://github.com/sgl-project/sglang/pull/38473)（main，合入2026-10-20T14:48:44Z；`github:sgl-project/sglang:pr:38473`）。源码基线 `eb4b167d2749840eae18167fe9e0a608308e45aa`。

**[Fix] radix-cache: Separate logical and reclaimable value lengths in tree components。** 统一radix树把逻辑索引长度与真正可回收pool行数分开：Python/Rust Full/SWA/Mamba账本改读reclaimable长度，负delta拒绝且保持计数，释放backing但未删除节点也算eviction进展。意在防止未来共享/已释放backing时admission夸大可用内存；当前in-tree组件两种长度相等，不能宣称现有模型因此省显存或提速。建议新cache组件同时测ledger与allocator实际归还。

来源：[#45238](https://github.com/sgl-project/sglang/pull/45238)（main，合入2026-10-20T20:24:21Z；`github:sgl-project/sglang:pr:45238`）。源码基线 `0b536ca112652008d5fa8766df7258b0700109f3`。

**[Fix] muse_spark: support unified KV memory with TRT-LLM。** Muse Spark unified KV pool按token行交错layer，旧fused decode/verify writer要求连续layer行，会在graph capture拒绝。unified模式改走已有QK norm/RoPE准备和backend的strided writer；target FP8与BF16 DFLASH draft分池，防dtype错配。作者4×B200完整GSM8K 1285/1319，且固定输入graph replay逐位对照通过；fallback多次kernel启动更慢，HiCache等组合未覆盖。建议把统一池容量收益与额外writer开销一起测。

来源：[#45356](https://github.com/sgl-project/sglang/pull/45356)（main，合入2026-10-20T19:54:54Z；`github:sgl-project/sglang:pr:45356`）。源码基线 `cfec1b91bb33f3267d1811680b51fd581e0df458`。

**[Feat] sgl-router: Route prefill buckets by uncached tokens。** 实验Rust router新增默认关闭的 `--route-by-uncached-tokens`，配cache_aware/bucket-config：warm请求按全prefill可复用最长prefix后的新增token量匹配bucket，cold/indexer无答复走全worker负载选择，bucket全忙时可spill到真正idle DP rank。作者MI355X agentic trace比较同时改变prefill worker布局，tok/s/GPU 47646→50778且p95 TTFT18.46→15.02秒，不能当flag单变量收益。建议固定拓扑A/B，并把chunking中未排队rank计作busy。

来源：[#45103](https://github.com/sgl-project/sglang/pull/45103)（main，合入2026-10-20T12:56:15Z；`github:sgl-project/sglang:pr:45103`）。源码基线 `df13d5dce31abfeff0e701a62e8c39a136758e6c`。

### pytorch/torchtitan

**[rl] Support context parallelism in the RL trainer。** RL的advantages、generator_logprobs、mask、temperature以前绕过labels的CP分片。最终merge把它们作为loss_kwargs并入 `Decoder.preprocess_inputs`，使用labels相同SPMD布局，经 `_cp_shard→context_parallel.shard_tensors`共同置换/分片，再拆出第四返回值；batcher/模型接口同步调整。PR正文仍描述旧helper及拒绝load-balancer方案，源码不支持照搬该限制。CP仍须满足所选attention/metadata及分片尺寸契约；建议多rank比较整序列梯度与masked loss，并检查自定义model preprocess返回值。

来源：[#5424](https://github.com/pytorch/torchtitan/pull/5424)（main，合入2026-10-20T01:04:51Z；`github:pytorch/torchtitan:pr:5424`）。源码基线 `9371a2ffcb8655d74e6e1c099d7ca7b756259cae`。

**[dist_moe] Use fast fused SwiGLU mega kernel for inference。** Dist-MoE把已有MXFP8 mega fused-SwiGLU路径接给RL generator：32×32 scale不变，W13 gate/up交错使SwiGLU进入W13 epilogue，FSDP刷新复用operand地址。最终API为 `mxfp8_mega_fused_inference=True`，要求MXFP8、pipeline=mega和inference runtime，不支持backward，默认staged不变。作者B200 Qwen3形状层级相对BF16 1.5–1.7×，8B200短RL run总step相对BF16 1.06×→1.10×；不能把层级倍数推为整训练速度。依赖dist_moe kernel内部未重审。

来源：[#5465](https://github.com/pytorch/torchtitan/pull/5465)（main，合入2026-10-20T21:05:52Z；`github:pytorch/torchtitan:pr:5465`）。源码基线 `ed3c713afeb21a514714ff59fec4fd4bb2d3b09d`。

**[TorchTitan] Load Kimi K3 compressed-tensors MXFP4 checkpoints。** Kimi K3可用DCP quantized HF reader读compressed-tensors MXFP4专家并反量化为BF16；adapter映射packed名字、恢复KDA padded A_log/flattened dt_bias布局，HF导出保留96-file分配而非单文件。普通导出不重新量化，text-only去掉vision/projector索引，避免空dtype header。149项针对性测试为作者结果，DCP reader内部是依赖边界；建议保存再加载同时覆盖quantized标志、空DTensor shard和纯文本/多模态索引。

来源：[#5337](https://github.com/pytorch/torchtitan/pull/5337)（main，合入2026-10-20T06:29:15Z；`github:pytorch/torchtitan:pr:5337`）。源码基线 `7d563223d4ee10c0b8a7fc70d0cd119bcbcff93f`。

**[kimi] Use the published rms_norm_eps for Kimi K2.x decoder-layer norms。** Moonlight/Kimi-VL/Kimi-K2.5 decoder block曾继承DeepSeek V3的1e-6 RMSNorm eps，而发布配置为1e-5。builder新增norm_eps并由Kimi传1e-5；MLA q/kv norm仍1e-6，DeepSeek默认不变。作者在真实embedding输入看到可观归一化差异，但未给整模型质量提升结论；建议checkpoint兼容测试核对每类norm而非只看final norm。

来源：[#5548](https://github.com/pytorch/torchtitan/pull/5548)（main，合入2026-10-20T02:30:45Z；`github:pytorch/torchtitan:pr:5548`）。源码基线 `d06d37f53cb206ad008fb5869ae36a5c46adba50`。

**[attention] Flex attention: recompile limit 16, and fail instead of running eager。** FlexAttention单进程共享compile对象，Muse训练/prefill/decode需要超过默认8个graph，原先静默eager fallback导致慢与capture失败。现在默认和deterministic compile统一fullgraph=True、recompile_limit=16，第17个graph会报错。结合 #5433，Muse三个既有local regions默认启用到base/debug/RL recipes；其融合改变求和/FMA顺序，非逐位相同，LoRA SFT显式覆盖列表不自动受益。建议混合shape压测关注编译上限和数值容差。

来源：[#5442](https://github.com/pytorch/torchtitan/pull/5442)（main，合入2026-10-20T15:53:21Z；`github:pytorch/torchtitan:pr:5442`）。源码基线 `aabe6eff1e6e35379e0c3a3d5041981a1ab60d5f`。

**[profiler] add opt-in Cuspy profiling backend。** 可选Cuspy替换Kineto的CUDA活动采集，加入PM sampling、环境counter、同步/graph依赖与可选Perfetto protobuf；torch/libcupti无能力时启动明确失败。#5136进一步把CUPTI annotation_config传到forward-backward/optimizer graph capture，重放kernel保留mark_kernels；event-node关联需按exec graph id。首次capture起CUPTI subscription持有至run结束，与Nsight Systems/dynolog同进程附加冲突；建议选择一种采集方案，并核查PM warmup+active在lookback内。

来源：[#5525](https://github.com/pytorch/torchtitan/pull/5525)（main，合入2026-10-20T03:37:48Z；`github:pytorch/torchtitan:pr:5525`）。源码基线 `789740c02578d288ee6e5a5651c6f96162bf9b84`。

### 组合项补充基线

- [vllm-project/vllm #62642](https://github.com/vllm-project/vllm/pull/62642)：main，合入2026-10-20T14:19:39Z；`github:vllm-project/vllm:pr:62642`；源码基线 `97a82d3c92e65429f6571df01253e573f90fbf06`。
- [pytorch/torchtitan #5433](https://github.com/pytorch/torchtitan/pull/5433)：main，合入2026-10-20T15:50:02Z；`github:pytorch/torchtitan:pr:5433`；源码基线 `39bd2054d9c4fcbcabc343d262abad85f19020fb`。
- [pytorch/torchtitan #5136](https://github.com/pytorch/torchtitan/pull/5136)：main，合入2026-10-20T16:17:28Z；`github:pytorch/torchtitan:pr:5136`；源码基线 `b93d5c0cf518b4404ceab998adc529fd60ebf301`。

### 分支、发布与旧缺口

Megatron的 [#8364](https://github.com/NVIDIA/Megatron-LM/pull/8364) 已把CheckpointWithoutOutput改造列为core_r0.20.0的cherry-pick，另有 [#8349](https://github.com/NVIDIA/Megatron-LM/pull/8349) 的GDN graph-state回补；这是release分支合入事件，不等于新release发布。`dev` 的[main同步#7960](https://github.com/NVIDIA/Megatron-LM/pull/7960)与[optimizer staging #8233](https://github.com/NVIDIA/Megatron-LM/pull/8233)也不当成main新增功能。TorchTitan [#5568](https://github.com/pytorch/torchtitan/pull/5568) 只在release/0.4把版本改为0.4.0rc1，公开release collection仍最新v0.3.0。slime主窗0 merged PR、0 main commits；回溯4个PR均先与历史去重，未重报。

**旧待办补录：Megatron运行时初始化顺序。** [#8272](https://github.com/NVIDIA/Megatron-LM/pull/8272)（main，2026-10-18T20:12:35Z，`ce852bfb93f364f64a556602fc71a5d9ed2c377e`；`github:NVIDIA/Megatron-LM:pr:8272`）昨日库存已列pending、未入正文。本轮打开关键patch：`initialize_megatron`先经过run-config校验，再构建tokenizer/依赖其vocab的writer等runtime服务，随后才distributed/RNG setup；集中保留显式/恢复vocab，并传递MiMo的build_tokenizer=False。其意义是无效配置在服务副作用前失败、入口保持单一初始化边界；作者针对性405测试是外部结果，未重跑。建议自定义training/inference入口检查是否还重复构造服务。

**P2P consumer寿命仍未闭合。** 今日重开 [issue61532](https://github.com/vllm-project/vllm/issues/61532) 与 [PR61667](https://github.com/vllm-project/vllm/pull/61667)，仍open、merged_at=null、head `4fd16ddb155fd0f7e676dece9abd6aa9209b2b1a`。本窗main `532a7d19d2ae70b1fd9188d82620d26f028564fb` 的 `ClientRole.collect_results` 仍在10秒abort-ack超时后报success=False并移除load；无法据此证明远端WRITE已停止。本轮Mooncake #62282是另一个connector的相关正确性进展，不能关闭P2P quarantine、RDMA/serving验证待办，也不重报PR现有TCP probe为新进展。

**覆盖边界。** all-base merged PR主窗229条（21/82/5/96/0/25）、其中main224；回溯644条、其中main630。main按commit时间枚举637条，主窗220条，满100项页均继续分页至短页；commit时间与合入事件时间不同（Megatron尤其明显），不互相代替。release最新10条（Titan全部5），最旧返回早于回溯起点；该范围不能排除任意历史创建、迟发布的release。库存完整不等于全部语义审阅，未选实质候选与历史10-05→10-08等缺口在core-coverage.json保留，全球完整覆盖水位不推进。当前没有访问失败。

<a name="engineering"></a>

## 扩展开源工程与模型接口修订

### DeepSpeed 0.19.8：把Muon、AutoEP和多项梯度/通信修复送入正式版本

**DeepSpeed；2026-10-20T15:21:47Z正式发布，tag0eb3a8bcc6d9a8775478e5377482c4bf2dd23c1c。** [完整release](https://github.com/deepspeedai/DeepSpeed/releases/tag/v0.19.8)。本轮增量是**版本可采用性**：此前数周main上的per-head Muon、ZeRO-3每optimizer step只执行一次Muon、CPU offload/gather-buffer约束、AutoEP共享DeepEP buffer/重放dispatch layout、AutoTP词表并行与chunked causal-LM loss等进入同一正式版本；不把这些旧PR重新包装成10-20首创。

实际升级价值还包括ZeRO-1/2 CPU非连续reduction静默丢梯度、FP32 accumulation被降精度、FP16 MoE grad norm误用权重、ZeRO-3 coalesced gather未等完所有异步请求、FPDT GQA K/V梯度buffer等修复。以上为release列明范围，本轮未逐一复跑GPU/NPU训练。**迁移前查配置**：release也列出移除elastic training、legacy curriculum、PLD、旧compression/LoCo Zero++、APEX AMP、graph harvesting等；不能只当兼容性补丁。建议依赖Muon/AutoEP或offload的任务**考虑升级验证**，固定旧checkpoint做梯度、optimizer step与恢复对照，尤其检查已弃用选项；没有跨版本同配置E2E速度证据，不给统一加速倍数。

稳定ID：`github:deepspeedai/DeepSpeed:release:v0.19.8`。

### PyTorch FSDP2原生collective copy：减少中间buffer，同时公开不利形状

**PyTorch；10-20默认分支新增，非已验证随2.15正式wheel发布。** [PR200034/准备与执行分离](https://github.com/pytorch/pytorch/commit/ffd1d24d04e6c5d898a9c1c2d64c52e4db83565e)、[PR197701/原生copy opt-in](https://github.com/pytorch/pytorch/commit/d3c397fef74cebde85dde5ca5e6638a2cb2da9c5)。先把AG copy-out/RS copy-in准备与执行分开，再导出`torch.distributed.fsdp.experimental`原生callback；RS已有公开`FSDPModule.set_reduce_scatter_input_fn`可opt-in，AG在此快照尚无公开安装setter，测试通过private `FSDPParamGroup._all_gather_output_fn`，需等待后续layout API或自行承担private hook兼容性。默认保持旧路径。`AllGatherInput(dim=...)`明确量化payload的拼接维，不能依赖形状猜测；payload数量、尺寸/dtype变化会提前校验，旧private callback签名可能TypeError。

原生copy能省contiguous tensor中间buffer；混合dtype AG按byte拷贝，payload小于缓存output回退默认路径，非零维分片的非连续gradient仍需重排。作者在**一张B200、world-size-8布局的copy微基准**中，[2,8192,4096] Shard(1) copy-out的GPU/CPU时间为默认0.53/0.41倍，copy-in GPU约0.38–0.45倍，部分布局峰值省24–830MB；**不是8卡完整训练加速**。Shard(2)专家copy-out反而1.7–1.8倍GPU时间、最高5.6倍CPU时间；多outer rows的Llama8B block也有CPU回归。建议**考虑复现**真实专家形状、payload dtype和AG layout后再opt-in，尤其测host调度和总step时间。

### PyTorch编译、checkpoint和退出：三类会影响训练可用性的修复

以下均为10-20默认分支可定位commit，本轮未验证正式wheel包含，也未独立跑CUDA/ROCm测试。

| 变化与来源 | 证据、边界与实践建议 |
|---|---|
| [CooR RNG/196152](https://github.com/pytorch/pytorch/commit/7ffd47d270087d6dbafb876c0549d949aabf0a5a) | `compile_on_one_rank`下RNG device还是FX node，旧替换会直接把Node传给要求Device的seed op而失败；现在仅在需真实device处解析、替换graph保持rank无关，并归并同设备seed/offset。覆盖rand/randn/randint/dropout；**考虑复现**跨rank图缓存与RNG，而非从编译通过推断随机流完全等价。 |
| [低精度zero-mask/201029](https://github.com/pytorch/pytorch/commit/52e81ec19967ac04c98cc89ff55d91d3827e955c) | FP16/BF16显式rounding cast令`PreservesZeros`收到NotImplemented，原先后续读expr崩溃；修复让零保持分析保守退回。影响低精度prologue融合matmul编译，**建议跟进**已有InductorError最小复现；没有声称改变训练数值或加速。 |
| [MXFP4 DCP/201291](https://github.com/pytorch/pytorch/commit/f21bd8f7b7494c83a7f01da3a761c6c67865b0b6) | DCP reader识别compressed-tensors MXFP4 `uint8 weight_packed/weight_scale`并解量化，重写size、chunks和目标dtype，允许Shard(1)边界落在32值group中间。对NVFP4/int4/3D MoE等非匹配格式保持raw；DCPv2 resharder不在范围。作者CPU/gloo合成fixture有对照，多GPU例有跳过；**考虑复现**跨文件scale和真实分片，不能说通用4bit checkpoint已支持。 |
| [communicator退出/200673](https://github.com/pytorch/pytorch/commit/adc4a4125deaced07f7930c6b9707660f97feefc) | NCCL2.31开始graceful finalize可能等同机rank barrier，逐communicator且顺序不同会互锁；现在用NCCL group并发finalize。abort后再best-effort释放TorchComms/NVSHMEM，按对象去重，不在abort前等挂起kernel。**建议采用前回归**多PG/P2P和故障退出；标题ROCm不意味着根因只影响ROCm。 |

### Transformers：专家并行梯度与长上下文编译值得单独回归

**Hugging Face；10-20 main增量，最近核到的正式release仍5.20.2，不能写成已发布5.20.3。** [EP梯度/49710](https://github.com/huggingface/transformers/commit/9f5750c67ad4c2e077dd3883d20503f3d77b2fe5)：PyTorch2.7–2.10的functional `all_to_all_single`没有autograd kernel，dispatch/combine导致expert和hidden state无梯度；改用autograd版本并让wait传梯度，EP backward测试逐参数对unsharded reference。本轮静态核patch，未独立分布式复跑。**建议训练使用者优先检查**torch版本、实际梯度存在性，loss能下降不足以排除此类错误。

[LongRoPE/49555](https://github.com/huggingface/transformers/commit/823a6ea6d8d1fbc2c598e64fa1e1b8ac9cab8224)把long/short频率各预计算，再用`torch.where`替代基于position tensor的Python分支，修复fullgraph失败、错误cache key和long buffer设备问题；MiniCPM3配置的可选factor也不再无条件读。测试明确跨原上下文阈值才走long branch；不把其它dynamic RoPE修复包含进来。[BLOOM/49481](https://github.com/huggingface/transformers/commit/71727af2206ef9d8ba5d028592bf2573786c1b5a)修静态cache/compiled generation的mask路径，[Compass/49715](https://github.com/huggingface/transformers/commit/d4fcbccd0966ace9505b9cbd3041d28ad822d8a7)把CohereCompass text RoPE对齐Qwen3-VL interleaved M-RoPE。建议**考虑复现**真实长分支、padding、batch与cache重用，不能只跑短prompt。

线性/混合attention缓存改为`get_recurrent_state/get_conv_states/is_recording_past`一组显式getter并迁移多模型（[线性cache迁移/49375](https://github.com/huggingface/transformers/commit/3bb8b72786c7cb5abc9bd05c2e1f451e44dcc9b3)）；`rope_scaling` setter走弃用并规范legacy key（[RoPE配置/49515](https://github.com/huggingface/transformers/commit/55928cfcc4f1b8253cd7e6e06ed2fed19bd0d2b0)）。这是集成契约变化，**继续跟进**自定义Cache/模型wrapper兼容；不是新长上下文能力或质量提升。

### TRL：兼容vLLM0.32生成logprob协议，并允许实例化异步GRPO模型

**10-20 main。** [vLLM0.32/7799](https://github.com/huggingface/trl/commit/2b0826efca107358cd309214f3889724e9a6f11c)适配`/inference/v1/generate`的新logprob wire shape：从字符串token占位符改为整数token_id，并保留sampled token、排序候选、把NaN转None；同步放宽受支持范围至0.22–0.32；[实例化model/6676](https://github.com/huggingface/trl/commit/4b81b9215a7fc0497b8d234de3d524fd9cdc9851)让AsyncGRPOTrainer接受已实例化model，切换到tree-attention packed训练后端，从config取服务模型名；此时model_init_kwargs会告警并忽略。二者是训练—rollout接缝的可用性变化；本轮未做多卡异步训练，亦未将0.32兼容误写成新GRPO算法或全模型兼容。建议**考虑复现**已有PEFT、量化、server/colocate组合的weight sync和logprob一致性，再升级实验环境。

### TensorRT-LLM KVCM V2：跨turn复用、保留优先级和外部缓存计账一起补齐

**10-20 main，以下不因源码version升1.4.0rc2而自动成为已发布wheel。** [生成KV复用/19710](https://github.com/NVIDIA/TensorRT-LLM/commit/fac6d3f794a1d1b2094d16c43f494c5477466c2e)把generation资源在response/transfer可能结束request前完成更新，已接受草稿token完成relocation后提交可复用块，令后续turn可复用生成上下文；使用零拷贝beam0 token view。该增量只开在block reuse/ALL_REUSABLE、无connector与sharing/lender、无compression-history、非draft、beam宽1且非dummy/context-only的路径；connector仍prompt-only，最后一个仅采样而尚未forward的token没有KV、不会提交。[retention/20086](https://github.com/NVIDIA/TensorRT-LLM/commit/41913522c7f9f085c022421fb6183bd7592e076f)实现原生page retention：prompt范围及decode的priority/duration，过期且无consumer才回默认优先级，partial copy不覆盖既有consumer claim，落地页继承source retention；冷层offload最小priority默认30。要求完整prompt length；**不支持每request GDS/POSIX transfer mode或directory**，offload tier由manager配置。多layer-group且会传KV的connector收到显式retention配置会请求级拒绝（priority列表无法按group表达）；capacity-only connector例外，原生无connector retention仍支持多group。

[connector计账/20180](https://github.com/NVIDIA/TensorRT-LLM/commit/2706f4dfb611063c39e29ef5bf8ad9401a1a3556)另把connector已加载token/完整block从miss重记为reuse，部分仍需forward写的block保留miss，并单列connector统计。**计数时点是load被接受、异步传输完成之前**；缓存命中指标不能当DMA完成或request成功证明。建议**考虑复现**多turn、取消、beam/prompt尾块、speculation与offload组合，监控connector与本地reuse分账；没有E2E吞吐对照则不量化收益。

### TensorRT-LLM GLM与CUTLASS：量化scale必须跨EP一致，后端能力也需早检查

[全部专家scale/20547](https://github.com/NVIDIA/TensorRT-LLM/commit/9f83fc48056dd7f1652bcfc449b4b79c5552f3fd)在GLM5-next NVFP4 loader中读取**全部专家标量input_scale**，权重仍只读本地专家，因为token在dispatch前量化，layer-wide scale必须所有rank一致。[GLM融合/20389](https://github.com/NVIDIA/TensorRT-LLM/commit/5cf7459e972090e576c1541e532d27056ebf1c15)共享KDA verify offsets、各阶段融合mHC，router使用checkpoint BF16 weight做GEMM且输出/校正bias保持FP32；不能从BF16输入推导路由逐位一致或无质量回归。本轮未重跑模型评测，建议**优先验证**EP切分与投机verify下的路由/输出。

[CUTLASS边界/19753](https://github.com/NVIDIA/TensorRT-LLM/commit/fe92fb923f12f98cfafc180bd9496126d2d9ef9e)恢复OSS MoE关闭时的legacy host/archive连接，并把`oss_cutlass_moe`纳入环境fingerprint与eligibility。legacy路径对MXFP8、部分W4A8/NVFP4、post-SiLU clamp、expert LoRA、all-to-all、unpadded output等拒绝，调用CUTLASS prefill的Blackwell后端也先检查。建议**按实际构建选项验证**，避免“后端存在”被误当所有量化组合均可用。

### vLLM-Ascend：持久KV池布局修正是升级事项，不能只重启worker

**10-20默认分支。** [KVPool物理页/19070](https://github.com/vllm-project/vllm-ascend/commit/7e081f466da1b07ac6e5b757d8610fe774cdf0e9)修复strided MLA物理页被误当跨scheduler block连续component的搬运。5760-token scheduler block/128-token kernel block、512/64 NoPE/RoPE场景，旧layerwise描述遗漏末5个物理页（640token）；现在按物理block分段保留stride及token区间，仅符合条件才coalesce。

官方新增迁移说明：**旧持久pool对象总字节数相同仍与新表示不兼容，读写端必须一起升级，并清理受影响entry或新namespace/cache_salt；仅重启worker不会失效持久对象**。本轮未连接AscendStore做E2E。建议使用该MLA KVPool组合者**优先采用修复并执行缓存迁移回归**，测bulk/layerwise与partial restore，不从新descriptor正确推断后端传输已验证。

### vLLM-Ascend：MLA分段归并、图replay和A5路径的新边界

| 10-20 main来源 | 机制增量与边界、建议 |
|---|---|
| [16路归并/19012](https://github.com/vllm-project/vllm-ascend/commit/8237ee6d690d1399bc6eb48ef5977ac2bd470f3f) | CANN `npu_attention_update`最多16对output/LSE；达到16先归并并保留aggregate，另存LSE anchor与relative量，避免大公共offset吃掉细微权重。是长prefill正确性/算子准入修复，**考虑复现**超过16 chunk及大LSE场景，不能只测两段。 |
| [PCP图padding/19192](https://github.com/vllm-project/vllm-ascend/commit/e5fd02646e2c9bf68c02bceb3e193f852f84e21f) | replicated PCP请求/token容量已padding而query offset仍止于31/32的情况不再早退，FULL replay恢复canonical layout。作者A3四NPU Qwen3-8B、TP2/PCP2/DCP2/Eagle3最终文件188 UT+1 E2E；这是作者结果，本轮未跑。**建议回归**同图bucket及末尾空request。 |
| [oproj容量/19262](https://github.com/vllm-project/vllm-ascend/commit/9a885476a25489b8cbd9c02dec171a5f9c5ce7cb) | fine-grained oproj TP decode buffer按decode/cudagraph容量分配，避免max-batched-tokens240对实际容量16使通信/计算/buffer膨胀15倍；此为规模算术而非实测15倍加速。该PD consumer路径过大budget现在报ValueError，需设`max_num_seqs*(1+num_speculative_tokens)`。**考虑采用**并核配置。 |
| [FlashMLA/18742](https://github.com/vllm-project/vllm-ascend/commit/bc078fc954cd9a14c8c1d4da54ea2b28ac34d363) | main接入external FlashMLA的MRv2 decode，prefill仍FIA；DCP普通decode与causal speculative分别处理，保留stream ready顺序和graph准入，PROLOG tile固定128而非writer页长。CPU契约测试不等于新增NPU/PD E2E；query None guard、PD layout和FP16 writer仍有follow-up。**继续观察/限定配置复现**。 |
| [A5 TP校验/19240](https://github.com/vllm-project/vllm-ascend/commit/b1acd72350c30ebfbbeffee689e3ca6a44027e7f) | 移除A5只能单rank的校验，DeepSeekV4/V4.1 Flash统一TP≤8、Pro≤16，保留V1 DSA-CP例外。patch为配置准入，作者未本地NPU运行，不能写“所有TP实测通过”。**继续观察**精确pin上的硬件回归。 |

### Triton/Helion：边界条件与错误参考值会让kernel调优选错答案

**10-20默认分支，非新正式版本声明。** [loop signedness/12350](https://github.com/triton-lang/triton/commit/29ef3c3320e42aa0b5c2c9e24110b53526660f10)修窄整数for边界/step提升按**源类型signedness**扩展，uint8上界200不再变−56；全宽unsigned loop比较另属未覆盖问题。[pointer减法/12318](https://github.com/triton-lang/triton/commit/6706b886650f13b79dac00b6df007d37e93fba42)在负向pointer offset前先把窄unsigned转int64，避免回绕反向寻址；作者仅CPU interpreter验证，GPU未本地跑。[CDNA5 barrier/12342](https://github.com/triton-lang/triton/commit/249c29ead385cfbd4272d1a96c90a7516ad754a5)在CDNA5 cluster barrier前加入workgroup barrier，避免慢warp消费错误轮次completion bit使连续barrier/atomic挂起；[Gluon边界/12400](https://github.com/triton-lang/triton/commit/72bcaf23ebd396d53696255a72a33b9496422c38)修Blackwell tutorial accumulator释放顺序，attention对CTA缩放后的BLOCK_M不整除长度明确拒绝，防不同partition循环数分歧和越界。建议kernel作者**考虑采用并回归**边界trip count、尾tile和延迟注入，不能把示例修复泛化为所有attention实现。

[MoE padding/4063](https://github.com/pytorch/helion/commit/7bb9cb141d42e387267a77c491316499acb498bf)尤其值得读：MoE padding lane被夹到专家首token，旧where/load/store把旧值写回同一地址，与真实lane冲突；默认参考本身也错，autotuner因此可能淘汰正确小block而接受错误结果。改为`hl.store(extra_mask=...)`让padding根本不写，CuTe旧fold pass已正确，影响主要在Triton/interpret例子。[attention匹配/4065](https://github.com/pytorch/helion/commit/9231718756086e2c7ebc1542333b8c224afcca9a)把attention backward独立dQ/dK scale识别和fa4_alt ReLU epilogue补齐，避免落SIMT或调优全失败；作者的约2秒→0.20ms是特定旧错误fallback与融合路径比较，**不作通用FlashAttention提速**。后续[fa4列表调度/4092](https://github.com/pytorch/helion/commit/76bf63b975d48717aa9f881590a748616094f3b9)扩展fa4尾wave切分、early QK、共享output tile和handoff fence，本轮只核关键patch，未审全部autotuning配置。建议**先固定独立正确性reference再比较kernel**，并保留fallback/未覆盖family日志。

### 已有权重的重要代码、接口与评测修订

- **Ling-3.1-flash，10-20T08:23Z。** [修订](https://huggingface.co/inclusionAI/Ling-3.1-flash/commit/93e96ab91e800998388563295340a4238ff3e962)将FP8排除名`b_proj`收紧为`attention.b_proj`、移除`mlp.gate`排除；`BailingMoeV3Model.forward`仅显式要求时计算MTP，MTP不再读写主KV cache；`BailingMoeV3ForCausalLM.forward`只在有labels时计算MTP loss。本轮对照配置/源码，safetensors对象未变；是推理与训练适配修复，不是新训练checkpoint。未运行数值/吞吐验证，更新remote code时需同时固定config与权重revision。ID：`hf:inclusionAI/Ling-3.1-flash@93e96ab91e800998388563295340a4238ff3e962:fp8-mtp-fix`。
- **Ming-3.0-asr-tiny，10-20T06:27Z。** [新卡](https://huggingface.co/inclusionAI/Ming-3.0-asr-tiny/blob/3ec2edd73a93a32efe853722af2b53ac9a6dea0f/README.md)改用[SGLang-Omni cookbook](https://sgl-project.github.io/sglang-omni/cookbook/ming3_asr.html)，一GPU共置encoder/LM、`/v1/audio/transcriptions`。文档给默认greedy、最多256新token、>30秒切片拼接；`stream=true`返回完成转写及DONE的SSE，不能解释为在线音频增量识别。没有逐字时间戳，srt/vtt不支持。权重未变、未执行服务；应验证长音频截断及错误language提示，而非仅测短句。ID：`hf:inclusionAI/Ming-3.0-asr-tiny@3ec2edd73a93a32efe853722af2b53ac9a6dea0f:sglang-omni-guide`。
- **K2-Horizon-0.9B，10-20T02:54Z。** [模板修订](https://huggingface.co/IFM/K2-Horizon-0.9B/commit/7c75f2f49e74ac97c1852e62b4c860091d6a00ea)补BOS、支持`reasoning_content`/`reasoning`，历史assistant缺少think字段即报错；reasoning_effort仅接受high/medium/low，未知值不再悄悄回退。删两份冗余模板，无权重变化。建议迁移历史消息序列时补思考字段，并验证unsupported effort的错误路径。ID：`hf:IFM/K2-Horizon-0.9B@7c75f2f49e74ac97c1852e62b4c860091d6a00ea:chat-template`。
- **Nemotron-3.5-Rerank-8B，10-20T21:03/21:28Z。** [固定卡/配置](https://huggingface.co/nvidia/Nemotron-3.5-Rerank-8B-BF16/tree/8edd41b8c76c976d3588c8340d63582fa4482d17)新增`keep_in_fp32_modules=["2.linear"]`，示例固定Sentence Transformers6.1.0。新评测用同一Embed1B检索top50，作者报ViDoReV3 image NDCG@10 **67.22**、MIRACL-VISION **78.69**；所列Qwen3-VL-Reranker为66.02/75.06，但两模型保留各自默认图像尺寸（1120长边 vs1,310,720像素），不当严格同分辨率比较。权重未变，不能写能力本日提升；要复现须固定候选集合、图像预处理和FP32 scorer。ID：`hf:nvidia/Nemotron-3.5-Rerank-8B-BF16@8edd41b8c76c976d3588c8340d63582fa4482d17:fp32-scoring-eval`。

<a name="hardware"></a>

## 硬件软件协同

### TorchTPU公开发布：原生PyTorch设备、三层编译缓存及真实迁移成本

**Google×Meta；官方介绍10-20，PyPI torch-tpu0.2.0首工件上传10-19T15:39:50Z；本轮源码基线905a0dfde04a01df8ee2c100ffd656694f38df84。** [官方架构与实验](https://pytorch.org/blog/introducing-torchtpu/)、[HF集成](https://huggingface.co/blog/hf-torchtpu)、[固定README](https://github.com/google-pytorch/torch_tpu/blob/905a0dfde04a01df8ee2c100ffd656694f38df84/README.md)、[公开API分级](https://github.com/google-pytorch/torch_tpu/blob/905a0dfde04a01df8ee2c100ffd656694f38df84/docs/public_apis.md)、[PyPI版本](https://pypi.org/project/torch-tpu/0.2.0/)。Apache2，Python3.11+，可钉包0.2.0；GitHub release列表为空不等于包未发布，也不能把当前main所有API承诺给已上传wheel。

新增的实质是PrivateUse1原生ATen→StableHLO/PjRT backend，运行时不依赖JAX；支持eager和Dynamo `torch.compile(backend='tpu')`、ProcessGroup/DTensor及生态集成。Strict eager默认逐op异步提交；DeferAndFuse需显式启用，延迟图在materialization/repeated-op边界编译融合。编译缓存分进程内、同host共享内存及远端持久层，最后一层默认关闭；这会影响冷启动、跨rank重编译和重启成本。自定义Pallas/JAX kernel是另一集成入口，不否定运行时无JAX依赖。

官方披露TorchTitan训练Llama3-405B在1024×TPUv7x、8K、BF16约500token/s/chip和50+%MFU；Qwen3-235B MoE在8192×v7x、4K、BF16的Case3具体表行为27.1%MFU/31.2秒step，Case5叙述概称约28%；本报保留这处内部口径差异。Case3固定每TensorCore batch16、全局batch随1024→8192 chips放大，MFU28.8%→27.1%，是weak scaling；BF16之外grad reduction/RoPE/optimizer states为FP32。二者模型/长度/规模不同，**不能互比扩展效率，也不是对GPU的统一优胜证据**。FLUX的eager→融合/分片/定制kernel/full compile多步叠加，须分清收益归因。HF报告已运行TPU测试96.5%通过、34 issue中19已修，分母不是整个Hub全覆盖；batch64的ragged-dot解码倍率也只是同TPUv6e-8默认实现对照。

采用边界直接影响研究可比性：默认matmul precision是`medium`，FP32输入可经BF16 MXU/FP32累加；`high/highest`约3/6趟计算，并非仅改一个无代价精度开关。XLA/PjRT仍偏静态shape，bounded dynamism及更广FP8/FP4在路线图；TPU单设备只能由一个用户进程访问，**RL colocation不能原样照搬GPU多进程方案，端到端verl RL仍是后续目标**。API文档将stream/event、DeferAndFuse、precision、Muon等大量接口列experimental。建议**深入阅读并考虑小规模复现**原生训练/服务迁移，固定dtype、padding/shape桶、编译冷暖时间和loss轨迹；不将“to(tpu)”可运行等同峰值效率或逐位等价。

稳定ID：`pypi:torch-tpu:0.2.0`；`github:google-pytorch/torch_tpu@905a0dfde04a01df8ee2c100ffd656694f38df84`。

### AMD MI355X megakernel：单请求解码有可复现路径，质量和软件分支限制必须一起读

**AMD；官方博客10-20；本轮所链ATOM qwen38-megakernels基线352229d6d4a756cf58146bc5cb37519dd5d5aefd，aiter适配分支29711d1798ab756cd49570ee09b6fb709e84c999。** [技术说明与完整配置](https://rocm.blogs.amd.com/software-tools-optimization/megakernel-intro/README.html)、[固定ATOM复现指南](https://github.com/proaditya/ATOM/blob/352229d6d4a756cf58146bc5cb37519dd5d5aefd/blog_benchmark/README.md)。**这些是官方链接的专用分支/plugin，不是已验证全部合入ATOM/SGLang/vLLM release的能力**。

增量组合是持久megakernel压launch/sync、producer/consumer按数据tag握手、MXFP4权重/MXFP8算子、图内draft/verify/accept和混合attention状态回放。Qwen3.8-27B/GPT-OSS可整个forward一kernel，Flash-Next每decoder layer一kernel；后者250K稀疏attention先并行选block再合并top512。GPT-OSS在DPX+NPS2把同一物理MI355X分两逻辑die，本地放权重/KV/同步标志，独立控制试验把跨die读与local placement区分。

单物理MI355X、batch1、GSM8K128 prompts，作者最快配置约1091/750/1472token/s（依三模型）；Qwen27B约2倍对同ATOM分支MXFP4常规路径，Flash-Next约3.3倍、相同MTP；GPT-OSS最快方案1.9倍**改变drafter、precision和partition**，同DFlash时约1.6倍。只计首token之后decode；不是完整agent或prefill倍数。Qwen27B仍有LM head/drafter精度差异，不能称完全单变量。

质量边界比峰值更重要：Qwen27B MXFP4对BF16的GPQA-D低3–5pp；GPT-OSS chat格式GSM8K/MMLU-Pro接近，但5-shot raw GSM8K随attention split出现35.5–42.8%，BF16权重参考49–56%，官方明确仍调查；“每draft由target验证”不能消除量化/attention kernel误差。GPT-OSS仅greedy单请求、32K preview且不batch prefill；Flash-Next其它引擎仍planned。ROCm10需`GPU_MAX_HW_QUEUES=1`，默认4可使优化路径慢10–25%，baseline也受其影响。建议**考虑复现**低并发交互负载，连同chat/raw格式质量和服务总时长，批量吞吐场景继续观察。

稳定ID：`amd:mi355x-megakernel:blog:2026-10-20`。

### cuTile Mamba-2：前向三个kernel的分解与B300实测，补录10-19技术文章

**Jane Street×NVIDIA；原文10-19，本轮补录。** [官方教程与配置](https://developer.nvidia.com/blog/accelerating-mamba-2-with-high-performance-forward-pass-kernels-in-nvidia-cuda-tile/)、[TileGym代码入口](https://github.com/NVIDIA/TileGym)。把Mamba2 chunk前向拆为log-decay cumsum、跨chunk state与融合输出，利用MMA；Blackwell上交换MMA操作数使融合结果走Tensor Memory，转[batch,head,token]布局适配TMA，并把latency hint和tile/occupancy纳入搜索。是实现与调优方法，非新架构。

作者单B300 SXM6/SM10.3、1920MHz、1100W、268GiB，BF16输入/FP32 state；对mamba_ssm2.3.2.post1/Triton3.7.1，cuTile1.5.0/tileiras13.4、PyTorch2.13/CUDA13.3，前向kernel约1.14–1.82倍。示例batch4、16384token、8head、head64/state128/chunk128；图跨多长度/state，不能把最大倍数赋给示例或全模型。未给后向/E2E训练速度。建议**深入阅读并复现kernel微基准**，包括指数mask的数值稳定性，待backward与整层内存账后再评估替换。

稳定ID：`nvidia:cutile-mamba2-forward:blog:2026-10-19`。

### IBM Spyre公开编译中层：KTIR明确scratchpad布局，调度器生成数据流流水线

**IBM；10-20官方技术文。** [完整编译流程](https://pytorch.org/blog/from-ktir-to-spyrecode-an-mlir-dataflow-backend-for-ibm-spyre/)。Inductor先决定core划分/tiling与scratchpad位置，KTIR把全局/各core本地memory view、分布tensor及显式访问记录为约束；scheduler在此约束内经过ScheduleIR→DataflowIR生成load/compute/store流水、嵌套循环、double buffering和多instance。架构描述提供数据通路和PDL device pattern，而非调度器硬编码所有硬件。

公开范围是KTIR与Apache2 Dataflow Scheduler；DataflowIR以下仍接Spyre设备相关symbolic binary、metadata、SpyreCode及runtime，**不把上层开放写成全工具链每部分均已核开源**。符号地址/有界shape留待launch填入，解释了为何placement必须成为显式契约。本轮只核官方设计文，未审编译repo commit/执行LLM benchmark，没有推出性能优胜。建议编译器/新硬件适配者**深入阅读**IR边界与scratchpad lifetime；LLM部署采用仍需模型覆盖、端到端数值和公开二进制条件。

稳定ID：`pytorch:spyre-ktir-dataflow:blog:2026-10-20`。

### torch_npu、XLA与MLX：保持随机流、通信顺序和专家路由语义

以下是10-20默认分支的独立正确性增量，本轮均为静态证据，非本地硬件复现。

| 来源 | 问题、变化与采用边界 |
|---|---|
| [NPU RNG门控](https://github.com/Ascend/pytorch/commit/b9f98688f2bad1db1643da4bb3e093978bcf8273) | TE后端`fallback_random=False`会把uniform/bernoulli/rrelu分解到host philox，和NPU eager aclnn随机流不一致，dropout族出现失配；NPU设备门控返回NotImplemented并显式fallback，CPU/CUDA分解保留，部分图失去capture资格。**考虑采用/验证**RNG重放与图模式，不把fallback当新增随机kernel加速；也未关闭旧issue其余CPU污染/缺lowering簇。 |
| [collective执行依赖](https://github.com/openxla/xla/commit/bd5b6fb67b298076c2ecc56dd51cece356a64bd3) | CPU异步While/Conditional thunk的Execute返回时内部collective可能没完成，纯scheduling edge允许后续collective并发，导致Gloo tag0串消息/rendezvous崩；改execution edge等真实完成，IFRT79及JAX回归测试联动。**建议跟进**嵌套异步控制流；这是XLA main，不证明JAX0.12 wheel已包含。 |
| [batched-dot纠错](https://github.com/openxla/xla/commit/71892d702987093758c76e3818481aa6fa4263b3) | transpose(dot)代数化简遇非identity batch维且不与P交换时，旧式把batch slice配错并可算错shape；现在规范新batch维、组合outer permutation并用inverse定中间shape。**建议回归**非平凡3-cycle batched dot，非普通GEMM提速。 |
| [GEMM候选限制](https://github.com/openxla/xla/commit/296b7da06928c7bf8a7d59b0e852adda4c50af7a) | 排除collective进入GemmFusionV2候选，避免isolated autotuner缺device assignment/collective参数而所有配置失败。**考虑采用**对应编译失败修复，不能推导communication被普遍禁止融合。 |
| [ERNIE路由/1959](https://github.com/ml-explore/mlx-lm/commit/a28cdd0dc2662dafba47ae16a9aee06d485f1dcb) | 保存checkpoint `e_score_correction_bias`、以偏置后score选top-k而用未偏置gate算权重，conversion保FP32，缺失bias回零；**建议ERNIE4.5 MoE使用者回归**，避免把能加载当路由正确。 |
| [AFM7 offset/1990](https://github.com/ml-explore/mlx-lm/commit/72be8c955d9a0d3099bd01865af288e0e947c3fa) | AFM7共享KV reuse层改用起始cache offset，修left-padded batch RoPE；新回归用真实prompt cache构造。**考虑采用**并测padding和增量cache一致性。 |

MLX Metal另新增staged affine quantized matmul/split-K（[Metal QMM/4682](https://github.com/ml-explore/mlx/commit/bff10220c742907f0c68e55ff174aaea56f74e3d)），本轮核到算子dispatch与实现但没有同配置E2E数据，**继续观察**小batch量化decode收益，不给速度倍率。

<a name="data"></a>

## 训练数据工件

### Nemotron Personas UK/Germany：各100万合成人设，英国职业字段发生实质重写

**NVIDIA与Locai Labs / Soofi；两卡release自标10-20、v1.0。** [UK固定版本](https://huggingface.co/datasets/nvidia/Nemotron-Personas-UK/tree/0bb75b1469b90795eb56fa8a64927dac77dd0a11)、[Germany固定版本](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Germany/tree/dc3131fdb38b558830cb38fbffc1dd08a482bc7b)。最初数据10-19已上传；UK在10-20T00:48Z及14:56Z重写occupation/industry相关数据，后者位于主窗，全部7个Parquet的LFS/Git对象均相对窗前变化；Germany主窗仅citation拼写更正，作为同批发布补录。不要将改卡或报道日期等同两个数据集均本日首次上传。

数据用人口普查分布PGM与Gemma4-31B-it生成persona，可作为LLM合成对话或agent模拟的种子；不是100万真实用户，也不是100万现成对话。英国以三地普查PGM合并并做类别映射，德国以2022普查及专有PGM为基础。全部17分片footer核验**各1,000,000行**（UK1.49GB、Germany1.93GB压缩文件），schema有六类persona文字与人口属性；只验证元数据，未验证每条人设一致性或宣传的总体分布。

两卡CC BY4.0，英国另列原统计来源许可。两者只含成人，若干敏感属性/姓名/地址或细粒度企业字段没有作为公开列提供；英国职业与行业修订应固定revision再生成语料。分布拟合、独立性假设和小单元抑制会丢失长尾，不能从“按普查生成”推出训练后偏差消失或避免模型坍塌；本轮也没有下游训练证据。建议对生成语料做交叉属性分层抽检，并保留种子、模型和validator版本。

稳定ID：`hf-dataset:nvidia/Nemotron-Personas-UK@0bb75b1469b90795eb56fa8a64927dac77dd0a11`；`hf-dataset:nvidia/Nemotron-Personas-Germany@dc3131fdb38b558830cb38fbffc1dd08a482bc7b`。

<a name="evaluation"></a>

## 评估与可靠性

### FairStress：程序公平性评测数据上线，发布卡split路径与工件不一致

**Ai2；[官方介绍](https://allenai.org/blog/fairstress)日期10-20，dataset04:23Z上传，04:44Z卡修订。** [固定工件](https://huggingface.co/datasets/allenai/fairstress/tree/a5cdf125f8a59d605efd0ea9deefe75083588208)。它以300个双候选场景、29领域、不同身份提示强度及成对劝服前缀，衡量有依据的选择是否同样抵抗干扰；这是特定“对称公平”定义，不涵盖所有公平观。本条核查数据发布，不将官网实验概述当作本轮全文论文复核或真实部署歧视证明。

本轮读取全部**47个Parquet footer**：46个`full-*`共**12,978,183行**，一个`core-*`为**48,005行**，验证实际schema包含身份对照、candidate结构、干预家族、expected_correct和prompt；没有读取全量数据页或测漏标/重复。官网“13M/50K”为近似规模，不能把Core写成精确50,000。

[当前发布卡](https://huggingface.co/datasets/allenai/fairstress/blob/a5cdf125f8a59d605efd0ea9deefe75083588208/README.md)的full路径`data/*.parquet`也匹配Core；Core配置却指向工件不存在的`core_data/*.parquet`，示例又使用`PardisSzah/fairstress`命名空间。**建议固定本revision并显式指定`data/full-*.parquet`与`data/core-*.parquet`**，避免混入子集或找不到文件；这是tree/schema静态核查，未运行datasets端到端加载。CC BY4.0；比较后训练阶段时同时报告正确率、劝服下翻转率及身份交换对称性。

稳定ID：`hf-dataset:allenai/fairstress@a5cdf125f8a59d605efd0ea9deefe75083588208`。

### POPE读出审计：首token空格形式能移动6.13pp F1，尚不能据此扣除既有方法收益

**POPE token读出审计团队；2610.25964v1，2026-10-19T07:58:14Z；回溯补录。** [全文](https://arxiv.org/html/2610.25964v1)、[作者代码](https://github.com/Kesav2k04/ugaa-research)。§3/Tables1–2对LLaVA1.5-7B、4bit NF4、RTX3070Ti8GB、Transformers4.40.1、greedy一token跑9000题：模型只吐带空格“ Yes/ No”，两个无空格yes/no ID从未生成。把固定两ID logit比较改成tokenizer-derived可接受surface集合的max比较，adversarial F1 **.7608→.8221（+6.13pp）**，popular+.0537/random+.0316。

§5另四VLM三tokenizer家族验证，照搬ID在异词表甚至F1=0；建议迁移的是诊断算法，不是八个常量ID。§4/7强调六篇既有论文未说明readout，且本实验NF4与其精度未对齐，**不能算“修正后收益”、不能宣布某方法失效**。九种自做decode correction没有超过本修正baseline，也只代表本split设置。**建议立即纳入评测前检查**：固定模型/template/量化，保存首token ID和输出parser、重跑同精度baseline，再比较hallucination改进。稳定ID：`arxiv:2610.25964v1`。

### KEEP-Bench：压缩后直接检查证据链、条件和少数观点，比只看下游答对更能暴露丢失

**KEEP-Bench项目；2610.26827v1，2026-10-19T17:55:18Z；回溯补录。** [全文](https://arxiv.org/html/2610.26827v1)。§3系统fictionalize实体减少记忆捷径，4102实例分parallel900、conditional1364、imbalanced1838，原context均约2.5–3.6K GPT5 token；直接标注命题/完整链保留，不是长达百万token记忆benchmark。

Table3默认句数budget下，Qwen3-8B parallel All-Rec23.8%、conditional29.3%，Gemma4-31B57.6%/71.2%；EXIT并行Para-Rec41.9%但All-Rec3.0%，BRIEF-Pro conditional Cond-Rec61.6%但并行Para-Rec仅.3%。这些是不同压缩率/方法族，不能只按一列当统一质量排行；RECOMP-abstractive不可靠服从预算，未纳同样长度控制。

§5增加budget从.5×到2×，最强三LLM跨情景All-Rec均增32.8pp，最弱三仅3.9pp，表明某些结构识别失败不靠更长摘要解决。结果依赖构造任务、命题验证与英语数据，未证明真实agent保留同等。**建议复现/采用诊断维度**：在业务context压缩验收加入完整证据链、条件和少数观点召回，并同时报压缩率/成本，不拿答案accuracy代替保真。稳定ID：`arxiv:2610.26827v1`。

<a name="revisions"></a>

## 已报道论文的版本复核

### ReaDy、ETH与Jeva v2：正文/披露修订，未发现新性能表

本轮对已报道ID在replacement清单里出现的新版本逐一打开v1/v2全文，并比较全部HTML章节及表格；下列三项的表格文字均未改变，**不把旧性能重新当新结果**。

| 来源与版本事件（UTC） | 核实的实际增量 | 边界与建议 |
|---|---|---|
| ReaDy项目；[2610.20503v2](https://arxiv.org/html/2610.20503v2)，10-18T15:04:17Z；[v1](https://arxiv.org/html/2610.20503v1) | AppE删掉两句：checkpoint存在训练/评估重叠、prefix-survival不能建立held-out泛化且不能隔离语义检索机制；其余已解析正文与表格未变 | **删除免责声明不是解决重叠的实验证据**。仍按v1披露保留训练/评估重叠风险，不由相同表格推导新held-out保证。建议作者澄清数据切分及重跑；稳定ID `arxiv:2610.20503v2` |
| ETH项目；[2610.22035v2](https://arxiv.org/html/2610.22035v2)，10-19T04:40:59Z | §11仅删Alibaba Innovative Research Program资助表述；其余已解析章节/性能表无变化 | 属致谢修订，继续采用既有机制/实验条件，暂不需重做性能比较；稳定ID `arxiv:2610.22035v2` |
| Jeva项目；[2610.18773v2](https://arxiv.org/html/2610.18773v2)，10-18T16:36:21Z | 引言/related work更明确归属Jev typed Choice/Noul/Score接口，调整图3位置与引用；性能表无变化 | 属引用与表述纠偏，不能称新模型/新增训练结果；稳定ID `arxiv:2610.18773v2` |

HTML比较不覆盖PDF像素/打包资产变化；三项PDF没有逐页差分，故结论限定为已解析正文和表格。

<a name="coverage"></a>

## 实际覆盖、补录更正与未完成事项

本次先读取automation memory、持久state、全部本地日报与远端目录树，16个既有归档blob（15篇日报+索引）与本地逐一比对一致；另直接回读远端最新日报窗口和README。无待补传文件，未以历史日报替代本轮关键来源。

| 来源类别 | 实际检查与覆盖 | 未覆盖/失败边界 |
|---|---|---|
| 模型/数据 | 34登记/补充官方HF组织，另发现5组织；models/datasets×created/modified共156路线，均跨回溯边界或清单尾且检查日期排序；102候选记录去向。官网34请求/31URL，中英文主题搜索补名单外工件 | 枚举不等于逐项全文/权重核验；37未审或受阻候选保留。Meta commit API403、IFM rollout卡/commit401，不能推断无更新；OpenAI普通news/research403，Mistral/xAI超时，若干国内动态壳。Ai2部分正文经web恢复，Dory报告入口403仍未闭合 |
| 核心六仓 | all-base merged PR主窗229/回溯644；base-main224/630；默认分支commit主窗220/回溯637，满页完整分页。43选定PR以merge SHA/关键patch核查，slime主窗0匹配项 | all-base/commit/release分别计时；43PR不等于所有变更已审。最新10 release（Titan全部5）未见本主窗新正式发布，只限实际返回集合。旧P2P consumer和更早语义缺口保留 |
| 扩展生态 | 18仓默认分支commit主窗314/回溯746，完整分页；40选定commit关键patch。另查llm-d1.0、DeepSpeed0.19.8、TorchTPU包/源码和新硬件技术文 | 没有穷举扩展仓全部base/release分支合入；新正式release与晚合main分开。未执行硬件训练/服务验证 |
| 论文/arXiv | 六学科10-20官方公告2231次出现/1604唯一ID/595 replacement，全部595恢复元数据；回溯初稿805篇、355标题候选筛查、20篇关键全文及3组v1/v2差分；MemT报告由模型组另审 | 20篇初稿均10-17至19提交，本轮从10-20公告发现，明确补录；余785元数据候选和592 replacement待语义核查。不能用主窗submitted返回0推断当天无新工作 |
| 会议/正式发表 | TMLR列表16新forum ID；ACL/PMLR根目录入口差分；3个OpenReview forum/PDF尝试 | API/PDF403，forum200仅动态壳，正文未读；TMLR月粒度不构造精确发布日期。ACL/PMLR非逐volume/所有修订穷举 |
| 硬件官方入口 | NVIDIA/AMD博客、Intel Gaudi1.24.2、Huawei公告入口、TPU release notes、Neuron2.33入口重新检查；另读TorchTPU、Spyre、AMD megakernel、cuTile正文 | Huawei仍有动态内容、CANN/HCCL/libtpu及旧组件提交不全；Gaudi/Neuron已知版本不重报为新发。NVIDIA论坛10-20活动链接实际落到3-16 Rubin原文，未找到有日期的新技术增量，过滤 |

**arXiv更新时间异常仍存在。** 请求lastUpdatedDate的feed标题继续被改成submittedDate；主窗updated返回0不能用作“无修订”。本轮恢复replacement元数据和3组全文版本对，只推进这些ID，不关闭真实updated-date历史缺口。当前new列表快照也不能恢复每个历史时刻的所有replacement或未公告稿件。

**本轮补录/更正。** Muse Glimmer、MemT、Xing、d3-turbo、20篇论文及cuTile10-19文章按原始日期标注；Megatron8272从旧pending完成关键patch复核。SGLang44858已由45289撤回替换，只写最终设计；TorchTitan5424/5465以最终代码纠正旧PR文字；TorchTPU同文表格27.1%与叙述约28%分开。MiniMax未核实二手参数，保留昨天未披露边界；ReaDy删除风险表述不视为风险已解决。没有静默改写旧日报。

**持久化与后续续查。** 本轮证据、来源检查、候选及上传回执保存在本地`.monitor/research-2026-10-21/`，模型/论文/核心/扩展各有独立coverage账本；state分别记录本地保存和远端同步，上传前列pending，上传后按实际回读结果清空。下轮从 **2026-10-21T00:56:38Z** 续查并回溯72小时，同时继续下列旧区间。全球完整覆盖水位仍为null；**本日报不声称已全面监控**。

### 模型来源复查记录

MiniMax M3.1 Flash与Mercury3的10-19发布已于10-20归档。本轮重开[MiniMax正式发布](https://www.minimax.io/blog/minimax-m31f)、[API指南](https://platform.minimax.io/docs/guides/text-generation)、[Mercury3发布](https://www.inceptionlabs.ai/blog/introducing-mercury-3)与[型号页](https://www.inceptionlabs.ai/models)，没有取得新的正式权重或新增训练机制披露，故不重复写成今日模型发布。MiniMax发布页仍说权重稍后、34组织HF枚举中没有M3.1 Flash；二手“428B/23B”线索在重开的正式指南中未获证实，不覆盖昨天未披露结论。Mercury3绝对参数、扩散采样步数/块长、训练预算及1068TPS完整硬件负载配置仍缺失。此结论限已读入口，不能推断所有渠道均未发布。

34登记/补充HF组织全部完成models与datasets的created/modified双路线，另因发现补查MemTensor、XingChen-AGI、Cloudflare、perplexity-ai、vllm-sr；枚举与正文深审分开。DeepSeek、Qwen、Kimi、GLM、Seed、Google等已读列表/官网入口未发现可写入本节的新训练工件，不能据此声称厂商无发布。Meta Muse Glimmer的技术报告2610.24628v1已在模型报告节补录；8月旧权重的主窗改动因gated commits403尚未完成差分，不当新checkpoint。

未审或失败继续保留：IFM两rollout数据集卡与commits401、Meta gated commits403、Ai2官网直接403（web搜索恢复公开正文）、OpenAI常规news/research403、Mistral/xAI超时、部分国内官网JS壳；Anthropic新scorpion-binder数据与NVIDIA生物/物理、天气模型为已发现但不属本节LLM训练主线。Cloudflare10-19增加Typed Decisions评测、PPLX v1.2旧发布/改tag、其余待审工件均在coverage清单，不以数量上限丢弃。全局完整覆盖水位仍为空。

### 保留的历史缺口索引

以下状态沿用原区间，尚未完成的来源不因本轮枚举成功自动结案；详单和候选文件保留在state及其引用日志。8272仅关闭其单项待审，61532/61667仍open；UnOPD v2已在10-18完成，不重新开启。

| 来源/待办键 | 原始区间（UTC，或版本待办） | 当前处理 |
|---|---|---|
| `core_six_github_repositories` | 2026-10-05T00:00:00Z → 2026-10-08T00:00:00Z | 保留未完成区间与明细；本轮定点进展见对应来源节 |
| `pytorch_transformers_triton` | 2026-10-05T00:00:00Z → 2026-10-12T01:55:47Z | 保留未完成区间与明细；本轮定点进展见对应来源节 |
| `unreviewed_engineering_candidates` | 2026-10-11T02:34:06Z → 2026-10-15T01:55:47Z | 保留未完成区间与明细；本轮定点进展见对应来源节 |
| `Intel_Huawei_TPU_Neuron` | 2026-10-05T00:00:00Z → 2026-10-15T01:55:47Z | 保留未完成区间与明细；本轮定点进展见对应来源节 |
| `model_vendor_dynamic_channels` | 2026-10-05T00:00:00Z → 2026-10-15T01:55:47Z | 保留未完成区间与明细；本轮定点进展见对应来源节 |
| `papers` | 按原候选ID/版本续查 | 保留未完成区间与明细；本轮定点进展见对应来源节 |
| `formal_conference_and_Bolmo` | 2026-10-05T00:00:00Z → 2026-10-15T01:55:47Z | 保留未完成区间与明细；本轮定点进展见对应来源节 |
| `model_artifact_followups` | 按原候选ID/版本续查 | 保留未完成区间与明细；本轮定点进展见对应来源节 |
| `vllm_p2p_combined_lifecycle` | 按原候选ID/版本续查 | 仍open；定时失败不证明远端WRITE终止 |
| `arxiv_incremental_api_2026_10_16` | 2026-10-15T01:55:47Z → 2026-10-16T01:00:40Z | 补元数据/选定版本，不关闭真实updated语义缺口 |
| `new_model_artifact_boundaries_2026_10_16` | 按原候选ID/版本续查 | 保留未完成区间与明细；本轮定点进展见对应来源节 |
| `arxiv_updated_semantics_2026_10_17` | 2026-10-13T01:00:40Z → 2026-10-18T01:05:14Z | 补元数据/选定版本，不关闭真实updated语义缺口 |
| `formal_tmlr_2026_10_17` | 按原候选ID/版本续查 | 保留未完成区间与明细；本轮定点进展见对应来源节 |
| `engineering_selective_review_2026_10_17` | 2026-10-13T01:00:40Z → 2026-10-17T00:56:24Z | 保留未完成区间与明细；本轮定点进展见对应来源节 |
| `new_model_runtime_and_report_boundaries_2026_10_17` | 按原候选ID/版本续查 | 保留未完成区间与明细；本轮定点进展见对应来源节 |
| `engineering_selective_review_2026_10_18` | 2026-10-14T00:56:24Z → 2026-10-18T01:05:14Z | 保留未完成区间与明细；本轮定点进展见对应来源节 |
| `models_2026_10_18` | 按原候选ID/版本续查 | 保留未完成区间与明细；本轮定点进展见对应来源节 |
| `papers_2026_10_18` | 按原候选ID/版本续查 | 保留未完成区间与明细；本轮定点进展见对应来源节 |
| `core_selective_review_2026_10_20` | 2026-10-15T01:05:14Z → 2026-10-20T00:55:34Z | 保留未完成区间与明细；本轮定点进展见对应来源节 |
| `extended_and_hardware_semantic_review` | 2026-10-15T01:05:14Z → 2026-10-20T00:55:34Z | 保留未完成区间与明细；本轮定点进展见对应来源节 |
| `hardware_vendor_dynamic_and_components` | 按原候选ID/版本续查 | 保留未完成区间与明细；本轮定点进展见对应来源节 |
| `papers_replacement_review_2026_10_20` | 2026-10-15T01:05:14Z → 2026-10-20T00:55:34Z | 保留未完成区间与明细；本轮定点进展见对应来源节 |
| `model_candidates_and_artifacts_2026_10_20` | 按原候选ID/版本续查 | 保留未完成区间与明细；本轮定点进展见对应来源节 |
