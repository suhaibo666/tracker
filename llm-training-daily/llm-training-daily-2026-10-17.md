# LLM 技术动态日报 · 2026-10-17

> 时区：Asia/Shanghai（UTC+8）。增量扫描窗口：**2026-10-16 09:00:40 → 2026-10-17 08:56:24**；72小时回溯从 **2026-10-13 09:00:40** 开始。另续查此前失败的 **10-15 09:55:47 → 10-16 09:00:40** 论文区间与持久化旧待办；各来源实际完成范围见末尾，事件游标不代表全面覆盖。
>
> 实际生成时间：**2026-10-17T09:14:18+08:00（Asia/Shanghai）**。截止后的读取仅用于核实截止前事件或记录当前访问状态，不将新合入或新工件倒灌到主窗。除特别注明北京时间外，工程精确时间为UTC；论文提交日、公告/精选日、工件提交和发现日分别标注。
>
> 证据口径：一手论文正文、官方模型卡/配置、release、PR和具体commit。测试与性能均为作者/厂商报告；本轮只做静态证据核查，没有独立GPU/NPU/TPU训练或serving复现。main、dev、release分支、专用镜像及未合入PR分别记账。

## 本次重要变化

- **值得深读的新模型是Dory。** 6T预训前约4.5T使用原生NVFP4，再转BF16；共享循环块让同一权重用1/2/4loop换计算。更深loop并非每项评估都更好，单份KV模式也仅有定性质量说明。
- **Beam这次兑现的是开放权重与新证据。** Apache2的BF16/FP8/NVFP4、精确config、量化表和安全报告是本轮增量，10月初已报的万卡训练配方不重复算新发现。完整训练技术报告仍待发布，vLLM完整模型PR仍open，专用镜像与main支持须区分。
- **Hy-Memory把记忆写入器单独训练。** 跨harness、bank继承、局部归因与过存惩罚值得阅读，但没有各模块的因果消融；固定20%评估子集上的微小领先不写成显著胜出。
- **RL效率要看达到目标质量的时间。** STOPD、ReFlow、KV4-RL与新verl/TPU配置分别揭示截断估计、重放token分母、rollout瓶颈迁移和样本陈旧度。step更快不保证time-to-quality更短；Muon同数据与同时间也给出不同结果。
- **工程采用先排静默错误。** Megatron MXFP8 scale越界、verl FSDP2异步D2H参数损坏、Hopper Mamba/LoRA NaN、SGLang deferred release与flush页复用，以及TRL截断采样归一化均有明确触发条件。性能优化单列成熟度、默认开关与最终测试revision。
- **KV与服务系统证据继续补齐。** 远程KV分片大幅压低p99却仍可能比重算的p50慢；TensorRT-LLM分页/packed logits减少暂存，Ascend的逐层load与encoder直传也有新实现。单算子、传输字节、缓存输入和output TPS的倍率不可混用。

## 目录

- [模型与技术报告](#models)
- [预训练与架构](#pretraining)
- [后训练、蒸馏与Agent](#posttraining)
- [推理优化研究与新工件](#inference)
- [工业开源项目](#engineering)：[Megatron](#megatron)、[vLLM](#vllm)、[verl](#verl)、[slime](#slime)、[SGLang](#sglang)、[TorchTitan](#torchtitan)、[扩展项目](#extended)
- [硬件与软件协同](#hardware)
- [评估与可靠性](#evaluation)
- [覆盖、失败、补录与更正](#coverage)

<a name="models"></a>

## 模型与技术报告

### NVIDIA Dory1.0-4B：同一权重按循环深度换算力，原生NVFP4预训后转BF16

**身份/日期。** 官方[BF16卡](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16/blob/b366b87f762ea334ba3eb55912031baba1515810/README.md)标Release Date/GA **10-16**；权重上传commit `0ff16d446055f198ea8e68cff18c7382f66a5920`于10-16T18:26:17Z，当前固定版本 `b366b87f762ea334ba3eb55912031baba1515810`。另有MXFP8 `f35880f52368271136b48e3a4365f91e70a92308`、NVFP4 `41f02335e1a1ad3289b8b32485e2265cbe97cfdf`和GGUF `0963c8000791e77b2b1b49aa387d92a6da9988d6`；OpenMDW1.1。

**结构/训练。** 4B dense looped normalized-GPT，input→共享recurrent block反复应用→output，每次loop再注入input。卡按attention+MLP配对写 **8 input / 20 recurrent / 8 output blocks**；实际[config](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16/blob/b366b87f762ea334ba3eb55912031baba1515810/config.json)按attention、MLP分别计为16/40/16、`num_hidden_layers=72`，不是另一套72个完整Transformer block。hidden2560、16Q/4KV、head256、SWA1024与full attention交错、context262144、默认2loops。1/2/4loop不换checkpoint；更多loop增加每token计算，参数驻留量相同。可选`last_loop`每个recurrent attention只保留单份KV，相比`per_loop`避免cache随loop数增长；**卡仅定性称两模式精度接近、输出长度几乎不变，未给数值消融或等价保证**，不能直接称无代价。

预训 **6T token，前约4.5T原生NVFP4、余量BF16；另300B token扩到256K**。Megatron-LM预训后做NVIDIA curated SFT、多环境异步GRPO（math/code/science/IF/tools/multi-turn/structured output，NeMo RL+Gym）。披露爬取/合成code、math、science、general及43种编程语言，但未给各阶段loop采样、完整数据比例、GPU数/卡时/FLOPs或稳定性对照；不能把“原生NVFP4”外推为全程四比特训练，发布NVFP4推理配置是另一个 **W4A16** 工件（embedding/head豁免）。

**消融/边界。** 卡BF16同harness：1→2→4loop，SWE Verified **50.3→59.5→61.8**，TB2.1 **12.5→18.1→22.2**；但GPQA69.19→72.41→72.16、HLE8.25→10.89→10.43，更多loop并非单调提升。AA指标是内部估算，AA-LCR用了v1.1答案但旧Qwen judge，不能当官方最新AA榜单；跨模型个别†是自报。量化推理DGX Spark仅给batch1，ISL2560/OSL300时MXFP8/NVFP4相对BF16 **1.69×/2.17×**，ISL128/OSL1024为1.84×/2.44×；未给固定runtime SHA及完整量化质量表，也没有等成本loop比较，不能写4loop带来相同成本增益。vLLM卡明确依赖仍在review的[#60149](https://github.com/vllm-project/vllm/pull/60149)；llama.cpp用专门分支，不当稳定发布版通用支持。**建议深入阅读/小规模复现：先对齐loop数、KV模式和精度，再比较质量—时延—显存。**

稳定ID：`hf:nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16@b366b87f762ea334ba3eb55912031baba1515810`。

### Reflection Beam：预览兑现为Apache2权重，新增精确架构、量化表和安全训练证据

**本轮真实增量。** [10-16官方发布](https://reflection.ai/blog/beam-is-available)明确开放BF16/FP8/NVFP4和DFlash2 drafter。[10-05预览](https://reflection.ai/blog/introducing-beam)的23.8T预训、6144GB300四周内完成、10500GB300四周/100M+rollout等已在10-06日报详细收录，本轮只作背景；不能再当10-16新训练披露。BF16当前 `78f8f654d40a40919b8e9954b40dd879ecb9a692`，10-16T10:34:28Z的`06641076793dad51b8907f7eb2a14f4e08fd3ec8`标“Beam is available”；仓创建10-07不证明当时已公开。FP8 `077872da1757aa43e0330ab924b834825c54ecbc`，NVFP4 `3bd1dccf13c8f7aa3d6732c5aaa48ba107a2c086`，DFlash2 `163b276be744c8ce2b8385ea85de54edb3088211`。[LICENSE](https://huggingface.co/reflection/Beam-1.0-501B-A23B/blob/78f8f654d40a40919b8e9954b40dd879ecb9a692/LICENSE)已读为Apache2；卡明确不禁止蒸馏/输出训练/微调等，usage policy不修改权重许可。

**新工件补足旧架构缺口。** [固定卡](https://huggingface.co/reflection/Beam-1.0-501B-A23B/blob/78f8f654d40a40919b8e9954b40dd879ecb9a692/README.md)501.3B总/23.4B激活、text-only、1M context。config为52层（前3dense，49MoE）、hidden5120、48Q/8KV、head128、256expert top8，group8选4、router scale2.5；SWA512和full attention交错（多数5局部+1全局，末段另有全局层），词表151936、modified Harmony。不能因OCR API演示称原生多模态。卡仍写 **Technical Report (coming)**；已发布的是安全技术报告，不能称完整预训/RL算法报告已齐。

**量化证据。** 卡统一T=.7/top_p=.99：BF16→FP8→NVFP4，TB2.1 **80.1→78.9→78.4**，HLE-no-tools36.0→36.0→34.8、SciCode49.2→48.3→46.9；也有SWE Pro v2-Hard76.5→78.4→78.6。没有seed/CI、量化重训消融和足够匹配的完整harness版本，不能据个别上涨称量化增强/无损；跨厂商表不能一概作同条件排名。旧“3倍效率”按active params×生成token估算，排除prefill/attention/serving，和实测成本不同。

**新安全报告的可用机制。** [Safety Training节](https://reflection.ai/beam-safety-report#safety-training)披露单独alignment/safety teacher经SFT+RL后，与大规模RL teacher做MOPD合并；安全数据占该teacher SFT **1.62%**、安全RL环境固定 **5%**。Deliberative alignment先带policy生成/筛选轨迹，再重写为一般原则推理，训练时移除policy；不是把安全policy永久留在prompt。相同数据/配方的100B→501B teacher初步ISE对照，over-refusal相对降48.7%，同时harmful-request pass rate降3.14%，可看作容量与拒答权衡，非普遍安全保证。报告承认RL提高有害拒绝却降低无害服从，MOPD部分反向。

[Prompt Injection节](https://reflection.ai/beam-safety-report#prompt-injection)在同一Mirror harness、工具输出截到2000字符、固定非自适应冒充攻击上，加developer policy使ASR从3.1%/3.6%变0.3%/0%（949/560攻击episode），clean utility87.6→86.6、83.3→85.0。**不能与公开AgentDojo/AgentDyn直接横比，0%也不是零风险**；干净任务仅97/60个，后一项+1.7点相当于约1题。ISE工件与自蒸馏配方仍称未来数日/数周开放。**建议优先核实际权重/部署，安全配方与harness约束同时阅读；完整训练报告继续跟踪。**

稳定ID：`reflection:beam:weights-release:2026-10-16`；关联旧工作`https://reflection.ai/blog/introducing-beam`保留，不重复计新配方。

### Tencent Hy-Memory-30B-A3B：跨harness训练写入器，奖励归因有设计但未做模块因果消融

**日期。** [10-16三层开源官方页](https://memory.hunyuan.tencent.com/)，[权重/卡](https://huggingface.co/tencent/Hy-Memory-30B-A3B/blob/64a460a94d5498b7316fb02395c72d01c47a7347/README.md)当前 `64a460a94d5498b7316fb02395c72d01c47a7347`（10-16许可证提交）；上传权重两commit在10-15T14:34Z、模型卡10-16T07:16Z。论文[2610.22196v1](https://arxiv.org/html/2610.22196v1)首发 **10-15T16:46:47Z**，属于72h回溯补漏，不能说论文10-16首次公开；首次公开可下载时刻无法仅靠历史commit证明。Apache2.0。框架README已开，GitHub commit API403，未完成训练实现逐符号审计。

**机制/算力（论文§4–7）。** 该30B-A3B是替换Mem0/MemOS/Hy-Memory中LLM **write path** 的专用模型，不是增加模型内部无限记忆。config为HYV3、48层、hidden2048、128路专家top8+1shared、32Q/4KV、262144ctx。OPD从Hy3-30B-A3B instruct起，学生生成并实际写入bank，冻结Hy3旗舰teacher只返回top50 log-prob近似reverseKL；教师不另写理想bank。每row2rollout，constant LR1e-6，OPD step375进入RL。

RL按session切约10turn chunk、≤5chunk/window，每个parent bank复制8份rollout；局部probe和跨chunk问题按gold fact给来源chunk归因，加trap与bank-growth罚避免“全存”；同chunk跨rollout做GRPO，窗口末按各chunk总分选bank继承，**不是每chunk拼接不同rollout的bank**。128×H20、64window×8rollout=512trajectory/batch，LR5e-6、无KL、rollout MoE routing replay，28672prompt+4096response；模型/优化器/已选bank每10step一同checkpoint。总训练时长/总token/全部算力未披露，不能由128卡直接算训练成本。

**证据/缺口（§8/Table6/§10）。** reader/judge/retrieval/prompt固定，只换writer，各自从空bank建立；固定20% benchmark子集，6指标等权后4harness再等权，55.49 vs Opus5 54.62，Hy3-A3B base47.66。EverMemOS训练未见56.89 vs Opus54.04，较大差来自Travel/CMB；但Hy-Memory60.55 vs60.53、MemOS50.51 vs50.50，作者明确这0.01–0.02点视为持平，不能称统计显著超越。**没有**chunk-vs-window GRPO、归因-vs-broadcast、trap、增长罚、OPD初始化等控制消融，也没有多训练seed。gold-fact归因是证据关联，不证明某次write的反事实因果；跨optimizer步骤无法回传旧bank错误。**建议深入阅读reward接口与bank持久化失败检查；复现先固定writer以外组件，再消融归因/过存惩罚。**

稳定ID：`arxiv:2610.22196v1` + `hf:tencent/Hy-Memory-30B-A3B@64a460a94d5498b7316fb02395c72d01c47a7347`。

### 回溯补录：GPT-Live-1.1/Quip的系统卡公开了语音与对齐回退

官方[GPT-Live-1.1 system card](https://deploymentsafety.openai.com/gpt-live-1-1)标 **10-15**，本轮10-17发现；不以社区10-16帖子时间替代发布日期。1.1继任Live1、Quip继任mini，继续full-duplex可连续听/说；是语音与thinking模型组成的双模型系统，不能把它等同一个统一原生omni模型。闭源，无权重、参数/tokens/FLOPs/完整训练配方披露，训练来源只按公开、合作、用户/训练员/研究员提供或生成等类型描述。

卡§3：真实音频生产prompt评测与前代持平、无统计显著回退；合成边界prompt中1.1在sexual、mental-health、emotional-reliance、gore四类有显著回退。§4音频化alignment任务中，Respecting Warnings略回退；blocker disclosure为1.1略好/Quip略差，broken-search disclosure方向相反。前代基线是本次重测当前版本；任务是针对难例构造、非真实流量prevalence加权；alignment运行不含完整system防护/标准developer prompt，不能用它直接估算线上危险率，也不能用“总体持平”掩盖分项回退。**建议作为短讯，继续跟踪音频原生评测与delegate边界；不扩写成新的可复现训练成果。**

稳定ID：`openai:gpt-live-1.1:system-card:2026-10-15`。

<a name="pretraining"></a>

## 预训练与架构

### Learning to Read Out：早期行为跃迁可能来自unembedding学会表达

**Cambridge / Flower Labs / Isomorphic Labs；2610.21025v1，2026-10-14T23:45:26Z；72h补录。** [论文](https://arxiv.org/html/2610.21025v1)；[代码基线](https://github.com/hematteo/learning-to-read-out/tree/003811e7b6ffbe777524d98278f4d21f2bba8558)。

§2–4用共享跨checkpoint稀疏字典分解每个词表readout行轨迹，结合固定hidden-state probe/readout swap区分“可线性解码”与“本模型输出能表达”。静态核`src/readout/crosscoder/wu_adapter.py::preprocess_snapshots/build_crosscoder/batch_iter`把各snapshot中心化缩放、同token行跨32个head喂入crosscoder；`src/readout/probes/readout_swap.py::compute_margin`固定h后比较正确/错误readout行。crosscoder底层llamascopium未逐核，不能声称全算法独立验证。

Pythia160M/1B/6.9B各32checkpoint，重组集中0.4–1.1%训练；WordNet35类probe用于readout rows的类别组织；SVA/IOI与二tokencontrast hidden probes才支持hidden state信息领先于可输出表达。OLMo2-7B首阶段1–3%也有早峰，但未显著区别于circular-shift null（p=.12），不足以建立普遍阶段。四个31M控制run各10Btoken，readout LR .25/1/4倍使表达gap更早闭合，早0.27Btoken loss差可1.20nats，最终loss差≤.06。readout占31M模型42%而Pythia1B约10%；每条件仅一个run，3seed主要是字典fit，不是预训多seed。

**建议深读**：比较短预训配方时可同时测probe、native output、readout swap，避免把早loss领先误当最终能力增益。这里是二元可解码性，无开放式能力/加快大模型预训的证明。

### Tokenizer指标：压缩好不保证下游好，跨语言相关方向会翻转

**EleutherAI / Basel / NTUST / Academia Sinica / Rochester；2610.21679v1，2026-10-15T11:05:07Z；上轮缺口补录。** [论文](https://arxiv.org/html/2610.21679v1)。论文链接repo在本轮git返回Repository not found，不能宣称工件可用。

§4–7：约200个100M–1B模型、九语言、48benchmark；TokSuite14个1B、TIMTC实际45/54个350M（9个OOV错误排除）、新增VISTA130M/350M，比较29个intrinsic metric。TokSuite40B英语+另四语言各15B；VISTA作byte-premium数据量控制，但词表变化也改变embedding/head参数，不能称所有比较严格同参数。13/29总体显著，最大相关仅r=.27；控制参数量后额外解释方差最高R²=.021，单任务/语言相关强弱乃至方向会变。

新U-statistic把token内部PMI瓶颈与边界熵相加，在TokSuite英/土好、VISTA近乎无关系；汉字/Farsi字符统计导致跨语种量纲差。**建议深读/应用**：用多维intrinsic筛选+小规模训练验证，别靠fertility/压缩率单指标定词表。证据最多到1B、部分任务近chance，未证明metric永远不可能预测。

### LangDiT：CDLM骨干组合改动有效，最低GenPPL来自更长训练而非纯架构消融

**Xiaohongshu Dots / PKU / CUHK Shenzhen / UCLA / Polyadic；2610.22330v1，2026-10-15T17:52:31Z；上轮缺口补录。** [论文](https://arxiv.org/html/2610.22330v1)。未核得专用公开实现；不报已可复现。

§3组合backbone/QK RMSNorm、更宽noise conditioning、likelihood shortcut codebook梯度detach及有界校准、训练bootstrap self-conditioning熵gate；推理仍用未gate posterior mean。§4 shared LangFlow ladder：OWT1024、Qwen3词表151936、batch512、250kstep、BF16、一个seed42、EMA；总参数327.09M→334.45M（不是约100M总参数，100M只去掉codebook/head）。128NFE同250kstep GenPPL64.26→53.66（-16.5%），unigram entropy5.416→5.391。conditioning多7.36M参数，非严格等容量/等FLOPs。

660k/520k延长训练可达21.72/11.33 GenPPL，但training plan/NFE不同；Plaid128steps实际129calls，且baseline40k vs修改150k等不匹配。**建议深读/观察**：有价值的是同条件累计消融和训练gate/推理估计分离；不把最低GenPPL全归架构，也不称更快推理。只有OWT单规模单seed，低GenPPL与更低entropy并行，不是语义多样性/总体质量胜利。

### Continuum Attention旧待审：连续KV积分是注意力变体，仍是二次复杂度

**University of Luxembourg SnT；2610.20586v1，2026-10-14T17:57:40Z；72h旧待审补录。** [论文](https://arxiv.org/html/2610.20586v1)。正文未核专用公开实现。

§2把相邻KV连成分段线性曲线，每query沿连续位置做指数密度归一化积分，用0/1/2阶指数moments解析前反传，边界半token常值延展保证等score时的token质量约定。§3 Tables1–4：TinyStories byte vocab256、约5M六层、192/512context，各3.84M训练token、五训练seed。固定温度.30同softmax NLL1.029903→1.009888（192），1.174585→1.126540（512）。**摘要把.0200/.0480写成softmax/focal两基线，正文实际对应两个context长度下的同温softmax对比，应以正文为准。**

Apple Silicon MPS/PyTorch2.8、自定义Metal对SDPA；512训练tok/s31,130→28,867（-7.3%），forward127,098→77,450（-39.1%），不是性能提速。额外disjoint slice曾在比较中看过，虽有五seed paired CI仍是探索性，不能称未触碰测试集。**建议观察**：值得数学机制阅读；大尺度/长context/CUDA与实际任务增益未证，暂不建议生产替换FlashAttention。

<a name="posttraining"></a>

## 后训练、蒸馏与Agent

本节论文大多是10-15首发、上轮接口失败区间的补录；日期逐条列出。旧待审与回溯首发不当作10-16/17新算法。

### STOPD：截断rollout可以保留梯度和的期望，但不等于整个优化轨迹无偏

**Cornell / Georgia Tech / Alibaba；2610.21337v1，2026-10-15T06:22:08Z；上轮缺口补录。** [论文](https://arxiv.org/html/2610.21337v1)。未发现专用实现链接；未代码核查。

传统OPD把学生完整生成、教师打分、学生反传全部跑完；固定前缀截断则永远失去后段监督。STOPD在生成前随机选停止block，用token被保留概率的倒数加权；block向上取整抬高尾token的最小保留概率，避免极端权重。归一化分母来自前一batch的加权长度估计，并先用25步完整OPD预热。§3.1–3.2证明的是给定完整rollout与局部反馈、预算不改变前缀分布等条件下的**梯度和/长度和**无偏；归一化梯度仍存在分母尺度差，Adam/Muon非线性更新仅给一步误差界，不能写成与完整OPD严格等价。

§4–5、Table1/AppendixD–F：Qwen3-4B teacher→1.7B student，math/code/IF各250步、3训练seed、batch256、K128；每run两张H100。4B/8B学生由30B-A3B teacher蒸馏，分别两张H200/GB200；均non-thinking。主设置少生成25–48% token，完整训练step加速math1.8x/code2.2x/IF1.3x。T=.7/p=.95，math为5基准混合avg@8/3、code为5基准avg@4/8（含200个held-out分布题）、IF为IFEval prompt-level strict avg@4；这些聚合指标最终math30.1→29.4、code28.5→28.1、IF59.8→60.1，属于近似保持。固定100token ESR最终math23.5，且回答变长、截断增多；无IPW随机截断也常保持准确率，IPW额外改善长度/终止行为，不能把所有准确率收益归给IPW。

**建议深读/复现**：适合OPD生成占主成本的任务。先复现完整step时间、尾权重、终止率与多seed精度，再测不同batch与长度；SFT/self-distill扩展在正文是推测，没有对应实验。

### ReFlow：限制重放与弹性资源池减少等待，2.06x的吞吐包含重放token

**HKU / USTC / Alibaba / SEU；2610.21640v1，2026-10-15T10:21:41Z；上轮缺口补录。** [论文](https://arxiv.org/html/2610.21640v1)。正文称基于verl实现10.6K Python行，未给可核专用repo；无代码验证。

一般异步RL仍需足够新轨迹才能开训，增加lag又会损害学习。§4–6将未使用与已使用轨迹放到同一replay buffer，限制batch重放占比σ、重放loss权重τ及版本lagδ；实验σ=τ=.5、δ=1。再用预初始化训练/推理混合worker与CPU offload切换角色，调节rollout并发和两侧GPU比例。理论仅是无相关误差等假设下的局部梯度MSE/一步进展界，未证明全程收敛或任意重放都有效。

§7：最大16节点128 H800，节点间8×400Gbps，vLLM+FSDP2；Qwen3-4/8/14/32B，DAPO数学、APPS多轮coding、ReTool。相对verl同步、one-step-off及verl-async，数学trainer吞吐1.19–2.06x，coding1.07–1.92x；32GPU角色重分配5.72s/9.20s；8B/32GPU trainer idle55.8%→15.5%。**吞吐是trainer处理token/step时间，含重复轨迹；收敛主证据为training reward，而非held-out模型能力。** benchmark只测5或10步稳态窗口，也未把全部质量/新样本效率转化成预算同等的测试集比较。

**建议深读/复现**：优先对比time-to-held-out-target、新轨迹数与总GPU小时；别把2.06x直接用于采样成本预算，也别称比所有异步框架更快。

### Waypoint-OPD：用教师中途checkpoint判断哪些策略变化值得迁移

**Alibaba Cloud；2610.21601v1，2026-10-15T09:47:19Z；上轮缺口补录。** [论文](https://arxiv.org/html/2610.21601v1)；[代码基线](https://github.com/hhhbxloveai/WaypointOPD/tree/69241267f325382bc6920fca9598b9f07458da4c)。

§3把teacher初始→中途、以及中途→最终的logprob变化，在学生top-k候选分布下中心化，计算带权cosine；方向相反就压低最终endpoint shift，防止部分抵消的临时偏好被照搬。全batch gate均值归一化保持总体scale。静态核两处：`verl/verl/workers/actor/dp_actor.py::_compute_waypoint_opd_rm_scores`实现FP32 detach、q-centering及非负cosine gate（近零范数fallback为1）；`verl/verl/trainer/ppo/ray_trainer.py::_normalize_waypoint_opd_reward_by_gate_mean`在组装batch后按有效prefix均值缩放。

§4–5/Tables1–3/AppC/F：Qwen3-4B数学/代码RL教师step0/600/1200，学生8B/14B/30B-A3B，同Qwen兼容tokenizer；8 H20，100迁移更新。相对Direct-OPD，数学七项mean@32提高2.18/1.59/1.89pp，代码五项mean@8提高2.66/1.52/1.19pp；8B math去gate56.14→53.50、打乱gate→51.20，位置相关性有控制证据；邻近waypoint500/600/700平均仅差.05pp。

**限制/建议**：多一次冻结教师评分，但单个8B math timing更快3.73%受生成长度影响，不能推出评分本身免费；各表是所选checkpoint，非统一终点全部多seed结论。建议保留中途teacher快照，并复现同预算迁移，跨模型族/词表尚未证明。

### TF-OPD：同底座专家可用激活二阶矩闭式合并，跨架构蒸馏仍不在范围

**UIUC / Tsinghua；2610.22236v1，2026-10-15T17:15:01Z；上轮缺口补录。** [论文](https://arxiv.org/html/2610.22236v1)。正文未核得专用代码链接。

§3把每层“合并权重在各专家自己输入上匹配专家输出”化为RegMean式最小二乘，用专家自生成答案的输入Gram矩阵求解；无需反传。但层级精确解**不是整个模型生成分布的精确蒸馏解**。过滤重复到length cap与验证失败的rollout比求解器更影响难题；学生自生成rollout反而未改善闭式解。

§4/Tables2–3/AppB：同底座Olmo-3-1025-7B→RL-Zero Math/Code专家，各生成256训练题，只保留175 math与92 code；48GB GPU单进程，整体约10 GPU-hours（未给GPU型号）。七benchmark平均61.1，math专家59.8、code53.6；AIME25合并38.3 vs math43.3，不能写所有单项超过专家。发布专家以BF16评估、合并权重FP32存储FP16评估；匹配FP16后AIME25距teacher<1pp，精度格式会改变此量级比较。数学难题各仅30题、4采样。

**建议观察/复现**：适合同架构同初始化领域专家整合；复现必须控制输出存储格式、评估精度和去loop规则。不能把标题理解为通用OPD不需要训练，也没有与强训练OPD完成同预算质量比较。

### RDC：路由漂移可作为筛样信号，不能把native routing称为完全无偏

**PKU / SII / Zhejiang / SJTU / Infrawaves；2610.21682v1，2026-10-15T11:09:10Z；上轮缺口补录。** [论文](https://arxiv.org/html/2610.21682v1)。未核得专用实现。

§3用每个response各层expert使用频率的total-variation距离衡量rollout/train漂移，在长度bucket里丢最高8%response，其余保持训练router的原生expert选择，另加TIS。相对R3固定rollout route，避免丢掉当前router本会激活的expert梯度；但过滤本身改变样本分布，正文只说明避免**replay相对native routing**的特定梯度偏差，不是整个RL目标无偏保证。

§4/AppA：post-trained Qwen3-30B-A3B、non-thinking，DAPO17K，Megatron BF16训练/vLLM FP8 rollout，20,480训练/30,720评估长度；每5步验证取各自峰值。异步Table2 AIME24/25：TIS65.0/54.8→RDC71.2/59.6，均值+5.5pp；ZebraLogic53.7→53.0，未全面获胜。同步高staleness(2048 rollout,128 update)时AIME24+6.4、25+4.3pp；低staleness有退步。

**建议复现**：MoE异步RL值得测routing drift分布、丢样率、质量/吞吐；GPU型号数量、重复训练seed及筛样对稀有难题影响未给充分证据，不能据此全面替换R3。

### MRTPO：把每项reward的优势分给相关token，额外replay让单step慢10–20%

**UMN / NUS / UIUC / USC / UPenn / MBZUAI；2610.21439v1，2026-10-15T07:47:04Z；上轮缺口补录。** [论文](https://arxiv.org/html/2610.21439v1)；[代码基线](https://github.com/zhaihaotian/MRTPO/tree/865c484a5c40b02c0a3394622edda54693c80636)。

§4保持GDPO各reward的组内标准化优势，replay原响应在“全部verifier信息”和“移去一个verifier信息”下的next-token分布，以JS distance定位相关token，β=.5混合均匀权重后再分配。静态核`verl_mrtpo/verl/workers/actor/dp_actor.py::compute_criterion_js`确实读full/minus两分布；`verl_mrtpo/mrtpo/advantage.py::compute_criterion_advantage`逐criterion分配并约束residual二阶矩，保持response平均credit，代码还含10步smooth ramp。

Tables1–7/AppB：1.5B–7B六模型，SQL100步、IF300步、tool100步；SQL8 H20、IF8 H200、tool4 A100；分别3/5/5seed。SQL相对GDPO平均约+1.3/+1.6pp；IFBench+2.4/+3.2pp，tool最终精度差小而格式更早学会。打乱token权重后收益消失支持位置效应。replay只对reward组内变化的pair执行，12–21%pair，单step仍慢10–20%，以更少step达标与wall-clock加速需区分。

**建议复现**：适合可文本化的多verifier任务；length reward或不可表述的reward不能这样定位，LLM-judge rubric及更大模型未测。JS依赖性不是因果credit真值，奖励本身错误仍会被更高效优化。

### Mosaic：多session奖励要区分压缩与subagent，session数是成本代理

**HKUST(GZ) LARK / Huawei / CityU；2610.22174v1，2026-10-15T16:32:44Z；上轮缺口补录。** [论文](https://arxiv.org/html/2610.22174v1)。项目页已给，未核专用实现。

§3针对compaction/subagent使一条trajectory跨多个context，把终局优势按同结果组内session数rank调幅（混合组不改正负号），再用真答案在session前后log-likelihood变化提供无critic局部信号，按normal/compact/subagent类型归一化，并排除最终session防答案泄漏。

§4/Table1：Qwen3.5-4B、REDSearcher1k一epoch、batch16/group8，每session128K，Serper/Jina工具、deepseek-v4-flash判分，BrowseComp1266题。对multi-session broadcasting，compact准确率35.31→38.86、session3.17→2.35；subagent31.44→32.46、4.43→3.77，后者收益较小。轨迹级与session级消融均有贡献，但DAG subagent的即时信息增量未必代表长期贡献。

**建议深读**：训练harness应对齐推理的session生命周期；减少session不直接等于减少token/GPU成本，论文未充分给硬件预算与训练重复方差。只在一个模型/深搜benchmark检验。

### EliteReplay：pass@k并不计正确解法的种类，持续replay比先SFT更能保留策略

**Ohio State；2610.22290v1，2026-10-15T17:43:31Z；上轮缺口补录。** [论文](https://arxiv.org/html/2610.22290v1)；论文提供[仓库](https://github.com/skai-research/EliteReplay)，本轮未静态核其实现。

§2–3用冻结base的MAP-Elites搜索得到不同SQL AST策略的正确解archive，GRPO中持续加长度归一化replay NLL（μ=.05）。另提出U@k计不同正确策略、effective diversity和base策略保留率，避免把至少一个正确解的pass@k当解法多样性。

§4–5/Tables1–2/AppB：Qwen3.5-4B Base，BIRD5000题训练100步，1000题每题72候选构建1500条archive；4 A10080GB（训练/rollout各2），vLLM+verl/Trinity。BIRD pass@12874.8 vs MaxRL73.7，U@1289.1 vs6.2；Spider U9.4 vs5.3。IID replay可得相近pass@k但U7.4，先Elite SFT再GRPO U3.0，支持archive构建+持续replay的联合价值。

**建议观察/复现**：归档搜索额外72k候选不是免费的同预算GRPO；策略定义是SQL AST操作而非通用语义推理，SQL execution等价也不能保证所有数据库状态逻辑等价；单4B家族、无大规模跨任务证明。适合关心重采样覆盖和歧义解释的训练评估。

### Muon证据并读：数据效率、时间效率与长程RL成功率分别报告

**2610.21555v1，2026-10-15T09:15:49Z；新论文补录。** Mushir Akhtar等四作者；正文未给完整机构归属，不从邮箱推机构。[官方TeX/PDF](https://arxiv.org/abs/2610.21555v1)，HTML404后已下载官方src读Main.tex；未核公开实现。

§2–4/主结果表：单A100-SXM4-40GB，17.7M/33.6M LM、2.7M ViT，FineWeb-Edu/CIFAR100，LM context256/batch4096token/5000step，23候选单seed开发后五个matched训练seed、一次冻结测试。Muon同数据LM validation AUC改善3.25%/4.46%、test loss4.77%/7.01%；同时间AUC反而AdamW胜6.96%/3.38%。参考Muon optimizer调用含aux AdamW占总训练56.1–69.6%，只有34–47% AdamW吞吐；不能把全部时间归到Newton–Schulz，因为未分解。多数LM参数仍走aux AdamW，调参同候选数也不等于同搜索空间/同调参时间。

**实践意义**：应记录同token、同FLOPs/同wall-clock、同调参预算三条曲线；单机小模型参考kernel不能推翻大规模优化Muon。

**旧待审2610.20256v1（10-14T14:25:54Z）** [Long-Horizon RL Needs Muon](https://arxiv.org/html/2610.20256v1)也重读：RUC/Duke/CAS，MiniCPM5-2B/Qwen3.5-4B、R2E-Gym、ClaudeCode2.1.118、单rollout异步+TIS+token平均+长度/tool-error处罚；正文主要是选最长run训练成功曲线，汇总有“90–100完整窗口，否则最后观测值”，full recipe同时改多项，无充分held-out测试/多seed预算/硬件表。可补录其工程配方与局限，**不从曲线猜数，不把Needs当必要性定理**；两论文的任务/规模不同，不互相证伪。

### Librarian Agents：学习在问题到来前组织语料，收益需跨多查询摊销

**Stanford / CMU；2610.22273v1，2026-10-15T17:35:17Z；上轮缺口补录。** [论文](https://arxiv.org/html/2610.22273v1)；给[官方repo](https://github.com/teresaz05/librarian-agents)，本轮未代码核查。

§2/AppendixA.7把builder改写文件系统建成MDP，以每8turn的answerability变化+terminal QA reward做GRPO。训练query只用于奖励、不暴露给builder；primary实际上construction/nav-cost权重全为0，不能称已直接优化所有成本。Qwen3.5-4B builder用LoRA32，680任务训练/50 heldout测试；32turn，1k生成/60k轨迹上限；Create/Merge由固定35B-A3B executor，history由Gemini3.1FlashLite压缩，answerer冻结。

§3/附录：BrowseCompPlus Qwen answerer准确率.75 vs embedding.42/rerank.44，Gemini answerer.77 vs.30/.42；不直接搬摘要“最强基线33pp”。扩展amortization435题匹配run，10queries/corpus时按正确答案归一化token少33.8%；raw token/query仍更高，原始token成本point crossover31queries、bootstrap18–54，两个口径必须区分。

**建议观察/复现**：适合静态共享语料的多查询；只改单个知识库问题未必摊平，内容更新与验证synthetic摘要/回原文路径仍是待解问题。50题评测小、依赖35B工具和闭源模型，不应归为“仅4B模型”的收益。

<a name="inference"></a>

## 推理优化研究与新工件

### 远程KV：分片解决排队，仍可能在中位延迟上输给重算

**Multiverse Computing；2610.21500v1，2026-10-15T08:37:22Z；上轮缺口补录。** [论文](https://arxiv.org/html/2610.21500v1)。正文列LMCache PR5545/5547/5548/5550/5551/5553及FlashInfer3969；本轮未审这些PR，不把论文结果当合并状态。

§4–5把remote hit拆为固定engine开销+KV字节/有效带宽。GLM-5.3 FP8 MLA在8×B300、PCP8下，原rank0集中存储成为瓶颈。按chunk索引静态分到8个rank及8个Valkey实例，复用现有lookup广播协议，再加异步读取；并修丢completion、missing-key GET悬挂、异步GET错key三类client问题。单纯异步单实例会让49%请求超时回退重算，不能把其低延迟当完整cache成功。

§6 Tables1–3：450条16k命中重放、98.5%token来自L2，单实例同步p99 83.32s→全修复8实例.67s（124x），p50 9.70→.40s；但同请求重算p50仅.31s，cache仍慢.09s。固定成本约.33s，约17k以下即使无限带宽也难赢只是模型预测，长度阈值尚未直接实测。200k并发prefill，分片与独立FP8 requant kernel修复合计43.2k→57.3–58.0k tok/s（34%），不能归给缓存单项。端到端11→5节点配置同时改变并行度与PD布局，作者明确不作为缓存消融。

**限制/建议复现**：每条件一次，<10%差异未立稳；latency只14.7–17.9k桶、单engine私有L2；6 needle×4路径只能证明有限语义正确，不能证明bit-exact。建议把load-vs-recompute决策加入真实延迟模型，优先减小每chunk collective固定开销。

### KV4-RL旧待审补录：INT4 rollout需要保护关键位置，1.80x rollout仅变为1.27x整轮

**KAIST / NAVER Cloud / AITRICS；2610.14448v1，2026-10-10T01:24:03Z；旧待审补录，早于72h。** [论文](https://arxiv.org/html/2610.14448v1)。自研in-house框架，未核得专用公开实现；不称vLLM/verl原生功能。

§4–5说明持续并发把KV访问集中到更大的decode batch；权重读已被摊薄，KV成为更值得量化的对象。KV4-RL使用paged INT4全历史，Q/K Hadamard旋转、group128，并用有界INT8 side-cache保护前16个sink与近期128token，保护区PV用BF16、区域结果做log-sum-exp合并。跨更新保留token与原behavior logprob，更新后re-prefill重建缓存；这里“异步”指partial rollout跨更新，**生成/训练仍在同四卡交替**。

§6、Table1、AppA/B/E：每run4 H200，Qwen3-4B/8B Base，GRPO、DAPO-Math-17k，1024prompt/16384response上限；4B600轮、8B400轮。8B累计rollout1.80x、完整迭代1.27x；4B1.57x/1.17x。统一BF16评估六项平均距BF16 rollout训练<.5pp；纯KV4在8B MATH500落后5.24pp，TIS不足以自动消除量化采样偏差。**每精度仅一个训练run，三个seed为同checkpoint评估采样seed**，不是三次训练复现。

**建议深读/复现**：较高工程价值，先用真实长度分布看完整迭代收益。末50轮learner占60.5%，量化rollout后瓶颈迁移；不能拿更短、更差的纯KV4响应作为同工作量加速。

### LoopAhead：早loop多token草稿+中loop暂停错误分支，收益限于单请求decode

**SJTU / Tongji；2610.21594v1，2026-10-15T09:38:11Z；上轮缺口补录。** [论文](https://arxiv.org/html/2610.21594v1)。未核得专用官方repo，论文对LoopSpec/DFlash使用本地复现/改造。

§3用独立训练两层512宽LAD读取早loop64state窗口+native anchor，预测2–3额外token；中loop反馈暂停不合分支但保留state，最终full-depth决定输出，sampling包含校正。草稿用8192词表、最终验证全词表，不能把draft vocab缩小当目标分布改变。

§4/Table1–3/AppA.5：A40 48GB、每GPU单请求、BF16 target/cache，Ouro4loop与Raven32loop；greedy六checkpoint×五任务全部30组复现AR输出，decode-only2.82–7.82x，最大是Raven-TinyLlama/HumanEval。sampling仅Ouro1.4B/Raven-Llama两模型、T1/top-p.9/3seed，3.00–4.32x AR、比LoopSpec+16.2–36.8%。计时含draft/调度/验证，排除prefill/加载/warmup。

**建议复现**：looped模型单用户低延迟有价值；没有大batch serving、端到端TTFT或标准成熟后端对比，局部DFlash adaptation不代表其官方实现上限。分布保持是验证规则与假设保证，非实测抽样有限集合能证明所有分布相等。

### 两个新drafter工件：不把接受长度写成服务吞吐倍率

- **Tencent DSpine**：10-16创建/上传 `tencent/Qwen3-8B-DSpine-b16@9fc2ab3d4db06ca06dd30d91b2f8d28537673add`，[固定卡](https://huggingface.co/tencent/Qwen3-8B-DSpine-b16/blob/9fc2ab3d4db06ca06dd30d91b2f8d28537673add/README.md)、config、历史和[旧论文2609.36173v1](https://arxiv.org/html/2609.36173v1)已开。新增是Apache2 drafter及inference code，论文属9月旧工作。并行draft在每层把前驱预测feature注入后继，因果依赖沿深度传播；config为5dense层、block16、target层[1,9,17,25,33]、最后write用选中前驱refine。卡将DSpine/DSpark/DFlash2都在 **Open-PerfectBlend同配方重训**，Qwen3-8B七任务平均AL：T0 **7.83/7.44/7.55**，T1 **6.98/6.65/6.53**；旧论文附录则是ShareGPT六epoch，不能把旧论文训练预算直接套给新权重。训练代码/Qwen3.8-27B仍在roadmap；此处无固定硬件吞吐对照，不宣称相同比例加速。
- **NVIDIA Muse Spark Open DFlash2**：10-16卡 `fbec887a79866e9ecff6c57adfd9fd02aa3cadea`，[固定卡](https://huggingface.co/nvidia/Muse-Spark-Open-816B-A42B-DFlash2/blob/fbec887a79866e9ecff6c57adfd9fd02aa3cadea/README.md)。5dense层4.38B（不含共享embedding/head）、noncausal SWA2048、两tap in-block conv+pairwise path selector；ModelOpt0.47预发布训练，3.48M多语言text prompt由target合成五effort响应。GB300 TP4+EP4、concurrency32、xhigh、output4096、NVFP4 target、draft7，MT-Bench AL greedy/T1 **3.87/3.49**，SPEED qualitative3.84/3.47，32K throughput split3.35/3.03；T1无top-p/top-k截断。**实测只有text，目标支持image/audio/video不等于drafter完成多模态性能验证；AL不是TPS。** NVIDIA Open Model Agreement并关联Meta Muse许可，不能写Apache。该工件是昨天SparkOpen发布后的新适配，不重复816B模型发布。

<a name="engineering"></a>

## 工业开源项目

主窗为 `2026-10-16T01:00:40Z → 2026-10-17T00:56:24Z`，72h 回溯起点 `2026-10-13T01:00:40Z`。下列 main/dev 均为源码合入状态，六仓本轮 release 列表未见主窗新 release；不能等同正式发行版本。所有测试、性能数字均为原作者披露，本轮仅做一手资料与关键 patch 静态核查，没有独立运行 GPU/NPU/TPU 或训练。

<a name="megatron"></a>

### Megatron-LM

#### NVIDIA Megatron：MoonEP 式虚拟专家进入 main，保持逻辑路由并转移计算位置

稳定ID `github:NVIDIA/Megatron-LM:pr:7450`；[#7450](https://github.com/NVIDIA/Megatron-LM/pull/7450)，main，2026-10-17T00:21:49Z，merge `36731acc3f0dac859ee7369bdc3a6af58bac546c`。

新增默认关闭的 `--moe-virtual-expert-load-balance`，在 HybridEP 上给每 rank 的 L 个原生专家增 L 个临时 slot。GPU planner 统计全局负载、复制选中的权重并重映射 runtime expert ID；token 仍由 HybridEP dispatch/combine，副本梯度归还原参数，optimizer/checkpoint 不增加独立副本。已读 MoE README、配置/dispatcher 关键 patch 与测试入口；不是把专家概率或逻辑选专家改成另一算法。

门槛很具体：固定且各 EP rank 相等的 token 数、EP2–64、ETP1、Triton≥3.5、TE grouped tensor、BF16 或 native MXFP8。MXFP8 要求 param gather/复用梯度 buffer；FSDP v1/v2、变长、capacity/drop、DeepSeek expert bias、Sinkhorn、部分 MoE graph scope、全 MoE recompute 和 FC1 offload 等仍不支持。副本 grad 默认 FP32；BF16 partial 先丢失的 cancellation residual 无法靠后续 FP32 相加恢复。作者列 parity/determinism 测试，未给可用于容量规划的本 PR 端到端提速表。建议**深入阅读并小规模复现**真实路由偏斜、额外 workspace 和同步成本，不将 MoonEP 名称直接换算训练倍率。

#### NVIDIA Megatron：原生 P/D 拆分加入 KV/SSM 所有权协议

稳定ID `github:NVIDIA/Megatron-LM:pr:6675`；[#6675](https://github.com/NVIDIA/Megatron-LM/pull/6675)，main，2026-10-16T21:05:11Z，merge `09462ef0e2aedb5699f7275f96768df7b429b281`。

role-tagged inference shards 允许 prefill/decode 使用独立并行布局，coordinator 只传请求与生命周期元数据；KV suffix 和最终 live SSM state 由 worker 经 NCCL 或 NIXL 直传。decode 可复用完整块前缀，部分末块仍归请求所有，收到所有 MP rank 完成结果后直接解码，不重放 prompt。已读协议说明、handoff ownership 和配置/负路径测试关键 patch；86 文件复合 PR 未逐行全审。

源码明确“计时器不是完成屏障”：已 claim 的读取不会因 offer 超时释放；取消期间源状态保持 pin，目标可隔离，NCCL source 还须本地 send handle 完成。主管只有确认 adapter 与所有 decode ranks 已退出且不能重启后才能退休 owner。前后两角色必须启 prefix cache、后端/模型几何兼容，native coordinator 仅文本，prompt/top-N logprob 不随 handoff 传，disaggregated RL 不支持 CP。未读得本 PR 可泛化吞吐对照。建议**深入阅读/先验证故障注入**，尤其 abort、disconnect、容量满与 refit；它不证明 vLLM P2P WRITE 的旧问题已解决。

#### NVIDIA Megatron：MXFP8 分片 scale 越界会静默改坏邻块

稳定ID `github:NVIDIA/Megatron-LM:pr:8247`；[#8247](https://github.com/NVIDIA/Megatron-LM/pull/8247)，main，2026-10-16T18:03:06Z，merge `adcd90f4b2a7f693c88b289ef29a0fc7c2980d57`。

MFSDP v2 `QuantizedDBuffer.quantize_` 曾把紧凑 scale view 交给 TE MXFP8 quantizer；当 local shard 行数不满 64-row chunk，TE 仍写 padded rows，可能越界到邻 rank scale 或触发 illegal memory access。现在先用 `get_tensor()` 构造 padded scale，量化后仅将 compact 区拷回；data plane 继续共享 view，只有需要 padding 的参数多临时分配/复制。

已读真实 `quantized_dbuffer.py` patch 与 canary 回归契约；旧顺序量化测试会把邻块损坏覆盖掉，故单看最终数值可能漏检。建议**优先核查**MFSDP2+MXFP8 的大 DP/不齐 shard 初始化，升级后复测邻块 canary 和目标 TE 版本，不把一般 FP8 训练都归入该故障。

#### NVIDIA Megatron：streaming load 降低量化推理加载峰值，严格拒绝训练路径

稳定ID `github:NVIDIA/Megatron-LM:pr:7405`；[#7405](https://github.com/NVIDIA/Megatron-LM/pull/7405)，main，2026-10-16T04:53:27Z，merge `6e7e6ed435e41af04fc41e7007ef862049153fdc`。

默认关闭的 `--stream-ckpt-dequant` 将全模型 BF16 副本改成逐目标 tensor scratch，目标读取完整后才整体量化；fully-parallel 只支持 broadcast 编码及 scales，避免重新量化。已读 planner/serialization、入口 guard、配置和测试 patch。作者单 GB200/TE2.17、64 个8192×4096 参数（BF16内容4GiB）实验，额外GPU峰值从4.00降至0.12GiB；这是合成加载峰值，不是整模型显存总量。

回归源自旧实现会用重新量化值初始化 FP32 master weights。本次要求 no-load-optim、禁止 finetune/load-main-params、传入 optimizer 直接报错；不支持 delayed scaling。tensorwise/NVFP4/blockwise 的 SwiGLU FC1 等仍可能 upfront（作者估计 dense 参数约40%），Megatron-Bridge 独立策略不读此开关。作者2/4 ranks 多配方/reshard测试，新的动态 minitron functional golden 仍待首轮生成。建议**考虑复现于推理加载**，记录仍 upfront 的张量和峰值，不能用来优化训练 resume。

#### NVIDIA Megatron：checkpoint 防误删与缓存身份修复

稳定ID `github:NVIDIA/Megatron-LM:pr:8165`；[#8165](https://github.com/NVIDIA/Megatron-LM/pull/8165)，main，2026-10-16T12:05:18Z，merge `438ef0025d805bdae486ec1bfaef2002ec3cf4d5`。

稳定ID `github:NVIDIA/Megatron-LM:pr:8251`；[#8251](https://github.com/NVIDIA/Megatron-LM/pull/8251)，main，2026-10-16T18:03:06Z，merge `f5186bfe732778dbbd667d6218b6504564933d16`。

`--save-retain-interval` 过去重启到更早迭代后可能删 tracker 指向的新 checkpoint（如保存100删2700）；现在仅删除 `0 < prev_iteration < iteration` 且不在保留间隔的旧项。PG distribution cache 也从最小 rank 命名改为有序全成员 SHA256，避免 [0,1,2,3] 与 [0,2] 等组共享最小 rank 时串缓存。

已读两个实际判定 patch。PG旧文件不会再读取，须 `--ckpt-pg-tensors-cache-create` 重建；训练入口 save/load group 一致性 assert 仍保留，不能当解除拓扑限制。作者8×H100披露多个checkpoint suite通过，本轮未独立运行。建议**优先采用正确性修复并重建缓存**，先演练回退/重启和 checkpoint 保留策略。

#### NVIDIA Megatron：batch-invariant Mamba 前缀缓存开放，同时修零预算等待

稳定ID `github:NVIDIA/Megatron-LM:pr:8214`；[#8214](https://github.com/NVIDIA/Megatron-LM/pull/8214)，main，2026-10-16T10:33:20Z，merge `3888b34882eb85aecf6ec9ecc3f4020b7608b497`。

稳定ID `github:NVIDIA/Megatron-LM:pr:8242`；[#8242](https://github.com/NVIDIA/Megatron-LM/pull/8242)，main，2026-10-16T02:29:18Z，merge `90dfb4be65e5f1331197c291e8a65eb9d00228be`。

batch-invariant Mamba 现在可发布完整 chunk 的 FP32 SSM state，在 block size 为 scan alignment 整数倍时复用；仍要求 FP32 state 与非 speculative 单 token decode。另修600-token prompt/512缓存/88计算预算被错误 round 至0的等待：计算预算排除 cached prefix，再返回完整 span。已读 context/mixer/scheduler patch。

作者单 GB200、TE2.19、tiny BF16 hybrid 的重排/尾段/graph/cache invalidation 比较 tokens 与logprobs exact；多GPU、跨TP/EP invariance和完整模型仍未证实。独立 #8242 修 batch-invariant `mean` 空维列表返回sum及`dim=None`异常，将空/None归约到全部维度。建议**优先做确定性回归**，覆盖缓存命中后一token尾段及全量mean；不要以batch-invariant标签替代数值验证。

#### NVIDIA Megatron：GTP 去掉双重舍入，RL UBR offload恢复池所有权

稳定ID `github:NVIDIA/Megatron-LM:pr:7889`；[#7889](https://github.com/NVIDIA/Megatron-LM/pull/7889)，main，2026-10-16T09:34:18Z，merge `54d99a30b649eece0191b766946ac2dea5f35519`。

稳定ID `github:NVIDIA/Megatron-LM:pr:6925`；[#6925](https://github.com/NVIDIA/Megatron-LM/pull/6925)，main，2026-10-16T04:53:27Z，merge `1ddc6e43421d807c70ba1d1a774cab4bd79bdd4e`。

#7889 在 opt-in FP32 accumulating GTP reduce-scatter 保持 output 与 async cache 为FP32，直到加入 main_grad 才舍入；BF16示例(1001+2)由双舍入1000恢复单舍入1004。仅group>2且非symmetric registered group进入该路径，新增FP32 shard buffer成本。已读dtype、cache及4/8rank回归patch。

#6925 则为非CUDA-graph RL、`perform-rl-step + rl-offload-optimizer-during-inference + use-nccl-ub`拆分持久parameter与可抛弃gradient pool，offload先全部deregister再释放，restore重新分配并显式register；拒绝param/grad alias的MXFP8布局。partial deregistration失败不可原地重试，restore注册失败仍可重试；默认丢弃并清零grad。作者历史合成GRPO反而慢0.9%/2.7%，多出2.10GiB headroom才是收益，最新配置生命周期未重跑完整E2E。建议**按需核查**通信注册trace与可用容量，不称免费加速或数值修复。

#### NVIDIA Megatron：MFSDP 梯度窗支持声明 multiplicity，完成性仍由调度器负责

稳定ID `github:NVIDIA/Megatron-LM:pr:7581`；[#7581](https://github.com/NVIDIA/Megatron-LM/pull/7581)，main，2026-10-16T19:55:16Z，merge `1fc08e50e339d58d68267da71ca2cf1f33fadadb`。

`fully_shard(..., grad_accumulation_count=...)` 可在构造FSDP unit时声明一个 reduction window 的参数梯度callback总数，避免 combined1F1B 的多个GraphTask共用权重时首批回调就reduce/reshard；未声明保持每trainable参数一次。计数归零后自动rearm，普通参数用post-accumulate hook，TE delayed wgrad走其callback。

已读PR与对应FSDP/countdown patch入口。它是总数记账，不逐参数追踪，不能定位漏掉哪个参数，也无法检测额外callback恰好完成另一个整窗。作者最新17个CPU模拟storage/communication检查；1000step/16GB200历史集成跑早于调度简化，不能当当前PR独立GPU长跑。建议**深入阅读并观察**后续#7582接入，窗口边界必须校验partial countdown。

#### NVIDIA Megatron dev：释放路由引用、全局批预取与 optimizer graph 数值契约

稳定ID `github:NVIDIA/Megatron-LM:pr:8231`；[#8231](https://github.com/NVIDIA/Megatron-LM/pull/8231)，dev，2026-10-16T20:04:45Z，merge `bbff3309a6cf81c6ed9af9936299372740fffe0f`。

稳定ID `github:NVIDIA/Megatron-LM:pr:5771`；[#5771](https://github.com/NVIDIA/Megatron-LM/pull/5771)，dev，2026-10-16T07:54:11Z，merge `c0d3b03f584d07eb1ae94c576604cef4637c1be6`。

稳定ID `github:NVIDIA/Megatron-LM:pr:8103`；[#8103](https://github.com/NVIDIA/Megatron-LM/pull/8103)，dev，2026-10-16T04:10:41Z，merge `87effd07995c95b21e1a53bcfcc11dabbff8c485`。

稳定ID `github:NVIDIA/Megatron-LM:pr:7811`；[#7811](https://github.com/NVIDIA/Megatron-LM/pull/7811)，dev，2026-10-16T14:48:06Z，merge `3b49df5b164239f6a75ae1a90504b809c21523c6`。

这四项进入 **dev**，不能写作main已发布。#8231 在combine最后使用后释放router概率图，真实patch先record_current_stream，再清理引用；Flex只清理概率，保留HybridEP graph需要的静态metadata。作者8/16层H200全重算峰值降10.8%/12.5%，DeepEP8rank降18.5%；CUDA graph、HybridEP/NCCL EP及A2A overlap未实测，不能将所有backend视为已认证。

#5771 扩充文本varlen的Nemotron-SFT/RP2/CodeForge字段、tool/schema/config/split处理，并让packing scheduler预取一个global batch；32GB200、4layer+MTP proxy mean48K/GBS512，eager36.26→33.65s、chunk graph34.96→32.21s，代价是host多一批内存，非VLM数据支持。#8103 chained optimizer把sqrt放设备float64，clip系数真正除法后仅向scale kernel舍入FP32，保持旧Python-float语义；作者20步Muon parity，无可测整步加速。#7811 dense fallback CSA topk按1GiB score预算分query行，full dense loss仍物化全矩阵；158passed/27skip的SM8环境未执行真实SM9+ cuDNN DSA kernel。建议**继续跟踪dev进入main与目标路径验收**。

#### NVIDIA Megatron：MLA RoPE 连续读取保留显式 FMA 舍入

稳定ID `github:NVIDIA/Megatron-LM:pr:8141`；[#8141](https://github.com/NVIDIA/Megatron-LM/pull/8141)，main，2026-10-16T23:00:20Z，merge `a4b2beebc08d503fb06e5f49b190b0d46487ffc5`。

原两次stride-2加载改一次连续load，再寄存器split；remove_interleaving也连续store。已读Triton patch，作者VR200、DeepSeek-V3 query BF16[4096,1,128,192]测18.7→11.3µs，仅算子。显式FMA配对避免编译器随BLOCK_H改变舍入；FP32 q+BF16 cos/sin+inverse的旧输出本来依BLOCK_H变化，本次固定为旧small-BLOCK_H输出。建议**考虑采用并验证dtype组合**，不说所有输入逐bit等价或整模型加速。

<a name="vllm"></a>

### vLLM

#### vLLM：Hopper Mamba2 与 LoRA 的 NaN 是不同路径上的编译 hazard

稳定ID `github:vllm-project/vllm:pr:61718`；[#61718](https://github.com/vllm-project/vllm/pull/61718)，main，2026-10-17T00:24:52Z，merge `783380af397ceeae91a40f4f759439eb8c1ab690`。

稳定ID `github:vllm-project/vllm:pr:61906`；[#61906](https://github.com/vllm-project/vllm/pull/61906)，main，2026-10-16T23:42:31Z，merge `523d74e2ac78867aa35d535835e14f78fb7433af`。

#61718 在CUDA SM90为Mamba2 chunk scan移除 `num_warps%4==0 && BLOCK_M%64==0 && stages>1` 的racy configs：register-A异步wgmma跨循环仍运行而输入寄存器被复写。作者H20-3e/Triton3.7.1复现23配置8坏→12配置0坏，95kernel tests，受测shape最优速度不变；**没有跑NemotronH服务E2E**。ROCm即使capability(9,0)也不应用此过滤。

#61906 针对SM90+fp32输入CAST_TYPE的LoRA expand，选config后强制stages1（亦覆盖自定义tuned配置）。作者H20/Torch2.14/Triton3.8：Qwen3-8B随机r64 LoRA prompt logprob NaN2047/2047→0，误差与HF+PEFT对照；最坏r128/T8192/add_inputs kernel慢21%，多slice实测基本不变。MoE LoRA类似cast路径尚未核实、留后续。已读两项实际guard patch及回归，建议**优先回归Hopper线上模型/LoRA**，不要将workaround等同Triton/ptxas所有问题已修。

#### vLLM：MiniMax M2 长工具参数流式解析减少重复扫描

稳定ID `github:vllm-project/vllm:pr:62035`；[#62035](https://github.com/vllm-project/vllm/pull/62035)，main，2026-10-16T22:46:03Z，merge `b088eba94431b711ffaa2f356e33361cf38b8fc5`。

新增受条件保护的arg_value_extender，避免每delta重跑整个converter和safe-prefix而出现二次CPU成本。作者Qwen3 token/delta、write_file16KiB、7次fresh-process A/B中位数5921→25.6ms；44K随机stream differential fuzz逐delta一致，测试覆盖split close tag、schema、未闭合和回退。已读extender与guard关键patch。

这是parser CPU累计耗时，不是模型TTFT或整服务吞吐；只有稳定的最后JSON value且opener不会被后续重新解析才延展。建议**考虑采用**于长代码编辑/tool参数，保留逐delta差分、断流和截断用例。

#### vLLM：SM100 FP4 DeepGEMM 跳过 padding 工作，但工作区容量不变

稳定ID `github:vllm-project/vllm:pr:59044`；[#59044](https://github.com/vllm-project/vllm/pull/59044)，main，2026-10-16T20:20:52Z，merge `68ebea789a4652a6b16665d50d79805866b933ec`。

专家GEMM用prefix-sum endpoints，packed SiLU/mul/FP8 quant只处理live rows；不增加count host回读或额外launch，workspace仍按capacity分配。SM120、FP8专家与无metadata dense caller维持既有路径。已读expert endpoint和activation masking patch。

作者8×GB200/DSv4-Flash/EP8-DP8-TP1/DeepEPv2条件下，prefix sums only→加mask的FFN425.5→385.7µs(-9.36%)；activation76.0→20.7µs不能直接当整FFN收益，更不能当服务加速。使用旧固定镜像overlay，后续main冲突合并只重跑静态检查未重跑GPU；GSM8K单次几题差被作者明确归为验收/噪声而非能力提高。建议**按稀疏度复现**full-buffer与empty专家，实际峰值显存不因少算padding自动下降。

#### vLLM：adaptive verification 少算bonus row，Mamba2接通varlen decode

稳定ID `github:vllm-project/vllm:pr:61451`；[#61451](https://github.com/vllm-project/vllm/pull/61451)，main，2026-10-16T19:42:49Z，merge `663e5356404332d6c764180aef41cc07015cd34a`。

稳定ID `github:vllm-project/vllm:pr:53036`；[#53036](https://github.com/vllm-project/vllm/pull/53036)，main，2026-10-16T18:24:00Z，merge `3dc6def36088a4139182e1a79cfe9b4e5953aa8b`。

#61451 V2 rejection sampler按request边界辨bonus logit，短draft不再做无用全词表归约，保留末行OOB guard。H100/vocab151936/K7、500次中位数：256request/1095logits adaptive block verify795→666µs(1.19×)，固定full draft1248.5→1250.2基本持平；这是sampler微测。

#53036 只做wiring：Mamba2 decoder已支持device offsets，开放CPU/device query length不同；ReplaySSM仍退出，FULL graph长度受max_decode_query_len限制。B200/Nemotron3.5Lightning30B/MTP8，GSM8K200 AV181 vs noAV184是小样本烟测，batch128/maxq8 ragged1330.5→900µs反映token工作量减少，不是固定token算子免费提速。已读两项核心patch。建议**考虑复现**真实acceptance与batch分布，保留graph、ReplaySSM负用例。

#### vLLM / NVIDIA：Kimi K3 专用 decode opt-in，合成接受率不能当真实服务结论

稳定ID `github:vllm-project/vllm:pr:61867`；[#61867](https://github.com/vllm-project/vllm/pull/61867)，main，2026-10-16T02:50:42Z，merge `642c18d64c4cb6db3ded8fbe1786718cd4005fcb`。

新增 `--additional-config '{"decode_backend":"latency_specialized"}'`，localized CuTeDSL runtime；不支持的prefill/大batch/PIECEWISE回stock并补final norm。已读model选择/回退、runtime入口及性能披露，50文件kernel实现未逐行全审。支持GB300 TP8/16，NoSpec至B64，DSpark3–7draft、总verify≤64token及DCP=TP或关闭。源码还要求MRV2/FULL decode graph、PP1、无EP/DBO/LoRA/sleep/AV/ReplaySSM/RecoverSSM；小batch回退和部署不受支持时直接拒绝是两回事。

作者FP8 KV/FP32 SSM、TP16/ISL1/无投机B1，decode7.38→3.90ms(1.89×)；DL7多处吞吐测试采用**synthetic acceptance6**，不是自然acceptance服务流量。质量评估则标准rejection/probabilistic drafting，TP/DCP及并发与stock并不全配平。建议**小规模复现**同拓扑真实draft与长上下文，分别报告decode step、prefill和E2E，不用标题1.5–2×作全场景承诺。

#### vLLM：DeepSeek V4.1 的 token分片投影与 PCP 支持有不同成熟度

稳定ID `github:vllm-project/vllm:pr:58872`；[#58872](https://github.com/vllm-project/vllm/pull/58872)，main，2026-10-16T02:31:35Z，merge `05ad0a8ace9bfdbf29acb0554c80d6c4895a215b`。

稳定ID `github:vllm-project/vllm:pr:59857`；[#59857](https://github.com/vllm-project/vllm/pull/59857)，main，2026-10-16T06:17:52Z，merge `fa75d3456d60a5ffe923818c4b495295675b38e1`。

#58872 对SM90、无SP/LoRA/microbatch/batch-invariance且TP NVLink连通的≥2048token batch，分片fused_wqa_wkv计算后all-gather。H20 TP8/8192token此投影1794.2→346µs，但服务TTFT仅约降4–5%，DSpark服务测试用synthetic AL3.51。2048门槛对H100/H200是缩放估计未实测；SM100明确不启。已读gate和projection patch。

#59857 PCP仅CUDA MRV2 FlashMLA sparse、文本colocated单实例，PP=DP=DCP1，可TP×PCP；不支持PD/KV connector、prefix cache/bounded replay、spec、graphs、LoRA/MM等。作者在review拆分后明写**Validation TODO**：旧global-order compressor GPU结果不证明当前rank-local revision，且依赖另一个generic MoE padding fix #60946。建议**继续观察PCP验收**，不把它与较成熟投影优化合称“多并行已全面可用”。

#### vLLM：DP wave唤醒归engine所有，解决多前端pause/resume后挂起

稳定ID `github:vllm-project/vllm:pr:59559`；[#59559](https://github.com/vllm-project/vllm/pull/59559)，main，2026-10-16T23:41:55Z，merge `a94c3157119e52adacb4d580d4e950983d14e353`。

旧前端FIRST_REQ可能先唤醒仍paused的peer而被丢弃，随后真实ADD到另一rank，其误以为wave已启动而独自进DP collective挂起。现在idle且unpaused engine收到请求就announce start_wave，coordinator去重，Python/Rust前端移除乐观唤醒。已读engine/coordinator及frontend核心patch。

作者H200、多API server、MP/Ray、pause模式及DP16等测试；这不是减少一条消息的通用提速，兼容要求新frontend配新engine，正常整包部署满足。建议**优先验证**异步RL pause/resume、elastic EP切换与延迟ADD race，混版本前后端需单独验证。

#### vLLM：Beam发布镜像可用，完整模型PR仍open；main只合入组件

稳定ID `github:vllm-project/vllm:pr:62042`；[#62042](https://github.com/vllm-project/vllm/pull/62042)，main，2026-10-16T22:39:47Z，merge `596e53067548fab65de6fc3abdcd020f752e7ae7`。

稳定ID `github:vllm-project/vllm:pr:62052`；[#62052](https://github.com/vllm-project/vllm/pull/62052)，main，2026-10-16T21:08:30Z，merge `153a17166e5d07bc38184dedeea542d2e8885e55`。

稳定ID `github:vllm-project/vllm:pr:61978`；[#61978](https://github.com/vllm-project/vllm/pull/61978)，main，open，尚未合入，head `00fe254cd01b1d6b9f69370750f113cffb90e474`。

10-16 [Inferact/Reflection/NVIDIA官方博文](https://vllm.ai/blog/2026-10-16-reflection-beam)指向专用 `vllm/vllm-openai:reflection-beam`。完整Beam+DFlash2 #61978仍open，依赖FlashInfer nightly；main的#62042只是router5120×256/BF16 GEMM output、#62052只是显式beam parser，两者不构成E2E模型支持。原主PR及两slice已核，博客AgentX固定短commit为8e0922，未锁该镜像digest。

作者GB200 TP8、SPEED-Bench低熵greedy，adaptive c128输出15012→16632tok/s(+10.8%)而c1 708→707；需显式启adaptive。AgentX约88K total tok/s/GPU包含cached input，命中91.4–95.6%，不是新生成速度。建议**先锁镜像/依赖复现**，保留主线合入和发布边界；其agent kernel优化相对最初未优化实现，不是对成熟通用baseline的2.71×。

#### vLLM：structured output在reasoning阶段屏蔽非法隐式结束符

稳定ID `github:vllm-project/vllm:pr:61574`；[#61574](https://github.com/vllm-project/vllm/pull/61574)，main，2026-10-16T04:53:42Z，merge `4ac1f8d2ac14911e9709a2fd57ca1845f2a7ecc2`。

engine-based reasoner曾允许GLM的tool_call等既结束reasoning、又构成正文首token的marker，若该request grammar拒绝它，采样后产生HTTP500。现在先询问grammar，reasoning bitmask仅屏蔽grammar无法接受的implicit ends，显式think-end及required/auto工具合法marker保持可用，draft/bonus同样处理。已读adapter/manager缓存与bitmask patch；作者CPU真实组件189passed/1xfail等，6636项宽测有环境排除，未做模型GPU服务验收。建议**优先回归JSON schema与tool_choice组合**，不将它写成全面禁止思考中工具调用。

#### vLLM：External LB模式接入elastic EP，需要外部编排执行生命周期

稳定ID `github:vllm-project/vllm:pr:43202`；[#43202](https://github.com/vllm-project/vllm/pull/43202)，main，2026-10-16T01:29:02Z，merge `dd9e07a25febb05f1a44f2ef9e6c56803ee1a9b6`。

为每次scale临时建立TCPStore/handshake coordinator，复用已有EP worker状态机；不会替外部系统创建/销毁Ray actor。扩容须所有旧rank调用scale，外部启动新rank，完成后才入LB；缩容先移出LB，再让所有旧rank进入scale，移除rank最终退出。启elastic_ep且external_lb、DP>1，hybrid LB仍拒绝，PP也仍不支持。已读parallel兼容gate与控制流说明；作者Qwen3-30B-A3B多次DP2–8增减烟测可继续服务，未提供硬件/性能/故障恢复完整矩阵。建议**小规模验证编排原子性与失败退出**，不能把单rank API返回等同整集群全局事务完成。

<a name="verl"></a>

### verl

#### verl：FSDP2 CPU offload 改阻塞D2H，避免repadding持久损坏参数

稳定ID `github:verl-project/verl:pr:8187`；[#8187](https://github.com/verl-project/verl/pull/8187)，main，2026-10-16T03:18:32Z，merge `edd2c63b6a77b551c20cdded1c802d82a9cc751d`。

uneven shard的FSDP2 `_apply/reset_sharded_param` 可能在D2H完成前从CPU目标拷贝到新padded storage，故整个model.to返回后再sync也无法补救已复制的坏值。runtime offload改 `non_blocking=False`，initial full-state load复用helper且empty_cache=False；H2D仍nonblocking，但存储pinning变化意味着旧overlap不能保证。已读真实helper/入口patch及回归。

作者NPU Torch2.10独立复现10/20 rank-trial错，CUDA2.11排队matmul后50/100错；最终变更13CPU mock policy检查与双NPU默认/非默认stream数值回归通过，CUDA最终数值suite列入CI但未宣称本轮本机已跑。下游两节点Qwen3-Omni通过是正确性证据，未量化阻塞成本。建议**优先采用并做参数精确roundtrip**，不要将性能重叠保留为默认假设。

#### verl / TPU：100步更快但更陈旧，升级package不是提速的充分解释

稳定ID `github:verl-project/verl:pr:8259`；[#8259](https://github.com/verl-project/verl/pull/8259)，main，2026-10-16T03:31:53Z，merge `d400b3eccc304a5f0c64817b1f295462752b6b48`。

只改TPU GRPO example：google-pytorch/torchtitan ea7fec0f、JAX/JAXLIB0.11.1、libtpu0.0.50、torchao0.17、plugin f5876f75；core verl TorchTitan引擎未变。Qwen3-4B脚本默认warmup batches1→2、maxbatched1536→4096、padding bucket1024→512，分别补decode slots、缩partial-rollout重prefill、减少trainer padding。已读脚本默认值patch、3文件变更清单与原始实验说明。

2×v6e-4（4train+4rollout chips）、100steps，相同新image旧设置steps6–100均值25.02s→22.18/21.97s；但旧main+新设置已21.57s，不能把11–12%归JAX/libtpu升级。多一批排队使前期reward约0.80vs0.86，达到0.85的step-time19.1–19.4vs15.8min，**学习到目标奖励更慢**；128验证样本的step100准确率差不显著。作者fork TPU CI与100步通过，私有TorchTPU registry仍是复现门槛。建议**同时比较reward/time与step/s**，重视新鲜度可保留warmup1。

#### verl：禁止engine_kwargs覆盖受管vLLM拓扑参数

稳定ID `github:verl-project/verl:pr:8260`；[#8260](https://github.com/verl-project/verl/pull/8260)，main，2026-10-16T07:40:12Z，merge `6a54261bf517aa4a00ad6e59b7a746c6813d05ed`。

TP/DP/EP/LoRA/routing replay参数过去有的用户kwargs覆盖verl，有的被静默改回，enable_expert_parallel=True甚至会变false而每卡持全专家、KV不足。现在server启动即拒绝这些kwargs（含hyphen拼法），报应改的verl config；其他参数仍透传。已读managed map与preprocess guard，作者33CPU CLI tests通过。建议**升级前迁移配置**到rollout.tensor_model_parallel_size/data_parallel_size/expert_parallel_size及model.lora，不将启动报错误判资源不足。

同窗[#7958](https://github.com/verl-project/verl/pull/7958)（稳定ID `github:verl-project/verl:pr:7958`，main，2026-10-16T08:48:00Z，`18cfb74852a053a27dcf301d3318a093be81b628`）将hybrid resource上是否构造rollout的判定由hybrid_engine改enable_switch；该PR正文没有实测说明，暂仅作为代码跟踪，不据标题扩展运行保证。

<a name="slime"></a>

### slime

#### slime / THUDM：GLM-5.3 Flash原生Megatron训练接入，公开验证仅三步

稳定ID `github:THUDM/slime:pr:2478`；[#2478](https://github.com/THUDM/slime/pull/2478)，main，2026-10-16T11:57:43Z，merge `5bec7a10859f29ae1bf2c9ddd5a67aa70ead0fac`。

新增KDA、no-RoPE sparse MLA、mHC、vision wrapper与HF双向权重映射，镜像nightly-dev-20261016b（SGLang0.5.21/MCore0.19.2+patch）。原生attention要求TP1，CP>1须contiguous `--allgather-cp`；64H100 recipe CP2/PP4/EP16，frozen indexer、文本训练冻结vision。已读README、启动/guard、mapping与optimizer patch，33文件复合PR未逐行审计全部kernel。

作者300B CP1与CP2各3steps/GBS128/非零LR/R3，64inference ranks初始权重相等，response token replay audit通过；配置response32768但实测最长约5338，**没有32K输出/完整多模态训练长跑证明**，CP1使用random smoke reward。重要正确性补丁：checkpoint reload须刷新已FP32且CPU offload的optimizer副本，否则首step把mHC/KDA覆盖成旧初始化；resume映射也补全。权重同步587→61groups、97.9→21.5s是特定64H100实验，不能当训练总步提速。建议**先锁镜像与patch做resume/weight roundtrip**，再扩训练长度和多模态。

#### slime / THUDM：奖励在filter前按trajectory采样，避免只看筛后高分

稳定ID `github:THUDM/slime:pr:2475`；[#2475](https://github.com/THUDM/slime/pull/2475)，main，2026-10-16T03:33:47Z，merge `57ff5d669be6b998411f33b04ecef5e297538faa`。

新增 `rollout/raw_reward_before_filters`，在自定义filter可能改reward前统计terminal trajectory reward，跨context-window segments只算一次；worker只回传sum/count，包含被丢弃候选，同步及两类async路径收集。旧恢复结果缺stats或尚未评分导致候选不完整时**不发布部分均值**；未启filter时也不额外报该项。PR正文为空，本轮结论来自8文件实际patch与tests，未独立跑。建议**考虑采用**，并列raw reward、accepted reward及drop ratio定位筛选偏差，不能把筛后奖励上升当policy能力增益。

<a name="sglang"></a>

### SGLang

#### SGLang：PD deferred release必须阻止flush，避免KV页双重释放

稳定ID `github:sgl-project/sglang:pr:44386`；[#44386](https://github.com/sgl-project/sglang/pull/44386)，main，2026-10-16T02:17:52Z，merge `f9df33494a3bd861d1b4153b109154048d318bf1`。

failed/aborted PD transfer默认延迟释放，KV页/request row/tree lock可能仍由远端write占用；旧is_fully_idle只看queue，held request已移出queue，abort→pause_generation→update_weights(flush_cache)可把held页释放后再deferred release，造成free list重复/新请求共享页。实际patch仅在idle gate加has_pending_deferred_releases，连同storage attach/detach和release_memory_occupation一起保护。

已读scheduler一行修复与既有测试调整；作者H200真实allocator复现旧272/256可用页、一次alloc重复16页，新版拒flush并恢复256，14cleanup tests通过。该PR不重新设计30s超时drain协议，不证明所有late WRITE寿命闭环。建议**优先检查**异步RL/PD的flush与释放组合，禁止把暂停当传输完成。

#### SGLang：存储预取超时从首次阻塞计，deadline不等于数据缺失

稳定ID `github:sgl-project/sglang:pr:44614`；[#44614](https://github.com/sgl-project/sglang/pull/44614)，main，2026-10-16T17:37:12Z，merge `453b251dbd0f9ca6e3de2ddec99a4ce7fe723e0f`。

稳定ID `github:sgl-project/sglang:pr:44605`；[#44605](https://github.com/sgl-project/sglang/pull/44605)，main，2026-10-16T17:36:47Z，merge `989ccf4863c6f9aa2ed68ffd32abbaacbc755a06`。

#44614 为timeout policy维持request级预算，首次blocked admission poll开始，refetch不续命，rank协商过期后取消retry并防晚到rank重发。完整KV只等sidecar时不cut。deadline标记区别storage_transfer，避免把超时误当存储不存在并“治愈”错误belief；已读request state、timer与deadline关键patch。作者309passed/277subtests CPU，没有服务提速claim。建议**考虑采用并监控deadline归因**。

#44605另给KV backend `should_verify_kv_checksum(peer)` opt-out：TP不齐peer的local-byte digest不一定可比。默认True且内置backend未改，返回False会未验证放行，**不是更强checksum校验或自动修复所有TP mismatch**。建议仅由掌握布局的backend使用，另建正确数据/索引验收。

#### SGLang：CUDA bounded replay扩DP并修draft SWA缓存所有权

稳定ID `github:sgl-project/sglang:pr:43813`；[#43813](https://github.com/sgl-project/sglang/pull/43813)，main，2026-10-16T22:38:34Z，merge `aaf5f397a5a6565ad2cbcfe64bab56bdebbe4b75`。

稳定ID `github:sgl-project/sglang:pr:44673`；[#44673](https://github.com/sgl-project/sglang/pull/44673)，main，2026-10-16T08:18:59Z，merge `fad7d95a7f1a942e039f74eb2d319da29311b8cb`。

稳定ID `github:sgl-project/sglang:pr:44783`；[#44783](https://github.com/sgl-project/sglang/pull/44783)，main，2026-10-16T22:05:45Z，merge `8e3f11b2193d18a62e9480c29e1e0514e9192094`。

稳定ID `github:sgl-project/sglang:pr:42716`；[#42716](https://github.com/sgl-project/sglang/pull/42716)，main，2026-10-16T17:45:17Z，merge `80c54e6ac52c769cb66c4bda53942898965e9bc3`。

#43813 encoder SWA replay扩至attention TP1的DP，计入折叠tail行、同步round、无hit rank idle参与，至少每attention-DP rank128prefill tokens；TRTLLM/prefill graph仍不支持。2GB300 TP2/DP2/EP2、native DSpark GSM8K1314样本，replay1278→1281 correct、0errors，单次几题差不称质量提高。

#44673 CUDA draft SWA从radix组件改request ring（窗口+draft长度），prefix hit后valid_from屏蔽旧slot残留，避免draft SWA被驱逐连带切断target仍可用MainKV；HIP仍paged，#42716另跳过其draft SWA匹配gate，不能混称HIP也改ring。#44783 window commit仍按全输入rows launch，但mask掉discarded rows的KV loads/stores并同launch写tag，**没有缩launch grid**。16K micro142.24→14.79µs；4GB300/TP4EP4的replay E2E单baseline两candidate中位数约降6.9–7.4%，用simulated AL5.4，未重跑最新main。已读布局/DP计数/commit关键patch，建议**结合slot复用、graph metadata和真实acceptance回归**。

#### SGLang / AMD：decode radix复用减少PD payload，SWA/状态仍整块传

稳定ID `github:sgl-project/sglang:pr:30929`；[#30929](https://github.com/sgl-project/sglang/pull/30929)，main，2026-10-16T09:39:28Z，merge `f05da43d44067e4e4f95302b6d3bd40f7deb3b1d`。

显式 `--disaggregation-decode-enable-radix-cache`现在允许DeepSeekV4 ROCm unified KV BF16或FP8；仅共享compressed full-attention pages，request-scoped SWA ring和c128 state整块传。已读validation guard，paged KV、hierarchical cache/SWA compress等仍拒绝，flag默认False。

作者MI355X单node1P1D/MoRI/DSpark，120shot约18K多轮GSM8K，长前缀BF16 payload190.5→52.7MiB、FP8131.3→42.2MiB（-72%/-68%），是传输字节而非TTFT/吞吐等比改善；28–31MiB固定状态限制收益。不同配置准确率有升有降，off/on并非逐输出一致；DP需session保持同decode rank，内置metric还受旧状态计数bug影响。建议**按会话路由复现传输计数与质量**。

#### SGLang / AMD：DeepSeek V4优化系列的默认、依赖和负收益要分开

稳定ID `github:sgl-project/sglang:pr:44016`；[#44016](https://github.com/sgl-project/sglang/pull/44016)，main，2026-10-16T08:50:17Z，merge `ab5858ac8a8f2eb4dac3835e22e1a3f1cf9f1504`。

稳定ID `github:sgl-project/sglang:pr:44003`；[#44003](https://github.com/sgl-project/sglang/pull/44003)，main，2026-10-16T08:58:01Z，merge `f0ff18353d2c6bd8d0c1896563e197d3f4fcfa2b`。

稳定ID `github:sgl-project/sglang:pr:44004`；[#44004](https://github.com/sgl-project/sglang/pull/44004)，main，2026-10-16T09:09:33Z，merge `9fde8bcc466f5f8b5c8222d266feff95551c71da`。

稳定ID `github:sgl-project/sglang:pr:44006`；[#44006](https://github.com/sgl-project/sglang/pull/44006)，main，2026-10-16T09:16:07Z，merge `6efc96abf975cb0a2658f7cdfa164c3e5ab5eb95`。

稳定ID `github:sgl-project/sglang:pr:44007`；[#44007](https://github.com/sgl-project/sglang/pull/44007)，main，2026-10-16T09:28:32Z，merge `42eb556e66f1d5f288ee8b232a8a545431a25226`。

稳定ID `github:sgl-project/sglang:pr:44405`；[#44405](https://github.com/sgl-project/sglang/pull/44405)，main，2026-10-16T08:13:38Z，merge `bdc4ec15ce4fbe2a05c43b3d6b79d2d657232863`。

#44016 gfx950 FP8 unified KV decode/verify默认FlyDSL（auto选aiter/in-tree），128local heads默认不启该路径；split model/HCA persistent另有默认on但作者受测多格ITL几乎没差。依赖aiter符号/pin与carry kernel，15文件之一大kernel patch未返回，未审其全文。#44003把FP8 verify SWA ring写并入norm/RoPE，作者183launch/step省去，DSpark block6测step降1.3–2.1%。#44004小MoE sort仅32<P≤256用histogram+bitonic，受测E385/topk7 launch14.8→5.2µs，decode约降4%；不是所有batch都换kernel。

#44006 HIP V4.1 low-ratio rowgroup默认on但C4 rowgroup默认off，缺aiter kernel回退warning；C4部分小verify shape约慢5%。#44007 e8m0 mxscale **默认off**，仅power-of-two scales/whitelisted dense shape且aiter支持；作者ITL降1.5–6.5%，output throughput仍在约5%跑间噪声。#44405 HIP仅当backend无需CPU seqlens且无confidence-budget准备时跳host同步；simulated bonus fold还要求设置simulate AL，其额外1–3%不能当自然acceptance收益。已读关键选择/guard patch及原始body，未独立验证整系列。建议**逐开关做目标shape A/B**，保留FP32-scale/asm fallback，不将系列百分比累加。

#### SGLang / NVIDIA：2shot_push通信调优覆盖SM100，但B300证据不能外推所有系统

稳定ID `github:sgl-project/sglang:pr:42811`；[#42811](https://github.com/sgl-project/sglang/pull/42811)，main，2026-10-16T20:43:19Z，merge `07950be745136dc90de8a6f7f61d6fc734ae1858`。

稳定ID `github:sgl-project/sglang:pr:44387`；[#44387](https://github.com/sgl-project/sglang/pull/44387)，main，2026-10-16T05:50:34Z，merge `83940471b46911dcb28da800235d7d997e0cdd8c`。

稳定ID `github:sgl-project/sglang:pr:44622`；[#44622](https://github.com/sgl-project/sglang/pull/44622)，main，2026-10-16T20:53:42Z，merge `2948662026a8addefdba7697060aa109d821ec23`。

#42811 CustomAllReduceV2新增Lamport push-plane两段reduce-scatter/gather，固定rank顺序归约；SM100 TP4在640KiB以上、TP8在256KiB以上至8MiB优先2shot_push。slot sizing使每rank symmetric memory TP4 24→32MB、TP8 12→32MB。8B300 BF16/graph微测TP8约1.19–2.01×，没有整模型服务倍率；B200/GB200继承SM100表但作者列为待复测。已读picker/layout关键patch与测试矩阵。

#44387 MNNVL fusion在跨node时核group是否同NVLink clique，失败提前禁用并给拓扑warning；NCCL fallback仍发生，**没有修跨clique通信成MNNVL**。旧verify3.65×退化是生产定位，新跨block真实跑未做。#44622 CPU>2rank的MLP同步行由gloo ring allgather换gather+broadcast，2rank/device tensor保留原路径，8rank子组bitwise测试；明确无benchmark。建议**先查group拓扑和实际trace，再复现阈值**。

#### SGLang：统一内存补TRTLLM-MHA graph/spec边界与compaction首用OOM

稳定ID `github:sgl-project/sglang:pr:42930`；[#42930](https://github.com/sgl-project/sglang/pull/42930)，main，2026-10-16T09:24:07Z，merge `8d229d96e00b3c02641fc12b7ee441c4f355bd25`。

稳定ID `github:sgl-project/sglang:pr:44592`；[#44592](https://github.com/sgl-project/sglang/pull/44592)，main，2026-10-16T23:47:51Z，merge `36cf0283460cda42fcb054a1d2b7b604a82f8c73`。

#42930在per-iteration KVLocPlan基础上开放trtllm_mha decode graph与相应spec rails，handover页清零避免0×NaN，要求packed page table，分离fmha_v2 scratch与XQA semaphore，prefill只拷实际请求页。SM90作者GSM8K和poisoned pool测试；部分EAGLE3/性能实验叠四个独立修复，SM100相关draft未在该硬件跑，不能宣称本PR单独完成所有组合。已读backend/gate/zeroing关键patch，11文件非全行审计。

#44592启动在sink page0预热compaction commit并同步，按physical page envelope搬移，默认每launch≤4096pages且同一次urgent compaction在返回前完成所有chunk；同stream先copy再更新映射。只对显式unified-memory pool，启动额外分配在后续pool sizing前计入；作者单GPU字节smoke，未报受控服务提速。建议**优先核高压首次compaction与page reuse**，不要把启动预热视为无资源成本。

#### SGLang：MegaMoE W4A16 / SiTU是opt-in兼容集成，默认仍W4A4

稳定ID `github:sgl-project/sglang:pr:44036`；[#44036](https://github.com/sgl-project/sglang/pull/44036)，main，2026-10-16T07:38:51Z，merge `7098ec387c9f9d9f85a0c2c16bc1233f40f1d407`。

复用 `SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16=1`，serialized ModelOpt与nvfp4_online保留NVFP4权重、BF16 activation/EP/combine；SM100/103，attention-DP=TP=EP，W4A16需BF16。默认FC2外部FP32累加，opt-in in-kernel可允许非确定BF16 atomic。reload维持allocation/graph pointer及workspace owner；FlashInfer0.7.1的W4A4输入scale变化需重启，不能任意热换。

已读adapter与reload guard关键patch及原始验证说明。Kimi K3 SiTU wiring存在，不含完整K3 E2E或专门NVSHMEM环境验收，Nemotron ReLU²不支持。单node还设local peer NVSHMEM默认，显式env不覆写、跨node不自动设。作者GPU结果早于rebase、最新CPU stub harness不能替代CUDA；**未做受控serving性能对照，不claim提速**。建议按模型activation与热更新契约小规模采用。

#### SGLang：MiniCPM-V4.7与DeepSeek V4.1多模态输入契约

稳定ID `github:sgl-project/sglang:pr:41993`；[#41993](https://github.com/sgl-project/sglang/pull/41993)，main，2026-10-16T05:24:49Z，merge `2a08e261ecd4cdfb04ec354f9005f18b88108b7d`。

稳定ID `github:sgl-project/sglang:pr:44751`；[#44751](https://github.com/sgl-project/sglang/pull/44751)，main，2026-10-16T15:39:51Z，merge `a038a0dca05cefca1e0edc4ac43e3bfbec2e064c`。

#41993接入MiniCPM-V4.7 1B与35B-A3B，canvas3D MRoPE按model_type路由（checkpoint version字段仍旧），per-request downsample/slicing进入media profile与cache identity，避免相同像素不同几何误共享；补quant ignore命名映射、draft config嵌套读取。已读processor/config关键patch，未逐行审18文件。作者4GB300全GSM8K与5种image probe、19305布局fuzz，不能由语言GSM8K泛化全部视觉能力；FP8/INT4工件缺processor文件时需显式base tokenizer-path，trust-remote-code及extra_buffer仍需。

#44751将上游HF `314113f8`编码修复搬入DSv4.1 encoder：保留tool-result image顺序与当前turn thinking。建议**同步模型工件revision与服务编码器**，回归工具图像混排/多轮上下文，不仅看模型加载成功。

<a name="torchtitan"></a>

### TorchTitan

#### TorchTitan / Meta：多轮Muse renderer保留采样token，避免重复整段history训练

稳定ID `github:pytorch/torchtitan:pr:5448`；[#5448](https://github.com/pytorch/torchtitan/pull/5448)，main，2026-10-16T22:18:41Z，merge `215ddfe4af1e1c97d17693327ec0801ab9f40a92`。

稳定ID `github:pytorch/torchtitan:pr:5464`；[#5464](https://github.com/pytorch/torchtitan/pull/5464)，main，2026-10-17T00:07:21Z，merge `c9c86265166a111ec641fbb1105066186f6c2200`。

#5448 bridge_to_next_turn不再总None；沿用previous prompt+completion原tokens，只渲染新tool/user tail，避免解析strip whitespace/重写tool JSON后不再保持prefix而另起训练sample。必须completion以EOT闭合、无assistant tail、tool results与calls数量/顺序/可选name匹配，否则回退全render。已读bridge实现和负tests。作者131K上下文SWE两次各3步，samples312/291/264→64/100/76，forward-backward110/86/62→58/33/16.9s；rollout不同，不能当受控吞吐倍率。

#5464按10-15发布template revision0fb02478更新low16→8、medium64→32、拒minimal，tool schema缺additionalProperties默认false。15个对照prompt/token ids由12不符→0，但namespace tools/deferred loading仍未实现。建议**优先对齐template并验证token prefix与loss mask**，不把15例相等称完整模板功能等价。

<a name="extended"></a>

### 扩展项目

本节时间为 UTC。除单列正式 release 外均指所列默认分支 commit，不能据此认为已进入现有 pip/容器版本。已阅读 PR 正文及对应变更；PyTorch 的 ghstack PR 另回到真实 main commit 验证，避免将 `merged=false` 误判为代码未落地。以下测试均为作者报告，本轮没有 GPU/NPU 运行复现。

#### TRL AsyncGRPO：把训练概率重新归一化到采样时的候选集

**项目/版本。** TRL [#7627](https://github.com/huggingface/trl/pull/7627)，10-16 07:42:55，main `a516f43a015177986c5d2f385600aa43b1eafa99`（完整SHA见本地来源账）。它晚于 v1.16.0 首发，采用时应检查实际构建是否含补丁。

**问题与增量。** top-k/top-p rollout 的 processed log-probs 在截断后的集合归一化，旧 trainer 却用完整词表，导致 importance ratio 系统性偏低。现在从 vLLM token-in/token-out 端点取得每 token 的候选 ID，经过 reconciler、tree packing、collator 传到 fused LM head，以同一支持集计算概率与梯度；熵仍按完整词表计算。已读 `ChunkedLogProbFunction`、`support_ids` 的移位/掩码与 worker/server 检查变更。

**证据与边界。** 作者 Qwen3-0.6B/vLLM0.31、top-p .95/top-k512 的 trainer−sampler log-prob 偏差 −.028→−.0005。R1-Distill-Qwen1.5B、Sanity-Test-R1D-1.5B、128 completions/step、lr1e−6、300步的小实验中，末50步 reward .28→.71，接近未截断参考 .67；这是指定 sanity bed 的训练结果，没有独立评估/多种子泛化证明。需 vLLM≥0.28、`--return-sampling-mask --enable-scale-out`；top-p<1 还需正 top-k 限制候选数。旧服务会警告并回退，外部自主管理且不记录集合的 OpenEnv loop 会拒绝截断配置。

**建议：优先核查并考虑采用。** 适用于用 processed log-probs 做截断采样的异步RL；先在固定权重上比对 sampler/trainer 概率及梯度，再看长程 reward。不能仅凭 ratio 接近1宣称训练无偏的所有条件均已满足。

#### DeepSpeed：词表并行蒸馏统计与 fused linear CE，用重算换显存

**版本。** master 10-16 [#8791](https://github.com/deepspeedai/DeepSpeed/pull/8791) `53f59dbf834027c8f1981d2793702edf615c580c`；[#8792](https://github.com/deepspeedai/DeepSpeed/pull/8792) `036230e4f97c728204f89b1f8214addc9465b5c5`。

**具体增量。** 前者在本地词表分片上计算 token/top-k log-prob、entropy 与双向 KL，不 all-gather 完整词表；用 MAX/SUM 归约 token 统计，反向局部解析计算。输出 FP32 并在 TP 组复制；假设各 TP rank 对同一 loss 反传，只支持一阶梯度，student/teacher 必须同组同分片。全 −inf 行输出 NaN 但梯度为零，调用者仍要正确掩码。**本地 logits 仍存在**，不能把它称为 fused-linear 蒸馏。

后者直接接受 hidden/本地 head weight：前向按 token 分块、反向按词表块重算 logits，消除完整 logits/softmax 激活，`dh` FP32 累加后 TP all-reduce，不保留整份 FP32 dW 副本。代价是额外 logits GEMM；接口支持不等长词表分片和 bias。已读两个实现与 full-vocabulary oracle 测试。

**证据。** #8791 的 1×Intel Arc Pro B70，T8192、local V16032、BF16、排除输入的前反向峰值：reverse KL 3012→698MiB，但37.2→70.2ms；不是同时省显存并提速。#8792 的 **单卡本地分片模拟**，H4096、BF16、local V按256K/TP2、T16K：28672→1921MiB，633.6→678.9ms；这些显存/时延不是完整2-rank训练。另有XPU/XCCL TP2不等分片值与梯度测试。原作者仍把 HF forward 集成、整模型多rank内存、ZeRO2/3 head hook 交互列为后续。

**建议：考虑复现 functional API。** 大词表长序列 SFT/蒸馏可能直接受益；先测自己的 TP 通信、ZeRO hook 与 step time，不能将单卡 local-shard 表直接用于集群容量承诺。

#### DeepSpeed NPU inference：修复静默 RoPE 错位，KV 改成增长式预分配

**10-16 master [#8776](https://github.com/deepspeedai/DeepSpeed/pull/8776)**，`a7c01f574e0af1206b384f6336716005d64bfa34`。旧 decode 忽略 seq_offset，始终用 position0；prompt 的 interleaved cos/sin 配错默认半区配对，`rotate_half=True` 又没有对应实现。修正绝对位置、两种配对及 token 推进，缓存 FP32 inv_freq；KV 从每步 cat+contiguous 全历史复制改为原生 BSH 预分配、单行写入、容量2倍增长，并修 GQA head reshape。

作者2×910B4(64GB)、torch2.10.0/torch_npu2.10.0.post7/CANN8.5.1、FP16，3 warmups+7次交替顺序取中位：32K **单个 attention decode op** 5.490→2.311ms，KV存储子路径1.863→.176ms。2.4倍和10.6倍分别对应这两层口径，**不是整模型生成吞吐**。Qwen3-4B/36层、8KV/32Q head、rope_theta1e6 的单输入32 token与HF相同，logit仍有FP16误差；未覆盖任意模型/长程质量。

**建议：优先回归正确性。** 若使用这个 DeepSpeed NPU inference op，应先测 offset、配对方式和GQA，再测容量增长点；普通HF/vLLM-NPU路径不能据此推定受同一bug影响。

#### PyTorch FSDP2 与 SDPA：特定布局与跨 graph 边界的三项修复

**均为10-16 main真实提交。**

- **FSDP2 post-forward小mesh**：[#200985](https://github.com/pytorch/pytorch/pull/200985)，[main `841e3dc165963ca6550560eab70a84d1c166b9eb`](https://github.com/pytorch/pytorch/commit/841e3dc165963ca6550560eab70a84d1c166b9eb)。`reshard_after_forward=int` 的扁平 shard 过去按 Shard(i) 形状重组，使i>0触发IndexError；现在按flat layout拷贝，并在修改任何参数前整组验证。dim0不可被新mesh整分、DTensor与某些多payload扩展仍明确 NotImplemented，不能理解为任意切分全部支持。
- **size1 shard mesh**：[#200986](https://github.com/pytorch/pytorch/pull/200986)，[main `1e233a2e93d21402e1be990dc1418dc4d306c07e`](https://github.com/pytorch/pytorch/commit/1e233a2e93d21402e1be990dc1418dc4d306c07e)。HSDP单rank shard或replicate过去只复制all-gather extension第一payload；现在逐payload复制并校验数量变化，恢复FP8等扩展的scale/附属数据。dtype变化及缩小payload仍有严格兼容边界，不是自动支持所有可变长度布局。
- **memory-efficient SDPA dropout backward**：[#201128](https://github.com/pytorch/pytorch/pull/201128)，[main `74bf05d58bae8c538dce58063e7fb81ca167e61e`](https://github.com/pytorch/pytorch/commit/74bf05d58bae8c538dce58063e7fb81ca167e61e)。captured/replayed forward的seed在device，eager backward原按“当前stream是否capture”决定host解引用，可能segfault；现在按seed实际设备选Philox状态，并验证seed/offset设备一致。作者GB300复现从exit139到0，新增及相关graph/dropout测试通过；ROCm源码路径未本地编译，不能冒充同等硬件验收。

已核 `_fsdp_collectives` / `_fsdp_param_group` 与 attention backward 的实际落地patch。**建议：考虑采用并对照自己的dtype/mesh/graph边界回归**；这几项解决触发条件明确的正确性/崩溃，不提供通用训练加速数字。

#### PyTorch precompile：把直接调用的 Triton JIT 缓存带到每个 replay rank

**10-16 main**，[#198955](https://github.com/pytorch/pytorch/pull/198955) C++ binaries（`20622c55f02cac33b97b340e6b371a97d3c44424`）、[#199153](https://github.com/pytorch/pytorch/pull/199153) 导出（`d45baa06bcfb21f0b40cebe859b4e7efca56d840`）、[#199154](https://github.com/pytorch/pytorch/pull/199154) 校验/导入（`6ba77d26ab608ca66f4b80d8fffdde27c5e77287`）、[#198789](https://github.com/pytorch/pytorch/pull/198789) capture/load接线（`d46faaf4f39c91cb76efb7cd813cf0ecd4dc3e27`）。

过去冻结Inductor图仍遗漏custom op直接调用的 `@triton.jit/autotune`，每个新rank首步重新编译/调优。现在capture独占临时cache并导出二进制及调优结果，load先校验SHA256、ABI/build identity再放入匹配/全新目录，且早于图artifact加载；并行导入用staging+rename避免半写入。已读真实main的runtime_cache、transport和scope接线patch。

**证据与限制。** 作者生产FSDP2推荐模型“整组补丁”16rank×100batch严格replay无编译，对照缺runtime-cache组全部拒绝服务；旧harness损失差异有混杂，不能归因于此单PR，更不能推成LLM质量提升。必须在capture块内首次运行所有所需kernel/autotuner；之前已JIT的、带pre_hook而不持久化调优的kernel可能仍遗漏。不同模型cache后来切目录也可能让未运行的旧kernel重新编译；不是任意硬件/版本的通用二进制包。

**建议：深入阅读、在自己的LLM冷启动路径复现。** 先用严格禁止编译模式验cache覆盖，再衡量部署首步和版本滚动；不要用是否存在cache文件代替完整性检查。

#### Transformers 5.20.1：Beam发布与加载回归修复；main另补专家分片遗漏

**正式版 [v5.20.1](https://github.com/huggingface/transformers/releases/tag/v5.20.1)**，10-16 13:02:29。除Beam入口外，修复torch2.7 `from_pretrained`被错误DCP guard卡住：`hf_storage`实际从2.8才有，将分布式checkpoint门槛升到2.8，**不是把全库基础torch最低版升到2.8**。[#49652](https://github.com/huggingface/transformers/pull/49652) main `f74cb53b9efd2d9155e5599ba7a7e5c1827e4576`。同版[#49428](https://github.com/huggingface/transformers/pull/49428) `eb943c26b92634df05a98488780b0b6ddcc95bb7`让DeepGEMM MegaMoE接管 `ep_dispatch_experts`，由kernel内部dispatch并要求EP=TP，避免双重dispatch/缺process group；[#49557](https://github.com/huggingface/transformers/pull/49557) `742d847414ca98b82b5e8eb0079ffdcc3feae69b`按图像拆分HunYuan-VL features，防止batch beam拿错图。均已核实际patch；不是新增模型质量提升。

**另有main变更 [#49206](https://github.com/huggingface/transformers/pull/49206)**，10-16 22:25:03合入、`8498ed425fa01bbee7d086efdf1a6634c2962aed`。若TP/EP plan没覆盖experts，输出仍可能正确但每rank保留全部专家；现在对多个MoE补plan及覆盖检查。部分新TP plan只分experts、attention仍复制；Granite/Aria/Jamba router返回契约改变，外部直接调用方需核对。作者2GPU FP32误差≤2e−6，未给端到端显存表，不能凭“TP auto”假定所有参数均分。它是后续main独立增量，采用时锁commit核计划。

**建议：考虑采用5.20.1的相关修复，并审查自定义parallel plan。** TRL [#7636](https://github.com/huggingface/trl/pull/7636) 同日main `41f8d4e88f15f5839105ba55b3722790ef51593b`补“合并LoRA后routed expert确有变化”的测试，揭示只检查adapter变化不够；它本身只改测试，实际PEFT修复在上游#3946，不应把TRL新测例写成运行时修复。

#### TensorRT-LLM DSA：限制跨请求 packed logits，避免 prefix reuse 放大暂存

**10-16 main [#20164](https://github.com/NVIDIA/TensorRT-LLM/pull/20164)**，`dac84b01a6c82059f189e2d7d4fb0fe4fbe4d62d`。MLA chunked prefill原把不同请求拼为 `sum(q)×sum(cached+new compressedKV)`，跨请求块最终虽被mask却已分配；prefix命中越多反而可触发巨大logits/OOM。按连续请求的乘积预算分组，**代码默认 `1<<28` FP32元素=1GiB**，再受单请求tile预算约束；过大单请求仍走原query tiling，不能解释为任何单调用峰值绝对≤1GiB。context tokens≥`q_split_threshold`（默认8192）时先划分TP query份额再组chunk、最后一次all-gather top-k，避免小group丢失TP并行。请求准入/调度本身未改。已读indexer与metadata实际patch；自动摘要曾写1<<26，正文/代码一致为1<<28，以代码为准。

**证据。** GB200仿真单TEP8 rank，FP8 indexK、64head、topk512、满q8192/936K KV复用batch：3792→525MiB、.78→.51ms；低于split阈值的原bugbatch q4290/KV500K：8266→981MiB，但1.00→1.10ms，并增host chunk准备。8×B200/DeepSeekV4Flash、64个约64.5K prefix族、729个OSL1请求、4.05req/s，早期256MiB cap版本从248/729后OOM到两次729/729；**不等于当前1GiB+TPsplit最终版完整E2E重跑**。先前#19959预留也可能已遮住原OOM，作者未重核该对照；rebalance约10s host re-pin暂停仍是独立未修问题。

**建议：考虑复现。** 对prefix密集、短增量prefill服务，记真实logits峰值、host耗时与TP分摊，不能只看命中率。同日[#19077](https://github.com/NVIDIA/TensorRT-LLM/pull/19077)新增KVPoolV2 rebalance可调阈值（2000样本、120s cooldown等默认不变），仍为prototype且需opt-in；它不解决上述host暂停。

#### TensorRT-LLM MiniMax-M3：unquantized KV 的 dense 层直接读页

**10-16 main [#20159](https://github.com/NVIDIA/TensorRT-LLM/pull/20159)**，`4271011718a350bbf40f296127879d3713502936`。由 gather全部KV、展开GQA后的SDPA，切到 native paged `TrtllmAttention`；主KV用HND并向稀疏层提供零拷贝NHD view。只在支持native且KV未量化时生效；FP8/NVFP4 KV保留旧路径，**NVFP4权重+BF16KV则可进入新路径**。P/D worker必须匹配构建，因为Triton unquantized KV布局改变。已读模型选路、cache views与fallback patch。

作者GPU实验明确为测试revision `566e0303f70e6d53416150fb8f94eed9e96ced46` 对main `b22283a1b69ddc5ed54e900513f1d4b7be08e472`，与最终head `7065e350e5cd0c11895af6ad7fed1ded1285521d`有后续cache-view/NVFP4处理等差异，未见最终head完整GPU重跑证据。2×B200/26.09容器，batch32/KV32768的单dense decode op额外峰值4.50GiB→80KiB，**另有预分配48.1MiB共享workspace，不含KV池/模型总显存**；7.51→.095ms也是operator口径。TP2真模型72 dense对比及1368 sparse检查通过；114题MMLU只是回归样本。端到端固定32输出长/偏斜prompt有降时观察但噪声较大，反向顺序短请求40次差−.14%、95%CI跨零，作者未建立统计显著E2E提速。

**建议：考虑采用省暂存路径、复现真实服务负载。** 不能把约79倍单算子耗时比值写成整模型吞吐倍率；量化KV OOM未在此PR解决。

#### TensorRT-LLM：投机 acceptance RNG 也要遵守每请求 seed

**10-16 main [#20147](https://github.com/NVIDIA/TensorRT-LLM/pull/20147)**，`7fafb022ee281f5b3a8f3a8ded30cfe0a8d5fd72`。旧one-model draft/target采样已用每请求seed，但linear acceptance及dynamic-tree rejection只读seed[0]/offset[0]、用batch行号作subsequence；请求结果依赖前一个请求和排列。新增chain sampler并对两类acceptance统一每row RNG契约，dynamic target采样接入AdvancedSamplingMode。已读C++采样接口与Python分派变更。

Qwen3-1.7B+EAGLE3、draft3、ISL/OSL128/256、AL约1.5、batch1–64、3对交替中位：H100/B200 linear outputTPS约+.25–1.61%，主要意义是可复现性。dynamic-tree作者给+11.2%/+17.6%，但**B200 CUDAgraph原本就有mask buffer越界**，双方都用了本地扩大buffer的workaround，修复不在本PR，不能当原版可直接运行结果。GSM8K1319题、Llama3.1-8B+EAGLE3、t.6/p.95、3seed得分相近，不能据采样噪声判断能力回退。

**建议：优先检查seed与batch重排不变性**；有B200 dynamic-tree需求继续追独立mask修复。

#### TensorRT-LLM CUTLASS MoE：回收burst workspace，保留warmup下限

**10-16 main [#19981](https://github.com/NVIDIA/TensorRT-LLM/pull/19981)**，`a0835b3d4fb91217317baa20beb499f710fe245d`。每model-forward跟踪owner/device/stream下的eager runner，连续3次需求≤保留容量一半才缩，缩到warmup最大值与窗口需求的较大值；growth、失败和低收益会重置。先分配替代块再释放旧块，保守退出重叠/未scoped等情形，原 `TRTLLM_RECLAIM_WORKSPACE=0`同时关闭attention/MoE回收。已读native scope、workspace策略与DecoderRunner接线。

**限制比百分比更重要。** 历史B300、BF16、Moonlight16B-A3B/FP8KV、TP1/eager、128in/8out及burst负载的E2E，普通最大shape warmup已覆盖需求，**实际回收事件为0**；观测约.1–.2%开销不能证明真实回收收益。operator fixture能回收2.284/2.967MiB active allocation，但reserved/NVML峰值没变，也不等于腾出KV容量。完整GPU数据来自rebase前revision，迁移到DecoderRunner后仅CPU source-isolated检查，不能称最终head完整GPU认证。

**建议：继续观察/定向复现。** 先证明线上确有超过warmup floor的增长和回收，再看allocator、尾延时和容量；不要把旧228MiB宣传数带到本版。

#### vLLM-Ascend：main再次默认MRV2，C8 MXFP8缓存补全静态V scale生命周期

**main [#17681](https://github.com/vllm-project/vllm-ascend/pull/17681)**，10-16 15:39，`f978348e9045fb045a4281d189339ef2ca1f1a45`。未设 `VLLM_USE_V2_MODEL_RUNNER` 时由白名单转黑名单，默认选V2；310P、encoder-decoder/Whisper、poolingKV、DSAcp、VL encoder disagg、`cudagraph_mm_encoder`、draft-window、suffix/draft-model/ngram/p-eagle及dflash2+graph等仍回退V1，显式0/1优先。**这与已读v0.30.0rc1 release线曾回退默认V2并不矛盾，分支状态不同。** 不能把新main称为所有配置一律V2或已在所有release默认。

同日[#18485](https://github.com/vllm-project/vllm-ascend/pull/18485)，`fe686bc79a81bcc2cbcff5b1717b07ed67a66404`，为V2重写MXFP8 FullAttentionSpec、组合raw allocation视图、初始化/页重分配恢复static V scales，并让block zeroer保留第四分量scale。该补丁解决低比特KV的布局和回收契约，不是提出新量化算法；作者主要新增UT，未给整模型质量或显存/吞吐表。

同日[#17334](https://github.com/vllm-project/vllm-ascend/pull/17334)，main `b6006612602f50f9219cb0aad53e067049d033ea`，latent MoE的MegaMoE约束改按 `routed_expert_hidden_size` 检查，避免把residual宽度当投影后的算子宽度。synthetic8193/3584回归与349项CPU/mock测试支持该配置判断，但测试在容器匹配旧基线上，未证明最终main/NPU质量与性能。

**建议：升级main时显式记录runner选择和cache dtype**，测试回收/重分配后的输出；不要只验证初次prefill。已读mrv2选择与attn_utils/model_runner/zeroer patch。

#### vLLM-Ascend KV Pool：global layer窗口与prefix命中时的全范围load

**10-16 main**，[#18647](https://github.com/vllm-project/vllm-ascend/pull/18647) `4f00748442020c0b59cd69450e0a3359a8c53fd1`：KVPP每rank只拥有连续layer shard后，旧“成功提交P个本地层”会在开头把晚shard全部预取，随后实际仅提前1层；改为global model-layer窗口，使load(L)在max(0,L−P)提交，并修group-local层号映射。作者边界UT不等于NPU全服务验证，未报E2E倍率。

[#18987](https://github.com/vllm-project/vllm-ascend/pull/18987) `2ab4f449d980d17f8a86b468075a7d12cd06ec04`：layerwise offload复用少量buffer时，device prefix命中也必须从pool重载；旧30s负缓存/短路可能跳过必需load，且独立层与shared层不能共用load range metadata。现对offload关闭该短路并分开metadata，普通layerwise pool保持原逻辑。已读scheduler/worker patch与新增边界测试。

同日性能补丁[#18907](https://github.com/vllm-project/vllm-ascend/pull/18907) `0720f2ee2cc3cada3d00badae9e33122da13c04b`减少HBM-prefix集合、过滤chunk和重复reachability mask。A3/32NPU、P DP4×TP4+D DP16×TP1、c49、196条固定chat、4次warmup、OSL1，**在已装v0.30上的兼容port** inputTPS208662→242531（+16.23%）；所谓“Prefill metric”为input tokens/所有TTFT之和，并非集群瞬时prefill TPS。各case只一次正式replay，P侧外部命中为0，且warm/cold输出一致性在基线及port中**都未解决**，没有full-vocab logits对照。

**建议：先采用/复现必要load与窗口正确性，性能结果继续观察。** 已合main、port实验及端到端质量验收是三种不同证据，不因表中吞吐接近HBM参考就认为pool路径可无条件上线。

#### vLLM-Ascend ECMooncake：NPU encoder输出直传，收益取决于E/PD配比

**10-16 main [#16124](https://github.com/vllm-project/vllm-ascend/pull/16124)**，`7373f6c3719c818bd53bd5a03fc37480e741faa9`。复用上游EC调度/目的预约/producer管理，NPU侧提供2MiB对齐注册池；staging放不下时，未对齐前缀经共享bounce arena，剩余直接注册原始storage，分wave批量写，并让tensor/registration/lease持有至传输完成。bounce arena是**额外注册内存**，不是已含在ec_buffer_size中。已读memory/transfer/worker及工厂接线；不把这一EC机制当作vLLM P2P KV旧WRITE寿命问题的修复。

作者“two910C”平台、Qwen3.6-35B-A3B-W8A8，750条MMIU多图、c96、8client、prefixcache off、graph on、3GiB E/consumer池：2E2PD 3.022req/s，DP4EP4 2.107，1E3PD仅1.692；2E2PD中位TTFT28.93s优于41.18s，但P95 TTFT62.85s劣于57.30s。三者750/750 HTTP200，parsed-choice准确率55.73/55.73/54.93%，不是独立通用多模态评估。E2E含一字节staging强制fallback与重复请求；作者用V2测试，但上面的新默认选择对VL encoder disagg仍fallbackV1，复现必须明确runner。

**建议：考虑复现多模态EPD分离**，同时扫E/PD比例和尾延时。另[#18715](https://github.com/vllm-project/vllm-ascend/pull/18715) `363af1c27954ff1f7ae5bd66eef06da62c2e3cd0`补packed SFA C8逐层字节传输与计数初始化；只支持匹配group/block-scale和连续main cache，TurboQuant及不匹配DCP仍不支持。作者18项是CPU/mock harness，未做NPU整链路。

#### Triton与FlashAttention：资源预算和warp barrier语义；hd256通用反向仍opt-in

- **Triton [#12326](https://github.com/triton-lang/triton/pull/12326)**，10-16 main `4a18983590bf26cf1f79e81d9b1ed471986ce9ef`：max_occupancy预算纳入CUDA每CTA driver shared-memory reservation及128B分配粒度，修复GB300要求15CTA却只能14的情形；kernel实际shared/register/硬件限制仍优先，不承诺任意kernel达到设定值。作者153项定向测试，已读driver属性及launch预算patch，未报LLM吞吐。
- **Triton [#12321](https://github.com/triton-lang/triton/pull/12321)**，10-16 main `2cd5519881ff9f92fa6f2a83db86b51eeea6292e`：重新落地warp-specialized分区的non-aligned named barrier，避免要求全CTA到达同一指令。此前曾因性能回退撤销；作者当前GB300微测及GB200 990匹配timing总体接近不变，上游A100/H100/GB200通过，不写成普遍性能提升。
- **FA4 [#2966](https://github.com/Dao-AILab/flash-attention/pull/2966)**，10-16 main `965c9212e6be33c821ee9f2b461564c5c91a2503`：统一hd256 backward第3/5步，须 `FLASH_ATTENTION_HD256_GENERIC_BWD=1`，默认仍专用kernel。合并dQ/dKdV将7GEMM变5，增加d256可用的softcap/sink/local/确定性等功能。B200锁1400MHz、32K总token/shape、交替两轮：MHA/varlen在长序列部分提升3–17%，2K GQA却慢19–29%；短GQA的后续#2967不能当本PR已具备。作者测试仍有34个非确定重复/dSink容差失败，不能从“5489 pass”略去；变更报告中parent CUBIN一致只对所述检查矩阵有效，不代表新d256路径未变化。

**建议：按kernel与shape深入阅读/选择性复现**。这三项都是可定位机制或边界，不能合并成“新版本训练统一提速”。

### 非默认分支补充

#### NVIDIA TensorRT-LLM feature分支：先拒绝padded draft token，避免xgrammar进程崩溃

稳定ID `github:NVIDIA/TensorRT-LLM:pr:20489`；[#20489](https://github.com/NVIDIA/TensorRT-LLM/pull/20489)，base **feat/musespark-open**，2026-10-16T18:31:52Z，merge `79e45c5df8788cc00f13da189434f0672c8a0bed`。

Muse Spark Open tokenizer201818词，但LM head202048行；draft未受grammar约束可能提出padding ID。旧xgrammar<0.2.6拒绝这些ID时还试图打印缺失decoded string，导致越界segfault；TRT-LLM本分支pin0.1.32。实际 `XGrammarMatcher.accept_token` 现在在token_id≥encoded_vocab长度时直接False，不调用xgrammar，draft验证在此停止而由target grammar继续，未改变合法token的grammar接受规则。

已读4文件完整guard与3个CPU回归：padding拒绝不前进matcher、合法ID仍走xgrammar；作者单4GB300/Muse+DFlash2/guidedJSON重放旧crash序列通过，未报加速。只合 **feat/musespark-open**，不是main或正式release。它是#20463后续崩溃修复，不能当第二次Muse模型首发。额外工件核对说明drafter新config多output_multiplier/final_logit_softcapping，而TRT当前DFlash不读；“两revision相同”不能扩张为与vLLM logit变换等价。建议**该feature分支部署优先纳入修复**并测试padding draft/rollback；保留门控模型访问与镜像基线。

#### NVIDIA TensorRT-LLM release/1.3：回移三项P/D服务发现修复

稳定ID `github:NVIDIA/TensorRT-LLM:pr:20396`；[#20396](https://github.com/NVIDIA/TensorRT-LLM/pull/20396)，base **release/1.3**，2026-10-16T10:11:07Z，merge `41c0ec856258a3aabb2faf7c25030df31577de1f`。

这是原main [#19427](https://github.com/NVIDIA/TensorRT-LLM/pull/19427)（原始 **2026-09-27T03:57:21Z**，`e0448239a57a70a7e357d1d0e11e2bebb6b37cbf`）的release/1.3 backport，不是新算法。当前API实际6文件patch包含行为变化：cluster storage以 `_service_now=monotonic−event_loop_unserviceable` 计TTL、0.1s探针结算stall，避免自己的event-loop阻塞让排队refresh被当失联；worker heartbeat timeout在保守本地expiry后继续一个inactive_timeout grace，明确未刷新或grace仍静默才放弃；server_info通过off-loop RPC优先取runtime tokensPerBlock，避免FlashMLA实际64而发布旧32使路由hash全miss，RPC空结果/encode-only错误回退配置。

已读原main与release files并核对同6文件的patch完全相同。PR正文附带自动审阅称“仅格式/没有cluster_storage diff”，与此次API真实patch冲突，不能采纳那段bot总结。release作者只跑py_compile/precommit，明确本地未跑3个pytest suite；main原PR的formal verify也不能替代当前release验收。测试新增loop outage后活worker存活、真正不刷新者到期及runtime block size回退，未披露服务吞吐A/B。建议**release用户检查服务发现/条件disagg correctness**，不把TTL补偿当无限容错或速度提升。

#### vLLM Ascend RFC分支：层级KV offload约束固定DSA形状，暂无实测收益

稳定ID `github:vllm-project/vllm-ascend:pr:18999`；[#18999](https://github.com/vllm-project/vllm-ascend/pull/18999)，base **rfc/vllm_cann**，2026-10-16T09:33:03Z，merge `aefc78918df625afbfa6f6f9ab00e4dd6f3b54a7`。

实际base是 **rfc/vllm_cann**，并非release，更不是main。83文件大改新增Host device-visible DDR存完整MainKV，HBM用层级selection/workspace；producer跨层4arena、2layer滚动prefetch，arena复用需source Future全完成且compute_release_event完成，两条边界不等同kv_ready。关键 `LayerWorkspaceFence.wait_workspace_reusable` 即使source失败也先同步compute，再抛原错误；timeout也不会成功返回“可复用”。request_generation/chunk_id与workspace_generation防身份混淆。

已读能力默认False、workspace fence源码/对应FakeEvent单测及platform启动guard。显式 `layerwise_host_kv_offload` 默认关闭；只PD producer/consumer、block128、MTP禁用或3token，index_topk若声明必须2048；producer需MultiConnector恰一AscendStore(range+layerwise)并包含MooncakeLayerwise。GLM5.2固定首版形状不能写通用所有DSA/模型全支持。PR测试栏为空，仅勾选checklist无数值/正确性运行记录，部分巨型store files未返回patch，未全审lease/drain/cancellation，不关闭跨进程传输寿命风险。建议**继续观察RFC/只做受控试验**，先验证DDR可见性、传输失败、arena再用与真实长上下文收益；Omni-EPLB也属独立opt-in配套，不与offload收益相混。

#### vLLM Ascend 0.30 release回滚A3 SiTU融合，性能恢复尚未测

稳定ID `github:vllm-project/vllm-ascend:pr:18927`；[#18927](https://github.com/vllm-project/vllm-ascend/pull/18927)，base **releases/v0.30.0**，2026-10-16T09:24:32Z，merge `63b471bfd34af81085ba4bb04b91df3b25f36d7f`。

因原[#16655](https://github.com/vllm-project/vllm-ascend/pull/16655)（main **10-06T14:32:44Z**，`9f0f60de677972fba9de69996b7699c98a521f81`）被报告性能回退，releases/v0.30.0移除A3 W4A8 `gmm_dequant_situ_quant` ACLNN实现/构建、Torch/Meta注册、capability与自动dispatch，恢复 `npu_grouped_matmul` 后 `dequant_situ_quant`。旧融合对eligible输入自动开、无opt-out，故这是真执行路径和API撤回；GMM2与另一个grouped_matmul_situ_quant不变。必须重建extension/custom op，单改Python不能当完成回滚。

已读W4A8删除dispatch、hardware/build与注册patch。正文称backport相关main rollback [#18902](https://github.com/vllm-project/vllm-ascend/pull/18902)，但本轮metadata #18902 **仍open/merged_at=null**；应写release已回滚、main方案仍审阅，不能笼统写main同步回滚。作者仅删除范围/AST/lint等静态核，pytest未安装无法启动、缺CANN/NPU，完整lint未完成，原始回退负载与倍率尚未提供。merged API胜过正文残留“Draft pending”，但合入不证明性能恢复。建议**release A3用户优先核路径与重建验收**，不发布恢复倍率或归因某kernel瓶颈。

#### vLLM Ascend 0.30：GLM V1 PP切面恢复full-token布局

稳定ID `github:vllm-project/vllm-ascend:pr:18870`；[#18870](https://github.com/vllm-project/vllm-ascend/pull/18870)，base **releases/v0.30.0**，2026-10-16T15:32:23Z，merge `24e7054878b5d01c2923ee409f89b72f0e355b3c`。

release-only修复GLM DSA V1在PP>1时虽报FlashComm1开启，但upstream PP1 guard让decoder SP未启：局部MoE+MLA layer启SP并关attention o_proj预归约，使decoder接管reduce-scatter；把hidden+residual allgather提前到PP return前，随后裁TP padding，runner receive/dummy/profile也保持full-token长度。无新flag，限global SP且dense prefix后连续MoE（moe_layer_freq1）；MRV2、非GLM、交错dense/MoE保持原逻辑。未增加任意SP→dense转换、nullable residual或empty stage兼容。

已读5文件关键代码及selection、奇数3/偶数4token、PP首中末stage回归。作者Ruff/diff通过，tensor tests、NPU correctness/perf仍pending，先前CI在证书/apt key安装阶段失败，不可写测试已通过。所引旧main[#16819](https://github.com/vllm-project/vllm-ascend/pull/16819)是09-19 DSV4 PP非首rank input_ids=None的sp_shard guard，并非本GLM补丁已在main的证明。建议**限定0.30 V1 GLM PP采用并做端到端回归**，暂无性能增益测量。

<a name="hardware"></a>

## 硬件与软件协同

### Beam多轮服务：缓存路由与分层KV的系统配方，先用已修启动兼容的版本

**日期与来源。** [Google Cloud官方文章](https://cloud.google.com/blog/topics/ai-infrastructure/day-0-support-with-5x-perf-for-reflection-beam-on-google-cloud)搜索快照标10-16，本轮实际页面及JSON-LD为 **10-17**，`published_time=2026-10-17T08:00:03+0800`，落在本轮截止前；按原页时间记账，不能用搜索摘要日期代替。代码基线 **AI-Hypercomputer/gpu-recipes@8f6846537d1462ff139b5204dd1de9f9e997e5a0**，对应10-16已合入main的[#287](https://github.com/AI-Hypercomputer/gpu-recipes/pull/287)。

**技术增量。** llm-d/vLLM或Dynamo/SGLang把prefix与负载路由、turn-priority flow control、P/D分离、DFlash2及HBM→DRAM/NVMe分层KV组合到多轮AgentX服务，解决后续turn换replica重算历史及HBM溢出后cache thrash。文章的Dynamo在32×GB300/A4X Max，llm-d在8×RTX PRO6000/G4，**两图不能横比框架或硬件优劣**；A4X为最高5倍吞吐/8倍低延迟，G4为最高2倍吞吐/8倍低延迟，均为customer benchmark相对各自基线的最大值，不是单项改动收益。正文未完整给出每个曲线点的长度/缓存分布/运行次数与置信区间，本轮不据此做容量或成本结论。

更可立即采用的证据是#287：首版recipe在当前 `lmsysorg/sglang:beam` 会因Dynamo仍调用已删除的 `ServerArgs.get_model_config()` 启动失败。补丁按方法存在性接入 `model_config_of`，将镜像锁到digest `0ea912aadc3d7827858de9aa5a312c4d51c254e7dbe670dbbe8181895b92071d`，为所有worker加nodeSelector确保同一NVLink domain，传正确namespace并避免重复装cluster-scoped CRD。已读两个recipe、Helm与启动patch。作者8个4GPU GB300节点、operator1.4.2，聚合与4P4D各跑c32、3600s，分别4788/4750请求、0error；**这是可运行性烟测，不是5倍/8倍对照复现**。

**建议：考虑复现系统配方。** 先锁模型、镜像digest、operator及GPU域，再分层测output TPS、TTFT/ITL、prefix hit和offload字节；不要从“day0”推断通用main镜像已兼容。AWS同日[Beam部署原文](https://aws.amazon.com/blogs/machine-learning/introducing-reflection-beam-an-open-weight-model-for-agentic-coding-on-aws/)也已开，主要补充JumpStart/HyperPod部署入口，本轮没有新的可比较训练或性能证据，合并为同一模型的多来源线索。

### AMD RDNA FP16 GEMM：逐步解释LDS与occupancy，优化方向不能跨架构照搬

**AMD，10-16 [官方教程](https://rocm.blogs.amd.com/software-tools-optimization/rdna-fp16-gemm/README.html)。** 八步从wave32 WMMA、复用tiling/宽load，到LDS布局、buffer descriptor越界归零、K深度、block重排、交错prefetch与重新选择单双buffer。RDNA3的operand/accumulator寄存器宽度与RDNA4不同，同一改动会改变各自驻留率；这是一套可追踪的kernel分析与实验，不是新LLM架构或生产GEMM库发布。

ROCm10.0、HIP `-O3 -std=c++17 -mcumode`，FP16输入/输出/**FP16 accumulation**，TN存储，10warmups+30次均值，对比hipBLASLt heuristic返回的方案。RX9070XT/gfx1201上8192³，通用最终tile8.71ms仍慢于8.05ms；进一步192×224、单buffer调优到7.73ms，wide FFN型8192×28672×8192则29.55→26.64ms，对照28.54ms。W7800与共享系统内存8060S结果不同，不能拿同一TFLOPS标尺解释内存瓶颈。

**限制/建议：深入阅读，复现自己的shape与数值。** 正确性用随机±1构造、整数可精确累加，不能证明一般FP16输入的误差；两个大矩阵也不覆盖decode skinny GEMM、MoE小M或端到端模型。可以借鉴LDS/寄存器/trace分析，不宜直接替换通用库。FP8移植是后续Part2，不写成此次已发布。

### NVIDIA RTX Spark：Windows on Arm CUDA生态扩展，公告性能仍缺完整负载协议

**10-16 [官方全文](https://developer.nvidia.com/blog/code-test-and-deploy-on-one-machine-with-nvidia-rtx-spark/)。** 本轮可核的新方向是GB10/最多128GB统一内存PC的Windows Arm软件生态：公告称首个PyTorch-CUDA build、Python3.15 native-default与更广的Arm包覆盖；Linux/WSL与Windows原生是不同运行环境，本文未给每条安装物的固定版本/完整验收矩阵，本轮也未安装。

其4.6倍coding/2.7倍LoRA/1.8倍编译分别以AA-AgentPerf-local+Qwen3.8-27B、Unsloth+Llama3.1-8B、Linux kernel build对照Core Ultra X9 388H，不是同模型训练/推理的统一倍率；batch、长度、精度和热状态等未完整披露，不作为选型结论。文中Bolt“will be able to run at launch”与decision模型“最多200倍”不能扩写为所有运行时已验证或通用agent加速。Bolt权重及旧训练证据已在前日报收录，不重算新模型。

辅助核查：[cuda.core1.2.2 release notes](https://nvidia.github.io/cuda-python/cuda-core/1.2.2/release/1.2.2-notes.html)实际新增Python3.15（含实验free-threaded）wheels，在Python3.15下，CUDA12需cuda.bindings≥12.9.10，CUDA13需≥13.4.4；notes标10-15、GitHub tag发布10-16，两日期分别保留。这只验证包兼容范围，不证明上面全部Windows Arm训练链路已经可用。**建议继续观察并做现有依赖的小规模兼容验证**，不因宣传倍率改变部署基线。

### 补录：Gaudi1.24.2版本说明访问恢复，需关注训练组件支持收缩

**版本1.24.2-481；bridge tag原发布10-07，本轮10-17补录。** 此前错误猜测的带空格/重复命名URL多次404；本轮从官方文档首页解析出 [GAUDI_Release_Notes.html](https://docs.habana.ai/en/latest/Release_Notes/GAUDI_Release_Notes.html)，已实际打开对应1.24.2正文。文档未独立标明每段更新日，不能把补查日期写成软件新发日期。

**实质内容。** 版本说明明确停止支持Slurm与Megatron-LM，并增加FP8 Dynamic等软件支持；vLLM Gaudi0.30基于上游0.30，列出GLM5.2 FP8 DSA、Nemotron-H量化Mamba/non-gated FP8 MoE、Qwen GDN prefix-cache/compact正确性与超长RoPE修复，decode warmup block bucket受物理KV容量约束。正文没有可直接比较的整模型性能表，插件逐PR和全组件commit尚未完成审计；“移除支持”也不等于既有环境立即不能运行。

**建议：现有Gaudi训练栈优先核升级兼容矩阵**，尤其仍依赖MCore/Slurm的任务；推理新模型先锁插件/driver/framework组合验证。此处仅关闭“版本说明不可读”缺口，不关闭Gaudi全软件栈覆盖。华为正确公告入口本轮恢复为 `/productbulletins`，CANN/HCCL本体的增量仍未扫完；TPU与Neuron release入口及组件release列表也不代表全部底层库已覆盖。

<a name="evaluation"></a>

## 评估与可靠性

### RLVR去污染：语义相似阈值可能大量误删，但88%只属于小样本

**独立研究者Esther Xin；2610.21229v1，2026-10-15T04:19:03Z；上轮缺口补录。** [论文](https://arxiv.org/html/2610.21229v1)；给[code/data](https://github.com/ethxin0011/rlvr_data_contamination)，本轮未代码核查。

§3–5/Tables1–3：16个RLVR math corpus共1,673,514 corpus-specific item（去跨库后1,197,691文本），六benchmark1949题；594pair双盲复核、按corpus/题分calibration与holdout，另80item全部83candidate edge复核。复用exact/lexical/Qwen3Embedding候选与judge，不是新检索器。cosine.80下sample removed59中52误删；整个16067条population删除并不能据此说88%全误删。judge policy heldout resolved precision98.1%、**candidate-frame** recall98.2%；exact observed precision100%但frame recall7%。

§7明确recall无法测候选外污染、同两reviewer且仅raw agreement、judge removal样本仅6个且1未决、数学英语限定；没有下游RL训练实验。**建议深读/实践**：去污染报告应同时给pair precision、完整item误删审计、候选范围召回与数据损失；不因为semantic更强就降低阈值全删。

### LURCH：长context评估还应报告证据范围、离散度和依赖深度

**University of Edinburgh；2610.22067v1，2026-10-15T15:28:33Z；上轮缺口补录。** [论文](https://arxiv.org/html/2610.22067v1)。未核专用实现。

§3–5把evidence token占比、首尾evidence间无关文本占全部无关文本比例、推理DAG最长依赖路径作为三个难度属性。6550题来自17个已有源，32k–256k token（Qwen3-.6B tokenizer），四模型GPT5.6Luna/Gemini3.8Flash/Qwen3.8-27B/Qwen3.5-4B每题3次响应。条件GLM控制source/model/长度后，scope与depth四分位对比约-2.14/-1.60pp，长度-1.66pp；scope对Gemini却+1.55pp，不能称每模型三属性均负。dispersion/depth某组合难点受SciTrek占62%影响，移除后差显著缩小。

**建议采用评估分层，观察因果**：evidence自动检索扩成段/整表、judge三次通过才保留，图由模型生成；是条件相关性不是控变量干预。可以设计相同evidence/推理仅改distractor分布的复现，别把长度统一后仍困难直接归因“推理能力不足”。

<a name="coverage"></a>

## 覆盖、失败、补录与更正

### 实际来源覆盖

先读取自动化memory、`.monitor/state.json`、本地历史日报与索引，读取GitHub最新索引/上期日报，并用远端树校验全部历史：**12篇日报及README共13个Git blob逐一一致**。起始远端main为 `0989b6bd7a8b01d4efc256298812f61cde5c4db0`，无遗留待上传。扫描窗口从历史明确截止点恢复，没有用mtime或调度启动时刻代替。

| 来源类别 | 本轮实际检查 | 完成边界 |
|---|---|---|
| 模型机构与工件 | 31个官方/扩展HF组织，modified前100与created前50双路62/62请求成功；17工件卡/config/history三路51请求、50成功；官方研究站与模型卡/安全报告 | capped列表不是全量分页；只有选中候选精读。GGUF无config404保留，未下载全权重、未执行remote code。Reflection加入后续清单。 |
| 核心六个开源项目 | 默认分支72h+主窗743 commits：Megatron62/vLLM310/verl19/SGLang287/slime5/TorchTitan60；主窗所有分支merged搜索164；66个PR正文、64个files列表；release各前20或不足20 | 指定清单分页取尽、search incomplete=false；关键patch选择性审阅，不等于所有大PR/依赖逐行审计。合入事件时间与commit自身日期分开。未把dev或专用镜像算main发行。 |
| 扩展九项目 | PyTorch339、DeepSpeed13、Transformers79、TRL68、OpenRLHF0、Triton35、FlashAttention5、TRT-LLM149、vLLM-Ascend130，共818 commits；主窗merged搜索148、release各前8；34个PR及files、7个PyTorch真实main提交核查 | 指定列表到不足100且merged搜索完整，但只详核有实质候选；PyTorch ghstack搜索0不能代替main提交。分支补查另核TRT/Ascend五项，见正文分支说明。零计数只限该查询与窗口。 |
| 论文发现与正文 | HF10-13至16共176唯一候选；arXiv cs.CL/LG/AI new/recent；new1187次出现/884唯一ID、319个Replacement；论文方向20篇正文形成19条，另Hy-Memory在模型章节详核；3仓库固定SHA静态审 | 不把HF精选日或new列表公告日当首发。MuonHTML404后官方TeX补读；无训练/算子复现。未精读全部标题候选，未审完所有修订。 |
| 正式学术来源 | TMLR当前与昨日保存列表对比新增11个stable forum ID；PMLR首页及v350/v348；ACL首页/EMNLP入口与OpenReview | TMLR只给月份，不能确定日级接收事件；Delta-Rule Writes等正文仍challenge/403。PMLR两卷是9-30旧发布，不包装成当天成果；ACL指定入口404，非全会议枚举。 |
| 硬件软件协同 | NVIDIA/AMD技术博客、Intel/Gaudi、Ascend公告、TPU/Neuron release notes、AWS技术入口；NCCL/cuda-python/Gaudi bridge/JAX/NeuronSDK/NxD/NxDI/NKI八仓release列表，另Google Beam recipe PR | CUDA/JAX初次connector传输失败后恢复；release列表不代表CUDA/ROCm/CANN/HCCL/XLA/libtpu/NKI所有组件commit。Gaudi正确正文访问恢复但插件逐PR审阅仍不全。 |

模型站点包含OpenAI、Anthropic、Google DeepMind、Meta、Microsoft、NVIDIA、DeepSeek、Qwen、Seed、Kimi、智谱、MiniMax、腾讯、百度、阶跃、小米、美团、Mistral、xAI和Ai2，HF还覆盖IBM/Cohere/LiquidAI等。中英文主题搜索用于补Reflection、远程KV、低精度与硬件等名单外线索。站点/搜索没返回候选，**不意味着该机构没有发布**；有技术实质才进入正文。Anthropic量子电路/ECDSA应用研究保留发现线索，经济/就业报告、纯部署可用性或营销倍率未扩成训练突破。

### 访问失败、语义异常与尚未完成区间

- **arXiv原缺口只部分恢复。** 请求10-15 01:55:47Z→10-17 00:56:24Z的lastUpdatedDate API虽200，返回Atom的title/link却改成 **submittedDate**；396/396只是这组新提交的索引返回。72h组1542/1542也只限首次提交在窗内的论文（其中49有更新版本）。不能据此关闭旧论文版本更新覆盖。精确主窗返回0也不证明没有新研究或修订。官方new的319个Replacement metadata分7批仍429/一次断连，候选、精确区间与失败均保留供下轮续查。
- **GitHub匿名API全面403限流**，本轮改认证connector恢复清单与选定证据。两次初始分页参数错误造成冗余页已识别并排除后补齐；扩展清单写入曾受shell单参数长度限制，最终分块重存且解析验证，不把失败输出算完整记录。SGLang大kernel一个files项无patch，保留未读全文边界。上述恢复不等于完整代码审计。
- **模型动态/门控渠道。** Meta gated raw/config仍403；MiniMax API陈旧与IR差异未解；Qwen/腾讯首页/百度/Step动态正文不全。Seed blog首20有has_more，paper API200却无实际条目，不能记录论文零新增。Kimi blog本轮真实列表可读，只关闭入口访问小项，未穷尽闭源产品更新。
- **模型待续。** NVIDIA retrieval卡仍写未来10-20，当前上传与正式ReleaseDate不一致；Super软件中训67B/报告65.4B差异仍在。Step NVFP4独立质量/输出限制、UserRepGen正式工件、GLM-Legal权重与Ai2 Bolmo原文等旧待办保持；Dory/Beam runtime PR未合入及完整训练报告继续跟踪，不把未来承诺记为能力。
- **旧工程审阅未完成。** 核心10-05→10-08剩余候选、PyTorch/Transformers/Triton10-05→10-12旧清单及既存Mooncake等候选继续保留。本轮新清单未选入项也仍是选择性审阅，不因列举成功关闭“内容已审完”的缺口。
- **硬件组件。** Gaudi先前notes404定位为路径错误，正确GAUDI_Release_Notes已读；华为正确入口`/productbulletins`也恢复。只关闭对应访问问题，CANN/HCCL/XLA/libtpu/NKI等组件本体仍有旧区间缺口。NVIDIA BlueField论坛10-16活动实际指向8-24原文，已过滤，不能当当天新硬件。
- **正式来源与修订待办。** TMLR新增11列表项、Un-OPD2610.05373v2的v1→v2实质diff、剩余replacement和标题候选待核。Un-OPD不是2610.21646；仅获得v2正文不把旧整套方法重新宣称首发。

### 补录与发布前校订

1. 本轮论文详报主要是10-15首发的上轮缺口补录；readout为10-14，KV4-RL为10-10，Muon/Continuum旧待审分别保留10-14日期。Hy-Memory官方10-16发布与论文10-15提交分开。GPT-Live1.1系统卡10-15补录、DSpine为9月论文的新10-16权重、Gaudi为10-07旧版本补录，均非10-17首次研究发布。
2. Beam仅报本次开放权重、配置/量化/安全新证据，万卡预训/RL旧披露不重复；Dory卡对last_loop质量只有定性“接近”，未给数值消融/等价保证。
3. 交叉审阅补足Ascend `cudagraph_mm_encoder`回退、TensorRT-LLM TP query split的8192阈值与MiniMax-M3测试revision；代码里的packed budget为1<<28，自动摘要的1<<26不采用。最终head与旧GPU验证、兼容port、CPU/mock harness分别限定。
4. Readout的WordNet probe针对readout rows，与hidden可用性的SVA/IOI证据分开；OLMo2有早峰但统计未显著，不写成完全没有。STOPD的math/code混合avg@k与IF prompt-strict口径补全。所有小样本/单次实验避免推成普遍能力结论。
5. Google Cloud Beam文章按实际原页10-17北京时间08:00记账，搜索10-16快照不作定论；A4X的5倍/G4的2倍与各8倍低延迟分开，recipe的0error烟测不证明这些倍率。cuda.core notes日与tag发布日分别保留。

本轮日报及索引仅保存/同步到指定 `llm-monitor` 和 `tracker/main/llm-training-daily`。逐来源请求、固定版本、正文与同步回执分别保存在本地 `.monitor/research-2026-10-17/`、`.monitor/state.json` 和运行/归档回执；本地生成与远端确认分开登记。**全局完整覆盖水位仍未建立**，没有声称全面监控，也没有修改llm-knowledge的wiki。
