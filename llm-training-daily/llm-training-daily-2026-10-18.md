# LLM 技术动态日报 · 2026-10-18

> 时区：Asia/Shanghai（UTC+8）。增量扫描窗口：**2026-10-17 08:56:24 → 2026-10-18 09:05:14**；72小时回溯从 **2026-10-14 08:56:24** 开始。另续查此前 **10-15 09:55:47 → 10-16 09:00:40** 的论文失败区间及持久化旧待办。事件游标与各源实际覆盖水位分别记录，不能据此声称全面覆盖。
>
> 实际生成时间：**2026-10-18T09:19:05+08:00（Asia/Shanghai）**。截止后的读取用于核实截止前事件或访问状态；未将新合入倒灌到主窗。除特别注明北京时间外，工程精确时间为UTC；论文提交/修订、厂商公告、权重上传和本轮发现分开标注。
>
> 证据口径：实际打开的一手论文正文、官方卡/配置/日志、release与PR/commit。测试、性能与质量分数均为作者或厂商披露，本轮只做静态核查，没有独立GPU/NPU/TPU训练或serving复现。未合入提案、dev/专用分支、main与正式发行分别标注。

## 本次重要变化

- **DeepSeek-V4.1-Flash-1017是可核的新权重更新。** 48份权重分片hash均变，已查配置/参考推理代码沿用旧版；self-feedback RL只有公告级披露。新旧Terminal-Bench表的harness口径不同，质量分差不能全归因于权重。
- **训练与热更新先看语义正确性。** Megatron补GLU checkpoint/master布局及有效token归一；TorchTitan补YaRN scale与EMA恢复；vLLM逐参数M2N布局和权重版本span为异步RL提供更清晰的接点，但失败不等于原子回滚。
- **KV和专家scale修复值得优先核查。** SGLang MegaMoE EPLB漏搬派生scale、HiCache跨stream顺序、Ascend C8定点写入与跨rank快照发布，都有具体触发条件；已有P2P WRITE寿命问题只推进到open隔离方案，尚未关闭。
- **默认优化也可能改变数值语义。** SGLang的DSv4.1 decoder replay进入兼容CUDA/HIP standalone默认路径，同窗又补prompt-logprob全extend回退。RL训练对齐仍应保留full-prefill对照，不能当普适等价加速。
- **补录的研究更适合按瓶颈复现。** MirrorMoE针对共享专家下的路由探索；二比特n-gram表、DropKV区分容量与算子提速；Un-OPD v2的真实增量是实验配置/边界分析，四张主表在v1已有。
- **硬件结果必须对齐分母。** Meta/AMD的生产主表是DLRM、相同卡数/batch且功率不同；Apple的816B本地结果是batch1、多Mac的PP/TP比较。两者均提供实践线索，不能泛化为所有LLM训练或成本优势。

## 目录

- [模型与技术报告](#models)
- [预训练与架构](#pretraining)
- [后训练与论文修订](#posttraining)
- [推理与内存优化研究](#inference)
- [工业开源项目](#engineering)：[Megatron](#megatron)、[vLLM](#vllm)、[verl](#verl)、[slime](#slime)、[SGLang](#sglang)、[TorchTitan](#torchtitan)、[旧待办与提案](#backfills)、[扩展项目](#extended)
- [硬件与软件协同](#hardware)
- [评估与可靠性](#evaluation)
- [覆盖、失败、补录与更正](#coverage)

<a name="models"></a>

## 模型与技术报告

### DeepSeek V4.1 Flash 1017：新权重兑现质量—输出效率优化，架构与参考实现沿用9月版

**机构/日期/版本。** DeepSeek [10-17官方公告](https://api-docs.deepseek.com/news/news261017/)和[更新日志](https://api-docs.deepseek.com/updates/#deepseek-v41-flash-1017-update)确认Web/App/API同步升级；API仍用`deepseek-flash`、调用方式与价格不变。独立MIT权重仓[固定卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/README.md)为 `1bae0e1088993b9b82de08f59de5be5a34e42170`；上传权重10-17T03:35:48Z、release commit 05:46:55Z，均在主窗。

**真实增量与机制。** 这是已有图文模型V4.1-Flash的小版本权重更新。新旧48个safetensors分片的LFS SHA256全部不同，排除单纯改卡；但与旧版 `314113f8a7dadb388215396ba5fbee2c610a04d6` 比对，**config、tensor索引、inference/model.py、encoding.py及encoding说明逐字一致**。因此保留原40层、1M context、Engram/压缩KV/DSpark等配置，不能把这些旧机制重新算作1017的新架构。公告只披露以 **RL with self-feedback** 改善从零完成项目，未给反馈器、奖励、loss、训练数据/步数/算力或控制消融；目前无法说明是哪项训练改动造成多少增益。

**新卡并列自报及口径限制。** 表列旧/新：Terminal-Bench4.0 **31.2/42.3**，ProgramBench **20.3/29.1**，DeepSWEv1.1 **74.2/77.0**，HLE无工具 **36.8/42.6**；视觉工具任务Chartography **78.9/83.0**、BabyVision **89.6/92.5**。新卡TB4.0用Claude Code、DeepSWE用mini-swe-v2，而旧固定卡将TB2.1/3.0/4.0归于DeepSeek Harness Minimal；新表沿列旧31.2，**未证明旧权重已在新harness重跑，故不是已证实的纯权重受控A/B**。HLE括号44.0仅纯文本子集。ZeroBench-main旧49.0在旧卡标Pass@5，新卡51.0未重述，但**10-17官方更新日志的新1017节明确其评估为Pass@5**；不能当Pass@1，也未证明其余服务/采样条件完全相同。作者另称输出更短、完成更快，并给DeepSWE的质量—每任务费用图，但卡/公告未列完整数值、服务硬件、并发、所有harness版本、seed/CI；不能推成普遍吞吐或固定成本倍率。新卡仍链接旧仓的V4.1报告，并无1017训练细节报告的独立版本。

**实践建议。** 优先做旧/新固定权重的同harness A/B，分别记录成功率、生成token、端到端时间和工具费；API别名已自动切版，应重新验收。DeepSeek Harness建议≥0.2.1-rc.1；新仓移除Jinja template，提供Python encoding和另发`deepseek-recipe`协议工具包，后者仍把模型推理、工具执行、HTTP交给调用方，不能当完整serving框架。

稳定ID：`deepseek:v4.1-flash-1017:release:2026-10-17`；工件ID：`hf:deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170`。

### 72h回溯补漏：Jais 3把阿语适配和循环抑制拆为可对照阶段，但完整报告/训练代码仍待发布

**机构/日期。** IFM（MBZUAI）/Inception42、与Cerebras合作，[10-15官方技术博客](https://ifm.ai/blog/jais3/)正式介绍0.9B、3.7B、7B、32B、36B-A4B、375B-A23B六款Apache2模型，本轮10-18首次补录。HF仓/上传历史可早至9-29，不能据此断言当时已对外公开，亦不能把10-16署名修改当新训练。已逐一打开六仓当前卡/config/history；7B固定为[11dcac6d60aaf87e427c33d070dd6cf412f7270a](https://huggingface.co/inception42/Jais-3-7B/blob/11dcac6d60aaf87e427c33d070dd6cf412f7270a/README.md)，375B为[1b5d78983cb36203253256f5085f3a3d435060a2](https://huggingface.co/inception42/Jais-3-375B-A23B/blob/1b5d78983cb36203253256f5085f3a3d435060a2/README.md)。

**为何这套分阶段设计。** 从同尺寸已post-train的K2 Horizon继续适配，CPT以阿语/原模型replay **75/25** 平衡新语言学习与旧能力保留，再cool-down、SFT、FTPO、可验证RL。FTPO只优化重复循环开始的token，避免对整条回答做DPO；六模型平均loop rate自报 **13.8%→1.2%**。**Agent轨迹来自SFT，RL所列任务不含agent**，不能把所有agent增益归给GRPO。以上为博客披露，尚未代码审计。

**工件核对。** 四个dense；36B-A4B是MoE+MoVA **attention**，并非多模态视觉版。7B配置36层/hidden4096/32Q-8KV；375B为61层/hidden6144/192expert top8+1shared，词表250624。0.9B配置131072 context，其余524288。7B与375B的refs实际返回`CPT/Cooldown/SFT/FTPO/main`分支及各自SHA，支持分阶段复现入口；没有逐一下载这些分支的权重或核完训练日志。

**证据边界。** 官方清洗过阿语题并采用generation scoring，不能与旧的log-prob榜单直接相减。7B卡在high effort下TB2.1 **46.4（Avg@3）**、SWE Verified **70.2（offline Pass@1）**；同卡Qwen3.5-9B分别29.2/50.8，仅表示作者当前harness内比较。375B卡Apex-Agents仅420个text-only loop-agent子集，WildClaw仅English text-only；不把这些写成全量原生阿语agent测量。0.9B不是512K，SFT长文训练披露到128K也不等于已独立验证512K质量。博客强调开放代码，但**六张当前卡仍将Technical report和Code repository列为Coming soon**；不能称完整训练代码/正式报告已齐。建议深入读阶段差分与FTPO消融，优先7B小规模复现，保留清洗集和原始集两套评估。

稳定ID：`ifm:jais-3:official-announcement:2026-10-15`；六仓SHA见`models-supplement-versions.json`。本条是72h回溯公告/机制补漏，非10-17新架构。

### Hy-Memory新增FP8工件：静态激活/KV量化需要单独验收，旧BF16分数不能代言

腾讯[FP8固定卡](https://huggingface.co/tencent/Hy-Memory-30B-A3B-FP8/blob/40d31ceadee7318e0aa35443efe2d38487b83644/README.md) `40d31ceadee7318e0aa35443efe2d38487b83644` 在 **10-18T00:05:50/55Z上传文件、00:06:26Z加卡/Apache2许可**，属主窗；10-16T07:57Z只是initial commit。它是昨天已报Hy-Memory writer的FP8部署版本，不重复OPD/归因RL成果。

[config](https://huggingface.co/tencent/Hy-Memory-30B-A3B-FP8/blob/40d31ceadee7318e0aa35443efe2d38487b83644/config.json)保留48层、128expert top8+shared、262144ctx，增加`quant_method=fp8`、static activation/static KV；K/V为per-tensor静态FP8 E4M3，Q为per-token-per-head动态FP8，embedding/lm_head豁免。卡称vLLM v0.31.0测试部署，但**表头仍是原Hy-Memory-30B-A3B，未另给FP8-vs-BF16质量、校准集、显存或时延对照**，不能把55.49均分写为FP8复测成绩或无损证明。建议只作可部署工件短讯；在同一memory bank/reader/retrieval下补量化质量、bank增长与长上下文漂移测试。

稳定ID：`hf:tencent/Hy-Memory-30B-A3B-FP8@40d31ceadee7318e0aa35443efe2d38487b83644`。

### 72h短讯：Liquid d1-ultra把多模态决策直接变成概率，复现细节不足

Liquid AI [10-15公告](https://www.liquid.ai/blog/d1-ultra)的d1-ultra输入文本/图像和问题，一次前向返回yes/no、choice或score概率，不生成回答token；用更大的未命名开放基座加定制训练，特别优化概率校准。与10-05 d1属于同系列的新API模型，不能与开放的`d1-omni-600M`或其GGUF当同一权重。

该公告比较112项评估并以图给准确率/校准，但未披露基座身份、参数、完整训练损失/数据、逐任务表、评估样本量或置信区间；本轮未取得可下载的d1-ultra权重，不按可复现训练突破写长条。API按输入token计费且每个问题各自计整份输入，单次forward不表示多问题免费或固定服务时延。建议有概率决策需求再深入：先量Brier/ECE、分布外校准和多问题成本，勿只看accuracy或“SOTA”标题。

稳定ID：`liquid:d1-ultra:announcement:2026-10-15`；发现日10-18。

### 旧待办一手补证：Ling-3.1-flash的1M宣传、256K发布配置与服务窗口要分开

10-08旧日报仅列二手待查，本轮打开InclusionAI[当前固定卡](https://huggingface.co/inclusionAI/Ling-3.1-flash/blob/a4ccd6b45c002ddb236f9cb77d52cc427482bf19/README.md)、config/history及[官方文档](https://developer.ant-ling.com/en/docs/models/ling)（页标10-16更新）。当前SHA `a4ccd6b45c002ddb236f9cb77d52cc427482bf19`是10-15 README commit，权重上传10-13；**不把它写成10-17新发布**。

MIT、560B总/25B激活，72层按卡为63 KDA+9 Gated MLA，512expert top8+1shared，hidden4096，配置另带1个MTP层；是text模型，不能把同页介绍的Ling-3.0-flash-VL移植为3.1的模态能力。卡称1M，但发布config `max_position_embeddings=262144`，官方文档明确“上限1M、当前服务256K”。卡的长周期代码成绩结合未公开LingCode（Pi改造harness），因此不把成绩直接当模型脱离harness的收益。SGLang cookbook入口已读，未逐项复现其多硬件/压缩推理表；训练内容摘要链接也不等于完整技术报告。建议旧待办改为“一手配置已补齐；1M配置/实测、训练细节、LingCode开放与压缩质量待核”。

稳定ID：`hf:inclusionAI/Ling-3.1-flash@a4ccd6b45c002ddb236f9cb77d52cc427482bf19`（旧工作补证）。

<a name="pretraining"></a>

## 预训练与架构

### MirrorMoE：共享专家压低路由差异时，先缩小有效探索空间再逐步放开

**ETH Zurich / HKU / MPI Intelligent Systems / WU Vienna；2610.22353v1，2026-10-15T17:58:02Z；旧缺口补录。** [论文](https://arxiv.org/html/2610.22353v1)；[代码基线b63acf2](https://github.com/BingcongLi/MirrorMoE/tree/b63acf2e81066d02d9636bb8dc862779341d8676)。

§2的控制干预显示，共享专家主导表示时，替换route对层输出影响较小，细分专家带来的组合空间却更大。§3用正logit与负logit的softmax混合来**选择**top-k，真正输出仍按正logitsoftmax加权：低logit专家可占一个选择槽，却几乎不贡献输出，因而自适应减少“功能活跃”专家数。ρ从.4余弦退火到0，末期恢复标准top-k。静态核`Megatron-LM/megatron/core/transformer/moe/router.py::TopKRouter.mixed_softmax_routing`确认选择混合、聚合用原概率；`TopKRouter.forward`结合`router_ablation.py::cosine_rho`确认迭代调度；独立评估若拿不到训练iteration会fallback到ρ_max，README要求最终checkpoint显式设ρ=0。

§4/Tables1–4/AppB：DCLM、Llama式MoE、seq2048，100M–2.4B总参数/38M–270M激活参数，4.4–12.8B训练token，2或4 H100；64/128 routed+2 shared，每token选6 routed。2.4B PPL21.17→20.50，六常识任务均值45.23→46.87；100M均值仅36.61→36.82，不能泛称所有任务显著提升。迭代耗时近似相同，**固定实际激活专家数，功能压低不等于省掉expert计算**。无共享专家时三种较大模型几乎无收益，625M还略退；不同有无共享配置的GPU数/global batch亦变，不能当严格同并行度控制。

**意义/建议深读并小规模复现**：为“增专家/增top-k仍没增益”提供路由与表示两层诊断，优先测shared expert占比、router干预、有效专家数。论文未给多次完整预训seed方差、超过2.4B/12.8B token或真实MoE serving验证；单步时间表的波动不能当预训种子置信区间。稳定ID：`arxiv:2610.22353v1`。

<a name="posttraining"></a>

## 后训练与论文修订

### Baselines and Score Centering：正则化IS后，奖励中心化与score中心化的关系取决于粒度

**University of Illinois Chicago / Perplexity AI；2610.22263v1，2026-10-15T17:28:50Z；旧缺口补录。** [论文](https://arxiv.org/html/2610.22263v1)。理论note，未给专用实现/训练实验，不报能力或性能提升。

§2–3指出clipping/truncation/masking后的IS加权score期望一般不再为0，奖励平移会改变梯度。作者改以reward与加权score的**协方差**为estimand，恢复平移不变；无ratio正则时才还原相应policy gradient。sequence-level IS针对真实目标，token-level本来针对替代目标，不能把后者也叫全序列目标无偏。

§3.3/Proposition11：sequence级用group-mean reward时，再减population-mean score会在有限样本中精确抵消，因而多做无益。§3.4–3.5/Proposition15：token级prefix的mean score随trajectory变，group reward均值无法消掉；不做条件score中心化会多一个协方差项。known-mean estimator可group1，但比group-mean低方差取决于baseline是否接近prompt-dependent方差最优值，偏离可任意更糟；不是“小group必然更好”。理论依赖独立rollout、有限期望终止/可积性与可定义IS，未对所有实际PPO/GRPO clipping程序给即用等价结论。

**意义/建议深读**：复查异步RL或train/inference mismatch修正时，应先写清sequence/token ratio、正则函数、baseline及实际estimand，再审reward常数平移测试。先实现受控数值校验，缺少训练实验证据前不建议只凭note替换生产loss。稳定ID：`arxiv:2610.22263v1`。

### Un-OPD v2：核心方法与四张主表已在v1，新增的是实现细节和静态/大block边界实验

**CityU Hong Kong / AMAP Alibaba / CUHK / CityU Dongguan；2610.05373v2，修订2026-10-15T15:52:33Z；旧修订缺口补录。** [v1，10-04首发](https://arxiv.org/html/2610.05373v1)、[v2](https://arxiv.org/html/2610.05373v2)。未找到专用已发布实现；v2 reproducibility statement补充的是算法、实验与证明的文内定位，并未给可核代码发布链接。

本轮对照两个完整HTML：**Table1–4的正文文本逐字相同**；context-boundary对齐、support rebalance、rollout reuse及152→72 GPU-hours等结果均已在v1，不能当本次新方法。v2新增AppA算法、AppB训练/基线/评估配置、AppC五类边界分析、AppD证明与AppE例子；版本diff证据另存。

AppB/Table5把主实验具体化：SDAR1.7B/4B学生、TraDo8B冻结teacher，8 H20/BF16，64prompt×1rollout、最大2000token、LR2e-7、1000更新、λ.4、reuse10；主math avg@4，AMC avg@16。新增AppC显示：静态每步只unmask1token时，1.7B v2报告step1000 MATH50060.8、比TOPD+2.9pp；对称block4下4B两方法约71–72%，无决定性差距；student block16动态解码后期约67 vs TOPD61，属于新增较大block证据。图阴影是**最近五checkpoint局部标准差**，不是独立训练seed区间；动态decode下G1可暂胜G3，提示置信度/解码策略与“内在能力”的区分还需控制。

**建议续读版本增量**：重点AppB/C，复现实验需同时固定static/dynamic schedule、token预算与输出长度；新增response-length图仅1.7B、MATH500，长度约560增到650–670再平台，不能据此消除全部长度混杂。标题unbiased不意味着所有优化步骤、reuse off-policy分布与能力提升皆严格无偏。稳定ID：`arxiv:2610.05373v2`（前版`v1`）。

<a name="inference"></a>

## 推理与内存优化研究

### N-Gram Memories at Two Bits：压缩可寻址表记录，主要收益是内存可行性

**Dnotitia / KAIST；2610.21810v1，2026-10-15T12:57:19Z；旧缺口补录。** [论文](https://arxiv.org/html/2610.21810v1)。未找到本篇专用公开实现链接；以下机制来自论文，未代码验证。

n-gram memory每token只读少量hash行，普通dense权重量化的跨行编码可能让一次lookup解码无关数据。§3的Conditional Program给每行独立Gaussian trellis code与局部norm恢复scale，共用decoder；保留hash地址、backbone及readout，按需解码。norm恢复优于直接最小化行重构MSE说明**表重构误差与输出行为不是同一目标**；这是后训练表量化，并非重新预训练模型。

§4/Table1、AppA–B：DeepSeek-V4.1-Flash、Qwen3.8-Flash-Next、LongCat-Flash-Lite三种已有hash memory，40任务/五组、greedy生成。2-bit指code，含每行BF16 scale实际2.06/2.10/2.06 bpp；表体积202.7→50.7GB、102.4→13.4GB、62.8→8.1GB。整体分79.8→79.9、75.6→75.5、68.2→67.3，但LongCat HumanEval68.9→63.4（164题greedy），不能用整体均值写无损。相对2-bit RTN输出KL低41–63%，对QTIP只低3–9%且多一行scale。DeepSeek编码约4 H200×3.1h，是一次性额外成本。

§5/Table5、AppG：4 H200、batch8/4k prompt/256 decode，FP8→CP prefill64.66→13.27s，decode795.7→792.9ms几乎不变；8 H200显存宽裕时共同可运行点step时间差≤2.4%。更大batch/长prefill收益主要来自释放HBM、避免OOM/内存压力，非所有请求快4–8倍。DRAM lookup只占eager forward≤.8%，SSD每行仍一次4KB read，压缩不会直接减少每次miss的I/O粒度。Gemma PLE边界实验需保护BOS/高频行，未支持把uniform无旋转方案推广到所有embedding。

**建议复现**：适合大hash表挤占HBM的部署；分别测logical table bytes、实际padding/RSS、lookup、完整step、质量与编码成本。单点容量测试部分仅一次、八卡结果“至少2048batch/128k”是测试网格上界，不是极限。稳定ID：`arxiv:2610.21810v1`。

### DropKV：单token输出扰动评分便于kernel化，理论“近优”有条件且代码pooling与论文不同

**SUNY Albany / IBM Research / Northwestern；2610.22130v1，2026-10-15T16:05:17Z；旧缺口补录。** [论文](https://arxiv.org/html/2610.22130v1)；[代码基线afb97c2](https://github.com/AozhongZhang/DropKV/tree/afb97c2af09418ca58399dba50e6f956bb8ef153)。

§3把联合eviction的组合目标拆成每token单独移除对attention输出的扰动，兼看attention权重及value距当前attention输出的残差，在尾部query窗口累积再保留高分token。Triton路径先算输出/LSE，再分块重算权重累积score，避免完整attention矩阵。静态核`dropkv_triton.py::_dropkv_score_kernel`对应平方概率比×残差平方；`dropkv_cache.py::DropKVCache._compute_keep_indices`负责评分、保护末窗口和top-k。**差异保留：论文Algorithm1写MaxPool1D，作者当前代码用avg_pool1d**，本轮未运行对比，不能称代码完整精确复现论文设置。

§4/AppC：常数近似保证要求候选池的被删attention小、残差处于同向cone；三模型128k被测(layer,head,query)同时满足两条件的比例仅89.1%/87.5%/73.4%，相应最坏界factor中位10.4/11.8/11.3，远非无条件接近最优。小子集穷举经验比值较好，也不能覆盖整个cache组合或未来query。

§5/Table2、4/Fig3/AppA：Llama3.1-8B、Mistral7B-512K、Qwen2.5-7B-1M，A10080GB、5%cache budget。LongBench平均45.51/36.22/42.66，最强其他eviction约44.64/35.54/41.72；FullKV仍49.91/40.15/46.47，有明确质量损失。128k/B1/window8 scoring相对naive PyTorch19.8x、scratch约9GB→17MB，**匹配batch prefill仅-0.1%到+9.9%**。1.75–8.22x decode throughput按各方法最大可行batch比较，主要是压缩KV/容量收益；不等于Triton decode kernel提升，作者当前只加速prefill scoring。

**建议复现**：长context显存约束下值得测，但先对齐pooling、预算、recent-window与数值格式；同batch TTFT/完整生成和同质量吞吐分开。不要把带条件的单层扰动界当最终任务准确率保证。稳定ID：`arxiv:2610.22130v1`。

### FlashAttention with Statistics：从同一次proxy前向提取检测信号，仍需额外模型与有标签训练

**Applied AI Institute / Yandex / Risk AI Research Lab；2610.21483v1，2026-10-15T08:23:42Z；旧缺口补录。** [论文](https://arxiv.org/html/2610.21483v1)；[作者匿名代码](https://anonymous.4open.science/r/attention-guardrails-0C16)本轮403，未代码核查。

§2–3针对“给定上下文能否支持现有回答”的teacher-forced proxy检测：熵、prompt mass等可随online softmax累积（T1）；少数已知位置权重用LSE稀疏恢复（T2）；列和/sink统计需第二遍tiled score（T3）；不能bounded-state计算的量仍可能需要dense图（T4）。分级描述的是当前实现路径，不是所有统计的复杂度下界。输出统计只计算response rows，prompt仍可用生产kernel。

§4–5/Table2、AppH/I/L：单H100 NVL BF16、batch1，Qwen3-8B全层全head统计，2k→35k输入的全前向附加延迟16%→1.4%，**22.5KiB/response token是统计存储，实际峰值增量约28KiB**；0.6B短输入附加约170%，固定Python host开销47–60ms不能称免费。RAGTruthQA/HaluBench/TRIVIA+分组拆分，冻结0.6–8B proxy+监督logistic probe；4B在后两者观察F1超过九个4–9B judge中的最好者，同backbone probe也优于prompt judge。检测器用了标签和阈值选择，不能解释为零样本judge能力胜利；有些feature增益经Holm校正显著，有些没有，150–546 test groups使F1差异仍不稳。

**建议深读/复现**：适合已有有标签grounding样本、可承担一次小模型前向的场景；低开销统计不等于总检测成本低。实现尚非生产kernel、无生成中逐token路径，不支持sliding-window/softcap，只支持padding mask及head dim64/128/256。测的是给定context支持度，不能当一般事实真伪或已校准错误概率。稳定ID：`arxiv:2610.21483v1`。

<a name="engineering"></a>

## 工业开源项目

<a name="megatron"></a>

### Megatron-LM

#### Megatron：GLU checkpoint转换必须先于FP32 master初始化

稳定ID `github:NVIDIA/Megatron-LM:pr:7606`；[#7606](https://github.com/NVIDIA/Megatron-LM/pull/7606)，main，2026-10-18T00:32:04Z，merge `65ac8216e481b8a2e10d0f366865528fc010342a`。

启用专家GLU interleave后，checkpoint的连续 `[all gate | all up]` 行布局不能直接当运行时分块交错布局。新增 `prepare_glu_checkpoint_for_load/save`，读取时先按每个专家重排FC1 weight/bias，再加载模型与初始化FP32 master；保存模型重新归一为连续布局，optimizer master/moments保持运行时布局。索引专家、单grouped weight和独立shared expert设置均有处理。

已读MoE README、`glu_checkpointing.py::_permute_glu_rows`及save/load接入patch。只支持torch/torch_dist的显式model加载；DTensor、flattened model shard、in-place FSDP加载拒绝。FP32模型参数不支持交错；受影响FC1的GTP weight remat须为1。完整optimizer resume要求相同交错尺寸和专家ETP（shared为TP）；改布局须finetune/no-load-optim。正文概述“显式layout metadata”，实际README明确**不新增字段**，沿用保存配置；旧自定义交错模型checkpoint须外部规范化。作者CPU数值/Adam resume回归，未披露端到端GPU加载性能。建议**优先验证已有交错训练checkpoint的模型与master一致性**。

#### Megatron：MFSDP2按全局有效token归一，恢复dense/expert不同求和口径

稳定ID `github:NVIDIA/Megatron-LM:pr:7711`；[#7711](https://github.com/NVIDIA/Megatron-LM/pull/7711)，main，2026-10-17T18:23:07Z，merge `31265d3c102d850e8f087d868f099422a0649ba8`。

原先MFSDP2直接拒绝per-token loss；现在dense/expert各自mesh先归约，通过 `grad_scale`抵消mesh平均，累积结束后 `scale_gradients()`再除全局有效token总数，之后才clip/update。普通rank-mean loss保留dense系数1、expert系数1/EP。已读 `mcore_fsdp_adapter.py::_get_grad_scales/scale_gradients`与parameter-group patch，避免把各rank的均值再平均误当全token均值。

初始支持TP=PP=1、CP/EP与单optimizer instance；partial梯度尚未finalize时拒绝scale。作者4×GB200、DeepSeek proxy、EP2/四microbatch、EOD masking、50步三次全精度loss相同，golden从误用26.08更新到26.09；新的五次CI尚待。表里的iteration-time变化达-1537%来自首attempt慢启动，**不构成提速**。建议**考虑复现不等有效token的梯度与clip norm**。

#### Megatron：embedding直接累加main_grad，稀疏触行仍保留旧舍入次序

稳定ID `github:NVIDIA/Megatron-LM:pr:8142`；[#8142](https://github.com/NVIDIA/Megatron-LM/pull/8142)，main，2026-10-17T16:45:49Z，merge `a747c1bf6a61ea9d0395ee8c33791f542bd947f6`。

gradient_accumulation_fusion路径的 `EmbeddingWithGradAccumulationFusion.backward`自行累加weight.main_grad，向autograd返回无存储ZeroTensor，DDP hook仍运行但不重复加。Triton将(token,position)排序，按PyTorch旧算法每10行FP32 partial再顺序合并、先舍入梯度dtype再加main_grad；仅触及出现token行。已读 `fused_embedding_wgrad.py::can_accumulate_embedding_wgrad/accumulate_embedding_wgrad`及tensor-parallel接入。

快路径限3073–8192 token、GPU BF16/FP32梯度、连续hidden维及31bit可打包索引；其余回dense backward。MXFP8 LM head的ZeroTensor dummy需TE版本读取属性，旧TE会忽略。作者给VR200、129280×7168词表旧dense加约1ms/embedding use，是消除开销的动机，未给本PR训练整步倍率。建议**在MTP/重复token与目标dtype上验bit-exact，再量整步收益**。

#### Megatron：混合层PP前缀缓存统一决策，不能仅让部分stage继续

稳定ID `github:NVIDIA/Megatron-LM:pr:8250`；[#8250](https://github.com/NVIDIA/Megatron-LM/pull/8250)，main，2026-10-17T14:43:20Z，merge `8442db193ced81584c2e94faa352ec7307d18d83`。

Hybrid模型PP可能一段有Mamba/GDN、另一段只有attention，各stage原按本地层判定，导致Mamba缓存容量协商挂起或跳prompt token数不一致。把has/-has recurrent标记并入原block-count MIN all-reduce，无新增collective；所有stage得到pipeline级状态。memory-only前缀命中按recurrent规则统一不跳token；若正Mamba缓存预算且并非每stage都有recurrent层，则所有stage一致ValueError。

已读 `DynamicInferenceContext.__init__/_compute_prefix_match`与三回归。作者8×H100最新head上3679 inference测试、130 hybrid与58 GDN测试通过；该修复选择**拒绝不合法布局**，并未开放任意PP层分配的SSM缓存。建议**优先采用并检查PP层分布**。

#### Megatron：MFSDP多组owner规划与量化平面多轴gather补齐构件

稳定ID `github:NVIDIA/Megatron-LM:pr:7974`；[#7974](https://github.com/NVIDIA/Megatron-LM/pull/7974)，main，2026-10-17T05:08:40Z，merge `4ab3af12d274b3a9c330b7a88db8f8e2768be971`。

稳定ID `github:NVIDIA/Megatron-LM:pr:7727`；[#7727](https://github.com/NVIDIA/Megatron-LM/pull/7727)，main，2026-10-17T16:42:44Z，merge `e6913b8f3225215c773b2094c8dbaad6b2593f3c`。

owner规划从每buffer独立改为同DP mesh所有buffers联合计负载：先计单一holder张量，再按估算Newton–Schulz五次成本降序给最轻合法holder，tie按buffer/tensor/offset稳定处理。跨mesh输入拒绝。它是后续owner-compute的规划构件，没有端到端Muon加速证据。

`QuantizedDBuffer.allgather`接收多个mesh轴，先将一次性iterator物化成tuple，确保data/scale四平面使用同一轴序列并保留out复用。作者2×RTX A6000测试的2D mesh为(2,1)，7项真实MXFP8量化/GEMM跳过；字节平面正确不等于多非平凡轴MXFP8算术认证。建议**跟踪adapter/optimizer接入并补真实二维mesh验证**。

#### Megatron dev：DSv4确定性修复付出归约与初始化成本

稳定ID `github:NVIDIA/Megatron-LM:pr:7441`；[#7441](https://github.com/NVIDIA/Megatron-LM/pull/7441)，dev，2026-10-17T11:14:29Z，merge `7a39a747919e5a6ece25ea043cd2bec7cb6bfdae`。

仅进入dev；deterministic-mode下固定cuDNN compact top-k ID/softmax顺序、请求确定性sparse backward workspace、对indexer dK用无原子有序归约，并将grouped-MLP prob乘法移到FC2输出，拒绝FC2 bias与router fusion不合法组合。已读determinism文档、compact/indexer与experts关键guard patch。

要求cuDNN frontend≥1.28与SM100 deterministic sparse backward，不满足就fail-fast。作者CP2/PP1/EP2十步日志逐bit相同，未覆盖CP>2、多节点和混长THD。相对非确定性step慢14.8%；作者另关PyTorch全局fill_uninitialized_memory后慢6.1%，该全局改动**未被PR启用**且仅DSv4验证。建议**继续观察dev迁移，别把确定性等同默认数值路径或零成本**。

#### Megatron：其余状态变化与去重

#8287为已在main的#7522 KPool/NoPE/precision控制移植到dev（2026-10-17T08:22:21Z），不重复计新主线功能；KPool indexer loss仍须关闭。#8139把RL序列计数归到 `_samples_seen`，skip-train的统计口径会变；#7932 TrainState接管可变训练状态与checkpoint恢复，#8163删wandb悬空checkpoint artifacts并过滤下划线私有args。后三项已读正文，其中#8139已读patch；#7932/#8163深审仍待，不能借“维护”名义声称所有训练状态恢复路径已验证。#8105为dev的SP+MTP mask、batch dict重放alias与recompute线程CUDA context等修复，尚未逐文件审，保留候选。

<a name="vllm"></a>

### vLLM

#### vLLM：OpenAI路由可返回每段token的权重版本

稳定ID `github:vllm-project/vllm:pr:61347`；[#61347](https://github.com/vllm-project/vllm/pull/61347)，main，2026-10-17T02:08:08Z，merge `b3a7e2ff6e7d182cd9c230cdb04dd66043b9f413`。

completions/chat请求opt-in `return_weight_versions:true`，每choice返回半开输出token区间(version,start,end)，prompt不计；stream只在最后chunk返回整段列表，abort零token也补finish chunk与空列表。已读protocol、两个serving loop与async-RL文档patch。它暴露既有engine记录，帮助判断一个rollout跨了哪些权重更新，**不自动修正off-policy偏差**。

作者H20/Qwen3-1.7B测pause keep跨zero-lm-head更新、abort、n=2、async scheduling及reasoning parser；span边界对齐首个新权重token。Beam search与Rust frontend不在范围；reasoning隐藏token IDs时spans仍计全部生成token，重提旧输出需自行偏移。建议**考虑用于异步RL staleness审计**。

#### vLLM：M2N允许逐参数源mesh，失败传播不等于回滚

稳定ID `github:vllm-project/vllm:pr:60934`；[#60934](https://github.com/vllm-project/vllm/pull/60934)，main，2026-10-17T01:41:17Z，merge `6b6a6c5c98e4a23c77bb07d344c515f48c157193`。

同一训练rank集合可让dense用(DP,TP)、expert用(EDP,EP)，每个参数声明global shape/dtype/source mesh/placements；versioned有序manifest和SHA256摘要先全员达成一致，再建NCCL data plane、逐参数reshard。已读 `M2NParamMeta/M2NLayout`、manifest source、trainer/engine初始化与cleanup关键patch。

仍只支持同一完整trainer rank区间上的二维mesh，DTensor placements限Replicate/Shard；destination mesh仍session-wide，接收端reload策略未改变。vLLM校验身份/顺序/本地shape/dtype，native描述符/contiguity与容量属nccl4py/NCCL边界；raw pointer寿命由producer负责。作者4×GB200、NCCL2.30.7/extension0.1.0、205pass/24skip。中途错误可留下部分新权重与不确定collective序列，Python不能取消已运行RPC；应重建session，不能当原子热更新。建议**深入阅读并故障注入，而非直接默认上线**。

#### vLLM：EPLB NIXL与共享block-table分别补足复用屏障

稳定ID `github:vllm-project/vllm:pr:62081`；[#62081](https://github.com/vllm-project/vllm/pull/62081)，main，2026-10-17T23:18:02Z，merge `9c778e9ff8d0c78a241d7f530c441c2dc1d67533`。

稳定ID `github:vllm-project/vllm:pr:61174`；[#61174](https://github.com/vllm-project/vllm/pull/61174)，main，2026-10-17T08:25:02Z，merge `61bb590877521e2f4516912297a40f593e222ee9`。

#62081的NIXL READ绕过GPU stream，下一层可在前层expert_buffer拷出完成前覆盖它；ROCm分支在每层transfer context先sync current/EPLB stream。作者MI300X多层冗余专家测试idle重排5/100错误→0/100，注入延迟/模拟async也由稳定失败到通过；同步有成本，未给吞吐表，CUDA路径不受该guard修改。

#61174在batch缩小/dummy切换后清理shared BlockTables直到max(本次padding行,前次dirty行)，避免更大的MTP graph用旧行寻址别的请求KPool/状态；独立PCP out不改变共享dirty账。作者8×H20/GLM5.3Flash TP8/MTP3、CUDA graphs+CPU offload的47请求正确；这是cache生命周期回归，distributed PCP serving仍未测。已读两项真实patch，建议**优先回归跨层重排、缩批与缓存恢复**。

#### vLLM：MRV2先收KV结果再跑drafter，save完成仍在末尾

稳定ID `github:vllm-project/vllm:pr:62041`；[#62041](https://github.com/vllm-project/vllm/pull/62041)，main，2026-10-17T17:27:58Z，merge `0c5dd18e7a0ace18d99438fb16b8375572ea76b5`。

post_forward在target采样后启动deferred load并采集transfer results；有speculator时仅把wait_for_save/clear_metadata延后到drafter之后 `finalize_step()`。这样已完成load可及时报告，异步load也不被drafter阻塞，而draft写入仍在save结束前。已读runner/connector与顺序回归patch。KVConnector公共接口未改，未给服务性能对照；建议**按transfer结果、draft写入、save完成三个边界验trace**，不要将提早汇报等同远端WRITE停止。

#### vLLM：split top-p修跨块tie预算，并给小batch更多并行

稳定ID `github:vllm-project/vllm:pr:62046`；[#62046](https://github.com/vllm-project/vllm/pull/62046)，main，2026-10-17T02:26:58Z，merge `ce49c47cbb241b59a32ef110079a5261dbd08e49`。

稳定ID `github:vllm-project/vllm:pr:61996`；[#61996](https://github.com/vllm-project/vllm/pull/61996)，main，2026-10-17T04:22:12Z，merge `7b6310c109135b1b4aba42ed363697987380fd10`。

正确性修复在每个8192-entry块后累加boundary duplicate计数，避免下一块重新保留同一tie额度。仅top-p-only、B≤64且slice跨块时触发；作者L4/词表151936/4096ties例中旧保留概率质量0.994，正确应0.700，单块对照不变。已读一行carry与跨块回归，常用greedy GSM8K不经过此路径，不能代替采样分布验证。

性能项允许split超过一wave、每slice至少7168 vocab，并由MRV2给出has_topp_only_rows以跳全top-k时七个空kernel。作者B300/H200多词表算子最高数倍；B300/Llama3-8B c48 seeded top-p中median ITL约降5.1%，MiMoV2.6Pro TP4/DFlash7的22.7ms decode只少约0.27ms非MoE GPU工作。未给hint的全top-k caller可能慢2–10%。建议**正确性修复优先，性能按真实row mix验收**。

#### vLLM：batch-invariant开关补上Inductor编译设置

稳定ID `github:vllm-project/vllm:pr:61118`；[#61118](https://github.com/vllm-project/vllm/pull/61118)，main，2026-10-17T18:53:06Z，merge `03e8300d2cac77b24f7ab6013a803369ee64c19e`。

自定义RMSNorm在torch.compile下被native lowering替换，跨worker/编译piece的autotuned reduction block不同，SP MoE会让同token落到不同TP chunk后改变舍入。现在 `override_envs_for_invariance`设 `torch._inductor.config.batch_invariant=True`，保留SP并修编译原因；未设VLLM_BATCH_INVARIANT时不启。

作者4×GB300/Qwen3-30B-A3B-FP4，DP2×TP2+EP、32prompts独立/合批、固定seed/temperature.6/max8token等逐step logprob比较，旧可间歇1–14/32不一致。W4A16实验还叠未合入#61038，不能当纯本PR支持认证。已读实际三行patch及条件，建议**验证完整编译产物与跨batch逐bit行为**。

#### vLLM：MXFP4专家LoRA接通TRTLLM，需显式backend

稳定ID `github:vllm-project/vllm:pr:61979`；[#61979](https://github.com/vllm-project/vllm/pull/61979)，main，2026-10-17T12:17:57Z，merge `837e87e1c5024843b66383ef1bd60ca5672a9575`。

新增 `TrtLlmMxfp4LoRAExperts`，gate/up delta在激活前送入GEMM，down delta从postactivation取值后FP32 final reduce；GPT-OSS静态FC1 bias并入BF16 delta，处理[up|gate]布局。已读expert class、oracle opt-in/拒绝与output unpadding patch。auto选择保持旧值，须显式flashinfer_trtllm；MXFP8 activation分支明确拒绝，TP>1依赖已合入#61235。

作者GB300、旧arm64 nightly+patch overlay，gpt-oss20B TP1/c64/256输出时LoRA tok/s：Triton4593、Marlin7047、新7699；未在本PR base原生重建。random small/large adapter只是作用方向/近似一致性测试，large会破坏模型；EP、多loaded adapters、fully_sharded_loras与TP4/8仍未验。建议**对真实adapter做质量及多adapter复现**。

#### vLLM：Qwen3 XML工具长值不再逐delta重扫全串

稳定ID `github:vllm-project/vllm:pr:62027`；[#62027](https://github.com/vllm-project/vllm/pull/62027)，main，2026-10-17T18:49:07Z，merge `5c1c903501927ea7c6c699aabd4c271015e66e71`。

复用arg_value_extender，安全的最后open parameter只对新增delta做JSON转义；重复key、闭合tag/下一parameter跨delta时回converter，保留首尾换行与MiMo verbatim差异。已读 `_param_value_end_window/_param_value_extender`，其guard要求converter/regex同步演进。

作者Apple M5/Python3.12、每delta一Qwen3 token、七次交替：12KB write_file parser累计2.1s→11ms，25KB8.4s→22ms；27500随机stream逐DeltaMessage一致。没有GPU server E2E，不能把parser CPU倍率写作模型提速。建议**考虑用于长文件tool参数，保留split-closer与截断差分测试**。

#### vLLM：CuTe cached-context prefill可选，非全部MLA自动替换

稳定ID `github:vllm-project/vllm:pr:62069`；[#62069](https://github.com/vllm-project/vllm/pull/62069)，main，2026-10-17T05:21:56Z，merge `90d6d871d6700dd229a76b788bf0dca22bab6c70`。

TRTLLM_RAGGED可opt-in cute-dsl，并要求query quantization；只处理cached-context attention，新token prefill及default dispatch保持原路。Kimi K3要求FP8 KV；head dim192要求FP8 query，256不支持，strided V为CuTe才额外打包，其copy可抵消小负载收益。已读选择/metadata兼容关键patch。

作者GB300/FlashInfer0.7.1、Kimi12local heads、8K cache的CUDA graph微测1024/8192/16384 query约1.20–1.28×。8×GB300两节点K3 MXFP4+DSpark4、maxbatched256强制chunk，trace确认CuTe被调用，GSM8K1271/1319；无匹配baseline的服务吞吐，不能据此称全模型快28%。建议**同cache长度与V-layout复现**。

#### vLLM：小显存Gemma4图捕获与DeepSeek权重加载修峰值

稳定ID `github:vllm-project/vllm:pr:61365`；[#61365](https://github.com/vllm-project/vllm/pull/61365)，main，2026-10-17T11:04:50Z，merge `ec8b20fbf01e04d33bb059e555a9b5f39fb4cc46`。

稳定ID `github:vllm-project/vllm:pr:58857`；[#58857](https://github.com/vllm-project/vllm/pull/58857)，main，2026-10-17T09:37:57Z，merge `026f6ab2689397e64c1f53a5ac57d37c4dc46b67`。

Gemma4 encoder把graph按soft-token row宽70/140/280与budget分别捕获，buffer上界budget×9 patches，修所有row都按2520patch分配的启动OOM及video gather宽错。作者RTX5090 32GB/Gemma4-26B-A4B-NVFP4，旧4次启动均OOM，新3次成功，21图pool2.61GiB；video/eager mean cosine.9994。已读model patch，不泛化为其他模型图管理器优化。

DeepSeek V4/V4.1 VL loader以 `stream_language_model_first`让LM边到边加载、延后clone vision/aligner，维持child一次finalize，去全流sorted持有所有tensor。默认mmap不是同等匿名内存OOM；4×DGX Spark/TP4旧0.28.1测试sorted248s、stream202/212s，fastsafetensors88s还依赖未upstream fixes，不算本PR单独结果。建议**按loader格式和启动峰值验证**。

#### vLLM：不齐PP的CPU offload可显式统一块数，自动尺寸尚未修

稳定ID `github:vllm-project/vllm:pr:61674`；[#61674](https://github.com/vllm-project/vllm/pull/61674)，main，2026-10-17T02:11:13Z，merge `ce8853d27a977bd5d955c7e03c3d1dd9c735ef86`。

SimpleCPUOffload新增共享 `num_cpu_blocks`，worker按本地真实byte/stride容量拒绝过大/非正/非整数，调度与worker用同count；省略仍各stage独立自动计算。异构PP层数若scheduler25而worker23，旧DMA会越界；本配置需operator选所有rank都容得下并一致下发。

作者2节点16×B200/KimiK3 TP8/DCP8/PP2、FP8KV/no spec、128MiB/rank，在旧v0.30镜像backport：统一23后OOB copy40→0/1088，强制CPU reload16rank输出16tokens一致。它提供显式替代，不是自动协商，也不证明所有历史crash源于此。已读manager/worker校验patch；建议**先修部署尺寸并在拓扑改变时重算**。

#### vLLM：AMD逐层选择MXFP4布局，单节点allreduce恢复MNNVL优先

稳定ID `github:vllm-project/vllm:pr:61278`；[#61278](https://github.com/vllm-project/vllm/pull/61278)，main，2026-10-17T09:00:29Z，merge `2067297b2f15f0776be6d2a7e1b435bd20af60c1`。

稳定ID `github:vllm-project/vllm:pr:61989`；[#61989](https://github.com/vllm-project/vllm/pull/61989)，main，2026-10-17T09:55:28Z，merge `2b85d9cb65c1f56c347cb7f56f1596a6b8b28373`。

ROCm a4w4不再全局env强开preshuffle，而在load时要求AITER对ASM prefill及preshuffled Triton decode均有tuned config才启，避免Kimi/DeepSeek等无调优decode退步。作者8×MI350X、Llama3.3-70B MXFP4、1024/1024随机负载、dummy weights：TP1大c吞吐可升53.3%，但TP8/c16 E2E反慢5.2%；性能测试dummy不证明精度，另250题GSM8K/GPQA有限eval单列。不能把未合入AITER#6546的405B增益算当前默认。

FlashInfer allreduce node-local组auto恢复mnnvl，缺NVSwitch multicast可回trtllm；多节点Blackwell/禁Hopper策略不变。作者8×B200/Qwen3.5-397B NVFP4、8K/1K c128在v0.32rc2上5462→5728tok/s、TPOT21.78→20.22ms。已读两项dispatch条件；建议**按实际互连与shape验收，别只看TTFT**。

#### vLLM：MoRIIO hybrid WRITE等待全组状态，仍有限模型边界

稳定ID `github:vllm-project/vllm:pr:60544`；[#60544](https://github.com/vllm-project/vllm/pull/60544)，main，2026-10-17T04:57:59Z，merge `41b8c158097c57ccf61626680d0fd13790bd120b`。

decode以transfer-group序宣告blocks，prefill选最终live prefill的KV/conv/SSM slot、排除spec scratch，发布N−1 prompt状态，decode等全部transfer再重算末token。wait_all有有限deadline，旧MoRI bounded polling回退。已读group、final-slot、wait_all关键patch。

要求两端read_mode:false、matching布局/TP、mamba-cache-mode align，DSpark同draft；SWA WRITE与其他hybrid speculation不支持。作者真实KimiK3 TP8/DCP8/DSpark4，两次GSM8K93.40/95.00 vs direct94.31%，自然接受率67.86%，289回归；这是运行正确性/波动，无吞吐配对。本轮未读取GPU具体型号，故不报告其性能数字。建议**先验全组completion和本地cache全命中退出路径**。

#### vLLM：Hadamard末warp越界与beam allowlist补正确性

稳定ID `github:vllm-project/vllm:pr:61944`；[#61944](https://github.com/vllm-project/vllm/pull/61944)，main，2026-10-17T07:52:17Z，merge `08a65e287e55099255ce178c5b5f58ce42dfd461`。

稳定ID `github:vllm-project/vllm:pr:47630`；[#47630](https://github.com/vllm-project/vllm/pull/47630)，main，2026-10-17T08:50:40Z，merge `eb8de8d1a1a79fce8bb2ce90be8620768d0f8347`。

稳定ID `github:vllm-project/vllm:pr:58778`；[#58778](https://github.com/vllm-project/vllm/pull/58778)，main，2026-10-17T09:08:14Z，merge `f98ae6a9fd2c73a2f80dd9fe4c534676f689394d`。

Hadacore masked variant仍对不存在chunk发cp.async，42×128 BF16、每warp两chunk的末warp多读512B；仅allocator恰落段末才fault。现在不load超过real_num_chunks，作者H200 MIG重放原8文件顺序三次244pass/25skip，避免将孤立文件通过误作排除证据。

beam路径原丢allowed_token_ids且候选来自未mask logprobs；现在透传并ranking前过滤，小allowlist精确取logprobs，offline还与grammar相交。Qwen3-0.6B单token banana允许集旧返回Paris、新正确。另async resumable请求若preempted时旧step才触stop，清理会误删新队列项、fold以computed0截掉prompt；#58778同时补re-enqueue与保留旧token再重算，单独一半修复仍失败。作者GB200 streaming集14pass。已读这些实际patch，建议**按边界用例优先回归**。

#### vLLM：模型支持与autotune成熟度

[#55671](https://github.com/vllm-project/vllm/pull/55671)进入main `71c20f73f2`（完整元数据见证据），managed autotune v2只自动给单rank/单node/PP1/nonelastic且API满足者，多rank仍legacy同步。当前Torch2.14/FlashInfer0.7.2栈只核CPU/source；历史B200 GPU9例使用0.7.0.post1且非本head重建，不称新依赖栈已验。#61455以显式draft-layer tag替代KV group猜测，PP投影保留全局flag；新flag不改变KVCacheSpec equality/grouping，untagged runner仍保守全flag。其DFlash hybrid E2E因模型draft自身startup问题未测，Qwen3.5 MTP小负载hit-rate未变，不报告普遍命中率提升。

<a name="verl"></a>

### verl

#### verl：主窗未见新增合入或release

已读默认main的13个回溯commit列表及主窗merged查询（0）；最新GitHub release仍v0.9.1（2026-09-20）。72h内FSDP2阻塞D2H、TPU配方与Blackwell BF16 rollout等已在10-17或更早日报报告，本轮按PR+状态去重，不重复写作今日发布。此结论限已核release/main/merged inventory，不表示所有open提案或分支无活动。

<a name="slime"></a>

### slime

#### slime：长上下文recipe默认扩大，完整长度训练尚未认证

稳定ID `github:THUDM/slime:pr:2479`；[#2479](https://github.com/THUDM/slime/pull/2479)，main，2026-10-17T07:11:01Z，merge `859938dcb2b0791dbae9084a4d9ae91a4f38f3b7`。

这个复合PR除CI修复，还把维护中的RL recipes默认response上限改128K（GLM5.3为256K）、打开适用SC/R3与省内存配置并迁移旧Qwen启动例。已读参数/launcher与相关TMS pageable-host fallback关键patch；不能仅按标题过滤为维护，也不能把配置上限当已完成长序列训练。

作者称受影响GPU回归、20生成器测试、31严格CLI、72shell检查通过；Flash64条真实sample最长88764token，replay的训练/save/offload/weightsync通过。**完整128K/256K训练未验证**，额外debug recovery checkpoint尝试host OOM。建议**先按最大response长度预算host/GPU内存并分阶段复现**。

<a name="sglang"></a>

### SGLang

#### SGLang：DSv4.1 decoder replay转默认，logprob请求另走全extend

稳定ID `github:sgl-project/sglang:pr:44403`；[#44403](https://github.com/sgl-project/sglang/pull/44403)，main，2026-10-17T02:52:03Z，merge `360382454170073d54c202745e9884ee51f72fa3`。

稳定ID `github:sgl-project/sglang:pr:44850`；[#44850](https://github.com/sgl-project/sglang/pull/44850)，main，2026-10-17T08:25:34Z，merge `a2953d7b7fe9c7bb705e681e5011d3db80adfac7`。

稳定ID `github:sgl-project/sglang:pr:42901`；[#42901](https://github.com/sgl-project/sglang/pull/42901)，main，2026-10-17T21:12:00Z，merge `05ce0e980f00aa80864190d6f1cd5d9b685802f1`。

CUDA或HIP满足条件的standalone DSv4.1默认启decoder SWA bounded replay，保留显式full-prefill opt-out；实际resolver要求disaggregation=null、无prefill-CP/attention-CP/DCP冲突、受支持attention-DP/backend组合，且未要求返回全部hidden states。PD、CP、full prefill graph及不兼容配置不自动启；本改动不自动开启encoder replay，但可与显式encoder replay组合，另受encoder专属校验。它改变晚层SWA语义，非数值等价。默认PR同时修候选bitmap跨1,048,576 token边界，按capacity选16/32KiB bitmap、拒>2M容量；这不扩大模型context上限。

同窗后续#44850已让prompt-logprob/需hidden rows请求late-layer whole-extend，tail超图块时eager，并DP投票一致；**最终main不能沿用#44403正文的“prompt logprobs unsupported”**。要求与trainer匹配logprobs的RL仍应显式关replay。#42901把layer20拆为全token写compressed KV，再仅尾128做indexer/attention/MoE，ROCm留旧融合路径；最后128token的SWA也缩短，属于有意数值变化。

已读arg resolver、tail layout、DP graph vote与layer20关键patch。作者4×GB300/DSv4.1Flash TP4/EP4/DSpark5，固定token三次：32K/64K输入单请求TTFT约480.92→346.89ms、974.77→700.62ms，c16/32K+128输出256.38→337.22tok/s。准确性历史single run无cap AIME469/480两边同，DeepSWE78→77/113；MMMU变长更多且E2E更慢，不能称等价或普适加速。这批完整eval早于最新default patch，最新仅resolver/边界回归。layer20后续4×B300新head短样本TTFT再约4–8%改善但GSM8K有run波动，**不能把两表倍率相乘**。建议**深入阅读并按RL/agent负载保留full-prefill对照**。

#### SGLang：encoder replay禁止重写缓存，ROCm attention-DP加拒绝门

稳定ID `github:sgl-project/sglang:pr:44857`；[#44857](https://github.com/sgl-project/sglang/pull/44857)，main，2026-10-17T07:56:18Z，merge `ef26f6ad49b03d85bea33d080c2c7d46e2f84097`。

稳定ID `github:sgl-project/sglang:pr:44934`；[#44934](https://github.com/sgl-project/sglang/pull/44934)，main，2026-10-17T22:47:42Z，merge `563cd60345e0d8940b6ae04e96e638f483d69036`。

folded encoder replay旧fused c1/c2/indexer仍写cached row，unfused还把筛过输入与未筛slot错配。现在replay row的compressor out设padding slot0，保留原KV/compressed/index key；新row保持自身slot，删分散的compress_skip。作者GB300对prefix-hit前后bytes快照，旧3712行中3308–3712变动，新两路径均0，已读metadata与backend patch。

encoder replay+attention DP在ROCm现在启动拒绝：其prefix-hit另起forward，只有命中rank进入collective而非CUDA folded同batch。该门在CUDA放开DP时被误删，本次恢复ROCm专用约束。它不影响CUDA，不是ROCm encoder replay+DP已可用。建议**优先回归多次prefix hit并尊重平台门**。

#### SGLang：HiCache写回和Full图load-back补前后两道stream依赖

稳定ID `github:sgl-project/sglang:pr:44800`；[#44800](https://github.com/sgl-project/sglang/pull/44800)，main，2026-10-17T08:31:41Z，merge `f75232cb42a80a267f56e5547ace5c35a648e39c`。

稳定ID `github:sgl-project/sglang:pr:44220`；[#44220](https://github.com/sgl-project/sglang/pull/44220)，main，2026-10-17T11:13:58Z，merge `ae28306a74fc9ba56daab9ccb06dcf45659f58af`。

write_back旧insert-time SWA backup及blocking drain可能在forward写完前D2H，日后恢复陈旧KV。Python/Rust tree插入时不再发write-back BackupKV，驱逐再备份；blocking drain令D2H stream wait forward stream，异步flush/write-through不变。已读tree、controller、scheduler接入patch，作者单CUDA两tree backend的注入延迟负对照由失败到通过，Rust931tests；无通用吞吐收益。

反方向Full prefill graph跳过Python逐层H2D wait，load仍in-flight时可能读未到状态并将错KV缓存。现在 `PrefillCudaGraphRunner._wait_for_layer_loads`只在Full与active counter等最后层，再replay；eager/breakable保留逐层overlap。作者Inkling小测试H200/GB300延迟注入旧KL4.4e−3/1.7e−2，新0；4×GB300 InklingSmall TP4缓存160K的TTFT113→115ms，较eager195.4ms仍低。breakable捕获state-reading KDA另属#44414，**本PR未修**。建议**优先采用并故障注入复制延迟**。

#### SGLang：MegaMoE EPLB必须一起迁移派生scale

稳定ID `github:sgl-project/sglang:pr:44847`；[#44847](https://github.com/sgl-project/sglang/pull/44847)，main，2026-10-17T05:22:41Z，merge `9259553f0957fbbdaa4d226497c83acafd99f69f`。

SM100 MegaMoE的UTCCP转置scale、NVFP4 alpha/activation scale原只是普通tensor副本，EPLB只搬专家parameters后副本仍属旧专家。现把这些派生项注册为requires_grad=False参数，get_moe_weights重排时一起搬；SM90/ROCm原直接读被改参数，保持原路。

已读两个register_parameter patch。作者2×GB300/DeepSeekV4.1Flash DEP2、MegaMoE、每32iter rebalance、greedy GSM8K200：旧EPLB72.5%、新97.0%、关EPLB97.5%；旧因输出更长产生更多rebalance，不是负载均衡提高能力。建议**开启EPLB部署优先核对所有派生scale随专家身份移动**。

#### SGLang：joint reclaim少驱逐，Mamba压缩不再先gather大临时量

稳定ID `github:sgl-project/sglang:pr:39294`；[#39294](https://github.com/sgl-project/sglang/pull/39294)，main，2026-10-17T11:46:49Z，merge `ee30dc4d40993045273f0a90260db7efa302c214`。

稳定ID `github:sgl-project/sglang:pr:44097`；[#44097](https://github.com/sgl-project/sglang/pull/44097)，main，2026-10-17T11:56:41Z，merge `025655068ca9d3c571200d0210221416445de619`。

joint FULL/SWA/Mamba分配先用只读充分回收谓词，证明不可能的byte/ID需求在驱逐前拒绝，按有界quota回收、阶段边界再实际准备校验。作者fixture六token需求驱逐5→3，四tokentri-pool从失败变成功只驱逐1，不可能200token需求保留全部96缓存。几何谓词只是充分证书，不是完备可行性求解。

compaction对disjoint physical state envelopes直接copy_pages，去主state gather scratch，保留siblings/COW与current stream语义；CPU fallback/sibling callback仍可分配scratch。已读reclaim phase与move关键patch，作者最新H200 combined head87tests/211subtests、早期B30018个scratch配置旧失败新通过；**没有新model-serving/多GPU/完整HiCache性能认证**。建议**在内存逼满时验证payload、reader completion与缓存保留量**。

#### SGLang：block-FP8-as-MXFP8三条prefill优化须分别计量

稳定ID `github:sgl-project/sglang:pr:44838`；[#44838](https://github.com/sgl-project/sglang/pull/44838)，main，2026-10-17T09:45:54Z，merge `a126e8d640c707f855e8da31f906035883f4ff70`。

稳定ID `github:sgl-project/sglang:pr:44837`；[#44837](https://github.com/sgl-project/sglang/pull/44837)，main，2026-10-17T06:47:42Z，merge `14c4dda24e1cf623935331dd77831e842e301f93`。

稳定ID `github:sgl-project/sglang:pr:44842`；[#44842](https://github.com/sgl-project/sglang/pull/44842)，main，2026-10-17T07:19:36Z，merge `bce7b27aa6aabda681351f8f4c20776981b5d5d4`。

#44838在load时把query head布局折进wq_b，32row scale精确细成16row后同置换，跳每层pack_query的zero+strided copy。#44837共享专家从仅128×128 block gate放宽至K block为32倍数，scale展开到1×32 UE8M0后并入MegaMoE；非power-of-two scale仍有UE8M0舍入。#44842让已ready的block-FP8-as-MXFP8 FlashInfer CUTLASS/CuTe模型参与既有autotune，deterministic/batch-invariant仍pin tactic。已读三项实际选择/scale转换patch。

同类GB300 DEP2、DSv4.1Flash、MegaMoE/Mega-mHC、decoder replay、eager16K/rank，三个独立A/B：query布局16K chunk247→235ms、128K1906→1764ms；shared fusion241→235ms、1925→1842ms；autotune247→235ms、1906→1773ms。不是同一累计配置的三步收益，**不能相乘**；前两项GSM8K native-acceptance gate作者仍未跑，cache-hit结果有正负波动。建议**先验数值，再profile实际启用的layout/fusion/tactic**。

#### SGLang：BCG metadata显存缩减与QSA FP8 scratch各有边界

稳定ID `github:sgl-project/sglang:pr:44920`；[#44920](https://github.com/sgl-project/sglang/pull/44920)，main，2026-10-17T23:05:29Z，merge `e4512a5a17bac4a050df738e984682c88af06e56`。

稳定ID `github:sgl-project/sglang:pr:44114`；[#44114](https://github.com/sgl-project/sglang/pull/44114)，main，2026-10-17T18:50:39Z，merge `3b34d9f8b6edc23216f29c1edf02d1ff0306f9de`。

BCG限low-ratio页表到已有graph context limit，serial capture buckets复用full index storage、增长时保旧地址，c1/c2 fully-overwritten combined scratch可共用，c0/c128及full/tail仍隔离。作者4×GB300 DEP4/DSpark5/encoder replay OFF/decoder ON，capture8K的增量16.18→12.24GiB/GPU、2K6.77→5.10；这是capture结束增量，不是server总量或显存峰全面降24%，重base后未复测该内存表。

QSA在SM10x且FP8 E4M3 KV pool让packed scratch保FP8、Q仍BF16，减流量并免dequant；保留tail zero/int64 offsets修复，SM120/FA fallback/prefill不改。作者VR200/Qwen3.8FlashNext NVFP4 TP1/MTP3、c64，decode20.686→19.908ms(-3.76%)、三对1h agent serving总tok/GPU+2.29%。AIME差+1.56pp的95%区间跨0，不能称质量提高。已读storage复用与dtype guard patch；建议**分别量capture内存和真实服务收益**。

#### SGLang：Mooncake直连L3支持SWA+Mamba，依赖图加载屏障

稳定ID `github:sgl-project/sglang:pr:41343`；[#41343](https://github.com/sgl-project/sglang/pull/41343)，main，2026-10-17T20:24:09Z，merge `05c15390984d80a8bb4bbc415a7653e174c27c0b`。

稳定ID `github:sgl-project/sglang:pr:43817`；[#43817](https://github.com/sgl-project/sglang/pull/43817)，main，2026-10-17T11:08:18Z，merge `5fcb07da34abb98a5210ff40d03290a74ac0df01`。

稳定ID `github:sgl-project/sglang:pr:43387`；[#43387](https://github.com/sgl-project/sglang/pull/43387)，main，2026-10-17T12:20:50Z，merge `86e67f7d93cb8720538d25123c5ff45a8683c4ba`。

#41343直连linker为FULL/SWA打包每层KV对象，Mamba按checkpoint node末页hash存slot状态，仅可恢复到存过state的边界；LOAD分配slot，PREPARE接管request状态并取消旧CoW/ReplaySSM cursor，ABORT释放。依赖已在同窗合入#44220，否则Full图可能读未完成load。作者GB300/微型Inkling+Mooncake TCP，四次及延迟注入KL0；int8 Mamba、scaledKV、unified memory、side states、UMBP及FULL+Mamba无SWA仍不支持。

另HF3FS MLA原所有TP共享KV key但Mamba分片，#43817给Mamba加TP rank/size后缀，旧未标rank的条目不复用。#43387在no_buffer+spec CPU grammar/finish截短后，只有live state长度等owned KV并保留完整verify语义才捐缓存，否则仅跳本请求插入，保已有prefix/正常清理。作者GB300/Qwen3.5-0.8B NEXTN长度截断，旧KL0.877/4.107，新约.000235/.000942但cache hit变少。已读slot adoption/key与finish guard，建议**优先回归状态身份而非只看cached token数**。

#### SGLang / Ascend：FIA TND替RingMLA，FuseEP共享专家独立修复

稳定ID `github:sgl-project/sglang:pr:40131`；[#40131](https://github.com/sgl-project/sglang/pull/40131)，main，2026-10-17T02:12:42Z，merge `2d3dd3168d4a8e5b69ebd4c9b41a28fa93e6c16e`。

稳定ID `github:sgl-project/sglang:pr:40516`；[#40516](https://github.com/sgl-project/sglang/pull/40516)，main，2026-10-17T08:19:44Z，merge `7a8804edb5fcf72687fc9e352e6517969c3c497b`。

MLA prefix path现在两次FIA V2（current causal及prefix）再FP32 attention update，恢复query dtype；gather处理PA-NZ逻辑顺序，此路径不再依ASCEND_USE_FIA或ATB。Gemma3另遵守默认关闭的native RMSNorm开关。已读npu attention与norm patch；最新59 CPU兼容测试用AST抽生产定义、FIA/merge由PyTorch reference替代，**不验证完整import或真实NPU kernel**，不能当A3验收。

FuseEP mode2已combine routed token，旧Qwen2Moe/Qwen3.5仍用TP-sharded shared expert并allreduce不同rank token行。现在shared expert设TP1、初始化EP metadata、走deepep式combine后加完整shared贡献。作者历史A3/Qwen3.5-35B-A3B W8A8 TP4对齐input-logprob MAE5.7167→.05182，最终删测试改动后未新跑NPU；未给吞吐。建议**按真实CANN/torch_npu镜像做端到端prefix与shared-expert数值回归**。

#### SGLang：省掉allreduce的DLPack重复包装，收益受host调度限制

稳定ID `github:sgl-project/sglang:pr:44251`；[#44251](https://github.com/sgl-project/sglang/pull/44251)，main，2026-10-17T10:44:31Z，merge `400c1dfe6eb42b2b3ff79c8a1f773227c84094ec`。

tvm-ffi启Torch DLPack exchange后已经返回torch.Tensor，再from_dlpack只是torch→DLPack→torch绕行；现在已是Tensor直接返回，否则保fallback。已读三行guard，返回别名与stream语义不变，graph/compiled路径不改。

作者2×GB300 Grace/torch2.13/tvm-ffi.1.11，eager TP allreduce host提交42→20.7µs；TP2 eager/QwenImage2.1 512²端到端1462.4→1298.3ms、FLUX1dev512²3794→3559ms，每10对交替、输出MD5同。SRT eager prefill适用但未测，不称LLM服务自动快11%。H200跨server有3.5%噪声，更应实测host-bound程度。建议**profile提交时间再决定优先级**。

#### SGLang：online NVFP4显式4over6保配方，尚无runtime parity

稳定ID `github:sgl-project/sglang:pr:41745`；[#41745](https://github.com/sgl-project/sglang/pull/41745)，main，2026-10-17T03:56:11Z，merge `29a7e19de10bf44d5f13e90df84345b0316358f2`。

将448/MSE/fast candidate-error配方交给FlashInfer NVFP44Over6Config，scale上界从同config导出，删loader四个全局环境覆盖；剩exact-FP4-math scope在quantizer用引用计数，首caller设、末caller恢复，异常也恢复，lock不串行quantize。已读online quantizer/loader patch，依赖已pin FlashInfer0.7.1，activation recipe不改。

作者明确**本次只做macOS静态检查，无GPU数值/运行parity与加载性能测量**。这项是配置/并发scope整理，不是新增量化精度或免费提速；建议**观察并补原模型加载及并发reload数值对照**。

<a name="torchtitan"></a>

### TorchTitan

#### TorchTitan：EMA checkpoint修全局元信息与加载后reseed

稳定ID `github:pytorch/torchtitan:pr:5357`；[#5357](https://github.com/pytorch/torchtitan/pull/5357)，main，2026-10-17T21:15:20Z，merge `dc9ea005e411dd9d264fb4b04feb75318948e38d`。

稳定ID `github:pytorch/torchtitan:pr:5141`；[#5141](https://github.com/pytorch/torchtitan/pull/5141)，main，2026-10-17T21:14:21Z，merge `ba53205255cc89e0479096cd061fccbb71e30112`。

offloaded EMA用local shard推global shape会把5×4两rank的3/2行误还原为6/4行； `_materialize_dtensor`显式传live parameter的shape/stride，支持uneven/empty tail/transposed等。作者CPU/Gloo2/4rank保存载入覆盖，host storage身份保持；pinned-memory、CUDA顺序与真实GPU续训未测。

reseed从DCP私有加载移到BaseCheckpointManager.load，让TorchCheckpointingManager也在model-only或exclude ema时从新权重重置shadow，完整resume保留恢复的EMA；TorchFT per-replica dataloader直接走私有入口不被误重置。已读两个实际patch与回归，作者CPU115pass、macOS缺O_DIRECT的两个HF用例同main失败。建议**优先回归seed模型加载、跳EMA resume与offload不齐shard**。

#### TorchTitan：RL vLLM替换attention不能丢YaRN scale

稳定ID `github:pytorch/torchtitan:pr:5347`；[#5347](https://github.com/pytorch/torchtitan/pull/5347)，main，2026-10-17T04:35:42Z，merge `8a3f5bcaa4fd51ea6af6fb4bbb90ef03bcb139d0`。

把model-specific softmax_scale提升到attention config属性，GPT-OSS/DeepSeek保YaRN magnitude，构造vLLM attention透传，每次传入scale若与构造不一致就ValueError。已读模型config、wrapper与forward校验patch；作者GPT-OSS20B KL mismatch .055→.0006，但正文未提供硬件/完整harness，**DeepSeek未测**。建议**优先复现trainer/rollout同token同scale对照**，不外推两模型数值已对齐。

#### TorchTitan：生成引擎与ActorSystem接口可替换，当前实现仍是vLLM+Monarch

稳定ID `github:pytorch/torchtitan:pr:5317`；[#5317](https://github.com/pytorch/torchtitan/pull/5317)，main，2026-10-17T23:48:30Z，merge `3c7c1479151a9ea7c9d5be9fd43a687390c5f1db`。

稳定ID `github:pytorch/torchtitan:pr:5391`；[#5391](https://github.com/pytorch/torchtitan/pull/5391)，main，2026-10-18T00:51:34Z，merge `f1bb643a71fbddda5fb94d2d3f62c6e9bca4cf70`。

Generator拆service、EngineLoop、dispatcher和GenerationEngine，vLLM调用集中到VLLMEngine，配置移入generator.engine；rank0汇FinishedRequest及policy版本，decide→broadcast→admit→step→fanin→load顺序保留。ActorSystem再接管provision/spawn、DP TrainerHandle、TorchStore初始化与shutdown；默认Monarch，RDMA微批通过read/drop协议注入，缺registration明确错误。已读接口、构造选择、engine-loop与actor生命周期关键patch，未逐行审36+25文件。

这为后续SGLang/Ray留接点，**本PR没有实现这两套backend**。作者GB300随机Qwen3 debug model、确定reward、六步on-policy/BI全部非计时metric逐bit同，DP2/RDMA也同；async完成六步后仍与main同一个teardown router-inflight错误，不能称完整异步生命周期无错。#5317 CPU492、#5391 CPU497pass各自基线；建议**深入阅读配置迁移并单测setup/teardown失败**。

#### TorchTitan：Ulysses旧NCCL graph注册与eval图共享池

稳定ID `github:pytorch/torchtitan:pr:5393`；[#5393](https://github.com/pytorch/torchtitan/pull/5393)，main，2026-10-17T23:16:31Z，merge `8899cee5bcf520cbeca62ec7b320af8d6b112428`。

稳定ID `github:pytorch/torchtitan:pr:5241`；[#5241](https://github.com/pytorch/torchtitan/pull/5241)，main，2026-10-17T15:05:23Z，merge `3c7b0c9d9441cf91cb0034a9223a147ed1e8b24b`。

普通Ulysses在NCCL<2.32.3可能复用不完整IPC mapping，现CUDA+真实NCCL且model遍历到Ulysses时在process-group初始化前设NCCL_GRAPH_REGISTER=0；fake非CUDA跳过，固定版本不改。已读 `configure_nccl_for_ulysses`与training/TorchFT入口。作者小规模复现，仍待KimiK3大规模验证；影响其他collective的推测未证实，故门只限Ulysses。上游fix指NCCL#2362，工具链版本需单核。

eval graph通过validator.enable_cuda_graphs opt-in，train/eval/optimizer共pool，先eval capture以让临时量复用grad storage，_CUDAGraphGradientState管理生命周期；eval前等checkpoint staging、PP metadata捕获外准备。作者MuseGlimmer30B train+eval峰61.531→61.530GiB是**不额外增显存**，不是降低2.5GiB（training-only59.024不同口径），正文未给硬件/速度，勿外推。建议**按配置启用并验stage/eval梯度存储恢复**。

#### TorchTitan：SFT去额外assistant header与双BOS，generation tags可解析

稳定ID `github:pytorch/torchtitan:pr:4993`；[#4993](https://github.com/pytorch/torchtitan/pull/4993)，main，2026-10-17T21:13:57Z，merge `ce7b0594f442c7faf80b7a838c164822a0ebf6e5`。

稳定ID `github:pytorch/torchtitan:pr:4992`；[#4992](https://github.com/pytorch/torchtitan/pull/4992)，main，2026-10-17T22:37:30Z，merge `f0444a79323bd37a75f2d88dd93640b988d63181`。

HFBackendTokenizer.apply_chat_template不再默认加generation prompt，encode发现文本已有BOS就不再添加，避免完整conversation监督中多assistant header及双BOS；外部inference caller依赖旧默认须显式true。generation/endgeneration标签转换为with/endwith并保空白控制/变量scope，范围只experimental Transformers backend，不新增HF assistant mask功能。已读tokenizer与回归patch。

作者macOS没有torch，以真实定义+stub harness比对Llama/Qwen/Seed/SmolLM/LFM/TRL模板；新CPU pytest仍待CI。两步A10G SFT未固定seed且还换CUDA nightly，loss差不能归因此改动。另#5471下载tokenizer assets补chat_template.jinja（正文已读、patch未深审）。建议**先检查实际labels/BOS与模板字节，再比收敛**。

<a name="backfills"></a>

### 旧待办补录与未合入提案

#### 旧待办补录：MTP每深度分母与runtime CP MoE统计

稳定ID `github:pytorch/torchtitan:pr:5055`；[#5055](https://github.com/pytorch/torchtitan/pull/5055)，main，2026-10-06T18:34:43Z，merge `ed35bdaede8b2723f43ebfe1f91a6de95a20eb74`。

稳定ID `github:NVIDIA/Megatron-LM:pr:7046`；[#7046](https://github.com/NVIDIA/Megatron-LM/pull/7046)，main，2026-10-08T05:49:07Z，merge `159b443c3e1a23ff68321973b936fef810bb3c61`。

这两项是此前候选库存的语义补审，**不是今日新合入**。#5055于10-06合main：data pipeline按packed document边界/padding逐深度生成loss_token_counts与routing_token_counts，普通trainer累加归约，RL batcher提供global分母；每MTP objective及MoE aux用其实际适用token，避免深层有效位变少仍除depth0。已读 `get_mtp_token_counts/roll_mtp_sequence`、aux denominator传递patch；作者4×H100/FSDP及TP+CP各十步和431focused CPU tests，非完整长训认证。

#7046的default-branch commit时间为10-07，但**merged_at是10-08T05:49:07Z**，不沿库存日期当merge。PackedSeqParams贯通MoELayer/Router，runtime CP1只TP reduce，CP>1按runtimeCP→TP；in-place allreduce前clone本地token计数，保持global aux scale与DP metrics各自口径。z-loss/graph不在范围。作者早期4/8×H100含CP1→2→4→1、uneven/empty rank，最新rebase patch-identical仅CPU11pass/5skip，GPU表不是新SHA再跑。建议**深入复现不齐packing与拓扑切换的梯度规模**。

#### 旧待办补录：hybrid Mamba前缀命中的block单位错位

稳定ID `github:vllm-project/vllm:pr:55601`；[#55601](https://github.com/vllm-project/vllm/pull/55601)，main，2026-10-07T17:09:51Z，merge `3d6853b32c7c872df00d9e3489f642dc22e974e0`。

10-07已合main、本轮补审。`MambaHybridModelState.add_request`原用cache_config.block_size算running-state列，但它会被更小的prefix-cacheable drafter attention组拉到64/1024，Mamba实际仍7168；短命中可静默还原错误状态，长命中可越block-table。现在除mamba_block_size并assert存在；已读真实五行修复及新增regression文件。正文旧称无unit test，files现含测试，按patch为准。

作者DGX Spark/GLM5.3Flash EXL3 TP1/DFlash2 K5/cache on，57344token八块命中旧Xid31/500，新完成且50176cached；133请求三小时无fault。warm/cold TTFT不同缓存工作量，**不把8.7 vs139s作本修复普遍速度倍率**。建议**优先核对hybrid多组block几何与warm/cold分布**。

#### 旧P2P寿命待办：consumer quarantine已有open实现，尚不能关闭风险

[#61532](https://github.com/vllm-project/vllm/issues/61532)仍open，最后issue更新时间2026-10-14T10:08:46Z；本轮冻结main `9c778e9ff8d0c78a241d7f530c441c2dc1d67533`重新读client/server，producer已持pins/AbortAck等terminal（先前#59790），但consumer仍在10s abort-ack timeout后报告failed。lookup超时/控制面线程及MRV2提前收结果都不是远端WRITE停止证明。

新补审候选 [#61667](https://github.com/vllm-project/vllm/pull/61667) **open/base main/merged_at=null**，head `4fd16ddb155fd0f7e676dece9abd6aa9209b2b1a`，不是发布。引入memory_safe=False将失败通知与可释放权限分开：失败key转miss而CPU槽保持quarantine，late terminal确认才释放且不重复成功计数；close若再收不到确认就保留隔离，reset拒绝未settle quarantine。已读PR与client/tiering/scheduler关键patch；另防延迟handshake中retired Fetch在ID重用后发出、失败reset破坏request账本。

作者WSL2/RTX4060Ti8GB、NIXL1.4.1/UCX-TCP单host：真实1GiB WRITE于PROC时停止目标/producer，timeout后槽未复用，late ack才可释；close保持隔离。1289回归与tiny dummy Llama生成/reset检查不能代替RDMA、多节点服务压力或预训练精度，CI尚因新贡献者策略受阻。**待办实质推进为“候选实现+真实单机transport探针已读，等待合入及RDMA/E2E验证”**，不能标已解决。

<a name="extended"></a>

### 扩展项目

#### PyTorch cuDNN varlen SDPA：反向暂存按真实 token 数分配（10-14 补录）

**项目/版本。** PyTorch main [commit `6b25e6c6fb0f`](https://github.com/pytorch/pytorch/commit/6b25e6c6fb0fb234fe49335ee55131840897135f)，10-14T01:46:04Z；对应 [#200847](https://github.com/pytorch/pytorch/pull/200847)。PR 页面显示 closed/`merged=false`，实际默认分支 commit 已存在；按该 commit 记录能力，不能仅靠 PR 的 merge 字段判断 ghstack 落地。

**变化。** nested/varlen cuDNN SDPA backward 原先未设置 `max_total_seq_len_q/kv`，frontend 把 fp32 dQ 累积和 softmax 暂存按 batch×最长序列分配。文档长度偏斜时，packed 输入没有节省这部分内存。`aten/src/ATen/native/cudnn/MHA.cpp::build_graph_backward_nestedtensor` 现在传入 `q.size(0)`、`k.size(0)`，使这部分 workspace 按总 token 数配置；forward 不变。

**证据/边界。** 作者同一 B200/SM100、CUDA13.0、cuDNN9.20、frontend1.30、BF16、T=16384、H32/D128，比较同一构建撤销两行修改前后：递减文档长度、128篇时，fwd+bwd 峰值（输入之外）**17156→902 MiB**，512篇 **66692→902 MiB**；等长文档不变。这是特定反向暂存的修复，不能写成任意 LLM 训练显存降到 1/19。dK/dV bitwise 相同，dQ 仍有原有 atomic 非确定性；作者报告 sanitizer 0 error。cuDNN<9.6、head dim 非16倍数、SM80/SM120 等不支持条件由 frontend 警告并忽略该属性；短序列≤128在此测试版本还有既有限制。**建议考虑复现/采用：长短文档混合 packing 且使用 cuDNN backward 的训练最直接受益，先确认底层版本确实接受属性。**

稳定ID：`pytorch/pytorch@6b25e6c6fb0fb234fe49335ee55131840897135f`。

#### PyTorch profiler：修复元数据占位 tensor 造成的后续事件错配

**项目/版本。** main [commit `09673565957c`](https://github.com/pytorch/pytorch/commit/09673565957c49fb174ece4a84187cd98f9d5962)，10-17T06:08:13Z；[#200964](https://github.com/pytorch/pytorch/pull/200964)。

`record_shapes=True` 会取输入 storage 地址作为身份键。被 rematerialization 或 cudagraph trees 标为不可取 data pointer 的占位 storage 仍可合法执行 view，但旧 profiler 的取址异常发生在“已追加 op、尚未追加 inputs/collective metadata”之间；异常只记日志，之后同线程事件与元数据整体错位。`torch/csrc/profiler/collection.cpp::storageData` 先查 `throw_on_immutable_data_ptr`，对这类 storage 记 null，保留 shape/dtype，避免触发此异常；并未放开真实数据访问。

作者在 TorchTitan Llama3、FSDP2×TP2、activation checkpointing、eager Kineto 场景报告：原来224个 `record_param_comms` 中95个丢失正确 collective metadata，122个 NCCL kernel 中54个无进程组归属；修复后122个均归属，NCCL/NCCL2均测。CPU回归覆盖 poisoned tensor 后另一个 add 的输入 shape。**尚未把所有 profiler 记录操作改成原子提交，其他异常仍可能触发位置错配**；没有吞吐收益测量。**建议采用并重采相关 trace：以往看到的通信归属缺失可能来自采集错误，不能直接据旧 trace 判定无通信。**

稳定ID：`pytorch/pytorch@09673565957c49fb174ece4a84187cd98f9d5962`。

#### PyTorch precompile：允许显式补编译，同时修复捕获和安装边界

**项目/版本。** 10-17 main 四个独立 commit：[serve-time 编译 `80c4d80b86cf`](https://github.com/pytorch/pytorch/commit/80c4d80b86cf67ec7a94244810551409f0e71634)、[无图子帧 `a36677e34575`](https://github.com/pytorch/pytorch/commit/a36677e345753dfafe2e9cd4859d02ef6831c768)、[bundled cache `716a92192510`](https://github.com/pytorch/pytorch/commit/716a9219251038a8c391c170f1f84cf849766609)、[预先使用的 Triton kernel `a96720f45ee1`](https://github.com/pytorch/pytorch/commit/a96720f45ee1388e932c1765c366d278b63c3519)。均按实际 main commit 核查，尚非新稳定版发布。

- `torch.compiler.precompile.load(..., allow_serve_time_compiles=True)` 让 **installed** artifact 先复用捕获变体，未覆盖的 entry 或 graph-break 子帧再按原 backend 编译，适合无法容忍漏捕一批就中断的长训练。默认仍拒绝补编译；standalone、旧版无开关 artifact 会报错，`compiled_autograd` 限制仍在，`fail_on_recompile` 等全局 stance 仍生效。补编译不写回捕获包，变体用完 recompile limit 后走 eager。
- 无 graph、无 continuation、且不是 entry 的纯 Python 子帧若因 weakref guard 无法保存，改为递归 eager 执行；真的产生过计算图的帧仍保留原来的完整性门槛，不能概括为“忽略不支持的 guard”。
- 捕获前后向以及安装同帧多 backend 时关闭 `bundled_autotune_remote_cache`，绕过“同一 compile context 只能开一个 bundle”的断言；单 kernel 的 local/remote cache 不受此改动影响。该限制本身未根治，捕获/安装的首次 autotune 不再成 bundle 上传；加入忽略项还使既有 FX graph cache key 一次性失效。
- 对捕获块之前已用过、且块内再次执行的 Triton JIT kernel/autotuner，先清内存缓存，使它们在块内重编译/重调优后入包；新调优可能选不同配置并在块后继续使用。未在块内执行的旧specialization会恢复；不剪枝且带pre_hook的autotuner仍不携带磁盘决策，服务端首次使用还要调优。开关捕获块时其他线程不能并发launch Triton；块内其他线程产生的kernel也可能入包。

依据是 `torch/_precompile.py::load`、`torch/_precompile_driver.py::_build_installed_forward`、Dynamo package/session 与新增回归；没有本轮独立跨进程/集群复现，也没有通用训练提速数字。**建议深入阅读并小规模验证：先决定服务是否允许编译、准备工件是否需要重捕，再验证未覆盖 shape、graph break 和冷缓存路径。**

稳定ID：上述四个完整 commit；对应PR `#200835/#200836/#200832/#201003`。

#### PyTorch FlexGEMM：把 MoE GLU 反向两路梯度和路由归约接入 epilogue

**项目/版本。** 10-17 main [interleaved `bae3ba19f25f`](https://github.com/pytorch/pytorch/commit/bae3ba19f25f50366a48f73c5baef0d253ac7510)、[chunked `d862fcbaaf4e`](https://github.com/pytorch/pytorch/commit/d862fcbaaf4e5843c52b93fee0464b8a934c1aa1)、[组合/offset `31ba9a56657b`](https://github.com/pytorch/pytorch/commit/31ba9a56657bc887c95792dfe270706142d44f1a)、[TMA capture `7eee4754295f`](https://github.com/pytorch/pytorch/commit/7eee4754295fbdd10823f3fac40912458133e57d)。

GLU backward 的 `dy @ w2[e]` 结果是 M×N，但后续两路权重梯度需要 M×2N 的 gate/up 梯度。新 epilogue 支持两路交错写出和 `cat` 成两段的写出，能在同一次 grouped GEMM 内读保存的两路 pre-activation、算 VJP、写梯度；组合改动还容纳保存 accumulator、局部归约与捕获输入。`quack_ops/grouped_reduce.py::_store_groups` 补 `cu_seqlens_m[expert]`，避免 varlen-M 不同专家都写相对第0行。

边界仍明确：varlen-M 只开放**沿 N、只存出的** grouped reduction；跨专家 M 归约、把归约回馈主输出等不支持组合仍拒绝。chunked 两半要求16字节对齐；不满足 TMA stride 条件的 capture 会复制到连续对齐布局，有额外内存/复制成本，能直接读的 row-strided view 保持原位。作者给出布局/激活/版本12组输出及四类梯度检查；性能图片不足以建立可复用的 LLM 端到端倍率，本报不取图中数字。**建议深入阅读/考虑复现：这是 MoE backward 融合的实际接口增量，验证时必须覆盖不均衡专家长度和两种 gate/up 布局。**

稳定ID：上述四个 main commit，对应 `#201087/#201199/#201127/#201086`。

#### PyTorch 两项数值/性能落地：XPU 的 E8M0 block scale，超大短归约退出 combo

**项目/版本。** 10-17 main [XPU `ec91f0c15cbb`](https://github.com/pytorch/pytorch/commit/ec91f0c15cbbd64cc87db07195e99c0bb05b4b76)（[#195565](https://github.com/pytorch/pytorch/pull/195565)）与 [Inductor `64373c6e0d22`](https://github.com/pytorch/pytorch/commit/64373c6e0d225f9809e0f9cb98741a77cf67603d)（[#200700](https://github.com/pytorch/pytorch/pull/200700)）。

XPU `_scaled_mm` 的 DeepSeek-style 1×128/128×128 block scaling 接受 E8M0 scale，并把类型传给 oneDNN `e8m0`；meta 校验也区分该格式与 MXFP8 1×32。两侧 block scale 要同 dtype。它是指定数值格式在 XPU backend 的接通，未提供模型级质量/吞吐证据，不能写所有 XPU 都获得原生 MXFP8 训练提速。**建议相关 Intel GPU 实践者核查设备与 oneDNN 支持后复现。**

combo kernel 的分区规则原先仅把 rnumel>2048 的大归约分出，超大而短的 reduction 会继承小子算子的差配置。现在在不开 `benchmark_combo_kernel` 时，只要总元素超过 `LARGE_NUMELS` 就单列；开 benchmark 时保留旧长归约限制。作者 H100 示例472MB、737280×112的 persistent reduction 单跑0.2ms，混入128×768后1.2ms；生产模型曾有18.7%回退，但模型、精度、完整负载未披露，不能当普遍提速。**建议遇到 combo 回退时对照分区，而非一律融合更多 kernel。**

稳定ID：`pytorch/pytorch@ec91f0c15cbbd64cc87db07195e99c0bb05b4b76`、`@64373c6e0d225f9809e0f9cb98741a77cf67603d`。

#### TensorRT-LLM Kimi K3/Rubin：CuTe量化、KDA重放与路由协同

**项目/版本。** [#19960](https://github.com/NVIDIA/TensorRT-LLM/pull/19960) 于10-17T09:51:38Z 合入 main，commit [`e88d19ca3bda`](https://github.com/NVIDIA/TensorRT-LLM/commit/e88d19ca3bda5bb761cf582e0379ed5397a7e60b)。PR顶部 bot 还描述“三文件import格式调整”，与最终68文件diff不符；这里依据最终patch。

SM107 的 FP8 weight-read linear 改用 CuTe DSL MXFP8 GEMM 的 K32/R128c4 E8M0 scale 布局，SM100/103 保留 DeepGEMM；增加预量化输入入口，使满足条件的 KDA verify 输出可把融合归一化/门控后的 FP8 activation 与 scale 直接传入 o_proj。CuTe FP8 MLA 改由设备tensor读 BMM scale，去掉逐层 `.item()` host读取。配套 SiTU FC12 epilogue、KDA tensor-core verify/replay、路由与 latent-down overlap，以及共享专家 all-reduce 顺序修复，减少中间转换并补启动/挂起恢复路径。

**采用边界。** KDA 的 BF16 tensor-core operand 是 `TRTLLM_KDA_MTP_BF16_MATH=1` opt-in；FP4/NVFP4 KV MLA **不在此PR**，混合线性注意力/PD路径仍拒绝。drafter独立KV池改32-token page，不再随target的128-token page误配；`TLLM_DRAFT_KV_GPU_BUDGET_GIB` 也是默认未启用的预算上限，保留allocator floor。作者称drafter区域约36%更快和DEP8 context rank加载峰值约少22.6GiB，缺少完整同条件端到端A/B，不能推广为整机/整模型倍率。最终新增测试覆盖backend、路由、replay、量化加载等，但本报未运行Rubin/GPU验证。**建议深入阅读并按实际SM/backend逐项验证，尤其比较恢复后状态和最终输出；不把大PR整体当无条件可用开关。**

稳定ID：`NVIDIA/TensorRT-LLM#19960@e88d19ca3bda5bb761cf582e0379ed5397a7e60b`。

#### TensorRT-LLM 流式输出与镜像回归：末尾 UTF-8 flush、固定 compat driver

**项目/版本。** main [#20424](https://github.com/NVIDIA/TensorRT-LLM/pull/20424)（10-17T12:45:57Z，`2f5bb049f33f68e36d4c4694318d22f021bc10e2`）与 [#20252](https://github.com/NVIDIA/TensorRT-LLM/pull/20252)（10-17T21:15:06Z，`8b2ac2d08ec53e4b61104c74bb02ce30ef0cfa93`）。

HF `DecodeStream` 遇到不能独立解码的UTF-8片段会暂不产文本，而旧路径没传请求结束的 `flush`，导致最后片段消失，甚至单token响应只有metadata/DONE。新路径记住pending ids和最后已解码id，最终按full-decode补尾；作者3000组DSv4随机流、chunk1–20与full decode一致。生产线索是GB300、单token/warmup/ctx-only、10695请求中15个空响应；**正常跳过EOS形成的空文本是另一个问题**，测试端同时调整特殊token处理，不能把所有空响应归因于同一bug。建议有流式文本一致性要求的部署核查此修复。

26.09容器把CUDA forward-compat用户态 `libcuda` 从615.65.02换成615.71.09，作者GB200的DS-R1 FP4、128K/8K、con1、PP8/TEP8、MTP3、ctx-only场景出现约6% total-token throughput回退；同cubin FMHA每调用慢约7–13%。控制实验只把compat目录换回后，吞吐从约5334–5352恢复约5703–5704 tok/s，NCCL更换未复现差异。修改保留26.09/CUDA13.4 toolkit，从26.08镜像拷compat库；脚本要求相同CUDA major.minor，非NGC目标默认不替换。**这是特定容器/driver/算子组合的定位，不是CUDA13.4或新硬件普遍变慢；建议镜像升级A/B记录进程实际映射的libcuda，而不只看宿主driver版本。**

稳定ID：`NVIDIA/TensorRT-LLM#20424@2f5bb049f33f68e36d4c4694318d22f021bc10e2`、`#20252@8b2ac2d08ec53e4b61104c74bb02ce30ef0cfa93`。

#### TensorRT-LLM 补录：KV路由事件、LiLiCorr drafter、LoRA失败保留缓存

**10-14合入main的三项漏收。** [#19440](https://github.com/NVIDIA/TensorRT-LLM/pull/19440)，`2442169f07a1313d1316c2e0ac4db8817689ac41`；[#19856](https://github.com/NVIDIA/TensorRT-LLM/pull/19856)，`39ea47f9715f055504b13d3a1eb587bdf62af0be`；[#20212](https://github.com/NVIDIA/TensorRT-LLM/pull/20212)，`51b993b814ec5954bd6175190e3bd2a8d0ebd752`。

OpenEngine暴露原生ZMQ KV事件和逐DP rank负载，并校验Dynamo传入的目标rank，部署按节点设sidecar；把prefix locality从引擎内部接到外部路由。作者旧双节点B200测试证明过8rank发现和预热前缀选rank0/4，但早于最新discovery RPC重构；最终六节点smoke、多frontend/签名affinity等仍无完成结果。**建议观察并集成验证，不能称最终head的跨节点路由性能已证实。**

LiLiCorr checkpoint可通过 `DFlashDecodingConfig` 选择相关候选评分drafter，支持量化元数据和checkpoint自带head，context buffer按activation dtype分配。需要非causal attention、共享词表、非空 `target_layer_ids`、draft长度小于block_size；不兼容词表/Laguna组合会拒绝。硬件无关评分与TP归一化测试、H100/B200量化加载测试有代码/CI注册，不能等同已完成全模型质量或吞吐测量。**建议有匹配drafter工件时小规模复现。**

LoRA `claimPagesWithEvict` 在可逐出页总量仍不足时先失败，避免“清掉已有adapter后仍分配失败”；`put`用 `throw;` 保留原异常类型。单测检查失败不逐出、清理失败task、同ID重试和恰好够页的边界，未提供硬件性能结果。**建议多adapter负载采用，重点验证cache-full重试语义。**

**旧待办分支界限。** [#19868](https://github.com/NVIDIA/TensorRT-LLM/pull/19868)（10-14T04:52:49Z）合入的是 **`feat/m3_with_msa`**，commit `1759a47318dc4b15bbca13e601abf835237834b2`，与本窗main比较仍diverged。其Mooncake prefix失败重算仅允许整页由单rank持有（attention-DP/单rank），TP失页仍fatal；后台async loader为可选。它有实现和测试，**仍不能算main已发布能力**；本轮关闭该候选的“尚未审读”待办，保留分支采用验证问题。

#### vLLM-Ascend：C8 K-scale定点写入、layerwise快照与跨rank发布 fence

**项目/版本。** 10-17 main [#19064](https://github.com/vllm-project/vllm-ascend/pull/19064)，commit `e6c0f4b7d2e2be253ecfc0d0d34f4864ff7fe107`；[#19039](https://github.com/vllm-project/vllm-ascend/pull/19039)，`8e88ec3fdf34c1a7e25f8e0486598647497bf69b`。

C8 K-scale的高级索引赋值会把packet-strided cache view物化后整段写回，覆盖另一个并发KV传输正在写的**未选中页**，损坏E8M0 scale，表现为NaN/重复感叹号。新Triton scatter按实际六维stride只存目标字节；空批次不触碰cache，token元素数作为runtime scalar避免按token数量重复编译。它不计算/校验scale，也不解决对同一字节的竞争或reader等待问题。

作者950DT的64项NPU测试涵盖offset、padded stride、graph replay与40种并发SDMA交错，另50项UT；kernel核验hash `da8c6ffac57dd3d7292808f38022c3cf21a39364c2823849fdd9af12eb4edd41`。旧35B服务A/B/A损坏27/512→0/512→3/512、之后2048请求无复现，但测试在0.30兼容依赖栈，**不证明最终整个新分支依赖组合已完成集成，更不覆盖独立的397B V-scale生命周期问题[#19065](https://github.com/vllm-project/vllm-ascend/issues/19065)**。

layerwise GVA+reuse则分开“请求上一步实际保存的快照身份”和投机拒绝后回退的logical prefix；没有scheduler hash的完整block先放请求快照namespace，之后再提升为full hash。worker在drain deferred save之后、准备下一轮load之前增加跨rank发布barrier；仅在layerwise GVA、reuse、TP>1条件启用，不能由rank本地请求数决定是否参与。作者当前head主要为12项CPU逻辑测试（含25subtests），早期A3/TP16/GLM5.2硬件检查不是最终rebased head验证；新barrier可能增加等待，没有零性能损失保证。**建议优先核相关正确性修复，并分别验证K/V scale、rollback和远端发布完成，不能以单个scatter修复关闭整个KV生命周期问题。**

稳定ID：`vllm-project/vllm-ascend#19064@e6c0f4b7d2e2be253ecfc0d0d34f4864ff7fe107`、`#19039@8e88ec3fdf34c1a7e25f8e0486598647497bf69b`。

#### vLLM-Ascend：Kimi draft/head与MLA布局修复，FIA v2迁移依赖CANN能力

**项目/版本。** 10-17 main [#18845](https://github.com/vllm-project/vllm-ascend/pull/18845)（`33dd6e797de414db6981ec4afb09ab71baaddd60`）、[#18752](https://github.com/vllm-project/vllm-ascend/pull/18752)（`7c0c846c1640e457ea01e0d72a8424f4842044a5`）、[#19061](https://github.com/vllm-project/vllm-ascend/pull/19061)（`93ebeef5f270a472b65d8a0680598f72d7e75e68`）。

宽Eagle3 draft的两个7168维输入拼成14336，CANN PROLOG_V3不接受：eligible BF16 draft保留原strided KV，用既有RMSNorm/RoPE writer和scatter，加载时准备有界workspace供graph复用；受支持宽度继续PROLOG。A5 FP8 MLA请求native TND输出，在多token、无head padding、`1<=heads×kv_lora_rank<65536`时才走fused V projection；单token或其他布局回退，不能把三条路径混成统一融合。

DCP MLA的eager/capture/update/workspace共八处改走FIA v2，使支持dim0 stride的CANN可直接消费component-major cache，避免隐式contiguous大复制。**避免复制依赖CANN运行时能力**（作者核对9.2.0-beta2源码），不是Python调用更名就自动获得；没有实测提速数字。

K3 DSpark加载器在继承mapper丢弃 `lm_head` 前截取checkpoint自带head，校验shape、dtype/device并成功加载后才设 `has_own_lm_head`，防止target共享/QuaRot对齐再覆盖或旋转它；无head工件保持fallback，重复/错shape报错。新增22项CPU加载测试，一部分依赖stub，不能当NPU TP/PP全链路验证。**建议有上述模型/模式的部署采用并复核真实head、graph布局和CANN版本；本报没有硬件复现。**

#### vLLM-Ascend：主分支兑现SiTU回滚；Mooncake关闭线程与完成通知状态

**项目/版本。** [#18902](https://github.com/vllm-project/vllm-ascend/pull/18902) 10-17T11:11:58Z合入main（`29092af7d8a75b2b57685337a649ee693ea91b59`），是10-17日报仍open条目的**状态增量**；[#19075](https://github.com/vllm-project/vllm-ascend/pull/19075)（`cccfd2338e4249b717c3f5c1bbee10a9b664566f`）；[#18956](https://github.com/vllm-project/vllm-ascend/pull/18956)（`22730f46da2d821dc3955f47d56d16284eb2de81`）。

A3 W4A8 SiTU自动fused GMM出现性能回退后撤回原#16655，恢复 `npu_grouped_matmul` + `dequant_situ_quant`，移除对应新增op/注册/dispatch；其他GMM2及旧融合op不变。回退幅度和瓶颈仍未确定，需重建extension后对照，不能称精确恢复某倍率。KDA direct compiled launch则撤回三个constexpr实参，匹配当前Triton仅九个runtime参数的签名，修复18参/21参TypeError；作者八台950DT、4P+4D、CANN9.2.0、torch-npu2.10.0.post4样例成功，不能代表所有依赖版本。

Mooncake V2从永久累积completed-ID set、失败DONE无限回队，改为用pending delayed-free去重，未知/晚到DONE仍ACK；通知重试由有界socket机制承担，失败依赖producer delayed-free timeout。shutdown唤醒/join控制线程，并在拥有线程关闭ZMQ、清空请求状态。**它处理的是控制面清理，不证明GPU/RDMA WRITE已停止，也不关闭已有P2P生命周期待办**；作者未附性能benchmark。建议长期运行/频繁engine重启的PD服务核查。

#### DeepSpeed与Transformers补录：BF16溢出开关、static cache和导出量化

**DeepSpeed。** [#8818](https://github.com/deepspeedai/DeepSpeed/pull/8818) 10-16T01:27:35Z合入master，默认分支commit `bb3f873e9d`（本轮补录）。普通ZeRO1/2与ZenFlow step现在只在BF16且显式 `check_grad_overflow=false` 时跳过post-norm invalid检查；FP32/FP16仍拒绝invalid norm，修复了原先“BF16开关无效”和粗略按flag判断会放过FP32这两个边界。作者两H200/NCCL、PyTorch2.11+cu130的12配置/36step每rank、CPU offload warmup等通过；ZenFlow selective/top-k、GPU offload、ZeRO3与收敛未覆盖。**建议已有显式BF16 opt-out配置的作业复核分支语义，不从此推导一般提速。**

**Transformers static cache。** [#49317](https://github.com/huggingface/transformers/pull/49317) 10-14T13:36:51Z合入main（默认分支 `245167e066`，补录）。把nested dict/list/tuple的输入tensor递归移到model device；提前建static KV时只在TP plan的key projection确为 `colwise` 分片时除head数，空plan、gathered或仅experts分片保留原head数。**当前patch仅识别 `colwise`，正文提到 `colwise_units` 尚需跟进**，不能写覆盖所有TP计划。作者含Neuron/Qwen3-VL-8B TP2检查和回归测试，无跨模型吞吐结论。

**Transformers导出量化。** [#47747](https://github.com/huggingface/transformers/pull/47747) 10-14T10:47:32Z合入main（`d15ebba04d`，补录）。统一export config的 `quantizer`：PT2E/NNCF在`torch.export` FX图上、转换前量化；ONNX Runtime/NNCF也可在后端模型上量化。FX quantizer返回GraphModule后重新export为ExportedProgram；OpenVINO量化放在state folding之前，让校准样本的cache仍作为输入。带校准输入捕获与依赖检查，不是每种backend、每种量化都互通，更没有全模型无损/性能结果。**建议部署工具链实践者深入阅读并对目标model/backend校准验证；不可把接口接通写成通用量化精度保证。**

<a name="hardware"></a>

## 硬件与软件协同

### Meta × AMD：生产DLRM达到B200相近吞吐，1.5x来自AMD移植基线上的整栈协同

**Meta / AMD；2610.22230v1，2026-10-15T17:09:08Z；旧缺口补录。** [论文全文](https://arxiv.org/html/2610.22230v1)。工业系统报告；声明贡献进入PyTorch/Triton/ROCm，本轮未逐项追溯对应upstream commit，不能把报告中的每项优化标成已核发布版本。

压力来自推荐模型大量不规则GEMM、embedding通信、jagged attention及反向kernel：CUDA→HIP能解决功能移植，却保留了32线程warp、NVIDIA缓存/流水深度与atomic设计假设。§III–V把模型、内核、通信和硬件资源一起调：Origami用分层缓存/计算代价与Stream-K尾部模型选GEMM配置，在生产shape上达到穷举调参平均93%选择效率；RCCL按collective分别选QP数，LL64贴合64B缓存行；Triton调PID映射、occupancy与较浅软件流水，GQA反向按KV-head组局部归约，减少跨XCD atomic竞争。后一JFA案例32→14ms，报告模型吞吐+25%，仅属于这个case；agent工作流“>80%成功率”未在本篇给完整任务分母与独立对照，暂作作者经验报告。

**§VI/Table III：10个生产DLRM，8–128 GPU，同GPU数、同batch，MI350X/B200吞吐比82–130%，几何均值101.2%。** 这里总参数含稀疏embedding，不能把3.6T DLRM解释为3.6T稠密LLM。§II-B/Table I明确：MI350X为288GB HBM、1000W，Meta的B200为180GB、750W，同为空冷；并非标准HGX B200 1000W，也不是等功率对比。B200仍占性能/瓦优势，本文不足以独立推导采购价格或普遍TCO结论。TF32依赖重的Model A落后18%，transformer Inductor kernel仍约慢30%，不能称全面超过NVIDIA。

§V-A/Fig.15的**1.5x分母是最初直接移植到MI350X的baseline**：增batch、去activation checkpoint、改embedding placement合计1.26x，再叠kernel优化到1.5x；不是MI350X相对B200普遍快50%。与LLM更直接相关的控制例是GPT-2 1.3B：MI350X可完全取消AC而96.4 examples/s，B200需AC预算≤.35而71.5；收益来自内存可行性差异。另1K GPU广告模型HSDP报告+6%且多750MB/GPU；本轮无法据此核长期收敛、精确成本及全部软件版本。

**意义/建议深读**：先按实际shape、dtype、通信重叠、显存约束和部署功率重建性能分母，再评估硬件替换。推荐优先读§II-B、III-B、IV-C、V-A、VI；不要把单kernel比值、kernel耗时求和或扩batch后的吞吐直接当同配置端到端收益。稳定ID：`arxiv:2610.22230v1`。

### ETH：跨pipeline分配不同数据量，比只按层数平衡更细；公开实现范围需与Ascend实验分开

**University of Macau / Great Bay University / Sun Yat-sen University / independent researchers；2610.22035v1，2026-10-15T15:10:27Z；旧缺口补录。** [论文](https://arxiv.org/html/2610.22035v1)；[代码基线58a11e5](https://github.com/YuMJie/ETH/tree/58a11e5b1e12b2febb12b3ff12f397f6968fcede)。

异构设备只按层数划分会遇到显存与整数层粒度限制。§3–5令不同pipeline使用不同local batch，pipeline内部再选PP/TP/DP、层数和算子级重计算；先优化stage、分配local batch、再refine，属于启发式搜索，**没有全局最优保证**。不同pipeline的TP切片让gradient地址和长度不同，block-wise AllReduce先按最大TP粒度切分/对齐，再融合相同通信组，避免全设备padding和逐tensor大量小collective。静态核`eth/searching/search.py::workload_balance_pipeline_v1`确实按下一个microbatch的预测完成时间分配local batch；`grad_sync/bucket_plan.py::_bucketize`按通信组/dtype/归约语义融合，`GradientBucketExecutor.execute`pack→AllReduce→按scale归一化→unpack，与论文两处机制相对应；未执行分布式正确性测试。

§7/Figs.9–13、Tables1–2：Ascend910B2/B3/B4分别测试，以及A100/RTX3090异构GPU；**不是Ascend与NVIDIA在同一任务混合**。GPT-3 350M–175B与LLaMA2-70B，主设seq2048/global batch1024，模型对应4–1024设备。Ascend相对HexiScale 1.36–2.00x、相对Metis1.20–1.84x迭代加速；NVIDIA350M–13B分别1.67–1.95x/1.41–1.73x。搜索时间与训练迭代是不同指标：175B对Metis搜索38.22–42.99x，而非模型训练快40x。13B global batch4时只1.02–1.10x，数据量更大才有平衡粒度；TypeSplit-HPP在两模型已相当，另三模型才有1.17–1.36x收益。

**限制/建议复现**：当前搜索模型限PP/TP/DP与1F1B，未覆盖CP/EP；pruning可错过高吞吐配置；70B Enwiki2000步只给接近loss曲线（约.1%差），不足以证明最终能力或任意不等batch下完全等价。公开README支持Ubuntu22.04/Python3.10/CUDA12.4/PyTorch2.6，当前发布路径是GPT CUDA环境；不能据论文Ascend实验宣称现仓已支持NPU一键复现。优先核全局样本加权、梯度同步与搜索含profiling的摊销成本。稳定ID：`arxiv:2610.22035v1`。

### Apple MLX：四台Mac运行816B MoE的RDMA/并行实测（10-15补录）

**机构/来源。** Apple [10-15官方实测](https://machinelearning.apple.com/research/mlx-distributed-inference)，MLX LM [v0.32.0](https://github.com/ml-explore/mlx-lm/releases/tag/v0.32.0)于10-15T18:30:46Z发布、[#1976](https://github.com/ml-explore/mlx-lm/pull/1976)支持Muse Spark Open；本轮补录，非10-18硬件首发。macOS26.2的Thunderbolt5 RDMA与MLX≥0.30.1 `jaccl` 是前提，实验还启用 `MLX_METAL_FAST_SYNCH=1`，为实验性快速CPU/GPU同步。

Muse816B/A42B用NVFP4，约493GB权重加中间态后单台512GB仍装不下。厂商在**batch1、8192输入、128生成token**测两台M5 Ultra pipeline→两台TP→四台TP：decode **15.74→24.99→36.73 tok/s**，TTFT **8.4→8.2→4.8s**；比较基线是两台pipeline，不能把它写成相对单台的线性加速。65,536上下文，四台TP每对双线比单线TTFT **50.5→40.6s**；双机TP在该长prefill反而可慢于pipeline。chunk8192的prefill pipeline可重叠两机工作，逐token decode仍按层串行。

同文的“最高4.6×TTFT”来自能单机容纳的dense模型跨M3/M5比较，不适用于单台装不下的Muse。文章未给完整benchmark commit、重复次数/置信区间或并发服务SLO；没有量化质量对照、整群成本或独立复现。**建议已有多Mac环境考虑复现：主要价值是本地大权重可运行性和按prefill/decode选PP/TP的实测边界，不据此外推数据中心性价比。**

### MLX v0.32.4：SDPA/Gated Delta反向kernel与发布后正确性修复

**项目/版本。** [MLX v0.32.4](https://github.com/ml-explore/mlx/releases/tag/v0.32.4)发布10-14T23:37:14Z，补录；其中 [SDPA VJP #4563](https://github.com/ml-explore/mlx/pull/4563)、[Gated Delta VJP #4565](https://github.com/ml-explore/mlx/pull/4565)原始合入日均为9-29，不冒称本周首次算法提出。

SDPA反向fallback物化context²矩阵；Metal实现用分块matmul/steel组合减少中间矩阵。Gated Delta增加sequential Metal与chunked Neural Accelerator反向路径，使线性注意力微调可以使用专门kernel。作者benchmark是M5 Ultra、batch1/4或1/4/8的算子形状扫描，不等于完整训练质量/吞吐；新release还包含双线mesh与异常cleanup先commit GPU streams后sync的死锁修复，后两者原PR也早于本扫描窗。**建议Mac训练/LoRA实践者按模型算子coverage测试，不把反向kernel倍率直接写成训练倍率。**

[MLX LM v0.32.2](https://github.com/ml-explore/mlx-lm/releases/tag/v0.32.2)于10-16T02:49:02Z修复MiMo-V2-Flash value scale：[#1971](https://github.com/ml-explore/mlx-lm/pull/1971)把checkpoint的 `attention_value_scale=0.707` 放在V projection后、写KV前，SWA/full attention均生效；无该字段保持旧行为。这里是补齐官方数值定义，没有独立quality benchmark；相关旧版输出不宜拿作同模型等价比较。

### OpenXLA跨主机接收：发布fabric handle前等接收端allocation完成

**项目/版本。** 10-17T01:21:02Z main [commit `a61c6012cfe2`](https://github.com/openxla/xla/commit/a61c6012cfe262acdaa367e67cf7546becae48a1)。

在 `kComputeSynchronized` 分配器复用HBM时，接收端compute stream可能仍在使用旧内容。旧实现只往接收端D2D stream排wait，却立即发布fabric descriptor；真正copy由发送端D2D stream执行，因此本地排队不约束远端写。`CrossHostReceiveBuffersInto` 现在在host侧等待transfer dependencies与物化的compute allocation event ready后才通知发送端，并把前置错误报告notifier；`ScheduleRemoteSend` 又把imported `dst_base`保留到usage event完成，防止异步copy中unmap。

新增专门cross-host测试源；本报只核C++patch，没有多主机NVLink运行或性能测量。这是GPU PJRT跨主机buffer协议的修复，**不等于TPU/libtpu发布，也不证明任意collective/RDMA协议都安全**。**建议相关GPU PJRT后端采用并做复用/失败注入验证，特别关注“本地stream wait”与“允许远端写入”的不同完成边界。**

### JAX custom VJP：remat/jit/scan保留gradient refs，vmap边界显式报错

**项目/版本。** 10-17T03:47:22Z main [commit `8ff5d87c8fda`](https://github.com/jax-ml/jax/commit/8ff5d87c8fda879438b47bf75f38740daf5499e7)。

`defvjp_with_accums` / `custom_gradient(with_accums=...)` 在先stage再求导（remat/jit/scan/checkify）时现在把调用方的RefAccum传进backward；旧版会退成dense ValAccum或抛类型错。对embedding/sparse-update等自定义反向，保留原位累加可避免无意构造整块dense梯度；这是机制意义，提交没有LLM显存/时间benchmark。

**边界。** 先 `vmap(f)` 再求导的custom VJP不能给每例一个调用方ref，现明确ValueError并建议有batch规则的hijax HiPrim；不能称所有变换组合自动原位。文档示例还要求batch规则直接转交原输入；搬轴形成新值会正确地退成dense路径。代码与docs/301/custom-derivatives.md有配套测试，未独立执行。**建议使用activation remat与自定义稀疏梯度的JAX训练核查变换顺序和backward jaxpr。**

<a name="evaluation"></a>

## 评估与可靠性

### Shortcut Sensitivity：标签可预测、观测相关与guard真正依赖，需要三级证据

**Kontext；2610.22243v1，2026-10-15T17:18:05Z；旧缺口补录。** [论文](https://arxiv.org/html/2610.22243v1)；给[公开聚合code/results](https://github.com/hasandemirkiran/shortcut-sensitivity-agent-safety-evaluations)，本轮未代码核查。

§2区分只看弱cue能预测label、cue与guard决定观测相关、保持构念的配对改写能改变决定。§3–4/Table1对LinuxArena去重为576完整shell trajectory，命令平均长度弱诊断AUC.910；AgentDojo exact-tool lookup mixed-domain F1.901→domain holdout.495；AgentHarm全判unsafe也可F1.836而balanced accuracy.5。它们揭示构造签名和分母问题，**没有证明guard就用了捷径，更不能直接宣布基准无效**。

§5/Table2：固定64 AgentDojo steps/58interactions，四域安全/不安全平衡，每条件3次交错调用、含JSON空白placebo；Haiku4.5共960calls。marker从INFORMATION改成UNTRUSTED_EXTERNAL_CONTENT，全体block率-8.33pp，opaque tool alias+11.46pp，unsafe均仍block；原准确率73.44%，改marker81.77%，alias61.98%。GPT5.6Luna同960calls marker0、alias+1.04pp，范围含0，未复现Haiku大小的效应。不是模型强弱排行榜，也不是cue sensitivity普遍率。

**限制/建议采用审计方法**：64例不是概率样本，guard选择用过观测screen；marker改变了provenance文字，alias不是全局一致改名、无盲人类等价审查，结论限这些精确编辑。上线评估建议按interaction/trajectory去重、域外holdout、配对干预+repeat/placebo；把label预测性与实际causal reliance分开报告。稳定ID：`arxiv:2610.22243v1`。

### FEB-30：直接检查记忆状态能发现QA总分遮住的更新与归因错误

**AWS / Technical University of Cluj-Napoca；2610.22209v1，2026-10-15T16:58:12Z；旧缺口补录。** [论文](https://arxiv.org/html/2610.22209v1)；给[benchmark仓库](https://github.com/amazon-science/feb-30-benchmark)，本轮未代码核查。

§3–5从已知事实演化轨迹反向生成文档，保留每个ingestion后的目标fact state和create/update/reinforce/supersede动作，再并行测fact store coverage与最终QA。30轨迹、六类/三个难度、147文档（127state+20noise），多模型生成+三个judge一致核对，并有人工修正；“确定ground truth轨迹”不等于实际scorer无误，matching仍先embedding>.8再LLM entailment。

§6/Table3–4：一个extract/embed/classify/synthesize架构11次run，主要变化来自retrieval，不是11种独立系统。intrinsic60.6±1.6%、extrinsic48.8±2.9%，逐轨迹绝对差均值.168±.025；差>.15的轨迹占43.6%，**不是44%答案错误或44%生产系统失效**。action分类20.1±2.7%、false update22.2±1.2%说明更细诊断有用。四配置含早/公开corpus、同一pipeline的改版与换backbone，MAD .158–.181，不能包装为覆盖四种不同memory架构。

**限制/建议采用状态级评估**：仅30 synthetic技术/业务轨迹，不适合细微排名；未覆盖episodic/procedural memory。intrinsic是各中间state平均，extrinsic是最终state问答，测量时点本身也不同，不能把全部差值都因果归给retrieval。正文Table4与其下概述的若干难度值不一致，本条以表为准，不引用概述的旧数。不要把330个重复trajectory-pair当330个独立任务来夸大显著性；优先在真实序列上验证annotation/entailment与端到端效用。稳定ID：`arxiv:2610.22209v1`。

### 异构pipeline精确makespan：可读理想模型，现有实机日志不验证pipeline上界

**Yantai Vocational College of Culture and Tourism；2610.21815v1，2026-10-15T13:03:06Z；旧缺口补录。** [论文](https://arxiv.org/html/2610.21815v1)；给[heteropipe](https://github.com/ytidb6688/heteropipe)，本轮未代码核查。

§2–5对纯forward、固定stage时间、顺序fill-drain、无限buffer、固定延迟且无带宽争抢的传输，给出首microbatch遍历时间加其余microbatch瓶颈stage时间的closed form；阻塞模型把传输并入stage service，完全异步模型仅把链路延迟加进填充项。可用于检查将所有stage粗略取最大值的GPipe估算偏差，不能用于training1F1B、变长decode、有限buffer/backpressure或共享链路争用。

§7.1的16万随机/手工配置比对的是**同一理想模型的递推与闭式**。§7.2/9明确8571条生产请求只有单请求compute与queue日志，没有pipeline makespan；最短duration还混入输出长度差。故这些日志不能证明真实runtime总落在async/blocking sandwich内，也不能证明通信优化只影响首批延迟。§10承认带宽限制、bounded buffer、prefill/decode混合和时变service均未覆盖。

**建议观察**：适合作为planner的理想下界/单元算例参考，优先复现带宽资源与buffer扩展；暂不据此改真实集群通信优先级。稳定ID：`arxiv:2610.21815v1`。

<a name="coverage"></a>

## 覆盖、失败、补录与更正

### 本地与远端连续性

本轮先读运行memory、`.monitor/state.json`、历史日报与GitHub main。远端起点为[`b8203bee351b`](https://github.com/suhaibo666/tracker/commit/b8203bee351bfa010c3962fbac98fd0394b2b5c8)，递归树完整；13篇历史日报和README共14个blob均与本地一致，最新日报/索引又按固定commit正文逐字节核对。没有旧待补传文件。本轮只新增10-18日报并更新日期索引，不改写llm-knowledge wiki。最终归档回读及提交回执单独保存在本地运行记录。

### 六核心项目实际范围

| 仓库 | 默认分支 | 72h回溯commit | 主窗commit时间计数 | 主窗merged PR（main / 全base） | 主窗release |
|---|---|---:|---:|---:|---:|
| NVIDIA/Megatron-LM | main | 56 | 8 | 10 / 13 | 0 |
| vllm-project/vllm | main | 272 | 45 | 45 / 45 | 0 |
| verl-project/verl | main | 13 | 0 | 0 / 0 | 0 |
| sgl-project/sglang | main | 256 | 46 | 46 / 48 | 0 |
| THUDM/slime | main | 6 | 1 | 1 / 1 | 0 |
| pytorch/torchtitan | main | 53 | 10 | 10 / 10 | 0 |

commit时间与merge时间为不同口径；例如Megatron#7606 merge在18日00:32但committer时间17日20:05，#7974 merge在主窗而commit较旧。SGLang另两条base是sglang-miles、oss-sync分支，未当main功能；Megatron三条dev也分列。release列表每仓取最新10条（Titan5），最新published均早于回溯窗口，非匿名API报错当0。

全部主窗merged候选元数据已核，正文/关键patch为选择性深审，不宣称所有patch逐行认证。旧10-05→10-08共434commit库存只关闭本轮三项的语义阅读，余量继续保留；72h中未选大型kernel、依赖与新open PR仍待。历史日报稳定PR/commit URL去重，本轮新增主线条目无已报告URL命中；Dory/Beam、KPool dev port、P2P风险属于状态跟踪单列。所有证据保存在同目录core-inventory-*.json、core-pr-*.json与core-p2p-*，逐仓进度/未审ID见core-coverage.json。

### 扩展工程与硬件检查

| 扩展项目 | 默认分支 | 回溯commit数 | 审阅状态 |
|---|---|---:|---|
| pytorch/pytorch | main | 302 | 清单分页完整；实质patch选择性核查 |
| deepspeedai/DeepSpeed | master | 13 | 清单分页完整；实质patch选择性核查 |
| huggingface/transformers | main | 59 | 清单分页完整；实质patch选择性核查 |
| huggingface/trl | main | 46 | 清单分页完整；实质patch选择性核查 |
| OpenRLHF/OpenRLHF | main | 0 | 清单分页完整；实质patch选择性核查 |
| triton-lang/triton | main | 28 | 清单分页完整；实质patch选择性核查 |
| Dao-AILab/flash-attention | main | 4 | 清单分页完整；实质patch选择性核查 |
| NVIDIA/TensorRT-LLM | main | 125 | 清单分页完整；实质patch选择性核查 |
| vllm-project/vllm-ascend | main | 112 | 清单分页完整；实质patch选择性核查 |

九仓合计689个默认分支commit；31项PR的正文/files取回，另12个PyTorch真实main commit核对。这里只将正文实际展开的机制记作已审；没有把另外数百个commit、所有release更新或非默认分支标成完整语义审计。release各读最新20项，页面/版本号变化本身不入正文。

NVIDIA/AMD技术博客归档、Gaudi官方1.24.2 release notes、Cloud TPU官方release notes、Huawei productbulletins及Neuron SDK/四组件notes均实际打开。Neuron组件页当前列8-17的2.32.0（NKI0.6.0、runtime2.34.10等），补齐了组件入口读取，未据此回报本窗新发布；compiler页顶部与详细小节版本还不一致，未替它推断新版本。Huawei页面可读不等于全部CANN/HCCL动态内容完整。

另枚举NCCL/RCCL、NKI samples、XLA/JAX、Ascend/pytorch与MLX/MLX-LM默认分支回溯，共371个commit，选择性核原始patch；NCCL/RCCL在此清单返回0仅是该分支/时间口径。错误尝试`mlx-explore/mlx`、`aws-neuron/nki`返回404，前者已用正确`ml-explore/mlx`恢复；后者不能代表NKI未更新。**CANN/HCCL、libtpu及未选组件实现/旧窗口仍不完整。** 不把没有检索命中或没有新release写成厂商没有发布。

### 模型来源、去重与访问边界

- **Dory/Beam**：本轮HF各格式SHA与昨日相同；重新打开Dory BF16卡/config/history，Beam卡仍写Technical Report coming，官方release/safety页已读。检查入口未得到独立Dory完整报告，不把“未找到”写成全球不存在。core方向核实vLLM **[Dory #60149](https://github.com/vllm-project/vllm/pull/60149)仍open、merged_at=null**（head `55076950b6c155f384d6766d63563c3bc20a0e3f`）；**[Beam #61978](https://github.com/vllm-project/vllm/pull/61978)仍open、merged_at=null**（head `00fe254cd01b1d6b9f69370750f113cffb90e474`）。官方“day-zero支持”不能替代main/release合并状态；运行专用分支与稳定版需分开。旧Dory loop训练调度/last-loop数值消融、量化质量，Beam完整训练报告/ISE/self-distillation工件仍待续。
- **Step5访问修复**：本轮[平台模型页](https://platform.stepfun.ai/docs/en/guides/models/step-5-preview)HTTP200，明确**Maximum output 64k**，同页又称`max_tokens`默认INF；固定HF卡仍写Maximum Output up to1M，且1M含输入+输出。可关闭“平台页SSL超时”的入口故障，但不能将请求参数默认INF当无限输出、或把托管服务64K与自托管工件1M合并成一个通用上限。NVFP4独立质量/时延仍未验证。
- **Liquid d1-omni GGUF**：主窗commit `8eb2d2a1...`仅修正`base_model`大小写；此前GGUF上传10-07，不记新模型。
- **UserRepGen**：同一`bfc905b...`，卡末尾仍明确paper link/BibTeX待确认；检索没有恢复正式论文/训练仓，保留待办。未将重新读卡当新成果。
- **Meta**：固定SparkOpen config再次403；两仓SHA未变，门控缺口保留。**NVIDIA retrieval/Super**相关SHA与昨日相同，未重新作全表技术审计；10-20日期与可见上传、67B/65.4B差异沿用旧待办，未关闭。
- **其余未重复**：MuseSparkOpen/DFlash2、Wonder、DSpine、GraniteSwitch、Nemotron3.5/Bolt、Step5、GPT-Live1.1已有稳定工作；Timestral为时间序列模型，scFM/天气/雷达/机器人工件不列LLM主条。DSpark fusedscale tensor diff、GLM-Legal权重、Bolmo Nature全文本轮未完成，显式延续。

- **HF**：sources.json的**31组织×createdAt/lastModified双路=62请求，全部200**。以每页100为分页单位，按各路日期单调降序校验；所有路首个返回页已跨过72h边界（Reflection modified到公开清单结尾，Anthropic两路为空），所以无需后续页，**不是固定Top100后直接声称全覆盖**。本轮公开列表范围内70个72h候选均保留于inventory；按模型与日期去重，选择性读卡/config/history。另读inception42/IFM/inclusionAI双路与Jais六仓，未声称私有/门控/删除记录、所有分支或闭源模型穷尽。
- **官方站**：已抓取Anthropic news/research、DeepMind blog/model cards、Meta research、NVIDIA Nemotron、DeepSeek updates/news、Qwen blog、Seed两类API、Kimi blog、AutoClaw、MiniMax news/IR/research API、Tencent/HyMemory、Baidu、Step、MiMo、LongCat、Reflection及Liquid。OpenAI直接403、Microsoft/Mistral/xAI直接超时、Ai2直接403，均通过web实际打开相同官方入口恢复可读列表；仅代表这些入口成功。IFM博客/主页直接403，经web成功，HF工件直接可读。
- **动态/陈旧范围**：Qwen仍只7字，Tencent/Baidu/Step入口仍动态壳；Kimi研究列表实际可读但最新可见7-16。MiniMax IR可见到10-13，research API到1月、news API到5月，渠道不一致仍在。Seed paper第一页200但0项且has_more/total246；沿`page_token=20`再抓到1个旧条目，仍不能补回第一页未显示的20条。blog第一页返回12个可见条目并跨到旧日期；隐藏/过滤项与分页完整性仍有缺口。**检索未命中或列表未见新项均不表述为厂商没有发布。**
- **范围限制**：逐来源状态/URL/HTTP失败/实际条数/边界及哈希在本地`models-coverage.json`；抓取时刻晚于报告cutoff，只将能确证发生在cutoff前的事件列为主窗，新观察与旧事件区分。没有把本轮所有HF候选、所有官方页面或训练实现都称为完整审计。

### 论文与正式学术来源

- **已恢复明确访问缺口**：昨日319个arXiv replacement候选，本轮7批id_list全部200，319个稳定ID+最新版本/更新时间已保存；新增cs.DC独有9个replacement也200，总328个。仅恢复元数据，不等于328篇全部完成版本diff或正文审阅。
- **仍未恢复updated枚举语义**：主窗、72h、原旧缺口三个lastUpdatedDate查询虽200，Atom title/link仍被服务改写成submittedDate；本轮新提交分别返回0/962/402条（含cs.DC），不能据主窗0条推断无修订。CL/LG/AI new/recent与昨日字节相同，公告停在10-16；new四学科合计902 unique IDs/328 replacement，不能补足所有历史修订或后续尚未公告稿件。
- **HF与正式来源**：HF10-14/15/16各50条，10-17空列表，10-18请求400；这些是发现层。TMLR相比昨日快照又增11 stable forum IDs，含Language Modeling with Hyperspherical Flows（原arXiv2605.11125）及Separable Pathways（2604.20039），仍只有October2026月粒度，不能标本窗首发；OpenReview API403，Delta-Rule旧待查PDF与新Hyperspherical PDF均challenge，正文/精确版本未核，不转述指标。ACL/PMLR根页可读但未完成所有卷/版本事件枚举。
- 尚未逐篇精读的候选仍保留：teacher selection/critique/action OPD、奖励/agent训练、Cyclotron等。不能把“未纳入”解释为无价值或宣称全覆盖。完整请求、清单、失败、证据定位与后续待办见papers-coverage.json。

### 本轮补录、更正与后续区间

本报的研究正文多数为10-15漏收论文补录；Jais3、d1-ultra、Apple MLX等是72h回溯补漏，Ling是既有待办的一手补证，工程另含明确10-06/07/08/14/16日期的补录。Un-OPD只把v2新增证据算增量；Ascend #18902从open变为main合入属于状态增量。既有报告不因此被覆盖或重新生成。

发布前交叉复核已纠正两处表述：DeepSeek TB4.0新旧harness不一致，改为新卡并列自报而非受控权重A/B；新ZeroBench Pass@5由10-17官方更新日志确认，卡未重述的缺口仍说明。SGLang decoder默认条件按最终patch补全CUDA/HIP，与ROCm encoder+attention-DP拒绝门分开。另按最终patch处理TRT bot陈旧摘要、Megatron checkpoint元数据与PR正文冲突。以上是本稿发布前校正，不冒称修改历史日报。

仍须续查原论文失败区间 **2026-10-15T01:55:47Z→10-16T01:00:40Z** 及回溯修订：328篇元数据恢复并未完成逐篇diff，`lastUpdatedDate`被改写的问题也未解除。工程旧缺口10-05起的已枚举库存继续选择性审阅；模型动态站、门控工件、完整训练报告与正式会议版本枚举按原待办保留。下轮事件游标从 **2026-10-18T01:05:14Z** 续扫并再回溯72h，另处理这些未闭合区间；全局完整覆盖水位继续为空。

逐源请求结果、候选稳定ID/版本、失败、审阅与本地/远端同步分账保存在本地归档`.monitor/research-2026-10-18/`、`state.json`、`run-2026-10-18.json`及归档回执。仅本轮日报和README进入tracker的既有`llm-training-daily/`目录。
