# LLM 技术动态日报 · 2026-10-20

> 时区：Asia/Shanghai（UTC+8）。增量扫描窗口：**2026-10-18 09:05:14 → 2026-10-20 08:55:34**；72 小时回溯自 **2026-10-15 09:05:14**。10-19 运行留下未归档研究草稿，本轮从最近成功归档恢复连续性，并重新核实这些线索；没有伪造一篇10-19日报。更早未完成区间另列于文末。
>
> 实际生成时间：**2026-10-20T09:08:16+08:00（Asia/Shanghai）**。下文精确事件时间除注明北京时间外均为 UTC；公告、论文初版/修订、权重上传、合入与正式发版分别计时。截止后读取用于核实截止前事件/访问状态，不能作为把截止后新事件加入本窗的依据。
>
> 证据：本轮打开的一手论文全文、官方说明/模型卡、release与具体PR/commit；性能、质量和测试结果均为原作者/厂商披露，本轮没有独立GPU/NPU/TPU训练或服务复现。默认分支实现、专用分支、正式版本、Beta与未合入提案明确区分。扫描游标不等于逐源完整覆盖水位。

## 本次重要变化

- **模型发布先看披露边界。** MiniMax M3.1 Flash增加稀疏化与思考档位、Mercury3把扩散语言模型扩至1M上下文，均有可核API发布；前者权重待发布，后者没有公开训练配方和完整测速协议。本报保留可验证规格，不推测架构。
- **原生PyTorch上Trainium有实质工件。** Neuron2.33/TorchNeuron公开Beta源码、NKI0.7、混合架构实验kernel和通信/编译正确性修复一起到来。静态shape、fallback、LNC、packed routing与新旧同名包的兼容边界影响迁移。
- **训练系统的新机制值得小规模复现。** TorchTitan MLA传压缩latent并合并Ulysses通信、RL接通PP；verl改变QAT/低精度rollout的量化与refit边界；Megatron补checkpoint恢复、loss-mask和图内梯度存储。Zephon数据流水线补录强调扩缩容后相同global batch仍须固定recipe和lane约束。
- **推理工程不能只读版本号。** vLLM0.32、SGLang0.5.22正式发布，与晚于tag的主线KV/replay/专家分页变化分开记录。TRT长Gemma mask与host KV渐进分配有具体收益和资源代价；Transformers5.20.2修MoE零梯度回归。
- **有损投机必须测完整任务。** vLLM显式接入FLy，论文补录发现更长接受往往伴随更长输出/质量损失；DLite论文与当前checkpoint配置又有12/14层顺序差异。接受率、论文倍数和当前工件性能不能互相替代。
- **后训练关注机制与可验证失败。** GRAFT把monitor用于梯度路由；CriticOPD、CCE-OPD及多教师/skill蒸馏提供可复现方向，同时保留surrogate、额外教师成本、per-task回归和缺少多seed等限制。

目录：[模型与技术报告](#models) · [预训练与架构](#pretrain) · [后训练](#posttrain) · [多模态](#multimodal) · [推理优化](#inference) · [开源工程](#engineering) · [硬件软件协同](#hardware) · [评估](#evaluation) · [覆盖与缺口](#coverage) · [工程版本账本](#ledger)

<a name="models"></a>

## 模型与技术报告

### MiniMax M3.1 Flash：更激进稀疏化与可调思考，权重仍待发布

**MiniMax；官方发布日期2026-10-19。** [正式发布说明](https://www.minimax.io/blog/minimax-m31f)、[模型API指南](https://platform.minimax.io/docs/guides/text-generation)、[release notes](https://platform.minimax.io/docs/release-notes)。正式型号`MiniMax-M3.1-Flash`，不要把此前Preview名称和规格直接移植到正式版。发布稿称更激进的稀疏化提高在线推理效率；**未披露正式版总/激活参数、attention稀疏算法、预训数据与预算、后训练配方或训练代码**。Space Bunny累计处理200T token是匿名运行流量，不能写成预训练token数。

正式API指南给**1M上下文、文本/图像/视频输入、文本输出**；思考始终开启，`low/medium/high/xhigh/max`五档、省略默认max。兼容协议分别使用`output_config.effort`、`reasoning_effort`、`reasoning.effort`；OpenAI兼容返回中`reasoning_content`与最终`content`分离，禁止思考/`none`返回400。这里的协议事实来自MiniMax自身文档，不代表OpenAI模型行为。

发布方自报DeepSWE v1.1 **73.7%**、OSWorld V2 **61.7**、ALE **29.6**、CyberGym **82%**、BankerToolBench **76.2**；正文未给完整harness、重试次数、工具预算和置信区间，不将其解释为业务成功率保证。厂商在线速度声明200+ TPS、P50 170+ TPS，未给本页测量的硬件/并发/输入输出长度/effort分布；“近3倍更快、agent成本低70%”是特定比较口径，不是统一硬件基准。公布每百万token $0.15输入/$0.60输出/$0.015缓存输入，任务成本还受思考token、工具和失败重试影响。

发布稿明确**API、M Plan、Code已提供，权重稍后**；API文档虽宣传open-weight，本轮MiniMaxAI公开HF清单未取得M3.1 Flash工件，不能写“权重已开源”或推断许可。建议先固定effort与harness，在长任务上同时量成功率、完整任务时间、推理token和重试成本，再判断稀疏化是否带来实际收益。

稳定ID：`minimax:m3.1-flash:official-release:2026-10-19`。待办：正式技术报告、权重/许可、稀疏机制与训练消融。

### Mercury 3：扩散语言模型扩至1M上下文，速度声明仍需受控验证

**Inception；官方发布日期2026-10-19。** [发布页](https://www.inceptionlabs.ai/blog/introducing-mercury-3)、[官方型号与迁移说明](https://www.inceptionlabs.ai/models)、[API入门](https://docs.inceptionlabs.ai/get-started)。机制增量是继续扩大扩散式文本生成与推理模型，厂商称总参数较Mercury1扩大40倍以上、上下文从Mercury2.5的260K升至**1M**，仍保持高生成速度；**绝对参数、稠密/MoE组成、噪声/去噪目标、块长与采样步数、数据和训练预算均未披露**。不把“扩散”自动解释为非Transformer，也不把厂商“3倍智能”当严格统一指标。

厂商报告NVIDIA GPU上**1,068 tokens/s**、相较所列快模型4–8倍；已读发布正文未给具体GPU型号、batch/并发、prompt/output长度和完整计时协议。用户案例中的50道live Mind2Web导航任务成功50%，属于合作方生产harness样本；不等于完整Mind2Web、独立复现或所有agent任务水平。应分别测首个可用答案延迟、完整响应时间和任务级成功率，不能从tokens/s推导端到端agent加速倍数。

官方型号页给API ID`mercury-3`；常规每百万输入/输出$0.20/$0.70，发布期50%折扣$0.10/$0.35，缓存输入另为$0.01（发布期）。旧Mercury2/2.5不再向新用户开放，10-19前已有使用者可用至11-15，企业按协议保留。**这是真实迁移事项**：要测长上下文、tool schema与streaming语义，不能仅凭OpenAI-compatible认定零变更。发布页未提供可下载权重或训练代码；本轮直接docs访问403，经web恢复入门入口，尚未完整审计接口/版本差异。

稳定ID：`inception:mercury-3:official-release:2026-10-19`。待办：正式技术资料、可复现速度/质量协议、旧版本迁移细节。

### Ming系列：TTS/Audio补齐代码与部署，ASR权重删视觉张量属于打包修订

**蚂蚁inclusionAI；本轮10-18/19工件与代码增量。** [TTS代码固定版本](https://github.com/inclusionAI/Ming-3.0-tts/tree/e999c39fa676722aaa9ea29b5aa19a90525b271f)、[Audio代码固定版本](https://github.com/inclusionAI/Ming-3.0-audio-flash/tree/cc4e155f156bb35464669be2fcd0762ee42728c3)、[ASR打包commit](https://huggingface.co/inclusionAI/Ming-3.0-asr-tiny/commit/d08d77d4b32f0707eb430a995ccbf933f0b04583)。本轮逐一对照主窗前与当前HF tree的Git/LFS oid，详见`models-diffs.json`。

**TTS代码可用性。** 两款HF权重9-24已上传；10-19卡/架构图更新均**未改变权重oid**。独立GitHub仓10-19T08:58Z初始发布，提供推理、Docker、任务cookbook。Dense0.5B和LingV3 MoE tiny（backbone总7.9B/激活1.3B）共享LM+DiT patch head+audio tokenizer框架；任务控制分模式，tiny另支持多说话人podcast，不能把情绪/方言/freeform任意叠加。MIT；0.5B与tiny依赖环境不同，后者固定Python3.10.15、torch2.7.1+cu118、Transformers4.46.3。技术报告仍roadmap，训练数据/预算未披露。榜单与卡是作者汇总、不同模型遵从各自推理设置，不能等同统一硬件质量—速度比较；本轮不重报旧权重为新模型。

**Audio代码/接口。** Flash卡报126B总/6.5B激活，原生audio encoder/projector+MoE backbone，Talker可选语音输出；Audio tiny卡报9.3B总/2.7B激活，勿与TTS tiny的backbone口径混用。HF主窗变化是首版完整卡、语音preset/文档，**未改权重**。GitHub当前提交将选择统一到`--voice`，vLLM-Omni客户端用`audio.voice.id`；标准128K profile为8×H80080GB，SGLang参考4GPU/text-only，不能混写成所有runtime已支持同样speech output。当前harness依赖适配分支/补丁，未审计上游merge状态，不能说主线均已原生支持。

**ASR重新打包。** 10-18T02:18Z commit明确“移除unused vision tensors，保留ASR tensor值”；本轮index核验**移除351个张量名、零新增、保留10181个名称**，两分片总大小18,885,677,168→17,732,860,448字节。作者的“保留值”未做全量逐tensor下载比对，不能升格为本轮bitwise验证；变化不证明新训练。当前单轮ASR支持热词/对话上下文，输出无逐字时间戳/diarization、示例不支持streaming。新SGLang cookbook入口和benchmark图亦属可运行/评测材料增量。建议固定checkpoint与适配器commit，并将ASR/TTS/Audio三套环境、模型预算和任务模式分别验证。

稳定ID：`github:inclusionAI/Ming-3.0-tts@e999c39fa676722aaa9ea29b5aa19a90525b271f`；`github:inclusionAI/Ming-3.0-audio-flash@cc4e155f156bb35464669be2fcd0762ee42728c3`；`hf:inclusionAI/Ming-3.0-asr-tiny@d08d77d4b32f0707eb430a995ccbf933f0b04583:audio-only-packaging`。

### Nemotron Parse：新模型、同输出scorer修订与畸形块解析修复应分开记

**NVIDIA；整包commit10-18T02:52:38Z，scorer材料18:00:22Z，解析代码修复10-19T02:02:21Z。** [固定当前版本](https://huggingface.co/nvidia/Nemotron-3.5-Parse/tree/e7ea79242a87ea0ad3919b672bfedb429a683206)、[发布模型卡](https://huggingface.co/nvidia/Nemotron-3.5-Parse/blob/dca5b2df89d00a671509204501d8f1f379d5e94f/README.md)、[回归说明](https://huggingface.co/nvidia/Nemotron-3.5-Parse/blob/e7ea79242a87ea0ad3919b672bfedb429a683206/reference/parse_3_5/README.md)。卡自标release10-17，仓库createdAt9-03均不证明当日已公开整包；本轮以可见commit事件定位。

约0.631B可训练参数（含MTP，作者口径），C-RADIOv4视觉编码器→压缩adapter→10层mBART式causal visual-prefix解码器，输出文本/语义类/bbox/阅读序，适合PDF语料构建。OpenMDW1.1模型/代码、tokenizer MIT；训练精确配比/预算未披露。ParseBench的single/double/triple为72.74/74.92/75.20、15.42/11.65/10.62页/s（2078页、单H100）；double/triple增加区域重读，不能把75.20当单前向。竞争分数来自各自发布协议，速度才是作者重跑，非全表同scorer端到端比较。

10-18 full500（250表格+250图像页）同一批原始输出，仅将Table body经过已发布HTML postprocessor，轻量回归总分70.17→74.49、table quality70.41→83.38；bbox/类/GT不变。**不是模型能力涨4.32pp，也不是官方TEDS。** 单H10080GB/BF16/MTP/concurrency128为9.03197页/s，排除preload/warmup/转换/scoring；19页截断、2个malformed Table仍计分，Pillow版本有偏差。

10-19新增`engine/blocks.py`令不合法坐标块不能吞掉后续块；`parse_results.finalize`独立校验坐标/类，保留`source_raw_out`，区分partial/raw_fallback；fixture覆盖`<x_0.org>`页脚畸形坐标。**这是真正后处理可靠性修复，与前一天table scorer纠偏不同**。发布、10-18回归、10-19修复的model.safetensors LFS SHA256均`81dd7609...787e9`，未发现权重更新。本轮静态审计未执行全部回归，不宣称fix已本地通过。建议把权重、parser、scorer分别固定版本，业务PDF评测单列截断、malformed、fallback和表格错误。

稳定ID：`nvidia:nemotron-3.5-parse:release`；新增代码事件ID：`hf:nvidia/Nemotron-3.5-Parse@e7ea79242a87ea0ad3919b672bfedb429a683206:parser-fix`。

<a name="pretrain"></a>

## 预训练与架构

### Looped Models Done Right：循环位置与输入注入比额外latent state更值得先做控制实验

**2610.23036v1，2026-10-16T07:12:41Z；补录。** [论文全文](https://arxiv.org/html/2610.23036v1)；[官方代码755c7b3](https://github.com/ifm-ai/xllm-loop/tree/755c7b3a91081fe0f83500eabd53bd2490b8b745)，静态核查，未训练。

§3/Tables2–4固定730M物理参数、112次逻辑block执行，对比28层全栈×4与8层prelude→12层core×8→8层coda。TxT360，42/84/168/336B token，序列8192、global batch512、AdamW及3档LR取最优。sandwich在336B时MATH500 **20.8→32.8**、GSM8K54.97→62.02；ARC-C39.40→38.45、HumanEval+29.88→28.66，收益依能力而异。中循环每次注入固定prelude输出，BBH-CoT48.30→54.94、HumanEval+28.66→34.15，但MATH50032.8→29.2。不能把“持续看到原输入”推成数学推理必涨。

随机初态或共享H/L双状态没有一致额外收益；H/L相对输入初始化MATH500再降7.2pp。§4/AppA的MoE是25专家top2、无共享专家、专家宽2432、8B物理/约793.9M激活参数（不含embedding），unrolled active约3.2B；训练到500B token后Huginn比Ouro八项更好，但112层feedforward MoE仍在七/十项领先，九项可比均差4.96→1.71只是跨架构差距缩小。routing冻结干预支持每轮重选expert有用，不足以把更均衡routing当唯一因果解释。

实现对应：`xllm/models/looped.py::Huginn.forward/_run_recurrences`按prelude→重复core→coda执行；`InputInjection._decayed_state_and_update/forward`确为指数衰减state加softplus步长×线性input；`xllm/paper_part1/recipes.py::_model_fields/_RECIPES`固定8/12/8、注入/初态消融。论文未给可复算的GPU型号/节点数；§5仍把多seed作为后续工作，matched logical depth不等于已测端到端时延相同。**建议深读并复现最小三项消融**：先固定数据、FLOPs和物理参数比较全栈/中循环，再开输入注入；按数学、代码、知识分别看回归。稳定ID：`arxiv:2610.23036v1`。

### Frozen Vocabulary：168M词表参数可换成16.8MB二值码，但不是零代价压缩

**2610.22428v1，2026-10-15T06:34:18Z；补录。** [论文](https://arxiv.org/html/2610.22428v1)；[官方工件822177f](https://huggingface.co/QLNI/embeddings-are-overpriced/tree/822177f4003d65629068339d5be558d1509a057d)，代码/核函数静态阅读，未跑模型。

§2–4把Gemma3-270M预训embedding中心化、取512 PCA方向再sign为每token512bit，配512↔640可训练接口与unigram输出prior。三组body从随机初始化训同序同seed的2.05B FineWeb-Edu token，1张RTX5090、seq1024、262144 token/step、7820步；**二值表继承预训embedding知识，不能称全模型从零学习、也不能与A无先验地等同。** A学词表268.1M trainable，B冻结表浮点body101M，C冻结表三值body101.2M；C还改变LR schedule和可训prior，不是单独量化消融。

Tables2–3：B相对A validation loss3.155→3.358（差+0.203nats），LAMBADA+1.6pp、ARC-Easy−3.4pp，HellaSwag/PIQA差的paired区间跨0；C loss3.439、ARC-Easy−4.9pp。单seed，误差条只对测试题bootstrap。Table5在Xeon W-11955M、8线程、128token context、greedy、16bit KV比较C自制AVX2 kernel615–740tok/s/62MB与A llama.cpp Q8_0 116tok/s/471MB；5.3–6.4倍只属于该模型/软件栈的decode，C质量低，prefill逐token，B没有专用kernel。不要照抄作者关于所有quantizer都不量化词表的泛化说法。

`code/train.py::frozen_table`核实PCA/sign及固定log-unigram，`Model.__init__/hidden/logits`核实table为buffer、接口map与output prior；C的`TLinear`替换仅body线性层，attention/KV保持浮点。§5新增1万PubMed词可省3.2%token，但B/C每字符loss仍增3.3%/3.1%；不是无损扩词。**建议在极小模型/CPU内存预算下复现**，先补等token多seed及更充分训练A，再测prefill和量化embedding对照。工件非商业研究许可并继承Gemma条款。稳定ID：`arxiv:2610.22428v1`。

### Priming Base Models：用软件工程自然数据和少量轨迹做长上下文midtraining，收益要在相同SFT后检查

**2610.22019v1，2026-10-15T14:58:21Z；旧缺口补录。** [完整论文](https://arxiv.org/html/2610.22019v1)。本次未在论文入口找到专用训练实现；机制以下按论文核查，未代码验证。

§3/Table1把merged issue–PR、import依赖排序的完整仓库、多文件diff/commit历史、CI日志与agent轨迹混合；正式500B run的token占比约40.8/20/3/1.2/10%，另20%预训replay、5%长上下文数据。轨迹只对assistant动作算loss，环境tool输出mask；自然代码不套chat模板。问题是短上下文预训很少教跨文件状态与交互，而直接大幅延长SFT可能损伤后续RL训练性；**本文没有等token长预训/长SFT对照，不能据此证明midtraining必然最省预算。**

§4/Tables3、6、AppB：Nemotron3.5 Lightning总30B/激活3B和Super120B/12B，GB300、Megatron-LM、BF16、packed256K、global batch64，Adam；分别500B与280B midtraining tokens，之后都用相同100B-token agent-focused SFT。Lightning SWE-Verified60.7→64.5、Multilingual55.8→62.8、Pro32.0→35.5；Terminal-Bench2.0只22.6→23.6，且300B checkpoint高于500B。Super DeepSWE11.7→15.7、SWE-V72.9→74.3，**Multilingual69.0→67.9、Pro53.9→51.8反而下降**。不能把两规模写成所有基准均增益。

§5/Table9的tool-response masking对照为早版SFT：SWE-V27.5→37.8，不应与最终主表直接拼接。提高LR有回归风险：Lightning在5e-5下MATH50079.7→61.7；20%replay不保证任意LR安全。作者承认每对比单run，小差异可能噪声；AppD仅单seed20步GRPO，证明优势在RL开始时仍在，**没有证明RL能额外提升更多**。跨harness结果没有单scaffold轨迹控制，不能把全部收益因果归给scaffold多样性。

**建议深读/复现**：采用固定短SFT筛recipe，再做完整SFT与一般能力回归检查；分别预算自然数据、轨迹、replay和256K训练成本，补等token替代方案。不要只按base BPB/LoCoDiff挑配方，本文这些proxy未可靠追踪最终post-training排名。稳定ID：`arxiv:2610.22019v1`。

### When Is Muon Optimal：随机半径破坏的是某种“精确最优”解释，不是Muon训练有效性

**2610.21549v1，2026-10-15T09:13:19Z；旧缺口补录。** [完整论文](https://arxiv.org/html/2610.21549v1)。理论论文，无训练规模、硬件、语言模型benchmark或速度提升；未发现专用实现链接。

§2–3把固定球面的Isotropic Curvature Model改成半径独立随机、方向均匀的RR-ICM，刻画scaled polar update精确最优的梯度谱集合。关键是半径分布的原子恰落在凸曲率profile的kink时，才保留非标量谱的subgradient自由度；半径无原子时该自由度归零。对满列秩梯度，如果其正polar factor有不同特征值，scaled polar方向便不是此surrogate的**精确**最优；任意薄的连续shell也可触发这一现象。

§4/Theorems2–3保留半径与曲率，改为对球面所有方向概率分布取worst case：只要存在非零global minimizer，其归一化方向就是梯度polar factor，步长由核范数与一维凸问题决定。这提供一个minimax解释，但最大ambiguity set是作者刻意构造，未从真实训练数据估计。

**边界/建议观察**：假设曲率有限、凸、非减，半径独立、有效profile有限且梯度满列秩；§6明确未证明近似最优误差很大，也没覆盖实际Muon的momentum、有限Newton–Schulz和完整训练动力学。适合纠正“薄shell≈固定sphere所以精确最优也近似保持”的推理；不能据此切换optimizer或宣称Muon失效。稳定ID：`arxiv:2610.21549v1`。

### Indic合成预训报告补漏：翻译质量、配比和评测有效性必须一起看

**NVIDIA / AI4Bharat；文档署10-15、HF PDF上传10-16T02:16:20Z；10-19草稿发现、本轮重核。** [44页固定报告](https://huggingface.co/datasets/nvidia/nemotron-indic-pretraining-report/resolve/11ada57b3bc8e59ec2eb40571d48cfe62bc4c204/Scaling%20Synthetic%20Pre-training%20Data%20and%20Evaluation%20for%20Indic%20LLMs.pdf)。本轮重新下载PDF，与上轮逐字节一致，SHA256`d63120c3...7bd901a`；沿用已读§3–6、Tables5.4/5.7/5.8的分析，非10-20新论文。

报告约2826B目标token（Gemma4 tokenizer口径），2601B native-script+225B romanized，22语言/2.44B文档。罗马字文本来自相同native材料，不是额外独立知识源。top12语种整文翻译→同模型MQM judge→低于75定向rewrite→再judge，余10语言流程不同；92篇Hindi初审不能覆盖全语种，同源translator/judge存在偏好盲区。

训练不是2.8T全量：100B decay/CPT实验中Indic45B，NPI25%+Sangraha20%；12语种均值MILU63.2→64.5、MMLU-Pro40.2→41.8、ARC-C84.2→87.6、GSM8K67.7→75.7，但**只用first-pass accepted**，不能证明rewrite回收收益。200B三配比每设定仅一run。英语Table5.8有多指标下降，不能泛称“英语无损”。作者还说明CPT尚未在修复后的V1/MILU1.5重评，100B checkpoint未保留；新benchmark审计不能追溯保证旧对比。

§6.4译后benchmark自动通过者经人审仅73.1%有效、26.9%仍无效；这是**benchmark翻译**误差，不能当2.8T预训corpus整体错误率。报告仓CC-BY4.0不证明完整corpus可下载/同许可，数据分片与插件代码仍未完整审计。建议分离first-pass/rewrite消融，追踪源文档去重、tokenizer计数、英语回归、人审与judge错判，并在审计后benchmark重评。

稳定ID：`hf-dataset:nvidia/nemotron-indic-pretraining-report@11ada57b3bc8e59ec2eb40571d48cfe62bc4c204:report`。

### Zephon：扩缩容后保持在线数据流水线的全局批次，适合隔离数据配方消融的混杂因素

**DatologyAI；10-16 官方技术说明，本轮补录。** [官方文章](https://pytorch.org/blog/bringing-elastically-deterministic-data-loading-to-torchtitan-with-zephon/)与[固定版本集成说明](https://github.com/datologyai/torchtitan-zephon/blob/cdaaa5ac402d21598d626c592028783f49b49d64/examples/zephon/README.md)。代码基线为该 fork `cdaaa5ac402d21598d626c592028783f49b49d64`（10-12），不是本窗新合入 TorchTitan 的能力。

在线 tokenize/packing/mixing 的状态通常与 worker/rank 分配纠缠。Zephon 用固定逻辑 lanes 保存状态，accumulator 顺序作决策、无状态 worker 并行处理后重排；checkpoint 记录最近 flush 后尚未完成的输入，恢复时重放重建状态。官方对 1B 模型在 16 GPU 的 step150 checkpoint 恢复到 8/32/64 GPU，展示全局批次摘要相同；另在 20B token 训练中，普通 loader 的 DCLM/FineWeb 分数跨 GPU 数波动达 0.82/0.64 点，Zephon 为 0.05/0.011 点。这些是作者实验，未披露足够多 seed/硬件细节来估计普遍收益。

**建议复现。** 固定配方、tokenizer、seed、global batch 与 lane 数；lane 必须整除预期 DP 规模，并按完整 lane window 消费。适配器 CPU demo 的比较忽略同一步内部顺序，不能据此宣称逐 token 排序在任意配置都相同；数据一致也不保证浮点归约和 loss 逐位一致。目前适配器只支持在线原始文本。先跑 2→1/1→2 的数据恢复 demo，再验真实训练；本文不把它当吞吐普遍提升或所有模态可直接替换。

稳定ID：`blog:pytorch:zephon:2026-10-16`；`github:datologyai/torchtitan-zephon@cdaaa5ac402d21598d626c592028783f49b49d64`。

<a name="posttrain"></a>

## 后训练

### GRAFT：把监控结果用于梯度路由，避免用奖励惩罚训练出隐蔽作弊

**MATS / Stanford / FAR AI / Anthropic；官方博客2026-10-19；论文v1为10-12T16:28:28Z、代码10-13/14，均早于本轮回溯。** [主窗官方解释](https://alignment.anthropic.com/2026/graft/)、[固定论文v1](https://arxiv.org/abs/2610.17062v1)、[代码固定基线](https://github.com/jnward/GRAFT/tree/b0767d1c40e5a03d78fc5ff4b6abc92829168386)。历史日报/已报ID未匹配，本条归为**主窗官方技术说明触发的旧论文补漏**，不伪称10-19首发论文或新代码。

GRAFT在每层MLP加入RETAIN/FORGET分支。被监控器标记的目标轨迹只更新FORGET，未标记轨迹更新两者；训练完成移除FORGET。同时用独立anchor环境训练仅RETAIN的部署路径，缓和训练时双分支与部署时单分支的不一致。监控器决定更新哪些参数，**不改变目标环境样本奖励**。固定README明确RL使用GRPO、发布vLLM0.17补丁；`SplitMomentAdamW`用路由梯度的一阶矩与未路由梯度的二阶矩，不能用普通AdamW替换后宣称同配方复现。

作者在Qwen3-8B的单轮Countdown-Code中，用只检测第一类作弊的启发式monitor；奖励惩罚基线学出约30%的第二类漏检作弊。GRAFT训练后移除FORGET，所有类型作弊接近零，真实任务解答率仍较base高近20pp。另有135M六个toy环境与32B多轮SFT证据，但**32B不是多轮RL成功验证**，其任务能力未见强提升。所有prompt可作弊的主设置对方法有利；仅半数可作弊时inoculation prompting同样有效。尚未验证LLM monitor、真实大规模多轮RL与anchor构建成本，不能泛化为解决reward hacking。

建议先复现发布图表输入与Countdown基线，再在实际训练中同时测可检测/不可检测作弊、anchor任务能力、FORGET移除前后表现，并保留优化器与采样配置。不要只观察monitor报警下降就判“对齐改善”。

稳定ID：`arxiv:2610.17062v1`；主窗事件ID：`anthropic:graft:official-blog:2026-10-19`；代码ID：`github:jnward/GRAFT@b0767d1c40e5a03d78fc5ff4b6abc92829168386`。

### CriticOPD：学生带批评重写，训练时只让教师保留批评上下文

**2610.20572v1，2026-10-14T17:54:34Z；旧待办补录，早于72h起点。** [论文](https://arxiv.org/html/2610.20572v1)；[官方实现91af0ce](https://github.com/kzhao5/CriticOPD/tree/91af0ce7d7d8cd0539f897441ef617abab64091e)。已静态核查，未训练。

§2–3指出teacher顺着学生错误prefix继续写时可能无法恢复，尽管仍能指出错误。CriticOPD仅对错误/截断轨迹请求看参考答案的teacher critique，保留到最后一个被批评段落，学生看critique重写后缀；训练序列剥离critique，仅重写后缀算loss，teacher scoring仍看critique，student scoring不看。**生成后缀与评分上下文不同，作者明确它是surrogate，不是reverse-KL的无偏梯度。** 正确或无有效critique轨迹仍普通OPD；推理无需critique。

§4/Table1、AppC/D：Qwen3-0.6B/1.7B non-thinking，teacher Qwen3-4B-Instruct-2507，DAPO-Math17k、batch128×1、LR1e-6、response16384、critique2048；主1.7B每run4×A10080GB（2学生/2教师），部分seed4242基线用8GPU，cost用同4GPU的43/44两seed。六数学任务均值OPD34.80→38.86、50.08→54.65，三seed均值/标准差；相对RelayOPD只+0.76/+0.97pp。Minerva1.7B 37.44仍低于OPSD-T37.53；非数学平均仅70.47→70.69、两code任务回退。评估AIME/AMC每题32样本，其余8样本，最多32768token，不是greedy准确率。

独立validation选checkpoint：seed4242选步数后沿用于所有seed，CriticOPD1.7B40步 vs OPD80步。AppD的17.2 A100 GPUh、0.42倍总成本是达到各自所选checkpoint；同40步cost约0.96倍，且后期每步critique/重写更慢，不能宣称每步省58%。critique错误真值判定依赖模型judge，标注错误中73–81%被判断真实。

实现对应：`overlay/verl/experimental/agent_loop/critic_opd_agent_loop.py::CriticOpdAgentLoop._run`对重写构造prefix0/suffix1 mask、剥离feedback，并通过`critic_fb_prompt_ids/n_kept/n_repair`传teacher重评分上下文；`overlay/opd/scripts/criticopd/train.sh`启用after/last/fb/参考答案/2048配置，与Python通用默认不同。**建议小规模复现**：必须同时验证teacher/student prompt隔离、错误前缀mask、grade与截断fallback；先限可验证任务，不扩写到无参考答案开放任务。稳定ID：`arxiv:2610.20572v1`。

### CCE-OPD：对比完整与遮条件题目，修正教师对错误prefix的偏好

**2610.21195v1，2026-10-15T03:55:34Z；补录。** [论文](https://arxiv.org/html/2610.21195v1)。正文未链专用实现，按题名补搜未找到官方代码；机制未代码验证。

§4使用同一student prefix，让冻结teacher分别看完整题与遮掉一个关键条件的题，将两个分布log差clip到±5、乘0.2后重权完整teacher分布；support为两边top16并集。对完整teacher全词表entropy≥0.8nats的response位置增加forward-KL辅助项，普通sampled-token OPD仍作用于全部轨迹。MATH7498题中只3402题通过DeepSeek-Flash annotation和语义复查，其余4096题仅OPD；student rollout始终看完整题。不是用遮条件题训练学生答错再纠正。

§5/Table3、AppD：Qwen3-8B→Qwen3-0.6B/1.7B Base；六数学任务8响应、T1/top-p.8、8192上限。对OPD avg@8宏平均+1.60/+1.95pp，1.7B AMC23 35→40；相对每task最强competitor仅+1.33/+1.45pp（0.6/1.7B）。**avg@8是样本平均正确率，pass@8覆盖提升不一致**：0.6B AIME25 OPD13.33→10；entropy gating让1.7B Minerva avg@8提高2.48但pass@8下降0.74。200个人工构造诊断state的invalid preference27%→16%不是部署失败率。

AppD/Table10训练为AdamW3e-6、cosine、weight decay.01、clip1、每update1个PPO epoch，训练prompt/response各2048、T1/top-p1、thinking关闭；8192是评估response预算。正文未列可复算硬件成本；额外teacher view、条件标注和review有开销，缺多seed训练方差及等总算力对照。β=0同查询/同support辅助loss才是重权的匹配对照，不能把全部增益归给log比值。**建议深读/小规模复现**：先审遮条件是否保留请求、条件语义与标注隔离，再分别报avg@k/pass@k、teacher forwards和训练seed；β与entropy门槛需目标任务复核。稳定ID：`arxiv:2610.21195v1`。

### Token-Level Teacher Selection：两教师逐token路由能组合技能，但相容性与置信度谁更好依任务而变

**2610.21234v1，2026-10-15T04:26:33Z；旧缺口补录。** [完整论文](https://arxiv.org/html/2610.21234v1)。本次未在正文外链找到专用公开实现，未代码核查。

§3在学生自己生成的prefix上，为两教师计算softmax权重；对照人工section routing、教师负熵routing，以及学生–教师Hellinger距离/末层feature距离routing，再用sampled-token advantage或Top-K KL蒸馏。feature距离适用于共享结构，异架构用分布距离。理论关于可存在更优router与特定feature模型/成功集合覆盖的结论，**不保证任一启发式在一般任务胜过固定teacher**。

§5/Tables1–3、AppF/G：Qwen3-4B学生，300updates、batch128×1rollout、AdamW1e-6，采样T1/top-p1；数学×Triton由8077训练pair组成，测试30 AIME题×100 KernelBench L2组成3000pair，输出上限16,384。Hellinger joint correctness pass@16为29.30，对math-teacher OPD20.12；pass@1仅4.37 vs2.84。**3000组合不等于3000独立数学/kernel题，fast0只测正确性，不测kernel加速。**

Triton-only学生thinking开，教师Qwen3-30B-A3B-Thinking和Dr.Kernel8B，100 L2任务/16samples、32,768输出上限；最佳变成entropy routing，pass@1 22.44、pass@16 82，对固定Dr.Kernel OPD4.50/49，Dr.Kernel本身12.63/78。算法代码25,276 Eurus训练prompt、HumanEval+164/MBPP+378题：feature routing HumanEval+ pass@64 94.14 vs OPD93.80，但pass@1 80.03低于80.90；是小幅覆盖增益，不是全面更强。AppF未给可据以复算硬件成本的设备清单。

**建议小规模复现**：先对齐两教师推理模式、tokenizer/feature可比性、总teacher forward成本和验证集温度搜索；分开报告pass@1、pass@k、生成预算与fast0/速度。本文扫温度与loss实现，且主实验缺独立training-seed方差，最佳routing策略应在目标任务重新选。稳定ID：`arxiv:2610.21234v1`。

### SCOPED：先独立训练视觉能力专家，再用教师专享skill约束多专家OPD

**2610.22217v1，2026-10-15T17:01:41Z；旧缺口补录。** [完整论文](https://arxiv.org/html/2610.22217v1)；[官方实现基线274949a](https://github.com/NEUIR/SCOPED/tree/274949a1bf9db3e9069c4f798c812b266433097c)。仅静态核查，未GPU训练。

§3先在感知、OCR/chart、符号推理、多图四个subset独立RL专家，把训练轨迹按reward分成功/失败，分别反思再合成每能力一个procedural skill；OPD时**教师看原问题+对应skill+学生prefix，学生仅看原问题**，无需推理时附加skill。`src/verl/verl/experimental/teacher_loop/teacher_manager.py::AsyncTeacherLLMServerManager._get_skill_token_ids/_inject_skill_as_user_suffix`证实teacher-only user suffix；`compute_teacher_logprobs_single`剥离加长后的teacher prompt并对齐学生response位置，`trainer/distillation/losses.py::compute_distillation_loss_reverse_kl_estimator`按response mask计算sampled KL。没有把skill当system消息或混入学生输入；代码允许单行空skill显式关闭，论文统一skill设置与自定义行级设置需区分。

§4–5/Tables1–2、AppA：Qwen3-VL4B/8B，37,999训练例/四能力近等量，A80080GB、BF16。专家3epochs、128prompt×8responses、prompt/response8192；OPD3epochs、256×1、prompt12288/response8096，LR均1e-6。八benchmark均分8B67.30→73.13，Mix-RL69.31，无skill OPD72.39：**完整增益中，skill相对已有多专家OPD只+0.74pp，不能把全部+5.83归给skill。** 4B66.06→69.96，但MMSI30.20→29.40；8B VisPuz46.40→46.58几乎不变，Mix-RL的WeMath82.90还高于SCOPED82.47。

**边界/建议复现**：是先训四专家再蒸馏的额外预算，未给等总算力Mix-RL对照或多seed置信区间；八任务简单均分混合accuracy和OCR micro-F1，不等于通用VLM排名。梯度cosine、轨迹embedding相似度、attention hit rate支持作者的干扰/regularization解释，不能独立证明reward hacking已消除或注意力即因果依据。优先对比同专家的无skill OPD、技能提取成本/验证集隔离与per-task回归。稳定ID：`arxiv:2610.22217v1`。

### MiMo RL环境删96项：必须固定任务集合，剔除原因未披露

**XiaomiMiMo；10-18T12:40:59Z数据、12:41:01Z映射修订。** [code变更](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss/commit/6b9046cc42d85c2cf3d6dd3aa9d007897b9a5ccf)、[映射变更](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss/commit/971cab2ab421fd88db29770db0d0e52014b63615)。本轮重新打开info/history/卡/tree与两commit，SHA与10-19未整合稿一致，主窗事件相对上次成功归档未报。

9-25已有Apache2环境集合本次code **2698→2602**；上轮对两版Parquet完整记录、多重集与映射ID核验为**恰删96、零新增、所有保留记录一致**，mapping3764→3668；该计算证据保存于10-19`models-mimo-diff.json`，本轮未重复下载计算。官方commit称excluded，未说明删项动机，不猜去污染/授权/质量改善。Docker镜像是否同步、任务可执行性未验。建议固定旧新SHA及删项ID，避免把训练任务集合改变归因为算法进步。

稳定ID：`hf-dataset:XiaomiMiMo/MiMo-V2.6-RL-oss@971cab2ab421fd88db29770db0d0e52014b63615`。

<a name="multimodal"></a>

## 多模态训练

### MSP：用单真实模态生成另一模态的文本代理，省的是audio–vision配对而非所有配对数据

**2610.20965v1，2026-10-14T21:41:32Z；旧待办补录，早于72h起点。** [论文](https://arxiv.org/html/2610.20965v1)。正文及题名搜索未找到专用公开实现，机制未代码验证。

§3先独立训CLAP音频/MetaCLIP视觉512维feature→LLM的MLP projector，之后Phase1模型从唯一真实anchor生成另一模态caption并缓存。proxy经相应text encoder和ridge线性bridge对齐至模态空间，与真实feature使用相同projector/单token槽位；Phase2训练projector+Qwen3-8B LoRA，推理换回两个真实模态、无proxy生成。每sample只用一个真实模态；**仍需模态–文本数据、预训练contrastive encoder，以及每模态1000 caption–modality校准例（500fit/500holdout）**，不能称无paired supervision。

§4/Tables1–4、AppD/I/L：Qwen3-8B LoRA128/256/512，冻结encoder与bridge；音频用AudioCaps/WavCaps/Clotho-AQA等，视觉LLaVA665k/VQAv2及视频抽帧。每run单H10080GB、BF16/ZeRO2、batch16、LR5e-5，两phase各1epoch，先2000步只训projector；seed4242。rank512 AVQA85.58、MUSIC-AVQA65.65；**rank256 OmniBench43.26仍远低于Qwen2.5-Omni56.13**，跨论文/模型基线并非相同训练配方控制。

AppI更有价值的对照：Phase1→仅QA→完整MSP，AVQA70.21→77.46→85.58、MUSIC45.75→55.18→65.65；不能把Phase1到最终的全部15.37/19.90pp都算proxy composition。真实模态替换为learned-null/乱序等token匹配控制均变差，支持模型使用内容而非只读位置。AppL另一个4B且直接在AVQA训练的paired upper bound87.45 vs MSP86.00，**不是上述8B zero-shot的同设置上限**。仅单全局token/视频4帧均值限制细粒度时间/空间任务，未测高token或更大规模。**建议观察并复现缺配对数据场景**：分别验bridge、QA格式和proxy成分，补训练proxy退化控制；隐藏态可分而输出logit相近是关联分析，不能证明唯一对齐机制。稳定ID：`arxiv:2610.20965v1`。

<a name="inference"></a>

## 推理优化

### DLite：草稿端不留历史KV，用目标hidden与轻量顺序头做局部预测

**2610.23787v1，2026-10-16T17:18:54Z；补录。** [论文](https://arxiv.org/html/2610.23787v1)；[官方实现07aece6](https://github.com/SJTU-DAI-Sys/DLite/tree/07aece6deefdd25f73e798c45761b99b2ae57491)，静态核查，未跑H800。

§4将历史建模交回target：当前target多层hidden加depth embedding，anchor+mask本地位置一次并行预测；低秩RNN-style head仅恢复块内token依赖，再投影词表。target自己的KV仍保留，所谓history-free仅指drafter。训练先同结构但有历史KV的teacher CE8epochs，student forward-KL2epochs，再直蒸馏target CE+L1分布差6epochs；这不是免费去KV。

§5/Tables2–4、AppA：target Qwen3-4B/8B，draft5层、宽2560/4096、取12/14层hidden、head rank320/512、K7；约1.9M prompt target重生成，seq4096、batch32、AdamW4e-4，32×H800训练；服务实验H800但正文未完整披露各表GPU数量/精度/运行方差。Qwen3-8B LongBench-v2四类对vanilla decode为2.32/2.74/2.53/3.96倍，对照DSpark为1.98/2.28/2.02/3.02；SWE-bench2.74倍vsDSpark1.60。**不是训练吞吐提升，也不是单任务得分涨2.74倍。** 高并发LCB8B c1为3.95倍，c64只2.13倍；Alpaca c64为1.63倍。DFlash2/DFlow部分数字引自外部，不统一视作同机重跑。

**工件差异已核**：[4B config@88b2f1e](https://huggingface.co/SJTU-DAI-Sys/Qwen3-4B-DLite/blob/88b2f1ee8018e8adbeaf679a20167f8feb36915e/config.json)实际`chs_num_layers=14`、`architecture_version=dlite_v1`，[8B config@0b00176](https://huggingface.co/SJTU-DAI-Sys/Qwen3-8B-DLite/blob/0b001769556a8df6347f20eb647fdbe63a8abf29/config.json)为12、dlite_v2，和论文4B/8B的12/14顺序不同；两者5层、宽2560/4096、rank320/512、BF16、block_size8（anchor+7proposal）。上面性能标为论文实验，不能当已发布checkpoint复现。

实现对应：`dlite/modeling/dlite.py::DLiteDraftModel.initialize_inference_condition/update_inference_condition`返回/保留空history；`_apply_chs_depth_embedding`按target layer ID加嵌入；`dlite/modeling/sequential_head.py::DLiteSequentialHead._compute_step_latent`将上token嵌入和RNN state门控融合后结合parallel hidden，避免先为每位置重复全词表投影。论文缺完整服务尾延迟/SLA和多模型独立复现，训练/推理成本需分开。**建议复现长agent上下文**：锁target、采样、并发、长度与验证规则，分别计target KV、draft KV、hidden提取/传输及顺序头开销。稳定ID：`arxiv:2610.23787v1`。

### Acceptance Is Not Speed：lossy投机接受长度更长，经常换来更长输出与质量损失

**2610.23818v1，2026-10-16T17:41:19Z；补录。** [论文](https://arxiv.org/html/2610.23818v1)；作者给匿名代码入口，本轮[访问401](https://anonymous.4open.science/r/lossy-token-eff/)，未实现核查；以下为作者评估，不是本轮复现。

§2–3不提新drafter，而在vLLM0.26.0 rejection sampler重实现五类training-free relaxed规则，四target/12对target–draft、六数据集，Ndraft6、T1/top-p1、batch1、每case每arm一次；H100 PCIe/SXM多服务器，约750 GPUh。主对为GPT-OSS20B+nebius EAGLE3及Qwen3-8B+RedHatAI EAGLE3，另含P-EAGLE/DSpark/独立小模型。matching接受长度只是best-effort，不同method无法在每组精确匹配。

Table3与§3：最松设置360cell中248输出更长，长度倍率中位1.08/均值1.54；114cell verifier rounds反而更多，300有可执行/确切评分的cell中198比lossless降≥2pp。接受增长要除以输出长度膨胀才能估算round收益；时间还受每round提交token数、节点差异影响。primary pairs时间比中位1.01，即没有中位加速，不能推所有lossy规则都无用。温和设置和Qwen推荐T.6下部分规则不再膨胀，说明采样是重要边界。

§4/AppF对仍未结束reasoning的1132条强制闭合轨迹，正确126→439，同时22条原正确变错，说明部分额外token在拖延作答；这不是可以普遍无损早停的证明。AppC披露首服务器fresh/reused混合且时间噪声、其他节点异速；AIME30题区间宽、部分规则在第二sampler path只改acceptance未改residual，均限制跨实现结论。**建议立即采用评测口径**：同时报告完成长度、总wall time、verifier rounds、截断率和任务质量；接受率、accepted tokens/s、单round收益均不能替代端到端有效完成率。稳定ID：`arxiv:2610.23818v1`。

### Hy-Memory GGUF：新增官方本地工件与独立量化测试，非新训练模型

**腾讯混元；仓库创建10-19T07:52:21Z，GGUF上传11:44:16Z，最终使用说明13:38:32Z。** [固定模型卡](https://huggingface.co/tencent/Hy-Memory-30B-A3B-GGUF/blob/7778309f5109ec6e86820c687e638b9589b541a9/README.md)、[上传commit](https://huggingface.co/tencent/Hy-Memory-30B-A3B-GGUF/commit/e22324b35a9f0a90d88f9df6f559c00fe7cbf0dd)。继10-15的memory-writer训练模型、10-16 FP8后，**GGUF量化与本地部署是独立工件增量**，Apache2.0。

Q6_K imatrix/Q8_0/Q4_K_M imatrix分别24.71/31.99/18.24GB。作者本轮新增独立GGUF表：LoCoMo Judge **154/262=58.78%、151/262=57.63%、148/262=56.49%**。条件是Mem0 **2.0.19**、固定20% LoCoMo（2 users/262题/59 memory chunks）、top_k10、每chunk10turn、memory温度0.7、no-think、4096维`qwen3-embedding-8b-for-online`、答案与judge均`gpt-5.6-sol`、llama.cpp `5aeac37a...`、F16 KV。样本很小，无重复seed/置信区间；不能由Q6领先三题推出量化普遍优于Q8，也不能与旧卡全套跨harness宏平均55.49混用。

三工件作者称通过check-tensors和真实生成，但这是作者验证，本轮未下载全部GGUF/运行。卡没有同一llama.cpp/harness下BF16基线，也未披露imatrix语料、校准细节、硬件、实际显存或延迟。建议固定版本复现memory bank写入，不只测答案模型；先用Q6、保留Q4/Q8对照，并固定judge版本。单slot `ctx-size65536`与双slot`131072`的总context含义已解释，应按每请求64K配置。

稳定ID：`hf:tencent/Hy-Memory-30B-A3B-GGUF@7778309f5109ec6e86820c687e638b9589b541a9`。旧FP8独立质量验证仍未关闭。

<a name="engineering"></a>

## 开源工程

### 正式版本

#### vLLM 社区：v0.32.0 发布，升级先核接口与默认值

2026-10-18T06:50:48Z；稳定 ID `github:vllm-project/vllm:release:v0.32.0`；tag `5df63a7203cdb9328d824a27b682e77c7b9928be`，正式非 prerelease。[发布说明](https://github.com/vllm-project/vllm/releases/tag/v0.32.0)。

已打开完整 notes 和 tag ref。对训练/部署直接相关：Python 最低 3.11；Transformers >=5.16.1,<5.20.0、tokenizers>=0.23.2；CORS 默认不再允许所有来源；compile sequence parallelism/AsyncTP pass 和相应 PassConfig 参数移除（model-level SP 不受此条影响）；`mamba_cache_mode=all` 不再可用。新默认包括 shared-memory tensor arena、ROCm MRV2、DeepEPv2 hybrid；`max_tokens` 改为输出上界而非预留上下文。notes 还汇总 token-level weight-version spans、睡眠释放 workspace/状态、DeepSeek replay、Mamba prefix checkpoint 等此前两周提交，应按“正式版本汇总”去重，不能把每项算作本窗新实现。

**建议：采用前先迁移配置并固定镜像做 trainer/rollout 数值及 sleep/weight-sync 回归。** 本轮只核发布声明/资产元数据与 tag，未下载 wheel、检查各容器内容或逐个复审整版数百 PR；notes 中的加速数字不作为本轮独立性能结论。#61938/#60735 等下列主线事件与 tag 分开记账。

#### SGLang v0.5.22：发布日期与tag快照分开

2026-10-19T02:55:19Z；稳定ID `github:sgl-project/sglang:release:v0.5.22`；正式非prerelease。[官方说明](https://github.com/sgl-project/sglang/releases/tag/v0.5.22)。本轮核release JSON、annotated tag `35ce313293052f8362e0829e7512bc9bc3d336c9`及peeled commit `baa9befa381511cd4d56f83980951e5a45fffc6d`。**tag创建于10-15，release发表于10-19；下面10-18/19主线功能不自动属于此包。**

正式notes汇总950 PR、245贡献者；涉及DSv4.1/AMD DSpark、HiSparse speculative、GLM DCP+MTP、InstantTensor加载、PD调度及hybrid模型spec+structured-output恢复错误修复，并要求Python>=3.11。2.3x/1.9x/10x等为发行说明在各自特定基线上的摘要，未逐项重新复现实验；它们涵盖此前已报实现，按版本化汇总而非本窗新增机制计数。建议先以固定tag验证模型、HiCache/SSM和P/D两端，再挑选tag之后所需主线修补；未下载wheel或验证所有平台镜像内容。

### Megatron-LM

#### #8230：MFSDP v2 expert-parallel checkpoint 与 compute-weight 恢复

来源：[NVIDIA/Megatron-LM #8230](https://github.com/NVIDIA/Megatron-LM/pull/8230)（2026-10-18，main；`github:NVIDIA/Megatron-LM:pr:8230`）。

此前 BF16 EP=2 能训练两步，却在首次 save 因多 device mesh 抛 NotImplementedError。现在 optimizer state 直接用 MCore 全局参数名；TE>=2.18 走直接序列化，TE<2.18 保留 empty-shard workaround，按每个参数的 dense/expert-DP mesh 收集缺失键，placeholder 只在旧 TE 分支生成/剔除。model state 加载后把恢复的 FP32 main weights 同步到 BF16/量化 compute buffers，避免第一步 forward 读旧 storage。

已读 `FullyShardedOptimizer._add_empty_shard_state/sharded_state_dict/load_state_dict`、adapter `load_state_dict` 以及 standalone checkpoint loader。作者 4 GPU 14 测试覆盖 EP2/EDP2、EP4/EDP1；小模型 BF16 EP2 四步中断/恢复的 84 个 model/Adam tensor 和 metadata 共459项精确一致。**旧 TE 兼容分支未加/运行测试，端到端是 tiny mock-data hybrid，并非长训/弹性重分片认证。建议 EP 用户优先复现 save/restart 后首步 loss 和 optimizer state。**

#### #6981：GPT loss-mask 缓存不能别名首个样本

来源：[NVIDIA/Megatron-LM #6981](https://github.com/NVIDIA/Megatron-LM/pull/6981)（2026-10-18，main；`github:NVIDIA/Megatron-LM:pr:6981`）。

`GPTDataset.__getitem__` 首次缓存 loss_mask 原 tensor，随后原地把 pad 标签置零，使首样本的零位永久写入缓存；后续虽 clone，clone 的也是污染基底。换成缓存 `loss_mask.clone()`。两个 dataset 实例若首个服务样本不同，后续同一 index 也会产生不同 mask，因此 checkpoint 本体相同也可能恢复后立即梯度分歧。

已读单行生产 patch 与定向回归：masked sample 先访问导致后续 index0 mask 不一致，修改后通过。真实训练默认丢弃末尾 partial sequence，实际触发还依赖 pad-id 出现在数据中；validation 可保 partial，mock-data ramp 明确包含 pad id。SFTDataset 自有 __getitem__ 不在此修复内。**建议把“样本访问顺序不同仍返回相同 loss_mask”纳入续训排查，不外推所有训练必受影响。**

#### #8180：deterministic mode 接受 NCCL 自选，但有硬件与版本条件

来源：[NVIDIA/Megatron-LM #8180](https://github.com/NVIDIA/Megatron-LM/pull/8180)（2026-10-18，main；`github:NVIDIA/Megatron-LM:pr:8180`）。

未设置 NCCL_ALGO 仍填 Ring；显式空串改为保留，让 NCCL 自选，rank0 警告比特一致性依赖所选算法；Ring,NVLS 也被允许，裸 NVLS/Tree/纯空白仍拒绝。Hopper 架构9对空串与包含 NVLS 的列表整体拒绝，代码并不查询驱动/Fabric Manager 修复版本。创建第一个 process group 后通过 libc getenv 检查空串未被 PyTorch 偷换，不能只读 Python os.environ。

已读 `apply_determinism_env/apply_determinism_to_args/assert_nccl_algo_still_empty` 与 init 调用。作者 B200/NCCL2.27.7 单节点 multicast replay 和64GPU固定软件栈重复/节点平移比特对齐；H100 空串在初始化前被拒，unset 控制正常。**建议保持 Ring 默认，需要吞吐时在固定 topology/driver/NCCL/tuner 上显式实验。不能从64GPU几种 dense layout 证明任意 MoE/网络/算法可复现；GB200 CI 专用测试当时尚未执行。**

#### Full-iteration图池复用FP32梯度转换存储

来源：[NVIDIA/Megatron-LM #7775](https://github.com/NVIDIA/Megatron-LM/pull/7775)（2026-10-19，main；`github:NVIDIA/Megatron-LM:pr:7775`）。

eager optimizer原本在forward/backward图池外另分配FP32 optimizer-gradient shards。现在把梯度转换也捕获到同一池，在前反向结束后复用scratch；zero_grad后重新绑定输出storage，输入地址变化拒绝重放，training graph reset会重建转换图。覆盖distributed/layer-wise mixed-precision optimizer；optimizer-cuda-graph、precision-aware零拷贝、CPU offload和FSDP保原路径。

已读FullIterationGradCopy及两类optimizer选择门。作者256GPU NT4例最大显存241.10→225.02GiB（-16.08GiB），声称无吞吐退化但未给完整统计表。建议显存受限的full-iteration用户复现，并严格保forward/backward→转换→optimizer顺序；不能拿这一个模型数字当所有拓扑节省量。

#### MFSDP v2预取按模块调用次数区分后继

来源：[NVIDIA/Megatron-LM #8207](https://github.com/NVIDIA/Megatron-LM/pull/8207)（2026-10-19，main；`github:NVIDIA/Megatron-LM:pr:8207`）。

A→B→A→C中两次A需要不同prefetch后继，原IndexedOrder禁止重复模块。ExecutionOrder以弱引用保每次出现，唯一模块可按identity定位，重复模块必须匹配cursor，next_item只向前看而不消费；预算仍由caller控制。

已读order及prefetch consumer。作者24条生产VPP schedule trace与旧行为一致，8H100上36个focused cases通过，完整CI未核；运行时自动录制执行顺序仍是后续工作。建议复用层/共享module及VPP用户检查预取序列，不能据此宣称任意动态控制流均已支持。

#### GDN初始状态直接在设备分配，解除CUDA图捕获错误

来源：[NVIDIA/Megatron-LM #8322](https://github.com/NVIDIA/Megatron-LM/pull/8322)（2026-10-19，main；`github:NVIDIA/Megatron-LM:pr:8322`）。

无initial_state时旧torch.zeros(...).to(value)在capture中先分配CPU tensor再搬到CUDA，报operation not permitted。改value.new_zeros，维持张量shape/device契约；给定initial_state路径不变。已读单行生产修补与16-case回归。

作者NeMo26.10 RC5、8GB300原Qwen3.5 9B配方捕获失败→10步有限loss并完成eval，另16原生图和4TE图检查通过；完整current-main pytest尚未跑，GPU实验是RC5同函数补丁。建议GDN图训练核capture、变化输入replay及续训，勿当整版MCore已正式发布。

### vLLM

#### #61938：Triton MLA decode 64位 offset 与 workspace 计费必须同时修

来源：[vllm-project/vllm #61938](https://github.com/vllm-project/vllm/pull/61938)（2026-10-19，main；`github:vllm-project/vllm:pr:61938`）。

MI355 在长上下文/大 graph batch 时，split-KV logits 的 `cur_batch * stride_mid_ob` 超过2^31；第一行乘法现先转 tl.int64，覆盖普通/grouped stage1 stores 和 stage2 loads。第二个独立问题是 logits workspace 在内存 profiling 后才保留，KV cache 被算得过大；新增 backend reserve_workspace hook，由 MLA profiling forward 提前预留最坏 decode rows × heads × splits × (kv_lora_rank+1)。

已读 ops、TritonMLAImpl/builder 及 profile hook patch。作者 MI355/TP1：Kimi1M/LongCat320K 由 access fault/OOM 变为可服务，384并发零失败；只补一个缺陷仍分别 OOM 或 fault。8.6GB logits 回归触发旧 fault、修改后通过。**建议长上下文 ROCm MLA 优先回归，并接受 KV 容量因真实计费下降。** DCP 在特定 backend 下预留可能比 builder 更保守；32K吞吐数字来自指定模型/5次实验，不能外推所有硬件。该条合入晚于正式 release。

#### #60735：DeepEP HT × DBO 补跨 stream tensor 生命周期

来源：[vllm-project/vllm #60735](https://github.com/vllm-project/vllm/pull/60735)（2026-10-18，main；`github:vllm-project/vllm:pr:60735`）。

DBO 的 dispatch 输出在 comm stream 分配、compute stream 读取，combine 输入反向跨 stream。等待 dispatch 完成并不阻止 allocator 在另一 microbatch 的 comm 分配中复用已释放 Python tensor 的内存。新增 `_dbo_record_stream`，分别标记输入、dispatch 输出、combine 输入；DBO关闭时 no-op，已有 ROCm host-sync 路径跳过相应输入标记。

已读完整单文件 patch，作者 main基线 GLM5.3 FP8/8H200 DP4×TP2/EP8 启动崩溃2/2，修复后 GSM8K 可完成；32H200补丁移植也有运行记录。**无新增自动化测试能可靠复现原 GLM bug**；原有 DeepSeek小模型 DBO测试会执行修改路径但补前也通过。MRV2 DBO、ROCm及main多节点未跑，重基后未重跑 GLM。每层每microbatch最多9次record_stream和free时event的独立成本未测。**建议采用并以真实 MoE/backend 做压力和数值回归，不把流水线等待当 allocator 寿命证明。**

#### #62129：MRV2 × speculative × PP warmup 所有 stage 必须同轮数

来源：[vllm-project/vllm #62129](https://github.com/vllm-project/vllm/pull/62129)（2026-10-18，main；`github:vllm-project/vllm:pr:62129`）。

额外 greedy warmup 原来由 rejection_sampler 是否存在决定，只有末 PP rank 有 sampler，前面 stage 不做 forward，末 stage 等 activation 死锁。改用 speculative_config 条件，让每个 stage 同步执行。已读 `gpu/warmup.py` 与两个定向测试。作者34pass，GLM4.7Flash TP2/PP2 BF16 MTP1含JIT/graph warmup及1024 token生成成功；**未测任务准确率**。建议 PP speculative用户优先补启动回归。

#### #61002：ROCm hybrid MLA 的非整除 P/D block 使用 gcd 单位配对

来源：[vllm-project/vllm #61002](https://github.com/vllm-project/vllm/pull/61002)（2026-10-19，main；`github:vllm-project/vllm:pr:61002`）。

Kimi hybrid 的 SSM 页随 TP分片，P2944/D5760 token block 互不整除，原 handshake断言。新 `_split_remote_blocks` 在 hybrid MLA **pull** 路径把远端看成 gcd128的23个子block，由本地45子block配对并对未传 tail清零。最多128份，要求远端紧密可分 stride、无 PLE layout；不满足就保留原逻辑拒绝。push WRITE未验证，不走此代码。

已读base_worker几何/调用patch与28个失败转通过回归；作者MI355 AITER MLA、双向TP异构运行及GSM8K报告，等block控制性能近似相同。**D端prefix caching仍拒绝相应hybrid组合，不能称任意异构TP/P-D兼容。建议先核transfer mode/layout，再做两方向restore与tail校验。**

#### #52477：结构化输出的 grammar mask 以GPU实际logit数为准

来源：[vllm-project/vllm #52477](https://github.com/vllm-project/vllm/pull/52477)（2026-10-18，main；`github:vllm-project/vllm:pr:52477`）。

scheduler 为每请求固定 `num_spec_tokens+1` mask block，MRV2 按设备端cu_num_logits应用实际活跃位置；避免 adaptive verification 零预算/bonus-only后CPU scheduled-draft数量失配。batch sharding按整block切，MRV1还原packed布局。已读manager、GPU structured_outputs、batch_shard及warmup patch。作者102pass/1xfail，**GPU kernel和MRV1/MRV2真实生成测试待跑**。建议对adaptive verification+JSON schema+batch sharding组合验证合法性及逐token行为，勿宣称GPU端到端已认证。

#### #61819：ROCm DeepSeek-V4 非融合共享专家与MegaMoE重叠

来源：[vllm-project/vllm #61819](https://github.com/vllm-project/vllm/pull/61819)（2026-10-18，main；`github:vllm-project/vllm:pr:61819`）。

只有未融合shared expert且stream启用、step不超过默认256tokens时才走aux stream并与router+routed MegaMoE重叠；关开关、大prefill、shared expert已融合都走相应原分支。已读AMD model的构造gate和 `maybe_execute_in_parallel` 调用，库内实现只按其调用契约引用。

作者8MI355X DP8/EP8、AgentX1200s单次测量：c128 P90每用户27.13→27.87tok/s（+2.7%），c64 48.26→51.33（+6.4%），后者总tok/s/GPU却-1.5%。**建议特定ROCm场景复现；不能只摘用户交互增益当总吞吐提升，单次/不同job也限制统计结论。**

#### PCP接通APC与encoder侧SWA replay，边界仍窄

来源：[vllm-project/vllm #62183](https://github.com/vllm-project/vllm/pull/62183)（2026-10-19，main；`github:vllm-project/vllm:pr:62183`）。

在各PCP段分别重建SWA尾部并按rank-major聚合可缓存slot，使已有压缩KV/indexer key可复用；所有rank即使无replay行也参与metadata collective。decoder尾部裁剪仍关闭。已读attention/model-state关键patch。

作者两台8H20分别跑baseline/candidate，TP2/PCP4，80次1152输入/1输出的重复前缀测试TTFT 640.73→386.73ms（-39.64%），cache hit79.29%；GSM8K全量96.97→97.35%，配对p=0.1797，不能称质量改善。仅NVIDIA MRV2/eager/sync，PP=DP=DCP=1；connector/spec/microbatch/MegaMoE仍不支持。建议以高复用前缀和真实长输出分别评估。

#### 异构P/D按每类KV与SSM的实际分片传输

来源：[vllm-project/vllm #61726](https://github.com/vllm-project/vllm/pull/61726)（2026-10-19，main；`github:vllm-project/vllm:pr:61726`）；[vllm-project/vllm #58471](https://github.com/vllm-project/vllm/pull/58471)（2026-10-20，main；`github:vllm-project/vllm:pr:58471`）。

MoRIIO WRITE旧MLA P2→D4只向首个目标rank写，其他rank永等；当前patch进一步按attention/Mamba布局描述建立fan-out/slice/拼接映射，并等所有传输状态与通知。文档要求TP互为整数倍、KV heads与TP可整除、相同token block；hybrid MLA需显式统一到较大自动block-size。滑窗WRITE和非DSpark hybrid spec仍不支持。作者单MI355X节点DeepSeekV2-Lite P2/D4，128请求及1319 GSM8K均完成；跨节点和全部hybrid组合不由此实测证明。

NIXL #58471则为MLA target下的GQA draft另建head-sharded TPMapping，防各decode rank都读draft head0；pull支持P≤D，P>D/DCP拒绝，push异构TP仍拒绝。作者H200 GLM5.3+DFlash2旧0/80→80/80，accept length3.221 vs本地3.253；后续B300持续310请求零错。最终代码经过generic helper重构，部分E2E数据先于最后修订；CUDA graphs、fp8 draft KV、MultiConnector等未验。建议分开核target/draft/SSM几何与传输方向，不能把一个connector的修复泛化给另一个。

#### 滑窗页回收与EAGLE共享前缀分别补寻址和保留边界

来源：[vllm-project/vllm #60767](https://github.com/vllm-project/vllm/pull/60767)（2026-10-19，main；`github:vllm-project/vllm:pr:60767`）；[vllm-project/vllm #62206](https://github.com/vllm-project/vllm/pull/62206)（2026-10-19，main；`github:vllm-project/vllm:pr:62206`）。

#60767在每步gather出的block table中把完全低于窗口的页替成null block，防旧页被其他请求复用后含FP8 NaN，masked tile仍经0×NaN污染输出。只改forward副本，不动真实write slot；CP>1关闭该mask。已读GPU gather路径；作者DFlash2/B300复现出NaN与接受率下降，建议64token页配FA4 128-key tile者重点回归，不能把mask当CP布局已修。

#62206把稀疏prefix retention中EAGLE SWA shared-junction尾部向下再保一block，使仅共享部分前缀的兄弟请求可命中。Step5/MTP示例原连续0 hit，修后第3起544 hit tokens；代价是每junction每SWA group多存1block。真实例在Step5支持分支上跑，CPU620pass外有2项需GPU未执行。建议同时核命中量与resident KV容量。

#### 多模态encoder双向look-ahead死锁与队首阻塞分别修复

来源：[vllm-project/vllm #60553](https://github.com/vllm-project/vllm/pull/60553)（2026-10-19，main；`github:vllm-project/vllm:pr:60553`）；[vllm-project/vllm #59030](https://github.com/vllm-project/vllm/pull/59030)（2026-10-19，main；`github:vllm-project/vllm:pr:59030`）。

#60553移除encoder输入释放/下一item admission两侧反向look-ahead偏移：两幅图只能各自放入cache且间距<2L时，旧路径永远0 scheduled tokens。作者Qwen3.5-4B/MTP1、GB200、16384-token encoder cache，gap0/1原300秒无完成，修后均完成且贪心文本等同no-spec；已读scheduler/cache生命周期patch。不能把它等同于Mamba alignment旧修复。

#59030在waiting request因encoder预算得到0 token时skip/requeue并继续，避免整个队列被队首挡住，保持drain-first和deferred计数。作者248pass/1既有失败，无模型eval。该项不解除请求自身的encoder循环依赖；建议两项一同覆盖双图相邻、encoder饱和及后续短请求。

#### split top-p在概率域保留强制token，避免thinking-budget后反复token0

来源：[vllm-project/vllm #58231](https://github.com/vllm-project/vllm/pull/58231)（2026-10-19，main；`github:vllm-project/vllm:pr:58231`）。

强制</think>设1e9 logit；旧split kernel把概率pivot转回logit，FP32舍入使pivot等于最大值，严格>删除整行。现在在probability域比较并保最大logit，避免全-inf/NaN恢复成token0。适用于CUDA小batch split top-p、spec两runner，也会影响走该Triton路径的non-spec；大batch/active-top-k monolithic路径没有改。

已读完整mask patch。作者A100完整sampler135pass/2skip，MRV2 Nemotron BF16/FP8 patched端到端强制token恢复、212vs65tok/s只是原失败场景的恢复值；V1 patched E2E未重跑。建议用真实top_p默认值和budget边界验证，不能宣称修复另一top_k相关budget issue。

#### cuDNN frontend最低1.31，修图节点仍引用已卸载JIT代码的寿命

来源：[vllm-project/vllm #61465](https://github.com/vllm-project/vllm/pull/61465)（2026-10-19，main；`github:vllm-project/vllm:pr:61465`）。

依赖floor从1.19.1升1.31.0，让CUDA graph持有execution plan；旧plan最后owner被cache驱逐后，graph可跳到已释放/重新占用的kernel代码地址，造成fault或静默错误。已读requirements修补，PR提供coredump/CUDA-GDB及定向reproducer链条。当前FlashInfer0.7.2自身capture-retention已另缓解GEMM缓存驱逐，但不能替代frontend修复。

作者没有released wheel全暴露场景E2E故障率A/B，保留plan的显存成本也未量化；backend版本单独不能判暴露。建议升级构建时核实际解析frontend版本并做大量shape图重放/缓存驱逐回归，勿仅看requirements lower bound。

#### DSv4.1修PP首层漏Engram与cumem混allocator collective hang

来源：[vllm-project/vllm #62208](https://github.com/vllm-project/vllm/pull/62208)（2026-10-19，main；`github:vllm-project/vllm:pr:62208`）；[vllm-project/vllm #62269](https://github.com/vllm-project/vllm/pull/62269)（2026-10-19，main；`github:vllm-project/vllm:pr:62269`）。

#62208在非首PP stage入口的mHC pre之前补执行该层Engram；14,26分区可让layer14此前静默漏算。已读NVIDIA model patch，作者双8H20对照；AMD同类问题此PR未修。建议PP分区落Engram层时逐层核执行，不能只看服务能启动。

#62269预分配receive buffer并与send用同一allocator，解除host-offloaded、DP-sharded Engram在cumem开关下首轮all_to_all_single挂600秒。作者GB200 DEP2/4，改后capture+48/48不同长度请求完成；只改存储来源，未给全面模型数值A/B。建议保留cumem/NCCL_LOCAL_REGISTER控制组排查。

#### 量化输出融合与PDL路由消费者各省一次小开销

来源：[vllm-project/vllm #60118](https://github.com/vllm-project/vllm/pull/60118)（2026-10-19，main；`github:vllm-project/vllm:pr:60118`）；[vllm-project/vllm #59737](https://github.com/vllm-project/vllm/pull/59737)（2026-10-19，main；`github:vllm-project/vllm:pr:59737`）。

#60118让MegaAttn wo_a直接输出wo_b消费的MXFP8数据和128x4 UE8M0 scales，跳独立quantize/swizzle；限per-32 recipe、o_lora_rank%128=0、支持该scale的wo_b且无GEMM-RS。已读完整_o_proj gate。GB200单层测20尺寸bitwise，decode约省0.4–2.0μs，384tokens反慢0.6μs；依赖vLLM DeepGEMM fork dev@727f847，外装旧库会缺新参数。建议先核库pin，再看端到端占比。

#59737为fused MoE top-k启SM90+/CUDA12+的PDL consumer。GB200 MiniMax-M3 3组配对TPOT仅改善0.61–1.16%，纯prefill TTFT反退约0.5%；GSM8K每臂2次不证明质量等价。已读kernel launch flag，适合小batch decode实验，不宜把router微秒收益当整机大幅提速。

#### FLy作为显式有损verification策略合入MRV1/MRV2

来源：[vllm-project/vllm #53987](https://github.com/vllm-project/vllm/pull/53987)（2026-10-19，main；`github:vllm-project/vllm:pr:53987`）。

target entropy较高且后W个draft都通过原生verification时，可跨过一次孤立拒绝，无额外target forward。采用top-k概率子集熵且不重归一，默认top-k3，并非论文full-vocab熵；改变target分布，标准SD仍默认。已读配置及verification关键patch，未逐行审全部12文件。

作者4模型×2平台×6batch的MRV2 decode速度相对standard SD为MI355X1.055–1.310x、B3001.041–1.273x；测试排除prefill/load，质量单次pass@1，部分AIME25少1题，最低retention96.30%。建议仅在可接受质量变化的工作负载显式opt-in，以target-only为质量对照；不能当无损spec加速或通用质量保留率保证。

#### P2P提交错误快速回报、peer fault升级和重连退避已合入

来源：[vllm-project/vllm #60374](https://github.com/vllm-project/vllm/pull/60374)（2026-10-19，main；`github:vllm-project/vllm:pr:60374`）。

NIXL提交异常/ERR现在回TransferDone(False)并收尾；raising state-check立即fault peer，普通失败连续3次才fault，同poll内成功会重置strike。fault后停止新WRITE，等inflight drain、最长10秒再disconnect/re-register；反复故障退避0/1/2…60秒，稀疏流量超时可忘却退避但fault计数保留。

已读transport/session关键patch。作者P2P308pass、tiering554pass/12skip；两H200 pod真实NIXL1.5带故障注入，提交错误20请求1.6–2.3秒本地重算并恢复cache hit。**10秒强制teardown不是远端WRITE已停止的证明；旧consumer quarantine #61667仍open，见后续待办。** 建议采用错误回报与可观测性修复，同时保留跨主机RDMA迟到WRITE/目的块复用验收，不能据此关闭生命周期风险。

### verl

#### #8196：score centering 自动按每token完整support证据选择

来源：[verl-project/verl #8196](https://github.com/verl-project/verl/pull/8196)（2026-10-18，main；`github:verl-project/verl:pr:8196`）。

完整support不再由 configured K、top-p开关或“head质量近1”推断。vLLM provider检查真实FlatLogprobs offset/rank连续、全局ranked-head来源、sample存在性/概率一致性；只有记录整个词表，或真实top-k中出现-inf证明省略token均零概率，并通过unit-mass等校验，才发 Boolean `response_support_complete`。缺offset/rank的legacy records标unknown，自定义logprob_token_ids/greedy不冒充采样分布；手动 `score_centering_top_p` 移除。

trainer逐token complete时在该support上重归一trainer分布，并同时修正sampled scoring；unknown继续“已记录head + full-vocab trainer tail代理”，epsilon尾部正则仍在，可能把质量分给未知采样support外。记录预算不改变真实sampling params，K小于effective top-k仅warning，不决定估计器。

已读 `extract_response_topk_logprobs_with_coverage`、provider.generate、`score_centering_auto_correction/support_correction/ppo_loss`关键patch。作者CPU176pass/10skip，H100 provider/top-p overflow targeted和Qwen0.6B验证；最终Boolean schema followup 19测/静态通过，remote成功结果对应前一head，按维护者要求未为最后有界改动重跑。**建议先看sc_support_aligned/unknown/tail_mass，再复现loss/梯度；不把原作者单seed Countdown400步或不同head结果当本轮最终训练质量/速度保证，Megatron扩展#8199仍独立。**

#### QAT和低精度rollout改为trainer量化、engine只加载/refit

来源：[verl-project/verl #8287](https://github.com/verl-project/verl/pull/8287)（2026-10-19，main；`github:verl-project/verl:pr:8287`）。

Megatron导出HF命名权重后由trainer转compressed-tensors布局，再经checkpoint engine分发；vLLM/SGLang加载预量化权重，避免各rollout worker重复BF16量化。支持FP8_BLOCK/MXFP8/MXFP4/NVFP4 exporter及ModelOpt QAT模式，dense/routed专家可分别配置；不是FSDP的等价新实现，也不是TE原生NVFP4 real quant。

破坏性配置迁移：rollout.quantization=fp8改quantization.linear/moe=FP8_BLOCK，新增ignore/config_file且启动校QAT精度一致；TRT-LLM quantization路径删除。已读config、worker dispatch和exporter/refit关键patch，60文件未逐行全审。作者compileall、pre-commit/类型/配置生成通过，targeted pytest与GPU E2E因依赖缺失未跑。建议升级先迁配置，再跑rollout_logprob_diff和多次refit数值对照；不能把新增模式清单视为全格式硬件验收。

#### VeOmni成为独立uv训练backend，增加GPU/Ascend工作流

来源：[verl-project/verl #8274](https://github.com/verl-project/verl/pull/8274)（2026-10-19，main；`github:verl-project/verl:pr:8274`）。

无需额外fsdp/veomni-sft extras即可--extra vllm --extra veomni；锁VeOmni0.1.12@10a26ac与Megatron core_v0.19.2@4b4acac9。L20×8工作流覆盖两种FSDP/SP配法、MoE导出，另增Ascend变体；公共CUDA13 uv镜像移至dev4。已读lock/install及E2E recipe patch。

作者真实uv离线预取/构建安装通过，H20×8/A800×8 PPO/GRPO smoke产出loss/reward；新L20 CI全绿尚待，Ascend新workflow不等于本轮NPU E2E实测。建议按固定lock和镜像复现环境后再做训练比较。

### SGLang

#### #43325/#43326/#43168：统一内存接纳非对称K/V、SWA与recurrent draft

来源：[sgl-project/sglang #43325](https://github.com/sgl-project/sglang/pull/43325)（2026-10-18，main；`github:sgl-project/sglang:pr:43325`）；[sgl-project/sglang #43326](https://github.com/sgl-project/sglang/pull/43326)（2026-10-18，main；`github:sgl-project/sglang:pr:43326`）；[sgl-project/sglang #43168](https://github.com/sgl-project/sglang/pull/43168)（2026-10-18，main；`github:sgl-project/sglang:pr:43168`）。

原统一池拒绝K/V宽度不同，以及包含SWA/recurrent层的MTP draft。#43325传递真实full/SWA K/V head geometry，非对称仅允许 triton/fa3/fa4；K和V各自检查16字节对齐。#43326按host kind注册 full/swa/mamba，把draft层放入生命周期覆盖其读取的target子池，runner各占不重叠lane；draft window宽于target时尝试full fallback，几何不兼容则decline。SWA draft限制为有逐步window索引的Triton multi-step；recurrent draft状态随target整entry clear/copy/move。

读了placement/binder/config gate关键patch及正文，未逐行审50文件。保留历史统一为target窗口与draft reach较大者；#43168把非页对齐window先向上取整页，避免prefix插入边界后只剩半窗，例如window127/page64额外最多保1页。作者455测试/921subtests等是**早期combined-stack**验证，非每PR独立模型/最终head全套认证；后续fixture/CI metadata也有改动。**建议以MiMo/Inkling的draft层型、实际backend、window/page/多depth graph组合逐项复现，不能当所有backend的通用新默认。**

#### #43333：HiCache direct连续复制不能用于统一池strided rows

来源：[sgl-project/sglang #43333](https://github.com/sgl-project/sglang/pull/43333)（2026-10-18，main；`github:sgl-project/sglang:pr:43333`）。

统一池一条token entry包含所有层，单层view有stride。旧page_first_direct仍把page_size×row_stride当连续有效数据，Inkling可复制实际页的8倍，写邻近host页、其他device层或越界。参数解析在storage-layout normalization后，把unified+page_first_direct改page_first/kernel；MHA/MLA direct入口额外拒绝非packed rows。

已读hicache_hook及两个hostpool保护patch。作者H200三项assertion旧失败新通过，0.35GB Inkling小pool旧版秒崩、修改后服务；**未提供错误输出复现，KL测试未发生host load-back而无检测力。建议优先采用并强制验证真正evict/load-back字节与logits，尤其Mooncake会自动选择direct layout的配置。**

#### #41907/#43816：MLA共享KV key不代表Mamba state也能跨TP共享

来源：[sgl-project/sglang #41907](https://github.com/sgl-project/sglang/pull/41907)（2026-10-18，main；`github:sgl-project/sglang:pr:41907`）；[sgl-project/sglang #43816](https://github.com/sgl-project/sglang/pull/43816)（2026-10-18，main；`github:sgl-project/sglang:pr:43816`）。

file/SeaweedFS分别为Mamba component key添加TP rank/size；普通replicated MLA KV仍共享。file eviction改为每rank负责自身TP分片，rank0另负责shared；目录总占用可超过单rank max_size。旧无rank Mamba文件/对象无法判归属，故不复用。

已读key构造及LRU ownership patch。作者GLM5.3Flash TP4/file L3 16K前缀restore，旧408/544为错rank payload，新544/544正确；SeaweedFS新两rank往返与模块测试通过。**HF3FS等同类backend不由这两PR证明，建议清理/隔离旧state并检查目录总quota，做L3-only恢复而非只看L1命中。**

#### #44930/#44952/#45035/#45037：DSv4.1 replay扩展到CUDA breakable图与PD，伴随wire-state修补

来源：[sgl-project/sglang #44930](https://github.com/sgl-project/sglang/pull/44930)（2026-10-18，main；`github:sgl-project/sglang:pr:44930`）；[sgl-project/sglang #44952](https://github.com/sgl-project/sglang/pull/44952)（2026-10-18，main；`github:sgl-project/sglang:pr:44952`）；[sgl-project/sglang #45035](https://github.com/sgl-project/sglang/pull/45035)（2026-10-18，main；`github:sgl-project/sglang:pr:45035`）；[sgl-project/sglang #45037](https://github.com/sgl-project/sglang/pull/45037)（2026-10-18，main；`github:sgl-project/sglang:pr:45037`）。

encoder SWA replay允许CUDA breakable prefill graph，其他prefill graph及ROCm graph仍拒；padding行无window/不commit。CUDA PD prefill可启encoder replay，target request-window SWA按whole-page列表发送，decode仍paged SWA；PD decode不运行encoder replay。#45035使符合其余兼容gate的CUDA两PD角色**decoder replay默认启用**，ROCm PD仍显式opt-in，`--no-enable-decoder-swa-bounded-replay`保full-prefill。

#45037为DSpark draft SWA传每slot `swa_valid_from`（新REQUEST_ROW state、Mooncake/NIXL），防prefix hit后的短extend把prefill未写的旧slot行当draft历史。读取CUDA pool状态、decode payload、torch/Triton window gather clamp等关键patch；对大PR属选择性审阅。作者GB300 DEP2 PD+Mooncake GSM8K/传输检查；43-token短extend的accept length两轮3.16/2.87→3.85/3.80，**是接受长度而非吞吐**；质量对照97.12/96.89%与新96.74%，不能称逐token等价或无差异。未测Mooncake custom mem pool，decode有额外local前缀时可能隐藏本来有效行（影响acceptance）。**建议同时升级P/D并做短prefix-hit、wire-state及full-prefill对照。**

#### #44948/#45040/#45047/#45042/#45051：replay行必须参与跨rank协议、预算和最小chunk

来源：[sgl-project/sglang #44948](https://github.com/sgl-project/sglang/pull/44948)（2026-10-18，main；`github:sgl-project/sglang:pr:44948`）；[sgl-project/sglang #45040](https://github.com/sgl-project/sglang/pull/45040)（2026-10-18，main；`github:sgl-project/sglang:pr:45040`）；[sgl-project/sglang #45047](https://github.com/sgl-project/sglang/pull/45047)（2026-10-19，main；`github:sgl-project/sglang:pr:45047`）；[sgl-project/sglang #45042](https://github.com/sgl-project/sglang/pull/45042)（2026-10-19，main；`github:sgl-project/sglang:pr:45042`）；[sgl-project/sglang #45051](https://github.com/sgl-project/sglang/pull/45051)（2026-10-19，main；`github:sgl-project/sglang:pr:45051`）。

#44948把late-layer DP row-count CPU all-gather从model forward前移到 `ForwardBatch.prepare_mlp_sync_batch`：否则一rank graph replay跳过Python forward、idle rank eager却进入collective，发生MegaMoE grid-sync hang。新增每extend step、每rank一个int的CPU gather成本。

#45040先从chunk扣最多128 cached replay rows，再向页边界取整extend，避免超过MegaMoE cap后落入不兼容Triton fallback；#45047 admission携带replay_len，选入与commit共用，host load-back改变prefix时重算，计算预算收费extend+replay、KV预算仅页取整extend。#45042 decode转extend清旧encoder_swa_reset，防多fold未计入DP padding。#45051启动要求每attention-DP rank chunk至少128+page_size（page256即384），并拒非DSv4.1使用encoder replay；太小的chunk此前每次零页admission使队列永久阻塞。

已读以上配置/collective/预算/状态patch。作者GB300真实超过cap触发旧crash、新request完成及边界unit回归；最终最小chunk改变也收紧gfx950原来能运行的129–383tokens配置。**建议按最新组合核chunk、DP、page、MegaMoE cap，不能只升级前半replay功能而遗漏后半修补；这些并非任意backend性能承诺。**

#### #45043：移除不安全的speculative-skip-dp-mlp-sync参数

来源：[sgl-project/sglang #45043](https://github.com/sgl-project/sglang/pull/45043)（2026-10-18，main；`github:sgl-project/sglang:pr:45043`）。

attention DP中用旧开关允许一rank prefill、另一decode，decode batch会被送入prefill路径AttributeError崩溃；原来其他配置并无运行效果。现在scheduler/spec planner移除该field与条件，未启 `SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION`时正常pre-sync；协调环境变量是另一有安全协议的路径。已读两个运行条件patch。**旧启动命令会报unrecognized argument，建议删除参数并回归DP混合负载。** 作者GLM4.7Flash MTP/DP2复现，非全面DP组合验证。

#### #43991：KDA verify的Q/K归一化改与reference相同舍入次序

来源：[sgl-project/sglang #43991](https://github.com/sgl-project/sglang/pull/43991)（2026-10-18，main；`github:sgl-project/sglang:pr:43991`）。

fused verify用x/sqrt、reference已经改1/sqrt后乘，浮点舍入不同；现同样先reciprocal再乘。已读完整kernel patch。作者MI355X两个bit-exact从失败转通过、文件25pass/1skip，并有NVIDIA CI结果；**代码数值变更适用于所有平台，其他仍用除法的recurrent/replay kernel未改**，另一KDA parity故障由#44089独立处理。建议有严格数值对齐要求者固定kernel版本比对，勿称所有KDA路径bit-exact。

#### #39264：packed weight receive按metadata直接分配，主线补入

来源：[sgl-project/sglang #39264](https://github.com/sgl-project/sglang/pull/39264)（2026-10-18，main；`github:sgl-project/sglang:pr:39264`）。

此前receiver先每tensor empty再cat未初始化字节，broadcast马上覆写；新 `FlattenedTensorBucket.empty` 一次分配uint8连续buffer，按name/dtype/shape/byte offset重建共享storage view。只用于distributed `load_format=flattened_bucket`，保持一次broadcast和begin/restore、load、end/postprocess生命周期；拒空meta、负shape、长度不一致及dtype对齐错。

已读receiver和factory完整patch。**此前#39265落sglang-miles，本条才进main**，非重复同分支事件。作者CPU混合dtype/bit-pattern/alias和拦截broadcast harness，未验证GPU/NCCL/真正loader整合，full-model benefit未测。建议深入读receiver内存分配；不能把少一次cat外推权重同步吞吐或原子更新。

#### #44666：Muse Spark text/Omni运行时主线集成，B300验收由本轮后续补上

来源：[sgl-project/sglang #44666](https://github.com/sgl-project/sglang/pull/44666)（2026-10-18，main；`github:sgl-project/sglang:pr:44666`）。

82文件、+9175/-224；加入MuseSparkForConditionalGeneration、torch-native NaViT图像/视频与audio encoder、MoE/量化与runtime defaults。已实际读muse_spark_hook和Omni构造/forward关键patch：仅识别该architecture后设graph dedup/pool borrow/precarve/postcapture sizing等默认，显式env优先；modelopt_fp4才默认per-token activation scales。Omni按encoder-only/language-only选择组件，维度与projector激活有校验，非decode多模态forward映射IMAGE/VIDEO/AUDIO特征；breakable纯文本也显式materialize embeddings避免graph buffer保留零值。

**仅选择性架构审阅，未逐行审model/layers/量化kernels。** 作者594 targeted tests、4B200若干真实多模态/非Muse回归与smoke；合入时8B300 CI在tokenizer加载前失败，0测试执行。**本轮后续#44975已用专用B300重跑补证并恢复注册，详见后文；不能再沿用“完整验收仍待”的昨日结论。** AMD支持和Helion megakernel仍另层#44667/#44668，不能称本PR单独提供全部平台能力。

#### Paged Experts把GPU专家working set与host pinned权重分开

来源：[sgl-project/sglang #42557](https://github.com/sgl-project/sglang/pull/42557)（2026-10-19，main；`github:sgl-project/sglang:pr:42557`）。

每层GPU只保K个专家slot，所有E专家放pinned host，按路由缺页搬运后仍在GPU算；eager可分多wave，GPU侧LRU/页入可捕获单wave decode，最多K//top_k并发。已读method/executor及capture限制；单GPU，BF16/FP16、block-FP8 Triton、GPTQ Marlin，尚无TP/EP、NVFP4/AWQ/MXFP4或disk tier。

作者小样本logprob BF16/FP8 bit-identical；GPTQ受Marlin batch数值噪声限制。16GB卡int4对调优llama.cpp为205.7vs194.4tok/s，绝非宣传12x（12x是对SGLang全量CPU offload）；8GB卡K26/E128、4请求时仅11–12vs41tok/s，工作集抖动可更慢。短prompt TTFT也可能更差。建议把K与KV容量、并发、PCIe带宽一起调，优先作容量可行性工具。

#### NVIDIA megakernel仅新增插件契约与运行时绑定入口

来源：[sgl-project/sglang #44945](https://github.com/sgl-project/sglang/pull/44945)（2026-10-19，main；`github:sgl-project/sglang:pr:44945`）。

--enable-nvidia-megakernel默认off；外部spec声明GPU/shape/数值/功能要求，启动前与每TP rank一致校验，post-load后构建，覆盖step走plugin、其余走原forward。binding传当前KV/attention/Engram，seal检查来源替换，拒weight update与没有真正覆盖decode graph的安装。DSpark draft可提供独立spec/sampler。

已读guards/runtime/binding接口，模型kernel本身位于独立sgl-nvidia-mega，未审该仓全部实现。作者CPU154pass/1GPU skip；GB300 TP4+外部plugin、DSv4.1 DSpark 200题197/200，needle128/135，长16K/24–32K通过。主仓PR没有自身E2E提速表；建议仅按plugin明确兼容矩阵试用，不能把入口合入视为所有NVIDIA模型自带megakernel。

#### DSv4.1 PD默认MegaAttention，并把真实KV布局加入握手

来源：[sgl-project/sglang #45075](https://github.com/sgl-project/sglang/pull/45075)（2026-10-19，main；`github:sgl-project/sglang:pr:45075`）。

SM100/103两PD角色不再退FlashMLA；packed V4.1与V4虽同dtype却不兼容，prefill注册主布局及各compression ratio，Python/Rust bootstrap中继，decode拒KV layout mismatch。已读pool属性、默认backend及decode gate。作者两GB300、DEP2+DSpark Mooncake，GSM8K96.66%与双FlashMLA相同；故意只改decode则首请求准确拒绝。

跨GPU型号PD需prefill显式flashmla，旧decode无法解析新字段，应两角色一起升级；TRTLLM PD仍因未测试拒绝。建议先做握手负例，防“dtype一样即可互传”的错误判断。

#### decoder replay tail图扩展到BCG和attention-DP

来源：[sgl-project/sglang #45059](https://github.com/sgl-project/sglang/pull/45059)（2026-10-19，main；`github:sgl-project/sglang:pr:45059`）；[sgl-project/sglang #45107](https://github.com/sgl-project/sglang/pull/45107)（2026-10-19，main；`github:sgl-project/sglang:pr:45107`）。

先将eager prefill的tail按行bucket图重放，再让>=2048-token的breakable prefill shape在尾切换处eager break，只对live tail row重放；DP rank按最大tail bucket统一，idle rank也参加。MegaMoE仅eager prefill支持；capture stub跳过未计算activation上的indexer，避免捕获期间DeepGEMM越界。已读row-bucket runner和tail条件，未全审所有kernel。

作者GB300 DSv4.1+DSpark：TP4 BCG冷8K TTFT146.4→100.4ms；DP4+TP MoE eager saturated prefill93.7K→140.9Ktok/s，GSM8K各臂不同但没有等价性证明。基线在#45059之前，不能把两级combined收益各归给一个PR。encoder replay inside BCG、CP、unified-KV Triton等仍off；MAX_ROWS=0可关闭。建议按实际MoE backend和prefill图方式分别验收。

#### P/D路由延后选择decode，流式首包统计另设兼容门

来源：[sgl-project/sglang #44689](https://github.com/sgl-project/sglang/pull/44689)（2026-10-19，main；`github:sgl-project/sglang:pr:44689`）；[sgl-project/sglang #44402](https://github.com/sgl-project/sglang/pull/44402)（2026-10-19，main；`github:sgl-project/sglang:pr:44402`）；[sgl-project/sglang #45102](https://github.com/sgl-project/sglang/pull/45102)（2026-10-19，main；`github:sgl-project/sglang:pr:45102`）。

两个router分别实现--decode-dispatch-limit N：所有请求先去prefill，每prefill worker/DP rank只有N个提前进入decode，余下FIFO等待到更接近KV到达时才选rank，减少长prefill队列导致decode running batch失衡；需optimistic-prefill-attempts>0。取消/超时释放gate并abort prefill，默认不开。已读Rust gate和dispatch关键patch；作者MI355X AgentX两组一小时比较，gateway总输出约+1.5–4.0%，不是任意负载保证。

#45102另加默认off的--pd-stream-after-prefill，使sgl-router等prefill成功才交decode首包，用于对齐gateway；并不推迟decode计算。作者192会话时0.7–0.9秒decode输出随首token一起到达，改变TTFT/ITL归属。建议比较router时先统一首包策略，再谈推理性能。

#### 修复两类借用buffer寿命：retraction恢复与流式量化待配对

来源：[sgl-project/sglang #42702](https://github.com/sgl-project/sglang/pull/42702)（2026-10-19，main；`github:sgl-project/sglang:pr:42702`）；[sgl-project/sglang #44500](https://github.com/sgl-project/sglang/pull/44500)（2026-10-19，main；`github:sgl-project/sglang:pr:44500`）。

PD overlap中刚完成请求的forward仍可写已回收slot，resumed request恢复KV必须先wait forward_stream；否则host_pool/cpu_tensor恢复都可能被旧写覆盖。已读单stream屏障及定向延迟测试，代价是恢复那一步让出overlap；建议在显存压力retraction下测真实KV，而非仅正常请求。

RunAI callback返回即复用streamed tensor，在线NVFP4暂存gate/up或FP8 weight/scale等配对会借到覆写数据；#44500给storage传播borrowed标志，仅进pending队列时copy到owned CPU。12 CPU回归覆盖到达次序、packed和detach alias，GPU量化替为记录函数，无新full-model eval或加载成本表。建议大分片流式权重加载检查NaN/scale及跨文件配对。

#### hybrid attention-DP不再把decode语义误当prefill

来源：[sgl-project/sglang #34535](https://github.com/sgl-project/sglang/pull/34535)（2026-10-19，main；`github:sgl-project/sglang:pr:34535`）。

per-worker max_running_requests最小1，解决全局cap小于DP数时floor0无法启动；引入logical_forward_mode穿透DP padding的decode→EXTEND标签，使Mamba/KDA/GDN继续构建decode状态，不读不存在的prefill tracking seqlens。TBO不支持这种重标路径并显式拒绝。

已读容量下限与mode消费patch。作者4GB300/Nemotron-H并发sweep旧cap1起不来、仅容量修补后>=64仍崩，两修同时后完成；该模型无GDN，所以Qwen GDN布局未由E2E覆盖。当前rebase只重施divisor重命名，GPU数据属之前head。建议低并发、idle rank、track boundary分别回归，别把“标签统一供collective”当计算语义统一。

#### MoE路由缩放只计一次，归一化只含真正routed专家

来源：[sgl-project/sglang #44901](https://github.com/sgl-project/sglang/pull/44901)（2026-10-19，main；`github:sgl-project/sglang:pr:44901`）；[sgl-project/sglang #44904](https://github.com/sgl-project/sglang/pull/44904)（2026-10-19，main；`github:sgl-project/sglang:pr:44904`）。

#44901统一route-aware的routed_scaling_factor_fused_in_topk，修TBO rsf²、Waterfill/每rank共享slot的1/rsf误权重，以及aiter偏置/无偏置/hash路线差异。#44904在每rank融合shared slot前只选真实routed topK再归一，避免选K+1后覆盖末列，使剩余routed weights仅0.85–0.97。FlashInfer router会强制renormalize的契约也单独分流。

已读flag/epilogue与topk remap关键patch，19+11文件选择性审阅。作者CPU与H200定向kernel、812项区域用例通过；最终清理/rebase未重跑accelerator，ROCm native/full-model长生成和GSM8K仍待。建议用模型实际quant/backend/TBO/shared融合组合做路由权重sum及输出A/B；大量unit通过不能代替质量验收。

#### Kimi长prefill attention-residual地址乘法改int64

来源：[sgl-project/sglang #43438](https://github.com/sgl-project/sglang/pull/43438)（2026-10-19，main；`github:sgl-project/sglang:pr:43438`）。

NB8/H7168时stride57344，token37450起乘积越2^31，>=37451tokens可非法访问；Blackwell主TMA路径虽安全，DSpark aux-hidden仍会走旧_mix_fused。三处program_id先int64，数值算术不变。作者GB300边界回归旧fault新pass，短shapebitwise；32K耗时约+0.4%，HIP改动没在ROCm跑。建议长chunk/DSpark用户优先测溢出边界，不把异步报错后的那个kernel错当根因。

#### Muse Spark专用B300 PD三项重跑通过，恢复CI注册

来源：[sgl-project/sglang #44975](https://github.com/sgl-project/sglang/pull/44975)（2026-10-19，main；`github:sgl-project/sglang:pr:44975`）。

这是#44666后续状态推进：先前tokenizer前HF cache清理竞态导致0测试执行，本PR只移除临时disabled并把est_time900改800，未修cache cleanup本身。官方B300 job117911539097在25ecb5556c1ae908f1de7eaa1c0575511569a629跑3测试通过，746.859秒；后续仅时间估值改动。

因此可将“专用B300验收未执行”收敛为已跑通过；500tok/s和accept-length4.7仍是门槛，不是报告观测值。建议阅读真实job与固定测试配置，不能把恢复注册扩大成所有多模态/AMD/megakernel支持认证。

### slime

#### SGLang可用独立Python子进程运行，解耦训练依赖

来源：[THUDM/slime #2484](https://github.com/THUDM/slime/pull/2484)（2026-10-19，main；`github:THUDM/slime:pr:2484`）。

--sglang-python-executable或server-group覆盖让slime仍负责启动/健康/退出，但SGLang在自己的Python/conda环境构造ServerArgs，通过stdin JSON传参，避免继承父进程已导入Torch类。子环境保CUDA可见设备，剔除expandable_segments allocator选项，保其他配置，默认Mooncake端口范围防冲突；退出等本server进程树释放GPU。

已读runtime.py和配置guide patch；CPU测试覆盖特殊字符参数、env隔离、Popen/multiprocessing子树退出，使用fake streamer/模块不构成真实多环境GPU E2E。建议训练/推理依赖冲突者试用，核两个环境都能import slime且weight-sync所需协议兼容。

#### Straw checkpoint在原队列回退，并为坏rollout state保cursor退路

来源：[THUDM/slime #2482](https://github.com/THUDM/slime/pull/2482)（2026-10-19，main；`github:THUDM/slime:pr:2482`）。

恢复保存prompt并删除未来rollout索引/归档，使相同step可重新保存；pending/ready输出重新生成，不复用旧response/training状态。保存原输入引用可跨fanout去重；旧checkpoint从dataset ordering重建，整体校验cursor后再改producer。rollout state坏时警告并退到可读dataset cursor或offset0。

已读rewind.py及queue/recovery入口；作者346CPU测试和两节点Ray rewind通过。fallback0可能重读数据，重新rollout也不是复现同一采样轨迹。建议恢复演练中同时核prompt身份、训练step和重复数据，而非只看模型checkpoint能load。

### TorchTitan

#### #5470：保留non-special added token，避免decode吃掉reasoning/tool分隔符

来源：[pytorch/torchtitan #5470](https://github.com/pytorch/torchtitan/pull/5470)（2026-10-18，main；`github:pytorch/torchtitan:pr:5470`）。

tokenizer配置已保留AddedToken.special=False，注册却调用add_special_tokens，把普通added token也升为special并在默认decode被过滤；改add_tokens保原标志。已读 `HuggingFaceTokenizer._infer_special_tokens` 与离线CPU regression patch。

作者Qwen3-0.6B固定tokenizer revision：reasoning从reasonanswer恢复<think>...，tool_call wrapper亦保留；EOS仍正常过滤，token IDs/模型参数不变。52CPU测试通过/1gated下载skip/5subtests，无GPU训练。**建议先做tokenizer roundtrip和tool/reasoning解析回归，而非仅比loss；这与昨日chat-template/BOS修复是不同缺陷。**

#### MLA CP传压缩latent，Ulysses合并Q/KV通信

来源：[pytorch/torchtitan #4760](https://github.com/pytorch/torchtitan/pull/4760)（2026-10-19，main；`github:pytorch/torchtitan:pr:4760`）。

all-gather先传normalized KV latent与head-shared key，再各rank本地投影/展开，交换额外投影计算以减少通信；Ulysses把Q+packed KV合成一次all-to-all，shared key独立all-gather，取代3次Q/K/V all-to-all。HF adapter键保持兼容，native attention.wkv_b.weight改nested键且不自动映射旧键。

已读cp_mla、config sharding/gate关键patch，50文件未全审。作者32H100、CP8/FSDP4/EP8、DeepSeekV3 16B单run，all-gather TPS/GPU+4.02%、Ulysses+9.80%，step SD较大；2H100固定batch100步首步logit/lossbitwise但gradient已有0.1475–0.2716%差异，不能称训练等价/收敛不变。fused_mla+CP拒绝；vLLM wrapper当前拒MLA，KimiK3 MLA inference/RL parity tests被禁。建议先查checkpoint迁移和backend，再复现通信收益及训练漂移。

#### RL trainer接通PP，microbatch loss参数与stage权重共同发布

来源：[pytorch/torchtitan #3980](https://github.com/pytorch/torchtitan/pull/3980)（2026-10-19，main；`github:pytorch/torchtitan:pr:3980`）。

每microbatch独立传advantages/logprob/mask/temperature及loss metrics，TorchStore各stage发布到同一个model_state_dict key；actor mesh按pp/dp/cp/tp切，metrics由真正末stage持有rank返回，不假定固定rank。GraphTrainer仍共享loss kwargs，不支持这种per-microbatch语义；RL CP和PP+RDMA microbatch prefetch仍拒绝。

已读engine/loss wrapper/controller关键patch。依赖PyTorch step(loss_kwarg_mbs)，主审已核实际main commit9a2d9d7b6434e9aa4e048aa8d1b76939e5fe9962（10-15）落地，PR200238本身的ghstack merged=false不能推翻commit证据；nightly wheel是否包含未核。作者当前版8GPU整合未跑，旧workaround版8step比特对齐不代言最终版。建议固定含该commit的torch并跑PP2/FSDP2真实rollout。

#### chunked loss保原dtype梯度，并恢复冻结decoder下head训练

来源：[pytorch/torchtitan #5408](https://github.com/pytorch/torchtitan/pull/5408)（2026-10-19，main；`github:pytorch/torchtitan:pr:5408`）；[pytorch/torchtitan #5389](https://github.com/pytorch/torchtitan/pull/5389)（2026-10-19，main；`github:pytorch/torchtitan:pr:5389`）。

#5408 chunk梯度只是写互斥slice，原FP32 assembly再转BF16/FP16没有加法价值；改hidden dtype直交decoder backward。H100 16K少128MiB、128K少1043.983MiB，72行为组合/12graph replay/12debug训练步精确一致；速度0.3–1.07%小于部分重复波动，作为显存优化看待。

#5389冻结decoder但lm_head/其LoRA可训练时，chunk内部仍反传head，返回可backward leaf；保持no_grad验证只forward。36loss tests及head-only对照通过；两GPU FSDP/PP、C4收敛、GraphTrainer autograd.grad桥未验。返回scalar后再scale不会缩放已累积head grads，使用者须理解该已有副作用接口。已读两项loss.py patch，建议head-only微调验证真实梯度而非仅loss.requires_grad。

#### HF streaming惰性restore期间保存pending cursor，避免二次续训读错文档

来源：[pytorch/torchtitan #5501](https://github.com/pytorch/torchtitan/pull/5501)（2026-10-19，main；`github:pytorch/torchtitan:pr:5501`）。

HF load_state_dict到下一次read才生效，packer在read前拍source状态时会读到旧cursor；现set_state保原bytes，第一次成功read才清，epoch/HF字段格式不变。真实JSONL Grain packer两次checkpoint原第二次拿错文档，修后input/label/position/mask与不中断一致，prefetch开关均覆盖。

已读cursor patch，作者CPU127pass/1accelerator skip/9subtests，6新增反例旧版全失败；缺torchvision的多模态suite和GPU训练未跑。建议连续两次save/restore且中间跨document remainder测，不要只测一次恢复。

#### 梯度裁剪归optimizer所有，提供backend局部替换契约

来源：[pytorch/torchtitan #5394](https://github.com/pytorch/torchtitan/pull/5394)（2026-10-20，main；`github:pytorch/torchtitan:pr:5394`）。

Optim持有一次构造的GradientClipper，默认保持原dist_utils行为，标准和TorchFT路径共享dispatch，外部设备backend可实现自己的分布式norm/scale而不全局monkeypatch eager工具。已读interface和两调用点；作者44CPU/mock测试通过，未提供backend-specific GPU性能。建议外部加速器移植用该扩展点，并保跨mesh全局norm、nonfinite和返回tensor生命周期契约。

### 扩展项目

#### Transformers 5.20.2：MoE 的 BF16 零梯度和多模态输入类型回归进入正式补丁

**Hugging Face；v5.20.2，10-19T17:45:37Z。** [release](https://github.com/huggingface/transformers/releases/tag/v5.20.2)、[专家归约修复 #49688](https://github.com/huggingface/transformers/commit/d88946453f945ecdc7bc5cf4427084a5d6c215fe)、[输入类型修复 #49731](https://github.com/huggingface/transformers/commit/b19c4a6669eab3d446789405d5bd00c3ace65863)。

当 EP 组小于 FSDP 组，FSDP2 的非组大小 divide factor 会走 PREMUL_SUM；patch 注明 PyTorch 至 2.13 的 NCCL BF16 路径可能把专家梯度归零。现在对该专家 FSDP 路径统一先求和、再本地除，不能只让非 NCCL 后端绕开。另一修复保留 `generate` 中 dict-like 输入的具体类型，避免 `ModelOutput` 等被设备搬运工具降成普通字典。两项都已列入正式 release，优先于只在 main 的新增特性检查。

**建议考虑采用。** 对 EP×FSDP 的训练验专家梯度非零与单步更新；多模态回归保留原模型输入类型。此处静态核查没有独立训练重放，修复不等于所有 BF16/通信后端问题已解决。稳定ID：`github:huggingface/transformers:release:v5.20.2`。

#### Transformers main：embedding 并行语义、非整除 TP 和续接式 chunked prefill

**10-19 主线提交；不一概归入 v5.20.2。** [#49570 / 718d12a](https://github.com/huggingface/transformers/commit/718d12a9012a100c86afa105145b723200d3fffc)把 linear 的 row/col 并行与 embedding 分开：词表切片 lookup 后 all-reduce，hidden 切片后 all-gather；有 TP plan 且名为 `embed_tokens` 的 embedding 默认按词表分片。linear 用模块全局宽度构造 DTensor，允许末 rank 较短的非整除切片。它有具体模块命名与 plan 前提，不是所有自定义 embedding 自动覆盖。

[#49662 / 90a1fdb](https://github.com/huggingface/transformers/commit/90a1fdb63dc95cd52d6faae131229f5c0ceb13ea)让 chunked prefill 从已有 Cache 的真实长度继续，裁掉 `input_ids` 中已缓存的旧前缀，避免重填和位置错位。多模态另有 [SmolVLM #49725](https://github.com/huggingface/transformers/commit/5b20befc1c565c1f46ba9659993b352d42ef146e)移除图像 token 数必须整除 patch_size 的错误约束，改验占位与特征数量；[Molmo2 #49736](https://github.com/huggingface/transformers/commit/1d9654e78a23fb167ef539cbd699dfbd3e4830cf)将视觉特征移到文本 embedding 设备，[Qwen3-Omni #49505](https://github.com/huggingface/transformers/commit/c75d679c2ec7edb6f1ddab29a2a4fe6799aa7a58)在未显式给 `return_audio` 时尊重 text generation mode。

**建议按触发条件采用。** TP 验不整除宽度、tied weights 和自定义 plan；缓存验完整输入与仅增量输入两种调用；多模态验真实图像和多设备。没有受控端到端性能表，不从减少复制推断服务吞吐。稳定ID：上述仓库/PR 号与固定 SHA。

#### TRL AsyncGRPO：以 KV 压力调节准入，Blackwell 转向 FA4

**10-19 main。** [#7610 / d296691](https://github.com/huggingface/trl/commit/d296691ae5ca2db37887942fa6b60265f2581595)在已有 staleness 上限之外增加活跃并发控制：从 KV token 容量/最大上下文起步，每 5 秒取服务 metrics，使用率低于 0.6 且无排队时按完成一轮的节奏增 25%，高于 0.8 时减 30%，等待自然排空而不取消运行请求。metrics 缺失时初始退回固定上限，后续格式/查询失败保持旧 cap；不是无条件最优调度。

同一 patch 的权重切换请求显式 `pause(mode=keep, clear_cache=false)`，避免清缓存后抢占和重算；精度检查改比 trainer 实际 BF16/FP16 compute dtype，而不是 FP32 master weight。保留旧 KV 的语义仍依赖上游异步训练与权重版本协议，本文不把它当 on-policy 等价证明。[#7792 / 03877ea](https://github.com/huggingface/trl/commit/03877eada0c8a92de2fe229bd13af81869c27466)为 async GRPO/蒸馏在 Blackwell capability≥10 默认选 FA4，其他 NVIDIA 仍 FA3、AMD 仍 aiter；文档继续排除该 Blackwell/AMD 路径上的 attention-logit softcap 模型。

**建议复现**真实长短混合 rollout 下有效训练样本/s、staleness、KV 抢占和重填，而非只看生成 tok/s。无完整控制实验，不量化吞吐收益。稳定ID：`github:huggingface/trl:pr:7610`、`:7792`。

#### TRL：MoE LoRA 漏训告警与 VLM 占位一致性

**10-19 main。** [#7771 / 88365e5](https://github.com/huggingface/trl/commit/88365e546b5be84821f1a8bd9f167ef3230950e2)对 Transformers v5 + PEFT 0.19.0–0.21.2 的特定 MoE LoRA 配置发告警：routed experts 训练了，dense/shared expert 的目标 linear 层却可能被跳过。该提交是探测和告警，**没有在 TRL 内修复 PEFT 的注入机制**；原文建议使用有修复的 PEFT main，需要另钉版本验证。

[#7698](https://github.com/huggingface/trl/commit/3a05623594c268a8972f2ff17068471e54f25139)为 VLM 经 vLLM 生成时默认对 image placeholder 加 -100 logit bias，减少生成无对应图像特征的占位 token；用户 generation kwargs 可覆盖此默认，不应写成形式上的绝对禁采。[#7773](https://github.com/huggingface/trl/commit/dd614b44e89fa2ed752e8194cb98c0cb92f707b3)用模型 config 的 image token 放置特征，避免 processor 属性缺失/不一致。

**建议采用相关回归检查**：列出实际 trainable 参数和梯度、核 shared expert 覆盖，并在多轮 VLM rollout 验 token/feature 对齐。稳定ID：`github:huggingface/trl:pr:7771`、`:7698`、`:7773`。

#### DeepSpeed continuous rollout：固定物理槽位进入实验性 CUDA Graph decode

**#8732，10-19T14:34:23Z 合入 master。** [PR 与实验条件](https://github.com/deepspeedai/DeepSpeed/pull/8732)、[代码 0eb3a8b](https://github.com/deepspeedai/DeepSpeed/commit/0eb3a8bcc6d9a8775478e5377482c4bf2dd23c1c)。固定 batch capacity、KV 行和 decode 张量地址，空槽 mask 后补新请求；参数 storage/version、设备、dtype 或静态形状变化重建 graph，匹配调用复用 capture。

范围限定 CUDA、greedy、每 prompt 一个样本、同一 padded prompt 宽度，且不支持 align_decode_fronts、cache trimming 和 FA2 的该 capture 路径。作者在 RTX A4500、PyTorch 2.9.1+cu128、Transformers 4.57.1 的小 GPT-2/Llama 验 token 一致；随机 medium GPT-2、batch1、32→32、两次 warmup/五次稳态测量从 250.88ms 到 31.45ms，是合成 launch-bound sanity check。**不能写成生产 LLM 普遍 8 倍。**

**建议复现**首调用 capture 与稳态分开，加入 EOS 槽复用、部分空 batch、热更新和真实 prompt 长度。另 [#8846](https://github.com/deepspeedai/DeepSpeed/commit/cc7b5125d52bf0878ead8237379e08e569e407f6)清理销毁后的 process-group 缓存，针对 CPU/Gloo 重初始化与退出生命周期；原始 CI abort 的同根因尚未由 core dump 确认。稳定ID：`github:deepspeedai/DeepSpeed:pr:8732`、`:8846`。

#### PyTorch：Flex decode 的 tile 边界和 NCCL2 等待覆盖修正

**10-18/19 main。** [#201328 / 9d18b95](https://github.com/pytorch/pytorch/commit/9d18b95219c1c16b653b5e43e8201c0d80e98b8f)把 `SAFE_M_BOUNDARY` 放到每个最终 BLOCK_M 配置之后计算，避免覆盖 tile 后误省略 Q 长度 mask；[#201329 / 780c30c](https://github.com/pytorch/pytorch/commit/780c30cf047428e017a51d69af18303333b17f09)按 GQA 折叠后的 query 数验稀疏 block 整除，并限制默认 tile，避免单 tile 跨稀疏 Q 行和不合理共享内存需求。代码注释提到旧大 tile BF16 慢 5–8 倍，但没有通用端到端测量条件，本文仅报告机制修正。

[#199585 / 278c1a4](https://github.com/pytorch/pytorch/commit/278c1a44a8a93029b343aae6b6418ffccf808d48)使新 `ProcessGroupNCCL2` 的 `endCoalescing` work 覆盖窗口中所有操作，包含普通 collective 与 coalesced/P2P 的组合，而非只保留最后一类。**建议对采用该后端或自定义 sparse/GQA decode 的系统优先验等待、跨 stream 可见性和非整除长度。** 不是所有现有 ProcessGroupNCCL 都有同一问题。稳定ID：三个 PR 号及固定 commit。

#### PyTorch：预编译/加载与 checkpoint 的具体可靠性增量

**10-19/20 main。** [#201314](https://github.com/pytorch/pytorch/commit/10ca444331aa499968e66558a5a2cfdfedbd22cf)记录 capture 中 Dynamo 放弃编译的 frame，加载后相应调用仍 eager 并一次告警；recompile-limit 的 RUN_ONLY 仍是 coverage gap。[#201319](https://github.com/pytorch/pytorch/commit/a9003555d275e9bfac3aa9bee0054f4162507036)允许为非默认 PG 的 DDP 保存可序列化 guard 状态，但 guard 若读取被丢掉的 PG/reducer/logger 属性则拒绝打包，不是把活 PG pickle 过去。

[#201261](https://github.com/pytorch/pytorch/commit/e5e5dde90b4f696a31eab48f686084b307a657dc)把 AOTI 加载 JSON 的共享 static 对象改为局部值，消除并发加载竞态；[#201267](https://github.com/pytorch/pytorch/commit/69baf2222996f582ba55afda2a1ad230dd972579)避免 GPU fallback 代码生成提前求参时遗失临时 tensor handle。[#200958](https://github.com/pytorch/pytorch/commit/40fdcfec1df62cd9c2f441b4727233ea0bb38d9d)在 DCP 写 safetensors 前 contiguous，覆盖转置 weight view 的导出失败，代价是可能发生复制。

**建议考虑采用对应路径修复**并测冷/热加载、并发打包与权重导出；这些提交没有发布统一性能提升。稳定ID：上述仓库/PR 与 commit。

#### TensorRT-LLM：Gemma mask 减少二次方中间张量，最终 dense mask 仍存在

**#20405，10-19T16:21:18Z 合入 main。** [PR](https://github.com/NVIDIA/TensorRT-LLM/pull/20405)、[代码 395e1e1](https://github.com/NVIDIA/TensorRT-LLM/commit/395e1e10fc6a541eac6460c55a7bae1524f1e533)。Triton 直接写 packed bool mask，辅助的 vision-blob 元数据为线性大小，保留 causal/window、同 vision blob 双向和 Gemma4 audio 规则。最终每请求仍需 Q×(P+Q) 字节。

作者单 B200、无 cached prefix、window1024、已 warmup 的 mask-only 测试：64K Gemma3 峰值 20→4GiB、Gemma4 12→4GiB；Gemma4 构造 46.905→2.255ms。完整权重的 Gemma3-12B/Gemma4-31B 在短文本/真实图像等八例上报告 logits 与后续 token 精确一致，但用兼容旧 native build 加当前四文件 overlay，**未证明当前 main 全栈兼容**；长例约1.4K，不能代替64K全模型精度验收。

**建议深入并复现**长多模态 prefill 的峰值瓶颈；20.8×属于 mask 构造，不是 TTFT 或推理吞吐提升。稳定ID：`github:NVIDIA/TensorRT-LLM:pr:20405`。

#### TensorRT-LLM：渐进 host KV 分配与 overlap 槽位，优化有明确资源代价

**10-18/19 main。** [#19650](https://github.com/NVIDIA/TensorRT-LLM/pull/19650)、[0cd9d0b](https://github.com/NVIDIA/TensorRT-LLM/commit/0cd9d0b5415b82696ecbaba8b0b6034aa9c09997)预留固定虚拟地址、后台按512MiB推进驻留/注册水位，首块可用即可服务。改用 fault-in + `cuMemHostRegister`，降低多 rank 争驱动全局锁造成的 executor 停顿；slot 不跨 registration，staging 另用固定分配，THP 默认关闭以限制碎片节点的 direct compaction 尾延迟。

作者8×B200、8rank各64GiB的填充/探针：旧路径填充6.4–10.4GiB/s、最坏探针停4.6–34s；注册路径填充5.2–5.6GiB/s、最坏0.3–2.6s。它牺牲填充带宽换响应，不能只摘停顿降低写成全面提速，也不等于服务时延 SLO 已验证。

[#20250](https://github.com/NVIDIA/TensorRT-LLM/pull/20250)为 non-ADP、非PP且开启overlap/KVCM2等适用路径扣除 retiring 请求并补一批物理 seat，消除 context-only 3/1交替而配置batch4的准入空洞。GB200 Qwen3-235B-FP4、8K/1K、con64、ctx TP1/gen TEP4 的受控overlay对比45,227→46,659 tok/s（+3.17%）；不是原回归全部恢复。seat从B到2B，增加惩罚词表和draft概率等内存，patch补了懒分配空间预留；hybrid/SSM、MRoPE等仍排除。

**建议考虑采用并复现**NUMA碎片、并发rank启动、带penalty/投机请求的真实峰值。稳定ID：`github:NVIDIA/TensorRT-LLM:pr:19650`、`:20250`。

#### TensorRT-LLM：GLM 量化 kernel 加速，与低精度/投机正确性修复分开验收

**10-19 main。** [#20390](https://github.com/NVIDIA/TensorRT-LLM/pull/20390)、[47de5e4](https://github.com/NVIDIA/TensorRT-LLM/commit/47de5e4c9cf1e7df58a077efa934ab90d8b185a7)向量化BF16→FP8 1×128 quant，满足对齐/stride条件才走新kernel，其余回退；增加288expert×4096 router的低时延实例。B200量化算子可到1.89×，88例量化字节/scale等同；router改变累加顺序，**不是整个patch逐位等价**。4×B200、GLM-5.3-Flash TP4/EP4、MTP3、13,312输入/33输出，作者C1 TTFT p50 372.9→366.6ms（1.7%），C8吞吐变化处于噪声。建议只按瓶颈复现。

另 [#20443](https://github.com/NVIDIA/TensorRT-LLM/commit/4d5f55a301106c5b79f4276b39265d385735d277)让MXFP8 fused QKV/gate-up的UE8M0 scale与weight使用同样行切片，含不均匀/复制TP；[#20223](https://github.com/NVIDIA/TensorRT-LLM/commit/85b1b8323b78a4717aeb772aa66eed9b8346d687)按Blackwell SwapsMmaAb实际layout求spec tree mask容量并传递边界；[#20477](https://github.com/NVIDIA/TensorRT-LLM/commit/72cd8b57a3399460549f65937ece92eba2cc7c2e)阻止LoRA超出真实problem的线程读module_out_sizes。

[#20048](https://github.com/NVIDIA/TensorRT-LLM/commit/c6350002f7e1d8e0f50a1aab86188b3d351f223d)在KVCM2直接保存完整string salt，区分无salt/空串并保留嵌入NUL的哈希语义；[#20442](https://github.com/NVIDIA/TensorRT-LLM/commit/df28262e26e16861e00c0762c32dbf4a3e2d5b38)让MiniMax-M3 streaming与非streaming按实际prefilled thinking模式一致解析，避免把可见正文/截断思维错误分区。**建议相关部署采用正确性回归**，不要把这些独立修复计成同一性能增益。稳定ID：上述PR及SHA。

#### vLLM-Ascend：KV 传输、Mamba 恢复与 sleep 生命周期的正确性

**10-18/19 默认分支。** [#18901 / a219257](https://github.com/vllm-project/vllm-ascend/commit/a219257748820aa15a998bb86b0cacb57cb06a04)把Mamba状态bulk copy推迟到同步非layerwise KV load完成之后，避免命中远端cache却复制旧state；实例级connector包装保持runner隔离。作者对同机制runtime专用port在8台A3节点PD环境验重复20,965 token请求，第二次实际加载20,736 token；这支持机制可行，**不是此默认分支SHA的原样八机验收**。

[#19101 / 2581a69](https://github.com/vllm-project/vllm-ascend/commit/2581a69bcc16ee6af56e72985a15799bb5f61808)处理Memcache返回失败时的请求级报告、queue/fence收尾、lease释放与关闭顺序；不会对可能共享GVA盲发WRITE_FAIL删除key。虽然已在默认分支，提交说明仍明确：配套vLLM `c4df37df...` 的`failed_recving`仅覆盖WAITING_FOR_REMOTE_KVS，**同步hybrid运行中请求的终止尚不闭合**；必要offload save的失败也不在全保证范围。不能写“KV错误恢复已全部修好”。受支持路径还需在顶层`KVTransferConfig`（含MultiConnector）设`kv_load_failure_policy="fail"`，才按失败策略终止请求。

[#18838 / 469318b](https://github.com/vllm-project/vllm-ascend/commit/469318b1fbd5800f0ade2e48d8ec829f8546e1ea)在level2 sleep前按配置pattern备份不由sender重发的参数，weights唤醒后恢复，KV-only唤醒不消费备份。作者单卡Qwen3-0.6B真NPU验证1,024值、token/logprob一致，未测多卡/graph。**建议相关RL/PD部署逐条回归**；分清状态复制、错误上报和调度器最终处理。稳定ID：上述仓库/PR与SHA。

#### vLLM-Ascend：block verification、decoder replay 与稀疏空工作边界

**10-18/19 默认分支。** [#19138](https://github.com/vllm-project/vllm-ascend/commit/ee29cc6fc24d20d47948776764059f05601170a6)接入MRV2的block rejection verification，利用累计target/draft比率与residual采样；需显式`rejection_sample_method=block`，默认standard，不能和synthetic acceptance混用。分布保持是上游算法语义，本文未做统计复现，接受长度和吞吐仍取决于draft与负载。

[#18936](https://github.com/vllm-project/vllm-ascend/commit/e8b7bc990df1c7b3c9e877784edc21110a7fa7ea)扩展DSV4.1 bounded replay到V1/V2，处理SWA重基址和runner元数据，移除0.30兼容shim；CPU检查通过，但A3四卡E2E卡在离线模型cache解析、未到replay，**硬件完整验收未完成**。[#19034](https://github.com/vllm-project/vllm-ascend/commit/5c9b1134d161b793632a4087eab678dec956f29f)按PCP front/back本地段将replay元数据容量加倍，修复max_seq16但本地27/29条时越界。

[#19073](https://github.com/vllm-project/vllm-ascend/commit/a5e97927ac100bb70b68cb2388852f1c3185ba7d)为A3 GLM KPool的有query/无完整pool空工作加dummy物理工作并在索引打包时丢掉候选，避免TND分配T却初始化B×T行的内存破坏；[#16936](https://github.com/vllm-project/vllm-ascend/commit/eb7c45831206a1b3bd2ffb5b3025d4c831fa9044)修MRV2共享top-k MTP graph capture的生命周期，使capture与runtime选择同一skip_topk分支，目前NPU graph/GSM8K仍未验。

**建议按精确后端复现**稀疏空rank、FULL/PIECEWISE graph、prompt尾部与MTP多步，不能从合入推出所有组合可用。稳定ID：上述PR号及固定SHA。

#### vLLM-Ascend：MegaMoE 初始化修复；短卷积去拷贝仍欠当前版本硬件证据

**10-19 默认分支。** [#18684](https://github.com/vllm-project/vllm-ascend/commit/19c2f7a7cb1504d4624620ea59d6c7fa9eadc422)删除DeepSeek W4A8 MXFP子类重复post-load覆盖，使父类执行CANN MegaMoE的packed FP4/E8M0布局与四组per-expert list准备，修启动AttributeError。原checkpoint int8装入uint8参数发生在copy阶段，删除的view不是新量化转换；作者有A5/950开启fused_mc2=2和非MegaMoE请求验证，约6%单卡吞吐缺模型/负载条件，不用作可比较收益。

[#18917](https://github.com/vllm-project/vllm-ascend/commit/24534c8636e347905e67f4a3d49042d9ec5ad7c0)对channel连续的SD short-conv state直接按page/row stride原地更新，省gather/scatter两次copy launch，并共享固定地址conv索引；DS等不支持布局仍staging。**最终runtime修订未执行/未profile，旧修订测量不能替代它。** 建议先复现state原地语义、spec padding/no-draft与真实NPU，再评估是否采用该性能路径。稳定ID：`github:vllm-project/vllm-ascend:pr:18684`、`:18917`。

<a name="hardware"></a>

## 硬件与软件协同

### Neuron 2.33 / TorchNeuron：原生 PyTorch 已有公开 Beta 源码，兼容边界比演示代码更重要

**AWS/PyTorch；10-19公告与SDK2.33.0 release。** [SDK发布记录](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/2.33.0.html)、[官方架构说明](https://pytorch.org/blog/introducing-torchneuron-native-pytorch-backend-for-aws-trainium/)、[公开代码/README固定版本](https://github.com/aws-neuron/torch-neuronx/blob/b1b6c66a2be28bce3bed6ef20632f0d3164119ef/README.md)。公告称10-20 public Beta；本轮已能实际读取Apache2源码，当前默认分支可见基线commit时间为10-17 `b1b6c66...`，不以commit时间推定首次公开时间；分别记录公告、代码工件与计划可用日，不把未来时态当已完成硬件部署。

PrivateUse1注册`neuron`，eager用于调试，`torch.compile`经FX/AOTAutograd→Torch-MLIR/StableHLO→NEFF；NKI自定义op、DTensor/DeviceMesh、原生collective和torch.profiler衔接。原README明确这是原生后端，**较早同名torch-neuronx是XLA产品**；示例要求PyTorch2.14、相应torch-neuronx2.14系列及Trn2/3，不能只升级旧包名就认为迁移完成。

实际README与[API兼容表](https://github.com/aws-neuron/torch-neuronx/blob/b1b6c66a2be28bce3bed6ef20632f0d3164119ef/docs/torch_api_compatibility.md)仍有限制：compile静态shape/contiguous，dynamic=False、mode默认，adaptive eager实验且默认关；部分op CPU fallback，FlexAttention支持有限，TorchAO/MXFP8未在此release公开可用。博客10万模型为缩小代表配置的load/forward/backward/compile扫测，非10万完整预训或性能验收；测试跳过项不能作通过率分母移除后泛化宣传。

**建议深入阅读并小规模复现**最小eager/compile、BF16训练、collective、真实模型算子fallback和冷编译成本，暂不由合作方美元吞吐口号推断LLM成本。稳定ID：`aws-neuron:sdk:2.33.0`；`github:aws-neuron/torch-neuronx@b1b6c66a2be28bce3bed6ef20632f0d3164119ef`。

### NKI 0.7：Python tracing、可变长通信前置准备和混合架构 kernel 同时落地

**AWS；10-19，Neuron2.33.0。** [NKI0.7 notes](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/components/nki.html)、[NKI-Lib notes](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/components/nki-lib.html)。specialization改用Python tracing/ISA发IR，可用更多Python元编程；tiles/blocks区分计算与传输粒度，view本身不搬数据。`all_to_all_v`/`all_gather_v`可prepare后再trigger，使已知counts的元数据准备与填源tensor重叠；每prepare只配一个trigger且中间不能穿插另一prepare，不能写成任意异步通信流水线。

NKI-Lib新增25项**experimental** kernels，包括DSV3.2稀疏MLA/indexer/router、prefill MXFP MoE、Qwen3.5 GDN和KDA token decode、GPT-OSS采样tail；这些是明确模型/阶段的算子，不能据此宣称DSV4.1或Kimi-K3全模型serving已经完整支持。另修scatter-add重复索引应累加、attention多batch布局、MXFP8 scale swizzle与尾块等正确性问题。

**迁移建议：考虑复现，勿直接批量升级kernel。** 默认frontend不再接受旧dynamic_range/while形式，需fori_loop/while_loop，过渡可设`NKI_COMPILER_FRONTEND=custom`；NKI与compiler LNC不匹配现在编译报错。Lib移除TensorView、A2AV改为required packed routing且插入新形参，MXFP8调优选项改config对象。新机制有明确代码契约，但本轮未运行Trainium算子或E2E性能复现。稳定ID：`aws-neuron:nki:0.7.0`；`aws-neuron:nki-lib:2.33.0`。

### Neuron compiler/runtime：低精度语义、内存与通信正确性值得随 Beta 一起验

**10-19；compiler2.28.7631.0、runtime2.35/driver2.31/collectives2.35。** [compiler notes](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/components/compiler.html)、[runtime/collectives notes](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/components/runtime.html)。compiler默认核NKI LNC一致，整数窄化从饱和改为补码wrap；unique-index整数/FP8 scatter避免全尺寸FP32目的tensor复制，broadcast weight batched-linear可降为一次二维matmul；修全mask softmax NaN等。内部大MoE backward编译“20×”、MXFP8特例“7×/30×”缺公开模型/shape和完整对照，**仅记厂商特定工作负载声明，不作为通用LLM提速**。

runtime引入无需NEFF的device cast/ADD/MIN/MAX及异步DMA路径，dma-buf默认用于EFA注册；profiling的H2D/D2H通知默认每NeuronCore额外64MB应计入显存预算。修DGE stale表、越界gather和collective同步/重叠buffer等；TRN3 LNC1支持AllToAllV，跨两rack最大256rank。仍有LNC2 AllGatherV小于64MB、LNC1某些跨node ring不支持、replica group不能任意混合rack布局等限制。driver sysfs路径重组会影响现有监控。

**建议采用前作定点兼容验收**：重新编译受影响NEFF、验证整数/低精度边界，按真实LNC和拓扑测collective；不要从SDK版本一致推断所有组件及应用都兼容。稳定ID：`aws-neuron:compiler:2.28.7631.0`、`aws-neuron:runtime:2.35:2026-10-19`。

### Helion × CompileIQ：把 PTXAS 控制作为调优维度，收益和许可都需单独看

**NVIDIA/Meta/PyTorch；10-19技术文章。** [官方实验](https://pytorch.org/blog/one-search-two-layers-compiler-controls-inside-pytorch-helion-using-nvidia-compileiq/)、[Helion固定代码](https://github.com/pytorch/helion/tree/ceafab07aab63876d87af36b5a7286cfb060ea0f)、[Booster Pack记录](https://github.com/NVIDIA/CompileIQ/releases/tag/booster-packs-2026.08.18)。本轮实际核`runtime/config.py`的`advanced_controls_file`和`settings.py`的`autotune_search_acf`，不是纯路线图。tile/warp/stage与寄存器分配/指令调度耦合，联合候选评测后固定ACF；无ACF默认也保留为候选。

GB200 sm100、ptxas13.3.73、CompileIQ1.0.3、Triton3.8、锁频的NVFP4 W4A4 decode GEMV，N8192/K28672，受控重测约0.0975→0.0930ms（1.048×），多目标候选约1.069×；不是完整模型吞吐。现成24份Helion pack标注B200/CUDA13.4，与实验13.3不同，不能直接混用。pack原8月发布，10-19增量是方法与实验说明；CompileIQ工具受NVIDIA专有EULA，**不能因Helion开源就称整套工具开源**，生成ACF可分发。

**建议考虑离线复现**最热kernel并钉GPU/toolkit、ACF摘要与数值检查；每份ACF增加编译搜索成本，架构/toolkit不匹配可能产生错误结果，默认候选存在不等于自动正确性保障。稳定ID：`blog:pytorch:helion-compileiq:2026-10-19`。

### JAX 0.12.0 与 XLA：MXFP4 和重计算/通信契约有正式与主线两层增量

**JAX v0.12.0，10-19T11:18:24Z。** [release](https://github.com/jax-ml/jax/releases/tag/jax-v0.12.0)新增`scaled_dot_general`的MXFP4（E2M1、每32元素一个E8M0 scale），gfx950原生、其他后端dequantize+dot；修fallback中量化rounding被excess-precision折掉和scale不用FP32计算。增加自定义VJP remat规则、显式sharding下禁止指定mesh轴隐式collective的`no_collectives`，以及ProfileSession/IFRT proxy后端，分别服务显存控制、通信意外检测和分离执行。

移除`custom_remat`、新增规则有新实现开关前提，x86最低Haswell/AVX2/FMA、NumPy≥2.2；最低libtpu变0.0.50。升级先检查自定义梯度与wheel依赖，release没有通用训练提速表。

**10-19 XLA main 独立于该release：** [bab0f6a](https://github.com/openxla/xla/commit/bab0f6aa0c7af4aa7a4aee51f1b934f80592a1b4)在实验collective kernel选择中纳入Triton all-to-all，补one-/two-shot策略和多进程测试；[e646c97](https://github.com/openxla/xla/commit/e646c974796caf1f134393934236eb91888ab69f)修variadic reduce-scatter padding program的输入tile越界；[1e8d90e](https://github.com/openxla/xla/commit/1e8d90ecf354a779466abc338a7353a1b0d93345)删除dimension squeeze替代后的有副作用原op，避免inline asm被执行两次。**建议关注/定点回归**MoE通信和自定义Triton编译；没有验证上述晚于release的XLA提交已进入JAX0.12 wheel。稳定ID：`github:jax-ml/jax:release:jax-v0.12.0`及三项XLA commit。

### ROCm/Ascend/MLX：后端数值、缓存失效与内存边界的定点变化

**10-19/20默认分支提交。** PyTorch [#200788 / 376702f](https://github.com/pytorch/pytorch/commit/376702f84049e0d74551e656432395979fd14da5)为gfx942的FP32且开启TF32 GEMM增加私有`TORCH_GFX942_ENABLE_RN=1`，先round-to-nearest-even以接近NVIDIA TF32输入转换；默认关，不改MI350/gfx950。它新增两次elementwise launch与临时operand，适合数值对齐诊断，不能当免费性能优化；原内部回放性能/质量未独立核实。

Ascend/pytorch默认分支 [6e3f766](https://github.com/Ascend/pytorch/commit/6e3f766d5b48943cd1de3be67018599f6ce16bbc)把执行中的triton_experimental后端Python源码digest纳入FxGraphCache key，防源码变而版本未变仍用旧artifact；[c0c5921](https://github.com/Ascend/pytorch/commit/c0c5921a11adb3191ac37aa63a54cf6bca7eaf67)使autotune自定义生成输入按实际分配shape/stride重解释，避as_strided存储越界；[17a3010](https://github.com/Ascend/pytorch/commit/17a301038147d353c935468a5c2839ff7f63b6f6)接NPU host pin/unpin到DCP stager；[52b8a09](https://github.com/Ascend/pytorch/commit/52b8a09a5fdbe62f92b1912c2b324760728e3608)让batch_invariant codegen用runtime轴长而非batch相关constexpr，非静态sibling轴仍拒绝。这是局部实现，非所有NPU模型batch invariant的证明。

MLX [#4719](https://github.com/ml-explore/mlx/commit/0390bd835d8675c949b78b2429409c93a483f255)补M1/M2的FP16 head_dim256 TK1 SDPA vector实例；[#4708](https://github.com/ml-explore/mlx/commit/c60afdd5b57fd0e82d10aa0ec36970ecfff7213f)在d512线程组过限时回退两pass；[#4721](https://github.com/ml-explore/mlx/commit/f41bec734ac9fc96723ab7d4e64e385796f15561)处理未对齐scalar读取与contiguous view存储范围。**建议对应设备/模型采用边界回归**，没有给出统一LLM吞吐增益。

DeepSpeed [#8799](https://github.com/deepspeedai/DeepSpeed/commit/91694726fef08b260ad72626f3ae5b8f72c81e30)新增昆仑芯KLXPU accelerator与自动探测，复用torch_xmlir的CUDA表面接口；算子builder清单包含显式NotImplemented项，Triton不支持，不能把“full op-builder coverage”误读为所有fused op和推理引擎可用。稳定ID：以上仓库/commit；建议有相关硬件时小规模验证。

### Cyclotron：显式表达通信依赖后由编译器合成send，收益已验证在线性代数而非LLM训练

**2610.20981v1，2026-10-14T21:49:40Z；旧待办补录，早于本轮72h回溯起点。** [完整论文](https://arxiv.org/html/2610.20981v1)。正文描述Python前端、C++ CPU/GPU runtime及Rust DAM simulator，本次未找到专用公开实现链接，机制未代码验证。

§3–5把tile递推、space/time映射、通信schedule分开；用户写receive依赖，编译器反转依赖自动合成send与guard，去重相同发送；block-cyclic映射后同物理processor的虚拟tile可复用一份网络接收；按依赖ready执行，避免跨kernel的强制全局barrier。它降低换通信schedule时改写计算的负担，不是自动替用户选择所有拓扑的最优schedule。

§6/Figs8–11、Tables2–3：Perlmutter16CPU nodes，GPU为4node×4A100，机内NVLink、机间Slingshot11；同precision、tile/kernel/thread/MPI条件，2warmup+4trial并报best。CPU Cholesky最佳相对ScaLAPACK1.34–1.68x，也有schedule更差；GPU matched host-staged相对SLATE最高1.24x；device-direct/out-of-order相对cuSolverMp1.12–1.28x。**相对SLATE4.57–4.74x混入SLATE在该机器缺GPU-direct的传输差，不应概括为纯编译收益。** CPU TRSM ping-pong较分开执行1.72/1.83x，相对ScaLAPACK1.97/2.07x。

**边界/建议观察**：测试是GEMM/Cholesky/TRSM，没有Transformer、TP/EP collective或训练吞吐测量；CPU节点型号/LLM序列等不适用。cuSolverMp调度原因来自作者profile，闭源内部机制未验证；best-of-four不是平均端到端SLA。可借鉴通信依赖和完成边界表达，在小型collective DAG复现；不能直接替换NCCL或把线代倍率写作LLM训练提升。稳定ID：`arxiv:2610.20981v1`。

<a name="evaluation"></a>

## 评估与泛化

### Pre-linearity：预训知识能被线性probe读出，比会不会答原问题更能预测一种受控重标记泛化

**2610.22300v1，2026-10-15T17:47:52Z；旧缺口补录。** [完整论文](https://arxiv.org/html/2610.22300v1)；[官方工件基线ec49160](https://github.com/NightMachinery/active-inert-facts/tree/ec4916030eb9afdfb0c513e05f86c39b5737b823)。工件静态检查，未重跑757个fine-tune。

§3把模型已有的关系答案随机一一映射到动物等标签，只在部分subject上教新映射，再看未见subject和新prompt模板是否跟随原关系泛化。pre-linearity由subject token隐藏态PCA白化至16维、正则线性分类器的nested10×5fold预测衡量，PCA每折只fit训练集，layer只在inner fold选；held-out subjects不参与probe训练/选择。`artifacts_v1/code/linroute/scripts/probe_ce_readout.py::ce_readout`重新按outer fold训练索引调用`_project_layers/solve_ce`并校验原工件预测一致；`materialize_newrels.py::build_cell`校验关系标签、首token和每例映射一致，是两处公开实现核对，未声称验证了整条训练pipeline。

§4–5/Table1、AppB/H：11个2B–27B开源模型、70关系，涵盖BEAR/Hernandez/300town池；全参20epochs，4措辞、每fact80次、batch16、BF16前反向/FP32参数与AdamW状态，最多8GPU但正文未列GPU型号。所有模型训练subject准确率1；跨关系pre-linearity与held-out generalization Pearson .87–.92，控制原关系Recall、答案数、subject数、Dolma频次后仍.70–.82。与之相比Recall控制pre-linearity后相关降为.40–.58；关系级10,000bootstrap区间，不是757模型都独立采样的置信保证。

**边界/建议观察**：该实证是关系层面的关联，任务为受控任意标签映射、候选内首token准确率，不能外推自由生成、复杂推理或证明线性可读性导致泛化。Dolma频次只是各模型未知预训暴露的proxy。可把nested probe加到知识注入前诊断，对“原知识答对却新任务学不会”的失败比单看Recall更有用；先在真实任务验证，不宜当部署质量保证。稳定ID：`arxiv:2610.22300v1`。

<a name="coverage"></a>

## 覆盖、失败、补录与更正

### 本地与远端连续性

本轮先读automation memory、持久化state、14篇本地历史日报和GitHub main。远端起点[`044168a843e8`](https://github.com/suhaibo666/tracker/commit/044168a843e875adeab49d33ebb4db533c705022)；14篇日报与README共15个blob与本地Git对象摘要逐项一致，最新索引正文另从固定commit读取。启动时无待上传文件。10-19只有研究草稿，未形成正式归档；其未报内容本轮重核后并入10-20，不篡改扫描水位。仅新增本日报并更新日期索引，不改llm-knowledge wiki。最终提交/远端正文回读回执保存在本地运行记录。

### 模型来源：部署说明增量、去重与待查

- **Nemotron Embed8B/Rerank8B/Embed1B**：10-19正式卡将vLLM示例0.31→0.32、固定`transformers==5.19.0`，增加`compile_mm_encoder:true`；独立Transformers5.20环境继续分离。8B embed/rerank tree差分均只有README。它是多模态encoder编译/依赖可运行性增量，未给量化速度收益，不报新权重。固定[Embed卡](https://huggingface.co/nvidia/Nemotron-3.5-Embed-8B-BF16/blob/f30256172b5192672824e1d718251aae0ed8bc45/README.md)、[Rerank卡](https://huggingface.co/nvidia/Nemotron-3.5-Rerank-8B-BF16/blob/39aa01525864f736e63f1d5742fc4cf59e754abd/README.md)。稳定事件ID `nvidia:nemotron35-retrieval:vllm032-mmcompile:2026-10-19`。
- **GoTo27B**：媒体10-19公告线索实际官方HF当前`db3fb073...`相对8-25`277207b...`只有README新增license/language front matter；权重commit早至2025-12-03。没有本轮新训练/新权重证据，**排除新模型主条**。此结论只限公开可见history，不断言旧仓当时是否已公开。
- **Ling3.1Flash**：95feb312相对a4ccd6只有将内网图片链接换为公共HF CDN，未改权重；排除新能力更新。**Bolt BF16**：567021只补竞争模型MCP-Atlas=48.4，非自家涨分；LiLiCorr当前仍需额外diff，未把lastModified当新权重。
- **Dory、Beam、DeepSeek1017、HyMemoryFP8、Jais3**：本轮卡/config/history重开，SHA同10-19稿。Beam卡完整报告coming、Jais报告/代码coming，Dory专门报告入口仍403；DeepSeek新harness成对重跑、HyMemoryFP8独立质量/校准等仍未取得。本次GGUF成绩不替代FP8对照。
- **Anthropic RSP10-19**：路线图含自动红队8题×3代分类器24组、23组优于人类的厂商自述，但无本轮可复现实验工件；覆盖记录保留，不独立写训练突破。OpenAI官方API changelog本轮可读，最近列10-16 GPT-Live语音功能；未从该入口发现主窗模型训练新证据，不能推断其所有官方渠道无发布。常规news/research仍直接403。
- **候选边界**：Real5-OmniDocBench、OpenAoE2000h等dataset有主窗改动，尚未逐commit深审；HF模型124、dataset32候选不等于全部已读。Cosmos/物理仿真、天气/时间序列、单细胞等非LLM训练主线候选保留覆盖记录。

### 模型来源：覆盖范围

31登记官方HF组织+inception42/IFM/inclusionAI，共34组织完成models/datasets的createdAt、lastModified双路线；各路limit100，返回页均跨回溯边界/到公开清单末端，附排序单调性核对。models实际72请求包含重复baidu与探索nvidia-nemotron空组织，不冒称36个官方组织；datasets68请求。另由主题搜索发现GoToCompany单仓并深审，未宣称遍历该组织。

官方入口直接抓取31处，正文可读程度单独记录；Mistral/xAI超时、Ai2/OpenAI/Dory部分403；Qwen/腾讯/百度/Step/MiMo等JS页可能仅壳。Seed列表API与MiniMax多种列表API保持独立边界；官方发布文章和模型卡胜过列表摘要。中英文日期/模型/技术报告检索只用于补充发现，二手源不承担最终技术结论。HTTP原文、hash、取证时间、差分和候选处置见`models-coverage.json`与`models-evidence/`。

未穷尽私有/删除/门控工件、所有分支、每篇厂商报告或所有候选；**全局完整覆盖水位仍为空**。事件截止时间不等于完整覆盖截止时间。

### 论文与正式学术来源

- 六学科cs.CL/LG/AI/DC/CV/stat.ML官方new页实际均为10-19公告、每页2000上限内取尽，共1255次出现/922唯一ID；360个replacement ID的最新Atom metadata已用4批id_list全部恢复，updated范围10-09T16:48:20Z→10-16T17:58:51Z，这个具体快照内没有主窗更新时间；未逐篇完成版本diff。
- 主窗submitted与updated各返回200/0，回溯updated返回200/1052；但是updated请求的Atom title/link明确被改成submittedDate。1052只能视为首次提交在回溯范围的元数据，不能关闭旧论文修订全枚举。补恢复的当前360也不覆盖10-19草稿里此前384/56独有replacement的全部旧待办。
- HF10-19、10-16各50候选，只作发现；10-19精选含10-16及更早论文，不等于当日首发。对本次13个稳定ID扫描所有本地已归档日报，没有重复。10-19六篇未归档草稿恢复，另三篇旧未读本次完成，外加四篇新的全文审阅。新候选与旧复用全文均保存在本目录。
- TMLR列表相对10-19新增7个forum：5ChJTFLi5r、ppOrbSsMM0、S2SdvYaPKm、xFFYq6NFQT、UibXoy4h3u、AKQ2b7VLA4、S98skK3FfD。仅给October2026；TrustVLM论文PDF403，OpenReview API403，不能判定准确接收日或全文结论。ACL/PMLR根页200、href集合无增量，未枚举全卷/正式版本。
- 中英文主题搜索补预训/循环架构/蒸馏/推理等名单外线索；本文数字均回原全文。2610.23257 HTML404保留未读；匿名lossy代码401，CCE/MSP未找到官方实现；这些不伪装成代码验证。DLite、CriticOPD、xllm-loop与冻结词表均钉SHA；另两份10-19冻结代码重开检查，均未执行GPU实验。
- 旧失败窗10-15T01:55:47Z→10-16T01:00:40Z、10-13起真实updated枚举，以及current replacements中的FastOPD2610.21337v2/N-Gram2610.21810v2等实质diff继续开放。本文没有把已归档v1换个摘要当v2增量。覆盖清单只代表选择性正文审阅，**全局完整覆盖水位没有建立**。

原始证据、查询URL/状态、候选、读过的正文定位、工件基线、稳定ID及剩余区间见`papers-coverage.json`、`papers-reviewed-evidence.json`、`papers-fetchlog.json`、`papers-extra-fetchlog.json`、`papers-replacement-fetchlog.json`、`papers-web-discovery.json`。

### 核心工程及P2P未闭合事项

本轮重开[issue61532](https://github.com/vllm-project/vllm/issues/61532)和[PR61667](https://github.com/vllm-project/vllm/pull/61667)：issue仍open、候选仍open/merged_at=null，head仍`4fd16ddb155fd0f7e676dece9abd6aa9209b2b1a`。在主窗末main `2296c12e47be87d44037d14e2a74028710c29f36`的ClientRole.collect_results，abort-ack仍10秒超时后报success=False并移除load；这不足证明远端WRITE不再接触目的storage。#60374新增peer-fault drain/reconnect是相关进展，**不关闭consumer quarantine及RDMA/serving寿命待办**。证据core-p2p-*及core-pr-vllm-project--vllm-60374.json。

六仓主窗all-base merged PR共298条（15/109/5/152/3/14），main294条；Megatron另外2条在core_r0.20.0，SGLang2条在sglang-miles。回溯all-base PR共801条（80/293/15/346/9/58）；默认main按commit时间枚举769条（50/294/20/336/11/58），已分页终止。主窗commit数见coverage，不等同merged PR数。Release各仓最新10条（TorchTitan总5条），最旧返回均早于回溯起点；本窗正式发布仅vLLM0.32.0与SGLang0.5.22。

库存完整不等于全部语义审阅。正文深读与patch检查是选择性，特别大型Muse、统一池/量化stack、外部plugin和kernel依赖未逐行全审；其余实质候选保pending，维护/CI/doc按标题及正文选择过滤。公网api.github.com首先403，已改认证connector重新完成库存；该访问失败不记作零变化。10-05→10-08及其他历史未审缺口延续，不通过本轮枚举关闭。该研究子任务未修改全局状态；本轮归档由主任务统一完成。

### 扩展工程与硬件实际覆盖

| 扩展/硬件仓库 | 回溯默认分支commit | 本轮读关键patch |
|---|---:|---:|
| Ascend/pytorch | 35 | 4 |
| Dao-AILab/flash-attention | 5 | 0 |
| NVIDIA/TensorRT-LLM | 112 | 9 |
| NVIDIA/nccl | 0 | 0 |
| OpenRLHF/OpenRLHF | 0 | 0 |
| ROCm/rccl | 0 | 0 |
| aws-neuron/nki-samples | 1 | 0 |
| deepspeedai/DeepSpeed | 18 | 3 |
| huggingface/transformers | 64 | 6 |
| huggingface/trl | 38 | 5 |
| jax-ml/jax | 135 | 0 |
| ml-explore/mlx | 12 | 3 |
| ml-explore/mlx-lm | 7 | 0 |
| openxla/xla | 216 | 3 |
| pytorch/pytorch | 283 | 9 |
| triton-lang/triton | 13 | 0 |
| vllm-project/vllm-ascend | 95 | 10 |

17仓共1034个默认分支commit，分页到尾；52个commit取回并选择性读关键patch，另读PyTorch loss_kwarg_mbs的实际main提交、数项PR正文和Helion固定源码。各仓release读最近10项（不足则全返回），只按实际发布日期判断；未穷尽所有分支合入或任意旧创建/新发布的release。表中0仅表示该默认分支/时间查询返回0，不表示项目、厂商或其他分支没有变化。未审commit清单完整留存，旧窗口继续开放。

NVIDIA与AMD官方技术博客归档、Gaudi1.24.2 notes、Cloud TPU notes、Huawei productbulletins和Neuron2.33及NKI/NKI-Lib/compiler/runtime均实际打开。华为入口200仍主要为动态壳，完整CANN/HCCL和libtpu逐版本覆盖未完成。Neuron2.33是本窗确认新release，不能沿用10-18缓存的“最新详细记录8-17”；Gaudi/TPU入口未在已读内容找到主窗新条目，不作厂商无发布结论。

公网GitHub API因共享IP rate limit返回403，已以认证connector重取并分页，访问失败与覆盖恢复分别留痕。MiniMax/GRAFT的web入口在主代理复读时工具报不可访问，模型子任务已直接HTTP200保存官方正文；这两处不依赖搜索摘要。

### 补录、更正与下一轮

13篇论文是明确标日的补录，含10-14旧待办；GRAFT论文10-12、Zephon代码10-12和10-16博客、Indic报告10-15/16等不冒充10-20首发。10-19稿中的Ming/Parse/ASR和模型数据差分均重新打开关键来源；MiMo完整Parquet计算沿用先前证据并明确未重算，HF改卡与改权重分开。未对旧日报盲目覆盖。

发布前校正：SGLang0.5.22的10-15 tag与10-19 publication分开；TorchNeuron10-17 commit时间不等于首次公开时间，当前公开源码与10-20 Beta计划并列；Ascend19101已合入但同步hybrid终止仍欠调度器支持，受支持路径需显式fail策略；DLite论文/工件hidden层配置冲突保留。另纠正研究稿误把Un-OPD v1→v2差异重新列待查：该具体diff已于10-18归档审阅完成，不重复开启。以上为本稿发布前校正。

下轮从 **2026-10-20T00:55:34Z** 事件位置继续并回溯72小时，同时续查已持久化旧缺口：核心工程10-05→10-08、扩展工程10-05起未审库存、原论文失败窗10-15T01:55:47Z→10-16T01:00:40Z、真实updated与replacement版本差分、正式会议卷与门控/动态厂商材料。不能通过本轮清单枚举关闭这些缺口。**全局完整覆盖水位仍为空，本报不声称全面监控。**

来源清单、各源进度、稳定ID/版本、失败、剩余区间、本地归档和远端同步分别记录在本地`.monitor/sources.json`、`state.json`、`run-2026-10-20.json`及`research-2026-10-20/`。归档失败时正文保留并进入待传；只有远端回读一致后才清除同步待办。

<a name="ledger"></a>

## 工程版本账本

|仓库 / PR|base|merged_at UTC|merge SHA|
|---|---|---|---|
|[NVIDIA/Megatron-LM #6981](https://github.com/NVIDIA/Megatron-LM/pull/6981)|main|2026-10-18T07:01:48Z|`85c950c074533bd4386ea7b48fe87e4727d8d48e`|
|[NVIDIA/Megatron-LM #7775](https://github.com/NVIDIA/Megatron-LM/pull/7775)|main|2026-10-19T21:11:42Z|`66a6d3f71610ce209e0fbf7d16049d94f841dbff`|
|[NVIDIA/Megatron-LM #8180](https://github.com/NVIDIA/Megatron-LM/pull/8180)|main|2026-10-18T08:36:11Z|`62c063f2b731b57b1c2031b2b4f4c6a769271ccb`|
|[NVIDIA/Megatron-LM #8207](https://github.com/NVIDIA/Megatron-LM/pull/8207)|main|2026-10-19T18:10:04Z|`60547a3f419bd8027bb6ffb91793c2fa99ad97ed`|
|[NVIDIA/Megatron-LM #8230](https://github.com/NVIDIA/Megatron-LM/pull/8230)|main|2026-10-18T05:39:16Z|`db3a6cd8a03eb550e9fa4532a0db96778bb56fc5`|
|[NVIDIA/Megatron-LM #8322](https://github.com/NVIDIA/Megatron-LM/pull/8322)|main|2026-10-19T21:19:05Z|`e592c042d575ad86b9a0b9b20239e4c1a1c1f060`|
|[THUDM/slime #2482](https://github.com/THUDM/slime/pull/2482)|main|2026-10-19T03:46:57Z|`4f204660abcd531882dd4918ee301f107a952eb4`|
|[THUDM/slime #2484](https://github.com/THUDM/slime/pull/2484)|main|2026-10-19T08:41:49Z|`a4202e0595467d9374d9a68d45a34f45c3f8d230`|
|[pytorch/torchtitan #3980](https://github.com/pytorch/torchtitan/pull/3980)|main|2026-10-19T06:51:12Z|`a4a62b349d669aea0bd413518bed6cb58a691b2b`|
|[pytorch/torchtitan #4760](https://github.com/pytorch/torchtitan/pull/4760)|main|2026-10-19T21:16:13Z|`0980e1cb1c9ab866ac855755bca2928c67bd1c8a`|
|[pytorch/torchtitan #5389](https://github.com/pytorch/torchtitan/pull/5389)|main|2026-10-19T23:10:07Z|`9391c0110aa9705b4bf8a1410c3c847ea6cc8fa1`|
|[pytorch/torchtitan #5394](https://github.com/pytorch/torchtitan/pull/5394)|main|2026-10-20T00:05:29Z|`9698a733623ca997ef824a03cb1f44706ee631e7`|
|[pytorch/torchtitan #5408](https://github.com/pytorch/torchtitan/pull/5408)|main|2026-10-19T07:08:18Z|`619cd4cef2ef3ff7e9667d1571cf81c51c8580e7`|
|[pytorch/torchtitan #5470](https://github.com/pytorch/torchtitan/pull/5470)|main|2026-10-18T23:21:19Z|`5e27dc87b8cc8f0f1dd0904bec60a704d9781ad8`|
|[pytorch/torchtitan #5501](https://github.com/pytorch/torchtitan/pull/5501)|main|2026-10-19T22:05:49Z|`c0eb136092402581bc1b184434bd2e46fea90bbc`|
|[sgl-project/sglang #34535](https://github.com/sgl-project/sglang/pull/34535)|main|2026-10-19T08:24:42Z|`33dd8f6a2eee41d4ad44fb397cf9b794df07aa54`|
|[sgl-project/sglang #39264](https://github.com/sgl-project/sglang/pull/39264)|main|2026-10-18T07:04:11Z|`520df8a4917fd1c4828ac988b12144c38a35c2a0`|
|[sgl-project/sglang #41907](https://github.com/sgl-project/sglang/pull/41907)|main|2026-10-18T07:10:51Z|`a11fa06be78dd3d033020f0f97f8f2be7befba53`|
|[sgl-project/sglang #42557](https://github.com/sgl-project/sglang/pull/42557)|main|2026-10-19T15:10:05Z|`8e7601ae4a48c1127afd1bcdccf0a0ad61dbf043`|
|[sgl-project/sglang #42702](https://github.com/sgl-project/sglang/pull/42702)|main|2026-10-19T18:55:05Z|`0ce41c5a06f829bb74e3d582c82d8d51837742df`|
|[sgl-project/sglang #43168](https://github.com/sgl-project/sglang/pull/43168)|main|2026-10-18T12:36:02Z|`47eeddcd180e2b5228ea35dae7a0ff327c5d07ab`|
|[sgl-project/sglang #43325](https://github.com/sgl-project/sglang/pull/43325)|main|2026-10-18T12:37:51Z|`19cb24f1e1c39edbdba7de4d46719150e6e55dec`|
|[sgl-project/sglang #43326](https://github.com/sgl-project/sglang/pull/43326)|main|2026-10-18T12:39:10Z|`3edb156decff8c2d1a4b5ddb9317260c3cebf0b6`|
|[sgl-project/sglang #43333](https://github.com/sgl-project/sglang/pull/43333)|main|2026-10-18T07:09:27Z|`e6c2cd47a93e5bdb7aabe0c21c6799208afccd4a`|
|[sgl-project/sglang #43438](https://github.com/sgl-project/sglang/pull/43438)|main|2026-10-19T22:12:13Z|`49766a74f9de2b5cdd3c11269671329f3d716ac2`|
|[sgl-project/sglang #43816](https://github.com/sgl-project/sglang/pull/43816)|main|2026-10-18T07:09:59Z|`ffc4573c981b35a45b89084c8dbea7d5ef393750`|
|[sgl-project/sglang #43991](https://github.com/sgl-project/sglang/pull/43991)|main|2026-10-18T07:01:12Z|`3626978a13694eb4c6e3a6080e4f1c4d3e5f4d52`|
|[sgl-project/sglang #44402](https://github.com/sgl-project/sglang/pull/44402)|main|2026-10-19T07:11:41Z|`f86a04a3314c673e3a9025451d5a6467213ff353`|
|[sgl-project/sglang #44500](https://github.com/sgl-project/sglang/pull/44500)|main|2026-10-19T14:41:54Z|`417606527d0059fbbb5f87b6eb2b8508f5647433`|
|[sgl-project/sglang #44666](https://github.com/sgl-project/sglang/pull/44666)|main|2026-10-18T08:00:25Z|`d33dd0566a1a8b77bdad83b64cbc45510307052b`|
|[sgl-project/sglang #44689](https://github.com/sgl-project/sglang/pull/44689)|main|2026-10-19T18:16:22Z|`678913a0f029394ad1f4ee64d1439421e9ea56ef`|
|[sgl-project/sglang #44901](https://github.com/sgl-project/sglang/pull/44901)|main|2026-10-19T23:54:28Z|`3a059de94c1642b35441cb16cd365b07fbae6990`|
|[sgl-project/sglang #44904](https://github.com/sgl-project/sglang/pull/44904)|main|2026-10-19T23:56:22Z|`48ea4869280dfaace60840b5c967ee00a5598968`|
|[sgl-project/sglang #44930](https://github.com/sgl-project/sglang/pull/44930)|main|2026-10-18T03:51:18Z|`ff73d63121e9fc28f430a37700920115762b9938`|
|[sgl-project/sglang #44945](https://github.com/sgl-project/sglang/pull/44945)|main|2026-10-19T18:54:45Z|`722669e6066fec2266e8a7e2f6bb738709a2f53f`|
|[sgl-project/sglang #44948](https://github.com/sgl-project/sglang/pull/44948)|main|2026-10-18T04:46:46Z|`d20d010574a4d1376e1759a79ae519c3dd9b4869`|
|[sgl-project/sglang #44952](https://github.com/sgl-project/sglang/pull/44952)|main|2026-10-18T07:07:56Z|`df966f234504523ab3ff4bd88521c99796405bc6`|
|[sgl-project/sglang #44975](https://github.com/sgl-project/sglang/pull/44975)|main|2026-10-19T20:40:03Z|`642a671c300babc061b0200c69b1c9882465696d`|
|[sgl-project/sglang #45035](https://github.com/sgl-project/sglang/pull/45035)|main|2026-10-18T18:53:36Z|`0f4aedfecc3e1e420292541af4e9da8909b9b4f7`|
|[sgl-project/sglang #45037](https://github.com/sgl-project/sglang/pull/45037)|main|2026-10-18T22:06:21Z|`fe89a7c8901216237e1acf67ec08e648ca41d16d`|
|[sgl-project/sglang #45040](https://github.com/sgl-project/sglang/pull/45040)|main|2026-10-18T22:43:10Z|`1461152cb0857ea2d8d95d3e87c491e4438c54ab`|
|[sgl-project/sglang #45042](https://github.com/sgl-project/sglang/pull/45042)|main|2026-10-19T00:09:19Z|`1637a4e3059d7576a34500fc5f50f6b202da1dfa`|
|[sgl-project/sglang #45043](https://github.com/sgl-project/sglang/pull/45043)|main|2026-10-18T23:41:22Z|`67084267289333faf784e895a88cf39d559922bd`|
|[sgl-project/sglang #45047](https://github.com/sgl-project/sglang/pull/45047)|main|2026-10-19T00:17:39Z|`3c560089e1818a5217a73f2168273485a5350341`|
|[sgl-project/sglang #45051](https://github.com/sgl-project/sglang/pull/45051)|main|2026-10-19T00:55:36Z|`59fec011dddec722d9a0dda8e208efe572ab2c7a`|
|[sgl-project/sglang #45059](https://github.com/sgl-project/sglang/pull/45059)|main|2026-10-19T06:07:51Z|`a3330237db9d0288e1bab603d8650e44344f66be`|
|[sgl-project/sglang #45075](https://github.com/sgl-project/sglang/pull/45075)|main|2026-10-19T04:16:46Z|`82c9a8b0372984316665efc302337b91d78a9982`|
|[sgl-project/sglang #45102](https://github.com/sgl-project/sglang/pull/45102)|main|2026-10-19T17:13:26Z|`4ef9765e76be4952d25f6b8ed160600aefc0757f`|
|[sgl-project/sglang #45107](https://github.com/sgl-project/sglang/pull/45107)|main|2026-10-19T20:02:07Z|`f1a2ee763d57d2066f7adde3dcbcb7fefe936326`|
|[verl-project/verl #8196](https://github.com/verl-project/verl/pull/8196)|main|2026-10-18T04:36:44Z|`8ced5d2f4b148c2991217554e0c6f03028fa1470`|
|[verl-project/verl #8274](https://github.com/verl-project/verl/pull/8274)|main|2026-10-19T05:08:10Z|`263653e749306287a5032140cce3d7833c1959d0`|
|[verl-project/verl #8287](https://github.com/verl-project/verl/pull/8287)|main|2026-10-19T14:01:44Z|`8f016846e3010e225072a947270dc0c1b98c1cd0`|
|[vllm-project/vllm #52477](https://github.com/vllm-project/vllm/pull/52477)|main|2026-10-18T18:29:22Z|`6540d5c8f8c73bb2dd6194e56ed002fc32a31837`|
|[vllm-project/vllm #53987](https://github.com/vllm-project/vllm/pull/53987)|main|2026-10-19T07:30:10Z|`c1dc2e9cf992995953548c392f8c67db7283495b`|
|[vllm-project/vllm #58231](https://github.com/vllm-project/vllm/pull/58231)|main|2026-10-19T07:51:51Z|`852989f580820849113d67d13fd8053e279d0444`|
|[vllm-project/vllm #58471](https://github.com/vllm-project/vllm/pull/58471)|main|2026-10-20T00:40:10Z|`45ce4050bcb90710e45c19907d330396d1d1d246`|
|[vllm-project/vllm #59030](https://github.com/vllm-project/vllm/pull/59030)|main|2026-10-19T04:01:50Z|`43caf5b3520b704fa766f00fc2b24da25c78e084`|
|[vllm-project/vllm #59737](https://github.com/vllm-project/vllm/pull/59737)|main|2026-10-19T19:17:28Z|`3c9beb82d1a92eb5a84cca4408a0a1b479d4dd03`|
|[vllm-project/vllm #60118](https://github.com/vllm-project/vllm/pull/60118)|main|2026-10-19T07:39:32Z|`018f5ff71aa9cf7138d562dade7e8f1caebfa7cc`|
|[vllm-project/vllm #60374](https://github.com/vllm-project/vllm/pull/60374)|main|2026-10-19T14:31:51Z|`4bd75e952a9f423742223c97c8cc9fc6db362950`|
|[vllm-project/vllm #60553](https://github.com/vllm-project/vllm/pull/60553)|main|2026-10-19T09:38:25Z|`1641765ffb53cb10f2753ece0efb41399f7381d3`|
|[vllm-project/vllm #60735](https://github.com/vllm-project/vllm/pull/60735)|main|2026-10-18T22:47:49Z|`36e0ae2df2882de8027bb484da9179c10646ccee`|
|[vllm-project/vllm #60767](https://github.com/vllm-project/vllm/pull/60767)|main|2026-10-19T20:25:37Z|`e1f3141647b93f12ac2ffa95f0cf7699fcaca21f`|
|[vllm-project/vllm #61002](https://github.com/vllm-project/vllm/pull/61002)|main|2026-10-19T00:07:55Z|`8f17dbf707459686eb7e500748552be98612218c`|
|[vllm-project/vllm #61465](https://github.com/vllm-project/vllm/pull/61465)|main|2026-10-19T15:48:14Z|`a3cee21d0bf986b519b48f13a3e7fa3814cbabca`|
|[vllm-project/vllm #61726](https://github.com/vllm-project/vllm/pull/61726)|main|2026-10-19T06:24:42Z|`e7023e596d94443014bae65065aefd0c3c933fb9`|
|[vllm-project/vllm #61819](https://github.com/vllm-project/vllm/pull/61819)|main|2026-10-18T14:39:39Z|`e80b6c642f21fd5b24439825592f5b9ced36add8`|
|[vllm-project/vllm #61938](https://github.com/vllm-project/vllm/pull/61938)|main|2026-10-19T00:39:15Z|`689effee6d07bfeb0afe42d44046141657238c23`|
|[vllm-project/vllm #62129](https://github.com/vllm-project/vllm/pull/62129)|main|2026-10-18T06:51:33Z|`7c9042c05addf034366b9a1da5b1d3e3687c052b`|
|[vllm-project/vllm #62183](https://github.com/vllm-project/vllm/pull/62183)|main|2026-10-19T06:52:21Z|`fc47cc8fff8282b7fa2b257468b8595f91a1a741`|
|[vllm-project/vllm #62206](https://github.com/vllm-project/vllm/pull/62206)|main|2026-10-19T12:38:19Z|`a608b6e6e05f4b309f77981f010e93c2caa1c677`|
|[vllm-project/vllm #62208](https://github.com/vllm-project/vllm/pull/62208)|main|2026-10-19T19:49:33Z|`8a1c02efca9bb0660f70b404e1ac4cb0629144ae`|
|[vllm-project/vllm #62269](https://github.com/vllm-project/vllm/pull/62269)|main|2026-10-19T03:55:55Z|`ed871c91c749d23d1d9a0110ea7655f3b7d4e048`|
