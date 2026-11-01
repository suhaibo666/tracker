# LLM 技术动态日报 · 2026-11-01

> 时区：Asia/Shanghai（UTC+8）。增量扫描窗口：**2026-10-31 09:02:00 → 2026-11-01 08:57:10**；72小时回溯：**自2026-10-28 09:02:00起**。实际生成时间：**2026-11-01T09:13:19+08:00**。下文原始UTC时刻保留Z；发现日期、论文提交日、工件更新日和合入日分别标明。
>
> 已先读自动化memory、持久化state/sources、25篇本地日报、GitHub main树/索引及上一期完整正文。26个历史归档blob（日报与索引）与本地一致，远端基线`0476f0f6001b5697c8fee39e9e9128e433eb5a25`，无旧待补传。仅维护llm-monitor和tracker/llm-training-daily，未改llm-knowledge wiki。
>
> **性能与模型成绩为作者报告；本轮独立核对源文、版本、部分最终代码、文件树和索引，不等于独立复训或GPU/NPU/TPU/多机复测。** 已合main、develop、支线、正式release与未合PR分别写明。cutoff只用于继续枚举事件，完整语义覆盖水位仍未建立；全部旧失败及未审队列继续保留。

## 本次重要变化

- **训练精度与资源寿命值得优先回归。** 新补读的loss-spike研究给出输出头有限精度机制；Megatron把GDP衰减参数留FP32、把RoPE/YaRN常量交给graph owner，并减少MXFP8专家双份常驻。来源中的小规模/高学习率和指定硬件实验不能外推成通用增益。
- **长上下文与低并发推理有可复现的具体路径。** vLLM persistent MQA scoring+top-k、FlashInfer MSA ICP/GDN block、TRT MiniMax-M3 GB300专用target各自处理不同成本；GDN单行decode可能更慢、部分收益仅在MTP和特定几何成立。
- **混合状态与外部KV的正确性修复比版本号更紧要。** SGLang收紧P/D目的页发布、sidecar、共享LRU及host checkpoint；Ascend补MLA外部KV等待和request-major采样mask。接收HTTP响应、取消请求、发布事件与物理内存可重用是不同完成条件。
- **部分新能力适合研究，采用条件仍未闭合。** vLLM direct reload缺少完整更新和失败执行隔离；QSA池化的硬件表使用较早栈；ATOM调度的测试checkpoint与最终head不同，且吞吐/TTFT权衡依负载变化。
- **架构质量必须和真实成本同看。** 本次补录Dual DeltaNet、loop scaling和GDN低比特状态研究；更强固定记忆、更多循环、标称4bit都不能自动换算成更快或更少完整存储。
- **数据和模型卡出现影响预算的实质修正。** Vela3完整router包和多次前向的分母已披露；Indic Specialized交付的是295.8B目标token子集，非整份2.8T语料；FairStress/WBench修正复现入口。已查表面未确认新基础模型权重，不据此声称全行业没有发布。
- **保留负面结果与旧结论更正。** Sandman暖池取得快不等于训练快，预注册组合实验质量下降；量化可能拉长CoT；跨token压缩统计可破坏因果；ParaTTT实现链接目前仍是待发布说明。普通日志/CI、文档补丁和重复PR报道已过滤。

目录：[模型与数据工件](#models) · [预训练与架构](#pretraining) · [后训练与评估](#posttraining) · [推理与服务论文](#inference) · [旧正式论文补读](#formal) · [核心开源工程](#core) · [扩展工程与硬件协同](#extended) · [覆盖与续查](#coverage)

<a id="models"></a>
## 模型与数据工件：部署口径修正与旧待审补录

### Vela3-fast：0.38B 是决策主干，全路由需要 6–7 次前向

**2026-10-31 11:00:50／52 UTC；fast `fa60e3aa`、fast-vision `c1f651ca`。** [fast 固定模型卡](https://huggingface.co/vllm-sr/vela3-fast/blob/fa60e3aa41132c60bf05a84cabfdced39cb6d4f7/README.md)和[vision 固定模型卡](https://huggingface.co/vllm-sr/vela3-fast-vision/blob/c1f651cafddca43860a855ee434a35697949bce1/README.md)补充部署分母：fast 的 UltraBERT 决策主干为 0.38B／1.5 GB，连同各任务适配器的完整包为 **1.65B／6.6 GB**；vision 每类问题调用一个约254M主干，包内却包含三个视觉主干、文本编码器及适配器，合计 **1.48B／5.72 GB**。

fast 将表述明确为“每个问题一次 encoder pass”；完整 router 调用为6–7 passes，不能把单决策约14 ms的口径套给完整路由。卡提供忽略 `*_adapter.safetensors`、只加载 `DecisionEncoder` 的示例，并称纯决策输出相同；`vllm_srun` 当前仍加载完整包。训练、原有评测与许可证未随本次更新变化；fast为Apache-2.0，vision为MIT，基础组件各有条款。

**建议：立即重核常驻内存、完整路由时延与下载预算。** 本轮对比前后两棵文件树，两个仓库都只有README对象变化，权重、配置与代码OID相同；这是重要模型卡修正，**没有新权重或新速度实测**。包大小／参数量与输出等价性属于作者披露，本轮未运行推理、未逐张量重算参数。

### Indic Specialized：可下载的295.8B-token子集，不等于整份2.8T语料

**回溯补录；仓库创建2026-10-26，最新上传10-29 07:01:56 UTC；`72d7d77a`。** [固定数据卡](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Indic-Specialized-v1/blob/72d7d77ab56569cf0a09069542df42d0a39c6185/README.md)公开的是来自 Nemotron-Pretraining-Specialized-v1／v1.1 的翻译子集，卡称167,954,507行、**295.8B输出tokens**，以Gemma-4-31B-it tokenizer统计目标 `text`；191.9B输入tokens另计，不能相加为目标语料规模，也不能把存储中的多语种／罗马字表示当作独立去重知识。本轮完整翻页文件树确认34配置、10,959个Parquet、1,253,140,731,121字节；平台索引34配置的行数合计亦为167,954,507（首轮一配置超时，重试恢复）；这是当前平台索引交叉核对，未逐行独立重数，也未重算输出tokens。

12个主要语言的原文与罗马字配置采用翻译→自动评估→低于75重写→复评淘汰，Gemma系列承担生成／评估；另10个低资源语言走Bodhan-MT及单独处理，**没有相同的整文rewrite阶段**，也不提供罗马字配置。21字段保留 `source_uuid`、`_source_repo`、`_source_config` 和逐行 `license`；低资源未使用的中间字段可为Arrow `null`，Manipuri中间字段有实际结构，混配置加载应统一nullable类型。

**建议：先按来源和语言审计一个CPT切片，再衡量英语保留与原语种质量。** 这不是新模型评测或完整预训练recipe：源数据和生成器的不可变版本未钉住、Bodhan生产checkpoint与部分转换细节未披露，自动judge未做人类校准，未给跨来源去重测量；本轮未重算全语料tokens、未复训。许可按来源组件区分CC BY4.0，以及部分CC BY-SA4.0／GFDL，不能统称CC BY4.0。先前已报的2.8T总语料技术报告不在本轮重发；配套[Indic-CC-v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Indic-CC-v1)的commit接口仍401，尚未完成该部分工件核查。

### 评测数据修正：FairStress分片选择与WBench资源路径

**72h回溯的旧待审修正，均非新benchmark。**

| 工件／版本／时间 | 核实变化与实践建议 | 证据边界 |
| --- | --- | --- |
| [FairStress `1642fd5e`](https://huggingface.co/datasets/allenai/fairstress/blob/1642fd5e021904cf196e68aa06e6697f898080d5/README.md)，10-29 04:19:08 UTC | `full`由会混入core的`data/*.parquet`改为`data/full-*.parquet`，`core`从不存在的`core_data/`改为`data/core-*.parquet`；示例命名空间改为`allenai/fairstress`。修复了10-21日报已指出的加载问题，建议更新固定revision后重查split样本数。 | 50项文件树前后仅README变化，Parquet对象相同；没有新样本／新公平性结果。本轮静态核验，未运行`datasets`端到端加载；CC BY4.0。 |
| [WBench `16b560d7`](https://huggingface.co/datasets/meituan-longcat/WBench/blob/16b560d77afaa51ed127eaf4fe81d9340289dd6a/cases/case_4.json)，10-30 07:52:33 UTC | `case_4`的图片／mask从带`data/wbench_preview/`的路径改成仓库根相对路径，并移除重复顶层字段。建议多轮world-model评测固定新revision，避免单个case因资源定位错误而缺失。 | 783项文件树中内容对象只有`cases/case_4.json`改变；资源文件未变。本轮未重跑视频生成／评分，不能据此改写榜单；卡标MIT，未扩展为全部外部资源许可认证。 |

### nesformer机制观察：把6502电路编译进DeltaNet状态

**名单外补录；2026-10-30 15:59:16 UTC；[作者仓库 `26b1b053`](https://github.com/dbreunig/nesformer/tree/26b1b053f826154498926892815574580dc45c24)。** 作者计算权重，将6502、2 KB RAM与32 KB NROM映射到7层Gated DeltaNet／MLP，另加零attention层满足标准`Qwen3_5ForCausalLM`缓存接口；配置为hidden768、词表263,808、bf16。主机负责图像渲染、控制器和中断／I/O输入，CPU逐token输出执行轨迹。

源读核实`weights.build_checkpoint`生成电路权重、`hfcheck.HFEngine.load_state`把机器初态写入recurrent cache，再由`step`执行贪心预测。作者报告原生Transformers在teacher-forced reference trace上检查2,651位置零差错；hf_check逐项喂入参考token并跳过host注入位置，不能与专用引擎长轨迹free-running混同。本轮未运行。**建议作为线性注意力状态表达与精确程序执行的机制阅读材料。** 无训练数据／优化过程，不能作为模型学会玩游戏或Qwen通用能力增强的证据；仅NROM／官方opcode、无声音等限制，原生CPU约10 token/s，专用numba引擎约1,650不能替代通用引擎吞吐。代码MIT，ROM分别授权。

本分支未在已查组织确认新的基础模型权重发布；这只描述已查表面。52组织208次模型／数据createdAt与lastModified查询到达回溯边界，仍保留动态官网历史、访问失败及未完成语义审查；元数据查询成功不等于全来源完整覆盖。


### ParaTTT既有论文工件核验：链接仓库仍只有待发布说明

**MIT CSAIL／Liquid AI／MPI；既有 `arxiv:2610.38255v1`，10月30日报已收录；本轮仅补核工件状态。** 10月30日定位的[作者仓库](https://github.com/ntumm120/paraTTT)本次冻结在 `a103dfcc4a0848e4a450e4e0db5d5ac1f7cde566`（10月27日20:51 UTC），完整树没有实现文件，README仅称“Code to come for ParaTTT”，无许可证。先前“作者实现入口”只代表链接已找到，不能解释为kernel已交付。本轮不重报论文结果、不登记新版本。

**建议继续观察代码发布。** 已重新打开[固定v1全文](https://arxiv.org/html/2610.38255v1)；Appendix D明确主训练序列4096，模型评估的4K—32K上下文是另一条件，保持两者分开。论文实读不等于可运行实现核验完成。

<a id="pretraining"></a>
## 论文：预训练、递归架构与训练稳定性

本轮优先清理前轮未审队列。以下有 13 篇 arXiv 论文为 **10 月 28—29 日原稿的回溯首次纳入**，另有一项既有论文工件核验及两篇明确标为旧正式论文补读；没有把周末未变的公告页当成今日新论文。方法、结果与局限均来自已打开的固定 v1 全文或正式 PDF；数值为作者报告，本轮没有独立复训或硬件复测。


### Loss spikes：输出头均值漂移本身无害，有限精度使它变成故障

**Zaiqian Chen 等作者团队；`arxiv:2610.38277v1`；原始提交 2026-10-28 04:53:13 UTC；实现MIT。** CE 梯度沿词表轴和为零，但 Adam 逐坐标预条件破坏该性质，输出头／logit 均值可长期漂移；精确算术中共同平移不改 softmax，BF16 中则可能抹去决定概率的 logit 差。论文用同一 hidden state 的 FP32 shadow GEMM 对比 native logits，区分数值误差与抽象优化动力学（§3.2—3.4）。[固定版本全文](https://arxiv.org/html/2610.38277v1)。

404M／96B tokens、4×B200 的高 LR 长训练出现延迟且不恢复的 time-bomb；28M／16B tokens、五 seeds、LR0.1 的受控精度实验中，BF16 true 最早失败、mixed 稍后，FP32 在所测 horizon 稳定。仅输出头用 SGD-M、其余保留 AdamW，可控制误差；对 AdamW 输出头同预算调 LR／提高 β1 至0.99能缩小优化差距，但最大 trunk LR 下仍不稳定。快速 spike 的 Table 3 中，404M tuned SGD-M eval loss 3.178±0.002，AdamW 3.916±0.278，但这不是正常生产 LR 下的普适收益。

**建议训练稳定性团队复现小模型精度对照，记录 shadow-logit RMS／KL。** 证据主体是人为提高 LR、过训练与28M精度消融；不能把所有 spike 归因于输出头。作者[JAX/Flax代码](https://github.com/tingtang2/quantization-loss-spikes)冻结 `71f48081`；已读 `training/optimizers.py::build_optimizer` 的 LM-head SGD 分流与 `diagnostics/quantization.py::get_output_logit_and_optional_metrics` 的同输入 FP32 shadow 测量，未执行实验。

### Dual DeltaNet：额外从 value 侧重排地址，召回收益与时延开销并存

**浙江大学／新加坡国立大学／湖州师范学院；`arxiv:2610.38578v1`；原始提交 2026-10-28 10:09:14 UTC；实现MIT。** KDA 从 key 侧修正已存 value；DDN 再用受门控 value 方向读出相关地址并修正，合并为 rank-2 recurrent update，仍每步读写一次固定大小状态，额外计算不为零（§3—4）。[固定版本全文](https://arxiv.org/html/2610.38578v1)。

§5／Table 2—6：130M—1B、FineWeb-Edu 约5B／15B／50B tokens、等参数及等有效 state size、2048训练上下文。1B 七任务均分 KDA50.37→DDN50.59，RULER四类needle跨1K—4K均分64.27→74.52；真实检索31.20仍低于GDN-2的31.80。H200、batch64、输入及输出均8192时，prefill+decode为33.37秒，KDA29.34秒、GDN-2 33.11秒；不能把更高算术强度写成 E2E 加速。

**建议线性记忆研究者阅读并复现精度／时延联合对照。** 规模止于1B，真实检索输入最多2048；训练效率各基线沿用自身默认 recomputation，非相同显存策略。作者[实现](https://github.com/ReichtumQian/DualDeltaNet)冻结 `d62f906b`；已对读 `fla/layers/ddn.py::DDN.forward` 的方向／gamma投影及 chunk/recurrent切换，`fla/ops/ddn/naive.py::naive_recurrent_ddn` 对应归一化value selector和payload。该层明确拒绝context parallelism；未独立训练或验证 kernel 数值。

### Loop scaling：两次循环可改善过训练算力效率，平行循环须重建依赖

**FAIR at Meta／MIT；`arxiv:2610.40325v1`；原始提交 2026-10-29 15:56:09 UTC。** 比较full-stack、per-layer、中间块及attention/MLP循环，进一步把跨loop同token依赖改为前token状态，使当前token各loop可并行；分别归一化当前embedding和前loop状态、加入rank8 gate，并保留每loop独立KV（§2—3）。[固定版本全文](https://arxiv.org/html/2610.40325v1)。

70M—1B标签尺度、10—160 tokens/parameter、同预训练混合与8192上下文，主要看DCLM BPB。§3.4／Appendix H：在所测过训练区间，100M—1B的两loop、80TPP能以较少训练FLOPs达到不循环160TPP的同等BPB；四loop未保持同样效率。归一化shifted PLT在1B／两loop接近unshifted控制，四loop差距扩大。**参数标签排除input embedding**，1B标签实际共1.230B可训练参数（Table1）。

**建议架构团队深入阅读对照设计，先小规模复现。** 结果限从零dense decoder、≤1B标签尺度，主要是似然评估；没有证明自由生成、后训练、MoE或前沿模型同样获益。PLT改版不再沿用原版KV sharing／gated sliding-window，不能同时归给它原版缓存节省；本轮未核实作者公开实现／权重，也不把依赖允许并行换算成实测服务加速。

<a id="posttraining"></a>
## 论文：后训练、检索与可靠性评估

### PIVOT：老师须在学生走错后接手，且直接监督接手片段

**国科大／清华／中科院软件所；`arxiv:2610.40576v1`；原始提交 2026-10-29 17:45:07 UTC。** 仅塞入成功老师轨迹不保证学生学会；仅在失败学生轨迹上给dense KL又可能缺少可恢复路径。PIVOT让学生先写512 tokens、老师接一段、学生继续；GRPO加只覆盖老师片段的KL，默认六条学生轨迹+两条辅助轨迹、top30并把剩余概率收为tail bucket（§3—4）。[固定版本全文](https://arxiv.org/html/2610.40576v1)。

Table2／Appendix A：Qwen3-1.7B逻辑题、Qwen3-4B数学，老师Qwen3-4B-Thinking；逻辑三epoch、数学一epoch。reverse-KL版Survo/Minesweeper/Kakurasu avg@8为72.90／25.44／8.38，对KDRL41.68／13.88／2.93；Math-ID23.05对最强baseline POPE22.09，增幅较小。AIME24/25采用avg@64，forward-KL63.33／52.45，对RL-only60.62／49.74。等老师token预算时少而长的干预优于多而短，但并非等GPU墙钟成本比较。

**建议稀疏奖励任务团队阅读并复现中段干预。** 数学筛出30K低成功率题，逻辑每类2K训练／500测试，范围有限；老师模型、轨迹长度、KL方向均影响结果。论文无完整公开实现链接，本轮未核实硬件规模／成品权重，不能将其写成已验证的训练提速。

### xPRT：无标注文档的跨段关系有用，精挑正例比复杂解码损失更关键

**Microsoft Research；`arxiv:2610.40122v1`；原始提交 2026-10-29 14:18:02 UTC。** 用文档代表anchor及四个相关siblings做跨段对比与masked-content预测；训练后丢掉一层decoder，仅保留12层encoder，推理结构不增加。正例由冻结E5和TF-IDF预先筛选，并不是无模型参与的自动监督（§2—3）。[固定版本全文](https://arxiv.org/html/2610.40122v1)。

Table1—3：E5-base-unsupervised继续训练，FineWeb100K／500K／1M命名语料，MTEB英文41任务均分53.06→54.86／54.93／54.89；“1M”指最初百万文档中满足5—64段条件者，并非必有百万有效文档。BERT初始、100K命名语料、六epoch、三seed，xPRT47.12，SimLM42.73；换成同文档随机正例降至42.22，移除跨段content项仅降至46.67。仅TF-IDF排序47.10接近完整47.12，不能把主要增益归给dense筛选或复杂预测项。

**建议检索团队阅读，先复现正例curation。** 每组五encoder+十个单层decoder前向，非与单段基线等FLOPs；英文、512-token评估不证明长文／多语泛化，E5保留正则对全部对手一致。论文称代码为匿名supplement，本轮未取得可公开复核的完整仓库／权重，未独立复现。

### CATT：数据归因分数会被无关token稀释，找回支持文档不等于证明因果

**Northeastern University／MIT；`arxiv:2610.40655v1`；原始提交 2026-10-29 17:59:53 UTC；实现MIT。** 为合成函数映射／routing规则控制每事实文档冗余和词面干扰；将文档平均CE梯度改为negative-log-odds并只聚合最高若干token，避免已学会的关键答案token被低CE梯度压掉（§2）。[固定版本全文](https://arxiv.org/html/2610.40655v1)。

§3／Fig3—7、Appendix B：Qwen3.5-4B LoRA，每配置三seed；50事实×50文档、1:1干扰的Grad-Dot例中，Recall@N从全token-CE的0.01，经仅top10的0.30、仅NLO的0.35，到二者组合1.00；这是特定格，非全部基准满分。外部FTRACE-TREX采用删文档后重训的准确率变化另检验。完整归因扫描约2400 GPU-hours，使用A100-80GB/H100/H200。

**建议数据筛选／可解释性团队深入阅读并采用强词面基线。** 单token、单任务微调，BM25本就很强；token梯度是含前文的局部信号，非删除该token的精确反事实，top-k在真实数据无真值时难调。作者[代码](https://github.com/Lamsheeper/catt)冻结 `de4de13a`，已对读 `losses.py::log_odds/sample_loss` 的排除target logsumexp与 `token_aggregation.py::doc_topk_mean` 的有符号top-k平均；工件可访问不等于本轮复训。

### MSafeTemp：多语模板扩增能降攻击成功率，也会改变误拒与数学能力

**ETH Zurich；`arxiv:2610.40516v1`；原始提交 2026-10-29 17:26:02 UTC；实现MIT。** 将英文安全seed拆成模板／实体槽、翻译和本地化，生成harmful、benign与wrapped变体，再SFT混一般指令数据、用模型自身采样构造DPO偏好；HarmBench保留evaluation-only（§3—6）。[固定版本全文](https://arxiv.org/html/2610.40516v1)。

Table1—3：HarmBench200 seeds×60语言、WildGuard2500 seeds×44语言。Llama3.1-8B IT→SFT+DPO后，HB/WG多语ASR15.9／10.5→2.0／0.8%，但MMLU72.0→63.6、GSM8K84.0→68.5；HB误拒4.5→5.3，不能称安全和能力无损。Apertus-v1.5-8B IT的MMLU70.2→64.0也有明确代价。Qwen3.5-27B在wrapped benign上误拒31.1%，即使wrapped ASR<0.1%，仍须同时读两类指标。

**建议多语安全团队考虑小规模采用并保留能力／误拒回归。** 主要LLM judge，独立judge只复核5K响应；人工语言覆盖少，每语1—2人，伤害标签由英文seed继承，单轮固定模板不能覆盖多轮自适应攻击。作者[代码](https://github.com/swiss-ai/msafetemp)冻结 `03fb65da`，数据构建／训练路径已取得，未下载完整语料、成品权重或执行训练；上游数据与模型许可证仍须分别读。

### OpenConjecture：随机抽题改善能力评估，但“模型评审通过”不是解决开放问题

**PNNL／宾夕法尼亚大学／UCSD／华盛顿大学；`arxiv:2610.40382v1`；原始提交 2026-10-29 16:24:55 UTC。** 从arXiv数学论文提取和二次筛选开放猜想，先均匀抽200篇论文，再每篇抽一道，避免按每篇猜想数或选定成功案例加权。三轮递进GPT-5.6 Sol、Sol Pro、GPT-6 Astra Pro，仅未决题继续，工具／输入上下文也随轮次改变（§3—4）。[固定版本全文](https://arxiv.org/html/2610.40382v1)。

§5—6／Fig2—4：200题产出70个候选反例、55个候选证明；同模型新实例检查中第一二轮约80—85%、第三轮全部通过。**这不等于125道新开放问题获解**：部分题后续文献已解决；许多反例是标准对象或利用遗漏假设，作者明确专家核验不在范围内。不同轮次同时改变模型、上下文、search权限，不能直接当模型排名。

**建议评测团队深入阅读抽样／文献状态／正确性三层检查。** 同族solver/reviewer可能共享错误，本文没有专家或形式化全面验收。[公开数据管线仓库](https://github.com/davisrbr/conjectures-arxiv)本轮冻结 `d2e3afe6`，仍是7月12旧snapshot与GPT-5.4 pilot README；不能宣称它完整复现本篇最新200题实验，数据revision和最新运行产物还待核实。

### Rewindable oversight：验证者少问oracle的形式保证，需要可恢复prover与鲁棒计算

**MIT／Succinct；`arxiv:2610.40535v1`；原始提交 2026-10-29 17:34:28 UTC；Apache-2.0实现。** 对少量oracle答案改动不变的鲁棒电路，先让prover固定编码见证，再每次查询都恢复到固定proof string的状态，防后续答案随查询适应；由此把昂贵oracle检查降为抽样（§2、§5）。[固定版本全文](https://arxiv.org/html/2610.40535v1)。

§6／Table1—2：单Graviton4核、50个预筛鲁棒MNIST图像、oracle为预计算table lookup；最大2^26 views时验证者问80次、online time为直接全量0.12×，prover却约58.8×。BLAKE3同IOP对SNARG编译，prover约1.77—2.15×更快，比较的安全／交互合同也不同。此处**没有实测LLM监督任务**。

**建议安全理论研究者阅读，工程采用继续观察。** majority基准甚至可由验证者自行抽样估计，无prover也能近似；一般电路预处理、真实模型状态save/restore成本未计，现实现开证据成本约sqrt(trace)，不是理论polylog。作者[Rust实现](https://github.com/ZiyiGuan/oversight-rip)冻结 `a9546b55`，有结果和重跑脚本；README说明最大MNIST重跑需约370GiB驻留内存，本轮未重跑，不把低验证成本写成低总成本。

<a id="inference"></a>
## 论文：推理预测、状态压缩与服务系统

### Sparse-MoE SD：专家重叠和GPU利用率共同决定验证成本

**Cohere；`arxiv:2610.38981v1`；原始提交 2026-10-28 22:04:53 UTC。** 仅按多token激活的专家字节预测投机解码，会漏掉连续token路由重叠、不同GEMM带宽利用率和L2复用。论文把不含acceptance的target efficiency（AR单步耗时／SD验证耗时）与E2E speedup分开，再用pairwise overlap约束的最大熵IPF造合成路由（§3、§6—8）。[固定版本全文](https://arxiv.org/html/2610.38981v1)。

十个MoE、默认draft3、vLLM `7c080dd3c`，H100为主，Kimi-K2.5／DeepSeek-V3用B200。Table1、5：Qwen3-30B FP8 TP1 batch1，专家字节模型TE0.511、实测0.818；H100中MoE占67%权重却仅33%GPU时间，attention GEMM带宽22—25%peak，MoE45—49%，L2 hit AR5.72%→SD27.79%。Mixtral大专家的对应TE0.428→0.488，不能套同一修正倍数。IPF对所测unique-expert数最大误差<4%，不代表能等精度预测所有服务时延。

**建议MoE推理团队深入阅读并在本机profile低batch区间。** 不少细粒度MoE最佳speedup在batch1附近，非一概中batch最佳；32K长context只补测Qwen3-30B，其他硬件kernel会变机制。MoE drafter部分是dummy权重／合成路由设计研究，没有已训练drafter接受率收益；论文无作者实现链接，本轮未核实完整公开代码，未独立复测。

### GDN low-bit state：把最近写入暂缓并入低精度历史，必须把队列计入位宽

**浙江大学；`arxiv:2610.39931v1`；原始提交 2026-10-29 12:33:21 UTC。** GGA按预测loss梯度敏感度全局分配head精度；Pure-Write把最近8次低秩写入独立传播／读取，老分支退出窗口再合入每步量化的历史state，减少最近结论过早损坏（§3—4）。[固定版本全文](https://arxiv.org/html/2610.39931v1)。

Table1、3、7／§5：Qwen3.5-4B／9B、24 GDN层、每层32个128×128 heads，A100-80GB、SGLang0.5.12、最高100K thinking输出、七任务。9B标称@4的STEP均分37.42，GGA67.26、GGA+PW78.63，FP16 state79.58；完整存储分别约4.755、4.749、**5.780 bits/value**，含scale、FP16 pivots和队列，不能宣传成纯4bit。近预算GGA@5为5.679bit／78.22分、平均输出26.78K；GGA@4+PW为23.89K／78.63分。

**建议hybrid模型团队阅读，先验证长推理质量再测runtime。** 仅两模型、固定窗口、有限样本；某些@6+PW还用PW轨迹重新校准分配，非完全只加窗口。§6明确尚无专用高效PW kernel，**未报告E2E latency／throughput收益**；短输出不等于低单步成本。论文未给作者公开实现链接，本轮未核实代码／权重发布。

### Prefill长度预测：先量同prompt的随机性，理论界依赖分布假设

**西安交通大学／山东大学；`arxiv:2610.40541v1`；原始提交 2026-10-29 17:36:01 UTC。** 将输出长度视作随机停止时间，显式区分采样自身方差与预测器误差；在条件高斯工作模型下，给出MAE下界约0.798σ和容差准确率上界，而不是声称任意长度分布都有同一数值界（§3）。[固定版本全文](https://arxiv.org/html/2610.40541v1)。

§4—5／Table2、4：Qwen3-32B no-thinking、ShareGPT18K训练／验证和300测试prompt、每测试prompt200次decode，temperature0.6/top-k20/top-p0.95；MLP-Concat+LN MAE168.13、Acc@10048.63%，估计的Gaussian界129.5／57.9%。26.5%非退化prompt在Shapiro-Wilk检验不拒绝正态，75.8%在KS不拒绝；模型只是对有关容差的近似，不能把拟合界当严格实测天花板。跨域LMSYS→该测试MAE182.40，数据组合会改变任务难度。

**建议服务团队考虑加入每prompt多次采样方差与相对误差指标。** prefill与decode在不同worker／batch产生，kernel与determinism未严格控制，作者承认可能混入数值偏差；没有调度SLO收益证明。[代码](https://github.com/ShwStone/DecodingPredict)冻结 `306dbf0c`，核心预测／分析脚本可访问、README称MIT但API未识别许可证文件，本轮未运行；不要仅按绝对MAE跨模型排名。

### Sandman：沙箱生命周期的完成条件，比“建pod更快”更影响RL质量

**Yiming Yan 等作者团队；`arxiv:2610.39711v1`；原始提交 2026-10-29 09:45:57 UTC。** 共享Kubernetes上把每次create交给provider持续持有，调用者取消后仍等其settle；API报错可能已经存pod，以nonce回查，delete必须确认，命令超时则清process group。trainer、pool、provider与execd各有边界，动作／grading用独立sandbox（§3—4）。[固定版本全文](https://arxiv.org/html/2610.39711v1)。

§5／Table2、4：10月3—17日122个训练run、约21,400 B200-hours、3.13M rollout+grading sandboxes；典型Qwen3.5-35B-A3B异步GRPO16×B200。warm pool三对实验使grading acquisition中位数6.2→0.12秒，但sandbox vCPU-hours变2.46×；到step24快9.4%，steady-state间隔CI跨零，不能宣称50×训练提速。12K控制面ramp用0.1vCPU/128MiB小pod+缓存43MB镜像，只证明该速率下并发承载，非真实SWE持续吞吐。

**建议agent-RL平台团队深入阅读故障合同和预注册负结果。** 六对150-step训练的“删三任务+warm/reuse+token”组合，让held-out resolve48.4→46.1%，差-2.33点、CI[-4.22,-0.43]，三件事不能分开归因。生产期间并非所有当前默认防线都已开启（Table4／Appendix A）；root runc共享内核且信任自家trainer，不代表隔离任意敌对多租户。正文未给作者完整公开实现，本轮未核实可运行发布，未独立复测。

<a id="formal"></a>
## 旧正式论文补读：因果性与低比特服务评估

### 序列轴压缩统计：attention有causal mask，前向仍可能偷看未来

**NVIDIA；ACL正式版 `2026.emnlp-main.2432`；论文集标注2026-10-24—29，在线首发精确日未核实；旧作补读。** `Breaking Causality in LLM Training with Sparsity and Quantization`说明，跨token挑2:4 mask或共享block量化scale，会使past token变换依赖future activation。shuffling可以推迟利用，却不移除路径；前向用逐token因果算子，非因果算子只留WGRAD的forward-bypass才断开预测依赖（§2、§5—6）。[正式PDF](https://aclanthology.org/2026.emnlp-main.2432.pdf)。

Table2／Appendix B、D：162M机制实验主要8×H100、2.5B Nemotron-CC-v2 tokens；Llama3.2-1B checkpoint续训5B tokens后，hidden-axis causal的train/causal-val为2.418／2.411、HellaSwag60.9%；sequence-axis+shuffle64为1.726／4.969、HellaSwag31.7%，训练loss更好而生成质量大坏。shuffle512仍2.427／2.695；小规模暂未失败不能证明安全。

**建议低精度／稀疏训练团队深入阅读并审查前向统计轴。** 多数实验单run，仅证明故障存在，不预测必然发生时间；bypass有前后向不一致成本。作者明确现代NVFP4／部分激活稀疏recipe已采用bypass，不能泛化成所有现有FP4训练均泄漏；本轮未取得作者独立代码仓库或复训。

### Quantization Inflates Reasoning：少每token字节，未必少整题成本

**UIUC／Microsoft／Anyscale／Snowflake；ACL正式版 `2026.emnlp-main.2059`；论文集2026-10-24—29；对应早期 `arxiv:2606.25519` 首稿2026-06-24，旧作补读。** 按任务先比较量化／BF16平均CoT长度，再对任务等权宏平均CTIR，另看双方都答对的子集，避免只用准确率或tokens/s评价量化（§5—6）。[正式PDF](https://aclanthology.org/2026.emnlp-main.2059.pdf)。

Table2：Qwen3-4B GPTQ INT4宏平均CoT膨胀12.0%、INT3 48.7%；Qwen3-30B对应3.7%／19.0%，并非每任务每quantizer都增长。H100上Qwen3-30B-A3B、1K prompt、batch64的**受控长度压力实验**，人为多10%CoT使p90 first-visible-token1050→1400秒、throughput0.055→0.047req/s；这是隔离长度效应，不能写成该INT4 checkpoint真实比BF16慢33%的直接测量。

**建议量化部署团队考虑同时报告质量、token长度、单token速度与整题时延。** 主实验weight-only INT3/4、固定抽样子集greedy；QAT／prompt／采样减长的代价依配置，非通用修复。first-visible-token指隐藏思考后首个用户可见token，和常规engine首decode-token TTFT不同。正式PDF已读，完整作者实现／服务原始trace未核实，未独立复测。

<a id="core"></a>
## 核心开源工程：权重寿命、长上下文索引与服务完成边界

主窗 2026-10-31T01:02:00Z → 2026-11-01T00:57:10Z；回溯自 2026-10-28T01:02:00Z。本节采用维护组织/社区一手 PR、最终 changed-files 与固定 merge SHA；作者测试和本轮静态核验分开，本轮未运行 GPU/NPU。

### FlashInfer MXFP8 专家只保留 routed 权重副本

[NVIDIA/Megatron-LM #8560](https://github.com/NVIDIA/Megatron-LM/pull/8560)，2026-10-31T14:08:31Z，main `dda809cad59315fcb304bb5e8a36ea30880f3753`。

原来canonical MXFP8与内核读取的Major-K布局同时常驻，refit后还要重排。现给canonical对象装writeback，copy_/quantize_直接量化并写入routed slot，原data/scale换成保形占位，refit仍同步后才replay。作者Nemotron-3-Ultra两节点TP4/EP8、128×512 token实验峰值189.9→131.9GB、生成128/128相同。 有训练/rollout同卡与MXFP8 refit需求时优先读；自定义代码须从routed stack读取权重，不能再dequantize占位data/scale。 **证据限制：**显存与输出结果为作者指定实验；本轮只核对最终代码，未复跑。

### RoPE/YaRN 常量寿命从普通 LRU 提升到 CUDA graph 所有权

[NVIDIA/Megatron-LM #8358](https://github.com/NVIDIA/Megatron-LM/pull/8358)，2026-10-31T23:23:52Z，main `dae43b6c9c0c1adf8a444543c14b40f1154f6493`。

同卡RL切换阶段清理缓存后，已capture的位置编码地址可能被训练张量复用。普通cache保留32项且用弱引用区分embedding实例、device/dtype/CP等；capture时register_capture_constant将tensor交给实际graph/callable持有，销毁前先同步，再reset/release。未找到graph owner的capture保留到进程结束并告警。 同卡零logprob-mismatch训练应回归清cache、GC、graph重建后再rollout，区分缓存释放与图完成。 **证据限制：**作者20步KL为0是有限实验；代码覆盖local/full-iteration/TE owner接力，本轮未执行NeMo-RL外部调用链。

### GDP 衰减参数在 BF16 训练中真正保留 FP32

[NVIDIA/Megatron-LM #8528](https://github.com/NVIDIA/Megatron-LM/pull/8528)，2026-10-31T21:01:55Z，main `cc2d9c9ee189c68d2a39a49c67b2640b5aae1952`。

A_log虽以FP32初始化仍会被Float16Module下转，dt_bias原本直接BF16初始化；两者共同改变每头衰减。现dt_bias用FP32随机/逆softplus初始化，两参数均mark_keep_in_fp32。旧非分片optimizer master列表在参数dtype迁移后按顺序、shape与低精度舍入匹配；不能匹配则报错，避免错位恢复。 GDP训练与旧checkpoint恢复应检查参数dtype和optimizer加载，不能只看消费者内部曾upcast。 **证据限制：**改变了初始化精度，不能说旧权重精度自动恢复；新增checkpoint匹配检查失败时需按报错选择无optimizer状态或sharded格式。

### GDP 的 GTP checkpoint 按本地片段保存，去掉整矩阵 all-gather

[NVIDIA/Megatron-LM #8530](https://github.com/NVIDIA/Megatron-LM/pull/8530)，2026-10-31T17:09:32Z，main `f397320061616df2929961febd3b43b0b232b19a`。

GDP in_proj的z/V/K/Q/b/a段界与GTP行片不对齐。新factory对每段与本rank行区间求交，用本地view和全局offset保存；pad行不落盘，全pad rank用本地非持久空对象保留key/device，load重建本地片并清零pad。tensor-aware恢复改为在完整state dict上合并factory。 大GTP checkpoint可读其按区间交集建chunk的实现；回归TP×GTP重分片、纯pad rank及GTP↔非GTP加载。 **证据限制：**验证合同是checkpoint布局/往返；本轮没有I/O吞吐实测，不把去掉gather直接折算成训练加速。

### MFSDP 按真实调用边界包 ShortcutMoE，并拆开构造作用域

[NVIDIA/Megatron-LM #8312](https://github.com/NVIDIA/Megatron-LM/pull/8312)，2026-10-31T18:09:34Z，main `f44e54a36e62a1e2e08a0c2f2c3d673e20fc4057`；[NVIDIA/Megatron-LM #8479](https://github.com/NVIDIA/Megatron-LM/pull/8479)，2026-10-31T02:14:55Z，main `c94825af68c10bec9b9dcd384ba839aca05a93ab`；[NVIDIA/Megatron-LM #8585](https://github.com/NVIDIA/Megatron-LM/pull/8585)，2026-10-31T19:15:58Z，main `3249e40088dcc36c4b90537fdfa49af12be9de2c`。

ShortcutMoE调用子层helper会绕开子层__call__ hook，故外层block拥有residual/norm及helper层参数，experts保留独立expert-DP单元；collector按子先父后包裹，普通MoELayer不额外拆成单元。fully_shard_context现接收显式FsdpContext，执行hook和梯度回调数归SchedulePolicy；state-dict安全hook独立保留。 升级实验MFSDP v2调用方须迁移构造API、自定义unit列表和SchedulePolicy，并检查residual_stream recompute被明确拒绝的边界。 **证据限制：**ShortcutMoE比较为作者同环境有限配方；本轮未运行多机训练。配置搬迁本身不等于新的prefetch算法。

### 无 EP 同伴时每步刷新工作共识，避免空闲 dummy forward

[NVIDIA/Megatron-LM #8504](https://github.com/NVIDIA/Megatron-LM/pull/8504)，2026-10-31T23:54:29Z，main `38dd3e993a600bd226ade3eba94380b220a0a8dd`。

EP=1沿用20步缓存的global_work时，prefill已排空仍会走dummy_forward；recurrent slots可能全被未完成P/D handoff占有而触发断言。现单rank路径每轮刷新本地共识，不再凭旧工作量造dummy；EP>1的摊销共识与专用dummy slot保持原路径。 混合模型P/D部署应复现decode饱和、全部state slot被handoff持有后的prefill排空。 **证据限制：**作者复现为GB200 Nemotron-3-Nano、3P+1D；并非任意队列/EP死锁的通用修复。

### 融合 sigmoid router 的最终数值合同小于 PR 标题所称范围

[NVIDIA/Megatron-LM #8296](https://github.com/NVIDIA/Megatron-LM/pull/8296)，2026-10-31T14:08:31Z，main `faf27c82f9b91c628ea941d8cb869e1489ad19fe`。

最终diff默认选择Triton融合sigmoid、带bias选择、归一化/scale，限CUDA FP32 logits、专家≤512与top-k 2–32；batch-invariant、group routing、replay等保留Torch。源码明确top-k<8概率逐位一致，而top-k≥8因tree reduction可差最后1bit。 读最终sigmoid_topk.py与InferenceTopKRouter选择条件；零mismatch路径应维持batch-invariant验证。 **证据限制：**PR正文仍描述大top-k仅替换aten.topk并声称全量bit-exact，与最终代码不一致；其早期候选speedup表不当作最终实现性能结论。

### 训练启动与大 DP 验证边界修复

[NVIDIA/Megatron-LM #8629](https://github.com/NVIDIA/Megatron-LM/pull/8629)，2026-11-01T00:40:36Z，main `c9d72ebaf46e382fb20239bd2310c6f52fa99926`；[NVIDIA/Megatron-LM #8579](https://github.com/NVIDIA/Megatron-LM/pull/8579)，2026-10-31T11:27:27Z，main `8b2d9b0e825d5d81536c72c332017fbb9ab48bf7`。

unset eval_interval现在按训练后一次验证估算样本，恢复训练时也不再除以None；eval_iters=0可得零验证样本。DDP gradient scaling断言改用rel_tol=1e-9的math.isclose，避免如1/72/5与1/360的浮点运算次序差导致360卡启动失败。 受影响训练配方回归不设eval interval、train-only、resume与非整齐DP组规模。 **证据限制：**这是样本预算与断言修复，不改变梯度缩放数学或证明所有360卡配置可用。

### 补查：单 rank reduce-scatter 走恒等路径，checkpoint 删除无效 GC

[NVIDIA/Megatron-LM #8449](https://github.com/NVIDIA/Megatron-LM/pull/8449)，2026-10-28T20:20:43Z，main `79f559413f93abb9d81a63d4375740631ff7f3c7`；[NVIDIA/Megatron-LM #8522](https://github.com/NVIDIA/Megatron-LM/pull/8522)，2026-10-27T17:31:18Z，main `29319fd9fd828cda40c212369dd4bb3f00015abe`。

10-28的reduce_scatter_with_fp32_accumulation在group size1时只在地址不同时copy，async返回已完成handle、sync返回None；跳过all-to-all与FP32归约。10-27的checkpoint保存删除FP8专属gc.collect，作者认为临时反量化引用已释放。 极小DP组与checkpoint telemetry可分别核对完成handle和暴露保存时延，避免只看训练日志内计时。 **证据限制：**8522是回溯窗外旧队列补录；作者约30%指checkpoint exposed time，未给完整负载/本轮实测，不写为训练吞吐提升30%。

### fused SwiGLU 注册到检测到的 accelerator

[pytorch/torchtitan #5466](https://github.com/pytorch/torchtitan/pull/5466)，2026-11-01T00:01:54Z，main `c25592f246c15732fc5dc520ec0064c7e1ff07af`。

DSv3/Muse配方默认override的两个custom op曾固定device_types=cuda，XPU首forward会报未实现。现按Torchtitan device_type注册，算术Triton kernel未改；测试按torch.accelerator与注册device是否一致决定执行。 XPU用户可检查默认配方override；其他仍CUDA限定的override需单独核验。 **证据限制：**作者PVC、torch2.16dev/triton-xpu3.9运行17项通过；不能从注册泛化推定所有accelerator/Triton后端均已验证。

### SM100 稀疏 prefill 把 scoring 与 top-k 合为一次 persistent launch

[vllm-project/vllm #60618](https://github.com/vllm-project/vllm/pull/60618)，2026-10-31T19:01:19Z，main `25fc60723cf6c92c041b0c665450023b96544e9a`。

避免写回再读取rows×keys logits：8-row任务块经采样gate筛候选、24-bit score code选择；候选不足/过多、不安全weights或scratch不足则块内FP32 dense rescan。默认仅FP8、32×128 index heads、top-k512/1024/2048，planner最多16K rows/1M keys，按32K/64K可见key阈值及末波占用选择。 长上下文GLM/DSV4.1团队优先读dispatch与fallback，量TTFT及workspace挤占KV预算。 **证据限制：**B300 fixed workspace作者报296MiB；BI、MXFP4、纯DCP和两级candidate层走dense，边界ties不保证相同集合顺序；准确率表早于最终refactor，本轮未复跑，不能把4.52×kernel数当全服务收益。

### direct 权重重载已合主线，但尚缺失败期间执行门禁

[vllm-project/vllm #64336](https://github.com/vllm-project/vllm/pull/64336)，2026-10-31T06:50:27Z，main `b5755c3ba5d4f751445be39c8ad8a50d9a8f93c1`。

opt-in VLLM_RELOAD_MODE=direct在冷启动记checkpoint布局，begin于原storage建view并恢复loader属性，finish同步后重跑post-load；减少逐层meta重建，默认仍layerwise。拒绝LoRA、online quant、torchao、offload、quantized draft head与sharded_rdt等组合。 适合深入读RL权重同步生命周期；生产试用前先补全更新完整性、地址稳定与pending/failed执行隔离验证。 **证据限制：**PR明确没有rollback、没有完整性校验或运行门禁；失败后可能继续用checkpoint-layout bytes服务，部分平台重绑定与派生常量仍是后续工作，因此不直接建议生产打开。

### NIXL 用共同 layer 身份配对每次 KV transfer

[vllm-project/vllm #63416](https://github.com/vllm-project/vllm/pull/63416)，2026-10-31T05:51:20Z，main `8b55f9195a66363a64a0388fad31f9b6ea099b51`；[vllm-project/vllm #64230](https://github.com/vllm-project/vllm/pull/64230)，2026-10-28T08:50:33Z，main `066d9799347c88c14e91baa9d849e9595ff8a3d3`。

握手统一advertise region_members/layer_group_ids，按共同layer建立remote/local/group三元配对，支持K/V分region、shared region和packed-row展开，拒绝缺layer、数量/布局不一致；connector版本15→16。另10-28补录transfer_mode=push让通用NixlConnector解析到既有Push类。 P/D双方一起升级并回归shared-layer、packed caches与异构TP；Dynamo类仅允许通用名字的集成可用push选择器。 **证据限制：**PP+HMA pull与SSM hybrid PP仍被拒绝，group不一致仅纯MLA pull有region-mapped fallback；本轮没运行NIXL/RDMA，不把配对重构说成放开所有拓扑。

### Rust frontend 预计算 prompt block hashes，移出 engine GIL 热路径

[vllm-project/vllm #63842](https://github.com/vllm-project/vllm/pull/63842)，2026-10-31T19:38:39Z，main `e318ac09796468ac31139d64d609143874e0ed3b`。

engine握手给算法/block size/NONE_HASH/lookahead，Rust对所有engine相同配置生成byte-identical SHA256 pickle/CBOR链并随request传入；engine先验hash数量，后续生成沿已有链继续。LoRA、salt和lookahead被编码。 长缓存prompt在decode中持续到达的服务可读CPU瓶颈与cross-language vectors；保留未知输入的engine fallback。 **证据限制：**多模态、prompt embeds、xxhash、pickle超单帧与可能对象memoization的请求仍回退；作者Qwen3-0.6B/GB300特定负载+9.6%/+39%依hash算法，非大模型通用幅度。

### MRV2 prompt logprobs 降低 chunk 拼接峰值

[vllm-project/vllm #64841](https://github.com/vllm-project/vllm/pull/64841)，2026-10-31T18:48:17Z，main `42258ca4afe8d3cee728a2d7cf52091f3ace15ff`。

每chunk结果直接写预分配输出，避免列表累计后torch.cat双份保留，token id改int32，单chunk仍直接返回。作者2048-token、152K词表full-vocab输出约从7GiB降到2.3GiB加一个chunk工作集。 需要full-vocab prompt logprobs或MIG小显存时检查warmup可用KV容量。 **证据限制：**CPU副本未改；跨scheduler step的PromptLogprobsWorker仍可累计/拼接，所以不能说所有logprobs瞬态已消失。

### batch invariance 固定 FlashInfer bmm 与 W4A16 backend

[vllm-project/vllm #62192](https://github.com/vllm-project/vllm/pull/62192)，2026-10-31T15:25:10Z，main `54beee831b0fd2df2a19967b7dcea661c32044c5`。

VLLM_BATCH_INVARIANT下FP8 bmm从auto固定cutlass，避免autotune按M bucket换backend；NVFP4 W4A16改选Humming，修原强制W4A4-only CUTLASS初始化错误，并对冲突linear-backend告警。 确定性训练/rollout路径应在autotune前后、跨batch与padding组合比较输出。 **证据限制：**已读vLLM分派和新增测试；FlashInfer/Humming内部只按外部接口合同，本轮未审计内核或扩写成所有路径逐位一致。

### PP warmup 共识与 DeepSeek-V4.1 PCP 边界收敛

[vllm-project/vllm #64338](https://github.com/vllm-project/vllm/pull/64338)，2026-10-31T15:24:05Z，main `4aba9bf3a4da72745b1bcb2e0870a1dd6e9c4861`；[vllm-project/vllm #63098](https://github.com/vllm-project/vllm/pull/63098)，2026-10-31T04:41:30Z，main `2b009fc6d1493a9a1ad19f4d0c7230b7e6b7e63a`。

FlashInfer warmup用CPU collective让PP ranks作一致决定；最终V2在任一rank有FlashInfer时全员参与，V1要求全员满足。DSV4.1移除模型PP=DP=1限制，并在sequence-parallel的PP发送端gather、接收端shard pre_mix。 混合target/drafter backend与PP/PCP组合应重新回归启动、跨stage shape和真实生成。 **证据限制：**最终diff不同于正文旧的PP+DP排除描述；仍DCP=1、无microbatch/KV transfer/MegaMoE，PCP+PP不支持spec；作者仅验证两种H20拓扑，不推出所有组合可用。

### MRV2 draft 配置与 DSpark 私有量化 lm_head

[vllm-project/vllm #64751](https://github.com/vllm-project/vllm/pull/64751)，2026-10-31T20:21:57Z，main `d8fad9202a52744efc7c34a67df6036b8e7cc023`；[vllm-project/vllm #64368](https://github.com/vllm-project/vllm/pull/64368)，2026-10-31T10:23:37Z，main `7b77ed581d92fa14e34ab43344837bb1ca1b4257`。

standalone draft_model现统一apply_draft_overrides，真正覆盖MoE/attention/KV dtype且保留target配置。DSV4/V4.1 DSpark可显式用FP8/NVFP4 target-head私有副本，target仍用原head验证，helper检查可共享与head dtype。 投机解码调优先检查加载后的实际backend；量化draft head后再量接受长度/ITL，不把小kernel变快直接等同服务提速。 **证据限制：**作者MI355X FP8 c1 ITL改善2.4–4%，c8以上近中性；非所有DSpark架构、V1 runner或不共享head可用。

### MoRI EPv2 backend 在可用拓扑走 CCO 对称窗口

[vllm-project/vllm #61361](https://github.com/vllm-project/vllm/pull/61361)，2026-10-31T08:54:47Z，main `d5301a4f5487db1d7e8d87daf7828225b95b1774`。

新增all2all-backend=mori，默认在符合架构/版本、BF16与激活合同、LSA拓扑等条件时用EPv2，否则警告并选EPv1；显式VLLM_MORI_EP_V2可强制版本。EPv2启动先查实际LSA team，按op arena计费复用窗口，AITER输出可直接写combine buffer；EPv1跨节点按per-rank token budget选择LL/V1。 AMD EP集成先核对MoRI>=1.2.4、gfx950/1250、实际LSA team、激活scale格式和AITER shared-expert fusion开关，再跑真实通信/服务。 **证据限制：**跨节点RDMA EPv2尚非本实现；强制1仅跳自动fallback，不豁免LSA失败。作者8×MI355X DSV4-Pro 8K/1K约13%输出吞吐优势测于更早命名revision，本轮未复跑；不是所有FP8/FP4激活路径已支持。

### ROCm 长上下文与投机缓存修复看 query-length 和 checkpoint 边界

[vllm-project/vllm #63116](https://github.com/vllm-project/vllm/pull/63116)，2026-10-31T02:49:26Z，main `c4f54cf5b369eb29e4f02c94bd4433a9c3078379`；[vllm-project/vllm #63297](https://github.com/vllm-project/vllm/pull/63297)，2026-10-31T02:47:57Z，main `e3eec51570d7a44a452bd3148f1146c3b26ffb3c`；[vllm-project/vllm #64094](https://github.com/vllm-project/vllm/pull/64094)，2026-10-31T02:46:49Z，main `138dfa54636c5c51c86b25346f38667e390c03e2`。

AITER MLA persistent metadata对1..decode threshold每个qlen求各维最大容量，修非单调tile需求。Kimi-K3 gluon普通步继续AITER，携checkpoint步交给可用fused KDA；DSV4.1 prefix-cache SWA bounded replay仅在已验证CDNA4/gfx950解除强制禁用。 MI355X的DSpark、chunked-prefill与prefix hit应分别回归warmup、backend切换、snapshot内容和cache hit。 **证据限制：**MLA验证有dummy-weight部分，KDA存在同base复现的11项builder失败；CDNA3仍禁prefix replay，不能把三个修复合成所有ROCm模型支持。

### MoE-LoRA ≤64 token 改用并行 shrink/expand

[vllm-project/vllm #64653](https://github.com/vllm-project/vllm/pull/64653)，2026-10-31T02:24:52Z，main `042f30f6e8a9ce4c1c7769d528a96b654164bbb0`。

原one-shot串行K循环、decode grid过小；新分派令小batch走既有两kernel路径，并对TMA 16-byte stride和显式TensorDescriptor import加保护。 低并发LoRA-MoE可测weight冷/热cache和batch切换，检查精度与BI路径。 **证据限制：**本PR阈值是64；256阈值、FP32 intermediate与删除small-batch移到后续PR。本版intermediate仍BF16，作者说明误差略变，不能报后续栈的单次舍入语义。

### EPLB 默认补齐最少 redundant experts

[vllm-project/vllm #30614](https://github.com/vllm-project/vllm/pull/30614)，2026-10-31T01:35:55Z，main `8ecf86ba83c547ea33eb5c75ffa88004c86cab2a`。

omitted值从0改为None，再按effective EP=TP×PCP×DP计算(EP-logical_experts%EP)%EP；disabled时归零，显式非负值保留，负数拒绝。 非标准EP规模应检查physical expert总数与新默认显存预算，显式值仍需自己满足模型约束。 **证据限制：**配置归一化修复，没有新负载均衡算法或吞吐测量。

### skinny GEMM 调度补 SM121/SM103 的低 token 形状

[vllm-project/vllm #62956](https://github.com/vllm-project/vllm/pull/62956)，2026-10-31T13:55:24Z，main `e5196e2959a42908313510f55fa21f213a48802e`；[vllm-project/vllm #64650](https://github.com/vllm-project/vllm/pull/64650)，2026-10-31T07:50:08Z，main `900b8b08e2552c28a004e075866b5e1a39b99ac3`。

SM121新增按capability、本地(N,K)和M=1/2/4查表的通用BF16无bias linear/lm_head入口，不满足计划/stride则F.linear；SM103增加Qwen3.8 TP1专用shape计划。 DGX Spark/Station部署按实际TP local shape核对命中率，再跑服务而非只看microbenchmark。 **证据限制：**作者SM121 GLM5.3-Flash双Spark TP2 c1无spec TPOT64.0→54.1ms，MTP收益更小；SM103本次只有kernel测试、无E2E，不横向相加。

### GLM-5.3 FlashKDA 直接消费 row-strided Q/K/V

[vllm-project/vllm #61614](https://github.com/vllm-project/vllm/pull/61614)，2026-10-31T03:29:25Z，main `602811546b035a5ff35256a0a008b61071b13904`。

升级FlashKDA固定commit到47fc9fec后，删除三份.contiguous()，把packed q|k|v的对齐token-stride视图直接交给fwd；g仍contiguous，beta布局保持原接口。 短prefill服务可确认消失的copy kernels，并固定匹配的FlashKDA构建。 **证据限制：**作者4×H200 TP2/DP2+MTP的2048/1负载饱和吞吐约+1.9%；消费者diff证明传参，依赖内部正确性仍需其接口与实测，非零拷贝所有KDA路径。

### ColBERT token head 跨请求批量投影

[vllm-project/vllm #64269](https://github.com/vllm-project/vllm/pull/64269)，2026-10-31T02:35:28Z，main `16332714512d30a134f8a14840352c4856367b2a`。

AllPool且非chunked-prefill时先对total_tokens作cast/project；相同dimensions/activation一并finish后split，混合设置只共享project再逐请求finish。late chunking、BI保留原路径。 rerank服务可按文档数、混合维度/归一化与输出buffer ownership回归。 **证据限制：**作者指定33项/L128/C32/eager配置H100+B300有33–40% req/s提升；旧SciFact表未在最终refactor重跑，本轮不推广到任意embedding负载。

### 语音服务新增 Canary，保留 CohereASR 运行 dtype

[vllm-project/vllm #64215](https://github.com/vllm-project/vllm/pull/64215)，2026-10-31T06:46:24Z，main `886b845ef14c25dd890b9750e5fb5d67e0ca96d0`；[vllm-project/vllm #54399](https://github.com/vllm-project/vllm/pull/54399)，2026-10-31T15:42:58Z，main `78161c28da3b17f95d27c16f7d329cc1f1987f27`。

Canary注册native Transformers checkpoint adapter，复用HF FastConformer encoder并用vLLM decoder，支持transcription/translation入口。CohereASR loader删除按checkpoint dtype重绑pos_bias，交正常loader转换到既有FP16 buffer。 语音部署核对checkpoint布局、实际语言/翻译参数、FP16 bias共享与WER回归。 **证据限制：**Canary作者token parity为有限短clip、earnings22 tiny切片WER；本轮未评估25种语言。Cohere修复本轮来源没有新GPU WER结果。

### loader 明确 tensor ownership 后省掉多余 clone

[vllm-project/vllm #63174](https://github.com/vllm-project/vllm/pull/63174)，2026-10-31T16:02:24Z，main `50f622bdcf17071cf2e715c47e4a0bf13cf6cf2e`。

runai stream_files若支持owned参数则显式owned=True并直接yield，否则保留clone；pooling prefix probe只对白名单owned formats跳过clone，fastsafetensors因允许的旧版本仍排除。 加载大权重时固定runai>=0.17.1并回归自定义iterator buffer复用语义。 **证据限制：**作者尚未测峰值CPU内存与load-time；这是省拷贝合同，不是已经量得的启动提速。

### RADIO/NaViT 编译让 packed token 维与变长 metadata 可寻址

[vllm-project/vllm #63415](https://github.com/vllm-project/vllm/pull/63415)，2026-10-31T09:11:53Z，main `10da92ff285612aa0c6947d1dd8c51fc2062771e`。

RadioLayer加compile decorator，动态维改hidden_states dim1及cu_seqlens dim0；metadata从dataclass展开tensor，single-image也构造合法[0,n]，仅编译路径pack uniform batches。 Nemotron Parse等视觉encoder编译需回归单图/多图/video及不同packed长度，确认首真实请求不会触发fatal EngineDeadError。 **证据限制：**本轮读编译入口和metadata路径，未跑GPU；不把修复描述成所有vision encoder已完整可编译。

### 长 tool stream 通过增量状态避免反复全量解析

[vllm-project/vllm #64675](https://github.com/vllm-project/vllm/pull/64675)，2026-10-31T01:51:14Z，main `c7a683bde5909bf2862d5d4500239dd1e3bd44bc`；[vllm-project/vllm #64673](https://github.com/vllm-project/vllm/pull/64673)，2026-10-31T01:51:09Z，main `0aa72a450c9a98642c698e36a26e26c750400157`；[vllm-project/vllm #64674](https://github.com/vllm-project/vllm/pull/64674)，2026-10-31T01:37:28Z，main `8777b4c86ce89ba21cb8376621ad8a7d7a9a1a59`；[vllm-project/vllm #61542](https://github.com/vllm-project/vllm/pull/61542)，2026-10-31T20:28:35Z，main `cfa9f96f5efbb45c5c02ab127b51eaf8b86bb4ca`。

Qwen3缓存已closed参数并从末尾继续；ParserEngine在重复key破坏prefix时装非streaming extender，跳过后续无效转换；Responses custom工具只解新字符串尾并保留完整escape/surrogate，不能满足编码前缀即回退；HYV4参数名不再strip以匹配精确schema。 长参数、重复key、Unicode分块和带空格schema key应一起回归event deltas与health。 **证据限制：**作者小模型/H200实服务减CPU证据是对应负载；Qwen机制不覆盖独立MiMo converter，Responses优化仅non-Harmony，某重复值情形delta分组时机有意改变。

### gRPC 退出共用 shutdown-timeout 预算并等待 RPC 之外的 KV lease

[vllm-project/vllm #62967](https://github.com/vllm-project/vllm/pull/62967)，2026-10-31T05:41:55Z，main `5a233286c6318c6fa579d3aac02860f64fa28742`。

从dispatch时计入非subscription RPC，覆盖仍收包/预处理阶段；SIGTERM先拒新RPC，待两次idle或deadline后才shutdown engine，把剩余预算加5秒teardown交给engine drain；第二信号强制cancel。 P/D滚动重启应分别测试长stream、慢媒体预处理、RPC已回但KV未读取以及KV-event subscription。 **证据限制：**默认0保持原行为，正值RPC grace至少5秒；这是有界drain，不能等同无限保证，作者验证是双B300 NIXL和指定模型。

### 补录：KV offload 控制 socket 关闭顺序与 lazy walk 预算

[vllm-project/vllm #63207](https://github.com/vllm-project/vllm/pull/63207)，2026-10-28T10:27:53Z，main `f4fd1ab68e2ba3a76bf6b903662eb6a8ee8202ab`；[vllm-project/vllm #64229](https://github.com/vllm-project/vllm/pull/64229)，2026-10-28T09:22:51Z，main `77f58e530450416d217e4816727a35910bf3d9bf`。

P2P先disable DEALER monitor再关reader，避免libzmq I/O线程阻塞导致全context失联，保留500ms linger。SimpleCPUOffload把不prefix-cacheable的bounded-replay SWA组也计入下一step free-queue需求，防缓存块被复用前尚未offload。 P2P做断peer/connecting-close回归；lazy offload同时检查free-queue顺序的#59383依赖。 **证据限制：**两项10-28旧候选补读；lazy证据是scheduler simulation，无E2E。它们不关闭仍open的#61667跨consumer/producer生命周期缺口。

### 补录：输入预算与拒绝请求的正则开销

[vllm-project/vllm #57411](https://github.com/vllm-project/vllm/pull/57411)，2026-10-28T06:10:10Z，main `26302eb938966fbe2d17c2ad9cbbd75bbdc7bf9a`；[vllm-project/vllm #52120](https://github.com/vllm-project/vllm/pull/52120)，2026-10-28T05:46:43Z，main `665bfc8aaeb7eca008ca4102cbba68efdc4efabb`。

FunAudioChat把dummy profile与runtime共用sample/frame预算，在feature extractor前拒超长waveform，默认约300秒；错误消息sanitizer增加无dot快速路径与atomic slash group，针对长slash输入的回溯开销。 对外服务回归边界音频长度、正常路径脱敏与长异常输入，检查实际发布包是否带修复。 **证据限制：**10-28主线补录；FunAudioChat正文还描述短音频padding优化，但最终diff增量主要是duration guard，本轮不将其重复算作新padding提速。

### P/D 目的页发布要等旧 forward，staging 保留已散布标记

[sgl-project/sglang #46373](https://github.com/sgl-project/sglang/pull/46373)，2026-10-31T23:13:13Z，main `fc32cbff6983c7573f4b60eab8b44815668130a0`；[sgl-project/sglang #47652](https://github.com/sgl-project/sglang/pull/47652)，2026-10-31T17:55:57Z，main `e624174f0734ef605fad45edffa4603e4ae62d80`；[sgl-project/sglang #46898](https://github.com/sgl-project/sglang/pull/46898)，2026-10-31T05:34:12Z，main `566dbd2cb31841a50c6995d618848d37f0f1c814`。

decode复用刚完成请求的device pages后，prefill可能先写、旧forward又覆盖。inline metadata现在由forward-stream event和所有rank MIN readiness共同放行；未发布失败请求不等不存在的drain ack。staging另用CHUNK_SCATTERED标记挡跨socket迟到STAGING_REQ；device admission首拒绝后只跳后续host-staged分配，仍处理失败与device commits。 回归overlap、取消、异构TP及staging ring reuse；注意首拒绝后小请求不再在同pass越过大请求。 **证据限制：**作者本地Qwen3-8B/GB300为Mooncake TCP，未验证RDMA/NVLink；PP当前无overlap。新增事件证明旧forward完成，不能用metadata发送当DMA完成。

### P/D 传首 token 的 customized_info，并固定注册 wire 帧

[sgl-project/sglang #40107](https://github.com/sgl-project/sglang/pull/40107)，2026-10-31T09:32:47Z，main `f78d906f3ed0c3802068cc4ee283eb8f0c4b27ab`；[sgl-project/sglang #47638](https://github.com/sgl-project/sglang/pull/47638)，2026-10-31T15:37:49Z，main `5eb70183472d56326920e70a4dde5cfbabe6f376`。

prefill采样出的第一token metadata现经有界MessagePack buffer送decode，tensor还原CPU，rebootstrap不重复append；默认capacity0，显式启用需两端同正容量。后续修复把aux_item_lens始终置于aux_ptrs旁，关闭时发空帧，消除后续帧的feature-dependent位移。 自定义sampler应核对首token、序列化失败、超capacity的request-local abort；P/D两端统一版本和容量。 **证据限制：**47638改变Mooncake/NIXL注册布局，混版本不支持；40107来源的GPU E2E仍待CI，不把CPU fixture通过写作GPU传输已验收。

### Mamba checkpoint 可只驻 host，ReplaySSM ring 按活跃 request 计费

[sgl-project/sglang #44718](https://github.com/sgl-project/sglang/pull/44718)，2026-10-31T06:20:24Z，main `2f6e7f347526034a173424cad0e7a72a5fb61ad2`。

SGLANG_OPT_MAMBA_HOST_ONLY_CHECKPOINT默认关闭；开启后finish先把state D2H并等待完成才发布host index/释放source slot，prefix checkpoint不再捐GPU slot，load-back写request slot。KDA ReplaySSM ring用request index，不再按所有mamba slots大小。 有HiCache的Kimi-K3可读其节省常驻state方案；同时调低max-mamba-cache-size才能兑现pool显存，并回归并发anchor load-back与host不足。 **证据限制：**要求hierarchical-cache且拒int8 checkpoint，request-ring排除仍按slot写的CuTe DSpark验证核。作者MI355指定配置conv+ssm13.00→7.81GB、ring9.14→1.84GB，不是自动提高吞吐。

### HiCache 的 sidecar 跟随全部 source transfer，MXFP8 L3 连 payload 与 scales

[sgl-project/sglang #43110](https://github.com/sgl-project/sglang/pull/43110)，2026-10-31T07:04:11Z，main `1f834c196414db0dd201f0a1ecc9fd143aaaf119`；[sgl-project/sglang #46936](https://github.com/sgl-project/sglang/pull/46936)，2026-10-31T15:25:20Z，main `9780c521554153b70bd62dbaee00a93dcfec83c7`。

derived sidecar从只取第一source改为按顺序merge每个独立transfer，保留重复host index/显式device index，字节计费相加，单source保留alias。MXFP8 host page在旧K/V key下分别用payload+UE8M0 scales多buffer对象，注册也覆盖scale缓冲。 缓存作者重点测split D2H尾页、一个host state恢复到两个device slot以及L3往返scales完整性。 **证据限制：**flat MXFP8 storage page仍不支持；MXFP8 E2E作者来自带变更的fork，原Mooncake storage套件未重跑；本轮无存储服务实测。

### 统一 Full/SWA 共享 LRU，SWA tail 分配去掉 host sync

[sgl-project/sglang #46827](https://github.com/sgl-project/sglang/pull/46827)，2026-10-31T17:13:58Z，main `a65c9c5969055788fce6b99ba78e3eb133e7e6d0`；[sgl-project/sglang #47542](https://github.com/sgl-project/sglang/pull/47542)，2026-10-31T04:22:37Z，main `573649879bfe2a7b547b25658707c2c4a0d9db62`。

opt-in joint_lru把Full leaf与SWA entry放同一访问时钟/heap，按共享bytes满足需求，SWA victim不连带折叠Full；只在能compaction且无per-component target时联合evict。tail新页根据extend顺序取new_pages最后num_swa_pages，代替device unique+isin。 Full/SWA unified-memory负载可比较命中率、reclaim次序及admission CPU等待；先覆盖split/cascade/restore recency。 **证据限制：**joint_lru限两端ChainAllocator、无state slots、Python tree core，session cache/rust core组合拒绝；无benchmark证明joint policy收益。tail优化只消同步，不改绑定集合。

### HiCache host budget 按保护字节计信用，并尝试 mlock 注册内存

[sgl-project/sglang #47557](https://github.com/sgl-project/sglang/pull/47557)，2026-10-31T20:41:59Z，main `181380682b50f4606d1a09df3cbbbfc21de931e3`；[sgl-project/sglang #47543](https://github.com/sgl-project/sglang/pull/47543)，2026-10-31T04:22:25Z，main `9dbc462c652cda456a5a5e1468f9a43458e3915f`。

cgroup headroom不再因任一小子group的memory.min撤掉全部page-cache credit，而扣direct children的min(protection,usage)，读值不明仍保守撤信用。cudaHostRegister后mlock、unregister全完成后munlock；soft memlock限可升到hard，失败仅告警、仍保留注册。 容器部署检查实际VmLck、memlock限制和host pool sizing；ATS平台再测aging后copy延迟。 **证据限制：**mlock有启动成本且使内存unevictable，torch-pinned/NPU等路径不在此变更；没有GPU/ATS吞吐测量，不把内核行为解释写成已观测提速。

### DeepSeek MegaMoE 对称 buffer 先于 KV sizing 分配

[sgl-project/sglang #47577](https://github.com/sgl-project/sglang/pull/47577)，2026-10-31T06:59:47Z，main `6e4d53fbc445fe9678062a5c95ad5e292efacb4f`。

buffer此前首forward才落地，目标与draft的大token cap会吃掉activation/graph slack而OOM。initialize现在对ready DeepseekV2MoE提前从同一个key取得buffer，执行复用它，让KV预算事先扣除常驻字节。 高prefill cap+DSpark团队可核对启动日志的target/draft buffer与KV容量，不再仅降低static fraction兜底。 **证据限制：**作者GB300 DSV4.1 cap65536示例22.31+11.78GB/卡；Qwen/MiniMax/Kimi和AMD FlyDSL仍可首forward分配，不能说所有MegaMoE都完成预算闭合。

### Rust/Python frontend 收紧 abort、DP relay 与 SIGTERM 生命周期

[sgl-project/sglang #47321](https://github.com/sgl-project/sglang/pull/47321)，2026-10-31T09:14:48Z，main `ba78bce729d8b2ba05c4b3ae95c32b9bf940a859`；[sgl-project/sglang #47489](https://github.com/sgl-project/sglang/pull/47489)，2026-10-31T05:41:43Z，代码经 main #47321 集成 `d99d572904bc69568005049beb8574186909942f`；[sgl-project/sglang #47532](https://github.com/sgl-project/sglang/pull/47532)，2026-10-31T17:13:28Z，main `60b183d5c3cebd56033cd00b74310bc2f040c800`；[sgl-project/sglang #47691](https://github.com/sgl-project/sglang/pull/47691)，2026-11-01T00:06:35Z，main `cda284e47b51bfa71a3414ed6a0ea3425b409cb3`；[sgl-project/sglang #47546](https://github.com/sgl-project/sglang/pull/47546)，2026-10-31T16:44:53Z，main `005b8e033f38846cb30b7ccc0503e2ecb92900df`。

main #47321保留HTTP abort的partial-output/abort finish语义，scheduler接受后等待其terminal output；同一最终diff已携带SIGTERM drain实现。#47532让DP0 base listener向其他rank best-effort relay abort并发布listener map，#47691启动后集体验证每attention-DP rank恰一listener和地址族；Python在最后await后筛掉已abort未dispatch请求并释放MM transport。 服务网关回归早期取消、media preprocessing、全DP abort和滚动重启；分别观察客户端返回、scheduler终止与资源释放。 **证据限制：**47489自身合在专项支线，但冻结main的shutdown.py/drain.rs已确认其代码由47321带入。drain继续接新请求、无自身deadline，main还写attention-DP shutdown广播TODO；relay5秒/64并发且返回不等peer，不能承诺全拓扑原子取消。

### TTFT response metadata 给出条件化 phase breakdown

[sgl-project/sglang #47159](https://github.com/sgl-project/sglang/pull/47159)，2026-10-31T16:45:00Z，main `571ddad70fae4cf7a8556e42bd14019c92af7aaf`。

由已有scheduler timestamps拆storage_prefetch_wait、other_queue与包含chunk间隔的prefill_latency；before_queue/after_prefill只在零retraction且时钟与顺序有效时返回，无效phase省略。 排查首token慢时用这些字段分离queue/prefetch/prefill，保留原始timestamp做交叉检查。 **证据限制：**after_prefill是elapsed remainder，不能当KV-transfer计时；有retraction时缺字段是合同而非零耗时。无本次服务性能结果。

### 小 forward metadata 可由 kernel 参数填入，绕开 copy engine 队头阻塞

[sgl-project/sglang #47550](https://github.com/sgl-project/sglang/pull/47550)，2026-10-31T16:43:02Z，main `9767ce3a5a5bbea1ee912e815ce41a1e0695a739`。

默认关闭的SGLANG_ENABLE_MLP_SYNC_METADATA_FILL在CUDA用Triton写DP token counts、extend lengths和有效token数，代替在大HiCache DMA后排队的pinned H2D。整数先编码16v+3避免Triton按1/16整除分化cache，长list按256和power-of-two padding限编译形状。 direct-IO HiCache与persistent kernel并用时可试验开关，量metadata enqueue到forward开始的等待。 **证据限制：**来源只做单GPU整数/编译cache验证，serving开关测试未完成；会用compute资源替代copy，不报未经量测E2E收益。

### Kimi-K3 residual bank shard carry 可用普通 NCCL

[sgl-project/sglang #47452](https://github.com/sgl-project/sglang/pull/47452)，2026-10-31T22:59:39Z，main `39082103118a35dce2783facaddf6986c40a495b`。

去掉bank-slice eligibility必须custom collective的条件，保留SP-MoE、residual-bank、PP1及连续MoE等要求；没有custom gather时不先做fusion preparation，按普通ReduceScatter/AllGather处理边界。 NCCL部署可试现有SGLANG_K3_SP_ATTN_RES，核对最后bank read与draft hidden-state capture排除。 **证据限制：**减少读写的bank rows，不缩physical bank allocation；作者8×B300、65K prefill c1 eager约4%收益，仅对应配方。

### decode indexer 分摊长行，MiniMax radix ties 固定取低 block id

[sgl-project/sglang #47387](https://github.com/sgl-project/sglang/pull/47387)，2026-10-31T04:40:51Z，main `2ccbdab65fb1fb4dab302f823b78cc575723d19b`；[sgl-project/sglang #45174](https://github.com/sgl-project/sglang/pull/45174)，2026-10-31T01:53:07Z，main `006ad251695d524705a52272c738d3de9de307f7`。

DSV4.1 FP4 paged indexer用固定num_sms×8网格均分全部rows的tiles，保持BF16舍入和head sum，长context不再只压少量CTA；默认开关路径限≤4096 rows。MiniMax CUDA radix final ties用ballot/prefix rank选最低id，与小路径/ROCm语义一致。 混合短长context和重复文本负载分别测straggler与多次冷cache greedy复现。 **证据限制：**作者8×H200 DP+DSpark混长负载decode+16.5%、单900K尾请求+49%，不是均匀短请求收益；tie修复不是所有浮点/采样路径确定性保证。

### Qwen3 decode 融合保留舍入，SM100 采样开放 Cake softmax

[sgl-project/sglang #46908](https://github.com/sgl-project/sglang/pull/46908)，2026-10-31T13:27:15Z，main `8368f86fd7a71b07e2b7dda5d4146695ec54cbfa`；[sgl-project/sglang #44168](https://github.com/sgl-project/sglang/pull/44168)，2026-10-31T12:26:00Z，main `c0f710f7dccc5d6c7cf05f0cb0109b76e8b5b60d`。

Qwen3 native CUDA在现有flag下复用table QK-norm+RoPE，match_cuda重现FP32归约、BF16中间舍入与FMA顺序，只走支持的BF16/NeoX/head shape decode。Cake softmax gate从SM103扩到SM100，dtype/shape/非确定性条件保持原合同。 分别用CUDA graph replay、exceptional values和稀疏grammar mask验证，而非只看argmax。 **证据限制：**Qwen prefill/verify、deterministic/on-policy与不支持shape仍fallback；Cake最新SM100数值验证有待CI的边界，不能写所有sampling都逐位不变。

### AMD MoE wire format 服从消费者，Quark W4A16 修非本地专家

[sgl-project/sglang #47408](https://github.com/sgl-project/sglang/pull/47408)，2026-10-31T03:53:57Z，main `fb3396260545bad839a393102f82816e15a2edca`；[sgl-project/sglang #37367](https://github.com/sgl-project/sglang/pull/37367)，2026-10-31T08:17:48Z，main `ef4f3f1151f16a5cc67ed1d6601a4b8292810e01`。

MoRI以AITER resolve_activation_dtype、quant type与scale布局决定BF16/FP8/MXFP8/FP4，缺API、batch-dependent或缺收发能力回BF16，不再按weight dtype猜。Quark W4A16将EP -1 id映射额外bucket、忽略其ragged rows并清零输出，路径收窄gfx1151且拒未实现SwiGLU变体。 AMD集成固定AITER接口/版本，读实际dispatch日志；EP>1回归全非本地、mixed本地与activation variants。 **证据限制：**AITER内部选择是依赖边界，本轮未复审其实现；保守BF16会增加接收端量化开销。gfx942回AITER路径，非泛化新W4A16支持。

### Ascend GLM 视觉与 DSA offload 修真实布局/预算

[sgl-project/sglang #44442](https://github.com/sgl-project/sglang/pull/44442)，2026-10-31T03:27:18Z，main `9d33af853b21ad7080e9e9abac78315ca52a30b5`；[sgl-project/sglang #45302](https://github.com/sgl-project/sglang/pull/45302)，2026-10-31T02:21:47Z，main `d2cf2371b0361db4b4f37d163bc9b4ecfa7a2d0d`。

GLM4.1V支持uncompressed ModelSlim W8A8S的dense INT8/encoded deq_scale，视觉forward一次准备CPU lengths给各ViT block复用，下采样把NHWC自然布局flatten后F.linear。DSA sparse offload estimator按本PP范围真正indexer owner层数计费，保留draft/dense fallback。 NPU部署核对W8A8S需要的FP16输出、vision metadata及owner-layer budgeting，复跑真实多图并发。 **证据限制：**DSA作者config-only GLM5.2 index budget容量26880→99840 tokens，固定16384 tokens张量footprint不变；不是HBM减少3.71×或服务吞吐已涨3.71×。

### MiMo 把 value scale 折入浮点 o_proj，NPU dispatch 保留 int32 ids

[sgl-project/sglang #40890](https://github.com/sgl-project/sglang/pull/40890)，2026-10-31T01:43:48Z，main `c7290006d1db3188ee524e1f31da9234a426da9a`。

MiMo load后把attention_value_scale乘到FP16/BF16/FP32 o_proj并标记folded，runtime与fused-KV-write不再缩V，cache改存未缩V；量化o_proj保留原multiply。NPU非ZBAL DeepEP保持原生int32 topk ids，避免先int64再int32往返。 回归reload、KV producer/consumer一致、MTP/DFlash独立loader和低精度误差；重点看本地实际适用路径。 **证据限制：**线性代数等价不保证低精度逐位相同，源码MiMo folding也未限定NPU；本轮不把标题扩写为所有量化checkpoint或draft已覆盖。

### DFLASH 小 M BF16 投影进入既有 tuned GEMM 分派

[sgl-project/sglang #45876](https://github.com/sgl-project/sglang/pull/45876)，2026-10-31T05:27:14Z，main `b20fdd2eb3a0e1176b3f5a5d4395be35c22f2362`。

kernel_projection从nn.Linear改ReplicatedLinear，candidate head在CUDA BF16且cutedsl/gemv backend时用bf16_gemm_dispatch，补GB300 TP4/8、M≤32计划；不满足条件留torch.matmul。 低并发draft可按实际M/N/K核对计划命中，再量acceptance和ITL。 **证据限制：**形状表也可能服务同shape非draft层；本轮未复跑整体性能，不把每格microbenchmark优势当所有DFLASH模型收益。

### MiniMax-H3 diffusion 优化把通信、norm 与 VAE 分别闭合

[sgl-project/sglang #47565](https://github.com/sgl-project/sglang/pull/47565)，2026-10-31T12:09:10Z，main `67bd749ada5bebf3f1ed9be16a814a3243f348bd`；[sgl-project/sglang #47567](https://github.com/sgl-project/sglang/pull/47567)，2026-10-31T12:27:16Z，main `e93f817325b7cc482421c3488c6e985b328baaff`；[sgl-project/sglang #47569](https://github.com/sgl-project/sglang/pull/47569)，2026-10-31T14:12:30Z，main `7585ccc2c060f09c07e2652d8608360a6c7bb47a`；[sgl-project/sglang #47570](https://github.com/sgl-project/sglang/pull/47570)，2026-10-31T12:05:17Z，main `6e011a7097530703f386b54c7ed1fdc4aa89814a`。

CUDA head-group门槛从4降3 waves；ROCm新增两组Ulysses，先排两组input A2A、在下一组attention时交换前组output。ROCm128-wide BF16 QK norm直接读strided QKV，第一见BitExactFusionGate通过才融合norm/RoPE；gfx950 VAE replay math-SDPA运算链，省无mask场景无用safe-softmax/downcast。 分别读每条gate、head独立性与fallback，在同分辨率/质量/并行配置复验输出与时延。 **证据限制：**作者4×MI355X H3 720p报告各分项收益，但有组合实验；不相加为独立总加速。CUDA3-wave结论不替代RCCL两组选择，非所有attention backend都可拆head保持bits。

### Blackwell diffusion 的 attention 选择同时看模型、执行模式与质量

[sgl-project/sglang #47586](https://github.com/sgl-project/sglang/pull/47586)，2026-11-01T00:41:48Z，main `8817f63c2cf9acb0fe8c8750863b17bbb1c6ab1b`；[sgl-project/sglang #47301](https://github.com/sgl-project/sglang/pull/47301)，2026-10-31T12:29:07Z，main `504f9cf720c835beb96563ec5186cb4e6802cc74`。

GQA在SM10x且Q-head整除KV-head时保留cuDNN，MOSS主DiT移除硬钉FA默认；sub-tile attention在eager lossless/high走cuDNN，BCG warmup/capture保FA4，exact保参考路径。 部署以实际eager/BCG breaks及请求quality核对backend，而非单凭GPU kernel时间选全局默认。 **证据限制：**cuDNN/FA4舍入不同；作者B200 Cosmos3端到端-13.1%是指定少量请求，LTX最终E2E还合并其他布局/阶段修复，不将组合结果归本PR。

### Diffusion 服务 warmup、取消与 KV admission 边界补齐

[sgl-project/sglang #47571](https://github.com/sgl-project/sglang/pull/47571)，2026-10-31T12:04:35Z，main `5c46d20db9507804a9fb6954840fcd72e52bc70d`；[sgl-project/sglang #47240](https://github.com/sgl-project/sglang/pull/47240)，2026-10-31T01:55:05Z，main `7bc92100ad741206b68b9065d941e4c16bfaac9c`；[sgl-project/sglang #47516](https://github.com/sgl-project/sglang/pull/47516)，2026-10-31T08:24:57Z，main `a4a6eddd25343905245db7a8b54bcf24d9483c8b`。

server warmup按replica显式固定，完整列表并发覆盖所有DP副本且不消耗真实round-robin；等待所有warming收敛后再抛错。取消检查从progress bar移到全rank stage与采样边界、forward后保存前；Nomas把reserved-allocated视作可用，并提前拒video token数不被SP整除的segment，flush除外。 多副本视频服务回归首请求、低频progress、group cancellation、输出归属与缓存后第二次请求。 **证据限制：**取消只在检查点生效，不是中断in-flight kernel；Nomas只有CPU测试/无SP多进程E2E。warmup的冷路径音频差异不能解释为DP数值bug。

### Pika-H3-Flex 接入的是 LoRA 加固定六步 re-noising 合同

[sgl-project/sglang #47378](https://github.com/sgl-project/sglang/pull/47378)，2026-10-31T12:24:25Z，main `37fc17b8d60aa8a380b63fae0401c5d343ae604a`。

注册overlay固定base/adapter revision并校验，选择FL2VA或Ref2VA；每output独立device RNG，按release knots预测x0再以新noise到下一sigma，终点不抽noise。metadata显式stochastic_sampling与lora_adapter，非仅把普通H3的num_inference_steps设6。 使用Pika配方应固定overlay/sampling contract，检查adapter partition、seed与输出扩展。 **证据限制：**这不是新base模型；plain --lora-path的uniform Euler不等价，re-noising显式拒rollout组合，本轮未下载或运行视频模型。

### Cosmos3 conditioning cache 用 GPU BLAKE3，只回传 digest

[sgl-project/sglang #46028](https://github.com/sgl-project/sglang/pull/46028)，2026-10-31T09:26:17Z，main `03a27e2a17567d04a4309a83df8771d012bc0950`。

Triton实现unkeyed BLAKE3-256分chunk/tree hash，Cosmos3把digest连tensor dtype/device/shape/stride和encoder settings作key，复用既有cache invalidation；避免整tensor D2H后CPU SHA256。 有重用conditioning的单GPU路径可量fingerprint等待，保留metadata/settings避免错误cache hit。 **证据限制：**当前集成限CUDA且world size1的cache路径；不把通用kernel当所有视频模型或多GPU已接入，也不以自定义64-bit hash的速度替代正确性证据。

### Qwen-Image-2.1 半 warp norm/RoPE 与 text encoder graph 缓存

[sgl-project/sglang #47465](https://github.com/sgl-project/sglang/pull/47465)，2026-10-31T01:52:59Z，main `96a76f5d5fe4b7fa0f9809fbbf6f6101dae69dae`；[sgl-project/sglang #47467](https://github.com/sgl-project/sglang/pull/47467)，2026-10-31T01:52:29Z，main `ab86251288cf475d5c4976e1d42b29456f8b6f50`。

128-wide行由半warp、16-byte访问处理，仍按原四元素分组/树次序归约并检查工作行数可用32bit。text-only encoder按prompt length第二次请求capture，最多8长度，参数/buffer地址变化则丢图，失败永久fallback；上传先pin后nonblocking。 resident单GPU T2I可回归奇数head、edit输入、reload/offload地址变化及改变输入的replay。 **证据限制：**image-conditioned/TP encoder不capture；作者B300编码阶段约41→18–20ms，小幅E2E收益与另外PR有组合，不能宣称整体快一倍。

### dummy MXFP8 的 UE8M0 scale 改为中性值

[sgl-project/sglang #47337](https://github.com/sgl-project/sglang/pull/47337)，2026-10-31T04:23:23Z，main `800de62d7ceec6a619bbea2422f9327ae249a10d`。

dummy loader原来只初始化float，uint8 weight_scale_inv留0意味着2^-127，使MXFP8权重近零并可能向下游传播NaN。现在这类scale填127即2^0。 做profiling、warmup和容量估算时检查dummy数值有效，真实checkpoint仍用自己的scale。 **证据限制：**只改变--load-format dummy，不是修复真实模型权重或提高模型精度。

覆盖说明：六仓 releases、72 小时回溯 merged-PR 搜索与默认 main committer-time 提交清单已分页；selected PR 已取完整 changed-file 页，但只对正文涉及的关键机制做语义读审。无正文条目不等于仓库没有实质更新。verl 主窗两项为文档/CI；slime 在本次所查 merged/main/release 面无主窗增量。旧 open vLLM #61667 与 issue #61532 仍 open、head 未变，不把相关其他修复当作其已交付。旧队列、分支和未审候选均留 ledger；不推进全局完整覆盖游标。

<a id="extended"></a>
## 扩展开源工程与硬件软件协同

以下日期为原始合入日期，保留 UTC；代码基线逐项列出。本节测试和性能均为贡献者报告，本轮只核对 PR、最终 diff 与可定位工件，未运行设备实验。合入开发分支不等于正式 release。

### Transformer Engine：BF16 grouped MLP 也可融合，但性能证据只是一个 MoE 层

**NVIDIA；10-31，main。** [#3646 / `63451d6b`](https://github.com/NVIDIA/TransformerEngine/pull/3646)新增 `GroupedLinear → ScaledSwiGLU/ScaledClampedQGeGLU → GroupedLinear` 的非量化 BF16 融合，覆盖 forward、dGLU/dgrad、wgrad；此前该可选路径主要服务 MXFP8/NVFP4。最终 `GroupedMLP_CuTeGEMMGLUBf16.is_supported` 限 Blackwell（SM10.x、minor<7），要求 cuDNN Frontend≥1.32.0 **且四个 wrapper 实际声明 BF16 canonical-layout 能力**，只看版本号不够。每组及总行数需256对齐；GeGLU常数受限，需要原子scale梯度的非空训练路径拒绝确定性请求（空batch等例外以guard为准）。

作者在**单 B300、BF16 eager** 上模拟 Qwen3.5-397B-A17B 的 EP32 单卡形状：16个本地专家、hidden4096、FFN1024、每专家2560行。CUDA-event测量，7轮×50次、3次交错配置；per-expert权重的模块训练步2.556→2.400ms，packed权重2.557→2.393ms，约1.065–1.068×。这里“train step”是该 grouped MLP 的完整前后向，**不是整模型端到端训练提速**。建议复现自身 token/expert 分布、padding和确定性要求，再决定是否打开融合。

两项配套增量也值得一起检查：[布局优化 #3641 / `70ee2bbb`](https://github.com/NVIDIA/TransformerEngine/pull/3641)按 wrapper 能力将 MXFP8 的天然二维/packed operands 直接传给 cuDNN，减少每次调用构造3D/6D view；保留旧wrapper回退。[验证修复 #3785 / `9ad8fd04`](https://github.com/NVIDIA/TransformerEngine/pull/3785)把一般 GroupedLinear 默认split对齐按recipe调整为BF16/FP16等1、Hopper current FP8为16、block-scaled为128，并始终拒绝负split、检查16字节leading dimension。**一般GroupedLinear可用小split不等于BF16 fused MLP免除256约束。** 验证修复的测试checkpoint `565bb89a`在H100 NVL/GB200报告整套fusible-op回归通过，API当前head与之不同，未据此宣称final-head全套重验；这也不能替代上述新融合的自身验证。

### TensorRT-LLM：MiniMax-M3 的 GB300 TP4/TP8 专用 decode 路径合入

**NVIDIA；10-31，main。** [#20306 / `f405e10b`](https://github.com/NVIDIA/TensorRT-LLM/pull/20306)为 MiniMax-M3 NVFP4 新增 `modeling_v2` 的 `sm_103__tp4`、`sm_103__tp8` target，针对小decode batch的碎kernel、graph replay之间host工作和通信边界。PR共400个文件，本轮已取齐文件页，重点读路由、runtime-hook协议、dispatch与测试入口；没有逐个证明全部生成kernel。

公开实现把每engine的输入staging/未变化metadata上传省略、私有graph metadata pool、采样与投机step-fold接到模型声明的 `RuntimeHooks`，默认无hook的模型维持原路径。这些target仅服务text、拒绝image input，要求对应modelopt混精度打包、FP8/NVFP4主KV与指定MSA runtime；相同architecture/shape但不同checkpoint量化打包也可能被拒绝。TP4的纯generation≤48 token走融合decode，prefill、mixed及更大step走一般layer；TP8另有1–8 token的router+experts融合。`modeling_v2: require`下，受支持形状若缺前提会报错，不应把隐式回退当成功。TP4还为两个BF16 head提供可逐位验证的12-bit编码，验证失败回BF16；TP8无该coded副本。长上下文block selector对相等score的破平局规则与原atomic到达顺序不同，不能据“精度不变”推出所有token完全一致。

作者报告TP4单测1156项、runtime-hook548项、TP8单测1019项，另有2项需跨两tray八rank的通信测试跳过，仅以2/4rank替测。GSM8K greedy/sampled对照和自然接受长度未见显著劣化；PR声称约1.55×/1.84× interactivity，但该摘要没有足够完整的输入输出长度、并发及基线配置，**本日报不将其当可直接比较的端到端速度承诺**。建议深入阅读、在GB300对应拓扑考虑复现；其他模型、SM或batch不可继承。

### FlashInfer：MSA index context parallel 与 GDN block 有新实现，收益边界不同

**FlashInfer；10-31，main，experimental。** [MSA ICP #8122 / `35ea8e25`](https://github.com/flashinfer-ai/flashinfer/pull/8122)交付 MiniMax-M3 的分片index score、Top16候选和合并路径，接口为 `flashinfer.msa_ops`/`cake_msa_icp_indexer`。只支持FP8 Q/K、head dim128、TP2/TP4，以及P128跨rank交错或P256连续分片；SM100/103/107有工件，不能扩成任意上下文并行。

作者整模型serving实验固定 MiniMax-M3-NVFP4、EAGLE3 k=3、ISL163840/OSL8192、并发14和14+1，CUDA13.4、PyTorch2.14；基线是MSA#46@`f380d0e88e2e`经vLLM#63671@`a76900d6eee7`调用。GB300/VR200表中decode约1.01–1.09×，output throughput约1.01–1.08×；所称decode是rank0 GPU-time中位数，轮数GB300仅2轮、VR TP2仅1轮。没有B200端到端测量；整套比较依赖上述调用栈，**不是装一个已发布wheel就自动获得的收益**。建议按同一服务栈复现长上下文吞吐和尾延迟。

[GDN block #8131 / `56472bf4`](https://github.com/flashinfer-ai/flashinfer/pull/8131)把BA projection、conv状态更新、gating、gated-delta递推、gated RMSNorm与FP8量化包装成一次调用；conv和SSM池原地更新，verify保留逐token状态。`frost`后端当前限SM100、Qwen3.8-27B TP1的5120 hidden/48V/16K heads/128维/conv4几何，verify最多4请求，并有token数gate；先prepare、eager warmup后才graph capture，不满足条件返回typed decline。

作者B200 1000W、冷L2 graph replay的**每层**实验：T6单请求verify23.42→14.47µs（1.62×），4请求34.17→33.14µs，8/16请求不服务；单行decode7.80→8.72µs，反而慢。基线已经把可重叠的BA GEMM排除，不能拿串行“+BA”列夸大收益。FP8输出≥99.99%逐字节一致、SSM误差在作者给定2.5e-4范围，不是全程bit-exact。建议优先复现低并发MTP验证，普通decode和大batch继续观察。

### FlashInfer：训练host状态、graph workspace 与依赖接口修复

**10-31，main。** 以下关键修复影响正确性，不只影响microbenchmark：

- [LM-head多线程与binding #8126 / `785e84e4`](https://github.com/flashinfer-ai/flashinfer/pull/8126)：deterministic chunked loss的hidden-valid-row pinned cell/event、递增常量、dW side stream原为设备级共享，多线程可覆盖计数或重录event，导致错误输出乃至非法访问；改为线程内、设备内状态，并对常量使用stream记录保护寿命。logprob backward的lse/dlogp改按layout缓存binding，仍按调用重绑实际tensor；X因alignment前提仍按地址。作者在B200/GB300有强制交错回归，**kernel数学代码没有变化，deterministic命名不能替代host并发安全**。建议多线程训练服务优先回归。
- [空dX K-slice #8108 / `96ca3d01`](https://github.com/flashinfer-ai/flashinfer/pull/8108)：V256只有4个K步，切3片会成2/2/0，空片未初始化accumulator仍写入并参与求和。planner将3修到2、非法显式pin报错，并加launch guard。作者T1000/H6144/V256下B200/GB300的dX相对FP64误差由约0.68–0.70降至约0.0023–0.0026；loss/logp/dW并非同时出错。V≥512及未受影响plan保持原样。建议小词表/合成模型训练回归，不能泛称所有LLM梯度曾损坏。
- [mHC workspace #8100 / `fc6d2311`](https://github.com/flashinfer-ai/flashinfer/pull/8100)：capture中首次`torch.zeros`只录制清零而未执行，默认workspace可能被之后eager调用读到未清ticket。现在capture cache miss直接拒绝，要求同模式/桶warmup或调用者提供workspace。默认仍是同设备同token桶跨stream共享，**没有解决并发stream复用，需独立workspace**。作者B200 mHC相关550项通过；建议graph与并发测试同时做。
- [cuDNN单expert边界 #6853 / `ddf3e7bc`](https://github.com/flashinfer-ai/flashinfer/pull/6853)：cuDNN≥9.28的单expert要求两条显式边界，旧`m_indptr[:-1]`会BAD_PARAM。统一helper在该条件传`[start,cum_m]`，末段仍到实际buffer总行数；多expert及旧cuDNN维持原格式，device上构造、无host同步。建议升级cuDNN或EP每rank仅1个expert时回归，不能把旧版本CI通过当新接口兼容。

### FlashInfer 的两则调优纠偏：抢占下不挂死，比kernel少几微秒更重要

[MegaMoE #8095 / `18fb0bfc`](https://github.com/flashinfer-ai/flashinfer/pull/8095)，10-31合main，修复实验384-expert/top6/hidden5120 rank变体在TP4/TP8、EP1、T1024且GPU被其他进程抢占时的hang；并重生成rank programs。作者GB300/CUDA13.4、交错多进程graph replay测得新旧约0.996–1.001×，收益是稳定性，不是速度提升。建议共享GPU环境做抢占故障复现；其他variant不据此宣称全解决。

[撤回 #8075 / `5f1eae4d`](https://github.com/flashinfer-ai/flashinfer/pull/8075)恢复MiniMax-H3 SM120两种NVFP4 packed-varlen attention的head-major计划顺序。原#8040把尾部按长任务优先，省kernel时间却多60–160µs host建表；每次新建plan时，RTX PRO6000/RTX5090分别46/52、45/52个单元更快，而复用plan会交回原kernel-only约18µs收益（该数只测一个shape）。建议将新建/复用plan分开测。这是多模态attention局部反例，不推广到自回归LLM全栈。

### TensorRT-LLM：分离服务的取消与KV清空事件有更清楚的完成边界

**10-31，main。** [#20491 / `5ec25c55`](https://github.com/NVIDIA/TensorRT-LLM/pull/20491)在generation-first服务监视CTX/GEN两腿，失败或取消时取消并drain pending sibling；保留已经完成的GEN响应和worker错误，清理被取消的placement选择，SSE按增量解析识别完成/错误。已读`openai_disagg_service.py`、`openai_client.py`、`router.py`及配套测试。**HTTP取消不是GPU/DMA/KV物理静默的证明，不能据此回收远端内存**；placement在CTX dispatch之前的某些取消仍依赖expiry，完整配对重试未在此实现。PR的407项原生测试是旧head证据，最新SSE修订的原生执行曾受环境限制。建议复现取消竞态及资源退休，不当作完整pipeline全部通过。

[#21142 / `17bfe4e0`](https://github.com/NVIDIA/TensorRT-LLM/pull/21142)在native prefix-cache成功清空后按rank发布有序`AllBlocksCleared`，即使推理暂停/idle也在finally路径drain streaming事件；下游Dynamo才能及时撤销驻留记录。`streamingEventSink`、`KvCacheEventManager`和reset路径没有新增跨rank gather，buffered模式仍等iteration边界；后台ZMQ仍best effort，**入队不保证路由器收到**。建议对暂停、后续reset步骤失败、消息丢失分别验证。

[#20017 / `2d1a7d17`](https://github.com/NVIDIA/TensorRT-LLM/pull/20017)另修MoE `util_convert_type`返回局部stack fragment视图的寿命错误，改为owning tensor并约束layout连续性。贡献者只有host ASAN和nvcc编译验证，没有端到端GPU复现；同模式flashMLA与exported树未在此修。建议编译器迁移时优先查该补丁与未覆盖同类位置，不称整库已消除该问题。

### vLLM-Ascend：MLA外部KV与投机采样的静默错误修复

**10-31，main。** [#19639 / `1bebfa67`](https://github.com/vllm-project/vllm-ascend/pull/19639)修复短prefill被分为decode时跳过layerwise外部KV等待的问题。普通/strided preprocess在Q/KV投影后、cache访问前等待，保留通信重叠；PROLOG_V3在任何decode/prefill cache写之前仅消费一次completion，fused-only decode也补等待。decode还补`record_attention_compute_start`释放下一层prefetch，否则仅加wait可能在layer1死等。建议外部KV+短prefill/混合模型部署优先回归；HTTP200不是结果正确性验收。

[#19866 / `0f2eee9d`](https://github.com/vllm-project/vllm-ascend/pull/19866)适配vLLM0.32/main@`544f182d`的request-major sampling-mask：每个已接受token读`cu_num_logits[req]+slot`，输出`num_requests×rows_per_request`槽，inactive槽mask/count归零，offset/count在设备端随graph replay变化，保持compact token_ids零宽的精确bitmask回退。旧“一请求一行”无法表达一次投机提交多个token。作者报告36个NPU回归通过，建议验证拒绝采样/RNG、动态接受数和graph重放；不能仅检查接口可import。

### vLLM-Ascend：混合cache的新增支持，要区分warm重放与模式等价

**10-31，main。** [QSA池化 #20261 / `aa6f94ba`](https://github.com/vllm-project/vllm-ascend/pull/20261)使Qwen3.8 QSA/GDN混合cache进入AscendStore bulk DRAM/SSD池化，并协调QSA、PLE和Mamba状态的逐层load、延迟copy/save与MTP finalize。最终diff包括connector已加载layer记录、scheduler与model-state转发，CPU回归162通过/1跳过。作者A3 TP8、9.2 toolkit/9.3算子、旧#16981栈+相同wrapper修复曾测到9217-token prompt重放8192 external-hit tokens和SSD rewarm；**不是完整rebased packed-page最终head的NPU验证**。其中1TiB SSD是sparse-file/loop逻辑容量，GPQA新跑尚未有最终分数。建议深入读表示和状态寿命，考虑在固定最终head复现；不直接以旧栈硬件结果采用全部新路径。

[RecoverSSM #20369 / `95753074`](https://github.com/vllm-project/vllm-ascend/pull/20369)允许实验GLM KDA在Mamba `align`模式启用prefix caching，并在调用上游builder前准备物理state IDs，复用已有accepted-state恢复和block checkpoint。作者A3 TP8+EP、GLM-5.3-Flash W8A8、BF16 model/KV、MTP5、FULL_DECODE_ONLY的30题：不开cache、enabled冷cache和warm均28/30，30条cold/warm token完全匹配，warm每条至少命中1280token；但disabled/enabled冷token仅2/30相同。**证据支持相同cache模式冷/热重放，不支持跨cache模式逐token等价**。A5仅12项算子测试通过，端到端待验。建议按上述两个等价命题分别回归。

### vLLM-Ascend：AllGather的副本路由、物理block单位与prefill开销

**10-31，main。** [MRV2 EPLB #18549 / `712d2e5e`](https://github.com/vllm-project/vllm-ascend/pull/18549)：AllGather后每rank看相同token，原rank相关副本选择可能重复/漏算route；新AllGather路由表与rank无关，load只在物理owner计一次，排除padding/非采集phase，放置变化原位刷新graph表。`AscendEplbLayerState.record_load`统一统计；mask过滤负载统计，不改GMM token数。MC2/AlltoAll及MRV1保留原选择。建议冗余expert+AllGather部署回归放置迁移与draft padding，不外推所有通信路径。

[DeepSeek-V4 block #19125 / `0fe5b3bf`](https://github.com/vllm-project/vllm-ascend/pull/19125)阻止上游platform hook把已选物理block32/64/128改成压缩backend的逻辑token单位4096而在padding表KeyError。`update_block_size_for_backend`对该model提前返回，保留已确定配置。建议A5无需显式block-size的启动做回归。

[PCP metadata #19007 / `f9e063e8`](https://github.com/vllm-project/vllm-ascend/pull/19007)按compressor/SWA/attention/indexer消费者区分需求，跨cache组复用每step的restore indices、padding和RoPE准备；仅需cache update的全局PCP不再重复造query元数据。建议用实际多cache组trace验证CPU建metadata下降，不以静态重用推出整模型提速。

[PLE prefill #20344 / `6c2020db`](https://github.com/vllm-project/vllm-ascend/pull/20344)只把模型局部prefill binding切到Ascend Triton，decode/spec保留上游；四tap连续权重与通道加载、2048通道tile，单独writeback保留所有读取所需旧历史。作者950DT对应vLLM基线的4项最终NPU算子测试通过，不能当所有Qwen预填充速度已验证。[MiniMax #20290 / `e1312563`](https://github.com/vllm-project/vllm-ascend/pull/20290)为A5 CEIL-scale、`SWIGLUOAI_UNINTERLEAVE`融合clipped SwiGLU+MXFP8量化，并启用AscendC index-score decode；193个CPU用例不替代NPU精度和吞吐。两项都建议按模型/硬件条件复现。

### ROCm：低精度scale越界、attention在线softmax与DP调度

**10-31。** [AITER #7047 / main `0dd27160`](https://github.com/ROCm/aiter/pull/7047)修batched FP4 GEMM的ragged-K scale加载：数据虽masked为0，越界e8m0若恰为0xFF仍会让`dot_scaled`产生NaN；非整K路径给scale加mask、默认127（2⁰），整K保持原载入。已读kernel与故意在scale后放0xFF的回归。建议FP4不规则K形状优先回归，不能把数值被0遮住理解为scale也安全。

[rocKE #14542 / develop `809c17de`](https://github.com/ROCm/rocm-libraries/pull/14542)修online softmax把leading全mask tile的临时安全参考0错误地作为running max传后续tile：当真实score极负会下溢。最终代码传raw max，只在指数参考处替换；keyless guard用`l==0`，不再把NaN伪装为0。作者gfx942/gfx950分别65/69个新增失败转通过，D128/GQA32:8、8K的选定kernel性能约持平。gfx942的4-warp、decode及部分3D reduce未改，D256超LDS、多序列、gfx1250没有GPU测试。建议复现滑窗、空segment和NaN传播；这是develop新修复，不是ROCm release保证。

[ATOM #2727 / main `43eed2fd`](https://github.com/ROCm/ATOM/pull/2727)修DP prefill delayer把“capped pending总数不增长”误判为不会再有到达的问题。共享归约增加blocked/unreachable信息：能在剩余TTFT预算内结束decode的rank继续等对齐；KV-blocked rank参与dummy prefill且不虚增pending。预测基于max_tokens和可接受draft上界，EOS可早、拒绝可晚，TTFT hard bound仍需保留。PR中段的18.0%来自旧base/head，不能代表最后一轮。底部另报main `e893780f`对测试checkpoint `81a8ba75`，固定AITER `50873674`、DeepSeek-V4-Pro/TP8、FP8 KV/FP4 index、无prefix cache、随机8192/1024、c128：DPA 1455.30→2819.05 output tok/s，TTFT8733.99→4863.12ms；DSpark 3139.72→3751.36（19.5%），TTFT4709.91→6017.74ms。四腿均完成1280请求，但每配置只有一组完整配对。该测试checkpoint与本轮PR head `9bd1f821`不同，后者还修PP中只有local verify才按draft接受上界换算步数，不能称最终merge已按相同GPU实验重验。硬件具体型号未在该表明确；最新accuracy记录10/11通过，GLM-5.3-kpool仍失败，unchanged base亦失败，不能归因此PR也不能称全过。建议在最终固定commit复现吞吐/TTFT与正确性。

### Apple MLX 与 JAX：分片表示和极端尺寸的正确性

**10-31，main。** [MLX #4715 / `33927952`](https://github.com/ml-explore/mlx/pull/4715)把ShardedToAllLinear新建bias置零，避免各rank随机bias在sum后各加一次而输出不一致；量化层的整数split按原输入feature数变成精确fraction，使packed weight与scale同边界，无法按group/rank整分时报错。建议多机量化模型转换回归各rank一致性。

[MLX #4772 / `93e65c0b`](https://github.com/ml-explore/mlx/pull/4772)将Metal layer_norm forward/VJP中越过行尾thread的剩余元素数夹到0，避免负数fill循环越界/拖住SIMD。作者M2 Pro/FP16、每次2²⁶元素、小行16/64前向236.82→20.45ms、48.01→5.20ms；行256基本持平，**不是典型LLM大hidden层的通用倍数**。[#4776 / `f7f86bd7`](https://github.com/ml-explore/mlx/pull/4776)对cross_entropy forward/VJP按Metal grid上限分slice，行数改int64，避免超过UINT32_MAX线程时静默漏行；≤128 classes时2²⁷行是具体触发示例。建议长packed训练/小词表回归，大词表按实际threadgroup另算界限。

[JAX #42201 / `f3bad1a6`](https://github.com/jax-ml/jax/pull/42201)使Explicit mesh/shard_map中的unreduced/reduced数组与replicated常量组合按语义验证；原4设备pad常量1会被sum成4。最终pad/concat/select规则要求匹配轴，full_like与transpose零值保留归约语义，而非只加警告。[#42204 / `98285a14`](https://github.com/jax-ml/jax/pull/42204)为sort补VMA检查、拒绝unreduced输入，并修sort/gather梯度的标记。建议TP/自定义shard_map训练回归常量padding、排序和梯度；新报错可能揭出旧的静默错误，不能当普通兼容性噪声。

### Dynamo 与 Helion：可用的基础设施接口，与尚未交付的系统能力分开

**10-31，main。** [Dynamo #17331 / `2cb12a1a`](https://github.com/ai-dynamo/dynamo/pull/17331)新增Global Discovery Service**库**，定义pool/replica list-watch与heartbeat lease；memory注册和MCS/Karmada EndpointSlice发现已在代码，HTTP binary另在#17371。memory仅保留本实例收到的注册，需单副本；MCS连续错误一分钟保留旧视图但标incomplete，恢复需两分钟无错。消费者对partial list应merge而非删除缺席pool，对complete才可replace；PoolListing不含动态load/KV/readiness。DynamoDB持久化仍planned，不能把参数规划表写成已实现后端。建议深入读多集群发现与不完整快照契约，继续观察服务与路由器集成。

[Helion #4190 / `d4dab6b3`](https://github.com/pytorch/helion/pull/4190)新增`hl.prefetch(tensor,index,extra_mask)`，让编译器识别L2预取而非把inline asm当opaque内存操作。PTX上SM90+的连续行可降bulk prefetch，其他索引用逐元素hint；其他target/backend丢弃。它**不等待store、不建立可见性、不返回值**，只是调度/缓存提示，不能替代通信完成fence。建议自定义LLM megakernel尝试KV/权重预取，并单独计host与同步成本；目前没有本轮可比的整模型提速证据。

### PyTorch：默认分支已落地的动态slice与ROCm backend修复

**10-31，默认分支commit证据。** [dynamic slice `40783e4a`](https://github.com/pytorch/pytorch/commit/40783e4a68eb41c5f41f19c17441e60c39bb8ed7)关联#202378：避免容器/切片的Python常量检查强迫符号shape特化，`dynamic=True,fullgraph=True`的不同slice长度可复用图；[ROCm addmm `97d0fc02`](https://github.com/pytorch/pytorch/commit/97d0fc0212aebf2cc58a13c1ff0d05e45e0ceb54)关联#202594：不再把首次是否禁用Lt永久缓存，运行时切preferred BLAS后可重选。[AVX2 `43ae9ac8`](https://github.com/pytorch/pytorch/commit/43ae9ac8a8d4e3ee449d2a1e76e290234a0f7db7)关联#202480修Half/BFloat16 zero_mask的每元素两bit错误，先pack到byte再取16bit mask。

本轮PR接口这三项`merged_at=null`，不能写成“PR已merge”；它们确已出现在截至cutoff的默认分支commit枚举，按该commit报告。建议分别回归编译次数、backend切换和CPU半精度mask，未提供未经核实的吞吐倍数。

### 支线观察与release去重

[#21364](https://github.com/NVIDIA/TensorRT-LLM/pull/21364)只合TensorRT-LLM `feat/musespark-open-agentx`；[AITER #7253](https://github.com/ROCm/aiter/pull/7253)、[ATOM #2874](https://github.com/ROCm/ATOM/pull/2874)/[#2872](https://github.com/ROCm/ATOM/pull/2872)只合 `alizaidy/gfx1250-qwen-support`，分别涉及页ID边界、Qwen3.5 FP8 KV/LMCache支持。Ascend [#17200](https://github.com/vllm-project/vllm-ascend/pull/17200)为`releases/v0.23.0`修dummy load和ACL graph更新。均不当作新main交付，继续观察。

回溯补核 [DeepSpeed v0.19.9（10-29）](https://github.com/deepspeedai/DeepSpeed/releases/tag/v0.19.9)、[Transformers v5.22.0（10-29）](https://github.com/huggingface/transformers/releases/tag/v5.22.0)，分别聚合此前已逐PR记录的训练/数值修复与FP8 adapter backward等；本轮只补release落点，不重复报相同机制。[v5.22.1（10-30）](https://github.com/huggingface/transformers/releases/tag/v5.22.1)只移除WIP文档，不作技术新增。

<a id="coverage"></a>
## 覆盖、补录与待续查

### 实际检查范围

| 来源类别 | 本轮实际完成 | 仍未覆盖的部分 |
| --- | --- | --- |
| 六个核心开源项目 | Megatron-LM、vLLM、verl、SGLang、slime、torchtitan的release、merged搜索与默认分支提交分页；768个PR搜索对象、747个默认分支commit；109个PR取元数据/changed-file页，正文按56个机制组组织 | 清单取得不等于每项语义审查；217个当前候选与277个继承未闭合项可相交，不能相加为独立成果 |
| 扩展开源与设备生态 | 34仓，包括PyTorch/DeepSpeed/Transformers/TRL/OpenRLHF/Triton/FlashAttention/TRT/TE/FlashInfer/Dynamo/Ascend/ROCm/JAX/XLA/MLX等；日期放宽搜索取1982个PR对象，精确提交窗2168个默认分支commit，release最近页到达回溯边界 | 非正文候选仍需精确merge-time、分支及最终机制审查；PyTorch搜索仅3个对象，已用273个commit表面补查，不能当无活动；1692个未闭合候选保留，且与历史队列相交 |
| 模型/数据与厂商 | 52个HF组织、208次createdAt/lastModified模型/数据查询均成功到回溯下界；196候选；46官网入口中39直取成功，7首次失败里3经web补访 | 数据行、tensor、许可证和所有官网分页并非全部深审；OpenAI部分入口/Recursive等访问缺口、IFM/Indic-CC gated接口与动态站点历史仍保留 |
| 论文/会议 | 14篇固定arXiv v1全文、2篇ACL正式PDF，正文13篇arXiv首次收录、1旧论文工件跟进、2旧正式补读；8个论文仓库入口及部分关键代码 | exact submitted/updated语义、回溯尾页、OpenReview全文与会议全卷历史仍不完整；没有设备独立复现 |
| 硬件厂商 | NVIDIA/AMD技术博客、Gaudi notes、Ascend公告入口、Cloud TPU notes、Neuron2.33.0及索引；实质新代码见工程节 | Ascend公告为动态壳，不能据200确认版本已查；CANN/HCCL、libtpu与Neuron组件历史未闭合。Gaudi1.24.2、Neuron10-19 release、TPU旧入口不能代表所有渠道无新增 |

无正文条目不等于该厂商/仓库没有更新。各清单存在交叉和补录，PR、commit、模型卡、数据工件不按篇数相加为独立成果。

### 论文检索失败与未完成区间
- 七个arXiv类别new页面均200，仍显示10月30日公告且与前轮逐字节一致；这只说明这些listing未变，不代表主窗没有上传或revision。主窗submitted请求返回0，但query title把范围变成引号；真实updated请求又被服务端规范化为submittedDate，均不能用来证明“无新增／无修订”。回溯请求收到2000／2530条，第二页先SSL失败、重试500，日期语义和尾页仍未闭合。
- HF daily-papers：10月30日39条、10月31日200空数组（前轮400访问缺口恢复）、11月1日400。按日接口没有主窗严格语义；未把HF列表摘要当全文研究证据。中英文主题检索用于发现，正文均回到固定全文。
- OpenReview的TMLR页面可读，但notes API与Kuramoto／spectral-rank／weight-clustering三个旧失败PDF、两个新增候选Complete Scion／Thinking Is Not Telling PDF均403。jmlr.org的TMLR正式列表恢复，比10月31日多12个forum，仍只知月份，未读正文者保留候选；不能据“有正式列表”关闭全文缺口。
- ACL EMNLP2026事件页和PMLR首页可访问；本轮只完成两篇ACL正式PDF补读，未穷尽volume、PMLR所有新卷或发表历史。14篇arXiv固定全文、8个链接仓库库存／README和部分实现源读，各自与训练／硬件复现分账；未审库存继续保留，不宣称论文全生态语义覆盖。

### 补录、更正与续扫水位

- 13篇10-28—29论文、两篇旧EMNLP论文、Indic Specialized等旧待审工件和部分开源PR均标原始日期；Megatron #8522为10-27窗外队列补录。首次收录是本归档的状态，不是今天首发。
- ParaTTT已在10-30日报收录，本轮只补“固定仓库无实现/许可证”核验，并保持4096训练长度与更长评估长度分开；不重复记新论文版本。FairStress关闭已证实的加载配置缺口，不扩大为新数据/新结果。
- 本轮证据审查保留并修正Megatron router标题/最终核差异、ATOM新旧性能表及head差异、TE确定性例外与测试checkpoint、nesformer teacher-forcing范围；未用“文档已取得”替代最终执行证据。
- **未完成区间继续保留：**核心旧队列最早10-05—08，扩展PyTorch/Transformers/Triton旧区间10-05—12，动态厂商/硬件历史从10-05起仍有缺口；arXiv真实updated及替换版本历史至少从10-13起未闭合。当前72h回溯至cutoff仍有大量语义待审。每个来源的原始区间和候选文件保留在本地`.monitor/state.json`及本轮账本，未以本日cutoff覆盖它们。
- 下轮从**2026-11-01T00:57:10Z**继续并回溯72h，先处理待传和高价值旧缺口。MTG-AD `2610.39955`只获卡片线索/论文路由，全文仍待审；vLLM #61667/issue #61532仍open；TE #3552原生失败未因本轮其他修复而关闭。
- 本地归档与远端同步分别记账；上传回执保存固定commit、逐字回读与历史blob保留结果。来源失败不记成扫描成功，上传失败保留待传，不重复生成已保存正文。
