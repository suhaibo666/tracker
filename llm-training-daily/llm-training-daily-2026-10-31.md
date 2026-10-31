# LLM 技术动态日报 · 2026-10-31

> 时区：Asia/Shanghai（UTC+8）。增量扫描窗口：**2026-10-30 09:03:12 → 2026-10-31 09:02:00**；72小时回溯：**自2026-10-27 09:03:12起**。实际生成时间：**2026-10-31T09:24:38+08:00**。文中原始UTC时刻保留Z；只有日期或会议月份时不推定精确首发时刻。
>
> 本轮先读自动化memory、持久化state/sources、24篇本地日报、认证远端main索引/树及上一期完整正文。25个历史归档blob（日报与索引）与本地一致；远端基线`720dd21358b7eaf14c3b48a46443e91bdd435e99`，无旧待补传。只维护llm-monitor与tracker/llm-training-daily，未改知识库wiki。
>
> **作者自报结果、代码/工件静态证据与本轮机械核验分开；未独立运行训练、模型GPU/NPU/TPU、集群性能或设备故障注入。** “已合入main”“支线合并”“正式release”“未合入PR”分别标明。扫描cutoff是续扫事件边界，不代表历史和语义全覆盖；所有旧失败与未审区间继续保留。

## 本次重要变化

- **小模型训练开始把执行环境作为监督的一部分。** Mage-VL-Cowork交付4B权重和361题复现overlay，教师harness消融与学生增益分开测；Apriel Voice公开语音适配、全文词表蒸馏流水和完整系统的评测口径。
- **循环架构有新公开工件，也暴露导出与服务边界。** SenseNova Loop把共享物理层、两路mHC和逐访问缓存落到代码；9片原始tensor头与卡的BF16说法存在差异，SGLang支持仍在open PR。
- **预训练结论需要强基线。** RethinkingPPT在初始化和等算力控制后未复现所测合成预热的普遍优势；AIR、Multi-View MoE和Soft-MTP分别探索有效学习率、专家输入表示与训练梯度复用，收益和实验规模不能混为一谈。
- **推理可靠性比版本号更值得跟。** vLLM EP scale-down有外部编排和不可回滚边界；SGLang修复TP renorm漂移、graph资源寿命、HiCache事件等待和无法恢复的KV retract。大部分是开发分支增量。
- **训练系统要同时读最终实现与失败记录。** DeepSpeed共享K/V梯度归约值得复现；CPU-offload最终代码没有PR所写的固定分块上界；TE延迟wgrad补丁仍披露SM100未闭合失败，不能按“测试已过”采用。
- **低精度、缓存和调度各有可复现条件。** DeepSeek新NVFP4工件依赖未合入MXFP4 Engram运行时；FA4 SwapAB最终dispatch与最初PR描述不同；dLLM-KVQ、DWISE、Raincheck、Rimuru需要把队列收益、延后补算和模型质量分别比较。
- **评测与公开工件的偏差有新证据。** Trident公开checkpoint不对应论文分域表格且不支持直接低维截断；Gemma tokenizer修复可机械核验；Perplexity/Octen揭示答案/完整证据、热门/冷页面及私有训练评测之间的区别。

目录：[模型与报告](#models) · [预训练与架构论文](#pretraining) · [后训练论文](#posttraining) · [推理论文](#inference) · [多模态与检索](#retrieval) · [工件与评测更正](#artifact-corrections) · [核心开源工程](#core) · [扩展开源工程](#extended) · [部署验证](#hardware) · [正式会议与修订](#formal) · [覆盖与缺口](#coverage)

<a id="models"></a>
## 模型与技术报告

### Mage-VL-Cowork：先改教师 harness，再用五阶段训练学会 GUI 与终端协作

**Microsoft Mage；10-30 发布报告、权重和 OSWorld 复现代码。** 权重基线 [`6fbe7f91`](https://huggingface.co/microsoft/Mage-VL-Cowork/tree/6fbe7f91da477c1c73790dbb1f08d001b8626be1)；代码与 [30 页报告](https://github.com/microsoft/Mage/blob/d213d430de4875c4c5c90e84b792410f704c8a23/assets/mage_vl_cowork_tech_report.pdf)基线 `microsoft/Mage@d213d430`。Apache-2.0。

问题是纯截图操作昂贵，而直接给教师更多工具，也可能把低效的通道选择习惯蒸馏给小模型。作者从中间版 4B Mage-VL 出发，依次做 GUI mid-training、GUI SFT、终端 SFT、混合 SFT、在线混合 GRPO；先把两种通道分别练好，再由同一策略选择点击或写代码。上游把任务生成改成可复用的 skill/workflow，生成 293,806 个任务、保留 318,837 条混合轨迹。§6.7 的 672 题受控实验中，去掉冗余 AX 通道等 harness 变化使教师成功率 89.6→92.0%，相应学生 25.1→37.6%；这组消融不是下述 OSWorld 总分。

§6.1、Table 2：**361 个 OSWorld-Verified 任务、排除 Google Drive、50 步、两次运行均值 66.0%**。同一训练序列 Hybrid SFT 为 55.84%；Table 6 的平均调用数 21.57→18.77、输出 token 7,374.4→5,498.6。跨模型表混有不同步数、作者引用成绩和缺失配置，不能据此作严格统一排名。Table 3 的任务生成成本 $0.906→$0.158 是**经常性成本，排除一次性 workflow 编写和调试**。

系统方面，§4.5/§5 的异步 RL 每更新 96 组×8 episode，最多滞后 3 次更新，TIS 截断 2.0；32×B200 中 24 卡训练、8 卡推理，另有 1,120 个 CPU GUI 槽。7.79 ZFLOPs 是作者的名义 LM 计算估计。活跃训练/推理窗口 32K，配置的 262K 架构上限不代表已经验证 262K agent 任务。公开树交付报告、推理权重和评测 overlay；本轮未核实完整数据合成、训练及云端 RaaS 的可运行交付。

已读实现对应：`MAGEHybridAgent._build_messages` 保留近期截图/回答并折叠旧 Action；`parse_response`/`execute_bash_actions` 与 runner 区分 GUI、bash 执行和回传；[复现说明](https://github.com/microsoft/Mage/blob/d213d430de4875c4c5c90e84b792410f704c8a23/mage_vl_cowork/osworld_eval/README.md)固定 SGLang 0.5.2、Torch 2.8/CUDA12.8、Transformers5.7，采样 0.7/0.8、每步4096 token。**建议深入阅读并考虑复现教师 harness 消融与 361 题评测**；要一起固定 VM、依赖包注入与错误重采样，不能只换权重比较。

### SenseNova-Loop-9B-Preview：循环层、mHC 与逐次访问独立缓存落到公开工件

**商汤 SenseNova；卡标 10-30 发布，权重文件上传记录从 10-29 13:46Z 可见，仓库 10-28 初始占位不是权重首发。** [固定权重、配置与自定义实现 `27abea77`](https://huggingface.co/sensenova/SenseNova-Loop-9B-Preview/tree/27abea779b1e6634b136805b6519371739d551f7)，官方代码基线 [`a521b0af`](https://github.com/OpenSenseNova/SenseNova-Loop/tree/a521b0af970f0c363c6845ff1ddb465e995178a7)，Apache-2.0；完整技术报告仍 TBD。

配置有32组物理 decoder 权重，先执行0–3，再把4–27执行两次，最后28–31，共56次层计算；每四层是三层线性注意力、一层全注意力。中间区间用两路 mHC 残差流与20次 Sinkhorn 归一化，增加执行深度而不复制重复块权重。已读 `execution_plan`、`MHCRouter.forward/merge` 与 loop forward：同一物理层复用权重，但 `_VisitCache` 为每次虚拟访问分开 KV/卷积/递归状态；已填充的外来执行计划缓存会被拒绝。**权重共享不能理解成两次访问也共享同一缓存。**

卡给约9B参数、约14.6B“推理参数”等效口径，后者不代表额外存储一套参数；训练上下文最多262,144。导出精度需特别核对：卡写 BF16 主干、FP32 mHC，但当前固定工件的 HF safetensors 元数据及索引显示**约38.14 GB、9.656B参数；本轮9个safetensors头部逐片统计为9,412,959,856个F32参数、243,290,624个BF16参数**，不能按“9B BF16约18GB下载”估算。mHC代码还显式要求路由参数保持FP32；设备加载与峰值显存须实际验证。

Transformers 自定义代码已在工件中可读，尽管资源表仍写实现发布详情待定；卡推荐 Transformers5.18/Torch2.11/CUDA13。SGLang [#47281](https://github.com/sgl-project/sglang/pull/47281)本轮仍 **open、未合入**，卡也要求使用含该PR的构建。MTP标可用；DSpark、DFlash2仍在训练，不能写成已经交付的加速。公开性能图缺少足以重建统一比较的完整协议，本轮不转写排名或速度倍数。**建议深入读执行计划和缓存实现，继续观察训练报告与主线服务支持**；复现时优先检查循环访问、mHC精度与模型导出dtype。

### GPT-Rosalind-6：同 Astra 权重的新部署与评测增量

**OpenAI；10-30 系统卡增补及 API `gpt-6-rosalind`。** 一手来源：[系统卡增补](https://deploymentsafety.openai.com/gpt-rosalind-6)、[API changelog](https://developers.openai.com/api/docs/changelog)。这是受审查组织的生命科学研究部署。**§2明确与GPT-6 Astra使用相同权重，没有为本次发布添加训练数据或额外后训练**；不把名称变化当作新架构或新训练方法。

§3按本次相同题集重评5.5基线，在所测最高推理预算下：MedChemBench v7的99题为49.6% vs28.9%；GeneBench Pro v13的129题、每模型每题7次响应估计单次成功率37.7% vs16.7%；LifeSciBench Gold排除6题后94题为61.3% vs44.0%。GeneBench使用与生产不同的专门配置；新评测版本也不能直接和旧系统卡数值拼接。

这是关于部署边界和评测可比性的证据，底层架构、数据配方与完整训练实现没有本次新披露，亦无公开权重供本轮复现。安全表还注明部分比较的版本、推理/评分和样本覆盖不同，不构成受控因果比较。**建议深入阅读评测口径与“相同权重、不同部署配置”的分界，继续观察可复现评测材料**；不据此推断新增领域训练。

### DeepSeek-V4.1-Flash-1017 NVFP4：专家值转换可无损，Engram 与激活量化仍有损

**NVIDIA；10-30 新量化工件，ModelOpt v0.48.0rc0，源revision `1bae0e10`。** [固定卡、配置与48片权重 `480550d3`](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4/tree/480550d349ede47a048eb65a276c5c01a06256e2)，MIT；原模型不是本日新发。

原路由专家已经是MXFP4。本次将40层×384专家的w1/w2/w3转为NVFP4 W4A4、group16，Engram从FP8改为MXFP4；注意力、共享专家、视觉与MTP等保留各自源精度。作者检查169.87亿个专家weight block，报告反量化值保持（带signed-zero规范化）；**这不包括有损Engram，也不证明激活或全模型输出相同**。激活校准用cnn_dailymail与Nemotron-Post-Training-Dataset-v2各512样本、长度512、batch4；46,080个投影条目中24个用了fallback scale。体积约475→400GiB，并非整个模型压到原来四分之一。

测试为4×GB300、TP4、FP8 KV、MXFP4 indexer、CPU offload Engram。vLLM0.31.0加定制改动：GPQA92.77→92.93、AA-LCR80.41→80.19、SciCode57.58→57.32、Terminal-Bench2.1为81.81→81.11；常规任务temp/top-p=1、max effort、最多262K输出，重复次数随任务不同，Terminal-Bench另用每题8轨迹、两次运行均值。没有统一吞吐/延迟改进证据，MTP未测，SGLang未验证。

关键落地边界是vLLM [#61530](https://github.com/vllm-project/vllm/pull/61530)：**本轮仍open、未合入**，常规镜像无法直接加载MXFP4 Engram。**建议考虑小规模精度和装载复现，继续观察运行时PR**；固定checkpoint、ModelOpt和运行时patch，把专家值等价、Engram误差、激活误差分别验收。

### Joey-1：2B 决策模型公开可运行工件，训练长度与部署口径需分开

**Base Labs / Baseten；10-29公告，10-27 20:23Z已有权重，本轮补录。** [固定权重/卡 `9efbe485`](https://huggingface.co/baselabs/joey-1/tree/9efbe485efbed9598dfca5294cbbe8da00e21908)、[公告](https://www.baseten.co/blog/decisions-for-a-penny-introducing-joey-1/)、[家族训练报告](https://labs.baseten.co/articles/mammoth-1-and-elephant-1-frontier-decision-models)。MIT。

从Qwen3.5-2B对文本/图像决策做语言模型全参数微调、冻结视觉编码器；全注意力层改block-bidirectional，线性层仍因果。一次共享前缀prefill后读出各问题选项概率，避免长文本生成再解析。卡披露Mix8的6.21M样本、CE+Brier（权重1）、含“I don't know”、1epoch/43,390步、16×H100、BF16；家族报告主要论证9B/27B，2B具体参数以本卡为准。

卡报Jev Decision Index0.4 public split63.30（elephant72.24），Artificial Analysis **内部复刻**62.9，不是官方榜成绩。公告20ms P50对应生产平均2,000-token请求，但未披露硬件/并发与延迟分布，不可外推自部署。公告说专用部署可到262K，**工件明确训练和接收上限16,384，超长拒绝而非截断**，不能用配置的262K架构上限代替训练支持。

`setup.sh`固定OneJev `81ce62f1`、Decision Index kit `f5a6e4fd`再打`serve.patch`；patch的`DecisionEngine._split/_blocks`约束非因果块边界必须与共享cache边界一致，证明不能直接替换普通自回归serve路径。配套数据集本轮401，未验证完整训练数据可获取。**建议考虑路由/分类/拒答校准复现**，与生成式小模型在同负载上比较概率质量和端到端成本，先用16K内输入。

### Apriel Voice 27B Preview：语音适配与全词表在线蒸馏公开报告

**ServiceNow；权重实际在10-29 16:55Z上传，10-13仅初始占位，本轮补录。** [固定卡与模型代码 `a4feb3f7`](https://huggingface.co/ServiceNow-AI/Apriel-Voice-27B-preview/tree/a4feb3f79ed1f03a3c181a05b4ec068557a43c71)、[官方完整技术报告](https://servicenow-ai-apriel.hf.space/assets/Apriel_Voice_TechReport-B5MRQync.pdf)。卡标MIT；底座及第三方数据仍需遵循各自条款。

把冻结Whisper large-v3-turbo接到Gemma4 26B-A4B：约47.7M projector将8帧一组压到语言hidden维，先只训投影，再训投影/语言骨干/router，保留冻结的视觉接口。按数据源选择CE、forward-KL与文本replay，让学生听音频、冻结教师看对应文本。开源推理`AprielVoiceModel._prepare_audio_embeds`把投影特征写到指定audio span；processor负责16kHz单声道及30秒分块。输出是文本/工具调用，合成语音、轮换与执行工具由应用完成。

报告§7的工程增量比“接一个语音塔”更值得读：8节点×8 H10080GB，每节点2卡teacher/6卡student，teacher BF16全词表logits走节点内P2P，student单独FSDP；单batch lookahead重叠teacher(k+1)与student(k)，存checkpoint前排空请求再重启流水。融合FKL按词表tile重建概率，用同一全行normalizer，并按监督token数加权序列分块；**完整logits仍物化，只省FP32概率等中间量**。阶段二17.5B token/epoch、3epoch约52.6B暴露量，不能当52.6B独立数据；作者报优化后约3天，未给每项独立加速归因。

Table6/§8：thinking模式EVA-A/X为51.27/74.13、τ-Voice成功65.51、完整语音系统均值延迟1.91秒；Cartesia Sonic3.5合成计入延迟，EVA/τ-Voice用三域等权均值，**不同于公开榜按case池化**。卡的约2×输出吞吐来自4×H10080GB的定制优化镜像与baseline Apriel-vLLM对比，不能归为模型单项加速。英文优先、长音频会退化；冻结视觉塔也不保证更新后的decoder保持全部图像能力。**建议深入阅读在线蒸馏流水与比较协议，考虑按原镜像复现**；完整训练数据/所有训练内核可运行交付本轮未确认。

### Apex-35B-V1.0：研究型中训练/后训练模型的开放权重补录

**Apex Intelligence / 超衍智能；10-28 10:01Z首个实物上传，本轮补录。** [固定卡、配置、权重 `5105f8f9`](https://huggingface.co/Apexintelligence-AI/Apex-35B-V1.0/tree/5105f8f9ed8fd8212a72da546bcbb4f01d2cfd68)、[官方发布说明](https://www.apexin.com/apex-35b/)。基于Qwen3.6-35B-A3B继续中训练/后训练，35B/3B激活、262,144输入输出共用窗口、Apache-2.0；BF16与官方GGUF工件可见。

可验证增量是可下载的研究/代码/工具任务checkpoint及部署说明。作者称Apex Research参与数据构造、实验与训练/推理优化；但数据配方、阶段token、算力、完整RSI闭环实现和关键消融没有足够披露，**不把“自主参与研发”解释成已证实的递归能力增益**。卡比较SciCode主问题62.6 vs同规模Qwen37.5、PaperBench60.9 vs45.8等，但未给足以重建全部比较的统一评测配置；CFD/RoboRSI/Real2Sim含内部任务，不能当独立榜验证。

部署特别要求专用chat template末尾`<think>`；卡推荐SGLang/vLLM/TokenSpeed等版本，说明不同框架是否保留往轮thinking会影响agent表现。**建议继续观察完整训练报告；有研究agent负载时考虑本地同harness对照**，先复核模板、上下文和任务集，再判断是否优于底座。

<a id="pretraining"></a>
## 论文：预训练、优化器与架构

本节按论文实际提交历史去重。除明确标出的旧作补录与修订外，下列首稿发表于 **10 月 28—29 日，属于回溯首次纳入**，不能写成 10 月 30 日首发。已打开固定版本方法、结果表和局限，未独立训练、测量加速器性能或复算私有数据集；代码可访问与论文结论已复现分别说明。

### Soft-MTP：把已有交叉熵梯度改接为自蒸馏，收益需要按增量读

**`arxiv:2610.38331v1`；原始提交 2026-10-28 06:31 UTC；MIT。** 标准 MTP 的梯度可拆成使 drafter 接近本模型 next-token 分布的 KL 项，以及数据交叉熵残差。Soft-MTP 在共享线性输出头条件下，用前向数值为零的 stop-gradient 路由抵消后者；MTP 头与 trunk 收到 KL 梯度，输出头仍收 CE 梯度，复用 fused CE 而不显式保存全词表教师概率（§4—5）。[固定版本全文](https://arxiv.org/html/2610.38331v1)。

§6／Table 2—6：167M—1.2B LLaMA 风格、3 个 DeepSeek 式 MTP 头、FineWeb-Edu 31.5B tokens、Muon，五域各 256 个 512-token prompt。相对**标准 MTP**，平均接受长度增 1.8—2.2%、推理速度增 1.4—1.5%；1.2B 的 FineWeb-Edu 自投机速度为 1.74×→1.76×，不是新增 1.76×。同尺度零样本均分 56.80→57.31，部分单项仍下降。单 H200 成本对比中，Soft-MTP 吞吐与 MTP 相差 0.07% 内；显式 KL-MTP 慢 5.6%，峰值多 21GB，但各方法吞吐采用各自可容纳的最大 microbatch，显存比较才匹配 microbatch19。

**建议优先复现梯度等价与小规模训练。** 证据止于 1.2B／约 31.5B tokens，前沿尺度并行兼容性是设计讨论。作者 [MIT 代码](https://github.com/JaydenTeoh/Soft-MTP) 冻结于 `4e1cf661`，已对读 `combined_mtp_loss` 的位置对齐、按有效 target 数重新缩放和 CE 分支；仓库含梯度测试并链接权重集合，本轮未下载权重或运行训练。不能把“一行改动”省略成无需处理 loss normalization。

### Rethinking Pre-pretraining：先把初始化和等算力基线调好，再判断合成预热

**`arxiv:2610.40051v1`；原始提交 2026-10-29 13:45 UTC。** 论文重新比较 NCA、k-Dyck、DeclRef 合成预预训练与继续使用自然文本，把“额外加算力”和“固定总算力分配”分开，并为真实文本控制保留连续 LR／优化器状态，避免阶段切换本身伤害基线。[固定版本全文](https://arxiv.org/html/2610.40051v1)。

§3—7／Table 1、8—16：400M Llama、约 9B OpenWebText 的复现中，NCA 原有 5.4% PPL 优势被仅改变 embedding 初始化带来的 9.0% 基线改善消除。336M 的加算力实验在 50M／100M 自然 tokens 有收益，1B 时落入调好初始化基线的种子波动。FineWeb、Muon／AdamW、1e18—9e18 FLOPs 的等算力实验中，所测合成 PPT 未在自然语言 BPB 上显著胜过继续自然文本训练；9e18 时 baseline 0.8733±0.0002，NCA1% 0.8751±0.0024、k-Dyck1% 0.8765±0.0021（越低越好）。少数 DCLM／BLiMP 子项有收益，C4 控制也能达到相近水平。

**建议深入阅读，用作合成 curriculum 的实验设计检查。** 这是否定所测 recipe 的算力收益，未否定所有合成数据。9e18 以上未做完整 HPO，1%／5% 配额不能代表最优配额；交错混合能保住所学合成能力，但该较大实验未充分调参。作者 [MIT 代码](https://github.com/alexkstern/rethinking-ppt) 可访问，冻结 commit `aec57c02`；树接口初次SSL失败，重试恢复127文件库存，未声称完整实现审计，未独立复现。

### AIR：把调度对象从名义 LR 改成参数块的有效 LR

**`arxiv:2610.40362v1`；原始提交 2026-10-29 16:15 UTC；HTML404 后恢复 PDF 全文。** 名义 LR 相同并不等于相对参数更新相同；论文用“名义 LR／块范数”解释固定范数 WN 与 weight decay 的轨迹差异，再把 inverse-square-root 曲线做仿射变换，匹配有限训练终点、最终 LR 与累计 LR（§2—4）。[固定版本 PDF](https://arxiv.org/pdf/2610.40362v1)。

Table 1／Appendix C：Muon+WN，4.7B 总／110M 激活、128B tokens、16 GPU×48h，以及 35.7B 总／790M 激活、500B tokens、32 GPU×168h。等 warmup、最终 LR、累计 LR 且调过 cosine 基线后，大模型 eval loss 1.843→1.832、11 项均分 56.22→57.78；DROP 56.0→51.2，非逐项胜出。尾段交换消融支持最终 LR 形状对终点影响更大。

**建议做低成本调度对照，先记录逐块范数。** 理论的最优性依赖固定 oracle 方向、切向定幅更新等理想化条件；报告为单条训练轨迹终点，未跨训练种子平均，GPU 型号未明确。正文没有作者完整实现链接，本轮未核实公开权重或可运行代码；不能据此保证把 AIR 名义 LR 直接搬到任意 AdamW 训练都获益。

### Multi-View MoE：让 router 同时决定选谁和喂给专家什么表示

**`arxiv:2610.40577v1`；原始提交 2026-10-29 17:45 UTC。** 多个 router 对完整 hidden state 的 sigmoid 分数先求和再 top-k；进一步把每个专家在各 router 上的相对分数作为多个 latent view 的混合权重，先混输入再执行一次专家。这样保持 top-8 激活与通信向量数，避免每个 view 各跑一次专家（§3）。[固定版本全文](https://arxiv.org/html/2610.40577v1)。

§4／Table 1、Appendix B：24 层、10B-A1B、256 experts+1 shared、latent512、MLA、708B tokens，同数据／优化器配置；九任务均分 LatentMoE 33.07、仅 Multi-Router 33.69、Multi-View 34.25，GPQA +5.60 点而 BBH -0.34 点。参数约增 0.19%；**保持专家 FLOPs 不等于总算力零开销**，额外 router、投影与混合仍需计算。附录吞吐表主要是 router／view-fed 消融配置，不能当主方案无开销的直接证明。

**建议架构团队阅读并观察复现。** 主比较共享单一 seed42、单一 backbone 与 checkpoint；有受控输入混合干预，未建立跨规模规律。正文未给公开作者代码／权重链接，本轮未核实完整工件；先在相同 expert budget 下复验收益和真正 E2E 吞吐。

<a id="posttraining"></a>
## 论文：后训练、蒸馏与学习评估

### Unified Training：同一加权似然接口混静态数据和成功 rollout，配比仍有代价

**`arxiv:2610.40564v1`；原始提交 2026-10-29 17:42 UTC。** 将二值奖励 MaxRL 写为拟合成功条件分布，成功 completion 权重为采样成功率的倒数，失败权重为零；静态语料、SFT 和 RL 以 token 权重及数据比例进入同一 trainer，generator 异步补数据（§3—4）。异步生成策略与当前策略不同后，更新一般不再严格等于当前策略的 MaxRL 梯度。[固定版本全文](https://arxiv.org/html/2610.40564v1)。

§5／Appendix C—D：123M 控制推理题验证阶段交错、保留与泛化；0.76B 从零预训练模型的 12K-step 数学 RL 中，混 0.5% 静态 midtraining 数据使 held-out loss 相对底座变化由不混时 +0.0737 变为 -0.0871，SciQ 50.2→62.0，但 HumanEval+ 15.17→11.97，不能称能力全面保留。之后做同样 4K-step coding RL，前期5%混合模型的 HumanEval+／MBPP+ pass@1 为24.96／32.69，阶段训练为22.94／29.81（Table22）。

**建议阅读并复现小比例 replay 与退火。** 论文未给普适最优配比，实验主要小模型、单 RL 任务；算力比较不是预测性 scaling law。[Apache-2.0 实现](https://github.com/yudasong/unified-training) 冻结 `5b3eaa0f`，公开训练／评估脚本并链接起始模型与数据；代码安装要求 CUDA13 对应驱动。本轮做源读核验，未运行完整实验。

### WISD：把模型故意削弱后，利用原模型与弱副本的分歧产生监督

**`arxiv:2610.38681v1`；原始提交 2026-10-28 18:55 UTC。** 用原模型生成固定轨迹，再用原模型和经3-bit RTN／幅度剪枝的冻结副本在同一 prefix 上打分，构造 log-probability 差值与 tilted target；学生从原模型初始化。没有外部答案或 verifier，**实现是固定轨迹的 offline DOPD**，不能把标题中的 on-policy 读成每次更新都重新 rollout（§3）。[固定版本全文](https://arxiv.org/html/2610.38681v1)。

Table2—3：五个4B—35B-A3B推理模型，每域3,200 prompts×4轨迹，200 updates；数学 Avg@32、代码 LCB v5/v6 Avg@4。量化／剪枝在25个模型任务格中分别21／23格改善；Qwen3.5-4B五项均分55.95→67.23／69.34，但Qwen3.6-27B量化76.58→76.29。GPTQ3-bit和普通W4A4不稳定获益，不能仅按位宽选择“弱老师”；部分比较方法直接引用已发表分数，非全部等预算重跑。

**建议小规模复现，检查削弱方式而非只看教师强弱。** [Apache-2.0代码](https://github.com/jet-ai-projects/WISD) 冻结`012993f8`；已对读 `rtn.quantize_weight` 的128列分组、QDQ与 `pruning.transform_weight` 的maxabs/6阈值、整行norm恢复，对应§3.3。量化只作用弱 scorer，学生不是低比特模型；上游权重／数据各自许可证，未核实作者成品学生权重或独立复训。

### MTA-OPSD：逐 token 选老师，但测试集挑 checkpoint 和七教师成本要一起看

**`arxiv:2610.40574v1`；原始提交 2026-10-29 17:44 UTC。** 同一学生接收七种参考解截断，形成 privileged-context 教师池；先按教师对采样 token 的相对支持方向做多数票，再在多数组选择与学生 KL 最大者，而非直接选最远老师（§4—5）。[固定版本全文](https://arxiv.org/html/2610.40574v1)。

Table5—8／Appendix C：Qwen3-1.7B／4B／8B thinking、DeepMath、4×H200、300 steps、训练上限4096而评估38912 tokens。AMC23/AIME24/25平均：1.7B底座51.1、OPSD54.0、MTA57.0；8B底座76.4、Paint78.1、MTA79.2。1.7B仅“多数票后随机老师”已56.6，最近老师53.3，七教师逐token选择需额外教师前向。

**建议观察并做严格独立验证集复现。** 作者每50步在测试集评估并报告最佳Avg@12，收益带有checkpoint选择偏差；没有跨领域证据，也没有等墙钟成本优势证明。[MIT代码](https://github.com/xingzhejun/MTA-OPSD) 冻结`9567be28`，有训练入口／verl依赖，本轮未核实可下载成品权重或完整复现实验。

### OPD 的不确定性形状：选老师前，比较在哪些位置分叉

**`arxiv:2610.39020v1`；原始提交2026-10-28 22:53 UTC。** 论文把“哪些 token 是候选”和“在轨迹哪里犹豫”分开，用逐位置熵的 Pearson 相关性 ESC 衡量后者；再用逐token温度移植，保留一个老师的token排序却匹配另一个老师的熵，提供干预证据（§4）。[固定版本全文](https://arxiv.org/html/2610.39020v1)。

Table1—2：R1-Distill-1.5B、Qwen3-1.7B，两组各五教师；7K“双方都错”与7K“双方都对”的子集均可获益，例如JustRL老师在R1学生上分别+6.1／+4.4点，同recipe7B老师约无收益。熵形状移植到学生自身，100步可得+2.2点，原成功老师+4.8点。1,000 probe rollouts上的训练前ESC在**十个配对**中排序相关0.98，两个学生均选中最好老师；这只是很小的数学套件，不能称通用预测器。

**建议深入阅读机制，ESC先作诊断量。** “双方都错”按单次采样判定，不证明错误知识总是有益；训练200步、六数学benchmark Avg@4，无前沿规模测试。[作者仓库](https://github.com/maybenotime/opd-esc)可访问，默认分支master、冻结`5c9a4f3a`；根README404，树接口SSL重试后恢复359文件、可见OPD／温度移植相关脚本，但无LICENSE且API未识别许可证；完整运行契约与许可待补查，未独立复现。

### Turing C-MOPD：多领域教师损失校准与动态 MoE 执行必须分开归因

**`arxiv:2610.39876v1`；原始提交2026-10-29 11:52 UTC；Turing-20B-A2B。** 按任务选冻结专家，在每个老师Top-K局部词表内归一化并做reverse KL，再按老师loss标准差缩放与截断。系统把容量过滤后的路由宽度限32、重放最终专家分配，并使rollout与训练异步重叠；宽度溢出时确实丢弃低分assignment，不是无损压缩（§2）。[固定版本全文](https://arxiv.org/html/2610.39876v1)。

Table4—9／Appendix A：51.5K prompts、400步consolidation；K32等条件下完整校准macro整合分数0.9962，无校准0.9724；所选K16为0.9988，是相对SFT→各领域专家增益的归一化恢复率，非原始准确率。8K响应上限四数学／代码均分85.92，对照Qwen3-8B80.33；放宽Qwen到32K后84.64，其中MATH50095.0高于Turing的82.4。广域GPQA40.40→35.35。PPU profile step398.7→139.3秒（2.86×）来自完整组合；异步单独仅398.7→330.1，未独立剥离RW32和route replay。

**建议动态路由后训练团队阅读，暂作报告证据。** 局部KL不约束support外总质量，clipping饱和token梯度为零；异步有一步策略滞后。PPU型号／规模与独立组件收益不足以跨硬件外推。本轮未核实公开权重／完整实现，未把已知8月底底座报告当成本次发布。

### BookNavigator：结果奖励会让模型少查书，探索与工具信用需分别激励

**`arxiv:2610.39671v1`；原始提交2026-10-29 09:21 UTC。** 从158本数学教材提取5,183题、移除参考材料中的习题与解答，训练模型通过目录／检索／读节工具获取知识。正确且用工具的轨迹获得退火探索奖；冻结tool judge看下一轮推理评估本轮检索，将优势按回合重分配、保持token加权均值（§2）。[固定版本全文](https://arxiv.org/html/2610.39671v1)。

Table1、4：Intern-S2-Preview-35B、4,557训练题、50 GRPO rollout steps、8轮／64K轨迹；GLM5.2八次保守评审判证明。完整方法总体79.5%，top-k预检索73.1%，无书RL68.5%；相同互动协议仅正确性RL为74.9%，加tool credit76.2%、加探索77.0%。纯结果RL工具轨迹占比从底座36.9%降至6.7%，完整方法58.4%；OOD书／学科有收益，但CL-bench总增仅2.3点，程序执行类略降。

**建议阅读奖励消融，在自身可验证任务复现。** 主要是LLM判分数学证明，未以形式化验证证明正确；单底座、50步和特定教材分布，不能称持久参数自学习，测试时不更新权重。正文未给可验证完整数据／代码／作者权重链接，训练硬件未明确，本轮未核实开放许可证与可复现工件。

<a id="inference"></a>
## 论文：推理、缓存与服务系统

### dLLM-KVQ：扩散模型的缓存量化，首先要隔离缓存复用本身的误差

**`arxiv:2610.39853v1`；原始提交2026-10-29 11:31 UTC。** dLLM的K/V会因mask逐轮变化，AR式单pass校准不能代表真实轨迹。TRACE用受块窗口／每步reveal数约束的min-cost-flow重建轨迹，再逐层选择哪些K token保留高精度；V仍走简单QDQ（§3—4）。[固定版本全文](https://arxiv.org/html/2610.39853v1)。

Table1—4：LLaDA-Instruct与Dream-Base-7B、H200140GB、生成256／block32、threshold0.9、prefix／dual cache、四数学代码任务。LLaDA2-bit prefix均分44.38，高于KIVI43.62、KVQuant41.45，但低于未量化cache45.08；Dream2-bit dual48.83，对照KIVI48.10、未量化49.65。4-bit K策略实际有效位宽prefix4.77／dual4.60，并非所有K都恰好4bit。

**建议深入阅读误差分解，先复现质量后评估runtime。** 报告只给QDQ任务质量和有效位数，**没有融合低比特attention的E2E速度证据**；固定长度、两模型不能外推所有dLLM调度。正文未给作者公开实现／权重链接，本轮未核实可运行工件；不能把质量改进换算成推理加速。

### DWISE：按剩余 denoising 工作量调度，消除 prefill-only 空转

**`arxiv:2610.39852v1`；原始提交2026-10-29 11:31 UTC。** admission时用prompt轻量预测，首轮denoise后以已有hidden state修正剩余轮数，再按剩余prefill块+denoise工作做可抢占排序；同轮混prefill与denoise，暂停请求保留GPU缓存并预留最坏内存（§4—5）。[固定版本全文](https://arxiv.org/html/2610.39852v1)。

§6／Table2、13—15：SGLang0.5.20、单GH20096GB、LLaDA2.x-mini16B-MoE、SDAR-8B、DiffusionGemma26B-A4B共五模型，4,096请求+独立2,048预测器训练请求；全部同时到达或半数突发+超服务率Poisson。十设置mean E2E比原SGLang快几何均值5.4×，LLaDA吞吐约3.34—3.90→6.39—8.21req/s；SDAR最大14.9×主要来自把长prefill请求后排，吞吐几乎不变。DiffusionGemma不能混phase时仅约1.1×，Poisson下p99变差；2048输出预算的单次复核仍4.45—5.18×。

**建议读实现边界，等代码后在真实到达分布复现。** 主结果是积压／超载负载，非稳定低负载SLO承诺；对手沿用prefill-first，因此组合收益不能归给预测器单项。当前代码仅承诺将公开，无作者可运行仓库链接；本轮未独立复现，也不把调度改善当成kernel算力提高。

### Raincheck：先答后补视觉 token，质量恢复与解码拖慢是同一交易

**`arxiv:2610.39562v1`；原始提交2026-10-29 08:01 UTC。** 初始仅保留30%视觉token，未选token按源顺序在decode空隙补prefill；全注意力靠不同可见性mask插入缓存，Gated DeltaNet用记录的suffix仿射映射修正live state。旧输出／历史KV不重算，状态修正近似全量重跑（§2）。[固定版本全文](https://arxiv.org/html/2610.39562v1)。

Table2—5：Qwen3-VL-8B和Qwen3.5-9B；质量用Transformers、时延用单H20／单运行请求、每步最多31 deferred tokens，固定输出轨迹对照。MultiVerse中MMTok硬裁剪48.77→Raincheck55.22，Full55.43；9B37.53→43.12，Full44.90。图片TTFT降约18—24%，TPOT约不变；视频TPOT反而+14.9—19.9%，E2E仅少4.83—6.56%。视频中42%／69%请求在答案结束仍未补完，补算尾巴不计入该E2E。

**建议多轮视觉服务团队深入阅读。** 多轮质量由特定LLM judge评估，少数turn回退；主时延仅48个对话、单请求，后台空隙与完成后补算需真实调度保证，未证明高并发下同收益。正文未给公开作者代码／权重链接，本轮未核实完整工件；应同时报告答复时延、补算尾时延与下一轮可用性。

### Rimuru：专家权重与 KV／SSM 共用可伸缩页面池，吞吐不等于单请求更快

**`arxiv:2610.40305v1`；原始提交2026-10-29 15:47 UTC。** 在vLLM上用多dtype／shape视图与三级页分配统一expert、KV、SSM、卷积状态；按调度后内存压力收缩或扩展expert hotset，专用page-table-aware FusedMoE读非连续权重，主机权重异步预取（§5—7）。[固定版本全文](https://arxiv.org/html/2610.40305v1)。

§8／Fig9—12：Qwen3-30B、Qwen3.5-35B／122B，1／2／8×RTX5090或2×4090D，ShareGPT；所有可用系统均禁CUDA Graph。对vLLM／SGLang throughput1.1—2.88×，相对Fate最高11.34×；“19.03×对KTransformers”出现在不支持AMX、仅AVX2的老CPU平台，5090+AMX平台最高2.30×。高并发改善来自释放缓存供更多请求，部分TTFT／TPOT反而更高；离散页kernel平均延迟+0.83%，单token最大+9.63%。

**建议受显存约束MoE部署团队阅读，待工件后复现。** 非所有vLLM模型都已验证；一配置仍因Triton GDN autotune OOM而省略vLLM，Rimuru同受其影响。正文有实现规模说明但未给作者可核实完整仓库，本轮未核实代码／权重许可与可运行发布，不把CPU指令集劣势当算法倍数收益。

### DolphinRouting：是否拆分 PD 应按实际缓存命中与工作负载混合决定

**`arxiv:2610.40292v1`；原始提交2026-10-29 15:42 UTC。** 按原始输入长、P／D两侧有效输入长和预测输出长聚类，每个cluster选择本地聚合或远端prefill；先扫描，再用epsilon-greedy搜索联合二值表。离线用真实历史输出长，在线需预测器，worker共享令各cluster收益并不独立（§3）。[固定版本全文](https://arxiv.org/html/2610.40292v1)。

§4／Table1—4：Qwen3.8-72B、每节点4×H20-SXM141GB、800Gb/s RDMA，vLLM+Mooncake，P3D1／P2D2。Hybrid／公开Mooncake trace比所实现Dynamo长度规则P90 TTFT少约6—8%、TPOT少约3—10%；单型ArXiv、L-Eval没有一致收益，HumanEval多项变差。SLO由近空载Naive P90的10倍TTFT／5倍TPOT定义，并非外部固定业务SLA。

**建议观察并做轨迹回放。** 对照只实现Dynamo的长度／命中率规则，不覆盖其额外load policy；结果不能称胜过完整Dynamo。正文未给可验证作者代码、精确后端commit或公开权重，本轮未核实完整实现；先评估聚类漂移、预测成本与真实SLO，勿只优化归一化百分比。

<a id="retrieval"></a>
## 论文：多模态生成与检索证据

### AB-Attention：对抽样块作概率补偿，保留被略过的尾部贡献

**`arxiv:2610.39663v1`；原始提交2026-10-29 09:15 UTC。** 用query块内小聚类估计KV块分数，按inclusion probability采样并以逆概率修正softmax分子／分母，再将4-token选择块拼成硬件tile。分子分母估计各自无偏，**二者比值仍一般有偏**，不能称attention输出无偏（§3）。[固定版本全文](https://arxiv.org/html/2610.39663v1)。

Table1—2：H20/BF16、FA3共同后端、Penguin600 prompts、720p，Wan2.1-1.3B／14B81帧、Hunyuan13B129帧，50步中前10步及首transformer块保持dense。89—92% attention sparsity对应E2E1.84—2.23×而非同倍算力节省；Wan14B时间4743.52→2571.52秒，PSNR30.788，高于SVG229.236。VBench部分维度低于dense，不能只凭reference相似度断言生成质量无损。

**建议阅读、等待实现后做硬件复现。** [官方仓库](https://github.com/InfRec-Cardinal/AB-Attention)冻结`9fa3ed49`，MIT但明确“Code coming soon”，树仅36文件、无Python实现；不能把仓库存在写成kernel已开源。只支持所测H20 forward推理，无backward或跨硬件保证。

### SpecSlot：把像素读取密度、输出频段与全局通信 token 解耦

**`arxiv:2610.40647v1`；原始提交2026-10-29 17:59 UTC。** 每个空间anchor展开多个互斥DCT频段slot，仅低频carrier跨anchor全局通信，其余在anchor内部交互；频域预测加局部空间修正，避免为细节一律扩大二次全局attention（§3）。[固定版本全文](https://arxiv.org/html/2610.40647v1)。

Table1—3／Appendix A—D：约138M参数、ImageNet120epochs，单H800 BF16/Inductor profile。512px的W256/K4：135.64GFLOPs、FID9.14；compute-matched JiT-CM141.21／11.98，更高质量JiT-B/16为219.56／8.45，质量／成本交易仍在。256px T2I400epochs、50步Heun，DPG62.21胜JiT-B/861.15，而GenEval26.99低于29.78；CFG也分别4.0／2.9，不能合并成全面更优。

**建议像素空间生成研究者阅读并小规模复现。** compute-matched256px FID差落在五seed区间内，部分吞吐来自单次profile。作者[MIT实现](https://github.com/JiongfanYang/SpecSlot)冻结`883d7ca0`，有训练／评估脚本，README链接预训练权重；本轮未下载checkpoint、未核权重单独许可证，未独立复训。该工作是视觉生成架构证据，不是语言模型缩放结论。

### Perplexity Contextual Embeddings：答案命中之外，还要找齐条件与解释证据

**`arxiv:2610.38854v1`；原始提交2026-10-28 20:34 UTC；10-30官方模型卡新增报告链接，本轮首次深读。** 9B用context-compressor给文档内chunk分级相关性，再结合document MaxSim InfoNCE，避免单gold-chunk训练把支持片段误作负例；0.6B表示级蒸馏9B的1024维前缀，混25%图像数据，共享embedding空间。索引时每chunk一向量，teacher仅训练使用（§2）。[固定版本全文](https://arxiv.org/html/2610.38854v1)。

Table3、5、7：训练约57M pairs、440个公开／内部数据集、50余语种。Q2D-Chunk含6,069 base／共11,719 queries、9.17M chunks，9B Answer@10=86.4对voyage-context4的83.9，完整覆盖median1954对2412 tokens；但corpus evidence46.6低于非contextual Nemotron49.3。公开ConTEB八任务平均9B83.4、0.6B80.7、voyage80.4；合成context-bench上0.6B Answer@10仅21.9，低于旧4B26.4。仅contrastive的Q2D71.1→联合86.3，直接支持chunk蒸馏的必要性（同step687消融）。

**建议检索团队优先阅读，按答案／支持证据／阅读token预算分别验证。** Q2D-Chunk仍为私有hosted leaderboard、标签由LLM与裁决模型产生，训练含PII过滤生产搜索日志，不能完整独立复现；存储图的Voyage INT8等质量是作者假设。权重与许可见模型工件小节，本条不把新增报告link当新权重发布；未核实context-compressor教师、完整训练数据／代码开放。

### Octen-WebGR：两 token 直接生成 URL，导航新 query 与长尾冷页面要分开

**`arxiv:2610.35256v1`；原始提交2026-10-26 11:39 UTC；旧作补录，超出本轮72小时回溯；本轮发现先前未监控官方工件。** 把Qwen3的URL向量做两级65536码本与无碰撞分配，新增两组SID token；训练query→两token URL，click平方根加权并混URL字符串索引样本。URL文本包括热门query，不包含网页正文（§3—5）。[固定版本全文](https://arxiv.org/html/2610.35256v1)。

Table2—4：4B覆盖完整201.23M universe，9月50K-click样本的导航hit@1=59.6，click-table lookup已58.9；lookup后GR相对dense fallback仅多0.7点，routed最终63.1。真正Held-Nav12967新query：GR32.0对dense20.6，但冷target14.3低于dense21.8；公开ORCAS-Nav为16.2，dense+click prior21.0。单空闲H200参考实现56ms／9.7GiB；batch64的2.3GPU-ms来自另一个KV复用decode，不能直接当开源参考实现性能。

**建议有导航搜索业务时读完整分层表。** click数据、SID构建和训练代码不公开；dense基线只训线性映射、约2%训练compute，非同预算最强retriever。论文所有评测用完整universe，而开放URL表经过删减至195.92M，不能无条件复现59.6。权重／表CC-BY-NC4.0，推理代码Apache2.0；模型卡与下载核验由工件小节承接。本轮不把记忆已见query的高命中泛化成任意网页搜索。

### Tokenizer Tokens Tell Training Tales：词表公开也能泄露数据集成员关系

**`arxiv:2610.39919v1`；原始提交2026-10-29 12:25 UTC。** 以相近分布shadow BPE词表估计token稀有度，计算候选数据实际激活的rare-token mass，再以校准融合判别成员；关注词表是否含稀有token，而非merge排序（§3—4）。[固定版本全文](https://arxiv.org/html/2610.39919v1)。

Table1—3：英语C4按4,133网站划分，五种80K—200K词表、50/50成员；200K时AUC0.875对Vocabulary Overlap0.771，TPR@1%FPR36.8%对34.6%。80K时AUC仍更高，但低FPR TPR23.5%低于26.8%；min-count64防御后AUC0.768对0.717，低FPR尾部却弱于旧方法。不能把AUC改善等同所有实际误报约束下更好。

**建议数据披露／tokenizer团队阅读并观察跨语料复验。** 只验证BPE网站级C4，需要攻击者能训练同过程、近分布shadow；未验证WordPiece、SentencePiece unigram、production跨分布，也未证明DP tokenizer失效。正文未给作者代码／数据新发布链接，本轮未核实完整实现和许可；这种信号不是对任意模型某条训练记录的确定性证明。

<a id="formal"></a>
## 正式会议旧工作补查与版本修正

### UltraX：把网页清洗做成可审计编辑程序，插入操作也必须受保留约束

**`acl:2026.emnlp-main.1132`；EMNLP2026正式版，条目只标October，本轮旧作补录，无本窗口新版本证据。** 从专家端到端清洗文本反推出行对齐的删除／替换／插入程序，用动态扩展上下文消除重复子串歧义；0.6B refiner预测紧凑function calls，滑窗合并、校验后在原文执行（§3）。[正式条目](https://aclanthology.org/2026.emnlp-main.1132/)、[完整PDF](https://aclanthology.org/2026.emnlp-main.1132.pdf)。

§4／Table2、13：1B MiniCPM从零、五语料各约20B tokens、同模型／预算，FineWeb十任务Raw45.08、ProX-C45.05、UltraX46.14；100B复核50.85／51.58／52.13。16B UltraX45.49已过20B Raw45.08。不是所有任务提升，Head高质量文档更适合保守编辑；无公开RefineX实现故未纳入，专家生成和LLM清洗判分可引入偏差，主要限英语及1B规模。

**建议语料团队阅读，优先验保留率和错误编辑。** [作者仓库](https://github.com/OpenBMB/UltraX)冻结`cb0c2fae`，包含程序构造／推理／训练入口；代码Apache-2.0；[0.6B预览模型](https://huggingface.co/openbmb/UltraX-0.6B-Preview)与[清洗数据预览](https://huggingface.co/datasets/openbmb/UltraX-Preview)卡也标Apache-2.0，API可访问，7月已更新；本轮未下载张量／数据行或独立复训。不能据“最高平均分”批准无监督大规模改写原始语料。

### GradPower：优化器前加幂变换，最优幂次随噪声与 batch 改变

**`pmlr:v306/wang26is`；ICML2026正式版，会议2026-07-06—11；本轮旧作补录，未证明本窗口有修订。** 先对每个梯度分量保留符号并取绝对值的p次幂，再交给原AdamW／Muon；目的在现代优化器归一化后改变信噪比，而非单纯放大梯度（§3、5）。[正式条目](https://proceedings.mlr.press/v306/wang26is.html)、[完整PDF](https://raw.githubusercontent.com/mlresearch/v306/main/assets/wang26is/wang26is.pdf)。

§4／Table1、Appendix B：8×A10080GB、LLaMA66M—2B及Qwen2MoE0.5B—2B、C4／OpenWebText、cosine／WSD、约20 tokens／parameter。p只在0.2B调为1.2后迁移；2B六任务均分47.72→48.26，ARC-C26.45→26.28；小模型step0.7534→0.7565秒（+0.4%）。部分配置六seed复核，但更大batch的最佳p下降、可低于1，不能机械推广p1.2。

**建议优化器研究者小规模复现。** flat-direction解释是近似噪声模型，正文承认全梯度在LLM上不可直接验证；没有前沿尺度结论。本轮未定位可验证作者完整代码／成品权重链接，许可与复现需补查，不能把一行公式当即插即用的生产加速保证。

### Proposing Better Rollouts v2：归一化 token occupancy，结果表未改

**`arxiv:2610.36477v2`；v1为2026-10-27 01:33 UTC，v2为2026-10-28 22:30 UTC；已报道v1的修订补录。** [v1](https://arxiv.org/html/2610.36477v1)、[v2](https://arxiv.org/html/2610.36477v2)。逐版全文对照发现Appendix F把token occupancy由轨迹内直接求和改成除以期望轨迹长度Mq的规范化分布，并把risk/objective明确为平均逐token量；所有表格文本不变。它修正理论量的长度口径，不能读成方法或实验新获提升；阅读／复现理论界时应采用v2。

另核对[Verification-Aware Training 2608.30135v2](https://arxiv.org/html/2608.30135v2)（v1为8月31日、v2为10月28日08:06 UTC）：去除引用编号变化后方法、实验表数值均无变化，主要增加相关引用与措辞，不重复此前性能正文。其余replacement与旧候选继续保留待审。


<a id="artifact-corrections"></a>
## 模型工件、评测与复现更正

### Trident：公开的合并训练 checkpoint 不能复现论文分域表格

**OpenBMB；10-30修订卡及独立推理文件，`ed4ff93b`。** [固定更正与工件](https://huggingface.co/openbmb/Trident-Qwen3VL-2B/tree/ed4ff93bd1ed814a8479681b2d8edb1d5eaf3e0c)。这是对10月已有模型的实质披露，不是新权重首发。

作者明确：论文Table1的视觉文档模型、Table4的自然图像模型分别训练于`only_vdoc`/`only_nature`，**两种分域checkpoint没有发布**；目前公开权重使用二域合并数据。重新评估平均NDCG@10，视觉文档Origin/Mix为78.28/70.96，论文分域模型80.51/73.90；自然图像64.86/65.91，论文69.81/76.78。不能把论文最好结果标在当前权重上。

公开checkpoint还**没训练Matryoshka loss，必须保留完整2048维**，论文附录的低维实验属于另训版本。新`trident_embedder.py`提供免clone入口，要求query/doc不同指令和最后token池化；不能据此推定低维截断或任意prompt等价。卡称数据和代码支持按论文另训，完整数据使用仍受原图像许可约束。**建议优先复查已采用Trident的检索结果与维度配置；分域复现需另训**，这比新增单个排行榜分数更有实践意义。

### Gemma4-12B base/IT：删掉通向 glitch token 的 BPE 合并路径

**Google官方HF仓库，10-30合入；贡献者的修复PR分别为[base #13](https://huggingface.co/google/gemma-4-12B/discussions/13)、[IT #55](https://huggingface.co/google/gemma-4-12B-it/discussions/55)。** 固定revision：[`1a2c7729`](https://huggingface.co/google/gemma-4-12B/tree/1a2c7729392586f5ecddae8807ee2ef741eabb9a)、[`43c9d014`](https://huggingface.co/google/gemma-4-12B-it/tree/43c9d014b231474fa7212dbb48a762566c2b255d)。不属于新训练。

作者用embedding分析定位异常token，选择在tokenizer侧阻断输入路径：分别删238/239条BPE merge，覆盖209/210个目标字符串，词表内容和token ID保留。**本轮对前后JSON做了结构比较，确认词表完全相同、无新增merge及上述删除数量**；部分目标有多条合并路径，不能只删一条。作者示例`adiganik`从异常单token改成两个正常子词后，复制任务恢复；本轮没有运行模型验证其全语言/下游质量。

效果边界是常规文本编码路径；保留词表并不从原理上禁止直接传异常ID，或在输出端生成它。缓存/离线语料若存的是旧token ID，不会自动重分词。**建议考虑更新tokenizer并重跑异常字符串、代码和多语言回归**，训练和部署共同固定tokenizer版本；不能声称权重被修好或所有异常都消失。

### MiniCPM-V4.7量化工件：移除指向已不随包交付 processor 的旧配置

**OpenBMB；10-30 06:59–07:00Z的六组工件修复。** [1B-int4基线 `0fdd1c14`](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-int4/tree/0fdd1c14ce647809751eb6ad14f6c2d7a31890df)、[35B-int4基线 `612e2272`](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-int4/tree/612e2272ef6382894095ca060ab7be0155a7234a)，并核对[1B-AWQ `b0c2575f`](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-AWQ/tree/b0c2575f2c2509978f0d13ca8439c3c21427536b)、[35B-GPTQ `d804fac1`](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-GPTQ/tree/d804fac1c1872d5830d4e830e4ba72f173d4b72f)、[AWQ `e83f81c2`](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-AWQ/tree/e83f81c27e20d01d79cebd37f7c1b38afa86aad3)和[FP8 `4b7b0537`](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-FP8/tree/4b7b0537539b0ba6216bc4c4d0247d65956d40e2)。

前后配置显示，多个`processor_config.json`由嵌套的旧`MiniCPMV4_7Processor`/remote-code `auto_map`改为`{}`，相应tokenizer/preprocessor配置去除陈旧指向。解决的是量化包自动processor解析与实际交付文件不一致的问题，**不是重新量化、不是模型质量提升**。本轮读了六组固定版本配置diff，未用各Transformers版本实际装载；空配置本身也不证明图像/视频路径已全部可用。**建议已遇到装载错误的部署优先回归检查**，保持成套模型/processor依赖，不混用旧缓存。

### Dory、ZwZ2与LFM3：只记录这次工件/条件增量

| 项目及日期/版本 | 本次变化、证据与边界 | 实践建议 |
|---|---|---|
| NVIDIA Dory1.0 4B，10-30；[BF16 `f6c832e6`](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16/tree/f6c832e6963ddb74f62c75813a3e8c74b1af67b1)，另有[MXFP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-MXFP8/tree/acb538da38032a1daba9c411d2380d3812eab0a2)与[NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4/tree/64b4039a3468bef85c6744c1e24d5a80801fe965)同日配置 | 前后diff只新增`recurrent_kv_cache_mode: per_loop`，将循环独立缓存的默认口径显式化；不是新增权重或实测提速。vLLM [#60149](https://github.com/vllm-project/vllm/pull/60149)实际已于10-25合入，本轮重新核实其状态；不再沿用10-17的open状态。 | 固定循环次数和cache模式；区分代码支持与设备实测。 |
| inclusionAI ZwZ2 4B/9B/27B，10-30新GGUF；[27B `d8c2afdd`](https://huggingface.co/inclusionAI/ZwZ2-27B-GGUF/tree/d8c2afddeab0ab4faa3620001da36260467aebc1) | [4B](https://huggingface.co/inclusionAI/ZwZ2-4B-GGUF/tree/2998cc8c43b4a130e42541237e98c952d37bb846)、[9B](https://huggingface.co/inclusionAI/ZwZ2-9B-GGUF/tree/206f7713767984556757237bd0ce69addf8d643f)及27B新交付FP16/Q8_0/Q4_K_M语言权重和对应vision/projector。需近期支持Qwen3.5多模态的llama.cpp及匹配`mmproj`；原BF16成绩不能当各量化组合的质量证据。Apache-2.0。 | 考虑本地视觉检索/理解装载复现，分别评测语言和视觉量化组合；新格式不等于新底座。 |
| Reflection Beam NVFP4，10-30；[`c5621fc9`](https://huggingface.co/reflection/Beam-1.0-501B-A23B-NVFP4/tree/c5621fc926758c3fb0a2944645760bcdc17cf084) | 本轮对前后卡逐字比较，LiveCodeBench列87.4%改为88.3%；BF16/FP8列仍88.3/88.1。没有权重变化或解释这次是重测还是排版纠正，不能当新量化算法的提升。 | 更新引用这列分数的笔记；若据此选择精度，重跑相同评测配置。 |
| LiquidAI LFM3-Audio-2B，10-31 00:51Z；[`0295e413`](https://huggingface.co/LiquidAI/LFM3-Audio-2B/tree/0295e4130594c06800fd25451d97139ac2da1aff) | 补充原评测解码条件：audio temperature0.4、top-k0、重复惩罚1.2/window64；文本多用greedy，但VoiceBench用采样0.1及自己的重复窗口。这是复现条件补全，未见新权重/新分数。 | 复现旧结果时补齐参数，避免把greedy与VoiceBench采样混为一谈。 |


<a id="core"></a>
## 核心开源工程：训练并行、推理正确性与恢复边界

主窗 2026-10-30T01:03:12Z → 2026-10-31T01:02:00Z；回溯自 2026-10-27T01:03:12Z。下列为维护组织/社区的一手 PR 与最终 diff；版本使用完整 merge SHA，已与默认分支提交交叉核对。作者性能/准确率结果均有限定环境，本轮未运行 GPU/NPU。

### NCCL EP 为 THD/packing 固定共同容量边界

维护方：NVIDIA 社区；合入 2026-10-30T01:55:48Z；[NVIDIA/Megatron-LM #8458](https://github.com/NVIDIA/Megatron-LM/pull/8458)，main `0f3e582f6e13b94162994710a9780cccad89f8b9`。

各 EP rank 的 packed token 数可变，不能再用首个微批的本地长度初始化通信容量。NCCL EP 现从 max_seqlen_per_dp_cp_rank 或 max_seq_length_per_sample 推导共同上界，按 CP/sequence-TP 切分并向64取整；padding 路由设为-1、概率置零，超界报错。BSHD 仍按首批定容。建议用不等长 rank、padding、后续增大微批做前反向与全零行回归；本轮没有重跑 NCCL EP 内核。

### Megatron-FSDP 预取重计算需要的 row-wise 权重

维护方：NVIDIA 社区；合入 2026-10-30T05:28:41Z；[NVIDIA/Megatron-LM #8515](https://github.com/NVIDIA/Megatron-LM/pull/8515)，main `90e40493e6d71a67ef51ca72a4bb47b8de0bc533`。

MXFP8 等格式的 backward 已取 transpose 权重，但 activation recompute 的 forward 重跑还需已释放的 row-wise 副本，原来每桶按需 gather 会阻塞主流。现记录重计算实际访问且存在 transpose buffer 的桶，在以后 backward 预取窗口补取其 row-wise 副本。无 transpose 时无额外动作，fsdp_double_buffer 禁用此扩展。建议观察第二轮以后的 gather/compute overlap 和峰值显存，不能将此推定为所有 FSDP 配置提速。

### DeepSeek V4 CSA 少保留一份输出 activation

维护方：NVIDIA 社区；合入 2026-10-30T02:26:58Z；[NVIDIA/Megatron-LM #8399](https://github.com/NVIDIA/Megatron-LM/pull/8399)，main `a77288b18f1a3e5a7bfe2e680f6f693a7366c2e5`。

fused CSA 在自身保存的输出上完成 inverse-RoPE，backward 用独立存储重建原 attention 输出与梯度，避免整个前反向期间多留一份输出；非旋转前缀在同一 kernel 复制，保留共享 alias 和上游梯度。覆盖 SBHD、THD/CP 与 fused indexer-loss；依赖 mscale=1 的纯旋转约定，BF16 重建新增一次舍入。作者4GB200短程测试不能代替收敛验证，建议对比显存、梯度和长程 loss。

### RMSNorm gamma 可单独跟随权重衰减计划

维护方：NVIDIA 社区；合入 2026-10-30T23:04:06Z；[NVIDIA/Megatron-LM #8246](https://github.com/NVIDIA/Megatron-LM/pull/8246)，main `e9ddf9b5efc470824fd165348ed776f5e6a19c25`。

新增 rmsnorm_weight_decay_mult，构造器给 PyTorch/TE（含 fused norm）、ModelOpt、Mamba/GDP 的 gamma 打标，优化器用既有 wd_mult 乘全局 scheduled decay。未配置保留原豁免，显式值优先于 Q/K norm 策略；zero-centered gamma 衰减的是存储参数。作者未本地运行 CUDA 测试。建议检查实际 param group、保存恢复和0/0.5/1三档收敛，避免只按参数名猜归组。

### 学习率可叠加 sample 阈值倍率与线性过渡

维护方：NVIDIA 社区；合入 2026-10-30T23:19:46Z；[NVIDIA/Megatron-LM #8524](https://github.com/NVIDIA/Megatron-LM/pull/8524)，main `a2946a03b1c8f889ca79bf53adbbe6d0bc1de2c8`。

step_lr_rescale_schedule 在原 warmup/decay/min_lr 后乘分段倍率，step_lr_rescale_warmup_samples 把后续跳变替换为线性 ramp；适合 batch-size ramp 时同步调 LR。阈值须从0严格递增，倍率有限且为正，过渡不能跨越下一阈值；配置进入 scheduler checkpoint 并按恢复策略核对。默认关闭，建议复现阈值和半 ramp 恢复测试，而非视为新的基础 LR scheduler。

### TorchTitan 启动超时应用到 mesh 子组

维护方：pytorch 社区；合入 2026-10-30T17:33:57Z；[pytorch/torchtitan #5674](https://github.com/pytorch/torchtitan/pull/5674)，main `ff96670a26c7079bc1b9049f5b6942adf35567c1`。

DeviceMesh 子组不继承 world group 的超时，导致配置1800秒却在模型/FSDP初始化时按子组600秒退出。共享引擎和 TorchFT 现于 ParallelismContext 建成后调用 set_pg_timeouts，首次更新后仍切换 train_timeout_seconds。该调用只覆盖 context 跟踪的组，包含一次启动 barrier/device sync；CPU76测未直接证明 NCCL超时。建议在慢启动多节点环境核对实际子组。

### SDC replay 对 RNG 状态逐字节比较

维护方：pytorch 社区；合入 2026-10-30T06:22:36Z；[pytorch/torchtitan #5702](https://github.com/pytorch/torchtitan/pull/5702)，main `1e30d69a3df2c027c7a5ea65d2fd5439bf068f69`。

CPU一次随机抽样的两次计数器更新可能在 XOR tensor hash 中相消，漏报 replay 多消费随机数。CPU及 accelerator RNG 状态现保存为原始 bytes 精确比较，loss/gradient/buffer 仍用摘要。回归测试固定到旧 hash 会碰撞的位置。建议把它用于随机数消费漂移排查，同时保留其他 tensor digest 的碰撞边界，不能据此宣称全部状态都精确比较。

### 自定义 positions 的长文档 packing 不再越过 RoPE cache

维护方：pytorch 社区；合入 2026-10-30T01:29:55Z；[pytorch/torchtitan #5664](https://github.com/pytorch/torchtitan/pull/5664)，main `8b136eb27e38955d59e0e41985817b59dfedab6f`。

uncapped ConcatThenSplit 原来直接透传自定义 processor 的 positions，长文档一行可出现超过 max_context_length 的索引。现以输入 positions==0 识别文档起点，对文档内 token 下标按 context 长度取模；缺省 positions 路径保持同样规则。内置 Text/ChatProcessor 不提供 positions，因此主要影响自定义 processor。建议覆盖多文档、长文档及 max_num_documents=None 的回归。

### RL resume checkpoint 改为不透明 resume 状态

维护方：pytorch 社区；合入 2026-10-30T22:58:36Z；[pytorch/torchtitan #5718](https://github.com/pytorch/torchtitan/pull/5718)，main `8e4990d8c61997249e2f7444ccbb03622639114a`。

trainer 不再包装或理解 rollouter 的 dataloader 状态，optim_step 接收 resume_state 并以 checkpoint键 resume 保存；acknowledge 改名 commit，工作缓冲 INFLIGHT 改为 CLAIMED，体现提交发生在 optimizer/checkpoint 事务边界。旧 RL checkpoint 的 controller/dataloader 状态不会自动迁移。迁移实验须先验证 resume round-trip、pending prompt replay 和指标名；作者CPU503过、两项依赖失败已说明。

### verl 恢复 TensorDict 拼接的首份 metadata 语义

维护方：verl-project 社区；合入 2026-10-30T06:53:24Z；[verl-project/verl #8453](https://github.com/verl-project/verl/pull/8453)，main `8e48d72de4766228c36623c90f9f5c49ee5b45f3`。

tensordict>=0.11 会把不同 rank 的 NonTensorData 拼成 NonTensorStack，metrics 从字典变列表后破坏 SFT validation 和 MFU读取。regular/nested 两条 concat 路径现把首输入的 NonTensorData 放回结果，与旧版本约定一致。它保留首份 metadata，没有替各 rank 聚合指标。建议在实际 tensordict 版本下用不同内存指标和 nested batch 复现，核对后续指标聚合位置。

### verl 新增 Intel XPU 多阶段 rollout/训练镜像

维护方：verl-project 社区；合入 2026-10-30T03:10:12Z；[verl-project/verl #8401](https://github.com/verl-project/verl/pull/8401)，main `321e4e450fcfc47a3491c946cd01dacd9ea7281a`。

Dockerfile 将 driver、运行时、构建和最终层分开，安装 vLLM0.32.0+xpu、torch2.14.0+xpu 与外部 verl-hardware-plugin；保留 Triton launcher编译和视频解码需要的运行库，并检查 transformers等版本相容。作者在B60验证单卡GRPO和双卡FSDP2/xccl，不能外推大规模集群。建议固定 VERL_REF/PLUGIN_REF，先做 sleep恢复、collective和小模型两步回归；插件内部未审计。

### MRV2 speculator 可按声明式 config 扩展

维护方：vllm-project 社区；合入 2026-10-30T17:52:24Z；[vllm-project/vllm #63205](https://github.com/vllm-project/vllm/pull/63205)，main `42379a4c246ce48472f165571d89f962e6c8d97e`。

register_speculative_method将方法名绑定轻量config类，声明hidden-state、lookahead、PCP/watermark等traits及实现qualified name，scheduler无需import具体speculator；显式method与checkpoint声明冲突启动时报错，不再静默改写。custom方法只支持MRV2，options须在supported_options中。作者主要做单元/构造回归，custom plugin及全家acceptance E2E尚未完成。建议最小plugin先验证注册、实例化和失败gate，再做真实draft质量/吞吐。

### gfx942 Kimi-K3 requant 按数值量分块

维护方：vllm-project 社区；合入 2026-10-30T11:07:18Z；[vllm-project/vllm #64004](https://github.com/vllm-project/vllm/pull/64004)，main `8a2a5e4c40f92914fa670d28e348e6ffffe19632`。

MXFP4→group32 int4原按expert数分块，pure TP持所有expert的窄分片导致大量小块。现按展开元素总数限制chunk，预分配输出避免cat双份live set，末chunk回退起点保持同shape以复用allocator，重算重叠expert但字节一致。作者8MI325X warm-page-cache TP8 requant约19分降25秒、启动22.5降3.5分；建议核对权重bytes、峰值及冷磁盘，数字不适用于所有模型/量化格式。

### Intel sparse MLA 补逻辑 top-k ready hook

维护方：vllm-project 社区；合入 2026-10-30T07:26:30Z；[vllm-project/vllm #64065](https://github.com/vllm-project/vllm/pull/64065)，main `fa9356aba6bff6adde15f3e3b273675fe3852741`。

MLAAttention每sparse layer无条件调用record_logical_topk_ready，但XPU实现缺方法导致DSv3.2/GLM5首forward AttributeError。现提供no-op：XPU共享top-k buffer但不参加sparse-MLA index group协议，并把稀疏测试纳IntelCI。作者B60五项kernel/backend测试通过，缺小checkpoint的LLM级验证。建议用真实模型做首次forward及prefix/decode回归，不能把no-op解释为新增跨流同步。

### vLLM EP 容错新增需编排器驱动的 scale-down

维护方：vllm-project 社区；合入 2026-10-30T18:16:02Z；[vllm-project/vllm #46370](https://github.com/vllm-project/vllm/pull/46370)，main `913fccd11847f5b4d4769e3a69c8925f955bc29e`。

永久死 rank 不能靠原 retry 恢复。scale_down 保留原 DP/EP槽位坐标、mask死EP rank，以冗余专家槽恢复缺失逻辑专家并从磁盘reload受影响权重，重建存活rank的Gloo组，全体成功投票后恢复。仅 external LB、MRV2、DP+EP、DeepEP-LL/NIXL-EP及冗余EPLB；不支持PCP和MoE draft，reload依赖默认loader。apply的202仅代表受理，需轮询status；迁移失败需重启，/health仍可能200。建议做死rank、超时、二次scale-down与失败注入。

### 离线 DP 异步调度在无 forward 步补齐 collective

维护方：vllm-project 社区；合入 2026-10-30T23:07:06Z；[vllm-project/vllm #64457](https://github.com/vllm-project/vllm/pull/64457)，main `4c38b653af940bdd8e70e1e92d9ce7b114facb02`。

离线 in-process LLMEngine 在某rank只清理占位或队列时，可能直接进入 has_unfinished_dp 的all-reduce，另rank仍等forward的DP同步而挂起。引擎现记录本步是否execute_model，无forward时执行dummy batch；删除V1 external_launcher的重复补偿。作者GB200 DP2不均匀请求完成，建议同时覆盖MRV1/V2和async scheduling；多进程serve已有自身补偿。

### HiSparse admission 预留尚未分配的 import host pages

维护方：vllm-project 社区；合入 2026-10-30T12:56:06Z；[vllm-project/vllm #62979](https://github.com/vllm-project/vllm/pull/62979)，main `8493b34b7b8738b379c6cc6850e1117bc66814d8`。

等待外部KV导入的请求持有host页且不能被抢占，若admission只看已分配页，后来的请求会挤占它完成全序列所需容量。coordinator现记录已准入import的目标页数并从可用量扣除未分配余额，allocate/free持续更新；spec步额外计入1+draft+lookahead token。建议用多个同时import、失败释放及spec边界测容量，不能把GPU pool空闲量当host页无竞争的证明。

### KV offload 事件保留前缀顺序并描述 recurrent checkpoint

维护方：vllm-project 社区；合入 2026-10-30T12:28:33Z；[vllm-project/vllm #63249](https://github.com/vllm-project/vllm/pull/63249)，main `dfe4d99e42fa7c49426fe3fe1ad134eaaf544765`。
维护方：vllm-project 社区；合入 2026-10-30T09:24:26Z；[vllm-project/vllm #62613](https://github.com/vllm-project/vllm/pull/62613)，main `fac2ce2eb3633c44a104a66dbb30d0e6c7f4493e`。

store job的keys改成保留首次出现顺序的tuple，避免set导致子块先于父块、路由器丢映射；顺序保证仅限单job。Mamba/线性attention组的自描述事件只公告chunk最后hash对应的完整state，带父hash且不复制token_ids；windowed attention仍保留placeholder。建议按所有hybrid组的共同可恢复边界建索引，并测跨tier/job乱序重放；两修复均未保证全局事件拓扑顺序。

### Mooncake Store 将异构 TP 共享扩展到 hybrid KV

维护方：vllm-project 社区；合入 2026-10-30T10:30:28Z；[vllm-project/vllm #54307](https://github.com/vllm-project/vllm/pull/54307)，main `4d1376a4922e28d814d79fae3b16d8de9d2d46cf`。

attention按固定Store TP/chunk切分，Mamba/GDN按完整state及可整除的unit映射，schema指纹隔离布局；本地page大小可以不同，lookup须交汇各组可用边界，writeback按组追踪确认进度。要求共同Store TP/PP、显式相同prefix-match-unit、兼容state布局，PCP/DCP及cross-layer block排除；不相容配置回退隔离namespace。建议跨TP存取同一前缀，覆盖partial-tail、失败PUT重试与完整state恢复；本轮未审计Mooncake内部。

### MoRIIO hybrid READ 接入多个 attention 组和共享组 draft

维护方：vllm-project 社区；合入 2026-10-30T06:10:55Z；[vllm-project/vllm #60867](https://github.com/vllm-project/vllm/pull/60867)，main `1353e1926f94c774f2f8b0fba90af5f980855b57`。

GLM-5.3-Flash同时有MLA/indexer、kpool环与KDA，原单attention组限制及Uniform包装误分类会漏传state。READ payload现按各attention组对齐、逐层路由，并以组内所有layer spec判定Mamba；允许与target共享attention组的MTP/DSpark。CircularBuffer仅hybrid READ放行；WRITE仍保留原DSpark限制。建议分别核对full-local-hit仍需传的scratch组、KV zeroing排除和recurrent state；不泛称WRITE也支持所有draft。

### 多模块 MTP 与 PP structured-output 保留正确 draft

维护方：vllm-project 社区；合入 2026-10-30T23:08:29Z；[vllm-project/vllm #62111](https://github.com/vllm-project/vllm/pull/62111)，main `dd5f5eebd7a004b7366e38927417fb0748bcbbd1`。
维护方：vllm-project 社区；合入 2026-10-30T02:53:31Z；[vllm-project/vllm #63219](https://github.com/vllm-project/vllm/pull/63219)，main `322f90a2a356a83276bc07236e0e620a4e4fc93a`。

Qwen3.5/Next多MTP模块现按spec_step_idx选模块，MRV2避免丢guard的单次compile固定到模块0；已发布单模块checkpoint行为保持。另将PP下draft交接修到请求级保留/丢弃生命周期，撤回仅mask尾部draft的性能退化补丁。建议用合成双模块和PP2混合grammar请求复现，检查结构合法性及acceptance；index-share、MiniMax专用组合仍受选择gate限制。

### DSpark Markov 按 batch owner 分片，greedy argmax 分词表归约

维护方：vllm-project 社区；合入 2026-10-30T17:23:16Z；[vllm-project/vllm #63886](https://github.com/vllm-project/vllm/pull/63886)，main `5423fb8b93a539b24e7f80075d132d4f4dcab95b`。
维护方：vllm-project 社区；合入 2026-10-30T20:20:44Z；[vllm-project/vllm #57770](https://github.com/vllm-project/vllm/pull/57770)，main `c8dbb1cb262ee5663f8983ef5536e1d57e1b47e1`。

batch-sharded路径以request i%TP归属owner：本地vocab投影经all-to-all转为owner的完整词表，owner运行Markov链，再all-gather draft/confidence恢复批次顺序；复用TP PyNCCL communicator，固定split可capture。greedy路径另用4096词块两阶段argmax，保留首索引tie和int64输出。最终代码把NaN当-inf，与torch.argmax遇NaN行为不同，正文同语义说法不成立。建议测TP padding、AV及非有限logit；probabilistic路径不使用新greedy归约。

### RL reload 完成后统一重建模型级派生状态

维护方：vllm-project 社区；合入 2026-10-30T22:53:16Z；[vllm-project/vllm #64357](https://github.com/vllm-project/vllm/pull/64357)，main `14a5bb4adcc86e4c953236e048bae28512c92427`。

NCCL/IPC/M2N/RDT等按bucket喂权重，原model post-load hook仅冷加载执行，导致融合buffer或广播副本过期，部分模型又在每bucket上过早处理。新增finalize_model_weights，在层级finalize之后每次完整load/reload执行一次；sparse-NCCL原地patch也于finish调用。建议更新DeepSeek、DSpark/DFlash、Muse等模型后对比全新server权重与输出，并验证缺少可选head的bucket不会错误清空状态；大型模型逐项数值回归仍待做。

### level-1 sleep 以精确大小 host backup 复用分配

维护方：vllm-project 社区；合入 2026-10-30T09:44:54Z；[vllm-project/vllm #64630](https://github.com/vllm-project/vllm/pull/64630)，main `deacc3670153845db2f013ea24085d6eaf338874`。

CUDA/ROCm backup改为mmap+可选cudaHostRegister，避开CachingHostAllocator按2次幂取整；buffer随allocation首次建立、后续sleep复用，discard/level2/free才释放，并统一状态转换防止恢复旧backup。注册失败可回退pageable；XPU部分复用生命周期但仍由torch分配。建议测多轮sleep/wake、discard后再sleep与host RSS，不能把精确CUDA backup结论直接套给XPU allocator。

### NCCL 权重同步结束释放 receive 缓存

维护方：vllm-project 社区；合入 2026-10-30T22:41:05Z；[vllm-project/vllm #64691](https://github.com/vllm-project/vllm/pull/64691)，main `be712f5601dc7eebf43aa42848be6a84256dcada`。

finish_weight_update末尾empty_cache释放闲置receive和post-load临时块，让非torch分配者可用；不改变已分配权重或KV容量。作者GB200/Qwen3-30B-A3B-FP8 TP2每rank空闲增加3.7–4.4GiB，但下次同步需重分配，中位同步约+7%，仅一组测量。建议对照reserved/allocated/device-free和下一轮同步延迟，不能将释放reserved算成模型显存下降。

### compile cache 可选设备无关模式，depyf 接受新 Inductor 参数

维护方：vllm-project 社区；合入 2026-10-30T08:01:16Z；[vllm-project/vllm #63247](https://github.com/vllm-project/vllm/pull/63247)，main `a20280f30e3169d33a7e21d1d117e9719bace00b`。
维护方：vllm-project 社区；合入 2026-10-30T20:04:41Z；[vllm-project/vllm #60873](https://github.com/vllm-project/vllm/pull/60873)，main `2fab35893b1a0ef78cb2392363cc7de3b00cb425`。

torch>=2.14的CUDA/ROCm/XPU可显式开VLLM_COMPILE_ON_ONE_RANK，使缓存运行时解析device index，并将开关纳入cache key；最终代码默认关闭，与PR正文默认开不符，graph partition/debug dump有兼容限制。另按depyf hook签名有条件补kwargs转发，修debug_dump_path autotune的set_sys_modules错误。建议分别做换device-id的warm-cache启动和禁cache的debug冷编译，勿用命中缓存证明debug编译成功。

### 长 token/大 GEMM 地址在乘法前提升到64位

维护方：vllm-project 社区；合入 2026-10-30T20:27:51Z；[vllm-project/vllm #63761](https://github.com/vllm-project/vllm/pull/63761)，main `2f6f7861ba075b72b530b3d0b7889fc55b500ae4`。
维护方：vllm-project 社区；合入 2026-10-30T01:34:22Z；[vllm-project/vllm #63054](https://github.com/vllm-project/vllm/pull/63054)，main `f81a414faed63b576768487236f98ea5e655fc57`。

Qwen3.5的QK/RoPE、GDN/checkpoint/三角求逆等在token×stride>=2^31时可溢出，现把基址或row索引在乘法前转int64；单在乘后cast无法恢复已溢出的值。ROCm W4A16同样修activation/output行偏移，权重offset保留原类型。建议按实际stride构造跨界prefill、多sequence后段及AWQ/GPTQ回归，不只增加单请求max-model-len。

### AITER tuned GEMM 越过地址界限时回退

维护方：vllm-project 社区；合入 2026-10-30T05:16:41Z；[vllm-project/vllm #64567](https://github.com/vllm-project/vllm/pull/64567)，main `6d018934838a0c8c6beed244aa1ca327e1ed7182`。

对M×N>=2^31或A/B跨度>=4GiB的非量化GEMM避开tuned AITER，走F.linear/hipBLASLt，防止gfx950 FlyDSL地址溢出导致MiniMax多模态profile崩溃。源端只提供guard与fallback，外部AITER修复没有本轮验证；gate实际写在共享ROCm调度。建议用边界两侧尺寸、bias和视觉profile验证dispatch和输出，勿仅凭AITER版本升高删除保护。

### Qwen NVIDIA norm/gate 融量化，NVFP4 PDL按规模启用

维护方：vllm-project 社区；合入 2026-10-30T23:04:16Z；[vllm-project/vllm #64509](https://github.com/vllm-project/vllm/pull/64509)，main `6c19299fca37eabb719002b0249501a89430c3e6`。
维护方：vllm-project 社区；合入 2026-10-30T08:23:14Z；[vllm-project/vllm #64494](https://github.com/vllm-project/vllm/pull/64494)，main `8b10bda57ceab3a741d2fb3718af48cef691023e`。

Qwen3.5 NVIDIA将Gemma add-RMSNorm、GDN norm或sigmoid gate的结果直接生成consumer所需static FP8/NVFP4 activation，减少独立quant kernel；block/dynamic/per-token FP8等不匹配格式回退。SiLU-mul+NVFP4另以M×N/16<=SM数×4096门槛开PDL，避免大prefill回退。作者B300限定checkpoint/并发报告TPOT收益，建议逐consumer检查quant key和bitwise scale，再做端到端，而非把两项百分比相加。

### persistent DSA top-k 直接读取 BF16 logits

维护方：vllm-project 社区；合入 2026-10-30T22:43:09Z；[vllm-project/vllm #64138](https://github.com/vllm-project/vllm/pull/64138)，main `e31cf3dec7cf14369b0c7a8447cdfb8b387af6e0`。

SM100无DCP decode若选persistent，可省BF16→FP32整张logits物化，kernel加载后再扩FP32做原binning/refinement；不是减少DeepGEMM算分或新增top-k算法。cooperative、prefill和DCP保留旧边界，equal-score索引顺序与生成token一致性未保证。建议记录auto实际backend、峰值临时显存、选中score和tie，单后处理微基准不代表整模同比提速。

### MegaMoE preparation 按 token 数扩大 row tile

维护方：vllm-project 社区；合入 2026-10-30T07:32:32Z；[vllm-project/vllm #64467](https://github.com/vllm-project/vllm/pull/64467)，main `2bfb8f1674640c4cf91f525111a7757c9c9e2940`。

prepare_megamoe_inputs在>=1024 token用32行tile，64–1023仍8行，小批1行，保留MXFP8字节和top-k/scale布局。作者GB200 preparation为1.3–1.6倍，DeepSeek-V4.1-Flash DEP2整forward仅约0.3% GPU时间节省；精度表同时应用另一PR。建议用实际prefill桶做bitwise输出和trace验证，勿拿kernel倍率预测整模吞吐。

### Muse Spark 新CUDA路由及分片投影融合

维护方：vllm-project 社区；合入 2026-10-30T01:18:45Z；[vllm-project/vllm #64553](https://github.com/vllm-project/vllm/pull/64553)，main `36ed4327d2caeb38543b75b80db2a291e88883bc`。
维护方：vllm-project 社区；合入 2026-10-30T16:06:04Z；[vllm-project/vllm #64638](https://github.com/vllm-project/vllm/pull/64638)，main `becbaea9f58f5a492f36d4b932fbc0484fa68198`。

CUDA routing接入固定256-expert几何及FP32 weights、int32/int64 ids，Blackwell可选原生warp-max，保留padding处理。hybrid MoE的pooler列分片后可将all-gather与norm/quant合并，并复用buffer的int32 id视图喂TRTLLM路径；SP状态与backend决定gate。PR正文缺完整验证说明，本轮只核实dispatch/diff。建议先用路由reference和padding/tie回归，再测NVFP4/SP组合；不据标题承诺吞吐。

### 非 Triton CPU 默认接入独立 MRV2

维护方：vllm-project 社区；合入 2026-10-30T22:52:04Z；[vllm-project/vllm #61995](https://github.com/vllm-project/vllm/pull/61995)，main `040abb262c5b09b8325abe0379ff70246fd92bba`。

Arm、IBM Z/PowerPC和RISC-V等无成熟Triton的CPU路径新增standalone runner，用torch/CPU C++算子替换GPU Triton，并将batch、sampler、block table等放在cpu目录独立维护。x86现有Triton路径不在此开关范围，spec decode暂未支持。建议在目标架构验证多模态、PP、LoRA和structured output的实际支持；大量移植模块只做关键路径审阅，不宣称等同全部GPU MRV2功能。

### Step-5-Preview 扩展到 ROCm 的 sparse-attention 生命周期

维护方：vllm-project 社区；合入 2026-10-31T00:31:07Z；[vllm-project/vllm #61837](https://github.com/vllm-project/vllm/pull/61837)，main `17550580e932f15e772185847382d3ae4211f5ec`。

新增AMD实现并抽出共用CSA summary/tail/indexer逻辑，主paged KV走AITER unified attention，支持prefill/chunk/decode/MTP及图像。主要验证gfx942 TP8，gfx950为次级；FP8 KV须block32。NIXL/Mooncake增加compressed page split guard，不能拆summary page。建议固定checkpoint做text/image、prefix和PD回归；外部AITER内核及早期约2.3倍MTP测量未本轮复现。

### Qwen4Exp PLE 表可从 checkpoint 按需读盘

维护方：vllm-project 社区；合入 2026-10-30T13:28:30Z；[vllm-project/vllm #59958](https://github.com/vllm-project/vllm/pull/59958)，main `b44a48cef1c08cbdc30a0953c724f18cd434e612`。

disk_offload让PLE n-gram表留在原checkpoint，host按步读取所需行后送固定device buffer，CUDA graph无需读盘节点；BF16/FP8/NVFP4复用既有GPU解码。仅CUDA Qwen4Exp，不支持elastic EP，优先于cpu_offload。作者GB10统一内存环境少27GiB常驻表，decode约±3.5%但冷TTFT和网络存储成本更高。建议先在本地NVMe测冷热page cache及压力回收，不能把OS可回收cache等同零host占用。

### Fast Start preload 任一 rank 退出即清理全组

维护方：vllm-project 社区；合入 2026-10-30T07:23:24Z；[vllm-project/vllm #63692](https://github.com/vllm-project/vllm/pull/63692)，main `19e24f0ca807e2938a3373674bc4b6cbe6994685`。

launcher不再按顺序join而漏看后rank退出，启动和READY后均监视全部process sentinel；先保留原失败status，再terminate/reap兄弟进程，统一10秒grace后kill，并关闭health端点。正文保留过时OPEN描述，当前metadata和main commit确认已交付。建议在健康前后分别杀不同rank、测SIGTERM及清理超时，避免只看rank0健康。

### Hub/分片加载保留模型身份并减少无用下载

维护方：vllm-project 社区；合入 2026-10-30T20:31:18Z；[vllm-project/vllm #64779](https://github.com/vllm-project/vllm/pull/64779)，main `644a9f5d278cc24cea5ad5a7990f397df0fa26b7`。
维护方：vllm-project 社区；合入 2026-10-30T20:31:41Z；[vllm-project/vllm #63482](https://github.com/vllm-project/vllm/pull/63482)，main `212f3666a84e3943b3b1b1658d8162d8ba5baef5`。
维护方：vllm-project 社区；合入 2026-10-30T20:14:07Z；[vllm-project/vllm #63922](https://github.com/vllm-project/vllm/pull/63922)，main `ed14230f4fc4d889a6a60c187a4b6449ef9fd77c`。
维护方：vllm-project 社区；合入 2026-10-30T19:18:39Z；[vllm-project/vllm #61380](https://github.com/vllm-project/vllm/pull/61380)，main `c9af9acc6ac22f6351e214eb0d7d9a7a26dfa870`。

要求huggingface_hub>=2、tokenizers>=0.23.3/timm>=1.0.30；加载local tokenizer snapshot时复制config并还原原Hub ID，修DeepSeek-Coder路由丢空格。snapshot过滤补齐ot/tflite/CoreML/NeMo/NumPy/Paddle权重，sharded loader先_prepare_weights解析Hub路径并尊重revision。建议锁依赖，对照token ID、下载文件清单及model_weights覆盖；Hub跨仓Xet去重为上游契约，本轮未实现审计或测量。

### DiffusionGemma 默认 sampler 分tile生成 logits

维护方：vllm-project 社区；合入 2026-10-30T07:10:59Z；[vllm-project/vllm #51953](https://github.com/vllm-project/vllm/pull/51953)，main `89a9a03710810bea21cc42e2c52a85526a606655`。

并发下整批logits使首decode OOM，现sampler每tile调用投影，并在profile_run预留transient，缓解KV预算挤满显存后的运行时峰值。作者RTX5090/NVFP4-26B并发8将稳定KV容量由508k升至689k，0.92利用率仍在prompt-logprobs启动OOM；tile改变数值归约，不是bitwise等价。建议同时测decode并发峰值、profile和prompt-logprobs，勿只测启动可用。

### GLM-5.3 的 DFlash 可沿用 FP8 KV 配置

维护方：vllm-project 社区；合入 2026-10-30T15:22:11Z；[vllm-project/vllm #64612](https://github.com/vllm-project/vllm/pull/64612)，main `d15be0856ded087df49b67a82efaa75243b98251`。

模型配置的draft dtype排除规则现把dflash与mtp/dspark一同保留，避免强制回BF16；PR题述每rank省2.85GB对应其GLM-5.3 TP4/特定DFlash设置。实际diff仅改变一处method gate，不构成所有DFlash模型都支持FP8 KV的证明。建议核对draft实际KV dtype、scale、acceptance和任务准确率后再采用内存数字。

### GLM-5.3-Flash EPD 可传 media metadata 而不重做视觉预处理

维护方：vllm-project 社区；合入 2026-10-30T02:42:01Z；[vllm-project/vllm #64071](https://github.com/vllm-project/vllm/pull/64071)，main `a17e79abde45715e79a00deb99ad34be49162231`。

encoder发布image_grid_thw，video另带每temporal patch时间戳；consumer凭元数据重建placeholder并由EC connector接收embedding，检查grid与embedding row匹配。作者8H200 image/video greedy小样本与raw路径一致，约1.14倍是image吞吐，不能套给video。建议验证混合media、metadata-only与显式embedding错误尺寸，并确认所用的是Python EPD路径。

### 多模态 processor LRU 在单请求内保留未来 hit

维护方：vllm-project 社区；合入 2026-10-30T09:21:14Z；[vllm-project/vllm #64495](https://github.com/vllm-project/vllm/pull/64495)，main `8472c01d81cbaeeccb3f18407282f23f3f1d101e`。

请求[new C,cached A,cached B]超过两图cache时，插入C可能先逐出稍后要用的A。merge前以不改LRU顺序/统计的peek保留本请求payload和prompt update，随后evict仍可继续构造。只覆盖本地processor-only payload，cache预算不含请求瞬时引用。建议用重复图、多图顺序和超过预算请求对比uncached token/placeholder，另看host瞬时峰值。

### PyNvVideoCodec caps 以设备和格式缓存

维护方：vllm-project 社区；合入 2026-10-30T13:45:48Z；[vllm-project/vllm #64618](https://github.com/vllm-project/vllm/pull/64618)，main `6a95c89c371939997d9ce3b616360b529760f0bc`。

GetDecoderCaps约180ms持GIL，原每视频请求查询可冻结API事件循环。现以(device,codec,chroma,bit-depth及模块)的64项LRU缓存结果，不改变解码数据；返回dict须视为只读。作者GB300短clip收益较大，长clip受decode本身限制。建议先区分首次caps miss与后续hit，做并发text/video混合负载，不能把cache收益计为视频decoder算子提速。

### QSA 去掉 FlashAttention 类依赖，OOT layer 可按实际路径注册

维护方：vllm-project 社区；合入 2026-10-30T06:48:19Z；[vllm-project/vllm #64110](https://github.com/vllm-project/vllm/pull/64110)，main `d492ee786266e17b984a5772715186a416425e17`。
维护方：vllm-project 社区；合入 2026-10-30T11:16:03Z；[vllm-project/vllm #60218](https://github.com/vllm-project/vllm/pull/60218)，main `24420b7a341af37a9bc3186e6fc6062512e0aa58`。

Qwen4Exp QSA改用自有attention backend/metadata，保留已有kernel、FP8/Octave cache与builder校验，不声称数学或吞吐改变。另公开hw_agnostic.resolve，插件可对Transformers backend实际构造的类register_oot；缺port回退in-tree，已有port的依赖import错误继续抛出，native模型及部分norm子类仍有覆盖边界。建议分别验证无FlashAttention导入和OOT override生效，不把注册成功当所有模型替换成功。

### ROCm SWA metadata 单launch与 MegaAttn int64 位置接线

维护方：vllm-project 社区；合入 2026-10-30T02:53:35Z；[vllm-project/vllm #63885](https://github.com/vllm-project/vllm/pull/63885)，main `e49ea84e929e3d48a2bc500454dfcd4377f9a722`。
维护方：vllm-project 社区；合入 2026-10-30T07:32:47Z；[vllm-project/vllm #61971](https://github.com/vllm-project/vllm/pull/61971)，main `53624fc952d1a98af0ac77f51ab6fe0de9a669f4`。

DSv4.1 --monokernel在gfx950也可用小decode<=32行的Triton SWA metadata单launch，其他操作仍按原gfx942 gate。NVIDIA FlashMLA固定到d1a1c92并直接传int64 positions，去掉每层cast；PR性能表来自较早共享一次cast版本，不能当最终revision实测。建议各自回看kernel count、metadata/position dtype与AV开关；外部FlashMLA内部未本轮重审。

### HTTP下载与 run-batch 前端修补计时/错误边界

维护方：vllm-project 社区；合入 2026-10-30T07:41:35Z；[vllm-project/vllm #62674](https://github.com/vllm-project/vllm/pull/62674)，main `8b9a4341dbad4979a03e1776fe86e33944a72963`。
维护方：vllm-project 社区；合入 2026-10-30T07:04:40Z；[vllm-project/vllm #61973](https://github.com/vllm-project/vllm/pull/61973)，main `e7aad419dee663aa19ed199715d174cb94477525`。
维护方：vllm-project 社区；合入 2026-10-30T09:09:55Z；[vllm-project/vllm #63005](https://github.com/vllm-project/vllm/pull/63005)，main `96c80f8af766a1d206d92d2fb01bb92234b9b443`。

不完整HTTP body的ChunkedEncodingError/ClientPayloadError现进入既有次数/退避重试，仍执行下载size limit。audio handler调用移出准备阶段try，让engine错误保持500、可重试503，下载/校验仍400；TypeAdapter改模块级复用，省逐行schema重建。建议分别做截断图片恢复、audio状态码和大batch CPU基准；约11倍为指定解析流程，不是推理吞吐。

### GLM-4.7 非 strict 工具参数避免 grammar 锁死

维护方：vllm-project 社区；合入 2026-10-30T01:25:02Z；[vllm-project/vllm #64615](https://github.com/vllm-project/vllm/pull/64615)，main `dc420193b71d3a2e4c8d6ee08ef3eec98af51933`。

array/object（及含这些类型的union）改为到</arg_value>截止的原文，避免Windows路径等非JSON文本被强制到近零概率escape而循环；integer/number/boolean/null和enum继续约束。只影响非strict结构tag适用的glm47/ling3请求。建议覆盖raw数组/对象、反斜杠和strict模式，不能把此放宽当作保证JSON合法。

### Responses benchmark 将工具 delta 计入生成

维护方：vllm-project 社区；合入 2026-10-30T07:15:51Z；[vllm-project/vllm #62753](https://github.com/vllm-project/vllm/pull/62753)，main `c05e9edb788ec87a65ca366a8fb45771d8002062`。
维护方：vllm-project 社区；合入 2026-10-30T02:35:29Z；[vllm-project/vllm #64557](https://github.com/vllm-project/vllm/pull/64557)，main `32a73034d511ff2f695036ffa3742fb94f7466c4`。

Python和Rust基准客户端均把function/custom-tool/MCP/code-interpreter的argument/code delta纳入TTFT/ITL，修HTTP200但tool-only被判失败；仍只把output_text写入generated_text，token总数按usage。建议重跑工具工作负载并标注基准客户端revision，旧新成功率或TTFT不可直接比较为模型能力进步。

### Rust frontend 支持跨 launcher 的 DP rendezvous 与局部 rank

维护方：vllm-project 社区；合入 2026-10-30T07:49:15Z；[vllm-project/vllm #64449](https://github.com/vllm-project/vllm/pull/64449)，main `aa85ff7f13c6822bbfd183f8c4193bc12cd3e6a6`。
维护方：vllm-project 社区；合入 2026-10-30T07:49:16Z；[vllm-project/vllm #64450](https://github.com/vllm-project/vllm/pull/64450)，main `c20fc8a8570b5cfda377cf9a7ce5f10056bd3eaf`；原目标 bz/rust-dp-rendezvous-init，已验证集成 main。
维护方：vllm-project 社区；合入 2026-10-30T07:49:17Z；[vllm-project/vllm #64451](https://github.com/vllm-project/vllm/pull/64451)，main `85591000403fc3bc3b4d0303e3a97bedab007763`；原目标 bz/rust-handshake-owner-dp-range，已验证集成 main。

握手owner将统一DP地址/端口放INIT，rank0握手所有rank但只注册自己的区间，其他frontend跟随coordinator wave；CLI接入hybrid/external LB与全局rank路由。#64450/#64451合入目标是栈分支，但merge SHA已出现在本轮main提交集，计为已集成main而非把base字段等同交付边界。作者单GB200节点多launcher验证，跨物理节点仍须自己测；建议检查错rank/配置hash拒绝与只有单frontend有流量时全组推进。

### Rust 请求禁词与 template 检查，GLM native rendering

维护方：vllm-project 社区；合入 2026-10-30T08:33:48Z；[vllm-project/vllm #48951](https://github.com/vllm-project/vllm/pull/48951)，main `8bad43e2c04bee07bf76366b5d89c39a3da03a73`。
维护方：vllm-project 社区；合入 2026-10-30T07:19:26Z；[vllm-project/vllm #64274](https://github.com/vllm-project/vllm/pull/64274)，main `a70e48deba60c53fd1c171bdb3d8adac1e0c070d`。
维护方：vllm-project 社区；合入 2026-10-30T06:35:44Z；[vllm-project/vllm #62424](https://github.com/vllm-project/vllm/pull/62424)，main `d14f72c7fe1316ae9d186f1eed554e8099dba705`。

completions现透传bad_words到sampling，修请求成功却禁词被静默丢弃。native renderer遇chat_template明确拒绝而不忽略；GLM5Next原生编码器按added vocabulary写control token，GLM5.3-Flash可auto，GLM5.3须显式tokenizer-mode。建议逐route校验禁词与template错误，比较工具/思考模式的token序列；参数接受不代表限制已传到engine。

### Rust 返回真实 weight-version spans 与 prompt logprobs

维护方：vllm-project 社区；合入 2026-10-30T08:43:19Z；[vllm-project/vllm #64218](https://github.com/vllm-project/vllm/pull/64218)，main `52737e03b85a31a57e524a0d070823a0e8318777`。
维护方：vllm-project 社区；合入 2026-10-30T08:44:16Z；[vllm-project/vllm #64220](https://github.com/vllm-project/vllm/pull/64220)，main `d8a7945791c66440cdc4024112f7ff27d5cfe3e1`。
维护方：vllm-project 社区；合入 2026-10-30T10:14:18Z；[vllm-project/vllm #64219](https://github.com/vllm-project/vllm/pull/64219)，main `c817cc0f8be5f41477ca1944129d36c93724eba3`。

weight_versions为最终choice/末chunk的token-index spans并遵守stop截断，使RL权重更新期间的输出来源可追踪。echo+logprobs现请求并返回prompt logprobs，修stream路径遗漏；completion_tokens_details只在chat有reasoning parser可计数时存在，其他返回null，避免把未知reasoning计数写成0。建议固定流/非流契约，比较stop截断、prompt offsets及更新跨越的token区间。

### Rust frontend 不再静默丢弃 UMBP transfer 统计

维护方：vllm-project 社区；合入 2026-10-30T08:42:21Z；[vllm-project/vllm #64512](https://github.com/vllm-project/vllm/pull/64512)，main `2b313cd97bf6052f213be407ca32fef1c2f991b6`。

UMBPStoreConnector已能产出统计，但Rust原解成unknown map后丢弃，导致submitted/completed/failed/bytes不可观测。现按load/store解析bare及MultiConnector子payload，四类counter按model/engine/operation在interval间累加，首次报告才出现series。bytes代表成功transfer的range字节，不是prefix-hit率或提交即完成。建议用两段stats和注入失败核对sum与labels，再以它判断实际传输瓶颈。

### SGLang TP sampling renorm 按最终代码选择确定性路径

维护方：sgl-project 社区；合入 2026-10-31T00:07:32Z；[sgl-project/sglang #38565](https://github.com/sgl-project/sglang/pull/38565)，main `a2cafa56766cdd7a8c51f04008bdc0fbb1b64ebb`。

共享sampling coin若落在跨rank float-atomic归约差异之间，rank会提交不同token并最终collective挂起。普通sampler和spec verify现统一renorm包装，最终规则是tp_size>1（含CP）或deterministic mode即请求确定性FlashInfer变体；并非正文的attn_tp_size，最终代码也没有正文所述env override。缺该API时退回平台kernel，不能推成所有旧依赖均受保证。建议在真实TP/DFlash/DSpark场景对比probs/token与hang，grammar既有同步不因此取消。

### 多 graph variant 保活 attention metadata 与 SP workspace

维护方：sgl-project 社区；合入 2026-10-30T19:32:30Z；[sgl-project/sglang #46720](https://github.com/sgl-project/sglang/pull/46720)，main `8ecabd3c1e1227ca91b0bbb48d75cf33b4626a73`。

同batch size捕获LoRA/base等多variant时，原按batch重建wrapper或workspace会释放旧graph仍引用的buffer。FlashInfer/TRTLLM共享并保留capture元数据、SP保留旧workspace generations；LoRA dual graph改为SGLANG_ENABLE_LORA_DUAL_GRAPH显式开启、默认关。建议对同一batch捕两图交替replay并扰动allocator，不能用单图测试或某GPU暂未出错证明生命周期安全。

### CUDA graph executable 首次使用先在私有流 upload

维护方：sgl-project 社区；合入 2026-10-30T23:35:20Z；[sgl-project/sglang #47295](https://github.com/sgl-project/sglang/pull/47295)，main `be556ff0e9ce48716149e824d054ceda555b38cb`。

首次cudaGraphLaunch的隐式upload可能等忙碌launch stream清空并阻塞host。dedup registry现每executable只在首次replay前以私有nonblocking stream upload，记录event再让launch stream等待，close统一销毁；不把全部upload移到startup。建议看首次命中各shape时host stall、后续replay和共享executable仅一次upload，不误算为steady-state kernel提速。

### HiCache/KDA 图内等待绑定当批真实 load event

维护方：sgl-project 社区；合入 2026-10-30T06:14:20Z；[sgl-project/sglang #45839](https://github.com/sgl-project/sglang/pull/45839)，main `b572b8ea827fc236786cf64d025337f60895d16d`。
维护方：sgl-project 社区；合入 2026-10-30T23:13:18Z；[sgl-project/sglang #46884](https://github.com/sgl-project/sglang/pull/46884)，main `9f445266c2556a377b2aa16410b77f0c0ff92f78`。

KDA breakable graph跳过Python pool accessor，现capture每层wait并在replay前用cudaGraphExecEventWaitNodeSetEvent绑当批H2D事件，无load时用completed placeholder。dedup/full graph、非CUDA与其他不支持路径保留保守等待。另HiCache controller的offload已各自fence forward时可早发布prefill shared-read-done；LMCache、无fence decode-offload仍等整forward。read-done不是KV-ready，建议延迟H2D/D2H做数值及竞态注入。

### PD 地址发布与尾块 state 索引读回避免阻塞 scheduler

维护方：sgl-project 社区；合入 2026-10-30T15:16:34Z；[sgl-project/sglang #45328](https://github.com/sgl-project/sglang/pull/45328)，main `d51a1e5d971892d2406441bbdd3d78a9f1c100e6`。
维护方：sgl-project 社区；合入 2026-10-30T23:41:39Z；[sgl-project/sglang #47177](https://github.com/sgl-project/sglang/pull/47177)，main `3e83f3a71a15b3bac41d040afae290c745eadc9d`。

decode在overlap+WAR等gate下把已分配目标地址延至下一迭代，从独立流回读并等allocation event后再publish，转移等待的执行位置；PP、HiSparse、staging等不适用组合保留旧路。prefill最后chunk的SWA/full-KV页索引也纳入host tensor异步D2H，send worker等同一indices_event才消费；不支持该能力或需host过滤CP继续同步。建议测abort/释放、最后chunk和不同sender，不能把队列入队当传输完成。

### 无法重建 KV 的 retract 改为可见 abort

维护方：sgl-project 社区；合入 2026-10-30T23:19:32Z；[sgl-project/sglang #47521](https://github.com/sgl-project/sglang/pull/47521)，main `c105a91b436c3af8dea28bbc829462db6eac38b0`。

PD rebootstrap只携token IDs，multimodal/input_embeds/position override不能复建原KV；local re-prefill的input_embeds在已发输出后重启也会拼接错流。release_req现统一判断可恢复性，pause/retract对不可恢复请求503 abort，未发输出的local input_embeds仍可重启。/detokenize另拒绝越词表ID。建议覆盖stream/force flush、embed override和PD角色；这是正确性拒绝边界，客户端须按可重试失败处理。

### PD rebootstrap 保留已输出 token，handoff 统一提交

维护方：sgl-project 社区；合入 2026-10-30T03:15:22Z；[sgl-project/sglang #47366](https://github.com/sgl-project/sglang/pull/47366)，main `03aba9e6479788470750b884333f20cc270e06f6`。
维护方：sgl-project 社区；合入 2026-10-31T00:18:14Z；[sgl-project/sglang #47544](https://github.com/sgl-project/sglang/pull/47544)，main `4df3895bddc7d543e9174efcedab84793298f6d4`。

rebootstrap不再pop末输出再append，payload仍重算prompt+output[:-1]，transfer commit跳过已交付token/logprob/mask，避免completion_tokens倒退或重复流。新请求handoff改走共用commit_handoff_token及单token logprob helper，同时更新reasoning/mask；未请求top-k不再发送首chunk的[null]空行。建议在retract窗口持续消费stream、对齐token/logprob和reasoning计数，不能只核对最终拼接文本。

### scheduler abort 在出队时定终态并退休 beam 组

维护方：sgl-project 社区；合入 2026-10-30T21:09:44Z；[sgl-project/sglang #43897](https://github.com/sgl-project/sglang/pull/43897)，main `7146002be8a7c65a482c27778eadb21fcf34034d`。
维护方：sgl-project 社区；合入 2026-10-30T08:09:06Z；[sgl-project/sglang #47381](https://github.com/sgl-project/sglang/pull/47381)，main `f96a6e42f6048fa5175e3de603d146daaaa95a29`。

waiting/dLLM队列pop后立即finish_outside_step，后续同批abort不再对已释放请求写悬空to_finish；释放时可正确归还PP ticket或session slot。chunked-prefill beam leader abort另调用retire_group，避免无member却永久计为live组。建议同intake重复abort、chunk中断和session借用slot回归，观察资源计数归零而非只看HTTP返回。

### DSpark 越 capture grid 回 eager，DP verify 不走空闲 tail replay

维护方：sgl-project 社区；合入 2026-10-30T20:21:13Z；[sgl-project/sglang #47333](https://github.com/sgl-project/sglang/pull/47333)，main `a8d944f5e4a027a1e069b19e115919fbdf793510`。
维护方：sgl-project 社区；合入 2026-10-30T06:42:25Z；[sgl-project/sglang #46676](https://github.com/sgl-project/sglang/pull/46676)，main `f362716a21eca74c060f36854a88d2bd3c817481`。

ragged verify先判超grid再round_up，oversized tier返回eager，修confidence scheduling且max-running大于graph容量时scheduler异常。DeepSeek-V4.1 DP spec/prefill协调时，verify rank虽无prefill tail却有真实verify行，现整步避开decoder-tail idle graph，防止以他rank的tail/stale logits替换输出。建议覆盖DP有/无prefill尾、verify行多于桶和AV配置，确认fallback与acceptance。

### 移除 PD multiplexing 及专用 split-prefill 路径

维护方：sgl-project 社区；合入 2026-10-30T22:56:51Z；[sgl-project/sglang #43898](https://github.com/sgl-project/sglang/pull/43898)，main `a076243ceacf376b9632d8da37dc8e17005595bd`。

--enable-pdmux、--pdmux-config-path、--sm-group-num及其重复TP组、per-stream graph、green-context extension和split-prefill API被删除；维护者指出用户和CI覆盖不足。现有不启用pdmux部署不涉及该入口，旧启动参数将直接无效。建议升级前扫描命令/配置并移除这些flags；不把删除理解为普通PD disaggregation也被移除。

### ROCm MoE 保留 layout、reduction 与 LoRA 正确性

维护方：sgl-project 社区；合入 2026-10-30T06:20:09Z；[sgl-project/sglang #45486](https://github.com/sgl-project/sglang/pull/45486)，main `196ee6b6115f6c39584d53f53b59c8662d299520`。
维护方：sgl-project 社区；合入 2026-10-30T15:37:36Z；[sgl-project/sglang #44268](https://github.com/sgl-project/sglang/pull/44268)，main `51d5502aff999cf3a80e0c2d211d56523310561c`。

compressed-tensors CHANNEL FP8只在实际AITER runner时shuffle，修显式Triton收到错误layout及非block FP8错误减去128维。fused-moe-sum-all-reduce提前处理平台共享输出，非CUDA不再从未写buffer重做combine；有after_down LoRA hook时关闭fusion，空hook仍可开。建议用1/33token、flag开关及LoRA delta对照unfused，不能把两条不同quant/kernel分支合成通用FP8修复。

### ROCm attention 按 layer scale，Quark shared fusion 检查每层

维护方：sgl-project 社区；合入 2026-10-30T06:20:45Z；[sgl-project/sglang #46833](https://github.com/sgl-project/sglang/pull/46833)，main `85af8131e145bc843d2bb8415db0662f87253a2e`。
维护方：sgl-project 社区；合入 2026-10-30T06:21:05Z；[sgl-project/sglang #46800](https://github.com/sgl-project/sglang/pull/46800)，main `16a3bf3d9db77f9f6804b3fd4515a44a061a6bfb`。

AITER NHD/5D prefill及ragged decode现传layer.scaling，避免Granite等非默认softmax比例生成错误。Quark融合gate不再只探dense layer0，而遍历named target MoE层，发现MXFP8 shared与MXFP4 routed不同则拒融合；draft条目单独处理。建议用自定义attention scale、dense-first模型和混合shared精度回归，已有slider/unified attention路径不代表所有site原来都正确。

### 确定性模式检查实际 phase backend 并绕开 gfx950 AITER unquant

维护方：sgl-project 社区；合入 2026-10-30T11:01:12Z；[sgl-project/sglang #44438](https://github.com/sgl-project/sglang/pull/44438)，main `56a8d3a8d76ddb43d4697ae56972294364dbcd94`。
维护方：sgl-project 社区；合入 2026-10-30T06:33:44Z；[sgl-project/sglang #43788](https://github.com/sgl-project/sglang/pull/43788)，main `0fd1e3a799266e5d4b641b20b37b1d034270d49d`。

prefill/decode override可覆盖主attention-backend，现都须位于确定性allowlist；gfx950确定性模式的BF16 linear/MoE改用batch-invariant/Triton，显式AITER MoE冲突会拒绝。量化模型内部BF16层也受影响，但DSv4/GLM仍有FP8/MXFP4等其他非确定性来源。建议对照不同batch composition精确token/hidden-state，不能把单一guard当全模型确定性证明。

### Nemotron-H 显式放行 Blackwell CuTe W4A16

维护方：sgl-project 社区；合入 2026-10-30T12:59:27Z；[sgl-project/sglang #44750](https://github.com/sgl-project/sglang/pull/44750)，main `5d4a6f499e028405ecd4c8362c1e8b2443df1758`。

ModelOpt mixed检查现允许SM100家族+SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16=1的显式flashinfer_cutedsl；auto仍Marlin，A2A必须none。它放行既有BF16 activation/NVFP4 weight实现，不是新的NVFP4 activation路径。作者B200小样本GSM8K差异属噪声，建议固定checkpoint做backend/flag/hardware gate及吞吐回归。

### GLM CP prefill 配合 TP decode 恢复 fusion 与 graph 容量

维护方：sgl-project 社区；合入 2026-10-30T23:51:55Z；[sgl-project/sglang #46857](https://github.com/sgl-project/sglang/pull/46857)，main `a959f319f4bd4c44ebfa4b44f6067725fb097468`。

修packed scale、deferred-reduction和图尺寸，使既有CP query-row分片可与小批TP decode接合；只在full-TP CP、无attention-DP/LayerNorm-SP/LoRA/two-batch/quant-communication等gate下延后all-reduce，派生权重cache涉及图引用，更新路径受拒绝协议保护。作者8B300长262k输入组合TTFT下降来自CP与这些修补共同作用，不是新top-k算法。建议核对实际拓扑、图bucket与weight-update拒绝。

### H20 与特定 resident-thread SM 采用形状适配调度

维护方：sgl-project 社区；合入 2026-10-30T23:37:21Z；[sgl-project/sglang #47070](https://github.com/sgl-project/sglang/pull/47070)，main `319b98226f9e2feaf3b6179f7d41798f8ee8cbeb`。
维护方：sgl-project 社区；合入 2026-10-30T03:41:58Z；[sgl-project/sglang #47149](https://github.com/sgl-project/sglang/pull/47149)，main `4201b8fa996d80e09b4ab391d9b467a015ad43c5`。

H20新增Triton3.8的E128/N768/block128 FP8 MoE配置，覆盖moe_tp=1几何，TP8的N96并不匹配。DSv4.1 amax8以两波resident CTA限制grid并grid-stride扫key；1024-thread SM的MegaMoE predispatch可半线程处理两16B chunk，top-k不合条件回原launch。建议记录精确GPU/shape dispatch、padding与bitwise scale，勿按模型名或Blackwell泛化全部性能。

### AMD DSv4.1 wo_a 与 Qwen hyper/GDN 扩展 fused 路径

维护方：sgl-project 社区；合入 2026-10-30T06:41:39Z；[sgl-project/sglang #46677](https://github.com/sgl-project/sglang/pull/46677)，main `2b11736be17270862fa7dec13af341f67efba3ac`。
维护方：sgl-project 社区；合入 2026-10-30T06:44:10Z；[sgl-project/sglang #41880](https://github.com/sgl-project/sglang/pull/41880)，main `fcecdad5db2c228e32df65b0eb104a7e6da30ddd`。
维护方：sgl-project 社区；合入 2026-10-30T06:35:01Z；[sgl-project/sglang #44302](https://github.com/sgl-project/sglang/pull/44302)，main `5035bc97fbc675fda5e7388f84ac538979ffcec3`。

gfx950 wo_a保留32×32 block-FP8并preshuffle，activation按32量化；monokernel/确定性等仍gate，不改变DSv4的128块。Qwen hc_mix在可导入AITER/FlyDSL且BF16/gfx950/shape合格时用packed-weight cache；最终row上限128，正文16已过时。AITER GDN packing增加48/16=3的head ratio，替copy+cat。建议分别测scale/layout、权重更新重打包及ratio3数值；外部AITER内核未穷尽审阅。

### Muse Spark 按硬件推导 launch 默认值

维护方：sgl-project 社区；合入 2026-10-30T23:36:58Z；[sgl-project/sglang #45683](https://github.com/sgl-project/sglang/pull/45683)，main `98979301a188f3d98f41a83ea00c1ca4eb4adefd`。

启动解析NVFP4元数据，Blackwell默认TRTLLM、Hopper默认Marlin及不同memory fraction；Blackwell LayerNorm-SP、EP=TP>1且非PD decode可自动开MegaMoE，并提供显式关闭。DFlash2按draft metadata推FA4和8draft token，显式配置优先。正文含旧release阻塞说明，main SHA已确认合入。建议dump解析后的参数对照旧命令，检验BF16/多节点/显式opt-out，不能将recipe简化当新增所有硬件能力。

### Ascend DSv4 DSpark 接 arch35 算子并修 graph KV长度

维护方：sgl-project 社区；合入 2026-10-30T09:23:57Z；[sgl-project/sglang #39947](https://github.com/sgl-project/sglang/pull/39947)，main `72576df88dbc9837878c7c2c6ad007ccd906447f`。
维护方：sgl-project 社区；合入 2026-10-30T06:53:05Z；[sgl-project/sglang #46278](https://github.com/sgl-project/sglang/pull/46278)，main `585bfe32a724bde0fd65a04245561eb72614da35`。
维护方：sgl-project 社区；合入 2026-10-30T02:08:02Z；[sgl-project/sglang #47216](https://github.com/sgl-project/sglang/pull/47216)，main `9f0beaa43b2e71dbc66ef3900254edd4299e96be`。
维护方：sgl-project 社区；合入 2026-10-30T02:08:23Z；[sgl-project/sglang #40746](https://github.com/sgl-project/sglang/pull/40746)，main `0f3f619850f222b4f238ff3d336926e99a2a5c56`。

arch35使用专用quant sparse-attention与CPU query/KV mirror，draft禁wo_a量化时不触发MXFP8断言。DSpark原生NPU projection/add/argmax及sampling/verify接入sgl-kernel-npu；最终diff也接了MarkovSampleStep，正文称排除已过时。full-graph DFlash-family直接用已含verify宽度的CPU长度，避免1000+8再加8；MTP在FIA-v2 BSND开关下也走v2。建议核对目标CANN/kernel包和host mirror，回归因果边界与随机抽样；本轮未验证外部NPU kernel实现或重跑准确率。

### Intel XPU 支持 pure-SWA page1并整合受限模型路径

维护方：sgl-project 社区；合入 2026-10-30T02:56:17Z；[sgl-project/sglang #42725](https://github.com/sgl-project/sglang/pull/42725)，main `6162b071994fcf67090ebe3d45f7aebcdf03f4d7`。
维护方：sgl-project 社区；合入 2026-10-30T02:25:45Z；[sgl-project/sglang #46623](https://github.com/sgl-project/sglang/pull/46623)，main `80afdc1280bf37c1440e6e7009741cbc94e2fba0`。

MHA page1走KV gather+varlen workaround，需禁graph，MLA仍仅16/32/64/128；Unlimited-OCR补新的model_type/config注册。整合PR另以intel_xpu作为BF16 HiSparse支持backend，提供不支持dtype的明确错误，避免选择CUDA-only recurrent小批路径，并补模型/vision/运行库接线。它是多source集成，本轮只审核关键admission/dispatch；建议逐目标模型验证B60设备与graph/dtype gate，不泛称全XPU功能齐备。

### RL 验权和 prompt KV 版本追踪各自保留限制

维护方：sgl-project 社区；合入 2026-10-30T14:12:55Z；[sgl-project/sglang #47087](https://github.com/sgl-project/sglang/pull/47087)，main `75a5da7fb9319d9e83a39a20e720b86390b9727f`。
维护方：sgl-project 社区；合入 2026-10-30T14:09:01Z；[sgl-project/sglang #47085](https://github.com/sgl-project/sglang/pull/47085)，main `e7dd1d094dfe91194b665d403ff1da7cebf1d418`。
维护方：sgl-project 社区；合入 2026-10-30T14:12:27Z；[sgl-project/sglang #47086](https://github.com/sgl-project/sglang/pull/47086)，main `9cf2bc89250d50ccd0915d41747992f15ac8268d`。

raw_checksum按名称对当前持有的原始tensor bytes做SHA256，过滤非持久buffer、缺名报错，D2H发生在诊断操作中。prefill_weight_versions给prompt KV slot记版本并返回区间，默认关，HiCache/PD/Rust server等不支持组合明确拒绝。KV canary-info另随已committed token发布prefix/token/sampling摘要供复核，限定PP1、无PD、指定spec形态；它不是密码学安全或实际权重一致性的替代证明。建议分别注入权重差异、复用旧KV和输出token置换。

### msgpack IPC 保留多模态对象身份与 dtype

维护方：sgl-project 社区；合入 2026-10-30T20:46:01Z；[sgl-project/sglang #47445](https://github.com/sgl-project/sglang/pull/47445)，main `734aa5503ea8a299d1f742962f4f73a27bb0c7f8`。

opt-in SGLANG_USE_PICKLE_IPC=0原把tuple/dataclass等object字段解成不同类型，transport proxy也丢子类/共享owner。现仅不忠实的object值以每request共用pickler的ext封装，tensor/ndarray等保留既有路径；hidden-state单vector行和weight-update dtype名可往返。默认codec不变，建议对PIL、deferred preprocessing、VMM共享owner及更新dtype做round-trip；不能把此称为完全移除pickle。

### detokenizer 每请求一次header、每步一列token

维护方：sgl-project 社区；合入 2026-10-30T07:22:24Z；[sgl-project/sglang #47260](https://github.com/sgl-project/sglang/pull/47260)，main `8080507c3e578458af6749c8e428efdff737de93`。
维护方：sgl-project 社区；合入 2026-10-30T08:03:04Z；[sgl-project/sglang #47415](https://github.com/sgl-project/sglang/pull/47415)，main `15ef67584698eec9459d780ec47aaffaefc020f2`。
维护方：sgl-project 社区；合入 2026-10-30T17:57:38Z；[sgl-project/sglang #47106](https://github.com/sgl-project/sglang/pull/47106)，main `60a347b6f304e7fd953c5cf823081cf90a2ad0b2`。

不变的prompt tail/flags/route以DetokenizeHeader首次发送，detokenizer持有状态；abort显式release，同rid新header覆盖旧state。每步输出改flat int64列+row offsets，multi-tokenizer重新分区；Rust logprob flatten只扫有数据行，但保持列arity和空family语义。建议混合stream、beam、logprob、abort/rid重用并对齐row顺序；内部IPC schema已变，消费者须同版本升级。

### Rust native frontend 返回缓存分层和 scheduler 请求统计

维护方：sgl-project 社区；合入 2026-10-30T05:02:11Z；[sgl-project/sglang #46961](https://github.com/sgl-project/sglang/pull/46961)，main `93a936d007ada66544cb5ac855de540a596c2ca4`。
维护方：sgl-project 社区；合入 2026-10-30T20:45:27Z；[sgl-project/sglang #46963](https://github.com/sgl-project/sglang/pull/46963)，main `76820e89a97b652d62a54b3990edb3c5d065860d`。

cached_tokens_details从scheduler带device/host/storage来源并修counter cache_source，不再把所有hit记device；terminal meta_info补retraction、metrics启用时的scheduler timing、spec acceptance/histogram。部分中间frame统计仍与Python条件不同，缓存来源也由实际配置决定。建议用L2/L3命中和PD decode逐frame核对，确认末帧才有的统计再做延迟/acceptance分析。

### Rust 请求计时、loads 与 metrics flush 纠偏

维护方：sgl-project 社区；合入 2026-10-30T21:46:13Z；[sgl-project/sglang #47160](https://github.com/sgl-project/sglang/pull/47160)，main `a3517976f8a84c0e8050746c410674fe40dcb14f`。
维护方：sgl-project 社区；合入 2026-10-30T17:57:43Z；[sgl-project/sglang #47319](https://github.com/sgl-project/sglang/pull/47319)，main `dde1ef21a03da18fe226308fb27043013297e38e`。
维护方：sgl-project 社区；合入 2026-10-30T17:57:28Z；[sgl-project/sglang #47113](https://github.com/sgl-project/sglang/pull/47113)，main `dfade835a612d6148353c8f704d416c89f3c999e`。

HttpLayer可用RequestStart让handler前置工作进入E2E计时，避免Rust比Python漏算前处理。多节点attention-DP Rust listener改读node-local SHM的自身rank负载，修全503造成路由/扩缩信号失真；metrics flush先锁再snapshot/write，default priority先落值再选label。建议用慢前置层、各listener loads和并发scrape检查时长/单调性；listener自身rank数据不等于全局汇总。

### native embedding 与 prefill-only scoring 减少无用工作

维护方：sgl-project 社区；合入 2026-10-30T22:50:50Z；[sgl-project/sglang #42248](https://github.com/sgl-project/sglang/pull/42248)，main `783b3905b14ab0383b604bf45dfe873393680922`。

Engine.encode非float encoding可保留原dtype的detached CPU tensor，经专用IPC字段传递；这是in-process接口，不新增JSON编码。MIS小metadata转到CPU构造后一次H2D；raw-K/V FA路径明确要求prefill-only-disable-kv-cache。仅无量化dense CUDA模型省padded-tail逐层zero，MoE/quant/nonCUDA及idle rank保留。建议混合embedding batch、prefix-offset与动态量化amax回归，不能对所有padding取消清零。

### DP controller 只向已配置且健康 rank dispatch

维护方：sgl-project 社区；合入 2026-10-30T13:33:58Z；[sgl-project/sglang #45496](https://github.com/sgl-project/sglang/pull/45496)，main `227189488449d97d89d8da2fd5c1ee4bf1e2f955`。

非elastic update_active_ranks原直接更新status却不刷新active缓存，可能把未provision的扩容slot视为活跃。现以dp_active∩reported status重建active_workers/count，round-robin可直接取该列表；单线程event loop是其一致性前提。建议模拟rank掉线/恢复和尾部未配置slot，验证无active时的既有行为与路由计数，不把它当跨进程原子集群视图。

### 请求 cache_salt 与图片错误保留正确接口语义

维护方：sgl-project 社区；合入 2026-10-30T17:13:31Z；[sgl-project/sglang #46937](https://github.com/sgl-project/sglang/pull/46937)，main `5008726431693ef0883f83b230c08f77434e9b3b`。
维护方：sgl-project 社区；合入 2026-10-30T23:22:43Z；[sgl-project/sglang #43866](https://github.com/sgl-project/sglang/pull/43866)，main `e83c9d57bcd9efaf3a3f1552b8800021d7543fc7`。

Anthropic Messages现接收并透传cache_salt到Chat缓存路径，空/null/遗漏保持各自值；Pillow DecompressionBombError加入client-media异常，超大尺寸图片变400而非500。前者作者仅unit suite，server级same/different salt命中隔离仍待验。建议对stream/非stream及实际缓存namespace做回归，图片区分小文件字节数与解压后pixel上限。

### profiling/benchmark 可见真实 segment 与重 rank 容量

维护方：sgl-project 社区；合入 2026-10-30T23:08:01Z；[sgl-project/sglang #47353](https://github.com/sgl-project/sglang/pull/47353)，main `6441f6ba8dbbe5b16219eec794e7c76885e5fa97`。
维护方：sgl-project 社区；合入 2026-10-30T09:34:05Z；[sgl-project/sglang #46485](https://github.com/sgl-project/sglang/pull/46485)，main `e3718813ec5f47df0ad368b9a4713fbd3ad0a3bf`。
维护方：sgl-project 社区；合入 2026-10-30T21:34:20Z；[sgl-project/sglang #47130](https://github.com/sgl-project/sglang/pull/47130)，main `6fb0fd97fdc5ef0336663fb8134ead490956816a`。

FpmTiming保留interval，ready后按需生成只读segment，多个listener各收到一次；未ready读segment报错。新增sglang:scheduler_iterations_total按loop累加（不是正文误写的scheduler前缀），用于观察迭代速率。ragged one-batch基准按i%DP分配计算最重rank容量，避免总量能放却分两波测成假batch。建议对照GPU span与sum-segment、metric实际名称、ragged rank分布；该修正影响测量解释。

### diffusion LoRA merge cache 使用加载时文件身份

维护方：sgl-project 社区；合入 2026-10-30T23:25:26Z；[sgl-project/sglang #45834](https://github.com/sgl-project/sglang/pull/45834)，main `b5a0c9dc18832959d71df69faca1ce07e043c97c`。

缓存键使用resolved本地路径、加载时size/mtime快照及有效strength/alpha等配置信息，防文件加载后被替换却写到新身份；所有adapter设置完再merge，覆盖最后adapter不含的层，put失败只删temp。正文提到inode/LRU eviction，但最终diff没有这些新增，不能照抄。作者CPU测试无GPU验证，建议本地同路径替换、alpha/config变化及失败put回归，检查缓存命中确实对应已加载权重。

### MiniMax H3 Acc LoRA 修 gate/value 排列与 OOT admission

维护方：sgl-project 社区；合入 2026-10-30T12:58:56Z；[sgl-project/sglang #46907](https://github.com/sgl-project/sglang/pull/46907)，main `8776a94f2935e9a58b6dfc164c0d0e86f4d6b748`。
维护方：sgl-project 社区；合入 2026-10-30T12:51:40Z；[sgl-project/sglang #44725](https://github.com/sgl-project/sglang/pull/44725)，main `c8054551012b3df729a0f7c409af95f74286e416`。

离线build工具把Diffusers fc1 delta的[value,gate]两半交换成native[gate,value]后再merge，旧工具产出的Acc8权重需重建。full-loop denoise另允许is_out_of_tree平台，但后台仍需实现所需设备API/算子。建议以零base+单block LoRA验证矩阵行顺序，再测视频质量；CPU gate测试不证明任意OOT平台能完成denoise。

### HunyuanImage-3 复用不变 text-prefix KV

维护方：sgl-project 社区；合入 2026-10-30T20:24:30Z；[sgl-project/sglang #46454](https://github.com/sgl-project/sglang/pull/46454)，main `60e84e91cd41ea85203b1954d5d4e1a502c20c09`。

首denoise步保留每层causal text prefix的KV，后续只算image/timestep suffix并注意到完整KV，减少重复attention/MoE。CLA factor!=1回完整路径，default仍SDPA，FA3需显式选择且非bitwise等价，生成结束清cache；cot_text等请求字段接入。正文依赖旧CUDA noise-fix栈，不重复将前项修复算为本PR新交付。建议跨连续请求验证cache隔离、think prompt及FA3质量，不能套用前缀占比为整图线性速度。

### diffusion 修饱和转换并减同步、归约和帧复制

维护方：sgl-project 社区；合入 2026-10-30T17:06:31Z；[sgl-project/sglang #46277](https://github.com/sgl-project/sglang/pull/46277)，main `60725d0e176b0f08c4e45863e1ae74cdef7900f9`。

Wan-Animate-2先clamp[-1,1]再转uint8，避免越界wrap噪点；LTX Bongmath把收缩迭代求固定点，适用|h*a21|<1且分母非零。rollout init不再做finalize会覆盖的CPU复制，SDE/CPS logprob约掉方差逐元操作并保留标量项，ODE不分配零张量。NaN扫描可显式禁用但默认保留，fresh decode帧可原地归一化，stream tile仍非原地。建议先逐数值回归再测显存/host同步；作者A10为synthetic microbench，非整模倍率。

### LTX-2.5 native Gemma text encoder 与 eager launch 修补

维护方：sgl-project 社区；合入 2026-10-30T14:02:46Z；[sgl-project/sglang #47411](https://github.com/sgl-project/sglang/pull/47411)，main `e3e372931a31dd285bbaf95216327635a640333d`。
维护方：sgl-project 社区；合入 2026-10-30T13:59:34Z；[sgl-project/sglang #47270](https://github.com/sgl-project/sglang/pull/47270)，main `24268cc92bcdfebf9c61d697dac318217884c0ca`。

Gemma4Unified只建text stack，输出embedding+48层hidden states并保留layerwise/host表offload声明，融合QKV/gate-up，k_eq_v按K载V；transformers格式仍有adapter。LTX eager另缓存launch元数据、避免singleton AdaLN重排，并只在Hopper或Blackwell+BCG自动启stage2 CFG-rank Ulysses。建议比全49组hidden states、mask/RoPE和image-conditioned性能；后项明确还需其他Blackwellattention修复，不能独自宣称全回归恢复。

### AMD diffusion varlen 与 vision-cache 调度调整

维护方：sgl-project 社区；合入 2026-10-30T03:55:44Z；[sgl-project/sglang #47062](https://github.com/sgl-project/sglang/pull/47062)，main `0b05d49d1fba324dafbfec4db4cec7cf541b4fc5`。
维护方：sgl-project 社区；合入 2026-10-30T16:57:26Z；[sgl-project/sglang #47466](https://github.com/sgl-project/sglang/pull/47466)，main `dee613b283fc17b82ed573926921b3b4ebd3da60`。

AITER varlen撤销gfx942一律Triton fallback，改FAv3并传RTZ转换模式，旧MiniMax-H3 hang在作者覆盖尺寸未复现，不证明任意旧版本均安全。Qwen3-VL diffusion去掉单vision feature跨request cache，因pixel D2H hash和pin成本超过约11ms tower；stage级条件与VAE缓存仍有。建议用零长segment和真实image reference A/B，单请求e2e改善不能归因所有联合优化。

### Humming0.1.18 集成保留多级 scale 与分层 dtype

维护方：sgl-project 社区；合入 2026-10-30T05:36:48Z；[sgl-project/sglang #46018](https://github.com/sgl-project/sglang/pull/46018)，main `ca939407423088dffd33304e0ea7d092cba355d2`。

接schema weight quantization、static/secondary/per-shard scales，按各MoE子层dtype分buffer、按不同M tile分别align路由，并将cache/tmp纳SGLANG_CACHE_DIR；这部分是版本迁移所需的实质集成代码，外部kernel未逐项审核。建议用w13/w2不同tile、DeepEP FP8 dispatch及masked quantization验证shape/scale和checkpoint恢复，不能仅升级依赖包就推定所有量化格式相容。

覆盖与观察：六仓库 release、merged PR search、默认分支 commits 均实际读取并分页；主窗无新 release，回溯里的 Megatron Core0.20.0/TorchTitan0.4.0已历史去重。Slime主窗没有新的默认分支提交。SGLang #47135 LoRA upsert、#47355/#47358 token-space策略仅合入集成分支，未计为main交付。vLLM #61667仍open同head，旧late-WRITE/外部传输完成保证不因此关闭。历史398个前轮pending、其他旧区间及本轮未审overlap候选继续保留，枚举完成不等于机制审计完成。

<a id="extended"></a>
## 扩展开源工程与硬件软件

本节主窗口为 **2026-10-30 01:03:12Z—2026-10-31 01:02:00Z**，72 小时回溯从 10-27 01:03:12Z 开始；以下日期均为 UTC。已按 24 篇归档的 PR/commit 稳定 ID 去重。除明确的 release 条目外，均指开发分支的新提交，不能推定已进入此前发布的安装包。以下性能、准确率与测试结果均是作者自报或仓库测试设计，本轮只核读公开源代码/diff，**未运行 GPU/NPU/TPU 或多机复现**。

### 训练正确性、分片与编译

- **DeepSpeed 社区｜10-30，master：GPTBigCode AutoTP 的共享 K/V 梯度终于跨 TP 求和。** 每个 TP rank 持有自己的 Q 行和完整 K/V 行，旧实现只更新本地 Q heads 对共享 K/V 的部分贡献，副本会分叉。[#8928](https://github.com/deepspeedai/DeepSpeed/pull/8928) 在 `fused_LinearLayer._sum_replicated_kv_gradients` 给可训练 weight/bias 注册 hook，只 all-reduce K/V 行。该 PR 于 **01:04:53Z** 合入，正是上轮 cutoff 后暂存项；本轮正式纳入。作者在 macOS CPU、TP2/4、SGD、关闭梯度裁剪下比对单设备；**尚未跑 GPU，ZeRO-3、torch.compile、默认梯度裁剪下重复计算 K/V norm 的问题未覆盖**。建议有 BigCode/MQA AutoTP 训练的团队优先读 diff，复现参数梯度与跨 rank 一致性，不能只看 loss。

- **DeepSpeed 社区｜10-30，master：ZeRO-3 CPU optimizer offload 改为 pinned FP32 回传后在设备转换。** [#8840](https://github.com/deepspeedai/DeepSpeed/pull/8840) 为 CPU master partitions 固定页，把完整 subgroup 的 FP32 master 先传到 GPU 再由目标 copy 完成 dtype 转换，避免先在 CPU 产生 pageable BF16 临时量。[合并版本 `c6291b7e` 的 `stage3.py`](https://github.com/deepspeedai/DeepSpeed/blob/c6291b7e16d65eab1c387f52f0acd5e206c5a996/deepspeed/runtime/zero/stage3.py)中 `_copy_pinned_master_to_device` 是整分区 `src.to(device)` 再 `dst.copy_`，随后 event 同步，**没有 PR 描述中的 2^26 元素分块，不能给出 0.25GiB 临时显存上界**；临时 FP32 空间随 subgroup master 大小变化。PR自报 Qwen2.5-32B、8×H200、全部 optimizer 放 CPU 时 step12.08→9.42秒和峰值不变，但未确认该数字对应最终整分区实现，不作合并版本性能保证。建议 CPU-offload 用户检查 pinned 主存预算和完整 step 时间，同时验证更新后参数一致性。

- **DeepSpeed 社区｜10-30，master：AffineIR 能直接描述 AutoEP+EDP+ZeRO 的真实 expert 存储。** [#8941](https://github.com/deepspeedai/DeepSpeed/pull/8941) 的 `expert_storage_affine_maps()` 把 expert placement 与 ZeRO storage map 嵌套，ZeRO0/1/2 列出 EDP 副本持有者，ZeRO3 映射到去掉尾部 padding 的 `ds_tensor`，避免先还原 packed EP tensor。map 在本地构造，不额外 collective；expert TP、量化/NVMe ZeRO3 storage 和 expert optimizer-state maps 仍显式不支持。建议 checkpoint/训练转推理集成人员阅读映射与拒绝条件，勿将“能规划 transfer”写成所有运行时已接通。

- **NVIDIA Transformer Engine｜10-30，main：延迟 wgrad 和分片 dummy gradient 两类训练错误分别修正。** [#3552](https://github.com/NVIDIA/TransformerEngine/pull/3552) 将 Linear、LayerNormMLP、两种 GroupedLinear 的 `backward_dw()` 从覆盖 `.grad` 改为 None 时建立、已有时累加，保留 tied weights 与多 microbatch 贡献；CUDA Graph 路径另保留 graph-pool 梯度并在 replay 时合入调用方梯度，避免 `zero_grad(set_to_none=True)` 释放 capture 地址。[#3748](https://github.com/NVIDIA/TransformerEngine/pull/3748) 则让 fused-DDP dummy gradient 匹配原始本地 shard shape，而非完整 GEMM main_grad shape。新增测试定义覆盖 drain 顺序、reset、bias、grouped/FP8 支持门槛；**PR同时披露 reviewer 在SM100的50个eager用例中看到24个失败，尚未重跑或从逐例日志诊断；本次revision没有native SM100、CUDA Graph或FP8通过声明。** 私有CPU控制和测试定义检查通过不能替代这些GPU失败的闭环。建议流水线延迟wgrad与分片训练优先复现，采用前须核查失败与后续修订。

- **NVIDIA Transformer Engine｜10-30，main：MoE aux loss 增加可选确定性规约，GroupedLinear 收紧隐式 fallback。** [#3587](https://github.com/NVIDIA/TransformerEngine/pull/3587) 新增 `deterministic=False` 参数，打开后使用固定顺序两阶段 FP32 reduction 和最多 256 个 partial 的 workspace，原 atomic 默认路径保留；保证范围是同硬件/软件、同输入的重复调用。[#3613](https://github.com/NVIDIA/TransformerEngine/pull/3613) 减少 grouped 路径无用 Python 工作，并对单 grouped 参数或 delayed-wgrad 下不受支持的 split-quantize fallback 报错。建议重现训练漂移时明确是否开启确定性，并检查旧配置是否依赖自动 fallback；本轮无新的端到端 CPU-overhead 数值验证。

- **PyTorch 社区｜10-30，main commit fe586824：dropout Philox offset 少预留会重用随机数。** [提交及对应 #202606](https://github.com/pytorch/pytorch/commit/fe586824ec8d03067aaf5a05614c962973a3f9a5) 在选择 vector width 后按实际每轮 RNG 消耗预留 counter，并修正窄向量路径。旧 VEC8 half/BF16 路径可能实际取 8 个随机数却按 4 个计，后续 RNG 调用重叠。影响随机数状态推进与 eager/compile 对齐，不能只当性能更新。建议依赖可重复训练、图 replay 或 dropout 对齐的用户读 `Dropout.cu` 与随机状态回归。GitHub PR 的 `merged_at` 为 null，本轮落地依据是默认分支 commit。

- **PyTorch 社区｜10-30，main：precompile capture 补齐配置、autotune 与 custom-op 注册依赖。** [#201940 对应提交](https://github.com/pytorch/pytorch/commit/08c2d21eeac45d7bb2b95f497938e99c0250222d) 让 `capture(mode=, options=)` 沿用 torch.compile 配置语义；[#202255 对应提交](https://github.com/pytorch/pytorch/commit/8c03c528651c04532e121f4d56c4bc4b4e7136b2) 在 capture 编译阶段完成 Triton autotune，避免没有被样本执行的 backward kernels 保留多个 launcher，同时禁用不能冻结选择的 multi-kernel；[#202258 对应提交](https://github.com/pytorch/pytorch/commit/1dbd30475f96a60645f5acfdbd76b7a615d5d896) 将 custom-op 模块/ops/dispatch-key 需求写进 artifact，在 replay 前导入和校验。部分与冻结/caches 冲突的 options 会明确拒绝。建议部署预编译训练/推理 artifact 的团队在干净进程验证 backward 与自定义算子，勿仅重放 capture 当时已注册的环境。

- **PyTorch 社区｜10-30，main：PDL 静态 launcher 和依赖链补齐。** [a257a35b / #202126](https://github.com/pytorch/pytorch/commit/a257a35b006015c2161a593b7cd7f108a0a39f32) 为 static Triton launcher 加入 `cuLaunchKernelEx` 路径；[a3222da1 / #199812](https://github.com/pytorch/pytorch/commit/a3222da1fd41062c158297749d63d593cc8a584a) 跟踪所有可能仍在运行的 kernel，到真正等待处清除依赖，而非只盯相邻 kernel。建议编写 combo kernel、extern-call 和模板融合的用户阅读 scheduler/launcher diff，结合 correctness 测试复现；这些提交本身不建立普遍的加速幅度。

- **PyTorch 社区｜10-30，main：动态形状范围与 dict truthiness guard 的正确性修复。** [8cf5c79f / #202505](https://github.com/pytorch/pytorch/commit/8cf5c79f94ffd6cf2301b754049ccac82c6ac1fa) 把 value-range 实际变化纳入 ShapeEnv cache version，防止区间收紧后 simplify/floor-div/static-evaluation 仍命中旧结果；[bc7e151b / #202374](https://github.com/pytorch/pytorch/commit/bc7e151b196582c7c6b83ea1d17d8004e5f99c65) 调整 dict 长度相关 guard。建议动态 batch/sequence 和依赖字典分支的 compile 用户运行对应最小回归；这是防错，不是减少 recompile 的量化保证。

- **PyTorch/ROCm｜10-30，main：混合 dtype 和 CK GEMM 多个 silent-wrong-result 路径修正。** [81e20498 / #202782](https://github.com/pytorch/pytorch/commit/81e20498b3650d0d5a6e98d4e51730bae4ece42f) 将向量化 elementwise load 转换到 functor 参数 dtype，避免 FP32 操作数先舍入到 BF16/FP16 输出 dtype；[#202096](https://github.com/pytorch/pytorch/commit/7768be02b3f66feeb05945ccbb70880b545bda08) 让 CK 不支持的问题回退默认 BLAS、补 fp16 实例并保留 beta；[#200195](https://github.com/pytorch/pytorch/commit/5f9684c2db892325470f836a2725c1cd6e68e65f) 在 gfx950 的 fast-math 编译加入架构限定的 `-fno-associative-math`，处理间歇错误的 CK addmm。建议 AMD 训练栈用混合 dtype/out/in-place、非整齐 shape、beta≠0 和 fresh-cache autotune 回归，不能仅以常见大矩阵吞吐判断正确性。

- **PyTorch/ROCm｜10-30，main：profiler 停止和退出的异步生命周期修复。** [b0e48e7b / #202758](https://github.com/pytorch/pytorch/commit/b0e48e7b450ff27ab02c171a520c1183da9eed93) 对 in-flight dispatch record 给予有界等待（配置默认 20ms），处理设备同步已经返回但 rocprofiler completion handler 尚未交付记录的窗口；[45b310dd / #202665](https://github.com/pytorch/pytorch/commit/45b310ddadcbad7f9980c10e721e19030432e001) 在 TLS 销毁后避免退出阶段 HIP tracing 访问已释放 correlation state。建议排查空 GPU timeline、CUDA Graph replay 缺 kernel 或进程退出崩溃时读这两项；等待是有界的，不等同保证外部 recorder 永不丢数据。

- **PyTorch/MPS｜10-30，main：macOS 27 的原生 FP8 MPP matmul 路径。** [24d8b9a7 / #202545](https://github.com/pytorch/pytorch/commit/24d8b9a75c33b427ae7284d672707064f8a16299) 为 `_scaled_mm_v2` 增加 Metal 4.1/macOS 27 的独立 metallib，使用 MPP 原生 e4m3 operand，避免逐元素先解码到 half。需要新 SDK 与运行时；不是所有 Apple 设备/旧 macOS 都能使用。建议 Apple 低精度实验者检查 library 选择与精度测试，本文不把 PR 的局部硬件结果泛化为 MLX 或整模型加速。

- **PyTorch 社区｜10-30，main：两项优化已撤回，不能照早先落地记录直接启用。** [c1f7cc16](https://github.com/pytorch/pytorch/commit/c1f7cc164ffe53079ea3d2247bec68bd0c0ead12) 撤回 #197100 的 zero-copy non-contiguous matmul folding，commit 明示破坏 torchao；[4e1cfdf6](https://github.com/pytorch/pytorch/commit/4e1cfdf6c4f95cd16b401f0289c20bcd8f57e1ab) 撤回 #201248 的 gfx11/12/1250 wave32 int4mm，原因是 ROCm 7.0 build compatibility。建议跟踪修复后重新落地主线的 commit，当前 cutoff 不把这两项算作仍可用能力。

- **Hugging Face Transformers｜10-30，main：PEFT 保存区分“所有 rank 参与 gather”与“保存 rank 写文件”。** [#49316](https://github.com/huggingface/transformers/pull/49316) 移除 Trainer 仅按 `requires_grad` 过滤 state dict 的做法，让 PEFT 自己选 checkpoint 所需状态，并让各 rank 调用 `save_pretrained` 所需 collective，只有指定 rank 写文件。它依赖 PEFT 的提前过滤能力（关联 peft#3963），不证明旧 PEFT 版本也有同样行为。建议 FSDP/ZeRO+adapter 用户阅读测试并复验 adapter checkpoint 完整性和重载，尤其有非训练参数仍须保存的配置。

- **Hugging Face Transformers｜10-30，main：混合 dtype FSDP2 与裸 attention-sink 参数的 TP 边界补齐。** [#49501](https://github.com/huggingface/transformers/pull/49501) 在加载时把必须保留 FP32 的 router 等模块先独立 fully_shard，避免 BF16/FP32 同组触发 FSDP uniform-dtype 断言；作者以两张 H100 的 ERNIE-4.5-MoE 训练步自测。[#49567](https://github.com/huggingface/transformers/pull/49567) 加 `local_params` style 放在参数所属模块上，把已分片 sink 参数在 forward 视为 local tensor，避免 DTensor 与本地 attention activation 混用。建议全参 MoE 训练和 sink-attention TP 用户读 plan 生成与测试；LoRA 不等同相同触发条件。

- **Hugging Face Transformers｜10-30，main：组合模型 ZeRO3 初始化与 GPT-OSS MXFP4 kernel 映射回归修正。** [#49982](https://github.com/huggingface/transformers/pull/49982) 在 `_init_weights` 可能访问直接子模块权重时，额外 gather 非 PreTrainedModel 直接 children 的参数，避免空 shard 上计算 fan-in/out；[#50150](https://github.com/huggingface/transformers/pull/50150) 恢复 `Mxfp4GptOssMLP` 的 MegaBlocks hub-kernel decorator，处理 XPU 退回慢 Triton、CPU 无 Triton driver 的回归。后者作者自报单 Intel Arc Pro B60、gpt-oss-20b、BF16 greedy 的首 token 4.27×、后续 1.90×，**仅是该退化基线的恢复**，CUDA 的 dequantize 路径不受此修复影响。建议相关设备跑首次 forward、生成和保存重载回归。

- **Hugging Face Transformers｜10-30，main：可显式要求 SDPA 用 GQA backend。** [#49459](https://github.com/huggingface/transformers/pull/49459) 最终代码的开关是 `TRANSFORMERS_SDPA_FORCE_GQA=1`，使 `use_gqa_in_sdpa` 越过原 heuristic，避免主动重复 K/V；PR 描述提及的 `force_gqa` 参数不是最终接口。应先确认所用 PyTorch/backend 的 masked GQA 支持，默认路径未改变。适合定位 K/V 展开带来的显存或性能差异，不宜在未支持组合上无差别开启。

- **Hugging Face TRL｜10-30：v1.17.1 修 LFM2.5 参数括号截断；main 增加 DeepSeek-V3/R1 工具训练模板。** [v1.17.1](https://github.com/huggingface/trl/releases/tag/v1.17.1) 对应 [#7942](https://github.com/huggingface/trl/pull/7942)，工具参数中普通 `)` 不再提前结束调用，但字符串里恰有 `), name(` 仍有 regex 歧义。[#7961](https://github.com/huggingface/trl/pull/7961) 让拒绝 dict 工具参数的原模板明确不满足 prefix-preserving 条件，切换训练模板，并新增 DeepSeek-V3 response template，处理 think-prefill 与工具输出后续 turn；新式解析测试要求 Transformers≥5.13。后者不能从发布时间推断已包含在 v1.17.1。建议 GRPO/蒸馏工具循环同时验证模板渲染、解析与完整工具轮次。

### Attention、kernel 与低精度接口

- **Dao-AILab FlashAttention｜10-30，main：typed forward config 成为真实配置入口。** [#2733](https://github.com/Dao-AILab/flash-attention/pull/2733) 让 `_flash_attn_fwd` 根据 host metadata 选择或校验 `FwdConfig`，kernel 不再静默覆盖配置；compile key 归一化 config，split 数量 2/4 仍共享 on/off specialization，降低新增字段漏进 cache key 的风险。建议自定义调优者读 `config.py::select_fwd_config/validate_fwd_config` 和调用点；这是主线接口/策略变化，不可假定已进入 10-28 的 beta37 包。

- **Dao-AILab FlashAttention｜10-30，main：swap-AB split-KV decode 已接入自动 forward dispatch，旧 PR 说明过时。** [#2973](https://github.com/Dao-AILab/flash-attention/pull/2973) 用 KV token 占 MMA 的 128-row 维、query heads 占较小 N 维，减少小 GQA group 的 padding；最终 diff 的 `SwapABPolicy`、`select_fwd_config`、`_flash_attn_fwd` 已含 paged KV、按序列 split 元数据和 fallback。**PR 开头仍写“独立 dense 入口、不支持 paged、不改 dispatch”，与最终代码冲突，应以最终 diff 为准。** 限制是 SM100/103、单 query/sequence 推理、FP16/BF16、受限 head-dim/page/alignment，训练、sink、block-sparse 等会拒绝或回退。自动策略避开作者在 B300 CUDA Graph sweep 测到的多短序列退化；建议 B200/B300 用户按自己 KV capacity/actual lengths 冷缓存复测，不照单一带宽图外推。

- **FlashInfer 社区｜10-30：v0.7.3rc1 发布；Prims-TS 主线将逻辑长度与编译拓扑分开。** [rc1](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.7.3rc1) 的官方 release body 自身注明被长度上限截断，完整范围应看 compare，不能把 release notes 尾部当全部变更。[#6925](https://github.com/flashinfer-ai/flashinfer/pull/6925) 扩 NHD、sinks/LSE、FP8/MLA shape 并引入 SM-bounded runtime split-KV；统一 `softmax_scale/output_scale`、`*_dtype`、`seq_lens_kv` 等接口名。[#7106](https://github.com/flashinfer-ai/flashinfer/pull/7106) 将 Q/KV 逻辑长度移到 launch bounds/live metadata，复用同一 topology kernel，并在运行时选小/大 reducer；资源拓扑和存储布局仍参与 specialization。建议动态批处理集成者先迁移参数名、校验 plan capacity 与实际长度，测 JIT/cache 命中；不能把容量越界当动态支持。

- **FlashInfer 社区｜10-30，main：SDSA 的单头 BF16 paged indexer 组件落地。** [#6422](https://github.com/flashinfer-ai/flashinfer/pull/6422) 新增 `single_head_paged_mqa_logits`，针对 SM100/103、D128、page64、每请求 1–8 query rows，不用多头权重/ReLU；按 context capacity 定 grid，支持 CUDA Graph。它只是 SDSA 六步中的第 3 步，未证明完整 indexer top-k 与稀疏 attention pipeline 都已交付。建议读 public wrapper、trace template 与边界/graph 测试，观察后续 #6423 等整合。

- **FlashInfer 社区｜10-30，main：PDL 提前放行 MXFP8 GEMM 与 BF16 PDL autotune。** [#5982](https://github.com/flashinfer-ai/flashinfer/pull/5982) 在 quantizer 的 upstream wait 后调用 `griddepcontrol_launch_dependents`，让下游初始化重叠量化但仍等待真正依赖；作者 GB200 固定 M/N/K=4/6144/1024、split-K4、旋转权重的 graph 测试为 8.559→8.175µs，整服务改善更小，不能混用两种基线。[#8032](https://github.com/flashinfer-ai/flashinfer/pull/8032) 让 `mm_bf16(pdl=None)` 才会同时调优 PDL on/off，默认仍 False，untuned opt-in 也先走 off；动态产生的 weight 必须满足 PDL 输入依赖约束。建议 kernel 调用者读依赖合同再测实际 serving，不将“可 autotune”当“默认安全启用”。

- **FlashInfer 社区｜10-30，main：Mamba SSD 的 dstate≠chunk_size 与 KCP workspace 跨 stream 生命周期修复。** [#3494](https://github.com/flashinfer-ai/flashinfer/pull/3494) 为 X 与 B 的 TMA MMA-B partition 传各自 NK tile，避免 dstate=chunk_size=128 掩盖维度错误；测试扩到 dstate64。[#8008](https://github.com/flashinfer-ai/flashinfer/pull/8008) 在 KDA 共享 KCP workspace 换 stream 时为活跃 storage 做 record，补足单纯 wait_stream 无法阻止 allocator 提早复用的缺口。[#7985](https://github.com/flashinfer-ai/flashinfer/pull/7985) 新增同 context 的 q/k/v 多卷积入口，一方向一次 halo all-to-all，保留各自计算/梯度。建议 CP 训练团队读 allocator 压力与不均匀 rank 到达测试，通信次数减少不等于所有模型的等待都消失。

- **FlashInfer 社区｜10-30，main：ragged candidate-support LM head 可用于采样掩码重放与 teacher KL。** [#7956](https://github.com/flashinfer-ai/flashinfer/pull/7956) 增加 SM100/103 Cake 实验接口，在每个 token 自己的 `ids/offsets` 候选集上归一化，输出 packed log-prob 或 detached-teacher forward KL；按候选密度选择 sparse/dense 各 pass，避免必建完整 T×V logits。**其 softmax 定义域是候选集，不能替代全词表 KL 而不改变目标；teacher 不求导，reduction、weight、temperature² 由调用方处理。** 作者的大倍数比较以 eager dense-mask 为基线，本文不作为整训练收益。建议 RL/蒸馏开发者先核对目标函数再复现 backward 与峰值显存。

- **FlashInfer 社区｜10-30，main：实验 Qwen block 扩到 prefill/extend 与 packed KV pool。** [#7983](https://github.com/flashinfer-ai/flashinfer/pull/7983) pin cuDNN Frontend `4c8e2435549`，给 Qwen3.8 Flash-Next 加 uniform chunk 的 extend/prefill，经 serving cache 接续 decode；[#8036](https://github.com/flashinfer-ai/flashinfer/pull/8036) 为 Qwen3.5 block 增 Qwen3.8-27B dense-attention geometry 与 `kv_layout="UNIFIED"` 的一块 K|V pool 半视图绑定。二者属实验 block，不能推定完整 serving stack 或全模型已接入；后者没有改 kernel。建议读 geometry、rounding、cache mutation 合同，复现连续 chunk→decode，相关 vendor 内部行为仅按固定依赖合同理解。

- **Triton 社区/AMD｜10-30—31，main：gfx1250 descriptor buffer 与数值语义分离。** [#12512](https://github.com/triton-lang/triton/pull/12512) 暴露 `ttgl.amd.gfx1250.buffer` 的 descriptor/load/store、scope/cache policy；gfx1250 没有 buffer-to-LDS load，num_records 仍 32bit，越界向量行为和 gfx950 不同。[#12560](https://github.com/triton-lang/triton/pull/12560) 将 CDNA5 原生近似 tanh 仅用于 `fast_tanhf`，precise `tanh` 回到 OCML。作者的 lit/IR 验证已跑，但真实 gfx1250 数值误差验证在 PR 中仍列 TODO。建议 AMD kernel 作者分别检查边界访问与精度 API，不把 ISA 的近似指令等同精确数学函数。

- **Triton 社区｜10-30，main：libdevice fma/fma_rn 删除，sort/topk 接受负末维。** [#12558](https://github.com/triton-lang/triton/pull/12558) 属 breaking change，CUDA/HIP 与通用 stub 删除冗余默认舍入 wrapper，调用者迁往 `tl.fma/ttgl.fma`；非默认 rounding 的 CUDA variants 保留。[#12559](https://github.com/triton-lang/triton/pull/12559) 在末维校验前归一化负轴，使 -1 合法，其他非 minor dim 仍不支持；作者尚未跑 GPU。建议升级 main 前先搜索旧 wrapper imports，再跑 fused arithmetic/排序回归，勿把它说成 10-27 v3.9.0 已有变化。

- **PyTorch Helion（补充发现）｜10-30，main：CuTe 后端补 ragged GEMM、TMA 对齐和 split-K 搜索。** [#4193](https://github.com/pytorch/helion/pull/4193) 对足够大的静态二维/三维、TMA stride 不合法 operand 在 host prelude 做 padding copy，保留逻辑 shape，付出额外读写；[#4194](https://github.com/pytorch/helion/pull/4194) 让静态 edge tiles 用 TMA zero-fill/向量 store，动态 shape 仍 generic；[#4195](https://github.com/pytorch/helion/pull/4195) 加 in-kernel split-K/workspace schedule；[#4196](https://github.com/pytorch/helion/pull/4196) 将 two-CTA edge 搜索门槛降至256并补 128-row pair seeds。作者 B200 对齐/奇数/K-heavy 个案收益不能拼成统一加速比。建议用新 [96-cell matmul sweep #4192](https://github.com/pytorch/helion/pull/4192) 分别看 burst 与 sustained timer，并审计 staging copy 成本。

- **PyTorch Helion（补充发现）｜10-30，main：Muse Spark EP4 verify MoE megakernel。** [#4198](https://github.com/pytorch/helion/pull/4198) 将 T=8、EP4、NVFP4 的 W13/SwiGLU/量化/W2、post-expert norm 和 rank-order all-reduce 合在一条实验 megakernel 路径，原地读取 FlashInfer TRT-LLM weight layout，并有 FP64 reference/check 入口。它是固定模型几何的 pretuned kernel，不代表通用 MoE backend 已切换；建议模型/kernel 研究者先运行四 rank correctness，再与相同 cold-L2/PDL 配置下的 SGLang 基线比较。

### 推理运行时、缓存与集群服务

- **NVIDIA TensorRT-LLM｜10-30，main：每层每 forward 的 DynamicEPLB 原型。** [#19646](https://github.com/NVIDIA/TensorRT-LLM/pull/19646) 新 `per_layer_dynamic` 模式按当前 routing 在 GPU 选热点 expert，COPY 高优先级 stream 用 HALO-Q+NVLink multicast/TMA 填 helper slots；MAIN 在 route epoch 后启动 MegaMoE，并以 READY generation 保护尚在复制的 helper reads，利用量化/shared-expert 工作重叠。**限制包括 NVFP4、MEGAMOE_CUTEDSL、≤384 experts、SM100/103/107、单 NVLink multicast domain、attention DP+纯 EP2–32，无 CUDA Graph/DWDP；配置标记 prototype。** 建议负载不均的 context-heavy 服务读 planner/arena/cleanup 合同，先验证硬件拓扑、额外权重槽与失败清理，不能用主线合并替代生产资格认证。

- **NVIDIA TensorRT-LLM｜10-30，main：Kimi K3 hybrid KDA+MLA 可选 NVFP4 KV。** [#20888](https://github.com/NVIDIA/TensorRT-LLM/pull/20888) 给 hybrid V2 manager 接 FP4 MLA layout，默认保留最后两个 MLA layers 为 FP8（还受每个 PP stage 至少一个 NVFP4 MLA 的约束），可用 `TLLM_KIMI_K3_FP8_KV_LAYERS` 改选；补 chunked prefill、context-only 与部分 disagg 路径。chunk 必须满足16对齐，hybrid reuse 需要 chunked prefill，C++ transceiver/非 NIXL 等组合未支持。PR 虽已 merged，说明中的 SM107 验证/准确率基准仍有未完记录。建议低精度 KV 团队先做混合 batch、chunk→decode 和质量回归，不能报成已完成 SM107 end-to-end 验证。

- **NVIDIA TensorRT-LLM｜10-30，main：KV Cache Manager V2 的 reuse policy 下沉到 attention/SSM buffer。** [#21088](https://github.com/NVIDIA/TensorRT-LLM/pull/21088) 用 REQUIRED 约束 prefix match、OPTIONAL 在匹配端点显式选用、PRIVATE 不发布可复用 checkpoint；buffer 可覆盖 layer policy，新增 buffer-group 查询，对旧 layer API 的歧义 split layout 明确报错。**`AttentionReusePolicy` 被 `ReusePolicy` 替代且无兼容 alias；Mamba 独立 recurrent/replay indices 和 wrappers/connectors 的 split-buffer 消费仍待跟进，不会默认把 replay buffers 变 PRIVATE。** 作者更新后的最终自测声称154 C++、2431 Python通过，自动摘要仍带旧“待验证”信息；本文保留来源差异、不将其当独立验证。建议底层缓存接入者读 config、index converter 和 split-layout拒绝路径。

- **NVIDIA TensorRT-LLM｜10-30，main：TriAttention 约束 workspace，Mamba MTP 取消 whole-pool copy。** [#20871](https://github.com/NVIDIA/TensorRT-LLM/pull/20871) 让 eviction score/phase workspace 按有界 decode window 存储，预热 resident 单请求 fallback，batch allocation OOM 时串行退回，budget>16384 提前拒绝；allocation-free 限 warmed TP1，TP>1 union 仍有 allocating all-gather。[#21105](https://github.com/NVIDIA/TensorRT-LLM/pull/21105) 将实际64bit slot stride 传进 CUDA op，直接更新 hybrid pool 的 padded SSM slots，避免整个 state contiguous+copy-back；Python/native 必须成对升级。建议长上下文和 MTP 用户测真实 pool memory、OOM/fallback 与图 replay；局部内存结果不能代表整服务吞吐。

- **NVIDIA TensorRT-LLM｜10-30，main：MoE 原生 FP8 LoRA 权重可保留到执行路径。** [#18555](https://github.com/NVIDIA/TensorRT-LLM/pull/18555) 在 PEFT cache 保留 E4M3 adapter，以 grow-only workspace 在 grouped MoE LoRA 执行中转换，并修 grouped-GEMM problem metadata 生命周期。作者有 H100 eager/CUDA Graph 与 BF16 reference 的 integration 验证；初版连 BF16 adapter 也预留转换 workspace，仍有开销跟进。建议多 adapter MoE 服务比对 adapter-vs-base/logit delta 和图 replay，不把 FP8 cache storage 当全程 FP8 GEMM。

- **NVIDIA TensorRT-LLM｜10-30，main：idle disagg cancellation 进展与 pipeline NCCL 初始化修复。** [#20702](https://github.com/NVIDIA/TensorRT-LLM/pull/20702) 让等 peer/KV 的 model-idle 请求无需下一个 batch 才处理取消，但**eligibility 不是 retirement 证明**，仍等 transceiver 确认、KV/AUX claims 未释放不能回收；部分能力受 ownership bridge gate 限制。[#20805](https://github.com/NVIDIA/TensorRT-LLM/pull/20805) 在 pipeline communicator 建立前也设置未显式指定时的 `NCCL_GRAPH_REGISTER=0`，避免复用 MPI workers 时 NCCL 首次缓存值造成图 capture hang。建议 P/D 服务复现取消晚到 readiness、多 rank 空 batch 和进程池连续模型用例。

- **NVIDIA TensorRT-LLM｜10-30，main：删除旧 executor/config API 是明确兼容性变化。** [#21041](https://github.com/NVIDIA/TensorRT-LLM/pull/21041) 删除 `BatchingType`、`ExtendedRuntimePerfKnobConfig` 以及 `EagleDecodingConfig.num_eagle_layers`；`max_draft_len` 保留。建议自编 C++ executor、旧 Python imports 和 serve YAML 的用户先扫描引用并重建 bindings；不把“obsolete”误读为下游完全不用迁移，公开 CI 摘要也不足以证明全部下游兼容。

- **NVIDIA Dynamo｜10-30，main：SGLang LoRA KV 命名空间与 KVCR inventory 接口补齐。** [#15746](https://github.com/ai-dynamo/dynamo/pull/15746) 仅在 KV events 已启用且未指定 format 时默认 `"dynamo"`，把 adapter name/hash namespace 交给现有 router，旧 SGLang0.5.19 XPU image 因缺 field 保持原样；[#17655](https://github.com/ai-dynamo/dynamo/pull/17655) 接 scheduler inventory/peer endpoint 生成 KV hints。后者当前 linker inventory 限 TP1，早期 revision 的两节点 Qwen3-32B RDMA 1984-token restore 不等于最终 quick-start image 或 TP>1验证。建议 prefix-router/LoRA 服务读 compat gate 与 hint expiry，测试 adapter 隔离和 eviction 后不再报 peer hit。

- **NVIDIA Dynamo｜10-30，main：active-active PoolRelay 及 replica 一致的负载视图。** [#17347](https://github.com/ai-dynamo/dynamo/pull/17347) 为一个同名/同 namespace DGD 创建 `DynamoPoolRelay`，默认两 active Pods 与 Service；[#17686](https://github.com/ai-dynamo/dynamo/pull/17686) 将 serving-load/KV-load 放到同连接，按 Relay replica 分开存储，从同一 fresh view 选择池，避免重连/双副本交叉拼报告。建议多 pool 部署者核对 CRD/RBAC、readiness、TTL 与 failover，不能把“两 active”当任意跨集群一致性保证。

- **NVIDIA Dynamo｜10-30，main：上轮 open 的 Decisions qualification stack 已合并。** [#15857](https://github.com/ai-dynamo/dynamo/pull/15857) 补 native 数值、wire contract、官方 OpenAI/TypeSafe 同步 SDK 的 opt-in qualification profiles，固定 SGLang0.5.19/0.5.22，需显式选择 model/server。此变更将旧 backlog 从“open”更新为“tests merged”；共享 CI 不负责部署大 Qwen3.8-27B，也不自动认证其他 runtime/config。建议关注 Decisions API 的团队按 profile 自行 provision 后跑全栈，不把测试定义当已运行的兼容性证书。

- **NVIDIA KVCR（补充发现）｜10-30，main：staging-only 集成不再强制伪造 framework pin callbacks。** [#133](https://github.com/ai-dynamo/kvcr/pull/133) 允许四个 framework pin callback 全部省略；启用 pin 时 request/poll/release 仍须成组提供，cancel 可选。关闭 pin 后 progress thread 能直接服务已有 local subset、报告缺失/不兼容源，pending deposit 仍保持等待。建议 cache 集成人员阅读 callback 校验和 reservation 状态测试；这是共享缓存接入接口变化，不是 Dynamo 全拓扑可用性的证明。

- **llm-d 社区｜10-30，main：GLM-5.2 SGLang P/D recipe 加 shared-storage HiCache tier。** [#2924](https://github.com/llm-d/llm-d/pull/2924) 将 L3 storage backend/prefetch/write policy 参数化，提供 RWX PVC 组件及 L1/L2/L3 对比 overlays，并补 GKE COS Mooncake DMA-BUF 注册相关 recipe。默认 size0/空串不启用，指定 storage 却未开 host cache 会 fail-fast。建议 Kubernetes 服务团队从 rendered Kustomize 与 mount 合并验证起步；扩展 benchmark 与 host-tier tuning 在 PR 中明确留后续，没有新三层收益保证。

### AMD、Ascend 与 TPU/Apple 后端

- **AMD AITER｜10-30，main：MI300X 的 fnuz MXFP8 MoE 路径与 SonicMoE hipBLASLt 备选。** [#6219](https://github.com/ROCm/aiter/pull/6219) 在 gfx942 避开不支持 fnuz 的 dot_scaled，按32宽K片段做 FP8 MFMA 后应用 E8M0 scale，修 fnuz max240 与64bit大权重偏移；不需为该路径把整 expert weights 解到 BF16。[#5970](https://github.com/ROCm/aiter/pull/5970) 的 `SONIC_MOE_GROUPED_GEMM_BACKEND=hipblaslt|multistream|auto` 是 opt-in，默认 Triton，blockwise FP8 仍 Triton，auto 可退回。建议 MI300/MI350 用户分别验证编码、scale、split-K 和 backend failure；作者 gfx95044通过不覆盖全部 gfx942/多 GPU 路径。

- **AMD AITER｜10-30，main：fused custom all-reduce 保留舍入与传输边界。** [#6270](https://github.com/ROCm/aiter/pull/6270) 将 residual add 或只对 latent rows 的 RMSNorm 与 AR 融合，residual 在规约舍入后相加，保留独立算子的数值次序；按 (arch,TP,bytes) 限定，row options 仅2-stage且≤80 rows。[#7155](https://github.com/ROCm/aiter/pull/7155) 允许 gfx1250 IPC 在 graph capture 时注册原 input供 peer 直接读取，省掉 D2D staging；VMM/symm_mem 保留 copy-in。作者4×gfx1250的一次graph计时改善约4–16%，非广泛统计。建议多卡 AMD 服务验证 IPC/VMM、TP尺寸和 changed-input replay，再测慢 rank 延迟。

- **AMD AITER｜10-30，main：FlyDSL paged decode 在 PV 前清除无主 V/scale，防 0×NaN 污染。** [#7092](https://github.com/ROCm/aiter/pull/7092) 在 context tail 清零 V 和 V-scale；仅把无效 K logits 设 -inf 不够，P=0仍会乘到 stale PAGE checkpoint 的非有限字节。作者 Qwen3.5-35B-A3B-FP8/ATOM 个案 GSM8K0.06→0.97是该故障恢复，不能当模型质量提升。建议 hybrid KV/state 共池实现优先读该根因和 poisoned-tail 回归，确认自己的 backend 是否也把无效 V 视为“必须有限”的调用方条件。

- **AMD AITER/ATOM｜10-30，main：gfx1250 Mega-MHC 与 Kimi K3 GEMM 适配。** [AITER#7131](https://github.com/ROCm/aiter/pull/7131) 以 TDM/LDS ring+wave32 WMMA 实现 DeepSeek-V4.1 delayed Mega-MHC seam，fused MXFP8 可输出硬件 GEMM布局或 row-major scale；[ATOM#2810](https://github.com/ROCm/ATOM/pull/2810) 按实际 `scale_row_major` 能力接入，旧 AITER 不满足就让 QKV GEMM 自量化，防读错 layout。[AITER#7142](https://github.com/ROCm/aiter/pull/7142) 给 Kimi-K3 f_b K128 与 DSpark PTPC/blockscale 补双缓冲 prologue/drain及 shape调优。建议 gfx1250集成者固定两仓版本与 scale合同复验，不把某批 shape tuning 外推到所有模型。

- **AMD ROCm Libraries｜10-30，develop：hipBLASLt skinny GEMM 与 persistent K-loop 设置。** [#13489](https://github.com/ROCm/rocm-libraries/pull/13489) 把 vLLM wvSplitK BF16/FP16 TN、n=1–5 引入 gfx950 GFA，增加 alpha/beta/C-D stride 合同；作者 MI355X约590 shapes/type、自报平均22%/26%，只代表该 tiny-N workload 与指定 baseline。[#14719](https://github.com/ROCm/rocm-libraries/pull/14719) 保留 packed StaggerU寄存器，避免persistent第二tile后丢 stride/mapping 导致相同行不同求和次序。建议 decoder tiny-batch 与 uniform-summation 用户检查 develop/package 边界并复现 cache-hot/HBM两种计时。

- **AMD ROCm Libraries｜10-30，develop：rocKE attention 调度与 hipDNN autotune 的返回输出合同。** [#14367](https://github.com/ROCm/rocm-libraries/pull/14367) 给 gfx942/950 unified split-KV/dense attention 补GQA非整除、D256 split和LLC-aware cache policy；其性能表测在较早 revision，PR 自称最终refresh仍进行中，本文不报最终倍数。[#14694](https://github.com/ROCm/rocm-libraries/pull/14694) 在autotune排名后重跑winner并同步，确保返回成功时caller buffers是winner输出，代价是额外execute+sync；in-place写输入的graph不享此保证。建议benchmark工具与provider集成人员分别验证调度适用范围和“返回完成”的语义。

- **AMD ATOM（补充发现）｜10-30，main：LMCache MP失败要及时暴露，但DMA buffer仍不能提前重用。** [#2837](https://github.com/ROCm/ATOM/pull/2837) 对future query/result异常抛 `LMCacheTransferUnprovable` 并保留lease，ModelRunner异常退出先通知frontend，缺失/失败引擎的health改503；默认transfer deadline1200→30秒、completion margin60→10秒。40秒是检测策略，不是量测SLO，也不是远端DMA已停止的证据。建议offload服务读异常/健康状态回归，验证故障可见性与资源隔离同时成立。

- **AMD ATOM（补充发现）｜10-30，main：DeepSeek-V4 prefill indexer 可按TP轮转分query rows。** [#2722](https://github.com/ROCm/ATOM/pull/2722) 将每TP rank重复做的indexer score/top-k改为每第W行分配，再all-gather rank-major选择，避免连续切块把长prefix高成本集中尾rank。作者较早revision在单节点8×MI355X、DSV4-Pro-0813/DSpark、TP8、151K/520K prefix-hit prefill测step下降15.1%/35.4%，不能直接视为最终head质量与服务结果；该split gate排除DP attention、TBO、PCP/DCP与gfx1250 OPUS等路径。建议长上下文团队看 `score_topk`、pack顺序及实际硬件/拓扑限制，复验mixed-batch负载均衡和最终输出。

- **AMD ATOM（补充发现）｜10-30，main：Kimi K3 state传输与MegaMoE recipe合约。** [#2570](https://github.com/ROCm/ATOM/pull/2570) 把KDA per-request state与paged MLA blocks分开注册，write-mode MoRIIO用同一WRITE_DONE覆盖，不能携带state就拒绝，read mode/ReplaySSM仍不支持；[#2765](https://github.com/ROCm/ATOM/pull/2765) 通过AITER MegaMoERecipe统一stage1/2、wire和geometry合法性与VMM sizing，并按依赖所有权清理arena/communicator。建议P/D与MoE集成人员固定AITER接口、验证state slot握手/完成信号，不能只比较KV blocks字节。

- **vLLM Ascend 社区｜10-30，main：native SFA DCP布局提速与PCP+DCP cache-address纠错。** [#16350](https://github.com/vllm-project/vllm-ascend/pull/16350) 把BF16 partial output与FP32 LSE按raw bits一次AllToAll，FP32 merge，大payload用有界head/feature tile，query gather推导维度，native C8写融合scale与cache发布；16K元素阈值是tile选择而非固定token上限。[#20079](https://github.com/vllm-project/vllm-ascend/pull/20079) 在metadata builder预备匹配“单份decode前缀+各rank prefill”的cache slot mapping，避免mixed batch写错地址。建议CP团队复现pure-prefill/mixed、padding和实际token数，勿把query-token T当context length。

- **vLLM Ascend 社区｜10-30，main：EP/GDN前处理省拷贝与FX fusion真正覆盖多个model prefix。** [#19515](https://github.com/vllm-project/vllm-ascend/pull/19515) 在uniform SP/DP1跳过all-gather后select+cat和reduce-scatter前identity重建；[#19954](https://github.com/vllm-project/vllm-ascend/pull/19954) 用一Triton launch构造GDN mixed/spec metadata，仍在model graph外；[#19756](https://github.com/vllm-project/vllm-ascend/pull/19756) 将pattern去重改成每matcher，并补bias-less int8 norm+quant，防第二个draft prefix完全无fusion；[#19778](https://github.com/vllm-project/vllm-ascend/pull/19778) 补shared-first/foreach-add MoE匹配并保留中间舍入与多consumer排除。建议投机解码部署者先核对实际graph fusion与metadata，再量ITL，PR明确整服务收益仍需验证。

- **vLLM Ascend 社区｜10-30，main：KV-transfer layout与A5 sparse边界收紧。** [#19824](https://github.com/vllm-project/vllm-ascend/pull/19824) 把contiguous single-local-head NHD仅以byte-equivalent HND metadata发布，原tensor/地址/stride不变，多头/歧义/非连续组合明确拒绝；[#19880](https://github.com/vllm-project/vllm-ascend/pull/19880) 单独处理GLM跨group共享physical slot，解layerwise MRV2初始化误判；[#20116](https://github.com/vllm-project/vllm-ascend/pull/20116) 在A5 QSFA不论是否需LSE均裁到有效sparse index前缀，全无效时清输出；[#20375](https://github.com/vllm-project/vllm-ascend/pull/20375) 给实际launch传入max_spec_len，避免以batch维推导draft-position网格而漏写recovered slots，warmup仍保留廉价(1,1)。建议PD/层传输/A5团队跑真实cache registration和全-1 index回归；主线修复不自动属于10-29 v0.30.0。

- **Google torch_tpu｜10-30，main：FlexAttention mods落到StableHLO/Kiln。** [#5976](https://github.com/google-pytorch/torch_tpu/pull/5976) 将score_mod与BlockMask mask_mod trace成StableHLO，再splice到每个listed KV block，使partial-block causal/sliding-window/document mask与ALiBi等可用；wrapper返回log2 LSE，block size仍须整除T/S。建议TPU attention用户验证自定义mod、两类BlockMask布局和numerics/error测试，不把forward kernel支持扩写成全部训练/反向已覆盖。

- **Google torch_tpu｜10-30，main：stream wait真正约束transfer启动，ready callback推迟到snapshot。** [#6077](https://github.com/google-pytorch/torch_tpu/pull/6077) 将H2D/D2H交给stream scheduling，wait_event依赖会阻塞后续transfer，防host提前填满仍在用的device memory；[#5833](https://github.com/google-pytorch/torch_tpu/pull/5833) 对TensorCore eager不再每op注册device-ready callback，而在EventSnapshot记录最新输出；SparseCore/compiled任务保留直接ready tracking。[#5997](https://github.com/google-pytorch/torch_tpu/pull/5997) 让XLA_FLAGS/LIBTPU_INIT_ARGS中的RANK在memoized read之前展开，避免cache key与libtpu配置不一致。建议异步pipeline用户分别测enqueue、device completion和event wait，而非用host返回当执行完成。

- **Google JAX｜10-30，main：unreduced值全面经HiJAX，collective轴校验收紧。** [#42140](https://github.com/jax-ml/jax/pull/42140)、[#42142](https://github.com/jax-ml/jax/pull/42142)、[#42170](https://github.com/jax-ml/jax/pull/42170) 将manual shard_map/unreduced collectives下沉到HiJAX，在manual边界squeeze/expand隐含轴，并在同日默认启用后删除旧config；[#42135](https://github.com/jax-ml/jax/pull/42135)、[#42169](https://github.com/jax-ml/jax/pull/42169) 对all_gather及多轴collectives要求**所有请求轴**属于相应varying/unreduced/reduced集合，避免只有intersection就放行的错误类型。建议分片训练自定义collective用户复验grad与auto_pcast=False，旧flag不应作为长期逃生开关。

- **Google JAX/OpenXLA｜10-30，main：Mosaic register allocation与低精度转换纠错。** [JAX#42160](https://github.com/jax-ml/jax/pull/42160) 只在Mosaic GPU的自包含PTX编译路径禁用nvJitLink，避免relocatable编译忽略setmaxnreg而spill；不是全XLA禁用linker。[XLA#51935](https://github.com/openxla/xla/pull/51935) 让NVIDIA float-n转换pass拒绝先将F64截成F32再转F8/F4，以免double rounding，交后续合法lowering处理。建议warp-specialized kernel与量化数值测试读IR/regression；本轮没有独立profile证明spill或误差幅度。

- **OpenXLA 社区｜10-30，main：Triton scan先撤回再以CUDA门槛重新默认启用。** [#51913](https://github.com/openxla/xla/pull/51913) 撤回昨日`6a3fedfa`的默认开关；随后 [#51933](https://github.com/openxla/xla/pull/51933) 再启用并明确只让CUDA跑ScanRewriterTriton，ROCm仍走AssociativeScanRewriter/ScanExpander。不能把中间revert当cutoff最终状态，也不能说ROCm已默认Triton scan。[#51951](https://github.com/openxla/xla/pull/51951) 另外避免symmetric-memory ragged-all-to-all仍分配signal buffers。建议compiler/collective用户固定最终commit检查HLO fusion/backend选择，不仅看flag值。

- **Apple MLX｜10-30，main：JACCL断链从崩溃/挂住变成持久失败。** [#4767](https://github.com/ml-explore/mlx/pull/4767) 检查completion状态，stall时周期query QP，group共享failure latch，等待所有wires结束后重抛错误；失败group后续调用仍抛，应用须teardown。作者两台Mac/Thunderbolt5拔线9/9次两rank在300ms内报错是特定测试，非所有RDMA拓扑SLA。建议Mac多机用户复现断链与group重建流程，确认没有把这当自动恢复/重试功能。

### 仅观察支线，不能写成主线交付

- **AMD AITER｜10-30，支线 `zhiding512/gfx1250-pa-merge-partitions`：** [#7147](https://github.com/ROCm/aiter/pull/7147) 修大K-scale/logits下softmax/V-normalization次序及offset scratch的64bit atomic对齐，目标是#7079的feature branch。值得有gfx1250 PA路径的用户观察后续main merge；本轮不把它与main的#7092混成同一个已发布修复。
- **AMD ROCm Libraries｜10-30，支线 `users/khuagarw/rocke/aick-2712-cpp-parity-1-stopgaps`：** [#14523](https://github.com/ROCm/rocm-libraries/pull/14523)、[#14524](https://github.com/ROCm/rocm-libraries/pull/14524)、[#14525](https://github.com/ROCm/rocm-libraries/pull/14525)、[#14526](https://github.com/ROCm/rocm-libraries/pull/14526)、[#14527](https://github.com/ROCm/rocm-libraries/pull/14527)、[#14528](https://github.com/ROCm/rocm-libraries/pull/14528)、[#14758](https://github.com/ROCm/rocm-libraries/pull/14758)、[#14759](https://github.com/ROCm/rocm-libraries/pull/14759) 合并一组rocKE C++/Python unified-attention parity栈，含2D/3D routing、D256、sliced-K/i64 offsets、GQA与LSE/mask-band。这是可读的porting/parity设计，但默认develop本轮不能仅凭这些子PR认定全栈交付；继续核验umbrella和默认分支可达性。
- **AMD ATOM（补充发现）｜10-31 00:52:53Z，支线 `alizaidy/gfx1250-qwen-support`：** [#2869](https://github.com/ROCm/ATOM/pull/2869) 追踪PAGE checkpoint释放后遗留NaN bytes进入KV tail、导致Qwen3.5接受率崩溃的传播链，加入released checkpoint units清零等防线；与AITER#7092相关但不是同一主线事件。建议hybrid-state缓存作者读故障链与清零时序，等待main集成和实际attention backend验证。

覆盖说明：31个原监控仓库的releases、回溯merged-PR列表、默认分支commits均分页枚举，另补ATOM/KVCR/Helion；**枚举完成不等于语义全覆盖**。未读完的候选、普通维护的过滤理由、旧支线/依赖/qualification缺口均在本轮ledger保留；无本节条目不代表对应仓库没有实质更新。

<a id="hardware"></a>
## 硬件软件协同与部署验证

### AICR：把 Dynamo 下方的集群依赖、配方和验证结果一起版本化

**NVIDIA；10-30新技术文章。** [官方说明](https://developer.nvidia.com/blog/build-a-validated-kubernetes-runtime-for-nvidia-dynamo/)，静态核验基线[`NVIDIA/aicr@db626d1b`](https://github.com/NVIDIA/aicr/tree/db626d1bcd2f3040c621ba4517ed660b16c6ab72)。文章实例为AICR1.0.0+Dynamo chart1.4.2，不能与main任意新组件混搭。

价值在于把“模型服务pod能起”扩为可重建的基础设施验收：snapshot识别集群→选带版本/约束的recipe→bundle→部署→deployment/conformance检查，通过后才装应用；性能验证另起临时负载、占用GPU。文章区分Dynamo Frontend自行转发与GAIE/EPP只选端点、Gateway实际转发；1.4.2请求平面用TCP、KV事件用ZMQ，默认无NATS并不等于没有事件面。1.5的原生Rust EPP在本文属于展望，不写成1.4.2能力。

已读`docs/user/validation.md`和GB300 EKS recipe：离线`--no-cluster`只判声明式约束，行为检查会skipped，不能把它记成集群运行验证通过；recipe还对Kubernetes版本、gang scheduling和负载给出具体约束。main中的GB300验证把并发门槛降至128/GPU，是为了避开单frontend瓶颈；配方注释中的历史测量不是本日新的GPU对比结论。**建议Kubernetes推理团队深入阅读并考虑将recipe和原始验证报告归档**；实际集群仍需独立跑行为与性能验收，本轮未部署。

### gds-diag：为训练数据/检查点存储路径补上确定性诊断入口

**NVIDIA；10-30发布教程，工具代码基线[`MagnumIO@4eb93247`](https://github.com/NVIDIA/MagnumIO/tree/4eb93247fac9cd7a5cda60f1601d3d2499b1c259/gds-diag)最后commit为10-09，明确是新说明、旧工具补录。** [官方教程](https://developer.nvidia.com/blog/how-to-check-and-configure-nvidia-gpudirect-storage-for-your-platform/)。

CLI把host readiness、已装cuFile/GDS栈、指定mount、容器可见性和config审查拆成可复跑检查，区分native nvidia-fs、PCIe P2PDMA/Grace C2C、RDMA与CPU bounce fallback，输出JSON和PASS/WARN/FAIL。`pre-install`不要求先有GDS；大部分其他检查需Linux/Python3.8+和相应GDS环境。源码README/CHANGELOG还把device-mapper路径标为native/P2P不支持，不能因磁盘是NVMe就认定走直通。

其意义是先定位数据/检查点I/O没有走预期路径的原因，**诊断通过不是带宽/训练吞吐基准**，没有本次LLM提速数。CLI提供缓解建议而不自动修改系统；文章可选agent层不替代确定性检查。**建议有GDS装载/容器问题时考虑采用检查脚本并归档JSON**，再用真实存储和训练负载验证；本轮未运行设备诊断。

<a id="coverage"></a>
## 实际覆盖、补录与未完成区间

- **本地/远端状态分账。** 24篇旧日报和索引已核对远端；本轮正文先在本地保存，远端必须完成固定commit内容回读才能记录上传成功。运行记录、稳定事件和来源证据保存在本地`.monitor/research-2026-10-31/`，不会上传到其他仓库。
- **核心六仓。** 实读release、merged搜索和默认分支commit，回溯搜索返回858条PR、默认分支838条commit；本轮正文136个main已集成PR、87个机制组。前轮398条未审队列、本轮285条pending及更早区间仍保留，枚举与选定diff审阅不等于穷尽kernel或外部依赖审计。主窗没有在该release检查面发现新release；不能扩写为项目没有发布任何内容。
- **扩展工业工程。** 31个既有仓加ROCm/ATOM、ai-dynamo/kvcr、pytorch/helion，共34仓、136个成功surface页；枚举2444个PR、2661个默认分支commit、258个release对象。正文123个稳定事件中95为默认分支merged PR、16为PyTorch直接落地commit、1正式release、1 prerelease、10个明确标为支线的观察项。119个候选PR元数据及全部changed-file页被读取，语义深读只覆盖所选关键代码；4406当前候选和4424继承记录仍未完成语义审阅，二者及不同队列可能相交，不能相加成独立成果数。release仅检查最新对象至回溯边界，FlashInfer rc正文自身有截断，旧release编辑历史没有完整闭合。
- **模型与厂商。** 47个既有HF组织加Baselabs、Apex、ServiceNow-AI、xiaomi-research、Octen，共52组织，models/datasets×创建/修改排序的208条查询在4次TLS重试后均到回溯下界，发现290个唯一model/dataset对象。212个listing与上轮相同，继承其原审阅状态；23个模型revision在本轮正文中讨论，另有按论文归属合并的工件；7个访问/revision失败和25个新待语义审候选留存。取得卡不代表权重张量、完整代码、所有数据行均已验收；GGUF的config.json404不按整工件不可用处理。
- **官网与硬件。** 检查63个厂商/研究/硬件入口，53取得快照、10失败：OpenAI部分news/research/publications入口403、MS Research403、Kimi308、xAI news403、Reflection safety超时、Dory报告403、RecursiveSI TLS。个别OpenAI系统卡、Mage GitHub等可访问不关闭上述网站整体缺口。Qwen/腾讯/百度/阶跃/昇腾等动态正文与历史仍不完整。NVIDIA/AMD博客、Gaudi、TPU、Neuron及NKI入口实际打开；AMD既有RL/EAGLE教程去重不重报。CANN/HCCL/libtpu及较老组件历史仍待查，不能从快照相同推断厂商没有更新。
- **补录与更正。** Joey、Apex、Apriel按实物上传原日期补录，未把占位仓建立日当发布日；Octen、UltraX、GradPower明确旧作补查。Dory显式cache配置、LFM3原评测参数、Beam卡分数修订各只记本次增量。DeepSpeed8840以最终整分区实现为准，TE3552披露已知GPU失败；Trident与SenseNova保留卡/论文/工件冲突。对既有日报正文未作无依据改写。

## 论文覆盖与仍未闭合的缺口

- 24篇历史日报建立497个work／327个显式版本去重；当前7个arXiv类别公告已恢复到10月30日，合计2029行／1466 unique works，其中replacement761行／534 works。HF10月27—30接口分别50／50／17／39条，10月31仍400。公告恢复只关闭陈旧入口的访问问题。
- 主窗submitted查询超时；单独日期查询虽200/0，返回标题把范围规范化成引号。真实updated查询仍被改成submittedDate并返回0，不能据此声明没有新稿或修订。72小时回溯分页收到1312条，旧失败28—30日类别查询收到753条，均作为发现库存保存；尚未证明服务端范围语义与所有精确区间完备。
- OpenReview API与3个针对性PDF仍403；TMLR比10月27列表新增26个forum ID，但只知月份，正文／确切发布日期未恢复。ACL／PMLR本轮补读两篇正式PDF，未枚举全部volume历史与版本。
- 本轮固定版本全文读审、实现源读与论文E2E结论的层级分开记录。小尺度Dynamic Wave全文已读，因10.94M TinyStories单epoch、compile不对称、无增量state-cache且相对fused attention无显存优势，本轮未推荐其为LLM架构进展。未审发现与旧replacement383／657／594／292等相交队列保留；不相加，也不宣称全局完整覆盖。

本轮按稳定标识记录308个讨论/观察事件，包含同一工作的多种工件和明确未进入主线的支线项，**不等于308项独立研究成果**。跨分支去重、日期/版本检查、关键一手证据交叉抽查与发布前修正已记录；没有完成所有历史区间或全生态语义监控，`last_fully_covered_global_window_end`继续为null。下轮从本次cutoff续扫并回溯72小时，同时优先处理旧失败、待审候选与任何待上传文件。
