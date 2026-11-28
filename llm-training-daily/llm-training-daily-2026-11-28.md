# LLM 技术动态日报 · 2026-11-28

> 时区 **Asia/Shanghai（UTC+8）**。增量扫描窗口 **(2026-11-27 08:58:22, 2026-11-28 08:59:46]**；回溯72小时至 **2026-11-24 08:58:22**，同时继承此前未完成区间。实际生成 **2026-11-28T09:16:41+08:00**。本轮先读取运行记录、本地50篇历史日报和GitHub归档，按论文版本、模型快照及仓库合入提交去重。本文为选择性技术监控；目录获取完整不等于候选全文已经全部审查。

LLM为大语言模型，MoE为混合专家，KV为键值缓存，SFT为监督微调，RL为强化学习，RLVR为可验证奖励强化学习，GRPO为组相对策略优化，MTP为多词元预测。TP／PP／DP／CP／EP分别为张量／流水线／数据／上下文／专家并行，PCP／DCP为预填充／解码上下文并行；P/D为预填充与解码拆分，FSDP为全分片数据并行。TTFT为首词元时延，TPOT为每输出词元时延，HBM为高带宽显存，LDS为设备片上共享存储；MLA为多头潜在注意力，DSA为DeepSeek稀疏注意力，GDN为门控DeltaNet。GEMM为通用矩阵乘法，RMSNorm为均方根归一化，RDMA为远程直接内存访问，IPC为进程间通信，JIT为即时编译；BF／FP后的数字表示浮点位宽。数值均归于作者／项目方，未标独立复现；本轮没有独立复训、设备压测或付费API实测。

## 本次重要变化

- **GigaChat 4值得读系统配方。**83页报告公开混合GDN／DSA、长窗转换、局部稳定性控制、FP8分片聚合及多教师蒸馏；220K实验注意力核快4.32倍，整GPU步快1.68倍，10K反慢14.5%。708B与37B的许可、优化器和交付条件不同，vLLM集成仍是开放PR。[固定技术报告](https://github.com/salute-developers/gigachat4/blob/81b0dd849dcbbd4c2f909214386c7e5e90680327/GigaChat_4.pdf)
- **训练方法要先给普通基线同样的选参目标。**SFT新论文在联合优化新能力与保留能力后缩小与自蒸馏的差距，但MBE等任务仍明显偏向自蒸馏；BReD压缩持久优化器状态也有吞吐代价。SSA的400M实验保留检索亮点与预填充／语言建模负结果。[SFT选参](https://arxiv.org/html/2611.34579v1) · [BReD](https://arxiv.org/html/2611.33951v1) · [SSA](https://arxiv.org/html/2611.33625v1)
- **投机与权重同步继续减少冗余状态。**DFly2共用目标KV与稀疏索引，三种草稿和训练入口已交付，通用vLLM服务支持仍未完成；verl sharded_nixl减少大MoE聚合与无用传输，但小模型实测反慢，异步失败门禁也有限制。[DFly2](https://arxiv.org/html/2611.34776v1) · [verl #8395](https://github.com/verl-project/verl/pull/8395)
- **弹性与缓存优化需同时验收停顿和正确性。**SGLang在线缩容默认重捕图和可选图复用代价不同；OpenRLHF修复工具观测插入后的优势／折扣错误，Gaudi修复坏KV加载后仍输出与异步回滚污染。[SGLang #33111](https://github.com/sgl-project/sglang/pull/33111) · [OpenRLHF #1430](https://github.com/OpenRLHF/OpenRLHF/pull/1430) · [Gaudi #2038](https://github.com/vllm-project/vllm-gaudi/pull/2038)
- **新工件开放程度和复验范围须分开。**Innata-0发布理解、世界预测和动作权重，但部分新路径闭环仍未完成；Parse新增三份合成数据与适配配方，其开发checkpoint的结构和完成性改善、低并发投机倍数都保留原评测条件。[Innata固定交付](https://github.com/baai-eai/innata-framework-inference/tree/d49cc178d4b2067aed3a954d646c89601f10c998) · [Parse官方说明](https://huggingface.co/blog/nvidia/building-on-nemotron-35-parse-open-data-fine-tunin)
- **硬件收益不能越过证据边界。**Firefly是显式实验NVFP4 MoE后端，核收益伴随工作区与非逐位等价；昇腾固定KV、接近显存边界的prefill收益不外推到所有负载；AITER、FlyDSL和Blackwell同步修复分别保留编译验证与真机故障证据的区别。[Firefly](https://github.com/flashinfer-ai/flashinfer/pull/9995) · [昇腾 #22983](https://github.com/vllm-project/vllm-ascend/pull/22983)

[模型与技术报告](#topic-1) · [预训练与架构](#topic-2) · [后训练](#topic-3) · [推理与部署](#topic-4) · [开源工程](#topic-5) · [硬件软件协同](#topic-6) · [评估与可靠性](#topic-7) · [覆盖与续查](#coverage)

<a id="topic-1"></a>
## 模型与技术报告

### GigaChat 4：开放混合稀疏模型与中间 checkpoint，系统报告的价值高于单项榜单

**机构／日期／版本：GigaChat Team；技术报告随11月26日仓库提交 `81b0dd849dcb` 公开，权重历史同日标记发布；本期补录报告与工件，27日增加聊天模板修订。** 两档为708B总参数／41B激活和37B／4B，分别64／32层，按3∶1交错门控DeltaNet（GDN）与带DeepSeek稀疏注意力（DSA）的多头潜在注意力（MLA），每次稀疏选择2,048项；最终base与instruct配置支持1,048,576词元。公开FP8、BF16、GGUF及中间训练checkpoint，但中间分支有4K／32K配置，不能全部当1M模型。[固定报告](https://github.com/salute-developers/gigachat4/blob/81b0dd849dcbbd4c2f909214386c7e5e90680327/GigaChat_4.pdf) · [固定708B卡](https://huggingface.co/ai-sage/GigaChat4-708B-A41B/blob/57f5a39960c14d83c65a5d914aefe250a290bd20/README.md) · [固定37B配置](https://huggingface.co/ai-sage/GigaChat4-37B-A4B/blob/f39b9385f2cf78446d88ef035edf2481231c92c9/config.json)

83页报告主要解释708B训练：约15.35T词元语料池、按重复因子采样的约20T训练目标，末1T用于更重合成数据的mid-training；AdamW、逐层残差预热、局部专家负载触发的学习率缩放及分投影Z-loss共同控制稳定性。扩窗先做约336B词元的128K稠密适配，再冻结骨干训练indexer，最后约400B词元联合训练到1M。后训练先训领域RL专家，再以多教师on-policy distillation融合即时／推理模式；37B卡另述其Muon预训练，不能把两档优化器配方混为一体。报告披露数据构建、训练监控及消融，但未提供完整语料和可直接重跑全部训练的交付。

系统部分尤其值得训练团队**深入阅读并考虑定向复现**：跨分片边界重建128×128量化块尺度，专家参数直接FP8 all-gather，保持FP32主权重；两分块专家计算／通信重叠与激活卸载互有约束。报告明确CPU复制即使在独立流也会拖慢DeepEP，因此调整复制窗口；约20%峰值显存下降的FP8检查点存储没有与两分块调度同时使用，不能叠加收益。推理附录H.4在220K输入配置中对分块prefill各步取平均，累计注意力核时间433.38→100.33ms（4.32倍），整GPU步825.77→492.16ms（1.68倍）；10K时稀疏路径反慢14.5%。另一220K输入／1,000输出的解码融合实验中，累计注意力核时间下降30.3%，整步只下降3.9%。这些是报告内不同实验，不能拼成统一端到端收益；跨代吞吐图为四GPU、TP4／EP4／注意力DP4、两阶段分测，缺少足够硬件型号条件，不作跨平台成本比较。[报告§2.5、§4、§5.3及附录H.4](https://github.com/salute-developers/gigachat4/blob/81b0dd849dcbbd4c2f909214386c7e5e90680327/GigaChat_4.pdf)

**采用边界：**708B权重是MIT；37B使用“Noncommercial License for Individuals 1.0”，条文还明确禁止所列教育／公共研究等组织使用，不能因根仓MIT或“开放权重”而按同一许可采用。部署示例固定SGLang `ff142897935e`；vLLM示例来自开放的#69762 head `c0c90837262b`，本次回读尚未合入，不能写成正式发行版原生支持。榜单含7小时任务超时等设置，Terminal-Bench2.0的71.26%并非与所有公开短超时榜直接可比。优先复现长窗交叉点、卸载和通信竞争及同预算蒸馏，采用前分别核权重许可和服务commit。[708B许可](https://huggingface.co/ai-sage/GigaChat4-708B-A41B/blob/57f5a39960c14d83c65a5d914aefe250a290bd20/LICENSE.txt) · [37B许可](https://huggingface.co/ai-sage/GigaChat4-37B-A4B/blob/f39b9385f2cf78446d88ef035edf2481231c92c9/LICENSE.txt) · [固定部署说明](https://github.com/salute-developers/gigachat4/blob/81b0dd849dcbbd4c2f909214386c7e5e90680327/README.md) · [vLLM开放PR](https://github.com/vllm-project/vllm/pull/69762)

### Innata-0：理解、世界预测和动作模型交付，须区分论文成绩与新发布路径的复验范围

**机构／日期／版本：北京智源研究院（BAAI）；报告 `2611.30254v1` 提交11月23日，本期补录；11月27日发布Apache2.0 checkpoint与推理候选版，代码 `d49cc178d4b2`。建议多模态与具身团队深入阅读，考虑复现接口和受控闭环。** 论文用三塔Mixture-of-Transformers衔接理解、未来视觉预测和动作／触觉，以完整任务、当前子任务和已验证历史为共同条件，让Agent的记忆、验证、恢复与训练标注一致。新交付分开提供单塔Understanding（约8.77B）、双塔动作条件世界模型ACWM（约15.75B）及RoboCasa／RoboDojo三塔动作模型，不能把单塔问答权重当完整机器人策略。[原始报告](https://arxiv.org/html/2611.30254v1) · [Understanding固定卡](https://huggingface.co/BAAI/Innata-0-Understanding/blob/726ef0ebe8d4baf07b512bb4d58a2893e6505ce8/README.md) · [ACWM固定卡](https://huggingface.co/BAAI/Innata-0-ACWM/blob/90cfcd3df899bb9d545d1243b6cb35d598c34312/README.md)

论文WorldArena2.0分数67.63、RoboCasa365总成功率60.2%是原实验口径。发布卡坦率给出较窄的新路径证据：理解模型18指标均重测、只有4项在记录精度上完全相同，最大差0.6993个百分点；ACWM已生成1,000集、15指标只完成8项评分，新的总分尚未计算；RoboCasa同机代表性请求的动作逐元素相同，**完整闭环评测仍未完成**。发布29k handoff还按Atomic／Composite任务使用不同采样shift，评测文档区分了论文比较所用checkpoint／设置，不能把一次输出相同推广为整套论文已复现。[RoboCasa固定卡](https://huggingface.co/BAAI/Innata-0-WAM-RoboCasa365/blob/b5fbe70a7647cdcee071cc9ee545232ec9b92879/README.md) · [冻结评测协议](https://github.com/baai-eai/innata-framework-inference/blob/d49cc178d4b2067aed3a954d646c89601f10c998/docs/release-evaluation.md)

输入契约也是复现条件：ACWM使用736×544初始图像、16Hz、每帧20维归一化动作；RoboCasa三相机加16维state，输出16×12动作、20Hz，不能互换控制器。推理仓标release candidate，依赖外部Wan2.2 VAE、匹配模拟器／资产／评测器及CUDA13环境，通用legacy domain路径不等于四个发布模板。当前证据支持研究复现入口；跨机器人迁移、完整新路径成功率和训练重现仍应继续观察。[固定推理说明与适用范围](https://github.com/baai-eai/innata-framework-inference/blob/d49cc178d4b2067aed3a954d646c89601f10c998/README.md)

<a id="topic-2"></a>
## 预训练与架构

<!-- item:2611.33625|预训练与架构 -->
### State Space Attention：学会留下哪些上下文，400M 仍未证明更好的准确率／成本（补录）

**机构／日期／版本**：Harvard Pehlevan 研究组；2026-11-25，2611.33625v1。[论文 §3–5、Table 1–2](https://arxiv.org/html/2611.33625v1)。

SSA 用可训练评分器保留固定数量的历史块，再用普通注意力读取；被淘汰块的反事实 recovery 为评分器提供训练信号。约400M、15B SlimPajama 词元、576可读词元预算下，后训练后8K MKVT 为96.57%，同预算滑窗 Transformer 为8.13%；但 RULER 为56.40%，仍低于8K Transformer 的76.86%。限制很实质：400M只有单种子，复跑有30%对90%的波动；2K外推8K语言建模退化，预填充慢2–6倍，训练也更慢，论文明确未证明400M更好的准确率／成本。**建议深入阅读并考虑小规模复现**：价值是学习式记忆保留及失败分析，采用前应同时核检索、语言建模、预填充和长输出成本。实现入口已给出，本轮未审代码。

<!-- item:2611.33951|预训练与架构 -->
### BReD：重放量化扰动以控制优化器状态误差，节省的是持久状态（补录）

**机构／日期／版本**：港中大（深圳）、深圳大数据研究院、华为等；2026-11-25，2611.33951v1。[论文 §3–7、Table 2–3、Appendix C](https://arxiv.org/html/2611.33951v1)。

指数移动平均（EMA）状态反复量化会累计偏差；BReD 在写入和读出时重放同一块级扰动，减少偏差而避免逐元素随机数成本。真正打包的4bit状态已在120M五种优化器、1.1B／3.4B AdamW与Muon上评估；3.4B AdamW 持久优化器状态25.53→3.58GB，但端到端吞吐0.101→0.096M词元／秒。不同规模吞吐损失5.1%–16.0%，不是峰值显存下降86%；文中未清晰给出这组计时的硬件条件，不能跨系统比较。3bit在已测设置收敛但质量偏移更大；最大预训练3.4B，较长100B词元实验只有单次运行，也未对齐Adafactor等。**建议考虑复现4bit默认点**，先验证状态压缩能否换取更大批次，再看真实吞吐。

<!-- item:2611.33468|预训练与架构 -->
### Preference write-back：把深循环生成的答案写回共享块，收益依赖输入注入（补录）

**机构／日期／版本**：Wisconsin–Madison、Microsoft；2026-11-24，2611.33468v1。[论文 §2–3、Table 1／3／4、Appendix C](https://arxiv.org/html/2611.33468v1)。

从 OLMo-2-1B 改造成循环模型、继续训练50B Dolmino词元后，用深循环贪心轨迹做单次循环的完成段交叉熵训练，只更新共享块，无正确性过滤。四循环模型 GSM8K 从56.18%→62.70%，单循环从40.86%→48.52%；同一目标用四循环训练仅得56.86%。但单循环训练优势在移除输入注入的匹配模型上消失；非循环控制匹配数据／词元，未匹配完整循环算力，部分设置优化器也不同，不能把对控制组的相对增益全归于 write-back。**建议深入阅读、考虑复现固定教师轨迹的训练深度对照**；它支持共享计算可被浅层更新改善，而不是证明更多循环免费。

<!-- item:2611.34707|预训练与架构 -->
### Mismatch-corrected bandits：小代理的数据排名需要大模型配对审计（补录）

**机构／日期／版本**：Orange Research、École Polytechnique；2026-11-25，2611.34707v1。[论文 §3–7、Table 1–2](https://arxiv.org/html/2611.34707v1)。

用小模型选数据时，各数据干预在大小模型上的收益排名可能不同；方法以少量同更新配对观测估计逐臂残差，并把审计预算分散到训练期间。CivilComments-WILDS上以BERT-Small指导BERT-Large，H64、10个配对种子、同48次目标观测，最差群体准确率65.63%→68.15%，差2.52个百分点，95%区间[0.96,4.28]。这是分类概念验证：目标模型每块都训练，预算限制目标收益信息而非算力；还未对比DoReMi／DoGE或生成式LLM数据混合。**建议深入阅读，继续观察LLM验证**；可借鉴的是审计代理失配和时间漂移，不宜直接采用其节算结论。

<a id="topic-3"></a>
## 后训练

### Nemotron 3.5 Parse：新增开放合成数据与适配配方，GRPO改善开发检查点的结构和完成性

**机构／日期／版本：NVIDIA；11月26日官方说明及公开数据，本期按原日期补录。** 本次增量是3份训练数据、SFT→RLVR适配入口与DFlash2草稿，不把此前Parse3.5模型重算首发：Charts含9,980,000对图像／标注，Tables含1,000,000对（同时带HTML和LaTeX样式表格），多语OCR含7,907,809对、149个语言／文字条目覆盖28种文字系统。均以渲染源同步生成结构化真值，当前卡标CC BY4.0；许可与开放访问在26日更新，不以12日占位仓创建日断言当时已可下载。[官方增量说明](https://huggingface.co/blog/nvidia/building-on-nemotron-35-parse-open-data-fine-tunin) · [Charts固定卡](https://huggingface.co/datasets/nvidia/Synthetic-Charts-V1/blob/9f1db4f54d230fbc19ca3da178692bad15a3dcf9/README.md) · [Tables固定卡](https://huggingface.co/datasets/nvidia/Synthetic-Tables-V1/blob/f9770ad231601ed847e62a8909e7f96d58bcc077/README.md) · [OCR固定卡](https://huggingface.co/datasets/nvidia/OCR-Synthetic-Multilingual-v2/blob/9d842c74647e2c56b3c2481d210907fefca7c938/README.md)

在相同2,078页ParseBench、相同评分／解码条件的**开发checkpoint**对照中，SFT→GRPO总分66.24→71.38，重复标记页126→29、达到9,000词元上限185→35；重复检测不是一般幻觉率，也不是最终发布权重榜单。牙科适配的256张同生成器留出页、每页8采样估计pass@1为95.85%→96.58%，不代表从8次尝试择优的成功率；牙科奖励检查精确内容、零归一化可见文字错误与自然结束，不保证框、块类别或标记正确，更不能外推真实未见版式。配方明确交接数据与模型哈希，SFT全模型、RLVR解码器GRPO，当前不实现优化器状态续训；定制业务还需替换数据、奖励和评测。**建议优先深入阅读并小规模复现**，先用独立真实模板验证收益。[固定测量说明](https://huggingface.co/datasets/nvidia/Nemotron-Parse-Blog-Assets/blob/2588319/measurement-notes.md) · [固定配方与限制](https://github.com/NVIDIA-NeMo/Nemotron/blob/4bdc0d7bcf304675232955208393273b2fdd0e76/docs/nemotron/parse/workflow.md)

推理数据同样保留负载条件：单H200、BF16、相同2,078页、8,744输出上限、原生单遍，DFlash2对同版关闭投机时并发1为7.62倍、并发512为1.52倍；后者13.48→20.44页/秒。计时含预处理／prefill／decode与补位排空，排除启动、加载图像、warmup、序列化和HTTP，各并发择优固定草稿长度；内容与输出长度可不同，不能把低并发7.62倍当通用在线收益。多遍裁剪的质量改善属于另加请求的预算。[性能与计时协议](https://huggingface.co/datasets/nvidia/Nemotron-Parse-Blog-Assets/blob/2588319/measurement-notes.md)

<!-- item:2611.34579|后训练 -->
### SFT 保留能力的结论要先对齐选参目标：同策略采样不是充分解释（补录）

**机构／日期／版本**：TTIC、Apple；2026-11-25，2611.34579v1。[论文 §4–7、Table 1–2、Appendix B／D](https://arxiv.org/html/2611.34579v1)。

作者把仅最大化新任务得分改成同时优化收益与旧能力保留，重做监督微调（SFT）与自蒸馏微调（SDFT）的学习率／批次扫描。Qwen2.5-7B、2epoch、三种子、单B200 BF16：旧能力损失≤2个百分点时，Science新任务增益SFT为34.6±1.2，SDFT为31.2±1.8；但MBE允许损失3个百分点时，SDFT增益18.5±1.9，SFT仅7.3±0.3，不能说两者处处等价。14B有同方向验证，SDFT约1.3倍端到端／1.9倍训练成本且需额外EMA参数扫描。**建议优先深入阅读并复现保留能力曲线**：上线新算法前，应先给普通SFT同样的验证目标和选参预算；均分保留也不保证每项能力不降。

<!-- item:2611.34777|后训练 -->
### SPLICE：全失败组改学局部可续接选项，额外续跑仍有成本（补录）

**机构／日期／版本**：上海交通大学、上海创智学院、京东；2026-11-25，2611.34777v1。[论文 §4–6、Table 1、Appendix D](https://arxiv.org/html/2611.34777v1)。

可验证奖励强化学习（RLVR）的整组零奖励不给相对优势信号；SPLICE只在全失败组调用提议模型，从失败轨迹中选接点、给3个短选项，每项再由当前策略续跑3次，用正价值加权局部监督，其他组继续GRPO。相同3200任务子集、Qwen3-8B冷启动点，SWE-bench Verified为15.9±0.7→25.7±0.8；Qwen3.5-9B为58.4±0.4→61.2±1.0。训练总32张H200，SPLICE把一节点给122B-A10B提议模型；它对齐可用任务而非计算量。**建议考虑复现全失败任务子集**；必须统计提议模型和环境续跑的总成本，3次续跑价值估计也有采样误差。

<!-- item:2611.33648|后训练 -->
### PRIOR：保留工具能力要看调用前推理，调用率不是最终目标（补录）

**机构／日期／版本**：PRIOR论文作者团队；2026-11-25，2611.33648v1。[论文 §3–4、Table 1–2](https://arxiv.org/html/2611.33648v1)。

Qwen2.5-VL-7B、每次两H200，在近似匹配词元预算的浅／中／深工具调用SFT数据后接同一RL配方：最终答题准确率接近，但深调用组工具启动率50.60%→5.38%。从该组按每次调用前推理长度筛样、用更多epoch近似匹配累计SFT词元，LF25最终工具开启准确率66.24%→69.75%，同预算匹配随机筛选为68.39%；纯总推理长度筛选为65.15%。结果不意味着推理越长越好，也未证明工具启动率稳定增加，数据难度等混杂仍需留意。**建议深入阅读并考虑复现筛选对照**：同时报告工具开／关正确率与调用后成功，而不是优化调用次数。

<!-- item:2611.34703|后训练 -->
### Maco-RL：异步多任务每批配额混合，配合新鲜轨迹与难度缓冲（补录）

**机构／日期／版本**：中科院自动化所、华为；2026-11-25，2611.34703v1。[论文 §3–4、Table 1、Appendix B](https://arxiv.org/html/2611.34703v1)。

方法为不同任务分配异步rollout槽位，但每次学习批都满足逐任务配额，超出策略版本滞后的轨迹丢弃；Tri-Buffer按未解／部分掌握／旧任务分配采样，并对旧题重新生成当前策略轨迹。Qwen3.8-27B在统一作者框架下，SWE-bench Verified70.2%→79.0%，Terminal-Bench2.1为68.9%→72.3%；交替单任务更新消融出现奖励崩溃，混合批更稳定。前沿闭源行有外部排行榜来源，不能当同条件比较；跨任务数据效率对照也不等于同总算力。**建议深入阅读并考虑工程复现配额与滞后门禁**，先审长尾任务供给速度和丢弃率。

<!-- item:2611.33670|后训练 -->
### 扩散语言模型蒸馏：更多教师样本未必更好，最终PPL与覆盖率仍有取舍（补录）

**机构／日期／版本**：UCL、HKUST、北大、MediaTek Research等；2026-11-25，2611.33670v1。[论文 §3–5、Table 1–4、Appendix B／F](https://arxiv.org/html/2611.33670v1)。

在掩码扩散语言模型上，静态教师样本的前向KL蒸馏可随数据增加而退化；工作先做离线蒸馏，再在学生反向轨迹上监督，加入教师置信度权重和信赖域锚。Dream-7B→93M、256长度时，Qwen-2.5评估器的生成PPL从25.23→19.83，但仍高于教师12.07；MAUVE从离线0.6034升至0.6677，却低于只加轨迹监督的0.7643。这里是外部模型对生成文本的PPL，不能说学生真实似然超越教师。只测同架构家族／分词器，Dream总1827 H100 GPU小时且新增多状态教师查询。**建议深入阅读、考虑复现消融**，应同时检查重复、覆盖和推理成本。

<!-- item:2611.34440|后训练 -->
### vSFT：把失败反馈编成LoRA，先付超网络元训练成本（补录）

**机构／日期／版本**：高德／阿里、深港河套学院、同济等；2026-11-25，2611.34440v1。[论文 §3–4、Table 2–4](https://arxiv.org/html/2611.34440v1)。

监督者从失败轨迹写自然语言反馈，421M可训练参数的超网络把反馈转成作用于MLP下投影的rank-8 LoRA，直接更新代理权重；未经相应元训练的预训练超网络几乎零成功。Qwen3-4B在WebShop最多10次试错时，Reflexion成功率30.8%、vSFT49.0%；但vSFT单次仅24.45%，GRPO44.2%、GiGPO50.0%，不能写为全面超过RL。强监督者和不同试次设置需分开看。**建议深入阅读并继续观察跨环境泛化**：这是训练好的反馈到参数映射，采用前要计入元训练、监督者调用和反复尝试预算，本轮未核实现。

<!-- item:2611.34635|后训练 -->
### ZeroHero：小代理从专门轨迹起步，语言对齐也可能损失任务成功率（补录）

**机构／日期／版本**：Cognitive AI Systems Lab、MIRIAI；2026-11-25，2611.34635v1。[论文 §3–5、Figure 5、Table 1](https://arxiv.org/html/2611.34635v1)。

每个环境单独从零训练5M／10M／50M代理，先模仿大模型成功轨迹，再做近端策略优化（PPO）；PPO-RA在RL中加教师语言锚和辅助监督。匹配交互预算时，WebShop末训练区间成功率50.0%→67.2%，但采用的是1000商品合成目录且仍低于教师70%；Alfworld-Dialogue验证集反而93.3%→83.3%。Crafter语言重复减少不等于解释语义正确，重复探针的检查点预算也不同。**建议考虑专用小环境复现，继续观察通用性**；它证明受限任务的小参数可行性，未证明少于50M能替代通用代理。

<!-- item:2611.33183|后训练 -->
### REI／SPOD：从仓库演化训练提案，类别多样性与真执行效果须分开（补录）

**机构／日期／版本**：阿里、武汉大学、MBZUAI等；原始提交2026-10-28，2611.33183v1；本轮在11-26公告目录发现，按原始日期补录。[论文 §3–4、Appendix D／L／M](https://arxiv.org/html/2611.33183v1)。

REI把提交变更与后续采用信号变成可学习的技术提案；SPOD显式标方法类别，校正类别奖励偏差、保留每类最佳项以避免少数方法族垄断。内部推荐系统真执行的Success@3为REI-8B0.79、加SPOD0.83；公开nanoGPT speedrun／rustc-perf主要评估类别覆盖与离线回放，不能当重新训练得到同样收益。论文修正分析也提示最强证据在分布对齐／降低类别集中，非每项质量指标都显著；新域需要可靠分类。**建议深入阅读公开基准与纠正说明，继续观察独立复现**，不把内部业务收益直接迁移为LLM训练提速。

<!-- item:2611.33302|后训练 -->
### 遗忘有两个时间尺度：输出不再出现，表征可能仍可用（补录）

**机构／日期／版本**：Principia／UCL、Université Paris-Saclay；2026-11-24，2611.33302v1。[论文 §3–6、Figure 2／4](https://arxiv.org/html/2611.33302v1)。

在随机层级语法上预训练、用部分规则微调后，删除规则的生成概率先下降，支持它的隐表征较晚才随参数漂移丢失。8层英法双语从零训练模型再单语SFT也看到类似分离；它支持对“输出遗忘”与“表征擦除”分开测量。局限是受控窄分布SFT与小模型，未覆盖现代大模型多轮RL或证明所有被压制能力都会保留；表征相似度也只是本文操作定义。**建议深入阅读并考虑做恢复／探针对照**，不能仅凭最终答题率下降判定已完成知识删除。

<a id="topic-4"></a>
## 推理与部署

<!-- item:2611.34776|推理与部署 -->
### DFly2：草稿共用目标KV与稀疏索引，长窗收益要按部署规模读（补录）

**机构／日期／版本**：腾讯DFly2作者团队；2026-11-25，2611.34776v1。[论文 §3–5、Table 8–10、Figure 7](https://arxiv.org/html/2611.34776v1)。

三层MoE草稿读取对应目标层已存的键值（KV）和上一验证步已选的稀疏位置，不维护独立KV或重做全窗评分。各草稿按同数据／步数、32K序列、5词元草稿块训练；H20计时中，相对强基线单请求解码一般快2%–6%、长窗6%–18%、Agent12%–30%，接受长度提升不能冒充吞吐。DS-V4.1-Flash的64K CodeSpan／SWE，同用户速度下相对DSpark吞吐+62%／+81%，但采用DP8×TP1／EP8及8个预填充实例对1个解码实例，不是单实例或同卡成本结论。**建议优先深入阅读并考虑复现长窗供给／KV占用对照**。交付增量：11-26已合入的[AngelSlim #409](https://github.com/Tencent/AngelSlim/pull/409)固定AngelSpec到[e02ea36e](https://github.com/Tencent/AngelSpec/blob/e02ea36e65d44e6e165c4ba73c139d500e12f2d9/README.md)，公开训练入口及三种B6草稿；[DS模型卡固定d08e55a6](https://huggingface.co/AngelSlim/DS-V4.1-Flash-DFly2-B6/blob/d08e55a6897b1974307ea9771038838de5986e66/README.md)明确FP4专家／其余BF16、共享目标embedding／输出头，vLLM服务支持仍在进行中，不能写成通用发行版已支持。

<!-- item:2611.33919|推理与部署 -->
### DiCALM：将卷积—注意力混合模型改为掩码扩散，当前仅小模型SFT（补录）

**机构／日期／版本**：Qualcomm AI Research；2026-11-25，2611.33919v1。[论文方法、Table 1、速度协议与Limitations](https://arxiv.org/html/2611.33919v1)。

方法把LFM2／2.5因果门控卷积渐进扩为双向局部卷积，配合掩码预热与AR—扩散联合监督，补上仅适配全注意力模型的转换空白。同样微调的350M／1.2B基线、H100 80GB、batch1、最多1024新词元、eager GQA注意力，仅计generate调用：0.9置信阈值下七任务平均快1.42／1.43倍；1.2B GSM8K62.55%→60.73%。是质量／并行度折中，未测显式长思考、持续预训练或服务批处理。**建议考虑小模型复现转换与阈值曲线**，不能外推到大规模混合架构生产吞吐。

<!-- item:2611.33080|推理与部署 -->
### Shifter v2：让小分组量化跨完整归约做整数MMA，质量与速度配置须拆开（补录）

**机构／日期／版本**：Yufeng Huang；v1 2026-11-24，v2 2026-11-25，本轮核v2补录。[论文 §3–6、Table 5–9](https://arxiv.org/html/2611.33080v2)。

SQ把共享位追加、移位、截断成原生整数操作数，让矩阵乘加（MMA）跨完整归约维执行；SHI再用短归约补偿做INT4。RTX5090上的Qwen3.8-27B，单4096词元预填充，SQ2对llama.cpp Q2_K为2.71倍；SHI2为3.77倍，但其速度用A4、质量测试用A8，不是同配置质量—性能点。四位SQ对NInfer整数基线有优势，SHI4对NVFP4则在所报服务负载更慢；NInfer值来自固定公开版本，KV格式也不同。**建议深入阅读、考虑同引擎同精度复现**；需分开序列化位数、展开驻留平面和临时工作区，本文非独立复现。

<!-- item:2611.34693|推理与部署 -->
### ShowNow：显示前准备多种未来，主要收益是连续性与临近截止响应（补录）

**机构／日期／版本**：NUS Show Lab；2026-11-25，2611.34693v1。[论文 §3–4、Table 1／3](https://arxiv.org/html/2611.34693v1)。

交互视频世界模型在播放期间预备1–8种动作未来，用单潜变量提交、共享历史树执行减少切换后重生成。单H100、5B、480p16fps，完整部署输入到显示P50从SolarWM-FT的620ms降到208ms；这同时改变块大小、训练模型、执行与读前量，不能全归因于多未来。固定ShowNow检查点／运行时的对照，Reactive与Eight Futures P50为211／208ms，重启空拍1.56%→0.09%，才隔离准备收益。**建议深入阅读并考虑匹配运行时复现**；32fps是把16fps内容2倍播放，非原生32fps生成，八动作覆盖也有限。

<a id="topic-5"></a>
## 开源工程

本节逐项核实官方PR正文、最终文件差分与合入状态；性能/复现均为作者提交的证据，本轮未上设备运行。除特别注明，都是源码落地。TP为张量并行、EP为专家并行、CP为上下文并行、PP为流水并行，P/D为预填充与解码拆分；TTFT为首token时延，TPOT为每输出token时延。推理内核改善、能力/质量与整步成本分别判断。

### verl sharded_nixl取消训练端全量聚合，但小模型和异步恢复仍有边界

**项目／日期／版本：** verl-project/verl；合入／主干落地UTC 2026-11-27T03:06:43Z；`f4fb2b63fd00`。

原先NCCL/NIXL从trainer rank0发送整模型，每个rollout worker又收进自己不加载的参数。新后端让每个训练rank导出自己持有的分片，按rollout的TP/PP/EP所有权过滤，以NIXL单边WRITE写入GPU或锁页host桶；首轮缓存布局，vLLM本身的load_weights干跑生成TP切片表。作者在8卡B300节点/RoCE、Qwen3-235B-A22B、16或32卡Megatron→16卡vLLM的关闭切片哈希的稳态第2–4轮，同步约20–22秒降至6.35–6.36秒；30B-A3B六步GRPO中位9.4→7.5秒，0.6B却3.3→6.1秒。235B独立harness未随PR交付，设备环境用vLLM0.24，PR说明当时pin的0.32尚未复验，不能外推更新版本。TP局部切片要求plain BF16、TP>1及本地checkpoint，量化、LoRA、投机和PP回退整张量路径。失败后manager阻止恢复生成至重试成功，但fully_async rebalance可绕过该门禁；只实测v1 separate_async，SGLang、one_step_off等未跑。建议大MoE分离式RL深入阅读并考虑复现布局和失败恢复；小dense先测固定开销。

[verl-project/verl#8395](https://github.com/verl-project/verl/pull/8395)

### SGLang elastic EP合入在线退卡，默认重捕图与可选复用代价不同

**项目／日期／版本：** sgl-project/sglang；合入／主干落地UTC 2026-11-27T01:27:12Z；`c8186c7b4bd1`。

退卡从等待故障超时变为PREPARE→DRAIN→主动deactivate_ranks→翻active mask→重配；仅关闭离开slot的新请求，既有请求排空后才退，存活rank修专家映射并重载改名slot。设置external退役生命周期后，外部编排器可等待safe_to_terminate_ranks再结束进程；默认self_exit自行退出。最终源码默认缩容后重捕decode图，NIXL路径可用--elastic-ep-reuse-cuda-graph保留固定宽度图并将退役rank的token计数置零。作者GB200、8节点各4卡、DeepSeek-V2-Lite-Chat BF16、EP32→16、64流/最多600输出token、五轮中位：冷重启停顿101秒；默认缩容最长流停顿11.4秒、总重配23.5秒；复用图0.74秒/13.9秒。不能把早期H100示例13ms停顿写成最终默认配置。限制包括attn_tp_size必须1、只退高rank连续后缀、降级拓扑拒绝resize；operation_id只是关联标签，不是幂等键，迟到回复仍有无call-id窗口。建议弹性MoE服务深入阅读，先复现流量排空、图模式及故障/缩容交错。

[sgl-project/sglang#33111](https://github.com/sgl-project/sglang/pull/33111)

### OpenRLHF修复工具回复介入GAE和REINFORCE折扣的错误

**项目／日期／版本：** OpenRLHF/OpenRLHF；合入/落地主干UTC 2026-11-27T01:18:34Z–2026-11-27T01:38:30Z；`e95bb621cba0`, `4d6c7f6aae68`。

多轮序列把工具回复插在策略动作间，旧广义优势估计（GAE）按上下文相邻位置bootstrap，工具token既读取masked value又重复打折；REINFORCE回报同样按所有位置推进。现在均只在action_mask动作处更新递推。γ=0.9、λ=0.8、两动作value0.25/0.75、末reward1时，首优势应0.605；插两工具token的旧实现变0.193232。作者A100三seed小Llama/DeepSpeed stage0重放三步PPO，旧actor/critic首步梯度偏差约24–31%/22–24%，修复后相对标量FP64动作递推≤1e-7；REINFORCE另有masked-KL与真实PolicyLoss梯度回归。γ=λ=1的GAE对照不受这个折扣问题影响，不能据此解释默认超参所有训练变化。建议非1折扣的Agent PPO/REINFORCE优先采用并重审旧实验。#1439正文为空，结论来自实际合入patch及其回归。

[OpenRLHF/OpenRLHF#1430](https://github.com/OpenRLHF/OpenRLHF/pull/1430) · [OpenRLHF/OpenRLHF#1439](https://github.com/OpenRLHF/OpenRLHF/pull/1439)

### OpenRLHF把均匀奖励组的浮点噪声清零，EMA保留FP32累积

**项目／日期／版本：** OpenRLHF/OpenRLHF；合入/落地主干UTC 2026-11-27T01:17:00Z–2026-11-27T01:17:09Z；`92688d6464af`, `8b0f3f2e5e4f`。

group_norm对完全相同的0.3/0.6/0.9奖励，FP32 mean/std残差约1e-8再除std+1e-9，旧优势可接近±0.9；新按组max==min将该组显式置零，其余组沿用旧算式。EMA则修BF16/FP16每步平均被舍入吃掉：每rank在EMA设备额外保存每可训参数4字节FP32平均，再写回原dtype导出。作者CPU/真实DeepSpeed ZeRO2 actor+stage0 EMA、600步、β0.992，修复后与FP64参考一次舍入一致；未跑整套Ray PPO或GPU。ZeRO3 EMA不持久更新的旧问题未修。建议分数型RLVR和启用EMA用户采用，但将新增显存/内存及ZeRO阶段作为门槛。

[OpenRLHF/OpenRLHF#1383](https://github.com/OpenRLHF/OpenRLHF/pull/1383) · [OpenRLHF/OpenRLHF#1419](https://github.com/OpenRLHF/OpenRLHF/pull/1419)

### OpenRLHF短rollout对齐各DP rank的前向次数

**项目／日期／版本：** OpenRLHF/OpenRLHF；合入／主干落地UTC 2026-11-27T01:16:57Z；`b67e80ecf17f`。

静态批最后672样本/42微批分4卡曾给各rank11/11/11/9次forward，ZeRO3参数collective会等缺席rank。新把微批数补成有效actor数的倍数并均分样本；整批原分割保持，短批通常更小，极短批也可能略超过micro_batch_size。两rank CPU/Gloo/DeepSpeed重现阻塞并有负对照回归，未跑多GPU。样本数不能整除rank时PPO更新微步仍可能不齐，另由#1360负责；reference/critic/reward卡数与actor不同也未覆盖。建议尾批常见的分布式PPO优先核对这两个独立边界。

[OpenRLHF/OpenRLHF#1434](https://github.com/OpenRLHF/OpenRLHF/pull/1434)

### OpenRLHF数学奖励修正大整数、千分位与fbox，0.12已打包

**项目／日期／版本：** OpenRLHF/OpenRLHF；合入/落地主干UTC 2026-11-27T01:19:03Z–2026-11-27T01:38:11Z；`9c5d940fee61`, `7e8071c4328b`, `fc7a7e6f368e`。

奖励判分先转binary64会把9007199254740993与9007199254740992混同；现字符串经Decimal保留整数精度，保留原1e-7近整数容差。相邻千分位分隔符改lookaround一次删除，避免正确1,234,567,890获零分；fbox也按对应前缀提取。回归调用实际grade_answer与示例reward，证明reward/score/accuracy同步变化，未证明所有数学表达式等价判定可靠。后两项为CPU测试，千分位修复保留原局部格式规则，不增加通用格式验证。建议数学RLVR数据与奖励审计采用并抽查历史误奖/漏奖。三项均在11月27日发布的0.12.0 changelog中。

[OpenRLHF/OpenRLHF#1438](https://github.com/OpenRLHF/OpenRLHF/pull/1438) · [OpenRLHF/OpenRLHF#1409](https://github.com/OpenRLHF/OpenRLHF/pull/1409) · [OpenRLHF/OpenRLHF#1374](https://github.com/OpenRLHF/OpenRLHF/pull/1374)

### Prime-RL接通Inkling 2 Mini循环模型RL，路由按层访问次数重放

**项目／日期／版本：** PrimeIntellect-ai/prime-rl；合入／主干落地UTC 2026-11-27T20:04:42Z；`1ac98cc474dd`。

循环核心同一权重被多次访问，trainer/router replay、capture buffer和KV/conv状态均改按48次layer visit而非24个存储层编号；转换器解NVFP4/MXFP8混合专家到BF16，原生非loop config遇loop字段拒绝静默误载。8×H200、4训练卡EP4+4单卡BF16 rollout、8K序列、batch64/组8，20步均值训推KL无重放0.0013、有重放0.00009；少数token仍有大偏差。一次100步数学任务带中reward后40步约0.55–0.57、早30步0.50–0.52，截断下降，median34秒/step、trainer峰107GiB/卡；无配对能力对照。未含FP8 rollout、视觉、DSpark、可变loop或CP；trainer导出也不能直接给原生vLLM服务。建议研究循环MoE后训练深入阅读，先验每次访问的路由与状态编号，不能用这次单任务曲线证明普遍质量提升。

[PrimeIntellect-ai/prime-rl#4669](https://github.com/PrimeIntellect-ai/prime-rl/pull/4669)

### Prime-RL把rollout检查点写盘移出权重发布屏障

**项目／日期／版本：** PrimeIntellect-ai/prime-rl；合入／主干落地UTC 2026-11-27T02:33:43Z；`4d5954348448`。

环境快照拿到可恢复名字即可放行，箱体恢复及重试后台继续；只在发布屏障内记录pipeline和切点batch数组，其他训练数组及fork writer随后在sink锁内处理，避免trainer的weight broadcast被慢恢复拖住。ScaleSWE同配置并行A/B、35B-A3B、各1训练/1推理节点、NCCL、每2步保存，四个切点屏障38/67.2/59.3/301.5秒→6.9/8.8/5.2/2.8秒，最多15条在途rollout重来；硬件型号未披露。这是发布等待缩短，不是整检查点写入或训练总时长同倍下降。rollouts_save_timeout语义从每pool总预算改每rollout，默认600→120秒；未回复pool还可等待额外600秒。建议长Agent训练复现恢复点完整性、取消与真正持久化完成时刻。

[PrimeIntellect-ai/prime-rl#4787](https://github.com/PrimeIntellect-ai/prime-rl/pull/4787)

### Prime-RL每节点进程镜像DCP分片，metadata最后才发布

**项目／日期／版本：** PrimeIntellect-ai/prime-rl；合入／主干落地UTC 2026-11-27T19:52:36Z；`575c0c5689ad`。

训练rank不搬远端数据，每节点local rank0启动同步进程按DCP .metadata枚举自己的有效分片；node0等全节点上传后最后传.metadata作为完整标记。磁盘删除排在上传后，远端/本地保留策略分别管理；训练进程死亡时同步子进程立即退出以免提交重启正在清理的旧step。远端独有trainer checkpoint可在建模时并行拉取，但只镜像trainer，RL仍需run目录且远端超前时拒绝。当前证据是LocalStore多节点逻辑与故障单测，R2对象存储尚无端到端。建议检查点容量/容灾需求者深入阅读，采用前补真实对象存储失败和resume验证。

[PrimeIntellect-ai/prime-rl#4514](https://github.com/PrimeIntellect-ai/prime-rl/pull/4514)

### Prime-RL与Dynamo1.7打通rollout和ModelExpress成员变化

**项目／日期／版本：** PrimeIntellect-ai/prime-rl；合入/落地主干UTC 2026-11-27T02:29:33Z–2026-11-27T02:29:35Z；`a2f5a639d26f`, `32b844e125e9`。

Dynamo discovery每轮重读活worker，ModelExpress以instance-id命名局部rank；全成员pause→安装同一版本→post-transfer→resume，任何安装失败保留暂停。后续PR再去掉无用prl_replay:false与缺失usage阻塞，才使generation真的穿过Dynamo前端。单节点Qwen3-0.6B五步、vLLM0.34/ModelExpress0.8/Dynamo1.7，NCCL单worker及ModelExpress双worker均0调用失败、KL约0.0002–0.0007；成员更换后存活/新worker最终fingerprint相同。top-p/top-k截断、score-centering/logprob-head请求需extra_args而sidecar拒绝，配置现在提前拒绝；跨主机、worker内TP/DP>1和P/D未测。在两次更新间加入者会先用启动权重，无token级版本时只能按dispatch版本估staleness。建议弹性RL服务考虑复现，勿把动态权重发现等同所有采样模式可用。

[PrimeIntellect-ai/prime-rl#4802](https://github.com/PrimeIntellect-ai/prime-rl/pull/4802) · [PrimeIntellect-ai/prime-rl#4800](https://github.com/PrimeIntellect-ai/prime-rl/pull/4800)

### Megatron把shortcut MoE collective收回主流，保留专家计算重叠

**项目／日期／版本：** NVIDIA/Megatron-LM；合入／主干落地UTC 2026-11-27T20:56:00Z；`c755fd1d9d65`。

旧routed branch连NVLS metadata/AllGather-V/ReduceScatter-V放侧流，与主流mix/shared expert交错，EP多轮Agent负载出现设备忙等死锁；新仅无collective的专家grouped GEMM上侧流，router/dispatch/combine统一主流顺序。作者两EP rank/MTP2相同seed负载，旧2/2次约4分钟挂、新约3.7小时无挂；精确循环等待未被识别，且源码仍拒MTP+overlap组合，不能把此压力实验当受支持配置。无MTP配对实测TP1/EP2或4、8K入/1K出与Agent回放比旧overlap慢约1–5%，但仍可比关闭overlap快；需已合入#9598保护专家权重存储。建议有该组合的服务优先采用并计入稳定性代价，而非宣称纯性能提升。

[NVIDIA/Megatron-LM#9627](https://github.com/NVIDIA/Megatron-LM/pull/9627)

### Megatron异步compact后，MTP草稿KV提交改回原请求行

**项目／日期／版本：** NVIDIA/Megatron-LM；合入／主干落地UTC 2026-11-27T19:41:18Z；`69615f134430`。

异步调度只把待验forward的logits压成survivor顺序，hidden/input仍原序；旧commit用新row算hidden_idx，完成请求左侧空位被另请求占用时，把他人的hidden/token写入存活请求draft KV。新把token_row_indices一路传到commit的decode段。target验证保证生成token不因这项错写直接变化，但后续草稿接受率可持续下降；不能只靠答案一致排除。作者用真实engine+scheduler配确定性模型在旧版73行中抓到2错行，新增回归仅旧版失败；H100八rank及FA3/FA4相关套件通过，无吞吐量化。legacy、prefill与不用draft KV者不变。建议异步MTP缓存用户优先采用并加入请求身份而非只位置的检查。

[NVIDIA/Megatron-LM#9234](https://github.com/NVIDIA/Megatron-LM/pull/9234)

### Megatron Muon用无需通信的组填流水空档

**项目／日期／版本：** NVIDIA/Megatron-LM；合入／主干落地UTC 2026-11-27T15:43:38Z；`fd8d02c0bfaa`。

完整张量落在单rank的组省gather/scatter，均匀插入其他组的异步传输之间，最后也放一个无通信组，以Newton–Schulz计算遮住在途通信；布局固定的requires_communication缓存，owner直接用本地view。作者DSV4-Pro七层、64×GB200，同一分配三轮含反序，叠#9702基线Muon GPU窗口838/856/838ms→829/851/833ms，均值少约6.5ms（不足1%），统计是rank0/32/48/63平均。#9703 foreach动量属于另一项，不能归给本PR；无完整671B或整训练步加速结论。建议FSDP Muon实践深入阅读调度结构并复现自己的组布局。

[NVIDIA/Megatron-LM#9682](https://github.com/NVIDIA/Megatron-LM/pull/9682)

### Megatron修learnable attention-sink梯度范数，并限制DCP元数据反序列化

**项目／日期／版本：** NVIDIA/Megatron-LM；合入/落地主干UTC 2026-11-27T15:25:43Z–2026-11-27T17:54:20Z；`b4b2e5ec6314`, `9767f25ac163`。

按head切在TP rank的softmax_offset未标tensor-model-parallel，clip-norm只保留部分rank的sink梯度；TE与原生attention构造处补分片属性，这只解决#7452的finding8。另SafeDCPUnpickler把DCP .metadata反序列化换成限定类型路径，在core导入时patch DCP的pickle接口，并有合法metadata/拒绝危险对象回归；并非对所有checkpoint内容作通用安全证明。前者有CPU/Gloo复现，后者没有本次大模型恢复或性能对照。建议learnable sinks训练核梯度裁剪，使用DCP者核兼容类型和恢复路径。

[NVIDIA/Megatron-LM#7530](https://github.com/NVIDIA/Megatron-LM/pull/7530) · [NVIDIA/Megatron-LM#9488](https://github.com/NVIDIA/Megatron-LM/pull/9488)

### TorchTitan门控局部编译：算子倍数与整步收益分开

**项目／日期／版本：** pytorch/torchtitan；合入／主干落地UTC 2026-11-27T08:55:05Z；`90ae735851d3`。

Qwen3.5/3.8 attention output gate三次eager与GDN decay/update约七次eager分别改local_compile融合；全block编译会破torch_remat saved-tensor hooks及FA4/Flex边界，所以仅包两个区域并保留非连续输入。BF16舍入可差1ULP。B200当前nightly重测output gate、T65536/131072/262144的无图速度比0.93/1.18/1.70，早表小shape1.43倍已不成立；小shape有回退。Qwen3.5-0.8B、24层、12步、full activation remat，B200/FA4/65536长序列整步514.30→510.16ms（约0.8%）；A100/flex/16384仅少4.22ms。Flash-Next已接同区域但无该模型端到端loss对照。建议长序列训练考虑复现，别把单gate图内1.8–2.8倍写成训练倍数。

[pytorch/torchtitan#6321](https://github.com/pytorch/torchtitan/pull/6321)

### SGLang修prefill尾图首次捕获的DP collective失配

**项目／日期／版本：** sgl-project/sglang；合入／主干落地UTC 2026-11-27T04:26:27Z；`82b9b10f5de9`。

DeepSeek-V4.1 DP attention/MegaMoE/breakable prefill图中，大bucket走尾部break并在首次捕获进barrier，小bucket把尾层录在图内；不同rank跨2048阈值时一侧等barrier、一侧等MoE，冷启动可反复挂。新启动统一prime尾图，服务时只在所有rank都到break才捕缺图，否则该tail eager；空闲rank不再强制忙rank全部eager。作者8×B200/1P1D、SGLang0.5.23回移同改动线，120/120冷波及两次一小时约3.8万请求0服务器错；最终树本地主要CPU验证。1K入/200出60req/s对照meanTTFT462.6→164.1ms，长60K入场景无稳定收益；启动多3.3秒和0.049GiB/rank。新key/bucket仍可能首轮eager，最大capture shape过小时prime会空跑。建议这组图/DP/MoE部署优先审查并在正式版本复验。

[sgl-project/sglang#48850](https://github.com/sgl-project/sglang/pull/48850)

### SGLang补齐HiCache idle、PP flush和分叉备份三个完成边界

**项目／日期／版本：** sgl-project/sglang；合入/落地主干UTC 2026-11-27T02:38:59Z–2026-11-27T15:23:20Z；`4791be0149a4`, `9840eb3724ee`, `f3ff5320e926`。

write-back ack已从ongoing队列转入fenced队列时，D2H尚未结束却可被判idle；现在真正空闲时同步并退休fenced event，再允许flush/存储切换/权重释放。此项关闭上轮对#50469的idle/drain待查。PP flush则让PP0 relay探针、全stage投票idle、所有stage reset完再处理下一storage ticket，避免先空闲stage把仍工作的peer丢下；有两stage真实Gloo/file-backed字节回读，DSV4 HTTP端到端仍待测。buffer_only下父节点被split且原备份尚在途，子节点改以父末页stored或in-flight覆盖作准入，免整串分叉备份丢弃；该项主机元数据回归，不提供吞吐数字。建议有PP、慢存储或频繁flush的服务优先采用并保留完整完成事件。

[sgl-project/sglang#52437](https://github.com/sgl-project/sglang/pull/52437) · [sgl-project/sglang#51780](https://github.com/sgl-project/sglang/pull/51780) · [sgl-project/sglang#52954](https://github.com/sgl-project/sglang/pull/52954)

### SGLang将每路径Mamba软上限扩到host副本

**项目／日期／版本：** sgl-project/sglang；合入／主干落地UTC 2026-11-27T01:17:06Z；`51c77c89f889`。

旧软上限只裁device副本，仍活路径的过期host状态挤占host Mamba池，迫使有用KV被逐出。新在HiCache cache模式把host副本计数，裁状态时一并删可删host副本；host锁定或传输在途、fork/tail/device-leaf仍保留，buffer_only或无限额不启用。Qwen3.5-0.8B、97状态小host池、单GPU三轮，crowding命中81.4→90.7%、缓存token+11.5%，延迟差在运行波动内；504输出502一致，恢复与重算两种舍入的两条差异未查logprob。早期Kimi单轮+12%并非本head独立收益。代价是最深状态miss时较浅host fallback已被剪掉。建议混合模型多轮缓存复现命中与质量，避免把软上限当硬一状态/会话。

[sgl-project/sglang#50954](https://github.com/sgl-project/sglang/pull/50954)

### SGLang Qwen CP×整专家EP与NIXL切片打包：TTFT收益伴TPOT代价

**项目／日期／版本：** sgl-project/sglang；合入／主干落地UTC 2026-11-27T21:34:09Z；`c37fc5f33faf`。

CP/TP共享组允许EP==TP整专家，避免窄专家640维切TP2成320而不对齐FP8块；KV head切片先在每transfer worker有界64MiB scratch打包，再合并写decoder KV，失败不确定的buffer隔离至不再复用。4×H200，P TP2/CP2/EP2→D TP2/EP2、两轮每100请求/c10、目标65536入1000出、BCG与receiver staging均关，93.75%名义共享prefix时TTFT均值46.21→1.06秒，output174→753token/s，但TPOT9.36→12.06ms；无cache则TTFT49.55→4.99秒、TPOT5.87→10.75ms。不要用正文早期53.89→1.46秒概括最终表，也不当P侧最大吞吐。patch只在CUDA、VRAM、非MLA/hybrid-MLA、PP1/DCP1开打包；建议P/D拓扑与缓存负载下复现完整时延权衡。

[sgl-project/sglang#47920](https://github.com/sgl-project/sglang/pull/47920)

### SGLang staging按16字节对齐向量化，测的是拷贝核

**项目／日期／版本：** sgl-project/sglang；合入／主干落地UTC 2026-11-27T21:37:49Z；`37a52b4427ce`。

从device指针表取地址让Triton看不出KV对齐，旧payload被标量load/store；注册时验证KV基址/stride，每传输只验staging偏移，满足16字节才给multiple_of提示，不满足保原generic。GB300四rank、BF16/80层/1024token/页1/KV8头D128、P TP4→D TP1，graph replay gather37.29→34.86µs，scatter151.90→105.00µs；含每次检查为35.65/105.93µs。1/2/4字节及错位路径逐字相同，scatter共享内存8KiB→0。未含RDMA、请求处理或端到端。建议有P/D staging拷贝热点者考虑复现，不能把scatter少31%外推服务吞吐。

[sgl-project/sglang#39236](https://github.com/sgl-project/sglang/pull/39236)

### SGLang统一decode/投机verify的logits处理阶段

**项目／日期／版本：** sgl-project/sglang；合入／主干落地UTC 2026-11-27T23:19:34Z；`e8317f420c59`。

penalty→grammar→logit_bias→watermark→custom processor一次定义，各verify fast-path共读VerifyRoute，避免图key、vocab-shard和各drafter分别判断。真实行为变化是verify把watermark放custom之前，并对保留的原logits副本也执行custom；内置Unlimited-OCR v2 page guard+水印+投机可观察，防水印把被屏蔽token加回来。B300单卡Llama/Qwen固定seed的decode/EAGLE树链/NGRAM/DFLASH/DSPARK差分逐token/logprob/logits哈希相同，另约93.5万gate组合对照；这是作者给定配置的差分，不是所有模型能力认证。MLX每launch新增两次O(batch) host扫描，未给性能对照。建议维护采样插件者深入阅读阶段组合与fast-path门禁。

[sgl-project/sglang#52406](https://github.com/sgl-project/sglang/pull/52406)

### SGLang修NGRAM P/D接续并补prompt语料

**项目／日期／版本：** sgl-project/sglang；合入／主干落地UTC 2026-11-27T22:12:06Z；`aac1ef2cc86a`。

P/D decoder从不走extend，既漏NGRAM draft-input构建而首prebuilt batch崩溃，又漏把prompt尾写进NGRAM trie。新入站给零accept状态并覆写复用pool row，首次只靠host token历史draft；同时插prompt语料。2×A10080GB/Mooncake/Qwen2.5-Coder7B，四个实测通过，移除入口分支即崩、移除prompt seeding复制prompt请求accept length3.5→1.01。较早GB300/NIXL回移也有取消清理与聚合输出对照。TP>1、DP attention、HiCache、内存压力retraction、PD页>1等仍未覆盖；正文对grammar覆盖叙述与末尾未覆盖清单不完全一致，因此只确认所列小测试，不扩大。建议NGRAM分离式服务采用入口修复并补自身组合回归。

[sgl-project/sglang#52551](https://github.com/sgl-project/sglang/pull/52551)

### SGLang统一真实请求长度，并改变prefill预算与首token发送策略

**项目／日期／版本：** sgl-project/sglang；合入/落地主干UTC 2026-11-27T09:50:31Z–2026-11-28T00:55:28Z；`29e67c3294f8`, `6547f87787e3`, `ddda5e37abee`。

长度唯一来源为min(model context, KV容量)，prompt上限扣1、5、dLLM block与预留草稿token并按inclusive接受；KV不足而请求显式输出预算原被静默截短，现在400或按allow-auto-truncate截；多模态绝不截placeholder，超长embedding abort。Python/Rust入口与scheduler同步，有CPU/Rust回归但无模型吞吐对照。另max_prefill_tokens自动至少等于最终chunk大小，显式小预算也会被抬高，可改变多请求prefill批量。P/D first-token-stream新增admission/first_decode，默认后者；只是发送prefill首token的时点改变，TTFT与按(e2e−TTFT)/(out−1)算的TPOT互换等待，不让token更早计算。建议升级时检查显式budget、KV容量和SLA测量口径。

[sgl-project/sglang#53454](https://github.com/sgl-project/sglang/pull/53454) · [sgl-project/sglang#52964](https://github.com/sgl-project/sglang/pull/52964) · [sgl-project/sglang#53293](https://github.com/sgl-project/sglang/pull/53293)

### SGLang load-diff先合入格式，尚无运行时caller

**项目／日期／版本：** sgl-project/sglang；合入／主干落地UTC 2026-11-27T22:10:44Z；`507ac7e2da14`。

仅参数/buffer共享会漏post-load的plain tensor属性、scale标记、dict缓存和view；新增walk/storage/view/带tag的普通值及BLAKE3摘要格式，保对象别名与shape/stride/offset，拒部分重叠、地址值和非法manifest，process-local字段只带形状不带字节。decode不导入新模块或执行新代码。CPU广泛格式回归及B200 hash对照支持本格式，**此PR没有caller**；record/export/install和实际weight-cache服务是后续层，不能写成模型跨进程加载已提速或完整能力发布。建议做快启动/权重共享基础设施者深入阅读格式契约，部署继续观察接线。

[sgl-project/sglang#51728](https://github.com/sgl-project/sglang/pull/51728)

### vLLM异步EPLB策略移持久CPU子进程，绕开worker的GIL

**项目／日期／版本：** vllm-project/vllm；合入／主干落地UTC 2026-11-27T09:43:56Z；`8ad82dae1db5`。

仅CPU placement策略离开推理worker，负载汇总、D2H、权重迁移及map commit保原路径；每worker复用子进程，跨rank协调IPC超时/死亡，异步use_async默认启用时走新路径。2×H20、Qwen3.6-35B-A3B BF16/TP2EP2、128入1出/c16、8192请求，稳态52.15→54.95req/s，P99 TTFT476.9→382.7ms；8×H20/DSV4.1 FP8 KV、256入1出、固定39.35req/s的P-only样本P99 313.5→131.6ms，不含KV handoff/decode，闭环吞吐仅+1.21%。性能取早head但热路径相同，最终修过IPC/FT；首token仅7470/8192相同，不能称服务输出逐位不变。策略和stats必须可序列化。建议CPU调度干扰GPU提交明显者考虑复现。

[vllm-project/vllm#68606](https://github.com/vllm-project/vllm/pull/68606)

### vLLM接Mooncake EP后端，仍只有小规模正确性证据

**项目／日期／版本：** vllm-project/vllm；合入／主干落地UTC 2026-11-27T20:43:00Z；`944ec109da42`。

optional mooncake_ep加入A2A registry、CUDA communicator和modular FusedMoE prepare/finalize，复用标准专家算子集成。两H20、torch2.13/CUDA13、Mooncake0.3.13.post1、不同token数BF16 dispatch/combine与本地专家reference六测试通过。没有多节点、FP8或整模型吞吐/显存对照，不把“SGLang已有”当vLLM性能证据。建议依赖Mooncake通信栈的MoE服务深入读适配点，采用前测目标拓扑。

[vllm-project/vllm#58375](https://github.com/vllm-project/vllm/pull/58375)

### vLLM修Mamba混合前缀在attention块内的UMBP恢复

**项目／日期／版本：** vllm-project/vllm；合入/落地主干UTC 2026-11-27T04:33:28Z–2026-11-27T08:22:50Z；`1185d98ea797`, `b7a65380bc03`。

中间checkpoint落在attention页内部，旧只存Mamba state而漏该attention页，跨组不能在同边界命中。新手递state时同存attention页，以core当前block table兼容lazy_offload；匹配粒度比scheduler块细的Mamba hybrid默认partial_hash_hits打开。Granite4.0H350M恢复0→832/840token且tokens/logprobs同本地命中，NemotronH8B 200题五轮复用98.8%prompt后分数在重算波动内。Kimi MI355X单节点旧定制build、synthetic DSpark acceptance和水位不同的单轮21%收益无法单独归因本patch。代价每checkpoint多一attention页；lookup不pin，4GiB压力池4/200请求失败，recompute策略或64GiB池避免该观测。另embedded默认水位0.9/0.7→0.99/0.97使约98%驻留；同节点含其他修复的单轮仅+1.3%，勿叠加百分比。建议混合多轮缓存复现边界与逐出竞态。

[vllm-project/vllm#67942](https://github.com/vllm-project/vllm/pull/67942) · [vllm-project/vllm#67944](https://github.com/vllm-project/vllm/pull/67944)

### vLLM Mooncake可向HiSparse host池导入异构KV分组

**项目／日期／版本：** vllm-project/vllm；合入／主干落地UTC 2026-11-27T03:33:00Z；`86b6d3adc9c3`。

普通sparse-MLA prefiller把source/indexer KV放一group，HiSparse decoder却source在host、indexer在GPU；新按layer匹配region、用各端自己的group ID配末尾缺页，共享host池仅decoder TP0注册/请求host写，各rank仍履行GPU索引与completion。GLM5.2 FP8、16H200 TP8→8有token/logprob逐位P重放对照，24H200 TP8→16实际传输零connector失败；均eager/PP1、32GiB host、buffer2048。仅相同token块、未拆物理页、每group缺页为连续逻辑后缀，内洞和window截页不在契约；TP需整除关系。建议HiSparse P/D容量扩展者深入阅读，不能按任意页面集合或PP宣称可用。

[vllm-project/vllm#66502](https://github.com/vllm-project/vllm/pull/66502)

### vLLM把Kimi indexer_logits_dtype真正传给DSA核

**项目／日期／版本：** vllm-project/vllm；合入／主干落地UTC 2026-11-27T06:05:26Z；`18d8b5e7024d`。

recipe虽配置BF16，旧Kimi prefill稀疏索引logits仍FP32；新增透传，默认保FP32。32×GB300、Kimi-K3.1-Pro MXFP4、DEP16P+16D、AgentX/c352、各60分钟，BF16+deep_select相比同配置旧忽略参数总token/s/GPU35756→37130、P90TTFT15.90→11.93秒；decode本已BF16，收益来自prefill。131K prefill378.5→327.7ms，BF16不用deep_select需FP32 copy约378.9ms无益。作者给50个GPU核回归，但本PR无配对能力评测，不能只看吞吐就默认改所有索引dtype。建议该recipe用户先复现top-k/numerics与完整负载。

[vllm-project/vllm#69838](https://github.com/vllm-project/vllm/pull/69838)

### vLLM让既有SiLU+FP8融合真正可达，列出舍入与greedy分差

**项目／日期／版本：** vllm-project/vllm；合入／主干落地UTC 2026-11-28T00:32:34Z；`269fca48a146`。

CUTLASS block-FP8 linear过去不暴露group128输入quant key，导致已有fused producer从不选中；新接QuantizedActivation并跳独立quant，同时必须把scale改列主序，旧行主序在2token起会有46–71%相对RMS错。B200/Qwen3.6-35B-A3B-FP8/TP1、512入512出、双GPU同时跑并交换卡六轮，c1/4/8 medianTPOT平均少3.8/2.5/2.4%；另B300收益随c变化。GSM8K采样8seed差−0.13pp、贪心10轮均值96.36→95.95%（−0.41pp），作者归近tie/批组成，不能当已证明零质量代价。最终未保曾试过的bit-identical舍入变体。建议低并发共享专家热点复现速度与答案稳定性，尤其RL训推对齐。

[vllm-project/vllm#69778](https://github.com/vllm-project/vllm/pull/69778)

### vLLM Qwen Flash-Next的HC与FP8 QSA两处Blackwell优化

**项目／日期／版本：** vllm-project/vllm；合入/落地主干UTC 2026-11-27T05:17:22Z–2026-11-27T22:02:17Z；`4b70a4a659f8`, `8c5ffad660f8`。

HC对SM100家族、BF16 merged权重、hidden2560/rank320/hc4、M>48融合down+inject+SiLU到tcgen05，BF16落点从SiLU前移到后，batch-invariant与wrap模块保旧分派。4×B300、Qwen3.8-Flash-Next FP8、MTP3、FP8 KV/indexer、8192入1024出，c1–64 output+2.0–3.7%，不等于孤立HC最高7.5%。另FP8 QSA用split-K、tiny-M页表融合、PDL及PTX9.2直接FP8→BF16 K转换（ptxas<13.2回退），GB300微核最高1.92倍；4×GB300/19.2K热prefix+1入256出、两轮c4–64，ITL仅少0.68–1.46%，TPOT受MTP接受率影响。后者GSM8K只报候选96.44%无同配置baseline；两PR测法不同，收益不能相加。建议目标模型/Blackwell长窗复现端到端收益。

[vllm-project/vllm#70160](https://github.com/vllm-project/vllm/pull/70160) · [vllm-project/vllm#70187](https://github.com/vllm-project/vllm/pull/70187)

### vLLM top-k短行跳split/merge，收益由实际行长决定

**项目／日期／版本：** vllm-project/vllm；合入／主干落地UTC 2026-11-27T16:27:58Z；`361f362033fa`。

logits容量≥20万但真实行≤k时，输出仅有效索引加−1，旧仍split/merge且短行会拖全batch；新device长度分支绕过，图可随长度重放。MI325X宽262144/1048576、k512/1024/2048，短行43–142→5–10µs，混合长短batch最高2.5倍；8×MI325X/MistralLarge4/TP8/262K上限、1K输入、不同并发TPOT少15–27%，c1 19.21→14.08ms。8K输入无短行对照差≤0.5%；长行多一次长度load，H200也可多≤4%微核时间。建议长容量、小真实context及异质batch复现，不按max_model_len推断收益。

[vllm-project/vllm#65427](https://github.com/vllm-project/vllm/pull/65427)

### vLLM修Inkling2 padded short-conv，并将家族扩到Ampere/Ada

**项目／日期／版本：** vllm-project/vllm；合入/落地主干UTC 2026-11-27T02:49:59Z–2026-11-27T04:12:08Z；`da11057ed97a`, `8c102bebd5bc`。

prefill>256token的末tile带padding时，旧拿末row的request start=0决定不读conv cache，把继续请求上下文当零；改取tile最大真实start，B300新增负对照核差2.5–3.6修至容差内，未跑端到端。Ampere/Ada支持另禁SM90专属PDL/native conv、对旧卡用兼容E4M3转换；2×RTX3090 TP2、Inkling2Mini BF16，PPL7.4444对HF7.4406，单流约100输出token/s，DSpark约177但不是同硬件理论FP4倍数；测试跳过83个缺硬件feature。SM90/100抽得477 Triton launch PTX同旧版，完整大集群未证。建议旧GPU部署者考虑复现精度/缓存与内存预算，勿把跳过项写支持。

[vllm-project/vllm#70246](https://github.com/vllm-project/vllm/pull/70246) · [vllm-project/vllm#69165](https://github.com/vllm-project/vllm/pull/69165)

### vLLM Gemma4批不变与LoRA确定性选项各有作用边界

**项目／日期／版本：** vllm-project/vllm；合入/落地主干UTC 2026-11-27T01:12:48Z–2026-11-27T22:49:28Z；`48e1793769ab`, `c15fb3688a8d`, `3bf64ce10587`。

Gemma4混合head维在BI关闭FA4后误走不适用FA2，现选Triton fallback；H100 BF16/TP1、E4B、8输出token的单图/混图与encoder图replay目标token/logprob相同，只证该范围。LoRA新增split-k-reduction=kernel用固定8份FP32 scratch+独立固定序归约替atomic，global BI也选它，但匹配tuned shrink配置优先，且不使整模型BI、不影响fused-MoE LoRA，未给速度测量。另embedding/lm_head LoRA改默认不wrap，必须opt-in vocab modules或target allowlist；只含被跳过权重的adapter仍可加载成base输出；lm_head启用后prompt-logprob请求会被拒。建议升级LoRA部署优先验adapter确实生效和配置优先级。

[vllm-project/vllm#70234](https://github.com/vllm-project/vllm/pull/70234) · [vllm-project/vllm#70197](https://github.com/vllm-project/vllm/pull/70197) · [vllm-project/vllm#69354](https://github.com/vllm-project/vllm/pull/69354)

### vLLM HY4 Flash增加PP与DFly P/D，DSV4.1 PCP图像仍有限

**项目／日期／版本：** vllm-project/vllm；合入/落地主干UTC 2026-11-27T07:13:38Z–2026-11-27T14:20:34Z；`5f1df80d3663`, `6311f4474f34`。

HY4 Flash不再强制PP1，layer split与DFly可配NIXL/Mooncake P/D；GB200/官方FP8，TP2×PP2和TP4的PP-equivalence及AIME25/OCRBench对照，1P1D TP4+DFly全GSM8K约93.85–93.93%、零KV失败，接受率接近聚合服务；未给此变更吞吐提升。DSV4.1 NVIDIA MRV2 PCP图像则复用encoder cache与sequence segment、跳空PCP窗口；较早H20/TP1×PCP4/DSpark5 eager33/33包含另共享Engram补丁，本次rebase只有CPU源级回归，正式nightly/全GPU套件未新跑，PCP图限制保留。建议HY4 PP/P-D使用者考虑复现；DSV PCP图像先观察目标组合验证。

[vllm-project/vllm#70300](https://github.com/vllm-project/vllm/pull/70300) · [vllm-project/vllm#64435](https://github.com/vllm-project/vllm/pull/64435)

### vLLM多模态占位符与启动profile改信真实processor输出

**项目／日期／版本：** vllm-project/vllm；合入/落地主干UTC 2026-11-27T09:08:13Z–2026-11-27T11:13:11Z；`d83003b72257`, `3fadb96f1cae`。

视频请求级采样kwargs改变帧数/时间戳，旧LLaVA/Qwen3-VL/Molmo2/GLM4.1V重新按原视频计placeholder；新直接读processor产物，GB10真实processor167项全过，旧50失败，数个实权重模型请求级/全局/预采样tokens与logprobs一致。OCR/MiniCPM则修实际max-crops、slice/downsample和最宽padding profile，免比准入允许的帧数更小的dummy造成OOM。MiniCPM-V4.6在L4默认offline、同wheel，启动activation4.74→5.47GiB，KV容量1127095→1065269token（少5.5%），换来原拒16帧视频被正确服务；不是显存优化。超宽图/视频比率新增拒绝，依模型而异。建议VLM部署按实际processor版本、请求kwargs和最坏padding重测容量。

[vllm-project/vllm#68583](https://github.com/vllm-project/vllm/pull/68583) · [vllm-project/vllm#68623](https://github.com/vllm-project/vllm/pull/68623)

### vLLM及时释放异步媒体解析环，并为VLM草稿训练保留配置键

**项目／日期／版本：** vllm-project/vllm；合入/落地主干UTC 2026-11-27T09:16:53Z–2026-11-27T10:51:29Z；`2bd5514b4791`, `6dc0b3131b61`。

AsyncMultiModalItemTracker resolve先snapshot后清callbacks，打断parser↔tracker引用环，使raw base64不必等循环GC；失败fetch同样释放。作者4 API进程/Qwen2.5VL3B、23.4MB视频/c32、4分钟+45秒idle，API RSS空闲21.59→17.43GiB、peak少2.25GiB，单轮吞吐+3.6%；200×20MiB合成tracemalloc19倍不是整服务19倍。另extract_hidden_states保原decoder/generator/text_encoder typed config键，免额外text_config让NemotronParse启动拒；H100/Qwen3VL两图full vs chunked promptIDs相同、hidden cosine≥0.999，未证明所有层逐位一致。建议大媒体长跑与VLM投机数据采集分别采用，注意证据测量范围。

[vllm-project/vllm#70166](https://github.com/vllm-project/vllm/pull/70166) · [vllm-project/vllm#69064](https://github.com/vllm-project/vllm/pull/69064)

### LMCache FP8序列化修字节污染，multi-chunk与预取唤醒继续落地

**项目／日期／版本：** LMCache/LMCache；合入/落地主干UTC 2026-11-27T05:12:48Z–2026-11-27T15:06:01Z；`a8e61ee57d5f`, `599642360e17`, `7af5be3d8b7c`。

vLLM FP8 KV以uint8代码存储，fp8 serde旧把代码数值cast成float8再转回，L2往返改变大多字节；新对所有1byte元素逐字copy，较宽dtype才照旧量化，CPU枚举256字节/两FP8格式与每向独立回归。payload无dtype/版本头，新旧编码无法辨认，升级必须作废旧版为1byte KV写入的L2条目，且新旧服务器不能共读写同L2；无本次真实GPU质量对照。multi-chunk最后一层把有效chunk协商扩到SGLang/ATOM/TRT-LLM并按KV块对齐，scheduler/worker同size；未给性能数字。MP prefetch在L1释放capacity主动notify，CPU1MiB池/八64KiB块交接DRAM约0.49秒→亚毫秒，file约0.5秒→4–7ms，属无其他唤醒的队列微场景、非模型TTFT。建议FP8 L2用户优先采用，chunk混用和排队负载再复现。

[LMCache/LMCache#6129](https://github.com/LMCache/LMCache/pull/6129) · [LMCache/LMCache#6893](https://github.com/LMCache/LMCache/pull/6893) · [LMCache/LMCache#6966](https://github.com/LMCache/LMCache/pull/6966)

### FlashAttention4扩展Ampere与RTX Blackwell的paged KV前向

**项目／日期／版本：** Dao-AILab/flash-attention；合入／主干落地UTC 2026-11-27T19:14:08Z；`4b6dee0c070a`。

把SM90 paged-KV manager与边界mask接到SM80/SM120通用前向，V保token×dim布局、逻辑head维padding和页边界分开mask；同时让block-sparse/score-mask坐标使用实际容量。最终interface显式拒SM12x pagedKV+custom score_mod+aux_tensors，pagedKV+top-k gather也仍拒；不能只凭标题泛化全flex组合。diff含相应paged/不等head维/稀疏与padding回归，但PR正文只有简短能力说明，未给系统软件版本、设备性能或正式wheel范围。建议相关GPU的paged attention实现深入读guard与测试，部署速度继续观察实测。

[Dao-AILab/flash-attention#3060](https://github.com/Dao-AILab/flash-attention/pull/3060)

### llama.cpp把量化scale纳入GGML，Inkling2Mini可保留原NVFP4量化

**项目／日期／版本：** ggml-org/llama.cpp；合入/落地主干UTC 2026-11-27T13:51:57Z–2026-11-27T14:44:38Z；`a386950136a8`, `4c7344447cb1`。

GGML MUL_MAT/MUL_MAT_ID可关联FP8/NVFP4 scale，模型图统一通过支持scale的build_lora_mm，CUDA在epilogue乘scale，其他backend目前显式拒带scale运算；224文件全分页diff说明这是接口/模型图改造而非所有硬件已支持。Inkling2Mini额外适配3D fused专家，使NVFP4+MXFP8+BF16原工件可转NVFP4+Q8_0，converter重排nibble/交错gate布局而非宣称文件字节相同；列宽非64倍仍回退反量化，只转文本backbone、跳视觉/MTP，conversion仍由环境变量门控，不等于任意混合模型通用转换。两PR未给独立质量/吞吐表。建议本地低精度运行者深入阅读权重/scale配对和backend支持，再测目标GGUF。

[ggml-org/llama.cpp#28898](https://github.com/ggml-org/llama.cpp/pull/28898) · [ggml-org/llama.cpp#33062](https://github.com/ggml-org/llama.cpp/pull/33062)

### llama.cpp recurrent状态直接view，明确读取先于覆盖

**项目／日期／版本：** ggml-org/llama.cpp；合入/落地主干UTC 2026-11-27T09:48:11Z–2026-11-27T11:15:31Z；`7b74cb92d673`, `2a8bf0f59347`。

状态选择是identity时用cache view免get_rows往返，RWKV6/7先读token-shift旧状态再存新状态，非融合chunk DeltaNet也约束输出先于state覆盖。RTX5090/Windows/CUDA13.4、512prefill/128decode/3轮，Qwen3.6-27B Q4_K_M单流约+1.9%，32并行约+16.2%，Llama控制≈不变；Vulkan部分模型可慢1.5%，HIP未正式基准。logits多后端逐位同，RWKV7Metal的微小PPL差与command-buffer拆点相关。另NVFP4 stream-k尾tile曾读y_scale越界但不写无效列，padding scale多J−1消memcheck错，359核例在5090/GB10通过、SASS不变，无速度收益。建议recurrent并发负载复现图复用与读写顺序，并优先采用越界修复。

[ggml-org/llama.cpp#30139](https://github.com/ggml-org/llama.cpp/pull/30139) · [ggml-org/llama.cpp#33136](https://github.com/ggml-org/llama.cpp/pull/33136)

### Transformers修kernel化norm丢TP梯度归约

**项目／日期／版本：** huggingface/transformers；合入／主干落地UTC 2026-11-27T15:17:46Z；`1aa850f87ec6`。

ReplicatedWithGradAllReduce原包module.forward挂梯度all-reduce，后续use_kernels再换forward就丢wrapper；新改forward pre-hook使替换不移走hook。影响Qwen3 q/k norm及同类replicated norms，模型仍能训练但梯度错。新用kernels/no-kernels norm-grad回归，作者在默认kernel化候选栈上七项TP/EP/FSDP2比较从失败到通过；未给大型训练收敛或性能对照。建议Transformers TP训练且启kernel者优先核查，不能据当前norm受影响推断未来所有模块hook都安全。

[huggingface/transformers#51339](https://github.com/huggingface/transformers/pull/51339)

### Transformers serve分离API与模型进程，性能表用假模型

**项目／日期／版本：** huggingface/transformers；合入／主干落地UTC 2026-11-27T11:56:26Z；`f6b7cbef762d`。

HTTP/template/tokenize/detokenize与模型生成不再共享Python进程GIL，多API进程可共享同一model process，需命令行force model。模型进程死则health503并退出，stream/usage行为也按请求显式字段。作者256客户端、73或10000入、32或1000出、模型被立即回答fake替代的前端对照，双API约124K对vLLM90K流token/s；这是API工作，不是H100模型吞吐。H100/M2仅说明试过架构，未给真实模型配对端到端。建议轻模型/高CPU前端压力者考虑复现，评测保留tokenization、队列和模型时间。

[huggingface/transformers#50927](https://github.com/huggingface/transformers/pull/50927)

### Transformers多模态与assisted-decoding修复需同步下游模板

**项目／日期／版本：** huggingface/transformers；合入/落地主干UTC 2026-11-27T06:33:25Z–2026-11-27T17:15:52Z；`a779a0d3598a`, `e6f44c33fe6f`, `e5bbd1d6fe33`, `44ee1da4158b`。

Qwen3VL视频processor曾只替video_pad，保留chat模板外vision wrapper又给每帧wrapper，5.10起多包一层；现整体替换恢复5.9格式，vLLM曾适配旧错版需协调。PaliGemma不等长未padding输入逐序列构labels，保prefix−100，避免NumPy ragged-array异常。assisted decoding改跟踪上次assistant input长度再crop，不用易错长度推算；正文未给广泛设备精度对照。Gemma4/Unified audio_outputs在5.24.0/1误丢返回值，现恢复并透传；logits/generate本不受影响，不能说修转录能力。前三者有对应CPU/processor或已有代码回归，建议更新processor/生成工具链者比对promptIDs、labels和cache回滚，而非只做能启动检查。

[huggingface/transformers#51229](https://github.com/huggingface/transformers/pull/51229) · [huggingface/transformers#51222](https://github.com/huggingface/transformers/pull/51222) · [huggingface/transformers#51384](https://github.com/huggingface/transformers/pull/51384) · [huggingface/transformers#51301](https://github.com/huggingface/transformers/pull/51301)

### TRL按训练模板结束token停止在线生成，H100/B200 SDPA默认pad128

**项目／日期／版本：** huggingface/trl；合入/落地主干UTC 2026-11-27T09:49:14Z–2026-11-27T13:40:21Z；`b00c6f3e8dec`, `39a92dee8eb7`, `8a59361a235e`, `b5266b7219d1`。

GRPO/RLOO/蒸馏/AsyncGRPO及实验online trainers把conversational模板的end-of-turn补进generation_config及vLLM stop_token_ids，免Qwen/Gemma漏EOS一直生成、被当截断并对轮后文本算reward；用户generation_kwargs仍能覆盖。另七trainer的auto padding在CUDA9.x/10.x且实际model用SDPA时才选128，FA/eager保None，避免cuDNN按每新长度积host graph；H100 torch2.14.1示例RSS3.54→2.13GB稳定，历史最高57%为旧测不泛化，generation自身仍可能增长cache。vLLM0.34 colocate还cap batched_tokens≤max_num_seqs×max_model_len，修小256窗口warmup越表非法访问，H100两次训练smoke通过。建议在线训练优先核停止token与attention实现，升级对batch形状/显存重测。

[huggingface/trl#8606](https://github.com/huggingface/trl/pull/8606) · [huggingface/trl#8607](https://github.com/huggingface/trl/pull/8607) · [huggingface/trl#8554](https://github.com/huggingface/trl/pull/8554) · [huggingface/trl#8608](https://github.com/huggingface/trl/pull/8608)

### VeOmni新增MiniMax M3 VL SFT，完整模型与rebase硬件验证未完

**项目／日期／版本：** ByteDance-Seed/VeOmni；合入／主干落地UTC 2026-11-27T07:14:05Z；`d60841cf56fe`。

Transformers5.16.1 pin下接packed TND MSA、SwiGLU-OAI fused MoE/EP、Ulysses序列并行、processor和runtime checkpoint转换，GPU/NPU有配置。作者rebase到VeomniOp前测约8B减层模型四卡EP2/Ulysses2、混图文视频406步，A3/A5对H100后期loss相对差≤约0.25%；全深度减4专家约30B十六设备仅文本走通。完整MiniMax M3未训，最终registry重构后的设备复核仍在进行；CPU/源级回归不是新GPU签收。建议需要该模型微调者深入读算子/权重/模板，采用前补最终head与完整规模验证。

[ByteDance-Seed/VeOmni#1315](https://github.com/ByteDance-Seed/VeOmni/pull/1315)

### Megatron-Bridge修交错GLU权重转换，并统一MXFP8配置所有权

**项目／日期／版本：** NVIDIA-NeMo/Megatron-Bridge；合入/落地主干UTC 2026-11-27T16:37:40Z–2026-11-27T16:39:08Z；`e779077354a8`, `3197f33e7cae`。

interleaved FC1 gate/up原按连续两半切，形状正确却权重错；新每TP shard按实际block布局转canonical，先反量化再重排、拒rawFP8导出避免scale错位，checkpoint同用MCore ShardedTensorFactory。403项CPU测试及真torch_dist往返通过，GPU/rollout未测，NeMoRL自身旧wrapper需随Bridge bump去掉以免双反交错。MXFP8另让model.recipe压optimizer.recipe、optimizer reuse（含False）压DDP副本，在optimizer finalization前派生；离线822正式配方有效设置无变，旧GB200两卡六步验证不是本最终优先级跟进新测。建议权重迁移与手配MXFP8者优先核保存/导出/恢复一致性。

[NVIDIA-NeMo/Megatron-Bridge#7101](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/7101) · [NVIDIA-NeMo/Megatron-Bridge#7027](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/7027)

### SpecForge增加FP32状态FSDP2及单向旧检查点迁移

**项目／日期／版本：** sgl-project/SpecForge；合入/落地主干UTC 2026-11-27T04:31:45Z–2026-11-27T06:02:34Z；`214c5b494f6c`, `2e66bd0df2ea`, `566220756a3a`。

新fsdp2 runtime把可训参数/浮点buffer与微批累积、归约置FP32，compute/artifact精度分开，冻结target不升；直接驱动registry optimizer，拒CPU optimizer-offload和USP。FSDP1/DDP旧master/moment可按host replay布局逐参数还原到DTensor，只有精度相同的fsdp→fsdp2迁移可用，反向/改精度先拒。检查点进一步等所有rank原子分片写成功后才发布training_state.pt完整标记，四阶段collective错误一致；负注入包括rank1写失败、rank0rename失败与旧完整step保留。两rank Gloo/CPU精确恢复及轨迹对照支持契约，三PR均无CUDA验证。建议草稿模型训练需要高精度累积/迁移者深入阅读，先补设备与吞吐成本。

[sgl-project/SpecForge#1244](https://github.com/sgl-project/SpecForge/pull/1244) · [sgl-project/SpecForge#1245](https://github.com/sgl-project/SpecForge/pull/1245) · [sgl-project/SpecForge#1247](https://github.com/sgl-project/SpecForge/pull/1247)

### SpecForge减少DFlash2/DSpine重复全词表扫，修恢复与Mooncake并发

**项目／日期／版本：** sgl-project/SpecForge；合入/落地主干UTC 2026-11-27T08:59:39Z–2026-11-27T12:00:32Z；`d3c36cda0bba`, `5426ef6d3327`。

CE/CE+L1同一pass带top-k候选，DFlash2/DSpine复用term0统计免第二全词表candidate pass；diagnostic teacher-L1可detach免保teacher logits反向。fused=True不支持就报具体拒绝理由，Triton探测读实际cuda/hip/npu backend，7蒸馏token loss注册NPU并不等价已设备验证。另resume分片读错/坏shared字段跨rank同步失败，UUID claim/文件名免容器PID1冲突；Mooncake close与put/get准入同锁、读在途延后reclaim，HTTP inbox保UTF8跨块与截断失败offset不推进。证据是CPU stub与真实CPU回归，未有本轮CUDA/NPU性能或端到端训练。建议分离草稿训练者深入读生命周期与恢复，性能收益继续观察。

[sgl-project/SpecForge#1250](https://github.com/sgl-project/SpecForge/pull/1250) · [sgl-project/SpecForge#1253](https://github.com/sgl-project/SpecForge/pull/1253)

### slime更新MI300/325的ROCm整栈镜像，性能证据尚未披露

**项目／日期／版本：** THUDM/slime；合入／主干落地UTC 2026-11-27T08:04:26Z；`e3b669fd1919`。

镜像改基于rocm/primus:v26.5，保其torch/Triton/AITER/FA/TE，SGLang0.5.21、Megatron core0.19.2与patch stack对齐CUDA线；构建gfx942的sgl-kernel和memory saver。启动入口取消Primus预置NVTE注意力覆盖，避免与--attention-backend冲突，Ray不再重写HIP_VISIBLE_DEVICES。Qwen3-4B AMD教程与已有recipe统一；slime仍构建时clone main，非全栈不可变快照。PR未给同机吞吐、长期训练或模型数值比较。建议MI300/325部署者采用这套版本契约再考虑复现，不把镜像更新写成量化提速。

[THUDM/slime#2506](https://github.com/THUDM/slime/pull/2506)

### ms-swift动态重采样不再重复同一prompt批

**项目／日期／版本：** modelscope/ms-swift；合入／主干落地UTC 2026-11-27T07:48:38Z；`7bbbe488b19f`。

主训练RepeatSampler为num_iterations×steps_per_generation×SP重复micro-step，旧resample迭代器继承它，动态零方差/截断重试反复拿同prompt。新resample_mode把repeat_count设1并以finally还原seed与模式，正常训练复用保留。七CPU真实dataloader回归旧失败新通过，completion/reward为stub；完整多GPU/NPU vLLM rollout未跑。建议动态采样RL训练采用并验证“重试确实换新prompt”，勿从重复消除推断最终reward提升。

[modelscope/ms-swift#10632](https://github.com/modelscope/ms-swift/pull/10632)

### vLLM-Omni MRv2恢复teacher hidden-state逐请求返回

**项目／日期／版本：** vllm-project/vllm-omni；合入／主干落地UTC 2026-11-27T06:35:10Z；`631f273e7b42`。

请求typed return_hidden_states沿sharedCUDA/NPU MRv2路径返回完成prefill的rows到teacher_hidden_states，未opt-in免copy，主流等slice copy防图replay覆写。chunked prefill、prefix hit或缺counter都明确拒，OPD teacher必须禁chunking/前缀缓存。H800节点Qwen3Omni相关MRv2测试，另GSPO两GPU/FlowGRPO四GPU原包recipe smoke通过；hidden-state e2e本身用server double不导verl，完整OPD未跑，CUDA图只用tiny权重，NPU虽共享实现本次未设备验证。建议多模态OPD数据链路考虑复现，锁MRv2与teacher配置。

[vllm-project/vllm-omni#11000](https://github.com/vllm-project/vllm-omni/pull/11000)

### llm-d多集群会话亲和：退避与已迁会话表仍局限单hub

**项目／日期／版本：** llm-d/llm-d-router；合入／主干落地UTC 2026-11-27T07:37:30Z；`cd793eea4fa6`。

weighted rendezvous以ready pod数哈希session（agent identity/session header/prefix），满/不可用才spill并在30min本地表记新归属，避免每请求逐prompt block给每cluster算cache分。两站约30ms链路、三小时配流对照TTFT p50/p99 3.0/72→1.1/39秒、prefix hit34–42→52–73%，但模型/硬件/精确流量未完整披露不能作通用SLA。moved表每hub独立，active-active曾有30–60%spill session翻回，单hub/active-passive才0；failover仍有在途请求错，primary kill约8秒错误。建议跨集群服务深入读亲和/负载取舍，active-active先继续观察共享状态后续。

[llm-d/llm-d-router#4475](https://github.com/llm-d/llm-d-router/pull/4475)

### PyTorch NVGEMM动态形状扩至多算子，图复用不等于kernel工件复用

**项目／日期／版本：** pytorch/pytorch；合入／主干落地UTC 2026-11-27T22:36:51Z；`02966beb7f87`。

在前轮dense-mm guard机制上扩addmm/BMM、NVFP4/MXFP8 scaled-mm与未scaled grouped-GEMM；候选筛选不装guard，只有autotune胜者提交layout/scale数量/对齐/index/grid要求，symbolic只接审过的零workspace provider。scaled-grouped仍拒；scale要求32byte而Inductor只保16byte的候选必须排除，可能改变静态选择。两GB200/CUDA13.1同binary SDK，基线357通过/2跳过→367/2，无性能表。Dynamo一图能收多个shape，但artifact key仍具体shape/stride，新签名可JIT且cache无界。建议动态低精度编译深入阅读launch契约与缓存成本；此项按主干commit核实，GitHub PR为closed-unmerged，不写GitHub“已合并PR”。

[pytorch/pytorch#200458](https://github.com/pytorch/pytorch/commit/02966beb7f87bee7dac795c5f02c1d8bc98972dd)

### PyTorch保持TokenSwitch原生group寿命，B2B启发式避免symbolic特化

**项目／日期／版本：** pytorch/pytorch；合入/落地主干UTC 2026-11-27T01:38:49Z–2026-11-27T07:32:17Z；`2cd56519bc79`, `8006bc95d291`。

NCCL-EP routing handle可能比创建它的switch活得久，原生析构仍解group；新intrusive_ptr强持group至handle销毁，4×GB200释放switch后再dispatch的行distinct payload回归旧报internal error新过。另一项B2B GEMM利润判定用backed优化hint、不为代表值添guard，free unbacked先回退；XPU早已可用，本次移旧skip并验证真走B2B，未合入历史B60自定义候选性能方案。建议通信生命周期优先采用，动态B2B按实际shape和编译成本复现；两项均依据实际默认分支commit而非GitHub merged flag。

[pytorch/pytorch#204971](https://github.com/pytorch/pytorch/commit/2cd56519bc79d6875925ba4d9d25b8cf331a0abe) · [pytorch/pytorch#189675](https://github.com/pytorch/pytorch/commit/8006bc95d29126baf4f57d8e71b39da53fc47480)

**交付版本：** [OpenRLHF v0.12.0](https://github.com/OpenRLHF/OpenRLHF/releases/tag/v0.12.0)，2026-11-27T01:39:16Z正式发布，含上文新正确性修复并升级vLLM0.34/DeepSpeed0.20.1。其余本节按指定源码commit叙述，不自动视为已出稳定wheel。

<a id="topic-6"></a>
## 硬件软件协同

本节优先展开可操作机制和正确性边界。下列性能均为作者实测或其报告，本轮未独立跑设备实验；已合入PR不代表已发布稳定包。

### Firefly把NVFP4 MoE的路由、GEMM和中间量量化一起设计，仍是显式实验后端

**项目／合入日期／版本：** flashinfer-ai/flashinfer#9995 `077b3b1337`（11-27 21:06，UTC+8）。

FlashInfer新增CuTe DSL的NVFP4权重×NVFP4激活专家后端：短批直接按路由计算，中批由路由计数驱动grouped GEMM，长批扫描并打包专家词元；第一层结果直接在epilogue量化，省去常规BF16中间量落地。`FireflyNvfp4Config`只接受SM100/SM103，须显式列入候选；已合入不等于默认auto会选，也不因SM107测试可通过就宣称对Rubin正式支持。H须256对齐、I须128对齐，top-k≤32、专家≤1024；只接受预计算路由，拒绝融合共享专家、4over6和不执行finalize的配置。作者以九种模型形状、M=1–16384、每表135格，与11月21日基线各格最快NVFP4后端比较，B200几何均值1.031–1.094倍、GB300为1.045–1.106倍；并非全格更快，也没有整模型吞吐或能力榜；激活量化与top-k路由不计时，测的是MoE核段。完整四表的独立harness未随PR交付，树内bench仅短burst，严格复现尚需向作者取得测量脚本。非融合finalize固定FP32归约可确定，融合finalize用BF16原子而不确定；与旧后端并非逐位等价。最大词元8192时，九形状非融合workspace 0.4–3.8GiB，融合0.06–0.58GiB；非图调用还有50–81µs主机开销。

**建议：** 值得深入阅读并按真实路由和并发考虑复现；把能力、确定性、workspace及原有量化误差一起验收。 [#9995](https://github.com/flashinfer-ai/flashinfer/pull/9995)

### TensorRT-LLM把JIT编译分工放进预热，收益主要在启动阶段

**项目／合入日期／版本：** NVIDIA/TensorRT-LLM#22374 `4e0fddd85a`（11-28 08:23，UTC+8）。

新增KernelBuilder，将构建模式与运行时模式分开；节点内进程默认共享DeepGEMM/CuTe DSL JIT工件，长编译可交给每rank进程池，工件claim失败或900秒默认超时后由当前进程回退构建。池仅用于显存剖析预热，在KV cache定容前退出；实际只给MiniMax-M3 MSA等长构建启用，短核放池反而更慢。14个GB300性能sanity配置从空缓存比较，MiniMax-M3示例TP4并发64启动1533→1156秒，服务3307→3290tok/s；Kimi Helix16并发32启动476→479秒，并非所有配置改善。部分端到端测量还带日志读取修复#22782；这是冷启动证据，不能写成稳态吞吐普遍提速。

**建议：** 建议关注多rank重复JIT的部署；锁构建源、节点本地目录和超时回退，并区分构建共享与分布式调优。 [#22374](https://github.com/NVIDIA/TensorRT-LLM/pull/22374)

### 多模态breakable prefill图进入TensorRT-LLM，但输入匹配与回退是核心契约

**项目／合入日期／版本：** NVIDIA/TensorRT-LLM#21333 `b976c4136c`（11-28 06:03，UTC+8）。

新hook将当步embedding、MRoPE等输入放进图持有buffer，重放前刷新；shape、dtype、步长、地址或主机值不匹配即eager回退，attention数据并行各rank先一致投票。编码器和embedding融合仍在图外，启动只接受已声明完整输入契约的wrapper以及单流水线stage。GB200单卡TP1、并发4、32请求、1输出词元筛查：Qwen3.5-35B-A3B FP8图像TTFT p50 347.7→155.1ms、文本167.9→52.7ms；四模型图片降9–55%、文本降19–69%。这些是较早revision的共享节点筛查，后续少拷贝和扩大重放范围未重测。Qwen3-VL示例启动约18→25秒、额外staged输入24MiB；Gemma4双向vision注意力仍回退。多checkpoint输出词元匹配不代表所有logit逐位一致，长输出吞吐和未测TP组合仍空缺。

**建议：** 建议考虑复现短prefill和多模态TTFT受主机启动开销限制的服务，保留强制回退与跨rank一致性测试。 [#21333](https://github.com/NVIDIA/TensorRT-LLM/pull/21333)

### Transformer Engine修正MXFP8尺度越界，最终支持范围比PR正文更窄

**项目／合入日期／版本：** NVIDIA/TransformerEngine#3710 `0c4a13f48b`（11-27 10:31，UTC+8）。

原量化核假设scale buffer已经完整padding，裁剪外维或非16字节对齐view可能写越界、漏写或MisalignedAddress。最终`quantize_mxfp8.cuh::quantize`增加尺度行数边界、2维布局校验和对齐选核，CuTe DSL不适用布局回退C++实现。**以最终diff为准：unswizzled只允许省外维padding；连续维仍严格保规范padding**，rowwise列数为ceil(cols/128)×4，columnwise列数为ceil(cols/128)×128，不能照正文把任意每32元素一个尺度的紧凑矩阵都当合法。正文B200 180例旧基线仅53通过证明原故障面，不是最终更窄契约的通过数；GEMM对compact scales的支持也未在本PR扩大。

**建议：** 建议采用越界与对齐修复，先核调用方尺度shape；文档和源码冲突在此明确保留。 [#3710](https://github.com/NVIDIA/TransformerEngine/pull/3710)

### FP8张量view保留转置缓冲，图捕获退出则恢复临时全局状态

**项目／合入日期／版本：** NVIDIA/TransformerEngine#3921 `beb5a64133`（11-27 12:39，UTC+8）；NVIDIA/TransformerEngine#3862 `37be77c987`（11-27 22:11，UTC+8）。

Float8Tensor最后一维不变的leading维view/reshape现在复用转置storage，并使用解析后的shape处理-1，避免归一化恢复三维后丢掉转置、反向重复转置以及torch.compile真假元数据不匹配；最后维变化仍失效。CUDA图工具另先验证所有模块，再用ExitStack和finally恢复继承的类调用、FP8/RNG状态、输入梯度、GC与临时hooks；warmup流结束前与调用流建立依赖，捕获和清理同时失败时保留原错误为主异常。图修复作者在A6000的Python模块覆盖TE2.20.2原生二进制验证13例，另有Megatron组合回归；未完成原生重编译、完整L0与FP8图执行认证，临时改类dispatch仍要求主机串行访问。

**建议：** 建议采用对应正确性修复；不要把异常后可重试扩大成致命CUDA错误可恢复。 [#3921](https://github.com/NVIDIA/TransformerEngine/pull/3921) · [#3862](https://github.com/NVIDIA/TransformerEngine/pull/3862)

### 昇腾短prefill分片同时收紧MegaMoE预算，大收益有固定KV与内存压力条件

**项目／合入日期／版本：** vllm-project/vllm-ascend#22983 `3b66dc3d3f`（11-27 15:42，UTC+8）。

MRV2的PCP（预填充上下文并行）与DCP（解码上下文并行）组合不再因空尾块把225词元请求整份复制给PCP8每个rank：保留完整head并平衡tail后每rank28或29词元，Indexer仍按一致30行计价。最大仍需复制长度界收紧到3P−1，并用于启动profile/MC2预算。在Kimi K3.1 Pro W4A8 MXFP、32张A5、TP1/PP4/PCP8/DCP8、eager、MegaMoE2、3草稿、固定246 KV块、24576批词元、并发20、131072输入/1输出的prefill-only三轮A/B中，吞吐0.8394→2.2861req/s，平均TTFT22.889→8.652秒，PP3分配重试194/192/194→0/0/0；规划器每rank估少7.68GiB，与采样设备HBM变化是两种口径。相同工作量与重试消失支持“预留减小释放瞬时分配余量”的解释，具体重试分配点未证实。16384预算同KV的1P1D则基本中性，增大KV池又出现重试；64NPU GPQA 184/198与另配置185/198不能隔离本PR准确率。

**建议：** 建议深入阅读并在接近显存边界的同配置复现；不能把2.7倍泛化为所有PCP负载或端到端P/D收益。 [#22983](https://github.com/vllm-project/vllm-ascend/pull/22983)

### 昇腾复用GDN缓存组的请求布局，主机元数据成为可量化优化点

**项目／合入日期／版本：** vllm-project/vllm-ascend#20345 `0bace00d47`（11-27 22:08，UTC+8）。

MRV2对相同cache spec和builder类型只完整构造一次GDN（门控Delta网络）请求布局，其余组绑定独立物理状态索引并批量更新各自捕获buffer；checkpoint组及带step hook的builder保留独立完整构建，不混用物理页。作者同一A3、Qwen3.5-9B BF16、DFlash7草稿、TP1、同置P/D、完整decode图、16K输入/256输出、预热后两轮，c1平均TPOT5.787→3.829ms，c4为6.941→4.836ms；10组配对词元完全匹配。诊断运行c1同34目标轮，完整build816→34，另782次组内更新；746单测及29NPU算子/图例通过。

**建议：** 建议考虑复现多GDN缓存组的混合模型；收益取决于元数据是否在临界路径，不能只按核时分析。 [#20345](https://github.com/vllm-project/vllm-ascend/pull/20345)

### HiSparse把Device GET也限为256MiB，解决读租约到期而非增加带宽

**项目／合入日期／版本：** vllm-project/vllm-ascend#23442 `84894c1de0`（11-27 21:01，UTC+8）。

MemCache Host与Device GET都取读租约，旧256MiB预算只限Host，A5 TP8近1M GLM-5.3前缀每rank8183个C8键一次Device读会报-3106并返回HTTP500。现按两方向字节分批、超大单key保持完整，逐批保留SDK状态。A5 950DT/MemCache1.3 host_tcp、60秒租约三轮独立SDK比较，Device单批与256MiB分批总时40.18/40.28秒，最长单调用40.30→1.39秒；默认10秒时单批八rank全失败、分批成功。模型16K/131K/近1M回放完成证据另带Resident DMA修复#23336，仅证明请求完成与外部传输；不是本PR单独的整模型质量或TTFT验证。

**建议：** 建议采用租约边界修复，验收分批失败与重试；不要把更短单调用误写成更高总带宽。 [#23442](https://github.com/vllm-project/vllm-ascend/pull/23442)

### 昇腾先量化再PCP AllGather，通信字节减少但INT8舍入并非逐位相同

**项目／合入日期／版本：** vllm-project/vllm-ascend#22230 `ae3168938f`（11-27 16:04，UTC+8）。

W8A8 INT8及A5 W8A8 MXFP8 MoE在本地padding行量化后，按相同行序gather量化值与scale，省浮点激活传输及跨rank重复量化；LoRA、定制路由和把路由权重先乘输入的层保留浮点。最终两张950DT、CANN9.3、torch-npu2.10.post7真实HCCL回归覆盖空rank和padding：INT8路由/专家计数一致，值最多差一量化步；MXFP8字节匹配。较早32张910B3、DeepSeekV4 W8A8 P/D实验固定负载输入吞吐+3.7–6.6%，准确率配对区间均跨零；这批性能/能力结果早于重基线，且P/D放置随arm变化，不能当最终MXFP8栈同机单变量认证。

**建议：** 建议在通信受限prefill考虑复现，保持最终revision、精度与部署放置一致，并单独验收量化误差。 [#22230](https://github.com/vllm-project/vllm-ascend/pull/22230)

### 昇腾split-query DCP与DSpark C8融合分开交付，图缩批修复有独立价值

**项目／合入日期／版本：** vllm-project/vllm-ascend#23220 `2b05dcb303`（11-27 22:32，UTC+8）；vllm-project/vllm-ascend#23015 `a5a652874c`（11-27 17:44，UTC+8）。

DCP路径可分传latent/RoPE及当前KV页，历史与当前输出各用独立mask，pack/combine时规范化空行并预热mask变体；#23220单独使用仍走原attention provider，没有新增device tiling、也未消除投机host长度同步，最终验证为CPU。#23015对合适A5 Kimi C8草稿以npu_mla_prolog_v3融合BF16投影/RoPE，再转FP8，DCP先gather头；capture预留metadata并把replay padding拆进每请求query上限，修正缩批图故障。较早93层8专家目标的TP2/DCP1草稿预处理134.9→90.1µs/层；完整八节点目标P/D尚待验证。

**建议：** 建议按两个边界分别验收；不要把执行接口、设备tiling与完整P/D支持合并宣称。 [#23220](https://github.com/vllm-project/vllm-ascend/pull/23220) · [#23015](https://github.com/vllm-project/vllm-ascend/pull/23015)

### Gaudi修复KV加载失败仍解码，以及异步流水线的回滚污染

**项目／合入日期／版本：** vllm-project/vllm-gaudi#2038 `a35c24eddb`（11-27 19:17，UTC+8）。

HPU runner原只回报完成传输，模型步中加载失败的block不进scheduler，默认fail策略也可能从未写KV继续输出。现在同步/异步都上报invalid_block_ids；recompute时回滚到所有在途步之前，清理输出占位、草稿与旧采样，丢弃基于坏KV的在途输出。Gaudi3、固定vLLM基线、故障连接器明确注入不写KV的外部命中，旧版三种场景均词元不符、默认fail也错误结束为length；新head同步/异步重算输出与清缓存参考逐词元相同，fail只让受影响请求error。请求级failed_recving等其它字段仍未全透传，共享坏块同一步的行序依赖未覆盖，异步投机+KV连接器尚未运行。

**建议：** 建议优先采用并注入真实错误路径验证；“重算成功”必须同时证明故障确实注入与坏输出未泄漏。 [#2038](https://github.com/vllm-project/vllm-gaudi/pull/2038)

### Gaudi compact GDN前缀缓存回收死行，但默认路径原本已避开浪费

**项目／合入日期／版本：** vllm-project/vllm-gaudi#2019 `fcf8f15359`（11-27 19:19，UTC+8）。

显式VLLM_COMPACT_GDN=1且prefix caching开启时，旧state每组分配num_blocks×组数+1行，核仅索引每block一行，三组约2/3为死行；改共享tensor为num_blocks+1行，依赖全局BlockPool的跨组唯一block ID保持不碰撞。0为null guard，末行承接-1 padding；非prefix的私有slot布局不变。作者Qwen3.5混合模型比对prefix与非compact输出逐位相同并恢复KV容量；默认配置过去会自动禁compact，已经得到同布局，不能将容量收益计给全部HPU用户。compile非lazy compact路径还默认关闭Synapse持久输入reuse以免共享GDN状态被污染，显式环境设置仍优先。

**建议：** 建议显式compact部署采用；核清block编号唯一性与共享storage身份。 [#2019](https://github.com/vllm-project/vllm-gaudi/pull/2019)

### AMD跨GPU标志轮询改system scope，尚没有本head多GPU运行证据

**项目／合入日期／版本：** ROCm/aiter#8832 `0f8e5c18dc`（11-27 21:16，UTC+8）。

gfx9 ROCm custom AllReduce的start/end flags由对端system-scope写入，旧device-scope读尤其不能正确配对中间barrier的release。补丁所有poll改relaxed/system，观察到flag后仅中间barrier做一次system acquire fence，避免每次自旋都失效L2。源码只有该头文件，CUDA和gfx1250独立实现未变；gfx942/gfx950编译汇编核对支持指令差异，但本head没有多GPU正确性/性能运行。旧PR作者报告的生产挂起频率是动机，不能当本head已消除挂起的实证。

**建议：** 建议优先审阅内存模型修复并做多GPU压力验收；当前不宣称已获生产稳定性或吞吐增益。 [#8832](https://github.com/ROCm/aiter/pull/8832)

### FlyDSL FP8 attention补等待，普通随机算子测试漏掉了LDS竞态

**项目／合入日期／版本：** ROCm/FlyDSL#1503 `3c3b692707`（11-27 15:37，UTC+8）。

gfx950默认stagger双wave组中，后组第三阶段barrier早于其K/V搬入LDS完成，前组下一迭代可读半新数据且结果仍有限。现在释放前组前补waitcnt。作者MI350X、ROCm7.0、torch2.10压力复现：每次先写1GiB时100/100错→0/100，空闲307/600→0/600，HBM竞争233/400→0/400；151个原随机FP8测试在旧新版都通过。另一inline-asm LDS读追踪问题未在此修复。计时随机输入下新旧变化接近波动，不将此项包装为速度优化。

**建议：** 建议采用正确性修复，并保留带构造尖峰attention与带宽竞争的负对照。 [#1503](https://github.com/ROCm/FlyDSL/pull/1503)

### Blackwell同步修复要分TMEM可复现竞态、内存模型符合性和编译器约束

**项目／合入日期／版本：** NVIDIA/TensorRT-LLM#21848 `c67e03ec34`（11-27 13:43，UTC+8）；flashinfer-ai/flashinfer#10049 `811a244cd3`（11-27 10:16，UTC+8）；flashinfer-ai/flashinfer#9863 `62b46eabb8`（11-27 14:26，UTC+8）。

TRTLLM MiniMax低M split-K GEMM在MMA二次arrival前先等待barrier phase0，防止phase1 wait误将上一阶段判完成而提前释放TMEM；B200注入epilogue延迟并允许第二核共驻时，旧版20/20 launch错误、补丁0错。原出厂满共享内存占用掩盖了数据被别核覆写。跨CTA mailbox另改cluster relaxed wait+restricted acquire fence，未在硬件复现其错误，属内存模型符合性。FlashInfer给PDL（程序依赖启动）inline asm补memory clobber以阻止编译器跨指令移动访存，亦未复现坏结果；GB200/GB300回归通过。SM120 MegaMoE另把result-chunk步长从按64路由计数改按32计数，避免小batch偏斜路由的source-slot别名；该项为代码推导，尚待SM120真机验证，不写成已观测死锁修复。

**建议：** 建议采用对应正确性补丁，保留不同证据级别；结构AST检查不能代替并发压力运行。 [#21848](https://github.com/NVIDIA/TensorRT-LLM/pull/21848) · [#10049](https://github.com/flashinfer-ai/flashinfer/pull/10049) · [#9863](https://github.com/flashinfer-ai/flashinfer/pull/9863)

### ReplaySSM修正flush写回所有权，KDA残余开销尚未验收

**项目／合入日期／版本：** ROCm/ATOM#3618 `2610191c64`（11-27 11:21，UTC+8）。

GDN/KDA fallback必须所有value切片读完旧k/g记录后才能flush覆盖；补load wait与barrier再count arrival，由最后切片写k/g并reset，counter归layer记录池、WAIT在创建池时确定，native/SGLang图重放均传持久storage。gfx942修订把ownership获取放到replay收缩后，将GDN寄存器占用272→256、occupancy1→2，批256/T4的非flush相对main−0.39%、flush+0.81%。gfx950 32数值例通过，但KDA bs256 flush仍多7.86–17.36µs/调用；不把已合入写成性能验收结束，也不把小于漂移的非flush负差当提速。

**建议：** 建议采用读写顺序修复并继续观察flush代价；按目标设备、输出及全部可变状态一起验收。 [#3618](https://github.com/ROCm/ATOM/pull/3618)

### AITER的精确BF16 pull融合通信从注册图输入直接读，适用窗受驻留限制

**项目／合入日期／版本：** ROCm/aiter#8574 `a3ecc89426`（11-27 14:24，UTC+8）。

新增TP4/gfx950/H6144、BF16且Gemma RMSNorm专用pull路径，省inbox打包与中间拷贝；图capture交换注册输入地址，再按payload ladder选带读确认oneshot、单行两阶段pull或stream。策略表最终1–1024行，单行spin核不能越驻留上限，超过640行会有等待风险，故大行数改stream并验证ladder；非注册capture、其它宽度/精度仍回退。MI355X TP4、64调用图、ABAB各rank最大值对当前融合接口：M8约8.25–8.49→6.34–6.44µs、M32为13.4→8.6µs，无端到端测量。作者报告对所选reference数十亿元素一致；不能扩大为对所有旧两阶段rank归约逐位一致，host文档保留稀有求和顺序差异。

**建议：** 建议考虑复现图内小批decode；固定reference的求和/denormal语义并验证多次混合桶重放。 [#8574](https://github.com/ROCm/aiter/pull/8574)

### AITER单核AllToAll同时搬MXFP8码与尺度，端到端增益仍在噪声内

**项目／合入日期／版本：** ROCm/aiter#8587 `a76f1ad65b`（11-27 17:46，UTC+8）。

GroupCoordinator新增通用all_to_all/all_to_all_buffer，把peer块16字节pull直接写到GEMM所读布局，scale同launch搬运；生产者可直接写注册池，省两次RCCL及transpose。自定义路径限二维连续、scatter0、gather0或末维、16字节行、2/4/8rank、gfx9 IPC，其他走torch.distributed回退，peer shape一致性仍是调用前提。MI355X TP4/H6144图重放，4096/8192/16384词元为49.9/88.1/164.9µs，对pynccl send/recv为1.6–2.4倍且对RCCL字节一致；仅TP4已测，TP8/gfx942未实测。组合栈c32提升0.1–0.6%在噪声内，旧Triton版本单次16K prefill−1.8%也未反复ABAB。

**建议：** 建议深入阅读数据布局与注册池接口；用模型临界路径判断算子收益能否兑现。 [#8587](https://github.com/ROCm/aiter/pull/8587)

### QuickAllReduce舍入和小消息几何分别优化，后者改变浮点等价口径

**项目／合入日期／版本：** ROCm/aiter#7093 `8989520534`（11-27 10:12，UTC+8）。

INT4/5/6量化用FP16 magic-number加法替代round-even转换，穷举65536个FP16位型验证codec逐码一致；peer地址放kernel参数、仅本卡使用的epoch放普通device memory。gfx950/xGMI、2MiB以下mesh从256改64线程，使8KiB–1MiB更多workgroup并发；MI355X/hidden4096、10调用图、两轮均值，TP2/4/8延时约降6–16%。codec替换本身逐位一致，但块几何改变chunk owner，整个AllReduce在新小消息档只保证统计误差门槛通过，不能沿用“整算子逐位一致”。未采用的倒数近似曾让部分融合形状慢3–6%，并非所有低级变换均有益。

**建议：** 建议按通信量与数值容差考虑复现，分别评估codec和调度变化。 [#7093](https://github.com/ROCm/aiter/pull/7093)

### MoRI为EP8实模型形状调dispatch网格，收益在大词元而非普遍decode

**项目／合入日期／版本：** ROCm/mori#1106 `2a2296c5a3`（11-27 14:40，UTC+8）。

给机内HIP block-per-token派发补MI300X/MI355X形状/dtype分桶表，维持≤64词元每rank的64-block旧档，对大批改96/128等；没有增加网络带宽或减少payload。八卡EP8、pull combine、同进程hip对hip@64、交换顺序两轮，GLM4.7 Flash FP8的每rank16384词元dispatch在MI300X从818.1→570.7µs，MI355X878.3→673.7µs；其它形状少2–30%但部分中点持平或微退。≤128词元50迭代噪声可达数十%，不能声称小decode收益；gpt-oss FP8的尺度行90字节不满足dword约束，未覆盖。

**建议：** 建议按EP形状表复现prefill/训练交换，勿把dispatch核时间当完整MoE或模型吞吐。 [#1106](https://github.com/ROCm/mori/pull/1106)

### gfx942扩展FP8大头attention与Kimi MXFP4路径，基线精度须分清

**项目／合入日期／版本：** ROCm/aiter#7516 `9da5928186`（11-28 05:42，UTC+8）；ROCm/aiter#8211 `1c96e23ebe`（11-28 03:44，UTC+8）。

FP8 MHA仅head_dim_v=256切新配置，8warps与MFMA非K维16等替代原默认；MI325X、24query/4KV头、varlen、10次中位数，4096因果1.743→0.543ms，长形状最高3.8倍，比较的是旧FP8调度而不是BF16同数值精度；d>256会pad512并越LDS，明确排除。Kimi K3另一opt-in在gfx942/304CU、MXFP4权重×MXFP8激活、SiTUv2、M1024–32768用单阶段汇编，MI300X/MI325X比原a16w4快1.15–1.41倍；权重必须有专用relayout tag，bias/其它activation/scatter不适用。该对照激活精度不同，仍缺完整模型能力和吞吐认证，M≤512未由此覆盖。

**建议：** 建议按确切硬件、head维和混合精度路径选择复现；不将大形状调度收益或a8/a16比较扩成通用模型加速。 [#7516](https://github.com/ROCm/aiter/pull/7516) · [#8211](https://github.com/ROCm/aiter/pull/8211)

### AITER修正大packed反向地址与FP32 MLA漏写，两项证据强度不同

**项目／合入日期／版本：** ROCm/aiter#8748 `37303aabc9`（11-28 01:09，UTC+8）；ROCm/aiter#8655 `db4fabceb8`（11-27 10:21，UTC+8）。

gfx950 BF16 head256 varlen反向的三个二进制对象把每document基址乘积扩至64位，避免packed张量超过4GiB时错误梯度；作者把同文档拷到4GiB以上位置比较反向一致，前向64位仍未配套，本轮不反汇编重证二进制。MLA的kv_lora512/rope64、多KV头GQA prefill原写死每head64向量，FP32四元素向量应为128，导致半行漏写；现按维数/向量宽度计算。后者只有gfx950编译与Volta ISA符号验证（旧核拒绝，2304输出验证通过），尚未GPU运行；当前vLLM/ATOM已知BF16调用常数不变。

**建议：** 建议训练大pack采用地址修复；FP32 MLA修复需补真机，并区分算子接受的dtype故障与现有BF16模型实际受影响面。 [#8748](https://github.com/ROCm/aiter/pull/8748) · [#8655](https://github.com/ROCm/aiter/pull/8655)

### gfx950 paged attention可显式选BF16 partial与4-wave归约，默认不改后端

**项目／合入日期／版本：** ROCm/aiter#8085 `6f032bbd3a`（11-27 11:51，UTC+8）。

显式pa_persistent_fwd可用compact_partials=True把SplitO从FP32改BF16，LSE和累加仍FP32，原默认调用仍FP32；两native启动器按实际tensor布局选split_o_format，支持真实2/4/8wave符号与Q4分桶。限gfx950、page16、D128、GQA16、2–4query的MTP、BF16/FP16 Q/O、per-token FP8 KV及匹配mask/softmax配置。MI355X/ROCm7.2.4/PyTorch2.10，11形状、图内1次metadata+3层attention，较早测速revision对FlyDSL0.5.1几何平均延时−3.81%，部分真实形状不稳定；对stock FP32 PS的−12.49%包含规划及精度路径变化，不能只归因BF16。最终provider修订未重跑整表，无模型吞吐。

**建议：** 建议在支持契约内复现，锁比较后端与revision，单独评估partial精度。 [#8085](https://github.com/ROCm/aiter/pull/8085)

### FlashInfer sparse MLA重排PDL与TMA流水线，默认调用并未启用全部开关

**项目／合入日期／版本：** flashinfer-ai/flashinfer#9701 `dd58283ac5`（11-27 10:54，UTC+8）。

DeepSeekV4.1 DEP稀疏MLA将FP8主池tile用TMA直接送入保留K/V tile，Q在最后QK后先释放；可选PDL等待推迟到descriptor/barrier初始化后，但必须在TMEM分配及所有依赖全局读之前。额外MXFP8 scale四字节合存仅给提供store_scales hook的量化器。GB300、每rank64头、128SWA+512MAIN、图重放的旧revision示例1252行118.4→106.4µs，尚无端到端。FlashInfer现有入口都不启用deferred wait，输出quantizer也保留逐block存scale；TensorRT-LLM CSA2直接调用才启用两者。负对照延迟前驱并投毒Q/索引，不能实际重叠时测试记inconclusive skip，不能当成功。

**建议：** 建议深入阅读异步依赖与TMEM分配顺序；按真实入口核对哪些优化确实可达。 [#9701](https://github.com/flashinfer-ai/flashinfer/pull/9701)

### cuDNN前端修正FROST split和placement，合入的是有限形状选择规则

**项目／合入日期／版本：** NVIDIA/cudnn-frontend#2936 `55db0abf9c`（11-27 19:28，UTC+8）；NVIDIA/cudnn-frontend#2932 `cce3d38a59`（11-27 11:46，UTC+8）；NVIDIA/cudnn-frontend#2941 `cefbd2f4f8`（11-28 01:32，UTC+8）；NVIDIA/cudnn-frontend#2931 `34f347ec1b`（11-27 10:08，UTC+8）；NVIDIA/cudnn-frontend#2938 `6d31a2ce1b`（11-27 23:00，UTC+8）；NVIDIA/cudnn-frontend#2939 `29d6960236`（11-28 01:27，UTC+8）；NVIDIA/cudnn-frontend#2942 `1ae1b4fe27`（11-28 02:11，UTC+8）；NVIDIA/cudnn-frontend#2949 `03a66632d4`（11-28 06:48，UTC+8）。

develop连续更新SDPA计划选择：Blackwell/Rubin paged THD prefill可在已声明到131072 keys的缓存上提出split；短滑窗FP8 decode若unsplit已满1.5波CTA且窗口≤20KV tile则不盲按全cache split。Rubin的部分FP16 MTP/decode可借用已测BF16 placement，dense causal chunk、paged MHA、非pack GQA及exact ragged split-prefill则按实际计划/launch类别分键，修正好核已存在却选不到或误选的情况。它们没有改底层算子能力或计算核，也没有证明所有形状更快；若干初版规则holdout失败已被排除。每个PR的阈值、训练/留出shape和反例均独立，不能把局部极值几倍提升相乘或当整模型加速。

**建议：** 建议后端集成者深入阅读split chooser与placement的耦合；升级后用实际图声明、dtype、cache上限与滑窗形状复测。 [#2936](https://github.com/NVIDIA/cudnn-frontend/pull/2936) · [#2932](https://github.com/NVIDIA/cudnn-frontend/pull/2932) · [#2941](https://github.com/NVIDIA/cudnn-frontend/pull/2941) · [#2931](https://github.com/NVIDIA/cudnn-frontend/pull/2931) · [#2938](https://github.com/NVIDIA/cudnn-frontend/pull/2938) · [#2939](https://github.com/NVIDIA/cudnn-frontend/pull/2939) · [#2942](https://github.com/NVIDIA/cudnn-frontend/pull/2942) · [#2949](https://github.com/NVIDIA/cudnn-frontend/pull/2949)

### Triton AMD修正scale组验证与重复寄存器，并开放descriptor原子

**项目／合入日期／版本：** triton-lang/triton#12691 `2b7c54326a`（11-28 00:09，UTC+8）；triton-lang/triton#12964 `b5c80913c2`（11-28 02:06，UTC+8）；triton-lang/triton#12857 `430c09cf92`（11-28 00:16，UTC+8）。

scaled_downcast verifier按目标使用CDNA4两unique值、CDNA5八值一组，lowering移除重复寄存器基再索引scale，避免合法CDNA4被拒绝及重复寄存器取错scale；FP4累加器不再冗余置零，但每字节仍须转换后才读。buffer atomic RMW/CAS新增显式descriptor与soffset，复用其bounds，指针旧调用不变；高层descriptor atomic API属后续工作。另移除MoveUpPrologueLoads旧pass，使前序/循环load发射顺序交给pipeliner，减少issue-history不一致迫使保守vmcnt的情形；PR没给统一LLM实测，机制理由不等同已测吞吐收益。

**建议：** 建议编译器/内核维护者深入阅读，验收scale重复映射与descriptor边界。 [#12691](https://github.com/triton-lang/triton/pull/12691) · [#12964](https://github.com/triton-lang/triton/pull/12964) · [#12857](https://github.com/triton-lang/triton/pull/12857)

### FlyDSL浮点转换按芯片降低，fast-math边界会改变NaN与双舍入

**项目／合入日期／版本：** ROCm/FlyDSL#1533 `478e1abe6b`（11-27 17:15，UTC+8）。

ROCDL转换pass按gpu.module target处理舍入与FP8/BF8，避免LLVM忽略请求舍入或翻译崩溃；支持的FNUZ/OCP用芯片packed转换，不支持格式在源位置报编译错。gfx942 FP32→BF16、FNUZ全2^32输入及五舍入模式等有作者硬件对照，gfx950/gfx12本地只编译。没有fast-math时部分NaN规范化会改变payload；nnan/fast_fp_math可省NaN检查，少数NaN位型可变0/Inf，afn/unsafe下FP64经FP32双舍入可差1ulp。展示的指令数减少不是模型吞吐测量，直接调用ROCDL内建的另一gfx12转换问题仍未覆盖。

**建议：** 建议升级时审查fast-math契约，把数值语义与可编译性变化纳入回归。 [#1533](https://github.com/ROCm/FlyDSL/pull/1533)

### ATOM的DP贪心LM head只交换max/id，收益不能外推到采样或投机

**项目／合入日期／版本：** ROCm/ATOM#3353 `4cc9fbe71e`（11-27 09:35，UTC+8）。

显式ATOM_DP_LM_HEAD_MODE=argmax把每vocab分片的最大值/最低tie-ID交换，避免传全词表logits并绕过采样。启动仅允许一个DP-sharded head、纯DP且无投机，入站拒绝非零温度、logprobs、结构化输出或required channel，不是修改默认mode。MI455X DeepSeekR1、DP4/TP1、每rank1536行head图从2.567→1.236ms；约占58ms整步2.3%只是trace推算，非隔离MLPerf端到端A/B。DeepSeekV4.1服务1K/1K/c512单轮+1.7%吞吐，all2all基线另一次在波动内。

**建议：** 建议纯贪心离线负载考虑复现，保持tie规则、入站拒绝与无投机约束。 [#3353](https://github.com/ROCm/ATOM/pull/3353)

### ATOM让只有等待的DP组保持idle，prefill角色默认还受delayer限制

**项目／合入日期／版本：** ROCm/ATOM#3777 `63d3ba61d9`（11-27 09:12，UTC+8）。

将“还有未结束请求”投票细分为“本次schedule确实可推进”：tier lookup/load、P/D接纳及延后free等待不再强迫全组跑dummy forward；idle仍完成PAGE checkpoint、推进KV，shutdown恢复普通投票排空隐藏工作。ATOM_DP_IDLE_WHEN_BLOCKED在P/D producer角色默认请求启用，但DP默认prefill delayer仍会禁用，需ATOM_ENABLE_PREFILL_DELAYER=0才真正生效；DP1、PP及其他角色默认不同。最早4P4D c1消去约110ms dummy；后续1P1D显式开关同实现c1 TTFT均值−2.8%、c256中性，最后角色默认/idle PAGE补丁未作为新性能对照。

**建议：** 建议低负载P/D复现，先核实际启用条件及等待期间进展，避免只抄“默认开启”而无收益。 [#3777](https://github.com/ROCm/ATOM/pull/3777)

### ATOM按PCIe拓扑选Mooncake NIC，跨节点旧新对照不能隔离吞吐贡献

**项目／合入日期／版本：** ROCm/ATOM#3884 `7bcaa56785`（11-27 11:29，UTC+8）。

P/D原按HIP序号选rdma同号，但KFD顺序并非PCI顺序；现在从GPU PCI地址找同switch ACTIVE HCA，producer/consumer、advertised rail与Store requester一致，找不到唯一合法NIC直接拒绝启动并给override。同host bridge的旁支HCA或更接近另一GPU的HCA不再猜选。MI355X一小时MiniMax3.1 1P2D/PP4/c240实际每GPU选到本地NIC，零错误；发送CPU/GB及KV传输p50优于先前不同node pair且带多补丁的运行，因此只能支持拓扑选择正确和现场观察，不能把差值归于本PR。未做模型准确率评估；mlx5候选名额外扫描仍有限制。

**建议：** 建议核查GPU/NIC绑定并采用明确拒绝/覆盖策略，在同节点固定patch栈做性能对照。 [#3884](https://github.com/ROCm/ATOM/pull/3884)

### MLXcel的Raon优化同时保留了不值得采用的融合与批投机负结果

**项目／合入日期／版本：** lablup/mlxcel#3461 `521ba0f717`（11-27 09:28，UTC+8）。

M3 Ultra上单行mHC selector改row-major产品、编译mHC小图，使Raon mini decode33.5→54.2tok/s；前缀快照包含RDA FP32 state/conv/offset及MLA latent，2132词元重复turn TTFT1.00→0.04秒，另有缺层/effort等拒绝条件。ModelOpt NVFP4原规范triplet改device reinterpret，加载63–70→2.2秒，驻留与decode不变。数值求和顺序改变会在near-tie改变旧版文本，cache miss多一次历史边界分段也可能改变字节；不是所有路径逐位等价。RDA投影融合仅+0.6–0.8%且不逐位一致而撤回；B>1 DFlash2推算/实测开销使其继续关闭，默认草稿宽从5降3，现有原生9组平均1.18倍不代表所有任务。

**建议：** 值得深入阅读性能归因与负结果，采用时锁checkpoint和cache/采样设置；CUDA/ROCm未运行。 [#3461](https://github.com/lablup/mlxcel/pull/3461)

### MLX输入INT8量化是显式精度选项，小batch会只增加成本

**项目／合入日期／版本：** ml-explore/mlx#4802 `c91080830a`（11-27 12:33，UTC+8）。

Metal affine quantized_matmul新增quantize_input=False默认参数，显式True将激活按group量化为有符号INT8再用NAX整数MMA。作者BF16、group64、四矩阵形状的M≥64测量，M5 Max在M1024时4-bit/6-bit/8-bit权重几何收益1.23/1.06/1.20倍，M64 8-bit反而0.94倍；M<32不走INT8 MMA，只额外付量化开销。随机4096方阵M1024时8-bit权重相对FP32参考RMS误差0.76→0.95%，并非不损失精度，未有LLM能力/吞吐。性能来自前版QQMM、M≥64核与路由相同的声明，需最终接口复核。

**建议：** 建议大prefill矩阵考虑复现，不默认用于单词元decode；同时测能力和激活量化成本。 [#4802](https://github.com/ml-explore/mlx/pull/4802)

### Primus用实训校正projection，预测误差不是训练本身提速

**项目／合入日期／版本：** AMD-AGI/Primus#1446 `30dd51aa6d`（11-27 09:14，UTC+8）。

benchmark projection补DDP/优化器/all-gather与FSDP2在位成本，TorchTitan按完整checkpoint/compile block计时，去掉重算双记及backward通信重复计费；hybrid Mamba/GDN/KDA用完整重复层模式而不是1层上限裁成零层。MI325X、nightly镜像、mock数据30步稳态tokens/s/GPU，作者30个单节点Megatron点平均绝对误差9.8→3.1%、9个TorchTitan点27.5→2.1%；25个两/四节点点2.3%，但全局batch随节点增长、无损RoCE TC106。训练路由强制平衡等配方改动也为校准前提，不能将此当真实路由跨拓扑预测保证，更不能把误差下降写成训练加速。

**建议：** 建议研究系统容量/吞吐投影时深入阅读，按目标通信与路由重新校准，并区分测量模式与纯simulate。 [#1446](https://github.com/AMD-AGI/Primus/pull/1446)

### Trainium3视频生成报告：主机、图大小和并行粒度共同决定可用性

**机构／发布日期／版本：** Hugging Face，2026-11-27；官方博客源码 `22c0b0e901b124640dde7d1a76b086cc3a11210c`。

LTX-2.5实录用TP4×SP8把transformer分到32 NeuronCores，rank0运行主机pipeline，另外解决线程争用、参数打包、非连续拷贝、attention长度对齐和FP32 GELU等。10秒片从4核/960×544/320秒，变为32核/2048×1024/22–23秒；资源、分辨率与两阶段流程同时变化，14倍不是同资源同形状对照。参数打包示例每核22.96→10.2GiB，CPU文本重复编码消除是重要因素。约3分钟加载、首片80秒，整图NaN等三问题未解释，上游parallel配置仍在推进；没有视频质量标准评测。

**建议：** 值得深入阅读设备适配中的主机与图规模瓶颈；按公开代码考虑复现，不能据此比较Trainium与GPU成本或宣称上游已交付全部能力。 [官方报告](https://huggingface.co/blog/video-trainium3) · [固定源码](https://github.com/huggingface/blog/blob/22c0b0e901b124640dde7d1a76b086cc3a11210c/video-trainium3.md)

<a id="topic-7"></a>
## 评估与可靠性

<!-- item:2611.33307|评估与可靠性 -->
### 多语言决策模型：英文置信阈值不能直接迁移，机器翻译评测会缩小差距（补录）

**机构／日期／版本**：NUS、NTU、VinUniversity等；2026-11-24，2611.33307v1。[论文 §3–5、Table 1–3](https://arxiv.org/html/2611.33307v1)。

25个开放决策模型、17任务、14语言的平行题，非英语准确率平均降3.5个百分点、置信仅降1.8；按英文95%准确率校准的门槛在泰米尔语约剩90%。同题人工／LLM翻译的23个基准—语言组合，后者全部更易答对，但只有12组差异达到文中配对显著性。81次控制微调显示基座和语言影响很大，几千目标语言题可缩小差距而未完全消除；不保证所有线上分布。**建议优先深入阅读并复现逐语言选择性风险曲线**，路由和护栏应在目标语言校准，避免只用LLM翻译的英文题验证阈值。

<!-- item:2611.34934|评估与可靠性 -->
### 监控器评测：攻击策略选择会改变漏报估计，仍缺部署合法性论证（补录）

**机构／日期／版本**：Charlie Griffin等作者团队；2026-11-25，2611.34934v1。[论文 §1–3、Appendix B](https://arxiv.org/html/2611.34934v1)。

固定Qwen3.8-27B代理、30个控制任务和一个退役生产GPT-5.4监控器，仅换攻击生成方法／优化量，RL不同阶段漏报率0.5%–20.8%，各方法最佳观测点6.4%–23.0%；攻击只针对代理监控器开发，GPT-5.4作为留出目标。它表明简单攻击可能给出虚假安心，但更高漏报也不自动等于真实部署风险：编辑轨迹可能泄露“侧任务”等离分布伪影。**建议深入阅读并采用多攻击策略敏感性检查**；要另行解释代理轨迹为何代表或保守覆盖部署危险行为，不把23%当线上漏报率。

<!-- item:2611.34825|评估与可靠性 -->
### RAP：先审计模型组合是否可路由，再解释路由器收益（补录）

**机构／日期／版本**：Sorbonne／CNRS、Jožef Stefan Institute等；2026-11-25，2611.34825v1。[论文 §3–9、Table 1–2、Appendix B](https://arxiv.org/html/2611.34825v1)。

在四基准×66模型得分矩阵上，RAP先量化单模型最好与理想选择器的余量、并列最好比例、提示到性能的可预测性，再比较路由。IFEval理想余量仅4.5个百分点，中位并列最好达组合的88%；有余量也不保证可学习。论文主动降级／撤回部分PCA和路由收益；按重叠划分的Nadeau–Bengio敏感性校正，无headline行保留显著性。**建议深入阅读并考虑采用审计口径**：报告并列处理、组合构造与多重检验，工具是已有完整矩阵的回溯审计，不是低成本预测未测组合；成本仍是参数量代理。

<!-- item:2611.34398|评估与可靠性 -->
### 技能单独成功不保证可组合：软提示接口仍有伙伴依赖（补录）

**机构／日期／版本**：Newcastle、Melbourne、INSAIT等；2026-11-25，2611.34398v1。[论文 §3–6、Table 1与Limitations](https://arxiv.org/html/2611.34398v1)。

冻结语言模型、独立学习软提示技能，再按已知身份／顺序组合：四数字任务基座和三记录编辑模型，单调用平均正确率最高7.6%，分两调用为88.2%–100%；后者增加计算和显式中间结果，并非同成本胜出。替换为同操作且单独≥90%正确的伙伴也能使组合崩溃，修复组合又可能伤害原技能。是合成短链软提示实验，未证明真实Agent规划／文本技能同样失效，内部干预也未定位统一根因。**建议深入阅读并采用组合回归测试**：新增技能需同时验旧技能保留和新伙伴组合，而不只看单项得分。

<!-- item:2611.33184|评估与可靠性 -->
### Agent记忆密度不能可靠预警崩溃：先检查容量与直接任务表现（补录）

**机构／日期／版本**：独立研究者、Vasuki、Piramal Finance；原始提交2026-10-28，2611.33184v1；本轮11-26公告目录发现后补录。[论文 §2–4、Table 1](https://arxiv.org/html/2611.33184v1)。

13种记忆方法、805快照上，全局平均语义相似度的预警领先中位数为零；局部拥挤对事后峰值定位有时领先，却不稳定优于条目数量。分类器不改参数迁移到最长5000题的PropWorld：稳定限容存储56.9%检查点误报，任务成功已归零后只识别18%；该长跑唯一崩溃是无限整库提示超出输入容量。**建议深入阅读并保留负结果**：几何统计可描述当前存储，但不能代替任务回归、词元增长和容量门禁；结果受这些环境／记忆实现限制，不代表所有语义指标都无用。

<!-- item:2611.34331|评估与可靠性 -->
### Puzzling-Bench：人写谜题有价值，但当前模型排名受未生效长度上限影响（补录）

**机构／日期／版本**：UMass Amherst、USC、Adobe等；2026-11-25，2611.34331v1。[论文 §3–5、§8](https://arxiv.org/html/2611.34331v1)。

从Puzzling Stack Exchange筛得23777题与被采纳答案，保留提示、日期和图片标记，支持滚动新题和去提示对照。数据可用，排行榜却需谨慎：Pilot-255仅255题，后截点子集56题；虽请求4096输出词元，CLI封装未执行限制，各模型7%–63%回答超限且未截断。不同封装还混杂推理力度，题目参考答案部分由LLM抽取，“被采纳”也不是形式证明。**建议深入阅读数据与失败记录，暂不直接采用名次结论**；复现应先修预算执行，并隔离提示、旧题变体和翻译／裁判误差。

<!-- item:2611.34232|评估与可靠性 -->
### OS-Omni：跨系统长期任务按最终持久状态评分，首波跨平台仅12题（补录）

**机构／日期／版本**：OS-Copilot作者团队；2026-11-25，2611.34232v1，benchmark v1。[论文 §2–4、Table 1、Appendix D](https://arxiv.org/html/2611.34232v1)。

首波300题包含六平台各48题和12个桌面—Android流程，提供平台重置、逐步轨迹及最终状态可执行裁判，能识别重复转账／漏订项目等“界面已推进、任务未完成”。Opus5.5宏平均68.5%，GPT-6-Astra66.1%；按300题微平均则66.7%与67.0%，小跨平台集合被宏平均放大。所有跨平台通过桌面上的scrcpy单屏操作，不能外推任意设备编排；步数／超时与模型配置也需固定。**建议深入阅读并考虑采用状态验证设计**，复现排名前先核每平台任务分布、控制面和完整预算。

<!-- item:2611.34852|评估与可靠性 -->
### DRBench Pro：把深研报告覆盖拆为知识树广度与深度（补录）

**机构／日期／版本**：浙江大学、腾讯；2026-11-25，2611.34852v1。[论文 §3–4、Table 2及Appendix H](https://arxiv.org/html/2611.34852v1)。

从Wikipedia局部实体关系树生成查询与证据rubric，以关系边为单位，自底向上计算研究方向广度、连续知识链深度及其几何平均；144题、24领域、44102项rubric，区分“罗列很多点”与“沿关系展开”。16套系统中，同名Qwen3.8 Max在作者本地研究框架与Native Search的覆盖率47.94%／36.53%，提示框架会显著改变覆盖；这是不同产品／工具预算配置，不是纯模型能力差。树只代表预定知识空间，不能单独保证事实正确、来源可信或时效。**建议深入阅读并考虑作为深研评估的一维**，与事实／引用校验联合使用，避免训练成机械命中rubric。

<!-- item:2611.34813|评估与可靠性 -->
### 视觉推理的代码基线：同输入裁判领先视频模型，计算预算并未对齐（补录）

**机构／日期／版本**：Solve Vision with Code作者团队；2026-11-25，2611.34813v1。[论文 §2–4、Table 1–2](https://arxiv.org/html/2611.34813v1)。

给编码Agent相同首帧、指令和视频输出格式，由其写程序渲染答案，使用未修改的VBVR-Pro裁判。500个合成几何／规则任务，Codex得0.923，所引用最强视频模型榜值0.670；视频模型未由作者重跑。对照揭示“必须原生视频推理”的结论需要代码基线，但Agent可在CPU沙箱迭代最多30分钟，视频模型只生成一次，且平色几何场景天然适合程序重建，不能当同成本或真实视频能力胜出。**建议深入阅读并采用符号／程序基线**；后续复现应固定工具、总时延、采样预算和真实场景条件。

<a id="coverage"></a>
## 覆盖、补录与未完成区间

- **归档与去重：**本轮读取持久化state、逐轮稳定事件和本地50篇历史日报。GitHub main固定基线`706591c646be0aca00d753f4ea233cdf4887acb8`；51个目录文件（50日报与README）的Git blob逐项与本地一致，远端索引已读；没有待补传文件。论文ID＋版本、模型快照、PR／合入SHA分别去重；旧稿上热榜不自动视为新研究。
- **模型／厂商／数据：**157个官网、研究和技术文档入口，直连95个HTTP200，其中动态空壳与少量内容不足页另记未完成；并非95个来源均完成全文审查。187个HF组织、374个模型／数据集目录经重试均取回，当前376页；新旧目录差分175个候选选择性审读。全站创建目录得到4,117个模型和598个数据集候选，模型目录越过主窗起点，数据集尾页缺可用时间字段，不能关闭其创建窗口；创建枚举也不覆盖全站修订。补查中英文主题检索及ai-sage、BAAI、腾讯与NVIDIA固定工件；没有条目不意味着某厂商没有发布。
- **论文／评估：**arXiv十分类目录完整取回，但均仍返回11月26日公告；精确回溯Atom为603／603条，最新更新时间仍11月25日18:59:38Z，不能据此断言27日没有新版本。本轮1,338个唯一目录候选、734个历史未见宽关键词候选，711个仍待细审；26篇入选均有固定版本完整HTML（25个v1与1个v2）。OpenReview两入口403，ACL／PMLR等历史会议与长尾缺口继续保留。
- **开源工程：**64仓主窗默认分支目录全部取回，507条commit；Megatron-LM、vLLM、verl、SGLang、slime、TorchTitan六核心回溯合入目录745条、全分页取回，20仓release目录。95个PR正文与文件差分、3个PyTorch默认主干commit定向审读；大文件列表补页与上游导入状态单独记录，981个当前／回溯候选继续待审。合入不等于正式发版，开放PR与作者fork明确标状态。
- **硬件与软件栈：**31仓、39个commit目录页、1,603条回溯目录记录（主窗296），31个release目录；117个PR元数据及完整文件列表取回，仅其中有实质技术证据者展开。NVIDIA、AMD、Intel／Gaudi、昇腾、TPU／XLA／JAX、Neuron等71个官网入口直连55个完整HTTP200；1,405个未审或未分类目录行留查。CANN正式包／组件映射、libtpu行为、MLCommons原始日志及历史官网失败不因本轮目录成功关闭。
- **补录与更正：**论文原始日期与当前发现日分开，REI／Semantic Density按原文10月28日记补录，Shifter使用v2并分开质量和计时精度。GigaChat报告和26日权重、Parse开放数据、Innata旧报告均明确补录／本次增量。Transformer Engine #3710按最终源码保留连续维padding限制；SGLang #52437补齐此前fenced writes的idle／drain具体缺口，本次只关闭这项已验证问题，不关闭整个HiCache领域的未审区间。跨线审阅补齐GigaChat数据重复采样和GPU步口径、Parse牙科奖励边界及Innata新路径复验范围。
- **续查水位：**下轮从`2026-11-28T00:59:46Z`继续目录发现，再回溯72小时，同时继承全部未完成区间；全局完整语义覆盖水位仍为空。当前主窗论文版本可见性、未审长尾及失败入口均留有台账，不把“未找到”写成“没有发布”。本地保存、远端同步与失败状态分开记录。

本轮取证和状态位于`.monitor/research-2026-11-28/{models,papers,engineering,hardware}/`、`.monitor/run-2026-11-28.json`与`.monitor/state.json`；上传回读结果另存`.monitor/archive-receipt-2026-11-28.json`。仅维护llm-monitor及tracker/llm-training-daily，未改写llm-knowledge Wiki。
