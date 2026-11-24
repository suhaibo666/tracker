# LLM 技术动态日报 · 2026-11-24

> 时区 **Asia/Shanghai（UTC+8）**。增量目录扫描窗口 **(2026-11-23 10:14:20, 2026-11-24 09:01:25]**，回溯72小时至 **2026-11-20 10:14:20**，继承此前未完成区间。实际生成 **2026-11-24T09:25:42+08:00**。另对 DeepSpeed ReFlow 单项补充取证至09:18:36：该PR于09:02:19合入，比目录截止晚54秒，明确列为截止后补充，不提升其他来源的覆盖水位。

文中 LLM 为大语言模型，MoE 为混合专家，KV 为键值缓存，SFT 为监督微调，RL 为强化学习，RLVR 为可验证奖励强化学习，GRPO 为组相对策略优化，OPD 为在线策略蒸馏；TP/PP/CP/DP/EP分别指张量/流水线/上下文/数据/专家并行。TTFT 为首token时延，TPOT 为每输出token时延，ITL 为流式输出间隔，HBM 为高带宽显存，MLA 为多头潜在注意力，MTP 为多token预测，SDPA 为缩放点积注意力，GDN 为门控DeltaNet，QSA 为Qwen稀疏注意力，SSM 为状态空间模型，RoPE 为旋转位置编码；BF/FP后的数字表示浮点位宽。下面的实验均为来源作者或项目方报告，本轮未复训、未进行设备压测或付费API测试。

## 本次重要变化

- **循环架构进入旗舰发布，但公开消融仍需分清规模。**GLM-5.4复用28层执行两次，并以门控更新、跨轮稀疏KV与索引复用控制额外开销；30B同参数实验有效，750B旗舰仍缺完整可复现量化对照。Ling-3.2-air、Xing4.1和WePM等开放工件提供了新的研究入口，量化、MTP、许可证与部署条件分别列出。[GLM官方说明](https://z.ai/blog/glm-5.4)
- **预训练与后训练的新结果，更需要强基线和成本口径。**QueryLM把搜索引入预训练，Model Carving在训练中重新分配层与token预算；OPD覆盖不足、RL记住链内自造事实、Gumbel替换的残余偏差，则提醒不能仅看最终分数或理论边缘分布。[QueryLM](https://arxiv.org/html/2611.26809v1) · [Model Carving](https://arxiv.org/html/2611.27962v1) · [RL事实研究](https://arxiv.org/html/2611.26863v1)
- **训练系统出现实质的CPU卸载与通信重叠增量。**ReFlow拆分低精度参数就绪和持久优化器状态更新；FsdpMuon跨组重叠传输与正交化；TorchTitan让GPU融合Adam直接访问锁页主存。三者硬件、计时阶段和实跑/fake进程组条件不同，不能按最大倍数排序。[ReFlow，截止后补充](https://github.com/deepspeedai/DeepSpeed/pull/8719) · [FsdpMuon](https://github.com/NVIDIA/Megatron-LM/pull/9681) · [TorchTitan](https://github.com/pytorch/torchtitan/pull/5881)
- **长上下文优化继续从“减少attention”转向检索、通信与状态维护。**LongCat LRSA把分页检索和增量索引一起设计；其35.60倍是选键+attention相对dense局部阶段，1M端到端对DSA是1.865–1.984倍。vLLM ShardQ和ATOM则在适用prefill上改搬KV，避免query汇聚与输出归并。[LRSA](https://arxiv.org/html/2611.27233v1) · [vLLM同机对照](https://github.com/vllm-project/vllm/pull/66027) · [ATOM](https://github.com/ROCm/ATOM/pull/3547)
- **采用判断应优先看负面结果和正确性边界。**Ling法律评测经幻觉过滤后全通过率0.8%，VideoGym组合/跨任务泛化接近零；MemFabric HiSparse在已测组合更慢，投机在高并发时优势也会缩小。MoRI晚到写fence、混合缓存完整状态、replay专家游标与TPU容量同步等修复，实际影响可能高于又一个局部kernel峰值。[VideoGym固定卡](https://huggingface.co/OpenMOSS-Team/VideoGym-LTX-2.3/blob/64a410d9f00071eb1ea0e675e760b4b3ad0c817c/README.md) · [MoRI fence](https://github.com/ROCm/mori/pull/848)

[模型与技术报告](#topic-1) · [预训练与架构](#topic-2) · [后训练](#topic-3) · [推理与部署](#topic-4) · [开源工程](#topic-5) · [硬件软件协同](#topic-6) · [评估与可靠性](#topic-7) · [覆盖与续查](#coverage)

<a id="topic-1"></a>
## 模型与技术报告

### 智谱 GLM-5.4：把已有750B基座的28层循环两次，关键证据仍是30B消融

**机构／日期／版本：Z.ai，11月23日官方技术说明。建议深入阅读，等待公开工件后考虑复现。**相较继续堆参数，GLM-5.4把预训练好的GLM-5基座中28层共享权重执行两次，再做持续训练。官方解释直接循环会放大激活范数，因此每次迭代加入独立的逐token写入门；第二轮复用第一轮选中的稀疏键值（KV）与注意力索引，另保留自己的局部滑动窗口KV，使新增缓存随窗口而非完整上下文增长。上述机制来自官方说明，尚无公开实现可核。[官方正文](https://z.ai/blog/glm-5.4)

**证据和限制：**官方同参数30B模型、200K长上下文监督微调（SFT）后、开启思考的对照，循环版总训练计算多约10%，AIME25为19.17→30.42，GPQA-Diamond为47.58→56.26；后者平均输出少11.7%，AIME25仅少2%。不能把该消融当作750B旗舰成绩，也不能把输出缩短当作总计算下降。文中“相当于2T–3T模型”没有旗舰量化对照；200 tokens/s非高峰、70–80高峰缺硬件、负载和长度条件。官方承认会出现很长的思考过程；Coding Plan等可用，通用API写明Coming Soon，权重须安全加固后计划开放，尚无可确认的权重许可证。[官方30B结果图](https://z-ai-bj.oss-cn-beijing.aliyuncs.com/prompts-rich-media-resources/5.4-blog/img_v3_0217n_2d1a655c-5bce-4b3f-9115-2ad7c397ce5g.png)

### Thinking Machines补充可变循环结果：第五遍并不在所有任务上获益

**机构／日期／版本：Thinking Machines；权重11月20日已收录，本次11月23日评测卡`c90921bd5f78`增量。建议深入阅读同权重的深度对照。**新增结果包括GraphWalks BFS 1M上下文、长窗MRCR、SWEBench多模态及PMPP-Hard，而非重发模型。可变循环工件在1–5遍下，GraphWalks BFS 1M为5.7%、39.4%、52.1%、56.2%、57.8%；MRCR八针512K–1M为20.6%、49.7%、72.3%、81.4%、79.9%，第五遍退步。PMPP-Hard的0–10小时解题面积也从第四遍437.2降到第五遍424.0，不能当百分比或只看“更多循环更好”。[固定研究卡§4–5](https://huggingface.co/thinkingmachines/TML-d24-varloop-research/blob/c90921bd5f78df6c8168173b9f8a3a850320c594/README.md)

这是同权重推理计算的可复查新证据；每额外一遍约增加1B有效激活参数／token，未给各任务实测能耗／时延或置信区间。mid-training循环采样1:2:6:2:1，后训练只在3遍完成，不能当各深度分别充分训练的等算力比较，也不能直接对照训练不同的Inkling2Mini。卡同时修订了256K GraphWalks等旧行；本轮保留前后卡片差分，后续引用应锁版本。研究权重Apache2.0，不建议生产使用。

### 蚂蚁 Ling-3.2-air：124B混合多模态权重落地，法律全通过率暴露幻觉约束

**机构／日期／版本：inclusionAI；11月23日；固定`1613b1a4dcc6`。建议按文档和自动化场景考虑复现。**模型共享Ling-3.0-flash-VL的原生多模态架构：124B总参数／5.5B激活，35层Kimi Delta Attention（KDA）与7层门控多头潜在注意力（MLA），512路由专家每token选8。交付配置为42层、262,144上下文、一个多token预测（MTP）层，权重量化配置含FP8动态激活及128×128块尺度；不能因启动参数为BF16就称全模型BF16。[固定模型卡](https://huggingface.co/inclusionAI/Ling-3.2-air/blob/1613b1a4dcc6aa0d8f346606628cd9a92061a525/README.md) · [固定配置](https://huggingface.co/inclusionAI/Ling-3.2-air/blob/1613b1a4dcc6aa0d8f346606628cd9a92061a525/config.json)

卡片报告相对前代AutomationBench-AA 15.7%→61%、GDP.pdf全通过8.4%→16%，但未给完整同预算细节；法律Harvey LAB-AA v1.1的单项判据通过87.7%，**全通过经material hallucination检查后从8.3%降至0.8%**，两种指标须并读。训练数据、算力与消融未充分披露，图表用了本轮无法公开审查的内部图片地址。推荐专用SGLang镜像，256K配4×141GB级GPU或4卡Blackwell，80GB卡建议TP8；另有厂商vLLM fork。卡标MIT，根目录LICENSE请求404，采用前应补核许可文件、镜像与任务成功率；本轮未运行模型。

### 中国电信 Xing4.1-Preview：国产训练报告对应285B开放工件，量化版本不能混读

**机构／日期／版本：中国电信星辰；BF16仓库11月22日创建、11月23日更新并新增FP8／NVFP4；报告2611.22115v1提交于11月17日、已在11月19日日报收录；本次增量是权重与量化工件。建议深入阅读并考虑部署复现。**新权重采用45层、285B总参数／约17B激活、MLA+DeepSeek Sparse Attention（DSA），256路由专家选8，配置1,048,576上下文。报告交代在MindSpore HyperParallel上以8,192张Ascend 910C训练285B，并描述超节点内分片／节点间复制的混合分片数据并行（HSDP）、矩阵拥有者执行Muon正交化、细粒度重算与异步partial rollout。公开权重让这些系统研究有了可定位的模型工件；报告中的系列级mHC、IndexShare和投机解码实验不能自动归给每个发布版本。[固定BF16卡](https://huggingface.co/XingChen-AGI/Xing4.1-Preview/blob/35a6cecba33d2e0f512f227447e4b46edbda7f94/README.md) · [报告§2–5](https://arxiv.org/html/2611.22115v1)

§5.2仅将CPU绑定与隔离组合报告为512卡+1–2%、8192卡+5–10%，不能把卡片合并表述的15–30%归因给单项措施。评测卡的Terminal-Bench 4.0跳过GPU沙箱任务，HLE仅文本，其它模型取各自公开报告，不能视为统一复测。BF16／FP8的`num_nextn_predict_layers=0`，NVFP4为1，需核权重和运行栈再启MTP；NVFP4亦非所有模块4bit。三工件卡标Apache2.0，量化质量／吞吐未给充分同条件对照；1M配置不等于8×80GB即能跑满1M。[固定BF16配置](https://huggingface.co/XingChen-AGI/Xing4.1-Preview/blob/35a6cecba33d2e0f512f227447e4b46edbda7f94/config.json) · [NVFP4配置](https://huggingface.co/XingChen-AGI/Xing4.1-Preview-NVFP4/blob/6b3478dc99c561848b024fa50e5a6f9bf0baa823/config.json)

### 腾讯 WePM：用一个2B／4B骨干统一生成、检索与结构化判断

**机构／日期／版本：微信AI；11月23日开放WePM 1.0；论文2611.27365v1提交11月20日，补录。代码固定`8cfe184934d6`，4B权重`d57517468ff5`。建议深入阅读并考虑复现多任务冲突消融。**多模型常驻会重复视觉编码器和语言骨干；WePM按固定视觉token预算分配全局／区域／细节信息，借两类冻结教师做列表级与生成蒸馏，再按固定任务权重、曝光计划依次前后向、最后一起更新。检索与判断分别使用独立学习槽及读出头，生成保留自回归路径。[官方发布及机制](https://github.com/Tencent/WePocketModel/blob/8cfe184934d6d26fb4ed63dd0adc9124a55ccc55/README.md) · [论文§3、§4.4](https://arxiv.org/html/2611.27365v1)

论文共享表示／生成切片中，固定等权方案Joint49.78对FAMO46.23；这是特定数据与曝光计划的结果，不能据此否定通用自适应加权。HCD（异构能力密度）比较的8GB对21–43GB**只计BF16权重**，不含KV、激活、碎片及运行时开销；README的4B实际CUDA建议16GiB，CPU FP32建议24GiB。论文§6承认相对Qwen骨干的Agent能力有所退化，尚无专门Agent后训练。已提供Transformers/GGUF与2B/4B权重，4B独立许可Apache2.0；宜按目标端侧设备测三能力共同常驻与延迟，不据权重内存推断手机运行表现。[固定4B卡](https://huggingface.co/tencent/WePocketModel1.0-4B/blob/d57517468ff546df8cb9c6da5c76319810d3523b/README.md)

### 腾讯 WeMM-Ecom：电商视觉理解开放0.8B至35B家族，多格式需分别验收

**机构／日期／版本：微信视觉；11月23日权重实录；35B固定`1d8bc85c58d7`。建议有商品理解需求时定向复现。**基于Qwen3.5的文本／图像／视频理解家族提供0.8B、2B、4B、9B与35B-A3B，并交付FP8、GPTQ、GGUF等格式；35B配置262,144上下文、支持显式选择思考与直答。官方以28个电商基准和64个通用基准汇总，35B电商平均比Qwen3.5-35B-A3B高3.92点、通用平均仅高0.17点，价值主要在定向商品任务。[固定卡与评估设置](https://huggingface.co/tencent/WeMM-Ecom-35B-A3B/blob/1d8bc85c58d7f3a6469bb455054375d33f7a2b18/README.md)

完整数据配方、消融和量化质量尚不足以独立重做；详细成绩主要以图片交付，不把“胜过数百B模型”当统一能力排序。评测输出预算65,536，而quickstart仅4,096；部署示例32,768上下文也不同于最大窗。卡要求vLLM≥0.31.0或SGLang≥0.5.15；FP8/GPTQ/GGUF文件存在不表示精度、硬件和吞吐等价。Apache2.0并保留第三方组件条款，本轮未运行评测。[GPTQ工件](https://huggingface.co/tencent/WeMM-Ecom-35B-A3B-GPTQ/tree/284a6cdb1562cda610d3b86af512bdb1e3c757d4)

### MemTensor MemT-1.0-Pro：Nemotron骨干的持久记忆权重，按11月20日补录

**机构／版本：MemTensor；主要权重上传11月20日，最新卡11月21日`046a1626642f`；本轮发现，非11月23日新上传。建议阅读并考虑复现跨会话记忆。**Pro以Nemotron-3-Ultra-550B-A55B为骨干，卡称554B总参数／59B激活、4.23B可训练记忆参数、13M动态记忆状态。Native Memory Block在注意力旁读取局部动态矩阵及可训练超记忆，轮末／段末通过GDN-2写入；session可保存状态并在没有原始历史文本的上下文中恢复。[固定模型卡](https://huggingface.co/MemTensor/MemT-1.0-Pro/blob/046a1626642f804033cd05330dced47955acdf58/README.md) · [固定配置](https://huggingface.co/MemTensor/MemT-1.0-Pro/blob/046a1626642f804033cd05330dced47955acdf58/config.json)

“128K历史节省约66倍存储”是固定大小记忆对完整KV的对照，随历史和基座而变，不能当全模型显存或时延减少66倍。卡明确强文本记忆系统在长历史细节检索上仍领先；本轮打开的2610.26825v2报告表2仍主要列Flash/Nano，不能用它代替Pro完整评测证据。当前Pro配置、卡片与实际权重可读，采用须使用自定义代码及匹配Mamba/causal-conv1d依赖；许可OpenMDW-1.1，训练与压测均未独立复现。[已打开报告v2](https://arxiv.org/html/2610.26825v2)

### Salesforce BLIP-Live：流式视频“何时回答”训练与指标，发布卡明确合并后未重评

**机构／日期／版本：Salesforce；论文2611.21553v1为11月16日、权重仓库11月19–20日建立，本轮补录；11月23日增量是报告链接串接。Qwen卡固定`96b112b11302`。建议深入阅读数据与指标。**BLIP-Live将内容正确性、响应时机与保持沉默共同训练，提供780K数据、滑动视觉窗口／上次回答记忆，以及惩罚过量触发的PAUC-F。论文表6在ProactiveVideoQA、GPT-4.1裁判、ω=0.5下，Qwen3.5-A3B配作者harness均分42.94，BLIP-Live-Qwen49.08；与MMDuet2的44%相对提升包含不同模型与harness，不能称纯训练增益。[论文§3–6、Table6](https://arxiv.org/html/2611.21553v1) · [固定Qwen卡](https://huggingface.co/Salesforce/BLIP-Live/blob/96b112b1130251d1cdd4d54b7969a0cb30ebab25/README.md)

发布Qwen模型是rank64 LoRA合并的完整工件，1FPS／32帧窗口／每帧144视觉token；普通chat调用不复现流式协议。卡上的四分项成绩来自**合并前适配器**，仅做张量回读、加载和最小CPU前向，未做合并后生成或基准验收。模型修改及发布材料现标CC BY-NC4.0，上游Qwen保留Apache2.0；Gemma版本与数据须分别核许可。对实时助手有价值，采用前应复现触发精确率、延迟与合并后质量。

### SenseNova RoboRSI-27B：正式版联合更新模型与Harness，真实机器人证据仍小样本

**机构／日期／版本：商汤；11月23日正式权重`48be9e0ae25b`，不同于10月已收录的Preview。建议考虑复现具身Agent训练闭环。**正式版将交互反馈用于Harness修订，再由教师采轨迹、筛选、微调模型；同checkpoint可调用视觉语言动作（VLA）策略、审查动作或以代码控制末端位姿。作者RoboDojo均分34.7对Preview10.0，所测七模型用通用Harness且不加任务技能，但发布未提供可立即重做该RoboDojo分数的脚本，仍写明提交榜单后开放。[固定卡](https://huggingface.co/sensenova/SenseNova-RoboRSI-27B/blob/48be9e0ae25b000c3bb85b4d1a701da93eeb642d/README.md)

374个动作的VLA2Agent消融中，50mm内末端命中24.9%→49.2%，中位位置误差119.7→51.6mm，仅对可解析坐标求中位数；双臂需都命中才计成功。无真实机器人数据的后训练版本在Piper取放6/10、Piper X堆叠9/10，Preview分别2/10和3/10，样本不足以推断稳定泛化，且两平台执行模式不同。权重标Apache2.0；应把模型更新、Harness更新与控制接口适配分开评估。

### 千问音频新增Live与Captioner：任务委派与异步全信息标注的接口边界已公开

**机构／日期／版本：千问AI平台；11月23日模型更新。建议有实时语音或音频数据管线时考虑小规模验证。**`qwen-audio-3.1-live`在后台任务处理期间继续前台对话；Client模式由应用执行并回传，Responses模式由平台调用指定后台模型。需区分URL中的服务ID与`session.model`：当前完整示例的前台模型是`qwen-audio-4.0-realtime-preview`，全双工还需`turn_detection.type=duplex`，不能把服务入口名称直接等同前台模型版本。`delegation_id`关联任务进度、追问和结果；该接口可将长推理移出语音主循环，但本轮未实测可感知时延。[更新记录](https://platform.qianwenai.com/docs/changelog/models) · [Live完整协议](https://platform.qianwenai.com/docs/developer-guides/speech/qwen-audio-live)

Captioner支持带时间信息的语音、说话人、环境声、音乐描述及分离，走异步文件URL接口，每次≤2GB、≤5分钟、≥8kHz，7语种；不是实时流式标注。模型页虽列音频输出，使用指南主要展示转写URL中的标注结果，分离音轨交付细节仍应验收。两者未公开训练架构、数据和可比基准；Live空闲／静音也计会话时长，后台模型另计Token，因此任务等待成本须纳入设计。[Captioner使用限制](https://platform.qianwenai.com/docs/developer-guides/speech/audio-tagging)

<a id="topic-2"></a>
## 预训练与架构

以下论文均固定到 arXiv v1，提交日期为 11 月 11–20 日（UTC）；本轮补录，不当作 11 月 24 日首发。下述结果均为作者报告，本轮核对正文中的机制、配置、结果与限制，未独立复训，也未将论文中的实现描述视作已合入主流框架的能力。

### QueryLM：把搜索调用与证据使用提前到预训练（2611.26809v1，补录）

**机构／日期／版本**：EPFL、University of Toronto；2026-11-19 UTC 提交，v1。

传统配方通常在后训练才引入搜索。QueryLM 先用小标注模型为普通文档插入搜索位置和查询，经本地 BM25 索引检索回复，再对原文与查询计算下一个词元损失，**检索回复本身屏蔽损失**。Wikipedia 的 360M 和 DCLM-Edu 的 400M／1.4B／3.3B 从头训练对照表明，搜索预训练的知识问答优势在双方接受搜索 SFT 后仍存在；同计算比较计入了无损失回复词元的前反向开销。限制是搜索数据制备与推理外部调用仍需额外资源，训练 FLOPs 匹配不等于端到端成本匹配；更“理想”的回复提高上下文依赖，却未必提高换检索源后的问答效果，小模型通用能力也有下降。建议深入阅读并考虑复现等总成本、等原文暴露量的对照，适合以小模型承担知识密集搜索任务的路线。[正文 §3–6、Tables 1–2、Appendix A.2/A.5](https://arxiv.org/html/2611.26809v1)

### Model Carving：训练中删去子层，把省下的计算换成更多训练词元（2611.27962v1，补录）

**机构／日期／版本**：Harvard、NYU；2026-11-20 UTC 提交，v1。

对注意力和 MLP 子层增加标量门，依据在线学习曲线估计“保留这个子层的计算可换来多少额外数据收益”，形成删层压力；门归零后真正停止执行该子层。在 FineWeb-Edu、2K 序列、Muon＋AdamW 的固定 FLOP 预算下，标准 Transformer 测到 2.7B、循环模型测到 1.3B，比较完整模型、相近推理成本的浅模型、剪枝及蒸馏。作者报告推理 FLOPs 少 8–34%；单 H100、4K 上下文，**同批量**解码最多 1.45 倍，各模型取各自最大批量时最多 1.61 倍。后者包含容量收益，不能混写成同负载加速。主要删除发生在前 20% 训练，门初始化与压力预测存在启发式，尚未验证更大规模和不同训练配方。建议复现；价值在训练计算与最终拓扑联合分配，而非训练后免费剪枝。[正文 §2–5、Table 1、Fig. 4、§8](https://arxiv.org/html/2611.27962v1)

### cmHC：双随机约束保证稳定，不自动保证多流有效混合（2611.27193v1，补录）

**机构／日期／版本**：中国电信星辰团队、港科大（广州）等；2026-11-20 UTC 提交，v1。

论文区分残差流混合过弱和过强：前者近似独立流，后者抹平差异；cmHC 通过向多面体内部收缩、主要施加于浅层，保持精确双随机性并控制混合程度。小模型有三次重复；6B MoE 训练到 71.5K 步，300B MoE 仅展示到 2K 步的损失／梯度趋势，不能据此断言完整大模型训练收益。另在 Ascend 910C、CANN 9.1.0、B=1/S=4096/四流下融合 Pre/Post 的前反向流水，hidden 2048／3584／7168 相对 mHC 总算子时延下降 44.8%／31.0%／24.1%。**该计时排除了 Transformer 子层，并非整模训练加速**。建议研究多流残差的团队阅读；NPU 团队可评估融合实现，但应单独验证端到端占比与数值行为。[正文 §3–4、Tables 1–3、Appendix F](https://arxiv.org/html/2611.27193v1)

### CoFrGeNet-FNG：非扩张门控与连分式通道混合的可训练实现（2611.26835v1，补录）

**机构／日期／版本**：IBM Research；2026-11-19 UTC 提交，v1；文中标注 NeurIPS 2026 接收。

将 FFN 的扩张改成非扩张门控，并用对数／指数重写连分式中的除法，以免深层训练依赖旧方案的逐层增量训练。GPT-2、Llama-3、Mamba-2 与四专家 Llama-MoE 的实验比较原 FFN、旧 CoFrGeNet 和无连分式的非扩张门控；MoE 版本近半参数且验证损失 2.92 对原模型 3.01。收益伴随结构和参数量改变，不能归结为单算子提速；论文也明确缺少内存优化的 Triton 实现，更大 MoE 未测。建议继续观察，或以小规模预训练检验其数值稳定性和真实硬件吞吐，暂不直接替换成熟 FFN。[正文 §3–5、Tables 2–8](https://arxiv.org/html/2611.26835v1)

### DistP：本地更新训练的学习率迁移，须区分扩批与加 Worker（2611.27808v1，补录）

**机构／日期／版本**：Cambridge、SOFAIR、ZIB、Flower Labs 等；2026-11-20 UTC 提交，v1。

将宽度／深度参数化迁移规则推广到多 Worker 本地更新：外层平均时，增大每卡本地批量与增加 Worker 数不能套同一学习率缩放；外层 Adam 另需修正，外层带动量时内外学习率可能需要联合调参。823M、17.18B 总训练词元的 Nesterov 对照中，小模型调好再迁移的平均下游分数为 45.18，对直接使用既有 DiLoCo 拟合律的 41.10，约 **+4.08 个百分点／相对 +10%**。这不是对重新充分调优的大模型上限的比较。实验以内层 Adam、同步更新和周期性优化器状态同步为主，异步、陈旧版本和其他内层优化器未验证。建议通信受限预训练团队深入阅读迁移条件，再考虑复现。[正文 §3–4、Table 2、Appendix A/G，PDF](https://arxiv.org/pdf/2611.27808v1)

### 多语词表分配：表征更对齐，跨语迁移仍可能不变（2611.27491v1，补录）

**机构／日期／版本**：University College London；2026-11-20 UTC 提交，v1。

39 个从头训练的 1B 模型固定 64K Unigram 词表、30B 训练词元与架构，比较词表分给 419／20／5／2 种语言的差异。更充分分配给目标语言后，早中层表征对齐、跨语言事实神经元共享上升，**跨语任务迁移却基本没有相应改善**；只测最后一层会漏掉对齐变化。每配置单种子、英语为固定中心；文字系统与语言身份仍有混淆，因此“跨文字系统障碍”是有约束的描述。词元固定也意味着实际文本暴露量变化，作者的等质量回归控制属于稳健性检查。建议阅读，避免将对齐指标直接当作迁移收益的代理。[正文 §3–4、Limitations](https://arxiv.org/html/2611.27491v1)

### VIG：多模态预训练损失下降，未必来自更好地读取图像（2611.26917v1，补录）

**机构／日期／版本**：Rice University；2026-11-20 UTC 提交，v1。

以同一文本在真实图像、打乱 patch 和纯色图像下的损失差，分开估计全局视觉统计与布局信息。158M–1.04B 编码器无关模型、最多 40B 词元和 19 个公开检查点显示，语言建模损失相近时对布局的依赖可以不同；冻结 SigLIP 2 带来的损失优势主要体现在原布局完整的图像上。公开大规模无编码器模型已能获得与有编码器模型范围重叠的布局信息，**不能把小规模结果写成无编码器路线失败**。布局信息是特定干预下的损失差，非通用视觉能力分数。建议采用该诊断补充多模态 scaling 实验，尤其检查 OCR 与批量变化。[正文 §3–6、Fig. 5、Appendix H](https://arxiv.org/html/2611.26917v1)

<a id="topic-3"></a>
## 后训练

### Sentovara First Time Player：把动作token当Q值训练，等采样预算下并未全面胜过冻结骨干

**机构／日期／版本：Sentovara；11月23日；权重`680974a6bd2c`。建议作为小规模负面对照阅读，谨慎扩展。**在Qwen2.5-7B-Instruct的注意力投影上训练rank16 LoRA，用动作数字token的logit作为Q值做double Q-learning，最后合并权重；当前／旧游戏各半的经验回放控制顺序学习遗忘。这是直接动作打分模型，不能按自由文本聊天使用。[固定模型卡](https://huggingface.co/Sentovara/qwen2.5-7b-first-time-player/blob/680974a6bd2ce6dd2504d0de8403a8f7d6a94d76/README.md)

十个自制60步文字游戏、每游戏50个未见布局；4,000次练习的四训练运行均分34.3，对冻结模型加小学习器28.3，但后者只有2,000次。**等2,000次预算时29.4对28.3，整体基本持平**，主要优势集中在需读文字的Parcel；部分游戏退步，Cargo各法均接近0。发布的是seed1平均权重单checkpoint，单A10G完整训练约4.3小时；通用聊天和新游戏迁移未测。Apache2.0，训练任务与代码仍须进一步核验，不能据此称通用自主学习突破。

### 知识任务的结果奖励会强化推理链中的自造事实（2611.26863v1，补录）

**机构／日期／版本**：Cornell；2026-11-19 UTC 提交，v1。

研究在 QuALITY 的 CPT→SFT→GRPO 和 TriviaQA 直接 GRPO 两条流程中，提取模型自己生成的错误事实，比较第一次强化前后检查点的选择题回忆概率。六个 Qwen／OLMo Dense 与 MoE 模型、每数据集 3,609 训练问题的结果是：训练中被强化的自造事实平均回忆分数上升 4.83 点，未受训练信号的对照约 0.17 点；在链中撤回的错误也被记住。它不是“最终问答错误率上升 4.83%”的测量。错误事实由 GPT-4.1-mini 识别并转成探针题，探针和标注误差仍是边界；主实验没有 KL 惩罚，正文另做惩罚消融。建议深入阅读并复现事实型 RL 的链内事实审计，不能仅依赖最终答案奖励或自我纠正措辞。[正文 §2–5、Tables 1–2、Appendix A.4](https://arxiv.org/html/2611.26863v1)

### OPD 的伪一致：教师偏好的词元可能一直得不到足够更新（2611.27125v1，补录）

**机构／日期／版本**：University of Minnesota、Virginia Tech、Amazon、Ohio State 等；2026-11-20 UTC 提交，v1。

论文把在线策略蒸馏（OPD）的重复与长度失控，追到学生在教师 top-p 词元集上的概率质量不足、跨位置梯度干扰及停止词元错配。提出按状态将默认采样词元反向 KL 切换为教师 top-K 前向 KL。Qwen3-1.7B／4B／8B、8B 教师的数学对照中，从 Base 起步的平均 Avg@8 增加 2.36／2.50／1.72 点。Base 与 SFT 两面板的教师分别采用非思考／思考模式，**不能跨面板将全部差异归因于 SFT 初始化**；也不是所有单任务指标都赢。建议已有 OPD 训练的团队阅读，先记录覆盖质量、EOS 概率和长度分布，再决定是否复现状态路由。[正文 §4–6、Tables 1–2、Appendix F](https://arxiv.org/html/2611.27125v1)

### 稀疏教师支持集：省时来自离线缓存，截断 KL 本身几乎不提速（2611.14850v1，补录）

**机构／日期／版本**：NVIDIA；2026-11-11 UTC 提交，v1；本轮从历史待查线索补录。

在同一位置仅保留教师最高概率的少数词元，双方在保留集内重新归一化；Qwen3 4B→1.7B、三种子，k=15 的 MMLU-Pro 与全词表蒸馏接近，但**主实验rank-k支持集对照的MMLU-Pro中，所有蒸馏学生均低于原始冻结学生**，不能称普遍能力提升。64张 GB200、BF16/FSDP2 的实测中，在线截断仍需教师全词表前向，840 对853 ms/步几乎不变；提前缓存后519 ms/步（约-38%），再把学生输出头缩到缓存行后峰值显存73.4→55.8 GB（-24%），没有额外提速。缓存约108 GiB，预计算和分词时间不在该训练步计时内。Nemotron在更小支持集发生思考结束标签失效，Qwen还需标签支持修正。建议深入阅读并复现总成本和格式边界，不能将该离线固定轨迹方案直接等同OPD。[正文 §3–6、Tables 1–2](https://arxiv.org/html/2611.14850v1)

### 复用教师 RL rollout：先做兼容性筛选，再 SFT→DPO（2611.26803v1，补录）

**机构／日期／版本**：Carnegie Mellon；2026-11-19 UTC 提交，v1。

利用 GRPO 已存的成功率和教师 log-likelihood：全对问题做 SFT 热身，部分成功问题按易到难做偏好训练，正负样本选教师高似然的轨迹。数学与 Python 代码、4B／8B 教师到 1.2B–4B 学生的实验显示，直接随机复用 rollout 的 SFT 比重新从最终教师采样差，筛选加两阶段训练可接近常规蒸馏。新增数据构建不需再次推理，**不代表学生训练零成本**。各配置只有一个训练种子，八次采样是评测重复，容量差距更大及非二元奖励未验证。建议考虑复现；当前就可采用的工程动作是保留奖励、轨迹和生成时教师似然，避免丢弃高成本 RL 工件。[正文 §4–7、Table 1、Limitations](https://arxiv.org/html/2611.26803v1)

### CSPD：冻结同一基座，将 RL 专家蒸馏成可拼接软提示（2611.27376v1，补录）

**机构／日期／版本**：UCSB、CMU、UCSC 等；2026-11-20 UTC 提交，v1。

每个 RL 专家独立蒸馏成 256 个输入向量，以前缀扰动训练增强组合性，新增专家时拼接提示即可。23 个单专家设置中 22 个恢复至少 90% 的教师增益；四领域组合的主要实验仅一个 Qwen3-4B 系基座、单训练种子。B200／BF16／SGLang、20,480 输入＋512 输出的多适配器负载下，软提示吞吐接近无适配器基座；32 并发较未合并 rank-4 LoRA 为 2.40 倍。该负载的 LoRA 额外核开销不能推广成普遍速度差。**部分检查点和合并系数在评测基准上选择**；每专家增加上下文与 KV，换基座要重蒸馏。建议研究性复现组合与持出验证，先不要据此认定持续后训练已解决。[正文 §2–5、Tables 1–2、§3.6、Appendix B.3](https://arxiv.org/html/2611.27376v1)

### Gumbel Substitution：去掉采样器分布漂移的一部分，不等于无偏在线 RL（2611.27375v1，补录）

**机构／日期／版本**：Reliquary；2026-11-20 UTC 提交，v1。

协调器给定共享 Gumbel 随机数，采样器按自身 logits 生成，训练器用自身 logits 重算同一随机数对应的词元并替换评分，从而不依赖采样器上报概率。在遵循因果生成协议、同一输出头等条件下，固定前缀上的替换词元有正确边缘分布；但 **前缀和回报仍来自旧采样器，cross-credit 偏差仍在**。9B 异构网络运行 158 步、15.75B 词元、1.21% 替换率是生产可行性证据，不能单独证明算法相对其他校正更优；生产 fp32 选词、bf16 评分仍有约 0.01 的每位置 L1 残差。可看未来噪声、选择性 withholding、准入门限、未支持投机解码等均限制保证。建议深入阅读 §4/§8，先在受控环境复现偏差分解，暂不直接采用其“免信任”表述。[正文 §3–6、Tables 9/12、§8](https://arxiv.org/html/2611.27375v1)

### 受损 SFT 后仍能恢复推理：小规模对照不支持全面放弃数据清洗（2611.26756v1，补录）

**机构／日期／版本**：University of Michigan；2026-11-19 UTC 提交，v1。

OLMo-2-1B／GSM8K 上系统扫描 361 种数值错误配置，SFT 性能随污染下降，而随后用**完整、未污染 GSM8K 问题与答案验证器**做 GRPO，多数配置恢复到金标准 SFT 初始化附近。仅八条错误“格式示例”热身也能取得较好最终分数。主指标是 **pass@2**，不是通常默认的 pass@1；证据只覆盖单模型、单英文数学集、可用正确信号的后续 RL。建议阅读这个诊断结果，考虑复现格式热身与等预算对照，不能外推成错误事实或领域 SFT 数据无害。[正文 §3–6、Fig. 6、Table 1、Limitations](https://arxiv.org/html/2611.26756v1)

### CreditFilter：词元熵排名近似分叉价值，比较时必须核算生成预算（2611.27227v1，补录）

**机构／日期／版本**：NTU、Amazon AGI；2026-11-20 UTC 提交，v1。

以策略下一词元选择造成的成功概率方差定义“分叉价值”，再用响应内熵排名的指数窗口重分配优势，配合难度抑制和熵门限回退。Qwen3-8B-Base／DAPO-Math 三种子、20,480 输出上限的作者结果优于 DAPO；但 **相同步数下约训练两倍词元、回答更长**，故摘要中的分数差不是等 rollout 成本收益。精确分叉价值依赖昂贵续采样；实际窗口仅近似其中一部分，理论未覆盖过长惩罚。建议复现等 token／等时间对照，作为信用分配候选，而非无成本的高熵 mask 替代品。[正文 §3–5、Table 1、Limitations、Appendix F.1](https://arxiv.org/html/2611.27227v1)

### SH-PRM：把长文奖励切到语义段，收益需与裁判和预算分开验证（2611.27348v1，补录）

**机构／日期／版本**：SH-PRM 作者团队；2026-11-20 UTC 提交，v1。

在相同 GRPO 框架下，将长文按章节／段落逐步生成，针对遵循、连贯和相关性给局部 rubric 奖励，计算段级优势。6,163 条合成训练提示、三个 8B／32B 模型，使用同一预检索材料后，Qwen3-8B 的 DeepResearch Bench 总分由结果奖励基线 33.42 到 35.61，WritingBench 63.7 到 70.6。训练与主评估均使用 Gemini-2.5-Flash，另有有限 Qwen-A30B 复核；同裁判不消除奖励偏好迎合。建议阅读并考虑复现独立裁判、等总生成／评分成本和段边界消融，不能把预检索长文实验称作完整搜索 Agent 的提升。[正文 §3–5、Table 1、Appendix D.3](https://arxiv.org/html/2611.27348v1)

### DM-PRM：外部概率式决策模型提供步骤信用，额外服务是复现边界（2611.27068v1，补录）

**机构／日期／版本**：UIUC／NYU／Yale 联合作者团队；2026-11-20 UTC 提交，v1。

对每个 action→observation 询问是否推进任务，将外部 Jev 1.13 API 的概率在任务内标准化后加入原算法优势。Qwen2.5-1.5B、150 轮、每轮 16 任务×8 rollout，ALFWorld 对 GRPO 增加 23.7／29.2 点，WebShop 约 13.1 点。无需拟合 critic，但**仍有外部模型推理和费用**，作者不知道该服务训练数据；评分日志还包含最终成败，因此不是不看结果的在线进度预测。ALFWorld 单次运行约付 19 美元评分费，步骤减少抵消了部分时间。建议继续观察或对可控本地裁判复现，先核服务可得性和环境外泛化。[正文 §2–3、Table 1、Appendix C.2/C.4](https://arxiv.org/html/2611.27068v1)

### THRIFTY：合成数据按多样正确轨迹分配 token，不等于学生能力已提升（2611.27436v1，补录）

**机构／日期／版本**：Quantiphi Phi Labs；2026-11-20 UTC 提交，v1。

把教师采样分配写成带预算的多臂老虎机，联合成功率、解法新颖性、边际价值递减和长度，停止反复采已饱和或极难问题。固定 Qwen2.5-32B-Instruct，AIME／MATH500／LiveCodeBench 九个预算格中八格效率最佳，最多比均匀分配多 22% 的“不同正确轨迹／token”，包含约 9–12% 的校准开销。多样性使用 self-BLEU，教师难度不是学生难度；**没有学生 SFT 下游验证**，也不能将测试集上的数据采样实验分数当作最终学生泛化。建议作为数据制备对照复现，保留对稀有题和语义解法覆盖的检查。[正文 §3–5、Tables 1–3](https://arxiv.org/html/2611.27436v1)

### Prism：不需验证集的批次筛选，半批更新仍可能每步更慢（2611.27578v1，补录）

**机构／日期／版本**：四川大学作者团队；2026-11-20 UTC 提交，v1。

用预条件化 ghost-gradient 内积和正射线覆盖替代原始欧氏距离，选择批次子集并给径向权重，避免大梯度离群点支配。四个 1.5B–8B LoRA 配置中，16 选 8 的下游指标优于全批基线，但端到端单步 **0.83 对 0.52 秒、1.92 对 1.28 秒**，仅比其他更重的选择器快；不能直接写成节省训练时间。早期新格式学习的 300 步冷启动下反而更差，1,000 步才反超；理论针对预条件 SGD，实际用 AdamW。建议阅读其几何修正，复现应报告到达目标质量的总时间，并保留全量热身。[正文 §4–6、Tables 1–2、§6.5、Limitations](https://arxiv.org/html/2611.27578v1)

### Gemma-4 梯度病态：提高精度未必能修复训练路径（2611.27243v1，补录）

**机构／日期／版本**：Samsung Medical Center／成均馆大学；2026-11-20 UTC 提交，v1。

在 Gemma-4-12B 图像提示上，状态替换实验把低层 bf16／fp32 梯度方向分离，定位到 19–21 层图像词元的 MLP 反向路径。保持前向不变、在这些位置停止梯度，私有 CT 定位任务四 LoRA 种子、300 步的验证 NLL 从 0.336±0.019 到 0.296±0.003；单种子 fp32 仍有梯度尖峰。**替代梯度是有偏的，且未证明更准确估计原计算图的标准 fp32 梯度**；它在这个任务上训练更好，不能据此称数值更正确。只有一任务／模型／数据顺序，长纯文本的机制尚未定位。建议深入阅读诊断流程，其他模型遇到精度问题时先验证扰动敏感性，不直接复制 detach 修复。[正文 §2–5、Tables 1–2、§7](https://arxiv.org/html/2611.27243v1)

<a id="topic-4"></a>
## 推理与部署

### SenseNova Loop DFlash2：0.97B草稿带来四卡实测，LightLLM集成仍未合入

**机构／日期／版本：商汤；11月23日；草稿`02eb882d73b8`。建议考虑复现，保持实验分支状态。**967M／5层BF16草稿读取循环目标模型执行层3、27、51、54以及最终归一化隐状态，分享embedding与输出头；它是另加载的DFlash2工件，不是目标内置MTP。四H800、TP4、BF16、线性注意力状态FP32、每轮5草稿、3030固定提示／九子集、最大输出2048、同prompt种子下，c1输出吞吐125.61→268.37 tok/s（2.137倍），c32为1899.32→2550.62（1.343倍）。每点仅一次，节点另四卡有独立负载，不能当隔离机器稳定均值。[固定卡](https://huggingface.co/sensenova/SenseNova-Loop-9B-DFlash2/blob/02eb882d73b80e8623e81ef038680887427c8d4d/README.md)

LightLLM #1757核实为**open／未合入**；GPU集成在旧基线`38793654`通过，rebase到`0c0b0683`后只重跑CPU与风格检查，未重跑GPU。权重Apache2.0，启动须使用该PR分支，并验收目标分布、统计和完整端到端负载。[PR状态及验证边界](https://github.com/ModelTC/LightLLM/pull/1757)

### Red Hat Qwen3.8 DFlash2补充并发曲线：单用户优势在满载时基本消失

**机构／日期／版本：RedHatAI；草稿11月9日发布，本次11月23日新增RAG／摘要对照；固定`2981a6bd815c`。建议深入阅读并按线上并发复现。**四H10080GB、TP4、关闭前缀缓存、7草稿、最大输出1024的closed-loop测试，作者DFlash2在单用户RAG／摘要为target-only的2.68／2.76倍，到128并发只剩1.02／1.03倍；原生MTP3在128用户RAG超越DFlash2，摘要基本持平，表明更高接受长度不保证满载吞吐更高。[新增曲线与条件](https://huggingface.co/RedHatAI/Qwen3.8-27B-speculator.dflash2/blob/2981a6bd815c90f128ebe1e54eefccd484ded5d2/README.md)

每负载点单次测量；DFlash2／target为vLLM0.33.0，MTP为0.34.0，存在运行栈版本混杂，不能把交叉点全部归因给草稿结构。卡片原有0.30.0 HumanEval／数学测试与本次RAG曲线亦须分读；x轴是1000／平均token间延迟而非首token时延。Apache2.0，本轮新增的是条件化性能证据，未重发草稿或自行压测。

### Red Hat Hy4 NVFP4：给出非专家张量逐位校验，质量测量仍受思考预算约束

**机构／日期／版本：RedHatAI；11月21日工件、11月23日补充创建及评测说明；固定`88e74418d42f`。建议考虑复现量化与权重转换。**仅77个MoE层的路由专家用NVFP4权重／激活，其余注意力、索引器、共享专家、视觉、路由及DFly2草稿保留BF16，文件约1.55TB→479GB。采用512条公开perfectblend样本、长度2048校准；因Transformers反向命名转换有损，另做名称和配置恢复，并逐张量检查全部非专家张量与基座逐位一致。这是量化格式落地与转换正确性的证据，非整网运行显存或提速证明。[固定卡与创建脚本](https://huggingface.co/RedHatAI/Hy4-NVFP4/blob/88e74418d42f061b06c2b99e23e5a0a8df9c1fa3/README.md)

4×B300／TP4、同vLLM与lighteval、high思考、单seed、64K总预算时，GSM8K严格匹配96.28%对FP8的96.77%，GPQA89.90%对91.41%；GPQA标准误约2.1点，两模型各7/198在一次重试后仍因思考耗尽而无最终答案，计错。提取格式也影响GSM8K，不能把该表称无损。内含DFly2权重；**卡片仍写依赖PR未合入，但连接器回读确认vLLM #69274已于11月23日16:24Z合入**，运行时应核所安装版本是否含此提交，不能沿用卡片旧状态。许可沿腾讯Hy条款；本轮未验证合入后该量化工件的投机运行。[DFly2实时合入状态](https://github.com/vllm-project/vllm/pull/69274)

### OpenJev 1.1：重读提示改善部分判断，却付出2.6–3.3倍整集时间

**机构／日期／版本：OpenJev独立项目；11月23日创建，11月24日00:12Z卡片更新；BF16固定`f2db5ff4c648`、FP4`ad43ef860870`。建议对文本路由任务考虑复现。**Qwen3.8-27B衍生决策模型将请求选项映射为首输出位置的字母logits，再固定校准为概率；`READOUT_REPEAT=1`在同次请求中重复整段状态／问题／选项，不增加模型轮次。11,650公共问题、H100加载FP8、16并发、单次回答的作者配对bootstrap显示重读让EvalSafe-O*NET+1.71点、When2Call+2.33点，HLE无明确变化。[固定卡](https://huggingface.co/openjev/OpenJev-1.1/blob/f2db5ff4c648f2cae596ac7c764741f6d30993b4/README.md)

When2Call中“应当调用工具”子集反而下降0.46点，改善主要来自应拒答情形。H100整集墙钟491→1625秒，B200 FP4为281→719秒，**不是每请求时延或负载测试**；FP4相对B200 FP8只286→281秒，不能宣称显著量化加速。64K输入可接受，但>16K与截图质量未测；EvalSafe过滤6611题也不同于AA的6582题，非官方Decision Index。权重CC BY-NC4.0，helper Apache2.0，完整训练数据未充分披露。

### LRSA：索引也需要分页、异步维护和前缀复用（2611.27233v1，补录）

**机构／日期／版本**：美团 LongCat；2026-11-20 UTC 提交，v1。

面向 DSA（DeepSeek Sparse Attention）选键仍扫描全部历史的瓶颈，将时间桶内的相似 key 做容量均衡分组，每页保存完整组及各组中心，先选页再精排；SGLang 实验实现异步建桶与匹配前缀后的索引复用。H20、BF16 权重与原始 KV、FP8 索引、64K–1,024K 上下文下，1M 选键对 DSA 为 4.70 倍，选键＋注意力对全注意力为 35.60 倍；**端到端对 DSA 为 1.865／1.957／1.984 倍**，对应最终 K=2048／1024／512，包含建索引等开销。质量只在 Qwen3-4B、Hunyuan-7B、K=2048 的十任务平均上较 DSA 低 0.07／0.08 点，单任务差异更大；页预算按相对DSA Top-K参考集合的全query平均注意力质量召回95%标定，不是dense全注意力或逐query保证。建议深入阅读并考虑复现，先验证索引内存、预算标定、前缀复用及异步竞态；论文集成不等于上游已发布能力。[正文 §3–6、Tables 1–2、§6.6](https://arxiv.org/html/2611.27233v1)

### CTRL：让训练和 rollout 在每张 GPU 共驻，代价转移到主机（2611.27119v1，补录）

**机构／日期／版本**：港科大（广州）；2026-11-20 UTC 提交，v1。

每卡用 CUDA MPS（多进程服务）并行运行常驻 rollout 与 learner，优化器和版本权重主要放主机，GPU 只保留有界逐层窗口；先训练已就绪批次，再在发布新策略前等待，以保证有界陈旧且不丢样本。H100／A800／4090、Qwen3 的 24 个设置对较强 verl 异步基线吞吐为 1.11–2.24 倍；同 learner 的执行布局对照才更接近隔离共驻收益，约 1.10–1.26 倍。指标包括输入和输出词元，窗口为第 5–12 步并到最终权重发布。14B 的主机训练状态约 220 GiB，每租用 BF16 策略映像另 27.5 GiB；不能把省显存写成总内存下降。150 步质量对照只两个种子。建议阅读并考虑在主机带宽充足环境复现，单独核 MPS 共享、PCIe 和策略版本一致性。[正文 §3–4、Tables 1–2、§4.5](https://arxiv.org/html/2611.27119v1)

### Contexture：结合工具语义与全局 KV 压力，决定何时删历史（2611.26980v1，补录）

**机构／日期／版本**：NTU、A*STAR 等；2026-11-20 UTC 提交，v1。

工具后续读写／测试可标记被替代的旧结果，但立刻删除会破坏已缓存前缀。系统把完整会话与执行上下文分开，结合宽限窗口、全局 KV 压力和重 prefill 上限，延迟提交删除。Qwen3.8-27B/H200、Qwen3.6-35B-A3B/H100、FP8、vLLM 0.26、Hermes/OpenClaw 两基准八配置，在 **p90 相对空载慢 10 倍** 的 SLO（服务水平目标）下，比除 observation masking 外最强基线多完成 20–73% 会话；对 masking 仅五个配置提高 7–63%。性能主要来自录制会话重放，另做筛选过任务的 live-agent 准确率探测，均值 +1.4 点、最差 -3.7 点，非无损保证。建议阅读并考虑复现，保留负载、失败任务和历史删除的语义审计。[正文 §5–6、§6.2.1/6.2.3](https://arxiv.org/html/2611.26980v1)

### DICE：把验证随机数提前提供给 draft 预算分配（2611.27891v1，补录）

**机构／日期／版本**：Carnegie Mellon；2026-11-20 UTC 提交，v1。

不改变 drafter 权重，而让树构建和异步预取利用与验证阶段同步的预采样随机数，将有限节点预算放到本轮更可能执行的路径。H200、温度 1、并发 1、18 个模型与后端配置，Spec-Bench 吞吐较原树方法提高 4.3–21.5%，SPEED-Bench 为 5.6–24.6%；这是相对吞吐增幅，**不是准确率百分点**。异步 SSD 预取另提高 4.0–13.7%，温度 0 时无增益。分布不变是正确随机数耦合及验证实现下的算法保证，不等于跨数值实现逐词一致；高并发和不同目标／草稿对仍要重测。建议考虑复现，重点核随机状态按位置和请求的归属、拒绝后的更新及可重复性。[正文 §3–5、Tables 1–2、Appendix B](https://arxiv.org/html/2611.27891v1)

### ALD：并行计算草稿层代理，必须把缓存重建计入收益（2611.27477v1，补录）

**机构／日期／版本**：IIT Bombay、Singapore-MIT 等；2026-11-20 UTC 提交，v1。

从共享隐藏状态并行计算多层代理变换，以隐藏空间相似度决定能前进多少层，草稿模型无需额外训练；目标模型仍验证。Qwen 4B→30B、Llama 3B→70B 的五类任务给出草稿耗时及对自回归的端到端速度。并行代理不是免费跳层：K=10 的一轮自身约耗单层的 2.4–2.7 倍，需平均跨越足够多层才能盈利，并新增同步、相似度计算和 KV 重建。相似度阈值与后端实现决定接受率，不能把最快草稿倍数直接当全系统加速。建议继续观察，需开发环境复现完整重建路径与现代强投机基线后再考虑采用。[正文 §4–5、Table 1、Appendix A.8](https://arxiv.org/html/2611.27477v1)

### Vega-Fafnir：低 margin 回退单 token，仍可能读到“另一条计算路径”的 KV（2611.27554v1，补录）

**机构／日期／版本**：Intel 作者团队；2026-11-20 UTC 提交，v1（文稿落款 11-19）。

Qwen3-0.6B、RTX4090、BF16、128 输出、prompt-lookup draft 的对照中，四 token 验证与单 token 参考的整段文本在 90/150 提示上不同。只在低 logit margin 时回退单步，却复用宽验证写入的 KV，仍有 58/100 不同；重建为单步路径 KV 后 0/100 不同，代价 7.24 次目标前向／输出 token。固定宽度能让 draft 与普通固定宽度执行一致，但**两者仍有 66/100 不同于单步参考**，不是恢复某个唯一正确答案。证据仅一软硬件栈，0/100 也非普遍精确性保证，前向次数不等于真实时延。建议深入阅读，为训推对齐和投机正确性测试补充“缓存由哪种数值路径生成”的条件。[正文 §3/6–10/13、Tables 4–5，PDF](https://arxiv.org/pdf/2611.27554v1)

### IsoMoE：专家在 CPU/GPU 间移动，输出仍可保持一致（2611.27604v1，补录）

**机构／日期／版本**：中国科学技术大学等；2026-11-20 UTC 提交，v1。

GPU 专家核按 CPU 核的运算次序回放，配合可用 AVX-512 点积直接处理的无损 12-bit 编码和双微批流水，避免专家放置改变贪心结果。Xeon 8488C＋RTX A6000／PCIe4×16、三个 MoE 模型、批量 8–32，在所测专家显存预算下较最快基线平均 1.26 倍、最高 1.54 倍；Qwen3 的 160 问与动态换入子集验证了放置一致。它保证的是**同一数值参考下的放置不变**，非等价于任何 BF16 实现，12 bit 是主码字，异常行段另有旁表，实测平均约 12.1 bit，不能当作固定 12 bit 有损量化。基线 KTransformers 被接入作者的 GPU slots／流水以支持公平比较。建议混合推理团队考虑复现，先确认权重可编码范围、CPU 指令集和精确模式成本。[正文 §IV–VI、Tables I–II](https://arxiv.org/html/2611.27604v1)

### Weave：按路由结果调整 MoE megakernel 的 warp 角色（2611.27765v1，补录）

**机构／日期／版本**：上海交大、NUS、复旦等；2026-11-20 UTC 提交，v1。

在专家并行（EP）融合核中同时决定 SM／warp 的通信计算比例，并让 warp 在头尾阶段切换角色，以运行时成本模型适配各卡路由差异。EP8 的 SM103／SM90 对 MegaMoE 层耗时几何平均加速 1.30／1.24 倍，双节点 CDNA3 EP16 对最强官方 RCCL＋AITER 为 1.44 倍。**权重和激活为随机数，路由来自真实 ShareGPT**，不是完整真实模型精度验证；SM90 整模 decode 仅约 1.08 倍，prefill 的收益更小。部分替代基线因正确性或网卡注册内存限制无法在全范围比较，NPU 未测。建议阅读调度与成本模型，复现时保留真实路由、跨节点条件和整模开销。[正文 §III–V、Table IV](https://arxiv.org/html/2611.27765v1)

### MixFold：为手机 NPU 的原生整数矩阵单元折叠 2/3-bit 权重（2611.26976v1，补录）

**机构／日期／版本**：MixFold 项目；2026-11-20 UTC 提交，v1。

通过行和／位平面折叠，使非原生 W2／W3 不必在解码时先扩成更宽整数；混合精度按与矩阵 tile 对齐的 32 输出通道分组。Snapdragon 8 Elite Gen5 Hexagon、Qwen3 0.6B–8B、A16 和 FP16 KV、单流 tg128，按每权重平均位数匹配 llama.cpp b11444 的亚四位格式，作者报告 1.16–1.41 倍吞吐及 1.5–4.3 倍整机 tokens/J。质量大多由 GPU AIMET 模拟，NPU 只核对部分困惑度；优势对单线程向量扩展基线最成立，多向量线程可缩小差距。M>1／投机／多请求端到端未验证。建议相应移动 NPU 团队考虑复现，不能迁移成通用低比特硬件结论。[正文 §3–5、Appendix A/E](https://arxiv.org/html/2611.26976v1)

### BitPrefill：将 ternary 权重和 A4 激活落到 INT4 Tensor Core（2611.26657v1，补录）

**机构／日期／版本**：Case Western Reserve University；2026-11-18 UTC 提交，v1。

面向低比特 decode 已优化而 prefill 仍跑 BF16 的缺口，对 BitNet-b1.58 权重映射到原生 INT4、激活量化到四位，加入按形状选核和 epilogue 反量化融合。RTX4090 的 3B、Jetson Orin Nano 的可运行 700M 对照，分别报告约 1.42／1.13 倍 prefill 加速；不是所有设备和规模都达到同一倍率，Orin 3B 的部分原基线内存不足。A4 通过 85,805 样本的量化感知 SFT 恢复质量，因此不是对任意现成 ternary 检查点的无训练替换。建议继续观察或复现算子边界，比较时包含量化、打包和反量化成本。[正文 §III–IV、Tables I–III](https://arxiv.org/html/2611.26657v1)

### PLoQ：按提示保护低秩激活方向，质量收益需换取额外状态（2611.27092v1，补录）

**机构／日期／版本**：PLoQ 项目；2026-11-20 UTC 提交，v1。

prefill 时估计每个提示的主导低秩方向，四位量化后在 GEMM 内恢复，避免只靠离线固定高精度子空间。六个 Llama／Qwen 模型，按平均激活位数与 ResQ 匹配，最多恢复其激活量化所损困惑度的 46%；**γ、b 曾在测试数据选择**，数值应待真正持出集复核。4090、Llama3.1-8B 形状、batch1／8K 时，相对 ResQ prefill 多 4.1%、decode 多 7.9%；每序列状态约 424 对 336 MB，非零内存代价。2K prefill 开销达 14.2%。建议继续观察或考虑复现独立校准与质量—总容量对照，不能只看平均位数。[正文 §3–5、Tables 7–8、Appendix H](https://arxiv.org/html/2611.27092v1)

### RelaxQAT：连续改变训练位宽，而非突然切换量化网格（2611.27056v1，补录）

**机构／日期／版本**：RelaxQAT 作者团队；2026-11-20 UTC 提交，v1。

将量化感知训练（QAT）的位宽做连续周期调度，缓解低位网格平坦区，并实现 Triton 版本以控制训练开销。Qwen3／Qwen3.5／Gemma3 的 0.6B–4B、10B 词元、**仅权重量化**实验中，Qwen3-0.6B 的综合均分 47.04 对 ParetoQ 39.34，+7.70 个百分点，仍低于浮点 54.74。H100 的 361 对 370 GPU 小时被作者视为误差内相近，不能称确定训练提速；也没有由此证明 W2A2 推理核加速。建议低比特 QAT 团队考虑复现位宽调度，重点看最终离散网格、端到端质量和部署格式。[正文 §3–5、Tables 1/3](https://arxiv.org/html/2611.27056v1)

### AVERT：每请求平均深度预算，可让同批不同 token 执行不同层（2611.27738v1，补录）

**机构／日期／版本**：山东大学等；2026-11-20 UTC 提交，v1。

路由器按当前 token 状态和请求预算决定是否执行层；跳过时经过轻量适配器，并不写入该层 KV。一个检查点可混批不同预算，三种 Llama 基座、相同请求集合的实验中，避免按预算分组导致欠批后，输出 token 时延改善 4.3–21.2%。**目标是平均执行深度，不是精确时延预算**；1.15 点是所测组的最大平均预算误差，单请求可更大。prefill 仍全深度，激进预算会损失 GSM8K 质量，生成中途改预算尚未实现。建议研究可变计算的团队阅读，考虑复现混批容量与稀疏 KV 布局；不宜据此承诺逐请求时延。[正文 §3–5、Appendix E/L](https://arxiv.org/html/2611.27738v1)

### Baton：把小视觉模型的 KV 映射成大模型可用的历史记忆（2611.26902v1，补录）

**机构／日期／版本**：University of Pittsburgh、HKU；2026-11-19 UTC 提交，v1。

跨模型逐层线性映射若只拟合大模型自己的 KV，缓存更像却不提升任务；Baton 改由大模型答案蒸馏映射，大模型仅重编码当前帧。RoboMME、Qwen3-VL 2B→8B、最多64帧，在25%升级预算时，相对重新看历史的级联准确率差约1点、分析计算量少28–47%。**TFLOPs是解析计数，未提供等价设备时延结论**；RoboMME 的两模型先有标签 LoRA 训练，仅映射无标签。每个同族 writer–reader 对都需训练专用映射，定位任务收益更弱。建议继续观察，若已有小大模型级联，可复现真实解码时延和跨域持出验证。[正文 §4–6、Tables 1/3、Limitations](https://arxiv.org/html/2611.26902v1)

<a id="topic-5"></a>
## 开源工程

除明确写正式发行的补录外，以下为默认分支源码增量，不能自动等同当前稳定发行。日期分别记录原始合入/commit时间与本次发现日；性能均为来源作者的测试，本文没有独立设备实测。

### ReFlow：ZeRO-3 把 CPU 优化器计算与反向传播重叠

一手来源：[deepspeedai/DeepSpeed #8719](https://github.com/deepspeedai/DeepSpeed/pull/8719)；[deepspeedai/DeepSpeed #8714](https://github.com/deepspeedai/DeepSpeed/pull/8714)。

版本/原始日期（UTC）：#8719 2026-11-24T01:02:19Z / `fc8d7ac8621e`；#8714 2026-11-23T23:45:15Z / `1b5261a05d30`。发现日期：2026-11-24。

**补充取证：ReFlow于主目录截止后54秒实际合入，本条纳入截至01:18:36Z的定向补充；不把目录覆盖延长至该时刻。**

ReFlow 新增 CPU Adam/AdamW 与 Lion 路径：BF16 梯度在 SIMD 寄存器中提升至 FP32，中间微步在 CPU 累积，末微步才做 GPU 最终累积及梯度范数；桶级异步更新把 GPU 需要的低精度参数生成与 FP32 持久状态更新拆开。它解决 CPU 卸载后优化器尾部暴露的问题。

作者披露 OPT-30B 对 ZeRO-Infinity：1×B200 的 166.41→671.29 TFLOPS/GPU，4×B200 为 366.40→681.18；标准 AdamW 的暴露更新尾部下降超过 96.8%，与 clipping 专项的 2440→850ms 是不同实验。作者称确定性执行下训练逐位相同；论文尚未公开，PR 没给出所有 CPU、批量与序列条件，不能把 4.0 倍推广到任意卸载系统。另 #8714 修复 CPU Adam/Adagrad/Lion 对非连续参数更新写入临时 contiguous 副本后丢失的问题。

建议深入阅读并考虑在 CPU 卸载是瓶颈的长跑中复现；先复用其正确性矩阵和配置 zero_optimization.reflow，再比较完整迭代与主机拓扑。本文未运行训练。

### Megatron FsdpMuon：跨参数组流水化通信，优化器跨度缩短

一手来源：[NVIDIA/Megatron-LM #9681](https://github.com/NVIDIA/Megatron-LM/pull/9681)。

版本/原始日期（UTC）：#9681 2026-11-23T05:20:59Z / `a924ee5d5111`。发现日期：2026-11-24。

FSDP（Fully Sharded Data Parallel，全分片数据并行）Muon 原先逐组 gather→Newton–Schulz 正交化→scatter 并等待。新 gather/scatter 返回在途句柄，FsdpMuon.step 提前发布后续组，并延后应用已发送更新；num_groups_in_flight 默认 8，0 保留串行次序，更多在途组占用更多临时缓冲。

作者在 DSV4 7 层代理、64×GB200、EP64、MXFP8，抽 rank 0/32/63 的第 9–11 步 Nsight：旧串行 1186ms，新代码关闭重叠 1217ms，在途 2/8 组为 986/848ms；分布式 Muon 参考为 728ms。848ms 相对旧串行约缩短 28%，这是优化器阶段，未给整步同幅加速。每设置仅一次；该实验优化器峰值仍低于前反向 12–19GiB，不代表其他模型无显存代价。8×H100 单测检查 0/2/8 组与未分片训练一致。

建议做 Muon+FSDP 者优先阅读，按可用缓冲与通信占比复现。

### TorchTitan Adam：固定页锁定主机矩支持零拷贝卸载

一手来源：[pytorch/torchtitan #5881](https://github.com/pytorch/torchtitan/pull/5881)。

版本/原始日期（UTC）：#5881 2026-11-23T17:28:28Z / `ed0206d0e682`。发现日期：2026-11-24。

Adam/AdamW 的 moment_cpu_offload_fraction 将整参数粒度的一阶/二阶矩放到 pinned CPU 内存，再以共享地址 CUDA 视图交给原 fused kernel；计算与主机链路读写在同一 kernel 内推进。省去双缓冲 GPU slot；state_dict 暴露真实 CPU 张量，保存前同步当前流，加载保留地址与 dtype，因此可继续使用已捕获优化器图。

关键条件：只支持 CUDA fused Adam/W，拒绝 ROCm、TorchFT、非连续或 CPU 参数；需 NUMA 绑定与 pinned_max_round_threshold_mb:1。所有优化器 checkpoint load（即使卸载比例 0）依赖 2026-10-30 后的 torch.optim.preserve_state_dtype_on_load，旧版本恢复会报错。

作者的 Muse Spark 816B 两套容量/吞吐实验均为单 GB300 只执行模拟 64/256 卡布局的 rank 0（fake process group），不能写成真实 64/256 卡训练。模拟 256 卡、96K 配置全卸载：reserved 253.06→228.84GiB，3301→3273tok/s，AdamW 26.2→166.5ms；真实性能依赖共享 Grace 带宽。性能使用 cudaHostRegister 分配，最终 pin_memory 和部分卸载只在单测检查。建议容量受限且有高速主机链路者考虑复现，并单独验证双 GPU 共享 socket 和 checkpoint。

### TorchTitan Ulysses：复制 KV 头解除部分 GQA 的 CP 度数限制

一手来源：[pytorch/torchtitan #6285](https://github.com/pytorch/torchtitan/pull/6285)。

版本/原始日期（UTC）：#6285 2026-11-23T17:27:15Z / `cb58b4603c90`。发现日期：2026-11-24。

当 KV 头数小于上下文并行度（CP）时，Ulysses 原先因无法给每 rank 分配完整 KV 头而拒绝。现在在 all-to-all 前相邻复制 KV 头，例如 4KV/CP8 将每头复制两份，使每 rank 的 Q 分组仍对应正确 KV；这增加 KV 通信与存储，换取 packed 长文档下更均匀的注意力工作量。

只放宽 full-attention 的整除条件；GDN、KDA、MLA 仍保留严格规则。作者 8×H100、CP8、Qwen3.5-9B 的四层代理（2GDN+2全注意力）、256K packed、FA3，第6–15步：Ulysses 1.152s，K/V all-gather 的 head-tail/连续分片 1.346/1.453s；不同方法后续 loss/梯度有小数值差，非全程逐位等价。建议少 KV 头且 CP 负载不均者考虑复现，不能据四层代理认定全模型同幅收益。

### TorchTitan RL：基础设施失败不再进入奖励与组均值

一手来源：[pytorch/torchtitan #6116](https://github.com/pytorch/torchtitan/pull/6116)；[pytorch/torchtitan #6182](https://github.com/pytorch/torchtitan/pull/6182)；[pytorch/torchtitan #6183](https://github.com/pytorch/torchtitan/pull/6183)；[pytorch/torchtitan #6204](https://github.com/pytorch/torchtitan/pull/6204)；[pytorch/torchtitan #6344](https://github.com/pytorch/torchtitan/pull/6344)。

版本/原始日期（UTC）：#6116 2026-11-23T19:34:16Z / `bb81f1446d7f`；#6182 2026-11-23T19:35:11Z / `1715fcafb406`；#6183 2026-11-23T19:56:46Z / `56a07ca65d8e`；#6204 2026-11-23T19:57:33Z / `a7887db53aad`；#6344 2026-11-23T23:25:08Z / `2559eca0e12f`。发现日期：2026-11-24。

四个状态处理 PR 将故障归属贯通 native worker、TokenEnv、Verifiers 与 TrainingSampleBuilder：网络/沙箱/生成异常保留为 ERROR_INFRA 供计数，既不打分也不训练；首次提示已占满预算单列 FIRST_PROMPT_TOO_LONG，之后的策略超长仍可训练。坏 rollout 只丢自身，剩余可训练兄弟重新计算组均值；保留零 advantage 样本以免改变 token 归一化，并新增按原因丢弃和 zero_advantage_frac 指标。

证据是 CPU 回归与一段 Qwen3.5-27B Terminal-Bench 20步日志：17504 条中 116 条外部失败以前会被当作策略失败；这是旧日志重分类，未证明重训后的能力提升。缺失或未知失败类型默认不训练。另 #6344 删除 alphabet-sort 等配方的 BF16 master 覆盖，恢复 FP32 master/梯度/Adam，前向及下发仍 BF16；trainer 参数、梯度和优化器状态的内存随之增加。大 Qwen3.5 的 A100 本地训推差仍未归因，不能写全部已逐位对齐。

建议 RL 实践优先审查此类样本语义和 master dtype；它们直接改变实际优化数据与更新。

### verl：LoRA+ 与共享外因子适配器打通优化器和 rollout

一手来源：[verl-project/verl #7484](https://github.com/verl-project/verl/pull/7484)；[verl-project/verl #8032](https://github.com/verl-project/verl/pull/8032)。

版本/原始日期（UTC）：#7484 2026-11-23T16:33:05Z / `de742368df5d`；#8032 2026-11-23T16:32:14Z / `8d6ae4696540`。发现日期：2026-11-24。

Megatron 后端用 model.lora.lora_plus_ratio 控制 B/A 学习率比（默认1）；共享外因子 MoE LoRA 可按 SGLang 原生共享因子、逐专家/堆叠格式或 vLLM 布局导出，HF checkpoint 则展开成 PEFT 可加载形式。Bridge 0.6.2 的兼容路径关闭可训练 EP 复制投影的 wgrad fusion，以让既有 hooks 归并真实梯度；legacy 共享投影的可训练或已分片情形要求 pipeline-stage 与 EP 成员一致。

另新增默认关闭的 lora.resync_base：adapter 模式每轮先重新传基座、再传 adapter，换取睡眠期释放。vLLM HYBRID 合条件时用 level2 丢弃基座；COLOCATED、MTP 和 NPU 仍 level1。SGLang 虽释放 GPU 基座，HYBRID 的 CPU backup 仍存在，不能宣称同时释放主存。重新同步代价接近全权重 refit再加 adapter。LoRA+有8×L20 E2E，resync_base 主要20个CPU/Gloo测试。

建议 LoRA RL 内存吃紧者考虑复现；按后端分别核 GPU、CPU峰值和每步全基座传输代价。

### DeepSpeed ZeRO-1/2：超大梯度必须延长到 reduction stream 完成

一手来源：[deepspeedai/DeepSpeed #9031](https://github.com/deepspeedai/DeepSpeed/pull/9031)。

版本/原始日期（UTC）：#9031 2026-11-23T07:24:52Z / `8df84cf58bf9`。发现日期：2026-11-24。

超过 reduce_bucket_size 的梯度绕过持久 IPG bucket，overlap_comm 下非所属 rank 清空 param.grad 时，reduction stream 可能仍在读取原存储。分区跨边界后 flatten 复制也可能没有记录源流；修复在 average_tensor 前 record_stream，阻止 allocator 提前重用。

作者在8卡PCIe、无NVLink、BF16、Qwen2.5-7B形状、ZeRO2+checkpointing 的旧版本第3步NaN；当代master的溢出保护把症状变成40步跳过39次更新、权重冻结。额外成本一次record_stream/超大参数/步，12步观察3.542→3.565s，不能据此保证所有负载0.6%。附三rank的确定性延迟/复用注入复现，natural小模型未自然触发。建议使用相应组合者优先审阅与回归，有限权重/低loss不能替代更新有效性检查。

### DeepSpeed AffineIR：把布局并集、传输分析与候选排序接起来

一手来源：[deepspeedai/DeepSpeed #9052](https://github.com/deepspeedai/DeepSpeed/pull/9052)；[deepspeedai/DeepSpeed #8967](https://github.com/deepspeedai/DeepSpeed/pull/8967)；[deepspeedai/DeepSpeed #9145](https://github.com/deepspeedai/DeepSpeed/pull/9145)；[deepspeedai/DeepSpeed #8847](https://github.com/deepspeedai/DeepSpeed/pull/8847)。

版本/原始日期（UTC）：#9052 2026-11-23T04:26:38Z / `5eb0b3c9cdc7`；#8967 2026-11-23T04:03:45Z / `9d53dfb2cd1a`；#9145 2026-11-23T09:35:06Z / `991375804cc2`；#8847 2026-11-23T23:11:40Z / `6d6aa58959b2`。发现日期：2026-11-24。

union_affine_maps 将同一参数的存活TP分片与ZeRO等多个布局在边界切齐，保留每片真实持有者和步幅，允许重分布继续复用存活副本；切片数可能越过默认4096预算，千分区列切例要显式提高规划预算。analyze_transfers 从真实executor调度导出轮数、字节和staging分布；local_copy_bytes 对原地副本是上界。

新的独立TransferCostModel按每轮最忙链路、调用与区域开销排序，cheapest作确定性选择。在两台机器各取4×A100/4×H200、五类移动16个候选中，所选相对实测最优regret为0–1.4%；但绝对估时可低估62%，不含准备和跨阶段重叠，未做跨节点实测；**并未自动接入训练、checkpoint或ZeRO热路径**。

另AutoEP+AutoTP folding 接通可选 fused_weighted_sum，保留router精度并融合恢复，默认仍weighted_sum；2×H200的BF16恢复前反向微测，32768token/top8/256expert/hidden6144为91.9→38.8ms、反向峰12.38→6.38GiB，非完整训练加速。建议重分布/训推权重更新团队深入阅读接口与预算边界。

### Megatron DSv4 分组输出投影保留优化器和 checkpoint 契约

一手来源：[NVIDIA/Megatron-LM #7971](https://github.com/NVIDIA/Megatron-LM/pull/7971)；[NVIDIA/Megatron-LM #9557](https://github.com/NVIDIA/Megatron-LM/pull/9557)；[NVIDIA/Megatron-LM #9598](https://github.com/NVIDIA/Megatron-LM/pull/9598)。

版本/原始日期（UTC）：#7971 2026-11-23T21:11:32Z / `74f265977aba`；#9557 2026-11-23T19:18:28Z / `101ba65387fc`；#9598 2026-11-23T23:12:00Z / `5265d33902fc`。发现日期：2026-11-24。

TE BatchedLinear 适配器以零拷贝3D视图计算分组投影，底层仍保留一个旧2D参数供Muon整矩阵更新和模型/优化器checkpoint；TE缺该API时走einsum，primary MXFP8缺columnwise数据/scale时前置拒绝。2×B300隔离TE版本每rank20测试通过，覆盖累积/初始化/恢复；当前main依赖与该实测TE不同，早期dev报告4%迭代收益未在最终main端口复测。

推理侧 #9557 将 FP32 residual 输入在 fused norm+linear 入口按norm权重dtype转换，修复 inference_optimized 与BF16参数首forward类型冲突，保留原残差张量。#9598 接通MXFP8 shortcut overlap，作者列Nemotron请求数32/128/768为1.248/1.218/1.12倍，但模型具体规模、硬件、长度/负载未充分披露，不能直接选型。建议读源码整合与兼容条件；性能数字继续观察。

### Prime-RL：整层编译的稳态收益与冷启动成本都需计入

一手来源：[PrimeIntellect-ai/prime-rl #4617](https://github.com/PrimeIntellect-ai/prime-rl/pull/4617)；[PrimeIntellect-ai/prime-rl #4687](https://github.com/PrimeIntellect-ai/prime-rl/pull/4687)；[PrimeIntellect-ai/prime-rl #4551](https://github.com/PrimeIntellect-ai/prime-rl/pull/4551)。

版本/原始日期（UTC）：#4617 2026-11-23T20:49:55Z / `519cd83d0bda`；#4687 2026-11-23T20:50:14Z / `953ed9a54422`；#4551 2026-11-23T02:59:32Z / `8662e3b88a6b`。发现日期：2026-11-24。

将 FLA、Mamba scan、带sink的FA4包装成custom op，避免activation checkpointing层内graph break；CP时把每packed row的FLA CP上下文移到层外，解锁Qwen3.5/Qwen3.8/GLM线性注意力的fullgraph。

作者8×H200、8层随机初始化代理、真实packed SFT、每卡8192token（CP2为4096）、全重算，整层编译稳态1.18–1.58倍，后续FLA CP补齐1.26–1.45倍；首步多30–135s，跨归约形状还会重编译，Qwen3.5的20步总时196对103s，CP补齐301对250s，短跑反而更慢。

同日另接入Inkling2 Small真实594B权重的trainer、vLLM插件及数学/SWE RL：公开配方六trainer节点加两推理节点，每节点8×H200；需要BF16转换、专家FP8和CPU卸载等完整条件，不能写8卡装下全部训练。建议长期线性混合架构训练考虑复现，先量化编译摊销；Inkling支持可作为开放RL复现起点。

### Automodel：FP32 LM-head logits 与不均匀词表蒸馏分片对齐

一手来源：[NVIDIA-NeMo/Automodel #4574](https://github.com/NVIDIA-NeMo/Automodel/pull/4574)；[NVIDIA-NeMo/Automodel #4746](https://github.com/NVIDIA-NeMo/Automodel/pull/4746)。

版本/原始日期（UTC）：#4574 2026-11-23T20:51:03Z / `082770550f01`；#4746 2026-11-23T09:25:43Z / `3f261a7f189b`。发现日期：2026-11-24。

独立 logits_dtype 让BF16/FP16操作数的LM-head以FP32累积并输出FP32，目标是对齐vLLM head_dtype=float32；原生模型统一head/project入口，pipeline shape和chunked loss跟随该dtype。HF模型类忽略并告警，head上LoRA在设置该选项时拒绝；FP32输出会增加logits驻留量，是否多通信取决于具体路径，不能只当无代价精度开关。

蒸馏bridge改用student DTensor的真实本地词表offset/size切teacher：7词/TP3的student为3/3/1，旧tensor_split误给3/2/2；全局词表不等则每rank预先拒绝。4进程CPU/Gloo实路径与KL/梯度参考通过，当前头GPU/NCCL及完整recipe仍未跑，已有零宽词表三维flatten限制未消失。建议低精度RL与异构teacher/student并行者阅读并做集成回归。

### Automodel DSpark 可从已发布 draft 权重热启动

一手来源：[NVIDIA-NeMo/Automodel #4725](https://github.com/NVIDIA-NeMo/Automodel/pull/4725)。

版本/原始日期（UTC）：#4725 2026-11-23T06:49:02Z / `1d02e4d99e5c`。发现日期：2026-11-24。

draft_weights_path 在FP8/compile/FSDP包装前加载draft，严格核key及target_layer_ids、mask_token_id；冻结embedding/head时保留target版本。DeepSeek-V4.1只打开含mtp.*的3/48个shard，MiniMax MXFP8权重与scale成对反量化到BF16。它是仅权重的新训练，优化器、RNG和step重新开始，正式resume优先。

CPU真实工件回环逐位一致；2×H100、6层截断MiniMax目标可继续训练，但发布draft原用3–59层特征，代理改1–5层，所以不能从其loss判断发布draft质量。空Adam moments配满学习率会出现loss跳变，guide建议warmup。建议目标微调后要重适配draft者考虑复现，注意热启动与断点恢复语义。

### Automodel 增加双向文本 Deciffusion 决策后训练配方

一手来源：[NVIDIA-NeMo/Automodel #4821](https://github.com/NVIDIA-NeMo/Automodel/pull/4821)。

版本/原始日期（UTC）：#4821 2026-11-23T23:25:57Z / `e5d03f7ef008`。发现日期：2026-11-24。

配方从固定 serving prompt 的单question/options构造训练行，以一次 prompt+[mask] 双向读取、冻结无bias diffusion head上的option codes完成选择；支持硬/软标签CE和加权KL，LoRA或全参数训练后导出完整权重，以验证准确率再CE选阶段，test不参与选择。

只接受bidirectional text NemotronLabsDiffusionModel；block-causal、vision、quantized在setup拒绝，超长行计数跳过而非截断。94个CPU单测与tiny checkpoint双GPU训练/导出重载给出概率一致性条件，未展示生产任务能力提升。建议需要低生成成本结构化决策的团队深入阅读契约与数据划分，再决定复现。

### VeOmni：逐算子位置选择与低精度路径统一配置

一手来源：[ByteDance-Seed/VeOmni #1472](https://github.com/ByteDance-Seed/VeOmni/pull/1472)。

版本/原始日期（UTC）：#1472 2026-11-23T02:36:47Z / `9f533f4613e8`。发现日期：2026-11-24。

model.ops_implementation 统一linear、gated MLP、MoE和KV rounding；OpsSelection+SiteOverride以re.fullmatch匹配site，后规则优先，同实现合并kwargs，切换实现重置kwargs，硬件与compile检查覆盖可达配置。TorchAO动态FP8改为普通nn.Linear上的functional路径，FP4 QAT仍模拟，原生FP4 emulated=false明确拒绝。

本PR只有DeepSeek-V4/Qwen3/Llama完成linear逐site接线；其他模型仅已有op族级选项，原DiT的Float8路径本轮未迁移。未给训练端性能/收敛实测。建议需要逐层保精度或可控消融者阅读配置迁移，继续观察模型覆盖而非直接假定全模型支持。

### ms-swift Ascend：Ling-3.0-tiny packing 与 GDN 归一化修复

一手来源：[modelscope/ms-swift #10536](https://github.com/modelscope/ms-swift/pull/10536)；[modelscope/ms-swift #10567](https://github.com/modelscope/ms-swift/pull/10567)。

版本/原始日期（UTC）：#10536 2026-11-23T07:26:49Z / `5ef0daa3d34d`；#10567 2026-11-23T07:20:54Z / `ec4f5b6bfe5d`。发现日期：2026-11-24。

Ling-3.0-tiny通过Transformers后端补齐RoPE、MLA mask、KDA短序列/左右padding和singleton短卷积梯度，支持FSDP2 LoRA sequential packing。Ascend A3、torch/torch_npu2.10、CANN9.1、Transformers4.57.1/5.14.1各84回归，两NPU每配置3步；旧非packing有8卡2000步对照，不能当本次packing长跑验证。

另一修复在MegatronAdaptor packed native GDN fallback缺少Q/K L2归一化时，以FP32、eps1e-6归一化后恢复输入dtype；刷新缓存alias且保留真实FLA分派。950DT四卡Qwen3.5-35B-A3B LoRA、TP2/EP2/ETP1、MTP1三步中real-FLA/no-FLA loss差约0.01，属于短smoke而非数值等价或性能证明。建议相应NPU配方优先回归正确性。

### vLLM 稀疏 MLA DCP ShardQ：长 prompt 首 token 时间降低

一手来源：[vllm-project/vllm #66027](https://github.com/vllm-project/vllm/pull/66027)。

版本/原始日期（UTC）：#66027 2026-11-23T09:05:50Z / `b26e0840134e`。发现日期：2026-11-24。

最后一组Kimi K3.1优化只影响decode context parallelism（DCP）。当TP=DCP时，把prefill KV/indexer页推入对称workspace，按query token分片处理完整head集合，prefill避开query all-gather、LSE合并和top-k合并；decode继续DCP。SM100 direct A2A LSE reduce为2次launch代替4次；workspace装不下自动回到旧DCP，需FP8 indexer K cache。

优先看同机新对照：官方Kimi-K3.1-Pro固定revision、同8×B300、TP8/DCP8、FP8 KV、无prefix/autotune、温度0、128输出上限，三条约108K prompt各三轮；main首token时间（TTFT）中位6.805s，合并候选开启4.913s，关ShardQ6.782s，预期code均9/9。较早8K输入/512输出服务对照在不同节点且每点一次，不能拿来增强同机结论。建议长prefill与decode混合负载者优先复现；不是所有DCP模型或硬件的通用开关。

### vLLM AWQ：向量化写回显著改善 BI 退化路径

一手来源：[vllm-project/vllm #63590](https://github.com/vllm-project/vllm/pull/63590)。

版本/原始日期（UTC）：#63590 2026-11-23T22:48:02Z / `5a215c27fd76`。发现日期：2026-11-24。

awq_dequantize 每线程原以8个2B store写出8个half，改为一个对齐16B uint4 store；反量化数学与输出逐位不变。H100/CUDA13、Llama3-8B各层L2-cold CUDA graph微测2.44–3.66倍，Nsight显示写指令/sector减少8倍，收益来自写事务而非溢出寄存器。

仅在采用该算子的路径有收益。Meta-Llama3.1-8B AWQ、单H100、batch-invariant（批次不变，BI）模式、1024输入/128输出，2次服务器启动各3轮：并发1 TPOT34.94→15.96ms，并发32吞吐761.7→1407.1tok/s，GSM8K及35组greedy/logprob相同。默认Machete路径约4.3ms且未变，**优化后的BI仍明显慢于默认**。建议需要BI或相关AWQ fallback者考虑复现，不据2倍数字宣称默认AWQ同幅提升。

### vLLM TRITON_MLA：布局和head块排布改善 kernel，端到端仍有限

一手来源：[vllm-project/vllm #69029](https://github.com/vllm-project/vllm/pull/69029)。

版本/原始日期（UTC）：#69029 2026-11-23T17:21:30Z / `5eca5e982bff`。发现日期：2026-11-24。

stage1将共享同一latent KV的head块排到grid首维，提高复用；K_Buffer用token×channel行优先tile合并加载，减少转置及shared bank冲突，stage2归约顺序不变。H100 DeepSeek-V2-Lite几何的kernel矩阵最高1.71倍；A100也有1q1s2k的0.95倍退化，不能说所有点都加速。

实际DeepSeek-V2-Lite-Chat只有16Q头，单head块无法体现重排收益；H100+FP8 KV的端到端多数点近似持平，80K输入点吞吐50.22→49.89，100K为34.92→35.03。基准脚本还需未包含在PR中的形状适配。建议多head块MLA用户复现真实大模型与自有负载，保留kernel收益和服务收益的区别。

### vLLM MXFP8 低延迟 EP 专家后端可用，但单机未胜默认

一手来源：[vllm-project/vllm #66456](https://github.com/vllm-project/vllm/pull/66456)。

版本/原始日期（UTC）：#66456 2026-11-23T23:16:16Z / `ab6ce7d23086`。发现日期：2026-11-24。

FlashInferCuteDSLBatchedExperts 新增W8A8 MXFP8 masked GEMM，将BF16行分派到DeepEP low-latency/NIXL EP后仅量化有效行；此前这两种batched分派在MXFP8无可选专家backend。要求SM100/103、FlashInfer0.7.3相关API、hidden/intermediate为128倍数、SiLU，无expert bias或SwiGLU clamp/scale，启动时拒绝不支持组合。

4×B200锁频、Qwen3-30B-A3B-Instruct-2507 MXFP8、128输入/512输出、三轮交替，并发16/64/256：DeepEP LL为默认allgather-reducescatter+TRT专家吞吐的0.92/0.96/0.94，NIXL为0.84/0.90/0.88；GSM8K相近。此处是新增可用组合，**不是已证实的单节点性能提升**。建议需要这些分派机制或更大拓扑者考虑复现，默认用户继续观察。

### vLLM HY4 与 Muse：新增量化/投机组合各有证据边界

一手来源：[vllm-project/vllm #68770](https://github.com/vllm-project/vllm/pull/68770)；[vllm-project/vllm #69274](https://github.com/vllm-project/vllm/pull/69274)；[vllm-project/vllm #64552](https://github.com/vllm-project/vllm/pull/64552)。

版本/原始日期（UTC）：#68770 2026-11-23T17:13:03Z / `e4ac61ae44d1`；#69274 2026-11-23T16:24:48Z / `1d9f8058bf8b`；#64552 2026-11-23T04:03:51Z / `306f47fcfb2b`。发现日期：2026-11-24。

HY4 Flash SM100优化作者列TP4、随机8192输入/512输出，并发1/32吞吐3.36/2.46倍；TP/TEP/DEP多配方均提升，但只写SM100且无充分重复信息，accuracy也有OCR835→830、GSM8K和AIME细微波动，建议按完整配置复现，不当无损承诺。

DFly将target层量化方案与draft条件分离，允许compressed-tensors NVFP4 target，只拒KV方案与transform；draft仍仅BF16/指定FP8。770B、4×B300、TP4，300题单轮NVFP4为289→288、accept≈2.955；AR基线并发不完全一致，作者未声称加速，图像/视频尚未验证。

Muse Spark允许把原BF16的第0/61层routed experts在线量化为NVFP4 per-token，同时保留post-expert RMSNorm，可减少约35GiB权重。单GB300 Station的收益同时依赖已合入CPU-offload修复，在线量化增加加载时间；不是把新量化checkpoint能力无条件推广给其他backend。建议容量受限和需draft配套者分项复现。

### vLLM 共享 PLE 与 Grace EGM：容量换取查表带宽

一手来源：[vllm-project/vllm #68684](https://github.com/vllm-project/vllm/pull/68684)；[vllm-project/vllm #68685](https://github.com/vllm-project/vllm/pull/68685)。

版本/原始日期（UTC）：#68684 2026-11-23T02:34:08Z / `842f74fa8dff`；#68685 2026-11-23T07:55:25Z / `dddd8b7dd3e7`。发现日期：2026-11-24。

多个独立引擎的FP8 PLE表复用Engram pool，每peer只加载自己行，NVLink可共享HBM，RDMA可共享HBM或pinned host；要求TP1/PP1、sync fetch、统一checkpoint/revision，变更权重需整组重启。GB200两个真实引擎各持23.84GiB表数据；短样本相同，但长上下文生成12–21token后有分歧，不能写端到端逐位一致。

另Grace Extended GPU Memory（EGM）可作为NVLink peer shard后端。DeepSeek-V4.1-Flash四GB200把总188.83GiB表移出HBM，使此前OOM的两组TP1/DP2+EP prefill引擎启动；16K输入、每引擎并发4、单输出token，p50/p95为553/719ms，对各引擎host-copy563/735ms。相对HBM，8192token查表p50 0.205→0.858ms、p95 0.210→1.563ms，容量并非免费。

EGM还依赖host-NUMA CUDA VMM、FABRIC handle、IMEX和足够Grace节点内存。建议查表大模型多副本部署者深入阅读所有权与启动校验，重点比较可服务容量和尾延迟。

### vLLM KV：失败回退、块几何、混合checkpoint三类关键修复

一手来源：[vllm-project/vllm #67957](https://github.com/vllm-project/vllm/pull/67957)；[vllm-project/vllm #59586](https://github.com/vllm-project/vllm/pull/59586)；[vllm-project/vllm #69048](https://github.com/vllm-project/vllm/pull/69048)；[vllm-project/vllm #67185](https://github.com/vllm-project/vllm/pull/67185)；[vllm-project/vllm #69433](https://github.com/vllm-project/vllm/pull/69433)。

版本/原始日期（UTC）：#67957 2026-11-23T19:48:14Z / `0fcd102378d9`；#59586 2026-11-23T15:28:33Z / `75d8770263d6`；#69048 2026-11-23T07:52:13Z / `18af054a09ab`；#67185 2026-11-24T00:36:18Z / `66e8914fc93c`；#69433 2026-11-23T17:24:49Z / `4f0ecfff6be6`。发现日期：2026-11-24。

MooncakeStore在recompute策略下新增按request的load_failed记录，GET失败后不再相信残留hit并重复远程加载，保证可落回本地重算。NIXL注册在kernel block大于全局NIXL block时，仅对token-major布局拆成dense页；head-major多头布局不能等价切页，明确拒绝并提示更小block或LBNHC，避免误传字节。

混合模型CPU partial-tail命中后，后续store从包含命中的chunk起算，而非向上取整跳过该chunk。HiSparse仅在容量不足时提前分配Hot、释放resident页供下一prefill/stream-resume chunk，保持其他request预留和水位保护。MLA continuation合并前clone旧prefix LSE，避免native merge覆写其他线程仍未读取的值；作者CI形状计时282.0→288.8µs，修复有少量复制开销。

这些均为已进主分支源码的正确性/活性修复，验证主要是定向回归和作者场景，不能泛称所有KV connector已稳定。建议涉及P/D、层级缓存、混合状态模型者优先核对命中边界和失败路径。

### vLLM PP 与 loader 生命周期：减少无用权重，公开KV阶段元数据

一手来源：[vllm-project/vllm #67349](https://github.com/vllm-project/vllm/pull/67349)；[vllm-project/vllm #67998](https://github.com/vllm-project/vllm/pull/67998)；[vllm-project/vllm #57011](https://github.com/vllm-project/vllm/pull/57011)；[vllm-project/vllm #69229](https://github.com/vllm-project/vllm/pull/69229)。

版本/原始日期（UTC）：#67349 2026-11-23T05:40:12Z / `a3309b31387d`；#67998 2026-11-23T04:47:42Z / `8399da37800d`；#57011 2026-11-23T13:51:50Z / `afb35f85e7e2`；#69229 2026-11-23T02:15:09Z / `f17a0e40d415`。发现日期：2026-11-24。

Qwen3-MoE/Qwen3.5/gpt-oss/Gemma3等只在实际使用的pipeline stage构造embedding/final norm/lm_head；tied embedding或draft复用仍需末stage副本，不能按所有stage一刀切删除。作者PP2已测模型每个省略模块的stage少0.58–1.08GiB，tied模型需PP≥3中间stage才省。

KVCacheConfig.pipeline_stages将planner推导的各stage块字节等元数据统一供scheduler/worker connector读取；SimpleCPUOffload据此得一致pool，删除专属all-reduce/RPC。自定义loader新增on_sleep/on_wake_up和reload_weights_inplace钩子，供P2P/RDMA权重服务在显存失效前停止、恢复后重建；默认no-op或旧reload。

私有worker-local tensor LoRA可用于verl式更新，但调用者仍负责同步和prefix-cache失效，不是新的跨EngineCore公共传输。建议自研权重分发/训推协同者阅读接口与时序，按自己的loader完成失败和热更验证。

### vLLM 量化修复：保留FP8容量、在整数域扣除零点

一手来源：[vllm-project/vllm #69446](https://github.com/vllm-project/vllm/pull/69446)；[vllm-project/vllm #69085](https://github.com/vllm-project/vllm/pull/69085)；[vllm-project/vllm #67413](https://github.com/vllm-project/vllm/pull/67413)。

版本/原始日期（UTC）：#69446 2026-11-23T20:02:42Z / `822bbd577ba3`；#69085 2026-11-23T19:49:54Z / `0f06eb7e4eee`；#67413 2026-11-23T14:26:35Z / `abad4cc48213`。发现日期：2026-11-24。

DSv4.1的SM90 TP1不再为MXFP8线性层常驻额外BF16副本；作者截断模型推算/实测此前可额外占约10.9GiB换约1% decode收益，TP2代价收益不同而保留旧优化。未在完整510GB checkpoint或8卡做accuracy，不能写所有DSv4.1回归均已消除。

ROCm asymmetric INT8把activation zero-point校正提前到精确int32累加器，再缩放到BF16，避免两项先舍入后相减造成消去误差。MI300X、Llama3-8B INT8 asym的完整GSM8K作者值ROCm10.0为0.720→0.744、10.1为0.713→0.748；不是新模型训练进步。moe_wna16同时遵循GPTQ checkpoint dynamic/modules_in_block排除规则，保留未量化MTP专家。

建议相关checkpoint用户优先阅读和回归；精度修复需保持权重、框架及评估一致。

### SGLang QSA：去同步与AMD融合稀疏decode分开评估

一手来源：[sgl-project/sglang #50622](https://github.com/sgl-project/sglang/pull/50622)；[sgl-project/sglang #45196](https://github.com/sgl-project/sglang/pull/45196)；[sgl-project/sglang #48649](https://github.com/sgl-project/sglang/pull/48649)。

版本/原始日期（UTC）：#50622 2026-11-24T00:09:35Z / `4593b484412b`；#45196 2026-11-23T13:44:53Z / `ac0380c7faf9`；#48649 2026-11-23T13:16:10Z / `633209130352`。发现日期：2026-11-24。

extend row map从CPU已知长度提供repeat_interleave的output_size，MTP末行种子改scatter-max，省去两处device→host读回；CPU长度不匹配仍fallback。GB300 metadata微测在排队100ms替代kernel时100.2→约0.75ms，空队列约0.8ms；它证明主机线程不再被前序prefill拖住，**不是整服务100倍**。

ROCm gfx950 QSA decode/verify接AITER FlyDSL融合kernel，直接从KV pool读selected rows并跨CTA分片，要求BF16 Q、BF16/FP8 KV、D256、GQA≤16；workspace固定到graph最大bucket。单MI355X Qwen3.8-Flash-Next Quark MXFP4+PLEFP8、MTP3/1/4、AgenticX c16，作者模拟accept2.32的性能臂总吞吐+3.3%、ITL均值-10.8%，真实accept的accuracy单独测且差异在小样本噪声内。

另ROCm indexer padded Q及每CTA四组的launch缩短长context串行扫描；25形状logit/topk逐位同，提交描述仍保留一项decode-selection测试失败记录，不能把full stack GSM8K96.8%归因该单PR。建议按完整AITER版本与真实接受率复现。

### SGLang Muse 专家 TP：低并发略快，高并发有退化

一手来源：[sgl-project/sglang #50065](https://github.com/sgl-project/sglang/pull/50065)。

版本/原始日期（UTC）：#50065 2026-11-24T00:08:51Z / `1b85fbdb4ea3`。发现日期：2026-11-24。

NVFP4 Muse从只支持EP=TP扩展到EP1的专家tensor parallel：每rank持全专家的一部分intermediate，先对每token/expert行all-reduce再做post-expert RMSNorm与加权top-k，不能用普通先合并再all-reduce顺序。SM10x、无DP attention、分片128对齐，BF16端层还需TRT专家backend；不支持则拒绝或使用既有finalize fallback。

B300、DFlash、ShareGPT输出1024：TP8/EP1相对TP8/EP8并发1为+2.8%，并发32为-12.6%；TP4分别+2.9%/-5.8%。作者仍建议EP=TP，新增配置不能当更优默认。尤其--ep-size默认1，旧只指定TP的命令会由拒绝变成启动此新路径。建议Muse部署者读配置变化并按负载复现。

### SGLang 自调优持久化迁移仅覆盖单rank

一手来源：[sgl-project/sglang #51512](https://github.com/sgl-project/sglang/pull/51512)。

版本/原始日期（UTC）：#51512 2026-11-24T00:05:38Z / `e9167b2869e1`。发现日期：2026-11-24。

单rank从FlashInfer v1整JSON切到managed v2逐条原子持久化：crash保留已调条目，缺失/坏/过期只重调单项，环境版本/GPU由FlashInfer管理失效；target与draft共享process store且warmup前挂载，消除draft先用默认后用tuned的冷命中差。

只在max_world_size=1、SM90/SM100族、非MegaMoE、启用reuse与tuning时生效，TP/PP/EP/多节点仍旧JSON。作者H100/B100/GB200、FlashInfer0.7.4等验证重启零profiling和并发store有效；首次升级要重调。建议单卡多副本与有draft的服务考虑采用；不把v2选择算法或多rank一致性写成此次新增。

### SGLang P/D与外部KV：释放要晚于所有stage和请求消费

一手来源：[sgl-project/sglang #47618](https://github.com/sgl-project/sglang/pull/47618)；[sgl-project/sglang #52510](https://github.com/sgl-project/sglang/pull/52510)；[sgl-project/sglang #49463](https://github.com/sgl-project/sglang/pull/49463)；[sgl-project/sglang #49467](https://github.com/sgl-project/sglang/pull/49467)。

版本/原始日期（UTC）：#47618 2026-11-24T00:00:38Z / `c64f2256d583`；#52510 2026-11-23T22:10:08Z / `e1dbd0ed451a`；#49463 2026-11-23T14:35:44Z / `ac0243723b0d`；#49467 2026-11-23T14:33:26Z / `576e00a5afe7`。发现日期：2026-11-24。

pipeline prefill由PP0汇总所有stage transfer结论再统一释放；任一失败保留失败，但另一stage尚在transfer或staging cleanup时不能提前释放其KV/metadata。单stage也本地poll以结算trace副作用。195定向测试及Qwen/Mooncake压力场景支持该修复，吞吐/TTFT只小幅变化。

finished prefill先flush release再交observer，避免SpecForge prefill-only capture要求defer已释放request时崩scheduler。linker将slot location查询移到forward fence前，允许下一轮CPU/GPU重叠；published prefetch KV纳入request session，避免尚排队的已预取prefix缺session引用而被优先驱逐、重复计算。这些是不同生命周期边界，不能只检查传输已提交。

建议自研capture plugin、外部KV或PP P/D用户优先回归取消、部分失败、延迟消费和drain；未做本地设备复测。

### SGLang draft megakernel：兼容性拒绝可回退，target仍严格校验

一手来源：[sgl-project/sglang #52495](https://github.com/sgl-project/sglang/pull/52495)；[sgl-project/sglang #52438](https://github.com/sgl-project/sglang/pull/52438)；[sgl-project/sglang #51138](https://github.com/sgl-project/sglang/pull/51138)。

版本/原始日期（UTC）：#52495 2026-11-23T21:45:05Z / `1f713857aa37`；#52438 2026-11-23T04:42:36Z / `2544c9ac34be`；#51138 2026-11-23T13:15:40Z / `b119180763ec`。发现日期：2026-11-24。

DFLASH plugin把worker自己的round phases绑定给组合runtime，使外部runtime可以调用真实draft过程；不满足draft spec的TP/SM等条件时，各TP rank一致decline并保留普通draft forward，而target spec不满足仍报错。#52438描述中的GPU整服务器检查仍待完成，所以只当已合入启动逻辑，不当所有fallback实跑认证。

独立router正确性修复把softcap的x/c夹到[-20,20]后计算exp表达式，避免大logit溢出inf/inf产生NaN与错误topk。最终实现是clamp，**不是早期sign/exp(-abs)形式**；L4对±1500回归与FP64参考吻合。建议megakernel集成与高幅度router数值场景阅读相应边界。

### Transformers：MoE并行迁移、只读cache与投机零概率修复

一手来源：[huggingface/transformers #50897](https://github.com/huggingface/transformers/pull/50897)；[huggingface/transformers #50996](https://github.com/huggingface/transformers/pull/50996)；[huggingface/transformers #50993](https://github.com/huggingface/transformers/pull/50993)；[huggingface/transformers #51112](https://github.com/huggingface/transformers/pull/51112)；[huggingface/transformers #48989](https://github.com/huggingface/transformers/pull/48989)。

版本/原始日期（UTC）：#50897 2026-11-23T18:56:13Z / `52bac5d0e0da`；#50996 2026-11-23T05:48:58Z / `1097d1570e62`；#50993 2026-11-23T11:27:30Z / `7ad60adc9e33`；#51112 2026-11-23T10:17:51Z / `bc7bd0030206`；#48989 2026-11-23T12:53:55Z / `2f6f6912aff2`。发现日期：2026-11-24。

MoE专家TP方案转向EP并弃用旧MoE TP布局；同时恢复显式tp_plan={}表达的仅EP装载，不能把所有模块TP一概当不可用。cache layer update新增write=False，返回本次应读state而不写回，Dory末loop以此替代读raw keys，兼容static cache。

投机采样accept比较由<=改<，防止uniform恰为0时接受target概率为0的draft token；禁词、top-k/p等置零都受此边界影响。作者强制零随机数和万轮flake测试定位，未改正常性能路径。Nemotron-H Mamba2还修复runtime time_step_limit被初始化floor覆盖，同权重H100旧栈KL0.033659→0.003206，当前头只CPU回归；不能写所有跨框架差异已清零。

建议维护MoE自定义plan、循环模型cache和投机生成者查看迁移/回归；这些合入在默认分支，不等于已包含于此前v5.24.0发行。

### PyTorch FSDP2：自定义 all-gather 布局由 FSDP 管存储

一手来源：[pytorch/pytorch #199726](https://github.com/pytorch/pytorch/commit/25acc7329a9d9266fc9dfcb2dcf218fdf9ad7cab)。

版本/原始日期（UTC）：#199726 2026-11-23T22:51:39Z / `25acc7329a9d`。发现日期：2026-11-24。

AllGatherLayout 以prepare/prepare_output/copy_in/finalize_outputs契约支持backend自选打包与输出布局，默认prepare_output可返回None回退。输出tensor要view持久buffer，由FSDP核形状/dtype/所有权后接管；reshard释放storage、后续按记录重分配，避免自定义layout让reshard_after_forward失去省显存效果。跨copy-in/compute流的buffer要record_stream。

这改动涉及私有接口：AllGatherResult新增字段、函数签名和旧_all_gather_output_fn移除；替换私有hook的扩展需迁移set_all_gather_layout。PR为ghstack closed/merged=false，但相应变更已出现在默认分支精确commit，本条按commit核验，不冒称GitHub merge事件或稳定版能力。建议FSDP通信backend团队深入阅读生命周期与拒绝条件，未见端到端性能证据。

### PyTorch TokenSwitchMoRI：MI355X 专家通信接线与明确退路

一手来源：[pytorch/pytorch #201225](https://github.com/pytorch/pytorch/commit/fd770cc55b3374aaecd2156e5dc98bffb626d805)。

版本/原始日期（UTC）：#201225 2026-11-23T22:08:39Z / `fd770cc55b33`。发现日期：2026-11-24。

新增MoRI EP v2+FlyDSL intranode TokenSwitch backend，支持flat与expert-major；首次routing确定布局、后续重放，反向同时给token和top-k weight梯度。为绕过MoRI handle复用缓冲，复制dest map和recv count；同一op连续dispatch/combine时额外all-reduce fence，交替时不加，避免快peer覆盖。

作者MI355X、MoRI e73ccc539源码、FlyDSL0.5.0、2/4rank各38测试通过；MI300X默认几何会在构造失败，需要尚待发行的调优修复。当前CI镜像无依赖会skip，未给吞吐优势。按默认分支commit记录（PR ghstack不显示merged），建议AMD MoE训练者考虑复现并核依赖/同op连发边界。

### PyTorch 数值转换与 Inductor 索引复用修复

一手来源：[pytorch/pytorch #204075](https://github.com/pytorch/pytorch/commit/d2e0c99a0338b5761780b36f843064b9f3b5d51b)；[pytorch/pytorch #204208](https://github.com/pytorch/pytorch/commit/29ada738653956ff90dbadd397b98e9e72007912)；[pytorch/pytorch #205499](https://github.com/pytorch/pytorch/commit/00bc9df982a1476f275d80cb3c6e69967f96defd)；[pytorch/pytorch #194414](https://github.com/pytorch/pytorch/commit/69197b1962aaa286767d9ab30cafedd4ff810d82)。

版本/原始日期（UTC）：#204075 2026-11-23T12:21:33Z / `d2e0c99a0338`；#204208 2026-11-23T12:21:33Z / `29ada7386539`；#205499 2026-11-23T18:22:37Z / `00bc9df982a1`；#194414 2026-11-23T21:55:27Z / `69197b1962aa`。发现日期：2026-11-24。

float64→FP16/BF16/FP8改用round-to-odd的中间FP32，消除双次nearest舍入；只有历史错误边界等会改变输出，旧header内联/AOT包需重建。E8M0另修float32 subnormal在最小两档之间错误上舍入，强调E8M0无零且tie规则不同，扩展也在重编译后生效。

Inductor不再把同kernel用于indirect-indexing（含从该buffer派生）的buffer原地当输出，避免跨warp复制lane在索引覆写后读到任意地址；代价是相应kernel多一个输出分配。已见MI300X推荐模型故障，但触发取决于layout/warps，不能归因AMD硬件本身。Host TMA epilogue descriptors则移到host生成，operand和epilogue同buffer需不同shape时仍fallback。

均按已读默认分支commit而非ghstack merge标志记录；建议数值回归、AOT/自研layout用户审阅，不把实现正确性修复包装成模型质量提升。

### LMCache：大 KV 的 int64 地址与 inline 分页传输

一手来源：[LMCache/LMCache #6872](https://github.com/LMCache/LMCache/pull/6872)；[LMCache/LMCache #6756](https://github.com/LMCache/LMCache/pull/6756)。

版本/原始日期（UTC）：#6872 2026-11-23T23:10:32Z / `b27f6bd9e2c8`；#6756 2026-11-23T03:20:53Z / `2a5d026c9959`。发现日期：2026-11-24。

CacheBlend per-token CUDA传输offset的int32乘积在pool超2^31个scalar时溢出，现逐乘积先提升int64；H200的Qwen3.8-27B约62K CacheBlend场景由illegal address/等待不终止变50请求全成功。block粒度store/prefix-hit不是该bug；SYCL同型代码无生产caller，本PR未修。

默认无SHM pool的engine-driven inline KV以前单消息，撞gRPC2GiB/ZMQ msgpack4GiB上限。超过128MiB现按chunk range分页，retrieve保留未读位置，失败/COMMIT清理剩余锁并记录failed；CPU真实server的2.03/4.25GiB checksum测试通过。新旧ZMQ不能混部，gRPC仅旧worker→新server兼容；不是任意混版本零停机升级。建议长context/CPU缓存和CacheBlend者优先回归。

### LMCache 注册 profile 与 RawBlock I/O：契约先于性能

一手来源：[LMCache/LMCache #6248](https://github.com/LMCache/LMCache/pull/6248)；[LMCache/LMCache #5450](https://github.com/LMCache/LMCache/pull/5450)；[LMCache/LMCache #6523](https://github.com/LMCache/LMCache/pull/6523)。

版本/原始日期（UTC）：#6248 2026-11-23T16:40:19Z / `e23b1f8df806`；#5450 2026-11-23T03:06:45Z / `4d52596cf4bc`；#6523 2026-11-23T07:26:44Z / `9c98203ffb0f`。发现日期：2026-11-24。

KVEventProfile(profile_id,block_size)在注册时核world/block整除、full-attention地址空间及active registration一致性；typed rejection在gRPC/ZMQ有各自code与超时面。生产caller本轮仍传None，首个profile发送/核echo属于后续PR，不能写router事件端到端已启用。ZMQ请求帧数改变需同时升级两端；full-model/Dynamo/llm-d E2E未跑。

RawBlock对AES-GCM等非块对齐payload，在L1物理分配有足够padding时直接扩展aligned view并清零padding，去bounce copy；余量不足仍fallback，POSIX不变。作者Qwen3-8B两轮三请求，uring_cmd TTFT-20.4%、io_uring仅-0.8%在噪声内，硬件条件未完整列，不作通用承诺。

bench l2同时修复迟到旧round completion提前完成当前round的计账错误，只接受本round task IDs；这是基准可信度修复，建议先升级再比较L2吞吐。

### TensorRT-LLM GDN MTP：verify 重放与接受后 commit 分离

一手来源：[NVIDIA/TensorRT-LLM #19586](https://github.com/NVIDIA/TensorRT-LLM/pull/19586)；[NVIDIA/TensorRT-LLM #21207](https://github.com/NVIDIA/TensorRT-LLM/pull/21207)；[NVIDIA/TensorRT-LLM #22596](https://github.com/NVIDIA/TensorRT-LLM/pull/22596)。

版本/原始日期（UTC）：#19586 2026-11-23T12:03:46Z / `5aa1250da135`；#21207 2026-11-23T17:10:46Z / `1e87a1d027a4`；#22596 2026-11-23T09:29:46Z / `7bdea49f6b59`。发现日期：2026-11-24。

PyTorch GDN mixer用CuTe DSL replay一次/层融合Conv4与accepted history+新候选递推，验证时不改正式Conv/recurrent state；接受后一次/步跨所有本地层commit，16行history满时折叠checkpoint。history按request slot迁移且计入V2 KV预算，P/D之间不传。只选SM100/103、V2、BF16状态、D128、4tap、固定线性draft≤8等组合，其他回非replay，modeling_v2有自己的机制。

B200、Qwen3.5-397B形状、45层CUDA graph、L2 flush、accept≈3.62：T4/B64的每层区段81.17→50.74µs，而B4为7.83→8.59退化；不是全模型吞吐。另默认关闭TRTLLM_GDN_FUSED_CONV补其它prefill形状，FORCE_DETERMINISTIC禁用，小eager还多约9µs launch开销。

Mamba2非MTP单token把对齐x/B/C view直接交FlashInfer，省三copy；GB200作者eager decoder GPU区段降2.2–6.5%，非端到端，主分支pin的FlashInfer0.7.3未实跑、graph逐位未核。建议混合架构投机者深入阅读选择矩阵并复现真实batch。

### TensorRT-LLM KV：复用更早可选checkpoint，SWA scratch看pool划分

一手来源：[NVIDIA/TensorRT-LLM #22352](https://github.com/NVIDIA/TensorRT-LLM/pull/22352)；[NVIDIA/TensorRT-LLM #20971](https://github.com/NVIDIA/TensorRT-LLM/pull/20971)；[NVIDIA/TensorRT-LLM #19898](https://github.com/NVIDIA/TensorRT-LLM/pull/19898)。

版本/原始日期（UTC）：#22352 2026-11-23T06:06:14Z / `8727fd0b19e2`；#20971 2026-11-23T02:25:00Z / `22c18f587d4b`；#19898 2026-11-23T10:12:03Z / `ffb55507b433`。发现日期：2026-11-24。

DeepSeek-V4.1 CSA2的Global required prefix仍保留P，Encoder OPTIONAL group可找回[P-R,P]内共同较早端点S，仅重放S到P；支持partial-page，但奇数P的ratio2 compressor要回P-1重放一行。cold recovery补齐writer未算的页头，pending recovery在post-forward resize完成，避免读未写状态；需要新C++ binding，未单独测速。

Gemma3/4文本+TRT attention+V2默认开启SWA scratch跨层复用，改正page-table extent；多模态和外部shared-KV MTP默认关闭，FlashInfer拒显式启用。在B200受限KV容量实验，未钉pool ratio几乎持平甚至-1.2%，按8K预期长度钉ratio后才有6.9–59.6%收益，不能把最大值当普适。

multimodal scheduler wrapper还补传AsyncTransferManager与prefill-demote阈值，消除在途send被误判KV deadlock及长prefill降级失效。建议对应模型/多模态P/D用户优先核正确性与容量。

### TensorRT-LLM 与 FA4：TMEM 完成与 cluster 作用域需要显式同步

一手来源：[NVIDIA/TensorRT-LLM #21851](https://github.com/NVIDIA/TensorRT-LLM/pull/21851)；[NVIDIA/TensorRT-LLM #21847](https://github.com/NVIDIA/TensorRT-LLM/pull/21847)。

版本/原始日期（UTC）：#21851 2026-11-23T08:09:09Z / `d9d5e18bdfa9`；#21847 2026-11-23T08:08:59Z / `5f70adef14ad`。发现日期：2026-11-24。

paged MQA logits的TMEM allocator warp原先可能在别的math warp还未完成tcgen05.ld时dealloc；新代码先等每thread load完成，再以256math-thread barrier汇合，之后释放。B200注入stall+同CTA重用marker时旧FP8/FP4均20/20错、修复0/20，跨stream重用旧1/20错；这是受控故障注入，未给自然服务故障发生率。

GLM-5.2四CTA routed/shared-expert的peer bulk copy完成在cluster作用域，原FC1 acquire只CTA不足；改relaxed cluster轮询+shared acquire fence，并统一FC2。该完整专家核只编译sm103a和源码测试，未硬件执行；旁证mailbox实验不能替代完整核验证。建议Blackwell自研算子者深入阅读内存模型，区分实际读写损坏复现与规范性修复。

### FA4 稀疏 MLA backward：读取 accumulator 后先等异步 load

一手来源：[Dao-AILab/flash-attention #3208](https://github.com/Dao-AILab/flash-attention/pull/3208)。

版本/原始日期（UTC）：#3208 2026-11-23T06:17:56Z / `b038b4974f26`。发现日期：2026-11-24。

dKGemmKernel在tcgen05.ld读取每32列subtile后增加fence_view_async_tmem_load，再释放accumulator stage给MMA复用；使用载入寄存器本身不构成TMEM访问完成契约。作者B200/ptxas12.9当前SASS调度原本碰巧保证正确，改后345相关测试343通过、2个sink梯度失败两侧相同，dK逐位同，时间-0.06%至+0.3%。建议实现异步TMEM producer/consumer者阅读这一行修复，不能把规范补齐描述为已发现普遍自然数值错误。

### TensorRT-LLM Rubin fused MXFP8 MoE 与对称pool基础设施

一手来源：[NVIDIA/TensorRT-LLM #21589](https://github.com/NVIDIA/TensorRT-LLM/pull/21589)；[NVIDIA/TensorRT-LLM #20308](https://github.com/NVIDIA/TensorRT-LLM/pull/20308)。

版本/原始日期（UTC）：#21589 2026-11-23T07:57:41Z / `d7d81834c8ad`；#20308 2026-11-23T19:08:35Z / `8a82e1f400d5`。发现日期：2026-11-24。

SM107 CUTEDSL新增持久fused FC12 leaf，一kernel融合gather、FC1、SwiGLU、MXFP8重量化、FC2和top-k finalize，替代该格式退回Cutlass。仅BF16输入、指定blockscale、hidden512/intermediate128对齐、fused finalize，无EPLB/expert bias；缺依赖或未对齐走Cutlass。与权重partial-refit interleave一起接线，不能只看一段kernel。

PR堆栈拆分后本条仅其own最后提交；Blackwell port、decode优化和workspace修复在其他PR，未核它们本轮合入，**不写成本次SM100/103完整支持**。先前4×SM107整栈117测试通过，当前拆分头本地H100会skip Rubin，不能等价看待。

VisualGen的SymmMemPool提供rank一致slot/signal/失败协商与collective teardown，但树内尚无consumer；copy-engine all-gather/reduce-scatter仍后续，不能写已服务加速。建议源码团队深入阅读，部署继续观察当前头运行与依赖资格。

### FlashInfer SM107 小 M MXFP8：精确 allowlist，真实形状加合成值

一手来源：[flashinfer-ai/flashinfer #9588](https://github.com/flashinfer-ai/flashinfer/pull/9588)。

版本/原始日期（UTC）：#9588 2026-11-23T20:48:27Z / `e2d163fb3d46`。发现日期：2026-11-24。

specialized backend在SM107将E4M3+UE8M0一次缩放提升BF16，再用warp MMA FP32累加，按shape族编译launch；极小scale有精确反量化分支。auto仅对34个精确(M,N,K) allowlist和dtype/layout/device条件优先选用，并参与autotune；显式specialized不支持则报错，不能静默换backend。

真实serving trace只提供shape与权重分布，**正确性tensor值是seeded合成数据，不是模型权重/activation**。Rubin silicon CUDA-graph作者计量allowlist相对0.7.4/当前autotuned main加权几何均1.4621/1.4324，整个43形状集1.3563/1.2792；这是kernel集合且筛选过受益形状，不是LLM服务1.46倍。最终补齐跨CPU/跨GPU operand拒绝，作者101native测试0skip，official CI资格与真实模型E2E仍需区分。建议SM107低batch团队复现同shape并加真实值与端到端验证。

### FlashInfer TF32 router GEMM 与边界mask优化

一手来源：[flashinfer-ai/flashinfer #9694](https://github.com/flashinfer-ai/flashinfer/pull/9694)；[flashinfer-ai/flashinfer #9846](https://github.com/flashinfer-ai/flashinfer/pull/9846)；[flashinfer-ai/flashinfer #8815](https://github.com/flashinfer-ai/flashinfer/pull/8815)。

版本/原始日期（UTC）：#9694 2026-11-23T20:48:08Z / `e4f8d55957c2`；#9846 2026-11-23T06:10:15Z / `7b1f4400b112`；#8815 2026-11-23T17:26:41Z / `d6ab9eb1a43d`。发现日期：2026-11-24。

新增FP32输入/输出、TF32 MMA累加的mm_tf32，SM100/103/107的low-M CuTe DSL可避免cuBLAS splitK外部reduce；auto按tile占SM比例/调优选，cublas依赖caller已允许TF32。SM107 Muse路由N256/K8192/M1、60个cold-weight graph calls：cuBLAS5.12µs、default3.61、+PDL2.62，M512无PDL则7.44对6.89反慢，非全shape胜。

Prims-TS把可见K范围缓存到task/SMEM，边界tile才mask；B300作者33案例几何均-2.98%，若干window形状-20%左右，最差+0.9%，70个MLA几何均仅-0.14%。D256特例因ptxas寄存器复制退化而保持每tile重建。

NVFP4 per-token线性scale无排列时new_empty省一次zero-fill，swizzled/permuted仍清零。GB300 MiniMax-M3.1 TP4+DSparkK7图时间小batch省约1.2%步，高batch与router重叠后无收益。建议分别复现router、mask及量化critical path，不能累乘这些微优化数字。

### Triton 与 Helion：跨 CTA scan、DSMEM 与 verify megakernel

一手来源：[triton-lang/triton #12924](https://github.com/triton-lang/triton/pull/12924)。

版本/原始日期（UTC）：#12924 2026-11-23T22:08:08Z / `856833f7901e`。发现日期：2026-11-24。

Triton scan按逻辑segment索引而非CTA rank寻找前缀，支持跨CTA且正反方向都保持scan次序；通过现有layout conversion/DSMEM交换segment总和，并将scratch读标记为跨CTA以在重用前同步。寄存器carry按最后consumer立即应用以缩短生命周期。未给本PR的LLM端到端量化收益，建议scan/线性递推kernel开发者深入阅读正确性与同步生成。

### Helion：DSMEM split-K 与跨rank in-band传输增强，仍有驻留限制

一手来源：[pytorch/helion #4570](https://github.com/pytorch/helion/pull/4570)；[pytorch/helion #4569](https://github.com/pytorch/helion/pull/4569)；[pytorch/helion #4572](https://github.com/pytorch/helion/pull/4572)。

版本/原始日期（UTC）：#4570 2026-11-23T05:21:16Z / `34db444c647a`；#4569 2026-11-23T04:42:12Z / `ec578eae5d8b`；#4572 2026-11-23T19:27:56Z / `6b38dbd68bf4`。发现日期：2026-11-24。

默认关闭的cross_loop_dsmem让同cluster的splitK producer以st.async将partial发到rank0 shared，再固定顺序归约，减少global partial+counter；限制cluster2–8、slab等，不满足回counter。runtime把compile/patch/launch/restore串在每kernel锁内，并按TMEM列驻留填shared防单launch死锁；**不同stream两waiting launch仍可能共享SM超列容量**。

in-band新增Lamport三slot NaN-key编码及gated push/同gate peer读，减少epoch/tag和无用数据；特定NaN符号语义有差，不能称任意比特完全一致。另新增DeepSeek-V3 NVFP4 TP1/4、Llama3.2-1B、Qwen3-8B、Muse TP4/8预调verify megakernel；PR未披露服务性能或完整模型资格。建议编译器/megakernel团队读，生产采用继续观察。

### llama.cpp：GDN replay 减少投机状态快照，Metal 可读 NVFP4

一手来源：[ggml-org/llama.cpp #31669](https://github.com/ggml-org/llama.cpp/pull/31669)；[ggml-org/llama.cpp #32825](https://github.com/ggml-org/llama.cpp/pull/32825)；[ggml-org/llama.cpp #32828](https://github.com/ggml-org/llama.cpp/pull/32828)。

版本/原始日期（UTC）：#31669 2026-11-23T23:06:04Z / `d35dc4d9e682`；#32825 2026-11-23T19:02:32Z / `51831d47d0a4`；#32828 2026-11-23T20:50:31Z / `c212d25ad34f`。发现日期：2026-11-24。

GDN从为每投机位置保全量快照改成accepted-history replay，作者RTX PRO6000/Qwen3.8-27B Q8_0、n_rs_seq7每序列1197→176.8MiB；32slot约37.4→5.5GiB。DFlash2单seq性能约持平，server4/16/32slot为+2.1/+7.9/+3.8%，batch verify的更大收益不是服务同幅；SYCL尚未完成后端接入。

Metal新增NVFP4 dequant/matvec/MM/row/copy支持，避免该GGUF类型退CPU，但PR描述未给Apple GPU的数值或速度实测，所列DGX Spark不验证Metal。prompt cache另修target恢复成功但draft失败时提前清理cache entry，改为全restore成功后再erase，保留可重试状态。建议混合模型投机用户考虑复现容量，Apple部署先核实际硬件资格。

### llama.cpp 量化可按文件或 bpw 预算分配逐tensor位宽

一手来源：[ggml-org/llama.cpp #15550](https://github.com/ggml-org/llama.cpp/pull/15550)。

版本/原始日期（UTC）：#15550 2026-11-23T17:06:27Z / `30ec35ae2e27`。发现日期：2026-11-24。

--target-size/--target-bpw通过imatrix采样估计各tensor量化误差，剪Pareto支配格式，再以拉格朗日预算分配混合类型；--state-file保存估计以便续跑/复用。目标是相同字节预算下改善分配，而非固定Q4_K_M标签比较。

作者34模型238测例中93% KLD匹配或更好，但优化的是采样局部误差代理，非直接最优perplexity/能力；首次通常慢5–10倍，需新imatrix包含activation统计，不同kernel格式混合也可能影响速度。建议端侧精确容量受限者考虑复现并同时测实际质量/时延，不把93%当所有任务赢。

### Dynamo：worker发布负载事件，饱和基准修正计账

一手来源：[ai-dynamo/dynamo #18903](https://github.com/ai-dynamo/dynamo/pull/18903)；[ai-dynamo/dynamo #19335](https://github.com/ai-dynamo/dynamo/pull/19335)。

版本/原始日期（UTC）：#18903 2026-11-23T22:01:37Z / `69ed0387786c`；#19335 2026-11-23T19:02:12Z / `a2d8eb5e001e`。发现日期：2026-11-24。

Ray Serve选择router不在response路径，request在router退出后仍可运行；新增worker直接PUB prefill-complete/free给每个selector，非阻塞发布、origin区分且不重广播，早于booking的事件最多暂存30s/65536条。它解决只由booking router转发导致其他副本负载滞留，不等于PUB/SUB获得持久可靠投递。

selection_replay_bench按wait结束时空闲admission占比计harness_wait_share，避免512slot已满时把issuer等待误判bench饱和。此4行只改计量，作者可计入场景的req/s基本不变；有排队时engine时序仍可能控制速率而guard没报。建议分布式路由/性能平台团队阅读事件所有者和基准口径，勿报告router吞吐因计账改动而增长。

### llm-d：未知延迟排后，非 P/D 路径不找 prefill

一手来源：[llm-d/llm-d-router #4200](https://github.com/llm-d/llm-d-router/pull/4200)；[llm-d/llm-d-router #4335](https://github.com/llm-d/llm-d-router/pull/4335)；[llm-d/llm-d-router #4283](https://github.com/llm-d/llm-d-router/pull/4283)。

版本/原始日期（UTC）：#4200 2026-11-23T09:02:49Z / `5b08e40f8d1c`；#4335 2026-11-23T09:04:53Z / `f8f47d9372e1`；#4283 2026-11-23T09:46:35Z / `51a947ae2323`。发现日期：2026-11-24。

predicted-latency producer只给已scrape过endpoint生成预测，避免零值metrics被当真实观察；scorer将部分候选中无prediction的排后，全无时保留既有fallback，配套修复要一起看。单测验证策略，不是推理模型更准确的实测。

P/D handler与sidecar共享HTTP/gRPC route集合，embeddings/images/speech等未支持P/D的路径直接decode-only，修复prefill池为空时本不该依赖prefill的503；实际chat/Generate仍要求prefill，不能把全部请求悄悄降成decode-only。kind/Istio及vLLM CPU gRPC验证路由与计数，SGLang gRPC是否真正P/D仍取决sidecar connector配置。建议多API入口的P/D部署优先核路径与监控。

### vLLM-Omni：输入时钟、打断fence和Qwen音频MTP图的增量

一手来源：[vllm-project/vllm-omni #10788](https://github.com/vllm-project/vllm-omni/pull/10788)；[vllm-project/vllm-omni #11032](https://github.com/vllm-project/vllm-omni/pull/11032)；[vllm-project/vllm-omni #11031](https://github.com/vllm-project/vllm-omni/pull/11031)；[vllm-project/vllm-omni #10443](https://github.com/vllm-project/vllm-omni/pull/10443)；[vllm-project/vllm-omni #10871](https://github.com/vllm-project/vllm-omni/pull/10871)；[vllm-project/vllm-omni #10616](https://github.com/vllm-project/vllm-omni/pull/10616)。

版本/原始日期（UTC）：#10788 2026-11-23T18:55:57Z / `7d0f18e7f2c9`；#11032 2026-11-23T18:21:19Z / `5d6f661ac784`；#11031 2026-11-23T17:11:22Z / `2280fd690cb7`；#10443 2026-11-23T10:04:35Z / `f79cc5a05aa2`；#10871 2026-11-23T05:32:17Z / `085d7dba33ec`；#10616 2026-11-23T21:09:25Z / `597372343ff4`。发现日期：2026-11-24。

MiniCPM-o4.5成为首个input-clock音频session opt-in，以Talker segment结束加Code2Wav标记flush完成精确ack，不靠timeout；双工append fence取不低于已接受turn，避免用户插话后所有后续音频被判stale。Qwen3-Omni在MRv2接Thinker DFlash和可续speech，JoyAI四stage ASR profile补connectorless角色/中间token/prewarm以启用async_chunk。

Qwen3-Omni residual-codebook采样保持每请求RNG draw形状/次序，并恢复whole-MTP graph/per-row generator capability；该#11031提供结构CPU测试，但作者Mac缺PyTorch/vLLM环境、未能本地执行pytest，描述将CUDA/ROCm READY作为运行资格，不能据先前3.5倍回退声称最终头已恢复相同倍数。

MOSS长codec分块按request轮转batch。910B3、vLLM0.33移植、c10、三轮原/新/原共360请求：RTF3.050→2.583，首包6.324→4.780s；仍非实时。跨batch wave数值未过1e-3筛选，11组单人盲听无差不足以宣称音质普遍无损，当前base完整套件也未认证。建议双工系统优先读生命周期，性能/音质按最终环境复现。

### miles-diffusion：AV 联合动作训练接通，音频奖励仍缺

一手来源：[radixark/miles_diffusion #406](https://github.com/radixark/miles_diffusion/pull/406)；[radixark/miles_diffusion #414](https://github.com/radixark/miles_diffusion/pull/414)；[radixark/miles_diffusion #415](https://github.com/radixark/miles_diffusion/pull/415)；[radixark/miles_diffusion #412](https://github.com/radixark/miles_diffusion/pull/412)；[radixark/miles_diffusion #413](https://github.com/radixark/miles_diffusion/pull/413)；[radixark/miles_diffusion #389](https://github.com/radixark/miles_diffusion/pull/389)。

版本/原始日期（UTC）：#406 2026-11-23T07:15:46Z / `705a461731c7`；#414 2026-11-23T07:21:43Z / `52aeb5fc84a6`；#415 2026-11-23T07:22:00Z / `0a942cc6d90d`；#412 2026-11-23T07:22:32Z / `be7a08dcb74f`；#413 2026-11-23T07:22:48Z / `e6129d0fe8cc`；#389 2026-11-23T23:30:27Z / `4cb1a2909ce2`。发现日期：2026-11-24。

Flow-GRPO把多latent stream按元素数结合成一个action logprob与一个PPO ratio/clip；支持每stream SDE/noise和独立reference beta。ODE stream只作条件不进ratio，可单独用velocity MSE约束；SDE与CPS混合拒绝，最终#414允许CPS不同noise/grid，但此时joint ratio不再是真联合log-density的统一倍数，符号都可能变。不能沿用#406早期描述写成最终禁止不同noise。

LTX2.3 --audio接joint forward与5×H200（4训练+1PickScore）E2E，**奖励只看video**；音频从同advantage学习且元素占比小。四步joint diff低于clip，但audio diff可高于1e-5、visual较video-only差9–23%，未证明残余数值因果。非已证实音频能力改善。

另FSDP初始化可meta构造后每rank由safetensors直接读本地shard，避免rank0全模CPU/broadcast；仅可见且匹配的文件、无LoRA等合条件路径，缺条件fallback或全rank一致报错。建议AV RL/大DiT载入团队深入阅读并独立验证奖励与启动矩阵。

### 训练工具的可复现边界：TRL assistant mask 与 SpecForge 水位

一手来源：[huggingface/trl #8471](https://github.com/huggingface/trl/pull/8471)；[huggingface/trl #8463](https://github.com/huggingface/trl/pull/8463)。

版本/原始日期（UTC）：#8471 2026-11-23T12:20:15Z / `5fa3563cdc94`；#8463 2026-11-23T11:27:43Z / `56e48a8933f0`。发现日期：2026-11-24。

TRL补DeepSeek-V3.2-Exp训练模板，generation标记只覆盖assistant，保持reasoning、工具参数JSON与thinking开头，修复assistant_only_loss初始化拒绝。AsyncGRPO文档明确reward/tool/env在daemon子进程，不能再起multiprocessing Pool/ProcessPoolExecutor；建议thread/async/subprocess，它是原行为披露，未改变daemon机制或性能。

建议对应SFT与异步奖励实现者核模板mask和子进程约束；不要把文档限制说明当新算法。

### SpecForge 规划时即检查 producer 水位能否喂满一次更新

一手来源：[sgl-project/SpecForge #1222](https://github.com/sgl-project/SpecForge/pull/1222)。

版本/原始日期（UTC）：#1222 2026-11-23T09:24:28Z / `4e7068c9d9b7`。发现日期：2026-11-24。

规划与运行复用同dispatch-window计算，含DISAGG_IN_FLIGHT覆盖；避免--plan成功而producer在SGLang加载后退出。作者Qwen3-8B、8×A100的1producer/7trainer、accum5例需要35sample，旧low-watermark32会漏检，改35才通过。187规划/调度/schema测试及原节点复查支持，未宣称训练加速。建议分离式draft训练者采用这个前置容量检查。

### Megatron-Bridge：转换保持模型语义，70B Rubin配方调整无公开绝对计量

一手来源：[NVIDIA-NeMo/Megatron-Bridge #7103](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/7103)；[NVIDIA-NeMo/Megatron-Bridge #7018](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/7018)。

版本/原始日期（UTC）：#7103 2026-11-23T23:32:16Z / `deed50bc03e5`；#7018 2026-11-23T05:53:07Z / `accf35b12f51`。发现日期：2026-11-24。

转换修复DeepSeek-V3导出YaRN误继承reference factor、FP8导入硬编码128块、Llama不支持RoPE被静默变default、GPT-OSS特殊square BF16 gate_up未转置；已发布GPT-OSS20B/120B不受最后一项影响。530 CPU测试，包括非128block和往返，未给生产训练性能。

Llama3-70B VRNVL72 MXFP8配方开FP8 attention/fused QKV+RoPE/TE SwiGLU/去dataset mask，并以PP4/VP5把80层分20/21/21/18，末VP仅2层+loss减尾部。作者称64卡全配方各轮更快，但数值仅内部可索取，mock loss不是质量证据。建议当可审阅配方设计阅读，独立计量后再采用性能结论。

### DeepSpeed autocast KV：修复只适用于纯全注意力 prefill

一手来源：[deepspeedai/DeepSpeed #9124](https://github.com/deepspeedai/DeepSpeed/pull/9124)。

版本/原始日期（UTC）：#9124 2026-11-23T12:17:46Z / `f1d37eb2b3ef`。发现日期：2026-11-24。

FP32 Llama在BF16 autocast下，RoPE的FP32 cos/sin把K提升回FP32，V仍BF16，不能只按autocast dtype分配同dtype cache。decode写前cast到cache dtype；纯full-attention prefill改用按首K懒分配的DeepSpeedStaticCache，保持AutoTP-local KV头。窗口/混合模型保留HF StaticCache，Mistral、Gemma2/3、Qwen3-Next/3.5相应autocast prefill失败并未解决。

新Llama逐token参考及cache路由测试通过，但GPU为T1000 SM75的BF16模拟，6个邻近失败在base也相同；未验SM80+原生BF16、FP16 autocast、多卡/ZeRO3/AutoTP完整rollout。建议相关RL混精度推理用户优先看限制，不能把一项KV补丁概括成全模型autocast支持。

### vLLM GLM Hopper融合：短prompt decode收益需连同MTP条件复现

一手来源：[vllm-project/vllm #69001](https://github.com/vllm-project/vllm/pull/69001)。

版本/原始日期（UTC）：#69001 2026-11-23T21:59:19Z / `a289c722dd59`。发现日期：2026-11-24。

已将all-reduce+mHC+RMSNorm融合扩到受支持Hopper TP4 decode并默认选择。作者GLM-5.3-Flash配方含MTP5、max_num_seqs48，随机2token输入/1024输出、并发1、32请求/4warmup，两轮输出吞吐约391→425tok/s（约8.7%）；accept length也由4.11变4.18，不能只归给单kernel。AIME25仅小样本分数，stderr很大。

没有明确披露具体Hopper型号及完整checkpoint量化条件，其他TP仍待支持。建议对应低并发Hopper GLM服务考虑复现完整配方，不以8.7%承诺prefill或任意batch收益。

### 补录正式发行：Transformers 5.24、Prime-RL 0.11、LMCache 0.5.8

一手来源：[huggingface/transformers #51124](https://github.com/huggingface/transformers/pull/51124)；[huggingface/transformers #51085](https://github.com/huggingface/transformers/pull/51085)；[huggingface/transformers v5.24.0](https://github.com/huggingface/transformers/releases/tag/v5.24.0)；[PrimeIntellect-ai/prime-rl v0.11.0](https://github.com/PrimeIntellect-ai/prime-rl/releases/tag/v0.11.0)；[LMCache/LMCache v0.5.8](https://github.com/LMCache/LMCache/releases/tag/v0.5.8)。

版本/原始日期（UTC）：#51124 2026-11-23T12:46:33Z / `4b401cf208f3`；#51085 2026-11-23T11:17:53Z / `90e9ee1d25ac`；v5.24.0 2026-11-21T12:28:26Z / `eeee11099135`；v0.11.0 2026-11-22T01:49:08Z / `4cec52773bec`；v0.5.8 2026-11-22T15:39:57Z / `651617efcd55`。发现日期：2026-11-24。

这些是11月21/22日原始发行的补录，不是今日首发；本轮许多main/dev PR发生在tag之后，不能认为已随这些版本发布。

Transformers5.24.0新增Inkling2 Mini、Apertus1.5、Dory等原生装载，cache名统一为cache并提高PEFT到≥0.22；旧字段有过渡，但inspect.signature过滤旧参数、remote ModelOutput和已删除modeling_attn_mask_utils仍会实际破坏remote code。11-23主分支#51085补前两项，#51124一次性恢复旧mask模块解vLLM阻塞，**不能写5.24发行本身已含修复**。建议远程代码用户先阅读兼容更正。

Prime-RL0.11.0把datatrove预训练数据mix/按源恢复、逐token staleness、重放sampler kept-set的IcePop centering、env worker pools等组合进发行。完整组件wire/config需同版，旧LoRA DCP不能resume；MegaMoE还依赖闭源prime-mega-moe，不能把项目主仓开放等同此kernel可复现。建议作为训练全流程版本深入阅读，评估迁移而非只升包。

LMCache0.5.8聚合MP缓存身份/传输/服务正确性；stable与0.5.8rc3 tag均指同一651617ef源码，增量是正式发行/工件状态，不是又一轮性能提升。旧CUDA12.9分发/NIXL打包迁移要看release，今日dev profile和分页等不得混入此tag。

<a id="topic-6"></a>
## 硬件软件协同

本节代码状态以各链接对应的合入提交或固定标签为准；除明确写正式 release 的条目外，均为默认分支能力。设备数据来自作者及项目测试，本轮没有上机复测。

### AMD MI455X FP4：新公开的内核设计和随机输入测量，比峰值更值得读

**AMD，11月23日技术博客；建议深入阅读并考虑复现计时方法。**文章解释 FP4 矩阵核吞吐增长快于供数能力后，如何联合选择 tile/cluster、多播、Tensor Data Mover（TDM，张量数据搬运器）、权重/尺度离线重排和四级 LDS（片上共享存储）流水。新增公开证据尤其有用：单 MI455X 预发布硬件、M=N=K=16384 的 A4W4，静态输入22.25 PFLOPS，随机输入17.16；三类测试的随机输入比静态低23%–30%，作者归因于功耗下的持续时钟。测量实际做于6月30日，11月23日才发表；不应当作当天硬件首发。计时排除输入准备与重排，当前 AITER 二进制也可能不同，不能将数字换算为模型端到端吞吐。实践上应同时报告静态/随机输入和时钟，并把在线激活重排成本计入集成决策。[设计、实测与脚注配置](https://rocm.blogs.amd.com/software-tools-optimization/mi450-asm-fp4-gemm-opt/README.html)

### CUTLASS v4.9.0dev：Rubin 数据驱动同步与 MoE 算子扩展，同时引入必须处理的 TMA 接口变更

**NVIDIA，11月23日发布，固定标签 `v4.9.0dev @ f8acf1aa78f9`；建议算子维护者深入阅读并做升级验证。**release 包含 Rubin TMA 有效性报告、FP4 结构稀疏和 FP6 MMA、producer/consumer GEMM 的数据驱动同步，以及 Operator API 0.4 的 Blackwell MoE 权重梯度和 MXFP8/MXFP4/NVFP4 分组 GEMM。门控 epilogue 可在单 MMA 后融合 SwiGLU 等；这类支持有具体持久化调度/精度组合边界，不能理解成所有 GEMM 都已支持。

采用前首先处理正确性：device-dynamic TMA 张量必须在创建及指针/形状/stride 改变后显式 `tma_update`，并使用返回值；epilogue 的 TMA store 不再自动排空，最后一次调用后需 `tail()`。release 还列有 Rubin packed 混合类型/PTX 错码、extent=1 的动态 stride 折叠、TMA descriptor prefetch 被提升到 warp0 等已知限制。GitHub `prerelease=false` 与标签里的 `dev` 同时存在，本报保留原名，**不称4.9.0稳定版**。约25%导入提速与约50% host launch 路径改进均为官方局部指标，无本轮模型实测。[完整 release 与已知问题](https://github.com/NVIDIA/cutlass/releases/tag/v4.9.0dev)

### NCCL 2.32.7-1：正式更新通信与暂存内存控制，收益尚无可比测量

**NVIDIA，11月24日08:46（Asia/Shanghai）正式发布，`351f33fd087e`；建议相关集群考虑验证升级。**修正 Vera Rubin merged-NIC 系统的 NVLSTree 性能，调整 AllGather/ReduceScatter 峰值配置；未加载 profiler 插件时减少 bookkeeping，尤其针对多 peer 小消息 AlltoAll；新增 `NCCL_SYM_GINBUFF_MAX_SIZE` 限制 symmetric GIN 核估算的暂存缓冲；另修复面向 PTX 编译时 device runtime 链接冲突。release 未给模型、集群规模或端到端改善数字，因此价值是具体升级候选和显存控制入口，而非已证实的训练提速。[正式 release](https://github.com/NVIDIA/nccl/releases/tag/v2.32.7-1)

### Transformer Engine：BF16 激活查表和 dbias 流水化，保留数值角落与算子级比较

**NVIDIA，11月23日，#3761/#3852，合入 `5a2385e42f1e` / `123eafd8991d`；建议定向复现。**CuTeDSL 寄存器驻留 MXFP8 量化增加 BF16 激活表、导数及融合偏置梯度；查表以 `NVTE_ACTIVATION_TABLE=1` 显式启用，默认关闭。GeLU/SiLU 的负无穷和极小输入导数存在文档化差异，不能笼统称逐位等价。单 GB300、BF16输入/E4M3输出、4096×18432、500次中位且冲刷L2，GeLU开表78.06→57.54µs，SiLU不开表75.91→61.79µs；不是整网增益。[#3761](https://github.com/NVIDIA/TransformerEngine/pull/3761)

dbias 的独立改动以64线程 CTA、`cp.async` 四级流水增加列并行，保持原行求和顺序；单 GB300 的 cast+dbias 4096×7168 为28.83→25.15µs，4096×36864 为100.58→95.25µs。该比较只计 CuTeDSL 配置，其他共用 reducer 的调用未测速；较低 occupancy 不妨碍覆盖更多 SM 后变快，值得作为性能分析案例。[#3852](https://github.com/NVIDIA/TransformerEngine/pull/3852)

### Transformer Engine 交叉熵：局部全掩码分片不能污染全局 softmax

**NVIDIA，11月23日，#3891 / `a93fc0eedb0a`；建议相关训练栈关注修复。**online-softmax 合并遇到局部全 `-inf` 时，旧式 `exp(-inf - -inf)` 会产生 NaN 并污染损失/梯度。新增有限 shift 使该分片贡献零质量，覆盖单核长前导掩码和三 rank 的两块全掩码测试。修复针对仍有有效类别的全局行；不扩写为“所有 logits 全掩码也有定义的正常损失”。[合入 PR、回归范围](https://github.com/NVIDIA/TransformerEngine/pull/3891)

### ATOM 稀疏 MLA：按工作量选择搬 query 或 KV，长前缀的小增量仍可能反向

**AMD ROCm/ATOM，11月23日，#3547 / `097d898b47ff`；建议深入阅读并按真实负载复现。**DCP（解码上下文并行）的旧 prefill 路径汇聚 query，再以 LSE 合并分布式注意力；新路径汇聚每token约576B的FP8 KV，让各rank用自己的heads处理完整候选，省去prefill的query汇聚和输出合并。默认 `auto` 按本步总上下文、新token数及缓冲上限选择；只用于FP8、MLA page=1、非插件且无PCP（预填充上下文并行）。

MI355X、64K冷prompt、仅输出1token、3次比较，GLM-5.2-FP8 TP4/DCP4为6.10→4.15s，TP8/DCP8为3.65→2.61s，Kimi-K3.1-Pro为5.49→4.58s。960K缓存前缀只增64/1K token时强制新路径反而更慢；`auto`系数主要由GLM-5.2单请求空闲服务拟合，不能视为普适。部分质量与性能数据来自早期修订；旧 `q` 暂存缓冲仍未完整计入显存profile，0.95利用率基线故障的因果未确认。本轮将这些边界与收益同时保留。[机制、配置、实测及限制](https://github.com/ROCm/ATOM/pull/3547)

### ATOM Mono 与 GIL：层级融合和短回合交付延迟是两种不同成本

**ROCm/ATOM，11月23日，#3736/#3469；建议分别复现内核与服务层。**`83865cfa5010`把 DeepSeek-V4.1 TP8 的层内K1/K2及压缩器融合，按列分片残差，并以节点共享的锁页 Engram 表代替每rank重复装载；同时校正draft固定缓冲覆盖前的D2H事件等待。PR在FP8 KV、DSpark5条件报告133K单请求、固定接受长度3.51时2.79ms中位解码步，但没有同配置整网基线，不据此计算加速倍数。[#3736](https://github.com/ROCm/ATOM/pull/3736)

`ecb2d3020a77`在忙轮询空闲分支和输出排队后主动释放 Python GIL（全局解释器锁），减少完成请求的最后chunk等到约5ms线程切换的现象。8×MI355X、另一hi Mono集成版本、强制接受率的10token探针，ITL约1.418→0.910ms；1200s单轮AgentX的P90 interactivity仅979.1→999.4，说明短回合局部收益不等于全负载等比例提升。[#3469](https://github.com/ROCm/ATOM/pull/3469)

### ATOM MiniMax NVFP4：跨 TP 的页面结构和 draft dtype 成为显式传输契约

**ROCm/ATOM，11月23日，#3472/#3616，`59c401ababb7` / `71b2d9863e8b`；建议采用前做P/D往返测试。**NVFP4页将K/V数据与scale分成四个不同宽度的head run，Mooncake P/D据此切分TP；Store命名空间纳入目标/draft dtype，连接器注册时分别声明能否搬NVFP4页及独立dtype的DSpark草稿。此前仅LMCache路径获准，现在Mooncake及Store接通；EAGLE3独立dtype及未声明能力的传输仍被拒绝，MXFP4没有随之放开。

作者给出单MI355X节点、MiniMax-M3.1-Flash、PP4×TP1 prefill→TP4 decode的质量与Store重载检查；#3472部分整网数据早于最终表驱动gather，而#3616记录当前stack的BF16/FP8草稿GSM8K和重载needle。不能把早期AgentX吞吐全部归因于格式声明修复，也不能据此推定任意异构TP组合。[页面格式与早期证据](https://github.com/ROCm/ATOM/pull/3472) · [独立draft dtype及当前验证](https://github.com/ROCm/ATOM/pull/3616)

### ATOM Qwen3.8：卸载必须保存 QSA、GDN 与 PLE 的完整状态

**ROCm/ATOM，11月23日，#2636 / `811b02bb4177`；建议研究混合模型缓存实现。**新LMCache offload/MP发布覆盖QSA的K/V、raw/compressed index、可选图像位置及草稿平面，且把PLE（逐层嵌入）的卷积/ngram窗口放入GDN（门控DeltaNet）状态检查点。layout ID记录SSM转置存储、各平面shape/dtype，避免相同模型名下读错布局。由于forward内部没有快照PLE窗口，Qwen3.8的midstep读被显式禁用；MTP边界延续既有verify-image契约。价值是消除“只存KV就能恢复混合模型”的错误假设；本轮不从新增codec推断端到端收益。[合入PR及恢复边界](https://github.com/ROCm/ATOM/pull/2636)

### ATOM KV-store：保存流水与租约续期已合入，仍为默认关闭的实验路径

**ROCm/ATOM，11月23日，#3475/#3533，`a90691df56f3` / `03244191644c`；建议继续观察。**保存流水让打包与native put重叠，以连接器级credit限制在途批次，失败时等已提交native操作完成再结算/隔离slot；另有显式异步租约续期。保存窗口1–64由`ATOM_STORE_SAVE_BATCH_CHUNKS`启用，默认0，当前路径限制单保存线程。608→358ms来自人为50ms pack/put的CPU模拟；最终修订无GPU/多rank实测，早期GPU比较无收益。租约项也无独立flag-on/off硬件结果。因此只收录机制和验证缺口，不写成已证明的吞吐优化。[保存流水](https://github.com/ROCm/ATOM/pull/3475) · [异步租约](https://github.com/ROCm/ATOM/pull/3533)

### ATOM P/D 与 ReplaySSM：发布范围、stride 与状态恢复的正确性修复

**ROCm/ATOM，11月23日，#3589/#3614，`3252233fec5c` / `93f31c640914`；建议相应部署关注。**Mooncake producer仅发布prompt占用的KV块，避免将未来decode预留块当作已就绪数据；作者同一gpt-oss-120b 1P1D作业的GSM8K由0.8461到0.8954，属于修错恢复质量，非算法能力提升。KDA ReplaySSM改按token stride读取融合projection产生的beta等视图，保留非dense布局的复制fallback；其wrapper收益来自非flush算子对照，已有flush次序问题被留在后续修复范围，不能视为整个ReplaySSM已无竞态。[prompt发布](https://github.com/ROCm/ATOM/pull/3589) · [stride修复及未覆盖flush](https://github.com/ROCm/ATOM/pull/3614)

### MoRI：接收端 fence 补齐晚到 RDMA 写入的回收条件，HBM复制遵守host注册片边界

**ROCm/mori，11月23日，#848/#1073；建议通信/缓存维护者深入读。**`6c2f7f50e08d`增加接收端fence：切断某peer的RDMA队列对、拒绝新握手并失败结算在途传输；只有返回`ok`才构成“晚写不再到达”的回收保证，失败时不能释放。unfence还要由调用方在abort确认后执行。作者两MI355X节点RoCE测试及vLLM集成分支故障注入有证据，**vLLM调用方仍是独立后续分支，不能称默认已接通**。[fence契约与测试](https://github.com/ROCm/mori/pull/848)

`032a34284783`修复HBM↔DRAM复制把已经按host注册片切开的segment再次合并的问题。4×MI355X、DeepSeek-V4.1-Flash、384GiB DRAM、256 session/600s AgentX，128token chunk的失败get约154→0，256token chunk的85失败put→0；分片内仍合并。意义是恢复可缓存性和避免整prompt重算，不能把不同chunk尺寸的吞吐差当作单纯内核加速。[注册片边界修复](https://github.com/ROCm/mori/pull/1073)

### MoRI EP：小批EP32与gfx1250 EP4各有收益，网络和配置条件不能省略

**ROCm/mori，11月23日，#937/#1043；建议定向复现。**`f105d21736d2`修正16B staging对齐、提前RDMA post并增加EP32表。4节点×8 MI355X、一GPU一AINIC、top16/hidden3584、FP8 dispatch/BF16 combine，1 token/rank时245.8→174.3µs，128token为295.4→262.8µs；含新launch表，且两臂都带未合入的host-ring选项。另一hidden7168/top8在128token处无明显收益。`100d0a8fa760`针对gfx1250 EP4把量化前移到ring slot获取前、重排producer warp并按token认领归约；16384token/rank时FP8 combine483→343µs、FP4为283→239µs，dispatch未变。两项都是通信算子，非整网吞吐承诺。[EP32](https://github.com/ROCm/mori/pull/937) · [EP4](https://github.com/ROCm/mori/pull/1043)

### AITER gfx942 FP8：FNUZ格式、probability尺度和QSA预填充一起决定可用性

**ROCm/aiter，11月23日，#6199/#8149，`8c2471aff5c0` / `c21216dd4386`；建议按格式和集成版本复现。**稀疏MLA改用gfx942原生e4m3fnuz解读缓存、相应MFMA及probability尺度，拒绝另一架构编码。MI325X、GLM-5.3-Flash TP4形状的16K prefill为BF16缓存10.55ms、FP8缓存/FP8 dot5.78ms；但FP8 dot对FP32注意力的误差增大。131K入/1K出服务几何均值155.1→165.7 tok/s、KV容量12.2M→23.1M来自另带未在上游提供的Gluon dispatch helper的vLLM，不能当stock部署结果。[MLA格式与条件](https://github.com/ROCm/aiter/pull/6199)

QSA新增MI308X FP8 K/V、按token/head或tensor尺度、packed与slot pool及超过4GiB地址；`compute=fp8`还量化Q/P，`bf16`则走解量化后路径，并保留opt-in batch invariant模式。卡片提到缓存容量增加89%是客户配置背景，不能单独归为本PR端到端增益。[QSA实现、数值与布局范围](https://github.com/ROCm/aiter/pull/8149)

### AITER 0.1.30正式发布：补录可部署版本，今日main改动需另判

**ROCm/aiter，11月22日09:53Z发布，正式tag `v0.1.30`；本轮补录，建议固定依赖时阅读。**本轮精确URL去重未见历史日报收录该release。它将此前kernel/硬件支持与正确性变化汇成正式版本；相对之前逐PR记录，本条增量是发行状态。不能把11月23日才合入的Triton、Top-K等改动都写成0.1.30已包含；升级需比对tag和所需PR的提交关系，本轮没有新安装或设备验证。[正式release与完整changelog](https://github.com/ROCm/aiter/releases/tag/v0.1.30)

### AITER Triton 3.9迁移：显式FMA、scale流水、资源配置和已知失败需一起审

**ROCm/aiter，11月23日，#7544/#8266/#7742/#8175；建议升级栈时一起核查。**`12238413ce8d`用显式`tl.fma`并前取scale，避免3.9不再默认隐式收缩乘加后的退步；gfx950指定4096/K7168混合形状FP8段129.9→77.8µs，同时含两项优化，不把全部收益归因FMA。迁移PR修正MXFP4编译及小M LDS超额，并改善安装版本解析；旧验证记录仍有MHA/dropout和gfx942特殊head dim的工具链故障及CI deselect，绿色CI不能说明这些路径通过。[FMA/scale](https://github.com/ROCm/aiter/pull/7544) · [3.9迁移及未验证项](https://github.com/ROCm/aiter/pull/8266)

`37651f91df07`把MHA backward stride运算扩到int64，处理超2^31元素地址溢出；`c66cfc3fff09`仅取消positions对齐/缓存row offset的多余特化，以字节相同的amdgcn证据减少重复编译，非内核执行提速。另同日gfx950 FP8 MQA #7183因未隔离架构builtin、破坏gfx942+gfx950双架构编译而被#8504撤回，本轮不列为可用新能力。[地址正确性](https://github.com/ROCm/aiter/pull/7742) · [编译键](https://github.com/ROCm/aiter/pull/8175) · [撤回](https://github.com/ROCm/aiter/pull/8504)

### AITER 路由与融合：以实测dispatch决定Top-K，独立RMSNorm融合仍有集成开关

**ROCm/aiter，11月23日，#5677/#8258/#8158；建议阅读选择机制。**adaptive decode Top-K用`arch/CU/emit/k/width`的测量表选核，两个公开入口共用一个gate；未列硬件保持原路径，gfx1250没有测量因此不启用。混合长短行和bound不紧的情况可能改变收益，不能只挑最大倍数。[自适应Top-K与长尾限制](https://github.com/ROCm/aiter/pull/5677)

384专家sqrtsoftplus的gfx950 HIP路由复用wave64 vwave，限定top-k≤8并保留tie/归一化顺序，其他路由不变；RMSNorm新核则合并DeepSeek的q_a/kv_a归一化和MXFP4量化，严格限制q宽度1024<n≤1536、kv≤512及特定scale布局。作者约8.1→3.7µs是decode算子历史比较，ATOM集成另由默认关闭的flag控制；不把算子合入写成所有DeepSeek部署已获益。[路由](https://github.com/ROCm/aiter/pull/8258) · [RMSNorm融合与输入契约](https://github.com/ROCm/aiter/pull/8158)

### 昇腾混合模型：GDN RecoverSSM、Qwen PCP与GLM DFlash2各有不同交付边界

**vLLM-Ascend，11月23–24日，#22045/#21693/#21831；建议按各自边界考虑复现。**`6ef9f12125ac`实验性接入Qwen3.5/3.6/3.8-27B的GDN RecoverSSM，保存卷积、原dtype检查点及FP32 replay记录，以接受前缀反向commit支持投机、align前缀缓存和指定Mooncake P/D。PR给出静态检查和新增验证套件，不能把列出的待执行命令写成设备测试已通过。[RecoverSSM](https://github.com/vllm-project/vllm-ascend/pull/22045)

`a0a7343822c6`在MRV2启用Qwen3.8-Flash-Next PCP，要求DCP=1且GDN heads能被TP×PCP整除；早期A3长prompt连续24token相同，混batch短continuation两token后分歧，支持矩阵仍标未验证。`9316dd8089b0`接通GLM-5.3-Flash的DFlash2+PP，A3 16NPU、PP2/TP8、PCP1 eager仅一条80token smoke，不能把70/70接受草稿当成稳定接受率；PP+sequence-parallel MoE仍不支持。[PCP](https://github.com/vllm-project/vllm-ascend/pull/21693) · [DFlash2+PP](https://github.com/vllm-project/vllm-ascend/pull/21831)

### 昇腾 KV Pool：角色默认、完成事件和分片恢复需要一并迁移

**vLLM-Ascend，11月23日，#22713/#22514/#22621/#22834；建议现有配置维护者阅读。**Decode查找改由`enable_lookup`控制，consumer默认关闭；Decode保存统一`save_decode_cache`，旧consumer开关明确拒绝并提示迁移。CPU KV事件按真正写该块的rank聚合，并可跨iteration等待，避免只认worker0或假定所有rank写每块；save event改用完整128bit UUID并在所有worker确认后释放KV。仅移除构造拒绝的TP+PCP Mamba支持不等于任意P/D布局可互读，双方仍需相同传输模式和cache layout。角色迁移主要CPU测试，UUID有8×950DT冷/热重载小样本；旧32/64bit性能表不代表最终128bit实现。[角色接口](https://github.com/vllm-project/vllm-ascend/pull/22713) · [事件聚合](https://github.com/vllm-project/vllm-ascend/pull/22514) · [UUID完成确认](https://github.com/vllm-project/vllm-ascend/pull/22621) · [Mamba配置放行](https://github.com/vllm-project/vllm-ascend/pull/22834)

### 昇腾稀疏卸载与indexer：流水收益和跨层复用不能混为同一归因

**vLLM-Ascend，11月23日，#20289/#22720/#22656；建议在固定版本上定向复现。**MRV2稀疏KV改为冻结请求计划、专用stream异步D2H、下一iteration按TP完成集体确认后再复用slot；并重新启用prolog_v3/mlapo。作者跨host P/D、prefill TP16 / decode DP16×TP2、MTP3数据为TPOT22.6→21.2ms（仅延迟D2H）→19.6ms（再加融合），所以总15%输出吞吐不能全算给D2H；PR未完整披露模型、硬件及输入输出，不能直接横比别栈。[D2H及消融表](https://github.com/vllm-project/vllm-ascend/pull/20289)

A5 C8 indexer同时迁移CANN V2并复用跨层metadata；每build仍刷新值，保留graph地址且隔离draft buffer。GLM-5.3、两8卡950DT、PP2/PCP8、DFlash7、16384入/20出、三轮中，C32吞吐+2.63%，C1/C8无明确收益；只是一次正序比较，且当前缩scope修订未设备复跑。PCP的draft启动profile另改为rank-local target预算，但全draft context仍需gather；#22656没有NPU容量测量，不写具体显存收益。[indexer与证据归属](https://github.com/vllm-project/vllm-ascend/pull/22720) · [启动profile](https://github.com/vllm-project/vllm-ascend/pull/22656)

### 昇腾正确性：replay专家游标和每层SwiGLU限制直接影响输出

**vLLM-Ascend，11月23日，#22420/#22405/#22761；建议命中配置优先检查。**bounded replay新context需从外层继承MoE游标，否则DeepSeek-V4.1-Flash后段层读前段专家；作者下游debug runtime、不同950节点的三轮GSM8K，完成答题准确率75.22%→97.38%，计入全部调度题则73.41%→96.87%。有timeout和首轮retry差异，但逐层trace与回归支持游标根因；这是恢复正确性。[游标修复](https://github.com/vllm-project/vllm-ascend/pull/22420)

Ling的`swiglustep`按层在SiLU后clamp，unset/0使用7；fused MC2/MegaMoE不能表达该限制时回退regular MoE。INT8 target搭配BF16 draft的SFA packed缓存则按owner spec dtype reshape，修复二倍element count启动失败，设备证据使用dummy权重且只验证启动/请求，不证明模型质量。[激活限制与回退](https://github.com/vllm-project/vllm-ascend/pull/22405) · [draft dtype](https://github.com/vllm-project/vllm-ascend/pull/22761)

### 昇腾融合与生命周期：A3共享专家放行、A5 eager缓冲退役，以及两项同日撤回

**vLLM-Ascend，11月23日；建议按提交核当前树。**#22207使A3 W8A8/W4A8可用`enable_fused_mc2=3`，INT8共享专家遵循(in,out)/NZ权重和FP32封装uint64尺度；PR的约1.6–1.8×吞吐缺完整模型、负载与原始表，仅保留为作者声称，不作为可比较结论。#22592在A5 eager packaged MLA提交stream等待事件后退役query/output的记录使用；旧8×950、8192入/1出、C64诊断中等待复用显存6.266→0.680GiB、重试4→0，graph捕获路径不变。[A3共享专家](https://github.com/vllm-project/vllm-ascend/pull/22207) · [eager生命周期](https://github.com/vllm-project/vllm-ascend/pull/22592)

共享DSpark MegaMoE原生context的#21727同日被#22964撤回，Mamba align block-table融合#21196也被#22924撤回。撤回正文部分复制了原“修复收益”，最终补丁实际删除/恢复相关改动；**本轮不把这两项写成当前能力**。撤回原因没有充分证据时不推测。[context撤回](https://github.com/vllm-project/vllm-ascend/pull/22964) · [gather融合撤回](https://github.com/vllm-project/vllm-ascend/pull/22924)

### 昇腾 HiSparse：旧HDK通过MemFabric启动，但该实测组合更慢

**vLLM-Ascend，11月23日，#22600 / `913f9d79f15c`；建议有容量需求再考虑复现。**HDK25.5.1缺native AIV Host映射能力时，可显式选MemFabric AccOffload LOCAL池；1.3.0/1.3.1有不同arena分配器。它解决本机Host池可达性，不代表P/D SHARED backing已完成整网资格验证。A3、GLM-5.3 W8A8、TP16/EP、MTP4、INT8 KV/index、32K入/1024强制出、C1两轮，HiSparse ON相对OFF的TPOT+10.35%、输出吞吐−6.85%；全GSM8K为1281/1319对1287/1319。旧Core组合已跑，当前Core pin仍待服务验证；应权衡容量而非称纯性能优化。[配置、容量背景及负面结果](https://github.com/vllm-project/vllm-ascend/pull/22600)

### 昇腾 DCP rank tile：最终代码采用A3静态预热、A5动态，旧PR描述不能代表交付

**vLLM-Ascend，11月23日，#19555 / `6450b60e37c6`；建议深入读dispatch与预热。**PR正文仍保留旧“动态A3退步、A5禁用”记录；本轮核最终diff，`_rank_tiled_mode()`实际为A5 dynamic、A3 static、其他off。A3只用预热过的完整head/token/dim/DCP/stride签名，未预热回退scalar/batched，不在在线请求临时编译；A5使用动态token/head。BF16、head scatter、local贡献连续、DCP8–16、head dim≥512等条件仍在。随补丁文档给出算子/8rank HCCL比较，但all-to-all约53µs未变，不能据combine倍数推导服务TPOT；旧模型质量基线也不当候选质量。该项首先是编译器差异改变实施方案的工程案例。[合入PR](https://github.com/vllm-project/vllm-ascend/pull/19555) · [固定提交](https://github.com/vllm-project/vllm-ascend/commit/6450b60e37c6)

### TPU混合缓存：修正engine/worker容量分歧，投机状态导出仍待Runner接线

**vLLM TPU，11月23日，#4017/#4007；建议相关Ray/multiproc部署检查。**`b876f4043b48`把compact-mamba计算的attention block override传回engine core，避免worker计划27376块而core按统一padding仅算713块；Qwen3.8-2.4T-A95B-FP8、16×v7x、TP4/attention-DP8的作者日志显示每rank约11Ktoken上限及等待队列卡住。这是特定布局下恢复调度容量，非硬件显存增加。[容量同步](https://github.com/vllm-project/tpu-inference/pull/4017)

`c3b3d4fe4b8e`增加设备侧GDN state slot解析和block边界导出，避免verify窗口连续写覆盖其他请求的prefix slot。但main在投机开启时仍强制mamba cache mode=none，新增align+spec分支**处于休眠，需后续Runner PR**；本轮只报告底层实现，不写前缀缓存与投机组合已经启用。[底层状态导出与休眠说明](https://github.com/vllm-project/tpu-inference/pull/4007)

### SGLang-TorchTPU：显式页面布局与SparseCore有效区读取

**SGLang TPU，11月23日，#2418/#2363；建议定向读缓存接口并复现混合长度。**Raiden P/D从具体MHA类检查改为`kv_pool_layout()`的block-major契约，注册整页tensor；混合SWA子池布局冲突会拒绝，当前主要CPU接口验证，无吞吐数据。[页面注册](https://github.com/sgl-project/sglang-torchtpu/pull/2418)

SparseCore Top-K只读每行有效区并减少一行占用subcore数，DCP merge保持旧路。单tpu7x host、k512、batch含一长行时4/8/16/32行算子42/67/110/228→32/42/58/121µs；全短行约30µs未变。作者66测试、索引比对及两次AgentX零失败支持实现，不能据算子1.9×推导模型同等吞吐。[Top-K](https://github.com/sgl-project/sglang-torchtpu/pull/2363)

### hipBLASLt / CK Tile：A2A+GEMM实验性API与按mask范围拆分KV

**ROCm libraries，11月23日，#15352/#15681；建议算子与通信集成者阅读。**`3341d91bcbb7`把融合AlltoAll prologue暴露给hipBLASLt C API，gfx950 BF16 world2/4/8；仍由默认关闭的实验编译开关保护。调用方须在任何rank launch前保证所有B就绪，库未补跨rank就绪协议；旧实验API直接重命名且无alias，不可当作稳定ABI。[A2A prologue](https://github.com/ROCm/rocm-libraries/pull/15352)

`082a2cbc05c1`让gfx1250单query的split-KV只拆causal/window有效tile，mask下合并GQA query heads以共享KV，并修复每head attention sink读取；d128小row用较薄pingpong tile，d256改变occupancy目标帮助提前读取。性能只对应限定decode路径，其他架构仍按原seqlen选split；本轮未转述跨形状最大倍数。[split-KV布局与正确性](https://github.com/ROCm/rocm-libraries/pull/15681)

### MLX：布尔mask广播大幅缩小注意力临时内存，另修大logits与线程资源寿命

**MLX，11月23日，#4934/#4938/#4936；建议CUDA用户考虑复现。**`f04e1f27bf0e`让SDPA广播布尔mask而非物化expanded中间量。单H100、B1、Q32/KV8 heads、8192×8192、head dim128的eager forward17.123→3.769ms，报告临时/峰值指标8256→192MiB；这是特定算子布尔mask比较，非LLM整网质量/显存结论。`c23a35c9e6f4`另修输出2^32元素矩阵乘法的batch计数溢出（如16384token×262144词表）。`4b15effb7bc6`用进程主线程识别，避免首个调用MLX的worker退出后线程ID复用导致Metal queue留在进程级map。后两项没有性能提升数字。[mask](https://github.com/ml-explore/mlx/pull/4934) · [大矩阵](https://github.com/ml-explore/mlx/pull/4938) · [线程资源](https://github.com/ml-explore/mlx/pull/4936)

### MLXcel：Inkling2算子移植与Mamba2循环状态ABI，批服务范围不同

**MLXcel，11月23日，#3367/#3369/#3371；建议小模型/端侧后端研究者定向复现。**Inkling2 Small/Mini支持保留每次loop visit的独立缓存及卷积rewind，Mini48次访问/24物理层；后续Metal融合在M3 Ultra同checkpoint对照，native decode32.07→35.29tok/s、4bit36.81→44.76，三次均值，128greedy输出逐字一致。grouped expert核仅在2–8token speculative verify内使用，padded prefill仍拒绝、batched decode是逐行fallback，不是完整批优化。[模型接入](https://github.com/lablup/mlxcel/pull/3367) · [Metal融合](https://github.com/lablup/mlxcel/pull/3369)

XLA/IREE shim接入可选KV+循环状态列表，Mamba2-130M在HIP和CPU local-task对40token HF oracle一致；只完成single-sequence，循环模型server/continuous batching仍由后续任务实现，某些分组norm形状拒绝。失败prefill清空状态保护下一请求，但同一请求decode中途失败的重试仍可能面对部分推进状态，不能称事务化恢复已完整。[循环状态ABI、验证及剩余边界](https://github.com/lablup/mlxcel/pull/3371)

### MLXcel数值与加载：修正舍入、YaRN与EOS，仍保留LongCat残差

**MLXcel，11月23日，#3336/#3306/#3313/#3335；建议后端对齐者深入读。**Qwen3-VL XLA视觉塔按eager BF16存储边界舍入，而非全FP32，GB10 fixture greedy从第8token分歧延至27，但部分chained block/main merger误差变大，不能称整体逐位对齐。MLA终于按checkpoint应用YaRN，GigaChat3长段显著减少分歧，LongCat仍有39/20486个decided差异且超参考自身噪声，原因未确认。[舍入边界](https://github.com/lablup/mlxcel/pull/3336) · [YaRN及残差](https://github.com/lablup/mlxcel/pull/3306)

XLA EOS集合合并generation/config/text_config/tokenizer来源，避免缺generation_config时越过结束符；六MoE家族在FFI stack前验证专家shape/dtype，把checkpoint不一致从进程abort改为带key的加载错误。两者是部署可用性修复，无新能力分数。[EOS](https://github.com/lablup/mlxcel/pull/3313) · [专家加载校验](https://github.com/lablup/mlxcel/pull/3335)

### AWS：HyperPod把常变节点软件改成受管期望状态；G7新增裸金属规格

**AWS，11月23日正式公告；建议已有平台团队考虑采用，模型性能继续观察。**HyperPod Slurm extensions首先支持GPU/EFA/Slurm监控与统一用户管理，配置声明后校验并应用到新增/替换节点，回报健康状态；EKS的Studio入口则支持PyTorch/Ray作业与模型部署、可复用资源preset、YAML预览和队列优先级。这两种编排入口各有范围，不能合称所有HyperPod工作流已统一。公告无训练提速测量，价值是减少节点配置漂移和操作负担。[Slurm extensions](https://aws.amazon.com/about-aws/whats-new/2026/11/sagemaker-hyperpod-extensions/) · [EKS Studio](https://aws.amazon.com/about-aws/whats-new/2026/11/sagemaker-studio-hyperpod/)

G7 metal在us-east-1提供8×RTX PRO4500 Blackwell Server（每卡32GB）、192vCPU/768GiB主存、700Gbps EFA，适合需要自有虚拟化/driver控制的部署。官方“相对G6最多4.6×推理”没有模型、精度与负载，故不采用该倍数作LLM选型结论；也不把8卡显存总量理解为单GPU可寻址池。[G7 metal公告](https://aws.amazon.com/about-aws/whats-new/2026/11/amazon-g7-metal-size/)

<a id="topic-7"></a>
## 评估与可靠性

### VideoGym：视觉推理视频的收益高度依赖时长，组合规则几乎不泛化

**机构／日期／版本：OpenMOSS-Team／VideoGym；11月23日；LTX-2.3工件`64a410d9f000`，代码`4cde89707bac`。建议深入阅读并优先考虑复现负面结果。**发布245,884程序化视频／246任务的训练集、评测与约22B视频DiT全参数SFT权重；32×H200、BF16/FSDP、10epoch／76,840步，其余编码器与VAE冻结。无外部LLM改写、逐帧全视频验证的pass@1中，1230个域内问题在参考轨迹时长下56.34%，强制81帧降到32.20%；组合仅1/170（0.59%）、跨任务1/165（0.61%）。基座域内为0%，但熟悉模式内提升不等于学会可组合规则。[固定模型卡与完整表](https://huggingface.co/OpenMOSS-Team/VideoGym-LTX-2.3/blob/64a410d9f00071eb1ea0e675e760b4b3ad0c817c/README.md)

变化的是视频生成模型的受控后训练／评测，不直接证明语言模型推理边界。图像重绘、长度外推、多步与单步的退化程度也不同；参考时长本身提供额外信息，必须与固定时长分开报告。论文当前仍Coming Soon，结论来自作者代码、数据与卡片，尚无独立复现；权重继承LTX Video2许可，原创代码Apache2.0。适合检验“生成可视轨迹即推理”的主张，并复现时长、规则组合与完整轨迹验证的交互。[固定代码说明](https://github.com/VideoGym/VideoGym/blob/4cde89707bac6642ad4c1df47b694c0becb34425/README.md)

### Hugging Face agentic-evals：把验证器漏洞、提前结束与拟合能力分开诊断

**机构／日期／版本：Hugging Face；11月23日官方案例。建议深入阅读并采用验证方法。**作者从Terminal-Bench的Raman拟合任务扩展八个合成谱，公开任务、运行轨迹和分析。故意只读峰高、不做拟合的捷径能通过八个“只查D/G比值”的判定，补充逐峰拟合检查才拒绝；这把验证器对答案的检查与对所需工作能力的检查明确分开。[官方案例§2、Appendix A](https://huggingface.co/blog/agentic-evals)

Terminus-2＋gpt-oss-120b的24次运行中19次首响应就声明完成；总成功11/24，pass@3为7/8，但三次全成功的pass^3仅1/8，按任务聚类标准误约0.11，重跑10/24。固定DeepInfra后为13/24，提示提供方也属复现条件；其余三模型24/24并不支持继续在这八个开发任务上排名。该案例不是普遍模型能力结论，价值在于先测oracle／no-op／捷径、查轨迹，再选适合无人值守的稳定性指标。[任务数据](https://huggingface.co/datasets/burtenshaw/raman-lab) · [运行证据](https://huggingface.co/datasets/burtenshaw/raman-lab-runs)

### Vetto Terminal Tasks Open 1.0：30个长任务开放环境、验证器与参考解

**机构／日期／版本：Vetto；11月23日14:29Z release v1.0.0；HF镜像`ac4d37aaab34`。建议考虑加入Agent回归集。**30任务横跨软件、科学、机器学习等七类，包括投机解码语法回滚、JAX事件流分块不变性、训练曲线恢复等；每题开放指令、Docker环境、单独容器运行的隐藏于Agent的验证器及参考解，以Harbor运行。评分检查最终状态，不依赖模型口头“完成”。[正式release](https://github.com/vettoai/terminal-tasks-open/releases/tag/v1.0.0) · [固定镜像卡](https://huggingface.co/datasets/VettoAI/terminal-tasks-open/blob/ac4d37aaab3418013d98d1716cc35a9ed862cb60/README.md)

每任务8小时、1–12CPU／2–32GB，无GPU要求；工作阶段仅允许声明的模型端点，镜像构建与验证器联网。**没有固定预构建镜像**，仅锁任务版本不足以消除构建依赖漂移，先验证oracle再读分数。30题是评估集，公开参考解也增加污染风险，不应直接并入训练；本轮未运行任务或审计全部验证器，因此不认可其“更难”或任何榜单绝对排序。

### Salesforce Swarm公开更多原生回执：ProgramBench新增1728次最终评估，可审查重试选择

**机构／日期／版本：Salesforce；11月23日数据增量；固定`d2d1b318e4c3`。建议深入读原生轨迹，继续观察统计结论。**新增ProgramBench六campaign、18配置、32任务×3重复×N=1/4/8，共1728次最终官方评估；连同此前TB2.1/NL2RepoBench，卡片列5652个计分运行。新增清单区分模型、沟通机制、调整后时间、分批启动，以及重评分恢复与原始执行，便于复核多Agent收益究竟来自并发、通信还是额外预算。[固定数据卡](https://huggingface.co/datasets/Salesforce/swarm-report-runs/blob/d2d1b318e4c39b0490b27c1ef95d7db0eaa9193b/README.md)

作者声明替换尝试按campaign记录选取而非挑高分，超时缺失原生reward的一次计0，并保留重评分历史；27运行缺最终计费记录，时间指标保持缺失，不能补0。交互viewer需要共享密码，底层数据公开；本轮读卡与版本史，未解包核完24,108份worker journal，也未重算跨campaign显著性，因而只收录证据开放，不将运行数或viewer汇总当独立复现结论。

### 四位奖励模型：准确率近乎不变，选中的候选仍可能大幅更换（2611.27155v1，补录）

**机构／日期／版本**：Illinois Institute of Technology、UW-Milwaukee 等；2026-11-20 UTC 提交，v1。

对多奖励模型家族、NF4／FP4／INT4-RTN／GPTQ 的权重四位化，直接追踪每一对偏好是否翻转。RewardBench 的 Llama3.2-1B/NF4 行，准确率仅 -0.03 点，却有 3.99% 决策翻转；数学 best-of-64 的某对照有 34.5% 唯一最大值候选更换，最终答案却仅约4.5%不同。**候选变化不等于伤害，BF16也只是数值参考，非正确性真值**。高斯 margin 模型不能稳定胜过直接测翻转率，路由效果也不优于朴素 top-k；有害翻转方向依赖模型与量化器。建议深入阅读并采用决策级审计，量化裁判时至少同时报告准确率、偏好翻转、最终答案与有害／有益方向。[正文 §3–8、Tables 1/9–10](https://arxiv.org/html/2611.27155v1)

### Tasklets：为架构研究建立统一的小规模计算能力单测（2611.27607v1，补录）

**机构／日期／版本**：Mila、MBZUAI、RIKEN 等；2026-11-20 UTC 提交，v1。

将关联回忆、复制、记忆、计数／排序、栈、形式语言和状态追踪归入共享任务与训练流程，对 softmax／线性注意力、SSM 和 RNN 等混合层做对照。约15万–20万参数、状态容量小于序列长度，学习率三档取最佳；回忆和复制偏向 softmax／部分 delta-rule，而状态追踪展示不同能力边界。该结果是**合成小模型的学习与容量诊断**，不等于大语言模型排名，也非自回归长输出测量；各任务还使用不同的 token／整例准确率。建议新架构团队考虑采用为便宜回归测试，并配真实语言建模和等资源验证。[正文 §3–5、Appendix A](https://arxiv.org/html/2611.27607v1)

### Open CDLM：生成困惑度与单条文本熵都变好，仍可能完全坍缩（2611.27536v1，补录）

**机构／日期／版本**：Open CDLM 项目；2026-11-20 UTC 提交，v1。

同时审计连续扩散语言模型（CDLM）的似然上界松弛与生成分布指标。在 OWT 受控构造中，把同一段挑出的文本重复512遍，生成困惑度25.96→2.46、条内unigram熵4.275→4.419，却只有一个输出支持点；MAUVE识别这类坍缩，但对重排等错误又依赖特征。论文引入KTD等分布诊断，尚不能用一个指标代替质量、覆盖与人类偏好。**这些是短文本编辑构造，不是新训练的CDLM失败率**；似然实验只核一个归档LangFlow检查点，不能宣称已测其他模型的精确PPL。建议扩散模型评估团队深入阅读并保留采样分布、文档独立性和上下界条件。[正文 §3–6、Table 2、§8](https://arxiv.org/html/2611.27536v1)

### 裁判的示例标签比例能制造虚假提升，但单向性证据有限（2611.26881v1，补录）

**机构／日期／版本**：独立作者、University of Essex；2026-11-19 UTC 提交，v1；BlackboxNLP 2026 论文。

固定 LLaVA-NeXT-Video-7B 生成答案，仅改变 Llama3.1-8B 二元 Yes/No 裁判的 few-shot 示例。每数据集200例人工对照中，75%肯定示例让通过率比共识高29.5／19.5点；25%肯定示例没有对称下降。一次示例抽取间的波动还可大于条件均值差。**每种偏斜只有一套示例、单生成器—裁判对**，示例类别构成也变化，不能概括为所有裁判只会单向偏高。建议阅读并采用平衡标签、多个示例抽取、固定生成答案及人工小样本校准；结论尚不覆盖连续评分与pairwise裁判。[正文 §3–5、Table 2、Limitations](https://arxiv.org/html/2611.26881v1)

### PreRun：凭任务描述估运行时，小型回归器仍是必要基线（2611.26875v1，补录）

**机构／日期／版本**：Texas A&M 等；2026-11-19 UTC 提交，v1。

将“模型运行多快”的预测目标拆为训练循环、前向和完整generate调用，避免口径混合。额外受控集为单A100-PCIe40GB、FP32、Qwen2.5-0.5B/T5-small的64配置、三次独立启动，16配置留出；两个开放模型零样本平均低估90–99%，Claude零样本的平均绝对误差为小样本随机森林的4.5倍。五示例能改善部分目标，也有负面迁移。历史数据来自单集群且受运行结果筛选，受控集是合成token、小模型，**数据与代码公开托管仍待处理**。建议作为估时方法的负面结果阅读；实际资源决策仍应使用本栈历史样本与测量模型，不能外推成任意LLM都无法做性能预测。[正文 §4–6、Tables 2–3、Limitations、Appendix D](https://arxiv.org/html/2611.26875v1)


<a id="coverage"></a>
## 覆盖、补录、更正与归档

本轮读取本地47篇历史日报、持久状态及GitHub既有索引/昨日日报，并将远端所有历史归档blob与本地逐一核对一致后续扫。检索覆盖厂商模型与研究入口、论文/会议、工业开源工程、硬件软件栈，并使用中英文主题检索补充。原始响应、固定卡片、PR元数据/补丁、正文与检查记录保存在本地 `.monitor/research-2026-11-24/`。以下区分目录读取与语义审查，不能据此声称全面覆盖，也不能把检索未返回写成某机构没有发布。

| 来源线 | 实际覆盖与尚未完成的范围 |
|---|---|
| 模型与工件 | 157个官方入口，125个完整HTTP200；181个HF组织364页可读，回溯修改候选326个，其中127在主窗；49个定向工件取证。全站创建枚举7554模型、2225数据集并跨过主窗起点，只是元数据与创建目录覆盖，未完成长尾语义和全站修订审查。官网403/404、动态入口、gated历史及部分许可文件缺口留存。新增openjev、oahzxl、VettoAI、rimelabs等发现入口。 |
| 论文与评估 | 10个arXiv分类目录均完整，所返最新公告为11月23日；精确更新窗Atom完整189项，与分类合并830个唯一ID。历史去重后477个词法候选不是全部合格论文；当前38篇与11月11日历史漏收1篇共39篇固定v1，37篇完整HTML、2篇HTML404后补完整PDF。其余792版本/语义候选未审，与旧队列合并后3020项继续留查。OpenReview两个接口403，ACL/PMLR仅目录不当语义全审；11月24日公告批次未返回。没有证明主窗首发项，不等于主窗没有新工作。 |
| 开源工程 | 81个仓库地址尝试，65个成功默认分支目录共1961条commit；六核心项目的合入PR目录600条翻页，21仓release目录按各自台账记覆盖。按最终PR/commit与代码选择审查，保留1414个待审候选；417项只按标题判为维护或历史线索，不当作代码全面审查。PyTorch ghstack若PR显示未合入而默认分支已有commit，按具体commit记，不伪报PR merge。ReFlow为单项截止后补充，未将全仓目录截止外推。 |
| 硬件软件栈 | 29仓32页commit目录、29页release目录，64个PR元数据及64组文件补丁；87次官网/发行文档请求中68次完整HTTP200。目录/传输成功不等于全文语义成功；CANN/GitCode接口429、MLCommons403/404、部分动态页面继续留查。libtpu包行为、CANN组件对应正式包、库存长尾和原始性能日志未全审，旧缺口不关闭。 |

**补录与状态增量。**论文按原提交日、v1和实际发现日注明补录；Effective Teacher Support为11月11日漏收，未改写为本日首发。Xing本轮增量是权重/量化工件，11月19日已收录的报告不重复计入；WePM、BLIP等报告与权重新增分别记版本。AITER0.1.30、Transformers5.24.0、Prime-RL0.11.0、LMCache0.5.8是11月21–22日正式发行补录，不能将后续main/dev提交混进旧tag。Google11月23日残差连接博客回顾DCA旧论文，本轮未发现足够新实验/工件，作为检查记录，不另包装成新架构。

**交叉复核与更正。**GLM表图独立核实为30B而非旗舰消融；Ling幻觉过滤、VideoGym时长/组合分母、LRSA局部与端到端基线均已核对。TorchTitan容量/吞吐的fake进程组、Gumbel偏梯度和蒸馏离线缓存成本保留。昇腾DCP最终代码为A3预热静态/A5动态，纠正了旧PR描述；Hy4卡所写vLLM #69274“未合入”已过时，按当前合入状态记录；同日撤回项不列成可用功能。TPU align+spec底层已合入但主路径休眠，经固定平台源码复核。正文所指独立复核是来源交叉检查，**不是独立设备复现**。

**归档与续查。**本轮只维护本地 `llm-monitor` 以及 `suhaibo666/tracker` 的 `main:llm-training-daily/`。今日日报与README索引的本地文件、待同步状态和远端提交分开记录；同步后另回读两文件、父提交、树和变动路径。没有改写 `llm-knowledge` 的wiki。全局完整语义扫描水位仍未完成；下一轮从 **2026-11-24T01:01:25Z** 续扫并回溯72小时，按稳定版本去重ReFlow补充项，同时继承全部未关闭来源/区间，失败不当扫描成功，上传失败不触发重复生成正文。
