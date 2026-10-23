# LLM 技术动态日报 · 2026-10-23

> 时区：Asia/Shanghai（UTC+8）。增量扫描窗口：**2026-10-22 08:58:39 → 2026-10-23 09:00:15**；72小时回溯自 **2026-10-19 08:58:39**。由上一轮已归档事件游标续查，历史未完成区间继续保留；事件游标不代表所有来源和候选的语义审阅已完成。
>
> 实际生成时间：**2026-10-23T09:25:03+08:00（Asia/Shanghai）**。除核心六仓章节明确用北京时间外，精确事件时间以UTC标注；首次提交、公告、权重上传、修订、合入和正式发版分别记录。截止后读取仅用于核实本窗事实与访问状态，越过截止的新事件不计入正文。
>
> 本轮实际打开论文全文、官方技术文、固定模型卡/config/工件目录及选定PR/commit。指标均为作者/厂商实验或其引用，不是本轮独立复现；没有执行GPU/NPU/TPU训练推理或证明重放。已做选定代码路径、最终补丁与失败边界审阅，大型生成kernel和未选择候选仍有待审。

## 本次重要变化

- **新工件与新报告分开看。** MiniMax-M3.1-Flash首次可见整包权重及DSpark drafter在截止前落地，部署配方仍有缺口；IQuest-Q1的320B权重是9月旧工件，本次新报告披露29T训练、mid-training和多teacher蒸馏。pplx-embed-v2-late新增报告也补齐了teacher、采样及逐token蒸馏细节。
- **正确性修复应优先安排。** FlashAttention变长K/V尾部NaN修复又需3071解决descriptor越界预取；vLLM修NCCL权重buffer跨stream寿命，SGLang修HiCache/HiSparse页面复用竞态，verl修shape相等但CP token身份错配的蒸馏。main修复不能等同已进入当前wheel。
- **低精度与省内存有明确代价。** Transformers FP8 KV在作者单prompt测试中KV减半、延迟却增加16%；Ada FP8的独立FP32加法恢复精度也有性能开销。ProQuant、PACE及稀疏/量化同成本研究值得对照，存储bit、kernel时间和模型质量不能互相替代。
- **后训练研究同时给出方法与反例。** prefix clipping、冲突感知蒸馏、参考随机遮挡和reverse-KL理论各有数据分布与预算边界；Delta N-gram数值稳定但跨位置问答失败，提醒长期记忆应验证知识能否被读出，而非只看写入与遗忘指标。
- **训练和服务系统出现可用阅读入口。** TorchTitan发布规模配方与TitanRL，Megatron完善GTP图下权重预取和global-batch路由平衡；Dynamo1.6、TRT1.4RC1均有具体发布快照与已知故障。FlashInfer KDA CP和Spark八节点实验值得定向复现，模块倍率不是完整训练或服务收益。
- **形式验证工件可重放，但证明范围要读定理。** Anthropic/Theorem开放的60入口、138条agreement证明不包含halts证明；它支持指定二进制/语义下的sanitizer一致性，不能扩成全库、任意平台或必然终止的安全保证。

目录：[模型与技术报告](#models) · [预训练、架构与数据](#pretrain) · [后训练与持续学习](#posttrain) · [推理优化与系统研究](#inference) · [核心开源工程：训练、推理与正确性](#core) · [扩展开源：发布与采用边界](#engineering) · [训练与推理底层：正确性与编译器](#correctness) · [硬件软件协同与设备后端](#hardware) · [评估与可靠性](#evaluation) · [模型与数据版本观察](#versions) · [覆盖与缺口](#coverage)



<a name="models"></a>

## 模型与技术报告

### MiniMax M3.1 Flash 权重落地：全层 MSA、量化感知训练和独立 DSpark drafter 现可核

**MiniMax；API 发布于 2026-10-19，本轮增量是 10-23T00:59:04Z 的首次可见整包提交，固定 `9e7ebacac350497ad4a1272a2e817639bf11561c`，处于本次截止前。** [固定模型卡](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/9e7ebacac350497ad4a1272a2e817639bf11561c/README.md)、[工件目录](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/tree/9e7ebacac350497ad4a1272a2e817639bf11561c)。此前“权重稍后”的缺口有实质进展：可核 48 个主权重分片和独立 `dspark/model-dspark-00001.safetensors`（2.30GB），合计 safetensors 约252.42GB；这是下载体积，不能直接当作运行显存。仓库 createdAt 为10-17，不作为公开权重日期。

卡述约428B总参数、23B激活，图文/视频输入、文本输出、最多1M上下文；[配置](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/9e7ebacac350497ad4a1272a2e817639bf11561c/config.json)可核60层、128路由专家top4+1共享专家、64Q/4KV，`sparse_attention_freq` 全为1：相对M3，最初三个全注意力层也改用MSA。配置同时列 `quant_method=mxfp8` 和 `moe_quant_algo=NVFP4`，不能统称全模型FP4。卡述专家与attention路径进行了QAT，但训练数据、预训练/后训练阶段token、QAT日程、训练预算及消融尚未披露；所链2606.13392仍是既有MSA论文。[drafter配置](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/9e7ebacac350497ad4a1272a2e817639bf11561c/dspark/config.json)为5层、block7、4096滑窗、rank256 Markov head与confidence head，抽取target第3/17/31/45/59层特征；它替换先前EAGLE式MTP，不能脱离target独立使用。

作者卡提供较完整的评测harness条件：DeepSWE v1.1为mini-SWE-agent2.4.5/512K上下文，得73.7%，低于同表DeepSeek-V4.1-Flash-1017的77.0%；Terminal-Bench4为Codex0.149.1，38.1也低于42.3；不能写全面领先。10-19公告的200+TPS/约三倍速度未给可复算硬件、batch和序列条件，本轮不据此预估部署收益。**建议深读并考虑复现**原始QAT权重的长上下文质量、DSpark接受率/吞吐与关闭spec的差异。[TokenSpeed配方](https://lightseek.org/tokenspeed/recipes/models#minimax-m3-1)已列Blackwell、TP4、FP8 KV、7个draft query/8-token verify window；模型卡所链SGLang与vLLM专用配方本轮均404，Transformers/KTransformers等仍标集成中，不能概括为所有发行版开箱即用。FlagOS附件虽列9厂商W8A8再量化版本，本轮未逐仓核实，不写成已验证硬件兼容。权重为[MiniMax Community自定义许可](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/9e7ebacac350497ad4a1272a2e817639bf11561c/LICENSE)，商业使用有展示、通知及收入阈值授权条件，非MIT/Apache。

稳定ID：`hf:MiniMaxAI/MiniMax-M3.1-Flash@9e7ebacac350497ad4a1272a2e817639bf11561c`。

### IQuest-Q1首次技术报告：29T训练、稳定性边界与多harness后训练配方有实质披露

**IQuest Research；新报告`2610.30949v1`于2026-10-22T09:01:29Z提交。** [完整报告](https://arxiv.org/html/2610.30949v1)、[版本时间](https://arxiv.org/abs/2610.30949v1)、[固定模型卡](https://huggingface.co/IQuestLab/IQuest-Q1/blob/5c21b0630586ef77d38cff1094b8cf37a417fcd8/README.md)。官方HF权重9月已公开，最新提交仍为09-29；本轮是新报告，不能写10-22新发320B模型。工件可核175个BF16主分片及独立MTP文件，合计约649.98GB，不含runtime/KV开销；文本输入，512K上下文，无原生视觉/音频。

报告的价值首先在预训练：320B/15B激活、88层、256专家top8，局部4096滑窗与全局attention混合，query-aware sink、QKNorm Clip、Stable SplitNorm、Muon与宽/深缩放规则共同组成训练配方。预训练+mid-training合计**29T token**，code占比在四个phase由40%升70%；Muon用于多数矩阵，embedding/LM head/norm/router用Adam；mid-training32K→128K→512K，跨phase重新初始化optimizer但不重置单一cosine进度，loss可按token span选择而非仅按role。作者同数据对Qwen3-MoE式基线得到约1.93×训练loss/FLOPs效率；这是整套设计的受控训练loss比较，不能归因Muon单项，也不是实际集群吞吐倍率。对safe/optimal学习率拟合外推的“约572B激活/12T总参数CMS”**不是已训练模型或实证稳定上限**。

后训练披露SFT两阶段240B+/50B+ token，之后按领域/harness/版本固定条件做GRPO，同task比较多个rollout；环境/评测器故障与policy错误分开归因，污染轨迹不消费。MOPD由四个冻结领域expert给student自生成token打logprob，截断teacher/student对数比优势，按轨迹长度归一化并屏蔽重复输出；teacher按需换权、分片预取，共用训练引擎算logprob，避免每个teacher常驻一组GPU。还给SWA halo交换与local partial attention/LSE合并的CP机制，但缺少完整训练系统、语料与全阶段算力工件，不能当完全可复现训练。所谓human-on-the-loop RSI给出一次人工审核诊断/训练决策的2.35B token、938.7GPU小时工具schema修订案例，不能扩成无人监督递归自改进。

评测保留边界：DeepSWE v1.1使用mini-SWE-agent，其余coding多用Claude Code，CyberGym/ProgramBench6小时、Terminal-Bench4为10小时、各模型最高reasoning档；表混合公开分数与作者重测。IQuest DeepSWE64.6、TB4 22.7，明显低于同表Opus5的73.7/53.9；内部IQuest-CLIBench不公开，不能独立复算用户体验优势。模型卡配方依赖IQuest专用SGLang/vLLM镜像与parser，HF自动生成的通用安装示例不能证明上游现成兼容。[许可](https://huggingface.co/IQuestLab/IQuest-Q1/blob/5c21b0630586ef77d38cff1094b8cf37a417fcd8/LICENSE)为**Modified MIT**，商用产品UI须展示IQuest-Q1，不能直接标标准MIT。**建议深入阅读并考虑复现**小模型的safe/optimal学习率转移、mid-training optimizer重置、跨harness故障归因与MOPD一致性；现有证据更支持方法研究，不支持泛化“便宜一半训练出同等agent”。

稳定ID：`arxiv:2610.30949v1`；既有权重基线`hf:IQuestLab/IQuest-Q1@5c21b0630586ef77d38cff1094b8cf37a417fcd8`。

### pplx-embed-v2-late 新技术报告：公开teacher训练、低差异采样与逐token表征蒸馏细节

**Perplexity；报告 `2610.29523v1` 于2026-10-21T10:10:26Z提交，72小时回溯补录；两模型卡10-22改为引用报告，权重未变。** [完整报告](https://arxiv.org/html/2610.29523v1)、[版本元数据](https://arxiv.org/abs/2610.29523v1)、[9B固定卡](https://huggingface.co/perplexity-ai/pplx-embed-v2-late-9b/blob/e6f9fd05b69356dec68b9b15443d3e963f3d2c09/README.md)。不是再报10-07的模型发布：本次新增可读训练机制与评估限制。18B teacher从Qwen3.5-27B裁去末20层，full-attention层去causal mask、GDN递归层保留；先pair InfoNCE再hard-negative triplet。186M pair按语言/域切为981子集，来源28数据集；实际训练采样配比为文本56.5%/自然图30.9%/文档图12.6%，不可把原始样本数比例当训练配方。以随机平移的黄金比例Kronecker序列映射分组CDF，让子集及父来源按权重且较均匀地贯穿训练；teacher LoRA rank128、对比batch16,384、gradient cache及不物化token相似度矩阵的Triton MaxSim核有明确披露。

两个student都对齐同一teacher的128维、L2归一化逐token向量，query/document位置分别平均再等权合并，使9B建索引、0.6B在线query成为可测配置。9B的末8文本层全参训练、其余LoRA；0.6B从Qwen3.5-0.8B每块三个GDN中裁两层，并全参训练。**这比笼统说“裁掉一半层”更具体**。作者称9B索引/0.6B查询在域均值较0.6B双侧升1.6个百分点，在线query编码器不变；不代表整个检索系统零额外代价，文档索引构建、多向量存储与MaxSim候选计算仍须计入。

限制同样值得读：§3.1明确72任务的完整清单未公开，81.3%/78.0%域均值不能独立复现；Q2D-Web为生产日志与LLM补充相关性判断，存在来源偏好。公开ViDoRe-v3图像检索9B/0.6B为65.2/62.3 nDCG@10，vision-only EVIE-8B为66.4，不能称所有检索模型第一。BrowseComp+采用gpt-oss-120B high-effort，9B准确率64.0%，对Reason-ModernColBERT高4.9个百分点；仍是作者实验，无独立重跑。训练总步数、学习率和总GPU预算未完整给出。权重MIT，报告文本许可另行区分。**建议深入阅读、考虑复现**公开ViDoRe与跨模型索引组合，再用自己的长文档/多语负载衡量质量、索引GB与query p95。

稳定ID：`arxiv:2610.29523v1`；卡修订 `hf:perplexity-ai/pplx-embed-v2-late-9b@e6f9fd05b69356dec68b9b15443d3e963f3d2c09:report-citation`，0.6B卡`7bba8e7cf2075dec747fa8557f41eb90e70361ed`。

### Ling-3.0-flash-Sante 补齐医疗域模型卡：权重10-19已有，本窗新增评测与部署披露

**inclusionAI；权重上传2026-10-19T17:52:04Z，本窗模型卡于10-22T14:53:04Z从仅许可头补为完整说明，固定`f74bd27bd26ffc71a50dd847805f8f7666ff0b43`。** [固定卡](https://huggingface.co/inclusionAI/Ling-3.0-flash-Sante/blob/f74bd27bd26ffc71a50dd847805f8f7666ff0b43/README.md)、[固定工件](https://huggingface.co/inclusionAI/Ling-3.0-flash-Sante/tree/f74bd27bd26ffc71a50dd847805f8f7666ff0b43)。对主窗前tree核对，只有README变化，27个权重分片未变。本轮收录的是首次完整技术披露及回溯的公开工件，不写10-22重新训练。卡述以Ling3 Flash为基座、124B/5.1B激活、KDA+gated MLA混合MoE、训练上下文至256K，面向医疗推理、文献研究和工具调用；医疗数据规模、来源、后训练/RL配方、训练算力和相对基座消融均未披露。

已实际打开卡中评测图，作者HealthBench Professional为54.3、MedXpertQA-Text51.0、MedMCQA80.3；同表GPT-5.6 Sol对应60.8/60.2/87.6，不能写医疗全面领先。AFUMED/AFUSAFE为内部评测，未给可复用题集；卡说所有模型用最高reasoning档，temperature1/top_p0.95；BrowseComp单agent采用256K、到40%上下文即discard-all，多agent是最多64子agent自研harness，两种设置不能混为模型单次推理能力。公开卡可读不等于这些内部benchmark可复现。

卡标MIT，给出Ling3专用SGLang dev镜像、4×141GB或8×80GB GPU配置及vLLM源安装示例；本轮未运行，不将示例视作性能/容量保证。**建议继续观察、按研究用途定向评测**相对原始Ling3的域增益、引用可靠性、多语言与长文档外推；当前披露不足以复现医疗域训练，也不足以推导临床能力。

稳定ID：`hf:inclusionAI/Ling-3.0-flash-Sante@f74bd27bd26ffc71a50dd847805f8f7666ff0b43:full-model-card`。



<a name="pretrain"></a>

## 预训练、架构与数据

### OpenLVD-200M 开放约1.98亿图文URL元数据：层级聚类均衡抽样可检查，图像与来源许可需分开

**TII；仓库10-21创建，10-22补齐分片，固定`e01081e09440254888b3cab0658e566fadb8217d`。** [数据卡](https://huggingface.co/datasets/tiiuae/OpenLVD-200m/blob/e01081e09440254888b3cab0658e566fadb8217d/README.md)、[工件](https://huggingface.co/datasets/tiiuae/OpenLVD-200m/tree/e01081e09440254888b3cab0658e566fadb8217d)。这是既有SigLino视觉基础模型研究所用数据的更新版开放工件，不是新SigLino论文，也不直接证明LLM训练增益。当前tree可核200个parquet；相对主窗前快照增加82个parquet及卡修订，卡报197,657,490行，仅托管URL/原始caption/hash/图像尺寸/来源/四层cluster/NSFW分数，不托管图像。

作者从COYO、DFN、Re-LAION抽200M，以DINOv3 ViT-L/16 CLS表示做四层球面hierarchical k-means，再均衡层级随机抽样；最顶层24,800簇全覆盖。NSFW阈值0.2过滤约2.34M行后剩197.66M，最终来源约115.59M/43.98M/38.08M。相比不透明的“高质量图文集”，其价值在可按来源/聚类重采样并核下载hash；但本轮未下载parquet内容或图像，未核URL存活率、图文匹配率、重复/污染与过滤误差，未有本次更新独立下游消融。

元数据选择/聚类标CC-BY4.0，DFN选择仍列Apple研究使用条款、Re-LAION为gated research来源，图像版权另归原持有人，不能把整包视为无附带条件的商用图像集。**建议多模态预训练团队深读并抽样复现**层级均衡、语种与领域覆盖、URL/hash可恢复性，再决定是否纳入视觉塔训练/蒸馏。

稳定ID：`hf-dataset:tiiuae/OpenLVD-200m@e01081e09440254888b3cab0658e566fadb8217d`。

### 平行语料预训练：先成对学习、再恢复单语采样，比一直拼译文更稳妥

**澳门大学、UCL、UNSW 等；2610.28838v1，原始提交 10 月 20 日 19:07 UTC，本轮补录。** 一手：[全文 §2–5、Tables 1–2](https://arxiv.org/html/2610.28838v1)。这项研究不增加语料，而是比较同一英法／英斯瓦希里语料的排列方式：SPT 独立采样原文与译文，PPT 将二者拼进同一 2,048-token 窗口、交替语言先后；再将 PPT 接 20B token SPT，减少训练时拥有译文、测试时只有单语的错配。

1.3B decoder、每语言约 150B 可用 token，基线从相同初始权重训至 200B；阶段实验从 60/100/160B 分叉，各续训 20B。英法 120B 的六任务双语均分 SPT/PPT/PPT→SPT 为 47.74/46.18/47.79，翻译 BLEU 为 19.25/14.23/34.03；英斯瓦希里 120B 均分 49.38→50.11，BLEU 9.07→13.18。**理解收益很小，主要稳定增量在翻译，反向 SPT→PPT 的斯瓦希里 BLEU 仅 1.54。** 同 token 不完全等优化：阶段模型重启 optimizer 并 rewarm，连续基线保原 schedule；仅两语言对、单规模，非现代大规模多语配方定论。建议深入阅读并复现相同 optimizer/reset 的对照；有机器翻译语料时先实验排列和末段分布适配，别把简单拼接当普遍增益。

### Looped Transformer 梯度诊断：测到丢失多少梯度，仍不能直接预测训练代价

**独立研究者 Aur Shalev Merin；2610.28871v1，10 月 20 日提交，本轮补录。** 一手：[§3–5、§7](https://arxiv.org/html/2610.28871v1)。论文用一次完整反传分解各循环对共享权重的梯度，精确计算仅保最后 k 次循环时被截断的份额；九个模型包括 Huginn、Parcae、Ouro 和两种 SmolLM2 转换，避免仅凭 Jacobian 最大奇异值或单一衰减率判断。

当循环数远大于 k，部分 Parcae 的丢失曲线从 32 到 64 循环变化约 ≤0.013，但拟合衰减率随拟合窗口漂移；Ouro 最后 2/4 循环保留反传仍丢最终输出梯度约 81%，**未测这对 Ouro 训练的实际损失**。四个模型 600-step GSM8K 文本微调、每配置五 seed 的主要负面结果是：跨两个新模型，梯度指标预测代价的平均误差 0.037，简单搬用 Parcae 同 k 代价为 0.036，未胜出。评测按相同步数，非相同计算；LR 仅为 full-backprop 调参，训练中指标还会下降。建议深读可重复诊断方法，但据自身模型和 optimizer 做短程实训选截断窗；别把“丢 80% 梯度”直接写成“损失 80% 训练收益”。

### CFDN 架构搜索：先修共享上下文、再修地址记忆，收益比 Agent 团队叙事更值得复核

**UCSB、Penn State、LinkedIn；2610.28887v1，10 月 20 日提交，本轮补录。** 一手：[§3、AppD1、Table2](https://arxiv.org/html/2610.28887v1)。16个带学科profile的研究角色、80iterations中63个候选训练，发现Context-First DeltaNet：在GDN2矩阵记忆旁添共享vector；先用预测误差更新vector，再让matrix拟合余差，分别承担所有query共享的上下文和query相关修正。不是每个研究者训练独立LLM，也不是80个成功实验。

FineWeb-Edu、按GDN参考FLOPs分配各架构token，150M搜索、1.3B验证。1.3B GDN2/CFDN token87.18/87.17B，WikiPPL16.05→15.74、八CSR52.65→54.53、NIAH66.86→67.29、真实检索32.52→33.41；不是搜索反馈里NIAH28.91→50.00那种大增幅。150M full-suite的CSR还略退。团队组织ablation未重复独立search seed，fresh-memory/uniform复用主运行头16轮，不能据单次utility胜出证明普遍组织最优。建议考虑复现CFDN本身及等训练budget/种子；优先检验架构新意，Agent团队提高研究生产率的外推继续观察。



<a name="posttrain"></a>

## 后训练与持续学习

### Delta N-gram：forward-only 写原生记忆可保数值稳定，但跨位置问答仍失败

**Illinois Institute of Technology；2610.31125v1，10 月 22 日提交。** 一手：[§3–9、AppA](https://arxiv.org/html/2610.31125v1)。冻住Qwen3.8-Flash-Next backbone、原生n-gram表与read-out，仅离线训4.1M writer；每个evidence block为被hash寻址的行生成target/gate，对同一行的事件先平均，再用delta rule写入稀疏patch。原表不变、同block共享旧快照、patch可单独保存／删除；有界性来自平均与有界target，不等于语义不会漂移。

100M raw-text训练、一个训练seed；两版本各4.1M参数：plain writer编码证据时只读原表，memory-aware delta writer还读取已写patch。**plain writer**的native continuation相对empty NLL减.068nats/token、相对donor减.024，16篇干扰后保93%总收益、55%文档特异收益。固定T⁰-only编码的同batch对照中，delta weights的文档特异收益保留约70%，plain约54%，但delta的self与donor绝对NLL都变差，不能只凭特异差距增大宣称更好。两writer均35K连续写无数值故障，**七形态任务净acquisition却为负；delta writer在LoCoMo categories1–4共1,540题上，self/empty/donor都同答拒绝，F1 .44（0–100标度），完整对话入context为56.79**；低遗忘不能掩盖未学会。A800/H200状态相差4–6%，一般输入drift仍不可由数值有界性保证。建议深入阅读可归因patch设计并复现跨位置读取；目前适合同位置continuation记忆研究，不能当可用长期agent memory或已解决持续学习。

### MCPD：对同一参考随机遮挡四次，再平均teacher概率，降低完整答案造成的目标敏感性

**Khouloud Saadi、Di Wang；2610.30880v1，10 月 22 日提交，论文HTML未披露机构。** 一手：[Method、Experiments、AppC–D](https://arxiv.org/html/2610.30880v1)。冻结初始模型作privileged teacher，保持学生实际prefix不变，只随机mask参考解的30%/40%token，四个teacher分布在**概率空间平均**。该分布是期望view的forward-KL最优目标，不代表所有KL方向或任意mask都最优，也不消除reference错误。

Qwen3-4/8/14B、R1-distill7B、OPSD同17K OpenThoughts、LoRA64、100steps、response≤1024；相对OPSD，Qwen4/8B AIME25约+2.3/+2.0pp，四backbone均分最高。4×A10080GB完整run比OPSD多10min约11.6%；**GRPO500steps/8rollouts对MCPD100steps，所谓7.8×训练省时不是等compute算法胜出**。学生no-thinking、teacher启用thinking，配方与更长reasoning部署有距离；未见多训练seed统计。建议考虑复现reference改写稳定性和同teacher-token预算的K1/K4/entropy-matched对照。

### GIFT：在logit空间筛选蒸馏token、逐步撤掉teacher，与参数梯度投影是不同干预

**上海交大、腾讯；2610.30796v1，10 月 22 日提交。** 一手：[§3–4、AppD–I](https://arxiv.org/html/2610.30796v1)。把每token蒸馏logit梯度分解为相对GRPO梯度的平行与正交分量，先少选冲突、再偏好正交新信息，在sub-batch内按精确quota保留；teacher覆盖逐步退到零，保留token的loss权重不重新放大。它是局部logit几何与筛选定理，不能据此保证共享参数经AdamW更新后全局RL loss不升；有quota时仍可纳少量最弱冲突token。

Qwen1.7/4/8B、DeepMath难度分层和KK，数学mean@12均分base→GIFT为51.65→54.25、74.67→77.20、76.67→78.96；4B最强所列LUFFY76.85，所以不是比全部baseline都高3.39pp，3.39%是相对base增幅。简化conflict-only在部分较易题更强，AIME25支持完整规则；末段可跳过teacher，但45.3% nominal supervision不等于45.3%wallclock。建议考虑复现与CORD/naive联合loss的等预算比较，重点测teacher成本、零advantage题与筛选范围；当前不宜称普遍最优训练整合。

### Reverse-KL蒸馏理论：在线数据不自动解决访问分布错配，完整teacher分布能改变样本复杂度

**University of Virginia、Google Research等；2610.30844v1，10 月 22 日提交。** 一手：[§2–4、Theorems3.1–3.5/4.2](https://arxiv.org/html/2610.30844v1)。有限horizon、可能misspecified的候选policy类中，即使每轮精确拟合当前学生访问状态的局部reverse-KL，也会偏爱错误访问分布而卡住。作者用prefix importance ratio截到1、累积局部loss也截到1，并在policy类上做exponential weighting，得到相应二阶／轨迹likelihood-ratio条件下的最优阶保证；offline full-feedback也需clip，原始ERM的逼近代价可多一阶。

这是**统计理论而非可直接替换大模型optimizer的实现**：算法需比较有限policy类，full-feedback是访问state上teacher全部action分布，action-only只知已采样动作概率。定理区分两者及假设强弱；不能从offline在某假设下可更省样本，推成“on-policy总是不必要”。论文未给现代LLM训练吞吐／效果实验。建议深入阅读以设计OPD的分布错配诊断、支持覆盖与teacher反馈接口；考虑复现小MDP反例，暂不将样本界等同工程cost优势。

### DConstrain：把schema上下文留到运行时，避免任意字段顺序在冷编译阶段爆炸

**香港浸会大学、Princeton、蚂蚁集团；2610.30920v1，10 月 22 日提交。** 一手：[§4–5、AppB/D5](https://arxiv.org/html/2610.30920v1)。编译context-parameterized transition program，把stack/已用字段等运行时状态保为symbolic参数，warm阶段只实例化实际可达后继，再合并同token-effect；适配多种joint decoder，CPU warm编译与GPU forward重叠。它不改schema语义，但并未消除任意深层grammar推理的最坏指数复杂度。

LLaDA2.2/SGLang块32、DiffusionGemma/vLLM块256、H20-3e；BFCL完整2,495题的strict-schema validity100%，任意字段顺序相对固定顺序semantic accuracy72.71→81.60%、37.27→81.56%。冷编译同fixed-depth4对优化DINGO快3.88–4.83×；**TPS/延迟大倍数取双方都成功的cohort，Any-order与fixed-order的cohort还不同，不能横向串算**。JSB仍有13.7%因不支持oneOf exclusivity/general allOf等、4.8%因预算而失败；非全部JSON Schema支持。建议深入阅读接口和编译边界，复现相同success cohort、字段顺序与semantic质量；项目是否上游合入需另核代码。

### Gandalf：混合attention的复用差异可少写SSD，寿命数字来自TBW模型

**Chicago、TensorMesh、Samsung；2610.30791v1，10 月 22 日提交。** 一手：[§3–5](https://arxiv.org/html/2610.30791v1)。full-attention历史片段与linear recurrent state复用模式不同，统一spill-to-SSD会把大量不会再用的linear状态反复写盘；Gandalf先按attention type分层，再按省重算收益与写wear费用选择DRAM保留、SSD迁移或丢弃。它优化持久cache的存储写量，不是decode每token读HBM的KV压缩。

KimiK3约72%linear层、100GB DRAM+1TB SSD，Bailian/ClaudeCode/AIPerf trace replay；最低miss点SSD写量比最低miss baseline少2.0/3.3/6.5×。**寿命由TBW÷每日重复trace写量估算，WAF假设1，不能说真实SSD已运行多年；延迟也是10GB/s SSD、50GB/s PCIe与60Ktok/s重算的重叠模型**。ClaudeCode某点寿命17对4月、加载.78对.66s，明确以额外重算/读取换wear。baseline为适配两层所扩展，端到端发布集成未本轮验证。建议考虑复现真实SSD SMART/NAND写放大与业务trace，用寿命—miss—时延曲线选策略，不引用单一7.3×平均当硬件保证。

### EPIC：按整个 prefix 校准 clipping，在深度复用 rollout 下优于 token／sequence clipping

**清华 AIR、字节 Seed SIA-Lab；2610.29999v1，10 月 21 日提交，本轮补录。** 一手：[§3–4、App7–10](https://arxiv.org/html/2610.29999v1)。从 autoregressive MDP 的性能差异式出发，用累积 prefix importance ratio 而非当前 token ratio；通过精确 KL 的非负坐标和行为 prefix 几何平均概率校准上下限。实现再对 ratio 取长度归一化几何均值，straight-through 只让当前 token 回传，避免整个 prefix 梯度相乘；O(T) cumsum，不额外跑模型。

Qwen3-30B-A3B-Base、DAPO-Math17K、32×A100、verl/Megatron/vLLM、300steps、每题8响应、16K response cap，一 rollout batch 分8次更新；三独立训练 seed、每5steps测评并按四套题均分选单 checkpoint。四套数学 mean@8 均分 EPIC39.34，GRPO28.65、GSPO32.84。**130道题同时用于 checkpoint selection，无独立外层测试；默认 clipping 对新方法调参公平性仍有限，且只一个模型和高复用配方。** 理论下界对应精确目标；几何均值、straight-through 与 clipping 是实用近似，不能声称实际优化每步保证单调或满足 hard KL。driver RSS 较 routing replay 低是 CPU 进程观测，不是 GPU 显存。建议深入阅读并复现不同 update-depth、异步程度和全套调参；当前不足以普遍替代 MoE routing replay。

### CORD：RL 与自蒸馏冲突时只投影一侧，保护方向和 loss 权重需要一起选

**港中文、腾讯、北大；2610.29965v1，10 月 21 日提交，本轮补录。** 一手：[§3–4、Table1、AppB–C](https://arxiv.org/html/2610.29965v1)。在已有 GRPO+OPD/Skill-SD/SDAR 上分别取梯度，保留一侧作为保护方向，只在内积为负时把另一侧投影到其正交平面；几何来自 A-GEM 单约束，区别于 PCGrad 同时改两侧。保证仅是固定 minibatch 上的原始梯度一阶下降，AdamW、动量和有限步长后不等于任务成功率保证。

Qwen2.5-3B-Instruct、8×A10040GB、150updates/8rollouts、ALFWorld≤50动作、WebShop≤15动作。GRPO+OPSD 的 ALFWorld success65.0→80.9，但 WebShop64.8→64.5且score80.5→77.5；其余两配方在两环境均增。**一个训练 seed、四次 decode 评测，±不是训练方差**；外引 WebShop128题 baseline 与本作500题不同，应优先看相同配方 Base–CORD 增量。保护 distillation 时总是投影可把某配置 success58.35降至38.90，权重／目标／batch aggregation 交互很大。建议小规模复现并联合扫保护方向、gate、loss与权重；不把投影包装成通用自动冲突消除器。

### LongZero：先造可校验的虚构世界，再扩成长文，可用模型自己生成长上下文 RL 数据

**清华；2610.30132v1，10 月 21 日提交，本轮补录。** 一手：[§3–4、Table2–3、AppA](https://arxiv.org/html/2610.30132v1)。模型先写实体／事实／3–5跳问题的结构化 spec，程序核对 supporting-fact ID、分配到8篇文档再扩写，混入易混淆实体；另合成多轮 MRCR 类检索任务，筛掉始终答对／答错题，以2:1混合做GRPO。所谓 zero data 指不需外部文档／强 teacher，仍用预设 domain/style/noun 表与大量自产采样，并非无先验或无成本。

4B-Thinking与8B各2,304 reasoning+1,152 retrieval训练样本；8×A80080GB、约213/230GPUh，五benchmark均分54.68→57.42和44.63→45.66，4B高于 teacher-synth baseline55.99。**后者文档长度和构造方式并不匹配，不能只归因 teacher 大小。** 检索单独训练反而降分；8B LBv2下降，AA-LCR的200响应有120个因131K总窗等原因没有答案。扩写的语义忠实性不由字符串存在检查完全保证；原文未提供多训练seed统计。建议深入阅读结构化答案构造和困难度过滤，复现时审计生成事实一致性、计入合成成本，并避免把特定窗口评测误当长程推理上限。

### CSD：跨 tokenizer 蒸馏应匹配同一文本事件，残余概率保留未匹配质量

**Mingyu Yang、Pavlo Molchanov 等；2610.28895v1，10 月 20 日提交，本轮补录。** 一手：[§3–4、AppA/C/E](https://arxiv.org/html/2610.28895v1)。用字符offset寻找拥有相同历史文本的chunk；一对多／多对一通过固定已观测prefix、改变最后token构造多个同文本事件，概率用chain rule相乘，不做长度归一化；再将所有未匹配质量汇入residual事件，在同一归一化分布上做JSD。多对多只保观测事件+residual，是明确的计算折中，不是完整跨词表等价分布。

Llama3.2/Gemma3 1B学生、Qwen3-4B/Phi4-mini teacher、Nemotron-ClimbMix、30Ksteps/batch768/context4096；四种1B配对相对X-Token五任务均分改善0.66–5.38pp，不需随配对切换loss。8B学生2Ksteps也优于CPT/ALM。256×H100、32nodes的**step计时**里，teacher top128对full-logit CSD为3.66s对7.45s，均分33.59对33.84；不是相对全部蒸馏算法均快2倍，ALM仍略快。未给跨训练seed置信，larger teacher能力／数据仍影响增益；目前是离线共享文本蒸馏，跨tokenizer在线OPD未验证。建议深入阅读support构造和数值守恒，复现时审计Unicode、特殊token、代码空白、many-to-many比例及teacher总成本。

### LIFT：把内容要求纳入指令奖励，可减少“格式全对、正文为空”的训练捷径

**AI21 Labs；2610.28819v1，10 月 20 日提交，本轮补录。** 一手：[§3–6、AppC/J](https://arxiv.org/html/2610.28819v1)。将约束绑定到文档某section/表格/代码块，分别用程序或Gemma4-31B判定，结构占10%、body占90%且包含内容要求；同topic/outline下改变局部constraint density，避免将约束总数和局部负载混成一种难度。

17模型、每例8次生成；305个人标constraint-response对上，scoped judge的macro-F1约.73，scope本身比换judge多11–19pp。训练对照中IFTrain奖励到1.0同时可见长度114→28.5→18词，出现空标题／无意义约束token；LIFT-Train同base未见该型坍缩。AdvancedIF的base/IFTrain/LIFT为.392/.374/.401，ComplexBench .324/.316/.327，**外部分布增量小，不能只看自家LIFT .423→.822的大幅提升**。flat-only控制仍无法消除topic差异，.73 judge也有误判；生成器未全局检查约束冲突。建议深入阅读reward hacking样例并复现内容门槛；上线需人工抽验scoped裁判，不能把“有内容判分”当已杜绝投机。

### ReAPO：失败后带 verifier 反馈重试，训练目标仍落在第一次作答能力

**Columbia、Yale、UT Austin；2610.28770v1，10 月 20 日提交，本轮补录。** 一手：[§3–4、AppC](https://arxiv.org/html/2610.28770v1)。初始G条rollout中只对失败者追加反馈和最多K次重试，把每一次attempt自己的奖励归给自身token，不将第二次成功奖励广播给失败前缀。SSP推导还假设失败历史可由奖励充分概括、未来失败代价与attempt位置无关，作者承认这是简化，不能视为一般多轮策略梯度恒等式。

Qwen.6/1.7/4B与OLMo1B，G16/K2，GSM8K或Polaris53K，累计每轨迹2K/4K response cap；**评测仅第一次回答、没有verifier**，mean@32是32次平均正确率。小模型里达到GRPO双group最佳accuracy省39%生成token、双训练steps最佳accuracy省31%；GRPO在训练早期按token/time仍有竞争力，所以不是每阶段都提速。SCoRe仅StageII且报第二次回答，不能写成完整SCoRe公平胜出。建议考虑复现困难度自适应采样和attempt credit，记录生成总token/时间与pass@K；部署没有答案verifier时，不要求复用训练时重试流程。

### EMPO：同一可执行动作的多种措辞共享 ratio，减少文字实现与动作 credit 错位

**天津大学、复旦、人大等；2610.29427v1，10 月 21 日提交，本轮补录。** 一手：[§4–5、AppC–D](https://arxiv.org/html/2610.29427v1)。将batch中上下文可比且解析到相同执行effect的响应分组；先按响应长度归一化ratio，再在至少两响应的cell中平均共享，仍保每个响应原来的GRPO/GiGPO advantage，无需新critic或额外环境交互。孤立、无效、截断响应用原token ratio回退。

1.5B/7B文本与3B-VL、ALFWorld/WebShop/Sokoban、三训练seed、同base配置；1.5B GRPO→EMPO ALFWorld69.5→83.3%、WebShop62.8→70.3%，GiGPO→EMPO94.0%/77.3%对90.1%/73.8%；VL Sokoban72.7→84.4%。可用cell覆盖ALFWorld64.6→79.5%，WebShop后段仅23.4%，收益依赖重复可比effect的出现。理论是局部分组方向，不保证含自然语言参数的任意tool call等价；context/effect key分错会耦合无关动作。建议考虑复现已有明确动作parser的环境，先验effect cell的边界与覆盖，再用于开放式工具／SWE。

### Uni-Mem：建设与检索分阶段 online RL，评测协议变化足以改写 baseline 分数

**华为诺亚、东南、中山等；2610.29671v1，10 月 21 日提交，本轮补录。** 一手：[§4–5、AppC/E/F](https://arxiv.org/html/2610.29671v1)。单Qwen3-4B manager多轮操作Mem0/Qdrant，建设与检索交替更新；用原始证据映射与更新／删除provenance定位应负责的建设event，错误QA再用已知证据重答，区分建设失败与检索失败。该归因使用训练证据标注，不是部署时免费获得的oracle。

LoCoMo train/val/test2:1:7、GPT-OSS120B回答/GPT4o-mini评判，SFT起点83.56，标准GRPO84.02，组合88.80；无SFT76.58→80.99。Memory-R2(OSS)换Mem0-aligned judge prompt重评87.10→77.87，**不能拿原87.10和本轮88.80当同协议1.7pp比较，也不能把重评分差当模型退化**。大部分其他baseline沿引旧协议；SFT仅4个自建会话任务，RL200steps/8rollouts，训练Qwen3.8-27B兼作回答和judge，成本高且不同于最终评测模型。建议深读归因与协议敏感性，先复现相同harness/SFT起点与新旧judge并行评分；公开主表不足以把全部优势归给stage-decoupled GRPO。

### FlashNext-RL：长轨迹 RL 先解 offload 和状态生命周期，不能把开发迭代差值当同配方倍速

**Snowflake AI Research；2610.28998v1，10 月 20 日提交，本轮补录。** 一手：[§3–5、Tables10–16](https://arxiv.org/html/2610.28998v1)。Qwen3.8-Flash-Next的大MoE与冻结n-gram表使优化重点先落在CPU offload：参数跨microstep驻留、expert梯度留GPU，再换MoK通信／融合residual／原生GQA sparse attention；权重每层gather、重排expert，以≤1GiB bucket发给rollout engine，receiver用clone避免view持有全部receive buffer。

8层受控ladder从2,391到17,678 REAL tok/s/GPU（7.4×），baseline已有必需correctness fixes；全48层395→4,438的开发记录还改变bin/batch，**不能报11倍受控加速**。四个8×B200节点的live SWE RL，step5–15平均22.4min/iter，trainer约1,013s、等rollout331s，约1,393采样中1,024个可训；实训4,737 packed tok/s/GPU包含padding，与REAL token不同。短run entropy和mismatch KL稳定，未证明长期收敛或任务质量。8–11训练节点节省只是理想线性外推；本地NVMe DCP有节点丢失风险。建议深入阅读系统记账、峰值host-memory和广播生命周期，在同shape复现增量；未验证论文全部优化已在上游release提供。



<a name="inference"></a>

## 推理优化与系统研究

### Hy-Memory 官方新增专家低比特GGUF：8.03GB工件可核，但证据只覆盖固定20% LoCoMo

**腾讯混元；2026-10-22T06:50Z新增Q3_K_M及两款专家混合IQ工件，固定`38e10a37701e1caff4a6dc3831448aaf8c2e15d1`。** [固定量化卡](https://huggingface.co/tencent/Hy-Memory-30B-A3B-GGUF/blob/38e10a37701e1caff4a6dc3831448aaf8c2e15d1/README.md)、[目录](https://huggingface.co/tencent/Hy-Memory-30B-A3B-GGUF/tree/38e10a37701e1caff4a6dc3831448aaf8c2e15d1)。本次不是重报已收录的记忆writer训练：新增下载体积8.03GB的`Q8_0-expsIQ1_M`、9.14GB的`Q8_0-expsIQ2_XXS`及14.45GB Q3_K_M。方法是在llama.cpp既有格式上用imatrix校准及MoE分配精度，路由专家压IQ，attention、共享专家、embedding/output留Q8_0；不是全模型1-bit/2-bit，也不是专家裁剪。

卡报告在Mem0 2.0.19、固定20% LoCoMo（2用户/262题/59记忆chunk）、top_k10、no-think、F16 KV、llama.cpp `5aeac37a35b44fde4bdffaec9159f2fcb253b622`、GPT-5.6-Sol answer/judge的单轮结果：Q6_K58.78%(154/262)，8.03GB混合款57.25%(150/262)，约差1.53个百分点；9.14GB款55.73%，并未随位数单调提升。没有多种子/置信区间或完整全量LoCoMo，不能据此证明无损或所有记忆框架性能不变。下载体积也不含KV与runtime显存。

卡还给出特定第三方Q1/Q2文件在H20/同llama.cpp/8个基础JSON记忆写入提示下失败的作者负面检查，**只适用于所列文件与环境**，不能扩成所有低比特量化失败。Apache2.0。**建议考虑复现**自己记忆写入/更新/删除工作流，先检查JSON有效率和bank质量，再测完整检索问答；不要仅以能加载或更小文件判断量化可用性。

稳定ID：`hf:tencent/Hy-Memory-30B-A3B-GGUF@38e10a37701e1caff4a6dc3831448aaf8c2e15d1:expert-mixed-gguf`。

### Nemotron3.5 SuperVL修复remote-code权重映射与非正方视频格点：应更新部署基线

**NVIDIA；2026-10-22T17:24Z，两款BF16仓同步修订：instruct `fb0ee0bc6b3f70342afd8d84cb80dbadbaa8a1fe`、base `5d80455a9bf1305b898a1088f8824b02ba451c93`。** [具体修订代码](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16/blob/fb0ee0bc6b3f70342afd8d84cb80dbadbaa8a1fe/modeling_nemotron_h_omni.py)。主窗前后tree比较仅processor/preprocessor配置与`modeling_nemotron_h_omni.py`变化，权重未换。

`_register_checkpoint_conversion_mapping`现在同时用model_type和类名注册映射。维护者注释解释：Transformers≥5.18原生port与remote class同名，原生类名映射会优先把`mlp1.*`转去remote model不存在的`multi_modal_projector.*`，令真正vision projector随机初始化；本次覆盖两key以保持`vision_projector.mlp1.*`等目标名。另将视频`_project(...,keep_grid=True)`保留真实 `(T,H,W,hidden)`，pruning直接用前三维，移除由token数开平方猜正方形的路径。processor还补齐16kHz feature-extractor配置并删旧downsample键。这些是影响多模态正确性的具体代码变化，不是新模型能力。

**建议现有remote-code部署者优先采用新固定revision并回归**图片/非正方视频、checkpoint missing/unexpected-key告警与已知问答。维护者明确该全局映射覆盖会影响同进程后续加载原生port，应避免混载；本轮只审差异，未在GPU上复现旧错误或量化准确率修复幅度。

稳定ID：`hf:nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16@fb0ee0bc6b3f70342afd8d84cb80dbadbaa8a1fe:processor-video-fix`；base修订如上。

### Decisions API正式可用：把频繁的有界决策从开放生成热路径拆出

**OpenAI；2026-10-22正式发布。** [发布说明](https://openai.com/index/decisions-api/)、[已打开的官方开发文档](https://developers.openai.com/api/docs/guides/decisions)、[原生语音集成](https://developers.openai.com/api/docs/guides/decisions-voice)。相对public beta的具体增量为共享输入prompt cache、原生JSON、题目图片、每请求独立问题上限200→400、GPT-Live原生触发，以及US/EU推理驻留；不称新基础模型。`POST /v1/decisions`目前只支持`gpt-6-luna`，返回predicate概率、固定choice或rubric score；tool call/history只作证据，动作仍由应用执行，文件/音频/item引用不直接支持。

工程价值是将分类、路由、下一步选择等有界输出独立调度。共享input前缀至少1,024token才可缓存，questions（含题目图片）不在可复用前缀内；400题是共享证据上的独立问题，依赖前一题答案的链须分请求。发布方称对同GPT-6 Luna Responses API最高10×速度，但未给统一公开硬件/负载/输入长度基准，客户5×/40×等证词不能混作通用实测。底层推理优化机制、模型架构和可复现实验未披露；概率输出也不能自行当作经过业务校准的置信度。

**建议考虑小规模采用评测**频繁低时延路由，比较同输入/固定选项的错误率、拒答/低置信处理、p50/p95、缓存命中与总调用成本；其适用性取决于任务是否能写成有界选择，不能据发布倍率推出整个agent加速。

稳定ID：`openai:decisions-api:ga:2026-10-22`。

### RIFT：draft 的 prefill KV 会随长输出过时，刷新与可压缩性训练需一起考虑

**NVIDIA；2610.31098v1，10 月 22 日提交。** 一手：[§3–5、AppD/E/J、Limitations](https://arxiv.org/html/2610.31098v1)。针对外部autoregressive drafter与KV型self-speculation，把历史重要token和recent sliding-window配额结合，再周期或按acceptance/entropy/query drift刷新重要KV；训练时给drafter施加类似压缩mask，适配实际缺失的上下文，避免仅训练dense再硬压。

SPEED-Bench8任务、draft length11、output≤4K、T0/1、active draft KV8K。Qwen3.5-4B→122B、100K input/low-entropy split/T0时，保留full-KV acceptance length的SnapKV59.7%，RIFT静态85.6%、周期95.5%、自适应96.0%；32K/4K/8K结合训练可保>98%。**这些是AL保留率，不是端到端speedup。** 刷新仍假定完整历史驻GPU或offload，总系统KV没有压到8K；AppD2将forced-AL vLLM replay测得的decode延迟加上估算重评分耗时，投影开销最高约6%，**不是完整RIFT的实测端到端开销**；未来SRAM设备成本仍待测。tail-augmented MTP/EAGLE/DFlash未纳主验证，代码部署性能待验证。建议深入阅读刷新成本与完整历史边界，复现时同时报系统总内存、AL、refresh次数和实际延迟。

### ProQuant：极低平均位宽时，删掉被破坏的专家可能比保留一位专家更好

**上海交大；2610.30779v1，10 月 22 日提交。** 一手：[§3–4、AppB/C](https://arxiv.org/html/2610.30779v1)。用router score加权的expert输出误差、按被路由token数归一化，避免校准集activation frequency主导重要性；ILP在{0,2,3}bit与最大剪枝率下分配预算。再只训少量RMSNorm，逆向缩放router输入权重，抵消由norm重标定导致的router logits改变；该等价保证限同一hidden输入，不能消除前层量化累计误差。

Qwen3-30B-A3B与Qwen3.5-35B-A3B、2×A800、C4 128×2048校准，expert平均1.5/2/2.5bit，QKVO4bit、router等保高精度。1.5bit六生成任务均分67.30/60.85；另在Qwen3-30B-A3B-Instruct-2507的2.0bit expert平均预算下，{0,2,3}消融74.07，对{1,2,3}68.46和{0,1,2,3}66.98，说明收益不只是剪枝。30B存储估61.06→8.09GB、量化耗时约GPTQ1.21×。**expert平均bit不是整模bit；存储估算非实际kernel吞吐，未验证数百B规模或多训练seed。** 建议考虑复现真实生成而非仅PPL，并验证剪枝后的top-K替补路由、packing和部署kernel。

### T²Quant：扩散模型应按任务／去噪步分配activation位宽，同平均bit不等同硬件成本

**Technion、Ben-Gurion University INSIGHT；2610.31198v1，10 月 22 日提交。** 一手：[§3–5、AppE2](https://arxiv.org/html/2610.31198v1)。先测局部量化KL与对后续预测的影响，用无标签任务prompt估precision收益，再在16个时间窗、{3,4,6,8}bit上做固定平均4.00bit预算分配；保留原quantizer和sampler，避免把整个denoise过程当同等敏感。

Dream7B、LLaDA1.5/8B、六任务，三种基线共九组平均分都提高，42/54单任务改善；相对最强独立基线均分+3.55/+1.13/+1.40pp，最高6.7pp指特定baseline增量。关键条件：**全部为BF16 arithmetic模拟整数值，未测低bitkernel速度／能耗**；4token/step的schedule到2仍有收益，到8收益消失。random对照只一个固定schedule draw，不能概括所有随机策略。建议观察／复现任务分布转移和真实混合bit执行成本；不能将平均bit预算解释为相同FLOPs或已实现服务加速。

### SetCommit：协调一次提交哪些位置，给尚缺上下文的token留下修订机会

**北航、港大、北大；2610.31120v1，10 月 22 日提交。** 一手：[§3–4、AppC/E/F](https://arxiv.org/html/2610.31120v1)。从host的高置信集合出发，在候选中综合top2 margin与相邻不确定token的共提交风险，保留一定baseline比例；再用当前posterior检查旧commit，有限remask并多提交相同数量，维持host每步net quota，不增forward。它修改解码策略、非模型训练。

LLaDA8B/Dream7B、GSM8K/MATH500/MBPP/HumanEval，32NFE质量增4.9–7.4pp；相同质量目标取native128分数减3pp，配对L20计入选择/remask/cache-refresh，geomean增量加速1.44–1.74×。**Dream模块同时改为确定性top1，因此是完整pipeline对比，不能把其收益全归选择位置；EB亦为重实现且独立policy。** 同状态干预支持位置集合会影响后续质量，但收益随进度并非单调。建议考虑复现原host配对、confirmation split和实际延迟，避免仅以NFE代替总成本。

### R²D：随机remask让冻结的错误可回退，逻辑题的缩放不等于开放域生成law

**Carnegie Mellon University；2610.30744v1，10 月 22 日提交。** 一手：[§4–5、AppA/B](https://arxiv.org/html/2610.30744v1)。把clean序列上的remask→denoise视为Markov kernel，同noise level迭代到argmax序列连续τ步不变，再降noise；理论收缩受denoiser误差floor限制。可直接用pretrained MDM，另有TCT在自己生成trajectory的已揭示和masked位置都算CE，专训纠错，二者贡献要区分。

6.8M小模型、Sudoku-Hard和3SAT，TCT+R²D97.5%/99.3%；同TCT和预算下比ReMDM correction高5.2/2.8pp，early-stop平均NFE约其1/4和1/2。未训TCT时Sudoku M1→256为19.7→93.5%，近log线性仅观察区间。LLaDA2.1-mini16B HumanEval+85.4%，高于引用的官方Q-mode82.9%，**后者非本次统一运行，真实code测试仍需控制采样/长度/硬件**。建议考虑复现训练与sampler两维交叉消融，报告早停实际NFE/时间；token不再变化只是heuristic，不是正确性证书。

### PACE：减少循环状态的反复四位舍入，比仅设计量化尺度更关键

**Patrick Haller、Filipe Laitenberger、Alan Akbik；2610.29545v1，10 月 21 日提交，本轮补录。** 一手：[§3–7、Table 1、Limitations](https://arxiv.org/html/2610.29545v1)。Gated Delta 状态每生成一个 token 就写回并舍入，会累积误差；PACE 将最近 token 的 key/value 放入 decode buffer，每 8 token 更新并量化一次状态，沿 key 轴分组、让同一 value 列共用尺度，再将误差衰减最慢的四分之一 head 保为 8bit。它不训练模型；代价包含 buffer 与尺度，不能只报四位 payload。

四个 Qwen3.5/3.6 hybrid、greedy/no-thinking、五项任务；PACE-4/8 实驻 6.82 bit/value，对 DAMP INT8 9.90。9B 均分同为 83.7；27B 87.3 对 87.4；35B-A3B 85.0 对 86.3；4B 79.1 对 80.9，仍有损失。吞吐只测 **uniform INT4+int8 buffer**，单 H100 PCIe、512 context、batch 1–128，batch≥16 TPOT 优于逐 token BF16，**大部分速度来自 BF16 buffer 本身，完整 PACE 的分组与 mixed-head 未计时**。建议深入阅读并复现 resident-memory/质量/TPOT 三联对照；暂不将其外推到纯 recurrent、Mamba 或超长输入，prefill 固定 256-token chunk 的舍入频率也未消融。

### 2:4 与量化的同成本比较：先降精度通常更合算，最低位宽上的稀疏另当别论

**独立研究者 Xiuyi Qin；2610.28947v1，10 月 20 日提交，本轮补录。** 一手：[§3–6、Tables 2/5](https://arxiv.org/html/2610.28947v1)。研究把 index/scale/zero-point 都算入存储，并分别比较同 bit 与同 tensor-core 理论吞吐档位。2:4 INT8 的实际存储接近 dense INT5，不能直接按“砍半参数”估算；在 Llama3.1-8B、Qwen3-8B 六类操作数、逾 220K 配置中，FP16 稀疏的算子误差为 dense FP8 的 5–16 倍，且在所测 GPU 的同档位通常更慢。

**结论限定在尚可继续降精度的区域。** 已到 FP8 floor 时，稀疏重训练可再获益：单 H100/vLLM、64×2,048 prefill、256 decode，dense FP8→稀疏 FP8 的 prefill 58.9→74.6 Ktok/s、batch64 decode 8,556→9,914 tok/s，权重 8.5→6.1GiB，但使用不同的已重训练 Sparse-Llama 权重；GSM8K 差异不显著。Blackwell sparse NVFP4 实为相邻对的 4:8 格式，不能沿用 elementwise 2:4 权重；1B token 蒸馏恢复了大部分 perplexity，却只恢复约四分之一丢失的 GSM8K 能力。建议深入阅读格式核验和同成本基线；先试 dense 低精度，再预算稀疏重训，恢复终点需看 reasoning 而非只看 PPL。本文未做独立重跑。

### Looped UDLM：在 denoiser 内复用层可省训练 FLOPs，生成的等 NFE 比较并不等计算

**MBZUAI、Sony、MIT；2610.30066v1，10 月 21 日提交，本轮补录。** 一手：[§3–5、AppC/F–H](https://arxiv.org/html/2610.30066v1)。保持 Duo++ 的 uniform corruption、训练目标和 sampler，只循环两层中间 block、每步抽循环次数；比较均匀、按FLOP比例和噪声相关的抽样，推理可按 hidden-state 相对变化 <0.1 提前停。

170M、12blocks/width768、1,024序列、OWT与FineWeb-Edu10BT、batch512；以估算训练FLOPs对齐时达到非循环模型 NLL 需2.7–3.3倍更少 FLOPs。噪声越高多循环的直觉未获支持，noise-proportional 比另两分布差约 .029NLL；93–94%的NLL改善来自更好预测被替换token，非仅辨认污染位置。推理 S6/S12 每次denoiser成本分别1.48×/2.05×，**同NFE的最高9.8%生成PPL下降不等于同FLOPs加速**；FLOP-proportional 的 MCQ 更好，却平均生成PPL差1.55%。目前小模型和 likelihood proxy，无大模型服务时延验证。建议考虑复现训练效率，分别画训练FLOPs与实际采样FLOPs/时间的Pareto，不用统一“更深更好”解释任务差异。

### UlaCache：稳定逻辑 KV 地址把 GPU/CPU 历史存储与计算副本分开，增加可入场并发

**Scitix.AI；2610.29820v1，10 月 21 日提交，本轮补录。** 一手：[§3–6](https://arxiv.org/html/2610.29820v1)。稀疏attention只读取少量历史，却仍需保留未来可能选择的全部KV。UlaCache为历史分配稳定逻辑地址，用mapping追踪GPU/CPU物理slot；TP按rank分存完整历史，选中KV再入本地计算buffer，CPU扩展用parallel bucketed LRU与bitmap制定交换，完成传输后才更新mapping。这样GPU池也是历史容量，不像HiSparse只作CPU历史的缓存副本。

GLM5.3、FP8权重/KV与BF16activation、8×H200或8×B300，CPU预算每GPU128/192GB；Multi-LexSum64–128K输入、平均98,125token、greedy2048输出，全部prefill完再decode。B300 TP无spec的峰值吞吐为SGLang峰值3.51×；有2步MTP时相对HiSparse为H200 DP/TP1.16/1.70×、B3001.73/2.51×。**各系统各取自己的最佳batch，不是同并发kernel快3.51倍；BS1仍慢于GPU驻留SGLang。** 主要收益来自CPU与GPU合并容量、去副本后容纳更大batch；未测在线到达/排队/TTFT与SLO，也非跨节点任意扩展证明。建议考虑复现容量—吞吐—TPOT曲线，并按真实长短混合负载验证；本轮只核论文，SGLang集成发布状态需另看已合入代码。

### AKV Cache Engineering：直接删历史 KV 与重排 RoPE，可避免改文本后重算剩余上下文

**清华等；2610.28577v1，10 月 20 日提交，本轮补录。** 一手：[§3–5、AppA1.4](https://arxiv.org/html/2610.28577v1)。提供append/drop/repos三操作：drop只移除选中KV，保留其余hidden-state计算历史；repos旋转保留key的RoPE并重置后续位置。由于保留状态仍携带过去读入的信息，**它不等价于删除原文后完整prefill**；AKV的radix树必须把操作历史加入前缀键，恢复时按drop/repos边界分段attention，防止不同操作轨迹错误复用。

三模型、四Agent benchmark，最新12条tool-response rolling-drop均分相对Standard增1.3–2.1pp，部分单任务退步；按供应商token费率估成本平均省27.7%，不是实测整机电费。服务测8×A80080GB、BF16、4prefill+4decode、TP4；选上下文最长80个SWE任务、各方法回放自身轨迹。各自峰值并发AKV14对Standard12，均值/P95延迟快1.36/1.51×，output throughput1.18×；input吞吐计全部逻辑token，含drop内容。**回放规定生成长度、重放tool结果，非完整在线工具执行；大并发Standard还有一小时截尾未完成请求。** 建议深入阅读cache身份与恢复语义，在自家业务审计删工具响应后的长期信息丢失；论文未给linear-attention支持，集成发布状态也须与已合入代码区分。

### LAW：只传播预期 embedding、最后才解码，把 self-conditioning 变成完整生成状态

**EPFL、Basel、ELLIS、Microsoft AI等；2610.30068v1，10 月 21 日提交，本轮补录。** 一手：[§3–4、AppC–D](https://arxiv.org/html/2610.30068v1)。从uniform token prior开始，中间不采样离散token，逐步传播denoiser词分布的expected embedding；多pass训练用stop-gradient并随机temperature暴露不同锐度状态，终点才raw/argmax解码。与把连续SC作为离散状态辅信号的LDDM不同，删除离散路径可减少每步信息坍缩。

12层DiT、TinyGSM11.8M训练、每题生成并执行一个Python程序，GSM8K T1/NFE1024为42.2%，LDDM-U34.5%；从Duo继续50Ksteps后48.1%，但这不是零成本改sampler。OWT1024长度、16GPU、同训练FLOPs的LAW308K对baseline400Ksteps；64NFE在生成PPL/单样本unigram entropy曲线上接近ARM，**NFE非相同FLOPs/服务时间；1024NFE时相同entropy可更好但跨样本多样性较低**。温度regularization和mode embedding对防collapse关键，不同任务最优配方不同。建议考虑复现连续状态与离散SC对照，同时测跨样本多样性、正确性与实际计算；尚无大模型开放域能力证明。



<a name="core"></a>

## 核心开源工程：训练、推理与正确性

本节版本状态以北京时间标示合入时间，源码基线为每项merge commit；所有benchmark/测试均为作者或PR引用证据，本轮未独立执行GPU/NPU/CPU框架测试。main合入与公开release发布分开处理。

### NVIDIA/Megatron-LM

**Global-batch Quantile Balancing：合并直方图后更新路由偏置。** 此前按microbatch求分位数再平均，现在新增global_batch：为每expert累计路由margin直方图，在梯度累积结束后沿TP/DP/CP归并，以topk/专家数为目标分位点插值、中心化bias并更新下轮bin范围；失败的dropless重跑先清直方图，避免重复计数。默认仍micro_batch，global_batch限sigmoid、无aux loss、无group-limited routing/replay，EMA须0；融合需TE相应直方图接口。正文主要是模板，尚无可用收敛或吞吐对照。本轮打开router、finalize及分位数实现，不把实现接入当作Kimi结果复现。建议考虑复现全局batch不平衡与直方图近似误差，并验证重算/重试只计一次。

来源：[#7394](https://github.com/NVIDIA/Megatron-LM/pull/7394)（main，合入10-23 05:14，`92641a421f40eeb40fc83f3f13ec83ca652c8b13`）。

源码阅读点：`megatron/core/distributed/finalize_model_grads.py::_update_router_expert_bias_with_quantile` → `megatron/core/transformer/moe/moe_utils.py::get_updated_expert_bias_with_quantile`。

**GTP在CUDA graph下保留FC1/FC2独立预取链。** 捕获时原先把dense、expert FC1/FC2并成一链，使FC1预取当前FC2，且消费时再记完成event会把后续无关gather也包含进等待。现在三条逻辑链独立但共享AG stream，底层Work.wait先建立通信依赖、随即记录完成event，消费者只等自己的权重；grad-ready hook回到参数实际RS stream。作者4×GB300针对性测试89项/rank；256×GB300、DSv4-Pro 30层+MTP1代理、MXFP8、TP1/PP1/CP1/EP64/GTP64/EGTP4、GBS512/MBS2/4K、同profiling设置，full-iter图吞吐3799→5146 token/s/GPU，eager4223。约35%是相对旧图、相对eager约22%（正文另写19%，口径不一致），不是全量DSv4-Pro通用增益。建议深入阅读完成event与buffer所有权，按实际拓扑复测。

来源：[#8345](https://github.com/NVIDIA/Megatron-LM/pull/8345)（main，合入10-22 22:30，`9e5aad0c6a3571b682e06a7d569be27e5194e7c4`）。

源码阅读点：`megatron/core/tensor_parallel/generalized_tensor_parallelism.py` → `megatron/core/tensor_parallel/gtp_cuda_graphs.py`。

**MTP逐head提前反传，编译交叉熵保留原地语义。** mtp_loss_early_backward让每个MTP head在detached leaf上算projection/loss并立即反传，保留head输入梯度，通过_MTPHeadGradInjector在主反传时注回，避免多个词表级logits/softmax同时常驻。pipeline在forward前设置与主loss一致的scale；warmup/capture禁用提前反传。CE把pointwise原地exp/scale和reduce/scatter分开编译，避免Inductor为输入mutation额外分配词表级临时tensor；GroupedMLP按实例解析offload开关。未见可用于全模型收益归因的最终对照，早反传与graph组合不能直接推广。建议检查loss scale、共享head累积梯度及峰值HBM。

来源：[#7478](https://github.com/NVIDIA/Megatron-LM/pull/7478)（main，合入10-23 01:06，`6e5946498fe1ed0157c7e307ef117b791dbdec3e`）。

源码阅读点：`megatron/core/transformer/multi_token_prediction.py::process_mtp_loss` → `megatron/core/fusions/fused_cross_entropy.py::calculate_predicted_logits`。

**Packed MTP把逐文档CP边界交换合成向量。** 每次roll将所有文档的边界打成一组neighbor向量，用CP group上的batch_isend_irecv交换，再向量化本地roll与HSM边界映射；独立DP lane不借WORLD通信。zigzag CP>1要求物理文档长能整除2×CP，否则明确报错；末文档后的padding独立成段并以0结束。作者仅CPU/Gloo与格式检查，未GPU验证。建议考虑复现多文档不同长度、padding与多DP lane，先看通信次数再量吞吐。

来源：[#8389](https://github.com/NVIDIA/Megatron-LM/pull/8389)（main，合入10-22 18:44，`50f94464f6ae5fa7f68aac29b6c5da8d3ed16eff`）。

源码阅读点：`megatron/core/transformer/multi_token_prediction.py::_exchange_cp_roll_boundaries`。

**Batch-invariant MXFP8 MoE按活跃行工作并修复int32溢出。** graph静态buffer按max_tokens×EP×topk分配，但BI量化/置零/unpermute原先每步扫描最坏容量。新persistent kernel读device live count，只处理活跃行及真实padding；量化寻址改int64，修复≥2³¹元素时非法访问。作者Nemotron-3.5-Nano随机权重、4×GB200 EP4、NVLS、52层/128expert/top6、ordered collective、局部图，max_tokens8192下1/40 live行decode中位16.6→7.9、19.4→10.8ms；73728旧版直接crash，不能算速度倍率。162项测试及4prompt生成/logprob bitwise证据由作者提供；未用行不再初始化，消费者必须以offs限制。建议优先采用对应正确性修复并做生产模型回归。

来源：[#8385](https://github.com/NVIDIA/Megatron-LM/pull/8385)（main，合入10-22 18:34，`77232c387ad51e3e0724cf9da9bf50306db98015`）。

源码阅读点：`megatron/core/inference/moe/fused_moe.py::_quantize_mxfp8_used_rows` → `megatron/core/inference/quantization/mxfp8_quantize.py`。

**Batch-invariant Mamba移除z回放buffer并只算目标行。** z只参与当前输出行的逐元素gate，无须按chunk为每slot存整份z；decode直接用当前z，并将scan/CB缩成目标行tile、只在跨chunk时合并state passing，bookkeeping合成一kernel。row kernel复用训练固定配置以保持累积次序；未固定autotuner、dstate>128或warpgroup/tcgen05 tile回退原路径。作者GB200单层Nano形状、chunk128、graph、best of 3×500，BF16 64/256行143.9→75.3、419.3→212.2µs；属于单层微测。移除z的并发容量收益来自形状推算，非E2E实测；剩余replay buffers仍在inference预算外惰性分配，可超额。建议深入阅读并测真实max_requests显存。

来源：[#8221](https://github.com/NVIDIA/Megatron-LM/pull/8221)（main，合入10-22 15:11，`aa46a57229995b4c537ed2f554e78d10473e4964`）。

源码阅读点：`megatron/core/ssm/ops/mamba2/batch_invariant_decode.py` → `megatron/core/ssm/ops/mamba2/batch_invariant_decode_rows.py::row_kernel_configs`。

**模型转换与provider不再静默错配HF布局或RoPE。** Mixtral转换同时兼容Transformers4逐expert w1/w2/w3与5的mlp/fused gate_up/down布局，明确gate-before-up，拒绝转置fused权重，版本比较改语义版本；作者tiny真实模型逐tensor相等，另引8×GB200 TP4/PP2损失对照。Nemotron MIMO则把原硬编码none改为用户position_embedding_type，并传rotary base/percent/interpolation；provider50项通过。两者均改变此前错误加载/构造结果。建议采用前对checkpoint逐tensor及position配置做确认，别只放松版本号。

来源：[#7937](https://github.com/NVIDIA/Megatron-LM/pull/7937)（main，合入10-23 03:29，`888887d35e378b59d03e34855c1e6e21aed918f1`）；[#8371](https://github.com/NVIDIA/Megatron-LM/pull/8371)（main，合入10-22 22:02，`964b1deefb95a339a9297ef11b10125cffcd5b88`）。

源码阅读点：`tools/checkpoint/loader_mixtral_hf.py::set_mlp_state` → `examples/mimo/model_providers/nemotron_moe_vlm.py::language_model_spec`。

**Sequence-packing scheduler修复中间PP stage首步崩溃。** packed数据普通路径在每个PP stage的TP0建loader，但dp_balanced scheduler只允许首/末stage持iterator并广播packing metadata。现在pretrain_gpt/hybrid在scheduler启用时只在首末stage建数据，消除PP≥3无VPP首步assert及VPP多余loader。中间stage MTP仍不支持，不能把跳过MTP shortcut解释为接通。作者CPU模拟TP2/PP4并给GPU测试，未实跑GPU。建议先复现原首步错误，再测真实packed多stage训练。

来源：[#8378](https://github.com/NVIDIA/Megatron-LM/pull/8378)（main，合入10-22 10:56，`ede2c586cf70dd452f93b9726b31cbedc8df0dd0`）。

源码阅读点：`pretrain_gpt.py::is_dataset_built_on_rank` → `pretrain_hybrid.py::is_dataset_built_on_rank`。

**dev分支：optimizer自有master权重终于受offload chunk约束。** precision-aware FusedAdam自有state[master_param]现在与moments一起分块H2D→update→D2H并计入chunk planner；MCore在更新后读的master仍整窗恢复，混合两种所有权仍拒绝。staging改每device一份byte buffer，避免不同dtype各留一窗。作者单H200、268M BF16 MLP、128MiB chunk、offload1.0/int16 remainder，相对dev c3010d6 step active峰值768→320MiB，70.8→66.4ms，参数与无offload bitwise。只合入dev，非main或release能力。建议继续观察主线迁移，并区分master所有权再复现。

来源：[#8232](https://github.com/NVIDIA/Megatron-LM/pull/8232)（dev，合入10-22 16:51，`a5aa637fa5dbe2b08af7b413e32306dbc086aa01`）。

源码阅读点：`megatron/core/optimizer/cpu_offloading/chunked_optimizer_state_offload.py::ChunkedOptimizerStateOffloader`。

### verl-project/verl

**MXFP8 RL训练—导出—rollout闭环有可运行配方。** Megatron/TE MXFP8路径补齐THD每序列32-token对齐，并拒绝QAT与TE FP8 recipe混用；BF16权重/checkpoint保持，rollout端lm_head/embedding/router保持BF16。Qwen3-8B单B200、vLLM0.29/TE2.16.1/MCore0.19.2、GSM8K1319题、10step，BF16准确率0.329→0.819，MXFP8 0.372→0.803，单次短跑不足判断1.6点差异；rollout KL为BF16约0.0006–0.0008、MXFP8约0.0063–0.0071，低精并未降低该组mismatch。文中30B的11%step收益属于旧ModelOpt/vLLM0.24配方，当前30B示例尚未本分支实跑，不应直接引用成新闭环收益。建议考虑复现同kernel、同KV精度、长程收敛与吞吐。

来源：[#8127](https://github.com/verl-project/verl/pull/8127)（main，合入10-22 10:35，`d1390eea0c94467d4ae1387eeda9e2f78f8d73ea`）。

源码阅读点：`docs/low_precision/mxfp8.md` → `verl/models/mcore/util.py`。

**Top-k蒸馏同时对齐低精度padding与CP token身份。** THD预处理统一收quant_align_size整数，teacher top-k与student共享量化padding；随后的8332继续透传cp_layout/local_cp_size，修复dsv4_hybrid student contiguous、teacher默认zigzag时shape相等但token错配的静默KL污染。作者CP2/17token示例rank0有128/136有效token错配；CPU49项含反例与真实调用参数检查，GPU CP未跑。8325的2×B200五步测试来自8127旧head，不能覆盖8332。BSHD forced_max_seqlen问题8061仍open；dynamic CP+蒸馏仍显式拒绝。建议优先采用token身份修复，以唯一token编号核每rank配对。

来源：[#8325](https://github.com/verl-project/verl/pull/8325)（main，合入10-22 14:11，`b1a98e1d425e44180ddbf8d87aa074dde71c5466`）；[#8332](https://github.com/verl-project/verl/pull/8332)（main，合入10-22 16:03，`67bf1df3f2d294f4c5fa65a5dd4585f24d642816`）。

源码阅读点：`verl/workers/engine/megatron/transformer_impl.py::_lm_head_logits_processor` → `verl/trainer/distillation/megatron/losses.py::compute_forward_kl_topk`。

**QAT ignore模块规则现在覆盖子quantizer。** 忽略self_attention.linear_proj原先匹配不到其input_quantizer/weight_quantizer，actor仍fake quant而rollout BF16。现在保留原direct规则，追加pattern.*，保护模块边界及别名。作者H100/Qwen3-0.6B、10GRPO更新/单seed、Megatron ModelOpt FP8 QAT+vLLM FP8_BLOCK，56个误启quantizer→0、投影QDQ误差→0，目标BF16 tensor refit11/11相等；两臂reward反而不同，不能宣称reward/KL改进。建议采用精度配置修复并在实际ignore层核forward/backward。

来源：[#8329](https://github.com/verl-project/verl/pull/8329)（main，合入10-22 17:43，`5d09d065b2a69116bf9d5fe80b0510b68c8aeac7`）。

源码阅读点：`verl/utils/qat/modelopt.py::_ignore_patterns_to_quant_cfg`。

**VeOmni低精度梯度通信接线，当前小模型结果未提速。** 默认关闭low_precision_reduce_scatter_comm直传已有VeOmni构建器；仅CUDA FSDP2混合精度且确认同节点的multi-rank shard group用BF16 transport、FP32累积/输出，其余保留native。作者早版2×H200 NVL、428M Qwen3、seq512/batch4/micro2、eager无AC，native117.0/118.9ms，低精130.9/119.2ms，显存reserved略增；最终去掉兼容guard后的代码只CPU复核。建议继续观察，按真实bucket/带宽A/B后再开；不把字节减少等同加速。

来源：[#8330](https://github.com/verl-project/verl/pull/8330)（main，合入10-22 15:03，`fb55b9288560f0c7bf380c1c1c3b45ea170538b0`）。

源码阅读点：`verl/workers/engine/veomni/transformer_impl.py::_build_model_optimizer`。

**训练/rollout一致性增加raw logprob零差计数。** exp会把1ULP logprob差映射成同一prob，旧prob diff无法证明严格一致。新增有效response token raw不等计数及绝对差max/mean，V1跨迭代count按sum；all-masked保留valid=0，count0不应误当有效比较通过。rollout.calculate_log_probs启用后现有三个trainer caller都得到指标。作者23项CPU测试含1ULP反例。建议采用于exact-match实验门禁，同时检查有效token数与NaN。

来源：[#8025](https://github.com/verl-project/verl/pull/8025)（main，合入10-22 11:39，`16ec093cc4a235c603c29aed1f2c4e9f551fab48`）。

源码阅读点：`verl/utils/debug/metrics.py::calculate_debug_metrics`。

### vllm-project/vllm

**XPress因果refiner接入DFlash投机解码。** 为block drafter增加读hidden/global summary/previous token的rank256 head和下三角mixer，用固定次数Jacobi迭代取代逐token串行修正；复用DFlash KV与sharing，Triton融合latent与add/argmax避免大logits临时量。实际配置仍method=dflash并按Qwen3XPress architecture分派；V2 runner、fill-in布局，sample_from_anchor拒绝，默认6轮/top-C512；top-C会限制可达token，不等价完整词表refiner。作者单H200 Qwen3-8B/FA3/graph/prefix、7任务×batch1/4/16/32/64、最多2048输出，相比DFlash汇总吞吐+18.5%，DSpark+4.1%，部分高batch反降；5-shot GSM8K0.889、accept length5.654。建议深入阅读并考虑复现自身batch分布，不能只用平均accept length预测收益。

来源：[#54448](https://github.com/vllm-project/vllm/pull/54448)（main，合入10-23 05:04，`ce9467b757c024a55fea4d0919ac2e682f28a894`）。

源码阅读点：`vllm/v1/worker/gpu/spec_decode/xpress/speculator.py::XPressSpeculator` → `vllm/model_executor/models/qwen3_xpress.py::XPressRefinerHead`。

**NCCL权重更新修复跨stream复用导致的MoE静默损坏。** packed consumer的unpacked views可能被layerwise reload留到后续pack、在另一slot stream上消费，原buffer已释放给allocator重用。现在每次receive串接先前ready event，且对所有slot/caller stream record_stream，直到所有潜在reader结束才可复用，保持双buffer重叠。作者4×H100/torch2.14真实NCCL NET复现10/10损坏→0/10，OLMoE TP2逐次weight_checker匹配；1GB pack sync约4.95→4.93s。慢transport+layerwise保留一至几个pack更久，reserved memory可能增。建议优先采用，故障验证要检查tensor/data_ptr和reload后权重，服务输出相近不足排错。

来源：[#62555](https://github.com/vllm-project/vllm/pull/62555)（main，合入10-23 00:06，`c723f6d3ee5c832f7d4a64aaa8dfda5cecbf9eef`）。

源码阅读点：`vllm/distributed/weight_transfer/packed_tensor.py::packed_nccl_broadcast_consumer`。

**P/D容错在失败边界保住KV进度并排空响应队列。** worker异常不再丢finished_send/recv/invalid_blocks，而把KVConnectorOutput随失败状态带出；executor在FT路径收齐所有worker response后抛错，salvage output下次正常aggregate重放。sentinel消费batch futures中的get-and-clear进度，而非直接清队列；cache events故意丢弃，避免破坏活worker quorum。补丁已核到exception→defer→aggregate→scheduler消费链，未独立运行故障注入；不能视为所有connector/RDMA取消安全已证实。建议考虑复现传输中worker失败/retry、多次失败与KV页回收。

来源：[#56430](https://github.com/vllm-project/vllm/pull/56430)（main，合入10-23 02:14，`e7bb7e2f20752ad38f8d44a6e7258cf002e29bea`）。

源码阅读点：`vllm/v1/executor/multiproc_executor.py::get_response` → `vllm/distributed/kv_transfer/kv_connector/utils.py::KVOutputAggregator` → `vllm/v1/fault_tolerance/engine_core_sentinel.py::_drain_batch_queue`。

**DFlash/DSpark不再错误丢末尾prefix block。** 原use_eagle大类同时覆盖DFlash/DSpark，却把仅EAGLE-style依赖next token的draft KV规则套到它们，Mamba align checkpoint不可复用、后续回复全上下文重算。新增needs_eagle_block_drop，仅EAGLE/EAGLE3/MTP保留drop，其他lookahead/encoder逻辑仍用原大类。作者2×L40S旧布局三轮共享prefix命中0/0/0→0/560/560；当前V2 acceptance对照近似持平，外部验证也由PR引用并非本轮独立复测。建议采用前后测多轮prefix、checkpoint恢复与token/logprob。

来源：[#54163](https://github.com/vllm-project/vllm/pull/54163)（main，合入10-23 08:03，`0bfeb8d0e367c21d4406c87d0807729cc8787322`）。

源码阅读点：`vllm/config/speculative.py::needs_eagle_block_drop`。

**NemotronHMTP在每个PP stage共享target embedding/head。** drafter只载mtp.*，embedding/head先留None并声明无自有副本，再由sharing helper接target；target refit直接对draft可见。原PP2仅share head，作者GH200 TP1/eager/MTP3/12prompt×64token，accept9.0%→70.4%，与PP1输出逐token相同。若checkpoint真有不同MTP head会被忽略，属于兼容边界。建议优先修复PP+RL weight update组合并核模块identity，不只比一次质量分。

来源：[#62194](https://github.com/vllm-project/vllm/pull/62194)（main，合入10-22 23:26，`f235b89bdf62b5fc6356fc8d5e9d59572e48b145`）。

源码阅读点：`vllm/model_executor/models/nemotron_h_mtp.py::NemotronHMTP`。

**FA4 FP8单KV-head descale显式修正head stride。** scalar.expand(1,1)的stride(0,0)被PyTorch视为contiguous，调用contiguous不能满足CuTe leading_dim=1必须stride1的要求。现仅对2D且head stride≠1的descale做contiguous-format clone，不拷KV。问题来自B200/Nemotron-H两KV head/TP2/每rank单head。作者36项CPU生产wrapper回归和layout环境检查，未本轮GPU重现。建议采用于FA4 FP8服务，回归batch1及TP分片退化维度。

来源：[#58304](https://github.com/vllm-project/vllm/pull/58304)（main，合入10-23 02:36，`2a038794e0fc1400d404489f8b902e03b7725a63`）。

源码阅读点：`vllm/vllm_flash_attn/flash_attn_interface.py::flash_attn_varlen_func`。

**UVA专家权重的小batch改走按expert排序路径。** VRAM内小batch跳过align可省kernel，但host-resident experts若每(token,expert)独立block会重复走PCIe读相同权重。现在读UVA marker强制排序聚合expert，其余device路径不变。作者RTX4090/PCIe4×16、Qwen3-30B FP8 offload16GiB、128输出、并发2/3/4输出吞吐约+7/+10/+15–19%，单并发及8/16基本不变；250题GSM8K逐响应相同。235B TP8本身跨启动不确定，不能用它证明逐token一致。建议按PCIe流量与并发考虑采用。

来源：[#62054](https://github.com/vllm-project/vllm/pull/62054)（main，合入10-22 22:28，`3b364baef5cd731b5e1f14ff16387feb69d01f10`）。

源码阅读点：`vllm/model_executor/layers/fused_moe/fused_moe.py::_prepare_expert_assignment`。

**ROCm graph局部关闭expandable segments消除AITER staging。** VMM指针不能hipIpcGetMemHandle导出，AITER captured AR退到逐次copy-in。现在仅capture作用域关闭expandable_segments并finally恢复，限定可延后registration的AITER≥0.1.26、非gfx1250、TP开启且DP无AITER collective；之前分配的权重/KV不变。作者MI355X Qwen3-0.6B TP2 trace staging57次→0，batch1/4 decode约-7/-8%；DSv4.1 TP8仅约-1.5–2.9%，graph pool增加0.4–0.5GiB。旧AITER测试有补丁叠加；建议在锁定版本上核registration marker、KV容量与真实TP/DP拓扑。

来源：[#57481](https://github.com/vllm-project/vllm/pull/57481)（main，合入10-22 09:34，`a4aa37d1a0ed89dd6d5f37ee3071e3ddc0d85e61`）。

源码阅读点：`vllm/distributed/parallel_state.py::graph_capture` → `vllm/device_allocator/alloc_conf.py::non_expandable_allocations`。

**Quark MoE新增MXFP8 W8A8与逐层混合量化。** Quark把MXFP8 routed experts分派到已有MXFP8 MoE栈，在gfx950走AITER_MXFP8/dynamic activation，逐层MXFP4/MXFP8无需另造格式。仅1D per_group/group32/E8M0 spelling，2D per_block experts与bias明确拒绝。作者MI350 dispatch矩阵60项、重点5项；没有完整模型吞吐/质量结果。本轮核schema分派与method构造，未深审已有底层kernel。建议考虑复现真实Quark混合checkpoint逐层加载与正确性。

来源：[#63125](https://github.com/vllm-project/vllm/pull/63125)（main，合入10-23 01:48，`b6031d71beb219b9c2cbf580745c7307bf15ec91`）。

源码阅读点：`vllm/model_executor/layers/quantization/quark/quark_moe.py::QuarkW8A8Mxfp8MoEMethod`。

**Beam TP all-reduce融合norm与MoE finalize，主模型PR仍待合并。** SM90/SM100 AOT kernel把MNNVL one/two-shot AR与FP32 sandwich norm、可选TRTLLM-Gen NVFP4 top-k finalize合并，保留本地BF16 rounding并禁FMA收缩；shape/TP/multicast不符回标准AR。作者GB200 4GPU178项、跨2节点TP8 AR测试；相对同一kernel旧JIT的8192 hidden微测有6–16%差异但workspace原因未隔离，端到端相对标准TP AR仍待测试。Beam基础集成61978仍open：子PR已进main不等于发布版完整Beam就绪。建议继续观察主模型合入及E2E。

来源：[#62366](https://github.com/vllm-project/vllm/pull/62366)（main，合入10-22 09:31，`e4e6bcaae60241d5519f0a21d1601eac4d7a8d94`）。

源码阅读点：`csrc/beam/allreduce_norm.cu` → `vllm/models/beam/nvidia/allreduce.py`。

**V2多模态PP省掉重复encoder，spawn worker并发启动。** V2只首PP stage跑encoder，现在后续stage不再构建/加载tower；作者两sm80、Qwen3.5-4B PP1载入5.27→4.65GiB，部分真实image/logprob测试通过，V1/首stage不变。另spawn以thread pool并行start，保留rank顺序与失败清理；fork/CPU/NUMA绑定/单worker仍串行。作者8×H100/Nemotron-3-Ultra NVFP4、两次NCCL-ready启动55–56→15s，为启动阶段而非稳态吞吐。建议按部署痛点采用，回归多模态真实输入及部分启动失败。

来源：[#60238](https://github.com/vllm-project/vllm/pull/60238)（main，合入10-22 21:27，`168a86ab8f418b0e12c21c3d582ddf5c6d2927ef`）；[#60028](https://github.com/vllm-project/vllm/pull/60028)（main，合入10-23 06:40，`52f44839f2b7a05664f2ac74b5ba766c0045d6d3`）。

源码阅读点：`vllm/model_executor/models/interfaces.py::_mark_tower_model` → `vllm/v1/executor/multiproc_executor.py::_init_executor`。

**LoRA/MoE边界：保留运行集合、禁跨stream同communicator分叉。** scheduler把本步未调度的running request也计入max_loras，避免稍后恢复时超额；fully-sharded LoRA的shared experts关闭aux stream overlap，因其和routed专家在同TP communicator跨stream发collective，graph replay可跨rank错序死锁。另moe_forward custom ops打needs_exact_strides，EPLB把element_size=1的低精tensor按uint8 view搬运，避免NCCL不识E8M0/FP4x2。都是正确性边界而非统一性能优化；建议对MM跳过/PP、图下TP LoRA及低精EPLB分别回归。

来源：[#60306](https://github.com/vllm-project/vllm/pull/60306)（main，合入10-22 10:58，`3e58641d1f29b61d534c889c9625d4bc56f0ecf7`）；[#61284](https://github.com/vllm-project/vllm/pull/61284)（main，合入10-22 10:59，`7bac28707d318a1c9c8f2011828913de5b19dd51`）；[#61749](https://github.com/vllm-project/vllm/pull/61749)（main，合入10-22 13:16，`15a78aa88e92ef5073ad4afa08544252d4c2ca8b`）；[#61786](https://github.com/vllm-project/vllm/pull/61786)（main，合入10-22 11:22，`876d5da8316a81180b8c7f10dea5ffab5b5a849f`）。

源码阅读点：`vllm/v1/core/sched/scheduler.py::schedule` → `vllm/model_executor/layers/fused_moe/runner/shared_experts.py::_disable_shared_experts_overlap` → `vllm/model_executor/models/interfaces.py::set_eplb_state`。

**ROCm构建拆分经历回退后以offload-compress重新合入。** 昨日拆paged-attention TU的62475复制多arch GPU debug info，使9arch库超过PC32约2GiB可达范围、mold/lld都链接失败；62983先回退，63105再加offload-compress后恢复12TU并行。作者9arch/RelWithDebInfo/MAX_JOBS64冷构建85m28→21m25，库约125.9MiB；测试ROCm7.2与10，未验证6.x。压缩保留debug info但增加模块解压，约0.6s从首次attention移到import。建议维护ROCm wheel者采用并验证完整arch矩阵；原单arch7.6倍不能直接代表发行构建。

来源：[#62983](https://github.com/vllm-project/vllm/pull/62983)（main，合入10-22 10:17，`9d61ade7f657833df27e43c4561455523c78c8fb`）；[#63105](https://github.com/vllm-project/vllm/pull/63105)（main，合入10-23 07:31，`cc1a6d31353205bf38b6dbecf4ad134ce3b2b678`）。

源码阅读点：`CMakeLists.txt` → `csrc/rocm/attention_dispatch.cu`。

**补录10-21：MI355X Kimi-K3专用decode路径。** latency_specialized默认关闭；TP8/gfx950把≤8行、完整uniform spec-verify请求的92层KDA/MLA与latent MoE拆persistent FlyDSL MonoKernel，AR在peer memory里做，其他step回普通layer。明确拒绝PP/DP/EP/EPLB/LoRA/KV connector/offload/refit等；wait10秒超时、每16step TP达成fail-stop，因此不是立即故障上报。作者8×MI355X、Kimi-K3/DSpark k7/FP8 KV/V2 FULL、8K输入1K输出/c1/3seed，TPOT5.44→4.31ms；测量用旧20commit镜像mount本代码，ISA相同不等于整树完全验证。建议考虑复现该窄低延迟配置。

来源：[#60683](https://github.com/vllm-project/vllm/pull/60683)（main，合入10-21 14:53，`a9bf45ed8a856a0b1839197152d0d26fabb6599c`）。

源码阅读点：`vllm/models/kimi_k3/amd/specialized_decode/runtime/decode.py`。

**补录10-21：CPU AMX长上下文MTP改为逐请求GQA计划。** 不再要求整batch全部q1才能做grouped-query调度，按每request因果性/query长/AMX几何与split-KV成本选择，支持BF16 attention及FP8 E4M3/E5M2 KV。作者Qwen3.5-4B TP1/BF16/100K输入1K输出/c1/无prefix，auto KV MTP3输出8.01→9.48token/s；FP8分支仅一次计时、native amx_fp8/ARM未测，MRCR n8两臂都未过门槛，不能写全部质量测试通过。建议在实际CPU平台复现，保持普通FP8 prefill fallback以避免TTFT退化。

来源：[#59082](https://github.com/vllm-project/vllm/pull/59082)（main，合入10-21 12:18，`900828023a9b0eeae042ebd9869497b8d0a6ad81`）。

源码阅读点：`csrc/cpu/cpu_attn_impl.hpp`。

**SM90 DSv4.1 sparse prefill使用真实8/16/32个local head。** FlashMLA依赖pin更新后，Hopper sparse prefill不再统一把query head pad到64，FP8 decode和其他架构继续原padding。作者TP4/DSpark k5、随机4096输入128输出/并发32、128prompt、prefix关闭的单组结果输出365.65→393.10token/s，但具体SM90设备及重复测量披露不足，不能推广为稳定7.5%增益。本轮核最终native-head分派与固定FlashMLA commit，未深审其kernel。建议考虑复现自己的TP-local head数，连同padding区输出与accuracy检查。

来源：[#61615](https://github.com/vllm-project/vllm/pull/61615)（main，合入10-22 15:50，`b579949edf936304ac9e2b52a1613e2871bbcd54`）。

源码阅读点：`vllm/models/deepseek_v41/nvidia/flashmla.py::_get_prefill_num_heads`。

**token embedding和late chunking进入标准embeddings入口。** token_embed服务器的/v1/embeddings与LLM.embed默认返token向量，设late_chunking_params后在完整上下文token表示上聚chunk并带token/source范围，复用已有core。仅float encoding、chunking须纯文本；矩阵embedding及metadata是vLLM扩展，Cohere/batch/invocations未扩。作者A10080GB/E5-small/BF16/512context、V1/V2 HTTP与离线正确性检查，未测检索质量；最终offline validation调整仅CPU复核。建议检索系统考虑复现offset、并发与长文截断，不据此宣称RAG质量提升。

来源：[#61772](https://github.com/vllm-project/vllm/pull/61772)（main，合入10-22 12:00，`79d5c9379600751b189d6e56a2fce3b3c0ddaae1`）。

源码阅读点：`vllm/entrypoints/pooling/token_embed/serving.py::ServingTokenEmbedding` → `vllm/entrypoints/pooling/offline.py::embed`。

### sgl-project/sglang

**Plan-driven LoRA整栈合入main，支持边界仍严格。** 21-part历史栈作为同一141文件实现合入：dense triton_v2按phase/shape/adapter rank选grouped/per-row shrink/expand，MoE支持per-expert/shared-outer与Triton/CuTe BF16/FP8、Marlin NVFP4，路由workspace按plan所需视图建立，LoRA-free decode graph独立。MoE仅SM90/SM100、Standard dispatch、trivial placement，无ROCm/elastic EP/EPLB/redundancy；V2拒绝DP-attention>1、PDmux/TBO等，draft仍无adapter，DSPARK要求static ragged。旧栈CI不等于此合并上下文通过，当前PR状态面板仍有失败，不引用泛化TPS。本轮只审注册/配置/计划合同与主干补丁，141文件kernel未全审。建议深入阅读并在精确device/shape上复现。

来源：[#45892](https://github.com/sgl-project/sglang/pull/45892)（main，合入10-23 04:40，`13999673ef26f7e1ccd55f7b6e47e7fbd7199119`）。

源码阅读点：`python/sglang/srt/arg_groups/lora_hook.py::check_lora_moe_runner_args` → `python/sglang/srt/lora/moe/configs/README.md`。

**KVShard把GLM-5.2 DSA attention/indexer接到page-sharded cache。** 将pool读用logical page IDs与raw kernel用scratch地址分离，extend先聚prefix/chunk scratch；index-K fused写同时写scratch全部行与owner-local持久行，各target沿原量化路径，避免多次地址翻译及非owner落盘。sharded prefill禁dense MHA one-shot，走显式scratch sparse路径。本轮核metadata、write_targets与attention接线；正文未填实际模型速度/准确率表，不能宣称显存倍率或吞吐收益已证实。建议考虑复现owner分布、prefix命中、量化字节一致与跨rank输出。

来源：[#44280](https://github.com/sgl-project/sglang/pull/44280)（main，合入10-23 00:16，`cc811ff0c3d5e06b3d6ca10c0a0388bc720ab2c2`）。

源码阅读点：`python/sglang/srt/layers/attention/dsa_backend.py::init_forward_metadata` → `python/sglang/srt/mem_cache/page_interleave_pool.py::write_targets`。

**HiCache write-through补forward→D2H fence防陈旧KV。** overlap调度在处理N完成时已启动N+1，最后输出token KV仍在forward stream写，scheduler已把它插树并D2H可读到旧值。最终代码不是旧正文所写load_fence_stream，而是移除start_writing(wait_for_forward)的可选门，让所有write在write_fence_stream后执行；scheduler已把该字段接forward_stream。GPU侧wait只约束已排队工作，之后仍可overlap，write ack可能延后一forward，eviction同步路径也更晚。已核真实L2TransferEngine顺序单测，未本轮GPU重跑。建议优先采用，验证多轮对话device-hit与host-loadback逐token一致。

来源：[#42701](https://github.com/sgl-project/sglang/pull/42701)（main，合入10-23 05:46，`3b0003dac15f32b27cf247c5da555edf459a2e3f`）。

源码阅读点：`python/sglang/srt/managers/cache_controller.py::start_writing` → `python/sglang/srt/managers/scheduler.py`。

**HiSparse取消/失败时host目的页跟随decode KV一起延迟释放。** 此前失败分支先request_finished归还HiSparse host页，再决定等待prefill sibling drain；旧RDMA WRITE仍可落到已复用页。现在归还移到_release_request，与req slot/KV持有期一致，deferred ack到达才释放，不能通知prefill时沿原立即路径一次释放。作者B30085项回归、真实host pool576页：旧失败即全归还且可重分配，修复后释放前0归还。它修的是目的页所有权，不等于已独立验证所有RDMA取消。建议优先采用并注入中途取消+快速重分配。

来源：[#45108](https://github.com/sgl-project/sglang/pull/45108)（main，合入10-22 11:19，`6929afb528a8f923bc42872d4217f01ecec7e7de`）。

源码阅读点：`python/sglang/srt/disaggregation/decode.py::_release_request`。

**Mooncake prefill KV发送把D2H等待移出scheduler线程。** page indices先异步拷pinned host并附event，transfer worker读NumPy/发KV前等event；scheduler只读长度继续排下一forward。CP需host过滤或backend不声明host tensor支持仍同步，末chunk state payload也仍同步。作者GB300 DSv4.1Flash/10×DEP2 prefill+DEP16 decode/AgentX6400会话/900秒，叠加44623，总token/s/GPU521838→543328、p90TTFT36→29.3s；不是独立消融且字段包含input token，勿当output TPS。建议考虑复现CPU空洞与copy事件完成边界。

来源：[#45492](https://github.com/sgl-project/sglang/pull/45492)（main，合入10-22 11:11，`8dc7f98005dce2f9b297840d95045fc4e8cf9de7`）。

源码阅读点：`python/sglang/srt/disaggregation/prefill.py::_send_kv_chunk` → `python/sglang/srt/disaggregation/mooncake/conn.py::transfer_worker`。

**Breakable prefill graph修复DP barrier死锁与SM90 tail动态行。** 有idle DP rank时，row-bucket tail首次capture含TP barrier，活rank replay、idle rank eager会分叉死锁；新增capture时rank_barrier标记，统一从全rank相同token metadata决定全部eager。SM90 tail另避免用全extend长度重建少量live row索引，并把source-KV host compressor放eager break，防捕获时行数在真实replay变动触发repeat_interleave assert。作者DEP4/lone request与GSM8K复现，覆盖有限；建议优先采用并测空闲rank、首请求、row bucket切换及真实H200。

来源：[#45881](https://github.com/sgl-project/sglang/pull/45881)（main，合入10-23 08:36，`97b79f0796687b26335d3b2d4c051f6c830f53db`）；[#45407](https://github.com/sgl-project/sglang/pull/45407)（main，合入10-23 08:57，`26396bf56696b8edc53d955cdf232fdfc167611a`）。

源码阅读点：`python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py::_has_inactive_dp_rank` → `python/sglang/srt/models/deepseek_v4.py::_low_ratio_sources_break`。

**确定性rejection sampling补齐draft proposal RNG。** target/verify已有request seed而draft仍用全局exponential_，换batch/graph仍变输出。新fast_sample按seed/绝对位置/draft step/vocab做域隔离hash，在(0,1)生成uniform、FP64 exponential-race抽样，接通普通与多层EAGLE的eager/graph seed缓冲；默认非确定性路径保留。作者Qwen3.5-9B TP2温度1/seed42，12变batch输出12种→1；Step3.5Flash TP8/B200多层图也有检查。只CUDA兼容设备，hash分布sanity不是严格独立性证明；作者B200/vocab163840/graph微测batch128抽样138.9→544.1µs、增量峰值241.25MiB。建议考虑复现logprob与prefix/位置变动，并衡量确定性的实际成本。

来源：[#33395](https://github.com/sgl-project/sglang/pull/33395)（main，合入10-23 05:04，`25ad569ff93a43ce5c66cd94edbf75e2eb29b779`）。

源码阅读点：`python/sglang/srt/speculative/spec_utils.py::fast_sample`。

**Gemma4单步frozen-KV draft终于捕获CUDA graph。** capture早返<=1改<=0，一步仍有assistant seed forward可捕获；零步/关图保持原逻辑。作者单H200/Gemma4E2B+assistant/1step2verifytoken、随机256in/256out、main→PR→PR→main，c1中位TPOT5.30–5.33→3.53–3.54ms，c8约-30%，accept length约1.95不变；200题GSM8K79.5%两臂相等。收益限原本漏捕的一步路径。建议考虑采用并先确认server_info.draft_decode capture非0。

来源：[#45771](https://github.com/sgl-project/sglang/pull/45771)（main，合入10-23 06:37，`c394a90c2b4d82483ed908d2afbca1901d9d59d2`）。

源码阅读点：`python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py::_capture_cuda_graphs`。

**DSPARK prefill也可提前发布shared-read-done。** 先在target forward前用batch plan/positions/SWA ring算draft注入位置，forward后只从本次结果选择/写draft KV，不再读取scheduler共享表，故可加入PRE_REPLAY read-done allowlist。作者GB300 DEP4 DSv4.1Flash在ring与paged SWA两种映射逐prefill比前后placement相同并做200题GSM8K；没给可归因吞吐收益。它不关闭仍open的45580 chunk-stash寿命缺口。建议深入阅读read-done之前必须完成哪些读，再以overlap竞态验证。

来源：[#45682](https://github.com/sgl-project/sglang/pull/45682)（main，合入10-22 09:40，`d61f618276c6453886ccf263aed146476c344175`）。

源码阅读点：`python/sglang/srt/speculative/dspark_components/dspark_kv_inject.py::TargetHiddenKvInjector` → `python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py::_forward_prefill`。

**QSA CP移除host同步并重排K/V gather。** host长度镜像/预留RoPE表/max_q消掉每层多处item/tolist同步；最终代码在index-key gather之前向side stream发K/V gather，attention消费时join，末尾反向wait保护buffer复用；不是PR旧正文写的key gather之后。仅zigzag CP，prefix仍从replicated pool拼，通信本身未消失。作者表CP4 32K step68.45→58.94ms，但正文未明确模型/hardware完整配置，暂不把百分比写成通用收益。建议考虑复现同rank collective发射序与trace空洞。

来源：[#43792](https://github.com/sgl-project/sglang/pull/43792)（main，合入10-23 04:33，`154e29c1ec7582d6eed287c58050060548cd2f41`）。

源码阅读点：`python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py::prefetch_cp_kv` → `python/sglang/srt/models/qwen4_exp.py::self_attention`。

**分布式确定性/graph guard扩展，根因结论保持审慎。** NCCL_GRAPH_REGISTER现在在任一多rank phase可能捕获collective时保守关掉；作者H200跨节点TP16/batch128重现17–37分钟挂、关后5小时/76K请求干净。但源码明确pool地址复用只是suspected、根因仍open，不能照旧正文写成已证实机制。另ROCm确定性模式拒绝TP>2且禁custom AR，因为固定Ring/Simple RCCL仍不batch-invariant；TP2保留。建议采用guard并测自身吞吐，确定性要核逐logprob不能只固定seed。

来源：[#43216](https://github.com/sgl-project/sglang/pull/43216)（main，合入10-23 04:56，`786cdfc040be49f2b0dfbe59ca40745abd48cff2`）；[#43809](https://github.com/sgl-project/sglang/pull/43809)（main，合入10-23 00:31，`c80285ca9adc23946447128926c22e8e7dfb2b80`）。

源码阅读点：`python/sglang/srt/arg_groups/parallel_hook.py::_graph_may_capture_collectives` → `python/sglang/srt/arg_groups/overrides.py::_deterministic_rocm_rccl_check`。

**NIXL UCX显式环境选择优先，修正昨日CLI新增兼容边界。** 43482由disaggregation-ib-device生成device_list会盖掉UCX_NET_DEVICES并丢lo，AMD RoCE host可初始化失败。现优先级是显式backend device_list > 非空UCX_NET_DEVICES > CLI映射，环境存在时跳过CLI并告警，不改env或其他backend。作者parameterized/source-isolated初始化回归，完整AMD runtime尚待。建议采用于显式UCX配置部署，核实际选卡日志；不能以CLI存在推断已生效。

来源：[#45722](https://github.com/sgl-project/sglang/pull/45722)（main，合入10-22 20:07，`8a6770d5660153f808da9d2b2697b7baa7688485`）。

源码阅读点：`python/sglang/srt/disaggregation/nixl/conn.py`。

**DSv4.1 MXFP8 producer融合从decode扩到eager prefill。** Mega-mHC与wo_a einsum直接产生下一wqkv_a/wo_b所需的MXFP8+DeepGEMM packed scales，省BF16中间写读及独立量化；权重加载时同时备两种backend scale。Blackwell/UE8M0/aligned shape才准备，decode/verify≤2048行，后续45903让eager extend也可走，breakable prefill graph和SP仍排除。作者只有定性profile/GSM8K不变，未公开充分A/B数字；不能将两PR各算一次加速。建议考虑复现量化字节、scale layout与实际kernel分派。

来源：[#45632](https://github.com/sgl-project/sglang/pull/45632)（main，合入10-22 09:27，`aa10927271f0302fc7f2394a66b5568d394f2c88`）；[#45903](https://github.com/sgl-project/sglang/pull/45903)（main，合入10-23 06:29，`f9b7225f2e169b3a41d0087508fb607bd014e4fe`）。

源码阅读点：`python/sglang/srt/layers/quantization/fp8.py::prepare_deepgemm_mxfp8_input` → `python/sglang/srt/models/deepseek_v4_mhc.py::fused_mxfp8_rows`。

**DSv4.1 prefill减少重复反量化、copy与残留tensor寿命。** 共享kv_source按(source地址/gather IDs/行数)缓存一次compressed KV dequant，每forward/realloc/plain get失效；MegaAttention einsum直接写prefill graph out buffer；Engram/DSpark临时引用及调用frame不再把多GiB residual留到step末。作者B300/GB300 profile声称重复kernel/copy消失、峰值下降及GSM8K不变，没给完整时间/HBM表；这三项不应叠加估算收益。建议深入读workspace失效与无后续writer条件，再以大batch/连续step检查allocator retry。

来源：[#45909](https://github.com/sgl-project/sglang/pull/45909)（main，合入10-23 08:08，`9c7dd3c45f3044a42568425a91f1c3aa96b1d677`）；[#45901](https://github.com/sgl-project/sglang/pull/45901)（main，合入10-23 05:56，`9c0f6d6a610531bd16ad3513572e55226450b244`）；[#45900](https://github.com/sgl-project/sglang/pull/45900)（main，合入10-23 05:55，`818d7932d7f7a8305d3c60d0c461d54fa50e1e26`）。

源码阅读点：`python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py::SparsePrefillKVBuffer` → `python/sglang/srt/models/deepseek_v4.py::_mega_core`。

**DSv4.1 prefill图收敛到two-level，metadata搬到每replay一次。** SM100发布checkpoint的two-level candidate路径继续captured，非该配置ratio1/2改eager break并删未消费full-topk metadata；移除开发开关，不表示SM90/HIP新增同样图能力。index-K slots与ratio计划每replay snapshot，tail共享slots但独立plan；作者GB300 DEP4/4K与8K639次replay校验，4K scheduler gather约4.0→0.9ms。MLA默认capture ceiling由2048改min(chunk,8192)，非MLA保留chunk上限，显式设置不变；最终默认变更仅CPU配置测试，原4×GB300显式8192实验不能替代。建议按图pool与最高chunk重新配容量。

来源：[#45783](https://github.com/sgl-project/sglang/pull/45783)（main，合入10-23 03:26，`0d6b283d7e6ac79edc10b2ca788f309e972f527b`）；[#45741](https://github.com/sgl-project/sglang/pull/45741)（main，合入10-22 14:34，`82f7249ce6c1f2b6c5d293dabdf11d2b21cc667a`）；[#45565](https://github.com/sgl-project/sglang/pull/45565)（main，合入10-22 14:49，`779adb4b82243e7743be60a305dc62ca3bee3a3a`）。

源码阅读点：`python/sglang/srt/layers/attention/deepseek_v4_backend.py::_snapshot_low_ratio_k_slots` → `python/sglang/srt/arg_groups/memory_hook.py::handle_gpu_memory_settings`。

**SM100 V4.1 index logits默认BF16，选块tie允许差异。** 未设环境开关时，SM100 head weights/logits改BF16，其他架构FP32，显式false可回FP32；flat prefill tile仍按FP32预算分块，避免BF16表面省空间却把selection scratch翻倍。单测允许selection floor上BF16 tie引起block/pick差别，所以不是bitwise无变化优化；无完整任务质量/吞吐表。建议继续观察或精度对照，长上下文选块不能只看单算子误差。

来源：[#45880](https://github.com/sgl-project/sglang/pull/45880)（main，合入10-23 05:29，`840a6313c95735c46e55176b41ab263cfdc842c0`）。

源码阅读点：`python/sglang/kernels/ops/attention/dsv4/index_logits.py::index_logits_dtype`。

**Request-window metadata用Triton融合pointwise阶段。** 既有request-group scan之后两kernel建立query/history索引、length/commit mask，保留stride/floor/padding；CPU/HIP走旧实现，row/history大小runtime参数减少recompile。作者GB300/torch2.14.1/Triton3.8、4K1group/32K8group、window128、100call×5sample，metadata中位约0.89→0.41ms，非服务吞吐；独立测试/脚本依review移出最终树，历史fc63b690可回取，已有4项copy测试保留。建议考虑复现真实request-run边界与graph refresh。

来源：[#45726](https://github.com/sgl-project/sglang/pull/45726)（main，合入10-22 15:13，`9d5386fd13f7a91e9e84dc0c0ab61c5b1fbaba98`）。

源码阅读点：`python/sglang/srt/mem_cache/dsv41_request_window.py::window_layout`。

**Dense SwiGLU融合block-FP8仅开放窄decode路径。** 融合activation与动态block128量化，先匹配原dense BF16 rounding再量化，DeepGEMM直接吃packed UE8M0，不兼容layout拒绝。限SM100 CUDA/TP1/BF16普通decode，无LoRA/确定性/compile/override，prefill与spec verify保留原activation。作者B200 Qwen3-8B-FP8受控单请求TPOT约-1.92%；并发及full-patch prefill非退化证据仍受限。建议按准确分派条件考虑采用，不能把此kernel向全MoE或所有FP8配置推广。

来源：[#45219](https://github.com/sgl-project/sglang/pull/45219)（main，合入10-23 07:49，`bb9e9d5632e33fb10c7e6e4a026df749fca7017a`）。

源码阅读点：`python/sglang/srt/layers/quantization/fp8_activation.py::DenseSiluAndMulQuant`。

**同日logits直接写buffer优化立即补精度修复。** 45631把BF16/FP16 head直接输出FP32 buffer，省临时tensor/copy，却使未开启FP32 head的buffer路径绕过原低精rounding，与unbuffered prefill/LoRA不一致。45855最终仅self.use_fp32_lm_head允许直接FP32输出，普通head恢复原dtype再copy，Muse Spark显式override仍直写。原DSv4 DEP8约0.8–0.9%数字不能直接套修正后的普通head。作者最终CPU控制流验证，GPU/DSpark/LoRA端到端仍pending。建议两项一起看，优先精度回归，避免只摘优化。

来源：[#45631](https://github.com/sgl-project/sglang/pull/45631)（main，合入10-22 11:32，`f1e26dc658bea745e8e83ee8bd37a11c212fc810`）；[#45855](https://github.com/sgl-project/sglang/pull/45855)（main，合入10-23 03:46，`d67d1ae02c2c9f62757289e6453df29bc98082c7`）。

源码阅读点：`python/sglang/srt/layers/logits_processor.py::_compute_lm_head`。

**Idle SWA store先选dummy/idle布局，不读陈旧attention metadata。** CUDA/HIP idle forward在访问可能未设或带旧tail的forward_metadata前早返request-window idle layout或dummy slot，避免用上一个batch的tail store地址与长度。作者GB300 DEP4/MegaMoE/DSPARK有bounded-replay开关两种GSM8K200题验证，HIP硬件未测；无request-window时stale tail实际到达路径未追完。建议采用并覆盖空闲rank/非均衡DP，不夸为HIP完整验证。

来源：[#45699](https://github.com/sgl-project/sglang/pull/45699)（main，合入10-22 11:56，`17babd67b4495d6e58a6b6f43de549a284685b9f`）。

源码阅读点：`python/sglang/srt/layers/attention/deepseek_v4_backend.py::get_swa_out_cache_loc`。

**补录10-21：EAGLE KV索引构造启用token-block并行。** 把已有AITER并行kernel接到FlashInfer draft decode及共用verify/draft-extend：长context>32K时按token block并行复制int32索引，短上下文保持原grid。最终helper以host可得live长度和作为最长行上界，受table/window cap限制；不是旧正文称的平均长度，device长度不为估grid新增同步。window与tree索引语义不变，kernel本身未改。作者旧v0.5.19容器回移kernel、8×H200/Qwen3.5-397B FP8/TP8/EAGLE3step/并发10/agent trace各1小时，ITL中位6.55→6.05ms；最终grid sizing又变，未重测该整体验证。建议考虑复现90–170K上下文trace，同时验证短请求大table不会过量launch。

来源：[#42803](https://github.com/sgl-project/sglang/pull/42803)（main，合入10-22 06:26，`991b8462278503fbdfcdd30213bc0bd5c7662eb9`）。

源码阅读点：`python/sglang/kernels/ops/attention/kv_indices.py::spec_kv_index_token_blocks`。

### THUDM/slime

**MUSA启动层迁至musa_titan，旧环境变量失效。** 加速器选定后的一次性bootstrap改导入musa_titan，读MUSA_TITAN_PATH，优先apply_rl_patch并兼容patch_after_import_torch回调；缺包和缺内部依赖分别处理。MUSA_PATCH_PATH不再读，旧PYTHONPATH需迁移。补丁含path/回调/fallback/缺依赖测试，未本轮执行真实MUSA训练。建议MUSA使用者采用前先迁launcher并做全新进程启动smoke；不视为新硬件性能成果。

来源：[#2489](https://github.com/THUDM/slime/pull/2489)（main，合入10-22 13:59，`18868a79b0b083a093798a94ed0f8cb4f71b9c14`）。

源码阅读点：`slime/utils/accelerator/__init__.py::_import_musa_titan`。

### pytorch/torchtitan

**Muse Spark RL trainer从256GPU改128GPU并引入CP。** 816B DAPO-Math GB300配方从FSDP64×TP4/EP4改FSDP8×TP4×CP4/EP16，Ulysses transform同时覆盖global/sliding varlen层，降低跨rack expert AG/RS并分摊token激活；generator保持16GPU，EP16四host必须同NVLink domain。作者证据来自另一个SWE-rebench、128K context、3RL step，峰值186GiB、MFU15→26→28%，DAPO65K配方本身尚未重跑。建议深入阅读通信拓扑并复现目标workload，不把三步MFU当收敛或稳态性能。

来源：[#5581](https://github.com/pytorch/torchtitan/pull/5581)（main，合入10-23 02:17，`43d1bb98014e0dab19a83e5220ec7ea1ab06d49e`）。

源码阅读点：`torchtitan_recipes/rl/dapo_math.py::muse_spark_816b_a42b_dapo_math_gb300_config`。

**Verifiers0.4适配同时修正timeout仍带reward的训练边界。** Taskset.head→take、依赖Verifiers0.4/Harbor0.24；trace/episode is_timeout先映射ERROR_TIMEOUT，避免ok=True但超时的rollout按完成样本训练。仍需区分训练样本reward与validation error_reward规则，Terminus沙箱独立0.22安装不随host换。作者CPU49项、H100 Qwen3-0.6B 1trainer+1generator三步math smoke；Terminal-Bench sandbox端到端没跑，CI不覆盖Harbor。14层后续栈仅本第1层合入，不能把drain/cancel等全写已完成。建议优先复核timeout样本掩码。

来源：[#5456](https://github.com/pytorch/torchtitan/pull/5456)（main，合入10-23 07:31，`4103c7291ffeb568ebf78a522b5fb2a8c01f9b06`）。

源码阅读点：`torchtitan/rl/examples/verifiers/rollouter.py::rollout_status`。

**局部编译默认接通，但varlen默认并未改为Flash优先。** DeepSeekV4默认regions补rope并把ComplexRoPE recompile_limit设32，避免多site/shape/no_grad超过8而fullgraph报错；Muse Spark补fused_binary_activation，但已发recipe的fused_swiglu override绕过它。作者单GB300/16K/两层DSv4训练-1.1%，四层Muse微测-4–5.9%，后者非bitwise，约27%输出/grad差1BF16 ULP，未测长loss曲线。5152旧正文称强制Flash，最终代码却只新增可配置sdpa_backends，默认仍[CUDNN_ATTENTION, FLASH_ATTENTION]，不能声称默认cuDNN backward OOM已修复。建议配置明确backend并核实际kernel，性能按真实override路径测。

来源：[#5561](https://github.com/pytorch/torchtitan/pull/5561)（main，合入10-22 16:27，`3521b50957285059a41d5b2148148b8c0e990aba`）；[#5152](https://github.com/pytorch/torchtitan/pull/5152)（main，合入10-22 16:04，`094c884685c2583323b76be244b76526b85fb520`）。

源码阅读点：`torchtitan/models/common/rope.py::ComplexRoPE` → `torchtitan/models/common/attention/attention.py::VarlenInnerAttention`。

**补录：TorchTitan规模结果与TitanRL公告；本窗仅有修订记录。** 公告首发北京时间10-22 03:45:44，早于本主窗起点10-22 08:58:39，属于72小时回溯补录；修改时间10-23 04:33:58在主窗内（HTML原时区-07:00）。未取得修订前完整正文，无法精确还原本窗新增或更正了哪些段落；以下是当前公告内容，不能当作主窗首次披露或已确认修订增量。作者DSv3 671B+MTP1/MXFP8/GB300结果覆盖256–512GPU；Muse Spark长context表将learned routing与forced-balanced分开，后者不能作真实路由吞吐。TitanRL的bitwise train/generator logprob只限受支持on-policy配置。代码基线`4103c7291ffeb568ebf78a522b5fb2a8c01f9b06`已打开DSv3/Muse recipe及GraphTrainer文档，确认可复现入口存在；GraphTrainer仍有实验特性和MXFP8组合缺口，未核全部FlexShard/新checkpoint实现或重跑规模实验。建议深入阅读配方与支持矩阵，结合本日报具体PR决定采用。

来源：[PyTorch官方公告](https://pytorch.org/blog/torchtitan-update-new-sota-models-performance-at-scale-titanrl/)；[固定基线模型配方](https://github.com/pytorch/torchtitan/blob/4103c7291ffeb568ebf78a522b5fb2a8c01f9b06/torchtitan_recipes/models/deepseek_v3.py)。

### 发布、持续待办与实际覆盖

六仓公开release collection共254项逐页检查，主窗无新release发布；回溯期SGLang v0.5.22为10-19已报旧发布，不能把今天main合入写成该release已含。Megatron8232只合入dev；SGLang多个MoE/LoRA优化在cheng/hot-fix分支，miles的logits/Engram变更亦非main；TorchTitan release/0.4的PyTorch2.15 RC兼容提交不是公开release。

P2P consumer61667、XPU prefix43299、chunk-stash45580、Dory60149、Beam基础61978及BSHD蒸馏8061均重新打开且仍open。P2P main基线`432b1003f8ea7c871750f27bc82815a7ad0252d5`仍有10秒abort-ack timeout；其他connector或host页修复不关闭late WRITE/quarantine/RDMA serving验证缺口。

库存：主窗all-base merged PR 232，main 214；含72h回溯共823、main 793；default main按committer time列798 commits，与merge事件时间独立。正文选择72项PR、合为56组机制，另核官方TorchTitan公告。库存检索无失败；两项公告代码旧路径探针失败、底层库实现未全审，明确留待办。其余实质候选逐仓留在core-coverage.json，历史10-05→10-08等语义缺口保留，未推进全球完整覆盖水位。



<a name="engineering"></a>

## 扩展开源：发布与采用边界

本节各条基线为链接旁的已落地commit。PyTorch使用ghstack流程，部分原PR仍显示`merged_at=null`；相应事实由默认分支commit确认，不能仅按PR开关判断。发布工件与当天较晚main改动分别列示。

### TensorRT-LLM 1.4 RC1：默认缓存栈迁移已进入预发行，取消流仍有已知故障

**NVIDIA；[v1.4.0rc1](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.4.0rc1)（发布2026-10-22T01:38:13Z）。** tag固定到`0372799d1543ef5920c1318b829cf127800dbcb9`，代码时间10-19，发布事件在本窗。官方说明明确默认KV manager V2与Python transceiver是破坏性变更；已有Mooncake V2、DSpark ragged验证、逐请求随机种子、KV lending API、DeepSeek-V4.1-Flash和NVFP4 GLM-5.3-Flash等进入该预发行，不能都当成当天新算法。需要先读迁移项与依赖约束，再按自身P/D和spec组合回归。

**限制直接影响生产采用：** release记录A100上流式请求取消后，默认Python KV传输可能导致后续生成请求失败；Qwen3 JSON-schema示例可能无界输出。下面10-22才合入的FP4 MLA disagg、GLM共享draft KV和forward存活修复均晚于该tag，不能以“1.4 RC1已修好”替代固定commit验证。建议升级评估；涉及取消、P/D和结构化输出的服务继续观察并做故障注入。

### Dynamo 1.6：路由策略、sidecar和运维能力形成正式发布快照

**NVIDIA/Dynamo；[v1.6.0](https://github.com/ai-dynamo/dynamo/releases/tag/v1.6.0)（发布2026-10-22T22:31:45Z），tag `701da25292ecca8f64d770cc0b17d389200019a1`。** 官方发布说明给出可插拔请求分类/延期/定向、统一worker选择核心和`--router-policy-config`；experimental sidecar覆盖TRT OpenEngine、vLLM LoRA生命周期及多模态encode→prefill→decode。Planner可在线改GPU预算，Operator支持引擎与Dynamo独立容器，Snapshot用于active/standby引擎切换。这些是具体部署契约的扩展，适合从自己的流控、伸缩和故障恢复需求评估；本轮未重放完整发布测试，也没有可用的统一成本收益数字。

其引擎基线为SGLang0.5.21、TRT1.3.0rc29、vLLM0.31.0，不能与TRT1.4RC1或当日main混为一套已验证栈。发布说明的`Looking Ahead`是未来事项，不能视为已提供`/router_policy`热更新和RL artifacts等。已知故障中，SGLang的`stop_token_ids≥vocab`在同batch任一请求设置`min_tokens`时可触发device assert并让全部inflight失败；Snapshot/GMS与默认breakable prefill图有启动冲突，官方临时方案是禁CUDA graph。另列Frontend丢弃`logit_bias`、Planner从mixed live指标外推容量可严重失真（建议预发布AISimulate模型），Slime示例/NCCL权重同步也未闭环。采用应逐项匹配原notes的配置与规避条件。**建议阅读并在现有服务做小流量验证。**

### Dynamo main：双容量准入、P/D失败收尾和索引内存

**项目：ai-dynamo/dynamo；下列10-22 main增量晚于1.6 tag。**

- [#14369](https://github.com/ai-dynamo/dynamo/pull/14369)（2026-10-22T06:31:12Z，`5df849a0`） 增加有界FIFO backend admission，同时保留“引擎请求总容量”和“等待首token容量”；队列请求不占引擎容量，观察到首生成token或完成才释放等待额度，响应stream结束/丢弃才释放总额度。每DP rank独立限额且受worker合计约束；默认过载HTTP529，worker标为overloaded。源码`admission_gate.rs`区分中间metadata与生成token，取消释放恰一次；无队列年龄过期或抢占，不能当带时限公平调度器。适合突发流量验证goodput，未给统一吞吐提升。
- [#16766](https://github.com/ai-dynamo/dynamo/pull/16766)（2026-10-22T12:15:25Z，`58e83f6d`） 在bootstrap P/D中将typed decode容量拒绝传回独立prefill context，并传播SGLang原生prefill错误/取消，避免一端继续跑或另一端永久等KV。模糊dispatch失败和客户端断开仍保留detached prefill，因为decode可能还需要数据；不能概括为任何取消都立即回收KV。建议对拒绝、early EOF及KV尾部故障做回归。
- [#17109](https://github.com/ai-dynamo/dynamo/pull/17109)（2026-10-22T17:23:11Z，`a4df2bc2`） `BlockLookup`把64bit hash拆成两32bit字段，将带NodeId的slot从16改12字节；编译期size assert保持此契约。作者两frontend、2048 mocker worker、每worker524288块、并发6144、均值82K输入/866输出、90秒单次同allocation测试，tables从63.4降47.8GiB；吞吐约-1.6%处在约3%漂移内，TTFT改善也不能归因此改动。这是CPU索引内存优化，建议大缓存集群复现，不能写GPU KV节省25%。

**状态提醒：** MiniMax M3.1 recipe [#17154](https://github.com/ai-dynamo/dynamo/pull/17154)（2026-10-23T00:29:10Z，`4b02e692`） 之后被 [#17155](https://github.com/ai-dynamo/dynamo/pull/17155)（2026-10-23T00:37:31Z，`f3b4b9aa`） 回滚。原配方提到的NGC tags尚未发布；截止main不具备该现成recipe，不影响模型权重本身的公开事实。

### Transformers：FP8 cache、稀疏索引位置与专家格式契约

**Hugging Face；均为main，尚未核实进入公开wheel。**

- [#48677](https://github.com/huggingface/transformers/pull/48677)（2026-10-22T11:23:56Z，`43cc0abd`） `QuantizedCache`新增`backend="fp8"`，以`float8_e4m3fn`连续保存K/V；首个update按K/V各自张量校准scale并固定，后续追加无需整数cache的全精度残留窗或周期重数量化。首次update返回原始K/V，后续decode每步attention前仍完整反量化；`QuantizedCache`当前只接收全full-attention模型，其他layer types显式拒绝。作者A10080GB、Qwen3-4B-Instruct-2507、1024输入+256 greedy输出：KV175.2→87.6MiB，峰值8098.9→7917.4MiB，耗时8396.0→9723.7ms，单prompt输出256/256相同。**建议复现长序列误差与动态范围漂移；当前是存储折中，不能声称加速或普遍无损。** 直接FP8 attention消费是后续工作。
- [#49840](https://github.com/huggingface/transformers/pull/49840)（2026-10-22T13:38:25Z，`5c65f7c6`） Qwen4-Exp的QSA历史position不再挂在顶层cache，而随`DynamicIndexedLayer`/`StaticIndexedLayer`管理。`reset/crop/reorder/repeat/select`让索引位置与历史K一致，修复跨请求reset、裁剪与beam重排可能误选key block的机制；这是缓存正确性修复，未给端到端质量增益。建议复用cache的服务回归这些生命周期。
- [#49848](https://github.com/huggingface/transformers/pull/49848)（2026-10-22T12:10:43Z，`9a6925a9`） fine-grained FP8配置显式携带`expert_dtype`，V4旧字段仅在quant config缺省时迁移；FP4打包专家按半K存储分配，避免V4.1的`expert_dtype="fp4"`落到默认FP8分配。源码`FP8Experts`从量化配置接收格式。masked expert parallelism仍是另一PR，不能据此宣称所有EP路径闭环。
- [#48825](https://github.com/huggingface/transformers/pull/48825)（2026-10-22T10:32:48Z，`a672eb5f`） 通过`modeling_utils.slice_logits_to_keep`把选位放到lm_head前。二维bool mask可按每样本不同数量挑监督位置并展平；二维整数是逐样本位置索引而非mask。329文件迁移已枚举，本轮重点核helper和选定模型，未逐个重跑。对仅completion有监督的训练有直接省大词表激活的入口；需同步labels/loss的展平与移位约定，不能把bool转整数后继续当mask。

### DeepSpeed与TRL：减少无用logits和CPU打包，保留训练语义

**项目与版本：DeepSpeed master、TRL main，下列为本窗落地。**

- DeepSpeed [#8839](https://github.com/deepspeedai/DeepSpeed/pull/8839)（2026-10-22T09:24:39Z，`ff34be65`） 在HybridEngineRollout图捕获prefill及显式static-cache分支，仅当`forward`签名声明时传`logits_to_keep=1`，避免分配B×T×V。作者随机权重4层Llama、hidden1024、vocab151936、BF16、单Intel Arc Pro B70，16×2048 prefill544.7→149.5ms、峰值9.84→1.27GiB，首token一致；小模型lm_head占比高，不能用3.6×估算真实大模型rollout。Transformers5.x的generate分支原本已优化，增益不覆盖该分支。建议同实际模型验证。
- TRL [#7798](https://github.com/huggingface/trl/pull/7798)（2026-10-22T12:54:04Z，`ea8cb1b5`） AsyncGRPO把CPU上超过98% padding的候选mask改成flat IDs+每位置counts，在device重建矩阵；sampler的logprob字典变数组，fused head/损失kernel保持原逻辑。作者R1-Distill-Qwen1.5B、128条最长8K、2trainer+2vLLM GPU，rank0 collate22.7→5.9秒/step；“安静step”37.0→22.1秒仅取before2步/after5步，所有步有1–8分钟权重同步噪声，不能把9.2K→21.1K trained tok/s当稳定因果提升。建议复现loss/gradient一致性及真实队列停顿。
- TRL [#7793](https://github.com/huggingface/trl/pull/7793)（2026-10-22T10:08:27Z，`a707c77f`） batcher在原group行满时，允许放进最轻且不增大全局最大load的行；`TreeRow.cost`用实际trie增加的load，避免把仍有空间的microbatch过早关闭。未报告统一性能表，建议对多轮共享prefix任务观察利用率。
- TRL [#7859](https://github.com/huggingface/trl/pull/7859)（2026-10-22T19:16:53Z，`e2dc4081`） Online DPO保留并传递`mm_token_type_ids`，生成completion补0作为text，使Qwen2/2.5/3-VL的多模态RoPE/logprob位置正确。涉及Transformers5.3起的输入契约，测试覆盖tiny三系列；建议训练者优先修复，不能仅靠loss不崩溃判断位置正确。



<a name="correctness"></a>

## 训练与推理底层：正确性与编译器

### FlashAttention：NaN尾部修复已延伸到反向，但最终方案必须包含3071

**Dao-AILab；[#3012](https://github.com/Dao-AILab/flash-attention/pull/3012)（2026-10-22T06:02:44Z，`2bdeddd1`）、[#3013](https://github.com/Dao-AILab/flash-attention/pull/3013)（2026-10-22T06:09:31Z，`767386d8`），随后[#3071](https://github.com/Dao-AILab/flash-attention/pull/3071)（2026-10-23T00:01:35Z，`10b74d98`）。** 昨日报告的SM100 forward尾部V NaN问题不是全部：SM100 backward中未使用K或V也能经`0×NaN`污染dQ；Hopper CuTe forward/backward亦同。前两PR用逻辑seqlen约束TMA读取，但它们沿用的环绕5D descriptor仍允许硬件预取越过tensor映射末尾，B200可出现`cudaErrorIllegalAddress`。

最终`10b74d98`移除环绕helper，改为每CTA复制descriptor，并在换batch tile时用真实row extent执行`tensormap.replace`和proxy fence。接口为每CTA/descriptor分配128byte workspace，按所有scheduler补齐grid上界并在越界时trap；只覆盖varlen/`seqused_k`的SM90/SM100相关CuTe路径，普通dense、paged KV与MLA按原路径，FA3 C++不在此次修改范围；`use_dedicated_hd256_kernel`专用backward亦明确排除，hd256套件通过不代表该专用路径同样修复。

作者B200 guard mapping独立子进程probe12/12通过，相关226个unused/seqused用例通过；部分大套件仍有与base相同的容差失败。kernel时间开销约0.2%–2.7%，约100µs host-bound形状wall time约+5%，H100未复现旧descriptor越界但共享修复机制。**建议相关训练/变长服务优先验证最终commit，而非只挑3012/3013。** beta36早于这些commit，不含此闭环。本轮审阅descriptor和接口补丁，未运行GPU实验。

### TensorRT main：隔离请求错误、KV同源与FP4 P/D

- [#19465](https://github.com/NVIDIA/TensorRT-LLM/pull/19465)（2026-10-22T16:37:35Z，`b3893e42`） `_forward_step`失败返回None后，非PP overlap/普通loop按empty-batch做对称收尾，避免错误response发出后executor再解引用崩溃。多模态验证hook可在调度前拒绝坏payload；forward时失败仍会波及同batch请求。最终代码还将仍被unpinned传输触碰KV的失败park到`_transfer_held_failures`，待`sweep`有访问终止证据后再失败/释放。**PP仍明确fatal**，不能泛称任意分布式前向故障都可继续服务；源码及新增executor回归可读，建议做真实故障注入。
- [#20742](https://github.com/NVIDIA/TensorRT-LLM/pull/20742)（2026-10-22T22:30:17Z，`82b61901`） GLM-5.3-Flash单engine MTP且无attention DP时关闭separate draft KV manager：glm slot table沿用target IDs，独立draft pool分配可能越界且host offload迁移不镜像。合并成同manager让slot和移动同源。作者TP4/EP4/MTP3 GSM8K观察旧分配95%行slot不同但acceptance相同，故不能把修复写成已证accuracy改善；attention DP及P/D原已不建独立draft pool。
- [#20544](https://github.com/NVIDIA/TensorRT-LLM/pull/20544)（2026-10-22T02:02:56Z，`2770bbfc`） FP4 MLA P/D新增Python+NIXL限定路线，传K/K-scale/BF16高精度tail，在首次generation forward前本地重建V-scale和packed-V。代码核`get_disagg_ignored_roles`、global layer映射计数与重建hook；其他runtime/backend显式拒绝。测试新增DeepSeekV3Lite NVFP4 GSM8K配置，但本轮未取得最终硬件CI闭环，尤其SM107不可泛化，建议观察并复现。
- [#20129](https://github.com/NVIDIA/TensorRT-LLM/pull/20129)（2026-10-22T08:54:08Z，`d4cfc12f`） Python transceiver进一步强制KV manager V2/NIXL，V1需CPP；这是部署约束收紧。[#19822](https://github.com/NVIDIA/TensorRT-LLM/pull/19822)（2026-10-22T04:21:32Z，`27842d51`） 则抽出deadline/watchdog和retirement仲裁，强化callback异常/重入处理，所有权六状态仍在native；**不是新非fatal隔离协议**，不能仅由代码搬迁宣称传输故障已全可恢复。

### Triton与PyTorch：精度、异步资源及自动调优副作用

| 机构/事件 | 本次具体变化、证据与限制 | 实践判断 |
|---|---|---|
| Triton [#12462](https://github.com/triton-lang/triton/pull/12462)（2026-10-22T22:48:26Z，`389dd29e`） | SM89默认非零`max_num_imprecise_acc`，阻止把独立FP32 add折叠进低精度FP8 MMA。作者L40S全1累加16384→20000，24个separate-add回归先失败后通过；10个matmul形状延迟+2.8%–12.0%，中位+7.4%。显式链式dot/参数0保留旧语义。 | Ada FP8用户优先校验数值；这是精度与速度折中。 |
| Triton [#12461](https://github.com/triton-lang/triton/pull/12461)（2026-10-22T20:53:04Z，`1ec1eee8`） | 同partition发起的异步MMA也须等scale消费完成后才复用TMEM，使用aref producer/consumer周期并在销毁barrier前等末消费者；变化scale/空循环回归。 | 自写warp-specialized FP8/FP4 kernel值得阅读，不能凭同warp就省掉异步依赖。 |
| Triton [#12341](https://github.com/triton-lang/triton/pull/12341)（2026-10-22T02:19:08Z，`93035c44`） | Gluon异步copy前端和IR verifier拒绝pointer/共享内存元素dtype不同，避免raw bytes错解释或misaligned fault；本地编译前端可核，C++ verifier rebuild由CI覆盖。 | 将静默错误前移为编译错误；NVIDIA与AMD路径均涉及。 |
| Triton [#12420](https://github.com/triton-lang/triton/pull/12420)（2026-10-22T22:49:45Z，`b10bbbd3`）、[#12419](https://github.com/triton-lang/triton/pull/12419)（2026-10-22T22:49:44Z，`2fecfecf`）、[#12421](https://github.com/triton-lang/triton/pull/12421)（2026-10-22T22:49:46Z，`941849d2`） | TMA view需覆盖CTA splits；ConSan按真实recipient布局及每issuer写入footprint记录可见性，区分合法不相交写与真正越界。 | sanitizer诊断更可信，不是LLM吞吐结果。 |
| PyTorch [#200358](https://github.com/pytorch/pytorch/pull/200358)（2026-10-22T04:09:17Z，`c693e1ad`） | 重建用户Triton Autotuner时保留`reset_to_zero/restore_value`，避免atomic累加输出跨benchmark污染首个真实调用。作者jagged attention backward旧dK约放大28000倍；源码四个重建路径传参。 | 训练正确性修复，已有自写autotune kernel建议优先检查。 |
| PyTorch [#200911](https://github.com/pytorch/pytorch/pull/200911)（2026-10-22T14:57:53Z，`c7c92789`） | max-autotune生成example storage时保留view相对input数据指针的负offset，分配能覆盖buffer和view的跨度，避免eager as_strided拒绝以及候选kernel读到分配之前。 | sliced/strided输入的编译与调优回归有价值，无通用性能倍率。 |
| PyTorch [#196195](https://github.com/pytorch/pytorch/pull/196195)（2026-10-22T16:05:49Z，`e04a100b`） | `GradScaler()`缺省device由当前accelerator解析，避免XPU/MPS无CUDA时静默禁用FP16溢出保护；CPU/no-accelerator保留历史warn/disable行为。 | XPU/MPS训练者应核is_enabled与溢出路径。 |
| PyTorch [#201488](https://github.com/pytorch/pytorch/pull/201488)（2026-10-23T00:50:19Z，`be5da999`） | AOTI loader对>48KiB shared-memory kernel设置prefer-shared，与static launcher一致；作者Jetson Thor限定BF16 GEMM/固定tile测试指出先前15%–30%差距。 | 针对AOTI部署验证同kernel occupancy，不可外推所有GPU或模型。 |

**回滚纠正：** 本窗PyTorch main [2f8c573c](https://github.com/pytorch/pytorch/commit/2f8c573ca4054bf2447175d3dfba67068dfecbed)撤回COW事件生命周期方案（原#200800），[b5a6f980](https://github.com/pytorch/pytorch/commit/b5a6f9805c93977afe7ff19fd3a3f2d3fcecbb17)撤回默认module-level Triton kernel（原#200502），另两preamble同栈亦撤回（[50121e3b](https://github.com/pytorch/pytorch/commit/50121e3bc425ab6bff4891a70a18080d5c5a974f)、[a9fc3918](https://github.com/pytorch/pytorch/commit/a9fc3918557937980955f882600b194ef120f557)）。昨日已报告的合入事实是历史事件，截止代码能力已变化；不将原方案当继续默认可用，待后续重落地。回滚commit在本窗默认分支清单，原PR正文仍可能保留正向设计，需按commit判断。



<a name="hardware"></a>

## 硬件软件协同与设备后端

### HAWK：把 GPU→NPU 的 KV 布局转换并入 RDMA，可接近同构服务吞吐

**ETRI、Georgia Tech 等；2610.29873v1，10 月 21 日 15:25 UTC，本轮补录。** 一手：[§4–5、Tables 7–9](https://arxiv.org/html/2610.29873v1)。HAWK 为 NPU 分配符合其 DRAM-channel/tile 布局的 RDMA buffer，将 GPU 连续 buffer 分别配对到目标片段，接收后直接消费；常规单 buffer 虽可直接传输，却因通道分布不均导致服务时 OOM。它依赖 NPU peer-memory/driver 改造，非仅替换传输 API。

Llama3.1-70B、每 phase TP4：prefill 4×A100 PCIe40GB，decode 4×p150a 对 4×A100 80GB，200Gbps IB、vLLM0.10.0/Mooncake0.3.6；ShareGPT 平均 783/140、LongBench 2,186/86 输入/输出，Poisson 到达。90% SLO 下吞吐距 GPU-direct 基线 −0.48%/−7.88%；TTFT/TPOT SLO 分别 2s/200ms 与15s/200ms。1.26×功效比只计 **ASIC 功耗**：NPU 只暴露 core power，GPU 由 board 测量和模拟估 ASIC 份额，未等同整机耗电；1.72×成本比用公开购价，不是云租赁 TCO。当前仅一个模型，代码计划开源、尚未作发布验证。建议深入阅读异构内存布局与注册机制，迁移前验证自己的 NPU/驱动和实际长上下文负载；不将 Tenstorrent 结果外推到昇腾。

### AMD AITER：大缓存寻址修复、受限W4A16与BF16递归状态

**ROCm/AITER；下列均为10-22 main，晚于10-19的v0.1.26发行。**

- [#6661](https://github.com/ROCm/aiter/pull/6661)（2026-10-22T17:04:11Z，`272586aa`） 修复gfx950 unified attention对packed K/V view只按`nelement`判断32bit offset的错误；最终函数为`max_addressable_bytes`，按stride实际跨度选buffer load。作者MI355X、Qwen3-32B+EAGLE3、单层2.6GiB cache的高block IDs原错，回归由56%不匹配转通过；不能把正文旧helper名当最终实现。
- [#5320](https://github.com/ROCm/aiter/pull/5320)（2026-10-22T14:07:54Z，`31c66db0`） VarCtx FP8 paged MQA的block×stride会越2³¹并让高block区域logits静默为0，改64bit指针`gl.load`；[#6758](https://github.com/ROCm/aiter/pull/6758)（2026-10-22T15:12:17Z，`0e83a830`） Triton split-K写/归约partials也在乘stride前转int64。后者MI325X、ROCm7.2.2/Triton3.6重现4096×36864×2048/16split越界，gfx950/1250 Gluon同类代码尚未纳此PR，不能说全部寻址闭环。
- [#6325](https://github.com/ROCm/aiter/pull/6325)（2026-10-22T16:06:38Z，`16efefb7`） **最终补丁只加拒绝条件**：KV block必须2的幂，V2还不能大于partition；PR旧正文讲“支持3/5/48和mask修复”，与一文件最终diff不符。建议依最终guard迁移，非2幂block支持仍待后续。
- [#6639](https://github.com/ROCm/aiter/pull/6639)（2026-10-22T13:42:43Z，`a0f7b103`） gfx942 FP8 persistent/blockscale fused-MoE修正overlaunch，并通过真实has-kernel查询避免给不支持model_dim选1-stage；[#6750](https://github.com/ROCm/aiter/pull/6750)（2026-10-22T13:00:02Z，`2a3ef2b3`） gfx950预编译qh64 MLA二进制去掉错误±64输出clamp并保留FP32 denorm。前者有参考与模型形状复现，后者没有可读kernel源，机制仅核二进制替换与作者probe，需额外独立验证。
- [#6778](https://github.com/ROCm/aiter/pull/6778)（2026-10-22T15:52:49Z，`b818d357`） 新`gemm_a16w4`支持GPTQ v1/v2、AWQ重排、group scale/zero与act-order，配置只覆盖gfx942/1100；MI300X和W7900各356pass/8skip。其他架构缺配置会失败，group32 decode更慢，MI300X prefill也慢于BF16 hipBLASLt。建议小batch内存带宽受限decode复现，不由4bit存储自动推断模型加速。
- [#6672](https://github.com/ROCm/aiter/pull/6672)（2026-10-22T18:19:22Z，`cf84c52f`） gfx950 Gluon KDA decode/spec/chunk可用BF16 state，内部recurrence与workspace仍FP32；gfx1250仍FP32-only。作者H12/KV128、batch64 decode约17.6→11.2µs，batch1则4.8→4.9µs，prefill基本不变；172pass/238skip、原RMS阈值不变不等于长期模型质量无损。建议验证长期状态误差与vLLM集成，不能把算子能力当默认服务已启用。

### FlashInfer：KDA训练CP与低精度MoE的公开接口

- [#7002](https://github.com/flashinfer-ai/flashinfer/pull/7002)（2026-10-22T21:09:43Z，`4c731817`） packed KDA训练新增CP，FP32 affine摘要跨rank传递状态；host plan把document切成fragment链，支持contiguous/zigzag/per-document ragged，配width4 conv halo及保存或checkpoint replay backward。作者GB300、H64、约250K packed token、CP2/4单节点至CP16四节点同NVL72域、CUDA13.0，两次session最慢rank median比较FLA KCP：模块边界倍率1.3058–2.4679×。这是forward或forward+replay+backward模块，不是完整训练吞吐。选定host plan/入口已核，22个大generated文件在PR API无patch，device kernel未全审。**建议长序列线性层训练研究者复现跨文档状态/梯度边界。**
- [#7023](https://github.com/flashinfer-ai/flashinfer/pull/7023)（2026-10-22T23:51:00Z，`7b7db109`） 同一packed KDA单设备BF16+FP32 gate变体更新SM100/103/107 kernel，作者42个shape×GPU组合的forward/backward/checkpoint边界1.435–1.892×相对前Cake版本；只在特定H、S与saved-set模式成立。与CP表是不同基线和测量边界，不能连乘成整体训练倍率。
- [#5487](https://github.com/flashinfer-ai/flashinfer/pull/5487)（2026-10-22T02:41:50Z，`eed63cd1`） SM100 CuTeDSL NVFP4 MegaMoE可传每token FC1 scale，恢复尺度发生在clamp/SwiGLU之前；每次staging验证presence/shape/dtype，避免工作区复用旧batch scale。[#6403](https://github.com/flashinfer-ai/flashinfer/pull/6403)（2026-10-22T02:32:43Z，`338b336e`） 把显式4over6 recipe贯穿TRT-gen/CUTLASS/CuteDSL，但per-token等资格限制仍在，不支持的组合提前拒绝。建议量化训练/部署者读参数契约；未给统一模型质量收益。
- [#6414](https://github.com/flashinfer-ai/flashinfer/pull/6414)（2026-10-22T20:11:50Z，`fb644f5e`） deferred routing weights按最终frontend配置返回，阻止漏乘/双乘；workspace复用要求capacity和live-token profile均相等。[#6894](https://github.com/flashinfer-ai/flashinfer/pull/6894)（2026-10-23T00:25:37Z，`ba8bd032`） 调优cache命中还需所有rank选择一致，现有一次reduction携带选择摘要，分歧时共同retire并重调；serving不额外加collective。建议多rank warmup/持久cache部署者优先回归，host tests不等于Blackwell全硬件验证。
- [#5651](https://github.com/flashinfer-ai/flashinfer/pull/5651)（2026-10-22T22:08:09Z，`0602a5b9`） sampling返回预留前的RNG offset，并为spec rejection恢复draw把预算从B×(N+1)改B×(N+2)，保留四draw对齐，防连续调用重用Philox draw。改变随机序列可复现契约，不宣称已测模型质量提升。

[v0.7.2rc6](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.7.2rc6)（发布2026-10-22T07:58:40Z） 的tag为`8152053015328128c177fc16df52ecb1fba5a78e`（06:45Z），晚于该时间的上述改动不在此发行。它只有compare链接，本轮已解析tag，不把main都装入rc6。

### Ascend：缓存预算和传输修复应优先于峰值优化

**vllm-project/vllm-ascend main，核心基线仍按其固定vLLM pin，不代表任意upstream版本可组合。**

| 事件 | 技术增量、证据和限制 | 建议 |
|---|---|---|
| [#19264](https://github.com/vllm-project/vllm-ascend/pull/19264)（2026-10-22T07:38:12Z，`9df8e7fc`） | Gemma4 MRV2全/滑窗attention在等物理页尺寸下共享allocation，预算与物理布局一同恢复，避免五/六group被缩到1/5、1/6。最终资格限Gemma4、dense FP16/BF16、compact layer/block、无KV transfer、PCP/DCP=1等；A5指定配置有容量回归，最终简化版完整CI未全绿。 | 优先复现目标配置容量与隔离，不放宽到所有hybrid/量化模型。 |
| [#19457](https://github.com/vllm-project/vllm-ascend/pull/19457)（2026-10-22T13:16:53Z，`9d39709e`） | GLM-Next已共享物理池的spec也免除generic per-layer缩预算，修1M context startup容量判断。最终shared exemption有CPU回归；正文A3硬件数字来自更早MRV1-only版本。 | 当前版本NPU/P-D/accuracy与吞吐仍待验，不把旧实验充当最终head。 |
| [#19251](https://github.com/vllm-project/vllm-ascend/pull/19251)（2026-10-22T10:51:38Z，`b5231d02`） | 移除worker每step RPC结果与scheduler hit缓存，每请求重查readability/lease/GVA，防DRAM→SSD迁移后过期命中；保留opt-in deferred last-save drain。作者Qwen3.8特定池设置命中93.68%、0失败、吞吐几乎同值。 | tiered KV pool正确性回归优先；单测试不证明所有负载零开销。 |
| [#19273](https://github.com/vllm-project/vllm-ascend/pull/19273)（2026-10-22T08:12:27Z，`d5923bcd`）、[#19480](https://github.com/vllm-project/vllm-ascend/pull/19480)（2026-10-22T10:49:23Z，`55ae17c2`） | bulk partial KV传输按请求token interval与每物理segment相交，修非零offset/多segment错误；layerwise PP同步注册扩展后的物理层数。 | 用部分块、PP和SSD组合核完整传输，不能只测full block。 |
| [#19164](https://github.com/vllm-project/vllm-ascend/pull/19164)（2026-10-22T03:46:39Z，`b012182f`） | mixed-dtype packed transfer解包先clone byte slice再view，避免对齐与alias；HCCL在packing stream提交但仍保留receive→sync→unpack屏障。失败drain会保留未证完成的buffer至进程退出，重复失败可耗尽内存。 | 训推权重同步优先回归异常/回调；不能把failed-drain保留列表当可无限重试恢复。 |
| [#18878](https://github.com/vllm-project/vllm-ascend/pull/18878)（2026-10-22T01:24:51Z，`353212d6`） | padded decode的CausalConv1d null slot从-1改新增非负专用state行，保留真实slot0，避免UPDATE不识别负null导致MTE underflow。 | GLM图填充批次故障修复，验证5live→8row replay及空batch。 |
| [#18931](https://github.com/vllm-project/vllm-ascend/pull/18931)（2026-10-22T08:39:29Z，`7b13f387`） | MTP+PP末stage从共同checkpoint加载embedding并保留独立MTP权重优先，零元素RoPE/NoPE组件跳过zeroer。作者Atlas800IA3/16×91064GB、CANN9.2、TP8PP2/EP/MTP5、GSM8K前160题：217.87→508.70输出tok/s，acceptance1.358→4.080。 | 这是故障修复前后特定配置，非一般2.3×优化；需核权重和PP加载顺序。 |
| [#19364](https://github.com/vllm-project/vllm-ascend/pull/19364)（2026-10-22T10:31:43Z，`ecbf20d2`）、[#19220](https://github.com/vllm-project/vllm-ascend/pull/19220)（2026-10-22T01:26:37Z，`59746539`） | MRV2接入独立draft model；DSpark SparseFlashMla限定A5+BF16 KV，以logical top-k和[T,1]长度建立metadata。前者400 MT-bench prompt、Qwen3-8B/1.7B eager输出1869tok/s没有target-only对照和完整硬件条件。 | 新能力可复现，不据acceptance/单次吞吐声称加速。 |
| [#18020](https://github.com/vllm-project/vllm-ascend/pull/18020)（2026-10-22T03:24:34Z，`4831904e`）、[#19408](https://github.com/vllm-project/vllm-ascend/pull/19408)（2026-10-22T08:56:12Z，`06c14a32`） | EPLB map用pinned CPU+nonblocking copy减少host drain；GLM复用conv输出并清一次metadata padding，移除重复KDA clear。最终GLM shared-buffer硬件验证仍待，旧accuracy表只覆盖较早clear移除版。 | 建议观察最终head的HCCL/graph消费与缓冲区寿命，暂无模型吞吐证据。 |

### DeepSpeed CPU Adam：RISC-V向量化接入，收益限FP32状态与单核优化器步

**DeepSpeed；[#8817](https://github.com/deepspeedai/DeepSpeed/pull/8817)（2026-10-22T03:01:41Z，`11ac9f86`）（master）。** riscv64主机有RVV1.0时自动选择FP32参数/FP32状态的向量Adam热循环，m4按vlmax处理、尾块截断无需scalar tail；其余dtype组合保留scalar。builder解析ISA扩展并选rv64gcv，避免仅字符串子串误判V支持。作者SG2044/VLEN128/gcc14.3/torch2.15 CPU，单核、100step：1B参数13.07→3.15秒，2B26.13→6.31秒，约4.1–4.2×相对原scalar；48pass/1skip。**建议RISC-V offload研究者考虑复现**，这是单核optimizer kernel，不包含GPU通信、整机多核带宽或端到端训练吞吐；FP16/BF16路径并未由此向量化。

### TorchTPU/XLA与Apple：低精度执行、重算屏障及设备内存

- TorchTPU [b7469fea](https://github.com/google-pytorch/torch_tpu/commit/b7469feac7baf0355bd4e89f38513f7c1a9048e9)（2026-10-23T00:04:34Z） 保持三类ragged grouped-mm的原生FP8输入进入MXU，累加/scale epilogue仍FP32；TPU专用Meta放宽不必要的上游stride/transpose限制以支持torchao FP8 MoE的compile路径。[724193ce](https://github.com/google-pytorch/torch_tpu/commit/724193ce40ccbc3dc4c9a171463fc7a383915957)（2026-10-22T21:40:31Z） 补rowwise/per-group scale形状。是低精度训练接口落地，未给模型端到端效率/收敛表，建议MoE小规模验证。
- TorchTPU [923254bf](https://github.com/google-pytorch/torch_tpu/commit/923254bf39b7d1ed50b501a2f21153ada4e6c0af)（2026-10-22T18:14:40Z） 将每叶输出barrier改为反向checkpoint region入口的单个variadic barrier，绑定边界激活、参数及incoming gradients，避免CSE合并重算回forward并让激活一直存活，也防重算提前调度。源码重建flat backward区域并拒绝interleave；旧PyTorch元数据另补。**建议对重算训练深读并用XLA HLO/峰值内存核实，未提供通用节省百分比。**
- TorchTPU [c572c9ce](https://github.com/google-pytorch/torch_tpu/commit/c572c9cecec61051ca7aa8320e767c42ea6a2258)（2026-10-22T14:36:47Z） 修size1 query/key broadcast mask被按整tile读取，改1行/列加载后VMEM广播，避免错误值/NaN；[056815da](https://github.com/google-pytorch/torch_tpu/commit/056815da8701a9f9d03c544bcfa342b33afb7a8b)（2026-10-22T12:32:31Z） 仅底层transfer utility统一异步启动，CPU↔TPU调用者仍按`non_blocking`选择等待并登记事件/错误，不能写所有用户copy默认异步。
- XLA [3421a9e0](https://github.com/openxla/xla/commit/3421a9e0f4f5089161647f972351dddf85d8fda2)（2026-10-22T20:20:03Z） split-K结果继承原dot物理layout且split维最major，避免reduction附带转置；[d37f27c2](https://github.com/openxla/xla/commit/d37f27c26ed4992ff2481f7b11574fd6298211bf)（2026-10-22T11:02:39Z） 允许≤128 scan维的producer融合（例如softmax→top-p cumsum），**默认flags无效果**。[b6ebee76](https://github.com/openxla/xla/commit/b6ebee7600fc36f41f98ae7d044752d323dcf6c7)（2026-10-22T13:46:23Z） RaggedAllToAll仅显式允许NCCL fallback时分配相关host临时buffer；[fdddccf8](https://github.com/openxla/xla/commit/fdddccf86ae6a3bd114f0cd6582008495a720b73)（2026-10-22T13:11:21Z） 未启用experimental symmetric flag时window注册用NCCL_WIN_DEFAULT。都是主线特定边界，未测LLM收益。
- Apple MLX [#4712](https://github.com/ml-explore/mlx/pull/4712)（2026-10-22T08:22:20Z，`c6abf0db`） 为6–16行量化矩阵向量组引入qmv_mma/qmv_nax，每块权重读取一次后矩阵单元消费。作者M5Max、Qwen3.8-27B NVFP4、512输入/128输出、batch16生成149.5→298.4tok/s，batch1无变化，prefill无观察到提升；适合并发/spec形状同机复现。MLX-LM [#1955](https://github.com/ml-explore/mlx-lm/pull/1955)（2026-10-22T09:45:57Z，`33fdd163`） 在convert保留NVFP4 global scale FP32，dequantize按专家scale恢复，避免默认BF16保存后首forward失败或权重尺度错误；对Muse Spark Open等专家转换优先回归。

### DGX Spark八节点：小消息通信与小模型预训的公开系统实验

**NVIDIA，官方技术文2026-10-22：[原文](https://developer.nvidia.com/blog/running-frontier-class-ai-models-across-8-nvidia-dgx-spark-nodes/)。** 本次可读增量是8×128GB统一内存拓扑/实验与小消息collective权衡：RoCEnante采用all-to-all交换后本地reduce，减少小消息通信轮数，增加每rank发送量，要求switch任意点互通；大消息/高并发/大rank仍可能由NCCL更合适。作者称已并入FlashInfer，但本轮未定位到对应PR/commit，因此只作厂商披露，不提升为独立核实的已发行能力；NVIDIA Sync八节点Cluster Assistant仍计划年底提供。

同精度NVFP4 Qwen3.8-27B、无spec/MTP、llama-benchy、8192输入/256输出、并发1：单Spark11.86输出tok/s，八节点switch+该backend52.58（4.43×）；八节点物理ring40.11（3.38×），比较混合拓扑和backend，不能把全部差异归通信算法。完整753B GLM5.3约470GB NVFP4/MTP3在八节点并发1/4/8的单用户decode31/21/18tok/s，不能拿另一个Flash模型约80tok/s替代。约896.5M nanochat、20层hidden1280、序列2048、global batch524288token、每GPU microbatch32seq，作者单→八节点8741→68079训练tok/s（7.79×）；这是小模型数据并行scaling，不代表大模型/不同global batch效率。

**建议网络/内存受限本地研究团队深入阅读并复现**，文中链接[训练代码](https://github.com/zyang-dev/spark-nanochat-multinode)与[网络配方](https://github.com/eugr/spark-vllm-docker/blob/main/docs/NETWORKING.md#automated-cluster-setup)，本轮未审该两仓固定commit或复跑，后续需先固定版本。路由示例92case的质量及1.5–3×吞吐是工作负载分流原型，不能当普遍同质量收益。

### GPUDirect Storage新官方指南：成功read不等于走了GDS

**NVIDIA；2026-10-22：[KvikIO/cuFile实操](https://developer.nvidia.com/blog/how-to-read-data-into-gpu-memory-in-python-with-nvidia-gpudirect-storage/)。** 新文把Python训练输入/权重加载的路径验证讲清：`KVIKIO_COMPAT_MODE=off`只要求使用cuFile；cuFile仍可按自身兼容模式走CPU/POSIX。需在进程存活时打开`CUFILE_STATS_LEVEL=3`并用`gds_stats`核nvfs/posix计数。作者1GiB文件、默认4MiB task分成256read，nvfs256/posix0才给出该次路径证据；成功返回完整字节数不足以证明直达。

KvikIO task/thread与cuFile内部并行是两层调度，任务小于内部8MiB阈值时不会按预期再拆分。指南不是新存储算法或LLM训练性能报告；**建议有实际I/O瓶颈者按指南定位数据路径**，无LLM端到端吞吐对照，不能宣称GDS自动解决训练供数。



<a name="evaluation"></a>

## 评估与可靠性

### Claude把fuzz入口变成可重放Rocq证明：公开子集与“终止/全库安全”边界必须分开

**Anthropic与Theorem；2026-10-22发布，公开proof工件`theorem-labs/oss-verify-proofs @ 8f5f46768858af236a4270f9628e6beed9aa0e72`。** [官方研究文](https://www.anthropic.com/research/formal-verification)、[固定工件README](https://github.com/theorem-labs/oss-verify-proofs/blob/8f5f46768858af236a4270f9628e6beed9aa0e72/README.md)、[校验步骤](https://github.com/theorem-labs/oss-verify-proofs/blob/8f5f46768858af236a4270f9628e6beed9aa0e72/docs/checking.md)。问题是fuzzing只能采样输入，难触及大输入/极罕见状态；该流程把既有OSS-Fuzz harness及sanitizer检查编译为RISC-V二进制，生成形式语句，模型写证明、Rocq内核检查。作者称几周内完成239入口/123库、约1.8亿行Rocq，并在证明过程中定位、修补bug；这些是作者全量实验，并非公开可下载全量。

公开release2026.10.0实际为**60入口/25库、138条agreement证明**；2,148入口的20,164条语句只是待证明规格清单，不能计为通过。其声明是在内存预算等前提下，plain与ASan/UBSan版本具相同可观察行为且sanitizer不报错，针对指定静态riscv64/musl二进制及LPS语义。关键限制：README明确**本次没有任何halts证明，不能排除永不终止**；不能把博客“always exit normally”的概括直接套在这批公开agreement结果上。也不覆盖sanitizer未检测错误、未达代码、整个库、x86/ARM动态链接部署。LPS及真实OS/ISA的偏差、theorem陈述、Rocq和ELF转换/check wrapper均属可信边界。

作者另用36小任务比较每题3次独立单agent、每次5M-token上限：Opus5.5达到约一半成功任务所需token较Opus5少约40%；共享proof工作区声称少17%token。它们是特定长程证明任务效率，非所有代码生成、训练算力或单请求时延结论；没有公开完整成本与失败分布。允许模型修patch还可能将被证程序改弱，作者说对patch惩罚/审计，不能只信checker消除了规格投机。

**建议深入阅读并先重放公开证明**：固定Rocq9.2、LPS `f1006fe8178338a499b5e5c41be83006c4edd058`及checker容器digest，校验二进制hash、定理wrapper和允许axiom列表。文档称16核单proof最多约20GiB；`rocqchk`独立二次校验为可选，当前CI verdict不运行它。本轮未执行proof重放，不能称独立验证成功。此工件适合作为长程agent、证明复用和规格忠实度研究基线；采用前先对照实际部署平台与要证明的性质。

稳定ID：`anthropic:formal-verification:2026-10-22`；`github:theorem-labs/oss-verify-proofs@8f5f46768858af236a4270f9628e6beed9aa0e72`。

### “Wait”更多未证明RL发明反思：标记来源与错误告警含义需要分开测

**港科大、华为诺亚、北大；2610.30882v1，10 月 22 日提交。** 一手：[§3–5、AppD、Limitations](https://arxiv.org/html/2610.30882v1)。跨checkpoint追踪lexical revision marker来源，再向正确prefix注入同类算术错误，测marker的signal-detection敏感度。18个0.5–32B数学outcome-RL run中，从缺marker的base开始多数始终<.05/1Ktoken；mid-training／reasoning-rich预训练／trace SFT更常为来源，RL再放大。

OLMo3 RL-Zero的错误敏感度d′ .57→1.09，但criterion .11→−.24，对无错prefix也更常报警。60句盲审consensus的revision rate1.3→15.9%，一个LLM judge却测26.0→12.9%，反转方向；另一个judge方向正确。**provenance仍观察性，系列多为单训练run，区间是evaluation采样；只测词形而非底层反思能力**，人审亦非独立双标。建议深入阅读并将base marker频率、可控错误和人审抽样纳训练分析；不以“Wait词多了”证明新能力，也不反推没有marker就没有纠错。

### Subliminal learning延伸到预训练base：换初始化、数据或SGD并不足以阻断隐性偏好转移

**EleutherAI SOAR、Cincinnati及独立研究者；2610.30884v1，10 月 22 日提交。** 一手：[§2–3、AppA4/A6/A14–17](https://arxiv.org/html/2610.30884v1)。用带动物偏好的teacher产生过滤后数字串作hard-label蒸馏，或在无显式偏好的学生trajectory上提供soft logits，控制初始化、data order、corpus、size与优化器；检验偏好跨看似无关数据的传递，而非笼统“知识泄漏”。

Pythia160M–6.9B、OLMo1B及Qwen兼容tokenizer配对，soft transfer在30配对中26个出现；hard transfer也能在full-FT+plain SGD发生，须足够LR扫，不支持“adaptive optimizer必要”的概括。Pythia把teacher数字边际频率重采样仍保40–92%转移，OLMo仅5–36%，机制不能统一；跨预训练13×13 grid拟合teacher/student因子可解释90–97%交叉验证变化，但不是因果分解。论文还发现PolyPythia某weight-seed发布实际没换初始化并自行重训。建议深入阅读校验与小模型复现，distillation审计同时看显式词过滤和目标偏好；目前人为偏好/小base证据不等于现实所有有害特征都可如此迁移。

### 文档LoRA open-book高分多在测阅读行为，随机外来adapter可恢复82%的专属增量

**Bosch、LMU、MCML；2610.31017v1，10 月 22 日提交。** 一手：[§2–6](https://arxiv.org/html/2610.31017v1)。150篇英文百科、Llama3.1-8B-Instruct/rank8/4bit训练，对同一document换specialist、foreign或random adapter，并加不提供文档的cloze probe。random adapter可恢复specialist相对base的open-book增量82%[77,87]，cloze增量2%[−19,17]；同域foreign恢复88%[77,101]，因此大部分open-book优势不能归文档特异记忆。

同时给出四个会制造“漂亮机制”的silent bug：默认1024截断使44%文档所有训练样本都丢答案、缓存混用repetition penalty、自己将LoRA两因子均乘1/N造成1/N²缩放、4bit训练却merge到BF16评测。**正文结论基于修正后统一配置；PEFT合并实现无该缩放bug**。单model/rank/genre、每文档8字符串重复约1000次，不能推全部LoRA不存知识。建议深入阅读并复现context-swap、closed-book与长度/precision审计；不要以open-book合并保持率直接宣称记忆成功合并。

### minItems会让抽取模型补够数量，来源字符串检查也拦不住语义错项

**Tribhuvan University、St. Xavier’s College；2610.30997v1，10 月 22 日提交。** 一手：[§3–4、AppM](https://arxiv.org/html/2610.30997v1)。558真实文档、九数据集、八open-weight模型、k1/3/5，配对仅展示schema minimum与强制decoder minimum；greedy、vLLM.31/XGrammar.2.7，BF16或gpt-oss原生MXFP4。n<k时，仅展示minimum就使over-count34.5→55.4%；新增项常来自真实原文，约半数可过source occurrence检查，却不符合所问类别。

“fewer is fine”平均逆转补项但真项漏掉率17.0→24.3%；将短列表连validation error重问，Qwen9B过量10→69%、KimiK3 14→78.3%。**结果限string-list抽取和所测open-weight／host服务，非所有closed API；允许N/A仍半数补项，thinking未修复**。建议深入阅读并移除语义上无依据的minimum，验证后重试也需允许真实少项；schema validity、source-groundedness与任务准确率应各自测。

### Reward跨prompt分数不可由pairwise accuracy背书，盲目按prompt居中还会损伤选择

**Warsaw University of Technology；2610.31090v1，10 月 22 日提交。** 一手：[§3–9](https://arxiv.org/html/2610.31090v1)。同prompt两答案排序会消去所有prompt-shared offset，无法验证跨prompt绝对分数。作者用between-prompt variance share、换错prompt及sibling人评回归作诊断，再直接测试全局筛选的目标质量，而非把prompt clustering自动当bias。

六7–8B RM在HelpSteer2 between-prompt份额.34–.64，超对应人评，但HelpSteer独立评分reference本就有.26份额。挑top30%时，按sibling均值居中反而让HelpSteer有用度降.14–.21；独立12参考响应居中仍降.05–.33。0.6/1.7B、五seed同base/data训练臂表明loss和target幅度信息共同影响错prompt辨识，pairwise shift invariance不能独自解释。**英语、每prompt仅2–4人评响应，未重复训练全部released RM**。建议深入阅读并给跨题数据筛选单独校准；不要因pairwise reward benchmark高就直接全局混排，也不要未经目标验证自动减prompt均值。

### 工具可能提高均分却放大prompt换写下的模型排序波动

**Warsaw University of Technology；2610.31148v1，10 月 22 日提交。** 一手：[§3–6、SupplementS4](https://arxiv.org/html/2610.31148v1)。在分子、图、音乐、表格、数学上分离first-token/answer-line读出、reasoning、是否call、工具结果及直接提供报告；标准化同一题多wording下的两模型分差，而非只比较各自accuracy。

plain-text工具环相对无工具reasoning，使37/60分子model-pair的wording spread变宽，其他域也有部分；音乐／分子15对各4对在不同wording间出现>±.05的排序翻转。**native tool interface能消除分子域这类widening，工具有无不是唯一原因**；图first-token改answer-line即可把median spread.25降至.05。多数结果greedy，同模型/题目pair不独立，多wording平均选择损失仍较小。建议深入阅读评测分解，报告accuracy、分差、call rate与接口；不要从工具平均提分推“工具让排名更可靠”。

### Context augmentation现实检查：多加解释能提分，也可能同时打破原来正确的预测

**Copenhagen、Nokia Bell Labs；2610.30983v1，10 月 22 日提交。** 一手：[§3–6、Limitations](https://arxiv.org/html/2610.30983v1)。七个1–72B instruction模型、MedQA/MMLU有多句背景子集/DRUID，将空白、重复、改写、自解释和teacher解释插在单句后；用rescue/break与sign-randomization检验净accuracy变化是否体现稳定朝正确答案移动。

MedQA teacher解释六模型增2.0–9.3pp，但最大两增幅的方向检验未拒绝null；21个model-dataset对仅8个teacher、6个self解释通过方向检验，无信息repeat/blank各4。MMLU某配置净+.1pp同时改变19.1%位置的预测，净分会隐藏反向扰动。**这不是“解释全部无用”**：部分模型/任务有效；实验为单句单次干预、选项log-likelihood，不覆盖整段解释、多轮组合或开放生成，且未纳reasoning模型。建议考虑复现自家业务的rescue/break和干预位置选择，不能只用一次均分增长判断可靠采用。

### ReflectPO：从可恢复环境快照采纠错分支，短窗找循环、完整prefix树分credit

**美团；2610.30933v1，10 月 22 日提交。** 一手：[§4–5、AppA](https://arxiv.org/html/2610.30933v1)。归一化action/observation签名，保存动作后的完整环境状态；用轨迹差异与重复短窗选节点，由更强guide产reflection hint、从snapshot重走恢复轨迹。原轨迹在完整prefix tree上由叶结局给action advantage，恢复分支用与原目标轨迹的reward差；找相似短窗与credit树不能混成一种状态等价。

Qwen3-4/8B、EnvScaler训练，BFCL-v3 Multi-Turn与τ²Bench，overall33.2→36.6、39.6→45.4，8B BFCL+6.3pp/τ²+3.8pp。**额外guide与recovery rollout是成本，收益未等于零额外计算；可精确snapshot/restore含user-simulator内部状态，是迁移真实tool环境的重要条件**。同签名未必蕴含外部世界完全同态，论文主表不足以证明任意SWE/真实用户都能重放；独立训练seed稳健性仍待检验。建议考虑复现已有可控环境的错误恢复，计入guide/重放总成本，再评估生产部署。

### SANE：按 scorer 的真实规则生成变形测试，比通用固定库找到更多缺陷

**Cisco、Anduril；2610.28886v1，10 月 20 日提交，本轮补录。** 一手：[§4–6](https://arxiv.org/html/2610.28886v1)。先解释每个judge/scorer意图，从通用行为要求生成scorer专属metamorphic recipe，再验证recipe和实例是否真满足预期不变／有方向变化；用相同生成模型重建JudgeReliabilityHarness作对照。

五scorer上SANE133候选中77个validated family，固定库35中21，转化率57.9%对60.0%；**优势来自适用recipe多，不是每条更容易验证成功**。31对10个family发现至少一故障。11个judge的faithfulness排序与FaithBench人标AUROC Spearman .852（精确置换p=.0015），固定库.100；这只验证judge排序，非逐测试case的人类正确性。验证panel仍是LLM，未独立人审oracle错误率，其他四领域也缺人标meta-eval。建议深入阅读并用于上线前测试reward/judge的具体实现；新增fault数量不等价于已证明的模型缺陷率，需人工抽检。

### FrontierCS 2：在论文原仓库里补回贡献，比对原实现的全部性能／质量门槛

**UC Berkeley、Princeton等；2610.29932v1，10 月 21 日提交，本轮补录。** 一手：[§3–5、AppA–D](https://arxiv.org/html/2610.29932v1)。48任务多数从作者代码中删去贡献实现，保依赖和调用入口；agent只能编辑指定文件、5小时、无公网，development judge给聚合反馈，最终只测agent指定的一份patch，在新seed/query/graph上与恢复原代码的private参考比较。pass需所有指标／所有workload都过阈值，单一final score高于参考未必pass。

12个模型各配一个harness，每任务仅一run，另一个anchor agent参与FCI拟合；90%区间对任务bootstrap，非重复agent run方差。FCI结合逐metric胜负和相对分差，10点/一个logit、KimiK2.7为0；不可直接换算为“研究员百分位”。预算曲线只让预算内完成的最终提交得分、超时一律0，单调性由定义决定；价格估计尤其影响低budget排序。建议深入阅读任务和隐藏评测设计，用于评估系统研发代理；当前同时测模型+harness、原仓库先验与受限补实现能力，不代表从零自主发现论文贡献。



<a name="versions"></a>

## 模型与数据版本观察

- **Open-SWE-Traces默认配置变化，不是本窗新轨迹。** 截止内固定[卡`46d5987b`](https://huggingface.co/datasets/nvidia/Open-SWE-Traces/blob/46d5987be9241414edf7ca133ace3dda18632ff8/README.md)把既有Qwen3.8-Flash-Next轨迹从v1.2分离为默认v2.0；对主窗前基线tree只README变化，无新增parquet。依赖默认config的训练流程应显式固定版本；v1.2现在只指Qwen3.8-27B。01:01:24Z的后续卡提交越过本轮截止，未用于正文。
- **Flora三模型统一研究许可。** 10-22[Rewarder卡](https://huggingface.co/nvidia/Flora-Rewarder/blob/031299e46e452fc0cedb4e0975eb9fa95fc37783/README.md)、TempSeg与Captioner提交统一NVIDIA OneWay Noncommercial License（2022-03-22）；Rewarder前后tree仅README变。属许可归属修订，不是新权重/新性能；物理AI视频奖励模型侧线未作完整训练与代码审计，准备采用者应重新核该revision条款。
- **Ming Audio Flash部署卡改4卡配置，长音频仍留8卡回退。** [10-22固定卡](https://huggingface.co/inclusionAI/Ming-3.0-audio-flash/blob/77e851a4675a93181c30f50426507d076cc55a1b/README.md)把128K reference profile由8×80GB H800改4×80GB，并明确长音频可能OOM、需切`*_8gpu.yaml`；本轮tree仅README变化，未核部署脚本/运行，不能称任意128K音频4卡保证可用。TTS Tiny另更新vLLM-Omni说明，未新发权重。
- **其余卡级更新已筛去。** Beam FP8只同步既有NVFP4 LCBv6分数修订；Glimmer新增eval元数据，d3改max-model-len示例，Dolci-Think7B纠正32B链接，均不重报模型发布。Dolci Hybrid Tool-Use实际删2条无user-turn样本（1597→1595），作为数据清洁记录保留，未扩成训练能力增量。RE-USE两仓本窗tree改变为文档/演示音频等、权重未变，卡标10-22发布日期但不足以重建首次公开时刻，未写成当天新权重。



<a name="coverage"></a>

## 覆盖、补录、更正与归档


### 本轮模型扫描覆盖说明

实际重查40个HF机构（包括上一轮名单外RecursiveSI），模型/数据集各按createdAt与lastModified两种排序，160页请求，均到72小时边界或公开库存末端，合并102模型/27数据候选；本窗模型28、数据6。卡/commit/tree、权重与文档对象比对均只对明确选择的主窗候选及回溯重点开展，**列举完成不等于全部内容语义全审**。旧SHA与已收录ID对照后未重复报告Flint、DSpark、Youtu、Ming TDS等已收录事件；Perplexity报告为明确标日期的回溯补录；论文侧发现IQuest-Q1新报告后，补查IQuestLab既有权重/commit/配置/许可，仅作目标工件核实，未声称该新增机构全库存已扫描。

官方实际打开/请求了OpenAI、Anthropic、Google/DeepMind、Meta、Microsoft、NVIDIA、DeepSeek、Qwen、Seed、Kimi、智谱、MiniMax、腾讯、百度、Step、小米、美团、Mistral、xAI、Ai2、Reflection与Inception入口，并用中英文主题检索补充。Anthropic新研究页实际读正文及公开proof仓；Google最新森林类型制图与生物/健康应用不作LLM技术增量收录。OpenAI/MS研究列表403，OpenAI Decisions正文经web打开且官方开发文档200；Mistral/xAI部分入口两次超时，Ai2列表403，Kimi入口308，Qwen博客/百度/Step仍有JS或轻量壳，不能由这些结果断言厂商没有发布。MiniMax的SGLang/vLLM新配方404、IFM两rollout数据卡仍401，旧欠账不关闭。完整逐源进度、候选去留、原始响应hash与尚未审查候选见本轮`models-coverage.json`；未运行GPU推理、未下载大权重或大规模数据、未独立重跑所引指标。

### 论文来源覆盖、补录与未完成区间

本轮论文共收录40个此前未归档的精确ID/版本：19条10月22日主窗新提交、21条72小时回溯补录。直接打开各条arXiv版本的机制、实验设置、结果表与限制；均为作者报告，未作独立训练／部署复现或全面代码审计。已查cs.CL/LG/AI/DC/PF/stat.ML的10月22日公告列表（1,070次出现、748唯一ID、289替换），另取cs.CV列表、HF10月22日50条精选；精选日期不当发布日期。

arXiv精确／扩展日期API多次429、timeout或断连；官方advanced search恢复language model130条、quantization7条、pretraining30条主窗日级关键词结果，但其中含旧论文修订，**不能据此声称全类别、精确时间窗完整枚举**。289个替换恢复250条metadata，余39条请求失败；已对2610.21810v2→v3做归一化正文比对，无文本技术增量。2605.11125v3→v4出现表述／证明和显式bootstrap、PPL/entropy协议说明，保留版本审阅记录，未冒充新性能。

TMLR列表相对昨日新增7个forum ID，仅知October月份；OpenReview API及新试两PDF403，forum200只是动态shell。ACL/PMLR根目录已重查，未完成逐volume精确日期／版本核查。仍有860个去重发现候选未完成语义审阅、249条已取替换metadata待进一步筛审，既有319/328／CV-stat56及更早待办原样保留；当前日级关键词入口不关闭2026-10-22T00:58:39Z—2026-10-23T01:00:15Z的全类别精确提交枚举缺口，也不关闭2026-10-19T00:58:39Z起的全球修订枚举缺口。

### 核心六仓检查结果

核心六仓覆盖：主窗北京时间2026-10-22 08:58:39至10-23 09:00:15，回溯至10-19 08:58:39。GitHub认证连接器GET已逐页枚举all-base/base:main merged PR、main committer-time commits与全部公开release collection，Search incomplete_results均false；库存抓取无失败。语义为选择性审阅，不能声称所有PR/源码已审。

| 仓库 | 主窗PR all-base/main | 含回溯PR all-base/main | main commits | release对象 |
|---|---:|---:|---:|---:|
| NVIDIA/Megatron-LM | 18/16 | 56/47 | 49 | 51 |
| THUDM/slime | 1/1 | 4/4 | 6 | 11 |
| pytorch/torchtitan | 7/5 | 44/41 | 41 | 5 |
| sgl-project/sglang | 106/93 | 374/357 | 357 | 63 |
| verl-project/verl | 9/9 | 20/20 | 20 | 16 |
| vllm-project/vllm | 91/90 | 325/324 | 325 | 108 |

合计：主窗232/214，含回溯823/793，main commits798、公开release对象254。主窗无新release发布；SGLang0.5.22为回溯已报旧发布。正文72个PR分为56组，另有TorchTitan官方公告；所选main PR merge SHA全部位于commit库存，Megatron8232明确dev。72项与本地历史URL/稳定ID无重复。

重要变化：Megatron全局batch quantile balancing、GTP图下专家链与MTP/BI显存；verl MXFP8端到端配方与teacher/student CP静默错配；vLLM XPress、packed NCCL权重损坏、P/D容错；SGLang LoRA计划栈/KVShard、HiCache forward→D2H与HiSparse取消生命周期、预填图DP死锁和SM90 tail；TorchTitan128GPU RL配方、timeout样本边界、默认编译接线。

本轮原始证据纠错（含时间分类更正）：SGLang42701实际write_fence_stream；43792最终先发KV gather再发index-key gather；43216地址复用仍为suspected，非根因已证实；42803用live length sum上界而非均值；TorchTitan5152默认仍CuDNN后Flash，未默认强制Flash；SGLang45631直接FP32 logits在45855中收窄；Megatron8345表值相对eager约21.9%，PR另称19%不一致，35.45%是相对旧CG。

待办保留：P2P61667、XPU43299、chunk-stash45580、Dory60149、Beam61978、BSHD蒸馏8061均仍open。P2P main仍10秒abort-ack timeout，未验证late WRITE/quarantine/RDMA serving闭环。未选实质候选逐仓在core-coverage.json，历史10-05→10-08等旧语义缺口继续保留。SGLang LoRA141文件及大kernel仅负载关键路径选择性审阅；无独立benchmark/框架测试。本轮两项公告旧代码路径探针失败（flex_shard.py/checkpoint.py），不影响库存成功；FlexShard/torch_checkpointing全实现尚未审，recipe imports与GraphTrainer支持矩阵已核。

TorchTitan官方公告HTML记录：首发2026-10-21T12:45:44-07:00，即北京时间10-22 03:45:44；修改2026-10-22T13:33:58-07:00，即北京时间10-23 04:33:58；首发早于本主窗起点10-22 08:58:39，仅在72小时回溯内；修改落在主窗。首发为72小时回溯补录；本主窗内仅确认修改时间，未取得修订前完整正文，无法精确还原修订增量，当前内容不得写成主窗首次披露。代码核查基线4103c7291ffeb568ebf78a522b5fb2a8c01f9b06。

### 扩展工程与硬件来源的实际覆盖

已用GitHub连接器枚举24个扩展/设备仓默认分支的72小时commit，去重后1944条，主窗457条；主窗merged-time检索464条，并读取每仓最近20个以内的公开release列表。包括PyTorch、DeepSpeed、Transformers/TRL、OpenRLHF、Triton、FlashAttention、TRT-LLM、vLLM-Ascend、FlashInfer、Dynamo、llm-d、NCCL/NVSHMEM、AITER/RCCL、Ascend/PyTorch、XLA/JAX、MLX/MLX-LM、TorchTPU、NKI samples及Run:ai。64项选定PR取元数据和完整分页文件清单，另核9个设备commit、发布tag及4个PyTorch回滚。大kernel/329文件迁移只重点看契约及选定路径，不能视为全部代码审计；未审候选记录在本地运行清单。72小时非默认分支全部merged事件尚未枚举，旧区间保留。

实际请求NVIDIA、AMD、PyTorch官方技术文入口与TPU、Gaudi、Neuron/NKI、NCCL notes，重点打开Spark8与GDS新文。Huawei公告仍为动态壳，CANN/HCCL/libtpu组件审阅不完整；RoCEnante合入说法只核到厂商文章，未定位具体PR。旧GEAK V3文实为7月内容，无本次新增证据，未作为新进展。检索未命中不能推导厂商没有发布。

### 补录与更正

72小时内首次纳入的旧提交逐条写原日期与“补录”，Perplexity报告、IQuest旧权重/新报告、Sante旧权重/新完整卡分开；已收录Youtu/Flint/DSpark等无实质新证据不重报。Open-SWE-Traces仅默认config调整，未新增parquet；M3.1的Dynamo配方同窗回滚，模型权重仍有效。FlashAttention最终descriptor方案、AITER非2幂KV guard、TorchTitan SDPA默认、SGLang传输stream/CP gather先后、PyTorch两项回滚等依最终代码注明；这是对旧结论的增量与适用范围更新，不改写旧日报。

### 未完成区间与下轮续查

全局“完整覆盖水位”仍为空。保留10-05→10-08六仓历史语义待审、10-05→10-12扩展仓缺口、10-15T01:55:47Z→10-16T01:00:40Z及10-13→10-18的arXiv修订/原API语义问题，以及各后续轮未审候选。当前arXiv精确增量API受限，高级检索、新列表和HF入口提供补充但不是全分类精确版本清单；OpenReview/ACL/PMLR与动态厂商入口也未全面完成。来源读取、事件枚举、语义核验和远端同步分账，不能从任何一项成功推导其余完成。

下轮从 **2026-10-23T01:00:15Z** 续查并回溯72小时，同时优先处理持久化待办与待上传文件。来源清单与本轮检查、稳定ID/版本、失败/未审候选、证据hash和上传回执保存在本地`.monitor/`。本次只维护llm-monitor归档及tracker的`llm-training-daily/`，不改写llm-knowledge wiki。
