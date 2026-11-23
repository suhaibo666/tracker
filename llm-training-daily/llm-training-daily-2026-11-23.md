# LLM 技术动态日报 · 2026-11-23

> 时区 Asia/Shanghai（UTC+8）。增量扫描窗口 **(2026-11-22 09:03:50, 2026-11-23 10:14:20]**；回溯72小时至 **2026-11-19 09:03:50**，并继承更早待查区间。实际生成 **2026-11-23T10:55:16+08:00**。论文提交、首次公开、模型工件上传、修订、合入与发现时间分别记载；本轮覆盖为选择性审读，未完成的来源不作扫描成功处理。

文中LLM为大语言模型，MoE为混合专家，KV为键值缓存，SFT为监督微调，RL为强化学习，GRPO为组相对策略优化，OPD为在线策略蒸馏；TP/PP/CP/DP/EP分别为张量/流水线/上下文/数据/专家并行。TTFT为首token时延，TPOT为单输出token时延，HBM为高带宽显存，DRAM为动态随机存储器，MLA为多头潜在注意力，RoPE为旋转位置编码，SDPA为缩放点积注意力，MTP为多token预测，CUDA/NPU graph为捕获后重放执行图；FP/BF后的数字表示浮点格式位宽。性能、硬件运行与质量成绩均标明来源，本轮未进行设备复训、压测或付费API实测。

## 本次重要变化

- **预训练进展需要固定验证集，也需要强基线。** OpenAI公开内部代码快照上的预训练损失读数，跨机构比较仍缺公开共同基准；Muon补录则说明拆开Q/K/V本身已有收益，谱修正须在更强基线上计量。[OpenAI原文](https://openai.com/index/perplexity-and-intelligence/) · [Muon论文](https://arxiv.org/html/2611.24644v1)
- **新权重的交付配置与质量证据应先于采用决策。** LingSante文本、视觉及MXFP4工件落地，但Flash卡片1M上下文与交付配置262144不一致，MXFP4图表仍属于源模型；SingVL换权重后的分数也缺对应新清单。[固定LingSante卡](https://huggingface.co/inclusionAI/LingSante-3.1-flash/blob/af553189673cc8cf33b0bbc1c7d5b097e23d116d/README.md) · [SingVL修订](https://huggingface.co/inclusionAI/SingVL_S1_4B/blob/c0846efa6a53bd21537f0f78e262dadcdabf39f7/README.md)
- **后训练与Agent推理有可复用设计，成本和统计边界同样具体。** AutoMaster改变固定教师的示范行为，4B两项收益区间跨零；Seer共享9B基座和两个适配器筛选候选动作，但无设备成本实测；e-RWSC保证的是生成分布中的多数答案，预算回退不获认证。[AutoMaster](https://arxiv.org/html/2611.26510v1) · [Seer](https://arxiv.org/html/2611.26423v1) · [e-RWSC](https://arxiv.org/html/2611.26413v1)
- **训练和推理主线继续减少通信、等待与冗余计算。** Megatron的上下文并行、重算预取和graph捕获，TorchTitan的MTP激活策略/XPU GRPO，以及SGLang的HiCache与DSA MTP均有具体合入；HiCache作者A/B吞吐提高伴随TPOT略退，尚有idle/drain生命周期边界待审。[HiCache PR](https://github.com/sgl-project/sglang/pull/50469) · [Megatron graph修复](https://github.com/NVIDIA/Megatron-LM/pull/9408)
- **正确性与负面结果优先进入采用清单。** ATOM旧restore状态会造成跳算或过早释放缓存；TPU主机页迁移需同时维护元数据。MLXcel真权重验收发现NVFP4类型和MTP精确性缺口，MTP未提速；Aurora极低bit量化缩小文件却明显损失能力，当前loader也没有运行内存/速度收益证据。[ATOM固定提交](https://github.com/ROCm/ATOM/commit/36adbfddb6379a038c2a6ac0ea2e3552607e507d) · [MLXcel实测修复](https://github.com/lablup/mlxcel/pull/3278)

[模型与技术报告](#topic-1) · [预训练与架构](#topic-2) · [后训练](#topic-3) · [推理与部署](#topic-4) · [开源工程](#topic-5) · [硬件软件协同](#topic-6) · [评估与可靠性](#topic-7) · [覆盖与续查](#coverage)

<a id="topic-1"></a>
## 模型与技术报告

### 蚂蚁 LingSante：医疗文本、视觉与MXFP4权重落地，交付边界比宣传上下文更需要先核对

**版本：11月22日上传权重，仓库11月19日创建；机构：inclusionAI。**本次与已收录的MedGuard评估工作分开：`LingSante-3.1-flash`固定`af553189673c`、`air-VL`固定`1f398c937d60`、`flash-MXFP4`固定`94db797206cf`均已有实际工件目录。Flash沿用Kimi Delta Attention（KDA）与门控多头潜在注意力（MLA）的7:1混合专家架构，63层KDA+9层MLA、512路由专家每token选8；训练叙述覆盖医疗/通用混合预训练、mid-training、监督微调、任务强化学习与多教师在线策略蒸馏。卡称主模型约560B/激活25B；HF张量计数572.12B属于另一统计口径，不能混成同一个精确参数数值。Air-VL卡称124B/激活5.5B、256K，面向图像报告、医疗搜索与GUI操作，后两者依赖外部工具和状态环境。[固定Flash卡](https://huggingface.co/inclusionAI/LingSante-3.1-flash/blob/af553189673cc8cf33b0bbc1c7d5b097e23d116d/README.md)；[固定Air-VL卡](https://huggingface.co/inclusionAI/LingSante-3.1-air-VL/blob/1f398c937d6069994a156689636832c47d8c09be/README.md)

**建议先定向复现，暂不据图表采用。**训练token数、数据清单、算力和完整消融未披露，主要评测结果以图像交付、复现条件不足。Flash卡写1M上下文，但所交付`config.json`为262,144且`rope_scaling=null`，示例没有明确升到1M；1M能力和实际运行配置尚未对齐。Flash目录为FP8主权重；MXFP4则是无校准数据、无再训练的最近舍入（RTN）转换，仅路由专家用E2M1/32元素共享尺度，其余保留FP8/BF16等。卡明确**所展示评测来自源模型，未测该量化工件**，配置上下文也只有262,144。推荐4×H20-3e、TP4/EP4、Humming 0.1.12；专家激活W4A8或W4A16须另配，`--dtype bfloat16`不会覆盖环境中的FP8激活设置。卡标MIT，本轮未找到根目录LICENSE；采用前仍需核许可文件及运行栈。未独立运行医疗评测或吞吐测试。[固定Flash配置](https://huggingface.co/inclusionAI/LingSante-3.1-flash/blob/af553189673cc8cf33b0bbc1c7d5b097e23d116d/config.json)；[MXFP4格式、部署与限制](https://huggingface.co/inclusionAI/LingSante-3.1-flash-MXFP4/blob/94db797206cf019959a8f0b464b99f4321c368bc/README.md)

### NVIDIA / MIT等 SANA-Robot：4.5B世界动作模型开放混合记忆权重，实时运行需要配套异步执行

**版本：11月22日开放预训练及RoboDojo、RoboTwin2.0、LIBERO三个后训练工件；建议深入读并考虑复现记忆消融。**SANA-Robot从多视角观测、本体状态和语言指令联合生成32步动作及未来视频；8个块各含3层门控DeltaNet（GDN）+1层softmax注意力。AR版以固定大小GDN状态压缩全部历史，softmax只保留首帧与有限窗口，避免随episode增长持续增加每次请求的历史读取量。官方称用19.4K小时、23数据集训练；发布的4.5B预训权重用于再微调，未含Gemma2-2B文本编码器和LTX-2.3 VAE。[官方项目页](https://nvlabs.github.io/Sana/Robot/)；[固定预训练卡](https://huggingface.co/Efficient-Large-Model/SANA-Robot-AR-Pretrain/blob/486387da56e8374a9602cfbee3670808d099401a/README.md)

官方实物Shell Game仅20次rollout，AR成功75%对仅看当前观测的base30%；展示了记忆价值，样本不足以证明普遍稳健性。RTX5090、窗口1、4步去噪、三视角条件，优化后NVFP4每chunk115ms、FP8为135ms；**30Hz是动作执行控制频率，不是每秒30次模型推理**。Memory RTC为不同执行相位保留多份记忆来匹配训练时chunk布局，有额外内存成本。LIBERO工件只在原LIBERO示范上后训练，LIBERO-PRO扩展步数下68.09%，回到原四套件步数限制是65.36%；两种预算不能混比。许可Apache2.0，完整优化栈、量化质量和其他机器人迁移需另验；本轮未运行机器人或设备测试。[固定LIBERO卡与预算说明](https://huggingface.co/Efficient-Large-Model/SANA-Robot-AR-LIBERO/blob/24905be1f5d2ecfe54e7d4a274b991732f39d897/README.md)

<a id="topic-2"></a>
## 预训练与架构

### OpenAI：公开一组长期追踪的预训练损失；跨机构困惑度比较仍缺共同基准

**发布日期：11月22日；机构：OpenAI，Jakub Pachocki。建议深入阅读。**《Perplexity and Intelligence》的技术增量是公开部分内部预训练评估读数，并提出把困惑度用于跨机构进展与风险讨论。文中实际报告的是**对数困惑度，即交叉熵损失**：GPT‑3.5约0.99、GPT‑4约0.91，近期模型低于0.8，并描述约0.76模型在数学上的新能力；评估数据是**内部代码库的固定留出快照**。这些是作者报告的关联，无法据此推导某个损失阈值必然触发某项能力。[官方原文与脚注2–5](https://openai.com/index/perplexity-and-intelligence/)

未公开该数据集、逐模型原始读数、tokenizer统一口径和可重做的因果实验；文中也承认数据配方差异、未知熵下限及思维链收益不反映在该损失上。对实践的意义是持续维护稳定、无污染的预训练验证集并联看下游能力；不能把0.76或作者展望的0.68–0.70当成可跨模型套用的“智能刻度”。关于2027年预训练算法加速属于作者预测。

### Muon 曲率诊断：先拆开 Q/K/V，再判断谱修正的额外收益（2611.24644v1，补录）

**机构／日期／版本**：JPMorgan Chase LLM Suite；2026-11-18 提交，v1；本轮补录，未确认首次公开秒级时间。

Muon 对动量各奇异方向近似分配相同步长，但曲率并不相同。论文用留出梯度和 Hessian–vector product（海森矩阵与向量乘积）检验同更新范数下的步长重分配，并把“分块收益”拆成更强基线与额外修正：在 FineWeb-Edu、124M、100.7M 训练 token、序列 512、三配对种子下，单独正交化 Q/K/V 已使损失比 fused-QKV Muon 低 0.0250 nats；cooldown 中的谱倾斜在该基线上再低 0.0120，与 NorMuon 合用低 0.0158。**限制**：仅 124M／355M、欠训练区间；局部二阶诊断常数未完全实测，QKV 解释仍属假设，部分旧基线跨代码版本。建议深入阅读并考虑由开发会话复现拆 QKV 的对照；不能把小规模损失收益换算成大模型训练加速。[正文 §3–6、§9、Appendix E](https://arxiv.org/html/2611.24644v1)

<a id="topic-3"></a>
## 后训练

### AutoMaster：固定教师也能通过行为引导产生更好的 SFT 轨迹（2611.26510v1，补录）

**机构／日期／版本**：中国人民大学；2026-11-19 提交，论文落款 11-20，v1；本轮补录。

只筛成功轨迹会保留盲目假设、重复操作和浅验证。AutoMaster 用代理设计自然语言技能，先改变教师 rollout 行为，再筛轨迹做监督微调（SFT）；技能保留在训练输入中，评测时去掉。固定 DeepSeek-V4-Pro-Preview 教师、任务池和每次 512 轨迹／三 epoch 配方，在 Qwen3.5-4B／9B 的 Terminal-Bench 2.1、BrowseComp Lite-185、Doc2Repo 上，较普通成功轨迹 SFT 的均值增加 1.42–8.99 个百分点；9B Terminal 为 26.59%→35.58%。**限制**：代理可进行五轮外循环，成本并未与普通 SFT 完整匹配；三次是评测重复，并非训练种子，4B 的 Search／Doc2Repo 配对置信区间跨零。建议复现技能保留／移除及等生成预算消融，尤其适合已有教师蒸馏流程。[正文 §3–5、Table 1、Appendix A.6](https://arxiv.org/html/2611.26510v1)

### RL 的稀有行为学习：成功的 rollout 组数比总采样数更能解释训练是否起效（2611.26084v1，补录）

**机构／日期／版本**：University of Deusto；2026-11-19 提交，v1；本轮补录。

在先验和最优动作均可精确计算的合成工具游戏中，论文把训练是否学会“默认动作失败后换工具”，关联到基座策略预计产生的**至少有一次成功切换的 rollout 组数**。固定总采样预算时，扩大每组样本会减少独立的有效更新组，因此未必更容易学会稀有技能；特定配方约 35 个预期成功组对应一半运行起效。该计数在 1.3M–85M、组大小 4–256 和多种 GRPO（组相对策略优化）配置中比只数问题／尝试更有效。**限制**：阈值依赖任务与配方；OLMo-3-7B 仅做单步测量，未能可靠完成游戏，因而没有大模型 RL 验证。论文也观察到技能向未尝试的具体动作泛化。建议阅读并将有效成功组覆盖作为实验诊断，不把 35 当通用门槛或据此否认 RL 泛化。[正文 §3–6、Appendix A–C](https://arxiv.org/html/2611.26084v1)

### Seer：共享 9B 基座的预测与评分，仍需核算额外推理成本（2611.26423v1，补录）

**机构／日期／版本**：四川大学、UIUC、联想；2026-11-19 提交，v1；本轮补录。

Seer 将动作提出、工具响应模拟和状态价值评分分开：由真实研究轨迹离线训练预测与评分两种适配器，**共享同一 Qwen3.5-9B 基座权重，分别使用 LoRA（低秩适配）**。对候选动作做一步想象，仅执行选中的动作，正式历史只写入真实工具响应。候选宽度 32 时，Qwen3.5-4B 在 GAIA 从 13.6% 升至 41.7%，9B 从 22.3% 升至 50.5%；宽度 4 的 4B 为 29.1%，表明增益依赖额外推理预算。**限制**：模拟器消除未选动作的真实调用，也在部分消融中损失准确率；更宽并非总更好。论文未实测设备时延、峰值显存或能耗，共享基座不消除逐候选预测／评分的计算；任务为可判正确性的问答，未验证长报告质量。建议深入阅读其“预测内容不得成为引用证据”的状态边界，部署前由开发会话核算端到端成本。[正文 §4–5、Tables 2–5、Appendix H](https://arxiv.org/html/2611.26423v1)

<a id="topic-4"></a>
## 推理与部署

### H2O-Lightning v1.3.5：用隐状态预测“思考是否值得”，收益伴随长尾排队

**机构：H2O.ai；本轮11月23日核见tag v1.3.5=`fc5ee73d7c67`，相较旧v1.3.0新增自适应推理；首发日期因gated历史接口401未核实，按版本补录。**Gemma4-31B衍生模型读取两层末token残差，由小头预测思考收益，符合条件后在同权重/同引擎追加最多2,560 thinking tokens。只覆盖不含图像、单选择题且prompt≤8,192；基准`decision_index`/`jevbench`模式默认禁用，因此不能拿旧榜单说明新增功能收益。[固定卡“Adaptive reasoning”](https://huggingface.co/h2oai/h2o-lightning-31b/blob/fc5ee73d7c673141471f29416600758055e1ec15/README.md)

作者在单RTX PRO6000 96GB、313道未参与路由头拟合的HLE文本选择题上报告+10.86个百分点（配对bootstrap 95%区间+6.39至+15.02）；另200题见过训练，单独列示。231题JevBench在`aa`串行模式218→224正确，但平均时延126.8ms→4,468ms；16并发时升级请求p95达289.3秒，最多4个思考槽，默认排队而不降级。一般决策控制集的独立半折收益区间跨零。**考虑复现路由/长尾代价，继续观察采用。**需要H2O自有vLLM fork，权重受非商业、非生产许可约束；托管服务只写“下次更新启用”，不能称已上线。本轮gated历史/配置访问失败，未复跑。

### Aurora2发布QAT与SFT：极低bit MoE给出清晰负面结果，27.5MB是存储指标

**机构：AuroraAI-Research；11月22日新工件，QAT固定`5d93093f2a7b`、SFT固定`80cffeaca6cf`；建议作为小规模复现实验，继续观察工程采用。**399.6M/激活19.1M、32路由专家top-1的2K上下文小模型，QAT用LittleBit-2二值低秩因子及teacher logits/隐状态/路由分布蒸馏；FineWeb-Edu约2.6B tokens。路由专家约0.48 bit/weight，含其它INT4及尺度后**全模型约0.551 bit/weight**，文件799MB FP16→27.5MB。相同零样本设置下作者自定义、机会水平归一化的Open-SLM Intelligence Index为21.38±0.78→10.83±0.80，HellaSwag38.36→28.17，说明广泛能力明显损失；该指标不是Artificial Analysis同名指数。[固定QAT卡](https://huggingface.co/AuroraAI-Research/Aurora2-400M-A20M-QAT/blob/5d93093f2a7bb849ea566d38eaf2e491144bfdf4/README.md)；[位宽与训练配置](https://huggingface.co/AuroraAI-Research/Aurora2-400M-A20M-QAT/blob/5d93093f2a7bb849ea566d38eaf2e491144bfdf4/littlebit_config.json)

本轮读实现确认`from_pretrained→_unpack_and_load`展开二值因子为FP32，`LittleBitStack.all_weights`每次拥有者forward重建全专家权重、仅当次缓存；因此当前代码**没有给出内存或速度收益**。作者也明确需要专用二值矩阵乘法核。另SFT工件采用5B tokens，约半对话、半通用/数学重放，assistant-only损失、Muon+AdamW；对话验证损失1.505→1.254，网页2.566→2.612，是留出损失证据，缺完整交互能力复评。两工件卡标Apache2.0，尚无独立复现；不把存储压缩比当部署加速。[固定loader代码](https://huggingface.co/AuroraAI-Research/Aurora2-400M-A20M-QAT/blob/5d93093f2a7bb849ea566d38eaf2e491144bfdf4/modeling_aurora2_littlebit.py)；[固定SFT卡](https://huggingface.co/AuroraAI-Research/Aurora2-400M-A20M-SFT/blob/80cffeaca6cf10e26ddd252bc14ddbae67c6aa20/README.md)

### e-RWSC：自适应停采样可以证明“答案占多数”，证明对象仍需明确（2611.26413v1，补录）

**机构／日期／版本**：IIT Kanpur；2026-11-19 提交，v1；本轮补录。

直接把随采样变化的领先答案代入经典顺序检验，会破坏错误率保证。e-RWSC 为每个新答案维护可选停止仍有效的 e-process，奖励模型仅影响下注速度；它在独立同分布采样假设下控制“任何时刻错误认证非多数答案”的概率。MATH500、Qwen2.5-7B、ArmoRM、最多 24 候选、α=0.1 时，准确率 81.4%、平均 13.06 候选，对 Best-of-24 的 77.8%、24 候选。**边界**：保证的是生成分布中占比大于一半，非答案正确；预算用尽后的普通奖励加权自一致性回退没有该认证。Appendix D 的 RTX 6000 Ada 计时另用最多 512 候选，不能与上述预算直接拼成加速结论。建议复现明确答案可归一化的数学／代码任务，同时报告认证率、回退比例和真实时延。[正文 §2–4、Theorem 1、Table 1、Appendix D](https://arxiv.org/html/2611.26413v1)

<a id="topic-5"></a>
## 开源工程

以下为默认分支已合入增量；数字均来自相应作者的实验或回归记录，本轮未自行复训或设备压测。张量并行（TP）、上下文并行（CP）、专家并行（EP）、流水线并行（PP）、数据并行（DP）、多 token 预测（MTP）在下文沿用这些缩写。

### Megatron：混合模型可统一连续 CP，代价是因果注意力负载不均

[NVIDIA/Megatron-LM #7834](https://github.com/NVIDIA/Megatron-LM/pull/7834) · 11-22 11:34（上海）· [`6dfb6533ac23`](https://github.com/NVIDIA/Megatron-LM/commit/6dfb6533ac236bba71e838d1b6831283209cec84)。

混合线性/softmax 注意力原先分别使用连续与 zigzag token 归属，层间需要补齐及重分布。现在 `attention_cp_layout="contiguous"` 经 Transformer Engine 的 `NO_LOAD_BALANCE` all-gather 路径支持 packed THD，并适配 dense SBHD；MTP 与隐藏状态滚动也按连续归属处理。这里换掉的是布局转换，不是免费得到更均衡的注意力计算。

新增八卡前后向、MTP roll 与实际 padding 元数据测试；PR 未给生产吞吐对照。**建议深入阅读并考虑复现**：同时含线性注意力与 softmax 的模型应测层间搬运下降能否抵销 causal attention 的 rank 不均；还需满足所用 TE 路径条件。

### Megatron MFSDP v2：把激活重算纳入记录式预取

[NVIDIA/Megatron-LM #9549](https://github.com/NVIDIA/Megatron-LM/pull/9549) · 11-22 11:28（上海）· [`6ca2dc52ed02`](https://github.com/NVIDIA/Megatron-LM/commit/6ca2dc52ed02da77183df7238d0177b046905f42)。

MXFP8 反向预取只取 columnwise planes，而重算又需要 rowwise planes，过去在反向中临时 all-gather，无法重叠。`PrefetchOrder.RECORDED` 现在将重算记录为 FORWARD 事件，由前一事件预取它；校验使用事件阶段，避免“模块已处于 backward”误判。

reentrant 与 non-reentrant checkpoint 新测试确认重算开始前 rowwise 已就绪，并与不重算保持逐位一致。`REGISTRATION` 顺序仍按需聚合，PR 未给端到端提速。**建议考虑复现**：适合 MFSDP v2、MXFP8 与重算一起使用时，核查通信是否移出关键路径。

### Megatron：GTP 分片权重的局部 CUDA Graph 反向捕获恢复

[NVIDIA/Megatron-LM #9408](https://github.com/NVIDIA/Megatron-LM/pull/9408) · 11-22 09:22（上海）· [`1928d46c6faf`](https://github.com/NVIDIA/Megatron-LM/commit/1928d46c6faf2fb69dfbb8e91d242cd946f345a1)。

GTP 权重的梯度直接归入 `main_grad`，autograd 叶子收到 None，旧 `AccumulateGrad` 的 stream 不会像普通权重那样被改写，首次 backward capture 因等待非捕获事件而报 isolation 错。**最终 diff** 让局部图 warmup/capture 与 DDP 初始化共用 `get_shared_capture_stream()`，并在结束后正确回到默认 stream；PR 正文仍描述较早的 fork/join 方案，机制解释以最终 diff 为准。

作者在 torch 2.13 四卡回归连续五次通过；16 卡 GTP8/EP2、18 层 hybrid+MTP、40 步由首捕获崩溃恢复训练，初始 loss/grad norm 与 eager 相同。三个稳态抽样为 1276–1308ms，对照 1320–1327ms，不能当长期均值；测试仍记录全 H100 bucket 首次 replay 偶发漂移根因未定。**建议优先阅读**，已有 GTP+local graph 部署应回归训练一致性。

### Megatron：推理 Triton 核启用 PDL，训练保持原路径

[NVIDIA/Megatron-LM #9406](https://github.com/NVIDIA/Megatron-LM/pull/9406) · 11-22 13:48（上海）· [`e54d3896a1c5`](https://github.com/NVIDIA/Megatron-LM/commit/e54d3896a1c5ce14b08d0a493a078c77c01a182c)。

11 个推理核增加 programmatic dependent launch（PDL，程序化依赖发射）的 wait-only 路径：只在 inference_mode、NVIDIA compute capability≥9.0、Triton/launcher 支持时启用；每个 block 先等待依赖，未主动提前释放 dependent。MXFP8 scale padding 同时合入 quantize 核，减少一次发射；`MCORE_INFERENCE_TRITON_PDL=0` 可关闭。

作者 Rubin 四卡 MXFP8 hybrid MoE、744 请求、输出1024、三次计时报告 decode 时间17.444→17.261ms（约1%）；未披露完整模型和输入长度。11核逐位回归与故意移除 wait 的负对照支持排序正确性。**建议继续观察或按部署复现**，不要据此推断所有 SM90 核都能同幅获益。

### DeepSpeed AutoTP：未绑定输入 embedding 可按词表分片

[deepspeedai/DeepSpeed #8859](https://github.com/deepspeedai/DeepSpeed/pull/8859) · 11-22 12:55（上海）· [`6c8e3f1f70e5`](https://github.com/deepspeedai/DeepSpeed/commit/6c8e3f1f70e54b307bceab9fc9124d2ed61016fd)。

`tensor_parallel.vocab_parallel_embedding=true`（默认关）把未与输出头绑定的输入 embedding 改为均匀词表行分片，覆盖 HF tp_plan、显式/预设及启发式路径，并处理 ZeRO Init、重复模块注册、checkpoint 元数据。已有 tied shard 保留；与复制输出头绑定、重复切分、DeepCompile autotp 等冲突显式拒绝。

作者 Qwen2.5-72B、4×8 H200/EFA、TP8×ZeRO1 DP4、BF16+FP32 master/Adam、每步64×8192 token、无重算：rank0 峰值114.1→107.0GiB，步时15.88→15.77s；实验还叠有独立的 sequence-sharding 改动，旧版与终版在这组配置的切分相同。新测试含ZeRO0–3和保存恢复，但offload、多种低精度组合与通用导出仍未端到端验证。**建议考虑采用**：主要价值是词表及优化器状态显存，吞吐变化较小。

### DeepSpeed AutoEP：跨微批暂存专家权重梯度接上 ZeRO-3

[deepspeedai/DeepSpeed #9115](https://github.com/deepspeedai/DeepSpeed/pull/9115) · 11-22 11:40（上海）· [`c00a7f68c0cd`](https://github.com/deepspeedai/DeepSpeed/commit/c00a7f68c0cdb20ed37f356a38af6b440cf70f5d)。

`deferred_dw_flush_period` 现在可与 ZeRO3 联用：无梯度的 staged backward hook 提前返回，依赖已合入#8981保证一轮内首次梯度覆盖旧分区；ZeRO3 gathered weight 若未16字节对齐，grouped_mm路径才复制。重新聚合权重用于反向，避免 detached alias 长期持有整块专家权重。

H200双卡、EP1/2、累积4、flush3、两次optimizer step验证staged/unstaged梯度；ZeRO0–3均复核。单GPU模拟释放96MiB权重，separate gate/up暂存由98→2MiB；fused gate/up仍保留66MiB打包副本，重打包会使特定8 rows/expert模块前后向慢30%。**建议深入阅读并选择性复现**：这是兼容与生命周期修复，未给端到端吞吐/峰值/收敛；多节点、DeepEP、offload/NVMe未验证，对齐复制成本未测。

### TorchTitan：激活重算策略覆盖 MTP 与多模态编码器

[pytorch/torchtitan #6246](https://github.com/pytorch/torchtitan/pull/6246) · 11-22 23:05（上海）· [`69e31a2d0e92`](https://github.com/pytorch/torchtitan/commit/69e31a2d0e921c393a3d7c7f71343e17e050f063)。

以前重算策略只走 `model.layers`，DeepSeek式 `mtp_layers` 保留全部激活，容易在大模型上意外OOM。现在一次枚举编码器、主干和MTP block containers，共享一个 ActivationOffloader；编码器region名称含完整路径，避免和decoder撞名，并在MTP裸算子前声明重算所需tensor。

四GH200、FSDP4、seed42、10步的DSV3 debug MTP（Default及16K Full）loss/grad norm与基线逐位相同；DSV4仅局部重算对照，完整MTP训练另有已知问题。畸形block override名称现在可能静默匹配不到。**建议优先检查现有MTP训练配置**，但PR没有给可泛化的峰值节省比例。

### TorchTitan：Intel XPU 使用同一 GRPO 训练与生成路径

[pytorch/torchtitan #5402](https://github.com/pytorch/torchtitan/pull/5402) · 11-22 14:49（上海）· [`a75b09a7d5b5`](https://github.com/pytorch/torchtitan/commit/a75b09a7d5b530f317c35771745ce1f93f6a856b)。

RL进程按设备选择可见卡变量；XPU保留TorchTitan runner，并让训练和生成请求相同flex attention，避免静默换路径。CUDA graphs、varlen、压缩MLA cache、GDN/KDA、batch-invariant模式在不支持的设备上明确报错。

Aurora一节点、torch2.15+xpu、vLLM带#69044变更，Qwen3-0.6B完成5步GRPO；较早1–16节点仅能支持设备可见性/accelerator部分，不能等同当前全功能扩展测试。还要求allocator expandable_segments=False。**建议继续观察并考虑小模型复现**：可作为异构RL接入范例，当前证据不支持大模型性能或收敛结论。

### verl：每步同步时去掉从不读取的整模 CPU 快照

[verl-project/verl #8659](https://github.com/verl-project/verl/pull/8659) · 11-23 09:59（上海）· [`499cbe6d3a74`](https://github.com/verl-project/verl/commit/499cbe6d3a741de605c6aaf495c4bd13c1ac1fff)。

separate-async PPO的 `parameter_sync_step=1` 时 local_trigger_step始终为0，旧策略快照slot0从未restore，但每global step仍执行整模device-to-host拷贝。现在调用方仅在同步间隔>1时保存slot0，而不是让后端猜测快照是否还会用。

CPU回归覆盖间隔1/2的save/restore/clear顺序；默认间隔4及需要旧策略恢复的语义保留。没有GPU/NPU/TPU端到端速度测量，不能从整模拷贝被删除推导固定提速。**建议符合条件的异步RL配置考虑采用**，并观察host RAM与D2H流量。

### vLLM：DFlash/DFlash2 正式接受 anchor-first 草稿布局

[vllm-project/vllm #68827](https://github.com/vllm-project/vllm/pull/68827) · 11-23 00:07（上海）· [`e2052f5c7ad6`](https://github.com/vllm-project/vllm/commit/e2052f5c7ad64564f00dd4592e77c27785c9333f)。

此前 `sample_from_anchor=true` checkpoint在启动被拒绝，尽管输入核已支持。现在config以统一 `num_query_tokens` 驱动K（anchor-first）或1+K（fill-in）的query分配、KV lookahead、warmup和DFlash2 grouped-conv；checkpoint顶层与子配置冲突会报错，默认fill-in保留。

作者B200，Qwen3.8-27B+DaoCloud anchor DFlash2，GSM8K 0.920、accept length6.03；fill-in128提示输出完全一致。私有Omni ASR草稿的同驱动对照4090/4096字节一致，控制重复4094/4096，统计量不足以证明完全等价；驱动变更本身明显影响输出。**建议深入阅读并复现公开权重**，这首先解除模型兼容门禁，未证明布局自身普遍更快。

### vLLM KV offload：超容量分批存储、ARC复用前缀与读写分工

[vllm-project/vllm #66719](https://github.com/vllm-project/vllm/pull/66719) · 11-22 14:20（上海）· [`e72fc9ae50c2`](https://github.com/vllm-project/vllm/commit/e72fc9ae50c221180c4b8f4a94c5009714a4b199)；[vllm-project/vllm #69110](https://github.com/vllm-project/vllm/pull/69110) · 11-22 18:17（上海）· [`e29c741da8d1`](https://github.com/vllm-project/vllm/commit/e29c741da8d11e2ae7ba381bb78b6f703812cae1)；[vllm-project/vllm #67961](https://github.com/vllm-project/vllm/pull/67961) · 11-22 16:43（上海）· [`9ae9c15c5691`](https://github.com/vllm-project/vllm/commit/9ae9c15c569176e52783dd2ef8f4fedf8aadc70a)。

#66719修复CPU tier小于单请求时整批prepare_store反复失败：manager只接受容得下的较早位置chunk，以keys_to_retry把尾部留给后续step；各cache group token范围对齐，结束/hand-off后无法重试仍计丢失。无次级tier时可能循环驱逐自己的旧前缀，容量不足没有消失。#69110把复用T1前缀加入ARC运行中请求的tail-first驱逐次序，防止留下失去前缀的后缀。

ARC作者GB10、SmolLM2-135M、CPU240MiB/GPU2GiB、四个3073-token交错请求：三轮replay命中0→1664token、单token replay中位268.3→133.9ms，generation总时2.856→2.868s；tokens/logprobs一致。这是人为制造缓存压力的回归负载。#67961新增主/次级 `access_mode=read|write|readwrite`，用于CPU→文件写和GDS直读GPU分工，未附性能证明。**建议缓存密集服务优先阅读和压力回归**，特别检查完成请求剩余chunk与shared-key owner。

### SGLang HiCache：写回驱逐移出 scheduler 主机等待，初始化预算消除竞态

[sgl-project/sglang #50469](https://github.com/sgl-project/sglang/pull/50469) · 11-23 00:46（上海）· [`de41ccfae097`](https://github.com/sgl-project/sglang/commit/de41ccfae097c629a3928150b1f7139e33dfd183)；[sgl-project/sglang #52120](https://github.com/sgl-project/sglang/pull/52120) · 11-22 09:53（上海）· [`7e62e8361211`](https://github.com/sgl-project/sglang/commit/7e62e8361211a3e04deb7fa6bbb31d9a58778701)。

write_back原先D2H复制排在当前forward后，scheduler再synchronize复制，阻塞下一批准备。#50469将保护改成“复制等旧forward、下一forward/load等复制”的设备event依赖，允许host立即demote；overlap时schedule stream不额外等待。L3 storage、buffer pipeline、linker、统一内存及PD decode继续阻塞，避免其他路径提前读写释放槽。

DSv4.1-Flash、GB300、14个prefill server+DEP16 decode、agent多轮回放1200s×每臂2次（两臂均含#47650）：作者报告端到端958.8k→998.9k tok/s/GPU（+4.2%），TTFT p50 4.26→2.73s，TPOT p50却6.75→6.92ms。#52120在显式hicache-size/ratio时先取每rank主机预算快照再barrier，避免兄弟rank并发pin导致反复缩小预算；GB300双节点四rank/节点复现启动失败并修复，gfx950显式尺寸未设备验证。独立复核还发现 `_fenced_write_acks` 未纳入 `transfers_in_flight`，blocking `writing_check` 对该旧队列只query；reviewer也提醒flush/storage-attach的idle/drain边界，尚未证明这些生命周期全面安全。**建议考虑复现**，先匹配stream与缓存拓扑，补齐drain边界检查，不能只看吞吐。

### SGLang：单层 DSA MTP prefill 跳过非末 token 的注意力与 MoE

[sgl-project/sglang #46784](https://github.com/sgl-project/sglang/pull/46784) · 11-22 21:11（上海）· [`b2ce7f8811bc`](https://github.com/sgl-project/sglang/commit/b2ce7f8811bc40fd0d67000030c22edbbe0f2a37)。

DSA NextN草稿只有一层，非末prompt token只需写该层KV与indexer key，attention/MoE输出不会被后层消费。新路径把prefill拆为KV-only全体非末token及末token完整forward。适用CUDA上的GLM5系/DeepSeekV3.2；其他draft不变，attention DP、prefill/decode CP、若干FP8 fresh/cache数值路径和其他未验证组合被排除。

H200 GLM-5.3-Flash、MTP5/1/6，8K/32K部分并发负载input throughput +2.8–3.7%；另有贡献者在8B300/8H20复核KV字节和结果，B300某些4–8K单请求TTFT减3.5–5.3%。拆批新增约3.5ms CPU，仅较大prefill启用；B300无target prefill graph覆盖的行可慢约2%，补graph后才转正。GSM8K/RULER2差异在作者统计误差内，不是所有hidden state逐位相等。**建议深入阅读并按CPU/GPU比例复现**。

### SGLang GLM：LayerNorm序列分片节省 mHC，短prefill会亏

[sgl-project/sglang #50906](https://github.com/sgl-project/sglang/pull/50906) · 11-22 22:28（上海）· [`cb337923d576`](https://github.com/sgl-project/sglang/commit/cb337923d576f48a4accba439af9cfbc9b720a2c)。

`--enable-layernorm-sp`把GLM-5.3-Flash残差/mHC留在1/TP行，attention和FFN再gather；根据输出是否还欠TP求和选择reduce-scatter或slice。mHC每token独立，因此可少做重复行；代价是每层collective由2次增为4次。门禁拒绝attention CP、MoE a2a、attn-tp-input-scattered，当前GLM MTP也不可与之组合。

8H200 TP8代理128K冷prefill，mHC80.96→11.43ms，但collective总成本多15.4ms。真实权重同栈4–128K单请求TTFT下降5.8–9.1%，1K单请求反慢4.9%；可用已有min-tokens=2048保留小批复制。**建议长prefill考虑复现**，不能将代理mHC缩短86%当成全模型速度提升。

### SGLang：DFlash 捕获hidden states按真实行分片恢复全量

[sgl-project/sglang #47047](https://github.com/sgl-project/sglang/pull/47047) · 11-22 12:59（上海）· [`448764da636a`](https://github.com/sgl-project/sglang/commit/448764da636a8500015f461afbb91e3d2d23c94f)。

GLM draft辅助hidden states先前只对枚举的几种MoE backend all-gather，megamoe等路径拿到1/TP行后索引越界。新gate使用与layer boundary相同的 `is_moe_input_scattered_across_dp_ranks()`，并检查当前batch实际layout，既补漏也避免整批被重复聚合。

8H200 TP8/EP8、GLM-5.3-Flash+megamoe+incoai DFlash2，原先首请求崩溃，现在可服务。8.2K输入/1024输出时，作者TPOT由无draft基线8.92→3.40ms(c1)、30.06→23.13ms(c128)，收益包含从不能开到可开投机解码，不能称此all-gather自身提速2.6倍。GSM8K96.89–97.27%对97.42%，处于重复服务波动范围；其他a2a仅机制覆盖未全实测。**建议相关配置优先检查正确性**。

### SGLang deterministic decode：固定block预算循环split，避免大量空CTA

[sgl-project/sglang #47410](https://github.com/sgl-project/sglang/pull/47410) · 11-23 08:39（上海）· [`e2d88a64f05d`](https://github.com/sgl-project/sglang/commit/e2d88a64f05dec87a23980a52215eb1491b44421)。

batch-invariant模式按最大context分配split预算，短请求过去照样为大量空split发block。现在grouped decode每request/head tile发足以覆盖设备且至少32个block，block循环处理该请求真实splits；仅deterministic merge提前停在真实split数。普通≤32预算不变，MHA非grouped核不变；MLA merge可差几个ULP但保留batch invariant。

MI350X/ROCm7.2.4，同一套额外deterministic改动同时叠在两臂，gpt-oss-20b TP1、1K输入/1K输出、c256吞吐5024→12147 tok/s，c64为3983→7504；64K少量长请求接近不变。优化后该模型仍只有普通模式约0.50倍(c256)，不是追平normal。**建议确定性服务考虑复现**，先记录context上限与split预算以解释收益。

### SGLang：Mamba verify状态stride与大LoRA行偏移的静默正确性修复

[sgl-project/sglang #51164](https://github.com/sgl-project/sglang/pull/51164) · 11-22 16:14（上海）· [`c0bdbb306b2c`](https://github.com/sgl-project/sglang/commit/c0bdbb306b2c60557e2516b0d419eb293544a494)；[sgl-project/sglang #51199](https://github.com/sgl-project/sglang/pull/51199) · 11-22 16:15（上海）· [`d40fb11e7139`](https://github.com/sgl-project/sglang/commit/d40fb11e71396e11440d695b2b72355099edb67f)。

Mamba2自适应验证的buffer容量可大于当前active steps，原核用active steps推导batch stride，第二请求起写入前请求空槽且漏写accept状态；#51164改传真实stride(0)，同时用于存储和tree parent读取。A10上容量4控制及容量8回归、chain/tree、图replay通过；独立mixer检查旧输出可相同但1536/3072 live state错误，说明只比当前logit会漏检。

#51199把csgmv shrink/expand读出的physical row转int64再算地址，修复offset超过2^31元素的溢出；新增65538×32768 FP16张量末行测试覆盖普通/split-K shrink与expand。**建议混合模型投机及大批LoRA服务优先回归**：两项都是边界正确性，没有吞吐提升主张。

### FlashInfer：SM120 原生NVFP4稀疏MLA prefill融合落地

[flashinfer-ai/flashinfer #9356](https://github.com/flashinfer-ai/flashinfer/pull/9356) · 11-23 10:01（上海）· [`637f9c530909`](https://github.com/flashinfer-ai/flashinfer/commit/637f9c5309093704b9500495068627761f6ebc97)。

sparse MLA mega API新增native NVFP4 cache prefill，CuTe C++/CUTLASS barrier组织IO warp gather与V转换、math warp QK/online softmax/PV；非RoPE部分block-scaled FP4，RoPE尾保留BF16。该mega路径要求prefill plan，decode plan显式拒绝并由独立#9439负责，普通NVFP4 decode不受此提交扩展。

RTX PRO6000 Blackwell Server、SM120/CUDA13.4，warm-L2图replay，B1/Q128/H128/D512/KV8192/page64：选128/512 keys时attention单独1.170/1.207倍；含RoPE与输出量化完整序列65.49→43.18μs、163.49→124.99μs。1451定向测试通过，GB10/CUDA13.0验证因集群prolog失败未跑。**建议SM120稀疏prefill深入阅读/复现**，这是核级microbenchmark。

### FlashInfer Cake：BF16长KV扩split，FP8 reducer扩CTA，但未普遍胜过其他backend

[flashinfer-ai/flashinfer #9857](https://github.com/flashinfer-ai/flashinfer/pull/9857) · 11-23 01:33（上海）· [`1b4d210d62d0`](https://github.com/flashinfer-ai/flashinfer/commit/1b4d210d62d00f4cac7b57c99e41ca2a98ec2217)；[flashinfer-ai/flashinfer #9858](https://github.com/flashinfer-ai/flashinfer/pull/9858) · 11-23 09:29（上海）· [`6046d47f2907`](https://github.com/flashinfer-ai/flashinfer/commit/6046d47f2907e7abb89c7061e63c2f7766c00446)。

#9857让B200单query row从64K、GB300从128K起的BF16 KV可用最多64个split（原32）；#9858则在FP8 page64、1–2个query row且≥8split时，把每row归约从最多16扩到最多128 CTA，split数不变。auto仍按tuned shape选backend。

CUDA13.5 JIT/torch2.14.1，冷L2、图replay、6轮medians：BF16改动形状相对旧Cake几何均值1.314/1.328倍(B200/GB300)，对最快trtllm-gen或cute-dsl却仅1.033/0.999倍；FP8为旧Cake1.148/1.145倍，对最快其他backend0.924/0.890倍。**建议读选择边界并按形状复现**，单一backend进步不等于全局最快，尤其不宜强制Cake跑所有FP8请求。

### FlashInfer：deferred MoE FP32累加与autotune图pool元数据膨胀

[flashinfer-ai/flashinfer #9749](https://github.com/flashinfer-ai/flashinfer/pull/9749) · 11-23 09:52（上海）· [`83a5b03c44b8`](https://github.com/flashinfer-ai/flashinfer/commit/83a5b03c44b8310e0f7f029235a33bf8403d8624)；[flashinfer-ai/flashinfer #9554](https://github.com/flashinfer-ai/flashinfer/pull/9554) · 11-23 04:59（上海）· [`3adb5ac7ca74`](https://github.com/flashinfer-ai/flashinfer/commit/3adb5ac7ca7423a3883eff4609933fb9cf3f6cd3)。

#9749将一个deferred MoE finalize+all-reduce Lamport核内top-k专家和shared expert的加权求和改为FP32累加、最终store只舍入一次；另一个 sibling `moereduction...`核仍逐项低精度舍入，不能称全部MoE路径已修复。

#9554解决每次profiling capture各建private pool后，PyTorch memory_stats保留成千pool条目、监控取数持GIL卡scheduler的问题，四处capture共享每device pool。作者B300、FP4 MoE E16/H=I3072/T4096/topk4：pool新增2800→1、memory_stats21.9→0.06ms，reserved峰值1.23GiB/分配次数/调优20.7s均不变。不是显存容量降低；共享图pool不得并发capture/replay。**建议使用重调优并监控CUDA内存的服务优先检查**。

### vLLM-Omni：Ulysses合并QKV交换，通信调用4次变2次

[vllm-project/vllm-omni #10134](https://github.com/vllm-project/vllm-omni/pull/10134) · 11-22 21:15（上海）· [`711a720c9a34`](https://github.com/vllm-project/vllm-omni/commit/711a720c9a347f4ba73f7b350c8122b99090aa4f)。

`VLLM_OMNI_PACKED_QKV=1`可把eligible strict-layout的Q/K/V打包一次forward exchange，加上output exchange，每attention调用由4次collective降到2次；逻辑payload和精度不变，平台无关、默认关。显式ulysses-a2a-permute优先，其他layout保留旧路径。

当前修订的真实2/4rank Gloo与CPU配置测试108通过；CPU上collection的AMD NIGHTLY数量不算MI325执行，当前头未重跑CUDA/ROCm/NPU服务及图像质量。历史MI350X数字不能归到本修订。**建议继续观察后做本机复现**，小collective次数下降是可核代码事实，收益取决于pack成本与后端。

### vLLM-Omni：语音残差采样隔离请求，启动副本故障限制在stage内

[vllm-project/vllm-omni #10966](https://github.com/vllm-project/vllm-omni/pull/10966) · 11-23 05:28（上海）· [`a4f1f6f80d34`](https://github.com/vllm-project/vllm-omni/commit/a4f1f6f80d34a7f03c46957debb7402a74b57246)；[vllm-project/vllm-omni #10881](https://github.com/vllm-project/vllm-omni/pull/10881) · 11-23 02:35（上海）· [`acda6ed6161b`](https://github.com/vllm-project/vllm-omni/commit/acda6ed6161b76b133c67f042d29d20c4ef924e0)。

Qwen3-Omni codebook0已用request seed，但residual codebook消费进程全局RNG，batch顺序可使长音频跑偏。#10966提供持续request-local generator，显式tts_local_seed优先，否则从Talker seed作域分离；unseeded保留旧行为。CPU回归覆盖顺序独立/流状态；seeded row绕过外层MTP graph，内部code predictor图仍可用，当前头设备质量验证不能由CPU测试替代。

#10881把READY前单个EngineCore退出隔离：stage仍有可用replica即可继续启动并保留原ID；必需stage全失仍退出。MiniCPM-o4.5/4090D注入故障验证幸存replica完成双工语音。显式duplex group全失时普通chat可留HTTP200、duplex报不可重试replica_unavailable；满组才是可重试resource_exhausted。**建议语音/多replica部署优先审查**，它界定了确定性与可用性的不同故障边界。

### Megatron-Bridge：Qwen3.5-VL CP、多模态grad缩放与autograd CUDA上下文补齐

[NVIDIA-NeMo/Megatron-Bridge #7062](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/7062) · 11-22 11:10（上海）· [`70ba3586db7a`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/commit/70ba3586db7a7d3e834b1d43d4781292ba49560e)；[NVIDIA-NeMo/Megatron-Bridge #7060](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/7060) · 11-22 11:10（上海）· [`e39a1507fb02`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/commit/e39a1507fb024f7992b5c2d3fd474809f166daec)；[NVIDIA-NeMo/Megatron-Bridge #7061](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/7061) · 11-22 11:10（上海）· [`b16087df1a30`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/commit/b16087df1a306b916ac7591eea5dafe216430d6c)。

#7062为per-token loss模块单独关闭DDP average_in_collective，并避免已经rank-local/packed的RoPE position ids被CP再次切分。#7060修复MIMO finalize向新版MCore scale_gradients传Python float的接口崩溃，改为grad设备上的0维FP32 tensor，且保留除法表达式避免倒数乘法引入ULP差异。

#7061在torch_dist_init后做一元素backward，让CUDA0的持久autograd worker先获得context，避免TE重算norm成为其首次CUDA工作时invalid context。作者H100、torch2.15/TE2.20.2，Ultra LoRA 32GPU及Super32K/16GPU完成短训练与checkpoint；Super首次Triton编译仍需独立延长10min超时。MIMO/provider-only未走该入口，所以不能据此宣布所有TE上下文路径修好；根治仍应在TE。**建议组合使用MIMO/CP/重算的训练优先核对版本**。

### Megatron-Bridge：ERNIE5.5导出补回共享MTP body名字

[NVIDIA-NeMo/Megatron-Bridge #6963](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6963) · 11-23 10:09（上海）· [`f53143723543`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/commit/f53143723543834d1b0ade6fa5914eedad1116b0)。

共享MTP body在训练模型中只存一次，但已发布HF checkpoint还在MTP层名下保存副本，推理engine只按这些名字读draft。新导出依据--hf-model实际名字补副本（发布版1573份），无MTP或来源不含副本则不造新键；处理外部提供conversion tasks和FP8来源先反量化再导出。

新无模型单元测试进入默认CI，toy export逐位/strict shard/两种MCore基线验证；真实全模型从该分支导出后再vLLM启动没有重跑，SGLang还需要独立config修复#6905。**建议训练后要用MTP部署者优先阅读**，严格保存成功不足以证明推理可读。

### vLLM：Nemotron Shortcut适配含text/MTP/视觉，但依赖与KV精度需逐项核对

[vllm-project/vllm #68198](https://github.com/vllm-project/vllm/pull/68198) · 11-23 05:04（上海）· [`0fa666893b29`](https://github.com/vllm-project/vllm/commit/0fa666893b2989efc61dd25f7e915ef21d23cdd0)。

新增独立模型包接入SDSA稀疏注意力、GDP mixer及让专家从block输入提前dispatch的shortcut MoE；支持text/image/video及TP/PP/DP+EP/MTP。运行依赖layer-compact mixed-page KV #67941，MXFP8权重另需unaligned-N线性层#68353；config/processor来自checkpoint并要求trust-remote-code。

作者单节点4GB300、torch2.14、transformers5.22.2，定向1496通过/1个基线同样失败；review后165通过/1跳过。BF16/FP8 KV已有路径，NVFP4 KV明确仍等FlashInfer bump，NVFP4 expert测试亦因当前pin缺routing_weights_before_fc2跳过。**建议继续观察配套依赖并复现公开checkpoint**，模型注册合入不能等同所有声明精度都已在发布依赖上可用。

### PyTorch/Triton：编译期死锁、解释器精度与零步长的边界修复

[pytorch/pytorch #206050](https://github.com/pytorch/pytorch/pull/206050) · 11-22 12:48（上海）· [`60a7fa585a70`](https://github.com/pytorch/pytorch/commit/60a7fa585a70d2daefe9f3ab7889fbcc89f1cee2)。

Gloo worker在workMutex锁内销毁最后一个AsyncWork引用可能为Python对象decref获取GIL，而主线程启用Python dispatcher时持GIL进入enqueue等待workMutex，形成锁反序，表现为PT2分布式compile偶发hang。补丁先移出引用、unlock，再销毁work，queue协议保留。作者CPU双rank负对照2/2挂住、修复5/5通过；测试基线为内部fbsource且提供可公开运行的复现代码。PR API显示closed/merged=false且给出另一merge_sha，但默认main目录和已打开的固定commit `60a7fa585a70`均含此diff，本条按main commit收录，保留PR元数据冲突。**建议遇到compile-sync all_gather_object卡死时优先排查**。

### Triton：解释器浮点参数遵循launcher类型，constexpr零步长提前拒绝

[triton-lang/triton #12620](https://github.com/triton-lang/triton/pull/12620) · 11-23 08:25（上海）· [`7d4f88e3c706`](https://github.com/triton-lang/triton/commit/7d4f88e3c7065f134b598b2832a46549babf4368)；[triton-lang/triton #11242](https://github.com/triton-lang/triton/pull/11242) · 11-23 08:04（上海）· [`2e45f872f730`](https://github.com/triton-lang/triton/commit/2e45f872f7307b16c4ed6ec21518f8cda524cbdf)。

#12620修复解释模式将Python float保留fp64而compiled默认fp32的分支差异：未标注按fp32，有fp16/bf16/fp32/fp64注解就按该类型，bf16遵循launcher先fp32再截高16位的约定。例如1.00000001>1.0现在默认false、显式float64才true；作者80解释器测试及7156语言用例无新增失败。

#11242在编译前端拒绝constexpr step=0的range，避免lower成无报错GPU无限循环；runtime step未因此新增全面保护。**建议kernel开发者考虑采用并重看解释器对照**，这是调试可信度和失败边界改进，无LLM吞吐主张。

### LMCache：MLA共享对象不能用worker0的写入mask判定所有指纹

[LMCache/LMCache #6836](https://github.com/LMCache/LMCache/pull/6836) · 11-22 18:09（上海）· [`b1f50a83f2df`](https://github.com/LMCache/LMCache/commit/b1f50a83f2dfb229b3afd61332739a69fcfcfcbf)。

最终diff区分每rank独占read group与TP共享MLA latent：共享对象由先reserve的rank写入，worker0的store结果不能为其背书，曾令Kimi-Linear TP8仅约1/8 chunk注册fingerprint。现在只用单reader对象组判断worker0提交，lookup仍确认共享对象已loaded；纯MLA无独占组时还排除content-dedup跳过的chunk，避免错误注册。

PR正文无实质说明，本条依据已读代码/文档diff及新增TP/DCP布局、跳过chunk回归；未给执行回执、真实服务命中率或性能数据。**建议继续观察并优先补真实KV blend复现**，代码修正了提交判定，不足以单独证明所有共享cache生命周期正确。

### 同日撤回：PyTorch低精度转换不能算已交付改进

11月22日合入的float64→低精度单次舍入#204075及E8M0 subnormal舍入#204208，已分别在[`aabd19c7bc9b`](https://github.com/pytorch/pytorch/commit/aabd19c7bc9b1ae4922427a10ba9d3357f864611)和[`bd3e2c245090`](https://github.com/pytorch/pytorch/commit/bd3e2c245090920d286be339942ecbe1044a1eed)撤回；本轮打开了两次revert的固定diff。不能将早先合入或测试说明当作当前main已含的数值语义。撤回原因在已读commit材料中未充分说明，继续跟踪替代修复。

<a id="topic-6"></a>
## 硬件软件协同

### ATOM：重分配必须清空旧 restore 状态，避免跳算与异步写入后释放

**项目／合入时间／版本**：ROCm/ATOM 2026-11-23 01:55:25 UTC @ `36adbfddb637`。

ROCm/ATOM 在请求恢复、抢占、重新停驻到分层 KV 缓存时，旧的 loaded/failed/token-count 字段可能沿用：一条路径把未填充的 token 行误算为已缓存，另一条在新 load 仍写入时提前释放 block。本次在成功分配后、通知 connector 前统一重置四字段，覆盖普通和 early-load admission；已完成且保留 block 的 resume 不重新分配。补丁含三个能在旧 main 失败的状态序列回归；作者也披露缺 msgpack 导致的 85 个收集/初始化错误在两版相同。**建议采用前优先回归抢占＋重载＋取消链路**，这是静默错误及资源寿命修复，没有吞吐收益证据。

一手来源：[ROCm/ATOM @ 36adbfddb637](https://github.com/ROCm/ATOM/commit/36adbfddb6379a038c2a6ac0ea2e3552607e507d)。

### 昇腾 Mooncake：删除 prompt 尾 token 后截断 hash，保留前缀复用与旧版兼容

**项目／合入时间／版本**：vllm-project/vllm-ascend 2026-11-22 06:35:30 UTC @ `c12397a088f7`。

vLLM-Ascend 的 Hybrid、Mooncake V1/V2 调度路径在 prefill 去掉最后一个 prompt token 后，以 engine 的 hash block size 和 lookahead 截去失效后缀。可调用时委托 vLLM 原生截断接口；旧于 0.34 的回退原地修剪 hash 并清空 embedding hash 缓存，避免重算全部前缀。作者在早期修订、Windows/Python3.12、131072 个文本 token、块128、零 lookahead 下报告调度方法中位数 3.39ms→7.6µs（31次）；不含 prompt 构造、初始哈希及 NPU 执行，不能解释为服务加速。当前补丁增加原生/回退、嵌入 prompt、边界与重入回归。**建议深入读并考虑回归后采用**，关注 hash 语义而非仅比较 CPU 时间。

一手来源：[vllm-project/vllm-ascend @ c12397a088f7](https://github.com/vllm-project/vllm-ascend/commit/c12397a088f7e63ed9defae66ac98faffee88b81)。

### TPU 分离部署：主机页数据落地后，元数据也必须先恢复再发布

**项目／合入时间／版本**：sgl-project/sglang-torchtpu 2026-11-22 18:19:33 UTC @ `9975d2518202`。

SGLang-TorchTPU 为 decode DRAM offload 增加 host-page metadata 的导出、传输、恢复与清理，覆盖全注意力和滑窗（SWA）页。KV 搬运完成后先恢复元数据，经各rank同意后才把页发布为可供前缀命中的状态；registered owner要求的元数据缺失或recall失败时拒绝发布；无owner时空metadata可用于清旧状态，重用/设备回写时清除peer保留元数据，避免“字节已在、页身份不对”的缓存复用。失败拒发布不保证元数据事务回滚。补丁包含 push/recall/validate 和 DMA HiCache 回归；本轮未做 TPU 运行。**建议重点审读失败、取消及页重用边界后采用**，收益是正确性；不能把新增测试当生产大规模验证。

一手来源：[sgl-project/sglang-torchtpu @ 9975d2518202](https://github.com/sgl-project/sglang-torchtpu/commit/9975d2518202bb5fbe68a9e2d80acdf57af00da8)。

### MLXcel：Inkling 真权重验证发现 NVFP4 类型与 MTP 精确性缺口

**项目／合入时间／版本**：lablup/mlxcel 2026-11-22 22:36:46 UTC @ `4d1d2c209c1f`。

项目在 M3 Ultra 512GB 上验证真实 Inkling 工件：量化声明只在 hf_quant_config.json，旧 loader 将 BF16 dense 权重错误转成 FP16，导致反复输出 token 0；修复保留 checkpoint dtype。多 token 预测（MTP）原先跳过 block-vs-chain 精确性检查，宽度2/3可能偏离普通贪心解码，现接入共享检查和回退。作者报告普通 decode 41.6 token/s，而三个 MTP 配置为26.6/24.2/10.8，**未达加速标准**；128-token 局部 dtype 修复11.6s→5.1s不能替代整体对照。原生NVFP4峰值约274GB，报告只覆盖少量文本/视听样例。**建议参考其真权重验收流程并优先修正确性**。PR 已合入，随附报告中“Draft PR open”是未同步的历史文字。

一手来源：[lablup/mlxcel @ 4d1d2c209c1f](https://github.com/lablup/mlxcel/commit/4d1d2c209c1f8dd60989ebb8e18daac71fa2fa4e)。

### MLXcel：DeepSeek V3.2 的 MLA 与索引器同时接通 YaRN，长窗复测仍待完成

**项目／合入时间／版本**：lablup/mlxcel 2026-11-23 00:17:21 UTC @ `a9860ece908a`。

旧实现只从 rope_scaling 读取 softmax 修正，位置旋转仍用普通 RoPE，忽略 DeepSeek-V3.2 的 YaRN factor40、原始窗口4096配置；本次让 MLA 查询/键和 DSA 索引器六个旋转入口共用解析后的 YaRN 表，保留 default 配置行为，并拒绝非法旋转宽度。已合入PR披露 ROCm gfx1151 的99个定向测试通过，但该主机无法装下4bit真模型；M3 Ultra 的8K/16K困惑度重跑、top-k overlap仍待完成。**建议采用前复测长窗质量**；旧日志7.3→1403的困惑度崩塌不是修复后的成绩，DeepSeekV3另一实现同类缺陷另待处理。

一手来源：[lablup/mlxcel @ a9860ece908a](https://github.com/lablup/mlxcel/commit/a9860ece908a3cd18d95b6aa379e72c28319ee98)。

### 昇腾路由融合：在 TopK 输出前映射物理专家，并保留旋转门控的残差基底

**项目／合入时间／版本**：vllm-project/vllm-ascend 2026-11-23 01:16:34 UTC @ `29f19afb04e6`；vllm-project/vllm-ascend 2026-11-23 01:48:52 UTC @ `984ee29a6341`。

vLLM-Ascend 两项合入分别压缩 MoE 路由和 Engram 门控开销。TopK＋专家并行负载均衡（EPLB）映射仅自动启用于受支持 A3/A5、非分组 sqrtsoftplus 路由；先按逻辑专家算权重，再在片上映射物理ID，每次读取当前 ordinary/AllGather 表，跳过后续独立Map。逻辑ID捕获等不支持配置保留原路径，首次路由前配置须固定，Python与编译算子须配套。另一项在A3保留 FP32 旋转矩阵乘，融合其后归一化、评分、mask和残差加法，并分开评分输入和原残差输入以保留坐标基底。作者给出定向算子/图重放验收，但TopK最终router重构后的NPU/E2E、A5及配对profiling未证实；旋转门控验证环境的vLLM较旧，绕过了不相关初始化，不能当完整服务E2E验收。**建议深入读并按目标硬件回归**，不据省一次launch推算整网提升。

一手来源：[vllm-project/vllm-ascend @ 29f19afb04e6](https://github.com/vllm-project/vllm-ascend/commit/29f19afb04e61e9a613edf3225808499469d4d4e)，[vllm-project/vllm-ascend @ 984ee29a6341](https://github.com/vllm-project/vllm-ascend/commit/984ee29a6341c7e04ec5d418614b2d05ca36c63f)。

### ROCm：FlyDSL 先撤回寄存器约束回归，AITER 随后更新依赖

**项目／合入时间／版本**：ROCm/FlyDSL 2026-11-22 12:25:28 UTC @ `fe81c4077343`；ROCm/aiter 2026-11-22 13:52:29 UTC @ `183a6ce66ebb`；ROCm/FlyDSL 2026-11-23 01:07:24 UTC @ `39d1b66bee66`；ROCm/FlyDSL 2026-11-22 07:06:03 UTC @ `23b9fd75a2b7`；ROCm/FlyDSL 2026-11-23 01:06:31 UTC @ `287a01ce877a`。

FlyDSL 源码0.5.2对应提交撤回0.5.1引入的 strict AGPR（累加通用寄存器）约束路线，恢复相关FP4/FP8 GEMM内联汇编累加路径；AITER明确将依赖从0.5.1改为0.5.2以规避gfx1250性能回归。主窗源码版本随后升至0.5.3，补入导出HSACO标为binary而非fatbin、嵌入AOT运行库去调试信息等修复。另一个重要诊断修复：进程内LLD链接会重置LLVM选项，原先 final_isa.s 的二次codegen因此与实际二进制不同；本次重施选项，只修正汇编转储，实际二进制此前不受此问题影响。REST最新Release仍为0.5.0，0.5.2/0.5.3标签接口404，与Atom目录有冲突；此处只确认源码/依赖增量，不确认正式Release发布。**建议跟随依赖锁定版本并复核性能结论的ISA来源**；版本号不作为独立算法进展，不能把诊断输出修复宣称为运行加速。

一手来源：[ROCm/FlyDSL @ fe81c4077343](https://github.com/ROCm/FlyDSL/commit/fe81c40773431335c9f77ccd196a7adc0e3126d7)，[ROCm/aiter @ 183a6ce66ebb](https://github.com/ROCm/aiter/commit/183a6ce66ebb975223534f3bdeaca4132d2f4530)，[ROCm/FlyDSL @ 39d1b66bee66](https://github.com/ROCm/FlyDSL/commit/39d1b66bee66c2df86097eb61237e044e04ca4b7)，[ROCm/FlyDSL @ 23b9fd75a2b7](https://github.com/ROCm/FlyDSL/commit/23b9fd75a2b7988634869f3d8a8ce5de9c1704df)，[ROCm/FlyDSL @ 287a01ce877a](https://github.com/ROCm/FlyDSL/commit/287a01ce877a366ab437143c6665ac937b0d42ce)。

### ROCm MoRI：按128字节边界切分专家分发元数据，仅优化明确的EP4路径

**项目／合入时间／版本**：ROCm/mori 2026-11-23 02:01:31 UTC @ `0883c2f2f8bb`。

在 gfx1250、专家并行度4（EP4）、每block16 warps 的FP8/FP4 dispatch中，MoRI把元数据段切到字段128字节行边界，并用TDM搬运两端不完整行以替代远端masked store。作者单机256-CU设备、hidden7168、top-k6、每rank4096/8192/16384 token下，FP4 dispatch耗时72.54→63.99、100.70→96.03、181.78→161.26µs；FP8相应90.06→84.93、145.84→143.37、272.46→261.53µs。少于40行、小于等于每warp两token、BF16或独立metadata warps计划保留旧路径。**建议在匹配拓扑复现通信段**；数据是作者微基准，没有模型端到端或跨节点证据。

一手来源：[ROCm/mori @ 0883c2f2f8bb](https://github.com/ROCm/mori/commit/0883c2f2f8bb32d6f628df31494de86bb984a299)。

### AITER OPUS：补齐head-dim64和attention sink，同时拒绝不成立的反向路径

**项目／合入时间／版本**：ROCm/aiter 2026-11-22 09:39:06 UTC @ `35c23e610341`；ROCm/aiter 2026-11-22 09:00:13 UTC @ `cd78917e243b`。

gfx950 的BF16 OPUS attention扩展到对称head-dim64，并支持每查询头一个FP32 sink logit参与softmax分母但不带value；无有效key时输出0、LSE为sink，强sink采用重标定避免溢出。D64不支持滑窗，sink也不能配滑窗/非对称head；反向仍未实现sink梯度，所以对开启梯度的sink调用显式报错，避免前后向对应不同softmax。另一项gfx1250 GEMM把cluster_m由编译变体移到运行时参数，AOT枚举需覆盖非单调M边界，并把split-K输出描述符限制到实际stripe，避免4GiB跨度。**建议按模型几何和自动微分需求选择**；补丁与回归证明接入边界，没有统一性能提升可引用。

一手来源：[ROCm/aiter @ 35c23e610341](https://github.com/ROCm/aiter/commit/35c23e610341b2e9d98be989b48728405e77d7d1)，[ROCm/aiter @ cd78917e243b](https://github.com/ROCm/aiter/commit/cd78917e243b994d9023f02d4b0d2786c3c28ef5)。

### rocRoller/TensileLite：选择器声明捕获核的真实尺寸与寻址限制

**项目／合入时间／版本**：ROCm/rocm-libraries 2026-11-22 19:40:13 UTC @ `215d06c2d317`。

GridBased查找可能把RR_GEMM捕获核发给相邻但不合法的矩阵尺寸。本次为该核族补上M整除MT0、K整除DepthU、batch=1及每个A/B/C/D/MX-scale地址跨度小于4GiB的谓词；stride padding计入，size1维未用stride不计入，N不额外约束。作者在MI355X上分别构建启用/关闭rocRoller并做host与kernel测试，但最终rebase后未重新构建；后续捕获核栈尚有独立依赖，不能据本PR推定整栈已可用。**建议优先纳入动态shape/大stride正确性审计**，这是调度域修复而非新模型加速。

一手来源：[ROCm/rocm-libraries @ 215d06c2d317](https://github.com/ROCm/rocm-libraries/commit/215d06c2d31756b5602492f18996ec6332c20db4)。

### MLX：半精度注意力反向与CPU浮点量化矩阵乘保留FP32中间量

**项目／合入时间／版本**：ml-explore/mlx 2026-11-23 01:49:23 UTC @ `25f42f69c76b`；ml-explore/mlx 2026-11-22 09:31:16 UTC @ `07fff01caf4c`；ml-explore/mlx 2026-11-22 09:30:15 UTC @ `8e98fb8a454c`。

Metal 的缩放点积注意力（SDPA）向量雅可比积（VJP）把score临时块固定为FP32，半精度Q/K先转FP32再算score，以减少反向误差；梯度的其他中间量仍可能舍入，测试对BF16 dQ/dK允许0.3相对误差，不能称全FP32反向。CPU FP量化矩阵乘则把反量化scale、组内和与跨组累加改为float，最后转输出dtype，并覆盖普通/gather及转置两种路径。另项Metal分发拓宽部分causal head_dim256、512–1023查询行的fused路径，实际选择仍受设备和batch×head等条件约束。**建议先运行已有数值回归再采用**；三项源码变化均无本轮独立设备测速。

一手来源：[ml-explore/mlx @ 25f42f69c76b](https://github.com/ml-explore/mlx/commit/25f42f69c76bf5a5db6570df423abf9087fbfb0e)，[ml-explore/mlx @ 07fff01caf4c](https://github.com/ml-explore/mlx/commit/07fff01caf4c01a183f0b32c0861374bd4da2713)，[ml-explore/mlx @ 8e98fb8a454c](https://github.com/ml-explore/mlx/commit/8e98fb8a454ca247cf8e892b2b94407503234666)。

### CANN：交叉熵正反向不再先截断int64忽略标签；MoE形状契约同步补齐

**项目／合入时间／版本**：cann/ops-nn 2026-11-23 01:54:59 UTC @ `df61298137e9`；cann/ops-nn 2026-11-23 02:00:41 UTC @ `9fe0317f9638`；cann/ops-transformer 2026-11-23 01:55:15 UTC @ `91ad3cc1e0b9`。

Ascend950/arch35交叉熵正向和梯度以target的真实类型比较ignoreIndex，避免超过int32的标识被截断后误忽略合法标签；int32 target且正ignoreIndex超过2147483647时在tiling拒绝。负ignore约定和int64索引仍有各自分支，不能概括为所有超32位取值一律拒绝。MoeTokenPermute/V2补shape前置校验，950非INT8高维tokens通过展平再恢复形状；FP4的两个逻辑值虽然共用一字节，公开隐藏维仍为H而非H/2。**建议检查自定义loss和量化路由契约**，测试与patch已读；本轮只确认源码合入，未核这些commit对应哪一个正式CANN安装包。

一手来源：[cann/ops-nn @ df61298137e9](https://gitcode.com/cann/ops-nn/commits/detail/df61298137e9b2ed0380e54bd9f4047e30c7d589)，[cann/ops-nn @ 9fe0317f9638](https://gitcode.com/cann/ops-nn/commits/detail/9fe0317f96382cbebcdff97d2314c1f698ad4c63)，[cann/ops-transformer @ 91ad3cc1e0b9](https://gitcode.com/cann/ops-transformer/commits/detail/91ad3cc1e0b9929abe12e432ea22a4bac5e301ad)。

### CANN低精度MoE：流水、缓存绕过与上下文复用的证据成熟度不同

**项目／合入时间／版本**：cann/ops-transformer 2026-11-23 02:12:12 UTC @ `299909fb8ff9`；cann/ops-transformer 2026-11-23 01:37:04 UTC @ `387e8d225b85`。

GroupedMatmul MX A8W4新增按dtype、对齐、核数、并行度和K-stage收益选择的Blaze流水及回退策略，接口保持原样；提交说明仍明确泛化精度、二级冒烟及性能对照待补，**适合继续观察与源码阅读**。MegaMoE A4W4小batch decode则改动L2 bypass、L1预取、wave内上下文复用和受容量约束的L0 K256：作者在Ascend950八卡、MTE、Static OFF，对master ab6afe25与PR 5c5b03b2相同输入，每例480样本，六例平均耗时减少0.33%–15.86%，96项卡级精度检查通过且输出逐字节一致。六个编号用例不是完整模型负载，收益不宜外推。**后者可考虑对照复现**，正式包映射与更广shape覆盖仍待核实。

一手来源：[cann/ops-transformer @ 299909fb8ff9](https://gitcode.com/cann/ops-transformer/commits/detail/299909fb8ff974e9326656ec7d2430136b419eb5)，[cann/ops-transformer @ 387e8d225b85](https://gitcode.com/cann/ops-transformer/commits/detail/387e8d225b85a9fc4a4b192f165c545a53a7d27c)。

### HCCL：VCache预热沿业务流有序执行，并更正CCU_MS保序适用条件

**项目／合入时间／版本**：cann/hccl 2026-11-23 01:40:38 UTC @ `1a695e9e0b7c`；cann/hccl 2026-11-23 01:57:59 UTC @ `ec7eeedc9144`。

原独立预热流与业务流复用启动notify可能在大图触发602同步顺序冲突；本次复用param.stream，保留预热完成同步和buffer清理，删除预热流销毁以免误销毁调用方资源，并在构造DFX参数前实时检查捕获状态，捕获时跳过阻塞预热。另项仅更正文档：950DT单机、单层全Mesh、rank≤8、sum、非inplace、FP16/FP32/BF16的ReduceScatter可用CCU_MS保序（同段还有rank≥3）；不满足回退AI CPU，AllReduce与CCU_SCHED按各自回退规则。**建议优先审计通信stream寿命及确定性配置**；文档更正不是本日新算子，源码合入也不等于正式包已发。

一手来源：[cann/hccl @ 1a695e9e0b7c](https://gitcode.com/cann/hccl/commits/detail/1a695e9e0b7c73d86919f2f40ef4a3a318a5a9b5)，[cann/hccl @ ec7eeedc9144](https://gitcode.com/cann/hccl/commits/detail/ec7eeedc914456b6edb69db5cb4b0d2644e89905)。

### TPU接入与可观测性：修复上游同步边界，并按实际供给层归属缓存命中

**项目／合入时间／版本**：sgl-project/sglang-torchtpu 2026-11-22 15:16:26 UTC @ `9a2c157e26ba`；vllm-project/tpu-inference 2026-11-22 23:45:08 UTC @ `561451c94e98`；vllm-project/tpu-inference 2026-11-22 18:31:53 UTC @ `61899c6dff45`；vllm-project/vllm-torchtpu 2026-11-22 03:55:45 UTC @ `2d9c11777ba8`。

SGLang-TorchTPU为DeepSeek-V4.1修复重复row hook、mHC平台算子选择、Engram图像行和tracer参数可见性，并用优化屏障阻止XLA生成无法window的布局；作者在tpu7x-8、21层dummy权重、DPA8＋EP8启动与多种请求波形验证，不能当真权重质量验收。vLLM TPU另修Llama原生实现注册及DeepSeek routed-scale枚举适配。Raiden offload新增host/p2p/disk来源命中计数：相同token的KV和线性状态若来自不同层，记入较慢层；抢占后再admit不重复计，无法归属用external_unspecified。**建议用于故障定位和容量诊断**，命中率变化不能单独证明时延提升。

一手来源：[sgl-project/sglang-torchtpu @ 9a2c157e26ba](https://github.com/sgl-project/sglang-torchtpu/commit/9a2c157e26ba46810fd0e27106697a615b5bd8dd)，[vllm-project/tpu-inference @ 561451c94e98](https://github.com/vllm-project/tpu-inference/commit/561451c94e98fedaeb31ef41d949b805ba46e70d)，[vllm-project/tpu-inference @ 61899c6dff45](https://github.com/vllm-project/tpu-inference/commit/61899c6dff45253fd69b71acda7de425ad5345f6)，[vllm-project/vllm-torchtpu @ 2d9c11777ba8](https://github.com/vllm-project/vllm-torchtpu/commit/2d9c11777ba816743a159580911c038c25b71b49)。

<a id="topic-7"></a>
## 评估与可靠性

### SingVL S1换OptMerge权重后分数更新，复现清单仍指旧版

**机构：inclusionAI；11月22日权重提交`351afa6dbfc6`、卡修订`c0846efa6a53`。建议继续观察，先补齐新版本证据。**新卡把Decision Index 0.3公开Text/Vision写成70.95/77.158，140,178/12,210请求、图像上限1,638,400像素；相较上期是权重与结果变化。可是同卡下载命令仍固定11月18日`dd9fcbe2`，`inference/release.json`也仍指旧评估权重`93f15cd6`与旧batch分数69.75/73.7495。不能拿旧manifest、逐项结果和私有榜单资格为新分数背书，当前无法按既有清单重做新结果。此处记录发布事实与复现缺口，未独立确认提升。[固定新卡](https://huggingface.co/inclusionAI/SingVL_S1_4B/blob/c0846efa6a53bd21537f0f78e262dadcdabf39f7/README.md)；[仍旧的发布清单](https://huggingface.co/inclusionAI/SingVL_S1_4B/blob/c0846efa6a53bd21537f0f78e262dadcdabf39f7/inference/release.json)

### CompAct-Bench：上下文压缩是否保留工作状态，应通过后续执行检验（2611.25673v1，补录）

**机构／日期／版本**：复旦大学、腾讯等联合作者；2026-11-19 提交，v1；本轮补录。

基准从压缩前连续三次均成功的任务筛出 200 个实例，涵盖 BrowseComp、SWE-bench 与内部 CompanyBench；在执行 25%／50%／75% 边界压缩历史，恢复同一环境状态，并限制相同剩余动作数。12 个压缩模型在校准至 **10% 字符保留率**、每次摘要三次继续执行下，平均成功率 56.7%、最好 68.5%。遗漏常与重复工作／预算耗尽相伴，状态歪曲可能导致提前结束；代码任务还受到磁盘上既有修改的保护。**限制**：这是激进压缩、特定执行器和成功筛选任务的过程评估，不能外推成所有生产压缩均损失同样比例；内部任务限制完整复现。建议采用其恢复状态与等动作预算设计，另做无摘要继续、不同压缩率和持久化状态消融。[正文 §2–3、Table 2、Appendix F–G](https://arxiv.org/html/2611.25673v1)

### 可执行解释的代价：拟合全部演示仍可能无法泛化（2611.25896v1，补录）

**机构／日期／版本**：Gibran.AI、UTSA；2026-11-19 提交，v1；本轮补录。

研究用 Gemini-3.5-Flash 在 ARC-AGI-2 的 120 个公开评估任务上比较直接输出、可执行程序和演化搜索。严格按每任务全部测试输入正确、且三次运行多数成功计，直接输出解 93 题，最佳领域专用语言（DSL）程序方案解 68 题；该数是受限方案和预算的结果，并非等 token 的通用代价。更关键的是，106 题找到过完全拟合演示的程序，最多只有 68 题成功泛化；直接输出独有的 28 题中，27 题也找到过拟合演示但测试失败的程序。复用两份既有程序做演化搜索较相同种子的 best-of-2 程序多解约 9.2 题，不增加模型调用。**限制**：仍花 CPU 搜索，模型在公开集上的预训练记忆无法排除；首次找到可验证程序即停止会丢解。建议阅读，完善训练检查器时保留测试分布与选择误差审计，不能将“通过演示”当任务正确性。[正文 §2–4、Table 1、Appendix D–F](https://arxiv.org/html/2611.25896v1)

### AltRA-Test：稀疏人工标注也能检验替换裁判，但需同时报告人工信号质量（2611.26073v1，补录）

**机构／日期／版本**：Technion、University of Haifa、Google Research；2026-11-19 提交，v1；本轮补录。

不同样本由不同人标注时，要求固定标注组的旧替换检验难以使用。AltRA-Test 同时重采样样本和每样本的人类标注，构造纯人工／替换一人的混合面板，对标注者一致性（IAA）做非劣检验；默认容忍人类分歧增加 10%。论文给出渐近检验性质，并在 20 数据集、18 模型、五种提示下测试，约一半设置通过，任务影响大于模型／提示差异。**边界**：相对人工池非劣并不保证标签有用，人工一致性极低时尤其如此；模型与人共同犯的错无法识别，群体观点保留也需专门变体。建议考虑用于奖励／评估标注流程的每任务验收，同时报告原始 IAA、容忍幅度及人工群体构成；“约 100 样本常足够”不是固定样本量保证。[正文 §3–6、Appendix B–C](https://arxiv.org/html/2611.26073v1)

### Represented but misread：探针可读与模型可用之间，后训练改变的是哪一段（2611.26584v1，补录）

**机构／日期／版本**：University of Potsdam；2026-11-19 提交，v1；本轮补录。

对六个 4B–12B 开放权重基座及后训练版本，作者比较线性探针、下一答案词概率与模型拼接／激活替换。在德语人机文本识别中，信息可被探针读取，答案却可能反向排序；后训练平均敏感度变化 Δd′=-0.34。两个模型的分阶段检查把变化关联到 SFT，提示文本后位置的读出路径参与其中；公开可复现的否定命题任务反而被后训练修好，说明方向取决于训练信号。**限制**：作者身份语料不能共享；指标取两种答案的 log-odds，非自由生成准确率；因果干预集中于少数模型，训练研究还有文本长度／主题混杂。建议深入阅读并给后训练回归测试增加“表示是否可读、答案是否使用”的区分，避免据单一探针推断模型可靠能力。[正文 §2.2–2.11、§3–4／Limitations](https://arxiv.org/html/2611.26584v1)

<a id="coverage"></a>
## 覆盖、补录、更正与归档

本轮继续检查厂商研究/模型卡与权重、论文与正式会议入口、核心及相关开源工程、硬件与软件栈四类来源，并用中英文主题检索补充。检查记录及固定证据保存在本地 `.monitor/research-2026-11-23/`；以下是实际到达的范围，不代表厂商或全领域没有其他发布。

| 来源线 | 实际检查与局限 |
|---|---|
| 模型与工件 | 157个官方入口、125完整HTTP200；181个HF组织的362页目录可读，406个回溯候选中81在主窗，18个固定仓库审读。全局创建目录枚举翻过主窗起点（3354模型/1915数据集候选），只覆盖创建目录，未完成长尾语义和全站修订审读。H2O历史/配置401及动态页面、许可文件等缺口保留。 |
| 论文与评估 | 10个arXiv分类7完整，LG/AI/CV部分超时；目录仍见11月20日公告，不能据此否认周末新工作。9篇固定v1均为11月18–19日提交的补录；4篇完整正文直连，其他结合部分正文及web原始实验/限制段落核对。2278个候选仍待筛；精确更新窗口Atom超时、OpenReview两接口403、HF当日日榜超时、ACL部分与PMLR仅索引等缺口继续。 |
| 开源工程 | 84仓尝试中默认分支目录15、REST提交目录20、merged目录16、release feed10完成本轮枚举；连接器核39个PR及7个commit，1290候选未审。核心六项目及相关项目按各自ledger记录，优先固定PR最终diff。未审分页和库存继续保留；PR正文或随附报告的Open/Draft文字与已合入状态不一致时，以固定merge SHA和当前元数据分别说明。PyTorch当日被撤回的浮点舍入/E8M0改动不当作上线能力。 |
| 硬件软件栈 | 301次传输记录中163完整HTTP200；解析593个窗内版本目录，按机制选择审读。部分继承REST URL仍带旧查询边界，已另以当前边界补请求并记录403/429，旧响应不推进主窗水位；第一页面/Atom目录未普遍跨过回溯起点。CANN组件与正式包版本映射、libtpu行为、动态官方页面和未审库存继续。FlyDSL Atom与REST发行列表冲突保留，只确认源码/依赖变动。 |

**更正与证据边界**：整合前独立复核更正Seer为“一个共享9B基座、两个LoRA适配器”；确认AutoMaster的4B两项区间跨零及e-RWSC回退不获认证；TPU元数据恢复不等于事务回滚，Ascend路由融合的最终分布式/A5实证未齐。MLXcel两PR已合入但随附技术报告仍带Open/Draft旧标记；不因此误写为未合入，也不把未完成的长窗复测写成已通过。没有把旧工作补录当作今日首发。

**归档进度**：本轮通过用户网页授权恢复的GitHub连接，已将11月21、22日日报与索引补传到tracker/main，提交[84bd667](https://github.com/suhaibo666/tracker/commit/84bd66776f54474a4feaca6afda32be2fec897b5)，三文件回读逐字一致。今日日报与索引的本地/远端结果分记在运行回执，上传后另做内容与提交回读，不因上传失败重复生成正文。全局完整语义扫描水位继续保持未完成；下一轮从本轮发现截止续扫并回溯72小时，同时继承此前所有未关闭区间。
