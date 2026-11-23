# LLM 技术动态日报 · 2026-11-22

> 时区 Asia/Shanghai（UTC+8）。增量扫描窗口 **(2026-11-21 08:58:21, 2026-11-22 09:03:50]**；向前回溯72小时至 **2026-11-18 08:58:21**，并续查更早候选。实际生成 **2026-11-22T09:21:06+08:00**。日期仅精确到日时保留来源原日期；论文提交、首次公开、工件修订、合入与发现时间分别记载。

文中KV为键值缓存，MoE为混合专家，SFT为监督微调，RL为强化学习，GRPO为组相对策略优化，OPD为在线策略蒸馏；TP/PP/CP/DP/EP分别为张量/流水线/上下文/数据/专家并行，PCP/DCP为prefill/解码上下文并行。TTFT为首token时延，TPOT为单输出token时延，A2A为全交换通信，PPL为困惑度，MFU为模型计算利用率，SDPA为缩放点积注意力，MLA为多头潜在注意力，RoPE为旋转位置编码，HBM为高带宽显存，LDS为GPU片上共享存储。

## 本次重要变化

- **开放后训练配方值得读，奖励与可靠性回归同样具体。** Munin2.0补录公开SFT/异步GRPO与sandbox条件，SWE成绩改善伴随IFBench下降、Agent泄漏风险上升及训练奖励误判。不要只移植正向配方。[官方报告](https://www.foundationmodels.dk/news/2026/11/18/munin-20-release-note.html)
- **长窗推理先看完整负载和容量账单。** TPU专家路由表成段复制有32芯片、8K/1K负载的+6.2%吞吐对照，但基线是特定开发分支；Ascend PCP/DCP联合优化含其他补丁与高前缀命中，另有干净单轮显示容量增加而性能略退，不能归成单项加速。[TPU实验](https://github.com/vllm-project/vllm-torchtpu/pull/3388) · [Ascend固定提交](https://github.com/vllm-project/vllm-ascend/commit/dcf0f96c43ece47edabcf11d859a74ba390a5a2d)
- **通信重叠进入训练集成，也有静默未生效的修复。** Primus MegaUlysses在8×MI355X长序列训练中重叠投影/全交换，适用域限TP1/单节点CP且MLA依赖权重缓存修复；FSDP2的SDMA“installed”日志则曾掩盖AllGather退回计算核。[MegaUlysses](https://github.com/AMD-AGI/Primus/commit/abbbcb8dc9f0a7276dea339a4a989971eac0303e) · [SDMA修复](https://github.com/AMD-AGI/Primus/commit/ad2fd2b6150b61afdce6deed5fb799ca8f35a23f)
- **优先处理会改变数值、状态或可用性的边界。** 本轮包括动态var/std分母、FP16梯度、量化指数、缓存尾块竞态和草稿配置污染；Rubin预发布镜像的SM107错误结果规避还需显式切换ptxas，镜像未自动代办。SGLang TorchTPU三项Kimi稀疏DCP改动当天撤回，按撤回后的能力状态评估。[动态归约](https://github.com/google-pytorch/torch_tpu/commit/a2161629e4f1f9d03897d1e7bc0953a9d973bc92) · [撤回链](https://github.com/sgl-project/sglang-torchtpu/commit/9981cfc8bdc7a9789c43bdd9444d19e1e2491615)
- **新权重、研究结果和评估数据各有可复现边界。** Muse Spark Open的DFlash2交付长度/并发曲线，接受长度不是吞吐或答案正确率；补录TripleKV真机预取与Aggron模拟控制器，另外收录GNS批大小校准、OPD容量权衡和评测隔离。[DFlash2固定卡](https://huggingface.co/RedHatAI/Muse-Spark-Open-816B-A42B-speculator.dflash2/blob/c8961501a9cb36f6a682dbe32aa4d05f146a068b/README.md) · [TripleKV](https://arxiv.org/html/2611.25112v1) · [Aggron](https://arxiv.org/html/2611.26435v1)

- **TorchTitan把新训练机制和前缀复用接入主线，同时给出内存代价。** Flash-Next的PLE行分片避免每步全表聚合，GDN/KDA共享前缀状态支持rollout重用，但状态槽3→5、9B默认并发上限512→428，不能只看较低并发的提速。[训练集成](https://github.com/pytorch/torchtitan/pull/6053) · [前缀状态](https://github.com/pytorch/torchtitan/pull/6172)

[模型与技术报告](#topic-1) · [预训练与架构](#topic-2) · [后训练](#topic-3) · [推理与部署](#topic-4) · [开源工程](#topic-5) · [硬件与软件协同](#topic-6) · [评估与可靠性](#topic-7) · [覆盖与续查](#coverage)

<a id="topic-1"></a>
## 模型与技术报告

### Munin 2.0（11月18日，补录）：开放小语种 Agent 后训练配方，也披露奖励噪声和能力退化

Danish Foundation Models 11月18日发布；本轮补录，非今日首发。固定主模型 `f357ab54bb2ee87bb69984afb29ab092284707f5`、SFT `5a344e21d454b8c54642c54df5e3406154597764`、NVFP4 `587a57d0e7ae1d2365bf5f75604ca0897b78dc99`，均附 NVIDIA Nemotron Open Model License。沿用 Nemotron 3 Super Base 的120B/12B激活、Mamba-2与注意力混合架构和128K训练上下文；一轮6.8M会话/11B tokens SFT，仅3.4B assistant tokens参与损失，再用异步组相对策略优化（GRPO）训练约119K软件工程/终端等episodes，奖励全部程序化、RL不含丹麦语。公开Prime-RL分支与sandbox服务；完整训练内容清单仍待发布。训练约2.5天LUMI128GPU加2天Gefion128H100，总约1.5e21 FLOP；公开说明约千并发sandbox才是吞吐瓶颈。

同vLLM FP8、temperature0.8及bash/opencode/pi三harness，SWE-bench Verified **首128题**66.2%对SFT58.1%/Nemotron49.3%；全500题仅bash另为65.7%对46.6%，不能混成同一榜单。检查点部分按这些benchmarks选择，成绩可能乐观；IFBench49.7对Nemotron74.0，Agentic Misalignment中的泄漏79%对SFT56%，并非能力与可靠性同步提高。作者事后审计还发现SWE-smith Rust/C++/Java的参考修复18–42%被误判失败，修复已入公开训练配置；这影响训练奖励，不是评测标签。**建议深入读并考虑复现**sandbox隔离、奖励重置和指令遵循回归；128K末端质量未验证，不把配置长度当实测能力。[官方发布与完整限制](https://www.foundationmodels.dk/news/2026/11/18/munin-20-release-note.html)；[固定主卡](https://huggingface.co/danish-foundation-models/munin-2.0/blob/f357ab54bb2ee87bb69984afb29ab092284707f5/README.md)；[SFT卡](https://huggingface.co/danish-foundation-models/munin-2.0-sft/blob/5a344e21d454b8c54642c54df5e3406154597764/README.md)；[NVFP4卡](https://huggingface.co/danish-foundation-models/munin-2.0-nvfp4/blob/587a57d0e7ae1d2365bf5f75604ca0897b78dc99/README.md)。

<a id="topic-2"></a>
## 预训练与架构

### OpenGNS：梯度噪声尺度不能直接充当临界批大小（2611.24331v1，补录）

**机构与时间**：Jülich Supercomputing Centre；v1提交11月18日，本轮补录。开放400余条梯度噪声尺度（GNS）轨迹，覆盖11M—1B的ResNet、GPT、DiT；语言模型以AdamW在FineWeb上训练20B tokens，按最大更新参数化（muP）扫描批大小与学习率。相同loss附近GNS跨规模较稳定，但对临界批大小（CBS）的偏差随任务改变：分类倾向高估，生成任务倾向低估。

**证据边界与建议**：原始梯度在AdamW预处理前测量，未记录optimizer state，无法据此重建预条件噪声尺度；最大仅1B，不能直接外推百B长训或泛化。**建议深入读并考虑复现**本方模型的GNS—CBS曲线；这是批量调度的校准数据，不能把单个GNS估计当最优global batch。[原始报告§3–4、附录A](https://arxiv.org/html/2611.24331v1)。

### FlashMorph2：用离散混合模型自己的隐藏态选择全注意力层（2611.24809v1，补录）

**机构与时间**：复旦／字节Seed／南洋理工；v1提交11月18日，本轮补录。将FlashMorph的连续分支混合、教师强制隐藏态与冻结权重选择，改为采样离散混合架构端到端运行，并联合更新层选择分数及恢复权重，减少“选择时有效、部署后失效”的分布差异。

Qwen3主实验0.6B/1.7B，以DCLM数据、相同全注意力预算和恢复配方比较；线性分支包括Lightning Attention、GLA和Gated DeltaNet，默认线性:全注意力为3:1。作者汇总相对FlashMorph在RULER／LongBench平均提高2.42／1.08点；0.6B—8B的0.72—3.57 GPU小时**仅为层选择**，不含完整转换和恢复。**建议考虑复现**预算匹配的选择消融，先核验逐项召回与总训练成本，再谈架构转换收益。[报告§4–5](https://arxiv.org/html/2611.24809v1)。

### Kernel Ridge Attention：协方差引导的定长记忆仍是小规模架构证据（2611.23896v1，补录）

**项目与时间**：KRA作者团队；v1提交11月18日，本轮补录。以递归岭回归写入关联，结合学习非线性特征与内容相关遗忘；状态包含协方差和系数矩阵，大小不随上下文增长。

6层约31M模型、100M相同训练tokens、单seed、2K训练窗下，PPL为88.00，Transformer++为97.21；但自然稀有重复召回25.41%低于Transformer++的32.00%。8×RTX5090完整训练step31.77ms，比Gated DeltaNet的27.97ms慢13.6%；8K后状态1.165MiB，后者0.615MiB。**继续观察，适合小规模机制复现**；当前证据支持比所测递归基线更好的质量，未证明大模型长窗、吞吐或召回全面占优。[报告§4及训练条件](https://arxiv.org/html/2611.23896v1)。

<a id="topic-3"></a>
## 后训练

### Ai2 Dolci-Think-SFT-Clean-v2：主分支合入残余身份泄漏清理，复现需固定数据版本

11月21日主分支合入 `b83abf4073b66569ba788ba25f4a092f9ad2a116`，删除63条模型自认DeepSeek R1或Alibaba开发的记录（61 Aya、2 Precise IF），2,050,381→2,050,318行，tokens减少172,952；本次是既有Clean-v2的数据修订，不是新数据集。卡内把准备时间记为11月7日，所指数据revision `526926efcb3418af15aa57708e5956019f37be8a`的API元数据为11月8日；与本轮主分支合入时间分开记录。发布逐ID清单、脚本和manifest，47 shards改写、81保持不动，均为作者声明，本轮未重下约30GB逐shard复核。

筛选先人工核读122条名称命中，再补读215条不同拼写等候选，保留正常讨论；不能用全局关键词删除替代语义判定。清理后也不剩以Olmo命名的身份回答，训练方需另补身份数据；tokenizer.json与Transformers5加载路径还会给63行不同token计数。**建议采用版本固定并审查数据配比**，当前没有完整Clean-v2的新训练验证，不能把63行删除直接换算成质量提升。[固定修订卡](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-Clean-v2/blob/b83abf4073b66569ba788ba25f4a092f9ad2a116/README.md)；[删除脚本](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-Clean-v2/blob/b83abf4073b66569ba788ba25f4a092f9ad2a116/meta/drop_identity_residual.py)；[manifest](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-Clean-v2/blob/b83abf4073b66569ba788ba25f4a092f9ad2a116/meta/manifest.json)。

### OPD目标的理论分析：前向KL的优化优势不等于所有学生都该换它（2611.23397v1，补录）

**项目与时间**：Strauss／Fowl／Razin；v1提交11月17日，本轮补录。研究在线策略蒸馏（OPD）的序列级与token级目标：序列级有偏好未来可与教师对齐路径的偏差，可能先学后部token；在给定表格化学生与梯度更新假设下，反向KL及Jensen–Shannon散度可在师生相距很远时梯度消失，前向KL更易学习低概率目标。

**限制不能删掉**：MATH的Qwen3-0.6B/1.7B学生、4B—32B非思考教师、三seed实验却显示容量差越大，token级反向KL越有利；0.6B对32B教师，MATH500每题8采样平均、三seed约51.5%对前向KL47.5%。理论讨论的是优化障碍，容量与任务改变最优目标。**建议深入读并复现**冷门API／低初始概率与已有数学能力两个情境，避免统一替换loss。[论文理论与§4、附录E](https://arxiv.org/html/2611.23397v1)。

### COPD：先向黑盒教师提问，再用特权上下文监督无教师轨迹（2611.23673v1，补录）

**机构与时间**：北大／京东等；v1提交11月18日，本轮补录。学生在任务执行中自主询问DeepSeek-V4.1-Flash，保留成功交互作为特权参考；随后对新生成、无顾问的学生轨迹，用有／无参考两种视图的token对数概率差做自蒸馏，并联合任务PPO。教师只提供文本，算法仍需访问学生两种视图的概率；不能误写成黑盒教师logits可用。

Qwen3.5-4B/9B各100次更新、6,400 episodes；640项CAMG-II主测关闭顾问。4B平均45.3→59.8%，最强黑盒对照ActionOPD为51.2%，所谓2.46倍是**相对基座的增益比**。外部迁移只测无网络子集，部分域仍有对照领先。**建议考虑复现**顾问预算、异步行为概率校正及去掉特权参考的消融。[论文§4–5、Table1](https://arxiv.org/pdf/2611.23673v1)。

<a id="topic-4"></a>
## 推理与部署

### Red Hat / Inferact：Muse Spark Open DFlash2 草稿权重，把长窗接受率与实际加速分开交付

11月21日固定 `c8961501a9cb36f6a682dbe32aa4d05f146a068b`，附Muse Model License与使用政策。五层BF16 drafter读目标42/46/50/54/58层隐状态，每轮提议7 tokens，针对NVFP4 Muse Spark Open verifier。作者主套件用4×B200、TP4+EP、FP8 E4M3缓存，temperature1/top-p0.95/top-k64、xhigh、输出上限4096，标准接受率并发64，MRCR并发4；3,543 prompts聚合接受长度4.215，**包含verifier本身贡献的1 token**。另4×B300、100个原始八针MRCR历史，实际输入132K–1.044M tokens，接受长度5.450；这是草稿与verifier匹配程度，不能当答案正确率或5.45倍吞吐。

同卡附非投机/自适应验证开关、并发1/4/8/16/32的速度曲线；每点仅一次30秒warmup+100秒测量，无重复试验置信区间，当前固定目录不含原始速度数据和MRCR选择manifest。**考虑复现**本方长度/并发下的时延—吞吐曲线，再决定是否采用；已有1M缓存配置不等于业务长窗质量通过。[固定卡与测量设置](https://huggingface.co/RedHatAI/Muse-Spark-Open-816B-A42B-speculator.dflash2/blob/c8961501a9cb36f6a682dbe32aa4d05f146a068b/README.md)；[固定配置](https://huggingface.co/RedHatAI/Muse-Spark-Open-816B-A42B-speculator.dflash2/blob/c8961501a9cb36f6a682dbe32aa4d05f146a068b/config.json)；[工件目录](https://huggingface.co/RedHatAI/Muse-Spark-Open-816B-A42B-speculator.dflash2/tree/c8961501a9cb36f6a682dbe32aa4d05f146a068b)。

### Red Hat：Hy4 NVFP4 专家量化附同栈质量对照，DFly2 部署仍有门禁

11月21日固定 `b036a52330c601e486d8417dfb5fd1f4e4b52393`。只把77个MoE层的routed experts做NVFP4权重/激活量化，16元素FP8尺度再加FP32全局尺度；注意力、共享专家、router、视觉模块和DFly2保持BF16。checkpoint约1.55TB→447GB是文件大小变化，未给吞吐、峰值显存或端到端成本对照。发布512条、最长2048tokens的校准与名称转换配方；作者称逐tensor核对非专家权重bitwise一致。

4×B300、TP4、同vLLM/lighteval、high reasoning、temperature0.9/top-p1、单seed，NVFP4对官方FP8：GSM8K Platinum96.28%对96.77%，GPQA Diamond89.90%对91.41%；198题GPQA标准误约2.1个百分点，差异不足以证明稳定质量退化或完全等价。未覆盖图像、视频、长上下文与Agent任务。虽checkpoint带BF16 DFly2，卡明确vLLM当前DFly仅接受FP8/MXFP4 Hy4，不能据“带drafter”假定该工件已可投机。**考虑定向复现**，先检查本方vLLM版本和多模态任务；许可沿腾讯Hy条款。[固定卡/配方/质量表](https://huggingface.co/RedHatAI/Hy4-NVFP4/blob/b036a52330c601e486d8417dfb5fd1f4e4b52393/README.md)；[交付目录与LICENSE](https://huggingface.co/RedHatAI/Hy4-NVFP4/tree/b036a52330c601e486d8417dfb5fd1f4e4b52393)。

### TripleKV：预测工具返回时间，把SSD预取放在主机内存（2611.25112v1，补录）

**机构与时间**：UIUC／IBM／雅典国立技术大学；v1提交11月18日，本轮补录。工具执行期间预测下一次KV复用时刻，将SSD块提前读入DRAM；避免预占GPU缓存，并通过去重、被GPU淘汰时的victim refill及可超时保护减少无效预取。

单H100 NVL94GB、Qwen3-30B-A3B BF16，KV预算HBM32GiB／DRAM64GiB／SSD1TiB；vLLM0.24，12/16/20并发Agent。四个各50任务的真实轨迹回放中，相对补上SSD层的vLLM，P95首token延迟最多下降35/49/23/33%，不是相对所有新版vLLM。SSD层不淘汰、工具时间按轨迹重放，限制向多租户和大规模存储推广。**建议考虑复现**本方工具时延分布与预取失准压力。[设计与§6实验](https://arxiv.org/html/2611.25112v1)。

### Aggron：在队头KV阻塞时安排摘要，头条性能来自模拟（2611.26435v1，补录）

**机构与时间**：UIUC／IBM；v1提交11月19日，本轮补录。正常增长阶段仅摘要可压缩的prose、冻结工具原始输出；达到接近模型窗口的reset ceiling时，会连同冻结工具段一起摘要并重启，因此不保证工具原文永久保留；建立独立摘要队列，仅在主队列因KV空间队头阻塞、且预计摘要能及时完成时利用剩余资源，避免摘要与正常请求抢同一队列。

准确性使用单H100 NVL、Qwen3.6-35B-A3B FP8等真实任务；**1.34—1.40倍吞吐、P99下降27—31%是1000程序轨迹和Poisson到达的离散事件模拟**，对照Slipstream/APEC。模拟以单GPU测时校准，端到端验证主要是vanilla；不可改写为Aggron大规模实测。摘要可能改变行为且收益依赖负载。**建议深入读并先复现控制器与模拟误差**，再在目标并发上压测。[§4–6与模拟边界](https://arxiv.org/html/2611.26435v1)。

### PRKV：多跳重要性传播改善淘汰决策，但计分会增加首token开销（2611.24969v1，补录）

**项目与时间**：PRKV；v1提交11月16日，本轮补录。用注意力图上的个性化PageRank传播重要性，覆盖prefill后压缩及decode周期淘汰；全局变体用隐式低秩运算，避免完整token两两矩阵。

同硬件、BF16、FlashAttention2、相同固定生成长度对照中，Llama3.1-8B-Instruct的RULER16K、50%压缩，PRKV／全局变体TTFT增加2.4／6.3%，decode46.49→49.12／49.14 tokens/s；计时部分使用RTX PRO6000 Blackwell，不能混用主实验A100口径。90%压缩并非质量无损，Qwen3-8B的LongBench显著掉分。**建议考虑复现**质量—内存—TTFT曲线；FlashAttention不返回注意力图，本文实现会重算窗口注意力。[§6、附录B/H](https://arxiv.org/html/2611.24969v1)。

### 高温长生成：先审采样候选集合，再把崩溃归因于训练（2611.22148v1，补录）

**项目与时间**：Roush等研究团队；v1提交11月17日，本轮补录。29个1B—122B开放模型、H200、统一vLLM logits processor，对比温度提升如何放大不同采样器的候选集合。高温下top-p／Top-H纳入大量低概率token；先确定集合再施加温度，可明显改善稳健性。

SWE-bench Verified固定100题、mini-swe-agent中，Qwen3.5-9B在温度2的top-p/top-k/Top-H解决0题，top-nsigma解决51题；**温度1时差距小得多**。这不是长上下文退化全部由采样造成的证明；写作部分有模型裁判，题量和模型设置限制泛化。**建议考虑复现**采样顺序及候选集合日志；生产参数默认温度1时不要预期相同幅度。[报告§4–5](https://arxiv.org/html/2611.22148v1)。

<a id="topic-5"></a>
## 开源工程

### TorchTitan：Flash-Next 的 PLE 行分片与图可重放训练进入主线

**版本与来源**：[#6046 · 2026-11-21](https://github.com/pytorch/torchtitan/pull/6046)（代码基线[fd913e6c7f](https://github.com/pytorch/torchtitan/commit/fd913e6c7fe0adff461b6600c87ba70cc8fe8f30)）；[#6047 · 2026-11-21](https://github.com/pytorch/torchtitan/pull/6047)（代码基线[7bc80f098b](https://github.com/pytorch/torchtitan/commit/7bc80f098b2ddfb5584654f84e0bf20d3a585b7a)）；[#6052 · 2026-11-21](https://github.com/pytorch/torchtitan/pull/6052)（代码基线[a00933d2f1](https://github.com/pytorch/torchtitan/commit/a00933d2f14dafed6d3224dec19eec0d0fbcad2d)）；[#6053 · 2026-11-21](https://github.com/pytorch/torchtitan/pull/6053)（代码基线[b9c588a15f](https://github.com/pytorch/torchtitan/commit/b9c588a15fa49d583d3e709337155252691041bf)）；[#6054 · 2026-11-21](https://github.com/pytorch/torchtitan/pull/6054)（代码基线[0a762a88e1](https://github.com/pytorch/torchtitan/commit/0a762a88e1079dd648641be6d3da7ec26cccc87b)）。

新增Qwen3.8-Flash-Next文本/多模态模型、HF转换、四路门控残差、Qwen Sparse Attention（QSA，稀疏注意力）与哈希n-gram嵌入（PLE）。51.2B参数的PLE表若走普通全参数分片数据并行（FSDP）会每次聚合约95.37GiB BF16；实现改为DP按行持有、通过显式前后向全交换完成查表，排除FSDP all-gather。后续两项把QSA的动态nonzero/bincount及PLE的主机计数改成固定形状/固定容量槽，解除图捕获的主机同步和批次形状障碍。TP+序列并行也已接入，索引器teacher的KL梯度只贡献一次；**DistMuon尚不支持TP，配置会拒绝这一组合**。

125B-A6B配方在16张GB300、FSDP16/EP16、自然路由、16K打包行、每步约2M tokens、完整重计算及前后向整图下，作者在**较早修订（1/n router改动前）**记录步骤4—30为378.5TFLOPS/GPU、模型算力利用率（MFU）15.14%、峰值243.4GiB；418.2TFLOPS/16.73%是强制轮询路由，不能用作真实路由成绩。融合QSA teacher、门控残差和GDN投影分别减少中间张量/复制；部分融合只有误差界，非逐位一致。B200集成测试当时尚无golden，能检查capture/replay但未完全排除陈旧状态；GDN split迁入模型后该测试未重跑。当前API正文另列current-stack自然路由378.6—379.0TFLOPS/GPU、243.59GiB，强制轮询416.4TFLOPS/GPU；不能把较早分项消融数值与该表混作同次测量。第6项PP仍是Draft/状态待核，**不写为已交付流水线并行**。建议深入阅读并按真实文档分布复现，先预算PLE通信、容量槽与~243GiB峰值，再比较真实路由。

### TorchTitan：GDN/KDA rollout 复用前缀状态的收益伴随状态容量代价

**版本与来源**：[#6172 · 2026-11-21](https://github.com/pytorch/torchtitan/pull/6172)（代码基线[0e8a05baf0](https://github.com/pytorch/torchtitan/commit/0e8a05baf048c8f4efd54cfec3e03966008b8c82)）；[#6261 · 2026-11-21](https://github.com/pytorch/torchtitan/pull/6261)（代码基线[34ed86f1d3](https://github.com/pytorch/torchtitan/commit/34ed86f1d3319e971e34ac9f046ddb93d68c1658)）。

#6172把线性注意力的no_buffer换成extra_buffer：prefill在64-token边界写入卷积/递归状态，KDA同时保存未完成chunk的重放尾部；overlap scheduling重新开启，同组rollout可等待首样本prefill再共享前缀。150-token提示重复时复用到128，续轮200-token状态对齐到192。每请求状态槽从3增至5，Qwen3.5-9B/GB300默认上限512→428；Attention Gym暂固定main@755691f，不能只升级TorchTitan包而忽略依赖。

作者在单GB300、TP1、1000输入/512输出、breakable图、6轮中位数下，9B的32提示×8样本由15.22降至11.50秒；128独立提示9.27→8.22秒。测试负载都低于新的428并发上限，**没有证明满状态池仍同幅提速**。#6261再让KV与递归状态共用内存池，Qwen3.5-9B/TP1/单GB300在512×16K入/8K出约快6%，1024×512入/4K出反而慢6%；长短负载权衡真实存在。该项还带临时backend allow-list适配，后续上游API迁移待跟踪。建议考虑复现GRPO共享提示和状态受限负载两种曲线。

### TorchTitan：低精度堆叠权重的非零维分片静默错序修复（补录）

**版本与来源**：[#6259 · 2026-11-20补录](https://github.com/pytorch/torchtitan/pull/6259)（代码基线[50da8142a6](https://github.com/pytorch/torchtitan/commit/50da8142a6ff2da74fbaa660c152a82be46a0fb0)）。

NVFP4/MXFP8权重包装的all-gather前后hook把非零分片维移到首维再移回，但FSDP2已沿原分片维重组，导致[2,F,D]、Shard(1)等堆叠FFN的行次序被重排两次，形状仍正确而无异常；Shard(0)不受影响。现在保留参数原布局，并经真实FSDP2验证，而非用简单cat模拟聚合。作者另称作业loss接近BF16，未披露足以归因的完整收敛/规模配置。**建议优先审查并考虑回迁**，用量化前后参数坐标与一次优化步对照，不能仅凭shape和loss未发散验收。

### Megatron：FsdpMuon 支持 HSDP/HFSDP 混合数据并行网格

**版本与来源**：[#9005 · 2026-11-21](https://github.com/NVIDIA/Megatron-LM/pull/9005)（代码基线[da0d3d473c](https://github.com/NVIDIA/Megatron-LM/commit/da0d3d473cd501514491d775b912c8c4a333382e)）。

Muon的所有者规划从每网格独立改成构造时全局收集元数据，平衡dense/专家矩阵正交化；实际gather/scatter只联系共同分片该buffer的rank，点对点通信走默认进程组。HSDP在各副本内独立step，HFSDP跨整个网格，支持num_distributed_optimizer_instances>1与outer_dp_sharding_strategy=no_shard/optim。代码仍要求非量化主权重/梯度、固定参数成员顺序，并要求各rank一致地识别有梯度参数。作者称1节点8×H100的MFSDPv2目录3330通过，(2,2)网格对照未分片Muon；未给性能提升或长训练收敛结果。建议深入读所有者资格与混合网格通信，具备混合FSDP需求时考虑复现。

### Megatron：共享训练配置补齐五项原生消费者

**版本与来源**：[#9428 · 2026-11-21](https://github.com/NVIDIA/Megatron-LM/pull/9428)（代码基线[12541d892c](https://github.com/NVIDIA/Megatron-LM/commit/12541d892c73eeb1b48e244fb6b01c2589938e79)）。

合并Bridge配置的一部分：可控制优化步成功标志在模型并行rank间的AND归约、只跳过**报告梯度范数**的MAX同步、以max_steps固定学习率/权重衰减全程而train_iters控制本段停机、外部launcher单卡可见时用CUDA0、尖峰loss阈值可配置。max_steps须为正且≥train_iters，不支持按samples计的训练；跳过grad_norm报告同步不改变梯度裁剪，关闭step-success同步则依赖rank本地结果一致。该PR有实际运行消费者，值得作为统一配置迁移的一项检查，**不等于Megatron-Bridge整体已并入**；没有性能/收敛新对照。建议按现有launcher与分段训练需求考虑采用。

### Megatron M2N：按专家族批量重分片，收益取决于网络规模（补录）

**版本与来源**：[#8606 · 2026-11-21补录](https://github.com/NVIDIA/Megatron-LM/pull/8606)（代码基线[ae90c188f4](https://github.com/NVIDIA/Megatron-LM/commit/ae90c188f44080a8a76da0cec1ee608e6d0d864a)）。

将每个MoE专家逐张量调用改为按expert family堆叠成EP分片张量，支持跨步长GLU shard并默认拆分communicator，面向训推分离权重refit。作者在GB200 NVL72、BF16、训推GPU分离、预热3次后10次中位数、两臂均split communicator下，Nemotron-3-Ultra的16训练GPU→2×8推理GPU，M2N1602→706ms，对send/recv4597ms；32节点跨两机架时35891→22615ms，对send/recv32042ms，仅1.42倍。不能把单机架6.51倍推广跨机架。

目的权重逐tensor bit fingerprint与send/recv一致；该证据验证refit数值，不含全RL训练吞吐/最终质量。Dense模型的前后两列抖动且plan不变，作者明确不归因新优化。建议深入读专家堆叠、切片stride与通信分域；考虑以本方拓扑和权重更新频率复现。

### Megatron：动态梯度累积与 CPU offload 恢复补齐正确性（补录）

**版本与来源**：[#9349 · 2026-11-20补录](https://github.com/NVIDIA/Megatron-LM/pull/9349)（代码基线[25bf0e10d5](https://github.com/NVIDIA/Megatron-LM/commit/25bf0e10d512aa1a3b13c1d2174747c61506c910)）；[#7449 · 2026-11-20补录](https://github.com/NVIDIA/Megatron-LM/pull/7449)（代码基线[fbc9562e70](https://github.com/NVIDIA/Megatron-LM/commit/fbc9562e70851ec2575a2567493616e894d2f33c)）。

#9349不再用上一次迭代的backward次数推断本次GTP输出头何时reduce-scatter，改为当前累积窗口计数；缺失边界、no_grad或晚到forward退至最终fence，必须先于DDP finalize。优化合并仅对PP=1、合资格eager输出头成立。它修复动态microbatch数量可能提前提交不完整梯度的问题，不能看固定GA测试推断所有外部RL调度已安全。

#7449修复precision-aware HybridDeviceOptimizer的多条恢复路径：Adam步数被占位step覆盖、CPU/FP32主副本未随checkpoint刷新、torch格式dummy_step把初始化权重写回，以及重叠传输中的梯度cast竞争。权重单独加载和FP32模型也在影响面内。**建议使用相关offload/动态GA的系统优先核查并做连续训练与保存恢复逐步对照**；本轮只核原始PR/固定版本，未独立训练，不能据此声称所有checkpoint格式已等价。

### Megatron：bias-free SwiGLU 的 Triton 前后向落地

**版本与来源**：[#9060 · 2026-11-21](https://github.com/NVIDIA/Megatron-LM/pull/9060)（代码基线[3fc2893ec6](https://github.com/NVIDIA/Megatron-LM/commit/3fc2893ec6d1d6a337cea56a712fb424faa9537d)）。

对plain/clamped/router-weighted且无bias的SwiGLU，用Triton一次写入两半输入梯度，router权重梯度与激活在同一pass归约；F≤1024时多行tile，配置只依赖行宽、不做运行时autotune，适合图捕获。bias、SiTU-GLU、CPU及无Triton环境仍走旧路径。作者报告BF16差异在一ULP内，router梯度的FP32求和次序可变，不能写逐位不变；Qwen3-30B-A3B及Llama3-8B MXFP8有步骤改善描述，但缺统一量化表，不摘录倍数。建议考虑复现无TE op fuser的MoE短行负载，并按确定性要求验收梯度。

### DeepSpeed：MQA AutoTP 修复不能替代默认梯度裁剪验证

**版本与来源**：[#8962 · 2026-11-21](https://github.com/deepspeedai/DeepSpeed/pull/8962)（代码基线[7681845b3e](https://github.com/deepspeedai/DeepSpeed/commit/7681845b3e20b371e28f1affe6cd6b813f1aa237)）。

GPTBigCode的融合QKV含共享单KV头；TP>1时原逻辑会产生空Q shard，非连续KV梯度与特定torch_npu post-accumulate hook又使累积梯度重复归约。实现按Q头分片/复制KV、连续暂存非连续梯度，并让fallback只归约本microbatch新增梯度；TP超过Q头及该布局的显式partition/HF plan被拒绝。

作者验证4×910B4、CANN8.5.1、torch/torch_npu2.10、Transformers5.19；另有4×A100部分组合，TP2/DP2、ZeRO0—3及TP4。**仅FP32且parity关闭clipping**：复制KV在默认范数中仍被TP重复计数，tiny模型的正确clipped对clipped四步差4.72e-4；BF16/FP16、generate/past-KV和无offload外配置未完整覆盖。建议优先阅读并考虑正确性回归，不能直接宣称MQA训练已在所有精度收敛一致。

### DeepSpeed：静态 rollout 图按本地 KV 形状与正确 mask 重放

**版本与来源**：[#9127 · 2026-11-21](https://github.com/deepspeedai/DeepSpeed/pull/9127)（代码基线[388f6a3f05](https://github.com/deepspeedai/DeepSpeed/commit/388f6a3f0528875531e57edffd350919ed1fc91f)）。

静态缓存改从真实prefill本地KV tensor推导尺寸；eager/SDPA使用additive四维mask，捕获step kernel保留boolean mask；重放前重置捕获期mask写入并补repetition penalty。整步argmax只用于单行、无惩罚greedy快路，避免批量请求误用；batch1仍每16次检查EOS，多余至多15步会裁去。

这是segment-KI改动后的静态图回归修复；作者在Intel Arc Pro B70/XPU、torch2.14dev、Transformers5.17做26项聚焦测试，本地没有CUDA/ROCm、多rank AutoTP及CUDA采样/惩罚端到端证明。惩罚历史每步cat仍为二次累计复杂度，未顺带解决性能。建议有静态图rollout的系统优先复现输出等价与cache形状，而非只跑启动。

### DeepSpeed：部分 offload 的优化器状态与 AutoSP 标签对齐（补录）

**版本与来源**：[#8973 · 2026-11-20补录](https://github.com/deepspeedai/DeepSpeed/pull/8973)（代码基线[25eaac4f2e](https://github.com/deepspeedai/DeepSpeed/commit/25eaac4f2e4df516ce138f74ca5a1a0464cb21de)）；[#8811 · 2026-11-20补录](https://github.com/deepspeedai/DeepSpeed/pull/8811)（代码基线[adcabcf95e](https://github.com/deepspeedai/DeepSpeed/commit/adcabcf95ea2a02bbe74ca63bfcb87a780ab8185)）。

#8973修复ZeRO3 offload_optimizer.ratio<1时device backup optimizer忽略LR/betas/eps/weight_decay调度、未存Adam状态、恢复后共享param-group字典导致CPU子组被多step等静默错误；Adam非W模式改用L2，bias_correction=false直接拒绝。作者示例中lr=0仍移动权重3e-3、一步恢复差4.9e-3；修后声称bit-identical并可跨ratio加载。SuperOffload其他路径另由#8972处理，不能归为全修复。

#8811针对AutoSP模型内部causal-label shift导致每个SP分片末token错配：labels保持完整复制，在全序列移位后按logits全局窗口取目标分片，不需要新增通信；支持的静态切片/reshape模式有明确限制，部分等价图重写仍可能绕过检测。建议优先检查partial-offload恢复和SP loss/梯度对照；属于正确性补录，非新训练算法。

### vLLM：16-bit GDN MTP decode 的 kernel 收益不能等同整机吞吐

**版本与来源**：[#69065 · 2026-11-21](https://github.com/vllm-project/vllm/pull/69065)（代码基线[78f20a5a34](https://github.com/vllm-project/vllm/commit/78f20a5a3454bcdcc0717cf63d26acf4075c1161)）。

新增BF16/FP16递归state专用融合CUDA多token预测（MTP）decode kernel，重排寄存器直接加载/归约以缓解warp shuffle瓶颈；FP32和小batch的V-split保留旧kernel，ROCm AITER仍不接FP16 state。作者GB300、H16/HV48、K=V128、MTP3、cold L2/CUDA graph下，4—128请求kernel约快1.33—1.49倍；240组输出/state逐位一致。

4×GB300、Qwen3.8-Flash-Next-FP8、DEP4+MegaMoE、FP16 state、MTP3、30720缓存前缀+1输入/1024输出、greedy下，作者比较**decode步时间（TPOT×接受长度）**，32请求约16.9→16.1ms，即约5%，单请求无收益。接受率跨轮变化，不能写端到端TPOT同幅下降；GSM8K1319题88.02→87.79%，不是实测完全不变。建议考虑复现本方state dtype和并发，先保证接受率/质量可比。

### vLLM：ZAYA1 CCA checkpoint 避免冷长提示额外 prefill

**版本与来源**：[#68921 · 2026-11-21](https://github.com/vllm-project/vllm/pull/68921)（代码基线[9920e7c7c8](https://github.com/vllm-project/vllm/commit/9920e7c7c8aabd90384f03ef75143cfac8068737)）。

CCA层原来不导出边界state，只有全部Mamba类层都可checkpoint时调度器才取消末尾切分，长提示会多一次prefill。现在保存所需原始q/k尾部与延迟value，在align模式声明alignment1，无checkpoint行写null block。作者ZAYA1-8B@67d34da515、TP1、64token block、prefix caching开启、8000输入、B300/MI355X同环境下，cold prefill由7936+64变8000一次，均值TTFT降15—38%/18—33%（并发1/4/8）。每请求一块checkpoint约占KV token容量0.386%；MI355X两臂均关闭有既有故障的AITER linear，不能泛化全部ROCm设置。建议考虑复现长prompt冷启动与缓存容量代价。

### vLLM：DiffusionGemma 混合 mask 与 Qwen 音频 encoder 图支持

**版本与来源**：[#62516 · 2026-11-21](https://github.com/vllm-project/vllm/pull/62516)（代码基线[bd53af0bbe](https://github.com/vllm-project/vllm/commit/bd53af0bbec2296e36f9413b69b63664839267e3)）；[#68831 · 2026-11-21](https://github.com/vllm-project/vllm/pull/68831)（代码基线[f8b29fab41](https://github.com/vllm-project/vllm/commit/f8b29fab414884206460fe63b983ca34fc6e3e9b)）。

#62516让DiffusionGemma的causal prompt/commit与bidirectional denoise混批走FlashInfer TRTLLM-Gen，SM100/103纯文本设为默认，依赖FlashInfer≥0.7.3；Hopper与多模态不变。1×B200、26B-A4B NVFP4、TP1、SPEED-Bench约1.2K入/3.1K出、并发1—8、32请求/档、三次均值、无prefix cache，对Triton BF16 KV：BF16 KV输出吞吐1.10—1.45倍，FP8 KV1.15—1.54倍；GPQA68.69/69.67对69.95，有质量差异，FP8还改变query精度。

#68831把Qwen3-Omni/ASR audio transformer/output MLP接到既有cudagraph_mm_encoder，卷积前端仍eager；仅FA/TRITON attention，默认encoder预算5120token，超出回eager；SDPA、FlashInfer及ROCm AITER没有图资格。没有在此引用完整同负载端到端表。建议按实际ASR长度和图覆盖率考虑复现，不能把所有audio backend写成已图化。

### vLLM：多租户服务的地址偏移、鉴权和输入预算修复集中合入

**版本与来源**：[#53131 · 2026-11-21](https://github.com/vllm-project/vllm/pull/53131)（代码基线[bafe5cc3e8](https://github.com/vllm-project/vllm/commit/bafe5cc3e803601b490ea6090c52b9b1e4e53497)）；[#52794 · 2026-11-21](https://github.com/vllm-project/vllm/pull/52794)（代码基线[3a1176475f](https://github.com/vllm-project/vllm/commit/3a1176475f25255923cbf881b277a9f676c762ca)）；[#54721 · 2026-11-21](https://github.com/vllm-project/vllm/pull/54721)（代码基线[0bf9c07376](https://github.com/vllm-project/vllm/commit/0bf9c07376a4091627e6cb50dd85464361db452c)）；[#53473 · 2026-11-21](https://github.com/vllm-project/vllm/pull/53473)（代码基线[468d10ae51](https://github.com/vllm-project/vllm/commit/468d10ae51ed432ff0612a7e3a57ec308439846c)）；[#68249 · 2026-11-21](https://github.com/vllm-project/vllm/pull/68249)（代码基线[6e0991c296](https://github.com/vllm-project/vllm/commit/6e0991c296f77de07d7d9007580544d91ae4d4ce)）。

#53131在LayerNorm/FP8 norm、MoE routing及AllSpark W8A16中先把token索引转int64再乘stride，修正32位偏移回绕；hidden8192时约524288tokens就触及32位范围，可能读写同batch其他用户状态，不能只按普通数值优化看待。#52794使root_path裁剪与Starlette按路径段边界一致，避免/v与/v1一类字符串前缀让鉴权和真实路由分歧。

#54721补齐stop IDs默认128上限/集合查询、stop strings与bad words前置校验，以及DeepSeek工具消息反向扫描；#53473把prompt-count限制扩到pooling/scoring/rerank/classify，IO processor仍自行校验插件payload；#68249把整批HTTP请求的下载/解码byte预算共享，防止逐子请求放大。证据以CPU回归和具体源码为主，部分端到端服务验证仍待CI，**未证明所有入口资源上界已完备**。建议维护对应版本多租户服务者优先核查回迁与自己的路由/批量接口回归，不摘取性能宣传。

### vLLM：batch-invariant norm 与 encoder-only 视频裁剪修复

**版本与来源**：[#68600 · 2026-11-21](https://github.com/vllm-project/vllm/pull/68600)（代码基线[ad9053380f](https://github.com/vllm-project/vllm/commit/ad9053380fe83234bcdc1f262abb7143c9cf0cfb)）；[#68314 · 2026-11-21](https://github.com/vllm-project/vllm/pull/68314)（代码基线[dd8c157895](https://github.com/vllm-project/vllm/commit/dd8c157895f01875bc5f44c51003bbe330647330)）。

#68600在VLLM_BATCH_INVARIANT=1时禁用手写FlashInfer all-reduce+RMSNorm/Gemma norm fusion，防止不同batch走不同算术路径。作者2×B200/TP2/BF16/hidden6144复现，原输出相对显式参考1193/6144元素不同；修后检查单行及更大batch两端逐位一致。测试环境FlashInfer0.7.2.post1与仓库pin不同，不能写所有依赖组合已验。

#68314对Qwen3-VL启用视频pruning的encoder-only/EC producer保留marker token embedding表，修复启动时宽度探测崩溃并与完整实例的裁剪视频embedding对照。实测RTX6000Ada48GB、2B-Instruct、pruning0.5；会保留额外embedding内存，未给量化内存/吞吐收益。建议分别按RL确定性需求和encoder拆分部署考虑采用。

### Inkling 2 Mini：vLLM 与 Transformers 的本轮集成状态

**版本与来源**：[#69028 · 2026-11-21](https://github.com/vllm-project/vllm/pull/69028)（代码基线[ab18334a4f](https://github.com/vllm-project/vllm/commit/ab18334a4f60da74db7de89fe95594e5ed0cda36)）；[#51050 · 2026-11-21](https://github.com/huggingface/transformers/pull/51050)（代码基线[046bdb8cc2](https://github.com/huggingface/transformers/commit/046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3)）。

本轮是已发布模型的框架增量。vLLM接入inkling2_loop和variable-loop研究checkpoint，循环每次访问保留独立KV/conv state，并接DSpark、图像输入与推理/tool parser；variable-loop覆盖仅研究权重，固定循环checkpoint拒绝任意改循环，DSpark也要求预期pass数。覆盖SM90/100/103/120/121为项目声明，不能省掉驱动/低精度kernel条件。GB200分支回归及GSM8K200题有披露，部分百万上下文/视觉评测来自早期pre-squash分支，不能当最终head全面重测。

Transformers #51050也已合入主线，提供loop模型实现和cache处理；本轮未确认包含它的正式release。vLLM另有B200驱动610.57与tile-8专家GEMM非法访问报告，620.70未复现，根因/修复状态仍待核；SM120低M优化的PR状态与清单冲突，不转述为已交付速度。建议按固定环境复现主线集成，模型本身的架构/数据不在此重复。

### SGLang：惩罚历史改由采样流拥有；block verify 的 exact row 仍待核

**版本与来源**：[#52035 · 2026-11-21](https://github.com/sgl-project/sglang/pull/52035)（代码基线[cee614b4f1](https://github.com/sgl-project/sglang/commit/cee614b4f16d84a0d05a3584d7b9ca73bdc99bf7)）；[#51961 · 2026-11-21](https://github.com/sgl-project/sglang/pull/51961)（代码基线[bae0ec95dd](https://github.com/sgl-project/sglang/commit/bae0ec95dd10b99dd099e3ba82bcd4af367a3259)）。

#52035把惩罚计数从scheduler维护的逐batch表，搬到请求级device池，采样后立即在采样流提交；重prefill/retract按历史回放，退出释放，无penalty请求不占表。它修复overlap下一步看不到上次token、min_new_tokens多阻挡EOS一步及跨流读写问题，也改为f×count并固定先additive再repetition，末位结果可变。**verify各row仍共享block起始惩罚；逐row exact是下一项#52054，状态冲突未当已交付**。DLLM、Unlimited-OCR复用窗、bound-runtime DFlash仍拒绝penalty。

#51961把block rejection的词表扫描拆为三个Triton tile kernel；只在FP32 p/q且请求数<128时走新路，其余保留classic。GB300有相同随机数下逐位输出、CUDA graph重放证据，后加127/128边界测试尚待GPU；9slots/163840vocab/b64在部分accept分布反而更慢，即便<128也非全胜。建议优先复现采样分布正确性与small-batch kernel，再按业务accept分布评估性能。

### SGLang：MXFP4 MoE workspace 预算与 K3.1 verify bucket 边界

**版本与来源**：[#51995 · 2026-11-21](https://github.com/sgl-project/sglang/pull/51995)（代码基线[3626134c72](https://github.com/sgl-project/sglang/commit/3626134c729430d784d579e4a54642f84de5a2ce)）；[#51983 · 2026-11-21](https://github.com/sgl-project/sglang/pull/51983)（代码基线[313429e551](https://github.com/sgl-project/sglang/commit/313429e551b99a9d9963207b93e5d75a434cf486)）。

#51995把SM100 TRTLLM-gen MXFP4 finalized MoE的scratch在载权重时预留共享buffer，使KV sizing看见它；此前约2.83GiB暂存到首次16K prefill才申请，可能只差12—33MB而OOM。只对最大prefill token预算内、finalized调用生效；deferred-finalize、prepared-routing、超预算forward仍动态分配，PD decode不预留。PR初始描述未GPU测量且最终影响未完整披露，不能写成显存下降，实际是**提早扣除KV可用空间**。

#51983让插件自行决定Kimi-K3.1 Q8 verify服务bucket，修复host硬编码1/2/4而拒绝5—8请求导致eager fallback。TP16/GB300的316.6ms是旧batch8回退基线，GPU修后速度与准确性仍标in progress，不能据7.65ms batch1推断41倍收益。建议有该配置的部署考虑复现；先验实际graph命中和总显存。

### SGLang：最短 prefill 优先增加负载感知边界与预算

**版本与来源**：[#51957 · 2026-11-21](https://github.com/sgl-project/sglang/pull/51957)（代码基线[ca9126e903](https://github.com/sgl-project/sglang/commit/ca9126e903a3ca587846e45a4972111b3e0aaeff)）。

对shortest-prefill-first新增可选模式：按等待及延续chunk的未缓存token backlog，指数收缩“短请求”类别阈值，重载时逐渐按到达顺序排长请求；空队列又缩小prefill batch token预算，以减少新短请求等待完整chunk。每个attention-DP rank用相同前缀匹配信息计算，TP应得一致顺序。开关默认0、原策略不变；单位测试覆盖预算传到admission与长请求阻断插队，**没有公开benchmark**。建议深入读策略与尾延迟公平性，再用混合长短trace复现，不能直接声称降低P99或解决所有饥饿。

### SGLang：Marlin MXFP8 与 JIT family target 是两种有边界的扩展

**版本与来源**：[#51936 · 2026-11-21](https://github.com/sgl-project/sglang/pull/51936)（代码基线[fe7159e66f](https://github.com/sgl-project/sglang/commit/fe7159e66f0ac15b75470b41709ec23fc67bd9a6)）；[#51929 · 2026-11-21](https://github.com/sgl-project/sglang/pull/51929)（代码基线[7915828551](https://github.com/sgl-project/sglang/commit/7915828551915de669b9586462da6988b1e61081)）。

#51936区分E4M3权重的E8M0 group32 scale与原16-bit folded scale，接Marlin MoE W8A16路径；新canonical MXFP8 fused路径用BF16激活，E8M0与16-bit scale不能因同weight dtype而混派kernel。SM100形状/路由精度测试及旧路径对照可读，SM90仍待对应CI，未benchmark，不把“支持”写成更快。

#51929让单个JIT模块可选specific/family，CC10.x及nvcc≥12.9可共享sm_100f缓存，减少同族设备重复编译；默认仍specific，**本PR没有任何模块默认切到family**，需要架构专属指令或静态共享内存者仍specific。CPU测试覆盖缓存键/编译flag，未实测冷启动收益。建议按异构Blackwell构建缓存需求考虑试验，不能全局强开后假定数值/性能等价。

### SGLang：Ulysses 超时失败归属与 IPC 池扩容代价

**版本与来源**：[#52030 · 2026-11-21](https://github.com/sgl-project/sglang/pull/52030)（代码基线[fa5f133b0b](https://github.com/sgl-project/sglang/commit/fa5f133b0b58ac981459972f4780d44a59baf7a6)）；[#52001 · 2026-11-22](https://github.com/sgl-project/sglang/pull/52001)（代码基线[c7290667d4](https://github.com/sgl-project/sglang/commit/c7290667d43c636063b68cace265066221a80858)）。

#52030在denoising/forward结束的所有rank统一检查timeout，让计算了不完整交换数据的**当前请求**失败，标health503；head-group pipeline加入有预算的wait kernel，避免原cuStreamWaitValue32无限等。它不继续用未知buffer走NCCL。但二rank旧transport仍在下一请求检查；永不返回的rank仍可让后续NCCL collective挂住，分组顺序路径又可能只报单output错误而不标整个服务坏。不能写成完整容错。

#52001保留仍容纳新shape的IPC池做first-sight逐位对照，扩容时加25%headroom，减少每种时长/提示形状重建和跨rank重新开IPC handle。8×B300、MiniMax/Pika、1344×768固定种子序列对照中，最终池每rank约746MB→932MB；保持池和reference同时存在时还需额外瞬态容量。收益是特定新shape/增长请求启动成本，非稳态算子提升。建议优先复现故障注入，再按时长分布与内存预算考虑采用。

### SGLang：MiniMax/Pika 多模态运行时的缓存与并行优化

**版本与来源**：[#51965 · 2026-11-21](https://github.com/sgl-project/sglang/pull/51965)（代码基线[301d90b3da](https://github.com/sgl-project/sglang/commit/301d90b3da8db7d5a522df95f3a4883d934b4a17)）；[#52051 · 2026-11-22](https://github.com/sgl-project/sglang/pull/52051)（代码基线[f2519b602d](https://github.com/sgl-project/sglang/commit/f2519b602d9a3a4b02c6a4de0935e88a7cfa4afd)）；[#51934 · 2026-11-21](https://github.com/sgl-project/sglang/pull/51934)（代码基线[90e21ed75d](https://github.com/sgl-project/sglang/commit/90e21ed75dc24fd59669c1d4ca011277a25b9ec1)）；[#51933 · 2026-11-20补录](https://github.com/sgl-project/sglang/pull/51933)（代码基线[0b86ed2239](https://github.com/sgl-project/sglang/commit/0b86ed22391ec6478138207c5033d98be4ec7467)）；[#51908 · 2026-11-20补录](https://github.com/sgl-project/sglang/pull/51908)（代码基线[2fa9203b32](https://github.com/sgl-project/sglang/commit/2fa9203b32059805e9de941c35c013e1db8f26c3)）。

#51965对相同timestep集合缓存MiniMax-H3 AdaLN投影，最多32项LRU，权重重绑/epoch变化失效；只适用resident CUDA、TP1，compile/图、Cache-DiT、layerwise offload等不走，不能与原本剪掉投影的online cache模式重复叠加。#52051在CUDA多decode rank把resident audio VAE放侧流，与其他rank视频tile重叠；ROCm为rank0音频优先、首组减负，一rank/音频offload保持串行。2×H200/2×MI350X多shape MP4逐字对照可读，二rank速度未正式A/B，单次约2.33→2.34s/4.50→4.45s不足以宣称收益。

#51934让ROCm BigVGAN卷积复制MIOpen既定求和次序，经每shape逐位门禁才替换；#51933对gfx950单live-sequence packed attention接AITER batch/MHA-v4，后者要求含#6841 race修复的版本，旧版保留v3、其他设备/边界保留varlen。#51908（11-20补录）让Qwen3 text encoder TP图在TP group内capture collective，并在销毁进程组前释放图；32提示Z-Image图像逐字一致。建议按具体配置复现，以上不是一个可相乘的统一加速倍数。

### SGLang：Hunyuan-4-Flash 视觉支持补齐（补录）

**版本与来源**：[#51937 · 2026-11-21补录](https://github.com/sgl-project/sglang/pull/51937)（代码基线[5a16e6be9c](https://github.com/sgl-project/sglang/commit/5a16e6be9c0700d53d75305fa4a60136048d22e9)）。

在已合入文本模型上加ViT/空间projector、图像视频processor、帧采样和interleaved MRoPE，Engram hash跳过媒体pad并恢复媒体ID。processor尚未随Transformers正式版交付，不能只装现有Transformers假定可用。作者Hy4-flash-FP8、B300/TP4、262144配置、128最大请求、sgl-eval0.1.3，MMMU-Pro前300题三轮均74.78%，与未拆stack均74.00%有随机波动；不是全MMMU结论。建议涉及Hy4视觉部署者考虑复现媒体token/位置与video行为。

### FlashAttention：MXFP8 PV 的 BF16 边界带修复因果量化语义

**版本与来源**：[#3022 · 2026-11-21](https://github.com/Dao-AILab/flash-attention/pull/3022)（代码基线[cc7b69d4b6](https://github.com/Dao-AILab/flash-attention/commit/cc7b69d4b67145f3822bd75344e3877eccd3c34e)）。

V沿key方向每32个元素共享量化scale；即使attention mask挡住未来token，量化边界块的scale仍可能由未来V决定，间接影响更早输出。实现对跨越causal边界的块使用原始BF16 V和BF16 P，完全可见的历史块仍用MXFP8 V/E4M3 P，共用输出累积；normalizer按实际P表示求和，LSE保留FP32指数和。B200/B300记录有future-V扰动不改变早期输出/LSE及persistent replay测试，参考自身就是混合精度recipe，**不等于相对完整BF16 attention无误差**。本项为主线机制增量，stack里的backward/更多normalizer选项仍各有状态，不能一并写为发布。建议深入阅读，复现低精度causal kernel时把未来输入扰动测试列为核心验收。

### NeMo RL：按环境配置 GRPO 与 rollout checkpoint 写流水

**版本与来源**：[#5321 · 2026-11-21](https://github.com/NVIDIA-NeMo/RL/pull/5321)（代码基线[33f611a543](https://github.com/NVIDIA-NeMo/RL/commit/33f611a5432d1cee8d6fc8efc50eab8ad4ab23f0)）；[#5176 · 2026-11-21](https://github.com/NVIDIA-NeMo/RL/pull/5176)（代码基线[9d9d34e9a8](https://github.com/NVIDIA-NeMo/RL/commit/9d9d34e9a85b71720b421ec796957d4c5395458b)）。

#5321在SingleController+token capture多环境路径加grpo.agent_overrides，按Gym agent或task_id.taskset逐行覆盖overlong_filtering与normalize_rewards，未设置继承全局；解决rubric连续奖励中0.80/0.85的小差被组标准差放大，以及不同环境截断策略不一致。该改动提供控制能力，**未证明哪种设置训练更好**。

#5176把prefix cut准备与TQ写入重叠，按行/token/byte界合并，TQ PUT在本节点额外1-Ray-CPU writer actor中进行；确认写成功后才发receipt，失败批次逐行重试一次，保留freeze/seal线程语义。4节点早期表25.24/72.27秒对52.44/107.60秒包含后来移除的direct-column encoder，**最终交付pipeline+writer+单encoder尚未重新benchmark**，不能当最终2.1倍。建议深入读失败与恢复契约，并考虑在真实checkpoint负载复现。

### Megatron-Bridge：量化 HF checkpoint 现在携带 KV scales

**版本与来源**：[#7057 · 2026-11-21](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/7057)（代码基线[726c31f8fc](https://github.com/NVIDIA-NeMo/Megatron-Bridge/commit/726c31f8fc0c3713128f24699a5a9aa499730c89)）。

save_hf_pretrained_modelopt不再直接拒绝KV quantizer，按QKV映射/TP规则输出k_scale/v_scale，query被量化时才输出q_scale，并写kv_cache_scheme。离线checkpoint行为与online refit不同：FP8 K/V scale至少1，不能把两种导出逐字段等同。MLA等无融合QKV映射仍退ModelOpt exporter，output quantizer仍拒绝。

2GPU功能测试涵盖Qwen3 FP8/TP2与Qwen3.5-VL mixed/PP2/ETP2，Qwen3与ModelOpt导出逐字对照；较早GB300/EP8/2节点Qwen3.6-35B NVFP4权重+FP8 KV示例4.5分钟及vLLM0.31加载正确，不是当前head端到端质量/性能复验。建议考虑复现量化checkpoint→部署闭环，先核每层scale和后端实际读取。

### LMCache CacheBlend：重复文档全部命中，再去掉重复 H2D（dev 分支）

**版本与来源**：[#6812 · 2026-11-20补录](https://github.com/LMCache/LMCache/pull/6812)（代码基线[52d149e706](https://github.com/LMCache/LMCache/commit/52d149e706a92a3f2a8a0b8c0e44ec630fcd14ee)）；[#6813 · 2026-11-21](https://github.com/LMCache/LMCache/pull/6813)（代码基线[80f3622f82](https://github.com/LMCache/LMCache/commit/80f3622f82dcbccc2ab9094a282da4cb265c9fdf)）。

#6812修复matcher只报stored chunk第一次位置：Agent对话第一次副本已被vLLM prefix cache吞掉时，后面副本本来应blend却重算。现返回全部非重叠匹配，key只读取/解锁一次；#6813每retrieve step同chunk只host→GPU一次，后续走device staging，且在re-RoPE改写前拷贝。

作者H200/gpt-oss-120b、v0.5.8rc1与固定plugin、10次中位数：A两次的H2D377→189MB，retrieve kernel6.96→3.67ms；A一次3.55→3.54ms。全部5120tokens加载且0 degraded，三副本测试逐位等价。只测retrieve，不代表端到端同倍；新planner使用host_offset=-1，旧cuda_ops会误当主机地址，**须一起升级二进制和Python planner**。两PR进dev，非宣称已进稳定发行。建议有重复文档Agent负载者考虑复现。

### VeOmni 与 TRL：非标准专家激活和 Mamba 树打包扩大范围

**版本与来源**：[#1480 · 2026-11-21](https://github.com/ByteDance-Seed/VeOmni/pull/1480)（代码基线[12acd6e326](https://github.com/ByteDance-Seed/VeOmni/commit/12acd6e326f375e1c9d5b81e3cdc1f75f7ea41f9)）；[#8457 · 2026-11-20补录](https://github.com/huggingface/trl/pull/8457)（代码基线[585e096200](https://github.com/huggingface/trl/commit/585e0962006fb884acc4b92140be929ec565f138)）。

VeOmni让MLP hidden_act传到routed experts，六条fused/eager路径统一EP dispatch；修复非SiLU在EP下旧eager全局expert ID索引越界。BF16 fused_quack/NPU/MLU不支持激活可退eager，FP8 DeepGEMM/QAT则必须拒绝以免偷偷变回全精度；不是每条kernel都支持所有activation。

TRL #8457（11-20补录）为AsyncGRPO加入Mamba2 tree conv/scan，避免兄弟分支串递归state；按模块函数family完整识别，函数重命名则初始化拒绝。Nemotron-H的MoE/MLP逐token块可接受，Falcon-H1/Zamba2的attention+Mamba同块hybrid仍拒绝。建议深入读分支state隔离与显式拒绝，按需要考虑复现；未从功能扩展推断训练质量/吞吐提升。

### Transformers：TP cache 内存估算与 no-grad 训练态缓存（补录）

**版本与来源**：[#48566 · 2026-11-20补录](https://github.com/huggingface/transformers/pull/48566)（代码基线[d0da3b84c6](https://github.com/huggingface/transformers/commit/d0da3b84c6b5851e7506fcde99d33e7fa793e1f7)）；[#50890 · 2026-11-20补录](https://github.com/huggingface/transformers/pull/50890)（代码基线[f2c9ee2401](https://github.com/huggingface/transformers/commit/f2c9ee2401bce932d38d48eafde581a65af7fcc9)）。

#48566在KV头确实按TP分片时，将PagedAttentionMemoryHandler的query/newKV/attention peak估算也改为本rank头数，避免每batch token预留TP倍内存、无谓缩小cache pool；KV未分片时仍size1。PR早期吞吐记录缺完整当前配置/硬件，因此本轮只采纳机制与分片估算对照测试，不复述6523→11277tok/s为通用结论。

#50890仅在grad enabled时启用activation checkpoint：训练态但no_grad生成应保留cache/use_cache，修复ZeroSyncGRPOTrainer连续批处理在cache参数迁移后被置空而失败。用train+checkpoint+no_grad四token前向回归，未给实际RL训练收益。建议使用Transformers原生连续批处理/训练中rollout者考虑升级并验证API组合；StaticCache混dtype与反量化save两个PR仍有状态冲突，单独留查。

### vLLM-Omni：Ming 与 Phonon 的运行时集成明确限定设备/并发

**版本与来源**：[#9970 · 2026-11-21](https://github.com/vllm-project/vllm-omni/pull/9970)（代码基线[3789f7a230](https://github.com/vllm-project/vllm-omni/commit/3789f7a2305dc3e152cbe9073c2267af9dfc85ce)）；[#9995 · 2026-11-21](https://github.com/vllm-project/vllm-omni/pull/9995)（代码基线[9c82e5bdf3](https://github.com/vllm-project/vllm-omni/commit/9c82e5bdf310af494e18242f55b0f4653d975b5a)）。

Ming dense0.5B与MoE16.8B-A3B的两阶段在CUDA默认MRv2，补req_id识别、防止一直生成到max_tokens、MoE图捕获返回包装，以及abort后stage1 state清理；ROCm/NPU/XPU/MUSA未资格化仍V1，V1可显式选择。迁移本身含正确性修复，不能用一个model_runner字段代表全部模型已兼容。

Phonon31M EN是单阶段MRv2初始集成，paged AR attention+flow采样/Mimi解码，slot池持有音频状态、环形KV和history；当前仅B1/TP1/default pretrained voice，不是任意voice或高并发已支持。建议考虑复现stream取消/slot重用与首音频时延，性能与音质尚需业务级对照。

### vLLM-Omni：NPU 专用图池与 XPU 原生 MXFP8 修复

**版本与来源**：[#8312 · 2026-11-21](https://github.com/vllm-project/vllm-omni/pull/8312)（代码基线[673daeada7](https://github.com/vllm-project/vllm-omni/commit/673daeada725f325fb488f9582c035411d60ef71)）；[#10278 · 2026-11-21](https://github.com/vllm-project/vllm-omni/pull/10278)（代码基线[dc8214f410](https://github.com/vllm-project/vllm-omni/commit/dc8214f410b32bcdef699b39572681425a312b92)）。

VoxCPM2 AudioVAE为每实例使用专用NPU graph pool，缓存重复(batch,seq_len)图；旧全局共享池超过8图曾出现CANN分配器冲突/尾音截断。交付max_graphs32，作者生产观察每天13—17shape、graph池<1GB/总HBM约15.6GB于64GB设备、utilization0.7；不能把该观察推广全部CANN/并发，也未在此摘录通用加速比。

HunyuanImage3 AR+DiT/XPU改正online MXFP8 MoE构造的keyword-only参数，SigLIP2的4304维不满足group32时保持非量化，并仅startup LoRA才建MoE LoRA bridge，以避免native XPU MoE被迫EMULATION。代价是启动后再添加MoE LoRA不应用专家。建议按现有NPU图故障或XPU量化需求考虑复现，特别验收长音频尾部与LoRA生效范围。

### vLLM-Omni：停机、取消对话边界与请求 token 预算

**版本与来源**：[#10808 · 2026-11-21](https://github.com/vllm-project/vllm-omni/pull/10808)（代码基线[0f363a5ccf](https://github.com/vllm-project/vllm-omni/commit/0f363a5ccfd348f28f0472dab978d061606439ec)）；[#10890 · 2026-11-21](https://github.com/vllm-project/vllm-omni/pull/10890)（代码基线[6ce3e37a52](https://github.com/vllm-project/vllm-omni/commit/6ce3e37a52771195868bb91cfe4a6a9354c98c16)）；[#10812 · 2026-11-21](https://github.com/vllm-project/vllm-omni/pull/10812)（代码基线[9e50e66287](https://github.com/vllm-project/vllm-omni/commit/9e50e66287481a9fcfcaeb1996e4a7b0062565fa)）。

#10808在serving loop同一callback关闭注册、给所有在途请求排终态错误再停output pump，阻止停机快照与新generate竞争造成永久等待；CPU race barrier回归可读，未泛化全部多进程停机。#10890保留JoyAI取消语音后仍被视觉观察引用的空assistant边界，防止新图/新问紧接旧观察而错序；有真实prompt replay由旧图答案改为新图答案的证据，但不等于整体对话质量评估。

#10812使Step-Audio2 speech endpoint的max_new_tokens传到stage0，遗漏时保留默认，不改后续stage/停机条件；CPU回归原显式7/8192都传256，修后正确。没有同时解决截断上报或所有TTS模型，CoVo对应PR仍Open/状态冲突。建议相关服务优先做取消后下一轮、SIGTERM与请求budget边界验收。

### vLLM Rubin：cu135 迁移同时记录 SM107 的错误结果 workaround

**版本与来源**：[#68036 · 2026-11-21](https://github.com/vllm-project/vllm/pull/68036)（代码基线[25d7765e63](https://github.com/vllm-project/vllm/commit/25d7765e632beca1edd0ce3e6a6dc8186e9d9bb3)）。

Rubin prerelease镜像把PyTorch cu134 nightly与旧builder迁到cu135，pin torch2.16dev20261116，Triton改用其要求的3.9.0 wheel；同日torchvision/torchaudio仍无cu135，使用同日cu134 nightlies。技术关键是ptxas13.5.37被报告会把部分SM107a Triton kernel静默错编译，GLM-5.3输出乱码；文档提供显式docker选项，让Triton使用wheel携带的ptxas-blackwell13.4。

**镜像本身没有自动切换Triton实际使用的ptxas**，不能把装新wheel写成错误输出已自动修好；这是Rubin预发布路径，非所有GPU/CUDA的普遍问题。建议使用SM107测试环境者优先核查compiler实际路径，并运行输出对照，不能只验pip依赖无冲突。

### DeepSpeed AutoEP：跨 microbatch 合并专家 dW，用暂存显存换大 GEMM（补录）

**版本与来源**：[#9114 · 2026-11-20补录](https://github.com/deepspeedai/DeepSpeed/pull/9114)（代码基线[d70adc325c](https://github.com/deepspeedai/DeepSpeed/commit/d70adc325caa4c5d4e3fcebe19abbdbe41638a2e)）。

新增可选deferred_dw_flush_period与每expert行容量，将多个backward的专家输入/梯度行暂存，周期或累积窗口末尾集中做权重梯度GEMM；溢出可先算但延迟交付到统一backward，空expert也交零梯度，避免复制expert在不同rank提前flush导致ZeRO2 collective错位挂住。默认period1关闭，拒绝非托管累积、ZenFlow、DeepCompile、AutoTP folding和ZeRO3；DeepEP staging/offload/4rank EP2未完整实测。

作者H200/BF16、module级fwd+bwd对照，在小row/eight-microbatch可达约1.6—1.9倍，而512rows/expert的部分形状退到0.66—0.90倍；128experts/hidden2048/intermediate768、period8的暂存每层每rank随8—512rows需0.20—12.5GiB，溢出还额外留一份dW。**这些计时早于最后collective修复且未重测，不是最终端到端训练提速**。建议深入读flush窗口和复制expert条件，仅在小专家GEMM/足够显存时考虑复现。

### LMCache：Device-DAX 共享 L1 的生命周期与可见性前置落地（dev，补录）

**版本与来源**：[#6121 · 2026-11-20补录](https://github.com/LMCache/LMCache/pull/6121)（代码基线[831e050281](https://github.com/LMCache/LMCache/commit/831e0502818ec6e5a444e03a3e57c9ae72806a9d)）。

加入CLFLUSHOPT/SFENCE显式cache可见性、GPU注册SharedDevDaxRegion、已发布对象删除与store失败后的保留/中止契约。committed对象可全局失效但现有lease仍可释放，extent暂不回收；关闭时导出tensor未释放则保留mmap/fd以供重试，不能在仍被设备使用时假定已释放。无硬件coherence的共享映射需要显式flush。

**这只是共享Device-DAX三项中的前置部分：DevDaxL1Manager仍拒绝shared配置，SHADOW单独不提供peer对象发现或共享服务，borrowing adapter待后续**。当前head有两host各RTX PRO6000Blackwell的功能证据，但不能写已交付跨主机共享KV系统或性能收益。建议跟踪协调PR，并研究lease、GPU复制drain与物理extent回收边界。

### TRL：文本异步蒸馏冻结视觉模块，避免重复同步未训练权重（补录）

**版本与来源**：[#8412 · 2026-11-18补录](https://github.com/huggingface/trl/pull/8412)（代码基线[713aeffe60](https://github.com/huggingface/trl/commit/713aeffe60dd6c392e1ba6fde81d22b63f6275bd)）。

AsyncDistillationTrainer对VLM student先冻结全模型，再启用文本主干与输出embedding；原先纯文本蒸馏没有视觉梯度，但参数仍标trainable，vLLM sync按是否冻结筛选，导致每轮反复发送未变化视觉权重。tiny Qwen3-VL集成验证视觉冻结且数值不变、其他参数更新，要求Transformers≥4.57。没有同步字节/端到端吞吐表，不能量化节省；这也不适合要学习视觉塔的多模态蒸馏。建议纯文本异步蒸馏VLM者考虑复现。Phi4/DeepSeekV3.1 assistant-only模板两项仍单PR Open，与分支清单不一致，留查不写已发布。

<a id="topic-6"></a>
## 硬件与软件协同

### OpenAI / AMD 芯片设计研究（11月18日，补录）：RTL优化已进生产流程，公开RISC-V评测仍有条件差距

11月18日官方正文，本轮补录。OpenAI称模型改写RTL并用Synopsys检查行为等价与PPA（功耗、性能、面积）；Jalapeño向量/矩阵单元标准单元面积约减13%/10%，已硅验证；AMD GPU IP最高14%PPA改善，有改动进入已送制造设计。**未披露IP、指标权重与同条件全芯片测量，不能理解为LLM吞吐或能效提高14%。**

另两个32位RISC-V核的9,881次评测用仿真正确性、隐藏工作负载、Yosys频率估计，失败或面积超过原核16倍记零；约112K tokens同预算下新模型更好，从规格重写又优于改旧RTL。作者承认前者可省性能计数器和额外指令；面积预算宽松且非布局布线后时序，不能视为生产芯片设计能力等价证据。**建议深入读评测边界，继续观察**代码、测试集与物理实现复现条件；GPT-Synopsys属后续训练合作，尚非可用发布。[官方技术正文](https://openai.com/index/advancing-chip-design-with-our-models/)。

### Primus MegaUlysses：投影与全交换重叠，先检查适用域和权重缓存依赖

AMD把Primus-Turbo的融合GEMM＋通信接入Megatron Ulysses上下文并行：GQA/MHA把QKV与输出投影分别和全交换重叠，MLA则先聚合压缩latent再做局部head上投影。序列由zigzag切分改连续切分，保留完整RoPE频率。默认关闭，要求TP=1、CP组在单节点；packed序列、非因果mask和推理等调用不支持，不能自动回退。MLA必须有Turbo #678及#679权重缓存修复，否则可能复用首步kv_b权重；缺算子时告警回退。

作者单节点8×MI355X、BF16、MBS1/GBS16、mock数据12步，以第4—12步中位数比较同配置stock CP：完整48层Qwen3-30B-A3B在32K为6433→5831ms、64K16914→15654ms，按stock/new−1为10.3%/8.0%吞吐改善。其余大模型多为4层代理，最大已分配内存约−0.5%～+0.8%；与同attention串行通信路径的损失接近，但stock切分与attention kernel也不同，Kimi损失最大差0.5%。**建议深入读并定向复现**长序列单机CP，先固定Turbo依赖和loss对照，不能外推跨机或完整大模型。

**版本与来源**：默认分支提交UTC日期 2026-11-22；具体SHA见[Primus abbbcb8d](https://github.com/AMD-AGI/Primus/commit/abbbcb8dc9f0a7276dea339a4a989971eac0303e)。除明确撤回项外为源码已合入，未据此承诺正式发行包包含。

### Primus：日志显示安装，不代表FSDP2真的用了SDMA

补丁修复已经按值导入的fully_shard别名未被替换，使TorchTitan的AllGather缓冲没有注册为RCCL窗口；并对跨机NCCL_CROSS_NIC=0导致退回计算核给出告警。Flux改用同rank顺序、nccl-lazy创建的twin group以供copy engine通信，跨机HIP IPC copy明确拒绝。作者8×MI355X、4层Llama3.1-70B、40步中，被CE拒绝的AllGather由444降至4、末步loss同3.507；具体最终分支和Flux完整MLPerf仍有待验项，提交头称4节点测试良好但未给完整对照，不合并为全面验证。**建议核对通信日志并考虑采用**；收益首先是实际路径生效，所读证据没有完整训练加速率。

**版本与来源**：默认分支提交UTC日期 2026-11-21；具体SHA见[Primus ad2fd2b6](https://github.com/AMD-AGI/Primus/commit/ad2fd2b6150b61afdce6deed5fb799ca8f35a23f)。除明确撤回项外为源码已合入，未据此承诺正式发行包包含。

### vLLM TorchTPU：prefill路由表由接收端排序改发送端排序＋成段复制

发送端按expert稳定排序自身路由pair并携带token号和scale位，接收端一次run copy构造gather表和scale slab；总路由pair数T×topk≥160×expert总数（平均门槛）且VMEM预算满足时启用，小decode保留原路径。作者32×TPU8i、Qwen3.8-2.4T-A95B-FP8、DP16×TP2/EP32、1488并发、8192输入/1024输出：吞吐45104→47911token/s（+6.2%），混合prefill/decode步286.2→263.8ms，decode82.0→81.8ms。两组是同一个11月9日main派生开发基线、同时有#1585及其他服务改动，**不能视为当前干净main的同幅提升**；仅rank0性能分析，慢rank缩短是作者推断。20个greedy输出/logprob相同及GSM8K对照有限。**建议考虑复现**真实路由偏斜和通信额外scale载荷。[完整实验与限制](https://github.com/vllm-project/vllm-torchtpu/pull/3388)。

**版本与来源**：默认分支提交UTC日期 2026-11-21；具体SHA见[vllm-torchtpu e132eda8](https://github.com/vllm-project/vllm-torchtpu/commit/e132eda819d67617c46a5e93d233806dbe3345a8)。除明确撤回项外为源码已合入，未据此承诺正式发行包包含。

### TorchTPU：SDPA按VMEM与填充代价路由，小head布局阻断向上游传播

先按score在片上向量内存（VMEM）的实际占用和128-lane填充成本决定StableHLO/FlashAttention；后续同日补丁让所有单split tile（≤256tokens）、每GQA组live score可放入VMEM的推理形状优先StableHLO。在禁用XLA HLO FlashAttention rewrite的条件下，TPU v6e、BF16、H8、D128/T128/B3264对旧Flash路径14.88→2.93ms，**对23.61ms等更老基线的高倍数包含前序优化**；首补丁331模型套件step几何均值反而+0.54%，未见最终组合全套复测。小head的另一补丁在FA边界加sequence-minor布局约束，避免QKV/RoPE低lane利用率，但会增加至多三次边界relayout；作者Llama3.2-1B eager约6%—7%改善、未给该数完整硬件配置。**建议按实际shape复现**，不把微基准倍数当整网收益。

**版本与来源**：默认分支提交UTC日期 2026-11-21；具体SHA见[torch_tpu a14816af](https://github.com/google-pytorch/torch_tpu/commit/a14816afbb34f53a3b43767c541594f30d302a44)；[torch_tpu 40840059](https://github.com/google-pytorch/torch_tpu/commit/4084005932b97cef704b5913ae571eacce4e5255)；[torch_tpu 6d86b584](https://github.com/google-pytorch/torch_tpu/commit/6d86b5845a1a75e4f2027eaba58d695cad950c27)。除明确撤回项外为源码已合入，未据此承诺正式发行包包含。

### TorchTPU：动态var/std的分母与strided view分解修正

var/std/var_mean由host静态或上界N改运行时元素数，并修FP16计算及correction精度；strided view只在stride小于后续维实际跨度时视为重叠，避免错误unfold逆变换和空slice改写导致进程abort。作者v5e-1、torch2.11/libtpu0.0.50与torch2.15/libtpu0.0.51对照：98个动态归约case修前44失败、修后匹配CPU；每版本56个view复现修前29 abort。complex128动态和旧libtpu复制问题仍未解决。**建议优先核对并回归**动态训练统计和非连续缓存视图，不以这些局部修复声明后端完全等价。

**版本与来源**：默认分支提交UTC日期 2026-11-21；具体SHA见[torch_tpu a2161629](https://github.com/google-pytorch/torch_tpu/commit/a2161629e4f1f9d03897d1e7bc0953a9d973bc92)；[torch_tpu 7957a6eb](https://github.com/google-pytorch/torch_tpu/commit/7957a6eb2d8f0c728039af4ea5bc708e57e6a66c)。除明确撤回项外为源码已合入，未据此承诺正式发行包包含。

### vLLM TorchTPU：ModelOpt逐层混精度和Mamba分块SSD算子

混精度checkpoint按quantized_layers逐层选择已有TPU FP8/NVFP4实现，补hf_quant_config.json回退和融合投影scale加载。NVFP4与W4A16_NVFP4默认均以FP4权重＋BF16激活运行，**不是照搬GPU的FP4激活路径**。Mamba另新增Pallas chunk_state/chunk_scan两阶段、按VMEM预算分head、FP32高精度状态累加，含分组B/C投影与Mamba3 SISO/MIMO形状；只是SSD流水线中的两个阶段，不等于全Mamba3服务已交付。两者无完整模型质量/吞吐新证据。**建议深入读后定向复现**混合权重映射与长序列状态。

**版本与来源**：默认分支提交UTC日期 2026-11-21；具体SHA见[vllm-torchtpu d35dd0d2](https://github.com/vllm-project/vllm-torchtpu/commit/d35dd0d292d19fee88e67d8a81334f60ff27a586)；[vllm-torchtpu 52883535](https://github.com/vllm-project/vllm-torchtpu/commit/52883535f274db0a6c8b21e10445526716b5bfeb)。除明确撤回项外为源码已合入，未据此承诺正式发行包包含。

### vLLM TorchTPU：并行启动先解决spawn管道背压与rank环境隔离

原约75KB配置超默认64KiB pipe，父进程等待子进程完成导入才继续，8worker启动约每6秒串行一次。新实现将多worker spawn管道扩至最多1MiB/系统上限，只把写rank环境和start放在锁内并恢复父环境，使子进程并行导入。载荷仍超上限会再次串行，不能扩容时告警，DP engine core不覆盖；每worker pipe容量仍计入用户额度。**建议考虑采用并复测冷启动**，这是启动开销优化，无稳态吞吐证明。

**版本与来源**：默认分支提交UTC日期 2026-11-21；具体SHA见[vllm-torchtpu c676da07](https://github.com/vllm-project/vllm-torchtpu/commit/c676da0751d0bb568a2a8ac786e7224e07b81cba)。除明确撤回项外为源码已合入，未据此承诺正式发行包包含。

### SGLang TorchTPU：Kimi稀疏DCP三项同日撤回，不能按已支持采用

明确revert移除#2239稀疏MLA解码上下文并行、#2240请求handle scratch所有权/可选DMA all-gather及服务修复、#2352 DFlash草稿回MHA，理由为未经review合入。固定revert删41文件中的约2837行；先前合入记录只作为状态链保留，**本轮截止以撤回后状态为准**，不将其描述为已发布能力，也不推定撤回原因是性能或正确性失败。**继续观察**重新审核与重合入。

**版本与来源**：默认分支提交UTC日期 2026-11-21；具体SHA见[sglang-torchtpu 9981cfc8](https://github.com/sgl-project/sglang-torchtpu/commit/9981cfc8bdc7a9789c43bdd9444d19e1e2491615)；[sglang-torchtpu 25ffb64b](https://github.com/sgl-project/sglang-torchtpu/commit/25ffb64b35c5468bcfac7fad9d3587de3f9b129e)；[sglang-torchtpu 7eda979d](https://github.com/sgl-project/sglang-torchtpu/commit/7eda979df43107010a5e34a2b56561527583bdc8)；[sglang-torchtpu 5b0f0588](https://github.com/sgl-project/sglang-torchtpu/commit/5b0f05881d874eefbf2390f586f38ddb3d10e61f)。除明确撤回项外为源码已合入，未据此承诺正式发行包包含。

### SGLang TorchTPU：DSA原始key按页写，最终门禁是周期成本模型

把对全raw cache逐token scatter改成小页缓冲内写后整页回填，最终实现以TPU7x的周期估计选择路径；不能沿用提交早期说明的固定1024tokens门槛，最终测试含256/512tokens命中新路径。GLM-5.3-Flash、TPU7x、TP8/EP8、8K输入/1K输出：raw-key写11.82→1.00ms/pass，但c1首token296.3→285.9ms（−3.5%）、c64吞吐1773→1800token/s（+1.5%），decode14.99ms不变；GSM8K90.9→90.8。缓存字节等价测试覆盖跨页/多请求/padding。**建议考虑复现**具体cache规模与门禁，不能把写核11倍直接换算服务收益。

**版本与来源**：默认分支提交UTC日期 2026-11-22；具体SHA见[sglang-torchtpu 7d182792](https://github.com/sgl-project/sglang-torchtpu/commit/7d182792043711595974f285f0b9649d5b926953)。除明确撤回项外为源码已合入，未据此承诺正式发行包包含。

### SGLang TorchTPU：会话保护应区分host与device缓存，隐式session须管理关闭

可从HTTP header填充session_id，显式body/session_params优先；增加prompt末尾注册叶子、被驱逐叶释放覆盖及仅host按会话保护的选项，均默认关闭。目的是避免重渲染assistant文本使每轮分叉积累叶子，以及空闲会话占device滑窗池挤走活跃请求。隐式session不调用close仍可能增长，日志只提高可见性；最终补丁有有界池驱逐、分支编辑及两端点测试，无真实多租户吞吐结果。**建议深入读并定向复现**会话生命周期、host回载及并发内存占用。

**版本与来源**：默认分支提交UTC日期 2026-11-22；具体SHA见[sglang-torchtpu 4bd8c442](https://github.com/sgl-project/sglang-torchtpu/commit/4bd8c442c1c438503857bc70ae470d33ff38d7ef)。除明确撤回项外为源码已合入，未据此承诺正式发行包包含。

### SGLang TorchTPU：cache-only DSA写入仍未接服务；DeepSeek-V4.1启动兼容已修

DSA新op只写latent/旋转/索引或pooled缓存，复用完整attention/indexer的writer，逐字节比较且拒绝CP等不同语义路径；提交明确尚未接入serving，后续供chain speculative worker使用。另一提交修上游Engram hash-column分片后TPU仍均分行、误调用CUDA设备探针及out-of-tree KV pool被提前拒绝三个启动阻塞。**cache-only继续观察接线，启动修复考虑采用并回归加载**；都没有可比较整网性能证据。

**版本与来源**：默认分支提交UTC日期 2026-11-21；具体SHA见[sglang-torchtpu 7ac84f45](https://github.com/sgl-project/sglang-torchtpu/commit/7ac84f4554eb35b6914e514afcf932e5f76da5aa)；[sglang-torchtpu ce81c35f](https://github.com/sgl-project/sglang-torchtpu/commit/ce81c35fe37bb6e58ae703122456020208a8ddff)。除明确撤回项外为源码已合入，未据此承诺正式发行包包含。

### vLLM Ascend：PCP与DCP组合进Kimi DSpark，容量收益与联合优化收益要分开

prefill只聚合引用的DCP主缓存/索引页、Q按本地行计算；没有Q行的rank仍参与集合通信。草稿复用物理PCP/DCP组，但复制Q视图暴露PCP1；host长度取拒绝采样修正后的真实device长度，使用blocking copy。作者64张A5、32P＋32D、完整896expert Kimi-K3.1-Pro W4A8 MXFP、PP4/PCP8、约131160输入/1输出、并发12：未profile轮首token6.576→6.232秒、QPS1.753→1.853。**优化组同时改DCP1→8、开启MegaMoE并加#22656预算补丁，不能将5.7%全归本提交**；两组前缀命中89.09%，不是全冷128K。另一完整部署DCP1容量不足的事实也不能与这张可运行基线混淆。同份补丁后半另有前缀缓存关闭、同payload的三臂单轮历史修订实验：main DCP1、PR DCP1、PR DCP8；约16K/1并发8及128K/1并发2中，DCP8对main输入吞吐−1.38%/−1.59%、server TTFT+3.46%/+1.06%，逻辑KV容量1,140,192→5,296,666 tokens（4.65倍）。该容量不等于HBM物理节省或可持续并发；GPQA198题185对187仅单次随机结果，最终修订三轮回归与完整rank HBM采样未完成。**建议深入读后复现容量—性能—质量权衡**，当前证据不能证明DCP8普遍加速。

**版本与来源**：默认分支提交UTC日期 2026-11-21；具体SHA见[vllm-ascend dcf0f96c](https://github.com/vllm-project/vllm-ascend/commit/dcf0f96c43ece47edabcf11d859a74ba390a5a2d)。除明确撤回项外为源码已合入，未据此承诺正式发行包包含。

### vLLM Ascend：草稿配置隔离与All2AllV拆分D2H重叠

draft attention浅拷贝上游配置再改parallel_config，防覆盖target的PP/EP；作者vLLM0.33、A3、EAGLE3/PP4/EP首prefill挂起，同修复请求14.60秒HTTP200，但只是一条定向复现。All2AllV另复用pinned split缓冲、一个event保护D2H、先排token permute后等待，最终numpy metadata独立拷贝以免复用竞态。A3两卡、CANN9.1、Qwen3.8-Flash-Next W4A8、TP2/EP/eager、8192输入/1输出/并发1共16请求：host dispatch均值−7.9%，首token仅−0.63%，16题GSM8K相同。**建议优先采用配置修复并定向复测同步优化**，小样本不能证明所有模型/负载数值等价。

**版本与来源**：默认分支提交UTC日期 2026-11-21；具体SHA见[vllm-ascend c64ace67](https://github.com/vllm-project/vllm-ascend/commit/c64ace678d12d5a48671d929c76f723f8158876c)；[vllm-ascend 2dad709b](https://github.com/vllm-project/vllm-ascend/commit/2dad709bfa86ea7f4a3ddb050ee83adc7c2442a4)。除明确撤回项外为源码已合入，未据此承诺正式发行包包含。

### AITER：mHC用Gram统计减少归一化访存，保留消去情形回退

gfx950/256CU、连续对齐BF16、fused RMSNorm与20轮Sinkhorn的decode新路径由producer生成舍入后残差及四流Gram矩阵，finish用混合系数求折叠范数；强抵消/非有限时改从真实折叠值求范数，并固定FMA顺序。最终门禁表经MI350X三类计时筛选，只覆盖隐藏4096/7168及特定行区间；prefill另为packed权重2049—65536行保留融合GEMM，作者36行数点1.07—1.20倍算子收益。取消流测试存在约4.17%元素超过既有容差的难例，两候选相同，不能说绝对零误差。**建议深入读后复现**数值消去与整网稳定性。

**版本与来源**：默认分支提交UTC日期 2026-11-21；具体SHA见[aiter 9f3cba25](https://github.com/ROCm/aiter/commit/9f3cba25ee1117e8152112e46b8984d94c90b676)。除明确撤回项外为源码已合入，未据此承诺正式发行包包含。

### AITER：K平滑门控融合与MLA无位置编码写缓存接口

MHA-v4把约10个Torch门控操作收成一次host调用/两个确定性HIP核，运行在K的设备并先Python校验，避免跨卡stream错误或非法headdim触发进程abort。gfx950 eager、MXFP4、head_dim256/H16/S4096调用210.6→135.2µs，长64K改进较小。MLA另一提交在gfx1250/gfx950优化参数预载、Q load顺序与head分组，并提供独立NoPE Q/K concat＋FP8写缓存入口，避免虚构identity RoPE缓存；要求单GPU、连续int64 slot、FP32 scales等。**建议考虑复现**真实短decode；不同阶段微基准不能相加，NoPE入口不能用于真实RoPE模型。

**版本与来源**：默认分支提交UTC日期 2026-11-21、2026-11-22；具体SHA见[aiter e7728184](https://github.com/ROCm/aiter/commit/e7728184c6ae0775f8164c9eb921641aa431b24d)；[aiter 09497204](https://github.com/ROCm/aiter/commit/09497204a230a265b419f27965563ed5325c9562)。除明确撤回项外为源码已合入，未据此承诺正式发行包包含。

### AITER：gfx942大head共享内存、gfx1250-strict编译及非法expert保护

gfx942 BF16 head257—511补3D split-KV tile32，修head320填充512后需128KiB却只有64KiB LDS的编译失败；48shape仅编译对照，不承诺性能提升。gfx1250-strict避免借用不适用OPUS配置，稀疏prefill ASM按family能力门禁、MoRI JIT按运行时架构定标，不能把A0/B0泛作同一编译目标。机内FlyDSL EP另过滤负值和越界expert id，走现有drop/sentinel路径并加入多卡测试。**建议核对硬件代际后采用并回归**padding、masked routing与fallback；未给整网质量/吞吐。

**版本与来源**：默认分支提交UTC日期 2026-11-21；具体SHA见[aiter e9316997](https://github.com/ROCm/aiter/commit/e9316997c70800518f048f57f62f868f4cdd1cf2)；[aiter 3d2db502](https://github.com/ROCm/aiter/commit/3d2db5027e0c0e66498ec0eddad53303ce557cbe)；[aiter fb6c1706](https://github.com/ROCm/aiter/commit/fb6c17066ef497c2eaae1aa799a8d49d99b93d4a)；[aiter 810f722c](https://github.com/ROCm/aiter/commit/810f722c2182eca11bcd7191052152cb3a6a82fb)；[aiter fc623e74](https://github.com/ROCm/aiter/commit/fc623e74580fefddb7f6e20b4ab3cbb7ca7eb05e)。除明确撤回项外为源码已合入，未据此承诺正式发行包包含。

### ROCm hipBLASLt：在线调优由实际调用分摊搜索，图捕获不参与探索

HIPBLASLT_TUNING_MODE=online对同key的实际matmul调用分摊K候选×R次采样，用后续调用读取event，不等待GPU；默认K5/R3、冷调用3、event pair最多4096，winner仍写缓存供后续重用。只调null algo的hipblasLtMatmul，显式algo、扩展/分组GEMM、rocRoller、pointer-array、capture等均不探索；不能据功能名认为所有LLM GEMM都在线调优。另一修复清理tune scratch分配失败自身留下的HIP OOM状态，避免成功matmul被数值检查报INTERNAL_ERROR。**建议深入读后考虑在稳定shape服务复现**首次探索代价、噪声与最终收益，未见整网吞吐证明。

**版本与来源**：默认分支提交UTC日期 2026-11-21；具体SHA见[rocm-libraries 00243b01](https://github.com/ROCm/rocm-libraries/commit/00243b012cf8ef44ff1217b019d044dad14d5743)；[rocm-libraries 00611734](https://github.com/ROCm/rocm-libraries/commit/006117343a723455e2a774b0e3e71b0e9d3c30ed)。除明确撤回项外为源码已合入，未据此承诺正式发行包包含。

### ROCm：Tensile构建压缩父进程对象，geko按收敛自适应测量

TensileCreateLibrary在父进程只留压缩solution和必要摘要，worker按需hydrate；96核/1.9TB主机、MI300X/ROCm7.2.4、843logic/356190solution/214059kernel、96jobs全gfx942构建：进程树峰值PSS91.0→81.3GiB、1778→1559秒；这是构建内存与耗时，非运行时模型显存。完整单测仍有607失败/4错误，与develop相同，不能写全套绿。geko从固定迭代改按均值相对标准误/稳健离散收敛或上限停止，balanced先快筛再每GEMM重验50候选；调度chunk默认值含模拟推导，不能当硬件实测通用最优。**建议考虑用于构建/调优流程复现**并保留测量噪声。

**版本与来源**：默认分支提交UTC日期 2026-11-21；具体SHA见[rocm-libraries 2f4a2fff](https://github.com/ROCm/rocm-libraries/commit/2f4a2fff5c8255e2deb9866ee05a736b3b7f5d6b)；[rocm-libraries 6798e891](https://github.com/ROCm/rocm-libraries/commit/6798e891e0a320e1210d15deeb70b0530d0ad2b9)。除明确撤回项外为源码已合入，未据此承诺正式发行包包含。

### ROCm hipDNN：默认模型化引擎选择，SDPA拒绝不满足契约的图

默认heuristic改ModeA按L1预测排序，精确缓存/手写配置仍优先，无预测回静态顺序；显式FALLBACK可保留无模型选择。ASM SDPA补group stats head stride、连续head、输出不重叠、FP32 backward compute、真实operand与FP8 scale检查；FlyDSL matcher及公共launch也拒绝单轴超2^32−1 work-items。拒绝旧错误图后**可能没有后备引擎**，不能写透明无损回退。约4GiB base-offset、gfx942长causal FP8与部分backward问题仍分开待修；某提交gfx950执行路径验证未齐。**建议升级前回归自己的layout与引擎选择**，这些是正确性门禁并非新性能承诺。

**版本与来源**：默认分支提交UTC日期 2026-11-21；具体SHA见[rocm-libraries 6c1b3b83](https://github.com/ROCm/rocm-libraries/commit/6c1b3b83f14fc37959c563f2c3187ec38e4ca043)；[rocm-libraries e3f81c33](https://github.com/ROCm/rocm-libraries/commit/e3f81c33ebf19a4935411f48748b5f15bf68bf64)；[rocm-libraries 9fa8f642](https://github.com/ROCm/rocm-libraries/commit/9fa8f642cdeb4bd104b5706b0cd342aec978fcf8)；[rocm-libraries 9f4051a3](https://github.com/ROCm/rocm-libraries/commit/9f4051a3d7a0dde7a3df9dca6472d8e892a02e77)。除明确撤回项外为源码已合入，未据此承诺正式发行包包含。

### CK Tile：gfx1250 MX调度流水线、批量Stream-K与LDS填充

新增gfx1250 comp_tdm_scheduled MX GEMM，FP4/FP8/BF8同宽输入、E8M0/K32尺度、packed RCR、4wave、TDM epilogue；该新pipeline不支持Stream-K。另一个提交把batch×tile纳入Stream-K全局划分并接batched dispatcher，同时修非2幂tile的TDM LDS row-aligned padding与容量门禁。两项能力不能合写为“新MX流水线已支持Stream-K”。有gfx1250设备与host用例，未据单shape示例建立LLM整网收益。**建议深入读支持矩阵后定向复现**小GEMM和批次分布。

**版本与来源**：默认分支提交UTC日期 2026-11-21；具体SHA见[rocm-libraries 6e8e3117](https://github.com/ROCm/rocm-libraries/commit/6e8e3117065633cd26f8b5cf98a85cd429ce9d26)；[rocm-libraries be0f4f6e](https://github.com/ROCm/rocm-libraries/commit/be0f4f6e28a390dd9fd12a52abfc1a6cc5a1b147)；[rocm-libraries 1d570395](https://github.com/ROCm/rocm-libraries/commit/1d570395445db15c5d012282e44fdd67d78cab01)。除明确撤回项外为源码已合入，未据此承诺正式发行包包含。

### MLX：低位宽Metal路径扩展与DSA head求和编译

affine qmv_mma新增2/3/5/6bit、group32/64/128，明确不含7bit；M1/M2 Ultra从4行可选，M3 Ultra及更新仍留vector路径，不能笼统说所有Ultra开启MMA。MLX-LM DSA把逐head ReLU、weight和加法按原顺序编译成shapeless图；保留求和顺序用于控制索引差异。**建议考虑在目标Mac复现**多并发/短批次与缓存；补丁未给统一吞吐提升和模型质量实验。

**版本与来源**：默认分支提交UTC日期 2026-11-21；具体SHA见[mlx 6d38ff93](https://github.com/ml-explore/mlx/commit/6d38ff93d77cd08b471d1cbc0930f464082f939a)；[mlx 5cabdf50](https://github.com/ml-explore/mlx/commit/5cabdf5091713e331cd8f0433f1d527ec6e641c9)；[mlx-lm 1bf8da05](https://github.com/ml-explore/mlx-lm/commit/1bf8da05a1ef30a3eb2e8a634180be6557fd3ca0)。除明确撤回项外为源码已合入，未据此承诺正式发行包包含。

### MLX：FP16梯度、量化临时分配和并发容量检查

分片梯度范数先把FP16平方和升FP32、缩放后再转回，防溢出/下溢；GELU近似用截断中间值和stop_gradient分离避免inf×0梯度。Metal量化分配移至外层并注册temporary，qqmm/gather_qqmm显式拒绝缺bias的affine模式；调度活动任务计数原子化并在容量压力时提交/等待，allocator统计受锁保护，多个算子补空输出早退。**建议优先核对采用版本并回归训练、并发及零token batch**；这些是所见具体修复，未做本轮Mac/GPU实测，不宣称所有并发风险已解决。

**版本与来源**：默认分支提交UTC日期 2026-11-21；具体SHA见[mlx 6ac6c0f4](https://github.com/ml-explore/mlx/commit/6ac6c0f4bc43889097ec9cc0c8cf25f3a3b546ab)；[mlx dc5ea6e9](https://github.com/ml-explore/mlx/commit/dc5ea6e98e97a97ffb1abce98fe7deb20ca5f39c)；[mlx 06712fba](https://github.com/ml-explore/mlx/commit/06712fbafdd32af67eeedf2628e2f99d805cf4be)；[mlx 0c93d91e](https://github.com/ml-explore/mlx/commit/0c93d91e8dd5863128388538b68996d87202465b)；[mlx 1fd565f6](https://github.com/ml-explore/mlx/commit/1fd565f699df298424892001a14b069ae15b11c3)；[mlx a3a610bb](https://github.com/ml-explore/mlx/commit/a3a610bb890ef3ebabbce8dd019328517ec5bc98)；[mlx 777d7417](https://github.com/ml-explore/mlx/commit/777d74171952d1fd99c664a63b692fa4a4183a5d)。除明确撤回项外为源码已合入，未据此承诺正式发行包包含。

### CANN：DynamicMxQuant指数选择与SiTU分组接口需要迁移审查

DynamicMxQuant scaleAlg2原把FP16/FP32 block max先缩至BF16，可跨UE8M0指数边界；现仅BF16保留快路，其他走通用计算。作者950/CANN9.3自编包62定向＋27其他算法抽样全过。SiTU Group新group_list_type默认0为累计终点、1为count求和，**最终diff已支持两种，提交说明“只支持1”已过时**；旧count调用必须显式1，零有效行/尾行不写不能当已清零。新GroupGrad交付950PR/DT输入及路由weight梯度，但文档对group_index单调性比前向更严格。**建议优先核对数值修复，迁移时逐接口复测**，源码合入未映射为正式CANN发行包。

**版本与来源**：默认分支提交UTC日期 2026-11-21；具体SHA见[ops-nn 20ab60f3](https://gitcode.com/cann/ops-nn/commits/detail/20ab60f3ca67c4b68f048b9ddf9abfe1154b5656)；[ops-nn b4ea74bd](https://gitcode.com/cann/ops-nn/commits/detail/b4ea74bd9c5ea158af7d5685ffa0d56904e3eace)；[ops-nn dfcbb2d9](https://gitcode.com/cann/ops-nn/commits/detail/dfcbb2d9e2ab26dfc3f17141fa316842dbdaaf4c)。除明确撤回项外为源码已合入，未据此承诺正式发行包包含。

### CANN：text_gated_residual尾块固定行所有权，消除三槽复用WAR

原按actualM划分AIV，mTask128/尾块126使分界64变63，后任务AIV1可能覆盖前任务AIV0尚在读的xn行；改固定mTask所有权后再裁剪有效行。作者A2/950完整32766×10240 BF16输入100次五输出逐位稳定，A5缩小但保持126尾块的racecheck旧版检出GM写后读冲突、新版无错；完整A5 sanitizer因124GiB cgroup内存中止，未完成。小shape优化另限D10240/R320/C4/H2560及专用核数，A2对主线耗时−17%～35%、A5约−8%～27%；只正向、A3未上板。**建议优先考虑采用并复现尾块与训练缓存**，不能外推所有shape或反向。

**版本与来源**：默认分支提交UTC日期 2026-11-21；具体SHA见[ops-transformer 4d6e0004](https://gitcode.com/cann/ops-transformer/commits/detail/4d6e0004be9660cbf1416d2ce57eb38af183a876)。除明确撤回项外为源码已合入，未据此承诺正式发行包包含。

### CANN：稀疏MLA地址预计算有大收益，同时保留旧快路回归

950混量稀疏MLA接Vec0向量PA地址预计算，按物理页跨度选uint32或int64偏移，并给跨核FD各自地址表。作者950DT、32Cube/64Vector、新DSL接口，四轮交替/每次前清8GB设备内存：新增decode49.01→15.59µs、prefill544.68→115.99µs；原145shape最大退0.75%，需新squeeze接口wheel，非所有shape3倍。Arch22另一改动移出重复索引获取，仅FP16提前CMP首波通知，BF16保留原搬运；910B3重测18组13组相对同进程SMLA增幅在6%内，开发极端FP16改善10.64%仍比SMLA慢6.47%。**建议深入读并复现**同shape/精度/包对照，不混用两硬件性能表。

**版本与来源**：默认分支提交UTC日期 2026-11-21；具体SHA见[ops-transformer edc3505e](https://gitcode.com/cann/ops-transformer/commits/detail/edc3505e55eb2b14aedd5b41c8d92707d6c72824)；[ops-transformer bb3640b9](https://gitcode.com/cann/ops-transformer/commits/detail/bb3640b9b07e8fe849d12deb59f010e2cd0c975c)。除明确撤回项外为源码已合入，未据此承诺正式发行包包含。

### CANN：sink软最大数值稳定、MSA分数下界与分页存储描述修正

A3推理attention把sink作为零value token初始化max/sum，替代尾部exp(sink−max)叠加，split-KV仍由FD处理sink；MSA score在写回前统一有限下界，防mask或FP16溢出留下−inf；GBSA删除把非连续分页view的storage_shape覆成逻辑shape的代码，避免sanitizer误判合法跨页越界。后者是存储边界描述修复，不能据标题写真实越界已修。sink/MSA提交缺完整设备实验数据。**建议定向回归**大sink、全mask、尾页/非连续view，按代码机制理解，不编造故障率或质量提升。

**版本与来源**：默认分支提交UTC日期 2026-11-21；具体SHA见[ops-transformer 265d2174](https://gitcode.com/cann/ops-transformer/commits/detail/265d2174fbe16e1a73cf15e180dd1d5822db888c)；[ops-transformer b84699a5](https://gitcode.com/cann/ops-transformer/commits/detail/b84699a5d562e5dd2bffdc923edc8b8be497a328)；[ops-transformer 61a3223c](https://gitcode.com/cann/ops-transformer/commits/detail/61a3223cf6e0d219b6d9fada662607ae013ce03c)。除明确撤回项外为源码已合入，未据此承诺正式发行包包含。

### CANN：MoE路由gather去重、可选梯度输入与通信索引同步

moeInitRoutingV3在arch35按源行去重后一次搬入/多次写出，扩静态/HIF8/MXFP4/FP8 per-block等路径，full-load不走gather且MXFP4列数须偶数。MoeTokenUnpermuteGrad只在probs存在时要求permuteTokens，作者950DT三dtype双空逐位对照。EP通信另按是否有topk weight计算WQE/地址buffer容量，避免flatten topk全局偏移进入int32，并补V/S/MTE同步；grouped_matmul非量化950新增L0C双缓冲，1000泛化精度及多架构编译记录没有可比性能表。**建议考虑采用并回归**路由padding、可选输入和大索引；未给端到端加速结论。

**版本与来源**：默认分支提交UTC日期 2026-11-21；具体SHA见[ops-transformer 84b913ca](https://gitcode.com/cann/ops-transformer/commits/detail/84b913ca9f9d3211903d5b07b4cec19c0e374715)；[ops-transformer 07d90f55](https://gitcode.com/cann/ops-transformer/commits/detail/07d90f550c21e5cf556918a81eef03c6dd49de39)；[ops-transformer e5d252a1](https://gitcode.com/cann/ops-transformer/commits/detail/e5d252a10ff1a8aae04e3e34e661d76c03dbd6f5)；[ops-transformer 01557b94](https://gitcode.com/cann/ops-transformer/commits/detail/01557b94f57a75b0f4394479d0c795ce7dc23010)。除明确撤回项外为源码已合入，未据此承诺正式发行包包含。

### CANN HCCL：卡死计数重合入，等待包含排队，不是纯kernel耗时

AICPU/AIV/CCU插入配对head/tail原子计数，取址/发射失败fail-open；AICPU缓存录制与重放共用计数路径，三地址任缺一侧都不发，防永久误报。AIV只block0计数、必要时tail前补本地同步；CCU按kernel粒度trampoline，不定位内部阶段。head−tail包含host NotifyWait与队列等待，阈值必须大于正常排队；host/device字段布局要同版本。**建议深入读并定向验证**卡死定位与误报，不能把计数器差值当纯通信运行时，也无独立设备验证。

**版本与来源**：默认分支提交UTC日期 2026-11-21；具体SHA见[hccl 21791ba6](https://gitcode.com/cann/hccl/commits/detail/21791ba6eed2564519444080eb56a104353c3b8f)。除明确撤回项外为源码已合入，未据此承诺正式发行包包含。

### CANN Runtime：StreamStop后返回明确错误，ACL Graph控制面继续演进

停止队列不能推进时，普通同步/销毁从持续等改返回107046，提示先StreamAbort清理恢复；销毁失败沿用对象处理，**不保证旧句柄可重试**，Stop=false也不代表整个恢复成功。作者276单测与ACL透传通过、真实设备说明未入库。ACL Graph V6涉及TS agent swapbuffer、runtime管理ID、物理CQE及异常处理，最终V8把报告stream/task ID改uint16并修组合提升类型；“V6/V8”是提交标题，不是SDK版本。另增PCIe ATU映射/直接NotifyRecord仓间接口，驱动缺符号明确不支持。**建议跟踪正式包映射并回归错误恢复**，未给host-bound改善量，不按标题声称已提速。

**版本与来源**：默认分支提交UTC日期 2026-11-21；具体SHA见[runtime a6d62de0](https://gitcode.com/cann/runtime/commits/detail/a6d62de03ccc47c9d2d962ae2dc4503f6cde1729)；[runtime a501311d](https://gitcode.com/cann/runtime/commits/detail/a501311db5b75b13d82f74f11b0507f027307a27)；[runtime fd20a65b](https://gitcode.com/cann/runtime/commits/detail/fd20a65bd212f2b85bd18758174c2036643a47f6)；[runtime ff0269e2](https://gitcode.com/cann/runtime/commits/detail/ff0269e24f778357fb8082fc32e9b01a8803ca1a)。除明确撤回项外为源码已合入，未据此承诺正式发行包包含。

<a id="topic-7"></a>
## 评估与可靠性

### Sarvam：Samvaad v2.1 开放多语客服下一步决策，另补旧模型评测配置

11月21日数据固定 `3bc927f79c0372328b7b57f9429da86122d1a666`，Apache2.0。按对话历史/指令/工具检查下一步该说话、静默调用还是边说边调用，以及工具参数、状态转移、语言与script。**本轮独立读取JSONL核实338例、13语言、620确定性检查、3,340 rubric实例**；英语132例，语言分布明显不均。卡列158纯说话/146纯工具/34混合，只有33例显式状态转移。交付是文字消息与工具schema，未含音频字段，不能作为ASR、TTS、打断时序或端到端语音时延证据。checker已读，本地因缺jsonschema未执行全量checker；未独立调用模型/裁判。**考虑复现**为多语Agent回归集，先人工审核rubric与业务流程适配。[固定数据卡](https://huggingface.co/datasets/sarvamai/samvaad-bench-v2.1/blob/3bc927f79c0372328b7b57f9429da86122d1a666/README.md)；[确定性checker](https://huggingface.co/datasets/sarvamai/samvaad-bench-v2.1/blob/3bc927f79c0372328b7b57f9429da86122d1a666/deterministic.py)。

另11月21日 `sarvam-105b-1711`固定 `50d799f9e6963c5839cee1dc6d8b0698b42c1d0f`新增`benchmark_config.yaml`，文件内部updated为11月6日；是旧模型复现条件补全，未换权重。默认4采样/64K输出、GPQA8采样、数学64采样，IFBench用strict而官方口径为loose；GDPval-S用209题和LLM rubric，与expert win-rate不同。**建议回读配置后重审旧比较**，缺少逐次原始输出与硬件，不能把新披露配置写成独立复现。[新增配置](https://huggingface.co/sarvamai/sarvam-105b-1711/blob/50d799f9e6963c5839cee1dc6d8b0698b42c1d0f/benchmark_config.yaml)。

### Sakana CoffeeBench v2：补齐102次六Agent经济模拟的完整轨迹

11月21日新数据工件 `4cea233c2d892ca7b728be4688a2af0d43e301a6`；论文是既有`2606.16613`，本次增量为可检查的输出和模拟器账本。公开54个90天主实验（8模型×高低reasoning×3 seeds，加规则基线）和48个30天KPI压力实验，包含六Agent消息、交易、commands、成本、daily books与原始事件；可追究高收入是否来自循环交易，而非只看总分。Apache2.0附第三方模型输出条款提示；卡明确这些是**评估输出，不是训练数据**，文本含canary并做可逆混淆。

**建议深入读并考虑复现论文表格/账本核对**；本轮读取固定卡与index，未下载102个完整raw runs重算，也未独立验证全部经济机制。异构多Agent、指定对手/席位与KPI压力的结果不应外推成单模型普遍能力。[固定数据卡](https://huggingface.co/datasets/SakanaAI/coffeebench-trajectories/blob/4cea233c2d892ca7b728be4688a2af0d43e301a6/README.md)；[index](https://huggingface.co/datasets/SakanaAI/coffeebench-trajectories/blob/4cea233c2d892ca7b728be4688a2af0d43e301a6/index.json)。

### MiniMax 内部模型密码分析（11月20日，补录）：可执行工件增加可信度，模型与搜索配方仍未披露

11月20日官方发布、固定研究仓 `57842cef92e84eaf8e2535b12f48ba5586080ddd`，本轮补录。内部模型MIM针对历史密码竞赛候选、同行评审构造及公开数学挑战，发布8组新技术报告、证明/验证工件与代码（另有10月旧BentSum）。价值是把长程推理落到可核验结果，区分构造缺陷、参考实现bug与求解器加速；不是新开放权重模型。官方明确不影响已部署密码标准。

公开仓有Apache2.0许可，但未披露MIM架构、训练数据、后训练过程、搜索/工具调用预算或同预算模型消融；本轮核文章、固定仓树及README，未执行攻击或逐篇审计密码学证明。因此不把“首次/加速倍数”升级成独立确认，也不从运行攻击的秒数推导模型研究成本。**继续观察并建议领域专家深入审读验证材料**，对通用LLM训练方法的可迁移证据仍不足。[官方文章](https://www.minimax.io/blog/practical-cryptanalysis-with-minimaxs-latest-internal-model)；[固定研究目录](https://github.com/MiniMax-AI/Research/tree/57842cef92e84eaf8e2535b12f48ba5586080ddd/Cryptography)；[发布索引](https://github.com/MiniMax-AI/Research/blob/57842cef92e84eaf8e2535b12f48ba5586080ddd/README.md)。

### RSIBench-Infra：把Agent优化提交与评测进程隔离，测端到端收益（2611.24395v1，补录）

**机构与时间**：清华AIR／字节Seed；v1提交11月18日，本轮补录。20项任务含16推理、4训练，GPU／昇腾NPU，1—32设备；从可运行官方配方出发，Agent提交OCI镜像层，独立部署任务加载修改，Benchmark Job不加载该层，以固定任务质量门槛和完整负载计性能，减少只优化kernel或篡改计时器的问题。

130个正式计分结果中86个超过1.10倍；这既不是20任务都提升，也不是每个Agent的成功率。官方配方基线未必最优，质量门槛按任务、且不等于逐bit等价；部分配置因API错误只完成子集。**建议深入读并借鉴评测隔离与负对照**；作者内部“独立评测任务”仍属论文证据，不能当第三方复现。[§3–4、附录B](https://arxiv.org/pdf/2611.24395v1)。

### MoE干预归因：后续路由切换会让局部梯度漏掉专家影响（2611.26075v1，补录）

**机构与时间**：芝加哥大学／eBay／佛罗里达大学；v1提交11月19日，本轮补录。对专家替换，除二阶项外，用模型原生专家输出和选择规则估计后续路由切换影响，修补纯导数的局部近似。

Qwen3.6-35B-A3B与Gemma4-26B-A4B-it，158道MMLU-Pro／SuperGPQA／HLE文本多选题、FP32且禁TF32；相对五步二阶导数对照，作者称答案分差预测误差下降60／47%。这是**干预效果预测误差**，不是任务准确率提高；需要白盒权重、激活及逐干预HVP，编译与查询成本不能只按结构项计。**建议深入读并考虑复现**固定后续专家身份的机制对照，避免直接拿梯度排名裁剪MoE。[§3–4、附录B/C](https://arxiv.org/html/2611.26075v1)。

### Judge量表反转：先扣除自身重测噪声，再判断语义不忠实（2611.25588v1，补录）

**项目与时间**：Rubric Semantics审计团队；v1提交11月19日，本轮补录。保留评分语义、只反转数值锚点，再把反转不一致与正常量表重复运行的不一致比较；避免将模型自身随机性误判成量表偏置。

5个元评测、7种开放／API裁判显示显著残余方向偏差；但不同5/10分量表的缺口不能跨benchmark直接比较。双向评分映射后取平均，在21组pairwise设置中17组改善、2组持平、2组退步，仅9组改进区间完全高于零，且多一次评分调用。**建议考虑复现**本方reward/judge的正反锚点与重复对照；平均化是经验缓解，未保证所有任务无损。[§3–5与Table3](https://arxiv.org/html/2611.25588v1)。

### 自一致性证书：同题反复采样不能直接认证总体可靠性（2611.26592v1，补录）

**机构与时间**：CUNY／Rutgers；v1提交11月19日，本轮补录。交换潜变量相关模型下，传统逐样本累积证据仍可认证给定潜在状态的命题，却不能自动认证跨题总体命题；有效样本量的饱和针对总体均值问题，不能解释成“采样三次以后全无用”。

Claude Haiku4.5、温度1、GSM8K300题及AIME150题各30rollouts；AIME的逐题触发24%与跨题程序0%是描述统计，**不是实测总体I类错误由5%变成24%的证明**，论文明确强调此区别。任意题内串行依赖仍未覆盖。**建议深入读**用于停止采样／可靠性SLO的证书定义，复现时分清认证对象、题目聚类和独立性。[理论§3–7与实验§8](https://arxiv.org/html/2611.26592v1)。

<a id="coverage"></a>
## 实际覆盖、补录与续查

本轮记录 **160 个稳定版本事件**；同一工作多工件或撤回链会有多个事件，不能把该数当独立研究数。所有性能/质量数值均按原始来源归因，本轮未执行独立复训、GPU/NPU/TPU压测或付费模型API实测。

模型与工件线实际检查157个官网/研究/评测/发现入口（106个完整HTTP200；23连接失败、9个403、9个308、10个404），181个定向HF组织的模型/数据目录共363页完整200。HF全站创建目录跨过主窗起点，枚举3199模型、2082数据集；这是创建枚举与选择性语义审读，不是任意更新或长尾全覆盖。读取48个固定工件的历史、模型卡、配置、许可和文件目录，连同定向补查共845条来源记录，11个稳定事件合并为9个工作组，5个事件按原始日期补录。

本轮补录Munin2.0（原11月18日）、MiniMax密码分析工件（11月20日）、OpenAI芯片设计文章（11月18日）。Dolci修订按11月21日main合入、卡内11月7日准备和数据revision11月8日分别记载。Sarvam评测yaml的内部updated11月6日与11月21日公开提交分开；没有新权重。历史日报全文及稳定ID作去重线索，关键正文重新打开核实。

未完成：失联/动态壳官网、全站修订与长尾语义；MemTPro卡新增说明但API/history401，权重/配置尚不可核；amd/Nitro-P仅四个512/1024px safetensors，README/config/license404，披露不足不收；Edge0第三方vLLM插件待代码及负载复核，不作为upstream能力或采纳性能；Granite GGUF重传待二进制模板核对，不能从fix-template标题推正确性。WeMM本轮大部分仅元数据、另音频图新增；Inkling框架链接与IFM latent-feedback引文变化未重复算模型发布。

Jais六个格式工件和K2-Horizon补入完整Apache2.0文件已核；不作为新模型事件。Samvaad数据338行/13语种、620确定性检查、3340rubric实例已独立计数，无音频字段；本地checker因缺jsonschema未执行，未冒充验证通过。DFlash当前固定目录无原始速度点数据与MRCR选择manifest；未实机压测、复训或调用付费API。继承所有旧缺口，不关闭全局语义水位。详见models/ledger.json及verification.json。

论文线检查主窗为 (2026-11-21T00:58:21Z, 2026-11-22T01:03:50Z]，向前回溯至2026-11-18T00:58:21Z，并继续处理继承候选。本轮13项均为此前未收的固定v1论文补录，原始提交11月16—19日；未将其写成11月22日首发。首次公开秒界未证明，主窗新增数不能据此判为零研究。

实际尝试52次直连：10个arXiv分类、精确lastUpdatedDate Atom、HF日报、OpenReview ICLR/TMLR、ACL、PMLR，以及18个候选的abs/HTML。arXiv和PMLR DNS失败，HF连接超时，OpenReview两项403；ACL HTTP200但仅接收199,941/7,656,804字节后超时，均不算完整扫描。

改用web实际打开10个arXiv分类入口、15个固定abs、机制／实验条件／结果／局限段落，以及两篇HTML404对应的PDF正文。八类入口显示11月20日，cs.DC仍11月19日、cs.PF仍11月13日，体现返回页缓存差异；不能把它当精确主窗或版本更新闭合。PMLR只读索引、ACL web仍超时、ICLR web不可访问；HF11月21日聚合不可访问。中英文主题检索只发现线索，正文关键结论均回到原始论文，未采用二手稿作指标证据。

本轮保存182条来源请求／摘录记录（52直连、130web片段记录），审阅13篇入选论文的关键正文段落；原始完整HTML下载不可用，不能宣称完整逐行文档或源码审计。无独立复训、压测和付费API实验。继承候选合并去重后2,237项继续待审；历史未完区间及替换版本差分、OpenReview、ACL／PMLR／SC26缺口均未关闭。

跨正文去重额外剔除GLiDE2611.26509v1（11月21日模型章节已完整报道，稳定标识缺arXiv别名）与EchoMoE2611.22215v1（11月18日已收同工件及核心结果，本轮没有新版本）。建议将这两个arXiv别名加入既有工作身份，以免跨分类重收。A CPU Inside an LLM2611.26450v1已打开，但零梯度电路构造、编译器和保留能力协议需更充分核对，暂不入正文。

建议独立抽核四处：Aggron头条性能为模拟、TripleKV为固定轨迹真机测量；PRKV计时硬件与重算窗口注意力；OPD前向KL理论条件与反向KL容量差实验；自一致性24%逐题触发不是总体I类错误率估计。

工业工程：尝试84个仓库的默认分支、merged PR、release Atom及REST历史，另打开TensorRT-LLM release正文。完整清单仅default HTML 7、merged 7、REST 12、Atom 2个；传输中断即便HTTP200也未计完整。Megatron默认及合入目录、verl REST/合入目录、slime默认/合入目录、TorchTitan合入目录恢复；vLLM/SGLang分页、FlashInfer/Triton/PyTorch等未查完。67个入选版本按固定补丁/PR原文核实，37组，原始patch恢复84/106；其余通过当轮一手PR正文支持，未写全源码审计。

状态冲突的18项未当已发布能力：含vLLM Inkling低M/TMA、masked-token，SGLang exact-row penalty/ROCm视频VAE、KVCR两项，Omni PersonaPlex/RAS/CoVo/OmniVoice、Transformers两项及TorchTitan PP。CI tag/ciflow不是正式release。TensorRT1.3旧发布去重，不因当前页面可读重收。主窗和72h回溯都有语义缺口，314个关键词候选留在follow-up-candidates.json，未以条数配额截断；其中可能有已报别名和普通维护，待逐项筛选。所有历史缺口继续继承。

补录标注以合入时间为先，不把commit Author Date当当天首发。历史链接按仓库大小写归一交叉去重（避免DeepSpeed大小写别名漏判）。本轮没有独立GPU/NPU压测、复训或在线模型调用。

- **硬件与软件协同**：472次原始HTTP请求、226次完整200（包含重试与固定diff，并非独立来源数）；21个Git传输均失败。直接读取NVIDIA、AMD、Intel、TPU/Neuron、昇腾、Apple及国产加速器入口；成功目录恢复698个窗内事件版本，关键入选读固定patch/GitCode提交与files。多个高频仓首屏不足、候选语义/旧缺口未完，CANN发行包对应和libtpu内部行为继续查，不声称全面覆盖。

- **补录与更正**：13篇论文均是11月16—19日固定v1的漏收；Munin2.0、MiniMax密码分析与OpenAI芯片设计保留原始日期。GLiDE、EchoMoE跨分类同版工作剔除并补稳定ID别名。CANN SiTU按最终diff修正过时提交说明，SGLang TPU按明确revert记录状态。工程源状态冲突项继续观察，不能视为可用能力。

- **未完成区间**：继承此前全部待查组，另保存本轮缺口与候选清单。论文分类页日期不一致、API/直连失败和高频仓分页不足均未冒充成功；全局完整语义覆盖水位仍为空。下轮从本次发现截止续扫并再向前回溯72小时，按各来源已记录区间补查。

- **归档同步**：本轮新clone回读tracker/main，44篇已上传旧日报与本地逐字一致；11月21日报尚未上远端。标准Git无写凭证、gh不可用、GitHub插件未安装且浏览器未登录；实际push先遇DNS失败，待恢复后仍须认证。今日日报、上期待补传日报及最新README保存本地，待同步状态与本地生成分开记录。不得将本地准备提交当上传成功。
