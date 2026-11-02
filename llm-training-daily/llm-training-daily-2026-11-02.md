# LLM 技术动态日报 · 2026-11-02

> 时区：Asia/Shanghai（UTC+8）。增量扫描窗口：**2026-11-01 08:57:10 → 2026-11-02 09:42:47**；72小时回溯：**自2026-10-29 08:57:10起**。实际生成时间：**2026-11-02T10:07:14+08:00**。UTC时间保留Z。原始提交、合入、工件更新与本次发现分别标注；LongCat仅有日期，单列为运行中发现、是否在截止前未定。
>
> 本轮先读取自动化memory、持久化state/sources、26篇历史日报及GitHub main树/索引/上一期全文。已有本地归档与远端blob及上一期全文一致，远端基线`d5b7bb7138d0ce87becd43f2e06e7d097537dc35`，无待补传。只维护指定归档，未改llm-knowledge wiki。
>
> **所有性能、模型成绩和作者测试均不等于本轮独立复现。** 本轮重开一手材料、固定版本、最终文件及若干真实main提交，未运行GPU/NPU/TPU或多机训练。事件枚举截止用于下一轮续查，不能当作全行业完整语义覆盖水位；旧失败及未审区间全部继承。

## 本次重要变化

- **训练正确性优先于小算子倍率。** TorchTitan修复TP交叉熵高置信塌缩；PyTorch新增独立FMA收缩开关，并修正CooR跨卡缓存／同步目标；Megatron MFSDP补outer归约重叠、真实顺序预取及MXFP8重计算，两项策略默认仍需显式开启。
- **量化工件有真实交付，也有测试集选型边界。** d3-ultra-NVFP4替换了权重，512条新校准改善Box精度但未恢复BF16；Metis新增GGUF/MLX混合精度工件。模型卡的权重大小、单请求时延、实际加载路径与泛化成绩分别解读。
- **长上下文服务的布局、内存与完成边界继续变化。** vLLM补PP共享索引、按层KV编码与offload请求上下文；SGLang修大地址溢出、compile下KV alias、bootstrap及外部缓存写回。局部kernel支持、集成分支和正式release分开报告。
- **线性注意力训练出现值得实验的实现选择。** FlashInfer新增GDN/KDA非确定性单卡与KCP后向、lean workspace及SAC修复；默认数值模式、SM数量、保留workspace和整模峰值显存不是同一合同。
- **研究阅读重点是阶段策略与基线公平。** PDScope的prefill/decode专家搬运策略适合受限显存实验，但交付工件缺learned predictor；SCARL默认脚本与论文配方不同；投机能耗收益会随CUDA Graph、实际batch和负载明显变化。Dynamic Wave对fused SDPA的显存优势消失，是应保留的负面证据。
- **补录系统卡揭示退步。** GPT-6 Luna 10月29日增补中的直接指令高分伴随间接注入、长度调整HealthBench与事实性退步；不因HTTP修改时间而写成今日新修订。LongCat-2.5目前核到正式API文档，训练报告和匹配新权重证据仍缺。

目录：[模型与技术报告](#models) · [预训练与架构](#pretraining) · [后训练](#posttraining) · [推理研究](#inference) · [评估与负面结果](#evaluation) · [核心开源工程](#core) · [扩展工程及硬件协同](#extended) · [实际覆盖与续查](#coverage)

<a id="models"></a>
## 模型与技术报告、部署工件

### d3-ultra-NVFP4：本轮实际更换权重，校准精度改善伴随测试集选型边界

**vLLM Semantic Router；新权重提交 11-02 00:22:18Z `cf65de7c`，结果卡 00:22:52Z `53353641`。** [固定工件](https://huggingface.co/vllm-sr/d3-ultra-NVFP4/tree/53353641644960cb9d37bcb92a5729cfcbd72caa)是非生成式、多模态决策模型的 Blackwell W4A4 版本；原始 26.09B 模型不是本日新发。已对完整目录逐路径比较：旧版 `becd81ec`→`cf65de7c` 改变 `model.safetensors`、manifest 和卡；随后 `53353641` 只改 manifest/README。因此本条是**重新校准的量化权重交付**，不能只归为 README 刷新。

新配方用 INT4 校准池打乱后的前 512 条请求（316 文本、153 Box、43 图片选项；无视频），代替含视频的旧 175 条。作者采用 AWQ 后接针对 NVFP4 激活的 GPTQ；[配置](https://huggingface.co/vllm-sr/d3-ultra-NVFP4/blob/53353641644960cb9d37bcb92a5729cfcbd72caa/config.json)确认 16 值块、4 位权重/激活和 E4M3 scale，视觉塔等部分保留较高精度。[运行代码](https://huggingface.co/vllm-sr/d3-ultra-NVFP4/blob/53353641644960cb9d37bcb92a5729cfcbd72caa/d3q/kernels.py)的 `load_backbone` 保持 packed 权重，`NVFP4Linear._gemm` 把激活量化后交给 FP4 `scaled_mm`；默认 Transformers 路径会解量化为 BF16，不能沿用低显存数字。

作者在单 RTX PRO 6000 Blackwell、同 runtime、单请求串行的文本测试中报告 **32.5ms vs BF16 63.4ms**；**17.2GB vs 52.2GB 是权重大小**。Box 测试 5,836 项、IoU≥0.5：BF16 **0.8717**、旧量化 **0.8436**、新版 **0.8538**；改善仍未恢复基线。卡明确两版均用 test splits 参与选型，故不能视作完全独立的泛化估计；文本 Decision Index 的 BF16 公开基线还来自不同 GPU/engine。Apache-2.0，权重与推理实现可得，量化构建脚本仅链外部研究分支；本轮未重跑 GPU 评测。**建议针对票据定位、知识分类和生产分布重新留出测试集，再决定是否接受精度换成本。**

### GPT-6 Luna 10-29 系统卡补录：直接防注入高分不能覆盖间接注入与事实性退步

**OpenAI；原发布 10-29，本轮发现补录。** [16 页增补报告](https://deploymentsafety.openai.com/gpt-6-luna-october-28-2026-addendum/gpt-6-luna-october-28-2026-addendum.pdf)封面/官网均标 10-29，URL 的 `october-28` 不作发布日期。当前 PDF SHA256 `766262a7…`；HTTP 修改时间 11-01 01:42:04Z 只作快照元数据，不能证明内容在本主窗改版。

更新目标是简洁、减少附和，§4.2 继续 GPT-Red 自动对抗训练。直接指令层级鲁棒性 **99.91%**；但 Table 4（PDF 第8页）跨可用推理档等权的间接注入指标，Luna 10-07→10-29 为 **95.80%→94.50%**。同页 Table 5 的长度调整 HealthBench Hard 为 **27.2→22.8**；§6.1 说明六个事实错误指标均高于 10-07 版。不能把更新概括为全面提升，也不能据此归因于输出变短。§2仅称数据和训练类型沿用旧版，未披露本轮配方、架构或可复现权重；旧模型比较列可能使用后续版本。**建议优先复测长任务间接注入与事实核验。**

### Metis-1.6 新增 GGUF/MLX：量化敏感的小张量决定大部分精度损失

**Lernex；11-01 19:20–19:45Z 交付六个 GGUF/MLX 仓库。** 10-27 的训练报告已在 10-30 日报分析，此处只记部署工件增量。[Logos GGUF `f5ad2cb6`](https://huggingface.co/Lernex/Metis-1.6-Logos-GGUF/tree/f5ad2cb6d230309a5bb1afe2c34e22f4778a4f42)、[Praxis GGUF `4c8d94cb`](https://huggingface.co/Lernex/Metis-1.6-Praxis-GGUF/tree/4c8d94cbbf699e101cd56a047b2e1f806643375b)各含 BF16 和五种量化权重、importance matrix；另有两模型各 4/8 位 MLX。Apache-2.0。

作者把 Mamba 输出投影留 BF16，潜空间投影留 Q8，Mamba 输入留 Q6/Q8；MLX [固定 `Model.quant_predicate`](https://huggingface.co/Lernex/Metis-1.6-Praxis-MLX-4bit/blob/b33f92530904d133c8094af90610917345858037/metis16.py)亦保留路由器/深度组件，主要压缩 routed experts。GGUF 卡在 wikitext-2、512 token、40 chunks 的同 BF16 对照下报告：Logos Q4_K_M **6.7GB、PPL15.998 vs15.873**，Praxis **2.0GB、24.568 vs24.121**；不能推广成任务质量不变。MLX 4 位实为混合精度，Logos 卡报平均5.14 bits/weight。llama.cpp 需含 `metis16` 支持的版本，Ollama/LM Studio 还受各自打包版本限制；MLX 自定义实现需要 `trust_remote_code`。**建议先验收运行时支持和自己的长上下文/工具任务，不沿用卡中未充分交代长度与批量的 tokens/s。**

### LongCat-2.5 正式服务文档出现：本轮发现，精确上线时刻未披露

**美团；[官方日志](https://longcat.chat/platform/docs/zh/ChangeLog.html)标 11-02，本轮首次抓取 01:45:23Z（北京时间09:45）。** 较前轮快照新增正式版条目及视觉文档入口，强调原生多模态、长程编程与多种 harness。因抓取晚于本日报 01:42:47Z 截止，且日志只有日期，本条单列为**运行中发现、是否早于截止未能确认**。

[参数文档](https://longcat.chat/platform/docs/zh/api/model-params)列输入输出合计 **1,048,576 tokens**、最大输出/思考预算 **262,144**；开启/关闭思考时温度固定 **1.0/0.7**，其他值会被覆盖。[视觉合同](https://longcat.chat/platform/docs/zh/image-video-understanding)支持图片 Base64/URL、视频仅 URL；每请求≤50图/10视频，单图≤10MB、单视频≤100MB且<3600秒。这些是接口声明，本轮未调用服务验证。已查页面未披露正式版架构、训练配比或受控评测；本轮 HF 官方组织枚举也没有形成匹配新增权重的证据。**建议继续观察技术报告/权重，以生产截图和长程工具链验证接口兼容。**

<!-- 覆盖与未纳入项由 ledger.json 提供；未命中检索不等于厂商没有发布。 -->

<a id="pretraining"></a>
## 预训练、长上下文续训与混合架构

### Hybrid Mechanics 补漏：滑窗与线性混合在外推、续训之间交换优势

**`arxiv:2610.10114v1`；原始提交2026-10-07 14:02 UTC；旧文补录，11月1日中文报道不是新论文。** OpenMOSS/Fudan/SII在376M、776M、1B进行50B token、4k预训练及5B token、32k续训，另用3B验证关键现象。SWA-NoPE在免训练长度外推更强，GLA/GDN-NoPE在长文续训后的训练窗口内更强；固定短滑窗会把优化集中到局部依赖，扩大窗口与LongCE可缓解（§3、§5）。[固定版本全文](https://arxiv.org/html/2610.10114v1)。

其SWLA把4096长窗口以1024步幅重复计算再拼接，并给NoPE注意力加log-scale；这是一种分块近似，不能直接等同原生精确滑窗递推。776M GDN-NoPE-LH在64k的NIAH-SK1达到100%，但同表SK2/SK3只有90%/53%；GLA-NoPE-LH则为100%/59%/64%，不能把单针检索外推为通用64k能力（§6.2，Table1）。训练吞吐测试用8×H200，3B训练需16×H200；GDN/GLA状态FP32，滑窗KV FP16。公开仓库在固定提交`1c19814`仅README/PDF，未交付训练或SWLA实现；本文只覆盖预训练文本任务，未证明后训练、多模态和复杂推理迁移。

<a id="posttraining"></a>
## 后训练、课程与持续学习

**建议深入阅读并做同预算短滑窗／GDN续训对照。** 它直接区分免训练外推与长文续训，代码尚缺，暂不直接采用headline结论。

### SCARL：同符号优势内重分配更新，但论文配方与部分脚本不同

**`arxiv:2610.40407v1`；原始提交2026-10-29 16:38 UTC；72h回溯补录。** SCARL把旧策略整段响应log-prob聚合为组内置信度：低置信成功、高置信失败获得更大权重，再分别归一化正负类，保留基准优势的方向与每类平均预算；不增加模型前向（§2，Appendix D）。[固定版本全文](https://arxiv.org/html/2610.40407v1)。

DAPO-Math-17K、150步、8×A100 40GB、每题8条rollout下，Qwen3-4B-Base五数学集平均Pass@32由DAPO的67.1升至69.5；16个估计器×模型组合有14个改善，两个SAPO组合均下降0.2。去掉分类归一化反而从56.4降至47.0（Table1–3）。这里每5步评估并选测试集平均最优checkpoint，Pass@32是对32条响应进行有放回bootstrap，不能直接当标准无偏pass@k或固定终点结果。

官方仓库`b1d9d37`的`core_algos.py`确有log-prob求和与正负类归一化；但`run_scarl_dapo_qwen3-4b_base_single_node.sh`默认`logit/tau=3/power=2`，不同于论文`old_log_probs/tau=5`，而GSPO脚本设为`prob`；DAPO还设`lam=16`、GSPO为`lam=1`，logit分支实际聚合`logsigmoid(selected_logits)`。因此有代码不等于默认脚本复现论文。结果仍限于≤4B、二元数学验证奖励，未独立复现。[固定代码](https://github.com/kezhh163/scarl/tree/b1d9d37336e1b7a36a520c8218a800a4f9339f09)。

**建议考虑小模型复现。** 先固定论文真正使用的置信度定义、lam和checkpoint选择规则，再比较固定终点与独立测试集。

### ClosedLoopGaussian 补漏：课程反馈有稳定性信号，尚无等算力显著收益

**`arxiv:2610.39056v1`；原始提交2026-10-28 23:35 UTC；早于本轮72h窗的pending补录。** 在Countdown按操作数2–5分桶，Gaussian课程中心决定难度推进，宽度决定复习范围；每100优化步用每桶100题greedy probe，根据超前/落后和低难度遗忘调整两者，仍回归固定参考轨迹（§II）。[固定版本全文](https://arxiv.org/html/2610.39056v1)。

Llama-3.2-3B-Instruct、LoRA rank32、1600步、每题4个GRPO样本、4 seeds下，均分42.4±14.0，对比固定E2H-G的35.0±11.4和普通GRPO的26.4±21.6；后者两次降至7.7，均值受崩溃影响很大。单次计时闭环12.94h、固定课程11.49h、GRPO10.82h，探测成本未抵消；seed42参与校准、seed43的消融系事后选择，作者也未作统计显著性结论（§III，TableI）。硬件与精度未在实验设置披露，未比AdaRFT/DUMP等自适应课程。官方`55d73db`中调度中心/概率与probe更新均可对应论文，未运行复现。[固定代码](https://github.com/yuvrajpant56/Closed_Loop_Curriculum/tree/55d73db66dc1ec9db507741f506f4b3cd19b2401)。

**建议继续观察并复现多seed稳定性。** 先按等墙钟和等总生成预算校准probe成本，当前证据不足以替代既有课程。

### RACE：区分响应塌缩与rubric失效，按奖励饱和切换训练角色

**`arxiv:2610.40096v1`；原始提交2026-10-29 14:00 UTC；72h回溯补录。** 低奖励方差可能来自答案雷同，也可能来自评分rubric无法区分答案。RACE用奖励方差/并列率及文本、embedding多样性分流：前者屏蔽，后者缓存成对答案供rubric训练；当高分低方差持续出现时切到rubric优化，改进停滞再切回policy，两个角色共享Qwen3-8B权重，冻结Qwen3-1.7B作judge（§5）。[固定版本全文](https://arxiv.org/html/2610.40096v1)。

实验用271K条Tulu3去重prompt、500次policy更新、2节点16×H800；八任务均分73.0，去掉gradient filtering降到70.8，固定50/50切换加routing为71.3（Table4、6，Appendix B）。但Appendix B明确RACE为3次独立run、baseline使用既有报告，500次policy更新也不等于等总算力：每次rubric更新还需512条rubric生成和1024次judge评分（Appendix E）。IFEval需移除think前缀后比较，本文给78.2对重现EvoLM的77.6；不可拼接不同评分口径。训练精度未披露，未核实公开训练实现，结论限单一共享骨干、固定judge和单轮任务。

**建议深入阅读奖励失效诊断，并保留复现。** 采用前需要同judge、同总算力和同IFEval处理的对照，不能仅凭八任务均分切换训练系统。

### Surprise Alone Learns Lies：保留旧知识与拒绝假知识是两种检验

**`arxiv:2610.39840v1`；原始提交2026-10-29 11:19 UTC；72h回溯补录。** 对流式短事实，仅挑surprising样本会把“新知识”和“假知识”一起学入。论文分开检查新颖性、冻结基座先验冲突、judge真假及旧事实回归；基座冲突先搁置，再由可信来源匹配开第二条接纳路径（§3–5）。[固定版本全文](https://arxiv.org/html/2610.39840v1)。

主要学生Qwen2.5-1.5B，BF16+rank16 LoRA，在16GB笔记本GPU，合成120人×5关系及CounterFact反事实，关键实验复核至3B。30%污染、5个held-out seeds、模拟judge准确率0.9时，加先验门把合法更新学入率从0.90压至0.24；全可信来源可恢复至0.98，却把保留旧事实从0.97降至0.91（Table6）。当攻击者给30%的已知事实谎言伪造匹配来源，学入样本污染达65/1000，比同条件judge-only的41更差；真实judge也会通过拒绝无来源真话获得表面安全（§5.6、5.8）。

官方`2369302`可对应`Student.prior_gap`的adapter-off评分及`run_system`的证据旁路/回滚；未独立运行。无检索基线，事实模板、≤3B与非自适应投毒限制明显，不能推导安全持续学习保证。[固定代码](https://github.com/Metalesaek/sura-metales/tree/2369302ffb687021b0fe6cca098942e05389a2af)。

<a id="inference"></a>
## 推理、量化与服务系统研究

**建议将其作为持续学习的失败用例阅读。** 同时验收新真知识接纳与假知识拒绝，并对来源伪造做压力测试。

### PDScope：prefill积极搬专家，decode优先避免PCIe过量预取

**`arxiv:2610.39216v1`；原始提交2026-10-29 02:37 UTC；72h回溯补录。** PDScope按阶段独立预测下一层专家，以跨层成本模型协调当前层CPU/GPU分配和下一层预取；执行前用真实路由重算，误预测影响性能而非改写应执行的专家。prefill倾向搬热专家，decode以CPU/GPU平衡目标限制I/O，并用预注册buffer、异步拷贝与静态图降低开销（§4）。[固定版本全文](https://arxiv.org/html/2610.39216v1)。

主实验为BF16 Mixtral-8×7B、Qwen3-30B-A3B、DeepSeek-V2-Lite、Moonlight-16B-A3B；A100+双6230R或RTX3090+EPYC7452，均PCIe4×16，GPU限24GB；ShareGPT输入128–4096、输出64、十次平均。相对不同baseline的最高prefill加速为2.5–10.3×、decode为2.1–6.1×，不是所有模型同一增益；不支持prefill/decode重叠，离线校准每模型仍需<30分钟（§5–6）。

**工件限制已可核实：** 官方10月30日提交`1771e79`包含AdaptSched和执行代码，但learned PhasePer未交付，改用下一层gate作用于当前activation；仅重现Qwen3单模型单请求，未提供论文五套baseline和多批次比较。代码可对应`plan_pdscope`与`PDScopeCudaRuntime.forward`，不能从仓库直接重现headline比值；README的参考CPU也换为5318Y。未独立复现。[固定工件及验收范围](https://github.com/nezha-Anonymous/PDScope/blob/1771e79b7ba8b0d74e59d310e9e9c03db5f1786d/docs/AE.md)。

**建议考虑复现已交付的单模型受限显存路径。** 多模型与论文headline比较须等待缺失组件和基线补齐，当前不直接作为部署倍率依据。

### Hessian统一PTQ：用输出损失敏感度统一变换、范围与舍入

**`arxiv:2610.40345v1`；原始提交2026-10-29 16:06 UTC；72h回溯补录。** HZ-PTQ从量化导致的最终分布KL构造二阶代理，再将层输出Fisher按token/feature作Kronecker分解；先定等价线性变换，再用同一敏感度代理选权重scale/rounding与activation clipping，避免各步骤分别最小化局部重建误差（§3）。[固定版本全文](https://arxiv.org/html/2610.40345v1)。

用DCLM-Edu的2048×2048-token序列校准，三种校准seed；测试Llama3.2-1B-Instruct、Llama3-8B、Ministral-8B-Instruct、Qwen3-8B的W4A4KV4及group32量化。Table2中Llama3-8B W4A4KV4的WikiText PPL由FlatQuant+GPTQ的7.19降为7.06，但六任务均分67.32对67.25，精度并非逐项全胜；group32的提升也较小（Table3）。增加梯度校准时间/显存，依赖二阶、Kronecker与误差独立近似；无端到端吞吐或特定硬件加速证据，未披露校准硬件、未核实公开实现（§5，Appendix B）。

**建议深入阅读统一PTQ目标，继续观察实现交付。** 先比较校准额外成本与任务质量，再讨论实际kernel和服务收益。

### 投机解码能耗审计：开启CUDA Graph后，MoE收益可能大幅缩水

**`arxiv:2610.40287v1`；原始提交2026-10-29 15:37 UTC；72h回溯补录。** 作者在SGLang、CUDA Graph开启、BF16权重/KV、greedy下测11模型，覆盖GQA、SWA、MLA、GDN混合；H100为主，3模型批量扫描用H200，固定512输出token、实际batch≤64，代码/对话/长文及英-罗曼什新闻五类负载（§3）。[固定版本全文](https://arxiv.org/html/2610.40287v1)。

投机与graph都摊薄launch开销，关graph会高估相对收益：Table3的Qwen3-Coder-30B-A3B单流DFlash从71%节能降至32%，GLM-4.7-Flash MTP从44%降至19%。同一DFlash在Coder模型batch64的代码省51%，换对话则耗能增加；接受率必须与验证宽度、模型和负载一起看（§4.1）。NVML累计计数只含GPU board，忽略host/网络/制冷，因此会偏袒host上运行的n-gram；没有动态到达、长上下文大批量或整机节能证明，部分drafter验证宽度也不相同。作者强调核对实际并发、chat template、重复prefix、GPU型号与采样窗口；未核实公开完整harness，未独立复现。

**建议复现测量方法。** 保持CUDA Graph、实际batch、输出长度和负载一致，并补host／整机功耗后再做成本判断。

### 浏览器GPU常驻投机：精确性只针对同一量化运行时的greedy输出

**`arxiv:2610.39761v1`；原始提交2026-10-29 10:14 UTC；72h回溯补录。** 把树验证、greedy接受、KV压实和上下文匹配drafter留在GPU，CPU只批量提交两个在途window；窗口之间根据最近接受数和实测成本调树宽。W4+A8的小batch核与减少CPU readback共同贡献速度，固定reduction顺序使不同树宽的结果等于本运行时单token greedy（§3–4）。[固定版本全文](https://arxiv.org/html/2610.39761v1)。

Chrome155在RTX PRO5000、GTX1650及同笔记本Vega上评估Qwen2.5/Llama七模型；11任务每模型220 prompt，部分另扩1060。Table2工作站Qwen2.5-1.5B从WebLLM的19.14降至2.23ms/token，但自家greedy已是6.05，不能全算作投机；所有WebLLM比较采用其成功完成的相同prompt子集。GSM8K均分BF16为60.0、两量化运行时均52.3，因此“lossless”不是对原BF16无损。冷启动另含加载/编译3–11秒及校准，首个短请求未必更快；仅greedy，无Apple/mobile实测，§4写明代码拟于发表后开放，本轮未见交付。

<a id="evaluation"></a>
## 评估与负面结果

**建议继续观察代码开放。** 有价值的是GPU常驻控制与冷启动代价分解，当前不宜把同量化greedy等价宣传为BF16无损。

### Dynamic Wave小实验：fused SDPA足以消除“线性显存优势”

**`arxiv:2610.39854v1`；原始提交2026-10-29 11:33 UTC；72h回溯补录。** Dynamic Wave用输入相关指数衰减和cos相位调制更新逐通道状态；同一递推可先批量计算系数，再作scan或简化循环。作者在10.94M、4层、TinyStories 500K故事、单epoch上比较纯Transformer、纯Wave与两层各半混合；T4 16GB、训练FP16、profile FP32（§III–VI）。[固定版本全文](https://arxiv.org/html/2610.39854v1)。

batch8、4096长、含LM head时，纯Wave峰值1404.3MB，对未融合attention的6601.6MB省78.7%，但对fused SDPA的1403.6MB没有优势；512长训练显存反而比fused基线高8.3%。只把指数/cos移出循环就提速2.53×（TableIII、V、VII）。混合模型PPL为5.95，纯attention6.93、Wave7.26；不过只有一个小规模短故事训练，未比较成熟Mamba/RWKV，未实现state-cached增量解码，训练墙钟还混入只有Transformer启用compile的差别。本轮未核实到公开实现，结论应限于基线与测量口径，不推导成熟线性架构普遍无益。

## 论文实际覆盖与待续查

本轮新拉取cs.CL/cs.AI/cs.LG/cs.DC/cs.PF/stat.ML六类`new`页，均仍显示10月30日且与前轮逐字一致；这是周一上午尚未滚动的发现面，不是“没有新论文”的证明。提交日期回溯API收到1543条metadata（返回声明1543），主窗API首次SSL失败，重试返回0条；`lastUpdatedDate`查询被返回页改写为`submittedDate`，不能作为修订覆盖。HF 10月30日39条、10月31日/11月1日空列表，11月2日400；TMLR、EMNLP及PMLR首页与昨日一致，OpenReview API及两个待审PDF仍403/挑战页。

本轮对12篇固定v1打开全文并按方法/表格/实验限制作选择性核读，去重排除10月29日已报的EnvSched、MuonGS，收录10篇；全部收益均为作者报告，未执行独立复现。旧pending、532条replacement、正式卷目录/更新历史与全部候选全文仍未穷尽；保留具体队列，不前移全局完整覆盖游标。当前最值得继续核验的是SCARL论文配方对应的确切run脚本、PDScope缺失predictor与多模型基线、OpenReview的Scion及Thinking Is Not Telling全文。

**建议深入阅读基线与profile消融。** 可作为线性架构实验检查清单，当前规模和实现不足以支持生产替换。


<a id="core"></a>
## 核心开源工程：重计算、长索引与服务完成边界

主窗 2026-11-01T00:57:10Z → 2026-11-02T01:42:47Z；72h 回溯自 2026-10-29T00:57:10Z。以下来自维护组织/社区一手 PR、重新打开的最终 changed-files 与完整 merge SHA；作者测试与本轮静态核验分开，本轮未运行 GPU/NPU。

### MFSDP v2 为 DP-outer 梯度归约分配独立 stream

[NVIDIA/Megatron-LM #6973](https://github.com/NVIDIA/Megatron-LM/pull/6973)，2026-11-01T07:51:18Z，main `037b883c9e7996b361943d27342ebd9757a5417a`。

HSDP/HFSDP 最后一个 microbatch 的 outer 归约原来排在同一 reduce stream 的 inner 归约后。`--overlap-outer-communication` 默认关闭；启用后每组 outer 等待本组 inner 完成，再与下一组 inner 重叠，复用持久 main_grad 及 optimizer-layout view；finish_grad_sync 同时 join 两个归约 stream。旧 unify_communication_stream 改为 overlap_allgather_reduction，默认保持 AG/归约分离。 **实践：**有跨节点 outer DP 的配方可显式试开，回归最后 microbatch、zero_grad 两模式和 caller-managed finish_grad_sync。 **证据限制：**仅实验 MFSDP v2；outer all-gather 尚未分离。作者两张 RTX A6000 的 2×1 mesh 不能验证两轴均有通信或跨机收益，不把单节点 collective 微基准写成训练提速。

### MFSDP 预取可记录首次真实执行顺序

[NVIDIA/Megatron-LM #8478](https://github.com/NVIDIA/Megatron-LM/pull/8478)，2026-11-01T02:17:55Z，main `2d7ea246f936e1f0a6b747202010b2d2a2808fcf`。

模块注册顺序与实际 forward/backward 不同时，按构造顺序预取会挑错下一个单元。FsdpContext(prefetch_order=PrefetchOrder.RECORDED) 首次普通 eager pass 按需 gather，并分别记录 forward/pre_backward；post_backward 结束录制，之后沿记录顺序使用既有预取预算。重计算不追加 forward 事件，已在 BACKWARD 的重复共享调用沿用原 guard。 **实践：**有反序模块或共享层时先核对录制结果和跨 rank 一致性，再启用 recorded prefetch。 **证据限制：**REGISTRATION 仍为默认；录制依赖普通 module hooks，combined 1F1B 绕开它们。后续各 DP peer 必须稳定重放同一调用顺序；首次录制异常应丢弃 context。作者双 A6000 测试不代表动态控制流支持。

### MXFP8 重计算可显式预取 row/column 两套平面

[NVIDIA/Megatron-LM #8432](https://github.com/NVIDIA/Megatron-LM/pull/8432)，2026-11-01T06:20:09Z，main `88312648a9dd30caedd517233e59c11d11915daf`。

backward 默认只 gather columnwise MXFP8；模块内部的重计算若不重入 owning module 的 forward hook，TE 会因缺 rowwise 数据失败。新增 SchedulePolicy.unshard_all_planes_in_backward=True，把该单元 backward（含前一单元的预取）的 phase 置为 None，一起 gather 两套平面；其他 dtype 不变。新增 reentrant/non-reentrant 重计算与无重计算逐位对照测试。 **实践：**使用 root-owned eh_proj 等权重的自定义 MFSDP 单元需显式配置，并检查额外通信/显存及 checkpoint 重计算边界。 **证据限制：**默认关闭，MCore adapter 本 PR 尚未自动开启；重新进入 forward hook 的重计算仍按需 gather。它不是所有 MTP/full recompute 配方已开箱修复的声明。

### ScheduleNode 以所有权转移替代强制缩小激活 storage

[NVIDIA/Megatron-LM #8302](https://github.com/NVIDIA/Megatron-LM/pull/8302)，2026-11-01T07:13:00Z，main `1dc6b4b936149e123063e3c97c48ec6dc616a118`。

NCCL EP MXFP8 dispatch 返回 GroupedTensor，没有可统一 resize_(0) 的 backing storage，旧节点手工释放方式不适用。free_input 路径用无 storage 的 ScheduleNodeInputWrapper 留住梯度入口，真实 detached tensor 一次性移入消费方 autograd 图；输出和 side output 的生产方用 GradientEdge 表示，避免持有输出 tensor。dispatch→experts 的局部作用域也收紧，GroupedTensor 依靠引用寿命释放。 **实践：**定制 combined 1F1B 节点需回归梯度边、shared/side output 与 offload，不能再假设 get_output 持有真实输出。 **证据限制：**普通必须留给 backward 的输入继续使用 detached leaf；细粒度 CPU offload 对普通 tensor 仍可强制释放。作者 DSv3 四节点、TE main、MXFP8 NCCL EP+PagedStashing 的基线另加 #7992，测得性能/显存近似不变，不能外推为提速。

### TP 交叉熵避免高置信目标的 softmax collapse

[pytorch/torchtitan #5661](https://github.com/pytorch/torchtitan/pull/5661)，2026-11-01T03:56:04Z，main `d86c77d609b4c35d71b4ed7a18e0c69f2a4f11fb`。

目标 logit 极大时，logsumexp(z)-z_target 与 p_target-1 会因舍入同时变成零，而非目标梯度仍非零。最终 loss.py 先从求和中排除目标，再用 softplus(log(non_target_sumexp)-target_shifted) 求 NLL；目标概率达到 1/4 后改用 -p_non_target 求目标梯度。保留低精度输入和逐行统计，backward 重算概率，仍为 forward 三次 TP all-reduce、backward 零 collective。 **实践：**高置信训练回归 FP64 参考、ignore labels、非均匀 vocab sharding、eager/compile，并留意 loss golden 的预期变化。 **证据限制：**正文早期称把 softmax-onehot 搬到 forward 后无需 backward 重算，与最终代码不一致，应以最终 diff 为准。TP>1 数值改变；测试是误差容限比较而非逐位一致。正文 T=16384/V=128256/TP2 时延表未充分固定硬件与最终版本，不引用为通用提速。

### RL peer 结果 drain 失败现在使整个 engine loop 失败

[pytorch/torchtitan #5716](https://github.com/pytorch/torchtitan/pull/5716)，2026-11-01T18:00:51Z，main `04fe1074fbe6af1869eedacf4fcf5b108aaa731f`。

rank0 后台接收/解析 peer DP 结果的任务异常原来独立退出，未完成 generation 和后续调用可永久等待。异常回调用 ResultDrainFailureMessage 关闭队列并唤醒主循环；rank0 先广播 CLOSE 让 peers 离开 collective，再抛原异常，复用 engine crash cleanup 失败未完成 generation、load 和排队调用。 **实践：**RL 生成服务可回归 drain task 异常、EngineLoop.wait 和 endpoint 拒绝，避免只监控前台 loop 存活。 **证据限制：**最终实现回调名为 on_result_drain_failure；不是自动恢复或重试。新增未知 peer completion 注入测试检查原 KeyError 传递及后续 generate 立即拒绝；本轮未运行分布式服务。

### MoE 识别按直接子模块修正自定义 block 的静默遗漏

[pytorch/torchtitan #5715](https://github.com/pytorch/torchtitan/pull/5715)，2026-11-01T03:37:44Z，main `06a4adfc8c9177d8cdca266d1c99390a7868afd6`。

负载均衡 hook、dense symmetric-memory 范围和 FSDP 曾依赖 TransformerBlock/feed_forward，漏掉 HF layer.mlp 或自定义 block 内的 MoE。三处改按直接 MoE child 判断，使 expert_bias 更新、dense 范围排除和 MoE 参数分片一致；新增 mlp 命名/非 TransformerBlock 回归。 **实践：**自定义块检查 MoE 所在层级，并观察 expert_bias 在 optimizer step 后真正变化。 **证据限制：**这是直接 children 的合同，不是任意深层嵌套 MoE 自动发现；apply_fsdp_to_decoder 取首个直接 MoE。作者 CPU 与 transformers backend 测试支持修复，未证明任意 out-of-tree 结构。

### RL GDN 的 Q/K 直接以 strided view 进入 l2norm

[pytorch/torchtitan #5690](https://github.com/pytorch/torchtitan/pull/5690)，2026-11-02T00:23:01Z，main `1d1fa9444d07b79351c9ed660b3ac4a5f52dfef9`。

packed prefill/mixed 步曾先复制 Q/K 为 contiguous，而下游 l2norm 已能读取 strided head row 并输出紧凑张量。VLLMInnerGDN._split_qkv 删除 Q/K 的两次 contiguous，仅保留 view；V 仍复制，因为 chunk kernel 对 strided V 会另行复制。 **实践：**复用该思路前确认 l2norm 的 stride 合同，并分别回归 packed 和纯 decode。 **证据限制：**作者 GB300、Qwen3.5-27B TP4/35B-A3B TP2、2K token、CUDA graph paired replay 下每层少 5.3–8.6µs，输出/SSM/conv state 在指定 eager及一次 replay 中逐位相同；普通 decode 路径与 trainer 未改。约 1% mixed-step 结果含其他 open PR，不能归给本 PR。

### KV 逻辑编码在 attention runtime 按层解析

[vllm-project/vllm #56155](https://github.com/vllm-project/vllm/pull/56155)，2026-11-01T15:03:45Z，main `8530a778f78dc7cd9a31e108ebc035c4b27ceb3c`。

全局 auto 无法表达 checkpoint 按层 KV 编码，而物理 uint8/FP8 storage spec 不能代替内核的读取编码。QuantizationConfig.get_kv_cache_dtype(prefix) 默认 None；auto 时 Attention 可接收该层编码，FlashInfer 从组内已加载层解析。fp8/fp8_e4m3 与 NVFP4 store 变体归一化，同组 E4M3/E5M2 等不兼容编码显式报错；sliding-window MLA merge 保留 kv_quant_mode。 **实践：**自定义量化配置可实现该 hook，并回归显式覆盖、组内冲突和 MLA merge；不要把它当成完整混合 KV 功能已发布。 **证据限制：**显式全局设置优先；GPU runner 与 AttentionGroup 未改。这仅是 #56156/#56116 前置，未启用 ModelOpt 混合 KV recipe。作者 GPU constructor/serving 验证仍缺，本轮仅静态核验。

### 共享 indexer top-k 随 PP 中间张量跨 stage 传递

[vllm-project/vllm #59961](https://github.com/vllm-project/vllm/pull/59961)，2026-11-01T06:47:36Z，main `63d3d39c84724daeaf098eae209fb3b6bdec02d2`。

GLM-5.x/DSv3.2 类层复用上层 top-k；PP 边界落在共享组内时，后一 stage 曾读到上一批次留在本地 buffer 的索引。公共 helper 检测 stage 首个 indexer 使用者是否复用前层，必要时让每个边界传 token-major int32 topk_indices，并在接收后复制到共享 buffer；覆盖 DSA、HY-V4 和 dots3 的对应模型路径。 **实践：**PP 精度回退时检查共享组与分层边界，并回归中间 stage 接收后继续转发。 **证据限制：**新增边界每步传 (num_tokens,index_topk) int32。MiniMax-M3 AMD 的 top-k 按 TP heads 分片，此处仍不支持并显式拒绝，需改 PP_LAYER_PARTITION；DSv4.1 已由既有 KV-sharing 分组限制。作者精度对照有指定 B300/MI350X 配方，ROCm 有运行波动，本轮未复跑。

### FlashInfer warmup 显式 autotune DSv4.1 indexer wq_b

[vllm-project/vllm #60543](https://github.com/vllm-project/vllm/pull/60543)，2026-11-02T00:05:40Z，main `72a340466f0b0c6358b50967f26616cd552d82ae`。

短上下文 dummy run 提前返回，漏过 wq_b MXFP8 GEMM；真实长步骤再 JIT 可能触发执行超时。既有 autotune context 内遍历 DeepseekV4Indexer，直接运行 wq_b，按同一 tuning bucket 建缓存；模型 forward 路径不增加 tuner 状态判断。 **实践：**开启 enable_flashinfer_autotune 后检查每 rank 缓存和 No tuned config 日志，并按实际长上下文/capture shapes 验证命中。 **证据限制：**作者对照主分支用 GB300/DEP2、修复用 GB200/DEP4，不能用它作硬件等价性能比较。其共同 22 bucket 的缓存证据显示 (1280,4096) 从 0→22 项、警告消失；CI 未运行该 SM100/checkpoint 路径，只有作者实测。

### DSv4.1 首层 HC 展开只读取一次输入

[vllm-project/vllm #64816](https://github.com/vllm-project/vllm/pull/64816)，2026-11-02T00:55:54Z，main `ea836290a7166f7d5b8efb1ae6ef1e2b2faa8b56`。

首层 expand().contiguous() 为 hc_mult 条 residual stream 重复读取输入。模型局部 Triton hc_expand 将每个输入元素读一次，再写到各连续 [M,hc_mult,H] 输出位置；保留 stride、空输入和 int64 索引处理，未改共享 mHC ops。 **实践：**可借鉴一读多写，复核 strided/odd/NaN/-0 与 graph replay；先测整模型收益再决定回移。 **证据限制：**作者 GB300、DSv4.1-Flash TP1/DP2/EP2、DSpark K5 的整 forward 增益很小且接近漂移；仅主张有限 32K 场景。动态 C32 输出本就不稳定，不能宣称并发生成逐位一致或泛化吞吐提升。

### Qwen GDN 恢复 CUDA in_proj_ba 重叠选择

[vllm-project/vllm #64999](https://github.com/vllm-project/vllm/pull/64999)，2026-11-01T18:27:26Z，main `b2d07f59c7a5909287f9ef5f037fd575ebdbbed0`。

CUDA GEMM dispatcher 改名后，重叠路径仍拿 default_unquantized_gemm 作 identity 判断，导致已实现 overlap 静默关闭。选择条件改成 cuda_unquantized_gemm，并补 Qwen3.5 GDN overlap 回归。 **实践：**升级过 CUDA GEMM 分发的 Qwen3.5 配方应检查 _gemm_impl 与 overlap 选择，不只看配置已开启。 **证据限制：**仍限 CUDA、UnquantizedLinearMethod，且无 ubatching、无 LoRA 的既有 guard；未提供新的端到端性能数字。

### KV offload 事件保留请求上下文，并减少 GC 扫描对象

[vllm-project/vllm #63927](https://github.com/vllm-project/vllm/pull/63927)，2026-11-01T18:27:47Z，main `b8c04dd3b455affd8d91f0ab436b815491c9af38`；[vllm-project/vllm #63695](https://github.com/vllm-project/vllm/pull/63695)，2026-11-01T07:34:53Z，main `17e675965d54bef8a3eb77ea92c6bae195ef793c`。

异步 FS/OBJ store 完成后需要原请求上下文；同时大 secondary tier 的长期 event metadata 会扩大 Python full-GC 扫描停顿。#63695 保留完整 TransferJob，成功 raw store 事件携带 ReqContext，CPU及级联 tier 走同一合同；#63927 把 metadata 改为 gc=False 的 msgspec.Struct，以 bitmask 代替 residency set，保留重存时多 tier residency 的并集。 **实践：**事件消费者区分 raw/internal 与公开事件；大 tier 部署用真实 gc.callbacks/ITL 检查收益，并回归 reoffload 后 secondary residency。 **证据限制：**前者仅内部运输，公开 BlockStored/Removed payload 和 removal 语义未改；inventory/removal 无新 context。后者时延与 7% req/s 来自 mock model/GPU copy 的 CPU 模拟，不能写成真实 GPU 服务提速；gc=False 结构必须持续避免引用环。

### CUDA grouped/Muse router 在融合 kernel 内屏蔽 padding

[vllm-project/vllm #64955](https://github.com/vllm-project/vllm/pull/64955)，2026-11-01T07:06:26Z，main `2a4792ddcc4ba2091aaf743358866c86bc53bf75`。

grouped routing 需要额外后处理 mask，而 Muse prepared TRT-LLM 路径直接调用 router，曾漏传图 padding mask。CUDA 四类 grouped top-k kernel 接受可选 mask，padding 直接写 expert id=-1、weight=0；Muse prepared 路径仅在 token layout 匹配时传 mask，Triton 用最终 where 写零避免 NaN×0。XPU 保留旧 ABI 与后置 mask。 **实践：**对不满 graph bucket 和 nonfinite padding 做主动输出/route parity 回归，并保留 VLLM_MOE_SKIP_PADDING=0 对照。 **证据限制：**依赖下游 expert 支持 -1 sentinel、backend opt-in 和 layout guard。作者 GB200 单算子测试通过，但 Muse timing 用 instrumented graph，诊断 harness 在 NCCL graph teardown hang 后仍标 FAILED；不引用为 stock 模型提速或完整成功退出。

### 超大权重与 KV 池的寻址乘法升为 int64

[sgl-project/sglang #47066](https://github.com/sgl-project/sglang/pull/47066)，2026-11-02T01:41:07Z，main `8088adb4d31a5cf62a9fe89b41d527fea8885a86`；[sgl-project/sglang #47029](https://github.com/sgl-project/sglang/pull/47029)，2026-11-01T04:13:21Z，main `8d12b65e541ac6d4513a8fad03fed877557702ca`。

gpu_tensor_hash 的 pid*TILE 在超过 2^31 个 32-bit word（8GiB）后溢出；extend attention 的 int32 slot×stride 也可越界或静默读错。#47066 在乘 TILE 前提升 pid；#47029 在两种 extend kernel 读取 KV slot 后、任何 page/stride 地址运算前提升，涵盖 int32 调用者。 **实践：**大 engram checksum 与长期运行大 KV 池分别回归；内核只把最终地址 cast 为 int64 仍不足，必须在乘法前提升。 **证据限制：**hash 不是 forward 热路径；extend 风险阈值随 stride，8 KV heads×128 dim 对应约 2,097,152 token 的 slot，不能把所有长上下文都说成受影响。aiter legacy verify/NGRAM 等 int32 index 路径是作者复现入口，Triton 自有 int64 buffer 原不受影响。

### KDA/GDN recurrent kernel 清零图 padding 输出尾部

[sgl-project/sglang #46225](https://github.com/sgl-project/sglang/pull/46225)，2026-11-02T01:34:57Z，main `a8c69d0f2029da0a159976df63b00228806a24c6`。

batch-8 图只有七个活跃请求时，第八行未被 recurrent loop 写入，旧 NaN 可经下游污染有效输出。varlen kernel 由最后一个 sequence 的 program 将 cu_seqlens 末端后的 output tail 写零，不新增 launch，保留真实 token 输出与 state slot 更新。 **实践：**连续 replay 不同 active count，主动 poison padding 并同时检查有效输出和 recurrent state。 **证据限制：**针对该 fused sigmoid-gating recurrent 输出尾部；不等于所有 padding/NaN 来源已消除。新增同一图改变活跃 token 数的 KDA/GDN 测试，本轮未运行 GPU。

### replicated linear 输出退出全局 TP symmetric-memory 窗口

[sgl-project/sglang #47744](https://github.com/sgl-project/sglang/pull/47744)，2026-11-02T01:27:37Z，main `009b167f109a396edaf6af801034b99f5010afca`。

RowParallelLinear 的 replicated 模式没有 TP group，但仍进入全局 TP 对称分配窗口，给完整本地输出施加无关 collective 分配约束。forward 在 self.tp_group 为 None 时禁用全局 TP symmetric-memory context；attention-TP 和真正 sharded linear 路径保留。 **实践：**replicated shared expert 配方检查分配池归属，并让 replicated batch size 跨 rank 不同以发现隐含对称假设。 **证据限制：**作者 CPU 测试与八张 B300 的 12 例/rank allocation+AllReduce smoke 支持该边界；不是完整 Kimi-K3 模型精度或吞吐测试。

### torch.compile 下让 K/V 使用独立 allocation

[sgl-project/sglang #47670](https://github.com/sgl-project/sglang/pull/47670)，2026-11-01T05:04:37Z，main `2d8462fef2e07d8e8759405f77026a1b7391c6e9`。

K/V 合并为同一 storage 的两个 view 后，store_cache 同时 mutate 两输入，会令 AOTAutograd 经 synthetic base clone/copy 整个每层 KV 池。MHATokenToKVPool 仅在未开 torch.compile 的 CUDA 路径保留 fused K/V allocation；compile 时分别分配，ROCm 既有独立分配也不变。 **实践：**升级 KV layout 后用全新 inductor/AOT cache 重测，profile 是否有全池 as_strided_clone/copy，再比较热 cache。 **证据限制：**作者在 Qwen1.5-MoE A2.7B、B200/H100 的冷 compiler cache 复现；旧 cache 可掩盖 alias 问题。只支持这条 allocation/functionalization 因果，不能把报告的 12–23 倍退化当成普遍 compile 性能。

### OOT fused op 获得独立的 compile 注册入口

[sgl-project/sglang #47229](https://github.com/sgl-project/sglang/pull/47229)，2026-11-02T00:39:28Z，main `a2029238dea8d858868c9e3213eefb76aa1c1f12`。

插件 eager 实现原以无类属性的 bound method 绑定，compiler guard 序列化按名字查找时可失败，也缺独立 compile dispatch 注册。新增 register_oot_compile_forward(op_cls,fn,platform_key)，exact op type/平台的注册函数优先；实例保存原函数，由类上固定 _forward_oot/_forward_oot_compile 传 self 调用。 **实践：**OOT 插件分别注册 eager/compile 实现，回归 enter/leave、guard 保存与 fullgraph tracing，避免将此理解为任意设备 kernel 自动可编译。 **证据限制：**注册需在首次使用/enter_torch_compile 前完成；只注册不会改变任意外部 torch.compile 调用。Dynamo 仍会追踪函数体，fullgraph 下不支持操作会失败，设备图捕获需插件自行满足。

### 先处理需要同步的旧结果，再规划下一批请求

[sgl-project/sglang #43900](https://github.com/sgl-project/sglang/pull/43900)，2026-11-01T21:49:38Z，main `d3bf0b4e52115f87ac41b6bca51915e9dc6baab2`；[sgl-project/sglang #47716](https://github.com/sgl-project/sglang/pull/47716)，2026-11-01T21:54:37Z，main `1c0dc0293d4cce8d13872080b2a20936dffb2232`。

overlap loop 原在排好下一批后才提前处理上一结果，刚结束的请求仍在新批中发射；NGRAM grammar 与禁用 prefill overlap 可触发。#43900 统一 launch hook，并按每个 Req 记录已发射而结果未处理的 inflight_steps；#47716 把 early-result 判定和处理移到 get_next_batch 前，NGRAM 按每请求 count>1 判断是否拼接前轮 accepted tokens。 **实践：**回归 EOS/stop、extend→decode/mixed、DP 转换和 grammar 状态推进，并检查已 finished 的请求不再进入下一 launch。 **证据限制：**inflight_steps 本身不是 KV 内存释放修复；本轮只是记账与结果/规划次序。原开关保留废弃别名，新 SGLANG_DISABLE_OVERLAP_AFTER_PREFILL 在任意 extend 后可提前处理，不再要求下一步也是 prefill；PD overlap+spec+grammar 仍不支持。

### P/D bootstrap 修复优化模式副作用与故障清理边界

[sgl-project/sglang #47727](https://github.com/sgl-project/sglang/pull/47727)，2026-11-01T09:17:38Z，main `43881a719e42165ff987fb0273a6bf12eaeb7662`；[sgl-project/sglang #47725](https://github.com/sgl-project/sglang/pull/47725)，2026-11-01T07:57:52Z，main `76066d6908411bf9afcd740cfac61fdf0d5d8050`；[sgl-project/sglang #47704](https://github.com/sgl-project/sglang/pull/47704)，2026-11-01T07:22:12Z，main `3edea93cbc0d857026253df0535a4d89791032a3`。

finalize_bootstrap 被写在 assert 中，python -O 会删掉初始化；server 线程尚未建立 loop/runner 即 close 会抛 AttributeError；故障地址前缀匹配还会误删相邻端口连接。分别将 finalize 调用移到 assert 外、先初始化可空 loop/runner 并仅清理已创建资源、按完整 bootstrap_addr 加下划线分隔符驱逐连接。 **实践：**有 -O 启动或快启停的部署检查 sender 初始化；故障注入时同时放 host:3000 与 host:30001 验证健康连接保留。 **证据限制：**前两项不是新的 bootstrap 重试/恢复协议；相邻端口修复限 connection-pool key 匹配。#47727 未新增测试；另外两项有对应 unit regression，本轮未跑 P/D 服务。

### KV events 支持多 DP 的 ipc 端点并容忍满队列退出

[sgl-project/sglang #47732](https://github.com/sgl-project/sglang/pull/47732)，2026-11-01T09:29:30Z，main `c757c4bb7c55d1eb69fdb2bdc6e6fb8507acb7a0`。

ipc:// 在 DP rank>0 构建 publisher 时不被 offset helper 接受；shutdown 的 bounded queue 满时抛 queue.Full 并跳过 socket 清理。ipc 路径加 _dp{rank} 后缀；_running=False 后允许 sentinel 入队失败，继续既有 drain/cleanup 路径。 **实践：**多 DP+IPC 启动与拥塞时 shutdown 都应回归，并核对实际 socket 回收。 **证据限制：**只修端点命名与满队列退出，不保证任何慢消费者下事件不丢或持久交付。每项各有 unit test，本轮未执行 ZMQ 压力测试。

### P/D decode 生成 KV 写回 linker，跳过已委托 prefill 的前缀

[sgl-project/sglang #46395](https://github.com/sgl-project/sglang/pull/46395)，2026-11-01T12:52:16Z，main `cb0a7a953fa6b8409b67959d445c95c802e50c14`。

只有 prefill 写外部缓存时，下轮缺上轮 output KV；decode 也全写又会重复上传 prompt 页。bootstrap 宣告 stores_to_linker，decode 每次预分配记录 raw-token skip 边界；UnifiedRadixCache 转成 page-aligned radix key（含 bigram），linker 保留本地树插入但裁掉该传输前缀，并轮询 offload completion 释放 node lock。 **实践：**需要外部 store、decode radix+linker 和两角色协同时可试用，回归 rebootstrap、页对齐、SWA/Mamba side pools 与写失败后的 miss。 **证据限制：**完全跳过的节点仅标记本地 stored，它没有等待 prefill 持久化 ACK。外部存储仍 best-effort，prefill 写失败可能让下轮重算；不能称 exactly-once 或可靠持久化。作者 Mooncake 多轮命中实验与本轮静态核验分开。

### DCP 可显式使用 breakable prefill graph，前缀命中仍回 eager

[sgl-project/sglang #38943](https://github.com/sgl-project/sglang/pull/38943)，2026-11-01T09:09:11Z，main `928e285bc10fc4119277e73ee2d3d29b6412bec6`。

显式 breakable 加 DCP>1 时 capture dummy batch 缺 attn_dcp_metadata，MLA extend gather 会解引用失败。capture 建 bucket-stable DCP gather buffer，replay eager 规划 live indices 并复用该 buffer；KV assembly 成为读取 live forward context 的 graph break，fresh prefill 无 prefix 时只拷本地 K 而跳过 collective。 **实践：**显式选择 breakable 后观察每次 admission/fallback 与 buffer 地址，并回归 prefix hit、draft speculation 和超 bucket prompt。 **证据限制：**只允许 fresh prefill replay；prefix-hit 和 chunked continuation 明确回 eager，默认 auto-disable 规则未放宽。作者 Kimi-K3、cutedsl_mla、DCP8/两节点的 TTFT 表未固定完整硬件，故不写通用提速。

### HiCache 增加有容量边界的本地 fast_file 存储

[sgl-project/sglang #39880](https://github.com/sgl-project/sglang/pull/39880)，2026-11-01T04:41:51Z，main `cb7eaa2ea0ed0903ab9ffac0de57bc8c4f02a4fd`。

参考 file 后端串行 staging、非原子写且无磁盘容量约束，实际本地 NVMe tier 需要控制 I/O 和逐出。fast_file 对 page-first host buffer 使用 readv/writev、bounded read workers，其他布局保留 staging；临时文件+rename 发布完整页，按 model/parallel/host-layout namespace 隔离，并以双水位 LRU/max_size/min_free_space 限制磁盘。 **实践：**本地 L3 可从小 cap 和故障注入开始，验证短读、并发同 key 写、重启临时文件清理、磁盘满及 reload 后生成一致。 **证据限制：**沿用 file 的原始 page 格式但 namespace 独立；拒绝 unified key scheme，以及 TP>1 replicated MLA namespace 下 rank-sharded side pools。sidecar 与 KV 可独立逐出。作者局部 NVMe/模型实验缺完整硬件条件，AMD/NPU未跑，不复述倍数为普遍收益。

### gfx950 MiniMax 内核层落 main，服务集成仍待合并

[sgl-project/sglang #46690](https://github.com/sgl-project/sglang/pull/46690)，2026-11-01T22:36:01Z，main `03f90543787cb4ff35892054d350cee44193aca1`；[sgl-project/sglang #46691](https://github.com/sgl-project/sglang/pull/46691)，2026-11-01T23:16:09Z，main `e37c1fb020acec0efa386f83d35d2a03d887132a`；[sgl-project/sglang #46692](https://github.com/sgl-project/sglang/pull/46692)，2026-11-01T23:15:43Z，main `fdf2266c8861597690e978e7346d140856220689`。

MiniMax-M3.1-Flash AMD 系列把 attention、MoE/GEMM 和通信拆成可独立审查的内核 PR。#46690 增加 packed NVFP4 sparse decode/prefill、verify top-k 与 fp8 O epilogue；#46691 增加 sort+MXFP4 quant、router/FP8 producer 与受 aiter signature guard 保护的 runtime patch；#46692 增加 gfx950/TP4/BF16 JIT quick all-reduce，INT4 请求低于默认64MiB改走INT8。 **实践：**gfx950 用户先对照精确 shape、Gluon CDNA4与aiter接口；只部署内核层时区分既有 M3 调用点和尚未合并的 M3.1 接线。 **证据限制：**重新打开的 #46693 服务接线和 #46694 indexer CP 仍 open，不能写为完整 MiniMax-M3.1 服务已支持。NVFP4 decode 限 block128、每KV头16个Q头、每层cache<2GiB且无sink/HiSparse；quick AR默认缓冲约4GiB/GPU。作者部分收益来自集成分支且 DSpark acceptance 人工固定，不作main性能结论。

### 分支增量：MiniMax BF16 融合 QKV 保留在线更新 view

[sgl-project/sglang #47612](https://github.com/sgl-project/sglang/pull/47612)，2026-11-01T19:51:31Z，sglang-miles `fe89fc7a3df5a895b4e18117fbd20c1015e9e456`。

融合 qkv_proj/index_qkv_proj 后释放原 weight，会令普通 tensor loader 的首轮 RL 同步写到空 tensor。BF16 原 weight 改成 fused weight 的 split views，loader 原地更新同一 storage，已捕获 graph 可继续读相同地址，未额外保存副本。 **实践：**使用 miles 集成分支且 BF16 rollout 才考虑回移，逐步做 weight-equal 与 train/rollout logprob 回归。 **证据限制：**实际 base 为 sglang-miles，merge SHA 不在本轮 main inventory，不能标成主线能力。MXFP8 因 scales 重排仍释放旧权重，在线更新依然不支持；作者四层 MiniMax/miles smoke 不代表完整模型零 logprob mismatch。

### diffusion 动态批处理保持每请求 rollout 轨迹与 conditioning

[sgl-project/sglang #46274](https://github.com/sgl-project/sglang/pull/46274)，2026-11-01T23:59:04Z，main `38d9e4a3fe4a493b0ae468477a274e83defda81a`。

split OutputBatch 曾丢失 rollout_trajectory_data，concat 又只留下第一请求 denoising_env，可让后续 trainer 用错 prompt conditioning 而静默算错 importance ratio。统一 range slice，并递归拼接/切分 per-row conditioning；共享 RoPE cache keys 保持共享，txt_seq_lens/img_shapes 列表按行处理，scheduler split 返回每请求对应轨迹。 **实践：**batching-max-size>1 的生成式RL应以不同 prompt 混批，核对轨迹、正负 conditioning 和重新计算 logprob 的逐请求归属。 **证据限制：**针对当前 conditioning tree 与显式 shared/per-row key 集合。作者仅 dataclass CPU 失败/通过对照，scheduler wiring 测试未本地运行；没有完整 GPU RL 收敛证据。

### diffusion LoRA merge cache 按最近采用顺序逐出

[sgl-project/sglang #47605](https://github.com/sgl-project/sglang/pull/47605)，2026-11-01T23:51:48Z，main `e8bf40b8eed8605e9e006857cb8562a076fa4bf7`。

磁盘不足时原 merge cache 直接放弃持久缓存，且 is_complete 后被另一实例删除会让 get 抛异常。采用 store 时刷新目录 mtime，写前按 mtime 逐出其他旧 store，直到有 expected_bytes×headroom 空间；并发删除导致的 FileNotFoundError 变为 cache miss。 **实践：**大量 LoRA 切换部署关注命中率和磁盘 free-space，并测试多实例并发逐出后重新合并。 **证据限制：**仍是 best-effort 磁盘缓存；空间仍不足则退回匿名内存，未引入跨进程事务或硬容量保证。作者8项CPU测试，未给服务吞吐结论。

### H3 Ulysses pipeline 跳过仅用于对齐的 attention 段

[sgl-project/sglang #47566](https://github.com/sgl-project/sglang/pull/47566)，2026-11-01T02:03:23Z，main `5032f9881bad04f2fd8503e0346742b066186b3a`。

packed sequence 的尾部对齐行被放在独立 segment，cuDNN varlen 每个 head group 仍为它单独发射 attention/copy。仅识别 DynamicCudnnSDPAImpl 且 cu_seqlens=[0,used,total] 的纯 padding tail，真实行照常计算，可 forward_into 时直写输出，尾行归零。其他布局继续原 varlen。 **实践：**视频模型自定义 packing 可检验这一不变量，并比较真实输出而非只数 launch。 **证据限制：**不是任意 ragged batch 都能删除尾行；依赖 tail 不与真实 token 互相注意、后续算子逐行。作者4×B300、叠加另一 three-waves PR 的视频结果逐位相同，时延不可归为独立主线总收益。

### PP 首阶段同节点时保留多模态 shared-memory transport

[sgl-project/sglang #44574](https://github.com/sgl-project/sglang/pull/44574)，2026-11-01T05:53:45Z，main `07efafb546364e7ddee1b46f9c4ae01da417f90b`。

nnodes>1 一律 pickle decoded image features，会让 tokenizer event loop 因大图请求阻塞；部分逐GPU launcher 即使单机也这样报告 nnodes。若无 Ray、DP=1 且 nnodes<=pp_size（TP组可留同节点），允许 tokenizer→PP0 的本地 CUDA IPC/shared-memory；跨节点TP或DP继续原fallback。 **实践：**大图PP部署先验证实际进程放置与 transport mode，检查健康接口 event-loop 延迟，并单独衡量后续 stage 传输。 **证据限制：**只改 tokenizer 到首阶段；PP0 转发后续 stage 仍 pickle。正文延迟结果叠加 #44575/#44576，不作为本 PR 独立速度证据。

### 覆盖与待核范围

六仓库 release 集合均枚举到回溯下界之前，本主窗无新 release；Megatron core_v0.20.0（10-29）已在历史日报，未重复报道。已合 PR 搜索共 645 条、main 提交 624 条，搜索分页均完整且 incomplete_results=false。主窗 verl/slime 未发现新提交或已合 PR。vLLM #61667 仍 open、head 未变，#61532 仍 open，未作重复动态。

本节完成 32 组、39 个新 PR 的最终文件与语义边界核查（38 个 main、1 个集成分支）；其余具体候选保存在 core/pending-candidates.json。完整枚举不等于全部语义已审：旧 217 项当前待核队列、前轮继承 277 项及更早 10-05 起区间均保存在 inherited-backlog.json；未宣称清空旧账或推进完整语义游标。

<a id="extended"></a>
## 扩展开源工程与硬件软件协同

以下状态以本轮重新打开的最终 PR 元数据及完整 changed-file 分页为准；主分支合入、发布支线合入和正式 release 分别标明。没有在本地运行 GPU/NPU/多机实验。大体量变更按所读机制总结，未读路径继续保留待审。

### Inductor：可单独关闭 FMA 收缩，补上确定性模式的一处边界

**PyTorch；[main `e6b1fb9c8c59`](https://github.com/pytorch/pytorch/commit/e6b1fb9c8c59513c03d55538fe2a3acc92eeb22c)（2026-11-02T01:15:48Z，关联 PR #202218）。** PR 的 `merged_at` 本轮仍为 null，以上依据默认分支真实提交，不把 ghstack 的开放 PR 写成已合 PR。

问题是同一逐点表达式在 XBLOCK=128/256 时可以选择不同 FMA 组合；计时选配置的噪声于是变成末位数值差异，既有 deterministic 模式并不固定全部逐点配置。新增 `config.triton.enable_fp_fusion`／`TORCHINDUCTOR_ENABLE_FP_FUSION`，默认仍开启；关闭时将 `enable_fp_fusion=False` 传给生成的 Triton kernel，避免该类乘加收缩依配置变化。它没有自动改变 `torch.use_deterministic_algorithms` 的含义，也不等于整个训练栈的逐位确定性保证。

作者在 H100 的一个 16,384 元素表达式中看到 4,042 个结果末位不同，并在 A100 多卡与64卡训练定位到同类现象；这是局部反例和调试证据，非所有模型的误差或性能统计。**建议需要逐位复训的团队先做开关 A/B，记录速度与数值代价。** 已读 `torch/_inductor/config.py` 与 `codegen/triton.py::triton_meta_common` 的真实 main diff；该选项也不同于会改变更多精度行为的 `emulate_precision_casts`。

### 跨 rank 编译缓存：移除编译卡编号，同时在载入时恢复当前设备

**PyTorch；[main `44f99009d01f`](https://github.com/pytorch/pytorch/commit/44f99009d01fbdd4013d40499b475677c865bb43)（2026-11-01T01:53:17Z，关联 PR #196153）；[main `1f4d2d5e820d`](https://github.com/pytorch/pytorch/commit/1f4d2d5e820dd870626dca4f9603e3a49879dbde)（2026-11-01T02:40:58Z，关联 PR #196154）；[main `0f53a707a91a`](https://github.com/pytorch/pytorch/commit/0f53a707a91af21a7b8f519b354bcf06268a941a)（2026-11-01T21:34:42Z，关联 PR #197384）。** PR 的 `merged_at` 本轮仍为 null，以上依据默认分支真实提交，不把 ghstack 的开放 PR 写成已合 PR。

`compile_on_one_rank` 原来把 example-input 的设备编号带入 FX/AOT 缓存键，使不同 rank 互相 miss；仅去掉键又会让载入的 CUDA graph 使用编译 rank 的 device、stream 和 pool。现在仅在 CooR 的当前单 accelerator 条件下去掉键中的编号，`CompiledFxGraph.device_index()` 在使用时取当前卡，repro storage 同样省掉固定编号。多设备普通编译保留原有标识，不能全局删除 device index。

另一修复让普通编译下不带 index 的 `synchronize()` 在 tracing 时解析调用者当前卡并安装 guard，避免总同步 device 0；CooR 路径仍留到各 rank 运行时解析。代价是普通路径按当前设备专门化，切多张卡可能增加重编译并触及 recompile limit。**建议多卡缓存共享和复现脚本使用者回归跨设备载入、graph capture 与同步目标。** 作者提供双卡／缓存命中测试；16卡键差异调查是定位证据，没有整训启动时间提升的可比表。

### ROCm INT4 matmul：wave32 WMMA 路径重新落到 main

**PyTorch；[main `80a60b3ec993`](https://github.com/pytorch/pytorch/commit/80a60b3ec993a1da4a5c88cac9a595af51b819b3)（2026-11-01T01:55:01Z，关联 PR #201248）。** PR 的 `merged_at` 本轮仍为 null，以上依据默认分支真实提交，不把 ghstack 的开放 PR 写成已合 PR。

原 `_weight_int4pack_mm`／pack 路径依赖 wave64 MFMA，RDNA/gfx1250 被拒绝。新 main 提交按 gfx11/gfx12/gfx1250 选择 wave32 WMMA，保持 packed-weight 形状，分离 CUDA 与 ROCm 实现；部分架构将 scale/zero 修正移到每个量化组的 FP32 累加后，gfx1250 保留更快的逐权重反量化。还用普通 C 替代 LLVM 无法正确识别 hazard 的内联 asm，避免 gfx1250 覆盖 WMMA 输入寄存器。

**版本沿革必须保留：10月29日首次落地，10月30日撤回，11月1日 `80a60b3e` 再落地；本条是重新落地的增量。** 作者报告六类 GPU 单测及950配置对照扫描，分组重排会改变部分架构的末位结果，不能承诺与旧 kernel 逐位一致；本轮未实测，不把小算子 timing 当端到端 tokens/s。建议 RDNA/CDNA 部署者核对实际构建包含本提交，再测精度和小 batch 权重量化负载。

### Transformers 修复 PaliGemma 2 丢失的最终 logits softcap

**huggingface/transformers；[#50156](https://github.com/huggingface/transformers/pull/50156)（2026-11-02T01:00:09Z，`main`，`3a667d7cb51b`）。**

自 v4.52 从 CausalLM 包装改为 AutoModel 加独立 `lm_head` 后，PaliGemma 2 的 `final_logit_softcapping=30` 没有执行。现在在 `lm_head` 后应用 `cap * tanh(logits / cap)`；PaliGemma 1 没有该字段时保持原行为。作者的 CPU 复现从最大 logit 72.03／loss65.31，修至与 Gemma2 参考的29.51／31.77匹配，并加失败→通过的回归测试，未跑 GPU／slow suite。

**建议相关推理及微调链路优先回归。** 修复会改变采样和 loss，也包括此前以错误路径微调的 checkpoint；它恢复模型公式，不证明旧微调权重会自动恢复原始质量。本轮没有宣布它已进入 v5.22.1 或更早 release。

### DeepSpeed：CPU 预算遵守 cgroup，HybridEngine 图捕获走设备能力接口

**deepspeedai/DeepSpeed；[#8930](https://github.com/deepspeedai/DeepSpeed/pull/8930)（2026-11-01T16:12:20Z，`master`，`2d80cedd2bda`）；[#8933](https://github.com/deepspeedai/DeepSpeed/pull/8933)（2026-11-01T15:51:02Z，`master`，`ab7fd98faac9`）。**

两个不同问题分别落地到 `master`。CPU accelerator 的 `total_memory/available_memory` 从宿主 RAM 改为进程到根 cgroup 路径中最紧的限制／可回收后余量，处理父级共享占用与子树挂载；默认开启，`DS_CPU_RESPECT_CGROUP_MEMORY=0` 可回退。影响 CPU ZeRO/自动调参的预算依据；这不代表所有 accelerator 的 CPU offload 内存规划都被统一改造，仍需在实际容器限制下检查。

HybridEngine 把 CUDA 专用门禁改为 `create_graph/capture_to_graph/replay_graph/is_current_stream_capturing` 能力接口，新增 `enable_graph_capture`，旧 `enable_cuda_graph` 作为弃用别名迁移；Ascend attention 改用图模式 v3，并按需扩展进程内多个 engine 的 KV 槽。ZeRO3、释放 inference cache、inference TP>1、不固定生成长度等拒绝条件仍在，捕获之后超出长度也不能安全随意退 eager。作者提供单 Ascend、OPT1.3B、FP16、256输入/128输出的早期容器测试；软件栈及新旧运行路径分开披露，不作通用加速承诺。**建议容器训练先校预算，NPU 训推切换先做固定长度与失败路径回归。**

### TensorRT-LLM：encoder-decoder 的 cross KV 必须按模型 dtype 分配

**NVIDIA/TensorRT-LLM；[#19864](https://github.com/NVIDIA/TensorRT-LLM/pull/19864)（2026-11-01T22:36:02Z，`main`，`0fd58c90ede9`）；[#20900](https://github.com/NVIDIA/TensorRT-LLM/pull/20900)（2026-11-01T03:01:49Z，`main`，`7a62522fb820`）。**

cross-attention 实际写未量化 K/V，却继承 self-attention 的 FP8/NVFP4 分配成本，可能越界或错输出。修复通过配置副本清除 cross 池的量化标记，保留 decoder self 池量化；V1 各池按自身 byte share 和真实 token 成本算容量，避免重复套 free-memory fraction，显式 max_tokens 上限仍保留。正确分配后固定预算可容纳的 cross tokens 可能下降。

作者正文明确本 revision 的 native/GPU 复现与 CI 待验，76项隔离检查及更早86/768/48对照均不能替代 CUDA 验证；末尾勾选框与正文冲突，以具体测试说明为准。另一个 PR 在构造 Python/C++ micro-batch scheduler 时，提前拒绝 FORCE_CHUNK 下 `max_num_tokens < chunk unit`，避免服务接受请求后在 executor 循环才失败；KV cache V2 与其他策略边界不同。**建议 encoder-decoder 量化 KV、混合状态块复用部署优先检查容量和启动负例。**

### TensorRT-LLM OpenEngine：限制图像准备并复用 prefill 元数据

**NVIDIA/TensorRT-LLM；[#21037](https://github.com/NVIDIA/TensorRT-LLM/pull/21037)（2026-11-01T16:05:36Z，`main`，`309426dd94d2`）。**

每 frontend 默认最多四个 media preparation，取消请求仍占用 admission，直到后台 executor 工作结束。Qwen vision 把展开 token、mRoPE 与多模态缓存标识经现有 KvSessionRef 属性交给 decode，减少重复取图／预处理；versioned JSON 限2MiB，未知版本走普通准备，已知版本的坏元数据拒绝。两端须匹配模型/tokenizer revision，fingerprint 不等于认证或权重校验。

作者10月26日候选 `88caa8cba` 的 Qwen3-VL-2B、BF16、TP1、RTX5880 Ada48GiB、224/896/1792图像、并发1/4，两次反序试验共900实测请求中，1792/C4 的 P/D TTFT1919.9→1578.7ms，吞吐1.636→1.951 req/s；224/C1基本持平，aggregate 的较大图像反而增加2.1–4.8% TTFT。该测试用旧 compiled runtime 加 Python overlay，**最终刷新没有重跑 GPU/perf**，只补隔离 CPU 检查。建议复现取消后的 admission 寿命和同机 P/D 预处理次数，再测自身图片分布；不能把这张旧候选表当当前编译 wheel 的普遍提速。

### TensorRT-LLM AgentX 支线优化并入 main：显式暴露有损与部署选项

**NVIDIA/TensorRT-LLM；[#20815](https://github.com/NVIDIA/TensorRT-LLM/pull/20815)（2026-11-01T06:20:13Z，`main`，`5d86481f60bd`）。**

223文件的变更把 Qwen3.5-397B NVFP4／GB300 P/D 的注意力、低 M GEMM、MoE、通信、投机、前端及调度路径带回 main。值得读的是配置归属：`mnnvl_allreduce_fp8_payload` 明确为有损 E4M3 通信；`draft_lm_head_dtype=mxfp8`／`draft_moe_dtype=nvfp4` 改的是 draft 数值与接受率；embedding replication 以显存换时延，frontend affinity 和 shortest-uncached-first 是部署策略。新增选项默认 off/auto，不能认为升级后所有路径都会启用。

本轮读完整文件清单及上述配置、KV草稿预留、部分选择／fallback diff；没有逐个审完223文件或重跑服务。PR 的早期 side-branch 性能、review 前测量和最终 head 不是同一基线，故不搬运汇总倍率。**建议作为架构与采用条件的深入阅读项，按需要单独回归通信误差、target输出、graph replay、取消与KV转移；剩余路径保留待审。**

### Ascend：ModelSlim W4A8 Linear 与 PLE 分片／INT8 工件适配

**vllm-project/vllm-ascend；[#20469](https://github.com/vllm-project/vllm-ascend/pull/20469)（2026-11-01T09:17:26Z，`main`，`2697e7f1fa7d`）；[#20276](https://github.com/vllm-project/vllm-ascend/pull/20276)（2026-11-01T07:13:16Z，`main`，`71aee7d12b58`）；[#20548](https://github.com/vllm-project/vllm-ascend/pull/20548)（2026-11-01T10:36:07Z，`releases/v0.32.0rc`，`1de10119d4cd`）；[#20443](https://github.com/vllm-project/vllm-ascend/pull/20443)（2026-11-01T10:36:29Z，`releases/v0.32.0rc`，`d9b3281c1510`）；[#20502](https://github.com/vllm-project/vllm-ascend/pull/20502)（2026-11-01T07:30:56Z，`releases/v0.32.0rc`，`d16dc393675c`）。**

主分支为普通 Linear 注册 ModelSlim 对称 per-channel INT4 权重＋动态 per-token INT8 激活，含 packed 加载、TP scale/bias 处理，小输出分区不满足 NZ 对齐时留 ND；per-group INT、非 ModelSlim 及其他 MXFP 格式不在这项承诺内。19项 CPU 单测通过是作者报告，最终 NPU 加载／精度／性能未重跑。

ETP 修复让 PLE 表分片可独立于 TP；PCP>1 隐藏配置时重新建立 TP×DP 的组，避免退回仅TP。**最终 #20276 只有3文件，PR文字提到的 Engram contiguous 修改与12组合测试不在本次 diff 内，不能当作本 PR 新交付。** 发布支线 `releases/v0.32.0rc` 另带回 PLE INT8（每32值FP32 scale）、PP/SP 的 stage-local QSA/GDN/PLE 所有权，以及 hidden2560 的 A5 MegaMoE和非量化MTP draft通信选择。它们是支线合入，非正式v0.32发版。

PP/SP的16卡 TP8/PP2/MTP3服务证据来自原系列；PLE INT8仅有有界lookup/dequant smoke，release全模型待验；MegaMoE的32次smoke和GPQA来自混合开发栈，不能单归本修复。**建议先固定 vLLM、Ascend 与算子包版本，按 checkpoint 格式、PP阶段、ETP rank 排序和 target/draft 分别验收。**

### Ascend Qwen3.8 Next：减少 CPU 长度同步，按逻辑 KV 长度切换 prefill

**vllm-project/vllm-ascend；[#20327](https://github.com/vllm-project/vllm-ascend/pull/20327)（2026-11-01T10:44:26Z，`main`，`af7c3fc267f8`）；[#20304](https://github.com/vllm-project/vllm-ascend/pull/20304)（2026-11-01T07:09:28Z，`main`，`6123e27d64b8`）。**

初始化时审核过的 MRV2 QSA/GDN/PLE target/draft consumer，默认跳过冗余 exact-count D2H 和同步CPU镜像更新；position、KV slot和接受后状态仍按device数据处理。决定在初始化后冻结，先 drain 旧 copy stream；未知consumer、CP、稀疏offload、实时媒体等继续同步。21参数化用例为本地 stand-in，最终默认开启 head 的原生 NPU 输出／cache／性能未证，不能从消除一次同步直接推出服务提升。

另一个主分支策略按**每请求当前 chunk 起始的逻辑可见 KV**判断：prefix<24K用 FIA＋selected-token mask，达到24K改GBSA；跨阈值chunk不拆，下一chunk再切，纯decode始终GBSA，混合batch逐请求分派。32K→1、并发1的旧 stacked 测试 TTFT13475.8→10325.0ms；仅一次请求，正文未完整给硬件拓扑与隔离基线，本轮不作普遍23.38%承诺。**建议对长prompt分布和混合batch复现阈值策略，并补最终head准确性。**

### Ascend 正确性：PCP packed KV、MXFP 尾部写与 Engram 旋转

**vllm-project/vllm-ascend；[#20512](https://github.com/vllm-project/vllm-ascend/pull/20512)（2026-11-01T09:08:17Z，`main`，`1deb03579065`）；[#20539](https://github.com/vllm-project/vllm-ascend/pull/20539)（2026-11-01T10:33:15Z，`main`，`0fcbee26875a`）；[#20428](https://github.com/vllm-project/vllm-ascend/pull/20428)（2026-11-02T01:37:21Z，`main`，`38261455c4c1`）；[#20209](https://github.com/vllm-project/vllm-ascend/pull/20209)（2026-11-01T07:42:50Z，`main`，`2ba8cbfb57ce`）。**

GLM5.3-Flash将 KPool query/index rows、PCP两个prefill chunk的NoPE metadata、replicated decode政策和去重后的 packed KV写入对齐；没有新增decode-sharding模式。270项worker测试来自当前更新，A3 PCP8/MTP5与AIME结果来自更早refactor前的组合栈，不能当最后初始化重构后的严格baseline等价。

SwigluGroupQuant四路packed store改用共享剩余列计数，修复 intermediate640等尾向量越界／激活不稳定；作者在Ascend950/CANN9.2报告22算子回归，GPQA180/198是混合部署结果，非独立归因。DeepSeek4.1 Engram支持 `diag(B0,B1,...)` 的非重复32×32块，压成block table；任意dense非块对角旋转仍不支持，作者只列py_compile/whitespace与相关支线审阅。

GLM静态编译则补入 MTP uniform-decode 向上取整后实际capture的shape（例如56→60）、让PoolKeyIndexer保留dynamic binary，并适配安装后扁平AscendC头文件。作者称NPU手工验证成功，尚无自动化测试。**建议按上述实际失败条件回归，分别核对最终算子包，避免用“支持某模型”掩盖布局与编译条件。**

### Ascend RecoverSSM：分开边界／非边界提交，承认小负载回退

**vllm-project/vllm-ascend；[#20474](https://github.com/vllm-project/vllm-ascend/pull/20474)（2026-11-01T07:12:22Z，`main`，`41d3969b31a8`）。**

接受token不跨snapshot边界时用更大tile，跨界单独处理，A5 grouped-head支持两路；请求保持外层，num_layers留运行时参数，避免新增JIT状态。作者A3/A5各212算子测试、1552个FP64比较，支持算子范围正确性。

早期非跨界路径局部时延下降27.25–49.68%／45.64–70.21%，后续grouped-boundary相对前revision 2.288→1.738ms；小／边界密集负载完整上下文有最高43.81%／18.71%回退。旧整模 `ed8ebdef` SSE ITL改进与最终算子head不同，GSM8K重复结果波动，严格整模等价未立。**建议按跨界比例分桶，测完整状态提交与服务ITL，不能照搬算子最大收益。**

### FlashInfer：GDN/KDA 训练新增显式非确定性单卡与 context-parallel 路径

**flashinfer-ai/flashinfer；[#8157](https://github.com/flashinfer-ai/flashinfer/pull/8157)（2026-11-01T04:10:54Z，`main`，`c3f2e2ad1ff3`）；[#8172](https://github.com/flashinfer-ai/flashinfer/pull/8172)（2026-11-01T09:09:26Z，`main`，`1f7e30ae2b38`）；[#8191](https://github.com/flashinfer-ai/flashinfer/pull/8191)（2026-11-01T19:41:26Z，`main`，`47879e4bc798`）；[#8218](https://github.com/flashinfer-ai/flashinfer/pull/8218)（2026-11-01T22:34:17Z，`main`，`06c5bdcda2a3`）。**

这批能力是带全部输入梯度的chunked training，而不只是prefill kernel：GDN和KDA各补齐单rank及KCP非确定性实现，facade默认 `deterministic=True`，明确设False才走新实现。GDN用BF16 q/k/v、FP32 gate/beta、head-dim128的packed长文档；KDA gate为FP32、beta BF16，q/k已在调用前归一化。KCP由进程组切token并执行状态扫描、merge与交换，不等于把单卡倍率直接扩到多机。

基线包括同族确定性Cake，以及满足精度预算的FLA0.6.0、cuDNN frontend1.32.1/FROST、FlashQLA等适用实现。#8172公开的R200 CP2/4表显示新实现较确定性Cake仅近持平（1.00–1.02×），相对某些非Cake基线的3.7–5.2×是另一分母；B200/GB300表当时仍待补。#8218有B200/B300节点及GB300 NVL72跨节点CP测试，但只测KDA步、不含short convolution。**建议作为线性注意力训练实验候选，复现时统一输入预激活、state精度、checkpoint、计时和最慢rank；暂无整模训练加速证据。**

### FlashInfer KDA lean arena：确定性不变，用少量时间换 workspace

**flashinfer-ai/flashinfer；[#8174](https://github.com/flashinfer-ai/flashinfer/pull/8174)（2026-11-01T09:17:55Z，`main`，`3d971efa707b`）。**

`WorkspaceDetKcp(local_arena="lean", lean_passes=2)` 以多次局部pass复用scratch；`replay`仍默认，前向不变，作者逐字节对照输出和梯度。W=1的KCP层测试：B200 H4 workspace4,729,623,808→3,925,615,956字节（17.0%），反向4.9050→4.9462ms；GB300/R200所测几何节约22.7–22.9%，反向增加约0.37–1.07%。

条件为head-dim128、BF16 q/k/v/beta/dout、FP32 gate/state，H4为261,393 tokens／4文档，H64为16,380 tokens／12文档；CUDA events、同进程交错采样、PyTorch2.14.1+cu130。**这是reserved workspace，不是整模峰值显存减少同一百分比。** W8布局只在单节点NVLink验证，不能推成跨两节点结果。建议显存卡在KCP scratch的训练任务先试这一显式权衡。

### FlashInfer 训练算子：非确定性梯度选项与确定性的设备边界

**flashinfer-ai/flashinfer；[#8190](https://github.com/flashinfer-ai/flashinfer/pull/8190)（2026-11-01T11:42:36Z，`main`，`e302d382bb65`）；[#8160](https://github.com/flashinfer-ai/flashinfer/pull/8160)（2026-11-01T04:03:00Z，`main`，`6c522eff1abd`）；[#8208](https://github.com/flashinfer-ai/flashinfer/pull/8208)（2026-11-01T22:59:42Z，`main`，`d5438ad3c8ea`）；[#8195](https://github.com/flashinfer-ai/flashinfer/pull/8195)（2026-11-01T13:50:07Z，`main`，`cd0ae364ef07`）。**

dense projection新增split-K/stream-K的非确定性weight-gradient，data-gradient及默认forward留确定性路径，只在明确适用几何选择新程序；部分router route须调用者显式split数。mHC扩到SM100/103 hidden5120/7168、4 residual streams，以FP32 atomic合并映射梯度，SM107仍仅4096。RMSNorm新增显式 `deterministic=False` 的dw路径，dx仍逐位可复现，默认不变；#8208在合入时描述中的速度表仍“pending”，故本轮不给该项提速数字。

确定性RMSNorm另澄清dw只保证**相同SM数量的设备之间**可复现；低于测量窗口SM数时改补偿dw程序，避免误把同架构不同SM设备视为相同比特基线。**建议将数值模式、SM数量和梯度误差纳入实验配置；这些局部新增API不等于默认训练自动更快。**

### FlashInfer LM-head loss：跨线程常量发布与选择性checkpoint修复

**flashinfer-ai/flashinfer；[#8152](https://github.com/flashinfer-ai/flashinfer/pull/8152)（2026-11-01T01:31:08Z，`main`，`6528657cd1ac`）；[#8222](https://github.com/flashinfer-ai/flashinfer/pull/8222)（2026-11-01T18:48:13Z，`main`，`27ccbb48e2e5`）。**

BindingCache的lookup→LRU更新、检查→evict、owned_bytes迭代之间可被其他线程打断；修复对已被移除项容错并快照重试。共享device常量在stream-pool stream填充后记录event，由host等待填充完成再发布，并以setdefault选择唯一缓存；capture路径例外仍在capturing stream创建。它并非对整个device做synchronize，但首次调用仍可能等待pool已有工作或lazy loading。

SAC按操作位置重放，原“loss-only forward／fused recompute”及不同线程首次状态初始化让两次操作序列不同，还会触发缓存tensor被原地改写检查。现在det entry在SAC内暂时弹出**全部torch dispatch modes**并按普通non-reentrant checkpoint运行；auto placement在recompute形成梯度交给backward，常量初始化也避开active modes。代价是用户FlopCounter等看不到内部算子，FakeTensor／compile trace在此组合可能显式失败；依赖私有`_disable_current_modes`，缺接口时回到旧行为。**建议做多线程/多stream首次调用、MUST_SAVE与compile组合回归，不能把修复解释为任意SAC策略完全透明。**

### FlashInfer GDN/Mamba：拒绝错误几何，并缩窄SM107自动CP

**flashinfer-ai/flashinfer；[#8196](https://github.com/flashinfer-ai/flashinfer/pull/8196)（2026-11-01T13:49:06Z，`main`，`7ec785bbf6cb`）；[#8213](https://github.com/flashinfer-ai/flashinfer/pull/8213)（2026-11-01T16:34:53Z，`main`，`1feaf68dfbb8`）；[#8225](https://github.com/flashinfer-ai/flashinfer/pull/8225)（2026-11-01T19:56:36Z，`main`，`443ee13db88a`）。**

SM107 Cake GDN的CP自动带原只用FP16验证，却接纳BF16并出现token串行state repair，作者某些行从默认0.07–0.24ms退到51–433ms。现在auto带限FP16；BF16＋FP32初/末状态的指定长序列低并行度区域显式报Unsupported，调用者应选默认backend；`use_cp=True/False`不随auto修正，不能假定所有调用都会自动fallback。

非CP SM107在DV-split没交付对应key时改用已有full-DV record，限无checkpoint、packed state、多chunk、非低精度state等条件；opt-in `backend="cake_gdn"` 不影响auto。Mamba2 `seq_idx/chunk_indices/chunk_offsets` triple只描述一条packed token stream，现在batch!=1在launch前拒绝，避免其余行未写和部分chunk越界；没有新增kernel或提速。**建议低batch长状态推理及Mamba packed调用者优先核请求几何和错误处理。**

### FlashInfer：softmax误差修正与QKV交换ready-word压缩

**flashinfer-ai/flashinfer；[#8236](https://github.com/flashinfer-ai/flashinfer/pull/8236)（2026-11-01T23:32:48Z，`main`，`69265901d35f`）；[#8231](https://github.com/flashinfer-ai/flashinfer/pull/8231)（2026-11-01T21:35:53Z，`main`，`d2d59e47d545`）。**

Blackwell FP32 row-major softmax把指数形成改为单次FFMA及预先舍入shift，作者float64对照最大相对误差在B200/GB300约从1.0e-6降至7.8–7.9e-7；相同launch的41个几何平均速度约0.984/0.986×，是精度与代价的交换。仅另一些改dispatch的shape稍快，不能统称全softmax加速。

`QkvNormRopeExchangeOverlap` 将投影、QK RMSNorm/RoPE与Ulysses QKV all-to-all配合；SM107两/四rank主机计划改为每目的端发布一个readiness word，wait扫描world_size个word而非按128-row tile×目的端。kernel源码和其他架构表未变，环境shape gate仍能覆盖。作者4×R200、CUDA13.5测到H3072特定行最高1.342×，H6144/8192大致持平；只代表这个通信融合调用。**建议关注ready-word协议如何减少等待扫描，采用时测本机rank/hidden/token分布。**

### ROCm AITER/ATOM：MXFP4缓存生产端与单launch DSA indexer

**ROCm/aiter；[#7057](https://github.com/ROCm/aiter/pull/7057)（2026-11-02T01:39:47Z，`main`，`f7374b5d4cae`）。**

`fused_qknorm_idxrqknorm`可将index-K及主K/V写成128-token页的MXFP4（e2m1＋每32值e8m0 scale），q_out可为饱和unit-scale E4M3。它是**producer**：gfx950外拒绝MXFP4，消费端score／attention kernel及ATOM集成各有独立PR，不能只据本提交宣称完整MXFP4模型服务已可用。测试已并入普通op_tests脚本，不应把已删除的pytest hook当当前CI覆盖。建议读缓存格式与形状守卫，先做生产／消费两端字节及精度对照；本项未给整模显存或吞吐收益。

### ATOM mono indexer与前端协议：融合范围和支线边界

**ROCm/ATOM；[#2871](https://github.com/ROCm/ATOM/pull/2871)（2026-11-01T15:52:45Z，`main`，`99ebf633ac7a`）；[#2844](https://github.com/ROCm/ATOM/pull/2844)（2026-11-01T02:45:57Z，`main`，`d246797dcc7f`）；[#2878](https://github.com/ROCm/ATOM/pull/2878)（2026-11-01T09:24:31Z，`alizaidy/gfx1250-qwen-support`，`d3b8e34a4d1f`）。**

main新增共享DSA indexer mono：一个FlyDSL launch合并wq_b、index-cache写、FP8打分和exact top-k到MLA slot；按IndexerShape构建，部署不满足则在路由任何层之前拒绝。DCP各rank持独立keys、非DCP按128-key run分工；verify由batch_id_per_q_token定位请求、padding不写。Kimi-K3 mono可到64行，plain decode／uniform verify／ReplaySSM的适用条件分别保留，不能泛化所有模型都单launch。

另main统一Chat/Responses推理控制优先级，但模型adapter决定合法值；Responses流输出stable reasoning item与completed/incomplete/failed，去掉Chat的[DONE]，未知模型在enqueue前报错。适配策略依checkpoint选择，不能仅改served alias就套DeepSeek规则。

gfx1250 Triton-MoE修复 #2878 **只合入 `alizaidy/gfx1250-qwen-support` 支线**，涉及mori counter/arena、调用参数及Qwen无意SwiGLU clamp；依赖另外aiter PR的并发2048对照不是main能力或孤立性能证明。GSM8K差异小于报告标准误，不能写精度改善。建议main使用者按indexer失败条件和HTTP/SSE语义回归；支线部署继续观察依赖集成。

### MLX：NAX量化矩阵路径与自定义RoPE频率

**ml-explore/mlx；[#4781](https://github.com/ml-explore/mlx/pull/4781)（2026-11-01T11:13:59Z，`main`，`914732419f25`）；[#4732](https://github.com/ml-explore/mlx/pull/4732)（2026-11-01T12:20:51Z，`main`，`3d2fbcdf11de`）。**

NAX GPU上新增FP格式split-K、staged qmm和affine 6–16行qmv，按core count限制分块；NVFP4部分窄矩阵仍留旧路径。作者M5Pro16核/M5Max40核/M6十二核、macOS27.0.1，36 projection形状×5格式、M=1/6/8/12/16/17/32/64/128/256的算子测试：FP格式17–256行通常1.3–2.6×，**M=1不变**；不能据此推断单请求逐token decode同幅加速。

`nn.RoPE`补`freqs`与`partial_rotary_factor`，将常量频率传给已有fast op；比例编码分母仍是完整dims，未旋转部分用无限波长，freqs不注册为训练参数。它减少自定义位置编码的Python重写，非新增长上下文训练结论。建议Apple本地推理先测多行prefill/验证批次，位置编码适配另做参考数值测试。

### Triton/Helion：RDNA4缓冲描述符与CuTe数值／PDL合同

**triton-lang/triton；[#12571](https://github.com/triton-lang/triton/pull/12571)（2026-11-01T21:48:51Z，`main`，`a380b0f498f3`）。**

Gluon新增`ttgl.amd.rdna4.buffer.make_buffer_descriptor/load/store`，复用gfx12 CachePolicy；coherence scope与性能cache hint区分，RDNA4默认WGP跨两个CU的cta scope映到SE，CU mode才映到CU。gfx1201测得bounds以4字节piece（或更窄整项）检查，不能照搬gfx1250的规则。建议自写RDNA量化／attention kernel时把边界与scope视为正确性条件；这里没有端到端LLM加速实测。

### Helion CuTe：half atomic默认逐次舍入，零填充可参与PDL

**pytorch/helion；[#4217](https://github.com/pytorch/helion/pull/4217)（2026-11-02T01:19:53Z，`main`，`2afe5adb8dfa`）；[#4218](https://github.com/pytorch/helion/pull/4218)（2026-11-02T01:20:39Z，`main`，`ff73085593b1`）；[#4219](https://github.com/pytorch/helion/pull/4219)（2026-11-02T01:21:25Z，`main`，`8f6ddb99dcfa`）。**

CuTe对fresh FP16 atomic output不再默认放大到FP32临时缓冲再cast，而按half atomic语义逐次FP16累加，并用packed f16x2/bf16x2降低开销；`cute_promote_half_atomic_outputs=True`／对应环境变量可恢复提升精度模式。作者split-K64分区示例最大绝对误差0.125→0.97、输出约100，虽在例子atol=1内仍是**真实数值变化**，应先审误差再看微秒级收益。

开启cute_pdl时，存活的host zero-fill可变empty＋dependent fill kernel，主kernel把wait延迟到第一次接触被填充tensor；alias/descriptor等每条访问都需覆盖，无法证明参数别名就保持wait-first。>=2^31元素保留memset/zero_回退，未开knob不换路径。GDN recurrence另允许fixed-size tile loop进入mma_dag搜索，并拒绝有未定义/超前loop index的tcgen05 emitter。**建议DSL开发者读数值、别名和wait门禁，微基准不等于整训加速。** 本窗有1.5.2 release，但这些提交时间晚于该release，未据此宣称均随1.5.2交付。

### 扩展分支的证据与未完成边界

34仓库的72小时回溯共枚举1,581条已合搜索结果、1,683个默认分支提交（两种时钟不同，不能相加为成果数）；release分页越过回溯边界。本窗Helion1.5.2仅确认tag事件，未用版本号代替技术增量。51个候选的PR与完整changed-file页均重开，页数与changed_files一致；PyTorch另核真实main提交，保留首次合入/撤回/重落地。普通CI、日志、security依赖维护、重复backport及纯形状tuning没有逐条堆入正文。未审的历史、回溯候选、dsx/run-ai与大量Cake shape优化继续留队列；“已查元数据”不等于机制审阅完成。


<a id="coverage"></a>
## 实际覆盖、补录与下一轮续查

| 来源类别 | 本轮实际检查 | 边界与未完成内容 |
| --- | --- | --- |
| 模型与官方材料 | 52个HF组织、208次按createdAt/lastModified的模型/数据查询，分页到回溯下界；123候选；57个官网入口中55个可访问（含2个web恢复）；固定模型卡、配置、树、代码及Luna PDF | Recursive博客与Dory报告入口仍受阻；动态历史和全局社区发现不完整。SenseNova Loop仅README改动，权重/代码/config对象不变，未当新权重；PiVo纯TTS、无证据量化与重复工件未纳入 |
| 核心六仓库 | release分页过下界，645条已合搜索/624个main提交；39个新PR，完整最终文件页及分支身份核对 | 38main＋1sglang-miles；351具体待审候选和全部继承旧队列保留。verl/slime本窗已查表面未见新事件，不推断项目没有其他发布 |
| 扩展34仓库 | 1,581条已合搜索/1,683个默认提交；51候选PR/完整文件页，PyTorch另读真实main提交 | 50个来源事件写入23组摘要；部分只是明确版本增量或支线。未审形状优化、其他候选及TRT223文件中的剩余路径不记语义完成，1,332个候选保留到具体清单；没有靠版本号凑内容 |
| 论文与正式学术来源 | 六类arXiv new共1,202唯一work，回溯API1,543条metadata；12固定v1全文，收录10篇；6作者仓库目录、4实现对应；HF按10/30–11/2，TMLR/ACL/PMLR当前入口 | new页仍10/30；主窗查询SSL后重试返回0只代表可见metadata；lastUpdated查询被改写，不能冒充修订全覆盖。HF11/2为400，OpenReview API及两份PDF为403/挑战页，正式卷与全文未穷尽 |
| 硬件厂商与软件栈 | 重新打开并保存NVIDIA/AMD技术博客、Intel Gaudi、Ascend公告、Cloud TPU与AWS Neuron release入口；相关NCCL/RCCL/Ascend/PyTorch/XLA/MLX仓库在工程扫描内 | 六官网入口直取成功，不等于动态公告完整。Gaudi1.24.2、Neuron2.33.0及已报Dynamo/Primus文章未重发；TPU返回面含旧记录，CANN/HCCL/libtpu与组件历史未关。没查到新条目不等于厂商没有发布 |

**补录与更正。** 8篇10月29日论文为72h查漏；ClosedLoopGaussian为10月28日旧pending，Hybrid Mechanics为10月7日旧文，Luna为10月29日系统卡，均保留原日期。PyTorch INT4路径明确首次落地→撤回→本次重落地；TorchTitan交叉熵以最终backward重算代码纠正PR早期文字；Ascend ETP测试/contiguous描述与最终三文件不符，正文已限定。没有重写历史日报，新的证据边界在本期说明。同日若再跑，继续更新这一文件。

**下一轮。** 从2026-11-02T01:42:47Z继续事件枚举，并回溯72h；LongCat精确上线时刻和正式报告、SCARL论文run配置、PDScope完整predictor、多模型基线、SGLang开放集成PR及最终硬件验证优先追踪。旧10月5日起区间、论文7,540候选和532条replacement及来源失败继续保留；本轮不推进全局完整语义水位。

运行状态、来源清单、证据快照、逐项稳定ID/版本、失败与待审队列保存在本地`.monitor/research-2026-11-02/`及`.monitor/state.json`、`sources.json`；本地生成和远端上传状态分别记录。108个稳定来源事件（核心39、扩展50、论文10、模型/报告9）可包含同一工作的多个工件，**不是108项独立成果**。正文不代表全来源全面监控。
