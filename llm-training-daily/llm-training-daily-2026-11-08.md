# LLM 技术动态日报 · 2026-11-08

> 时区 Asia/Shanghai（UTC+8）。增量扫描窗口 **2026-11-07 09:01:16 → 2026-11-08 09:00:30**，向前回溯 72 小时至 **2026-11-04 09:01:16**。实际生成时间 **2026-11-08T09:15:21+08:00**。日期未披露到时分时保留原日期。首次发布、工件或代码更新、会议报告与本轮发现分别标注。
>
> 生成前核对本地 32 篇历史日报及索引，与 GitHub `main` 的 `598a6c872e7212c6f1c07f0ff8c535e592f50244` 逐字一致。前次待上传队列为空。
>
> 本轮核验公开的一手报告、工件、发布记录及部分代码补丁。性能与能力指标为原作者或项目方的口径。

## 本次重要变化

- **训练正确性与升级。** DeepSpeed 修复 NVMe offload 把 I/O 失败当成成功及重复预取的问题。vLLM 默认分支移除 Model Runner V1，摩尔线程刚补录的 vLLM-MUSA 0.28.0 仍默认使用 V1，迁移前要区分两条版本线。
- **Megatron-LM 主干更正。** 四个提前合入的补丁已撤回，其中有 MFSDP v2 的 Muon 集成和 EP-invariant combine。上一期报道的 MFSDP v2 主合入和梯度累加改动属于独立提交。
- **后训练的复现边界。** Qwen 在十个 BF16/FP8 仓库更新 assistant-only loss 需要的模板标记。Reflection 的 Beam 报告给出了异步 RL 权重滞后、长轨迹跨版本及大规模训练恢复的数据，当前只核实到 PDF 版本时间。
- **长上下文需要按真实负载复测。** SGLang 合入 Kimi K3.1 的 PP 与 prefill CP 图执行路径，昇腾配方补录给出分开的索引与 KV 量化、分片状态交换及 950DT 测量边界。部署模拟显示，在细化 P/D 副本粒度后，AFD 的优势适用范围收窄。
- **论文优先读复现审计。** AACL-IJCNLP 的 PEFT 配对实验表明超参数和精度口径会翻转部分 LoRA 变体的收益，多语 MTEB 审计显示总榜可能掩盖目标语言的反向排序。

[模型与技术报告](#models) · [后训练与数据工件](#posttraining-models) · [评估与科研 Agent](#model-evals) · [研究论文](#papers) · [开源工程](#engineering) · [硬件与软件协同](#hardware) · [覆盖与续查](#coverage)

<a id="models"></a>
## 模型与技术报告

### 补录｜Beam 技术报告拆开异步 RL 的权重滞后和训练恢复

**Reflection，官网 PDF 的 HTTP 最后修改时间为 11 月 7 日 02:22:45（北京时间），首发时刻未核实。** 10 月 16 日已开放的 501B 总参、23B 激活的 Beam，本次补的是 156 页训练报告。Reflection 披露两阶段中训练把有效上下文扩到 1M token，随后约 10,500 张 GB300 在四周异步 RL 中生成超过 1 亿条 rollout，峰值指标包括约 11 万并发 rollout 和 12 秒权重更新。长轨迹可跨多个策略版本：同一轮解码保留旧 KV，下一轮以新权重重算完整上下文并重放此前的专家选择，逐 token 的策略版本随数据送到训练端。生产记录中，策略版本差异主要来自同组 rollout 彼此进度不齐。即使缓存命中率超过 94%，不少阶段新 prefill token 仍多于输出 token。报告也记录了序列仍在长度限制内、路由过度集中却触发 OOM 的情况。[报告 §4.4、Fig. 33–35](https://reflection.ai/documents/beam-model-tech-report#page=65) · [官方模型卡](https://reflection.ai/beam-model-card)。

同一份报告的 6,144 张 GB300 预训练统计中，21.9 天端到端保留进度的 goodput 为 77.0%，合并基础设施修复后最后 20% 步数为 92.3%。103 次恢复后，到首个恢复训练步的中位数为 9.7 分钟，另有 MFU 回升过程。**建议深入阅读**，对照异步队列的同组策略版本差、专家负载分布和故障恢复时实际保留的训练进度。这些数字来自 Reflection 的生产运行，官方公开的是模型工件和技术报告，训练服务、任务混合与轨迹的完整复现材料尚待核实。[报告 §6.7、Table 26 与 Fig. 44](https://reflection.ai/documents/beam-model-tech-report#page=86) · [完整报告 PDF](https://reflection.ai/documents/beam-model-tech-report) · [原始权重发布](https://reflection.ai/blog/beam-is-available)。

### PLaMo 3 Translate 更新四档模型的翻译模板，补录蒸馏配方

**Preferred Networks，11 月 5 日正式发布权重，11 月 7 日 16:22–19:40（北京时间）更新模板。** 0.8B、2B、8B 和 31B 四档翻译模型支持 53 种语言。主窗 GGUF 更新允许将手写 PFML3 作为最后一条用户消息传给只支持 chat 的客户端，新增系统设置同一行写法，允许设置键大小写混用。简明日语的渲染代码改为 `ja-easy`，模板把旧 `ja(easy)` 归一化。开发报告披露两阶段 SFT 从大规模合成翻译过渡到更精筛、面向真实场景的数据，再做 DPO。8B 及以下在 DPO 前向 31B 蒸馏，参考翻译上用前向 KL，学生以温度 0 自己生成的翻译上用反向 KL。PFN 的 0.8B 模型卡中，GGUF Q4_0 为 0.55 GB、BF16 为 1.78 GB，所测 FLORES+ 日语对多语 chrF++ 为 34.97 对 36.08，WMT24++ 英译日 COMET-22 为 0.8723 对 0.8724。多语对照只测 53 种中的 15 种，日英汇总还包含内部评测。**可复现量化与模板兼容性**，分开验证术语表、简明日语和直接传 PFML3 的客户端。权重按 PLaMo Community License 提供，企业商业使用须核对其营收门槛。[正式发布与许可更正](https://www.preferred.jp/en/news/pr20261105) · [技术报告](https://www.preferred.jp/en/blog/tech/plamo-3-translate-weight-release) · [31B GGUF 第一处固定差分](https://huggingface.co/pfnet/plamo-3-translate-31b-GGUF/commit/5b33ad543d91720364ee8fc4489bdcdd5557e3f8) · [简明日语固定差分](https://huggingface.co/pfnet/plamo-3-translate-31b-GGUF/commit/0dc5dffa8dab91cbd92fba60211a8f001e85f0d8) · [0.8B 模型卡](https://huggingface.co/pfnet/plamo-3-translate-0.8b)。

<a id="posttraining-models"></a>
## 后训练与数据工件

### Qwen 为十个 Qwen3.6／3.8 仓库补上助手训练掩码

**阿里 Qwen，11 月 7 日 12:41–12:46（北京时间），BF16／FP8 版本同步提交。** Qwen3.6-27B、Qwen3.6-35B-A3B、Qwen3.8-27B、Qwen3.8-Flash-Next 和 Qwen3.8-2.4T-A95B 的五对模型仓库加入响应模板与生成标记。已打开的 Qwen3.8-27B-FP8 前后差分把助手角色头移到 `{% generation %}` 之外，把实际保留的 `<think>`、回答、工具调用和 `<|im_end|>` 包在生成区间内，工具结果属于独立分支。Qwen3.8-27B 的 tokenizer 配置另增加结构化响应模板，定义思考段和 XML 工具调用的解析字段。Transformers 可据此返回 `assistant_masks`，供 TRL 的 `assistant_only_loss` 使用。**建议复现一个带工具往返的样本**，对比固定版本的渲染、token ID 和损失掩码，特别是 `preserve_thinking` 的两种设置。本次核对了模板工件和训练接口，尚无这些改动带来的模型质量测量。[Qwen3.6-27B 固定差分](https://huggingface.co/Qwen/Qwen3.6-27B/commit/8f33b98937c6031bb49e713dee3e8e185aafd26f) · [Qwen3.8-27B 响应模板](https://huggingface.co/Qwen/Qwen3.8-27B/commit/37561816e66b5b9153b1b64c10968c18c1e6f4a0) · [Qwen3.8-27B-FP8 固定差分](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/commit/c3a462cc3d38cf351f3b26cbd763469eeb74ef81) · [Qwen3.8-2.4T-A95B 固定差分](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/commit/4f885e51d6bbfee5866446d56d3f96f336c39723) · [Transformers 掩码契约](https://huggingface.co/docs/transformers/main_classes/tokenizer) · [TRL 训练模板](https://huggingface.co/docs/trl/chat_templates)。

### 补录｜REINFORCE Zero 先对齐训推数值，再校正异步权重滞后

**NVIDIA Research、Inferact，11 月 6 日 21:40:38（北京时间）提交论文和算法工件，22:59:56 的 vLLM 官网源提交补上旗舰模型配方。** 异步 RL 的采样与训练 logprob 差同时来自旧权重和不同内核。REINFORCE Zero 先让同权重、同历史下的两端逐 token 数值对齐，再用训练概率除以采样时记录的真实概率，不截断比率。优势以同题组平均奖励为基准，每个回答内部按生成动作 token 平均。DeepSeek-R1-Distill-Qwen-1.5B 训练 6,000 步后，以 8K 回答预算评估，五项数学宏平均 pass@1 为 48.80%（标称滞后 2），同步对齐为 48.91%，标准内核异步重要性采样为 45.32%。另一次 16 GPU、12,000 步测速中，两边都对齐内核，同步用 164.77 小时，异步用 68.55 小时，得到 2.40× 步吞吐，计时排除评估。[固定论文与实验表](https://github.com/hijkzzz/REINFORCE-Zero/tree/ac19d189c4caaa4a5946146bd3eeaef3db244719) · [固定算法说明](https://github.com/hijkzzz/REINFORCE-Zero/blob/ac19d189c4caaa4a5946146bd3eeaef3db244719/ALGORITHM.md)。

旗舰工件验证的是另一组口径。DeepSeek-V4.1-Flash 在 16 节点、每节点 4 张 GB200、128K SWE-bench Pro agentic RL 设置逐步保持采样 token logprob 一致。Kimi K3.1 的 KDA 解码复用 prefill 对应 chunk 行的算术顺序，在 4×4 GB300 的 100 步 DAPO LoRA 验证中保持一致。官网把两款旗舰的长程异步训练列为后续工作。**建议先复现同权重下的逐 token 差值**，分别覆盖不同批组、前缀缓存命中和恢复，再按实际策略版本差观测异步训练。[vLLM 官网 §3–6](https://vllm.ai/blog/2026-11-06-zero-mismatch-async-rl) · [固定公告源](https://github.com/vllm-project/vllm-project.github.io/blob/8cc9eecb09ec9f45cc65554f8b3e41ada92c0f36/_posts/2026-11-06-zero-mismatch-async-rl.md) · [各模型复现配方](https://github.com/Inferact/zero-mismatch-rl)。

### 补录｜Harness adapters 让同一底模在三套 Agent 运行时分别做 RL

**harness-hub，11 月 6 日 22:13:59（北京时间）发布报告，270 项环境工件当日 16:55 固定。** 项目用 TRL `AsyncGRPOTrainer` 给 pi、opencode、hermes 分别训练 rank 32 LoRA。每模型三个 trainer 共用一套 vLLM 底模，每题八个 rollout 都获评分后才入训练队列，超过四次更新的旧 rollout 被丢弃。代理 `safeagents` 在执行前检查工具调用，命中高严重度规则时把调用替换为提示，本次 rollout 得零分。拦截动作保留原始 token 和 logprob，学习信号仍能覆盖这次尝试。作者报告在 pi、LFM2.5-24B-A2B 的 50 步有无代理对照中，留出 pass@1 变化为 0.0 ± 4.1 个百分点。270 项环境中 253 项是从零构建，243 项出自同一轨迹库，最终训练在公告时仍持续记录。**建议小规模复现**，按项目和轨迹源留出，并分别计量任务通过率、通过任务的调用次数和拦截误判。[训练配方与消融](https://huggingface.co/blog/harness-hub/harness-adapters) · [固定环境数据](https://huggingface.co/datasets/harness-hub/traj2envs-rl/tree/191a751bda511eada76f6a9a76513d143321a64a)。

<a id="model-evals"></a>
## 评估与科研 Agent

### 补录｜GlobeBench 用真实执行结果验证文本环境模拟器

**Patronus AI，11 月 5 日作者解读，论文与公开工件始于 10 月 15 日 UTC。** 评测给定已渲染的 Agent 行动和观测历史，要求模型预测下一条环境观测，再用真实执行参考、确定性检查和语义裁判评分。共五类场景、19 种技能、1,632 项。20 个模型的同题结果中，GPT-6 Astra 为 809/1,632（49.57%），Grok 4.6 为 459/1,632（28.13%），当操作效果需传播到依赖对象时，20 个模型平均通过率降至 5.10%。631 项没有任何参评模型通过，拒绝和无有效输出计失败。**建议用公开轨迹离线重算**，分别看依赖传播、无响应和语义判分。它测的是给定历史的单步预测，长程自回归误差与用模拟器训练下游 Agent 的收益需要另测。[作者解读](https://www.patronus.ai/blog/globebench) · [论文 v1](https://arxiv.org/abs/2610.22708v1) · [固定复评分代码和结果](https://github.com/patronus-ai/globebench/tree/5f29c086d4604fb63789918175076201b13a083b) · [固定数据集](https://huggingface.co/datasets/PatronusAI/globebench/tree/da50af530cdfe5a97a2281fe4d9ac254efe04e31)。

### 补录｜PSI 开放科研 Agent 产出的 Aaronson–Ambainis 证明工件

**Physical Superintelligence，11 月 6 日公告，公开仓库初始提交为当日 10:47:20（北京时间）。** PSI 的内部研究系统调用未公开的 Anthropic 模型，报告第三天得到完整证明并通过 Lean，后来两次改进结果。固定仓库用 `PrintedConjecture` 声明有界低次多项式的影响力结论，公开证明、命题、依赖锁和验证记录。开发者报告两种 Lean 内核实现均重检通过。论文对量子查询复杂度的推论与形式化覆盖范围要分别读。**建议复现 Lean 重检和原命题对应核对**。公告未披露模型版本、完整轨迹和计算预算，因此无法据此复现生成证明的过程。[PSI 公告](https://psi.inc/research/aaronson-ambainis/) · [固定 Lean 工件](https://github.com/physical-superintelligence/aaronson-ambainis/tree/2a6ceaef872ae993708950f951989134084b851a) · [验证与覆盖边界](https://github.com/physical-superintelligence/aaronson-ambainis/blob/2a6ceaef872ae993708950f951989134084b851a/notes/VERIFICATION.md)。

<a id="papers"></a>
## 研究论文

本期把 AACL-IJCNLP 11 月 7 日的口头报告记作会议日程事件。arXiv 九分类最新公告仍标 11 月 6 日，以下首次收录的 arXiv 预印本均按原提交日期补录。官方常规下一批公告排在美东 11 月 8 日 20:00，即上海 11 月 9 日 09:00。公告批次、稿件提交和版本修订各用独立时间口径。 [AACL-IJCNLP 口头报告议程](https://2026.aaclnet.org/program/oral_program/) · [arXiv 公告规则](https://info.arxiv.org/help/availability.html) · [arXiv cs.CL 公告](https://arxiv.org/list/cs.CL/new)

### 预训练、架构与训练系统

#### SLA 将 3D 空间监督放进持续预训练

**补录｜2026-11-05 17:52 UTC｜arXiv v1。** SLA 沿用 π0.5 策略主干，在训练时用冻结的 3D 感知教师监督中间视觉表征。论文 §4.4.1 表 6 在 RoboTwin Clean2Random 的 50 个任务上比较 5%、25%、50%、100% 数据量及两个训练阶段的对齐开关。满数据时，两阶段均开启的成功率为 72.3%，只在持续预训练开启为 63.2%，只在后训练开启为 57.1%，均关闭为 48.2%。训练辅助模块可在部署时移除，结论来自 VLA 操控实验。复现宜保持动作数据混合、样本比例和后训练预算一致，并分别开关两阶段的空间监督。 [SLA v1 §3、§4.4.1 表 6](https://arxiv.org/html/2611.08373v1#S4.T6)

#### 强化学习环境与训练器双端账本定位静默故障

**补录｜2026-11-05 15:05 UTC｜arXiv v1。** 作者在 Gymnasium 环境与 SB3 PPO 之间注入观测、动作、奖励、终止语义和日志故障。低至中等注入比例下，训练收益阈值最多发现一成异常运行。双端逐步对账在实际启用的 Pendulum、Hopper、Pusher 上分别定位 189/189、207/207、189/189 次注入，在 Pendulum 的中位时间开销为 3%。Hopper 中有 12 次终止观测被改写，但训练器在真实终止时不读该值，权重仍逐字节一致。实验局限于稠密奖励仿真、单种 PPO 实现及三颗注入种子，且 Hopper 每次账本约 150 MB。将同样的检查迁到 LLM 工具环境时，可以先在小样本对齐 action、tool result、终止原因与 reward，再量跨机日志成本。 [环境—训练器边界 v1 §III、§VII、§IX](https://arxiv.org/html/2611.08144v1#S3)

### 后训练与数据

#### JanusPO 在小批量 GRPO 中分开约束概率上调和下调

**主窗｜2026-11-07 10:30–12:00 上海｜AACL-IJCNLP 口头报告排程，论文集版本。** JanusPO 分别配置正负优势的裁剪区间和 KL 权重，并用轮次权重分配末端奖励。Qwen2.5-3B-Instruct 的 GSM8K 单次消融峰值从 clean GRPO 的 74.8% 到 84.5%，作者跨两颗种子报告 84.3% ± 0.2%。同起点的多轮 DeepSearch 中，best@8 重测后 GRPO 的 GAIA/HLE F1 均为 0，JanusPO 为 0.32% 和 0.35%。离线模拟搜索、3B 的绝对 F1 低于 1%，且 7B 的 MATH-500 有负向迁移。复现时先锁定格式 SFT、优化器精度和预算，按任务报曲线与多种子终值。 [JanusPO 论文集 PDF §5、表 1、5](https://aclanthology.org/2026.aacl-long.200.pdf) · [会议 Session 3.2](https://2026.aaclnet.org/program/oral_program/#session-32-llm-agents-and-learning)

#### SFT 与 GRPO 的 LoRA 更新采用不同的谱变换

**补录｜2026-11-05 18:17 UTC｜arXiv v1。** 在同一套低秩适配参数化下，SFT 的配方在 A 使用 Adam，在 B 使用全谱展平的 SGD 方向。GRPO 对 A、B 都保留 Adam，仅展平 B 的更新中垂直于当前列空间的分量。Llama-3.2-3B 在 GSM8K 上，SFT 均值从 42.99% 到 45.72%，GRPO 从 42.36% 到 47.66%。表 4 的 MATH 扩展实验中，Llama-3.2-3B 的 GRPO 反而从 9.41% 到 8.21%。这些比较按配置各自选择最佳学习率、用三颗种子汇总，支持分训练阶段扫描谱形状和学习率，也应在 MATH 等不同难度任务保留标准 Adam 对照。 [LoRA 动力学 v1 §4、表 1、附录表 4–7](https://arxiv.org/html/2611.08414v1#S4)

### 推理优化与服务

#### 从同机架宽 EP 实测转向整套部署，AFD 仍取决于供给粒度

**补录｜2026-11-05 17:02 UTC｜arXiv v1。** 上期 FastAFD 在具体机架和模型上实测 decode。SiliconForge 则用已公布结果校准模拟，在 128 kW、TTFT 5 秒等约束下重新优化预填充、decode、设备和副本数，再比较 P/D 与追加 attention–FFN 拆分的 P/D+AFD。图 7 的 768 个组合中，579 个的单位功耗 SLO goodput 差异落在 ±9% 包络内。细化 P/D 池尺寸后，原本超出包络的 101 个 AFD 优势组合只剩 18 个。模型还纳入未来加速器类别和不确定规格角点，属于条件模拟。评估部署时宜固定功耗、TTFT/TPOT 与前缀缓存率，并对两边分别优化副本粒度和专家并行。 [SiliconForge v1 §IV–VI](https://arxiv.org/html/2611.08299v1#S4)

#### 扩散解码的两种并行目标，精确保分布与近似降延迟

**补录｜2026-11-05 17:41 UTC｜SpecCTMC arXiv v1。** 对满足附录 C1–C8 的掩码扩散过程，事件时钟只依赖剩余掩码数，因而可预采样。候选事件依次在当前已提交状态下核验或按残差分布校正，保留 DirectCTMC 的完整带标记路径分布。在 LLaDA-8B-Instruct 的五项任务和五种输出长度中，平均顺序目标调用减少 2.79 倍、墙钟快 1.55 倍。Dream 的对应均值是 2.85 倍和 1.36 倍，GSM8K 平均墙钟只有 0.97 倍。定理的对照是相同随机 CTMC，表中调用量、被批验证的状态量和真实延迟分开统计。 [SpecCTMC v1 §5、表 2–3、附录 B](https://arxiv.org/html/2611.08362v1#S5)

**补录｜2026-11-05 17:10 UTC｜Measured Coupling arXiv v1。** 该工作近似测量揭开位置 j 对位置 i 候选 margin 的影响，再批量提交互相支持的 token。LLaDA-8B-Instruct 的 GSM8K 相比 mean-field，顺序调用 65.4→40.1，单回答墙钟 8.10→5.86 秒，准确率差 −0.3 个百分点，配对 95% 区间为 −2.5 至 +1.9。计时用前 200 道 GSM8K，设定输出 256 token。同样配置迁到 LLaDA-1.5 时，MATH 下降 1.3 点且显著，使用双 KV cache 的部分设置也会失去时延优势。复现时要在同一设备、同一缓存与质量目标下对照，并报告逐步探针开销。 [Measured Coupling v1 §4.3、§4.6、§6](https://arxiv.org/html/2611.08307v1#S4)

#### DictLLM 用共享字典压缩权重，以稀疏系数保留层差异

**补录｜2026-11-05 14:31 UTC｜arXiv v1。** DictLLM 按激活统计分解线性层，共享字典，逐层剪系数，再重拟合字典。可对两因子额外量化到 NVFP4。表 1 中 LLaMA-7B 在 2× 压缩时 WikiText-2 PPL 为 10.59，Basis Sharing 为 19.99，未压缩为 5.68。NVFP4 5× 时 PPL 8.48、七项零样本均值 0.47，未压缩均值 0.52。论文的非结构化系数会引入不规则访存。复现宜固定真实字节和校准数据，另计稀疏索引、scale 及运行时实际显存，并和密集 W4A16 在同样批量比较端到端吞吐。 [DictLLM v1 §3、§4 表 1、附录 Limitations](https://arxiv.org/html/2611.08082v1#S4.T1)

### 评估与可靠性

#### KV cache 量化对齐诊断 v3 增加校准不确定性与逐模型开销

**补录｜2026-11-05 05:19 UTC｜arXiv v3，相对 8 月 8 日 v2。** v2 已报告 vLLM 的 FP8 拒答翻转，以及 Qwen2.5-7B 在 4-bit 下将 L0–1 保留为 FP16 后的 97.8% 恢复、7% KV 开销。v3 将九个模型在各自塌陷位宽下的 AdvBench 结果汇入表 5，例如 Phi-3.5-mini 的 3-bit 方案将翻转率由 43.8% 降至 6.9%，KV 开销为 47%。附录 C.8 加入以 20 条基线拒答提示起步的 bootstrap，当区间跨缓解策略的决策阈值就扩大校准集。附录 C.9 的两模型探针在各自关键层测得，3-bit per-tensor 误差投向拒答方向的幅值，Qwen2.5-7B 是 Mistral-7B 的 8.2 倍。这是两模型、两层的局部证据。复测需区分诊断量化与真实服务配置，在相同模型和位宽下同时报告翻转分母、区间和 KV 开销。 [v2 §5.1、§5.5](https://arxiv.org/pdf/2606.09864v2) · [v3 §5.5 表 5、附录 A 表 12、C.8–C.9](https://arxiv.org/pdf/2606.09864v3)


#### PEFT 配对复现审计显示超参数口径可以翻转结论

**主窗｜2026-11-07 10:30–12:00 上海｜AACL-IJCNLP 口头报告排程，论文集版本。** 四种 LoRA 变体与原 LoRA 在 16 个实验单元上做配对审计，15 个确认性单元经多重检验校正后仅 2 个支持原优势、4 个方向反转。PiSSA 在 Qwen 3B 的共同配方下高 2.62 点，按原论文配方低 1.30 点。Gemma-7B 按原配方和数据管线低 1.58 点。rsLoRA 在 BERT-base/MNLI 的 303 次扫描中，低于约 0.0005 的有效适配学习率区域 40/40 单元稳定，高侧 32 个中有 11 个出现坍塌。该阈值由特定模型和任务校准，另一 7B 设置控制 dtype 后在高侧仍可稳定。评估 LoRA 更新规则宜同时控制 rank、缩放和精度，保留配对种子与完整扫描。 [PEFT 审计论文集 PDF 表 1、图 2、Limitations](https://aclanthology.org/2026.aacl-short.32.pdf) · [会议 Session 3.1](https://2026.aaclnet.org/program/oral_program/#session-31-resources-and-evaluation-3)

#### 多语 MTEB 全局分数会掩盖目标语种的模型排名倒置

**主窗｜2026-11-07 10:30–12:00 上海｜AACL-IJCNLP 口头报告排程，论文集版本。** 对 17 个语种的 MMTEB 检索审计中，25 模型的实用名册上，按总分选任意模型对的语言内排序中位倒置率为 33%。先对每种语言求均值，再跨语言宏平均，在留一语言比较中将中位倒置率降到 14.5%。10 个语种只有两项检索任务，下游 RAG 只核了意、日、印地语的金融和法律。检索选型时可用目标语言与业务域的成对排序校验榜单，再报告有效任务覆盖数。 [MTEB 审计论文集 PDF §4.3、表 2、Limitations](https://aclanthology.org/2026.aacl-short.28.pdf) · [会议 Session 3.1](https://2026.aaclnet.org/program/oral_program/#session-31-resources-and-evaluation-3)

#### HorizonBench 按时间截断检索，测试科学构思能否接近后发表方案

**主窗｜2026-11-07 10:30–12:00 上海｜AACL-IJCNLP 口头报告排程，论文集版本。** 在发表前语料约束下，系统按生成构思与后来真实论文方案的嵌入相似度评估。表 1 中，仅给研究目标的朴素基线已达方法预测 soft recall 0.805，OSCAR 的学习重排为 0.807，平均相似度则为 0.759 对 0.753。51 名专家的相对排序与自动评估 Spearman 为 0.86。英文机器学习任务每个目标只有一篇参考论文，指标会漏记方向不同但有效的方案。复现宜同时固定时间截断、候选数量，报告最佳候选与平均候选，保留只给目标的基线。 [HorizonBench 论文集 PDF §5–7、表 1](https://aclanthology.org/2026.aacl-long.202.pdf) · [会议 Session 3.2](https://2026.aaclnet.org/program/oral_program/#session-32-llm-agents-and-learning)

#### 数学建模终端测评把可行性与数值接近度分开

**补录｜2026-11-05 16:18 UTC｜arXiv v1。** TB-MathModel 把 18 道竞赛题放进断网终端，使用多份人类获奖解的数值作为参照，并用结构化规则和 Lean 义务验证可判定的约束。表 2 的三次种子均值中，GPT-6 Astra + codex 为 64.58 ± 2.93，Fable 5.1 + claude-code 为 63.11 ± 1.33，DeepSeek V4.1 Flash + opencode 为 59.81 ± 1.25。顶两名间距处在种子波动内，成本是该次 18 题测试的 API 费用。两层验证覆盖可由产物判定的单位、范围和内部一致性。外部物理真值需要可独立重放的模型。复核应先比较判分器、题目方向及参考解覆盖，再比较端到端组合。 [TB-MathModel v1 §3、§4.2 表 2、Limitations](https://arxiv.org/html/2611.08237v1#S4.T2)

<a id="engineering"></a>
## 开源工程

### 升级与状态更正

#### vLLM 主干移除 Model Runner V1

**11-08 06:50（上海时间），默认分支，待发布。** GPU 路径统一为原 MRV2，`VLLM_USE_V2_MODEL_RUNNER` 开关和旧 runner 实现已删除，`vllm/v1` 仍是引擎的代码命名空间。配置校验仍会拒绝 `external_launcher` 下的多阶段 PP。仍依赖旧 runner 的 torchrun 部署应先在固定 commit 上复验启动参数、定制 logits 处理、speculator 和 pooling，再确定迁移基线。 [vllm #63373 `aac1b2a4f3`](https://github.com/vllm-project/vllm/commit/aac1b2a4f36e742c1f55bcb34ca9aaa5e6e02592)。

#### Megatron-LM 撤回四项提前合入的补丁

**11-08 06:55（上海时间），默认分支状态更正。** 11-07 23:00:26 至 11-08 06:37:20 先后落主干，11-08 06:54:07–06:54:40 撤回。按撤回顺序为 EP-invariant combine #8261 `ae47f9b5ea` → `a87e3059bf`、MFSDP v2 的 Muon 集成 #8548 `bc82433a06` → `6f501b6a04`、hybrid prefetch 窗口边界 #8787 `4e9de2872d` → `2ceec36c18`、整数 MoE 全局 aux-loss 计数 #8520 `d3660cba37` → `4f5f507a5e`。#9101 说明这四项须等完整批准及实际进入合并队列。上期记录的 MFSDP v2 权重释放和梯度规约拆分 #7902 `f1b38a8d8b`、免打包累加 #8949 `43be94fcd3` 属于各自独立的已合入变更。依赖被撤回四项的试验应改用已知 commit 固定配置并追踪重新合入。 [Megatron-LM #9101 `ba2b8fdf23`](https://github.com/NVIDIA/Megatron-LM/commit/ba2b8fdf23f3d11048ec1eff458b92d45ec708bb)。

### 训练、训推数值与恢复

#### DeepSpeed 修复 NVMe offload 的静默 I/O 失败和重复预取

**11-07 16:48–11-07 23:26（上海时间），默认分支，待发布。** DeepNVMe 检查异步完成结果，I/O 失败在等待或同步 API 抛错，对不满足 `O_DIRECT` 对齐条件或非连续的 tensor 用 bounce buffer。PR 在 GH200、ext4/md RAID 0 上复现了错权重，修复后 Qwen2.5-0.5B 初始化错参由 60/290 降到 0。ZeRO-3 同一 NVMe 预取窗口对参数去重，处理反向紧邻前向、activation checkpoint 重算及共享层。使用 NVMe 的训练可对照 CPU offload 检查初始化权重、恢复后 loss 及读写错误回报。 [DeepSpeed #8985 `1b022dcd4d`](https://github.com/deepspeedai/DeepSpeed/commit/1b022dcd4d45782f0af5b106cb3a2b2ca1092e16) · [DeepSpeed #9014 `6c7a6e0e76`](https://github.com/deepspeedai/DeepSpeed/commit/6c7a6e0e769bd519e49029814e51917025821463)。

#### DeepSpeed 修正 CPU optimizer 恢复精度与 AutoEP 折叠裁剪

**11-07 17:43–11-07 23:56（上海时间），默认分支，待发布。** CPUAdam、CPULion 和 CPUAdagrad 恢复时按 `fp32_optimizer_states` 取回磁盘中 FP32 状态，避免继承的 loader 按 BF16/FP16 参数 dtype 转换 moments。ZeRO-1/2 的 AutoEP+AutoTP 梯度范数改按实际 expert-DP 分片和 EP owner 求和，避免漏掉 TP peers 上不同专家。两条回归应分别比较不中断与恢复训练的状态，以及裁剪前后的全逻辑梯度和更新。 [DeepSpeed #8974 `b43c383ffb`](https://github.com/deepspeedai/DeepSpeed/commit/b43c383ffb8f1ff8a69c3e0094b13bb48e62da4f) · [DeepSpeed #8823 `a7ebdf96f0`](https://github.com/deepspeedai/DeepSpeed/commit/a7ebdf96f062461af2073f7c7b93d5e1febe39f0)。

#### Megatron 修正 GDN、MoE 辅助损失和 fused mHC 的训练路径

**11-07 10:20–11-08 04:06（上海时间），默认分支，待发布。** GDN 的复制 output-norm 参数按 TP rank 汇总梯度。确定性 MoE 模式允许走专门的 fused aux-loss 路径，含 TE wrapper 与路由调用回归。fused mHC 的 Sinkhorn backward 增补共享 scratch 同步。适合在 TP 切换、相同路由输入和 CUDA kernel 的数值回归中分别定位差异。 [Megatron-LM #7613 `9e5fb8ae25`](https://github.com/NVIDIA/Megatron-LM/commit/9e5fb8ae257ec579d04c32eb80f5f410c07fd651) · [Megatron-LM #7697 `efc46c5893`](https://github.com/NVIDIA/Megatron-LM/commit/efc46c5893e9a32541f276f58ba8a39672fe0173) · [Megatron-LM #7704 `fe283c254d`](https://github.com/NVIDIA/Megatron-LM/commit/fe283c254dde5529c512592c8f017c90a45ab73f)。

#### Megatron 修复 packed CP RoPE 频率与 mRoPE 角度精度

**11-08 01:31–11-08 04:06（上海时间），默认分支，待发布。** Packed mRoPE 保留全序列频率给 CP 路径消费，packed fused RoPE 保留真正的 per-token frequencies，mRoPE 用逐元素乘积计算旋转角，避开 matmul 在 TF32 设置下的舍入。值得用非均匀多序列、CP rank 重排以及启用 TF32 的多轴 position id 和非 fused 路径对照。 [Megatron-LM #7692 `54d22fe1ec`](https://github.com/NVIDIA/Megatron-LM/commit/54d22fe1ecffcf2f88617927dbb33d544b9c62d7) · [Megatron-LM #8377 `075972a2e4`](https://github.com/NVIDIA/Megatron-LM/commit/075972a2e44ce1726033234266549c95e868bc5d) · [Megatron-LM #8843 `7bc1aaec8d`](https://github.com/NVIDIA/Megatron-LM/commit/7bc1aaec8d3cdfba50c6a2d68ad8fa75eae538bf)。

#### Megatron 单 rank 恢复组保留 checkpoint metadata

**11-07 23:00（上海时间），默认分支，待发布。** fully parallel load 的单 rank 分支维持底层 checkpoint loader 返回的加载状态，避免提前从未读 metadata 的缓存覆盖。可用多 rank 保存、单 rank 组恢复的分片 metadata 与实际 tensor 内容做对照。 [Megatron-LM #8521 `eb6c402390`](https://github.com/NVIDIA/Megatron-LM/commit/eb6c402390093841f8f6ffe843d9402749189b78)。

#### PyTorch FSDP2 支持非零维度的不均匀切分

**11-07 14:37（上海时间），默认分支，待发布。** all-gather 对短 shard 和空 shard 按 rank 补齐，copy-out 重组原始张量。reduce-scatter 按各 rank 对齐并补零，回写时去 padding 并把非连续梯度压成紧凑 shard。作者附训练和集合通信测试。值得覆盖不能整除 mesh size、尾 rank 为空、与 Shard(0) 共同进入一个通信批次的参数。 [pytorch #203465 `398b4e5d02`](https://github.com/pytorch/pytorch/commit/398b4e5d029bf56d39a29c8f046c84917bbebf52)。

#### PyTorch 修正动态 scatter 和 Inductor 确定性配置

**11-07 09:34–11-07 17:23（上海时间），默认分支，待发布。** `scatter`、`scatter_add` 的 out variant 在 meta 路径用 SymInt 访问 index 的元素数和形状，修复 `torch.compile(dynamic=True)` 跟踪期崩溃。Dynamo 清理通过 C++ 接口恢复全局 deterministic algorithms 状态，避免覆盖独立的 Inductor `deterministic` 配置。可分别验证动态尺寸重复编译和独立的 Inductor 确定性开关。 [pytorch #194611 `039689b782`](https://github.com/pytorch/pytorch/commit/039689b782cf6a3f723f6a54e18cc461b0d2bc7d) · [pytorch #198572 `1a3f3982b1`](https://github.com/pytorch/pytorch/commit/1a3f3982b1cd4158460068c9e358895a7371b4ee)。

#### TorchTitan 把 rollout 换权重的 drain 放到 generator，把温度缩放放到编译区

**11-07 13:55–11-07 17:18（上海时间），默认分支，待发布。** `hot_swap=False` 时 generator 暂停纳入新请求，排空已有生成，再装载新版本。路由继续保持已有 session 的 sticky 归属，全部 generator 正在 drain 时按负载分流新会话。DAPO/GRPO 将逐 token temperature 的除法并入编译的 logprob/loss。PR 在 Qwen3-4B、TP2、10,240 token 的输出阶段报 28.1→22.2 ms/microbatch，约占训练前反向 0.7%。复验时分开记录换权重排队与训练步耗时。 [torchtitan #5905 `a9953e7b36`](https://github.com/pytorch/torchtitan/commit/a9953e7b3635720ce23be726598b6ba394a70366) · [torchtitan #5921 `201e79f610`](https://github.com/pytorch/torchtitan/commit/201e79f610dd380c3ec2fc37987079d86823d55f)。

#### TRL 在聊天模板重分词发生偏差时始终新建训练样本

**11-07 19:50（上海时间），默认分支，待发布。** Async GRPO 移除 `fork_threshold_tokens`，重分词结果只在精确延长已持有 token 时追加到原 row，否则分叉。之前实际采样的 token、logprob 与候选集仍有训练信号，同一 rollout 的各 row 共用 reward。建议用会删除 reasoning、改写 tool 结果或重分词边界的多轮模板，核对 completion mask、`fork_frac`、训练 token 数与额外前向成本。 [trl #8159 `65d1a055a4`](https://github.com/huggingface/trl/commit/65d1a055a415579afe2099c0922ba8c19595bc54)。

#### VeOmni 为 AdamW 加入主机侧状态 offload

**11-05 10:24（上海时间），默认分支补录。** `state_offload: cpu` 让 AdamW 的持久 optimizer state 驻留 host，按参数组块搬运并和计算重叠，DCP 检查点补充对应的保存恢复和 DTensor 状态处理。测试覆盖 CPU 逻辑以及 GPU/NPU 套件的配置入口。适合以相同 batch 比较设备峰值、每步传输和中断恢复后的参数/状态一致性。 [VeOmni #1286 `e6486d3847`](https://github.com/ByteDance-Seed/VeOmni/commit/e6486d3847536d6a5cf72a433040914b69f1957d)。

### Serving、KV 与投机解码

#### SGLang 扩展 Kimi K3.1 的 prefill CP、分段 CUDA graph 和 decode CP

**11-07 09:11–11-07 14:26（上海时间），默认分支，待发布。** Prefill 可共享 CP/TP 通信组，后续补丁使 MegaMoE、DFlash、可分段 CUDA graph 和 PP×prefill CP 接入。Decode CP 侧另接 DSpark、DFlash 和 HiCache，补了 draft KV pool 与分片位置处理。相比上期 DCP 基线，值得用实际模型分别覆盖 chunked prefill、PP×CP、长前缀、draft rollback 和缓存命中，按 PR guard 核对运行拓扑。 [sglang #48709 `6d2e648803`](https://github.com/sgl-project/sglang/commit/6d2e64880322d4fb95f702aeba81666d56dedc90) · [sglang #48710 `79a3b21956`](https://github.com/sgl-project/sglang/commit/79a3b2195624aa34fb8480afc99f39418dc6127d) · [sglang #48712 `4bef574039`](https://github.com/sgl-project/sglang/commit/4bef574039d853f5df2ec52d8c4ad02d6705f76b) · [sglang #48739 `709b8b3c79`](https://github.com/sgl-project/sglang/commit/709b8b3c79467c67bff11397b62ebe251a62559e) · [sglang #48877 `b165323cb6`](https://github.com/sgl-project/sglang/commit/b165323cb648e89d0922ce15613f984f4f84b4fd)。

#### SGLang 调整 DeepSeek V4.1 的 CP 通信和 padding

**11-07 09:26–11-07 19:08（上海时间），默认分支，待发布。** MoE 输出在可用分支用 reduce-scatter 代替先 all-reduce 再本地 scatter，减少通信及中间结果。MLP 同步补齐的 extend 中，attention 只消费真实 token 行，另一提交把 DSpark replay tail 在 prefill CP gather 前按 rank 散开。适合用真实不齐 chunk、不同 CP/TP 组和可接受长度核对输出、通信量与 speculative acceptance。 [sglang #47247 `8920463c90`](https://github.com/sgl-project/sglang/commit/8920463c90c81127d9413f093240aeba17983f3a) · [sglang #48980 `c840718849`](https://github.com/sgl-project/sglang/commit/c84071884945274205ffd62aa935571fe7564047) · [sglang #45534 `6de2991cec`](https://github.com/sgl-project/sglang/commit/6de2991ceca930b68e55dc2ee8a47df9535cbf3f)。

#### SGLang 在 P/D 握手检查投机 KV 布局并修复调度器释放

**11-07 16:46–11-07 17:30（上海时间），默认分支，待发布。** 引入 `spec_kv_layout`，使 DSpark/DFlash 的 prefill 与 decode endpoint 在注册时校验兼容性。PP 路径开始遵守暂停生成，同时清理中止的 P/D room 和被拒会话。适合用不匹配的双端版本、流量排空和预填充中止进行早失败和资源释放回归。 [sglang #48375 `043cfc4d3a`](https://github.com/sgl-project/sglang/commit/043cfc4d3a2cc2a931d15862f08c4b60b692c957) · [sglang #48992 `3572b9c318`](https://github.com/sgl-project/sglang/commit/3572b9c318b955e5f2e04754ce7c660591d237ca)。

#### SGLang Rust 前端补访问密钥保护与多 DP 排空

**11-07 12:44–11-07 16:59（上海时间），默认分支，待发布。** 服务信息、日志、启动命令和 crash dump 脱敏访问密钥，Rust 前端对支持的 API key 执行校验，对不支持的部分参数和 warmup 失败提前报错。DP rank 0 的 drain 会等待所有 attention-DP listeners。上线 Rust 前端时可用带认证、故障启动及滚动下线的服务回归验证行为。 [sglang #48518 `e93aacbd06`](https://github.com/sgl-project/sglang/commit/e93aacbd0663cea45c35f2677eb1618783592d32) · [sglang #48512 `18256b2b19`](https://github.com/sgl-project/sglang/commit/18256b2b196b9a247fe816150ce4826132cc1333) · [sglang #48492 `99833a441e`](https://github.com/sgl-project/sglang/commit/99833a441eb53fb2afedf36d97f9b0b400b6a275)。

#### vLLM 的 CPU KV offload 可为每 rank 分配一块 HugeTLB 内存

**11-07 15:12（上海时间），默认分支，待发布。** `SimpleCPUOffloadConnector` 每 rank 用一块 host backing，Linux 5.14+ 可选 2MB 或 1GB HugeTLB，未设置使用普通页并可请求 THP。显式 HugeTLB 配额或池不足会导致启动失败。容器需配额与各 rank 向上取整后的大小匹配，NUMA 仍跟随 worker 策略。建议并测 pin/register 时间、跨 NUMA 拷贝和重启配额失败。 [vllm #60130 `7a07e2342e`](https://github.com/vllm-project/vllm/commit/7a07e2342e7e857133d3964bd4b7022103a5b030)。

#### vLLM 调整 MLA DCP A2A，并修复 encoder-only full graph

**11-07 20:54–11-08 03:29（上海时间），默认分支，待发布。** MLA decode 在满足条件时先对局部 value 做 up-projection，再合并 DCP 的 A2A 结果，测试比较不同 LSE/投影布局。Encoder-only FULL CUDA graph 则沿用 capture 承诺的 `max_query_len`，使实际 query 与 replay 的 launch 上界一致。可分别压测 DCP 和小长度捕获后大长度重放。 [vllm #64994 `1ecc9cc4ce`](https://github.com/vllm-project/vllm/commit/1ecc9cc4ce6626352df39e4364579ec376c10761) · [vllm #65275 `35b4c82ee4`](https://github.com/vllm-project/vllm/commit/35b4c82ee4e9a1b9dab9d09102056da6de4671b5)。

#### TorchTitan 在混合生成步拆开 FA4 decode 与 prefill

**11-08 08:04（上海时间），默认分支，待发布。** 原先同一次 varlen call 由长 prefill row 决定 `max_seqlen_q`，单 token 行也会用到慢配置。新代码在 device 上切分 offsets，两次 FA4 写同一输出缓冲，适配 CUDA graph。PR 在 GB300、Qwen3.5-27B、FA4 beta38、128-token 页和约 9K KV 下，100 个 decode 加一个 650-token prefill 的单层时间为 6.59→1.92 ms。适合以真实多轮 RL 请求复验 P99 ITL、空分段和输出一致性。 [torchtitan #5941 `239a73f902`](https://github.com/pytorch/torchtitan/commit/239a73f902504435c723a42444a0998b398e2bcc)。

#### vLLM 合并 rollout 稀疏权重补丁的验证阶段

**11-07 16:41（上海时间），默认分支，待发布。** 把 sparse patch 的元数据检查和条目准备合并为一次遍历，并让完整校验在张量写入前完成。新增测试包含无效补丁和目标不被部分更新的检查。适合比较小补丁与大批更新的 refit 耗时，并对非法末条条目注入故障。 [vllm #66021 `dfebb42d09`](https://github.com/vllm-project/vllm/commit/dfebb42d096d5c6a04aa5a1722195e37cb8e0b2d)。

#### TensorRT-LLM 主干新增 Rubin 专用 GLM 路径，并调整 NVFP4 kernel 路由

**11-07 14:11–11-08 08:55（上海时间），默认分支，待发布。** GLM-5.2 NVFP4 的实验性低延迟 decode megakernel 需显式 `TRTLLM_GLM5_MEGAKERNEL=1`，适用 SM107 的 212 SM 设备、TP16、单序列和特定 verify 形状。另一提交把 NVFP4 activation quant 并进 SM100+ SwiGLU 与 SM12x add+RMSNorm。Ada/Hopper 上 Qwen3.5/3.8 的 dense NVFP4 checkpoint 默认走 Marlin。可按三类硬件分别核对实际 dispatch、逐 token 对照与尾延时。 [TensorRT-LLM #21590 `fe6938adb1`](https://github.com/NVIDIA/TensorRT-LLM/commit/fe6938adb1cd0a17f16eb7da68d4b824e2909f51) · [TensorRT-LLM #21273 `c16aa42c1c`](https://github.com/NVIDIA/TensorRT-LLM/commit/c16aa42c1cc4b0c401fb9d53f2eec05d4f562d77) · [TensorRT-LLM #21434 `bcb64e1459`](https://github.com/NVIDIA/TensorRT-LLM/commit/bcb64e1459646102a03d1fa4f925ef8ad863c24b)。

#### TensorRT-LLM 补 native P/D 的 KV 接收结果回执

**11-07 22:33（上海时间），默认分支，待发布。** 原生 disaggregation 的 decode 侧生成 KV 结果回报 token，让 scheduler 消费接收结果并处理失败。升级 P/D 运行时宜在固定双端版本上注入接收失败和请求中止，对照记录实际等待与资源回收。 [TensorRT-LLM #21538 `9484c59210`](https://github.com/NVIDIA/TensorRT-LLM/commit/9484c5921027f2024e888bc129f3b73a582c1939)。

### 算子、编译器与部署工件

#### FlashInfer 将批次不变数值扩展到 attention 和 logprob

**11-07 09:26–11-07 15:16（上海时间），默认分支，待发布。** 三类 paged/ragged attention wrapper 显式 `batch_invariant=True` 时使用固定 128-key tile、512-key chunk 和合并顺序，目标是同 token 的 prefill/decode、页表和批次变化逐位一致。范围为 SM100/103/107、BF16、D=64/128/256 的因果路径，低精度 KV、soft cap、LSE 输出等组合报错。采样侧新增 batch-invariant softmax 与 `log_softmax`、指定 token 的 `log_probs`，固定逐行归约，输入支持 BF16、FP16、FP32。PR 的 GB200/GB300 对照表报告部分形状的性能代价。值得用自身 rollout 的长度和 batch 分布联合测逐位匹配、吞吐及尾延时。 [flashinfer #8590 `be2c1dccee`](https://github.com/flashinfer-ai/flashinfer/commit/be2c1dccee2b73f0b6dd0dcd1bc20f855491602f) · [flashinfer #8999 `91e76b20d3`](https://github.com/flashinfer-ai/flashinfer/commit/91e76b20d38803ce95ebeb758a91ee7306571deb)。

#### FlashInfer 修复 Blackwell GDN 的 TMEM 回收顺序

**11-07 09:11（上海时间），默认分支，待发布。** GDN 的 CP、prefill 和 chunked kernel 在 allocator warp 释放 TMEM 前同步会读取同一块的 compute warps，补齐 `tcgen05.ld` 到 `tcgen05.dealloc` 的跨 warp 依赖。适合固定 CuTe 及驱动版本，对照不同 CTA 驻留和连续 launch 的 recurrent state。 [flashinfer #8854 `56d946ed1b`](https://github.com/flashinfer-ai/flashinfer/commit/56d946ed1b54f30b8a1c5786f4a09f2e569860ca)。

#### Triton 优化 NVIDIA 布尔归约

**11-07 16:58–11-07 17:15（上海时间），默认分支，待发布。** 子 warp 布尔 any/all 用 ballot，适用时全 warp 继续使用 vote。宽寄存器归约交错三条谓词累加链，最后在谓词寄存器合并。作者的重复 fold 微基准 256 values/lane 报 20.27→5.20 μs，收益取决于每 lane 工作量。可检查真实 mask kernel 的 PTX、寄存器与端到端耗时。 [triton #12647 `527657e26b`](https://github.com/triton-lang/triton/commit/527657e26b33e1cfd13a5fae38ae2369ff65daaf) · [triton #12650 `e8dda3400e`](https://github.com/triton-lang/triton/commit/e8dda3400e82957bf3408c8251619790c78d8d5c)。

#### FA4 显式传递 backward tile 坐标并覆盖 SM107 autotune 编译

**11-08 02:56–11-08 07:22（上海时间），默认分支，待发布。** CuTe SM100 家族 backward 显式传 work tile 和 head 坐标，SM107a 另加入候选编译覆盖。适合固定 beta38 与两个后续源码提交，复验目标架构的 forward/backward 候选兼容和变长梯度结果。 [flash-attention #3141 `bd964619e4`](https://github.com/Dao-AILab/flash-attention/commit/bd964619e487a8c9be8a11f65736dd781f851135) · [flash-attention #3121 `b2edfab9f8`](https://github.com/Dao-AILab/flash-attention/commit/b2edfab9f8fa81205884cdc9c0b8859cd32934d3)。

#### SGLang 的 AMD custom all-reduce v2 接入 MI450

**11-08 05:27（上海时间），默认分支，待发布。** MI450/gfx1250 使用单独的启发式表和 HIP VMM FABRIC handles，另更新 handle 导入及设备资格。应在目标 ROCm 和拓扑上先测 peer 映射、进程退出和大小分界，再比较 NCCL/RCCL 路由的吞吐。 [sglang #49001 `ba48fc1a9e`](https://github.com/sgl-project/sglang/commit/ba48fc1a9eb3c0e6046d2069bc073c3a8b88e20a)。

#### AITER 修复 gfx1250 HCA 压缩 attention 两段的 padding

**11-07 12:21–11-07 22:44（上海时间），默认分支，待发布。** 两处 FlyDSL 补丁分别在 Phase 1 和 Phase 2 归零不应消费的 KV，防止无效或上一次迭代的数据进入当前聚合。用不齐块、复用 workspace 和含极端值的输入对照 reference，能覆盖两处边界。 [aiter #7601 `2530d10943`](https://github.com/ROCm/aiter/commit/2530d1094356f644a77835897aaa6abdc1952f7e) · [aiter #7550 `d67fc6d240`](https://github.com/ROCm/aiter/commit/d67fc6d24099bb01ed73b7bdec5e7998aada40df)。

#### Transformer Engine 修正 graph-safe grouped 层开销及 CPU 量化 tensor dispatch

**11-07 10:25–11-07 14:29（上海时间），默认分支，待发布。** graph-safe GroupedLinear/GroupedMLP 减少 CPU 侧操作，CPU-resident QuantizedTensor 的操作改走可终止的分派以修复 RecursionError。分别用 CUDA graph replay 和 CPU 存放后再迁回 GPU 的操作回归验证。 [TransformerEngine #3752 `f0ee13b011`](https://github.com/NVIDIA/TransformerEngine/commit/f0ee13b0115fc15317cba3e14b0c38d9c00cc2d8) · [TransformerEngine #3812 `c1c2d4d2fa`](https://github.com/NVIDIA/TransformerEngine/commit/c1c2d4d2fa008386f247c2fec56a998e27134e0a)。

#### llama.cpp 增加 MoE cache miss 的 CPU 计算并限制 SYCL 显存分配

**11-08 00:20–11-08 07:13（上海时间），默认分支，待发布。** MoE cache 可在 CPU 上计算未命中的专家并提供调参脚本，SYCL 分配时检查设备剩余 VRAM。适合在紧显存条件比较 cache hit、CPU miss、host/device 数据搬运和 token 尾时延，并验证多 buffer 时显存估计。 [llama.cpp #30966 `e95426f169`](https://github.com/ggml-org/llama.cpp/commit/e95426f16982054c379ca7bdc8186996dd22f0fe) · [llama.cpp #31285 `818ac6b078`](https://github.com/ggml-org/llama.cpp/commit/818ac6b078e345e59ffa4b1ed3b1ce1b0ae83827)。

#### NeMo RL 扩展 Single Controller 的 forced-ratio 组合

**11-07 17:18（上海时间），默认分支，待发布。** `ready_first` 取样现允许 GRPO 使用 forced ratio，一轮 PPO 可用相应配置。补丁补足配置验证和测试。值得在与 `v0.8.0` 相同配方对比采样日志、ratio 权重及策略陈旧度。 [RL #5079 `cc2329e0ba`](https://github.com/NVIDIA-NeMo/RL/commit/cc2329e0bad8a58a427f3496c41b1fb83154baf9)。

#### LMCache 为 MUSA 增加原生 CacheGen codec

**11-07 13:43（上海时间），默认分支，待发布。** MUSA 路径使用原生 CacheGen codec。PR 覆盖平台 dispatch 和编解码，使用时可对照其他已支持后端的序列化、恢复正确性和单独的缓存吞吐。 [LMCache #5865 `9857feb35a`](https://github.com/LMCache/LMCache/commit/9857feb35aa8aa8336b056d4e4e486d9573853bd)。

#### Infera 从 Pareto sweep 选择不同延迟档的部署组合

**11-05 14:13（上海时间），默认分支补录。** `inferasim mix` 按模型、场景和延迟档选一个配置，先筛掉违反约束的点，再用每 GPU 吞吐和 SLO 余量排序，输出 JSON/CSV。结果来自投影数据，适合把候选缩到少数再用真实负载核验排队与硬件表现。 [Infera #230 `82fd851bbe`](https://github.com/AMD-AGI/Infera/commit/82fd851bbe4b65b7dede309444fae0659ff12a2d)。

#### LMCache 发布四种平台的 0.5.8.dev61 nightly wheel

**11-07 15:35（上海时间），nightly 预发行工件。** CUDA 12.9 和 13.0 各有 12 个 Linux wheel，覆盖 amd64/arm64。ROCm 7.2 工件为 gfx942/gfx950 的 CPython 3.12 x86_64，声明匹配上游 vLLM ROCm 镜像的 Torch 2.13。Rebellions RBLN 工件针对 torch-rbln 0.12.0 构建，驱动、torch-rbln 和编译器由宿主提供。四个可移动 nightly tag 应按实际 wheel 文件和 SHA256 固定后再复现。 [nightly-cu129](https://github.com/LMCache/LMCache/releases/tag/nightly-cu129) · [nightly-rbln](https://github.com/LMCache/LMCache/releases/tag/nightly-rbln) · [nightly-rocm](https://github.com/LMCache/LMCache/releases/tag/nightly-rocm) · [nightly](https://github.com/LMCache/LMCache/releases/tag/nightly)。

#### attention-gym 0.0.18 发布可自动调优的 paged MLA decode

**11-06 03:28（上海时间），正式版补录。** 自定义 op 在 SM90 走 FA3，SM100/103 走 FA4，支持 `torch.compile(fullgraph=True, dynamic=True)`。FA4 可按 head、页长、行数和 KV 长度桶调候选，CUDA graph capture 内要求提前完成 tuning。PyPI 已发布 wheel 和 sdist。PR 在 B200、BF16、64-token 页的形状组合测得 1.074× 几何平均、范围 0.975–2.14×，换成同桶混合 KV 长度后范围 0.910–1.99×。值得固定 FA4 版本和 KV 分布验证动态编译及尾时延。 [正式 release](https://github.com/meta-pytorch/attention-gym/releases/tag/v0.0.18) · [MLA autotune PR](https://github.com/meta-pytorch/attention-gym/pull/690) · [PyPI 工件](https://pypi.org/pypi/attn-gym/0.0.18/json)。

verl、slime 与 OpenRLHF 在已检查的默认分支、merged PR 和 release 来源中，主窗无值得收录的新合入或正式分发。核心仓库的 PyPI 清单未出现主窗新包。上述结论限于本轮可达入口。

<a id="hardware"></a>
## 硬件与软件协同

### 补录｜昇腾 Kimi K3.1 Pro 配方拆开索引量化、稀疏 KV 和 KDA 的上下文并行

**昇腾 CANN · MR !1252 于 11-05 09:25:36 合入，技术文章 11-06 09:21:26 发布，均为上海时间。** K3.1 的 DSA 路径把打分、head 加权和 Top-K 选择融合，直接读分页索引 K。索引 Q/K 用 32 元素一组的 MXFP8，主稀疏注意力的 KV cache 用每 token、每 head 的 128 元素 FP8 tile，投影保持 BF16。DSA prefill 把序列首尾分片配对来平衡因果索引工作量。FlashKDA prefill 则通过 AllGather 交换分片状态摘要、合成各卡起始状态，再并行执行本段。

厂商给出 32 张 950DT、100K 输入、global batch 32、DSpark 接受长度 4 时的平均 decode 模型计算时间低于 10 ms/token。文章只统计模型耗时，没有给服务端到端时延、分位数、对照配置或量化质量结果。固定配方指向 CANN 9.2.0 weekly、950 对应算子包及 torch_npu 2.10.0.post4，并说明两个 SiTU 自定义算子需要同包安装。**建议考虑复现**。先核对量化对 Top-K 和输出的影响，再分开测 prefill、目标模型 decode 和 DSpark 接受率。 [已合入 MR !1252](https://gitcode.com/cann/cann-recipes-infer/merge_requests/1252) · [固定提交的部署配方](https://raw.gitcode.com/cann/cann-recipes-infer/raw/4a09b2a095bd9f2d9120cffb7ddffd2065ab0d55/models/kimi_k3_1/README.md) · [昇腾技术文章 §2–6](https://cann.csdn.net/6aed2b9c40858c2d8818d037.html)

### 补录｜MLxcel 在 Strix Halo 上公开 ROCm 低精度移植的基线和失败路径

**Lablup · 技术报告标 11-06，v0.8.1 于 11-03 01:47:05（上海时间）发布。** gfx1151 缺少原生 FP8/FP4 矩阵指令，ROCm 后端在 kernel 内将块量化权重解码成 FP16 或 BF16。作者对 MXFP4 prefill 按专家分组，并将 NVFP4 的 CPU 反量化移到 GPU，避免 CPU 读取 GPU carve-out 时拖住执行。小行数选择融合 WMMA，大形状可走反量化加 hipBLASLt。v0.8.1 固定到 `2be2ab8b`。

作者在 Ryzen AI Max+ 395 / Radeon 8060S、Debian 13、ROCm 10.0、96 GiB BIOS carve-out 下，以 512 输入、128 输出、batch 1 跑通选出的 44 个模型。gpt-oss-20b 的 MXFP4 prefill 从移植初期的 7.5 到 526 token/s，decode 从 8.3 到 63.6 token/s。这组倍数基于项目自身早期实现。文章里 Gemma 4 31B 的 llama.cpp 对照来自其他机器，不能直接用于引擎选型。**建议深入阅读并考虑复现**。固定量化文件与内存配置，比较短输入和专家分组的拐点，分别记录单请求和并发吞吐。 [移植报告](https://www.backend.ai/blog/releases/2026-11-mlxcel-version-0.8-released) · [v0.8.1 release](https://github.com/lablup/mlxcel/releases/tag/v0.8.1)

### 补录｜WHIRL 0.2.5.1 把 MXFP4 解包融合进 8060S prefill，并拆开投机效率与 kernel 时间

**WHIRL · 11-06 03:54:02（上海时间），v0.2.5.1，提交 `fc42f1f5`。** 在 Radeon 8060S 上，MXFP4 prefill 矩阵乘内解包权重，Swift 的投机校验也更新了 int8 KV attention 和矩阵向量路径。作者用 80 W 的 ROG Flow Z13（64 GB UMA）、同量化文件和提示对比上一版，每点取三次中的最佳结果。Swift 27B MXFP4-A 的 balance 档在 2K–128K prefill 提升 4.2%–9.1%，128K 文档问答的 decode 为 25.0 token/s（+12.1%）。Ornith 35B-A3B MXFP4 的 64K balance decode 反而由 76.3 降到 71.5 token/s，作者同时记录了草稿接受数 0.71→0.60 和单步约 0.5% 的提速。二进制在两模型、三档模式的 golden reference 及 12 个 4K/16K 长提示检查上与上一版相符。R9700 表格复用旧版数据，部分 llama.cpp 对照也复用旧测且 HIP runtime DLL 不同。**建议考虑复现**。固定输出轨迹、配对比较每步耗时与接受数，再比较端到端吞吐。 [正式 release 与测量条件](https://github.com/tsaipifong/whirl-llm/releases/tag/v0.2.5.1)

### 补录｜vLLM-MUSA 0.28.0 补齐混合 KV、PD 分离及小批 MoE 路径

**摩尔线程 · 11-03 20:49:24（上海时间），正式 v0.28.0，早于本轮回溯窗。** S5000 插件随镜像对齐上游 vLLM 0.28.0（`2cf0a69`）。此版本仍默认 Runner V1，V2 需显式选择，DSpark 有单独覆盖路径。Mooncake 支持 MUSA FlashAttention 的 K/V-first cache 布局及 1P1D 示例，并修复未上报失败 KV load。混合 attention 的 64-token fast path 遇到显式 block size 会停用，GLM DSA 的 MATE prefill 只在 64 head、576 width、Top-K 2048 的形状启用。

在 S5000 上，厂商把小 M（M ≤ 32）的 MoE 默认 K-loop pipeline 从四级改为一级。Nemotron-3.5 MTP6、4096 入/128 出、单请求时为 11.635→9.563 ms/token，九条请求输出哈希一致，TTFT 1648→1678 ms。该结果限定于这条配置。**建议先读升级说明再复现**，核对镜像内耦合版本，测试 Runner V2 迁移和 Mooncake 错误回退。DeepSeek-V4-Flash-0731 的 128K 1P1D 示例还要求 RoCE 与约 400 GiB 空闲主机 DRAM。 [v0.28.0 release](https://github.com/MooreThreads/vllm-musa/releases/tag/v0.28.0) · [小 M kernel 配置补丁](https://github.com/MooreThreads/vllm-musa/pull/242)

### 补录｜cann-local-lm 增加 ExecuTorch 的昇腾端侧执行入口

**昇腾 CANN · Runtime 于 11-04 09:29:42（上海时间）合入，社区 11-06 介绍仓库。** 新的 `executorch_npu/` 把 PyTorch 图划分、编译为 PTE/PTD 工件，再通过 C++ backend 加载。仓库列有 manifest 校验、stateful API 及 Qwen3.5、Gemma 4 的示例。既有 `cann_local_lm` 路径则是 HF 模型经 epair 和 omc 编译后由统一推理 pipeline 调用。主 README 的模型与硬件表当前只列 Gemma4 / KirinX90，示例范围、交付版本和验证矩阵需分别核对。**建议继续观察**公开工具链、芯片与模型的版本配套，再在匹配分支上测状态缓存及端到端内存峰值。 [已合入 Runtime MR !35](https://gitcode.com/cann/cann-local-lm/merge_requests/35) · [仓库说明](https://gitcode.com/cann/cann-local-lm) · [社区公告](https://cann.csdn.net/6aed9e8a40858c2d8818f153.html)

### 补录｜Ascend 310P 热切模型暴露事件句柄寿命

**昇腾社区 · 11-06，单卡推理故障复盘。** 停旧服务再启新模型时，第二次部署报 `halResourceIdAlloc eventid failed`。复盘把两次分配日志对齐，发现第一次部署分配的句柄未回收，第二次耗尽案例所述 1,023 个 event ID。文章给出显式销毁事件、检查 stream/context 清理和读设备资源计数的方法。未披露服务框架、模型及实际运行 CANN 版本，API 参考指向商用 8.3.RC1。**建议深入阅读**日志对齐方法。在使用 AscendCL 的端侧服务中，验证多次切模后 event 计数能否回落到空载基线。 [官方技术复盘](https://www.hiascend.com/developer/techArticles/20261106-107)

### 补录｜沐曦披露 128 卡全交换与跨芯片的推理拆分

**沐曦 / 东方算芯 · 11-05，厂商展会披露。** 曦景 S600 展示 C600、MXMACA 和 MetaXLink 的单柜 128 卡全交换架构，厂商称交换无带宽收敛并支持内存语义。另一个异构机柜展示两种映射。MoE 的注意力放 C600、专家计算放 DF1000。投机解码由 DF1000 生成草稿、C600 批量校验。披露没有给可用软件包、通信矩阵、指定模型的负载、对照基线或交付状态。**建议继续观察**实测。跨芯片链路在 token dispatch/combine 或 draft/verify 间的开销将影响这些设计的适用负载。 [厂商技术披露](https://www.metax-tech.com/en/ndetail/12690.html)

<a id="coverage"></a>
## 覆盖与续查

- **模型、厂商技术报告与后训练。** 定向 Hugging Face 查询覆盖 91 个组织的模型与数据集、创建与更新排序，共 364 条请求，初始 363 成功。LG 组织一条超时后返回空数组，仍保留来源身份复核。定向目录有 203 个重叠窗候选，对 50 个主窗仓库尝试固定版复核，部分文件受 401、429、超时和格式差异限制。93 个厂商及基准站点入口中 73 个成功，20 个失败，动态历史仅选择性核对。HF 全局模型创建索引首次第二页超时，重试取得 5 页、4987 个截至截点的重叠窗候选，最早拉至 2026-11-06T13:30:05.000Z，只作目录和抽筛，语义待审。本节主窗 2 个归并事件，回溯 5 条。GPT-6 Cyber、Inkling 2 Small、Arete、Drex、Cola2、ToyBlocks、VIALS-AA 按历史去重。腾讯 DSpine-27B 与 Overnight 续至下一期。
## 论文来源覆盖

检查 arXiv 九分类 11-06 的现有公告（918 个稳定作品 ID），HF 11-06 发现页 32 个 ID，11-07 API 返回空数组。AACL 核对 11-07 四篇正式口头报告排程与全文，并按原提交日补录八篇 arXiv 预印本与一项旧稿实质修订。宽 Atom 末端 11-05T18:59:48Z，真实修订查询和 OpenReview 时间线待补。 另比对 KV 量化对齐论文的两版固定 PDF，抽查五个作者仓的固定提交源码。

未闭合的窗口与来源见 ledger.json 中的 pending。
- 检查 59 个 GitHub 仓库入口，54 个默认分支履历完成分页，54 个 release Atom 返回 HTTP 200。GitHub merged PR 分页闭合主窗 26 个仓库、回溯窗 15 个仓库，另查 23 个 PyPI 包公开文件清单及 attention-gym 定版工件。打开 129 个提交或 PR 补丁，收录 61 个工程事件（补录 3 个）。未逐条深核剩余候选、维护分支和部分工件。PyTorch 的 GitHub merged 查询不能覆盖其镜像落地方式，本文使用其默认分支 committer 时间。GitHub Atom 的更新日期仅用于发现。
- verl、slime 与 OpenRLHF 在已检查的默认分支、merged PR 和 release 来源中，主窗无值得收录的新合入或正式分发。核心仓库的 PyPI 清单未出现主窗新包。上述结论限于本轮可达入口。
- **硬件及组件。** 直读 NVIDIA CUDA/NCCL/Triton 技术与版本入口、AMD ROCm 博客和版本索引、Intel Gaudi/Compute Runtime/oneDNN、Ascend 产品公告与技术文章、Google Cloud TPU 和 libtpu 索引、AWS Neuron 文档及组件源、Apple MLX release，并用中英文检索补入 MLxcel、WHIRL、vLLM-MUSA、CANN 端侧仓和沐曦。Ascend 原请求返回 HTTP 200 内嵌 406，补上 Referer 后核实了两页公告和技术文章列表。GitCode 组件 tags 恢复，HCCL/runtime 均可钉住旧 v9.2.0，分组件 release 列表仍为空，9.2.1 组件映射续查。
- **硬件未覆盖项。** libtpu 0.0.52 的行为变化、AWS Neuron 部分包的发行历史和 A5X 的 3.6/3.0 TB/s 口径冲突仍未闭合。Azure HPC 博客 403，AMD PyTorch 大会页面的直连失败，部分聚合入口只覆盖到 11-06。摩尔线程 Kimi K3.1 Pro 基础推理有报道，已打开的正式 v0.28.0 release 缺少该模型的配方，本期保留待查。Bedrock Kimi Highspeed 官方公告有显式 prompt caching 与跨区推理信息，但未公开可比吞吐或硬件栈，本期作为部署可用性线索记录。

- **续扫边界。** 本轮发现截止为 2026-11-08 09:00:30。下轮从该位置续扫并回溯 72 小时，同时继续处理分源待查记录。全局完整语义水位仍未闭合。本轮未做独立复训、设备压测或付费 API 实测。
- **更正与补录。** Megatron-LM 已撤回的四项补丁按截点状态记载，并保留原合入和撤回的时间。Beam 报告仅核实到 PDF 版本与内容。补录条目保留原始发布、工件或提交日期，会议演讲时间与论文集的出版日期分开。
