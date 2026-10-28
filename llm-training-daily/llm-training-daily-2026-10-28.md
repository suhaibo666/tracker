# LLM 技术动态日报 · 2026-10-28

> 时区：Asia/Shanghai（UTC+8）。增量扫描窗口：**2026-10-27 09:00:54 → 2026-10-28 09:02:09**；72小时回溯：**自2026-10-24 09:00:54起**。实际生成时间：**2026-10-28T09:24:21+08:00**。原始精确时间戳保留UTC；只有日期的公告不推定具体首发时刻。
>
> 已读取自动化记忆、持久化状态、21篇本地历史日报及认证远端main目录、索引与提交；历史日报及索引的22个Git blob均与本地吻合，远端核对基线为 `aba2fae654919b78e041c338ddcb7dae85caea6b`。本轮无待补传旧文件；只维护llm-monitor与tracker/llm-training-daily，未改知识库wiki。
>
> 以下论文和性能数据均区分作者报告、来源测试、本轮文件/数据检查与独立复现。**本轮未做模型训练、GPU/NPU/TPU性能或全量数据污染复现**。来源枚举、正文语义审阅与远端归档进度分别记账；扫描水位推进不表示历史缺口关闭。

## 本次重要变化

- **新权重带来明确的复现入口，也带来配置边界。** LFM3交付1.7B预训练基座及后训练工件，披露22T预训、末3.5T mid-training；Ling-3.1-flash-1019开放560B/25B权重，但卡称1M与发布配置262K冲突。许可、端侧构建与评测条件分别核对。
- **低精度研究应同时验收误差与完整步成本。** S3R-EC在TPU修相关舍入误差，dense短程整步快、MoE却慢；MXFP8 RL的匹配网格改善部分一致性，但短序列可反慢，显存也可能增加。没有把微测当长期收敛证明。
- **训练工程优先看恢复精度和对象寿命。** DeepSpeed跳过optimizer moments仍保留FP32 master；Megatron固定pinned offload buffer、FP32 token缩放、MFSDP事件所有权与TE的量化回写/hook路径都有实质增量。
- **压缩存储不等于扩大decode容量。** DCKV的56–60倍针对压缩payload，解码前恢复BF16且峰值略增；RIPE近期窗口仍缺真实投机decoder计时。KV/SSM混合布局、CoW、prefix命中和P/D完成顺序的修复更应优先回归。
- **RL探索与Agent harness研究更可操作，但成本不可省略。** proposal rollout、solve-none引导、critic/branching和联合harness演化均给出实验，教师、critic、验证和最佳checkpoint选择的额外成本逐条保留。
- **投机解码从草稿质量走向全链路契约。** 新DFlash2 checkpoint、目标重生成的负面结果、grammar-invalid proposal处理、NPU/Mac processor历史等说明AL、分布正确性与端到端吞吐要分别验收。
- **正式release与main/功能分支分账。** Triton3.9.0公开发布；同日warp-ID折叠优化已回退。TensorRT-LLM部分Helix/DFlash、AITER融合通信及SGLang分支项按实际base标注；新增backend可能更慢，默认量化也可能损伤长推理质量。
- **评估新增强调可验证的失败边界。** InfraDev与BIEvolve补数值/资源/批不变性验收；Beam文档、novelty judge和专家agent实验展示配置、prompt与人类判断的混杂。SynthDocQA、Open-SWE及ChartNet新工件须先核格式、失败轨迹和同源切分。

目录：[模型与报告](#models) · [预训练与架构](#pretraining) · [后训练](#posttraining) · [推理与部署](#inference) · [核心开源](#core) · [扩展工程](#extended) · [硬件软件协同](#hardware) · [评估与可靠性](#evaluation) · [版本与覆盖](#coverage)

<a id="models"></a>

## 模型与技术报告

### Ling-3.1-flash-1019：后训练升级权重公开，1M 上下文声明与发布配置仍有差异
**inclusionAI；主窗新工件。** 仓库 10-27 05:46 UTC 创建，权重上传至 **10:46 UTC、`dddfee4b`**。模型卡给出 **560B 总参、25B 激活**，72 层由 63 KDA 与 9 Gated MLA 组成，前四层 dense、512 routed experts/top-8、一个共享专家；实见 54 个 safetensors 分片，配置为 FP8 block quantization，并列有不量化的 embedding、LM head 和门控投影。[固定模型卡与文件](https://huggingface.co/inclusionAI/Ling-3.1-flash-1019/tree/dddfee4bb43519a92761fbc5cefe3f0d3ff4aded)。

作者报告进一步后训练后 AA Intelligence Index v4.3.2 **41→44**、AutomationBench-AA **62%→68%**、CritPt **18%→26%**。这些是卡中列出的对比，未公开足以重建同预算对照的全部训练数据、token、算力和评测日志。**卡称 1M 上下文，但该 SHA 的 `max_position_embeddings=262144`、`rope_scaling=null`；不能据此保证直接加载配置即可运行 1M。** SGLang 示例也仍指向无 `-1019` 的旧仓名，应明确选择新 checkpoint 并验证实际服务配置。卡标 MIT；下一步优先查上下文扩展配方、MTP/FP8 排除路径和本地任务质量。

稳定 ID：`hf:inclusionAI/Ling-3.1-flash-1019@dddfee4bb43519a92761fbc5cefe3f0d3ff4aded`。

### LiquidAI LFM3：三类层混合的小模型家族，同时交付预训练基座
**官网 10-27 公告；主窗权重交付。** LFM3-1.7B 当前首个可见提交为 **10-27 05:58 UTC、`40ebe0aa`**，Base 为 **17:35 UTC、`545fc8f4`**，早于这些时间的仓库创建记录不能当成已交付权重。[官方公告](https://www.liquid.ai/blog/lfm3)、[后训练模型卡](https://huggingface.co/LiquidAI/LFM3-1.7B/tree/40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd)、[Base 卡](https://huggingface.co/LiquidAI/LFM3-1.7B-Base/tree/545fc8f4c7b5fa9f405105274cd29c4f8f44b526)。1.73B 的 24 层组合 **13 gated short convolution、5 sliding attention、6 global attention**；配置滑窗 256、上下文 262,144，BF16 基座可用于 CPT/后训练。家族同时覆盖视觉、音频、omni 与决策模型。

作者披露预训练 **22T tokens**，末 **3.5T** 为更高质量、含 agent traces/tool calls 的 mid-training 并扩上下文；后训练列出 **1.75T SFT tokens**、混合 RL、六领域专家 on-policy distillation 与 agentic RL，未交付全量原始语料或训练算力。作者同 vLLM/BF16 表里 SWE-bench Verified/OpenCode **43.4**，低于 Qwen3.5-9B 的 51.2；RULER128K **65.56** 也低于对照 87.11，不能概括为所有 9B 能力均被替代。端侧速度另用 batch1、32K、Q4_0、100 输出 tokens；LFM3 使用专用支持构建，对照 llama.cpp b11744，须计入版本差异。**LFM Open License v1.0 含商业使用年营收 1,000 万美元门槛**；不能写成无条件商用开源。建议复现同构建、同精度下的本地工具调用、长上下文质量与峰值内存。

稳定 ID：`hf:LiquidAI/LFM3-1.7B@40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd`；`hf:LiquidAI/LFM3-1.7B-Base@545fc8f4c7b5fa9f405105274cd29c4f8f44b526`。

### d1-2B / d1-Omni-2B：把分类、抽取和定位变成一次前向的带置信度决策
**LiquidAI；主窗工件，非 10-07 旧 d1-3B 的重新发布。** 两仓 10-24 创建，但当前唯一首提交分别是 **10-27 16:24 UTC、`97c57047` / `4cccbfd5`**。[d1-2B 固定卡](https://huggingface.co/LiquidAI/d1-2B/tree/97c57047d5852d77e02dfac211f8768aff77a8b2)、[Omni 固定卡](https://huggingface.co/LiquidAI/d1-Omni-2B/tree/4cccbfd5ba3405ee0f973134bd03e0d9d60ef1d8)。分别为 **1.92B / 2.17B**，沿用 LFM3 三类层 backbone，接受 state 与一组命名问题，直接返回概率、类别、数值/日期/文本 span 或 box；无自回归 token/speech 输出。Omni 增音频输入，不能把它当 ASR/聊天模型。

同题同评分的 Decision Index v0.3，作者给 d1-2B **51.1**、Omni **49.0**，旧 d1-3B **49.0**；图像任务最多各 1,000 行且改写成选项决策，不能等同完整生成式视觉基准。**5.7ms/H100 PCIe** 是 BF16、`compile()` 后 warm 调用、20 次中位数且 H100 取三次最佳；不含约一分钟编译与服务排队。默认一次读取 16 个问题，放大 set size 会损失文本准确率。卡明确文本 state 超约64K后可靠性下降，Omni 长语音和非拉丁文字性能更弱；未披露完整后训练语料/计算。沿用含营收门槛的 LFM Open License。建议先用自有分布校准阈值、缺失值拒答及数量区间，再测端到端批处理收益。

稳定 ID：`hf:LiquidAI/d1-2B@97c57047d5852d77e02dfac211f8768aff77a8b2`；`hf:LiquidAI/d1-Omni-2B@4cccbfd5ba3405ee0f973134bd03e0d9d60ef1d8`。

### NVIDIA 更新 Muse Spark DFlash2 草稿：确有权重变化，接受长度不能换算成同倍吞吐
**主窗 10-28 00:34 UTC，`22e2669e`。** 这是 10-16 已发布草稿的 checkpoint 与评测更新；对旧新两个固定 revision 的 `model.safetensors` 发 HEAD，尺寸均 8,768,333,792 字节，但 ETag 不同，支持确有工件内容变化，未下载张量核对。[固定模型卡](https://huggingface.co/nvidia/Muse-Spark-Open-816B-A42B-DFlash2/tree/22e2669e4c60689f9bff0749a34d9d8e59ce7cd6)。草稿 **4.38B、5 dense decoder layers、非因果滑窗2048**，不含与 target 共享的 embedding/head；每子层两 tap 卷积与 pairwise path selector 构成 DFlash2，必须配 Muse-Spark-Open target 使用。

一台 GB300 node、TP4+EP4、并发32、xhigh、4096 输出、draft7 的作者评测中，SPEED-Bench qualitative 平均 AL **4.08/3.68**（greedy / temperature1无top-p/k截断）；32K throughput 的三类各前100 prompt则 **3.59/3.26**。这些数据也用于开发选 checkpoint，不能当完全独立留出集；AL 不是实测端到端加速。训练披露约348万多语种合成样本、来自 Nemotron 数据集 prompts 和 target 输出，但 token 范围非常宽，精确训练量/算力不足。模型卡列 NVIDIA Open Model Agreement，并额外要求 Meta Muse terms；NVIDIA许可正文已读，Meta许可正文直取403，条款仍待核。它不是独立多模态 target 权重。建议复现目标版本、量化精度、采样方式及 TPOT/TTFT，而非只看 AL。

稳定 ID：`hf:nvidia/Muse-Spark-Open-816B-A42B-DFlash2@22e2669e4c60689f9bff0749a34d9d8e59ce7cd6`。

### Genie 4：原生声音与可交互角色进入世界模拟预览，但没有公开训练配方
**Google DeepMind；主窗10-27，页面元数据16:13 UTC。** [官方正文](https://deepmind.google/blog/genie-4-a-new-generation-of-world-simulation/)。Google称新世界模型联用Gemini Omni的外观/行为表示和其他Gemini模型的推理，在交互视频世界中同时生成环境声音、音乐和人物对话；用户可用键盘导航和文字改天气、角色、物体，Gemini也持续监控并引导场景。这是**trusted testers研究预览**；Project Genie/AI Ultra与Gemini API/AI Studio早期合作方属于接下来开放范围，不能写所有开发者已经可用。

正文主要给精选实时交互录屏，未公开参数、具体架构、训练数据/计算、权重或可复现物理/持续性基准，也未给本轮可确认的FPS、时延与分辨率表。官方明确长会话、复杂编辑与移动中可能发生物体恒常性/细节漂移，物理、音频、文字和空间精度仍有限。建议观察面向agent/robot训练的可控状态与长期一致性评测，不能把视觉逼真直接等同物理仿真准确。

稳定 ID：`google:genie4:research-preview:2026-10-27`。

<a id="pretraining"></a>

## 预训练、持续训练与架构

### S3R-EC：共享随机舍入阈值后，必须修正会向低层传播的误差
**2026-10-27 v1；主窗新作；`arxiv:2610.36587`；机构/团队：Google。** [论文全文](https://arxiv.org/html/2610.36587v1)。逐元素 stochastic rounding 的随机数生成可能耗尽 TPU vector unit，抵消 INT8 GEMM 收益。S3R 共享阈值，S3R-EC 再从 dX/dW 中扣除已知舍入误差的行列均值分量，避免相关误差沿激活梯度累积（§III）。

TPU v5p、JAX/XLA 上，12 层 decoder 的 32.8M C4-token、三种子验证中，校正后的 loss 为 4.919±0.022，独立 SR 为 4.929±0.019，BF16 为 4.892±0.015（Table I）。每设备batch4、序长4096，12步warmup后20步中位数的整步计时中，Llama3.1-8B/Qwen3-8B/Gemma3-12B三个 dense 模型比 BF16 快 1.24–1.32×；但两种 MoE 仍比 BF16 慢，且这些大模型只有合成输入短程计时，不能继承小模型的收敛证据（Table IV）。**实践：分别验收随机数成本、dX 误差与完整训练步；不能把 FFN 算子加速当作 MoE 训练加速。**

### Edge of Stability：语言模型预训练的稳定边缘依赖 batch 和学习率阶段
**2026-10-27 v1；主窗新作；`arxiv:2610.36778`；机构/团队：UC Berkeley／清华大学／ByteDance Seed。** [论文全文](https://arxiv.org/html/2610.36778v1)。论文用补偿后的局部随机二次动力学估计临界学习率，区分负曲率方向的逃逸与正曲率导致的数值爆炸，而非直接把 Hessian 最大特征值套入全批 GD 判据。

43M–787M GPT、100 余次预训练、3,000 余 checkpoint 中，EoS/EoC 经常出现；但小 batch 或充分降 LR 后，EoS 可以消失。GPT-extra-small 的 FineWeb 扫描中 B=512/1024 比 B=256 更常靠近稳定边缘；过大 peak LR 还会跨过边缘并恶化 loss（§6）。这是有限时域局部诊断，非证明所有 LLM 必在 EoS，也未证明贴着临界点最优。**实践：调 batch/LR 时同时看局部稳定性与最终 loss，避免把 EoS 当通用调参目标。**

### Select-TTT：已知查询后，按相关性控制 fast-weight 写入强度
**2026-10-27 v1；主窗新作；`arxiv:2610.36794`；机构/团队：中国科学技术大学／NUS／华为诺亚方舟实验室。** [论文全文](https://arxiv.org/html/2610.36794v1)。沿用 In-Place TTT 的 MLP down-projection fast weights，用冻结 LM head 的 query-token 分数为 chunk 更新乘上不同学习率，抑制无关内容冲淡相关记忆（§4）。

相同 TTT checkpoint 下，LongMemEval-S 的 Llama-3.1-8B 平均为 baseline 33.4、均匀 IP-TTT 33.6、Select-TTT 37.4；Qwen3-4B 为 27.2/25.4/28.8（Table 1）。正文一处把 Llama 的其他均值误写进 LongMemEval-S 叙述，本条按表格报告。仅两种至 8B backbone、两个英文 benchmark；query 已知和启发式 relevance 是关键条件，不能推广为在线未知未来查询的通用记忆。**实践：先做同 checkpoint、仅更换写入调度的对照。**

### TUON：跨层变换后逐切片极化，增益需与 AdamW 和墙钟一起看
**2026-10-26 v1；72小时回溯补录；`arxiv:2610.35130`；机构/团队：北京大学／琶洲实验室（黄埔）。** [论文全文](https://arxiv.org/html/2610.35130v1)。将同形状层的momentum先沿层轴混合，再逐slice作polarization并逆变换，避免把多层直接展开成一张失衡大矩阵（§2）。

GPT124M/350M、Qwen0.6B/4B上，TUON验证loss均低于Muon/TEON；但124M仍高于AdamW（2.9357对2.9230），4B对TEON仅降0.0006（Table 2）。每步较AdamW慢3.5–12.5%，对Muon慢1.7–6.1%（Table 4）；单seed42，附录对context/PP配置还存在前后不一致，不能将小loss差写成确定的算力节省。**实践：先做等墙钟、多种子及明确并行配置对照，再决定替换优化器。**

### Error Diffusion：低比特残差可代替二值训练的潜在权重，但尚是小模型证据
**2026-10-25 v1；72小时回溯补录；`arxiv:2610.34566`；机构/团队：Pierre-Olivier Bonin（独立研究者）。** [论文全文](https://arxiv.org/html/2610.34566v1)。每个二值weight累积未实现更新，跨阈值时翻转并保留量化残差；随机舍入网格必须在翻转阈值上有精确刻度（§3、§5.1）。

RTX4080SUPER上的小Transformer，8-bit残差加1-bit权重按核心每权重9bit计，50K步Shakespeare loss1.583对latent基线1.556；加update cap后六个scaling点loss差约−2.3%至+2.9%（Tables 2–3）。最大binary core19M，含token embedding约70M；未计入FP embedding/norm/head及Adam和activations，4-bit残差随规模/时域明显恶化；275MB CPU预算还排除约750MB Python/PyTorch基线。**实践：这是持续训练状态压缩研究，不是十亿参数模型整体训练显存缩减19倍或已证明加速。**

### INTRA：静态Scatter/Gather传递长距信息，但检索质量不等于通用长上下文质量
**v1 2026-10-25；本轮按v2 2026-10-27核查；72小时回溯补录；`arxiv:2610.34056`；机构/团队：New York University。** [v2全文](https://arxiv.org/html/2610.34056v2)。v2只新增代码链接，实验正文/表格未发现变化。静态互补分组让信息跨层经relay传递，并由CQS-aware kernel打包非连续token（§3–5）。

A100、Llama-3.1-8B、相同LoRA预算后，teacher-forced NIAH平均0.923，dense1.000；相对MoBA/NSA的最高8.4×/10.4×仅为attention计时，非整模prefill。SCROLLS六任务中五项低于两个基线，均分36.01对36.88/36.44；图像2K FLUX端到端才是1.53×（§6、Table3–4）。无质量CI，稀疏pattern与计时配置也有区别。**实践：按实际生成任务、adaptation成本与端到端延迟验收，避免从needle retrieval推广通用质量。**

<a id="posttraining"></a>

## 后训练、训练与评估数据

### CIGA：数据选择先抑制提示表面形式对梯度的干扰
**2026-10-27 v1；主窗新作；`arxiv:2610.36755`；机构/团队：中科院计算所／中国科学院大学。** [论文全文](https://arxiv.org/html/2610.36755v1)。同一验证题的语义不变改写会改变梯度归因。CIGA 先选更新强且跨改写稳定的层，再将该层梯度相似度和语义相似度做自适应 reciprocal-rank fusion（§3）。

Llama-2-7B/Qwen3-4B 的 Top-5% 指令数据选择中，跨五种改写的平均 Jaccard 达 58.33%/48.68%，最强全局梯度基线为 49.06%/40.03%；三次重训的五任务平均比随机子集高 2.11/1.04 点（Tables 1–2）。需要目标任务高质量验证集，仍要遍历候选池做前后向并存储梯度，未解决千万级数据归因成本；也非每个任务都优于全量或其他方法。**实践：评估数据选择器时加入语义不变改写和等预算重训，不能只看一次排名。**

### MPOPD：把教师支持用于学习 rollout proposal，而非只改蒸馏目标
**2026-10-27 v1；主窗新作；`arxiv:2610.36477`；机构/团队：UIUC／Stanford。** [论文全文](https://arxiv.org/html/2610.36477v1)。方法按教师相对学生的 log-prob 支持倾斜采样分布，训练一份独立 proposal policy 近似该分布，再用截断重要性权重更新学生（§3）。它改变会访问哪些前缀，区别于对已经采到的轨迹重加权。

GH200、Qwen3-4B-Base 的八个数学 benchmark 平均准确率为 35.37%，标准 OPD 28.46%、最强 VETO 31.06%；平均输出长度 9.63K，VETO 9.79K（Table 1）。proposal 增加训练计算：正文报告平均墙钟多约 13%，峰值显存 141.99GB 对约 124GB（§4.3）。主证据集中同族 Qwen3，跨族、动态倾斜强度和轻量 proposal 尚待验证。**实践：比较 OPD 配方时固定 rollout 分布，并计入额外 proposal 的训练与内存。**

### PICO：全组失败时，用教师与学生共同支持的轨迹恢复学习信号
**2026-10-27 v1；主窗新作；`arxiv:2610.36849`；机构/团队：中国科学技术大学／腾讯／上海人工智能实验室。** [论文全文](https://arxiv.org/html/2610.36849v1)。教师只在训练时读专家解，joint decoder 限制在教师与学生 top-k 的交集中，再用 mixed-behavior policy optimization；不是把专家答案直接放入学生上下文（§3）。

四张 H20、DAPO-Math-17K、每题七条 on-policy 加一条引导轨迹，4B/8B 的六 benchmark 平均分别比 GRPO 高 4.1/3.6 点；4B agent 的 ALFWorld/ScienceWorld 汇总为 74.3%，GRPO 67.5%（Tables 1–2）。训练单步由 28.3 增至 43.6 分钟，rollout 利用率由 57.2% 增至 91.9%（Appendix C），因此并非免费探索。**实践：遇到 solve-none cliff 可比较这种兼容引导与直接离线示范，但应按总墙钟及教师计算比较。**

### SetGRPO：校准的是 grader 引起的更新系数变化，不是答案真值
**2026-10-27 v1；主窗新作；`arxiv:2610.36807`；机构/团队：SetGRPO作者团队（机构未核）。** [论文全文](https://arxiv.org/html/2610.36807v1)。同一响应组两次评分经组归一化后会产生相关系数偏移；方法校准联合椭球集，在投影的局部更新几何中恢复仍与第二次评分方向一致的系数，并筛掉不满足门槛的组（§4）。

Qwen3.5-4B 的十 benchmark 宏平均为 60.26，RubricHub 58.65、普通 rubric-GRPO 55.41；Qwen3.5-9B 为 63.51 对 62.11（Table 1）。代价是单步 82→103 秒、累计 GPU 时 22.8→28.6（Table 6）。覆盖保证依赖校准组与目标组可交换；第二评分也非真值，“full geometry”仍是固定投影空间，策略漂移只有有限 checkpoint 的经验检查。**实践：将它视为 scorer 噪声下的更新过滤配方，不能写成真实奖励或训练安全的无条件保证。**

### CoHARL：策略权重与可执行 harness 在同一训练循环中演化
**2026-10-27 v1；主窗新作；`arxiv:2610.36469`；机构/团队：UNC Chapel Hill／RIKEN／Cisco。** [论文全文](https://arxiv.org/html/2610.36469v1)。一份共享策略在 harness population 中收集经验；更强 proposer 根据 trace 修改工具执行代码，新版本到下一 step 才准入，组内优势始终在相同任务和 harness 内计算（§3）。

Qwen3.5-35B-A3B rank-32 LoRA、150 步 SWE 训练中，110 题 SWE-rebench 的 avg@3 为 27.6%，harness-only 21.2%、policy-only 20.3%；跨数据集 SWE-bench Verified 为 59.8% 对 52.0%/49.8%（Tables 1–2）。单训练运行、无 CI、额外 proposer/验证计算没有完整对齐；Terminal-Bench 结果来自另训 agent。**实践：值得把 harness 纳入训练变量，但需单独测总算力、同 harness 下的策略收益及验证集选择过拟合。**

### HiLo-RD：把大预算的表示蒸馏到小预算，收益不等于同比例省时
**2026-10-27 v1；主窗新作；`arxiv:2610.36901`；机构/团队：HKUST（广州）／浙江大学／NTU／阿里巴巴／HKUST。** [论文全文](https://arxiv.org/html/2610.36901v1)。论文诊断 masked diffusion 在低预算下过早锁定尾部答案，再用同模型高预算轨迹在共有位置的 hidden-state MSE 训练低预算 LoRA（§3–4）。

八张 A100-80GB 训练的 LLaDA-8B，同低预算 pure-diffusion/fast-rollout 设置中，GSM8K 为 78.24%，vanilla 68.84%、Diffu-GRPO 72.93%；readable tokens 362.8→271.3，但单请求时间仅 73.35→71.71秒。MBPP 则为 38.60→41.40%、19.51→11.20秒；HumanEval 49.39 仍低于 dOPSD 50.00（Table 1）。**实践：区分可读长度、去噪预算和真实时间；结果局限一个 backbone 和数学训练向代码的迁移，不能称普遍自我进化。**

### S3AC：训练好 critic 后可做单rollout RL，但每更新仍更贵
**2026-10-26 v1；72小时回溯补录；`arxiv:2610.35518`；机构/团队：上海人工智能实验室／独立研究者。** [论文全文](https://arxiv.org/html/2610.35518v1)。混合不同强度策略轨迹覆盖长且正确的回答，用小value-head初始化和更宽critic clipping训练价值模型；critic既给token级优势，也用于生成前的难度采样（§3）。

35B组rollout最佳checkpoint七benchmark平均80.5，GRPO-X76.4；单rollout80.9，对REINFORCE++70.4。397B组rollout85.1→87.3（Tables1–3）。完整配方而非孤立critic组件比较；**每actor梯度步摊销墙钟多27–40%**，未系统多seed，任务集中可验证数学/科学/多模态推理，未测多轮agent。**实践：在rollout浪费高的RLVR里评估critic和DAPS，按达到目标质量的总墙钟验收。**

### MXFP8 RL：同量化网格改善一致性，host overhead 决定何时更快
**2026-10-26 v1；72小时回溯补录；`arxiv:2610.34915`；机构/团队：Wengee Zhang作者团队（机构未核）。** [论文全文](https://arxiv.org/html/2610.34915v1)。B200上把verl learner和vLLM/SGLang sampler的量化网格对齐，并检查weight sync的层集合、grid、派生layout、graph storage、完整性、输出和可观测性七条条件（§4、§9）。

同engine固定权重的dense对照，完整匹配sampler比BF16 sampler的train-rollout KL低16.1%（95%prompt-bootstrap14.1–18.1%）；BF16端到端仍常更一致。四张B200、Qwen3-30B-A3B十步中step506.0→451.8秒、generation−23.8%，但actor峰值60→75GiB。短序列dense反慢5.1%；约4.5K tokens/microbatch的盈亏点只适用于该模型/host（§6–7）。全部单seed、10–60步，不排序最终质量。**实践：同步验收byte/layout与输出，并测host dispatch和microbatch，而非只打开“FP8”。**

### InterQ-RL：多轮量化RL还要纠正抵达当前状态的历史
**2026-10-26 v1；72小时回溯补录；`arxiv:2610.35303`；机构/团队：华南理工大学／腾讯。** [论文全文](https://arxiv.org/html/2610.35303v1)。W/A及KV对齐后，当前token的比值仍看不到历史策略造成的state偏移。方法按执行前缀做有界校正，并按effective support分配segment校正预算，不增加model forward（§3）。

H20、dense8卡/MoE16卡、400步、共享W8A8-KV8 rollout，ALFWorld成功96.4%对QAT90.4%和BF1697.1%；ScienceWorld30B-A3B宏分69.9，对QAT69.3、BF1667.0（Table1）。部分多seed只覆盖ALFWorld/WebShop的三条主要arm；AIS/QuRL是作者重实现，MoE的UAQ有适配，不能把差异全归结于通用算法优劣。**实践：分别看endpoint、学习曲线面积、成功轨迹turn和cap耗尽，低K3本身不代表高质量。**

### FOCUS：把额外rollout花在更值得校准的中间状态
**2026-10-26 v1；72小时回溯补录；`arxiv:2610.35801`；机构/团队：Johns Hopkins University。** [论文全文](https://arxiv.org/html/2610.35801v1)。通过可恢复环境状态的branching收集中间continuation，用critic先验与递归backup校准value，再按预期误差下降选择新增rollout起点（§2）。

Qwen3.5-9B、等每步generation预算且均有compaction，BrowseComp-Plus为55.4%，GRPO48.2%、VinePPO46.6%；DeonticBench总体56.5%对52.4%，但hard子集26.8%对28.2%并未占优（Table1）。BrowseComp报告的是训练中最佳held-out checkpoint；paired CI不等于独立训练复现。critic另需1张H100，主训练2张H200加4–6张H100 rollout，并非等总硬件。**实践：适合研究长任务credit与compaction交互，需补总算力、checkpoint选择及多seed验收。**

### Fork：rubric覆盖奖励增加“必要且正确”的响应单元约束
**2026-10-24 v1；72小时回溯补录；`arxiv:2610.33512`；机构/团队：KAIST AI／Carnegie Mellon University LTI。** [论文全文](https://arxiv.org/html/2610.33512v1)。把rubric recall与response-unit precision合成F-beta奖励，减少靠冗长、堆事实满足checklist的投机（§3）。

Qwen2.5-7B-Instruct、8H100训练，医疗域Table1的beta=1结果precision53.2、recall32.5、macroF1 37.5，对OnlineRubrics为26.8/38.7/28.8；GPT-5.4 rubric-free win68.0%对10.6%。正文把F1写成38.6，和该表不一致，本条按37.5报告。长度明显缩短但recall存在下降，外部benchmark及win rate仍依赖LLM grader和judge prompt（Table4换prompt后数值显著变化）。**实践：将coverage、precision、长度和独立评价分开，不能仅凭训练rubric分数确认质量。**

### DTCA：扩散RL按去噪贡献分配token credit，当前证据限数学
**2026-10-26 v1；72小时回溯补录；`arxiv:2610.35201`；机构/团队：浙江大学／NUS／SAP。** [论文全文](https://arxiv.org/html/2610.35201v1)。估计某token提交后对剩余masked位置置信度的改善，按POS聚合，再选去噪状态并用entropy变化校准credit（§4）。

LLaDA-8B-Instruct、MATH训练、256-token rollout，128预算MATH50039.8%对base30.8%和RLDF36.2%；512预算44.0%对39.2%/42.6%，GSM8K85.4%对82.9%/82.7%（Table1）。它不是仅比未训练base：对RLDF仍有增益，但各arm保留方法特有设置。POS表来自MATH/GSM8K训练集，未证明对代码等token分布泛化；贡献是启发式置信变化，非因果credit真值。**实践：先在同rollout、同过滤阈值下拆分状态选择和credit校准。**

### ScaleRLT：按rollout原调度重放logprob，可严格校验但尚未换来更好质量
**2026-10-25 v1；72小时回溯补录；`arxiv:2610.34427`；机构/团队：University of Tübingen／西安交通大学／UIUC。** [论文全文](https://arxiv.org/html/2610.34427v1)。Ouro循环模型的deferred rollout与dense learner浮点路径不同；hybrid logprob用原schedule无梯度重放的值、dense路径的梯度，让同权重重要性比率精确为1（§4）。

Ouro-1.4B、两节点八GH200的one-wave/CUDA-graph/释放KV组合使更新195.6→72.2秒，bitwise gate通过；四节点弱缩放近似保持，强缩放受decode/gate最长响应限制。关键负面：mixed-depth读出虽每次更新快4–13%，100更新准确率82.4%，native90.0%，后者20更新已84.6%；单卡BF16有/无gate为77.2/77.0（§7）。**实践：重放一致性是可调试契约，不能据此宣称更高RL质量；精确比值还依赖每rollout batch只做一次optimizer step。**

### SynthDocQA：微软公开 100 份合成商务 PDF 的 9,798 条元素级 QA
**主窗 10-27 17:04 UTC 复制到微软官方组织，17:26 UTC 更新许可，`fe6bc0bb`；不等于此前个人仓首次创作日。** [固定数据卡与文件](https://huggingface.co/datasets/microsoft/SynthDocQA/tree/fe6bc0bb92ad6ed7643ff59ce5e5eb1d3690564e)。本轮实际下载 `ALL_queries.json`，复核 **9,798 唯一 ID、100 个 PDF 引用、无空 refs/assertions**；元素为 table3961、figure2899、form1496、annotation941、text block501。每题带 PDF、元素 ID、生成 recipe 与预期答案断言，适合分解 PDF parser/RAG 的元素定位与回答错误。

**实际 assertions 是 `{level,text}` 对象数组，卡里的示例却写字符串数组**，使用时应以数据为准。PDF 与 manifest 文件可见，未逐页人工验真；生成模型、原始来源覆盖和系统性人工标注质量未披露，合成商务内容也不能代表真实生产文档分布。数据卡全文列 CDLA-Permissive-2.0。建议按文档而非 QA 行划分开发/测试，检查断言执行器、表格/图形失败与同模板泄漏。

稳定 ID：`hf-dataset:microsoft/SynthDocQA@fe6bc0bb92ad6ed7643ff59ce5e5eb1d3690564e`。

### Open-SWE-Traces v2.0：增加 Qwen3.8-Flash-Next 的五种 harness 轨迹
**NVIDIA；主窗新数据上传 10-27 19:xx UTC，分布表20:27更新，`9b073ef0`。** 卡内新闻标10-26，按实际工件交付与卡内公告分开记录。[固定卡](https://huggingface.co/datasets/nvidia/Open-SWE-Traces/tree/9b073ef0468b948a0c5c0e94d03fc56efbec1db9)、[版本分布表](https://huggingface.co/datasets/nvidia/Open-SWE-Traces/blob/9b073ef0468b948a0c5c0e94d03fc56efbec1db9/data_distribution.md)。新增 Claude Code、OpenCode、OpenHands、SWE-agent、mini-SWE-agent，覆盖代码任务的完整 messages/tools/patch/metadata。文件清单实见 **212 个 `qwen38_flash_next` parquet**；作者 v2表列 **168,824 trajectories**，每 harness 26,027–36,598 条，非跨列去重后的唯一任务总数。

卡称有 xhigh/medium/low reasoning effort，但目前分布表主要列 xhigh；不能把表内168,824自动扩成三档各同规模。轨迹含 resolved=0 与 unknown=-1，**不应默认全是成功示范**；本轮未全量计数、去污染或训练复现。来源任务为 Scale-SWE 与 SWE-rebench-V2；数据声明 CC-BY-4.0，代码仓和上游数据条款仍需保留。建议核对 `instance_id` 交集、harness与effort配比，并在同一目标模型上做成功轨迹筛选消融。

稳定 ID：`hf-dataset:nvidia/Open-SWE-Traces:v2.0@9b073ef0468b948a0c5c0e94d03fc56efbec1db9`。

### ChartNet 扩展：多面板与11种语言图表，许可只覆盖明确的新子集
**IBM；回溯补录10-26数据发布，10-27卡说明更新，`ee4f3ce8`。** [固定数据卡](https://huggingface.co/datasets/ibm-granite/ChartNet/blob/ee4f3ce8fbfa83a6a995dbbba18c493a2556491d/README.md)。作者列新 `multichart` **413K** 图表，字段 image/CSV/summary；`core_permissive_multilingual` 约 **27M**，从原英语 chart翻译到阿拉伯语、中文、日语等11语，保留源 ID 与语言后缀、code/CSV/summary及chart类型。文件清单实见新子集与11语 parquet 路径，规模数字仍来自卡，未全量行计数或核验翻译质量。

**27M 不是27M独立原始图形**；同图跨语种划分会造成内容泄漏，应按去掉语言后缀的源 ID 分组。CDLA-Permissive-2.0 覆盖卡中明确的新扩展和 `core_permissive`；原始 `core/reasoning/grounded_qa/human_verified` 仍有仅供复核评估的 Notice 及商业限制，不能因仓顶标签就把所有历史子集统一标成可商用。未新增可归因于这批扩展的受控模型收益。适合复现多图表抽取、跨语种一致性与数据切分。

稳定 ID：`hf-dataset:ibm-granite/ChartNet:october-extensions@ee4f3ce8fbfa83a6a995dbbba18c493a2556491d`。

<a id="inference"></a>

## 推理优化与部署研究

### DCKV：大幅压缩的是存储中的 prompt cache，解码前仍恢复 BF16
**2026-10-27 v1；主窗新作；`arxiv:2610.36950`；机构/团队：MIL Team／DAIMLD。** [论文全文](https://arxiv.org/html/2610.36950v1)。codec 联合训练跨层 K/V 变换、量化器和条件熵模型；Qwen3.5-2B 的 recurrent state 保持原样，首 4/末 128 个 cache 位置不压缩（§3）。

单请求 A100-40GB、greedy、4K–32K 上，较高保真率点需 205–221 bytes/压缩 token，而 BF16 为 12,288；RULER4K/LongBench/RULER32K 分别损失 0.56/1.75/0.45 点（Table 1）。**56–60×只是 payload**，还要加固定状态和 176MiB codec 权重；完整 BF16 cache 在解码前恢复，因此不降 decode 峰值或每 token 时间，峰值反增约 182MiB。KVTC 是作者重实现，跨设备熵解码的 bitwise 一致性未验证。**实践：适合研究冷 cache 的存储/传输成本，不能据此采购更多并发 decode 容量。**

### 将 AM bias 折入 K：让压缩 cache 使用现有 attention kernel
**2026-10-27 v1；主窗新作；`arxiv:2610.36520`；机构/团队：Utkarsh Kanwat（独立研究者）。** [论文全文](https://arxiv.org/html/2610.36520v1)。Attention Matching 的 retained-key bias 缺少标准 kernel 接口。论文为每头拟合一条共享方向，把 bias 编进 K，保持 head dimension 和 query 路径不变；误差取决于新 query 在该方向上的投影是否仍接近 1（§3）。

equal-head-budget 下，vLLM 对 Qwen3-4B 的 51 题答案与 HF 一致，对 Qwen3-14B 的 200 个 LongBench 样本有 180 个 greedy 输出一致（§6）。容量/吞吐收益主要来自上游 cache compaction；14B/32B 扩展计时用随机权重和随机 cache，**不验证答案质量**。远离原上下文的 query、ragged per-head budget 和重复在线 compaction 仍有缺口，部分模型/设置会损失 AM 优势（§9）。**实践：先验收真实权重、相同 cache 的质量和服务一致性，再看容量曲线。**

### TDLM：冻结扩散 backbone，让未解 token 读取自身历史表示
**2026-10-27 v1；主窗新作；`arxiv:2610.36940`；机构/团队：Cambridge／Oxford。** [论文全文](https://arxiv.org/html/2610.36940v1)。temporal attention 保存并读取前几次去噪的同位置 hidden states，只训练新增模块；每 backbone 使用 3,500 条自生成轨迹和 750 次更新（§2–3）。

GH200、batch1、256-token预算、编译后计时，LLaDA GSM8K 为 79.08%、0.585秒，Fast-dLLM 为 78.85%、1.062秒；Dream MBPP 为 57.98%、0.467秒，原模型 58.37%、3.334秒（Table 1）。它并非所有质量/延迟点最优：例如 LLaDA MATH-500 比原模型低 1.2 点。历史缓存有额外内存，且有待补齐同容量 current-only adapter 与长输出、重复训练统计。**实践：将历史复用与阈值、NFE、实测延迟、质量一起比较，不只报减少 forward 次数。**

### DeepInfer-Mac：24GB 笔记本可运行大型稀疏模型，但吞吐与 swap 边界必须保留
**2026-10-27 v1；主窗新作；`arxiv:2610.36564`；机构/团队：东南大学。** [论文全文](https://arxiv.org/html/2610.36564v1)。在 M4 Air 24GB/1TB SSD 上按需读取 DeepSeek-V4.1-Flash 的专家和 Engram，使用混合2/3/4-bit转换；磁盘 checkpoint 为337.6GiB，常驻权重约7.6GiB（§2）。

短英/中/代码示例为0.21–0.72 token/s；3,764-token检索首次 token 等待309秒，采样到的最大进程 footprint 为16.73GiB（Table 2）。共享桌面原已有swap，运行中swap增长，**不能称无swap**；模型版本不一致的API对照不能分离量化损失，78项测试也不证明原模型能力保留（§5.3、§8）。**实践：这证明“能执行”及其延迟成本，不证明可交互吞吐或低比特质量等价。**

### dSpark：蒸馏提交日程，提高parallel decoding并保留质量边界
**2026-10-26 v1；72小时回溯补录；`arxiv:2610.35804`；机构/团队：MBZUAI VILA Lab。** [论文全文](https://arxiv.org/html/2610.35804v1)。从不确定性引导的去噪日程构建可提前提交的目标，用schedule distillation与KL约束训练LLaDA/Dream（§4）。

A100-SXM4、batch1、LLaDA GSM8K，dSpark78.94%/249token/s，base78.82%/15，d3LLM74.21%/220；因此16.6×对的是逐token去噪base，对强加速baseline约13.2%。自有static CUDA-graph cache提高到313token/s/79.23%（AppendixA.1）。代码任务并非完全无损：LLaDA MBPP40.0对base41.7；Fast-dLLM cache在部分512-token设置反降质量且不更快。**实践：用同prompt长度、cache和generation预算比较真实TPS，不把TPF当普适速度。**

### RIPE：循环模型旧prefix可只存最深plane，近期窗口的速度尚未实测
**2026-10-25 v1；72小时回溯补录；`arxiv:2610.34436`；机构/团队：University of Tübingen／UIUC／西安交通大学。** [论文全文](https://arxiv.org/html/2610.34436v1)。旧位置仅留最深loop的KV，近W位置保持全depth；W不少于draft block时，exact arithmetic下batched verify与逐token读相同planes（§4）。

GH200/BF16、Huginn、1024-context时W64占full cache11.9%，48个prompt的teacher-forced replay把KL>0.1的位置9→1。**W>0没放进真实speculative decoder**；2.58×来自deepest-only eager decoder，固定显存吞吐来自WaveFront emulator，不能称RIPE窗口端到端加速（§5、§7）。尾部统计只由四prompt承载，Raven/Huginn表现也不同。**实践：先做模型特定的误差尾部与真实decode验证，再将存储模型换算并发。**

### MoE推测解码的expert去重：省字节不保证省时间
**2026-10-25 v1；72小时回溯补录；`arxiv:2610.34440`；机构/团队：ReClan AI Studio, LLC。** [论文全文](https://arxiv.org/html/2610.34440v1)。将expert请求字节除以各kernel实效bandwidth建立step-time模型；只有去重后字节占比低于其相对带宽，deduplicating kernel才更快（§2–3）。

llama.cpp CUDA/Metal与MLX-LM，7B–30B、batch1中，去重kernel带宽仅为per-slot的0.32–0.75，通常到5–8 tokens/pass才获益或始终不获益。12配置有10个crossover预测误差不超过一个pass size（Table3）；Metal有明显失配。理论假设几何acceptance且忽略draft模型成本，多数拟合/预测发生在已有整模结果之后（§5）。**实践：调draft长度时实测具体expert shape的kernel带宽，不依据专家重用率单独判断。**

### T2D：联合裁剪与对齐训练，让draft结构服从延迟目标
**2026-10-25 v1；72小时回溯补录；`arxiv:2610.34103`；机构/团队：University of South Florida／Arizona State University／UC Davis。** [论文全文](https://arxiv.org/html/2610.34103v1)。以可学习gate裁剪target的层、attention heads及FFN channels，同时用CE与latency predictor优化，裁剪后继续对齐训练（§4）。

B200服务器上，Llama-2-7B整体推测加速1.74×，Wide1.59×；Vicuna13B1.89×对1.78×；Qwen3-14B只有1.20×，roleplay仍约1.00×（Table1）。0.86B T2D draft比1.27B Sheared-Llama-ft的1.18×更快，但该对照模型规模/预训练预算不同（Table3）。Wide架构也由Llama-2调出再迁到各target，非每模型穷尽搜索。**实践：在目标硬件与工作负载重新标定latency predictor，并把强draft/结构搜索预算纳入对照。**

### 受约束解码：应固定prompt和输出空间，再归因masking的影响
**2026-10-26 v1；72小时回溯补录；`arxiv:2610.35328`；机构/团队：Télécom Paris／Institut Polytechnique de Paris／Nokia Bell Labs。** [论文全文](https://arxiv.org/html/2610.35328v1)。区分format、validity和reasoning约束；在同自由推理trace之后只改变答案解码，帮助识别“结构化输出降低推理”中的prompt/格式混杂（§3–5）。

LastLetter的Qwen3.5-9B同NL-to-format JSON流程，unconstrained89.6%变为FC+VC91.6%（Table2）。但validity实证主要是字母任务；强制特定推理开头在Qwen3.5-4B AIME2026反降6.6点，复杂schema加复杂推理仍待测（§5.4、AppendixA.1）。**实践：把free reasoning、答案提取、格式可解析率和答案合法性分开做因果对照，不能理解为任意grammar都不伤质量。**

### Target Regeneration：draft目标函数的大增益，可能被目标模型重生成抵消
**2026-10-21 v1；旧候选补录，超出72小时回溯；`arxiv:2610.29013`；机构/团队：Korea University／Yonsei University Mirae Campus。** [论文全文](https://arxiv.org/html/2610.29013v1)。同prompt/图像分别用外部回答、target重生成及matched-length版本训练CE/KD/生存gate，检查目标函数收益是否只是修补corpus–target错配（§2）。

Qwen3-VL-8B的ALLaVA实验，外部回答KD比CE的MAT高14.4%，重生成后仅3.7%；gate从+4.9%变−0.5%。text DFlash/EAGLE-3复核也呈收缩（Tables1–2）。最长26,196样本子集、每cell约两次训练；matched-length仍不完全隔离截断/anchor/padding，MAT不等于实际墙钟或答案质量。**实践：新draft loss必须在生产实际的target-regenerated训练数据上对CE报告增益。**

<a id="core"></a>

## 核心开源工程

UTC主窗：2026-10-27T01:00:54Z→2026-10-28T01:02:09Z；72h回溯起点：2026-10-24T01:00:54Z。六仓官方all-base merged PR共623项（主窗222），默认main commits共598项、公开Release共254项均完成实际分页，六仓本窗及回溯均无新公开Release。Megatron 0.20文档、TorchTitan release/0.4分支版本变动不等于正式Release。

以下61组、88个新稳定PR为选择性机制审阅：已打开PR状态/正文与关键最终补丁，默认合入目标均为main。未做本地GPU/NPU模型复现；测试、数值、性能均为作者按所述版本/硬件报告，旧revision与最终merge验证差异逐项保留。库存枚举、归档patch与完整语义审计分开，旧backlog不因回溯清空。仅SGLang三项分支观察另列，绝不混作main或Release。

### Megatron activation offload 可用固定 pinned buffer 封顶，图捕获地址保留独立计费

新增 fine_grained_offloading_buffer_size_gib，默认0关闭；首次 forward 直接 cudaHostAlloc 一块固定存储，按256字节对齐跨形状复用，避免可变 MoE activation 的 allocator 缓存增长与2幂取整。H2D 最后读者的 CUDA event 保护后续写入，容量耗尽明确报错、不会自动增长；需要cuda-python，并须在capture前完成buffer初始化。图捕获各自保留地址范围，reset/clear 不回收图保留区；混合 eager/graph 要同时预算，PP/VPP/通信重叠还可能需要容纳整轮多个 microbatch。152GiB→无需256GiB是分配取整示例，不是训练实测节省。建议监控 live/peak/graph_reserved_bytes，并在重捕获和异步 reload 下验证生命周期；它仅约束此 buffer，不约束整机 host 内存。

机构/社区：NVIDIA。证据基线：[NVIDIA/Megatron-LM #7961](https://github.com/NVIDIA/Megatron-LM/pull/7961)，2026-10-28T00:54:48Z，main @ 57018e45e456。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`megatron/core/pipeline_parallel/pinned_activation_buffer.py::PinnedActivationBuffer`；`megatron/core/pipeline_parallel/fine_grained_activation_offload.py::OffloadTensorPool`；`docs/user-guide/features/fine_grained_activation_offloading.md::Fixed Pinned Buffer`。

稳定ID：`github:NVIDIA/Megatron-LM:pr:7961`。

### Megatron 保留 FP32 token 缩放因子与视觉 RoPE 频率，修低精度转换损失

梯度 buffer 乘0维 CUDA factor 时不触发 dtype promotion，FP16 token 倒数可能先下溢、BF16先舍入；改用一元素 view 后在乘法保留 FP32，最终存储才舍入，独立 FP32/scalar gradients 保持原 scalar factor，CUDA graph replay 仍读取动态值。Pixtral vision RoPE 则把频率从随 module 转 dtype 的 buffer 改为 FP32 属性，首次使用缓存 device copy，避免先把频率舍入后再算角度。作者4×GB300报告缩放37例、视觉28例每rank通过；这是数值正确性修复，不是速度结论。建议低精度长序列训练和视觉模型升级做 eager/graph、转换前后 device move 回归。

机构/社区：NVIDIA。证据基线：[NVIDIA/Megatron-LM #8532](https://github.com/NVIDIA/Megatron-LM/pull/8532)，2026-10-28T00:03:23Z，main @ 53015a0d7bf4；[NVIDIA/Megatron-LM #8533](https://github.com/NVIDIA/Megatron-LM/pull/8533)，2026-10-27T19:55:26Z，main @ 3ca52d95d6f4。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`megatron/core/distributed/param_and_grad_buffer.py::scale_gradients`；`megatron/core/models/vision/vit_model.py::forward`。

稳定ID：`github:NVIDIA/Megatron-LM:pr:8532`；`github:NVIDIA/Megatron-LM:pr:8533`。

### Megatron MFSDP v2 把 unshard 完成状态移到参数组，并保留 retention 衰减标记

每个 FsdpParameterGroup 与自身 gathered storage 一起管理 readiness event：BF16/replicated 前后向共享完成状态，sharded MXFP8 的 rowwise/columnwise 分别记录，释放哪一平面就失效对应 event，module 仍负责 stream/prefetch 与 compute wait。replicated 权重直接别名持久 model_weight，避免混淆临时 gather 所有权。另在替换/分片参数时保留 is_wide_residual_retention_parameter，防 retention vector 被一维参数规则误关 weight decay。作者两卡 lifecycle probe 不含 MXFP8 GEMM，Blackwell模型级数值/性能仍待；A6000无计算微基准反而增加控制开销。建议混合 BF16/MXFP8 与重计算用户核对释放/重用顺序，并与未分片 optimizer 更新对齐。

机构/社区：NVIDIA。证据基线：[NVIDIA/Megatron-LM #8464](https://github.com/NVIDIA/Megatron-LM/pull/8464)，2026-10-27T19:48:41Z，main @ 17fc7ba10998；[NVIDIA/Megatron-LM #8516](https://github.com/NVIDIA/Megatron-LM/pull/8516)，2026-10-27T04:56:16Z，main @ 3ddfe0fe07ef。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/parameter_group.py::unshard_parameters`；`megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/module.py::unshard`；`megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/module_utils.py::_PARAMETER_ATTRIBUTES_TO_PRESERVE`。

稳定ID：`github:NVIDIA/Megatron-LM:pr:8464`；`github:NVIDIA/Megatron-LM:pr:8516`。

### Megatron GDP 在单 microbatch 与 chunkwise CP 保留 gate view，使投影重计算真正释放激活

gdp_in_proj recompute 过去丢弃投影输出，却被 gated RMSNorm 保存的 contiguous gate 副本抵消。现在仅在 recompute_in_proj、batch size1、CP1或chunkwise CP 同时成立时保留 projection-storage view，让 checkpoint 一起丢弃/恢复 norm 保存 gate；headwise CP 和更大 batch 仍 materialize。补丁是特定布局下的显存生命周期修复，未给通用吞吐或显存数字。建议按这些开关组合检查前后向数值与 saved-tensor storage。

机构/社区：NVIDIA。证据基线：[NVIDIA/Megatron-LM #8517](https://github.com/NVIDIA/Megatron-LM/pull/8517)，2026-10-27T04:07:13Z，main @ 63f687dc83f4。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`megatron/core/ssm/gated_delta_product.py::_postprocess`；`tests/unit_tests/ssm/test_gated_delta_product.py`。

稳定ID：`github:NVIDIA/Megatron-LM:pr:8517`。

### Megatron 推理 ETP 接入 tp_ep 集体组，路由权重在 FC2 累积器应用

动态推理 context、NVLS MoE 缓冲与 ZMQ 步进共识改覆盖 tp_ep，而非仅 EP，EP1/ETP>1 也能参与 dispatch；fused MoE 中间宽度按 ETP 切分。普通推理把 routing probability 放到 FC2 FP32 accumulator epilogue，再做未加权 top-k sum；batch-invariant 路径仍在 activation 施权并 FP64 sum。PR 正文仍列 NCCLAllGather dispatcher/测试 TODO，不能据标题称所有 dispatcher 与 ETP 组合闭环。建议先按实际选中的 NVLS 路径验证 ETP rank 同步、缓冲几何与数值。

机构/社区：NVIDIA。证据基线：[NVIDIA/Megatron-LM #4743](https://github.com/NVIDIA/Megatron-LM/pull/4743)，2026-10-27T18:03:45Z，main @ 2abdd0d1c601。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`megatron/core/inference/contexts/dynamic_context.py::DynamicInferenceContext`；`megatron/core/inference/engines/dynamic_engine.py::_ep_establish_consensus`；`megatron/core/transformer/moe/token_dispatcher_inference.py`；`megatron/core/inference/moe/vllm_fused_moe.py::vllm_fused_moe`。

稳定ID：`github:NVIDIA/Megatron-LM:pr:4743`。

### Megatron P/D KV handoff 支持 TP 大于 KV head，复制副本按接收端配对

KVShardLayout 允许 head 数与 TP 任一整除另一方，TP>KV-head 时按 attention 的连续 rank 复制规则导出全局几何；destination 选择自身 replica index 模 source replica count，分摊读取并让单源/全局规划一致，实际 heads_per_partition 不符直接报错。EP/ETP 去重规则不变。作者4×GB200 tiny GPT 实测 TP2→2 token/logprob bitwise一致，TP2↔1 token一致但 logprob有BF16跨TP差。建议 GQA/MQA 异构TP部署测试复制副本、PP变化与push/pull配对；这不是全模型质量验证。

机构/社区：NVIDIA。证据基线：[NVIDIA/Megatron-LM #8506](https://github.com/NVIDIA/Megatron-LM/pull/8506)，2026-10-27T21:12:56Z，main @ 259f6a7c2711。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`megatron/core/inference/disaggregation/kv_reshard.py::KVShardLayout、plan_kv_reshard`；`megatron/core/inference/disaggregation/transfer_backends/base.py::compute_buffer_geometry`。

稳定ID：`github:NVIDIA/Megatron-LM:pr:8506`。

### Megatron VLM 等待队列联合编码图像，编码失败按请求隔离

动态分辨率图像的 [P,F]/[1,P,F] patch 请求延迟到 prefill 前，按 context.max_tokens 的 projected embedding 预算合批编码、拆回请求；相同 media_cache_key 去重，整批失败或计数不符退回逐请求编码。视频、静态tiling和不支持布局保留原路径，预算不是 encoder patch 总数。作者 A10080GB×2、Nemotron3 Nano Omni、64个不同448×448图、每请求64输出token、TP2/EP2，571→1019 output tok/s（1.78×）；仍低于同场 vLLM，批组成引起的BF16差异存在。建议突发图像服务复现 OOM fallback、取消与embedding计数，不外推视频或全模态收益。

机构/社区：NVIDIA。证据基线：[NVIDIA/Megatron-LM #8408](https://github.com/NVIDIA/Megatron-LM/pull/8408)，2026-10-27T02:00:34Z，main @ 1142209a1905。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`megatron/core/inference/engines/dynamic_engine.py::_encode_waiting_images、_encode_deferred_image_requests`；`megatron/core/inference/inference_request.py::DynamicVLMInferenceRequest`。

稳定ID：`github:NVIDIA/Megatron-LM:pr:8408`。

### Megatron tensor metrics 接入训练，并把诊断迭代的 FT timeout 单独计费

新增 NAME:INTERVAL 形式 tensor_metrics 配置及训练 observer/optimizer source 接线；周期 router reduction 的整轮 forward+commit 放入 tensor_metrics FT section，避免昂贵诊断触发普通 step 的紧超时。需要为该 section 显式设置独立预算；未设置时诊断迭代不受此 section 监控，不能误称无条件增强hang检测。PR main 聚合包含原 #7471 定义与被删临时base后重建的 #6650 集成。建议先小频率启用并检查诊断开销和超时配置。

机构/社区：NVIDIA。证据基线：[NVIDIA/Megatron-LM #7472](https://github.com/NVIDIA/Megatron-LM/pull/7472)，2026-10-27T21:48:19Z，main @ 364eef70a30a。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`megatron/training/config/training_config.py::LoggerConfig.tensor_metrics`；`megatron/training/ft_integration.py::on_training_step_start`；`megatron/training/tensor_metrics/observer.py`。

稳定ID：`github:NVIDIA/Megatron-LM:pr:7472`。

### TorchTitan 同配置 RoPE cache 共享，降低稳态显存但不降低初始化峰值

Decoder 构造与 init_states 后，将同 RoPE 类型且 config 相等的只读、非持久 cache 指向一份 tensor；每个 PP model part 自然独立处理。to_empty 和各层 init 仍先生成独立 cache 再绑定，所以只降低稳态内存；之后 model.to 等 _apply 会再次复制，影响节省而非正确性。建议长上下文深层模型检查初始化峰值与后续device转换，不把共享计为峰值优化。

机构/社区：PyTorch/TorchTitan。证据基线：[pytorch/torchtitan #5583](https://github.com/pytorch/torchtitan/pull/5583)，2026-10-27T03:30:59Z，main @ e111f10d4db3。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`torchtitan/models/common/decoder.py::_tie_rope_caches`；`tests/unit_tests/cpu/models/common/test_rope_cache_tying.py`。

稳定ID：`github:pytorch/torchtitan:pr:5583`。

### TorchTitan Qwen3-235B 原始 flavor 的 RoPE theta 改回 1e6

235B-A22B flavor 原把2507版本的5e6 theta 与原始40960上下文混用，现改为原始公开checkpoint的1e6；作者CPU缩小模型对比显示 attention差显著收敛，但 MoE BF16 grouped GEMM 仍使logits非FP32精确一致。已有在TorchTitan用旧theta从零训练的checkpoint会看到不同频率，不能无条件套用“兼容修复”。建议加载官方原始权重采用新flavor，自训练恢复明确固定历史theta。

机构/社区：PyTorch/TorchTitan。证据基线：[pytorch/torchtitan #5634](https://github.com/pytorch/torchtitan/pull/5634)，2026-10-27T06:14:22Z，main @ 13806cbae05b。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`torchtitan/models/qwen3/flavors.py::_235b_a22b`。

稳定ID：`github:pytorch/torchtitan:pr:5634`。

### TorchTitan MoE 只在真正消费统计的 dispatcher 计算 per-expert count

MoE/Kimi/MuseSpark 传 routing_map，Local/AllToAll/TorchAO dispatcher 内部才求sum；Dist-MoE、DeepEP、HybridEP从expert IDs推导调度，不再先做丢弃的reduction，aux-loss统计不变。作者4×GB300、Qwen3.5-35B MoE block、EP4、8层、CUDA graph/no-grad decode、T32/64/256，单层耗时下降4.4–6.8%；无端到端、prefill、训练或DeepEP/HybridEP性能测量，且计时早于最终重构。建议对实际训练/服务后端单独profile，不把微基准当全栈收益。

机构/社区：PyTorch/TorchTitan。证据基线：[pytorch/torchtitan #5516](https://github.com/pytorch/torchtitan/pull/5516)，2026-10-27T21:04:49Z，main @ 72a3a126fe73。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`torchtitan/models/common/moe/moe.py::MoE.forward`；`torchtitan/models/common/moe/token_dispatcher.py::dispatch`；`torchtitan/models/common/dist_moe/routed_experts.py::forward`。

稳定ID：`github:pytorch/torchtitan:pr:5516`。

### TorchTitan 首步 optimizer 同步兼容 MPS；Muse Glimmer softcap 局部编译补入

首步清cache前仅在 backend 存在 is_initialized 时查询，避免 Mac 上 torch.mps 无该函数导致CPU训练也崩溃，同时保留CUDA/XPU未初始化时不创建context的guard。72h回溯补收 Muse Glimmer logits后处理 local_compile：从保存的BF16 logits重算tanh，减少保存FP32中间量的流量；temp与倒数消去仍TODO，不能称双重舍入已修。建议Mac首步与Glimmer反向数值分别回归。

机构/社区：PyTorch/TorchTitan。证据基线：[pytorch/torchtitan #5625](https://github.com/pytorch/torchtitan/pull/5625)，2026-10-27T08:10:12Z，main @ 2ffc9fd3f802；[pytorch/torchtitan #5630](https://github.com/pytorch/torchtitan/pull/5630)，2026-10-26T21:25:11Z，main @ 18e92bae9215。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`torchtitan/components/optim/optim.py::step`；`torchtitan/models/muse_glimmer/model.py::postprocess_logits`。

稳定ID：`github:pytorch/torchtitan:pr:5625`；`github:pytorch/torchtitan:pr:5630`。

### verl FSDPTurbo NPU 示例把 EP dispatcher 与 fused ops 解耦

Qwen3.5-35B actor 与 reference 示例统一 dispatcher=all2all，另以 use_fused_ops 区分设备；NPU配置16卡/EP16并开启 fused ops，GPU保持关闭。它修的是示例配置契约，不能解释为新增NPU kernel或完整训练性能结果。建议沿此脚本运行的Ascend团队同步actor/ref两侧参数，再做真实训练冒烟。

机构/社区：verl community。证据基线：[verl-project/verl #8400](https://github.com/verl-project/verl/pull/8400)，2026-10-27T11:46:59Z，main @ 4d3ed12a7617。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`examples/grpo_trainer/run_qwen3_5_35b_fsdp_turbo.sh::ACTOR、REF`。

稳定ID：`github:verl-project/verl:pr:8400`。

### verl RL-Insight 将训练标量按 step 记录，并观察权重同步与睡眠阶段

metric_log 可调用时写 step/global_step，否则退回旧 metric_gauge且每标量只写一次；已发布rl-insight0.3.0缺metric_log，不能仅按版本号判断。异步 decorator 在运行时解析 rank/actor lane，覆盖 update_weight 与 vLLM/SGLang sleep/wake/cache transitions，未开启logger不产生span。建议RL团队用timeline分离权重同步与rollout空档，升级时核对实际包能力；这是观测改进，没有训练吞吐结论。

机构/社区：verl community。证据基线：[verl-project/verl #8315](https://github.com/verl-project/verl/pull/8315)，2026-10-27T06:23:05Z，main @ 6b2b417a059f。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`verl/utils/tracking.py::RLInsightLogger.log、trace_state_decorator`；`verl/workers/engine_workers.py::update_weights`；`verl/workers/rollout/vllm_rollout/vllm_async_server.py`。

稳定ID：`github:verl-project/verl:pr:8315`。

### slime 外部 engine 重复注册先核对既有 payload，允许连续 rollout 复用

遇409 WORKER_ALREADY_EXISTS 后读取router worker列表，仅 url/worker_type/prefill bootstrap_port 与本次payload一致才保留；不同注册、其他409或错误仍抛出，2xx行为不变。修外部engine跨run存活而create-only router拒绝再次POST的问题。作者4×H100、Qwen3-0.6B、两个外部engine+SMG连续两轮3-step复现通过，但为旧基线cherry-pick，未把所有router列为已验证。建议长生命周期外部rollout集群回归冲突payload与故障重启。

机构/社区：THUDM/slime community。证据基线：[THUDM/slime #2483](https://github.com/THUDM/slime/pull/2483)，2026-10-27T03:06:27Z，main @ cd55985ff716。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`slime/backends/sglang_utils/server_control.py::register_worker、_holds_registration`；`slime/backends/sglang_utils/sglang_engine.py::_register_to_router`。

稳定ID：`github:THUDM/slime:pr:2483`。

### vLLM HiSparse 修 residency 跨 stream race，并为 staged prefill KV 留最小 backing

prefill不再延迟gather batch-ordered residency表，而把group-major持久表与request_state_indices直接传remap kernel，使既有event覆盖真正依赖，去掉未被等待的写入。另对staged KV至少分配256行、仍返回精确尺寸view，规避FlashInfer trtllm-gen对短cache起始越界读取。作者2×B200+FlashInfer0.7.0.post1/dummy8层GLM服务验证越界修复，但DBO另有#61487失败未闭环；8×B200真实GLM TP8在2类长度/并发16/64端到端基本持平。建议优先测短1–3页cache、side stream与DBO分别复现，不把kernel 2×提升套到服务吞吐。

机构/社区：vLLM community。证据基线：[vllm-project/vllm #62739](https://github.com/vllm-project/vllm/pull/62739)，2026-10-28T00:57:30Z，main @ 66f1d0fa169b；[vllm-project/vllm #61659](https://github.com/vllm-project/vllm/pull/61659)，2026-10-28T01:00:47Z，main @ ebaa8124755b。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`vllm/v1/attention/backends/mla/index_group.py::convert_logical_to_physical_topk`；`vllm/v1/hisparse/binding.py::bind_hisparse_kv_caches`；`vllm/v1/hisparse/runtime.py::gather_prefill_cache`。

稳定ID：`github:vllm-project/vllm:pr:62739`；`github:vllm-project/vllm:pr:61659`。

### vLLM 结构化推测解码保留首个 grammar-invalid draft，修输出分布偏移

同步路径保留首个无效proposal供 rejection sampling，异步路径不提前过滤draft slots；grammar accept失败必须原子不改变状态，XGrammar/Guidance分别补rollback。首个invalid proposal的验证行仍受有效prefix grammar约束，之后槽不推进grammar而填全通mask，交rejection处理，最后回滚此前有效推进。作者4000样本分布测试、Qwen3-4B+DSpark15/adaptive、单H10080GB，修前显著偏移修后不显著；十轮性能同步高并发可退约2–2.7%，异步c96约+2.3–3.4%，不能写普适加速。建议structured-output+spec服务优先验证分布与grammar回滚，再权衡吞吐。

机构/社区：vLLM community。证据基线：[vllm-project/vllm #61557](https://github.com/vllm-project/vllm/pull/61557)，2026-10-27T21:09:33Z，main @ c4f2bb4abbdb。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`vllm/v1/structured_output/__init__.py::trim_spec_tokens、grammar_bitmask`；`vllm/v1/structured_output/backend_xgrammar.py::accept_tokens`；`vllm/v1/structured_output/backend_guidance.py::accept_tokens`。

稳定ID：`github:vllm-project/vllm:pr:61557`。

### vLLM PCP 暂停 draft-logits batch sharding，修刚接入的索引组合

_init_draft_logits 增加 prefill_context_parallel_size==1 guard：PCPManager.sync_drafts 仍按完整request state index读logits，无法理解request-owner分片。PCP因此保留完整cache，不能沿用昨日#63554的TP分片节省到PCP场景；draft计算本来就未分片。建议PCP+推测部署采用此组合guard后再查cache峰值。

机构/社区：vLLM community。证据基线：[vllm-project/vllm #64151](https://github.com/vllm-project/vllm/pull/64151)，2026-10-28T00:00:24Z，main @ 820be160ce0d。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`vllm/v1/worker/gpu/spec_decode/speculator.py::_init_draft_logits`。

稳定ID：`github:vllm-project/vllm:pr:64151`。

### vLLM FlashInfer MLA DCP 将 ragged decode 拆为单token因果行

判定ragged改为总token数不等于requests×max_query_len，避免8+2被误reshape成5+5；展开block_table/seq_lens，并把每行global exclusive causal bound减去该请求后续query行数，兼顾DCP所需LSE。作者GB200 kernel/adapter23通过1跳过，未跑Kimi-K3端到端。建议DCP+DSpark测试不同draft数与“总数恰好整除”的反例，不只验证启动shape。

机构/社区：vLLM community。证据基线：[vllm-project/vllm #59392](https://github.com/vllm-project/vllm/pull/59392)，2026-10-27T23:59:59Z，main @ 8697bbae4c7d。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`vllm/v1/attention/backends/mla/flashinfer_mla.py::forward_mqa`；`tests/v1/attention/test_flashinfer_mla_dcp.py`。

稳定ID：`github:vllm-project/vllm:pr:59392`。

### vLLM NVFP4 MLA chunked prefill 用专用 gather 与 seq_starts，DCP 仍拒绝

nvfp4_ds_mla cached context不再落入只懂FP8的generic gather，专用upconvert kernel接chunk.starts以处理非block对齐起点；DCP在启动校验显式拒绝。BF16 workspace旧正文修改已删，因为实际Sparse builder另行分配；其他不可达generic调用留后续，不能称全MLA路径已支持。作者B300 SM103/source build记录17+27相关通过。建议prefix-hit/chunked-preload用户测非对齐chunk与warmup，保留DCP禁配。

机构/社区：vLLM community。证据基线：[vllm-project/vllm #58744](https://github.com/vllm-project/vllm/pull/58744)，2026-10-27T18:35:02Z，main @ 741584f8ca89。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`csrc/nvfp4_ds_mla_cache_kernels.cu::cp_gather_and_upconvert_nvfp4_kv_cache_kernel`；`vllm/model_executor/layers/attention/mla_attention.py::_compute_prefill_context`；`vllm/config/vllm.py::validate_nvfp4_kv_cache_with_mla`。

稳定ID：`github:vllm-project/vllm:pr:58744`。

### vLLM GLM DSpark draft 可继承 FP8 KV 默认，内存收益按固定slot计

GLM SM10x自动FP8 KV时，dspark与mtp一起免于把draft重置auto，因此单独DSpark也用FP8 KV。正文5.38GiB/rank是TP4、三draft attention层、940736固定KV slots条件下的容量推算；8192 slots只省48MiB，不能称所有服务固定省5.38GB。作者TP4、2输入/1024输出、c1性能近乎不变，AIME小样本分数差不能推断质量提升。建议按实际slot/层数核算，并验证采样质量。

机构/社区：vLLM community。证据基线：[vllm-project/vllm #63841](https://github.com/vllm-project/vllm/pull/63841)，2026-10-27T21:09:00Z，main @ c6f6e6a9835c。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`vllm/model_executor/models/config.py::GlmMoeDsaForCausalLM.verify_and_update_config`。

稳定ID：`github:vllm-project/vllm:pr:63841`。

### vLLM KVCR 按请求前缀更新 recency，并削减 pin/descriptor 的 Python 开销

load/store真正完成后调用align_sequence按请求prefix顺序对齐，结束时只touch已处理、非inflight、排除最后未提交sample token的前缀；相邻同request completion可合并，但跨request顺序保留。后续优化bulk转chunk IDs、缓存agent名和省掉全成功时无用successful-key累积，保留部分失败报告。只有回归测试，无端到端收益数字。建议使用KVCR用户检查prefix eviction与部分失败，而不是将submit当完成。

机构/社区：vLLM community。证据基线：[vllm-project/vllm #60469](https://github.com/vllm-project/vllm/pull/60469)，2026-10-27T20:04:20Z，main @ e3883b7a7ec4；[vllm-project/vllm #63889](https://github.com/vllm-project/vllm/pull/63889)，2026-10-27T23:41:01Z，main @ 36f5ff3172eb。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`vllm/v1/kv_offload/tiering/kvcr/manager.py::_poll_finished_jobs、_align_sequence、_resolve_pin`。

稳定ID：`github:vllm-project/vllm:pr:60469`；`github:vllm-project/vllm:pr:63889`。

### vLLM NIXL hybrid transfer 只读本组 regions，依赖 pin 同日回退1.5.0

hybrid attention/SSM共享allocation时，注册阶段记录每transfer group实际拥有的regions，descriptor只覆盖其层，避免每attention block重读其他组bytes；wire format不变，作者GLM-5.3-Flash跨GB300节点TP2→TP2/pull/FP8曾约38%wire bytes重复，这不是所有模型比例。同日#64085撤回昨日1.5.1 pin，CUDA12/13均回1.5.0，因上游意外带入libfabric2.7.0影响EFA用户。建议EFA部署核对实际NIXL依赖，混合模型测wire bytes与SSM/PLE/scratch组几何。

机构/社区：vLLM community。证据基线：[vllm-project/vllm #62317](https://github.com/vllm-project/vllm/pull/62317)，2026-10-27T18:02:23Z，main @ 4ddc7aa40f80；[vllm-project/vllm #64085](https://github.com/vllm-project/vllm/pull/64085)，2026-10-27T22:07:03Z，main @ 0855defde8ea。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py::_compute_desc_ids、register_kv_caches`；`requirements/kv_connectors.txt`；`requirements/kv_connectors_cu12.txt`。

稳定ID：`github:vllm-project/vllm:pr:62317`；`github:vllm-project/vllm:pr:64085`。

### vLLM hybrid KV copy-on-write 按完整 view 几何去重，修部分 prefix hit 静默漏拷

同data_ptr的Mamba/attention覆盖view每block跨度不同，旧去重可能只拷窄view，Kimi-K3高TP prefill的MLA多数token槽留空。现在key包含device/address/shape/stride/element_size，只跳过真正同几何别名。修的是local partial-prefix CoW，异构TP P/D放大了症状；不能当作新增异构TP能力。建议混合SSM模型强制preemption+prefix-match边界，逐token比较与检查各view覆盖bytes。

机构/社区：vLLM community。证据基线：[vllm-project/vllm #55112](https://github.com/vllm-project/vllm/pull/55112)，2026-10-27T02:32:05Z，main @ 8701c9b7513d。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`vllm/v1/worker/utils.py::copy_kv_cache_blocks_inplace`；`tests/v1/worker/test_attn_utils.py`。

稳定ID：`github:vllm-project/vllm:pr:55112`。

### vLLM prompt scoring 仅复用首个评分位置之前的 local prefix

request统一prefix_cache_hit_limit，上界为prompt_logprob_start，最后token仍重算；有界请求跳过external KV connector，因为其命中不能按该边界裁剪，prompt scoring默认仍不读cache。输出补齐未评分位置。作者batch-invariant Qwen3-0.6B评分精确匹配；H200默认BF16虽完整却最大差0.29844，原因未闭环，不能称默认score等价。建议评测/蒸馏数据管道启用batch-invariant并检查完整性与容差，区分local缓存与外部缓存。

机构/社区：vLLM community。证据基线：[vllm-project/vllm #60960](https://github.com/vllm-project/vllm/pull/60960)，2026-10-27T19:26:08Z，main @ e1e77a279d68。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`vllm/v1/request.py::get_prefix_cache_hit_limit、skip_reading_external_kv`；`vllm/v1/core/kv_cache_manager.py::get_computed_blocks`；`vllm/v1/worker/gpu/sample/prompt_logprob.py`。

稳定ID：`github:vllm-project/vllm:pr:60960`。

### vLLM DSv4.1 稀疏 indexer logits 接受 FP8 cache，满足条件时默认开启

SM100 FP8 indexer cache已有UE8M0 packed scale，可复用DeepGEMM sparse MQA kernels，无需先把K降为MXFP4。indexer_sparse_logits改三态：None按要求自动开、True强制并报错、False关；要求SM100、DeepGEMM≥2.8、DeepSelect、FP8或MXFP4，且无DCP/PCP。ROCm仍不自动开启。建议长上下文候选块工作负载分别测prefill/decode及短上下文开销，不能把此路径等同所有稀疏attention或改变模型精度。

机构/社区：vLLM community。证据基线：[vllm-project/vllm #62011](https://github.com/vllm-project/vllm/pull/62011)，2026-10-27T19:25:20Z，main @ a2ad4b74b9f7。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`vllm/config/attention.py::AttentionConfig.indexer_sparse_logits`；`vllm/model_executor/layers/sparse_mqa_indexer.py`；`vllm/model_executor/kernels/attention/dsa/sparse_mqa_logits.py`。

稳定ID：`github:vllm-project/vllm:pr:62011`。

### vLLM 图内存修复同时处理 nested output 强引用与 decoder replay stream

weak_ref_tensors递归处理list/tuple，避免每capture size持有aux hidden states；unknown nested type仍报错，int/LayerName保留。DSv4.1 decoder replay graph转到model capture stream，以共享pool空闲块，speculator/encoder及MRV1不变。作者GB200 DP4/DSpark/breakable PIECEWISE≤16384、两次冷启动测capture15.52→9.88GiB/rank；该测量基于0.32.0组合，不能当当前main普适节省。建议EAGLE/DSpark/DFlash检查多capture-size峰值、early aux生命周期和logprobs。

机构/社区：vLLM community。证据基线：[vllm-project/vllm #63427](https://github.com/vllm-project/vllm/pull/63427)，2026-10-27T03:04:50Z，main @ f81f0f250954；[vllm-project/vllm #63155](https://github.com/vllm-project/vllm/pull/63155)，2026-10-27T20:45:58Z，main @ 274a5e19a184。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`vllm/utils/torch_utils.py::weak_ref_tensors`；`vllm/distributed/parallel_state.py::graph_capture`；`vllm/models/deepseek_v41/decoder_replay_layers.py`；`vllm/v1/worker/gpu/model_runner.py`。

稳定ID：`github:vllm-project/vllm:pr:63427`；`github:vllm-project/vllm:pr:63155`。

### vLLM DSv4.1 禁止 sparse prefill/mixed FULL capture，保留有界 uniform decode 图

五类sparse/compressor metadata builder从ALWAYS降为UNIFORM_BATCH，query bound设为1+speculative width，避免不graph-safe的prefill metadata进入FULL capture；prefill走piecewise、uniform varlen decode仍可图执行。作者仅CPU82例与既有issue证据，B300端到端未重跑。建议adaptive-verification+并发混合batch先压力测，不把窄化capture范围当增加模型支持。

机构/社区：vLLM community。证据基线：[vllm-project/vllm #59986](https://github.com/vllm-project/vllm/pull/59986)，2026-10-27T08:07:31Z，main @ 8977b13a1c38。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`vllm/models/deepseek_v41/compressor.py::CompressorMetadataBuilder`；`vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py`；`vllm/models/deepseek_v41/nvidia/flashmla.py`。

稳定ID：`github:vllm-project/vllm:pr:59986`。

### vLLM level-2 sleep 保留 MRV2 runtime model state，避免权重恢复后输出仍坏

init_model_state在sleep开启的CUDA/ROCm运行时放入runtime cuMem pool，不再随weights池丢弃；Qwen PLE ngram_context_offsets等非module常量因此随sleep拷host并恢复。XPU不走nested pool；额外host备份包含per-request bookkeeping及部分模型的max_tokens×hidden inputs_embeds。建议RL反复sleep/reload测试state tensor和输出，而不只比较权重bitwise。

机构/社区：vLLM community。证据基线：[vllm-project/vllm #63548](https://github.com/vllm-project/vllm/pull/63548)，2026-10-27T08:32:09Z，main @ cebcc5d8c744。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`vllm/v1/worker/gpu/model_runner.py::load_model`；`tests/basic_correctness/memory/sleep_mode/test_sleep_mode.py::test_deep_sleep、_model_state`。

稳定ID：`github:vllm-project/vllm:pr:63548`。

### vLLM process checkpoint 增 prepare/status/restore HTTP 生命周期

开启sleep-mode后Python/Rust前端提供checkpoint prepare(wait/abort/keep)、status、restore及startup prepare，engine状态running→preparing→prepared→restoring→running，失败保留failed。prepare返回200只表示开始，status200才表示全部prepared；health仍可200，不能当checkpoint readiness。清KV并调用worker hooks，真正CRIU/cuda-checkpoint由外部编排器执行；DP引擎独立推进，部分拒绝可能已有其他engine开始，无全局回滚保证。建议编排器按status轮询与幂等重投，并测试失败/并发/睡眠冲突。

机构/社区：vLLM community。证据基线：[vllm-project/vllm #62570](https://github.com/vllm-project/vllm/pull/62570)，2026-10-27T08:02:45Z，main @ 05a303fdc868。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`vllm/v1/engine/core.py::checkpoint_prepare、checkpoint_restore、checkpoint_state`；`vllm/entrypoints/serve/checkpoint/api_router.py`；`rust/src/server/src/routes/checkpoint.rs`。

稳定ID：`github:vllm-project/vllm:pr:62570`。

### vLLM Hopper FP8 MoE 权重改 TMA load，收益受shape与架构 gate约束

Triton fused MoE仅在CUDA SM90、block FP8 W8A8、连续权重且N/K tile对齐时用TensorDescriptor读B，TD gather路径与TMA_B互斥，其余保留原load。作者H200 GLM-5.3 TP8/MTP5、8k输入/1k输出、并发4/16，P50吞吐+3.4%/+3.1%；缺并发1复测，微kernel15–20%不能当E2E增幅。建议Hopper按真实expert形状验证数值与并发曲线。

机构/社区：vLLM community。证据基线：[vllm-project/vllm #63820](https://github.com/vllm-project/vllm/pull/63820)，2026-10-27T22:21:39Z，main @ c56706e88e0a。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`vllm/model_executor/layers/fused_moe/fused_moe.py::invoke_fused_moe_triton_kernel、fused_moe_kernel`。

稳定ID：`github:vllm-project/vllm:pr:63820`。

### vLLM ROCm MLA 按真实head成本回退，并折叠 Kimi decode query 线性映射

gfx950 FP8 prefill若padding至16会让head数至少翻倍（每rank1–8），恢复BF16 prefill，9–15与16+及decode不变。Kimi-K3 decode在load时折叠q_b_proj与W_UK为一GEMM，限unquantized projection、无额外RoPE/DCP复制/indexer且走支持prep；prefill保留原路径。折叠读权重/FLOPs更多，作者全93层多约1.9GB，收益偏小M；不可称无代价fusion。建议TP8的Kimi分别看prefill/head padding和decode内存账。

机构/社区：vLLM community。证据基线：[vllm-project/vllm #61863](https://github.com/vllm-project/vllm/pull/61863)，2026-10-27T17:28:38Z，main @ d0c20bb840c7；[vllm-project/vllm #60110](https://github.com/vllm-project/vllm/pull/60110)，2026-10-27T16:34:59Z，main @ 6aa70c4beca4。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`vllm/v1/attention/backends/mla/rocm_aiter_mla.py::AiterMLAHelper.is_valid_fp8_prefill_num_heads`；`vllm/models/kimi_k3/amd/mla.py::_compute_w_fold、process_weights_after_loading`。

稳定ID：`github:vllm-project/vllm:pr:61863`；`github:vllm-project/vllm:pr:60110`。

### vLLM ROCm DSv4 稀疏 MLA 融合 inverse RoPE，candidate mask合为单次launch

AITER sparse MLA store直接做inverse RoPE，V4.1可同时写MXFP8 o_proj输入，需AITER0.1.25对应接口；V4的FP8 wo_a保留既有旋转路径，prefill每rank>64head仍回gather+OPUS。另把candidate flags与mask融为一kernel，去掉cast/zero launch，并显式等global stores后barrier；只保证[0,end)掩码语义。作者MI355X candidate mask微基准1.06–2.50×，E2E均值仅约+0.44%；fusion各模型/TP比较口径不同。建议逐路径profile与长context校验，不叠加两PR百分比。

机构/社区：vLLM community。证据基线：[vllm-project/vllm #60154](https://github.com/vllm-project/vllm/pull/60154)，2026-10-27T11:15:05Z，main @ 8d88bc5d815f；[vllm-project/vllm #59668](https://github.com/vllm-project/vllm/pull/59668)，2026-10-27T16:08:15Z，main @ 228b1e3d235a。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`vllm/_aiter_ops.py::triton_sparse_mla_fwd`；`vllm/models/deepseek_v41/amd/rocm.py`；`vllm/v1/attention/ops/rocm_aiter_mla_sparse.py::_mask_candidates_fused_kernel`。

稳定ID：`github:vllm-project/vllm:pr:60154`；`github:vllm-project/vllm:pr:59668`。

### vLLM RDNA3 扩到 AITER Triton 路径，dense W4A16 prefill cache另为opt-in

AITER RDNA gate从gfx12扩为gfx11+gfx12（排除CDNA gfx1250），接线FP8/MoE/GDN/sparse MLA；作者W7900×8 DSv4测试命令实际带enforce-eager，虽正文称cudagraph模式，不能据此证明图执行。另在RDNA hybrid W4A16提供默认关闭的dense权重cache，M≥256用FP16/BF16 GEMM换掉反复int4 dequant，小batch/verify/decode保留int4；cache约为int4层权重4倍并挤占KV。gfx1151、2048输入/c1作者TTFT约降21–26%，不是所有RDNA实测。建议大统一内存设备按KV容量换TTFT，离散卡先核内存预算。

机构/社区：vLLM community。证据基线：[vllm-project/vllm #52970](https://github.com/vllm-project/vllm/pull/52970)，2026-10-27T17:01:14Z，main @ eaf1627c01f5；[vllm-project/vllm #60710](https://github.com/vllm-project/vllm/pull/60710)，2026-10-27T16:57:53Z，main @ 94ad4dfbaec9。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`vllm/_aiter_ops.py::is_rdna_aiter_enabled`；`vllm/config/kernel.py::rdna_hybrid_w4a16_prefill_dequant`；`vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py`。

稳定ID：`github:vllm-project/vllm:pr:52970`；`github:vllm-project/vllm:pr:60710`。

### vLLM Qwen3.8 QSA 按量化方法决定融合，PLE/PP 放开但验证依赖需分清

QSA indexer从全局packed mapping移出，只有与q/k/v量化方法一致才融合，解决qkv量化而indexer BF16 checkpoint加载失败。PLE因MRV2已向各PP stage传input_ids而移除PP1限制，KV planning跳过本stage无层的uniform group，MTP按intermediate_tensors存在分支。作者H200 PP4/MTP验证曾加#62129；本轮已核该warmup deadlock修复于10-18合入main，但特殊PLE放rank1的验证仍有本地core workaround，不能称任意partition已完整复现。建议分别测试mixed quant加载、PP空group与draft warmup。

机构/社区：vLLM community。证据基线：[vllm-project/vllm #62318](https://github.com/vllm-project/vllm/pull/62318)，2026-10-27T11:54:15Z，main @ a91a1d06b72a；[vllm-project/vllm #60234](https://github.com/vllm-project/vllm/pull/60234)，2026-10-27T08:37:22Z，main @ 5c1eb11cb361。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`vllm/models/qwen4_exp/nvidia/model.py::packed_modules_mapping`；`vllm/models/qwen4_exp/nvidia/qsa.py::fuse_indexer_projection`；`vllm/v1/core/kv_cache_planning.py::get_kv_cache_config_from_groups`；`vllm/models/qwen4_exp/nvidia/mtp.py::forward`。

稳定ID：`github:vllm-project/vllm:pr:62318`；`github:vllm-project/vllm:pr:60234`。

### vLLM draft sharing 与 XPU MoE LoRA 修精度和packed权重边界

EAGLE/DFlash/DSpark共享lm_head前先确认两侧plain weight都为Tensor，packed quantized head无法证明相等时保留draft自己的head，避免启动AttributeError；作者未找到公开checkpoint端到端复现，不能扩大为全量quantized draft已验证。XPU fused MoE LoRA expand保留FP32 dot与residual相加，最后store才舍入；CUDA/ROCm顺序不变，B60两卡433测试通过但BMG CI差异未本地复现。建议量化draft与XPU LoRA分别回归启动、残差数值。

机构/社区：vLLM community。证据基线：[vllm-project/vllm #53104](https://github.com/vllm-project/vllm/pull/53104)，2026-10-27T15:47:23Z，main @ 43111fd1f0c3；[vllm-project/vllm #63894](https://github.com/vllm-project/vllm/pull/63894)，2026-10-27T10:09:58Z，main @ ef2f382fd119。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`vllm/v1/worker/gpu/spec_decode/eagle/utils.py::_should_share`；`vllm/lora/ops/triton_ops/fused_moe_lora_op.py::_fused_moe_lora_kernel、_fused_moe_lora_expand`。

稳定ID：`github:vllm-project/vllm:pr:53104`；`github:vllm-project/vllm:pr:63894`。

### vLLM packed NCCL consumer 只等本slot完成事件，恢复 HIP共享queue重叠

用slot_done event替代整个slot stream.synchronize，避免HIP stream共享硬件queue时把其他slot较晚post_unpack/reload replay也等进来；本slot完成后才复用packing tensor。作者MI300X强制GPU_MAX_HW_QUEUES=1重现，默认4queue不触发；consumer-only微基准最多7.9%，不能当RL权重同步E2E收益。建议多buffer/layerwise reload压测queue共享和event寿命。

机构/社区：vLLM community。证据基线：[vllm-project/vllm #63833](https://github.com/vllm-project/vllm/pull/63833)，2026-10-27T03:01:03Z，main @ 1e10a92d1da2。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`vllm/distributed/weight_transfer/packed_tensor.py::packed_nccl_broadcast_consumer`。

稳定ID：`github:vllm-project/vllm:pr:63833`。

### vLLM Triton prefill tile 服从实际 shared-memory 容量；NUMA默认变为best-effort

prefill按Q/K/P工作区估算逐次减半tile直到适配设备shared-memory，修99KiB设备EmbeddingGemma2 warmup OOR，同时保留原本能放下的大tile。NUMA配置改三态，auto仅在支持的local mp/spawn、多NUMA、numactl、未受限CPU亲和/默认内存policy等条件生效，偏好本地内存而非强制membind；探测失败可退CPU-only/unbound，Ray/external/remap排除，显式参数仍严格并选spawn。建议消费卡检查warmup，容器/PCP部署核CPU亲和与实际policy，不能将auto开启理解为覆盖所有launch方式。

机构/社区：vLLM community。证据基线：[vllm-project/vllm #60942](https://github.com/vllm-project/vllm/pull/60942)，2026-10-27T02:00:15Z，main @ fe77d0906be0；[vllm-project/vllm #61258](https://github.com/vllm-project/vllm/pull/61258)，2026-10-27T21:36:02Z，main @ 4d95f222e68b。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`vllm/v1/attention/ops/triton_prefill_attention.py::_fit_block_to_shared_memory`；`vllm/config/parallel.py::ParallelConfig.numa_bind`；`vllm/utils/numa_utils.py`。

稳定ID：`github:vllm-project/vllm:pr:60942`；`github:vllm-project/vllm:pr:61258`。

### SGLang DSpark draft metadata 不再回读 host lengths，兼容无 Triton 设备

DeepSeek-V4 draft block固定extend长度可直接从device tensors准备，移除seq_lens.tolist导致的每步D2H同步和glue-graph永久fallback。随后torch fallback也从device extend lengths构造start loc，补XPU等无Triton路径；capture与replay统一needs_cpu_seq_lens，CI下声明可捕获却失败会raise，普通运行仍fallback eager。建议DSpark服务检查glue graph是否实际启用、跨设备fallback和acceptance，不把去host依赖等同所有metadata已图化。

机构/社区：SGLang community。证据基线：[sgl-project/sglang #46891](https://github.com/sgl-project/sglang/pull/46891)，2026-10-27T23:19:03Z，main @ a0d4034d5b32；[sgl-project/sglang #46897](https://github.com/sgl-project/sglang/pull/46897)，2026-10-28T01:00:42Z，main @ 357c53885fe7。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`python/sglang/srt/layers/attention/deepseek_v4_backend.py::init_forward_metadata_dspark_draft_block`；`python/sglang/srt/speculative/ragged_verify.py::compute_uniform_extend_lengths`；`python/sglang/kernels/ops/attention/dsv4/attn_metadata_kernels.py::expand_prefill_causally`。

稳定ID：`github:sgl-project/sglang:pr:46891`；`github:sgl-project/sglang:pr:46897`。

### SGLang Engram 用显式JSON配置 placement/replication/lookup，并变更旧host默认布局

engram-config三维分别选择gpu/cpu、none/per_node、collective/peer_read；GPU peer_read仅CUDA，per_node只允许CPU且当前要求TP单节点。参数显式设置时旧env忽略，否则翻译并警告；旧host-table开但没layout现在从共享mapping变为每rank分片，是实质默认变化。建议大Engram模型先固定启动解析日志、host共享方式与TP拓扑，再比较容量/带宽；不是所有硬件都支持peer读取。

机构/社区：SGLang community。证据基线：[sgl-project/sglang #46893](https://github.com/sgl-project/sglang/pull/46893)，2026-10-27T23:15:20Z，main @ 90a733941999。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`python/sglang/srt/arg_groups/deepseek_v4_hook.py::_resolve_engram_config`；`python/sglang/srt/arg_groups/engram_config.py::EngramConfig`；`python/sglang/srt/layers/engram.py::EngramEmbedding`。

稳定ID：`github:sgl-project/sglang:pr:46893`。

### SGLang gfx950 MTP verify 按CU扩segment，MegaMoE eager tile看DP全局token

vattn-asm小batch从最多64段扩到num_cus，每段至少256token，>64段改Triton reduction，seqused_k按forward缓存减少重复plan。MegaMoE eager用DP-global token count（容量cap）为kernel和shared expert选择相同block_m，避免忙rank128k而idle rank按1token选tile拖慢全体；capture与prefill CP不使用此hint。作者MI355X AgentX TP4/c8 interactivity中位+9%，到TP2/c16无可测收益；两项硬件/负载不同，不叠加。建议profile闲忙不均prefill与小batch长context verify。

机构/社区：SGLang community。证据基线：[sgl-project/sglang #45555](https://github.com/sgl-project/sglang/pull/45555)，2026-10-27T23:26:17Z，main @ 2c83dc6656f8；[sgl-project/sglang #46093](https://github.com/sgl-project/sglang/pull/46093)，2026-10-27T22:35:29Z，main @ 7e142dd9fdc4。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`python/sglang/kernels/ops/attention/vattn_asm_gfx950/__init__.py::mtp_verify_attn_seg_max`；`python/sglang/srt/layers/attention/aiter_backend.py::ForwardMetadata`；`python/sglang/srt/layers/moe/mega_moe.py::_mega_moe_global_tokens_hint`。

稳定ID：`github:sgl-project/sglang:pr:45555`；`github:sgl-project/sglang:pr:46093`。

### SGLang AITER routing 保留 FP32 correction bias，EPLB加载每个本地expert副本

FP32 bias时先升logits而不把bias降BF16，radix4与native AITER路径保持相同排序精度；不能恢复上游已舍入的gate logits。EPLB loader另从physical_to_logical全映射找所有副本，避免dispatch候选优化只留首个local replica导致其他slot从未加载，覆盖冷load和reload。bias测试记录为rebase前版本，非最终merge重测。建议带冗余expert的初始化布局检查所有slot已写，跨batch router阈值比较winner。

机构/社区：SGLang community。证据基线：[sgl-project/sglang #44797](https://github.com/sgl-project/sglang/pull/44797)，2026-10-27T23:04:44Z，main @ f9f2665b6e63；[sgl-project/sglang #45363](https://github.com/sgl-project/sglang/pull/45363)，2026-10-27T20:48:19Z，main @ e80120b4fb8f。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`python/sglang/srt/layers/moe/topk.py::biased_grouped_topk_gpu`；`python/sglang/srt/eplb/expert_location.py::logical_to_all_physical`；`python/sglang/srt/layers/moe/fused_moe_triton/layer.py::weight_loader`。

稳定ID：`github:sgl-project/sglang:pr:44797`；`github:sgl-project/sglang:pr:45363`。

### SGLang EPLB 把 remap与统计合为单kernel，但最终版本验证不完整

CUDA decode graph可获得count buffer并捕获fused logical→physical remap+histogram；MegaGate已映射路径用count-only kernel，其他设备保留torch路径。减少原先每层多个elementwise/scatter节点，记录停止后graph仍可能执行计数工作，不等于消除全部EPLB固定成本。作者数值/服务检查与部分性能来自较早revision，最后四commit和ROCm未跑。建议对实际EPLB mode、padding与图replay统计逐轮比对后再启用收益判断。

机构/社区：SGLang community。证据基线：[sgl-project/sglang #46497](https://github.com/sgl-project/sglang/pull/46497)，2026-10-27T22:28:45Z，main @ baa349af7bd8。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`python/sglang/kernels/ops/moe/eplb_map_record.py::_eplb_map_record_hist_kernel`；`python/sglang/srt/eplb/expert_distribution.py::get_current_pass_count_buffer`；`python/sglang/srt/eplb/expert_location_dispatch.py::topk_ids_logical_to_physical_and_record`。

稳定ID：`github:sgl-project/sglang:pr:46497`。

### SGLang Qwen3.5/3.8 GDN 默认在线 MXFP4：速度收益伴随推理长度与更新限制

gfx95x+AITER、对应Qwen模型、未量化且N/K均256整除的BF16 in_proj_qkvz/out_proj在load后在线MXFP4，默认开，SGLANG_ROCM_GDN_ONLINE_MXFP4=0关闭；其余复用GDN的模型不变。作者仅对Qwen3.8-2.4T-A95B MXFP4、8×MI355X TP8的12形状测得TPOT几何均值降4.0%/5.1%（无/有NEXTN），但thinking-on GSM8K8×1319样本回答长10.7%、准确率0.9641→0.9597（p=0.0017），GPQA也有波动。最终在KV预算前量化，旧性能实验两臂KV相同；不支持在线权重更新，RL refit应关闭；Qwen3.5及其他TP未测，旧AITER路径虽正确但性能未测，当前结果依赖含#5998的AITER pin。建议按长推理质量与KV/速度共同验收，不能写无损量化。

机构/社区：SGLang community。证据基线：[sgl-project/sglang #43839](https://github.com/sgl-project/sglang/pull/43839)，2026-10-27T22:27:17Z，main @ 3145c7ae6650。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py::AiterMxfp4OnlineLinearMethod`；`python/sglang/srt/models/qwen3_5.py::_gdn_online_mxfp4`。

稳定ID：`github:sgl-project/sglang:pr:43839`。

### SGLang hybrid MTP 与 DSA padding 修逻辑行数，KDA图强制激活 raw beta

MoE padding后KDA verify只处理metadata逻辑行并补零回物理shape，DSA KPool仅写write-plan覆盖行；TileLang/Triton sparse输出去掉内部singleton batch轴，恢复token-first契约；KDA prefill图明确beta_is_raw=True，避免4D gate被误判跳sigmoid。DP attention padding改重绑host list，保留预先规划metadata与scheduler持有的原始列表，防query/page-table行数分歧。多项验证基于早期等价版本或局部adapter，建议GLM/Kimi在eager/graph、ragged verify与不均匀DP batch交叉回归。

机构/社区：SGLang community。证据基线：[sgl-project/sglang #43271](https://github.com/sgl-project/sglang/pull/43271)，2026-10-27T22:21:37Z，main @ 7f0751994362；[sgl-project/sglang #46424](https://github.com/sgl-project/sglang/pull/46424)，2026-10-27T08:29:53Z，main @ 8db96fd40202；[sgl-project/sglang #44778](https://github.com/sgl-project/sglang/pull/44778)，2026-10-27T09:29:34Z，main @ 3927a73a3407；[sgl-project/sglang #45796](https://github.com/sgl-project/sglang/pull/45796)，2026-10-27T05:04:42Z，main @ 58eb1740d21f。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`python/sglang/srt/layers/attention/linear/kda_backend.py::forward_extend`；`python/sglang/srt/layers/attention/dsa/dsa_indexer_kpool.py::_forward_cuda_target_verify`；`python/sglang/srt/layers/attention/dsa_backend.py::_forward_tilelang`；`python/sglang/srt/layers/attention/linear/kda_prefill_graph.py::kda_extend_in_graph`；`python/sglang/srt/model_executor/forward_batch_info.py::_pad_inputs_to_size`。

稳定ID：`github:sgl-project/sglang:pr:43271`；`github:sgl-project/sglang:pr:46424`；`github:sgl-project/sglang:pr:44778`；`github:sgl-project/sglang:pr:45796`。

### SGLang Qwen megakernel plugin API v2 与可选 host-work 图化同时合入

Qwen3.5 binding把SGLang-owned KV/GDN metadata与output交给外部NVIDIA plugin，EAGLE entry、batch size和KV-length graph variants需跨rank一致，API升2；plugin代码不在此PR。另一独立PR用默认关闭env把post-verify采样/commit、input staging、chain verify与TP同步缩短；grammar、penalties、per-request seed等不满足条件仍eager，DP attention等组合受限。作者GB300 TP8、单请求128K、强制接受4.80的较早组合：仅runtime +6.6%，runtime+plugin整栈+82.8%，绝非此PR真实流量单独收益。建议匹配plugin API、实际接受率和flags逐项做消融；不要直接发布整栈数字。

机构/社区：SGLang community。证据基线：[sgl-project/sglang #45266](https://github.com/sgl-project/sglang/pull/45266)，2026-10-27T21:02:54Z，main @ 8f1b7691caf4；[sgl-project/sglang #45474](https://github.com/sgl-project/sglang/pull/45474)，2026-10-27T21:42:46Z，main @ b74fd392e18b。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`python/sglang/kernels/plugin_kernels/nvidia/spec.py::MEGAKERNEL_API_VERSION`；`python/sglang/srt/kernel_plugins/nvidia/eagle_binding.py`；`python/sglang/srt/speculative/eagle_post_verify_graphs.py::EaglePostVerifyGraphs._key`；`python/sglang/srt/environ.py`。

稳定ID：`github:sgl-project/sglang:pr:45266`；`github:sgl-project/sglang:pr:45474`。

### SGLang QSA CP 直接读cache，并在prefetch源复用前建立stream依赖

prefill CP sparse GQA从paged cache读选中K/V，不打包整段prefix；最终小补丁删除多余metadata解析，并让qsa_cp_kv stream等当前cache-write stream后再复用prefetched K/V。正文宏观“去D2D copy”与本PR最终diff粒度不同，不能把全部先前路径优化算成此次新增。建议关注cached-prefix与prefetch storage寿命，按最终main补丁测增量。

机构/社区：SGLang community。证据基线：[sgl-project/sglang #45753](https://github.com/sgl-project/sglang/pull/45753)，2026-10-27T21:41:15Z，main @ 148bea73026d。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py::_forward_extend_cp`。

稳定ID：`github:sgl-project/sglang:pr:45753`。

### SGLang forward observer 与 deferred response 分开计算完成和外部发布完成

target forward后、sampling前observer可快照device输出；finished/unstreamed请求可交completion延迟响应，由scheduler在busy/idle/paused轮询、调用result完成metadata后仅发送一次，失败/abort退休所有权。共享future不应由单个兄弟请求cancel；plugin必须自行保留数据，graph replay可能覆写的view需复制。当前不支持draft/spec、DLLM、PP>1、MLX、PDMux，deferred response还拒PD。建议SpecForge/异步hidden-state出版插件测“已计算但未发布”、取消与故障，而不把返回future视为已入库。

机构/社区：SGLang community。证据基线：[sgl-project/sglang #43450](https://github.com/sgl-project/sglang/pull/43450)，2026-10-27T20:37:13Z，main @ 4a58dc983825。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`python/sglang/srt/model_executor/model_runner.py::supports_forward_observer`；`python/sglang/srt/model_executor/forward_observer.py::ForwardObserver.after_forward`；`python/sglang/srt/managers/scheduler.py::defer_response、stream_released_deferred_outputs`。

稳定ID：`github:sgl-project/sglang:pr:43450`。

### SGLang paged experts 缩少 host 同步与图后处理节点，captured容量仍有硬上限

host-planned eager把routed counts与LRU state一起读，所有wave slot map/state用pinned非阻塞copy，Marlin用已知entry count的nonzero_static免同步。captured decode则decide kernel直接写每entry slot/weight，padding映射slot0/weight0，原21节点降10；host eager路径不受此融合改变。capture路由entries必须≤K resident slots，否则仍拒绝，不能称任意batch可图执行。建议量化MoE offload同时测多wave与小batch decode，区分GPU residency状态与host planner成本。

机构/社区：SGLang community。证据基线：[sgl-project/sglang #46567](https://github.com/sgl-project/sglang/pull/46567)，2026-10-27T19:36:12Z，main @ a4a19332abd9；[sgl-project/sglang #46457](https://github.com/sgl-project/sglang/pull/46457)，2026-10-27T10:38:59Z，main @ 80632965286d。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`python/sglang/srt/layers/moe/paged_experts/executor.py::run、_run`；`python/sglang/kernels/jit/csrc/moe/paged_experts.cuh::paged_experts_decide_kernel`。

稳定ID：`github:sgl-project/sglang:pr:46567`；`github:sgl-project/sglang:pr:46457`。

### SGLang DFlash/DSpark 可服务 SpecForge MLA draft，沿target MLA backend

DFlashMLAAttention复用DeepSeek absorbed MLA、按SpecForge配置合并down projection和吸收kv_b，target context用latent+RoPE KV而非GQA cache；DFlash/DFlash2/Domino/DSpark共用masked accepted-prefix写入。CUDA默认FlashInfer、ROCm/XPU默认Triton，MLA draft遇trtllm_mha改FlashInfer，因为它只服务MHA/GQA；backend决策提前给KV planner。建议MLA draft训练/部署对齐projection、causality与accepted-prefix commit，后端覆盖不代表所有模型质量已验证。

机构/社区：SGLang community。证据基线：[sgl-project/sglang #44676](https://github.com/sgl-project/sglang/pull/44676)，2026-10-27T19:14:04Z，main @ b068417bfcc1。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`python/sglang/srt/models/dflash.py::DFlashMLAAttention`；`python/sglang/srt/arg_groups/speculative_hook.py::_keep_mla_draft_off_trtllm_mha`；`python/sglang/srt/mem_cache/memory_pool.py::set_mla_kv_buffer_prefix_valid`。

稳定ID：`github:sgl-project/sglang:pr:44676`。

### SGLang SWA tombstone 恢复不越过protected prefix，防canonical页被错误释放

恢复window统一从max(eviction cursor, prev_prefix_len)开始，避免把请求仅借用的canonical FULL页当自有重复页free后、node仍引用旧ID，后续allocation覆写其他请求KV。单chunk cursor0或protected prefix落在node内部尤其危险，独立window外原先缺guard。建议混合FULL/SWA缓存压测locked/unlocked tombstone、prefix内部边界与再次分配，优先验证跨请求隔离。

机构/社区：SGLang community。证据基线：[sgl-project/sglang #46700](https://github.com/sgl-project/sglang/pull/46700)，2026-10-27T18:46:41Z，main @ e7c55f69597c。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`python/sglang/srt/mem_cache/unified_cache/components/swa.py::update_component_on_insert_overlap`。

稳定ID：`github:sgl-project/sglang:pr:46700`。

### SGLang HiCache load-back批次暂停cached-prefix early send，host-receive恢复可异步准入

P侧若本batch build提交HiCache load-back，整批跳early-send，等forward逐层等待H2D后再send，避免页被重匹配成device hit但尚未写完；作者100K prefix复现约晚54ms发送，是正确性换overlap。D侧host-received KV提前start restore，按event.query与attention TP/CP CPU MIN collective一致准入，free/reset登记event保护；PP不能在此位置collective，仍同步等待。建议P/D两侧分别测checksum、取消与reset，不能把P侧gap与D侧异步改进互相视为闭环。

机构/社区：SGLang community。证据基线：[sgl-project/sglang #44081](https://github.com/sgl-project/sglang/pull/44081)，2026-10-27T08:42:35Z，main @ 2a7987f9009d；[sgl-project/sglang #46689](https://github.com/sgl-project/sglang/pull/46689)，2026-10-27T18:46:34Z，main @ b28f4fc06586。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`python/sglang/srt/managers/scheduler.py::run_batch`；`python/sglang/srt/disaggregation/prefill.py::maybe_send_cached_prefix_chunk`；`python/sglang/srt/disaggregation/decode.py::_start_host_restores、_host_restores_ready`；`python/sglang/srt/mem_cache/common.py::start_restore_kv_cache、finish_restore_kv_cache`。

稳定ID：`github:sgl-project/sglang:pr:44081`；`github:sgl-project/sglang:pr:46689`。

### SGLang DeepEP v2 masked GEMM slab 按当前图batch上界缩容，DSv4.1 KV容量按真实layout计

DeepEP v2仅在EP覆盖完整TP且无attention DP/CP、同一device group时，用含uneven shard/graph padding/alignment的最坏source-token总数缩小每expert masked slab；通信buffer不变，其他拓扑回旧容量。DSv4.1 ratio1/2 KV计费改layout-aware row bytes，main/indexer保留页和rounding单独扣，再按完整page降容量；不改物理layout或kernel。后一PR最终简化测试因缺依赖未重跑，模型质量未证。建议用真实pool bytes核预算，再压力测最坏路由；节省不能外推所有拓扑。

机构/社区：SGLang community。证据基线：[sgl-project/sglang #46309](https://github.com/sgl-project/sglang/pull/46309)，2026-10-27T08:30:18Z，main @ a09c1a096c6b；[sgl-project/sglang #41275](https://github.com/sgl-project/sglang/pull/41275)，2026-10-27T02:11:23Z，main @ 20ff40cb8409。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`python/sglang/srt/layers/moe/token_dispatcher/deepep_v2.py::_masked_token_capacity`；`python/sglang/srt/model_executor/pool_configurator.py::_compressed_bytes_per_full_token、_get_low_ratio_kv_padding_bytes`。

稳定ID：`github:sgl-project/sglang:pr:46309`；`github:sgl-project/sglang:pr:41275`。

### SGLang GLM DSA/MTP KVShard 接通P/D与CPU HiCache，支持矩阵严格限定

page-owner映射贯通target/draft latent KV、indexer keys/scales、Mooncake sender completion及native host pool；main最终diff聚合#43778实现，不能因依赖PR正文仍写unmerged误判main缺实现。启动要求CUDA、interleave prefill CP>1、PP1、TRTLLM prefill/decode、FP8 E4M3/page64、Mooncake、未压缩index_kpool1且不开cache layer split；默认选Python树，Rust另属#46445。建议先按此矩阵做owner-local H2D、empty final chunk与retraction回归，不宣称通用KVShard。

机构/社区：SGLang community。证据基线：[sgl-project/sglang #42973](https://github.com/sgl-project/sglang/pull/42973)，2026-10-27T07:55:03Z，main @ 8136cbce3dc6。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`python/sglang/srt/arg_groups/kv_shard_hook.py::validate_kv_shard_attention_backend`；`python/sglang/srt/disaggregation/mooncake/conn.py`；`python/sglang/srt/mem_cache/page_interleave_pool.py`；`test/registered/e2e/models/test_dsa_glm52_pd_kv_shard_hicache.py`。

稳定ID：`github:sgl-project/sglang:pr:42973`。

### SGLang checkpoint loader 删除不安全pickle fallback，旧tar需先转换

语言与多模态_load_pt_file不再遇legacy tar就转weights_only=False，其他OCR/Roberta/RIFE模型权重load也明确weights_only=True；恶意pickle测试确保不执行payload。旧tar格式现在直接报Torch错误，应离线转safetensors或zip .pt，开发者tensor dump调试路径不在此次范围。建议模型分发/部署先检查旧artifact格式，此变更不涉及forward精度或性能。

机构/社区：SGLang community。证据基线：[sgl-project/sglang #42760](https://github.com/sgl-project/sglang/pull/42760)，2026-10-27T07:51:47Z，main @ f82f8a94b5d2。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`python/sglang/srt/model_loader/weight_utils.py::_load_pt_file`；`python/sglang/multimodal_gen/runtime/loader/weight_utils.py::_load_pt_file`。

稳定ID：`github:sgl-project/sglang:pr:42760`。

### SGLang fmha_v2 直接读取连续K/V池，并统一FlashAttention选中kernel的调用

TRTLLM SM90/SM120 prefill用同allocation的连续、block对齐K/V halves及int32 offsets构造paged输入，避免重拷interleaved KV；不满足连续/offset约束仍copy并警告。FlashAttention另把14处module wrapper改到实例绑定，使SM12x FA4 decode静态max_seqlen_k与subclass选择真正贯通，SM120 wrapper补mask_mod。建议Hopper/消费Blackwell查实际dispatch与pool布局，不将选中FA4等同所有路径原先都已调用它；外部FlashInfer/FA内部不在本轮审计。

机构/社区：SGLang community。证据基线：[sgl-project/sglang #32272](https://github.com/sgl-project/sglang/pull/32272)，2026-10-27T07:35:50Z，main @ 1a120754e00b；[sgl-project/sglang #46695](https://github.com/sgl-project/sglang/pull/46695)，2026-10-28T00:11:58Z，main @ 586355a83bdf。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`python/sglang/srt/layers/attention/trtllm_mha_backend.py::_fmha_v2_paged_inputs`；`python/sglang/srt/mem_cache/memory_pool.py`；`python/sglang/srt/layers/attention/flashattention_backend.py`；`python/sglang/kernels/ops/attention/flash_attention_v4_sm120.py`。

稳定ID：`github:sgl-project/sglang:pr:32272`；`github:sgl-project/sglang:pr:46695`。

### SGLang AITER FP8 KV 使用独立Q与层级K/V scale，gfx942 CK GEMM改默认开启

AITER MHA decode传layer k/v scales，NHD prefill与unified decode用unit Q scale量化并传正确q_descale，避免校准K scale被误作Q scale导致garbage；无KV scale时旧bug被1掩盖，饱和quant也避免e4m3fnuz越界NaN。gfx942 dense block-FP8 CK GEMM另默认开，=0可退Triton，作者MI325X8-shot GSM8K/MMLU中Qwen3-32B仍约1点GSM差，不能写各模型无损；4×速度来自早先kernel路径工作。建议校准scale模型做端到端质量回归，并固定eval设置比较新默认。

机构/社区：SGLang community。证据基线：[sgl-project/sglang #46528](https://github.com/sgl-project/sglang/pull/46528)，2026-10-27T06:31:49Z，main @ 2b36c718cb27；[sgl-project/sglang #43815](https://github.com/sgl-project/sglang/pull/43815)，2026-10-27T10:58:11Z，main @ a6f5bd397be4。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`python/sglang/srt/layers/attention/aiter_backend.py::_quantize_q_fp8、forward_decode、forward_extend`；`python/sglang/srt/environ.py::SGLANG_AITER_GFX942_BLOCKSCALE_USE_CK`。

稳定ID：`github:sgl-project/sglang:pr:46528`；`github:sgl-project/sglang:pr:43815`。

### SGLang BF16 TRT-LLM MoE refit 先恢复layout和值，再单次重新pack

cold-load把expert变BlockMajorK后，routed与非routed两路径都要逆变换；现恢复canonical shape和值且保storage指针，校验permutation双射与元素数。begin_weight_update覆盖绕过loader的tensor/P2P写，disk reload仍由loader恢复，end session仅pack一次，避免多bucket反复pack及未更新expert静默乱序。建议RL反复refit测试部分更新、三轮更新与LoRA-wrapped traversal；该修复是BF16路径，不关闭旧NVFP4 refit gap。

机构/社区：SGLang community。证据基线：[sgl-project/sglang #33743](https://github.com/sgl-project/sglang/pull/33743)，2026-10-27T07:34:09Z，main @ 8353f86f0bb4。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`python/sglang/srt/layers/quantization/unquant.py::UnquantizedFusedMoEMethod.restore_weights_before_loading`；`python/sglang/srt/model_executor/model_runner_components/weight_updater.py`。

稳定ID：`github:sgl-project/sglang:pr:33743`。

### SGLang HiCache 可选并行THP host allocation，fork语义显式改变

SGLANG_HICACHE_HOST_PARALLEL_ALLOC_THREADS>0启用private anonymous mapping、MADV_HUGEPAGE/DONTFORK与多线程prefault，再pin；失败退原shared映射，显式hugetlb不变，默认关闭。作者4×B300同时每rank306GB host pool，32线程最慢rank总allocation+register185–189→46秒，两次运行；收益是启动，不是推理。DONTFORK意味着fork子进程不能访问此buffer（测试会SIGSEGV），weight-checker arena也受flag影响。建议启动瓶颈团队先确认进程模型/NUMA/THP实际情况，再启用。

机构/社区：SGLang community。证据基线：[sgl-project/sglang #43978](https://github.com/sgl-project/sglang/pull/43978)，2026-10-27T04:29:17Z，main @ 75468e4b4f24。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`python/sglang/srt/mem_cache/storage/mmap/mmap_allocator.py::_mmap_private_parallel_prefaulted、alloc_mmap`。

稳定ID：`github:sgl-project/sglang:pr:43978`。

### SGLang Qwen3-Next gfx950融合QK norm/RoPE，ROCm profiler清继承注册

Qwen3-Next全attention的Q/gate split、Gemma RMSNorm与partial NeoX RoPE用AITER fused op，限HIP+AITER+gfx95，import/shape不支持则native fallback；作者MI355X TP4、Triton3.8 nightly报告TPOT降2.3–3.0%，不能外推其他架构。另在import torch前清理子进程继承的ROCPROFILER_REGISTER_LIBRARY claim，修Torch2.12/ROCm10.1子进程GPU trace空但无错误；本进程已加载SDK等guard仍需保留。建议性能排查先确认trace确有GPU事件再比较fusion。

机构/社区：SGLang community。证据基线：[sgl-project/sglang #45739](https://github.com/sgl-project/sglang/pull/45739)，2026-10-27T05:15:53Z，main @ db90f584d2f4；[sgl-project/sglang #44292](https://github.com/sgl-project/sglang/pull/44292)，2026-10-28T00:41:42Z，main @ 291f38c390f8。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`python/sglang/srt/models/qwen3_next.py::_fused_attention_prologue、forward_prepare_fused`；`python/sglang/srt/debug_utils/rocprofiler_register.py::clear_inherited_rocprofiler_registration`；`python/sglang/__init__.py`。

稳定ID：`github:sgl-project/sglang:pr:45739`；`github:sgl-project/sglang:pr:44292`。

### SGLang Flint 架构/解析器接入main；XPU镜像升级是依赖pin

新增Flint config/model与reasoning/tool parser，启动明确要求attention-sink支持backend，MoE runner限auto/triton/flashinfer_trtllm_routed，并拒MoE a2a。PR正文无实质accuracy/perf结果，不能称RecursiveSI全模型质量已验证。XPU独立将镜像base设oneAPI2026.1.3、kernel wheel设0.5.0，Torch不变；PR CI拉预建nightly而不build该Dockerfile，不能据CI通过称新镜像已验证。建议Flint先测权重/特殊token/attention sinks，XPU切镜像先build与smoke。

机构/社区：SGLang community。证据基线：[sgl-project/sglang #45624](https://github.com/sgl-project/sglang/pull/45624)，2026-10-27T19:21:11Z，main @ 41fb719cf259；[sgl-project/sglang #43989](https://github.com/sgl-project/sglang/pull/43989)，2026-10-27T06:31:30Z，main @ cc96d552c3c8。均为main合入代码，尚无本窗新增公开Release。

源码阅读路线：`python/sglang/srt/arg_groups/model_overrides/flint.py::_flint_overrides`；`python/sglang/srt/models/flint.py`；`docker/xpu.Dockerfile`；`python/pyproject_xpu.toml`。

稳定ID：`github:sgl-project/sglang:pr:45624`；`github:sgl-project/sglang:pr:43989`。

### 非main合入与旧开放依赖：保留隔离边界

SGLang #46894仅合入sglang-miles：修LoRA wrapping后.base_layer名字使MoE误分配dense buffer；#46682也仅sglang-miles：把multimem TP跨节点probe从constructor推迟到首个eager build，方便mock-group权重副本构造；#42092仅amd_march：RDNA默认Triton/关闭Lean自动选择，最终patch按is_rdna gate，正文gfx1100/gfx1201枚举不能扩成main已生效。三者均未核对main counterpart，不作已发布能力。

vLLM #63232 MiniMax-M3.1-Flash仍open，head已变a0ea94109745；190文件已两页归档，只选择性看配置/关键patch，不将其叙述为main集成。SGLang #43299仍open且head更新b5d07691696a：候选修非CUDA early-send event及NIXL丢wait_event（含transfer worker绑定正确device后、staging gather前等待）；它与本窗main #44081的HiCache load-back门控是不同race，后者不关闭前者。vLLM #61667 late-WRITE slot隔离仍open且head未变；#61978仍closed-unmerged，需跟踪后继。verl #8061、SGLang #45580仍open；Megatron HDO #4046参数身份映射与#7785 GPU Adam step恢复两半均open，不因verl昨日onload修复就称resume已闭环。vLLM #62129已于10-18合main，本轮只关闭Qwen PP文中该旧依赖的状态疑问，不重新计新事件。

建议优先级：低精度训练先处理Megatron缩放/参数元数据；服务先回归HiSparse、CoW、SWA、校准FP8 scale与PD完成顺序；RL在线更新先关闭不支持refit的GDN在线MXFP4默认。保留未选主窗候选与历史间隔，下一轮从core-pending-candidates.json继续语义分拣。

分支一手证据：[sgl-project/sglang #46894](https://github.com/sgl-project/sglang/pull/46894)，base=sglang-miles，2026-10-28T00:47:07Z，9b5761445381；[sgl-project/sglang #46682](https://github.com/sgl-project/sglang/pull/46682)，base=sglang-miles，2026-10-27T22:10:42Z，82bb25e09bcf；[sgl-project/sglang #42092](https://github.com/sgl-project/sglang/pull/42092)，base=amd_march，2026-10-27T17:04:34Z，8176c5a6ea2e。

开放/依赖一手证据：[verl-project/verl #8061](https://github.com/verl-project/verl/pull/8061)（open，merged_at=None，head 2c26090b2f6c）；[vllm-project/vllm #61667](https://github.com/vllm-project/vllm/pull/61667)（open，merged_at=None，head 4fd16ddb155f）；[vllm-project/vllm #61978](https://github.com/vllm-project/vllm/pull/61978)（closed，merged_at=None，head 00fe254cd01b）；[vllm-project/vllm #63232](https://github.com/vllm-project/vllm/pull/63232)（open，merged_at=None，head a0ea94109745）；[sgl-project/sglang #43299](https://github.com/sgl-project/sglang/pull/43299)（open，merged_at=None，head b5d07691696a）；[sgl-project/sglang #45580](https://github.com/sgl-project/sglang/pull/45580)（open，merged_at=None，head 2723bea86913）；[NVIDIA/Megatron-LM #4046](https://github.com/NVIDIA/Megatron-LM/pull/4046)（open，merged_at=None，head a1a6ac334845）；[NVIDIA/Megatron-LM #7785](https://github.com/NVIDIA/Megatron-LM/pull/7785)（open，merged_at=None，head ccc454f5cc4c）；[vllm-project/vllm #62129](https://github.com/vllm-project/vllm/pull/62129)（closed，merged_at=2026-10-18T06:51:33Z，head 0d32cddfc6e8）。

<a id="extended"></a>

## 扩展开源工程与编译栈

> 以下日期保留UTC合入时间。每组列出最终PR基线；PyTorch按ghstack导出的实际默认分支commit与落地时间记述；PR的merged_at未登记时不推定其PR状态。除单列release，合入不表示已进入安装包；本轮未运行这些项目的GPU/NPU/TPU测试。

### Triton 3.9.0 正式发布：原子语义与低精度边界值得迁移前验证

**triton-lang/triton；2026-10-27T18:56:21Z；正式tag `v3.9.0`、目标 `release/3.9.x`。** [官方release](https://github.com/triton-lang/triton/releases/tag/v3.9.0)。

release汇集tl.atomic_load/store（默认acquire/release）与atomic_poll、近似FP32除法、Gluon/后端支持与正确性修复；RMW/CAS默认acq_rel及相应CTA barrier会影响原先依赖隐式relaxed的kernel成本。E8M0的0x00改解2^-127、0xFF改NaN，旧zero/inf口径的测试需要重建。multi-CTA同步/proxy fence及非128字节对齐TMA等修正有LLM kernel意义，但release中局部最高加速缺本文同等负载，不直接复用为模型收益。**建议深入阅读并考虑采用**，先核Breaking Changes、依赖兼容与数值基线；本窗main的新PR不能凭日期推定已进tag。

### TRL：分块 log-prob / KL / JSD 修正高置信度小梯度

**huggingface/trl；合入记录与基线：** [#7882](https://github.com/huggingface/trl/pull/7882)（2026-10-27T17:47:25Z，默认分支@[c886cf23](https://github.com/huggingface/trl/commit/c886cf23d74acd8dfed95b63943a8dba0f25d85f)）。

分块头改用排除目标项的指数和、补偿减法及 log1p/expm1 等稳定计算，避免目标概率接近1时梯度被相消，也让相同师生的散度梯度归零；分块投影仍不物化全词表 logits。作者报告74项新增数值回归在CUDA通过，RTX3090Ti仍有一项既有BF16/温度0.7失败；ROCm头部微测中KL额外开销约2%（32K词表）到24%（128K），JSD也并非总更快，未跑完整多GPU训练。FSDP2输出cast是Accelerate的独立问题。**建议考虑采用并复测**高置信度RL/蒸馏的梯度，不能将数值修正当免费提速。

源码入口：`trl/kernels/chunked_divergence.py`。

### TRL：Gated DeltaNet 树打包同时隔离卷积与递归状态

**huggingface/trl；合入记录与基线：** [#7686](https://github.com/huggingface/trl/pull/7686)（2026-10-27T09:40:22Z，默认分支@[b73853d2](https://github.com/huggingface/trl/commit/b73853d2a5f34a3d7bbb6b7bfafb4c0d2b0dd063)）。

AsyncGRPO树打包新增Qwen3.5/3.6、Qwen3-Next、OLMo-Hybrid：段从父段final state起步，反向把子段初态梯度相加；宽4因果卷积补上根路径前三个token。Mamba和LFM2等其他非attention层仍拒绝；linear_attention要求Transformers≥5.15.0，且须找到目标卷积/递归函数才能patch。作者用1×H100 80GB、BF16、FA3、梯度检查点测0.8B/2B真实优化步，但rollout为预置随机token、不计生成；旧序列打包基线存在跨样本状态泄漏，其耗时只能作计时参照，不能当正确训练效果对照。**建议深入阅读并复现**独立样本/共享前缀的梯度等价性，再决定吞吐收益。

源码入口：`trl/experimental/async_grpo/async_grpo_trainer.py`。

### TRL：空 completion 与失败 rollout 的训练边界

**huggingface/trl；合入记录与基线：** [#7913](https://github.com/huggingface/trl/pull/7913)（2026-10-27T11:47:15Z，默认分支@[33b871b5](https://github.com/huggingface/trl/commit/33b871b5eee4d7a51dcc42c1b5454a4d0364339a)）；[#7849](https://github.com/huggingface/trl/pull/7849)（2026-10-27T16:47:15Z，默认分支@[b1e43f49](https://github.com/huggingface/trl/commit/b1e43f497378f2986501ab398783eb44024830a0)）。

ORPO/CPO在数据预处理丢弃chosen或rejected被截成零个监督token的行，避免0/0使权重NaN或把空completion当log-prob0；应重新统计过滤后训练/评估集规模。Harbor AsyncGRPO把EOS IDs传给worker；仅失败会话且capture不可解析时返回不可评分空rollout，成功会话的坏capture仍报错。两者解决不同故障，不能据此推定所有失败轨迹都安全可训。**建议考虑采用**并记录被过滤/不可评分占比；本轮仅核代码与回归用例，未跑完整训练。

源码入口：`trl/experimental/cpo/cpo_trainer.py`；`examples/async_grpo_harbor/async_grpo_harbor.py`。

### DeepSpeed：跳过 Adam 状态也恢复检查点中的 FP32 master

**deepspeedai/DeepSpeed；合入记录与基线：** [#8891](https://github.com/deepspeedai/DeepSpeed/pull/8891)（2026-10-27T14:52:11Z，默认分支@[2dd22dc1](https://github.com/deepspeedai/DeepSpeed/commit/2dd22dc1493c66f8c8f40f2b0b67521a07a7b26f)）。

ZeRO-1/2过去在load_optimizer_states=False时从BF16/FP16模型重建master，丢失保存精度；现在非module-only加载保留master并重置moments。load_module_only仍是warm start，ZeRO-3权重分片语义另有约束；旧load_fp32_weights接口移除、zero配置项弃用。刚性ZeRO-1/2或ZeRO-3不能据此承诺任意DP规模迁移，缺失master文件和alignment不一致会明确报错。**建议优先核查恢复流程**：这是训练轨迹正确性和接口行为变化，需要用实际检查点验证。

源码入口：`deepspeed/checkpoint/universal_checkpoint.py`。

### DeepSpeed：通信日志不再制造逐步增长的测量与同步开销

**deepspeedai/DeepSpeed；合入记录与基线：** [#8795](https://github.com/deepspeedai/DeepSpeed/pull/8795)（2026-10-27T17:03:16Z，默认分支@[b5829596](https://github.com/deepspeedai/DeepSpeed/commit/b5829596d6230275b7464ecf5a704b9b5a6ceccd)）；[#8909](https://github.com/deepspeedai/DeepSpeed/pull/8909)（2026-10-27T17:40:04Z，默认分支@[322283b2](https://github.com/deepspeedai/DeepSpeed/commit/322283b2509f8fbf6bbf2bc1306f749f9d7c6a41)）。

timed_op只为被prof_ops/prof_all/prof选中的通信同步；另将反复扫描全部历史event改为复用event对并缓存区间读数，elapsed返回最新区间、elapsed_total返回累计值、mean从10%截尾均值改为普通均值；enabled=false也恢复有效。前一PR的52项测试使用mock后端，不能当NCCL/MPI实机速度证明。**建议考虑采用**，并重建日志统计基线：新的均值口径与旧报表不可直接拼接。

源码入口：`deepspeed/comm/comm.py`。

### DeepSpeed：ZeRO stage 0 的 loss-scale hysteresis 开始实际生效

**deepspeedai/DeepSpeed；合入记录与基线：** [#8647](https://github.com/deepspeedai/DeepSpeed/pull/8647)（2026-10-27T01:27:02Z，默认分支@[62656fe7](https://github.com/deepspeedai/DeepSpeed/commit/62656fe76e42f7866b8fa3bb6e0867f35de415a6)）。

fused/unfused FP16 optimizer接通delayed_shift和consecutive_hysteresis并保存剩余计数，默认hysteresis=2由此前被忽略转为生效。作者给出hysteresis=3、初始scale256的四次overflow序列从128/64/32/16变为256/256/128/64；验证为M2 Pro CPU的6项用例，GPU门控测试未跑，且未修复另一个stage0 loss未缩放问题。**建议考虑采用**并核对恢复后的scale轨迹，不能概括为FP16训练问题全部解决。

源码入口：`deepspeed/runtime/fp16/fused_optimizer.py`。

### DeepSpeed AutoEP：clamped SwiGLU 落地融合前反向

**deepspeedai/DeepSpeed；合入记录与基线：** [#8652](https://github.com/deepspeedai/DeepSpeed/pull/8652)（2026-10-27T07:30:28Z，默认分支@[7501610e](https://github.com/deepspeedai/DeepSpeed/commit/7501610ee00daa3400368cc06a54f81f12822845)）。

为DeepSeek-V4式swiglu_clamped和GPT-OSS/MiniMax式swiglu_oai补上Triton融合，分别支持独立gate/up和packed投影；FP32算术、写回一次舍入，保留clamp等号处梯度。作者在1×H200、torch2.8.0+cu126、Triton3.4.0，BF16、1,045,081×3072的单激活前反向中，swiglu_oai separate额外峰值41.86→17.94GiB、77.8→11.9ms；extra peak排除输入与upstream gradient，输入为6倍标准正态、alpha1.702/limit7.0。这是失衡rank的算子微测，不是全模型训练6.5倍。packed仍比普通SwiGLU慢，完整MiniMax训练及空专家的torch grouped-mm问题不因此消失。**建议考虑复现**MoE失衡下的激活峰值，部署前确认实际Triton/torch版本与higher-order梯度限制。

源码入口：`deepspeed/moe/ep_experts.py`。

### TransformerEngine：row-only MXFP8 master 回写支持缺省列方向

**NVIDIA/TransformerEngine；合入记录与基线：** [#3488](https://github.com/NVIDIA/TransformerEngine/pull/3488)（2026-10-27T05:51:51Z，默认分支@[38509867](https://github.com/NVIDIA/TransformerEngine/commit/38509867458c686b3ebc3af9a713c9fec7b1ad7b)）。

只打包存在方向的amax，仍用一次MAX all-reduce，不增加高精度重建collective；缺省columnwise数据/scale采用空native handle，兼容零元素但有底层storage的PyTorch view，混合row-only与双向批次和FSDP fragment输出有覆盖。作者在GB200/CUDA13.3源码构建验证1/2/4 rank；双向kernel旧新32组微测比值0.994–1.020，排除了初稿21–35%回归，但不证明端到端训练提速。**建议考虑采用**，仍需与Megatron独立copy-back、checkpoint/reshard整体验证。

源码入口：`transformer_engine/common/include/transformer_engine/recipe.h`。

### TransformerEngine：动态专家路由的 NVFP4 grouped cast 可图重放

**NVIDIA/TransformerEngine；合入记录与基线：** [#3649](https://github.com/NVIDIA/TransformerEngine/pull/3649)（2026-10-27T06:17:17Z，默认分支@[89668d9d](https://github.com/NVIDIA/TransformerEngine/commit/89668d9dee82aba9b9107b37ed85227542a578d2)）。

将逐expert量化循环改为packed buffer上的grouped amax与grouped cast两个launch；图安全变体直接读device first_dims/tensor_offsets，让同一capture随路由改变重放，host-split与图路径共用device实现。不是将amax和cast合成一个launch。最终guard限Blackwell（SM10.0+）、BF16，并拒绝RHT、随机舍入、2D量化、amax-reduction及4over6；PR清单并不提供完整训练收益。**建议深入阅读并复现**空expert、路由变化和graph replay逐字节一致性，再评估launch节约。

源码入口：`transformer_engine/common/cast/cast_grouped.cu`。

### TransformerEngine：saved-tensor hooks 不再抹掉 DistributedWeight 身份

**NVIDIA/TransformerEngine；合入记录与基线：** [#3730](https://github.com/NVIDIA/TransformerEngine/pull/3730)（2026-10-27T18:37:32Z，默认分支@[fa4b30f0](https://github.com/NVIDIA/TransformerEngine/commit/fa4b30f09b95031931f3f9378e7d30c2071cb2bb)）。

QuantizedTensor保存/恢复层集中保留DistributedWeight原对象引用，同时继续经autograd保存以保留原地修改检查；激活offload的unpack可能只返回普通tensor，单靠其结果会丢失分布式权重行为。torch.compiler.is_compiling()下该DistributedWeight保存路径仍明确RuntimeError，不能扩展为torch.compile也受支持。回归覆盖相关模块，当前来源没有可泛化的CPU开销数字。**建议考虑采用**并把offload、量化权重和反向组合纳入实际训练回归；这是对象语义修复。

源码入口：`transformer_engine/pytorch/quantized_tensor.py`。

### TransformerEngine JAX MoE：外部路由与 expert count 成为显式契约

**NVIDIA/TransformerEngine；合入记录与基线：** [#3596](https://github.com/NVIDIA/TransformerEngine/pull/3596)（2026-10-27T15:10:34Z，默认分支@[83d8c4e3](https://github.com/NVIDIA/TransformerEngine/commit/83d8c4e3370bc0117189adbbd88a50fd57440acd)）。

最终patch以RouterComputationInfo/ RoutingMapInfo区分内部gate与外部expert IDs/weights；外部权重仍可反传，expert索引不代表可微。collect_expert_counts默认True，控制全局计数与额外all-reduce，不消费时可关闭并返回零计数；Flax返回结构也改变。PR早期正文的moe_from_routing命名不能替代最终公开类型。**建议深入阅读**hash routing、无辅助loss的bias更新集成，先核接口返回值与通信成本；没有完整DSv4训练结果。

源码入口：`transformer_engine/jax/flax/moe.py`。

### TransformerEngine：大路由矩阵与进程退出的两类稳定性修复

**NVIDIA/TransformerEngine；合入记录与基线：** [#3569](https://github.com/NVIDIA/TransformerEngine/pull/3569)（2026-10-27T21:30:00Z，默认分支@[268fdd54](https://github.com/NVIDIA/TransformerEngine/commit/268fdd54d39ba401d69a2fc3fe1ccd37565aa2fb)）；[#3733](https://github.com/NVIDIA/TransformerEngine/pull/3733)（2026-10-27T17:16:06Z，默认分支@[dd628d3a](https://github.com/NVIDIA/TransformerEngine/commit/dd628d3a11b7b52908ea5b7bba7d38d739df6f55)）。

fused MoE auxiliary loss三条launcher以64位乘积拒绝超过INT_MAX的元素量，循环和grid算术改64位，flattened offset仍32位；不能称任意大tensor已支持。作者仅以stub headers编译真实kernel做边界实机检查，未有完整TE构建。另将NVRTC/cuDNN graph静态cache改为进程终止时不析构，避免CUDA已不可用后释放导致139退出。**建议考虑采用**并测试大规模路由与干净退出；后者选择进程寿命缓存，不能等同运行中资源回收优化。

源码入口：`transformer_engine/common/fused_router/fused_moe_aux_loss.cu`；`transformer_engine/common/fused_attn/graph_cache.h`。

### Transformers：仅量化 KV cache 的 checkpoint 恢复可加载

**huggingface/transformers；合入记录与基线：** [#50030](https://github.com/huggingface/transformers/pull/50030)（2026-10-27T16:06:22Z，默认分支@[455c9a04](https://github.com/huggingface/transformers/commit/455c9a0491e3f4b8438186ea15ed0206c9a6da46)）。

compressed-tensors不再无条件取config_groups[0]：只配置KV量化、无权重量化group的checkpoint可正常判别。新增end-to-end加载回归，改动不新增量化算法，也不证明任何质量或速度提升。**建议考虑采用**于KV-only压缩的导出/加载链路，并确认使用的包实际包含此合入。

源码入口：`src/transformers/quantizers/quantizer_compressed_tensors.py`。

### Transformers：保存键名与 remote-code revision 的可逆性

**huggingface/transformers；合入记录与基线：** [#49772](https://github.com/huggingface/transformers/pull/49772)（2026-10-27T10:38:51Z，默认分支@[9ed172cd](https://github.com/huggingface/transformers/commit/9ed172cd9167c8364c41aa28ddf85da611b856be)）；[#49786](https://github.com/huggingface/transformers/pull/49786)（2026-10-27T11:26:32Z，默认分支@[820b1538](https://github.com/huggingface/transformers/commit/820b15380937a71486815cf786a3503e9b73df05)）；[#49832](https://github.com/huggingface/transformers/pull/49832)（2026-10-27T12:09:41Z，默认分支@[0b6c4d4c](https://github.com/huggingface/transformers/commit/0b6c4d4ca11851f810385960f7f15a25aba4e91b)）。

MiniMax-M3-VL的反向重命名收窄到MoE子模块，避免save_pretrained把dense/vision MLP也写成expert键；转换pattern保留捕获组内转义点。pipeline的code_revision则统一传给config/model/tokenizer/processor，移除重复kwargs，并避免污染普通tokenizer保存配置。**建议考虑采用**并做加载→保存→重新加载的键名/固定revision核验；这些修复在v5.21 release之后，不能自动算进该已发包。

源码入口：`src/transformers/core_model_loading.py`；`src/transformers/conversion_mapping.py`；`src/transformers/models/auto/tokenization_auto.py`。

### TensorRT-LLM：Helix + attention-DP 的计算切片与转移共识

**NVIDIA/TensorRT-LLM；合入记录与基线：** [#20597](https://github.com/NVIDIA/TensorRT-LLM/pull/20597)（2026-10-28T00:36:04Z，默认分支@[8786940e](https://github.com/NVIDIA/TensorRT-LLM/commit/8786940e8b74d620a0b972ac20fc78cbf6a184a7)）；[#19575](https://github.com/NVIDIA/TensorRT-LLM/pull/19575)（2026-10-27T04:04:35Z，默认分支@[94348c4d](https://github.com/NVIDIA/TensorRT-LLM/commit/94348c4de6eb5526b0743b9e1c365506eb70868c)）。

Kimi K3 helix_a2a在tp_size=1下切MoE及残差行，attention仍读完整行，并对padded batch、层序及PP组合设限制；Python transceiver的generation共识纳入CP rank，防止只有部分rank收到KV就推进；共识逻辑虽含PP分支，manager仍拒绝attention-DP+Helix+PP>1。开发分支曾报告batch64步时下降5.6%，来源明确未在最终合入分支做同样层级/端到端计时，因此不将它写成当前main收益。**建议深入阅读并复现**CP rank准入、完成与取消的一致性。

源码入口：`tensorrt_llm/_torch/disaggregation/native/mixers/ssm/peer.py`；`tensorrt_llm/_torch/disaggregation/orchestration/coordinator.py`。

### TensorRT-LLM：aggregated Helix 的 KV 分配先行落地，默认仍关闭

**NVIDIA/TensorRT-LLM；合入记录与基线：** [#20759](https://github.com/NVIDIA/TensorRT-LLM/pull/20759)（2026-10-27T06:30:24Z，默认分支@[4b027a38](https://github.com/NVIDIA/TensorRT-LLM/commit/4b027a381d35b28bd717dac67e6d7d5cd6e78fc1)）。

每个CP rank保留完整prompt，以全局ledger block分配prefill KV；要求V2 manager，跳过profiling时采用保守0.6可用显存fraction。来源定位为分阶段基础工作，未完成端到端验证，因此不把它视为聚合Helix全面可用。**建议继续观察**集成验证与后续preemption；对长上下文服务可先读设计及容量约束。

源码入口：`tensorrt_llm/_torch/pyexecutor/_util.py`。

### TensorRT-LLM 功能分支：aggregated Helix suspend 到 host tier

**NVIDIA/TensorRT-LLM；合入记录与基线：** [#21139](https://github.com/NVIDIA/TensorRT-LLM/pull/21139)（2026-10-27T23:50:59Z，feat/helix-agg@[ca7bb431](https://github.com/NVIDIA/TensorRT-LLM/commit/ca7bb431ae8c7a841c8959992adf9b7fa09a604b)）。

已合入feat/helix-agg，**未合入本轮main基线**。GPU压力下先挂起KV到host、host满再recompute-pause；以各CP rank一致的slot和顺序决策，避免无可释放decode时context死锁。disk tier及connector仍拒绝，disaggregated generation不能靠重放prefill恢复。**建议继续观察**该分支集成；不可把功能分支结果写成main或已发版能力。

源码入口：`tensorrt_llm/_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py`。

### TensorRT-LLM 功能分支：DFlash 的 draft KV 随 target 一起转移

**NVIDIA/TensorRT-LLM；合入记录与基线：** [#20806](https://github.com/NVIDIA/TensorRT-LLM/pull/20806)（2026-10-27T17:30:14Z，feat/musespark-open-agentx@[0f6cff3c](https://github.com/NVIDIA/TensorRT-LLM/commit/0f6cff3c0f155e69a55e7fe3ecd420b8d1c10c31)）。

已合入feat/musespark-open-agentx，**不是本轮main的新交付**。把draft layers并入target manager，修复分离部署prompt draft KV缺失；用按pool及活跃行的page table抑制旧port的+26–33%步时/+7GB开销，并处理多发送rank的bounce fan-in。作者在GB300、Muse Spark Open/DFlash2、K=7的TEP/DEP4/8负载观察到acceptance与聚合部署在噪声内一致；C++ NVFP4 KV部分未验证、modeling_v2 drafter仍有VANILLA约束。**建议深入阅读**转移所有权，等待main集成后再采用。

源码入口：`cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/config.h`。

### TensorRT-LLM：ADP KV connector 所有权与全局 transfer ID

**NVIDIA/TensorRT-LLM；合入记录与基线：** [#19369](https://github.com/NVIDIA/TensorRT-LLM/pull/19369)（2026-10-27T18:36:58Z，默认分支@[0a03c3ed](https://github.com/NVIDIA/TensorRT-LLM/commit/0a03c3ed057fe8c488e221413640ca87f64e99d2)）；[#21043](https://github.com/NVIDIA/TensorRT-LLM/pull/21043)（2026-10-27T13:24:01Z，默认分支@[2d3e60e2](https://github.com/NVIDIA/TensorRT-LLM/commit/2d3e60e27e7293973e3b79f34392492a65386999)）。

V2 connector当前要求PP=1且CP=1，每个ADP owner维护自己的adapter、完成/取消/恢复与prefix reservation，兼容的外部content pool可共享；被取消load的响应须在对齐gate交付后释放routing/slot，末个响应后save仍须继续推进。OpenEngine context-only请求改用正int64范围OS随机ID并字符串交接，避免不同prefill engine相同local ID替换live session；是概率唯一而非中央编号保证。该PR正文仍有旧draft措辞，但API已核实合入。**建议考虑采用**并复测多P/单D重叠及取消，CPU序列化测试不替代GPU验证；connector默认600秒超时、60秒control grace后的fatal须重启，非单请求可恢复。

源码入口：`examples/llm-api/llm_kv_cache_connector_adp.py`；`tensorrt_llm/grpc/openengine/disagg.py`。

### TensorRT-LLM：KV 容量估算、CC 地址可达性与 QSA 冲突诊断

**NVIDIA/TensorRT-LLM；合入记录与基线：** [#17658](https://github.com/NVIDIA/TensorRT-LLM/pull/17658)（2026-10-27T21:00:09Z，默认分支@[b542326c](https://github.com/NVIDIA/TensorRT-LLM/commit/b542326c60555c1a0e7bc4eb1f1515af18a0b3c5)）；[#20772](https://github.com/NVIDIA/TensorRT-LLM/pull/20772)（2026-10-27T16:25:47Z，默认分支@[ab95e84b](https://github.com/NVIDIA/TensorRT-LLM/commit/ab95e84be21fb8e8ded7cd5879cc543b937737cb)）；[#20904](https://github.com/NVIDIA/TensorRT-LLM/pull/20904)（2026-10-27T09:14:28Z，默认分支@[8bd66f6a](https://github.com/NVIDIA/TensorRT-LLM/commit/8bd66f6a7101dff68cf718c6aba25e7ce0448ca1)）。

V2 hybrid容量计入retained snapshots并去重重合边界，保留live-state floor与显式pool比例；机密计算下不能由GPU直接解引用CPU page table，改gather活跃K-plane并异步拷到device再用原kernel展开；SWA/DSv4既有host-initiated上传不变。QSA的FP8/NVFP4同页尺寸碰撞已有#20276修复，本次#20904新增可定位RuntimeError及真实碰撞layout测试，不能再次把原修复算成本次发明。**建议考虑采用**，分别验证snapshot保留、CC与QSA量化布局；缺少统一性能对照。

源码入口：`cpp/tensorrt_llm/batch_manager/kvCacheManagerV2Utils.cpp`；`tensorrt_llm/_torch/pyexecutor/kv_cache/mamba_cache_manager/mamba_cache_manager_v2.py`；`tensorrt_llm/_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py`。

### TensorRT-LLM：投机选择、低 M GEMM 与 logprobs 工作区

**NVIDIA/TensorRT-LLM；合入记录与基线：** [#20940](https://github.com/NVIDIA/TensorRT-LLM/pull/20940)（2026-10-27T16:45:44Z，默认分支@[935dc4a8](https://github.com/NVIDIA/TensorRT-LLM/commit/935dc4a8018c2e145f42ad52ca78afdb34f12cd7)）；[#20942](https://github.com/NVIDIA/TensorRT-LLM/pull/20942)（2026-10-27T14:48:52Z，默认分支@[07129805](https://github.com/NVIDIA/TensorRT-LLM/commit/071298058728f652a416023cf1c94f2fc89368af)）；[#20803](https://github.com/NVIDIA/TensorRT-LLM/pull/20803)（2026-10-27T04:40:45Z，默认分支@[8354c4c3](https://github.com/NVIDIA/TensorRT-LLM/commit/8354c4c3984410a5f735ed0ffa818162c2adaf05)）。

词表并行token选择仅用于均匀shard等支持布局，自定义logits processor回退gather；DSpark Markov与tail、drafter TP归约增加融合。FP8 block-scale低M投影按权重row limit选CuTe kernel（SM100/103及专用SM107 tactics），不适用时回退；sampler仅为实际生成行分配processed logprob临时区，raw logprob tiled选择避免dense词表结果，beam仍保留输出但不走raw gather。**建议考虑复现**真实模型/TP与logprobs负载，来源测试覆盖不等于统一端到端提速数字。

源码入口：`tensorrt_llm/_torch/models/modeling_dflash.py`；`tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py`；`tensorrt_llm/_torch/pyexecutor/sampler/logprobs.py`。

### TensorRT-LLM：prefill 调度与确定性遵循显式策略

**NVIDIA/TensorRT-LLM；合入记录与基线：** [#20600](https://github.com/NVIDIA/TensorRT-LLM/pull/20600)（2026-10-27T04:49:44Z，默认分支@[27c339ab](https://github.com/NVIDIA/TensorRT-LLM/commit/27c339ab53e912def9fb0b1232807fd3300ef658)）；[#20227](https://github.com/NVIDIA/TensorRT-LLM/pull/20227)（2026-10-27T07:25:53Z，默认分支@[c554d448](https://github.com/NVIDIA/TensorRT-LLM/commit/c554d4486d2900ff3152f532d239487850ad88a2)）。

V2可按估计uncached token数把长prefill放到短请求之后，保留类内FCFS并可设等待轮数上限；两个knob默认0，要求chunking，无estimate不降级，重排计算不等同降低KV抢占保护。另只有FORCE_DETERMINISTIC或FORCE_MOE_KERNEL_DETERMINISTIC精确为1时关闭MoE finalize fusion，以减少累加顺序变化。**建议深入阅读并复测**SLO/公平性和数值确定性；无基准支持普遍吞吐改善或全系统bitwise保证。

源码入口：`tensorrt_llm/_torch/pyexecutor/llm_request.py`；`tensorrt_llm/_torch/model_config.py`。

### TensorRT-LLM：GDN ring replay 已接入，但当前更慢

**NVIDIA/TensorRT-LLM；合入记录与基线：** [#20008](https://github.com/NVIDIA/TensorRT-LLM/pull/20008)（2026-10-27T06:29:40Z，默认分支@[ae91047a](https://github.com/NVIDIA/TensorRT-LLM/commit/ae91047ac364b65567aa5ef7f20d6fb32deb0d5e)）。

V2 cache接入FlashInfer ring-buffer GDN MTP replay，管理分配、cursor、flush、reset和acceptance，省去原逐层commit/work-list路径；支持SM90/100/103、BF16 state、K=V=128、每步4或8 token等条件。**作者A/B明确仍慢于现有Triton**，因此TRTLLM_USE_GDN_FLASHINFER_REPLAY需opt-in，强制不满足条件报错，普通模式回退。**建议继续观察**性能工作，作为状态管理设计阅读；不能以新增backend推断加速。

源码入口：`tensorrt_llm/_torch/modules/fla/cache_manager.py`。

### FlashInfer：确定性 GDN chunked training backend

**flashinfer-ai/flashinfer；合入记录与基线：** [#7592](https://github.com/flashinfer-ai/flashinfer/pull/7592)（2026-10-28T00:17:40Z，默认分支@[6b6c9f99](https://github.com/flashinfer-ai/flashinfer/commit/6b6c9f996de3038ed139eab3fb3a985aaaa0cd61)）。

experimental.cake_gdn_det_train提供显式导入的前反向，head_dim128、chunk64、HV为H整数倍，支持BF16 Q/K/V、BF16/FP32 beta、可选FP32初末状态和packed varlen；不含context parallelism。来源声明固定layout/plan重复运行逐位一致，性能表仅特定packed变体，不能推广到所有head grouping及训练任务。**建议考虑复现**需要确定性的GDN训练，先测与FLA参考的输出/梯度及多步收敛。

源码入口：`flashinfer/experimental/cake_gdn_det_train/__init__.py`。

### FlashInfer：两条 NVFP4 专家并行后端的含义不同

**flashinfer-ai/flashinfer；合入记录与基线：** [#7557](https://github.com/flashinfer-ai/flashinfer/pull/7557)（2026-10-27T08:58:57Z，默认分支@[124955a4](https://github.com/flashinfer-ai/flashinfer/commit/124955a41bf2857a52e7d77687f9b2751c00670c)）；[#6999](https://github.com/flashinfer-ai/flashinfer/pull/6999)（2026-10-27T04:59:48Z，默认分支@[479c38bb](https://github.com/flashinfer-ai/flashinfer/commit/479c38bbfceb0228a35a6d03f402fb95a75a08a9)）。

SM100/103 Cake W4A16以BF16 activation、NVFP4 weight输入，kernel内把weight解码为BF16后用BF16 tensor core/FP32累加，再接counted top-8 combine；不是原生FP4×BF16 MMA。另SM90 unified MoEEpLayer接入NVFP4×FP8 CuTe MegaMoE，默认kperm64、容量表限定H7168/I3072/E384/top6/EP4，其他shape启发式；所给NV/MX对照把MX compact手动打开，部分小token NV更慢。**建议深入阅读并复现**目标量化格式与EP负载，勿混用两路算术/性能结论。

源码入口：`csrc/cake_mega_moe_w4a16/cake_ffi_checks.h`；`benchmarks/bench_moe_ep_sm90_nvfp4.py`。

### FlashInfer：KDA 状态布局与 paged prefill 的明确支持面

**flashinfer-ai/flashinfer；合入记录与基线：** [#7132](https://github.com/flashinfer-ai/flashinfer/pull/7132)（2026-10-27T22:34:03Z，默认分支@[f9ad55c8](https://github.com/flashinfer-ai/flashinfer/commit/f9ad55c831d3558a6a487a2067acc746dea884b3)）；[#7512](https://github.com/flashinfer-ai/flashinfer/pull/7512)（2026-10-27T06:17:37Z，默认分支@[07cc0d60](https://github.com/flashinfer-ai/flashinfer/commit/07cc0d60a9ba1aca87b7b803ab1810ca5e3738d6)）。

fused/packed KDA decode加入16/64 heads，head_dim128，单步接口兼容vLLM NIXL需要的DS conv-state stride，状态仍原地更新；布局通过编译期variant选择。Cake paged context新增SM100a/103a的BF16/FP8/NVFP4 KV程序，head_dim128、右下causal，LSE或attention sink继续旧route；不是任意head_dim或任意mask的新attention实现。**建议考虑采用**前先验证状态传输stride和实际prefill dispatch，本轮没有重跑微基准。

源码入口：`flashinfer/kda_decode.py`；`csrc/cake_fmha/cuda/context_bf16/sm_100a/enable_sink0_heads_per_group5_is_causal1_l2_swizzle1_num_m_blocks11_num_q_heads10_pack_g5_page_size32_return_lse0_single_mask_loop1_tok_per_stage25.cu`。

### FlashInfer：低精度写界、非连续索引与有效长度修正

**flashinfer-ai/flashinfer；合入记录与基线：** [#6278](https://github.com/flashinfer-ai/flashinfer/pull/6278)（2026-10-27T22:29:01Z，默认分支@[e9c4787c](https://github.com/flashinfer-ai/flashinfer/commit/e9c4787c3ba9e2f59df1fd94a6bb093e3e40815e)）；[#5385](https://github.com/flashinfer-ai/flashinfer/pull/5385)（2026-10-27T02:40:53Z，默认分支@[61855497](https://github.com/flashinfer-ai/flashinfer/commit/618554979285eaf2fa803e6803e1ccb014efd957)）；[#7438](https://github.com/flashinfer-ai/flashinfer/pull/7438)（2026-10-27T05:18:42Z，默认分支@[a3452cd4](https://github.com/flashinfer-ai/flashinfer/commit/a3452cd4051c32d6d0420266ef34c8b53565cb9a)）。

NVFP4 TMA量化padding仅写numPaddedRowsForSf以内，避免最后16行tile越过8x4/linear scale布局并破坏相邻tensor；plan()/size-query对strided indptr/indices/last_page_len作contiguous，避免CPU planner和GPU kernel读到不同布局。cuDNN token-offset prefill分开物理origin和显式有效长度，后者决定mask，offset保留各tensor单位并纳入graph reuse key，padding输出仍未定义。**建议优先考虑采用**并做带canary的边界、stride和ragged回归；这些是正确性修复。

源码入口：`csrc/nv_internal/tensorrt_llm/kernels/quantization.cuh`；`flashinfer/decode.py`；`flashinfer/cudnn/prefill.py`。

### FlashInfer：TMA / PDL 顺序与 masked route 语义

**flashinfer-ai/flashinfer；合入记录与基线：** [#7455](https://github.com/flashinfer-ai/flashinfer/pull/7455)（2026-10-27T08:33:20Z，默认分支@[8a087fcc](https://github.com/flashinfer-ai/flashinfer/commit/8a087fcc87b1e7baed17c67603fe3051c5066d8a)）；[#7490](https://github.com/flashinfer-ai/flashinfer/pull/7490)（2026-10-27T08:17:21Z，默认分支@[2725b4ec](https://github.com/flashinfer-ai/flashinfer/commit/2725b4ec0e378700c415e3354bdc90307f8bca4f)）；[#7062](https://github.com/flashinfer-ai/flashinfer/pull/7062)（2026-10-27T19:32:58Z，默认分支@[7bce78b1](https://github.com/flashinfer-ai/flashinfer/commit/7bce78b15d6462e054bbd565fbadaf4daa65ac0d)）。

SM120 MiniMax-H3 MLP的ab_empty之后补proxy fence，保证共享内存读完成再被TMA refill覆盖；作者同形状微测速度比0.995–1.002，价值在正确性。warp_decode链首若未先wait就读外部routing，则改用stream顺序，只有满足wait/trigger契约的首kernel保留PDL。Rubin MegaMoE允许expert=-1的masked槽带非有限分数并置零，active route仍默认查finite/范围/重复；关闭value check须在warmup/capture前设环境。**建议考虑采用**并复测异步生产者和图重放。

源码入口：`csrc/cake_minimax_h3_sm120_fc1_gemm_fp8.cuh`；`csrc/fused_moe/warp_decode/cake_warp_decode_binding.cu`；`flashinfer/moe_ep/backends/mega/kernel/sm107/mxfp8_mxfp8_bf16_cutedsl/backend.py`。

### FlashInfer：SM12x MoE workspace 分桶抑制在线重复编译

**flashinfer-ai/flashinfer；合入记录与基线：** [#3836](https://github.com/flashinfer-ai/flashinfer/pull/3836)（2026-10-28T00:39:30Z，默认分支@[959a113b](https://github.com/flashinfer-ai/flashinfer/commit/959a113b9467198d78c89431b0cbee621f7826b1)）。

static NVFP4/MXFP4 workspace的routed_rows容量按最小128的2次幂增长，避免每次新high-water mark改变kernel cache key、使旧shape在forward重新编译。代价为最多接近2倍scratch行；dynamic backend及W4A16不套用该策略。**建议考虑采用**于batch波动服务，并测p99停顿与scratch峰值；不能只看steady-state kernel时间。

源码入口：`flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py`。

### AITER：IPC 子分配偏移与真实 XCD 数

**ROCm/aiter；合入记录与基线：** [#5775](https://github.com/ROCm/aiter/pull/5775)（2026-10-27T06:02:45Z，默认分支@[84dd855c](https://github.com/ROCm/aiter/commit/84dd855c91746dbe96cf60e40801194db391c306)）；[#5182](https://github.com/ROCm/aiter/pull/5182)（2026-10-27T07:51:11Z，默认分支@[e902ad1c](https://github.com/ROCm/aiter/commit/e902ad1c8d98450f0d7b9cce812a29d475f04589)）。

custom all-reduce向peer发送hipMemGetAddressRange得到的allocation base相对offset，不再固定0；作者gfx950/ROCm7.2、PyTorch2.12.1、TP2、BF16[128,8192]在碎片分配下复现并修正，但未测两个真实共驻engine。HIP属性编号随版本变化会误判MI308，MI350P也不能按8 XCD处理；新增runtime查询/型号识别避免错误L2 swizzle。**建议优先考虑采用**于共池/碎片和MI350P部署，硬件识别正确不等于已证明端到端更快。

源码入口：`aiter/dist/device_communicators/custom_all_reduce.py`；`aiter/dist/device_communicators/rocm_version.py`。

### AITER：舍入、alias 契约及 RoPE / MoE 边界

**ROCm/aiter；合入记录与基线：** [#6665](https://github.com/ROCm/aiter/pull/6665)（2026-10-27T08:40:29Z，默认分支@[9054e899](https://github.com/ROCm/aiter/commit/9054e8999aed0aae8ecb7c4394ab3d24cbd70121)）；[#6902](https://github.com/ROCm/aiter/pull/6902)（2026-10-27T07:50:06Z，默认分支@[e3950535](https://github.com/ROCm/aiter/commit/e3950535309c47e9087727075cb22cbd2ffc24f7)）；[#6879](https://github.com/ROCm/aiter/pull/6879)（2026-10-27T08:29:10Z，默认分支@[c0578eda](https://github.com/ROCm/aiter/commit/c0578edaaf148cec4aee70c53f0e105448b47f02)）；[#6736](https://github.com/ROCm/aiter/pull/6736)（2026-10-27T07:28:05Z，默认分支@[44b4fea6](https://github.com/ROCm/aiter/commit/44b4fea64200eea638770a5815a7d9aa40c752ab)）。

gfx942的六个CK-Tile a8w8 GEMM从默认截断改round-to-nearest-even，gfx950/gfx12原native转换不受影响；gemm_a16w16登记mutates_args=[]，防止compile把packed bias storage当成被写而克隆越界；per-entry MLA RoPE始终clamp到cos/sin cache，修复Kimi单行NoPE缓存越界/静默零。gfx950 FP8 MoE补K128 tile同时明确拒绝不合约的unshuffled权重，并非所有inter_dim都可用。**建议考虑采用**，对照后端切换和packed权重的数值回归。

源码入口：`aiter/jit/optCompilerConfig.json`；`aiter/tuned_gemm.py`；`csrc/kernels/cache_kernels.cu`；`aiter/fused_moe.py`。

### AITER：短 query 投机验证与 virtual sink

**ROCm/aiter；合入记录与基线：** [#6896](https://github.com/ROCm/aiter/pull/6896)（2026-10-27T13:37:57Z，默认分支@[a4ce041b](https://github.com/ROCm/aiter/commit/a4ce041b5e47bc18daff539fbd8e9c6c053b5f82)）；[#6760](https://github.com/ROCm/aiter/pull/6760)（2026-10-27T06:30:04Z，默认分支@[836b336e](https://github.com/ROCm/aiter/commit/836b336ede3d1f2cffd6624faaadc24a9fafc009)）；[#6906](https://github.com/ROCm/aiter/pull/6906)（2026-10-27T02:04:31Z，默认分支@[9b1dfee2](https://github.com/ROCm/aiter/commit/9b1dfee264979d11babe0dadc77c08bc03cb2c2a)）。

gfx1250短query按实际query行缩小2D tile，低并行度选3D split-KV；3D补逐行causal与全mask防NaN。作者Qwen3.5形状Q/KV heads16/2、D256、q_len4、ctx1280的batch1/64/128微测由350/408/449µs降至17/24/32µs，不是整个模型同倍数加速。single-query causal prefill带sink时保留causal specialization，并阻止sink dispatch到不支持的assembly路径；非causal sink不在修复范围。**建议考虑复现**目标spec verify负载和sink边界。

源码入口：`aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_3d.py`；`aiter/ops/mha.py`。

### AITER：gfx950 MXFP8 BMM 接入 1×32 weight scales

**ROCm/aiter；合入记录与基线：** [#6060](https://github.com/ROCm/aiter/pull/6060)（2026-10-27T13:21:43Z，默认分支@[66c86301](https://github.com/ROCm/aiter/commit/66c86301960dc1c3e6ea901a98719d628d23bcaa)）。

原方块32×32/128×128 scale之外支持MiniMax-M3式每weight row、每32K一个e8m0 scale，不要求repack；dispatch/AOT及配置把scale布局纳入选择，XCD内tile重排旨在提升L2复用。收益随shape与配置变化，不能用格式支持承诺全模型加速。**建议考虑采用**于实际OCP MX checkpoint，并确认scale shape与旧CSV兼容默认值。

源码入口：`aiter/aot/flydsl/gemm.py`。

### AITER 功能分支：all-reduce / residual / RMSNorm / quant 一次 launch

**ROCm/aiter；合入记录与基线：** [#6924](https://github.com/ROCm/aiter/pull/6924)（2026-10-27T18:21:16Z，satya/gluon_allreduce_1250@[b9c1215c](https://github.com/ROCm/aiter/commit/b9c1215c48fdcb2b80ad78bfb6059000afba64c2)）。

已合入satya/gluon_allreduce_1250，**本轮不据此宣称main已支持**。gfx1250接收端获得整行后做residual、RMSNorm及可选128-value FP8 group quant；all-reduce先舍入BF16，residual和norm中间保持FP32。作者TP4、8 token微测约5µs对RCCL+add_rmsnorm约9µs，未等价于模型步时收益，且默认分支集成未核实。**建议继续观察**，需核通信拓扑、残差/scale语义及实际落点。

源码入口：`aiter/ops/triton/_gluon_kernels/gfx1250/comms/fused/all_reduce_rmsnorm_quant.py`。

### vLLM Ascend：Mooncake 生命周期与 dense DP endpoint 隔离

**vllm-project/vllm-ascend；合入记录与基线：** [#18118](https://github.com/vllm-project/vllm-ascend/pull/18118)（2026-10-27T11:41:52Z，默认分支@[80bfd9a9](https://github.com/vllm-project/vllm-ascend/commit/80bfd9a989ab2407fa1c7c95988cc9ecc709e9fe)）；[#19744](https://github.com/vllm-project/vllm-ascend/pull/19744)（2026-10-27T14:31:56Z，默认分支@[b006d843](https://github.com/vllm-project/vllm-ascend/commit/b006d843bcd1f5a75d5d7299a2c9dc10192ffba5)）。

sleep前先注销transfer buffer并断peer，wake kv_cache后重注册；DONE_RECVING新socket重试、未开始remote prefill的请求显式abort，缩短prompt后重建block hashes，reset prefix cache时清实际pool；调用方须先pause并等待所有writing ranks的put完成，scheduler/worker须同步升级tagged frames协议；layerwise传输仍不支持。dense DP计算rank会重置0，endpoint改用保留的data_parallel_index，避免同机独立replica绑同端口/IPC路径。原PR未把已经由上游处理的partial-wake patch再次引入。**建议优先考虑采用**并回归sleep/wake、取消、reset和DP2/TP1冷启动；多卡负载仍需实测。

源码入口：`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`；`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/base_scheduler.py`。

### vLLM Ascend：Engram private Host-UVA 对齐修复静默读错

**vllm-project/vllm-ascend；合入记录与基线：** [#19695](https://github.com/vllm-project/vllm-ascend/pull/19695)（2026-10-27T02:41:52Z，默认分支@[90cd6316](https://github.com/vllm-project/vllm-ascend/commit/90cd631625a569dc420f8235ed9d525b8c9eaede)）。

针对A3/CANN9.2.0上CPU发布正确而NPU alias读错的小private HostUvaBuffer，payload改64KiB对齐、保留原allocation用于free/rollback，每buffer至多额外65535字节，无热路径copy。作者16,384行INT8/FP32表原scale有122850字节不一致、32次索引25次错，对齐后检查一致。**建议考虑采用并实机复现**；范围限private buffer，不改变SharedUvaBuffer或证明所有UVA硬件问题已解决。

源码入口：`vllm_ascend/models/deepseek_v41/engram/npu.py`。

### vLLM Ascend：GLM-5.3 PCP 与 KDA 状态切分

**vllm-project/vllm-ascend；合入记录与基线：** [#18511](https://github.com/vllm-project/vllm-ascend/pull/18511)（2026-10-27T01:40:23Z，默认分支@[5b01b97c](https://github.com/vllm-project/vllm-ascend/commit/5b01b97c723380d056dba50203ac32b8ac7caf91)）；[#19518](https://github.com/vllm-project/vllm-ascend/pull/19518)（2026-10-27T01:42:13Z，默认分支@[749d743e](https://github.com/vllm-project/vllm-ascend/commit/749d743ebf2accafeeaaf75578a6f6f71bb48ad5)）；[#19620](https://github.com/vllm-project/vllm-ascend/pull/19620)（2026-10-27T08:12:39Z，默认分支@[0330bfd4](https://github.com/vllm-project/vllm-ascend/commit/0330bfd4f62d045ff9ebf14de5797c2e03ca92e3)）。

KPool压缩前恢复全局causal token顺序，sparse query仍local；TP1时projection/conv/state按PCP heads切并归约，TP>1保留TP head切分。Mooncake V2按TP-major/PCP-minor映射state，当前明确要求PCP=1或TP=1，不支持联合两者都大于1。移除recurrent_kda全padding masked_fill依赖消费者只读有效行/清padding。作者10.6µs节省缺完整负载与硬件，不与外部数字比较；packed SFA C8依赖#19079；GLM PCP补丁只报告42项隔离CPU检查，KPool decode当前仍复制到各PCP rank。**建议考虑复现**state transfer与graph/MTP边界。

源码入口：`vllm_ascend/attention/context_parallel/common_cp.py`；`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/base_scheduler.py`；`vllm_ascend/models/glm5next/ops/kda.py`。

### vLLM Ascend：MRV2 投机路径与 HCCL 训推权重传输

**vllm-project/vllm-ascend；合入记录与基线：** [#18061](https://github.com/vllm-project/vllm-ascend/pull/18061)（2026-10-27T08:22:55Z，默认分支@[f032ca46](https://github.com/vllm-project/vllm-ascend/commit/f032ca4638296b88b5d27f8d470737fcd80457a8)）；[#19296](https://github.com/vllm-project/vllm-ascend/pull/19296)（2026-10-27T01:23:01Z，默认分支@[3c6d0169](https://github.com/vllm-project/vllm-ascend/commit/3c6d01695de5c70c012cde449ec60bf138738af0)）；[#18944](https://github.com/vllm-project/vllm-ascend/pull/18944)（2026-10-27T06:09:44Z，默认分支@[ec26d254](https://github.com/vllm-project/vllm-ascend/commit/ec26d2541fcbec6b4d719de1524c45d75b26c386)）。

MRV2接入NPU ngram speculator与多KV-group MultiModuleMTP（Step3.7/Step5等）；ngram作者Qwen3-1.7B BF16 TP1、Sonnet100个prompt、并发10、输入128/输出100/prefix10的MRV1→MRV2同时改变runner，TPOT12.22→8.61ms不能归因纯ngram优化，未明确NPU型号。HCCL engine复用上游NCCL orchestration、layerwise reload与metadata校验，旧trainer静态API移除需迁移，仍保留NPU数据面/设备绑定，拒绝NCCL-only unique-ID rendezvous，trainer默认packed=True。**建议深入阅读并复测**依赖版本、reload完成边界和真实长输出负载；其真实NPU验证带#19127 overlay，不是PR单独验证。

源码入口：`vllm_ascend/patch/worker/patch_v2/patch_spec_decode_310.py`；`vllm_ascend/worker/v2/spec_decode/__init__.py`；`examples/rl/rlhf_async_new_apis.py`。

### PyTorch：UE5M3 作为 shell dtype 入库

**pytorch/pytorch；合入记录与基线：** [#201269](https://github.com/pytorch/pytorch/pull/201269)（2026-10-28T00:51:27Z，默认分支@[91542833](https://github.com/pytorch/pytorch/commit/91542833cc8f1e563920a718457b4879cb01cbca)）。

新增torch.float8_e5m3fnu，包含storage/序列化、CPU/CUDA转换与有限张量操作；是无符号E5M3格式，不等于现有E5M2，0xFF为NaN，高端有限编码需按实现解释。**没有通用算术kernel、type promotion或DLPack支持**。**建议深入阅读**量化scale/新硬件格式适配；仅dtype入库不能写成LLM全面支持该精度训练或推理。

源码入口：`aten/src/ATen/DLConvertor.cpp`。

### PyTorch：在线换权重不再被并发读取饿死，hook 地址回收会失效缓存

**pytorch/pytorch；合入记录与基线：** [#201827](https://github.com/pytorch/pytorch/pull/201827)（2026-10-28T00:43:37Z，默认分支@[b5a961b8](https://github.com/pytorch/pytorch/commit/b5a961b8618d74a7372e264f7e94a60bb8a4e747)）；[#202143](https://github.com/pytorch/pytorch/pull/202143)（2026-10-27T19:21:28Z，默认分支@[a9261ad3](https://github.com/pytorch/pytorch/commit/a9261ad30952dbb337deee41f202c42d1d85bae7)）。

AOTI model_exec_mutex使用带turnstile的写优先共享锁，避免持续run占读锁使swap_constant_buffer/constant folding等不到；作者H100每秒800请求回放曾见swap等5秒至25分钟，修复目标是准入公平性，不能保证零停顿。Dynamo对saved_tensors_hooks登记id_ref，释放旧hook即失效图，防止Python地址复用命中旧语义。**建议考虑采用**于在线权重切换与activation offload，验证服务尾延迟、writer公平性和旧图失效。

源码入口：`torch/csrc/inductor/aoti_runtime/model_container.h`；`torch/_dynamo/guards.py`。

### PyTorch：对称内存失败回滚、NCCL split 与 EP 构建来源

**pytorch/pytorch；合入记录与基线：** [#201284](https://github.com/pytorch/pytorch/pull/201284)（2026-10-27T19:44:54Z，默认分支@[16fc997b](https://github.com/pytorch/pytorch/commit/16fc997b24083c5181d2b23a5c946601908fe8c7)）；[#202072](https://github.com/pytorch/pytorch/pull/202072)（2026-10-27T15:32:07Z，默认分支@[1aa1528f](https://github.com/pytorch/pytorch/commit/1aa1528f66debd0fd1cd893a3ac73fa49cbfa52f)）；[#199948](https://github.com/pytorch/pytorch/pull/199948)（2026-10-27T19:25:52Z，默认分支@[afabe7b9](https://github.com/pytorch/pytorch/commit/afabe7b9bd1301b781e35098713afeb3ffd6d93e)）。

CUDA VMM部分map失败先回滚并只在成功后发布pointer，避免资源取得与AllocationRef接管之间泄漏；blocking NCCL split child立即标initialized，nonblocking仍等待完成。NCCL EP实际main代码从third_party/nccl-extensions/nccl_ep构建；该目录缺失时按pin.txt的ab38d31b10af0247ec0dbabb97e996b6185fda69自动FetchContent拉取，再针对已选NCCL编译。topk_idx限int32/int64并转int64供native handle。PR旧正文的“仅显式source、不自动获取”与最终默认分支不符，本条以实际commit为准。**建议考虑采用**并测allocation失败注入、嵌套split和部署包原生artifact；PR ghstack base与实际main commit分别登记。

源码入口：`torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory.cu`；`torch/csrc/distributed/c10d/NCCLUtils.cpp`；`CMakeLists.txt`。

### PyTorch：编译候选身份、向量宽度、舍入与 XPU 内存池

**pytorch/pytorch；合入记录与基线：** [#201695](https://github.com/pytorch/pytorch/pull/201695)（2026-10-27T01:35:45Z，默认分支@[d6557fba](https://github.com/pytorch/pytorch/commit/d6557fba8cf0f3b77863f57ac37407a143f377f8)）；[#202123](https://github.com/pytorch/pytorch/pull/202123)（2026-10-27T18:30:22Z，默认分支@[76f0997b](https://github.com/pytorch/pytorch/commit/76f0997bd6288b7cd9cc8c01e84be918fe37566e)）；[#201713](https://github.com/pytorch/pytorch/pull/201713)（2026-10-27T09:29:52Z，默认分支@[090a616f](https://github.com/pytorch/pytorch/commit/090a616f58c9f26eb8057ade7170ee00995014ef)）；[#201820](https://github.com/pytorch/pytorch/pull/201820)（2026-10-27T06:37:18Z，默认分支@[7b2b0144](https://github.com/pytorch/pytorch/commit/7b2b014443760e9f54ed0d9c3a022c96baa73417)）。

Inductor把非空Triton extra_options纳入候选hash/identity，避免误去重和复用旧tuning；非JSON值即使关闭cache也会更早报错，已有非空配置cache失效。FlexFlash score_mod按实际架构支持选择vector width，不能简单以SM100以上概括SM120；拆conv bias时补低精度舍入屏障，XPU嵌套use_mem_pool按最内层优先。**建议考虑采用**相关路径并重跑数值/性能baseline，conv修复与LLM卷积混合架构有关，未证明通用LLM加速。

源码入口：`torch/_inductor/runtime/autotune_cache.py`；`torch/_inductor/kernel/flex/aux_vectorization.py`；`torch/_inductor/config.py`；`c10/xpu/XPUCachingAllocator.cpp`。

### Triton main：同日 warp-ID 折叠优化已回退

**triton-lang/triton；合入记录与基线：** [#12498](https://github.com/triton-lang/triton/pull/12498)（2026-10-27T01:40:40Z，默认分支@[1b4a2041](https://github.com/triton-lang/triton/commit/1b4a2041b0e3eda43ad398636098ef9147e4663c)）；[#12518](https://github.com/triton-lang/triton/pull/12518)（2026-10-27T15:20:49Z，默认分支@[d3229a17](https://github.com/triton-lang/triton/commit/d3229a17e19855c704059d5aa2214659430deeac)）；[#12505](https://github.com/triton-lang/triton/pull/12505)（2026-10-27T11:42:23Z，默认分支@[75b16128](https://github.com/triton-lang/triton/commit/75b1612805abe2127abc14297a67f932dfd48a7b)）。

#12498把单warp specialization的relative warp ID常量化为0以去分支；#12518因增加register spilling和性能回归撤回。**不能把先前合入算作当前有效优化**。另#12505保留thread handoff的axis信息并避免普通load引入full-domain handoff，是独立修正。**建议继续观察并复现**自身kernel，少一条分支不保证少寄存器或更快；这些main事件与3.9 release需按tag内容分开。

源码入口：`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ConvertWarpSpecializeToLLVM.cpp`；`lib/Analysis/AxisInfo.cpp`。

### Dynamo main：多模态缓存按真实宽度计费，去掉丢弃的 token decode

**ai-dynamo/dynamo；合入记录与基线：** [#17249](https://github.com/ai-dynamo/dynamo/pull/17249)（2026-10-27T05:37:51Z，默认分支@[2e96c98b](https://github.com/ai-dynamo/dynamo/commit/2e96c98ba75372091b6f2bdc1f5b08e13f5a5dc5)）；[#17349](https://github.com/ai-dynamo/dynamo/pull/17349)（2026-10-27T20:15:46Z，默认分支@[cffc4fa7](https://github.com/ai-dynamo/dynamo/commit/cffc4fa75c15017bcd32e62f7d0fe4bad468c35d)）。

DeepStack image/video row按vLLM≥0.32启动测出的每modality encoder width计费（Qwen3-VL-235B约hidden4倍），老版本/未知modality回退hidden；同轮eviction从待save移除，避免scheduler已丢弃而worker仍存。无logprobs chunk跳过仅供logprobs消费的逐token decode；作者Qwen3-0.6B tokenizer、2048mocker/6144并发、双NUMA frontend CPU profile净省约0.35ms/请求、约2% user CPU，非GPU端到端时延。**建议考虑采用**，勿误认为这些main提交已进入只做维护更新的v1.6.1。

源码入口：`components/src/dynamo/vllm/multimodal_utils/multimodal_embedding_cache_connector.py`；`lib/llm/src/backend.rs`。

### TorchTPU：重复 P2P key、跨 stream 拷贝与动态 slice 梯度

**google-pytorch/torch_tpu；合入记录与基线：** [#5754](https://github.com/google-pytorch/torch_tpu/pull/5754)（2026-10-27T23:37:14Z，默认分支@[fb7bdfcb](https://github.com/google-pytorch/torch_tpu/commit/fb7bdfcb244b26a63b9b31e8442277f9153af090)）；[#5927](https://github.com/google-pytorch/torch_tpu/pull/5927)（2026-10-27T16:38:18Z，默认分支@[b808e9dc](https://github.com/google-pytorch/torch_tpu/commit/b808e9dc956c8c650391876e095f79597702db3e)）；[#5801](https://github.com/google-pytorch/torch_tpu/pull/5801)（2026-10-27T17:25:49Z，默认分支@[6aa47c27](https://github.com/google-pytorch/torch_tpu/commit/6aa47c27bfb946036dbeefaae522ecf0eef05ba1)）。

P2P key改用(peer,tag)方向上的单调tensor序号，防止重复tag读取Store旧descriptor；blocking TPU→CPU copy若source在另一stream则显式materialize该stream，避免release构建静默死锁。slice_backward按ShapeEnv上界确定窗口并清除超运行时长度的padding梯度。**建议考虑采用**于流水通信/异步D2H和动态长度训练，来源回归用例不等于多host长跑稳定性证明。

源码入口：`csrc/distributed/process_group_tpu.cc`；`csrc/ops/copy_from/tpu_to_cpu.cc`；`src/torch_tpu/_internal/compile/dynamic/view_ops_passes.py`。

### TorchTPU：块稀疏 FlexAttention forward 与 Kiln 输入流水骨架

**google-pytorch/torch_tpu；合入记录与基线：** [#5774](https://github.com/google-pytorch/torch_tpu/pull/5774)（2026-10-27T18:15:50Z，默认分支@[4f64ecc4](https://github.com/google-pytorch/torch_tpu/commit/4f64ecc4a2ec070494f7002de2af2eb1eb17f7c0)）；[#5769](https://github.com/google-pytorch/torch_tpu/pull/5769)（2026-10-27T15:34:35Z，默认分支@[ad93e3a7](https://github.com/google-pytorch/torch_tpu/commit/ad93e3a74ce027ff5065288cc18c3f97e83e59c3)）。

新Kiln forward只访问BlockMask列出的KV块，返回out与log2 LSE；当前列出块全可见，score_mod/mask_mod仍后续，不能称完整FlexAttention。EmitPipeline只实现HBM→VMEM输入流，runtime offset/size可共用compiled kernel，输出流、累积、自动边界等仍后续。**建议深入阅读、继续观察**接口集成与backward，不能以首个骨架推断已有高性能matmul/训练路径。

源码入口：`csrc/eager/tpu_aten_kernels.cc`；`csrc/internal/mosaic/kiln/kiln.h`。

### XLA：scheduler 内存围栏默认启用，VMM host 与 packed backward 工作区收紧

**openxla/xla；合入记录与基线：** [#51705](https://github.com/openxla/xla/pull/51705)（2026-10-27T15:18:34Z，默认分支@[934c7bff](https://github.com/openxla/xla/commit/934c7bff92df5df0682631c81847aa76767fad0e)）；[#51684](https://github.com/openxla/xla/pull/51684)（2026-10-27T12:01:25Z，默认分支@[037e0268](https://github.com/openxla/xla/commit/037e0268ce9491382d48357aa896837ed15d3216)）；[#51680](https://github.com/openxla/xla/pull/51680)（2026-10-27T10:34:54Z，默认分支@[2c60ccf0](https://github.com/openxla/xla/commit/2c60ccf0c78b4d6ee378fb0cb98ae197545a40a6)）。

scheduler fencing阈值默认0表示自动取memory limit的1%，slack window默认1；负值才关闭。通过控制依赖限制大buffer用户延后导致的live-range膨胀，有调度自由度代价。CUDA/ROCm VMM的pinned_host真正由host allocator供应，无host allocator明确失败；ragged cuDNN backward传B×S最大packed token数，避免按B×segments×S放大workspace。**建议考虑复现**LLM动态packing与offload峰值，三项都不是无代价的通用速度提升。

源码入口：`xla/debug_options_flags.cc`；`xla/pjrt/gpu/se_gpu_pjrt_client.cc`；`xla/stream_executor/cuda/cuda_dnn.cc`。

### XLA：AllGather 改 push scratch，ROCm FP8 output 缺 kernel 可回退

**openxla/xla；合入记录与基线：** [#51624](https://github.com/openxla/xla/pull/51624)（2026-10-27T15:28:22Z，默认分支@[84b0d10c](https://github.com/openxla/xla/commit/84b0d10c27feb09e76e1c70729fc97262b2a7458)）；[#51610](https://github.com/openxla/xla/pull/51610)（2026-10-27T08:53:48Z，默认分支@[30fda9d6](https://github.com/openxla/xla/commit/30fda9d6789e5db1c73b8416c5aea36b2e4a9885)）。

one-shot Triton AllGather从peer pull改整tile推peer scratch，scratch按output计，大小上限16→32MiB、grid上限64→128；作者16设备/NVLink、1KiB–32MiB报告1.22–1.54倍latency改善，硬件型号等不足不外推。hipBLASLt仅在fused FP8-output查询无algorithm时，改宽输出GEMM+独立D scale/饱和转换；有fused kernel仍优先，原FP8输出范围与舍入须保持。**建议考虑复现**拓扑与量化边界，该fallback还要求beta=0、无amax/aux及支持的FP8/bias dtype，deviceless/AOT无autotune结果仍无法query/选择，不能保证任意场景可编译；更不等于更快。

源码入口：`xla/backends/gpu/codegen/triton/collective_emitter.cc`；`xla/autotuning.proto`。

### XLA：while 边界与融合数值约束纠偏

**openxla/xla；合入记录与基线：** [#51648](https://github.com/openxla/xla/pull/51648)（2026-10-28T00:21:23Z，默认分支@[2b530f26](https://github.com/openxla/xla/commit/2b530f269d132118049207e9001d9c3cbee02765)）；[#51644](https://github.com/openxla/xla/pull/51644)（2026-10-27T05:57:10Z，默认分支@[61108ee4](https://github.com/openxla/xla/commit/61108ee47fc0178be947220c1cb3e4f73cebccac)）；[#51656](https://github.com/openxla/xla/pull/51656)（2026-10-27T18:52:36Z，默认分支@[9d81e487](https://github.com/openxla/xla/commit/9d81e48725ba04e9e6fbe3183cd5cc4de24381e4)）。

关闭while copy时保持loop-invariant layout跨边界一致，async chain解析兼容多caller while body，并保留带control successor的bitcast。ConstraintPropagator修正向融合中32位累加器错误播种16位范围的问题，跨tuple/GTE传播边界并为两个非常量add操作数分配界；这主要影响测试输入约束/生成，不能当BF16通用kernel精度突破。**建议深入阅读**复杂loop与融合的回归，缺端到端LLM性能证据。

源码入口：`xla/service/layout_assignment.cc`；`xla/hlo/ir/hlo_instructions.cc`。

### JAX：HiJAX unreduced 表达部分归约，Rubin FP4 舍入可选 .rz

**jax-ml/jax；合入记录与基线：** [#42043](https://github.com/jax-ml/jax/pull/42043)（2026-10-27T22:21:23Z，默认分支@[0fedbd32](https://github.com/jax-ml/jax/commit/0fedbd32f43724bca7ae1ce1ea353832da77b3dc)）；[#42045](https://github.com/jax-ml/jax/pull/42045)（2026-10-28T00:40:50Z，默认分支@[44fd3c4d](https://github.com/jax-ml/jax/commit/44fd3c4d7c9432aa57a3c0cf91e597a466a1a432)）；[#42046](https://github.com/jax-ml/jax/pull/42046)（2026-10-28T00:19:17Z，默认分支@[5f881aa7](https://github.com/jax-ml/jax/commit/5f881aa7394b5965d0e815a19f690dd6d1ca5404)）；[#42010](https://github.com/jax-ml/jax/pull/42010)（2026-10-27T09:26:33Z，默认分支@[3041dc31](https://github.com/jax-ml/jax/commit/3041dc3152497e3b6c5b4b765d4077bcb3019d8b)）；[#41785](https://github.com/jax-ml/jax/pull/41785)（2026-10-27T19:11:52Z，默认分支@[8ad1d149](https://github.com/jax-ml/jax/commit/8ad1d14961da362961cbbbd95380a3ed48b8cddb)）。

HiJAX引入UnreducedArray并在LoJAX以额外mesh维表示部分结果，jit/device_put受hijax_unreduced控制；reshard到普通sharding才插sum/max/min，sharded/replicated转入以零补齐，不等同把通信无条件删除。Mosaic GPU在SM107且PTX ISA≥9.4时支持窄化FP4的.rz并跨两步转换传rounding；Pallas pad_ref在TPU降为memref_pad。**建议深入阅读并小规模复现**分片契约/舍入，均是源码支持，缺统一LLM训练收益证明。

源码入口：`jax/_src/array.py`；`jax/_src/api.py`；`jax/_src/pjit.py`；`jax/experimental/mosaic/gpu/fragmented_array.py`；`jax/_src/pallas/core.py`。

### MLX：非连续 logits 的 Metal cross-entropy 反向 fence 修复

**ml-explore/mlx；合入记录与基线：** [#4702](https://github.com/ml-explore/mlx/pull/4702)（2026-10-27T08:03:10Z，默认分支@[c9ea4f1e](https://github.com/ml-explore/mlx/commit/c9ea4f1e911b2955b206273e9e29dbe927dd5ceb)）。

非row-contiguous输入copy被donate成输出后，旧逻辑又把它作为temporary从encoder输出移除，下一command buffer可能在梯度写完前读；现保留输出跟踪而非登记短命temporary。适用于logits[:, :-1]等训练常用view及足够大触发commit的输入，不改变损失定义。**建议考虑采用并复测**跨command-buffer梯度，而不是只测小tensor数值。

源码入口：`mlx/backend/metal/cross_entropy.cpp`。

### MLX-LM：投机 logits processor 现在看完整 prompt 历史

**ml-explore/mlx-lm；合入记录与基线：** [#2014](https://github.com/ml-explore/mlx-lm/pull/2014)（2026-10-27T18:35:12Z，默认分支@[f39c4aa2](https://github.com/ml-explore/mlx-lm/commit/f39c4aa2af5d61a5e85bd50cc3f4294fedaff116)）。

speculative generation以prompt前缀初始化processor历史，且num_draft=0时不阻断历史增长，修复重复惩罚等处理导致有draft/无draft输出分叉。作者以Qwen3-1.7B/0.6B 4bit、repetition_penalty1.3示例和回归验证，不能推广到所有随机采样配置逐token等价。**建议考虑采用**并比较自己的processor、draft长度和无draft路径。

源码入口：`mlx_lm/generate.py`。

### TorchNPU：FlexAttention 保留 LSE 融合，并修复 ATen fallback 生命期

**Ascend/pytorch；默认分支commit：** [7dc26b7a](https://github.com/Ascend/pytorch/commit/7dc26b7adfd17baa151fa6317c3bc4fa4f3c4e8e)（2026-10-27T11:17:41Z）；[f72a49f1](https://github.com/Ascend/pytorch/commit/f72a49f161b0e5ec001d105c4f1b46dc0bd5c0a4)（2026-10-24T07:58:41Z，此前已报，仅作基线）。

此前已报的前向存储/反向消费者LSE换底融合与局部ATen fallback作为基线；本轮新增修正原地fill_/mul_在C++ wrapper的schema dispatch、保持被alias的位置表活到反向结束、处理非连续梯度缩放。不能只收录先前fusion而漏掉配套修复。**建议考虑采用并验证**NPU eager/compile及非连续输入，来源仅声明功能验证，未给可比较的端到端性能。

源码入口：`torch_npu/_inductor/kernel/flex_attention.py`。

### TorchNPU：GraphTree 重录上限与 reset 恢复有效

**Ascend/pytorch；默认分支commit：** [bd26c622](https://github.com/Ascend/pytorch/commit/bd26c6222418c5109b112b7c3038d9aaabd24115)（2026-10-27T01:13:33Z）。

不再以已恒真的inline_inbuilt_nn_modules绕过rerecord limit；合法static input地址变化不耗异常预算，异常重录另计；内存池检查在GC后同步，reset生命周期配套强化。**建议考虑采用**于reduce-overhead长期服务，检查合法换权重与真正异常的计数、警告及资源释放；没有长期多模型实测就不承诺消除所有泄漏。

源码入口：`torch_npu/npu/_graph_tree.py`。

### TorchNPU：memory history 可按 action 过滤

**Ascend/pytorch；默认分支commit：** [2e45d03b](https://github.com/Ascend/pytorch/commit/2e45d03b2a5c8b4c1158d31df5f6f05eb5c5d9b6)（2026-10-27T10:48:43Z）。

skip_actions贯通Python、binding、主及workspace allocator，配置前校验名称，空列表/None恢复；过滤trace而保留实际segment/block状态和allocator tracker callback。**建议考虑采用**于长跑显存记录，降低保留事件量时仍需保存过滤配置，否则不同snapshot不可直接比较；未给记录开销基准。

源码入口：`torch_npu/csrc/core/npu/NPUCachingAllocator.cpp`。

<a id="hardware"></a>

## 硬件与软件协同

### CUDA 13.4 显存边界接入 Kubernetes：调度与节点强制执行各负其责
**NVIDIA；主窗新官方教程10-27 20:50 UTC，不是本轮新CUDA发布。** [正文](https://developer.nvidia.com/blog/memory-enforced-gpu-sharing-in-kubernetes-with-nvidia-cuda-13-4/)。CUDA提供对应用透明的显存上限，以及保证request、空闲可突发到limit的区间；GPU Fractioning Operator在节点启动/监督MPS、设置容器限制并采集fraction指标。KAI Scheduler从v0.18.0起以NvFractions选GPU、写每容器显存/compute-mode/device annotations；其他调度器也可写同一接口。

上限省略时等于request；多GPU的request按**每卡**应用，不能误当总量。time-slicing为默认且各工作负载独立错误域，SM-sharing并发但共享错误域；同一GPU不能混两模式。文章的部署例使用旧CUDA容器tag，说明容器tag本身不能证明节点具备新驱动能力；本文仅核官方教程与release notes，没有部署K8s验证，也没有端到端利用率提升基准。建议复测超限分配失败、burst回收、故障隔离与监控口径。

稳定 ID：`nvidia:cuda134-gpu-fractioning:article:2026-10-27`。

### DGX Spark 的 MoE 三项优化：真实 GSM8K 增益42.5%，合成负载185%不能外推
**NVIDIA；主窗新教程10-27 15:00 UTC。** [官方原文](https://developer.nvidia.com/blog/3-ways-to-boost-moe-inference-on-nvidia-dgx-spark/)。Ling-3.0-flash-fp4在batch1受读权重带宽限制，组合Humming MXFP4专家kernel、在线FP8 LM head与内置NEXTN/MTP3。32K输入/2K输出greedy合成负载由 **36.09→102.88tok/s**，AL3.90；完整1319题GSM8K、5shot、temperature0.6/top-p0.95/top-k29、max32768时，仅 **37.81→53.89tok/s（+42.5%）**，均值延迟15.21→11.49秒，TTFT0.207→0.214秒，准确率96.89→96.82%。

合成输出重复、采样和长度分布不同，不能把+185.1%当业务预期；GSM8K优化后平均输出也由567增到607，吞吐/延迟须同时看。**FP8 LM head与MTP必须共享整个head模块，只共享weight会把FP8当BF16，AL降到1并慢于基线**；这是比标题倍数更值得核查的集成边界。作者称准确率无统计可检测变化，本轮未独立复现、多seed或一般代码/agent任务验证。建议先确认startup实际启用路径，再测自己的输出熵与并发。

稳定 ID：`nvidia:dgx-spark-ling3-moe:article:2026-10-27`。

### AMD 给出 EAGLE3.1 从训练到服务的完整说明：提高吞吐也会增加 TTFT
**10-27新官方文章；草稿权重最早7月23、当前卡9月1已存在，不当新模型发布。** [文章](https://rocm.blogs.amd.com/artificial-intelligence/eagle3-training/README.html)、[固定草稿卡 `cadf8c9f`](https://huggingface.co/amd/MiniMax-M3-EAGLE3.1/tree/cadf8c9f22b9447fac65da2314baec6327a4bb79)。流程在单8×MI350X节点，把target的on-policy输出、多层hidden-state提取、FSDP2/TorchSpec冷启动训练、vLLM导出和SPEED-Bench连起来；EAGLE3.1是在head增加normalization的一层BF16草稿，target是MXFP4 MiniMax-M3。Quark0.13工作流仍标experimental；Qwen3-8B为validated quickstart，MiniMax大模型recipe需要替换数据manifest占位和许可。

作者vLLM ROCm0.27.1/native AITER、TP8、draft3、temperature0、并发48、max model12288/output2048的两次qualitative均值，吞吐/GPU **281.7→415.2tok/s（1.47×）**、E2E中位13.15→9.68秒，**TTFT130→154ms（+18.5%）**；AL2.81不是2.81倍实测加速。1K–32K AL另为单次完整split，未证明任意请求长度/并发恒定收益。HF标签MIT但卡末明确引入MiniMax Community License，附件正文含商业通知/授权条件，不能只凭标签认定无条件MIT商用。建议复测chat-template一致性与目标量化版本；训练token、精确耗时/算力未充分披露。

稳定 ID：`amd:eagle31-training:article:2026-10-27`。

### AMD warp-pipelining：用跨 wave 的阶段偏移隐藏准备工作，结果主要是接近手工优化kernel
**10-27新官方技术说明。** [正文](https://rocm.blogs.amd.com/software-tools-optimization/triton-warp-pipeline/README.html)。Gluon阶段标注让两个wave组保持一阶段偏移，用barrier协调执行、用单独wait保证LDS跨组和跨迭代依赖；独立ring buffer分开allocation可减少编译器误判依赖导致的多余等待。它与双缓冲、priority和warp specialization控制的是不同问题。

MI355X FP16 4096×4096×8192，8wave **1437TFLOP/s**对手调4wave1425，作者解释为**持平**；不是受控仅切换pipelining的消融。MI450三种dtype也组合cluster multicast/LDS布局/TDM，MXFP8/4约持平hipBLASLt，BF16反慢3.3%；in-loop matrix efficiency不能当整kernel或标称峰值利用率。文章引用持续变化的upstream Triton main与示例，未固定可直接重建全部测量的commit。没有FlashAttention性能测量；带宽受限decode、短compute阶段和register spills可能不获益。适合深入读wait位置和IR/ISA，不宜概括为新通用LLM加速倍数。

稳定 ID：`amd:triton-warp-pipeline:article:2026-10-27`。

### ROCm 10.0 / Ray 2.58：官方验证组合与多卡服务示例，移植边界仍在kernel与部署
**AMD；10-27新支持说明。** [正文](https://rocm.blogs.amd.com/artificial-intelligence/rocm10-ray/README.html)。文章给ROCm10.0.0、Ray2.58.0、PyTorch2.12、vLLM/ROCm的验证组合，Instinct自动发现，`torch.cuda`映射HIP；Ray任务/placement group/Serve接口不需要另写厂商分支。原生SKU约束覆盖MI300X/MI325X/MI355X，样例用Qwen2.5-0.5B、TP2/RCCL、真实chat completion验证服务链。

这是普通Ray Python API层的兼容与教程，**不能推出任意CUDA扩展、现有容器和性能调参均可直接迁移**；文章没有受控吞吐或训练扩展曲线。混合GPU集群中`accelerator_type=None`可落错硬件，错误SKU字符串则无法调度；需至少两可见GPU、较大`/dev/shm`及头维整除TP。未运行本地设备验证，建议安装前核官方支持矩阵、真实模型的collective与kernel路径。

稳定 ID：`amd:rocm10-ray258:article:2026-10-27`。

<a id="evaluation"></a>

## 评估与可靠性

### InfraDev：AI 基础设施补丁需要数值与资源行为的隐藏测试
**2026-10-27 v1；主窗新作；`arxiv:2610.36589`；机构/团队：University of Edinburgh／腾讯混元。** [论文全文](https://arxiv.org/html/2610.36589v1)。benchmark 从 Megatron-LM/verl/vLLM 人工选择 36 个已合入、改动至少千行且有测试的 PR，并联合生成任务说明、对抗错误补丁和隐藏 probes；隐藏测试不仅检查输出，还检查优化是否减少内存、通信或时间（§3）。

六模型 resolved 为 25.0–38.9%，虽然平均通过 74.3–83.3% 测试；十个推理任务上各模型都只完成一个（Tables 2–3）。原 PR 中 24/36 缺数值断言，13/15 性能 PR 缺资源收益检查。小而专家筛选的任务集限制代表性，每次评估需多 GPU 和大量工时；隐藏 probe 是合成后对金补丁/对抗补丁验收，仍应人工复核其规格。**实践：工程 agent 的验收应包含数值、执行模式、生命周期和资源收益，避免拿测试通过率代替任务完成率。**

### BIEvolve：kernel 通过容差正确性，也可能依赖同批其他样本
**2026-10-27 v1；主窗新作；`arxiv:2610.36623`；机构/团队：George Mason University STAR Lab／NUS。** [论文全文](https://arxiv.org/html/2610.36623v1)。用 DSL 约束 reduction partition、recurrence、RNG 等不得依赖 batch 上下文，再动态改变 batch 组成验收，作为 kernel-agent 搜索反馈（§3）。

H100 NVL、固定生成预算、三个 agent 框架的风险定向任务集中，batch-invariance pass 从总体 67.6% 提升到 91.9%；相对任务 reference 的几何平均加速由 28.91×降至26.54×，保留91.8%（Table 2）。常规 correctness 仍约95%，表明两者衡量不同属性。静态跨文件分析有限，剩余违规未清零；大倍率对的是各任务 reference，非整模加速，尚未测对端到端生成/RL的影响。**实践：独立样本语义的算子应增加 batch 组成扰动验收。**

### API-TAIL：API 模型替换审计要把合法部署差异写进零假设
**2026-10-27 v1；主窗新作；`arxiv:2610.36801`；机构/团队：TokenRhythm Technologies。** [论文全文](https://arxiv.org/html/2610.36801v1)。用本地 claimed-model 权重对返回文本打分，先对齐有效温度，再统计异常 tail；校准池覆盖合法温度、截断、penalty 与system prompt，因此无需供应商提供logprobs（§2–4）。

两模型、26个“plain”真实endpoint、200请求与10%合成替换混合中，四面板power为0.33/0.69/0.89/1.00，最大false-alarm为0.035/0.002/0.003/0.069（Table 1）。这不是发现这些供应商实际作弊；替换审计由真实响应离线混合，special endpoints另算，同族低差异替换存在接近零power的设置（Table 2）。需要claimed模型权重及校准池覆盖，0.10有效性门槛也不是零误报。**实践：可作开放权重API抽检方案，不能凭一次异常定性供应商替换。**

### Beam 新增评测指南：先核 endpoint 语义与推理内容保留，再比较榜单
**Reflection；主窗10-27 20:35 UTC新增 `EVALUATION.md`，`24595dc9`，不是新权重发布。** [固定指南](https://huggingface.co/reflection/Beam-1.0-501B-A23B/blob/24595dc986af66da3e40229d46aa5f6d1fd2d0c7/EVALUATION.md)。指南把评测差异落到可检查配置：`reasoning_effort=max`、temperature0.7、top-p0.99，一般输出32768，SWE-bench先65536、截断再升131072；对话必须原样保留 `reasoning_content` 与 tool calls、每个调用配对应tool结果。Beam Verifier先查streaming/tool/multiturn，再测GPQA/AIME；SWE示例用Mirror/bash-only、原始repo commit、一次尝试并计入缺失/失败。

这是重建评测条件的文档增量，未提供新的受控分数或独立复现；指南中“少约3倍reasoning tokens”的宣称缺少这里可重建的实验，应保持作者口径。比较BF16/FP8/NVFP4需分别用对应参考分数、保存服务端与模板版本；优先检查截断、重试和API配额是否被误计为能力错误。

稳定 ID：`hf:reflection/Beam-1.0-501B-A23B:evaluation-guide@24595dc986af66da3e40229d46aa5f6d1fd2d0c7`。

### Anthropic：随机赋给 agent 的分析偏好，能明显改变专家的判断
**官网只标10-27日期；元数据的00:00 UTC不足以认定精确首发时刻，无法可靠判断相对本轮09:00:54起点的归属。本轮首次收录，保留日期边界不确定性。** [官方正文](https://www.anthropic.com/research/agents-judgment-calls)、[论文PDF](https://www-cdn.anthropic.com/519c17a5d23a794e8b6d50a854048d0b61e0d73a.pdf)。61名通过因果推断筛选的专业分析师中，50人使用Claude Code/Opus4.8 High做虚构农业政策分析；每人四个有合理分歧的分析选择与最终建议，由隐藏系统提示逐项抛硬币赋予agent立场。作者观察四项选择合计 **86%随agent立场**，逐项随机立场的组间选择差 **62–82pp**；这衡量的是被强化并主动表达的偏好对选择的影响，不是86%的“答错率”。

48份事后问卷里77%至少一次复述agent论据、15%提到来源；预先表达过偏好的38个案例中，agent反对时60%改变、赞同时6%，差的95%CI为24–82pp且选择子样本不能作同样强因果解释。无AI的11人系自选且较少经验，55%对6%的严重错误率和90对65分钟不能证明随机分配的AI生产力收益。仅一任务、一模型、陌生农业领域，未测自然状态agent偏好的总体效应。建议在研究harness中记录关键判断的备选项、先验观点与来源，复测是否影响人类审查的独立性。

稳定 ID：`anthropic:agents-judgment-calls:2026-10-27`。

### Ai2 发布 novelty judge 研究说明：提示改几字，结论可翻转过半
**10-27新官网研究说明；对应论文2610.02022首发10-01，不能写成今日新论文。** [官方正文](https://allenai.org/blog/novelty-bench)、[论文记录](https://arxiv.org/abs/2610.02022)。从ICLR2026评审挖出高novelty154、低novelty145个人类研究想法，再与简单LLM生成的154条配对。相同Human+Generated对里，GPT-5.4提示从“恰有一条被评审认定novel”改成普通比较，作者准确率 **93.5%→40.9%**；Human-Only变化较小，表明仅在人类论文上验证judge可能掩盖生成想法场景的不稳。

“生成想法较不新颖”是弱标签；额外专家审查仅30对且偏重最强judge错例，并非全部随机标注。更多检索/高推理预算未稳定改善，专用系统与便宜prompt baseline的调用次数/投票不同，成本与F1不能脱离该设置。新官网宣布数据/代码，但未核到主窗中独立的新HF数据工件版本，按**官网证据说明**而非新数据版本处理。建议对自动研究系统做prompt敏感性、无标签成对翻转率与跨judge稳健性测试。

稳定 ID：`allenai:novelty-bench:official-blog:2026-10-27`（关联旧作品 `arxiv:2610.02022`）。

<a id="coverage"></a>

## 版本更正、实际覆盖与未完成工作

### 版本与补录

HyScale `2610.32057` v1→v2（10-26）：正文段落与表格未发现差异；不重复列作新成果。ALTO/MXFP4 `2610.32229` v1→v2（10-26）：硬件明确为MI300X、仍是quantize/dequantize模拟而非native MXFP4；表格不变，删去“尚未完成正式MLPerf提交”的一句**不能反推已经提交或通过审计**。LAW `2610.30068` v1→v2→v3（v3在10-27）：增加/澄清扩散bridge参数化、Maze数据及评分说明，v2→v3为记号和引文改写，实验表格不变；没有按同名论文再造一项新算法。相关全文与差分已保存，公式/图片并未逐资产审计。[HyScale v2](https://arxiv.org/html/2610.32057v2)、[ALTO v2](https://arxiv.org/html/2610.32229v2)、[LAW v3](https://arxiv.org/html/2610.30068v3)。

本报32项首次论文收录中，16项主窗v1、15项72小时回溯补录、1项超窗旧候选Target Regeneration（10-21）。INTRA按v2核查，v2仅加代码链接；旧论文对应的新官网说明和新工件分别标增量，未重复记为新算法。OPTIC已在10-24实质报道，本轮去重剔除。vLLM PCP draft-logits guard、NIXL1.5.1→1.5.0回退与Triton同日撤回等状态收缩在相应正文明确列出。

### 来源检查范围

| 来源 | 本轮实际检查 | 仍未完成或失败 |
|---|---|---|
| 核心六仓 | 72h回溯623个all-base合入PR、598个默认main commit、254个公开release对象，均分页；主窗222个PR | 61组/88个新稳定PR选择性语义审阅；保留296项当前未选候选及旧569/447队列，队列有交集；开放依赖与3个非main观察单列；无本窗新公开release |
| 扩展24仓 | 1183个回溯all-base合入PR（主窗524）、1331个默认分支commit、1722个release对象 | 56组/106个稳定事件，事件不等于独立成果；1091个未选回溯候选保留。23仓release完整分页，FlashAttention100项入口Transport closed，仅5对象回退可读，不能冒充全页 |
| 模型/数据 | HF41组织×models/datasets×createdAt/lastModified共164查询到回溯边界，195去重候选；官方组织卡/配置/commit与重点数据/许可另审 | 取得卡不是代码、权重张量或全量数据验真；初次批量92失败请求含正常不存在文件，恢复5仓定点访问，剩余429/401/404与动态站点单列 |
| 官网/硬件 | 62官网入口、27重点正文/报告/许可直接来源；中国与海外厂商及NVIDIA/AMD/Intel/Ascend/TPU/Neuron均有实际检查记录 | 入口快照不替代组件历史枚举，动态页、访问失败、CANN/HCCL/libtpu与旧区间保持待补查 |
| 论文/正式学术 | 7类arXiv公告、API提交窗口、替换元数据、HF论文列表和重点全文；详情见下文 | 全局更新时间API被重写、OpenReview403、ACL/PMLR各卷与TMLR精确日期不全，不能宣称全域覆盖 |

七类arXiv（cs.CL/LG/AI/DC/PF、stat.ML、cs.CV）的10-27公告共1,731个稳定ID：1,071新/跨投、660替换。主窗submittedDate恢复184条；回溯提交页共1,233条且补齐第二页，精确日期过滤后数量不变。660替换的当前版本元数据全部恢复，但仅对四项作品做实质文字/表格版本比较，**其余替换仍待审**。`lastUpdatedDate`请求依然被服务重写成`submittedDate`，所以真实全局更新时间窗没有闭合。

HF10-26/27列表为42/50项，10-28接口400不代表当天无论文。OpenReview API仍403；ACL/PMLR入口可读，但没有完成所有volume的版本/日期遍历。本节32项首次实质收录为16个主窗v1、15个回溯作品（INTRA按v2核查）与1个旧候选补录；独立复现0。旧候选、594条继承替换、上轮292条未差分替换及本轮未审候选均保留在台账/候选文件中，队列有交集不可相加。没有将列举成功或全文下载成功当作语义覆盖完成。

### 额外入口与失败说明

- Google Research 10-27的[Sleep新官网解释](https://research.google/blog/the-sleep-framework-a-new-way-for-llms-to-retain-knowledge-in-continual-learning/)对应[2606.03979](https://arxiv.org/abs/2606.03979)，原文记录首发6-2、v2修订7-10；是新说明，不是今日新算法。未将10M tokens/数学分数从旧论文包装为本窗新实验。
- Anthropic 10-27的[六机构药物开发案例](https://www.anthropic.com/research/early-progress-drug-development)已读官网：主要为早期、临床前合作；未逐篇核其技术报告，不把分子/文档案例包装为训练架构、医疗有效性或独立因果评测突破。
- NVIDIA 10-27[本地规则后训练教程](https://developer.nvidia.com/blog/fine-tuning-nvidia-nemotron-to-know-local-rules-not-just-local-languages/)已读，固定教程仓tag v0.2.0、persona+官方程序文本生成→LoRA→merge→评测；没有受控质量汇总。它不证明persona本身提供事实，也不证明公平性或新权重发布，留作实践参考。
- 已查询41个组织的models/datasets、createdAt/lastModified共164个HF入口并翻页到回溯边界；195个去重候选。批量卡/commit抓取后段触发429，少量401/404；腾讯Hy-Nomas/WeVisOne2B、Beam与AMD草稿已定点补抓恢复，其他缺口留账本。取得卡不等于完成代码/张量/数据行审计。
- 中国厂商入口实际检查覆盖DeepSeek、Qwen、Seed、Kimi、GLM、MiniMax、腾讯、百度、阶跃、小米、美团；Qwen/腾讯/百度/阶跃/Ascend动态页面正文不足，MiniMax API分页和多入口内容不一致仍未关闭。OpenAI部分官网403、Microsoft Research timeout、xAI news403、RecursiveSI TLS失败；不能据此声称这些厂商没有更新。
- 硬件实际检查了NVIDIA、AMD及Gaudi1.24.2、TPU release notes、Neuron2.33.0入口和Ascend公告；Gaudi/TPU/Neuron这里只是当前入口快照，CANN/HCCL/libtpu、组件历史版本及旧backlog仍未全覆盖。全局窗口完整覆盖标记应继续为空。

### 持久化与下一轮衔接

本地 `.monitor/research-2026-10-28/`保存一手快照、逐源检查、稳定事件账本、候选与独立QA；`.monitor/state.json`与`.monitor/sources.json`记录来源游标、历史缺口、版本和同步状态。本轮保留全部旧backlog并追加当前未审区间，不以文件mtime或当前聊天恢复进度。**全局完整语义覆盖水位继续为空**；下轮从2026-10-28T01:02:09Z续扫并回溯72小时，优先重试失败源及未审队列；若上传失败，先补传本轮已保存正文。

本轮独立复核补齐：DeepSpeed激活微测的H200与额外峰值口径、TE/TRL版本与compile边界、ADP connector的PP/CP及fatal边界、Ascend更新协议和overlay验证、PyTorch ghstack真实落地commit与时间。所有数字均保留来源范围，未以自动测试清单代替独立训练复现。
