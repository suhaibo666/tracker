# LLM 技术动态日报 · 2026-11-17

> 时区 Asia/Shanghai（UTC+8）。增量扫描窗口 **(2026-11-15 08:57:48, 2026-11-17 09:01:04]**，向前回溯 72 小时至 **2026-11-12 08:57:48**。实际生成 **2026-11-17T09:21:12+08:00**。来源仅有日期时保留原日期，提交、发布、会程与发现时间按各自证据记载。

> 11 月 16 日留有未归档的取证。本期从两端最后一致的 11 月 15 日截止点续扫，并重新核对 11 月 16 日已留存的固定版本。

## 本次重要变化

- **模型。** Qwen4-Max 正式版公开 2.4T 主干、额外 328B N-gram 嵌入、71B 激活和 1M 上下文，并更新地域与 API 契约。 [Qwen 技术披露](https://qwen.ai/blog?id=qwen4-max) · [接口文档](https://help.aliyun.com/zh/model-studio/qwen4-max)
- **训练。** LACCO 的外同步重叠收益从 0.5 Gbps 下 1.89× 降到 5 Gbps 下 1.07×，对照均为作者在 4 张 A100 上的每轮用时。Megatron-LM 修正 MTP 跨 CP 归一化，并合入可选主模型蒸馏。 [LACCO](https://arxiv.org/abs/2611.17991v1) · [Megatron #7498](https://github.com/NVIDIA/Megatron-LM/pull/7498) · [Megatron #8639](https://github.com/NVIDIA/Megatron-LM/pull/8639)
- **推理与升级。** vLLM 0.34.0 正式分发，与 RC4 指向同一提交。主线另有 Mamba 缓存状态与 prefill 合并；SGLang 增加有界前缀调度，并撤回一组合入过的 Blackwell/Rubin 优化。 [vLLM 版本比较](https://github.com/vllm-project/vllm/compare/v0.34.0rc4...v0.34.0) · [Mamba #66938](https://github.com/vllm-project/vllm/pull/66938) · [SGLang #48409](https://github.com/sgl-project/sglang/pull/48409) · [撤回 #50703](https://github.com/sgl-project/sglang/pull/50703)
- **系统评估。** MLPerf Training v6.1 首次发布 Qwen3.5-397B Agentic RL 的达标时间。开源复现场景值得优先看 rollout、权重同步与完整评估的墙钟占比。 [MLCommons 发布](https://mlcommons.org/2026/11/mlperf-training-v6-1-results/) · [结果表](https://results.mlcommons.org/training)

[模型与技术报告](#topic-1) · [预训练与架构](#topic-2) · [后训练](#topic-3) · [训练系统与数值一致性](#topic-4) · [推理与部署](#topic-5) · [开源工程](#topic-6) · [硬件与软件协同](#topic-7) · [评估与可靠性](#topic-8) · [覆盖与续查](#coverage)

<a id="topic-1"></a>
## 模型与技术报告

### Qwen4-Max 正式上架，披露 2.4T 主干和新的输出预算

**阿里巴巴 · 11 月 16 日，正式 API ID `qwen4-max`。** Qwen 团队给出 2.4T 主干、每 token 激活 71B，另有 328B N-gram Embedding；阿里云新闻稿将 2.4T 简称总参，这里按技术正文的参数口径。每四层有三层 Gated DeltaNet 压缩历史为固定大小状态，一层 Qwen Sparse Attention 用轻量索引选取上下文块；四路门控残差用于跨层信息传递，N-gram 在第二层按局部上下文查表。厂商称预训练未见 loss spike，四类内部语料较 Qwen3.8-Max-Base 的 BPB 低约 5–13%，激活参数少约 25%。该对照包含多项同时改动，未给出逐机制消融、完整训练配方或 Max 权重。

正式版仍支持文本、图像和视频输入及 100 万 token 上下文。思考和最终回答的共享上限由 Preview 的 393,216 调至 262,144，默认思考预算 131,072，强度 `high`。百炼中文页 11 月 16 日 09:28:58 UTC 的修订将所列六个地域的批量推理均标为不支持。Artificial Analysis 的 v5.0.1 九项综合为 43 分，按产出长度与端点速度估算约 32.1 分钟/题，缺同协议的 Preview 对照。**建议深入阅读**架构和预训练部分；迁移时先固定地域和思考预算，复核长输出截断及工具往返成本。 [Qwen 技术正文](https://qwen.ai/blog?id=qwen4-max) · [阿里云新闻稿](https://www.alibabacloud.com/en/press-room/alibaba-unveils-qwen4-max?_p_lc=1) · [百炼正式文档](https://help.aliyun.com/zh/model-studio/qwen4-max) · [预览版参数](https://help.aliyun.com/zh/model-studio/qwen4-max-preview) · [独立评估](https://artificialanalysis.ai/models/qwen4-max)

### NVIDIA Sigma 发布连续扩散语言模型，随后重导出基座与指令权重

**NVIDIA · 11 月 12 日报告 v2 与模型卡发布日，11 月 16 日 17:01–17:02 UTC 更新。** 3B/8B 档使用共同学习的 16 维 token embedding，在连续潜空间逐块去噪，可复用已完成块的 KV。官方集合列出 14 个检查点，涵盖 base、instruct、NFE 2/4/8/16 和单步自投机。报告在自身协议中给出数学约 8、代码约 4 token/forward 的低步数蒸馏结果，8B 蒸馏基座的自投机在 GB200 上相对 AR 基线约 3.5× 吞吐。11 月 16 日两档 base/instruct 重导出以保留 FP32 几何张量，8B instruct 的最后一片权重哈希和字节数改变，生成配置仍以聊天 EOS ID 11 为准；旧结果不能直接归到新工件。**建议深入阅读并考虑复现**，固定权重、词表、去噪步数与提示协议，再比较精度、端到端吞吐和多样性。 [官方集合](https://huggingface.co/collections/nvidia/nemotron-labs-sigma) · [报告 v2](https://arxiv.org/abs/2610.02665v2) · [8B instruct 固定差分](https://huggingface.co/nvidia/Nemotron-Labs-Sigma-8B-instruct/commit/e5aab187fd3312cdaebe3335e29a757eb52470ba) · [3B base 固定差分](https://huggingface.co/nvidia/Nemotron-Labs-Sigma-3B/commit/7eff02e0b72dace46d9084430330c192bf5a4386)

### LongCat-Video-Avatar 1.6 把续写误差带入后训练

**美团 LongCat · 11 月 15 日 02:57:51 UTC，#156 合入，MIT 权重在 11 月 13 日上传。** 在冻结视频骨干上分别训练两个 LoRA。GRPO 用感知奖励优化口型和动作；另一分支以 DMD2 将去噪从 50 步减至 8 步，再以学生前段生成帧作为续写上下文进行 on-policy 自蒸馏。EMA 教师额外读取真实续写上下文和一张上一段生成的边界帧，为同一去噪状态提供监督。报告展示超过 10 分钟的生成。在 508 个图像与语音对、六系统 33,051 张有效偏好票的发行方评测中，1.6 排第二，对 1.5 的平票折算胜率为 55.6%，对 OmniHuman-2.0 为 24.7%。商业 API 失败使各对照样本数不一，偏好榜与长视频演示采用不同评估设置。完整训练数据和复训工件待补。**建议深入阅读** §2.3、§4 和图 8，并在固定显卡和长度下复验口型、颜色漂移和耗时。 [合入 #156](https://github.com/meituan-longcat/LongCat-Video/pull/156) · [固定技术报告](https://github.com/meituan-longcat/LongCat-Video/blob/58eb336edc5c8c63389838727e40b8c18434e54e/assets/LongCat-Video-Avatar-1.6-Tech-Report.pdf) · [固定权重与卡片](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.6/tree/acfc628244201fc53fb6fe8a8903c5bac9bdf68f)

### Quark-150M 公开 128K 基座及直接 GRPO 的 Zero 权重

**Empero AI · 11 月 15 日 16:56/16:58 UTC，Apache-2.0。** 165.3M 总参，16 层中 12 层以 256-token 滑窗并联短因果卷积，4 层做全局 GQA。基座从近 496B token 的 32K 检查点继续以 10B token 的 131,072 长度序列训练，将全局层 RoPE base 提至 4M。Zero 从基座直接进行 GRPO，8×H100 上不足 1,300 步，约 95% 奖励组由程序判分。内部 15 类任务、每题 4 次采样，正确率由第 50 步的 12.8% 到最终 36.4%。留出基准和完整训练工件待发布。**建议考虑复现**，分别测 32K/128K 召回与显存，以及基座到 Zero 的留出任务质量。 [固定 Base 卡](https://huggingface.co/empero-ai/Quark-150M-128K-Base/blob/e670e11de3775818a19f67ac0a9a49fdefb4dd7f/README.md) · [固定 Zero 卡](https://huggingface.co/empero-ai/Quark-150M-128K-Zero/blob/7c1599b0764fb9a8ee0eb3b9396e1f07f1e1b144/README.md)

### Supra2.5-Pro 给出 65M 基座的 100B-token 小模型训练配方

**SupraLabs · 11 月 16 日，发布 Base 与 Instruct，Apache-2.0。** Base 是 512-token 上下文的 65M Qwen3 架构，使用自有 16,384 项词表、公开语料来源和末段 20B token 的配比切换。厂商同表五项平均 44.73%，旧 100M 基线 42.18%，主要变化包括 ArithMark-3 的 36.9%→49.6%，PIQA 和 HellaSwag 各自回落。对照同时改变数据和训练过程，训练流与留出评估还需独立复跑。**建议考虑复现**，先固定词表、训练量和评估版本，对比末段换配比前后的单项结果。 [固定 Base 卡](https://huggingface.co/SupraLabs/Supra2.5-Pro-Base/blob/4767bdeb947ed3b8965ae0685f18c9bdca91c2a0/README.md) · [固定 Instruct 卡](https://huggingface.co/SupraLabs/Supra2.5-Pro-Instruct/blob/78901d93df038d77d4253c48bdaffe8276e38879/README.md)

### 补录｜Salesforce SOV-Transcribe-1 可按术语与格式调整转录

**Salesforce Sovereign Intelligence · 11 月 12 日 16:43 UTC 官方公告，固定卡提交于 11 月 11 日 23:26 UTC。** 约 26B 总参、4B 激活，音频可随领域词表与自然语言格式要求一起输入，为 Apache-2.0 非流式 ASR。在 Contextual Earnings-22 的同批 15 秒片段上，厂商报告无关键词 F1 为 71.9，给片段关键词为 96.8，改用含干扰词的整场词表为 94.0。Artificial Analysis 的 2.28% WER 属于其非流式协议。完整训练报告待公布。**建议考虑复现**，以自己的真实词表对比无词表、准确词表和含干扰词表三组，另计关键词误插入和处理耗时。 [作者公告](https://huggingface.co/blog/salesforce-sovereign-intelligence/sov-transcribe-1) · [固定模型卡](https://huggingface.co/salesforce-sovereign-intelligence/SOV-Transcribe-1/blob/505d8fae243924c0d8acdc622f7f0abab344eba3/README.md)

<a id="topic-2"></a>
## 预训练与架构

### 主窗｜LACCO 用两套状态将 DiLoCo 外同步移出本地计算的关键路径

**MLIA、Mila、蒙特利尔大学，arXiv 2611.17991 v1，11-13 10:21:28 UTC 投稿，11-16 公告批。** 每个副本维护两组参数及对应优化器，在 lookahead 与 main 阶段互换角色。各副本在一组上做本地更新时，另一组完成全局聚合。外更新应用同一份聚合梯度，每轮仍按相同 token 预算发生通信。FineWeb 上 134M 模型、4 副本、每 30 个局部步通信时，验证 loss 为 2.988，优于 Eager/Delayed DiLoCo 的 3.020/3.117，同步 DiLoCo/DDP 为 2.961/2.936。在 4 张 A100 80GB、0.5B 模型的实测中，0.5、1、5 Gbps 下每轮速度为同步 DiLoCo 的 1.89、1.44、1.07 倍。时间收益随通信占比变化，两组状态也带来显存或 CPU offload 代价。按实际带宽与局部步数画等 loss 的时间曲线，并检查峰值显存，最容易判断方法对现有集群的价值。论文所列实现本轮返回 404。 [论文 §3.3、§4、表 2–4](https://arxiv.org/pdf/2611.17991v1#page=8) · [版本记录](https://arxiv.org/abs/2611.17991v1)

### 主窗｜Muon 的短期早衰减在移除后仍改变更新幅度

**Meta、Mila、蒙特利尔大学等，arXiv 2611.17876 v1，11-13 08:05:19 UTC 投稿，11-16 公告批。** 研究将初始化、数据顺序和学习率配对，改变权重衰减窗口。FineWeb100B 无 warmup、约 20 token/参数训练时，最早 10% 窗口在 162M–3.4B 优于同表基线。1.4B 使用从 162M 迁来的系数 0.15、首 44% 窗口，终点 loss 为 2.6479，恒定衰减最佳采样值为 2.6743。162M 的固定范数回放显示，移除衰减后才采用早衰减实验测得的相对更新步长，已解释 0.0221 nats 回放收益中的 0.0181 nats。适合复现同一训练终点与等后续步数的两种比较，并记录投影更新除以权重范数后的轨迹。当前结论来自测试过的模型和优化器配方，窗口排序随评估时刻改变。 [论文 §3–6、图 2/5、表 1](https://arxiv.org/html/2611.17876v1#S5.SS3) · [版本记录](https://arxiv.org/abs/2611.17876v1)

### 主窗｜Concertina 折叠相邻层，保留逐步 KV 槽位和可训练差分

**北京大学、Georgia Tech、清华大学、Lovart AI，arXiv 2611.18113 v1，11-13 12:36 UTC 投稿，11-16 公告批。** Qwen3-4B 的 36 层折为 18 个各执行两次的块。每个块用首层线性权重作共享底座，给层间差分做 rank-64 截断 SVD，token 门控按 token 混合差分，RMSNorm 仍逐步独立。2.27B 学生蒸馏 22B token 后，GSM8K/MATH 为 78.8/49.8，教师为 87.9/56.6。按同一 20B 训练配方对照，2.31B 截层模型的 GSM8K/MATH 为 1.7/3.8，折叠学生为 71.8/40.8。公开代码 `FoldedLinear.forward` 混合差分，`LoopBlock.forward` 逐步切回原层的 cache 索引。压缩的是权重，运行深度和 KV 槽位仍保留。读表 2 的同预算实验后，再固定输出质量比较显存与带宽，以区分模型容量收益和单 token 时延。 [论文 §3、§4.1–4.4、表 1/2](https://arxiv.org/html/2611.18113v1#S4.T2) · [版本记录](https://arxiv.org/abs/2611.18113v1) · [固定代码](https://github.com/myboker/concertina/blob/64d303029947b83c3011c2c2262d7c185cab7f4c/concertina.py) · [Apache-2.0 权重与配置](https://huggingface.co/myboker/Concertina-2.27B/tree/5eea7f216e269062110331da8b9862c2ef858bee)

### 主窗｜V-JEPA-LM 用反事实视频对训练时间变化读出

**汉阳大学，arXiv 2611.17831 v1，11-13 07:05:22 UTC 投稿，11-16 公告批。** 冻结 V-JEPA 2.1，视频按 32 帧采样，固定 PCA 旋转与缩放后经可训练投影输出 1,024 个视觉 token。字幕和问答之外加入原片与变换片的反事实监督，同一问题因速度、方向或事件顺序改变而有不同答案。在 2B、相同混合配方及终点的对照中，V-JEPA 上 12 项均分由 61.53 到 66.02，SigLIP 由 58.52 到 62.13。仅加时间反转题的 V-JEPA 为 61.17。未见速度比的 3 选 1 探针在 V-JEPA 组由 33.5 升至 91.5，随机线为 33.3，此探针是构造数据。发布实现 `JepaLlmModel.encode_video` 构建视频表示，`SpectralBridge.forward` 拼接经谱变换的区域特征及坐标时间信号再投影。值得复现 encoder × 监督的四格消融，并另测真实视频问答的能力转移。 [论文 §3–5、表 3/4/6](https://arxiv.org/html/2611.17831v1#S5.T6) · [版本记录](https://arxiv.org/abs/2611.17831v1) · [固定代码](https://github.com/kkt94/Llama-V-JEPA-LM/blob/c3f9d8715aef484acab3cdf15673c9b5424de2a1/llama_vjepa_lm/modeling_jepa_llm.py) · [Qwen3-4B 工件](https://huggingface.co/kkt94/Llama-V-JEPA-LM-Qwen3-4B/tree/8bc98e74c5a13d0e74133564a2d25d0bfcd6e60a)

### 主窗｜LeCat 将类别记忆、泛化与随上下文调整的能力分开测量

**卡内基梅隆大学、Bar-Ilan University，arXiv 2611.18419 v1，11-13 16:59:46 UTC 投稿，11-16 公告批。** 先在 Pythia 与 OLMo 3 的已知训练语料中，区分直接重复、实例化、加强语境与例外等表达，再在 TinyStories 中注入两组虚构类别-属性对，从零训练 68M GPT-2 做受控比较。完整混合数据在概率排序约束上总体通过率 0.51，直接重复为 0.47。重复组容易保存默认关联，加强语境高于默认语境的通过率只有 0.03，完整混合为 0.37。两组虚构关系及单一小模型限制因果结论的外推，开放模型部分是相关性证据。适合把类别先验与例外响应分开测，检验扩充事实语料时是否增加了仅会重复的样本。 [论文 §3–5、表 1/2](https://arxiv.org/html/2611.18419v1#S5.SS2) · [版本记录](https://arxiv.org/abs/2611.18419v1)

<a id="topic-3"></a>
## 后训练

### 主窗｜TIDE 将同步 on-policy RL 的 barrier 推迟到权重更新

**华盛顿大学、布朗大学、UC Berkeley、纽约大学，arXiv 2611.17924 v1，11-13 09:03:23 UTC 投稿，11-16 公告批。** 长尾 rollout 仍用同一轮策略，已完成轨迹先进入前后向计算，所有 rollout 完成后统一更新权重。系统把剩余 rollout 迁入少量引擎，按 KV 容量否决不合适的迁移，将空卡陆续用于训练。Slime/SGLang/Megatron 在 8×H200 的 Text2SQL 与 DAPO-Math 上相对同步共卡基线分别为 1.36×、1.16×，SkyRL/FSDP 在 4×L40S 和 4×H100 的对应实验为 1.21×、1.17×。4×H100 的消融里，只有负载均衡为 1.11×，加入迁移后降至 1.09×，再加入 KV 否决升至 1.14×。证据来自单节点、35/20 步，投机基线的相对参照也与其他系统不同。复现时优先记录长尾 rollout 占比、恢复时的 KV 压力与重分片代价。 [论文 §4–6、图 7、表 2/3](https://arxiv.org/html/2611.17924v1#S6.SS2) · [版本记录](https://arxiv.org/abs/2611.17924v1)

### 主窗｜Self-Disagreement 单独奖励模型在反驳之后的判断

**Microsoft Research，arXiv 2611.17563 v1，11-12 22:37:32 UTC 投稿，11-16 公告批。** 同一个模型给出初答、生成反方论证、裁决后再作答。奖励由参考答案验证，监督初答、裁决和最终答案。Qwen3-8B 在九项任务的挑战后均分从 50.0±0.8 到 55.9±0.4，SCoRe 为 53.2±0.5。固定初答与 Gemini-3.8-Flash 反驳后，Base 为 52.3±2.0、SelfD-RL 为 66.4±0.3。表 3 的消融中，逐题统计的正确初答被改错比例从 10.3% 到 1.0%，仅要求输出裁决且不给该裁决奖励时为 2.9%。当前依赖可核对的参考答案，开放问题与部署中的多轮反驳待验证。适合把初答准确率、同一反驳下的改正率与改错率一起列入 RL 验收。 [论文 §3–5、表 1/3](https://arxiv.org/html/2611.17563v1#S4.T3) · [版本记录](https://arxiv.org/abs/2611.17563v1)

### 主窗修订｜Gnani 在 Indic 问答中测到英语切换抵消了 RL 收益

**Gnani.AI，arXiv 2611.16228 v2，11-13 04:30:59 UTC 修订投稿，11-16 替换公告。** 30B-A3B MoE 的同一 SFT 初始化经过 GRPO，34 项同配置评估中，MATH 增 36.07 点，但 XQuAD-Indic F1 降 19.32 点，回答使用英语的比例从 8.7% 到 91.7%，而 XQuAD-English F1 增 11.71 点。另一条从同一 SFT 出发的实验同时加 Indic 训练环境、语言与重复奖励，改善了四项先前下滑的 Indic 基准，部分英语推理/代码任务仍退化。v2 删除图 2 展示中的两行，表 1 和 §5 的主要对照保持一致。这是单 seed、多变量变化，修复用的 RL-Indic 未公开。建议用本地原创语言提示和翻译提示分别测试，并同时报告回答语言、文字系统和答案正确率。 [论文 §3–8、表 1、图 2](https://arxiv.org/html/2611.16228v2#S5) · [v1 原文](https://arxiv.org/html/2611.16228v1) · [版本记录](https://arxiv.org/abs/2611.16228v2) · [公开 RL-Base 工件](https://huggingface.co/gnani/gnani-evon-v3.3-30B-A3B)

### 主窗｜近相等分布下，KL 与 JSD 的值和梯度需要分别检验数值稳定性

**Reflection AI，arXiv 2611.18511 v1，11-13 18:26:07 UTC 投稿，11-16 公告批。** 从两个几乎相等的 softmax 概率相减会丢掉小梯度，求 KL 各项和时正负项也会相消。论文将精确前向 KL 改写成非负二阶余项求和，用小参数级数避免一阶抵消，并分开处理反向 KL、JSD 以及 Monte Carlo K1–K3。附录对照 CPU FP32 JAX 与 80 位十进制参考，在 2、3、4、8、16 类上做性质测试。输入量化已抹掉的差值无法恢复，某些近相等改写遇到较远分布需要回退，论文未实测 LLM 训练改善。适合先将零 KL、极近 logits、常数平移加入蒸馏或 RL 正则损失的单元测试，同时检验 loss 和梯度。 [论文 §3–5、附录 A](https://arxiv.org/html/2611.18511v1#S4.SS1) · [版本记录](https://arxiv.org/abs/2611.18511v1)

<a id="topic-4"></a>
## 训练系统与数值一致性

### 主窗｜Vitamin-E 保留 kernel 逻辑结构，按运行时需求分配 GPU SM

**中科院软件所、国科大、上海交大，RigoLLM ’26 工作坊论文 #77，11-15 03:00 UTC 议程。** 运行时冻结 kernel 二进制、grid 和依赖，把已满足前置条件的操作绑定至兼容、预创建的 CUDA Green Context。同一 GPU 上有重叠 SM 的上下文必须等在途工作完成才能交接租约。作者在 H200 上做了 1,000 次 reduction / Split-K GEMM 试验，以及 Llama 3.1 在 SGLang 的同一首 token 状态重放 100 次，输出均逐位一致。重排 kernel 的对照在 5 次重放中改选 token。另在 H20 的八组训练共卡实验中，相对五种基线的归一化总吞吐几何均值为 1.38–1.63 倍。逐位验证覆盖固定执行结构，CUDA Graph replay 被禁用，动态 MoE 路由仍待验证。**复现建议**在同一硬件和输入下先测逐位一致，再把多 token、CUDA Graph 和 CUDA runtime 开销作为单独测试。 [官方议程](https://rigollm.github.io/rigollm26/#program) · [论文 §3–5、图 1/5/6](https://rigollm.github.io/rigollm26/papers/rigollm26-77.pdf#page=3)

### 主窗修订｜HetCCL v2 加入 TP+ZeRO 与 MoE 专家并行的跨厂商训练对照

**Moreh、首尔大学、Samsung Research，arXiv 2601.22585 v2，11-13 04:28:22 UTC 修订投稿，11-16 替换公告。** HetCCL 用统一 runtime API 接上厂商各自的节点内路径和跨节点 RDMA。v2 附录 D.4 新增 Megatron-DeepSpeed 实验。8 张 NVIDIA GPU 加 8 张 AMD GPU 时，TP4+ZeRO-3 的 7B、13B 吞吐分别为 1,806、1,021 token/s，为两个各 8 卡同厂商基线吞吐之和的 91%、93%。8 专家、4.1B MoE 同口径为 91%。异构集群的每个节点仍是单一 GPU 厂商，H100 与 MI300X 的高速互连实验分别在同厂商单机完成。作者固定代码 `4dfddf5b` 中 `deps/tacc/src/tacc.cc::taccMemcpyAsync` 依 runtime 选择 HIP 或 CUDA，`src/plugin/plugin.cc::hetcclPluginInit` 加载对应设备插件。**复现建议**固定总 GPU 数，明确区分节点内、节点间路径，逐项测集体通信、微批负载均衡和训练吞吐。 [v2 附录 D.4、表 6/7](https://arxiv.org/html/2601.22585v2#A4.SS4) · [v1](https://arxiv.org/html/2601.22585v1) · [版本记录](https://arxiv.org/abs/2601.22585v2) · [固定代码](https://github.com/mcrl/hetccl/tree/4dfddf5b5a66fbdf5be009e58302391d9fd8c8a6)

<a id="topic-5"></a>
## 推理与部署

### Ripple-65B 的 Spark 包修正滑窗与树验证对齐

**Ricursive Intelligence · 11 月 16 日 23:57:42 UTC，NVFP4 `b4b4e36d`。** 同一 Apache-2.0 权重在 stock SGLang 0.5.19 上由本地包启用 DFlash2 树验证，16-token draft 最多 32 个验证节点。确定性 Triton 路径让 target extend/verify 的滑窗键按绝对 tile 边界起始，通过位置 mask 隐去额外位置，处理 prefill 与树验证的 BF16 softmax 不一致。文档默认总长 131,072、输入至 131,065、并发 4、前缀缓存开启。厂商以 128 输出 token 的提示报告单请求约 19.5 tok/s、四并发合计约 48 tok/s。树解码仍为实验模式，另一份同日比较中 NVFP4/BF16 的 Terminal-Bench 2.1 为 76.0/80.5，量化效果和部署速度来自不同设定。**建议考虑复验**，固定 cache 命中率，分别测试 chain 与 tree 在跨滑窗边缘的 logits 和吞吐。 [固定 Spark 差分](https://huggingface.co/ricursiveAI/Ripple-65B-Preview-NVFP4/commit/b4b4e36de43b5401f95c8e467cb3d1a355fd1e9f) · [固定评测表](https://huggingface.co/ricursiveAI/Ripple-65B-Preview-NVFP4/blob/b4b4e36de43b5401f95c8e467cb3d1a355fd1e9f/README.md)

### Haru-4B 接入 speech.cpp，以帧耗时衡量本地全双工

**higurashi-ai / speech.cpp · 11 月 15 日 06:15:31 UTC，`v0.11.0`。** 11 月 13 日的 Haru 研究基座将 Qwen3.5、MeanFlow 和 VibeVoice 编解码器移入 C++/ggml 的单文件部署。每帧 133 ms，同时消费和生成音频。移植者在锁屏 Apple M5 Metal 上做三次定序测量，F16 / Q8_0 的每步 p50 为 89 / 60 ms，峰值内存约 10.84 / 6.90 GB，K/V 预留 60 分钟。4.52 GB 的 Q4_K 出现连续标点及离散字符退化。已有逐阶段和两段会话的参考实现对照，长对话质量及带桌面负载的实时性仍需单测。**建议考虑复现**，先比较 F16 与 Q8_0 的端到端打断与音质，再测更低比特。 [固定基座](https://huggingface.co/higurashi-ai/Haru-4B/blob/8d6e83bcd3701a76fc27a764965665da1df7da78/README.md) · [正式运行时](https://github.com/nyosegawa/speech.cpp/releases/tag/v0.11.0) · [GGUF 固定评测](https://huggingface.co/sakasegawa/Haru-4B-GGUF/blob/fc026182f7bbfaa0400b2d44358a5e9e327b957f/README.md)

### Sophon-Realtime 4B Flash 用较小后端换全双工首音延迟

**Adventists AI · 11 月 16 日 02:13 UTC，`v1.4.0`。** 在 DuplexJev-4B-Para 上合并声音标记适配器，Flash 语音后端逐 80 ms 帧出音，作为单个 vLLM-Omni 模型提供 Realtime 接口。发布方在一张 RTX PRO 6000、按各变体内存限额、单用户下测首音中位数，Flash/Tag 为 0.21/0.40 秒，对应 Full-Duplex-Bench v1.0 四项均分 87.7/94.3。Flash 标称约 21 GB 显存，4B 在评测者以转录答案判分的 Big Bench Audio 上为 44.3%，32B 的历史结果为 81.8%。**建议考虑复现**，以并发、打断成功率和回答质量分别校验吞吐预算。 [固定 Flash 卡](https://huggingface.co/adventists-ai/Sophon-Realtime-4B-Flash/blob/ae4a8591d0ae4c3b3da0250a613828bc2435963b/README.md)

### Seedance 2.5 Flash 在火山方舟开放有版本的 API

**字节跳动 · 11 月 16 日模型发布公告，`doubao-seedance-2-5-flash-261115`。** Flash 沿用基础版的文生视频、首尾帧、全模态参考、编辑和延长接口。输出为 480p/720p、单次 4–30 秒。全模态参考上限为 10 张图、5 段视频和 5 段音频，分别低于标准版的 30/10/10。Flash 未提供标准版的样片模式，开通关系到账户余额、节省计划或资源包。公告与教程已核实，精确生效时刻、权重、训练细节与独立速度对照待查。**建议继续观察**，调用方可重点验证长输出、参考输入上限和现有编辑请求兼容性。 [发布公告](https://console.volcengine.com/ark/region:cn-beijing/docs/model-release-announcement#202611) · [API 指南](https://console.volcengine.com/ark/region:cn-beijing/docs/seedance-2-5)

### 厂商固定部署工件

- **NVIDIA Nemotron 3.5 Super VL NVFP4 · 11 月 16 日 18:34 UTC。** SC26 发布是现有 120B-A12B 模型的应用扩展，模型权重卡标记初版为 10 月 15 日。此次 DGX Spark 文档去掉 vLLM 0.32.0 备选并固定新 nightly，GB10 路径明确为单并发、LiLiCorr/DFlash 的 7 draft token，支持 1M 上下文的文本与图像。其他硬件仍列 MTP。**建议继续观察**，升级时固定镜像哈希，复核加载及首 token。 [SC26 官方发布](https://nvidianews.nvidia.com/news/nvidia-introduces-nemotron-for-science-to-accelerate-discoveries-with-agents) · [固定部署差分](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/commit/ae1368329b1fc76b71534c5d04b87b35d9cc6e15)
- **NVIDIA AVF-Instruct · 11 月 15 日 16:58 UTC 发布转换，16 日补清接口。** 发布方称仅重排同一检查点的权重名，并重新生成配置、processor 和聊天模板。后续卡明确音频按 30 秒片段跟随对应视频帧、默认抽 128 帧和截至 600 秒音频。上游集成尚需 AVF fork，权重受 NVIDIA OneWay Noncommercial License 约束。**建议考虑复验**同一视频的帧音对齐、处理器产物和输出一致性。 [转换差分](https://huggingface.co/nvidia/nemotron-labs-audio-visual-flamingo-hf/commit/d09c69554f258b7398bbcad9408b5361349468c8) · [固定调用契约](https://huggingface.co/nvidia/nemotron-labs-audio-visual-flamingo-hf/blob/9006f38cfd43ae350ac28a48f090ad06c2b1caab/README.md)
- **Zyphra ZONOS3-Mini 与 ZAC-1 · 11 月 16 日 02:53–02:54 UTC。** 新 Apache-2.0 布局将 `zonos3_mini` 模型与 `zac` codec 检查点分仓。新卡的 Transformers 原生路径以 `AutoModelForTextToWaveform` 和 `AutoProcessor` 加载，无需仓内 remote code，但须使用实现这些模型类型的 Transformers；解码器仍是 `keep2_plain`。本轮核到工件，支持该路径的稳定发行版待查。**建议考虑复验**同说话人、同 codec variant 下的波形和显存。 [固定原生模型卡](https://huggingface.co/Zyphra/ZONOS3-Mini-transformers/blob/3f582bf74827c543628ce893a8d62b09567cafc4/README.md) · [固定 codec 卡](https://huggingface.co/Zyphra/ZAC-1-hf/blob/a5506815a3156e02158b4ef2179494d4d95add37/README.md)
- **Liquid AI LFM2.5-24B-A2B · 11 月 16 日 18:05 UTC。** 新 9.5 GB QAD `Q2_K_L` GGUF 将大多数专家权重放在 Q2_K，部分层用 Q4_K，其他权重包括 embedding 仍为 Q8_0。发布方给出 llama.cpp 用法，缺同版本 BF16/QAD Q4_0 的质量和长上下文速度对照。**建议考虑复验**目标设备上的内存、外部工具 valid-call 率和长上下文退化。 [固定 GGUF 差分](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF/commit/4a79de634b3ad965a5f49c8ae1ce3368112e6ad2)
- **Edge0 120B/145B · 11 月 15 日 12:25 UTC。** 四个 Q3_K/NVFP4 包增加 `mmproj-BF16.gguf`，把图像编码为原语言主干接收的嵌入。需 Strata 构建或预编译包附带的 `strata-vision`。编码器可用 CPU，或占约 1.4 GB GPU 内存；此处的显存会与专家缓存共用。多图质量和语言回归对照待补，商业模型服务受配套许可约束。**建议考虑复验**两种编码路径的视觉 token、首 token 和专家缓存命中。 [145B 固定差分](https://huggingface.co/Edge0/Edge0-145B-q3k/commit/10fc212b0c2747706be26aa1e4c36aa84310e8a5) · [固定部署说明](https://huggingface.co/Edge0/Edge0-145B-q3k/blob/a19f88cc07d78047c273cdb88f3250f36c143999/README.md)

### 主窗｜Callo 在精确 KV 重算期间用失效 KV 提前起草

**杜克大学，arXiv 2611.18459 v1，11-13 17:28:37 UTC 投稿，11-16 公告批。** Agent 改写会话中间内容后，Callo 将保留 token 的旧 KV 对齐至新位置并重旋 RoPE 键，在空闲 decode 槽中提案。Prefill 并行重算并逐层推送精确 KV，decode 等全量确认后才按精确状态做 greedy 验证和提交。双 H200 NVL 上回放 SWE-agent，vLLM 0.29 + NIXL 的同请求平均时延比在 52% 容量时为 1.05–1.19×，涵盖 FP8 Llama-3.3-70B 与 BF16 Gemma-4-31B、Mistral-Small-3.2-24B。主配置相对仅打开流式传输的对照，在 16 槽轻载时再降 8%，2 槽满载时再降 3%。96 卡数据是仿真。值得用自己的会话编辑轨迹做 A/B，将逐层传输和空闲 decode 投机的收益分开度量。 [论文 §4–5、表 3、图 5/7](https://arxiv.org/html/2611.18459v1#S5.SS2) · [版本记录](https://arxiv.org/abs/2611.18459v1)

### 主窗｜VelServe 以迭代尾延时校准 prefill 预算和 decode 准入

**Sprinklr、Microsoft，arXiv 2611.18392 v1，11-13 16:34:37 UTC 投稿，11-16 公告批。** 聚合服务在线估计迭代预测误差的尾部，为 prefill 留出停顿余量，并优先处理短提示。解耦服务则将新请求在 decode 侧的加入间隔与每请求允许的慢步数关联。单 H100、同 vLLM 0.30 构建、混合 128–120K 输入、64–512 输出和每请求 P99 TBT ≤150 ms 下，达到 90% SLO 的请求速率相对最强基线 Medha 为 2.4×（Qwen3-8B）、2.9×（DeepSeek-V4-Flash 的 4 路专家并行单分片）和至少 4.4×（DeepSeek-V2-Lite）。8×RTX PRO 6000 的同节点 4P/4D Mooncake 回放，对完整 V4-Flash 的达标率为 35%，vLLM/Niyama 为 15–16%。长请求在持续负载下仍会超时，解耦只测了一个配比。适合按响应长度分层，检查 90% 达标交点和每请求 TBT 分位数随流量变化的情况。 [论文 §III–IV、图 7/11](https://arxiv.org/pdf/2611.18392v1#page=6) · [版本记录](https://arxiv.org/abs/2611.18392v1)

### 主窗｜MMFlash 把草稿训练集中在视觉决定且尚未学会的 token

**新加坡国立大学，arXiv 2611.18482 v1，11-13 17:54:28 UTC 投稿，11-16 公告批。** 目标模型分别带图和不带图前向，逐 token 的 KL 经带图预测熵折减后，与草稿失误概率、前缀通过权重相乘。草稿从目标已融合视觉的隐藏态起草，长视频可使用稀疏草稿上下文。论文用单 H200、SGLang、thinking 关闭、并发 1 的七项图像任务测 decode，Qwen3.5-4B/9B 的 greedy 平均为自回归的 3.00/3.05×，较同数据 DFlash 高约 10–11%。9B ChartQA 在并发 32 时为 1.89×，同模型 MMStar 为 1.14×。作者代码的 `visual_kl_per_token` 配对计算带图/无图 logits，`_base_weights` 对每个草稿位置做平滑置信度累乘。建议按真实图文输出长度、并发和 prefill 负载分别测 TTFT、显存和端到端吞吐，复用已释出的 drafter。 [论文 §4.2、§5、表 1–3](https://arxiv.org/html/2611.18482v1#S5.T3) · [版本记录](https://arxiv.org/abs/2611.18482v1) · [固定代码](https://github.com/fscdc/MMFlash/tree/9abe8fa031762677caf266e6e2051b47a6944bdf) · [4B drafter 工件](https://huggingface.co/FSCCS/Qwen3.5-4B-MMFlash/tree/2429fbdc15cd4bb091f933c968ad121e8b0e0014)

### 主窗｜CarryOn 在紧凑任务状态之外按需恢复原始观察片段

**新南威尔士大学、香港科技大学（广州），arXiv 2611.17808 v1，11-13 06:43:19 UTC 投稿，11-16 公告批。** Actor 写进度和下一步，小型排序器从源观察挑带引用的片段，用后续引用及动作构造训练标签，使先前淘汰的证据能再次出现。AppWorld、OfficeBench、Tau2 Telecom 与三个 Actor 的等权均值为 75.7% 成功率，较全历史高 13.3 点，较 SWE-Pruner 高 6.7 点。AppWorld 固定 Claude Sonnet 5 时，完整法为 79.8%，去紧凑状态为 66.1%，同预算按近因选择为 76.2%。排序器是按基准单独训练后跨 Actor 复用，验证局限于文本工具交互。对长会话值得同时保留全历史、定长截断与学习选择三条对照，单独统计旧证据重新进入上下文的任务收益。 [论文 §3–5、表 1/2、Limitations](https://arxiv.org/html/2611.17808v1#S5.T2) · [版本记录](https://arxiv.org/abs/2611.17808v1)

### 主窗｜SC26 把 Agent 四类状态对齐到同一 I/O 时间轴

**Argonne National Laboratory、Rochester Institute of Technology，SC26 AgenticAI4HPC，11-15 17:00 UTC 议程。** 作者对齐 TraceLab、Mooncake、记忆基准和 HPC I/O 轨迹，区分 KV、执行记录、认知记忆和大工件，统一为包含逻辑/物理字节的事件。按有写入的 1 秒窗口统计，单会话口径有 62% 同时写至少两类状态，叠加最重的 48 个 TraceLab 会话后为 87%，后者是配额压力场景。另一条 Mooncake 工具 Agent 轨迹的 512-token 块有约 55% 曾出现，160 GiB LRU 主要覆盖公共提示，约两分钟后只留下约 1% 的会话自身前缀。KV 大小按 Llama-3-70B 的 320 KiB/token 推算。固定代码 `analyze_c4_sensitivity.py::sweep_k` 改变并叠会话数，`analyze_c6_kvreuse.py::_stack_distances` 从块哈希估计 LRU 命中。作者公开归一化轨迹与重放脚本，统一存储层仍是设计提议。可先用实机日志回放校准跨状态预算，再判断是否需要联动 eviction。 [官方议程](https://sc26.conference-program.com/presentation/?id=ws_aaihpc130&sess=sess120) · [作者讲义第 9–10、15–16 页](https://docs.google.com/presentation/d/1ZejCTE-Kq0fUeVGAmCfB35M2DCotVbIiKYNz9WegluM/htmlpresent) · [固定工件与来源口径](https://github.com/mauryaavinash95/agent-artifact-ckpt-traces/tree/cf01797022c5e029b4b5d89e14fd2f64c3a9d74d)

### 主窗｜DuetScale 把解耦推理扩容分解为并行配置预测与 worker 再激活

**香港中文大学（深圳），RigoLLM ’26 工作坊论文 #85，11-15 07:00 UTC 议程。** 作者分别拟合 prefill 与 decode 的 TP、PP、DP 性能。论文报告 TTFT 和 TPOT 的 p50 模型中位相对误差分别为 9.45% 和 5.59%。WorkerPool 预建候选进程组，保留 rank 分配及 NCCL 上下文，激活时从锁页主机内存恢复 rank 本地模型状态。在 H20 的 SGLang TP4 试验中，普通启动 51.9 秒，池预热后的再激活 6.5 秒。首次建池更慢并占主机内存，模型与池还没有联合验证为端到端扩缩容策略。**复现建议**把首建池、再激活、闲置主机内存和扩容期间的 SLO 违约放在同一负载回放里测。 [官方议程](https://rigollm.github.io/rigollm26/#program) · [论文 §3–5、图 4/5](https://rigollm.github.io/rigollm26/papers/rigollm26-85.pdf#page=4)

### 主窗｜SemServe 用 Agent 的完成条件决定可以取消哪些推理请求

**香港中文大学（深圳），RigoLLM ’26 工作坊论文 #25，11-15 07:15 UTC 议程。** Agent DAG 为每次执行声明等全部分支结束还是接受首个经验证的成功。后者允许验证胜出分支后取消未胜出分支、回收请求和前缀缓存的逻辑容量。八组双方均答对的 HumanEval/MATH 对照中，输出 token 降 55.51%，KV token-seconds 降 62.21%，旧响应路径的均值耗时从 12.48 秒升至 15.19 秒。另有七次触发取消两分支的实验，清理移出响应关键路径后，均值从 15.544 秒降至 12.881 秒。两个前缀失效请求遇到生命周期竞态，实验未度量物理 HBM 归还。**复现建议**同时看任务准确率、失败分支仍需等待的情形、p95 端到端耗时和物理显存。 [官方议程](https://rigollm.github.io/rigollm26/#program) · [论文 §4.4–5、表 3](https://rigollm.github.io/rigollm26/papers/rigollm26-25.pdf#page=4)

### 主窗｜Qwen3-8B 的低比特量化提高了逐字复制标识符的错误率

**北京化工大学、中央司法警官学院、中科院软件所，RigoLLM ’26 工作坊论文 #26，11-15 06:45 UTC 议程。** 5,000 个测试案例包含 12,500 个标识符填槽，Qwen3-8B 通过 llama.cpp 在 RTX 4090 上运行。在可定位的槽中，局部高熵字符串错误率从 Q8_0 的 0.032% 升到 Q4_K_M 的 0.416% 和 Q3_K_M 的 2.444%，A100 重放也出现同一趋势。作者给标识符槽使用来源候选构建字节前缀树，在受约束槽的试验中，精确复制率由 97.3% 到 100%，语义正确率由 97.4% 到 97.9%。约束依赖上游定位，保证完成的字符串出自候选集，选错记录仍可能得到合法标识符。**复现建议**为编号等字段分别统计槽定位、来源集合成员、精确复制和语义绑定。 [官方议程](https://rigollm.github.io/rigollm26/#program) · [论文 §3–4、表 2–4](https://rigollm.github.io/rigollm26/papers/rigollm26-26.pdf#page=3)

### 补录｜MegaOps 以张量 tile 级同步让独立 kernel 在多 stream 重叠

**康奈尔大学，arXiv 2611.16248 v1，11-12 03:43:34 UTC 投稿，11-13 公告批。** 消费者等相应输入 tile 的信号量，生产者完成 tile 就通知，独立 kernel 可在各自的 CUDA stream 上重叠。作者在 0.6B–70B 稠密模型的单批离线实验中，单 H100 短 decode 吞吐平均为 vLLM 的 1.27 倍，关闭 overlap 时仍为 1.05 倍。8×H100 的 prefill 和长 decode 分别为 1.18 倍和 1.11 倍。基线使用不同 kernel，低并发和 kernel CTA 波数少时通信较难被掩盖，论文基准未覆盖在线多租户。**复现建议**先跑相同 kernel 的 overlap 消融，再扫 batch、输入长度和并发，分别看吞吐与时延。 [论文 §2–5、图 4–6](https://arxiv.org/html/2611.16248v1#S5.SS2) · [版本记录](https://arxiv.org/abs/2611.16248v1)

<a id="topic-6"></a>
## 开源工程

### Megatron-LM｜MTP 修正跨 CP 的权重，再支持主模型分布蒸馏

11 月 16 日合入 `main` 的两项变更直接影响多 token 预测（MTP）训练。**#7498**（`1a82f514b8`）在一个归约里汇总 CP 组的主任务与每个 MTP 深度的有效 token 数，传统归一化再乘 CP 组大小，抵消后续梯度平均。原先，同一目标随 MTP 位移落到另一 CP rank 时，局部掩码的分母会变。作者用 BF16 完整 GPT+MTP 在 H100、TP=PP=1、CP1 对 CP2 比较 56 组梯度，全部在既有 2.5% 容差内，最大相对差异 0.904%，空掩码梯度为零。传统模式的 CP>1 路径每个深度新增一次双标量归约。**#8639**（`f23e3c241f`）则允许混合标签交叉熵和主模型→MTP 的 KL，教师通过共享输出投影产生并停止梯度，支持温度与 KL 权重配置。词表分片内做归约，避免收齐词表 logits。默认仍只使用交叉熵，KD 路径需要保留教师 logits，源证据覆盖数值和梯度用例。复现蒸馏配方时应同时锁定目标掩码和 CP 归一化。 [CP 归一化 #7498](https://github.com/NVIDIA/Megatron-LM/pull/7498) · [MTP 蒸馏 #8639](https://github.com/NVIDIA/Megatron-LM/pull/8639)

同日 **#6293**（`45c2e71932`）将注意力、动态 CP、部分模型构造和推理模块中的进程组依赖显式传入 `ProcessGroupCollection` 或相应 group 参数。仍走全局 fallback 的外部调用会收到 `FutureWarning`，源码标记 0.21 起告警、0.23 计划移除。维护自定义 block、推理 wrapper 或多组并行拓扑时，可先定位 fallback 告警。 [Megatron-LM #6293](https://github.com/NVIDIA/Megatron-LM/pull/6293)

### verl｜主线 PPO 切换到 V1，rollout 调用方需处理分片整除

11 月 16 日，`main` 的 **#8625**（`124d22c401`）移除 V0 的 PPO 入口及训练器，`verl.trainer.main_ppo` 统一使用 V1。旧 fully-async 和 one-step-off recipe 迁至 verl-recipe 并固定在 verl v0.9.1。**#8709**（`909bb91387`）去掉 `DataProto` dispatch 的隐式 padding 与回收结果 trim，调用方必须提供能均分给 worker group 的 batch，仍可调用显式 pad/unpad 工具。V1 的 sync、colocate_async、separate_async 有作者在 8×80 GB Hopper、Qwen2.5-0.5B-Instruct 和小型 GSM8K 子集上的 3 步 smoke test，覆盖缓存 replay 和部分 checkpoint 恢复。升级跟踪 `main` 的训练脚本时，优先复查旧入口、配置和尾批处理。 [V0 迁移 #8625](https://github.com/verl-project/verl/pull/8625) · [显式 padding #8709](https://github.com/verl-project/verl/pull/8709)

同日 **#8681**（`cc1773b2f7`）解析 vLLM V1 的 `drafter.model` 和 MRv2 的 `speculator.model`，两种布局都能进入 MTP 权重、buffer 更新及后处理。**#8693**（`0c0a7a0f3b`）则在适用的未量化 bucketed 传输范围内跳过单桶完整性校验，避免只有底座权重的桶被误拒。旧 vLLM 没有相应上下文接口时沿用原行为。 [MRv2 #8681](https://github.com/verl-project/verl/pull/8681) · [分桶 #8693](https://github.com/verl-project/verl/pull/8693)

### TorchTitan、Miles 与 NeMo RL｜训练精度、长序列 log-prob 与快照停顿

- **TorchTitan** 的 Qwen3-30B-A3B DAPO 8K 两个 recipe 恢复 FP32 主参数、梯度和优化器状态，all-gather 仍用 BF16，单个 microbatch 从最多 4 个满长 rollout 降到 2 个。作者指出 BF16 在该模型常见权重尺度可能舍掉学习率量级的更新，且 4 张训练 B200 上 FP32 原微批大小首步 OOM。较早奖励对照每组仅一次运行且有本地变更，重验收敛仍需同 seed 对照。#6128 于 11 月 17 日 UTC 合入 `4a7c17c9ca`。另一项 #5511 于 11 月 15 日合入 `4122eafc0b`，将 Verifiers group 及其 GenerationServer 放到 rollout worker，controller 接收完整 group。作者在 16 个 CPU rollout 上逐字段比对训练输入，并做 10 步 GPU smoke test。此前 RDMA 权重同步的 packed staging 后续 #6097 仍为 OPEN。 [FP32 recipe #6128](https://github.com/pytorch/torchtitan/pull/6128) · [Verifiers #5511](https://github.com/pytorch/torchtitan/pull/5511) · [待合入 #6097](https://github.com/pytorch/torchtitan/pull/6097)
- **Miles #4645**（11 月 16 日，`cbea8cb01e`）给 Megatron 后端增加 `fused_lm_head`，只投影需评分的 response token，按默认 4,096 行分块计算 log-prob 和熵，反向逐块重算并保留输出投影的梯度。作者在 16×H100、TP4、Qwen3.6-27B 的录制轨迹回放中，用相同 49,152 token/GPU 预算测得峰值分配 53.0→49.5 GiB，actor train 67.8→67.6 s。旧路径到 114,688 token/GPU OOM，新路径运行并分配 64.8 GiB。支持的 policy/SFT loss 路径值得测显存与重算，MTP 头、sampling-support replay 和多 LoRA 有独立边界。 [Miles #4645](https://github.com/radixark/miles/pull/4645)
- **NeMo RL #5137**（11 月 16 日，`ae3a80e649`）为内存型 TransferQueue 0.1.11 SimpleStorage 增加两阶段快照。全局写屏障内固定 storage 索引和张量引用、将 controller 元数据直接序列化，屏障打开后才持久化，调用方仍负责发布。作者的 CPU Ray/ZMQ 回归在持久化挂起时完成 41 次 PUT/CLEAR，并由新 actor 恢复捕获时的值。约 82 MB controller 元数据三次回放的中位捕获耗时由 8.28 s 到 0.64 s，测试写临时文件且未 fsync。配置默认关闭、SSD offload 不支持，rollout/trainer 集成 #5139 仍为 OPEN。 [快照基础 #5137](https://github.com/NVIDIA-NeMo/RL/pull/5137) · [待合入集成 #5139](https://github.com/NVIDIA-NeMo/RL/pull/5139)
- **DeepSpeed #8739**（11 月 15 日，`5dc44e8453`）给 AutoEP 加入可选的专家梯度重叠，将 DeepEP 的 dX combine 与专家 dW grouped GEMM 并行，保留 ZeRO 处理。作者较早 H100 实验约缩短 2% 总耗时，最终修订提供 CPU 守卫测试。适合在已有 AutoEP、DeepEP 训练作业中核对通信是否真正藏到 dW 计算下。 [DeepSpeed #8739](https://github.com/deepspeedai/DeepSpeed/pull/8739)

### vLLM｜0.34.0 正式分发，主线另有 A100/H100、DSpine 与 Mamba 增量

**0.34.0** 首个 PyPI 文件于 11 月 16 日 23:50:58 UTC 上传，GitHub 稳定 tag 和上一轮已报的 RC4 均指向 `62d4bc3c7e33b1b6c906e233e6af2d3506c482ca`。直接比对显示 0 个源码提交增量。下列能力按对应源码合入记录登记。 [稳定 tag](https://github.com/vllm-project/vllm/releases/tag/v0.34.0) · [RC4 与正式版比较](https://github.com/vllm-project/vllm/compare/v0.34.0rc4...v0.34.0) · [PyPI 分发](https://pypi.org/project/vllm/0.34.0/)

- **DeepSeek-V4.1-Flash 上 A100/H100。** 11 月 16 日的 #66497（`352dbd1db9`）按 SM80/SM90 选择 CuTe DSL 路径。SM80 的 sparse MLA 在 shared memory 解码压缩 KV 为 BF16，再用 tensor core 计算，SM90 另有针对 Hopper 的 decode 路径。作者用 8×A100 或 8×H100、TP8、DSpark 验证，在指定修订和负载上报告 GSM8K、GPQA-Diamond、AIME25 及最高 256K 的 RULER NIAH，对照 GB300 TP4。A100 是新增可运行平台，H100 证据含与 InferenceX 的外部基线对照，复验需固定专家后端、采样和 PR 修订。 [vLLM #66497](https://github.com/vllm-project/vllm/pull/66497)
- **DSpine 接入现有投机验证。** 11 月 16 日的 #67094（`8070069e84`）适配腾讯与 Speculators 权重布局，DSpine 提供候选转移分数，并复用 DFlash/DFlash2 采样验证。作者在 1×H100 80 GB、BF16、TP1、FA3、并发 8、15 个草稿 token、无 prefix cache 上用 Qwen3.8-27B + DSpine-b16 做五项测试。贪心草稿的 per-response 平均接受长度在 GSM8K 为 6.93、MT-Bench 为 4.69，两者的 pooled 指标为 5.90、4.13。接受长度适合用于挑选复验负载，服务时延仍要在同样硬件与调度条件下测量。 [vLLM #67094](https://github.com/vllm-project/vllm/pull/67094)
- **Mamba 前缀命中先纠错，再从三次 prefill 降到一次。** 11 月 15 日的 #67717（`75f3967ac1`）只在 Mamba state 对应块尾时发布整块哈希，修复缓存启用时的静默错算。11 月 16 日的 #66938（`5b028fb836`）在遵守该不变量的前提下，用请求私有的恢复块存中间 checkpoint，让跨块的续轮 prefill 可合并。作者在 8×GB300、Kimi K3.1、TP8/DCP8、128-token 命中粒度、Mooncake、OSL=1 的轨迹回放中，c16/c48 平均 TTFT 分别下降约 13%/11%，吞吐基本持平。适合将命中率近似固定后对比调度步数与 TTFT。 [块尾正确性 #67717](https://github.com/vllm-project/vllm/pull/67717) · [合并 prefill #66938](https://github.com/vllm-project/vllm/pull/66938)
- **在线量化模型支持直接刷新权重。** 11 月 16 日的 #65802（`a4910b4128`）让训练侧用 checkpoint 名称发送 BF16，服务侧直接量化写回原有存储，再做相应 post-load 处理，地址稳定以保留 CUDA graph。须选择 `VLLM_RELOAD_MODE=direct`，覆盖线性和 MoE 的若干 FP8/MXFP8 配置及部分 MoE FP4 配置。上线前可按支持矩阵检查本地 scheme 及权重交付路径。 [vLLM #65802](https://github.com/vllm-project/vllm/pull/65802)

### SGLang｜有界前缀调度、拼批正确性及一次合入后撤回

- **长队列只在 KV 竞争有实质影响时启用有界 LPM。** 11 月 16 日合入的 #48409（`f76ac26543`）为 `lpm` 增加估计受威胁复用量与 KV 容量之比的门槛。队列超过 128、比值至少 1.5 时最多选入 64 个固定成员，每轮只给这些请求重新匹配并排序，减少大队列全量扫描，也限制新请求持续插队。作者用 Qwen3-4B、RTX 4090、并发上限 8 回放相同请求。在 KV 容量 32,768 的早期 gated 修订中，未缓存输入 token −11.63%、吞吐 +2.13%、平均 TTFT −6.18%。另一个容量点平均 TTFT +1.40%，发表的性能表仍用旧 cooldown，需按最终提交复测。 [SGLang #48409](https://github.com/sgl-project/sglang/pull/48409)
- **GQA 和 DP 拼批修正。** 11 月 15 日的 #47802（`40723a52de`）让 Triton decode 按 KV head 划分完整 query-head tile，末尾部分 tile 用组内掩码。作者在 H100 上用 GLM‑4‑32B‑0414、48 query heads / 2 KV heads 的既定 GSM8K greedy 提示，报告从 50.87% 恢复到 93.48%，FP8 E5M2 KV 路径由 49.51% 恢复到 92.87%。#49170（`80a461512e`）让 DeepSeek‑V4.1 DP attention 的混合 prefill/decode 在各 rank 均可 replay 时才走尾部 CUDA graph。11 月 16 日的 #50146（`552e16968b`）则让推迟 prefill 的 DP rank 也参加 delayer 的 all-gather，CPU 多 rank 回归覆盖配对。 [GQA #47802](https://github.com/sgl-project/sglang/pull/47802) · [混合批 #49170](https://github.com/sgl-project/sglang/pull/49170) · [delayer #50146](https://github.com/sgl-project/sglang/pull/50146)
- **HiSparse 增加 HiCache 数据面。** 11 月 16 日的 #49473（`d9bf131d8b`）让 DSA indexer 独立于 attention KV 的页表寻址，新增设备/主机访问与每请求 LRU 的 buffer，目标是在共享前缀 KV 移出 GPU 后仍保留索引页读路径。GPU 单测面向 BF16/FP8 字节保持、共享页、请求复用和 graph replay。该提交提供索引和 buffer 机制，PR 未给端到端服务性能。 [SGLang #49473](https://github.com/sgl-project/sglang/pull/49473)
- **Blackwell/Rubin 混合模型插件接口合入，另一组优化已撤回。** #50157 于 11 月 16 日 UTC 合入 `0b28a39a2b`，向 GDN+QSA+MoE megakernel 暴露逐层 KV/状态池、step metadata、prefill graph 与草稿回调，仅由插件接管其声明支持的 step。独立的 #50156（`775db052b6`）随后被 #50703 于 04:17 UTC 以 `3981de0825` 反向提交，包含 verify 后处理、状态回滚和一组缓存等优化。工程评估可跟踪接口，性能需按撤回后的基线重测。 [插件 #50157](https://github.com/sgl-project/sglang/pull/50157) · [原合入 #50156](https://github.com/sgl-project/sglang/pull/50156) · [撤回 #50703](https://github.com/sgl-project/sglang/pull/50703)

### vLLM-Omni、FlashInfer 与 FlashAttention｜双工缓存和算子边界

- **vLLM-Omni 的 Mooncake 缓存扩至 Qwen3-Omni 双工会话。** 11 月 15 日 #10107（`0f55c39eb0`）给普通 turn 按一致边界恢复 KV 与 Thinker embedding、hidden state，失败时转重算。11 月 16 日 #10321（`4e7c184bd8`）让双工续轮从选中的会话历史重建输入，以内容关联的多模态哈希复用早先前缀，并补齐取消、不同音频和 hidden-state 读取失败的验收脚本。范围是 CUDA、MRv2、未量化 Thinker TP=1、单 full-attention KV group。接入时尤其需要验证音频身份变化和取消后的缓存行为。 [普通轮 #10107](https://github.com/vllm-project/vllm-omni/pull/10107) · [双工 #10321](https://github.com/vllm-project/vllm-omni/pull/10321)
- **FlashInfer** 的 #9406（11 月 16 日，`f25a0c25a0`）在 SM100/103/107 的统一 `MoELayer` API 中加入 per-tensor FP8 W8A8 batch-invariant backend。量化和两个专家 GEMM 声明确定的舍入，按 token 的专家槽顺序用 FP32 combine。支持完整专家集合和已 finalize 路径，拒绝 EP 分片。另有 11 月 15 日 #9438、#9500（`0132cbc015`、`67d996c0a4`）为 Blackwell Cake block-sparse attention 接入融合量化及小形状 V quantizer 优化。项目在 B200、CUDA graph、B1/H2/S4096、每 query block 选 6 个 KV blocks 下报告 15.0→12.5 μs，长序列大多持平。 [FP8 MoE #9406](https://github.com/flashinfer-ai/flashinfer/pull/9406) · [Cake #9438](https://github.com/flashinfer-ai/flashinfer/pull/9438) · [量化 #9500](https://github.com/flashinfer-ai/flashinfer/pull/9500)
- **FlashAttention** 11 月 15 日合入四个独立边界修复。FA2 将 LSE、反向 softmax 中间值和返回 attention probabilities 的乘法索引扩为 64 位（`ff622c0ff6`），并按 padded 布局写空 query block 的 LSE（`2dd89523b5`）。作者的 B200 双序列样例里后者将 320/640 个错误 LSE 位置清零，attention 输出此前已一致。CuTe 路径令零列 page table 走空 KV 分支（`8fa1e462fc9`），并拦截达到 2³¹ 的 page-table key capacity（`bd26bc8e10`）。适合用空分段、padding 和长页表检查下游调用。 [FA2 #3169](https://github.com/Dao-AILab/flash-attention/pull/3169) · [FA2 #3170](https://github.com/Dao-AILab/flash-attention/pull/3170) · [CuTe #3167](https://github.com/Dao-AILab/flash-attention/pull/3167) · [CuTe #3168](https://github.com/Dao-AILab/flash-attention/pull/3168)

### Accelerate、Helion 与 Transformers｜版本升级和训练正确性

- **Accelerate v1.16.0**（11 月 16 日，`f938aab2a8`）修复 FSDP1 在 meta 参数实例化后恢复优化器引用，修复 FSDP2 RAM-efficient 加载和 activation checkpointing 的分片，并支持未传 optimizer 时先 prepare 模型。最低要求升到 Python 3.11/PyTorch 2.6，并移除 PiPPy、MS-AMP 和旧 FP8 kwargs。使用旧环境或先评估再训练的工作流可优先检查迁移。 [Accelerate 发布](https://github.com/huggingface/accelerate/releases/tag/v1.16.0) · [PyPI 分发](https://pypi.org/project/accelerate/1.16.0/)
- **Helion v1.6.0**（11 月 15 日，`aaf24f4d9d`）增加 CuTeDSL 的寄存器与 MMA 路径、合并多维规约，并修复 partial-tile mask、原地更新、scan 等生成代码的正确性（#4456，`c4b4c7039d`）。可用项目回归对照现有 kernel 迁移，官方发布没有一套统一的端到端收益。 [Helion 发布](https://github.com/pytorch/helion/releases/tag/v1.6.0) · [CuTeDSL 正确性 #4456](https://github.com/pytorch/helion/pull/4456)
- **Transformers** 11 月 16 日的 #50727（`311accabde`）把 Mistral Large 4 的 LeRoPE `alpha` 从 buffer 改为参数，并恢复前向梯度。回归比较普通和非 reentrant checkpointing 下的有限非零梯度与参数更新，reentrant 用例明确跳过。同日 #50658（`8108baccfd`）移除 JetMoE 的重复 attention router logits 采集，回归检查每层仅记录两个路由器的输出。微调相关模型时应从这些提交之后重新核对 loss 与可训练参数。 [LeRoPE #50727](https://github.com/huggingface/transformers/pull/50727) · [JetMoE #50658](https://github.com/huggingface/transformers/pull/50658)

### Triton 与 PyTorch｜kernel 启动依赖、常量舍入和小 k 路由

- **Triton** 11 月 16 日在 Gluon 暴露 SM90+ 的 `gdc.wait()` 与 `gdc.launch_dependents()`（#12750，`ea5722944b`）。前序 grid 可发出后续 kernel 的启动提示，后续需启用 `launch_pdl=True` 并等待前序结果可见后才能读。PR 的检查覆盖 Hopper、Blackwell 和 Rubin 的 IR 生成。另有 #12778（`735d1b531e`）将 Python 浮点常量直接舍入至 FP16/BF16，避免先过 FP32 在临界点发生二次舍入。编译器与解释器均有临界值回归，依赖逐位对照的 kernel 测试需更新其预期。 [Gluon #12750](https://github.com/triton-lang/triton/pull/12750) · [舍入 #12778](https://github.com/triton-lang/triton/pull/12778)
- **PyTorch** 11 月 15 日在 Inductor 为常见 MoE 路由形状增加 CUDA small-k top-k 分解（`48f1cea6cf`）。末维至多 1,024、最大值方向、k=1–16 时，按有序整型键执行多轮 argmax 和删除，融合为 persistent reduction。删除用独立 sentinel，允许输入包含真实的负无穷，测试覆盖 NaN、正负零和并列值。另一个 DTensor 提交（`11b971398a`）允许按 shard 局部计算 `any/all`，再以 max/min 合并布尔 partial，已有数值 partial 则先解析。 [Inductor top-k](https://github.com/pytorch/pytorch/commit/48f1cea6cf0a29cf83612fb05c11f183e06a3554) · [DTensor any/all](https://github.com/pytorch/pytorch/commit/11b971398ac96eefe8b5d8296dabcc43092d8b6e)

### 补录｜两项 checkpoint 兼容与异步评分隔离

- 11 月 13 日的 **Megatron-LM #9334**（`8bb72c784b`）从 metadata 识别旧 wide-residual 键，恢复 torch_dist checkpoint 推理加载，并防止 global-batch quantile router 重新带入不兼容的 expert-bias 配置。**VeOmni #1394**（`150f6690ee`）将 Qwen4-Exp PLE checkpoint 按表的逻辑行保存，加载时补回分片零尾，兼容旧 padded checkpoint。四进程 CPU/Gloo 用例检查了恢复后的优化步。 [Megatron-LM #9334](https://github.com/NVIDIA/Megatron-LM/pull/9334) · [VeOmni #1394](https://github.com/ByteDance-Seed/VeOmni/pull/1394)
- 11 月 13 日的 **TRL #8330**（`678074b62a`）把数学 accuracy reward 放到可重启子进程，超时终止并替换 grader，避免单条答案迟滞卡住异步 rollout。锁串行化调用，补丁覆盖超时及恢复路径。这是 1.18.0 发布后的独立源码事件。 [TRL #8330](https://github.com/huggingface/trl/pull/8330)

<a id="topic-7"></a>
## 硬件与软件协同

### 主窗｜LWS 为推理时权重学习加入按层版本和可恢复矩阵工作

**弗吉尼亚大学，arXiv 2611.18523 v1，11-13 18:37:52 UTC 投稿，11-16 公告批。** Learning While Serving 在注意力等操作占用非矩阵资源时运行反向计算。它给每层权重独立的可见性截止时间，允许在途预测保留旧版本，并将长矩阵归约切成可挂起、恢复的块。九个模型的统一操作级 NPU 仿真中，数据中心侧预测吞吐为同硬件纯预测的 1.00，串行方案为 0.64，Equinox 为 0.75。面积与开销来自模拟、RTL 和 DRAM 模型，NLL 用 16 条 Books3 验证样本测量。适合先读 TTT-E2E 3B/128K 的层版本滞后消融，复查预测任务的质量是否也能承受早层短暂滞后。 [论文 §3–5、表 1/2、图 9–11](https://arxiv.org/html/2611.18523v1#S5) · [版本记录](https://arxiv.org/abs/2611.18523v1)

### MLPerf Training v6.1 首次给出大型 Agentic RL 的端到端达标时间

**MLCommons · 11 月 16 日，Training v6.1。** 新增 Qwen3.5-397B GRPO 的首批正式结果。任务是多轮软件工程，计时到验证集达到 `pass@4 ≥ 0.69`；每轮完整验证有 1,004 条轨迹，评估采样参数为 temperature 0.1、top-p 0.95。参考实现将 Megatron 训练与 vLLM rollout 解耦，用 NeMo Gym 和 Apptainer 隔离任务环境。训练和采样异步运行，参考实现仅用最多落后一版策略权重的轨迹训练，提交方可自选此值。结果同时受到 rollout、权重同步、训练和评估开销影响。 [MLCommons 发布](https://mlcommons.org/2026/11/mlperf-training-v6-1-results/) · [基准设计与规则边界](https://mlcommons.org/2026/09/mlperf-training-llm-post-training/)

以 MLCommons Explorer 的 Closed 条目为准。64 卡 FP8 linear 栏，Rubin（6.1-0121）用时 25.93 分钟，Google 的 GB300（6.1-0020）为 34.17 分钟，NVIDIA 的 GB300（6.1-0034）为 36.99 分钟。不同提交的软件配置和部署形态有差别。640 芯片 Ironwood（6.1-0023）为 18.02 分钟，Google 报告最终分数 0.765、训练芯片等待低于 3%；1,024 卡 Rubin（6.1-0117）为 10.44 分钟。后两项的设备和规模均不同。预训练的同卡数对照更直接：DeepSeek-V3 671B 在 1,024 卡 FP8 linear 下，Rubin 为 4.14 分钟、GB300 为 8.03 分钟（6.1-0116 / 6.1-0030），计时从基准检查点起。两家厂商博客对部分提交 ID 的映射与 Explorer 有出入，上述数值按任务、系统、卡数和用时交叉核对。**建议深入阅读**，用达标墙钟时间、验证成本和策略滞后评估整套 RL 系统。 [MLCommons Explorer](https://results.mlcommons.org/training) · [Google 提交说明](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/training-performance-benchmarks-tpus/) · [NVIDIA 提交说明](https://developer.nvidia.com/blog/nvidia-vera-rubin-nvl72-achieves-up-to-2x-blackwell-ultra-performance-in-mlperf-training-v6-1/)

**AMD、Crusoe · 同轮公布 GPT-OSS-20B MXFP4 配方。** Primus/Turbo 在 MI355X 上用 MXFP4 矩阵运算和 BF16 attention，权重采用 32×32 块供前反向复用，wgrad 两个操作数应用配对 Hadamard，输出梯度用随机舍入，并抑制小步更新引起的权重反复跨量化阈值。Crusoe 的 8/16/32 卡（6.1-0004/0005/0006）为 58.04/39.74/27.16 分钟；AMD 按规则允许的不同全局批量及达标样本数归一后，报告 16 卡和 32 卡扩展效率 94%、69%。这是算法、算子、RoCE 和配方的整体结果。**考虑复现**，AMD 已给训练镜像与配置路线，可先在同硬件固定达标条件，分开比精度配方与多节点通信的增量。 [技术拆解](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-training-v6.1/README.html) · [复现指南](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-training6.1-repro/README.html)

### Lambda 披露万卡 GB300 集群使用共封装光学交换机

**Lambda、NVIDIA · 11 月 16 日，SC26 部署披露。** Lambda 的魁北克集群用 144×800 Gb/s、液冷的 CPO 交换机连接逾 10,000 张 GB300，光引擎贴近 ASIC，每台保留 18 个可更换外置光源。相对可插拔光模块，Lambda 报告机柜至 pod 的网络延时降幅为 3.8% 至 10.1%，测试吞吐相当，光学器件温度中位数从 51°C 降到 36°C。公告未披露流量模型、尾延时、故障观测期或训练中断率原始记录。**建议继续观察**，用集群通信尾延时和实际作业中断率验证扩容收益。 [Lambda 部署数据](https://lambda.com/blog/worlds-first-large-scale-deployment-of-nvidia-photonics-switches-with-cpo) · [NVIDIA 交换机细节](https://blogs.nvidia.com/blog/lambda-photonics/)

### 补录｜MUSA SDK 4.3.10 修正变长注意力反向和 DeepEP 低延迟超时

**摩尔线程 · 11 月 13 日，SDK 4.3.10。** 发行说明记录 FlashAttention Varlen backward 越界及跨段数据污染、GQA head dimension 256 反向精度、大 batch SDPA 的 NaN，以及 DeepEP low-latency 多卡超时修复。MCCL 加入 all-reduce tree，Toolkit 新增异步批量内存传输与原子操作 API。配套版本包括 Driver 3.3.10-server、Toolkit rc4.3.10、muDNN rc3.1.10.PH1 和 MCCL rc2.1.10.PH1。厂商提及相对 4.3.8 的 DeepSeek 和 MCCL all-to-all 回归修复，未给模型、卡数、精度或长度。**建议深入阅读**，把注意力反向跨段隔离、低延迟通信和 GID 选择纳入版本回归。 [发布及组件版本](https://developer.mthreads.com/product-updates/92) · [发行说明](https://docs.mthreads.com/musa-sdk/version-4.3.x/releasenote/MUSA%20SDK%204.3.10%20releasenote)

### HCCL 拓展非 2 幂归约分片和 HostDPU 变长通信

**CANN/HCCL !3901、!3833 · 11 月 16 日，master。** ReduceScatter 新增 RHD。在非 2 幂 rank 数上用配对与前后处理围绕递归主体，覆盖 AICPU 和 HostDPU 的二级、三级执行器。只有显式 `HCCL_ALGO` 链式配置才参与选路，一级域和三级 HostDPU 分支有拓扑约束，ReduceScatter RHD 排除 64 位类型与 PROD。AllGatherV、ReduceScatterV 的 HostDPU Mesh 路径则新增长数据多轮传输、rank 本地拷贝和 DPU 收发，按 HostDPU 拓扑与 Mesh/CLOS 条件选路，ReduceScatterV 排除 FP64 和 UINT64。合入记录提供实现及测试文件，未见非 2 幂集群的时延曲线。**考虑复现**，比较 3、5 及跨域 rank 数下与现有执行器的结果一致性、分片不均衡和通信尾延时。 [RHD 固定提交](https://gitcode.com/cann/hccl/commits/detail/d1c67b9aa1424fcc502d5ffece0db8e46733d054) · [变长 Mesh 固定提交](https://gitcode.com/cann/hccl/commits/detail/0ac8226b578d0e6bb21c0c59cc0a6c299dce6d7d)

**CANN/HCCL !3875 · 11 月 16 日，master `8ec30f62`。** AIV、CCU、AICPU 通信算子新增起止计数器，覆盖普通启动、图模式和部分专用入口，取址失败可降级继续执行。对卡住位置的判定依赖相应路径启用且计数器能采集。**建议深入阅读**，在固定拓扑验证通信超时和图回放的可观测性，主源缺少观测开销和线上平均修复时间。 [合入记录](https://gitcode.com/cann/hccl/merge_requests/3875) · [固定提交](https://gitcode.com/cann/hccl/commits/detail/8ec30f62148991bf6282d95fcfc0cab1122f3f36)

### CANN 修正投机解码状态队列和低精度算子契约

**CANN ops-nn !14501 · 11 月 16 日，master `4e80e024`。** `AdvanceStep` V2 将 sampled token、spec token 暂存拆到独立 UB 队列，并同步缩紧 tiling 计算的每循环序列上限。作者在 Ascend 950PR 报告 racecheck ERROR 由 156 降至 18，53 个 golden 用例四轮通过。剩余 18 个报告集中于大形状跨循环，作者认为是队列建模限制，仍有复核空间。**考虑复现**，针对大批量、跨循环和混合接受长度核对 decode 状态推进并检查实际吞吐。 [合入及验证](https://gitcode.com/cann/ops-nn/merge_requests/14501) · [固定提交](https://gitcode.com/cann/ops-nn/commits/detail/4e80e02440db386fd9c13442f40073cf61f68f5f)

**CANN ops-nn !14554、!13580、!13930 · 11 月 16–17 日，master。** SwiGluQuant V1/V2 的 tiling 白名单收紧到 static/dynamic，尚无对应 kernel 的 dynamic_msd 在 tiling 拒绝，部分错误按 tiling 错误码上报；作者给出 8 个板上用例。Ascend 960 QBMM v4 的 MX A8/W4 路径复用 Blaze 滑窗与基本块计算，保持 v4 tilingKey 布局；随后一项 matmul 域改为数值 NpuArch 门控，覆盖 960DT/PR 的 DAV_9201/9202 并固定新的 opbase 依赖，主源称 x86 ophost 10,294 个用例通过。**建议深入阅读**，检查 wrapper 的错误码契约、旧 opbase 与新注册表搭配，以及目标固件上的性能和支持边界。发行包映射仍待确认。 [SwiGluQuant 固定提交](https://gitcode.com/cann/ops-nn/commits/detail/0faf65e1b95beac69fc4c6598e6b9535330fd18a) · [QBMM 固定提交](https://gitcode.com/cann/ops-nn/commits/detail/f31c78e32a66179e15cd08af2bfc28c59f18c5cc) · [架构注册固定提交](https://gitcode.com/cann/ops-nn/commits/detail/955fa5c92246c4851530034c798e7694412bca8f)

### CANN Runtime 暴露设备邻近 NUMA 和 L2 持久缓存上限

**CANN Runtime !5876、!6006 · 11 月 16 日，master。** `aclrtGetDeviceInfo` 可查询最近的 host NUMA 节点，覆盖 950PR/DT、Atlas A3/A2，系统无 NUMA 返回 −1。950 的 device limit 增加 persisting L2 的容量配置。前者便于在 KV host staging 和数据管道中选择 NUMA，后者可用于评估常驻工作集与其他 cache 流量的取舍。**考虑复现**，按板卡、驱动、CPU 亲和性与实际应用测主存搬运和整体服务延时。上游差分没有公开端到端结果。 [NUMA 固定提交](https://gitcode.com/cann/runtime/commits/detail/8fe84e036680d3bdef0e64c0d58e9babc17e8fb1) · [L2 固定提交](https://gitcode.com/cann/runtime/commits/detail/d2ba42a24565f6ae68013e0f35cbc0f72420cee0)

### ATOM 把共享 Mooncake Store 的路由、预取和状态回放接在一起

**ROCm/ATOM #3097 · 11 月 16 日，默认分支 `7e2f723a`。** `share_cache` 用每 rank 的复用与待 prefill 工作量选路，允许 Mooncake Store 提前开始读。当前缀命中落在 Store 的 256-token chunk 中间，启用该能力的 connector 可把 HBM 命中回退到下一个较低边界、重算差额，提前发起远端读；混合状态、滑窗和连接器能力仍约束这个路径。多 DP rank 要能访问同一个共享 Store；作者的 DP2 分池对照中 263 次读取均出现对象未找到并回退。提交正文含多个中间试验及回退，其中 4-tick prefill 等待上限最终撤销。**建议深入阅读**，优先验证 P/D 下的共享池拓扑、会话倾斜、早读和取消时的状态，而后重测尾延时。 [合入提交与差分](https://github.com/ROCm/ATOM/commit/7e2f723aeca8246534c6e316d9fb5ad5cb72f622)

### MLXcel 增加 CUDA 端原生 4-bit 解码和混合精度扫描

**lablup/mlxcel #2984、#3026 · 11 月 15 日，默认分支。** #2984 给 IREE CUDA 单序列 decode 补齐原生 4-bit affine GEMV，保留未编译 PTX 时的 FP16 路径，批量 decode 仍走原路径。作者在 GB10、CUDA 13.0、128 个生成 token、容量 256 下，Llama 3.2 1B 从约 20–21 到 58–69 tokens/s，Qwen2.5-7B 从约 5.8 到 19 tokens/s。Qwen 的原始提示轨迹 5/6 逐 token 相同，第 6 组在第 67 token 分歧。#3026 则修复部分 ModelOpt W4A16_NVFP4 加载路径的尺度组推断与重打包，增加错误布局的负对照，Metal、ROCm 未测。**考虑复现**，把权重布局校验、单序列、批量及逐 token 对照分开测。 [CUDA IREE 固定提交](https://github.com/lablup/mlxcel/commit/0afe4037ea5711406c00b90879ddf4f5a6b26809) · [ModelOpt 固定提交](https://github.com/lablup/mlxcel/commit/6f20b796c97e551592ec30594da993e9bea68416)

**lablup/mlxcel #3128 · 11 月 17 日（上海时间），默认分支 `ce8de83d`。** CUDA 的 Mamba1 graph-exact scan 接受 FP16 激活配 FP32 的衰减参数 A，在计算后按图路径舍入回 FP16，使 Jamba 的对应 CUDA 路径可用融合扫描。作者用 GB10、Jamba reasoning 3B 4bit、2,151-token 提示，对比分步图扫描的三轮 prefill 约 1,003–1,058 对 2,072–2,116 tokens/s，32-token greedy 一致。普通 Mamba / Falcon-Mamba 的默认前向仍走既有路径，8K 未测。**考虑复现**，分开固定扫描选择、上下文长度与精度，检查溢出及输出等价。 [固定提交和测试](https://github.com/lablup/mlxcel/commit/ce8de83d325ae7b6c25d22774632dc6e78690d2e)

### Primus 把三款 MoE 的 MaxText 默认配方切到无 token 丢弃

**AMD-AGI/Primus #1450 · 11 月 15 日，主分支 `a0ab9227`。** Qwen3-30B-A3B、DeepSeek-V2-16B 和 Gemma4-26B 在 MI300X、MI325X、MI355X 的默认配方改用 Primus-Turbo grouped GEMM、FSDP 8 及不限制专家容量的路由。MI300X/MI325X 用 hipBLASLt，MI355X 用 FlyDSL。MI355X 的 FP8 配方对前向及两个梯度使用 per-tensor E4M3。旧配方可用 `token_dropping` 名称调用。依赖 MaxText `release/v26.10`、JAX 版 Primus-Turbo 和相应 GEMM 后端。

作者在 8 卡、合成数据、20 步中统计第 5–19 步中位数。Qwen3-30B-A3B 在 MI355X BF16 从 10,038 到 20,232 tokens/s/卡，Gemma4-26B 在 MI355X FP8 从 13,750 到 17,505。批量、重计算、并行及 collective 合并参数均有变化，结果适用于整套配方。反例是 MI355X 上 Mixtral 8×7B FP8，dropless 为 15,451，对照为 15,864 tokens/s/卡。**考虑复现**，先固定训练 token 数和并行，跟踪路由分布、首次 collective 等待和长程 loss。 [合入与测试条件](https://github.com/AMD-AGI/Primus/pull/1450) · [固定差分](https://github.com/AMD-AGI/Primus/commit/a0ab9227a56b8b533a52a8118e88d64c0de23e26)

### 昇腾的两条解码改动有不同的验证边界

**vLLM Ascend #21549 · 11 月 15 日，主分支 `6abfc9de`。** Kimi K3/K3.1 的实验性 RecoverSSM 可与 DSpark 和 Model Runner V2 使用。验证保持持久循环检查点不动，临时记录按请求索引，采样后提交被接受的状态及卷积历史。K3.1 原生 MTP 草稿头未包含在该合入。作者在较旧镜像上以四节点 A5、K3.1 Pro W4A8、DP4/TP8/EP32、3 个候选 token、eager、关闭前缀缓存测试。65,536 输入、2,040 输出时总占用缓存块 100→91，但额外记录区约 0.30 GiB/NPU，不能由块数折算 HBM。单批 4 个 65,536/2,048 请求的中位流式 TPOT 为 154.53→149.00 ms/token，跨配置有输出差异，P/D 对齐精度及最终提交的分布式结果仍待验证。**考虑复现**，先用连续请求对比接受序列、状态提交、输出与 HBM，再测试 P/D。 [合入与实验记录](https://github.com/vllm-project/vllm-ascend/pull/21549) · [固定差分](https://github.com/vllm-project/vllm-ascend/commit/6abfc9dee6bc6e14067e0f7704ee5636416b87a5)

**vLLM Ascend #21403 · 11 月 16 日，主分支 `74e578ae`。** 310P MRV2 将采样结果和接收/拒绝计数打包做一次异步 D2H，在设备端改写草稿 token，并用版本化 block-table gather 缓存、向量化主机计算和 pinned staging 缩减步间空隙。作者用 Qwen3.8-27B、310P、TP2、MTP 分别量到 accepted-token staging 的步间隔 149.06→135.08 ms，以及 block table 与主机算术组的 139.66→130.14 ms。两个起点来自不同 profiling 对照，主源未给请求长度、并发和尾延迟。**考虑复现**，分开记录 host gap 与 device window，检查重排和 dummy batch 后的缓存失效。 [合入记录](https://github.com/vllm-project/vllm-ascend/pull/21403) · [固定差分](https://github.com/vllm-project/vllm-ascend/commit/74e578ae83a8adad926a8dd92bd4c12e3ce3da62)

### 昇腾 Mooncake 与 TorchTPU 分别解决分片命中和传输窗口

**vLLM Ascend #21254 · 11 月 16 日，主分支 `faa43c37`。** Mooncake layerwise 的 block key 支持独立 DCP。每个 DCP rank 有自己的对象键及 PUT/GET 会话，所有 PP stage 与 DCP rank 的对象都提交才算命中。DCP=1 保留旧键。当前要求 PCP=1，DCP+KVPP、滑窗/循环状态等未按 DCP 切分的 cache group 以及不满足约束的 block size 组合会被拒绝。作者给出 66 个 CPU mock 测试及 38 个 subtest，覆盖缺 rank miss、跨拓扑和 PP+DCP roundtrip。**考虑复现**，用实际 NPU 集群测部分 rank 超时、冷热重试及外部池命中退化，主源缺端到端吞吐。 [合入与测试](https://github.com/vllm-project/vllm-ascend/pull/21254) · [固定差分](https://github.com/vllm-project/vllm-ascend/commit/faa43c37df0fc67e63ed9775cb86693c3735a389)

**vLLM TorchTPU #2373 · 11 月 15 日，主分支 `15be35da`。** Raiden consumer 可用每 rank host staging budget 或显式页窗口，把大请求的 full-attention KV 切成子传输，lease 预留同存储上的 GDN 状态块。生产方按 consumer 的请求发送，双方都要求支持 sub-transfer chains 的 tpu-sync。逐层 full-attention tag 会回退为整请求 lease，预算放不下一页加状态时初始化会报错。差分附窗口预算、混合状态及请求派发测试，尚无跨主机 HBM/RSS 或 TTFT 对照。**考虑复现**，同时看 host RSS 峰值、长请求的子传输数和短请求排队。 [合入记录](https://github.com/vllm-project/vllm-torchtpu/pull/2373) · [固定差分](https://github.com/vllm-project/vllm-torchtpu/commit/15be35dad1228448758a139f6d72644a3a509000)

### ROCm ATOM 补齐 DeepSeek-V4.1-Flash 的交互路径与多源草稿 KV

**ROCm/ATOM #3302、#3130 · 11 月 15、16 日，主分支 `f3d3bcf9`、`10c300b6`。** 前者让 MI355X 的 DSpark high-interactivity monokernel 覆盖 TP2/4/8，并在并发缩减时一并清空已退出槽位的 indexer 边界，避免 Top-K 等待没有生产者的块。TP8 的共享专家 FP8 线性另处理 `K=288`，通过零填充到 320 满足 grouped GEMM 步长。作者测试覆盖 8→6→5→7 请求的槽位缩减，以及 `288→320` 的 kernel 对照。后者在 gfx1250 用两组 CSR 索引直接读取上下文与草稿 KV，省掉拼接；其他架构继续走原路径，新增 dense oracle 和索引测试。两组结果尚缺可横向比较的端到端服务数据。**考虑复现**，按架构和 TP 分开看活跃请求收缩、草稿接受率和尾延迟。 [MI355X 固定差分](https://github.com/ROCm/ATOM/commit/f3d3bcf9f532225be53e96ceb39904e93c8539d8) · [gfx1250 固定差分](https://github.com/ROCm/ATOM/commit/10c300b689a78030ccc712b3f3860645910d0111)

**ROCm/ATOM #2927、#3269 · 11 月 15、16 日，主分支 `ac966417`、`8220b81a`。** 编译缓存加入 HF 模型配置因子，层数等配置差异可使同一实现代码产生不同缓存键，避免形状近似时重用错误产物。路由侧在 KV 事件流基础上加入分片前缀索引，收到 seq gap 或 clear 只撤销该 rank 的持有记录；cache-aware 以 token 重用计分，媒体请求沿用已有路径。后者是分阶段合入中的索引和接线，暂无跨进程故障率或命中提升数据。**建议深入阅读**，可把配置变更、序列断档和媒体降级加入发布回归。 [编译缓存差分](https://github.com/ROCm/ATOM/commit/ac9664173d83165624e703af69fa7910b49c2474) · [KV 路由差分](https://github.com/ROCm/ATOM/commit/8220b81a4a3da3d90787d65210fdfe944c7f78b8)

**ROCm/AITER #7872 · 11 月 15 日，主分支 `95d21128`。** gfx950 MLA PS 汇编对象更新尾 tile 的页表寻址，避免无效 lane 读到 KV slot 0 后将 NaN 带进输出。最终差分替换 34 个二进制对象，提交说明给出涵盖部分 FP8 和 A16W16 变体的再构建、NaN 注入和干净输入对照，并列出未重建变体及原本就非逐位确定的变体。**考虑复现**，从已部署的 .co 变体和实际页布局入手，做正确性回归。 [固定差分与作者验证](https://github.com/ROCm/aiter/commit/95d21128485c56fbadb37d5d158508985da5615b)

### MLXcel 把长提示内存峰值拆到 logits、状态扫描和 KV 生命周期

**lablup/mlxcel · 11 月 15–16 日，主分支 #3090、#3087、#3096、#3085、#3083、#3104。** #3090 让单位置采样的自回归 prefill 在末端归一化和 head 前选一行，多模态 wrapper 透传该 hook。M5 Max、28K 提示、greedy 48 tokens 下，Gemma-3n-E2B 4bit 单次 prefill 的进程峰值 38.17→23.09 GB，两轮 greedy 输出一致。两款 VLM 的同轮对照只用文本。#3087 将 Falcon-H1、Granite Hybrid、PLaMo 2 的 Mamba2 SSD 扫描改为 256-token 分块，部分测试从 OOM 变为可完成。Granite 4.0 H Tiny 4bit 的 2,048-token 分段峰值 9.85→7.46 GB，作者的 teacher-forced 判定位置有 4/16,148 分歧，Falcon-H1 tiny 的长序列 greedy 也有变化。 [末行 logits `4713efdf`](https://github.com/lablup/mlxcel/commit/4713efdf2130178d09754139c1f42b1190958d63) · [SSM 分块 `05312c60`](https://github.com/lablup/mlxcel/commit/05312c602ebbe19e9635a1f5ea48fd7a569dc129)

同一波提交显式释放被替换的 FP16 KV 缓冲，并限制滑窗 cache 首次保留的历史。M5 Max 的 Qwen3-4B 4bit 在 28K/2,048 分段下峰值 10.94→7.26 GB，DeepSeek V2 Lite 则为 20.20→20.20 GB。Gemma 4 12B 4bit 在约 28.7K 单次 prefill 为 27.38→18.67 GB，2,048 分段均为 14.17 GB。另两项分别把部分模型的 MLA RoPE 分数折入 SDPA、对 Llama 4 单行 prefill 按 chunk 切开计算以省稠密 mask。前者在 Kimi-VL-A3B 4bit 单次 prefill 由 76.31→32.03 GB，但 greedy 输出有差异；后者在 Llama 4 Scout 4bit、19,717 tokens 的单次 prefill 由 72.88→68.41 GB，耗时受共享主机波动影响。**考虑复现**，按提交单独做内存曲线，并对采样、滑窗回退、序列分段和多模态输出做同协议检查。 [KV 增长 `af1ac454`](https://github.com/lablup/mlxcel/commit/af1ac45459760245926cfa3dcd74f55048b677ab) · [滑窗 `2524e820`](https://github.com/lablup/mlxcel/commit/2524e820db36d08f21d50075711777225885f491) · [MLA `cd6715c7`](https://github.com/lablup/mlxcel/commit/cd6715c74ef482a86b8ae8b8452ecbdbdae663f0) · [Llama 4 mask `ab2f1d07`](https://github.com/lablup/mlxcel/commit/ab2f1d07b4f877917126ae91155dbf8e8828242f)

### TensorRT-LLM 把 MoE 实现钉到 tactic ID

**NVIDIA/TensorRT-LLM #21778、#21585 · 11 月 15 日，主分支 `ef83ce24`、`c2ea6bc5`。** 原型 `MoeConfig.tactic` 可按注册 ID 和 checkpoint 格式约束 MoE 实现。选择项仍需满足 SM、激活、EPLB、LoRA 和依赖条件。选中族后，有不兼容的层会构建失败并报告原因。混合 BF16/NVFP4 的 checkpoint 要检查两种格式，草稿及 MTP 可用单独配置，少数模型保留专门路径。另一个提交在 experimental Modeling V2 接入针对 Rubin SM107 的 GLM-5.2 NVFP4 StairCase 高吞吐目标，差分包含 DSA 索引、MLA、MoE 与 MTP，主源未披露可比的服务时延或吞吐。**建议深入阅读** tactic 测试与格式映射，并在部署校验中同时固定 target、draft 及两类专家的实现。 [tactic 固定差分](https://github.com/NVIDIA/TensorRT-LLM/commit/ef83ce246944131336e657681a951fa1c786bc6d) · [Rubin 固定差分](https://github.com/NVIDIA/TensorRT-LLM/commit/c2ea6bc52ab48f106fef8a9963e2d629de93a0e5)

### CANN 和 TorchNPU 收紧量化、归约与编译契约

**CANN ops-nn !14378、!14225 · 11 月 16 日已合入。** `aclnnDequantBias` 的手写入口加入连续化和按输出 stride 回写，修正非连续输出可能返回成功却留旧值的路径。差分确认调用链，作者把 910B/950 板上验证列为待完成。另一个提交拆分 QuantizeAddLayerNorm 的 arch22/35 tiling，修复 950 上非对齐 Welford 尾块和极小 epsilon 的处理，并对齐 bias、shape、BF16+FP32 组合与可选输出。作者报告 950 的 Host/Kernel/Graph 定向用例通过，其中 Kernel UT 8/8。**考虑复现**，优先走实际 fused tensor view、非对齐尾块和生产 CANN 包，主源尚无正式组件版本映射。 [DequantBias `c7d4c74f`](https://gitcode.com/cann/ops-nn/commits/detail/c7d4c74f2fbc631665ada6c47367aafd33fd924b) · [合入与文件](https://gitcode.com/cann/ops-nn/merge_requests/14378) · [LayerNorm `a3315802`](https://gitcode.com/cann/ops-nn/commits/detail/a3315802f4f5ad0ce89961a27f73c0c6863243e9) · [合入与测试](https://gitcode.com/cann/ops-nn/merge_requests/14225)

**Ascend/pytorch !49407 · 11 月 15 日，master `f7d5fda4`。** FlexAttention 的 `enable_outer_main_loop` 默认关闭，前向捕获配置并将其带入图缓存键与 dQ，避免切换配置后暖缓存错用；dKdV 路径不传该参数。作者在 TorchNPU 2.13、Ascend 950、1K BF16 GQA causal 上验证前后向和缓存切换，当前开发包前向仍回退 DynamicCV=False，无完整调度收益或最终 CI 数据。**建议继续观察**，等待编译器配套，试用时分开核对 dQ 与 dKdV 的模板和缓存。 [固定差分与验证](https://github.com/Ascend/pytorch/commit/f7d5fda42ae4ee9b397b48c7f5a9176459f16e68)

### 昇腾移除动态块长 CPP，旧配置需迁移

**vLLM Ascend #21310 · 11 月 15 日，主分支 `839f5383`。** 合入时移除动态块长 Chunked Pipeline Parallelism 的调度实现、profiling 数据路径和文档。release fragment 将其列为 breaking，并要求从追加配置移除 `scheduler_config.profiling_chunk_config` 及旧的顶层 `profiling_chunk_config`。**建议深入阅读**，生产升级前可检查已有 YAML 与固定块长回归，主源未披露迁移后的吞吐对照。 [固定差分及发行说明片段](https://github.com/vllm-project/vllm-ascend/commit/839f538312e3b32b0f09935381c3e4eb78a88e2c)

### 补录｜TPU 稀疏 MLA 的 PCP、逐位置 logprobs 与 reduce-scatter

**vLLM TorchTPU #1940、#3152、#1769 · 11 月 12–13 日，主分支。** 实验性 PCP 将 prefill 分成两端对称的片段分给每个 rank，从 DCP 分片缓存按块 all-gather 到临时工作区。decode 在各 PCP rank 复制请求并走常规 DCP；配置要求 DCP=PCP×TP、KV interleave 为 4 的倍数，当前排除投机、LoRA、PP/DP、多模态与 KV connector。默认工作区约 0.6 GB MLA latent 加 0.27 GB indexer K，需从 KV 预算手工留出。另一提交让 TPUModelRunner 对请求的 prompt token 分块计算 logprobs 并在 prompt 完成时返回，top-k 受运行器上限约束，排除 PCP、DP 词表并行和 LoRA。第三个提交为 8-device TPU v7x 加入分层 TensorCore reduce-scatter，按 microbatch 分阶段执行远端 DMA 和本地累加，受维度和 lane 对齐约束。三个差分分别给出了文档、单元测试或参考路径，未提供共同的端到端吞吐对照。**建议深入阅读**，根据服务是否需要 echo/logprobs 或长 prompt 决定启用路径，测 PCP 通信、临时内存和已支持特性的组合。 [稀疏 MLA PCP `79865142`](https://github.com/vllm-project/vllm-torchtpu/commit/7986514248c3db9111dfddb1c351081a94603552) · [prompt logprobs `ce850dbf`](https://github.com/vllm-project/vllm-torchtpu/commit/ce850dbf2c6a4a5ce0b02097a50c46fc0e5c64b0) · [reduce-scatter `8c81640f`](https://github.com/vllm-project/vllm-torchtpu/commit/8c81640f2b5ffe5c0e6b683c99d6fd1a34e58dfd)

### 补录｜ATOM 在图捕获后复位 V4 真缓存，并增加对应的低精度路径

**ROCm/ATOM #3057、#2989 · 11 月 14 日，主分支。** 捕获预热曾写入真实状态槽和 KV block 0。在 gfx1250，高并发请求可能继承宽 bucket 留下的 NaN；#3057 在接收服务请求前回填 KV、indexer 池与状态 arena，差分提供 split arena 回归。#2989 改造 DeepSeek-V4.1 的连续 KV 来源与 OPUS/ASM A8W4 注意力，并在 gfx950 H64/H128 保留显式 FP4 KV 需求和对应池大小。作者报告 157 个 CPU 测试、82 个 gfx950 GPU 测试，以及 DP4/EP4 H64 FP4 的 16/16 请求完成；覆盖范围不等于同负载下的 FP4/FP8 服务对照。**考虑复现**，用同一批次形状检查 graph warmup 后的首批请求，再对照 cache dtype 下的准确率与 TTFT。 [状态复位 `244913cc`](https://github.com/ROCm/ATOM/commit/244913cc8bdf16a7526004875a24497166c20c9a) · [KV 与低精度路径 `b71273a3`](https://github.com/ROCm/ATOM/commit/b71273a323fab349b3d64f939d11af9648525152)

### 补录｜TensorRT-LLM 拆开磁盘 KV 副本、异步 I/O 和草稿池发布

**NVIDIA/TensorRT-LLM #21822、#19303 · 11 月 14 日，主分支。** KVCacheManagerV2 可选在恢复后保留已提交页的磁盘副本，或在有空闲盘槽时后台写穿。副本配额、异步完成和故障计数与普通 eviction 分开处理。异步磁盘拷贝用进程级线程池拆为 4 MiB 片。另一合入修复配对的 draft 池未发布可复用块，增加联合复用、上下文分叉与快照点测试。已读实现与测试，未见 SSD 型号、命中率分布、放大写量或端到端时延。**考虑复现**，按预热与稳态分别测 NVMe 写量，覆盖容量逼近和回写失败，检查 target 和 draft 的共享前缀可用性。 [SSD 下沉 `9c88c5fc`](https://github.com/NVIDIA/TensorRT-LLM/commit/9c88c5fc59117735f6dbf838634f431a562f7165) · [草稿池 `b754bfb8`](https://github.com/NVIDIA/TensorRT-LLM/commit/b754bfb8eb568623094f2267054788b065e3e472)

<a id="topic-8"></a>
## 评估与可靠性

### Databricks AI Index 并列发布网关使用与私有仓库编码评估

**Databricks · 11 月 16 日公开首期。** 一路是 10,000 多家 Unity Gateway 客户的去标识化聚合流量，首期主要数据为 9 月 21 日至 11 月 7 日；另一路在私有代码库的真实开发任务上用隐藏测试比较 Claude Code/Codex × 模型，成本按公开单价估算。厂商报告 Fable 5.1 通过率 96.0%、$7.47/题，Muse Wonder 为 94.9%、$1.53/题。同页展示的模型点使用其最接近成本—质量前沿的 harness。原始私有任务和完整运行环境没有随榜单开放，结果受客户结构和任务组合影响。**建议继续观察**，迁移前用内部任务做固定 harness 的交叉对照，记录通过、尝试次数和每成功任务成本。 [发行方方法及结果](https://www.databricks.com/blog/introducing-databricks-ai-index-state-enterprise-ai-tokenomics) · [首期指数](https://developers.databricks.com/ai-index)

### RepoFactory 固定发布 200 个仓库重建环境

**IQuestLab · 11 月 16 日 14:51:40 UTC，固定版 `2a2787f7`。** 用可执行测试给完整仓库重建任务反馈，主要覆盖 Python。最终工件目录对应 200 份 Harbor 任务包和 200 份预构建 Linux/amd64 容器镜像。该子集处于环境演化循环之前，当日较早的 1,000 环境快照已由维护者替换，报告提及的全量 10,159 环境属于更大集合。卡片未给统一开放许可，也未提供独立任务有效性结果。**建议考虑复现**，先固定版本按官方 oracle 和空白代理验同一任务，并确认参考实现与测试隔离。 [固定数据卡](https://huggingface.co/datasets/IQuestLab/RepoFactory/blob/2a2787f7a4acf664265b7be8c2fc662a77032c63/README.md) · [固定工件清单](https://huggingface.co/api/datasets/IQuestLab/RepoFactory/revision/2a2787f7a4acf664265b7be8c2fc662a77032c63)

### IH-CacheBench 度量 KV 驱逐后的指令优先级反转

**AuthorityKV 作者 · 11 月 15 日 04:53 UTC 发布 Apache-2.0 数据与原始输出。** 400 个冲突和 100 个正常例用精确匹配比较同一模型在完整缓存与压缩缓存下的决定。三款 7B–9B 模型、三种压缩器、20%/40% 保留预算共 7,200 次冲突比较，作者测得原压缩器共反转 2,055 次，加入角色及来源排序后为 176，再用句级检测为 52。使用人工提供跨度的配置为 0，属于给定额外标注的上界。**建议考虑复现**无跨度标注条件，同时在自己的长上下文任务上统计任务效用。 [固定数据卡](https://huggingface.co/datasets/XingjianZhangDev/AuthorityKV/blob/3df48fbd96e23c639c0e10e248fcccae5e76f4d3/README.md) · [固定代码](https://github.com/XingjianZhang-dev/AuthorityKV/tree/95b5149d9041b940da2a039faf28ea74ccbe34be)

### Sentinel S1 用基座的采样表现训练不确定性标记

**Sentovara · 11 月 15 日 08:09 UTC 发布 Olmo 7B/32B 适配器，16 日 02:50 UTC 发布 Qwen3 8B 适配器。** 对训练题各采样五次，答对至多一次的样本保留原回答并加 `UNSURE:`，答对至少四次则保留直接回答。发行方将 Olmo 7B 的 9,255 题原模型准确率修订为 20.3%，适配后未标记回答准确率为 75.1%，标记自身 94.0% 的错误。Qwen3 8B 关闭思考，使用基座自己生成的通用对话练习，IFEval 80.2%→81.7%，标记自身 83.2% 的错误。未标记准确率因覆盖率下降而更换分母，整体正确率约持平。**建议考虑复现**风险—覆盖率曲线，另计检索成本及分类漏报。 [Olmo 固定卡](https://huggingface.co/Sentovara/Sentinel-S1-7B/blob/a987cd611274cec07faf1c17c4430397e8a7a909/README.md) · [Qwen3 固定卡](https://huggingface.co/Sentovara/Sentinel-S1-Qwen3-8B/blob/f97f344922da8e9e9f4f55087ec17858ac25dc2b/README.md)

### 补录｜公开检查点保留可重放的初始化布局

**Unemployment Labs · 11 月 12 日分析，11 月 13 日伴随重放工件。** 作者从训练尚未充分改变的参数行检索 Philox 种子，用 PyTorch/cuRAND 的并行分配布局重放，在留出行检验匹配。DeepSeek-V3 样本匹配 132-SM Hopper 布局，V4-Flash 和 Kimi K3 分别匹配 148-SM 与 152-SM 布局。第三方 CPU 重放只覆盖其可取样的部分。匹配说明检查点里的初始化布局，完整训练拓扑、芯片型号唯一性与使用地点仍未确定。**建议考虑复现**，固定上游权重并保留错种子、错布局负对照。 [作者报告](https://unemploymentlabs.com/checkpoint-forensics) · [固定独立重放](https://github.com/peterjliu/checkpoint-forensics-repro/tree/8f8ed78cf731195bab9f990a6523e65ecd5b7108)

### OpenAI 图像安全分类器面向获准机构开放试用

**OpenAI · 11 月 16 日，`omni-moderation-csam-latest`。** 内部 CLIP 视觉编码器接三层 MLP，辅助审核人员判断潜在儿童性虐待材料的送审优先级。在自家上传流量同等检出量的比较中，厂商估计需要复核的图片减少 48%。报告将留出集、生产抽样与伙伴试用分开，披露近重复仍存在泄漏风险，部分人群标签未经独立核验。分数仅供审核排序，人工作业与后续流程仍需要保留。**建议继续观察**跨平台的独立召回和误报证据。 [公开卡与评估](https://deploymentsafety.openai.com/csam-image-classifier) · [访问公告](https://openai.com/index/strengthening-online-child-safety/)

### OpenAI 公布四家外部机构的驻场评估方案

**OpenAI · 11 月 16 日，首阶段公告。** Apollo、Goodfire、METR、Transluce 分别覆盖隐蔽违背目标、可解释性监控、前沿 RL 轨迹和既有后训练对齐评估的覆盖性。按项目范围开放选定检查点、训练或评测数据、日志和监控系统。安全委员会可收到未删节报告，评估方保有发布独立性，公开稿仍受机密删节和负责任披露时限约束。此次披露的是协议与已签首阶段安排，仍待看到外部报告。**建议继续观察**项目实际访问范围与公开报告的可验证程度。 [试点及披露协议](https://openai.com/index/our-first-phase-of-embedded-assessments/)

### 主窗｜Agent 裁判对同一条轨迹的判断会因隐藏执行证据而改变

**Manipal University Jaipur 等，arXiv 2611.18471 v1，11-13 17:44:01 UTC 投稿，11-16 公告批。** 固定任务结果、评分标准和实际轨迹，只切换裁判看到的执行证据，包括多步宏调用、委托或历史压缩表示。四项基准、七个裁判模型和四套裁判协议的合并结果中，原本答对的判断在三类抽象下转为错判或无判决的比例依次为 31.0%、30.9%、20.3%。恢复源证据后，85.5% 的失败判断恢复，但此前保留正确的判断仍有 5.2% 退化。比例的分母是相应原始正确或抽象失败集合，证据复原来自同一冻结轨迹。适合对自有摘要和委托机制做配对审计，记录列出的日志证据、裁判置信区间和无判决率。论文所列 GitHub 工件本轮返回 404。 [论文 §3–6、表 1/2、图 3/4](https://arxiv.org/html/2611.18471v1#S5.SS1) · [版本记录](https://arxiv.org/abs/2611.18471v1)

### 主窗｜AscendGYM 用真实负载形状和搜索成本衡量昇腾算子优化

**上海创智学院（SLAI）AI Training Platform Team，arXiv 2611.17621 v1，11-13 00:45:36 UTC 投稿，11-16 公告批。** 48 道任务从 MindSpeed-LLM、vLLM-Ascend 等负载的 profiling 筛选，保留调用阶段、shape、dtype 和部署配置。候选须先通过 CPU 参考数值校验，再由 AscendForge 用 msprof 测 device 时间，STS 对消耗较晚才取得的增益扣分。首批只评 17 个以访存为主的基础算子，约七成模型与 harness 组合的几何平均加速低于 1.2 倍。GPT-5.6 在三个 harness 均完成全部任务，Codex 与 Claude Code 的几何均值仍分别约 1.69 倍与 1.05 倍。剩余 31 项融合及复杂 attention 未报告实测，算子计时未给端到端加速。固定代码 `51b0b55b` 的 `bridge/sandbox_evaluator.py::parse_profiling` 解析设备计时，`eval/native_harness/core/metrics.py::sts` 从搜索轨迹计算预算分。**复现建议**对齐硬件、CANN、基线、评测尝试预算，分别记录优化成功率、搜索成本与真实训练 step 或推理时延。 [论文 §3–4、图 3–7](https://arxiv.org/html/2611.17621v1#S4) · [版本记录](https://arxiv.org/abs/2611.17621v1) · [固定代码](https://github.com/SLAI-AITP/AscendGYM/tree/51b0b55bc313d7cdeb4a4bbd53b071258dce07f4)

### 主窗｜生成 Agent 回归测试集应按受影响的工具与条件单独判分

**Pulkit Yadav，arXiv 2611.17632 v1，11-13 01:03:19 UTC 投稿，11-16 公告批。** 对 τ-bench 航空和零售的工具、规则做受控变更。若变更只影响有明确覆盖关系的用例，其对全套通过率可产生的最大降幅等于原来通过且受影响的用例占比。航空例中，预设 10 点门槛并要求配对置信区间排除零，生成套件发现 1/7 个变更，等大小子样和人工套件均为 0/7。有一个工具移除让原先通过的受影响测试全部失败，全套均值仍因别处波动而上升。仅验证单一 Agent 模型，两域共用同一模型做用户模拟和 LLM 裁判，局部评分和多样性消融属事后分析。**复现建议**对工具和行为路径预注册分层判分，保留重复 rollouts 估计波动。 [论文 §3–6、表 1](https://arxiv.org/html/2611.17632v1#S4.T1) · [版本记录](https://arxiv.org/abs/2611.17632v1)

<a id="coverage"></a>
## 实际覆盖、补录与续查

本轮收录 **138 个带稳定标识的版本事件**。正文标注回溯补录和更正。数值均来自所列来源，本轮未执行独立复训、设备压测或付费 API 实测。

- **厂商与模型工件。** 检查 145 个官站与评估入口，114 个直连 HTTP 200，另用固定页和网页复核关键主源。176 个 Hugging Face 组织的 353 个定向目录页面可读，两路全站创建目录接回上一轮重叠窗口，定向读取 34 张固定卡。创建索引只证明条目可见，长尾工件、已存在模型的修订及部分受限官网仍待补查。

- **论文与会议。** arXiv 十分类 new/recent 核对 11-16 正式批，653 个唯一 ID 全部关联 OAI 版本，CS/stat 元数据桶 1,055/114 行无续页。11-17 01:00 UTC 是名义公告，截止后复查 new/RSS 仍只看到 11-16，开放最后 64 秒的核对窗口。Atom 精确投稿/修订查询 429。HF 11-16 39 条、11-17 HTTP 400；OpenReview 三组 API 403，目录与论坛精度不足；EMNLP 全量卷传输中断。RigoLLM 四份固定 PDF 与官方议程已读，SC26 Agentic Memory Hierarchy 已核作者完整讲义及固定回放工件，论文 PDF 待查。十一月 16 日未归档的取证经复核合入 11-17，MegaOps 原 11-13 公告作补录。本轮无独立复训或硬件压测。

- **开源工程**　主窗为 11 月 15 日 08:57:48 至 11 月 17 日 09:01:04（北京时间，左开右闭），回溯至 11 月 12 日 08:57:48。核查 105 个 GitHub 仓库地址（含别名及扩展生态），75 个默认分支扫描跨过回溯起点，对应主窗 1,297 条提交。79 个 merged PR 仪表板完成分页，PyTorch 仪表板空结果与默认分支提交记录冲突，采用固定提交补证，交叉核验后的完整清单为 78 个。发布 Atom 57 个可读，补查分页后 53 个发布列表达到窗口要求，19/19 个 PyPI 查询可读。读取 90 组固定提交或 tag 差分。与 40 份历史日报去重后收录主窗 44 项、原日期回溯补录 3 项，包含 SGLang #50156 的撤回状态。性能与回归数字来自项目报告，本轮未独立运行设备实验。
- **工程续查**　GitHub 直连和匿名 REST 分别出现 429/403，部分提交列表、合入仪表板和发布分页仍缺页。逐仓实际扫描结果与失败记录保存在 ledger.json，47 个事件到固定提交、官方发布、PyPI 和台账记录的对应保存在 evidence-map.json。继续追踪 TorchTitan #6097 的 packed staging 和 NeMo RL #5139 的 rollout/trainer 快照接入，两条本轮仍为 OPEN。FlashInfer 的候选发行需继续核对发布时间、分发工件和提交。覆盖仅代表已打开的来源、分支和时间区间。

- **硬件与软件协同。** 本轮检查 NVIDIA、AMD ROCm、AWS Neuron、Google TPU/JAX、昇腾 CANN、MUSA、Intel、Apple MLX 及部分国产适配，发出 520 次请求，332 次传输成功。重新核对 11 月 16 日未归档的固定差分，新增检查 9 个 GitCode 合入与差分。MLPerf Training Explorer 可查，原始运行日志及部分厂商入口仍返回 403、429 或 404。GitHub 限流覆盖缺口从 11 月 16 日 09:37:58 至本轮截止，部分旧候选和维护分支同样待补扫。CANN 发行映射、libtpu 源码差分、MUSA Kimi 训练一手记录及 SC26 后续场次继续留查。

- 全局完整覆盖水位仍为空。下一轮以本次发现截止续扫并回溯 72 小时，各来源继承尚未完成的区间。本地 `.monitor` 保存访问回执、候选及已收录版本。
