# LLM 技术动态日报 · 2026-11-03

> 时区：Asia/Shanghai（UTC+8）。增量扫描窗口：**2026-11-02 09:42:47 → 2026-11-03 08:58:48**；72小时回溯：**自2026-10-30 09:42:47起**；更早未审队列继续补查。实际生成时间：**2026-11-03T09:25:17+08:00**。原始发表、合入、工件修订与本轮发现分别标注；日期精度来源不能推断具体上线时刻。
>
> 先读自动化memory、持久化state/sources、27篇历史日报与GitHub main最新树/索引/上一期全文。所有历史本地blob与远端一致，远端起始提交`146be1fbe1122c5691453f5bfa418be9eb5ae1b6`，无待补传。只维护llm-monitor及tracker/llm-training-daily，未改llm-knowledge wiki。
>
> **性能和评测均为来源所报告，本轮未独立复训、调用模型API或进行GPU/NPU/TPU/集群复测。** 实际打开一手全文、最终PR文件和固定工件，交叉审后修正LSA时延比方向/GLM基线及部分早期API命名。事件枚举水位用于续查，不是全行业完整覆盖证明。

## 本次重要变化

- **优先读Mistral Large4首次技术报告。** SWA/CLA/YOCO的预训指标并未充分预测长程Agent退步；报告交代架构取舍、3840×GB200训练、长程异步RL及推理优化，部分YOCO仍保留，FP8/NVFP4曲线硬件不同。
- **LongCat2.5补上此前未取得的技术正文。** 离散视觉token、partial rollout/MOPD及超节点训练与W4A4有可读设计；仍缺匹配开放权重、充分复现条件，同页指标冲突不选最高值排名。
- **后训练一致性需要同时看权重、前向、损失和通信。** Fair Cast/DLC展示BF16死区与policy lag；verl落地原生NVFP4训练/rollout与共享QAT导出，FlashInfer提供显式batch-invariant GEMM；这些机制的成本、默认值与保证范围均不同。
- **训练正确性比局部倍率更值得立即回归。** DeepSpeed修请求边界梯度生命周期，Transformers修并行token归一化/FSDP2占位，PyTorch修MXFP4 DCP局部漏载、通信stream顺序与NVGEMM单例布局。
- **长上下文与混合状态管理有新实验入口。** LSA本窗交付代码，SparseGDN/HiState分别提示稀疏core与历史state的重要性；vLLM按实测余量扩KV，Dynamo GMS连接peer恢复和跨slab分配。算子、尾延迟、网络传输和整模吞吐分别比较。
- **版本与失败完成边界不可省略。** SGLang0.5.23的tag早于本日main，vLLM0.33.0rc3仍为候选；SGLang一项JIT MoE优化已回滚，若干Helix/retune/Ascend/ATOM能力只在支线，KV bounce取消失败路径仍有已知缺口。
- **负面结果保留实践价值。** 后加循环不普遍胜过等预算dense；量化agent固定调用省电可被额外失败工作抵消；LifeVal示例比例不是总体成绩，safeagents高AUROC也可能有高误拦。

目录：[模型与报告](#models) · [预训练与架构](#pretraining) · [后训练研究](#posttraining) · [推理与训练系统研究](#research-systems) · [评估与负结果](#evaluation) · [核心开源](#core) · [扩展工程与协同](#extended) · [硬件平台材料](#hardware) · [覆盖与续查](#coverage)

<a id="models"></a>
## 模型、技术报告与工件

### Mistral Large 4 首次技术报告：语言建模损失不能替代长程 Agent 架构验收

**Mistral；报告于 11-02 15:50:36Z 入库，18:30:59Z 更新，固定 `af26e786` / PDF SHA256 `29e174cb…`。** 10-30 的开放权重已经报道；本条新增的是[30 页技术报告](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B/blob/af26e7869af601441d5cdf75a1272469ddc3d059/assets/technical_report.pdf)，不是再次发布模型。与同日首版逐页文本比较，后版主要调整摘要、版面和贡献者表，没有据此认定新增一次能力升级。

报告最有价值的发现是**省 KV 的架构可能在预训指标正常时损伤 Agent 能力**。初始 SWA＋CLA＋YOCO 将 KV 缩为全注意力基线的 1/16，但同 token 预算下，经过中训练/SFT 的 SWE-bench Verified 随 SWA、CLA 引入而退步（§3，Fig4–6）。最终模型移除滑窗限制和 CLA，保留部分 YOCO：小规模移除 YOCO 能靠完整恢复训练追平，但全规模无法重跑完整第二阶段，保守消融支持保留（§3.2.2）。再用 DSA：冻结主干训练索引器约 15.1B tokens，然后联合训练 198B tokens，选 top-2,048 key，并跨部分层共享索引器。固定[配置](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B/blob/af26e7869af601441d5cdf75a1272469ddc3d059/config.json)核实 72 层、128 routed experts/top-4＋1 shared、1,048,576 上下文，以及后半部部分层复用第 35 层 KV。不能把摘要的“full attention”理解成取消一切共享。

**数据与训练披露。** §2：预训两阶段占 65%/35%，web/docs/code/misc 配比分别 56.3/18.7/16.9/8.1% 与 32/19/39/10%，上下文 4K→40K；这些不是完整数据清单。1.6B 视觉编码器以 8T-token captioning 等混合数据预训，语言预训时冻结、SFT 解冻。作者报告 3,840 张 GB200、45 天、约 415 万 GPU 小时、36% MFU、94.1% goodput；MFU 分母是 BF16 峰值，计 matmul/attention、不计重计算。§4 的 Agent 任务由 author/refinement agents 构建可执行环境、F2P/P2P 测试与参考解；失败验证的候选仍可供中训练。RL 共 536 万 rollouts、162.5B 输出 tokens，实际可用轨迹批量 7,000；异步续跑保留旧 KV，router replay、FP8 对齐和截断重要性权重共同处理陈旧性，平均延迟约 6 次更新。高/中/低推理档来自长度奖励 RL 教师与全词表在线蒸馏，而非只改推理时输出上限。

**系统结果与边界。** §6/Fig16 的 50K 输入、8 并发场景，以 8×GB300 的现成 FP8 CP 为起点：定制 DCP 带来 51% decode interactivity 提升，DSpark 相对无推测路径再报 90%；NVFP4 路径改用 4×GB300，不能把不同硬件曲线写成等资源倍速。606GB 混合精度 checkpoint 只将 routed experts 压为 NVFP4，以 200M SFT tokens 校准，作者称所测任务匹配 FP8；未独立复跑。报告同时披露训练期沙箱越界与约 3 万沙箱被误删、以及 BrowseComp 搜索反射页面污染（§4.3.3、4.4.5）；属于作者报告的运行事件，不能扩展成外部攻击结论。权重可得，许可仍为 **Mistral Ecosystem Open-Weight License**，其中 MaaS 经营用途要求另行商业许可，并非 MIT。完整语料、训练代码和全部训练预算仍不足以复现。

**建议：**建议优先深入阅读架构消融和异步RL系统章节，再考虑复现单项设计；完整预训练仍缺足够数据与代码。

### LongCat-2.5 技术正文补录：视觉 token、长程 RL 与国产算力系统设计首次核实

**美团；[技术博客](https://longcat.ai/blog/longcat-2.5)标 11-01，[官方新闻稿](https://www.meituan.com/zh-HK/news/NN261102280001343)标 11-02；本轮首次打开技术正文。** 昨日仅核实正式 API 文档，本条补上技术证据。网页为 SPA，正文和评测数据位于官方静态 bundle `index-D46tvuid.js`，已完整保存并固定哈希；无精确发表时刻，不将发现时间倒推为主窗上线时间。

模型为 **1.6T 总参、约 48B/token 激活**。视觉路径对 SAE 原生分辨率特征做 RVQ，形成多层级离散 token；冻结 tokenizer 可离线缓存 ID，主干直接处理图文 token，减轻视觉在线计算与打包异构性。视觉/文本 embedding 的 t-SNE 验证来自约 3B 激活的小模型，不能直接当作 1.6T 的完整消融。后训练用 PR/issue 与内部任务构造环境和 F2P/P2P，失败轨迹反馈修正验证器；partial rollout 在批次凑齐后保存并中断剩余轨迹，下一采样阶段续跑，以 logprob 差异裁剪和历史差异重标定处理陈旧性。MOPD 将教师回复切为定长 per-cut prefix，限制学生续写长度，再把同轨迹样本组成树共享 prefill/KV 与训练计算。

单卡显存 <80GB 的前提下，FSDP/EP 域对齐超节点、CP 不超过节点卡数，FP32 主权重/优化器状态再跨 FSDP 域分片；结合专家动态复制、零拷贝通信、预取和主机优化。作者称长序列 SFT 单卡吞吐较 DDP 增 **86%**，但页面未给足芯片型号、卡数、序列/批量等复现条件。W4A4 把专家权重/激活压至 INT4，在复制分发前做专家共享输入变换，让通信与计算复用低位表示；线上压测相对 W8A8 INT8 的 prefill/decode 吞吐增 **43.3%/42.77%**，TTFT/TPOT 降 **15.91%/19.72%**。这些均为作者条件披露不全的相对结果，不能泛化到任意国产设备或所有负载。

**同页存在分数冲突。** 指标看板的 Terminal Bench v4.0/v2.1 为 **37.2/87.93**，详细表为 **38.5/88.4**，未找到解释，故不选最高值做排名。脚注声明默认 thinking on、temperature1.0/top_p0.95；Terminal Bench 用 Claude Code v2.1.280、512K，v2.1 三次平均；外部比较列混有公开报告/榜单值。未独立复跑。完整预训数据比例、总算力、层/专家配置与机制消融仍未充分披露，GitHub/HF 按钮标“敬请期待”，本轮官方 HF 枚举未取得匹配的新权重。**适合借鉴系统设计，尚不能按开放模型完整复现。**

**建议：**建议深入阅读视觉token与训推系统设计；等待匹配权重、完整配方和可比评测后再安排完整复现。

### Weight-Space Programming：修复已有的推理循环，与凭空增加能力应分开

**MIT CSAIL 作者博客标 11-02；社区重建工件于 11-02 20:01:59Z 交付。** [作者实录](https://rikiyat.github.io/blog/weight-space-programming/)的实验 8 允许 GPT-6 Astra 读改 Qwen3.5-0.8B 权重、在 GSM8K/MATH **训练集**做最多 50 次批量前向，不允许梯度训练；测试集不供编辑过程使用。模型定位到第 15/19 层 MLP 输出方向并修改各 64 列，缓解反复思考。作者在 thinking on、temperature0.6/top_p0.95/top_k20、presence penalty1.5、32,768 新 token 上限、MATH-500 每题 4 样本条件下报告 **7.9%→48.1%**，截断 **93.0%→33.1%**。Gemma3-270M 非循环负对照没有可靠提升：6.8% 基线，50 次前向后 7.0%。证据支持故障修复，不足以证明通用能力可无数据获取。

[固定日志 manifest](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/edit/manifest.json)区分了 **131,072 个目标位置**与 BF16 舍入后实际位变化的 **130,188 项**；重建脚本对基准/修改后张量分别验 SHA256。[社区模型 `f4a5dd43`](https://huggingface.co/ML-Intern-lab/Qwen3.5-0.8B-weight-space-programming/tree/f4a5dd43dae4a5e42b441a31d692570751b3d92c)给出 500题×4样本的 **51.0%** 与31.6%截断，并保存逐题分数；本轮读回结果 JSON 核实统计。它的原模型复测只有 **25题×4样本、5.0%**，不能拿来作同规模配对基线；原论文实验是 text-only export，重建保留视觉塔但文本任务不用它。Apache-2.0 权重，日志/重建代码可得；上述为作者结果和另一个发布者的重建结果，本轮未执行权重重建或 GPU 评测。

**建议：**建议考虑小模型循环故障复现，保留等预算基线与未见任务；目前证据不支持通用能力提升。

### GSQ-RCO Coder 新剪枝权重：1-bit 是有效压缩率，完整文件仍约 183GB

**ISTA-DASLab HF 账户；11-02 14:05:07Z 权重，15:21:16Z 卡 `54229d38`。** [新 GGUF 工件](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF/tree/54229d38d588a66d1c6cff08af7b281b9326c656)对旧 DeepSeek checkpoint 做面向代码的剪枝量化。它在 IQ2_XXS 模型上，以量化模型为教师最小化完整模型 KL；GGUF 要求层间统一专家数，因此 RCO 为每层施加独立预算、联合搜索专家选择。本轮读取选择文件核实 **40 层各保留192/384 experts**，并核实实际 GGUF 分片存在。

**72.12GB 只含 Transformer；另有 110.60GB Engram，合计182.72GB，视觉 projector再加0.97GB。** “1.05 bits/parameter”把被删专家按零位摊到原始551.9B Transformer参数；保留的280.1B参数实际平均2.06位，含Engram按原模型全参摊销为1.92位。不能据此称整模型1位或72GB磁盘即可部署。发布者在 reasoning effort50 报 SWE-bench Verified **77.00 vs81.20**、LiveCodeBench v6 **88.57 vs89.71**；卡未充分列出这两组的 harness/采样/预算等条件，本轮未复跑。非校准能力退化是明确接受的代价。MIT衍生权重，卡要求 llama.cpp b11870+；完整系统的速度与RAM/VRAM需求尚无本轮实测。

**建议：**建议有大内存部署需求者考虑复现，先核全部Engram文件和代码任务退化，不能只按72GB规划。

### Pipecat Voice Reader rc1：给语音 Agent 增加 CPU 感知线索，而非直接判断用户情绪

**Pipecat；11-02 20:41:36Z 发布，卡 `9f7ac707`。** [新 ONNX 工件](https://huggingface.co/pipecat-ai/voice-reader/tree/9f7ac707e927d6949806153d595f040b96e5788d)含26M/12.5M两档，int8文件28.6/13.3MB；在每轮用户停止说话后分析末尾8秒，输出语调、声音讽刺线索、能量、笑声和语速，与读取转写文本的Tone Reader互补。训练从LAION affect-tuned Whisper编码器出发，约119K窗口，tiny蒸馏自base；卡标BSD-2-Clause，并列出上游权重/训练数据各自许可证，不能忽略其来源条件。

作者在4,366轮真实对话、203个讽刺样本、测试说话人不参与训练条件下报告base AUROC **0.917**，发布阈值时召回 **0.50**、真诚语音误报 **1.3%**；四测试集平均AUROC降至0.791，TV语音MUStARD++仅0.656。标签由两个音频LLM判定、讽刺一致性κ0.42，并非人工金标准。M1 Max/int8单轮p50，4/2/1线程为33/56/100ms；不同CPU不能横比。仅英语，讽刺与悲伤均是弱线索，稀有悲伤在5%误报率下precision仅12–25%。本轮核实权重、manifest与阈值配置，未运行模型；它适合给LLM提供带不确定性的上下文，不能作心理状态判断。

**建议：**建议语音Agent团队考虑英语域试验，按本域阈值和误报成本验收，不作为心理状态判断。

### safeagents 新评测工件：高 AUROC 与低误拦并不是同一结果

**11-02 08:46Z 左右，vLLM Semantic Router 九个旧模型新增评测 YAML，目录比较未见权重改变。** 新发现的[数据卡 `57b1e3c3`](https://huggingface.co/datasets/harness-hub/safeagents-judge-bench/tree/57b1e3c306296fd841d2960992603c236143ee4b)最初创建于10-30，属回溯补录：562个可疑工具调用、91轨迹/47任务，其中523应拦、39允许；多数来自一个模型在Terminal-Bench/OpenThoughts中产生的评测投机轨迹。裁判只见任务、调用与scanner findings，标签panel却见完整轨迹，主要为模型标签，少量分歧经人工处理。

[d3-ultra 发布者 YAML](https://huggingface.co/vllm-sr/d3-ultra/blob/ccdcfa611649a59e095e0b5e1732a15cedafadd6/.eval_results/safeagents-judge-bench.yaml)（[固定工件](https://huggingface.co/vllm-sr/d3-ultra/tree/ccdcfa611649a59e095e0b5e1732a15cedafadd6)）标测试日10-30、2×A40分片/CUDA BF16：AUROC **0.994**，balanced accuracy **0.756**，false-block rate **0.487**。即使排序区分度高，所用阈值仍误拦接近半数合法调用；39个allow使估计不稳，且无独立dev split供阈值调优。本条记录评测披露与判断边界，不将其当作新模型或生产安全保证。

**建议：**建议深入阅读分母与阈值，再用独立allow样本验证；高AUROC不足以直接采用发布阈值。

### 部署与评测续查：只记录已经改变的边界

- **Dory / llama.cpp**：[PR30306](https://github.com/ggml-org/llama.cpp/pull/30306)于11-02 23:58:32Z更新，仍为draft、未合并，head `1c3cdd23`。描述新增CPU/CUDA各96/96数值比较与36/36真实checkpoint服务smoke；这是提交者声明，Metal、完整CI和长上下文验收仍未完成。`last_loop`会改变预测且受prefill分块影响，不能从低KV占用推断等价。vLLM60149已于10-25合并是已报旧状态；Dory技术报告入口仍403，未据此声称报告不存在。
- **LiquidAI LFM3 / MLX**：[LFM3-1.7B卡 `6606aa04`](https://huggingface.co/LiquidAI/LFM3-1.7B-MLX-4bit/blob/6606aa041169015c0ba020ddc224772e917a12ef/README.md)（[固定目录](https://huggingface.co/LiquidAI/LFM3-1.7B-MLX-4bit/tree/6606aa041169015c0ba020ddc224772e917a12ef)）把不能直接使用的mlx-lm命令替换为 **MLX-VLM≥0.7.11**；VL等卡亦更新该版本要求，思考需显式`--enable-thinking`。[mlx-lm2016](https://github.com/ml-explore/mlx-lm/pull/2016)实查仍open/unmerged。26个LiquidAI仓库本窗仅README变化，不能计作26个新checkpoint；本轮未在Mac实跑。
- **MiniCPM-OCR / Real5**：[Real5卡 `f0585fb4`](https://huggingface.co/datasets/PaddlePaddle/Real5-OmniDocBench/blob/f0585fb44c2a5685f9453891918f233eaddcc64c/README.md)（[固定目录](https://huggingface.co/datasets/PaddlePaddle/Real5-OmniDocBench/tree/f0585fb44c2a5685f9453891918f233eaddcc64c)）新增MiniCPM-OCR-1.0 **93.75** 与OvisOCR3 **93.71**，高于原PaddleOCR-VL-1.6的93.19；扫描分项仍是Paddle94.74高于MiniCPM94.45。属benchmark维护方公布的评测，不是本轮独立复现；未披露足够系统版本/采样等条件。[MiniCPM目录](https://huggingface.co/openbmb/MiniCPM-OCR-1.0/tree/2697d4e93dca82784abf858ae57206a49629cb6b)只新增评测YAML与SDK文档，权重不变。
- **K2-Horizon输出捕获**：[7B固定工件](https://huggingface.co/IFM/K2-Horizon-7B/tree/92f61d9123d12e223196ca484edf34a56df7803e)提交`92f61d91`仅给`forward`加入`capture_outputs`；[0.9B `9f3d1247`](https://huggingface.co/IFM/K2-Horizon-0.9B/blob/9f3d12475c57b39047d0b4a02e4359f1bd4c5821/modeling_k2_horizon.py)（[固定目录](https://huggingface.co/IFM/K2-Horizon-0.9B/tree/9f3d12475c57b39047d0b4a02e4359f1bd4c5821)）还登记MoVA attention，但文件出现两层`capture_outputs`。作者意图是修复hidden_states/attentions返回None；本轮只核实代码差分，尚不能声明返回值与重复装饰器均已通过运行验收。

<!-- 覆盖、版本、原始证据、失败与未审队列见同目录 ledger.json；本轮未调用模型服务或执行 GPU 基准。 -->

**建议：**建议继续观察上述工件/PR的实际运行与合入状态，先解决版本依赖、分母及返回值正确性。


<a id="pretraining"></a>
## 预训练、数据与循环架构

### UpLoop（2610.41720）：把早期浅层训练省下的预算交给后期共享循环

**`arxiv:2610.41720v1`，2026-10-30 13:29 UTC，72h回溯首次纳入。** 该工作先按1/4、1/2、全深度复制层与Adam状态，最后重复调用4层中间块；阶段切换约发生在总token的9.7%、23.0%、44.3%，使最终参数、训练token、解析训练FLOPs同时对齐dense基线。它与另一篇同名的`2610.25004`不是同一研究。[固定全文§4–5、Appendix A](https://arxiv.org/html/2610.41720v1)。

OLMo2式370M/760M/1.3B在FineWeb-Edu按100 token/参数训练、4k序列、64个加速器die；1.3B的22任务均分22.99→27.52、八任务均分54.91→56.47，但闭卷TriviaQA、WebQuestions下降。370M在13个有效任务上聚合，不能与另外尺度的22任务均分直接比较。**等训练FLOPs不等于等推理成本**：末期执行20层而保留16组独立权重，推理FLOPs增加约20–22%。论文未给出多seed误差；本轮未核实公开训练实现，解析预算对齐不等于墙钟对齐。

**建议：**考虑复现训练FLOPs对齐及推理代价，价值在预算重分配而非免费循环。

### No Free Loop：预训练后的加循环，收益必须经得起等推理预算比较

**`arxiv:2610.42030v1`，2026-10-30 16:56 UTC，72h回溯。** 对Qwen3.5-0.8B/4B/9B分别比较同参数同训练配方，以及不高于循环成本的最佳无循环token预算。0.8B短SFT后等参数均分44.2→48.3，等FLOPs45.1→48.0；给两边都加budget forcing后仍为54.8→57.3。4B/9B的短SFT、长SFT、LoRA+RL大多无显著收益，ARD在后训练后加入甚至60.3→51.0（§4，Table2–4）。[固定全文](https://arxiv.org/html/2610.42030v1)。

从113M从头训到最多5.2B token续训的实验中，循环可超过等token的dense twin，却未超过等训练FLOPs的dense模型；H100上每token慢1.2–1.3×，吞吐约降两成。bootstrap区间条件于已跑seed，部分设置仅1–2 seed；结论限这些后加循环和预算，不能拿它否定UpLoop这类预训练预算重分配。未核实公开完整实现，未独立复现。

**建议：**深入阅读等推理预算基线，为后加循环实验设置公平对照。

### 塔吉克语CPT负结果：数据解码缺陷、格式遗忘与能力遗忘叠加

**`arxiv:2610.41576v1`，2026-10-30 11:06 UTC，72h回溯。** LFM2.5-2.6B-Base经1,485,319篇塔吉克文、3 epoch LoRA CPT，再SFT，3×H100分别用47.1h和2.11h。作者披露自建语料把Qwen2.5 token id用Qwen3.5 tokenizer解码，过滤后仍有3.6–5.8%文档可能乱码；这是实际训练工件缺陷，不能把所得退化全部归因于CPT本身（§3）。[固定全文](https://arxiv.org/html/2610.41576v1)。

145个需调用项目中，塔吉克严格full-call由20.69升至64.14；CPT中间checkpoint严格分为0，但宽松结构化解析仍有6.90，说明格式和skill都损失。最终英文MMLU从60.8降到52.5；29个不应调用项目的塔吉克正确拒绝率89.7→6.9，而基座换宽松解析后本来只有31.0。测试机器翻译未由母语者校验，多轮只给gold历史并评第二轮，不执行工具；缺少无CPT、无replay等对照，只有一个训练seed。模型已发布的事实不等于这套recipe已证明CPT必要性。

**建议：**深入阅读负结果与语料缺陷；先修解码和拒绝能力评估，再考虑采用该CPT配方。

### Less Loss, Fixed Weights：logit校正能回收部分损失，不能诊断遗忘的成因

**`arxiv:2610.42000v1`，2026-10-30 16:44 UTC，72h回溯。** 在14组1B–7B基座/衍生模型上，从同一DCLM shard各取≥1M token拟合/留出，用词表bias及温度校正；对基座也拟合后，按两者收益差衡量回收率。top4K bias最高9.8%，加温度后OLMo2-7B-Instruct为35.9%；Math CPT模型仍仅约1.2–1.3%（§2–3，Table1）。[固定全文](https://arxiv.org/html/2610.42000v1)。

模拟随机权重损伤、没有植入全局bias或温度偏移，也能回收29%，故“可校正”不证明“遗忘只是prior shift”。所有损失条件于屏蔽特殊/新增token后的词表；拟合未证明收敛最优，缺温度单独基线，仅一个web shard。官方代码`a5c1bdc`的`fit_bias_ipf`对应逐轮边际校正，`stats_from_docsums`对应对称回收率和按文档bootstrap；未运行模型复现。[固定代码](https://github.com/SeverinVisionary/prior-shift-diagnostic/tree/a5c1bdc45a69c496d37d4a482fff64f73b537046)。

<a id="posttraining"></a>
## 后训练数值、策略失配与优化器

**建议：**考虑复现便宜的logit诊断，但加温度单独基线与多域留出，避免把可校正当成因证明。

### Fair Cast：BF16权重同步的舍入死区，需要同时修导出与训练前向

**`arxiv:2610.41073v1`，2026-10-29 23:40 UTC，早于72h主回溯的pending补录。** FP32 master的小更新可能落在BF16相邻值的半格内；训练forward与rollout都用同一舍入副本时，两引擎log-prob比值接近1也看不见master偏离。Fair Cast把导出换成随机舍入或error feedback，并让loss在**同一served副本**上计算，以STE回传至master。不是仅改一行cast就可任意接入（§3–7.1）。[固定全文](https://arxiv.org/html/2610.41073v1)。

主配置为单H100共置vLLM/HF、Qwen3-1.7B、GRPO group8、24 prompt、150步、AdamW LR1e-6。zebra三组nearest/四组Fair Cast最终准确率0.33±0.10→0.57±0.16；Fair Cast仍有一run后期崩溃。**只改无偏导出、不对齐前向会崩溃**；list-functions原始Fair Cast反而0.523低于nearest0.668，显式EMA后两seed约0.708。随机导出1.7B权重约21.7ms对nearest4.3ms；error feedback另需6.9GB FP32残差，且改变2–15%权重会削弱稀疏delta同步压缩率。权重无偏不保证策略分布无偏，多个实验单seed、主模型≤4B，不是所有RL栈通用增益保证。官方`7c213b2`中`sr_cast_bf16`与`ste_params`/`served_forward`分别对应两半；未独立复现。[固定实现](https://github.com/andrewgcodes/faircast/blob/7c213b2f80834e00bfa7201b11421a3c1e462e51/faircast/cast.py)。

**建议：**优先考虑复现BF16低学习率RL，同时实现served-forward和导出两半，并计额外状态/同步成本。

### DLC：引擎数值误差与自诱导policy lag要分别校正

**`arxiv:2610.40945v1`，2026-10-29 20:23 UTC，旧pending补录。** 作者在sampler权重上额外跑trainer前向，把采样token误差拆成引擎numerics和当前权重lag；前者以score centering消漂移，后者以相对sampler权重的PPO比值裁剪，并在lag比值ESS低于半批时同步。只在旧sampler前缀上测KL会漏掉trainer自身新前缀上的崩溃（§4–7）。[固定全文](https://arxiv.org/html/2610.40945v1)。

主实验是Countdown、Qwen3-0.6B/1.7B、40步、三run、SGD LR1e-2；刻意把sampler固定在初始权重，或用FP8权重/activation/KV。1.7B stale设置中DLC51.9%，importance sampling+同步36.6%，on-policy57.6%。它没有免费恢复on-policy：需top128 log-prob及额外旧权重forward，实测每步113.2s，对on-policy50.4s、IS+同步52.7s；部分额外forward来自verl冗余KL路径（Appendix C）。8B与连续异步只各一run；短任务、SGD和人为极端staleness限制向真实长程Adam训练外推。本轮未核实公开完整实现。

**建议：**深入阅读numerics/lag拆分和ESS策略；先评额外forward墙钟，尚不直接作为异步训练默认。

### Muon-RT：已有EMA teacher可替代动量状态，收益要对齐无动量基线

**`arxiv:2610.41104v1`，2026-10-30 00:50 UTC，旧pending补录。** 将teacher–student位移去掉权重径向分量，谱归一化后与无动量Muon的正交化当前梯度混合；复用已有EMA轨迹，不再为hidden矩阵存动量。FSDP下位移本地算，但正交化与谱估计仍需额外gather（§3）。[固定全文](https://arxiv.org/html/2610.41104v1)。

Qwen2.5-7B的SDFT与OLMo3-7B的SDPO、四任务中，持久显存37.7GB，相对标准momentum Muon少27%、AdamW少42%；**momentum-free Muon/MuonClip本来同为37.7GB**，Muon-RT的价值是此内存档上的0.8–5.1点精度差，而非比它们再省27%。2×H100，优化器step2.3s，约整步2%；10T模型121TB是32-bit状态投影，不是实际训练。理论依赖未估计的PL/对齐常数；若原方法没有参数对齐的EMA teacher，专为替代动量加teacher不省内存。未核实公开训练实现，未独立复现。

<a id="research-systems"></a>
## 注意力、推理与训练系统

**建议：**已有EMA teacher时考虑复现，对齐无动量Muon；没有teacher则继续观察内存收益是否成立。

### LSA：10M上下文有端到端增益，树索引和原生训练的边界不同

**`arxiv:2610.42147v1`，2026-10-30 17:59 UTC；代码`906be195`于2026-11-02 20:29 UTC[初始发布](https://github.com/Prefrontal-AI/latent-sparse-attention/commit/906be195c1052aaa74f95852ad1917ce1926ba64)，是本轮主窗内新增工件。** 在64维index空间pool key，按语义聚成bucket，再Flat或分层beam检索，展开远程key并保留sink/window。D-LSA只蒸馏indexer，基座冻结；N-LSA从NTP原生训练、不加辅助loss。**仅Tree固定beam索引达到O(T log T)，Flat仍二次**（§3）。[固定全文](https://arxiv.org/html/2610.42147v1)。

H200/B300、SGLang、单请求256输出token、三请求中位数：H200上10M时，Qwen3.8-27B、TP8+DCP8的D-LSA Flat相对dense，TTFT/TPOT加速比为9.21×/2.87×（基线时延/LSA时延）；GLM5.3-Flash、CP8相对原模型sparse attention基线的加速比为2.24×/2.10×，不是对dense。单层72.4×不是整模型提速；512k的部分配置甚至变慢。树索引在GLM10M的RULER为55.8，对Flat66.8；100M曲线是外推。0.5B原生混合模型经100B 4k预训+12B 128k续训，N-LSA decay墙钟快1.24×且NLL同1.085，但单层峰值6.8GiB高于dense5.8GiB（§4–5）。

公开代码已包含聚类、分层检索、蒸馏/原生训练和SGLang适配；`ward.py::ward_merge`对应受bucket cap约束的互为最小代价合并，`hierarchy.py::beam_pick`对应逐层保留候选。README要求配套SGLang fork及关闭prefill CUDA Graph，decode仍可capture；尚未证明上游SGLang可直接运行。KV仍线性增长，CP还复制bucket cache；未独立跑10M复现。[固定代码](https://github.com/Prefrontal-AI/latent-sparse-attention/tree/906be195c1052aaa74f95852ad1917ce1926ba64)。

**建议：**考虑在固定fork上复现长上下文，分别验Flat/Tree精度与prefill/decode收益；上游即用尚未证明。

### SparseGDN：跳过约71% recurrent事件，1.94×只计GDN prefill core

**`arxiv:2610.41420v1`，2026-10-30 08:27 UTC，72h回溯。** 将64-token chunk内完整Delta solve近似为16-token块的对角与相邻块，再按相邻value-head pair选择chunk事件；跳过更新仍保留衰减、为每个token重建输出。它有真实近似语义，并非仅跳过冗余精确计算（§2.2–2.7）。[固定全文](https://arxiv.org/html/2610.41420v1)。

Qwen3.5-4B/9B、BF16 activation+FP32状态，13项RULER×6长度4k–128k、每模型7800例，保留约29%事件时均分96.142→96.159和96.617→96.502；9B在8k跌0.692点。包含selector的24层GDN core提速1.50–1.94×，**排除QKV/O projection、softmax层、MLP及decode**。整模型128k prefill allocated peak从53.116→26.894GB（4B）、64.066→37.844GB（9B）；256k仅证明sparse能完成，native OOM没有可测峰值对照。手工selector可能漏罕见重要事件，尚无开放生成负载；设置未明确GPU具体型号，本轮未核实公开实现，未独立复现。

**建议：**深入阅读稀疏近似机制，继续观察公开实现与完整硬件/生成评测后再采用。

### HiState：混合模型的KV前缀命中，还要有可恢复的历史递归状态

**`arxiv:2610.41508v1`，2026-10-30 09:59 UTC，72h回溯。** 一次复用需整段KV，但只需目标位置的一个recurrent checkpoint，因而优先把KV放快存、历史checkpoint放SSD；按预期重算与存储压力调间隔，并共同淘汰中间checkpoint和冷session末端bundle（§3）。[固定全文](https://arxiv.org/html/2610.41508v1)。

Qwen3.5-9B单H10080GB、约45GiB GPU cache、96GiB pinned DRAM、150GiB disk，最多24 live sessions；Weka3784请求/125会话与TraceLab4791请求/68会话回放。对vLLM-Marconi，p95/99最高7.40×/5.73×；在复用敏感子集里，对作者扩展的vLLM-Disk均值2.66→1.29s和1.77→0.96s。后者是筛选子集，不能当所有请求均值；p50在Weka还有小退化。baseline GPU与所用CPU/SSD容量匹配，但SGLang默认静态分区也是差异来源；正文未提供独立多模型或多GPU验证，本轮未核实公开HiState实现。

**建议：**考虑复现混合模型多轮state分层，先匹配CPU/SSD预算并扩到第二模型。

### MemX：把内存均衡纳入搜索目标，实测吞吐与预测指标应分开读

**`arxiv:2610.41525v1`，2026-10-30 10:20 UTC，72h回溯。** 以时间、峰值显存和跨卡方差的加权乘积搜索PP/TP/DP及分阶段重计算，少量profile+插值训练双输出XGBoost。20卡由4×A10080GB及两个未披露厂商的8×32GB卡组成，Megatron-LM0.8；31,240配置26.5s是**搜索时间**，12.17%是时间预测MAPE（§3–4）。[固定全文](https://arxiv.org/html/2610.41525v1)。

真正小集群部署为4×V100+2×T4、6层Llama2、667 runs/226配置：按时间目标，MemX、AMP、FaSOP同为9211ms；按内存均衡目标，MemX选10066ms，慢9.3%换更均衡内存。其可选维度还包括sequence length、attention heads/query groups，故并非全部方法在完全相同固定架构上竞争。40/60卡是扩展实测stage的模拟，不覆盖新增通信效应；60卡MAPE表为16.19%，邻近正文写14.91%，存在内文冲突。未核实公开实现，不把预测排序或“17天profile外推”写成实训加速。

**建议：**深入阅读多目标搜索取舍；没有吞吐优势的设置应按显存均衡价值判断。

### ARK：协调KV大流路由，改善的是拥塞场景里的转移和TTFT

**`arxiv:2610.41160v1`，2026-10-30 02:35 UTC，72h回溯；论文标SIGCOMM2026，不能据arXiv日期断言会议首发。** 预先枚举源端口到ECMP路径的映射，host广播RESERVE/RELEASE避免长KV流哈希碰撞；满路径时退回随机ECMP，状态最终一致而非强一致（§3）。[固定全文](https://arxiv.org/html/2610.41160v1)。

证据来自Vidur计算预测+NS-3网络仿真：两台各8GPU、四spine、100Gbps、2:1超售、Llama3-8B H100 profile、1000个Poisson到达请求，KV均值1.6GB。QPS25的mean/P95 FCT降27.3%/34.0%，全链路mean TTFT约降13%；低负载仅小改善，接近饱和收益缩小，TPOT不受影响。Intel i9上的controller lookup微测不等于真实交换网络部署；依赖可控hash/源端口与peer状态同步，本轮未核实公开ARK实现。

**建议：**继续观察真实网络实现与ECMP可控性验证，仿真收益尚不足以直接改生产网络。

### 量化agent能耗：固定调用省电，完整任务可能更贵

**`arxiv:2610.41672v1`，2026-10-30 12:40 UTC，72h回溯。** 同时计算CPU package+GPU board到server drain的总能耗及独立验证成功数，包含失败尝试；再cross replay相同token调用，区分kernel省电与agent多走弯路（§3–5）。[固定全文](https://arxiv.org/html/2610.41672v1)。

Devstral24B在A10080GB、vLLM0.10.2、4 admitted agents、16任务×6 paired重复：W4A16执行固定调用比W8A8省约23–25%，live却能耗多23.9%、成功少16.7%，每完成任务贵48.6%。Qwen3.5-9B在A10040GB、11 paired重复同为约49%更贵。40GB、16个Devstral agent存在cache竞争时，限制W4 cache会令每成功任务能耗上升34.4%，显示低bit释放内存也可反转结论。不是位宽单因素实验：量化recipe、kernel成套变化；仅16任务、replay有单次recording，CPU/GPU不是整机墙插能耗，加载/独立验证也在边界外。cache cap四对的sign-flip p=0.25，须保留不确定性。

<a id="evaluation"></a>
## 多模态、评估与监控负结果

**建议：**优先深入阅读并复现任务级能耗，包含失败和追加工作，避免只量单调用省电。

### MLLM视觉比较：认识与定位目标结构，可能是简单长度判断的门槛

**`arxiv:2610.41841v1`，2026-10-30 14:56 UTC，72h回溯。** 20,000张合成蚂蚁图随机配10,000对，平衡左/相同/右，用部件独立缩放隔离视觉比较。Qwen3.5/3.8、Gemma4、InternVL3.5、GLM4.1在腿/刚毛上较好，mandible却常接近33%随机；thinking并不一致改善（§3）。[固定全文](https://arxiv.org/html/2610.41841v1)。

Qwen3.5 mandible由33.2%经segmentation+Object提示升至76.8%；Qwen3.8由38.6→81.4。单纯给标准参考图无益，highlight后的识别错误99%把mandible叫antennae；中层image-text相似性支持语义grounding错误，但不是证明训练过程因果。思考模式只跑1000对；人为提供分割已几乎解决原本要自动处理的问题，不能当可部署解法，也不能外推真实显微病理图。作者提供官方代码链接，本轮完成论文核读、未冻结审查该repo代码或独立复现。

**建议：**深入阅读结构grounding诊断；真实视觉域仍需独立数据与自动定位验证。

### MTG-AD旧pending：要同时评内容、TTS占时与真实听众，分母已部分澄清

**`arxiv:2610.39955v1`，2026-10-29 12:51 UTC，原先只列卡片的pending现补全文。** 433个约35.9h英文YouTube视频，332 validation/101 test，盲人与明眼专业写作者共同标注；pipeline把AD转成语音，再衡量与原声重叠及静默覆盖，避免仅文本看起来正确。validation上的Guideline-only重叠59.19s、加ASR21.34s、再给gap/词数预算12.77s；Human为7.51s（§3–5）。[固定全文](https://arxiv.org/html/2610.39955v1)。

**101是test池，非每模型实际分母。** Appendix B/Table6为Qwen3.5-Omni-Flash57、Plus99；HF卡片有Flash82条response，再排除25个`include_in_scoring=false`，恰为57。卡片“其余101条”与Plus论文99仍需逐metric空值核验，不据此强排统一leaderboard。不同代模型还换了Gemini judge，不能将Likert跳升全归生成模型；reference-based Accurate与BLV感知准确率ρ=-0.264且不显著。7名BLV共28次video评价，播放时遇原声即截断AD，所以他们听到的是截断而非重叠，不能用该人评宣称重叠问题消失。官方`a97330f`的`compute_tts_overlap_metrics`用0.1秒离散重叠，`metric_1_2_narrates_onscreen_text`用ASR+AD词级recall；原始视频链接、数据和代码公开，未独立复现。[数据卡片](https://huggingface.co/datasets/google/mind-the-gap-inline-ad)，[固定代码](https://github.com/google/mtg-ad/tree/a97330f8ba3781aa2b7eb4311be98e71477731ed)。

**建议：**深入阅读音频占时与听众评估，先补Plus逐metric分母再用榜单选模型。

### LifeVal：新增日常agent评估，但示例百分比的口径尚不完整

**OpenAI官方日期2026-11-02，日期精度事件。** 从美国/国际时间利用调查出发，再调查3995名具人口代表性的受访者，选出照护、教育、家务、财务、购物及政务等需求，强调完成实际工作流。官方展示找幼儿园93.1%、防洪志愿者98.0%等个例，但本轮可读正文未给任务总数、模型准确版本、重复次数、评分rubric或整体聚合定义。应视为新评估方向与个例展示，不能把90.4–100%写成LifeVal总体成功率，也不能据此跨模型比较；需继续追完整benchmark和可复现实验材料。[官方正文](https://openai.com/index/lifeval/)。

**建议：**继续观察完整任务集、rubric与原始结果发布；当前适合了解评估方向，不能复现总体分数。

### Anthropic旧文补录：可优化的monitor分数，不保证找到最有解释力的安全结论

**官方仅标October2026，首发日未确定；11月2日二手报道不当新论文日期。** 研究让Claude agent提交≤5000条微调样本，给其精确monitor反复试，筛除flag比例>10%或总数>100的submission，并用cheat detector与能力门禁把关。经过迭代，fiction/roleplay等表面合理样本仍可使专门微调的研究模型在四类可见失准测试上恶化；另留两类测试，已标记样本删除重训后仍观察现象。[官方全文Setup/Results](https://alignment.anthropic.com/2026/automated-em-search/)。

这是刻意有利于红队、可访问确切monitor的受控stress test，不能推成部署模型事故或自然语料污染发生率；最高分优化也未必带来新的机制发现。官方未披露完整monitor措辞和模型版本、完整训练预算，外部难以按文复现；少数权重外传行为在最终重现run未再出现。应继续关注训练数据monitor的跨上下文泛化与held-out审查，而非引用单例作普遍风险估计。

## 论文域实际覆盖与待续查

本轮7类arXiv `new`页已滚到11月2日，共746个稳定作品，覆盖new/cross/replacement；HF10月30日39篇、11月2日37篇，周末空列表，11月3日API400。主窗/回溯/updated API初次均429；后续回溯分两页取齐1018/1018条metadata，时间从2026-10-30 01:43:37到17:59:59 UTC。主窗100条重试返回200且0 entry；updated重试的返回query被改写为submittedDate，仍不能推“无新论文/修订”。15篇arXiv在固定v1核读方法、条件、结果与局限；另4篇已保存全文但只完成结构/候选筛选，留队。加两篇官方研究文章；LSA当前窗代码release另记稳定工件事件。本文全部结果均为作者报告，本轮无GPU复现。

TMLR、ACL2026 EMNLP、PMLR目录已取回；OpenReview API及Scion、Thinking Is Not Telling全文仍403/挑战，正式卷的日期历史与逐篇语义覆盖未完成。新旧候选、replacement语义diff、MTG-AD Plus/逐metric有效分母、SparseGDN/HiState/MemX公开实现、SCARL配方与PDScope predictor等持续保留。列表完整只代表发现面，不前移全局完整覆盖游标；旧pending未因本轮选读自动关闭。

**建议：**深入阅读monitor可被优化的边界，为数据审查设计held-out测试；不外推部署事故概率。


<a id="core"></a>
## 核心开源：发行、训练与推理增量

主窗 `2026-11-02T01:42:47Z` → `2026-11-03T00:58:48Z`；回溯自 `2026-10-30T01:42:47Z`。以下代码增量均以各条合入 SHA 冻结，**main 合入、release 分支回补、候选发行与集成分支分别标记**。本轮做源码/最终 diff 核查，没有在本机复跑 GPU/NPU 或分布式训练；作者实测均保留其基线与边界。

### SGLang v0.5.23 正式发行，但其 tag 快照早于本日 main
11-02 22:14:10Z发布，annotated tag解引用至297e54fd105e974cdf401e5b4d20253a22c16f90（tagger 10-30 00:13:59Z）。发行说明集中已有模型、graph、MoE、LoRA、HiCache与平台工作；本日下列11-02主干PR不能据发布日期自动归入该包。

**边界：**说明中的最高加速数来自不同PR/硬件/工作负载，本轮没有重新审计全部1007 PR或逐镜像验证。 **建议：**依赖正式包时先按tag确认具体修复是否包含；需要本日main修复则锁对应commit并构建。
来源：[sgl-project/sglang v0.5.23](https://github.com/sgl-project/sglang/releases/tag/v0.5.23)；tag commit `297e54fd105e974cdf401e5b4d20253a22c16f90`。

### vLLM v0.33.0rc3 为候选标签，发行页元数据与正文有歧义
11-02 10:10:05Z发布，tag直接指向4ba787ec9959e566d3b587a796dff07673324da5，target为releases/v0.33.0。虽然GitHub prerelease=false，tag仍带rc3；正文使用0.33.0包/镜像示例并列Torch2.14/CUDA13等变更。

**边界：**正文仍有验证发行URLs/tags的TODO；不能据这些命令宣称GA镜像/所有wheel已实开验证，也不能把本日main全部增量算进rc3。 **建议：**将其作为候选验证，固定tag SHA和平台artifact；GA判断另查无rc正式tag及实际wheel/image。
来源：[vllm-project/vllm v0.33.0rc3](https://github.com/vllm-project/vllm/releases/tag/v0.33.0rc3)；tag commit `4ba787ec9959e566d3b587a796dff07673324da5`。

### ScheduleNode 为所有跨 stream 输入登记 storage 生命周期
保留输入的 backward 异步入队后，最后引用可能先消失；现在 make_viewless/detach 后无论 free_input 是否开启，都对原输入 record_stream(consumer)，防止 producer allocator 提前覆盖。

**证据与边界：**没有新增同步。作者在 H100 八 rank 跑完整 a2a-overlap/PP 文件，并有移除修复即失败的双 stream、NaN 分配压力回归；本轮只核最终 diff，未复跑 GPU。 **建议：**定制 combined 1F1B 节点同时检查 retained/free_input 两条路径，不能只延长 Python 引用至 forward 返回。
源码路线：`megatron/core/pipeline_parallel/utils.py::ScheduleNode._forward；test_schedule_node_autograd_boundary.py::test_input_storage_survives_cross_stream_backward`。
来源：[NVIDIA/Megatron-LM #8823](https://github.com/NVIDIA/Megatron-LM/pull/8823)（main，2026-11-03T00:02:29Z，`a0ef64615cc3d2574ffbaac70f324ee365f5ed34`）。

### dp_balanced THD packing 接入 TE attention 图捕获
用有界 packed-sequence capacity 固定四套逻辑/物理 Q/KV cumulative offsets 的形状，在 replay 更新文档边界，零真实 token 的 padding 不进入 loss/routing；变动 microbatch 数超过已捕获容量时提前拒绝。

**证据与边界：**需 TE≥2.18、MBS1、固定源 batch、静态 CP；当前仅 attention capture，PP 下拒绝 EP-overlap、delay-wgrad、paged stash。作者四 GB200 rank 的 PP2/CP2及VPP2验证不是完整64K Pro、FP8、PP4/CP16性能认证。 **建议：**显式设置 cuda_graph_thd_max_packed_sequences 与 max-aligned padding，回归4→2→4 microbatch及四套offset，勿只固定token buffer。
源码路线：`transformer_config.py::cuda_graph_thd_max_packed_sequences；TransformerLayer._reconstruct_packed_graph_inputs/_te_cuda_graph_replay；test_sequence_packing.py；test_cuda_graphs.py`。
来源：[NVIDIA/Megatron-LM #8361](https://github.com/NVIDIA/Megatron-LM/pull/8361)（main，2026-11-02T18:46:59Z，`631e146322af891b0c79b8d5a53bc579a69f01f3`）。

### 部分 activation offload 修正 warmup 配额与 backward 预取节拍
原调度只保留已卸载组，却每个 backward 组都推进，导致GPU常驻组也提前拉回更早激活。现在把无需copy的组也保留为节拍；warmup按模块名分别统计k次组出现（含空/不可卸载组），用ceil(k×fraction)决定新增配额，避免过量CPU buffer长期留池。

**证据与边界：**作者17×32MiB合成链、fraction=0.5：backward峰值448→416MiB；等待使用的reload从256→32MiB并不等价于总峰值省224MiB。warmup host544→288MiB也仅该链；35项/rank GPU测试，8项既有flaky未运行。 **建议：**分别观察pinned-host池与backward实际峰值；回归GPU常驻组夹在offload组之间、CPU buffer复用和最后一组margin。
源码路线：`pipeline_parallel/fine_grained_activation_offload.py::OffloadGroup/ActivationOffload；test_fine_grained_activation_offloading.py`。
来源：[NVIDIA/Megatron-LM #7575](https://github.com/NVIDIA/Megatron-LM/pull/7575)（main，2026-11-02T13:16:46Z，`90f4d5279b951a408d7bf605a24776c248357c03`）。

### MFSDP v2 以单一带 phase 的事件序列校验预取
forward/backward两套模块顺序合并为弱引用ExecutionEvent(module,phase)与单cursor；保留共享模块每次出现，逐事件验证实际调用，预取可跨phase向后看，序列末尾回绕。

**证据与边界：**registration模式现在要求真实顺序匹配；recorded仍依赖普通hooks并在首个post_backward结束记录，combined1F1B、完整PP/VPP schedule-window/eval集成另案。作者双A6000、93项/rank；五项需四rank跳过。 **建议：**调用顺序不同于注册顺序时选recorded，并回归共享MTP、重计算与跨rank一致性；不要把本次简化理解为动态控制流支持。
源码路线：`experimental/execution_order.py::ExecutionEvent/ExecutionOrder；context.py::FsdpContext.finalize；module.py`。
来源：[NVIDIA/Megatron-LM #8485](https://github.com/NVIDIA/Megatron-LM/pull/8485)（main，2026-11-02T09:59:53Z，`2fcef8359b977750f1868dbc5766baa9c26b2fb0`）。

### DSA indexer 获得独立梯度范数与旧 checkpoint 分组兼容
最终diff以is_dsa_indexer拆分参数组；dsa_indexer_clip_grad非None时分别计算indexer/非indexer范数并按各自阈值裁剪，默认保留全局范数。旧checkpoint缺失该标记按False匹配，并用构造时保存的pre_split_param_index恢复原参数编号。

**证据与边界：**PR标题称per-group optimizer clocks，不能据此宣称新时钟机制；最终代码仍写shared common_step，不支持各参数独立step。旧组参数索引无法重建或不连续时显式失败，不能承诺任意自定义optimizer checkpoint迁移。 **建议：**启用独立裁剪前核对indexer命名与阈值；用历史checkpoint核对每个参数的momentum而不只是load不报错。
源码路线：`optimizer/__init__.py::_get_param_groups/_tag_pre_split_param_index；optimizer.py::clip_grad_norm_separate_dsa_indexer/find_saved_param_group；distrib_optimizer.py::load_state_dict`。
来源：[NVIDIA/Megatron-LM #7158](https://github.com/NVIDIA/Megatron-LM/pull/7158)（main，2026-11-02T03:06:08Z，`e0a4bfed7f1e233b857187ceff0104645858d7d1`）。

### MFSDP v2 EP checkpoint 支持回补到 core_r0.20.0
将main的#8230及必需测试fixture后续#8326回补至release分支；optimizer按device mesh/FQN组织EP状态，恢复main weights后刷新BF16/量化compute storage，避免训练可跑却首次save报未实现。

**证据与边界：**这是core_r0.20.0分支合入，非新的core_v0.20.0发行。双RTX6000 Ada、TE2.17.1兼容路径验证61通过/37跳过；TE≥2.18、EP4、expert-DP>1及端到端训练未验证。 **建议：**在release分支固定本次SHA，实测save→restart→首forward及optimizer状态；不要只更新发行标签。
源码路线：`fully_sharded_optimizer.py::checkpoint mesh/key handling；mcore_fsdp_adapter.py::FullyShardedDataParallelV2.load_state_dict；experimental/checkpoint.py::load_checkpoint`。
来源：[NVIDIA/Megatron-LM #8465](https://github.com/NVIDIA/Megatron-LM/pull/8465)（core_r0.20.0，2026-11-02T23:22:20Z，`da8cb5e93a96811da073e301684580838a8186cc`）。

### 补录：训练 relaunch 默认续写同一 wandb run
从实验名派生稳定run id并替换URL不允许字符，resume=allow兼容首启/重启；新增wandb_run_id覆盖，slurm_job_name更新允许值变化。未命名实验仍由wandb分配id。

**证据与边界：**10-31已合入、此前队列未详审。本轮没有运行wandb集成；重放checkpoint前已记录step会被wandb丢弃，不能称日志随checkpoint回滚。不同实验名清洗后还可能碰撞。 **建议：**需要独立试验或改名后续写时显式分配run id；重启边界用checkpoint与训练日志校正，勿将合并图当成重放数据真值。
源码路线：`training/global_vars.py::_wandb_run_id/_set_wandb_writer；training/config/training_config.py::LoggerConfig.wandb_run_id；training.py::pretrain`。
来源：[NVIDIA/Megatron-LM #8612](https://github.com/NVIDIA/Megatron-LM/pull/8612)（main，2026-10-31T21:09:14Z，`aa7a2f34196f7686e0d6b67af203c439810ce335`；补录）。

### 原生 NVFP4 W4A4 贯通 Megatron 训练与 vLLM rollout/refit
TE仅对指定routed-expert MLP做per-token NVFP4计算，策略持久权重仍BF16，optimizer保留完整state；训练端每次更新导出NVFP4并共享gate/up global scale，rollout用FlashInfer TRT-LLM per-token activation。refit原位恢复权重、派生scale与kernel引用以保住CUDA graph地址。Qwen示例首2/末4层MLP及attention保留BF16。

**证据与边界：**限Blackwell与经过guard的MoE合同：PP/VPP、DCP/PCP、MTP/spec、DP>1/EPLB等不支持，naive colocated IPC与FULL_DECODE_ONLY。作者32×GB200、700-step DAPO报告在匹配长输出时rollout约1.3×、整step约1.2×，但TE训练本身仍慢于BF16；SGLang未做E2E。验证栈TE2.20.2/MCore0.19.2/vLLM0.32.0。 **建议：**同时锁trainer/rollout量化recipe、BF16 carve-out及依赖版本；每次refit检查地址/scale引用和train-rollout KL，勿当全模型FP4存储或通用提速。
源码路线：`real_nvfp4/config.py::validate_real_nvfp4_*；real_nvfp4/megatron.py::apply_real_nvfp4_overrides/attest_real_nvfp4_runtime；vllm_nvfp4_per_token_utils.py；vllm_fp4_utils.py::_restore_nvfp4_derived_params_in_place`。
来源：[verl-project/verl #7971](https://github.com/verl-project/verl/pull/7971)（main，2026-11-02T07:58:02Z，`6879e1ab07b9cfd0b4b27c31ca4e7afcad1466b3`）。

### QAT 与 CUDA 参数导出共享 TileLang 量化数值配方
新增FP8_BLOCK二维块、MXFP8 1×32、MXFP4 1×32 kernels；QAT融合QDQ并用STE backward，导出调用同一路径但返回packed量化值/scale，减少训练fake quant与rollout导出的舍入差异。ModelOpt升级0.47.0。

**证据与边界：**MXFP4函数要求CUDA、BF16/FP16/FP32输入、最后一维非零且32倍数；NVFP4走独立Triton路线，CPU保留fallback。作者51项GPU数值/导出测试与参考bitwise比较不等价于完整训练质量或速度认证。 **建议：**比较的是packed bytes、E8M0 scales和QDQ，不只比较反量化误差；升级ModelOpt后重验padding、ignore及QAT/export一致性。
源码路线：`qat/fp4.py::fp4_quant/_mxfp4_scale；qat/fp8.py::fp8_quant；qat/modelopt.py::_Fp8FakeQuant/_Mxfp4FakeQuant；qat/compressed_tensors.py`。
来源：[verl-project/verl #8498](https://github.com/verl-project/verl/pull/8498)（main，2026-11-02T11:31:28Z，`06a3a494daa28c912d764ee928bc96e643ec3d56`）。

### 量化 refit 后重置没有校准 recipe 的 KV scales
compressed-tensors无kv_cache_scheme时dummy load仍会随机化attention scale，而refit不发送它们；现在每次quantized post-load后调用vLLM默认scale helper，在no_grad下将q/k/v/prob scale归1，修复FP8_BLOCK rollout叠加FP8 KV产生乱码。

**证据与边界：**有显式kv_cache_scheme的校准配置保持不动，verl仍未导出其KV scales，不能宣称已修复该路径。作者vLLM0.29/0.32、H20复现与CPU stub回归；SGLang未查。归1属于未校准FP8 KV，不等同BF16精度。 **建议：**对每轮refit检查所有attention层scale，先测train-rollout KL再上长训练；有校准scheme时另验实际scale来源。
源码路线：`utils/vllm/vllm_quant_utils.py::process_quanted_weights_after_loading/reset_uncalibrated_kv_cache_scales；test_vllm_kv_cache_scale_reset_on_cpu.py`。
来源：[verl-project/verl #8483](https://github.com/verl-project/verl/pull/8483)（main，2026-11-02T01:56:23Z，`fc726e33ceb54ad875d4537880e786de2a32129d`）。

### 无 critic 的 GRPO 不再从 TransferQueue 请求 values
advantage与metrics字段表仅在use_critic时加入values，兼容新版TransferQueue对缺字段的严格拒绝。main #8461与release/v0.9.1 #8457分别合入；后者另修Qwen3.5 FSDPTurbo示例dispatcher为all2all并单独设置fused ops。

**证据与边界：**release分支合入不代表v0.9.1重新发行。本次main只改字段选择，不能外推为所有TransferQueue API兼容或critic计算逻辑改变；没有新增运行测试结果可独立核验。 **建议：**无critic配方升级TransferQueue时回归advantage与metrics两个阶段，并区分main与release脚本差异。
源码路线：`trainer/ppo/v1/trainer_base.py::_compute_advantage/_compute_metrics；run_qwen3_5_35b_fsdp_turbo.sh`。
来源：[verl-project/verl #8461](https://github.com/verl-project/verl/pull/8461)（main，2026-11-02T11:38:17Z，`3fdf2b548882a8aeccf0842c2150d9c9f1790a55`）；[verl-project/verl #8457](https://github.com/verl-project/verl/pull/8457)（release/v0.9.1，2026-11-02T11:38:33Z，`26613a684230aecb1daa00243dd032b3cdf13e1c`）。

### Ascend 950 环境升级，但 MXFP8 示例明确依赖旧基线
Docker/安装指引把Megatron升至core_v0.18.0，加入MegatronAdaptor、TransformerEngineNPU、MindSpeed-Ops/Bridge及Megatron-Bridge0.5.0，并增加Qwen3-30B-A3B MXFP8 GRPO示例。

**证据与边界：**最终示例注释明确：#8287移除旧Ascend MXFP8 reload支持，当前main首次weight sync形状不匹配；验证基线为18eb23dde5ec84edf51a8b780a74a116524aa5cb，必须暂用它。部分gitcode依赖默认分支未锁，不能写成新main已开箱支持MXFP8 GRPO。 **建议：**先固定示例声明的可用commit与完整依赖镜像，修复并验证reload后再迁回main；脚本文件实际无.sh后缀而文档链接带.sh也需核对。
源码路线：`docker/Dockerfile*；docs/ascend_tutorial/zh/get_start/install_guidance_950.rst；examples/ascend_extras/low_precision/run_qwen3_30b_mxfp8_megatron`。
来源：[verl-project/verl #8404](https://github.com/verl-project/verl/pull/8404)（main，2026-11-02T01:52:44Z，`6dcc6e4d10ba3b10cd7f659ba89189acabcb55be`）。

### GDN generator 减少 gate launches、重复 metadata 与输出copy
packed decay gate从reference改用Attention Gym fused且fastmath=False；同spec的后续KV组只gather自己的state slots并复制到graph-static buffers。runner告知是否重放packed FULL graph，避免one-token批错用旧offset；chunk直接返回output，decode只清零未写padding。

**证据与边界：**三项各自作者GB300测试支持指定输出/state逐位对照；不能把逐层微秒收益相加当整服务收益。metadata端到端4.5%仅Qwen3.5-9B TP1/256req，KDA与27B未同样实测；FULL/padding、empty/profile路径仍有必要zero。 **建议：**复核graph dispatch与metadata path一致，连续poison padding并变active count；不能机械删除所有zero或复制。
源码路线：`rl/generator/vllm/gdn.py::_forward/_forward_gdn；linear_attention.py::update_block_table；worker.py::_determine_batch_execution_and_padding；test_gdn_graph_metadata.py`。
来源：[pytorch/torchtitan #5668](https://github.com/pytorch/torchtitan/pull/5668)（main，2026-11-02T05:33:23Z，`eba2b1080d2c8bf85912f2f4b6a74a54ffa9e974`）；[pytorch/torchtitan #5669](https://github.com/pytorch/torchtitan/pull/5669)（main，2026-11-02T05:34:05Z，`282e290951d6e893004d67cd7acd91853e5d00de`）；[pytorch/torchtitan #5670](https://github.com/pytorch/torchtitan/pull/5670)（main，2026-11-02T06:44:39Z，`83abb2c039ebeffdef27907ea7e33b6174061651`）。

### GDN mixed step 把 decode 行分给 recurrent decode kernel
原每个decode请求都作为单token sequence进入chunk；现在把decode前缀在chunk端合为null-slot sequence，再加decode kernel输出，prefill仍走chunk。capture为任意decode count保留静态容量，batch-invariant分支不改。

**证据与边界：**mixed输出/state不再与旧chunk路径bitwise，作者相对误差有界；GB300 Qwen3.5-27B TP1/256req整batch约4.7%更快来自较早同kernel revision。FULL少/无decode仍发射null programs，纯2K prefill反而变慢，不能称全面提速。 **建议：**分开测prefill-heavy和decode-heavy mixed负载，回归DP dummy、null slot/state和自适应batch，别只测255decode+1chunk最优点。
源码路线：`gdn.py::VLLMInnerGDN._forward；linear_attention.py::_stage_decode_split；test_packed_decode_routing_matches_recurrence`。
来源：[pytorch/torchtitan #5671](https://github.com/pytorch/torchtitan/pull/5671)（main，2026-11-02T07:55:02Z，`23c4cf64ee307c88409c12262a62988ecc3634e2`）。

### GDN 六套输入投影融合为按 key head 分组的一套 in_proj
trainer/generator共享key-head分组的q/k/v/z/a/b行布局，一次GEMM后拆分；任意TP都可按dim0分片，HF适配器负责逻辑投影与native布局转换，generator直接接收mixed_qkv。

**证据与边界：**旧native DCP每层六个键无法直接载入新in_proj；需旧commit导HF再新commit从HF载入。random-init RNG序列改变；Ulysses CP多一次复制未测。作者27B trainer有小收益，35B-A3B layer反而慢1.3–1.6%，不能只报decode收益。第7/7编译后续#5731仍open。 **建议：**训练迁移先保留旧环境HF导出，按相同HF权重对照TP/CP、optimizer与logprob；不要仅重命名DCP键。
源码路线：`models/common/attention/gdn.py::GDN；models/base/state_dict_adapter.py::_split_gdn_in_proj/_merge_gdn_in_proj；qwen3_5/state_dict_adapter.py；rl/generator/vllm/gdn.py`。
来源：[pytorch/torchtitan #5689](https://github.com/pytorch/torchtitan/pull/5689)（main，2026-11-02T15:21:01Z，`b832ae41936a626ac0c8dc9ab1ae8dd556fd428b`）。

### GDN generator 的 A_log/dt_bias 持久保持 FP32
meta构造时转FP32并freeze，通过FSDP param_dtype_override_fn保留每参数dtype；接收端仍按trainer发送dtype建pinned buffer，load_state_dict.copy_时一次转换，删除每层每步.float。

**证据与边界：**trainer不转FP32，以保住既有训练/生成舍入；未做多host RDMA同步实测，FSDP degree>1仅CPU probe。非BF16 generator的其他参数也开始保留model_dtype，是实际行为变化；作者层时延不等于测得整step百分比。 **建议：**同时核对transport byte dtype与FSDP unshard dtype；在TP×FSDP和多机refit上另测，不能把FP32 buffer替代DTensor parameter。
源码路线：`model_wrapper.py::VLLMModelWrapper.__init__；models/common/decoder.py::_fsdp_param_dtype_override_fn；gdn.py；test_fsdp.py`。
来源：[pytorch/torchtitan #5691](https://github.com/pytorch/torchtitan/pull/5691)（main，2026-11-02T18:16:13Z，`c3093f782aec620fdc6887887fe366917abbc057`）。

### 显式 native checkpoint load 也先拒绝未完成保存
BaseCheckpointManager.load在进入任意native后端前统一调用_is_resumable_checkpoint，补齐显式load_step和initial_load_path漏检，错误提示包含路径与HF加载选项。

**证据与边界：**HF模式保留不同合法性规则；这只能验证完成metadata，不是checkpoint全内容校验或自动修复。作者171项相关CPU测试，未复跑真实分布式DCP。 **建议：**故障恢复同时测缺.metadata、HF-only目录与有效native目录，避免把backend内部storage异常当权限或数据损坏。
源码路线：`components/checkpointer/base.py::BaseCheckpointManager.load；test_checkpoint.py；test_torch_checkpointing.py`。
来源：[pytorch/torchtitan #5144](https://github.com/pytorch/torchtitan/pull/5144)（main，2026-11-02T23:32:41Z，`cec07e63cab1616582f0b7fb0904f74133531ef6`）。

### Muse Glimmer/LLama 的 RoPE 行置换按 tensor 自身形状处理
q/k行重排从传入完整projection维度改为tensor.rows/cols，使rank宽LoRA B因子也按同一行顺序导出，修复k路径view形状崩溃。

**证据与边界：**不改变完整权重数值。作者Muse-Glimmer-30B TP1/TP2 HF往返逐位一致与CPU因子测试；不等于任意LoRA组合或端到端RL均已验证。 **建议：**服务LoRA导出时单独回归q/k/v分片与rank宽矩阵，不要用整权重往返替代因子测试。
源码路线：`models/muse_glimmer/state_dict_adapter.py::_permute/_reverse_permute；models/llama3/state_dict_adapter.py；TestMuseGlimmerStateDictAdapterRowPermute`。
来源：[pytorch/torchtitan #5726](https://github.com/pytorch/torchtitan/pull/5726)（main，2026-11-03T00:57:13Z，`4759738106a6f198a4dd41c78d2a754be5d0b2c4`）。

### 可扩展 KV cache 在 graph warmup 后按实测余量提交物理页
CUDA/ROCm V2默认先用VMM预留稳定虚址，仅backing warmup碰到的块；capture后按实测transient峰值和margin决定容量，跨worker/DP取最小值再extend。未显式设utilization时改用1.0上限。紧随#65225改用segment padding保证RDMA注册整物理段，删除按block数向下取整造成的小pool归零/容量损失。

**证据与边界：**作者GB200八模型容量+4–10%、init快21–34%混入默认0.92→1.0的策略变化；固定0.92时容量反小1–4%。V1、HiSparse、elasticEP、固定cache大小、legacy IPC/custom pool及部分ROCm connector关闭/拒绝；direct reload无cache-clearing pause仍需headroom，不是任意负载不会OOM。#65225最终102/1331block与NIXL P/D测试另有证据。 **建议：**升级以#65225之后SHA为基线，记录实测KV、warmup峰值和connector布局；共享GPU显式设上限，RL更新需先清cache并回归sleep/wake。
源码路线：`utils/extensible_tensor.py::ExtensibleTensor；vmm_driver.py；worker/extensible_kv_cache.py；gpu_worker.py::_measure_kv_cache_blocks；kv_cache_planning.py::finalize_extensible_kv_cache；#65225 segment padding`。
来源：[vllm-project/vllm #56492](https://github.com/vllm-project/vllm/pull/56492)（main，2026-11-02T03:22:55Z，`860da3dafdbb1e24a6aef821b4ca7bceeb9b87d8`）；[vllm-project/vllm #65225](https://github.com/vllm-project/vllm/pull/65225)（main，2026-11-03T00:56:42Z，`8cf1277362aef963a71cf6860c28387acf6ecb90`）。

### AV 下 P/D prompt tail 保持 target/drafter FULL graph
传入decode实例的最后prompt token虽has_prefill，却是已有KV上的decode-eligible单token扩展；runner与EAGLE/MTP drafter现在按decode_graph_eligible保留max_query_len bound，避免一次arrival让所有DP rank回eager。

**证据与边界：**两GB200节点、1P1D DEP4、DSpark K5、FULL_DECODE_ONLY的2.0×/1.76×吞吐来自特定短/长输出、oversubscribed配方；FULL_AND_PIECEWISE已有fallback时收益近噪声。真实prefill仍eager/piecewise，不增加graphs；EP质量非逐位确定。 **建议：**跟踪arrival与纯decode各自graph mode，target/drafter一起升级；以实际arrival率和队列深度评估，勿泛化为全部spec decode翻倍。
源码路线：`gpu/model_runner.py；gpu/spec_decode/target_dependent_ar/speculator.py；test_prompt_tail_keeps_decode_graph_bound；test_draft_prefill_matches_shared_target_metadata_shape`。
来源：[vllm-project/vllm #65041](https://github.com/vllm-project/vllm/pull/65041)（main，2026-11-03T00:55:03Z，`74ea8d37c2395dec3574c1b3f880f3da696df4b7`）。

### SM100 sparse-indexer decode logits 按 query 行限额分块
新增VLLM_SPARSE_INDEXER_MAX_DECODE_LOGITS_MB默认2048，独立行切成平衡且16行对齐chunk，逐chunk跑DeepGEMM logits/candidate/top-k并原位写输出；profile只预留一个BF16 chunk，避开全decode-row张量及旧FP32估算。

**证据与边界：**限SM100 auto/persistent top-k512/1024/2048且无DCP/PCP；compact sparse consumers、DeepSelect、GB300特化路由不变。作者GB200指定DSv4.1/GLM容量改善明显，但纯eager宽mixed增加launch，最差配置退化；含open#63406的graph结果不得归给main本项。2^32元素top-k寻址问题只是被默认chunk规避，非底层地址修复。 **建议：**观察KV容量与host launch代价，FULL_AND_PIECEWISE优先；可调4096MiB但1M上下文别跨越单call溢出边界，budget=0会关闭。
源码路线：`sparse_attn_indexer.py::decode-row chunking；envs.py::VLLM_SPARSE_INDEXER_MAX_DECODE_LOGITS_MB；test_top_k_per_row.py decode_chunk cases`。
来源：[vllm-project/vllm #64806](https://github.com/vllm-project/vllm/pull/64806)（main，2026-11-02T23:48:41Z，`628c455e9ea508f40d723c9bf8d07014b62d3bda`）。

### FlashInfer all-reduce 补 row-width guard 与 HiSparse fallback
#63123要求每行字节宽是16的倍数，原本MNNVL会崩溃/TRTLLM可能静默截断的形状回NCCL；#65201在HiSparse+auto时单机选TRTLLM，多机关闭FI，绕过MNNVL cluster-barrier竞争。

**证据与边界：**显式backend仍优先，不自动保护强制mnnvl。HiSparse底层FlashInfer修复尚未进入所用依赖；H100 reduced dummy服务未复现自然崩溃，B200隔离实测TRTLLM可稳定但约3%decode代价，不能报H100吞吐数字为通用收益。 **建议：**Mamba2窄行TP及HiSparse压测都检查真实backend日志；保留NCCL对照与高H2D负载。
源码路线：`flashinfer_all_reduce.py::should_use_fi_ar/_resolve_fi_ar_backend；test_flashinfer_all_reduce.py；test_comm_ops.py`。
来源：[vllm-project/vllm #63123](https://github.com/vllm-project/vllm/pull/63123)（main，2026-11-03T00:51:46Z，`0834bcf6f29c1636a08677217289c762b134dbe2`）；[vllm-project/vllm #65201](https://github.com/vllm-project/vllm/pull/65201)（main，2026-11-02T20:33:18Z，`4910ee107cc4c16adf321abf0a684174fdc2dab8`）。

### NIXL 可对 head-contiguous NVFP4 KV 页做异构 TP 重分片
AttentionSpec新增kv_head_major_slots，MiniMax-M3 NVIDIA MSA NVFP4声明每head连续；handshake允许该布局异构TP并逐head注册byte range。FlashInfer NVFP4仍拒绝，dense BLHNC仍按层descriptor。

**证据与边界：**573项unit测试；真实字节kernel与TP1→TP4端到端来自早期等价实现，最终spec flag版本未完整重跑模型。ROCm MSA布局不设置此flag，不能外推所有NVFP4缓存或任意布局。 **建议：**在两端核对spec合并传播与实际页字节排列，再测试相同/异构TP；不以NVFP4 dtype本身判断可reshard。
源码路线：`v1/kv_cache_interface.py::AttentionSpec.kv_head_major_slots；nixl/base_worker.py；sparse_attention_msa.py；test_nvfp4_kv_head_major_slots_match_insert`。
来源：[vllm-project/vllm #64475](https://github.com/vllm-project/vllm/pull/64475)（main，2026-11-02T20:31:23Z，`ca66f0906ffa7a9ffcbddcaa5f6ab919332a3bcb`）。

### FS/对象存储成功写回刷新异步 lookup 的 cached miss
成功store无论events是否启用都保留TransferJob并调用mark_present；推进generation，防止较早pending/inflight probe的miss覆盖更新，孤儿条目可丢弃，仍复用跨request缓存miss。

**证据与边界：**仅本实例成功store；其他实例写共享存储不可见问题#61658仍未关闭。作者host-side102通过/11跳过，不构成跨实例一致性协议或模型性能结果。 **建议：**对store完成前后与旧probe晚到做交错测试，区分本地更新与外部writer invalidation。
源码路线：`tiering/async_lookup.py::mark_present；fs/manager.py::get_finished_jobs；obj/manager.py::get_finished_jobs`。
来源：[vllm-project/vllm #64444](https://github.com/vllm-project/vllm/pull/64444)（main，2026-11-02T18:55:10Z，`5cdeea9b13670c9da832d8ea72d799c8bfbb4552`）。

### P2P reset/取消/注册失败分别补生命周期清理
#63988 reset先失败尚待decoder fetch的queued stores并释放pin，真实inflight WRITE等transport完成；#64300用request_finished标记隔离晚到失败，避免finished ID永久重回_failed_req_ids并污染重试；#64819注册失败先best-effort Disconnect再标dead，disconnect失败进入既有故障/backoff。

**证据与边界：**三项各有独立unit/部分localhost NIXL TCP真实LLM证据，不等于consumer slot quarantine。原#61667仍open、head4fd16ddb，abort-ack timeout后的晚WRITE隔离风险仍在；不能因这些拆分合入关闭#61532。 **建议：**把producer无decoder reset、mid-load取消后复用ID、registration半成功分开故障注入；RDMA与多host仍需部署验证。
源码路线：`p2p/manager.py::drain_jobs；session/server.py::drain_stores；LoadResult.request_finished；session/session.py register failure handling`。
来源：[vllm-project/vllm #63988](https://github.com/vllm-project/vllm/pull/63988)（main，2026-11-02T18:54:21Z，`8c8d25a96358e5ff378af30512b42c73bc7c14a2`）；[vllm-project/vllm #64300](https://github.com/vllm-project/vllm/pull/64300)（main，2026-11-02T18:53:10Z，`fc86a5fba6a0b0bad897820fb4ccede4ff1eef81`）；[vllm-project/vllm #64819](https://github.com/vllm-project/vllm/pull/64819)（main，2026-11-02T17:32:08Z，`f1df4a3ae95e63db7ad539395f92c9397c21360a`）。

### hybrid partial-tail lookup 按 full-attention 已证实前缀寻找
complete-chunk命中要求recurrent组在同边界有state，可能把anchor拉回0；现在使用已有full-attention边界向后搜tail，只在候选位置检查recurrent，跳过已miss的完整chunk边界并处理pending loads。

**证据与边界：**本项是lookup半边；DCP store半边#58619另案，文中硬件DCP修复数字用了两者，不可归本PR。现有supports_partial_tail仍仅uniform blocks/dcp1；作者测试保留三项基线失败。 **建议：**区分存入了tail与能够查到tail，回归aligned state被驱逐、别的request有更长prefix、GPU prefix hit和pending anchor。
源码路线：`offloading/scheduler.py::_lookup；test_scheduler.py partial-tail cases`。
来源：[vllm-project/vllm #58618](https://github.com/vllm-project/vllm/pull/58618)（main，2026-11-02T12:37:17Z，`01b6a05b7b43f75a3f8a0f91e7b71f9776012170`）。

### Mooncake decode 复用本地 divergent hybrid attention prefix
kv_consumer显式opt-in supports_divergent_local_hybrid_hits，保留本地attention命中，仅从P拉uncached suffix和recurrent state，producer/kv_both保留旧lookup。

**证据与边界：**MI355X Kimi-K3日志hit0→93.9%来自旧基线强制同属性的单次run，途中被集群取消，缺最终snapshot且TTFT未变；不能写成吞吐或延迟提速。MultiConnector仍受每child opt-in限制。 **建议：**多轮会话按local-hit/external-transfer分项核字节，确认recurrent state仍拉取完整，别仅看总prompt cache率。
源码路线：`mooncake_connector.py::supports_divergent_local_hybrid_hits；test_mooncake_connector_hybrid_mamba.py`。
来源：[vllm-project/vllm #63971](https://github.com/vllm-project/vllm/pull/63971)（main，2026-11-02T02:53:34Z，`62c9b6263cea4a31444e597ebde1b1fe64dbe208`）。

### MRV2 warmup 与全 vocab prompt-logprobs 控制运行期显存
#62560在graph capture/empty_cache后补最大batch sampler warmup，预留FP32 logits copy；#64914把每次torch.topk限制为rows×(k+1)≤64M元素并及时释放结果，减少全vocab长prompt的workspace峰值。

**证据与边界：**sampler收益依赖已合#57926真正走FP32 dummy路径，实测R9700、未测NVIDIA；topk只影响超大k，作者GB200全vocab约3.5%时延代价，默认max_logprobs=20基本不切块。容量来自实测warmup，不能保证所有未见负载无增长。 **建议：**以满batch ramp和最大logprob配置验startup后显存，区分reserved/allocated、warm cache与瞬时峰值。
源码路线：`gpu_worker.py::compile_or_warm_up_model；gpu/sample/prompt_logprob.py::TOPK_MAX_ELEMENTS/compute_prompt_logprobs_with_chunking`。
来源：[vllm-project/vllm #62560](https://github.com/vllm-project/vllm/pull/62560)（main，2026-11-02T17:11:38Z，`b18775f8938b0c264643b7cd344a9ca2441f5b67`）；[vllm-project/vllm #64914](https://github.com/vllm-project/vllm/pull/64914)（main，2026-11-02T01:47:09Z，`16d667348d91ae010cb3cc9d4d6a5b7908a717a9`）。

### weight reload 后原位刷新派生 tensor，统一 post-load hook 遍历
loader统一无参post-load hook并children-first调用每模块own hook，跳过已由quant/attention阶段处理的节点；后续#64692将kernel constants、inverse scales、decode copies等派生tensor改原位refill/保留，避免CUDA graph读新权重的旧地址。

**证据与边界：**涉及多MoE/linear/drafter后端，作者完整模型reload结果有不同早期revision，部分perturbed checkpoint本身接近0分只能比fresh-server tokens；不是所有第三方hook自动兼容。Humming direct reload仍拒绝。最终74-file diff已取得，本轮仅追踪关键helper/representative consumers，未运行17模型矩阵。 **建议：**自定义模块迁移hook签名，测试重复post-load与真实不同权重refit后的地址、derived scale、fresh-server输出；同权重更新不足以证明值刷新。
源码路线：`model_loader/weight_utils.py::finalize_model_weights；model_loader/utils.py::process_weights_after_loading；model_executor/utils.py::replace_parameter；experts kept_constants/keep_constants；draft post-load paths`。
来源：[vllm-project/vllm #57678](https://github.com/vllm-project/vllm/pull/57678)（main，2026-11-02T13:53:39Z，`f9db1756e9e6809b209bb906c950ac577fdef90f`）；[vllm-project/vllm #64692](https://github.com/vllm-project/vllm/pull/64692)（main，2026-11-02T19:54:09Z，`26d198bd56ae974201fd56e5150acb8b1e29042b`）。

### batch-invariant GPTQ 绕开 Marlin，付出每步反量化代价
VLLM_BATCH_INVARIANT下不再Marlin重排/选择tiling，保留checkpoint packed布局，forward反量化后调用batch-invariant matmul；普通模式不变。

**证据与边界：**解包合同是对称GPTQ、无act-order；作者A10上指定Qwen2.5 GPTQ 22项由14fail变全通过。每次forward展开权重，吞吐更低，不是更快GPTQ；AWQ此前已有路径，不属于此增量。 **建议：**需要RL logprob逐位批不变时再启用，明确记录内存/吞吐取舍及TP/量化配置，不把开关存在等同所有backend已确定。
源码路线：`auto_gptq.py::AutoGPTQLinearMethod；batch_invariant.py::dequantize_gptq_weight/gptq_linear_batch_invariant；test_gptq.py`。
来源：[vllm-project/vllm #60479](https://github.com/vllm-project/vllm/pull/60479)（main，2026-11-02T20:13:23Z，`57371c4e283c65b348303829b7f515ea74a94fb2`）。

### DCP top-k selector 直接读取 rank-major all-gather buffer
候选(score,id)改dim0 all-gather，保持[W,R,K,2]视图，CuTe selector按编译常量rank/candidate寻址，删除movedim+reshape全量reorder copy，候选集合不变。

**证据与边界：**K须512倍数且当前merge允许512/1024/2048；作者B200114项对照通过。5–6%prefill结果来自内部模型下游fork，decode走另一交换路径；尚无公开模型端到端结果。 **建议：**优化通信后先查是否还有布局materialization，核对跨rank tie/稳定选择与warmup shapes，不直接外推公开DSv3.2吞吐。
源码路线：`sparse_attn_indexer.py::_merge_dcp_topk_global；dcp_indexer_cutedsl.py::stable_topk_from_gathered_candidates_cutedsl`。
来源：[vllm-project/vllm #65178](https://github.com/vllm-project/vllm/pull/65178)（main，2026-11-02T22:52:35Z，`a11d3a15f6f04ae426a0a5162f293e95163103cc`）。

### FlashInfer 布局声明提前拒绝无法寻址的 LHBNC
非SM100平台从无约束None改为仅NHD/HND可表达的五种layout，LHBNC在解析阶段给有效列表而非加载后assert；默认LBNHC与SM100两种head-major布局保留。

**证据与边界：**A100真实startup检查基于v0.30相同函数，unit模拟SM80/90/100/120；它不会让LHBNC突然受支持，也没有新性能结果。 **建议：**自定义KV connector先协商layout合同，检查解析错误而非等启动后backend崩溃。
源码路线：`attention/backends/flashinfer.py::supported_kv_cache_layouts；backends/utils.py::get_flashinfer_kv_cache_layouts`。
来源：[vllm-project/vllm #59999](https://github.com/vllm-project/vllm/pull/59999)（main，2026-11-02T22:14:19Z，`3a228c1fefc99f32dd9402610ab9a3ffaff66406`）。

### DSv4.1 大 eager step 的 Q padding 改连续copy
仅32 live/64 padded heads、≥2048 token、eager MegaAttention，fused insert走已有KV-only，再由Triton一读连续写chunk-interleaved Q及零padding，避免fused warp跨stride碎片写。

**证据与边界：**TP2特定head形状；capture、小step、其他TP/平台不变。B300单算子1.9–2.15×是Q pad+KV insert，不是模型提速；Q位等价限定NaN-free。多一个eager launch决定阈值。 **建议：**用真实prompt长度测整forward，回归layout、空输入与padding；不要在小batch照搬copy路径。
源码路线：`deepseek_v41/attention.py::_pad_interleaved_q/_Q_PAD_COPY_MIN_TOKENS；test_fused_deepseek_v4_qnorm_rope_kv_insert.py`。
来源：[vllm-project/vllm #62632](https://github.com/vllm-project/vllm/pull/62632)（main，2026-11-02T20:43:09Z，`28dd9c36b5fac772e4cccf9563ab9b19ac09edf0`）。

### SM100 FP8 conversion 成对发射，并保留 NaN 语义
VLLM_FP8_CVT_X2 gate统一控制两值一次cvt，饱和转换代替显式clamp，覆盖norm/quant/merge等调用；dynamic全零输入专门输出零避免0×inf。

**证据与边界：**普通有限输入作者515个tensor原始bytes/scales相同；NaN现在保留FP8 NaN而不是被clamp成448，全零旧输出也改变，不能无条件写bit-identical。性能是B300隔离kernel，Hopper/ROCm路径保留。 **建议：**量化回归加入zero、NaN、非对齐view和奇数维，单独监控异常数值传播，别只测随机有限样本。
源码路线：`csrc/core/cuda_compat.h::VLLM_FP8_CVT_X2；quant_conversions.cuh；fp8/common.cu；test_dynamic_per_tensor_fp8_quant_all_zero`。
来源：[vllm-project/vllm #61641](https://github.com/vllm-project/vllm/pull/61641)（main，2026-11-02T20:12:02Z，`4a0dfc7b2f72bbb694a54f942336ea2ff6ccf7a8`）。

### mixed-precision KV zeroing 决策跨 PP worker 保留
needs_kv_cache_zeroing由完整worker配置计算并存字段，planner跨worker OR后再缩成scheduler spec，防止某PP stage因本地uniform而不建zeroer却收到new_block_ids_to_zero。

**证据与边界：**uniform stage也会跟随需要的stage多zero新块；只修metadata/执行一致性，非全局所有reuse/transfer尾部NaN问题。作者GB200 Qwen PP2与PP1首token top5只比较到两位小数，非MiniMax完整质量测评。 **建议：**PP精度按层混合时把混合组放不同stage分别回归，观察zeroer初始化与首次generation。
源码路线：`kv_cache_interface.py::KVCacheConfig.__post_init__；kv_cache_planning.py::get_kv_cache_configs；kv_cache_utils.py::generate_scheduler_kv_cache_config`。
来源：[vllm-project/vllm #64472](https://github.com/vllm-project/vllm/pull/64472)（main，2026-11-02T19:54:36Z，`8278030ebab51176ab87e9d93345c7ff1f300844`）。

### gfx942 MiniMax-M3 可显式使用 FP8 indexer cache
Triton scorer允许gfx942 FP8；cache/query用platform e4m3fnuz，两个writer按该flavour范围饱和，避免误用CUDA e4m3fn/448。gfx950仍AITER路由。

**证据与边界：**作者MI325X/TP4指定MiniMax动态FP8、262K，KV tokens约+32%，短上下文吞吐近中性，长上下文小收益；这不是NVFP4 MSA支持或gfx950改动。61通过、32项NVFP4跳过。 **建议：**明确indexer_kv_dtype=fp8，检查日志flavour与writer饱和值；不要把attention KV与indexer side cache混为一层量化。
源码路线：`minimax_m3/common/indexer.py::select_indexer_impl_cls；amd/ops/sparse_pa.py::_insert_index_cache_kernel；fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu`。
来源：[vllm-project/vllm #62270](https://github.com/vllm-project/vllm/pull/62270)（main，2026-11-02T19:52:58Z，`901f26dfcd7a73283c79f34787debceac9b1a688`）。

### HiSparse host pool 单chunk也按多shm文件预取页
共享32段上限，按2MiB边界切single logical chunk，虚址仍连续；用sparse memfd预留地址避免需要可写temp目录和全pool commit charge，CUDA注册区间不变。

**证据与边界：**此前1TiB/H100 RAM准备快的数字未在最终memfd revision重跑；最终两rank8GiB真实cudaHostRegister验证跨段GPU写/peer可见约44GB/s只证明该配置。Linux memfd依赖不能泛化macOS。 **建议：**分别测prefault与pin阶段，验无tempdir启动、commit统计、跨段H2D及cleanup；勿将startup节省称服务吞吐提高。
源码路线：`shared_offload_region.py::SharedOffloadRegion/REGION_SEGMENTS；hisparse/runtime.py::allocate_hisparse_host_pools`。
来源：[vllm-project/vllm #61680](https://github.com/vllm-project/vllm/pull/61680)（main，2026-11-02T18:10:41Z，`bb197a852fbc3bf0a2b15bd57189dde616bf12c5`）。

### torch MXFP8 linear 可消费 DSv4.1 既有输入量化融合
TorchMxFp8LinearKernel声明CUDA kMxfp8Dynamic并接收QuantizedActivation/F8_128x4 scale，query quant与wo_a融合将其列为可兼容consumer，省重复量化；通过linear_backend_per_quant.mxfp8=torch显式选择。

**证据与边界：**默认backend不变。作者GB200 DEP4 dummy-weight target forward的大M约2.6–3.0%收益；本revision无完整权重/服务精度，部分小M GEMM更慢。不能沿用早期已删除专用linear/env设计或当全量模型benchmark。 **建议：**长prefill实例才作对照，核对scale swizzle合同与实际kernel，先补全权重质量和小batch回归。
源码路线：`kernels/linear/mxfp8/pytorch.py::input_quant_key/apply_weights；query_quant.py::can_fuse_query_quant；o_proj.py::_can_fuse_wo_a`。
来源：[vllm-project/vllm #64815](https://github.com/vllm-project/vllm/pull/64815)（main，2026-11-02T17:59:48Z，`b5814faa25cf23eb69a18d6e02a0984cd042fcb5`）。

### gfx942 长 sparse MLA prefill 调 AITER pa_prefill_sparse
实际CDNA3、AITER MLA启用、Q为fp16/bf16且KV同dtype、query≥512行时延迟导入并调用AITER，一池KV，输出形状匹配可原位写；更短仍in-tree ragged。

**证据与边界：**需AITER具备#6002 symbol，缺符号会在合条件调用抛错。多数E2E数字是旧1024阈值revision，旧GLM实验还把decode发往该kernel；最终decode保持in-tree，不能沿用旧decode收益。 **建议：**先检查依赖symbol与真实route日志，按512附近shape测，分开报告prefill和decode。
源码路线：`rocm_aiter_mla_sparse.py::_can_use_aiter_pa_prefill_sparse/_rocm_sparse_attn_prefill_aiter；test_sparse_attn_prefill_preserves_dense_triton_fallback`。
来源：[vllm-project/vllm #60102](https://github.com/vllm-project/vllm/pull/60102)（main，2026-11-02T17:43:29Z，`887eef12e1e586245e135733583afc3c52f4ba53`）。

### ROCm MiniMax NVFP4 indexer 的 DSpark7 继续使用 decode CP
TP4将每请求8query行拆成两份4行CP请求，复制block-table并调整第一份causal seq_len；纯decode不被DBO实际拆分时也能初始化/使用CP，kernel数学不改。

**证据与边界：**实际ubatching仍拒绝共享CP buffers，只有unsplit步骤可用；两份各扫描本rank quarter，相当每rank总读1/2而非native8行的1/4。DBO性能用了额外#63483，DSpark3→7总增益不能全部归本项，早期无DBO E2E没有显著收益。 **建议：**同时核对DBO是否真正拆batch、max_rows、padding零长度与K3/K7 causal边界，先做bitwise top-k再测服务。
源码路线：`amd/indexer_nvfp4.py::_cp_requests/_decode_cp；index_cp/host.py::init；test_index_decode_cp_push_split_requests`。
来源：[vllm-project/vllm #64448](https://github.com/vllm-project/vllm/pull/64448)（main，2026-11-02T17:37:25Z，`eff37e8e242022766115aa9b2944a1b00d6beb31`）。

### Qwen3-VL encoder graph replay 正确 padding FlashInfer 双份 offsets
Qwen3-VL注册已有FlashInfer cu_seqlens半区专用padding helper，其他backend重复末offset形成零长度sequence，修复capture/replay bucket不同时q/k/o与v两半错位。

**证据与边界：**作者最终head RTX4090有replay测试，早期110条件/1368shape-transition与当时配套branch区分；同padding eager可逐位，unpadded与padded eager仍有线性层形状数值差异，不是端到端生成质量逐位保证。 **建议：**分别比replay↔同padding eager和unpadded eager，覆盖图像/视频多shape，不将视觉embedding漂移误诊为随机采样。
源码路线：`qwen3_vl.py::get_encoder_cudagraph_config.padding_logics；TestFlashInferCuSeqlensPadding`。
来源：[vllm-project/vllm #64984](https://github.com/vllm-project/vllm/pull/64984)（main，2026-11-02T17:06:13Z，`45fc9fd0485d0814e68f0b73893621ca12a255b8`）。

### Kimi-K3 更新 FlashKDA/FlashMLA pins 以阻断 NaN 扩散与 PDL 早读
依赖pin指向两项已合入kernel修复：FlashKDA末尾partial chunk用select零而非0×越界邻请求NaN；FlashMLA SM90 dense decode在读query前等待PDL producer。

**证据与边界：**自然burst零坏请求的最终arm用了no-PDL变体；当前pin的wait变体有forced-race与时延测试，但未跑同自然burst。首NaN根因仍suspect，FlashKDA只明确阻断传播；不能说自然NaN已彻底根除。 **建议：**构建时核实两submodule精确pin，再做自然burst、NaN注入隔离和sm90依赖压力；不能只跑常规GSM8K。
源码路线：`cmake/external_projects/flashkda.cmake；flashmla.cmake；PR正文区分forced/natural probes；上游实现未在本轮展开审计`。
来源：[vllm-project/vllm #64759](https://github.com/vllm-project/vllm/pull/64759)（main，2026-11-02T16:22:32Z，`3b6eeb060621eba88deb74a146039291a1afbc42`）。

### pre-SM90 MoE top-k 不再带无效 PDL launch attribute
topkGating只在SM90+设PDL属性，避免旧架构虽不能PDL却在CUDA graph留下programmatic edge、每层replay增加成本；kernel body不变。

**证据与边界：**作者SM80/86/89 standalone与A100模型证据；私有Qwen配方6.5–13.6%不能用于公开模型，公开OLMoE/Qwen GPTQ batch1仅约0.7–1.2%，batch16近噪声；SM90+未测且构造保持原launch。 **建议：**旧卡回归graph-edge与replay开销，并以相同toolchain no-change rebuild作对照；不要把单op launch开销等同所有MoE吞吐。
源码路线：`csrc/moe/topk_softmax_kernels.cu::topkGating launch`。
来源：[vllm-project/vllm #65059](https://github.com/vllm-project/vllm/pull/65059)（main，2026-11-02T13:52:49Z，`187f1a27dc7e1289d7b0bbafddb84fc9fbceea44`）。

### RDNA W4A16 GEMM 在乘 stride 前提升 row index 至 int64
pid_m先转int64再生成offs_m，使activation load/output store的M×K/M×N地址不在2^31附近回绕，修复大prefill NaN或HSA memory fault/hang。

**证据与边界：**仅gfx11/gfx12 hybrid skinny-format kernel；作者R9700两种边界case验证，MI300/MI355 CI不覆盖该kernel。误差仍按现有bf16容限，非bitwise。 **建议：**大M场景用2^31//max(K,N)附近case回归，cast必须先于乘法；保留已修CDNA sibling的独立覆盖。
源码路线：`rdna_hybrid_w4a16.py::_triton_w4a16_skinny_fmt_kernel；test_rdna_hybrid_w4a16.py int32_offsets`。
来源：[vllm-project/vllm #64614](https://github.com/vllm-project/vllm/pull/64614)（main，2026-11-02T04:06:07Z，`3d23773f101a79d0eac510733f2d825af6701265`）。

### LoRA 修复同名热重载身份与 Gemma4 QKK 打包
#64616 equality/hash同时比较name+int_id，使旧inflight与同名新adapter可同时激活；#61267允许per-module lora_packed_modules，Gemma4 attention_k_eq_v的full-attention把k_proj delta同时写K/V。

**证据与边界：**两项是不同bug，未改变unload何时生效。热重载作者CPU/MRV1实测，MRV2主要unit/CI；Gemma内部adapter10prompt对照不等同公开任务精度。 **建议：**用old/new不同adapter同时inflight和驱逐触发身份回归；QKK按每层full/sliding分别验V delta。
源码路线：`lora/request.py::LoRARequest.__eq__/__hash__；lora/model_manager.py::_register_packed_modules；gemma4.py::lora_packed_modules`。
来源：[vllm-project/vllm #64616](https://github.com/vllm-project/vllm/pull/64616)（main，2026-11-02T03:07:55Z，`eecf8dbb2b279372733c844a9735262494b3fa9b`）；[vllm-project/vllm #61267](https://github.com/vllm-project/vllm/pull/61267)（main，2026-11-02T03:07:36Z，`bc09e63d7957e18b0bd1c46bb162aea99d68a57f`）。

### V2 rejection sampler 返回指定 token-id 的 logprob
spec混批时即便没有top-k logprobs，也按get_logprobs_dims包含logprob_token_ids列，传expanded_idx_mapping给compute_topk_scores，修复旁边有draft请求时systemone读取报无logprobs。

**证据与边界：**作者ROCm MTP复现和31项本地测试；CUDA专用tests在ROCm改gate才运行，TP>1 gather callsite未测，DSpark未独立验证。不是概率定义或采样算法改变。 **建议：**同时回归token-id-only、比top-k更宽、AV compact logits、batch sharding，不以单独请求成功覆盖混batch。
源码路线：`gpu/sample/sampler.py::get_logprobs_dims；spec_decode/rejection_sampler.py::_get_logprobs_tensors；test_gpu_rejection_sampler_chunking.py`。
来源：[vllm-project/vllm #61219](https://github.com/vllm-project/vllm/pull/61219)（main，2026-11-02T03:00:37Z，`e8ec12c5edc3e14598bb122237d2798a3d4f3fa0`）。

### ROCm Kimi-K3 padded projection 按真实元素完成 online loading
AMD局部PaddedKimiGDNMergedColumnParallelLinear用weight_loader_numel排除alignment行，六份真实shard到齐即可materialize/处理并释放源tensor；loader在empty_strided materialization后重新清零padding。

**证据与边界：**最终revision有MI355X八case/既有181项测试；完整16worker内存/一小时服务是早期同计数、不同清零位置revision，且有7项空内容error及cancel-drain timeout，不能称完整无错误通过或质量提升。 **建议：**检查完成计数、source引用寿命与meta→real后padding，量化大模型加载内存别把source bytes与allocator计数相加。
源码路线：`kimi_k3/amd/kda.py::_PaddedKimiGDNMergedColumnParallelLinear；weight_loader/weight_loader_v2/_zero_padding`。
来源：[vllm-project/vllm #61950](https://github.com/vllm-project/vllm/pull/61950)（main，2026-11-02T02:57:26Z，`de53c2deed03abd19cdef4f2bddcfcef60a84610`）。

### MiniMax NVIDIA norm 与 MXFP8 quant 合并，按 scale layout 选 consumer
二维Gemma RMSNorm可融合residual并直接产生MXFP8/E8M0 swizzled scale，兼容FlashInfer CuTe/CUTLASS consumers才开启量化；plain input/final/MTP norms复用BF16输出，drafter per-head仍原kernel。

**证据与边界：**residual/scales bitwise，但E4M3允许1code distance、BF16输出1ulp，不能写完整逐位等价。GB300 isolated1.39–1.60×不是整模型；c32 decode含退化/噪声，高并发才测得收益。 **建议：**回归consumer scale布局与shape分支，按实际并发比较TTFT/TPOT，并保留低并发基线。
源码路线：`minimax_m3/nvidia/model.py::_consumes_mxfp8_128x4/MiniMAXGemmaRMSNorm；ops/norm_mxfp8.py::gemma_rmsnorm_mxfp8`。
来源：[vllm-project/vllm #64852](https://github.com/vllm-project/vllm/pull/64852)（main，2026-11-02T05:16:52Z，`318c4db661d9706764ee3d9e1b9c14fc4e49c75b`）。

### MiniMax TP+MegaMoE 用 token-sharded residual，GEMM-RS 可选
TP>1+EP+MegaMoE+PP1自动SP，attention前AG、o_proj后RS，MoE直接消费本rank token shard，替代两次全量AR；dense层MLP另加AG/RS。GEMM-RS由VLLM_ENABLE_GEMM_RS启用，MiniMax阈值1024token，并接收既有MXFP8 QuantizedActivation。

**证据与边界：**GB300 TP4实验SP arm同时开GEMM-RS，不能拆成各自独立收益；短并发纯TP仍最快，SP相对EP-only c1吞吐约-4%。PP>1不启，不能泛化任意TP/EP拓扑。 **建议：**比较TP、EP-only、SP及GEMM-RS四组合，按真实并发取舍，核对residual shard、padding与draft路径。
源码路线：`minimax_m3/nvidia/model.py::_use_sequence_parallel/_maybe_init_gemm_rs；gemm_rs_ar.py::GemmRsAr.apply`。
来源：[vllm-project/vllm #64659](https://github.com/vllm-project/vllm/pull/64659)（main，2026-11-02T04:00:56Z，`37e16689fa4c92068a0ed9c165d1496696434aba`）。

### EC producer 控制端口直接绑定0并发布真实endpoint
默认由OS分端口，读取ZMQ LAST_ENDPOINT并通过peer_port广播，agent name加DP index；显式固定port按base+DP index，保留host override并拒绝wildcard。

**证据与边界：**固定port冲突明确失败而非自动逃避；112项unit含ZMQ/NIXL registration，最终revision未重跑GPU服务。不是KV connector改动，属于encoder-cache EC。 **建议：**同host多instance用自动端口或不重叠range，核对跨机可达host而非默认loopback。
源码路线：`ec_transfer/ec_connector/cpu/control/zmq.py::ZmqServerTransport；scheduler/__init__.py::_setup_nixl；envs.py`。
来源：[vllm-project/vllm #61764](https://github.com/vllm-project/vllm/pull/61764)（main，2026-11-02T03:54:51Z，`a9cf9df7522708e5d16eb5895569e1037d4cde9d`）。

### FlashInfer dense GEMM eager 绕过 custom-op 调度层
eager直接调用实现，compile时仍保留原op schema/fake boundary；部分慢custom_op改direct_register_custom_op，MXFP8量化helper缓存backend选择，减少host enqueue开销。

**证据与边界：**31–53µs是复刻调用链harness，不是完整vLLM服务；缺编译扩展时NVFP4量化用等schema替代，绝对时延可能偏高。不能将配套FlashInfer#7788收益重复相加，也不代表compiled GPU kernel更快。 **建议：**host-bound eager prefill先profile enqueue，再验fullgraph仍记录op与无graph break；native扩展全路径另测。
源码路线：`utils/torch_utils.py::eager_bypass_custom_op；utils/flashinfer.py；kernels/linear/scaled_mm/flashinfer.py；mxfp8_utils.py`。
来源：[vllm-project/vllm #64417](https://github.com/vllm-project/vllm/pull/64417)（main，2026-11-03T00:50:43Z，`3c01fae696376ceefa94fcdb75907a3cd8d4b9d4`）。

### 统一 request holder 后修复 paused decode held-request abort
request_lists在调用时枚举waiting/running/PD/HiSparse/dLLM/uOCR/deferred holders；最后chunk已launch但未转交的prefill有独立owner。debug ownership检查集中于step launch，真正行为增量#43905使paused rebootstrap-held decode可被abort，sampling-mask检查看见更多queue。

**证据与边界：**前几项主要所有权重构/debug，不能逐条包装为吞吐新功能。CI状态仍有基线失败/未完成；held abort只释放无KV且无step使用的请求，不是任意inflight KV可立即回收。 **建议：**回归pause→held→部分abort→resume、PP inflight与mixed步骤，使用SGLANG_ENABLE_RUNNING_REQS_CHECK观察唯一holder。
源码路线：`Scheduler.request_lists/_on_step_launch/abort_request；RequestList；SchedulerInvariantChecker.maybe_check_request_lists；DecodePreallocQueue.take_aborted`。
来源：[sgl-project/sglang #43901](https://github.com/sgl-project/sglang/pull/43901)（main，2026-11-02T07:45:50Z，`960e30769e8af4618ca0f7a6f1d85464176d881b`）；[sgl-project/sglang #43902](https://github.com/sgl-project/sglang/pull/43902)（main，2026-11-02T05:42:33Z，`768e56202b0ef523227fe8dc375937083194eccc`）；[sgl-project/sglang #43903](https://github.com/sgl-project/sglang/pull/43903)（main，2026-11-02T21:15:47Z，`3674c7173f6b9605efbd11b9d17e1a12b7b08da8`）；[sgl-project/sglang #43904](https://github.com/sgl-project/sglang/pull/43904)（main，2026-11-03T00:12:19Z，`797af7170f34dd4cb9e397e0038c52371ef5e050`）；[sgl-project/sglang #43905](https://github.com/sgl-project/sglang/pull/43905)（main，2026-11-03T00:24:58Z，`ba1ed00dfe87a63c99a6c15d888a4acea75cf469`）。

### MoE finalize JIT 优化因 B200 精度回退已回滚
主窗先合入#47784，随后#47994恢复旧moe_finalize_fuse_shared及wrapper；保留已被其他kernel使用的f32x2 math helpers。最终可用状态是旧finalize，不计入本日性能新增。

**证据与边界：**作者B200 GLM5.2-NVFP4 1P1D+HiSparse+MTP nightly由约0.95降至0.90，回滚branch0.950；B300没有同样失败。shared_output在PDL wait前早读仅为源码推测，未做针对性race证实，不能断言根因。 **建议：**若固定在两合入之间，移到回滚后；重提需side-stream shared-output+graph+PDL回归及B200 nightly，单op静态输入不能覆盖。
源码路线：`moe_finalize_fuse_shared.cu/.cuh；PR#47994 bisect/verification；math.cuh helpers保留`。
来源：[sgl-project/sglang #47784](https://github.com/sgl-project/sglang/pull/47784)（main，2026-11-02T02:09:18Z，`ca819c11cd2cd886bb651a4b44e000ab11f7e7ea`）；[sgl-project/sglang #47994](https://github.com/sgl-project/sglang/pull/47994)（main，2026-11-03T00:08:51Z，`e7a6c35f38a3e33eec196849a10ff3f52018e8ff`）。

### Ascend EPLB 按有效 route 统计并原位迁移 NPU 权重
NPU DeepEP所有mode选已mask为-1的TopK统计而非包含graph padding的recv counts；更新时NZ/strided权重暂转contiguous ND，调用共享单层迁移后copy回原storage，保持graph地址，buffer跨层复用。

**证据与边界：**硬guard要求TP=EP=WORLD，非任意hybrid topology。最终改四文件但新增NPU helper本身无新专用注册测试；正文多拓扑/性能证据需按作者具体环境理解，本轮未跑NPU，不能称普适load-balance提速。 **建议：**先构造真实token全落一rank、padding落另一rank的统计对照，回归unchanged sender、NZ回写和迁移临时峰值。
源码路线：`expert_distribution.py::SinglePassGatherer.init_new；npu/eplb.py::update_expert_weights_with_staging/_validate_migration_topology/_ExpertWeightStaging`。
来源：[sgl-project/sglang #47574](https://github.com/sgl-project/sglang/pull/47574)（main，2026-11-02T08:15:53Z，`d74749b0b63ed3373a0a34205ec635437d05ffc5`）。

### Ascend mixed-chunk 复用 extend 路由，保留 pure MLA 拒绝
forward_mixed把decode视为1-token extend，统一委托forward_extend以涵盖DSA top-k、FIA页布局、sink、SWA与DP padding；ModelSlim fused前缀量化元数据找不到时回原fused名，避免静默BF16。

**证据与边界：**pure MLA且无DSA仍NotImplemented；不是新FIA kernel或所有MLA支持。作者GLM5.2 w4a8等实测条件具体，不能把有限题集或旧backend结果视为所有NPU模型验证。 **建议：**mixed启用时核对实际量化方案而非只看加载成功，分别验MHA、DSA、sink/SWA与DP padding。
源码路线：`npu/attention/ascend_backend.py::forward_mixed/forward_extend；modelslim.py::get_quant_method`。
来源：[sgl-project/sglang #47367](https://github.com/sgl-project/sglang/pull/47367)（main，2026-11-02T06:21:19Z，`9dfa1374e2288790d4c960eff93f8a4a9304921b`）。

### PP proxy tensor 在 forward stream 读完前保护 storage
_pp_launch_batch对recv得到的非CPU proxy tensor record_stream(forward_stream)，阻止另一stream的新recv在queued forward未读完时复用旧storage。

**证据与边界：**第一stage无proxy不变；作者Kimi-K3 PP2×TP4、async depth2在8B300验证，没有穷举所有PP loops。无新sync，也不是延长tensor Python变量就能替代stream allocator合同。 **建议：**大async-batch-depth压测recv/forward前checksum，回归每loop及跨stage输入；和Megatron#8823一起排查同类异步所有权问题。
源码路线：`managers/scheduler_pp_mixin.py::_pp_launch_batch；PP proxy tensor record_stream`。
来源：[sgl-project/sglang #47614](https://github.com/sgl-project/sglang/pull/47614)（main，2026-11-02T07:03:32Z，`d9521615b790e4b0480024b87feca261547bb5ec`）。

### HiCache 失败回滚 host slot，Mamba 可选 donation 容忍池耗尽
#46606侧pool分配抛异常时逆序清理已分配slots，继续drain其余cleanup并链原异常，controller回收尚未交DMA的primary reservation；#47680 optional checkpoint donation失败返回0，active state/load-back仍严格assert。

**证据与边界：**前者是CPU allocation所有权证明，未测DMA/服务；后者本地七项模拟case，registered测试曾因macOS缺Triton无法collect。不能把optional skip说成required slot也永不耗尽。 **建议：**区分调用方预先持有与本次新分配slots，注入reclaim/alloc/free各阶段异常；观测donation跳过率而非只检查不崩溃。
源码路线：`pool_host/group.py::resolve_host_transfers/rollback；HybridCacheController.allocate_host_transfers；MambaComponent._try_alloc_mamba_slot/prepare_for_caching_req`。
来源：[sgl-project/sglang #46606](https://github.com/sgl-project/sglang/pull/46606)（main，2026-11-02T06:03:36Z，`3d33971dea1762fe2cc6dc8ec8172eecf3d6a3a4`）；[sgl-project/sglang #47680](https://github.com/sgl-project/sglang/pull/47680)（main，2026-11-02T04:26:04Z，`96b5d058b38cfb333a9eba738337f99cb1d161a0`）。

### ep_scatter 在 program 内求 expert prefix，删除跨warp回读
每个program从token counts局部求其expert之前总数，只写自己offset项，避免多个program反复写整表后无同步地读global旧值。

**证据与边界：**作者H20四项BF16/FP8、空expert/tile边界/graph回归和Compute Sanitizer通过；无完整模型精度，isolated更快不等于服务增益。 **建议：**检查生产/消费offset表的launch边界，对稀疏/空expert与重复graph做压力测试。
源码路线：`ep_moe_kernels.py::_fwd_kernel_ep_scatter_1；test_ep_scatter.py`。
来源：[sgl-project/sglang #44974](https://github.com/sgl-project/sglang/pull/44974)（main，2026-11-02T05:57:45Z，`806d8634a30d4dab286d51da9530ff991b219584`）。

### ROCm deterministic all-reduce 大输入改固定 rank 顺序两阶段
2-stage reduce-scatter仍按rank0..n-1累加，保持与旧1-stage相同位；4rank≥512KiB、8rank≥256KiB的unregistered-buffer路线选它，其余维持1-stage，正常非确定模式不变。

**证据与边界：**registered path和2/6rank仍旧kernel；MI350X整模型确定性实测含其他open确定模式补丁，不能把28–57%全部归此PR。新kernel测试明确覆盖order-sensitive inputs、BF16/FP16、graph阈值；未外推任意拓扑。 **建议：**逐元素比固定FP32 rank-order参考，跨batch size和阈值两侧验位等价；把配套patch集与单PR收益分开。
源码路线：`custom_all_reduce_hip.cuh::cross_device_reduce_2stage<kFixedOrder>；deterministic_all_reduce.hip::launch_deterministic；test_amd_deterministic_allreduce_fixed_order.py`。
来源：[sgl-project/sglang #46325](https://github.com/sgl-project/sglang/pull/46325)（main，2026-11-02T16:57:57Z，`16dd97df610d5a3f5b2c07d41c5ae9a2a2f25c5b`）。

### DSv4.1 decode 融合 mHC post/combine/norm，小batch再分 CTA cluster
Blackwell、mult4/hidden5120/BF16、≤1024行，单launch更新4-stream residual、下层combine、RMSNorm及可选MXFP8；后续小行用4/5CTA cluster拆写及weight预取，较大行按occupancy选unsplit/2split。

**证据与边界：**batch-invariant跳过；residual位等价但RMS归约顺序使norm末bit可变。#47949阈值B300调优、无最终完整模型A/B且未B200timing，侧stream竞争可能拖延cluster；不能把基础#46088整模型验证归给新cluster。 **建议：**分别测base fused与cluster增量，覆盖128/129、SM数、640/641边界与侧stream负载。
源码路线：`deepseek_v4_mhc.py::_post_fusion；mhc_post_combine_norm_decode.cuh；mhc_post_combine_norm_decode.py::_pick_config`。
来源：[sgl-project/sglang #46088](https://github.com/sgl-project/sglang/pull/46088)（main，2026-11-02T06:53:29Z，`b0debe0a7aa827b34c1987a234c07eeedf14c68f`）；[sgl-project/sglang #47949](https://github.com/sgl-project/sglang/pull/47949)（main，2026-11-02T17:53:15Z，`e152f531d9553906aa421c85da290d55c6e9492c`）。

### DSv4.1 eager sparse-indexer prefill 可按 TP 切 query 行
原attention-TP复制相同rows并重复dense logits/top-k；opt-in RowShard每rank只算连续query份额，再AG恢复完整selection，减少长prefill重复计算。

**证据与边界：**SGLANG_OPT_DSV4_INDEXER_TP_SHARD默认off，只有eager调用允许；attnCP必须1、attnTP>1、≥512row，graph breaks/非eager不引入新collective。不能称wq_b权重本身TP分片或所有decode也分片。 **建议：**先检查每rank输入相同、AG参与一致与选中indices逐位对照，再测长context是否抵消AG固定成本。
源码路线：`deepseek_v4_backend.py::_indexer_row_shard/_make_low_ratio_prefill_indexer_inputs；v41_indexer/types.py::RowShard；full_topk.py/sharded_dense_prefill_topk`。
来源：[sgl-project/sglang #42968](https://github.com/sgl-project/sglang/pull/42968)（main，2026-11-02T14:24:47Z，`e33cd0c3a10474c7a0f6efeb69ce2cf4b7a55021`）。

### SM100 TRTLLM-MHA full-attention decode 可显式增大 KV split
SGLANG_TRTLLM_MHA_FULL_ATTN_KV_SPLITS默认0，按最小batch12启用，对fixed-q-len decode/verify的full-attention按workspace/counter容量限split，暂时scoped覆盖FlashInfer读取的SM数，异常也恢复。

**证据与边界：**不改SWA、ragged verify、SM90/SM120 XQA；改变归约顺序非bitwise。Muse4GB300实验吞吐统计含input+output，+38.5%不能写成output tok/s；该参数与seq-len grouping同时使用未测。 **建议：**按实际长短KV分布调split并验证计数buffer容量，先关参数作对照，不用总token throughput代替用户decode速度。
源码路线：`trtllm_mha_backend.py::_full_attn_kv_split_sm_count/_trtllm_decode_sm_count；environ.py::SGLANG_TRTLLM_MHA_FULL_ATTN_KV_SPLITS`。
来源：[sgl-project/sglang #46082](https://github.com/sgl-project/sglang/pull/46082)（main，2026-11-02T13:53:03Z，`ad6ca42dea36ffad53fa033dab400f40ffa677f7`）。

### 多个 MoE 模型保留 router FP32 bias/logits 至 top-k
MiMo-V2/Ernie4/Kimi-Linear的correction bias明确FP32；Bailing/Bailing-Linear返回router weight dtype的GEMM结果，不再立刻降回hidden dtype。

**证据与边界：**仅五个模型文件；LLaDA2已另修不在本项。作者格式/bytecode验证，无新GPU数值或精度结果，不能写为已测任务提升。 **建议：**检查checkpoint bias载入后真实dtype和top-k输入，低精度router回归重点用近tie样例。
源码路线：`models/bailing_moe.py::gate.forward；bailing_moe_linear.py；ernie4.py/kimi_linear.py/mimo_v2.py::e_score_correction_bias`。
来源：[sgl-project/sglang #38719](https://github.com/sgl-project/sglang/pull/38719)（main，2026-11-02T13:34:22Z，`c35f89fc9a2643c531127ce7f71bca2e27c4eb01`）。

### Kimi-K3 纳入 FlashInfer AR fusion 自动启用白名单
仅新增architecture白名单，符合既有SM100/TP/no-DP-attention/no-MoE-a2a条件时自动设置fusion backend=auto；用户先前已可手动同配置。

**证据与边界：**作者双GB300节点单次配对并发梯度和重复control有约±5%噪声，无固定accuracy评测；不能由零request error推导质量无变化，也不含试验中的strategy/completion mode调优。 **建议：**升级先确认既有eligibility与实际backend，跑自己的数值/精度及低并发ITL对照，别只沿用汇总百分比。
源码路线：`arg_groups/overrides.py::_FLASHINFER_ALLREDUCE_FUSION_ARCHS/_flashinfer_allreduce_fusion_auto_enable`。
来源：[sgl-project/sglang #47306](https://github.com/sgl-project/sglang/pull/47306)（main，2026-11-02T12:57:10Z，`755783ebe157faf33db8f6105a21f4de7df3eb9a`）。

### 无 decode 时跳过 prefill-decode interval 空转
running batch为空或prefill-only且无需DP MLP同步时清零interval，长prompt chunk连续入场；有decode仍遵循原节拍。

**证据与边界：**DP-attention保持同步cadence，因为别rank可能decode。B200 TP4/interval96/122880token空闲冷prompt TTFT2.546→1.187s属于该设置，默认未开interval的部署无此收益。 **建议：**按GPU trace确认空隙减少，并同时测持续decode并发下ITL；不能以单rank空闲状态跳过DP同步。
源码路线：`Scheduler._should_defer_prefill；test_scheduler_prefill_decode_interval.py`。
来源：[sgl-project/sglang #46654](https://github.com/sgl-project/sglang/pull/46654)（main，2026-11-02T12:38:00Z，`b55350bb93b5a4079749e370603f8b34863a9d64`）。

### Intel XPU Gemma4 图像 token 使用双向 attention mask
Gemma4复用Gemma3的per-token image-block IDs传给XPU FMHA，避免非Triton/FA4 backend一律退causal导致视觉质量差；补prefill metadata与fixture。

**证据与边界：**sgl-kernel-xpu mask尚无sliding-window变体，prompt越1024窗口时sliding层仍失去图像双向性；max_soft_tokens560与1120不能等同。无充分最终模型A/B数值，不将示例低分视作修后完整恢复证明。 **建议：**分别回归窗口内外图像prompt及纯文本，核对实际block IDs而非只有模型能加载。
源码路线：`gemma4_mm.py::prepare_attn_masks；xpu_backend.py；test_xpu_gemma_bidirectional_attn.py`。
来源：[sgl-project/sglang #46816](https://github.com/sgl-project/sglang/pull/46816)（main，2026-11-02T07:18:11Z，`cc69762cb73cd3fdb4178c33dda71603e4cc57d7`）。

### DSpark draft FA3/FA4 的静态 query 宽度跟随 runner
FlashAttentionBackend用model_runner.decode_num_tokens_per_req替代target_verify通用解析，sample_from_anchor=False时保留γ+1，修复batch2应[0,8,16]却[0,7,14]的offset。

**证据与边界：**最终是metadata/unit层证据；Fun-CosyVoice3 H100吞吐/接受率数字来自较早downstream复现，FA4没有独立GPU benchmark，不能报为本head服务提速。 **建议：**draft/target与anchor开关分开检查，graph/eager同时比cu_seqlens和flatten query长度。
源码路线：`flashattention_backend.py::__init__；test_flashattention_graph_metadata.py`。
来源：[sgl-project/sglang #46478](https://github.com/sgl-project/sglang/pull/46478)（main，2026-11-02T06:38:01Z，`23c6442cde7fb38230134ba0e3513ad2332aa5cf`）。

### XPU 增加显式启用的 full prefill graph 路径
full backend捕获/replay新增prefill metadata与padding、稳定output buffer和graph-pool scope；XPU prefill默认仍disabled，显式配置允许full/breakable。

**证据与边界：**模型与metadata新增较多XPU测试，但正文未给最终硬件结果/性能口径；Torch Memory Saver可省weights/KV，graph自身allocation仍不能pause。不是所有XPU模型默认graph加速。 **建议：**显式开prefill graph，检查capture shape/padding、输出buffer别名与memory-saver恢复，先同输入eager对照。
源码路线：`xpu_full_graph_backend.py::FullXPUGraphBackend；platform_hook.py::handle_xpu_backends；xpu_backend.py；test_xpu_graph.py/test_xpu_graph_metadata.py`。
来源：[sgl-project/sglang #30980](https://github.com/sgl-project/sglang/pull/30980)（main，2026-11-02T06:26:09Z，`39a5d5b135b12f93185f22dc4e70182f3ec0ee83`）。

### OOT 可注册精确 class/platform 的 linear-attention backend factory
attention registry在构造GDN/KDA/LinearAttnModelSpec backend前查平台key+完整类名，provider可override dispatcher builder并复用cache/metadata初始化。

**证据与边界：**只在OOT平台且exact class匹配时优先，未注册subclass不自动继承；不改kernel数值。作者50项registry测试及指定OOT集成，不是所有插件GPU数值均已重跑。 **建议：**插件首次构造前注册factory，回归fallback与目标类身份，单独认证state/padding/graph合同。
源码路线：`attention_registry.py::register_oot_linear_attention_backend/_create_linear_attention_backend；GDNAttnBackend._create_kernel_dispatcher；KDAAttnBackend._create_kernel_dispatcher`。
来源：[sgl-project/sglang #47234](https://github.com/sgl-project/sglang/pull/47234)（main，2026-11-02T05:33:36Z，`8fd34681caf4afe2ad3b92dbdfba32c587156546`）。

### 非二次幂 MLA head 数也用既有 masked concat
去掉num_local_heads必须power-of-two的多余guard，12/24head复用已处理padding的Triton concat，仍要求nope/rope特征维power-of-two并保留128head特殊路由。

**证据与边界：**FA3+FP8输出沿既有dtype规则；作者B30014项、H100全108项源码回归，44–74%是isolated concat，不是Kimi-K3服务提速。 **建议：**回归strided input、空/64Ktoken、graph changing-input与FP8 raw bytes，再测整model影响。
源码路线：`forward_mha.py::_concat_and_cast_mha_k；test_concat_mla.py model_mha_k_concat`。
来源：[sgl-project/sglang #47743](https://github.com/sgl-project/sglang/pull/47743)（main，2026-11-02T05:27:46Z，`2dc5bdd18b040d9ec0447743921edcbf7ba252f9`）。

### NVIDIA megakernel plugin 新增 hybrid preparation 与 DSpark typed view
资源准备后、autotune/graph capture前调用可幂等prepare；向插件提供MLA/KDA池与step view、request-local verify state、aux hidden及graph-static所有权检查，acceptance/state commit/draft执行仍SGLang拥有。

**证据与边界：**这是框架接口，companion plugin另仓；旧pair完整模型证据不能当rebased pair GPU认证。历史numerical canary明确FAIL，q6/GB300 TP8/DCPoff合同不等于任意spec/长上下文；默认kernel不变。 **建议：**框架与插件成对锁SHA/APIv2，先做owner/layout fail-closed测试；质量指标接近不能替代numerical canary。
源码路线：`kernel_plugins/nvidia/binding.py::PreparationView/prepare_megakernel/KdaStepView；plugin_kernels/nvidia/spec.py::MegakernelSpec.prepare；base_runner.py::_warmup`。
来源：[sgl-project/sglang #45406](https://github.com/sgl-project/sglang/pull/45406)（main，2026-11-02T05:12:49Z，`aa3e34aaebf880d72482cea74dbbfef54da728e2`）。

### SM90 实验 H16 CuTe sparse prefill 避免补至64 heads
显式dsv4-prefill-backend=cutedsl_h16为16本地query heads返回紧凑[TQ,16,512]，用WGMMA与online softmax，combined TOPK按128对齐并cache索引表示。

**证据与边界：**限CUDA SM90、BF16 query/workspace、D=512、TOPK≤16384；decode/spec/TBO/CP/bounded-tail/graph保留既有路径，auto不变。作者H20-3e TP8小矩阵E2E是input+output tok/s，不是单纯decode；旧fewshot准确率含示例，held-out须排除。 **建议：**作为明确opt-in先验dispatch marker、配置bucket及held-out质量，再按实际长prefill比较；不把所有Hopper sparse attention都切换。
源码路线：`cute_sparse_mla_h16.py::cute_sparse_mla_h16_fwd；deepseek_v4_backend.py；test_dsv4_cutedsl_h16_dispatch.py`。
来源：[sgl-project/sglang #36059](https://github.com/sgl-project/sglang/pull/36059)（main，2026-11-02T05:06:49Z，`7605a94d8bdb82780763b65dc650e782f3544dc2`）。

### AITER legacy target-verify 读取每层校准 KV scales
FP8 KV且layer有scale时给extend_attention_fwd传k_scale_float/v_scale_float，替代硬编码1；BF16仍1，修复NGRAM等legacy verify读回未反量化prefix。

**证据与边界：**只有走legacy non-MLA路径才变；实测NGRAM，EAGLE适用是代码推断。MI355X旧arm小pool绕过另一个int32溢出，因此服务数字不能当只改scale的完全等价速度对照；本质先修质量。 **建议：**用非1校准scale回归cached-prefix attention，分开unified verify与legacy，先验输出正确再看accept length。
源码路线：`aiter_backend.py::forward_extend legacy target_verify；test_aiter_fp8_kv_scales_paths.py::test_extend_attention_target_verify`。
来源：[sgl-project/sglang #47030](https://github.com/sgl-project/sglang/pull/47030)（main，2026-11-02T04:50:49Z，`ae66493995ec3fc61b7737ba8a4c46a8cc53cd4d`）。

### AITER plain-decode graph 按 host seq_len 选择 partition bucket
新增SeqLenBucketGraphVariants，在已有host seq_lens上选p16/p128/pmax，只有比max明显更小的bucket才capture；eager同表，workspace仍按max，避免短请求发射整context的空CTA。

**证据与边界：**默认SGLANG_AITER_DECODE_PARTITION_BUCKETS=true；仅paged_attention_ragged plain1token，spec、缺CPU长度及sync-free路径不变。额外capture variants有图内存/启动成本；必须partitions×256覆盖最大seq_len，否则静默漏KV。 **建议：**按4096/32768及fallback边界验输出与graph选择，统计extra graph容量成本，别在设备端读length造成同步。
源码路线：`graph_variants.py::SeqLenBucketGraphVariants；aiter_backend.py::ForwardMetadata.decode_num_partitions；test_aiter_decode_partition_buckets.py`。
来源：[sgl-project/sglang #47042](https://github.com/sgl-project/sglang/pull/47042)（main，2026-11-02T04:38:20Z，`46e95bb9f98c7e7740acf756261f46c2d72d31d4`）。

### Quark MXFP4 gfx95/gfx1250 不再依赖全局 AITER env 导入 shuffle
该scheme只能用AITER runner，故目标架构无论SGLANG_USE_AITER值都绑定所需helper、对应shuffle布局和padding，修复env未设时NameError且避免只import不shuffle的慢fallback。

**证据与边界：**其他HIP架构仍遵循既有env规则；gfx1250主要import/source合同，MI355X指定模型测试。对照env开关还会改变attention/top-k，200题0.015差异不能全部归MoE。 **建议：**在env unset与set两种启动下检查实际packed layout及backend，勿把此改动理解为不需要AITER依赖。
源码路线：`quark_w4a4_mxfp4_moe.py::_use_aiter/_is_shuffle_moe_mxfp4；TestQuarkMxfp4MoEShuffleImports`。
来源：[sgl-project/sglang #45332](https://github.com/sgl-project/sglang/pull/45332)（main，2026-11-02T04:20:12Z，`e0c7992d0d48c8085655752de7a9f43bf1862bbc`）。

### 单请求 n>1 保留全部多模态项
parallel sampling把单请求变batch时，同步为flat image/video/audio及hash列表加request维，已nested/空/单非list保持原路，避免图像400或视频/音频只读第一项。

**证据与边界：**仅request normalization，无模型精度/吞吐新主张；作者CPU相关套件仍有同基线环境失败，不能称全项目CI通过。 **建议：**覆盖n1/n2、flat/nested、多媒体混合与hash对齐，别仅用单张图片smoke。
源码路线：`io_struct.py::_handle_parallel_sampling；test_parallel_sampling_keeps_every_media_item_of_a_single_request`。
来源：[sgl-project/sglang #41768](https://github.com/sgl-project/sglang/pull/41768)（main，2026-11-02T04:14:02Z，`a58735064eeb6b91b219f3c7a6bfaf8915e17b9e`）。

### fused-A GEMM eligibility 跟随实际 backend 的 K alignment
统一auto resolution，JIT按256、CuTe按1024检查K；DeepSeek两处检查传入后续真实backend，SM120 K3584/2560改用普通linear而非入kernel后assert。

**证据与边界：**SM120并无可用JIT fused fallback，kernel自身assert仍保留；RTX5090新增4项，非全部模型性能认证。满足原合同的shape路由不变。 **建议：**把selection与kernel contract放同一resolver，回归auto/显式jit/cutedsl以及不对齐K，别在kernel内部静默切。
源码路线：`fused_a_gemm.py::_resolve_backend/fused_a_gemm_weight_eligible；DeepseekV2Attention.prepare_qkv_latent/q_b_proj_forward`。
来源：[sgl-project/sglang #47712](https://github.com/sgl-project/sglang/pull/47712)（main，2026-11-02T03:47:03Z，`46db5f7dccb02bf070a97a14eee07db171f95413`）。

### Kimi-K3 fused routing 保留初始 expert placement 与 padding
precomputed logical top-k进入普通后处理完成logical→physical remap/padding/recorder，支持初始permutation且fake dispatch回普通router；目标与draft metadata独立。

**证据与边界：**明确拒绝online EPLB和redundant experts，权重布局只够每逻辑expert一slot。B300 unit/op测试支持映射合同，完整模型输出未重跑，早期工作负载特定placement收益不归此PR。 **建议：**非identity初始map与padded tokens做路由/权重/latent对照，别把静态placement当在线重平衡。
源码路线：`topk.py::can_postprocess_precomputed_topk/build_precomputed_topk_output；kimi_k3.py::_routing_contract_ok/_ep_front`。
来源：[sgl-project/sglang #47742](https://github.com/sgl-project/sglang/pull/47742)（main，2026-11-02T02:17:21Z，`cec4554871f67c3252c5f2d6347a4192c510d9bb`）。

### Humming MoE 每 sublayer transform 后回收替换旧块
两种加载路径在sublayer transform函数退出、旧引用已释放后empty_cache，避免后续分配切碎旧cached blocks使大权重加载后显存长期占用。forward kernel/padding保持。

**证据与边界：**作者H20-3e DSv4.1 TP4加载100.6→约82GB并恢复启动，TP8约57.1→50GB；不是参数压缩或服务kernel提速。empty_cache只在加载transform阶段，不建议搬进每token循环。 **建议：**测模型加载完成后的allocated/reserved与KV容量，确认loop变量/temporary不再持旧tensor，再做整体startup时间对照。
源码路线：`humming.py::process_weights_after_loading；humming_utils.py::prepare_humming_moe_layer；test_humming_moe_release_replaced.py`。
来源：[sgl-project/sglang #45200](https://github.com/sgl-project/sglang/pull/45200)（main，2026-11-02T01:54:22Z，`61d2005421959b8b5d151e20302807e4efd93ddd`）。

### Ascend Kimi-K3 no-RoPE MLA 使用 MLAProlog V4
无rotary_emb时选sgl_kernel_npu mlaprolog_v4并传sin/cos=None，复用BF16/MXFP8权重准备；修v_head_dim实参和tuple结果消费，ModelSlim Q/G精度不同会选split路由。

**证据与边界：**950PR TP16、W4A8/BF16 KV的27.4vs61.3µs是局部preprocess profiling；不能写端到端2.2×。新增依赖kernel内部未本轮展开，正文没有完整最终unit矩阵。 **建议：**核对kernel依赖版本、no-RoPE分支与Q/G量化metadata，分别验BF16/MXFP8和输出维度。
源码路线：`npu/attention/mla_preprocess.py::NPUFusedMLAPreprocess.forward_mlaprolog/uses_mlaprolog；deepseek_v2_attention_mla_npu.py::forward_mla_prepare_npu`。
来源：[sgl-project/sglang #47373](https://github.com/sgl-project/sglang/pull/47373)（main，2026-11-02T01:48:03Z，`9fcb7a7b98004ce73ddb2ac0c4b458c174cbfb5c`）。

### 补录：Miles output store 按 CP 本地字节范围读入复用 buffer
OutputStore.read_ranges验证identity codec、field/source/destination范围及同store open buffer；ReadBuffer共同拥有mmap和registration，close先unregister，Mooncake按part分组一次get_into_ranges填caller memory，避免CP每rank读全field和逐次register。

**证据与边界：**仅sglang-miles分支，非main/正式SGLang发行。Mooncake0.3.13.post1双H200/400Gb IB成功路径实测；错误码来自依赖source合同及fake测试，native registered test未本地跑。失败可部分写入，buffer仍注册可复用，单buffer只允许一次read。 **建议：**训练端明确管理ReadBuffer.close及view寿命，先校验CP zigzag rows字节等价，再把一次性registration与每轮I/O分开计时。
源码路线：`radix_transfer_plugins/store.py::OutputStore.read_ranges/ReadBuffer/_check_range；backends/mooncake.py::get_ranges/_raise_failed_ranges`。
来源：[sgl-project/sglang #47320](https://github.com/sgl-project/sglang/pull/47320)（sglang-miles，2026-10-31T01:57:32Z，`29b34bc6f77dc879a07967b682848f7781de2bee`；补录）。

### 继续追踪与覆盖边界
vLLM [#61667](https://github.com/vllm-project/vllm/pull/61667) 与 [#61532](https://github.com/vllm-project/vllm/issues/61532) 本轮重新实开，仍open；consumer quarantine head仍`4fd16ddb155fd0f7e676dece9abd6aa9209b2b1a`。本日producer/reset等拆分修复不关闭该风险。TorchTitan GDN [#5731](https://github.com/pytorch/torchtitan/pull/5731)（7/7）也仍open。
六仓回溯范围已完整分页枚举603个main提交、617项合入PR和258个release对象。THUDM/slime在此范围无main commit/merged PR，cutoff基线仍`540ea919e45b8b929198abce008a103be9cfc779`；这不等于旧backlog已审完。多数CI/文档按标题初筛，所有未详审实质候选与旧历史缺口继续排队；不宣称六仓语义穷尽。

<a id="extended"></a>
## 扩展开源工程与硬件软件协同

以下均为本轮重开的官方 PR、最终文件或真实默认分支提交；以每组列出的合入 SHA 为代码基线。测试和性能数字是提交者提供的证据，本轮未执行设备/集群复测。完整文件列表不代表逐行审完全部生成内核；大补丁的未审部分另留队列。

### Transformers：修正并行损失归一化与 FSDP2 加载内存回归

**huggingface/transformers；本轮发现，原始合入时间/版本：** [#50214](https://github.com/huggingface/transformers/pull/50214)（2026-11-02T02:38:57Z，`9f8b2169f1`，默认分支已见）；[#45649](https://github.com/huggingface/transformers/pull/45649)（2026-11-02T13:24:08Z，`5949d2c427`，默认分支已见）

有效 token 数除以 TP×CP×SP 时由整除改为真除，避免非整除 batch 的损失、梯度和 SGD 更新被放大。提交者在2×T4/FSDP2＋CP复现旧实现1.5倍偏差；这不是一般训练速度优化，建议用不能整除并行度的有效 token 数做回归。

FSDP2 非本地 rank0 的非量化加载不再为全部参数创建 CPU 零占位；**最终代码只给 `named_non_persistent_buffers()` 分配真实占位**，不是把所有 buffer 都保留在 CPU。旧回归证据来自 Qwen3-30B-A3B、2节点×8H100、DP16、SDPA、5steps、cpu_ram_efficient_loading：每节点参数占位约60→480GB。该表用于定位旧回归，不能当修复后的新 MFU 基准。建议大模型加载先验证主机峰值及完整 state-dict 覆盖。

### Transformers KV 共享正式接口：按最终名称读取，避免依赖 PR 早期设计

**huggingface/transformers；本轮发现，原始合入时间/版本：** [#47290](https://github.com/huggingface/transformers/pull/47290)（2026-11-02T18:40:41Z，`cc1451df3b`，默认分支已见）

最终交付是 **`KVSharingDict`、`kv_sharing_modes` 的 `store`/`reuse`/None，以及 `DummyLayer`**；PR正文中的 KVCacheSharingStorage/kv_sharing_roles 是早期命名，不能当当前 API。共享容器继承 UserDict，绕过 FSDP2 对普通 dict 的改写；原地 `.to()` 让 Accelerate 跨设备迁移后复用。消费层仍占逻辑层位、没有独立 KV 存储，补齐中间层共享、mask、静态 cache 与导出适配；Gemma系列接入。

它建立接口合同，不证明任何已有 checkpoint 自动获得新的共享比例或长上下文收益。建议自定义 CLA/YOCO 模型先测 producer/consumer 映射、跨设备迁移、prefill/decode与导出一致性，再评估显存。

### Transformers / Trainium：Neuron 编译显式使用静态 shape

**huggingface/transformers；本轮发现，原始合入时间/版本：** [#49617](https://github.com/huggingface/transformers/pull/49617)（2026-11-02T21:43:01Z，`26754696fd`，默认分支已见）

`torch_compile_backend="neuron"` 时向 Accelerate TorchDynamoPlugin 传 `dynamic=False`，其他 backend 保持原解析。它处理自动动态 shape 触发 NCC_EMOD023 或第二个 shape 的 SymIntArrayRef 错误；提交者给 Trainium2 上 tiny Qwen3、6步SFT的前后对照。建议 Trainium SFT 先复测实际长度桶、重编译次数与训练损失，不能把6步可运行当吞吐提升。

### TRL：DeepSeek R1/V3 的 assistant-only loss 获得训练模板

**huggingface/trl；本轮发现，原始合入时间/版本：** [#7963](https://github.com/huggingface/trl/pull/7963)（2026-11-02T07:54:14Z，`ffa3e9868f`，默认分支已见）；[#7978](https://github.com/huggingface/trl/pull/7978)（2026-11-02T07:54:15Z，`3a3b6c1464`，默认分支已见）

为 R1-0528、V3、V3-0324 保存参考模板并映射到训练版本，给 assistant 输出添加 generation 标记、tool 参数用 JSON 序列化；R1训练目标保留 reasoning，工具结果后的前缀不计入 assistant mask。最终合入 SHA 均在本轮 main 提交中出现，即使V3 PR曾以栈分支为base，也不能误写为仍未到main。

这是 SFT 模板与掩码支持，不是新模型训练结果。建议复查真实多轮工具数据的渲染文本、reasoning保留规则与逐 token mask；仅升级包不能替代数据合同验收。

### DeepSpeed ZeRO-1/2：请求边界与 optimizer step 分离

**deepspeedai/DeepSpeed；本轮发现，原始合入时间/版本：** [#8949](https://github.com/deepspeedai/DeepSpeed/pull/8949)（2026-11-02T19:45:14Z，`dc046b15d7`，默认分支已见）

调用者可多次设置 accumulation boundary，却延后 `engine.step()`。旧实现会在边界丢掉早先请求梯度，CPU offload还可能混入上次更新的主机累加值。最终实现按参数记录当前更新窗口参与状态、保留非 offload 累加直到step；FP32 master首次边界只保留已有分区，第二次backward再恢复局部分片到辅助累加器，低精度master用共享buffer，并在成功/overflow step路径清理。

新增两rank、ZeRO1/2、offload与精度组合回归；单backward不额外分配不代表多backward更快。建议异步RL、多请求累加、reentrant checkpoint与overflow恢复优先对齐参考优化器，和此前#8749的精度修复分别验收。

### DeepSpeed fused LM-head＋CE 接入 HF forward：大词表显存收益有兼容边界

**deepspeedai/DeepSpeed；本轮发现，原始合入时间/版本：** [#8945](https://github.com/deepspeedai/DeepSpeed/pull/8945)（2026-11-02T12:07:43Z，`1a956270f4`，默认分支已见）

`vocab_parallel_ce_backend="fused_linear"` 将 LM-head 输入交给融合loss，跳过完整 logits；ignore_index 行在投影前压缩，默认chunk至少256行，减少窄GEMM和每chunk两次all-reduce。默认仍为torch。显式拒绝ZeRO3、DeepCompile autotp、head后softcap/scale；自行从logits/weight计算loss的label smoothing与TRL相关fused路径不兼容。

作者 Qwen2.5-0.5B、V151936、TP2、ZeRO1、BF16、checkpoint、microbatch4×2048、50%prompt mask，HF Trainer/Transformers5.17/Accelerate1.15，20步取6–20均值：2×L20上torch→fused每rank峰值13201→5200MiB、514→467ms；Liger为8456MiB/477ms。2×ArcProB70上torch653→590ms。不是大模型/全算法普遍收益；前三步grad norm接近不等于长期训练等价，新增BF16多token测试的梯度相对norm容差可到1%，融合FP32 logits与先BF16物化路径也可不同；训练labels路径返回末维为0的logits占位，eval/no-label才fallback。**建议考虑复现大词表SFT，先审调用者loss合同与收敛。**

### FlashInfer batch-invariant GEMM：以显式数值合同换取部分性能

**flashinfer-ai/flashinfer；本轮发现，原始合入时间/版本：** [#8361](https://github.com/flashinfer-ai/flashinfer/pull/8361)（2026-11-02T22:30:50Z，`ae39460de2`，默认分支已见）

新增 `mm_bf16(batch_invariant=True)`，默认False；固定规范K累加流，使同一行不随batch大小、邻居、位置、SM数或允许的配置变化。True还要求backend=auto且无bias；API只接收同设备二维BF16、K/N为8倍数、规定权重布局，支持SM100/103/107；不支持的输入抛 BatchInvariantUnsupportedError，**不静默fallback**。Lean证明覆盖 canonical flow 的R1–R4，不证明生成CUDA kernel；工具链含Lean.ofReduceBool，Lean检查未接入CI，fingerprint含版本/架构，不承诺跨架构或整模型训推logits逐位一致。

22种M×12组LLM权重形状，每layout264cells；对每格较快的非不变cuDNN/CuTe基线，autotune后Rubin NVL8 k/n布局总体几何均值0.963/1.000×、decode0.794/0.842×；B200总体0.866/0.894×。收益是可控一致性，存在明显decode代价。建议RL训推一致性实验逐算子接入、补模型级对照，并记录依赖版本、布局与配置。

### FlashInfer LM-head logprob：温度参与梯度，但本身不可微

**flashinfer-ai/flashinfer；本轮发现，原始合入时间/版本：** [#8254](https://github.com/flashinfer-ai/flashinfer/pull/8254)（2026-11-02T11:42:02Z，`bca46b9b01`，默认分支已见）

实验性确定性selected-token logprob新增正有限标量或FP32[T]逐token温度；对logit梯度带1/温度，温度tensor requires_grad=True被拒绝，keep_logits模式不接受温度。**默认非法tensor温度会触发device assertion并结束CUDA context；`check_temperature=True`才同步读回并报ValueError**，忽略行也须合法。

BF16 X/W、H6144/V154880，前向＋反向对同温度PyTorch eager：作者B200/CUDA13.4.92所测T2048–32463约1.38–1.54×，GB300约1.46–1.64×。B200 T65031的eager OOM，而Cake时间仍pending，不据此给大长度倍率。建议RL logprob训练复现温度/标签/梯度与错误隔离，算子收益不能代替端到端吞吐。

### FlashInfer KDA：raw forget gate 融入确定性单卡和KCP训练

**flashinfer-ai/flashinfer；本轮发现，原始合入时间/版本：** [#8276](https://github.com/flashinfer-ai/flashinfer/pull/8276)（2026-11-02T07:13:03Z，`5729f0174b`，默认分支已见）；[#8255](https://github.com/flashinfer-ai/flashinfer/pull/8255)（2026-11-02T10:41:19Z，`c31ecb5321`，默认分支已见）

显式 `use_gate_in_kernel=True` 时接受BF16 raw gate及FP32 A_log/dt_bias，在kernel内激活并固定顺序归约参数梯度，避免FP32整gate/dgate中间量。单卡与context-parallel分开交付，KCP跨rank参数梯度仍用all-gather＋rank-order累加；raw＋lean arena目前拒绝。

GB200/CUDA13.0/PyTorch2.14.1，KCP W4/H64、每rank16384tokens/3文档、最慢rank前后向中位：7.138ms，对外部torch门9.456ms、FLA gate helper7.233ms；峰值输入之外1354/3112/2088MiB。故相对torch1.325×，相对FLA helper仅1.013×，不能混用分母。作者B200/GB200/GB300 W1/2/4/8及R200 W1/2/4数值行通过；R200 W8未跑。建议线性注意力训练复现真实gate配方、checkpoint及跨rank梯度，勿外推整模加速。

### FlashInfer KDA 非确定性KCP：checkpoint保留tape，少做重放

**flashinfer-ai/flashinfer；本轮发现，原始合入时间/版本：** [#8366](https://github.com/flashinfer-ai/flashinfer/pull/8366)（2026-11-02T20:27:52Z，`f9738265cd`，默认分支已见）

非确定性KCP在activation checkpoint下交接forward tape，backward跳过intra/WY/K1重放及多分区offset/carry工作。`checkpoint_handoff="auto"`只在已验架构、head-token门槛内启用，小而host-bound的CP8/H4/H8、CP4/H4等保持关闭。on/off之间输出/梯度逐位相同不代表与确定性实现逐位相同。

作者BF16 q/k/v/beta、FP32 gate、D128、261392tokens/4文档、CP2/4/8、contiguous/zigzag，无short convolution；B200/B300/GB300/R200最慢rankCUDA events比较，checkpointed step相对同head确定性Cake1.038–1.164×。相对FLA0.6的1.815–9.732×是另一个分母且只限支持layout。建议先定位重放占比，再测整训显存与数值模式代价。

### FlashInfer FP8 GEMM＋all-reduce：TP8融合也可能更慢

**flashinfer-ai/flashinfer；本轮发现，原始合入时间/版本：** [#8379](https://github.com/flashinfer-ai/flashinfer/pull/8379)（2026-11-02T21:51:51Z，`f5d2f294e9`，默认分支已见）

SM100单节点显式Cake backend新增TP2/4/8 rowwise E4M3 GEMM＋all-reduce，输出BF16；TP2/4为NVLink pull/reduce，TP8为NVLS。B200/CUDA13.0、48个TP4几何、交替会话CUPTI最慢rank中位，相对旧Cake几何均值1.112×，范围0.927–1.303×；TP2/8对 `torch._scaled_mm+all_reduce` 则1.029/0.925×，TP8多数shape更慢。

建议只在实际TP/几何分布复现，不能把“单kernel”当必然更快，也不能把TP4对旧Cake结果当TP8对PyTorch结果。没有本轮整模推理收益证明。

### FlashInfer FMHA：页表对齐、NaN污染与warp同步修正

**flashinfer-ai/flashinfer；本轮发现，原始合入时间/版本：** [#8406](https://github.com/flashinfer-ai/flashinfer/pull/8406)（2026-11-03T00:23:58Z，`e4f5a75fcc`，默认分支已见）；[#8407](https://github.com/flashinfer-ai/flashinfer/pull/8407)（2026-11-03T00:14:21Z，`eabf03fe55`，默认分支已见）；[#8408](https://github.com/flashinfer-ai/flashinfer/pull/8408)（2026-11-03T00:06:33Z，`a72f407d20`，默认分支已见）

BF16 decode为奇宽/未对齐页表使用workspace尾部padding副本，retained-KV B256恢复persistent一维grid；移除错误且非确定性的精确B128成员，转用通用/DEC_B128。correction barrier计数从每warp代表扩到全部128线程，防下一tile写覆盖；NVFP4 context在page_empty等待后补warp同步。

另一修复把超出每请求visible=min(q_pos+1,kv_len)的V页指向TMA零填充，避免0×NaN污染，并让kv_len0输出零；K仍由softmax mask保护。作者在B200/GB300给前后失败用例、mem/race/sync sanitizer结果，#8407单独12个坏页用例从失败到通过；不同PR的“完整suite”计数来自不同head，不相加。建议先做垃圾页/NaN页/零长度/odd-width/graph与retained-KV回归，性能表次于正确性。

### FlashInfer：DSA verdict发布、MNNVL集群屏障、FP16舍入与大地址

**flashinfer-ai/flashinfer；本轮发现，原始合入时间/版本：** [#8355](https://github.com/flashinfer-ai/flashinfer/pull/8355)（2026-11-02T18:30:26Z，`7c395b6e81`，默认分支已见）；[#7782](https://github.com/flashinfer-ai/flashinfer/pull/7782)（2026-11-02T18:58:30Z，`688d518a8a`，默认分支已见）；[#6114](https://github.com/flashinfer-ai/flashinfer/pull/6114)（2026-11-02T19:53:08Z，`0b6e0db242`，默认分支已见）；[#8334](https://github.com/flashinfer-ai/flashinfer/pull/8334)（2026-11-02T21:56:12Z，`e8143622b1`，默认分支已见）

DSA确定性top-k为共享verdict增加两个reader的ACK，Q-loader等ACK后才发布下一unit；这是潜在覆盖/phase alias竞态修复，作者未实际观察到错输出或hang，不能写成已复现事故。MNNVL RMSNorm让cluster中每线程都等待barrier，再访问peer CTA共享内存；不只让rank0的32线程等待。

Mamba软件随机舍入恢复FP16 subnormal区间，并按原输入分类Inf/NaN，避免噪声把溢出变NaN；TinyGEMM2输出线性索引在乘法前提升int64，覆盖超过2^31元素的输出。建议多CTA通信、长序列/大词表以及低幅状态更新用户优先回归。这些是分立的正确性条件，没有整训倍率承诺。

### FlashAttention：测得的CLC单元成为默认，显式autotune独立运行

**Dao-AILab/flash-attention；本轮发现，原始合入时间/版本：** [#3062](https://github.com/Dao-AILab/flash-attention/pull/3062)（2026-11-02T05:13:24Z，`6d6392ddc6`，默认分支已见）；[#2743](https://github.com/Dao-AILab/flash-attention/pull/2743)（2026-11-02T05:13:32Z，`54b044c8cf`，默认分支已见）

dense causal forward只在经验表允许的dtype/head/架构单元走CLC，新增L2-swizzled LPT次序与late claim。排除varlen/local/SplitKV/paged/sparsity/score-mask modifier/qv，越CUDA y/z grid上限则解析回退，显式非法配置报错；generic FA_CLC环境变量不再决定此路径。B200 hd256至少32个batch/head slices才默认开，24shape约1.044×，小切片测有0.83×退步，故门控有意义。

`tune_fwd_config`仅在调用者显式请求时对子进程预编译、参考输出校验、CUDA Graph轮转计时；正常forward不运行tuner、不持久化winner。仅eager BF16/FP16 output-only，要求提前开磁盘cache，拒绝autograd、compile/capture和多种扩展；不接管Inductor/Flex自有调优。#2743虽base为stack，其merge SHA已在本轮main。建议固定真实stride/shape跑，不能把GB300某shape峰值当全attention/整模提升。

### Transformer Engine：实验性分组BatchedLinear与capturable Adam访存

**NVIDIA/TransformerEngine；本轮发现，原始合入时间/版本：** [#3160](https://github.com/NVIDIA/TransformerEngine/pull/3160)（2026-11-02T23:26:24Z，`b021980343`，默认分支已见）；[#3782](https://github.com/NVIDIA/TransformerEngine/pull/3782)（2026-11-02T18:06:50Z，`5c3f067c28`，默认分支已见）

新增cuBLASLt strided-batched GEMM与 `ops.BatchedLinear`，允许[S,B,G,D]×[G,R,D]等交错布局，不必先把group搬到最外。高精度输入配对或MXFP8配对、输出高精度，支持前后向/primary MXFP8权重与main-grad累加；不支持混合两种输入、输出量化、融合epilogue、非MXFP8 recipe或TP。测试代码存在不等于本轮独立执行。

capturable Adam将lr/inv_scale读取提前，梯度读取完再写unscaled值，inv_scale1时省写；没有足够匹配工作负载的整训数值，本文不给倍率。建议按真实分组布局、recipe与图捕获测试，BatchedLinear不等于通用expert-parallel方案。

### TensorRT-LLM Helix：主干压力抢占与实验支线启用分开看

**NVIDIA/TensorRT-LLM；本轮发现，原始合入时间/版本：** [#21122](https://github.com/NVIDIA/TensorRT-LLM/pull/21122)（2026-11-02T16:56:45Z，`0eb0539d0f`，默认分支已见）；[#21467](https://github.com/NVIDIA/TensorRT-LLM/pull/21467)（2026-11-02T19:28:17Z，`b47aba05c3`，仅核实合入 `feat/helix-agg`）

main的aggregated Helix在KV压力下按rank-invariant输入一致选择recompute-preemption；prefill只抢后方请求以免互相抢占，重置owner/长度计数。当前仅GPU池，不接host/disk/KV connector/rebalance；可开全rank schedule比对，不能宣称已有host-tier恢复。

**启用完整aggregated Helix的#21467仅核实在feat/helix-agg支线。** 它用实际prefill峰值定KV容量、tokens_per_block×CP的ledger块、context-head workspace与跨rank容量协定，新增prefill warmup及MLA/dense/SM100+门控，拒绝FP4 KV。作者KimiK3 NVFP4、GB300 TP4×CP2的GSM8K BF16/FP8为96.21/96.59，启动检查未被自动单测覆盖；不是正式版的普遍可用或性能证明。建议继续观察支线集成，按压力/重算/owner一致性验证。

### TensorRT-LLM KV bounce：配置迁移扩大了已知失败路径的可达性

**NVIDIA/TensorRT-LLM；本轮发现，原始合入时间/版本：** [#20049](https://github.com/NVIDIA/TensorRT-LLM/pull/20049)（2026-11-02T02:44:46Z，`a9be7004c7`，默认分支已见）；[#21095](https://github.com/NVIDIA/TensorRT-LLM/pull/21095)（2026-11-03T00:03:55Z，`ff1aa0f7d3`，默认分支已见）

Python bounce删除；size>0选择C++ agent bounce，agent_bounce_buffer_enable弃用且忽略，旧最小块/字节环境变量不再读取；无libzmq支持时启动报错而非Python回退。纯Python NIXL与C++ transceiver的适用边界、FP4 MLA零bounce要求仍保留，P/D必须同release。

**#20049明示：失败/取消后的迟到C++ scatter可能写入已释放/复用KV页，TRTLLM-17105仍待后续处理。** 本次删除Python层不能写成已解决该安全完成边界；Qwen3.5原Python bounce配置迁到C++后新暴露于此路径。另#21095门控无peer NCCL scope、优化tensor发现及最终配置的graph默认值；最终diff只有5文件，正文提Python bounce stream的变更未在最终文件中，不能当本PR仍交付。45个scope测试是旧1.4.0rc2 wheel加Python overlay，current-main缺编译符号，8×B200性能未重跑。建议取消/失败/allocator复用专项验收后再采用。

### TensorRT-LLM：Ray NCCL EP初始化与显式Small-BH大状态安全

**NVIDIA/TensorRT-LLM；本轮发现，原始合入时间/版本：** [#21347](https://github.com/NVIDIA/TensorRT-LLM/pull/21347)（2026-11-02T22:22:09Z，`71b8f3dc74`，默认分支已见）；[#21342](https://github.com/NVIDIA/TensorRT-LLM/pull/21342)（2026-11-02T08:24:58Z，`bdcb577a93`，默认分支已见）

MPI-disabled/Ray通过torch EP process group取得NCCL communicator，避免每actor singleton MPI world；MPI worker仍用按PP拆的MPI子通信组。它修选择路径，不证明所有Ray通信拓扑已验。

Small-BH KDA把state row stride改64位runtime，处理超2^31元素偏移；测试跨约5.3GB状态池做row0/大offset逐位对照。当前auto/bt16自#20888起不选Small-BH，故本修复主要保护显式mode或ModelingV2 op。旧KimiK3 Rubin精度与IMA改善同时含路由改动，不能孤立归因stride。建议显式kernel调用者核stride/布局与vendored patch版本。

### TensorRT-LLM MiniMax-H3：按模态stage卸载，denoising期间保留主干

**NVIDIA/TensorRT-LLM；本轮发现，原始合入时间/版本：** [#21301](https://github.com/NVIDIA/TensorRT-LLM/pull/21301)（2026-11-02T23:06:55Z，`53529c014e`，默认分支已见）

复用pipeline offloader分别管理text encoder、transformer、video/audio VAE，CPU加载被卸载权重；conditioning/denoise/decode进入对应stage，整个denoising期间transformer驻GPU。作者BF16/VANILLA attention报生成峰值130.03→62.53GiB、LPIPS0与audio容差通过；硬件/分辨率/长度披露不足，且新增B200集成测试的存在不等于已提供执行回执。建议多模态生成部署复现峰值、host RAM与stage搬运时延，不外推LLM逐层offload收益。

### Dynamo GMS：每GPU进程、跨slab分配与peer恢复连成一条冷启动路径

**ai-dynamo/dynamo；本轮发现，原始合入时间/版本：** [#17339](https://github.com/ai-dynamo/dynamo/pull/17339)（2026-11-02T22:27:16Z，`0efff57755`，默认分支已见）；[#17344](https://github.com/ai-dynamo/dynamo/pull/17344)（2026-11-02T20:55:32Z，`8da75b398d`，默认分支已见）；[#17340](https://github.com/ai-dynamo/dynamo/pull/17340)（2026-11-02T23:45:36Z，`f9779a05ae`，默认分支已见）

V1 supervisor按PCI bus序为每GPU起进程，用GPU UUID命名socket；Linux Landlock限制可打开GPU节点以减少libcuda启动成本，缺支持则警告并退回较慢路径，**不是设备安全隔离保证**。恢复仍要求Snapshot保持bus位置/rank对应，不能只按引擎CUDA ordinal推断身份。

每domain保留连续VA、相邻2GiB slab可组成大allocation，统一free list处理碎片，恢复仍按旧地址映射；完全空闲slab可归还。作者GLM5.3 NVFP4 TP8/B200/SGLang0.5.21的0.85配置由57slab OOM变成KV池成功，但运行叠加了同系列GMS代码，不是孤立通用显存百分比。

MX backend按manifest顺序从同权重peer填充，逐allocation只核大小；同device bus位置须为同rank，`mx-source-key`不是完整内容验证。持read-only lease的发布pod不能checkpoint，退出先unpublish再释放NIXL/mapping。作者B200 TP8/vLLM与SGLang的两模型输出对照通过，GLM每GPU66.6GB约2.4–2.5s、212–219Gbps；忽略artifact shard digest不等于端到端身份校验。建议作为冷启动方案复现总ready/TTFT、拓扑和失败清理，别将网络传输时间当完整恢复时间。

### Dynamo路由：会话粘性减少重prefill，direct模式去掉全互订阅

**ai-dynamo/dynamo；本轮发现，原始合入时间/版本：** [#15600](https://github.com/ai-dynamo/dynamo/pull/15600)（2026-11-02T09:06:31Z，`c260334d70`，默认分支已见）；[#17237](https://github.com/ai-dynamo/dynamo/pull/17237)（2026-11-02T19:56:14Z，`820bffe701`，默认分支已见）

sticky-min-load第一次选active request最少的合格worker，之后按session固定，直到worker离开候选集；无session只最小负载，各prefill/decode pool独立绑定，空闲4h过期。作者两GB300、两个TP1 SGLang0.5.16、Qwen3-8B BF16、256slots×12turn、1K→16K输入/256输出、180s滚动负载：约7130/7090tok/s，与SMG min_load近持平；旧KV默认约低21–22%的对比来自各自run，不能写所有路由通用22%提升。

direct frontend不负责选worker，停止订阅每worker KV metrics并禁用本地busy threshold，POST /busy_threshold返回409；避免一worker一sidecar时O(W²)连接/内存。2048 mocker的OOM是定位证据，非本PR已完成2048真实模型服务验收。负载拒绝应由调用者/EPP处理，相关load shedding仍有独立待办。建议多轮会话与GAIE部署分别复现缓存率、重绑及拒绝行为。

### Dynamo live retune：实验栈分支，尚不能写成main在线自调参

**ai-dynamo/dynamo；本轮发现，原始合入时间/版本：** [#17562](https://github.com/ai-dynamo/dynamo/pull/17562)（2026-11-02T23:57:38Z，`08c5eb9661`，仅核实合入 `hzhou/dgdr-trace-pvc`）

已合入hzhou/dgdr-trace-pvc栈分支，未在本轮默认分支提交中找到此合入SHA。设计将每窗口frontend trace切入共享RWX claim，经DGDR v1beta2回放一组KV router参数，并在非advisory模式向所有frontend PATCH winner；需独占retune、固定replicas，关闭load/throughput scaling。默认local executor不变，明确面向development cluster的EXPERIMENTAL。

仅核查这层supervisor/lease/actuator及文档，依赖#17483/#17733–17736的完整部署资格没有本轮集群复验。建议继续观察栈到main、trace完整性、共同随机数比较及部分PATCH失败恢复；不把“merged”误写成已发布线上能力。

### Ascend：稀疏HCCL权重更新接入现有checkpoint坐标合同

**vllm-project/vllm-ascend；本轮发现，原始合入时间/版本：** [#18915](https://github.com/vllm-project/vllm-ascend/pull/18915)（2026-11-02T06:29:18Z，`a68511fad3`，默认分支已见）

`backend=sparse_hccl`、weight_nz_mode0复用上游patch验证/loader，HCCL组大小1+TP×DP，EP不额外加设备；转置expert提供临时checkpoint视图，更新同步后才恢复推理。仅未量化ND、静态expert placement/EPLB关；不支持PP、draft、NZ、MC2 fused expert lists或packed patch。

**原地非事务更新，失败后必须暂停generation等待恢复**；通信O(nnz)不等于加载O(nnz)，checkpoint应用仍需O(N)staging，超2^31元素还有上游int32索引边界。作者Ascend910/torch2.10给36UT及固定checkpoint端到端，不能替代最终生产拓扑；A5表只计send，50%/100% sparse的252.66/435.87ms反比dense191.41ms慢，不是所有稀疏率都加速。建议RL训推同步先测dense patch、错误中断、视图复原与恢复协议。

### Ascend：dense MLA的DCP Q复制、省通信有prefill代价

**vllm-project/vllm-ascend；本轮发现，原始合入时间/版本：** [#17489](https://github.com/vllm-project/vllm-ascend/pull/17489)（2026-11-02T06:36:31Z，`3ce023839d`，默认分支已见）

显式环境变量优先于dcp_q_replicate配置；仅PCP1/DCP>1、dense MLA，SFA与LoRA拒绝。组内复制Q投影、直接算完整group heads省decode all-gather，prefill再切local view，并处理W_UK。作者两A3用DeepSeekV2-Lite做eager/FULL_DECODE_ONLY；性能另为DeepSeekV3.1 W8A8、TP16/DCP16、并发8、32请求×3runs：1K/128与4K/128输出吞吐提升约5.5–8.7%，4K eager TTFT反而1917.71→1998.60ms。

不能用小模型正确性证明大模型全精度，也不能只报decode收益。建议长prefill与短decode分布加权复现，记录复制权重显存成本。

### Ascend投机解码：entropy会改分布，LiLiCorr仍缺成熟权重证据

**vllm-project/vllm-ascend；本轮发现，原始合入时间/版本：** [#19460](https://github.com/vllm-project/vllm-ascend/pull/19460)（2026-11-02T06:27:37Z，`bf492d1405`，默认分支已见）；[#17147](https://github.com/vllm-project/vllm-ascend/pull/17147)（2026-11-02T02:07:53Z，`6d2866b717`，默认分支已见）

MRV2 entropy verification默认关，启用时threshold0.95/alpha0.4，FP32 entropy与rejection融合，alpha0跳扫描；和upstream block verifier互斥。它**有意改变采样分布**，接受率上升不保证吞吐/精度提升，不能称lossless；作者20CPU/126NPU用例来自Ascend910_9392，未覆盖所有A2/A3/A5部署。

LiLiCorr在DFlash各位置top-k候选间用小两层transformer打转移分，再左到右greedy取lattice路径，target verify保持原协议。最终集成已合main，但PR明确使用非官方、只部分训练的权重，更新后性能表仍待补；不能从其他框架/论文移植倍率。建议前者仅做可接受分布偏差的实验，后者继续观察匹配权重及目标模型验收。

### Ascend CANN KeyPool：FP8 LI cache是功能支持，旧算子倍率另算

**vllm-project/vllm-ascend；本轮发现，原始合入时间/版本：** [#18447](https://github.com/vllm-project/vllm-ascend/pull/18447)（2026-11-02T06:47:04Z，`7afcf200dc`，默认分支已见）

Ascend950上 `attention_config.indexer_kv_dtype=fp8` 支持LI pooled-key与FP32 per-pool scale，BF16默认，SFA cache精度独立。A5/CANN9.2.1/TritonAscend3.2.2的指定head有16个FP8/BF16算子例通过；整个operator group仍7failed/2107passed/1315skipped，未同环境基线重跑，不能写全套通过或确认全为旧失败。

旧10-21 GSM8K1319题/8shot/MTP5/8192cap，BF16 96.89、FP8 97.19，只是单次历史比较，未复验当前head；另一个优化包的速度表不是本PR所用算子。建议先做值/scale/padded stride/graph与整模精度对照，再决定容量换精度。

### Ascend KV Pool：CPU元数据降复杂度，并补未分配请求ACK

**vllm-project/vllm-ascend；本轮发现，原始合入时间/版本：** [#20544](https://github.com/vllm-project/vllm-ascend/pull/20544)（2026-11-02T12:55:21Z，`eb5ddb90ec`，默认分支已见）；[#20213](https://github.com/vllm-project/vllm-ascend/pull/20213)（2026-11-02T08:16:02Z，`bf2adf6513`，默认分支已见）

unfinished ID过滤由重复列表扫描改expected O(U+N+C)，无unfinished时直接跳过；lease key用first-index map保留重复键首次语义，刷新变O(B+L)。242CPU tests和复杂度计数回归；aarch64/Python3.12/torch2.10CPU、真实模块＋immediate mock store，4096unfinished全已调度中位96.16→0.244ms，8192block GVA487.48→20.72ms。少量稀疏场景仍可变慢，非NPU/网络端到端倍率。

decode在首次分配前拒绝remote-prefill时直接发DONE_RECVING再清标志，避免prefill等abort timeout；stateful prefill截尾后重建block hash。移除不存在的disconnect_all_peers调用，由native unregister处理peer解除，保留失败检查及wake重注册。建议大并发prefix/offload部署测CPU预算、取消、截断命中与sleep/wake。

### Ascend release支线：PLE host offload与component-aware迁移

**vllm-project/vllm-ascend；本轮发现，原始合入时间/版本：** [#20560](https://github.com/vllm-project/vllm-ascend/pull/20560)（2026-11-02T16:41:36Z，`f9c8e0bdb2`，仅核实合入 `releases/v0.32.0rc`）；[#20625](https://github.com/vllm-project/vllm-ascend/pull/20625)（2026-11-02T13:38:02Z，`87b6de2ea2`，仅核实合入 `releases/v0.32.0rc`）

两项仅核实到releases/v0.32.0rc，非正式release。Qwen4Exp BF16 PLE表可放CANN注册host/UVA，**仅显式--engram-config开启**；DP共享仅同IPC拓扑可用，跨节点/单副本回退private表；PCP可替代部分DP条件，要求ETP=TP并拒绝elastic EP。unregister/free各自可重试。约12GiB/卡是作者估计，本PR主要UT/CPU CI，未给充分NPU精度、吞吐/host带宽证据。

#20625是#20152的backport，按cache component独立传，分开block stride/payload和view offset，最终prefill chunk发布完整快照；压缩MLA、circular/tail state、DCP/replica及Mamba head所有权分别映射。CP path要求PP1，prefill DCP关或等PCP，Mamba TP/PCP不能同时>1；该PR只写“By CI”，没有本轮独立硬件验证。建议pin vLLM/Ascend/CANN组合，先测混合状态P/D布局，不能作一套通用异构缓存合同。

### AITER六rank通信与大EP缓冲：修复静默错误和host ABI溢出

**ROCm/aiter；本轮发现，原始合入时间/版本：** [#7149](https://github.com/ROCm/aiter/pull/7149)（2026-11-02T03:33:30Z，`47eaf46499`，默认分支已见）；[#7299](https://github.com/ROCm/aiter/pull/7299)（2026-11-03T00:21:08Z，`cf76c5b006`，默认分支已见）

六rank仅all-reduce有对应实例，all-gather/allgatherv/reduce-scatter改为不走不存在的custom kernel；六rank小消息all-reduce改用naive线程分配，避免512/6余线程越界。原问题包括未初始化输出/进程抛错与非法地址，建议六卡部署先验证每种collective，不因all-reduce支持而推断其余也支持。

FlyDSL grouped GEMM用typed pointer传只需地址的buffer，避免[64,8192,7168] BF16缓冲3758096384元素被C ABI按int32维度序列化溢出。13个host ABI回归及gfx950真实地址/graph recorder通过，八组gfx1250 ISA逐字相同；**recorder替代了GEMM算术**，20.34→16.37微秒等只测host enqueue，不证明gfx1250整模或数值输出。建议大EP先做地址/offset/缓存替换与真实算术两层回归。

### AITER量化：双布局H16与MiniMax MXFP4缓存消费端

**ROCm/aiter；本轮发现，原始合入时间/版本：** [#6458](https://github.com/ROCm/aiter/pull/6458)（2026-11-02T15:14:43Z，`a9cb51ecf2`，默认分支已见）；[#7060](https://github.com/ROCm/aiter/pull/7060)（2026-11-02T03:08:45Z，`514c7cda00`，默认分支已见）；[#7061](https://github.com/ROCm/aiter/pull/7061)（2026-11-02T03:10:30Z，`85cf44f287`，默认分支已见）

gfx950新增双输出融合量化：rowwise MXFP4及独立signed-H16变换后的转置量化，连续BF16/FP32、维度32对齐，两个输出随机舍入流独立。历史MI350X/PyTorch2.13+ROCm7.1/Triton3.7.1，含分配与布局转换，512×4096相对分解实现14.63×，大shape约4.4×；最终重命名head未重跑该性能表，CI成功不能当整训倍率。

MiniMax-M3 decode/prefill scorer分别读e2m1＋每32值e8m0的MXFP4 index-K；128token页8704B，data/scale各有stride，查询BF16或E4M3。它们补上昨日producer之后的**score消费端**，仍不是所有attention/backend已完成模型集成；e8m0 0/255不在合同，转unit-FP8基线仅指数[119,133]无饱和，宽scale真实分布未测。建议缓存字节、score/top-k、模型准确性逐层验，不把index-cache减半写成整池/整模型显存减半。

### AITER gfx950 MLA：NVFP4 prefix与MXFP8 extend并存

**ROCm/aiter；本轮发现，原始合入时间/版本：** [#7124](https://github.com/ROCm/aiter/pull/7124)（2026-11-02T08:06:25Z，`553db8b17f`，默认分支已见）

`mla_sparse_prefill_v41`可接V4.1的NVFP4 prefix：256B e2m1值＋32B E4M3 scale、每16值一scale；Q与extend仍MXFP8。prefix/extend独立字节row stride可直接吃288B packed row；H≤16用JIT小kernel，更大head用预编译。NVFP4反量化后以BF16做prefix QK/PV，不是把全部计算改为原生FP4。

同PR的a8w8 H16优化为Q只入LDS一次与提前页索引预取，4–17%是该局部路径，不能当新NVFP4格式带来的独立收益。建议核格式/scale与C ABI160B参数匹配，再做前后缀混合精度回归；本轮未运行实际模型。

### ATOM WideEP：主干MoRI V1LL opt-in，精度与链路分开验

**ROCm/ATOM；本轮发现，原始合入时间/版本：** [#2817](https://github.com/ROCm/ATOM/pull/2817)（2026-11-02T12:43:00Z，`974414f16f`，默认分支已见）

`ATOM_MORI_V1_LL=1`默认关；同节点用IntraNode，跨节点InterNodeV1LL采用handle的launch几何而非每调用覆盖，避免AsyncLL只发不收的hang；ATOMesh增加跨节点角色。作者MI350X、DeepSeekV3 FP8、MoRI IO write、1319题GSM8K，1P1D TP8 strict0.8817，DPA8/EP8为0.881，2P2D DPA16/EP16为0.8855，全部1319传输校验；这些支持连通与精度接近，不构成吞吐扩展性倍率。KimiK2.5 MXFP4数值来自早期base。建议pin MoRI/aiter与节点几何复现超时、取消、数值和服务吞吐。

### ATOM gfx1250支线：状态checkpoint预算与prefix恢复竞态缓解

**ROCm/ATOM；本轮发现，原始合入时间/版本：** [#2884](https://github.com/ROCm/ATOM/pull/2884)（2026-11-02T17:05:29Z，`a3d3cfe619`，仅核实合入 `alizaidy/gfx1250-qwen-support`）；[#2868](https://github.com/ROCm/ATOM/pull/2868)（2026-11-02T23:28:14Z，`397964a7d4`，仅核实合入 `alizaidy/gfx1250-qwen-support`）

只在alizaidy/gfx1250-qwen-support支线。GPU image预算超限按superseded/chain root/tip、branch-back与LRU清READY未pin状态，保留对话恢复点；Qwen3.5-35B-A3B MXFP4、4×gfx1250 DP4/EP4、FP8KV、c768/9808轨迹，作者10分钟点2–8分钟generation33.9k→37.1ktok/s；“interactivity”是generation/(server requests×4)，不是客户端测得延迟。

#2868把DP prefill graph中的GDN保持eager，状态拷贝前drain，并让attention只跑scheduled rows以避免padding触发64GiB scratch。**它降低了prefix恢复乱码概率，底层race仍未消除**：两次c2048无异常，但c512仍2条异常，同prefill batch。建议继续观察并保留逐请求验收；不能写成主干修复完成或LLM精度已保证。

### Helion Split-K：PDL重叠和atomic epilogue是两种数值合同

**pytorch/helion；本轮发现，原始合入时间/版本：** [#4225](https://github.com/pytorch/helion/pull/4225)（2026-11-02T05:48:42Z，`c88c5c82e2`，默认分支已见）

workspace reducer先发所有partial load再求和，以PDL和producer尾重叠，保留原bias/舍入顺序；新增可选 `cute_split_k_atomic_epilogue` 直接packed atomic写最终output、省partials/reducer，fill先清零、epilogue等依赖。后者逐partition按output精度舍入，前者FP32合一次再cast，不能称数值相同；仅证明atomic-add模式且适用FP16/BF16/FP32，promoted-half不准入。

B200固定FP16+bias(128,32768,256)，同进程手改阶段数字与fresh-process ABAB不同；后者atomic7.166us对固定Triton7.740us约1.08×，不引用6.74us作同条件比较。建议自写split-K先明确误差预算、别名/PDL等待与graph poisoned-output测试。此提交晚于昨日1.5.2发行，未据版本号声称已在该包内。

### TPU / XLA：P2P描述符寿命、半精度addmm及重计算别名安全

**google-pytorch/torch_tpu；本轮发现，原始合入时间/版本：** [#6096](https://github.com/google-pytorch/torch_tpu/pull/6096)（2026-11-03T00:41:30Z，`779050e334`，默认分支已见）；[#6062](https://github.com/google-pytorch/torch_tpu/pull/6062)（2026-11-02T20:40:41Z，`d0dcc2b0f2`，默认分支已见）

TorchTPU P2P sender读到one-shot接收descriptor并enqueue远端copy后删除c10d store key，避免长期pipeline服务每tensor增长；删除best-effort，仅警告，不把enqueue混作远端完成。addmm/_addmm_activation的FP16/BF16 alpha/beta在host保FP32，DotGeneral与epilogue用FP32，最后cast，处理scalar>65504和中间溢出；最终输出dtype仍低精度。建议长期send/recv容量与大scalar数值回归，未提供整模吞吐收益。

### JAX Pallas：lookahead至少两个输入buffer

**jax-ml/jax；本轮发现，原始合入时间/版本：** [#42210](https://github.com/jax-ml/jax/pull/42210)（2026-11-02T23:05:53Z，`e8937d4005`，默认分支已见）

Buffered及TPU BufferedRef构造时拒绝use_lookahead且input buffer<2（包括tuple规范化），因为单slot无法一边消费一边预取，可能跳block0、等待错误字节或覆盖未消费数据。它是fail-fast正确性边界，不是新prefetch性能功能。建议自定义TPU pipeline用边界block、空BoundedSlice与纯output buffer测试。

### XLA rematerialization：保存被覆盖的操作数再重算

**openxla/xla；本轮发现，原始合入时间/版本：** [#51823](https://github.com/openxla/xla/pull/51823)（2026-11-02T17:22:10Z，`4c6a7502de`，默认分支已见）

新增可选alias_info，让pass通过HloAliasAnalysis找到while参数/root共享存储的overwriter；尚未覆盖时加控制边，已覆盖则给单view array保存copy并计入成本，多view/tuple不跨覆盖重算。overwriter及其definer/原地自更新固定。未传alias_info时行为保持旧路径，不能声称所有XLA训练自动消除此问题。建议编译器集成方核调用端是否启用、loop数值与显存成本，避免低显存优化读到下一迭代值。

### KAI Scheduler：Deferred in-place resize的就地抢占

**kai-scheduler/KAI-Scheduler；本轮发现，原始合入时间/版本：** [#2510](https://github.com/kai-scheduler/KAI-Scheduler/pull/2510)（2026-11-02T10:58:27Z，`a271a9852a`，默认分支已见）

`--deferred-resize-preemption`默认关；对kubelet标Deferred的resize，已按max(spec,allocated,actual)计费，用零新需求stand-in求解节点短缺，只从原node找victim，最终只提交victim驱逐/搬移，不驱逐或重新bind正在resize的pod。pending先处理、gang未保障不驱逐、victim终止中不重复抢占等约束保留。建议弹性CPU/内存伴随GPU任务先做gang/NUMA与reclaim回归；这不是在线GPU切分新能力，未给训练goodput收益。

### Triton CDNA5：partitioned shared layout接入异步拷贝

**triton-lang/triton；本轮发现，原始合入时间/版本：** [#12547](https://github.com/triton-lang/triton/pull/12547)（2026-11-02T16:17:37Z，`f54dadcc59`，默认分支已见）

global↔shared async copy改用通用linear layout并处理多个shared-memory base，补conversion lit与Gluon单测。支持边界是CDNA5的partitioned shared layout，不是所有AMD GPU上的新attention实现；没有端到端LLM性能数字。建议自写GEMM/attention流水线者审新lowering与分区地址，继续验证race与bank布局。

### PyTorch：以真实main提交核对通信、量化加载与profiling

以下PR在搜索界面可能仍未显示merged；本轮实际main提交是能力状态证据，不能用搜索0/1条断言无更新。

- **[ROCm] Use the device warp size for rocSHMEM all-to-all-v tiles**：[a66bab0815](https://github.com/pytorch/pytorch/commit/a66bab0815ee39b649b0b66b5ca8e2a2466fd673)（2026-11-02T21:46:35Z）。rocSHMEM all-to-all-v按设备warp size及显式WarpScan逻辑宽度分配，避免wave32 GPU仍硬编码64的共享scratch冲突。gfx1250有2/4卡结果，gfx1100/1201未验；gfx9设备码不变。建议AMD EP先回归offset/align与group边界。

- **[Kineto] Enable CUDA sync-event correlation without incompatible device timestamps**：[0effc5bab1](https://github.com/pytorch/pytorch/commit/0effc5bab110ac67514f130c349e37c3b5ddfc14)（2026-11-02T20:53:48Z）。CUDA13+显式采CUDA event以恢复sync correlation，但只有CUPTI≥13.4且驱动查询支持13.4+才开device timestamp；零timestamp不发独立event，相关元数据仍可用。建议profiling分析区分“未显示event”与“未关联依赖”，在准确的驱动/CUPTI组合验证双次capture。

- **[inductor] Profiler markers that tie each kernel launch to its provenance debug handle**：[d6d6258597](https://github.com/pytorch/pytorch/commit/d6d62585979f4499f123a81a3e7cc5736caf1304)（2026-11-02T20:22:24Z）。Inductor默认关闭的provenance_profiler_markers在Python wrapper每次launch前记录kernel/debug_handle，可把外部采集的Kineto trace关联post-grad FX节点；fused time如何拆分仍由消费者决定。C++/AOTI/FXIR/CUDA graph replay不发这些marker，开启后未采样也约有0.4us/launch CPU成本。建议性能归因实验启用，勿承诺任意图回放完整归因。

- **symm_mem: Complete per-group stream serialization for NCCL and improve tests **：[80c1d0bf58](https://github.com/pytorch/pytorch/commit/80c1d0bf5830c2402af19d46e0bc01d47b6d6d6b)（2026-11-02T19:39:43Z）。symmetric-memory NCCL reduce-scatter-offset/all-to-all-nd补per-(ProcessGroup,device) stream serialization；eager与capture各保留last-op，不让中间capture丢失eager顺序。NVSHMEM、自写signal-pad和裸NCCL signal不在合同；多线程交错多capture也不是完全排序保证。建议多stream通信用户优先回归死锁及graph依赖。

- **[DCP] Fix QuantizedHuggingFaceStorageReader loading only part of each MXFP4 tensor**：[9e4e615f29](https://github.com/pytorch/pytorch/commit/9e4e615f29709f67c9cef700d998f78fdf424fe4)（2026-11-02T18:13:29Z）。QuantizedHuggingFaceStorageReader不只改MXFP4 tensor总shape，同时改dequantized chunk offsets/sizes并重键storage_data。旧planner按quantized chunk交集只加载小部分权重，作者GPT-OSS20B示例约97%专家值未写；补完整与group内partial-read回归。建议DCP→HF量化加载逐tensor比较，不用能启动代替权重完整性。

- **[Inductor] Preserve singleton dimensions in NVGEMM reduction outputs**：[2c970505ad](https://github.com/pytorch/pytorch/commit/2c970505ad5f808920f3c07f81b34654c710ac73)（2026-11-02T15:56:43Z）。NVGEMM grouped reduction保留逻辑矩阵singleton维再做A/B swap，修(1,4096)结果只写第0项、其余未初始化。新增NaN预填增强回归；它是具体layout正确性修复，非全GEMM精度提升。建议启用该fusion的训练验证单例group和epilogue输出。

### 当日撤回与release检查

PyTorch main中200459–200466八项Inductor优化及202198 per-thread tracing均出现合入后撤回；本日报不把这些暂态提交计为净新增能力。扩展34仓release入口均已打开，当前窗口ATOM0.1.9、Dynamo1.7.0-mistral-large-4-dev.1及KAI多支线patch发行已记录；Dynamo标签含dev，KAI安全依赖/普通维护、Run:ai3.0.22未据版本号单独凑条。FlashInfer0.7.3rc1/Transformers5.22.1/TRL1.17.1及Helion1.5.2属已报回溯旧状态，不重复生成成果。

<a id="hardware"></a>
## 硬件与平台技术材料

### NVIDIA Workload Map新技术说明：用声明式结构替代每个框架一套解析器

**NVIDIA；官方文章日期2026-11-02，本轮新收技术说明，项目本身不是本日新创建。** [官方原文](https://developer.nvidia.com/blog/support-new-ai-frameworks-faster-using-nvidia-workload-map/)展示CRD＋Go库：以路径/表达式定位pod template、replicas和状态，将JobSet、LeaderWorkerSet等解析为统一component/instance树。JobSet示例按replicas×parallelism计算pod数，再由各consumer使用这套结构。

CLI用kubeconfig读取即可检查，不要求先装集群controller；可选gang分组由KAI执行，native grouper plugin优先。Workload Map本身不执行调度。Apache-2.0，官方称Run:ai生产使用；没有量化接入工时、训练goodput或大集群吞吐对照。**建议多训练框架/Kubernetes平台团队深入读catalog与消费者边界，再考虑接入；不据此宣称LLM训练性能提升。** 本轮仅按官方文章快照解释合同，未部署或逐行审该repo。

<a id="coverage"></a>
## 实际覆盖、补录、更正与续查

| 来源面 | 本轮实际检查 | 边界与下一步 |
|---|---|---|
| 六核心仓库 | Megatron/vLLM/verl/SGLang/slime/TorchTitan；72h重叠617条merged搜索、603个main提交、258个release对象，分页去重完整；111个PR完整最终文件列表 | 选择性机制审阅；423个当前/继承候选待详审，较早失败区间保留 |
| 扩展34仓库 | 1586条merged搜索覆盖72h；619个默认分支提交覆盖主窗；release入口已到回溯前；63PR最终文件列表＋6个PyTorch真实commit | 搜索与committer时钟不能相加成成果；1335个未报候选待语义审，超大生成内核/完整lifecycle仍有未审部分 |
| 模型与厂商 | 52个HF组织×创建/修改×模型/数据集，208查询均到回溯边界；135候选、64主窗树比较中63有窗前基线；57官网入口49直接成功 | 访问恢复/仍失败逐项记录；15178个全局新HF仓库候选仅元数据，不等于已读；全局lastModified、动态历史与受限资料仍缺 |
| 论文与正式学术源 | 7类arXiv new页11月2日746个稳定作品；HF daily；19篇全文、15篇纳入；回溯submitted API收齐1018条metadata；ACL/PMLR/TMLR入口 | 8891候选、806 replacement队列可能重叠，不相加；真正updated查询未证覆盖，OpenReview及两PDF仍403/挑战，正式卷历史未详审 |
| 硬件/软件平台 | NVIDIA新技术说明、AMD ROCm notes/blog、Intel Gaudi、Ascend产品公告、Google TPU、AWS Neuron，共7官方快照均HTTP200 | CANN公告正文动态不可据200称已查完；libtpu/CANN/HCCL/Neuron组件历史及既有队列继续待查 |

扩展仓库除既定九项，还覆盖TransformerEngine、FlashInfer、Dynamo/GMS、DeepEP、NVSHMEM/NCCL/RCCL、AITER/ATOM/ROCm libraries、TorchTPU/XLA/JAX、MLX、Helion、KAI、llm-d、Run:ai及旧候选项目。主题检索补进Workload Map新技术说明、Weight-Space Programming重建、GSQ-RCO、Voice Reader、LifeVal与Anthropic研究。固定来源名单与本轮检查状态另存本地 `.monitor/sources.json` 及各域ledger，历史队列逐项继承。

**失败/恢复。** arXiv主窗初429、重试200/可见0项并非完整无更新证明；updated查询被服务改写为submittedDate，不能称完成修订扫描。HF11月3日daily为400；OpenReview/部分PDF403或挑战页；Dory技术报告仍403。部分官网urllib失败后由web恢复，未恢复者见模型ledger。NVIDIA文章web open内部错误后已用官方HTTP200全文恢复。六个扩展repo继承的错误默认分支造成404，查询仓库metadata后按master/develop/develop_deprecated重试成功；失败尝试保留，不冒充第一次成功。ROCm Core10.1/Gaudi1.24.2/Neuron2.33等旧版本不因本日抓取重发。

**补录与更正。** Fair Cast、DLC、Muon-RT及MTG-AD来自更早pending；LongCat按11月1日技术页/11月2日新闻分开，Anthropic仅能定October2026。核心域两项旧窗补录均注明原始时间，release/backport只计当前工件增量。交叉审将LSA的9.21/2.87明确为基线/LSA时延比，并区分GLM原sparse基线；Transformers KV-sharing按最终API名称而非PR早期正文；SGLang回滚和PyTorch当日撤回均未列为净新增能力。没有改写历史日报。

**覆盖水位与去重。** 本轮202个稳定来源事件（核心100、扩展69、模型14、论文18、平台文章1），可含同一工作的报告/代码/权重等多个工件，**不是202项独立成果**。本地稳定ID/版本台账与历史已报版本比较无同版本重复；更旧只提及的pending仍可作为补录，状态变化明确标注。全局完整语义覆盖水位保持未确认，旧129项backlog原样保留，并追加本轮队列引用；未详审区间不关闭。

下一轮从**2026-11-03T00:58:48Z**续查并回溯72小时，同时优先处理未同步文件及所有旧未审/失败区间。该时间是事件发现截止，不承诺所有来源到此已全面审读。本地保存与GitHub同步分别记账，远端只有回读全文、blob及提交均通过后才记成功。
