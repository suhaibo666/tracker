# LLM 技术动态日报 · 2026-10-15

> 主扫描窗口：2026-10-14 10:34:06 → 2026-10-15 09:55:47（Asia/Shanghai）；72小时回溯从2026-10-11 10:34:06开始。实际生成时间：2026-10-15 10:16:30（北京时间）。另补查上轮登记的10-05 08:00→10-08 08:00核心项目缺口及扩展项目/论文待办；各来源完成程度见文末。
>
> 进度恢复依据：本地持久化记录、历史日报明确窗口、GitHub tracker/main `23b0fc82945dcdf120fcf7b0753be394da280ad3`，11个归档文件逐项Git blob一致；不依赖自动化“Last run: never”或文件修改时间推断空白。正文中的GitHub/HF精确时刻默认UTC，未给时区的博客仅保留原始日期。事件筛选截止09:55:47，页面/代码核验持续至生成，动态页面为读取时快照，不保证历史页面状态可重建。
>
> 证据口径：实际打开论文正文、官方卡/代码/release/PR；所有性能与训练数值为原作者或厂商报告，本轮没有独立GPU/NPU复现。main/master、开发支线、候选release与未合入PR分开标记。旧工作补录与新工件/新修订分开记账。

## 本次重要变化

1. **Step 5 的新信息主要在工件。** 主窗有NVFP4上传，可读BF16/NVFP4配置揭示交错SWA/稀疏GQA与多token预测；BF16权重已有9月上传历史，不能把README刷新写成首次开源。输出长度文档和NVFP4独立验证仍有缺口。
2. **训练数值正确性先于性能。** DeepSpeed FP32梯度分区、router buffer dtype、AutoEP checkpoint/TP folding，以及TRL编译损失/aux scaling各有不同触发条件。loss看似正常或单测通过都不足以排除梯度错误。
3. **异步KV必须审计真正完成点。** vLLM P2P取消后仍可能继续写，SGLang radix页归属可在请求成功后才暴露，昇腾KV Pool故障也可能卡住调度器。两补丁组合及未合入后续修复仍不能视作全链路认证。
4. **低比特的主要收益和代价要逐路径测。** Engram表压缩减少host占用，但依赖未合入运行时；MiniMax NVFP4 KV改善来自host/graph开销；TRL/DeepSpeed融合loss的单算子倍率与整步收益差很多。
5. **预训练有独立值得读的进展。** Indic CPT揭示词表扩展、旧行冻结和英语保留的权衡；重复数据训练需联合选择batch/epoch/正则；VHA在训练中重参数化Q/O以折叠为更少KV头，但尚无通用端到端加速证明。
6. **环境与critic研究保留反证。** SoftwareGym不能据合成数据迁移推出替代native数据；ProgramGym的test通过率不是完整任务解决率；TPPO、DisaggKD都需要匹配预算、精度与强基线再复现。
7. **长上下文吞吐与交互时延可能方向相反。** AMD同节点DEP在96K/32K上吞吐+147%，median TPOT却升高；短8K/1K还慢26%。选并行拓扑要带自己的流量与时延约束。

## 阅读目录

- [模型与技术报告](#models)：Step 5、AesCode、UserRepGen、Engram与模型工件补录。
- [预训练与架构](#pretraining)：Indic CPT、重复数据、VHA。
- [后训练与 Agent 数据](#post-training)：SoftwareGym、TPPO、DisaggKD、ProgramGym。
- [推理优化研究](#inference-research)：DoSA旧候选补录。
- [工业开源项目：核心六项](#core-projects)：训练、KV、性能、并行能力及支线状态。
- [训练与推理工程：扩展项目](#extended-projects)：TRL、DeepSpeed、Transformers、FlashAttention、PyTorch、Triton、TensorRT-LLM、昇腾。
- [硬件软件协同](#hardware)、[评估与可靠性](#evaluation)、[来源覆盖与未完成范围](#coverage)。

<a name="models"></a>

## 模型与技术报告

### Step 5 Preview：600B 多模态 MoE 的权重与推理工件可核查，NVFP4 在主窗上传

**日期与增量。** 原 API/预览公告是 **9 月 20 日**，不能写成今天首次发布模型。官方 NVFP4 仓库创建 `10-14T02:59Z`，`542f8b345779a301533c14dbbf97ebb3463f536d` 于 `10-14T07:10:06Z` 标记 “Upload Step-5-Preview NVFP4 v1 weights”；截止前卡版本 `bc246cc2848e17b37f377eda8fd825c5b3018d1f`。BF16 截止前版本 `7734b7ffac118d70d8398bfa3eb0fab007eb45b1`，API 列出 48 个文件、604,339,504,832 stored parameters，公开非 gated；其历史已有 **9-20** 权重上传，不能以本轮 README 更新推断新训练/重新上传。FP8 元数据版本 `364b3b03fed2477aa0953fe855a2c4c8c48ee3cc`、49 文件，但 API 标记 `gated: manual`，本次无认证 raw README/commit 返回 401。公开工件可见性不等于精确开放时间，本次只能确认截止前状态。

**已披露机制。** 官方卡称 600B 总参数、27B 激活，text/image/video 输入、text 输出。BF16 固定 config 更具体：92 层语言 backbone，隐藏宽 4096；SWA 窗口 512，与 full-attention 层交错；full 层启用 sparse GQA，indexer top-k=512、region block=8；352 routed experts、top-8，另有共享专家，3 层 next-token predictor。config 的 layer_types 包括额外预测层，不能直接把整个数组长度写成 backbone 层数。NVFP4 config 为 ModelOpt 0.47.0、group16、仅部分模块量化，KV cache 未量化；不是全模型统一 4-bit。训练 token、数据配比、优化器、训练 FLOPs、后训算法未在已读卡/配置披露。

**部署证据与边界。** 卡提供专用 `vllm/vllm-openai:step-5-preview` / `lmsysorg/sglang:step-5-preview` 镜像与 step5 reasoning/tool parsers。FP8 示例为 4×B300 或 8×H200；BF16 为 8×B300 或 16×H200。这些是装载/服务起点，不保证 1M 请求加并发后 KV cache 仍能容纳。BF16 reference README 报告完整模型 streaming-weight 推理验证；顶层卡同时明确 **BF16 convert→file-shard generate 全流程尚未端到端验证**，两种路径应区分。NVFP4 卡沿用 FP8/BF16 示例，未给该量化版独立 benchmark/校准集/速度数据，不可把通用卡分数当 NVFP4 等价性证据。

**冲突与许可。** 顶层卡 Maximum Output 写 1M，平台 API docs 写 64k，后者配置表又称 `max_tokens` 默认 INF；卡表的 Input 仅 text/image，而介绍和 API docs 含 video。建议日报只报“总上下文可配至 1M”，明确输出/视频服务边界需按实际端点复核。许可是 **StepFun Model License 1.0**，含附加归属与商用条款，不能写 Apache2.0。许可附有归属与商用条件；采用前以原始条款为准，本日报不作商业授权解释。

**实践建议（判断）。** 优先锁定 BF16/NVFP4 revision 做自有长程 agent 回归；将上下文、输出上限、工具解析和量化质量分开测试。NVFP4 先验证装载与多模态路径，不据名称推断已具成熟专用服务支持。

来源：[9-20 官方预览](https://www.stepfun.com/news/step-5-preview)、[BF16 固定卡](https://huggingface.co/stepfun-ai/Step-5-Preview-BF16/blob/7734b7ffac118d70d8398bfa3eb0fab007eb45b1/README.md)、[BF16 config](https://huggingface.co/stepfun-ai/Step-5-Preview-BF16/blob/7734b7ffac118d70d8398bfa3eb0fab007eb45b1/config.json)、[reference inference说明](https://huggingface.co/stepfun-ai/Step-5-Preview-BF16/blob/7734b7ffac118d70d8398bfa3eb0fab007eb45b1/inference/README.md)、[NVFP4上传commit](https://huggingface.co/stepfun-ai/Step-5-Preview-NVFP4/commit/542f8b345779a301533c14dbbf97ebb3463f536d)、[NVFP4固定卡](https://huggingface.co/stepfun-ai/Step-5-Preview-NVFP4/blob/bc246cc2848e17b37f377eda8fd825c5b3018d1f/README.md)、[API 文档](https://platform.stepfun.ai/docs/en/guides/models/step-5-preview)、[英文许可](https://huggingface.co/stepfun-ai/Step-5-Preview-BF16/blob/7734b7ffac118d70d8398bfa3eb0fab007eb45b1/LICENSE.txt)。HF API、raw 文件与 commit 实际读取；web 对部分 HF 页面失败，不把失败冒称内容不存在。

### AesCode 8B/32B：以设计图拆分可验证内容与视觉奖励，再逐通道归一化 RL

**日期/版本。** arXiv `2610.18860v1` 提交 `10-13T17:59:31Z`（回溯窗）；8B/32B 权重 release commit 分别为 `cfe1f698e09b8e7a009901ca700c765e25240af6` / `3f9157f331c574b470149ddd88e67d5bd13481f8`，`10-14T19:28Z`，属主窗工件。仓库 9-29 预建不作发布日期。官方代码基线 `microsoft/AesCode @ e7796913f07dfe2c6a4034ab70e15f09524948fa`（10-14T19:49Z），已读 README；未逐函数审计或运行训练。

**机制与配方。** Qwen3-VL 8B/32B 以 prompt 确定内容，生成图只供布局/风格参考。把目标拆成含元素、属性和关系的 design graph；六个确定性验证器覆盖执行、文字、边界、表格/图表、布局与留白，第七路 VLM 用与图元素绑定的二值 rubric 判美学。七路分别做组内归一化后聚合，避免易饱和的内容奖励盖过稀疏美学信号。图为监督结构，不是额外塞给 policy 的输入。论文 §4/AppB：约1.8万源设计，经筛选形成3000 SFT例与7408 RL prompt；RL参考图用GPT-Image-2。卡给 verl FSDP-vLLM hybrid、无critic/独立训练reward model，128 prompts×8 rollouts/step，AdamW 5e-6；8B400步、32B520步。GPU时数、FLOPs未披露。

**结果与条件。** 300个 infographic、每题3次生成取均值而非选优；Rule与Visual均分为Overall，8B82.94、32B85.89，对照相同参考图Qwen3VL为51.93/61.10。8B推理去参考图Visual只降1.00点。**全部评测生成采集在8-28至9-18**；reasoning闭源端点用provider默认采样，不能写成10-14最新闭源模型全面横评。模型权重Apache2.0，代码MIT，需区分。

**限制/建议。** 所有系统Style<60；表格必须HTML table、图表必须ECharts，参考里的照片可对应占位槽，不能外推为任意网页/插画能力。AppF中，32B在适用46/30题的关系子图/语义图标重复失败率23.9%/23.3%，仍弱于GPT5.5的10.9%/3.3%。适合借鉴“确定性核验+视觉rubric+分通道归一化”，上线仍需渲染后纠错与复杂关系独立回归。

来源：[论文v1](https://arxiv.org/html/2610.18860v1)（已读§3–4、AppB/C/F关键段）、[8B固定卡](https://huggingface.co/microsoft/AesCode-8B/blob/cfe1f698e09b8e7a009901ca700c765e25240af6/README.md)、[32B固定卡](https://huggingface.co/microsoft/AesCode-32B/blob/3f9157f331c574b470149ddd88e67d5bd13481f8/README.md)、[固定代码基线](https://github.com/microsoft/AesCode/tree/e7796913f07dfe2c6a4034ab70e15f09524948fa)。

### UserRepGen-4B：把用户模拟的优化对象移到可复用行为表征

**日期/工件。** Microsoft官方HF创建 `10-14T20:30Z`；`bac09c8247deb9bc813007f11830d8f9ce6d8416`于20:39Z上传权重；固定卡 `bfc905b249365cab00e94ba308f2cea5b23bbc6e`（22:03Z）。是实际可读取工件，卡仍保留draft/论文链接待确认、私有仓库操作提示等残留，不能称完整论文正式发布。

**机制/数据。** Qwen3.5-4B，rank64 LoRA；输入多轮对话，输出JSON `task` 与 `user_behavior`，并非生成模拟用户下一句话。Online SFT：当前策略生成表征→固定模拟环境rollout→比较原对话与模拟行为→自然语言hindsight反馈改写表征→改写作在线监督目标。3200段英语对话来自7数据源，去近重复、safe过滤、至少4个user turns；训练上限8192 prompt+1024 output。推理直接生成表征，无需再跑反馈循环。MIT；GPU清单仅是推理服务硬件，未给训练算力/总成本。

**评测。** 700段独立对话、每源100；GPT5.4mini同时承担simulator、assistant与fidelity judge（调用分离），与训练Qwen/Gemma环境不同。1–10分制best-of-1/3为4.95/6.02，对Qwen3.5-4B基线3.41/4.42；SFT4.64/5.68、GRPO3.57/4.59。100对不等分样本、一位专家盲评与judge偏好88%一致。不要把test-time多轮反馈的6.10/7.07当checkpoint直接推理成绩。

**边界/建议。** 仅text/英语，保留VLM骨干不代表支持图像视频；不支持超9216训练长度外推。表征可能过度推断、混入任务细节；同用户匹配准确率51.02%（147人、4选1）高于原对话44.22%，说明表征不是匿名化。可用于构造agent交互评测人口，需同时核验模拟保真、任务泄漏和表征可链接性；不能拿合成反馈代替真实用户结果。

来源：[固定模型卡](https://huggingface.co/microsoft/UserRepGen-4B/blob/bfc905b249365cab00e94ba308f2cea5b23bbc6e/README.md)、[权重上传](https://huggingface.co/microsoft/UserRepGen-4B/commit/bac09c8247deb9bc813007f11830d8f9ce6d8416)。卡全文、模型API与commit已实读，论文链接尚未确认。

### NVIDIA DeepSeek V4.1 Flash：Engram查表FP8→MXFP4，体积收益伴随未合入运行时依赖

**日期/增量。** 原NVFP4模型9-16发布；本轮 `c48dca0146b56d64c8116dbb6fcace8b7cc835f8`（10-14T16:40:46Z）仅更新shards47/48、config、卡，将Engram layers1/14的lookup tables从源FP8量化为MXFP4。其余checkpoint沿用旧`3431dde3247c13b5957f682b1e3c6fcae2566079`。不是新基础模型、不是所有权重无损转换。

**技术事实。** 表用E2M1两数/byte、每32元素一个UE8M0 scale；卡报表relative L2误差约12%，checkpoint约492→400GiB；TP4且host-pinned lookup每rank23.6→12.2GiB。原routed expert MXFP4→NVFP4的lossless weight conversion与本次**有损**表压缩是两件事。ModelOpt v0.47.0rc0、源DeepSeek `dba1be0...`。卡评测GB300/vLLM，相对源混精checkpoint（MXFP4 routed experts，非旧NVFP4版）的GPQA91.035→90.783、AA-LCR78.563→79.562、SciCode54.401→53.994，混合升降；temp1/top_p.95、reasoning_effort100、maxnew262144，GPQA/LCR16 repeats、IFBench5、TerminalBench另8 trajectories。无置信区间或表压缩独立消融，不称无损/更准确。

**部署强约束。** 当前main旧vLLM镜像（包括卡所列deepseekv41-flash-0909）不能加载，需[vLLM #61530](https://github.com/vllm-project/vllm/pull/61530)。框架调研已核该PR **open、未合入**，head `1800929afdd3728860f794bcf69ae7beda95649d`；不能写“升级官方main即可”。SGLang尚不支持/未验证MXFP4表，卡要求pin旧3431dde。卡中DSpark保留不等于已测spec decoding，vLLM示例为text-only。MIT。

**建议。** 有host-memory压力的部署可做受控试验，锁定checkpoint与PR build；同时测查表位置、延迟和自有任务质量。PR报告GPU resident lookup慢约36%，host pinned慢约1–3%，且单次GSM8K没有隔离expert baseline，因此不把PR烟测当全模型吞吐结论。

来源：[固定模型卡](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-NVFP4/blob/c48dca0146b56d64c8116dbb6fcace8b7cc835f8/README.md)、[工件diff](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-NVFP4/commit/c48dca0146b56d64c8116dbb6fcace8b7cc835f8)、[运行时PR](https://github.com/vllm-project/vllm/pull/61530)。本轮已交叉核读卡全文、raw diff与运行时PR状态。

### Grok Voice 3.0：确认版本与别名变化，技术披露待补

10-14官方公告与release notes确认 `grok-voice-3.0` 已可用，`grok-voice-latest`自动切到3.0，2.0弃用；WebSocket/session/events接口沿用。工程增量是浮动别名行为发生变化，可复现实验应固定版本并重跑打断、turn-taking和端到端时延回归。

参数、架构、数据、预训练/后训练配方、算力和权重均未披露。本轮未核底层独立评测协议，不采用公告的横评分数作为可靠结论。建议继续观察技术报告与可复现实验，暂不据宣传判断训练或部署收益。来源：[官方公告](https://x.ai/news/grok-voice-3)、[release notes](https://docs.x.ai/developers/release-notes)、[接口文档](https://docs.x.ai/developers/model-capabilities/audio/speech-to-speech)。

### 模型工件与配方补录

#### GLM-Legal：10-13官方技术博客，领域mid-training与法律环境RL

- AutoClaw/Z.ai官方10-13文章，基于GLM5.3Flash，扩充法律语料做mid-training，再在“数万”法律环境RL，与harness的context/tool/feedback同步迭代。官方只给定性流程，未披露语料token/配比、RL算法、reward公式、训练GPU/FLOPs，也未找到对应权重卡；API目前beta申请，AutoClaw接入为将来时。
- 评测发布者Vals原始页标10-12：HarveyLAB19.17%±3.25、LegalBench86.23%±0.41、LegalResearchBench41.83%±3.43，默认temp1/top_p.95/maxoutput128000/reasoning max。官方19.2%是四舍五入，不能写“可靠完成所有法律工作”。Vals页“open-weight”字段与未找到官方权重的状态不一致，不能据聚合metadata写权重已发布。
- 建议（判断）：可作为领域mid-training+真实环境RL方向短讯；机制披露不足，不适合作本期可复现配方重点。只讨论技术评测，不提供法律实务建议。
- 来源：[官方全文](https://autoclaw.z.ai/blog/model/GLM-Legal/)、[Vals直接评测页](https://www.vals.ai/models/zai_glm-legal)。前者web一次超时后raw HTTP读取全文成功，落在`models-glm-legal.txt`。

#### Qwen-Image-NVFP4：10-12量化checkpoint，配有对齐质量条件但没有速度数据

- NVIDIA卡明确发布10-12，固定`4928c00477453b3570521e70a32e34b54c3a6624`；基于**原Qwen-Image**，不是上轮QwenImage2.1Turbo。20.43B denoiser /28.85B full pipeline；784个linear位于blocks2–57，支持算子W4A4，其余含text encoder/VAE保留高精度。ModelOpt0.47.0，StableDiffusion-Prompts仅用prompt校准。
- TensorRTLLM VisualGen1.3.0rc28/GB300验证；1024²、50step、CFG4、空格negative prompt，同条件BF16/NVFP4 QwenImageBench48.986/49.754，OneIG-EN51.337/51.563。1000 prompts(seed42)与1119/1120 prompts(seeds42–45)；有1题预先排除，未给延迟/吞吐/显存对照。不能从分数小幅上升推断量化提高能力，也不能从NVFP4名称推断倍速。
- 许可metadata与governing terms为NVIDIA Open Model Agreement，同时提upstream Apache2.0，不能只写Apache。适合Blackwell量化质量对照短讯，多模态训练日报优先级低于主条。
- 来源：[固定卡](https://huggingface.co/nvidia/Qwen-Image-NVFP4/blob/4928c00477453b3570521e70a32e34b54c3a6624/README.md)。卡全文与commit列表已读。

#### DeepSeek V4.1 Flash reference：补完上轮cache修复待查，10-12已有实质code diff

- `8b3cec70c307624fb098c3a04bdbfc118e7abda9`（10-12T08:37:23Z）只改`inference/model.py::Indexer.forward`：此前只有`owns_k and latent != None`（压缩组完成）才把自己的k_cache挂回共享对象；现把`shared_attn.index_k = self.k_cache`移到单独`if self.owns_k`，令未完成compression step也恢复该owner的cache。
- 实读固定code的Compressor/Indexer：decode压缩组未满时latent为None，但indexer仍随后读取shared_attn.index_k算score；这说明修复的是cache选择/所有权，不是append新latent的频率。可避免共享指针沿用其他层cache（根据所读控制流的判断）；本次未运行复现，不能声称具体长上下文分数/速度改善。不是10-12新权重。
- 来源：[修复commit](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/commit/8b3cec70c307624fb098c3a04bdbfc118e7abda9)、[固定code](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/8b3cec70c307624fb098c3a04bdbfc118e7abda9/inference/model.py)。本轮关闭“仅查到标题未读修复”的旧缺口，仍无执行验证。

#### MiMo-V2.6 preprocessing：10-09旧缺口补完，不当本期新模型

- Pro-MOPD `6a7f202e8dbd0e57b3fe1da03917d08d75abc4b8`（10-09T06:30:36Z）仅改preprocessor_config：min_pixels3136→8192，max_pixels12845056→8388608；image_mean从CLIP值改为[.485,.456,.406]，std改为[.229,.224,.225]。作者标题明确“align to training values”。这是输入预处理对齐，不是训练token/权重变化；未测质量/视觉token预算影响。
- 9-27 tool-call repetition博文已实际重开，API9-25上线MOPD、不是10月新发布。其“回退5步+repetition专项teacher+teacher-prefixOPD”及$9万成本可作旧机制阅读线索，不重复当本日新消息。
- 建议：复现MiMo视觉评测应固定包含修正的processor，不能只锁模型权重SHA而复用旧CLIP归一化。
- 来源：[processor diff](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD/commit/6a7f202e8dbd0e57b3fe1da03917d08d75abc4b8)、[9-27官方博文](https://mimo.xiaomi.com/blog/mimo-v2-6-tool-call-repetition)。本轮关闭旧“未读preprocessing修复”缺口。

<a name="pretraining"></a>

## 预训练与架构

### Indic CPT：词表扩展收益与英语遗忘必须一起量

- **身份与日期**：arXiv **2610.15373v1**，2026-10-11T05:53:00Z；交叠窗口补录，名单外主题检索发现。
- **方法变化**：在 Nemotron-3-Nano-30B-A3B 上继续BPE，保留旧token ID，将词表131,072→163,072；新输入/输出行按原子token均值初始化，约3B token嵌入warmup（90% Indic、10%英语），再全参CPT。不是新骨干架构。
- **正文证据**：§2–5、Table1、附录B/C/H。8种原生文字语言的平均tokens/word **3.55→2.18（-38.6%）**；CPT约1.27T token、256×H200，广泛训练974.5B、退火194.9B、128K长文约99.86B。退火checkpoint的MILU **40.63→53.52**，但英语MMLU **78.50→75.11**、MMLU-Pro **64.83→55.43**。token更少不等于已实测延迟更低，扩大的输出词表也增加每token计算。
- **值得直接借鉴的故障**：§3报告旧行虽然梯度mask为零，AdamW衰减仍使其8000步漂移约4%，训练loss正常而英语降至近随机；作者改成每步保存/恢复旧行，并在所有TP rank做bit-exact检查。Indic-heavy验证PPL最低选1e-4，而英语保留更好选1e-5，说明单聚合PPL会漏掉遗忘。
- **局限/工件**：无原词表匹配CPT控制，不能把全部benchmark提升归因于词表；英语混入的debug序列不是控制消融。官方下游HF权重另经过SFT/RL，不能用它充当本文CPT checkpoint的原样复现。本轮未核得独立CPT代码发布，机制遵循论文而非代码验证。
- **建议（分析判断）**：做多语/领域CPT时把旧行恒等性和旧域保留分开列门禁；先复现便宜的row-freeze检查。来源：[历史](https://arxiv.org/abs/2610.15373)、[正文与表格](https://arxiv.org/html/2610.15373v1)、[论文链接的下游模型](https://huggingface.co/gnani/gnani-evon-v3.3-30B-A3B)。

### Repeated-Data Pretraining：batch、重复次数和正则应共同选

- **身份与日期**：arXiv **2610.15965v1**，2026-10-11T22:42:39Z；交叠窗口补录，arXiv官方列表补入。
- **问题/方法**：独特数据有限时，单遍预训练batch法则不一定适用。联合扫batch、epoch、LR、weight decay，并比较固定packing、每轮shuffle、每轮重新packing，加入weight tying与embedding正衰减；把可用GPU和训练时限分开讨论。
- **证据**：§3–4、Figure2/Table1、附录A/B。dense decoder-only的153M/474M/1.5B，DCLM-Baseline、seq4096、独特池0.333B/1B/3B，AdamW+WSD。1.5B/1B独特token池，分别优化LR/WD/停止点后，batch512→64条sequence时验证CE **3.124→2.977**；选择batch64即约0.262M token/update，最佳观察点20轮。并非同optimizer step或同token预算比较；这是各batch的验证选择前沿。进一步的重packing+embedding配方会移动最优重复次数/batch。
- **关键边界**：小batch的梯度噪声起正则作用是与曲线一致的作者解释，非被单独隔离证明的唯一因果。下游accuracy噪声比CE/BPB大。H100的GPU—时限曲线和“70B/10T需22.6年”“额外GPU换2.7–3.75×”含模型/模拟外推，**不是对应规模端到端实测**；正文未给足原训练设备明细，不能给出生产GPU-hours承诺。结论限dense、1.5B以内和特定过滤语料。
- **实现状态/建议**：论文引用EMO/OLMo-core底座，本轮未逐机制核其实验配置与代码，不称已复现。建议有限高质量数据项目以验证loss—实际墙钟—GPU-hours三轴扫参，避免仅用利用率决定batch。来源：[历史](https://arxiv.org/abs/2610.15965)、[正文 §3–6/附录D/F](https://arxiv.org/html/2610.15965v1)。

### Virtual Head Attention：训练时重参数化，换取更少KV头的质量恢复

- **身份与日期**：arXiv **2610.17959v1**，2026-10-13T08:39:20Z；交叠窗口补录。arXiv页面标注NeurIPS2026主会。
- **方法**：同时压缩物理Q头和KV头后，用组内线性Q Premix生成虚拟头、低秩残差Linear Postmix混合输出；论文证明线性模块可分别吸收进Q/O projection，推理化为标准GQA-2，无需专用VHA attention kernel。
- **证据**：Tables1/3/5、附录A。1.7B、50B token、4K，八任务zero-shot平均：GQA16-8-8 **54.09**、压成GQA8-2-2 **52.38**、VHA8-2-2 **54.50**；BF16每层每token KV 4KB→1KB，**计算量估算**training/prefill FLOPs降低7.9%。30B-A3B、500B token、8K、FP8混合精度，12bench四类别宏平均：GQA-4 **47.76**、MLA **49.66**、VHA **50.36**。另一个10B-A1B混合注意力控制实验经500B预训+两阶段各100B长上下文训练后，RULER 8K–128K长度平均 **73.0→84.4**；只将full-attention层KV头4→2，滑窗层不变。
- **局限**：不能把KV缩小2–4倍写成实测端到端速度提升；论文的decode倍数是memory-bound理论推断。HELMET五类别为3胜1平1负，MLA的coding类别也高于VHA。更大模型、20T级训练、RL后训练未验证；论文未提供足以复现工业训练硬件/总成本的完整明细。
- **代码核对**：**PaddlePaddle/VHA@ab7ceded6e1ad6a6ba635afd19a152bc64fe8e2d**（master，2026-10-14T08:25:07Z）。`src/paddlefleet/models/gpt/gpt_layer_specs.py::get_attention_spec`按`use_vha_attention`选择`SelfAttentionVHA`；`src/paddlefleet/transformer/attention.py::SelfAttentionVHA._apply_vha_premix`与`_apply_vha_postmix`分别实现线性扩展和低秩残差。该公开训练路径明确assert TP==1；单卡example不是工业多卡/权重吸收demo。本轮单卡测试文件因GitHub SSL读取失败，未跑GPU验证。
- **建议（分析判断）**：适合新预训练架构小规模等预算对照，先验证权重折叠前后输出等价、TP支持和完整延迟。来源：[版本](https://arxiv.org/abs/2610.17959)、[正文](https://arxiv.org/html/2610.17959v1)、[实现基线](https://github.com/PaddlePaddle/VHA/tree/ab7ceded6e1ad6a6ba635afd19a152bc64fe8e2d)。

<a name="post-training"></a>

## 后训练与 Agent 数据

### SoftwareGym：用运行中 GUI 验证约束合成软件，再用于 CUA 训练

- **身份与日期**：arXiv **2610.12611v2**；v1 2026-10-08T18:31:16Z，v2 **2026-10-14T08:40:10Z**（主窗口）。昨日仅列候选，本轮首次详报。v1 对照已查：核心得分、匹配来源消融、A800 配置在 v1 已有；本轮未识别到足以宣称“v2 新增算法”的变化。
- **问题与方法**：原生软件环境集成贵，浏览器合成软件的交互深度有限。流水线先设计功能规格与差异化 UI，生成可执行应用，再以固定 GUI 用例、checkpoint 和迭代修复检验真实可用性；任务生成只接受有通过证据的功能树叶节点。构造 182 个应用、12 类，作者 GUI 用例平均通过率 91.5%。这不是最终 CUA 成功率。
- **正文证据**：§4.1、Tables 1–3、附录 C.3。Qwen3.5-9B 在 OSWorld **244 个任务子集**上 Pass@3 由 **41.39%→56.97%**（+15.58 点）；SoftwareGym-Bench 的 120 任务上 **25.8%→40.0%**。温度1、100交互步、每题3次独立rollout，不能与其他协议的全量 OSWorld 排名直接混排。训练为语言部分全参 SFT 1 epoch，视觉编码器/merger 冻结，BF16、seq16384、global batch32，SoftwareGym-only 用16×A800；过滤后49,445个序列化样本来自3,715条轨迹。
- **反证与约束**：匹配训练来源消融中，原生 OSWorld-only 在 OSWorld 为61.48%，高于合成-only的56.97%；混合来源为58.61%，也未超原生-only。合成-only在自建bench为40.0%，混合47.5%。所以证据支持“可迁移且互补”，不支持“合成完全替代原生”。自建bench与训练环境同属12类，不可称严格OOD。
- **实现基线与边界**：官方仓库 **MrKiing-Jin/SoftwareGym@1e2e8f072dd33fb20046936c6977929467b2a99c**（main，2026-10-14T06:36:11Z）。已读 `agent/loop_controller/runner.py::PipelineRunner.run_variant_rounds`：GUI cases仅首轮生成、后续修复复用，`overall_status == Pass`可提前停止；`PipelineRunner.run_eval_stage`：通过子进程运行GUI evaluator，非零返回抛StageError，完成后验证summary存在。对应论文“固定测试—修复—重测”的两处代码路径；未部署其VM/GUI依赖，也未审计完整训练工件。
- **价值与建议（分析判断）**：适合借鉴环境工厂和证据门控任务合成；复现优先保留native-only、synthetic-only、mixed三组，使用匹配token与GPU-hours。来源：[版本历史](https://arxiv.org/abs/2610.12611)、[正文 v2](https://arxiv.org/html/2610.12611v2)、[官方实现固定基线](https://github.com/MrKiing-Jin/SoftwareGym/tree/1e2e8f072dd33fb20046936c6977929467b2a99c)。

### TPPO：用累积log-odds critic跟踪前缀价值

- **身份与日期**：arXiv **2610.16794v1**，2026-10-12T13:47:36Z；交叠窗口补录。
- **方法**：标准critic分别回归前缀值，局部变化可能不稳定；TPPO将每位置标量累积成value logit，再sigmoid，用终局监督训练，并用于PPO优势估计。该结构不等于把outcome reward变成外部人工process reward。
- **证据**：§4–5、Table2、附录D/A。约30k Polaris数学题，训练每prompt采8条、最长8192、温度1，slime、每节点8×H200；Qwen3-4B-Base域内聚合 **PPO34.6→TPPO41.6**，8B-Base **37.5→45.9**；对应GRPO35.4/38.5。评测AIME/AMC Avg@32，其余Avg@4、最长24K，不能叫单次greedy分数。OOD平均4B62.8→66.9、8B67.6→69.7。未见总GPU-hours或大规模多seed不确定性足以保证所有收益稳定。
- **边界**：目前理论与实验设置假定二值终局奖励和gamma=1；最大模型9B，dense/nonbinary奖励的推广留待研究。critic诊断MC只在选定anchor/64个prompt估计，有采样噪声，不能当真值。
- **实现核对**：**LINs-lab/TPPO@9b11fef526834f8e25121a50eac6f6901862ab84**（main，2026-10-10T08:07:52Z）。`slime/slime/backends/megatron_utils/loss.py::_values_from_response_logits`在`critic_learn_tppo`下累计anchor+deltas并sigmoid；同文件`value_loss_function`选择logit-space BCE，区别于非累积BCE消融和MSE。`scripts/8b_polaris/tppo.sh`显式启用对应flag。仅静态读码，未跑训练。
- **建议（分析判断）**：已有PPO critic可做小改动消融，至少同时保留PPO、PPO+BCE、TPPO+BCE和匹配rollout成本。来源：[历史](https://arxiv.org/abs/2610.16794)、[正文](https://arxiv.org/html/2610.16794v1)、[固定实现](https://github.com/LINs-lab/TPPO/tree/9b11fef526834f8e25121a50eac6f6901862ab84)。

### DisaggKD：把教师部署位置和loss内存优化拆开评估

- **身份与日期**：arXiv **2610.17644v1**，2026-10-13T03:17:08Z；交叠窗口补录。
- **方法**：教师replica分节点流式发送hidden states；学生以row window和vocab shard计算全词表soft CE/KL，fused统计避免完整logits物化；最多4个slot的prefetch ring调节生产/训练节奏。精确算术下loss/梯度等价，不等于FP8与BF16数值完全相同。本工作用固定语料LM+KD，不是学生在线rollout的OPD。
- **证据**：§5、Table4。每节点8×H100-80GB、NVLink+InfiniBand、FineWeb-Edu、默认seq32768；Qwen3.5-35B-A3B→9B四节点，SQ-L+K同名loss的共置 **96.3s/it**，分离 **69.9s/it**；但教师精度不同：分离方案教师路由专家为FP8、其余保留BF16等混合精度，共置对照为BF16；文档顺序也不同。对Bal-L+K的176.8s主要包含loss balancer额外成本，不能把所有差值归因于分离。把同类fused KD loss移回共置，SQ-L+K fused-W在3节点 **72.0s/it**（model CE fusion off、batch126、仅15步），与分离3节点74.3接近，2节点共置还更快；不是所有条件完全匹配的单因素部署位置消融。
- **质量/局限**：200步附近MMLU跨layout差异与seed42/43的1.3–2.3点变化同量级，无证据说分离部署本身增强能力；只有一类节点、一种英语语料、短跑、主组合两seed。能单节点容纳的pair，分离会多用GPU；有些大pair共置失败涉及dispatcher NaN，不能把软件正确性故障全部称为显存不足。本文未核得独立官方实现链接，机制以论文为准。
- **建议（分析判断）**：先移植/评估fused loss，再按学生状态+教师权重的内存模型决定是否分离；对齐loss、精度、数据顺序、并行网格，报总GPU成本及teacher duty。来源：[历史](https://arxiv.org/abs/2610.17644)、[正文 §4–5/Tables3–4/附录](https://arxiv.org/html/2610.17644v1)。

### 【旧候选补录】ProgramGym：环境数据与Agent工作流的收益应分开

- **身份与日期**：arXiv **2610.13395v1**，2026-10-09T11:09:54Z，昨日待办补录，不在72h范围。
- **方法**：从C/C++/Go/Rust仓库构建reference executable、隔离sandbox和黑盒tests，通过build/test修复loop保可靠性；Program-Agent把analysis/implementation/review隔离，并用git revision、独立编译和last-stable恢复支撑长任务。
- **证据**：§6/Table2、附录D.1。1,648环境、665,512有效tests，构建总成本 **136.77B token**；64,505训练“轨迹”是context切分后的segment，非同数独立任务rollout。Qwen3.8-27B BF16全参SFT2epoch、seq262144、globalbatch32。ProgramBench相同Claude Code框架下，Tests Passed **34.87%→42.84%**；换Program-Agent（最多5轮）base/SFT分别 **55.71%/56.25%**。每任务8CPU、16GB RAM、30GB存储、8小时限时；训练GPU数量/总时长未给全。
- **不可丢的负面结果**：Tests Passed是每任务test pass rate的宏平均，**不是任务完整解决率**；base/SFT配Program-Agent的Resolved都仅 **0.50%**。SFT+Claude Code的Almost Resolved还从3.50%降到1.50%，Resolved0.50%→0。跨旗舰模型对比用了mini-SWE-agent不同scaffold，不能当同条件模型能力排名。
- **实现/建议**：论文写“will release”，本轮未核得公开ProgramGym代码工件。适合作为环境构建和revision管理设计参考；优先追踪Resolved/cheating flag与总token成本，分开SFT和scaffold两轴。来源：[历史](https://arxiv.org/abs/2610.13395)、[正文 §4–6/附录B/C/D](https://arxiv.org/html/2610.13395v1)。

<a name="inference-research"></a>

## 推理优化研究

### 【旧候选补录】DoSA：用保守证书跳过不可能进入Top-K的完整打分

- **身份与日期异常**：arXiv **2610.13990v1**；abs官方历史显示 **2026-09-26T17:26:54Z**，与编号月份直觉不一致；HF精选日10-14不能覆盖官方时间。按官方历史保留为旧候选补录，**不计主窗口或72h新发**。
- **方法**：seed集合的第K分数给真实阈值下界；用保留head、正head非负投影/残差上界和负head聚合下界构造score上界，只完整计算仍可能入Top-K的key，再按原score精确筛选。decode阶段仍全FP8 scoring，主要减少selection，而非同样跳过大部score。
- **证据**：§4、Figures4–8、附录C/D。GLM-5.2、DeepSeek-V3.2，K2048，SGLang0.5.18/LiteTopK；single-request冷prefix cache、TP8，prefill chunk32K/16K，128K–约1M token英文/中文/代码。1M时TTFT相对SGLang分别 **1.86×/1.94×**，相对LiteTopK **1.26×/1.50×**。Indexer microbenchmark最高2.74×不是端到端；decode Indexer约1M仅1.26×。论文仅称NVIDIA GPUs，未列具体SKU，不补猜。
- **边界/实现状态**：仍扫描每个key；短上下文或松界时可能不划算。lossless要求数值保守guard、tie顺序和candidate overflow处理；合法同分Top-K并不保证生成文本相同。官方 **llm-sys/flash-indexer@d0718a89c0d3c9ce2b62d0cdf524e3189ee7a2b9**（2026-10-08T07:22:43Z）只有README/PDF，README明确“Code coming soon”，没有可复现GPU kernel发布。
- **建议（分析判断）**：值得跟踪DSA indexer工程，等代码后先做FP8误差、ties、overflow和实际GPU型号核对。来源：[官方时间](https://arxiv.org/abs/2610.13990)、[正文](https://arxiv.org/html/2610.13990v1)、[官方仓库基线](https://github.com/llm-sys/flash-indexer/tree/d0718a89c0d3c9ce2b62d0cdf524e3189ee7a2b9)。

<a name="core-projects"></a>

## 工业开源项目：核心六项

本节时间均UTC；精确合入分支与SHA逐项列出。测试与性能均为原作者报告，本轮未运行训练/serving。

### 训练与 RL 正确性

#### Megatron：CUDA graph重放后MTP指标可能不再做跨组归约

[PR #7768](https://github.com/NVIDIA/Megatron-LM/pull/7768)，main，2026-10-14T19:58:25Z；SHA 80781ab0aba325514dd43010b23d631aaa8382b9。

- 增量：日志清理原先把reduce_group/avg_group一并设None。eager forward会通过Python重新设置，CUDA graph replay只更新tensor，不重放Python赋值；后续logging interval因此跳过所需归约。补丁清零累计tensor但保留group引用。
- 证据：实际patch只移除group清空并增加回归文件；回归覆盖sum、average及组合归约，eager/graph共三个logging interval。PR未给实跑通过数量、完整训练对照或性能结果，不能写“恢复模型收敛”。
- 意义/建议：监控异常不能只查loss tensor本身；为MTP metric增加多次日志间隔、跨rank一致性回归。优先核查graph训练的指标可信度。

#### Megatron dev：量化激活重计算须恢复forward的FP8/FP4上下文

[PR #7538](https://github.com/NVIDIA/Megatron-LM/pull/7538)，**dev分支**，2026-10-14T11:51:03Z；ec815ea9db6f55a9e25ce5ecb2f1fad6569c8c04。
[PR #8203](https://github.com/NVIDIA/Megatron-LM/pull/8203)，**dev分支**，2026-10-14T17:21:43Z；c3010d6d50364df1e5a1e4f7ed4c2111d4ebbbb8。

- 增量：tensor_parallel.checkpoint在autograd backward里重跑forward，原来只恢复RNG、没有恢复TE量化上下文，可能静默用BF16重算FP8 forward，FP8参数还可丢main_grad而报错。#7538记录FP8 enabled/recipe并重入TE recompute phase；#8203让GDN/KDA的FP8/FP4选择te_checkpoint。
- 证据：#7538在一GB200节点、MXFP8、TE2.20上，DSv3.2形状core_attn重算原第1迭代崩溃，补丁可运行；Qwen3.5形状GDN重算原第2迭代分叉，补丁6迭代逐digit一致。#8203另报告12个GPU回归（内部GDN/KDA `_forward_compute`替换为TE Linear+dropout，验证上下文/RNG边界，未覆盖真实FLA kernel），及四GPU单节点Qwen proxy，覆盖MXFP8/NVFP4/参数gather。都是作者验证，非本轮复现；短proxy不是全规模训练保证。
- 建议：低精度+selective recompute须把实际dtype/recipe及梯度比较纳入门禁。**本次只能说dev已合入，不能写main或正式release已修。**

#### vLLM：attention sink重载要经过当前parameter loader

[PR #57797](https://github.com/vllm-project/vllm/pull/57797)，main，2026-10-14T10:17:59Z；19aba3e4d59baf0bc1c1433cea3ccb3065ed00c9。

- 增量：RL layerwise reload中直接写sink绕过parameter的weight_loader，留下旧值；统一factory/loader，按checkpoint全局head数作TP切分一次、原地写local shard、padding恢复-inf。涉及DeepSeek V4/V4.1、HY-V4、MiMo含MTP/Omni MTP、GPT-OSS、Laguna、Qwen3-DFlash；GPT-OSS仍保留各量化分发。
- 当前证据：当前review revision在Windows/Python3.12/torch2.13 CPU的隔离回归35 passed/1 CUDA skipped；使用真实PyTorch/生产reload模块，但绕开启动import并对无关构造提供替代，不是标准suite/full model。旧revision H20的41 sink tests不能说成当前head已GPU复验。
- 边界：标准pytest启动受uvloop缺失阻塞；无当前head真实checkpoint全模型准确率、TP>2或多节点验证；backend-converted sink copies及GPT-OSS expert reload不在此修复范围。
- 建议：把修改sink值的两次重载、TP切片、padding及storage identity纳入RL权重同步门禁，同权重reload alone可能不暴露旧sink。

#### verl：Blackwell BF16 SGLang rollout默认切Triton，绕过首次权重同步崩溃

[PR #8230](https://github.com/verl-project/verl/pull/8230)，main，2026-10-14T11:58:25Z；e5a627cf55c94d85406556854c2c1a22db4ff5e7。

- 增量：SGLang在sm10x非量化Qwen3-MoE/DeepSeek类auto选择flashinfer_trtllm，expert block layout不能按当前tensor更新路径恢复，首次sync在_load_w13报64/512维度不符。verl仅对CUDA CC10.x、checkpoint/rollout都未量化、runner auto/unset、a2a none/unset时默认Triton。
- 证据：1×B200、随机2层Qwen3-MoE/8 experts、Megatron backend、BF16/GSM8K、4step，base在首次sync退出1，Triton 4/4步退出0；sglang0.5.20、FlashInfer0.6.18、TE2.16.1、torch2.13+cu130。同一有效case在条件收窄前revision测得；最终CPU条件测试经review要求删除，不能说最终diff保留单测。
- 边界：这是一项verl集成规避；作者指出SGLang相关上游PR未合入。量化/a2a/显式runner/Hopper/ROCm保持各自选择，没有吞吐A/B。
- 建议：升级后核实际runner和首个真实weight update，不能只以rollout启动成功判断可训练。

#### torchtitan：masked full-vocabulary logits的entropy保留有限值

[PR #5351](https://github.com/pytorch/torchtitan/pull/5351)，main，2026-10-14T03:55:45Z；8d95f48169a7d917096c86e2db88570d26a99844。

- 增量：全词表路径的0×-inf产生NaN；将有效分布masked项贡献归零，仍对无效分布保留NaN。作用于vocab_parallel_group=None及batch-invariant gather后路径；entropy仍no_grad，不是熵正则反传重写。
- 证据：31个CPU测试；float32/16/BF16含point mass、ignored targets、invalid NaN，48个CPU/CUDA有限输入controls保持logprobs/entropy/加权梯度exact；4个10step DAPO/GRPO组件optimizer controls exact；2rank Gloo probe验证gather。未做端到端LLM、多GPU NCCL、optional batch-invariant CUDA override。
- 成本：作者128×151936 CUDA shape的median1.629→1.840ms，峰值分配三形状一致，不可写成提速。
- 建议：采样/约束解码产生-inf mask后同时检查entropy数值；无效分布仍需报错诊断，不能全局nan_to_num掩盖。

#### torchtitan：KDA L2Norm backward的独立autotuner纳入确定性模式

[PR #5358](https://github.com/pytorch/torchtitan/pull/5358)，main，2026-10-14T03:04:12Z；508125f8e25eefb5475f08228db3b92cb3a39247。

- 增量：Attention Gym0.0.16→0.0.17；torch deterministic algorithms启用时对q/k l2norm传autotune=False。此前chunk KDA schedule固定但L2Norm backward仍会因benchmark/cache选择不同计划。
- 证据：2×B200、FSDP2/EP2、kimi_k3_debugmodel_mm_muon、10步，两个fresh export与既有golden byte-identical；GB300同拓扑三次独立10步loss/grad_norm也一致但不替换B200 golden。20个focused tests，final B200 assert-equal覆盖10个loss和grad_norm。
- 建议：严格数值回归逐kernel盘点autotuner；不能将十步固定拓扑确定性推广为所有设备/长训练bitwise保证。

### KV 缓存、PD 与异步生命周期

#### vLLM P2P两项互补修复：超时撤供与传输终止确认

[PR #59329](https://github.com/vllm-project/vllm/pull/59329)，main，2026-10-14T13:22:29Z；7115b70c0170797cb3f0ed98b7ae761f6fe9bcb3。
[PR #59790](https://github.com/vllm-project/vllm/pull/59790)，main，2026-10-14T15:31:32Z；e2aabc8b1521aad382b439e6be4e740d8e8211eb。

- #59329：lookup已将block绑定round并pin，但30秒store timeout释放pin后，round仍advertise旧slot；晚到fetch可成功读取已复用的其他请求KV。补丁从parked available/pending撤掉已失败job，晚fetch走失败并本地prefill。CPU macOS arm64回归原版明确提交slot7→20传输而失败，修后该case通过，tiering434 passed/1 skipped；不是RDMA实机。
- #59790：producer不能把release_xfer_handle成功或remove_remote_peer当作WRITE已停止。NIXL1.4.1+UCX/TCP/CPU DRAM实测512MiB WRITE取消后5/5仍落剩余字节；1GiB stalled WRITE解除后64/64 blocks仍写入。现在轮询PROC/PEND直到terminal，保留被读store pins，延迟AbortAck；remote-disconnect作terminal failure。作者tiering新head486 passed/1 skipped，旧head484通过不能累计。
- **未闭环边界**：producer无hard deadline；peer活着而WRITE不前进会持续持pin。正文明确consumer仍会在_ABORT_ACK_TIMEOUT_S后释放slot，另跟踪#61532；shutdown immediate fallback也保留。因此只能说修复producer侧提前释放，不能说P2P所有abort竞态解决。本轮静态复核还发现两补丁组合后的pending-job撤供集合与PR正文要求可能不一致，尚未运行复现；组合兼容性仍需回归，不将各自单测当组合认证。无RDMA/IB、GPU memory、serving或模型eval。
- 建议：按“地址可重用”的真正完成点写生命周期测试；同时追踪consumer修复与pin占用、stalled round，不把取消API返回等同硬件已停止写。

#### SGLang：PD decode radix cache按insert路径明确KV所有权

[PR #40857](https://github.com/sgl-project/sglang/pull/40857)，main，2026-10-14T21:19:52Z；d3dc14eb3869adcf2291b40c6903a40a77fc6f76。

- 增量：允许decode radix cache与EAGLE/EAGLE3组合，并修post-insert重新match可能短于实际insert的所有权问题。SWA只保留近期window，EAGLE bigram page边界后移，normal rematch不够全window；cache_protected_len过小可把树仍拥有的page释放回free list。改沿最终insert node→root拿canonical indices并断言完整覆盖。
- 证据：作者gpt-oss-120b、1P/1D、page64、EAGLE3、decode pool65536，单轮500 GSM8K/concurrency16干净，多轮在所有请求HTTP200后idle pool check崩溃；base四次三次1–5round复现。tree counters/sanity check与free list去重都正常，但6full+1SWA页同时被tree和free list拥有；这是实质双重ownership。patch包含针对回归测试，PR没有修后完整准确率/速度结果。
- 建议：上线多轮PD场景要检查idle与回收阶段，HTTP200及cache内部counter一致并不能证明allocator所有权正确；此项无硬件外推。

#### SGLang：HiCache buffer_only的storage信念必须消费跨rank归约结果

[PR #43108](https://github.com/sgl-project/sglang/pull/43108)，main，2026-10-15T00:25:09Z；3c28614f129b4a3b98308d8ba63d02c0b64b2ad1。

- 增量：撤销prefetch时过早用rank-local query更新storage existence belief，TP ranks可能做不同backup决策。现在只在同步hit queue drain更新；某rank没query的-1/无KV verdict不当storage miss，明确0-page仍视为miss。复用既有同步，不加collective。
- 证据：sidecar单测使用真实query/sync方法、stub storage和模拟MIN reduction，区分cancel时peer跳过query与显式miss。未给真实多GPU storage E2E/吞吐；PR页面不同CI状态混合，不写全绿。
- 建议：分布式缓存元数据也要明确“未知”和“确认不存在”语义。

#### SGLang：DFLASH包装backend漏判导致重放stale custom mask

[PR #43054](https://github.com/sgl-project/sglang/pull/43054)，main，2026-10-14T09:15:37Z；df1a089cd39e02ad56eb5dfe5e561a77cfb93cbe。

- 增量：prefill FlashInfer/decode TRTLLM_MHA、默认spec attention mode=prefill时，class-name skip list看不到HybridAttnBackend包内TARGET_VERIFY后端，graph按custom mask capture但replay给None，kernel读旧mask。补丁先解开Hybrid/Tbo wrapper，并在捕获mask而plan无mask时明确抛错。
- 证据：Nemotron-3.5-Super NVFP4+DFLASH/LiLiCorr、DGX Spark GB10、GSM8K5shot，flexible/strict从0.027/0.005→0.923/0.848；decode-mode参考0.922/0.850，接受长度3.75→5.51/参考5.50。这是特定错误配置恢复，不是通用模型能力或服务吞吐提高。
- 建议：可组合backend的graph metadata政策要按实际执行后端解析；重放mask缺失应fail loudly。

#### SGLang：GLM-5.2 PD HiSparse重现揭示“传字节正确”不等于格式一致

[PR #44256](https://github.com/sgl-project/sglang/pull/44256)，main，2026-10-14T12:40:29Z；7b53a134016c23f554647534a3b4f78d4ea055e1。

- 本次只改测试配方：prefill显式flashmla_kv并重新启用suite。Blackwell/FP8 KV默认TRTLLM使用576B/token，而decode flashmla_kv消费656B/token，Mooncake普通MLA按sender item length搬字节不转格式，导致负scale/NaN。
- 证据：4×GB300、GLM-5.2-NVFP4、P TP2/D TP2+attentionDP2、Mooncake TCP0.3.13.post1、FlashInfer0.7.1、HiSparse+EAGLE MTP+graph；原4198token请求开关graph都NaN，布局对齐后同请求及另16请求/c8/64output全部成功。
- 边界：非完整GSM8K/接受率评估，500题、长上下文和原8×B300 RDMA拓扑未重跑；不能写通用PD格式自动转换已实现。
- 建议：接入新PD后端显式核KV dtype、RoPE格式、scales及每token字节布局。

#### torchtitan：多轮KV holder新增可选能力，GPU高压收益待测

[PR #5220](https://github.com/pytorch/torchtitan/pull/5220)，main，2026-10-15T01:03:41Z；13ab79360297ec4eb992507ae7c8a28002a35a90。

- 增量：可选session_kv_holder在prefill刚结束时hold下一turn能命中的attention/GDN state；decode开始GDN状态就可能回free pool，所以等本turn结束才hold太晚。下一prefill替换hold，rollout/group结束释放；按engine watermark+下一步需求推导free floor，LRU释放idle session；空running却有waiting时额外释放防永久stall。TP ranks同序操作。
- 证据：真实vLLM KVCacheManager的CPU probe，full churn后下一turn 48/50 prompt token命中，对照0；80blocks/12held sessions的stall回归修后可admit。CPU suite486 passed/5 skipped。
- **无GPU收益结论**：旧GB300/35B棋类分支KV use52–66%，开关holder hit均93–96%，未触free floor，且用了本stack没有的admission gate。当前branch没跑GPU高压，默认关闭。拒绝partial hash hits、spec decoding；不兼容reset_kv_cache_on_model_state_dict_load=True。
- 建议：先做压力下TTFT/每轮重算量与slot占用对照，再决定开启；不能把48/50 CPU命中率写成GPU加速。

### 性能与新能力

#### verl TPU GRPO：先消除同步等待与编译形状抖动

[PR #8226](https://github.com/verl-project/verl/pull/8226)，main，2026-10-14T03:56:59Z；2bc6ac46043e31855cea194236ff7548c91817c8。

- 增量：Qwen3-4B示例MAX_NUM_SEQS32→256覆盖32×8rollouts；actor/ref microbatch1→4；packed length bucket256→1024，让形状在首步编完。
- 证据：2个TPUv6e-4 slices，4trainer+4rollout chips，GKE spot，示例Dockerfile/plugin main；每种10步、steps2–10均值从约72→32.0–32.8s；分步干预中microbatch4约67s，256seq35.5s，再bucket约32s。最终committed默认运行10/10，20次model compile均首步，mean32.0s。
- 边界：小规模10步reward0.55–0.65不证明长期收敛；plugin main未pin SHA；microbatch4虽能容纳但libtpu有更多可恢复fragmentation allocation retries。512seq+双batch in flight变43.5s，不能把更大并发当通用优化。
- 建议：按当前并行批次核quiesce等待、trainer/ref耗时及编译形状，再匹配实际内存；不能写TPU普遍2倍加速。

#### vLLM GDN：prefill内导出边界state，避免为prefix checkpoint再跑一遍tail

[PR #60050](https://github.com/vllm-project/vllm/pull/60050)，main，2026-10-14T06:49:47Z；badfcdbf919532624b2296eeacb85c6dd9fe2e58。

- 增量：FlashInfer GDN prefill在一次kernel中导出cache boundary recurrent state；所有Mamba layer可checkpoint才开启，align block落64token grid。Qwen3.5-35B示例1056→1088token；Qwen4Exp的PLE short conv无export，仍split、block保持800。
- 证据：Qwen3.5-35B-A3B BF16/1×GB200/MRV2/graphs/prefix caching，8000in/1000out、unique seeds、0%prefix hit；匹配1088block的main716961c基线TTFT c1/c4/c16为188.0/398.4/770.6ms，候选150.8/349.4/694.0（约-20/-12/-10%），TPOT±2%。后续review head有重测但没有同head新A/B，不能混作同表。GSM8K多次噪声达约2pp；支持路径测试含export recurrent/conv及MTP spec slots。
- 边界：仅SM100实测，MRV1配置base自身OOM未验证，Triton/FLA/CuTe/ROCm不变；细checkpoint grid最坏约0.5GiB workspace与8kstep约1ms额外成本。不能说所有Qwen hybrid都提速。
- 建议：prefix benchmark每并发使用不同prompt，避免seed重复触cache而掩盖split优化；核block边界与state恢复。

#### vLLM DSv4.1：候选选择按实际长度，不能把kernel19.8倍写成端到端

[PR #61039](https://github.com/vllm-project/vllm/pull/61039)，main，2026-10-14T20:15:59Z；7c3f7e5500c85aacb72aab50bd1ce05485b1d73a。

- 增量：graph固定1M logits宽度，旧candidate selection每行扫/排序全宽；新固定grid按row end只算有效block，用DeepSelect end并支持bf16 decode logits（DeepSelect且DCP off）。原fp32 fallback保留；处理空行和NaN候选。
- 证据：单GB300，384rows=64requests×6tokens、1,048,576width、2048个8tokenblock；131Kcontext的candidate kernel847.8→42.9us（19.8×），1Mcontext939→254.9us（3.7×）。更贴近服务的DSv4.1-Flash DEP4/GB300/125Kin+8Kout/256随机请求/DSpark5合成接受设置，target GPU busy24.11→23.36ms，medianITL26.52→25.65ms（约3.3%）。E2E用了development tree中同实现文件，unit在PR branch。
- 边界：GSM8K1319题短context，候选层保留全部valid block；92.57%不能证明131K长上下文选择/bf16精度完全无损。长上下文质量需另测；synthetic acceptance不同真实拒绝采样。
- 建议：静态capture容量与实际数据长度分开优化；自己的row/context分布及长文质量优先验证。

#### vLLM DFlash/DSpark：将context combine/anchor纳入draft graph，收益集中host-bound

[PR #59511](https://github.com/vllm-project/vllm/pull/59511)，main，2026-10-14T19:40:53Z；388a610dfb5f462261cc1ec27bc2e3bebc0249ca。

- 增量：fixed-address staging target auxiliary states，capture context combine、LiLiCorr anchor及KV准备；超容量和non-FULL仍eager，staging只FULL时分配。
- 证据：4×GB200、DSv4.1-Flash TP4、DSpark5/synthetic acceptance6、128或8192in/1024out；c1 medianITL约5.66/5.68→5.55/5.54ms，对前后base为2–6%（base本身漂移约5%）；c8/c64无可测差别。单GB200 prologue CPU约0.28–0.35ms/step节省，GPU略增；92tests。真实rejection GSM8K DEP4接受率相近，在数题噪声内。
- 建议：先确认host-bound，再试；不以纯CPU prologue倍率预估高并发服务收益。

#### SGLang Elastic EP：以完整DP-attention replica扩容

[PR #33728](https://github.com/sgl-project/sglang/pull/33728)，main，2026-10-14T17:17:02Z；87ed1b95266c4ccff9a9e5b0e2fb6143f6ecb7f8。

- 增量：移除scale-up要求dp=tp=ep的限制；replica width=attn_tp×attn_cp，API保持physicalEP size、scheduler/DPC/tokenizer改用logicalDP size。固定attnTP2可DP2/TP4/EP4→DP3/TP6/EP6→DP4/TP8/EP8。
- 证据：8GPU manual test eager及FULL decode graphs均通过；分别GSM8K63.28/64.45%，另width1回归62.50–64.06%，37focused tests。正文未提供模型/GPU型号及吞吐，不作准确率对比或扩容效率数字。
- 约束：attentionTP/CP/DCP固定、primary/joiner拓扑一致、targetEP能整除replica width、moe_dense_tp_size=1；已有方向/恢复限制不变，无拓扑协商。
- 建议：运行时扩容目标应以可健康加入的完整replica确定，先测试失败和恢复，不解读为任意并行布局在线变更。

### 其他实质变化与工程约束

- **vLLM移除Medusa/MLP speculator（breaking）**：[#61701](https://github.com/vllm-project/vllm/pull/61701)，main，10-15T00:29:24Z，8530aed2b78a19eb00e242031c22bb78d615eea2。method=medusa/mlp_speculator以及对应checkpoint现在配置验证报错。504passed/11skipped/2个本地DeepGEMM过旧的无关registry失败；GB200/Llama3.1-8B/V1 ngram/eagle3各16prompt greedy smoke输出一致。存量Medusa配置升级前需迁移，不能说其他spec算法新增提速。
- **vLLM依赖升级**：[#52183](https://github.com/vllm-project/vllm/pull/52183)，main，10-14T07:18:00Z，30e79c3a766c628fa005146d713ae72530be2b76。torch2.14.0/torchvision0.29.0、release channel；triton3.8、torchaudio2.11、cu130不变，CPU image恢复cache。PR自己列旧test-channel run中可能torch regressions，并要求新pins重跑；不是PyTorch当天首次发布，不能写CI全部验证通过。
- **vLLM FP8 scale clamp顺序**：[#58032](https://github.com/vllm-project/vllm/pull/58032)，main，10-14T13:56:43Z，b4b7eca0ef35cfcbec85170f894bb4ef1537e3bc。UE8M0 round后再clamp可能把zero/small groups scale改成非2幂，DeepGEMM assert；先clamp后round。GB300/Qwen3-30B-A3B-Base reproducer及edge test修后初始化通过，无端到端性能/质量结论。
- **vLLM KVCR失败启动保留映射**：[#57500](https://github.com/vllm-project/vllm/pull/57500)，main，10-14T12:43:05Z，a68dbbad0524a776b171860cef819ce9251c3b63。native thread仍活着时unmap已注册NIXL区域可造成写入已复用页；配对KVCRStartupError时retain_until_exit，普通error继续cleanup，interrupt保留并重抛。需paired ai-dynamo/kvcr#43；测试kvcr包缺失会skip，未做serving。
- **SGLang HiCache AUXILIARY_SWA**：[#43932](https://github.com/sgl-project/sglang/pull/43932)，main，10-14T23:43:25Z，e7d6a88bfce232fb3441baff1293e9bedeeb37ef。buffer_only L3为额外SWA window建独立pool，只H2D missing rows；修staged admission无界重试。GPU buffer unit136passed/230skipped、Rust rebuild focused通过；CPU四例与base同失败。Rust tree core仍只支持一个window，对辅助组件某些API明确NotImplemented，无E2E性能结果。
- **Megatron min-memory简化DSA kernels**：[#7152](https://github.com/NVIDIA/Megatron-LM/pull/7152)，main，10-14T06:39:18Z，1e3c0c62fbb79c26d083f506f7da46f5f7add49e。简化单head topk(QK)、sparse GQA的Triton/autograd分块topk避免seq² score matrix；这是四个stack的第一个kernels PR，不能说完整DSA训练栈已交付，未给可引用的训练吞吐/显存实测。
- **torchtitan graph PP参数准备外提**：[#5392](https://github.com/pytorch/torchtitan/pull/5392)，main，10-14T18:08:40Z，58a3027d954327f3fade5fea561305684dd45041。对PP+no FSDP把quant/cast等prep提至UNSHARD，避免每microbatch重复；正文无性能数字，宜作实现跟踪。
- **Megatron dev GDN2校验**：[#8198](https://github.com/NVIDIA/Megatron-LM/pull/8198)，dev，10-14T14:27:18Z，d9c876ffa8c7f8a28212ce464c652f4268966131。CP>1时两种linear_cp_mode都要求K/V heads能整除TP×CP，拒绝contiguous/dynamicCP；126config tests含12new，没GPU复现；main原已检查，不可说main新bug。
- **Megatron dev DSA empty row**：[#7434](https://github.com/NVIDIA/Megatron-LM/pull/7434)，dev，10-14T13:49:48Z，ef1693cbf5d29fd33228b540680b4ce449e1cccf。static安全tile+zero dO/+inf LSE/zero dQ消除dynamic nonzero路径，31unique focused cases；不改CSA/THD/CP语义，无端到端训练测量。

### 历史条目更新：Megatron MLA Q-RoPE race 的新核验证据

【更新补证，非新发布】[Megatron #5963](https://github.com/NVIDIA/Megatron-LM/pull/5963)，10-05已合入main，`8f8be7352775741a4503e268d07d338d05618b66`，10-06日报仅一行缺细节。本轮重新打开完整PR，补充机制：forward interleaved→split与backward split→interleaved的多warp原地读写，需要所有source load后无条件barrier。作者单H200、BF16 packed THD、32 local heads、Triton3.5.0/3.5.1，1000次replay中forward749错/backward8错→0/0，去单方向barrier负对照仍错；未做完整模型训练。固定tile/warps性能波动约-1.74%至+0.44%，不声称加速。建议采用融合MLA/RoPE的训练检查重复replay正确性，本次增量仅为证据深化。

<a name="extended-projects"></a>

## 训练与推理工程：扩展项目

本节同时补查上轮未完成的扩展项目。标为【补录】的项目保留原始日期；main/master 的合入不等于进入 release。下列测试、性能和复现均为维护者在原始材料中的报告，本轮未运行 GPU/NPU 训练。对有直接代码核读的条目列出冻结基线与入口；其余机制解释依据已打开的官方 PR 正文。

### TRL v1.15.0：大词表训练移除完整 logits，收益主要落在显存

- 日期/版本：【补录】2026-10-08T19:26:48Z；[正式 release v1.15.0](https://github.com/huggingface/trl/releases/tag/v1.15.0)。本次新增是对上轮扩展项目缺口的核验，不是10月15日首发。
- 增量：SFT、DPO、KTO、GRPO、RLOO 和蒸馏采用按 tile 投影的融合 LM head，直接归约 token log-prob/entropy，避免构建完整的 batch×seq×vocab logits。与仅将 CE 分块不同，它连前面的完整输出投影结果也不保留。原 SFT 默认 `chunked_nll` 已避免该问题，release 表中其可训练长度没有提升。
- 证据/限制：1×B300、allocator 限79GiB，随机 BF16 Gemma-3-1B（262K词表）、batch1、gradient checkpointing、SDPA；DPO 最大长度10,240→59,392，KTO batch2为9,216→63,488。8,192 token时 DPO峰值48.97→12.00GiB，但吞吐仅1.035×。这是B300上模拟80GB容量，**不是H100测量**；GRPO/RLOO用固定随机completion代替生成，只测scoring。未测多卡、真实生成或收敛，不能把“6.9×长度”写成训练加速。
- 迁移与建议：深入阅读并考虑在大词表小模型上复现。`lm_head`上的PEFT adapter会拒绝，需按release检查`modules_to_save`；额外metrics路径可能再跑完整logits前向。原文还打包SAC、vision assistant-only mask等能力，使用前核对具体版本依赖。

### TRL：修复 MoE 辅助损失随 world size 与累积步数放大

- 日期/版本：【补录】2026-10-08T13:03:16Z，main合入 `727a61dcc40b365b00ed5784a890eaf3e866a3fd`，见[PR #7570](https://github.com/huggingface/trl/pull/7570)；v1.15.0 release列入此修复。
- 增量：token归一化的CE需要world-size修正，per-microbatch均值的router aux loss不应一起乘该系数，也不应在累积时直接求和。修后在CE缩放之后添加aux，并在`num_items_in_batch`路径除以累积步数。
- 证据：2×B300，同一个样本复制4份、仅aux项、SGD；1GPU batch4、2GPU batch2、2GPU batch1/GA2，原router更新量4.94/9.88/19.76，修后均4.94。DPO/KTO不走该归一化路径，不受此缺陷影响。
- 建议：受影响的MoE SFT优先核查。改变GPU数或GA后，即使有效batch不变，旧实验也可能隐含改变router正则强度；该证据验证局部更新一致性，不证明长期能力收益。

### TRL：torch<2.11 编译损失的零 loss / NaN 梯度修复

- 日期/版本：2026-10-14T07:22:17Z，main `7e15ebf5f69f867669ada4ad77a0529149126307`，[PR #7688](https://github.com/huggingface/trl/pull/7688)；未确认进入正式release。
- 增量：旧Inductor分析Triton kernel写入集合时，用constexpr/Python float代入scalar；`.to(tl.float32)`抛错后，分析错误地认为所有输入被改写，导致误编译。改为`tl.cast`，同时覆盖chunked logprob和distillation divergence。
- 证据/限制：torch2.10 SFT开启`torch_compile=True`时原版每步loss0、grad_norm NaN，修后匹配eager；kernel编译测试覆盖2.7.0、2.10.0、2.14.1，2.6 fullgraph仍有独立限制。没有长期训练或速度结论。
- 建议：考虑立即复核自己的损失和梯度。单看“进程在跑”无法检测这种编译正确性问题；以eager数值对照为升级门禁。

### TRL × Trainium：保留分块头并减少动态 shape 重编译

- 日期/版本：main已合入[非Triton设备fallback #7669](https://github.com/huggingface/trl/pull/7669)，10-14T16:15:45Z，`7e465a58170e92140627ebf26a6966e5e80b6dc1`；[Neuron自动padding #7724](https://github.com/huggingface/trl/pull/7724)，10-14T20:18:33Z，`b54122a7fae6a61d973717088336294b152c2973`。未确认正式release。
- 增量：每个Triton tile计算增加同签名torch实现，保留tiling/recompute且不物化完整logits。Neuron上不再先按mask选出动态数量hidden states，而是计算全部位置；默认`pad_to_multiple_of="auto"`在Neuron取128，GPU仍为None，减少每个新长度一次编译。
- 证据/限制：维护者报告Trainium2上各trainer能训练，Qwen3-0.6B SFT eager loss匹配1.14.2并快约10%；20-step示例padding耗时130s对260s。短跑包含编译摊销，不能当稳态2×加速。原文未给完整Neuron软件pin/硬件规模；H100上的padding/FA2对照反而可能变慢，不能照搬设备策略。
- 建议：Trainium后训练值得考虑复现，分别记录首次编译、shape数量和稳态step；这里是框架适配合入，不是Neuron SDK本日新发版。

### DeepSpeed：FP32 accumulation 的结果不能再存回 BF16 分区

- 日期/版本：【回溯补录】2026-10-14T01:15:58Z，master `3b809329fb760de6832d02020e156a9870f5cbdd`，[PR #8749](https://github.com/deepspeedai/DeepSpeed/pull/8749)；未确认正式release。
- 增量：BF16模型设置`data_types.grad_accum_dtype=fp32`，ZeRO-2仍可能把归约后的owned gradient partition放入BF16。两rank梯度1.0与1.0078125的FP32平均为1.00390625，写BF16会回到1.0。新路径用紧凑FP32 partition替代旧BF16缓冲；ZeRO-3已有对应dtype语义，不是同一缺陷。
- 证据/限制：1B toy模型、2×H200、ZeRO-2、FP32通信与accumulation、GA4；峰值allocated仍14,336MiB，reserved16,386→14,338MiB/rank。**省的是本配置reserved，不能写allocated下降2GiB。** 本轮未做训练复现。
- 建议：使用高精度梯度累积时深入阅读；通过`safe_get_full_grad()`检查可见梯度，配置字段和通信dtype正确不代表持久存储也正确。

### DeepSpeed：ZeRO-3 保留模型声明的 FP32 routing buffers

- 日期/版本：【补录】2026-10-09T12:43:57Z，master `7b911151ed62b1c2ddfeb887d576c189dfe17679`，[PR #8699](https://github.com/deepspeedai/DeepSpeed/pull/8699)。
- 增量：`zero.Init`构造时会把显式FP32 buffer转为训练dtype，后续原地载入checkpoint已丢精度。`data_types.keep_in_fp32_modules="auto"`读取模型声明，在checkpoint载入前恢复所列buffers的FP32；本PR**不包含所列parameters的混合dtype分区**。
- 证据/限制：真实GLM-4.7-Flash、1节点8×H200、ZeRO-3 BF16，同样输入，保留bias前后首步loss7.1479→6.5999；平均34.7% token的expert集合变化→0。40步后差异进入run-to-run噪声，未测长收敛。已舍入的buffer再upcast不能找回原值，关键是载入前完成dtype处理。
- 建议：MoE SFT/RL优先自查，尤其偏置量级大、邻近值在BF16中合并的模型。不要将这个首步差异当成模型能力提升。

### DeepSpeed AutoEP：非重入 checkpoint 必须重放同一 dispatch 布局

- 日期/版本：【回溯补录】2026-10-12T07:24:37Z，master `9ec13539abaea3bc1c31c29558b4d9dfafeeadbb`，[PR #8731](https://github.com/deepspeedai/DeepSpeed/pull/8731)。
- 增量：DeepEP新dispatch的接收行序由到达时序决定；非重入checkpoint保留forward的autograd节点，却把新行序激活塞回旧handle对应的backward，造成静默梯度错配。新路径在重算中复用原handle，并约束只能匹配唯一live forward graph；保留图与新forward交错且无法确定归属时直接拒绝。
- 证据/限制：4×H100、DeepEP v2、torch2.10/CUDA13、均匀/偏斜routing及嵌套checkpoint的当前针对性6例通过；旧runtime在嵌套负对照的输入梯度相对L2约0.956/0.973，阈值1e-2。8×H100 Qwen3-30B-A3B，EP8、MB4/GA16、S4096的30步loss对照接近；非重入仍慢约0.7%，不声称加速。
- 建议：深入阅读并检查checkpoint与通信的组合。绝对误差容忍度若大于实际梯度，会使“parity test通过”失去鉴别力；纯重入且无嵌套非重入，以及collective backend不受原始布局缺陷影响。

### DeepSpeed AutoEP + AutoTP：ZeRO-3 folding 与真正 TP 层的梯度修复

- 日期/版本：【补录】2026-10-10T14:55:52Z，master `7d77213d9999f41022e1166ffb393fc651ef454d`，[PR #8670](https://github.com/deepspeedai/DeepSpeed/pull/8670)。
- 增量：允许AutoTP与AutoEP在ZeRO-3共享ranks、支持`zero.Init`；更关键的是修复所有ZeRO stage的folding梯度。各TP peer只持部分路由贡献，原来只能在复制参数的后续平均中恢复；真正TP层用本地不完整dx算wgrad就会错。新增MoE输入backward的TP平均，每MoE层增加一次tokens×hidden all-reduce。
- 证据/限制：旧4rank小GLM结构45个梯度中36个偏1–5.5%；新4/8rank、9种TP/EP布局×ZeRO组合共27例通过。8×H200上53B随机初始化截层GLM的loss/norm与4卡unfolded相近，reserved76.9对134.9GiB，但卡数与布局不同，**不是同卡数速度/显存比较**。offload、DeepCompile、expert tensor parallel>1仍不支持。
- 建议：考虑复现并优先对齐每个参数梯度。原测试所有权重都设skip，不能验证真正TP路径，是本次值得借鉴的测试盲点。

### DeepSpeed：分块 CE 与 gate/up 融合是两个独立的内存/算子切口

- 日期/版本：【回溯补录】[分块CE #8653](https://github.com/deepspeedai/DeepSpeed/pull/8653)，10-12T00:55:47Z，master `cd4cb971b43e2f341a2eaec2cd7a6ea81b5da28c`；[gate/up融合 #8733](https://github.com/deepspeedai/DeepSpeed/pull/8733)，10-12T07:33:08Z，`3a656d3ee711483f543909ed59dcbdfd33a8813b`。二者均opt-in，不将不同实验收益相加。
- CE增量/证据：保留输入logits与每token一个FP32 logsumexp，backward重算softmax，避免多个完整FP32词表张量；**仍有原dtype完整logits，区别于TRL融合LM head**。H100，8192×151936 BF16，单算子20.82→2.76ms；16×GPU、Qwen3-30B-A3B、EP16、ZeRO1、S8192的peak allocated62.224→54.439GB。整步1763.6→1752.8ms，95%区间跨零，不声称端到端加速。二阶梯度拒绝，不支持任意loss或vocab-parallel CE。
- gate/up增量/证据：每次拼接两个专家矩阵做一个grouped GEMM，参数/optimizer/checkpoint仍保留独立w1/w3；输入梯度改为一次FP32累积，不能保证跨shape逐位一致。8×H100、Qwen3-30B-A3B、EP8、MB4/GA16、S4096短跑约快1.9%，reserved约多0.4GB；部分新guard复用早期性能证据，未重新计时。
- 建议：考虑在对应瓶颈上复现，先分清显存来自输出投影、CE还是expert中间激活；不要用单算子倍率预测整步。

### DeepSpeed AutoTP：共享 embedding/head 保持同一物理 shard

- 日期/版本：【补录】10-09T08:29:04Z，master `3c6a5ca9540040fd71786fa393ac6108a06ea6d3`，[PR #8680](https://github.com/deepspeedai/DeepSpeed/pull/8680)。
- 增量：原替换遍历先改embedding、后检查lm_head时，可能已看不到weight alias，静默拆开tie。现在先快照tie关系，将embedding沿head相同hidden维分片，二者共享Parameter，再all-gather embedding输出。
- 证据/边界：2GPU三步训练对照包含不均匀hidden、ZeRO0/2、完整共享梯度与权重；相关suite225 passed/12 skipped于3×RTX5090D。`max_norm`、sparse/custom embedding forward等拒绝；row-parallel head仍须归约完整词表logits。
- 建议：采用共享词嵌入的自定义TP布局时深入阅读；同时检查对象alias、完整梯度和checkpoint恢复，不能只比首步logits。普通tied模型通常仍应比较vocab-parallel head通信成本。

### Transformers：专家拿到 all-to-all 完成后的真实 tensor

- 日期/版本：10-14T15:59:24Z，main `0814cc791d61147a3e947b2479c9ca3727a5c6b6`，[PR #49421](https://github.com/huggingface/transformers/pull/49421)。本次实际核读其代码diff及测试。
- 增量：`EpDispatchExpertsParallel.install_forward`中的`ep_forward`在dispatch后、调用`_run_local_experts`前明确`wait_tensor(tokens)`。AsyncCollectiveTensor在等待前可能仍是pending结果，直接按raw pointer启动expert kernel不会自动触发普通torch运算的等待。
- 证据/边界：新增2rank CPU/Gloo回归，dummy experts要求输入是实际`torch.Tensor`；这不是GPU raw-pointer kernel的完整模型认证，也未提供吞吐数据。入口为`src/transformers/distributed/tensor_parallel.py`，回归`test_dispatch_hands_experts_the_received_rows`。
- 建议：自定义EP/expert kernel值得立即核对。10-06正式[v5.19.0](https://github.com/huggingface/transformers/releases/tag/v5.19.0)已引入默认Qwen3-MoE/Mellum token-dispatch并解耦EP/TP大小；本次正确性修复仍按main状态列出，未确认release收录。

### Transformers：MTP 单层调用与权重转换接口开放给外部 drafting loop

- 日期/版本：10-14T16:14:47Z，main `5201fa1b04d176b15cba4d98e39a8682eccb1d21`，[PR #49544](https://github.com/huggingface/transformers/pull/49544)，代码核读`src/transformers/modeling_layers.py::MtpModel` diff。
- 增量：提取`forward_layer`，允许外部框架自行组织drafting循环，并复用每层mask、位置编码、cache及shared post-norm；提供`get_mtp_weights_regex`和`get_weight_conversions`处理checkpoint映射。`input_ids`和`inputs_embeds`必须恰选一个。
- 证据/边界：是可定位的集成接口变化，未给外部vLLM端到端接入或接受率/加速实测；不能写“所有MTP模型现已加速”。
- 建议：维护独立推理runtime的团队深入阅读，减少每个模型复制MTP层逻辑；采用前核对模型自带MTP模式和权重命名。

### FlashAttention：FA2 反向偏移先转64位，FA4 的 SM120 auto 不等于 SplitKV

- 日期/版本：【回溯补录】[FA2 v2.8.3.post2](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3.post2)，10-13T05:41:17Z；回移[PR #2994](https://github.com/Dao-AILab/flash-attention/pull/2994)的release分支合入 `49774b31c8d1172a5b28f72e9aed0f322bf4ccba`。另[FA4 beta35](https://github.com/Dao-AILab/flash-attention/releases/tag/fa4-v4.0.0.beta35)于10-14T08:20:58Z发布，仍是预发布。
- FA2增量/边界：`dq_accum`偏移的乘法在转`index_t`前已int32溢出，按batch×round128(seq_q)×heads×headdim超过2^31触发，可导致非法访问/NaN梯度；8,193个长度16的序列、16heads×128就足够，并非只在超长序列出现。修为乘前转换，不是新attention算法。release还含安全与wheel维护，本文不当性能进展展开。
- FA4增量/证据：[SM120 #2758](https://github.com/Dao-AILab/flash-attention/pull/2758)在auto `num_splits=0/-1`时选唯一支持的1；显式>1仍拒绝。RTX5070Ti本地回归/benchmark通过，B200 post-merge CI通过但不覆盖SM120；不能写SM120已新增SplitKV。
- 建议：FA2大batch/varlen训练优先核查offset边界，FA4消费级Blackwell可考虑验证beta35兼容性；没有本次通用速度提升的证据。

### PyTorch：collective 启动失败也必须撤销 communicator

- 日期/版本：默认分支已存在[commit `8d8507e6e460db3d4e2b4680518fe131579e6112`](https://github.com/pytorch/pytorch/commit/8d8507e6e460db3d4e2b4680518fe131579e6112)，committer时间10-14T23:55:07Z。关联[PR #199429](https://github.com/pytorch/pytorch/pull/199429)的`merged_at`仍null；这里以实际main commit和“Pull Request resolved”消息核验落地，不伪造PR merge时间。未确认正式release。
- 增量：nonblocking NCCL launch等待失败时，`WorkNCCL`尚未创建，watchdog无对象可追踪，comm状态仍SUCCESS；迟到kernel可让后续CUDA同步卡死。补丁在launch异常路径设TIMEOUT/COMM_ERROR并revoke或abort。
- 证据/局限：原始256GPU torchtitan+torchft故障恢复任务出现此挂起；本地未重现完整挂起，只验证到超时后仍SUCCESS。新测试让rank0在其他rank不加入时以500ms超时发起首collective，要求报错、stream排空和reconfigure恢复；没有给完整256卡补丁后重跑结论。
- 建议：有弹性恢复/异步通信的训练系统深入阅读。代码基线入口`torch/csrc/distributed/c10d/nccl2/ProcessGroupNCCL.cpp`，测试`test_launch_timeout_revokes_comm`；不要只监测已创建的Work。

### PyTorch：CUDA graph 在 capture 之前预留 RNG state

- 日期/版本：main [commit `9beae8fa8cb87e424fefc7a43ddf96418286e0d4`](https://github.com/pytorch/pytorch/commit/9beae8fa8cb87e424fefc7a43ddf96418286e0d4)，10-15T01:10:01Z；[PR #193993](https://github.com/pytorch/pytorch/pull/193993) API merge时间未置位，状态以已读main commit为准。
- 增量：RNG首次使用在capture内惰性分配seed/offset，可能在默认stream查询capture stream event，引发`cudaErrorCapturedEvent`。改由CUDAGraph在capture前拥有并预留buffers，首次使用才绑定generator；默认`num_rng_states=8`，超量拒绝而不是capture中再分配。
- 证据/限制：作者editable CUDA build针对性20例、2个integration checks、3个新进程reproducer通过，覆盖checkpoint/RNG/replay与cudaMallocAsync路径；未在正文给GPU型号，未给LLM端到端收益。
- 建议：维护graph capture、dropout或多个generator组合的团队考虑回归，尤其预热不完全的多线程场景。不能把这项修复概括为所有随机性已确定。

### PyTorch：functional collectives 公共入口及旧缓存兼容性边界

- 日期/版本：main [commit `245ba5d859973a90932312d38bcdcf00af1c9cea`](https://github.com/pytorch/pytorch/commit/245ba5d859973a90932312d38bcdcf00af1c9cea)，10-14T04:01:40Z；[PR #200479](https://github.com/pytorch/pytorch/pull/200479)。
- 增量：DTensor/FSDP2/torch.compile常用的私有functional collectives获得`torch.distributed.functional_collectives`公共模块，复用原实现，公开非弃用collective与`wait_tensor(s)`、`AsyncCollectiveTensor`；旧私有入口仍可用。
- 边界：对象`__module__`改为公共路径，新的pickle/compile cache可能无法被旧release反序列化。正文列测试计划，没有完整执行结果，本文不写“全suite已通过”，也不把API公开当通信加速。
- 建议：框架维护者深入阅读并考虑迁移；模型训练集群混版本回滚时单独验证缓存兼容性。

### Triton：大 descriptor shape 的符号扩展，以及 CAS 类型修复

- 日期/版本：main [大shape #12278](https://github.com/triton-lang/triton/pull/12278)，10-14T17:11:13Z，`34fffaf2b63195f6c864e2b866eea5bbd584df43`；[atomic_cas #12270](https://github.com/triton-lang/triton/pull/12270)，10-14T20:22:01Z，`6e55d637d9d6ce5188220b393e42abfce976a9a7`。未确认正式release。
- descriptor增量/证据：pointer fallback把signless i32 shape错误地符号扩展为i64，2^31至2^32之间的维度变负、所有非负访问被mask。改shape零扩展，index/offset仍有符号。A10上2^31±64的原issue用例通过；lit304 passed/2 unsupported。不是所有指针或索引统一改为unsigned。
- CAS增量/边界：本次核读`python/triton/language/semantic.py::atomic_cas`，新增对compare/value按pointer pointee类型校验转换；回归覆盖int/uint32/64及scalar/tensor，防止返回值符号语义错误。PR正文仅引用issue，未提供设备测试执行结果，不据此认定跨设备已认证。
- 建议：大tensor、kernel同步原语或量化低层开发者深入阅读，优先复现自己的边界shape；二者均为正确性修复，无性能倍率。

### TensorRT-LLM v1.4.0rc0：KV 分层能力进入候选版本，兼容性问题须随版读取

- 日期/版本：10-14T16:35:44Z，[v1.4.0rc0](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.4.0rc0)，**预发布**。release正文是本次证据，不把其中更早合入的每条PR都称为当天新增。
- 增量：Mooncake store pool与V2 scheduler preemption第一阶段、sparse KV配置/分层锁与offload发布GPU metadata、MiniMax-M3 NVFP4 KV、GLM-5.3-Flash；同时强化PD写入覆盖与回收顺序。移除KVCacheManagerV2的Python backend，属于接口/架构迁移。
- 证据/限制：release不含可统一比较的端到端性能表。明确known issues包括H100 TP2 streaming/RPC复用MPI worker pool初始化可能挂起、SM120/121 MLA BF16 BMM graph非法访问、SM120 GPT-OSS部分NCCL EP hidden size不支持、tied-embedding Qwen3 attention-DP+TP>1构建失败。不能把新增模型名等价成所有并行组合可用。
- 建议：深入阅读并在相应组合灰度复现，升级审查同时覆盖cache backend移除、并行拓扑和已知故障。下述10-14 main PR未逐项确认被该RC打包，单独标main状态。

### TensorRT-LLM：MiniMax-M3 NVFP4 KV 的瓶颈其实在 CPU 与 graph 覆盖

- 日期/版本：10-14T23:10:13Z，main `d68ea55c1213c60036df8270ea8cae83f5e0aa7d`，[PR #19974](https://github.com/NVIDIA/TensorRT-LLM/pull/19974)。
- 增量：NVFP4 sparse层与FP8 dense/draft层混合page size，旧实现每步为61层重建/复制无人读取的page table；NVFP4的QKV融合也未被piecewise graph capture覆盖。删除无用元数据、cache构建时固定view、补齐capture，而非再降精度。
- 证据/限制：B300 AgentX，对main `a1f562a696`，各臂单次3600s，强制Eagle3接受。TP4/C20/K3的P90 interactivity约+10%、tok/s/GPU仅+2%；其他点interactivity+2.1–4.6%，单次x90噪声±3–4%，小幅结果不稳。38个greedy prompts（含136K prefix复用）token相同；未给可独立重放的完整流量分布。
- 建议：考虑复现。低bit KV读字节更少却更慢时先查host bookkeeping和capture空洞，不用理论带宽直接推断收益。

### TensorRT-LLM：在存活进程中释放/恢复 GPU 物理内存

- 日期/版本：10-14T06:37:50Z，main `8d09d171348330aebc7e50c83120863b71d1d011`，[PR #19016](https://github.com/NVIDIA/TensorRT-LLM/pull/19016)，API标为prototype。本次核读新增`sleep-wakeup.md`、`llm.py::LLM.release/resume/get_memory_status`及worker状态处理diff。
- 增量：按tag关闭新请求并排空在途工作，保存到CPU/pinned RAM后释放GPU物理allocation，保留虚拟地址；恢复时重新映射同地址并按mode恢复/清零/丢弃内容。只有全部parked tags恢复后才重开请求；分布式状态不一致/部分mutation可进入failed。HTTP文本服务另有opt-in认证控制面。
- 边界：这是**同一活进程内的内存checkpoint**，不落盘、不跨进程恢复；GMS管理的共享权重不随普通weight tag释放；KV manager V2下KV-cache sleep被拒绝。无release/wakeup耗时或回收容量的可泛化性能数据，本次未跑多卡生命周期测试。
- 建议：训推轮换或GPU共享可深入阅读原型契约，先测host容量、请求排空与失败后恢复策略；不要把`resume`当作容灾checkpoint。

### TensorRT-LLM：KV lending 的所有权契约比拷贝 API 更值得读

- 日期/版本：10-14T14:37:20Z，main `5f2e6cdc8b6e53ca3de57e3d9047af9afe1c961b`，[PR #19722](https://github.com/NVIDIA/TensorRT-LLM/pull/19722)。本次核读新包`kv_cache/sharing/__init__.py::attach_lender`公共契约与变更清单，未审计约千行lender全部实现。
- 增量：KV manager V2可通过host staging把稳定块借给传输backend，Lease/RegionHold界定可访问期，cache所有权仍由manager保留；`scope`必须编码权重/adapter/量化/attention布局等影响KV字节的配置，token hash并不足够。
- 限制：当前契约拒绝CP/PP、recurrent state、已有KV connector、lossy cold codec等组合；fetch先锁device pages，V2不能驱逐parked fetch，集成方必须为运行请求留余量。不能确认完成的copy会隔离slot，容量可能逐渐损失。未提供吞吐/命中收益，也不能把新API写成现成分布式KV产品。
- 建议：维护外部KV存储/传输层的团队深入阅读，尤其检查scope隔离、completion与lease释放、所有请求parked造成的调度死锁。

### vLLM Ascend v0.30.0rc1/rc2：模型支持与 KV 低比特路径有明确边界

- 日期/版本：【回溯补录】[rc1](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.30.0rc1)10-12T13:46:38Z；[rc2](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.30.0rc2)10-13T15:24:29Z，均预发布。rc2仅修Dockerfile的CANN构建配套，不能当新推理算法。
- 增量：围绕DeepSeek V4.1、GLM-5.3/Flash、Qwen3.8-27B，整合MRV2、PCP/DCP、KV Pool、混合attention与投机路径。TurboQuant 4bit latent KV限定A2/A3和指定SFA/DSA模型；[release分支PR #18048](https://github.com/vllm-project/vllm-ascend/pull/18048) `c312e5a175bf00ff16a3f38f6d203460a9e0d312`补V1 runner的SFA打包cache，DeepSeek V4的compressed路径仍要求V2。
- 证据/限制：#18048为512 latent/64 RoPE构造386-byte fused slot，测试只验证spec/分配/配置约束，作者明确无TurboQuant端到端CI，需相应CANN算子；不能声称质量无损或所有设备可用。release还说明async CPP默认开启的尝试已回滚。rc2升级Dockerfile至`9.2.0_20260930220649`只是镜像配套。
- 建议：昇腾部署可深入阅读支持矩阵；先pin runner、硬件、CANN与量化路径，再验证精度、复用和并发，勿将release的模型名单扩成所有组合承诺。

### vLLM Ascend：三类独立的推理正确性/可用性修复

- **随机投机分布**：10-14T11:02:39Z，main `ba784cc0b844db58fd69d12a1e8b5c35e5b57bf7`，[#18657](https://github.com/vllm-project/vllm-ascend/pull/18657)。accept/reject与residual抽样共用随机域，条件化后产生偏差；target=[0.5,0.25,0.25]、one-hot draft0时拒绝后只出token2而非1/2等分。恢复resample SALT；A3、v0.30.0依赖集31例通过，旧版4个新例均失败。未跑完整serving/性能，固定seed输出可能变化；建议随机spec配置优先复现分布测试。
- **FA3 mixed batch**：10-15T01:12:49Z，main `5e1647cf998a27b3e0f13392a707c9438d51df81`，[#18068](https://github.com/vllm-project/vllm-ascend/pull/18068)。prefill block-table按请求数切片、重置query offset，多token验证启用causal。独立A2 dense-KV七例通过；早先192 token相同的模型测试同时带另一物理page补丁，**不能当本PR单独端到端认证**。strided page另需#18729，继续观察真实模型组合。
- **KV lookup失败挂调度器**：10-14T02:38:00Z，main `34c9170739ccc7ae6b05282febd51de72e9a5e47`，[#18017](https://github.com/vllm-project/vllm-ascend/pull/18017)。server异常漏发REP会让client永久recv；现在handler错误返回0 hit、client1秒超时后重建socket并清理context。作者未在本机跑NPU serving；另有协作者真实ZMQ/CPU报告。不能强制取消卡死的storage handler。建议KV Pool集成者深入阅读超时降级和清理路径。

### vLLM Ascend：混合页复用时恢复静态 V scale

- 日期/版本：【补录】10-07T13:13:55Z，main `9d56044a9bc1c7629fc52354d799ff534bf38008`，[PR #17956](https://github.com/vllm-project/vllm-ascend/pull/17956)，release rc1也列入。
- 增量：GDN/Mamba与full-attention复用物理页，SSM写入可覆盖初始化在同页的C8 MXFP8 V scales；后来分配回attention时只清零而未恢复checkpoint scale，产生NaN。补丁保留各层view，对新分配页恢复V scale。
- 证据/限制：Qwen3.6-27B W8A8+C8、TP1/eager、max_len4096、max_seq4、无prefix cache，112/112顺序/并发请求通过；同模型非C8为32/32。NPU活动测试在较早相同runtime改动树，rebase后未重跑；graph、TP>1、MTP、全数据集精度不在验证范围。
- 建议：混合架构的cache复用可考虑复现，至少覆盖“首请求正确、后续独立请求错误”和跨cache-group归属转换；不要只做一次warmup问答。

<a name="hardware"></a>

## 硬件软件协同

### AMD：GLM-5.3 的 DP8+EP8 长上下文吞吐提升伴随更高 TPOT

- 机构/日期：AMD ROCm Blog，2026-10-14新文章；脚注测量为2026年9月。[原文及完整配方](https://rocm.blogs.amd.com/artificial-intelligence/glm53-mi300x-vllm/README.html)。Web打开失败，本轮对官方URL直接HTTP GET成功，已读取正文与性能表。
- 增量：1节点8×MI300X 192GB、GLM-5.3 FP8权重/KV、vLLM0.28.0、ROCm7.2.3，比较TP8与attention不分片、expert跨8卡的DP8+EP8。后者每rank保留自己的MLA/DSA cache，服务窗口202,752并非原生1M都已验证。
- 关键对照：128 prompts、max concurrency128、synthetic random tokens、无prefix cache，96K输入/32K输出总output tok/s431.8→1065.6（+147%），但median TPOT26.8→62.7ms；8K/1K则1531.9→1125.6（-26%），TPOT66.4→81.2ms。吞吐提升不能解释成逐token时延改善，也不是混合流量trace。
- 复现限制：配方依赖写作时仍under review的AITER #4365/#5563补丁、关闭特定FP8/FP4 BMM、V2 runner与较长JIT超时；未固定所有组件SHA。尤其博客称patch anchor缺失会失败，但所给Python脚本若anchor不符会打印后正常return，Dockerfile仍可能记录applied；这是本次对可见脚本的核读发现，复现时应明确断言补丁实际生效，不能只看构建成功。
- 建议：考虑复现长上下文拓扑实验，带上TPOT约束评估；该文章证明指定负载的布局权衡，不证明DEP普遍优于TP或upstream当前默认即可运行。

<a name="evaluation"></a>

## 评估与可靠性

### 【旧候选补录】Thinking Inertia：关闭thinking不能替代输出行为检查

- **身份与日期**：arXiv **2610.11765v1**，2026-10-08T11:49:37Z，昨日待办补录，非72h新发。
- **方法**：把响应规范化为pre-answer T和答案A，分别量严格answer-only的ETR、问题与T相关度QRel、盲judge判断可见推理的EIR；短输出、缺thinking标签或相关文本都不能独立当作“不推理”的证明。
- **正文证据**：§3–4、Figure6、附录D/TableD.3。6模型（Qwen3三规模、DeepSeek-V4-Flash、GPT-5.4、Gemini-3-Flash），BoolQ/StrategyQA、MMLU/MMLU-Pro、GSM8K/MATH，6提示干预，能支持时采用确定性decode。全语料开放题EIR从native think-off **99.69%**降至strict answer-only **62.00%**；五个完全可关闭thinking模型的ETR均值近0→约40%，但accuracy均值约降15点。Qwen3-32B在同一组numeric题改写时，MCQ→Boolean的ETR10.5→62.5%，accuracy77.8→78.9%；改开放题accuracy52.6%，ETR0%。主judge GPT-5.5，600例人工审计平均二分类一致98.8%；另一family judge作稳健性检查，仍是作者校准。
- **边界**：只解释可见文本，不证明内部计算消失，更不证明延迟或billing下降。QRel是相关度辅助量，EIR依赖rubric/judge；多语、工具、多轮和随机采样泛化未验证。API endpoint会变化，不将历史分数当实时产品行为。
- **实现核对**：**thinking-inertia/code@8f8f1cddcea6560df672fb70e83f013ce9f530e3**（main，2026-10-09T05:27:45Z）。`src/thinking_inertia/eval_thinking_spectrum.py::_provider_payload_controls`分别映射Qwen/DeepSeek/OpenAI/Gemini的控制项；`src/thinking_inertia/metrics/eir/judge_pairs.py::judge_batch`只把id/question/pre_answer_text送judge，隐去模型与最终答案；`judge_rubric.py::evidence_is_verbatim`核证据span。git clone SSL失败后以GitHub API取得SHA、raw固定版本补读；未调用模型跑评测。
- **建议（分析判断）**：需要answer-only的业务将格式遵循、accuracy、可见推理、实测latency和reasoning usage分开验收；不要只靠开关。来源：[历史](https://arxiv.org/abs/2610.11765)、[正文](https://arxiv.org/html/2610.11765v1)、[固定代码](https://github.com/thinking-inertia/code/tree/8f8f1cddcea6560df672fb70e83f013ce9f530e3)。

### NVIDIA × Siemens：用最终工件验证器改进 Agent harness

- 日期/版本：2026-10-14，[NeMo Gym 官方案例](https://developer.nvidia.com/blog/optimizing-agent-reliability-and-tool-calling-efficiency-for-chip-design-using-nvidia-nemo-gym/)，新工程披露，无新模型权重版本。
- 增量：评价最终Liberty工件的覆盖、合法性、相对验证、数值容差与完整性；任一硬门失败reward归零，再用rollout诊断改skill/MCP描述与JSON schema。固定Claude Opus4.8/Claude Code，**没有证据表明本案例更新了模型权重**。
- 证据/限制：背景数据229 cells×36 PVT，但每轮调试仅用2 cells×36 corners、约25 rollouts、每轮1.5–2h。作者报告composite score约55%→95%、turn155→53、输出81K→29K；不是229-cell全量成功率，也未给独立留出集/置信区间，无法排除对开发任务的适配。
- 建议：深入阅读验证器设计，特别是防缺失/复用旧值的integrity门；可复现自己的工程工件评测，但不将“95%”当通用Agent可靠性或RL训练增益。

<a name="coverage"></a>

## 来源覆盖与未完成范围

以下只说明实际检查入口与覆盖程度，不能据“未检出”推断某机构没有发布。研究/模型工件、开源清单与重点原文审查相互补充；历史日报仅用于去重，不能替代本轮证据。

| 类别 | 实际检查 | 尚存边界 |
|---|---|---|
| 核心六项目 | Megatron/vLLM/verl/SGLang/slime/torchtitan：主窗+72h默认分支629条、旧10-05→10-08缺口434条，均分页取尽；另查主窗所有分支187条merged与各项目前20个release；38个已合入PR全文、18个PR files/patch，额外核未合入#61530 | 清单完整不等于1063条全部技术审计。旧缺口仍是重点选择性审查；vLLM requant/MRV2采样/ROCm、SGLang Qwen3.8 CP/DeepSelect及NPU/XPU等标题候选待细读。slime当前两条docs不收，旧已报修复不重报。 |
| 扩展开源 | PyTorch/Transformers/Triton从10-11 10:34起，DeepSpeed/TRL/OpenRLHF/FA/TRT-LLM/vllm-ascend从10-05 08:00起，共973条默认分支提交分页取尽；主窗merged搜索186条，release入口11仓各前6条；32个重点PR、5组patch、3个PyTorch实际commit另核 | 前三仓10-05→10-11旧缺口未补完；提交列表是committer时间，PR可能走ghstack/支线，未以单一字段宣布全部合入完成。其余候选保存在本地，不从标题猜机制。 |
| 模型厂商 | HF官方组织创建/修改两路，扩至23组织modified前100并另查具体工件；实际开OpenAI/Anthropic/DeepMind/Meta/Microsoft/NVIDIA/DeepSeek/Qwen/Seed/Kimi/智谱/MiniMax/腾讯/百度/阶跃/小米/美团/Mistral/xAI/Ai2等官方站点、API或定向检索 | HF不覆盖所有闭源模型。Meta/NVIDIA/腾讯动态研究页受限；Qwen列表未完整恢复；MiniMax API陈旧与IR不一致，不能标完成；Step FP8 raw/commits401。Seed动态列表已部分打通，但非全部火山渠道。 |
| 论文/学术 | HF10-12至10-15共128唯一候选；arXiv cs.CL/new可见642条按标题筛93；15全文候选中9项详核；中英文主题检索补名单外CPT/架构/蒸馏；5个官方repo固定SHA静态核对 | 不是全量arXiv/cs.LG/cs.AI/版本更新扫描。OpenReview、ACL Anthology、PMLR为主题搜索，未全量枚举正式会议；VRSD、多Agent信息奖励、Jeva等候选待详核，Ai2 Bolmo/Nature旧访问缺口未补齐。 |
| NVIDIA/AMD | NVIDIA博客首页、NeMo案例；NCCL/cuda-python releases；AMD posts及GLM-5.3部署原文完整脚本/表格。AMD Web失败后直接官方HTTP成功 | 未遍历所有CUDA/ROCm组件。论坛10-13 activity曾指向3月Vera旧文，回原文后排除，不能把活跃日当发布日期。 |
| Intel/昇腾/TPU/AWS | Gaudi bridge/JAX/NCCL/Neuron/NxD/NxDI release各前6；TPU与Neuron release notes；昇腾产品公告与vLLM插件；AWS ML入口及相关主题检索 | Gaudi确认10-07 v1.24.2-481 wrapper但正文notes不可读，旧latest路径404，未报猜测增量。CANN/HCCL/GitCode本体、XLA/libtpu、NKI等组件提交未全扫。设备相关有效增量从TRL、verl、昇腾插件补入。 |

**补录与更正。** 上轮DoSA、SoftwareGym、ProgramGym、Thinking Inertia待办已阅读全文；DoSA编号月份与官方9月26日时间异常同时保留。SoftwareGym v2有主窗修订，但核心结果v1已有，不称新算法。DeepSeek cache与MiMo processor旧缺口已补读；#5963仅深化旧证据。TRL/DeepSpeed/FA/Ascend等早于主窗的内容均保留原日期与补录标识，不算当天首发。

**失败与未完成区间。** 不能关闭的历史区间仍从10-05开始逐来源保留；本轮新窗至10-15 09:55:47的清单/发现已检查，但未详核候选和动态/访问失败来源继续待查，不设置虚假的“全局完全覆盖截止”。下轮从该事件筛选截点续扫并回溯72h，同时优先处理未完成区间。Intel版本说明、MiniMax当前内容源、Meta/腾讯动态内容、Step门控工件、会议全量与Bolmo原文均在待办。

**持久化与归档。** 本地 `.monitor/` 分别保存运行状态、来源清单/逐项结果、稳定ID/版本、失败/待补区间、正文状态及远端同步回执。上传仅含本日日报与日期索引；本地生成成功和远端回读成功分开登记，上传失败会保留待同步条目。本期没有改写llm-knowledge的wiki。

## 本日校订记录

- 首次发布前交叉审阅补充：DisaggKD共置fused-W的batch126、15步和model CE fusion off，及教师混合精度差异；不再暗示完全同条件的部署位置消融。
- 补充P2P双补丁组合尚需回归、Megatron量化回归替代内部compute的范围、Step 5历史上传/门控/文档冲突。未用作者单次烟测包装当前分支完整认证。
