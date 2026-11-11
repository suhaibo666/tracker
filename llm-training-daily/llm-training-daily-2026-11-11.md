# LLM 技术动态日报 · 2026-11-11

> 时区 Asia/Shanghai（UTC+8）。增量扫描窗口 **2026-11-10 08:58:40 → 2026-11-11 09:00:49**，向前回溯 72 小时至 **2026-11-07 08:58:40**。实际生成时间 **2026-11-11T09:18:55+08:00**。日期未披露到时分时保留原日期。首次发布、工件或代码更新、会议报告与本轮发现分别标注。
>
> 生成前核对本地 35 篇历史日报及索引，与 GitHub `main` 的 `3e0f8fc82f5918a256af458931f7eea3cee821ed` 逐字一致。前次待上传队列为空。
>
> 本轮核验公开的一手报告、工件、发布记录及部分代码补丁。性能与能力指标为原作者或项目方的口径。

## 本次重要变化

- **模型与工件。** Gemini 4 Carbon 扩大付费 API 可用范围。MiniMax H3-Image 公开六步图像生成与编辑权重，部署需要另行加载 H3 文本编码器。DeepSeek-V4.1-Flash-1017 更新对话模板，Granary v2 大幅调整已发布的数据。
- **预训练与后训练。** 801-30B-A3B 披露 11.8T token 训练前的两轮试跑和数据诊断。OPD 论文表明，训练端与 rollout 的失配预算需要按模型、量化和训练进度校准。
- **推理系统。** Muse Spark Open 把单步 62 层解码合为 TPU Pallas 程序，低并发延迟改善最大。ATOM 为 MI355X 与 gfx1250 分别接入 FlyDSL 解码路径。vLLM 新增跨进程编译缓存复用与异步 KV 加载限流。
- **训练与部署。** Transformer Engine 2.21.0 扩展低精度和编译路径，Megatron 增加固定归约树以对齐不同 EP 布局，verl 的短程对照量化了 MXFP4 rollout 的训推差距。昇腾 HCCL 撤回了前日报记录的一项能力协商提交。

[模型与工件](#models) · [研究论文](#papers) · [开源工程](#engineering) · [硬件与软件协同](#hardware) · [覆盖与续查](#coverage)

<a id="models"></a>
## 模型与技术报告

### Gemini 4 Carbon 开放付费 API，长输出与思考共用上下文预算

**Google DeepMind，11 月 11 日 00:00（北京时间）宣布正式开放。** 稳定模型 ID 为 gemini-4-carbon，基于 Gemini 4 Argon，接受文本、图像、音频、视频及 PDF，输出文本。输入、思考与回答共享 1,048,576 token，默认同步输出上限 65,536，long decode 仍受总窗口约束。思考默认为 medium，Carbon 在复杂任务上会比 Argon 花更多思考 token。11 月 30 日前每百万输入／输出 token 为 2／10 美元，随后为 4／20 美元。Google 引用的 CursorBench 4.0、WANDR 和 AutomationBench 结果属于特定评测配置。可用同一长任务回放 Argon 和 Carbon，按完成率、思考 token、P95 延迟和成功任务总成本决定路由。 [首方公告](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-4-carbon-now-available/) · [API 更新日志](https://ai.google.dev/gemini-api/docs/changelog#11-10-2026) · [模型卡](https://deepmind.google/models/model-cards/gemini-4-carbon/) · [迁移指南](https://ai.google.dev/gemini-api/docs/whats-new-gemini-4-carbon)。

### MiniMax H3-Image 以六步直接生成约 16 MP，另发 NVFP4 提示词扩展器

**MiniMax，11 月 10 日 22:50:23（北京时间）上传主模型，23:00:58 发布中文公告。** 模型沿用 H3 的文本编码器，移除音频分支，将文本与参考图表示合入一个序列，用 DiT 预测目标图像潜变量。六次去噪无需 classifier-free guidance，可直接输出 4096×4096 等约 16 MP 尺寸，编辑最多支持 16 张参考图。主仓约 68.65 GB，另需约 66.7 GB 的 H3 文本编码器，文生图可选 17.9 GB 的 NVFP4 PE。北京时间 11 日 00:21 的提交追加 Diffusers 格式。厂商公告的十秒内是其测试口径，部署可分别测含 PE 的总延迟、显存峰值及参考图遵循。当前社区许可对年收入超 100 万美元或面向第三方提供托管服务要求事先书面授权。 [中文公告](https://www.minimax.cn/blog/minimax-h3-image) · [固定模型卡](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/fbde899fffb5c5225ee8dd7027680113746af42e/README.md) · [固定权重目录](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/tree/fbde899fffb5c5225ee8dd7027680113746af42e) · [Diffusers 提交](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/commit/492fd8c4eca666f1085f57e30738c1e0d86b3439) · [固定 PE](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE/tree/35603d4b63949afa12cb098227fffefedf081afa) · [许可 §IV](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/fbde899fffb5c5225ee8dd7027680113746af42e/LICENSE)。

### Gemini Robotics 2.5 向合作方开放，工厂试点用两个模型分管任务和动作

**Google DeepMind，11 月 10 日 23:00（北京时间）宣布。** Apollo 2 工厂配料试点由 Gemini Robotics ER 跟踪流程，2.5 VLA 控制手臂和手，根据货箱状态完成抓取、排序和补位。DeepMind 报告在奔驰两处生产线的该项试点成功率为 96%，同时评估了约 200 项灵巧操作。训练中用 Gemini 4 Argon 辅助从实景生成模拟练习，与真机示范一起使用。公开公告只确认向机器人合作方开放，试点百分比未给出样本量或完整停机分布。迁移评估可分别统计任务跟踪错误、抓取失败和人工接管。 [首方发布及工厂试点](https://deepmind.google/blog/gemini-robotics-2-5-accelerates-real-world-physical-intelligence/)。

### NVIDIA 开放 Alpamayo 2 Edge，2B 模型同时输出驾驶轨迹与解释

**NVIDIA，模型卡确认 11 月 10 日开放，11 月 10 日 14:56:17（北京时间）更新固定模型卡。** 以 Cosmos 3 Edge 的 Nemotron 视觉语言骨干配合扩散动作解码器，读取六个摄像头的 22 帧、导航指令和 20 个历史位姿，输出 6.4 秒内每 0.1 秒一个航点。厂商报告 AlpaSim 的 875 个场景各跑三次，平均每 15.15 公里一次责任事件，开环 1,392 个样本的 5 秒 minADE₆ 为 0.52 米。模型权重采用 OpenMDW 1.1，推理代码为 Apache 2.0。仿真结果的适用范围取决于场景分布，可按候选轨迹数记录延迟，并分开复核开环轨迹误差与闭环事件。 [固定权重及模型卡](https://huggingface.co/nvidia/Alpamayo2-Edge/tree/b92b6aa326bd3e2235db28f248ad5a30b5890e64) · [官方实现](https://github.com/NVlabs/alpamayo2-edge)。

### 蚂蚁灵波 LingBot-Touch 让三类触觉传感器共用编码器

**Robbyant，11 月 10 日对外宣布，Small 与 Tiny 工件于北京时间 11 月 10 日 01:31:55 与 01:32:08 提交。** JEPA 训练用模态专用 tokenizer 将磁式、压阻和视触觉信号送入共享 Transformer，靠潜空间预测学习接触的时空结构。项目报告用 4,983 小时录制预训练，在两种灵巧手上以 26 项任务、每项 20 次对比原策略、直接触觉输入与冻结的触觉特征。Apache 2.0 权重包仅含冻结目标编码器及匹配配置，后训练需自己接任务策略。更换传感器时应先核对几何、采样点顺序和量程，再测相同策略的成功率。 [首方项目](https://technology.robbyant.com/lingbot-touch) · [论文 v1](https://arxiv.org/abs/2611.12806v1) · [Small 固定工件](https://huggingface.co/robbyant/lingbot-touch-small/tree/38cca106c5a989a7bb973d20b2d65027b4b2a2b4) · [Tiny 固定工件](https://huggingface.co/robbyant/lingbot-touch-tiny/tree/f23969ccd4b6bba2163e69cc05a577455a075fe9)。

### 补录｜GR00T 2 联合预测未来视频和机器人动作

**NVIDIA，11 月 9 日 22:00:01（北京时间）在 CoRL 公告。** 一个扩散 Transformer 联合去噪预测未来画面和动作块，小型机体专用头负责映射到执行机构。用 50 个目标机器未练过的任务评测，官方项目报告零样本成功率为 40%。预训练数据的人类视频从 0 增至全量时，72 个 Dexter 下游任务的真机任务进度从 50% 升到 70%。项目页给出可查的人类视频子集和白皮书，GR00T 2 自身公开权重待核。可在固定机器和初始状态下分开测新技能迁移与执行可靠性。 [CoRL 公告](https://blogs.nvidia.com/blog/robotics-simulation-advancements-corl/) · [项目与试验口径](https://research.nvidia.com/labs/gear/gr00t-2/) · [白皮书](https://research.nvidia.com/labs/gear/gr00t-2/assets/gr00t2_whitepaper.pdf)。

### 补录｜Sophon-Realtime v1 复用语音理解骨干，分出三种语音生成路径

**Adventists AI，11 月 8 日 16:14:32（北京时间）提交家族模型卡。** Tag、Flash 与预览版 E2E 共用理解骨干，分别采用表达优先的音频生成、低延迟路径和端到端研究路径。Tag 与 Flash 可通过 vLLM-Omni 的同一接口接入，支持连续监听、抢话和简短附和判断。作者报告 VoiceBench 82.8、Big Bench Audio 81.8%，其工具调用得分取多次测试中的最好一轮。复测时可固定噪声、并发和打断频率，同时比较首音延迟、误打断及工具调用成功率。 [固定家族卡](https://huggingface.co/adventists-ai/Sophon-Realtime-v1/blob/2e52bd4ade5ab9353a387de0b8930d305b1aff88/README.md) · [首方运行代码与权重入口](https://github.com/adventists-ai/sophon-realtime)。

### 已有模型的工件变更

- **DeepSeek-V4.1-Flash-1017，11 月 10 日 10:22:56（北京时间）。** 从仓内 Python 编码器移植成 275 行 Jinja，支持思考开关及别名、推理档位和 reasoning_content，工具调用参数要求为字典。提交只增加 chat_template.jinja。可用相同消息、工具调用和历史思考比较 Python 与 Jinja 的序列化结果，再固定版本上线。 [固定差分](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/commit/08c8657f02291e11b6f6b63e0e16014f8e748518)。
- **Ripple-65B Preview，11 月 10 日 17:31:23（北京时间）。** 单文件模型实现改动覆盖预热 KV cache 的新 token 块、补齐到容量的 attention mask、训练与张量并行处理。采用远程模型实现的部署可用左补齐批次与热／冷缓存做 logits 对照。 [固定差分](https://huggingface.co/ricursiveAI/Ripple-65B-Preview/commit/b172323ffd63e62de9ce1ba3830dfffe493c7671)。
- **LongCat-Next，11 月 10 日 17:43:01（北京时间）。** 仓库增加离散视觉 tokenizer dNaViT 的技术说明、图与四个占位目录。现有多模态模型用语义编码器加残差向量量化表示视觉输入。此次差分为技术文档更新，相关权重待核。 [固定差分](https://huggingface.co/meituan-longcat/LongCat-Next/commit/567781844d14891ac173c320f702e8052dd85bc7) · [基座模型卡](https://huggingface.co/meituan-longcat/LongCat-Next/blob/567781844d14891ac173c320f702e8052dd85bc7/README.md)。
- **K2-Horizon-0.9B，11 月 10 日 21:17:26（北京时间）。** 配置 dtype 从 float32 改为 bfloat16。使用自动 dtype 的客户端可复查实际加载精度与显存。 [固定差分](https://huggingface.co/IFM/K2-Horizon-0.9B/commit/fc5c9dc7c13df2360acee789fca2cd55d439e644)。
- **ZONOS3-Mini，11 月 10 日 10:58:41（北京时间）。** 新增完整解码器的 Web 分片。可分别比较两种解码器的冷启动下载体积与首音延迟。 [固定差分](https://huggingface.co/Zyphra/ZONOS3-Mini/commit/eaac314642440e8f1f497f1505ca376f8e618f2a)。
- **GR00T N1.7 六套检查点，11 月 11 日 06:10–06:12（北京时间）。** 厂商把许可改为 NVIDIA Open Model License 并增加 Cosmos 归属，涵盖基础、H、DROID、LIBERO、Fractal 和 Bridge。 [基础版许可差分](https://huggingface.co/nvidia/GR00T-N1.7-3B/commit/373d34e2723d928870157d07dc216a19a624b5bb)。

<a id="posttraining-models"></a>
## 后训练与数据工件

### Granary v2 剔除训练流水线此前标记的 574 万段语音

**NVIDIA，11 月 10 日 19:29:32（北京时间）发布修订。** 首次公开的清单包含流水线已经标记 _skipme 的 5,740,042 段，公开 schema 没有标识。此次替换全部清单并剔除 43,088 小时，标称规模由 463,062 降至 419,974 小时。NVIDIA 表示其他行逐字节一致，保留旧版本。复现实验宜固定数据 SHA，并在比较旧结果前统计受影响的语种和样本分布。 [固定提交与说明](https://huggingface.co/datasets/nvidia/Granary-v2/commit/f835695fb5cc2d7e9feb0bf8c048bde1f0ccf877) · [修订数据卡](https://huggingface.co/datasets/nvidia/Granary-v2/blob/f835695fb5cc2d7e9feb0bf8c048bde1f0ccf877/README.md)。

### SenseNova RoboRSI 公开两套冻结的视觉语言导航经验包

**商汤，11 月 10 日 15:49:13（北京时间）提交 v1.0.0。** LHPR-VLN 提供一套共享训练／探索数据，VLN-CE 的 R2R 和 RxR 提供两套，供评测框架加载为上下文。包包含完整性清单和分组件许可，仿真器环境与评测镜像分别交付。跨版本复跑可同时锁定数据 SHA 和模拟环境，再比较有无经验包的导航结果。 [固定数据卡](https://huggingface.co/datasets/sensenova/SenseNova-RoboRSI-Packages/blob/1ef6e2be44e6625ef5ef95d462275854a26aae8a/README.md) · [提交与清单](https://huggingface.co/datasets/sensenova/SenseNova-RoboRSI-Packages/commit/1ef6e2be44e6625ef5ef95d462275854a26aae8a)。

<a id="model-evals"></a>
## 评估与科研 Agent

### MotionGraphicsBench 用逐帧 SVG 检查动画是否满足文字要求

**Figma 等，11 月 10 日 13:01:26（北京时间）提交数据和脚本。** 100 道题各有静态 SVG、动作指令和确定性程序，程序按帧验证几何、位置、时间与事件关系，允许不同实现满足同一要求。32 个模型的 47 个设置中，最高 pass@1 为 GPT-6 Astra Max 的 67.7%。在平衡抽样的 180 段保留动画上，程序和人类标签的一致率为 93.3%，同组 VLM 评分为 83.3%。pass@32 测 32 次独立生成中至少一次成功，反馈后的迭代修复需另外评测。数据卡标注 CC BY-NC-SA 4.0，脚本有单独许可。 [固定数据及代码](https://huggingface.co/datasets/figma-ai/motion-graphics-bench/tree/89592d63cf9f5f4ae4ef501fa08166a2045bd20f) · [首方方法与对比](https://huggingface.co/blog/figma-ai/motion-graphics-bench)。

### Google 开放 MKA 的 10 项跨文件视觉工作示例

**Google DeepMind，11 月 10 日 18:02:28（北京时间）提交。** 示例要求 Agent 从图纸、栅格、表格等多文件工件中提取影响答案的视觉证据，交付表格、演示、文档或结构化报告。公开 Apache 2.0 目录按任务给出原始输入、指令与评分标准，覆盖十个领域。这是 MKA 的展示样本。可用同一任务对照仅 OCR 与原生视觉两条流水线，记录证据引用和交付文件的错误。 [固定任务目录](https://huggingface.co/datasets/google/multimodal-knowledge-agent/tree/46bce7ab8768f661750bd1016ec8cbdc517740b5) · [固定数据卡](https://huggingface.co/datasets/google/multimodal-knowledge-agent/blob/46bce7ab8768f661750bd1016ec8cbdc517740b5/README.md)。

### 评测工件补充

- **DexBench GA，11 月 10 日 22:55:01（北京时间）。** 发布固定开局状态，覆盖 46 个主要评估单元共 5,520 行，另有 3,000 行敏感性条件。每行绑定场景、机器人起始状态和摘要，可用于同一重置条件下对照策略，结合模拟器和策略版本复跑。 [固定条件集](https://huggingface.co/datasets/RLWRLD/DexBench-eval/blob/efc9aa12b849bd320a55b56a439930452588150c/README.md) · [评测实现](https://github.com/RLWRLD/DexBench)。
- **InfraOptBench，11 月 10 日 21:59:21（北京时间）。** 公开四个可冻结的性能优化题包，覆盖 SGLang 的两个起点、在线多 LoRA 窗口与 CPU Ragged-GQA。CPU 对照两模型均通过 60/60 项正确性，预算不同。SGLang 部分计时仍缺硬件资格。复测时可保留各自的基线，提交封存后再执行独立评分。 [固定题包与口径](https://huggingface.co/datasets/dabingzz/InfraOptBench/blob/9618d00a1a30438b256827bf709a6686f3d7ff5c/README.md)。
- **Gaokao-Bench v0.0.3，11 月 10 日 15:13:21（北京时间）。** 公开 2025 年五套数学卷的 74 道计分题，按卷面分值确定性判分，缺失与超时为零分。四题需要题图，文本和视觉模式均以 353 为分母，分别最高可得 334 与 353 分。公开真题存在训练集暴露的可能，横比应固定模态、版本与超时配置。 [固定数据卡与计分协议](https://huggingface.co/datasets/Knove/Gaokao-Bench/blob/78d6309923b0e6a678faffeccbb9d7dc21363991/README.md)。
- **TaxBench 2.0，11 月 11 日 07:17:41（北京时间）。** 820 题的专业税务评测公开 100 题，带 758 项加权专家评分项。样本中 87 题来自美国或印度，参考产物只供评分器使用，独立复测宜按地区和轮次分层。 [固定样本及口径](https://huggingface.co/datasets/intelligenceconsortium/TaxBench-2.0/blob/203562f22145e38564afd9d6ed4c959b453ca4da/README.md)。
- **PhyRubric-Bench，11 月 11 日 06:37:09（北京时间）更新发布卡。** 245 条预先写好各自物理评分细则的提示覆盖五个领域，公布十个生成器的 2,433 段短片和金标。训练集与评分器另有入口。可按物理类别测漏判与误判，模型生成视频仍适用各自许可。 [固定数据卡](https://huggingface.co/datasets/kittyhy/PhyRubric-Bench/blob/85f15837ae5999056fbb898b8bc65b4ba80b1a4d/README.md)。
- **补录｜Hugging Face 决策模型接口，11 月 10 日 08:00（北京时间）。** 公开 DecisionPipeline、AutoModelForDecision 和与 llama.cpp 共用的 /v1/systemone。相同输入可由专用决策头或生成模型的候选标签 logits 打分，两者应在同一分布上分别检验概率校准。文中 TRL 原生支持当时仍在 PR。 [HF 接口与评估说明](https://huggingface.co/blog/decision) · [TRL 工件状态](https://github.com/huggingface/trl/pull/8040)。

<a id="papers"></a>
## 研究论文与评估

### 主窗｜AACL-IJCNLP 11 月 10 日工作坊

11 月 10 日主办方排程覆盖 ARRSI、TrustAudio、PlurVA-LLM、WASP、From Reasoning to Agency 和 NORA。以下时间为单篇报告的排程，均为 UTC+8。PlurVA 主办方已在 11 月 9 日公告论文集上线，NORA 论文页只标到月份。 [会议排程](https://2026.aaclnet.org/program/workshops/#tuesday-november-10) · [ACL Anthology 会务索引](https://aclanthology.org/events/aacl-2026/) · [PlurVA 上线公告](https://plurvallm2026.github.io/#news)

#### 代理记忆需要分别评估持久写入、当次使用、核实与追问

**11-10 11:30–11:45｜NORA｜ACL Anthology 2026.nora-1.2。** 在 70 道英文主测试上，Gemma 4 E4B 从直接选标签切换到原始工具提示后，准确率由 .629 降到 .157，误写率由 .171 升到 .486，工具参数却都能通过最小校验。Qwen3.5-9B 在原始策略提示下的核实召回为 .944，追问召回为 .083。工具与标签条件同时改了措辞和角色框架，论文没有把影响单独归因于工具接口。部署评估可以分别报告各决策召回和误写率，再做同策略文本的接口对照。 [NORA 排程](https://nora-workshop.github.io/AACL2026/#program) · [论文 §6.2、表 5–6、§9](https://aclanthology.org/2026.nora-1.2.pdf)

#### 价值对齐审计把国家标签与可观测群体分布拆成独立变量

**11-10 11:59–12:11｜PlurVA-LLM｜ACL Anthology 2026.plurvallm-1.13。** 作者设置无提示、国家标签、相符群体分布和标签与分布冲突四个条件。在 GlobalOpinionQA 筛出的 200 个有向对照、四个 7B–9B 指令模型中，冲突时按选项似然算出的预测有 85.5%–97.5% 更接近证据群体。国家标签也带来小幅影响，但显式分布的平均作用在四个模型中更大。结果来自筛选后的调查题与受约束打分，适合作为标签/证据冲突的回归设计，覆盖面以外还需单测。 [报告排程与最佳论文](https://plurvallm2026.github.io/code/program.html) · [论文 §4–6、表 1–3](https://aclanthology.org/2026.plurvallm-1.13.pdf)

#### 三地价值对齐共享任务中，同一小模型配方拿下两条赛道

**11-10 14:35–14:59｜PlurVA-LLM｜ACL Anthology 2026.plurvallm-1.11、2026.plurvallm-1.8。** 总览按中国、印度尼西亚、斯里兰卡的准确率取宏平均，最终分别有三支和两支队伍进入两条赛道。同一 Gemma 4 E4B 方案用 LoRA 做选项判别和偏好学习，两条赛道均为 0.808，高于受限/开放赛道基线 0.600/0.780。获胜提交在印度尼西亚得 0.683，低于开放基线 0.696。三套数据分别考查价值标准、投票共识及可辩护性，宏平均容易掩盖这些差异。 [报告排程](https://plurvallm2026.github.io/code/program.html) · [总览表 2–3 与 Limitations](https://aclanthology.org/2026.plurvallm-1.11.pdf) · [系统 §3–5、表 3](https://aclanthology.org/2026.plurvallm-1.8.pdf)

ARRSI 日程另列 09:50 的 DeltaHarness 与 11:40 的评委/奖励模型选择报告，当前证据为议程与讲题。From Reasoning to Agency 日程列 13:30 的 InCoder-32B 与 14:00 的 Cortex，两项已有 3 月和 10 月预印本，保留为会议报告事件。 [ARRSI 日程](https://ar-rsi.github.io/#schedule) · [From Reasoning to Agency 日程](https://agent-aacl-workshop.github.io/#program) · [InCoder 版本记录](https://arxiv.org/abs/2603.16790) · [Cortex 版本记录](https://arxiv.org/abs/2610.20399)

### 主窗｜arXiv 11 月 10 日公告

以下论文都出现在 11 月 10 日的分类公告。常规发布时间为 11 月 10 日 01:00 UTC，逐篇公开秒未确认。各条另列版本及提交时间。 [arXiv 公告规则](https://info.arxiv.org/help/availability.html#announcement-schedule) · [cs.CL 公告](https://arxiv.org/list/cs.CL/new?show=2000) · [cs.LG 公告](https://arxiv.org/list/cs.LG/new?show=2000)

#### 801-30B-A3B 在 11.8T 预训练中用小规模试验筛选配方

**11-07 13:26:27 UTC 提交｜arXiv:2611.10719v1。** 团队把基础设施与数据诊断、检查点后训练潜力测量、数据质量指标构建做成可反复验证的小环节，完成 1T、3T 两次试跑后启动 11.8T 主训练，后两阶段上下文由 32,768 延长到 262,144。论文表 7 的 13 项 base 均值为 73.13，Qwen3-30B-A3B 为 66.16，Nemotron 3.5 Lightning 为 73.07，Qwen3.5-35B-A3B 为 76.67。8.1T 到 11.5T 的基础评测接近平台时，一轮轻量 SFT 后平均分从 70.54 升至 72.93，适合作为筛选检查点的补充信号。单次主训练与多项同步变更限制归因。作者仓库已列轻量 SFT 数据，并把 Base 权重标为即将发布。 [论文 §3–4、§7–8、图 10–11 与表 7–8](https://arxiv.org/html/2611.10719v1#S7) · [作者发布仓库](https://github.com/GAIR-NLP/801-30B)

#### 预训练蒸馏先扰动教师输入，再回到干净文本

**11-08 20:25:17 UTC 提交｜arXiv:2611.11598v1。** 固定训练预算，从头训练学生。前段用相邻 token 混合、跨文档混合或噪声，让学生拟合冻结教师的预测，后段恢复普通蒸馏。相邻 token 混合让 135M 与 0.5B 学生的验证损失相对同预算普通蒸馏分别下降 0.025 和 0.020 nats。135M 的噪声消融中，只扰动教师保留约九成收益，扰动太强或混合持续到 70% 训练时可退化。复现实验可优先做 teacher-only 与 clean-KD 同预算对照，现有结果来自两个小模型尺度，多数配置单跑。 [论文 §2–6、图 3–5、表 2–4](https://arxiv.org/html/2611.11598v1#S4)

#### MNN-Pocket-0.3B 把端侧延迟纳入模型尺寸和深度选择

**11-09 12:33:26 UTC 提交｜arXiv:2611.12311v1。** 报告把 iPad M5 的显存和速度约束用于 300M 稠密模型选型，之后以四个同源教师按领域做反向 KL 蒸馏，再用冻结的 BF16 学生作教师进行量化感知蒸馏。512-token、MNN 3.6.2、INT4、贪心且分别从空 KV 测量，iPad M5 Metal 的 prefill/decode 为 8,721/466 tok/s，骁龙 8 Gen 5 的 OpenCL 为 3,403/138，CPU 为 930/251。发布配置是 28 层、14 个 Q 头与 2 个 KV 头，位置字段为 40,960，论文标称支持 32K。端侧压测宜覆盖应用的真实上下文与后端。 [论文 §2、§4.2、§5.2、图 2、表 3–4](https://arxiv.org/html/2611.12311v1#S5) · [固定模型配置](https://huggingface.co/taobao-mnn/MNN-Pocket-0.3B/blob/48fae9c0be6ec505e80f36f69b8ba21097f9f822/config.json)

#### OPD 的训推失配预算随模型、量化配置与训练进度变化

**11-08 15:22:47 UTC 提交｜arXiv:2611.11450v1。** 用 rollout 引擎访问的前缀，在同一批已采 token 上对比 rollout 与训练端 log-prob，估计逐 token KL。测试的 7B 学生在 0.184 nats/token 及以下通过预设 held-out 教师 KL 标准，约 0.39 的加噪配置开始失败。在同学生、同引擎，REINFORCE 和 GRPO 对失配更敏感。只把第一层 KV 恢复 BF16，可让该 7B 的 vLLM FP8-KV 离线估计从 0.52 降至 0.0037。FP8 加最多八版本权重滞后达到同 KL 目标的时间快 1.62 倍，但 1.5B 学生表现及训练中超过预设离线裕量的例子显示，阈值应按模型校准并在线复核。11 月 10 日首次公开的代码中，`bench/tim_calibrate.py::score_samples` 用同批 token 重评分，`bench/tim_kl_eval.py::cmd_combine` 汇总 held-out 的学生与教师 KL。 [论文 §3–5、图 3–7 与附录表 6、8、11](https://arxiv.org/html/2611.11450v1#S5) · [作者代码首次提交](https://github.com/benyucong/train-inference-mismatch-opd/commit/1fc4571cc5cb07173193fec546827956ae647fe5)

#### 后训练的低稳定秩可能掩盖高能量秩

**11-09 01:43:20 UTC 提交｜arXiv:2611.11752v1。** 将阶段终点权重差的谱与相邻检查点位移分开测量。OLMo-2-7B 的三处 PPO attention 更新中，稳定秩约 9.6–18.3，但完整 SVD 需 532–1,222 个分量才能覆盖 80% 平方谱能量。公开 PPO 检查点的相邻位移有负相关，在匹配的三对局部种子上，group-relative 与 critic–GAE 的平均余弦分别为 +0.259 和 +0.012，两个受控配置都未复现公开轨迹的负相关。选 LoRA 秩时可直接检查目标能量覆盖率。跨发布轨迹的训练预算、检查点间隔和存储精度各异。 [论文 §3–5、图 1、表 2–5](https://arxiv.org/html/2611.11752v1#S4)

#### VerKV 按注意力误差估计选择性回取被移出的 KV

**11-08 11:43:34 UTC 提交｜arXiv:2611.11319v1。** prefill 后按保留概率抽样 KV，GPU 上的低位 key 摘要辅助估计误差，当前 query 超出容许误差时从 CPU 补取。Qwen3-4B 在 32K/128K 的 18 项 RULER/LOFT 汇总中，常驻 KV 为 6.25% 时保留 dense 质量的 95.0%，同预算 Fast KVzip 为 28.4%。单 H100、vLLM fork、单请求的 Qwen3-4B 解码相对全量 CPU offload 快约 1.6 倍。误差保证针对注意力估计，质量和吞吐仍受模型、CPU 链路与摘要阈值约束。 [论文 §4–5、表 1–2 与附录 A.3](https://arxiv.org/html/2611.11319v1#S5)

#### CVSA 用覆盖估计给稀疏 attention 配置逐头回退

**11-07 10:22:27 UTC 提交｜arXiv:2611.10634v1。** 用稀疏输出及 softmax 质量覆盖估计识别需要重读的 head，配合混合调度与选定 dense 层。三个约 7B 模型的 LongBench 16 项平均，dense 41.83、CVSA-c4 40.70、AdaKV 38.78，CVSA 实测 KV 读取 11.7%，对照标称 10%。RULER 上 c4 落后 AdaKV，且对齐读取量的消融显示增益也来自刷新节奏和 dense 层的位置。工程对照应对齐实际读取量并按任务拆开报告。 [论文 §3–4、表 1、4、8、19](https://arxiv.org/html/2611.10634v1#S4)

#### D-REAL 按验证器实际接收前缀来训练投机 drafter

**11-07 03:19:43 UTC 提交｜arXiv:2611.10432v1。** 从 drafter 自采前缀，以目标与 drafter 概率的重叠构造动态权重，直接优化期望接收长度。在 Qwen3-8B、SGLang、八卡 Hopper、32 并发的 GSM8K，DFlash2 的输出吞吐在温度 1 由 10,064 升至 10,668 tok/s，均值 E2E 时延由 824 降至 775 ms，DSpark 在同档吞吐提升 2.0%。十项任务确认接收长度收益，端到端测量只覆盖一个负载，复核时同时保留采样温度和 drafter 对照。 [论文 §3、§5–6、表 1–2](https://arxiv.org/html/2611.10432v1#S5)

#### MACaQ 把 MXFP4 校准和推理变换限制在单个线性层

**11-07 10:49:02 UTC 提交｜arXiv:2611.10649v1。** 在线按通道缩放并做分块 Hadamard，逆变换并入权重。校准时按 Hessian 近似挑选移动到邻近 FP4 码的权重，逐层处理。Llama-3-8B 的 WikiText-2 PPL 达到 6.97 用 31.3 秒，MR-GPTQ 的 7.11 用 267.9–319.1 秒，FP16 为 6.14。Qwen3-32B 的 vLLM 单批、512 token 解码中，在线缩放与旋转相对关闭两者的 MXFP4 路径增约 2.3% 开销。接入前可在实际批量重测，校准速度与解码吞吐是两项独立成本。 [论文图 2–3、表 6–7 与 Limitations](https://arxiv.org/html/2611.10649v1)

#### 极低比特权重量化的校准收益集中在特定算法和任务

**11-08 02:40:47 UTC 提交｜arXiv:2611.11079v1。** 将易损任务导向的 FragCal-DC 与通用 Pile-200 校准对比。在 W2G128，SignRound V2 六模型十任务平均提高 1.83 个百分点，OmniQuant 五模型提高 1.78。W3G128 下 GPTQ +0.23、AWQ −0.04，两者区间均跨零。SignRound V2 的预选易损任务增 3.86 点，严格非目标任务仅 +0.20 点。量化校准宜保留任务外对照，并按位宽、模型和量化算法分层报告。 [论文 §4–7、表 2–7](https://arxiv.org/html/2611.11079v1#S5)

#### 部分题面补全的污染测试会随剩余题长产生误报

**11-06 23:13:57 UTC 提交｜arXiv:2611.10307v1。** 在七个指令/蒸馏模型和 60 道 AIME 2025–26 负对照题中，逐字匹配在 40% 与 60% 前缀时未测到误报，80% 时合并误报率达 8.6%。60% 前缀时，Qwen2.5-7B 在 MATH-500 上 ROUGE-L 原始标记为 22.6%，按其他模型族的可补全性筛选后为 11.4%。负对照样本小，结果支持按剩余字数分层、同时提供跨模型族与训练截止后对照，无法单凭高标记率认定污染。 [论文 §3–6、表 1–2、图 1–2](https://arxiv.org/html/2611.10307v1#S4)

#### PACE 的长程缓存实验区分内容仍在与模型还能用

**11-06 21:25:45 UTC 提交｜arXiv:2611.10240v1。** 预算 2,048 KV token 的 Qwen3-8B 在 100K 合成编码会话中，按预登记对照用计划保留与有界位置时通过 145/150 项隐藏检查，单开任一项均为 94/150，H₂O 为 63/150。只保留计划指定 token 时，所需内容可以全部常驻，但远距离应用仍有错误。主要对照用已知依赖图生成计划，实际代理需单测计划错误、位置策略及再加载代价。 [论文 §3–5、§7、表 1–3](https://arxiv.org/html/2611.10240v1#S5)

#### BridgeKV 把大模型 prefill 接到小模型解码

**11-09 02:03:03 UTC 提交｜arXiv:2611.11761v1。** 将大模型 KV 按层混合并映射到小模型 cache，以缓存对齐及生成目标训练。单 A40、1,024 输入及 1,024 输出 token，Llama 8B→1B 的响应时间 9.13 秒，8B 原生 39.80 秒，1B 原生 9.05 秒。混合路径 TTFT 111.4 ms，1B 原生为 27.5 ms。最高 33.86 倍是独立的 70B→1B 服务模拟。是否采用取决于质量保留和输出长度，实际流量需同时比较两种原生模型。 [论文 §3–5、图 3、表 4–7](https://arxiv.org/pdf/2611.11761v1)

#### DForkTree 用分支感知的 n-gram 修正重排投机树

**11-09 13:06:52 UTC 提交｜arXiv:2611.12356v1。** 用目标模型自生成语料统计 n-gram，针对离开主干后的候选修正独立位置边际分数，并在 GPU 固定节点预算内组树。RTX 5090、batch 1、温度 1 的 Qwen3.5-4B/9B 上，相对自回归吞吐分别为 3.05/3.88 倍，DDTree 为 2.61/3.44，EAGLE-3 tree 为 2.80/3.64。作者未测连续批服务，离线语料也是额外成本。固定实现的 `Builder.run` 用 n-gram 路径函数给候选树评分，`fc1_fused.py::install.fused_commit` 在验收后提交对应分支状态。主窗 23:59 UTC 提交仅更新 README 的参考 GPU。 [论文 §3–5、表 5 与附录 I](https://arxiv.org/html/2611.12356v1#S5) · [作者提交](https://github.com/Perseus037/DForkTree/commit/57d3a187da189e94f460fc04c50c391b846433c3)

#### 对角 Fisher 筛查器的绝对 KL 预测在部分 INT4 配置失效

**11-09 16:11:49 UTC 提交｜arXiv:2611.12634v1。** 预注册实验把二阶预测和精确 KL 对照，并分别考察梯度精度与截尾汇总。Qwen3-4B 用 FP32 梯度后，三种格式的实测/预测比为 0.95–1.10。同协议在 OLMo-2-7B 的 INT4 G128 为 1.35，越过 0.85–1.15 预设区间。Qwen3-32B 的 INT4 为 5.18，后验检查把误差集中到少数结构。代理分数可用于排序，绝对质量门槛宜保留精确验证，尤其在低比特和新模型族。 [论文 §6–9、§12、表 1 与 3](https://arxiv.org/html/2611.12634v1#S7)

#### RDS-MOPD 按各领域相对进度调多教师蒸馏的采样量

**11-09 07:51:00 UTC 提交｜arXiv:2611.12012v1。** 按领域监测学生与教师的 KL 变化，抑制短时尖峰，并对较自身最佳点退步的领域增加采样。在一个 60 步同种子受控实验中，自第 30 步向代码教师的 top-16 log-prob 加随机噪声后，RDS-MOPD 在 41–60 步给代码的份额为 .325，D3-MOPD 为 .179。期末代码 KL 分别为 .0164 与 .0225。故障实验覆盖一种噪声注入与单个教师，训练监控可将领域预算和教师打分健康度并看。 [论文 §3、§4.2–4.5、表 1、图 4–5](https://arxiv.org/html/2611.12012v1#S4)

### 主窗｜已核正文差异的 arXiv 修订

#### Ghosted Layers v3 增补视觉语言模型裁层对照

**11-07 04:37:41 UTC 修订提交｜arXiv:2605.15491v3。** 延续 v2 通过原模型边界激活拟合裁层补偿矩阵的方法。新增 LLaVA-1.5-7B 对照，裁语言解码器 11/32 层，MME、SQA、MMMU 均分为 55.44，未恢复为 47.94，ReplaceMe 为 53.64，dense 为 61.28。结果把恢复证据扩展到单个 VLM，仍需能读取未裁模型激活，新增实验没有报告 VLM 端到端延迟。 [v2](https://arxiv.org/html/2605.15491v2) · [v3 §3 与新增附录表 A5](https://arxiv.org/html/2605.15491v3#A5.T5)

#### Steerling v2 加入指令版实测及扩散解码退化记录

**11-09 08:22:14 UTC 修订提交｜arXiv:2608.07594v2。** v1 已有带可加概念瓶颈的块因果扩散架构与预训练。v2 加入后训练 §11、表 20–21。Steerling-8B-Instruct 的六项生成评测均分 33.6，LLaDA-8B 为 54.7，OLMo 2 7B 为 51.7，基线分数来自不同既有来源。作者记录高置信逐位置解掩会产生全 EOS 块，评测时屏蔽 EOS logit 并用独立 EOT 标识结束。比较概念指标与生成质量时，应钉住该解码配置并保留独立性损失在 SFT 后升高这一结果。 [v1](https://arxiv.org/html/2608.07594v1) · [v2 §11、图 32、表 20–21](https://arxiv.org/html/2608.07594v2#S11)

<a id="engineering"></a>
## 开源工程

本节按 GitHub PR 的合入时间或默认分支提交时间记录，正式版和预发布构建单列。时间均为上海时间。

### 训练、蒸馏与训推一致性

#### Megatron 拆开多层 MTP 的反传并修正梯度桶生命周期

**11-10 21:37–11-11 07:07，`main` 默认分支合入。** EP all-to-all 与 1F1B 重叠时，多层 MTP 的最后一层曾提前走进前层的计算图并释放它。现在前层把 detach 的共享存储视图交给拼接，自己的 backward 再合入该视图的梯度。GPTModel 可用，重复 MTP 层、HSM、mHC、全量重算及主线 HybridModel 有独立限制。MFSDP v1 则在释放临时梯度桶前让其分配流等到 reduce-scatter 完成。复测应包含分片梯度、MTP 和通信流交错。 [Megatron-LM #8799 `bf01a58ebb`](https://github.com/NVIDIA/Megatron-LM/pull/8799) · [Megatron-LM #9114 `908d0ea007`](https://github.com/NVIDIA/Megatron-LM/pull/9114)

#### Megatron 可按固定归约树对齐不同 EP 大小的 MoE combine

**11-11 05:10，`main` 默认分支合入。** 默认关闭的 `moe_ep_invariant_combine` 采用固定顺序的 FP32 归约树，让 batch-invariant MoE 的合并结果可跨 EP 布局逐位比较。适用 EP 值是能整除专家数的 2 的幂。用于训练 EP8、rollout EP4 之类的对照时，应单独核对其他 kernel、路由选择及分片是否也满足一致性条件。 [Megatron-LM #9107 `76cf2e2d64`](https://github.com/NVIDIA/Megatron-LM/pull/9107)

#### Megatron dev 合入 Kimi Attention Residuals 和 DeepSeek V4 GB200 配方

**11-10 12:02–11-10 15:14，`dev` 特性分支合入。** Kimi 分支把 block attention residual 组合进 HybridStack 层循环和 MTP 输入，带细粒度 activation offload 和可变序列的非交错 PP 支持，显式拒绝 mHC、宽残差、shortcut MoE 和激活的逐层 CP 布局管理器。DeepSeek V4 Flash/Pro 配方固定依赖，修复 grouped 输出投影的 Muon 形状和跨模块直接读参数的就绪顺序。作者在 128/256 张 GB200 以 mock 数据、强制路由测到约 569–740 TFLOP/s/GPU，4K BSHD 与 64K THD 不宜直接比较，数据未覆盖收敛性。 [Megatron-LM #8913 `4ceb3c4ea7`](https://github.com/NVIDIA/Megatron-LM/pull/8913) · [Megatron-LM #9070 `43a2246f4e`](https://github.com/NVIDIA/Megatron-LM/pull/9070)

#### Megatron-Bridge 补齐 NCCL EP packed SFT 的逐 rank 容量上界

**11-11 04:46，`main` 默认分支合入。** 上期从 HybridEP 切换的九套 GB200 SFT 配方，THD 长度可变时在首个 MoE dispatch 缺少静态分配上界。现在设 `max_seq_length_per_sample=seq_length`，dispatcher 按 CP 和启用 sequence parallel 时的 TP 换算逐 rank 上限。升级后用真实 text/VL config 检查该值到达嵌套的 TransformerConfig，再执行首步。 [Megatron-Bridge #6865 `8dff4ae1da`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6865)

#### verl 接通 Megatron score centering 并收窄 Ray OMP 默认值

**11-10 14:37–11-10 22:16，`main` 默认分支合入。** 采样头概率以 TP 全局行最大值和归一化和计算，再按 CP 切头，交叉熵拿 logits 副本以避免其原位移位破坏 backward 重读。CPU 双 rank 测覆盖 vocab shard。VeOmni CPU offload 的 OMP 默认值改为训练 RayWorkerGroup 局部，避免 rollout、reward 和 TransferQueue 一起抢核。PR 在单卡 104 核的特定负载观测 84 s 与 44 s 每步的差距，迁移时检查实际 worker CPU 配额。 [verl #8199 `b363245c5b`](https://github.com/verl-project/verl/pull/8199) · [verl #8592 `4b39670207`](https://github.com/verl-project/verl/pull/8592)

#### verl 给 routed experts 增加 MXFP4 rollout 与 QAT 对照配方

**11-10 23:39，`main` 默认分支合入。** Qwen3-30B-A3B 用 DAPO-Math-17k、固定数据种子，支持 BF16、仅 rollout MXFP4、ModelOpt 专家 fake quant 加 MXFP4 rollout 三组。4 张 B300、各 10 步时 QAT 令 rollout/trainer KL 从 0.0150 到 0.0030，BF16 为 0.0018。QAT 每步 900 s 对仅量化 rollout 的 848 s。结论限这套短测和专家路径，可先跑同种子三组再比较奖励与 KL。 [verl #8426 `63913d1d5f`](https://github.com/verl-project/verl/pull/8426)

#### verl 扩充 Ascend 与 TPU 的 GRPO 启动配方

**11-10 09:53–11-10 14:36，`main` 默认分支合入。** 16 卡 Atlas A3 的 Qwen3.5-35B-A3B 分支接入 3-token MTP、按每序列 4 token 对齐的图捕获、两组 TP4 rollout 和复用 rollout 旧 log-prob，配套按路径触发的 e2e。其他改动也一起启用，评估 MTP 应单独消融。TPU 合入 Qwen3-30B-A3B 的 TorchTitan＋vLLM 双 slice 配方，trainer 用 FSDP、tpu-sync 更新权重，并统一六个 dense Qwen3 尺寸的 launcher。 [verl #8576 `1d4194cb28`](https://github.com/verl-project/verl/pull/8576) · [verl #8451 `0269aafa5f`](https://github.com/verl-project/verl/pull/8451) · [verl #8456 `1ec15fed25`](https://github.com/verl-project/verl/pull/8456)

#### TRL 约束 structured GRPO 的归一化域和异步 rollout 年龄

**11-10 17:35–11-10 23:08，`main` 默认分支合入。** GRPO 在 vLLM 施加 grammar mask 后，需要在采样候选集合重新归一化训练 log-prob，故 vLLM colocate ≥0.28 要求 `top_k>0`，缺候选集合的 server 模式会报错。vLLM 0.32 测例中差值从 5.5–8.9 nats 降到 0.03 以内。异步 GRPO 和蒸馏按 trainer 消费的 token、上一 optimizer step 用量及 `max_staleness` 控制新 rollout，整组获准后继续完成，防止半组停住。看丢弃样本、每步实际 token 和在途组数。 [trl #8145 `def29c3fac`](https://github.com/huggingface/trl/pull/8145) · [trl #8170 `f2ea8a9ad4`](https://github.com/huggingface/trl/pull/8170)

#### TRL 加入 GOLD 特权上下文蒸馏和抗标签翻转 UDPO

**11-11 07:25–11-11 07:40，`main` 默认分支合入。** GOLD 给 teacher 拼入逐样本附加上下文，student 保留原始 prompt，两侧在同一组 completion token 上做 JSD。支持文本和对话输入，拒绝 X-Token KD 等不兼容模式，PR 报告 CPU 小模型验证。DPOTrainer 的 `udpo` 将 `label_smoothing` 解释为翻转率，区间 [0, 0.5)，零值退回 sigmoid，用 logaddexp 实现稳定的归一化 α-UDPO。先确认数据噪声模型，再用同一评测比较。 [trl #6437 `65f8d2e3c1`](https://github.com/huggingface/trl/pull/6437) · [trl #8211 `d115286bc5`](https://github.com/huggingface/trl/pull/8211)

#### TorchTitan 把视觉输入贯通 vLLM rollout 与 GRPO 训练

**11-11 07:26–11-11 08:30，`main` 默认分支合入。** Qwen3.5 栈把带 hash 的图像和 placeholder 位置贯穿环境、vLLM generator 与训练样本，在同一个 trainer microbatch 中每张图只编码一次。Geometry3K 提供 GRPO 配方。验证时应比对 rollout 与 actor 对同一图像的 processor 输出和位置，并检查批次打包。对应 LoRA-only 同步和视觉 encoder graph 是另一组后续 PR。 [torchtitan #5824 `ef53e9c15f`](https://github.com/pytorch/torchtitan/pull/5824) · [torchtitan #5825 `1f47797aee`](https://github.com/pytorch/torchtitan/pull/5825)

#### TorchTitan 的 SGLang rollout backend 完成多段特性分支合并

**11-11 04:13，`sglang-1-paged-attn-flash`、`sglang-2-engine`、`sglang-3-moe`、`sglang-5-gdn`、`sglang-6-mla`、`sglang-7-kda`、`sglang-8-full-prefill-graphs`、`sglang-9-stagger-shared-prompts` 特性分支合入。** SGLang 管调度和 KV，TorchTitan 继续运行训练模型与注意力内核。分支链含 EP all-to-all/Dist-MoE/DeepEP/HybridEP、窗口语义转换、每头 MLA KV、按 64 token 回放对齐的 KDA、完整 prefill CUDA 图和分批放行同一 GRPO prompt 的副本。SGLang 在首请求 prefill 结果处理后才发布 radix KV，错开准入可减少重复 prefill。各段进入栈式特性分支，使用前核对根合并与 dispatcher 的 CUDA graph 限制。 [torchtitan #5812 `4f3be59e3c`](https://github.com/pytorch/torchtitan/pull/5812) · [torchtitan #5815 `41856d9716`](https://github.com/pytorch/torchtitan/pull/5815) · [torchtitan #5816 `54116a879b`](https://github.com/pytorch/torchtitan/pull/5816) · [torchtitan #5818 `2839ad3ad8`](https://github.com/pytorch/torchtitan/pull/5818) · [torchtitan #5819 `6dd00a4f1f`](https://github.com/pytorch/torchtitan/pull/5819) · [torchtitan #5853 `71d78ae299`](https://github.com/pytorch/torchtitan/pull/5853) · [torchtitan #5918 `ad06f8c9f2`](https://github.com/pytorch/torchtitan/pull/5918) · [torchtitan #5959 `5a4e2c7489`](https://github.com/pytorch/torchtitan/pull/5959)

#### TorchTitan 修复 DeepSeek V4 PP 与异步 checkpoint，编译 Kimi MLA KV 装配

**11-10 15:14–11-11 02:42，`main` 默认分支合入。** PP forward 从本 stage 持有的层推导执行范围，末段携带输出 head，避免部分模型读缺失层。Kimi MLA 的 strided cat 换编译路径，作者在 4096 token/rank、96 heads 的形状报告前后向 767→234 μs 且逐位相同。Pinned-memory checkpoint 的 staging 流等 optimizer 流，避免日志和保存周期错开时复制旧状态。做恢复对比时可故意设不同的日志/保存周期。 [torchtitan #4782 `1ff3ae648d`](https://github.com/pytorch/torchtitan/pull/4782) · [torchtitan #5682 `63e44072f6`](https://github.com/pytorch/torchtitan/pull/5682) · [torchtitan #5989 `78c55e7538`](https://github.com/pytorch/torchtitan/pull/5989)

#### DeepSpeed 对齐 torch 2.12 编译标准并修复 Ascend fused op

**11-10 18:21–11-10 18:30，`master` 默认分支合入。** torch ≥2.12 时 CUDA 12.0–12.4 的扩展也用 C++20，避免 nvcc 后置 C++17 覆盖 PyTorch 头文件所需标准，ROCm 同步按 torch 版本选择。Ascend 的 FP16 LAMB 镜像写回和最后一维分组的 FP16 INT4/8 反量化补齐 NPU builder 路径。验证选真实 JIT 编译与分布式首个 optimizer step，再测满足 fast-path 条件的权重。 [DeepSpeed #9041 `cbed412dc5`](https://github.com/deepspeedai/DeepSpeed/pull/9041) · [DeepSpeed #9043 `f7a7177751`](https://github.com/deepspeedai/DeepSpeed/pull/9043)

#### OpenRL 公开 Harvey LAB 长上下文 GRPO 的工程复盘

**11-10 01:00，`main` 默认分支合入补录。** Tinker 驱动把任务和 rubric 交给环境，Automodel 用 FSDP2＋CP、重算与分块 log-prob，四张 H100 运行 Qwen3.5-9B 的约 260K 训练上下文。另四张卡跑 vLLM 副本，llm-d 按负载及前缀选择，作者在其路由对照中报告单步 35→17.7 min。300 训练、50 跨场景留出，GLM-5.2 grader 下 Qwen 35 步从 0.54→0.78，配对四次复测增 13.2 个百分点（95% CI 8.4–18.1）。Gemma-4-31B 的直接 RL 从 0.53→0.55，SFT 后 RL 从 0.75→0.80。复现实验应保留场景划分并单列未完成、每步时间与 rubric pass rate。 [openrl #442 `f6912143e7`](https://github.com/openrl-dev/openrl/pull/442) · [OpenRL 作者文章](https://openrl.dev/blog/post-training-legal-agents/)

#### OpenRL TPU 特性分支按进程可见 HBM 定 vLLM 批上限

**11-11 06:24，`feat/tpu` 特性分支合入。** 采样器用 torch accelerator 的设备模块读 total_memory，v5e/v6e 仍为 64 序列档，tpu7x 由 64 到 256，显式 `VLLM_MAX_NUM_SEQS` 和 `VLLM_MAX_NUM_BATCHED_TOKENS` 优先。合入 `feat/tpu`，验证时按单进程可访问的 device HBM 核对配置并压测 prefill 批。 [openrl #435 `1c5d197a33`](https://github.com/openrl-dev/openrl/pull/435)

### 推理、KV 缓存与服务

#### Megatron Mamba 投机验证接入 ReplaySSM，并修复状态 slot 准入

**11-11 00:02–11-11 08:52，`main` 默认分支合入。** 启用 `--mamba-replay-ssm` 后，每层每 slot 留一份惰性 checkpoint 和 x/B/dt 环，整段 verify 从 checkpoint 及环缓存带因果掩码计算，中间不保存整份 SSM state，接收长度用设备侧指针提交。超出 `write_pos+2W>L` 时提前刷新，卷积状态另走原回滚路径。P/D 解码准入同步检查 recurrent slots，且 expert TP × EP 大于 1 时保留 dummy slot，覆盖 TP2/EP1 的空转 forward。应复测满 slot、部分接受和环刷新。 [Megatron-LM #6617 `47499f90ae`](https://github.com/NVIDIA/Megatron-LM/pull/6617) · [Megatron-LM #9121 `dd4c8e20dc`](https://github.com/NVIDIA/Megatron-LM/pull/9121) · [Megatron-LM #9123 `2bceb5b193`](https://github.com/NVIDIA/Megatron-LM/pull/9123)

#### vLLM 让编译缓存和 RDMA KV 注册跨进程可复用

**11-10 11:49–11-11 06:59，`dev`、`main` 默认分支合入。** D2D 控制面传带版本和校验的 manifest，Mooncake 传 FlashInfer cubin/JIT 产物，目标先校验并原子安装，失败则本地编译。作者在 B300 的 K3-like MoE 报 daemon 启动 337→34 s，范围是具备暖源的同构设置。外部 connector 通过 `kv_cache_access` 声明访问位标记，`MultiConnector` 合并各子连接器的标记。RDMA 注册与跨进程导出要求每段单一物理分配，legacy CUDA IPC 禁用 extensible KV。LMCache MP 按配置声明 legacy、FD 或 FABRIC。POSIX FD 导出在此补丁中尚未实现，升级时用实际 IPC 模式验证注册和回退。 [vllm #63753 `0d96fb6589`](https://github.com/vllm-project/vllm/pull/63753) · [vllm #66605 `0a37f464da`](https://github.com/vllm-project/vllm/pull/66605) · [LMCache #6487 `9239503d1c`](https://github.com/LMCache/LMCache/pull/6487)

#### vLLM 区分异步 KV 启动节流与 offload 命中边界

**11-10 11:06–11-10 19:25，`main` 默认分支合入。** 实验项 `--max-async-kv-loads-per-step` 对一轮 schedule 新发 load 限流，和 in-flight 上限共用 gate，等待中的异步请求进入独立队列。140K token 高负载实验中，cap=64 的慢 rank schedule p99 48.2 ms，限 2 次后为 9.0 ms，load 前等待约 0.05→0.17 s。另一 PR 让稀疏组缩短查找时只保留已确认的 full-attention junction。部署侧同时看慢 rank 尾延迟、等待和交界缓存命中。 [vllm #61725 `dc74b8cbbb`](https://github.com/vllm-project/vllm/pull/61725) · [vllm #66544 `9b38c833ee`](https://github.com/vllm-project/vllm/pull/66544)

#### vLLM 减少 P/D 重复 KV、CPU 复制碎片和 prefill 投机开销

**11-10 13:23–11-11 06:45，`main` 默认分支合入。** MoRIIO 单组 WRITE 跳过 decoder 已有的前缀块，MI355X 的 MiniMax-M3.1-Flash 1P1D 报每请求每 rank 写入 1650→130 MB。NVFP4 KV 的 MiniMax MSA 可显式选择 BLHNC，使 CPU offload 按 block 跨层连续复制。MRV2 的 DFlash 类 drafter 可在 prefill worker 开 `prefill_context_kv_only`，保留 decode 端所需的上下文 KV 与状态尺寸，但省去本地 draft、lookahead slots 和 draft CUDA graph。各项需按 connector、模型和布局单独验证。 [vllm #65822 `47236ef76e`](https://github.com/vllm-project/vllm/pull/65822) · [vllm #66892 `539366a773`](https://github.com/vllm-project/vllm/pull/66892) · [vllm #65217 `e3dc02e56c`](https://github.com/vllm-project/vllm/pull/65217)

#### vLLM 缩短 GLM-5.3 与 DeepSeek V4.1 解码路径，合并 EPLB 回拷

**11-10 15:08–11-10 16:27，`main` 默认分支合入。** GLM-5.3-Flash opt-in 专用解码在 GB300 TP4、MRV2 FULL decode graph、FP8 KV 且无 PP/DP/EP/DCP 的前提下用 CuTe DSL 与融合 AR/mHC，覆盖 ≤64 tokens 的 decode/均匀 MTP verify。合成 8K 输入并发 1 的中位 ITL 5.15→2.93 ms，并发 64 为 13.92→12.00 ms。DeepSeek V4.1 将 ≤336 tokens 的 shifted mHC 融进 cluster kernel，GPU EPLB 则把同一权重的多专家行聚成一次 Triton 回拷。压测应把低并发收益、重排发生时延迟和超界 fallback 分开。 [vllm #63102 `ade4feed90`](https://github.com/vllm-project/vllm/pull/63102) · [vllm #66652 `2858aa5901`](https://github.com/vllm-project/vllm/pull/66652) · [vllm #66660 `83f5de466a`](https://github.com/vllm-project/vllm/pull/66660)

#### vLLM 解锁三条模型特化路径

**11-10 11:49–11-11 01:18，`main` 默认分支合入。** DeepSeek V4.1 在 gfx950 可选 AITER MegaMoE（作者测 MI355X 的 TP1/DP8/EP8 + MoRI 与 DSpark），准确率对照的 token budget 不同。Kimi-K3 ROCm 的专用解码在扩展 KV 后重绑 views 并重跑 `prepare_kv`，覆盖首次 eager decode。Muse Spark Open 可在 Hopper 加载 BF16 checkpoint 后仅逐通道量化 routed experts，attention、dense、router 和输出头留 BF16，线性层泛化未验证。回归分别核对启动、权重热更后首次 decode 与任务指标。 [vllm #66644 `71a550ecd4`](https://github.com/vllm-project/vllm/pull/66644) · [vllm #66370 `dd5299b116`](https://github.com/vllm-project/vllm/pull/66370) · [vllm #66139 `24edda7de3`](https://github.com/vllm-project/vllm/pull/66139)

#### SGLang HiCache 收敛 prefetch 投票、短写确认和 worker 故障

**11-10 10:43–11-11 02:12，`main` 默认分支合入。** 把 timeout 的 rank vote 从每次准入 poll 移到固定 scheduler tick，避免 TP rank 因本地满批提前退出而多发一次 collective。分批 backup 仅确认 rank 共识的已写前缀，尾部仍可补写，避免位置对齐的 ACK 序列漂移。Worker 失败保序收尾并让 rank 显式失败。验证时给单 rank 注入 IO 短写或线程异常，核对后续前缀及队列释放。 [sglang #47617 `fadf83be20`](https://github.com/sgl-project/sglang/pull/47617) · [sglang #47620 `e886acfa8b`](https://github.com/sgl-project/sglang/pull/47620) · [sglang #47626 `daa171846f`](https://github.com/sgl-project/sglang/pull/47626)

#### SGLang 统一内存把 PP 容量和异步状态提交做成共同约束

**11-11 01:16–11-11 01:21，`main` 默认分支合入。** PP 各 stage 启动时交换 arena 几何，容量判断同时满足每级字节和 slot 需求。每 microbatch 处理结果后建立 write-after-read 屏障，回收页须等仍在读它的 forward。Mamba verify 回写复用该轮已解析的 slot，避免在请求行复用后再次查表写到新 owner。MPS/MLX 无法建立所需等待时启动拒绝，回归选混合模型并交错运行释放、复用和 verify。 [sglang #49299 `e38dff3e80`](https://github.com/sgl-project/sglang/pull/49299) · [sglang #49301 `1b51a11129`](https://github.com/sgl-project/sglang/pull/49301) · [sglang #49309 `347c07e89e`](https://github.com/sgl-project/sglang/pull/49309) · [sglang #49310 `e8a2bee913`](https://github.com/sgl-project/sglang/pull/49310)

#### SGLang 修复 Kimi CUDA graph 与 DSA 远端缓存，并扩展 NVLink gather

**11-11 06:15–11-11 08:50，`main` 默认分支合入。** FlashMLA 1.0.4 随 query 行数变化分片数，每个 decode/verify 图各自持有 schedule 并原位更新，避免大小 74/148 的张量互拷。Kimi-K3.1 direct linker 把 DSA indexer、MLA draft 和 KDA state 分池，TP 复制数据由 rank 0 写，DCP 分片单独入键。跨节点但同 NVLink clique 的 logits all-gather 可走 multimem，跨 clique、查询失败或无 multicast 指针回 NCCL。先按 TP/DCP、图尺寸与拓扑做逐位验证。 [sglang #49616 `214532133b`](https://github.com/sgl-project/sglang/pull/49616) · [sglang #49355 `7dc1e38e53`](https://github.com/sgl-project/sglang/pull/49355) · [sglang #49524 `21adb48ae0`](https://github.com/sgl-project/sglang/pull/49524)

#### SGLang 为 DSpark 与 ReplaySSM 增加低精度快路径和 KV 生命周期修复

**11-10 12:36–11-11 05:22，`main`、`qwen38-fn-rubin` 主线与特性分支分别合入。** DSpark 允许 Blackwell 选 MXFP8 draft 输出头，模块缓存权重、每次量化激活，target verify 继续独立计算。延迟 draft KV 写入只保留仍携带且未重新 retract 的请求。Qwen3.8 Rubin 特性分支增加 GDN u-cache，状态用 checkpoint 加环，在窗口满时同一 kernel 折回，减少逐步全状态写入。关注接受长度、最终精度、被撤回请求的页和未折回状态。 [sglang #49595 `9fdd39206d`](https://github.com/sgl-project/sglang/pull/49595) · [sglang #49436 `73bf84e90f`](https://github.com/sgl-project/sglang/pull/49436) · [sglang #48444 `970fa089b1`](https://github.com/sgl-project/sglang/pull/48444)

#### LMCache 允许 MP 保存尾块，并扩展 Valkey 与 RDMA 写通路

**11-10 13:20–11-11 04:37，`dev` 默认分支合入。** 可选 `lmcache.mp.save_unfull_chunk` 保存长 chunk 最后未满的部分。Mamba hybrid 需要更细的 `--prefix-match-unit` 才能命中可用尾部。Valkey 新增 blob-tcp，与 RDMA 走同一种大对象格式和 key prefix，要求服务端模块与 glide ≥2.3.0。NIXL/Mooncake 增加 write 提交与统一完成查询，目前是 headless DRAM pool 的传输基础。应按两种 transport 交叉回读，分别观察尾部命中和验证写入完成。 [LMCache #5886 `30dc05fba0`](https://github.com/LMCache/LMCache/pull/5886) · [LMCache #6400 `78ee369fa1`](https://github.com/LMCache/LMCache/pull/6400) · [LMCache #6401 `ac34cafe82`](https://github.com/LMCache/LMCache/pull/6401)

#### NeMo RL 在活跃生成中保存前缀，并给 SingleController 接上动态采样

**11-10 21:41–11-11 08:02，`main` 默认分支合入。** Gym 协调 checkpoint 时把活跃 vLLM 调用已生成 token 写入 prefix snapshot，恢复从剩余尾部继续，端到端 parity 测试在后续 PR。SingleController 的 DAPO 式采样剔除配置掩码下无奖励信号的 prompt 组，从后续已结束组或补采池填满，数据耗尽后按最小批次比例处理。复测中途重启的尾部/最终采样，以及零方差组占比很高的耗尽路径。 [RL #4506 `c459c448bd`](https://github.com/NVIDIA-NeMo/RL/pull/4506) · [RL #4588 `fc009e635e`](https://github.com/NVIDIA-NeMo/RL/pull/4588)

#### Transformers 修正静态 KV、缓存续接视觉位置与序列 bias 热路径

**11-10 09:49–11-10 22:12，`main` 默认分支合入。** Gemma 4 的 KV-shared prefill 截为最后一个 query 时也保留未填 cache 槽位的屏蔽。Qwen VL 续接新图或视频时只给未缓存的视觉块重算 3D M-RoPE，累加缓存有效前缀和旧 delta，支持全量 metadata 重建。SequenceBias 与 bad words 在首次调用按 prefix 长度分组，逐组广播匹配，再用 index_add_ 合并同一末 token 的偏置。分别对照 dynamic KV、完整多模态历史和重叠序列规则。 [transformers #50528 `ca774efd8a`](https://github.com/huggingface/transformers/pull/50528) · [transformers #50427 `9f8c7c9409`](https://github.com/huggingface/transformers/pull/50427) · [transformers #50532 `ed0ecab6c4`](https://github.com/huggingface/transformers/pull/50532)

#### TensorRT-LLM 加入 decode 槽位抢占、PrimsTS FP8 KV 与并发 JIT 暖机

**11-10 14:23–11-11 00:16，`main` 默认分支合入。** GEN 满槽时，可暂停在跑的请求让有 TTFT 风险的等待者进入，暂停期间仍保留 KV、recurrent state 和 draft slots，恢复优先并受最大暂停限制。非 MLA PrimsTS 支持 E4M3 KV，Q/K/V 走 FP8、输出保留 BF16/FP16，图重放使用设备端实时 scale。DeepGEMM 的可选线程暖机只给并发安全的 runner，C++ cache 非线程安全的 runner 继续串行。压测需把 TTFT 和被暂停请求 TPOT、内存占用一起记录。 [TensorRT-LLM #21334 `ae134b89eb`](https://github.com/NVIDIA/TensorRT-LLM/pull/21334) · [TensorRT-LLM #19396 `a1c22fc7a5`](https://github.com/NVIDIA/TensorRT-LLM/pull/19396) · [TensorRT-LLM #21610 `37e6e6770e`](https://github.com/NVIDIA/TensorRT-LLM/pull/21610)

#### llm-d 在 NVIDIA optimized baseline 接入 SGLang Rust server

**11-11 02:35，`main` 默认分支合入。** NVIDIA GPU overlay 对已固定的 SGLang 0.5.22 设置 `SGLANG_RUST_SERVER=1`，使 HTTP/tokenizer 入口走 Rust 进程。PR 验证 /v1/models 和 /v1/completions。迁移已有自定义请求处理时可按这个 overlay 核对 headers、tracing 和 prefix cache 路由。 [llm-d #2901 `b0621c6a57`](https://github.com/llm-d/llm-d/pull/2901)

#### vLLM 补 Kimi K3.1 主线分块、sharded RDT 与 CUDA 构建基线

**11-10 10:23–11-10 21:39，`main` 默认分支合入。** Kimi K3.1 第 4/6 段带来 ROCm TP8 及 NVIDIA 专用解码更新，FP8 indexer 固定 64-token kernel block 避免落入会破坏图捕获的 eager 回退，ROCm DCP/P/D 仍由下一段交付。sharded RDT 可用页锁 host staging 异步分发到 GPU，并平行从多 owner stream 拉取，默认仍是 GPU staging/单流。CUDA 主线构建从 13.0 改 13.2，与 torch 2.13 工件匹配。复测真实拓扑的 reload 以及 wheel/容器依赖解析。 [vllm #66532 `dc2137cd39`](https://github.com/vllm-project/vllm/pull/66532) · [vllm #61076 `c021a8b0a9`](https://github.com/vllm-project/vllm/pull/61076) · [vllm #55387 `3996083d37`](https://github.com/vllm-project/vllm/pull/55387)

#### SGLang 优化异构 TP 的 NIXL KV、BF16 KDA verify 与 prompt 编码

**11-10 10:23–11-11 06:25，`main` 默认分支合入。** 当 prefill attention TP 小于 decode TP 时，显式开启 KV packing 将对应 head slice 先聚成页粒度，再传给目标 rank，适用特定 NHD full/SWA 池且无 draft KV。Kimi KDA 用 BF16 recurrent state 时可走 fused CuTe verify，FP32 模式独立。DeepSeek V4.1 segmented fastokens 保持特殊 token 分界，作者在 4×B300 TP4 的 15 分钟 AgentX 测到 C4–C64 平均 TTFT 降 36–49%，吞吐约 ±1%。验证 token ID、传输后 KV 与图捕获。 [sglang #44615 `a6f190e8fe`](https://github.com/sgl-project/sglang/pull/44615) · [sglang #48613 `8d5411e478`](https://github.com/sgl-project/sglang/pull/48613) · [sglang #48928 `219b7e7a68`](https://github.com/sgl-project/sglang/pull/48928)

#### Dynamo 改进 runtime LoRA 放置和跨副本 KV 搬移

**11-10 10:23–11-11 07:29，`main` 默认分支合入。** Dynamo 将 LoRA hosted placement 和 KV-aware placement 分开，dispatch 前重验 adapter identity 与 worker 资格，runtime adapter 与 legacy demand accounting 分账。KVCR 先选 READY 子集再验证实际传输的 pinless destination，消息接收时间纳入 deadline，过期请求在占用资源前拒绝。PR 给出 Rust 与单测结果，GPU live serving 仍需做 adapter 热切及稀疏 KV 命中对照。 [dynamo #17895 `3a524d0d70`](https://github.com/ai-dynamo/dynamo/pull/17895) · [kvcr #182 `6d54642dbb`](https://github.com/ai-dynamo/kvcr/pull/182) · [kvcr #186 `b7aa81d05a`](https://github.com/ai-dynamo/kvcr/pull/186)

### 内核、编译与通信

#### FlashAttention 将 SM100 hd256 backward 并入通用 kernel，铺设 MXFP8 QK

**11-10 11:46–11-11 05:03，`drisspg/stack/81`、`main` 主线与特性分支分别合入。** 通用反传成为 hd256 默认，覆盖 deterministic、score/mask mod 等路径。GQA 的 hd256 selector 只有 2CTA 可用才让出 PackGQA，避免实际回退 1CTA 后重复读 KV。主线合入 E4M3 payload 及每 32 元素 E8M0 scale 工具。栈分支的 MXFP8 QK 保留 FP32 softmax 及 BF16 PV。按 page size、GQA 比值、确定性和量化尺度分别对照。 [flash-attention #2968 `c42f45c2b1`](https://github.com/Dao-AILab/flash-attention/pull/2968) · [flash-attention #3148 `c055bdc644`](https://github.com/Dao-AILab/flash-attention/pull/3148) · [flash-attention #3018 `dba62ef0f0`](https://github.com/Dao-AILab/flash-attention/pull/3018) · [flash-attention #3019 `69384e9c2e`](https://github.com/Dao-AILab/flash-attention/pull/3019)

#### Triton 改善 Blackwell TMA 描述符并扩充 CDNA5 分区 shared 布局

**11-10 17:58–11-11 07:48，`main` 默认分支合入。** SM90/100/103 编译期打包已知的 tensor map 字段，运行期仅替换地址与动态字段。SM100/103 + PTX ≥9.5 依据包含 padding 的跨度保留或动态选择大分配类。SM103 64-value FP6 unswizzled host TMA store 扩至 2–5 维，SM100 保持拒绝。CDNA5 的 TDM gather/store/scatter 支持按行分区的 shared 布局。验证应覆盖阈值两侧、动态 stride、FP6 guard 区和行分区外的拒绝路径。 [triton #12689 `2811380e19`](https://github.com/triton-lang/triton/pull/12689) · [triton #12685 `51e03b402c`](https://github.com/triton-lang/triton/pull/12685) · [triton #12651 `fbd70d9e44`](https://github.com/triton-lang/triton/pull/12651) · [triton #12702 `edb9e623fb`](https://github.com/triton-lang/triton/pull/12702)

#### PyTorch 改善 FlexAttention 边界索引、Inductor profiling 与 XPU linear fusion

**11-10 14:25–11-11 06:23，`main` 默认分支提交。** FlexAttention 对 captured-buffer 尾部 padding 索引用 clamp，避免 CUDA kernel 中成本更高的取模。Inductor 的 profiler 启用时沿用正常启动路径以减小测量扰动。XPU 把多路 linear 拼成一次宽 GEMM，切片保持真实宽行 stride，内部按 consumer 的布局要求 materialize，而对用户可见输出显式 clone。应对照编译/未编译产物的合法索引输出、profiler 开关和外露 view 布局。 [pytorch #204156 `47f5506dbe`](https://github.com/pytorch/pytorch/commit/47f5506dbe1a77618eb56c2575da27c89e013b65) · [pytorch #203702 `629c4e54cd`](https://github.com/pytorch/pytorch/commit/629c4e54cdb0da696711f4780b334dc183fe77e1) · [pytorch #201951 `2ef49f2780`](https://github.com/pytorch/pytorch/commit/2ef49f2780318ed900268e19ce91f88737a313ac)

#### PyTorch 补 NCCL2 合批 P2P 的完成信号与 collective 诊断

**11-11 03:43–11-11 05:45，`main` 默认分支提交。** NCCL2 coalesced send/recv 的 sentinel 共享整个 P2P batch 的结果 future，FlightRecorder 等合批真正完成再退役条目，计时属于整批。图捕获和未闭合的窗口没有这类完成事件。分布式 fingerprint 记录 GC 计数及单调时间，辅助区分 host 停顿与 collective 失配。用相同 trace 粒度和 batch 边界对照。 [pytorch #203539 `141cf7e1de`](https://github.com/pytorch/pytorch/commit/141cf7e1de1952890cdaf33e02f2c032b9c30b02) · [pytorch #194421 `8b4670c9f1`](https://github.com/pytorch/pytorch/commit/8b4670c9f1870471cfeed70ba3c7535162a58990)

### 发行与工件

#### Transformer Engine 2.21.0 扩充训练精度与编译范围

**11-11 04:56，正式发行。** 实验性 cuDNN KimiDeltaAttention、UE5M3 block scale 的 NVFP4 原型进入正式版。FP16/BF16 FusedAttention、Linear+Bias 和 FP8 首微批权重缓存扩充 fullgraph。FA3/FA4 支持序列间 padding 的 THD 和 P2P/all-to-all CP，hd256 backward 需要 FA4 ≥4.0.0b33。SM100+ 的 MXFP8 MLA RoPE 只覆盖 DeepSeek-V3 head 形状。迁移时核对 fused backend 与确定性配置，并对目标硬件跑带 padding 的反传。 [v2.21.0 `7457e47d26`](https://github.com/NVIDIA/TransformerEngine/releases/tag/v2.21.0)

#### TPU inference 0.33.0 发布 FlyWheel、MoE 路由及多步 hybrid decode

**11-11 08:04，正式发行。** 先前已记录同一 tag 和 PyPI 轮子，本次是 GitHub 正式发布及完整变更说明。FlyWheel 在 torchax attention/GDN 路径支持 FP8 KV，flax_nnx 会在启动拒绝。tokamax 的 fused EP/routing MoE、TensorCore gather/combine 及两阶段精确 top-k 都是可选。Qwen3-Next/3.5 接通异步 continue_decode。Gemma 4 视觉 encoder 重新 JIT，项目在 tpu7x 的 26B-A4B-it 测到 1.19→5.41 req/s。发行记录验证了特定 cut 和 vLLM LKG，部署应核对 TPU 代际、插件与 FP8 配置。 [v0.33.0 `a14edd7a4a`](https://github.com/vllm-project/tpu-inference/releases/tag/v0.33.0)

#### vLLM Ascend 0.32.0rc1 默认 MRV2 并拓展混合模型 P/D

**11-11 00:45，预发布构建。** 以 vLLM 0.32.0、CANN 9.2.1、torch-npu 2.10.0.post7 为基线。MRV2 默认启用且不兼容配置自动回 V1，Mooncake layerwise 支持按组件传 KV 和 recurrent state，AscendStore 扩展 Qwen3.8 hybrid pooling 与 GLM RecoverSSM prefix reuse。GLM pooled indexer 默认 BF16，FP8 限 950DT、INT8 限 A3。用官方 RC 镜像核对 runner、实际命中的缓存后端及 P/D 拓扑，评估废弃动态 CPP/GQA DCP 的替换路径。 [v0.32.0rc1 `24487f4f48`](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.32.0rc1) · [PyPI](https://pypi.org/project/vllm-ascend/0.32.0rc1/#files)

#### FlashInfer 0.7.4rc2 补 batch-invariant 低精度 GEMM 与边界修复

**11-10 17:22，预发布构建。** RC1→RC2 比较页列出 MXFP8 和 FP4 GEMM 的可选 `batch_invariant=True`、two-sided MoE combine 的空本地批次、FP8 ReplaySSM checkpoint scale 与 KDA 非有限 V padding 修正。PyPI 可见该 RC 的 flashinfer-python 工件，支持和性能应在目标 GPU 对比不同 M、空 rank 与 recurrent-state 数值。 [v0.7.4rc2 `6d41f103c5`](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.7.4rc2) · [RC1→RC2 对照](https://github.com/flashinfer-ai/flashinfer/compare/v0.7.4rc1...v0.7.4rc2) · [PyPI](https://pypi.org/project/flashinfer-python/0.7.4rc2/#files)

#### Attention Gym 0.0.19 发布 QSA grouped sparse gather attention

**11-10 10:28，正式发行。** `gather_attn` 的 grouped KV heads/独立 V 获得 Triton forward 与 backward，包含 dense tile 和小 grouped heads 的原子稀疏 dK/dV，并补 CUDA graph backward 测试。逐位可复现的 recurrent delta-rule scan 也进入这个 tag。接入时检查实际 schedule、共享存储梯度以及 FA4 编译/图路径。 [v0.0.19 `0d0197fb36`](https://github.com/meta-pytorch/attention-gym/releases/tag/v0.0.19)

### 部署与评测实践

#### llm-d 对照真实 agent trace 的保留字段与六种重放方式

**11-09 发布，补录。** 作者按六种工具能原生加载的格式对照数据。Mooncake 的 23,608 条请求保留到达时间、token 长度与前缀哈希，但没有 session 依赖。TraceLab 保留 4,265 个 session 的依赖信息，内容已去除。Inference-Perf 可用前驱响应推进图，vLLM bench serve 对独立请求按时间发送，DynoSim 模拟 engine。测 prefix cache 时可用哈希 trace，测完整交互尾延迟需保留因果依赖及相同重放目标。 [llm-d 作者文章](https://llm-d.ai/blog/from-agent-traces-to-replay)

<a id="hardware"></a>
## 硬件与软件协同

### TPU 与 Trainium 的推理路径

- **Muse Spark Open 在四颗 TPU v7 上将 62 层解码或验证合成一个 Pallas 程序。** PyTorch 与 RadixArk 的技术报告对应 SGLang-TorchTPU 分支 `haolin/ms-v11`，公开分支基线 `24e2b177`。四颗 Ironwood 暴露八个设备，模型按 TP8 分片，NVFP4 专家权重配 FP8 计算。每设备每步把 62 层和末尾归一化合在一次调用，层内用远端 DMA 传 TP 部分结果，提前搬运权重和 KV，嵌入、draft、KV 写回、LM head 与采样仍在外部。图 4 在较早 Muse Spark checkpoint、GSM8K 和相同 DFlash2 验证下，单请求每输出 token 由 3.78 降至 1.95 ms，两并发为 1.85 倍，高并发增益收窄。§7 将 host DRAM 用作全局、滑窗和 draft KV 池的 L2，按会话保护工具暂停期间的长前缀。图 1 的 1M AgentX 是 393 段结构化回放、替换了内容 token，吞吐混计输入和输出，GB200 对照来自 11 月 2 日外部结果。**建议深入阅读并复测**图 4 与 §7 的独立增益，特别是长前缀、低并发和高并发 prefill 上限。上游主分支合入仍在计划中。 [技术报告 · 11 月 11 日 00:45:24](https://pytorch.org/blog/optimizing-muse-spark-open-on-tpu-v7-with-sglang-torchtpu/) · [固定代码目录](https://github.com/sgl-project/sglang-torchtpu/tree/24e2b177e24270c52aa8a80a51d7f8bee74097b3/sglang_tpu/muse_megakernel)

- **补录 TrnRT v0.1.0，Trainium2 的 GLM-5.3-Flash 开源服务栈。** RunAnywhere 11 月 4 日发布技术博客，11 月 8 日发布 Apache-2.0 代码首版 `v0.1.0`（北京时间 11 月 9 日）。针对 decode 中的小指令、kernel 边界和裸露的集合通信，NKI 路径约每半层一个融合 kernel，复用 SBUF 并将集合通信移入 kernel。前缀缓存按 FP8 KV 页维护 radix tree，淘汰页和状态检查点可进主机内存。项目方以一台 `trn2.48xlarge`、16 芯片、Neuron SDK 2.32、GLM-5.3-Flash FP8、1M AgentX、32 并发、每点一小时测试。decode batch 12 达 3.57M 总 tokens/min、每用户 p50 22.3 decode tok/s，含已缓存的输入 token。报价用 11 月 3 日俄亥俄区 spot 8.60 美元/小时，换容量块则从 0.040 变为 0.167 美元/百万 token。指标测自较早构建，公开版报告 157 个编译图等价、384 条输出逐字相同，性能仍待复测。**考虑复现**同一固定权重与冷、热前缀切分，再与其他设备比较。 [正式版 · 补录 · 11 月 9 日 03:59:47](https://github.com/RunanywhereAI/trnrt/releases/tag/v0.1.0) · [NKI 设计与评测](https://github.com/RunanywhereAI/trnrt/blob/v0.1.0/docs/MEGAKERNELS.md)

- **SGLang-TorchTPU 将解码节点 DRAM 接为预填充 HiCache 的下层。** 预填充 host KV 淘汰前向服务过该分支的 decode 节点推送固定内存页，接收端预留并拉取，ACK 成功后本地副本才成为下一淘汰候选。复访时按各 rank 可用的连续前缀回拉，提交后再供预填充请求使用。两端均需开启且 attention TP 对齐，覆盖以 full-attention host 页为主。补丁附协议、调度与失败路径测试，缺少跨机成本数据。**考虑复现**长会话工具暂停后命中率、两次搬运与重算开销。 [sgl-project/sglang-torchtpu #1992 · 已合入 · 11 月 10 日 17:01:18](https://github.com/sgl-project/sglang-torchtpu/commit/3dabdc26f5f6f69eda60900d4d6d59cf5a6247be)

- **SGLang-TorchTPU 分散 DSA prefill 索引计算并扩大长上下文边界。** GLM-5.3-Flash 的 pooled DSA scorer 在多 attention TP 且达到最小 token 门槛时按 query token 切分并汇集结果，Kimi 配方共用该策略。独立提交为 GLM 补 1M 页索引路径并调整索引工作集，BatchedRPA 在 128 及以上 query heads 下缩小 prefill tile 以降低 VMEM 编译失败风险。**考虑复现**多 TP 1M 预填充及混合 batch，仓库补丁覆盖编译或功能边界，缺少共同吞吐基线。 [sgl-project/sglang-torchtpu #2040 · 已合入 · 11 月 11 日 01:22:51](https://github.com/sgl-project/sglang-torchtpu/commit/d62642fff9acefcd8a9e7789a2f2db444f5a2fc9) · [sgl-project/sglang-torchtpu #2045 · 已合入 · 11 月 10 日 13:52:34](https://github.com/sgl-project/sglang-torchtpu/commit/130612ca1df2b183939da879f45f4409a107da81) · [sgl-project/sglang-torchtpu #1920 · 已合入 · 11 月 10 日 22:26:18](https://github.com/sgl-project/sglang-torchtpu/commit/8346a015e3c6270a389e0e53c4efa16ae42a5128)

- **两条 TPU 推理栈修复 GDN 或 KDA 前缀状态。** vLLM-TorchTPU 为 MTP verify 跨 Mamba 对齐块保存真实接受边界的 GDN 状态，使 decode 填出的块可供后续前缀复用。SGLang-TorchTPU 处理以 extend 执行的 decode 行，补写应跟踪的 KDA 边界。两边各有回归用例。**优先验证**跨块接受、拒绝回滚及短请求混批后续命中与无缓存结果的一致性。 [vllm-project/vllm-torchtpu #2937 · 已合入 · 11 月 10 日 14:25:29](https://github.com/vllm-project/vllm-torchtpu/commit/c350823af74ba9cdf728daad64df7dd5bcf84e8e) · [sgl-project/sglang-torchtpu #2039 · 已合入 · 11 月 10 日 09:29:29](https://github.com/sgl-project/sglang-torchtpu/commit/1a897bfece9d3888af1f581ab0fc8810108b3c07)

- **vLLM-TorchTPU 增加 megakernel 路径判定与静态专家布局。** 一项提交补齐运行时 bucket policy 与逐步 eligibility 的守卫和单元测试，另一项在加载阶段按固定映射重排 MoE 专家并对齐 router 与权重布局。两项分别影响大内核调度边界和模型加载，公开证据为补丁与测试。**继续观察**对应内核覆盖、路由分布变化后的收益及端到端指标。 [vllm-project/vllm-torchtpu #2908 · 已合入 · 11 月 11 日 01:24:44](https://github.com/vllm-project/vllm-torchtpu/commit/0ef9144a66078fea32b994641769e2f1cccc088b) · [vllm-project/vllm-torchtpu #1408 · 已合入 · 11 月 10 日 12:12:25](https://github.com/vllm-project/vllm-torchtpu/commit/229f2e7d19ff1491ae98fbe41c6a049683178d6a)

- **TorchTPU 与 TPU Sync 修复异步拷贝、JAX donation 和注册寿命。** TorchTPU 对 pageable 目标的 D2H 即使传入 non_blocking 也等到数据可读，并修正负数 `donate_argnums` 的参数索引。TPU Sync 领取请求时刷新生产者注册 TTL，减少长期运行中缓存生产者过期。证据来自对应提交和回归用例。**考虑复现**异步输出消费、负索引包装及长时间缓存传输。 [google-pytorch/torch_tpu 3d1de23e · 已合入 · 11 月 11 日 06:41:19](https://github.com/google-pytorch/torch_tpu/commit/3d1de23edc853753cd5a7ee041c3633f4183c7cd) · [google-pytorch/torch_tpu ef3d62c9 · 已合入 · 11 月 11 日 08:41:53](https://github.com/google-pytorch/torch_tpu/commit/ef3d62c944060cd2854f3971045c4ddd70e84192) · [google/tpu-sync 63c6447c · 已合入 · 11 月 11 日 02:45:14](https://github.com/google/tpu-sync/commit/63c6447cc4850f923831b6f0636b7a63405ac246)

- **vLLM Gaudi 修复 padding 覆写真实 KV 页。** Mamba 混合模型将一个管理块分成多个 kernel 块，旧 padding ID 用管理块计数，落在真实 attention KV 页内。新代码按 kernel 块数量分配哑页，初始化核查容量与各组切分是否一致，Mamba 状态仍用其管理块索引。项目新增多种布局及不一致切分的单测。**优先回归**混合模型 decode 补齐、投机 draft 与高占用缓存。 [vllm-project/vllm-gaudi #1912 · 已合入 · 11 月 10 日 19:28:29](https://github.com/vllm-project/vllm-gaudi/commit/57cec1ec789a9eebe5338312039133e94f58fc47)


### AMD 推理、通信与训练

- **ATOM 为 DeepSeek-V4.1-Flash 合入两条 FlyDSL 低延迟解码路径。** MI355X 路径每 decoder 层一次持久 kernel，当前层推送 MoE 部分和、下层归并，验证 head、DSpark 写入和 draft 同步接入。自动启用有明确约束，包括 TP4、BF16 KV、FP4 index、DSpark 5 草稿、Engram host staging、无 KV transfer。gfx1250 路径另行融合整步并有配置门控，提交记录在 TP4、DSpark5、BF16 KV、FP8 index 的 48 行形状测得 kernel 7.48 ms。这两条是不同部署的项目方结果。**建议深入阅读**路由门控与原生页表交接，复现时分开记录低并发 TPOT 与回退比例。 [ROCm/ATOM #3091 · 已合入 · 11 月 11 日 00:30:50](https://github.com/ROCm/ATOM/commit/5b4fbfdec4b199f64922cae39967d02814e6b858) · [ROCm/ATOM #3109 · 已合入 · 11 月 10 日 23:29:46](https://github.com/ROCm/ATOM/commit/03b584b815c167d132396ba58efb111aab200db0)

- **ATOM 为投机解码接入基于 draft 概率的拒绝采样。** 在 drafter 发布分布 q 时，按目标 p 与 q 的比率接受，拒绝后从归一化正残差采样，缺省仍使用原 exact-match 路径。补丁记录了含概率元数据的调用路径和分布检验。**考虑复现**带温度的 drafter 在相同目标分布下的接受率、词表计算与端到端延迟，greedy draft 的 q 退化时没有接受率收益。 [ROCm/ATOM #2582 · 已合入 · 11 月 11 日 00:25:23](https://github.com/ROCm/ATOM/commit/06e693603bf83bf14354f7d28395331bbebc28f7)

- **ATOM 修正 Kimi P/D 状态与缓存完成语义。** PP8 prefill 引入分阶段 PAGE 状态检查点。in-process LMCache 状态 store 等待 slot→PAGE 拷贝 fence，避免在状态可读之前发布。另两项分别在状态层命中时刷新 LRU 与索引，以及在 engine 空步期间仍轮询 P/D 完成。**优先验证**短流量空窗、长前缀重入、缓存淘汰和跨 TP 失败传播，补丁无统一业务压测。 [ROCm/ATOM #2709 · 已合入 · 11 月 10 日 11:10:34](https://github.com/ROCm/ATOM/commit/cd5fa97dadf363ae3c4af2bed17350a4b7bbccf5) · [ROCm/ATOM #3155 · 已合入 · 11 月 11 日 00:10:40](https://github.com/ROCm/ATOM/commit/4d45f25103105df9772e3ea6a171e421042a944c) · [ROCm/ATOM #2668 · 已合入 · 11 月 10 日 10:39:15](https://github.com/ROCm/ATOM/commit/5f5019f7aaec6ae7e75a12668be7d0a9214e4827) · [ROCm/ATOM #3105 · 已合入 · 11 月 10 日 09:26:29](https://github.com/ROCm/ATOM/commit/456fb6605894ef870f69d9df44b577f0b5bbd00f)

- **AITER 增加混合精度 attention 并修复 MoE 与 MLA 边界。** gfx1250 unified attention 增加 FP8 激活、NVFP4 K、FP8 V 的 A8K4V8 路径。MoE 动态 per-tensor 量化前清零 stage 1 未覆盖的 a2 行，避免脏值抬高 scale 或产生 NaN。gfx950 FP8 GQA16 的 MLA merge 只读 stage 1 写过的 split，同时常规和 persistent decode 各补 LDS 同步。**考虑复现**量化前后长上下文误差、空 expert 分组与持久 kernel 的 NaN/LSE 回归。相关 PR 的内核测试与条件各异。 [ROCm/aiter #7421 · 已合入 · 11 月 11 日 08:20:28](https://github.com/ROCm/aiter/commit/a8780dfe65a801b62dcfc681d74d8b81ef571341) · [ROCm/aiter #6926 · 已合入 · 11 月 11 日 01:06:46](https://github.com/ROCm/aiter/commit/892e5d003df529516c864f90b2fc0e0f47f093a2) · [ROCm/aiter #7613 · 已合入 · 11 月 10 日 17:45:11](https://github.com/ROCm/aiter/commit/dcff721ff8f19706d1aa69a2ef78846fd7ead1f5) · [ROCm/aiter #7617 · 已合入 · 11 月 10 日 12:11:59](https://github.com/ROCm/aiter/commit/35aceb8c00ae77139719f1a72a673e5f41e56d5b) · [ROCm/aiter #7622 · 已合入 · 11 月 10 日 12:11:27](https://github.com/ROCm/aiter/commit/714e1944da61265aea831d1f9ed54f544cfef87c)

- **MoRI 为分层 KV 增加命中提示和传输校准，修正 gfx1250 RDMA 注册。** UMBP 的 BatchTouch 让上层设备缓存报告本地命中，下层据此续热，操作无需读对象或获得 lease。另一个提交加入 CPU/GPU、RDMA、SSD 实测校准工具，模拟参数可据真实链路拟合。gfx1250 的 CCO fabric window 改用 dma-buf IOVA 0 注册以配合相应地址语义。**考虑复现**多级缓存同时承压时的真实淘汰顺序、校准误差和链路故障回退。 [ROCm/mori #779 · 已合入 · 11 月 11 日 01:00:10](https://github.com/ROCm/mori/commit/5dc1f3688ec7313eaa7db1e72a1fb0fb3300909f) · [ROCm/mori #942 · 已合入 · 11 月 10 日 15:36:24](https://github.com/ROCm/mori/commit/651fb16adc51f4b8583de5f5479c5a2ddbe77df5) · [ROCm/mori #870 · 已合入 · 11 月 10 日 13:48:03](https://github.com/ROCm/mori/commit/7f4d309931d132635fe90f46c3d69a2d855a1394)

- **Primus 将 MI455X 的 MoE 代理预训练与融合路径合入主分支。** 组合提交包括 DeepSeek-V3 的 opt-in Triton router 和 BF16 expert wgrad 直积累至 main_grad，MiniMax-M3/M3.1-Flash 的 gfx1250 wave32 FlyDSL MSA 和分组 GEMM，以及 Kimi K3.1 Pro 四层 proxy 与 latent MoE DeepEP。作者记录 MiniMax 8 层 EP4 的代理迭代 1575→1024 ms，比较基线为 #1364，Kimi proxy 为 13,202→21,585 tokens/s/GPU。配方依赖对应 Primus-Turbo 版本，未给出全规模质量与训练稳定性。**建议深入阅读**配方与镜像依赖后再复现。 [AMD-AGI/Primus #1418 · 已合入 · 11 月 10 日 11:23:18](https://github.com/AMD-AGI/Primus/commit/7587ff517f79f51328674ba1c34f6a1ae3d27c12)

- **Primus 修复 HF→Megatron 分片兼容并推进 CP 与 MoE 训练路径。** native 转换器默认通过 Megatron 生成带全局形状的 torch_dist，恢复时按 TP、PP、EP 重切分，格式不符或 legacy 单分片尝试多并行时提前失败。其他提交将 Primus-Turbo attention 接入 CP 并保存选择性重算所需输出，合并 MoE router 与 expert 输入梯度写回，修正 persistent DEOSC 逻辑量化分片。项目给出单测与 GPT-OSS-20B MXFP4 短程数据，适用范围各自独立。**优先复现**跨拓扑 checkpoint round-trip 及长程 loss 对齐。 [AMD-AGI/Primus #1406 · 已合入 · 11 月 11 日 08:48:41](https://github.com/AMD-AGI/Primus/commit/4efe48eca7254a197dfa0e617ba823ba9c8047aa) · [AMD-AGI/Primus #1420 · 已合入 · 11 月 10 日 16:15:18](https://github.com/AMD-AGI/Primus/commit/865846318716b73136fca49bc7b1b44447cafd57) · [AMD-AGI/Primus #1372 · 已合入 · 11 月 10 日 10:27:20](https://github.com/AMD-AGI/Primus/commit/cb696c82200cc60453217574bd7cf0b22967513d) · [AMD-AGI/Primus #1371 · 已合入 · 11 月 10 日 10:26:34](https://github.com/AMD-AGI/Primus/commit/045ff7846d85be6c3be73a7356e50fc8375a4b37)

- **ROCm libraries 调整 SDPA dispatch 并修复分片合并。** HIP kernel provider 将 SDPA 接入 unified 与 dense kernel 选择。gfx1151 decode merge 为 padding split 初始化零值，rocBLAS 修复 batched GEMM 在 Tensile 与 hipBLASLt 的 split-K 路径。项目补丁与回归支撑正确性结论，缺少统一端到端对照。**考虑复现**不同 KV 分片数、空片及小 batch。 [ROCm/rocm-libraries #13115 · 已合入 · 11 月 11 日 07:44:37](https://github.com/ROCm/rocm-libraries/commit/f2d79d6eb356bcd41a18ee3e8a0b0e79d08b6432) · [ROCm/rocm-libraries #15235 · 已合入 · 11 月 11 日 04:07:34](https://github.com/ROCm/rocm-libraries/commit/9ed0627f8db59cd6173bc92a511322b912139a5b) · [ROCm/rocm-libraries #12145 · 已合入 · 11 月 11 日 07:12:20](https://github.com/ROCm/rocm-libraries/commit/65de3706b5689019b2cde963e4bcd7406acb1472)

- **MLxcel 扩大 ROCm 原生 4-bit GEMV 并修正性能口径。** q/k/v 与 gate/up 合并发射，ragged engine 让各激活行共享权重解包，Llama-3.2-1B-Instruct 4-bit 在 gfx1151 的 B=8 由 FP16 回退的 59.8 升至 74.5 tok/s，B=4 由 65.6 降至 62.2。ragged LM head 仍是 FP16，另有半写 KV 失效、设备内清零与编译 batch 限制。计时修复使吞吐分子只含 decode 窗口产出的 token，bench 与服务直方图均更新。**考虑复现**B=4/B=8 并按同一计时窗口重新测基线。 [lablup/mlxcel #2907 · 已合入 · 11 月 10 日 14:41:45](https://github.com/lablup/mlxcel/commit/f083bea6e7e67cb1c4947d4be63a62163fac0e61) · [lablup/mlxcel #2933 · 已合入 · 11 月 10 日 16:43:34](https://github.com/lablup/mlxcel/commit/92c07eb8e629b518737b1735d997b25c6eb1b863) · [lablup/mlxcel #2936 · 已合入 · 11 月 11 日 01:50:05](https://github.com/lablup/mlxcel/commit/311ad935776c6fa920752b5a9c53dce0e46013e7) · [lablup/mlxcel #2931 · 已合入 · 11 月 10 日 21:37:01](https://github.com/lablup/mlxcel/commit/f4d4c3ae546726b088581af8828a72a8c3f13cd1) · [lablup/mlxcel #2947 · 已合入 · 11 月 11 日 00:21:47](https://github.com/lablup/mlxcel/commit/37420f42c70aa7fa114d00c6bf0bb23f9f5c8b2d)


### 昇腾通信、算子与运行时

- **HCCL 更换 Host DPU 下发路径并撤回昨日的一项协商改动。** HCCL !3461 在 HCOMM ≥9.2.0.2 且有相应线程 API 时直接向专用 DPU 线程排队搬运原语，用线程通知同步，旧共享内存请求路径作为回退。随后 !3971 撤回昨日 !3738 的公共协商及 hostDPU inline reduce 能力协商，撤回说明未给出原因或回归结果。!3941 的选路改动是不同提交，本轮仅确认 !3738 被撤回。**优先重验**跨版本能力选择及数据一致性，部署依据应以当前固定提交为准。 [cann/hccl !3461 · 已合入 · 11 月 10 日 10:08:32](https://gitcode.com/cann/hccl/commit/4dbf9150ab7845a54b4a5ec5352cd0ace9a91898) · [cann/hccl !3971 · 已合入 · 11 月 10 日 14:08:58](https://gitcode.com/cann/hccl/commit/83b87a7ff518dd0b6111a8f2baad120cfc7fb1de) · [前日报对应的 !3738](https://gitcode.com/cann/hccl/pull/3738)

- **HCCL 将 UBX NDA 扩至更多集合操作并加入并行 AR/RS 对称内存。** Send/Recv/Reduce 在 UBX 350L 的能力约束下补 NDA 模板，FIN_ACK 保证 CCL buffer 可复用。Scatter/Broadcast 依据 CLOS 拓扑和消息大小选 Sequence 或 OmniPipe，零长度切片仍保留通知配对。独立提交为 concurrent AllReduce/ReduceScatter 启用对称内存并调整切分比例和 scratch。**考虑复现**支持机型的拓扑、阈值前后、空切片和并发稳定性，现有合入说明缺少横向性能数据。 [cann/hccl !3708 · 已合入 · 11 月 10 日 21:45:44](https://gitcode.com/cann/hccl/commit/03355d28bc34c5913cbeb6445df0fc4b6d85f45c) · [cann/hccl !3892 · 已合入 · 11 月 10 日 19:42:02](https://gitcode.com/cann/hccl/commit/e9bd3c0a0f6cf637229d9b3e193d3ffbcebbe678) · [cann/hccl !3911 · 已合入 · 11 月 10 日 21:00:10](https://gitcode.com/cann/hccl/commit/de8c245e2f46747cff450b8c324b69edf5bb515e)

- **vLLM Ascend 为 Atlas A2 大 BF16 TP 消息加 INT8 all-reduce。** 默认关闭。启用时先同步每通道 FP32 amax，再按 world size 留求和余量，将 BF16 payload 转 INT8 归约，最后反量化。作者在四张 910B3、TP4、Qwen3.6-35B-A3B、16K 输入/64 输出、32 prompts、2 QPS 下测得 64 MiB 单次 1.9→1.4 ms、平均 TTFT 1971→1473 ms、1.54→1.61 req/s。GPQA 变化在该次测量噪声内，长度 8K 截断或不可解析样本仍多。**考虑复现**目标模型的阈值扫描和长推理精度。 [vllm-project/vllm-ascend #19000 · 已合入 · 11 月 10 日 17:15:39](https://github.com/vllm-project/vllm-ascend/commit/53324b9dd134aebdbe54439f29484c25210ca498)

- **Ascend MegaMoE 推进 A5 共享专家融合与 A3 跨卡去重。** A5 路径把 W4A4 MXFP4 共享专家并入路由专家 kernel，要求激活种类与参数匹配，未匹配时保留独立计算。A3 W4A8 选配 dispatch 每 token/目标卡只传一行，combine 在目标卡按 FP32 加权累加再发回，舍入顺序会改变。A3 合入说明的内部 EP8/16 大 batch 延迟收益约 2.8%–20.9%，小 batch 可能增加开销，有收益门控。**优先复测**相同 batch 分布、跨卡 BS 一致及关闭态逐位兼容。 [vllm-project/vllm-ascend #18053 · 已合入 · 11 月 10 日 21:15:59](https://github.com/vllm-project/vllm-ascend/commit/e4d533d33211f0696bcdf65eda47245d52e428c5) · [cann/ops-transformer !14318 · 已合入 · 11 月 10 日 17:13:54](https://gitcode.com/cann/ops-transformer/commit/a7eb23f444597522475246317c18101d1b2078f7)

- **vLLM Ascend 主分支接入 Kimi K3.1，收紧混合精度 gate 和 MTP 通信。** Kimi 集成覆盖 KDA/PCP、稀疏注意力与 QLI、DCP、配对 KV 与 Mooncake 传输，证据限于 A3/A5 定向 Triton 验证，未证明旧 TP DSA-CP 与 PCP 等价。独立修复让不同量化的 MLA output gate 分开加载。GLM-5.3-Flash 的 MTP attention 接入 FlashComm1，覆盖 TP2 的多种 token 形状，设备性能仍待测。**优先回归**Kimi 多卡投机接受率、gate 精度及 GLM 的 TP/序列并行组合。 [vllm-project/vllm-ascend #20878 · 已合入 · 11 月 10 日 21:10:37](https://github.com/vllm-project/vllm-ascend/commit/8e07a81fc13d5a9c8e715738135fe09d0e5c12b0) · [vllm-project/vllm-ascend #21387 · 已合入 · 11 月 10 日 17:35:20](https://github.com/vllm-project/vllm-ascend/commit/6973892eb5d5cb2c1808890264aed73f7caf761f) · [vllm-project/vllm-ascend #20566 · 已合入 · 11 月 10 日 18:29:58](https://github.com/vllm-project/vllm-ascend/commit/9da543e2354519c0b74a3eb168eb2186902d30d5)

- **vLLM Ascend 收紧 xlite 热路径与 KV 恢复/保留配置。** xlite v2 直接消费 runner 维护的元数据，移除热路径阻塞 D2H 和 padding，并依赖 xlite 0.2.0rc5。另两项将同步外部 KV load 状态传入 SchedulerOutput，避免误跳过恢复，并恢复对上游前缀保留配置的遵从。**优先验证**外部 KV 连接器、MTP rejection 与图模式的混合流量，缺少统一性能基线。 [vllm-project/vllm-ascend #20938 · 已合入 · 11 月 10 日 18:53:29](https://github.com/vllm-project/vllm-ascend/commit/a90ead4271353cada94bd624d622bc17f0e2cfbf) · [vllm-project/vllm-ascend #20810 · 已合入 · 11 月 10 日 18:42:24](https://github.com/vllm-project/vllm-ascend/commit/b7ec7deb3380806ae64466db0c1f344403fe333f) · [vllm-project/vllm-ascend #21229 · 已合入 · 11 月 10 日 13:08:13](https://github.com/vllm-project/vllm-ascend/commit/3ed4259b2ee5a27f067e305b69e9e397a6b596b6)

- **Torch-NPU 改变建链默认值并补图编译入口。** 适用 A2/A3 RootInfo 路径默认每 128 rank 分层，域大小超过阈值且 HCCL 提供两项接口才启用，旧路径仍可回退。ACL Graph Trees 支持标记未完成 warmup，Inductor 新增 index→reshape→index 模式匹配和回归。**考虑复现**大通信域建链时间、graph tree 首次 warmup 与动态索引形状。 [Ascend/pytorch c85c8e3b · 已合入 · 11 月 10 日 15:53:52](https://github.com/Ascend/pytorch/commit/c85c8e3bfe137b9f174e3fb0bcd505b95e2e14ec) · [Ascend/pytorch 37f6a302 · 已合入 · 11 月 10 日 19:59:53](https://github.com/Ascend/pytorch/commit/37f6a302c7f282ca91677cf9fd3d02f2c3b636f1) · [Ascend/pytorch 72ef7477 · 已合入 · 11 月 10 日 20:40:38](https://github.com/Ascend/pytorch/commit/72ef7477cfc254e6a83ffcf3a2ac074f96a1e76b)

- **CANN 修复 attention 同步与 AICPU 失败传播，并扩展 QSFA。** Atlas A2 的 FIA split-KV 增加 MTE3→V 同步，确保 LSE 写出后才复用片上缓冲。QSFA V2 扩到 A2/A3 并支持 LSE、DCP 与 1–128 query heads，A5 也扩 head 范围。Ascend950 FlashAttention backward 保持 int64 dropout seed/offset，AICPU worker 初始化失败在唤醒前置退出状态，避免 host 误判成功后等待超时。**优先测试**交替 split/non-split、超过 2³¹ 的 seed 及订阅失败。 [cann/ops-transformer !15558 · 已合入 · 11 月 10 日 11:45:45](https://gitcode.com/cann/ops-transformer/commit/429fd6d9f3e95c8480638a6b642b4fa8b36d911c) · [cann/ops-transformer !14771 · 已合入 · 11 月 10 日 20:45:41](https://gitcode.com/cann/ops-transformer/commit/6efd8295ca750995fe9baa47c996134dc2d8fffc) · [cann/ops-transformer !15486 · 已合入 · 11 月 10 日 09:47:59](https://gitcode.com/cann/ops-transformer/commit/784d9a65768eb93bb0e995c90f0cca66e59e927e) · [cann/runtime !6157 · 已合入 · 11 月 10 日 19:25:24](https://gitcode.com/cann/runtime/commit/205c3ffcc0230876f64a553233e2898a6ca8dbc6)

- **CANN 补 CPU MoE 短批复用与 host ftrace 采集。** DeepSeek-V4-Flash 配方在 ARM/no-SVE 的 MXFP4 路径让最多四行共享权重块解码，并为 MTP 短批调整分组阈值。作者用锁定的 KT/SGLang/llama.cpp 源码和 A3/CANN 9.1 比较三组各 45 个真实权重用例，BF16 输出逐位一致。Runtime 的 ftrace 任务负责权限、节点锁和子进程寿命，数据解析与二进制在另两个组件。**考虑复现**短批延迟及 host stall 定位时先核对三组件配套。 [cann/cann-recipes-infer !1238 · 已合入 · 11 月 10 日 10:15:06](https://gitcode.com/cann/cann-recipes-infer/commit/a37f29f280f24b755b49deebc5aaee4d6153fb67) · [cann/runtime !6163 · 已合入 · 11 月 10 日 22:20:41](https://gitcode.com/cann/runtime/commit/28927c0aac547a90759ec496d0bcc0b707fe5d1e)

- **CANN 公布 DeepSeek-V4.1-Flash 小批量 MegaKernel 技术分享。** 活动详情原发于 11 月 6 日，本轮 11 月 10 日 19:00 排程，社区首页已列为回放。公开摘要描述 L20/L24 的 Indexer 与 Attention 融合边界、tiling 与流水优化。已确认活动与技术主题，未核到固定公开源码和可复测数据。**继续观察**回放配套材料的版本、卡型、输入长度与基线。 [技术活动排程 · 11 月 10 日 19:00:00](https://cann.csdn.net/6aed4fdb40858c2d8818ddae.html) · [回放状态](https://cann.csdn.net/)

- **昇腾 A2 的 236B verl 案例拆开首步长尾与 reward 排队。** 昇腾社区 11 月 10 日刊载两份项目方案例，仅披露日期。首步案例的 rollout 最终用 40 台、320 卡，TP8、EP40、DP5。跳过已离线完成的初始验证、HDK 25.5.2 IPC 权重同步及并行/绑核组合改动后，作者记录首步约 4–5 小时降至 1 小时内、稳态单步约 30 分钟，权重同步约 40→1 分钟。奖励侧用 Qwen3-235B-A22B，实际约 40% 样本需要模型打分，统一镜像、异步提交和多实例负载均衡后从 36 台降至 8 台 A2，未验证四台为生产方案。多项改动同时生效，不能单独归因。**建议深入阅读**阶段耗时拆分，复测时分别核算首次编译、权重传输和 reward 队列。 [昇腾社区案例 · 仅披露日期 · 11 月 10 日](https://www.hiascend.com/ascendgateway/ascendservice/tech/details?techColumnCode=20261110-2) · [昇腾社区案例 · 仅披露日期 · 11 月 10 日](https://www.hiascend.com/ascendgateway/ascendservice/tech/details?techColumnCode=20261110-1)

- **昇腾社区记录 256P 评估停顿及旧 HDK 单进程多卡启动冲突。** 两份 11 月 10 日案例只给日期。910B3、CANN 8.0.1、HDK 24.1.0.x 的 256P 作业在每十步评估处触发等待超时。作者从未正常上报超时的进程定位到 evaluate 与 get_data，调整评估和数据加载后恢复运行，未分离各措施贡献。另一案例将 HDK 23.0.6 单进程多卡启动失败定位到 expandable segments 与旧驱动兼容性，单算子独立运行正常。**建议对照阅读**应用栈、驱动日志和版本矩阵，避免把超时或 cast 首报直接归给硬件。 [昇腾社区案例 · 仅披露日期 · 11 月 10 日](https://www.hiascend.com/ascendgateway/ascendservice/tech/details?techColumnCode=20261109-201) · [昇腾社区案例 · 仅披露日期 · 11 月 10 日](https://www.hiascend.com/ascendgateway/ascendservice/tech/details?techColumnCode=20261110-3)

<a id="coverage"></a>
## 覆盖与续查

### 模型、后训练和评测来源覆盖

- 主窗为 (2026-11-10 00:58:40Z, 2026-11-11 01:00:49Z]，重叠从 2026-11-07 00:58:40Z 开始。逐项以 UTC 官方发布时间、固定工件提交或许可时间归档，并在正文转换为北京时间。没有精确开放时刻的 Alpamayo 2 Edge 按模型卡日期记，正文另标有证据的卡片修订时刻。已报告的模型首发与新的模板、解码器、数据、许可修订分别按稳定 ID 去重。
- 官网、新闻、研究、API、榜单和候选发现入口沿用前轮清单，121 个直接入口中 86 个 HTTP 200、21 个无响应、7 个 403、5 个 404、1 个 429、1 个 302。OpenAI 新闻与研究、Mistral、xAI、Kimi、Ai2 等部分入口由浏览或官方 RSS 复核当前内容。Qwen 旧博客出现迁移提示，DeepSeek API /news/ 返回入门页。动态历史、登录墙、未验的语言版本和先前缺口仍保留，单条源恢复不关闭整个区间。使用中英双语搜索、官方 RSS、HF、ModelScope 及候选追踪站，报告技术结论使用已打开的一手页面和工件。
- HF 全站按 createdAt 翻过重叠起点，模型 14 页、14,000 条元数据，其中 13,031 条落在重叠窗。数据集 8 页、8,000 条元数据，其中 7,066 条落在重叠窗。并检查 154 个定向组织的 308 个模型与数据集 lastModified 首批排序请求，均返回 200，筛得 268 条重叠窗候选。全站 lastModified、超出组织首批页的历史和全部候选的语义核验仍待查。
- 逐仓发起 1,051 个定向提交 API 查询，475 个返回 200，551 个返回 429，25 个返回 401。重点仓使用 57 条独立的官方提交网页作补充，重新打开关键差分和数据卡。候选的每次返回与可用记录见同目录 revisions.json、details.json 和 ledger.json。新增重点待查包括 XYBench、AgentPerf v2、两份 Muse Spark Open 变更、GR00T 2 权重、dNaViT 权重和 Nota AI 的 Nemotron 剪枝工件。保留上轮全部 269 个去重后的历史来源缺口。浏览成功、访问列表成功、提交元数据存在和发布内容已核实分别记状态。
- **arXiv 公告与版本。** 11 月 10 日九分类合并 2,110 个 ID，其中 1,175 个至少一次列为新稿或交叉列示，935 个只见修订。OAI 的 11 月 10–11 日 cs/stat 桶分页至末页，取回全部公告 ID 的版本历史。按版本提交秒，重叠窗看到 997 个 v1 与 857 个后续版本，最晚可见提交为 11 月 9 日 18:59:54 UTC。Atom 精确提交与独立更新时间查询受限流或连接错误，主窗实际提交和待索引修订继续补查。 [cs.LG 公告](https://arxiv.org/list/cs.LG/new?show=2000) · [OAI-PMH 接口](https://oaipmh.arxiv.org/oai?verb=Identify)
- **主窗末端公告。** 11 月 11 日 01:00 UTC 是常规周二 20:00 EST 的公告时刻。本轮 01:16 UTC 重新请求九分类的新稿页，带新查询且未命中边缘缓存，仍显示 11 月 10 日。cs.CL、cs.LG 的 RSS 也仍为 11 月 10 日。官方将 11 月 11 日周三列为假日，当地时钟页给出的下一提交截止为周四，下一批最早为 11 月 13 日 01:00 UTC。周二批的状态与公开时刻尚未确认，列为延迟可见待查。 [公告及假日日历](https://info.arxiv.org/help/availability.html#on-deferred-mailings) · [arXiv 当地时钟](https://arxiv.org/localtime) · [cs.CL RSS](https://rss.arxiv.org/rss/cs.CL)
- **会议与发现入口。** AACL 六个工作坊的入口已核，正文选读 PlurVA、NORA。OpenReview 的 TMLR、NeurIPS、ICLR API 均返回 403。EMNLP、NeurIPS、TMLR、PMLR、MLSys、OSDI 和 SOSP 仍以索引或局部列表为主。HF 日榜 11 月 9 日为 34 项、11 月 10 日为 38 项、11 月 11 日返回 400。机构入口和中英文检索作为候选发现，精确发表时间需逐项查。 [AACL 工作坊](https://2026.aaclnet.org/program/workshops/) · [EMNLP 论文集](https://aclanthology.org/events/emnlp-2026/) · [NeurIPS 会议论文](https://neurips.cc/virtual/2026/papers.html) · [TMLR 期刊](https://jmlr.org/tmlr/papers/) · [PMLR](https://proceedings.mlr.press/) · [HF 11 月 10 日日榜](https://huggingface.co/api/daily_papers?date=2026-11-10)
- **开源工程。** 检查 66 个仓库入口，61 个默认分支、61 个主窗 merged PR 分页闭合，重点 16 个仓库完成回溯窗 PR 分页。61 个官方 release Atom、27/30 个 PyPI 工件库存可读。取得 908 个已合入 PR 的正文并按候选筛选，入选事件中 82 项另有补丁。slime、OpenRLHF 主窗默认分支和已合并 PR 均无新增。维护分支、候选语义抽查、有限 feed 和受阻端点保留续查。
- **硬件与软件协同。** 发出 331 次请求，含补丁与代理复查。查阅 NVIDIA、AMD/ROCm、Google TPU/GKE、AWS Neuron、Intel Gaudi、昇腾/CANN、MUSA、寒武纪、MLX 与 MLCommons 的公开入口或组件仓。通过代理恢复 GitCode HCCL、Runtime 和算子目录，TorchTPU、Ascend/pytorch、ATOM、AITER、ROCm libraries 等补 REST 分页。昇腾公告与技术列表以页面 Referer 恢复首屏，公告第 2 页尚未核实，日期提前显示的 11 月 20 日条目留待后续。Azure HPC 403，部分沐曦、寒武纪、libtpu 入口失败或仅有浅层目录。CANN 产品与组件版本映射、旧维护分支、未审回溯候选仍待查。TrnRT 首版为补录。HCCL !3971 撤回前日报所记 !3738 的协商部分。

- **续扫边界。** 本轮发现截止为 2026-11-11 09:00:49。下轮从该位置续扫并回溯 72 小时，同时继续处理分源待查记录。全局完整语义水位仍未闭合。本轮未做独立复训、设备压测或付费 API 实测。
