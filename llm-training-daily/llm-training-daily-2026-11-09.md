# LLM 技术动态日报 · 2026-11-09

> 时区 Asia/Shanghai（UTC+8）。增量扫描窗口 **2026-11-08 09:00:30 → 2026-11-09 09:40:38**，向前回溯 72 小时至 **2026-11-05 09:00:30**。实际生成时间 **2026-11-09T09:54:28+08:00**。日期未披露到时分时保留原日期。首次发布、工件或代码更新、会议报告与本轮发现分别标注。
>
> 生成前核对本地 33 篇历史日报及索引，与 GitHub `main` 的 `90f951b12cd40166d35c87d6b5cf951a086c6db7` 逐字一致。前次待上传队列为空。
>
> 本轮核验公开的一手报告、工件、发布记录及部分代码补丁。性能与能力指标为原作者或项目方的口径。

## 本次重要变化

- **模型与训练配方。** PFN 开放日语金融 PLaMo-3-8B-Fin-Instruct，模型卡披露金融语料持续预训练、SFT 与 DPO。工具调用模板和 32K 金融继续训练长度给出了复测边界。

- **训练与蒸馏。** Megatron-LM 新增按全局主损失 token 数归一化 MTP 损失的开关。DeepSpeed 在 HybridEngine 的图执行路径上补齐 rollout 采样，并实现词表并行 JSD，适合分别核对采样分布与蒸馏梯度。
- **推理部署。** 腾讯补齐 Qwen3.8-27B 的 DSpine 草稿权重，公开的吞吐增益来自带缓存的工具轨迹回放。SGLang 扩展 Kimi K3.1 的解码上下文并行，vLLM 则补了 KV 传输失败后的清理和续接边界。
- **正确性与故障边界。** 昇腾 A3 的 DSA Host pinned 分配问题已在 0.30.0 正式镜像复发并转交上游排查。FlyDSL 0.5.0 的公开 wheel 纳入 JIT 磁盘缓存隔离和 gfx1250 低精度布局修复。
- **研究与评估。** AACL-IJCNLP 报告覆盖低秩 KV、片段级蒸馏以及批提示和系统日期对评估的影响。Decision Index v2 把置信度校准纳入评分，整份文档全对率与逐题平均之间的差距值得关注。

[模型与技术报告](#models) · [研究论文与评估](#papers) · [开源工程](#engineering) · [硬件与软件协同](#hardware) · [覆盖与续查](#coverage)

<a id="models"></a>
## 模型与技术报告

### PFN 开放面向日语金融文本的 PLaMo-3-8B-Fin-Instruct

**Preferred Networks，11 月 8 日 21:51:18（北京时间）建仓，21:57:22 的公开版本已列四片 BF16 权重。** 从 PLaMo-3 NICT 8B Base 出发，先在金融语料上持续预训练，再做 SFT 与 DPO，部分训练使用 PFN 的 MN-Core 2。基座上下文上限为 262,144 token，本轮金融持续预训练用到 32,768 token。模型先输出日语思考区间，随后输出答案或 JSON 参数工具调用。模板仅在每轮传入 `tools` 时才渲染历史工具调用。**建议先复测多轮工具结果回填与长文档分档表现**。模型卡披露测试范围为日语，未给可比较的金融分数，权重适用 PLaMo Community License。[官方工件与模型卡](https://huggingface.co/pfnet/plamo-3-8b-fin-instruct) · [固定权重版本](https://huggingface.co/pfnet/plamo-3-8b-fin-instruct/tree/3987ef72675694561040605411062769c265c0a6)。

### 补录｜腾讯开放 Qwen3.8-27B 的 DSpine 草稿权重

**腾讯，11 月 7 日 22:33:15（北京时间）上传权重，23:40:48 更新模型卡。** Apache 2.0 权重约 4.05 GB，草稿模型含 5 层、块大小 16、滑窗 8,192，配置从目标网络的第 1、16、31、46、61 层取特征。草稿块并行更新各位置，每层再把前一位置的预测特征经门控注入相邻位置，让块内依赖随深度传递。官方报告的 SWE-bench Pro `xhigh` 回放中，普通自回归、DFlash2、DSpine 分别为 248.17、711.60、791.30 输出 token/s，DSpine 相对 DFlash2 约快 11%。测试使用 SGLang 0.5.22、BF16、FlashAttention 3、并发 8，每个子集按顺序回放 20 条对话的已记录工具决策前缀并启用前缀缓存。DSpine 块长 16，DFlash2 为 8，DSpine 还启用了动态验证及 ReplaySSM。**建议按真实请求长度与并发复测**，同时记录接受长度和输出吞吐，分开核对草稿块长与验证路径。该结果衡量回放时生成速度，未测在线 Agent 的任务完成率。[提交历史](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16/commits/main) · [权重与许可](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16/tree/main) · [固定配置](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16/resolve/ea909c0f5fab8865f9e14600334f55f7ce47308f/config.json) · [模型卡与回放表](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16#results) · [论文 v2 §4.2](https://arxiv.org/html/2609.36173v2#S4.SS2)。

### 补录｜Fastino 用共享权重的循环层扩展 GLiDE 决策模型深度

**Fastino Labs，11 月 5 日发布五档 GLiDE 权重家族公告，来源只给日期。** 0.4B 至 27B 复用 System One 的分类、是非和打分接口，每题给出选项概率。9B 与 27B 让中间层段用同一组权重运行两遍，在内部表征上增加计算。27B 在单卡 RTX PRO 6000 上相对单遍的公开套件实测由 72.1 到 72.5，中位延迟由 83 到 112 ms。厂商预估的 Full 榜分从 72.1 到 73.3，Full 是估计值。2B 的同类消融在作者留出评测约增 1 分，延迟从 18 到 28 ms，因此发布版采用单遍。**建议复测循环深度的成本收益**，固定候选项和阈值，并按每请求问题数计成本。模型卡说明逐题需独立执行完整前向，权重附 Fastino Community License。[厂商公告与 Fig. 3](https://fastino.ai/blog/glide-decision-model-family) · [27B 模型卡](https://huggingface.co/fastino/GLiDE-27B) · [0.4B 提交历史](https://huggingface.co/fastino/GLiDE-0.4B/commits/main)。

### 补录｜Kakao 开放韩英多模态决策模型 Kanana-2-v-decide

**Kakao，11 月 6 日 09:42:31（北京时间）发布。** 3.67B 模型将输入状态与问题一同编码，把每个候选答案单独编码，再对归一化嵌入的相似度做 softmax，给出选项概率。发布版把候选嵌入按 token ID、设备与精度缓存。模型接收文本、JSON、对话及至多十张图片，长输入默认保留首尾。Kakao 在 Decision Index Vision 0.4.1 的公开子集上测得 61.12，StartLux 2B 为 59.90。纯文本公开子集为 61.34，低于 Joey-1 的 63.31。另有 15,909 道内部韩语文本题，Kakao 报 72.50。**建议用固定候选集复测**，分别测图片加入前后、超长输入截断、候选集变动造成的概率漂移。权重使用 KananaOpenLicense，厂商表格覆盖的比较模型有限。[发布时间](https://huggingface.co/kakaocorp/kanana-2-v-decide-3b-2611/commits/main) · [固定发布差分与代码](https://huggingface.co/kakaocorp/kanana-2-v-decide-3b-2611/commit/1489062def4145ae3a0a9261fbac4f4d88a8c6d1) · [模型卡与基准表](https://huggingface.co/kakaocorp/kanana-2-v-decide-3b-2611#decision-performance)。

<a id="posttraining-models"></a>
## 后训练与数据工件

### 补录｜Alpie Seed 9B 补齐后训练报告，身份指令随模板发布

**169Pi，11 月 6 日 15:10:09 上传报告，11 月 8 日 00:41:59 更新聊天模板（北京时间）。** 权重文件 10 月已上传，重叠窗新增的是报告和模板。报告 §3–4 披露，SFT 的 DoRA 适配器冻结为参考策略，GRPO 在同一份冻结的 NF4 双重量化基座上切换两个适配器求策略和参考 logprob，活动侧使用 DoRA 与 RSLoRA。8 张 H100 共跑五阶段 3,034 步，每题采 8 个回答，选第 4 阶段合并回 FP16 基座。模型卡的全精度测评以开启思考、温度 0 和最多 65,536 生成 token 运行，GPQA Diamond 为 84.32，MMMU-Pro 为 68.70。卡中 Qwen3.5-9B 的 81.7 和 70.1 取自原厂材料，评测配置可能不同。11 月 7 日 UTC 的模板差分给有无调用方 system 消息的两条路径都新增固定身份指令。**可先对比同一请求的渲染文本**，隔离模板效果，再用同一评测配方比较基座与后训练权重。报告公开奖励权重和训练设置，训练语料组成、各奖励函数实现与逐阶段分布未披露。[固定报告 §3–4](https://huggingface.co/169Pi/Alpie_seed/resolve/39f30ec5861453d5e40e67488f63f703e15be9a0/Alpie%20Seed%20Final%20Technical%20Report.pdf#page=4) · [模型卡训练与评测](https://huggingface.co/169Pi/Alpie_seed#training) · [报告上传提交](https://huggingface.co/169Pi/Alpie_seed/commit/39f30ec5861453d5e40e67488f63f703e15be9a0) · [固定模板差分](https://huggingface.co/169Pi/Alpie_seed/commit/021da6596d305151c4d3613386379b5f5c552b68)。

<a id="model-evals"></a>
## 评估与科研 Agent

### Decision Index v2 将置信度校准纳入决策模型比较

**Artificial Analysis，11 月 8 日官网发布，来源只给日期。** v2 把最高概率选项的对错与该概率的 Brier 误差共同计分，并把通用 LLM 改为结构化输出各选项概率。私有数据权重由 25% 升到 45%。新设的 AA-Paperclip 占 30%，在 1,500 份文件上同时评估每份文档的五项判断和十项抽取。15 项全对才通过，联合置信度以各项概率相乘，等同假设它们独立。Drex Max v1.1 与 Claude Haiku 5.5（Max）的逐题正确率分别是 97% 和 99%，文档全对率却是 63% 和 92%。新设的 AA-Symmetry 占 15%，测试重排选项、改标签或否定问句时的稳定性，稳定地选择弃答也可能得到高分。**建议在自有文档上并列报告文档全对率、校准和按人工复核阈值计算的成本**。私有样本限制外部复核，v2 更换了测试、权重与评分，跨版分数应分开记录。[官方发布](https://artificialanalysis.ai/articles/decision-index-v2) · [评分、权重与版本历史](https://artificialanalysis.ai/methodology/decision-index) · [榜单和准确率口径](https://artificialanalysis.ai/evaluations/artificial-analysis-decision-index)。

### 补录｜Overnight 公开四轮事实更新的逐题答案与遗忘曲线

**Fintan Labs，11 月 7 日 21:57:33（北京时间）发布。** 以 Apertus-8B-Instruct-2509 为底模，分四轮学习 295 条新闻事实。每轮只给原题问前几轮的全部事实。第四轮结束时，首日事实正确率为 Fintan 90.7%、LoRA 26.7%，两者对末日新事实分别为 91.7% 和 93.1%。LoRA 对所有线性层用 rank 16、学习率 2e-4，按作者设置连续训练。通用能力测试中，250 题 GSM8K 上原版 Apertus 为 69.6%、Fintan 服务为 69.2%、LoRA 为 4.4%。Fintan 闭源服务仅对判定涉及新事实的请求使用更新路径，其余由原版底模回答，因此该分数只衡量实际服务的路由结果。本轮用固定评分器重算 295 道题的 1,775 条可评分答案，全部吻合。**建议另用精确日期和数字匹配检查保持率**，原规则只要求 60% 的新词项命中。[固定数据卡](https://huggingface.co/datasets/fintanlabs/overnight-four-nights-2026-11/resolve/8d2126067bd9c987387a897e2666b02a61bd9bb5/README.md) · [发布时间](https://huggingface.co/datasets/fintanlabs/overnight-four-nights-2026-11/commits/main) · [固定评分器](https://huggingface.co/spaces/fintanlabs/overnight/blob/0a23c84a7ffd07a0a3307a004e5f2aa744f05cd1/scorer.py)。

<a id="papers"></a>
## 研究论文与评估

### 主窗｜AACL-IJCNLP 11 月 8 日口头报告

以下按上海会议日程记为报告排程事件，技术证据采用 ACL Anthology 论文集版。 [AACL-IJCNLP 口头报告日程](https://2026.aaclnet.org/program/oral_program/) · [ACL Anthology 论文集](https://aclanthology.org/events/aacl-2026/)

#### S4R 按提示词拟合低秩子空间，只重建本步需要的 KV

**11-08 10:30–12:00 上海｜AACL-IJCNLP Session 4.3｜2026.aacl-long.137。** 预填充从本次提示词采样拟合 K/V 的低秩基，解码保存系数，在潜空间用近期 query 选取历史位置，再重建命中位置和保留局部窗口。Qwen2.5-7B-Instruct-1M 在 128K、KV 压缩 2.5× 时，RULER 均分 93.61，完整 KV 为 93.97、xKV 为 93.47。另一个 Qwen3-4B 计时实验中，生成 512 token 为 61.54 秒，xKV 为 260.76 秒、完整 KV 为 54.89 秒。计时未做系统或内核优化，ShadowKV 也改用 GPU 常驻 V。复现宜单列拟合、选位和重建开销，并在相同 offload 及显存约束下测 TTFT、TPOT 与质量。论文链接的仓库在所查提交只见 README。 [S4R §3–4、表 2–3 与 Limitations](https://aclanthology.org/2026.aacl-long.137.pdf) · [作者仓库快照](https://github.com/Draconis98/S4R/tree/3f40ea0ba71708daa2efb46f336a2d09ecfb0bd8)

#### Masked diffusion 后训练用全块覆盖降低估计方差

**11-08 10:30–12:00 上海｜AACL-IJCNLP Session 4.2｜2026.aacl-long.263。** GRPO 的块级 ELBO 代理改用分层采样。样本预算 K 等于块数 C 时，每个块评估一次，以减少随机抽块重复或遗漏带来的方差。LLaDA-8B-Instruct 在 C=K=4、4,000 步时，GSM8K 从随机抽块的 76.20 升到 82.26，MATH-500 从 34.40 升到 38.00。固定 C=4 并加到 K=8 后，这两项差值分别收窄至 0.46 和 0.20 点。收益集中在低样本区，PPO/DPO 和更大 GRPO group 尚无实测。复现宜同时固定块评估次数、更新步数和 rollout 预算。 [See Every Block Once §4–5、表 1、10 与 Limitations](https://aclanthology.org/2026.aacl-long.263.pdf)

#### CHOP 在低可靠前缀切换到片段级蒸馏

**11-08 10:30–12:00 上海｜AACL-IJCNLP Session 4.3｜2026.aacl-long.167。** 学生自己生成轨迹。高可靠状态继续使用 token 监督，低可靠状态改对齐候选片段的分布，并按门控引入可用的特权信息。AIME 2024 的 Qwen3-8B 学生在 60K 预算下，三颗种子 avg@16 为 44.9，标准 OPD 为 28.4、Top-K LSM 为 42.8，30 题和配对种子的 bootstrap 给出 p=0.032。7K 时 CHOP 为 47.4，Top-K LSM 为 47.8。在 SWE-bench 中，论文把主要结果归为避免蒸馏退化。正文也分开比较学生更新量、教师预算和墙钟。复现宜先画长度扫描，分开报告原学生、可部署分支与 oracle，并把教师和验证成本计入。 作者实现固定至 `ba37ec5`，`chop/router.py::CellRouter.route` 交叉两道门，`chop/losses.py::chop_loss` 按分支选择 token 或片段损失。 [CHOP §4–5、表 2–4、附录 J.7](https://aclanthology.org/2026.aacl-long.167.pdf) · [固定代码](https://github.com/Galleons2029/CHOP/tree/ba37ec5998b96a3453b512623525f73927af22bf/chop)

#### 拼接多题的批提示会改变拒答行为

**11-08 10:30–12:00 上海｜AACL-IJCNLP Session 4.2｜2026.aacl-long.193。** 这里的批提示是在一个 prompt 内拼接多个问题。JailbreakBench 上，Llama-3.1-8B 的 12 题拼接条件中，单个有害问题混入良性问题时攻击成功率为 55%，同表长上下文攻击为 40%，测试的 ICL 攻击条件为 0%。论文也比较了面向批结构的偏好训练和外部审查。若应用用拼接降低请求开销，安全回归应分别测试单问、多问及各题在组内的位置，并逐问判定拒答与内容风险。 [Safety in Batches? §2.2、表 1、§5 与附录 B.2](https://aclanthology.org/2026.aacl-long.193.pdf)

#### 系统提示日期可改变同模型的评测得分

**11-08 10:30–12:00 上海｜AACL-IJCNLP Session 4.1｜2026.aacl-short.39。** 作者用 9 个开放权重模型扫过 2024 年的日期字段，在与日期无关的题目上仅改变系统提示里的日期。表 1 按模型与数据集计算最好日与最差日的准确率差，选择题平均为 2.52 个百分点，GPT-OSS-120B 在 GPQA 的差为 6.06 点。表 2 的 GSM8K 平均差为 7.75 点。这些值是日期扫描的极差，反映了该提示模板下的敏感性。跑版本回归时可冻结完整模板，对托管 API 记录评测日期，并用重复测量判断小幅排名变化。 [Dating the Model §3–5、表 1–2](https://aclanthology.org/2026.aacl-short.39.pdf)

#### 预训练末期的分数提升集中在哪些题

**11-08 14:00–15:30 上海｜AACL-IJCNLP Session 5.1｜2026.aacl-short.56。** 作者用按年份网页快照训练的模型计算题目负对数似然的变异系数，再按题长分层，在六个模型的预训练检查点上追踪 16,239 道选择题。终点高变异组在 34/36 个模型—基准组合中领先，机会校正准确率平均高 7.3 点（95% CI 5.5–9.2）。进一步控制似然、选项边际和准确率后，差距降至 1.4 点，22/36 组仍为正。区间反映设置间变异，每设置只有一次训练。复核训练曲线时可按难度与变异分层，分别报告知识和常识任务。 [What Improves During Pretraining? §3–4、表 2–3](https://aclanthology.org/2026.aacl-short.56.pdf)

#### ROCOB 为量化残差补偿按层分配秩

**11-08 10:30–12:00 上海｜AACL-IJCNLP Session 4.3｜2026.aacl-long.158。** ROCOB 用主激活子空间拟合校准误差，并给正交方向保留低秩残差补偿。它在相同新增参数预算内按各模块的谱收益分配秩。表 1 的 Qwen3-8B GPTQ W4A16 均分从 63.78 到 64.42，同预算 EoRA 为 63.88，BF16 为 65.01。换成 GPTAQ 后 ROCOB 为 64.16，略低于 EoRA 的 64.18。比较局限于 GPTQ/GPTAQ 及作者选取的校准偏移。部署复测可同时匹配真实字节、校准集与新增算子开销。论文所链仓库在所查提交只见 README。 [ROCOB §3–4、表 1 与 Limitations](https://aclanthology.org/2026.aacl-long.158.pdf) · [作者仓库快照](https://github.com/brinenick511/ROCOB/tree/7869d68fd5099f63a3193f389175bc0d877fd9dd)


### 补录｜按原提交日期

#### 同源 GRPO 专家蒸馏的算力优势取决于是否计入教师

**11-05 18:47 UTC｜arXiv:2611.08453v1。** 作者用累计训练 FLOPs 对照 reverse-KL OPD 和 GRPO。Qwen3-4B-Base 的数学实验以本学生的 GRPO checkpoint 作教师，蒸馏阶段达到该教师 33.9% 时，G=8 约需 1.62 EFLOP，G=1 约 0.2 EFLOP，对照 GRPO 达同分需 10.1 EFLOP。这两个倍率只计蒸馏阶段。若要交付同时覆盖数学和代码的单模型，作者分域训练专家后路由教师做蒸馏，将专家训练与蒸馏都计入后，到达联合 GRPO 同等准确率的总 FLOPs 减少 42%。外部教师的标准 OPD 在多项对照中没有类似优势。复核时应统一 rollout 计账、教师前向成本、目标分数和训练域混合，并将单域复刻和多域合并分别比较。 [OPD/GRPO 算力对照 v1 §3–6、附录 B.1](https://arxiv.org/html/2611.08453v1#S3)

#### HaluCheck 分开测幻觉发现和首个错误步骤定位

**11-05 16:15 UTC｜arXiv:2611.08231v1。** 中英评估由受控事实扰动和对已有代理轨迹的局部扰动组成，共 42,980 对。代理题把计划、工具调用与工具结果解释分开。附录 D 中，代理轨迹的样本级召回约 76%–86%，但明显、中等和细微扰动的首错定位率分别为 56.0%、47.2%、45.4%，定位分母仅含已检出的幻觉。合成扰动与自然错误可能有分布差异，论文也缺独立人类检测基线。部署门控可按业务中的自然失败轨迹补测，将漏检和定位失败分别计入。 [HaluCheck v1 §3–4、附录 C.1/D 与 Limitations](https://arxiv.org/html/2611.08231v1#S3)

#### ReKV 把低比特 KV 传输与后缀层重算重叠

**11-05 22:34 UTC｜arXiv:2611.08770v1。** P/D 交接时先发 BF16 边界隐藏状态，再传模型前缀层的量化 KV。decode 节点用边界状态重算后缀层，和剩余传输并行。跨节点 H100、8K、Qwen2.5-7B 的交接中位时间为 96.11 ms，纯 Q8 传输为 112.30 ms，纯 BF16 后缀重算为 149.14 ms，完整 BF16 KV 为 227.09 ms。相同 Q8 设置的 WikiText-2 PPL 增 0.077%，GSM8K 64 题两者均答对 46 题。带 512 token 的 Qwen2.5-3B 反而适合纯传输，且 Q8 尚未做分页缓存接入及端到端解码。验证时应按链路带宽、请求长度和并发扫描切分点，分别记录 handoff 与 TTFT。 [ReKV v1 §2–4、表 1–3、§6](https://arxiv.org/html/2611.08770v1#S2)

#### ResServe 在 FP4 线性层和解码注意力补偿残差

**11-05 21:09 UTC｜arXiv:2611.08713v1。** 线性层将激活的 FP4 基值与残差分别打包，折叠两份乘加结果。解码注意力将 query 的基值和残差分开乘 K，在 softmax 前合并分数。Qwen3-8B 使用 NVFP4 权重与 KV 时，LCB Hard350 pass@1 从纯 W4A4 的 24.29 到 35.14，BF16 权重与 KV 为 45.14。同一个方案在 Qwen3.8-27B 则由 79.43 降至 77.71。Qwen3-8B 的单 SM100、8,192 输入/256 输出、1 请求/秒 HTTP 均时延为 868.76 ms，W4A4 为 840.48 ms，所测 W4A8 为 1,119.39 ms。该试验限制在低负载且关闭前缀缓存，复现需扫并发、原始模型与 KV 格式，避免用算子精度替代端到端判分。固定实现 `bd33ded` 的 `csrc/nvfp4/mext.py::scaled_fp4_quant_mext_r1` 对应激活打包，`csrc/resattn/kernel_paged_decode.py::GroupedQueryAttentionDecodePagedFp4` 对应分页解码注意力。 [ResServe v1 §4–6、表 1–2、附录 C.5](https://arxiv.org/html/2611.08713v1#S4) · [固定代码](https://github.com/ShawRong/resserve/tree/bd33dedfc08019db77a4c11655cce5a8d4ce6c0d/csrc)

#### 取消超时请求之后，重试仍会消耗 prefill

**11-05 20:43 UTC｜arXiv:2611.08684v1。** 代理将取消传到引擎可回收已经离开的执行，但迟到的重试可能重复进入队列并消耗 prompt 计算。vLLM 过载脉冲后的恢复窗口中，Qwen2.5-7B 在同时启用重试和取消时仅交付 13.3%–15.2% 的后续请求，关重试、保留取消时为 89.0%–91.4%。重试预算和将相同请求附着到已有执行都在三种测试模型上恢复服务，合并还能救回过载期请求，但其中一部分超时。复测可分开记录逻辑请求、执行尝试、deadline 前交付和废弃 prefill。 [Cancellation Is Not Enough v1 §3–6、表 2](https://arxiv.org/html/2611.08684v1#S4)

#### ASP 按预测长度组批，再给并行 rollout 实例平衡负载

**11-06 01:47 UTC｜arXiv:2611.08864v1。** ASP 先预测响应长度，用动态规划形成长度接近的 minibatch，再调整批组顺序以避免长批集中出现，同时通过有界队列与权重刷新控制策略滞后。八台 Ascend 910B3 节点上，Qwen3-30B-A3B 的整系统迭代均时为 215.2 秒，verl 对照为 536.7 秒、MindSpeed-RL 为 927 秒。这个差值同时包含执行架构与实现差异。四节点 Maze 实验约滞后 3.3 个训练步，只测了 30 次 GRPO 迭代。复测宜在同一异步架构里只切换调度器，长跑分别追踪 padding、尾部完成、策略 lag 与质量。 [ASP v1 §3–5、表 3–4、附录 G](https://arxiv.org/html/2611.08864v1#S5)

#### 服务缓存分歧 v3 补明 llama.cpp 的内核路径

**11-05 23:24 UTC｜arXiv:2609.04748v3，相对 10-12 v2。** v2 已有重测与反量化对照，v3 在 §IV-D 补明 RTX 4090、llama.cpp `7e4c0a9` 的分支。量化权重的短调用走 MMVQ，超过八 token 的调用走 MMQ，两个路径都会把激活舍入到 8-bit 块。论文未隔离舍入与内核切换各自的影响。两版的反量化对照均显示同历史下路径分歧率 Q4_K_M 81.9%、反量化后 33.4%、F16 37.5%，后两者差异区间跨零。v3 同时将有序量化格式的趋势模型限定为统计编码，并补充归档 DOI。IEEE Early Access 状态来自作者的 v3 版本记录，独立期刊发表日期仍待核。复测宜固定引擎提交、权重格式和缓存路径，分别控制 activation 量化与 token 批几何。 [v2 §IV-D](https://arxiv.org/html/2609.04748v2) · [v3 §IV-D 与 Data Availability](https://arxiv.org/html/2609.04748v3) · [v3 修订历史](https://arxiv.org/abs/2609.04748v3)

<a id="engineering"></a>
## 开源工程

### 训练、蒸馏与训推一致性

#### Megatron-LM 让 MTP 损失可按全局主损失 token 归一化

**11-08 13:20（上海时间），默认分支 `main`。** 当不同 rank 或梯度累积微批的有效 MTP 目标比例不同时，局部归一化会改变各部分的相对权重。启用 `mtp_normalize_loss_by_main_tokens` 后，`_mtp_head_loss` 保留 MTP 损失和，梯度收尾统一除以全局主损失 token 数。开关默认关闭，要求 `calculate_per_token_loss=True`。可用相同样本的不同 packing 和 DP 划分核对梯度，并检查切换归一化后原 MTP loss scale 是否仍合适。 [Megatron-LM #8997 `7d2c1a12a0`](https://github.com/NVIDIA/Megatron-LM/commit/7d2c1a12a056ee89607f92ad93802bb3c88707d3)

#### DeepSpeed HybridEngine 的 CUDA graph rollout 支持非贪心采样

**11-08 16:29（上海时间），默认分支 `master`。** 只捕获 decode 前向，图外以 FP32 logits 依次执行温度、top-k、top-p 与 multinomial。图路径、连续批处理及其图路径共用采样器，温度非正仍用 argmax。PR 用固定 logits 检验采样分布，并记录 Transformers 4.57.x/5.x 的 SDPA mask 与缓存接口仍可能阻塞特定端到端 GPU 路径，相关修复仍在开放 PR #8602。值得先在实际依赖组合上对比采样分布和每步开销。 [DeepSpeed #8939 `02f7978481`](https://github.com/deepspeedai/DeepSpeed/commit/02f7978481cb3d3297f111c02b16d768b890b256) · [DeepSpeed #8602 开放 PR](https://github.com/deepspeedai/DeepSpeed/pull/8602)

#### DeepSpeed 增加词表并行广义 JSD

**11-08 23:33（上海时间），默认分支 `master`。** 蒸馏需要全词表分布。新 `vocab_parallel_jsd` 在 TP 上合并 softmax 统计，配对归约到混合分布的两项 KL，返回各 rank 一致的 FP32 逐 token 值。Triton backward 在寄存器重算概率，支持非均匀、空 shard 和分离 teacher 梯度。β=0/1 显式转入 forward/reverse KL。温度、mask 和 loss 归约由调用方负责，TRL 集成仍需独立验证。可用小词表的 dense 参考同时对照前向与梯度。 [DeepSpeed #8958 `da79d74479`](https://github.com/deepspeedai/DeepSpeed/commit/da79d744792cadda3af7674de43a88914895bf2f)

#### TorchTitan 优化 AllToAll token 重排与 MoE 路由排序

**11-08 11:47–11-08 11:48（上海时间），默认分支 `main`。** `AllToAllTokenDispatcher` 的置换反向和输出复位用逆索引 gather，省去累加写入的排序、清零和 scatter。稳定 expert argsort 按专家数用 uint8（≤256）、int16（≤32768）或 int64。PR 在单 GB300 用 DeepSeek-V3 EP2 单 rank、BF16 合成均匀路由测 16K token 重排前后向 9.253→4.083 ms。该数字只含重排 kernel，真实 EP 不均衡和通信重叠需另测。 [torchtitan #5162 `c0df32e306`](https://github.com/pytorch/torchtitan/commit/c0df32e306b95b27ba468f8d7dc52e4b23faa077) · [torchtitan #5436 `9771724696`](https://github.com/pytorch/torchtitan/commit/9771724696e96ec7bfa7390008d14f179d0f02c2)

#### TorchTitan 修复 Kimi K3 batch-invariant rollout 页尺寸

**11-09 07:05（上海时间），默认分支 `main`。** 原接口以 K 的 head size 估算 V，导致真实 KV 页 532480 字节与估算的 638976 字节不符，固定 16 token 块的 FA4 路径无法切分带 padding 的块。TorchTitan 的 DiffKV backend 在 `customize_spec` 先补实际 `head_size_v`，仍依赖每进程一种 V head size。PR 在 4 张 GB300、固定 vLLM nightly 上的三步 Kimi K3 RL 报告训推 logprob 最大差为 0，长跑仍待验证。 [torchtitan #5955 `b6d2c53412`](https://github.com/pytorch/torchtitan/commit/b6d2c534123e3a658ebaef66bbe277562199fb51)

#### NeMo RL 向量化 vLLM FlatLogprobs 的 score-centering 候选提取

**11-08 11:14（上海时间），默认分支 `main`。** 用向量化路径从平铺的采样 logprob 和索引还原逐 token 的 top-k ID 与 logprob 张量，保持旧提取契约并补单测。收益首先作用于 rollout 元数据整理，建议以候选数、响应长度和并发为维度剖析 CPU 时间，并核对输出与逐行路径的一致性。 [RL #5034 `8443b48388`](https://github.com/NVIDIA-NeMo/RL/commit/8443b483883ccedca2b826c92b56268e9390ce7a)

### 推理服务与 KV 正确性

#### vLLM 支持休眠时保留 offload KV

**11-08 10:38（上海时间），默认分支 `main`。** sleep/pause 或释放 KV 显存时可传 `reset_connector=False`，处理尚未提交的 store 后保留 CPU 或外部层，唤醒可命中。默认仍重置 connector。块哈希不含权重，热换版本必须保留默认重置或使用按权重版本隔离的 `cache_salt`，重点验证最后一步 store 和跨权重命中边界。 [vllm #57810 `5efb24d542`](https://github.com/vllm-project/vllm/commit/5efb24d5420b1f28737e4e8d1b62b354ddc18069)

#### vLLM 收紧 P2P 终止、流式续接和 Mooncake 查询的失败边界

**11-08 16:42–11-08 20:20（上海时间），默认分支 `main`。** P2P 将未发送的 fetch 在本地失败，避免向远端发送对应的 abort。流式请求抢占后刷新已重算的 offload key，保留未变化的块。Mooncake Store 前缀查询默认含排队最多等 1 秒，三次已到达 store 的连续失败触发 30 秒熔断，随后单探针恢复。超时按 miss 本地重算，已经执行的 master RPC、KV load/save 不受查询熔断约束。适合分别做断线、续接和 store 卡死注入。 [vllm #64298 `5dcc7fb261`](https://github.com/vllm-project/vllm/commit/5dcc7fb26109882912f4f463c6ee299da69d0a8a) · [vllm #65640 `b022bb7a48`](https://github.com/vllm-project/vllm/commit/b022bb7a4873a47cdcdf474edf3bd499c221f40d) · [vllm #65939 `71eb0c24fa`](https://github.com/vllm-project/vllm/commit/71eb0c24facfc5312c160026b4ba7baa3f047823)

#### vLLM 修复可扩展 KV 与同机 NIXL 的传输选择

**11-09 03:11（上海时间），默认分支 `main`。** VMM 分配在设备支持且进程能打开 IMEX channel 时请求 FABRIC handle，同机 UCX 由此可走 CUDA IPC/NVLink。设备内存 NIXL 在缺 IMEX 时默认禁用可扩展 KV，跨主机 RDMA 可显式开启。可用 `UCX_PROTO_INFO=y` 核对 `cuda_ipc` 与 TCP，避免只凭配置推断实际链路。 [vllm #65137 `80e2e486f2`](https://github.com/vllm-project/vllm/commit/80e2e486f2b895c06ea962e6146b8f97376dcaff)

#### vLLM 保留 batch-invariant FlashAttention 的滑窗约束

**11-08 09:10（上海时间），默认分支 `main`。** 滑窗配置不再随批内最长 KV 或 FULL graph 捕获上界变化，避免不同同批样本让同一序列走不同窗口路径。回归将 Gemma 滑窗请求分别单跑和混批，逐 token 检查生成和 logprob 逐位一致。上线 batch invariance 时应覆盖滑窗内外长度以及 graph/eager 的组合。 [vllm #65627 `d1879210a4`](https://github.com/vllm-project/vllm/commit/d1879210a4b10773d22a2a659152beca0d35d007)

#### vLLM 多模态 encoder 图按整批 replay，并保护共享输入输出

**11-08 21:01–11-08 22:39（上海时间），默认分支 `main`。** 按整批 token 总量选择最小可容纳预算，每条路径 replay 一次，超预算或图片数上限则整批走 eager。跨预算共享输入缓冲要求 replay 重填，漏填提前报错。Step3-VL 只包含全局特征的样本会 clone 输出，防止后续 replay 覆写。适合使用尺寸差异大的图像批、连续交错预算和引用持有期复验。 [vllm #65227 `325bc507bc`](https://github.com/vllm-project/vllm/commit/325bc507bcdb8ed92eced871e506025d40f2de6a) · [vllm #65351 `d2c5f9ad5f`](https://github.com/vllm-project/vllm/commit/d2c5f9ad5f3ee12c22c61a744d81fc4fb41bcf96)

#### vLLM 允许 DeepSeek V4 MegaMoE 同时用序列并行与流水线并行

**11-09 03:21（上海时间），默认分支 `main`。** 启用 EP、TP>1 且选 MegaMoE 后端时，PP stage 在传出 `IntermediateTensors` 前 all-gather token 并裁掉 padding。只因 DP 启用的序列并行仍限 PP=1，现有补丁测试覆盖选择条件。可进一步测 stage 边界正确性、padding 和 gather 对流水线气泡的影响。 [vllm #65766 `15166efabb`](https://github.com/vllm-project/vllm/commit/15166efabb4d7f27eaf807ee8e1c7454fd188886)

#### SGLang 扩展 Kimi K3.1 解码上下文并行部署

**11-08 16:15–11-09 06:17（上海时间），默认分支 `main`。** trtllm DSA 处理局部 attention 和 base-2 LSE。P/D 分离仅允许 Mooncake，显式把 DSA index-K 的 key/scale 按 token owner 发送，拒绝无可分片元数据的 kpool 路径。HiCache 分别处理 DFlash 的 DCP 分片 draft KV 与 DSpark 复制 draft KV。适合以不同 P/D TP 布局、长前缀 cache hit 及投机回退作联合回归。 [sglang #49106 `d55caa2675`](https://github.com/sgl-project/sglang/commit/d55caa267558962199bcafef50464df83fe13ad3) · [sglang #48741 `94d1e0230c`](https://github.com/sgl-project/sglang/commit/94d1e0230c47f9bc7757de19851268bcbe8711f0) · [sglang #49105 `033c03a9cd`](https://github.com/sgl-project/sglang/commit/033c03a9cd2816fbe52c04e9a9d1096da8db31f5)

#### SGLang 收紧 KV 页与请求释放边界

**11-08 19:28–11-09 07:59（上海时间），默认分支 `main`。** 统一内存池将 forward 在途时释放的页留在 pending，结果处理等待覆盖 forward 的事件后才交还 free list。拒绝请求及 offload 释放接入 request retirer。Mooncake KV 注册除已登记的重叠状态外，失败会中止启动。页回收补丁仍标出远端迟到写和 PP stable rows 的待查路径。建议分别用抢占、迟到写、终态输出计数和启动故障复验。 [sglang #48852 `de3636131d`](https://github.com/sgl-project/sglang/commit/de3636131d1727b1397be98a42d58d5db95a0936) · [sglang #49151 `620413be1a`](https://github.com/sgl-project/sglang/commit/620413be1a8e78c9d22563c7ad86a4a77f401647) · [sglang #49156 `842b473831`](https://github.com/sgl-project/sglang/commit/842b4738315fed2246b04a6fbe92fd6bb311ac25)

#### SGLang 修正 DeepSeek V4 未量化 checkpoint 的专家路由

**11-08 15:13（上海时间），默认分支 `main`。** 在识别出 BF16/F16/F32 未量化 checkpoint 时，避免将专家导入 MXFP4 路径。该改动影响装载和后端选择，可用官方 checkpoint dtype、专家权重形状和固定 prompt 的首 token logits 检查实际路径。 [sglang #44813 `a6989e70cb`](https://github.com/sgl-project/sglang/commit/a6989e70cb2c16c13cbd0b1e00b5529f792629d6)

### 算子、编译与运行时

#### FlashAttention CuTe 用持久调度减少填充 varlen 的空 CTA

**11-09 05:04（上海时间），默认分支 `main`。** SM100 在 request slots 多于真实 M tiles、超过一块且满足无 split-KV 等条件时走动态持久调度。load warp 获取 pipeline 槽后才领取下一 tile。32 位 semaphore 分别计领取和完成，最后 CTA 清零。每 semaphore 同时只容许一次 launch，且有 255 CTA 与小于 2²⁴ claims 的边界。PR 在 GB300 的 12 个重叠配置报告对 CLC 1.04–1.43×，范围为 64 slots 的 hd64/128 CUDA graph replay。 [flash-attention #3086 `f4cc3b0b32`](https://github.com/Dao-AILab/flash-attention/commit/f4cc3b0b325d278ec9e7bf7f9f054aad5f150ff6) · [flash-attention #3087 `8e3cb05219`](https://github.com/Dao-AILab/flash-attention/commit/8e3cb0521977706409081f82d938cde8baebd427)

#### FlashAttention SM100 扩展 hd256 GQA backward 的同 CTA 复用

**11-09 08:22（上海时间），默认分支 `main`。** 新路径把两个 Q heads 合装一 tile，在串行 2CTA 调度中循环同一 KV head 的剩余子组，K/V/Kt 保持驻留，dK/dV 在 TMEM 累积再直接存回。dQ 仍按各 scratch head 归约。score_mod、mask_mod 和块稀疏不走此折叠。适合比较奇数 Q/KV 比例、确定性与长 causal 序列的数值和资源占用。 [flash-attention #3059 `8fde8bf777`](https://github.com/Dao-AILab/flash-attention/commit/8fde8bf777db37222ca6e684df7744c47926d577)

#### Triton 修正 NVIDIA 发布者分析并简化单段 warp specialization

**11-09 00:05–11-09 03:59（上海时间），默认分支 `main`。** Membar 为 TMEM copy、MMAv5 completion、TCGen5 commit 等标记各 partition 的 warp-zero elected leader，并保留必要同步分析。无 worker 的特化折回默认函数。仅有一个且支配全部 return、不在 CFG 环内的特化用 warp-ID 直接分支，去掉共享命令读和退出 barrier。适合以 MLIR 回归及实际 Gluon kernel 核对同步与寄存器预算。 [triton #12661 `fc2c4016f8`](https://github.com/triton-lang/triton/commit/fc2c4016f891c650fac890fe1855e8c15f09b0c8) · [triton #12663 `248fe023da`](https://github.com/triton-lang/triton/commit/248fe023da7a2a6fc849adf66a74df90b5da65cb) · [triton #12664 `52e8a001c6`](https://github.com/triton-lang/triton/commit/52e8a001c6bad9e3017102e1202e047e9cf15875)

#### PyTorch XCCL2 主干加入异步工作与 communicator bootstrap

**11-08 09:51（上海时间），默认分支 `main`。** 新增 `WorkXCCL` 的 XPU 事件、超时、队列，和通过 c10d Store 交换 oneCCL unique ID 的 bootstrap。作者明确此系列第 2 部分的文件尚无调用，构建接线在第 7 部分。当前 oneCCL 无 communicator async-error 查询，故障预期由 work timeout 暴露。适合跟踪完整接线及超时回归。 [pytorch #202810 `ef091f1e3e`](https://github.com/pytorch/pytorch/commit/ef091f1e3e9643b73a5d617ffd31e472f6bf7a4a)

#### PyTorch 主干撤回 cuDNN 9.27.0 升级

**11-09 05:30（上海时间），默认分支 `main`。** 此次 commit 撤回 #202631 的依赖改动。复现从此前 nightly 得到的 cuDNN 路径时，应固定 torch commit 与实际加载的 cuDNN 版本，尤其需要避免把不同依赖组合归到同一性能基线。 [pytorch #203501 `4dc1d5db9b`](https://github.com/pytorch/pytorch/commit/4dc1d5db9bc2d81f28bb195f34bd99ff2183a058)

### 运行时与可观测性

#### TorchTitan 让 FSDP all-gather 可选使用计算流

**11-09 06:32（上海时间），默认分支 `main`。** 新增 `fsdp_unshard_in_current_stream` 并在启用时配置 decoder 显式前后向预取，降低多 CUDA 流分配造成的显存碎片风险。编码器及含 MTP 层的 DeepSeek V4 等缺失对应预取模型的路径明确拦截。PR 报告一项试验的 reserved 峰值 42.08→37.62 GB、吞吐 3771→3745 token/s，配置披露有限，适合在出现 allocator retry 的训练上分别采集 reserved、allocated 与重叠时间。 [torchtitan #5627 `7d88c59dbe`](https://github.com/pytorch/torchtitan/commit/7d88c59dbedc345040906da1567c39de7533e994)

#### TorchTitan 修正 DP FLOPs 归约和未知设备的 MFU

**11-09 01:50–11-09 01:54（上海时间），默认分支 `main`。** DP 平均使用参与求和的元素数。设备峰值 FLOPs 查不到时仍保留 token/s 和 TFLOPs，省去可能因假定 A100 而出现的错误 MFU。对比历史曲线时应固定 GPU 识别结果和 TP/DP 统计口径。 [torchtitan #5879 `a24a35c164`](https://github.com/pytorch/torchtitan/commit/a24a35c164882c88f0c254795619cf2ab84f0e82) · [torchtitan #5880 `c561eee640`](https://github.com/pytorch/torchtitan/commit/c561eee64056bb077b41022d8a65661b66f98a08)

#### LMCache 给同步与异步 gRPC 增加 API key 校验

**11-08 09:17（上海时间），默认分支 `dev`。** 同步与 `grpc.aio` 通过 interceptor 验证 authorization metadata，错误和缺失凭证返回 UNAUTHENTICATED。该入口在 LMCache `dev`，默认 insecure channel 无加密。适合验证拒绝路径和密钥轮替，同时明确实际 TLS 终止点。 [LMCache #5770 `a9db748507`](https://github.com/LMCache/LMCache/commit/a9db748507898234ffe8a2185d512e0a3719f554)

#### Dynamo 修复 reasoning 边界拆分的 tool-call 起始标签

**11-08 12:00（上海时间），默认分支 `main`。** 流式解析器在 reasoning 结束后保留跨 chunk 的部分起始标签。回归覆盖 Hermes 工具调用与最终普通文本，验证 finish reason、函数名与 JSON 参数。适合以真实 token chunk 边界对照聚合后 API payload。 [dynamo #18302 `27f6052cd5`](https://github.com/ai-dynamo/dynamo/commit/27f6052cd57c35d0da1eeb03945b7a3e2b89fcb3)

#### TensorRT-LLM 为 MiniMax-H3 接入 TRTLLM Ulysses attention

**11-09 01:11（上海时间），默认分支 `main`。** TRTLLM 后端在 SM100/SM103 的 Ulysses 处理显式 valid prefix，只把有效行交给 attention，再零填 padding。测试涵盖已有尾部 padding、全部无效及内部 padding 的拒绝。可比较不整除 SP 的序列与原路径，分别核对输出和通信开销。 [TensorRT-LLM #21222 `2d5b1fb68f`](https://github.com/NVIDIA/TensorRT-LLM/commit/2d5b1fb68fee16b49b43f501d9165af23fb83c33)

#### TensorRT-LLM 主干移除 FlashInfer 的量化 TRTLLM-Gen MoE 路径

**11-08 16:45（上海时间），默认分支 `main`。** 旧 `TRTLLM_GEN_FUSED_MOE_USE_FLASHINFER` 开关失效并告警，原可选的 NVFP4、FP8 block-scale 与 MXFP4 量化路径统一走原生 TRT-LLM。未量化 BF16 继续用 FlashInfer kernel，two-sided NVLink all-to-all 换成原生 `NVLinkTwoSided`。适合用部署的量化组合和 EP 拓扑核对实际 dispatch、精度及吞吐。 [TensorRT-LLM #21013 `c1544b47ec`](https://github.com/NVIDIA/TensorRT-LLM/commit/c1544b47ecdb370c26d41e72ba0da368a0a35f7f)

### FlashInfer 主干的确定性与正确性

#### FlashInfer 修复 Cake GDN 初始状态 prefill 死锁与训练跨流读写

**11-09 01:52–11-09 03:57（上海时间），默认分支 `main`。** 前两处修复 sm_107a 与 sm_100a/sm_103a 初始状态变体在 tile 边界的 barrier 竞争。R200/CUDA 13.4 注入延迟后，旧代码在 42 个用例格中挂 29 个，修复后 810 格、324000 次 launch 无挂。确定性训练另对缓存设备表建立写入事件及使用方引用，图捕获保有独立 workspace，避免未写读取或被回收重用。若其他线程在捕获期间构建表，仍须在捕获读取前同步其流。建议分开做 GPU kernel 边界注入与多流/多 graph 回归。 [flashinfer #9184 `9db6415713`](https://github.com/flashinfer-ai/flashinfer/commit/9db6415713d258650462487bdb1e9d719c881373) · [flashinfer #9187 `37c528ae76`](https://github.com/flashinfer-ai/flashinfer/commit/37c528ae766dd3ff7c68930c9fd1dc89daae4f1d) · [flashinfer #9186 `51062dcc81`](https://github.com/flashinfer-ai/flashinfer/commit/51062dcc811d9bf9115a0f28505f294451f3fdc6)

#### FlashInfer 阻止 paged attention 尾页 NaN 污染有效输出

**11-09 08:45（上海时间），默认分支 `main`。** 即使越过 `kv_len` 的概率为零，乘上陈旧页中的 NaN 仍会污染整块。SM100/SM103 的 BF16/FP8 路径在 P·V 前清零最后一块的无效 V 行，含 AOT 及 JIT。PR 报有限输入逐位一致，GB300 微基准中位开销 BF16 0.12%、FP8 0.27%。可用页池复用、尾页残留 NaN、split-KV 和带 LSE 的路径复验。 [flashinfer #9203 `432a0d35d6`](https://github.com/flashinfer-ai/flashinfer/commit/432a0d35d6ec1e34dd59344d1a57c7113ff961c6)

#### FlashInfer 新增确定性 DSA sparse MLA 与 packed HD256 GQA 训练内核

**11-08 13:29–11-09 04:04（上海时间），默认分支 `main`。** DSA 面向 BF16、64 Q heads、512 latent + 64 RoPE，提供 RoPE/NoPE 和 packed varlen。作者报告相同设备、版本、输入逐位复现，但三类 GPU 的确定性 backward 在 13 行配置上均慢于同版非确定性实现。GQA 入口限 BF16、24Q/4KV、head dim 256、bottom-right causal，支持 dK/dV 的 FP32 行映射写回和 non-reentrant checkpoint。两者适合单独比较复现性、梯度写回及端到端占比。 [flashinfer #9161 `8e4f178734`](https://github.com/flashinfer-ai/flashinfer/commit/8e4f178734f7908532724ea7dfced039829c38c5) · [flashinfer #9121 `15482962a1`](https://github.com/flashinfer-ai/flashinfer/commit/15482962a1f607944dc412f56523afadbbc72507)

#### FlashInfer 减少确定性 LM head 重算和 MoE 梯度写回

**11-08 10:05–11-08 20:21（上海时间），默认分支 `main`。** 默认限显存 logprob 路径在 non-reentrant checkpoint 重算时复用前向行结果。PR 以 BF16、hidden 6144、词表 154880、token chunk 4096，在 B200/GB300/R200 报前后向 1.20–1.30×。MoE grouped GEMM 将 FP32 wgrad 直接累加到 `main_grad`，目标与先算 dW 再相加逐位一致。1.254× 峰值限 GB300 的 E32、131072 条均衡 wgrad 测例。 [flashinfer #9124 `c8a4d8a2ed`](https://github.com/flashinfer-ai/flashinfer/commit/c8a4d8a2ed8dae9166da0e4161411cac09b2c6de) · [flashinfer #9132 `5c29500371`](https://github.com/flashinfer-ai/flashinfer/commit/5c29500371e87cf53610b09970447a01cd65c1a1)

### 回溯窗口主干补录

#### PyTorch Inductor 修复跨 graph partition 的 cat 缓冲未初始化

**11-06 18:43（上海时间），默认分支 `main` · 补录。** 当输入由不同 graph partition 生成，首次写入的 partition 分配并返回 concat buffer，后续写入或读取它的 partition 将它作为输入。规则限 `ConcatKernel` 写别名，TMA 等只读别名继续使用既有归属。回归覆盖三段独立产生并在原位写入的切片，适合针对 graph partition 开关验证 NaN 与 eager 一致性。 [pytorch #200865 `1b585fa49b`](https://github.com/pytorch/pytorch/commit/1b585fa49b588712e8db16a81ee1723769c5118f)

#### TRL 将 Qwen3 与 GLM-4-MoE 的预填 thinking 移出 assistant-only loss

**11-06 22:50–11-06 22:54（上海时间），默认分支 `main` · 补录。** 训练模板把 generation prompt 已给出的 thinking 前缀放到 generation 标记外，GLM-4-MoE 在 `enable_thinking=False` 时处理空 think block。Qwen 模板还处理空 reasoning 时换行合并导致的 token 边界。适合检查 `return_assistant_tokens_mask` 与实际 tokenizer，尤其是跨模板训练时。该项对应 TRL 的实现提交。 [trl #8072 `4917fe6f86`](https://github.com/huggingface/trl/commit/4917fe6f8639c0987042fc8599930006790931bf) · [trl #8122 `668ae15bbd`](https://github.com/huggingface/trl/commit/668ae15bbdf0326fc0fce2265e1eb8c3439e522d)

#### vLLM 增加 RL 权重同步的 load plan

**11-06 14:01（上海时间），默认分支 `main` · 补录。** 用无数据 `TraceTensor` 走一次模型的 `load_weights`，把可跟踪的 copy 归约为本 rank 所需的目标范围，后续可按 plan 传输和写入。不支持的变换按全张量回到原 loader，测试包含 overlap 与 manifest 不匹配。适合在 TP/EP 下分别比较首次规划成本、后续传输字节与冷加载张量一致性。 [vllm #65235 `d254f6a9bd`](https://github.com/vllm-project/vllm/commit/d254f6a9bdbb1afe3e6ca5b858ef4b06e547405f)

#### SGLang 修复 Mamba 状态复制峰值与 PP 下滑窗层偏移

**11-07 06:42–11-07 12:49（上海时间），默认分支 `main` · 补录。** MambaPool 的大 temporal state copy 分块执行，降低一次性临时开销。滑窗 KV pool 在 PP 下以本 stage 第一层 SWA 层为起点定位。两条各适合用长序列缓存搬移的显存峰值和 stage 边界层的 KV 索引做回归。 [sglang #47984 `784cffe3d9`](https://github.com/sgl-project/sglang/commit/784cffe3d90bb625041f9c0b4f06e66b0a122e42) · [sglang #46200 `c2580a7e1e`](https://github.com/sgl-project/sglang/commit/c2580a7e1e5b845fd91b8f97f1ec7e406e2b1a15)

### 项目与正式版补录

#### Google 开源 XLA Serving Engine

**11-05（Google 博客标注日期），开源公告补录。** XLA SavedModel 把 JAX 计算保存为 StableHLO，权重放 Orbax checkpoint，C++ runtime 直接通过 PJRT 执行，支持单机和多机并提供 gRPC 容器。官方的 2–5× 吞吐比较针对短请求与优化后的 Python server。Google 将它定位为单次推理运行时，长上下文自回归 KV 管理仍交给 vLLM/SGLang，TorchTPU export 和 AOT 编译是计划。仓库在公告前已有发行记录，这里按公开推广事件补录。可优先评估 embedding/评分等短请求。 [Google 开源公告](https://opensource.googleblog.com/2026/11/supercharging-xla-inference-xla-serving-engine.html?m=1) · [OpenXLA 项目](https://github.com/openxla/xla-serving)

#### NVIDIA Molt v0.2.2 保留 rollout 的 reasoning 历史

**11-06 13:39（上海时间），正式版补录，tag 指向 `293f9ee729`。** 旧 chat-agent 用模板默认值重建历史，Nemotron 3 模板默认截去较早的 thinking，可能使后续 token 前缀与采样时不同。版本允许透传模板 kwargs 并保留生成时历史，还把工具解析和数学评分移出事件循环，检查 rollout logprob 的负向偏差。PyPI 同日发布 wheel，项目仍建议使用镜像或源码安装以取得固定的 AutoModel。适合对照多轮生成后实际 token 轨迹、模板重渲染及训练动作 mask。 [Molt v0.2.2 发行记录](https://github.com/NVIDIA-NeMo/labs-molt/releases/tag/v0.2.2) · [Molt #251](https://github.com/NVIDIA-NeMo/labs-molt/commit/293f9ee729db7e0262d60aa6c13efdee2879791a) · [PyPI 0.2.2 工件](https://pypi.org/project/molt-rl/0.2.2/#files)

<a id="hardware"></a>
## 硬件与软件协同

### 昇腾 A3 正式镜像复发 Host pinned 分配失败，上游已开单排查

**vLLM-Ascend / TorchNPU。** 复发发生于北京时间 11 月 8 日 00:39，早于本轮主窗。11 月 8 日 12:22 起新增资源归属、分配路径和上游工单证据。两台 A3 使用 DeepSeek-V4.1-Flash-1017 W8A8、vLLM-Ascend 0.30.0、CANN 9.2.1、TorchNPU 2.10.0.post7，Prefill DP4×TP4、Decode DP8×TP2，开启 DSpark、CPU Engram offload 和两路 KV connector。Prefill worker 构建 DSA CP 元数据时，`qsl.pin_memory()` 的 64 B 请求进入 Host allocator，底层日志报 2 MiB 物理分配失败。外部 KV 池由独立服务持有，运行时 pinned 分配由模型进程发起。现场将每机外部池从 384 GiB 调至 256 GiB 后，在原镜像上完成了报告所列 6,915 次正常 SSE 请求，仍缺最小确定性复现和故障时的完整内存轨迹。驱动维护者在另一个栈上说明了 WithCfg UVA 路径的物理连续页约束，当前工单仍在排查。**建议深入阅读**，分别核对外部池容量、运行时 Host pinned 分配和后续 DSpark shape 失败，按现场版本采集证据。[正式镜像复发](https://github.com/vllm-project/vllm-ascend/issues/21094#issuecomment-6711241271) · [本窗分配路径](https://github.com/vllm-project/vllm-ascend/issues/21094#issuecomment-6717183598) · [后续测试](https://github.com/vllm-project/vllm-ascend/issues/21094#issuecomment-6717372155) · [TorchNPU 排查](https://gitcode.com/Ascend/pytorch/issues/5953) · [驱动讨论](https://gitcode.com/cann/driver/issues/225)。

### FlyDSL 0.5.0 随 PyPI 工件交付磁盘缓存隔离和低精度布局修复

**ROCm，Git tag 北京时间 11 月 6 日 17:04，PyPI 首个 wheel 11 月 8 日 11:49，基线 `226cde8d`。** 旧 JIT 磁盘键可能在切换 ROCm toolkit 后继续复用由旧设备 bitcode 或外部 LLVM 库生成的二进制。新版纳入设备 bitcode 和链接依赖的指纹，另有按追踪后 MLIR 区分编译产物的修复。gfx1250 的 WMMAScale FP4/FP6 A、B 操作数布局也已修正。**考虑复现**，用两个 ROCm 根目录做冷、热 cache A/B，并在 gfx1250 上将 FP4/FP6 结果与参考实现比对。依据是版本差分与项目测试，缺少同负载的端到端性能数据。[标签](https://github.com/ROCm/FlyDSL/releases/tag/v0.5.0) · [PyPI 工件](https://pypi.org/project/flydsl/0.5.0/#files) · [工具链指纹](https://github.com/ROCm/FlyDSL/pull/1381) · [追踪后 MLIR](https://github.com/ROCm/FlyDSL/pull/1338) · [WMMAScale](https://github.com/ROCm/FlyDSL/pull/1352)。

### CANN Runtime 主干为机密计算 DMA 映射和二维拷贝加上边界

**CANN Runtime，11 月 9 日 09:23:55，master `012e24c4`。** 在机密计算虚机的 PCIe 设备上，补丁加入 `halHostDmaMap` 和 `halHostUndmaMap`，并在二维拷贝校验复制范围与实际可用内存。变更横跨 HAL 接口、Runtime 及驱动适配，MR 给出的自测范围是机密虚机 H2D/D2H，缺公开性能与发行包对应关系。**建议关注**机密推理容器的普通 Host 分配和 SVM 路径，并对带 pitch 的拷贝单独做合法与越界测试。[合入提交](https://gitcode.com/cann/runtime/commit/012e24c45a1828a9c170287a6517a34cff3ebcc7) · [文件差分](https://api.gitcode.com/api/v5/repos/cann/runtime/pulls/6124/files)。

### 补录｜HCCL 修复保序 AllReduce 非整除尾块越界

**CANN HCCL，11 月 6 日 16:52:26，master `e556d8b3`。** 保序 AllReduce 把非整除余数放到末 rank，末 rank 暂存多份尾块时可能写穿 CCL buffer 后半区。MR 用 200 MiB buffer、8 ranks、FP32、26,214,399 个元素举例，暂存需要 104,857,792 B，超出 192 B。修复根据实际半区容量限制本轮尾块余数，把放不下的元素留到下一轮，保留原有归约顺序。**考虑复现**容量上限附近、非整除和最后一轮三类边界，项目 MR 缺公开实机覆盖结果。[合入提交](https://gitcode.com/cann/hccl/commit/e556d8b37d060656a92cc244b4ecef0cc346c285) · [文件差分](https://api.gitcode.com/api/v5/repos/cann/hccl/pulls/3914/files)。

### 补录｜CANN 修复初始化死锁和 Qwen3.5 量化配置误判

- **Runtime，11 月 6 日 15:22:34，master `c086f247`。** 初始化设备的 dcache lock 任务失败时，主线程在等 stream 完成，回收线程打印错误又等主 Context 引用初始化，两者互等。错误路径改为非轮询查找，Context 引用仍在更新时返回空值，回收线程可继续报告错误。**建议深入阅读**其调用边界，部署侧可在有故障注入的初始化测试中验证错误是否及时返回。证据是补丁和新增单测，MR 未披露设备故障注入结果。[提交](https://gitcode.com/cann/runtime/commit/c086f247840940b6868212d324fdedba55ee3e23) · [文件差分](https://api.gitcode.com/api/v5/repos/cann/runtime/pulls/6136/files)。
- **cann-recipes-infer，11 月 6 日 11:34:21，master `a07255c6`。** Qwen3.5 的文本子配置可能丢根级量化声明，旧代码在模型侧再次读盘。新代码统一解析文件与 YAML，规范化排除名单和模块前缀，按最终生效方案做硬件校验。贡献者在 950DT、完整 40 层 eager、无真实 checkpoint 下报告 BF16 与在线 MXFP8 的 prefill 加 8 步 decode 前后 logits 逐元素一致，FP8 测试依赖仓外诊断补丁。**建议优先复测**含排除名单的真实 checkpoint，特别是文件方案与 YAML 声明冲突时的融合层选择。[提交](https://gitcode.com/cann/cann-recipes-infer/commit/a07255c68401f789ea4af4f162d21f2b27de32ba) · [文件差分](https://api.gitcode.com/api/v5/repos/cann/cann-recipes-infer/pulls/1229/files)。

### 补录｜Gaudi 插件纠正异步解码输入和两处数值路径

**vLLM-Gaudi，11 月 5–6 日合入主干。** #1919 在沿用上一步请求行的 scatter 之前先写本步 Host token ID，修复新进 decode 请求读取旧 HPU 缓冲的风险。#1930 在 GemmaRMSNorm 中用 FP32 形成权重偏移和残差和，归一化后才转回原 dtype。#1948 在构造 RoPE 表前预热一次 CPU FP32 `cos`，绕开贡献者观察到的多线程首次调用差异。**考虑复现**异步 batch 进出时的 token 正确性，以及独立进程冷启动时 RoPE 表和 Gemma 路由的数值一致性。当前证据是已合入补丁及相关单测，未确认稳定包回移。[异步解码 #1919](https://github.com/vllm-project/vllm-gaudi/pull/1919) · [GemmaRMSNorm #1930](https://github.com/vllm-project/vllm-gaudi/pull/1930) · [CPU RoPE #1948](https://github.com/vllm-project/vllm-gaudi/pull/1948)。

### 补录｜tpu-inference 0.33.0 固化 JAX 路线和 Torchax 兼容修复

**vLLM TPU 生态，11 月 6 日 07:16 tag，次日 02:48 起上传 PyPI，基线 `a14edd7a`。** 标签 README 明确该代码线以 vLLM JAX 继续服务 TPU，PyTorch 原生 TPU 由独立项目交付。相对 0.32.0，tag 中含 Gemma-4 Torchax encoder JIT 的 capture axes 兼容修复，模板按 budget 和轴分组，避免碰到新轴时直接退出 JIT。MoE top-k 路由权重可用默认关闭的开关保留 FP32。**建议深入阅读**各后端的支持矩阵，并分别复测多模态 encoder 桶和路由精度。补丁和测试说明支持这些边界，尚无同负载的跨后端性能对照。[标签](https://github.com/vllm-project/tpu-inference/releases/tag/v0.33.0) · [固定 README](https://github.com/vllm-project/tpu-inference/blob/a14edd7a4af7d7597638da5bf36b4e440a203bba/README.md) · [PyPI 工件](https://pypi.org/project/tpu-inference/0.33.0/#files) · [Gemma-4 #3948](https://github.com/vllm-project/tpu-inference/pull/3948) · [MoE #3934](https://github.com/vllm-project/tpu-inference/pull/3934)。

### 补录｜AITER 0.1.28.post1 重切发行线并提供预编译矩阵

**ROCm AITER，11 月 5 日 14:05:56 正式发布，`ad0554a9`。** 维护者从 main 截至 #7429 重切 0.1.28 发行线，比较页显示新的版本提交带来广泛源差分。预编译 wheel 面向 gfx942/gfx950，提供 ROCm 7.2、7.14、10.0、10.1 与 CPython 3.10/3.12 的矩阵。**考虑复现**按明确 wheel 与前一版对齐镜像、GPU 和精度，先跑所用的 MoE、attention 和量化 kernel 断言。发布页未提供端到端对比，也未把 gfx1250 纳入该预编译矩阵。[发布与 wheel](https://github.com/ROCm/aiter/releases/tag/v0.1.28.post1) · [重切 PR](https://github.com/ROCm/aiter/pull/7490) · [版本差分](https://github.com/ROCm/aiter/compare/v0.1.28...v0.1.28.post1)。

<a id="coverage"></a>
## 覆盖与续查

- **模型、厂商报告与工件。** 定向检查 Hugging Face 156 个组织，模型及数据集按创建和更新共 624 个目录请求，475 项取得、149 项失败，183 个重叠窗候选。全局创建目录扫描到回溯起点，模型 14 页 13,845 条，数据集 8 页 7,622 条，正文按相关性抽筛。厂商、研究机构与评估站点 108 个入口，73 个取得索引、35 个失败。PLaMo Translate、MiniCPM-OCR、TwIL 的本窗固定差分受 429 限制，留待核查。
- **arXiv。** 九分类网页的最新公告标签仍为 11-06，Atom 主窗提交查询返回 0，72 小时重叠窗返回 399 条，末条是 11-05 18:59:42 UTC。再次确认 `lastUpdatedDate` 在响应中被回显为 `submittedDate`。OAI-PMH 的 `arXivRaw` 补到 11-09 元数据日期桶，按九分类去重后涉及 250 项，含重叠窗内 121 个 v1 和 101 个修订版本，所见末次提交为 11-06 02:31:07 UTC。OAI 桶日期和版本提交秒分别保留，逐篇实际公开时刻及未审候选继续跟踪。 [arXiv 公告规则](https://info.arxiv.org/help/availability.html) · [cs OAI 版本数据](https://oaipmh.arxiv.org/oai?verb=ListRecords&metadataPrefix=arXivRaw&set=cs&from=2026-11-08&until=2026-11-09) · [stat OAI 版本数据](https://oaipmh.arxiv.org/oai?verb=ListRecords&metadataPrefix=arXivRaw&set=stat&from=2026-11-08&until=2026-11-09)
- **会议与索引。** AACL 11-08 口头排程核对七篇全文，线下和线上海报查到目录。OpenReview TMLR/NeurIPS API 均为 403，TMLR 镜像仅月粒度，PMLR 仍为卷级浏览。HF 日榜 11-07/08 返回空数组，11-09 查询返回 400。 [AACL 日程](https://2026.aaclnet.org/program/) · [TMLR 镜像](https://jmlr.org/tmlr/papers/) · [PMLR](https://proceedings.mlr.press/) · [HF Daily Papers](https://huggingface.co/papers)
- **开源工程。** 检查 61 个仓库，默认分支履历和 release Atom 各取得 56 个，主窗已合入 PR 清单取得 56 个，回溯专门分页 15 个仓库全部取得。另读 98 份官方补丁。PyPI 工件查询 29 项取得 26 项。主干变化按提交或合入口径，正式发布单列。维护分支、语义候选抽查和受阻端点保留续查。
- **硬件与软件协同。** 检查 NVIDIA CUDA/NCCL、AMD ROCm、Intel Gaudi、昇腾 CANN、Google TPU、AWS Neuron、Apple MLX、摩尔线程、寒武纪、沐曦及 Qualcomm 的选定公告、包与仓库，另查 MLCommons 和中英文主题检索。核对了昇腾公告与文章 JSON、HCCL/Runtime 提交和组件 tags。AMD 11 月 7 日社区帖仅汇编往期文章。摩尔线程 K3.1 公众号需验证，NVIDIA feed 传输未完整，Azure HPC 403，部分版本 API 未返回有效记录。libtpu / Neuron 组件行为、A5X 规格冲突和 CANN 9.2.1 分组件映射仍待续查。

- **续扫边界。** 本轮发现截止为 2026-11-09 09:40:38。下轮从该位置续扫并回溯 72 小时，同时继续处理分源待查记录。全局完整语义水位仍未闭合。本轮未做独立复训、设备压测或付费 API 实测。
- **时间更正。** 上期把 DSpine 与 Overnight 留作下一期复核。固定提交表明两项工件都在上期截止前公开，本期按各自原始时间补录。FlyDSL 0.5.0 区分 11 月 6 日 Git tag 和 11 月 8 日 PyPI wheel。AACL-IJCNLP 报告按会议议程入主窗，论文首发日期另列。
