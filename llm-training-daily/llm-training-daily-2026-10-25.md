# LLM 技术动态日报 · 2026-10-25

> 时区：Asia/Shanghai（UTC+8）。增量扫描窗口：**2026-10-24 08:57:49 → 2026-10-25 09:00:32**。72小时回溯：**自2026-10-21 08:57:49起**；更早未完成区间继续保留。实际生成时间：**2026-10-25T09:24:49+08:00**。投稿、工件公开、代码落地、网站更新时间与本轮发现日期分别记录，不将它们混为首发日期。
>
> 已先读运行记忆、持久化状态、本地19篇历史日报，并认证读取tracker/main索引、目录与昨日日报全文；19篇历史日报及索引的Git blob均与本地吻合。仅维护llm-monitor与tracker指定归档目录。**来源和语义审阅仍有缺口，不宣称全域监控完成。** 本轮未做GPU/NPU/TPU模型训练或性能复现；作者实验、发布者自查与本轮源码核对明确分开。

## 本次重要变化

- **Dolci训练数据出现可定位的污染披露。** Ai2组织成员列出MATH等具体行ID；GPQA仍只是筛查命中，IFEval本体未发现污染。优先核对自己用的数据版本及clean评测口径。
- **KV与通信正确性仍比“已支持”标签更需要读代码。** vLLM修READ持有rank释放、UMBP取消/完成边界，SGLang修大输入确定性all-reduce；TorchTitan的FA2回退、NCCL版本及RDNA配置须以最终patch为准。
- **Kimi扩大联合RL，开放工件要等实际兑现。** K3.1 Pro服务发布但完整权重计划11月4日前交付；RoboRSI已可核权重与部署入口，其模型训练报告仍待发布。
- **训练与推理性能有明确适用区间。** FlashInfer的训练/通信kernel、低精度路径及专用Prism引擎值得定向复现；不同精度、质量、合成接受率和实验分支不能拼成通用提升。
- **论文补录保留正反两面。** 部分参数CPT与MTP监督研究独立收录；SARIA少写SSD要付TTFT代价，Apple投机主负载不加速仍有窄正例，自奖励更一致也可能更不正确。

目录：[模型与报告](#models) · [预训练与架构](#pretraining) · [后训练](#posttraining) · [推理与缓存](#inference) · [核心开源](#core) · [扩展工程](#extended) · [评估与可靠性](#evaluation) · [覆盖与更正](#coverage)

<a id="models"></a>

## 模型与技术报告

### Kimi K3.1 Pro扩大联合RL：服务已上线，完整权重仍待11月兑现

**Moonshot；官网研究索引日期2026-10-24；本轮读取10-25官网快照，未披露独立报告版本号。** [官方全文与评测脚注](https://www.kimi.com/en/blog/kimi-k3-1-pro)、[带日期的研究索引](https://www.kimi.com/blog/)。这是K3的进一步后训练，作者称扩大后训练算力，在数百下游领域联合RL，并改进rollout调度、长上下文推理和环境质量检查。公开信息尚不足以重建奖励、采样、优化器与算力配方；“3T级、1M上下文”也不能替代精确架构和上下文质量实验。服务提供Pro及Highspeed，**完整权重计划11月4日前公开**，当前不能按已交付开放权重描述；新工件配置、许可证与自部署条件仍待核验。

作者Terminal-Bench4.0成绩65.7%，Kimi使用Claude Code、默认max effort/temperature1/top-p0.95；其他模型取博客或榜单最佳值，不能当统一harness重测。更有研究价值的是跨阶段数据控制：作者对mid-training与RL同时移除网络安全数据，内部xdev-bench-hard pass@3从未过滤检查点90.5%降到81.0%，仍高于K3的33.3%；该内部集无限硬时长/token预算，不能外推公开漏洞发现率，也未给两阶段独立消融。Highspeed“快2–3倍”缺硬件、并发和长度条件。**建议深入阅读评测脚注，继续观察权重与训练报告**；目前适合定向API评估，尚不具备完整训练复现条件。本轮未独立复测。

稳定ID：`moonshot:kimi-k3.1-pro:blog:2026-10-24`。

### RoboRSI开放权重与评测入口：Harness演进和模型训练是两条路径

**SenseNova；官方GitHub News标2026-10-22，属回溯补录；HF初版`e0c0549dd8a9e829ddb4f5ab900ca4505e0b6092`，当前`9f793e67c753e4106cae742529ed3335f1493aea`；HF绝对提交时刻未核。** 本轮已打开[实际工件树](https://huggingface.co/sensenova/sensenova-roborsi-27b-preview/tree/9f793e67c753e4106cae742529ed3335f1493aea)、[config](https://huggingface.co/sensenova/sensenova-roborsi-27b-preview/blob/9f793e67c753e4106cae742529ed3335f1493aea/config.json)、[index](https://huggingface.co/sensenova/sensenova-roborsi-27b-preview/blob/9f793e67c753e4106cae742529ed3335f1493aea/model.safetensors.index.json)、[Apache2许可](https://huggingface.co/sensenova/sensenova-roborsi-27b-preview/blob/9f793e67c753e4106cae742529ed3335f1493aea/LICENSE)和NOTICE。12个safetensors分片，index标权重张量总字节54,713,457,120；config为64层text decoder、hidden5120、每3linear attention接1full attention、27层vision、262144位置；`mtp_num_hidden_layers=1`仅为config声明，未核MTP tensor与运行支持，不能推断该头已可用。没有下载54GB权重或实机加载。当前HF提交只改中英文README，不能把部署说明修订当新权重。

[固定GitHub `b513a1c`](https://github.com/OpenSenseNova/SenseNova-RoboRSI/tree/b513a1c7289ab11eb844856588b64c378705c405)要分清两条路径：Harness RSI只改prompt/skill/tool/feedback/memory、模型参数固定；27B preview才把验证过的交互经验继续训练到Qwen3.8-27B。模型卡已给vLLM0.21.0/CUDA12.9、BF16/TP4部署和LIBERO-Pro/RoboCasa评测入口，建议显式medium reasoning，chat template缺省却是xhigh；VLA+SAM3与直接VLM控制必须分设。

卡述RoboCasa成功率26.0→46.5%、输出59.5K→15.9K是作者报告，**详细模型训练技术报告仍Coming Soon**，本轮不将其升级为训练配方/算力/消融已公开的结论。RoboDojo复现入口也仍待开放；Harness的60.08/54.17%来自GPT-6 Astra+Harness，不能嫁接给27B的26.13/20.83%。README中RoboCasa Target50+task memory与官方pretraining-kitchen榜不等协议。旧[GPT-6具身经验报告](https://www.sensenova.ai/has-gpt-6-astra-achieved-embodied-intelligence)与此待发模型报告是两份材料，不能以打开前者宣称后者全文已读。**建议先定向复现公开评测，等待训练报告后再判断联合RSI的因果收益。**

<a id="pretraining"></a>

## 预训练与架构

以下论文均为回溯首次实质收录；逐项保留原始版本日期。全文方法、实验与限制已打开，代码链接仅作入口，未冒充完整实现审计。

### AQL-Lite：部分参数持续预训练省的是权重梯度和优化器状态，反向仍穿过全网

**`2610.30344v1，2026-10-21T19:04:28Z`，回溯补录。** 在 Gemma 3 12B 的所有线性投影加 RS-LoRA，同时解冻最后八个 block 和 tied embedding/head，较晚层用较大学习率。设计将分布适配留给上层全秩权重、embedding 和全网低秩路径；只解冻上层的消融弱于单用 RS-LoRA。

正文 Table 3 在**相同5B token、8×A100 80GB、4096 packed length、bf16、ZeRO-2**下：Full CPT 四个阿语基准均分73.58、121小时，AQL-Lite 73.19、63小时，RS-LoRA-only 70.88、24小时；“恢复94%收益”是相对 base 66.91 的增益比例，**不是保留94%准确率，也不是训练总成本减半**。全网 adapter 与 embedding 要求梯度仍穿过所有层，收益来自少算权重梯度和少存优化器状态。Table 4 同1B token 下 raw crawl 66.80、完整AQL清洗70.31；后三级清洗联合变化，不能单独归因某分类器。

限制很实用：所有配置仅一次训练；主40B实验没有等规模Full CPT；时间排除语料分类器与SFT；尚未测benchmark n-gram重叠与英语遗忘，100%标准阿语不覆盖方言。Table 6外部HELM榜与作者OALL协议也不同。**建议阅读并复现同预算数据/解冻消融，避免直接引用跨榜胜负。** [正文 §3–6、Tables 3–6、Limitations、Appendix B](https://arxiv.org/html/2610.30344v1)

### MTP图搜索：增强监督密度与改变输入编码是两件事，不能把合成任务结论直接搬到自然语言

**`2610.31439v1，2026-10-22T15:35:50Z`，昨日待办补录。** NTP 在路径上容易只学局部边查表；作者用 Ranking-into-the-Future 的 Plackett–Luce 排序损失，在同一隐藏态监督多个未来节点，迫使未显式给出的多跳间隔参与训练。它不是通常“四个固定MTP head”的同义词。节点列表 `+N` 与 structural annotation（STAN）另行改善路由和表示，两者与MTP联合才突破文中的规模瓶颈。

§5–6比较从零训练的11层小模型、ER/Euclidean/path-star图，每配置五种子、每次评估一万图路径样本，最多75k步、batch1000，以有效最短路径准确率TACC判定，而非只和某条参考路径逐token匹配；推理仍用NTP。主实验可扩到125节点，不能写成现成大模型自然语言推理速度/能力提升。

§7尤其保留反证：`+N`也改善本不受causal mask限制的prefix LM，STAN也改善无法做到完整置换不变的causal LM，作者最初解释并未被消融完全证实；真正为何组合有效仍未闭合。理论“一目标token对应一hop”在自然语言不成立；约34,800 GPU小时是按2900实验×平均12小时估算，非实测日志总和。**建议作为训练目标研究阅读，先复现小图的监督与编码正交消融。** [正文 §2–7、Appendices B/J/K/L](https://arxiv.org/html/2610.31439v1)

<a id="posttraining"></a>

## 后训练与持续适配

### MOER：专家前缀后的学生续写仍是偏分布样本，不能直接套on-policy损失

**`2610.30873v1，2026-10-22T07:54:45Z`，回溯补录。** 专家CoT前缀按cosine逐步缩短，让学生补完；答案验证失败就替换为已有完整专家轨迹。前缀条件化和“只留成功续写”共同改变分布，所以补完token也归off-policy，用trust-region式软化SFT；纯自身rollout用不做advantage标准差归一化、上下界0.2/0.24的GRPO。

Table 1同**Qwen2.5-Math-7B/OpenR1-Math-46k-8192**，MOER数学六任务均分49.1、通用三任务59.0，对比LUFFY48.4/57.3、SFT-then-RL48.2/44.8。AIME24/25与AMC为avg@32，其余数学和通用任务为pass@1，不能混当统一pass@1榜；7B的Minerva38.6反而低于LUFFY40.4。Table 3去rollback后数学均分46.2且有训练崩溃，说明退回完整正确轨迹不是可省步骤。

“cost-free rollback”仅指不需再生成专家答案；失败续写的采样与验证已发生，不是整轮训练无额外成本。1.5B数学均分与LUFFY同为38.1，14B的AIME25也低于LUFFY（14.3 vs18.5），不能概括全规模全任务提升。**建议复现verification、rollback率及同总rollout成本对照。** [正文 §4–5、Tables 1–3、Appendices D–G](https://arxiv.org/html/2610.30873v1)

### HERD：难样本记忆用于限制漂移，既有checkpoint与初始base各守一条边界

**`2610.31401v1，2026-10-22T15:03:20Z`，回溯补录。** 固定初始模型编写skill，仅当它降低当前窗口verified-answer NLL才提交；记忆保存当前模型+skill仍难预测的高residual完整记录。蒸馏以verified-answer prefix上的全词表forward KL训练新LoRA，用上一checkpoint锚定近期能力、base锚定累计漂移。**residual memory只提供约束probe，不作为replay训练目标；skill writer固定base，尚非自我共同进化。**

同**Qwen3.5-4B、25任务TRACE+17、每stage32×32更新、三种子**，Table 1/2 HERD最终AA0.706，uniform replay约0.699，差0.007的95%区间[-0.001,0.016]含零；forgetting0.010 vs0.020。general能力仍平均降6.3点、最差类别降13点，不能写“零遗忘”；低遗忘SDFT的AA仅0.021，是崩溃而非保留。

主结果需verified答案、已知任务边界、每stage一个任务且无回访，只有一个base/一条25阶段流；消融单种子。dual ascent加最终audit是有限probe约束，不是所有输入能力保证。**建议阅读并复现双锚点/记忆筛选消融，再评估真实用户流中的标签与边界来源。** [正文 §2–3、Tables 1–3、Limitations、Appendix A](https://arxiv.org/html/2610.31401v1)

<a id="inference"></a>

## 推理优化与缓存

### MoEZip：KV淘汰还会改变专家路由，保留锚点不能只看注意力

**`2610.30206v1，2026-10-21T13:06:30Z`，回溯补录。** 用instruction token的router Fisher trace估计脆弱性，再经instruction→context注意力把它传播到上下文位置；在固定每层/头KV预算中先留routing anchors，再由KVZip等基准补齐。未来答案token尚未生成，因此instruction信号是代理；局部各向同性小扰动推导不等于实际淘汰扰动分布证明。

Table 1/11的**Qwen3-30B-A3B-Instruct-2507、LongBench12任务、保留10%KV**：KVZip19.51，使用同一固定重建query且可跨请求复用的MoEZip-Recon41.95，下游query条件化MoEZip44.89。这样可分开路由信号与额外query条件化收益；后者缓存不再query无关。实验主体用2×A100 80GB；235B扩展用8卡，不能混成单卡结论。Qasper scoring从9.36到10.24秒，额外0.88秒，**没有新的整机decode加速倍数**。

需要可见router输出和attention统计，黑盒API不适用；跨请求复用与条件化质量存在取舍。**建议复现同预算、同query的Recon对照，并测选取阶段和完整请求时间。** [正文 §3–4、Table 1、Appendices C/D/E.6](https://arxiv.org/html/2610.30206v1)

### SARIA：SSD少写很多，要连同相对全量缓存更慢的TTFT一起看

**`2610.31169v1，2026-10-22T12:25:27Z`，主题检索补录。** DRAM内频繁复用不代表需要SSD持久化；SARIA用跨驻留期的淘汰次数/复用阈值决定写入，统一FA KV与GDN checkpoint的前缀metadata，防止只留FA却缺恢复所需状态。SSD命中后只部分晋升DRAM，并在GDS活跃时节流DRAM流量，给两路共享GPU入口留带宽。

§5主机为**单RTX PRO 6000 Blackwell Server 96GB、256GB主存、两块1.92TB PCIe5 NVMe**，BF16 Qwen3-8B/Qwen3.5-9B。WEKA/Qwen3.5-9B的E3阈值比Full Duplicate少写78.2%，但平均TTFT高78.4%；相比DRAM-only则低约94.9%。这是寿命—延迟交换，不是无损优化。TraceLab下相同E3对9B仍超一天写入预算13.3%，须改E5；阈值不能跨负载直接照抄。

关键边界：TraceLab/WEKA没有原始prompt，重建的是保持记录复用关系的确定性合成输入；每session最多一个活跃请求，默认最多四请求在途，**只生成首token以隔离prefill**。小A30消融还用了随机初始化的缩层1.7B/5.6B配置，并非同名发布权重；别把结果当完整agent质量或长decode收益。估计带宽失准会损失性能，GDS单独已饱和入口时节流可能变慢。**建议对自有trace复现SSD寿命、尾TTFT与完整prefill/外存读写曲线。** [正文 §4–5、Figure 10、Table 2](https://arxiv.org/html/2610.31169v1)

### Apple Silicon推理评估：投机接受长度增大，并不足以抵消完整draft–verify路径

**`2610.31479v1，2026-10-22T16:04:40Z`，回溯补录；HTML404后已实际下载36页PDF。** 主要投机对照固定**M2 Pro/M3 Ultra、MLX、Qwen3-4B的4bit与BF16 target、batch/concurrency1、100对话prompt（46工具/54聊天）、greedy、512输出上限、三次测量**；平均输入1776token，基线输出约54–62token。Classic、N-gram、Cascade、EAGLE3、DFlash、DDTree在这组负载中均无标准化端到端加速。4bit DFlash在M3 Ultra接受2.57个draft token，但本实现decode156.7→60.0token/s；必须用各自AR基线，不能跨实现绝对吞吐排榜。

正文同时保留正例，不能写成“Apple不能投机”：p9 Table 2在另一组M3 Pro/Qwen3-4B-BF16配对实验中，DFlash/DDTree的标准化端到端吞吐分别提高9.2%/15.5%；对应AR基线仅15.89token/s，不同实现不能构成受控硬件或方法排名。p12 Table 4的另一组**8B-4bit、256输出、两次中位数、小M Marlin port**，k7时stock为plain的0.99×，换kernel后1.13×；k2/4本来也可能受益。attention port在Qwen3-4B、16K、256输出下M4 Pro端到端1.20×，M2却0.970×，额外dispatch和partial往返是否划算随GPU而变。

public MSL缺公开异步copy接口不等于硅片没有该能力；作者明确存在私有intrinsic。KV pruning没有任务质量评测；megakernel负例只是一块M4 Pro上的一个MoE block，不能证明所有Metal megakernel不可能。**建议优先阅读负载定义与附录，再按自己的prompt/输出长度测TTFT、decode和总时延。** [PDF pp2–3/7–8/11–18/Appendix A、Figures 6/7、Tables 4/6](https://arxiv.org/pdf/2610.31479v1)

### StartLux-Prism补录：共享证据的树形批处理与确定性INT8，适用范围是专用决策模型

**StartLuxLabs；仓库创建于2026-10-13，本次补录10-22的main快照，固定`f32ed8eee9d89ea56dde03a978cf4128d2bbf497`；未另核初次公开实现日期，不作为10-25首发。** [固定README](https://github.com/StartLuxLabs/StartLux-Prism/blob/f32ed8eee9d89ea56dde03a978cf4128d2bbf497/README.md)、[设计](https://github.com/StartLuxLabs/StartLux-Prism/blob/f32ed8eee9d89ea56dde03a978cf4128d2bbf497/docs/design.md)、[完整基准条件与负面结果](https://github.com/StartLuxLabs/StartLux-Prism/blob/f32ed8eee9d89ea56dde03a978cf4128d2bbf497/docs/benchmarks.md)。它面向StartLux-Decision v1/v1.1的`/v1/systemone`概率判定请求，服务4B/9B/27B与条件思考路径。一次请求中多个问题共用同一证据，系统前缀→证据→各问题组织成分段树，body处理所有token，最后一层head只处理答案读取位置；两组CUDA Graph按bucket重放。跨请求缓存同时保存线性注意力递归状态/卷积尾及全注意力KV，命中须精确token与图像digest匹配。实现已读`Decider._probs → Engine.letter_logits/run`、`Engine.launch`及`PrefixCache.lookup/insert/discard`：缓存索引分配不等于GPU数据完成，后续读取依靠同stream顺序；图像准备失败会撤销尚未写入的新缓存。

数值路线以SmoothQuant、逐token激活量化、INT32归约和一次缩放，减少不同batch/tile/split的舍入差异。“bit-identical”主要指**该INT8/W4A8路径在batch、缓存及并行方式间一致**，不是与BF16原模型逐位一致；作者精度测试仍出现近似平局翻转，BF16 cuBLAS路径也不保证batch invariance。[测试说明](https://github.com/StartLuxLabs/StartLux-Prism/blob/f32ed8eee9d89ea56dde03a978cf4128d2bbf497/docs/testing.md)应与性能表一起读。

10-13/14作者测试的v1.1-27B、单H200、并发1、同证据三问且关闭思考时，Prism INT8为17.4ms，官方BF16工件经vLLM0.31动态FP8为92.9ms，形成约5.3倍；8,192token单问为439.8对531.2ms，仅约1.2倍。231项JevBench开启思考后，平均时延反而是1,212对1,150ms；两边单序列思考均94token/s，64客户端吞吐7.61对6.52请求/s，但对应正确数为219–222/231（均220.5）对221–228（均225.2）。两边权重舍入、SSM状态精度及随机数生成器不同，不能当等质量吞吐比较或同数值模型全面加速。表内Prism实验快照`6e47e4d`本轮GitHub查询422无法定位，公开当前代码不能自动充当原实验同源证明。

**建议深入阅读、考虑复现**共享证据和batch一致性设计，先用固定模型/精度/负载核时延与概率误差。当前仓库自己的代码明确“尚未选择许可证”，仅部分vendored文件为Apache2.0，权重为CC-BY-NC4.0；因此不能据“开源引擎”宣传直接建议商业采用。本轮未在GPU运行测试，支持范围也不能外推为通用聊天模型引擎。

稳定ID：`github:StartLuxLabs/StartLux-Prism:commit:f32ed8eee9d89ea56dde03a978cf4128d2bbf497`。

<a id="core"></a>

## 核心开源工程：状态寿命、默认行为与低精度边界

本节以UTC merged_at归窗（2026-10-24T00:57:49Z→2026-10-25T01:00:32Z），main、release/0.4和其他base分别判断。已打开官方GitHub发布集合、PR最终补丁与关键frozen源码；所有性能/质量数字均为作者报告，本轮未执行模型硬件实验。窗口内六仓公开Release全页枚举0新增，以下main合入不等于已进入正式wheel。对PR正文与最终代码冲突，以最终patch/merge commit为准。

### Megatron checkpoint转换：只在不支持的输出格式清除integrity标志

torch_dist源checkpoint带--verify-integrity转为torch时，旧代码把该标志一起存进目标args，随后加载不支持integrity的torch格式会受影响。setup_model_and_optimizer先按源设置完成load，再切目标格式；目标不是torch_dist才清verify_integrity，目标仍为torch_dist则保留。补丁的回归以mock load/save检查顺序和四种参数组合；作者最终只在CPU/Gloo且stub模型并行的环境复测，未重跑GPU/NCCL端到端转换。run_config.yaml sidecar与tools/checkpoint/convert.py另一入口未修。PR merged_at在主窗，但commit时间是10-23T23:17:30Z，以合入事件归窗。 **建议checkpoint工具维护者采用该修复，并分别检查sidecar和独立converter入口，复现时先验源校验顺序再验目标可重载。**

源码阅读路线：`megatron/training/training.py::setup_model_and_optimizer；tests/unit_tests/test_training.py::test_checkpoint_conversion_preserves_supported_integrity_settings`。

证据基线：[NVIDIA/Megatron-LM #6502](https://github.com/NVIDIA/Megatron-LM/pull/6502)，`main @ 2052769557dc`，2026-10-24T02:07:11Z。

稳定ID：`github:NVIDIA/Megatron-LM:pr:6502`。

### TorchTitan新增Qwen3-30B-A3B DAPO数学双精度配置

新配方把4张B200 trainer（FSDP4/EP4）和4张generator（DP4/EP4）用于8K回复、32 prompts×16 samples、50步DAPO；两端按bf16或mxfp8选择DistMoeTransform，MXFP8 generator用mega pipeline与fused SwiGLU，lm_head输出FP32。作者steps3–50测step 24.3→22.7s，改善主要来自等待rollout，reward约低0.01；AIME初值不同且只有50步，不能认定质量等价或长期RL收敛。适合复现训练/rollout低精度配比，配置不是独立新训练算法。 **建议训练平台团队考虑复现这套8卡配方，固定随机种子、起始reward和rollout长度，比较单位有效token成本与长程质量。**

源码阅读路线：`torchtitan_recipes/rl/dapo_math.py::_qwen3_30b_a3b_dapo_math_8k_config`。

证据基线：[pytorch/torchtitan #5598](https://github.com/pytorch/torchtitan/pull/5598)，`main @ df4e4dc22a33`，2026-10-24T13:04:29Z。

稳定ID：`github:pytorch/torchtitan:pr:5598`。

### TorchTitan缺FlashAttention wheel：实际合入是提前报错，PR正文的FA2回退未落地

最终merge commit的get_cuda_flash_attention_impl按能力选择FA4/FA3，再用find_spec检查flash_attn.cute或flash_attn_interface；缺wheel就raise ValueError，提示安装或显式选择其他attention backend。它没有实现正文所写的自动FA2回退，原has_cuda_capability仍是>=比较，也没有正文声称的未来SM family限制。正文L4上fallback与FA2的零误差表不能当最终两文件补丁的验证。已额外打开merge commit和完整utils.py核对；部署方应按提前诊断处理。 **建议使用Varlen的部署团队优先核查wheel存在性和显式backend选择；继续观察真正fallback实现，避免依正文误配。**

源码阅读路线：`torchtitan/tools/utils.py::get_cuda_flash_attention_impl、has_cuda_capability；tests/unit_tests/cpu/tools/test_tools_utils.py`。

证据基线：[pytorch/torchtitan #5546](https://github.com/pytorch/torchtitan/pull/5546)，`main @ 89e4339c963c`，2026-10-24T23:19:48Z。

稳定ID：`github:pytorch/torchtitan:pr:5546`。

### TorchTitan可选后端与release EnvServer隔离补齐

attn_gym的module可import但懒属性访问才因cutlass缺失失败；_load_export现沿显式__cause__识别后端ModuleNotFoundError，返回带原错误的placeholder，真正用到该kernel的模型在build时失败，其他模型仍能import。缺attn_gym内部模块、普通ImportError和隐式context链仍直接暴露，不能吞包损坏。另release/0.4的EnvServer在spawn pool前设置TORCHTITAN_DISABLE_TORCH_COMPAT=1，避免新worker仅import taskset就引入Torch；trainer继续原兼容shim。后者仅release分支，作者缺verifiers跳过目标Terminal Bench测试，不能写成main或已验证完整agent rollout。 **建议跨平台开发者深入阅读懒导出错误分类，并在真实verifiers环境复测release worker是否仍保持torch-free。**

源码阅读路线：`torchtitan/models/common/attn_gym.py::_load_export、_is_backend_error；torchtitan/rl/examples/verifiers/env_server.py::_run_env_server_process`。

证据基线：[pytorch/torchtitan #5565](https://github.com/pytorch/torchtitan/pull/5565)，`main @ 3ed5aa976fdf`，2026-10-24T01:16:54Z；[pytorch/torchtitan #5615](https://github.com/pytorch/torchtitan/pull/5615)，`release/0.4 @ 5804c17d5499`，2026-10-24T18:07:44Z。

稳定ID：`github:pytorch/torchtitan:pr:5565`；`github:pytorch/torchtitan:pr:5615`。

### vLLM KDA RecoverSSM：verify scratch移出KV页，CuTe verify/commit落地

每request只保留一份KDA checkpoint，verify把逐token correction/key/gate写到每层按max_num_seqs分配的step buffer，sampling后commit折入接受token；临时record不再跟Mamba page被KV connector复制，page回到conv+SSM state，因此解除RecoverSSM与KV transfer的互斥。CuTe在SM10x、BF16 activation/conv/SSM state、head_dim128、width4无bias卷积、bounded gate、窗口≤8（≤7draft）时接管，否则Triton；不对齐activation还可逐次回退。commit跨层一次launch且KV rebind重建context。必须Model Runner V2，DBO明确拒绝，以免共享record行。作者GB300真实rejection gsm8k/1P1D测试与kernel/preemption回归有证据；+1.9%吞吐/−4.2%均值ITL来自一小时synthetic acceptance的2P1D AgentX，不能外推真实接受率或所有拓扑，且P90 TTFT变差。 **建议Kimi-K3 P/D和KV-offload团队优先读record生命周期并考虑复现preemption/abort、KV rebind与真实rejection负载，确认scratch/HBM节省能否转为可服务并发。**

源码阅读路线：`vllm/models/kimi_k3/nvidia/kda.py::_recoverssm_verify；ops/recoverssm.py::KDARecoverSSMRecords、KDARecoverSSMCommitContext.commit；recoverssm/kernels.py::can_use_cute_recoverssm、CuteRecoverSSMVerify`。

证据基线：[vllm-project/vllm #63045](https://github.com/vllm-project/vllm/pull/63045)，`main @ ae065918640a`，2026-10-24T01:32:27Z。

稳定ID：`github:vllm-project/vllm:pr:63045`。

### vLLM adaptive verification接通batch-sharded sampling，以host wait换精确分片

draft预算受限时，GPU实际admitted counts与CPU scheduled上界不同，旧分片计划会让all-to-all的行数错位。新路径仅在预算绑定且启用batch-sharded sampling时在copy stream异步D2H counts，shard_sampler_inputs等event后重建GEMM/all-to-all/sampler计划；预算0也选与replicated一致的sampler。减少logits驻留的原有TP分片收益仍在，但CPU不能无限超前：作者4×GB200 TP4的c1多约0.05ms，主表吞吐/TPOT在噪声内，高并发median ITL小幅改善。作者为ABBA比较额外固定cost curves且排除一轮异常baseline，不能把表当无条件吞吐提升。 **建议大TP/大词表服务考虑复现logits峰值内存与CPU等待成本；吞吐对照应重复启动并保留cost-curve差异。**

源码阅读路线：`vllm/v1/worker/gpu/spec_decode/adaptive_verification.py::AdmittedDraftCounts.wait、reallocate_drafts；sample/batch_shard.py::BatchSharder.shard_sampler_inputs`。

证据基线：[vllm-project/vllm #58466](https://github.com/vllm-project/vllm/pull/58466)，`main @ 7050ef881ff0`，2026-10-24T20:35:18Z。

稳定ID：`github:vllm-project/vllm:pr:58466`。

### vLLM MoRIIO READ：按持有rank释放，异布局按head搬运

fan-in READ以前只ACK实际读取的prefill rank，其他持有副本的rank要等默认60秒reaper；现在completion、failure、abort都向get_moriio_read_holder_tp_ranks列出的全部holder发release。等TP但cache布局不同不能整块copy，改按head算offset；相同布局保留whole-block。flexible MLA mirror在RDMA仍轮转source，在单节点xGMI固定可达source。93个真实cache tensor+fake transport回归及MI355X dummy-weight P/D测试检查byte exact和释放；split-head/Mamba跨TP READ是另一个PR，且这里不关闭旧10秒P2P晚到WRITE生命周期问题。 **建议ROCm异TP P/D部署优先采用并按自身cache布局、abort/failure检查每个holder ACK计数。**

源码阅读路线：`vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py::get_moriio_read_holder_tp_ranks、_compute_read_attention_offsets、_send_read_release；tests/v1/kv_connector/unit/test_moriio_read_tp.py`。

证据基线：[vllm-project/vllm #62473](https://github.com/vllm-project/vllm/pull/62473)，`main @ 2aa7269a8375`，2026-10-24T11:53:21Z。

稳定ID：`github:vllm-project/vllm:pr:62473`。

### vLLM UMBP第3片：共享lookup、hybrid边界与转移取消协议

共享connector以request对象、local_tokens、num_tokens和hash序列识别异步lookup上下文，变化则取消旧future；把external hits交核心KV coordinator，full-attention保持prefix闭合，Mamba/稀疏组恢复自己的边界state。whole-block batch计划减少逐range对象开销；finished partial tail先pin整页，worker事件完成再归还，部分失败store不publish。preemption调用runtime.cancel；runtime没有cancel则wait以确保buffer可重用，不能把取消提交当DMA已排空。KV事件需各worker成功/驻留重验，另有bounded tracking上限。此PR是runtime之上的实现分片，并非完整UMBP产品发布；专门async/hybrid/events测试在后续PR，当前含core回归但本轮未运行。 **建议KV存储集成者深入阅读cancel/wait、partial-tail pin与publish完成契约，继续观察后续runtime和专门测试片段。**

源码阅读路线：`vllm/distributed/kv_transfer/kv_connector/v1/umbp/scheduler.py::get_num_new_matched_tokens、_coordinate_external_hits、register_finished_partial_tail；worker.py::_cancel_job、_drain_store_jobs`。

证据基线：[vllm-project/vllm #61791](https://github.com/vllm-project/vllm/pull/61791)，`main @ c0f9fff649e7`，2026-10-24T12:25:44Z。

稳定ID：`github:vllm-project/vllm:pr:61791`。

### vLLM ModelOpt mixed draft：先映射量化层名，再构建DFlash/DSpark

nested MIXED_PRECISION元数据现在选modelopt_mixed；draft量化设置与权重同走alias、runtime prefix和decoder offset映射，避免fc权重被构建成BF16却随后加载fc.weight_scale。映射只针对ModelOpt，其他backend保留原语义与既有exclusion alias。per-tensor FP8在FNUZ设备通过既有FN→FNUZ helper同时规范weight与静态input scale。CPU配置测试、小随机checkpoint及B200 GPT-OSS-20B短训draft smoke支持功能加载；后者35 prompts的1120 token与SpecDec OFF一致却仅16/2140 proposals接受，不是质量或加速证据。作者最新AMD normalization验证仍待CI，不能当MI300全链已关闭。 **建议量化draft使用者先复现checkpoint加载和实际量化层dtype，再等最新AMD CI闭合；不要用短训draft接受率判断模型质量。**

源码阅读路线：`vllm/model_executor/models/utils.py::map_draft_quant_config；qwen3_dflash.py::DFlashQwen3Model；quantization/modelopt.py::process；model_arch_config_convertor.py::_normalize_quantization_config`。

证据基线：[vllm-project/vllm #63001](https://github.com/vllm-project/vllm/pull/63001)，`main @ 5f63ea461ccd`，2026-10-24T12:28:36Z。

稳定ID：`github:vllm-project/vllm:pr:63001`。

### vLLM多模态cache释放变线性，Gemma4图预算与处理器对齐

EncoderCacheManager新增每request每mm_hash引用计数，释放每个input不再反复扫描其他输入，整request由O(N²)变O(N)，相同hash最后一次出现释放前仍不可驱逐。作者4096-item CPU微测235→1.2ms，无GPU端到端；代价是每request字典，直接修改request_cached_ids的外部subclass也要维护计数。Gemma4把processor max_soft_tokens解析收口_get_max_soft_tokens，encoder CUDA graph捕获预算、LoRA padding与token计数共用用户有效配置，避免仅按checkpoint默认长度捕获图。 **建议长视频/多图prompt团队复现释放延迟和重复hash寿命；Gemma4用户把自定义soft-token预算加入graph启动回归。**

源码阅读路线：`vllm/v1/core/encoder_cache_manager.py::_add_cached_input_id、free_encoder_input；vllm/model_executor/models/gemma4_mm.py::_get_max_soft_tokens`。

证据基线：[vllm-project/vllm #62850](https://github.com/vllm-project/vllm/pull/62850)，`main @ 49d634bd6a3d`，2026-10-24T05:58:07Z；[vllm-project/vllm #60687](https://github.com/vllm-project/vllm/pull/60687)，`main @ 4920df120d8e`，2026-10-24T05:21:38Z。

稳定ID：`github:vllm-project/vllm:pr:62850`；`github:vllm-project/vllm:pr:60687`。

### vLLM MLA单位FP8 scale可跳过读scale

concat_and_cache_mla新增skip_scale模板分支，只有plain FP8 cache且模型_k_scale_float==1.0时用常量1；fused norm/RoPE路径同步接线。它省的是单位scale的读取/运算，不是取消FP8量化，也不改变非单位scale和其他cache格式。作者GLM-5.3、B300 TP4/MTP5、低并发随机短输入测TPOT约1.2–3%改善并做AIME25对照，不能扩成所有MLA或GPU的服务收益。 **建议GLM/MLA服务按单位与非单位scale各跑cache数值回归，再在真实batch分布测收益。**

源码阅读路线：`csrc/cache_kernels.cu::concat_and_cache_mla_kernel；vllm/model_executor/layers/attention/mla_attention.py::update_kv_cache；vllm/models/deepseek_v32/common/kernels.py::_fused_norm_rope_kernel`。

证据基线：[vllm-project/vllm #62102](https://github.com/vllm-project/vllm/pull/62102)，`main @ 56e4e42868b4`，2026-10-24T20:26:20Z。

稳定ID：`github:vllm-project/vllm:pr:62102`。

### vLLM RDNA3/3.5 W4A8：MXFP4权重精确映射int8，激活数学已改变

最终通过--quantization-config.linear.activation int8_per_group_32为gfx11小M≤8 decode启用int8-dot GEMV：E2M1×2精确落整数查表，activation每32元素量化int8，fp32组scale合成；直接读Quark OCP checkpoint不额外repack。prefill保留现有仿真GEMM结构，但启用flag时不做checkpoint声明的MXFP4 activation QDQ，因此这是显式数值路线选择；PR正文残留env开关未出现在最终合入，不能照抄。用户activation override优先且kernel匹配检查该量化方案。Strix Halo作者5.8×对默认仿真数字用了额外量化lm_head的实验改动，合入不自动量化原BF16 lm_head；不可把该数字当stock checkpoint开flag即得。 **建议RDNA用户先以stock BF16 lm_head复现质量，再单独研究lm_head量化收益，防止把两项变化混为一个结果。**

源码阅读路线：`csrc/rocm/mxfp4_w4a8_gemv_rdna3.cu；vllm/model_executor/kernels/linear/__init__.py::_resolve_linear_activation_key；kernels/linear/mxfp4/base.py::_supports_quant_scheme`。

证据基线：[vllm-project/vllm #60413](https://github.com/vllm-project/vllm/pull/60413)，`main @ c0ba7e01ffdd`，2026-10-24T12:09:44Z。

稳定ID：`github:vllm-project/vllm:pr:60413`。

### vLLM MiniMax-M3 indexer-only CP：按block分读，保留外部kernel边界

ROCm gfx950的opt-in indexer CP使每rank投影全部index heads、只评分1/P context blocks，再交换各rank候选给head owner，削减随上下文增长的cache读；并非把整模型attention改成context parallel。最终配置字段与FP8 indexer guard、TP group IPC peer buffer/MTP路径已接入，外部AITER #6359已于10-13合入c09dc31787c2，仍需所用安装包含该实现。最终开关是--attention-config中的minimax_m3_indexer_cp=true（不是正文旧indexer_cp），必须decode_context_parallel_size=1；auto/bf16选Triton，fp8/fp8_e4m3才选gfx950 AITER MSA。本片段不把作者AgentX提高数字当独立复现。 **建议长上下文MiniMax ROCm团队对比indexer读带宽和候选block一致性，并固定实际AITER安装commit。**

源码阅读路线：`vllm/config/attention.py；vllm/models/minimax_m3/amd/indexer_msa.py::MiniMaxM3IndexerMSAImpl；amd/indexer_cp.py::IndexerCpPeers`。

证据基线：[vllm-project/vllm #57909](https://github.com/vllm-project/vllm/pull/57909)，`main @ cba669d45d12`，2026-10-24T12:03:35Z。

稳定ID：`github:vllm-project/vllm:pr:57909`。

### vLLM DSv4-Pro ROCm mono layer：TP8双persistent launch，性能对照含栈变化

DSv4-Pro新增独立FlyDSL双launch整层decode实现，K1后运行原compressor/CSA indexer，再K2完成attention/MoE与TP归约。明确要求CDNA4、TP8、无EP/DP/EPLB、AITER A8W4专家/FP8线性及padding、足够常驻CU；非decode、超64rows、piecewise/missing metadata等step回原路径，首次peer handshake必须在eager。作者历史AgentX baseline与新candidate同时换Torch/ROCm/vLLM且用synthetic acceptance，11.9–28.7%不可归因本PR；真实评测AIME26/30→24/30、诊断追加长预算为选择性重测，不足以证明质量等价。另MI325X配方启用已合入的V4.1 mono（TP4≤42rows，权重副本约170GiB/rank，MI300X不启）；MLA DCP_VERIFY env开关删除，按原auto启发式选cprr/segmented/Triton，升级脚本应清旧override。 **建议MI355X低并发部署先做数值与长输出回归，再用同栈ABBA重测；对生产配方和被删除override另做启动参数检查。**

源码阅读路线：`vllm/models/deepseek_v4/amd/mono_decode.py::_deployment_error、MonoDecodeLayer._decode_step；recipes AMD MI325X配置；vllm/v1/attention/backends/mla/rocm_aiter_mla.py`。

证据基线：[vllm-project/vllm #61482](https://github.com/vllm-project/vllm/pull/61482)，`main @ 139feb45348d`，2026-10-24T12:00:40Z；[vllm-project/vllm #62713](https://github.com/vllm-project/vllm/pull/62713)，`main @ 7822663cb480`，2026-10-24T12:12:01Z；[vllm-project/vllm #61628](https://github.com/vllm-project/vllm/pull/61628)，`main @ 39504080e56e`，2026-10-24T12:08:44Z。

稳定ID：`github:vllm-project/vllm:pr:61482`；`github:vllm-project/vllm:pr:62713`；`github:vllm-project/vllm:pr:61628`。

### vLLM MiniMax-M3.1配方扩成DEP2+DEP4，模型支持PR仍open

GB200/GB300配方改为一DEP2 prefill tray加一DEP4 decode tray，NVIDIA DSpark默认7tokens/FLASHINFER_TS；DP ready timeout到3600秒，NIXL engine_ttl=0避免闲置远端清理。作者在公开minimax-m31-flash专用镜像GB300上运行真实rejection并通过gsm8k/passkey，GB200及部分B200/B300单节点smoke仍待；recipe static JSON同机co-location与interactive builder两tray结果还有已知差异。主线模型支持#63232仍open，合入配方不等于stock wheel已支持，早期tuning的NIXL push/Mooncake pool/synthetic acceptance也不属于这套公开pull配方。 **建议MiniMax部署继续观察#63232主线落地，使用专用镜像时固定digest并按自己GPU和renderer核对角色拓扑。**

源码阅读路线：`recipes/models/MiniMaxAI/MiniMax-M3.1-Flash.yaml`。

证据基线：[vllm-project/vllm #63428](https://github.com/vllm-project/vllm/pull/63428)，`main @ c50aca14416e`，2026-10-24T03:43:49Z。

稳定ID：`github:vllm-project/vllm:pr:63428`。

### vLLM前端工具规则、profiling会话参数与LMFE弃用

ChatStreamProcessor与batch共用keeps_tool_calls，tool_choice none及null按既有batch语义丢tool calls、finish_reason=stop，覆盖Harmony及derender；不是重新定义null或阻止模型生成工具语法。start_profile从HTTP/LLM/AsyncLLM经engine/executor传profiler_kwargs，TorchProfilerWrapper每个inactive新会话从base配置重建override，active不改；不提供override则保留外部worker兼容。LM Format Enforcer只开始warning并宣布v0.35.0移除，当前仍可选。ViT mRoPE setup包装为PluggableLayer，允许外部平台替换；该PR无硬件验证结果，不能报Ascend性能。 **建议API与可观测性维护者回归stream/batch工具一致性和连续profiling会话；LMFE用户提前迁移并核查语法覆盖。**

源码阅读路线：`vllm/entrypoints/openai/chat_completion/message_assembly.py::keeps_tool_calls；vllm/profiler/wrapper.py::set_profiler_kwargs；vllm/config/structured_outputs.py；mrope_vit_setup.py::VitMropeSetup`。

证据基线：[vllm-project/vllm #63345](https://github.com/vllm-project/vllm/pull/63345)，`main @ f959137753c4`，2026-10-24T06:36:02Z；[vllm-project/vllm #51974](https://github.com/vllm-project/vllm/pull/51974)，`main @ dd5e754494b4`，2026-10-24T20:22:12Z；[vllm-project/vllm #63423](https://github.com/vllm-project/vllm/pull/63423)，`main @ 4e591657157a`，2026-10-24T22:19:15Z；[vllm-project/vllm #63017](https://github.com/vllm-project/vllm/pull/63017)，`main @ e9acf48614a2`，2026-10-24T06:44:58Z。

稳定ID：`github:vllm-project/vllm:pr:63345`；`github:vllm-project/vllm:pr:51974`；`github:vllm-project/vllm:pr:63423`；`github:vllm-project/vllm:pr:63017`。

### SGLang KVShard zero-SM gather：最终硬门槛是NCCL 2.31.2

prefill KV-shard prefix allgather原用NCCL SM核与attention竞争，现默认尝试专用NCCL_CTA_POLICY_ZERO communicator，双scratch slot以相同4KiB offset切入单个ncclMemAlloc symmetric window；最后tensor view销毁前同步后deregister/free。只支持CUDA单节点shard group、无memory-saver/custom pool、NCCL≥2.31.2（23102）；早期正文≥2.28及2.30 workaround已落后最终guard，2.31.2才修内部CE同步buffer和双rank multicast race。注册NULL仍落SM路径；跨节点shard group继续Torch scratch，部署内单节点group可在多节点server使用但需自行cuMem设置。engine在eligible单机默认NCCL_CUMEM_ENABLE=1会作用所有communicator；勿设置全局NCCL_CTA_POLICY覆盖专用策略。作者表中小消息单独gather更慢，优势主要是大消息/overlap，非整模型无条件加速。 **建议KVShard团队优先深入读最终version guard并用profiler确认无NCCL kernel、确有peer copy，再比较小消息负收益与overlap。**

源码阅读路线：`python/sglang/srt/mem_cache/page_interleave_pool.py::_symmetric_scratch_supported、_alloc_symmetric_scratch；distributed/device_communicators/pynccl_allocator.py::allocate_symmetric_buffer；test_page_interleave_gather_multi_gpu.py`。

证据基线：[sgl-project/sglang #44029](https://github.com/sgl-project/sglang/pull/44029)，`main @ 3ea7e210d28e`，2026-10-24T09:40:18Z。

稳定ID：`github:sgl-project/sglang:pr:44029`。

### SGLang Engram peer-shards：以每GPU HBM换掉DP lookup collectives

opt-in SGLANG_OPT_DSV41_ENGRAM_PEER_SHARDS让TP row shard驻CUDA VMM并映射所有peer，Triton以shard starts二分owner，直接按本rank token读取FP8 row/E8M0 scale并规范−0→+0；覆盖TP/DP/prefill-CP，不再all-gather ids、分配DP×tokens buffer或reduce-scatter。优先于host table，每GPU仍付table_size/TP的HBM，减少可用KV池；跨host必须同NVLink域FABRIC/IMEX，否则POSIX fd只可单host而明确报错。作者8GB300跨两节点和8B300单机真实表规模bitwise/CUDA graph测试，未给通用吞吐数字。 **建议宽DEP且有NVLink域的DSv4.1团队考虑复现peer映射与KV容量的权衡；无FABRIC跨机不应开启。**

源码阅读路线：`python/sglang/srt/layers/engram.py::_PeerShards、ShardedEmbedding.forward；python/sglang/kernels/ops/embeddings/engram_gather.py::_engram_gather_peer_kernel；utils/cuda_vmm_utils.py::exchange_shareable_handles`。

证据基线：[sgl-project/sglang #46180](https://github.com/sgl-project/sglang/pull/46180)，`main @ 79701a2549c3`，2026-10-24T02:01:26Z。

稳定ID：`github:sgl-project/sglang:pr:46180`。

### SGLang DSv4.1 prefill graph三处接线，MegaGate同窗启用后撤回

SWA location在无request-window/ring时每step只translate一次并原地址刷新；ratio2 compressor的request start/count形成int32[2,slots]图内buffer，复用kv|score投影走已有fused gate，非fused继续Torch。mega attention把FP32 accumulator直接产MXFP8 data/UE8M0 scale给wo_b，live/padded stride不同会临时live buffer再copy并清padding，数值跟eager而非原BF16中间表示，不应称bitwise保持旧图输出。MegaGate prefill #46232在03:31合入，04:15即#46242恢复decode/target-verify≤2048rows：GB300、DP attention4/MegaMoE、4096和8192两形状捕获在非零DP rank unspecified launch failure（sgl-deep-gemm0.2.3.post2）。截至cutoff不得报prefill MegaGate已可用；既有decode MegaGate仍在。 **建议DSv4.1团队回归两种prefill graph shape与真实padding；继续观察MegaGate prefill修复，不重新打开已撤回路径。**

源码阅读路线：`python/sglang/srt/layers/attention/deepseek_v4_backend.py::prepare_forward_metadata_for_breakable_cuda_graph_replay、forward_low_ratio_sources；models/deepseek_v4.py::deepseek_v4_mega_attention_with_mxfp8_output；layers/moe/mega_gate.py::should_use_mega_gate`。

证据基线：[sgl-project/sglang #46229](https://github.com/sgl-project/sglang/pull/46229)，`main @ 54df642fe663`，2026-10-24T04:33:16Z；[sgl-project/sglang #46230](https://github.com/sgl-project/sglang/pull/46230)，`main @ 7d996300b98e`，2026-10-24T04:33:43Z；[sgl-project/sglang #46231](https://github.com/sgl-project/sglang/pull/46231)，`main @ 564aec61c8b1`，2026-10-24T04:31:33Z；[sgl-project/sglang #46232](https://github.com/sgl-project/sglang/pull/46232)，`main @ 3c6a6c806865`，2026-10-24T03:31:50Z；[sgl-project/sglang #46242](https://github.com/sgl-project/sglang/pull/46242)，`main @ 3b95120f000c`，2026-10-24T04:15:39Z。

稳定ID：`github:sgl-project/sglang:pr:46229`；`github:sgl-project/sglang:pr:46230`；`github:sgl-project/sglang:pr:46231`；`github:sgl-project/sglang:pr:46232`；`github:sgl-project/sglang:pr:46242`。

### SGLang V4 paged dequant四token向量化，限定V4布局

paged KV dequant从每program一token改为四token批读FP8/scale/BF16 RoPE，用mask覆盖尾token和padding scale tile，保留int64 offset与caller output stride。补丁只改KVLayout.V4，不碰V4.1，也不再次宣称上游已有compressed-cache reuse。作者旧B200大gather kernel约4.3–4.4×，16K无收益；rebase后GPU和E2E TTFT未重测。本轮应按局部vectorization与零容差reference测试入口看待。 **建议V4长prefill团队考虑复现current-head零容差测试和TTFT，V4.1用户不据此改参数。**

源码阅读路线：`python/sglang/kernels/ops/attention/dsv4/dequant_k_cache.py::_dequantize_k_cache_paged_kernel`。

证据基线：[sgl-project/sglang #42929](https://github.com/sgl-project/sglang/pull/42929)，`main @ 052feb72a8c9`，2026-10-24T12:36:41Z。

稳定ID：`github:sgl-project/sglang:pr:42929`。

### SGLang AMD deterministic all-reduce修大prefill错误与注册表泄漏

启用deterministic/on-policy allreduce后，超默认16MiB工作buffer的大输入原每次register，既返回各rank自己的输入，又永久耗rank_data slot，长跑可overflow。新路径按max_size切块重复用已注册buffer，保持每元素固定rank相加顺序；恰好等buffer容量也走正常unregistered-copy路径。每片多一次launch，作者2×MI355X大输入单次/4095次回归验证且测试父进程会传播worker失败；旧错结果不存在可比较的正确性能baseline。 **建议AMD on-policy/deterministic服务优先回归大prefill及重复大allreduce，验数值也验长跑注册资源稳定。**

源码阅读路线：`python/sglang/srt/distributed/device_communicators/custom_all_reduce.py::_deterministic_all_reduce_oversized、_all_reduce_impl`。

证据基线：[sgl-project/sglang #34543](https://github.com/sgl-project/sglang/pull/34543)，`main @ 656331295619`，2026-10-24T22:36:01Z。

稳定ID：`github:sgl-project/sglang:pr:34543`。

### SGLang ROCm activation/Rotary与Marlin舍入边界补齐

DSv4.1 low-ratio source只在DP padding确有多余行才裁切，并对Mxfp8Activation/Fp8GridActivation逐struct field裁，修MI35x decode graph不可下标错误；RoPE HIP fallback调用已有cos_sin_cache_as(query.dtype)，保留QSA用FP32主cache并复用cast副本，修Qwen3-Next FP8 capture启动。Marlin GPT-OSS式activation改用FP32 fused swiglu_oai_split写既有intermediate buffer，公式/clamp相同但舍入改变，作者完整GPT-OSS Marlin质量仍TODO。gfx1250另加triton/gluon选择与kernel gather opt-in：后者必须triton，且无expert_map、无router-weight-on-input才把token rows先quant一次让GEMM1 gather，其他回原Torch gather。正文+30%同时换recipe，不能独归这两个开关。 **建议ROCm部署优先验启动和graph捕获，Marlin使用者进一步跑真实模型质量；gfx1250开关按EP与路由权重组合分别测。**

源码阅读路线：`deepseek_v4_backend.py::_first_rows；rotary_embedding/base.py::forward_cuda；fused_marlin_moe.py::fused_marlin_moe；aiter_mxfp4_w4a8_moe.py::aiter_w4a8_gfx1250_forward`。

证据基线：[sgl-project/sglang #46151](https://github.com/sgl-project/sglang/pull/46151)，`main @ 55f23a8b5913`，2026-10-24T03:55:14Z；[sgl-project/sglang #46161](https://github.com/sgl-project/sglang/pull/46161)，`main @ 7671d16ce8eb`，2026-10-24T06:44:41Z；[sgl-project/sglang #46134](https://github.com/sgl-project/sglang/pull/46134)，`main @ bf1a2529538f`，2026-10-24T14:27:08Z；[sgl-project/sglang #45758](https://github.com/sgl-project/sglang/pull/45758)，`main @ efc984952021`，2026-10-24T06:58:42Z。

稳定ID：`github:sgl-project/sglang:pr:46151`；`github:sgl-project/sglang:pr:46161`；`github:sgl-project/sglang:pr:46134`；`github:sgl-project/sglang:pr:45758`。

### SGLang external kernel contract搬迁与OOT speculative接口

plugin kernel contract搬到kernels/plugin_kernels/<hardware>并登记KernelBackend.PLUGIN，SRT绑定留srt/kernel_plugins；旧srt.megakernels/monokernels无alias，外部插件必须兼容新import，Docker先把sgl-nvidia-mega pin到3333e4c修main启动。新platform.get_speculative_worker_cls可替换worker并负责拒绝不支持flag；使用内置EAGLE worker的平台可按draft_decode/draft_extend给graph runner，返回None则eager且无graph capability不调用。adaptive target-verify用平台已有runner；这些是接口接缝和fail-loud guard，不等于任意OOT设备kernel已具备或全部spec模式已验证。 **建议外部平台与kernel插件维护者深入读新contract并回归import/flag/graph fallback，按声明能力拒绝不支持组合。**

源码阅读路线：`python/sglang/kernels/spec.py::KernelBackend；srt/platforms/interface.py::get_speculative_worker_cls、get_speculative_graph_runner_cls；srt/speculative/spec_info.py::create_worker；eagle_worker_v2.py::_select_spec_graph_runner_cls`。

证据基线：[sgl-project/sglang #45389](https://github.com/sgl-project/sglang/pull/45389)，`main @ b86d294a980b`，2026-10-24T12:12:01Z；[sgl-project/sglang #45144](https://github.com/sgl-project/sglang/pull/45144)，`main @ af3f9e4c8f7a`，2026-10-24T04:24:59Z；[sgl-project/sglang #45145](https://github.com/sgl-project/sglang/pull/45145)，`main @ a7183a824680`，2026-10-24T04:28:56Z；[sgl-project/sglang #46030](https://github.com/sgl-project/sglang/pull/46030)，`main @ 49a2fbf81726`，2026-10-24T00:58:05Z。

稳定ID：`github:sgl-project/sglang:pr:45389`；`github:sgl-project/sglang:pr:45144`；`github:sgl-project/sglang:pr:45145`；`github:sgl-project/sglang:pr:46030`。

### SGLang DeepSeek Python prompt/output改用官方recipe，最终builtin路由与正文有冲突

DeepSeek-V4/V4.1 prompt现由固定deepseek-recipe0.2.0配合served fast tokenizer编码，删除两套本地checkpoint encoder；Python Chat Completions/Responses streaming及batch输出从token IDs喂StreamProcessor，tool arguments改为增量JSON片段，客户端应拼接后parse，语法/think-end仍依赖配置的parser。实际#46219 frozen _BUILDERS中dsv4/dsv41只有builtin→recipe builder；#45907正文的--chat-encoding-backend recipe及builtin不变描述过时，照用recipe值反而无builder报错。输出在匹配reasoning/tool parser至少一项配置时启用，另一项可None，混用其他parser拒绝；native parse endpoints也必须由served模型tokenizer处理，模型不匹配返回400。编码的developer/system、tool配对、special-token文本与effort规则改变，不是纯内部重构。此处只确认SGLang adapter交接与最终选择，不把外部recipe内部行为当已审代码，也不把Python结果套到Rust。 **建议DeepSeek API维护者优先重测effort、tool配对和流式JSON客户端；以最终builtin路由验证，继续观察Rust同格式闭合。**

源码阅读路线：`srt/entrypoints/openai/chat_encoding_adapters.py::_BUILDERS、native_text_parser；serving_deepseek_recipe.py::DeepSeekRecipeChatEncodingAdapter.__init__；parser/deepseek_recipe_output.py::DeepSeekRecipeOutputParser`。

证据基线：[sgl-project/sglang #45315](https://github.com/sgl-project/sglang/pull/45315)，`main @ e22a53085994`，2026-10-24T01:05:36Z；[sgl-project/sglang #45907](https://github.com/sgl-project/sglang/pull/45907)，`main @ a77998e15718`，2026-10-24T02:23:23Z；[sgl-project/sglang #46219](https://github.com/sgl-project/sglang/pull/46219)，`main @ 17d92ea671b4`，2026-10-24T03:17:38Z。

稳定ID：`github:sgl-project/sglang:pr:45315`；`github:sgl-project/sglang:pr:45907`；`github:sgl-project/sglang:pr:46219`。

### SGLang GLM5.3分段编码cache与MiniMax nested工具参数修复

GLM Python renderer直接交special-token分段，安全cut点之间一次encode_batch并按tokenizer做1M-token LRU（单entry≤128K），复用system/tools/history；token overlap、backend truncation/padding、特殊kwargs等不安全组合回旧encode。作者6608 token-id比较无差异；TTFT表仍有小prompt无收益和部分并发p50回退，不能把32.6×编码微测当整服务提速。MiniMax解析无schema容器时推断nested对象/array及literal类型，计入模型自行打开reasoning块的token；全模型工具schema不再注入strict=False，保留key order、root role和completion_tokens_details.reasoning_tokens等协议变化。MiniMax request显式禁thinking优先于server默认，需API兼容回归。 **建议agent服务考虑复现重复tools/history的冷暖TTFT，并检查schema key order、thinking优先级与reasoning usage兼容。**

源码阅读路线：`parallel_prompt_encode.py::_SegmentEncoder、encode_prompt_segments；parser/glm53_chat_template.py::render_glm53_segments；function_call/minimax_m3.py::_infer_untyped_container；managers/schedule_batch.py::update_self_opened_reasoning_tokens`。

证据基线：[sgl-project/sglang #45399](https://github.com/sgl-project/sglang/pull/45399)，`main @ fe1bd108372c`，2026-10-24T13:11:15Z；[sgl-project/sglang #46133](https://github.com/sgl-project/sglang/pull/46133)，`main @ d0f517168dff`，2026-10-24T14:11:12Z。

稳定ID：`github:sgl-project/sglang:pr:45399`；`github:sgl-project/sglang:pr:46133`。

### SGLang Rust前端补weight版本/media计数及admission媒体hooks

scheduler→Rust扩optional batch columns，终帧带weight_versions span、各帧非零media token counts，/generate HTTP/gRPC meta_info补齐；未结束文本常见帧仍不分配extras，Rust没有weight-update endpoint，前置weight_version就是启动值。MM processor可在admission check_media拒400、fetch_media在blocking pool和byte budget内处理自定义scheme；未知scheme由原worker decode500变prefetch400，server backend异常与panic保留500，abandoned fetch仍持permit到阻塞任务结束。另Rust prefill启动自有bootstrap类型的transfer backend服务，内置CommonKV registry仍由Rust API监听。all-base的#46215/#46216只是oss-sync分支合入，此处不单列成main事件。 **建议Rust-mode P/D和多模态集成者优先验证终帧版本span、byte budget及自定义媒体异常状态码。**

源码阅读路线：`srt/rust_server/server.py::push_generation、push_abort；rust/sglang-server/src/multi_modality/worker.rs::MmProcessor；api_server/core/prefetch.rs；srt/managers/disagg_service.py::start_disagg_service_for_rust_server`。

证据基线：[sgl-project/sglang #46205](https://github.com/sgl-project/sglang/pull/46205)，`main @ c5e734e437fa`，2026-10-25T00:50:06Z；[sgl-project/sglang #46206](https://github.com/sgl-project/sglang/pull/46206)，`main @ 4b4ae831bb35`，2026-10-24T05:20:53Z；[sgl-project/sglang #46074](https://github.com/sgl-project/sglang/pull/46074)，`main @ fab0e896ab91`，2026-10-24T05:22:48Z。

稳定ID：`github:sgl-project/sglang:pr:46205`；`github:sgl-project/sglang:pr:46206`；`github:sgl-project/sglang:pr:46074`。

### SGLang PD可观测性：区分转移预留KV，并纠正长prefill忙时

decode新增prealloc_full/swa_token_usage，按pool observer同一page对齐私有token计费/同容量分母统计transfer_queue，排除host-staged并扣tail已逐出的SWA头，因此可由总usage扣出运行batch份额。prefill忙时不再丢≥2s的真实chunk，显式记录pause区间并扣与forward区间的重叠；decode/target-verify cadence仍保留上限。它影响router估吞吐/排队时间的输入而非提升模型吞吐；prometheus和pause回归可验证，生产拓扑校验仍有必要。 **建议路由/容量规划团队深入读计费分母与pause时序，先校验监控曲线再据此调max-running-requests。**

源码阅读路线：`srt/disaggregation/decode.py::DecodePreallocQueue.pre_allocated_token_counts；srt/managers/scheduler.py::_record_step_counters、pause_generation、continue_generation`。

证据基线：[sgl-project/sglang #45970](https://github.com/sgl-project/sglang/pull/45970)，`main @ 1335d372f0a5`，2026-10-24T05:24:50Z；[sgl-project/sglang #46127](https://github.com/sgl-project/sglang/pull/46127)，`main @ a99eaf9a7629`，2026-10-24T05:13:20Z。

稳定ID：`github:sgl-project/sglang:pr:45970`；`github:sgl-project/sglang:pr:46127`。

### SGLang diffusion失败收敛：不能把取消、部分输出或I/O异常当完成

云上传取消先shield并等待真实SDK线程结束，image/video siblings全join后才清本地文件；仍可能等待SDK timeout/retry且不回滚远端对象。API在tensor/file存在前先看worker error，失败不能成completed；Vertex全batch先validate才建coroutine。safetensors读失败不再自动删文件或shared symlink目标blob，修复交operator；保存失败仍在finally进入replica内存metrics collective，避免健康后续请求卡死。realtime per-session lock保证header/binary payload相邻，下一chunk错误先等前一发送；scheduler逐message反序列化/IPC重建，坏消息不吞同poll好请求。作者有H200/Wan/SANA与CPU/Gloo故障注入，不能扩大成任意CUDA故障恢复、取消GPU工作或持久化高可用。 **建议多模态服务优先复现取消上传、保存失败后恢复和恶意/损坏单message隔离，尤其检查完成状态与真实资源收敛。**

源码阅读路线：`multimodal_gen/runtime/entrypoints/openai/storage.py::upload_file；openai/utils.py::process_generation_batch；loader/weight_utils.py::safetensors_weights_iterator；managers/gpu_worker.py::_finalize_output_batch；realtime/realtime_video_api.py::_send_output_and_log；managers/scheduler.py::recv_reqs`。

证据基线：[sgl-project/sglang #46068](https://github.com/sgl-project/sglang/pull/46068)，`main @ f09cec65e908`，2026-10-24T04:19:11Z；[sgl-project/sglang #46162](https://github.com/sgl-project/sglang/pull/46162)，`main @ 6f0ac70cf106`，2026-10-24T06:41:55Z；[sgl-project/sglang #46095](https://github.com/sgl-project/sglang/pull/46095)，`main @ cdab6c1bcc30`，2026-10-24T04:17:47Z；[sgl-project/sglang #46235](https://github.com/sgl-project/sglang/pull/46235)，`main @ 31b352b8ea1b`，2026-10-24T07:42:21Z；[sgl-project/sglang #46250](https://github.com/sgl-project/sglang/pull/46250)，`main @ 625399c7d2ff`，2026-10-24T13:35:19Z；[sgl-project/sglang #46157](https://github.com/sgl-project/sglang/pull/46157)，`main @ 638374c3893f`，2026-10-24T07:45:10Z。

稳定ID：`github:sgl-project/sglang:pr:46068`；`github:sgl-project/sglang:pr:46162`；`github:sgl-project/sglang:pr:46095`；`github:sgl-project/sglang:pr:46235`；`github:sgl-project/sglang:pr:46250`；`github:sgl-project/sglang:pr:46157`。

### SGLang LTX两阶段：BCG有限默认、CFG空闲rank转Ulysses、prompt KV只算一次

LTX2TwoStagePipeline在validated checkpoint、CUDA非共享host内存、server warmup、≥130GiB、resident DiT、batch1、无torch.compile/Cache-DiT/user LoRA/offload等条件才自动BCG，LTX2.5同窗加入白名单；其他shape需warmup配置，不能称所有diffusion默认graph。stage2无CFG时把已同输入的CFG ranks视为Ulysses group，只有够长token才启；没有新NCCL communicator。LTX2/2.3 resident第二DiT可premerge distilled LoRA，但占额外权重，user stage1 LoRA等保留原路；LTX2仅在无per-step cross-attention AdaLN、无layerwise-offload/Cache-DiT等情况下每loop准备prompt K/V并原地址刷新供图重放。LTX2.3 uniform sigma的[B,T] AdaLN降[B,1]及all-valid prompt mask移除受lossless质量门控，exact保留参考路；数学值不变而GEMM/attention舍入变，作者端到端PSNR仅约15–24dB，不能把lossless标签理解为视频byte-identical。各PR测量不同叠加栈，不相加作总提速。 **建议LTX服务考虑按受支持shape复现BCG收益，并把resident显存、quality=exact和lossless输出差异分别留档。**

源码阅读路线：`multimodal_gen/runtime/server_args/server_args.py::_breakable_cuda_graph_default；parallel_state.py::use_sequence_parallel_group；ltx_2/denoising_av.py::parallelism_type_for；models/dits/ltx_2.py::prepare_prompt_kv；ltx_2/denoising.py::_ltx23_token_timesteps`。

证据基线：[sgl-project/sglang #45977](https://github.com/sgl-project/sglang/pull/45977)，`main @ e66c9e981d57`，2026-10-24T01:22:06Z；[sgl-project/sglang #46239](https://github.com/sgl-project/sglang/pull/46239)，`main @ af7d941fc3b6`，2026-10-24T07:42:58Z；[sgl-project/sglang #45965](https://github.com/sgl-project/sglang/pull/45965)，`main @ 5b8955e70fc4`，2026-10-24T01:17:42Z；[sgl-project/sglang #46108](https://github.com/sgl-project/sglang/pull/46108)，`main @ a08bb0ec9eee`，2026-10-24T01:23:08Z；[sgl-project/sglang #46105](https://github.com/sgl-project/sglang/pull/46105)，`main @ f2c634961b9e`，2026-10-24T04:17:05Z；[sgl-project/sglang #46109](https://github.com/sgl-project/sglang/pull/46109)，`main @ 25ba24e9a7b7`，2026-10-24T03:00:15Z；[sgl-project/sglang #46106](https://github.com/sgl-project/sglang/pull/46106)，`main @ 386fa9ecfdfe`，2026-10-24T01:19:13Z。

稳定ID：`github:sgl-project/sglang:pr:45977`；`github:sgl-project/sglang:pr:46239`；`github:sgl-project/sglang:pr:45965`；`github:sgl-project/sglang:pr:46108`；`github:sgl-project/sglang:pr:46105`；`github:sgl-project/sglang:pr:46109`；`github:sgl-project/sglang:pr:46106`。

### SGLang diffusion驻留与host-sync：以保守容量规则减少反复搬运

auto模式在缓存safetensors header可完整计大小、每GPU serving precision权重预算可估、最少free GPU≥2×per-GPU权重时，让未显式指定的组件全驻留；量化/替换权重/overlay/缺本地header等不能估算就保旧策略。不是warmup实测memory planner，2×规则无法证明任意shape都不OOM。共享hot path移除device.tolist作RoPE cache key、复用shape-based metadata并新增禁止host sync的测试guard，model-specific补Qwen-image等不变常量/特征的缓存；只确认已读路径，不声称所有40文件均完全无sync。Blackwell额外认可≤128且8倍数的FA4 head_dim（Qwen-VL72/80），默认进dynamic cuDNN/FA4 fallback，显式fa进FA4，Hopper/SM12x不变。 **建议大显存多模态部署先核对估算权重与最差shape峰值，再考虑启用新auto结果；用host-sync guard和profiler确认目标热路径。**

源码阅读路线：`multimodal_gen/runtime/server_args/auto_tune.py::_fits_all_components_resident；loader/utils.py::checkpoint_component_sizes；layers/rotary_embedding/mrope.py；platforms/cuda.py::_resolve_flash_attention_backend_cls_str`。

证据基线：[sgl-project/sglang #45978](https://github.com/sgl-project/sglang/pull/45978)，`main @ 4e2d5b17ad6c`，2026-10-24T01:26:14Z；[sgl-project/sglang #46045](https://github.com/sgl-project/sglang/pull/46045)，`main @ 19212d19b5df`，2026-10-24T07:40:30Z；[sgl-project/sglang #46046](https://github.com/sgl-project/sglang/pull/46046)，`main @ c9f40557e677`，2026-10-24T13:33:28Z；[sgl-project/sglang #46102](https://github.com/sgl-project/sglang/pull/46102)，`main @ 2e6d87e2ca8c`，2026-10-24T01:20:33Z。

稳定ID：`github:sgl-project/sglang:pr:45978`；`github:sgl-project/sglang:pr:46045`；`github:sgl-project/sglang:pr:46046`；`github:sgl-project/sglang:pr:46102`。

<a id="extended"></a>

## 扩展开源工程与GPU/NPU软件栈

以下基线为主干、release或feature分支的实际合入/落地，均不等同公开release；性能/硬件验证来自作者，本轮未做GPU/NPU/TPU复现。已读取所述入口、关键补丁与测试，生成设备程序未逐个审计。72小时未报补录明确标注；未闭合候选逐ID保留。

### FlashInfer PCIe/RDMA Ulysses 主干落地：用 copy engine 与mlx5直接写最终布局

[#4876](https://github.com/flashinfer-ai/flashinfer/pull/4876)（main @ `0dccfbf43e2cd4191ed5d9e3d1987092df573dfa`，2026-10-24T04:54:11Z）。

视频/图像扩散的 Ulysses SP 在无 NVLink 单机上，原 SM 逐项远写会产生小 PCIe transaction。新增 `UlyssesCommunicator(backend="pcie")`：两卡选 CUDA P2P copy engine，四/八卡优先 mlx5 all-RDMA；scatter/gather 借2D copy或interleaved MKey直接落最终head布局。`allocate_output`事先集体注册，`max_bytes`替换原构造参数`max_elems`；`input_buffer()`还可免一段D2D拷贝。

作者在8×RTX PRO 5000、双NUMA4+4、Gen5x16、每卡400G mlx5，seq37888/H56/D128/BF16，对NCCL `all_to_all_single`+permute测到qkv层两卡2.01×、四卡1.97×、八卡RDMA1.48×；八卡P2P fallback只有0.30×，须看实际路由。单机world1/2/4/8；RDMA要求batch1、head-row pitch≤65535B且拒绝CUDA Graph，P2P可capture。auto需所有rank一致开启experimental opt-in，JIT仍依赖rdma-core。

`ulysses.py`/`ulysses_topology.py`、原生transport与epoch barrier补丁将RDMA native失败转sticky abort；Python参数拒绝仍依赖SPMD一致，P2P失败要求进程退出，不能解释成可恢复容错。全组仍需CUDA P2P/IPC来访问signal barrier，RDMA仅替换payload传输；十秒只约束RDMA completion等待，epoch rendezvous仍可无界。缺席/乱序rank需launcher终止，若abort/stream或verbs drain不能确认完成，close拒绝同步或卸载。**建议无NVLink扩散服务团队深读并在自己的PCIe/NUMA/NIC拓扑复现**，验收fallback和异常退出；本轮未运行多卡硬件。

### FlashInfer GDN CP训练：token留本rank，只交换每文档仿射状态

[#7169](https://github.com/flashinfer-ai/flashinfer/pull/7169)（main @ `195ccacadc3b9c826a492885c8e89a8b32041ac4`，2026-10-24T12:06:48Z）。

新增实验入口`gdn_train.py::chunk_gdn_kcp`，处理BF16 q/k/v/out、FP32 scalar g/beta，Q/K头与value头原生分组，D128，HV必须是H_qk的整数倍（HV % H_qk == 0）。`cake_kcp.py`先为fragment形成FP32 `(B,M^T)`摘要，all-to-all发到每个head slice的owner，按文档次序组合再返回各fragment的初始状态；token不跨rank搬移。支持连续、整体zigzag、逐文档zigzag+ragged及saved/checkpoint replay，ring仅允许每文档跨至多三rank且不保存transfer。

作者GB300/CUDA13.4、H_qk16/HV48、CP2/4/8/16，60行相对同layout FLA GDN CP报告forward几何均值2.84×，包含forward+checkpoint重算+backward与通信的模块训练步3.57×；例如241817token/6文档、CP4逐文档zigzag，训练步59.362→15.445ms。它是单GDN模块，不能外推整模型训练3.57×。GB200开头与表后汇总数字冲突，本条不引用该汇总。

已读入口、计划/校验、状态交换编排及分布式测试；生成设备kernel未逐个审计，README仍留有scaffolding字样，当前实现已具完整调用与autograd路径。注册SM100a/103a/107a；GB200/GB300有硬件表，SM107仅同源编译而无同等实机证据；没有兼容性保证，也不进入自动backend选择。**建议长上下文Qwen混合训练团队深读/复现**，同时核对FP64精度预算、halo、checkpoint保留内存和跨文档隔离。

### FlashInfer KDA CP：BF16x3半宽state walk替换TF32 hi/lo

[#7176](https://github.com/flashinfer-ai/flashinfer/pull/7176)（main @ `7bc25aaefe1e5bf16c8e35e5604c392c8086ef8a`，2026-10-24T22:20:38Z）。

`cake_kcp.py`的公开默认transfer walk改为`bf16x3_hw64`：FP32状态乘积将每个operand拆三份BF16，用Tensor Core与FP32 accumulation，两CTA各算64列；`M/B/state`仍是FP32存储。跨rank仿射状态规则、单卡训练路径、saved/replay backward不变；旧TF32与全宽BF16x3只留测试门。

作者GB300/CUDA13.0/NCCL2.30.7，在CP2/4单节点、CP8双节点、CP16四节点同NVL72域，两独立会话144行中89行两次都快>2%；B1范围0.9498–1.0868×，包含checkpoint重算与backward的B3仅0.9576–1.0233×。两行只在首会话退化，第二会话反向，不能写成所有CP配置稳定加速。舍入末位变化、仍满足该家族精度预算，运行间bitwise确定不等于与旧版逐bit相同。**建议KDA CP用户定向复现并记录完整B3**，不要把前向state walk收益加到昨日checkpoint优化数字。

### FlashInfer Frost：FP16、普通/块FP8与四位weight-only扩展，仍需显式选核

[#7153](https://github.com/flashinfer-ai/flashinfer/pull/7153)（main @ `2f55a9b9b6d505dfa1c1c95e5e65fc8bc0686f86`，2026-10-24T04:16:36Z）；[#7162](https://github.com/flashinfer-ai/flashinfer/pull/7162)（main @ `c7c821c2a763dc35d9996cee6bf119b3fc827ed3`，2026-10-24T06:38:44Z）；[#7166](https://github.com/flashinfer-ai/flashinfer/pull/7166)（main @ `bd7f082ae896e32d4044f0224bc5624c022ee17e`，2026-10-24T10:31:05Z）；[#7174](https://github.com/flashinfer-ai/flashinfer/pull/7174)（main @ `e415578a3adfaad61cf13f636305874e82742054`，2026-10-24T12:05:00Z）；[#7184](https://github.com/flashinfer-ai/flashinfer/pull/7184)（main @ `92ec5f0976d0340be067d9ccbbbcd376756eead6`，2026-10-24T15:10:13Z）；[#7186](https://github.com/flashinfer-ai/flashinfer/pull/7186)（main @ `f5d4db2c488d77331afa653c282d2e03c87ba517`，2026-10-24T16:51:33Z）；[#7189](https://github.com/flashinfer-ai/flashinfer/pull/7189)（main @ `a9e02c23923b724293c583961915b438cece32eb`，2026-10-24T18:41:58Z）；[#7194](https://github.com/flashinfer-ai/flashinfer/pull/7194)（main @ `94b5e266a966146b5da8cba4e70f9cea6260de5f`，2026-10-24T23:37:00Z）。

共享两段FMA pipeline把FC1激活/量化、FC2有序combine接入更多契约：FP16输入与独立BF16/FP16输出；per-tensor E4M3、E4M3 weight-only；DeepSeek E4M3与任意FP32 128×128 scale；NVFP4/MXFP4/有符号MXINT4 canonical packed权重的BF16/FP16激活。runner/API补丁保留dtype/scale身份、运行时scale与原始checkpoint字节。一般限T≤8、top-k≤32、H按128对齐；DeepSeek FP8的I也须128对齐，weight-only逻辑I可32对齐。不支持EP，量化activation的输出仍BF16。

新增PowLU与SM120 W8的FC1重复expert复用；Hopper/SM120另把同expert的BF16 routes组织为Tensor Core候选，窄限H2048/I512/K10、SwiGLU、T4/8，保留FMA默认。所有这些仍显式opt-in并由autotuner挑选，默认最大token仍2；不是仅按T自动切TC。

H100的小专家H2048/I512/E512/K10，per-tensor FP8 T1相对CUTLASS作者测得38.551→10.688µs（warm3.61×/cold2.55×），但T8/shared降到0.79×/0.85×；四位H100 MXFP4 T8/shared仅0.574×，B300较大MXINT4控制也输。展开FP4的BF16对照耗4倍权重storage，不能当原生四位基线；不同预激活/partial舍入需各自reference。这些是已预先选路的完整routed-expert GPU区间，排除router/shared expert/EP通信/host enqueue，不能当模型吞吐。**建议小batch MoE内核团队深读完整输赢表并按真实路由复现**；SM120 grouped BF16部分表仍待score修复后的重测。

### FlashInfer SM120 W4A16：八token direct-route扩展补齐capture scratch

[#7190](https://github.com/flashinfer-ai/flashinfer/pull/7190)（main @ `316faec37a4706a2ecc7d46265087f8190b40baa`，2026-10-24T18:58:23Z）；[#7195](https://github.com/flashinfer-ai/flashinfer/pull/7195)（main @ `5e4555094c329cee6d035791eec1f73e96980f59`，2026-10-24T22:39:25Z）；[#7168](https://github.com/flashinfer-ai/flashinfer/pull/7168)（main @ `5d699c5cabe55d116b7f2a0831c518045abba793`，2026-10-24T10:28:08Z）。

BF16/NVFP4 direct Tensor Core decode仅在SM120、H≤2048/I≤512/top-k≤10把上限延至T8；大专家、SM121与FP16契约各保留原边界。随后修复多个route重复命中小expert bank时的scratch低估：`plan_w4a16_buffers`与动态workspace planner取packed容量和`min(routed_rows,8*topk)*block_size`的最大值，route-pack布局不变，仍受每SM scratch cap约束。旧eager临时分配掩盖了错误，capture会失败。

作者RTX PRO 6000 Blackwell、H2048/I512/E512/K10/T8、SwiGLU，增强native基线后，之前Frost约2.6×的warm优势收窄到1.11×，cold近持平；不是又新增2.6×。无约束延长TC曾让H7168/I2048大专家退化，因此继续拒绝。**建议已开SM120四位MoE graph的团队优先复现T8/E16重复route回归**，并重做native/Frost选核。

独立cuTile FP8/MX sorted MoE在大expert张量触发64bit indexing时，原shape给出int64 K循环上界却配int32 induction variable，编译器拒绝。#7168只在wide-index入口传model K的compile-time scalar，普通索引仍按tensor shape，地址/指针宽度不降级；强制wide的小分配与graph重绑测试核对逐bit。应先修这个native编译问题再比较Frost，不能把baseline编译失败算FMA加速。

### FlashInfer autotuner：cache身份、严格replay与分布式profiling前置协议

[#6959](https://github.com/flashinfer-ai/flashinfer/pull/6959)（main @ `03bc3f9c445b1bd97c40a6eb395aef1239853776`，2026-10-24T03:08:06Z）；[#7148](https://github.com/flashinfer-ai/flashinfer/pull/7148)（main @ `5d9c8d100c6a495df75d5d31cbdaf7fb0fbde578`，2026-10-24T03:26:38Z）；[#7150](https://github.com/flashinfer-ai/flashinfer/pull/7150)（main @ `a35cb6a536ab23a6f0c02bcf075485f4d1fd0a0f`，2026-10-24T05:06:11Z）；[#7152](https://github.com/flashinfer-ai/flashinfer/pull/7152)（main @ `39bbb50f7edc9d43f21d187063b45c4905cbd6d4`，2026-10-24T04:12:56Z）；[#7161](https://github.com/flashinfer-ai/flashinfer/pull/7161)（main @ `1bd9a70079e2d3e3936577ab90313ee1342df8ec`，2026-10-24T05:49:03Z）。

`autotune_v2(cache_scope="context")`可在一次显式身份中调优/lookup而不覆盖process ambient，排除legacy/bundled winner；捕获与调优必须同identity，CUDA Graph replay本身没有Python查找。新增`ReplayCoverage`区分真实hit、fallback miss与skip，`strict_replay=True`在identified context、单rank replay中对缺失/不兼容/未知profiling provenance直接失败，不重新profile或发布替代winner。hit仅说明选择命中，不证明backend最终执行了该kernel。

分布式`choose_one/rank_tactics`在profile前用固定大小phase/selection fingerprint确认op/runner次序、skip、profile数、计时策略和有序candidate一致，普通准备失败集体传播，OOM继续协调fallback；rank-local shape/cache根可不同。它无法救缺席rank、坏CUDA context或任意runner内部collective，仍需PG timeout；不调优的serving不增加collective。

计时另可opt-in热graph逐call interval，默认仍amortized，无backend自动启用；delay helper JIT前移到candidate warmup前，免首次候选被初始化扰动。**建议维护eager/graph两套选核cache的框架团队深读并复现miss与跨rank乱序测试**，这些是测量/身份正确性能力，不能据此宣称普遍吞吐加速。

### FlashInfer稀疏attention：Hopper原生GQA，QSA decode默认切Cake

[#7167](https://github.com/flashinfer-ai/flashinfer/pull/7167)（main @ `9a575dd5d8c9fa461bb5affa14dd03a2ab9545fa`，2026-10-24T14:05:17Z）；[#7147](https://github.com/flashinfer-ai/flashinfer/pull/7147)（main @ `2e1d7b0a6cd62df2719b40a6ca27694e7e4f7299`，2026-10-24T03:21:36Z）。

VSA Hopper显式`backend="cake"`原先靠重复K/V、scale和mask服务GQA，现在按KV头共享stream与稀疏pattern，query头h映射h//g；BF16与Sage-E4M3均支持g=2/4/8。契约仍SM90a、HND、D128、非因果64-token块、每query块选1–64 KV块，不支持LSE/位置编码/softcap。H10080GB/CUDA13.4的284 shape相对旧MHA复制路线（复制成本未计时）kernel-time geomean1.186×，不是整模型或连复制在内的收益；auto仍不选VSA Cake。

另一个NVFP4 KV、E4M3 block scale、BF16 query、D256的QSA paged decode API把默认改auto，SM80–90使用Cake，显式`default`仍可回reference。A100/H100、24×2头、top-k2051、1–128行的whole-call GPU event表，A1001.285–3.213×、H1001.353–1.991×；只覆盖该契约，精度在已有容差内而非逐bit。**建议稀疏KV部署分别复现这两个API**，不要把VSA显式选择与QSA默认变化混为一项。

### FlashInfer cuDNN：只重建已存GEMM winner，dense KV decode补LSE布局

[#6618](https://github.com/flashinfer-ai/flashinfer/pull/6618)（main @ `17555cab5819ea1315cdd3a8b8721304e8f2e894`，2026-10-24T02:45:37Z）；[#6619](https://github.com/flashinfer-ai/flashinfer/pull/6619)（main @ `dbc95c3e3001fe875518632ae4683d9a4f8ca0f3`，2026-10-24T03:11:14Z）。

GEMM加载保存的(engine,knobs)时，不再把它抹成tactic0并重建整套portfolio；无profiling graph只建单plan，刚调优过则weakref复用其graph而不额外延寿，覆盖BF16/FP8/MXFP8/FP4/BF16-FP4。作者FE1.31/cuDNN9.27/Torch2.14.1，M8/64/256、N1024/K2048的三形状fresh-process合计，B2002.072→0.675s、SM1201.589→0.191s；是单次组件启动观察，排除import与输入准备，无serving吞吐结论。

`cudnn/decode.py`在`block_tables=None`初始化optional descriptor，不传paged-only上界；LSE对graph绑定(batch,heads,query_tokens)view，同时保持公开token-major contiguous buffer。新增1/3 query token、不同KV长度及graph replay测试。**建议有保存tactic启动或dense cuDNN decode的部署定向复现**，关注plan身份和多query LSE，非新attention算法。

### FlashInfer RMSNorm与mHC训练核：延续普通launch与shape专用调度

[#7200](https://github.com/flashinfer-ai/flashinfer/pull/7200)（main @ `03f82fddadaebed69879b4e7d28f3df38c4e28d2`，2026-10-25T00:11:36Z）；[#7198](https://github.com/flashinfer-ai/flashinfer/pull/7198)（main @ `26d0d5dc906ca3738d02d796a7c71faf6952f690`，2026-10-24T23:37:11Z）。

RMSNorm继续使用昨日#7131普通chunk-claim main+PDL dweight reduce双核，通过uniform descriptor、H512 row-group issuer与SM107 bulk-copy程序优化；host launch manifest补充对应dynamic shared memory。BF16输入/weight、FP32 rstd/dweight，H512/1536/2048/4096/6144。作者同GPU同进程CUDA13.4.92、冷L2的CUPTI active interval，对#7131在T≥14928报告GB300 H5121.18–1.27×，其他H/残差型收益更窄；dweight跨grid/run确定，但chunk求和次序相对旧版改变。并未重证昨日常驻旁路kernel共存矩阵。

mHC保持4 residual stream/H4096、BF16边界与FP32 mapping/Sinkhorn/参数梯度、256B/token saved statistics，workspace和程序选择新增架构/SM count输入。对#6897，GB300两会话T1024–65536：forward1.02–1.05×、backward1.05–1.17×、checkpoint1.06–1.15×；仅这一mHC配置，backward一阶。生成设备程序未逐个审计。**建议对应训练团队复现单层计时、梯度与checkpoint内存，并保留小T/多stream验收**。

### TensorRT-LLM release/1.3：解耦传输晚到请求会失败应答，接收结果先验token与区间

[#20459](https://github.com/NVIDIA/TensorRT-LLM/pull/20459)（release/1.3 @ `a71d30a11abd85171abea7701cc67fc2f0b38b6c`，2026-10-24T10:57:46Z）；[#20723](https://github.com/NVIDIA/TensorRT-LLM/pull/20723)（release/1.3 @ `a34e839b7637791aef3620c2f9e4a3075ae55fd5`，2026-10-24T09:15:10Z）。

Python V2 context send原先把generation排队时间计入60秒transfer timeout；session清理后，晚到REQUEST_DATA被当generation-first早到请求搁置，可能让generation所有slot/KV永久占住。现在退休ID保留600秒并回FAILED，非attention-DP的120秒stale请求也回失败；所有receiver rank就绪后才起transfer时钟，等待peer最多3×timeout。已逻辑失败的receive等待所有writer终帧，最多2×timeout，后续cancel也不能提前放页。

**2×是计时上界，默认无physical ownership时不证明停顿RDMA已停止。** attention-DP下过600秒的旧请求仍可能静默drop，idle receive timeout本身未改；新增等待会延长过载时context KV占用。作者2×H100 NVL A/B覆盖Qwen3.5-4B，GPU未覆盖ctx TP>1/gen TP4与H100 SXM，其他waiver未一并清除。

配套接收校验冻结request的目标页区间和writer bounce容量，result token绑定预期writer，copy/完成前核对地址、正size、总容量与source base；native protocol升v2，双方必须同格式。**建议运行该release分支解耦服务的团队深读状态机并优先复现过载/重试/多writer失败**；两PR都不等于main或公开release发布。

### TensorRT-LLM DSv4 Rubin：Q投影融合接线只覆盖decode

[#19961](https://github.com/NVIDIA/TensorRT-LLM/pull/19961)（main @ `d7e0ecac925dc8f3fddfc2bff5ef5ab1d9149fbf`，2026-10-24T17:27:26Z）。

SM107 FP8 block-scaled q_b的CuTe路径接入GEMM+RMSNorm+RoPE+quant融合，generation只再处理KV侧RoPE；context/mixed batch因DSv4 metadata不提供cu_q_seqlens仍走unfused。O-LoRA两段投影在attention custom op内写单一mutable output，Rubin o_b用MXFP8 inplace GEMM。DeepGEMM JIT改回sm_100f家族target，规避bundled CUTLASS不识别sm_107f导致descriptor prefetch assert。

`TRTLLM_DISABLE_DSV4_Q_B_GEMM_NORM_ROPE_FUSION=1`可关闭Q融合。虽标题含DSpark，**DSpark自身的block-FP8投影并未启用SM107 CuTe kernel**；新增的是DSpark accuracy/acceptance与dummy slot回归，不能写成DSpark Rubin全链加速。**建议Rubin DSv4部署定向读attention接线和架构门禁并复现**，无可外推完整模型倍率。

### TensorRT-LLM Staircase：GLM-5.2 NVFP4的Rubin TP16专用target

[#20517](https://github.com/NVIDIA/TensorRT-LLM/pull/20517)（main @ `e707e62fc4d3f89aaabf81c006f89788487d75b6`，2026-10-24T02:28:25Z）。

`routing.py`仅匹配SM(10,7)、78层/H6144/256专家/q_lora2048、world=TP=MoE-TP16、EP1/无ADP/PP1/CP1、FP8 latent KV，线性one-model MTP-Eagle或无spec。auto不匹配回built-in，require则报错。target自有decoder/loader/MTP，把DSA key按16rank分片，精确histogram top-k合并，MNNVL Lamport hop融合residual/RMSNorm；单请求verify≤8行走快路径，更大batch回该target自有模块。

作者20 Rubin GPU的解耦c1（context DEP4+gen TP16）、K7、强制acceptance2.91的3600秒性能运行完成335请求/零error；这里强制acceptance不能当真实模型收益。真实acceptance GSM8K1319题严格0.9659、灵活0.9682、K7平均5.53 token/step。main旧版该部署warmup即失败，所以没有可比旧版吞吐倍率；Rubin GPU测试未进CI，local4卡414passed不等于20卡完整回归。**建议该特定部署团队深读/复现**，不推广到其他GLM形状、并行度、packed FP8-DS-MLA KV或动态树。

### TensorRT-LLM Helix：feature分支按bounded history chunk重建MLA KV

[#20924](https://github.com/NVIDIA/TensorRT-LLM/pull/20924)（feat/helix-agg @ `fb8c8fec68ce26ce75b3e7ec43607f65dd265ab6`，2026-10-24T02:02:09Z）。

`feat/helix-agg`中`_prepare_helix_mla_chunks`将全局历史chunk映成各CP rank本地读取区间；先本地load/FP8反量化到query dtype，再variable-size CP all-gather，零贡献rank也进入同一collective序列。CUDA reorder恢复request/token次序，并分出compressed KV与共享k_pe，持久cache继续分片。

仅dense MLA cached context、SM100+且chunked prefill；nonchunked显式拒绝，依赖同feature分支admission/owner-only write/allocation。旧base曾过GB300测试，**当前GitHub合并head的native重建/GPU回归、完整Kimi K3 checkpoint aggregate serving/prefix reuse/perf仍pending**。**建议研究CP prefill的工程师深读并观察整合**，不当成main可直接部署能力。

### TensorRT-LLM MiniMax多模态：保住已渲染prompt、像素尺度与spec原始token

[#20626](https://github.com/NVIDIA/TensorRT-LLM/pull/20626)（main @ `a0cee56d3a6eaea26206ddfbb28973e66e390f67`，2026-10-24T01:13:26Z）。

TextPrompt新增chat_template_applied，已渲染输入直接用，免图像marker二次套模板；先校marker/media计数。float[0,1]媒体关闭HF再次1/255 rescale，uint8/PIL保持原契约，同modality混用时归一到uint8；video token bracket按temporal patch计。纯文本不再强制加载image processor，首次media才懒加载。

multimodal prefill融合embedding后，MTP/Eagle/DSpark/DFlash通过orig_input_ids保住token；prefix reuse或chunk边界无placeholder行时跳vision，只按当前request窗口slice embedding。测试覆盖完全/部分cache、image span中断和mixed请求，不能仅凭CPU契约测试推断完整模型质量。**建议MiniMax VL serving团队优先回归真实图片/视频、prompt_token_ids与KV复用/spec组合**。

### TensorRT-LLM Gemma4：encoder按padded工作量计预算，避免八图OOM

[#20663](https://github.com/NVIDIA/TensorRT-LLM/pull/20663)（main @ `abb94ab8fc1c93ee377a5137c9b36466e2ae3664`，2026-10-24T05:38:41Z）。

Gemma4 image/video/audio现在报告encoder item的padded输入成本与有效输出行，最大item与dummy profiling帮助KV预算为一次encoder iteration预留空间；视频按声明frame数整段切分。google/gemma-4-31B-it、H10080GB、max_soft_tokens1120的8×2000²图片原先一次约80k patch，现按预算每轮至多两图。

chunked prefill+KV manager V2复用既有按chunk消费/释放item输出，八图可默认准入；无chunked prefill必须整请求输出常驻，超预算会admission拒绝。Gemma4 Unified的encoder-free投影仍禁item scheduling/profiling；executor单次forward异常后退出属于另一个问题，未在此修。**建议多图Gemma4服务优先复现预算与chunk边界**，不是所有多模态OOM统一解决。

### TensorRT-LLM MoE：补pre-quant smoothing，eager不再无条件分aux stream

[#20823](https://github.com/NVIDIA/TensorRT-LLM/pull/20823)（main @ `ecae3c97e2b646580415518a45a72bc5a5a67075`，2026-10-24T02:40:06Z）；[#20808](https://github.com/NVIDIA/TensorRT-LLM/pull/20808)（main @ `032eb456ecb145ba10f9c9dc043cfaa8254d657c`，2026-10-24T11:31:21Z）；[#20774](https://github.com/NVIDIA/TensorRT-LLM/pull/20774)（main @ `9666d7427a8c0b75a4159e70b594e23f70e25942`，2026-10-24T03:43:37Z）。

MegaMoE CuteDSL为NVFP4_AWQ/ARC先乘fc31_act_scale再量化，免raw activation对上按smoothed输入标定的scale/权重。最终patch也把streaming loader的w1/w3 pre_quant_scale成对传入、校一致并落到计算设备；缺一或非gated会拒绝，plain NVFP4无scale保持原路。PR正文还写其他CuteDSL leaves另修，最终diff含这些leaves相关整理，本条不把全部leaf正确性视为已证；作者B200反向控制旧版79.6%元素超容差，新MegaMoE通过。

CuteDSL async quantization仅在do_multi_stream或capture时fork，eager回主stream：GB300/B300 eager原先受event/stream/record_stream host开销拖慢10–20%，保留capture decode overlap。另把request的LoRA checkpoint格式沿enqueue→broadcast→LlmRequest→PEFT cache miss reload带到底，无per-request格式才回server默认。**建议量化MoE与混用HF/NeMo LoRA的服务分别回归数值、eager host时间和逐rank reload**，勿把stream fork一概视作GPU优化。

### AITER：gfx950四位GEMM避LDS溢出，gfx1250新增8-token页FP8 indexer

[#6874](https://github.com/ROCm/aiter/pull/6874)（main @ `26f4f1fb2de14d122c71f5fed7afb97a5c97cade`，2026-10-24T14:38:55Z）；[#6863](https://github.com/ROCm/aiter/pull/6863)（main @ `ddf072fc3b3ab9e65ff3ddd745d5dd33c9775dc3`，2026-10-24T02:56:10Z）。

Qwen3.8-27B-MXFP4 TP1的N96/K5120投影，M129–256与>4096桶原用3stage需214976B LDS，超MI355X163840B，profile/graph/prefill会直接失败。config把两桶降2stage，M32/128另用较快DEFAULT tile；MI355X/Triton3.8/PyTorch2.13测试40pass/40Gluon skip，M1288.24→6.84µs，大M原本不launch因此不能算相对旧版加速倍率。split-K求和次序允许约1e-3相对差。

FlyDSL gfx1250 MQA logits用wave32 WMMA16×16×128直接读原packed preshuffled FP8 KV，16token tile可横跨两个8token page；split按16column边界避免重复写，≥2GiB cache切64bit offset。限D128/H为16倍数、next_n≤8且next_n×H≤256、Preshuffle=True，支持page8/16/32/64/128/256；generic导出按arch懒dispatch。作者端到端GSM8K还联合另一个GEMM PR与ATOM，不能把结果单归此kernel。**建议MI355X该模型或gfx1250 DSv4.1用户定向复现启动/大cache与ragged MTP**，无通用ROCm倍率。

### vLLM-Ascend：MRV2恢复要消费preempt标记，MLA短prefill也必须等KV load

[#19693](https://github.com/vllm-project/vllm-ascend/pull/19693)（main @ `6197fec5c3520e75128795652f8145988f61fbf6`，2026-10-24T07:46:16Z）；[#19567](https://github.com/vllm-project/vllm-ascend/pull/19567)（releases/v0.30.0 @ `1b06b6cc60f8a726f4f9038dab5a45aa82785414`，2026-10-24T07:46:17Z）；[#19635](https://github.com/vllm-project/vllm-ascend/pull/19635)（releases/v0.30.0 @ `83de6c6d245d570657fd64861e2c0b3df9bf73e1`，2026-10-24T10:16:40Z）。

MRV2通过scheduled_new_reqs恢复，带完整block table；旧preempt标记没清，下一RUNNING delta却走preempt处理并覆盖完整表、重置save进度。main及v0.30.0分支现在在MRV2新请求路径discard标记，并撤销昨日窗口状态的save suppression workaround。MRV1保持cached-request消费标记。main只做UT/lifecycle复用，NPU集成/性能未测，不能据revert认为任意suffix-table保存都已证明正确。

独立v0.30.0 MLA修复将connector wait移到PROLOG_V3、generic preprocess及MLAPO所有cache访问之前，decode attention边界record start；避免短prefill被归decode后没等外部KV即读/写。作者full Kimi K3、2×Atlas A3/32NPU、TP16 PP2、eager LM-only/no-spec，实际恢复768external token的3次请求与cold64输出token/logprob全等；GPQA只跑103/198就停止，83正确且25截断，**不是完整80.58%准确率通过**，无unpatched全模型基线，main未上NPU。**建议AscendStore/MRV2与layerwise MLA用户优先复现preempt→resume→running和decode-only外部命中**，分清分支与硬件证据。

### PyTorch主干实际落地：NCCL2 capture stream池跟进程存活，symm-mem归还VA

[#199439](https://github.com/pytorch/pytorch/pull/199439)（main @ `39d5b15fac656ecaa70c9e2e8b1411d694093972`，2026-10-24T16:19:11Z）；[#201842](https://github.com/pytorch/pytorch/pull/201842)（main @ `8952da6ca22a81e6feabde2edc8149e82e84aa60`，2026-10-24T14:46:59Z）。

这两PR的GitHub merged_at仍null，已打开默认分支实际commit核对落地，不能只靠PR merged搜索判断。#199439最终代码位于`c10d/nccl2/ProcessGroupNCCLUtils.cpp`：按device/priority保进程期stream pool，capture中复用同capture-id的stream，否则取idle或新建，并在解锁前先join current stream；避免32固定stream被嵌套torch.cond耗尽/串入其他capture，recordStream引用也不再晚于PG销毁而访问已死stream。**限定实际nccl2代码**，不能把正文旧legacy NCCL测试描述当最终patch。pool扫描令N重嵌套有O(N²)query代价，NCCL_GRAPH_MIXING_SUPPORT=0仍是测试前提。

`CUDASymmetricMemory.cu`则在unmap、multicast unbind之后补cuMemAddressFree/hipMemAddressFree，免每local/peer/multicast映射泄漏一段保留VA；失败warn不throw，allocation失败路径另案。CUDA两卡测试可重新reserve旧base，ROCm/free错误路径未测。**建议分布式图控制流及长时对称内存用户优先深读/复现生命周期**，释放VA不等于减少同量物理显存，也使旧悬空VA可能被新分配复用。

### PyTorch编译正确性：encoder输出stride与torch.full标量dtype对齐eager

[#194044](https://github.com/pytorch/pytorch/pull/194044)（main @ `d1dd2542fbb73c9354a0d1f4ed08803281370b05`，2026-10-24T14:19:04Z）；[#199433](https://github.com/pytorch/pytorch/pull/199433)（main @ `18d8d0974472b71069c8a6d50f1262915e63cd20`，2026-10-24T01:32:32Z）。

实际main提交将非空`meta__transformer_encoder_layer_fwd`改为contiguous empty输出，匹配真实layer_norm/addmm分配；旧empty_like保留非连续dense输入stride，Inductor可能assert，关size_asserts还会让消费者按错布局读。零元素分支仍clone保stride，不能一刀切全部输入。

Dynamo分解动态Tensor scalar的torch.full为empty+fill_时，dtype缺省/None且out缺省/None才按Python数值类别推bool、int64、默认float或默认complex；显式dtype不覆盖，不把Tensor本身精度直接复制。**建议使用非连续encoder fallback、动态整型/bool fill的compile训练定向复现**，属于静默数值/布局修复，没有速度倍率。

### PyTorch CUDA index_copy：eligible宽行复用128bit scatter copy

[#201814](https://github.com/pytorch/pytorch/pull/201814)（main @ `199499c48eb5f76febc1fb3dd5c59a6facfd449b`，2026-10-24T05:56:41Z）。

实际main的CUDA dim0 index_copy在连续index、至少两索引、slice≥256B且未开deterministic algorithms时交给scatter；对齐/TensorIterator条件进一步限制向量化，128bit加载/存储，32thread/行、8行/block、最多8block/SM，少行宽slice可分到多个block。单index/短slice保留原路；unsigned byte loop避免接近2GiB增量溢出。

GB200 fp32/int64、65536 source行、随机唯一目标index、warm CUDA Graph，256/512/1024B slice为10.106→5.152、18.629→6.808、50.338→22.720µs；排除Python enqueue，非任意重复index、任意维度/布局或训练step收益。**建议KV搬移与宽row scatter/copy热点团队按真实shape复现**，同时核验deterministic开关与大slice。

### PyTorch MPS attention迁到Metal：奇异head维度也用padded/chunked核

[#201745](https://github.com/pytorch/pytorch/pull/201745)（main @ `f6585f7f189d6d457fcec520b399a39140a8ba8a`，2026-10-24T07:57:50Z）。

实际main扩大Metal推理SDPA覆盖：decode支持nearest-wider tile、实际head≤512、Q/V维度不同与FP32 mask；更宽走prefill，prefill≤256用padded tile，宽QK分块累积、V/O按128列slab。padding是线程内跳过越界列，未分配扩展tensor，返回attention权重为空，meta同步更新。

作者M5 Pro/M3 Ultra表同时列本PR及两个后续stack，不能把after-all列算本次落地：例如M5 Pro Q/K/V为1×8×1024×512，PR2仅0.29×，后续PR才1.18×；M3 Ultra该行0.33×。因此保留大head prefill退化，不用最大shape倍率概括整体。**建议Apple训练/推理用户按PR2列复现真实shape及精度**；本条不宣称训练反向也全面改成Metal。

### Triton warp specialization：scan区域与if内异步MMA获得正确建模

[#12472](https://github.com/triton-lang/triton/pull/12472)（main @ `ddfbfbe1ad647ec21e4299764cb1f971eb2664f2`，2026-10-24T22:36:16Z）；[#12482](https://github.com/triton-lang/triton/pull/12482)（main @ `1ed999d6b21d7bd9e89b830f9eeb7e5d8ba8934b`，2026-10-24T22:38:01Z）；[#12480](https://github.com/triton-lang/triton/pull/12480)（main @ `753b56ecb704cfc712daebcae554fbf78ca7343a`，2026-10-24T15:55:54Z）。

`tt.scan`现在和reduce一样给combine region传播partition，为tuple operand/result建独立port与partition-output，clone时保axis/reverse及region，修persistent tcgen05 epilogue的cumsum/associative_scan编译assert；不额外增加同步。NVWS另在遍历callback收集真实MMAv5 consumer，避免把scf.if祖先强cast成MMA。前者正文记录B200精确比对/ConSan，后者只有lit与cuda:100编译证据、未上Blackwell。

AMD target feature将gfx120x加入IEEE maximum/minimum指令支持表，lit改查LLVM min/max lowering，非新增通用GPU能力。**建议写scan/分支融合kernel的编译器团队深读并复现最小IR**，不把编译解除崩溃直接写成性能提升。

### FlashAttention SM100：CLC双CTA必须共享同一logical tile

[#3034](https://github.com/Dao-AILab/flash-attention/pull/3034)（main @ `98c1cee0ed59307833841cdbf567fe336ceec13a`，2026-10-24T15:10:02Z）。

2CTA cluster的两CTA本应共同算一个pair-level tile，旧CLC按CTA拆坐标导致两边读不同Q行，2CTA MMA混到不同tile。forward各scheduler现在统一logical work tile，`work_q_block=(work_m*q_stage+stage)*cta_group_size+cta_rank`决定row；mask、sink、LSE和O store都按同一映射，static persistent在cluster模式直接接收cluster数，backward/MLA scheduler不变。

GB300、K-only varlen packed GQA hd128，FA_CLC=1旧max error0.39，新输出与非CLC bitwise相等；新增覆盖1/2CTA×stage1/2、尾cluster、NaN-prefilled输出与LSE。CLC仍opt-in，hd256双CTA排除仍在。**建议开CLC的FlashAttention用户优先回归正确性**，不是改变默认scheduler或新增吞吐承诺。

### Ascend PyTorch：pipelined autotune分离编译/benchmark，FlexAttention LSE换底进核

[955b98d4](https://github.com/Ascend/pytorch/commit/955b98d43ba13cc824bef86a065deb2352f2783d)（master @ `955b98d43ba13cc824bef86a065deb2352f2783d`，2026-10-24T03:50:02Z）；[0c3f5564](https://github.com/Ascend/pytorch/commit/0c3f556423d8a5aa1fec77d93ee17ccb6327aef1)（master @ `0c3f556423d8a5aa1fec77d93ee17ccb6327aef1`，2026-10-24T08:02:50Z）；[f72a49f1](https://github.com/Ascend/pytorch/commit/f72a49f161b0e5ec001d105c4f1b46dc0bd5c0a4)（master @ `f72a49f161b0e5ec001d105c4f1b46dc0bd5c0a4`，2026-10-24T07:58:41Z）。

GitHub镜像master实际commit适配上游AsyncAutotuner/PrecompileThreadPool：启pipelined且无CATLASS候选时，外部kernel可先benchmark，同时预编译Triton，再等编译结果筛掉失败候选并合并timing；CATLASS保持原线程池，pipelined timing拒绝hint override。NPU subprocess初始化以npu tensor触发device初始化。模板config随后强制calling-thread串行预编译，避免嵌套线程池；vector多config继续并行。只有源码/新增host路由测试，未有编译加速实测或故障恢复保证。

FlexAttention只把LSE自然底→log2乘法融入forward store，backward消费者乘ln2恢复；位置表fill、dK/dV清零、sum和梯度dtype/layout收尾走局部ATen fallback，opaque empty_strided+copy_免未初始化分配被消费者错误inline。**建议NPU Inductor/FlexAttention用户深读并复现首次编译与梯度**，不能把标题fuse理解成整个反向全融合。

### TransformerEngine回溯：compiled Linear可复用FP8 weight cache

[#3505](https://github.com/NVIDIA/TransformerEngine/pull/3505)（main @ `9ea05cf2e8dc275d2fd3c88d39062ba20947b77d`，2026-10-22T17:04:03Z）。

10-22主干合入的未报补录。`Linear`在compiled first-microbatch不给functional custom op传旧workspace，而接回新workspace；随后microbatch复用，保持每minibatch只量化一次weight，免功能式op原地改input。GPU tensor控制的skip_fp8_weight_update仍显式fallback，fuse_wgrad_accumulation=True仍不支持。

作者RTX5880 Ada、PyTorch2.14内部构建、同一prebuilt TE core/extensions，18pass/12skip/1XPASS，含交错/延后backward和真实CUDA Graph replay。reduce-overhead要求每完整minibatch一次cudagraph_mark_step_begin，不能每microbatch重置；C++未重建，其他GPU/recipe/timing未验证。**建议有梯度累积与compile的FP8训练团队深读/复现**，不把cache复用能力写成已测训练加速。

### TransformerEngine回溯：mask-map容量上界与GroupedTensor stream寿命

[#3347](https://github.com/NVIDIA/TransformerEngine/pull/3347)（main @ `c9c92255465b4518380561b1bbb0aebb177c0b6d`，2026-10-22T23:33:15Z）；[#3701](https://github.com/NVIDIA/TransformerEngine/pull/3701)（main @ `2199b7166417a00c76d092b6d79337053f2184cb`，2026-10-23T18:59:50Z）。

`moe_permute`的Triton mask-map以前按无界cumsum生成目的row，output却只分num_out_tokens，可能越界写。现pass2把row_id≥capacity变-1，让后续计数、permute/unpermute及反向沿既有drop语义跳过；JAX调用仍传2³¹−1 no-drop sentinel，不等于JAX token_dispatch也落实capacity。L40S/Triton3.7.1复现compute-sanitizer48次invalid write→0；原套件16个FP8版本/头文件相关失败仍在，非全绿。

10-23 `GroupedTensor.__torch_dispatch__`给shape不变的identity view复制元数据并共享backing，requires_grad先false交autograd补differentiable-view；record_stream传播到每个CUDA backing tensor，wrapper自身没有allocation。测试检查alias/_base、dense梯度和data/scale/amax/offset等buffer被记录。**建议MoE dispatch、分布式优化器及多stream训练优先回归容量溢出与wrapper生命周期**；没有任意reshape无拷贝或任意并发正确性承诺。

### TransformerEngine回溯：JAX mHC VJP与cuBLASMp NVFP4有明确后端门槛

[#3680](https://github.com/NVIDIA/TransformerEngine/pull/3680)（main @ `53d1a3c71a8f54cbfcdbc5e9f548eba040e42523`，2026-10-21T17:38:52Z）；[#3634](https://github.com/NVIDIA/TransformerEngine/pull/3634)（main @ `6b5cd7e2e43cee6da5dff9b9f86bc2ec05d7b1ad`，2026-10-21T07:14:08Z）。

JAX mHC把projection/scale/Sinkhorn/aggregate/expand-combine的Triton调用接成custom_vjp，输入stream轴为(...,C,n)，与FlashInfer固定[T,4,4096]的接口布局不同。共同Triton Sinkhorn重算反向在写history后补tl.debug_barrier，避免其他thread先读；新增JAX单卡/分布式测试。正文称预期E2E收益，本轮没有可引用计时表，不能推倍率。

cuBLASMp comm-GEMM新增NVFP4：compile-time与实际加载runtime都须≥0.11.0，α由global scale在stream上计算并使用device pointer mode；只TN布局，拒绝row-scaled NVFP4、未swizzle scale和量化输出D。backward保留wgrad所需columnwise数据，all-gather仍只rowwise。**建议JAX mHC与NVFP4通信/GEMM重叠团队深读并分别复现**；这里验证的是TE接线/guard，cuBLASMp内部执行与通用训练收益不在本轮证据内。

### MLX-LM回溯：Flint模型适配full/SWA缓存与可求导MoE回退

[#1998](https://github.com/ml-explore/mlx-lm/pull/1998)（main @ `4cfc6478473b0668ede3eb0cc93f2a9181a15052`，2026-10-24T00:10:48Z）。

10-24T00:10Z合入，位于主窗起点前的72小时补录。新增Flint decoder按layer_types区分full/sliding attention、各自产mask/cache；只有sliding层使用RoPE，attention带per-head softmax offset，按配置选择dense MLP或带shared expert的SparseMoeBlock。推理有Metal时用expert_routing；training或无Metal用sigmoid→top-k→归一化权重的MLX路径，因Metal路由kernel没有VJP。shard方法切attention/experts并all-sum输出。

作者只说明用RecursiveSI/flint BF16 checkpoint测试；本轮已读实现和模型测试，未有完整质量/长上下文/分布式实测矩阵或性能表。品牌Recursive不等于此实现引入递归计算：当前源码按layer列表逐层调用。**建议Apple端Flint用户观察并复现cache、训练梯度与shard**，不由模型名推断新推理算法或端到端倍率。

### TorchTPU：大量参数tuple化打通融合训练编译，保持donation契约

[#5497](https://github.com/google-pytorch/torch_tpu/pull/5497)（main @ `03b9f4b742465fa6d0656900e567cf64d41ca041`，2026-10-24T00:58:46Z）。

TPU程序entry参数数有上限，4096参数的fused AdamW约20485实参超过v7的16313限制。统一`CompileAndLoadMlirModule`在TPU main参数数>2000时设parameter_is_tupled_arguments，把显式argument layouts包成tuple；StableHLO仍flat，调用方仍逐buffer传参，由PJRT在launch组tuple，donor对应tuple element。eager/compile/builder路径共用判断，compile key加tuple flag，binary cache版本5→6避免误复用旧executable。

tuple需要设备地址表和间接读取，有固定launch成本，所以未让所有小eager程序都tuple化；非TPU平台不套此门槛。新增大参数compile、in-place/out donation和key测试，本轮未运行TPU，也不将“保alias”泛化成无任何runtime开销。**建议大参数fused optimizer训练团队深读/复现编译上限与实际buffer donation**。

### TorchTPU：合并交错checkpoint区域，linear保symbolic shape

[#5792](https://github.com/google-pytorch/torch_tpu/pull/5792)（main @ `a058a6e6c2ca6bb4b6fe0f2aced07ce95f02d13c`，2026-10-24T07:39:03Z）；[#5810](https://github.com/google-pytorch/torch_tpu/pull/5810)（main @ `f4b2f5e6adcf086ce394b4a7ef65a581def8ca9d`，2026-10-24T06:18:45Z）。

FX backward里不同checkpoint region的remat节点可能交错，旧pass直接ValueError。新`_merge_interleaved_regions`按首次节点排序、维护已见最大remat位置，把重叠region并入更早region，由单个barrier gate；避免PyTorch2.11/2.14 checkpoint compile失败。2.11的fused fwd-loss-bwd未带autograd_backward标签时，memory-savings测试改skip，不能说这一组合已验证节省显存。

TPU linear autograd校验改用ConstTensor/sym_size/sym_sizes，bias以is_expandable_to做关系guard；免把out_features及broadcast bias的batch/seq每一维具体化，动态长度复用graph。contracting维相等、bias可广播仍必须guard，eager错误语义保持。**建议checkpoint+动态shape TPU训练优先复现重编译计数、barrier与峰值内存**；没有性能倍率或任意图重排保证。

### XLA GPU scan fusion：默认关，长scan按row数量择路

[#50730](https://github.com/openxla/xla/pull/50730)（main @ `a078bf45b0fe3f40a0704c5a6597f5cd802e5ea7`，2026-10-24T06:58:58Z）。

`xla_gpu_experimental_enable_triton_scan`默认false；启用后在AssociativeScanRewriter前把合格kScan标`__triton`，具体tile留给后续fusion/autotuner。只允许最后一维，避免tiled实现其他维算错；scan维≥32768时至少64row才取Triton，否则保hierarchical scan，且GPU须属于Triton支持架构。

作者H100/B200的FP32 cumsum+elementwise epilogue网格（B1–256、N1024–262144）中，短N分别1.05–2.9×/1.06–2.8×；长N只在B≥64采纳。top-p mask的softmax+cumsum+compare/select，K64/128可融合单kernel，H1003.3–4.5→1.3–1.5µs、B2005.9–8.4→2.5–3.1µs。是小sampling primitive，未测LLM服务吞吐。**建议编译sampling或长scan的团队按flag复现，并保最后维/row门禁**。

### XLA IFRT Proxy：取消从no-op变RPC，完成回复仍不等于设备停机

[#51545](https://github.com/openxla/xla/pull/51545)（main @ `a2a1d0dd83bbf169b28dde168bd5e94e7de5ede0`，2026-10-24T04:10:29Z）。

Proxy protocol v26携带每次execution的cancellation handle；client CancelExecution传非OK status，server在锁内extract对应backend handle并调用底层Client::CancelExecution，execution status完成时清理记录。老协议或非proxy handle客户端仍直接return；server对未知/已完成handle回复空response，RPC失败只异步log。

这补上MPMD reshard等client取消到backend的通路，**并不由RPC返回证明kernel、通信或资源已全部静止**；实际settle取决于后端取消与execution status。已读client/server/协议与测试，本轮未做设备取消故障注入。**建议使用IFRT Proxy的分布式执行团队深读并复现cancel/complete竞争与旧协议降级**。

### XLA/JAX数值与布局边界：显式FMA进IR，layout constraint变assert

[#51457](https://github.com/openxla/xla/pull/51457)（main @ `46f7c9918bd9770883ef89f558ad14496270cc63`，2026-10-24T01:20:24Z）；[#41959](https://github.com/jax-ml/jax/pull/41959)（main @ `4673fd4780dc48109fbd6374a5dce0b48c8298f8`，2026-10-24T23:25:12Z）；[#41955](https://github.com/jax-ml/jax/pull/41955)（main @ `c90d7ebb038a8aaa33bcf3ace1909c1533588a78`，2026-10-24T16:36:39Z）。

XLA加入三元kFma/mhlo.fma与StableHLO双向转换、shape/verify/layout/sharding/cost(每元素2FLOPs)支持；reference evaluator的f32/f64用std::fma，窄浮点经double与float两级Round-To-Odd避免double rounding。此PR建立IR/求值语义，不是已启用所有GPU/Triton的fused codegen，也不能当默认乘加融合策略改变。

JAX `with_layout_constraint`遇类型已有explicit layout时只assert相等并返原对象，改layout须显式relayout；AutoLayout仍bind constraint。另因下游numerics breakage回退#41914，logaddexp/logaddexp2 JVP恢复_replace_inf路径；只有回退代码证据，没有“全部数值问题修完”的结论。**建议写自定义低精度HLO/显式layout与log域导数的团队深读并定向回归**，区分IR契约、物理布局转换和回退。

<a id="evaluation"></a>

## 评估与可靠性

### METR补录监控系统实测：应监控流量、实际完成监控与当前覆盖率必须分开

**METR/Redwood；2026-10-22公开，补录；实验为7月21日至8月11日，对象是7月21日及以前的历史系统。** [评估方公开原文及限定说明](https://metr.org/blog/2026-10-22-red-teaming-openai-agent-monitoring/)。研究人员获内部代码、监控提示与配置访问权限，发现异步监控的覆盖和触发边界。其估计较强模型的内部Responses API agent流量75–94%属于“应送监控”的范围，内部Codex子集为99.2%；**这些不是最终完成监控的比例，也不包含所有内部推理服务**。当时只在交还用户时触发、且未监控compaction，持续行动即可推迟检查；作者估计5–7月约一成应监控Codex流量因此漏检，但日志不足使数值不确定；该compaction问题已由OpenAI在演练前独立发现，并在演练中修复。

工程启示是按消息数/时间触发、交叉核对推理与监控账本、核验两侧序列化上下文一致，并隔离agent工具环境与监控控制面。漏洞公开信息有限；61页为内部报告，38页删节版也仅向评估机构领导共享，不能称公开61页审计全文。OpenAI称影响生产防护的漏洞已修复，METR**未复核**后续修复与扩容，不能将历史结果当作今天覆盖率，也不能把修复声明当独立验证。**建议深入阅读并复现自己的监控完整性测试**；本次仅核原始披露，未接触内部系统。

稳定ID：`metr:red-teaming-openai-agent-monitoring:2026-10-22`。

### 训练数据纠错：Dolci披露具体污染行，避免把n-gram命中全部当实质泄漏

**本轮2026-10-25发现；原始绝对发布时间未核，不声称主窗首发。** Ai2 HF组织页面将作者`abhishekraok`标为“Ai2 org”；[32B讨论#2](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-32B/discussions/2)和[7B讨论#4](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-7B/discussions/4)披露相同审计，固定数据快照分别`7668c638`/`83f618e2`。两页是同一结果，合并记录。

报告先做13-gram筛查再手审：MATH的5000测试题中205题对应744训练行，其中13行/8题完全一致，其余改了数字或措辞；ifeval_mt为245行/197题，来自WildChat轮次；IFBench/MMLU各3题。GPQA Diamond 6行/3题及main/extended更大列表**仅n-gram重叠，未手审**；三个GPQA集合嵌套，不能相加。IFEval本体未发现污染，Precise IF共有约束措辞不算泄漏。

披露者称从MATH评测集移除这205题后，其SFT模型报告分数最多变0.19点；这不是删训练行后重训的去污染消融，原文未给重训证据，而污染子集相对难度level匹配clean题高5.7–8.2点；后者仅205题且噪声大，不能外推所有模型/基准，也不能拿两数互相抵消。页面给1001行ID并按`id`+`messages`hash确认7B/32B清单一致；本轮只审计披露正文，**未独立重扫Parquet、未验证训练运行或重算分数**。应优先复核所用数据快照，按行去污或在clean测试子集重评；不把数据仓全量内容视为当天新数据发布。

### OP-Print：常规membership分数接近随机，不代表自蒸馏没有留下可检测痕迹

**`2610.31616v1，2026-10-22T17:41:04Z`，昨日待办补录。** 灰盒沿同一rollout比较“普通题目”与“附参考解的privileged prompt”分布差，黑盒截断生成并多次restart，用续写文本一致性做代理。信号集中在不确定的纠正位置，因此平均likelihood/entropy可能掩盖它。

实验为**OpenThoughts数学500member/500held-out、Qwen3-1.7B/4B/8B、固定privileged初始teacher、3epoch、8×A6000**，不是生产规模训练隐私审计。Table 1的4B JSD灰盒max AUC0.611 vs最佳传统基线0.517；黑盒AUC0.575，TPR@5%FPR0.089，且RC-Rate该列0.104更高。因此结果是可检测的弱泄漏信号，不能说可可靠识别任意单样本。best accuracy通过全ROC阈值扫描取得，也不是部署前固定阈值性能。

黑盒成本随full generation、prefix比例和restart数增加；附录把其联系明确写成假设模型而非一般可识别性定理。**建议把重启稳定性纳入自蒸馏隐私评估，同时报告query成本、低FPR和多训练种子。** [正文 §3–4、Tables 1–3、Appendices A/B](https://arxiv.org/html/2610.31616v1)

### 自奖励道德论证：更稳定、更果断，并没有同步改善判断有效性

**`2610.30349v1，2026-10-21T19:08:39Z`，回溯补录。** Gemma-4 E4B对每prompt八回答做双顺序round-robin自判，56次judge；顺序翻转当tie，用Rao–Kupper强度估计、长度惩罚与RLOO advantage训练可见答案，thinking policy gradient被mask而KL仍约束全序列。

§6/Figure 5：156文学困境、两epoch后，MoralChoice高歧义顺序翻转14.3%→6.6%，但ELEPHANT迎合45.3%→44.0%基本不变，用户框架接受5.2%→7.5%；MORABLES准确率56.5%→53.4%（差-3.1点、item bootstrap95%区间[-4.8,-1.5]）。减少顺序敏感不能替代外部有效性测量。

每配置只训一次，区间反映题目而非训练随机性；无固定外部judge训练对照，不能把回归唯一归因“自判”；judges只在人类论证上验证，未对policy输出做人工质量评分。Z3只验证抽取出的布尔冲突，不证明道德结论正确。**建议优先阅读这组负面结果，在奖励设计里把一致性、迎合与任务真值分开测。** [正文 §3–7、Figure 5、Limitations](https://arxiv.org/html/2610.30349v1)

### CausalCoT：隔离父节点上下文增加可干预性，代价是错误计划不能再靠全局提示补救

**`2610.31639v1，2026-10-22T17:52:56Z`，回溯补录。** 先生成原子子问题DAG，再按拓扑执行；root可读原问题，后续只读自己的子问题与直接parent答案。该约束删除显式旁路，但不揭示模型内部计算，也不保证计划本身语义正确。

附录H.6固定同一DAG只改context：**GPT-5.4 Mini、MedQA+GPQA平均**Answer Change从隔离58.72降到NoTrim37.07/NoTrim-Full37.50，standardCoT40.95；这是上下文隔离贡献，不只“拆问题”的收益。主Table 1三benchmark平均准确率CoT68.96、CausalCoT66.36；Pass@3提高也消耗多次采样。另50例MedQA/GPT-5.4mini计时CoT50call/1.31min，CausalCoT237call/3.43min，属于作者API环境，不能按硬件通用外推。

局部错误由GPT-5.4注入/评分，作者审计中编辑有效率约73–82%，更高answer-change仍不是全部意义上的推理faithfulness。弱Gemma模型的Predictable Change未全面领先；graph遗漏、格式失败与摘要丢信息仍会失败。**建议小规模复现固定图的context消融与干预测试，保留准确率/调用数/总时延。** [正文 §3–5、Tables 1/2、Appendices H.6–H.11](https://arxiv.org/html/2610.31639v1)

<a id="coverage"></a>

## 实际覆盖、补录与更正

### 核心项目覆盖

六仓共枚举737个“72小时回溯＋本轮增量”all-base merged PR（主窗118：Megatron1、vLLM25、SGLang80、TorchTitan12；verl/slime0）；默认分支694 commits；254个公开Release对象，主窗及overlap均0新published对象。TorchTitan release/0.4升0.4.0rc4/rc5与8个分支合入不等于正式GitHub Release，最新公开仍v0.3.0。verl/slime本主窗未发现新merged/release，slime overlap #2489已报不重收。

118新合入中106以main为base，8以release/0.4为base，SGLang4个其他base（oss-sync-*、rust-server-render-via-processor、xpu/flux2-catfree-out-proj）不可直接冒充main；默认commit中已包含的最终汇总以实际commit基线解释。已保存102个新候选完整file-patch返回，正文只收实际读过关键实现的机制组；其余大patch/视频kernel/仅有清单候选在coverage待办，不宣称全量语义审核。737候选有206已报、413为old overlap未报候选，后者也不因本轮inventory完成而关闭。

旧依赖vLLM #61667/#61978/#60149/#63232、SGLang #43299/#45580、verl #8061全部仍open；#60149本窗head有更新但未合入。MoRIIO READ与UMBP修复不关闭独立P2P晚到WRITE生命周期，Megatron旧NVFP4 refit刷新待办继续保留。没有GPU/NPU/CPU模型复现实验。

### 论文、模型与学术来源覆盖

- arXiv七类列表仍为10-23公告：1152次出现、829唯一ID、312 replacement ID。精确主窗投稿API返回0；72h投稿元数据898条。`lastUpdatedDate`查询在返回feed标题被改成`submittedDate`，与投稿查询同898项，**修订全集扫描未完成，0返回也不等于主窗无新论文**。
- 另做pretraining/reasoning/inference/quantization/reinforcement learning/evaluation/language model七主题API与中英文网页主题检索，覆盖固定机构名单外的论文与SenseNova；主题feed分别95/171/259/35/83/627/302条，均低于单页上限。元数据发现与全文技术确认分开记录。
- HF 41机构模型/数据×created/modified全量枚举受网络超时阻断，本轮没有可靠新增总数；已中断长批量，不用“0候选”冒充覆盖。web实际读NVIDIA/Qwen/Ai2/moonshot/SenseNova组织入口，少量tree/config可经点击恢复，但官方API和固定revision入口仍不稳定。Cosmos-H卡重定向到10-20的Cosmos3-H-Dreams，原Cosmos-H为7-23旧发布；本轮只完成卡级筛选，不当新主窗模型，也未完成其权重/实现审计。
- OpenReview API403；TMLR列表TLS超时；ACL/PMLR首页可读，但未逐volume读取。Google EMNLP2026会议页作为补充发现入口，会期10-24至29不当论文发布日期；accepted目录未逐篇审完。
- 原1228论文发现队列与594修订队列、模型103/数据25候选和更早历史缺口均保留。LoRA Superposition已读但实验仅七个CLIP ViT-B/16视觉分类backbone，不推广成LLM通用adapter；FusionEval的abs与正文v1均写9-30，和10月ID不一致，暂不作本窗论文；CertMoE正文已读到smoothed sentence routing与induced-subnetwork，但形式证明如何覆盖实际token路由仍待专审，不将其认证宣传当已核保证。

证据原文/hash、请求成功/失败、稳定ID版本、完整待审队列见本轮`papers/source-checks.json`、`papers/events.json`、`papers/manifest.json`；本轮未推进全域完整语义覆盖水位。

### 扩展工程覆盖

24仓主窗all-base merged PR为123项，前置72小时回溯1200项；默认分支commit共1404项。PR搜索与实际commit分别检查，PyTorch等bot直接落地不依赖merged_at判断。其余仓release集合翻页至结束；FlashAttention每页100条的大页请求返回`Transport closed`，仅重试前五条，**该release集合未查完**。

已取87份PR元数据、91页changed files，纳入67个稳定事件；1700个候选仍待语义审核。部分大生成文件缺patch，已记录逐文件缺口；开关、入口、编排和测试核对不能替代完整设备kernel审计。固定分支、合成接受率、计时上界和负面性能行均保留，不把feature/release合入写作main正式能力。

### 官方研究与硬件来源覆盖

本轮对50个已有官网/API入口保存请求与文本快照，结合中英文主题检索补充METR、StartLux、SenseNova、信通院与FlagOS线索。OpenAI研究概览、Anthropic研究/新闻、Google DeepMind与Google Research、Meta、微软、NVIDIA、DeepSeek、Kimi、智谱、MiniMax、Seed、LongCat、小米等均有实际检查；入口可读不等于旗下全部报告、发布渠道和版本已覆盖。NVIDIA/AMD可见最新技术文章仍为10-23，已报内容不重复；不能据此推断厂商没有其他发布。

硬件与软件栈已读NVIDIA/AMD技术入口、Intel Gaudi1.24.2、Google TPU生产更新、AWS Neuron2.33及NKI组件说明。TPU页面页脚10-13更新时间不等于新增事件（最新带日期的生产条目仍6-01）。华为公告仍为动态壳，CANN/HCCL/libtpu历史组件缺口继续保留。没有把旧说明换版本号当新进展。

原始HTTP中OpenAI/Microsoft/Ai2返回403、Meta/xAI等超时；web回退恢复其中研究/新闻入口，Mistral仍超时，RecursiveSI仍TLS失败。Qwen/Tencent/Baidu/Step入口及部分研究索引只有动态壳，阿里云旧入口是内容级404，均不记为扫描成功。HF API的连接故障同时由两条研究路径复核，不能以少量可读模型卡替代41机构全量清单。

信通院词元服务监测报告目前只获得官方检索摘要，未打开完整监测方法，不将汇总TPS/TTFT作为可比性能条目；留待核并发、长度、地域、模型版本与样本量。FlagOS2.2新特性与官方发布计划页已读，属于更早发布线索，不把本轮二手转述当10-24新版本。VoxCPM2也由官方更新日志确认是4月旧发布，未作为当天新模型收录。

### 本轮补录、版本纠偏和续查

论文十篇、RoboRSI、Prism与METR均保留原始日期和补录属性；Dolci按本轮发现记录，绝对首发时间未核。Kimi未来权重日期、同窗MegaGate启用后撤回、非main分支、PR正文/最终patch差异均在对应条目显式说明。这些是本轮证据边界说明，不代表此前日报一定发布过相反结论。DeepSpeed #8838/#8782与TRL #7769同SHA已在10-22详报，本轮复核后去重，不再作为新补录。

此前42条待办原样保留，本轮新增候选、来源失败与未覆盖区间追加到`.monitor/state.json`，各来源进度分别记录。下一轮从2026-10-25T01:00:32Z继续并回溯72小时，同时优先处理待传文件和原有未完成区间；全域完整覆盖水位仍为空。原始证据、稳定ID/版本、逐来源检查、排除理由与候选队列保存在本地`.monitor/research-2026-10-25/`。本地正文保存与GitHub同步分账，只有远端内容及提交回读一致后才确认上传成功。
