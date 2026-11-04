# LLM 技术动态日报 · 2026-11-04

> 时区：Asia/Shanghai（UTC+8）。增量扫描窗口：**2026-11-03 08:58:48 → 2026-11-04 09:05:04**；72小时回溯：**自2026-10-31 08:58:48起**；更早的重要漏收单列原日期补录。实际生成时间：**2026-11-04T09:32:03+08:00**。发布日期、提交/合入、工件修订和发现分别标注，只有日期的来源不推断上线时刻。
>
> 已读取memory、持久化state/sources、28篇历史归档及版本事件用于去重，重点回读上一期；GitHub main起始提交 `acbced2eb14f98492f837c95589ee4ad8824761d`，28篇历史与索引共29个本地blob逐项匹配远端，无待补传。历史报道不替代本轮一手核查。仅维护llm-monitor与tracker/llm-training-daily，未改llm-knowledge wiki。
>
> **模型/性能数字均为来源报告；本轮没有独立复训、运行设备基准、调用收费模型API或做集群复现。** 交叉审区分了最终代码与早期PR正文、主干与支线、预发行与正式版。来源仍有访问与未审区间；事件发现截止点不是完整语义覆盖水位。

## 本次重要变化

- **先读NorthCUA与Hydrogen的报告和限制。** 前者把GUI经验生产、长任务SFT和联合RL串起，后者披露科学后训练的DSA/CP、Muon状态和续rollout工程。NorthCUA成绩尚未绑定精确发布checkpoint，Hydrogen成本含价格代理，均需先做小范围复核。
- **优化器不能只按预训练loss选。** 新补录的Muon微调前沿、OrthMuon与weight-decay控制实验，以及Sonar-MoE裁剪基线，提示把学习—遗忘、约束/状态精度及完整step成本一起比较；不是否定Muon或宣布所有配置反转。
- **训推一致性开始贯穿dtype、量化与导出。** verl NVFP4改per-token QAT且有接口迁移；TorchTitan统一precision合同，TorchStore少CPU整份副本却可能增加暂停；FlashInfer KDA/温度与DeepSpeed确定性均有明确支持矩阵。
- **KV与混合状态的完成边界仍是采用门槛。** vLLM扩展HiSparse、重载与NVFP4 stride修复，SGLang修投机采样/DCP与回滚，Ascend INT8 KPool当天撤回；TensorRT-LLM受限context-first保护不等于普遍解决迟到DMA，LMCache也需保留未完成transfer映射。
- **推理优化要分清局部收益与服务负载。** MXCode/整数microscaling、TPU FlyWheel、AMD Mono与多种投机/调度机制值得读；合成接受率、不同资源、量化状态误差和短请求退步都会改变实际收益。
- **正确性评估有更可执行的入口。** StateBench测动态状态而非静态needle；MIC用冻结小架构与真实usage回放；FlashProve是受限整数证明研究，110token/s不是已证明response生成速度。parser fallback、harness演化和领域数据删样本也有重要负结果。

目录：[模型与工件](#models) · [研究论文与方法](#papers) · [核心开源](#core) · [扩展工程](#extended) · [工程验证](#reliability-engineering) · [硬件协同](#hardware) · [覆盖、补录与续查](#coverage)


<a id="models"></a>

## 模型与技术报告

### NorthCUA：把经验采集、长任务 SFT 与桌面/Web RL 连成闭环；报告成绩尚未绑定到发布 checkpoint

**机构 / 时间 / 版本：** 上海奇骥智峰 NorthCUA Team；报告 [arXiv:2611.02102v1](https://arxiv.org/abs/2611.02102v1) 首次提交 **11-01 18:16:47 UTC，补录**。Pro 的 `Publish verified release` 提交为 11-01，Nano / Mini 为 11-02；11-03 三仓更新的是 README。固定 [Nano@317dd35f](https://huggingface.co/beipo-ai/NorthCUA-Nano/tree/317dd35f56527900d1c0fb4f344107f181598b07)、[Mini@ca2b95a4](https://huggingface.co/beipo-ai/NorthCUA-Mini/tree/ca2b95a4018a00c585cbef04d932fda3342ee32c)、[Pro@b5d64ab9](https://huggingface.co/beipo-ai/NorthCUA-Pro/tree/b5d64ab98d811775cd5a297a5d14df0df35b39b9)；[Harness@fb5f9e27](https://github.com/beipo-ai/NorthCUA/tree/fb5f9e2725c7cfe268fa0c2dc40f44cc132c12f4)。

**具体增量。** 不是从零预训练新骨干，而是在 Qwen3.5-9B、35B-A3B、397B-A17B 上训练计算机使用策略。报告的有用之处是经验生产与训练边界：Harness 保留完整轨迹和待检查的现场，模型上下文则用有界截图窗口、压缩与独立笔记维持长任务；Experience Flywheel 同时扩展任务意图、可执行环境和 verifier，对难题用经验证的 skill 与环境里程碑继续搜索；随后依次做广覆盖 GUI mid-training、完成结果经脚本检查的长任务 SFT、桌面/Web 联合 GRPO。报告给出的来源规模为 >10 万条轨迹、>400 万交互步、约4万 grounding 样本；长 SFT 约3,000轨迹/31万步；RL 准备池约1,500桌面任务+500 Web任务。它们是来源池数量，不能当作每个型号的实际训练消费量。[原始报告 §3–5、Appendix D–E](https://arxiv.org/html/2611.02102v1#S5)。

**关键证据。** 最有说服力的是同型号、相同任务/接口/预算下的阶段对照：Pro 从长 SFT 到 RL，OSWorld-Verified **85.9→88.6%**，WebArena-Verified **69.6→73.9%**。前者采用361任务（排除8个额外账户配置的 Google Drive 任务）、后者812任务，均100步上限；OSWorld 2.0 为108任务、500步上限，Pro 的 **66.8%是部分得分，完整完成率为30.6%**。Mini 的部分得分46.3→50.1%，完整完成率仍18.5%，不能把部分进步写成完成更多任务。Grounding 指标来自 SFT checkpoint；OSWorld-G 取三次完整运行中最好一次，ScreenSpot-Pro 则单次预测，二者不是统一 pass@1。[报告 Tables 6、12–13、Appendix I](https://arxiv.org/html/2611.02102v1#S6.SS6)。

**系统与时延的边界。** NorthCUA-Ultrafast 在 **pre-RL Nano** 上加4.21M选择头，或头+LoRA共47.49M可训练参数；有限的点击/目标选择走直接分类，需填文本等动作则重新走完整生成。812任务成功率 **64.2→60.6%**；**616.8→266.8秒**的2.3倍任务时间变化，仅来自计时快照中共同成功且日志完整的固定320任务，排除了失败任务，也不含环境准备/清理/打分。每副本一张GPU，但未给出可据以复现的统一GPU型号，且基线与混合路径的采样设置、并发与浏览器部署并不完全相同。另一个9B工程profile的97.2%是28批次**图像张量字节减少**，排除了其他字段和传输开销，不是端到端训练提速。[§6.6、6.8、Appendix E.7](https://arxiv.org/html/2611.02102v1#S6.SS8)。

**开源与限制。** 权重与 Harness 为 Apache-2.0，三档卡称约18.84/70.23/793.63GB，配置上下文262,144，发布标识为 r13 checkpoint975、无MTP；这些不是显存总需求。**模型卡明确说研究成绩尚未对应到这一精确导出 checkpoint**，Ultrafast 机制也没有随这些权重获得验证。Harness 仓只发布执行部分，Flywheel、训练数据与完整训练流水线不是该仓库的开源内容；其依赖锁在Transformers4.x，而模型导出记录5.20.2，项目要求把模型serving另放独立环境。还需注意版本呈现差异：卡仍引用“121页”及旧的部分待测描述，本轮原始 PDF 实读为123页，Table13包含三型号的 OSWorld2.0 结果；本文按固定 v1 报告和固定卡分别陈述，不替作者消除差异。本轮未加载模型或复跑基准。

**建议：深入阅读，并考虑先复现 Nano + Harness。** 优先检验训练时 actor 输入、终局 verifier 和分段权重如何保持一致，再测发布 checkpoint 的实际完成率。长任务训练者可直接借鉴“原始现场与完整执行记录留存、训练上下文单独裁剪”的接口设计；采用前先校准模型卡要求的 `normalized_1000` 与 Harness 的坐标配置。

### Hydrogen-1.0：科学领域全模型后训练，同时披露 DSA、Muon 状态与可续 rollout 的工程边界

**机构 / 时间 / 版本：** Deep Principle；**补录**。官方[技术博客](https://blog.deepprinciple.com/introducing-hydrogen-1.0/)未标精确发布日期；HF仓建立于10-31，三个版本最后更新均为11-02。固定 [BF16@b71dd62f](https://huggingface.co/deep-principle/Hydrogen-1.0/tree/b71dd62f26fa459ca548117c11c6ad567af2faba)、[NVFP4@a96c2b9d](https://huggingface.co/deep-principle/Hydrogen-1.0-NVFP4/tree/a96c2b9d359396b625a38e848c29d3d0ae5daf1e)、[MXFP4@5344d941](https://huggingface.co/deep-principle/Hydrogen-1.0-MXFP4/tree/5344d941f372950f8a688bc13f2841a5f7b92417)。本轮首次发现，不能按二手新闻日期写成11-03首发。

**具体增量。** 在 GLM-5.3 上做科学领域 SFT+RL，数据类别包含化学/材料/生命科学知识、研究场景、实验室全流程记录与完整发现轨迹；失败候选、观察与下一步选择也构成学习材料。更值得工程读者看的，是作者对全模型 MoE 后训练的具体披露：packed 样本保留完整序列并按全局受监督 token 数归一化；DSA跨CP分片Top-k显式加入全局因果offset和样本边界，重计算重放稀疏索引与MoE routing；CUDA把FP32 masters/momentum放CPU、逐矩阵上GPU更新，ROCm按rank确定所有权并在forward/backward期间把所属Muon动量放磁盘；ROCm checkpoint逐记录写入，恢复gate/up到同一分配的不同切片。RL共用GPU池交替生成与训练，未完回答留前缀到后续轮次，逐token保留真实行为策略logprob，完整prompt group才进训练；导出到BF16/分块FP8之前，先做**零梯度更新**对照以分离布局/量化漂移与训练收益。以上是**作者的系统披露**，本轮没有公共实现可逐项验算，也不表示CUDA与ROCm在同一个异构训练集群混跑。[官方正文“Training / Infrastructure”](https://blog.deepprinciple.com/introducing-hydrogen-1.0/)。

**证据与限制。** 当前Figure1数据给出 FSR max **38.33%**，GLM-5.3 max24.44%、GPT-6-Astra max33.33%；GPQA为92.93%，低于Astra max95.96%，不能概括为全面超过通用模型。原始样本量、完整评测脚本、训练token数/混合比例/算力和完整配方没有披露。内部场景仅科学计算39题、正向反应42题、NMR30题：相对Astra max分别+5.13、+4.76、**-3.33**个百分点。实验规划为117个共同问题，另两模型因拒绝>50%请求被排除。材料搜索从相同初始种群出发、每模型只接收自己的反馈，5个seed、5轮的best-so-far能隙均值4.318/3.689/3.353eV，是作者的受限搜索实验，不是完成新材料合成的独立复现。[图表源数据](https://www.deepprinciple.com/blog-assets/hydrogen-1.0/v20261102a/data/benchmark-cost.json)、[内部场景](https://www.deepprinciple.com/blog-assets/hydrogen-1.0/v20261102a/data/research-scenario-cost.json)。

成本图尤其不能当部署TCO：Hydrogen使用**GLM-5.3标价代估**；Figure1只算输出及推理token、不算输入与judge，Figure2还排除前期失败尝试，Astra的NMR成本是从其他场景外推。旧 `release.json` 与当前渲染图的数据还存在部分不一致（例如反应Astra83.33与85.71）；这里采用当前图所用 `benchmark-cost` / `research-scenario-cost` 文件，不混用旧值。

**发布与采用条件。** 模型权重MIT，独立代码/文档Apache-2.0；但BF16仓本轮显示 **gated=auto**，其模型API、提交史与config返回401，只有公开README与树可读；不能称无门槛下载已验证。量化两仓可读：NVFP4 routed experts，前3层dense MLP用FP8，其他部分保持BF16；作者测8×B300，示例实际部署只开262,144上下文。MXFP4为Quark的32元素组/E8M0尺度、动态激活，作者测MI350X/ATOM TP4、FP8 KV、关闭投机；卡记载128并发长思考填满KV池并驱逐时质量明显下降，48并发池占用<75%正常，**不是通用并发上限**。宣称1M上下文/128K输出不等于这些部署已实测1M。[量化部署卡](https://huggingface.co/deep-principle/Hydrogen-1.0-MXFP4/blob/5344d941f372950f8a688bc13f2841a5f7b92417/README.md)。

**建议：深入阅读系统部分，考虑分离复现实验。** 先复核“零更新导出漂移”和领域数据/反馈本身各自贡献，再在自己的科学任务上比较同推理预算、同量化和同verifier。尚不宜以公开示例推断可复现完整训练或科研发现效果。

### Sarvam-Prism：105B骨干的开放决策模型，源码给出了位置、掩码与FP8执行路径

**机构 / 时间 / 版本：** Sarvam AI；HF仓建立11-02，当前单个提交 `417cf26c` 时间11-03 13:16:18 UTC，按**新发现工件**收录。[固定模型卡/源码](https://huggingface.co/sarvamai/prism/tree/417cf26c2eea27286a803dd721b4e78698bdb875)，Apache-2.0。

**具体增量。** 基于Sarvam-105B，输入context和有限选项，直接返回choice/yes-no/有序score的概率分布。源码 `Encoder` 让每个option从相同position起步；`attention_mask` 使context内部及各option内部双向可见、option互不可见、末尾answer可见全部，再由两层小attention head读取各option末端和answer隐藏态。它避免逐token生成答案；多个问题仍分别形成序列，按长度排序、默认至多16个一批，不能理解为任意问题数都只需一个共享前缀forward。`prism.py` 的权重布局为按输出行缩放的FP8，激活按token动态缩放；routed专家走两次Triton grouped FP8 GEMM，其余投影用`torch._scaled_mm`，常见形状捕获CUDA graph。

**证据与限制。** 卡给出的p50 **44ms**、p95 **420ms**，是1张B200、加载后、Decision Index随机1,000请求**串行**测量，约106GB磁盘/120GB可用GPU内存；没有可直接推广到长context/并发负载的吞吐数据。配置131,072上下文，不是44ms覆盖这一长度的证明。定制FP8格式需用其`prism.py`，不能直接`AutoModelForCausalLM`或通用server加载。官方release博客本轮直接访问403，web也未恢复，训练数据/校准与跨模型准确率比较未在可读材料中充分核实，本轮只确认卡和推理源码，不补写“超过其他决策模型”。

**建议：考虑在路由/判别任务小规模复现，继续观察质量证据。** 它为较大MoE骨干提供了开放的非生成式推理参考；先测自己的校准误差、标签置换稳定性、长输入与批处理时延，再判断105B的资源投入是否划算。

## 后训练与数据质量

### Dolci-Think-SFT-Clean-v2：发布可重建的删样本版本；单种过滤有增益，但完整v2尚未训练验证

**机构 / 时间 / 版本：** Ai2；数据提交11-03，固定 [0c6818cd](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-Clean-v2/tree/0c6818cdf7c623242f51cbe5acf56fed4cc217ca)。实际数据/来源文件在 `ec947927` 已齐，后续为卡；总体ODC-BY，保留上游各来源许可。

**具体增量。** 固定源 `allenai/Dolci-Think-SFT@7668c638`，在Clean-v1的drop list之外，再用Kimi K3重解OpenThoughts3 Math，筛出“没有boxed答案、以矛盾/无解/欠定等语句结束、未给终答”的148,210行，排除证明题，其中8,521行已被v1删除。合并去重后删除203,303/2,253,684行，留下 **2,050,381行、19,160,262,205 token**，源token的15.6%被移除；保留行内容与顺序不变，未把Kimi生成文本加入训练。[模型卡](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-Clean-v2/blob/0c6818cdf7c623242f51cbe5acf56fed4cc217ca/README.md)、[manifest](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-Clean-v2/blob/0c6818cdf7c623242f51cbe5acf56fed4cc217ca/meta/manifest.json)。

**关键证据。** 对随机200个flagged样本人审，187个确有问题，precision93.5%、Wilson95%区间89.2–96.2%，**recall未测**。在Olmo3.5 base的单seed、固定token预算SFT中，仅施加Kimi删除、未加v1其他过滤：MATH七类合计n=5,000、T=1.0，+1.69点，95%CI[+0.69,+2.69]；GSM8K+2.20、PopQA-0.02。虽然区间排除0，未达到预注册的+1.15点增益门槛所要求的证据，作者结论仍为 **inconclusive**。不能把它写成完整Clean-v2已证实提升：完整v2尚未训练，实验也没分离“删哪些行”与“少了这么多数学数据”的影响。

**复现条件与局限。** 公布drop IDs、Kimi IDs、hash、精确源版本和build脚本，可从固定源确定性重建行选择；Kimi原始completions未公开，因此不能从零复现检测器。数学源占被删token的93%，Kimi筛选使该源约23%token消失，需要重新衡量配方权重。仍有benchmark污染、占位cutoff和未修复安全/语言问题；`Clean`不是已完全去污染的保证。新旧卡token口径不同，应用固定tokenizer/template比，不能把两个版本卡的总token直接相减。

**建议：深入阅读，并考虑做匹配删除量的消融。** 价值在于可执行的删行证据和诚实的负面边界；复现时至少加入等量随机删数学行、重复seed与独立去污染评测。

## 多模态架构与工件

### GenFirst：旧论文新增共享视觉latent与统一文本—图像权重，不能按新论文首发处理

**机构 / 时间 / 版本：** USTC / ByteDance Seed；原论文 [2608.29335](https://arxiv.org/abs/2608.29335) 为旧工作；**11-03工件增量**固定 [HF@62bdef08](https://huggingface.co/ByteDance-Seed/GenFirst/tree/62bdef08b75ce79f8f3e9d1516d14e9b0b70c212)，代码基线 [0b8bea81](https://github.com/ByteDance-Seed/GenFirst/tree/0b8bea81c38770eb5bbf30a5824f618cf5866552)。

**具体增量。** 对主窗前后文件树核对，新增 `shared-latents/{siglip,vlm-nll}`、配套Qwen3-VL ViT autoencoder和latent stats，以及 `unified-text-image/e2e-joint` 的SFT前/后检查点与Cola文本/FLUX图像VAE；两套Qwen-ViT decoder也替换为512px微调权重。原理是VAE与生成先验联合训练，先让生成目标塑造latent，再把prior/entropy权重各降10倍恢复重建细节；当前新工件进一步把表示监督与统一文本/图像latent建模带入可用权重。公开的是EiT扩散先验路径，EAR自回归路径仍未发布。[固定模型卡](https://huggingface.co/ByteDance-Seed/GenFirst/blob/62bdef08b75ce79f8f3e9d1516d14e9b0b70c212/MODEL_CARD.md)。

**证据与限制。** 512px共享latent的SigLIP/VLM-NLL方案GenEval为0.90/0.89、linear-probe79.25/81.69%，显示生成与表示质量取舍；统一模型T2I的86.57是**百分口径**，而共享latent表用0–1，不能直接摆在一起。统一模型的T2I成绩来自phase3 SFT，COCO caption来自phase2，不能当同一checkpoint同时获得两组成绩。T2I采用50步Euler、CFG7且只在[0.35,1.0]启用、无prompt改写；本轮未复现。工件是需项目代码的原始`.pt`，没有diffusers/from_pretrained一键路径，VAE必须配自己那份统计量。代码MIT，但**权重CC-BY-NC-4.0、限非商业研究**，还需遵守上游组件条款。

**建议：考虑复现共享latent与双模态权重的配对实验。** 对理解/生成统一架构的训练者有价值，主要增量是复现资源补齐，不是这一天才提出端到端VAE算法。

## 推理工件与正确性

### LFM3 NVFP4 GGUF：新文件随缩放因子修复可用，DGX Spark数据并不支持“4bit必然最快”

**项目 / 时间 / 版本：** Liquid AI / llama.cpp，11-03。[1.7B GGUF@cd0c74c5](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/tree/cd0c74c575b5c416f3011b5b4c298077f26ca8c2)、[VL2B GGUF@1add221a](https://huggingface.co/LiquidAI/LFM3-VL-2B-GGUF/tree/1add221a5ea6c047fcf1eaaf2d247939d7cf8849) 新增NVFP4文件；[llama.cpp#31608](https://github.com/ggml-org/llama.cpp/pull/31608) 于14:36:45 UTC合master，合并`d1044bc8`，**预发行**[b12088](https://github.com/ggml-org/llama.cpp/releases/tag/b12088)（11-04 00:54:14 UTC，API `prerelease=true`）。

**具体变化。** GGUF来自独立的ModelOpt NVFP4 checkpoint，前馈权重保留NVFP4，其他权重Q8_0；VL视觉编码器未在源checkpoint量化，沿用原mmproj。最终diff在dense/MoE FFN和attention gate传入此前漏掉的weight scales，因此旧版即使**能加载也可能输出错误**。这既是新格式落地，也是正确性修复，不是仅多了一个量化名称。

**证据与限制。** PR作者在1张GB10/DGX Spark、CUDA、`ngl99/fa1/t4`、pp512/tg128、5次重复测1.7B：Q8_0+NVFP4为1.18GiB、pp14,299.6/tg165.54 tok/s；BF16为3.22GiB、13,070.1/69.13；QAD Q4_0为993.13MiB、13,824.4/**190.43**。NVFP4相对BF16 decode更快，但比这份Q4_0更大且更慢，短基准不能推广到并发、长KV或VL图像输入。本轮核实最终patch和作者表，未实机复现。

**建议：采用前先验证含修复的b12088预发行或固定合并commit，按自己负载比较NVFP4与QAD Q4_0。** 架构支持和缩放读取要与权重一起固定；不能只验证文件成功打开。

### Metis-1.6：llama.cpp主干接入混合递归结构，现有权重不再只能依赖未合入分支

**项目 / 时间 / 版本：** Lernex / llama.cpp，[#31134](https://github.com/ggml-org/llama.cpp/pull/31134) 于11-03 15:24:46 UTC合master；固定合并/tag [b12089@9309bf1b](https://github.com/ggml-org/llama.cpp/commit/9309bf1bcfe8d52b0fb7ef906474db94041a5e0c)。模型卡当天把“需拉PR分支”改为b12089以上；本轮tag可读，但同名release API返回404，因此这里只确认**主干与tag**，不保证发行二进制已齐。

**具体变化。** loader/converter/graph支持Praxis约2.94B和Logos约10.2B的Mamba2+GQA、两遍固定深度递归、每pass attention LoRA、top4 LatentMoE、4路mHC和跨pass KV。转换器为每pass/层分配缓存槽而不复制物理层权重，并把只依赖权重/pass的Sinkhorn混合提前折叠；这是运行时对其结构的实质支持。权重本身未在此次README更新中变化。

**证据与限制。** 作者报告CPU logits与PyTorch reference的KL不高于reference BF16自身波动，三条prompt及143个arch测试通过；RTX5070Ti的Praxis Q4_K_M pp512约4033/tg128约250tok/s只是作者的该配置结果，没有同条件旧后端基线。本轮读了最终files，未运行模型。

**建议：现有Metis用户可考虑迁移并做logits/状态缓存一致性核对。** 新后端支持降低验证门槛，但Ollama/LM Studio还取决于各自打包的llama.cpp版本。

### Gemma4 E2B/E4B：tokenizer修复扩展到四个工件，部分词表字符串也有变化

**机构 / 时间 / 版本：** Google，11-03。固定 [E4B-it@4ecc2ba6](https://huggingface.co/google/gemma-4-E4B-it/commit/4ecc2ba63bcac23ba4a767fc87025b1450f84c9c)、[E4B@11b8250e](https://huggingface.co/google/gemma-4-E4B/commit/11b8250ef59eb3ebc5dae7b21144d7b84f2c06dd)、[E2B-it@4adf1574](https://huggingface.co/google/gemma-4-E2B-it/commit/4adf15748de9e1a0097b6274cee9c15836c41c37)、[E2B@433813a6](https://huggingface.co/google/gemma-4-E2B/commit/433813a669639fd7eaa83155828e9000ed357d53)。这与10-31已报12B的修复是**不同模型版本**。

**实读差异。** 四仓本次只有`tokenizer.json`对象变化。完整解析前后JSON，E4B base/it各删除253条BPE merge，词表相同；E2B-it删除267条、E2B删除246条，均把ID249788的旧字符串替换为`<glitch_pruned_249788>`。四者词表规模仍262,144、ID集合相同，没有embedding尺寸变化。提交标题所谓219/233/217个glitch token与删除merge数不是同一统计对象。

**证据与局限 / 建议。** 这是编码路径与一个ID解码字符串的修订，权重没变；本轮验证JSON，未验证完整tokenizer运行或生成行为，也不能说所有glitch已消除。使用这些模型的团队应**考虑同步固定tokenizer版本并重做缓存/评测基线**，尤其不要混用旧编码与新解码；需要精确复现旧实验时仍保留旧revision。

### Ling-3.1-flash与K2-Horizon-32B-FP8：分别更新数值语义与实际量化权重

- **inclusionAI / Ling-3.1-flash，11-03，[dde57b1c](https://huggingface.co/inclusionAI/Ling-3.1-flash/commit/dde57b1c6a365aa0e0a33e15cecce0f2be96f5a3)。** 只改自定义Python代码、未换权重。`BailingMoeV3SparseMoeBlock`现在按layer_idx读取routed/shared的SwiGLU limit，MLP对SiLU(gate)做上界裁剪、对up做对称裁剪，再相乘。当前config的72层中routed7层、shared8层值为10，其余0代表不裁；dense替换层不套该专家设置。旧remote-code实现忽略这些已有配置，所以可能改变HF前向数值。**建议考虑更新并验证与生产推理引擎一致性**；本轮未跑logits与任务回归，不能量化收益或保证所有后端等价。
- **IFM / K2-Horizon-32B-FP8，11-03，[716c30a6](https://huggingface.co/IFM/K2-Horizon-32B-FP8/commit/716c30a6933d37ffb1d7042cf88436feb744a686)。** 文件树确认两份safetensors对象均被替换，config的initializer_range 0.02→0.01；不是仅文案更新。作者提交称“同步final32B”，但卡仍写Stage1与“final checkpoint待发”，没有本次权重对应的新评测/校准误差，不能把旧表直接归给新权重。既有配方为除lm_head外128×128静态FP8权重、1×128动态激活。**建议现有用户固定旧/新sha作回归，继续观察正式卡与分数对齐**，不凭提交标题推定能力提升。

### Nota裁专家版Mistral Large4：4×B200装载与长上下文的取舍，仍依赖门控工件和专用patch

**机构 / 时间 / 版本：** Nota AI，HF11-02创建/更新，**补录** [Global-Pruned-20@61f4689a](https://huggingface.co/nota-ai/Mistral-Large-4-1T-A52B-NVFP4-Global-Pruned-20/tree/61f4689afeb4972b961fdec8497d41b138c37838)。

**具体增量与作者证据。** 在原NVFP4上非均匀去掉约20% routed experts：9,088→7,264，每层96/104/112/128，前后各两层MoE不裁；checkpoint568→460GiB。卡声称按vLLM recipe的262,144上下文/0.85显存利用率，4×B200 TP4得到428,658 token KV池，原版4卡装不下而需8卡TP8；512K示例把利用率调到0.90。八项自报测试并非全部保持：IFEval93.90持平，IFBench75.33→78.00，但GPQA87.50→84.34、LCBv6 86.26→84.73、KoBALT70.29→67.00；输出上限65,536含reasoning，未给足以复现的完整采样/重复次数/输入负载。

**限制与建议。** 四卡裁剪版对八卡原版不能作同资源速度或成本提升；核心价值是装载门槛与质量取舍。还需基于`vllm@bd4fa263`安装仓内patch来支持不等expert层数；本轮HF显示gated=auto，patch/pruning_info与提交史均401，只读到公开卡/metadata，不能声称已审代码或验证专家选择。许可继承Mistral Ecosystem Open-Weight License，另有访问时的Notification Commitment；不是Apache/MIT。**继续观察，已有访问权限的团队可考虑复现装载与质量曲线**，先核实patch和所保留专家，暂不把这组自报结果当生产替换依据。

## 评估与结论边界

### 权重空间编程有了独立的本地读权重试验，但没有复现“改LLM权重后数学提升”

**作者 / 时间 / 来源：** Simon Willison，11-03 16:29 UTC，[原始试验与日志链接](https://simonwillison.net/2026/Nov/3/qwen-can-interpret-model-weights/)。这是对昨日已报工作的**独立实践增量**。

**具体证据。** 他用原作者`exp1_which_game`的无工具prompt，让LM Studio中的Qwen3.8-27B量化本地模型只读浮点权重：辨认出Pac-Man、Pong、Breakout，Freeway认错。Pac-Man一次耗46分12秒/43,506 reasoning token，Pong64分54秒，Breakout52分25秒；他另用Kimi K3.1 Pro认出Breakout，用时9分41秒。prompt除权重外相同，模型输出与推理trace可查看；这是第三方实际运行报告，本轮没有重跑。原文游戏总述10×10、模型答案写8×8，也不应把自然语言描述完全正确与“猜对游戏名”等同。

**限制 / 实践意义。** 这是小策略网络语义识别、四个公开示例、未做盲测/随机权重控制/多seed准确率；不证明任意网络可被理解，更没有独立重现Qwen0.8B权重编辑或MATH收益。**建议深入阅读日志并考虑有控制的复现**：将无工具读权重、带代码编辑权重分开，报告失败例、对照与计算预算。

<a id="papers"></a>

## 预训练、优化器与数据重用

### Muon的预训练loss优势，在充分预训练后的微调前沿可能反转

**项目 / 版本 / 日期：** Pretraining Data Scale Reverses the Muon–AdamW Fine-Tuning Gap；`arxiv:2611.03633v1`，v1提交 **2026-11-03 02:56（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §3–4、Appendix C/D](https://arxiv.org/html/2611.03633v1)。

对OLMo式30M–600M模型、0.15B–154B DCLM token，分别调预训练与微调超参，并比较新任务学习和原域遗忘的整条前沿。数学、指令和音乐四组微调中，随token/参数比增大，AdamW基座在微调后的前沿越过Muon，尽管Muon预训练loss始终更低；反转所需预算随模型增大而下降。作者用线性关联记忆的margin-to-spread ratio解释对logit扰动的鲁棒性，理论比较是无动量精确正交Muon与GD，不能等同真实AdamW的完整理论。

这是对“只按预训练loss选优化器”的直接纠正。规模上限600M，结果主要是分布损失而非大模型任务成功率；不能据此宣称生产级Muon已被否定，也不能把微调时换优化器与基座差异混为一谈。

**建议深入阅读并在自己的mid/post-training配方上复现学习—遗忘前沿；仅比较预训练loss不足以决策。**

### OrthMuon控制实验：行正交的收益多数被球面约束和可学习幅度解释

**项目 / 版本 / 日期：** CMU / OrthMuon；`arxiv:2611.00593v1`，v1提交 **2026-10-31 11:42（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §4–6、Tables 1–9、Appendix F/G](https://arxiv.org/html/2611.00593v1)。

把block-Stiefel正交、每行单位范数、可学习gain、优化器实现与状态精度拆开。在C4/FPO-mix的360M–3.1B控制中，sphere+gain的d=1已优于调参Muon；360M、14.399B token的C4 loss为2.462，对Muon2.475，而加入32行正交为2.467。1.3B/3.1B中带完整SPD增益的正交版本可追平d=1，却未稳定超过；三seed配对区间给出不确定性。8×GB300的360M吞吐比Muon低约1.8%。

补充复查显示，AdamW moment从BF16改FP32会明显缩小甚至逆转若干正交方法原报优势。结论限这些参数化、优化器与数据；“无额外正交收益”不等于所有几何约束无用，跨语料代码/math loss还有较大seed噪声。

**建议深入阅读，复现时先匹配状态精度、范数控制和gain自由度，再为正交本身支付复杂度。**

### Muon缩放weight decay：先调LR×WD，原始收益大部分会消失

**项目 / 版本 / 日期：** Esther Xin / weight_decay_ladder；`arxiv:2611.02529v1`，v1提交 **2026-11-02 12:59（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §3–7、Table 1/2](https://arxiv.org/html/2611.02529v1)。

对Muon-SW和恒定decay分别扫描学习率与LR×WD积。最小L3的共享默认设置差0.0350 BPB，分别调优后完整二维比较只余0.0026 BPB，约保留7.4%；这个剩余收益在更大rung上增加，不能写成“全是调参假象”。同时WSD下最优积比值不符合论文预设的几种简单假说。

最大rung仅15M活跃/94M总参数；仅L3有多seed和完整二维扫描，L4/L6是单seed一维扫描，且部分拟合最优落在扫参边界之外；计划中的更大L9没有跑。无法外推到MAI等超大模型。

**建议考虑复现公平的LR/WD二维扫描；这比直接把新decay schedule纳入默认训练栈更能验证收益来源。**

### 固定语料多epoch：停止时间与ensemble应共同分配预算

**项目 / 版本 / 日期：** UT Austin、CMU / Multi-Epoch DMFT；`arxiv:2611.03590v1`，v1提交 **2026-11-03 02:34（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §2–4、Appendix A](https://arxiv.org/html/2611.03590v1)。

在10M–100M unique FineWeb token的小模型实验中，验证loss约9–14 epoch见底后上升，而训练loss继续下降；独立模型logit ensemble推迟最佳停止点。在匹配参数×token训练预算的一个对照中，4个基模型ensemble比单个两倍宽模型低0.16 nats/token。

理论是固定数据、Gaussian random-feature与SGD的动力学平均场分析，描述跨epoch/跨成员误差相关；它对真实Transformer只给定性对应，未包含完整feature learning与Adam动力学。ensemble会增加推理模型数，等训练预算不意味着等服务成本；9–14 epoch不是可直接套到工业数据配方的通用停止准则。

**建议深入阅读数据重用与早停机制；做小规模多epoch实验时同时记录验证曲线、ensemble预算和推理代价。**

### ZO-Curv把空闲推理设备用于曲率，半时长不代表半总算力

**项目 / 版本 / 日期：** 北大、南大等 / ZO-Curv；`arxiv:2611.02688v1`，v1提交 **2026-11-02 15:38（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §3–4、Table 1、Appendix B](https://arxiv.org/html/2611.02688v1)。

训练节点保留精确梯度，forward-only池用随机扰动损失估计对角曲率，并以一步延迟提供预条件；这比直接混入高方差ZO梯度有效。GPT-2式125M/355M/770M和1.5B GPT-NeoX、OpenWebText、1024长度、batch512；770M到AdamW目标loss的训练关键路径7.53h→3.98h。

主实验在单8×H100节点模拟forward池，q=512意味着每步2q次扰动forward；曲率池的并发成本被排除在该墙钟计价外，论文另有分离GPU组实验支持加速，但并非生产serving集群验证。SophiaG基线每步刷新曲率，比常见稀疏刷新更昂贵；模型广播假定不成为瓶颈。

**建议继续观察并先算资源账：只有确有闲置forward容量、快网络和可接受服务干扰时，才考虑复现。**

### Adam checkpoint有界有损压缩：证书省空间，编码时间和内存仍高

**项目 / 版本 / 日期：** Guarded Compression of Adam Optimizer State；`arxiv:2611.00589v1`，v1提交 **2026-10-31 11:34（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §2–4、Tables 1–3](https://arxiv.org/html/2611.00589v1)。

直接界定“共享同一梯度下原始与恢复Adam更新之差”，使用mixed partitions、native rounding界与精确escape，避免分别放大两个更新区间。OPT-125M百万坐标、误差预算0.01时，moment payload为原始FP32的28.28%，而同guard的BF16为51.57%、8bit为39.73%。120次完整OPT-125M/ResNet18恢复均满足首次更新容差。

这是checkpoint压缩而非持久8bit优化器。完整容器还含lossless权重等，分母不能只用moment；OPT半程codec中encode+restore约16.905s，对FP32 3.955s，峰值编码24.888GiB，对3.622GiB。100步续训差异仅描述性，首次更新界不保证长程训练轨迹；joint code search基本没有额外收益。

**建议深入阅读依赖保持的误差界；大型分片checkpoint与编码资源尚未验证，暂不直接采用。**

### Library of Layers：按window动态选择整层，连接循环与稀疏架构

**项目 / 版本 / 日期：** LoL / Library of Layers；`arxiv:2611.03526v1`，v1提交 **2026-11-03 01:57（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §3–5、Table 1](https://arxiv.org/html/2611.03526v1)。

固定prelude/coda，中间core每个window由top1 router从共享Transformer block库选下一层；strided-window attention让相邻window交换信息，router只读先前window，避免未来泄漏。改变库大小可从重复用层走向稀疏选层；360M有效计算规模、10B FineWeb token、三seed，与等独立层数Huginn损失接近，路径可在合成Sudoku/maze任务中被干预控制算法。

动态路径不等于免费test-time递归扩展。匹配约17EFLOPs的3:2 sparse MoE对照最终训练loss约2.75，LoL约2.8，未证明质量/吞吐全面超过MoE；router初始化带来更大seed方差，大模型训练稳定性、batch路由和服务kernel未验证。

**建议深入阅读动态层顺序这一架构选择，先复现同FLOPs/同active参数的实际时间与泛化。**

## 后训练、蒸馏与数据

### MEMENTO：把压缩笔记作为可训练的长上下文动作

**项目 / 版本 / 日期：** Columbia / MEMENTO；`arxiv:2611.00244v1`，v1提交 **2026-10-31 02:25（Asia/Shanghai），补录**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §3–4、Tables 1/2、Limitations](https://arxiv.org/html/2611.00244v1)。

Qwen3-8B先SFT于事后补注笔记的合成轨迹，再用答案、笔记启发式与confidence奖励做RL；只训练compaction，测试可接context folding/backtracking。四组英文book QA、最长书2.5M token、驻留context不超20K，表1 micro accuracy从同harness未训练36.9升至67.8%；仅SFT+RL folding为66.3%。

GPT-5.5对照每数据集最多100题，模型对harness的适配程度不同，“便宜50倍”不能当通用模型能力排序。强策略上的backtracking只加约1点却占43%token；37题Saga子集很小，答案由LLM judge评，笔记有效性奖励靠人工规则，未检验代码/web等迁移。

**建议考虑复现SFT笔记启动＋compaction RL，优先验证去掉昂贵backtracking后的收益。**

### SFTProbe：先探短轨迹的loss几何，再支付完整推理生成成本

**项目 / 版本 / 日期：** UCLA、Optum AI / SFTProbe；`arxiv:2611.00554v1`，v1提交 **2026-10-31 10:27（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §3–5、Table 1/2、Appendix D/E](https://arxiv.org/html/2611.00554v1)。

短暂LoRA拟合低秩方向，在预训练点和曲率拐点测部分reasoning token的loss signature，聚类近似梯度多样性并筛难度；teacher选择还用学生对teacher短改写的loss。医学、数学、物理三域有数据挑选/生成实验；5K数学示例匹配89K OpenThoughts训练的同学生结果，物理Qwen2.5-7B-Instruct从3.58到5K数据的9.76±0.74（论文自定平均分）。

这不是不生成任何轨迹：需要短前缀、LoRA probe、拐点搜索及最终入选样本完整生成。53倍“算力”比较包含不同teacher/生成筛选流水线的配置，不能视作训练kernel提速；理论依赖局部近抛物面/低秩几何，改变学生、任务或teacher后应重校准。

**建议考虑复现，用真实生成GPU小时、过滤损耗和最终学生效果一起比较。**

### L2RR：RLVR样本效率还取决于源训练分布与目标分布失配

**项目 / 版本 / 日期：** RIKEN-AIP等 / L2RR；`arxiv:2611.00586v1`，v1提交 **2026-10-31 11:28（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §3–5、Tables 1/2](https://arxiv.org/html/2611.00586v1)。

按目标验证梯度与源prompt梯度的对齐、再加rollout浪费正则，用指数梯度更新源prompt采样分布，接入GRPO。Qwen2.5-Math-1.5B在DAPO+MATH源域训练、六数学benchmark五训练seed：均分47.7±1.5→50.3±1.3，累计source responses7.6M→3.0M。

关键前提是能取得小型目标validation的聚合梯度反馈；“不见目标prompt”不等于无目标域信息。rollout条数不是生成token或GPU小时，长度与梯度计算会改变成本；理论目标loss界有条件，数学小模型结果不保证开放Agent或没有目标梯度的迁移。

**建议考虑复现适合有固定目标验证集的训练；与只按难度采样的baseline同时比较。**

### Bernoulli OPD：高置信学生token也可能需要被纠正

**项目 / 版本 / 日期：** Surrey、UCL等 / Bernoulli OPD；`arxiv:2611.02093v1`，v1提交 **2026-11-02 02:06（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §3–5、Table 1/2、Appendix A](https://arxiv.org/html/2611.02093v1)。

采样reverse-KL目标的token梯度含“1减学生概率”，会抑制高置信token。改成Bernoulli cross-entropy，使该token的梯度直接反映有界teacher—student概率差；另用移除抑制项的GC-OPD作诊断。DAPO-Math-17K、两类1.5/1.7B学生、四teacher设置，五benchmark均分相对vanilla OPD提高约0.98–7.38点，收益依teacher而变。

Qwen设置non-thinking、DeepSeek设置thinking；AIME24 Avg@8选checkpoint，再报包括AIME24的最终集合，有selection重用风险。该目标改变梯度几何并非原reverse-KL的等价改写，五项中前四Avg@32、MATH500 Avg@4，不能与其他论文的pass@1直接横比；尚无独立复现。

**建议考虑复现，监控按学生置信度分桶的teacher误差和梯度，避免只看总loss。**

### Order-Only OPD：teacher排名可比概率幅度更稳定

**项目 / 版本 / 日期：** Memphis、Missouri / O3PD；`arxiv:2611.02112v1`，v1提交 **2026-11-02 02:38（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §3–4、Appendix A](https://arxiv.org/html/2611.02112v1)。

仅匹配teacher top8 token排序，用相邻pair ranking loss，丢弃概率幅度，因此对teacher温度不变。六组Qwen3的4/8/14B teacher与1.7/4/8B student，在三竞赛数学任务上均超过文中OPD。8B→1.7B对照里，OPD早期38.1后下降，O3PD末期40.7。

这里OPD基线是teacher top8支持集上的Jensen–Shannon目标，并非上一条的完整采样reverse-KL；不可把两个百分点直接比较。最高三benchmark均分用于选checkpoint，SFT/GRPO数字取自外部同配方研究而非全部重跑；排序可丢掉本来有用的校准信息，非数学任务、跨tokenizer蒸馏仍待验证。

**建议考虑复现排名蒸馏与Bernoulli等目标的同配方比较，单独保留真正未用于选点的测试集。**

### SPTA：好teacher应会接着学生犯错的前缀继续推理

**项目 / 版本 / 日期：** 腾讯Weixin AI、北交大等 / SPTA；`arxiv:2611.03133v1`，v1提交 **2026-11-02 21:45（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §3–9、Tables 1/2](https://arxiv.org/html/2611.03133v1)。

先随机截断student rollout，让teacher在这些前缀上做verifiable-reward适应，再冻结teacher跑普通OPD。三teacher/student配对均改善；Qwen3-4B→4B-Base学生均分31.52→35.30，teacher干净prompt准确率却下降6.60点，说明standalone teacher能力不是唯一指标。

这还需要一轮teacher RL和verifier，非免费替换。没有teacher自身或无关模型前缀、等长上下文对照，前缀来源与长度效应未完全拆开；三配对、数学任务、一次适应，部分Prune-OPD参照来自引用而未独立重跑。

**建议考虑复现“teacher接学生前缀”的恢复能力评估，再决定是否值得额外teacher训练。**

### Team-GRPO：组依赖reward下，减组均值会改变真正的优化目标

**项目 / 版本 / 日期：** When Rewards Depend on the Group / Team-GRPO；`arxiv:2611.02404v1`，v1提交 **2026-11-02 10:26（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §2–4、Tables 1/2](https://arxiv.org/html/2611.02404v1)。

当diversity、coverage或distribution matching reward取决于整组response时，普通GRPO遗漏一个response对组内其他人的外部性，从最大化组价值变成组内竞赛；增大group不必消除偏差。Team-GRPO通过留一组总reward重算advantage，恢复目标梯度；pairwise reward无需额外模型调用。

GlobalOpinionQA、200步、每步256 rollout的分布匹配实验，K=2时普通GRPO的平方误差0.491，Team为0.081；不是所有RLVR reward都属此情形，固定的独立binary correctness reward不自动触发该问题。完整GRPO还含std归一化、clip与KL，理论固定点分析不能直接替代端到端RL稳定性验证。

**建议深入阅读；使用组内多样性或set-level reward时先检查计算图和留一外部性。**

### NAPO：全错组的负样本也能提供“首次解出”的学习信号

**项目 / 版本 / 日期：** 上海交大 / NAPO；`arxiv:2611.02747v1`，v1提交 **2026-11-02 16:35（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §3–5、Table 2、Limitations](https://arxiv.org/html/2611.02747v1)。

用早期不确定性与Fisher视角给全未解组的negative rollout分配不同负advantage，避免标准GRPO在全0组完全无梯度。固定rollout预算的数学实验中，AIME24 Pass@256从GRPO0.700到NAPO02的0.800，AIME25从0.4667到0.5333。

“能力扩张”不可只看大k：AIME25的pass@1从0.1346降到0.1220/0.1138，保多样性与单次可靠性有取舍；AMC23 Pass@256未超过GRPO。只测数学binary verifier，driver受group、温度与近可解样本比例影响，尚未证明长程Agent同样有效。

**建议考虑复现并同时看pass@1、pass@k、首次解出时间及token预算，按实际采样策略判断。**

### FADE：把动态能力与静态题难度分开，少量反馈更新整个题池

**项目 / 版本 / 日期：** Michigan、Edinburgh等 / FADE；`arxiv:2611.03625v1`，v1提交 **2026-11-03 02:53（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §3–5、Tables 1/2、Limitations](https://arxiv.org/html/2611.03625v1)。

在冻结prompt embedding定义的低秩skill坐标上，success logit分成动态能力状态与慢变题目intercept；用Pólya–Gamma闭式滤波更新，供课程selector和rollout allocator使用。同1.44M rollout预算，Qwen2.5-Math-1.5B的GVM五benchmark Avg@8为42.7，换FADE估计为45.5；Llama1B相应12.3→14.1。

FADE仅换difficulty估计，收益依宿主scheduler；0.06h估计开销不是全训练时间。冻结语义坐标可能不表示真正推理难度，组内binary条件独立假设不覆盖相关Agent轨迹，AIME只有60题，3B/7B外推仍较有限。

**建议考虑复现现有scheduler中的替换实验，保持selector和generation总预算不变。**

### KAPPO：一次新completion也可施加面向pass@k的训练压力

**项目 / 版本 / 日期：** EPFL、ETH / KAPPO；`arxiv:2611.03146v1`，v1提交 **2026-11-02 21:54（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §3–7、Appendix H](https://arxiv.org/html/2611.03146v1)。

从KL正则pass@k/max@k最优策略推导prompt自适应reward变换或regularization，让现有@1优化器调整压力；A*-PO实例每prompt只需一次新的在线completion。OLMo3-7B-SFT与Qwen3-4B-Base的数学/代码实验比较k=1及PKPO，强调easy/hard题不该被统一sharpen。

一次在线rollout之外仍需每题静态M-completion reference bank校准，不是总采样成本降到1；该bank的噪声和分配未优化。连续reward实测用win-rate surrogate，并未完整验证raw max@k的数值重建；参考分布连续的假设在离散同分reward上只是近似。

**建议深入阅读，复现时把离线bank和在线rollout一起计价，再对齐真实test-time k。**

### Mix-FT：少步扩散语言模型应拟合实际运行的采样器

**项目 / 版本 / 日期：** University of Zanjan / Mix-FT；`arxiv:2611.02666v1`，v1提交 **2026-11-02 15:24（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §3–8、Tables 1/2](https://arxiv.org/html/2611.02666v1)。

把少步masked-diffusion sampler写成对mask计划的混合分布，再按该混合似然微调，避免仅优化one-token/step的NELBO。110M BD3-LM与LLaDA8B实验显示少步质量改善；8B LoRA、64 prompt+192续写、block32，Mix-FT的16步相对起点32步在两LLM盲评中偏好约69%/71%。

这不是所有步数的蒸馏赢家：K=2时SDTT-4优于Mix-FT，8B的K=8仍落后原K=16；GPT2-large等judge NLL只在匹配entropy后比较，并非真实任务成功率。1.1B GSM8K自生成verified数据上并未超过继续CE。

**建议考虑复现具体sampler与预算，保留SDTT和继续CE对照，勿把减半model eval等同任务质量无损。**

### 持续多teacher OPD：replay份额比花哨loss更关键，parser会伪造迁移

**项目 / 版本 / 日期：** Yiqiao Qiu / FSAR；`arxiv:2611.02746v1`，v1提交 **2026-11-02 16:35（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §5–6、Tables 1/2、Appendix C](https://arxiv.org/html/2611.02746v1)。

DeepSeek-R1-Distill-Qwen-1.5B依次学logic/math/code/safety/IF，用旧student与历史prompt做replay；SATS/OPRD辅助信号只在replay稀少时明显有用。提高固定replay份额可接近单独末期recovery，且无需保存所有旧expert。

14benchmark均分中，足量replay约48.95，末期recovery49.19；同基座expert直接TIES merge达52.29，故不能称蒸馏全面最优。BBH/GYM的正backward transfer大量来自<answer>换成boxed的parser适配，MuSR主要读“answer:”格式；只有三evaluation seed，非三独立训练seed。

**建议深入阅读replay与snapshot保留策略；复现时同时跑可合并权重基线和统一答案解析。**

### RLVR策略泛化：ID会推导，OOD仍可能退回枚举

**项目 / 版本 / 日期：** UIUC / rl_meta_thinking；`arxiv:2611.03627v1`，v1提交 **2026-11-03 02:53（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §3–6、Table 1、Appendix H](https://arxiv.org/html/2611.03627v1)。

用可控字符串函数反演区分结构推导与暴力枚举。仅训练非双射反演的Mono在ID可推导、OOD却退回枚举；加入双射任务的Mixed在相近ID下恢复更多推导，数学零样本六任务均分35.1→39.8，Mono为35.9。

任务是有意简化的synthetic inversion，reasoning策略分类由Gemini judge并有人工验证；二分类不能涵盖现实策略混合。“每token信息比特”是受约束搜索模型下的解释，不能当真实自然语言思维的直接测量，也不证明所有混合任务都会泛化。

**建议考虑复现可控OOD策略测试，用训练任务组成干预检验reasoning质量，而不只看ID正确率。**

### 改reward model之前，用策略内协方差预测先出现的行为变化

**项目 / 版本 / 日期：** MBZUAI / Reward-Edit Forecast；`arxiv:2611.03244v1`，v1提交 **2026-11-02 22:53（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §3–6、Table 2、Limitations](https://arxiv.org/html/2611.03244v1)。

在同prompt的policy样本上，计算feature与新旧RM reward差的协方差，预测RL后格式、列表与长度供给变化。Qwen3-1.7B-SFT、verl RLOO、64prompt×8sample/步、六parent branch point，预注册因子实验中稳定变化的方向预测20/20，后段explained share约0.82；直接按counterfactual“价格”预测约0.50。

只覆盖一个policy/RM家族、200步、可规则删除的格式特征；gold仍由两RM定义，并非人类真值。远离branch后激励会变，部分summary大shock导致长度失控，无法事前可靠预测长期反转。

**建议考虑复现作为RM升级的短程预检；预测方向不能代替实训后的质量与reward hacking评估。**

### 递归自训练的“变自信”与“学会区分对错”需要分开测

**项目 / 版本 / 日期：** BRAC University / Self-Training Confidence；`arxiv:2611.01059v1`，v1提交 **2026-10-31 23:55（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §3–6、Table 1、Limitations](https://arxiv.org/html/2611.01059v1)。

Qwen3-1.7B五轮纯自训练、三seed，accuracy和整体AUROC几乎不动，confidence却逐轮sharpen。干预显示每个生成答案训练两次与top-p截断是主要来源，移除后生成entropy不再缩；entropy匹配可接近有标签温度校准，focal等改善区分度却可能转向欠自信。

主评仅600道multiple-choice、option-letter confidence，8B只重复温度诊断，部分机制臂单seed且未多重比较校正。没有外部正确性signal，不能外推到真实filtered synthetic-data训练或断言模型知识完全不变。

**建议深入阅读并在合成数据流水线里同时测calibration、discrimination与task accuracy。**

### SWE-Trail：历史经验用于分配采集预算，而不是直接提示coding agent

**项目 / 版本 / 日期：** SWE-Trail / History-Guided Collection；`arxiv:2611.02074v1`，v1提交 **2026-11-02 01:26（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §3–4、Tables 1/2、Appendix D](https://arxiv.org/html/2611.02074v1)。

把同issue先前attempt编成collector-side detector，在新轨迹遇到值得探索的状态时，从同checkpoint短暂执行多个候选并选一条继续；agent本身不见历史和reference patch。Ling3.0-flash-base124B/5.1B-active学生，同1500轨迹、其余背景mixture/训练设置固定、每臂3独立训练：Multilingual均分63.33→65.67、SWE-Pro41.22→42.82；Qwen3-14B重复方向一致。

收集解出更多轨迹本身不保证学生获益，H0的Multilingual只63.44；H10比H5多12.9%记录token却仅加0.22/0.14点。记录采集token不等全部VM/回滚墙钟，H0还漏计offline compiler下界；只一个collector模型/scaffold，各issue分配相同总预算，暂未验证跨领域。

**建议考虑复现可回滚环境中的数据采集，把“采集成功数”和“相同SFT下训练价值”分开验。**

### CUA-Master：把公开视频变成可执行seed，再用回滚扩展任务与策略

**项目 / 版本 / 日期：** CUA-Master（与NorthCUA不同项目）；`arxiv:2611.03126v1`，v1提交 **2026-11-02 21:41（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §2–3、Tables 1–3、Appendix E](https://arxiv.org/html/2611.03126v1)。

双层world-data-model飞轮：从YouTube/TikTok/Bilibili教学视频做视觉/语义双路径轨迹重建，以原子任务组合和snapshot rollback retry扩展可验证任务；再按当前模型弱点更新数据。Qwen3.5-27B起点、16×8H200，三轮SFT新增63K/38K/14K任务累计约115K，再约12K任务RL、每题8条VM轨迹。

同协议重评中，OSWorld-Verified得分80.2，对起点54.6；OSWorld2.0为36.9，Weave43.9、Agents’ Last Exam full-pass25.0。除最后一项外是平均得分，不全是完整任务成功率。各open模型用1–6节点、closed用API，8分钟等runtime受部署和API速度影响；不能把不同模型时间当算法纯效率。强annotator与大规模VM成本依赖明显，完整数据/权重发布条件本轮未核实。

**建议深入阅读任务可执行化与反馈回滚；考虑复现前先估环境复制、verifier与annotator成本。**

### ISA-Distill：teacher知道答案时，可能错误惩罚学生必要的搜索

**项目 / 版本 / 日期：** 北理工等 / ISA-Distill；`arxiv:2611.01987v1`，v1提交 **2026-11-01 23:24（Asia/Shanghai），72h回溯首次纳入**。本轮由submitted-date API补充发现，未核实精确公开列示时刻。[固定全文 §3–6、Tables 1/2、Appendix A](https://arxiv.org/html/2611.01987v1)。

从训练gold/support抽action-relevant fact，仅用student行动前可见状态做32次不检索local probe估可恢复性，再在fact-use与evidence-seeking两种teacher视图之间加权，修正局部credit。Qwen2.5-7B七QA任务宏平均EM48.0，对EviSD47.2/SD-Search47.6；3B为43.9，对42.6/42.7，收益主要应读准确率—搜索次数曲线。

额外probe和双teacher scoring仅训练使用，测得step161.9→168.8s（+4.3%），但不是固定检索长度/资源下普适开销。需训练期gold与support，NLL可恢复性不是可靠知识真值，小幅均分差需更多seed，不能认为privileged teacher在所有OPD中都会抑制有用动作。

**建议深入阅读信息状态失配，复现时加入同检索预算及teacher视图消融。**

## 推理、量化与服务机制

### RTP-LLM：把模型执行留在Python，调度与KV生命周期留在共享runtime

**项目 / 版本 / 日期：** 阿里巴巴 / RTP-LLM；`arxiv:2611.00967v1`，v1提交 **2026-10-31 22:10（Asia/Shanghai），72h回溯首次纳入**。本轮由submitted-date API补充发现，未核实精确公开列示时刻。[固定全文 §3–8、Table 1](https://arxiv.org/html/2611.00967v1)。

这是已开源多年引擎的新系统报告，不是今天才推出项目。Python/PyTorch模型层解释checkpoint与算子，C++ runtime共用request协调、KV分配/释放、distributed context；FlexLB做cache-aware worker placement，KV跨GPU、本地内存、remote cache与3FS。GLM5.2/Qwen3.5初始支持只改38/2行serving层，DeepSeekV4仍需5207行，边界并非一次设计后永不变化。

三种业务各一个cluster、七天小时聚合的mean TTFT<2s、TPOT<20ms及80–90%峰值cache命中，是作者production观察，未披露可比的模型/硬件/长度/并发组合，不能拿来排名其他引擎；80天后期error rate<0.01%也不是错误为0或证明每次升级因果有效。报告强调多层state/resource guard、故障回收与版本回归，规模披露不替代可复现benchmark。

**建议深入阅读分层接口与长期维护取舍，再对照自己的模型更迭速度决定扩展既有引擎或自建。**

### MXCode：约2bit存储重建原生FP4，显存收益可转成prefix容量

**项目 / 版本 / 日期：** MXCode；`arxiv:2611.00403v1`，v1提交 **2026-10-31 05:48（Asia/Shanghai），补录**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §3–4、Tables 2/3](https://arxiv.org/html/2611.00403v1)。

用PostScale/PreScale/PreScale+把VQ codebook和scale转成MX兼容FP4，直接重建FP4 operand而不物化BF16权重。三MoE的质量用MX emulation测；Qwen3-235B在SGLang/GB300、32K输入1K输出服务，约2BPE expert存储相对native FP4模型载入显存近减半。单GB300 prefix-reuse P90 TTFT为8.688s→0.477s。

94.5% TTFT下降来自内存约束下前缀驻留差异，不能当算子快18倍；总四GPU的DP/TP图里改replica数，并未把整个集群卡数减半。性能只实现Pre/PostScale共享backend，PreScale+和E5M3未计；非expert、接口activation和KV仍BF16，W4A4质量与native FP4并非完全持平。

**建议考虑复现显存紧张的prefix复用负载，分别测无复用吞吐、任务损失和实际HBM。**

### 整数精确microscaling：把合适MXFP4行化成一次INT8 GEMM

**项目 / 版本 / 日期：** mxfp4-int8-m5；`arxiv:2611.00486v1`，v1提交 **2026-10-31 08:18（Asia/Shanghai），补录**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §3–9、Tables 1–3](https://arxiv.org/html/2611.00486v1)。

若一行32元素block的MXFP4 exponent跨度不超3，可精确表示为INT8向量乘一个2次幂row scale；提出限制跨度的标准SC-MXFP4，以及16元素block、整数multiplier1–10的非标准4.26bit格式。M5 Max实机、dense/MoE含125B FP8来源模型，prefill整系统约1.10–1.74倍，125B约1.22–1.29倍。

“精确”指量化参考的整数计算，不是相对原BF16模型无损。SC-MXFP4质量逊于部分llama.cpp量化；125B整数格式PPL比NVFP4差1.1%。batch1 decode没有INT8算力优势，16block格式在该模型decode仅MLX的0.65倍；单台M5结果不能推广为所有Apple或GPU部署加速。

**建议考虑复现M5长prompt场景；交互短decode优先核实回退路径和整任务耗时。**

### LoReS：recurrent state的量化，应按衰减与decode长度配置

**项目 / 版本 / 日期：** MPI、ETH、MIT、Liquid AI / LoReS；`arxiv:2611.01864v1`，v1提交 **2026-11-01 20:55（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §3–5、Appendix H](https://arxiv.org/html/2611.01864v1)。

区分模型外部W/A与递归内部state量化。受控1.3B mixer zoo显示decay减弱残差outlier，但逐步state rounding仍会累积；LoReS以value轴分组、随机舍入和decay-based精度分配处理。Qwen3.5/Nemotron-H/Kimi-Linear上，INT8 state大多保留FP32-state质量；短GSM8K可降到5/5.5有效bit，代价在5%相对accuracy内。

长GPQA/AIME decode比32K短输出RULER更敏感，随机rotation也可能放大反复舍入。单H100 vLLM固定batch TPOT收益随batch增加，且不含prefill；kernel的>2倍不是全模型>2倍，hybrid的KV随context增大还会稀释state带宽贡献。

**建议考虑复现INT8 state起点，按真实最长decode验证质量，再试低bit按head分配。**

### SHq：学习网格间距和偏移，比固定NF/整数codebook更稳

**项目 / 版本 / 日期：** SHq / Learnable Grids；`arxiv:2611.02966v1`，v1提交 **2026-11-02 19:35（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §2–4、Tables 1/2](https://arxiv.org/html/2611.02966v1)。

用两个参数控制companding间距与offset，并借E4M3 scale符号选择网格镜像，保持block lookup格式存储结构。在3B–35B五模型、500K held-out token、RTN/GPTQ及2–6bit对照中，3/4bit的output KL都更低；4.5bpw相对NVFP4的KL降41–67%。

这是可查表格式的质量改进；改变lookup值不代表能直接使用固定E2M1原生Tensor Core。部分混合精度只有单seed，低layer MSE不必保证低模型KL，KL下降也不等于同幅任务提升或吞吐提升；不宜按标题把它当现成NVFP4硬件格式替代。

**建议考虑复现已支持可学习lookup的量化栈，分别核对任务精度、scale搜索成本和kernel兼容性。**

### Jet-QSparse：NVFP4 QK与稀疏高精度V聚合联合配置

**项目 / 版本 / 日期：** Jet-QSparse；`arxiv:2611.01022v1`，v1提交 **2026-10-31 23:06（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §3–5、Table 1、Appendix A](https://arxiv.org/html/2611.01022v1)。

以紧凑NVFP4存KV并用原生低bit QK选连续block，再高精度聚合选中V。四Qwen/Nemotron模型的数学、代码和Agent评测中，128block预算未检测到accuracy下降；两Qwen在64block也未检测到下降。独立GB300性能负载中，Qwen3.8-27B、128K context、batch16、1024输出，TPOT为BF16的1.94倍，packed KV含metadata为0.289倍。

“未检测到”受三run与任务方差限制；生成评测与长context性能是分开运行的，不能把图上两轴当同一次Agent实跑。KV-pool可容token实测只增2.58–2.76倍，低于packed字节理论值；分布式证据聚合在激进稀疏下会受损，2.4T模型还受expert计算/通信限制。

**建议考虑复现，联合调precision与block预算，除needle还应测试跨段信息聚合。**

### CISA：不训练indexer的dense稀疏改造，省读流量而不省KV容量

**项目 / 版本 / 日期：** NVIDIA / CISA；`arxiv:2611.03200v1`，v1提交 **2026-11-02 22:25（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §2–4、Tables 1/2、Appendix B/D](https://arxiv.org/html/2611.03200v1)。

按16token cache页建代表key，在KV-group内选远程chunk，prefill相邻query共享选择QShare；直接传现有paged-attention页ID。Nemotron/Inkling-Small的1M上下文保留部分长任务质量，适合dense模型无梯度校准改造。

完整K/V仍驻留并多存代表，不能说KV压缩16倍。8×B300、TP8、concurrency16的Inkling固定4096输出吞吐1.54倍，按平均verbosity线性调整只余1.12；更激进配置实测1.64倍，调整proxy为0.93，proxy也非真实端到端仿真。Ultra的≥1M GraphWalks仍低dense4.62点；传统单query decode暂不能用QShare。

**建议考虑复现attention-bound长prompt服务，同时测生成长度与task completion时间。**

### Penelope：先淘汰私有历史前部，保留重算昂贵的后部KV

**项目 / 版本 / 日期：** Penelope / SGLang研究扩展；`arxiv:2611.00391v1`，v1提交 **2026-10-31 05:14（Asia/Shanghai），补录**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §3–5、Tables 1/2](https://arxiv.org/html/2611.00391v1)。

保留共享prefix及active请求，对LRU选中的私有历史做head-first block eviction；重算缺口时复用保留下来的suffix，并统一GPU/CPU顺序。Qwen3-8B/32B/235B、1–8×H100/A100、coding-agent trace回放，无新GPU kernel；GPU-only相对SGLang mean TTFT最高降60%、JCT37%，两级cache最高76%/60%。

比较人为限制KV容量以制造可控内存压力，最长prefix scheduler要计入suffix，transfer需要额外同步；这是论文中约2.5K行扩展，不能当上游SGLang已合入。收益取决于多轮私有历史复用，短独立请求未必同样受益。

**建议考虑复现Agent trace和真实cache压力，重点测重算token、host流量与尾延迟。**

### SpecOPD：分支并行蒸馏parallel drafter，独立处理线性/稀疏历史

**项目 / 版本 / 日期：** 阿里Qwen、USTC等 / SpecOPD；`arxiv:2611.02846v1`，v1提交 **2026-11-02 17:47（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §2–3、Table 1、Appendix A/B](https://arxiv.org/html/2611.02846v1)。

在reference anchor采独立draft分支并共享target前缀评分，full attention用tree mask，sparse用分支一致indexer分组，linear用indexed state replay；E2E-TV surrogate直接优化acceptance length。覆盖8B full、27B hybrid、125B sparse-linear，最高AL提升41.45%，相对sequential-rollout OPD监督token吞吐16.09倍，生成对AR最高3.56倍。

三个数字分母不同，不能相乘；16.09倍是27B/DFlashV2、8张未命名高性能GPU的原生监督配置，排除offline reference准备，并非到同AL的训练时间比。相对原DFlashV2，27B跨五负载几何平均吞吐只提高最多2.83%，SWE在concurrency32还慢4.46%；T=1的Terminal AL还降1.02%。SWE/Terminal是历史context续写效率而非Agent任务成功率。部分8B只写“高性能GPU”未披露型号，许多评估关闭CUDA Graph/cache/overlap，27B标准任务归档greedy输出不完全一致；lossless算法不自动保证所有浮点实现逐token一致。

**建议深入阅读分支历史隔离，再在目标SGLang配置中同时验证输出语义、AL和真实服务吞吐。**

### Uni-LaSpec：少量相关视觉token和随负载变化的验证树

**项目 / 版本 / 日期：** Unisound、哈工大等 / Uni-LaSpec；`arxiv:2611.01617v1`，v1提交 **2026-11-01 14:09（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §III–IV、Tables I/II、XI](https://arxiv.org/html/2611.01617v1)。

六层parallel drafter复用verifier缓存特征，默认从三层全注意力各取相关视觉token，运行时在draft tree与截短linear proposal之间分配验证位。六VLM/17benchmark中198/204格提升accepted length；8个视觉token已接近全图输入收益，selector<1ms。RTX A6000 burst负载相对linear验证mean E2E latency降47.6%。

视觉drafter用1M prompt、10epoch和额外dynamic-head训练，并非免训；固定负载general benchmark每项最多100prompt、输出1024，上下游需持有辅助层特征。硬件包括A6000/Ada、5090/A800，不能把各图最大值当同一硬件通用值；更长树提高acceptance也会在并发下拖慢吞吐。

**建议考虑复现线上视觉任务的并发曲线，别仅按单请求AL选择验证结构。**

### Scute：一个batched LoRA schedule精确容纳多类PEFT

**项目 / 版本 / 日期：** Thomas Walker、Richard Baraniuk / Scute；`arxiv:2611.01089v1`，v1提交 **2026-11-01 00:24（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §3–6、Tables 1–3](https://arxiv.org/html/2611.01089v1)。

把LoRA/DoRA/IA3/SSF/VeRA/OFT写入六slot normal form，识别可离线fold与必须runtime保留的scale/向量/旋转，再用三种执行路径共享base GEMM；避免把高rank VeRA/OFT近似转换成LoRA。vLLM0.19.1、Llama3.1-8B FP16、RTX A6000、CUDA Graph/compile开启，多method burst比LoRA-only多5–17% decode时间。

“精确”是代数语义；FP16相对PEFT的logprob误差为adapter效应的0.3–1.5%，任务差不超0.4点。serving成本使用shape相同synthetic tenants，OFT的base GEMM切分本身会提速，论文加匹配切分对照；不是已发布vLLM主线通用PEFT支持。

**建议考虑复现多租户PEFT平台；先校验放大adapter的数值，再测adapter池容量与请求分布。**

### LayerLoom：在每层forward旁插入可随SLO让路的LoRA backward

**项目 / 版本 / 日期：** LayerLoom / FlexLLM扩展；`arxiv:2611.00652v1`，v1提交 **2026-10-31 13:29（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §3–6、Table 1](https://arxiv.org/html/2611.00652v1)。

跨microbatch维持反向wavefront，使ready backward任务按inference正向层次出现；每层按剩余SLO预算决定执行数，配host offload/prefetch及共享base权重GEMM。单RTX4090、Llama3.1-8B FP16、down_proj rank16、128token microbatch×64累积，joint TTFT≤5s与TPOT≤168ms的达标率比FlexLLM高13.5–21.8点，FT吞吐相当或最高多12%。

细粒度调度会扩大paused activation/host内存，长期序列PCIe可能成瓶颈；只测一个GPU/模型和这组宽松SLO，未开Legion trace以公平比较，不代表服务图捕获下仍同收益。

**建议考虑复现确有零散闲置容量的单卡LoRA共服；按自身SLO和host带宽重做cost model。**

### Apto：共享PC上按实时争用动态分派MoE expert

**项目 / 版本 / 日期：** 上海交大 / Apto；`arxiv:2611.01859v1`，v1提交 **2026-11-01 20:50（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §4–6](https://arxiv.org/html/2611.01859v1)。

基于llama.cpp把expert task放进共享deque，按GPU cache状态、token数及激活频次排序，CPU/GPU/fetch线程从适合的一侧取任务并允许尾部竞跑，适应并行编译或GPU程序造成的资源波动。RTX5080 16GB＋Ultra9 285K/96GB DRAM、限制12CPU核，三MoE含Qwen3-30B-A3B，batch1/4、ShareGPT每请求48输出，并补测DGX Spark。同DS-2/QW-3、batch4加ResNet共载时，相对FineMoE的prefill/decode平均1.44/1.75倍且ResNet吞吐相近；相对KTransformers却仅77.06% decode吞吐，换1.51倍后台ResNet吞吐。

其价值是动态资源适应而非固定最优静态分割；只覆盖少量本地模型/短decode与特定编译、ResNet共载，UMA与离散PCIe的成本结构不同。论文未证明分派收益能转移到大规模EP服务，模型质量也不是本轮测量目标。

**建议考虑复现本地coding-agent与编译共载环境，测吞吐之外的交互延迟和后台任务完成时间。**

### 精确相对位置attention：按可达relation分块并归约梯度

**项目 / 版本 / 日期：** Fast and Exact Relative-Position Attention；`arxiv:2611.01182v1`，v1提交 **2026-11-01 02:29（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §3–6、Table 2](https://arxiv.org/html/2611.01182v1)。

统一Shaw、Transformer-XL、DeBERTa与T5的加法相对位置项，避免先构造dense bias；只投影可达关系、复用恒offset tile，并按offset拥有梯度，减少atomic冲突。H200、FP16、8K长度，针对文中最快生产实现的forward/backward加速分别为Shaw6.4/21倍、TXL1.8/2倍、DeBERTa3.7/7.1倍、T54.6/13倍。

这些是attention子系统相对保留dense-bias等具体baseline，不是整LLM倍数。TXL identity map的投影仍二次，改tile内算省内存会慢2–2.9倍；非连续offset map会回退FP32 atomic并失去确定性，常见RoPE模型不会直接受益。

**建议考虑复现仍使用这些relative-position机制的训练或语音模型，先确认baseline本身是否高效。**

### PagedMoE：把不同精度当同一个expert的可递进缓存状态

**项目 / 版本 / 日期：** PagedMoE；`arxiv:2611.03173v1`，v1提交 **2026-11-02 22:09（Asia/Shanghai），72h回溯首次纳入**。本轮由submitted-date API补充发现，未核实精确公开列示时刻。[固定全文 §4–8、Tables 1/2](https://arxiv.org/html/2611.03173v1)。

以prefix-compatible tree codebook＋bitplane存1–8bit expert，runtime路由后、I/O前按gate权重和offline误差分配精度，只从SSD补缺失bitplane，CPU fused kernel直接用compact prefix。Ryzen9950X＋RTX5090/64GB DRAM/PCIe5 SSD，routed expert在CPU执行，GPU只跑常驻部件；Qwen3.5-397B、24GiB expert cache、平均2.5bit达到8.89tok/s，相对同分配独立精度对象1.33倍、uniform4为1.56倍。

同expert请求/精度回放中byte hit47.94→68.50%、SSD读1.14→0.69GiB/token，拆开了“少bit”与“跨精度复用”。这不是全模型2.5bit，非expert保持BF16/FP8；GLM5.2 GSM8K相对MXFP4的93.40降到90.52，质量不能笼统无损，random calibration与gate-weight误差代理可能漏任务关键expert。

**建议考虑复现SSD-offload本地大MoE，检查I/O、CPU解码、实际模型质量和内存分母。**

## 训练算子与硬件协同

### FLAMIA：低bit attention除了快，还要验证scale没有偷看未来

**项目 / 版本 / 日期：** Yandex Research / FLAMIA；`arxiv:2611.03344v1`，v1提交 **2026-11-03 00:00（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §3–4、Tables 1/5](https://arxiv.org/html/2611.03344v1)。

把MXFP8和混合NVFP4/FP8的量化布局与转置复用、forward/backward表示一致性、packed-document调度一起设计。B200/B300、causal 16head×128维、含量化的forward+backward，测试负载相对最快baseline约1.06–1.42倍；例如B200的4×4096为0.89ms，对最快1.06ms。

更重要的纠正：causal mask挡不住跨未来token的共享scale；默认仍可残留32token内依赖及whole-sequence dQ缩放依赖，exact causal选项仅覆盖支持的batched配置，dQ安全scale约加8–11% backward开销。6层byte-GPT实验里泄漏kernel自评loss0.811–0.827，换exact attention评同权重变0.888–0.940，对无泄漏训练0.862。单步gradient cosine高也未能预警后续发散。

**建议深入阅读并复现future-token perturbation与独立exact-kernel验证，再考虑接入训练。**

### ReWY：先简化delta-rule反向图，再比较完整训练边界

**项目 / 版本 / 日期：** 天津大学等 / ReWY；`arxiv:2611.01940v1`，v1提交 **2026-11-01 22:21（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §2–4、Tables 2–5、Appendix G/K](https://arxiv.org/html/2611.01940v1)。

为DeltaNet/GDN/KDA/GDN2共享reverse sensitivity，移除inverse adjoint并把部分局部矩阵乘从7次降3次，结合matrix-intra及Gram消除。H100/H200、chunk64、head128、六shape几何均值，完整low-level backward加速约1.18–1.70倍；matched recompute/save，保留原FP32工作矩阵。

Ascend950DT的GDN2最高2.89倍含组合优化，不能单归ReWY。完整loss.backward的收益小于core，仍排除forward、optimizer等，且模型随机初始化/合成token，不是Kimi/Qwen完整部署。公开wrapper末尾dtype转换被排除，部分旧降精度路径只作诊断；KDA fallback还有FP32 atomic导致非bitwise确定性。

**建议考虑复现既有FLA/NPU kernel，严格对齐dtype、recompute策略和最终输出转换后再计训练步吞吐。**

### FlyWheel：TPU原生attention同时改布局、静态anchor和流水

**项目 / 版本 / 日期：** NYU等 / FlyWheel；`arxiv:2611.01426v1`，v1提交 **2026-11-01 10:07（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §3–5、Tables 1–3](https://arxiv.org/html/2611.01426v1)。

softmax以head-major布局提高KV复用，固定首tile anchor并检查累积和，失败在同launch内用guarded anchor重放；交错QK/softmax/PV。GDN另用BF16三角逆与跨chunk排程。v6e/v7x、Qwen3.8-27B，RULER512K整应用分别1.53/1.35倍，短4K在v7x反为0.98倍；RULER/BABILong与Video-MME质量近似。

GDN core的3.19–3.53倍不是整模型倍数。512K在v6e用16-chip/TP8×CP2，其余≤256K多用8-chip；v7x的device只是一颗chip的一个TensorCore，不能按device数直接跨代比。JAX0.11/libtpu0.0.44、vLLM/tpu-inference固定论文baseline，非声明上游已发布。

**建议考虑复现TPU长prefill服务，包含projection/layout成本、回退率及短请求回归。**

### CLiS：用可证安全的固定stabilizer替代反复在线softmax归一化

**项目 / 版本 / 日期：** CLiS / TPU Attention；`arxiv:2611.00444v1`，v1提交 **2026-10-31 06:57（Asia/Shanghai），补录**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §4–6、Tables 1/2](https://arxiv.org/html/2611.00444v1)。

先为row选loop-invariant stabilizer、按head认证数值范围，失败回退online softmax；在同Pallas框架内比较移植的FD++/VFA/FA4式稳定机制。Wan2.2/LTX2、v6e-8/TPU7x-8、CFG2×Ring4，kernel延迟最高降24%、DiT step最高降19.6%，计预处理、安全检查和重算。

与FlyWheel聚焦LLM不同，主证据是视频diffusion；因对LLM训练也给初步continued-pretraining13.5%、SFT7.7%吞吐而纳入。720p81frame例中约1.78%位置回退，若固定anchor硬跑其中23site会NaN，不能删掉fallback再宣传同质量提速。style baseline仅统一TPU实现内稳定策略，非各原GPU系统整套性能对比。

**建议继续观察LLM训练证据；若移植固定anchor优化，安全认证与恢复必须保留。**

### Sonar-MoE：batched专家optimizer-in-backward省显存，提速需拆掉裁剪差异

**项目 / 版本 / 日期：** Sonar-MoE；`arxiv:2611.03577v1`，v1提交 **2026-11-03 02:30（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §4–6、Tables 1–3](https://arxiv.org/html/2611.03577v1)。

对每层两个batched expert tensor挂post-accumulate hook，完成梯度即fused AdamW并释放，避免全模型expert梯度常驻；按expert逐个hook会大幅增加调用开销。RTX PRO6000 Blackwell96GB、OLMoE1B-7B/Granite3B、Alpaca2000步、batch2、BF16权重/moment无master，OLMoE/SonicMoE峰值55.55→43.57GB（-21.6%）。

相对global-clip基线142.8→116.1ms，看似18.7%提速；去掉expert clip的普通基线已115.9ms，主要速度收益不是early release本身。更新只与“无expert裁剪”基线bitwise相同；无microbatch累积才可立即释放，累积保留收益受限，标准all-parameter optimizer-in-backward也达到相近或更低显存，不能称新算法全面占优。

**建议考虑采用batched hook粒度，但先确认可放弃global expert clipping及跨microbatch累积要求。**

### RankLane：静态MoE容量跟随负载名次，而非固定expert身份

**项目 / 版本 / 日期：** Qualcomm Cloud AI100 / RankLane；`arxiv:2611.00421v1`，v1提交 **2026-10-31 06:14（Asia/Shanghai），补录**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §3–5、Tables 1–3](https://arxiv.org/html/2611.00421v1)。

把每SoC专家按实时token load排序，再填入编译期固定大小lane；runtime只改索引，shape/placement保持静态；按token在本地core完成聚合，超容量chunk回退full-capacity重跑。四SoC单Cloud AI100 Ultra卡、MXFP6、batch1、128–2048token prefill，Qwen3-30B-A3B为dense固定capacity基线的1.27–4.18倍，Gemma4-26B-A4B为1.93–3.46倍。

基线“dense”指MoE dispatch/aggregation的稠密保留容量，而非换成同质量dense模型。性能计overflow检测和恢复支持，但容量校准与执行plan依赖特定编译器/芯片；无法直接套用GPU动态grouped-GEMM，也未证明decode/线上混合batch同收益。

**建议深入阅读静态图MoE排程，复现时保留held-out路由与overflow恢复的端到端成本。**

### Galaxy集群效率：便宜芯片需要快互联才能兑现每美元优势

**项目 / 版本 / 日期：** Tenstorrent / Cluster Efficiency；`arxiv:2611.02975v1`，v1提交 **2026-11-02 19:43（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §4–6、Appendix](https://arxiv.org/html/2611.02975v1)。

比较32×32GB Blackhole Galaxy与8×96GB RTX PRO6000 Blackwell server，三Llama/Qwen模型、BF16和8bit、8K–256K context，36配置的tokens/美元为1.07–2.24倍，tokens/J为1.00–2.27倍；在相同Galaxy芯片/批量上模拟TCP级collective后，美元优势全部消失。

这是整机不同总内存/芯片数与售价的capacity-aware比较，非同芯片算术快多少；8bit两边BFP8/FP8格式不同。功率分别来自core-rail与GPU-board遥测，不能当完整机架墙插能耗；TCP路径是native fabric上加成本的模拟，而非实际廉价以太网集群。采购价格与配置会改变结论。

**建议深入阅读互联与容量的因果对照；采购前须按自身上下文并发、整机功耗和实时价格重算。**

### Coffee：预算内动态精度，需要执行和存储一起可变

**项目 / 版本 / 日期：** Coffee / Elastic Precision；`arxiv:2611.00915v1`，v1提交 **2026-10-31 21:17（Asia/Shanghai），72h回溯首次纳入**。本轮由submitted-date API补充发现，未核实精确公开列示时刻。[固定全文 §4–6、Tables 1–3](https://arxiv.org/html/2611.00915v1)。

对当前两输入联合估计量化误差，offline分预算、online分bit，配nested读与reconstruction、专用mixed-precision阵列，并按attention重要性分KV精度。Llama3-70B平均4bit的六任务均分79.23，对FP16 79.95、PLENA74.37；128K RULER多needle也比uniform低bit更稳。

硬件是ASAP7/1GHz综合与288加速器系统模拟，匹配四B200的估算乘法器数、32TB/s带宽和可用KV容量；GPU基线实测，但不是已经造出的芯片对GPU实跑。最长context的最高4.96倍throughput/12.56倍能效依赖模拟功耗/通信/最大可装batch，短context能效反比PLENA低5–9%，不能据此作采购结论。

**建议继续观察；架构研究可复现误差与bandwidth/metadata账，待真实硬件验证系统收益。**

### PUD-LLM：借层间残差冗余暴露DRAM内计算的并行机会

**项目 / 版本 / 日期：** PUD-LLM；`arxiv:2611.02401v1`，v1提交 **2026-11-02 10:22（Asia/Shanghai），72h回溯首次纳入**。本轮由submitted-date API补充发现，未核实精确公开列示时刻。[固定全文 §3–5](https://arxiv.org/html/2611.02401v1)。

将相邻投影拆成可lookahead的主项与较小残差项，对后者用更少有效项近似，配SIMD packing/MIMD pipeline。十Llama/Mistral/Qwen/Gemma模型、4bit量化参考、WikiText2 perplexity增加≤5%的选点，Ramulator2模拟中相对PUD baseline几何均值1.17/1.18倍，最高Gemma12B modified-DRAM为1.41倍。

该“end-to-end decode”只计Transformer blocks，排除prefill、embedding、lm_head、sampling；batch1、context128/1024、DDR4-2400与host85%峰值带宽是假设，非真实部署。小残差允许粗近似的误差模型不保证长程任务，CPU剩余算子仍限制总加速，尚未扩展验证MoE。

**建议继续观察内存计算方向；先核quality约束、遗漏算子与完整系统成本，再考虑采用判断。**

## 评估、可靠性与机制诊断

### FlashProve补录：证明过哪套整数计算，比“110 token/s”更重要

**Anthropic / FlashProve，官网标2026-11-02；固定代码`9e2bdd8ff5bf9c62f18263d7dac189b0400b260c`的初始提交为10月30日20:56 UTC，README内部版本0.16.0、提交说明0.16.0-rc.26。补录，不作11月4日首发。** 面向MoE推理先commit权重，再按实际激活expert对输入和已完成输出作post-hoc证明，segment streaming让400B Maverick的4bit整数版本在单H10080GB＋24CPU核上运行；完整真实权重示例还要求至少330GB可用host RAM和245GB磁盘（计时public-weight steady host peak269.7GB、commit299.6GB）。代码、协议与Lean材料Apache2.0，Maverick权重另受Meta许可；仓库明确为不维护的研究工件。[官网](https://www.anthropic.com/research/provable-inference)，[固定报告§1.3/7–8](https://github.com/anthropics/FlashProve/blob/9e2bdd8ff5bf9c62f18263d7dac189b0400b260c/paper/paper.pdf)，[性能口径](https://github.com/anthropics/FlashProve/blob/9e2bdd8ff5bf9c62f18263d7dac189b0400b260c/guide/performance.md)。

报告的111.23±0.92 token/s（三run）是batch24、每request1024 **prompt token的证明吞吐**，每proof验证一个下一token；不是每秒生成110个已证明response token。约18.74–18.81MB/proof，C++单线程检查约21.14s，启动/验证不计入稳态率，首次commit约39分钟。对8H100 FP8 vLLM按GPU折算约93倍证明开销，格式、资源和计算边界不同，非原生inference吞吐公平竞赛。

**限制比博客摘要更实质。** Exact-match承诺整数模型greedy；216文的整数top1相对16bit浮点参考差6.81%，自然段子集9.41%，且无task-quality benchmark。两个stock vLLM各96条reply在Python参考scoring下均不满足exact match，不能直接给任意浮点/采样API附上同样保证。Lean soundness覆盖Lean verifier/协议，C++与CUDA代码对应关系是测试而非证明；batch completeness/ZK、seed派生mask、真实权重证明验证及若干参数条件有明示缺口，协议还披露每request一个weight-dependent bit，不能泛化为整套运行程序已证明零泄露。[已证与未证范围](https://github.com/anthropics/FlashProve/blob/9e2bdd8ff5bf9c62f18263d7dac189b0400b260c/guide/what-is-proved.md)。

建议深入阅读并从固定proof的CPU verifier开始考虑复现，优先审核relation、参数和权重commit信任来源；目前适合作为可验证推理研究原型，不宜直接宣传为现有模型服务的完整认证。

### XLA:TPU静默误编译：把真实autotune搜索变成差分验证

**项目 / 版本 / 日期：** Google / XLA from the Trenches；`arxiv:2611.00624v1`，v1提交 **2026-10-31 12:48（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §3–5、Table 1](https://arxiv.org/html/2611.00624v1)。

结合typed Jaxpr合成、生产autotune旁路差分、HLO→LLO→kernel逐级isolation及人工审核的Agent修复。论文报告在Google生产fleet定位并推动解决142项compiler bug，另103项fix仍在进行；已解决样本43%在TensorCore低层emitters、25%在HLO rewrite。

数字是作者报告的工作累计，不是本轮新release的142条上游修复。它针对静默数值错误，而不是“编译通过即正确”；真实fleet图和完整bug集合没有在本轮独立复验，统计分布受发现策略偏向影响。对每个AI patch仍需人审、ISA/语义约束和adversarial复核，不能把swarm产出直接当正确性证明。

**建议深入阅读，将已有性能搜索与差分correctness check结合，比只加静态unit test更有现实价值。**

### 同一输出不同分：解析失败fallback也会制造benchmark优势

**项目 / 版本 / 日期：** Nebius、独立作者 / Grader Audit；`arxiv:2611.03377v1`，v1提交 **2026-11-03 00:20（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §3–7、Tables 1/2](https://arxiv.org/html/2611.03377v1)。

审MMAD/OmniMedVQA/PMC-VQA/MMMU/ScienceQA/POPE/MathVista等released scorer，区分阈值无约束fuzzy snap、随机选项、默认class和拒判。五个7–12B模型、4bit、16GB GPU，在MMAD完整1670题中，把parse失败snap改为abstain，部分模型分数差17–22点；64token generation预算下artifact仍存在。

fallback大小在其他benchmark常只有1–5点，论文未发现只改fallback就有稳健排名反转；改完整parser可改变排名，但不是普遍结论。人类ground truth以binary MMAD最强，PMC-VQA只有60条人工audit；不能把所有正常format错误都归为奖励投机。

**建议考虑采用：发布parser版本、失败比例与fallback贡献，并报告abstain等敏感性分析。**

### StateBench：长context能找needle，不等于能追踪多次状态更新

**项目 / 版本 / 日期：** Ai2、NYU等 / StateBench；`arxiv:2611.03094v1`，v1提交 **2026-11-02 21:12（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §3–6、Appendix C](https://arxiv.org/html/2611.03094v1)。

按有限monoid层级区分不可变赋值、可变赋值、自引用与general finite四类state tracking，拆开理论表达力、直接训练和预训练模型实际能力。六类275M架构训练8–256次赋值、测试至64000；GDN/Hybrid的length generalization通常强于Transformer，但直接训出的可表示能力没有自动出现在现有模型里。

预训练评估包括7B OLMo3与Hybrid及closed frontier、多种文本格式；甚至soluble层也未稳健解决。报六format中的最高值，比单prompt保守失败诊断更宽容；理论对正/负eigenvalue和精度等假设敏感，不能把synthetic benchmark等同真实软件state完整正确性。Ai2研究页另标11/3发布，论文v1提交11/2。

**建议深入阅读并考虑复现长程工具状态/变量更新评测，补足只测RULER或静态recall的盲区。**

### Hybrid检索中的linear层也会构造query，不只是压缩历史

**项目 / 版本 / 日期：** HKUST、港理工、CUHK / LinearQuery；`arxiv:2611.01909v1`，v1提交 **2026-11-01 21:50（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §3–5](https://arxiv.org/html/2611.01909v1)。

在Qwen3.5 GDN与Granite4 Mamba2中定位Encode–Address circuit：早层留token身份，Address recurrent block构造最终softmax reader查询。跨不同key的prompt移植该表示，可在0.97–0.99比例重定向检索；把Qwen Address计算替进Granite可恢复0.92–0.94准确率，不更新权重。规模增大后该计算分布到多组block/reader。

证据来自受控associative retrieval与activation干预，未证明自然语言所有reasoning都遵循相同电路；可移植计算不等于任意模型state可互换。它纠正“linear层仅作无关压缩，检索全靠softmax”的过度简化，但不构成移除某类层的性能处方。

**建议深入阅读，压缩/量化Hybrid state时优先检查对query construction的影响。**

### 量化coding-agent：终局accuracy近似，也可能改变解题轨迹

**项目 / 版本 / 日期：** Traversaal AI、Kansas State等；`arxiv:2611.02368v1`，v1提交 **2026-11-02 09:30（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §3–5、Table 1](https://arxiv.org/html/2611.02368v1)。

Qwen2.5-Coder7/14/32B与Devstral-Small-2507，在同vLLM工具schema的OJBench上比较BF16、4bit AWQ/GPTQ、三seed。多重比较校正后未见显著accuracy损失，checkpoint约缩3.4倍；Qwen7B却从one-shot更多转向执行反馈loop，模型间failure profile改变不同。

115道Easy/Medium的overall baseline很低（如Qwen32B13.91%），Hard因全0被排除，统计power有限。“不显著”不是证明等效；论文未测整任务latency/能耗，因此不能依据存储压缩推出Agent成本节省。

**建议考虑采用轨迹/工具调用/失败类型联合评估，并补真实任务成本后再选低bit版本。**

### Kerkoporta：保住常规utility的权重后门，仍能劫持结构化工具输出

**项目 / 版本 / 日期：** Northeastern、Microsoft Research等 / Kerkoporta；`arxiv:2611.00138v1`，v1提交 **2026-10-30 02:41（Asia/Shanghai），补录**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §III–IV、Tables I/II](https://arxiv.org/html/2611.00138v1)。

在reasoning coding model的SFT数据植入触发—工具调用轨迹，再用轻量RL恢复普通任务utility，受控测试可跨repo/harness或限定特定schema。Qwen3.5-4B/9B、25%/50% poisoning等设置中部分OOD attack success达94.8%，另有设置最高96.2%；去掉poison样本reasoning trace后OOD ASR接近0，显示表示形式而非只加关键词起作用。

这是能改模型权重/训练数据的供应链攻击，不是普通未可信网页prompt injection；25–50%毒化等配置成本高，最大成功率不能当现实基率。恢复benchmark utility不能证明模型安全，false trigger小样本也不足以保证实际零误触。

**建议深入阅读防御评估；采用外部衍生权重时加入结构化工具行为测试和权重来源核验。**

### CrossSWE：跨仓库共同修改，揭示单仓库评估漏掉的一致性失败

**项目 / 版本 / 日期：** CrossSWE；`arxiv:2611.01484v1`，v1提交 **2026-11-01 11:16（Asia/Shanghai），72h回溯首次纳入**。本轮由submitted-date API补充发现，未核实精确公开列示时刻。[固定全文 §3–5、Tables 1/2、Limitations](https://arxiv.org/html/2611.01484v1)。

171真实任务、165仓库、13语言、20技术域，要求多个repo最终状态共同通过测试。统一mini-SWE-agent2.4.6、high effort、300步/$10上限，五模型最高Pass@1为58.48%；同问题只需改单侧的对照更容易，GPT6.1 Sol为64.55%，DeepSeek V4.1 Flash为61.81%，分别高6.07/9.18点。

165任务只含2仓库，≥3仓库仅6个；参与repo直接提供，未测试仓库发现/无关repo干扰。一次通过率不含多seed误差，模型排名受同一bash harness与预算影响；输入百万token常含cache/repeated context，不能直接等同unique文档大小。

**建议考虑采用跨repo契约一致性测试，尤其是SDK、服务端、schema需联动修改的agent场景。**

### Harness也会遗忘：新任务自改代码可破坏旧任务验证与流程

**项目 / 版本 / 日期：** Continual Harness Self-Evolution；`arxiv:2611.03219v1`，v1提交 **2026-11-02 22:43（Asia/Shanghai），72h回溯首次纳入**。本轮取得的new列表标11月3日，列示日不冒充提交日。[固定全文 §3–4、Table 1](https://arxiv.org/html/2611.03219v1)。

在Mini-SWE-agent上适配AHE与Meta-Harness，顺序演化repair、test-generation、feature-development三类任务，观察六个迁移方向。遗忘来自verification、workflow和budget三类代码/策略冲突；人工受控冲突例中，原best harness的SWE分96%可因验证冲突跌到0，workflow冲突降66%。

演化集仅50/50/30题，best harness也按这些小集选择，96→0不是全500题正式benchmark退化。方法是作者对原代码的适配而非原方法全部官方设置，二任务链并不足以证明任何长任务序列必然遗忘。其价值是提供harness级回归对象，而不是宣称模型权重遗忘。

**建议考虑采用跨任务harness regression，在自动自改工具协议或验证逻辑后重跑旧任务。**

## 开源工程架构报告



<a id="core"></a>

## 核心开源工程：训练、推理与平台增量

主窗 `2026-11-03T00:58:48Z` → `2026-11-04T01:05:04Z`；72小时回溯自 `2026-10-31T00:58:48Z`。来源为六个项目官方 GitHub。各条以合入 SHA 冻结，分清 main、release 回补与集成分支；本轮未在设备上复跑，实验数字均为作者陈述，正文给出可比较条件与边界。

### NVFP4 QAT 改为 per-token activation，并移除两条旧接口

QAT 按每个 token 的 amax 量化激活，替代跨步累积的静态全张量 scale；fake quant 同时匹配 FlashInfer per-token 舍入与 trainer 导出。NVFP4 rollout 默认使用 nvfp4_per_token。移除旧 SGLang 静态 NVFP4 refit 与此前已不可达的 FSDP QAT，当前 QAT 要求 Megatron。

**证据与局限：**作者 Qwen3-30B-A3B-Base、DAPO-Math-17k、32×GB200、100步，首2/末4层BF16，平均 k3_kl .060→.0033、logprob差 .124→.021，与原生NVFP4一致；这是训推失配指标，不能代替长期reward/泛化。另一个Qwen3.5-35B-A3B、EP8/vLLM TP2、3次中位的对齐脚本采用TE2.20.2/ModelOpt0.47，不能自动归到上述100步训练。per-token SGLang仍未支持。
**实践建议：**考虑复现：迁移先审breaking字段与后端，量化fast-math环境变量必须两端一致；用相同rollout逐token核KL，而非只比较最终reward。
源码路线：`verl/utils/qat/nvfp4.py::fake_quantize_nvfp4_per_token；qat/modelopt.py；compressed_tensors.py；workers/config/rollout.py`。
一手来源：[verl-project/verl #8494](https://github.com/verl-project/verl/pull/8494)（main，2026-11-03T14:38:11+08:00，`cd04d68d6e435e28c60c79b912ba718b494f889f`）。

### Grouped TileLang QAT/export 避免把专家权重先 stack

FP8、MXFP8、MXFP4 grouped kernels 通过指针表直接量化 TE 的分散专家；Megatron-Bridge 3D融合专家一次量化后拆成rollout的每expert wire格式，支持 w1/w2/w3 与 gate/up/down 命名，native vLLM Fp8Config 与 compressed-tensors 共用配方。

**证据与局限：**这是昨天共享量化kernel后的grouped增量；最终diff扩展专家target匹配并处理3D输出，不是NVFP4 grouped kernel。PR提供数值/导出测试，未给可比较的整步时延、显存峰值或训练质量实验。
**实践建议：**考虑复现：同时测专家数增长时的临时allocation和export时间，逐expert比较packed bytes/scales；不能仅凭减少stack推算整步加速。
源码路线：`verl/utils/qat/fp4.py；fp8.py；modelopt.py；compressed_tensors.py::_unpack_compressed_experts`。
一手来源：[verl-project/verl #8500](https://github.com/verl-project/verl/pull/8500)（main，2026-11-03T10:30:46+08:00，`e2ace870eba5e4cb850e839655f7be524a8fbe54`）。

### TorchStore 权重同步分别去掉 trainer 与 generator 的 CPU 整份副本

trainer把live GPU state_dict在线程内设置并同步push stream后直接put，SHM传输端完成cast，删除pinned staging；generator新增model_state_dict_load_mode=gpu_fetch，逐FSDP组直接fetch到GPU，失败后仍参加所有unshard并统一报错。默认cpu_prefetch保留。

**证据与局限：**作者GB200本地SHM、16GiB fp32 shard→BF16：push304→66ms、client RSS22.54→13.46GiB。Qwen3-8B generator RSS24.7→3.8–3.9GiB，但sync由可与生成重叠的.48s改为暂停engine的1.1–1.2s。gpu_fetch拒绝量化参数；非SHM仍可能逐batch建GPU cast副本，不能泛称零拷贝。依赖TorchStore#266/#252。后续#5774又去掉保留精度参数的临时dtype staging。
**实践建议：**深入阅读：按CPU内存预算选择prefetch或gpu_fetch，实测暂停占比与多机transport；保持push N完成先于optimizer N+1。
源码路线：`torchtitan/rl/trainer/trainer.py::push_model_state_dict/_put_model_state_dict；rl/generator/vllm/model_wrapper.py::load_model_state_dict；generator.py`。
一手来源：[pytorch/torchtitan #5659](https://github.com/pytorch/torchtitan/pull/5659)（main，2026-11-04T01:04:40+08:00，`411ce5b24cceb45ff83e078e1c5fa9927e6a8fc3`）；[pytorch/torchtitan #5652](https://github.com/pytorch/torchtitan/pull/5652)（main，2026-11-04T02:42:07+08:00，`c5c112d82e8d369a76368a62282b44718f1a42db`）。

### 训练、生成、checkpoint 统一 precision 与敏感参数 dtype 合同

dtype移至根PrecisionConfig，CPU offload移至ParallelismConfig，context长度仅由model配置提供。Module.Config.preserve_dtype_params按架构列出GDN/KDA衰减、DSv4 hc/ape等参数，FSDP all-gather、RL传输和export均保留较宽dtype；generator镜像trainer precision，删除独立model_dtype并以swap_tensors处理量化子类。

**证据与局限：**FP32 master路径会改变此前被BF16舍入的数值，BF16 master不会自动升为FP32。GraphTrainer SimpleFSDP遇保留参数报错；尚未接FlexShard。#5776本身是配置迁移，作者2GB300三类20步bitwise；其PP/FP32 kernel/CPU offload验证受环境限制。#5779另修两个原来不能启动的RL recipe，不能把全栈变化称纯重命名。
**实践建议：**深入阅读：迁移training.dtype/mixed_precision_*与generator.model_dtype字段，固定HF权重核对关键参数从build到传输再到导出的dtype和值；旧checkpoint不因新合同恢复已损失精度。
源码路线：`torchtitan/models/base/module.py::named_preserve_dtype_params；distributed/fsdp.py::mixed_precision_policy；rl/trainer/trainer.py::_preserve_dtype_keys；components/checkpointer/base.py::export_state_dict`。
一手来源：[pytorch/torchtitan #5776](https://github.com/pytorch/torchtitan/pull/5776)（main，2026-11-04T08:21:57+08:00，`e6ca6d86171a68ec0d5294c4e8212d21ee5b5990`）；[pytorch/torchtitan #5774](https://github.com/pytorch/torchtitan/pull/5774)（main，2026-11-04T08:25:15+08:00，`a3e458c735af12ba4eca4cd098c7ce8e32aeac5f`）；[pytorch/torchtitan #5779](https://github.com/pytorch/torchtitan/pull/5779)（main，2026-11-04T08:25:46+08:00，`e2b598851e818cfb7062ac8a6af8339b7cee16dc`）。

### GDN 7/7 已合入：按 requires_grad 编译两套 in_proj split

昨天仍open的第7项本日合入main。训练compile用cat limits=0和combo，把六路梯度直接写回各slice；no_grad仅用combo合成一次launch，两套函数共享region名但独立graph cache，依据projected.requires_grad选择。

**证据与局限：**作者GB300、单层per-GPU TP形状、FullActivationRemat：27B TP2 16k/32k fwd+bwd均-3.7%；27B TP4 decode64每层约-3.9µs。输出/输入梯度/state bitwise；未做完整RL-step、CP、64k训练或模型级冷编译。逐层乘48得整步收益只是作者估算，不能当实测。
**实践建议：**考虑复现：开启gdn_split_in_proj region再测，batch-invariant recipe可能清空regions；同时检查no_grad与remat走正确compile。
源码路线：`torchtitan/models/common/attention/gdn.py::_split_in_proj_with_grad/_split_in_proj_no_grad/_split_in_proj`。
一手来源：[pytorch/torchtitan #5731](https://github.com/pytorch/torchtitan/pull/5731)（main，2026-11-03T11:19:36+08:00，`cfef1691d0a68b7e9ec7b84bf37ef637c27fbd22`）。

### TitanRL 新增 Monarch 的 SLURM 多节点 launcher

把“RL recipe＋集群布局”放入MonarchControllerWithLauncher.Config；submit先规划trainer/generator host mesh，由SlurmLauncher提交普通rl.train，按节点放置角色，节点本地JIT cache可配置。删除旧setup_async(actor_system=...)入口。

**证据与局限：**作者两台4卡GB300，Qwen3-.6B GRPO，TP2×FSDP2 trainer与四个单卡generator分开两节点，10步每步sync、exit0；更大模型/调度器未据此验证。布局单测通过不代表任意SLURM站点配置可直接运行。
**实践建议：**考虑复现：集群运维可先跑小模型，核对空闲GPU、RDMA环境、head-node可达性与站点slurm_args，再扩展资源。
源码路线：`torchtitan/rl/actors/monarch/launcher/base.py::_layout；slurm.py；submit.py；controller.py::_build_actor_system`。
一手来源：[pytorch/torchtitan #5572](https://github.com/pytorch/torchtitan/pull/5572)（main，2026-11-04T04:32:21+08:00，`0f34815372413b72279080191978bb55a45c9ee7`）。

### HybridEP 默认 remat 留住通信结果，避免额外重放 dispatch/combine

将两段HybridEP通信放入同一ep_communication策略；默认重计算专家但保留forward通信，full remat仍重放。为tensor-only torch_remat0.2.2用CPU tensor ID引用handle并在GC释放，FakeTensor lookup避免host scalar读。

**证据与局限：**作者两GB300、FSDP2×EP2、10步loss/grad_norm bitwise；三个GPU remat/graph用例验证通信次数。正常forward与梯度通信仍存在，不能说移除all-to-all；上游opaque输出支持仍待后续。
**实践建议：**深入阅读：定制重计算策略时分别计forward、backward和replay通信，不把“保存通信”理解为保存整个MoE激活。
源码路线：`torchtitan/distributed/deepep/hybridep.py；models/common/moe/token_dispatcher.py::HybridEPTokenDispatcher`。
一手来源：[pytorch/torchtitan #5770](https://github.com/pytorch/torchtitan/pull/5770)（main，2026-11-04T03:11:33+08:00，`a87db08ea73cbdd567ceb055bd9118fd4a4507b2`）。

### Muse Spark 修复 norm region 在 remat 与 RL 下的编译失败

编译区域把各层不同gain_center从被特化的Python float改为FP32 CPU scalar tensor输入，避免forward与recompute匹配到保存tensor数不同的graph而CheckpointError；另把rmsnorm region重编译上限16→32，覆盖RL no_grad/训练及token数变化的22类graph。

**证据与局限：**作者4GB300、4层debug、FSDP2×TP2×EP2、10步：修后remat可训练，合法基线loss/grad_norm bitwise；算子级graph未保证与旧graph逐位。上限提升仅缓解特化数量，不是根除recompile或任意RL配置保证。
**实践建议：**考虑采用：复现时保留regions；CI的SPMD typecheck会清空regions，单看该CI容易漏掉故障。
源码路线：`torchtitan/models/muse_spark/model.py::MuseSparkRMSNorm._get_effective_weight/forward`。
一手来源：[pytorch/torchtitan #5713](https://github.com/pytorch/torchtitan/pull/5713)（main，2026-11-04T02:21:04+08:00，`a1a408f7881cf8a93d3e9b0281fe9aee656a43be`）；[pytorch/torchtitan #5733](https://github.com/pytorch/torchtitan/pull/5733)（main，2026-11-03T09:53:26+08:00，`3dc8d23d591ae727a74d11b463654eb89085dd95`）。

### QK-Clip 按 MLA 权重行缩放，允许 FSDP 切开一个 head

删除把Shard(0)局部权重view成完整head的要求，展开head scale成每行replicated scale，让DTensor在FSDP32/64且仅16head时仍能正确相乘，缩放过程不增加collective。

**证据与局限：**作者CPU gloo3/4rank对dense基线逐位相等，CommDebugMode无额外通信；AWS Neuron Moonlight16B-A3B text tower FSDP64十步成功。V行乘1也有额外工作，QK-Clip约.04s/step，不能称免费。
**实践建议：**考虑采用：大FSDP、小head模型尤其相关，回归NoPE/RoPE/V各段scale与已有DistMuon布局。
源码路线：`torchtitan/models/kimi_k2_7/qk_clip.py::_replicated_scales/_scale_mla_heads`。
一手来源：[pytorch/torchtitan #5737](https://github.com/pytorch/torchtitan/pull/5737)（main，2026-11-03T12:47:47+08:00，`196b4a5fcfb32508d0b3db6f0be6c8a9f4bb030b`）。

### 最后一步 model-only save 移入 exports，自动恢复不再误选

两种checkpoint后端统一把非零step model-only最终保存写到exports/step-N，自动恢复只扫描外层step-N；seed step0仍原位。Flux推理显式加载export，找不到则拒绝在初始化权重上运行。

**证据与局限：**作者169项CPU检查通过，未跑GPU HF往返/Flux集成。旧版本留下的model-only step-N仍可能被自动选中，exports不受keep_latest_k清理，已有training目录中可加载step-N仍优先于initial_load_path。
**实践建议：**考虑采用：先整理旧目录，推理使用独立dump folder与明确export路径；不把新目录约定当作历史数据迁移。
源码路线：`torchtitan/components/checkpointer/base.py::_create_last_step_id；dcp.py；torch_checkpointing.py；models/flux/inference/infer.py`。
一手来源：[pytorch/torchtitan #5748](https://github.com/pytorch/torchtitan/pull/5748)（main，2026-11-03T10:47:25+08:00，`1b59a2cc54e868b155abb341abf4101ef46e371d`）。

### 默认 HF tokenizer 恢复 chat template 中 BOS/EOS

apply_chat_template传入bos_token/eos_token（caller值优先），encode遇已以BOS开头的文本不再重复prepend，补齐多轮模板中的EOS。实验HFBackendTokenizer删去重复override。

**证据与局限：**这是SFT token序列的实际变化：依赖这两个变量的模板此前Jinja Undefined渲染空串。只传BOS/EOS，没有自动支持整个special-token map；不能将旧loss回归差异视为无关噪声。
**实践建议：**考虑采用：重新dump多轮文本、token IDs与answer-only labels，旧新SFT结果不能在忽略tokenization变化时直接比较。
源码路线：`torchtitan/components/tokenizer.py::HuggingFaceTokenizer.apply_chat_template/encode`。
一手来源：[pytorch/torchtitan #5757](https://github.com/pytorch/torchtitan/pull/5757)（main，2026-11-04T04:21:32+08:00，`89948030189159607e4c04699a34763a0f252926`）。

### Math-Verify 把验证器故障与答错分开记录

score返回MathVerifyResult(score,status)，status区分scored/timed_out/worker_died/invalid_output；rubric记录math_verify计数，非法worker输出也kill/wait清理，Verifiers适配器显式提取score。

**证据与局限：**基础设施故障仍可能贡献0 reward，但现在能看见来源；没有改变数学验证算法或证明所有0分可靠。最终diff覆盖所有调用者与回归，未提供完整训练收益。
**实践建议：**建议采用：监控timeout/worker death占比后再解释reward趋势或筛除样本，避免将验证器健康度误作能力下降。
源码路线：`torchtitan/rl/examples/dapo_math/math_verify_pool.py::MathVerifyResult/MathVerifyPool.score；rubric.py`。
一手来源：[pytorch/torchtitan #5741](https://github.com/pytorch/torchtitan/pull/5741)（main，2026-11-04T08:26:18+08:00，`8e3da8b50b39dd6239ca1ac1bb25172ac41133ec`）。

### SDC replay 用 int64 view 比较完整64位 digest

torch.hash_tensor结果view为int64，empty/None digest亦用int64，解决Ascend不支持uint64比较；不截断hash、不改哈希算法。

**证据与局限：**作者单rank Llama3 debug、100步、每步replay，与无replay同seed checkpoint比较loss/grad_norm/最终参数与RNG一致，注入梯度错仍报SDCReplayMismatch。单rank验证不能替代分布式故障注入。
**实践建议：**考虑采用：在NPU开启replay时先做可控错注入，确认hash能比较并不等于能覆盖全部SDC。
源码路线：`torchtitan/observability/sdc_replayer.py::_hash_tensor`。
一手来源：[pytorch/torchtitan #5769](https://github.com/pytorch/torchtitan/pull/5769)（main，2026-11-03T17:54:34+08:00，`6328f1e8c75ed716f74af8f1a84e1e843edda0d4`）。

### XCCL pipeline 暂停使用 per-edge P2P process-group split

按device默认backend判断，XCCL下PP>1不启用pipeline_per_edge_p2p，绕过后端无法split有向边process group；NCCL维持原路径。

**证据与局限：**作者声称内部XPU测试通过，但正文也保留未收集数值/性能的说明；没有公开可比完整训练配置与测量，属于兼容性gate。
**实践建议：**继续观察：XPU PP用户回归训练与通讯顺序，等待上游split支持后再恢复，不推断其他backend也适用此限制。
源码路线：`torchtitan/distributed/utils.py::init_distributed`。
一手来源：[pytorch/torchtitan #5750](https://github.com/pytorch/torchtitan/pull/5750)（main，2026-11-03T11:29:17+08:00，`1e948f5f819bd0a2cb314ffacef3e11d286d53bc`）。

### MInf 提供 token 级 live-request 快照与重启续写接口

snapshot_live_requests合并未失败请求的token、logprob、policy/KV epoch、接受步长和可用routing记录；RequestPromptPreparer可返回ResumedGeneration，经预算/元数据检查后直接返回已完成结果或重新入队续写。

**证据与局限：**快照不携带媒体，VLM必须重送同一媒体以重现expanded prompt；这是请求进度快照接口，不是KV物理buffer、采样RNG、整个engine进程的透明镜像。调用需在engine loop或已停止状态；完整外部持久化/故障编排仍由调用方负责。
**实践建议：**深入阅读：适合RL抢占/恢复，但要明确重复响应、token预算、policy epoch跨更换权重和routing完整性；先做中断-恢复一致性实验。
源码路线：`megatron/core/inference/engines/dynamic_engine.py::snapshot_live_requests/_admit_request；inference_request.py::LiveRequestSnapshot/ResumedGeneration`。
一手来源：[NVIDIA/Megatron-LM #8137](https://github.com/NVIDIA/Megatron-LM/pull/8137)（main，2026-11-03T17:55:41+08:00，`01fa4164c8b539958500c2314ad52eacfdd80774`）。

### batch-invariant squared-ReLU 与 MXFP8 量化融合且保持中间 BF16 舍入

在FC1→FC2之间一次kernel完成FP32平方/路由概率乘、原位置BF16 round及MXFP8分组scale/swizzle；alignment padding写零，n_used外不触碰data，替代两个kernel。

**证据与局限：**作者GB200 Nemotron3-Nano、MXFP8专家、TP2+SP/EP4/graphs、1–32请求、1k–8k上下文：每步省.07–.13ms，基线5.7–7.7ms；data/scales、logprob逐位相同。限对应batch-invariant ReLU²分支且未disable_fused_quant_kernels，不能泛化其他激活或所有硬件。
**实践建议：**考虑复现：改融合kernel要保留原中间舍入位置与未使用行合同，比较packed codes/scales和logprob。
源码路线：`megatron/core/inference/moe/batch_invariant.py::squared_relu_with_probs_quantize_mxfp8；fused_moe.py::mcore_fused_moe`。
一手来源：[NVIDIA/Megatron-LM #8645](https://github.com/NVIDIA/Megatron-LM/pull/8645)（main，2026-11-04T06:25:18+08:00，`4a4064f0631cd55a6f2a756cecb3e9ab9f5367cd`）。

### GRPO forward-only pass 不再提前卸掉 CUDA graph saved-tensor observer

create_cudagraphs在既未创建graph也无record时提前return，使old/reference logprob只前向期间保留observer，等真正训练capture后才清理。

**证据与局限：**这是TE2.21相关路径的生命周期修复，final diff仅增加empty-create guard与回归；不等于支持任意TE版本组合或修复所有graph捕获问题。
**实践建议：**建议采用：升级TE时回归先old/ref logprob后首次训练forward的顺序，单测直接训练首步不足以覆盖。
源码路线：`megatron/core/transformer/cuda_graphs.py::CudaGraphManager.create_cudagraphs`。
一手来源：[NVIDIA/Megatron-LM #8850](https://github.com/NVIDIA/Megatron-LM/pull/8850)（main，2026-11-04T00:27:35+08:00，`993bf83ec875a08261111f1a80aad5017fae8e6d`）。

### DSv4 HybridModel FLOPs 把 C/H/W attention 与 packing 纳入

从实际hybrid pattern/MTP深度统计projection、window、compression、ratio4 indexer/sparse及ratio128 attention，用DSv4 head维度并复用real-token与序列平方统计。

**证据与局限：**作者48项CPU算术测试；packed top-k饱和仍按token加权平均长度估计，缺逐文档长度；维持既有fwd/bwd FLOPs定义。改变报告的FLOPs/MFU分母，不改变执行速度。
**实践建议：**建议采用：校正性能看板时记录计数器commit，升级前后MFU跳变先排查分母，勿当训练吞吐提升。
源码路线：`megatron/training/training.py::num_floating_point_operations`。
一手来源：[NVIDIA/Megatron-LM #7972](https://github.com/NVIDIA/Megatron-LM/pull/7972)（main，2026-11-03T16:16:56+08:00，`e1afbb93db5cad5b7b652d10998bd4802cdbb89c`）。

### 启动阶段补 memory 报告，并在 checkpoint load 返回后清 allocator cache

新增model/optimizer build、checkpoint load、empty_cache后与首iteration前报告，MIMO按语言模型DP组打印；load局部临时buffer退出后再empty_cache，减少空闲但reserved的显存。

**证据与局限：**作者一大MoE rank build27.7GB allocated/36.3GB reserved，load后29.9/218.3GB；不能把reserved差额等同live tensor节省。本项不降低fully-parallel load内部瞬时peak，测试未给可比端到端训练收益。
**实践建议：**考虑采用：启动OOM先区分allocated、reserved和load transient峰，按这些新边界采样后再定向优化。
源码路线：`megatron/training/training.py::setup_model_and_optimizer/train/_get_memory_report_dp_group`。
一手来源：[NVIDIA/Megatron-LM #8611](https://github.com/NVIDIA/Megatron-LM/pull/8611)（main，2026-11-04T06:40:20+08:00，`e14be57fe1855cc61ef8af92e779239d0f5a5637`）。

### ScheduleNode activation ownership 修复回补至 core_r0.20.0

回补已在main的#8302，处理ScheduleNode输入输出所有权、grouped-MLP offload及release测试冲突，移除依赖旧TE#3355方案的MXFP8 carrier flags。

**证据与局限：**合入目标为core_r0.20.0，不能写成新的发行或main新机制；该分支没有main的full-recompute与grouped-MLP offload replay测试，回补覆盖范围较窄。
**实践建议：**继续观察：依赖release分支时锁本SHA并跑实际PP/offload配方，不能仅用同名版本号判断是否含修复。
源码路线：`megatron/core/pipeline_parallel/utils.py；models/common/utils.py；transformer/moe/experts.py`。
一手来源：[NVIDIA/Megatron-LM #8911](https://github.com/NVIDIA/Megatron-LM/pull/8911)（core_r0.20.0，2026-11-03T15:38:19+08:00，`748bc2aa72372fd181b9d3b0e85d2c43a8421b0a`）。

### FSDP-Turbo 在构建 Qwen3.5 前启用自身 fused model overrides

提前初始化accelerator兼容和turbo config，按后端模型注册表选择类并在finally恢复临时HF factory，应用fused overrides；forward-only也启用CPU offload策略，配套调整Qwen3.5示例。

**证据与局限：**最终diff可确认接线与配置变化，正文无匹配硬件/模型/负载的性能表，不能据“fused”宣称整训练加速。临时替换factory及CP patch顺序需要与依赖版本一起验证。
**实践建议：**考虑复现：先固定fsdp-turbo栈，用同一Qwen3.5 recipe核输出/梯度、CP切分与峰值，再评估融合收益。
源码路线：`verl/workers/engine/fsdp/fsdp_turbo_impl.py::_init_turbo_config/_build_module；transformer_impl.py`。
一手来源：[verl-project/verl #8508](https://github.com/verl-project/verl/pull/8508)（main，2026-11-03T21:16:46+08:00，`184687bb3b7c601627eed3660283ecabd2bb75ed`）。

### Extensible KV cache 补上 HiSparse、UMA sizing 与 profiling 释放边界

HiSparse移除昨天的排除gate，device KV走稳定虚址reserve/grow、host pool独立；warmup确保autotune访问页已commit并限制hot表宽。UMA通过统一memory helper加回可回收page cache，ROCm margin floor2→3GiB；profiling后unbind临时device视图，保留registered host pool。

**证据与局限：**作者B200 GLM5.3-NVFP4 TP4默认flag、16k入/1k出：GPU+host token容量1.314M→1.530M、吞吐近持平；默认utilization策略变化在内，不能归为同budget16%算法收益。GB10 Qwen3.8-27B NVFP4 util.8，UMA修复216→1302block/11.2→67.6GiB；实际可回收cache受OS状态影响。#64989最终修订仅CPU隔离生命周期检查，历史device验证未覆盖最终版，不报实测OOM节省。
**实践建议：**建议采用前逐平台验证：默认容量不等于绝不OOM，记录warmup transient/margin与host registration；P/D测试主要tiny dummy模型，真实生产栈仍须单独验。
源码路线：`vllm/v1/worker/extensible_kv_cache.py；hisparse/binding.py/runtime.py::unbind_cache；utils/mem_utils.py；warmup/flashinfer_sparse_mla_warmup.py`。
一手来源：[vllm-project/vllm #65214](https://github.com/vllm-project/vllm/pull/65214)（main，2026-11-04T03:50:18+08:00，`8ba4fe5f118ef188bafc162496a266b3a06661b1`）；[vllm-project/vllm #65413](https://github.com/vllm-project/vllm/pull/65413)（main，2026-11-04T07:57:43+08:00，`260b0c6d7e8759389991f6da9c7b88d986ad12e0`）；[vllm-project/vllm #65418](https://github.com/vllm-project/vllm/pull/65418)（main，2026-11-04T00:45:05+08:00，`bb86df34d78c2e325f4a66de2e29909d5af71fdf`）；[vllm-project/vllm #64989](https://github.com/vllm-project/vllm/pull/64989)（main，2026-11-04T04:19:29+08:00，`b210285e848dc054dd58ec3c8380c1f8a7546902`）。

### 满 running batch 时可限额提前启动异步 KV load

新增max_num_inflight_kv_loads；potential async request进独立队列，full batch或token预算已尽时也能load。cap同时统计传输中与已完成但仍等slot的请求；may_load_async默认识别P/D remote-prefill，MultiConnector按OR判断。

**证据与局限：**默认None维持旧行为。最终diff改scheduler接纳次序与KV持有上限；作者1084通过/9失败仍待确认是否基线问题，E2E P/D benchmark未完成。没有可归属的实际吞吐/尾延迟收益。
**实践建议：**继续观察/考虑小规模复现：扫cap同时测空slot等待、KV占用和调度公平性；不能把它等同per-step load数量限制。
源码路线：`vllm/config/scheduler.py::max_num_inflight_kv_loads；v1/core/sched/scheduler.py；kv_connector/v1/base.py::may_load_async`。
一手来源：[vllm-project/vllm #65319](https://github.com/vllm-project/vllm/pull/65319)（main，2026-11-04T07:47:05+08:00，`56e993a68990cd3bb0719d5d91dab549ce0d5a79`）。

### MXFP4 linear 可加 block Hadamard：质量增益伴随当前 PyTorch rotation 成本

InputHadamardLinearMethod在量化前对权重一次、输入每次做正交block Hadamard，新增mxfp4_hadamard32；Quark在线旋转checkpoint检查int8 Sylvester矩阵后丢掉冗余rotation tensor，避免二次旋转。

**证据与局限：**仅验证MXFP4 linear，MoE、其他online格式/已量化checkpoint再旋转等拒绝；CUDA路径未GPU验证。作者MI355X Qwen3.8-2.4T-A95B选择GDN projections，GPQA198×5rep相对RTN平均completion约短9.6%，accuracy差不显著；不是全模型任意4bit无损。dummy TP8、1024/1024、c4/64/256吞吐因FP32 PyTorch rotation低12.4%/6.2%/6.0%；#63272融合优化不能算本项已合入能力。
**实践建议：**深入阅读/考虑复现：同时计任务完成tokens与实际wall time，先选outlier敏感linear层；质量恢复不自动等于成本下降。
源码路线：`vllm/model_executor/layers/quantization/input_hadamard.py；online/base.py；quark/quark.py；config/quantization.py::QuantSpec.hadamard_size`。
一手来源：[vllm-project/vllm #62750](https://github.com/vllm-project/vllm/pull/62750)（main，2026-11-04T04:47:40+08:00，`275816d060b71bec752f26037fac4cd5381023b8`）。

### QuIP NVFP4 迁至 Humming，移除 QuTLASS 与 fp_quant

compressed-tensors Hadamard transform先接未量化输入，再走Humming NVFP4 A4，恢复SM100+此前启动失败的QuIP；删除QuTLASS扩展依赖与已弃用fp_quant入口。

**证据与局限：**breaking：fp_quant checkpoint不再加载。B300/humming0.1.18，Llama3.2-1B QuIP GSM8K600题Humming A4两次17.3–20.5%，float32.3%；不是宣称质量无损。旧QuTLASS baseline需harness补丁；其余FP-Quant性能/质量行来自删除路径的scratch prototype，不能当本main支持。
**实践建议：**考虑采用：先审checkpoint格式与A4/A16选择，别把A16较好分数归给新A4路径；固定Humming版本回归transform顺序。
源码路线：`vllm/model_executor/layers/quantization/compressed_tensors/transform/schemes/linear_humming_nvfp4.py；layers/linear.py；quantization/__init__.py`。
一手来源：[vllm-project/vllm #65230](https://github.com/vllm-project/vllm/pull/65230)（main，2026-11-04T03:12:00+08:00，`37753d9dd900712cee6a24b69a52a1a14e05edb9`）。

### NVFP4 SiLU-mul fusion 修复 K padding 的行 pitch

输出行距与scale tile按实际padded output宽度而非原K计算，显式写零padding，修复K≡16 mod32且batch>1时后续行错位，补齐此前只修scale的#63382。

**证据与局限：**作者RTX PRO6000 Blackwell SM120、torch2.14/cu130，K3696→3712多行GEMM恢复对齐；指定WikiText2模型compiled PPL1290959→29.45，eager29.43。该巨差是坏路径纠错，不是一般量化质量提升；24个micro形状不慢不证明全部吞吐不回退。
**实践建议：**建议采用：多TP切分后MLP宽度会触发，显式回归M>1、K-padding bytes与compiled/eager，不只跑单token。
源码路线：`csrc/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu::silu_and_mul_nvfp4_quant`。
一手来源：[vllm-project/vllm #61755](https://github.com/vllm-project/vllm/pull/61755)（main，2026-11-03T11:48:08+08:00，`eb2975565a81e6176399669d9c141355eeab8f23`）。

### Hadacore odd-row partial chunk 补回 cp.async wait

head_dim256时尾warp仅一chunk，diff_num_chunks原符号错误让switch落负值跳过wait；现在差值反向并对未覆盖case保守wait_group0，避免读取未完成shared-memory DMA。

**证据与局限：**作者A100、row1..512旧版39个奇数失败、修后0，错误可达1e36；涉及INT4 per-token-head KV旋转。修复同步race，不保证其他head shape/硬件全路径已认证。
**实践建议：**建议采用：用odd rows与尾chunk压力测量化KV，不能用通常偶数batch的accuracy suite代替。
源码路线：`csrc/quantization/hadamard/hadacore/hadamard_transform_cuda.cu`。
一手来源：[vllm-project/vllm #63614](https://github.com/vllm-project/vllm/pull/63614)（main，2026-11-03T20:57:32+08:00，`a625e62062417b60714cf5a9ee60b0e00f4a47f1`）。

### DSA indexer 恢复非 DeepSeek 模型的 FP32 logits，并给显式 dtype 控制

DeepGEMM固定dev ae4fd25，新增attention-config.indexer_logits_dtype。auto在SM100上GLM5/5.3/HY-V4回FP32，DeepSeek3.2/4/4.1保留BF16；SM90/120默认仍FP32，sparse-logits分支仍BF16。

**证据与局限：**作者GB300对FP64参考，FP32选中相同top2048，BF16改变.4–.75%索引；这不是端到端能力下降的直接证据。FP32增logits内存，影响昨天按MiB限额的chunk行数；DeepGEMM pin还含graph barrier等独立改动，不能全部归为dtype收益。
**实践建议：**深入阅读：从模型architecture hook确认auto解析，扫FP32/BF16时同时测选择集合、质量、logits临时峰与top-k后端。
源码路线：`vllm/config/attention.py::resolve_indexer_logits_dtype；sparse_attn_indexer.py；models/config.py；utils/deep_gemm.py`。
一手来源：[vllm-project/vllm #64580](https://github.com/vllm-project/vllm/pull/64580)（main，2026-11-04T07:13:24+08:00，`ab4e04d2dc15480acc4c6eb4cb88f94e01ebec2f`）。

### Direct reload 修复融合后 storage offset 与 derived Parameter 重注册

记录checkpoint layout时也记录storage pointer，若post-load把weight移至融合buffer则reload用当前offset，避免覆盖邻居。DSv4 ROCm衍生scale改ws.detach()且经refill_derived原位更新，取消重复Parameter注册并保持graph地址。

**证据与局限：**作者MI355X DSv4-Pro TP8、heterogeneous/graphs/fused compressor，两修复联合：冷/原/扰动/原权重GSM8K400 .965/.9675/.035/.9575，3480tensor地址无移动；不是两个独立E2E实验。仅offset修复CPU/GB300 toy回归也有证据，不能据联合结果宣布任意模型reload均安全。
**实践建议：**建议采用：RL refit回归原→扰动→原、邻接weight内容与每个derived地址；只检查load不报错会漏掉第一类错写。
源码路线：`vllm/model_executor/model_loader/reload/direct.py::record_checkpoint_layout/begin_direct_reload；models/deepseek_v4/amd/model.py/rocm.py::prepare_*`。
一手来源：[vllm-project/vllm #65281](https://github.com/vllm-project/vllm/pull/65281)（main，2026-11-03T17:19:57+08:00，`66360a5cd0ad1158b2c950f924671fca68a948ff`）；[vllm-project/vllm #65296](https://github.com/vllm-project/vllm/pull/65296)（main，2026-11-03T17:18:13+08:00，`22d05b87bbc98be5974922fa372076bd6fd579af`）。

### Triton ReplaySSM 把 decode ring planning 移上 GPU，支持 MRV2

按cache-slot ring counter在GPU规划，每KV group一次小kernel替代CPU per-request ring origin、tolist及H2D；删除V1专有metadata/input_batch plumbing，MRV2不再拒绝该后端。

**证据与局限：**作者单GB200 Nemotron3-Nano4B，V1/V2、align prefix cache、有无同步检查均与非ReplaySSM logprob匹配；无TP2或大集群性能表。不是FlashInfer ReplaySSM新算法，复用已有slot tracker。
**实践建议：**考虑采用：MRV2 Mamba用户测prefix hit后counter更新与连续decode，再评估host瓶颈减少是否变成吞吐收益。
源码路线：`vllm/model_executor/layers/mamba/ops/ssu_dispatch.py；v1/attention/backends/mamba_attn.py；mamba_mixer2.py`。
一手来源：[vllm-project/vllm #65033](https://github.com/vllm-project/vllm/pull/65033)（main，2026-11-03T20:34:38+08:00，`cf316179921e0470b4f4060899610d0da935aed7`）。

### Mamba2 使用共享 prefill checkpoint exporter，并跳过 null 目标块

删除单独ssd_checkpoint kernel，复用GDN/KDA的MambaPrefillCheckpointExporter；稀疏保留策略下不再把无目标prefill checkpoint写入block0。

**证据与局限：**作者B200、48个Mamba2形状39快，TP8少量case最多慢2.9µs，GDN/KDA部分形状也慢；没有发现block0真实reader，所以null写属于潜在缺陷而非已证实输出事故。模型internal checkpoint E2E不在现有CI覆盖，不能报全面加速。
**实践建议：**考虑采用：同时测小batch高TP和大state形状，poison null块与checkpoint selection；保留逐位导出检查。
源码路线：`vllm/model_executor/layers/mamba/checkpoint.py::MambaPrefillCheckpointExporter；mamba_mixer2.py`。
一手来源：[vllm-project/vllm #63742](https://github.com/vllm-project/vllm/pull/63742)（main，2026-11-04T04:22:15+08:00，`6dc6d2c0f31cbda948da727dc6d4bef62a02d38f`）。

### MRV2 standalone draft-model 接上 CUDA graph

共享AutoRegressiveSpeculator统一多步AR和graph管理，target-dependent保留hidden/MM逻辑，standalone保留输入及correction token处理；支持draft prefill及多步decode的capture/replay。

**证据与局限：**作者Qwen3-32B target/1.7B draft、k3、400 MTBench、greedy/256输出有接受长度2.80；性能表对eager与FULL_AND_PIECEWISE，不能把差值全归类重构。硬件等完整条件本条不复述，未独立复跑，不报可跨环境倍数。
**实践建议：**考虑复现：target和draft graph模式一起确认，prefix cache、correction token与MM路径各自验；比较端到端TTFT/TPOT及graph内存。
源码路线：`vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py；standalone_ar/speculator.py；target_dependent_ar/speculator.py`。
一手来源：[vllm-project/vllm #60989](https://github.com/vllm-project/vllm/pull/60989)（main，2026-11-04T07:59:24+08:00，`f352490fc0b5180c6e13a064dafe2e84ef5fa8f4`）。

### Suffix decoding 新增 GPU resident MRV2 路径与 async scheduling

MRV2的suffix speculator建立在ngram_gpu上，propose无host同步且无需Arctic；V1保留Arctic。suffix默认num_speculative_tokens改为8，并配合adaptive verification限制验证开销。

**证据与局限：**作者21k replay接受长度1.937，Arctic k8=1.923/k24=1.948；小模型部分并发Arctic仍快，无AV在高并发反而差。是后端/调度实现增量，不证明suffix对所有自然工作负载更好。
**实践建议：**深入阅读/考虑复现：重复性高的代码/结构化任务可试，按并发扫AV和k，保留无投机与V1 Arctic对照。
源码路线：`vllm/v1/worker/gpu/spec_decode/suffix/speculator.py/config.py；config/speculative.py`。
一手来源：[vllm-project/vllm #60339](https://github.com/vllm-project/vllm/pull/60339)（main，2026-11-03T09:43:05+08:00，`190df0b90007eead441d97d3abf6270f84f2c63e`）。

### XPress 终于向 Acceptance Estimator 填入真实 draft 置信度

用采样draft token的logits更新估计器，修复AV读取未初始化confidence而把每次draft剪到首token的路径。

**证据与局限：**作者单H100、Qwen3-8B/XPress b16、k15、SPEED-Bench1K、3trial：AV on修复前AL≈1，后约3.0–3.5；修后相对AV off c1/c8仍慢约8%，c32以上更快。不能报“AV必然提速”或把修复坏基线3×当普遍算法收益。
**实践建议：**考虑采用：监控实际AL和target步成本，按服务并发决定AV；先确认估计器确实被填充。
源码路线：`vllm/model_executor/models/qwen3_xpress.py；v1/worker/gpu/spec_decode/xpress/speculator.py`。
一手来源：[vllm-project/vllm #65175](https://github.com/vllm-project/vllm/pull/65175)（main，2026-11-03T20:27:11+08:00，`44e70e086ad6ae6a0726d3d830c9c8eee5db884b`）。

### DFlash2 edge scoring 合为 Triton kernel，并统一输出 FP32

Qwen3与Beam共用_score_edges，一program处理request×draft step，gather后用tl.dot计算候选K×K边分数；移除Beam独立kernel。

**证据与局限：**BF16输入保持，edge不再中途round BF16，output改FP32，故不是bitwise重排。作者GB200 102测试通过、FP64均绝误差7.2e-5→5.2e-5；不等于投机接受率/任务质量提升。
**实践建议：**考虑复现：检查path selection与acceptance而不只比较kernel误差，FP32 output略增内存需计入。
源码路线：`vllm/model_executor/models/qwen3_dflash2.py::_score_edges；models/beam/nvidia/dflash2.py`。
一手来源：[vllm-project/vllm #64916](https://github.com/vllm-project/vllm/pull/64916)（main，2026-11-04T07:10:28+08:00，`7b5492075db29db2c54716d8acbfe5d7b32a308c`）。

### 投机输入 padding 与 rejection resample 减少无效工作

#65488在同次launch加32program分担DFlash graph-static padding；#65100只对可能有top-k/top-p/min-p等mask的chunk，遇整词表block target=-inf时跳过draft load/Gumbel。

**证据与局限：**作者前者GB200 16k batched tokens 7.9–8.5→约2.1µs、host仍一次launch，特定TP4整step仅.2–.4%；后者GB300 Qwen3-.6B logits、256req/k3/top-p.95 resample166→55µs，未测E2E。未mask路径不开检测，以免约5%额外开销；两项不能把kernel倍数相乘。
**实践建议：**考虑采用：大capture capacity/稀疏概率支持有用，分别用sentinel padding和采样token逐位对照；总服务收益取决于它们占比。
源码路线：`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py::prepare_dflash_inputs；rejection_sampler_utils.py::_resample_kernel；sample/states.py`。
一手来源：[vllm-project/vllm #65488](https://github.com/vllm-project/vllm/pull/65488)（main，2026-11-04T08:26:18+08:00，`cfc9f681e9f3c101c87ed43b658ebc8c43f30a3e`）；[vllm-project/vllm #65100](https://github.com/vllm-project/vllm/pull/65100)（main，2026-11-04T08:12:16+08:00，`a512f0e7276d09d35073e91ed34e18f143204d89`）。

### DSpark NVIDIA Markov sampling 可选 inverse-CDF 融合

DSv4.1新增sample_sequential hook，把逐步Markov采样改为bias GEMM＋两Triton kernel；用cached logits inverse CDF替代Gumbel-max，additional-config dspark_fused_markov=true显式开启。

**证据与局限：**作者GB200 k5 kernel1.6–3.9×，DEP4 AgentX KV 64req×131k/128req×32k整step-1.7%/-3.2%；采样分布方法等价不意味着同seed逐token相同，默认仍旧循环。
**实践建议：**考虑复现：对随机分布、greedy及混合温度分别检验，并记录acceptance与最终任务指标。
源码路线：`vllm/models/deepseek_v41/nvidia/::sample_sequential`。
一手来源：[vllm-project/vllm #64442](https://github.com/vllm-project/vllm/pull/64442)（main，2026-11-03T12:56:58+08:00，`3d18166d453aa8ae73fbc2327c35f0827db55416`）。

### DSv4.1 小步 TP2/TP4 decode 用 swap-AB CuTe attention

对SM10x、≤16token、16/32 local head，把KV slots映到M、live query heads到N，避免mega attention按64head padding；其他shape继续原mega路径，无新flag。

**证据与局限：**作者单层Q pad+insert+attention1.4–2.15×，TP2 E2E median TPOT低4.5–7.6%来自特定c1–16 A/B；kernel对FP64误差不大于旧版1.1×、与mega在2%内，非bitwise。只支持对应interleaved FP8/NVFP4 records。
**实践建议：**考虑复现：低并发大模型服务可试，按record类型与local head确认dispatch，注意超过16token就不走新kernel。
源码路线：`vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py；ops/swapab_decode_attn.py`。
一手来源：[vllm-project/vllm #62633](https://github.com/vllm-project/vllm/pull/62633)（main，2026-11-04T02:39:57+08:00，`4b01cbad1bb65a6ce9290f55bb57fe9e1e732624`）。

### Qwen3.5 NVIDIA 可把 activation quant 融进 all-reduce＋RMSNorm

模型私有路径调用FlashInfer ARResidualRMSNormOutFP8/FP4Quant；超过workspace预算回显式AR＋Triton norm-quant，避免遗漏quant融合，未改共享compile pass。

**证据与局限：**作者Qwen3.8-2.4T-A95B NVFP4、TP8跨两4卡GB300、无EP、8192/1024：c1 TPOT8.28–8.30→7.95ms，c16/64接近噪声；92input norms中91与AR同kernel。仅特定static quant合同逐位验证，不代表全部Qwen/量化层可融合。
**实践建议：**考虑复现：先从低并发延迟看收益，同时核workspace fallback与FP4 swizzled scales，不用单点结果外推高并发。
源码路线：`vllm/models/qwen3_5/nvidia/ops/fused_allreduce_norm_quant.py；model.py`。
一手来源：[vllm-project/vllm #64723](https://github.com/vllm-project/vllm/pull/64723)（main，2026-11-04T06:36:10+08:00，`523991fec0661ca4a6ab348b227c6f7dfca3dc88`）。

### Muse Spark NVIDIA 各层共享同一 FP32 RoPE table

46个相同RoPE层从各自持表改为复用缓存对象，去掉45份重复256MiB table。

**证据与局限：**作者8B200/TP8 Muse-Spark-816B-A42B NVFP4、1M context，加载81→69.75GiB/rank、KV9.249M→10.590Mtoken；8个greedy prompt逐位一致，GSM8K100 96→95是小样本，不能据此证明质量完全等价。测试overlay基于nvidia/model.py相同的nightly，非完整新wheel。
**实践建议：**考虑采用：长上下文模型尤其受益，回归cache key与设备/dtype隔离，避免CPU测试污染全局RoPE缓存。
源码路线：`vllm/models/muse_spark/nvidia/model.py::shared RotaryEmbedding`。
一手来源：[vllm-project/vllm #65206](https://github.com/vllm-project/vllm/pull/65206)（main，2026-11-04T02:28:41+08:00，`866bccf0835bf69cabf4203a03698e01bbbdbacc`）。

### DeepSeek V4/V4.1 多图 encoder 改为每call packed pass

同call多图一次forward_packed通过ViT/aligner，替代逐图循环，复用既有encoder graph路线；没有新flag。

**证据与局限：**作者54个encoder calls、两H100：V4约2–2.6×、V4.1约1.6–2.1×；真实V4两B200四图c1 TTFT367–378→271ms。32图峰值额外约1.7/2GB，输出非bitwise；MMMU-Pro答对数相近但相对main答案变化多于main自身，不能只报精度持平。
**实践建议：**考虑复现：扫每call图数及encoder内存，直接回答与长推理任务都测，避免为了TTFT让多图突发OOM。
源码路线：`vllm/models/deepseek_v4/common/vision.py::forward_packed；vl_model.py；deepseek_v41/*/vl_model.py`。
一手来源：[vllm-project/vllm #65030](https://github.com/vllm-project/vllm/pull/65030)（main，2026-11-03T21:08:08+08:00，`0eeae6a473338959f3448d49f55327b7d7e6b7aa`）。

### DeepSeek-V4.1 EPD 可只传 image metadata，减少重复预处理

EC metadata协议传llm_grid，consumer重建相同placeholder跨度；DeepSeek chat template接受embedding reference并维持图顺序，raw-image路径保留。

**证据与局限：**作者GB200、2E TP1＋PD TP2共4GPU/Mooncake RDMA、MRV2/graphs、512 MUIRBench/c96，一轮总input＋actual-output吞吐17645→19618token/s，均延迟10.40→9.30s，accuracy318→317/512。缓存开启与单次实验限制因果外推，不是每GPU纯生成吞吐。
**实践建议：**考虑复现：分离encoder成本显著的部署，核EC reference寿命、placeholder与多图顺序，固定warmup和缓存条件。
源码路线：`vllm/models/deepseek_v41/common/mm_preprocess.py；renderers/deepseek_v4.py；tokenizers/deepseek_v41.py`。
一手来源：[vllm-project/vllm #62802](https://github.com/vllm-project/vllm/pull/62802)（main，2026-11-03T15:21:14+08:00，`67791dd94d95c55060a351eebb46e29387891d5d`）。

### MiniMax-M3 encoder graph 用零长度序列补 cu_seqlens

不再把capture未使用rows组成一段额外attention序列；利用已在graph内的attention输出清零，padding rows保持finite且不接触真实双向图像attention。

**证据与局限：**正确性动机是shared capture buffers；多padding时可减少无用self-attention，但没有统一服务收益可泛化。依赖已合入的FA output zero机制，机械挪到旧基线会重新引入NaN风险。
**实践建议：**建议采用：用poison padding、不同图像预算交替replay与shared buffer覆盖验证，不能只测满预算图。
源码路线：`vllm/models/minimax_m3/common/encoder_cudagraph.py`。
一手来源：[vllm-project/vllm #65226](https://github.com/vllm-project/vllm/pull/65226)（main，2026-11-03T14:20:50+08:00，`9a519c0585b6cccb3d2363380725d974789551fd`）。

### KV connectors 支持含 lookahead token 的 block hash

NIXL/Mooncake P/D、Mooncake Store、OffloadingConnector声明supports_lookahead_block_hashes，MultiConnector要求所有child支持；P侧截尾prompt后清空并重建hash，避免沿用缺失lookahead token的旧hash。

**证据与局限：**仅支持的组合不再额外丢EAGLE/MTP尾块，unsupported仍legacy。是缓存合法性与复用边界修正，不保证每次请求更多hit；依赖各child能力一致，不能由MultiConnector本身推断所有子后端支持。
**实践建议：**考虑采用：测试跨P/D截尾、hash边界及混合connector，不以传输成功替代内容匹配。
源码路线：`vllm/distributed/kv_transfer/kv_connector/v1/{nixl,mooncake,offloading}；v1/request.py::num_hash_lookahead_tokens`。
一手来源：[vllm-project/vllm #64096](https://github.com/vllm-project/vllm/pull/64096)（main，2026-11-03T14:18:32+08:00，`75be1384286d6f4c5ca3ad493bc9390159d93f5b`）。

### Mooncake Store 可只给实际 load 的 key 续 lease

opt-in selective lease先无租约probe候选key，再只lease当前hit要加载的SWA窗口/线性state边界；竞争淘汰缩短hit时二次lease，完整成功才返回hit，否则安全miss。

**证据与局限：**默认关闭，依赖Mooncake batch_probe_key；额外RPC与竞争失败可能降低hit/增时延。本项针对未读key被lookup无意保热导致的淘汰偏差，不能把存在probe当作load有效性保证。
**实践建议：**考虑复现：共享store压力大且SWA/Mamba模型优先，统计额外RPC、实际load集合和淘汰命中，而非仅cache hit率。
源码路线：`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/selective_lease.py；worker.py；envs.py::VLLM_MOONCAKE_STORE_SELECTIVE_LEASE`。
一手来源：[vllm-project/vllm #63420](https://github.com/vllm-project/vllm/pull/63420)（main，2026-11-03T09:17:07+08:00，`f1fd044b7000d40ced65b901224addcb07dc25d7`）。

### CPU offload 修复 SWA retry 页偏移，并按 request 去重 store_threshold 计数

失败store重试保留NULL_BLOCK_ID占位，防SWA释放重分配后CPU chunk后续页整体错位；每request的同key重试/抢占重报不再重复增加admission threshold或skip指标，cache reset后可重新计数。

**证据与局限：**前者作者real CPUOffloadingManager＋CUDA transfer复现逻辑页7/8读成9/脏值、修后恢复；后者影响threshold≥2，默认低阈值不变。没有性能结论，两项不能代替P2P consumer late-write隔离。
**实践建议：**建议采用：模拟tier满导致prepare_store失败与SWA block复用，核数据而不只核transfer完成；另测不同request仍可累加热度。
源码路线：`vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py；v1/kv_offload/cpu/manager.py`。
一手来源：[vllm-project/vllm #65279](https://github.com/vllm-project/vllm/pull/65279)（main，2026-11-03T20:15:33+08:00，`43067f6df60e2eafba5797908484cc1c12e4a48b`）；[vllm-project/vllm #64558](https://github.com/vllm-project/vllm/pull/64558)（main，2026-11-03T16:02:00+08:00，`eb239e34c29d8b51bf299aa9d674df5aafa7167f`）。

### P2P offload 修复双角色清理与 disconnect 后缓冲 terminal 结算

请求同时为source与decoder时分别finish两边session，相同session/id去重；断连后仍处理此前已收到并验证的TransferDone/AbortAck，工作启动类消息仍丢弃。

**证据与局限：**#64299作者RTX4060Ti/UCX TCP本地双进程、192token prefill/4token greedy故障注入通过，针对“成功terminal恰好随断连到达”。#64686组件tiering652通过。consumer slots late-write quarantine #61667仍open，本日拆分修复不关闭该风险。
**实践建议：**建议采用：独立测试终止消息顺序、双session角色与连接故障；不要把“已收terminal可结算”扩大为断连后任意消息可信。
源码路线：`vllm/v1/kv_offload/tiering/p2p/manager.py::on_request_finished；session/session.py`。
一手来源：[vllm-project/vllm #64686](https://github.com/vllm-project/vllm/pull/64686)（main，2026-11-03T23:22:00+08:00，`4d1c44780c971337f70d8850811dbdf01739c02e`）；[vllm-project/vllm #64299](https://github.com/vllm-project/vllm/pull/64299)（main，2026-11-03T16:53:13+08:00，`ffdf26702aec559f9e2fbf93ed4c74bd783836e7`）。

### Tier lookup 按完整 request group 及时发布结果

单FIFO worker完成某req_id的batch_lookup后立即发布，不再等本step后续全部group；组内迭代完成才发布，保持原子性，promotion策略不变。

**证据与局限：**作者较早commit测试与文件系统timing replay，最终test清理后未重跑完整head；没有可归属的E2E serving吞吐结论。不是多线程并行lookup，也不会发布部分group。
**实践建议：**考虑复现：慢尾group多时观察scheduler drain到promotion的等待，保留惰性迭代异常与FIFO回归。
源码路线：`vllm/v1/kv_offload/tiering/async_lookup.py::AsyncLookupManager`。
一手来源：[vllm-project/vllm #57474](https://github.com/vllm-project/vllm/pull/57474)（main，2026-11-03T23:22:09+08:00，`8210387296f61997474cad69b1f236edf4922c20`）。

### Offloading worker 改收已解析 GPU pointer，新增文件后端抽象

logical block→pointer/layout解析上移共享层，CPU worker不再重复做tensor布局解析；新增FSOffloadingWorker供POSIX/NIXL/cuFile等具体后端继承。

**证据与局限：**本项是GDS前置接口重构，GDS concrete实现另案，不能写成已具备GDS KV offload。作者offload相关组件测试通过，无SSD/RDMA服务性能实验。
**实践建议：**深入阅读：扩展新tier时复用pointer合同，验证SWA/group布局与DMA字节范围，等待具体后端再评估I/O收益。
源码路线：`vllm/v1/kv_offload/base.py；pointer_utils.py；fs/worker.py；cpu/gpu_worker.py`。
一手来源：[vllm-project/vllm #56076](https://github.com/vllm-project/vllm/pull/56076)（main，2026-11-03T23:58:46+08:00，`80ca61b6d9b9349c9b50b064f6af275be3c78d04`）。

### UMBP 7/10 加 node-local standalone server runtime

多个engine/DP rank共享DRAM池，KV bytes仍经HIP IPC，Unix socket只控制；namespace由有效模型dtype/quantization/hf_overrides构成，server不可达、无效pool配置与阻塞超时均fail closed。

**证据与局限：**只合入implementation切片，server不由vLLM自动启动，完整usage/tests在后续PR；不能当整套10/10已交付或GPU传输改走socket。
**实践建议：**继续观察：先核后续测试与生命周期，再用于跨engine共享pool；尤其压力下的blocked store、shutdown及不同模型隔离。
源码路线：`vllm/distributed/kv_transfer/kv_connector/v1/umbp/runtime/standalone.py/factory.py；worker.py`。
一手来源：[vllm-project/vllm #61795](https://github.com/vllm-project/vllm/pull/61795)（main，2026-11-03T10:37:07+08:00，`32d6ddea18b1a1773875f7305e392605a03cc2f3`）。

### LFM2 ShortConv 可启用 batch-invariant 模式

SHORT_CONV_ATTN声明supports_batch_invariance，依据causal_conv1d每token仅依赖自身窗口；PLE_SHORT_CONV_ATTN仍明确False。

**证据与局限：**作者A40 LFM2-8B-A1B完整22项determinism套件，重复generate与batched/alone差异归零；并非默认输出变确定，也未验证所有PLE/dilated卷积。
**实践建议：**考虑采用：需要重复greedy或RL训推对齐时显式VLLM_BATCH_INVARIANT=1，保留prefix-cache on/off与批形状对照。
源码路线：`vllm/v1/attention/backends/short_conv_attn.py::supports_batch_invariance`。
一手来源：[vllm-project/vllm #62367](https://github.com/vllm-project/vllm/pull/62367)（main，2026-11-03T22:49:36+08:00，`9226d6e59cd12ddd9a80b7d5ed37b16e186d29c6`）。

### LoRA 补 DSv4.1 语言投影路径，并关闭不相容的 KV-sharing fast prefill

DSv4.1注册language packed/MM映射，把grouped O与compressor直调GEMM接入LoRA，启用LoRA时禁GEMM-RS绕过wo_b；MRV2任何含LoRA的batch回普通prefill，base-only可恢复fast path；Transformers scaled embedding的LoRA delta在装载时乘embed_scale。

**证据与局限：**DSv4.1 tower/connector仍冻结，不含MTP draft适配。fast-prefill fallback牺牲整个mixed batch速度；scaled embedding作者tiny VaultGemma/PEFT0.21.1差回到约2.4e-7，尚无GPU完整服务验证。三项不能合称任意多模态LoRA已支持。
**实践建议：**建议采用：分别验证compressor FP32 base输出、mixed LoRA metadata和embed delta相对PEFT；实际RL训练需同步Megatron-Bridge对应适配。
源码路线：`vllm/models/deepseek_v41/common/grouped_wo_a.py/compressor.py；v1/worker/gpu/attn_utils.py::FastPrefillHelper.prepare；lora/layers/vocab_parallel_embedding.py::set_lora`。
一手来源：[vllm-project/vllm #57245](https://github.com/vllm-project/vllm/pull/57245)（main，2026-11-03T18:34:26+08:00，`a5d8174ea05f1092d4b20e926ed766a18490ac03`）；[vllm-project/vllm #61162](https://github.com/vllm-project/vllm/pull/61162)（main，2026-11-03T18:35:11+08:00，`0acdae3ec509ccb4cf9471b8d538446902f083bd`）；[vllm-project/vllm #62470](https://github.com/vllm-project/vllm/pull/62470)（main，2026-11-03T18:34:45+08:00，`b7486648bb027800e81d11ec51827a571d4f87a0`）。

### RDNA3.5 GDN prefill 新增寄存器驻留的 HIP chunk kernel

单block按head/sequence走32token chunk的WY/UT更新，FP32中间state与64KiB LDS内不物化h/w/u，减少DDR往返；auto只在实测gfx1151选hip，显式后端可覆盖gfx1150–53，BF16/FP16/head128，无法处理的call回Triton。

**证据与局限：**作者对FP64参考相对RMS误差约为原Triton的.53倍；该值是算子误差，非任务精度翻倍。硬件仅实测gfx1151，其他RDNA型号没有等价证据，无法外推CUDA或所有state shape。
**实践建议：**深入阅读/考虑复现：消费级共享内存机器可比较prefill完整wrapper、state峰与decode衔接；不只比较裸kernel。
源码路线：`csrc/rocm/rdna35_gdn_chunked.cu；vllm/model_executor/layers/mamba/ops/hip_gdn_chunk.py；v1/attention/backends/gdn_attn.py`。
一手来源：[vllm-project/vllm #56307](https://github.com/vllm-project/vllm/pull/56307)（main，2026-11-04T00:51:41+08:00，`a12e1222064c8bb15a9156ff3077bd14209262bb`）。

### MI350/355 的 Kimi-K2 int4 prefill 绕开 Triton3.8 慢 tile

TP8、E384/N256 tuned config的8192条目改BLOCK_N256→128、stages2→1，规避Triton3.7.1→3.8造成的特定fused_moe tile codegen退化，约6k+token batch才触发。

**证据与局限：**作者w13+w2、M8192/16384合计44.18→6.129ms；服务c32 TTFT1981→829ms、吞吐+6.2%，c1不触发该条目。没有全面编译器根因修复，不可给所有MI355工作负载套58%TTFT。
**实践建议：**建议采用：版本升级先以相同shape固定kernel/config对照，记录实际落入tuned key而非仅按模型名。
源码路线：`vllm/model_executor/layers/fused_moe/configs/E=384,N=256,device_name=AMD_Instinct_MI35*,dtype=int4_w4a16.json`。
一手来源：[vllm-project/vllm #62565](https://github.com/vllm-project/vllm/pull/62565)（main，2026-11-04T05:32:08+08:00，`81a28794036fa068d4cfb48925ec592e0be4dd71`）。

### ROCm 两条 attention 路径补 ptr_range，维持32位 buffer addressing

per-token-head INT8/FP8 TRITON_ATTN KV view与AITER unified非因果cross-attention fallback声明真实pointer range，避免编译器退成昂贵64位寻址。

**证据与局限：**作者gfx1100 W7900 Qwen2.5-7B默认extensible KV、INT8/FP8-per-head吞吐1.93→3.02、2.01→3.22req/s；gpt-oss例需关闭extensible，Whisper只约1%。不同cache/runner条件不可混成统一60%收益；AITER自己的causal kernel未动。
**实践建议：**考虑复现：检查virtual reserve与可访问字节范围一致、KV布局/时长参数相同，再比较实际address lowering及服务吞吐。
源码路线：`vllm/v1/attention/backends/triton_attn.py；rocm_aiter_unified_attn.py`。
一手来源：[vllm-project/vllm #64317](https://github.com/vllm-project/vllm/pull/64317)（main，2026-11-04T02:26:13+08:00，`8d3db204b6a17090909fb9895f873bdf4ff111e4`）。

### gfx1151 视觉 encoder attention 按真实 head-dim 拆 tile

仅encoder prefill kernel在gfx1151 BF16/FP16上将非2次幂head dimension拆main+tail，并按stride对齐给hint与更合适BLOCK_N/warps；同名LLM paged prefill模块未改。

**证据与局限：**作者head72 kernel约2.19×、Gemma3-4B TTFT低约21%，不代表语言prefill普遍加速。只实测gfx1151，nonpower-head/图像分辨率决定收益。
**实践建议：**考虑复现：多模态小模型优先，按encoder backend及patch数量测TTFT，不把语言attention benchmark混进来。
源码路线：`vllm/v1/attention/ops/triton_prefill_attention.py::context_attention_fwd/_fwd_kernel`。
一手来源：[vllm-project/vllm #56450](https://github.com/vllm-project/vllm/pull/56450)（main，2026-11-04T00:52:09+08:00，`c1d8fade8d392a010b80aaa0de55f6f0f02a4ebb`）。

### RDNA skinny GEMM 重调 decode tiles，并修 A_CHUNK16 的数据类型尺寸

int4 decode按gfx1150/1151、dtype/group-size/shape选择非spill tile；wvSplitK分别给wave32架构launch配置，bigType由错误64B改为与A_CHUNK16所需32B匹配，避免activation越界与LDS重叠。

**证据与局限：**作者cold rotated weights/交替A/B多形状观察小幅均值收益；wvSplitK求和与v_dot舍入次序变化，GSM8K少量答案变，不是bitwise。性能收益多在数个百分点，不报未核对的所有逐shape最优值。
**实践建议：**考虑采用：优先正确性补丁，bench含奇数M/K、bias与未调优形状，避免只测调参集。
源码路线：`csrc/rocm/skinny_gemms_int4.cu；skinny_gemms.cu`。
一手来源：[vllm-project/vllm #65322](https://github.com/vllm-project/vllm/pull/65322)（main，2026-11-04T02:18:11+08:00，`ac16e0caa44247d2a49d010ea45880d888e6379d`）；[vllm-project/vllm #63835](https://github.com/vllm-project/vllm/pull/63835)（main，2026-11-04T00:41:13+08:00，`7c284bf150f4673c198f13d3abbf914b9e215856`）。

### Kimi-K3 MI355X specialized decode 扩到 KDA16行与 MLA 融合

opt-in latency_specialized路径以FlyDSL K1做KDA prologue/epilogue，K2融合AttnRes、AR/router/MoE，MLA用M1a/M1b及DCP1下M1c减少launch；限TP8、DCP1/8及对应FP8 KV。

**证据与局限：**正文性能图未在本条转录不可核量纲数值；最终diff可定位专用guard/runtime，不能称默认全模型通用替代，也不能将图片上的收益当独立复现。
**实践建议：**继续观察/考虑复现：固定Kimi-K3量化与k值，分别测DCP1/8、fallback边界、输出/acceptance；保留通用backend基线。
源码路线：`vllm/models/kimi_k3/amd/specialized_decode/runtime/{guards,decode,layers,mla}.py`。
一手来源：[vllm-project/vllm #61435](https://github.com/vllm-project/vllm/pull/61435)（main，2026-11-03T09:34:14+08:00，`254a2a33908a9bd974e9656f05592f83de604298`）。

### ModelOpt 的 CSF NVFP4 scales 由 checkpoint stream mapper 无损重建

最终diff在ModelOptNvFp4Config及MixedConfig的get_checkpoint_weight_mapper接入WeightsMapper.weights_transforms，惰性收集同一scale的csf/csf_32x2_v1 streams，CPU NumPy解码为canonical E4M3 tensor再交普通loader；格式可由顶层或quantization/逐层recipe声明。

**证据与局限：**PR正文仍描述旧的MoE stream Parameter/weight_loader_target方案，最终实现并非如此；两decoder有dtype/shape/exception索引校验，未配套新增测试文件。作者旧版lossless/性能说法不能直接当最终stream-mapper版E2E证据，不断言只限routed experts。
**实践建议：**继续观察/考虑局部复现：固定本merge SHA，比较原scale bytes、缺失/混合streams与TP/EP常规loader结果，待当前实现完整模型与reload验证后再采用。
源码路线：`vllm/model_executor/layers/quantization/modelopt.py::get_checkpoint_weight_mapper；utils/nvfp4_utils.py::CSF_SCALE_ENCODINGS/csf_weights_transform/decode_csf_scale`。
一手来源：[vllm-project/vllm #64312](https://github.com/vllm-project/vllm/pull/64312)（main，2026-11-03T14:51:52+08:00，`608365b0f2a509678d9c1d1d61dcc6e101c5a009`）。

### Jetson Thor SM110 的 NVFP4 MoE 与 top-k 选择避开不支持 kernel

NVFP4 MoE从CUTLASS支持predicate排除SM110使auto fallback，dense FP4保持；QSA/cooperative top-k也排除SM110，改用persistent以避16block cluster超过Thor上限8。

**证据与局限：**作者Thor启动复现与修复，#62275单独仍卡在另一个QSA故障，需配套#56102；不能把一个predicate修复称全部模型已能服务，也不影响SM100/120。
**实践建议：**建议采用：edge部署按实际CC验证dense/MoE/attention三个backend，避免用“≥90/100”粗略判定新硬件支持。
源码路线：`vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py::_supports_current_device；layers/indexer_topk.py；models/qwen4_exp/nvidia/ops/qsa_indexer.py`。
一手来源：[vllm-project/vllm #62275](https://github.com/vllm-project/vllm/pull/62275)（main，2026-11-03T19:08:36+08:00，`c7a760751977e4e3396d586956d8767a9bc44a82`）；[vllm-project/vllm #56102](https://github.com/vllm-project/vllm/pull/56102)（main，2026-11-03T15:46:40+08:00，`9b14bef3be65331e0e7754a266b0d91c47b48a3e`）。

### MoE 对 routing tensor 要求完全 contiguous，拒绝静默读错

C++ align wrapper、shared dispatch、direct Triton与modular router边界统一检查ids/weights完整连续；不再只检查stride(1)==1，避免packed tensor列slice的跨行stride被kernel忽略。

**证据与局限：**ZAYA1自定义routing曾在token1读取token0 expert id作为weight；现选择fail-fast而不暗中copy。in-tree router原已连续，主要影响自定义扩展，没有新增通用速度收益。
**实践建议：**建议采用：自定义router在边界明确contiguous并做多token回归，M1无法暴露row-stride错误。
源码路线：`csrc/moe/moe_align_sum_kernels.cu；vllm/model_executor/layers/fused_moe/router/fused_moe_router.py::select_experts`。
一手来源：[vllm-project/vllm #57167](https://github.com/vllm-project/vllm/pull/57167)（main，2026-11-04T07:03:53+08:00，`fbb488f7d2d10e69edbcce02a4279c0046031cfb`）。

### Prefix hit 落在多模态 item 内时，scheduler 等待完整剩余 item

disable_chunked_mm_input路径将“缓存命中已在image内部”的剩余部分视为不可拆，在step预算不够时等待，补齐此前只保护item起点的逻辑。

**证据与局限：**针对item内部双向attention的正确性；不意味着所有多模态模型必须禁chunk，也不自动调整token budget以容纳任意大图。作者有scheduler与E2E回归，未报性能收益。
**实践建议：**建议采用：压测prefix hit恰在image/video span中间且本step预算不足的组合，核结果而不仅请求成功。
源码路线：`vllm/v1/core/sched/scheduler.py::multimodal scheduling`。
一手来源：[vllm-project/vllm #65271](https://github.com/vllm-project/vllm/pull/65271)（main，2026-11-03T20:06:41+08:00，`65a904538d294d9fafc044b164104b2f5f9a9391`）。

### FlashMLA sparse 的 BF16 KV 支持 decode context parallel

移除BF16 KV下DCP NotImplemented分支，局部cache attention后合并，BF16不沿用FP8 mixed-batch/head限制。

**证据与局限：**作者4GH200/TP4 GLM5.3-Flash、FP8 weights/BF16 KV，DCP4 KV token1.40×，GSM8K500 .894→.904（约1.3pp标准误），吞吐833→726token/s；用更多KV容量换通信代价。不是所有DCP越大越快。
**实践建议：**考虑复现：Hopper sparse MLA长上下文受KV限制时尝试，显式测试force_mqa让prefill也覆盖变更路径。
源码路线：`vllm/v1/attention/backends/mla/flashmla_sparse.py`。
一手来源：[vllm-project/vllm #61609](https://github.com/vllm-project/vllm/pull/61609)（main，2026-11-03T10:56:18+08:00，`bca1bc651624748313f9f9bfd3ef400e5f1061c0`）。

### Sampling 的 top-p/top-k 顺序在普通与投机路径统一

先按原概率mass确定top-p再top-k；稀疏top-k路径按保留mass调整p阈值，覆盖EAGLE/standalone/ngram与DFlash/DSpark/UNO target分布。CUDA非seed分支融合rescale，seed/非CUDA保留logsumexp；内建spec拒绝此前未应用的min_p>0。

**证据与局限：**作者H200多路径构造回归：原mass.8、小于p.82的token此前被top-k重归一化误删。最终FP32融合舍入仍可能改变cutoff，不能宣称与trainer或旧输出bitwise；GPU实测属于所写较早commit，非所有当前head CI。
**实践建议：**建议采用：对temperature/top-k/top-p/min-p联合设置做分布支持集与服务错误语义回归，普通sampling与spec不能各自解释同一参数。
源码路线：`python/sglang/srt/layers/sampler.py；speculative/{dflash_utils,eagle_utils,uno_utils}.py；managers/scheduler.py`。
一手来源：[sgl-project/sglang #41916](https://github.com/sgl-project/sglang/pull/41916)（main，2026-11-04T04:06:33+08:00，`71ef675ff584d01758d790e0c436f93e44f4097b`）。

### Fused RoPE KV store 在 DCP 上回退 owner-aware 路径

不再仅靠batch.dcp_kv_mask是否存在判断DCP，显式get_parallel().dcp_enabled即关闭generic fused KV write；交给attention backend做owner mask及loc//dcp_size翻译。

**证据与局限：**作者Qwen3-30B-A3B BF16、8A10/TP8/DCP2/Triton/BF16 KV、GSM8K1319/5shot/greedy，旧路径.99%→93.40%；是特定索引损坏修复，不能写模型能力普遍提升92pp。DCP1继续fusion。
**实践建议：**建议采用：CUDA eager缺mask的DCP组合尤其关键，回归长cache增长与illegal access；不能只用HIP mask存在的测试。
源码路线：`python/sglang/srt/models/utils.py::enable_fused_set_kv_buffer`。
一手来源：[sgl-project/sglang #47261](https://github.com/sgl-project/sglang/pull/47261)（main，2026-11-03T11:12:55+08:00，`f6db18e8c2d7fa52208543a2b49da0f6e3ee540d`）。

### Mega MoE 按 SM107 locality domain 放置权重，并修加载内存碎片

opt-in LOCALIZE用DeepGEMM domain-major物理存储，专家参数保留expert-major view供EPLB；非dense expert slice用contiguous staging P2P。超过token cap无非mega fallback而报错。随后先建scale再interleave并释放原shared/routed weights，降低reserved空洞。

**证据与局限：**仅CC10.7、sgl-deep-gemm≥.2.4/CUDA≥13.4，默认off。作者DSv4.1-Flash DEP4/EP4+DSpark，localized容量相对nonlocalized由-1.47%恢复到-.10%；不能称完全消除内存重复，部分loader segment仍被活weight牵住。性能未在第二项重新测，EPLB验证与weights布局是关键。
**实践建议：**深入阅读/考虑复现：同时核权重物理locality、EPLB交换与KV容量；高prefill上限先设足cap，避免原fallback被禁。
源码路线：`python/sglang/srt/layers/moe/mega_moe.py::build_mega_moe_*；eplb/expert_location_updater.py`。
一手来源：[sgl-project/sglang #47976](https://github.com/sgl-project/sglang/pull/47976)（main，2026-11-03T13:16:56+08:00，`4709a3a24e00e8d5408cb4968e4b919332006364`）；[sgl-project/sglang #48099](https://github.com/sgl-project/sglang/pull/48099)（main，2026-11-04T01:48:23+08:00，`7b88f092392a4b673c39b2a6b66a18131cdfb60e`）。

### HiCache write-back eviction 从逐victim等待改为一轮批量等待

同一eviction walk先排队所有D2H copy，再一次wait，最后按原顺序demote；Python/Rust树核心同步实现，处理同轮SWA/shared host状态，保持victim选择。

**证据与局限：**作者单GB300、gpt-oss20B、device约1/4history/host100GB、512client/96并行、1024首轮＋128后续/64输出/12round，req/s113→120.5、TTFT p99 813→559ms；Qwen3-8B+3.1%在噪声内，无eviction时中性。首copy仍等in-flight forward，不是无同步。
**实践建议：**考虑复现：eviction-bound多轮场景最合适，先核hit率、victim顺序与restore内容相同，再看scheduler idle是否减少。
源码路线：`python/sglang/srt/mem_cache/unified_radix_cache.py；unified_cache/unified_tree_core.py；rust/sglang-radix-tree/src/unified_tree_core.rs`。
一手来源：[sgl-project/sglang #46930](https://github.com/sgl-project/sglang/pull/46930)（main，2026-11-04T03:56:40+08:00，`d2a536fc532cc8cf1f70e0390adfa403c6fed09e`）。

### EAGLE/MTP 非 DSA draft KV 按真实 DCP-widened 大小计预算

draft每token价格用draft自身layer cell×dcp_size，替代target每层一行，修复MLA/GQA draft在DCP下被低估或误定价，分配布局本身未缩小。

**证据与局限：**作者MI355X DeepSeekR1 TP8/FP8 KV，DCP8 token capacity4.819M→4.330M，pool后free43.69→61.76GB；少容量是返还activation reserve。完整EAGLE+DCP在ROCm main仍因另一个draft-extend attention错误不能服务，因此没有吞吐对比。
**实践建议：**建议采用：先核静态预算与allocation一致，不把更大旧token数当优化；等另案修复后再评估真实服务。
源码路线：`python/sglang/srt/model_executor/pool_configurator.py::DefaultPoolConfigurator`。
一手来源：[sgl-project/sglang #39799](https://github.com/sgl-project/sglang/pull/39799)（main，2026-11-04T08:23:07+08:00，`dff6a0c3accf310f70eab81be1220a27b732c96b`）。

### Routed-expert 回传只在每个 DP group 的输出 rank 留 host cache

所有rank仍device capture/必要all-gather，只有scheduler is_rank_zero创建pinned host cache并D2H/readback，减少RL routing replay的重复CPU占用。

**证据与局限：**作者8H200 Qwen3-30B TP8、1024/256、三次中位：无DP host55→6.9GB，DP2约56.6→14.2GB，ITL基本不变。Nemotron101GiB/rank样例来自内部fork，不作为本main独立实测。
**实践建议：**考虑采用：CPU内存紧的GB300/RL优先，核PP/attn-TP/CP输出rank选择，不能删非输出rank参与的collective。
源码路线：`python/sglang/srt/state_capturer/base.py/routed_experts.py；runtime_context.py；managers/scheduler.py`。
一手来源：[sgl-project/sglang #47552](https://github.com/sgl-project/sglang/pull/47552)（main，2026-11-04T08:34:23+08:00，`91747384e5b1809d1977a9176ca79b7f78dc4184`）。

### Ascend950 的 MLA 接入 ModelSlim 静态 FP8 KV

新增KV-cache scheme接口与Q_FP8_DYNAMIC_KV_FP8，latent KV存FP8、side components留BF16，连通NPU pool、FIA v2和MLAProlog；要求匹配ModelSlim attention量化元数据。

**证据与局限：**作者950PR/TP16 KimiK3 W4A8C8、input16384/batch16，FIA decode140µs对BF16 KV204µs；仅算子不是整服务。GPQA .929另带#47811，不能当单项隔离质量实验。
**实践建议：**考虑复现：quantization=modelslim与kv-cache-dtype=fp8_e4m3同时匹配artifact，检查scale load/postprocess与跨rank KV布局。
源码路线：`python/sglang/srt/layers/quantization/modelslim/schemes/modelslim_q_fp8_dynamic_kv_fp8.py；hardware_backend/npu/attention/ascend_backend.py/mla_cache.py`。
一手来源：[sgl-project/sglang #46921](https://github.com/sgl-project/sglang/pull/46921)（main，2026-11-03T20:56:11+08:00，`99fd01047d67bb249641a8e50282a1a5218afc63`）。

### Kimi-K3 NPU DCP 接共享 compact 通信，并避 empty MoE launch

target MLA按物理page分片、draft KV复制；prefill重建完整prefix，decode/verify共享Q all-gather后compact A2A合并BF16/FP16输出与FP32 LSE，空query也参与；P/D限P侧DCP1→D侧N。empty TopK helper直接no-op避免NPU coreDim0。

**证据与局限：**#40825当前head只有CPU/CANN overlay 17回归及450配置重建测试，真实NPU kernels/完整serving/PD验证仍pending，正文旧模型数值不算当前head证据。#47821是独立empty-grid guard，不能代替DCP E2E认证。
**实践建议：**继续观察：先在匹配CANN/NPU环境测graph padded lengths、PP层配对、draft复制和P/D，然后再评估省KV收益。
源码路线：`python/sglang/srt/hardware_backend/npu/attention/{dcp,dcp_metadata,ascend_backend}.py；disaggregation/ascend/conn.py；kernels/ops/moe/fill_padded_rows.py`。
一手来源：[sgl-project/sglang #40825](https://github.com/sgl-project/sglang/pull/40825)（main，2026-11-03T09:19:17+08:00，`b8100b74bf4f6563f402c20a1e1af685cbd2246e`）；[sgl-project/sglang #47821](https://github.com/sgl-project/sglang/pull/47821)（main，2026-11-04T08:16:07+08:00，`d973326298e078c5959cdfe35e657094f1e9fff9`）。

### Paged Experts 默认 auto：先留 KV，再用剩余静态预算增加专家槽

初建每层K=top_k，按context/显式max-running-requests或max-total-tokens留KV，释放独立MemPool中的初始表后整轮增长所有层resident slots，最后一次建runner/graph cap；显式K保留。

**证据与局限：**仍单GPU且受既有paged-experts模型/量化支持边界约束；用户显式context放不下时报错，模型默认context放不下可缩并警告。不是动态运行时自动迁移或保证所有并发各有完整context；K/top_k限制full decode graph。
**实践建议：**深入阅读/考虑复现：小显存MoE部署记录自动K、KV tokens与请求目标，核多模态reserve/fragmentation，速度例K63→92不能泛化所有路由。
源码路线：`python/sglang/srt/layers/moe/paged_experts/sizing.py::kv_floor/ExpertSlotSizing；method.py::materialize；model_runner.py`。
一手来源：[sgl-project/sglang #47476](https://github.com/sgl-project/sglang/pull/47476)（main，2026-11-03T09:05:24+08:00，`8a0edcf799d06bdee8832356d382495e7fb50f22`）。

### MiniMax-M3.1 NVFP4 可经 unified-cache direct external linker 回存三类状态

pool group同时列KV payload/block scales、sparse INDEXER及sidecar DRAFT；原先没有MiniMax sparse group导致启动失败。

**证据与局限：**作者4GB300/TP4 MiniMaxM3.1-Flash+DSpark+Mooncake、9000token/96greedy，poison load target后两prompt KV/index/draft恢复与device hit相同；BF16 sparse pools、HiSparse、packed draft明确拒绝。2prompt是功能证据，不是可靠性压力认证。
**实践建议：**考虑采用：必须验证三组恢复；只核最终token可能漏draft缺失（验证步数会变），同时比较logprob与spec_verify_ct。
源码路线：`python/sglang/srt/mem_cache/hybrid_cache/linker_pool_assembler.py::_MiniMaxSparseStrategy.build_direct_linker_pool_group`。
一手来源：[sgl-project/sglang #48125](https://github.com/sgl-project/sglang/pull/48125)（main，2026-11-04T01:11:50+08:00，`1ea5f8c6f7f905e24758411ab59f09963ab68631`）。

### CUDA MoE router 用每 token 一 warp 的排序/归约路径

静态128–512experts、32倍数、特定sigmoid/sqrtsoftplus/softmax路由，把每lane专家先排序后只在lane heads做top-k；small wave一warp/CTA，否则四warp。覆盖group bias、共享专家、NaN-last、per-token bias与MuseSpark，未覆盖形状回Triton。

**证据与局限：**作者B200 graph router独立时间，例如Qwen3.5 T16384 63.1→15.6µs，DSv3 T1 1.88→1.43µs；不是整个MoE/服务4×。KimiK3 896expert、fp16、softcap等未覆盖，不可按标题泛化。
**实践建议：**深入阅读：router kernel可复现价值高，核最低expert id tie-break、renorm epsilon、padding与LoRA dispatch后再比较端到端。
源码路线：`python/sglang/kernels/jit/csrc/moe/route_warp.cuh；ops/moe/moe_route_warp.py；srt/layers/moe/topk.py`。
一手来源：[sgl-project/sglang #47641](https://github.com/sgl-project/sglang/pull/47641)（main，2026-11-03T20:30:45+08:00，`4494d682373e36ac89ca1bd8a52cd5266f9baf84`）。

### Kimi-K3 KDA decode 扩 NVIDIA head 合同，并调 gfx950 小batch tile

NVIDIA fused conv/recurrence/norm readiness从固定12head改核支持集合12/6/3，使TP16/32可选择既有kernel；AMD gfx950、12head/head128/≤4req改value tile32→8、1warp以增加workgroup。

**证据与局限：**NVIDIA TP16真实模型与kernel CI在PR中仍pending，不能报已实测融合加速。AMD作者MI355X TP8/DSpark k3+ReplaySSM、8192/1024，c1 TPOT4.37→4.30ms、c4 6.90→6.86ms；kernel10–24%快不等服务同幅提升。
**实践建议：**考虑复现：两平台分开验证ready gate与实际launch；支持head shape不自动证明完整模型/量化组合。
源码路线：`python/sglang/srt/models/kimi_k3.py::_prepare_fused_decode；kernels/ops/attention/kda_fused_decode.py；KDA recurrent launch selection`。
一手来源：[sgl-project/sglang #47526](https://github.com/sgl-project/sglang/pull/47526)（main，2026-11-04T08:39:50+08:00，`0f3c2c52ba7c1f3b8cdc0b65c449065467dd0a1e`）；[sgl-project/sglang #47064](https://github.com/sgl-project/sglang/pull/47064)（main，2026-11-03T17:26:34+08:00，`c97fe3559f54ed109e1cdbc0a8ec27f1bc272492`）。

### gfx950 deterministic 模式可显式选择 AITER paged decode

decode以每request固定256token KV partition归约具batch invariance，prefill仍用deterministic backend；只允许non-MLA、无spec、默认paged cache，SWA/unified/5D路径拒绝。

**证据与局限：**作者MI350/355 Qwen3系列deterministic吞吐有1.09–1.91×范围，但因模型/负载不同不能当统一收益；AITER prefill从约480token起观察到批形状影响，所以不能用attention-backend=aiter一键同时替换两阶段。
**实践建议：**考虑采用：显式decode-attention-backend=aiter与prefill后端分配，核batch/alone和SWA/spec guard，可靠性需求高时固定全部依赖。
源码路线：`python/sglang/srt/arg_groups/overrides.py::_deterministic_attention_backend；layers/attention/aiter_backend.py`。
一手来源：[sgl-project/sglang #44489](https://github.com/sgl-project/sglang/pull/44489)（main，2026-11-03T14:05:31+08:00，`cbfc0932e3be191e8bbb0873cac35c8ceb65313b`）。

### gfx115x unquantized BF16/FP16 weight 行加128B padding 避 cache alias

当本地weight row byte stride为2KiB倍数，加载后padding使L2/MALL通道错开；涵盖linear和真正lm_head/绑权头，untied输入embedding不动，逻辑shape和值保持。

**证据与局限：**作者gfx1151 Llama3.1-8B形状cold M1 hipBLASLt合计89→67.6ms，FP32 gate不pad；默认只gfx115x，未实测型号不等于同样提升。增加物理存储，opt-out可用；RL更新也需保留新stride。
**实践建议：**考虑复现：消费级RDNA推理记录stride、峰内存及LM head绑权，不能把kernel带宽收益当直接token/s。
源码路线：`python/sglang/srt/layers/quantization/unquant.py::process_weights_after_loading；model_loader/loader.py::finalize_gfx115_weight_padding`。
一手来源：[sgl-project/sglang #46327](https://github.com/sgl-project/sglang/pull/46327)（main，2026-11-03T15:58:11+08:00，`a470d74cdb2e92f851dd94269b8f82148be64de2`）。

### DSv4.1 在 SM107 的 dequant、pre-dispatch 与 indexer glue 减少小program/多launch

paged KV dequant每program八token/两warp并连续FP4 store；prefill-sized mega pre-dispatch在门槛上用四chunks/thread；低ratio indexer页表一个Triton kernel扩展；非paged BF16 top512也可按轮数选3CTA/SM。

**证据与局限：**作者SM107 isolated graph kernels：dequant特定大token形状低37–80%但只占prefill kernel和约.9%；页表ratio1/2合计约67→14µs；pre-dispatch16384rows低约24%。均未测整服务，dequant统一新launch在gfx950未验，不能堆叠百分比。
**实践建议：**考虑复现：SM107平台按真实rows/width/ratio分开测，保留尾mask、row stride、top-k集合与int32 edge case逐位对照。
源码路线：`python/sglang/kernels/ops/attention/dsv4/dequant_k_cache.py；jit/csrc/deepseek_v4/mega_moe_pre_dispatch.cuh；srt/layers/attention/dsv4/metadata.py；ops/attention/deep_select.py`。
一手来源：[sgl-project/sglang #48243](https://github.com/sgl-project/sglang/pull/48243)（main，2026-11-04T08:25:19+08:00，`cb7d232f6ec0e045de6d56fe2dfa9f8b67d38c3f`）；[sgl-project/sglang #48245](https://github.com/sgl-project/sglang/pull/48245)（main，2026-11-04T07:20:45+08:00，`3384ec8890d8cc6289696a6bde8da99464d971cf`）；[sgl-project/sglang #48229](https://github.com/sgl-project/sglang/pull/48229)（main，2026-11-04T06:58:51+08:00，`76b4c91360da67a2985500b82560f02278a2fff3`）；[sgl-project/sglang #48246](https://github.com/sgl-project/sglang/pull/48246)（main，2026-11-04T08:33:37+08:00，`f4b1dc820d519a94bb396893c5cee3ba4c38a0cc`）。

### Scheduler 补 HiSparse 异常退出释放与所有 request-list 的 weight-version 事件

KV checksum mismatch及chunked abort先request_finished再free KV，归还host/device/spec scratch；weight更新遍历统一request_lists，补grammar/PD/dLLM等排队请求此前漏掉的epoch事件。

**证据与局限：**前者泄漏由代码退出路径与回归确认，E2E自然泄漏复现仍在做；后者不改变已在running/waiting的常规记录，base/extra/AMD CI存在非全绿，不能称所有scheduler生命周期完成认证。
**实践建议：**建议采用：RL热更新与P/D用户分别注入checksum失败、chunk abort、grammar等待跨weight更新，检查资源回收和epoch审计。
源码路线：`python/sglang/srt/disaggregation/decode.py::_handle_kv_checksum_mismatch；managers/scheduler.py::process_pending_chunked_abort/record_weight_version_change`。
一手来源：[sgl-project/sglang #43911](https://github.com/sgl-project/sglang/pull/43911)（main，2026-11-04T08:54:50+08:00，`d091f25872bfc9cfa3dd3515e44c754008dab66a`）；[sgl-project/sglang #43909](https://github.com/sgl-project/sglang/pull/43909)（main，2026-11-03T16:53:40+08:00，`12e741c21fbc125da3c231719d937e075788bb2f`）。

### Grammar poll timeout 不再被永久缓存为无效 schema

timeout只取消/abort当前请求，删除负cache写入，后续同schema可重试；真正backend invalid仍缓存。此前executor排队也计poll，突发负载会把合法schema永久毒化至flush/restart。

**证据与局限：**Future.cancel不能停已运行编译，llguidance/outlines等无in-flight dedup的慢schema可能反复占executor；HTTP400与ready分支意外异常负cache尚未改变。不能称彻底解决grammar DoS或timeout分类。
**实践建议：**建议采用：验证超时后相同schema可恢复，同时监控编译队列与CPU；等待per-key dedup后再缓解持续慢编译。
源码路线：`python/sglang/srt/constrained/grammar_manager.py::get_ready_grammar_requests`。
一手来源：[sgl-project/sglang #45466](https://github.com/sgl-project/sglang/pull/45466)（main，2026-11-03T19:38:40+08:00，`035670b6854176b9176ddc7c9d506564c3a69a7c`）。

### Hash MoE layer 在 prefill fusion predicate 前先分流

breakable prefill dual-stream graph及vision top-k读取topk_config前先检查is_hash，保留带input_ids的HashTopK，避免不存在属性启动崩溃。

**证据与局限：**只改predicate顺序，无任务质量/速度实验；eager与禁prefill graph原本不触发。不能视为新增hash路由算法。
**实践建议：**建议采用：hash-routed模型开启breakable prefill graph时回归，保持非hash同路径对照。
源码路线：`python/sglang/srt/models/deepseek_v2.py::_can_dual_stream_graph/vision_topk`。
一手来源：[sgl-project/sglang #48041](https://github.com/sgl-project/sglang/pull/48041)（main，2026-11-03T20:34:44+08:00，`1ef3335da1142c19853996d93c5c1b8bc0e84b58`）。

### Qwen MRoPE decode 复用 fused table QK-norm/RoPE kernel

CUDA BF16、2D三轴positions、head64/128的decode用table kernel，warmup时将cache转activation dtype，减少norm/rotation多launch；extend、spec verify、1D及head256 MRoPE仍split。

**证据与局限：**head256 table BF16旋转与旧triton_mrope约14%element差1ULP，因此明确不启用；被支持布局与partial-head block有逐位回归，不能泛化所有MRoPE。
**实践建议：**考虑采用：图像模型decode关注position layout/stride和cache dtype，graph capture前完成转换。
源码路线：`python/sglang/kernels/ops/attention/table_qk_norm_rope.py；srt/models/qwen3.py`。
一手来源：[sgl-project/sglang #47993](https://github.com/sgl-project/sglang/pull/47993)（main，2026-11-03T14:30:20+08:00，`80121f264fff931d72c33b2d1faa89357c4d6fb6`）。

### 集成分支：output-store replay 按 layer blocks 分字段

sglang-miles分支接replay_layer_splits，把routed expert/indexer rows切成{name}.{start}-{end} Mooncake keys，使trainer PP/VPP只读自己的层；DSv4按C4 indexer slot重映射，未split也用单whole block。

**证据与局限：**仅合入sglang-miles，不是main发行。与Miles侧和router字段转发联动，旧裸routed_experts字段拒绝；SGLang自身PP capture仍明确禁止，不能把trainer PP节省误写成server PP已能返回全层capture。
**实践建议：**继续观察：验证consumer/provider字段同步升级与跨VPP切分，先单独证明读字节和writer布局；不要套用到正式包。
源码路线：`python/sglang/srt/managers/output_store.py；configs/model_config.py::indexer_slots_at；arg_groups/validation_hook.py::check_pipeline_parallel_compat`。
一手来源：[sgl-project/sglang #48123](https://github.com/sgl-project/sglang/pull/48123)（sglang-miles，2026-11-04T08:01:58+08:00，`321446ac8ffbd93c4d14f93ef676a15b0431de49`）。

### 集成分支：AuxOutput 可回放普通 sparse indexer TopK

rl/aux-output分支通过SHM connector输出int32[tokens,sources,topk]，offline属性与Python API base64.npy、Rust token-in/token-out均接通，支持DS3.2/4/GLM DSA与shared MLA reference。

**证据与局限：**不是main能力；retarget原#49555，diff带该分支落后的main更新，不把整份diff行数算成新功能。作者H20e TP2/8/DP2×TP2与并发隔离验证，不能扩到所有indexer模型。
**实践建议：**继续观察：训练replay研究可固定分支SHA，但正式服务需等main合并及完整消费链；验证DBO capture offsets。
源码路线：`vllm/distributed/aux_output_connector/*；model_executor/layers/indexer_capture.py；outputs.py；Python/Rust scale-out endpoints`。
一手来源：[vllm-project/vllm #65289](https://github.com/vllm-project/vllm/pull/65289)（rl/aux-output，2026-11-03T20:45:28+08:00，`562143ca7468ae161409db1a40b9505b8940fe68`）。

### 更正：撤回 Kimi-K3 写入共享 runner 的固定 FlashInfer tactic

回滚昨天#47653在model-specific代码中向共享autotune context灌入checked-in tactic；删除TP1/PP8/EP1、chunk16384、BF16、FI.7.2/SM103专用表，使用已有部署cache机制。

**证据与局限：**这是已合入的revert，不能继续按此前日报把该预置tactic当当前main默认优化；原因是部署/版本绑定过窄与职责位置，不是已证实所有该tactic数值错误。
**实践建议：**建议采用：需要复现实测tactic时放部署autotune cache并锁版本，升级依赖重新验证，不能把历史main能力沿用。
源码路线：`python/sglang/srt/model_executor/runner/flashinfer_autotune.py；removed flashinfer_kimi_k3_tactics.py/kimi_k3_sm103_fi072_prefill.json`。
一手来源：[sgl-project/sglang #48055](https://github.com/sgl-project/sglang/pull/48055)（main，2026-11-03T11:28:58+08:00，`af98fcd545f90eaaf4b26d20bfb42e3ec1d91020`）。

### Weight-cache restart 扩 online FP8/MXFP8 与 ModelOpt CUTLASS 路径

IPC复用已处理weight时重建运行时layout/逻辑output width，避免再quant/transpose/pad，NVFP4 CUTLASS MoE避免activation scale重复乘入。另支持ModelOpt per-tensor及per-channel/per-token FP8 linear的CUTLASS默认format。

**证据与局限：**online MoE、从已量化checkpoint再online quant、ModelOpt block FP8/其他backend不在范围。作者B300 torch2.13/cu130＋局部CUDA13.4兼容库、TP1/eager/固定小模型cold/missing-daemon/两次IPC逐位smoke，未验证上游torch2.14全矩阵、多GPU/ROCm或服务加速。
**实践建议：**考虑复现：冷启动频繁时试，但必须禁止silent fallback来确认确走IPC；记录共享weight bytes未被二次修改。
源码路线：`vllm/model_executor/layers/quantization/modelopt.py；online/{fp8,mxfp8}.py；kernels/linear/scaled_mm/cutlass.py；fused_moe/experts/cutlass_moe.py`。
一手来源：[vllm-project/vllm #63754](https://github.com/vllm-project/vllm/pull/63754)（main，2026-11-04T02:06:34+08:00，`39a7f9bf50c08ea146c4dca62883e1927f89de3e`）；[vllm-project/vllm #63672](https://github.com/vllm-project/vllm/pull/63672)（main，2026-11-03T21:57:10+08:00，`44965a10974ee417f6f937eee207409462b2a0ab`）。

### 集成 GPU weight load 先在 CPU prefault mmap 页

copy_weight_在CUDA integrated GPU、contiguous同dtype CPU source且元素数>1时先每page读一byte及尾byte，再copy，接入共享loader26处；discrete或需CPU转换的source不做该touch。

**证据与局限：**作者GB10指出约14万小tensor在driver copy内page fault显著慢于CPU prefault；本文不转录无完整条件的总加载倍数。额外CPU访问与Python检查也有成本，host page cache状态/NFS/NVMe影响很大；不是pinned memory或GDS。
**实践建议：**考虑复现：DGX Spark小tensor checkpoint加载测cold/warm page cache，保留same-dtype gate与empty/noncontiguous回归。
源码路线：`vllm/model_executor/utils.py::copy_weight_；model_loader/weight_utils.py；parameter.py；layers/linear.py`。
一手来源：[vllm-project/vllm #58868](https://github.com/vllm-project/vllm/pull/58868)（main，2026-11-03T15:14:20+08:00，`437288d18a613c5c8d0f06fbea704dd05c386435`）。

### MoRIIO native wait 改10ms切片，避免成功传输仍等满30秒

WRITE完成与READ barrier重复短wait_all并在间隙检查状态，保留整体transfer_timeout与fail_fast语义，绕开xGMI native wait单次poll后睡至deadline。

**证据与局限：**作者MI355X/MoRI1.2.3.post1，118层×128×32KiB跨GPU写：原6/6等29.999s、修后100/100中位.210s；DSv2-Lite P/D 16k/c8的原版会留请求不完成，修后三次96请求全部完成。特定MoRI版本/backend问题，不能宣称传输带宽140×。
**实践建议：**建议采用：同栈P/D优先，核实际copy状态与wait耗时，超时错误与故障退出仍需独立测试。
源码路线：`vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py::MoRIIOWrapper.waiting_for_transfer_complete`。
一手来源：[vllm-project/vllm #64330](https://github.com/vllm-project/vllm/pull/64330)（main，2026-11-03T16:46:58+08:00，`a8718969d2c3910625e381d503ecde28b4b50ab7`）。

### ROCm 无有效 grouping 的 sigmoid+bias top-k 换 register Triton

在统一AITER wrapper拦截topk_group≥num_expert_group，group扫描不能排除任何expert时直接flat top-k，覆盖GroupedTopKRouter与FusedTopKBiasRouter，保持bias/tie/renorm语义。

**证据与局限：**仅≤512experts/top8且graph路径；gfx950大top16实测反而慢，eager也不快，均保留AITER。作者routing-op gfx942约1.67–2.15×、gfx950约1.16–1.57×，无整MoE等幅结论。
**实践建议：**考虑复现：检查router实际传入group数，尤其人为合成2/2场景；不得简单将所有MoE换新kernel。
源码路线：`vllm/_aiter_ops.py::biased_grouped_topk；layers/fused_moe/router/rocm_sigmoid_bias_flat_topk.py`。
一手来源：[vllm-project/vllm #63076](https://github.com/vllm-project/vllm/pull/63076)（main，2026-11-03T12:11:32+08:00，`8a702b18a143ef55ceb639b227e897bfd997f5c4`）。

### Triton attention metadata 依据每个 KV group 的 head geometry

3D阈值和split-K scratch从group.kv_cache_spec取KV head数与head_size，修Gemma4异构组误用全模型8head/512dim而错选2D/过大scratch；homogeneous不变。

**证据与局限：**作者MI325X、Gemma4 group shape、ctx4096/batch24 kernel506→172µs；正文标+66%其实对应时延约-66%，吞吐倍数约2.94，避免混写。服务+6.7%/TPOT-11.5%属于旧v.29 MI355X，未在当前main重测；换reduction次序可改变输出。
**实践建议：**考虑复现：以本main重新测端到端，特别检查capture-size snap与max_num_seqs不在grid的scratch边界。
源码路线：`vllm/v1/attention/backends/triton_attn.py::TritonAttentionMetadataBuilder`。
一手来源：[vllm-project/vllm #56996](https://github.com/vllm-project/vllm/pull/56996)（main，2026-11-03T12:48:24+08:00，`a0d909f5395b626421b6f1d9634371a2f5546318`）。

### Pixtral 与 Mistral3 视觉 encoder 支持 CUDA graphs

变分辨率patch打包到固定capacity buffer，预计算attention metadata、RoPE和patch merger；保留max_seqlen在CPU避免capture内.item，unsupported attention回eager。

**证据与局限：**覆盖PixtralForConditionalGeneration与Mistral3ForConditionalGeneration。作者单H200 Ministral3-3B、224²单图、TP1/200请求与MistralLarge3 NVFP4 TP4服务验证；不同encoder/分辨率和backend不能由单点证明一致收益。
**实践建议：**考虑复现：vision encoder短小launch多时试，扫capacity padding、多图、后端fallback与峰值，不能把整请求数秒时延都归为encoder。
源码路线：`vllm/model_executor/models/{pixtral,mistral3}.py；v1/worker/encoder_cudagraph.py`。
一手来源：[vllm-project/vllm #56277](https://github.com/vllm-project/vllm/pull/56277)（main，2026-11-04T02:25:28+08:00，`2142c95e67bc9bebc9fd9fd89b2e323ee96c09c1`）。

### MiniMax-M3 sequence-parallel input norm 保留 BF16 到 gather 后再量化

MegaMoE+TP>1+EP自动SP时，input norm不再直接输出MXFP8 QuantizedActivation给all-gather，而由qkv_proj在gather后量化；SP关闭保持fusion。

**证据与局限：**作者4B300 MiniMaxM3.1-Flash，旧版profile因无.contiguous崩溃，修后GSM8K1319约.958/.964（是否GEMM_RS两配置）；KV路径在该机受CUDA版本限制，不能把这组实验当NVFP4 KV全路径认证。
**实践建议：**建议采用：升级模型私有norm融合时逐一检查collective消费tensor合同，测试SP/EP与quant输入的交叉组合。
源码路线：`vllm/models/minimax_m3/nvidia/model.py::input norm/qkv path`。
一手来源：[vllm-project/vllm #65496](https://github.com/vllm-project/vllm/pull/65496)（main，2026-11-04T05:51:19+08:00，`cc329910bd2ed7e9e57af468df2e6f1f803f906b`）。

### 两处启动 guard：空 compressed-indexer prefill 与 Step5 tail-ring page 大小

DSv4 sparse indexer以真实nonempty metadata chunks判断prefill，避免没有compressed KV却建空workspace；Step5 backend按CSA tail-ring所需字节自动提高默认block-size，替代要求用户手设32。

**证据与局限：**前者不增加算法计算；后者FP8高TP才需更大page，BF16及TP1/2可能保持16。作者Step5 MI300复现、MI325启动通过，两修复不是所有模型通用更大block策略或性能优化。
**实践建议：**建议采用：启服回归极短/无compressed token与未设block-size两路径，检查实际pool layout而非只看默认参数。
源码路线：`vllm/model_executor/layers/sparse_attn_indexer.py；models/step5/common/csa_tail_ring.py`。
一手来源：[vllm-project/vllm #65046](https://github.com/vllm-project/vllm/pull/65046)（main，2026-11-04T04:21:29+08:00，`b80941d3cad8e6cd6398081fd4aee321ba7f9db4`）；[vllm-project/vllm #65153](https://github.com/vllm-project/vllm/pull/65153)（main，2026-11-03T14:41:07+08:00，`893d65b0a6f9e30e83e1a85a79562250c2dd1059`）。

### 核心域覆盖、未完成区间与状态更正

六仓回溯范围完整分页枚举 637 个 main commits、656 个 merged PR 和 258 个 release 对象；release metadata/body 与上一轮同ID快照无变化，因此不重复抄录已有发行。THUDM/slime 本窗唯一 #2514 是 Megatron/image 依赖基线升级，未单独作为技术成果。
本文 93 组含 117 个稳定 PR 事件（114 main、1 release 回补、2 集成分支），并非同数独立成果。627 个当前或继承候选仍待详审；全部旧区间由 inherited-backlog.json 保留，枚举成功不等于语义覆盖完成。
状态增量：昨日仍 open 的 TorchTitan #5731 本日合入，已按新 merge SHA 收录；历史 URL 出现不是排除新状态的理由。SGLang #47653 的默认固定 tactic 被 #48055 回滚，本期作更正。vLLM #61667 仍 open，P2P consumer slot quarantine 不能被本日其他拆分修复替代。

<a id="extended"></a>
## 扩展开源工程、训练系统与硬件适配

本节逐项引用已打开的官方 PR 最终文件或真实主干 commit。各组列出合入目标和 SHA；主干落地不代表已经进入正式包。性能/设备测试均为提交者证据，本轮未独立复测。生成内核只审关键接口、同步/地址路径和测试合同，未声称逐行审完全部生成代码。

### DeepSpeed 确定性策略落地：固定算法选择仍需整训验收

**deepspeedai/DeepSpeed；本轮发现，原始合入时间及版本：** [#8917](https://github.com/deepspeedai/DeepSpeed/pull/8917)（2026-11-03T23:19:14Z，`dfc63adfbf`，默认分支已见）

新增默认关闭的 JSON 布尔项 `deterministic_mode`，当前只允许 NVIDIA CUDA。引擎初始化启用 PyTorch 严格确定性、禁 TF32/cuDNN benchmark，要求在首次 CUDA 使用前已设 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 或 `:16:8`，不会事后补设。FPDT 将状态传给 FlashAttention backward；legacy stochastic transformer 被拒；Triton grouped GEMM 绕过计时调优、固定首个配置，并显式选 IEEE dot，因为 PyTorch 的 TF32 开关不控制 Triton。

**保证范围有限。** 不改梯度精度、GAS、TP/EP、NCCL 算法或通信重叠；进程全局设置不在 teardown 恢复。作者单 L20、FP32 AdamW、ZeRO0/2/3、GAS2、3次更新、两独立进程逐位重复的32项结果早于自定义路径补充；后补自定义策略验证是 CPU 测试，不能当分布式整模逐位保证。建议深入阅读并在实际初始化、RNG、数据序、checkpoint恢复与通信拓扑上复验；固定 kernel 选择可能损失性能。

### Transformers：QSA NaN、专家负载损失与 RoPE 精度修复

**huggingface/transformers；本轮发现，原始合入时间及版本：** [#49923](https://github.com/huggingface/transformers/pull/49923)（2026-11-03T14:17:14Z，`23ab86437c`，默认分支已见）；[#50266](https://github.com/huggingface/transformers/pull/50266)（2026-11-03T12:21:56Z，`fa83babafe`，默认分支已见）；[#50275](https://github.com/huggingface/transformers/pull/50275)（2026-11-03T11:35:17Z，`89e9737d78`，默认分支已见）

Qwen4-Exp eager QSA 的选择 mask 保持布尔型，再与因果 mask 做交集/`masked_fill`，避免两个 minimum-finite 相加溢出为 `-inf`。左 padding 的全遮行 softmax NaN 会经 GDN 卷积/递归污染有效 token，不能仅忽略 padding logits；即使 token budget 大于输入长度也会触发。

Doge 将按排名的 router 概率 scatter 回 expert ID 后才与 expert load 配对，新增专家重命名不变性回归，修的是辅助损失目标。ZeRO3 构造前的 context manager 则只阻止 `torch.arange` 被 zero.Init 下转 dtype，保留设备选择，并在退出恢复包装器，保护 RoPE buffer。建议对应训练/加载用户优先做有限值、loss不变性与长位置精度回归；三项均无端到端加速主张。

### Mistral Large4 接入 KV 共享合同，indexer 仍需独立 cache

**huggingface/transformers；本轮发现，原始合入时间及版本：** [#50305](https://github.com/huggingface/transformers/pull/50305)（2026-11-03T15:28:37Z，`4b7a509a94`，默认分支已见）

在昨日通用 KVSharingDict 合同之上，Mistral Large4 从 `kv_source_layers` 推导 `store`/`reuse`；复用 KV 但自己计算 indexer 的层用 `reuse_but_indexer`，仍分配 IndexedLayer，其他纯消费层可走 DummyLayer。共享容器交给跨设备迁移处理，移除手工 `.to()` 和旧 kwargs共享状态入口。

这是 cache 表示与模型集成变更，不是改变权重的共享比例。建议 Large4 多设备/FSDP2 和长上下文用户对比 prefill/decode、indexer长度及 cache占用；目前没有可归因于此 PR 的整模显存百分比。

### Activation checkpointing 公共命名迁移及其兼容修补

**huggingface/transformers；本轮发现，原始合入时间及版本：** [#49248](https://github.com/huggingface/transformers/pull/49248)（2026-11-03T10:28:35Z，`a0cf918fd0`，默认分支已见）；[#50310](https://github.com/huggingface/transformers/pull/50310)（2026-11-03T15:53:25Z，`4a696b1c79`，默认分支已见）

公共参数/方法改为 `activation_checkpointing*`；旧 `gradient_checkpointing*` 保留带 FutureWarning 的别名，计划 v6移除。Trainer 改用新入口后，旧方法的子类/实例 wrapper 一度被绕开，可能报错，也可能静默失去 SAC/diffusion 补丁。随后 #50310 在过渡期检测旧方法 override 并回调，避免递归；直接替换 PreTrainedModel 基类方法的 monkeypatch 不在覆盖内。

建议库维护者检查真正入口与 `context_fn` 是否执行，并区分 TrainingArguments 的通用重计算与 fsdp_config 的 FSDP 实现，不能因名称相同混用。TRL 当天的版本分流补丁及回滚见下一条，升级应按兼容 commit 组合验收。

### TRL：GKD 按真实生成 token 归一化，SAC 回到兼容入口

**huggingface/trl；本轮发现，原始合入时间及版本：** [#7925](https://github.com/huggingface/trl/pull/7925)（2026-11-03T19:39:31Z，`f7d6dce07c`，默认分支已见）；[#8094](https://github.com/huggingface/trl/pull/8094)（2026-11-03T13:12:24Z，`5a976b9128`，默认分支已见）；[#8098](https://github.com/huggingface/trl/pull/8098)（2026-11-03T16:33:19Z，`5dd4878755`，默认分支已见）

GKD 原先先按数据集答案计 `num_items_in_batch`，随后 on-policy/seq-KD 又替换为生成答案，JSD 分母与实际目标错配。生成前移到 `get_batch_samples` 收集累加窗口阶段；作者复现分母172→16，即真实非 ignore 生成标签数。建议蒸馏用户以不同答案长度核loss/梯度，不能把旧学习率经验原样视为新目标等价。

SAC 的 #8094 曾按 Transformers 版本包装新/旧入口，#8098 在上游 #50310 修兼容后将其回滚，最终仍包装 `gradient_checkpointing_enable`。作者给5.22与上游修补版本的GPU/SFT+LoRA验证；不要将当天中间方案作为最终API。

### TRL：权重同步过滤参考适配器，教师长度错误提前暴露

**huggingface/trl；本轮发现，原始合入时间及版本：** [#8079](https://github.com/huggingface/trl/pull/8079)（2026-11-03T20:56:50Z，`15dda0c527`，默认分支已见）；[#8015](https://github.com/huggingface/trl/pull/8015)（2026-11-03T09:00:36Z，`9ec96f74c4`，默认分支已见）

GRPO（beta非零）/RLOO加载带 `modules_to_save` 的预训练 adapter 时，会把冻结的ref副本送给vLLM导致首同步失败。同步迭代器现在仅保留 default 副本，其他adapter和original_module跳过；同步改动也覆盖OnlineDPO/AsyncGRPO/AsyncDistillation。代码具体检查 `modules_to_save.default.`，不能扩展为任意动态adapter切换已验。

AsyncDistillation读取教师 `/v1/models` 的 `max_model_len`，在真实请求评分前检查 prompt+completion 还是否能多容纳1个评分token；不能容纳则明确报错，避免原来400被当传输错反复退避。缺该metadata时跳过检查，通用4xx重试策略未在此解决。建议复测满长rollout、多教师及PEFT同步，均不是训练倍率结果。

### DeepSeek 训练模板继续修正：保留 reasoning 与轮次前缀掩码

**huggingface/trl；本轮发现，原始合入时间及版本：** [#8061](https://github.com/huggingface/trl/pull/8061)（2026-11-03T08:51:29Z，`ce62ef1524`，默认分支已见）；[#8060](https://github.com/huggingface/trl/pull/8060)（2026-11-03T08:51:28Z，`6ffec466cf`，默认分支已见）

DeepSeek-R1/V3训练模板移除按 `</think>` 丢弃历史reasoning的推理习惯，使其保留在assistant训练目标；assistant开场 `<｜Assistant｜>` 与工具结果结束后开启assistant的前缀则移出generation块，避免掩码把提示前缀算入assistant loss。#8061先入栈分支，其merge SHA已出现在本窗main，故按主干落地记，不误写为仅支线。

建议对真实多轮/工具样本同时查看渲染文本与逐token loss mask。它修数据合同，不证明既有模型能力获得提升。

### FlashInfer GEMM workspace 按 stream 隔离，仍有共享边界

**flashinfer-ai/flashinfer；本轮发现，原始合入时间及版本：** [#8221](https://github.com/flashinfer-ai/flashinfer/pull/8221)（2026-11-03T21:52:35Z，`25cdfe1aaf`，默认分支已见）

GEMM临时缓冲由每device缓存改为现有的每stream缓存，涵盖split-K/stream-K中间量及grouped参数数组；分配来自FlashInfer私有MemPool，避免首次调用落入torch.compile cudagraph trees的临时私有池。低延迟FP8路径传固定长度slice，让调优key不随workspace增长变化。

作者GB300、K8192、N9216/4608/2304及64/32/16、16层、两stream同graph、200replays：cuBLAS/cuDNN/auto相对串行的逐位差异由最多3061/262/503次降至0（6400样本），grouped错输出894/960→0。**同一stream捕获的graphs仍共享workspace，不能并发replay；其他采样/prefill/GDN scratch后续仍需另查。** 建议多stream并行、RL overlap用户优先复现；小batch CUTLASS K-major scales另有间歇失败未由此解决。

### FlashInfer GDN MTP 状态池：64位寻址、stride与编译缓存修正

**flashinfer-ai/flashinfer；本轮发现，原始合入时间及版本：** [#7882](https://github.com/flashinfer-ai/flashinfer/pull/7882)（2026-11-03T20:44:09Z，`724920fdba`，默认分支已见）

逐token `[B,T]` scatter路径原先32位offset会在生产级pool越界；K=V128、HV12的FP32池，slot约10923起即可越过有符号32位元素offset。修复把64位slot/head偏移折入二维view基址，再使用局部32位tile；同时接入page-strided四维pool，编译key加slot数，避免profiling小池与实际大池错误复用。intermediate cache的batch×steps×heads也同受保护。

文档明确逐token scatter indices **没有负数padding约定**，每个都必须合法；应通过initial_state_indices为整请求padding。作者给FP32参考和真实服务pool故障证据，本轮未复跑。建议Qwen3.5类MTP集成覆盖大slot、页padding、小池→大池切换及负索引拒绝。

### FlashInfer 采样分布与接受率统计分开修

**flashinfer-ai/flashinfer；本轮发现，原始合入时间及版本：** [#8002](https://github.com/flashinfer-ai/flashinfer/pull/8002)（2026-11-03T21:34:51Z，`0a3a715d11`，默认分支已见）；[#8519](https://github.com/flashinfer-ai/flashinfer/pull/8519)（2026-11-03T17:04:10Z，`927754255e`，默认分支已见）

大词表top-k-first快速路径现在支持同device int32/int64一维indices，保持exact-k边界tie规则不因添加身份行映射改变。V≥65536、k≤256的indexed固定seed样本可能变化；有边界tie时原fallback保留全部并列项而快速路径只留k，分布语义也需要按新合同验收。

链式投机接受率原将首次拒绝位置再次计入，单token alpha=.5会报告期望.75；现在不计第二次但仍消耗同随机数，保持该统计修复的生成输出序列。AlphaMoE全NaN行路由则由越界expert修为0..top_k-1和NaN权重，阻止plan越界，不把坏数变有效预测；**部分NaN且有限项不足top_k仍可能重复expert**。建议采样基准重置统计口径，并测试tie/NaN/duplicate indices。

### FlashInfer FP4量化：不支持架构明确拒绝，FP8 TMA按真实tile门控

**flashinfer-ai/flashinfer；本轮发现，原始合入时间及版本：** [#7987](https://github.com/flashinfer-ai/flashinfer/pull/7987)（2026-11-03T21:35:15Z，`3346d95955`，默认分支已见）；[#6276](https://github.com/flashinfer-ai/flashinfer/pull/6276)（2026-11-03T17:05:28Z，`728a74876d`，默认分支已见）

FP4 CUDA/CuTe入口在实际使用的device上检查SM100+，避免旧架构kernel无实现却留下未写输出；不改变量化算术。FP8→NVFP4 TMA descriptor不再给UINT8类型请求仅允许浮点的OOB fill；FP8每producer step实际1024列，过去误用BF16的512列门槛。现在FP8列数为512奇数倍时回退非TMA，避免修复descriptor后又暴露scale尾写越界。

新增m1024/4096、n512/1024/1536/2048、三种scale layout与分块非TMA逐位对照。建议混合硬件部署和在线量化先验device选择、输出完整覆盖及边界列；仅版本升级不能意味着Hopper原生支持这些FP4量化kernel。

### FlashInfer RoPE append与FMHA同步：错误前移及合法调度回归

**flashinfer-ai/flashinfer；本轮发现，原始合入时间及版本：** [#7494](https://github.com/flashinfer-ai/flashinfer/pull/7494)（2026-11-03T19:40:14Z，`f22d616373`，默认分支已见）；[#8424](https://github.com/flashinfer-ai/flashinfer/pull/8424)（2026-11-03T02:31:46Z，`ce28c2fac0`，仅核实合入 main）；[#8384](https://github.com/flashinfer-ai/flashinfer/pull/8384)（2026-11-03T01:37:18Z，`626cf464b2`，仅核实合入 main）；[#8445](https://github.com/flashinfer-ai/flashinfer/pull/8445)（2026-11-03T05:29:42Z，`f583862840`，仅核实合入 main）

融合RoPE/FP8 paged append为非RoPE尾chunk补边界，尊重独立V cache stride；host要求RoPE/no-RoPE维为16倍数、各行16字节对齐，并检查K/V几何相同（页数不强制相等，各自须容纳页表）。页号值未做需要D2H的完整范围校验。

FMHA三种FP8/FP16 context JIT给最后row statistics独立barrier，避免两次相位先完成而correction漏等：作者给合法200µs延迟调度在B200/GB300复现hang，不能只用正常计时未挂作正确性证据。四个MoE/DeepGEMM家族则把cluster同步移到双CTA TMEM分配前。另一page32 slot修复只让被predicate关闭的地址也留在stage内，作者明确原来未实际越界写。建议用sanitizer与故障调度验收；这些不是整模加速项。

### FlashInfer 长上下文MQA：分页与dense分别按不同整数界限切分

**flashinfer-ai/flashinfer；本轮发现，原始合入时间及版本：** [#8315](https://github.com/flashinfer-ai/flashinfer/pull/8315)（2026-11-03T01:00:05Z，`0ff77be8f5`，仅核实合入 main）；[#8308](https://github.com/flashinfer-ai/flashinfer/pull/8308)（2026-11-03T01:15:15Z，`f5c654f493`，仅核实合入 main）

paged FP8 MQA logits超过有符号2^31元素时，按请求range分次launch并偏移输出row view；dense 32-head FP8 logits超过2^32元素时，按完整query block range切分。Plan对象对过大计划明确拒绝，避免把kernel内32位溢出藏在预分配足够大的输出后面。

作者GB300给paged约21.82亿元素从illegal access到零错误；dense Q=K65536（含stride padding约43.12亿元素）原有数十万错值，修后窗口内及窗口外均对齐参考。dense容差atol/rtol .1、paged按row max .01，不是FP32逐位证明。建议大prefill/大batch长context配置覆盖真实stride及边界range。

### FlashInfer KDA：融合Q/K归一化与ND raw gate进一步补齐

**flashinfer-ai/flashinfer；本轮发现，原始合入时间及版本：** [#8277](https://github.com/flashinfer-ai/flashinfer/pull/8277)（2026-11-03T08:02:43Z，`7dd8b5446b`，默认分支已见）；[#8259](https://github.com/flashinfer-ai/flashinfer/pull/8259)（2026-11-03T09:38:41Z，`8196950eec`，默认分支已见）；[#8550](https://github.com/flashinfer-ai/flashinfer/pull/8550)（2026-11-03T23:02:53Z，`3f0bf6d576`，默认分支已见）

确定性单rank及KCP可显式 `use_qk_l2norm_in_kernel=True` 接收原始Q/K，forward首reader归一化、不保存normalized Q/K，saved set仅增加FP32 rstd（单rank为8×T×H、KCP为每rank 8×T_r×H bytes），backward重建。flag-off不变，非确定性入口仍拒该归一化选项，KCP的`local_arena="lean"`也拒绝；不能把raw gate支持与Q/K norm支持混为一谈。

ND单rank/KCP本轮接上 `use_gate_in_kernel=True` 的raw gate、A_log/dt_bias梯度；KCP返回的参数梯度只是本rank partial，须调用显式跨rank归约，提供all-gather后按rank顺序求和的逐位确定性helper。raw gate的KCP亦只走replay arena、拒绝lean，不能假定所有checkpoint handoff模式照旧生效。作者BF16、16K至约261K tokens、多文档/H1–64，B200/GB200/B300/GB300/R200：对det包多数接近，去掉host-bound形状后单rank几何均值仅1.0003–1.0019；对FLA的约4×是另一基线且不含KCP。归一化forward更快也伴若干backward退步。建议复现保存显存、真实门控/归一化配方和整step；不把算子峰值外推训练加速。

### FlashInfer确定性KCP卷积：非cooperative启动用于已有并发kernel的场景

**flashinfer-ai/flashinfer；本轮发现，原始合入时间及版本：** [#8459](https://github.com/flashinfer-ai/flashinfer/pull/8459)（2026-11-03T09:41:21Z，`46924b97be`，默认分支已见）

显式 `cooperative=False` 让确定性KCP causal conv1d+SiLU backward用少1个SM的普通launch，再固定序FP32归约，避免cooperative grid等待其他stream resident kernel。默认仍cooperative；输出/dx与默认逐位相同，dweight保持重复运行确定性但跨两模式不保证逐位相等。

作者CUDA13.4/PyTorch2.14.1，C8192/10240、T16384/32768、多文档，GB200/B200/GB300；有1个2/4ms侧kernel时对旧cooperative报5–18×，空载最多约1.1%变慢；这衡量等待消除，不是算术吞吐骤增，SM107未给计时。建议有多stream重叠的训练复现并发负载，空载不应按峰值收益采用。

### FlashInfer NVFP4通信：先修wire/相位，再看TP4融合收益

**flashinfer-ai/flashinfer；本轮发现，原始合入时间及版本：** [#8325](https://github.com/flashinfer-ai/flashinfer/pull/8325)（2026-11-03T01:42:26Z，`0a0ee73d55`，仅核实合入 main）；[#8395](https://github.com/flashinfer-ai/flashinfer/pull/8395)（2026-11-03T04:26:08Z，`d1ff7c4088`，仅核实合入 main）；[#8419](https://github.com/flashinfer-ai/flashinfer/pull/8419)（2026-11-03T19:28:25Z，`1d2e5f9367`，默认分支已见）

预量化NVFP4的E2M1负零会撞轮询wire编码，现发送前canonicalize；E4M3 scale以无符号发布，避免NaN读成0xFF sentinel；call index在三buffer轮转内wrap，PDL wait后每thread发async-proxy fence。作者GB300 TP2/4对旧Cake速度近持平，价值在防hang/错误。

新增TP4 two-shot all-reduce+residual+RMSNorm，仅BF16输入/输出并放开非4倍数行；接口仍限hidden6144、单节点CUDA IPC peer access、各卡SM数一致的SM100/103/107配置。GB300、H6144、2049–8192行、最慢rank中位数：背靠背对已有Cake AR+norm为1.14–1.31×，以两侧均六次调用的CUDA graph对NCCL+norm为2.62–2.95×；**wire采用NVFP4有损通信，不能当无损BF16集体通信替代或整训收益。** SM107也补上既有MoE单边NVLink Lamport transport编译支持，64rank限制不变，未见独立性能证明。建议先做低幅/负零/NaN/长轮转数值与容错回归。

### 低精度Ulysses：FlashInfer与AITER交付不同通信合同

**flashinfer-ai/flashinfer；本轮发现，原始合入时间及版本：** [#4924](https://github.com/flashinfer-ai/flashinfer/pull/4924)（2026-11-03T02:13:02Z，`252965dc2e`，仅核实合入 main）

`UlyssesCommunicator.prepare_qkv/scatter_qkv` 把FP16/BF16序列分片量化为INT8 Q/K、FP8 E4M3 V；量化组沿全局序列网格对齐，准备期collective核配置并分配复用存储，执行拥有一次统计AllGather和一次合包AllToAll，attention仍由调用者执行。新容量是完整operand字节数 `max_bytes`，含scale/alignment；不是总显存上限，q.nbytes单份不足，out复用需串行并保持生命期。

作者8×H20-3e、SM90、D64/128、P8BF16/P4FP16/P2零V等真实NCCL+stock Sage2的6组验收；SM89/120仅编译/静态覆盖，不是硬件通过，最后完整上游GPU CI未给成功回执。建议长序列通信瓶颈用户考虑复现，先固定consumer精度与边界组，而不是把payload压缩率当端到端倍率。AITER的独立输入hop见AMD条目。

### TensorRT-LLM BF16 context-first生命周期保护：受限显式启用

**NVIDIA/TensorRT-LLM；本轮发现，原始合入时间及版本：** [#21272](https://github.com/NVIDIA/TensorRT-LLM/pull/21272)（2026-11-03T06:15:07Z，`db98553909`，默认分支已见）；[#20709](https://github.com/NVIDIA/TensorRT-LLM/pull/20709)（2026-11-03T06:07:06Z，`fee6cfd872`，默认分支已见）；[#20708](https://github.com/NVIDIA/TensorRT-LLM/pull/20708)（2026-11-03T09:53:54Z，`da980cce47`，默认分支已见）

新增双端显式 `TRTLLM_ENABLE_KV_TRANSFER_LIFECYCLE_ENFORCEMENT=1`，并要求coordinator/worker `TRTLLM_DISAGG_NO_RETRY=1`。仅KVCacheManagerV2、full-context BF16 MHA/GQA、对称TP1/2、PP/CP/DP1、无attention-DP、Python异步transceiver/C++NIXL-UCX直传、有限超时MPI。session持有源/目的K/V与操作roots直到安全settlement；不确定的取消/超时经不可重置grace进入fail-stop，可终止整个MPI launcher。

**未覆盖bounce、staging、layerwise、retry/reroute或一般in-flight取消。** 作者104 CPU与7 H100、4项B200生命周期用例；故障注入在native DONE之后遮完成，证明软件保留/containment，不证明在途DMA fencing或安全替换。因此不能宣布昨日迟到scatter缺口已普遍修复。

另按每view真实head数处理异构KV层、虚拟owner/混合精度及P/D TP切片，注册前拒绝不兼容；两端须同build。超时新增deadline owner与非阻塞progress快照只改善诊断，预算与取消政策不变。建议先在受支持部署做取消、页复用及rank失效验收。

### TensorRT-LLM Helix owner-only持久化进入main，聚合支线仍分开

**NVIDIA/TensorRT-LLM；本轮发现，原始合入时间及版本：** [#21245](https://github.com/NVIDIA/TensorRT-LLM/pull/21245)（2026-11-03T06:03:30Z，`26ffe4bba6`，默认分支已见）；[#21526](https://github.com/NVIDIA/TensorRT-LLM/pull/21526)（2026-11-03T16:54:01Z，`4aaa5dcd2f`，仅核实合入 feat/helix-agg）

main为fresh dense MLA prefill按global token→rank local cache slot映射，仅owner写persistent KV/scale，负slot不写持久池但仍生成attention临时Q/K；一次preprocess/append后复用。fresh且无prefix/chunked也初始化indptr，离开context清stale slots，unsupported cached context在mutate前拒绝。**不单独启用aggregate serving，也不交付历史KV gather。** dense FP4 KV attention在写前拒，不能从NVFP4 writer测试推得支持。

#21526只是将相同审查修正带到feat/helix-agg；该头本轮仅CPU隔离测试，native rebuild/GPU待验。建议Helix用户分别锁主干及支线基线，验证owner、chunk/reuse及dtype门控，不把两条合并成正式完整能力。

### TensorRT-LLM DFlash统一target/draft KV：避免P/D漏传prompt状态

**NVIDIA/TensorRT-LLM；本轮发现，原始合入时间及版本：** [#20486](https://github.com/NVIDIA/TensorRT-LLM/pull/20486)（2026-11-03T22:44:42Z，`d07468ba9e`，默认分支已见）

符合layer owner、dtype、页几何等门控的target/draft可放同一V2 KV manager，draft通过真实allocation metadata及block offset寻址；peer注册显式比较draft层数，避免只传target而接收端draft prompt KV未初始化。不支持的统一布局报错，分离manager保留private arena fallback。

作者报告24组聚合/分离配置接受长度均改善；同TP且无attention-DP时AL相同，其他设置生成文本可不同，不能拿AL相近证明输出等价。最终diff含分配/peer/offset测试，本轮未取得完整新头GPU验收回执或可复核的公开性能表。建议DFlash P/D优先复现两端同build、prefix复用、混合精度/TP与失败清理。

### TensorRT-LLM DSpark按置信度选验证预算，成本表必须匹配部署

**NVIDIA/TensorRT-LLM；本轮发现，原始合入时间及版本：** [#20078](https://github.com/NVIDIA/TensorRT-LLM/pull/20078)（2026-11-03T23:49:09Z，`63ef35ea7e`，默认分支已见）

实验性默认关闭：confidence head对draft位置评分，在纯decode step按实测整step成本表选较少验证行；draft仍生成完整block，target验证/接受规则不改。要求embedded drafter、padded graphs、PP1/CP1；LoRA、guided decode、动态EPLB、部分logits处理等组合拒绝，含context的混合step仍全验证。成本表核GPU型号、world/TP/EP/attention-DP及捕获batch，版本不同只warn；profiling会为各K加载模型并采样，需计准备成本。

作者8×B300、DeepSeek-V4-Pro、DEP8/FP8 KV、10240×768 token refill负载、同节点各两轮：BS1024相对最佳静态K3 46248→50204tok/s（+8.6%），BS512相对K5基本持平；不是任意batch通用提升。建议稳定线上负载考虑复现，重新生成成本表并验证接受分布及尾延迟。

### TensorRT-LLM复用策略与Lamport单位：新策略和已知数据损坏分别验收

**NVIDIA/TensorRT-LLM；本轮发现，原始合入时间及版本：** [#21432](https://github.com/NVIDIA/TensorRT-LLM/pull/21432)（2026-11-03T18:43:07Z，`ad998a67e1`，默认分支已见）；[#21283](https://github.com/NVIDIA/TensorRT-LLM/pull/21283)（2026-11-03T22:29:10Z，`b189a1525a`，默认分支已见）；[#21199](https://github.com/NVIDIA/TensorRT-LLM/pull/21199)（2026-11-03T09:48:57Z，`c26c166b8e`，默认分支已见）

KV v2新增仅滑窗且带sink层可用的SINK_REQUIRED_ONLY：sink决定prefix匹配与claim，window不缩短匹配；恢复时window完整才选，否则新分配。为后续mixed-precision KV做准备，PR所列GLM5.2/8GB300、并发128命中80.4→89.4%、p90TTFT333→72秒**含后续实现，不能归为本PR独立交付收益**。

Lamport one-shot/MiniMax/MoE共享workspace的clear size统一为byte、64位计算、reader向上对齐16B，防BF16/FP32交替复用时hang/NaN/错输出；two-shot与MNNVL不属此改动。作者4GB300对照及warmup/38chat通过。Encoder-decoder还在admission拒绝超单次encoder预算请求，避免坏请求结束executor；多个各自合法的请求合批仍可能超encoder_max_num_tokens，另留修复。建议分别做prefix/精度切换与异常请求隔离验收。

### TensorRT-LLM MiniMax DSpark新配方目前仅特性支线

**NVIDIA/TensorRT-LLM；本轮发现，原始合入时间及版本：** [#21475](https://github.com/NVIDIA/TensorRT-LLM/pull/21475)（2026-11-03T22:29:19Z，`90b6552caf`，仅核实合入 feat/minimax-m3.1-flash）

feat/minimax-m3.1-flash接入block-verified ancestral drafting、prefix-hit drafter context与page-aligned claim/host context allocation，附GB300 TP2/4 agentic配方。原始303单测和900秒自然接受AgentX是作者支线结果，不能写入main或正式release。

作者明示 `no_topk` 加block验证在7轮中5轮stall，默认采样6轮未stall、standard rejection+no_topk两轮未stall；配方保留默认模式。建议继续观察上游集成和该组合缺陷；不因作者TP2某组83.4K总tok/s/GPU就当全部输出生成速率或跨资源性能排名。

### Dynamo sidecar：引擎丢失后撤注册、drain并退出等待监督器重启

**ai-dynamo/dynamo；本轮发现，原始合入时间及版本：** [#17727](https://github.com/ai-dynamo/dynamo/pull/17727)（2026-11-03T22:21:44Z，`e4ad12ba9f`，默认分支已见）；[#17728](https://github.com/ai-dynamo/dynamo/pull/17728)（2026-11-03T22:22:32Z，`bd6aa892c3`，默认分支已见）；[#17972](https://github.com/ai-dynamo/dynamo/pull/17972)（2026-11-03T23:55:01Z，`2a0cc9a393`，默认分支已见）；[#17973](https://github.com/ai-dynamo/dynamo/pull/17973)（2026-11-03T23:58:07Z，`37144cc68d`，默认分支已见）

共用LLMEngine.engine_lost hook触发原有shutdown路径：撤discovery、grace、prefill drain、cleanup，返回错误使sidecar退出1。vLLM监听整体health空服务名，TRT监听liveness，不把sleep/pause/readiness当死亡；SGLang只以WatchEngineState流结束/错误/未打开为丢失，healthy=false与is_pause不触发。

**这是sidecar重启与重新注册，不自动恢复模型引擎或完成中的请求。** TRT旧镜像无health服务会warn并保持原行为；SIGTERM drain已开始则不改变原结果。SGLang直到自身gRPC停止才被识别，drain期间仍可能注册。建议部署者验收实际镜像协议、Kubernetes restartPolicy、shutdown grace与服务发现退出，勿当通用自愈保证。

### Dynamo KVCR0.2部署例补齐监督启动，故障替换仍有控制器缺口

**ai-dynamo/dynamo；本轮发现，原始合入时间及版本：** [#17290](https://github.com/ai-dynamo/dynamo/pull/17290)（2026-11-03T21:10:37Z，`22e65313a2`，默认分支已见）

同Pod例仅Guard挂pool目录，vLLM通过Unix socket收FD；不可变ConfigMap保存脚本，pool大小与HTTP/Guard/engine启动预算分开，readiness只读。需配套vLLM adapter的timeout forwarding，单覆盖KVCR wheel不足以更新native或引擎。

作者两节点B200在engine停机时Guard经RDMA服务451blocks/827.6MB/7216tokens，再交回重启engine；sidecar未重启。该验证是默认memory-backed，非HugeTLB/THP资格验证；Guard/state-agent失败仍需协调Pod或worker group替换，本PR不交付该控制器。建议KV持久化恢复用户考虑复现FD寿命、失败退出与混合版本拒绝。

### Dynamo成功decode终包可先发，仍等待两条腿结束做清理

**ai-dynamo/dynamo；本轮发现，原始合入时间及版本：** [#17484](https://github.com/ai-dynamo/dynamo/pull/17484)（2026-11-03T08:44:25Z，`2ec0b1112d`，默认分支已见）

SGLang bootstrap式P/D将成功decode terminal及最后tokens直接交付，不再等prefill响应；成功terminal之后的prefill失败记录日志，不反转已完成答案。实现仍读完decode、再等prefill结束后关闭stream，因此request寿命、取消与router记账并未缩短。

作者agentic场景p90用户输出速率约360→470tok/s，硬件/模型/输入分布披露不足，不能当可比吞吐基准。建议短答案多轮服务复现末包延迟、迟到prefill失败和cleanup；不要只测首token或假定prefill腿可以立即取消。

### Dynamo多模态资源：publish前清理与接收预校验不等于DMA完成保护

**ai-dynamo/dynamo；本轮发现，原始合入时间及版本：** [#15500](https://github.com/ai-dynamo/dynamo/pull/15500)（2026-11-03T01:19:00Z，`319ad3b906`，默认分支已见）；[#17935](https://github.com/ai-dynamo/dynamo/pull/17935)（2026-11-03T20:16:03Z，`607f8677a9`，默认分支已见）

sender在序列化/分配/注册/metadata构造/取消未能成功返回前持有cleanup责任，逐项失败回收，保留原始错误；未发布NIXL注册可直接释放。成功发布后的完成等待政策不变。作者ARM64/vLLM0.27.1真实POSIX roundtrip与隔离module41项通过，未验native NIXL/RDMA/GPU。

receiver在借pool buffer前检查READ kind、单descriptor、指针、size、notification，避免失败借用丢buffer。作者明确当前生产receiver pool max_items=0，无已知部署受影响；复现是CPU单buffer＋mock agent，不写成已发生线上泄漏。建议多模态接入者按资源发布边界验收，不把这些host修复当在途DMA安全回收证明。

### Dynamo KV hint可取单源continuation，仍不拼成多源假前缀

**ai-dynamo/dynamo；本轮发现，原始合入时间及版本：** [#17657](https://github.com/ai-dynamo/dynamo/pull/17657)（2026-11-03T03:39:06Z，`7e44a1dfea`，默认分支已见）

目标GPU/host已有blocks1–10时，可选择只持11–15的单源延伸，hint hash仍根对齐到15；从原local walk frontier接续、限制source活动区间，保留拓扑排除与保守prefill成本。一个hint只命名一个source，fetch进度不能自动作另一source或detached L2的已有前缀，Disk/External仍仅供评分。

建议缓存分布不完整的多worker服务考虑复现命中、排队后刷新和失效源；代码/单测支持机制，未有本轮可比较端到端性能证据。

### Dynamo批处理Planner：同快照决策、Redis lease与writer fencing接入native tick

**ai-dynamo/dynamo；本轮发现，原始合入时间及版本：** [#14196](https://github.com/ai-dynamo/dynamo/pull/14196)（2026-11-03T18:50:59Z，`d554c60a1f`，默认分支已见）；[#14197](https://github.com/ai-dynamo/dynamo/pull/14197)（2026-11-03T20:28:32Z，`740a7078ad`，默认分支已见）；[#14200](https://github.com/ai-dynamo/dynamo/pull/14200)（2026-11-03T23:10:35Z，`1564a73f46`，默认分支已见）

生产形状的Gateway/在线RPS与并发/Async backlog观测并发采集到同一快照，区分lazy未出现的metric与不可用：未知active-request不按零做idle downscale，durable job状态仍权威。Redis按server time原子writer epoch＋sequenced lease，拒stale writer、不同payload重试与过期decision，支持丢ACK后的同payload重试；启动先零速lease再拿Kubernetes writer fence。

native tick一次policy求值，batch floor/idle target与plugins合并后仍受GPU/功率budget，先apply drain lease再变副本；actuation错误视结果未知，不自动重放/scale。默认opt-in；这是控制面接口与故障排序，不是硬件吞吐结果。建议在线/离线混部用户深入读lease、缺观测及split-brain测试，再在真实Redis/Gateway版本压测。

### ATOM MXFP4与SWA：真实padding/allocator状态保护

**ROCm/ATOM；本轮发现，原始合入时间及版本：** [#2691](https://github.com/ROCm/ATOM/pull/2691)（2026-11-03T05:33:52Z，`8cb9eaa6fc`，默认分支已见）；[#2895](https://github.com/ROCm/ATOM/pull/2895)（2026-11-03T02:45:04Z，`b3a3ea0af3`，默认分支已见）

MXFP4 preshuffle ASM不mask K尾，Kimi K3 TP8 dense MLP局部K4224等几何会读未写scale/越行。现在成对column/row parallel linear分别补零输出/输入，按checkpoint实际量化方案核兼容；e8m0 shuffle对齐256、gfx1250 32×4为128，尾mask Triton例外。量化源跨格式需补scale的在线路径拒绝，module weight_loader同步也须遵守padding。

DSV4 MTP draft的SWA环改按首KV block对应allocator slot，graph每步随live position刷新indptr/dest，避免capture dummy与后续draft步覆盖。作者DSV4-Pro TP8/MTP3、并发8、5shot GSM8K配对401→366秒、接受44.6→64.1%；末提交draft_extend图路径仍未捕获、只4prompt probe，不把全run当最终所有路径通过。建议低位在线重载及投机用户优先回归真实allocator/prefix/graph。

### ATOM adaptive prefill：替代手调cadence，独立收益与融合叠加分清

**ROCm/ATOM；本轮发现，原始合入时间及版本：** [#2897](https://github.com/ROCm/ATOM/pull/2897)（2026-11-03T14:40:23Z，`b36791f027`，默认分支已见）

默认仍interval；显式adaptive按实测decode/prefill成本拟合rent-or-buy admission，设置冷启动/时间/KV水位上界，DP各rank复用all-reduce共享估计，保持一致决策。早期直接用decode拟合截距曾接近零且退步，最终改有界均值估计。

作者4×MI455X、DSV4.1-Flash DEP4+DSpark、Agentic并发512、每轮10分钟/各两次生成tok/s/GPU：手调interval24 11085，adaptive单独11090（基本持平）；stage1 fusion单独11322，fusion+adaptive11586。总+4.5%不能全归adaptive。建议负载/算子持续变化的服务复现p99TTFT与动态稳态，价值首先是减少调参。

### ATOM Engram UVA：锁页与sysctl联合缓解主机回收诱发GPU停顿

**ROCm/ATOM；本轮发现，原始合入时间及版本：** [#2898](https://github.com/ROCm/ATOM/pull/2898)（2026-11-03T15:45:46Z，`63e64d8dac`，默认分支已见）

原地注册checkpoint page-cache期间best-effort mlock，lease最后引用释放时先unlock再unregister；失败只warn并继续注册。mlock2(MLOCK_ONFAULT)加现有线程池分块fault-in。作者DSV4.1-Flash DEP4、MI455X、183GiB原地注册、c512小时压测，旧版14.6分钟开始stall/31.5分钟hang；开启mlock并同时关闭两项compaction sysctl后两小时轮次无stall/hang。

**没有分离mlock与sysctl贡献。** 主机共享cache锁定后内存压力可能转为OOM，copy路径原已锁页；此处只记录源码文档中的实际部署取舍，不执行任何宿主机设置。建议有evicted_ms与内存压力证据的部署先小规模复现，不能对无该瓶颈系统承诺吞吐收益。

### ATOM精确GDN中途快照目前只在gfx1250特性支线

**ROCm/ATOM；本轮发现，原始合入时间及版本：** [#2934](https://github.com/ROCm/ATOM/pull/2934)（2026-11-03T14:27:59Z，`6a54ea527a`，仅核实合入 alizaidy/gfx1250-qwen-support）

可选MIDSTEP将FP32递归寄存器状态直接写每chunk slot，避免从BF16 h恢复的精度损失；checkpoint targets用固定地址表参加prefill token graph，只超表容量回eager。默认仍关闭。

作者Qwen3.5-35B-A3B MXFP4、TP4/MTP4、FP8 KV、Agentic c64、10分钟：TTFT245→128ms、interactivity390.4→443.7、输出8640→8967tok/s；是alizaidy/gfx1250-qwen-support支线，不是main交付，且别的路径/状态布局需单独验证。建议继续观察集成，并核FP32 snapshot与切分forward的逐位一致性。

### AITER Ulysses融合输入hop与gfx1250低延迟all-gather

**ROCm/aiter；本轮发现，原始合入时间及版本：** [#5815](https://github.com/ROCm/aiter/pull/5815)（2026-11-03T09:28:18Z，`d4d80a02f4`，默认分支已见）；[#7265](https://github.com/ROCm/aiter/pull/7265)（2026-11-03T05:39:44Z，`5139375658`，默认分支已见）

AttentionA2AIntraNodeOp将BF16 [1,S_local,H,128] Q/K/V写入MoRI对称peer内存，可同时Q/K Hadamard和独立codec量化，输出MHA V4布局；只涵盖输入hop，不含RMSNorm/RoPE/反向all-to-all。packed length与MX dense consumer有32/128倍数限制，caller选择consumer支持的codec配对；跨stream consumer须event join后再复用双buffer。最终CI脚本已给测试入口，但作者说明MoRI依赖覆盖需另核，不把脚本存在当设备通过。

gfx1250 LL all-gather≤4MiB/rank改push peer scratch，与LL all-reduce共epoch/轮转，要求整8byte packet，未对齐指针另用byte-wise kernel。作者MI455X TP4、MiniMax-M3 MXFP4/EAGLE3、3×4096prompt、250步中位两轮：decode5.226→5.118ms；单logits gather61.1→17µs不能外推整step3.6×。建议TP通信瓶颈用户考虑复现多collective交错及fallback。

### AITER状态/页布局：BF16 GDN有精度取舍，FP4 MQA读取按真实stride

**ROCm/aiter；本轮发现，原始合入时间及版本：** [#6884](https://github.com/ROCm/aiter/pull/6884)（2026-11-03T15:55:59Z，`720b36ce0f`，默认分支已见）；[#5518](https://github.com/ROCm/aiter/pull/5518)（2026-11-04T00:33:51Z，`3a14384212`，默认分支已见）

gfx950 fused GDN decode新增BF16 recurrent state，读入FP32更新后回存原dtype，减少状态读写字节；BF16只走ratio-generic路径，旧FP32 pair kernel不变。作者MI355X、16 K heads/ratio3/D128、flat、B1–256：14.84→14.02µs（B1）、151.96→87.02（B128）。这是改状态精度的局部对照，缺长轨迹整模精度验证。

gfx950 FP4 MQA logits传真实K/scale page stride，并将physical page偏移64位rebase，适配vLLM混合block中跨层/类型区域；不能把page payload大小当block stride，也不重排整个pool。建议混合KV/大pool用户核writer preshuffle与reader一致，状态压缩用户另验误差累积。

### AITER MXFP8 Attention与hipBLASLt grouped C API：新增接口不等于全模式可用

**ROCm/aiter；本轮发现，原始合入时间及版本：** [#5578](https://github.com/ROCm/aiter/pull/5578)（2026-11-03T11:56:25Z，`bc12ddf440`，默认分支已见）

新增gfx950（CDNA4）的独立Triton MXFP8 FA2 forward/backward函数，32元素E8M0 block scale，架构运行时断言；支持MHA/GQA、causal。独立wrapper只接bshd/bhsd、拒varlen thd，forward直接拒bias/dropout，backward拒ALiBi。更关键的是，通用`flash_attn_func`接MX scales的路径目前**仅forward**：requires_grad直接报错，输出detach，并拒bias/dropout/sink/sliding-window及Gluon后端，不能把独立backward函数写成该入口已接autograd训练。最终测试以已去量化Q/K/V为reference，forward atol/rtol .2、gradient .35；它们不证明量化器梯度或整训收敛。原PR说明把gfx950括注MI300X，测试说明实际为MI350X；非power-of-two head在kernel文档仍列为未支持，不由padding代码推断普遍可用。

建议低精度attention开发者读实现并用目标精度复测梯度/长序列，未见足够端到端收益可推荐直接替换。下列hipBLASLt是另一个仓库/合同。

### hipBLASLt grouped GEMM C API接上MX scale，库内kernel覆盖仍有限

**ROCm/rocm-libraries；本轮发现，原始合入时间及版本：** [#13581](https://github.com/ROCm/rocm-libraries/pull/13581)（2026-11-03T04:52:52Z，`da1b11a6e8`，默认分支已见）

在已有grouped C API上接入MXFP8/MXFP4及block16/32 UE8M0/UE4M3/UE5M3描述符，A/B scale pointer是每group tensor的device数组，K按模式block倍数校验；修228byte MX record的pointer/stride和codegen offset，heuristic拒scale布局不匹配algo。

当前gfx950加入的是pre-swizzled TN grouped逻辑；unswizzled、block16及gfx1250 grouped logic仍有skip/后续项，C++ GroupedGemm device-user-args非MX record也未改。作者MI355X/ROCm7.2.1重生kernel后给16pass/8skip与smoke4pass，不能把接受全部enum写成每种已有可用kernel。建议MoE GEMM接入者核具体scale格式/启发式与空group/K0。

### Triton：TDM到达计数、dot类型与NVFP4资源预算

**triton-lang/triton；本轮发现，原始合入时间及版本：** [#12522](https://github.com/triton-lang/triton/pull/12522)（2026-11-03T00:59:17Z，`1fad2df060`，默认分支已见）；[#12442](https://github.com/triton-lang/triton/pull/12442)（2026-11-03T15:45:11Z，`3412130d63`，默认分支已见）；[#12612](https://github.com/triton-lang/triton/pull/12612)（2026-11-03T19:33:04Z，`f7f4df5229`，默认分支已见）；[#12613](https://github.com/triton-lang/triton/pull/12613)（2026-11-03T21:05:29Z，`90e50ecd9c`，默认分支已见）

AMD CDNA5的barrier-bearing TDM每个enabled逻辑操作每warp恰贡献一次arrival，包括无数据warp；拆硬件指令不重复arrival，防漏等/过早完成。dot前端按operand实际结果dtype检查accumulator，避免INT8+FP32 acc等组合在后端报无关错；blocked布局剩余CTA按CTAOrder最后轴wrap，修排序错误。

Blackwell FP8×NVFP4 fallback补计decoded BF16 tile及转换scratch，减少超shared-memory配置；GB300 M2048/K1024/N8192旧persistent236840、regular266256bytes均超过232448限额。作者若干warmed graph矩阵对手动block_n128/stages2，局部延迟改善，但不是attention/整模指标。建议自定义kernel、TDM同步与显式tile配置用户优先回归；不能将编译接受视作数值正确。

### JAX/Pallas移除Triton backend；XLA图与集合通信边界收紧

**jax-ml/jax；本轮发现，原始合入时间及版本：** [#42259](https://github.com/jax-ml/jax/pull/42259)（2026-11-03T16:12:20Z，`2361d533e7`，默认分支已见）；[#42257](https://github.com/jax-ml/jax/pull/42257)（2026-11-03T13:33:51Z，`a67386b646`，默认分支已见）；[#42258](https://github.com/jax-ml/jax/pull/42258)（2026-11-03T14:14:22Z，`748762f0f3`，默认分支已见）；[#42265](https://github.com/jax-ml/jax/pull/42265)（2026-11-03T22:57:21Z，`39ed35fb56`，默认分支已见）

Pallas Triton lowering/backend、测试与文档从main移除，属实际后端兼容变化，不是Triton项目本身移除GPU支持。现有Pallas/Triton代码应锁版本并评估Mosaic GPU等替代，不假定所有kernel能自动迁移。partial-manual shard_map与Explicit axes中的untiled collective同步修sharding specs，避免扩维/输出类型继承错误。

建议依赖该Pallas后端的训推/算子项目立即核迁移；本轮未执行设备迁移或提供跨后端性能比较。

### XLA：HLO runner核跨进程可执行指纹，command buffer追踪对称内存身份

**openxla/xla；本轮发现，原始合入时间及版本：** [#52188](https://github.com/openxla/xla/pull/52188)（2026-11-03T12:54:04Z，`a4eb10edbb`，默认分支已见）；[#52110](https://github.com/openxla/xla/pull/52110)（2026-11-03T14:31:58Z，`3936953a4f`，默认分支已见）；[#52204](https://github.com/openxla/xla/pull/52204)（2026-11-03T20:40:58Z，`6c34da6bad`，默认分支已见）

FunctionalHloRunner在多进程且有KV store时默认比较optimized HLO及schedule指纹，差异则编译失败并列进程，防独立autotune产生collective次序不同而hang。**不含backend config，不是所有JAX分布式执行入口的普遍新保护。** 可显式关闭，但建议先定位差异。

CommandBufferThunk按每device精确比较collective memory的(key,process-unique ID)，变更则重录，防地址复用骗过graph身份检查；FFI自己分配的对称内存仍不被检测。另保留mosaic_gpu_v2的schedule group annotation，使attention等可与async collective重叠；没有单项端到端测速。建议多host调优与FFI图复用用户深入读边界再测试。

### TorchTPU：动态kernel限eager，verifier目前仅骨架

**google-pytorch/torch_tpu；本轮发现，原始合入时间及版本：** [#6166](https://github.com/google-pytorch/torch_tpu/pull/6166)（2026-11-03T11:42:35Z，`2cf50f8d98`，默认分支已见）；[#6145](https://github.com/google-pytorch/torch_tpu/pull/6145)（2026-11-03T03:25:09Z，`c2cb7bf254`，默认分支已见）；[#6189](https://github.com/google-pytorch/torch_tpu/pull/6189)（2026-11-03T22:07:39Z，`3ef1a809aa`，默认分支已见）

无界动态kernel依赖运行时metadata，torch.compile序列化会丢；CompilationCache每请求policy默认Disallow，仅eager ExecutionTask可Allow，并拒序列化这类可执行物，避免并发eager/compile共享状态误选。Stream/Event同时改为真实类型，可用于isinstance/类型联合，仍为experimental。

新 `torch.tpu.pallas.verifier` 当前只交付入口、finding/report数据模型与基本输入校验；Jaxpr静态分析、CP-SAT并发检查、CPU interpret poisoning等都列为后续。**不能写成已经能无需TPU验证kernel正确性。** 建议编译/动态shape集成先回归，verifier继续观察后续passes。

### Helion调优可信度：异步执行错误不能被计成最快，OOM后换worker

**pytorch/helion；本轮发现，原始合入时间及版本：** [#4184](https://github.com/pytorch/helion/pull/4184)（2026-11-03T22:48:38Z，`16fc1cf67e`，默认分支已见）；[#4233](https://github.com/pytorch/helion/pull/4233)（2026-11-03T22:41:51Z，`35268fede5`，默认分支已见）；[#4246](https://github.com/pytorch/helion/pull/4246)（2026-11-03T22:54:19Z，`15ae0fd4b3`，默认分支已见）

Pallas基准对返回的TorchTPU tensor显式等待device结果，任一run附带错误则该配置时间记Inf，避免只调torch.accelerator.synchronize未看到输出错误。OOM后杀benchmark worker并补新进程，释放缓存/driver局部内存高水位/已载module；不让后续候选连续失败后错误选默认慢配置。

baseline比较在parent留下的空闲reserved GPU blocks也在搜索前empty_cache，使另进程worker可用。作者H100 nightly有41–170连续失败及默认120ms胜过真正15ms候选的定位，不是这套修复给LLM8×加速。建议自动调kernel项目考虑采用相同错误隔离/资源寿命验证，保留正确性与性能选择分离。

### PyTorch主干：NCCL2异步barrier、fallback输出寿命与masked softmax

**pytorch/pytorch；本轮发现，原始合入时间及版本：** [#202783](https://github.com/pytorch/pytorch/commit/d6db56f599beb3a4cd0a0430849af01021880d42)（2026-11-04T00:56:28Z，`d6db56f599`，主干提交已见；PR merged_at 为空，按真实提交确认）；[#202963](https://github.com/pytorch/pytorch/commit/b60d686fb5c5438283066fa9f99c14b7ab75ffe3)（2026-11-03T02:34:41Z，`b60d686fb5`，主干提交已见；PR merged_at 为空，按真实提交确认）；[#203045](https://github.com/pytorch/pytorch/commit/113a303a6ca72e868df2425a7a96bc980f7572b3)（2026-11-03T21:09:23Z，`113a303a6c`，主干提交已见；PR merged_at 为空，按真实提交确认）

nccl2 barrier传入asyncOp，避免异步barrier的block_current_stream已排spin kernel、getFuture却先同步该stream，永远无法注册释放callback。AOTI/cpp_wrapper则在未用fallback输出的unbacked symbol提取之后立即reset RAII handle，不让每层SDPA logsumexp活到整model返回；作者H100/timm B128的16/92MB减少只是视觉例证，未有LLM显存百分比。

masked softmax backward对mask读补row/lane边界；原匹配梯度为0故输出数值未变，但device ASan能报越界，不能把它描述成已观察到梯度错误。建议多stream collectives、AOT推理及sanitizer用户优先回归实际commit；PR merged_at为空，以上均通过真实main commit及最终diff确认。

### PyTorch Inductor八项本轮重新落地：先区分昨日回滚状态

**pytorch/pytorch；本轮发现，原始合入时间及版本：** [#200459](https://github.com/pytorch/pytorch/commit/dc5dbbe1b490bf59876d3ba4ba8fd5fe8601f1f4)（2026-11-03T19:49:24Z，`dc5dbbe1b4`，主干提交已见；PR merged_at 为空，按真实提交确认）；[#200460](https://github.com/pytorch/pytorch/commit/3aadcf52346620f4239f3b5abd1019789fcc7850)（2026-11-03T19:49:24Z，`3aadcf5234`，主干提交已见；PR merged_at 为空，按真实提交确认）；[#200461](https://github.com/pytorch/pytorch/commit/9b1ff1332ef196a7445047824846d9db0372794f)（2026-11-03T19:49:24Z，`9b1ff1332e`，主干提交已见；PR merged_at 为空，按真实提交确认）；[#200462](https://github.com/pytorch/pytorch/commit/d102913fc9a05ff090d84f8369c9d1500657b215)（2026-11-03T19:49:24Z，`d102913fc9`，主干提交已见；PR merged_at 为空，按真实提交确认）；[#200463](https://github.com/pytorch/pytorch/commit/2a79b2c5315a251dd9b9971c7d193cc648ebd8cb)（2026-11-03T19:49:24Z，`2a79b2c531`，主干提交已见；PR merged_at 为空，按真实提交确认）；[#200464](https://github.com/pytorch/pytorch/commit/6c9a3b578d90819359d1b5ae851950d2aed098ef)（2026-11-03T19:49:24Z，`6c9a3b578d`，主干提交已见；PR merged_at 为空，按真实提交确认）；[#200465](https://github.com/pytorch/pytorch/commit/7328a77751ef5ece5ca756fbc78e8c572f0d7d69)（2026-11-03T19:49:24Z，`7328a77751`，主干提交已见；PR merged_at 为空，按真实提交确认）；[#200466](https://github.com/pytorch/pytorch/commit/90a8fe20438e91c1547e57490c04525c91d10311)（2026-11-03T19:49:24Z，`90a8fe2043`，主干提交已见；PR merged_at 为空，按真实提交确认）

昨日同PR栈被撤回，本轮main有新的八个实际commit，是**重新落地**，不是重复宣称昨日净新增。包括复用已载static launcher；CUDA graph调优在实际current stream捕获并预热lazy候选；computed cat copy可融producer；非重叠仿射写证明；FP16 grouped reduction保FP32累加类型；packed32bit抽取16bit lanes；nested与persistent fixed-lane epilogue转发。

cat-copy融合旗标默认关，因为转置目的布局benchmark有退步；fixed-lane只在可证明单位stride、静态合法group、row-preserving且backend heuristic允许时生效。建议低位量化pack、norm/reduction图读最终代码并与现有baseline测，不把该栈当所有torch.compile工作负载通用加速。

### PyTorch ROCm：UALink fabric rendezvous与device-side集体通信

**pytorch/pytorch；本轮发现，原始合入时间及版本：** [#202830](https://github.com/pytorch/pytorch/commit/defd9f1f0f7766188d8bc555eaea558efa82171f)（2026-11-03T20:16:22Z，`defd9f1f0f`，主干提交已见；PR merged_at 为空，按真实提交确认）；[#202454](https://github.com/pytorch/pytorch/commit/fa6060d66f9d1a3f10701b163dca856c71f0651c)（2026-11-03T15:00:21Z，`fa6060d66f`，主干提交已见；PR merged_at 为空，按真实提交确认）；[#202475](https://github.com/pytorch/pytorch/commit/9ddaecf6304cce84ebae0a170edbce8a2ac74865)（2026-11-03T19:54:33Z，`9ddaecf630`，主干提交已见；PR merged_at 为空，按真实提交确认）

ROCm10.0+且设备报告fabric handle支持并通过provisioning probe时，symmetric-memory rendezvous可交换fabric handle，跨PID namespace容器不依赖POSIX FD；各rank类型不一致明确失败。expandable segments仍走POSIX，不能说allocator已通用fabric化。

rocSHMEM Triton backend以LLVM wrapper补context/team调用，开启device-side alltoall/broadcast。gfx1250支持列表与测试门控也更新，但native async-TP仍限gfx942/950，作者测试栈叠加其他PR、有未解决失败；不把支持列表变化当全套分布式测试通过。建议跨容器UALink/MoE通信开发者复现部署与句柄生命期。

### PyTorch ROCm重复索引归约：5.6×只属于受门控的局部工作负载

**pytorch/pytorch；本轮发现，原始合入时间及版本：** [#200477](https://github.com/pytorch/pytorch/commit/033b0f6d3b90eaafcd0b7c5fd808650dd6554733)（2026-11-03T07:52:02Z，`033b0f6d3b`，主干提交已见；PR merged_at 为空，按真实提交确认）

gfx942/950的index_add_在M≥100万、inner128/256、连续同rank tensor、alpha1、规定dtype/alignment等条件下，将目标index排序为连续runs、FP32寄存器局部归约后分chunk写，减少热点atomic；index_select backward随之受益，自动微分/compile入口不变，deterministic模式已提前转其他路径，不受该优化。

作者MI350X、D128/256、M1M/4M/7M、BF16/FP16/FP32、16分布288组合中287更快，几何均值5.6×；单热点最高约101×不是典型embedding整训收益，也不涵盖小M、任意stride或NPU。建议有重复高争用index_add热点的训练考虑复现，并测排序workspace、完整step及梯度容差。

### LMCache重要补录：MP描述符漏洞已在0.5.6修复，旧摘要不能代替当前状态

**LMCache/LMCache；本轮新发现 / 补录；原始合入时间及版本：** [#5644](https://github.com/LMCache/LMCache/pull/5644)（2026-10-12T06:22:09Z，`160b6dec46`，已核实合入dev）

**原始修复10-12、发行10-14，补录。** [v0.5.6](https://github.com/LMCache/LMCache/releases/tag/v0.5.6) 的published_at为10-14 22:21:42Z，tag固定`3e11e10812b448142c6cdb177a3058652bd826fa`；本轮回读tag内ipc_wrapper.py，已去掉pickle，使用versioned typed MessagePack和本地wrapper allowlist，ZMQ/gRPC共用安全descriptor，无pickle fallback。旧0.3.9–0.5.5及0.5.6rc1–3在可达MP端口配置下受影响；vLLM进程内模式不打开该MP端口。

[10-20官方说明](https://blog.lmcache.ai/en/2026/10/20/lmcache-v0-5-6-release/)与[当前GHSA](https://github.com/advisories/GHSA-w5f2-9q9c-ff4h)确认patched=0.5.6；GHSA Details仍残留截至10-07“无修复版”的旧句，不能断章据此宣称当日仍零日。descriptor wire不兼容，workers/server需一起升级。该修复只管MP device IPC，不代表engine-driven所有pickle、RPC认证与容器用户均解决；博客与advisory对镜像root/默认绑定的版本归属有差异，本文不代为确定。建议相关部署先核精确镜像、模式及同版本通信；本轮未运行漏洞利用或设备测试。

### LMCache共享ZMQ loop不再被单个背压客户端卡死

**LMCache/LMCache；本轮新发现 / 补录；原始合入时间及版本：** [#6256](https://github.com/LMCache/LMCache/pull/6256)（2026-11-03T08:39:09Z，`719f1d6435`，默认dev分支已见）；[#6236](https://github.com/LMCache/LMCache/pull/6236)（2026-11-02T06:25:21Z，`8a112e197b`，默认dev分支已见）；[#6193](https://github.com/LMCache/LMCache/pull/6193)（2026-10-31T06:50:49Z，`b1d939074a`，默认dev分支已见）

ZMQ client以NOBLOCK发送；socket满时在该client保持顺序队列并poll POLLOUT，其他client继续处理。作者一个down server堆1100请求后另client可应答、错把ZMQ接到gRPC端口可按mq_timeout失败并close，正常noop roundtrip近持平。建议多server/重启部署复现背压与错误端口，价值是故障隔离而非吞吐提升。

另两项为72h补录：gRPC正常shutdown和startup失败显式注销metrics插件，停止非daemon export thread，避免进程长期不退出；覆盖sync/aio/native及CLI。只管该startup边界，HTTP服务随后plugin/coordinator启动错误不在此覆盖。建议升级中验收实际SIGTERM与失败进程退出。

### LMCache SHM context换代先drain；超时后保留映射等待迟到操作

**LMCache/LMCache；本轮新发现 / 补录；原始合入时间及版本：** [#4072](https://github.com/LMCache/LMCache/pull/4072)（2026-11-02T11:49:15Z，`38ebdeac20`，默认dev分支已见）；[#6208](https://github.com/LMCache/LMCache/pull/6208)（2026-11-03T03:22:10Z，`236f3eec32`，默认dev分支已见）

#4072为11-02补录。engine-driven SHM transfer_guard包住prepare→gather/scatter→commit，closing后新调用返回False/miss；context register成功才发布，每调用读取一次context，重注册明确close被替换者。**最终close最多等30秒，超时交给后台线程无限等straggler后再unpin/unmap，不是超时就释放**；永不完成的transfer会保留映射到进程退出。最终diff比PR早期“bounded drain proceeds”描述更严格，已按源码核对。

peer adapter拆共用基类并使close失败可重试：失败adapter从lookup/status隐藏，但保留device range所有权；只有unregister成功才mark closed，避免借用仍在外时误释放。作者CPU线程/隔离module与B200相同failure-set证据，不是硬件故障后自动恢复保证。建议缓存服务重启和P2P借用用户优先验收延迟完成、取消、注册失败及旧context生命周期。

### LMCache命中语义：完整rank/group驻留、物理stride与首chunk分别修正

**LMCache/LMCache；本轮新发现 / 补录；原始合入时间及版本：** [#6119](https://github.com/LMCache/LMCache/pull/6119)（2026-11-02T07:10:52Z，`e781afcf18`，默认dev分支已见）；[#6127](https://github.com/LMCache/LMCache/pull/6127)（2026-10-31T05:20:27Z，`8166f24617`，默认dev分支已见）；[#6228](https://github.com/LMCache/LMCache/pull/6228)（2026-11-03T00:01:03Z，`bc3c6699d4`，默认dev分支已见）；[#6252](https://github.com/LMCache/LMCache/pull/6252)（2026-11-03T20:51:28Z，`2e2ce2fe91`，默认dev分支已见）

前三项为10-31至11-03 00:01的补录。CPU KV event只有所有必需rank/group对象的L1副本都committed才Stored；同对象多copy丢一份不Removed，最后所需copy消失才撤，reservation不充数。仍保留legacy unsalted/full-attention identity限制，不等于新namespace路由栈完成。

层分组把物理block stride加入identity，防同逻辑shape BF16页65536/73856元素stride被误并后错传；CacheBlend也登记prompt第一chunk，便于该文档以后换位置复用，和prefix重叠由connector去重。missing-registration RETRIEVE本轮**删除立即释放该worker lookup锁份额**，保留False本地重算，未消费锁由L1 TTL到期；不是锁泄漏已被立即清空。建议warm/cold等价、copy丢失、blend位置变化与TTL分别回归，无统一性能承诺。

### LMCache HTTP认证：可配置入口与出站key传递，不是默认端到端认证

**LMCache/LMCache；本轮新发现 / 补录；原始合入时间及版本：** [#5767](https://github.com/LMCache/LMCache/pull/5767)（2026-10-31T06:11:03Z，`ac34be47b2`，默认dev分支已见）；[#6174](https://github.com/LMCache/LMCache/pull/6174)（2026-10-31T01:43:44Z，`371108bfdb`，默认dev分支已见）

10-31补录。Frontend共用key middleware，可收Bearer或X-LMCache-API-Key，配置认证时除health/metrics/static等公共路由外要求key；代理剥离调用者两种credential header，再注独立backend key。Coordinator/MP server出站也从配置的key file取第一key，所有key可用于入站轮换，修之前配置文件只入站生效而出站401。

**无key默认仍无认证，controller api_server9000不在此变化，key也不加密传输。** 这是已合入dev的配置能力，不能据此把0.5.6或所有gRPC/ZMQ/P2P接口写成默认安全。建议多节点用户在精确版本验证每跳凭据、轮换和公共端点，采用前核release包含关系。

### LMCache RawBlock稳态checkpoint：消除永不落盘，仍留崩溃窗口

**LMCache/LMCache；本轮新发现 / 补录；原始合入时间及版本：** [#6185](https://github.com/LMCache/LMCache/pull/6185)（2026-11-02T21:02:24Z，`94cab1372f`，默认dev分支已见）

11-02补录。原周期checkpoint要求设备至少空闲100ms，持续store会永远不写metadata；现在首次未持久变更开始计时，到完整checkpoint interval即使I/O忙也写，只snapshot在同锁下已完成写的index。默认60秒，期间会walk index并与前台争I/O，不能当零成本耐久性。

作者SIGKILL/1秒interval、20ms连续store由0/1940恢复至1752/1940，仍丢最后checkpoint之后约一interval数据。slot在snapshot后可能evict/reuse的旧风险未改，须按meta_verify_on_load等配置另验；建议本地持久KV用户做持续负载crash/restart测试，不把“定期落盘”当事务原子保证。

### LMCache Valkey RDMA L2：主干接口已合，发布依赖和真实设备证据仍缺

**LMCache/LMCache；本轮新发现 / 补录；原始合入时间及版本：** [#5556](https://github.com/LMCache/LMCache/pull/5556)（2026-11-03T23:06:14Z，`ca439cad3e`，默认dev分支已见）

最终实现是既有ValkeyL2Adapter的 `transport=rdma` 分支（默认tcp），配ValkeyRdmaWorkerPool；每worker一次注册完整L1 arena，store/load通过Valkey large-object的BLOB.RDMA_SET/GET让服务器搬字节，不把value放RESP。不是PR早期描述的独立公开ValkeyRdmaL2Adapter类；普通GET/SET不能读写该key类型，transport应隔key_prefix。

需带RDMA的valkey-glide-sync（代码最小2.6.0，PR称依赖尚未正式发布）、libfabric≥2.4、关闭lazy L1；worker数×L1大小受注册内存上限，arena外buffer不回退TCP，request_timeout不界定RDMA完成。43测试以fake glide API为主，env-gated集成存在不等于设备通过，未给本轮可靠性能。建议继续观察依赖落地，先核内存注册、远端失败和完成边界。

<a id="reliability-engineering"></a>
## 评估与工程验证方法

### Hugging Face MIC：缩小架构、冻结使用路径与人工基线，仍不能替代真权重

**Hugging Face；[11-03官方说明](https://huggingface.co/blog/transformers-mic)，代码基线[transformers-mic@513ffe5d](https://github.com/huggingface/transformers-mic/tree/513ffe5db99a9db96b0a2075b9f01c819a0ad6ce)。** 用保留层类型/共享关系的小配置、原样tokenizer/processor、可执行usage及人工审定baseline组成contract，并以固定fleet revision重放。UMT5的SDPA因果泄漏在CPU小模型上回放可捕获，eager路径却漏检；13.4秒是所选CI runner步骤，不含setup。

这不是已证明有效的阻断门禁：截至11-01仅report-only；[固定证据汇总](https://github.com/huggingface/transformers-mic/blob/513ffe5db99a9db96b0a2075b9f01c819a0ad6ce/docs/evidence.md)初始五个月回放为捕获1、漏4、误报1，补usage后才改善，不能当真实PR前瞻召回率。Whisper过拟合fixture在合成故障18/24→24/24，但随机/过拟合都漏过真权重辅助解码bug。**建议考虑复现：**为自己的模型固定真实调用、dtype与attention backend，再测故障敏感性和误报；不因fixture通过替代真checkpoint设备测试。

### Transformer Engine overlap测试改查storage别名，尚无runtime修复交付

**NVIDIA/TransformerEngine；[#3718](https://github.com/NVIDIA/TransformerEngine/pull/3718)，11-03 20:59:08Z，main@`dc7302a2007e696a61e94c78570aedf393a66e99`。** 最终只改分布式layer runner测试：用saved-tensor hooks检查forward/backward保存tensor是否与跨层复用的comm+GEMM workspace存储重叠，避免单层数值测试看不到下一层覆盖。cuBLASMp需≥0.11.0才有可观测buffer view。

该项强化根因检测，不应写成新增算子或某线上梯度损坏已修复。**建议深入阅读测试方法：**多层重叠训练可借鉴storage范围检查，仍须在目标后端跑完整梯度与生命周期回归；本轮仅核测试源码，未执行多卡测试。

<a id="hardware"></a>
## 硬件软件协同的新材料

### AMD FlyDSL Mono报告：小batch融合收益随负载缩小，合成接受率不能代替自然Agent体验

**AMD；[11-03官方技术正文](https://rocm.blogs.amd.com/software-tools-optimization/flydsl-mono-atom/README.html)，代码引用固定[ATOM@53e4fb52](https://github.com/ROCm/ATOM/tree/53e4fb52403707ab07b69ee909501ea3cc25ca5c/atom/mono)，实验截至10-15。** 新增的是技术解释与配套比较材料，不是当天才首次实现Mono。跨stage提前读权重、就绪tile交接及TP归约重叠，需要模型专用调度与一致fallback。

8×MI355X节点上，MiniMax-M3 MXFP4/TP4、1M context/EAGLE3三draft：batch1 GPU-step 11.47→7.00ms（1.64×），额外indexer CP才到6.26ms；batch4仅1.16×。vLLM V4.1-Flash TP2的C1 p90 interactivity318.1→429.1tok/s/user，每mode仅一轮3600秒，**接受长度人为固定3.51**，TTFT1.18→1.28秒。ATOM AgentX表无Mono-off配对，含缓存输入的6580总tok/s/GPU中输出仅42.5，不能作生成吞吐。**建议考虑复现：**锁代码/长度/并发，用自然接受率同时测TTFT、输出吞吐与尾延迟，不把不同测量相乘。

<a id="coverage"></a>
## 实际覆盖、补录、更正与下一轮

**本轮实际检查。** 六核心仓完成72h merged PR与默认分支发现分页（656个merge、637个默认分支commit），检查258个release对象；123项候选打开最终files，正文按机制合并为93组/117个PR事件。扩展34仓加新纳入LMCache共35仓：1927个72h merged PR、786个默认分支commit（34仓主窗、LMCache回溯窗），129项当前候选打开元数据/最终files，另核10月安全修复。普通依赖、CI/格式维护不当成果；未审候选、部分生成/native大diff均留队列。核心627项、扩展1829项是候选/元数据过滤待核集合，含历史或普通维护，不是同等优先级的新成果数。

模型端52个固定HF组织208次createdAt/lastModified模型与数据集查询完成到回溯界限，76项主窗树比较；另外按问题发现NorthCUA、Hydrogen、Prism、Nota等。57个官方入口43个直接成功，其余部分web恢复，动态/门控材料仍缺。全局created模型15页/14195条仅发现与选择审读，未枚举全局lastModified。技术类别覆盖模型报告、预训练/架构、多模态、后训练/数据、推理服务、开源系统、GPU/NPU/TPU与可靠性，没有按类别配额收录。

论文七类arXiv new/current列表合并2058个work，64篇相关候选中63份固定正文取得，57篇论文及1份官方研究工件通过本轮选择审读；另检查HF daily/主题查询、Ai2/Anthropic原始研究。**再次直连取得的最新列表仍标11-03，无法据此证明11-04没有列表。** submitted-date API后续恢复四页3141个唯一记录，按秒级回溯边界排除1条早18秒记录后为3140；最晚返回11-02 18:59:32Z。主窗请求虽200却为0条，不作无新增证明，updated请求还与submitted逐字同体且查询被改写，因此主窗2026-11-03 00:58:48Z→11-04 01:05:04Z的精确新提交与真实更新仍未核实完整；所有论文明确写v1提交与列表日期。合并继承的10845个candidate与1480个replacement可能重叠，不相加，也不因当前列表成功就关闭更旧缺口。API在七类之外又筛出40个标题候选，6份全文另核、收录5项，其余留队列；ACL返回200但19MB正文只收229KB即超时，OpenReview API403，TMLR/PMLR历史未完整覆盖。

硬件端读取NVIDIA博客、AMD ROCm博客/发行注记、Intel Gaudi、Google TPU、AWS Neuron和Ascend公告等10个当前入口，再打开4份原始技术/发行/发布材料。Gaudi/Neuron/ROCm release首页与上轮相同，TPU正文相同；这只说明所读快照，不说明厂商没有新发布。Ascend HTTP200仍是SPA壳，搜索索引中CANN9.2.1/10-28仅作为线索，未恢复detail正文；CANN/HCCL/libtpu/Neuron逐组件历史仍续查。Lumai光学prefill部署声明缺可比工作负载、质量/功率方法，不采纳90%/50×目标；NVIDIA rel-26-11地址正文却是26.10，不把last-updated当新release。

**访问与恢复。** 初始shell代理127.0.0.1:7890拒连后，仅在检索进程用proxy-free直连恢复，未改全局配置，失败原记录保留。OpenAI新闻/研究、部分Microsoft/Mistral/xAI/Recursive/Dory及Sarvam正文有403、超时或动态历史缺口，部分官网可读也不代表历史全覆盖。Hydrogen BF16与Nota patch等401门控材料未取得；不把公开卡当私有代码已审。网页工具对MIC/NVIDIA notes报内部错，已直连保存原文。归档正文和状态分别记录上传结果，不能把本地保存当远端成功。

**补录和更正。** NorthCUA原11-01报告与11-01/02权重、Hydrogen/Nota等11-02工件、论文原10-30至11-02提交，以及LMCache10-12修复/10-14发行均标原日期补录。TorchTitan#5731从昨日open转已合入，不能仅按曾出现URL去重；PyTorch Inductor八项从昨日撤回到本轮新commit重新落地；SGLang#47653与Ascend#20567的回滚保留。llama.cpp b12088为11-04发布的预发行，b12089只有主干/tag核实。交叉审已在发布前修正AITER主MX接口forward-only、FP4背靠背/graph测速口径、KDA参数partial需调用者归约及不同训练脚本环境，未覆盖历史日报正文。

**持久化与续查。** 本文按新颖性、证据与实践影响展开，不以候选总数或固定Top N决定收录。台账记录316个稳定来源版本事件（核心117、扩展115、模型23、研究58、工程验证2、硬件材料1），同工作多个工件不等于独立成果；未审项不冒充通过筛选。全部134条继承backlog保留，单个PR状态解决不关闭整个区间。下轮从11-04 01:05:04Z续查并回溯72h，同时优先补未审/失败历史与待同步文件；完整语义覆盖水位仍未推进。检查日志与证据在本地 `.monitor/research-2026-11-04/`，报告/README是本轮仅上传的两个文件。

