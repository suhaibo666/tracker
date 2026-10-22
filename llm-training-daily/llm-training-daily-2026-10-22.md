# LLM 技术动态日报 · 2026-10-22

> 时区：Asia/Shanghai（UTC+8）。增量扫描窗口：**2026-10-21 08:56:38 → 2026-10-22 08:58:39**；72小时回溯自 **2026-10-18 08:56:38**。从上次成功归档位置续查，旧缺口继续单列；事件游标不是所有来源语义审阅完成的水位。
>
> 实际生成时间：**2026-10-22T09:22:55+08:00（Asia/Shanghai）**。下文精确事件时间除注明北京时间外均为UTC；首次提交、公告、权重上传、版本修订、合入与正式发版分别记录。截止后读取用于核实本窗事实及访问状态，不把后续新事件算入本窗。
>
> 本轮实际打开官方论文全文、模型卡/config/权重目录、技术文及PR/commit。所有模型质量、训练、性能、能耗和硬件测试均为作者/厂商披露；本轮未独立运行GPU/NPU/TPU实验。关键机制做了选定代码路径与失败边界审阅，不等于全部device kernel或所有候选已审计。

## 本次重要变化

- **预训练与架构有新可读工件。** 腾讯Youtu开放120B/7B多模态Base、KDA/门控MLA与模态路由配置可核；名单外Flint发布约40B/2.5B本地代码模型与量化栈，低比特输出头及离线蒸馏值得复现。长context CPT的新实验支持在1B级继续加长训预算，负offset经验law仍须限定拟合区间。
- **正确性修复比峰值kernel倍率更急。** vLLM MoRIIO需等所有READ落地，SGLang spec-KV移动改用仍有效的verify window；Ascend PP/SP防静默截token。FlashAttention未用V槽NaN污染已有main修复，但当天较早的beta36尚不包含它，P2P consumer旧超时/late-WRITE缺口也未随这些修复关闭。
- **后训练省算力必须分开目标变化与实现优化。** DeepSpeed词表分片损失新融合路径可消除大量临时量；DPO随机图研究说明主要显存收益来自保全监督的selected-head实现，单branch省约三成时间仍有小幅accuracy代价。TRL为MoE LoRA漏训、参考policy及多模态占位符增加明确契约。
- **推理优化按完整配置决定是否采用。** Ling Tiny UBTree有新drafter但公开SGLang集成尚未就绪；Liquid DSpark吞吐测量有公开协议、无质量评分。TRT长prefix索引并行、GLM FP8 sparse MLA、FlashInfer小token MoE/NVFP4与AMD融合路径各有硬件、精度、并发和端到端验证边界。
- **硬件软件协同出现可复现入口。** FlyDSL的PyTorch集成披露同机vLLM A/B，低精度与BF16的基线不同；XLA增长式显存池回收、对称内存录图与TorchTPU临时副本生命周期有具体main改动。llm-d的XPU/SGLang P/D目前是受限TCP/LevelZero路线，不能当RDMA组合已就绪。
- **旧结论出现实质纠正。** Tokenizer v2把U-statistic与英语/土耳其能力相关方向改为负，控制参数量后增量解释力也收缩；AITER带宽脚本数值跳升来自测量协议修正。负面结果继续提醒：排序指标、pass@K、记忆条数或低频词增加都不能替代最终任务可靠性。

目录：[模型与报告](#models) · [预训练与架构](#pretrain) · [后训练](#posttrain) · [推理与系统](#inference) · [工程正确性](#correctness) · [核心开源](#core) · [扩展开源](#engineering) · [硬件软件协同](#hardware) · [评估与纠正](#evaluation) · [版本观察](#revisions) · [覆盖与缺口](#coverage)


<a name="models"></a>

## 模型与技术报告

### Youtu-LLM-A7B-Base：120B/7B多模态基座开放，KDA与门控MLA混合栈可核

**腾讯优图；可见整包提交2026-10-21T07:53:34Z，`a4fe6c899679a4c66da0d6e52cdbfd2f1814b0fb`。** [固定模型卡](https://huggingface.co/tencent/Youtu-LLM-A7B-Base/blob/a4fe6c899679a4c66da0d6e52cdbfd2f1814b0fb/README.md)、[工件目录](https://huggingface.co/tencent/Youtu-LLM-A7B-Base/tree/a4fe6c899679a4c66da0d6e52cdbfd2f1814b0fb)。仓库createdAt为1月，不能当此次权重发布时间；本轮确认24个BF16权重分片、独立MTP参数、配置和自定义实现。作者称约120B总参数、每token激活约7B，11.5T多模态token、262,144训练上下文；未披露可复算的数据配比、清洗/去重、分阶段训练token、优化器、训练算力或后训练配方，这是base模型。

[固定配置](https://huggingface.co/tencent/Youtu-LLM-A7B-Base/blob/a4fe6c899679a4c66da0d6e52cdbfd2f1814b0fb/config.json)为48层、宽3072，按三KDA线性层+一MLA循环；首层dense，其后256路由专家top8、1共享专家，KV低秩512、1个MTP层。其意义是把递归状态与精确attention组合，而非仅由小激活参数就保证低服务显存。[实现](https://huggingface.co/tencent/Youtu-LLM-A7B-Base/blob/a4fe6c899679a4c66da0d6e52cdbfd2f1814b0fb/modeling_youtu_vl_moe.py)的`build_youtu_vl_attention`按配置选择两种层；`YoutuVLTopkRouter.get_topk_indices`按模态选择bias，在原始logit减bias后top-(k+1)取前k，权重另由sigmoid(logit)归一化得到。KDA执行交给FLA的chunk/recurrent kernel，本轮只核此交接与状态写回，未审计FLA内部或执行训练。

作者text表中5-shot MMLU-Pro为78.61，对照DeepSeek-V4-Pro-Base67.11；但SuperGPQA42.28低于54.20、RULER4K–256K92.38低于93.11，不能写全面超越。VL开放题用MiniMax-M2.7裁判，MCQ抽取选项后EM，须保留各自指标。**建议深读并小规模复现**长序列精确召回、模态路由与MTP，而非直接接聊天接口：卡要求Transformers5.12.1及自带SGLang0.5.17 adapter；高并发图像traffic开启overlap scheduler可能挂起。权重为[自定义Youtu许可](https://huggingface.co/tencent/Youtu-LLM-A7B-Base/blob/a4fe6c899679a4c66da0d6e52cdbfd2f1814b0fb/LICENSE.txt)，含EU地域条款，不能标MIT/Apache。

稳定ID：`hf:tencent/Youtu-LLM-A7B-Base@a4fe6c899679a4c66da0d6e52cdbfd2f1814b0fb`。

### Flint：新机构发布39.8B/2.5B本地代码模型，低比特输出头与离线蒸馏值得复现

**Recursive Superintelligence；名单外线索由Transformers新模型支持反查。BF16整包2026-10-21T14:24:03Z，`2a247be0c0aec895ccb0ac7ca92229ec6df37288`；三款量化变体15:03–15:08Z首次提交。** [固定模型卡/15个BF16分片](https://huggingface.co/RecursiveSI/flint/tree/2a247be0c0aec895ccb0ac7ca92229ec6df37288)、[官方发布文](https://recursive.com/updates/introducing-recursive-flint)。这是新权重事件，非仅框架支持；HF元数据统计39,785,287,456个BF16张量元素。卡的约2.5B激活包含输出头、排除完整输入embedding，body约2.2B，两种口径不能混用。

目标是在本地内存带宽约束下运行代码agent。[固定配置](https://huggingface.co/RecursiveSI/flint/blob/2a247be0c0aec895ccb0ac7ca92229ec6df37288/config.json)为49层、hidden2048、32Q/4KV：首两层dense，其后47层256专家top6+无门控共享专家；35层513-token滑窗（当前+此前512）用RoPE，14层全attention不用位置编码。每头QK RMSNorm及可学习sink logit控制归一化。发布方称从35B/3B配置搜索专家/宽度得到40B/2.5B；量化时把带宽占比较大的`lm_head`也降4bit，再以BF16 teacher对量化模型自己的预采样coreset做离线蒸馏。自动研究系统参与和训练步数节省是厂商陈述，未公开足以分离各设计贡献的完整消融。仅披露从零预训→midtraining→SFT→RL及公共/许可/爬取/合成数据类别，没有总token、阶段配比、RL算法或GPU预算；本轮未发现由发布入口链接的独立技术报告。

卡自报Terminal-Bench2.1 **54.9%**、SWE-bench Pro v2 **89.7%**、Verified **74.2%**；温度1/top-p0.95/top-k20，agent任务至少五次尝试取平均pass@1、最长5小时。TB用Terminus2并把网上直接检索得到的解答判失败，SWE用mini-swe-agent且无联网；部分对照引用开发者结果，不能视作统一harness重跑或与其他日报代码分数直接排名。博客称512K可用但建议≤256K，**发布config及本地配方是262,144且prompt/output共享**，先按发布配置验证。

部署应按格式与runtime区分：[NVFP4 GGUF](https://huggingface.co/RecursiveSI/flint-nvfp4-gguf/tree/6ec43e19b565fc8d560e2d04985d7724f20673bf)实际22.863GB，卡仍建议≥48GB GPU内存或Spark统一内存；[INT4 GGUF](https://huggingface.co/RecursiveSI/flint-int4-gguf/tree/9c47151036bc2c34331142dda04488bceab2bc9f)25.993GB、Mac要求≥48GB统一内存。不能按22.9GB文件宣称24GB全上下文可跑。[MLX INT4](https://huggingface.co/RecursiveSI/flint-int4/tree/3b93f9053b530488405ae9fca51d4900d6ce21f0)约22.6GB，projection/expert为group64 affine INT4、embedding INT8；rec专为单会话及指定M4/M5 Pro/Max优化。其[固定README](https://github.com/recursive-oss/rec/blob/e53663b262af4cb1ca498e596ca8b1a2d79e0800/README.md)报告M4 Pro48GB、4bit、1K–252K prompt/1024输出、每长度五次、排除prefill的1.4–2.6×相对llama.cpp吞吐，不能当prefill或多agent端到端收益。

**建议深入阅读/同机复现。** [Transformers #49824](https://github.com/huggingface/transformers/pull/49824)于10-21T15:04:23Z合入`0ea108d`；[实现](https://github.com/huggingface/transformers/blob/0ea108daf169dd66a1800e48a56d2251d4ff76bd/src/transformers/models/flint/modeling_flint.py)中`eager_attention_forward`把sink拼入softmax分母再丢弃该列，`FlintTopKRouter.forward`按sigmoid top6归一化、`FlintSparseMoeBlock.forward`直接加共享专家；测试覆盖sink、全attention无位置编码及checkpoint往返。SDPA不能应用sink，默认eager；长输入需sink-aware FA3/FA4，不能任意换attention backend。主卡14:24仍写“未提供Transformers建模代码”，15:04后的主线支持已是更新事实，但不证明已进公开wheel。权重Apache2.0；本轮未运行模型、量化训练或rec内核。

稳定ID：`hf:RecursiveSI/flint@2a247be0c0aec895ccb0ac7ca92229ec6df37288`；`github:huggingface/transformers:pr:49824`。建议将`RecursiveSI`加入后续观察来源，当前补查只覆盖这一个新机构的4个公开模型仓，不扩张为全球新机构全覆盖。

### Ling3 Tiny UBTree：树形drafter新增权重，Flash本窗只是卡与配置修订

**蚂蚁inclusionAI；Tiny权重2026-10-21T13:52:08Z上传，最终卡`05abc072b6547f49dc6920cbdf7b6f4402950e54`；Flash最终卡`59ced6e1e8fa434570be038e262c3e62552cf9a4`。** [Tiny固定卡](https://huggingface.co/inclusionAI/Ling-3.0-tiny-UBTree/blob/05abc072b6547f49dc6920cbdf7b6f4402950e54/README.md)、[Flash固定卡](https://huggingface.co/inclusionAI/Ling-3.0-flash-UBTree/blob/59ced6e1e8fa434570be038e262c3e62552cf9a4/README.md)。Flash虽有“Upload folder”提交，与截止前10-14版本逐文件比对，`model.safetensors`对象未变；它不是本窗第二个新权重。

问题是并行逐位置提出token时，候选间不一定相容。卡述五层DFlash作unigram proposer、学习bigram selector并行评分相邻候选，预算内建树交给target验证；先DFlash warm start，再联合训练backbone与selector。Tiny/Flash仓内分别183.1M/482.0M参数，**不含与target共享的embedding和LM head**；block16包含anchor，默认64非根节点+1anchor，不等于benchmark的树节点数。实际Tiny配置可核五层、hidden1536、16Q/16KV、候选top64、rank256 selector；完整训练数据量与GPU预算未披露。

卡报Tiny GSM8K接受长度9.53、相对target-only AR端到端3.055×，DFlash接受长度6.57；条件是block16、TP1、并发1、BF16、greedy/no-thinking、49节点，三轮中位数对一轮AR。Flash用TP4、32节点，不能与Tiny倍率直接横比；两卡未给GPU型号，本轮不据此估算生产收益。**建议观察集成后复现**：当前卡明确Ling3的SGLang UBTree适配尚未向开源社区发布，下载drafter不能直接按官方SGLang配方运行，也不能作为独立模型生成。权重MIT；2609.39972是既有方法论文，本轮不把它重报成新论文。

稳定ID：`hf:inclusionAI/Ling-3.0-tiny-UBTree@05abc072b6547f49dc6920cbdf7b6f4402950e54`；Flash修订`hf:inclusionAI/Ling-3.0-flash-UBTree@59ced6e1e8fa434570be038e262c3e62552cf9a4:card-config`。

### LFM2.5-24B-A2B DSpark：327.7M drafter公开快照，收益随并发和target精度变化

**Liquid AI；BF16/GGUF可见快照2026-10-21T01:11:30/31Z，target01:14:14Z添加drafter。** [固定BF16卡与工件](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-DSpark/tree/e052e5b1dd103c38cdd4f8ddc068d7dea4586f55)、[GGUF](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-DSpark-GGUF/tree/9222226d8d0fea777a25988e7389e543e99e4e0d)。两drafter仓当前历史均只留一次super-squash，createdAt分别10-19/20；因此记录本窗已可核的公开快照，无法由现存历史还原首次公开时刻或相对旧权重的数值差异，不称LFM2.5基座重新训练。

drafter为327.7M BF16、五层全attention、hidden2048、32Q/8KV、rank256 Markov head+confidence head、block9；生成候选由target逐token验证。作者在**每shard单B300、FP8 target、BF16 KV、temperature0**的8项完整evaluation split上，报告每GPU输出吞吐相对无spec baseline均值：并发1 **2.38×**、32 **2.69×**、256 **1.71×**；并发1平均5.16 token/step含target bonus、上限10。不是任意GPU或BF16 target的加速，也不是任务得分改善；卡明确batched验证的浮点差异可能令greedy输出不同。

**建议有该target部署者复现**[原始协议](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-DSpark/blob/e052e5b1dd103c38cdd4f8ddc068d7dea4586f55/benchmarks/protocol.json)：SGLang基线`488869c`另加仓内runtime.patch，关闭radix cache/torch compile，小数据集重复完整等频passes填足请求波次，输出cap512–2048；协议明确无quality scoring/置信区间，聚合吞吐不是streaming latency。卡链接#31041/#43939但本轮未单独核合入态，不能将所有既有发行版视为可用。训练语料、token与算力未披露；[LFM1.0许可](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-DSpark/blob/e052e5b1dd103c38cdd4f8ddc068d7dea4586f55/LICENSE)含年度营收10M美元商业使用阈值条款。

稳定ID：`hf:LiquidAI/LFM2.5-24B-A2B-DSpark@e052e5b1dd103c38cdd4f8ddc068d7dea4586f55`。

### Granite Speech：TurboCTC从英语扩到七语，Plus v2610改进对齐但一般WER并未全升

**IBM；TurboCTC整包10-21T13:56:08Z更新，最终`1dd23f46f6d4e6eae2531982ba2e33220086dd97`；4.1-2B-Plus整包10-21T14:56:22Z，`e69efe37f97b6283c61f5a8dadda2b3eb67765d6`。** [TurboCTC固定卡](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc/blob/1dd23f46f6d4e6eae2531982ba2e33220086dd97/README.md)、[Plus固定卡](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus/blob/e69efe37f97b6283c61f5a8dadda2b3eb67765d6/README.md)。本轮比对新旧tree，TurboCTC单safetensors与Plus三个分片的对象均更换，确为权重增量；Plus卡把版本标为10-19更新，上传时间仍单记10-21。

TurboCTC是470M非自回归CTC，16层Conformer、8倍时间降采样，输出词表扩成16,384多语BPE，新增法/德/西/葡/意/日，支持原生标点与大小写。卡披露约153K小时公共/合成语音及来源小时表、8×H100训练24天；不能把所有列出的重复任务或语料重新计为独立音频。作者英语九测试集平均WER5.78→5.51%；卡明确多语更新的橙点仍为**非官方榜单结果**，榜上条目仍旧英语版。要求Transformers≥5.16.0；本轮未跑ASR。

Plus v2610针对关键词、说话人标注和词级时间戳，2–5分钟拼接长音频的平均时间偏移AAS **814.3→328.3ms**；但新Open-ASR测试集的一般WER **6.01→6.10**，并非所有指标提升。该表为beam4、repetition_penalty2、无spec；旧榜测试集已变，不能与旧5.71直接比较。卡披露约32×H100/5天，词级对齐用MFA再由CTC复核，英语保留95%、非英语70%的低误差样本；完整可复现训练流水线未在卡提供。两模型Apache2.0。**建议按业务选择并复现**关键词召回、长语音时间漂移及重复生成；Plus不输出标点/大小写，时间戳每10秒回绕，需按例程展开。

稳定ID：`hf:ibm-granite/granite-speech-5.0-470m-turboctc@1dd23f46f6d4e6eae2531982ba2e33220086dd97`；`hf:ibm-granite/granite-speech-4.1-2b-plus@e69efe37f97b6283c61f5a8dadda2b3eb67765d6`。

### Voxtral 3：回溯补漏语音agent模型族，公开预览不等于可下载权重

**Mistral；官方发布日期2026-10-20，72小时回溯补录。** [发布文](https://mistral.ai/news/voxtral-3/)、[Small Transcribe 3模型页](https://docs.mistral.ai/models/voxtral-small-transcribe-3-0)、[Realtime模型页](https://docs.mistral.ai/models/voxtral-mini-transcribe-realtime-3-0)。问题是语音agent既要识别、结束轮次、推理/调用工具，也要及时合成回复；新SDK把这些阶段串成流式pipeline，不能等同“一个模型原生包办全部”。Small页标32K、`voxtral-small-transcribe-3-0`、Public Preview Premier v3.0；Realtime `voxtral-mini-transcribe-realtime-3-0`支持单WebSocket中的转录、说话人分离、context bias及声学/语义轮次结束检测，latency参数80–2400ms。

[TTS模型页](https://docs.mistral.ai/models/voxtral-mini-tts-3-0)为`voxtral-mini-tts-3-0`、预览版，现写支持9语言、约90ms首音频；发布文却称TTS35+语言/50+方言，两个一手页面口径冲突，应按具体语言实测，不能合并成已验证支持范围。转录launch的FLEURS45是所有对比模型共同支持语言交集，日/韩/泰/中文用CER，其余用WER；Realtime图另用35语言，不能直接合并。架构、完整训练数据/算力与多数参数规模未披露，未引用图中未取出的分数。

官方说可用EU-hosted API或自部署，本轮打开的mistralai HF库存没有对应新Voxtral仓，产品页也未给可公开下载的这一代checkpoint/权重许可证，故归**API/部署产品发布**。**建议观察/按真实语音链路复现**打断、数字/人名、关键词误引入、工具执行和首段音频延迟；80ms配置值或单TTS首音频不等于端到端agent回复延迟。

稳定ID：`mistral:voxtral-3:2026-10-20`；具体API模型ID如上。本轮页面快照时间与sha256在证据账本中。

### Nemotron 3.5 Retriever技术文：冻结语言塔延续文本向量空间，再训练视觉检索和重排

**NVIDIA；10-21新[官方团队技术文](https://huggingface.co/blog/nvidia/nemotron-retriever-world-class-agentic-retrieval)。** 本窗三款BF16 Embed/Rerank逐文件对照只README改变，不能把聚合公告写成当天全部新权重。部署基线：[Embed8B](https://huggingface.co/nvidia/Nemotron-3.5-Embed-8B-BF16/tree/781e779ee4ca9c2d2a4f38321993556baeab49a1)、[Embed1B](https://huggingface.co/nvidia/Nemotron-3.5-Embed-1B-BF16/tree/e24ca1e346bcc53572bd7bdc02b3e8ebf66c197c)、[Rerank8B](https://huggingface.co/nvidia/Nemotron-3.5-Rerank-8B-BF16/tree/f88f8d9104726747791283ab156d2fcdb29e90a7)。

读图表/扫描PDF只抽文本会丢证据。8B卡列实际8.38B，把Nemotron3文本embedding塔冻结，只训练来自Ministral3的视觉encoder及projector；1B卡列实际1.55B，以cosine+MSE蒸馏8B teacher，语言塔同样冻结。卡称保持对应旧Nemotron3的文本向量空间，支持复用旧文本索引；本轮未做逐向量等价验证。两者query只接文本，document接图/文/两者，32,768上限；mean-pool+L2后分别4096/2048维，截断降维后仍须重新L2。Reranker则joint编码query-document、输出raw relevance logit，不是校准概率。

训练卡区分**2.3M含图样本**与继承语言塔的文本训练量（8B50M+、1B8.5M+）；不要把它们全当本轮视觉阶段新消费的token。Reranker是文本阶段后图文阶段，卡列21M文本/2.3M含图样本。官方技术文的同agent/工具/任务/语料配置实验报告加reranker后agent输入+输出token约少13–14%；这是特定retrieval pipeline收益，非训练算力节省。NVFP4单GB200文档吞吐101.3→128.8篇/s也不能迁移到任意硬件/文档长度。总GPU训练预算未给。OpenMDW1.1。**建议深入阅读并复现**先验证旧文本索引兼容，再比较图文检索召回、top-k重排成本与最终任务成功率。

稳定ID：`nvidia:nemotron-3.5-retriever-technical-blog:2026-10-21`。


<a name="pretrain"></a>

## 预训练、架构与压缩

### 长上下文CPT：1B级模型训到约100B长token仍获益，但经验law不是无限扩张保证

**Don’t Stop Long-Context Continual Pre-Training Too Early；清华大学、OpenBMB；2610.29352v1，10-21T07:46:55Z提交，本轮发现；尚不归入10-21公告批次。** [§3–5、AppA3](https://arxiv.org/html/2610.29352v1)。研究把非embedding参数、短context token与长context token分开建模，用LongPPL观察远context敏感token；Constant-Last4保持常学习率训练、仅评估时平均最近四checkpoint，让观察不受预定终点LR衰减绑住。

16×H800、4K→32K，31M–821M非embedding参数（总165M–1.22B）；两组留规模外推分别为75配置拟合→25测试、100拟合→9测试，在**log(LongPPL)空间**R²>.95、MAPE<2%，不是下游accuracy误差。最大模型32K CPT至约100B token，RULER总体继续提升；同样128K追加2.10B、长SFT2.10B/agent SFT4.20B后仍见预算收益。**拟合offset<0，作者明确只是有限区间经验law**，代理分数提升亦非每项checkpoint单调。建议深读/复现固定模型的早期曲线外推，先验证业务远依赖任务及长数据质量，再扩预算；目前无多十B模型的直接验证。

### GaP-MoE：投影co-routed专家梯度可促分工，但当前证据仍是小BERT/GPT-2

**GaP-MoE；University of Seoul；2610.29355v1，10-21T07:48:07Z。** [§3–4](https://arxiv.org/html/2610.29355v1)。按router概率抽一位leader保原梯度，follower从输出梯度去掉leader近期正交buffer子空间；只把leader未投影梯度均值写回FIFO，避免立刻把干预结果当历史方向。它修改反向更新，前向TopK路由不变，条件理论也不等于全模型专家功能全局正交。

BERT-base12层每层4专家Top2、GLUE5epoch最佳checkpoint均分81.96→82.06；缩小六层GPT-2、仅后两层用GaP，WikiText103 ppl78.80→76.50、enwik8 BPC2.005→1.953。XMoE/SimSMoE对照为作者统一实现核心机制，非官方全配方等价。改善小、超参未系统扫、无现代大MoE训练吞吐/通信测量。**建议观察/小模型复现**buffer范数、短期梯度稳定性与layer选择；暂无证据支持大规模直接替换router或移除负载均衡。

### HMDLM：先预测层级再补token能改善生成指标，但每步多一次计算

**HMDLM；IIT Delhi、IBM Research；2610.29255v1，10-21T06:06:01Z。** [§4–5、Tables1–6](https://arxiv.org/html/2610.29255v1)。把token与hierarchy状态联合mask/reveal，level各用noise schedule；反向先采层级/是否揭示，再以更新层级预测token，已揭示状态保持。训练层级来自频率或任务规则，测试空位由模型自己预测，不能假设推理有gold层级。

LM1B约139M、33B/65B token，模型PPL上界MDLM≤27.04/25.49，HMDLM≤24.30/22.88；生成1000样本三seed，不能当三次训练。WikiHow约170M、256–1024长度，frequency版优于ADLM，但position版512长度PPL21.76反差于MDLM21.04；HMDLM训练稍长以匹配mask暴露，不是相同步数。Sudoku25M board greedy .83→.88。**建议观察/复现**有清晰层级的任务，比较等FLOPs、等有效监督token与更强排序baseline；小额参数开销不代表低运行开销，论文明确两阶段增加计算。

### Lafa：连续flow可借AR联合训练保住验证器，最高加速来自自投机而非单步flow无损生成

**Lafa（Large Flow Language Models）；作者机构本轮未核实；2610.28559v1，10-20T17:20:29Z提交，10-21公告、本轮发现。** [原文§2–3](https://arxiv.org/html/2610.28559v1)。为减轻连续flow扩至十亿参数的训练难度，Lafa保留prefix因果attention/KV cache，block内双向去噪并从loss学token时间进度；embedding只学方向、固定范数，联合AR交叉熵、去噪交叉熵和embedding误差。单步flow草稿再由同模型AR路径逐token验证，**只保证与这个训练后checkpoint的greedy AR输出一致**。

§3以Qwen2.5-Instruct 0.5/7B及Qwen3-8B Base适配，Dolci-SFT 9–27B token；部分baseline同初始化/数据/预算，其他公开checkpoint不等预算。单H100、batch1，7B block16在IFEval/GSM8K/HumanEval平均质量接近AR-SFT时达**2.54× TPS**；不是六项全平均或高并发吞吐。1–2步flow-map蒸馏仍有质量差距，置信接受亦会损质量；“至少少18×训练token”不计原AR预训。**建议深读/小规模复现**同初始化AR-SFT、块大小和准确率—延迟曲线，分别测flow、flow-map与自投机，不把三个模式的最优点拼在一起。

### AsymLoop：一次encoder供固定Global KV、decoder反复读写，效果含额外参数和优化器调度贡献

**AsymLoop；上海交大、中科大、南洋理工、南京大学；2610.28489v1，10-20T16:36:33Z提交，10-21公告；回溯补录。** [§3–4、AppD–E](https://arxiv.org/html/2610.28489v1)。整栈循环会不断重算全局表示；AsymLoop把物理层各半分为一次因果encoder和共享迭代decoder，每个decoder层把固定Global KV与当前Local KV放进同一softmax，Global投影跨loop复用。另采用既有谱学习率ALRS，不能把其收益全部归新attention结构。

FineWeb-Edu从零训练，标称60M–1B、1.68–20B token；Table3在同有效深度的“1B”配置Full Loop ppl**8.91→8.86**，Table1七任务zero-shot49.58→50.85。AppA实际总参数1.34→1.44B；AppD前向FLOPs额外9.38–15.23%，**同有效深度不等于同FLOPs**。AppE小规模匹配分析训练FLOPs仍有loss优势，但135M step .296→.367s、峰值33.6→33.8GiB。**建议观察/复现**架构与ALRS独立消融，预算同时报非embedding参数、实际FLOPs、wallclock和KV容量，尚不足据此替换大规模预训骨架。

### 随机token替换：重复小语料的后期loss改善多来自缓和过拟合，短预算反而可能受损

**Benefit and Cost of Random-Token Replacement under Repeated Data；Flitto；2610.27590v1，10-20T03:57:15Z；10-21公告、回溯补录。** [§3–7、Tables1–4](https://arxiv.org/html/2610.27590v1)。保持预测target，只以p=.15均匀替换输入token；已知熵Markov源把终点excess loss拆成近似、估计与后期memorization项，发现替换主要减少最后一项，并**没有消除训练文本记忆/逐字再现**。这是受控源的操作性分解，不能据此宣称真实语料因果项均已识别。

134/336M、十真实语料，每套约2.3–7.4M token，主配方训练至约22epoch；Table1韩/日web终点loss降.93/1.14nats，但各自最优checkpoint仅降.06/.09；代码最小loss反升.06，census最终亦升.06。Table3把cosine压到2epoch时两web均差.06nats；最佳预算的益处远小于过训终点。**建议小规模复现**资源有限且需多epoch的语料，必须与early stopping、按预算调学习率和不同p比较；论文无大模型/下游能力结论。

### E8-QAT：算术格点投影避免搜索大codebook，2-bit仍须计逃逸记录与非量化元数据

**E8-QAT；北京大学、香港城市大学、腾讯；2610.27699v1，10-20T06:17:09Z；10-21公告、回溯补录。** [§3–5](https://arxiv.org/html/2610.27699v1)。Hadamard后每8权重投影至无限E8格、STE训练，从有限65536格点PTQ初始化；导出以每组16bit核心索引加稀疏逃逸坐标保留训练格点。核心payload为2bit/weight，逃逸使其变为2+20ρ，**共享codebook、scale及格式元数据另计**，并非所有模型内存严格2bit。

Qwen3.6-35B-A3B、Llama3.1-8B，UltraData-SFT-Agent约50.8M token/pass，MoE八GPU/750steps。Table1 Qwen LCB：LC-QAT78.58→80.09，仍低BF16的81.42；BFCL75.22→74.16反降，IFEval80.96→82.99仍低BF16的85.40。Llama四任务均分62.12→65.11，仍低BF16的72.54。峰值−80%、吞吐1.97×用**各自最大可容batch**，不能当同batch kernel提速。**建议复现**时核导出逐权重等价、真实bits与同batch质量/时延；论文只验证两类模型，训练速度4.5%提升是配对375step特定条件。

### OptEP：联合选专家比逐个重要性排序更能保互补性，但校准与恢复都要成本

**OptEP；LinkedIn、MIT、Google；2610.27123v1，10-19T18:24:03Z；10-21公告、回溯补录。** [§3–5、Tables1–3](https://arxiv.org/html/2610.27123v1)。以整层输出重构最小化连续专家保留权，再top-s离散；逆频率权重避免常见专家主导。选择时层输入来自完整模型，各层独立；可选EWU则顺序用已剪且恢复的前层激活，仅更新保留专家。

Qwen30B/235B、GLM4.5-Air、Mixtral，4/8×H100/seed；仅数学/代码，以原模型**正确且未截断**的域内轨迹校准。Table1 Qwen30B保32/128专家，MATH500 REAP76.07、OptEP(w)83.47，未加权反为88.40；加EWU87.40，不能称加权或恢复在每格都最好。保64/128时95.47/96.13、原模型96.40。**建议复现域专用压缩**，保留完整模型、REAP、纯选择与EWU分组，另测OOD与真实加载/服务性能；删专家数量不直接证明吞吐同比提高，逐层优化是额外一次性成本。


<a name="posttrain"></a>

## 后训练与蒸馏

### DPO随机图剪枝：抽一条反向分支可省约三成时间，删位置/截前缀则会改变目标

**Stochastic Pruning of the DPO Computation Graph；华南理工大学、深圳技术大学；2610.29538v1，10-21T10:25:15Z。** [§3–4、Tables1–6](https://arxiv.org/html/2610.29538v1)。以chosen/rejected平均token log-ratio为reward，均匀抽一侧留反向、另一侧no_grad，按1/q补权，**条件于所选位置/前缀的期望梯度无偏**。随机位置只省词表head投影；非线性logistic引入Jensen gap。随机前缀更额外偏向早token，所以三种不能统称“原DPO无偏加速”；其baseline本身也是length-normalized配方。

主线8×Ascend910B32GiB、LoRA64，Qwen3-4B与Mistral-7B；两个Qwen3-4B seed及一个Mistral seed的Full–Branch wallclock省29.9–32.7%，七accuracy均分降.25–.45pp。核心4B187.8→128.4min，58.33→58.08；生成judge均分上升属单seed描述。Position显存收益93.3%来自**只对valid response计算head但保全部监督**，删除监督只占6.7%；需先实现这条无目标改变路径。建议优先复现selected-head与单branch，按真实resource/质量约束选配置，别把拼批不同profiler的时间和显存串算。

### Flat-RLVR：周期SAM改善多采样覆盖，单次accuracy并不总保住

**Flat-RLVR；UNIST；2610.29281v1，10-21T06:35:19Z。** [§5–6、Tables2–8](https://arxiv.org/html/2610.29281v1)。在group-relative RL每n步加SAM最坏扰动后的下降，按题难度调半径、两次梯度均做centralization，试图缓和对高概率正确路径的进一步集中。它保留rollout上的group advantage，未训练新的verifier。

Qwen2.5Math1.5B、Llama1B、MATH7.5K，另3B/code；Qwen用5×RTX8000，扩展用单A100。三seed表中Qwen GRPO平均Pass@1/256为15.6/64.8，Flat15.2/67.3；code56.7/85.6→54.5/89.6，说明**多样性收益换过单次损失**。n5/10总时间17.00/16.87h对16.91h，仅当前rollout主导配方；平均allocated百分比不是峰值GB。建议小规模复现同时报pass@1、pass@K、采样token预算与每步训练成本；不要把256次覆盖直接解读为单次更可靠或大模型可扩展定理。

### TeachBack：让学生反教privileged角色可改变后续教学轨迹，来源标签并不保证正确

**TeachBack；华中科技大学、清华大学、哈尔滨工业大学、华为；2610.29167v1，10-21T03:59:23Z。** [§3–4、AppC](https://arxiv.org/html/2610.29167v1)。同一共享参数模型两种prompt：学生只看题，teacher看reference后50%作为可验证working notes。双方各生成、两prompt交叉评分；学生侧偏好teacher响应，teacher侧反向偏好学生响应，再共同更新。**偏好按响应来源定，不由verifier过滤，teacher反向偏好亦非宣称学生更准确。**

Qwen1.7/4/8B、DeepMath分难度及KK，8×NPU、300steps；4B数学mean@12 Base75.4/OPSD76.0/Teach77.6，KK greedy61.8/61.3/65.7。AppC训练budget不统一：Teach4rollouts、4K cap，OPSD1rollout/1K，GRPO8/16K；不能把差异全归双向机制。λ消融是另一个8K实验，非主表配方。建议观察/复现matched token/compute、同prompt与冻结teacher控制，增加错误传播、长度和正确率面板；未给出跨随机训练seed置信保证。

### OnQ：量化学生自己的错误前缀需要teacher纠偏，Dynamic KL收益应与整套校准分开

**OnQ；密歇根大学、UCSB、Meta AI；2610.29378v1，10-21T08:03:24Z。** [§3–6、AppA](https://arxiv.org/html/2610.29378v1)。先逐block重构恢复可用2/3bit策略，再由量化forward自产轨迹、原BF16模型在相同前缀打分。已有candidate-wise Dynamic KL按teacher/student概率比混合FKL/RKL；训练更新latent权重与scale，zero point在OPD固定，响应cap从4K升8K/16K。

Qwen1.7/4/8B、group128/BF16activation、25.6K多域prompt、H100/H200/verl；4B2bit九项均分校准42.99→OnQ64.95，最强所列ReasoningQAT53.31，BF16仍79.93；LCB28.34→46.64。主表±反映3个**evaluation seeds**，不是3次量化训练；baseline数据、步数、部分输出cap不同。等token固定response DKL→onpolicy对照还换了prompt池，因此不完全隔离onpolicy。建议复现时对齐数据/总teacher cost并独立消融DKL；论文未给低bit部署kernel吞吐保证。

### Harness Scaling：训练时覆盖执行外壳变化能提升迁移，更多组合并非单调更好

**Harness Scaling；IQuest Research、北航、中国人民大学；2610.29458v1，10-21T09:15:50Z。** [§3–5、Tables1–4](https://arxiv.org/html/2610.29458v1)。把system/tool/observation/context/workflow各设3变体，243组合中用二阶覆盖结构选18训练harness，每task派2个相邻配置做多harness RL，保持任务语义。测试另18未见组合及外部Claude Code/OpenClaw；基于构造空间的泛化不同于任意工具/任意真实环境保证。

Qwen4B/30B-A3B、ClawGym13.5K任务、300steps、16/32GPU；GRPO六benchmark×harness平均29.81→39.04、37.29→42.78。Table5基准harness59.82→59.63、held-out48.12→58.60、Claude46.02→49.62。三次独立**evaluation**并非三训练seed；243全空间试验含测试配置，只是budget stress-test。建议深读训练coverage设计，复现对齐rollout总量并保Random18/RandomAssign/K1控制，别以训练harness数单独预测收益。

### DeepSpeed 词表并行损失：把内存节省路径的 eager 多 kernel 改成流式 Triton

**deepspeedai/DeepSpeed；本窗默认分支增量。** [#8838 / 0b0d4081](https://github.com/deepspeedai/DeepSpeed/commit/0b0d4081a00971585b693b0b4fbaecfaec70d824)（2026-10-21T16:05:12Z，master）。

上期词表分片 logprob/entropy/KL 虽避免完整词表 FP32 中间量，每个 row chunk 仍产生多次 eager 调用。本轮 `vocab_parallel_softmax.py` 按行分块扫本地词表，以在线 max、sum-exp 和中心化加权统计融合 entropy/KL 前向及反向；通信仍由 `sequence/vocab_parallel_ops.py` 管理，MAX/SUM all-reduce 没有被 kernel 消除。最终 `_triton_rows` 可直接处理两级 stride 的序列切片，其他布局先 contiguous，不能沿用初始正文“只有 contiguous 才走 Triton”的限制。

作者最终 B70 单卡、BF16、每 shard 75,968 列、8,192 行、前后向20次中位数：reverse/forward KL 从首次实现18.7ms到14.6ms，entropy9.2→7.1ms；额外峰值（不含输入/梯度）<1MiB。最初 master340.8→15.5ms 的表来自较早修订，不能与最终14.6ms混成一组 TP2 端到端收益。CPU eager/FP64参照、TP1/2、全-inf行和支持集不匹配都有测试；后者即使概率下溢仍保留无限KL。entropy/KL求和顺序变化，非逐位相同；本轮未独立跑硬件。**建议考虑复现**：RL/OPD 大词表损失有明确收益点，但应连同 collectives、logits布局和 loss masking 测实际 step，而非引用本地kernel倍率。


### TRL：MoE LoRA 不再静默漏训 routed experts，参考模型和 FSDP2 也收紧契约

**huggingface/trl；本窗默认分支增量。** [#7769 / 63d9074e](https://github.com/huggingface/trl/commit/63d9074e87f6c20bc49bc20a42387692afe225a9)（2026-10-21T12:50:43Z，main）；[#7808 / 816f6586](https://github.com/huggingface/trl/commit/816f6586a4b13cea1e9e7e40ebfb1e43f037ff15)（2026-10-21T13:47:03Z，main）；[#7797 / c304acf2](https://github.com/huggingface/trl/commit/c304acf2e550d508686b9a8da2734ca4cab09d23)（2026-10-21T13:16:20Z，main）。

三项是训练语义与依赖边界修复。#7769在 Transformers≥5、PEFT<0.19、模型含 `.experts.` 三维融合参数、LoRA目标含MLP或all-linear且没显式 `target_parameters` 时启动即报错；attention-only LoRA与显式参数方案保留。不能由“LoRA模块已插入”推定专家权重参与训练。作者本地SFT/DPO/Reward版本矩阵验证，CI未含这个旧PEFT组合。

#7808在DPO/KTO拒绝把 `PeftModel` 当 `ref_model`；遵循新接口时把适配器policy作为model并省略ref_model，让trainer按初始policy建立参照。#7797将 Accelerate 下限升至1.11.0，并让FSDP2始终调用 `get_parameters_from_modules` 传 ignored_params，删除旧版 `None` 降级；PR还说明旧autocast异常退出会泄漏BF16状态，但本轮只代码核验TRL依赖/参数处理，未重现上游异常。**建议优先采用兼容性检查并审训练参数清单**；若现有recipe依赖旧ref_model/依赖版本，需迁移，不应仅改安装号后继续假定训练等价。


### TRL Online DPO / GOLD：阻止 VLM completion 自行生成图像占位符

**huggingface/trl；本窗默认分支增量。** [#7708 / fa7b0332](https://github.com/huggingface/trl/commit/fa7b033271a24a82e5b99421f49b71fd2591c57b)（2026-10-21T12:29:22Z，main）；[#7699 / fa158d83](https://github.com/huggingface/trl/commit/fa158d8341dbe3eebd744eb0f6732b6b4b0c06f1)（2026-10-21T12:28:56Z，main）。

Transformer生成路径会把图像特征再次传给 prompt+completion 前向；模型若采到额外图像token，token数就超过feature数并导致训练中断。最终代码在ProcessorMixin路径的GenerationConfig抑制 `model.config.image_token_id`，测试特意使用processor与config图像ID不同的Gemma3小模型。Online DPO的自定义generation_kwargs在默认抑制项之后合并，因此可覆盖默认设置；不能将默认防护称为不可绕过的强约束。Online DPO的vLLM路径不受该问题的同一feature传递机制影响；本轮没有新训练质量结论。**建议相关多模态在线训练做边界回归**，把生成配置与数据中的占位符契约一并固定。


### R2S：教师能做完任务，不代表其局部token评分能教会学生及时行动

**R2S；NVIDIA；2610.28586v1，10-20T17:40:08Z；10-21公告。** [§3–5及Limitations](https://arxiv.org/html/2610.28586v1)。先从学生高分歧状态采8次teacher takeover，按可观察结果共识或judge选短完成轨迹；轨迹只给teacher生成privileged评分，再全参微调teacher让其闭卷重现该评分。冻结修好的teacher后照常OPD，学生始终不看takeover。主方案不用task reward，**仍使用筛选器/模拟环境且需可重放状态**。

Qwen3.5-9B→2/4B及Qwen3-MoE30B→4/8B，ALFWorld/WebShop/ScienceWorld，150steps；Table2 9B→4B成功率vanilla/R2S为65.8/81.9、34.6/55.9、34.6/47.8。表中±主要是同一最终checkpoint三次decode，非三训练seed；另有3次学生训练曲线共用同一个修复teacher。每cell最多2400×8=19200 takeover、筛选与teacher一epoch成本在前，不能称免费。**建议深读**rollout能力与scorer质量分离诊断，复现须计全部上游成本及重放可行性，避免把一组Qwen文本模拟任务外推到真实开放环境。

### Projected Distillation：把失败轨迹改成“离学生近的正确答案”，比直接灌gold更有利保留能力

**Projected Distillation；AWS AI Labs、University of Massachusetts；2610.27467v1，10-20T01:38:01Z；10-21公告、回溯补录。** [§3–4、Tables1–4](https://arxiv.org/html/2610.27467v1)。在base连续16次都答错的训练题上，teacher按最小修改修正学生当轮失败响应，verifier通过后以改动span上的对比损失加正确响应self-SFT更新，每轮刷新target。数学定义为正确集合内最大似然投影，实际仅近似搜索，**并不证明找到了全局最近答案**。

Qwen3-8B/OLMo2-7B、math/code/tools/science，主结果3训练seed，等更新与每步生成预算；Table1黑盒projector的Qwen GSM8K AUPF .262，OPD .145，但Qwen MATH .035远低SODA .159；AUPF是特化—保留前沿面积，不能当accuracy。§4.6 privileged自修正hard-set .219、held-out .924；强黑盒.560/.916，gold .489/.744。**建议复现**稀疏正确奖励的领域适配，记录verifier失败、修改量、在线刷新与所有teacher调用；收益依赖正确性检查，不能声称任意任务无遗忘。

### A2D：学习能引出兼容分歧的问题分布，收益不等于“越难或KL越大越好”

**A2D；UIUC、香港科技大学、Texas A&M；2610.27450v1，10-20T01:00:17Z；10-21公告。** [§2–3](https://arxiv.org/html/2610.27450v1)。从学生初始化独立问题生成器，冻结评分学生/teacher，以候选union上的分歧×teacher在学生Top-K内的概率质量作token代理信号，加长度/批内多样性控制，用GRPO训问句；再格式过滤、词汇/embedding去重，形成无参考答案的prompt bank。下游是candidate-restricted PPO样式OPD，**不等于精确全词表reverse-KL**。

三次独立run、A10080GB评估；Table2同4B学生/4B teacher六code项，Vanilla34.5→A2D39.6；换30B-A3B teacher30.2→35.7，仍远低teacher50.0。τ²-Bench三域均分40.4→41.9、teacher43.3。§3.9训练token曲线未计生成/评分问句成本；§3.10两额外轮仅再+.44pp，且按best observed checkpoint，属乐观比较。**建议小规模复现**目标student–teacher pair，固定总tokens、候选支持和筛选成本，保留随机/学生自产/KL筛问句基线；代理信号并非学习增益保证。

### OPTD：教层间预测变化比绝对隐态更贴近“怎么修正”，平均增益小且有跨域退步

**OPTD；香港理工大学、华为；2610.28048v1，10-20T11:37:31Z；10-21公告。** [§3–6及Tables1–5](https://arxiv.org/html/2610.28048v1)。沿学生on-policy响应读多个层的词表预测，保留最终层OPD并加teacher/student的层间logprob增量信号。实际更新的detach/归一化与精确transition-KL梯度不同，论文在AppB4区分；不是简单全层hidden-state对齐。

R1-Distill1.5/7B、Qwen3-4B，DAPO/DeepCoder及三域Rubrics-To-Tokens，50–150updates。Table1 7B双域OPD overall49.14→Late50.06；4B33.77→34.68但coding均分18.13→17.35。Table2三域1.5B37.89→39.57，code16.51→16.43仍略降。Avg@16、pass@k、CodeElo另列，不应混用。**建议观察/复现**层锚点、span与规范化，特别保留LayerOPD及各域面板；收益不足1–2pp的场景应补训练seed和全部额外中间读出成本。

### CA-MOPD：已会的域可降低teacher压力，但“能力差”控制不提供免遗忘或自动省算力

**CA-MOPD；清华大学；2610.26950v2，首投10-18T09:09:06Z、v2为10-21T07:30:12Z；本轮首次收录，旧成果补录。** [v2全文§3–6](https://arxiv.org/html/2610.26950v2)、[v1 PDF](https://arxiv.org/pdf/2610.26950v1)。Task Arithmetic先合并同MixSFT的三RL专家，再以held-out teacher能力与学生rollout verifier EMA差、历史最好回退幅度调域权，仍做token-share平衡。**本轮逐页文本提取对比v1/v2，实际增量只有作者Le Gui→Le Gai及版本标记；性能不是21日新增。**

SmolLM3-3B、2×RTX PRO5000 72GB、200steps。Table1六项均分Open-MOPD31.39、merge31.48、CA32.22；code两个LCB均略低Open-MOPD。Table3同MixSFT、step50只换headroom信号，capability31.38 vs discrepancy31.09，说明主表+.83不可全归controller。能力反馈33.3s/step、3.20%；未跳teacher forward。**建议观察/复现**同初始化消融和不确定度，噪声/校准分布、已知域路由及单3B家族限制外推。

### DPO调参研究：小模型与validation loss都可能选错配置，应直接在目标规模短跑筛选

**The Shape of the DPO Hyperparameter Landscape；ELLIS Institute Tübingen、Prior Labs、distil labs；2610.27895v1，10-20T09:14:22Z；10-21公告、回溯补录。** [§3–6](https://arxiv.org/html/2610.27895v1)。Qwen3 .6/1.7/4/8/14B先Dolci-SFT再DPO，共享3×3 LR/β core并按规模扩格；11下游项分静态能力与chat，逐项标准化后再平均。最小DPO validation loss对三aggregate平均归一化regret .50/.62/.53，且固定β后相关性仍不稳定；**chat排名更一致不等于judge给出客观真值**。

§4.3在600step（约30%）中间/最终rankρ .85/.90/.85/.76/.63；4B短跑选中零regret配置，直接转1.7B配置反为最大regret1。换UltraFeedback后8/14B在共享格内Dolci配置亦最大regret。各DPO配置仅**一训练seed**，25次HPO种子为离线surrogate搜索重复；不能写多seed实训证明BO优越。**建议深读并调整调参流程**：目标模型短跑、多能力与chat并列、另留最终验证；这些LR/β不是跨数据集默认值，边界赢家也未定位连续空间最优。


<a name="inference"></a>

## 推理优化与训练系统

### RISK-KV：重建历史KV可以省迁移字节，但predictor会吃掉单请求净内存

**RISK-KV；哈尔滨工业大学；2610.29544v1，10-21T10:30:46Z。** [§III–V、TablesI–III](https://arxiv.org/html/2610.29544v1)。保两侧层完整exact KV，以ridge预测目标层归一化attention输入、原冻结投影生成KV，校准按**实际替换后NLL风险**选组。逐组风险排序后只测组合prefix，捕捉非加性但没有全局最优保证；bootstrap上界门控可退回exact。

Qwen.5/1.5B/TinyLlama1.1B、WikiText2、512history+512teacher-force，24requests；raw KV省18.2–33.3%、ΔNLL区间跨零，不是长生成准确率保证。Qwen.5B单请求省2MiB，选定FP32 predictor约8MiB，**需多请求共用才可能净省**。RTX3060 Laptop batch1、predictor已热驻、局部restore对token replay1.51–3.31×；100Mbps节省167.8ms只是字节/速率计算，未实测无线迁移。建议先复现真实context/请求数的净state和恢复边界，再考虑分布式集成；decode仍重建完整cache驻GPU。

### 长向量FlashAttention：把vector轴转向query能去softmax横向归约，收益只对长query块成立

**Reorienting the Vector Axis；Barcelona Supercomputing Center、UPC；2610.29489v1，10-21T09:45:14Z。** [§III–VIII、TablesV–VII](https://arxiv.org/html/2610.29489v1)。从PyTorch CPU FA2 port到RVV，区分score product的跨步访存和value product的短向量。已有packing在全部方案保留；新方案将tile转向query轴，softmax max/sum改为逐元素纵向累积、向量状态驻寄存器，代价是score两遍读取。

EPAC16,384bit向量/512FP32、单core/head、d64/N1024，实chip stock51.94Mcyc→28.22（**1.84×**），对已fused为1.43×；FPGA2.71×来自不同带宽/周期条件，不能拿来代替硅片。N4096实测2.08×，d增大收益缩小；FP32是该硬件最窄dtype。**autoregressive decode query仅一行，论文明确自然方向才合适**。建议深读跨架构布局而非套GPUFA数字，复现要保packing、tile各自调优与numerical误差，尚无完整LLM端到端训练吞吐。

### TensorRT-LLM Mooncake Store 恢复零拷贝默认值，并补页复用前的 NIC 写入栅栏

**NVIDIA/TensorRT-LLM；本窗默认分支增量。** [#20108 / 69e06b8a](https://github.com/NVIDIA/TensorRT-LLM/commit/69e06b8afbffc4726a63b43fb4acb9fda340f3a7)（2026-10-21T14:38:57Z，main）。

KVCM V2把多个cuMemCreate物理分配映射成连续虚拟池；把整池一次交给RDMA注册会跨分配边界，导致peermem的EFAULT或dma-buf的EINVAL。`addressing.py::allocation_spans`依cuMemGetAddressRange切注册及传输范围，注册中途失败撤销已成功片段，worker退出逐片注销；driver lookup无法解析时将该段标unresolved、警告后仍整段尝试注册，跨分配注册仍可能失败并回滚，并非对所有driver都保证拆分成功。默认 `stage_through_host=False`，可显式True退回host staging；这会改变部署前提。`WITH_NVIDIA_PEERMEM=0`选择dma-buf；另一条路径需要模块和gpuDirectRDMACapable分配，进程首个client锁定路径，后续冲突拒绝。

更关键的完成边界是NIC不受普通CUDA stream排序：connector绑定KVCM的allocation fence，在向请求新页发起零拷贝load前等待旧GPU工作释放这些页；该栅栏不等同于RDMA完成。注册有效性还依赖pool映射保持稳定。已读addressing/worker/kv_cache_manager_v2及分配栅栏回归；PR给了真实pool、peermem/dma-buf及集成测试入口，本轮未运行。**建议分离式推理/分层KV使用者深入阅读并回归**，优先验证跨分配注册、忙页再分配、失败回滚和进程中多client配置；没有可直接泛化的吞吐倍率。


### TensorRT-LLM attention-DP：各 rank 必须同序 flush KV 事件，dummy角色读取队列内请求

**NVIDIA/TensorRT-LLM；本窗默认分支增量。** [#20314 / ec782566](https://github.com/NVIDIA/TensorRT-LLM/commit/ec782566f50ad9548e99818fa1d74c7c66cc572d)（2026-10-21T15:37:04Z，main）；[#20325 / afc1c168](https://github.com/NVIDIA/TensorRT-LLM/commit/afc1c168fd05c9b583823ae8d9b9079389d2a071)（2026-10-21T21:45:38Z，main）。

#20314把包含allgather的V2 buffered KV-event flush，从有条件执行的previous-batch helper移到overlap分支汇合处；即使本rank事件列表为空也必须参与。rebalance drain另补一次flush，避免修了正常循环却丢迁移路径。已读最终patch，包含真正逻辑与测试；PR末自动评语“只有import格式变化”与最终diff冲突，未采信。作者CPU71项/配对rank顺序测试为mock transport，不能写成真实多机压力证明。

#20325在路由后输入还是RequestQueueItem时，从 `candidate.request.request_type`识别P/D角色，替代对尚非LlmRequest对象读取字段；九类角色/混合顺序等单测，作者253项通过。**建议attention-DP/overlap使用者优先回归**：事件collective顺序和dummy角色都是服务不挂死的前提，无新吞吐主张。


### TensorRT-LLM MiniMax-M3：长 prefix 索引并行与保持 FP32 路由语义的 BF16 分解

**NVIDIA/TensorRT-LLM；本窗默认分支增量。** [#19975 / 0c9b463e](https://github.com/NVIDIA/TensorRT-LLM/commit/0c9b463e3411c6a5e3961309954b6571ff6efb09)（2026-10-22T00:03:09Z，main）；[#20699 / 522daa8c](https://github.com/NVIDIA/TensorRT-LLM/commit/522daa8cdf4183c4f0ba94fe0401555974a54a27)（2026-10-21T16:57:05Z，main）。

#19975将MSA proxy scoring除了query行外也沿cached prefix分片，最多16个KV切片；仅cached≥4096且新token>64时启用，并把query片裁成≤128行。这样约1K×131K评分不再只有4–5个CTA长扫prefix。B300 AgentX、forced Eagle3 acceptance、每arm单次3600秒：TP2/C45/K3的p50/p90 TTFT分别降15.6%/13.9%，tok/s/GPU+2.1%；交互指标噪声±3–4%，不是统一吞吐+16%。作者38prompt greedy token相同，自动split选择和fallback仍值得按实际shape验证。

#20699针对FP32 router被环境TF32 override改算及SIMT开销：load时将FP32权重拆成三个BF16项，BF16输入沿K重复后用一次tensor-core GEMM、FP32累加；其他输入用明确CUBLAS_COMPUTE_32F。GB300 batch96 router22.4→12.0µs，FP64参照测试中8192行/shape无top4翻转，而TF32有0.06%–0.16%行翻转。不能称任意输入严格FP32逐位等价或全模型1.87×；固定模型、输入dtype、权重重载及graph地址都重要。**建议考虑复现两项**，分别测prefix长度/调度shape与近并列专家路由，勿把两个局部收益相乘。


### TensorRT-LLM GLM-5.3-Flash：FP8 KV 直接稀疏 MLA、融合写缓存并去掉 host 同步

**NVIDIA/TensorRT-LLM；本窗默认分支增量。** [#20454 / ef913d39](https://github.com/NVIDIA/TensorRT-LLM/commit/ef913d39b3b5ba488dc0972bd400e52b41899767)（2026-10-21T05:59:14Z，main）。

旧glm_kpool先gather选中FP8行并解成BF16再跑FlashMLA；新路径在每rank16/32/64个query heads时对context和generation直接调用TRTLLM-GEN sparse MLA。query从BF16饱和裁到±448再转E4M3，scale进入BMM1/BMM2；indexer仍BF16、选中行不变，但attention数值精度已改变，不能称纯无损调度优化。其他heads（如TP8）保留fallback。

同时一次Triton写入latent+packed index state，把draft位置按顺序合并kpool更新，并用pinned异步H2D构造context rows；workspace在profiling前预留计入KV预算。作者NVFP4 checkpoint/FP8 KV、4×B200 TP4/EP4、ISL13312/OSL33、MTP3、max_num_tokens16384、三次中位数：C1 TTFT558→268ms/ITL3.20→2.87ms，C8总吞吐60.6→129.1tok/s。此组是三项叠加且只限该配置。已读native_decode饱和转换和kernel guard（二维position与request_ids组合拒绝、写入宽度须2幂）。**建议复现并同时评测质量**，不能以同top-k索引替代最终输出准确率验证。


### TensorRT-LLM DSA indexer 默认 CuTe DSL：移植 kernel 需同时带编译器兼容和数值边界

**NVIDIA/TensorRT-LLM；本窗默认分支增量。** [#19976 / 033f2ea0](https://github.com/NVIDIA/TensorRT-LLM/commit/033f2ea0205341b2f3ad55bbeb5593427030ec78)（2026-10-21T09:41:44Z，main）。

在SM100-family将paged MQA logits默认切到CuTe DSL，并统一metadata和indexer选择门；按步移植DKG的next_n分片、架构TMEM、stage与scheduler改进。直接替换DKG文件在固定cutlass-dsl4.8.0.dev0上可出现长编译/不兼容API，因此最终有选择地移植而非字节复制。

作者GB200 cold-L2/8GB flush、batch1/32/128、4K–128K多shape，FP8/FP4相对旧main的中位kernel倍率约1.051×/1.055×；不同rebase/GB300和next_n=7后续表需分开，不能只取最大值当整个模型提升。默认ReLU恒等变换在极大有限输入x≥2^127会溢出，`use_relu_trick=False`关闭但有代价；公开API没有把scale/weight限到安全范围。真实DeepSeek-V4-Flash、4×GB200、MTP3检查在13,776个选top-k行有相同top512，不覆盖所有规模/精度。**建议深入阅读兼容与数值取舍**，采用时钉dsl版本、next_n、输入范围和完整模型回归。


### TensorRT-LLM 多模态按 prefill chunk 编码：容量与复用变化先于吞吐结论

**NVIDIA/TensorRT-LLM；本窗默认分支增量。** [#19884 / b47eb8fe](https://github.com/NVIDIA/TensorRT-LLM/commit/b47eb8fe6b0965a85adf01121f9f0f2a1e79c7c2)（2026-10-21T02:04:28Z，main）。

KVCM V2+chunked prefill下，从真实prefix reuse/recompute之后的prompt window决定本轮需要的图片/视频items，所需items ready即推进LLM，已消费encoder输出释放。请求的全部encoder输出可超过cache容量，但同时必需的一个chunk单元必须装得下；共享entry按引用计账，reclaim失败不误删别的请求输出。V1继续全请求编码。DEFAULT能整体装入时仍整请求batch；EAGER可利用剩余预算预做future items。

这只交错encoder与LLM步骤，**没有新增两者CUDA kernel并发**。item scheduling要求PP=1，EAGER排除attention-DP及disaggregated，side-stream/EPD也有边界。PR的最高37.41%吞吐提升来自旧scope/rebase前Qwen3-VL-8B-FP8、2×B200、prefix-hit C8测试，包含跳过全部64张图编码；**最终V2-only head未重跑端到端性能**。作者977CPU/4000随机调度和6个H200实际allocator案例验证计账，最后一组模拟prefill结束、没跑LLM forward。**建议考虑复现容量/复用机制，性能继续观察**，不能把旧表写成最终代码已证实收益。


### FlashInfer Frost 小 token MoE：显式候选，局部低延迟而非分布式 EP 加速

**flashinfer-ai/flashinfer；本窗默认分支增量。** [#6626 / 2995461e](https://github.com/flashinfer-ai/flashinfer/commit/2995461e872e030b364a1a84d8ca94181f857876)（2026-10-21T18:40:04Z，main）。

新增 `frost_low_latency` 统一MoELayer后端，支持SM100/103/107，MXFP4或MXFP8权重+MXFP8激活/SwiGLU及BF16多种激活；输出BF16、路由预计算。FMA路径融合FC1激活与FC2 combine，TC路径有PDL但仍多个kernel。默认准入T≤2，可显式配置到8；**不加入默认backend列表，runner明确不支持expert parallelism**，需要cutlass-dsl≥4.8等约束。

作者B300固定tactic、warm/disjoint专家访问、H2048/I512/E512/K10、SwiGLU、T1：最佳受测旧backend27.736→15.460µs（1.794×），T2为1.385×；这些是split前记录修订的完整本地routed-MoE GPU调用，不含通信、不等于默认自动选择或模型tok/s。BF16和MX cohort不可混比。**建议单请求/小batch解码考虑复现**，保留原后端候选及真实autotune，把权重准备成本、cache温度和路由复用一起计入。


### FlashInfer MoE 通信收口与 MegaMoE 状态清理：迁移接口同时保住后续 launch

**flashinfer-ai/flashinfer；本窗默认分支增量。** [#6621 / 9d3837c4](https://github.com/flashinfer-ai/flashinfer/commit/9d3837c44bf364be84d4ea3a3a8635f68bfbb4ab)（2026-10-21T03:19:11Z，main）；[#6517 / da992ed9](https://github.com/flashinfer-ai/flashinfer/commit/da992ed9919604c5eb6dc255c63f5e89a466452c)（2026-10-21T07:09:48Z，main）。

#6621把Cake/two-sided all-to-all移入各自moe_ep backend，删除 `flashinfer.comm.moe_a2a_*` 与旧 `backend="trtllm"` 功能层/benchmark，通信bench转MoEEpCommunication；旧bench的每token额外payload并未迁移成新接口能力。one-sided模块对SM107使用native编译，修复继承sm_100f导致首次launch失败，作者8×GR100/EP4跑通fence/CFT。这是实际接口迁移与硬件适配，非单纯目录重命名。

#6517修复SM90 inactive dispatch warp清理workspace前没取expert counts，热专家高pool块留脏，影响下一次launch/graph capture；另按实测为H200/EP8小M启push，约省10µs、局部1.03–1.16×。新增prewarm/commit_staged供框架在graph capture前准备，权重必须先处理Marlin位序及scale范围（NVFP4 6×scale≤448），不会替框架自动修checkpoint。作者8×H20073项通过，其他geometry编译覆盖不等于全模型测过。**建议现有MoE调用者优先读迁移与权重契约，复现连续多次launch/graph场景**。


### FlashInfer MNNVL CuTe DSL 融合 all-reduce：单 launch 与动态 FP8 epilogue显式启用

**flashinfer-ai/flashinfer；本窗默认分支增量。** [#6603 / c952a5bc](https://github.com/flashinfer-ai/flashinfer/commit/c952a5bcffde71768bdb1969116ee1fa5a8ecf99)（2026-10-21T22:19:39Z，main）。

原BT发布→owner reduce→materialize三launch可选persistent单launch，用全局顺序ticket与负零Lamport readiness推进，融合residual/RMSNorm/动态每token FP8输出；不是靠grid barrier保证进度。统一API新增 `backend="mnnvl-cutedsl"`，**auto仍不选它**。编译时norm/量化配置固定，TP4/H8192等profile按10.3/10.7架构分别匹配；本地支持probe需跨rank先达成一致再构造，不能各rank独立分叉。

已读allreduce workspace契约、静态route说明与4rank实际接口回归入口；动态量化scale=max(amax/448,min-subnormal/448)，有FP32舍入差别。没有本轮独立跑通信或可泛化端到端倍率。**建议通信占比高且shape匹配的服务考虑复现**，尤其最少resident CTA进度、非默认eps/残差一致性及graph重复调用。


### FlashInfer QSA NVFP4 sparse attention：SM80–90实验路径以算力换 KV 容量

**flashinfer-ai/flashinfer；本窗默认分支增量。** [#6804 / 53f175d7](https://github.com/flashinfer-ai/flashinfer/commit/53f175d7d74319e47131424547cb9427c0297c20)（2026-10-21T23:00:06Z，main）。

实验包从与现有slot-mapping writer兼容的NVFP4 K/V及block scales直接解码到mma片段；split-K计算与LSE merge合并sigmoid gate，BF16乘法/FP32累加。当前BF16 query、head_dim256、SM80–90等wrapper guard均需满足。每个K或V head/token由FP8的256B变144B，但不是整个服务显存按同率下降，也没有等质量压缩保证。

作者SM80/250W、24Q/2KV heads、每row从8K选2051token、graph replay、不含KV写：1row为14.5µs vsFP8 Triton13.2µs（更慢），16row47.4 vs58.5µs，128row304.2 vs336.7µs。需显式实验入口，未自动接成vLLM默认QSA能力。**建议容量受限场景考虑复现质量/时延交换**，小batch未必获益；先纳入cache写入再比端到端。


### FlashInfer NVFP4 writer：去 host 同步改变错误语义，4over6则增加量化计算

**flashinfer-ai/flashinfer；本窗默认分支增量。** [#6204 / 38214409](https://github.com/flashinfer-ai/flashinfer/commit/3821440929ee342fe9f939dbf5186c3e667a851c)（2026-10-21T16:05:11Z，main）；[#6013 / 52fbbd09](https://github.com/flashinfer-ai/flashinfer/commit/52fbbd090ca9c85b69c3a94420e360d27779b366)（2026-10-21T07:40:21Z，main）。

#6204去掉eager每次把CUDA k/v_scale复制到host检查的同步，保留kernel写前trap；CPU/float仍ValueError，坏CUDA scale改为异步CUDA错误且可能使context不可复用，空写不launch也不检查。作者SM80、Qwen3.6-35B-A3B NVFP4/TP1/eager/prefix caching，18,640复用+2,860重算token、20append，同prompt中位499.4→451.7ms（各12次）；未测SM12x、多GPU或prefill graph/compile，top5 logprob本来有运行波动，不能称严格逐位模型等价。

#6013显式opt-in每16值块同时试amax/6与amax/4，仅严格降低重构MSE才选后者；tie留6、存储格式不变，reader不改。仅MSE/no-fast-math，独立JIT module，不自动读取GEMM环境变量，禁止JIT且无缓存会报错。SM80/BF16/8KV/HD128/16,384token写入0.51→1.49ms，另首次构建31秒：它提升局部重构而非写入速度，也未给模型质量提升证明。**建议分别决定是否采用同步优化和更贵的量化recipe**，不要把两项合并宣传为无代价FP4优化。


### BitMTP：低比特并行草稿要连同KV状态机一起优化，朴素实现反慢于AR

**BitMTP；香港理工大学；2610.28500v1，10-20T16:42:17Z；10-21公告。** [§3–5、AppB–D](https://arxiv.org/html/2610.28500v1)。BitNet/Bonsai联合NTP+block-causal MTP训练，同模型草稿/AR验证；runtime按宽度、KV容量segment和数值角色捕获CUDA Graph，用共享物理KV维护已提交但未物化的尾部，再预备可能用到的verifier输入。**只有speculative-only路径保证该AR验证器greedy续写；confidence/hybrid直接接受并无同样保证。**

H20、D8五生成任务平均hybrid对匹配SFT-AR加速2.2/2.4/2.1×（BitNet2B/Bonsai1.7/4B）。BitNet质量45.95→43.25，不能概括全部近无损；Bonsai1.7B47.36→47.91。§4.4 generic runtime143/244TPS均低AR373，完整方案722/806TPS；收益不是仅训练MTP。吞吐只计prefill后decode；Orin最高3.3×仅MBPP+，MAXN锁频，有冷启动成本。**建议复现**低bit长期服务，测接收率、冷启、全请求延迟和非CUDA可迁移性，较大模型/长context未验证。

### SSA：保heavy hitters再按proxy抽剩余KV，比纯top-k更稳；理论与GPU版本须分清

**SSA；Rice、UC Berkeley、Stanford、IIT Bombay；2610.28441v1，10-20T16:02:12Z；10-21公告。** [§4–5及Tables1–3](https://arxiv.org/html/2610.28441v1)。proxy质量决定确定性heavy集和残余重要性采样，对去重后样本按包含概率纠偏。GPU实现改为固定heavy比例、PQ16byte/key与分层采样，并把极小包含概率截至1e-4，**这引入偏差；不能把理论自适应分配/无截断保证原封套给kernel**。

Table1 50×稀疏、Llama3.1-8B RULER32K：PQ72.51、SSA83.67、dense88.76；128K为39.73/47.53/71.26，仍大幅损失。Qwen3.5混合架构对应点更稳，不能跨模型混成通用无损50×。Table2单H200 kernel计proxy/选择/采样/sparse decode后，对FlashInfer在1M context、batch1的50×仅1.79×，batch16为2.96×；是attention kernel加速。**建议复现**长context decode，保dense、oracle-top-k和纯PQ，测多seed质量及实用clamp/重采样，别把KV访问稀疏倍数当端到端倍数。

### TeCoP：未提交但稳定的预测可先当上下文，真正提交仍要规则把关

**TeCoP；台湾大学；2610.27583v1，10-20T03:48:01Z；10-21公告。** [§4–5及Tables2–6](https://arxiv.org/html/2610.27583v1)。训练免费dLLM decoder让多个block重叠推进：前沿已提交达到α才开新block；tentative预测放独立buffer作为下轮context，只有高置信或跨轮同token且达验证阈值才提交，fallback确保进展。**这是模型自身一致性验证，不是外部AR拒绝采样，亦无分布精确保证。**

单H200、batch1、默认256输出，LLaDA/Dream四模型×3任务；cached总体geomean **6.65×对uncached Original**、均分−.98pp。更直接相同缓存族比较对Fast-dLLM+DualCache为1.26–1.58×；各方法cache不同。Table4 pipeline K1→2使GSM8K78.2→76.7、61→71.7TPS，K4降74.3，存在明确质量取舍。**建议复现**真实输出长度/高熵任务，固定cache基线、阈值和有效非EOS token口径；论文尚未验证长文与开放生成。

### DStill：离线多步teacher轨迹能改善单步diffusion草稿，服务额外成本转移到了训练

**DStill；Capital One AI Foundations（原始PDF首页）；2610.27379v1，10-19T23:05:16Z；10-21公告、回溯补录。** [原始PDF§3–4、Tables1–3，pp4–9](https://arxiv.org/pdf/2610.27379v1)。先训练可补部分可见block的target-conditioned drafter，再收8步refinement轨迹；学生从轨迹root/中间状态预测同endpoint，训练逐渐偏向全masked root。推理仍一次草稿+原target验证，保分布的是标准拒绝采样器，不是蒸馏本身。

Table1 concurrency1/T0：GPT-OSS120B平均AR speedup3.45→3.68×、Gemma4-31B4.53→4.91×、Gemma26B-A4B2.34→2.78×；最大6.45×是单task相对AR。同target两方案保持drafter架构/语料/block16/一次草稿，平均acceptance包含anchor。Table3 Nemotron高并发MT/SWE有**speedup<1**，即比AR慢，不能用最高点概括生产收益。额外partial-mask训练、轨迹生成/存储和蒸馏需摊销，学生也继承endpoint错误。**建议复现**目标并发的真实wallclock与训练摊销，不把SWE prompt生成速度当完整修复成功率。

### Token-Energy Law：把共享权重、私有KV字节和实际时间分开，可解释受控dense decode能耗

**Token-Energy Law；NVIDIA、Texas Tech、TACC；2610.28576v1，10-20T17:32:08Z；10-21公告。** [§3–6](https://arxiv.org/html/2610.28576v1)。以实际有效batch摊权重字节、按输出token加权的历史长度计KV字节，再加入实际decode秒/token，局部校准板卡能耗。**它需要观测到的时间，不是只看参数量即可先验预测能耗**；各拟合项也不是独立电表测出的组件能耗。

Qwen2.5-7/14B、Mistral7B，BF16 full-history，GH200/GB300固定power limit、vLLM，关prefix/spec/chunk/offload/preempt；270次held-out静态执行54cells全部平均误差<3.73%，另104次受控请求退场。更大batch有33.2–46.8%较低J/token，但改变图捕获策略须重新验证，8192上限外eager fallback影响时间。**建议采用测量口径并复现局部回归**，同步能量/时间/有效输出边界；不含CPU能耗，TP2需另拟合，MoE不能直接把active参数÷batch代入。

### RACE：整对保留RoPE维度才能把压缩折进原attention，质量恢复也计入配方

**RACE；MIT CSAIL、MIT-IBM Computing Research Lab；2610.28385v1，10-20T15:34:02Z；10-21公告。** [§3–4、AppA7](https://arxiv.org/html/2610.28385v1)。共同Q/K低维选择若要先于原RoPE执行，coordinate-selection必须整对保留旋转维度；按joint Q/K能量选pair、value另选子空间，折入投影，推理不重建全维。最小化的是校准相关可分上界，非任意attention误差全局最优。

Qwen2.5-7B/Llama3.2-3B/Granite4.1-3B；RAP/RACE比较均用1M FineWeb-Edu token KL恢复。75%KV retention保98.5/97.5/96.5%的原zero-shot**相对分数**，非下降相同pp。单H10080GB、256K、50%retention作者测prefill1.76×、decode1.90×、overall1.79×，总allocated仅−28.06%，不是全显存减半。与token剪枝/量化组合会叠加质量损失。**建议观察/复现**校准迁移、原频率pair保留与同恢复成本，勿将此当无训练、通用长context无损压缩。

### KPage：把部分空block的有效KV页跨块拼满大tile，效果依赖碎片而非普遍kernel优势

**KPage；KlingAI Research、NVIDIA；2610.28096v1，10-20T12:12:16Z；10-21公告。** [§3–4](https://arxiv.org/html/2610.28096v1)。保原稀疏graph/有效行语义，以Page16加载映射把多个逻辑block有效页拼入大tile，query按有效行走M64/M128；不在HBM物化整份压缩KV。这是video sparse-attention布局论文，**尚非LLM端到端验证**。

BF16、head128、单GPU/CUDA13.2/CUTLASS-DSL；§4.3无padding对BSA geometric mean仅1.03×，有20.8–22.8%padding四case为2.29×；开放视频布局对最快exact sparse baseline1.13–4.25×。计时排除可复用的graph/page table/route构建；BSA当前partial-block backward语义不同，论文明确相关训练时序只能作硬件参考。**建议深读kernel布局**，复现先对齐forward graph、有效行、DSL版本和metadata摊销，不把图内kernel数当训练或LLM服务同倍收益。

### AMTP：把草稿存活率预测转成长度控制，但实验关闭KV cache，暂适合方法观察

**AMTP；Georgia Tech；2610.28619v1，10-20T17:56:32Z；10-21公告。** [§3–6](https://arxiv.org/html/2610.28619v1)。预测逐位置拒绝hazard，累计存活率求每个horizon预期有效token/成本；控制器在草稿前决策，用少量全长探索和逆概率加权更新，保持原exact verifier。理论需同一因果proposal的前缀截断、成本界及预测器条件。

§6三self-spec与两cross-model、AIME/GSM8K，**全部无KV cache、每格单次无seed运行，跨模型硬件不同**。cross-model Meta7B→Llama13B对最佳fixed-window最高+268% TPS；self-spec多为2–7%，Meta7B AIME反降15%。因此最高4.1×AR不宜作为cached生产服务收益。**建议观察**hazard目标和部分反馈修正，工程复现须先加cache/并发、计控制开销、多seed并用逐horizon最优fixed基线。


<a name="correctness"></a>

## 工程正确性与兼容性

### Transformers 拒绝静默丢掉 attention softcap

**huggingface/transformers；本窗默认分支增量。** [#49811 / 7d3ae42e](https://github.com/huggingface/transformers/commit/7d3ae42ee810431d4df15dcbf3bc5bce70221cc5)（2026-10-21T14:57:24Z，main）。

Gemma2等模型需要attention-logit softcapping，而SDPA没有对应参数；部分Flash后端也不接受softcap。最终patch不是PR初稿所述warning：`sdpa_attention_forward`遇非空softcap抛ValueError，Flash kwargs处理也在后端不支持时抛错，并把Gemma2/T5Gemma/VaultGemma的 `_supports_sdpa` 置False。Gemma3、Gemma3n、Muse Glimmer显式保留SDPA支持，不能扩大成“Gemma全系禁用SDPA”。可选eager/flex或确实支持softcap的Flash实现。**建议优先检查采用的attention后端**：过去跑通不代表执行了模型所需函数；升级可能由无声数值偏差变成启动报错，没有端到端准确率提升测量。


### FlashAttention 未用 V 槽中的 NaN 不能靠 softmax 的零概率隔离

**Dao-AILab/flash-attention；本窗默认分支增量。** [#2983 / a6fcdfdb](https://github.com/Dao-AILab/flash-attention/commit/a6fcdfdba17bf5c643d9d2efa275222677399787)（2026-10-21T23:57:51Z，main）。

CuTe SM100/SM110非paged V cache按容量装载时，逻辑长度外的NaN或packed相邻序列的NaN仍可能进入PV乘积：0×NaN仍是NaN。修复以ragged TMA视图/按tile的真实边界阻止越界逻辑数据进入V装载，而不是要求用户每步清空整块cache。回归覆盖dense/varlen、causal、splitKV、GQA、strided/broadcast V及FP8，检查无效尾部替换为NaN后有效输出和LSE保持一致；测试覆盖不等于任意架构已验证。

**版本边界。** [FA4 beta36](https://github.com/Dao-AILab/flash-attention/releases/tag/fa4-v4.0.0.beta36)于10-21T08:19:23Z发布，tag指向 `e7a6402a9ff099de5eb628ce207a88a3f0aa5378`；本项于23:57:51Z才合入，**beta36不包含本修复**。beta36汇集此前已报的paged graph capture与opt-in hd256 backward等，不能把新版本号写成后续main修复已发版，也不称FA4稳定版。**建议先复现和回归NaN污染，再按固定修订评估采用**；未运行本地GPU测试，没有吞吐收益主张。


### DeepSpeed masked softmax：mask 后重新定中心，避免全指数下溢

**deepspeedai/DeepSpeed；本窗默认分支增量。** [#8782 / 67ec1961](https://github.com/deepspeedai/DeepSpeed/commit/67ec19616b97d5734b841996b944304e88ca9972)（2026-10-21T17:10:38Z，master）。

旧Triton路径先减原始row max再加mask；如果原最大项被屏蔽，其余logit可能全部极负，exp均下溢为0，最后0/0。新代码加mask后再减一次最大值，首次shift保留以维护FP32 mask精度；回归覆盖有限大负mask与-inf。修复针对“最大项被mask且仍有有效项”的情形，不能从中推导全mask行已定义合法概率分布。**建议相关服务纳入数值回归**，尤其自定义大负attention mask；没有新性能证据。


### AITER attention 三类修复：极负 logit 的 NaN、FP8 尾概率及 MTP 未初始化

**ROCm/aiter；本窗默认分支增量。** [#6428 / 51e846d8](https://github.com/ROCm/aiter/commit/51e846d88550fbaea3b20c37fa19113705f3590a)（2026-10-21T16:58:33Z，main）；[#6138 / 0766f73d](https://github.com/ROCm/aiter/commit/0766f73d8fc2a2b06adc23d3737f3121b967ed4f)（2026-10-21T17:14:45Z，main）；[#5642 / 45d7ef38](https://github.com/ROCm/aiter/commit/45d7ef38074ee12ad692e817f12b7acb418610c9)（2026-10-21T04:57:57Z，main）；[#6606 / 2f24aad3](https://github.com/ROCm/aiter/commit/2f24aad379b7545f47c71577228d4bc09283fae4)（2026-10-21T02:04:12Z，main）。

- **训练反向NaN（#6428）。** 非整块长度的尾部key按0装载，lse<-88时未mask的exp(-lse)溢出、inf×0污染dq；在两个实现的dq loop补key-range mask。作者MI300X/Qwen2.5-1.5B Async GRPO捕获张量dq384NaN→0，并对FP32参照；只是具体数值故障，不是RL算法修正。
- **FP8概率下溢（#6138/#5642）。** gfx942 Sage把P乘128后转FNUZ，输出归一化除回128；paged ragged路径对gfx942用128、其他OCP e4m3用256，并把exp_sum归一化后移到PV之后。避免近均匀P落入粗糙subnormal，不改变精确算术目标，但仍有FP8舍入，Sage ASM另案未由Triton改动自动修复。已读两套scale前后配对代码及回归；不混用FN/FNUZ最大值。
- **GQA16 MTP（#6606）。** 八个gfx950 code objects更新qlen3/4 running-max初始化；作者2592配对case由2327→2488通过，16非有限与171不稳定case降0，但**104个既有FP8精度阈值失败仍在**。本轮只能检查二进制OID/提交和作者验证记录，未反汇编证明SP3实现；混合batch qlen1和性能不在验证范围。

**建议相关训练/服务优先回归这些正确性场景**，尤其极端logit、长尾概率、首次调用与重复graph；“没有NaN”不能代替精度认证。


### XLA/cuDNN 与 Pallas SparseCore：提交异步操作不代表已读写完成

**openxla/xla；2026-10-21T13:24:30Z，默认分支 `497b11c2f71ffd5dc5cc4f96d2378e17e2bc5131`。** [固定提交](https://github.com/openxla/xla/commit/497b11c2f71ffd5dc5cc4f96d2378e17e2bc5131)。

cuDNN9会在heuristic/plan创建内部临时capture graph，不经过XLA常规capture handle；并行autotune另一线程的VMM析构可能cuCtxSynchronize，使capture失效或驱动崩溃。CudnnHandle生命周期进入/退出allocator capture计数，延迟释放直到操作完成，而非删掉同步保证。已读RAII和move-owner转移，无独立GPU重现。

同窗JAX [`1937fa70`](https://github.com/jax-ml/jax/commit/1937fa70027e81886ab9878e57a58482b94abee6)把SparseCore push stream仅定义为VMEM出发；SPMEM→SMEM不能按push语义提前返回，调用者应使用正确wait等待读写完成。两者是不同层/设备的问题，不能视作同一补丁的全系统完成保证。**建议遇到随机编译崩溃或自定义Pallas DMA者优先回归**，不主张吞吐收益。


### vLLM-Ascend：GDN按真实token压索引，PP/SP不能把完整payload误切为分片

**vllm-project/vllm-ascend；本窗默认分支增量。** [#19378 / 87f2b060](https://github.com/vllm-project/vllm-ascend/commit/87f2b060adff3079bd63f995c6c628820fec0f9d)（2026-10-21T17:05:01Z，main）；[#19347 / 1c29071a](https://github.com/vllm-project/vllm-ascend/commit/1c29071a167af050270d50f5344c484b642fe570)（2026-10-21T13:16:53Z，main）；[#19165 / 4f3648ab](https://github.com/vllm-project/vllm-ascend/commit/4f3648abd258a5cbdd1a9ae34165a85f5e44beaf)（2026-10-21T08:26:17Z，main）。

#19378对CANN recurrent GDN要求每q/k/v行恰有一个state index：1D按真实T裁、2D均匀走flatten，变长spec根据query_start_loc只保留每请求实际query_len。修复30真实token/32padding请求、零token请求和达到max_model_len而未填满draft的长度错配；不是把所有state slot都重置。

#19347修Model Runner V1的PP边界：DeepSeek/GLM即使内部MoE开SP，PP仍可能传完整token，不能按全局SP标志只保1/TP行。buffer统一按完整payload分配/复用，MRV1 transport保TP scatter/allgather恢复，MRV2选择另行保留；显存profiling相对旧错误分配会增加，但真实需求才被计入。#19165另在MRV2 PIECEWISE dummy中区分逻辑query数和padding输入宽度，延伸到hybrid/MTP元数据；不是把graph容量当真实递推token数。已读helper、PP收发分支和Mamba metadata/回归；真实多节点组合仍须自行验证。**建议Ascend混合模型与PP/PD使用者优先回归**边界长度、空rank、padding及最大上下文，检查token未被静默截断。


### Triton NVFP4 scale真正向上取整；小batched MMA累加器stride与TMEM同步补齐

**triton-lang/triton；本窗默认分支增量。** [#12401 / 3419b568](https://github.com/triton-lang/triton/commit/3419b56873c5eed1d4885087c909a038614a46ff)（2026-10-21T19:01:16Z，main）；[#12404 / 70613630](https://github.com/triton-lang/triton/commit/70613630e05597634a6d81c41bb6f55b77fac5ef)（2026-10-21T04:04:08Z，main）；[#12402 / cd772b3d](https://github.com/triton-lang/triton/commit/cd772b3da9d343ea690f1c5daea020083503fd03)（2026-10-21T02:05:47Z，main）。

#12401修ROUND_UP仍用E4M3 nearest-even的问题：例如1.03125原落到1.0，新选最小≥原值的可表示scale（1.125，上限448）；kernel与reference同步，matmul reference保FP32到输出量化，避免先BF16舍入改变scale选择。它影响低精度数值契约，不保证新旧codes逐位相同。

#12404把MMAv2 batched accumulator offset从逻辑每线程元素推算改用loadC后实际寄存器数/repBatch，覆盖小M/N复制与packing；#12402让tcgen05_copy可传pred，跳过pipeline迭代时不拷未ready scales，im2col在重写单buffer前等待旧异步MMA读完。后者作者GB300两tutorial ConSan433过/64跳、TMEM与sanitizer57过，不能当任意LLM kernel均无race。已读尺度、累加器偏移及predicate的API/回归。**建议低精度/自写Gluon kernel使用者优先回归**subnormal、全E4M3区间、小batched维度和false-predicate保留旧目的地。


<a name="core"></a>

## 核心开源工程

### NVIDIA/Megatron-LM

**DSA-over-GQA 接入 hybrid/ GDP 栈，仍是受限训练能力。** 新增 learned indexer 选 token top-k，再让主 GQA 只计算所选 keys；dense warmup 可保持 dense 输出同时训练 indexer，显式 spec 从既有 hybrid/GDP spec 派生。默认 backend=none 在该组合被解析为 min-memory-triton。simplified实现只准一个全局KV group（num_query_groups=1），并拒绝 CP>1、sequence parallel、packed sequence、attention bias 与动态 batching；decode-time indexer cache 尚未实现。它是四段系列的第二段，不能当作完整 DSA 训练配方或通用推理支持；本轮只核查接线、模式选择与边界，未全审依赖的 min-memory kernels。建议先对 reference/min-memory 的输出和 indexer 梯度做小模型对照，再接冻结、重置及 optimizer-clock 后续工作。

来源：[#7156](https://github.com/NVIDIA/Megatron-LM/pull/7156)（main，合入2026-10-21T09:06:50Z；`github:NVIDIA/Megatron-LM:pr:7156`；源码 `d14b699755061d55cb918f68500fba24401a1846`）。

深入源码：`megatron/core/transformer/experimental_attention_variant/dsa_layer_specs.py::_with_dsa_gqa_attention` → `megatron/core/transformer/experimental_attention_variant/dsa_gqa.py::DSGQACoreAttention.forward` → `megatron/core/transformer/transformer_config.py::TransformerConfig`。

**LayerShardedMuon 逐批构建并消费本地 Newton–Schulz。** 旧实现即使 ns_batch_size 很小，仍保留整个本地参数组的 Nesterov 输入和 NS 输出。现在 _iter_ns 按同 shape 分块、即时应用更新并原地缩放，减少同时存活的中间 tensor；跨 rank routed 路径仍保留 materialized _run_ns。默认 batch=1保持原数值路径，batch>1有 kernel 舍入差异且要求 emerging-optimizers≥0.3.0；公开移植未重跑 GPU/性能，源分支 benchmark 不能当本次结果。建议用同一参数组测峰值显存，同时验证 batch1 逐位、batch>1容差及 unwired TP fallback。

来源：[#8379](https://github.com/NVIDIA/Megatron-LM/pull/8379)（main，合入2026-10-21T13:09:27Z；`github:NVIDIA/Megatron-LM:pr:8379`；源码 `6c956376e3ef22528413ff35412ad8b938272986`）。

深入源码：`megatron/core/optimizer/layer_sharded_muon.py::LayerShardedMuon._iter_ns` → `megatron/core/optimizer/layer_sharded_muon.py::LayerShardedMuon._step_groups`。

**GDN prefill graph 把 chunk 描述符交给 inference context。** FLA capture-time metadata 的缓存寿命不能保证匹配新的序列布局。动态 GDN prefill 改走树内单 Householder-update 的 GDP recurrence/varlen convolution；context 持有固定容量 descriptor，replay 前原地刷新，稳定 workspace 和 launch grid。仍明确不支持 chunked prefill；依赖 kernel 内部未全审。新增 changing-layout replay 与一/二次 Householder determinism 覆盖。建议以相同总 token、不同请求数/长度反复 replay，核对 recurrent/conv state 与 eager，不能只测 capture 时布局。

来源：[#8353](https://github.com/NVIDIA/Megatron-LM/pull/8353)（main，合入2026-10-21T01:08:22Z；`github:NVIDIA/Megatron-LM:pr:8353`；源码 `29e1a7b6697c61d184c95726865d417fc51b8574`）。

深入源码：`megatron/core/ssm/gated_delta_net/gdn.py::GatedDeltaNet.ssm_prefill` → `megatron/core/ssm/ssm_inference.py::ssm_chunking`。

**TE grouped MLP 由隐式环境约定改为 split_alignment。** 构建 GroupedLinear 时探测新 TE API，并显式传 split_alignment=256；旧 TE 保留其原有选择方式，同时取消 MCore 对 NVTE_CUTEDSL_FUSED_GROUPED_MLP 必须打开的检查。shared/单组 fused MLP 对未对齐 token 数直接 ValueError，参数本身不负责 padding。TE 内部 fusion matcher 仍是依赖边界，也不代表256行就保证 batch invariance。建议升级 TE 时同时验证 grouped routed/shared expert 的 padding、零token与非256倍数，观察实际 fusion 是否命中。

来源：[#8184](https://github.com/NVIDIA/Megatron-LM/pull/8184)（main，合入2026-10-21T11:21:25Z；`github:NVIDIA/Megatron-LM:pr:8184`；源码 `45b68f914cf15eae6eaa40117fe94cd6a3972d4c`）。

深入源码：`megatron/core/extensions/transformer_engine.py::TE_GROUPED_GEMM_EXPERT_TOKEN_ALIGNMENT` → `megatron/core/transformer/moe/experts.py::_make_fused_ops` → `megatron/core/transformer/moe/shared_experts.py::_fused_grouped_swiglu_no_comm`。

### vllm-project/vllm

**MoRIIO READ：所有读取落地后才进入 forward。** MRV2 FULL replay 在 pre_forward 尚无 FULL context，piecewise/eager 的 indexer cache 也没有逐层 wait hook；旧路径可能在 attention/indexer/tail READ 仍在飞行时消费 KV。现在 start_load_kv 统一等本步全部 READ；attention-only 失败不提前结束其他请求的等待，超时仍有 in-flight 就使该步失败，hybrid 失败继续明确抛出。作者 MI300X/GLM-5.3-Flash-FP8、EP8/MTP1/block8，基线另叠#60867的256K对照原36/36 admission竞态、修复63次0竞态，最终1242次FULL admission也0；尚未复现由此引起的错误答案，不能宣称准确率提升。建议 PD 部署优先做 delayed/failed READ与FULL/PIECEWISE混合故障注入，记录 pending-after-forward，而非只跑短prompt。

来源：[#60808](https://github.com/vllm-project/vllm/pull/60808)（main，合入2026-10-21T12:11:17Z；`github:vllm-project/vllm:pr:60808`；源码 `db3b0f0fae4bc98492b3863685e541e62bc59576`）。

深入源码：`vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py::_await_reads_issued_this_step` → `vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py::waiting_for_transfer_complete`。

**CPU offload pool 按所有 PP/spec worker 最小容量对齐。** 不同 PP stage/target-draft 的每block字节数不同，scheduler按轻rank估容量会给重rank分配越界block。workers在分配前 all-reduce MIN，engine仅在启用的 SimpleCPUOffloadConnector（含MultiConnector）查询对齐容量并要求有效结果，scheduler直接使用；显式num_cpu_blocks仍走容量检查。作者旧ROCm smoke使用dummy weights，当前review版本CPU144过/18跳，三项worker同步GPU测试未运行，不能把旧smoke当最终拓扑验证。建议覆盖不均匀PP、spec target/draft和缺失RPC结果。

来源：[#52921](https://github.com/vllm-project/vllm/pull/52921)（main，合入2026-10-22T00:22:11Z；`github:vllm-project/vllm:pr:52921`；源码 `30aded62d46dfcf52c815925a8acec5a8bebddc6`）。

深入源码：`vllm/v1/simple_kv_offload/sizing.py::sync_num_offload_blocks_across_workers` → `vllm/v1/engine/core.py::_initialize_kv_caches`。

**多模态 sender-cache 失效延迟到 render 之间。** engine cache-miss 回报曾直接删sender缓存，可能夹在 processor lookup 与 merge 中触发 Expected a cached item。现在先coalesce仍缓存的hash，在单worker多模态executor下次处理输入前drain；失效失败把未完成集合放回，输出处理不等待慢renderer。作者CPU57过及A800真实sleep/wake通过，但早期同步吞吐测量不覆盖async争用。建议测已排队render、重复/不存在hash、一次drain失败与sleep/wake，确认下一次确实重发payload。

来源：[#62164](https://github.com/vllm-project/vllm/pull/62164)（main，合入2026-10-21T06:21:24Z；`github:vllm-project/vllm:pr:62164`；源码 `70806b6318b9ffce9d11d54dbbd43fa325735bfa`）。

深入源码：`vllm/renderers/base.py::queue_mm_cache_invalidations` → `vllm/renderers/base.py::_drain_mm_cache_invalidations` → `vllm/v1/engine/async_llm.py::output_handler`。

**IPC weight cache 把 KV 量化配置纳入身份。** IPC daemon缓存的是后处理权重，KV dtype/skip-layers不同会误复用K/V scales。WeightCacheKey新增这两项，并排序去重skip列表；IPC loader冻结首次构建的cache key供standalone reload复用。已有mismatch/fallback机制处理不兼容；作者H800合成FP8 checkpoint中旧auto→fp8使scales变1并改token，修复后copy/zero_copy与disk加载一致。只验证该加载路径，非全量量化backend或模型质量结论。建议daemon与engine配置变化、reload及等价skip顺序一起测。

来源：[#61758](https://github.com/vllm-project/vllm/pull/61758)（main，合入2026-10-21T03:19:27Z；`github:vllm-project/vllm:pr:61758`；源码 `979be69b2c86c7fe33169a188293af3a990943b8`）。

深入源码：`vllm/model_executor/model_loader/weight_cache/protocol.py::WeightCacheKey` → `vllm/model_executor/model_loader/weight_cache/ipc_loader.py::_fetch_entries`。

**KV offload 事件补齐 pending metadata 与 group 类型。** HIT_PENDING时即快照真实tokens/hash/geometry但不提前宣布可读；失败primary分配通过metadata-only cleanup清理，并避免旧cleanup删掉retry持有的信息。另一项让placeholder保留recurrent/SWA group分类且跨reset保存，分类不等于residency，placeholder仍可block_size=0。两修复独立，消费端校验未放宽。作者当前主要CPU回归；较早paired Dynamo smoke及mock transport不构成最终GPU/RDMA failover验证。建议对secondary-only promotion、partial-tail、同key retry及recurrent placeholder逐项检查事件契约。

来源：[#59855](https://github.com/vllm-project/vllm/pull/59855)（main，合入2026-10-21T06:16:42Z；`github:vllm-project/vllm:pr:59855`；源码 `6af9fc27fc5fe95c02cadd1735ef84113e07ee34`）；[#59703](https://github.com/vllm-project/vllm/pull/59703)（main，合入2026-10-21T17:58:04Z；`github:vllm-project/vllm:pr:59703`；源码 `550c1cbb8b0a366f1bb2b5aea560a1d8e70f04cb`）。

深入源码：`vllm/distributed/kv_transfer/kv_connector/v1/offloading/events.py::OffloadingEventsTracker.record_lookup` → `vllm/distributed/kv_transfer/kv_connector/v1/offloading/events.py::_placeholder_stored` → `vllm/v1/kv_offload/cpu/manager.py::take_events`。

**compile clone elimination 保留跨图I/O的快照。** 若returned clone的源与graph input或另一output共享storage，图外in-place allreduce/RMSNorm可能覆盖应保留的snapshot。pass现保留这类clone，未知storage也保守保留，测试覆盖piecewise/fullgraph。该pass仍不保证任意图aliasing安全；例子来自vLLM-Omni thinker→talker TP2。建议扩展自定义模型compile boundary时保留input快照，并在下游原地修改后检查旧输出不变。

来源：[#60318](https://github.com/vllm-project/vllm/pull/60318)（main，合入2026-10-21T22:15:11Z；`github:vllm-project/vllm:pr:60318`；源码 `d29702fd75531ea5dec046a478101f66f390aa5d`）。

深入源码：`vllm/compilation/passes/ir/clone_elimination.py::would_alias_graph_io`。

**breakable graph 的 eager-break 开关改为调用时读取。** 装饰器原先import时发现flag关闭就不包wrapper，而batch-invariant配置稍后才开启，import顺序因此影响FA2短prefill token。现在总是返回wrapper，调用时决定是否eager break；flag关闭仍直接调用原函数。作者CPU反例覆盖真实attention op提前import，但未把CPU测试等同GPU token回归。建议覆盖cold model-info cache、in-process/default-fork及VLLM_BATCH_INVARIANT=1初始化顺序。

来源：[#59941](https://github.com/vllm-project/vllm/pull/59941)（main，合入2026-10-21T22:45:07Z；`github:vllm-project/vllm:pr:59941`；源码 `0eea5aca8451b25bf3419cd9460125a3132d3169`）。

深入源码：`vllm/compilation/breakable_cudagraph.py::eager_break_during_capture`。

**Cohere2MoE 并行 block 重叠 attention 与 MoE。** 二者读同一norm输入，现让MoE走side stream，join后attention+MoE共用一次TP all-reduce，再把residual add融合进下层RMSNorm。side输入clone防Inductor复用storage；EP all2all、内部必须reduce的MoE、DBO、piecewise/breakable capture等保留单stream。CUDA默认启用阈值控制，设VLLM_COHERE2_ATTN_MOE_DUAL_STREAM_TOKEN_THRESHOLD=0关闭。作者H100 TP2多个量化checkpoint单次测TPOT约1.14–1.36×，主要收益来自双stream，不能泛化全部拓扑。建议在目标量化/EP配置下测join正确性和Eagle3 aux hidden states再评估吞吐。

来源：[#62101](https://github.com/vllm-project/vllm/pull/62101)（main，合入2026-10-21T21:46:37Z；`github:vllm-project/vllm:pr:62101`；源码 `0d34366b14e5eef6b1f3c7b799e71cab155ce44a`）。

深入源码：`vllm/model_executor/models/cohere2_moe.py::Cohere2MoeDecoderLayer._allows_dual_stream` → `vllm/model_executor/models/cohere2_moe.py::Cohere2MoeDecoderLayer._fork_mlp` → `vllm/model_executor/models/cohere2_moe.py::Cohere2MoeDecoderLayer.forward`。

**DSv4.1-Flash GB300 opt-in specialized decode。** decode_backend=latency_specialized让模型自有executor接管FULL-graph decode，最多384行；prefill/mixed/更大batch回stock。严格要求SM103、MRV2、TP4、DSpark及指定FP8 KV/MXFP4 indexer，拒绝sleep、LoRA、KV transfer、batch invariance及额外并行；部分内部FlashInfer特化只接受0.7.2，否则回stock。作者GB300同build开关对照batch1含draft step 5355→4375µs（18.3%），AgentX P90交互16–24%；本轮仅审接线/guard，未逐核审数千行CuTe/同步kernel。建议先复现特定recipe和fallback命中率，保留GPQA不确定区间，勿推成通用DSv4.1性能。

来源：[#61868](https://github.com/vllm-project/vllm/pull/61868)（main，合入2026-10-21T23:45:13Z；`github:vllm-project/vllm:pr:61868`；源码 `d0787e48394b214f78694fce3520a8c11eca045c`）。

深入源码：`vllm/models/deepseek_v41/nvidia/specialized_decode/__init__.py::unsupported` → `vllm/models/deepseek_v41/nvidia/model.py::build_specialized_decode`。

**MiniMax-M3 ROCm indexer 以context分片换取长上下文带宽。** opt-in minimax_m3_indexer_cp把KV blocks按rank轮转分片，all-gather queries后每rank评分所有heads的本地块，再交换每head完整top-k候选并精确merge；每rank index-cache流量降为1/TP，增加两次小all-gather和merge。限ROCm、TP>1、BF16 index cache，与要求FP8的fused decode互斥，默认关闭。作者MI350X TP4在4K短context更慢约7–15%，128K获益9–27%；建议按实际context/并发找拐点，并验证FULL graph捕获的是CP路径而非Dynamo tracing fallback。

来源：[#57832](https://github.com/vllm-project/vllm/pull/57832)（main，合入2026-10-21T09:03:47Z；`github:vllm-project/vllm:pr:57832`；源码 `153b76b304ee6fc176c6ff442f0822764557e255`）。

深入源码：`vllm/models/minimax_m3/amd/indexer_context_parallel.py::IndexerCPDecode.__call__` → `vllm/config/attention.py::AttentionConfig.minimax_m3_indexer_cp`。

**MiniMax-M3 MXFP4 在MI355X新增MonoKernel FFN。** --monokernel在1–16 decode行把3–59层FFN的reduce/norm/router/共享与路由experts/最终reduce合到persistent FlyDSL launch；attention、dense前缀、prefill/mixed和超范围batch保stock。限gfx950 TP4及指定FP8 KV/indexer、AITER和shuffle flags，startup聚合拒绝原因，kernel故障fail-stop。作者同flags对照C1 TPOT约−19.1%，C16接近持平；大型FlyDSL内部未全审，不能与上条BF16-indexer CP直接叠加。建议先跑runtime fault与跨rank终止测试，再做相同flags的单变量A/B。

来源：[#62712](https://github.com/vllm-project/vllm/pull/62712)（main，合入2026-10-21T08:58:12Z；`github:vllm-project/vllm:pr:62712`；源码 `385d1dab3ba489a9f50f7eee704faab49cf760c4`）。

深入源码：`vllm/models/minimax_m3/amd/specialized_decode/runtime/guards.py::model_refusals` → `vllm/models/minimax_m3/amd/model.py::build_specialized_decode`。

**mHC 跳过NaN padding的Sinkhorn并调小batch形状。** NaN padding行的组合结果本已NaN，现跳过其剩余Sinkhorn iterations；Blackwell及以上、hidden4096且≤4token选单pass split配置。前者保同结果，后者只重排FP32 split sum并有小舍入；正常填满较大graph不预期获益。作者GB300 GLM-5.3-Flash同文件覆盖A/B，C1 tok/s/user+5.8%、C6+3.4%，C1收益大于孤立kernel预测且原因尚未profile。建议分开测真实行/NaN padding和精确填满bucket，别外推全部并发。

来源：[#62745](https://github.com/vllm-project/vllm/pull/62745)（main，合入2026-10-21T22:17:04Z；`github:vllm-project/vllm:pr:62745`；源码 `57d9bf768e0e3e2153acc3b01ae3fda3b7393ded`）。

深入源码：`vllm/model_executor/kernels/mhc/tilelang_kernels.py::mhc_fused_post_pre_split_config` → `vllm/model_executor/kernels/mhc/tilelang_kernels.py::mhc_pre_big_fuse_with_norm_tilelang`。

**DSv4.1 replay 为target-token scoring保留所有prefill行。** _keeps_rows_np条件补上prompt_logprob_token_ids，避免只识别普通prompt_logprobs后裁掉打分行；同batch generation仍正常trim。作者合成component回归能检测172条缺失行，但三个graph-vs-eager测试在基线与修复均遇host allocator/nested-capture环境失败，非完整模型E2E证明。建议target-token scoring跨chunk、非identity req-slot、late offset与slot reuse一起回归。

来源：[#62652](https://github.com/vllm-project/vllm/pull/62652)（main，合入2026-10-21T14:51:16Z；`github:vllm-project/vllm:pr:62652`；源码 `85752a43c3cee76d9234704e189cc2ca69f24712`）。

深入源码：`vllm/models/deepseek_v41/nvidia/model_state.py::add_request`。

**量化PWAL保留Parameter对象的第二阶段扩展。** 承接昨日#62326，141处PWAL新建Parameter改调用replace_parameter，覆盖多种MX/FP/INT量化linear/MoE、CPU与XPU路径，保留loader类与dim属性。layout匹配时copy保地址，shape/dtype变时仍可能重绑.data；删除/重命名留给第三阶段。作者GB20023项通过且stack端到端tensor对照，未证明每个backend或单项收益。建议RL refit不只比输出，还检查对象身份、processed_weight_dims和graph捕获storage。

来源：[#62405](https://github.com/vllm-project/vllm/pull/62405)（main，合入2026-10-21T08:08:57Z；`github:vllm-project/vllm:pr:62405`；源码 `fc377f77714b42f706640d5857a0f2a5f233758d`）。

深入源码：`vllm/model_executor/kernels/linear/mxfp8/flashinfer.py::process_weights_after_loading` → `vllm/model_executor/kernels/linear/mxfp4/flashinfer.py::process_weights_after_loading`。

**可选watchdog在engine/worker停滞时写线程栈。** 最终源码以watchdog_config.dump_dir非空才启用；默认timeout300s、10s检查，同一stall只dump一次。idle queue等待期间disable，worker RPC前后feed，engine step前feed；SIGTERM/父进程退出也保留诊断。它记录线程栈而不恢复deadlock，也不等同GPU kernel超时检测。PR正文部分旧描述说queue改5s timeout，最终patch保留原blocking并用disable包裹。建议压测用小timeout故意卡RPC，确认dump路径、空闲无误报和恢复feed后可再次dump。

来源：[#55700](https://github.com/vllm-project/vllm/pull/55700)（main，合入2026-10-21T15:47:37Z；`github:vllm-project/vllm:pr:55700`；源码 `1c879919786497217b3b5fb0258d1d72f2442cb0`）。

深入源码：`vllm/config/watchdog_config.py::WatchdogConfig` → `vllm/utils/watch_dog.py::ThreadedWatchDog` → `vllm/v1/engine/core.py::_process_input_queue`。

**generate可直接返回derender后的assistant message。** /inference/v1/generate新增output_mode=derender，render携带ParseContext，生成端复用engine实际output.text与chat parser，减少分离derender和adjust_request的偏差。只支持非stream；有tool/reasoning parser须提供parse_context，非法context在送engine前400；隐藏reasoning也隐藏相应token IDs/logprobs。其他mode忽略新context字段以保跨版本兼容，代价是render请求常带tools元数据。建议对工具选择、stop、n>1和logprobs与coupled chat逐项对照；不代表流式derender已就绪。

来源：[#60082](https://github.com/vllm-project/vllm/pull/60082)（main，合入2026-10-21T15:20:47Z；`github:vllm-project/vllm:pr:60082`；源码 `817fa3518cd4bcc94b31e044c534c002e701c801`）。

深入源码：`vllm/entrypoints/scale_out/token_in_token_out/protocol.py::ParseContext` → `vllm/entrypoints/scale_out/token_in_token_out/serving.py::serve_tokens`。

**Rust frontend只给请求自带template设fuel预算。** 固定500K MiniJinja预算曾让远低于context上限的长对话500。server自带template现无fuel限制，受trust-request-chat-template许可的request override仍默认500K，可用VLLM_CHAT_TEMPLATE_FUEL调整，0关闭、非法值startup失败。作者mock-engine长1730/3002条message由500→200；这是render契约，非模型long-context质量。建议复现长agent会话，同时覆盖请求template信任和预算错误。

来源：[#62374](https://github.com/vllm-project/vllm/pull/62374)（main，合入2026-10-21T08:55:59Z；`github:vllm-project/vllm:pr:62374`；源码 `1f5514ec46c2b4d12237282ace0869393e26e0d7`）。

深入源码：`rust/src/chat/src/renderer/hf/template.rs::request_template_fuel` → `rust/src/chat/src/renderer/hf/mod.rs::HfChatRenderer.apply_chat_template`。

**chunked embedding明确拒绝multimodal payload。** 之前tokens_input重写即使一chunk也会静默丢多模态payload/hash/placeholder。现在整batch在展开前检测multimodal并400，提示关闭enable_chunked_processing；文本chunk与关闭chunk的多模态继续支持。作者H800 SigLIP HTTP和CPU72项回归，未新增多模态chunk算法。建议混合batch和短image输入也测拒绝，防止只用超长输入验证。

来源：[#62859](https://github.com/vllm-project/vllm/pull/62859)（main，合入2026-10-21T14:39:47Z；`github:vllm-project/vllm:pr:62859`；源码 `65a886c69f44cf446d6afa03315f39590ccd843e`）。

深入源码：`vllm/entrypoints/pooling/embed/io_processor.py::maybe_pre_process_chunked`。

**XPU启用无需confidence head的adaptive verification。** FlashAttention metadata把XPU纳入ALWAYS cudagraph支持，解除可变verify长度的限制。作者8×Arc Pro B60 Muse-Glimmer+DFlash2的bench同时用了vllm-xpu-kernels#647：K5/K7部分负载获益，SPEED中K3不开adaptive仍最快，不能将最高11.7%归给本PR独立。建议按K与熵/并发比较最佳整体配置，并独立核对XPU kernel版本。

来源：[#62244](https://github.com/vllm-project/vllm/pull/62244)（main，合入2026-10-21T14:19:17Z；`github:vllm-project/vllm:pr:62244`；源码 `35d01660ff3644de128253950b8fcb4b7705668f`）。

深入源码：`vllm/v1/attention/backends/flash_attn.py::FlashAttentionMetadataBuilder._cudagraph_support`。

**ROCM_ATTN 两处V-cache布局契约修复。** opt-in QK-norm+RoPE+KV fusion原只把首层切到interleaved V；torch≥2.11 hoisted layer-name共享pattern使其余层仍以普通layout读cache。现在所有eligible层设置layout，只去重pattern；未匹配QK-norm却走RoPE融合的层在runtime拆成RoPE+支持interleaved的writer。第二修复不可省，否则第一修复扩大错误面。作者MI300 Qwen3-8B GSM8K由严重退化回约0.87，Llama也接近fusion-off；受限ROCM_ATTN且flag默认关，新增writer两kernel有成本。建议多层、混合有无QK-norm与compile-cache重载一起比真实KV bytes。

来源：[#62403](https://github.com/vllm-project/vllm/pull/62403)（main，合入2026-10-21T02:04:25Z；`github:vllm-project/vllm:pr:62403`；源码 `27aadb30961922ad0fa118a31bed3c7d5baf9492`）。

深入源码：`vllm/compilation/passes/fusion/qk_norm_rope_kvcache_fusion.py::QkNormRopeKvCacheFusionPass` → `vllm/v1/attention/backends/rocm_attn.py::do_rope_and_kv_cache_update`。

### verl-project/verl

**SGLang在线低精度refit以完整bucket序列重建layout。** 新增MXFP8与BF16 dense+MXFP4/NVFP4 routed-expert refit：先stage canonical checkpoint tensors，最后marker触发一次backend后处理并折回既有graph storage；异常恢复live storage，但源码不能据此保证已写权重数值全部事务回滚。拒绝delta_flush及MXFP8 dense+MXFP4 MoE组合。作者Blackwell Qwen3.5-35B-A3B测匹配QAT降低部分k3_kl mismatch，CPU suite受torchcodec动态库阻断且未提交新unit tests。建议多bucket失败/重试、graph replay及scale aliases联测，勿把恢复storage等同整个sync原子提交。

来源：[#8306](https://github.com/verl-project/verl/pull/8306)（main，合入2026-10-21T08:39:31Z；`github:verl-project/verl:pr:8306`；源码 `5265d561a3bf29f3af3d0aeae8634fe8bf4617c3`）。

深入源码：`verl/utils/sglang/sglang_quant_utils.py::load_and_reprocess` → `verl/utils/sglang/sglang_quant_utils.py::_abandon_sync` → `verl/workers/rollout/sglang_rollout/sglang_rollout.py::update_weights`。

**vLLM/SGLang低精度默认DeepGEMM，显式backend仍优先。** 最终合并代码对非NVFP4的有效quant config使用engine_kwargs.setdefault(...,deep_gemm)，两引擎都保留用户显式backend；NVFP4仍走既有FlashInfer CUTLASS。Hopper FP32 block-scale在选DeepGEMM时关闭E8M0重舍入；vLLM0.29缺MXFP8 Linear DeepGEMM仍有实际fallback。PR正文的VERL_ROLLOUT_USE_DEEP_GEMM=0退出开关和强制backend措辞已被最终patch替换，不能照抄。作者8项CPU refit通过，Blackwell mismatch只限给定Qwen拓扑。建议记录实际linear/MoE backend与train/rollout KL，采用engine_kwargs显式配置复现实验。

来源：[#8308](https://github.com/verl-project/verl/pull/8308)（main，合入2026-10-21T10:23:29Z；`github:verl-project/verl:pr:8308`；源码 `619c397dc9917fed278b6e47c52a6620aef05466`）。

深入源码：`verl/utils/vllm/vllm_quant_utils.py::set_default_vllm_quantization_backends` → `verl/utils/sglang/sglang_quant_utils.py::set_default_sglang_quantization_backends`。

### sgl-project/sglang

**spec verify的KV移动不再重读可复用请求行。** 接受tokens的目标地址改取out_cache_loc中原verify window前缀；read-done后scheduler可能复用已finish的req_to_token行，旧重读会写坏新请求乃至radix缓存prefix。影响CUDA NGRAM与UNO tree verify；EAGLE仍保整forward barrier。作者H200缓存prefix+短停请求压力后旧8/8 logprob probe不符、修复0/8，greedy4/4恢复；这不是所有spec/cache竞态的闭合。建议把EOS/stop、row复用与tree topk>1放到同一压力测试。

来源：[#45491](https://github.com/sgl-project/sglang/pull/45491)（main，合入2026-10-21T04:12:42Z；`github:sgl-project/sglang:pr:45491`；源码 `573b5716f5b47c77a5fb5f2c24ae9b0e596dc3d1`）。

深入源码：`python/sglang/srt/speculative/spec_utils.py::move_accept_tokens_to_target_kvcache`。

**external linker offload同时等索引与forward写KV完成。** prefill read-done让scheduler可提前推进，单在scheduler stream录event不足以保证待offload页已经被forward写完。registry传入真实forward_stream，Mooncake/UMBP linker记录两个event，后台copy先都synchronize。只影响external linker；作者由代码发现、尚无main现场复现，unit wiring不等于RDMA E2E。建议对长prefill stashed chunk施加延迟写与offload，核验host bytes，别把shared-read结束当KV写结束。

来源：[#45559](https://github.com/sgl-project/sglang/pull/45559)（main，合入2026-10-21T20:53:37Z；`github:sgl-project/sglang:pr:45559`；源码 `2c7a9e3b50828720951a5f7e71f42563d2df1b47`）。

深入源码：`python/sglang/srt/mem_cache/unified_cache/unified_cache_linker.py::record_offload_ready` → `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_direct_linker.py::offload_thread_func`。

**失败PD传输的KV延迟释放跨attention-TP一致。** 任一rank需要hold就MAX归约让全rank一起保留；release条件MIN归约，待每rankdrain或达到其既有device timeout才共同释放，host目的地始终须drain ack，不能由timeout替代。仅替peer hold的rank不等本地不存在的ack。作者真实双process Gloo队列回归59过/6subtest，GPU cache与transport被mock，无RDMA E2E。建议模拟ack/timeout错位，核对每rank admission容量与poll collective长度一致。

来源：[#45453](https://github.com/sgl-project/sglang/pull/45453)（main，合入2026-10-21T17:37:29Z；`github:sgl-project/sglang:pr:45453`；源码 `b57b5e22867f110ff577a9da18a9ff7e6619100c`）。

深入源码：`python/sglang/srt/disaggregation/decode.py::_all_reduce_release_flags` → `python/sglang/srt/disaggregation/decode.py::resolve_deferred_releases`。

**Mooncake失败session probe默认开并退避/清理。** 临时断网后旧failed_sessions可让健康decode持续拒流直到重启。现在独立线程probe默认开启，失败指数退避（base30s、cap300s），无人请求600s清除旧session，仍被路由命中的session刷新idle计时；成功probe才解除活跃黑名单。prune后再次请求会真send并可能重新黑名单，TCP probe成功也不证明KV数据面已无故障。作者100 dead-ID模拟每小时probe12000→400，非吞吐benchmark。建议做网络恢复与滚动更新，观察恢复次数、dead-ID增长及真实传输成功。

来源：[#45338](https://github.com/sgl-project/sglang/pull/45338)（main，合入2026-10-21T09:16:20Z；`github:sgl-project/sglang:pr:45338`；源码 `be65391abd01f711cdc443a3fe8f508b0c960357`）。

深入源码：`python/sglang/srt/disaggregation/mooncake/conn.py::_run_one_probe_pass` → `python/sglang/srt/environ.py::SGLANG_ENABLE_FAILED_SESSION_PROBE`。

**NVFP4 TRT-LLM修补gate/up异scale时的共同clamp。** 承接昨日#45289的canonical尺度顺序修复：TRT-LLM单一GEMM1 clamp无法同时匹配不等gate/up alpha。load时取两者max，把较小半区比例折进E4M3 block scales，再由共同alpha计算clamp/beta/output scale，无新增forward op。非二次幂ratio会重舍入、极小scale仍可subnormal/归零；作者GPU结果在rebase前，CPU最终20+5项通过，未拿到问题finetune checkpoint。建议非等scale、clamp与SiTU/无clamp controls跨reload/graph测试，不将representation invariance当通用质量保证。

来源：[#45386](https://github.com/sgl-project/sglang/pull/45386)（main，合入2026-10-22T00:37:07Z；`github:sgl-project/sglang:pr:45386`；源码 `9301ca37a0173fe6b5d5d53ef7ed40b76adf33e3`）。

深入源码：`python/sglang/srt/layers/quantization/modelopt_quant.py::ModelOptNvFp4FusedMoEMethod.process_weights_after_loading`。

**Kimi-K3 FULL+MAMBA MLA接入external linker。** linker group新增paged MLA KV与TP分片Mamba state，DSPARK sidecar draft共用target slot索引并作为DRAFT对象一起存取。拒绝非plain MLA、DCP>1、packed draft、多sidecar，以及unified memory/int8 Mamba checkpoint/额外side states。作者B300 TP8 device-hit/host-hit按token IDs/logprobs验证，跳过某pool load的负对照能失败；每rank仍复制MLA KV，带draft lookup不能命中无draft旧prefix。建议测pool完整性、flush后host恢复与draft acceptance，勿只量cache hit数量。

来源：[#44977](https://github.com/sgl-project/sglang/pull/44977)（main，合入2026-10-21T02:47:17Z；`github:sgl-project/sglang:pr:44977`；源码 `eb91a41f380bad9eb320fbe6bd52e6f15061e314`）。

深入源码：`python/sglang/srt/mem_cache/hybrid_cache/linker_pool_assembler.py::_build_mamba_device_pool_group` → `python/sglang/srt/speculative/base_spec_worker.py::_build_hicache_draft_plan`。

**Mamba mixed chunked prefill保留checkpoint tracking。** mix_with_running原merge清空prefill track字段，forward未写却把slot以已写checkpoint捐给radix，后续prefix可静默恢复旧state。现在保存prefill mask/seqlen/slot，decode tails标untracked；lazy预分配被mixed跳过时归还并记下batch，防finish快捷分支误判写入。仅extra_buffer/extra_buffer_lazy+mixed-chunk相关；作者CPU真实planner/backend路径回归，未GPU E2E。建议先造prefill/decode混batch后再prefix命中，检查捐出slot确实被写。

来源：[#45435](https://github.com/sgl-project/sglang/pull/45435)（main，合入2026-10-21T04:56:51Z；`github:sgl-project/sglang:pr:45435`；源码 `32c26788f3dafedcd87ee9d4a4dec65ea8ded716`）。

深入源码：`python/sglang/srt/managers/schedule_batch.py::mix_with_running` → `python/sglang/srt/managers/schedule_batch.py::mamba_lazy_undo_prealloc_at_boundary`。

**Mamba extend规划用host镜像与device gather移除隐式同步。** ping-pong空槽状态镜像在host，track索引在device一次gather，metadata经pinned非阻塞H2D；slot清零改index_fill_，multimodal placeholder mask用host排序+searchsorted替代可能sync的isin。slots所有权/写入点必须与host flags一致，非单纯替换.item。作者CPU回归，未提供可归因的end-to-end吞吐；建议在overlap下用sync-debug/trace确认scheduler可领先GPU，并逐分配/捐赠/释放验证镜像一致。

来源：[#45472](https://github.com/sgl-project/sglang/pull/45472)（main，合入2026-10-21T18:56:51Z；`github:sgl-project/sglang:pr:45472`；源码 `740d5f9f9e33aeebc6b5bcb1ba73f25bf6db062b`）。

深入源码：`python/sglang/srt/managers/schedule_batch.py::_gather_mamba_track_indices` → `python/sglang/srt/mem_cache/memory_pool.py::set_mamba_ping_pong_slot`。

**GLM-5.3-Flash CP与FlashInfer KDA prefill graph接通。** interleave BCG为prefix-free多请求按总token bucket replay，固定地址local embeddings/positions与rank-major IDs；target CP下DFLASH draft仍普通TP。FlashInfer KDA新增每bucket/每layer wrapper plan，刷新global token容量、sequence/state/checkpoint映射；实验wrapper缺失时仍eager。BF16 recurrent pool另把snapshot在Triton分支merge前cast，避免BF16/FP32编译类型冲突。CP图仍限制模型/拓扑、无prefix hit与MM；较早16384 capture测量用了显式上限，最终默认2048，最终restoration未重跑该GPU测量。#43182本窗已合并，不能照#45473旧正文再称open。建议分开验证capture、真实replay与吞吐，结合小bucket、多请求、checkpoint和BF16 state。

来源：[#43182](https://github.com/sgl-project/sglang/pull/43182)（main，合入2026-10-21T22:51:29Z；`github:sgl-project/sglang:pr:43182`；源码 `f4035cab0be4dd54099eeb044a701ad9aeecb5b4`）；[#45473](https://github.com/sgl-project/sglang/pull/45473)（main，合入2026-10-21T18:57:24Z；`github:sgl-project/sglang:pr:45473`；源码 `f06d1bb1f45df617aa455c103ce5ede2bc6c6a47`）；[#44886](https://github.com/sgl-project/sglang/pull/44886)（main，合入2026-10-21T07:58:35Z；`github:sgl-project/sglang:pr:44886`；源码 `a361f0c410b5f6c56b840fbf01e3ba21a58748e1`）。

深入源码：`python/sglang/srt/layers/cp/bcg/interleave_bcg.py::InterleaveCPBCGInput.allows_replay` → `python/sglang/srt/layers/attention/linear/kda_flashinfer_prefill_graph.py::FlashInferKDAPrefillGraph` → `python/sglang/kernels/ops/mamba/kda_prefill_track.py::_kda_track_ssm_kernel`。

**DSv4.1把跨层共享读移到batch metadata边界。** low-ratio index-K slots按ratio在metadata init收集一次，prefill index计划及request-window history_src跨layer复用；graph-captured metadata无snapshot时保留每层构建。随后V4.1/request-window允许PRE_REPLAY read-done，使安全的none/DFLASH组合提前调度；CP、HiSparse仍whole-forward，DSPARK明确不受益。两个metadata hoist合测4K step约106–110→98.7ms，未分别消融；#45582未用目标DSPARK负载测收益。独立chunk-stash fence #45580仍open，不能把这一串等同所有释放竞态已修复。建议先按实际spec/cache组合确认read-done触发，回归跨layer地址寿命。

来源：[#45577](https://github.com/sgl-project/sglang/pull/45577)（main，合入2026-10-21T21:00:35Z；`github:sgl-project/sglang:pr:45577`；源码 `bc412213c17cecfa826716e23d5ae00748bbe528`）；[#45579](https://github.com/sgl-project/sglang/pull/45579)（main，合入2026-10-21T21:01:20Z；`github:sgl-project/sglang:pr:45579`；源码 `f7d4f06242b76615a2cc95e87426b7123a90c7f6`）；[#45582](https://github.com/sgl-project/sglang/pull/45582)（main，合入2026-10-21T23:29:29Z；`github:sgl-project/sglang:pr:45582`；源码 `da7b97a737aad1dfcf7cf8721494e58174485b14`）。

深入源码：`python/sglang/srt/layers/attention/deepseek_v4_backend.py::_snapshot_low_ratio_k_slots` → `python/sglang/srt/layers/attention/deepseek_v4_backend.py::_snapshot_prefill_shared_reads` → `python/sglang/srt/mem_cache/dsv41_request_window.py::window_layout`。

**DSv4 request-window sparse prefill与DP tail恢复快路径。** request-window encoder rows重新允许sparse prefill，decoder late-tail仍dense；DP attention+MegaMoE tail可从大prefill图跳到row-bucket图，全rank通过既有vote/host metadata保持一致，CP/unified-KV等排除仍保留。作者GB300 DEP4单次测量request-window sparse约+5.3–9.6%，tail4K/8K step78.6→58.7ms及140.3→97.5ms；后者叠加#44627，前者200题准确率在无关crash前仅答198题。建议固定所有相关开关逐项A/B，别把组合数字相加。

来源：[#45573](https://github.com/sgl-project/sglang/pull/45573)（main，合入2026-10-21T20:54:33Z；`github:sgl-project/sglang:pr:45573`；源码 `97945831f9ec29755099284243ab7d91336d314f`）；[#45581](https://github.com/sgl-project/sglang/pull/45581)（main，合入2026-10-21T21:44:22Z；`github:sgl-project/sglang:pr:45581`；源码 `f3f0f7b68c9a8a062b63855bc86bb95de181f134`）。

深入源码：`python/sglang/srt/layers/attention/deepseek_v4_backend.py::_use_sparse_prefill` → `python/sglang/srt/models/deepseek_v4_replay_tail.py::breaks_out_of_prefill_graph`。

**DSv4.1 MXFP8融合与tactic覆盖decode尺寸。** rmsnorm_mxfp8扩展128-row scale tile寻址，移除q_norm→wq_b的≤8行限制；caller仍限Blackwell、BF16和1280维，kernel K需128倍数。MXFP8 autotune使用阈值4096→1，deterministic/batch-invariant继续固定tactic。作者给出1–1280行对照bit-identical及DEP8特定128K decode约2–3% step收益；未提供全模型广泛对照。建议跨128行边界核对swizzled scales，并分别测deterministic与auto tactic。

来源：[#45633](https://github.com/sgl-project/sglang/pull/45633)（main，合入2026-10-21T23:56:46Z；`github:sgl-project/sglang:pr:45633`；源码 `ed89693c4ac3f187792494723a240345eac440cf`）；[#45630](https://github.com/sgl-project/sglang/pull/45630)（main，合入2026-10-21T20:35:57Z；`github:sgl-project/sglang:pr:45630`；源码 `028de2c5d485385bee3b68f6463a0a887f42e4d2`）。

深入源码：`python/sglang/kernels/ops/quantization/mxfp8_epilogue.py::mxfp8_epilogue` → `python/sglang/kernels/ops/layernorm/rmsnorm_mxfp8.py::rmsnorm_mxfp8` → `python/sglang/srt/models/deepseek_v4.py::autotune_prefill_kernels`。

**DSv4.1 DP-attention padding不再进入真实index/history。** idle有padding行时EngramHasher早返零hash且不写history；eager extend按预建真实low-ratio rows裁x/q_lora/positions，避免repeat_interleave把未初始化request IDs交给compressor。仅DP pad/idle边界变化；作者原复现只跑初版，最终版未重复，CI无对应DP×attentionTP组合。建议用attn_tp>1的不平衡DP rank与idle/eager/graph切换验证无device assert。

来源：[#40424](https://github.com/sgl-project/sglang/pull/40424)（main，合入2026-10-21T21:37:56Z；`github:sgl-project/sglang:pr:40424`；源码 `29e78045fba1b4616e21e3d828b2c58df5e27879`）。

深入源码：`python/sglang/srt/layers/engram.py::EngramHasher.forward` → `python/sglang/srt/layers/attention/deepseek_v4_backend.py::forward_low_ratio_sources`。

**Ling-2.5/2.6默认归一化top-k路由权重。** 此前config缺norm_topk_prob时默认False，被旧FlashInfer总归一化路径掩盖；#44904正确尊重flag后模型MoE输出缩放错误。模型默认现True，显式配置保留。作者B200200题GSM8K0.43→0.87、accept length2.68→3.40，但测试是在旧nightly加同一行fix，非最终rebase完整模型；Bailing V3同默认尚未审。建议比selected expert weights的和及routed_scaling_factor，别回退正确的router gate。

来源：[#45662](https://github.com/sgl-project/sglang/pull/45662)（main，合入2026-10-21T21:53:58Z；`github:sgl-project/sglang:pr:45662`；源码 `aa5880e2b8f7807543a6d4745f96bd7a50821b2c`）。

深入源码：`python/sglang/srt/models/bailing_moe_linear.py::norm_expert_prob`。

**uvicorn多tokenizer worker可选独立reuseport listener。** 默认关闭的enable-http-worker-reuseport使每worker有自己的SO_REUSEPORT socket，由Linux分散新连接，避免共享accept队列让长stream/WebSocket永久偏到少数worker。supervisor替换/重启也给新listener，startup失败清理已启动进程；要求uvicorn≥0.51，拒绝单worker与HTTP2。仅连接分配不迁移已有会话，也没有受控端到端吞吐结论。建议按生产长连接分布对比每worker活跃连接、event-loop延迟并测试滚动重启。

来源：[#45482](https://github.com/sgl-project/sglang/pull/45482)（main，合入2026-10-21T18:57:26Z；`github:sgl-project/sglang:pr:45482`；源码 `7e5256d1ba3aaa628bb8b756fafb904a76a1123a`）。

深入源码：`python/sglang/srt/entrypoints/http_server.py::_ReusePortMultiprocess` → `python/sglang/srt/arg_groups/serving_hook.py::enable_http_worker_reuseport`。

**NIXL UCX可按process-visible GPU指定NIC。** 复用disaggregation-ib-device单值/list/JSON映射到UCX device_list，拒绝HCA端口后缀及与backend params冲突；不改变其他backend。作者H20096次full-buffer transfer校验和8请求PD smoke，独立GPU→pinned-CPU mapping约12ms接近原生NIXL配置，default约500ms是该host选择结果，非通用倍数。GPU→GPU同host仍可CUDA IPC/NVLink，选NIC不强制RDMA。建议按可见GPU编号核binding日志并分开测G2G/G2CPU。

来源：[#43482](https://github.com/sgl-project/sglang/pull/43482)（main，合入2026-10-21T09:54:12Z；`github:sgl-project/sglang:pr:43482`；源码 `2e7c9d851c868678ddfcd93b4e8000cd23e1f779`）。

深入源码：`python/sglang/srt/disaggregation/nixl/conn.py::_resolve_ucx_device_list`。

**XPU staging KV传输已接入，prefix依赖仍未合并。** staging改device-neutral streams/events，pointer table支持高位USM地址；NIXL在XPU每rank异步WRITE完成后才重用同staging源，避免下一head slice覆盖。device_type成为必需keyword，XPU DCP pack退回per-token transfer。作者Arc Pro B60 Mooncake/NIXL E2E均叠加仍open的#43299；单本PR共享prefix并发仍会在早发torch.cuda.Event崩prefill，需特定torch static-launcher workaround及Mooncake TENT/UCX设置。建议先等依赖或显式集成后复现，不把四项通过写成已发布组合就绪。

来源：[#40861](https://github.com/sgl-project/sglang/pull/40861)（main，合入2026-10-21T14:09:05Z；`github:sgl-project/sglang:pr:40861`；源码 `8e3e2980db83a1b020a6d273a2b653891af81bfc`）。

深入源码：`python/sglang/srt/disaggregation/common/staging_handler.py::register_decode_req` → `python/sglang/srt/disaggregation/nixl/conn.py::transfer_worker` → `python/sglang/srt/disaggregation/common/dcp_pack.py::init_dcp_pack_buffers`。

**Ascend decode retraction的host_pool配置与view修正。** backup-only decode也应用Ascend HiCache IO/layout配置，page_first_direct正确交换page/layer维形成layer views，防host页数少于layer数时错误转存或越界。作者最终8项真实NPU单测通过；rebase前MHA TP1 eager Llama3.2 PD重traction token与baseline一致，未重跑最终PD版本，host耗尽仍可能500。建议测真实retract→新device页恢复、partial page和容量耗尽，不泛化MLA/MoE或多TP。

来源：[#43814](https://github.com/sgl-project/sglang/pull/43814)（main，合入2026-10-21T06:48:01Z；`github:sgl-project/sglang:pr:43814`；源码 `fb6e6a09ce5e3dc2ebc93ee64d11572a4da4d758`）。

深入源码：`python/sglang/srt/hardware_backend/npu/utils.py::set_default_server_args` → `python/sglang/srt/mem_cache/pool_host/mha.py::_initialize_transfer_views`。

**Ascend backend初始化前移到parallel runtime入口。** init_parallel_runtime先装transfer_to_npu，再执行可能访问torch.cuda.*的prewarm，避免NPU-only torch启动报Torch not compiled with CUDA enabled。证据是小patch、原错误stack及作者after截图，无性能/模型质量结果。建议在全新进程验证prewarm与group初始化，不依赖其他import偶然提前装patch。

来源：[#43039](https://github.com/sgl-project/sglang/pull/43039)（main，合入2026-10-21T11:44:31Z；`github:sgl-project/sglang:pr:43039`；源码 `2815eed49c94dbc3266fb5dc4b01c61fc8eb73ca`）。

深入源码：`python/sglang/srt/distributed/bootstrap.py::init_parallel_runtime`。

**BF16→FP8按token量化保留NaN便于暴露数值故障。** UE8M0路径改NaN-preserving clip，FP32-scale路径BF16直接用saturating FP8 conversion，NaN不再被洗成+448；FP16因有限值scale也可能overflow而保留旧sanitize，FP32/INT8路径不变。作者GB300123项含graph replay通过，但全模型async-assert warmup可能读取未初始化buffer，仍需独立验证。建议故障注入区分BF16真实NaN与graph padding，并检查下游断言，不当作数值稳定性提升。

来源：[#42689](https://github.com/sgl-project/sglang/pull/42689)（main，合入2026-10-21T08:52:29Z；`github:sgl-project/sglang:pr:42689`；源码 `7a632cf998d437b0541700dd0b4f808c1f275ed5`）。

深入源码：`python/sglang/kernels/jit/csrc/gemm/per_token_group_quant.cuh::QuantTrait`。

**GLM-5.2 graph capture调整side-stream录制顺序。** 保留fork，把wq_b/q_b_proj的side work录制放到并行main kernels之后、join之前，避免graph replay主链被first-recorded-parent规则吸到许多side lanes。作者GB200 dummy-weight profile30ms slice stream45→3，TPOT0.4–0.8%在单次噪声内，不宣称提速；本轮只审发射顺序。建议用nsys对照graph replay lane归属与event依赖，性能另做重复A/B。

来源：[#45349](https://github.com/sgl-project/sglang/pull/45349)（main，合入2026-10-21T07:36:55Z；`github:sgl-project/sglang:pr:45349`；源码 `f7b8c6fe404ca4f29c3188cb50692f7735d45a99`）。

深入源码：`python/sglang/srt/layers/attention/dsa/dsa_indexer.py::_fused_q_prepare_and_store` → `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py::forward_absorb_prepare`。

### THUDM/slime

主窗0 merged PR、0 main commits；72h的3个merged PR和5个main commits已与历史核对，本轮不重复报道isolated SGLang runtime与Straw rewind。release collection当前11项，主窗无新发布。

### pytorch/torchtitan

**72h补录：Llama3.2 1B/3B的RoPE factor改32。** flavor曾继承Llama3.1的8倍缩放，加载发布checkpoint会算不同旋转频率。现只将1B/3B设32，8B/70B/405B不变；作者同随机权重CPU与HF对照误差下降并补frequency test。旧Titan从零训练且使用factor8的1B/3B checkpoint升级后也会改变频率，不能静默当兼容修复。建议恢复时明确保留原training RoPE配置，发布checkpoint则对齐config。

来源：[#5528](https://github.com/pytorch/torchtitan/pull/5528)（main，合入2026-10-19T08:27:25Z；`github:pytorch/torchtitan:pr:5528`；源码 `48bbce4434dfac2c38fedc9c7ef9c1998b04cf50`）。

深入源码：`torchtitan/models/llama3/flavors.py::_1b` → `torchtitan/models/llama3/flavors.py::_3b`。

**72h补录：HF保存的实际tensor、名字、layout与index一致。** GPT-OSS普通BF16导出默认使用HF unquantized名字与[in,out]布局，仅quantized-layout load使用_blocks和[out,in]，不支持保存成MXFP4；DeepSeekV3保存去掉未写的FP8 scales/不存在MTP层，必要MTP embedding/head拷贝独立clone，避免空dtype safetensors header导致无法加载。MTP拷贝也增加读取/临时内存（作者671B BF16约3.7GB），DeepSeekV4显式退出这套命名。作者87/89项CPU回归，DCP内部未重审。建议每次last_save_in_hf后独立reader reload并核index=实际keys及flag/transpose，不只测试内存roundtrip。

来源：[#5064](https://github.com/pytorch/torchtitan/pull/5064)（main，合入2026-10-20T06:21:59Z；`github:pytorch/torchtitan:pr:5064`；源码 `da54a64ede6983f52f5d66a1290eefc5e13e3260`）；[#5065](https://github.com/pytorch/torchtitan/pull/5065)（main，合入2026-10-20T06:26:56Z；`github:pytorch/torchtitan:pr:5065`；源码 `1d581795cd014a46e871cc9fc605325fbbceee85`）。

深入源码：`torchtitan/models/gpt_oss/state_dict_adapter.py::GptOssStateDictAdapter.to_hf` → `torchtitan/models/deepseek_v3/state_dict_adapter.py::DeepSeekV3StateDictAdapter.to_hf`。


<a name="engineering"></a>

## 扩展开源工程

### PyTorch Inductor 默认模块级 Triton kernel：回滚后重落地，序列化/预编译路径要一起看

**pytorch/pytorch；本窗默认分支增量。** [#200502 / 3d348547](https://github.com/pytorch/pytorch/commit/3d3485473ded9e269d5cac1fa8fcd5a0b43e5ad9)（2026-10-21T19:47:15Z，default-branch commit）；[#200501 / d10d6194](https://github.com/pytorch/pytorch/commit/d10d6194a2ee627f1f3e4215d52bf2fd1771055e)（2026-10-21T06:14:38Z，default-branch commit）；[#200500 / a7466f6b](https://github.com/pytorch/pytorch/commit/a7466f6b5084511b69f5f34e789d93646790f9a5)（2026-10-21T06:14:38Z，default-branch commit）。

Python wrapper默认从async_compile.triton字符串绑定改成module-level kernel定义，前置两项让预编译从生成文件加载Python，并从kernel字符串还原的源编译模块级函数，以适配其inspect/source依赖；C++/FX wrapper仍保留字符串形式。10-21T06:14Z先落地、13:38Z回滚、19:47Z重落地；本项默认行为基线固定到最后 `3d348547...`，不把中途回滚当最终状态。PR挂在ghstack base，落地结论来自main提交列表与固定提交。已读codecache/生成wrapper相关patch，未独立跑compile/cache reload。**建议扩展AOT/自定义codegen集成者深入阅读**，回归跨进程加载、源码提取和缓存复用，没有新模型吞吐结论。


### Dynamo：停止构建分发 KVBM，PdConnector迁入独立路径；多模态接收补失败清理

**ai-dynamo/dynamo；2026-10-21T16:08:56Z，默认分支 `45a4e56a5f4dd4111d937ac36f38ac669677468d`。** [固定提交](https://github.com/ai-dynamo/dynamo/commit/45a4e56a5f4dd4111d937ac36f38ac669677468d)。

main停止wheel/container/CI分发KVBM，旧示例标注要求Dynamo≤1.6.x；PdConnector保留并把module path改成 `dynamo.vllm.pd_connector`。这是交付与集成边界变化，不能说所有源码已经删除或当前1.6 dev release已撤掉该功能。直接依赖旧kvbm包路径的分层KV部署需要迁移，LMCache/NIXL组合示例也更新路径。

另 [`f5475886`](https://github.com/ai-dynamo/dynamo/commit/f5475886b583190f8dbb43097d388614b2c1e409)在mm_kwargs接收前验证整个batch的READ metadata、descriptor/size和remote编码，再借buffers/创建read；准备阶段失败借助ExitStack恢复借出资源，防止坏后项耗尽缓冲。已读 `_receive` 和Remote.decode_metadata分离及回归，不将其扩成任何飞行中RDMA都可回滚。**建议Dynamo升级前检查包依赖/connector路径，回归混合合法与非法多模态batch**。


### Dynamo HTTP 连接独立取消信号，sidecar恢复 visible stop token契约

**ai-dynamo/dynamo；2026-10-21T23:29:46Z，默认分支 `b2137c7b87607de865f3d361ee7f69405a71d46d`。** [固定提交](https://github.com/ai-dynamo/dynamo/commit/b2137c7b87607de865f3d361ee7f69405a71d46d)。

新accept loop用hyper-util/Tokio代替axum-server，每连接持有child CancellationToken，避免每个stream chunk唤醒都争用共享Notify waiter锁；有宽限期的graceful shutdown后关闭未结束连接，TLS握手timeout仍10秒。已读server入口与终止契约，本轮没有真实并发性能数据，不能由perf标题得出tok/s提升。

[`8e26089d`](https://github.com/ai-dynamo/dynamo/commit/8e26089d3f2ebddcad6c2117911c75d47a931ddf)在SGLang/TRT sidecar转发visible stop IDs并让TRT保留可见stop输出，帮助Harmony `<|call|>` 等抵达frontend decoder；仍需处理visible token等于end_id可能被丢弃及TRT旧版本min_tokens只拦end_id等限制。**建议长流式连接与工具调用服务考虑回归**，同时测慢客户端、停机和stop token边界，不把协议修复当模型能力变化。


### 另外两项接口变化：Nemotron TP 计划与 KV transfer backend 合同

**Transformers #48780，10-21 main `8e40ad7b`。** [固定提交](https://github.com/huggingface/transformers/commit/8e40ad7bc7019ba19ae8b0e7d98e60376d0fac2a)为旧 `model_type=nemotron` 新增q/k/v/up列分片、o/down行分片、lm_head gather-output计划；不是Nemotron所有混合/MoE系列自动获得同一能力。作者两rank CPU/Gloo对单进程最大绝对差4e-7，未多GPU验证。**建议该模型使用者考虑复现**，不推断多GPU吞吐。

**TensorRT-LLM #20117，10-21 main `a9000fe1`。** [固定提交](https://github.com/NVIDIA/TensorRT-LLM/commit/a9000fe10ecd62973dcf3670573419299b390acb)把backend合同与native extent/Attempt/Outcome类型分离，MemoryKind独立模块；是可定位的扩展接口整理，没有证据证明新增一种传输算法或提高RDMA可靠性。**建议自定义backend维护者深入阅读迁移路径**，普通服务可继续观察；不要把“定义合同”解释成已完成任意backend实现。


### Triton GSan access profiler：按观察到的storage寿命推荐检测粒度

**triton-lang/triton；本窗默认分支增量。** [#12338 / 6caee94f](https://github.com/triton-lang/triton/commit/6caee94f73b140877238364d3ce37b9688e7dfb9)（2026-10-22T00:00:48Z，main）。

新增gsan-profile插桩与AllocationProfile，专用allocator为storage lifetime记录访问、在stream顺序退休时汇总并重置，支持多device、IPC/FABRIC映射和Gloo/NCCL汇总；需在CUDA初始化前显式装PyTorch CUDAAllocator，unmanaged/raw allocator内存忽略。它为shadow granularity和write_once提供候选，**推荐仅反映已插桩执行，仍须独立GSan run验证**，不是race-free证明或通用算子耗时profiler。作者GB300110项profiler和其余sanitizer/编译测试，rebase后多节点FABRIC未重跑。**建议自定义通信/kernel调试者深入阅读**allocator接管与storage复用边界，先用人工race负例验插桩覆盖。


<a name="hardware"></a>

## 硬件软件协同

### FlyDSL 的 PyTorch 集成有端到端新披露，但 3.01×不属于 BF16 通用收益

**AMD / Meta；官方技术文10-21，新披露，部分kernel早已进入main。** [官方全文](https://pytorch.org/blog/accelerating-pytorch-on-amd-gpus-with-flydsl/)。gfx950近期ROCm PyTorch nightly加可选FlyDSL0.3.x：compile的dense/grouped/MXFP GEMM作为多候选实测选择，eager仅符合条件RMSNorm/TopK自动dispatch；SDPA经AOTriton≥0.14.2b内建kernel选择，不需另装FlyDSL。flex_attention仍独立Triton路径，AOTInductor/更广硬件/epilogue fusion等为后续。

新实测单MI355X、Llama3.1-8B/Qwen3-32B/Llama3.3-70B、输入256/输出固定512、并发8–256、相同KV及服务配置：BF16以ATen+Triton对照、加FlyDSL候选最高1.13×（多次ABBA）；MXFP8/MXFP4以ATen为对照最高2.04×/3.01×，低精度每格仅一次。后两者**不是比BF16精度方案的收益**，不能与kernel几何均值混比。TopK tie索引和NaN选择也可能不同。**建议gfx950用户复现真实负载的候选选择与质量**，固定nightly/包和精度；不要只取峰值宣传为PyTorch全场景提速。


### AITER 泛化 GDN decode 与 gated norm+MXFP4：混合架构的融合边界更完整

**ROCm/aiter；本窗默认分支增量。** [#6488 / d62da36c](https://github.com/ROCm/aiter/commit/d62da36c14bdb976a73b78829fd4237c1375ba11)（2026-10-21T13:38:38Z，main）；[#6038 / bc36872a](https://github.com/ROCm/aiter/commit/bc36872a6683343b413b4c861a6184daef55fa55)（2026-10-21T17:37:35Z，main）。

#6488将原gfx950二值head/一key head限定扩成ratio-generic sibling，支持Qwen3.5/3.8-27B的48V/16K（ratio3）和flat qkvz布局；每token/key-head group持有多份FP32状态，先发独立load、再conv/归一化/递推更新/门控RMS。wrapper支持query_start_loc跳过空graph行及预分配out、允许slot stride；ratio2旧interleaved路径仍独立。接口出现不代表所有上游框架已接入，配套vLLM PR须单独核状态。

#6038把GDN输出gate/RMS与MXFP4激活量化融合，支持norm前/后gate及silu/sigmoid；先舍入回输入dtype再量化以对齐原两算子，group须整除N且32倍数，scale可选转置以匹配后续GEMM。已读实现的dtype舍入、shape guard与状态测试，作者MI355X验证，未独立测模型收益。**建议混合架构ROCm部署考虑复现**，同时核对head比例、packing、空行state不更新和out_proj真实scale布局。


### AITER gfx1250 通信：QuickReduce已有受限路由，TP64仍只有16rank验证

**ROCm/aiter；本窗默认分支增量。** [#5965 / fa344a29](https://github.com/ROCm/aiter/commit/fa344a298b05bb09316e5d0b2efe72b24f689640)（2026-10-21T02:17:41Z，main）；[#6570 / 3a22ce72](https://github.com/ROCm/aiter/commit/3a22ce7236f4c8d137e1c4dce86b8ed19f74cbf5)（2026-10-21T02:20:36Z，main）。

#5965以三个同stream dispatch量化32KiB tile→单CTA发布/等待→按固定rank次序拉取压缩片，system-scope release/acquire与epoch奇偶双缓冲避免graph replay覆盖旧读。gfx1250只路由BF16→FP16+INT4；TP2≥16MiB、TP4≥8MiB、TP8≥16MiB才启用，小消息回custom AR或RCCL；不符合量化/架构约束则初始化禁用，不向所有gfx125x外推。该压缩路径不能由跨rank逐位相同推定与无压缩通信等价。

#6570将gfx1250、MoRI对称内存allgather实现扩到最大64rank，但作者只实测4node/16rank，8/16node机器不足；**“最大TP64”是代码准入，不是64rank已验证扩展性**。**建议匹配硬件/大消息的部署考虑复现**，继续观察跨节点规模、通信误差与真实LLM负载；未给本轮硬件验证。


### PyTorch ROCm grouped GEMM 新增 CK-Tile 候选；MX scale shape 检查收口

**pytorch/pytorch；本窗默认分支增量。** [#200307 / 6e33a88c](https://github.com/pytorch/pytorch/commit/6e33a88c262b6d2baf26e91c44eae607a8c3148b)（2026-10-21T22:03:59Z，default-branch commit）；[#201012 / d36355df](https://github.com/pytorch/pytorch/commit/d36355df345d6a3e79988775e90a13de556ca970)（2026-10-21T21:55:25Z，default-branch commit）；[#198195 / 5ec3d6c5](https://github.com/pytorch/pytorch/commit/5ec3d6c5e2f01fd3d01ae0cfb0eb87902f28c85b)（2026-10-21T06:48:05Z，default-branch commit）。

本轮实际默认分支提交证明落地；这些ghstack PR的merged_at可为空，不能仅按该字段误判未入main。CK-Tile首阶段grouped GEMM由 `ROCM_ALLOW_GROUP_GEMM_CK_TILE=1` 显式启用，需构建USE_ROCM_CK_GEMM，仅FP16/BF16、group≤1024及对应连续维16字节对齐；主要验证gfx942/950，不是通用默认backend或Inductor自动调优声明。MX scaled_mm v1/v2统一shape检查，ROCm v2的NO_SWIZZLE现在要求unpacked K为128倍数，原32×64×32等配置会报错；SWIZZLE_32_8的K为32倍数、N仍有16对齐要求，v1语义未变。Inductor另将NVIDIA reduction阈值限制为非HIP。已读固定main guard与测试，未进行MI系列benchmark。**建议在真实MoE shape上显式开启后复现，并回归scale布局兼容性**，勿从kernel可用推断所有precision组合已支持。


### XLA/PJRT 增长式显存池可回收整块空闲 region，修复容量未满却 OOM

**openxla/xla；2026-10-21T18:42:12Z，默认分支 `5b313eb620887bbd4c5c6d337954a82ec170c8ea`。** [固定提交](https://github.com/openxla/xla/commit/5b313eb620887bbd4c5c6d337954a82ec170c8ea)。

当preallocate=false时，新建BFC allocator启用现有whole-free-region回收重试，并遵循 `TF_ENABLE_GPU_GARBAGE_COLLECTION`（默认true）；已预分配池保持保留策略。问题是多个全空的小region占着增长额度，无法满足一个更大请求，回收后可在同cap下重分配。它**不移动活跃分配、不提高cap，也不解决含live allocation的region内部碎片**。

已读 `CreateBFCAllocator` 的明确条件及8类CPU可测allocator回归。作者早修订四GPU/Gemma2-9B与Gemma3-4B各101步有限loss，但最终rebase未重跑编译/训练，也无独立pre-fix训练对照；不能宣称新版本训练性能提升。**建议该类OOM使用者优先考虑复现**，固定运行时并记录reserved/allocated及preallocate开关。PR51216为Copybara导入，页面merged_at空不影响本main提交证据。


### XLA 对称内存 collective：RA2A可直接录图，模块边界仍需防御性复制

**openxla/xla；2026-10-21T11:44:20Z，默认分支 `810a16e6dbe6fc864c441ee6d54f5eb1243ce04d`。** [固定提交](https://github.com/openxla/xla/commit/810a16e6dbe6fc864c441ee6d54f5eb1243ce04d)。

Ragged-all-to-all的对称内存kernel提前在Init加载，避免录图时延迟加载产生死锁；command buffer显式记录pre-barrier→kernel→post-barrier节点并保存更新所需节点映射，符合条件时归COLLECTIVES_KERNEL默认录制。runtime还要求找到对应collective memory且LSA规模覆盖全部rank，不能称任意网络/跨域都走此路径。另本窗 [`e1e30b81`](https://github.com/openxla/xla/commit/e1e30b81d345346700a60a513da11a2c1d7f93d0)移除“persistent symmetric memory即可跳过模块边界复制”的优化，因为调用者输入/返回输出由PJRT普通显存分配，不保证对称。**建议MoE通信与graph集成者深入阅读**，先核真实allocation来源、barrier和节点更新；本轮没有通信速度实验。


### TorchTPU 新增等长 packed-document 自动折叠，并修临时 host buffer 生命周期

**google-pytorch/torch_tpu；2026-10-21T13:10:49Z，默认分支 `0db0f30bea8c75b4960b318460453d310c6ced22`。** [固定提交](https://github.com/google-pytorch/torch_tpu/commit/0db0f30bea8c75b4960b318460453d310c6ced22)。

flex_attention redirect首次materialize BlockMask并验证它能完整重建为相同方形块对角矩阵，才按document拆attention；结果按BlockMask memoize并在compile中常量折叠，替代调用者手工设置 `_tpu_doc_len`。不适用于任意变长packing或跨文档可见mask，首次检测仍有构造/比较成本，未给新端到端倍率。

同窗 [`548e7f47`](https://github.com/google-pytorch/torch_tpu/commit/548e7f4773ced0dc234cac20aaa6273bc769f388)对CPU→TPU非连续或dtype转换产生临时host副本时强制blocking，防止函数退出释放副本但DMA尚未结束；其余满足条件的non_blocking不因此全被关闭。异步split compile的 [`1f682796`](https://github.com/google-pytorch/torch_tpu/commit/1f682796c3682a9311a02005ca240bf003b35b9e)在debug_callback前resolve所有artifact，修空future列表提前回调；[`069328ca`](https://github.com/google-pytorch/torch_tpu/commit/069328cacd15da2343d4d88d7b92986726b8eaa8)批量drain ZMQ并直接按rank await已有队列减少Task开销，等待全部rank的语义仍保留。**建议TorchTPU试用者固定main基线回归**；这些10-21 main改动不等同此前0.2 wheel已全部包含。


### MLX FP4 prompt prefill：半精度解码快路保留 MXFP4/BF16 的精度退路

**ml-explore/mlx；本窗默认分支增量。** [#4709 / 52a6c234](https://github.com/ml-explore/mlx/commit/52a6c234ab35ddf46c8bd1517b849b59b610d078)（2026-10-21T07:52:10Z，main）。

Metal fp_qmm_t将可安全场景的FP4反量化改成FP16 tile并一次scale，长prompt用更大tile，减少先FP32解码/缩放再转换。NVFP4和MXFP4/FP16模型使用快路，MXFP4/BF16为避免精度损失保留原路径。作者Qwen3.8-27B-nvfp4、2048-token prompt：M1Pro54.4→76.5、M2Ultra275.0→378.1prompt tok/s，M4Pro132.1→144.4；生成速度未变，也没有多负载/质量统计。已读dtype/quant模式选择和数值测试。**建议Apple本地长prompt场景考虑复现**，勿把prefill改善当decode加速或扩到所有4bit格式。


### llm-d SGLang/XPU P/D overlay：跨节点 TCP + Level Zero 已有验证，RDMA 尚未随本项提供

**llm-d/llm-d；本窗默认分支增量。** [#2762 / 5489cf64](https://github.com/llm-d/llm-d/commit/5489cf641a9acaf6982b860915c5cabb82eb18ce)（2026-10-21T08:21:53Z，main）。

新增1P+1D、TP1社区支持overlay，镜像以UCX1.21.x/LevelZero+NIXL1.3.2构建，运行 `UCX_TLS=tcp,ze_copy` / `UCX_MEMTYPE_CACHE=n`，避免CUDA-only wheel及UCX缓存把XPU误判host。prefill禁early-send cached prefix，decode用cpu_tensor retraction backup以绕现有CUDA-only分支；因此不能把它当上游SGLang原生XPU全路径已无缺口。

作者实际两个节点42/42 recall完成、无NIXL/UCX错误、KV计数增长后回落；未测跨节点TP>1或性能，RDMA provider仍限vLLM，XPU/SGLang RDMA留后续PR。已读镜像patch、P/D env配置和guide支持矩阵。**建议有Intel XPU集群者考虑复现功能正确性**，继续观察上游依赖闭合与RDMA支持，不用TCP通过推断高带宽互联可用。


### vLLM-Ascend O-projection跨层共享scratch：稳定地址和准确profiling优先

**vllm-project/vllm-ascend；本窗默认分支增量。** [#19190 / 2164a24d](https://github.com/vllm-project/vllm-ascend/commit/2164a24d1e509fe211b60c4635fa266e292bd8b4)（2026-10-21T09:18:31Z，main）。

DeepSeek V4/V4.1串行DSA层共用model-owned OProjBufferPool，按shape/dtype/device复用exchange与reduce-scatter scratch；KV profiling前预留，按最大decode graph容量而非更大profile token budget定尺寸。空DSA rank仍进入OTP collectives，oproj TP的0/1都归禁用，避免单rank假启用。已读pool构造、各模型/DSpark绑定与预留；作者四项聚焦场景9项通过，完整DSA单测曾在Triton编译前后失败并中止，**未做多rank HCCL验证**，没有可泛化节省GB数。**建议深入阅读并按真实图容量复现**，核每层地址复用、常驻scratch计入KV预算和ragged DP collective顺序。


### XLA GPU补齐variadic reduce-scatter执行链路，仍保留布局/类型约束

**openxla/xla；2026-10-21T13:37:26Z，默认分支 `780b56d9f7c3ac7b51842fdba070c994118cfa0e`。** [固定提交](https://github.com/openxla/xla/commit/780b56d9f7c3ac7b51842fdba070c994118cfa0e)。

本轮把多operand reduce-scatter串到kernel支持判定、emitter和运行时执行，放开原先单buffer入口限制；可flatten性仍受replica group、reduction computation、元素类型、layout和size约束，并非任意异构tensor可打包。已读运行时BuildReduceScatterInfo和相关codegen/test变更，未独立测多卡吞吐或FSDP完整step；同窗有variadic计数上限工作，不能把接口放开等同无上限。**建议多小tensor通信显著的训练配置考虑复现**，固定XLA版本并检查实际发射的collective数/形状和数值对照。


### TrainBridge：GPU编译差异可给NPU优化指路，2×是初始移植版本的实训加速

**TrainBridge；港中大（深圳）、Shenzhen Loop Area Institute、Autokernel；2610.29171v1，10-21T04:00:48Z。** [§3–4、Limitations](https://arxiv.org/html/2610.29171v1)。先读同程序GPU eager→compile图/融合IR/kernel/profile的变化，归并semantic supernode、映射NPU同语义区并做本地改写，再NPU profiling处理残余瓶颈。迁移的是优化意图而非直接照搬CUDA kernel，每候选重验训练质量。

H100/PyTorch2.7/CUDA12.8对910C/PyTorch2.7.1/CANN8.5/torch-npu2.7.1.post2；Record50同12h调参、三次独立agent run，通用agent最佳实训均值419.2s、TrainBridge283.2s（−32.4%）。27历史NanoGPT任务单次最多18h调参，对初始NPU平均1.95×，NPU/GPU时间比4→2；额外四任务1.25–2.22×，均**排除调参时间**。原训练schedule/验证loss约束有价值，但单机、小训练任务与单次多任务选择限制外推。建议深读具体迁移操作并复现固定调参预算，先核初始port是否已有成熟融合，记录tuning amortization后再算成本。


<a name="evaluation"></a>

## 评估、可靠性与结论纠正

### Tokenizer v2纠正相关方向：U-statistic强相关仍不具跨语言预测力

**No Single Tokenizer Feature Reliably Predicts Downstream Language Model Performance；EleutherAI、巴塞尔大学、台湾科技大学、中央研究院、Rochester；2610.21679v2，10-20T12:09:27Z。** [v2全文](https://arxiv.org/html/2610.21679v2)、[v1](https://arxiv.org/html/2610.21679v1)。本轮打开两版全文、比较全部已解析章节/表格。三模型suite、九语言、29内在指标的基本研究问题不变，**§6把TokSuite英语U-statistic从“强正相关”改为强负相关，英语/土耳其ρ=−.85/−.58，低值反而配更好模型，与指标设计方向相反**；SLOPE-Zipf英语/意大利方向也反转披露。不可沿用旧版“语言学对齐分越高越好”的推论。

v2还改为19/29总体指标显著（v1为13/29）；控制总参数后19指标改善fit、最大额外R²仅.015（原23指标/.021）；两指标联合adj-R²≤.09、额外解释≤.04，仍弱。§8原English orthographic显著改为Bonferroni后仅Farsiρ=.86显著，**单40题**；AppD补分model-size展示及chance/Avg口径，AppE3独列参数量控制。v2显式揭示vocab大时embedding参数变化的混杂，因此不能把词表大小当纯tokenization单变量。

**建议深读并修正既有读书笔记/统计结论**，复现须对齐suite、语言、总参数、机会水平筛选和多重比较；本轮未重新计算作者统计。属于相关方向/报告口径实质纠正，非证明tokenizer没有影响。

### ECHO：排序分数可持续变好而生成坍缩，失败集中在高需求的平均token排序目标

**ECHO；东京大学、TIER IV、华为；2610.29410v1，10-21T08:35:39Z。** [§4–8及Tables1–3](https://arxiv.org/html/2610.29410v1)。把训练后final hidden位移分为跨位置/响应共享offset与残差，发现高需求ranking loss可几乎不改chosen/rejected margin却把概率推入单token“井”。注入/移除offset、对loss中心化及惩罚projection的干预支持该测试范围内的因果作用。

17个base lineage .5–14B、驾驶list与UltraFeedback，14/17在最多6×budget坍缩；旗舰held-out top1 .995却89%释放概率集中textt。**textbook DPO在1×/3×budget均未坍缩**，SimPO在通常预算也无井、全参3×才部分失效；不是“所有DPO都坍缩”。offset控制在四family至3×守住ranking但6×OLMo可走残差通道；旗舰另两seed，其他单seed。建议深读并加generation/常见token质量、offset投影监控，不能只按pairwise accuracy选checkpoint；预测预警目前是完成run后回看，代码/数据公告待出版释放。

### Graft：连续去噪的均值可抹掉低频词，补回词汇尾部不等于事实性改善

**Graft；电子科技大学；2610.29136v1，10-21T03:15:30Z。** [§4–7及Tables1–3](https://arxiv.org/html/2610.29136v1)。在ELF/LangFlow同一中间状态比较head，连续均值去噪比分类readout更早丢低频候选；Graft在候选尚未收窄时读词表、选择将丢失但有支持的rare词，固定其位置并从同noise重启生成，配整词/边界处理减少变形。该机制只对已打开的两连续dLLM实现，不代表全部diffusion模型。

冻结ELF105/342/652M、LangFlow652M、OWT1024token、200samples/arm；ELF-B rare share5.01→9.76%、human11.88，nonword2.64→2.57%；LangFlow固定dose反过冲1.25×human且JSD变差，另seed调剂量后1.03×、但nonword3.79%仍高human1.14。实测**2.2×wallclock成本**，非加速法；生成PPL可能奖励尾部缺失故未当主指标。建议观察/复现词频、非词、上下文恰当性和同预算best-of4，XSum未检测到unsupported entity改善，补“真的词”不能当补“真实的事实”。

### 后续问题压缩评测：已知问题选句的好处不一定留给未来，但“词级总更好”也不成立

**Evaluating Context Compaction on the Questions That Come Later；作者声明Independent Researcher；2610.27301v1，10-19T21:15:56Z；10-21公告。** [§2–6、Tables1–2](https://arxiv.org/html/2610.27301v1)。把压缩时可见known问题与随后hidden问题分开，随机删除token匹配、证据保留与KEEP–DROP干预一起评估。MultiRC300段/3355题，25%预算gpt-6-luna对known比random+11.4pp，而hidden−.4pp、95%CI跨零；disjoint hidden−2.9pp。LLMLingua2词级压缩的hidden优势部分来自留下更多内容词。

不能概括query-aware失效：录制τ-bench390states/837hidden identifier题，typed比random **+22.3pp**，LLMLingua2+4.0且区间跨零；工具ID/语法与普通语篇不同。FEVER label命中可能无证据支持，LoCoMo空memory四选项已81.4%。论文非live agent，亦受旧题污染风险限制。**建议立即加入压缩验收**known/hidden、token匹配、证据链/ID与support-based评分，算压缩构建成本；恢复检索亦未自动弥补损失。

### Anti-Persistence：保留轨迹、临时生成策略，避免把错误抽象直接固化为长期规则

**Anti-Persistence；HTML机构字段损坏，本轮未核清；2610.27406v1，10-19T23:58:46Z；10-21公告。** [§3–4、Tables1–2](https://arxiv.org/html/2610.27406v1)。持久存任务、轨迹、outcome与模式反馈，每次检索后bandit选直接用记录/模仿/策略/纠错，生成的指导执行后丢弃；**仍可经受指导影响的轨迹间接反馈，不是杜绝所有错误传播**。

三backbone×ALF/Web/Science、10更新轮、3seeds、held-out状态全冻结；九格均分trajectory53.5、abstraction49.0、joint49.3、anti59.9；anti只在7/9格居首，Gemma-Science joint56.3高于anti54.1。受控保留同套临时指导，Llama-ALF54.7→26.7，支持此配方中的持久化风险，不能证明所有维护策略必败。**建议复现memory设计**时保原始轨迹可追溯、严格冻结评估写入，另测LLM调用数以外的tokens/延迟；“更多记忆必更差”不是论文边界内结论。

### 组合任务脆弱性：单题三次都做对，拼批仍可因每题思考预算缩短而失败

**On the Fragility of Tasks under Composition；TU Munich、MIT CSAIL；2610.27190v1，10-19T19:23:48Z；10-21公告。** [§3–5](https://arxiv.org/html/2610.27190v1)。先取各模型独立3/3做对题，再组3/6/10/20题，同题独立评估另采样；用唯一tag把thinking分配至题，观察组合后缩短与accuracy损失，并控制context溢出。跨组合大小同模型fragilityρ≈.81，跨模型仅≈.24；**这是对模型特定脆弱性的证据，不是通用难度标尺**。

§4.2用固定embedding+逐模型线性预测器、5-fold cross-fit，held-out fragility batching平均省31.0%token、每数据集accuracy损失≤.5pp；已知经验fragility的更乐观约43%不能混为部署收益。需要大量组合试验打标签，未证明总采样成本能快速回收；研究主要同类独立任务拼接。OLMo3-7B RLVR比SFT/DPO更抗精简，但预算缩短依然存在。**建议复现批量问答**，固定标签提取和独立基线，补标注成本、延迟及不同任务混合，不直接套给长agent历史。

### AITER decode benchmark 修正：12 GB/s→5819 GB/s 不是 kernel 加速

**ROCm/aiter；本窗默认分支增量。** [#6690 / e52cac39](https://github.com/ROCm/aiter/commit/e52cac3940248c66fd1c01aa2bdb753cc7f16e4a)（2026-10-21T17:58:50Z，main）。

旧脚本把所有请求映到4个物理KV页且只按这些页算字节，再用含launch/同步开销的计时。新脚本默认为每sequence/block slot分不同页（每cache tensor上限4GB），按所有实际attended K/V字节计带宽，使用warmup后cudagraph计时；max_context_len变Python int以免wrapper同步。MI355X/BF16/64Q/8KV/HD128/batch8×8192，Triton显示12.2→5819GB/s、Gluon10.3→5420，代表**分母、cache命中和计时协议同时改变**。**建议采用新脚本并重建基线**，历史两列不能计算加速比，也不能把新数字解释成全LLM有效带宽。


<a name="revisions"></a>

## 版本复核与发布观察

- **Beam NVFP4为评测修订。** Reflection于10-21T21:21Z/23:26Z在NVFP4及主卡把LCBv6值87.4%改为88.3%；新旧tree只有README变化。没有新权重或实验协议增补，不能据此写量化模型重新训练/无损性证明。后续采用应固定[更正卡](https://huggingface.co/reflection/Beam-1.0-501B-A23B-NVFP4/blob/f8ecac0e16045babf60156f6ba996a77fcb48cf9/README.md)并复核原始输出。稳定ID：`hf:reflection/Beam-1.0-501B-A23B-NVFP4@f8ecac0e16045babf60156f6ba996a77fcb48cf9:eval-card`。
- **Ming3新增训练内容摘要。** inclusionAI于10-21T07:29–13:45Z在ASR Tiny、Audio Tiny/Flash、TTS Tiny/0.5B仓加入TDS2026 PDF，非新权重。打开五份七页摘要：§1.3勾选text>10T token/audio>1M小时、数据截止约2026-05；§2列公共/许可/爬取/合成来源类别。它们没有拆继承骨干/本阶段实际消费量、每checkpoint混合比例与训练预算，且保留未填写版本/更新日期等模板字段，不能用作精确训练配方。建议关注后续补充可枚举语料与阶段统计；固定[ASR摘要](https://huggingface.co/inclusionAI/Ming-3.0-asr-tiny/blob/ac95e7b1678c08ea6f4a2581a051e432ed63e9b7/Ming-3.0-asr-tiny-TDS2026.pdf)。
- **NVIDIA聚合稿中的日期要拆开。** 10-21[模型组合公告](https://blogs.nvidia.com/blog/open-models-europe-industry/)把已发布Nemotron3.5 SuperVL、Retriever、Kumo及新Cosmos工件集合到一起；Bolt E4B4GB仍预告10-23，Alpamayo2Edge是本月稍后。不能把预告计作权重可用。Cosmos3-Nano Sim-Auto/Transfer-Auto/Sim-Bimanual/Sim-Transfer确有10-21T02:33–03:28Z初始整包commit，卡标release10-20；属物理世界模型侧线，技术报告卡仍标coming soon，本轮只核卡/工件，未审其训练/仿真系统。
- **MiniMax M3.1 Flash与Mercury3继续观察。** 本轮重开[MiniMax发布文](https://www.minimax.io/blog/minimax-m31f)、[模型release notes](https://platform.minimax.io/docs/release-notes/models)、[Mercury3发布文](https://www.inceptionlabs.ai/blog/introducing-mercury-3)、docs与官方HF库存，仍定位10-19既有发布；MiniMax发布文仍说权重稍后。未核到本窗新权重/独立技术报告，不重复上一轮性能结论；相关缺口不以无搜索命中关闭。

### 其余八项修订：不将原有性能重新作为新结果

| 论文与版本事件（UTC） | 实际打开与核实的增量 | 处理与实践建议 |
|---|---|---|
| [Scaling Agentic Data 2610.13166v2](https://arxiv.org/html/2610.13166v2)，10-20T04:29:28Z；对照v1；中科大、高德/阿里巴巴、南方科技大学 | 已解析全部section/figure/table文字相同 | 元数据/资产仍可能变化；不重报旧训练性能，保持原数据与评估边界 |
| [知识线性可迁移 2610.22300v2](https://arxiv.org/html/2610.22300v2)，10-19T22:46:55Z；LMU Munich、MCML | “states”改“recalls”、例子/图注措辞及资助/引用更新；已解析数值表未变 | 表述澄清，不能当新增泛化实验；继续区分相关性与因果性 |
| [Muse Glimmer 2610.24628v2](https://arxiv.org/html/2610.24628v2)，10-20T10:14:52Z；Meta Superintelligence Labs | §11.2 contributors补名与重排，技术章节/表格未变 | 不当新模型版本/权重改变证据；权重变动须另查模型repo历史 |
| [MemT1.0 2610.26825v2](https://arxiv.org/html/2610.26825v2)，10-20T14:30:38Z；MemTensor Team | AppA1作者名旁NVIDIA附注/脚注删除，其余已解析正文表格未变 | 署名披露调整，不重报memory性能 |
| [Training-Free OPD 2610.22236v2](https://arxiv.org/html/2610.22236v2)，10-20T04:16:41Z；HTML机构字段损坏，本轮未核清 | 仅参考文献格式/链接改变；方法与实验表未变 | 无新增训练或推理对照，原方法成本边界继续有效 |
| [POPE读出审计 2610.25964v2](https://arxiv.org/html/2610.25964v2)，10-20T04:43:52Z；Sri Krishna College of Technology、Sri Eshwar College of Engineering | 全部已解析section/figure/table文字相同 | 昨日首token surface修正建议仍有效；v2不构成反证或新性能 |
| [STOPD 2610.21337v3](https://arxiv.org/html/2610.21337v3)，10-19T21:09:06Z；对照已报v2；作者机构本轮未核实 | 引言/related work增POPD/FastOPD/Prune-OPD/PG-OPD/KAT对比，结论删去部分外推措辞；机制、性能表未变 | 优先读局部梯度sum/IPW的条件；新增相关工作评述不等于全部对照实测 |
| [Looped Transformer截断研究 2610.26030v2](https://arxiv.org/html/2610.26030v2)，10-20T17:28:49Z；作者机构本轮未核实 | 重排附录/图表并扩写allocation推导、代价近似和边界；592runs/347配置、230fit+117holdout及已匹配主要性能表数值保留 | full-BPTT最优仍条件于拟合law/compute近似；v2强调整数loop窄区间例外、free-size/attention成本限制，不应写通用定理 |

上述HTML差分不覆盖PDF像素、图片内容和打包资产身份；CA-MOPD另做了v1/v2 PDF逐页文本提取比较。没有因为正文相同关闭历史修订枚举缺口。


<a name="coverage"></a>

## 来源覆盖、补录与未完成区间

本轮启动先读automation memory、state、来源账本、16篇历史日报和GitHub目录/索引；17个既有本地Markdown blob与远端逐一相符，待补传队列为空。历史日报仅作去重与进度依据，当前结论用本轮原始证据核实。归档范围仅为llm-monitor与tracker/llm-training-daily，没有自动改写llm-knowledge wiki。

### 实际覆盖

| 来源类别 | 本轮实际工作 | 覆盖边界 |
|---|---|---|
| 官方模型/数据 | 39机构按模型/数据createdAt、lastModified双排序，共156页；91模型/23数据候选；另从框架变化发现RecursiveSI并补查4仓 | 列举到72h边界/公开库存末尾，不等于所有候选已语义审阅；关键新条目比卡/config/历史tree与权重对象 |
| 六核心仓 | all-base merged PR主窗176/main171；回溯703/main689；默认分支commit库存698，主窗171；57PR、49机制组深审 | merge时间与commit时间独立；库存完成不等于全量机制审计，非main能力没有冒充main |
| 扩展工程与硬件仓 | 24仓默认分支72h库存1,676commit、主窗448，分页到短页；55个选定PR metadata/files及19个指定commit快照；另外补查rocm-libraries的projects/rccl与nki-samples，两条窗口内列表为0 | 只枚举默认分支committer日期，未全base审计；release多数只开最近10项，不能声称历史release修订全覆盖。0是指定表面/窗口，不代表项目没发布 |
| arXiv | 六分类10-21公告1,225出现/879唯一ID，300replacement metadata；API扩宽主窗204条、精确窗过滤196，扩宽回溯1,477 | 35项首次收录含13主窗新提交、22回溯补录；另9已报ID版本复核，共44版本。全文访问和选定章节审阅不等于所有候选已读完 |
| 正式学术来源 | TMLR列表增4forum ID；OpenReview API/两选定PDF、ACL/PMLR入口均尝试 | TMLR仅月粒度；OpenReview403、forum200是动态壳；ACL/PMLR根链接无变化不能替代逐volume版本枚举 |
| 厂商/硬件官方 | NVIDIA、AMD、Intel Gaudi、Ascend、Google TPU、AWS Neuron/NKI与PyTorch官方入口/选定全文 | 硬件组件与旧区间仍不完整；厂商主页/搜索没命中不能当无新发布 |

实际重查39个既有及扩展HF机构，每机构模型和数据集各按createdAt/lastModified两种排序，**156页请求、模型原始行4,874/数据集1,942（包含双排序重复）**；均核日期递减并到达72小时边界或公开库存末端，合并得到91模型/23数据候选。本窗及关键回溯候选已取卡/commit/tree，权重与文档分类仅对说明过的条目做对象比对。列举完成不是所有候选语义全审，更不是全球发布全覆盖。

官方已开站点覆盖OpenAI、Anthropic、Google/DeepMind、Meta、Microsoft、NVIDIA、DeepSeek、Qwen、Seed、Kimi、智谱、MiniMax、腾讯、百度、Step、小米、美团、Mistral、xAI、Ai2、Reflection等。OpenAI研究页和微软研究页直连403，xAI news403；Qwen博客是JS壳、Kimi旧入口308、百度/Step页为轻量壳。已用其他官方入口/一手搜索补充，但不据此宣称这些机构无发布。IFM两个rollout数据卡/commit再次401，仍未关闭旧待办；部分NVIDIA物理AI数据卡401。FairStress、MemT、Glimmer、d3的既有结论未复写。完整请求、失败、候选去留与未审清单见`models-coverage.json`。

### 访问失败、日期冲突与未审范围

- OpenAI研究/新闻与Microsoft研究入口直连403，xAI news403；部分国内厂商页面是JS/轻量壳，Kimi旧入口308。IFM两个rollout数据卡/commit401，NVIDIA部分物理AI数据401，Dory研究入口403；替代官方入口只补已核事实，没有关闭全渠道缺口。
- arXiv初轮429/timeout经顺序重试恢复200，但lastUpdatedDate查询仍被feed改写成submittedDate，不能用此枚举全部版本更新。当前保留**1,424条未审回溯metadata、291项未审replacement**及各自ID清单；这些含无关/可能有价值候选，未全部语义判断，不能当全部遗漏成果数量。PDF/HTML差分未覆盖图片像素或打包资产变化。
- 扩展工程最初NVIDIA/dynamo地址404，已改用ai-dynamo/dynamo并完成库存；kai-gpu-fractioning旧API返回迁移后，当前connector的numeric redirect端点不可用，保留访问待核。Ascend bulletin HTTP200仅壳，CANN/HCCL未据此判扫描完成；libtpu/Neuron更早实现区间仍欠账。
- Google Cloud A5X聚合页搜索结果为10-21、实际正文为10-22，且写GA“下月”；不把日期冲突条目收成窗口内GA。Run:ai旧10-13技术文已10-14归档，DSX设施集成缺少本轮可复现LLM执行证据，不重复或强行收录。
- 未完成的旧区间继续保留：核心仓10-05T00:00Z→10-08T00:00Z的选择性审阅、扩展工程10-05→10-12T01:55:47Z，以及arXiv 10-15T01:55:47Z→10-16T01:00:40Z和10-13→10-18历史修订枚举/语义缺口。旧319/328批次、先前CV/stat候选与正式会议逐volume检查不因本次API恢复而自动完成；完整28条原待办及本轮新增待办写入state。

### 核心发布、未合并依赖与connector旧待办

六仓公开release collection已逐页到短页，**主窗无新release发布**；72h内[vLLM v0.32.0](https://github.com/vllm-project/vllm/releases/tag/v0.32.0)（10-18发布）与[SGLang v0.5.22](https://github.com/sgl-project/sglang/releases/tag/v0.5.22)（10-19发布）均已在10-20日报报道，不把今天main合并写成这些release已含。Megatron本窗core_r0.20.0的Pillow版本变更、SGLang非main分支的LoRA/cherrypick/构建/weight-update不当成main能力；TorchTitan0.4.0rc1仍只是release分支版本提交，公开release collection最新v0.3.0。

**P2P consumer寿命继续待办。** 本轮重开[issue61532](https://github.com/vllm-project/vllm/issues/61532)与[PR61667](https://github.com/vllm-project/vllm/pull/61667)：均open，PR merged_at=null，head仍`4fd16ddb155fd0f7e676dece9abd6aa9209b2b1a`。截止窗内main `30aded62d46dfcf52c815925a8acec5a8bebddc6` 的 `vllm/v1/kv_offload/tiering/p2p/session/client.py::ClientRole.collect_results` 仍在10s abort-ack timeout后发布success=False并移除load；本地超时不能证明远端WRITE静默。新MoRIIO/SGLang完成边界修复不关闭这个connector的quarantine、RDMA/serving验证缺口。

**覆盖边界。** 六仓all-base merged PR主窗176条（9/69/4/94/0/0），其中main171；72h共703条、main689。main commit-time库存698条，主窗171；commit时间与merge事件时间独立。release collection共254项全部分页到短页。正文核查57项PR、合并为49条机制/实践条目；这不等于全部库存语义审阅。未选实质候选、10-05→10-08等历史缺口及两个未合并依赖保留在core-coverage.json；当前抓取无失败，全球完整覆盖水位不推进。

### 补录、更正与下轮续查

本日报只记本轮新证据：Voxtral3和22篇此前未收论文明确标原始日/补录；TorchTitan三项旧窗修复有merge日期；模型super-squash、卡与权重更新分开。TokSuite v2相关方向与统计口径更正已单列，不改写旧日报假装原结论从未出现；Beam卡分数更正、FlashAttention beta36包含边界与AITER测量协议均保留说明。新模型参数、性能与许可证仍以指定工件版本及披露条件为准。

原始请求、全文/patch、候选去留、稳定版本和独立复核记录保存在本地 `.monitor/research-2026-10-22/`，四路覆盖账本加Flint补充账本汇总到`.monitor/state.json`与`.monitor/sources.json`。本地保存与远端提交/回读状态分别记录；正文不会因上传失败在下一轮重复生成。下轮从**2026-10-22T00:58:39Z**续扫，回溯72h并续原未完成区间；全球完整覆盖水位仍为null。本轮不宣称全面监控。
