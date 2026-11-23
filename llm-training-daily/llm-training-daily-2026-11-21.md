# LLM 技术动态日报 · 2026-11-21

> 时区 Asia/Shanghai（UTC+8）。增量扫描窗口 **(2026-11-20 09:02:12, 2026-11-21 08:58:21]**，向前回溯 72 小时至 **2026-11-17 09:02:12**。实际生成 **2026-11-21T09:18:45+08:00**。来源仅有日期时保留原日期，提交、发布、会程与发现时间按各自证据记载。

文中 KV 为键值缓存，MoE 为混合专家，SFT 为监督微调，RL 为强化学习，GRPO 为组相对策略优化；TP／PP／CP／DP／EP 分别为张量／流水线／上下文／数据／专家并行，TTFT 为首 token 时延，TPOT 为单输出 token 时延，SLO 为服务目标，A2A 为全交换通信。

## 本次重要变化

- **循环架构有新交付，也有明确计算账单。** Inkling 2 Mini开放主模型与可变循环研究权重；约26.4B总参、一次独立激活约3.4B，计重复访问约5.4B/token。发布图中的速度是理论带宽上限，需按真实上下文、精度与并发复现。腾讯WeMM-Video同时进入多尺度有声视频权重交付。[Inkling官方](https://thinkingmachines.ai/news/inkling-2-mini/) · [WeMM固定卡](https://huggingface.co/tencent/WeMM-Video-35BA3B/blob/4375278c69e03c6ee3af6d34124c2888417cdbe4/README.md)
- **先检查会静默改变输出或训练的状态错误。** SGLang修复分块权重更新跨chunk遗失融合参数半边，非session接口仍有失败上报缺口；ATOM保留KV零填充块以隔离原始递归状态；SGLang TPU补齐缓存页写回等待。优先对照自己的权重更新与缓存生命周期。[SGLang权重](https://github.com/sgl-project/sglang/commit/7894bdf172df633b9770c06c07842eb92cb0c695) · [ATOM](https://github.com/ROCm/ATOM/commit/48537d30e940e63862e970d403279952d6c76a20) · [TPU写回](https://github.com/sgl-project/sglang-torchtpu/commit/4638039b577c)
- **后训练token预算可能改变测试时扩展能力。** Budget Internalization多种子主实验显示，低预算改善可伴随高预算退步，不能只在训练上限附近比较。TorchTitan同日把截断奖励改为保留部分得分的penalty，尚无新训练效果证明，两者适合一起审查本方奖励和评估曲线。[预算研究](https://arxiv.org/pdf/2611.26172v1) · [TorchTitan](https://github.com/pytorch/torchtitan/commit/1fdf04562e7bb02d8df9202a269668b43c42f503)
- **推理研究优先看完整负载与质量。** FrontierReplay给混合架构构造精确前缀状态；NariQ用Agent自身轨迹校准低于3bit权重。两者有具体复现工件/条件，但前者高倍数出现在接近失效的基线，后者不能跨不同Agent harness排名；稀疏注意力与INT4缓存还存在短窗减速和长窗质量回归。[精确缓存](https://arxiv.org/html/2611.26020v1) · [NariQ](https://arxiv.org/html/2611.26157v1)
- **正式版升级与新主线优化要分别验收。** TensorRT-LLM1.3移除旧TensorRT引擎/C++ Executor；Foundry图持久化进入SGLang主线，但要求固定恢复环境，部分gpt-oss decode可能慢约9%。Megatron MoK公开性能对照另有MRoPE遗漏，不能当正确模型的端到端收益。[TRT正式版](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.3.0) · [Foundry](https://github.com/sgl-project/sglang/commit/53b8e56e4a065af5993ade615f0dc434947038e2) · [MoK](https://github.com/NVIDIA/Megatron-LM/commit/87db06d436df8be21cf4605150164eab4b46b9cd)
- **评估分数要拆成它真正测到的能力。** MedChemBench的结构识别与解释占大部分rubric，不能把整体高分当药物候选决策已可靠；多项概率校准与judge研究提供负面结果，值得先读对照及限制。[MedChemBench](https://openai.com/index/introducing-medchembench/) · [Bradley–Terry负面结果](https://arxiv.org/pdf/2611.25594v1)

[模型与技术报告](#topic-1) · [预训练与架构](#topic-2) · [后训练](#topic-3) · [推理与部署](#topic-4) · [开源工程](#topic-5) · [硬件与软件协同](#topic-6) · [评估与可靠性](#topic-7) · [覆盖与续查](#coverage)

<a id="topic-1"></a>
## 模型与技术报告

### Thinking Machines：Inkling 2 Mini 把循环深度、原生低精度和开放研究权重一起交付

- **时间与版本**：官方11月20日发布；主模型固定 `ed40e7da625f594fc9c4990f5f66be24873c22d7`，研究变体 `657b53ba5da7baefe9ff15dca5cd25f5bd3dbaa2`。两张官方卡声明Apache2.0；本轮固定根目录`LICENSE`返回404，未把卡声明写成已审完整许可文件。
- **技术增量**：24层分四个6层block，中间12层共享权重运行三遍，共48次层应用；128专家选8、隐藏维2304、1,024滑窗与全注意力混合，config上下文字段为1,048,576，模型卡声明支持最高1M。总参约26.4B、一次独立激活约3.4B，**计重复访问后每token约5.4B权重激活**；不能据3.4B把循环计算量忽略。训练中从384专家剪掉三分之二，再用较大内部模型蒸馏1T tokens恢复；SFT后RL披露2.2T tokens/2,000 steps，并含on-policy distillation（在线策略蒸馏）对齐。
- **精度与系统边界**：作者称前20层专家从预训练至RL使用NVFP4、最后4层MXFP8，剩余参数BF16；约20GB checkpoint，建议至少32GB显存/统一内存。1M上下文能力与基础载入内存应分开预算，缓存随token与循环访问增加。HF集成博客的SGLang示例要求SM103、FA4，DSpark示例设128K；框架当日支持状态另见工程条目，不把厂商“支持”一词当各正式发行包已包含。
- **证据与局限**：同mini-swe-agent2.4.6下DeepSWE v1.1为48.2%，Qwen3.8-27B42.2%；但TerminalBench2.1为71.6%对73.0%，复杂约束与知识任务也未全面领先。发布图的“速度”是**614GB/s、BF16权重、batch1的理论上限**，不是20GB原生低精度模型实测吞吐。预训练完整配方、总算力及数据清单未披露。官方博客与固定HF卡均为26.4B；本轮只采用官方工件口径。
- **建议深入读并考虑复现**：重点是循环带来的计算/内存/生成token三者权衡，以及剪枝后恢复与训练、部署同精度；按实际长上下文和并发测性能。[官方发布](https://thinkingmachines.ai/news/inkling-2-mini/)；[固定模型卡](https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/ed40e7da625f594fc9c4990f5f66be24873c22d7/README.md)；[固定配置](https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/ed40e7da625f594fc9c4990f5f66be24873c22d7/config.json)；[HF集成说明](https://huggingface.co/blog/inkling2mini)。

**随附TML-d24-varloop-research是独立研究工件。** mid-training以1:2:6:2:1比例采样1—5次循环，RL只在3次完成；可用同权重控制深度，但不同于主模型后续训练，不能直接把二者成绩差归因于循环。作者在Think Fast子集上测得无思维链（CoT）50%可靠性的人类时间跨度由1遍16秒增至3遍37秒；GPQA正确于所有循环数的子集上，5遍思维链由3,703缩为1,661 tokens且文本仍可读。LinuxArena用184组主/副任务、1%误报率门槛的较弱monitor，未见随循环数增加而明确增强的隐蔽攻击趋势；这是小规模、特定训练与监控实验，**不是证明潜在推理总可监控**。研究卡明确无Tinker/API、不建议生产使用，1遍DeepSWE仅0.7%对3遍47.4%，减少循环并非免费加速。[研究卡](https://thinkingmachines.ai/model-card/tml-d24-varloop-research/)；[固定研究配置](https://huggingface.co/thinkingmachines/TML-d24-varloop-research/blob/657b53ba5da7baefe9ff15dca5cd25f5bd3dbaa2/config.json)；[固定研究权重卡](https://huggingface.co/thinkingmachines/TML-d24-varloop-research/blob/657b53ba5da7baefe9ff15dca5cd25f5bd3dbaa2/README.md)。

### 腾讯 WeMM-Video：从空仓进入全模态权重与复现管线交付

- **时间版本**：报告 `2611.26409v1` 提交11月19日；本次新增是11月20日2B/4B/9B/35B-A3B及FP8、AWQ、NVFP4、GGUF工件实际上传。仓创建11月17日不等于当时已发布：本轮回读前窗快照README为404。35B-A3B固定 `4375278c69e03c6ee3af6d34124c2888417cdbe4`，官方代码基线 `Tencent/WeMM-Video@6cefad232b94e3e190d747251f35799b8b142213`。
- **问题与方法**：以Qwen3.5小Dense和Qwen3.6-35B-A3B为主干，保留视觉编码器，加Whisper音频编码器及窗口Q-Former。16kHz音频按30秒分块，0.1秒压成一个音频token，与带时间戳视频帧按时间交织。报告§3披露约1.2T多模态tokens、45.4%联合视听；三段预训练先仅对齐音频，再解冻除视觉编码器外参数，最后扩窗。后训练用多教师OPD保留音频同时增强文图，再以离线重复坏例做多轮DPO（每轮LoRA合并重置）稳定解码，最后按任务路由规则/量表/教师一致性奖励做RL。
- **可定位实现**：固定HF的 `processing_wemm.py::WeMMProcessor.__call__` 明确区分`video`与`video_audio`并把音视频输入交给预处理；`modeling_wemm_moe.py::Qwen3_5MoeModel.get_audio_features`分批处理音频特征，`get_placeholder_mask`校验模态token与特征数。config对应40层、256专家选8、3线性注意力+1全注意力循环；这只核推理路径，不构成训练算法复现。
- **评测与边界**：§5/Table1的35B Thinking在Video-MME-v2有声54.9，对其Instruct47.5、Seed2 Mini51.7、Gemini3 Flash63.2；FutureOmni却由Instruct58.4降至Thinking56.2。原表混用对手官方结果与作者重测，部分任务依赖不同模型裁判，不能概括所有模态和思考模式均更好。35B BF16文件约67.39GiB；“3B激活”不等于只需3B存储。
- **部署限制**：官方锁定patched vLLM0.18/0.31与patched SGLang0.5.16/0.5.19，NVFP4用ModelOpt Mixed+Marlin、SM120参考；默认示例8K且关prefix cache。config的262,144不能替代长窗服务验证。模型卡称H20完成生成检查，固定serving文档却仍写“从Hub下载后GPU生成待完成”，本轮未消除此验证范围冲突。Apache2.0许可文件已读，第三方组件保留原条款。训练完整数据、算力与量化质量对照未完整披露。
- **建议深入读/定向复现**：视频原声的时间对齐、反重复稳定化与三阶段后训练具有实际参考价值，先按锁定环境重建并核验有声/无声、长视频、各精度质量。[论文正文](https://arxiv.org/html/2611.26409v1)；[固定模型卡](https://huggingface.co/tencent/WeMM-Video-35BA3B/blob/4375278c69e03c6ee3af6d34124c2888417cdbe4/README.md)；[固定服务指南](https://github.com/Tencent/WeMM-Video/blob/6cefad232b94e3e190d747251f35799b8b142213/docs/serving.md)。

### 百度 HiUni：共享视觉编码与Dense主干，输出侧补像素细节

报告`2611.26107v1`提交11月19日，本轮11月20日发布HF与Megatron格式权重，固定 `47cbef4bb6a155cd40c07b7c36c961632ad21345`。以Qwen3-VL-4B-Instruct共享ViT和4B Dense主干，另加0.9B Pixel Diffusion Head与HiGate，让理解、生成、编辑共用语义表示，图像直接在RGB像素空间扩散；HiGate向输出头补patch merger损失的局部细节。§4.1/附录A披露30K步冻结主干热身、310K步联合预训从256²到2048²、22K步SFT加入编辑；训练最大多模态序列8,192。完整训练数据未随权重交付，总算力未明确。

Table1的七项理解平均75.56接近初始化官方75.61，但OCRBench74.5对88.1、MMBench80.7对85.1，不能写“所有理解能力无损”；同设置重测Thinking基线仅四项胜出。Table2/3的生成/编辑比较包含不同来源结果与模型裁判，表明能力共存，尚未证明稳定跨任务增益。固定实现 `modeling_pixel_dit.py::PixelDiTForConditionalGeneration.predict_image`确实复用`model.visual`/`model.language_model`再进入10层diffusion head；`HiGate.forward`用两层MLP预测**无sigmoid/tanh约束**的门，对应附录A.2，不能按0—1加权解释。

交付主模型为EMA0.95、每50步更新到最终SFT8,000步，Base不含合成数据。Megatron文件**只有模型参数**，不含optimizer/RNG/dataloader，不能原状态续训。固定HF根LICENSE404，项目LICENSE回读也未成功；权重与代码许可范围均继续核实，报告不自行推定许可覆盖。**建议深入读并考虑复现共享计算消融**，尤其逐项理解退化、编辑监督迁移及高分辨率代价。[论文](https://arxiv.org/html/2611.26107v1)；[固定模型卡](https://huggingface.co/ernie-research/HiUni/blob/47cbef4bb6a155cd40c07b7c36c961632ad21345/README.md)；[固定推理实现](https://huggingface.co/ernie-research/HiUni/blob/47cbef4bb6a155cd40c07b7c36c961632ad21345/modeling_pixel_dit.py)。

### Baseten：bison-1 与groundhog-1把128K决策做成一次共享前缀prefill

11月20日权重上传；bison31B固定 `bf8350b2c59eb6c36e95d02a2af870aec18724a7`，groundhog26B/约4B激活固定 `f4602c43599114a7af2c158472b59a4ff2d13663`。分别基于Gemma4-31B/26B-A4B，文图输入、每请求一个state加多问题，输出选项概率而不生成文本；全注意力层改block-bidirectional、滑窗层保持因果。6.71M决策样本混合、交叉熵+Brier校准项λ1，约1/7样本是概率目标，含“I don't know”；128K、BF16训练分别192/128张B200、3,747/5,621步一epoch，视觉编码器/norm/embedding冻结，MoE再冻结router。

卡引用11月19日AA Decision Index v2.1的bison76、Brier0.14；groundhog未上该榜。作者LongMemEval M/S在128K分别84.8/87.1与77.4/79.3；这些只按模型卡归因，本轮未独立复评AA完整榜单。训练包含公开benchmark集合，缺少逐项污染审计，不能把榜单差值全归因于机制。两者LICENSE明确Baseten修改MIT+基座Apache2.0，并同步纠正旧mammoth/elephant/joey卡的单一MIT口径。**建议考虑复现**多问题共享prefix与概率校准，先在业务分布验证置信度及公开训练集重叠。[bison固定卡](https://huggingface.co/baselabs/bison-1/blob/bf8350b2c59eb6c36e95d02a2af870aec18724a7/README.md)；[groundhog固定卡](https://huggingface.co/baselabs/groundhog-1/blob/f4602c43599114a7af2c158472b59a4ff2d13663/README.md)。

### Red Hat：Mistral Large 4混合MXFP4/FP8工件附质量代价

11月20日固定 `8f3dc99221bd542c78cc27734b7395dd7cd9a411`，此前卡为WiP，本轮补充正式模型卡与评测。MLP/专家权重MXFP4+动态FP8激活，余下组block FP8；作者称checkpoint约575GB对上游FP8约1.06TB、存储约减46%。GPQA Diamond91.92→89.90、SWE Verified77.0→75.4、MMLU-Pro87.10→87.18；基座运行8×B300、量化8×H200，Inspect AI/vLLM，**没有同硬件吞吐对照**，不能从文件缩小推出同幅度服务显存或性能收益。量化源脚本/配方未提供，卡只有`license:other`且根LICENSE404。**继续观察，具备许可与部署条件时考虑质量复现**；先读日志分支和长上下文配置。[固定卡](https://huggingface.co/RedHatAI/Mistral-Large-4-1T-A52B-MXFP4xFP8/blob/8f3dc99221bd542c78cc27734b7395dd7cd9a411/README.md)。

### LlamaIndex OCRia-1.5（11月18日，补录）：2B/4B文档解析升级

原始官方日期11月18日，本轮发现，按补录而非今日新模型。OCRia1.5-4B进入LlamaParse Agentic，2B进入Cost Effective；在厂商ParseBench-2上表格分86.5→90.1/83.2→87.5，chart74.9→76.8/67.3→73.0，小档visual grounding77.0→88.5。该基准由同厂商制定，服务pipeline分数不能直接视为裸模型对比；升级文章未给硬件、延迟、训练过程与权重交付。**建议继续观察/以真实PDF做相同配置A/B**，表格跨列标题与bbox是RAG数据质量的具体收益点。[原始正文](https://www.llamaindex.ai/blog/introducing-ocria-1-5)；[基准定义](https://www.llamaindex.ai/blog/introducing-parsebench-v2)。

<a id="topic-2"></a>
## 预训练与架构

### Canon Layers at Scale：64次大规模对照支持局部混合，但不是今日新架构（2611.24963v1）

**时间与版本**：v1提交2026-11-12 18:51:19 UTC；11月20日分类公告恢复／补录；代码与结果更早已公开，首次公开秒界未确定。

Allen-Zhu Research／实验在 Meta FAIR 完成。**补录：固定版本提交11月12日；本轮从11月20日分类公告恢复**。正文注明16个Transformer权重于2025年7月发布，全部64模型配方／结果于2025年12月已公开，2026年1月已演讲，增量是整理成完整报告。1.3–8B、Nemotron-CC 1–2T tokens的64次从头训练，逐架构扫学习率、对齐线性模型参数分配与递归状态。Canon四位置短卷积加部分RoPE的组合，在8B/1T将MMLU从60.6提到62.4；按训练曲线达到原模型终值约省28% tokens。**限制**：各表可取不同学习率最好值，未重复多随机种子；“30%”针对MMLU而非等墙钟全能力；线性模型与Transformer的小规模训练窗口／步数不同。合成五事实检索在500背景token就暴露线性模型弱点，NIAH单针几乎满分却看不到。建议深入读并复现自然事实检索与局部卷积消融，勿把混合配方收益只归于卷积。[正文§2–5／Table1／Fig3、8](https://arxiv.org/html/2611.24963v1)

### HiSA：沿RoPE特征维裁计算，训练吞吐与运算量不同比例下降（2611.26363v1）

**时间与版本**：v1提交2026-11-19 16:31:46 UTC；本轮11月20日分类公告恢复，首次公开秒界未确定。

项目HiSA，Ambroise Borbely等。用token距离和旋转位置编码（RoPE）频率确定每对Q/K特征的有效窗口，保留完整投影但省略大旋转角的点积部分；需按该稀疏规则训练或持续训练。135M／360M、OpenWebText、单H200的证据中，135M 4K的Flash-HiSA只算34.8% QK乘积，吞吐238k对dense FA3的236k tokens/s；16K扩窗后只算21.7%，却为135k对143k，**训练反而慢约6%**。16K／batch128解码报告快42%、KV约小三分之一；这是该小模型和自定义缓存布局结果。建议复现特征窗口、cache布局与全前向一致性；运算量节省不能代替端到端吞吐或大模型质量证据。[正文§3–5／Tables2–5／AppendixB–D](https://arxiv.org/html/2611.26363v1)

### dLoop：给扩散模型未提交位置更多内部计算，代价必须计入（2611.26275v1）

**时间与版本**：v1提交2026-11-19 15:40:24 UTC；本轮11月20日分类公告恢复，首次公开秒界未确定。

项目dLoop，Handa Li等。在LLaDA-8B-Instruct的32层中重复16–21层，使用多深度SFT与prelude融合，再执行置信度提交。Table1中GSM8K为76.3%／调整后每前向生成6.76 tokens，native为72.6%／1.00；d3LLM为72.9%／9.06，存在准确率与并行度交换。调整后的TPF把额外6层按6/32计入；GH200单请求实测每步额外环路代价与估计接近，GSM8K相对dParallel每题1.38对1.71秒。**评估限制**：AppendixA用GSM8K前400道测试题挑loop块，相关GSM8K结果不能视为未触碰测试集；单模型，延时核验仅单GH200／batch1和选定解码设置。建议复现新的留出集和真实服务时延，再判断是否值得引入环路结构。[正文Experiments／Table1、AppendixA](https://arxiv.org/html/2611.26275v1)

### LLaDA-LED：扩散编辑从替换扩到插入／删除，收益有明显任务边界（2611.23750v1）

**时间与版本**：v1提交2026-11-18 03:05:15 UTC；11月19日公告的候选补录，首次公开秒界未确定。

项目LLaDA-LED，Shuaidi Wang等。**补录11月18日提交、19日公告的未完成候选**。在LLaDA2.1-Mini Base上用同样745K SFT样本比较replacement-only与Levenshtein编辑，后者能修复长度错位而不改写整个后缀；训练3.7K H200 GPU小时，默认block32、最长32,768输出。12基准均值54.86→57.10，TPF6.14→7.05；但AIME25由30.00降到24.01，BFCL与IFEval也退步。去置信度筛选时均值27.52→47.19显示编辑可恢复较差草稿；不等于最好单样本设置。block256时两者都明显崩坏。建议复现代码／数学的长度错位与吞吐，保留逐任务回归门槛。[正文§4–5／Tables1–2](https://arxiv.org/html/2611.23750v1)

<a id="topic-3"></a>
## 后训练

### Fastino GLiDE：旧模型新增报告，区分蒸馏、权重平均与循环深度的贡献

`2611.26509v1`提交11月19日；五档模型卡11月20日加入论文，不是重新发布全部权重。§2/§3把Qwen3.8-27B的16层全注意力改双向、48层Gated DeltaNet保持因果，在最终位置用255-way head输出候选概率。多问题实际为**各自独立序列**，每问题重新读全部context，不具备Baseten共享prefix机制。教师直接预测置信度<0.8才平均有/无思考分布，监督为gold CE权重0.3+教师KL0.7；最终V1.2阶段374,720样本、8×B200每seed约6小时，完整扩展lineage约3.9B tokens/3.4M问题。

§6/Table8的GLiDE27B把两段中间12层各运行两次、阻尼0.5，不加新参数；73.37对V1.2的71.52同时包含**checkpoint averaging和循环**，不能全算循环收益。单独held-out循环增益0.69/0.56点，单卡RTX PRO6000中位延迟92.9→122ms，ECE2.0→2.1%；三次循环反而差。§5.3转换消融仅单seed且改进落在seed变异内。权重为Fastino Community License及第三方条款，不因论文CC BY4.0写成Apache权重。**建议深入读并复现**，这是可校准决策蒸馏与推理计算预算的具体分解；不要用不同日期榜单直接比较名次。[报告正文](https://arxiv.org/html/2611.26509v1)；[本轮固定27B卡](https://huggingface.co/fastino/GLiDE-27B/blob/061c7d4312670f98261c007f5fd24851a64124dc/README.md)。

### OpenAI：精确推理轨迹蒸馏有预算依赖，提取防护同时改训练和API边界

11月20日官方研究文章：同预训练student从Astra级教师的合法编码/防御网络任务轨迹做SFT，再做轻量RL；精确CoT学生SFT后通常胜过summary学生，但RL后**只在最大测试推理预算仍更高，低预算summary更高**。无student规模、数据规模或开放复现实验，不能概括所有模型的能力转移或量化风险幅度。

新版防护同时含对抗训练、输入/输出classifier（覆盖streaming与tool-call参数）及API历史绑定；GPT6.2的受保护推理只在原历史后续可复用，改写早期消息后丢弃后续推理，并阻止旧弱防护模型读取。六个攻击模型聚合的自适应黑盒测试：提取前次请求ASR54.0→0.2%，当前请求22.9→0.9%，对照是**GPT6原防护与GPT6.2新防护整体**，不能分离模型与防护贡献。**建议深入读蒸馏预算条件与API兼容影响**；不把内部红队低ASR当普遍安全保证。[原始文章](https://openai.com/index/combatting-adversarial-distillation/)。

### IFM Feedback Shortcut：36套checkpoint与计时runner修订可复核

11月20日HF固定 `2077e693dbcd95fc3c7e88a67b1616868217575d`，官方`ifm-ai/LoRM@1367e7a46b17d8ad51852feae319084b82d93379`。提供135M/780M×六设定×三seed的FP32权重、配置和训练验证日志，针对latent feedback模型投机位置缺失上一步真实隐状态的问题比较zeros、MTP-state、state alignment与in-place替代。135M/780M训练分别读13.4B/77.8B Dolma3 tokens、全局256×2,048序列；测量通常单GPU FP16。

关键复现差异在固定README：论文原Spec-Bench runner把捕获CUDA graph放入初始计时请求，新`measure/specbench.py`先捕图、交错对照；**生成token/acceptance不变但新runner测得speedup高3—7%**。缺item-level下游得分，完整下游置信区间不能从`results/downstream.json`独立重算；Tokenizer访问另受Llama2门禁。alphaXiv链接是作者发布入口，不能编造为arXiv正式ID。**建议考虑复现**，先固定计时版本，再检查served BPB/采样KL及贪心near-tie，不能只看吞吐。[固定权重卡](https://huggingface.co/IFM/feedback-shortcut-checkpoints/blob/2077e693dbcd95fc3c7e88a67b1616868217575d/README.md)；[固定代码与计时说明](https://github.com/ifm-ai/LoRM/blob/1367e7a46b17d8ad51852feae319084b82d93379/works/feedback_shortcut/README.md)。

### TRL AsyncGRPO跨库benchmark：先读原始日志、草稿状态和比较边界

11月20日公开`trl-lib/asyncgrpo-stable-bench-runs@a119c671db31895d39f12060cac39e0af09f345f`，含各run固定库版本、config hash、step/eval、GPU/显存及部分episode/span日志。Space固定`cc714a00a627d5c0403ebd00b2afa20ec9920712`的README仍标**TRL1.18发布草稿**，代码在`asyncgrpo-benchmark`分支；不能据草稿文字把全部能力称为已正式release。

实验为每run单节点8×B200、Qwen3.5-2B/9B/35B-A3B，数学、terminal agent及text-to-SQL，每配置一seed。作者说明不同库未全跑通，TRL可在实验中修bug、其他库按as-shipped比较，存在不对称。草稿按useful token得出数学TRL与prime-rl接近，但TRL每step约双倍tokens，prime-rl到相同reward墙钟约半；**每step速度、每token吞吐与time-to-reward不可互换**。本轮抽读miles run meta也确实标`patched library ... excluded from the as-shipped comparison`，不能把所有有日志run混成有效榜单。**建议深入读并复现本方负载**，优先核混合架构packing状态、训推logprob及各库失败复现；本轮未重跑图表或确认草稿所列全部上游issue。[固定原始数据卡](https://huggingface.co/datasets/trl-lib/asyncgrpo-stable-bench-runs/blob/a119c671db31895d39f12060cac39e0af09f345f/README.md)；[草稿入口](https://huggingface.co/spaces/trl-lib/asyncgrpo-benchmark)。

### Budget Internalization：固定rollout预算会学进策略，低预算胜出可能损失高预算能力（2611.26172v1）

**时间与版本**：v1提交2026-11-19 14:20:41 UTC；本轮11月20日分类公告恢复，首次公开秒界未确定。

JetBrains Research，Alexander Kovrigin等。以三个模型家族在DeepScaleR和Reasoning Gym上做固定预算GRPO，主DeepScaleR/AIME组18次训练、三个种子，并同时检查选定与终态检查点。Qwen3.5-4B的4K训练模型相对base在4K多34.8个百分点，到64K却少30.2点；额外允许token也很少被利用。强制base提前答题明显改善低预算成绩，强迫训练模型继续则未恢复原来的长预算曲线。**限制**：GPQA／代码的损失较小，部分点还变好；仅主组多种子，不能把数学结论推广所有任务。建议优先复现截断奖励、成功轨迹长度与多预算评估；只报训练上限附近成绩会掩盖这一变化。[PDF§2–5／Table1–2、AppendixI](https://arxiv.org/pdf/2611.26172v1)

### RLCDR：校准决策概率可以当奖励，但相对同骨干softmax的策略优势未证实（2611.25135v1）

**时间与版本**：v1提交2026-11-18 20:15:18 UTC；本轮11月20日分类公告恢复，首次公开秒界未确定。

FuturixAI。27B决策模型对回答是否正确／首错步骤输出一次前向概率，平均选项顺序并用小量标签校准；GRPO训练Qwen3-1.7B 100步。MATH500 base72.2%、random72.8%、同骨干softmax75.5%、RLCDR76.8%、答案键76.9%；RLCDR对base／随机显著，但对softmax差1.30点，95%区间[-0.50,+3.10]，**不能宣称策略更强**。27B奖励训练1.7B策略、单种子／数学域；hard分布过度自信，8,192奖励上下文和3,072输出限制使AIME严重截断。模型Apache2.0已声明，训练／服务与逐题代码写为将发布。建议深入读校准和随机奖励控制，再做三种子与更大策略复现。[正文§3–6／Tables4–6、Availability](https://arxiv.org/html/2611.25135v1)

### CARE：跨问题奖励直接学低风险覆盖，平均校准好不保证能可靠拒答（2611.25292v1）

**时间与版本**：v1提交2026-11-18 23:41:15 UTC；本轮11月20日分类公告恢复，首次公开秒界未确定。

项目CARE，Hector Garcia Rodriguez等。把一批问题的confidence与正确性一起形成拒答阈值，奖励每个回答位于阈值的正确侧，并加置信度多样性；区别于独立将正确推1、错误推0。Qwen2.5-7B在HotpotQA训练后，5%风险下ID覆盖由RLCR34.7%到74.7%，OOD六类均值12.3%到23.6%。**限制**：覆盖是测试集经验风险曲线，不是线上分布保证；Gemma4-E2B某些OOD点反而退步，例如HotpotQA训练的C@5为7.8→5.0；较大历史buffer也会引入off-policy陈旧样本。建议复现按风险分层的覆盖和漂移监测，不能只看ECE或一项均值。[正文§3–5／Tables2–5](https://arxiv.org/html/2611.25292v1)

### BGPO：单个校准概率可遮蔽人群内部不同信念（2611.25168v1）

**时间与版本**：v1提交2026-11-18 20:48:01 UTC；本轮11月20日分类公告恢复，首次公开秒界未确定。

项目BGPO，Zirui Song等。策略先生成潜在belief列表及权重，posterior用带标签观测更新，再预测留出选择；通过held-out log score和复杂度罚项训练，推理不要求人口标签。Qwen3.5-2B／4B、五个文字／图像多标注者数据集、三种子，报告相对最强基线conditional JSD文字最多降44%、图像最多降12%；合成CLEVR另检查已知belief的恢复。**代价与边界**：每问题每训练步约13.6K生成tokens，基线约4.1K；真实数据belief不可观察，预测分布相同不证明心理解释唯一。适合有群体异质反馈的决策／偏好训练复现，不能当免费替换标准RLVR。[正文§3–4／Tables1–3](https://arxiv.org/html/2611.25168v1)

### DFRL：用下游决策后悔值训练预测器，最小预测误差不等于最低决策成本（2611.25597v1）

**时间与版本**：v1提交2026-11-19 06:44:35 UTC；本轮11月20日分类公告恢复，首次公开秒界未确定。

项目DFRL，Haotian Zhai等。LLM输出优化参数，下游求解器给负决策regret作为RLOO／GRPO奖励；另用近似奖励热启动、随机精确校正换取更便宜但方差更大的更新。H&M交易构造的4,013训练／1,251测试库存场景，Qwen3.5-4B由SFT的regret234.01到RLOO172.72；加额外reasoning-trace SFT与thinking后160.53，比SFT低31.4%、XGBoost低12.4%。**限制**：最好配置增加了GPT-6-luna生成、含真实销售条件的解释监督，31.4%不能全部归于奖励替换；三个合成优化实验使用仿射预测器，并非大模型扩展证据。建议复现纯RLOO与数据／预算匹配对照，面向可求解的决策闭环采用。[正文§3–5／Tables1–2](https://arxiv.org/html/2611.25597v1)

### 最小文本编辑训练：监督首个失败点能保留能力，在线版对新指令未必更好（2611.26342v1）

**时间与版本**：v1提交2026-11-19 16:18:25 UTC；本轮11月20日分类公告恢复，首次公开秒界未确定。

项目edit-based training，Alexandra Yakovleva等。对模型自身轨迹反复找首个指令违规，只监督局部替换span，其余续写配KL约束；编辑器由脚本和GPT-5组成。Qwen3-32B三训练种子，ReasonIF域内online97±1%、offline87±1%、全轨迹SFT40.3%；IMO-AnswerBench分别59、57、61，对base61。**限制**：未见指令online仅34±9%，offline68.4±0.4%；online并非全面优于offline。六类主要为格式／语言／字数代理约束，不能直接当一般数学推理提升；IMO用GPT-5.4裁判。建议复现精确首错定位、编辑预算及未见指令，而非只看域内97%。[正文§2、§4／Tables1–2](https://arxiv.org/html/2611.26342v1)

### CurvDelta：把参数更新限制在保持集的低曲率空间（2611.24654v1）

**时间与版本**：v1提交2026-11-18 15:56:20 UTC；11月19日公告的候选补录，首次公开秒界未确定。

项目CurvDelta／CurvLoRA，Zeguan Xiao等。**补录11月18日候选**。用Kronecker分解近似保持损失曲率，直接参数化低曲率delta，避免每步仅投影梯度后累积越界。Gemma-3-1B、Llama-2-7B在100K数学／104K代码样本适配，保持集仅256条FineFineWeb；Llama代码任务CurvDelta下游均值40.5、知识保持44.4，对FT38.5／40.4。**限制**：数学任务Llama的全指标均值39.0低于Replay40.2，不是全域最优；局部曲率与小保持集不能保证分布外能力不变，TOFU／MUSE的遗忘指标也不构成数据删除证明。建议复现能力取得—保留曲线与曲率估计成本。[正文§3–5／Tables1–3](https://arxiv.org/html/2611.24654v1)

### VETO：把视频微调的文本捷径与真正视觉增益分开（2611.25776v1）

**时间与版本**：v1提交2026-11-19 09:10:24 UTC；本轮11月20日分类公告恢复，首次公开秒界未确定。

项目VETO，Youngwoo Shin等。先以text-only激活标定子空间，每次更新投影到其正交补，降低无视频也能答对的训练捷径。七个VideoLLM、blind-filtered SSv2 57K／16帧／rank16 LoRA、单B200；Qwen2.5-VL-7B在Perception Test full-input为64.3±0.7%，SFT61.1±0.7%，而text-only增量由17.0降到4.9点。**边界**：这是过滤后选择题与特定训练配方；子空间投影约束不等于已证明所有视频推理因果路径，未做开放式视频对话泛化。建议复现text-only／full-input成对评估及等rank随机子空间控制。[正文§2–4／Tables2–6](https://arxiv.org/html/2611.25776v1)

<a id="topic-4"></a>
## 推理与部署

### NariQ：Agent轨迹校准让2.5／3bpw保住长任务，2bpw仍明显退步（2611.26157v1）

**时间与版本**：v1提交2026-11-19 14:13:27 UTC；本轮11月20日分类公告恢复，首次公开秒界未确定。

Nari Labs／NariQ。对Qwen3.8-27B做trellis编码，按BF16 Agent自己生成的tokens校准KL、混合位宽和尺度／norm微调，使用70:30代码与通用轨迹；训练仓与DeepSWE按repo去重。同一Claude Code harness、262,144上下文、DeepSWE每题一次：BF16解44/113，3bpw47、2.5bpw46、2bpw33；TerminalBench62.6对63.8／60.2／52.4。**限制**：前两档与BF16差异在噪声内，不是显著胜出；其他量化文件来自Unsloth不同harness，即使归一为BF16比例也不可直接排名。文本GGUF10.1／8.4／6.8GB，另加0.9GB视觉projector；KV／activation尚未量化、MTP未量化，llama.cpp fork主要调SM120。建议复现长Agent轨迹与低位宽失败，勿用短数学持平替代长任务验证。[正文§2–6／Tables1–2](https://arxiv.org/html/2611.26157v1)

### FrontierReplay：不切断prefill也能构造精确混合模型前缀状态（2611.26020v1）

**时间与版本**：v1提交2026-11-19 12:34:25 UTC；本轮11月20日分类公告恢复，首次公开秒界未确定。

北京理工大学。记录一小窗递归层输入，在完整prefill后只重放递归计算，GPU事件完成后才发布checkpoint；按会话frontier保留并回收死状态槽给KV。单A80080GB、vLLM0.30.0／BF16、32GiB cache、TTFT≤500ms和TPOT≤50ms，Qwen3.5-9B的ShareGPT goodput为vLLM的1.27倍；Falcon-H1-7B SWE-agent为4.26倍，但95%区间[3.50,5.01]且原基线接近失效。**限制**：6–37倍是逐token状态重放斜率比，非整体服务倍数；需可预测结构边界、保存起点与精度舍入信息，超1024-token窗会回退切分；log占2GiB。建议优先复现同token轨迹、bitwise状态与负载曲线，再看吞吐。[正文§3–6／Tables1、3](https://arxiv.org/html/2611.26020v1)

### TideCache：混合层间空隙搬运全局KV，透明重分配物理页（2611.26619v1）

**时间与版本**：v1提交2026-11-19 18:58:31 UTC；本轮11月20日分类公告恢复，首次公开秒界未确定。

MIT／ETH／UCSD。固定虚拟地址后重分配各cache物理页，利用有界状态层计算时间预取下一全局注意力层的host前缀，新增KV保持GPU驻留。SGLang实现、GB300／GH200七种模型平台组合，在40K–110K多轮Agent与短chat混合内存压力点，相对调过分区的HiCache最多2.04倍Agent turns，整体P50每token时延降20–59%；同batch有最高6.3%额外decode-step代价。**限制**：Kimi K3和Qwen3.8-2.4T只取前48／36层；主结果每配置一个压力点、依赖CPU–GPU高带宽，不能外推PCIe卡或完整超大模型。建议复现页重映射／CUDA graph正确性与实际host链路下的容量—时延交换。[正文§4–5／Table3／Figs9–13](https://arxiv.org/html/2611.26619v1)

### Twindex：主KV卸载后，索引缓存成为新的HBM瓶颈（2611.25078v1）

**时间与版本**：v1提交2026-11-18 19:19:42 UTC；本轮11月20日分类公告恢复，首次公开秒界未确定。

项目Twindex，Yuxuan Xia等。HBM存共享项加压缩残差粗筛，host存原始index rows并由模型原indexer精排候选；并非全局top-k完全精确。992K prompt／最多32K greedy生成，Qwen、DeepSeek-V4.1、GLM-5.2相对同优化HiSparse分别容纳1.78／1.62／3.03倍请求、decode吞吐1.44／1.25／2.54倍。**条件**：硬件只披露GPU-A为179GiB HBM／原生NVFP4、host4TiB、每GPU PCIe5×16约50GB/s；Qwen／GLM三份976K共享主前缀，index仍每请求私有，避免host先耗尽。Qwen峰吞吐step69.8→86.4ms，40ms TPOT约束只剩1.11倍；均分配共享索引并批量打分时收益约1.04倍。质量均值最多低3.1%，不是无损。建议按真实共享前缀、host容量与TPOT复现。[PDF§5／Tables2–5、AppendixE](https://arxiv.org/pdf/2611.25078v1)

### AscMLA：共享FP4 latent用不对称K/V scale，核函数收益与服务收益分开（2611.26106v1）

**时间与版本**：v1提交2026-11-19 13:37:28 UTC；本轮11月20日分类公告恢复，首次公开秒界未确定。

中国科大／快手。不同规约轴的K、V scale拆开处理并流水交付，使MLA单份FP4 payload直接参加QK和PV，避免另解量化或双编码。CC10.3／CUDA13、query1、latent512、RoPE64、batch16–128、16K–256K的40配置，kernel几何均值相对TRT FP8快1.35倍、CAKE FP4快1.26倍，cache比FP8少38.2%。在8卡、Kimi-K2.6 ModelOpt-NVFP4、prefill／decode各TP4、64K输入4K输出、无投机、1.8req/s时，输出3139tok/s，比TRT快18.8%；P90 TTFT1.11秒、mean TPOT44.05ms满足20秒／50ms目标。**限制**：kernel用高斯FP8输入，不能当实际整模服务倍率；软件依赖及特定FP4硬件门槛明确。建议按实际长窗和精度回归复现。[正文§III–IV／TableIV／Fig6](https://arxiv.org/html/2611.26106v1)

### LightningMoBA：动态阈值路由与KV列复用有效，64K质量回归不能忽略（2611.26268v1）

**时间与版本**：v1提交2026-11-19 15:38:08 UTC；本轮11月20日分类公告恢复，首次公开秒界未确定。

项目LightningMoBA，Yingqi Shi等。把固定Top-k换成每query相对阈值，按KV列组织CSC工作并切分高fanout，Hopper流水覆盖KV传输与不规则query gather。五checkpoint／TP8／batch1／完整6,500题RULER，匹配平均Top16、保留dense decode，整generation walltime快1.20–1.60倍。**回归**：五模型在64K全都下降1.41–5.67点，整体均值也有两模型退步；32K完整稀疏attention路径15.8ms慢于dense FA3的13.0ms，128K才明显占优。Qwen3-4B／PG19 128K／100步训练对照，Top16不加速，Top32／64为1.225／1.602倍；有效TFLOPS不是模型浮点利用率（MFU）。建议复现长窗分段质量与route成本，不能只引用全局平均。[正文§3–4／Tables1–4](https://arxiv.org/html/2611.26268v1)

### DTS：用局部变化选anchor降低稀疏打分，短上下文反而更贵（2611.25938v1）

**时间与版本**：v1提交2026-11-19 11:23:22 UTC；本轮11月20日分类公告恢复，首次公开秒界未确定。

项目DTS，Qinyu Chen等。局部向量差决定Q/K anchor密度，并行chunk scan替代串行packing，最终所选tile仍用原Q/K/V计算。四H100／batch1，Llama3.1-8B的RULER均值89.02对full89.07、LongBench47.51对47.53；128K打分pipeline比XAttention2,571→1,592ms，512K backbone prefill比PyTorch FA2快3.29倍。**限制**：时延不含LM head和decode、KV关闭；质量只测至128K，不能称512K质量保持；4K–32K打分更慢，Qwen的RULER平均也比full低0.58点。建议复现真实长度分布及打分／attention占比，关注近全精度操作点。[正文§IV–V／TablesI–V](https://arxiv.org/html/2611.25938v1)

### RoCoKV：不确定时少剪KV、重复内容降权，cache预算口径需核清（2611.25433v1）

**时间与版本**：v1提交2026-11-19 03:16:18 UTC；本轮11月20日分类公告恢复，首次公开秒界未确定。

项目RoCoKV，Payman Behnam等。在RocketKV上用上一时刻不确定性调top-k和压缩强度，对相似近期keys加冗余罚，融合三阶段CUDA decode并接vLLM。R1-Llama8B／R1-Qwen7B、A40，MATH／AIME最长16K；A40 R1-Llama8B、AIME batch16／预算512时，相对full vLLM时延196.2→154.5秒、吞吐247.3→341.3tok/s。**限制**：文中KV token预算主要是attention平均读取top-k流量，不可直接等同峰值驻留显存；AIME最大预算仍落后full7.5／5点；源代码注明录用后发布，当前不是可直接安装的正式插件。建议继续观察代码，再复现质量、流量和真实显存三条曲线。[正文§2–3／Tables1–4](https://arxiv.org/html/2611.25433v1)

### TCDA：扩散步数按任务校准，只有utility检查会放过循环退化（2611.25895v1）

**时间与版本**：v1提交2026-11-19 10:47:26 UTC；本轮11月20日分类公告恢复，首次公开秒界未确定。

PayPal／TCDA。对有限候选schedule做paired校准，以同时单侧bootstrap界联合检查质量容忍度和退化guardrail，选最便宜通过者，否则用完整schedule。ELF-B XSum128样本校准的500拆分中499次选prefix19，少3.37倍evaluations、wallclock快3.28倍；但仅utility准则会500/500次放过OWT循环的hybridH16，联合规则为0。**限制**：理论错误率是同分布／有限候选／非退化方差条件下渐近保证，不是有限样本分布漂移保险；校准和shadow运行有成本，随机时间grid还必须固定。建议复现quality＋重复／长度guardrail、漂移后重校准，避免统一默认步数。[正文§3–5、§7／Tables1–4、AppendixK–M](https://arxiv.org/html/2611.25895v1)

### CaRR：固定扩散forward预算内修复与恢复草稿，规则选择也有成本（2611.25763v1）

**时间与版本**：v1提交2026-11-19 09:03:43 UTC；本轮11月20日分类公告恢复，首次公开秒界未确定。

项目CaRR，Hao Wu等。对ME-DLM Stage3的原定mask/edit forwards重掩码不稳替换，再从轨迹恢复满足输出结构的候选；不额外调用backbone。64次／满预算1/8时六项代码数学均值31.90→39.34，MBPP+多10.05点；满预算只多0.36点且KLASS均值更高。**限制**：forward相同不代表wallclock相同，batch1暖测HumanEval／MATH中位时延多0.75%／3.37%；Math恢复规则基于1/8数据开发，换回TCR-only仍有7.25均值增益但该点非全盲；历史方法的转引数字未匹配预算。建议复现预冻结规则与同NFE／同时间双对照。[正文§3–4／Tables1–3、AppendixB/F/H](https://arxiv.org/html/2611.25763v1)

### Raptor：拒绝反馈同时调树宽深和草稿预算，接受长度不是效率（2611.25436v1）

**时间与版本**：v1提交2026-11-19 03:17:52 UTC；本轮11月20日分类公告恢复，首次公开秒界未确定。

北京理工／华中科大／EPFL。用覆盖不足与冗余扩展的不同拒绝信号，配EMA和momentum有界调树。两RTX3090、LLaVA1.6-7B／Qwen2.5-VL7B、三类drafter、七多模态基准固定子集、greedy验证规则，作者报告相对自回归最高2.39倍平均吞吐；EAGLE/Qwen组由1.38到1.90倍，即使接受长度下降也更快。**限制**：速度是每题tok/s的算术平均比，包含generation内prefill／视觉处理但不含函数外处理；不能当服务到达负载的goodput，未证明采样分布场景。建议复现draft成本、同样例逐题延时与greedy结果一致性。[PDF§4／Tables1–3、AppendixB](https://arxiv.org/pdf/2611.25436v1)

### Gated DeltaNet live migration：最早释放源卡与最短用户停顿是不同目标（2611.26153v1）

**时间与版本**：v1提交2026-11-19 14:10:37 UTC；本轮11月20日分类公告恢复，首次公开秒界未确定。

项目GDN live migration，Yongbo Yu等。比较snapshot、更新log、原生逐token精确重放与batch32近似重放，明确handoff的token输入边界。小模型Qwen3.5-0.8B／纯GDN1.3B，4090同机／3090到4090 TCP，软件限速10–250MB/s与0.5／50ms RTT，单请求每点三次。40共同完整点里精确重放总是最早释放源，而log在纯GDN中总是最短精确停顿；10MB/s时batch32约81ms、log445–464ms，但前者放宽数值精确契约。**限制**：log新增流量若高于链路就积压，hybrid可持续产生约17MB/s；Exact有30秒删失，不能丢掉超时后声称通赢。完整双GPU混合whole-request选择器和探针开销仍待测。建议复现退役安全、状态边界和后续生成，不直接部署启发式selector。[正文§2–7／Tables2–5、9](https://arxiv.org/html/2611.26153v1)

### Berth：以发射门控和已有GPU时间片保护推理SLO（2611.25091v1）

**时间与版本**：v1提交2026-11-18 19:29:56 UTC；本轮11月20日分类公告恢复，首次公开秒界未确定。

项目Berth，Mohammadali Khodabandehlou等。CUDA shim限制best-effort工作入队、利用驱动已有slice控制和迭代deadline slack插入后台计算，无需新内核抢占。单A10040GB，vLLM0.11的Qwen2.5-1.5B／0.5B、Azure到达trace／180秒、TTFT250ms＋TPOT25ms，最差服务99.8–100%请求双达标，后台训练保留17–38%单独吞吐。**限制**：shim看不到请求队列，突发／近饱和较弱；不管理显存容量、复制引擎或带宽，共享MPS有故障域和受信租户假设；新模型／batch需重新profile。建议复现自有负载的tail和后台收益，不把小模型SLO视为大模型隔离保证。[正文§3–6／Figs13–14](https://arxiv.org/html/2611.25091v1)

### Complete-SVD：压缩保留全logits几何，会改变校准域与跨域的取舍（2611.26059v1）

**时间与版本**：v1提交2026-11-19 13:04:57 UTC；本轮11月20日分类公告恢复，首次公开秒界未确定。

项目Complete-SVD，Alper Kalle等。将全模型logits的L2偏移线性化，以输入协方差和logit-Jacobian Gram做加权SVD，random probe用每序列K次反向估计输出因子，避免按词表×位置枚举。与同估计框架的KL-SVD对照，L2在WikiText2校准后的跨域C4／PTB困惑度更好，KL则常在校准域困惑度和下游准确率更好；不应只选一种指标证明全面最优。**限制**：正文主要7–8B，逐层统计取原模型，忽略前面层压缩后的误差累积；K次反向与因子存储有校准成本，无通用无损或服务加速保证。建议复现固定rank／校准集与跨域、IFEval共同评估。[PDF§3–5、Limitations](https://arxiv.org/pdf/2611.26059v1)

### NVIDIA Qwen3.8 NVFP4 QAD新增明确SGLang服务环境

11月20日固定`cd46ab575a5a687f0bc3e54251016fc2cc1a6aef`新增`lmsysorg/sglang:v0.5.23-cu130`与262,144 context命令，原卡只有vLLM。属于现有量化感知蒸馏checkpoint的runtime recipe扩展，无新权重/新评测，卡测试平台GB300，不能据此外推其他架构或新吞吐收益。**考虑按该镜像做加载与质量回归**。[固定卡差分基线](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4-QAD/blob/cd46ab575a5a687f0bc3e54251016fc2cc1a6aef/README.md)。

<a id="topic-5"></a>
## 开源工程

下列均是默认分支已合入工件；除明确标注的发行外，不代表已进入正式版本。日期按固定patch作者时间与main日期列表区分，未核的mergedAt秒界不作推定；作者性能数据均未经本轮独立设备压测。

### SGLang：分块权重更新不再静默遗失 fused pair 的前半边

DeepSeek 的 q_a/kv_a，以及 block-FP8 indexer 的 weight/scale 对，过去只存在一次 load_weights 调用的局部变量里；两半落入不同 chunk 时会遗失前半，融合参数静默保留旧值。现在 `_PendingWeightHalves` 放在模型上，暂存时 clone，跨调用配对；begin、磁盘加载及 rollback reload 先清旧半边。session 的 end_weight_update 汇总各 TP（张量并行）rank 缺失项并返回 success=False。**边界：IPC 非 session 路径遇到未配对项仍只记录 error、随后返回成功，不能推广为所有更新接口都可靠失败。**建议使用 bucket/chunk refit 的 RL 团队优先回归这些路径；本轮没有设备复训。

一手来源：[#51661 固定提交](https://github.com/sgl-project/sglang/commit/7894bdf172df633b9770c06c07842eb92cb0c695)（`7894bdf1`，作者时间2026-11-20T09:15:11Z）。

### Megatron MoK：Qwen3.5-VL 路由与图优化有明确数值正确性缺口

Mixture-of-Kittens（MoK）后端为 padding token 去路由，并把 router 与 megakernel 放进同一 per-layer/whole-layer Transformer Engine（TE）CUDA graph，减少填充计算和 host 开销。公开对照为 GB300 上 Qwen3.5-VL 35B/8 GPU/EP4 相对 native MoE +39.3%，397B/64 GPU/EP32、强制负载均衡 +63.8%/+64.7%。**作者随后披露两臂的 TE per-layer attention graph 都漏掉多模态旋转位置编码（MRoPE）**，因此不能把数字当成正确模型的端到端加速；修正后收益还会下降。固定 README 另要求 NVLink 域、TP=1、固定形状，per-layer graph 需 TE≥2.19、PP=1，并拒绝 router-only graph。建议深入读 padding/图状态机制，等位置编码修复后重跑，再决定采用。

一手来源：[#8654 固定提交](https://github.com/NVIDIA/Megatron-LM/commit/87db06d436df8be21cf4605150164eab4b46b9cd)（`87db06d4`，作者时间2026-11-20T08:12:49Z）。

### FlashInfer：SM103 paged-context FMHA 换静态调度以规避挂死

只把 SM103 的 trtllm-gen paged-context 从 Persistent 改 Static，一 CTA 一 tile；tiny-GQA generation 显式恢复 Persistent。作者在 B300 观察到多 tile CTA 的 mbarrier Missing wait：GPU 利用率100%、DRAM0、无 Xid且不恢复。8×B300 分两臂各约68 GPU小时，Persistent挂2次、Static0次；并发240/280/320吞吐差±0.7%，800个prefill batch重放总时间比0.998。**这是局部规避，SM100、ragged context、generation均未推广，上游本分支全套测试未跑。**建议对应线上挂死优先考虑回补并压力回归；0次挂死不是长期无故障证明。

一手来源：[#9702 固定提交](https://github.com/flashinfer-ai/flashinfer/commit/ceb7af388d41ae3532f27de3750375ef7efc29aa)（`ceb7af38`，作者时间2026-11-20T23:39:01Z）。

### SGLang／Foundry：CUDA graph 持久化进入 main，冷启动收益伴随稳态代价

选配 `--cuda-graph-persistence save/load` 保存图、kernel和地址布局，启动时预建 lazy communicator/dispatcher 状态，之后恢复而不重新capture。固定依赖是 Linux x86_64 的 foundry-core≥0.1.0rc5（PR旧描述仍rc4）。**必须保留限制：强制 TORCHDYNAMO_DISABLE=1，依赖该融合的 gpt-oss decode可慢约9%；FlashInfer autotune关闭，可先原生运行生成tactic cache再归档。**save/load须同模型、flags、并行布局和GPU；LoRA、elastic-EP recapture、若干图变体不支持，投机算法也有门禁。不支持配置在恢复前回退原生；一旦load已开始恢复则失败报错，启动OOM也不回退。建议重启频繁且配置稳定的服务复现冷启动与稳态两类指标，不能只摘“几秒启动”。

一手来源：[#42254 固定提交](https://github.com/sgl-project/sglang/commit/53b8e56e4a065af5993ade615f0dc434947038e2)（`53b8e56e`，作者时间2026-11-20T17:26:15Z）。

### FlashInfer SplitMoE：NCCL EP 0.3 增加接收窗口与量化通信，容错仍未闭环

`nccl-extensions≥0.2.0`交付EP0.3，新增独立dispatch/combine SM预算、HT流与元数据策略、LL Rank-Major零拷贝接收窗口，及量化dispatch forwarding。**eager_recv需host同步、不能capture**；NVFP4 combine是有损的BF16专家输出回传，与专家权重精度独立。4×SM107合成 H2048/I1024/E16/top-k4、64token/rank、capacity512案例中，FI接收数组8.09→1.01MiB/rank，却使uncaptured host-completed forward295.01→418.11μs；这不含provider staging或总显存。早期GPU资格测试为53回归+7完整forward，NVFP4对codec参考2%误差并不证明模型质量。丢peer探针虽让3幸存rank继续10轮，正常销毁仍卡在原communicator barrier，完整故障恢复未验证。建议按自己的图/内存/质量/故障负载复现。

一手来源：[#9623 固定提交](https://github.com/flashinfer-ai/flashinfer/commit/86ab1c74561d1ae7ef70ece38f2c34007848d030)（`86ab1c74`，作者时间2026-11-20T23:24:49Z）。

### Megatron：MFSDP v2 Muon 优化器工厂集成恢复合入

这是此前 #8548 被 #9101 撤回后的恢复：二维矩阵由FsdpMuon处理，其余由Adam处理，均经FullyShardedOptimizer并由ChainedOptimizer统一DTensor梯度范数与裁剪。各rank都创建Muon+Adam，避免PP（流水线并行）不同stage成员数不一致导致collective死锁；dense/expert可分别规划mesh，主权重更新与模型权重刷新分离。要求 `--muon-no-split-qkv`、一维DP mesh和分片的非量化主权重/梯度。历史4×GB200、TE2.18测试包含FP32/BF16及EP2/EP1参考；旧revision中默认medium Newton–Schulz精度曾约0.12%跨EP范数偏差。**恢复时未重跑runtime，checkpoint仍另PR。**建议考虑复现优化器/裁剪一致性，不把旧测试当新baseline全验证。

一手来源：[#9106 固定提交](https://github.com/NVIDIA/Megatron-LM/commit/7a4abaf64b27f595611dbae190c8f8cdfee36b0d)（`7a4abaf6`，作者时间2026-11-20T04:53:59Z）。

### Megatron：Mamba 训练把布局拷贝移到较窄 hidden state

SSM kernel本来是batch-first；原来在较宽in_proj/SSM输出上做sequence↔batch contiguous拷贝，现在让in_proj、SSM、out_proj保持batch-first，只在d_model宽的输入/输出转换。`MambaMixer._use_batch_first_ssm`按估计拷贝量决定是否启用，且仅memory-efficient训练路径、CP=1、无dynamic CP/SP、无residual_stream/mHC重算，也不能使用按dim0补序列的FP8 inference wrapper。**机制减少布局流量，不等于所有TP/CP配置变快**；高TP下投影已分片，反而可能不划算。建议用目标并行度profile contiguous流量与全步时间，本轮不引用旧Draft PR的未定稿结果。

一手来源：[#8696 固定提交](https://github.com/NVIDIA/Megatron-LM/commit/6d6d5af44c29070ef9b0ea544f7e0a5b3b2258c9)（`6d6d5af4`，作者时间2026-11-20T08:19:07Z）。

### TorchTitan RL：权重 push 在下一次前反向前完成，消除事件循环延迟

trainer actor的forward_backward没有await，会独占asyncio loop；后台push即使GPU→CPU拷贝完成，也可能等整个下一次前反向后才回复，从而拖延generator pull。Controller现在在optimizer及上一轮pull之后显式等待push，再后台pull并释放rollout group；pull仍与下一步计算重叠，push不再重叠。1 trainer+1 generator GB300、Qwen3-0.6B、10步A/B里，当main的push本就抢先完成时，这次改动额外阻塞0.022s/步、没有收益；较早TorchStore路径曾存在约0.15s等待。**27B多host A/B未跑，单host用UCX/TCP+CUDA IPC而非RDMA。**建议跨训推权重传输团队重点读其完成时序和fake-actor次序测试，勿写成无条件加速。

一手来源：[#5235 固定提交](https://github.com/pytorch/torchtitan/commit/cb790dd08ff64c6b601a24c9730b96e2b1ca714f)（`cb790dd0`，作者时间2026-11-20T12:56:57Z）。

### TorchTitan RL：截断 penalty 保留部分奖励，并改变配置兼容性

`truncation_penalty`替代把分数直接重置为常量的`truncation_reward`：例如测试通过率0.9的截断轨迹减0.2后得0.7，保留partial credit。默认penalty0关闭、负值报错；旧字段直接变unknown-field错误，validation共用rubric也会扣分。search_r1的length截断最终答案现在可被正常评分，而不再一律0；turn-limit/prompt-too-long且末尾为tool call的仍没有答案。**没有in-tree recipe启用非零penalty，也没有训练效果实验。**建议迁移时先核奖励分布与验证口径，不能把API变化写成已证明提升RL能力。

一手来源：[#5451 固定提交](https://github.com/pytorch/torchtitan/commit/1fdf04562e7bb02d8df9202a269668b43c42f503)（`1fdf0456`，作者时间2026-11-20T12:57:56Z）。

### TorchTitan RL：token budget 可非整倍上下文，但分布式条件仍保留

batcher按一条flat序列packing，本不需要每rank token budget整除max_context_length；现只要求至少容纳一个完整上下文，释放2.5倍等容量选择。TrainingEngine的SP×CP、Dist-MoE的CP×TP整除条件仍存在，超长sample仍丢弃。单GB300、BF16 debug Qwen3.5混合MoE，以context128、文档[128,128,50]比较320/384/512预算：306实token logits逐位一致，梯度不逐位一致，差异量级与两个整数倍预算之间相近。**未跑非整倍预算的多GPU Dist-MoE trainer step**。建议显存卡在两个整倍之间的训练考虑复现packing及分布式门禁。

一手来源：[#6264 固定提交](https://github.com/pytorch/torchtitan/commit/30b8b371b810931dda29b8f7f640ba55c3ac01e5)（`30b8b371`，作者时间2026-11-20T12:57:37Z）。

### verl：flat sampled log-probs 减少 rollout 的 Python 对象开销

`calculate_log_probs=True`、不要求top-k时logprobs=0，旧条件把0当false，未启flat列表而为每token建dict。现在只要非None便请求flat，并从每行第一项取sampled token；prompt reader改索引迭代，修复vLLM<0.29时单token prompt空slice报IndexError。输出接口仍float列表，greedy测试要求token及log-prob完全相同。作者vLLM0.30、单B300、Qwen3-0.6B量级模型，496请求/226K输出token、max_num_seqs256，每模式两轮：6.99→6.29s（约10%），CPU/GC时间同时下降。**只说明该CPU受限AsyncLLM负载，GPU受限收益会小。**建议开启rollout log-probs的团队考虑复现；SGLang/TRT-LLM路径不受此改动影响。

一手来源：[#8689 固定提交](https://github.com/verl-project/verl/commit/453954b3cb68575803ba4ff769032dda655b8f99)（`453954b3`，作者时间2026-11-20T06:09:09Z）。

### verl：IS 标准差改成先中心化再平方

重要性采样（IS）诊断过去用E[x²]−E[x]²，float32中序列权重近常数时相消，让同一有效行在补空行前后标准差可差约1e-4。`compute_is_metrics`改用masked_var(unbiased=False)，保留clamp与mask_count≤1时0。**这是监控统计的数值修复，不能写成policy-gradient目标或训练质量提升**。建议依赖rollout_is_std判断权重退化的用户回归padding不变性。

一手来源：[#8774 固定提交](https://github.com/verl-project/verl/commit/3a909894862c9d298073618e742e2f3ce51e86fc)（`3a909894`，作者时间2026-11-20T02:13:32Z）。

### Megatron：混合 MoE＋MTP 的 P/D handoff 修复，并区分 decode 缓存职责

Gated DeltaProduct的conv多Householder bands、recurrent维度顺序与Mamba2不同，旧SSM reshard会拒绝；现在携带num_householder并按实际布局handoff。动态server按本shard的TP/PP/CP/EP组建模，ETP>1即便EP=1也保留dummy recurrent slot；补上prefill首token的MTP acceptance计数。NCCL传输改为启动时预连接一个专用组，避免首次P2P连线与发布次序共同引发挂死。#9534先明确role：decode直接导入live SSM，不建Mamba prefix cache，prefill保留复用，构context前必须设role。内部模型ETP4/EP1的collocated对照位于噪声底附近，但**模型身份未公开，本轮未复测**。建议部署混合模型P/D时核SSM布局、空闲rank和首token计数；#9534作者日期早于主窗，作为本轮补漏、合入秒界未核。

一手来源：[#9120 固定提交](https://github.com/NVIDIA/Megatron-LM/commit/8a09c13ed77049e120701ad8f2e8e713b0335e27)（`8a09c13e`，作者时间2026-11-20T09:37:27Z）；[#9534 固定提交](https://github.com/NVIDIA/Megatron-LM/commit/84c7f168baa9f53b4db73d6ac4551d0f6734947f)（`84c7f168`，作者时间2026-11-20T00:31:48Z）。

### vLLM：VMM IPC 权重共享进入 preload，可减少导入开销但不能合并启动指标

固定文档新增Linux上的CUDA/HIP VMM后端，默认仍torch_ipc；daemon在postprocess和warmup后发布最终storage manifest，POSIX FD传递保留tied/shared storage、offset与stride，按默认1GiB打包减少FD和granularity padding。最终文档比PR初始CUDA/Linux说明扩展到ROCm，**实测仍仅4×GB200、BF16、TP4**。PR原型含未提交修改、单A/B、greedy8输出token；表格mean attach5.951955→4.774735s（−19.78%），同时materialization约+1.33s；摘要写−24.2%，两者不一致。作者明确不承诺稳态吞吐或一致的整engine启动收益，final revision待验证。建议对频繁重启服务复现完整startup分解及daemon失配fallback。

一手来源：[#68210 固定提交](https://github.com/vllm-project/vllm/commit/e6109f064ae7205a4164593817238180a4cbb59f)（`e6109f06`，作者时间2026-11-20T12:29:38Z）。

### vLLM：HiSparse 的 host indexer 按实际 offload 组判定 TP 单副本

#68785只用被offload的KV组重新计算TP复制关系，避免host-resident raw page不带TP元信息便把复制的indexer存成每rank一份；测试包含source/indexer/resident三组，安全门禁仍为MLA、TP>1、PP=1等。#68967为没有dense-MHA prefill backend的NoPE＋fp8_ds_mla路径照样生成HiSparse staging plan，测试使用GLM-5.3-Flash形状/192序列、block64、topk128。两项分别解决host重复副本和缺prefill计划，**没有完整服务显存/吞吐对照，不能推导固定倍数收益**。建议HiSparse多卡与NoPE用户考虑回归。

一手来源：[#68785 固定提交](https://github.com/vllm-project/vllm/commit/b492a09c10118f43bfaf5ff75f17fa10fe86e0a6)（`b492a09c`，作者时间2026-11-20T19:20:48Z）；[#68967 固定提交](https://github.com/vllm-project/vllm/commit/0e1d945bac0d646b0d5ae0933d36c5583c6d9a36)（`0e1d945b`，作者时间2026-11-20T18:08:37Z）。

### vLLM：低精度与分片保存的三个具体正确性修复

SM89开启VLLM_BATCH_INVARIANT后，INT8不再误进只支持FP8的fixed-config CUTLASS kernel；INT8的int32累加走普通dispatch即可。DeepSeek-V3.2在建backbone前把融合q_a/kv_a及wk/weights_proj映射写入quant_config，避免原checkpoint明确不量化的部分被融合层错误量化而加载失败。ShardedStateLoader以shape×stride算末地址并在保存时contiguous，支持ROCm row-stride padded权重；新增safetensors回读要求逐值精确。**这些是特定dtype/模型/布局范围修复，没有通用性能实验。**建议分别回归INT8 batch-invariant、NVIDIA V3.2 NVFP4和padded weight导出。

一手来源：[#67687 固定提交](https://github.com/vllm-project/vllm/commit/3302deb9b589b86686255e3ea51654f0facd8b5b)（`3302deb9`，作者时间2026-11-20T15:49:48Z）；[#68948 固定提交](https://github.com/vllm-project/vllm/commit/c334a4959a2d76ba4d4a7b12ee30ff0438989b83)（`c334a495`，作者时间2026-11-20T14:52:25Z）；[#61432 固定提交](https://github.com/vllm-project/vllm/commit/339d0bd69d520df75957b1f13efb42f717ee169c)（`339d0bd6`，作者时间2026-11-20T16:16:43Z）。

### vLLM Transformers backend：同请求处理 Omni 音频、图像与视频

已合入main，支持Omni模型的多模态输入组合，处理Qwen家族命名差异、video内audio对应关系及混合配置里的bidirectional encoder，并补processing/generation测试。PR明确要求Transformers5.23.0，现有dependency pin若未放开该版本，Omni CI仍会失败；固定patch包含相应registry与processor改动。**这不是音频输出能力或所有Omni模型零适配保证，也没有吞吐对照。**建议有Omni输入服务需求的团队深入读processor/placeholder配对边界，按匹配的依赖版本复现。

一手来源：[#60648 固定提交](https://github.com/vllm-project/vllm/commit/8f8f6c8fc85de978555bf705ac21ec7580ded88f)（`8f8f6c8f`，作者时间2026-11-20T20:06:22Z）。

### SGLang：request KV hints 接到存储层，external linker 补 KDA 图等待

KV hints原来停在Req，现经prefetch/lookup传给显式opt-in的HiCache storage；缓存salt/LoRA命名空间下同时提供KV event ID到storage key的映射。**提交说明没有in-tree backend消费hints，KVCR消费者仍另PR，不能写成已兑现跨worker命中收益。**KDA（线性注意力）captured state读的breakable replay等待条件现同时覆盖hierarchical cache与external linker，防逐层load未完成先replay；不是所有backend都需要该事件。attention-DP派生的attention-TP subgroup另恢复PyNCCL communicator默认策略，仍有process-group fallback。建议外部KV后端核hint命名空间与load完成顺序，暂无定量加速结论。

一手来源：[#51088 固定提交](https://github.com/sgl-project/sglang/commit/448efb13186f1ee8f3b0985900e25b1bee0cd508)（`448efb13`，作者时间2026-11-20T10:34:55Z）；[#50635 固定提交](https://github.com/sgl-project/sglang/commit/dc8f1c35bdb960b78065c2c7795fe0148cc08984)（`dc8f1c35`，作者时间2026-11-20T13:50:52Z）；[#51707 固定提交](https://github.com/sgl-project/sglang/commit/4f56795ec5485b9bf1c72aea23dd153c3ee07e50)（`4f56795e`，作者时间2026-11-20T17:24:34Z）。

### SGLang：Ascend P/D KV checksum 从 CUDA 诊断扩展到 NPU

按index tensor设备分派到sgl-kernel-npu的Adler-32 strided checksum，paged/FIA KV及nested state都检查设备和连续性。作者A3、Qwen3-32B、P TP2+D TP2覆盖10K输入、prefix复用、注入损坏拒绝及后续恢复；A3另与zlib参考比较20case，A5用Mooncake+TENT/MemFabric。**这是字节传输完整性诊断，非模型任务准确率证明；依赖NPU kernel交付版本，使用上游early-send修复的验证不能当任意组合保证。**建议排查静默KV损坏时考虑启用并测自身诊断开销，本轮不搬用缺完整对照的性能数字。

一手来源：[#41056 固定提交](https://github.com/sgl-project/sglang/commit/6ccb37eb58b1d14b5997f3930425d910d3e0183a)（`6ccb37eb`，作者时间2026-11-20T08:33:55Z）。

### SGLang：gfx950 引入 uTLX/TLX 的 Muse MoE 与融合路径

vendor的gfx950 GEMM/MoE/norm/router kernel可由uTLX接标准Triton；`MUSE_MOE_BACKEND=tlx`显式选择Muse融合MoE，不能支持的形状告警回退FlyDSL。dense BF16/FP16无bias GEMM、boundary norm、FP32 router由三个SGLANG_USE_TLX_*开关独立opt-in，均默认false；deterministic模式不走会随M变化的TLX dense plan，router还多一份约8MiB/层[E,K]权重。**PR旧快照仍Open，但固定commit已在main；未审完约30K行vendor kernel，也未独立benchmark，不引用其最好倍数。**建议MI355X/Muse用户考虑复现并核每条dispatch门禁。

一手来源：[#50354 固定提交](https://github.com/sgl-project/sglang/commit/2ff46c5979d1ce1c053433c6bbcd13f1debeb878)（`2ff46c59`，作者时间2026-11-20T21:27:03Z）。

### FlashInfer：SM12x 稀疏 draft lm_head 的 gather GEMV＋scatter 融合算子

新增gemv_bf16_fp4_gather：按row_ids读取NVFP4行、FP32累加，直接scatter进最终logits，避免独立scatter；SM120/121，BF16输入、线性E4M3 block scales、无分配/host同步，未选输出列不改。DGX Spark GB10、R215552/J16384、K2048/2560，CUDA graphs对照Triton gather+scatter，M1为1.19–1.25倍、M32为1.84–1.92倍；包含fill和dense-prefix copy，**不含scorer/top-k/dense GEMM完整成本**。vLLM sparse draft head消费者仍是提案#67220，不能写成vLLM已交付整体推理加速。建议带宽受限投机系统考虑算子复现，单测包含negative/duplicate IDs、stride与graph replay。

一手来源：[#9780 固定提交](https://github.com/flashinfer-ai/flashinfer/commit/e12c19376a66d7247a674415439645b295379739)（`e12c1937`，作者时间2026-11-20T23:28:03Z）。

### FlashInfer：NVLink 单边 all-to-all 补编译器 memory clobber

dispatch发flag前的release fence、combine读flag后的acquire fence及旧架构acq_rel inline PTX补上`::: "memory"`，限制编译器把payload/flag访问挪过硬件fence。移植自TensorRT-LLM #22388；**这是潜在重排风险修复，作者不预期当前编译器必然表现变化，没有确定触发复现，且本地未跑多GPU测试**。建议维护这份vendored kernel的团队考虑同步补丁，勿说成已复现数据损坏根因。

一手来源：[#9649 固定提交](https://github.com/flashinfer-ai/flashinfer/commit/bc69e528ef216bb7b827e27704595d5decbc9b5b)（`bc69e528`，作者时间2026-11-20T23:14:05Z）。

### TensorRT-LLM 1.3 正式版：先处理后端迁移，再评估新能力

NVIDIA 于11月20日02:45Z发布正式版 v1.3.0（release页tag指向 `c66df86`）。**本次增量是正式交付边界，不把此前 main/候选版已有特性重新当首次实现。**旧TensorRT引擎后端、C++ Executor、`trtllm-build`、engine-dir benchmark和双模型投机被移除，PyTorch成为唯一后端；升级清单另包括CLI优先于YAML、KV传输默认60秒超时、单请求logprobs上限100。发布说明列出混合模型前缀缓存、Router Replay、在途权重更新及Rubin预览支持，但无统一可比性能实验。已知问题仍含Qwen3-32B-FP8/B300拆分服务错误输出、encoder-decoder FP8 KV越界，以及DSpark拆分服务不转移draft KV导致acceptance下降。**建议深入阅读迁移与已知问题后考虑升级**；下面同日main修复不可据日期推定都在此tag中。[正式release](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.3.0)。

<a id="topic-6"></a>
## 硬件与软件协同

### VR-SparseQuant：Rubin双路2:4 FP4改善精度，不能读成比dense FP4更快（2611.25717v1）

**时间与版本**：v1提交2026-11-19 08:29:39 UTC；本轮11月20日分类公告恢复，首次公开秒界未确定。

项目VR-SparseQuant，Hanrui Wang等。用两路自适应2:4稀疏激活编码，复用同一权重tile并累积同一输出，提高有限FP4表示能力；按Vera Rubin原生sparse FP4做融合编码＋GEMM。Nemotron30B RTN W4A4的PPL6.85→5.57、AIME78.9→86.7%；原生kernel比BF16的GEMM／MoE快1.7–3.1／1.4–2.7倍。**代价**：相对最快适用dense W4A4，GEMM慢7–27%、MoE慢18–47%；G32每32逻辑激活的values＋metadata＋scale合6.5bits，比dense G16的4.5bits更多；G32权重也牺牲scale粒度。计时含激活准备、排除通信、用预平衡路由；质量仿真与原生kernel测量需分开。建议将其视作精度—延时交换候选，先在实际MoE不均衡和通信中复现。[正文§3–5、§7／Table2／Fig5](https://arxiv.org/html/2611.25717v1)

### TensorRT-LLM：RPC接收登记与拆分服务重试必须匹配交付语义

`rpc_client.py::_call_async`在发送可能让出执行权之前登记response future，避免响应先到而接收者尚未存在；close取消并收敛仍在发送的任务。拆分服务则把首chunk前失败改为HTTP错误／协议error事件；context可换ID重试，generation只在确认尚未抵达server的连接失败时复用ID重发，其他失败直接终止，因为一次性KV handoff可能已被消费。**优先考虑采用并回归**早到回复、取消、首包失败；这不是任意阶段都可安全重试的保证。

**版本与来源**：固定提交日期均为11月20日，已在默认分支列表确认；PR合入秒界未单查。固定SHA见[TensorRT-LLM 53c21a3a](https://github.com/NVIDIA/TensorRT-LLM/commit/53c21a3aeb669f751c5b10aa6645a30f58fec873)；[TensorRT-LLM 75749ff0](https://github.com/NVIDIA/TensorRT-LLM/commit/75749ff08ba33fe7915f5983f68a8876a70e41d7)。状态为main/master已合入，未据此承诺正式包已包含。

### TensorRT-LLM：TMEM回收前加栅栏，RocketKV迁至Manager V2

Blackwell MLA FP16/FP8 correction stage在`consumer_release`前加入`fence_view_async_tmem_load`，确保读取完成后softmax warps才可覆盖stage。RocketKV另用每层KT辅助pool共享KV slot生命周期和预算，但与固定步长K/V池分开coalescing，接入KVCacheManagerV2。**考虑采用前做缓存与数值回归**；补丁证明同步及资源记账路径改变，未给可核验整网收益，也不能把辅助KT池当免费空间。

**版本与来源**：固定提交日期均为11月20日，已在默认分支列表确认；PR合入秒界未单查。固定SHA见[TensorRT-LLM 9d392121](https://github.com/NVIDIA/TensorRT-LLM/commit/9d3921216c2df20adf2f46462173340677c69dce)；[TensorRT-LLM dccece1a](https://github.com/NVIDIA/TensorRT-LLM/commit/dccece1a9f707bdab7cd46facde349d65c2fc28a)。状态为main/master已合入，未据此承诺正式包已包含。

### TorchTPU：关闭争抢注意力图的XLA rewrite，并让ragged A2A反向可追踪

TorchTPU已有Kiln长序列FlashAttention；默认关闭XLA HLO FlashAttention／online-softmax rewrite，避免对某些128倍序列抢先改写、补齐短head并阻断原生融合。作者TPU v7x、331模型套件几何均值step time仅下降0.37%；v6e、BF16、H8、D32/T128、B3264的单shape才是23.61→2.10ms，不能写成全模型11倍。ragged_all_to_all反向另改走functional all_to_all_single＋wait_tensor，保留符号整数形状，避免fake process group失败与动态token维特化。**考虑复现**真实训练图与不同shape，未做本轮TPU实测。

**版本与来源**：固定提交日期均为11月20日，已在默认分支列表确认；PR合入秒界未单查。固定SHA见[torch_tpu 6a0d1ef7](https://github.com/google-pytorch/torch_tpu/commit/6a0d1ef786397667c0ee0aad35f1fb5cd358cde3)；[torch_tpu 70c827c6](https://github.com/google-pytorch/torch_tpu/commit/70c827c6b2d6fa8e52911aaf97401cd06146f822)。状态为main/master已合入，未据此承诺正式包已包含。

### SGLang TPU：三缓冲KV写回竞态与FP8缩放均有明确修复

Batched RPA预取tile t+2时会复用tile t−1的buffer，旧实现尚未等其copy-out，可能把缓存页撕成其他页的32字节片段。作者v7x seq-along-lane、decode batch>12观察约400次调用一次；新KV ref在自己的copy_out末尾等待，配相关测试。另一补丁修transposed/ones-V prefill的K/V scales、FP32 exp2缩放与BF16 mask下界，row-sum扩展lane不乘V scale。**优先考虑采用并回归**长运行缓存与量化，不将此特定发生率外推所有TPU负载。

**版本与来源**：固定提交日期均为11月20日，已在默认分支列表确认；PR合入秒界未单查。固定SHA见[sglang-torchtpu 4638039b](https://github.com/sgl-project/sglang-torchtpu/commit/4638039b577cff6d1b9eba100ab61149bcdb6ba0)；[sglang-torchtpu 191a967e](https://github.com/sgl-project/sglang-torchtpu/commit/191a967ed783140e312aff7931c01f7f474056c4)。状态为main/master已合入，未据此承诺正式包已包含。

### SGLang TPU：FlyWheel可选后端与可持久化direct_compile

新增`--attention-backend flywheel`复用RPA fused KV pool，仅替换softmax MHA/GQA kernel，支持BF16/FP8 KV；需可选flywheel依赖，版本进入编译缓存键。head_dim256另改直接分配目标布局，避免每次reshape复制整层KV；这不证明所有模型均加速。direct tracer可在显式unsafe bundle选项下保存/加载命名图的executable，以源内容、编译器、环境、参数及配置定键；每次启动重新hash源码，防同大小且mtime保留的编辑误用旧图。**建议深入阅读后复现**冷/热启动和失配回退，无开关默认不变。

**版本与来源**：固定提交日期均为11月20日，已在默认分支列表确认；PR合入秒界未单查。固定SHA见[sglang-torchtpu 736916e2](https://github.com/sgl-project/sglang-torchtpu/commit/736916e2b89d8c15a91b74f7cfde25003505e364)；[sglang-torchtpu 1e2c7427](https://github.com/sgl-project/sglang-torchtpu/commit/1e2c742740f419e6bd0828002b4ae68b24fcb967)。状态为main/master已合入，未据此承诺正式包已包含。

### SGLang TPU：非阻塞上传撤回，host回收需重算SWA缺口

默认host多模态放置路径已读回并等待encoder，再non-blocking上传共享行buffer没有省去等待；作者1P1D tpu7x、6-client、每组两次交替测试吞吐1.71→1.60req/s，因此两处恢复blocking。device-scatter上传仍non-blocking；最终补丁删除中途新增的复用测试，不能引用早期提交说最终仍保留它。host FULL叶回收会同时释放SWA页，现跨rank重算SWA缺口后再回收窗口，避免按旧缺口额外驱逐。**考虑采用并按实际放置路径复测**，不把non-blocking当必然优化。

**版本与来源**：固定提交日期均为11月20日，已在默认分支列表确认；PR合入秒界未单查。固定SHA见[sglang-torchtpu 7e0db935](https://github.com/sgl-project/sglang-torchtpu/commit/7e0db935dd42a9938da6c4c14a98b6ccc30e1d46)；[sglang-torchtpu 72bc71a3](https://github.com/sgl-project/sglang-torchtpu/commit/72bc71a324590d99d0096926ef210041a0219f19)。状态为main/master已合入，未据此承诺正式包已包含。

### ATOM：状态检查点不能占用注意力的零填充页

稀疏索引越界项和DSpark padding行会读物理block0；PAGE checkpoint却可能先把FP32/BF16原始状态放进去，字节被当FP8 NaN读取。作者报告受影响DeepSWE任务中约62%出现乱码工具调用。最终补丁在存在PAGE状态存储的pool构造时保留null block，即使关闭前缀缓存也保留，并按可分配容量做准入/水位记账；代价一个block，不能用保留填充页替代真实draft写入所需的lookahead分配。**建议优先采用/回归**前缀命中与投机状态；62%是作者受影响测试集比例，本轮未独立复验。

**版本与来源**：固定提交日期均为11月20日，已在默认分支列表确认；PR合入秒界未单查。固定SHA见[ATOM 48537d30](https://github.com/ROCm/ATOM/commit/48537d30e940e63862e970d403279952d6c76a20)。状态为main/master已合入，未据此承诺正式包已包含。

### ATOM：远端prefill查询要保持到真正排队，NVFP4 offload按connector准入

scheduler可能多轮查询却暂不接收请求；旧一次性yes会让后续在decode本地prefill，producer一直等WRITE_REQ，draft还可能缺lookahead页。新逻辑直到`update_state_after_alloc`排队receive才清远端prefill意图。MiniMax-M3.1另允许NVFP4目标与独立BF16/FP8 DSpark draft在进程内LMCache offload组合：按原始page字节搬数据/scale，namespace含完整投机配置和draft dtype，防跨精度读旧store。MXFP4目标、P/D重分片及其他connector仍拒绝；multi connector要求所有leaf合规。**考虑采用并回归**饱和decode与重新命中，不能外推到任意低精度P/D。

**版本与来源**：固定提交日期均为11月20日，已在默认分支列表确认；PR合入秒界未单查。固定SHA见[ATOM ecec8192](https://github.com/ROCm/ATOM/commit/ecec81921da0c27a088d542fcefad51fe83e4d5a)；[ATOM 17edc615](https://github.com/ROCm/ATOM/commit/17edc61575fd87b2aac9a7dfd94b532c07fc52fe)。状态为main/master已合入，未据此承诺正式包已包含。

### AITER：量化悬空指针、长序列索引溢出和Sol-Attn修正

per-row量化helper原返回局部vector指针，旧编译器偶然保留，ROCm10.3 hipcc7.18.26451/gfx1201复用栈后作者见65521/65536元素错误、scale仍正确；改按值返回并由调用方持有。MLA融合QK/RoPE/cache行与head偏移改INT64，新增T36864/QH128使偏移超过2^31的回归。gfx950默认256×128 MXFP8 Sol-Attn tilt二进制另修幅值，作者MI355X对FP32参考验证，64×64及无tilt不受影响；仅交付二进制，未审其内部算术。**优先核对采用版本并回归**，正常scale或旧栈通过均不足以证明量化正确。

**版本与来源**：固定提交日期均为11月20日，已在默认分支列表确认；PR合入秒界未单查。固定SHA见[aiter 2c415212](https://github.com/ROCm/aiter/commit/2c415212302208b47b7d846e80e103225fa7b432)；[aiter c813348a](https://github.com/ROCm/aiter/commit/c813348adc360f077c9bd040ee8de68bb7ade0f7)；[aiter f05d8ae7](https://github.com/ROCm/aiter/commit/f05d8ae7790814036341b8e2bdb1a18d38c0d742)。状态为main/master已合入，未据此承诺正式包已包含。

### AITER：inverse-Ulysses可直接交付量化投影布局

机内注意力输出重分布可将连续BF16 `[1,S_global,H_local,128]`送到各目标rank本地序列/全head布局，E4M3或packed MX格式可供后继投影消费。实现以MoRI对称双缓冲传输，加入extent、架构、peer指针一致检查和超时；结果payload/scales会在隔两次调用后被复用，跨stream消费者须显式join。作者加2/4/8卡及慢读者、NaN/Inf测试。**建议深入阅读后考虑复现**算子衔接与量化误差；不是任意shape/跨机A2A，也无本轮端到端加速证明。

**版本与来源**：固定提交日期均为11月20日，已在默认分支列表确认；PR合入秒界未单查。固定SHA见[aiter 50bfa78d](https://github.com/ROCm/aiter/commit/50bfa78d0006907a074d8a963bad118f2899304d)。状态为main/master已合入，未据此承诺正式包已包含。

### Transformer Engine：capturable Adam共享偏置修正，scale padding并入swizzle

capturable FusedAdam因step在设备，原每线程算两次pow；现在每block算一次后共享，作者8个各134M FP32元素tensor测7.5→6.0ms、权重/矩/梯度逐位相同，但提交未披露GPU型号，不能拿该数与他机比较。新增覆盖多次图重放、master weights和grad-scaler跳步用例。小MXFP8尺度另把compact padding与swizzle融合，选中buffer需16字节对齐。**考虑聚焦复现**优化器核和scale布局；不把内核改善折算为完整训练步同比例提升。

**版本与来源**：固定提交日期均为11月20日，已在默认分支列表确认；PR合入秒界未单查。固定SHA见[TransformerEngine 0f65420f](https://github.com/NVIDIA/TransformerEngine/commit/0f65420f8e4065d8fea18d636244f7f10eb4e1a5)；[TransformerEngine 6af366a0](https://github.com/NVIDIA/TransformerEngine/commit/6af366a0c010d458496a8dcc9eaa2bc5730f8643)。状态为main/master已合入，未据此承诺正式包已包含。

### 昇腾：KDA小范数修复必须跟随硬件原生归一化

先前RecurrentKDA升级后A3对平方范数加1e−6，RecoverSSM仍把小非零向量归一为单位长；#22523修verify和accepted-state重建并恢复跳过的测试，作者A3/CANN9.3、FLA26.10 feature六振幅及ragged测试通过。**同日最终#22686再纠正A5**：A5该版原生实现仅对非零范数除法，不加epsilon，因此用`USE_NORM_EPS`按设备分支，不能沿用早先“所有设备都加epsilon”的结论。**优先考虑采用最终组合并复验**原生/verify/commit一致性；新A5补丁自身未附完整新设备回归结果。

**版本与来源**：固定提交日期均为11月20日，已在默认分支列表确认；PR合入秒界未单查。固定SHA见[vllm-ascend 6ea27fff](https://github.com/vllm-project/vllm-ascend/commit/6ea27fffc48fcd9ec3bf119986b1923dac3ba304)；[vllm-ascend 7a7ed073](https://github.com/vllm-project/vllm-ascend/commit/7a7ed073c481d5e3a2fc2194c5b879efb2d373c0)。状态为main/master已合入，未据此承诺正式包已包含。

### 昇腾：HC与QSA融合撤回，MM+ReduceScatter保留严格门槛

11月19日A3 HC pre/post优化及QSA preparation融合在20日撤回，QSA回退仍保持当前packed-input ABI、V2索引和request页表；本轮保留状态变化，不把旧优化继续列为可用。另为MiniMax-M3注意力O投影接入BF16矩阵乘与ReduceScatter融合，仅纯prefill（num_prefills>0且num_decodes=0）融合，decode/mixed batch回旧路径；另需SP（序列并行）、token256..16383及dtype/layout/无bias/无LoRA等门禁，失败不会换一种collective悄悄重试。作者TP4+DP4+EP、BF16、输入64–2048/输出1、64 prompts、并发16、三次测量中位数报告1.16–1.30倍吞吐，**未给设备型号且仅单token输出**，不可当长生成收益。**建议核回退版本并考虑定向复现**。

**版本与来源**：固定提交日期均为11月20日，已在默认分支列表确认；PR合入秒界未单查。固定SHA见[vllm-ascend c6bf308c](https://github.com/vllm-project/vllm-ascend/commit/c6bf308c345a3ab7096dd6e90ca5493020f63c8c)；[vllm-ascend 6f9ec00e](https://github.com/vllm-project/vllm-ascend/commit/6f9ec00e4848ab5f4ce4f27938ea82e6b5d9b703)；[vllm-ascend 12d75d4c](https://github.com/vllm-project/vllm-ascend/commit/12d75d4c45b7acc5283f91107f2ca262992491a3)。状态为main/master已合入，未据此承诺正式包已包含。

### TorchNPU：原生profiling与独立条件分支流

ACLprof通过Kineto PrivateUse1接入`torch.profiler.profile`生命周期，每profiling窗口独立目录；启动失败若清理也失败，保留原始异常及未确认释放的资源，阻止新session，避免假定cleanup已完成。条件图IF/WHILE另从共享轮转流池改用独立创建、RAII持有子流，防深嵌套绕回正在捕获的流而报107027；外部流的capture/event/query和通信当前流限制随之调整，任务队列对外部流的原限制仍在。**建议深入阅读后复现**嵌套捕获、失败清理；本轮未确认最低正式包和全算子覆盖。

**版本与来源**：固定提交日期均为11月20日，已在默认分支列表确认；PR合入秒界未单查。固定SHA见[pytorch e6bb1260](https://github.com/Ascend/pytorch/commit/e6bb1260c53a93ccb4173caf0cd8a1dc9d433efd)；[pytorch 5e2d23f7](https://github.com/Ascend/pytorch/commit/5e2d23f74546109e5a3fa417b664af70deb46d91)。状态为main/master已合入，未据此承诺正式包已包含。

### CANN HCCL：修复图编译期野stream、对称广播布局和tag容量

Ascend950/hostDPU图模式离线选路的OpParam.stream未初始化，可表现为假ERROR或SIGSEGV；补丁给stream/comm确定默认值、构造点清零、空stream不查询捕获态。作者全量编译和25项hostDPU UT通过，**上板复跑仍待完成**。对称内存broadcast另以`fullLayoutTransit`使多跳中转直落用户output最终布局，序列化尾部追加，旧/截断载荷保留false；tag合法长度从191改159字符以匹配160字节实际buffer（含终止符）。**优先核对相关部署并考虑回归**；本轮未将源码提交映射为已发布CANN包。

**版本与来源**：固定提交日期均为11月20日，已在默认分支列表确认；PR合入秒界未单查。固定SHA见[hccl 727c577e](https://gitcode.com/cann/hccl/commits/detail/727c577e99ce005604615a3cc65635d226ca4e23)；[hccl 85ab51b3](https://gitcode.com/cann/hccl/commits/detail/85ab51b3bf9b2ea41e15dd9c79d8f3171150a3cd)；[hccl 3a0fbb47](https://gitcode.com/cann/hccl/commits/detail/3a0fbb47c22796ce5eea7edeb9fedac2d467ca58)。状态为main/master已合入，未据此承诺正式包已包含。

### CANN：AttentionUpdate竞态与稀疏MLA metadata修正

A2/A3 AttentionUpdate为转换/搬运补V→MTE同步，并按完整DMA分片跨度对齐，防下一片覆盖Cast仍读的UB；作者称300内存检测与8384功能例通过。Sparse MLA metadata删除“只有原始稀疏KV就直接返回top-k范围”的早退，使mask/window进入负载计算，避免错误FD归约及NaN。flash_attn的Torch扩展则在shape推导前拒绝非法layout组合，防TND输入/BSND输出静默变(0,0,N,D)空tensor；合法T=0跳过仍保留。**考虑采用并做边界回归**，不据这些单点修复声称所有稀疏模式正确。

**版本与来源**：固定提交日期均为11月20日，已在默认分支列表确认；PR合入秒界未单查。固定SHA见[ops-transformer fa1d8ead](https://gitcode.com/cann/ops-transformer/commits/detail/fa1d8eadbe5f2e05cb9c746240874e80d87c16e2)；[ops-transformer a1d4eab1](https://gitcode.com/cann/ops-transformer/commits/detail/a1d4eab109f5bf0642ff10a966d8f8218224ac15)；[ops-transformer 391c3863](https://gitcode.com/cann/ops-transformer/commits/detail/391c3863f2a983f1d00d1f1988755484af6364a0)。状态为main/master已合入，未据此承诺正式包已包含。

### CANN：全行MX缓存压缩和CCU AlltoAll容量界限

KvCompressEpilog合入MXFP8全行mode3与packed MXFP4全行mode4、2D/4D缓存布局及d%128门禁；新模式不沿用旧RoPE/NoPE分段含义，Torch枚举名字虽含nope_rope，文档明确整行量化。作者950PR真机逐位对照通过、性能与迁移前持平，960dt仍待CI，旧mode0/1/2保持。CCU AlltoAll另按单次≤256MB及最多63任务约束做分片/准入；MX scale要占一任务，数据预算降62×256MB，scale自身≤256MB，实际上限还受shape约束。**建议深入阅读后考虑迁移**，既不是新增全场景加速，也不能把总预算当单通信无限制。

**版本与来源**：固定提交日期均为11月20日，已在默认分支列表确认；PR合入秒界未单查。固定SHA见[ops-transformer cce6bc97](https://gitcode.com/cann/ops-transformer/commits/detail/cce6bc97326452402843f88a34c153a90c9fc110)；[ops-transformer 53ee1250](https://gitcode.com/cann/ops-transformer/commits/detail/53ee125049bc90c19a41ae41e573200c66b22e1a)。状态为main/master已合入，未据此承诺正式包已包含。

### Gaudi：KV异步保存完成后才重用块；图像归一化保留舍入顺序

HPU runner现在forward前先`handle_preemptions`再bind metadata，补齐异步offload save尚未完成就重用KV块的窗口。作者明确此风险来自代码审查，未在HPU复现内容损坏，CI也未核抢占后offloaded KV字节，需区分修调用顺序与已证实故障率。图像uint8 patch归一化另在FP32保持减均值再除标准差，再转vision dtype，防单affine融合改变近均值BF16舍入；浮点patch路径保持原行为。**考虑采用并设备回归**前缀重用及视觉embedding。

**版本与来源**：固定提交日期均为11月20日，已在默认分支列表确认；PR合入秒界未单查。固定SHA见[vllm-gaudi 4685af69](https://github.com/vllm-project/vllm-gaudi/commit/4685af698d11d610e0adfc5bd02556561e08976e)；[vllm-gaudi a1b0e4b0](https://github.com/vllm-project/vllm-gaudi/commit/a1b0e4b025356882c93c7902bb78959503aeb23a)。状态为main/master已合入，未据此承诺正式包已包含。

### MUSA：投机draft保留按请求/步骤隔离，FP64 Gumbel提前拒绝

FA3＋full graph的multi-step draft metadata就地更新；结构化输出draft按请求保留，非deferred步骤也取回，D2H拷贝各自使用stream/event，防其他batch清掉在途draft或后步骤延迟前一次拷贝。S5000驱动5.2、torch2.13+musa5.2另确认UVA往返可用，但`--use-fp64-gumbel`因MUDNN缺float64 argmax在配置时明确拒绝；不能解释成MUSA不支持真正FP64，更不能悄悄降FP32。**建议采用门禁并回归**结构化投机/图重放，本轮未跑MUSA设备。

**版本与来源**：固定提交日期均为11月20日，已在默认分支列表确认；PR合入秒界未单查。固定SHA见[vllm-musa badd1c27](https://github.com/MooreThreads/vllm-musa/commit/badd1c277352db285a81da8111fdd69ef98ea342)；[vllm-musa 658bd069](https://github.com/MooreThreads/vllm-musa/commit/658bd069778a67d69f8bf788f54a776909b235b3)。状态为main/master已合入，未据此承诺正式包已包含。

### llama.cpp：循环Inkling 2 Mini与Vulkan chunked GDN进入主线

Inkling-2-loop新增GGUF转换、独立架构/元数据和loop图，包含varloop标记，说明主线已接入循环模型，不等于任意GGUF量化及所有硬件已验证。Vulkan新增chunked Gated DeltaNet，按cooperative-matrix及设备feature选择变体，补workgroup/shared-memory处理与NVIDIA shader编译规避。**考虑在目标设备复现**混合模型prefill与循环推理；所开补丁未给可直接比较的整网吞吐，不宣称普遍提速。

**版本与来源**：固定提交日期均为11月20日，已在默认分支列表确认；PR合入秒界未单查。固定SHA见[llama.cpp 29b52d39](https://github.com/ggml-org/llama.cpp/commit/29b52d398b294509746ec0a3f28948a3107c0213)；[llama.cpp 16f80bd0](https://github.com/ggml-org/llama.cpp/commit/16f80bd0c0fa6b27e7d0ec7fcb5a0e0dc5f3fd47)。状态为main/master已合入，未据此承诺正式包已包含。

### MLXcel：显式INT4潜在缓存仍未通过默认质量门槛

新增实验`--mla-latent-quant auto|fp16|int8|int4`；auto维持原门禁，显式选择可越过原allowlist。INT4用group64 affine codec、composed decode，DSA indexer key仍INT8；缓存adoption拒绝不同latent宽度。固定报告在M3 Ultra/Metal、teacher forcing、`--decided 2`、1K/4K后续小样本中，四个模型虽0 decided分歧，却都超过0.11-logit near-tie界；GLM4.7在1K/4K后各有3个decided分歧。LongCat-Flash-Lite 16K decode FP16/INT8/INT4为99.3/50.2/54.5tok/s，省存储明显却更慢。**建议继续观察/定向复现**；真实DSV3.2质量、CUDA/ROCm及多个大模型未测，低perplexity也不能等同答案更好。

**版本与来源**：固定提交日期均为11月20日，已在默认分支列表确认；PR合入秒界未单查。固定SHA见[mlxcel d6ef8437](https://github.com/lablup/mlxcel/commit/d6ef8437faed7fca7fe8ee3a9c476a35ef45ba39)。状态为main/master已合入，未据此承诺正式包已包含。

### AMD FlashInfer：API兼容与实际FP8缓存互通要分开看

AMD 11月20日公布 `release/0.6.18+amd.2` 源码路线：自动在自有HIP与AITER间按dtype/layout/mask/架构路由，增加FP8 prefill、POD、MoE、MLA等。支持集固定ROCm10.1/PyTorch2.12/Python3.12、gfx942/gfx950，暂无wheel/预构建镜像。关键边界是自有kernel只读FNUZ FP8；gfx950 AITER FP8 prefill写的OCP KV不能交给现有decode读取。torch.compile还需单独K/V allocation，避免该PyTorch版对写入大tensor视图的静默误编译。**建议深入读支持矩阵后考虑迁移**，示例仅两层随机权重，不能当模型质量或吞吐验证。[官方正文](https://rocm.blogs.amd.com/artificial-intelligence/flashinfer-release3/README.html)。

<a id="topic-7"></a>
## 评估与可靠性

### Thinking Machines Inkling2 Small：训练mask标记与DAYJOB数字一起修订

11月20日主卡`885d78bfca61a7e9a498e33cfa7ad35850533b8c`及MXFP8`14f2cb29a5d0f4c1fda8f645fa339e936c2147a2`：chat template将assistant渲染包入`{% generation %}`区段，新增generation config明确eos200006/pad199999，便于下游正确识别assistant token区间；本轮没有实测mask API或重训。DAYJOB Healthcare6.4→11.2%、Finance14.3→13.8%，是卡内评测更正，未提供变更原因，不解释为当天新权重能力提升。**建议使用模板的SFT流水线检查mask，并更新历史指标**。[固定卡](https://huggingface.co/thinkingmachines/Inkling-2-Small/blob/885d78bfca61a7e9a498e33cfa7ad35850533b8c/README.md)；[固定模板](https://huggingface.co/thinkingmachines/Inkling-2-Small/blob/885d78bfca61a7e9a498e33cfa7ad35850533b8c/chat_template.jinja)。

### OpenAI MedChemBench：新公开评估把结构识别和候选决策能力分开

官方文章11月20日，本轮公开发布发现；配套PDF内部日期为**11月9日**，不能把报告撰写日期当公开日。99题取自117篇论文/专利、761结构，研究决策点后的历史结果从prompt中隐藏。PDFp7—9：结构/科学解释分别占平均rubric约37.49%/35.78%，排序选择17.51%、数值/类别8.92%；整体高分主要不来自真正候选选择。结构用RDKit精确canonical isomeric SMILES，解释用GPT6 Astra-medium裁判；每任务每model-effort十次保留单turn回答、等权平均，95%区间按99任务bootstrap，不是模型差的配对检验。

官方文章新增GPT6.2 Astra-xhigh84.5%，PDF仍以GPT6 Astra-max83.83%为最高，**二者是不同结果快照**。GPT6 Astra-xhigh结构95.4%但候选选择/预测仅约半分。无工具、反馈、实验循环；公开源训练污染不能排除，API失败恢复只统计捕获完成，旧Sol/Luna在9月图像编码修复前测得。本轮取得27页报告，未找到公开题集/评分代码交付承诺，不报“已开源可复跑benchmark”。**建议深入阅读评分构成和失败例**，适合诊断多模态证据整合，不能当模型可提升实际药物研发决策的证明。[官方正文](https://openai.com/index/introducing-medchembench/)；[固定PDF](https://cdn.openai.com/pdf/09c4ec60-a32e-4652-837b-1811368a3bb0/medchembench-paper.pdf)。

### 蚂蚁 LingSante MedGuard：医疗风控报告与实际数据集同时核对

报告`2611.25509v1`原始11月19日，本轮11月20日固定数据卡 `7703cc37190804e24781be7bc555d4302fe2b93e`新增完整说明与可下载评估。方法用一个风险教师和六个能力教师先监督蒸馏、再多教师OPD，覆盖safety/compliance/ethics；Table4的27B Full等权11子集F1为87.4，Mixed-SFT82.3；4B为82.3对相同规模Qwen3.5的76.3。这里是**不同benchmark等权平均F1**，不是14,588样本整体micro-F1，粗体/†也不代表统计显著。

本轮实际读取三份JSONL：Safety10,248、Compliance3,808、Ethics532，共14,588；每份safe/unsafe各半，字段含source/upstream_dataset，Rx-I/Rx-C不含候选response，不能套同一prompt。这是原始数据抽核，不是独立模型复评。卡称578张图，图像计数本轮未独立重算；Full11子任务还含拆分定义，不能由三文件名推断只有三benchmark。Apache2.0覆盖项目代码/新增注释与改造，第三方图文保留原许可。模型部署脚本可用不等于已公开相应权重。**建议考虑复现评估与蒸馏消融**，先防训练/评测同源与人工标签偏差；报告§6明确不能据当前子集保证其他临床人群/模态可靠性。[论文](https://arxiv.org/html/2611.25509v1)；[固定数据卡](https://huggingface.co/datasets/inclusionAI/LingSante-MedGuard/blob/7703cc37190804e24781be7bc555d4302fe2b93e/README.md)。

### Blind Spots Travel：judge的类别性盲点会迁移，更多知识不一定迁移（2611.25191v1）

**时间与版本**：v1提交2026-11-18 21:13:28 UTC；本轮11月20日分类公告恢复，首次公开秒界未确定。

项目Blind Spots Travel，Liang Yue等。814次RL运行让7–8B策略学事实短答／拒答，区分单个误奖答案记忆与类别偏见泛化。某Qwen judge在12个正确训练例中拒绝11个旧国家首都，120步LoRA-GRPO后Llama对未见旧国家首都拒答98%；禁止“I don't know”后94%仍答对，说明主要是回答策略而非知识丢失。只改这12个判决就大幅改变类别拒绝，较总体accuracy／拒答率更能定位问题。**限制**：短闭卷QA、KL0.2、手建首都probe；真实正确性部分借助Sonnet audit，人类一致93%；不能外推开放式Agent全部奖励。建议把类别级选择性拒绝和forced-answer控制加入RL评估。[正文§2–5／Tables1–2、Limitations](https://arxiv.org/html/2611.25191v1)

### Bradley–Terry负面结果：比较偏好拟合可扭曲本来校准的结果概率（2611.25594v1）

**时间与版本**：v1提交2026-11-19 06:42:02 UTC；本轮11月20日分类公告恢复，首次公开秒界未确定。

项目language world-model audit，Xinze Shi等。ManiSkill推方块、12离散结果、每条件80条可枚举随机分支，比较同批judge偏好用Bradley–Terry还是知道judge选择规则的likelihood训练。前者把约33–43个百分点概率移到该条件枚举从未出现的结果，理论目标最优点也预测这一步；完整channel likelihood无同样转移。**限制**：未出现不全等于物理不可能，四类几何不可能结果与有限枚举的零概率要分开；排除不可能事件后仍约29点质量偏移，但六种子下决策效用平均损失低于1点，原始约3点损失依赖继承阈值。建议复现概率、阈值重调和决策三层，不能泛称所有偏好训练都会损失效用。[PDF§3–4／Table1](https://arxiv.org/pdf/2611.25594v1)

### Anti-Hallucination训练审计：减少描述与廉价删句能解释大部分指标提升（2611.25814v1）

**时间与版本**：v1提交2026-11-19 09:34:18 UTC；本轮11月20日分类公告恢复，首次公开秒界未确定。

项目anti-hallucination accounting，Lingkai Bu等。先把base caption截到训练模型同object recall，剥离“少说”；再以500标注图拟合冻结11特征probe衡量剩余增益。LLaVA1.5-7B的八公开方法，probe在COCO7/8、AMBER8/8、ObjectHalBench6/8复现至少三分之二剩余移除量；RLAIF-V是稳定例外，InternVL部分自训DPO也不成立。**关键限制**：人审发现CHAIR误报判断的precision只有0.53–0.58（第三人条件0.68–0.71），低于预设0.80；COCO漏标也明显，不能据自动指标断言真实幻觉减少量。建议重做matched-recall和人工核验，再评判新DPO方法。[正文§4–10／Tables1–3、Limitations](https://arxiv.org/html/2611.25814v1)

### RouteVerify：路由可解码正确性、干预会改判决，仍不证明判决使用了正确性信息（2611.25780v1）

**时间与版本**：v1提交2026-11-19 09:11:52 UTC；本轮11月20日分类公告恢复，首次公开秒界未确定。

项目RouteVerify，Yucheng Du等。九MoE checkpoint、固定receiver文本，对router logits／MoE输出／残差做matched-donor与norm-random控制。router probe八模型AUROC0.80–0.88，全层router替换九模型都改变CORRECT/INCORRECT margin；但替换来自正确或错误步骤之差只有一模型显著为正，局部residual则八模型为正。**限制**：activation patch可能离开自然分布，correctness与文本／几何相关未完全隔离；九模型为回顾选集，不能由AUROC＋可干预就算“因果使用比例”。建议复现paired donor对照后再设计router verifier。[正文§4–8、AppendixB–C](https://arxiv.org/html/2611.25780v1)

### 训练数据审计：集合检测、文档识别和定位成员的统计功效相差很大（2611.26264v1）

**时间与版本**：v1提交2026-11-19 15:34:44 UTC；本轮11月20日分类公告恢复，首次公开秒界未确定。

项目multi-level training-data auditing，Yeheng Ge等。用Cauchy合并chunk证据，再校准文档p值、做集合energy检验和BH多重检验。Pythia6.9B／Pile-arXiv、每文档8×1024token，TPR@1%FPR由Mean-evidence7.4%到15.6%，AUC反由76.04到75.54；50%成员集合可几乎全被检出，但BH识别真正成员不到10%。**限制**：有限样本保证依赖non-member校准的可交换性，统计拒绝集合原假设不等于每篇属于训练，也不证明版权来源；模型／时间分布漂移必须另核。建议复现低FPR、集合power与FDR三套指标，不能只报AUC或集合成功率。[正文§3–5／Tables1–3、Figs2–5](https://arxiv.org/html/2611.26264v1)

### LLM Complexity：token长度不等于算法复杂度，表格也不支持所有相关性全面最优（2611.24943v1）

**时间与版本**：v1固定abs记2026-10-19 08:23:11 UTC；与2611编号及11月20日分类公告日期不一致，按补录保留冲突，首次公开时间未确定。

项目LLM Complexity，Yu-Hung Wu等。**补录与日期冲突保留：11月20日分类公告列出，固定abs的v1却写10月19日，不能定为当天首发**。用同一长轨迹在1K–16K预算截断后强制答题，取log预算空间accuracy曲线面积的补数，避免单纯以生成长度当任务难度。Table2的12算法任务跨模型平均Kendall为0.664，比length0.650好；平均Spearman却为0.834，略低于length0.836，与笼统“最好跨模型相关”表述不完全一致。**限制**：是特定模型／prompt／答案forcing协议下的经验难度，不是经典时间复杂度、也不对tokenizer或prompt无条件不变。建议复现跨prompt与任务排名，谨慎用于curriculum。[正文§3–6／Table2](https://arxiv.org/html/2611.24943v1)

<a id="coverage"></a>
## 实际覆盖、补录与续查

本轮收录 **122 个带稳定标识的版本事件**。正文标注回溯补录和更正。数值均来自所列来源，本轮未执行独立复训、设备压测或付费 API 实测。

### 厂商、模型与评估工件

- 157个官网/研究/评估入口，HTTP分布{"200": 99, "403": 9, "0": 31, "308": 9, "404": 9}；200可能只是动态骨架。176个归一HF组织、363目录页，主窗3805模型/2071数据集创建枚举均翻过起点；长尾语义与任意修订未完整审计。73个定向固定仓和7个长尾工件。

- 收录16项稳定版本事件，含OCRia1.5原始11月18日补录。WeMM前窗空仓不作发布，GLiDE旧模型新增报告；MedChemPDF内部11月9日与官网11月20日区分，RSS零点不能证明首发秒。

- 三份LingSanteJSONL实际计数14588行，标签/source分布已存；未复训/压测/调用付费API。MedChem27页PDF下载提取，选读评分和统计；四篇HTML论文及Inkling技术发布/模型卡选择性审读，不声称逐行完整审稿。

- SenseNova history401、部分HFtree429/LICENSE404、GitHubabuse及网络失败保留；Hy4 squash历史、Mini运行包/许可、WeMM验证冲突、HiUni权重/代码许可、MedChem题集开源、TRL草稿正式状态与继承待查区间继续查。请求记录及具体缺口见models/ledger.json。

### 论文、预训练／后训练与评估来源

- 本线实际打开 arXiv 的 cs.CL、cs.LG、cs.AI、cs.DC、cs.AR、cs.CV、cs.IR、cs.NE、cs.PF、stat.ML 十分类11月20日公告，及五分类 recent。**九分类公告正文完整；cs.AI 初次与压缩重试均有 HTTP200 但 curl exit28 截断，仍待分页补齐**。按每次已恢复 dt/dd 记录合并947个分类ID；当前完整工件解析的 new/cross 558、replacement 310仅是当前可见集合，不能据此宣称十分类全量。
- 修正首次 Atom 请求的 curl 方括号glob客户端错误后，原始API返回完整XML，722条且totalResults=722；**响应把请求的lastUpdatedDate改成submittedDate，并把秒级边界降为分钟**。这恢复了近期提交线索，不证明所有版本更新或精确扫描窗已查完。分类与Atom合并共1383个候选。本线未把“提交时间”当首次公开秒界。
- 本轮33篇固定v1原始正文读到实验、结果与限制：28篇采用完整HTML，Raptor／Twindex／Complete-SVD／Budget Internalization／Bradley–Terry采用完整PDF和逐页文本；部分其他论文另有PDF备份。每篇日期、主引用、条件、限制和实践建议均写入正文。候选没有固定Top-N截断；1320个当前未审及1253个继承候选保留，目录恢复不等于语义审完。33项均未与state的既有稳定ID重复。
- 共记录129次抓取尝试，包括46次失败／截断／无效响应；初次HTTP200但未完整取到的尝试保持失败标记，重试成功另记。OAI-PMH 200响应仅9535字节且XML中断，未取得可证明完整的分页元数据。OpenReview ICLR2027／TMLR API均403；Hugging Face 11月20日聚合页连接超时。ACL EMNLP main与PMLR官方索引已打开，但未完成按日期增量的正式论文正文盘点；继承NeurIPS／SC26、旧替换版本与CPTC等缺口保留。
- **补录／日期冲突**：LLaDA-LED和CurvDelta是11月18日提交、上轮未完成的候选；Canon报告固定v1为11月12日，正文说明2025年已公开权重／代码／结果，本轮补的是整理报告；LLM Complexity固定abs记10月19日，与2611编号和11月20日公告不一致，保留冲突。其余按11月20日公告恢复；不能证明首次公开落在主窗边界内，不作“主窗没有新论文”判断。
- 中英文主题检索用于补线索；二手日报只作发现，关键结论回到固定一手HTML／PDF。未独立复训、设备压测或调用付费API；未逐仓审计参考实现与许可证全链，机制／性能均按论文归因，不能视为主线工程项目已合入的交付能力。
- 取证：papers/ledger.json；fixed-metadata.json；body-review.json；inventory.json、atom-metadata.json；candidates.json、inherited-candidates.json、replacement-candidates.json；evidence/ 中的原始页面、PDF与抓取日志。下轮从本次截止续扫并回溯72小时，同时补AI分页、OAI/替换版本和继承语义候选；全局完整语义水位仍不推进。

- 工程线尝试84仓默认分支、merged PR与release入口：本机github.com/原始文件连接超时，REST枚举252请求均403；HTML默认分支0/84、merged目录1/84、release feed2/84完成，不能冒充全窗口覆盖。web工具恢复核心六仓的普通commit入口及部分其他仓，vLLM/SGLang仅前两页部分阅读，筛选28固定commit/22主题；28份patch均打开相关机制/条件，未逐行审完大型vendor补丁，精确mergedAt秒界未恢复。slime当前可读head仍11月5日；TRL普通commit页缓存停11月12日且与release较新记录不一致，均不据此宣布主窗无更新。
- 正式release定向回读包括MCore0.20.1、TRL1.18.2、vLLM0.34、SGLang0.5.23、TorchTitan0.4和FlashInfer。FlashInfer0.7.5.post1新release已确认，但compare受限、tag尖端只有版本号，功能范围留查而未单列正文；同日main改动未被写成post1能力。其他失败、分页、后续候选与493继承pending完整保留。
- 英文/中文主题检索用于发现，最终工程结论均回到固定GitHub commit/patch或官方PR；没有设备复训、压测或独立性能复现。MCore #9534作者时间在主窗前约30分钟，作为补漏，所有patch作者时间与合入时间明确分开。

- **硬件与软件协同。** 本轮280次原始HTTP请求中120次完整200；21个Git补查全部连接失败。GitHub具体commit REST受限，由网页固定patch/正文恢复入选证据；CANN八项差分另从官方GitCode API打开。检查NVIDIA/AMD/TPU/Neuron/Intel/昇腾及MUSA/沐曦/Apple等入口；支持正文不代表实际设备复验。多个高频仓首100条未跨回溯边界，未审目录另存，CANN正式包映射、libtpu行为与失联入口继续待查。

- 全局完整覆盖水位仍为空。下一轮以本次发现截止续扫并回溯 72 小时，各来源继承尚未完成的区间。本地 `.monitor` 保存访问回执、候选及已收录版本。

- **归档同步状态。** 本轮读取GitHub索引并核对远端main仍为上次已验证提交；当前Git写入无认证、GitHub插件未安装/连接，终端与浏览器同步通道亦受阻。日报和索引保存本地并登记待补传，未产生本轮远端提交。下轮优先重读main后补传，本地正文不重复生成。
