# LLM 技术进展日报

每日自动生成的大模型领域进展汇总，覆盖模型与技术报告、预训练与架构、后训练、推理与服务、工业开源项目、评估及硬件软件协同。

文件命名：`llm-training-daily-YYYY-MM-DD.md`

## 日报索引

| 日期 | 日报 | 说明 |
|---|---|---|
| 2026-10-22 | [查看日报](llm-training-daily-2026-10-22.md) | Youtu/Flint新权重、KV生命周期修复；长训与低精度边界、Tokenizer版本纠正 |
| 2026-10-21 | [查看日报](llm-training-daily-2026-10-21.md) | TorchTPU/llm-d1.0、MemT与Glimmer报告；KV正确性、蒸馏和评测纠偏 |
| 2026-10-20 | [查看日报](llm-training-daily-2026-10-20.md) | M3.1 Flash/Mercury3、Neuron原生PyTorch；训推系统与有损投机边界 |
| 2026-10-18 | [查看日报](llm-training-daily-2026-10-18.md) | DeepSeek1017权重与评测边界；KV/专家scale修复、MoE与异构训练补录 |
| 2026-10-17 | [查看日报](llm-training-daily-2026-10-17.md) | Dory循环与低精度预训、Beam权重兑现、Hy-Memory；RL效率边界与工程正确性 |
| 2026-10-16 | [查看日报](llm-training-daily-2026-10-16.md) | Muse Spark Open与Nemotron新报告；预训练、后训练负面结果及核心工程正确性 |
| 2026-10-15 | [查看日报](llm-training-daily-2026-10-15.md) | 续扫与72小时查漏；模型工件、数值正确性、KV生命周期及预训练研究 |
| 2026-10-14 | [查看日报](llm-training-daily-2026-10-14.md) | 手动首跑；补查10月8日以来进展，附来源覆盖与缺口 |
| 2026-10-08 | [查看日报](llm-training-daily-2026-10-08.md) | 历史归档 |
| 2026-10-07 | [查看日报](llm-training-daily-2026-10-07.md) | 历史归档 |
| 2026-10-06 | [查看日报](llm-training-daily-2026-10-06.md) | 历史归档 |
| 2026-10-02 | [查看日报](llm-training-daily-2026-10-02.md) | 历史归档 |
| 2026-09-24 | [查看日报](llm-training-daily-2026-09-24.md) | 历史归档 |
| 2026-09-23 | [查看日报](llm-training-daily-2026-09-23.md) | 历史归档 |
| 2026-09-02 | [查看日报](llm-training-daily-2026-09-02.md) | 历史归档 |
| 2026-08-28 | [查看日报](llm-training-daily-2026-08-28.md) | 历史归档 |
| 2026-08-27 | [查看日报](llm-training-daily-2026-08-27.md) | 历史归档 |

## 阅读说明

日报正文注明实际扫描窗口、补录日期、版本与原始来源。论文、厂商测试和开源 PR 的结论均保留实验条件；“已合入 main”不等于已经正式发版。
