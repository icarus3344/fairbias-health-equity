# Codex 接手执行与阶段验收

2026-09-15。用户已授权 Codex 完整推进：R12 验收、核心修复、对比适配、调查统计验证、本地 NHIS benchmark、论文结果整理；由 Codex 验收后进入下一阶段，不逐轮请求确认。允许 Luna 承担边界明确的基础编码。本文为当前执行规格，取代旧提示中限定 Gemini 只做 R12、不得开展后续工作的范围；保留历史提示作为历史记录。

## 不变的研究约定

- 以 `FAIRBIAS_PRIMARY_BENCHMARK_MASTER_PLAN_20260913.md` 为研究设计，近期候选见 `FAIRBIAS_BENCHMARK_EXPANSION_CANDIDATES_20260915.md`。FairBias-BM application-v1 为主方法；BM→AE、Joint 为登记扩展。
- 2022 F/C、2023 S、2024 T；T 已历史观察，称回顾性跨年评价。不依据 T 选择方法、预算、阈值或特征。
- 四臂语义、人群、同人群特征消融、调查权重和设计变量隔离不变。仅使用本地 NHIS；MEPS 不在本次范围。
- 用户尚未回复比较规模和资源选项时，先完成共同必需的审查和轻量合成实现，不据未回复提高资源预算。既有本地保守预算为单重型拟合串行、4 GiB、30 分钟；正式矩阵运行前登记实际资源与准入情况。
- 不改动继承基线、已有封存包和运行结果；不暂存、提交、推送。记录接手前已有修改，源码修改在新的验证记录中登记。

## 执行顺序与结束条件

1. **交付收尾**：独立核对 R12 清单、R10 22 份证据、120 项冻结来源。实际执行最小反例，不接受缺失或伪标通过的诊断。若历史封存器仍有问题，保留其未通过状态，由 Codex 的独立、限定本次冻结证据的核验记录接替后续信任依据；不把旧封存器推广为未来运行验证器，不再扩展通用封存平台。明确区分历史工具的缺陷与已验证模型证据。
2. **核心实现**：接通 F 学得的 FairBias 表示与最终模型，统一 p/q/硬决策；候选、部署编码和训练权重一致。有效算法路径、非恒等见证、F/C/S/T 隔离、原始 y 保留和输出语义测试通过后验收。
3. **方法适配**：核验官方论文/源码、锁版本及能力卡，传统对照优先；近期候选按选定规模逐个接通。无法适配者记录具体原因，不能伪装完成或按 T 表现筛选。匹配骨干，避免全笛卡尔积。
4. **调查统计**：年度完整设计域、配对复制权重、零支持、单 PSU 层策略、非光滑 gap 推断经独立参考及合成行为检查；不支持的推断不冒称有效。
5. **正式运行**：先只查数据支持与资源；冻结候选/种子/规则/比较家族，再运行 S 选择和 T 固定评价。每次运行独立目录；失败、耗时、警告和未支持条件完整保留。
6. **论文交付**：完整性能与公平性权衡、配对不确定性、四臂/跨年/消融、资源与局限、可复现方法和结果文本。不能把软件通过写成科学优势。

## 工具与依赖

Codex 可使用官方论文和官方实现作只读检索。为此次适配，HTTPS 来源白名单为 github.com、api.github.com、raw.githubusercontent.com、objects.githubusercontent.com、codeload.github.com、openreview.net、arxiv.org、proceedings.mlr.press、proceedings.neurips.cc、proceedings.iclr.cc、fairlearn.org、aif360.readthedocs.io、lightgbm.readthedocs.io、pytorch.org、scikit-learn.org、numpy.org、scipy.org、pypi.org、files.pythonhosted.org、cdc.gov、www.cdc.gov、ftp.cdc.gov、r-survey.r-forge.r-project.org、cran.r-project.org。其他必要官方来源先由 Codex核验并补登记，无须新增 worker 权限回合。无须账号的公开依赖可安装到项目独立环境；下载使用 `.part`、落盘 SHA-256 与来源记录，区分本地校验和出版者真实性。不得上传微数据或调用远程预测 API；需要收费资源、账号授权或新数据协议时再集中询问。

## 协作

Luna 默认只接触源码、文档及非敏感合成数据，负责清楚限定的模块/测试。Codex 负责统计与算法判断、集成、独立验证和阶段决策。子任务只写指定文件；有冲突时先协调，禁止还原他人的修改。每次交付使用协议第 9 节报告，worker 不自批验收。
