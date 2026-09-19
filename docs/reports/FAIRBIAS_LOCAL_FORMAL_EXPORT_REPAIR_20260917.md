# FairBias 主分析正式导出修复与验收

2026-09-17。主管结论：**本地正式导出 v2 已验收**。此前 `JSON_OBJECT_REQUIRED` 是报告格式处理错误，未使已完成的训练、925 个模型的预测或调查统计失效。本次没有重新训练、选模、计算预测或改变统计定义，也未连接任何服务器。

## 修复及版本边界

原冻结 `catalog_reporting.py` 的 SHA256 为 `ac8fde5e9cacf5d746ee0da91f6bfbfdaac156c8db1290f94210ae2aed9febb6`。主管使用这个原始文件与实际、哈希吻合的补充计划 Markdown，独立复现了 `JSON_OBJECT_REQUIRED`。原因是补充证据在核验文件哈希后，被无条件按 JSON 对象解析。

新版本将两种职责分开：所有补充证据仍必须通过原始 SHA256；仅用于绑定来源的 Markdown 或二进制材料不再被解析。study、selection、release、registration、catalog 等有内容语义的 JSON 文件仍须满足原来的严格结构验证。错误哈希、缺文件、伪装成 Markdown 的 release 仍拒绝导出。

另新增本地备份路径适配。文件读取按显式路径映射访问已经核验的本地副本；原始 JSON 中的服务器路径、运行身份、模型身份和哈希均不改写。不存在按修改时间、成功状态或“找到一个哈希吻合的副本”搜索替代结果的逻辑。输出不能落入备份或冻结源码目录。

原分析的 90 个冻结源码文件保留在备份，逐文件核验通过。新导出运行独立保存 92 个源码文件和授权计划；原 90 个文件中只有报告格式器有改变，新增的是备份读取适配器和导出入口。已验收统计结果仍对应原分析版本，导出版本明确为 `local_archive_export_v2_20260917`。原 v1/v2 控制器的失败记录没有被改成成功。

仓库基线另有一项历史记录差异：14 个继承文件与保护 tag 一致；`.gitignore` 与当前 HEAD 一致，但在既有提交 `b595e59` 已增加演示文稿、构建脚本及 PDF 忽略项，因而与保护 tag 不完全一致。本次没有修改它，也未擅自回滚；该既有治理差异不能写成“15 个保护文件全部与 tag 相同”，不影响本次冻结统计输入的逐文件验收。

## 已完成产物

正式目录：`artifacts/nhis/paper_export_v2_20260917/report/`。

| 文件 | 内容与分母 |
|---|---|
| `paper_results_tau_010.csv` | 138 行，冻结主分支 tau=0.10 的报告条目 |
| `paper_results_auxiliary.csv` | 334 行，其他阈值、固定消融等条目 |
| `paper_paired_contrasts.csv` | 300 行，保留配对模型身份与家族区间 |
| `paper_failures.csv` | 259 个未获 VALID 的主登记任务，包括预算、超时和其他失败 |
| `not_supported_conditions.csv` | 4 个不支持条件 |
| `registered_job_status_counts.csv` | 主登记 8,054 个任务的状态分母 |
| `attempt_status_counts.csv` | 9,166 次历史尝试，单独计数 |
| `paper_results_draft.md` | 聚合结果的方法说明草稿 |
| `report_manifest.json`、`export_receipt.json` | 输入、输出、运行源码、路径映射和执行环境绑定 |

138 行不等同于 138 个主要假设检验；472 行也不是 472 个独立样本。主检验家族仍是原登记的 10 个配对、20 个端点，未因导出扩大。三条待评价的补充分支只作为来源证据保留，没有进入主结果。

## 验证证据

- 既有报告测试与新增回归合计 **76 passed in 18.12s**。覆盖非 JSON 补充证据、篡改/缺失、release 语义、服务器路径整体迁移、历史失败完整性、源码闭包、路径逃逸和不允许覆盖证据。模拟中禁止加载模型或预测数组。
- 本地正式导出耗时 **117.71 秒**，完成全部目录元数据与来源验证。
- 主管使用独立核验脚本，逐行比对 472 行选择身份、种子模型、六类点估计、BA/EO 区间，以及 300 行配对身份、差值和家族区间；重新校验所有导出文件、新运行源码、原冻结源码和主要输入哈希。
- 主模型数 925 不变；主任务仍为 7,795 VALID、176 FAILED、40 BUDGET_EXHAUSTED、43 TIME_LIMIT。历史尝试与注册任务分母没有混用。
- study SHA：`b9531ebba6c95e66489255eb0a232ce2a67b585e48799673d8d0c6dad1a10b1e`。
- selection SHA：`ef11fe4864e0fc5d497c3c5731d1d633ecf16f2b038cca2ab03d661906b87838`。
- merged summary SHA：`f2c3edee002558de514a7dadc7598575e9be340a73d7e2e79d360fdf7cd4690f`。
- report manifest SHA：`e7960ffcdb6d1117532de747913c117c9af4ea28e92df907e0e3bbfbd99af924`。
- export receipt SHA：`2cad880eae02ecc156b506a15730e08cdcb7f5653b4cf062ea14932f030a043a`。

监督验收：`docs/reports/fairbias_recovery_20260917/evidence/local_formal_export_v2_accepted_20260917.json`。准确复现与合成验收证据在新导出目录的 `control/`。复核入口为 `scratch/fairbias_analysis_20260917/verify_local_export_v2.py`；正式入口为 `scripts/export_nhis_catalog_archive.py`。重现必须使用新的空输出目录与新的、绑定该输出路径的计划，不能覆盖现有报告。

## 仍需完成

1. AE40、strict Joint、BM22 的独立补充 manifest、冻结及评价入口需要实现和合成验证。AE40 仅 6/8、strict Joint 仅 2/8 配置有完整有效种子集合；BM22 仅局部恢复，不能构成增强预算全矩阵。该范围没有因看到 T 排名而改变。
2. 补充正式的亚组、队列及计算成本附表，把多节点/修复/超时成本与成功模型成本分开。不可把并行任务耗时之和称为服务器实际租用时长，也不可把不同硬件运行时间直接解释为方法速度优势。
3. 在既有真实轨迹图上完善论文叙事、图注和机制证据；可解释性指可追溯的特征变换与几何路径，不代表因果解释或总体公平性保证。
4. 完成论文写作与局限整合。已有证据足以写主稿，尚不能称为投稿包完成，也不支持 FairBias 全面优于所有方法。

**CPU 可以继续关闭并保留磁盘。** 后续先做本地开发；只有独立补充入口验收完成、确实需要额外计算时才安排短时服务器。不要为等待开发结果而保留空闲算力。
