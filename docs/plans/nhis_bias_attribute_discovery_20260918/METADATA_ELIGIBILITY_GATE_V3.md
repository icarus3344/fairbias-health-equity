# NHIS metadata and eligibility gate v3

Gate：`A1_METADATA_ELIGIBILITY_V3`

状态：`AUTHORIZED_NON_PERFORMANCE_WORK_ONLY`

## 目的

在不读取任何 NHIS 结局/模型性能、不生成预测、不访问 2025 的前提下，补齐 2022/2023 官方 Sample Adult codebook 证据，建立 2022-2024 可复算 harmonization matrix，并从完整 2024 public-use Adult codebook 建立变量 eligibility ledger 和排除分母。该 gate 不选择正式 O，不批准 delta，不授权 A3。

## 网络 allowlist

只允许以下四个 HTTPS URL；禁止重定向到非 `cdc.gov` 主机，禁止其他网络访问：

1. `https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHIS/2022/Adult-codebook.pdf`
2. `https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHIS/2022/Checksum-Filelist.pdf`
3. `https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHIS/2023/Adult-codebook.pdf`
4. `https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHIS/2023/Checksum-Filelist.pdf`

下载必须写入同目录 `.part` 临时文件，成功后原子改名；记录 URL、UTC 时间、bytes、SHA-256、PDF 页数、标题/年份与 HTTP 最终 URL。若 publisher file-list 提供 checksum，单独记录其算法和值；不得把本地 SHA-256冒充 publisher signature。

## 允许写入

- `docs/paper/manuscript_readiness_20260918/transformations/primary_sources/adult-codebook-2022.pdf`
- `docs/paper/manuscript_readiness_20260918/transformations/primary_sources/adult-codebook-2023.pdf`
- 同目录的 2022/2023 `checksum-filelist` 原始 PDF；
- `docs/plans/nhis_bias_attribute_discovery_20260918/` 下本 gate 的 manifest、harmonization matrix、eligibility ledger、方法说明、测试和 Section 9 报告；
- `work/` 或系统临时目录下的 PDF 文本/页面渲染中间件。

## eligibility ledger 规则

完整分母是 2024 Adult public-use codebook 的全部变量条目，不是现有 31 行 seed atlas。每个条目至少记录：变量名、module/section、label/description、page、data type、universe/routing、recode、构念角色、是否 X/Y/费用障碍邻近、是否 design/ID/weight/PSU/strata、是否 public-use、2022/2023 名称候选、嵌套/相关 cluster、准入阶段和排除理由。

机器筛选只能产生 `ELIGIBLE_FOR_HUMAN_REVIEW`、`EXCLUDED_BY_PREDECLARED_RULE`、`UNRESOLVED_METADATA`，不得直接产生正式 O。排除规则至少包括：标识符/权重/层/PSU、结局或近似复述费用障碍、预测/误差衍生字段、无成人适用域、无法建立跨年构念、纯管理/处理字段。临床/社会/功能/生活情境字段不能仅因敏感性自动排除，但须保留角色与规范性主张待人类决定。

## harmonization 规则

对现有 31 行 seed atlas 和 eligibility ledger 中进入人类复核的字段，逐年记录官方 label、codes、missing/NIU、universe/routing、construct/version、page 和 source hash。字段同名不等于等价；`URBRRL` 与 `URBRRL23` 必须单列方案版本与映射决定。无法证明等价时标为 `UNRESOLVED`，不得静默合并。

## 接受标准

- 三年原始 codebook 与来源 manifest 可复算；
- 2024 全 codebook 变量数和 ledger 行数闭合，无选择性遗漏；
- 31 行 seed atlas 的 2022/2023/2024 evidence rows 完整或明确 unresolved；
- 任何 downselect 均保留完整排除分母和规则版本/hash；
- 代表性页面视觉核验至少覆盖 anchor、功能、就业/路由、收入/插补、城乡版本、婚姻/移民、食物保障和精神健康；
- Reviewer 独立重算计数、抽样核页、检查 hash 与排除规则；
- 真实微观数据/性能行数、2023 replay 和 2025 读取均为 0。

## 明确禁止

- 读取或计算 2023/2024 真实性能；
- 访问任何 2025 微观结局/性能；
- 根据频数或性能选择 delta、O、分组、参照、方法或 family；
- 把 codebook frequency 当成模型支持度/precision；
- 改旧 runner、模型、预测、历史结果或受保护基线；
- 自批 A1/A3 或声称发现新的公平差异。
