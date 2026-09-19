# NHIS 31 项 atlas 跨年元数据状态 v3

日期：2026-09-19
状态：**LOCAL INVENTORY COMPLETE / 2022-2023 CODEBOOKS NOT LOCALLY AVAILABLE / HARMONIZATION NOT FROZEN**

## 1. 本次盘点结论

本地现有材料不足以完成 31 项 atlas 的 2022/2023 官方 codebook harmonization。仓库中可定位 2024 官方 Sample Adult Frequency Codebook PDF；2022 和 2023 本地官方 CSV 压缩包各只含年度 CSV 与导入提示 readme，不含年度 codebook PDF/HTML。readme 本身还明确建议用户到网站 codebook 对照频数，因而不能替代变量定义、代码、universe 或路由证据。

在不读取任何真实结局/性能、也不打印微观记录的前提下，本次只读了 2022/2023 CSV 的表头。结果是：24 个正式候选种子在两个年度均有同名列；7 个 atlas-only 扩展中 6 个在两个年度均有同名列，`URBRRL23` 不存在，但两个年度均有 `URBRRL`。这只能证明 schema 名称存在或发生更名，不能证明构念、代码、路由、NIU 或分组规则跨年等价。

## 2. 本地官方材料与身份

| 年度 | 本地材料 | SHA-256/内容 | 可支持的结论 |
|---|---|---|---|
| 2022 | `data/raw/nhis/2022/adult22csv.zip` | `25083298173acfff35c6635be0fbcaa3ff26b985e0a48fc5e7d9788761864dba`；只含 `adult22.csv` 与 `readme.txt` | 官方源压缩包身份、表头列名；不支持代码本语义 |
| 2023 | `data/raw/nhis/2023/adult23csv.zip` | `e6f0918e683e1d756e4298470737b4c99777e8a90ea9ee7488544b917f058585`；只含 `adult23.csv` 与 `readme.txt` | 官方源压缩包身份、表头列名；不支持代码本语义 |
| 2024 | `docs/paper/manuscript_readiness_20260918/transformations/primary_sources/adult-codebook-2024.pdf` | `04ef4aa4b86c1b341cfdbae4389d9d7726584c92e579e4acac98b7c240fda3c9`；636 页；NCHS/DHIS | 2024 官方描述、代码、频数与页面证据 |

压缩包内 2022 readme 的流式 SHA-256 为 `9c5bf485aedc99835400dac526f37a0c409d3f7fb23bc82788be3f70e03f041f`；2023 readme 为 `12d39526ae9e3f8ad8a8266633c4cf524f72d3a1a55fcca6c1d5c5dd8cc8e86a`。本 gate 无网络 allowlist，未尝试下载缺失 codebook。

`configs/nhis/features.json` 和 `artifacts/nhis/features/feature_registry_audit.csv` 含有既往“2022-2024 Adult Codebook”说明及年度可用性声明，但它们是本地派生注册/审计资产，不是本次可逐页复核的 2022/2023 官方 codebook，不能单独把 A1 跨年元数据状态提升为 frozen。

## 3. 31 项逐项状态

`schema present` 仅表示 CSV 表头含精确同名列；`name transition` 表示发现候选前身名称，但没有 codebook 语义等价证明。

| 变量 | 层 | 2022 schema | 2023 schema | 当前跨年状态 |
|---|---|---|---|---|
| SEX_A | anchor | present | present | annual codebook verification blocked |
| HISPALLP_A | anchor | present | present | annual codebook verification blocked |
| DISAB3_A | anchor | present | present | annual codebook verification blocked；构造项仍需逐年核验 |
| AGEP_A | seed | present | present | annual codebook verification blocked；分组需人类冻结 |
| EDUCP_A | seed | present | present | annual codebook verification blocked |
| REGION | seed | present | present | annual codebook verification blocked |
| PCNT18UPTC | seed | present | present | annual codebook verification blocked；分组需人类冻结 |
| PCNTLT18TC | seed | present | present | annual codebook verification blocked；分组需人类冻结 |
| RATCAT_A | seed | present | present | annual codebook verification blocked；收入插补/分组需核验 |
| EMPWRKLSW1_A | seed | present | present | annual codebook verification blocked；universe/路由需核验 |
| EMPWRKFT1_A | seed | present | present | annual codebook verification blocked；NIU/路由需核验 |
| NOTCOV_A | seed | present | present | annual codebook verification blocked |
| PHSTAT_A | seed | present | present | annual codebook verification blocked；审计角色需人类决定 |
| HYPEV_A | seed | present | present | annual codebook verification blocked；审计角色需人类决定 |
| CHLEV_A | seed | present | present | annual codebook verification blocked；审计角色需人类决定 |
| DIBEV_A | seed | present | present | annual codebook verification blocked；审计角色需人类决定 |
| ASEV_A | seed | present | present | annual codebook verification blocked；审计角色需人类决定 |
| VISIONDF_A | seed | present | present | annual codebook verification blocked；DISAB3_A 嵌套规则未冻结 |
| HEARINGDF_A | seed | present | present | annual codebook verification blocked；DISAB3_A 嵌套规则未冻结 |
| DIFF_A | seed | present | present | annual codebook verification blocked；DISAB3_A 嵌套规则未冻结 |
| COMDIFF_A | seed | present | present | annual codebook verification blocked；DISAB3_A 嵌套规则未冻结 |
| COGMEMDFF_A | seed | present | present | annual codebook verification blocked；DISAB3_A 嵌套规则未冻结 |
| UPPSLFCR_A | seed | present | present | annual codebook verification blocked；DISAB3_A 嵌套规则未冻结 |
| USUALPL_A | seed | present | present | annual codebook verification blocked；结局邻近性需人类复核 |
| URBRRL23 | atlas-only | absent；`URBRRL` present | absent；`URBRRL` present | name transition identified；2022/2023/2024 scheme equivalence not established |
| MARSTAT_A | atlas-only | present | present | annual codebook verification blocked；分组/未知状态需核验 |
| NATUSBORN_A | atlas-only | present | present | annual codebook verification blocked；审计角色需人类决定 |
| CITZNSTP_A | atlas-only | present | present | annual codebook verification blocked；与 nativity 的相关/嵌套处理未定 |
| FDSCAT3_A | atlas-only | present | present | annual codebook verification blocked；item/recode/universe 需核验 |
| ANXEV_A | atlas-only | present | present | annual codebook verification blocked；诊断史角色需人类决定 |
| DEPEV_A | atlas-only | present | present | annual codebook verification blocked；诊断史角色需人类决定 |

汇总：每年 30/31 个 atlas 名称精确存在；24/24 正式种子精确存在；6/7 atlas-only 扩展精确存在；第 7 项存在 `URBRRL` -> `URBRRL23` 名称变化线索。**0/31 项在本次盘点中获得了 2022 与 2023 官方年度 codebook 的逐页语义核验。**

## 4. 已核验、缺失与待决定

### 已核验的执行事实

- 2022/2023 官方 CSV 压缩包的路径、hash 和内部成员；
- 两个年度 CSV 表头对 31 项的名称存在性；
- 2022/2023 均使用 `URBRRL`，2024 atlas 使用 `URBRRL23`；
- 2024 官方 codebook 的本地身份、页数和作者机构；
- 本次没有读真实性能、没有输出微观行、没有访问 2025。

### 缺失的官方证据

- 2022 Sample Adult 官方 codebook 本地副本、来源 URL、获取时间、bytes、SHA-256 与页码索引；
- 2023 Sample Adult 官方 codebook 的同一组证据；
- 31 项逐年的构念描述、substantive/missing/NIU codes、universe、路由、构造来源和频数；
- `URBRRL` 与 `URBRRL23` 的分类方案、参考年份和可比映射；
- 可复算的逐项 harmonization matrix 与冻结 hash。

### 必须由人类/科学责任人决定

- 年龄、教育、收入、家庭计数和婚姻等变量的最终分组/参照；
- 临床、社会处境、功能与 access-context 变量允许支持公平主张还是只作描述切片；
- DISAB3_A 及六个构成项、nativity/citizenship 等相关或嵌套证据如何计数；
- 正式 expanded O 的准入、最多 5 个验证 O 的选择及其规范依据。

## 5. 完成 harmonization 的安全下一步

需要新的、明确的 gate 或主管授权，将 CDC/NCHS 官方域名列入 HTTPS allowlist，依协议以 `.part` 原子下载 2022/2023 codebook，记录 URL/时间/bytes/SHA-256，并验证 PDF 可读性与页面。随后为 31 项逐年建立 machine-readable matrix，至少包含 variable、year、official label、page、codes、missing/NIU、universe/routing、construct/version、mapping decision、reviewer 和 evidence hash。

在该工作完成并经 Reviewer 审查前，本文件只支持 `SCHEMA_PRESENCE_INVENTORIED`，不支持 `CROSS_YEAR_HARMONIZED`、A1 freeze 或 A3 启动。
