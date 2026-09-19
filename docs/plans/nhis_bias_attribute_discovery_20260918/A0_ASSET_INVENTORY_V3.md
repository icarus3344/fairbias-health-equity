# A0 资产盘点：NHIS 扩展公平审计 v3

盘点时间：2026-09-19（Asia/Shanghai）。范围：只读路径、大小、SHA-256、登记/manifest 和代码合同；**未读取或计算新的真实性能，未生成预测，未访问 2025**。

## 1. 仓库与保护状态

- 分支：`research/nhis-fairbias`；HEAD：`67e6659fa65249a8842e34af5d8969629efe4bca`。
- 工作树在本 gate 前已存在大量修改和未跟踪资产；全部视为用户现有工作，不清理、不 reset、不覆盖、不提交。
- 对 14 个受保护根文件和 `.gitignore` 执行 `git diff --name-only`，输出为空；本 gate 未改这些文件。
- v2 文件继续保留。本 gate 只添加 v3 隔离路径和测试。

## 2. 规范、计划和注册表输入

| 路径 | SHA-256 | 状态 |
|---|---|---|
| `docs/AI_EXECUTION_PROTOCOL.md` | `1a3d38929d65493f8352d6b46d32f65fa836502d08a3c2d2704dd713e5c3beac` | canonical protocol |
| `RESEARCH_PLAN_V2.md` | `1a90183fe1b3248be86bdbce3b4461443c77c6f13815282033e5de66e9f31152` | preserved input |
| `IMPLEMENTATION_HANDOFF_V2.md` | `bb6aeac0d81419fdf58a5be716589959de5a9d95b1a8cd00f4c6446239ae3600` | preserved input |
| `CANDIDATE_O_DRAFT.json` | `7a57832bf7944511190699b1ee54e4fc425d955c0e5b909ebe2f5ab4d8779007` | v2 seed list, not frozen |
| `configs/nhis/variables.json` | `fee56be0266b99988fa46010b6ecb7083029fe4d953bd18ddf36cfbf5a297a1e` | 3 anchor O + outcome/design registry |
| `configs/nhis/features.json` | `9fb42eb3b14c9d80b56476b1cbac5e4981843e020a00d7918702676026975018` | 21 core X registry |
| `configs/nhis/study.json` | `d0fa6b12b8fae5d77ddbb75aef91de9294c1c5d4b93fb5fbf3990c560dbbd964` | 2022–2024 source/role contract |

## 3. 官方元数据

本地官方 2024 NHIS Sample Adult Frequency Codebook：

- 路径：`docs/paper/manuscript_readiness_20260918/transformations/primary_sources/adult-codebook-2024.pdf`
- SHA-256：`04ef4aa4b86c1b341cfdbae4389d9d7726584c92e579e4acac98b7c240fda3c9`
- 大小：2,164,377 bytes；636 pages；NCHS/DHIS；codebook version 18 June 2025。
- 原 retrieval manifest 记录官方 CDC URL、原子下载、大小及本地哈希；无 publisher checksum，因此本地 SHA-256 只证明当前副本不可变，不证明发布者签名。
- 本 gate 重新检查 SEX_A p19、HISPALLP_A p25、DISAB3_A p181、VISIONDF_A p152、EMPWRKFT1_A p580、RATCAT_A p611 的完整页面，并复用已验收 21 个 X 的页码索引。
- 元数据扩展另核验 URBRRL23 p5、ANXEV_A p126、DEPEV_A p127、MARSTAT_A p541、NATUSBORN_A p563、CITZNSTP_A p565、FDSCAT3_A p628。它们只进入 atlas，状态为 `METADATA_ATLAS_ONLY_PENDING_DOWNSELECT`，不是正式候选，也不会触发性能扫描。
- `VARIABLE_ATLAS_V3_DRAFT.csv` 现有 24 个正式候选种子和 7 个 atlas-only 扩展条目。“官方已核对”只指 2024。2022/2023 全候选逐项跨年 codebook harmonization 尚未完成；已有局部教育/就业/收入检查不能扩写为全面核对。

## 4. 本地数据资产（只哈希，不作性能扫描）

| 文件 | bytes | SHA-256 |
|---|---:|---|
| `data/raw/nhis/2022/adult22csv.zip` | 3,654,579 | `25083298173acfff35c6635be0fbcaa3ff26b985e0a48fc5e7d9788761864dba` |
| `data/raw/nhis/2022/adult22.csv` | 28,118,163 | `2e814050ea06ca1ffed356f51979b87c20d6242b0453c928249a29bd5f1f29f1` |
| `data/raw/nhis/2023/adult23csv.zip` | 4,780,657 | `e6f0918e683e1d756e4298470737b4c99777e8a90ea9ee7488544b917f058585` |
| `data/raw/nhis/2023/adult23.csv` | 29,427,128 | `441f2bc17761cb1c755b24c1af551e34b91e084da4cce8d78888a688b1acc5ab` |
| `data/raw/nhis/2024/adult24csv.zip` | 5,543,360 | `5f926b2ec0af508fa84ebc791c83bfefa6f020724754aab599004706981abdaf` |
| `data/raw/nhis/2024/adult24.csv` | 32,782,569 | `31c5be8d630c13f6d0a232831a79654de7038d97fc4f33d803d8d2c84705c9d5` |
| `data/processed/nhis/nhis_2022_2024_core.parquet` | 1,538,218 | `7fe63d4b5e5329fd8b3b6f594fcd23b87677580dcb4b7cd831a7d534cf235bee` |
| `data/processed/nhis/nhis_2022_2024_features.parquet` | 2,740,478 | `49f415132ff0be0228f8533f9f74c48cd79ff6fa8be66db085f7329d9b083383` |

绑定 manifest：

- `artifacts/nhis/data/data_manifest.json`: `64d40a0a25131db6a8baf64f8beada422ed3899e052a4f9be11f03869e017eda`
- `artifacts/nhis/features/feature_manifest.json`: `02aea767b6a411b5a7f3285ec220721b5ead4f89dc2fbf4b2700854bc24c7184`

这些哈希说明当前本地字节身份；A3 仍须逐输入与 run manifest 重新绑定，不能因文件存在就直接运行。

## 5. 冻结模型/预测与历史评价资产

完成评价控制根：`artifacts/nhis/completion_evaluation_20260918/`。

| 控制文件 | SHA-256 |
|---|---|
| `control/admission_v1.json` | `2a83612b23048a54ce09196ec553b9881ee87bf5b5930f36fc53bc0e52d1a628` |
| `control/selection_v1.json` | `46e58b19a02ea5144b9c8fe11459c38b25db2fe9e503129125114a7e7167e69d` |
| `control/study_v1.json` | `3c4572ec327d8d97a2f108fe3c82a8619370a78981c8fe1abdb496677b2933fb` |
| `control/t_release_v1.json` | `64d34e601996ddcda1bf687b33fd9db2f6e259f94a29ae0ada41b08bef9964ff` |
| `control/supervisor_final_acceptance_v1.json` | `195a56a05cbb602ea83bc5b7b6d1169f3174b3e828a095311fd525ac0f856050` |
| `merged_summary_v1.json` | `641fe2ec523c8998cfe4c1f48ae2d5fb1e6ecbe9a5eb9e20b16af33d47c6b406` |

已验证历史范围：80 个新完成 BM+AE/Joint、269 个复用模型、共 349 个模型；2024 `known_T=true`。Arm001 2024 manifest 的 annual record-order hash 为 `48157948cf960585feccd0c90455d77acbdbc41c14c0674bf36ad36e2df22947`，domain order hash 为 `ca0ac725b7f28f804f6ba2d43a2612a60f12b920f683e639b2cd8c45c97e0e05`。

重要缺口：完成评价备份中没有 `*_predictions_S.npz`。A3 的 2023 发现不能复用 2024 T 预测或结果；须从 admission 绑定的模型/策略生成新的 2023 预测版本，并对 annual/domain record order、模型、源码和输出分别哈希。此动作尚未授权执行。

## 6. A0 结论

- `A0_ASSET_LOCATION_AND_HASHING`: **PASS**。
- `A0_OFFICIAL_2024_METADATA_SEED_MAP`: **PASS_WITH_LIMITS**；2024 的 24 个正式种子和 7 个 atlas-only 扩展可定位，扩展项仍待 downselect，2022/2023 全面跨年核对待做。
- `A0_2023_REUSABLE_PREDICTIONS`: **NOT_AVAILABLE_FOR_COMPLETION_PANEL**。
- `A0_2024_ROLE`: **KNOWN_RETROSPECTIVE_ONLY**。
- `A0_2025`: **LOCKED_NOT_READ**。
- `A0_SCIENTIFIC_FREEZE`: **NOT_READY**；delta、正式新 O、参照/分组与确认性方法对仍需 decision log。
