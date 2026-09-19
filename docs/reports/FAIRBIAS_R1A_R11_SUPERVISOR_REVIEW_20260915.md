# FairBias R1A-R11 独立监督复核

2026-09-15。**结论：REPAIR，R1A 尚未接受。** 保留本轮有效进展，剩余仍归入原 R7-04、R7-01 两类 P1，不新增一串算法缺陷。没有确认 P0、新的模型训练绕行或真实微数据泄漏。本次没有模型 fit、pytest 重跑、生产代码修改、网络、安装、暂存或提交。

## 1. 已核实的进展

对象：[R11 worker 包](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r11_worker_20260915_125510Z_a1d0e4c5/WORKER_REPORT.md)，复用 R10 run。分支 research/nhis-fairbias，HEAD 67e6659fa65249a8842e34af5d8969629efe4bca，无暂存。HEAD 不代表当前整个工作树；存在前序未提交修改，不能称 clean working tree。

| 核查 | 结果 |
|---|---|
| 冻结输入 | R10 的 120 项原输入全部未变 |
| 历史证据 | R10 监督清单 1285 项、原 worker/run 447 项均未变化 |
| 当前交付 | 5233 项清单全部 hash/bytes 匹配；无未入清单文件 |
| 实际交付的 R10 材料 | 22 个文件与已验证 R10 来源完全一致 |
| 诊断保留 | 6 个独立 attempt，含前四次失败；最后两次汇总均为 19/19 |
| 最终工具身份 | 最后一次诊断的 generator/validator hash 与当前一致 |
| 问题矩阵 | 48 项；AST 位置全部匹配 |
| 本次独立诊断 | 实际 generator main＋实际 stdlib validator，24 个用例；退出 0、意外权限拒绝 0、模型 fit 0 |

独立诊断中：正常包可生成；重复生成器和诊断器均抛 FileExistsError，人工已封存包不变；10 个分开的既有负例被拒绝，包括七类 trace 的上一轮指定错误值、错误源码 hash、错误来源计数、非法 JSON 日志。本次没有重跑全部 worker 19 用例或 88 项模型测试，已核验最后一次 worker 19 用例的保存结果与源码身份。

因此，独占 attempt、完整子进程 stdout/stderr、递归清单及重复写入保护均是实质进展。下面的缺口不表示实际 R10 材料已损坏，也不撤销既有模型契约的关闭结论。

## 2. P1 / R7-04：检查的文件与封存的文件不是同一份

[生成器 109 行](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r11_worker_20260915_125510Z_a1d0e4c5/generate_r1a_r11_delivery.py:109)验证 RUN_DIR；[124 行](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r11_worker_20260915_125510Z_a1d0e4c5/generate_r1a_r11_delivery.py:124)只在目标 trace 不存在时复制；[459 行](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r11_worker_20260915_125510Z_a1d0e4c5/generate_r1a_r11_delivery.py:459)最终清单遍历 REPORT_DIR。没有把目标包中的每份证据绑定到实际通过校验的内容。

独立反例 `packaged_trace_differs_from_validated_input`：保持输入完全正常，仅把交付副本 isolation_trace 的 fit_all_from_f 改 false、fit_source 改 T_test。实际 main 仍退出成功、写 MANIFEST、报告 FIXED_CANDIDATE、overall_verification_passed=true，封存了错误副本。这是必须先修的核心问题。

来源/trace 校验也仍没有落实全部既定关系。下列 **9 个分别执行的** 反例均被当前实际 main 接受，生成成功包：

| 反例 | 未落实的既定要求 |
|---|---|
| pre/post/loaded/compiled 全部只保留 enhancement.py，loaded/compiled 数量改 1，仍宣称 88 测试通过 | [164–270 行](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r11_worker_20260915_125510Z_a1d0e4c5/validator_r1a_r11.py:164)只检查所提供集合，不检查必需来源/实际执行测试集合 |
| 人工流 mono_step2_powers 清空，保持“cursor maintained=true” | [589 行](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r11_worker_20260915_125510Z_a1d0e4c5/validator_r1a_r11.py:589)未核对实际序列 |
| 正式 stream_length 改为 1 | [606 行](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r11_worker_20260915_125510Z_a1d0e4c5/validator_r1a_r11.py:606)只要求正整数，未绑定正式 1998 流 |
| cache 的 cand_fits_return 改 999，其余 reuse=true | [617 行](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r11_worker_20260915_125510Z_a1d0e4c5/validator_r1a_r11.py:617)未核对实际拟合次数关系 |
| entrypoint 改 Other.compute_metrics，keys_present 改为 28 个相同的 wrong | [631 行](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r11_worker_20260915_125510Z_a1d0e4c5/validator_r1a_r11.py:631)只拒绝特定字面量 invalid 和短列表 |
| 内部拟合摘要 all_model_features_y_verified=false，其余外层标记不变 | [640 行](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r11_worker_20260915_125510Z_a1d0e4c5/validator_r1a_r11.py:640)未校验内部摘要关系 |
| joint 事件改成仅含 cycle_detected=true 的一个字典 | [663 行](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r11_worker_20260915_125510Z_a1d0e4c5/validator_r1a_r11.py:663)未核对事件内容/路径 |
| candidate_boundary.equal_dphi_accepted 改 true | [674 行](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r11_worker_20260915_125510Z_a1d0e4c5/validator_r1a_r11.py:674)未校验严格候选边界 |
| sampled_max_rss_mb 改 JSON NaN | [324 行](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r11_worker_20260915_125510Z_a1d0e4c5/validator_r1a_r11.py:324)未检查有限性，NaN 绕过上下界比较 |

这些不是要求增加模型功能，而是 R11 原要求中的来源、七类 trace 关系和严格类型。上一轮合并变异被首个检查拒绝，并不能证明其余字段已被验证。

## 3. P1 / R7-01：诊断失败不阻止整体通过，运行记录仍有缺项

[生成器 167–180 行](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r11_worker_20260915_125510Z_a1d0e4c5/generate_r1a_r11_delivery.py:167)读取 diagnostic_results 的字段，但不验证其每个 case、源码绑定或通过条件，也不把失败纳入最终裁决。[413 行](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r11_worker_20260915_125510Z_a1d0e4c5/generate_r1a_r11_delivery.py:413)继续写成功状态。

独立反例 `diagnostics_explicitly_failed` 仅将 all_cases_passed 改 false。实际输出同时出现 **diagnostic_all_passed=false 与 overall_verification_passed=true**，仍有成功 MANIFEST。当前真实最后一次诊断是 19/19；反例说明裁决路径有缺口，不表示本轮实际诊断全失败。

[诊断器 319 行](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r11_worker_20260915_125510Z_a1d0e4c5/diagnose_r1a_r11_delivery.py:319)把任何“非零退出且没有 MANIFEST”算作负例通过。保留下来的 [第二次 attempt 错误日志](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r11_worker_20260915_125510Z_a1d0e4c5/diagnostics/attempt_20260915_131216Z_d16d189a/case_unique_missing_required_node/diagnostic_stderr.log)显示 unique_missing_required_node 实际报 NameError: hashlib 未定义，却被记为 PASSED。最后一次对应 case 已不再报此错，但诊断器的通用判定规则未修；基础设施错误不能证明目标契约生效。

[268–297 行](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r11_worker_20260915_125510Z_a1d0e4c5/diagnose_r1a_r11_delivery.py:268)计算了输入/输出指纹、命令和起止时间；[326 行](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r11_worker_20260915_125510Z_a1d0e4c5/diagnose_r1a_r11_delivery.py:326)保存的 case 只有时间差、退出码、文件数量等，没有把完整 command/cwd/env、起止时刻及指纹映射写入记录。实际输入文件和完整输出都保留了，不能再说“诊断没有保存”；但最终文件清单不是执行前后状态的记录。

## 4. P2：范围与报告准确性

附件有 47 条实际命令，报告仅概述了少数调用。出现了多次 sys.path.insert、在 /tmp 写入并删除 expected_test_nodes.py，以及清理本轮 __pycache__。这些与 R11 的明确范围不符，需要准确登记；操作对象是纯 stdlib 交付工具/临时源码，不把它们夸大为模型执行绕行、删除研究结果或 P0。

报告中的 clean working tree 不准确；R11 的 120 项输入未变不等于整个工作树干净。“首次编译 PARTIAL”“复用 R10”“没有新模型训练”应继续保留。采用 -I -S -B 不等于完整沙箱。文案需引用本轮事实，但不为排版或附加功能再扩张 gate。

## 5. 用更小的方案结束本关卡

本轮三个工具合计约 82 KB，已有 6 次诊断、5233 项清单。材料增加并未自动解决“验证哪一份、失败能否最终化”这两个根因。继续逐字段补通用 validator 不是本次最合适的收尾方式。

监督已生成 [FROZEN_R10_REFERENCE.json](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R11_SUPERVISOR_REVIEW_20260915_evidence/FROZEN_R10_REFERENCE.json)：固定 R10 的 22 份材料及 120 项源码/配置等输入，且逐项对照前序已封存指纹。本引用不新增训练、不重解释旧测试，不由 worker 重新计算“期望值”。

下一轮改成**仅接受这次已验证 R10 的逐字节复用**：直接核对实际要封存的证据目录；来源错误或任一字节不同都拒绝；诊断结果以真实规则和已测试工具身份决定封存。R11 的通用 validator 保留历史，不继续把它扩展成未来所有实验的验证平台。这是利用 R11 已允许的冻结证据引用来收敛实现，并明确限制验收声明。

参见 [R12 最小收尾提示](/Users/lkc/Downloads/code_v_0_3/docs/plans/GEMINI_FAIRBIAS_R1A_R12_FROZEN_DELIVERY_PROMPT_20260915.md)。这次仍不授权 R1B 实验；近期 benchmark 候选规划保持不变，R1A 接受后先完成 F01/F02 与适配注册，再进入调查推断和冻结比较。

## 6. 证据与边界

[静态核验](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R11_SUPERVISOR_REVIEW_20260915_evidence/verification.json)、[独立结果](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R11_SUPERVISOR_REVIEW_20260915_evidence/probe_run_134807_136263/results.json)。24 个用例由当前实际 generator main 和 validator 执行，代码从原字节编译；只替换了不影响校验的只读 git diff 子进程为明确记录的占位输出。诊断器只运行已封存包的拒绝分支，没有重跑它的完整 suite。

永久 Python audit hook 在三个交付模块执行前安装：只允许已列来源和新人工目录读写，禁止模型库导入、网络和子进程。全部故障只注入监督新目录的人工副本；不是在真实包上做覆盖测试，也不是 OS 全能力沙箱。

本次确认 11 个错误材料用例仍获成功包；10 个既有负例拒绝，加上正常与两项重复写入保护共 24 个用例。R7-03、R7-05、R7-06、R8-01 维持原关闭范围；R7-02 指定来源/路径关闭与首次编译 PARTIAL 边界不变。当前实际来源有效与交付工具尚未验收可以同时成立。
