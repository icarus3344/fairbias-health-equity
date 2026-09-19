# Gemini：R1A-R12 固定 R10 证据的最小交付收尾

状态：Codex 对 R11 判 REPAIR。只修新交付工具，不修改任何模型、runner、测试、配置，不跑 pytest、fit 或 benchmark。用户希望停止反复扩张修复范围；本提示明确取代前序“继续完善通用 validator、每个负例都生成整套报告”的实现路线。

先读 [协议](/Users/lkc/Downloads/code_v_0_3/docs/AI_EXECUTION_PROTOCOL.md)、AGENTS.md、GEMINI.md、[R11 监督报告](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R11_SUPERVISOR_REVIEW_20260915.md)。本轮唯一目标是：正确封存已经由 Codex 验证的同一个 R10 run。不是建立未来所有运行的通用验收器，不要求重做算法语义验证。

## 1. 固定信任来源

参考文件：[FROZEN_R10_REFERENCE.json](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R11_SUPERVISOR_REVIEW_20260915_evidence/FROZEN_R10_REFERENCE.json)。该文件 SHA-256 必须为：

`c3b16cbc7fafc8f14a77eec82384d221e554e188d566c74d1902a917b35cc84a`

它绑定 source_run_id=r1a_r10_20260915_105822Z_d582f745、source_phase=r1a-r10、22 份原始运行证据的 hash/bytes、120 项冻结输入。核对 R11 监督 MANIFEST 中的引用链；不得从待验证文件或当前可变目录重新计算并替换期望 hash。参考不匹配就停止依赖该输入的工作并准确报告。

仅新建 `docs/reports/fairbias_r1a_r12_worker_<UTC>_<id>/`。所有已有文件与历史目录只读，不修改 R11 已封存 validator，也不复制整个 R11 的 5233 文件重建大包。不要创建 /tmp 或 scratch 辅助文件；不清理旧文件、不写 sys.path/PYTHONPATH、不安装联网、不导入模型工程、不暂存/提交。所需纯 stdlib 辅助模块可用明确文件路径加载或独立 CLI 调用；不要用项目导入路径。

## 2. 用逐字节复用代替通用 trace 推断

实现一个小的 `check_reference_copy` 入口：

1. 验证可信参考文件固定 SHA；验证当前 120 项输入仍匹配及 source run 身份固定。
2. 在**实际将封存的 evidence 子目录**逐项核对 22 个文件的 hash、bytes 和完整名称集合，拒绝缺失、多余、冲突副本、目录穿越/外部链接。空 child_stderr.log 是合法的既定 0 字节文件，按参考判断，不能一律要求 bytes>0。
3. 不能验证原 RUN_DIR 后直接给另一份 REPORT_DIR 盖章。先构造新目录的准确副本，再核对它；已有冲突不得静默覆盖修好后冒称原输入通过。
4. 同样逻辑服务正常检查、负例诊断和最终封存，不另写较弱包装。来源变化、任一 trace 错误、NaN、丢来源条目都会改变既定指纹，被确定拒绝。
5. 明确限制：`FROZEN_R10_EVIDENCE_REUSE_ONLY`。支持的是这次已经独立验证的运行，不宣称对任意未来 run 完成通用日志/trace schema 验证。未来真实 benchmark 另有适配与验收规格。

允许 `--check-only` 调用同一实际 CLI/main 中的总检查，正常返回结构化成功，失败非零退出并给稳定错误码、文件名/检查项；不生成“FIXED_CANDIDATE”报告。它足以跑文件完整性的负例；不再要求每个完整性负例生成完整报告/历史矩阵。

## 3. 有限诊断，判定真正的目标错误

每次 attempt 排他创建，运行前保存工具源码、源码指纹、命令/cwd/必要环境覆盖、输入指纹和起始时刻。每个子过程的完整 stdout/stderr、结束时刻、退出码、输出指纹和状态落盘；中断保留 INCOMPLETE/UNKNOWN，不能事后伪造完成记录。

完整性检查的最小验收表：

- 正常 22 文件通过，120 输入不变。
- 22 个文件逐个缺失、逐个改动任意字节，各自被对应完整性检查拒绝；在人工副本中测试，原件不动。这覆盖 R11 的源码集合、七种 trace 与日志残余反例，不再逐值扩展通用 schema。
- 多余证据文件、冲突交付副本、错 source_run_id、错误参考 SHA 均拒绝。
- 单独执行 R11 的“输入正常、交付副本隔离 trace 错误”用例，必须拒绝实际封存目标。

只把明确的预期验证错误码及匹配检查项计为负例通过。NameError、ImportError、PermissionError、超时或语法错误标为基础设施失败，不计契约成功。验证这个诊断裁决规则时注入一个模拟 NameError 结果，不能把它算通过；不需要故意破坏真实工具。

不以固定“19/19”作为成功标准；总数从已冻结 case 注册表和实际执行记录生成。诊断不靠搜索输出中的 PASSED 字样。旧 R11 失败 attempt 已保存，保持其历史；本轮开始前的未知退出码不得补填。

## 4. 一次封存及诊断裁决

最终 seal 入口先检查：参考与实际 evidence 一致、当前工具与诊断测试工具的完整来源身份一致、已登记必需 case 都有完整记录且按预期通过、当前分支/HEAD/无暂存和 120 输入保持。诊断历史失败可以保留，但最后作为验收依据的完整 attempt 必须有效，不能只读取 all_cases_passed=true 自述。

必须实际验证最终 seal 拒绝：诊断结果缺失、false、必需 case 缺失、NameError 被伪标 PASSED、工具源码与被测版本不一致。用例只在新的人工包上做；任一失败都不产生成功 MANIFEST 或 FIXED_CANDIDATE。

最终目录唯一、所有材料完成后写一次 MANIFEST，递归覆盖本轮工具、参考引用、22 份证据、诊断记录和报告。不要把待验收包本身的清单当作上游可信参考。封存后生成器与诊断入口在任何输出写入前拒绝。

重复写入保护用新人工已封存目录验证两个实际入口，记录全目录前后 hash 相同，输出日志写在该人工目录之外的本轮 attempt 中；不必为了收集封存后日志再写入真实已封存根包。真实包最后 seal 一次后不再修改。

无需复制大量历史报告/源码快照，只通过不可变路径和 hash 引用；本轮工具和每次被测源码应有可读取的不可变版本。确需只读 git diff 时保存真实 returncode/stderr，不能忽略失败或声称工作树干净。

## 5. 报告与退出条件

报告明确：R12 交付、R10 模型执行来源；源运行 88/0/1、3.60 秒 pytest、6.11 秒 run、4 GiB 采样预算；没有 R12 训练或新 pytest。用固定来源/诊断记录生成事实，不把原 candidate_hashes.json 当成本轮工具清单。

R7-01/R7-04 只声明候选关闭，由 Codex 验收。R11 的 sys.path 调整、/tmp 辅助文件和 __pycache__ 清理按可见事实登记，不冒称模型绕行或删除研究结果。R7-02 首次编译 PARTIAL 保留；R7-03/05/06、R8-01 的已关闭模型范围不动；F01/F02 仍 OPEN/R1B，含义不变。

把 R11 仍有缺口的通用 validator 标记为历史未接受实现，不据本次逐字节复用宣称它已成为通用正确验证器。各项正常/拒绝/封存行为通过、无源输入变化、报告和清单一致即提交审查；不自行加模型、平台或额外安全工程。

按协议第 9 节完整 headings 交付。已授权按上述顺序完成，不需逐批询问。不得自批 ACCEPT、提交或启动真实数据。最后一行：

STOP — waiting for Codex review.
