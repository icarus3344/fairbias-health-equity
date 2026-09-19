# 新窗口启动指令：NHIS隐藏亚组偏差研究

本文件可直接作为新窗口任务提示词；建议新窗口使用当前同一项目目录，避免在仅含旧Git提交的空白worktree里误以为已验收资产不存在。

## 可直接复制的完整指令

请在 /Users/lkc/Downloads/code_v_0_3 继续我的NHIS医学信息学应用研究。我的优先目标已经从“必须证明FairBias最好”调整为“发现容易被常规公平性评价遗漏的模型亚组差异，并严谨验证”。FairBias保留为比较方法与机制案例，不预设获胜，也不要把现有负结果抹去。

首先阅读：
1. docs/AI_EXECUTION_PROTOCOL.md
2. docs/plans/nhis_hidden_bias_research_20260918/RESEARCH_PLAN.md
3. docs/plans/nhis_hidden_bias_research_20260918/PROTOCOL_DRAFT.json
4. docs/plans/nhis_hidden_bias_research_20260918/planning_verification.json
5. docs/reports/FAIRBIAS_COMPLETION_FORMAL_EVALUATION_ACCEPTANCE_20260918.md
6. docs/reports/intersection_feasibility_20260918/REVIEW.md
7. docs/paper/manuscript_readiness_20260918/READINESS_REPORT.md
8. artifacts/nhis/completion_evaluation_20260918/control/supervisor_final_acceptance_v1.json

先核对当前分支、工作树及产物源绑定，保留所有既有改动，不reset、不覆盖原产物、不暂存提交、不上传个体数据。原14个继承文件与.gitignore保持不动；若存在既往基线差异如实记录，不偷偷修复。现有已验收结果是80个完成版FairBias模型+269个复用模型，不是80个仍然在训练，也不能混成原925模型分析。

按计划R0–R6执行，先完成R0–R2再进入真实亚组分析。你负责内部阶段验收，常规可逆工作不需要让我逐轮确认；不要只输出新一轮计划然后停止。若必要可以将独立的基础编码/文献整理交给子代理，主管必须亲自核查；控制token和并发，不为分工而分工。

第一批具体任务：
- 盘点2023/2024可复用预测、模型、记录键/顺序哈希、概率与决策语义，不能只凭行数连接；找出需要补做冻结推断的部分。
- 核对CDC 2025成人公开数据的官方元数据、跨年定义与访问历史。当前只确认官方页面有链接，尚未下载或分析2025微观数据，也未证明全团队从未使用过它。
- 冻结主模型、共同域、群体语法、支持度、指标与统计家族。主发现模型默认原Arm001无干预GBDT；原LR做敏感性；RW/TO做主要改善比较，FairBias BM_AE/Joint保留案例。
- 实现新的隔离审计入口及合成合同测试。旧模型/旧选模/旧阈值不变；p_event、q_decision、base_p严格区分。使用完整年度分层PSU设计，不用IID置信区间。
- 2023作探索性发现：4个预设性别×残疾组，加最多10个受限规则发现组；所有候选和选择过程留档。2024已知结果，只能作回顾性复核，不得重新切分伪造未见验证。
- 2025若符合条件，必须先冻结模型/规则/端点/源、主管核验并写release后才加载个体数据评价。不根据2025重选群体、阈值、模型、特征或检验族。若未能确立独立验证资格，明确降级为探索性，不虚构确认性结论。
- 首轮不重新训练整套benchmark、不改FairBias几何/NMI算法。联合属性新训练是可选第二步，须在2025开放前完成并另行冻结，否则只能作为后续探索。
- 交付可复现表图、研究问题与结果对应表、Methods/Results草稿、局限和导师汇报材料；没有可靠新差异时如实报告，不无限加搜索找显著。

所有服务器保持关闭，不连接旧SSH、不租卡、不启用旧自动任务。默认本地2进程、每进程BLAS线程1，实际计时后有理由才增至4。只有明确资源缺口时再给我具体服务器需求和预算。不要循环监控、待机或启动持续goal。

所有新实现、冻结、分析和报告放在独立新版本路径，候选建议为：
- src/nhis_fairbias/audit_hidden_bias/ （新模块，实施时再建立）
- scripts/run_nhis_hidden_bias_audit.py （阶段化入口，实施时再建立）
- tests/test_nhis_hidden_bias_*.py （边界与统计合同）
- artifacts/nhis/hidden_bias_research_<run_id>/ （内部微观预测/哈希/统计）
- docs/reports/nhis_hidden_bias_<run_id>/ （聚合报告与接续STATUS）
- docs/paper/nhis_hidden_bias_<run_id>/ （公开可用聚合图表与文稿）

PROTOCOL_DRAFT.json中的数值是待R1审阅冻结的提案，不是已预注册方案。不得只把status改成FROZEN就算冻结；必须填全模型ID、源/输入哈希、类别字典、成员连接、支持规则、统计定义和冻结时间。发现列表在R3结束后单独冻结。2025 release必须绑定这两层冻结及评价源码。

每个阶段完成后说明：做了什么、证据、剩余问题、下一步。遇到记录或版本不能验证先隔离相关项，保留历史失败；不要反复重训已验收模型。数据接口疑问先查文件/官方资料；临床用途、作者伦理等确需人类事实时集中提问，不自行编造。

## 新窗口容易误解的事项

- 当前任务是新研究路线，不是继续清理已经完成的训练队列。
- 2024“known_T=true”是明确历史，不能被新的文件名或数据切分清除。
- 2025网页可用只是机会，不证明变量一致、下载可用或无人看过。
- 原单属性公平方法不等于联合属性已训练；本轮先研究公平改善是否迁移。
- 亚组高结局率、亚组高FNR、因果歧视是不同命题。
- 小组NA不是自动算法失败；不能删除后宣称全组公平。
- 不要求FairBias排名靠前，也不将它的合成反例当作NHIS实证因果解释。
- 交付的新图表要标明估计对象、支持度、区间和探索/验证身份。

## 接续信息

规划日期2026-09-18；规划时HEAD为67e6659fa65249a8842e34af5d8969629efe4bca，实际工作树与源manifest更重要。本轮仅新增规划文件；没有运行新研究、下载2025微观数据或启动任何服务器。

