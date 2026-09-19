# 2026-09-16 扩展与资源登记（T 评价前）

Codex 根据用户完整推进授权，登记本轮执行设计。此前三个开发批次只使用 2022/2023；尚未启动新的 2024 模型评价。既有历史 2024 结果已见，论文须写回顾性跨年评价。

## 与主方案的关系

主比较仍为四臂、LR/GBDT 完整四点骨干网格、FairBias-BM 八点 ε 比例、五个随机种子（确定性 LR 对照只运行 seed 0）、统一 C 阈值和 S 选择。WTFA 训练敏感性保持独立，调查设计不进入 X。完整配置由 `experiment_registry.enumerate_candidates` 与 `experiment_extensions.enumerate_extensions` 生成、落盘并计算 SHA；不手写候选数代替真实矩阵。

近期比较固定为 FairGBM（2023）、FRAPPÉ（2024）、OxonFair（2024）、fairret（ICLR 2024）。每类匹配普通预测器与 FairBias。FairGBM 同源普通 LightGBM fork 使用 100/200 trees ×15/31 leaves、learning_rate .05；公平模型在同骨干上使用 FPR/FNR 同值 slack .005/.01/.02/.05/.1/.2，multiplier_learning_rate .1；slack 不是统一评价 EO 的直接上限，不能如此叙述。FRAPPÉ 仅二组任务，100 epochs、batch 32、Adagrad .01、MMD Gaussian length .1、KL系数1、紧凑隐藏层16；MinDiff权重 .01/.1/1/10。OxonFair二组grid_width6；七组grid_width2，见下文资源事件。fairret的普通MLP与公平正则MLP固定相同结构。

现代预测器补充 TabM：使用官方锁定架构源码，CPU紧凑配置 k4、2 blocks、width64，Adam .001、100 epochs、batch1024、每epoch按注册seed洗牌。取消无来源的200输入维上限，资源约束由运行器执行。此配置是有记录的应用适配，不称原论文全部训练设置复现。TabICL-v1 尚未完成官方checkpoint与全F上下文资源准入，明确不列为已运行对照；完整容量 TabM 也不列为本轮已完成项目。后续新增必须单独研究轮次，不能依据本轮T排名决定是否纳入。

## 消融固定配置

BM→AE 与 Joint 不执行 512 个大网格配置。它们用于解释阶段作用，固定 LR C1/GBDT100 depth2、ε_ratio1、每臂五seed、各模式各一配置；保留同点BM参考，明确不声称AE完成了全面调参。BM最多50提交、AE最多10提交、utility500次、geometry20000次；预算停止为未完成优化，不当作成功收敛。

Arm 004 使用同一人群、15特征，固定上述骨干，比较 stress-elbow、fixed2同绝对ε、fixed2自身F参考ε 三种几何。同ε条件从该seed、同F、stress-elbow参考表计算；自身ε条件改变几何也改变阈值，必须分开报告。检查实际transform/commit轨迹；结果差异本身不证明路径机制。

## 资源与开发事件

每个重型拟合串行，单进程峰值4 GiB、30分钟；BLAS/OpenMP线程1。表示只在F学习，可在同F/seed/几何设置的下游骨干与正则参数间复用；缓存绑定数据、源代码与参数身份，模型文件校验SHA。超时/超内存事件完整保留，并在相同表示任务中传播失败，不能重试无限搜索或把缺失seed当作最佳seed。

- `benchmark_codex_20260916_003433`：OxonFair保存lambda失败；已修cloudpickle并用新进程验证，旧失败记录保留。
- `benchmark_codex_20260916_004100`：FairBias数值 nullable Int64 转换失败；修正 NA 与非整数中位数的float处理，旧失败记录保留。
- `benchmark_codex_20260916_005500`：第一项完整FairBias约170秒通过；OxonFair七组grid6超过4 GiB。官方代码 `fair.py` 说明 O(grid_width**groups)，因此登记七组grid2资源配置，在新独立批次验证；不是根据S/T公平排名改变参数。

这些批次为开发准入证据，不与后续完整注册批次拼接为一套论文结果。新T评价必须等待全矩阵已尝试、所有seed状态与模型哈希已核验、完整选择与分析代码冻结。失败率、资源限制和未可估条件进入最终报告。

追加资源核验：`benchmark_codex_20260916_011300` 在grid2下仍出现相同七组内存错误。Codex追踪官方0.3源码发现 `learners/fair.py:408–417` 的 `call_fast(grid_width)` 实际忽略参数，向 `efficient_compute.grid_search` 传入硬编码默认steps。适配层现以限定到单次拟合、finally恢复的转接使原搜索真实收到登记grid。保存effective_grid_steps供核验，官方包源码保持原样。前一批二组实际也用了默认网格而非所写grid6，因此该批仍为开发事件，不进入最终配置比较。新批次使用修正后的传参与source identity。

## 最终训练规则与数值准入

应用主方法显式使用 `author_max_pair`：每个上下文分别取两侧组对 RMS 的最大值，再取两者绝对差，最后对上下文求均值。证据为作者独立仓库 `zftang/MachineClassifer_BiasMitigation_beta` 的锁定源码；本仓库冻结的 `eval.py` 使用均值写法，不能代替作者源码作为该规则的证据。此设定与 stress-elbow、H=1、保留 NMI 筛选等共同构成有名称的应用版本，不声称逐行复现作者固定二维脚本。

每一次 MDS 拟合记录 geometry call、维数、stress、实际迭代数、迭代上限和随机种子。非有限结果拒绝；达到迭代上限表示该次几何尚未验证收敛，不能冒充零偏差或成功收敛。初始、BM或最终状态评价中的这类异常传播为配置失败/预算耗尽；AE候选guard会记录 `GEOMETRY_EVAL_FAILED` 并拒绝该候选，允许继续检查其他候选，最终表示仍须通过严格几何评价。因此要保留被拒绝的数值尝试，不能把有效终末策略描述成每一次搜索都已收敛。不放宽数值容差来挑选胜出的配置。Shapley 距离构造将重复 pandas 子集计算替换为同顺序 NumPy 缓存；开发等价检查 240 例最大绝对差为零，中位矩阵构造加速约 3.7 倍，这不等于全算法加速。特征名必须唯一。

`benchmark_codex_20260916_012000` 已完成 64 项修正后的 LR/OxonFair 任务（含七组）、各一项 FairGBM、FRAPPÉ、fairret、TabM、EG-DP/EO、BM→AE、Joint 真实开发数据准入。BM→AE 约214秒、Joint约530秒。之后训练源码规则固定时原批次按源文件变化停止；它仍是开发证据，不与新批次拼接为论文结果。新完整注册批次为 `benchmark_codex_20260916_014500`。

## 推断与预测结果规则（T 前固定）

主要 BA 及配对 BA 差使用完整年度设计上的 Taylor PSU 线性化和设计自由度 t 区间。EO/DP 是非光滑极差，主区间采用覆盖所有已冻结种子模型 × 组率坐标的同时 Bonferroni-t 区间投影；先逐模型投影再对 seed 求均值，不把平均决策概率的 EO 当作平均 seed EO。20项主要对比家族分母保持20，即使部分方法失败；近期方法为探索性比较。2000次共同 rescaled-PSU bootstrap 的 EO 区间保留为描述性敏感性分析，不据此宣称精确覆盖。区间是固定已学习策略条件下、公开设计变量支持的近似调查推断；seed变异单列。

AP、AUROC、Brier 与校准图仅用 event-probability 输出；随机决策概率 q 不冒充风险。TO/OxonFair 原始预测器风险单列为 untouched base，不能宣称后处理提升了该风险预测。加权及未加权指标同时报告。次要风险指标主要报告点估计与seed变异，未实现设计区间者明确标注，不借用 BA 区间。FRAPPÉ 的冻结预测在已验证的隔离 TensorFlow 环境执行，读取所有模型前先验证冻结文件，绝不在 T 上拟合。
