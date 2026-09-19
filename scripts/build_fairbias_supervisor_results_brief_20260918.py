#!/usr/bin/env python3
"""Generate a results-first supervisor briefing from sealed aggregate exports."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import zipfile
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/paper/supervisor_results_brief_20260918'
CURRENT = ROOT / 'docs/paper/completion_results_20260918'
OLD = ROOT / 'artifacts/nhis/paper_export_v2_20260917/report'
READY = ROOT / 'docs/paper/manuscript_readiness_20260918'
ARMS = {'arm_001':'性别', 'arm_002':'种族/族裔（7组）', 'arm_003':'残疾', 'arm_004':'残疾·去除6项功能特征'}
NAMES = {'FAIRBIAS_BM_AE':'BM+AE', 'FAIRBIAS_JOINT':'Joint', 'FAIRBIAS_BM':'BM', 'UNMITIGATED':'无减偏', 'REWEIGHING':'Reweighing', 'LFR_RECONSTRUCTED':'LFR重建版', 'EG_DP':'EG-DP', 'EG_EO':'EG-EO', 'TO_EO':'TO-EO', 'FRAPPE_EO':'FRAPPÉ-EO', 'OXONFAIR_EO':'OxonFair-EO','FAIRGBM_EO':'FairGBM-EO','FAIRRET_EO':'fairret-EO'}
ROLES = {'parent_tau10':'原冻结选择', 'parent_fixed_BM':'固定ε=1 BM', 'adaptive_completed80':'本次完整计算'}
INPUTS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def js(path):
    return json.loads(Path(path).read_text())


def verified(path, expected=None):
    got = sha(path)
    if expected is not None:
        assert got == expected, path
    INPUTS[str(path.relative_to(ROOT))] = got
    return path


def readcsv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def num(x, digits=4, scale=1):
    if x is None or str(x).strip() in ('', 'NA','N/A','None'):
        return 'NA'
    return f'{float(x)*scale:.{digits}f}'


def interval(r, metric, kind='family', scale=100):
    return '['+num(r[f'delta_{metric}_ci_{kind}_low'],2,scale)+', '+num(r[f'delta_{metric}_ci_{kind}_high'],2,scale)+']'


def table(headers, values):
    def cell(v):
        return str(v).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(cell(v) for v in row)+' |\n' for row in values)+'\n'


def txt(name, content):
    (OUT/name).write_text(content, encoding='utf-8')


def main():
    assert not OUT.exists(), 'Refuse to overwrite an existing briefing'
    meta = js(verified(CURRENT/'build_metadata.json'))
    assert sha(Path(meta['input'])) == meta['input_sha256'] == '641fe2ec523c8998cfe4c1f48ae2d5fb1e6ecbe9a5eb9e20b16af33d47c6b406'
    for name in ('completion_conditions_96.csv','completion_pairs_168.csv','completion_fairbias_8_conditions.csv'):
        verified(CURRENT/name, meta['output_sha256'][name])
    old = js(verified(OLD/'report_manifest.json', 'e7960ffcdb6d1117532de747913c117c9af4ea28e92df907e0e3bbfbd99af924'))
    for name,digest in old['output_files'].items():
        verified(OLD/name,digest)
    cohort_root = ROOT/'docs/paper/cohort_tables_v2_20260918'
    cohort_manifest = js(verified(cohort_root/'hash_manifest.json'))
    verified(cohort_root/'Table1.csv',cohort_manifest['outputs']['Table1.csv'])
    for folder in (READY/'missingness/v2', READY/'missingness/figures_v3'):
        manifest = js(verified(folder/'hash_manifest.json'))
        for name,digest in manifest['outputs'].items():
            verified(folder/name,digest)
    tf = READY/'transformations/extraction'
    for name,digest in js(verified(tf/'output_hashes.json')).items():
        verified(tf/name,digest)
    for path in [READY/'READINESS_REPORT.md', READY/'MANUSCRIPT_METHODS_RESULTS_v2.md', ROOT/'docs/paper/FAIRBIAS_COMPLETION_COST_AND_INTERPRETABILITY_20260918.md', ROOT/'docs/reports/FAIRBIAS_COMPLETION_FORMAL_EVALUATION_ACCEPTANCE_20260918.md']:
        verified(path)

    cond = readcsv(CURRENT/'completion_conditions_96.csv')
    pairs = readcsv(CURRENT/'completion_pairs_168.csv')
    cohorts = readcsv(cohort_root/'Table1.csv')
    original = readcsv(OLD/'paper_results_tau_010.csv')
    jobs = readcsv(OLD/'registered_job_status_counts.csv')
    assert len(cond)==96 and len(pairs)==168 and len(cohorts)==68 and len(original)==138
    assert sum(r['T_evaluation_status']=='VALID' for r in cond)==90
    assert sum(r['evaluation_status']=='VALID' for r in pairs)==156
    mainrows = [r for r in cond if r['origin']=='adaptive_completed80']
    assert len(mainrows)==16 and all(r['seed_count']=='5' for r in mainrows)
    lookup={(r['arm_id'],r['backbone'],r['method'],r['origin']):r for r in cond}
    def get(a,b,m):
        return lookup[(a,b,m,'adaptive_completed80' if m in ('FAIRBIAS_BM_AE','FAIRBIAS_JOINT') else 'parent_tau10')]
    unpaired=[r for r in pairs if r['comparison_method']=='UNMITIGATED']
    primary=[r for r in pairs if r['family']=='primary']
    joint=[r for r in pairs if r['kind']=='joint_vs_sequential']
    assert (len(unpaired),len(primary),len(joint))==(16,10,8)
    assert sum(float(r['delta_BA'])<0 for r in unpaired)==14
    assert sum(float(r['delta_EO'])>0 for r in unpaired)==16
    assert sum(float(r['delta_AUC'])<0 for r in unpaired)==14
    assert sum(float(r['delta_Brier'])>0 for r in unpaired)==12
    assert len({r['model_id'] for r in old['selected_models']})==925
    OUT.mkdir(parents=True)
    (OUT/'tables').mkdir(); (OUT/'figures').mkdir()
    copied = {}
    sources = {
        'current_conditions_96.csv': CURRENT/'completion_conditions_96.csv',
        'current_pairs_168.csv': CURRENT/'completion_pairs_168.csv',
        'current_main_8_cells.csv': CURRENT/'completion_fairbias_8_conditions.csv',
        'cohort_all_68.csv': cohort_root/'Table1.csv',
        'missingness_all_336.csv': READY/'missingness/v2/missingness_by_variable.csv',
        'eligibility_all_16.csv': READY/'missingness/v2/eligibility_exclusions.csv',
        'transformations_all_1680.csv': tf/'model_variable_matrix.csv',
        'transformations_by_variable_42.csv': tf/'variable_mode_summary.csv',
        'transformations_committed_1425.csv': tf/'committed_events.csv',
        'old_results_tau010_138.csv': OLD/'paper_results_tau_010.csv',
        'old_results_auxiliary_334.csv': OLD/'paper_results_auxiliary.csv',
        'old_paired_contrasts_300.csv': OLD/'paper_paired_contrasts.csv',
        'old_registered_status.csv': OLD/'registered_job_status_counts.csv',
        'old_attempt_status.csv': OLD/'attempt_status_counts.csv',
        'old_failures_259.csv': OLD/'paper_failures.csv',
        'old_not_supported_4.csv': OLD/'not_supported_conditions.csv',
    }
    for name,path in sources.items():
        shutil.copyfile(path, OUT/'tables'/name)
        assert sha(path)==sha(OUT/'tables'/name)
        copied[name]={'source':str(path.relative_to(ROOT)),'sha256':sha(path),'rows':len(readcsv(path))}
    for name,path in {
        'prediction_fairness.png':CURRENT/'prediction_fairness_scatter.png',
        'joint_vs_bmae.png':CURRENT/'joint_minus_bmae_paired_ci.png',
        'missingness_semantics.png':READY/'missingness/figures_v3/nhis_missingness_semantics_arm001.png',
        'search_operations.png':ROOT/'docs/paper/figures_completion_v2_20260918/feature_trace_feasibility.png',
        'compute_cost.png':ROOT/'docs/paper/figures_completion_v2_20260918/completion_cost_coverage.png',
    }.items():
        verified(path); shutil.copyfile(path,OUT/'figures'/name)

    wechat='''老师您好，跟您汇报一下FairBias在NHIS上的实验进展：之前受计算预算影响的BM+AE和Joint已经补跑完成，本批80个主方法模型都完成了训练及2024年跨年评价，连同冻结的对照结果，共比较了349个模型。

目前结果并没有显示FairBias普遍优于其他方法。相对同一骨干的无减偏基线，16组主方法配置中有14组的平衡准确率更低，16组的EO差距点估计都更大；这些是点估计方向，不代表每一项差异都显著。

Joint和顺序BM+AE在8个“人群属性×骨干”条件中的6个给出了完全相同的测试预测。差别较明显的是去掉6项功能特征的残疾分析：LR下Joint的平衡准确率提高约1.61个百分点，但EO差距也增大约5.00个百分点，校正后的差值区间都包含0。目前不能说交错搜索带来了稳定优势。

另外，调查加权会明显影响公平性判断；变换审计也发现了把缺失/不适用状态与实际类别合并的情况。因此，当前比较明确的发现是：内部几何约束达标，并不一定对应更小的输出群体差距；变换过程虽然可以追踪，临床语义是否保留还需要单独检查。

我已经把四个分析臂、全部对照、置信区间、较新方法的补充结果及计算成本整理成详细汇报。需要说明，这次计算扩展是在历史2024结果已知后完成的，按回顾性评价报告。想先向您汇报完整结果，再请您指导后续的研究定位。
'''
    txt('01_微信汇报.md',wechat)
    doc='''# FairBias / NHIS 实验结果汇报大纲

更新：2026-09-18。用途：向导师先汇报实验事实与结果，再决定论文定位。本文件不新增训练、选模或显著性检验。所有小数直接转录已验收聚合表；完整精度、种子SD、未加权结果和缺失状态保存在随包CSV。

## 一、开场：先讲当前状态和结论（约1分钟）

“本批完整FairBias主方法已经跑完，也完成了跨年预测、公平性评价和调查统计。现在最需要讨论的是如何解释结果。当前证据不支持FairBias普遍优于对照；内部几何可行和实际群体公平性达标并不是同一件事。”

- 本次：BM+AE40/40、Joint40/40；4臂×2骨干×2执行模式×5种子=80模型。
- 与269个冻结对照预测合并，当前比较覆盖349个唯一模型；96条件中90可评价，168配对中156可评价。
- 剩余6条件/12配对不完整均来自种族/族裔臂的历史对照或不支持配置。本批新80模型没有未完成或超时NA。
- 旧固定预算研究独立保留925个选定模型、138条tau=.10结果和334条辅助结果、300个配对；与349不能直接相加，因为存在复用。
- 当前主方法每格是固定ε比例1候选，已经完成该候选的搜索，不代表整个Joint参数空间已穷尽。

## 二、研究在预测什么、比较什么（约2分钟）

结局为NHIS Sample Adult的MEDDL12M_A：过去12个月是否因费用推迟医疗。它是医疗可及性识别，不是预测同一患者未来是否患病。2022、2023、2024为不同年度的重复横断面样本。

| 部分 | 用途 | 不能混淆的地方 |
|---|---|---|
| 2022 F | 拟合表示、缺失处理、编码和预测模型 | 只有F承担这些拟合 |
| 2022 C | 决策阈值和算法内部效用评价 | 参与AE发展，不是独立最终测试；与F按整PSU划分 |
| 2023 S | 固定新候选的公平性筛查；历史对照保留原冻结选择 | tau=.10约束EO；另记录.05/.20规则，不是T保证 |
| 2024 T | 冻结政策统一评价 | 扩展前历史T已知，不能称全新盲测或前瞻验证 |

| 分析臂 | 保护属性 | 输入特征 | 意义 |
|---|---|---|---|
| Arm1 | 性别，2组 | 21项 | 比较性别群体差距 |
| Arm2 | 种族/族裔，7组 | 21项 | 检验多组行为与稀疏亚组支持 |
| Arm3 | 残疾状态，2组 | 21项 | 保留6项功能限制相关特征 |
| Arm4 | 与Arm3相同的残疾状态及人群 | 15项 | 排除6项功能特征的消融；不是只取残疾人，也不是第4个独立队列 |

骨干是逻辑回归LR和梯度提升树GBDT。BM+AE先完成几何减偏再增强效用；Joint交错尝试两类操作，按本地冻结停止规则结束。两者均不保证全局最优；内部几何阈值也不是EO输出阈值。

主比较包含无减偏、BM、Reweighing、LFR重建版、EG-DP、EG-EO、TO-EO、FRAPPÉ-EO和OxonFair-EO，并保留固定ε=1的BM模块对照。部分确定性LR对照只有原登记的一个种子，不能写成每种方法都做了五个独立重复。FRAPPÉ/LFR在本地七组实现不支持，不能推断整个理论框架永远不能多组。

## 三、人群、事件与指标口径（约2分钟）

下表是全部16个分析域。加权事件率的分母是该分析域；F/C不能当两个全国独立样本相加解释。全部保护组的人数/事件见附录C及cohort_all_68.csv。

'''
    doc+=table(['分析臂','分区','年份','人数n','事件数','加权事件率%'],[(ARMS[r['arm_id']],r['partition'],r['year'],r['n'],r['event_count'],num(r['weighted_event_rate'],2,100)) for r in cohorts if r['group_code']=='ALL'])
    doc+='''2024完整年度32,629人，52个分层、662个PSU，自由度610。分析采用完整年度调查设计上的域估计；WTFA_A、PSTRAT、PPSU用于评价/推断，不作为预测特征。主训练策略未将调查权重嵌入FairBias几何。

种族/族裔中的非西班牙裔美洲印第安人/阿拉斯加原住民组，C有37人、1个事件；T有266人、17个事件。多组公平性差距可能由小组的率估计决定，须结合不确定性，不能把小组不稳定直接归因于某个算法。

| 指标 | 解释 | 方向与限制 |
|---|---|---|
| BA | 敏感度与特异度平均 | 越大越好；不是普通准确率；0.5常对应无区分性决策 |
| EO gap | 群体TPR范围与FPR范围的较大者 | 越小越好；0.20对应20个百分点的群体率差，不是20%的人被歧视 |
| DP gap | 群体正决策率的范围 | 越小越好；不条件于真实结局，与EO含义不同 |
| AP | Average Precision | 越大越好；不要不加说明地叫梯形积分PR-AUC |
| AUROC | 风险分数排序区分能力 | 越大越好；不能据此认为概率校准良好 |
| Brier | 概率预测均方误差 | 越小越好；单一Brier不能代替完整校准评价 |

p是事件风险输出，q是正决策概率。BA/EO/DP使用q；风险指标只在p有定义时报告。EG/TO等q-only单元的AP/AUC/Brier为NA，不是模型训练失败；未调整基础预测器的p不能替代公平决策策略的风险。

## 四、主方法和无减偏基线的全部8个场景（约3分钟）

每个单元为**BA / EO gap**，均为2024调查加权点估计。BM+AE/Joint均为5种子均值；表中“满足”指2023 S的EO≤.10。

'''
    view=[]
    for a in ARMS:
        for b in ('LR','GBDT'):
            u,x,y=get(a,b,'UNMITIGATED'),get(a,b,'FAIRBIAS_BM_AE'),get(a,b,'FAIRBIAS_JOINT')
            view.append([ARMS[a],b,*[num(t['T_BA'])+' / '+num(t['T_EO']) for t in (u,x,y)],'两者满足' if x['S_status']==y['S_status']=='FEASIBLE' else '两者不满足'])
    doc+=table(['分析臂','骨干','无减偏','BM+AE','Joint','S约束'],view)
    doc+='''应汇报：只有性别臂4/16个新配置在S满足EO≤.10，其余12个是边界评价。内部几何停止规则都满足，不能写成“所有模型都达到了公平性要求”。性别臂绝对EO较小也不意味着比无减偏改善。

## 五、完整预测能力：16个主方法配置

BA/EO已见上表；下面补齐DP、AP、AUROC、Brier。所有概率风险差值目前仅作描述，冻结分析未提供相应正式配对区间。

'''
    doc+=table(['分析臂','骨干','方法','DP↓','AP↑','AUROC↑','Brier↓'],[(ARMS[r['arm_id']],r['backbone'],NAMES[r['method']],*[num(r['T_'+m],5 if m=='Brier' else 4) for m in ('DP','AP','AUC','Brier')]) for r in sorted(mainrows,key=lambda r:(r['arm_id'],r['backbone'],r['method']))])
    doc+='''相对无减偏，14/16组合BA更低，16/16组合EO更大，14/16组合AUROC更低，12/16组合Brier更高。这些方向计数不等于逐项统计显著，也不表明算法在所有任务都失效。反例也要保留：Arm2 LR Joint的AP=.2643、AUROC=.7704、Brier=.07133，较无减偏变化+.01733、+.01783、−.00097，但EO增加约.04158。

下表进一步给出全部16个主方法−无减偏配对。差值和区间单位是**百分点**；BA正为预测更好，EO正为群体差距更大。区间为次要家族316端点校正结果。

'''
    doc+=table(['分析臂','骨干','方法','ΔBA','ΔBA区间','ΔEO','ΔEO区间'],[(ARMS[r['arm_id']],r['backbone'],NAMES[r['reference_method']],num(r['delta_BA'],2,100),interval(r,'BA'),num(r['delta_EO'],2,100),interval(r,'EO')) for r in unpaired])
    doc+='''## 六、Joint是否比顺序BM+AE更有价值？

8个场景中6个的五种子测试风险p和决策q数组均完全相同。只有Arm2 LR和Arm4 LR不同。这是这批数据与配置上的输出一致，不是证明两种算法普遍等价。

以下为**Joint−BM+AE**，百分点，次要家族校正区间：

'''
    doc+=table(['分析臂','骨干','ΔBA','ΔBA区间','ΔEO','ΔEO区间'],[(ARMS[r['arm_id']],r['backbone'],num(r['delta_BA'],2,100),interval(r,'BA'),num(r['delta_EO'],2,100),interval(r,'EO')) for r in joint])
    doc+='''Arm2 LR的BA约提高0.25个百分点，但校正区间包含0；Arm4 LR提高1.61个百分点，同时EO增大5.00个百分点，两项区间均包含0。可说“观察到不同取舍”，不能说Joint有稳定显著优势。

为什么部分相同预测仍有宽EO差值区间？目前EO正式区间由分别覆盖的群体率/差距界投影再相减，保守界未充分利用两套相同决策的相关性；因此它不是直接对完全相同数组的差值做出的精确置信区间。BA成对线性化则能给出相同决策的[0,0]。应将已验证的数组一致性与保守投影界分别报告，不把宽界当成真实运行差异。

## 七、与其他公平性方法相比：主要比较全部列出

主要家族预先固定为Arm1/Arm3、LR、Joint对Reweighing/LFR/EG-DP/EG-EO/TO-EO，共10对、20个BA/EO端点。这里“预先”是相对新预测而言；历史T已知，不是前瞻外部预注册。

差值均为**Joint−对照**，百分点。下表保留主要家族校正与合并336端点校正，不能只选有利的一种：

'''
    doc+=table(['分析臂','对照','ΔBA','BA家族区间','BA合并区间','ΔEO','EO家族区间','EO合并区间'],[(ARMS[r['arm_id']],NAMES[r['comparison_method']],num(r['delta_BA'],2,100),interval(r,'BA'),interval(r,'BA','union'),num(r['delta_EO'],2,100),interval(r,'EO'),interval(r,'EO','union')) for r in primary])
    doc+='''主要家族：BA有4个区间完全为正、3个完全为负、3个包含0；4个正向比较都来自EG。EO有2个区间为正（Joint更差），8个包含0，没有明确负向改善。合并校正后，BA为4正、1负、5跨0；EO为1正、9跨0。

因此不能把“战胜EG的BA”扩展为“优于全部公平方法”。应结合EG的随机化决策、优化目标与BA代价解释其低BA；同时展示其他方法的权衡。全96个条件的绝对值在附录A，全部168对的差值/三种区间在附录B。

## 八、近年方法已经产生了哪些结果？

下面32行来自**旧固定预算研究的扩展**，tau=.10、非调查加权训练。FairGBM使用其匹配原生GBDT，fairret使用MLP，TabM为紧凑配置。它们与新80不属于同一完成分支，不能将跨骨干最高数值称为严格同骨干胜负，也未在本次新主要家族中检验。

'''
    modern=[r for r in original if r['backbone'] in ('FAIRGBM_BASE','MLP','TABM') and r['training_weighted']=='False']
    assert len(modern)==32
    doc+=table(['分析臂','骨干','方法','BA','EO','DP','AP','AUROC','Brier','S状态'],[(ARMS[r['arm_id']],r['backbone'],NAMES.get(r['method'],r['method']),*[num(r[k],5 if k=='brier_p' else 4) for k in ('balanced_accuracy','eo_gap','dp_gap','average_precision_p','auroc_p','brier_p')],r['S_status']) for r in modern])
    doc+='''可直接说：较新方法确实已跑出结果。例如FairGBM在残疾臂的BA/EO点估计为.6883/.0322，而种族/族裔臂约为.5066/.1141，后者反映出明显预测代价且S不可行；不能只展示好看的残疾一格。旧研究的全部138条主工作点、334条辅助工作点和300对比较均随包附上，S不满足和缺模型的行没有删去。

## 九、调查加权、误差范围和结果稳健性

下面全部16格同时列加权与未加权BA/EO；二者目标分布不同，不选择更漂亮的一列。

'''
    doc+=table(['分析臂','骨干','方法','加权BA','未加权BA','加权EO','未加权EO'],[(ARMS[r['arm_id']],r['backbone'],NAMES[r['method']],num(r['T_BA']),num(r['unweighted_BA_mean']),num(r['T_EO']),num(r['unweighted_EO_mean'])) for r in mainrows])
    doc+='''最直观的例子是Arm2 GBDT：未加权EO约.3495，加权后.4680；不能把调查样本简单当便利样本。其他臂加权也可能让gap变小，方向并不固定。

推断口径：BA采用调查设计Taylor线性化，EO/DP采用同时群体率区间的保守投影；共用PSU重抽样2000次作为描述敏感性。正式区间条件于已经冻结的政策，不覆盖重新训练与选模的全部不确定性。BA/EO主要、次要、合并家族端点数为20/316/336，包含缺失槽位。

五种子新模型BA/EO基本一致并不等于总体误差为0：它们不是五次独立抽样。跨环境复核中C决策80/80一致，但风险概率字节哈希不同，差值大小未量化；不能宣称跨平台概率完全相同。

## 十、可解释性：已经有证据的结果是什么？

这里报告所有80个最终表示，并非挑选漂亮案例，也不将BM中间历史冒充AE后的最终状态。

| 审计对象 | 数量/发现 | 可以说明什么 |
|---|---|---|
| BM提交 | 560次：435合并、65幂变换、60删除 | 减偏阶段实际操作记录 |
| AE候选记录 | 37,515条：865提交、15,348有效但未提交、21,107几何约束拒绝、195循环拒绝 | 搜索代价与约束行为；几何拒绝不是实际EO失败 |
| 最终变量—模型对 | 820合并、80幂变换、60删除、600未变；120 Arm4设计排除 | 总计1,680行；操作次数不能与最终状态数相加 |
| 缺失与实际类别合并 | 80/80模型 | 表示可以追踪，但不等于缺失被合理插补 |
| 就业NIU与全职类别合并 | 60/80模型 | 输出类别应标明两类集合，不能只叫全职 |
| 非连续序数合并 | 70/80模型 | 未保证收入/健康严重度层次的语义连续 |
| 6项功能变量 | 360个可用变量—模型对：220合并、140未变、0学习删除 | Arm4的120个排除来自设计，不是算法自主去掉残疾信息 |

所有560+865个操作均独立重放至最终状态，80/80匹配。这证明记录和表示可复核，不证明临床等价、因果机制或已消除残疾代理信息。完整42行变量汇总和1680行最终映射随包提供。

缺失方面也有容易误读的结果：Arm1 T就业强度的结构NIU加权35.30%、条目无应答3.08%、源空值0.007%。合并38.39%并不是条目缺失率；未加权合并为44.31%。发布收入类别完整但含官方单次插补，未传播多重插补不确定性。见附录D的全部336行。

## 十一、为什么之前慢、现在到底完成到哪里？

本次完成80模型的累计进程运行时长为92.75小时。因为并行运行，这不是实际租用时长、CPU计费小时或从开始到结束的自然时间；也没有计入历史失败版本。

| 方法/骨干 | 完成模型 | 平均每进程分钟 | 中位数分钟 | 进程分钟之和 |
|---|---|---|---|---|
| BM+AE / LR | 20 | 40.05 | 26.15 | 800.94 |
| BM+AE / GBDT | 20 | 78.46 | 77.30 | 1569.16 |
| Joint / LR | 20 | 70.29 | 62.56 | 1405.80 |
| Joint / GBDT | 20 | 89.46 | 89.79 | 1789.25 |

可以说Joint本批进程时间更长且多数场景输出没有改变；不能在硬件负载、缓存不同的条件下把以上比值当严谨算法加速比。计算修复补齐了原80候选的完成条件，未悄悄把旧预算失败改成成功。

旧研究登记8054项，其中7795有效、259未成功；历史尝试9166条。259包括BM198、BM+AE21、Joint40。它们是原预算版本的完整记录，不是现在还排队等着跑的259个任务。旧BM根问题、更广Joint参数网格和单独预算敏感性不能因此宣称全部解决。

'''
    doc+=table(['方法','版本','角色','状态','登记任务数'],[(NAMES.get(r['method'],r['method']),r['method_version'],r['role'],r['status'],r['jobs']) for r in jobs])
    doc+='''## 十二、汇报时应明确保留的限制与未完成项

1. **回顾性扩展**：历史2024结果已经被查看；当前结果不能说成首次触碰的独立盲测。
2. **实现与文献**：应用实现和计算适配已记录；原文补充算法逐项忠实性尚未完整核实，不声称精确复现所有细节。
3. **预算和参数**：各方法搜索预算、恢复过程及种子数不同；本次新FairBias仅每格固定ε比例1，不是证明调尽参数后的最优表现。
4. **未完成的报告补强**：完整风险校准、风险指标配对区间、亚组绝对TPR/FPR展示及所有变量跨年分布尚未全面补齐。未算的区间保持NA，不能补写显著性。
5. **解释与应用**：变换语义有风险，尚无临床专家正式认可、部署或患者获益证据。
6. **当前操作状态**：主要80模型评价和本轮结果汇报无需服务器继续开机；全部工作基于本地已验证归档。

结尾可以这样说：“目前最明确的结果是，完整执行FairBias并没有自然转化成更好的公平性。我们同时拿到了预测、公平性、调查权重和表示语义四方面的证据。想先请您判断这些现象的研究价值和解释是否合理，再讨论论文定位或是否需要新的独立验证。”

## 附件与阅读顺序

- `01_微信汇报.md`：可以直接发送的文字，未代发。
- `03_全部96条件.md`：每个场景的全部方法和区间、所有NA。
- `04_全部168配对.md`：全部差值及名义/家族/合并区间，另含风险点差。
- `05_全部人群与缺失.md`：68行人群支持及336行变量描述。
- `06_旧研究完整结果.md`：旧138条主工作点可读表，完整334辅助和300配对CSV。
- `tables/`：上述全部机器可读精度、状态、种子数及标准差；另有最终变换和历史失败表。
- `figures/`：预测—公平性散点、Joint对BM+AE配对图、缺失语义、搜索及计算成本图。图是既有分析图，汇报时遵守表中口径。

本材料没有引入新的统计家族或重新选择对照。原结果和已有论文包均未改动。数据解释以当前分支为准，历史扩展单独标识。
'''
    txt('02_结果汇报详细大纲.md',doc)

    appendix='# 附录A：当前分支全部96个条件\n\n所有值为2024调查加权、注册种子平均。BA↑、AP↑、AUROC↑；EO↓、DP↓、Brier↓。BA/EO/DP方括号为名义95%设计区间，不是配对家族区间。风险指标当前无区间。NA按状态保留；q-only风险无定义与缺模型不同。\n\n'
    for a in ARMS:
        for b in ('LR','GBDT'):
            rr=[r for r in cond if r['arm_id']==a and r['backbone']==b]
            assert len(rr)==12
            appendix+=f'## {ARMS[a]} / {b}\n\n'
            data=[]
            for r in rr:
                cis=lambda m:num(r['T_'+m])+' ['+num(r[f'T_{m}_ci_low'])+', '+num(r[f'T_{m}_ci_high'])+']'
                data.append([NAMES[r['method']],ROLES[r['origin']],r['seed_count'],cis('BA'),cis('EO'),cis('DP'),num(r['T_AP']),num(r['T_AUC']),num(r['T_Brier'],5),r['S_status'],r['T_evaluation_status']])
            appendix+=table(['方法','来源','种子','BA [CI]','EO [CI]','DP [CI]','AP','AUROC','Brier','S','T'],data)
    appendix+='全部列含未加权、标准差及每个指标状态：`tables/current_conditions_96.csv`。\n'
    txt('03_全部96条件.md',appendix)

    appendix='# 附录B：全部168个配对\n\n差值方向为参考方法−对照方法；BA/EO差值及区间均为百分点。BA正为有利；EO正为不利。主要/次要家族20/316端点，合并336端点。缺失比较保留在分母中。“固定BM”由来源区分于原S选择BM。风险差值表为原始0–1尺度，未计算正式风险差值区间，DP配对估计/区间未提供。\n\n'
    for a in ARMS:
        for b in ('LR','GBDT'):
            rr=[r for r in pairs if r['arm_id']==a and r['backbone']==b]
            assert len(rr)==21
            appendix+=f'## {ARMS[a]} / {b}\n\n'
            appendix+=table(['参考−对照','来源/家族','ΔBA','BA名义','BA家族','BA合并','ΔEO','EO名义','EO家族','EO合并','状态'],[(NAMES[r['reference_method']]+' − '+NAMES[r['comparison_method']],r['kind']+'/'+r['family'],num(r['delta_BA'],2,100),interval(r,'BA','nominal'),interval(r,'BA'),interval(r,'BA','union'),num(r['delta_EO'],2,100),interval(r,'EO','nominal'),interval(r,'EO'),interval(r,'EO','union'),r['evaluation_status']) for r in rr])
            appendix+='风险指标点差（未作显著性结论）：\n\n'
            appendix+=table(['参考−对照','来源','ΔAP','ΔAUROC','ΔBrier'],[(NAMES[r['reference_method']]+' − '+NAMES[r['comparison_method']],r['kind'],num(r['delta_AP'],5),num(r['delta_AUC'],5),num(r['delta_Brier'],6)) for r in rr])
    txt('04_全部168配对.md',appendix)

    appendix='# 附录C/D：人群与逐变量描述\n\n## C. 全部68行队列/组支持\n\nKish有效样本量仅描述权重离散，不是完整聚类设计有效样本量。全部设计/权重分位字段在CSV保留。\n\n'
    appendix+=table(['分析臂','组','分区','年','n','事件数','加权事件率%','Kish ESS'],[(ARMS[r['arm_id']],r['group_label'],r['partition'],r['year'],r['n'],r['event_count'],num(r['weighted_event_rate'],2,100),num(r['kish_ess'],1)) for r in cohorts])
    miss=readcsv(READY/'missingness/v2/missingness_by_variable.csv')
    appendix+='## D. 全部336行变量描述\n\n表中类别为**n / 加权百分比**；未加权百分比可由n/分母得出，原始CSV也直接保存。NIU不是条目无应答；Arm4不纳入变量为NA。收入发布值含官方单次插补；“观测/发布值”不保证原始信息全被受访者报告。\n\n'
    for a in ARMS:
        for p in ('F','C','S','T'):
            rr=[r for r in miss if r['arm_id']==a and r['partition']==p];assert len(rr)==21
            appendix+=f'### {ARMS[a]} / {p}\n\n'
            appendix+=table(['变量','分母n','观测/发布值','条目无应答','结构NIU','源空值','未知/未映射'],[(r['official_name'],r['denominator_n'],*[r[k+'_n']+' / '+num(r[k+'_weighted_fraction'],3,100) for k in ('observed','item_nonresponse','structural_niu','raw_null','unknown_unmapped')]) for r in rr])
    txt('05_全部人群与缺失.md',appendix)

    appendix='# 附录E：旧固定预算研究\n\n此处是旧研究结果，不是新80完整计算的结果。共925个冻结模型；tau=.10表138行，其中125 VALID、13 NO_VALID_MODEL。另有334条辅助行及300个配对，全部CSV随包提供。相同模型可对应不同tau或比较，不把表行数叫训练模型数。以下风险指标是最终输出有定义时的p；基础分数另列于CSV，不替换q-only策略指标。\n\n'
    for a in ARMS:
        appendix+=f'## {ARMS[a]}\n\n'
        rr=[r for r in original if r['arm_id']==a]
        appendix+=table(['方法','骨干','训练加权','种子','BA','EO','DP','AP','AUROC','Brier','S状态','T状态'],[(NAMES.get(r['method'],r['method']),r['backbone'],r['training_weighted'],r['frozen_seed_count'],*[num(r[k],5 if k=='brier_p' else 4) for k in ('balanced_accuracy','eo_gap','dp_gap','average_precision_p','auroc_p','brier_p')],r['S_status'],r['T_status']) for r in rr])
    appendix+='## 失败与敏感性范围\n\n原始登记任务状态已列主报告，完整259条失败与4不支持槽位见tables。辅助334行包括其他tau/固定消融/敏感性；不是每个旧补充执行都完成正式评价。中间AE40、严格Joint和BM恢复尝试的完成记录，不得自动替代原登记失败或加入这张结果表。新80分支是独立完整执行/评价版本。\n'
    txt('06_旧研究完整结果.md',appendx if False else appendix)

    txt('README.md','''# FairBias / NHIS 给导师的结果汇报包

先复制01微信文字，再按02详细大纲汇报。03–06为完整可读附录，tables保留全部精度/状态与历史版本。这里只汇报results，不构建新的论文优越性结论。

当前分支96条件/168配对与旧研究138+334条件/300配对必须分开。新80已经完成；旧失败/不支持记录保留。所有csv为已验收源文件的逐字节副本；汇报摘要是这些聚合结果的描述，不新增统计检验。没有个体微观数据、模型或个体预测。也没有代发微信。

manifest.json记录输入/输出哈希和覆盖范围。源路径用于本地审计；不要求导师访问这些内部路径。文件中NA均保留原来的缺失/不适用意义，图表不得截取有利子集后声称普遍优越。
''')
    outputs={str(p.relative_to(OUT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(OUT.rglob('*')) if p.is_file()}
    manifest={'scope':'aggregate-only supervisor briefing, no new inference','input_sha256':INPUTS,'copied_tables':copied,'outputs':outputs,'coverage':{'current_conditions':96,'current_valid_conditions':90,'current_pairs':168,'current_valid_pairs':156,'main_configs':16,'completed_main_models':80,'reused_comparator_models':269,'current_unique_models':349,'old_selected_models':925,'old_tau010_rows':138,'old_auxiliary_rows':334,'old_pairs':300,'cohort_rows':68,'missingness_rows':336,'final_transformation_rows':1680},'generator_sha256':sha(__file__)}
    txt('manifest.json',json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    archive=OUT.parent/'FairBias_导师结果汇报_20260918.zip'
    assert not archive.exists()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,str(p.relative_to(OUT)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,spec in outputs.items():assert hashlib.sha256(z.read(name)).hexdigest()==spec['sha256']
    print(json.dumps({'status':'PASS','directory':str(OUT),'archive':str(archive),'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'members':len(outputs)+1,'coverage':manifest['coverage']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
