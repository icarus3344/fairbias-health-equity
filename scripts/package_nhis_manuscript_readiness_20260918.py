#!/usr/bin/env python3
"""Package approved aggregate paper material; preserve prior delivery bytes."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'artifacts/nhis/manuscript_readiness_20260918'
PAPER = ROOT / 'docs/paper/manuscript_readiness_20260918'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    archive = DEST / 'paper_bundle_readiness_v2_20260918.zip'
    inventory = DEST / 'delivery_manifest.json'
    proof_path = DEST / 'bundle_verification.json'
    for path in (archive, inventory, proof_path):
        if path.exists():
            raise FileExistsError(path)
    prior = ROOT / 'artifacts/nhis/completion_evaluation_20260918/paper_bundle_20260918.zip'
    assert digest(prior.read_bytes()) == '460b0b78469b632c8ddc5e791c2d8c07215fd32bf477881388ddae25a93c669d'
    members = {}
    with zipfile.ZipFile(prior) as z:
        assert z.testzip() is None
        for item in z.infolist():
            if not item.is_dir():
                members[item.filename] = z.read(item.filename)
    prior_members = len(members)
    paths = list(PAPER.glob('*.md'))
    for folder in ('submission','missingness/v2','missingness/figures_v3','literature','transformations/extraction'):
        paths.extend(p for p in (PAPER/folder).iterdir() if p.is_file())
    paths.extend(p for p in (PAPER/'transformations').iterdir() if p.is_file())
    paths.extend(p for p in (PAPER/'transformations/primary_sources').iterdir() if p.suffix in ('.json','.csv'))
    paths += [
        ROOT/'docs/plans/FAIRBIAS_MANUSCRIPT_READINESS_EXECUTION_20260918.md',
        ROOT/'docs/reports/FAIRBIAS_MANUSCRIPT_READINESS_ACCEPTANCE_20260918.md',
        ROOT/'scripts/build_nhis_predictor_missingness_20260918.py',
        ROOT/'scripts/build_nhis_transformation_review_20260918.py',
        ROOT/'scripts/plot_nhis_missingness_semantics_20260918.py',
        ROOT/'scripts/verify_nhis_manuscript_readiness_20260918.py',
        Path(__file__).resolve(),
        ROOT/'tests/test_nhis_paper_missingness_20260918.py',
        DEST/'supervisor_verification.json',
    ]
    for path in sorted(set(paths)):
        key = path.relative_to(ROOT).as_posix()
        assert key not in members, key
        assert path.suffix.lower() in {'.md','.json','.csv','.bib','.py','.png','.svg','.pdf'}, key
        assert not any(x in key for x in ('/data/raw/','/prepared/','/predictions/','/figures_v1/','/figures_v2/','/reproduction_check/')), key
        members[key] = path.read_bytes()
    members['START_HERE_READINESS_v2.md'] = '''# FairBias / NHIS 应用论文材料 v2

2026-09-18。先读[本轮完整报告](docs/paper/manuscript_readiness_20260918/READINESS_REPORT.md)，再看[新版英文正文](docs/paper/manuscript_readiness_20260918/MANUSCRIPT_METHODS_RESULTS_v2.md)和[摘要/大纲](docs/paper/manuscript_readiness_20260918/submission/ABSTRACT_AND_POSITIONING.md)。

本包保留上一版47个已验收材料，并新增缺失描述v2、最终变换审计、文献适配、投稿规范与缺失语义图v3。旧正文用于历史追踪；以新的主报告/正文为当前解释。变换图仍使用上一版的figures_completion_v2，新增缺失图使用manuscript_readiness/missingness/figures_v3，两者不是同一图的版本序号。

新80个模型已完成正式评价；本轮没有重训或改变选模。内部缺失统计初版和缺失图v1/v2不在本包。包内无个体数据、模型二进制、个体预测或密钥；官方代码本/期刊原文PDF也不打包，保留来源链接/本地检索哈希。聚合描述和代码域映射不等于临床验证。

包内脚本需配合本地封存输入才能重算；这不是包含全部执行环境与微数据的独立运行镜像。文献/来源清单中的内部绝对路径用于审计定位，不代表这些文件全部公开打包。当前Git HEAD不涵盖全部运行实现，正式发布版本和许可待作者落实。

本材料可以供导师审阅，不是已经签署的投稿终稿。作者事实表里的伦理、署名、资助、利益冲突、AI使用和全体批准必须如实确认；还需确定期刊、完成正文背景/讨论与最终可编辑排版。

delivery_manifest.json记录除自身外的包成员字节与SHA256。外部bundle_verification.json记录ZIP整体SHA256。原始已验收统计结果保持不变。
'''.encode()
    metadata = {'schema':'fairbias-manuscript-readiness-v2','status':'SUPERVISOR_ACCEPTED_BOUNDED_PACKAGE','prior_archive_sha256':digest(prior.read_bytes()),'preserved_prior_members':prior_members,'exclusions':['individual records','prepared objects','models','per-person predictions','credentials','third-party source PDFs','superseded missingness/figure versions'],'files':{k:{'bytes':len(v),'sha256':digest(v)} for k,v in sorted(members.items())}}
    inventory_bytes = (json.dumps(metadata,indent=2,sort_keys=True)+'\n').encode()
    inventory.write_bytes(inventory_bytes)
    members['delivery_manifest.json'] = inventory_bytes
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,data in sorted(members.items()):
            info=zipfile.ZipInfo(name,date_time=(2026,9,18,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,data)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist()) == len(members)
        for name,data in members.items():
            assert z.read(name) == data, name
    proof={'status':'PASS','archive':str(archive),'sha256':digest(archive.read_bytes()),'bytes':archive.stat().st_size,'members':len(members),'preserved_prior_members':prior_members,'all_member_hashes_match':True,'manifest_sha256':digest(inventory_bytes),'individual_records_exported':False,'official_source_pdfs_included':False,'note':'Source/provenance manifests intentionally reference internal files not distributed in this aggregate manuscript package.'}
    proof_path.write_text(json.dumps(proof,indent=2,sort_keys=True)+'\n')
    print(json.dumps(proof,indent=2))


if __name__ == '__main__':
    main()
