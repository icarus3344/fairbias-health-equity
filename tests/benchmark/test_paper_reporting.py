import json
import hashlib
import csv

from nhis_fairbias.benchmark.paper_reporting import write_paper_reporting


def test_paper_reporting_preserves_estimands_statuses_and_all_failures(tmp_path):
    registration = tmp_path / "registration.json"
    selection = tmp_path / "selection_freeze.json"
    summary = tmp_path / "summary_T.json"
    individual = tmp_path / "arm_001_individual_model_metrics.json"
    jobs = []
    registration.write_text(json.dumps({"candidates": [
        {"candidate_id": "c1", "status": "REGISTERED", "seeds": [0]},
        {"candidate_id": "c2", "status": "REGISTERED", "seeds": [0]},
        {"candidate_id": "c3", "status": "REGISTERED", "seeds": [0]},
    ]}))
    reg_hash = hashlib.sha256(registration.read_bytes()).hexdigest()
    selection.write_text(json.dumps({"registration_sha256": reg_hash, "selections": [
        {"arm_id": "arm_001", "method": "FAIRBIAS_BM", "backbone": "LR", "tau": .1, "training_weighted": False, "status": "FEASIBLE"},
        {"arm_id": "arm_002", "method": "BASE", "backbone": "LR", "tau": .1, "training_weighted": False, "status": "NO_FEASIBLE_CONFIGURATION"},
    ]}))
    summary.write_text(json.dumps({"selections": [
        {"arm_id": "arm_001", "method": "FAIRBIAS_BM", "backbone": "LR", "tau": .1,
         "status": "FEASIBLE", "evaluation_status": "VALID",
         "balanced_accuracy": {"mean": .7, "seed_sd": .02, "taylor_95": {"se": .03, "lower": .64, "upper": .76}},
         "eo_gap": {"mean": .1}, "average_precision": {"mean": .61},
         "untouched_base_average_precision": {"mean": .58}},
        {"arm_id": "arm_001", "method": "BASE", "backbone": "LR", "tau": .2,
         "status": "NO_FEASIBLE_CONFIGURATION", "evaluation_status": "VALID", "training_weighted": False,
         "balanced_accuracy": {"mean": .62}, "average_precision": {"mean": None}, "untouched_base_average_precision": {"mean": .58}},
        {"arm_id": "arm_002", "method": "BASE", "backbone": "LR", "tau": .1,
         "status": "NOT_ESTIMABLE", "evaluation_status": "DESIGN_NOT_ESTIMABLE"},
    ], "paired_contrasts": [{"arm_id": "arm_001", "delta_balanced_accuracy": {"estimate": .08, "se": .012},
                             "delta_eo_gap": {"estimate": -.04, "projection_95": {"lower": -.07, "upper": -.01}}}]}))
    individual.write_text(json.dumps({"c1_s0": {"risk": {"average_precision": .61}}}))
    (tmp_path / "evaluation_manifest.json").write_text(json.dumps({
        "selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest()
    }))
    value = json.loads(summary.read_text())
    value["evaluation_manifest_sha256"] = hashlib.sha256((tmp_path / "evaluation_manifest.json").read_bytes()).hexdigest()
    summary.write_text(json.dumps(value))
    for candidate, name, status in (("c1", "ok.json", "VALID"), ("c2", "failed.json", "MODEL_FIT_FAILED"), ("c3", "missing.json", "NO_VALID_MODEL")):
        path = tmp_path / name
        path.write_text(json.dumps({"candidate_id": candidate, "seed": 0, "status": status}))
        jobs.append(path)
    outputs = write_paper_reporting(registration, selection, summary, [individual], tmp_path / "report", job_result_paths=jobs)
    text = outputs["markdown"].read_text()
    csv_text = outputs["csv"].read_text()
    assert "失败数：2" in text
    assert "NOT_ESTIMABLE" in csv_text
    assert "average_precision_p" in csv_text
    assert "untouched_base_average_precision" in csv_text
    assert "seed SD" in text and "design SE" in text
    assert "不默认 FairBias 胜出" in text
    assert "S 状态计数" in text and "job result 状态计数" in text
    assert outputs["markdown"].parent.joinpath("paper_results_auxiliary.csv").exists()
    assert outputs["markdown"].parent.joinpath("paper_paired_contrasts.csv").exists()
    with outputs["csv"].open() as handle:
        main_rows = list(csv.DictReader(handle))
    assert float(main_rows[0]["design_se_balanced_accuracy"]) == .03
    with (outputs["markdown"].parent / "paper_results_auxiliary.csv").open() as handle:
        aux = list(csv.DictReader(handle))[0]
    assert float(aux["balanced_accuracy"]) == .62
    assert aux["average_precision_p"] == ""
    assert float(aux["untouched_base_average_precision"]) == .58
    with (outputs["markdown"].parent / "paper_paired_contrasts.csv").open() as handle:
        pair = list(csv.DictReader(handle))[0]
    assert float(pair["delta_balanced_accuracy"]) == .08
    assert float(pair["delta_eo_gap"]) == -.04
    assert float(pair["delta_eo_projection_upper"]) == -.01


def test_paper_reporting_rejects_incomplete_registered_job_coverage(tmp_path):
    registration = tmp_path / "registration.json"
    selection = tmp_path / "selection.json"
    summary = tmp_path / "summary.json"
    individual = tmp_path / "individual.json"
    registration.write_text(json.dumps({"candidates": [{"candidate_id": "c1", "status": "REGISTERED", "seeds": [0, 1]}]}))
    selection.write_text(json.dumps({"registration_sha256": hashlib.sha256(registration.read_bytes()).hexdigest(), "selections": []}))
    summary.write_text(json.dumps({"selections": []}))
    (tmp_path / "evaluation_manifest.json").write_text(json.dumps({"selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest()}))
    summary.write_text(json.dumps({"selections": [], "evaluation_manifest_sha256": hashlib.sha256((tmp_path / "evaluation_manifest.json").read_bytes()).hexdigest()}))
    individual.write_text("{}")
    job = tmp_path / "only_one.json"
    job.write_text(json.dumps({"candidate_id": "c1", "seed": 0, "status": "VALID"}))
    try:
        write_paper_reporting(registration, selection, summary, [individual], tmp_path / "report", job_result_paths=[job])
    except ValueError as exc:
        assert "coverage mismatch" in str(exc)
        assert not (tmp_path / "report").exists()
    else:
        raise AssertionError("incomplete registered job coverage was accepted")


def test_paper_reporting_refuses_overwrite(tmp_path):
    paths = []
    for name, value in (("registration.json", {"candidates": []}), ("selection.json", {"registration_sha256": "bad", "selections": []}), ("summary.json", {"selections": []}), ("individual.json", {})):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        paths.append(path)
    out = tmp_path / "existing"
    out.mkdir()
    try:
        write_paper_reporting(*paths[:3], [paths[3]], out)
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing output directory was overwritten")
