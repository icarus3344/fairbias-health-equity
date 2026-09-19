#!/usr/bin/env python3
"""Build the A1 metadata-only NHIS eligibility artifacts.

This program parses public-use frequency codebooks only.  It intentionally has
no dependency on microdata, predictions, models, or performance results.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PLAN = Path(__file__).resolve().parent
SOURCES = ROOT / "docs/paper/manuscript_readiness_20260918/transformations/primary_sources"
WORK = ROOT / "work/nhis_metadata_eligibility_v3"

CODEBOOKS = {
    2022: SOURCES / "adult-codebook-2022.pdf",
    2023: SOURCES / "adult-codebook-2023.pdf",
    2024: SOURCES / "adult-codebook-2024.pdf",
}
CHECKSUMS = {
    2022: SOURCES / "checksum-filelist-2022.pdf",
    2023: SOURCES / "checksum-filelist-2023.pdf",
}
URLS = {
    2022: {
        "codebook": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHIS/2022/Adult-codebook.pdf",
        "checksum": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHIS/2022/Checksum-Filelist.pdf",
    },
    2023: {
        "codebook": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHIS/2023/Adult-codebook.pdf",
        "checksum": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHIS/2023/Checksum-Filelist.pdf",
    },
}
RETRIEVED_UTC = {
    (2022, "codebook"): None,
    (2022, "checksum"): "2026-09-19T12:10:53Z",
    (2023, "codebook"): "2026-09-19T12:15:16Z",
    (2023, "checksum"): "2026-09-19T12:11:10Z",
}

ATLAS_PATH = PLAN / "VARIABLE_ATLAS_V3_DRAFT.csv"
FEATURES_PATH = ROOT / "configs/nhis/features.json"
LEDGER_PATH = PLAN / "CODEBOOK_WIDE_ELIGIBILITY_LEDGER_V3.csv"
MATRIX_PATH = PLAN / "CROSS_YEAR_HARMONIZATION_MATRIX_V3.csv"
MANIFEST_PATH = PLAN / "OFFICIAL_CODEBOOK_SOURCE_MANIFEST_V3.json"

ALLOWED_STATUSES = {
    "ELIGIBLE_FOR_HUMAN_REVIEW",
    "EXCLUDED_BY_PREDECLARED_RULE",
    "UNRESOLVED_METADATA",
}

# Exact-name scientific exclusions.  These are copied from the frozen feature
# registry and are not inferred from codebook frequencies.
OUTCOME_NAMES = {"MEDDL12M_A", "MEDNG12M_A"}
OUTCOME_PROXIMAL_NAMES = {
    "MEDDL12M_A", "MEDNG12M_A", "PAYBLL12M_A", "PAYNOBLLNW_A",
    "PAYWORRY_A", "DENDL12M_A", "DENNG12M_A", "RXSK12M_A",
    "RXLS12M_A", "RXDL12M_A", "RXDG12M_A",
}
DESIGN_NAMES = {"RECTYPE", "SRVY_YR", "HHX", "WTFA_A", "PSTRAT", "PPSU"}
ANCHOR_NAMES = {"SEX_A", "HISPALLP_A", "DISAB3_A"}
DISABILITY_CLUSTER = {
    "DISAB3_A", "VISIONDF_A", "HEARINGDF_A", "DIFF_A", "COMDIFF_A",
    "COGMEMDFF_A", "UPPSLFCR_A",
}
NATIVITY_CLUSTER = {"NATUSBORN_A", "CITZNSTP_A"}

FIELD_LABELS = [
    "Variable", "Module", "Section", "File(s)", "Data Type", "Length",
    "Question Text", "Fills", "Description", "Recode", "Universe",
    "Universe Description", "Sources", "Question ID", "Keywords", "Notes",
    "Evaluation Report", "Unweighted frequencies", "Frequency Missing",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def pdfinfo(path: Path) -> dict[str, str]:
    proc = subprocess.run(["pdfinfo", str(path)], check=True, capture_output=True, text=True)
    out: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            out[key.strip()] = value.strip()
    return out


def extract_pdf(path: Path, year: int) -> Path:
    WORK.mkdir(parents=True, exist_ok=True)
    target = WORK / f"adult-codebook-{year}-layout.txt"
    subprocess.run(["pdftotext", "-layout", str(path), str(target)], check=True)
    return target


def page_variable(page: str) -> str | None:
    match = re.search(r"(?m)^Variable:\s+([A-Z0-9_]+)\s*$", page)
    return match.group(1) if match else None


def parse_fields(text: str) -> dict[str, str]:
    positions: list[tuple[int, str, int]] = []
    label_pattern = "|".join(re.escape(x) for x in sorted(FIELD_LABELS, key=len, reverse=True))
    for match in re.finditer(rf"(?m)^({label_pattern}):\s*", text):
        positions.append((match.start(), match.group(1), match.end()))
    result: dict[str, str] = {}
    for idx, (start, label, value_start) in enumerate(positions):
        if label in result:
            continue
        end = positions[idx + 1][0] if idx + 1 < len(positions) else len(text)
        value = text[value_start:end]
        # Remove only printed page footer/header material from continuation text.
        value = re.sub(r"(?m)^\s*Page\s+\d+\s*$", "", value)
        value = re.sub(r"(?m)^\s*20\d{2} NATIONAL HEALTH INTERVIEW SURVEY \(NHIS\)\s*$", "", value)
        value = re.sub(r"(?m)^\s*Codebook for Sample Adult File .*\s*$", "", value)
        value = re.sub(r"(?m)^\s*PUBLIC USE\s*$", "", value)
        result[label] = norm(value)
    return result


def parse_codes(text: str) -> tuple[str, str]:
    match = re.search(
        r"Unweighted frequencies:\s*.*?\nCode\s+Description\s+Frequency\s+Percent\s*\n(.*?)\nFrequency Missing:",
        text,
        flags=re.S,
    )
    if not match:
        return "", ""
    substantive: list[str] = []
    missing: list[str] = []
    current: tuple[str, str] | None = None
    for raw in match.group(1).splitlines():
        line = raw.rstrip()
        row = re.match(r"^\s*(\S.*?)\s{2,}(.*?)\s{2,}[\d,]+\s+\d+(?:\.\d+)?\s*$", line)
        if row:
            code, description = norm(row.group(1)), norm(row.group(2))
            current = (code, description)
            marker = f"{code}|{description}" if code else description
            desc_l = description.lower()
            is_missing = any(
                token in desc_l
                for token in ("refused", "not ascertained", "don't know", "not in universe", "unknown")
            )
            (missing if is_missing else substantive).append(marker)
        elif current and line.strip() and not re.search(r"\d", line[-16:]):
            # Wrapped descriptions are uncommon; preserve their text without
            # trying to reinterpret the code.
            pass
    return " || ".join(substantive), " || ".join(missing)


def parse_codebook(path: Path, year: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    text_path = extract_pdf(path, year)
    pages = text_path.read_text(encoding="utf-8", errors="replace").split("\f")
    records: list[dict[str, Any]] = []
    zero_variable_pages: list[int] = []
    for page_no, page in enumerate(pages, start=1):
        var = page_variable(page)
        if var:
            fields = parse_fields(page)
            records.append({
                "variable": var,
                "page": page_no,
                "raw_text": page,
                "fields": fields,
            })
        elif page.strip():
            zero_variable_pages.append(page_no)
            if records:
                records[-1]["raw_text"] += "\n" + page
                records[-1]["fields"] = parse_fields(records[-1]["raw_text"])
    for record in records:
        substantive, missing = parse_codes(record["raw_text"])
        record["substantive_codes"] = substantive
        record["missing_niu_codes"] = missing
    return records, {
        "pdf_pages": int(pdfinfo(path)["Pages"]),
        "text_chunks": len(pages),
        "variable_count": len(records),
        "continuation_or_nonvariable_pages": zero_variable_pages,
    }


def role_for(name: str, fields: dict[str, str]) -> str:
    text = " ".join([name, fields.get("Section", ""), fields.get("Description", ""), fields.get("Keywords", "")]).lower()
    if name in ANCHOR_NAMES:
        return "ANCHOR"
    if name in OUTCOME_PROXIMAL_NAMES:
        return "OUTCOME_OR_COST_BARRIER_PROXIMAL"
    if name in DESIGN_NAMES or any(x in text for x in ("identifier", "variance estimation", "sample weight")):
        return "DESIGN_ID_WEIGHT"
    if name.endswith("_FLG") or "flag" in fields.get("Section", "").lower():
        return "ADMIN_PROCESSING"
    if name in DISABILITY_CLUSTER or any(x in text for x in ("difficulty", "disability", "functioning")):
        return "FUNCTIONAL_STATUS"
    if any(x in text for x in ("employment", "worked at a job", "hours per week", "job or business")):
        return "EMPLOYMENT"
    if any(x in text for x in ("poverty", "income", "food security", "food would", "housing", "basic needs")):
        return "SOCIAL_OR_MATERIAL_RESOURCES"
    if any(x in text for x in ("health insurance", "usual place", "health care", "healthcare", "medical care", "doctor")):
        return "HEALTHCARE_OR_ACCESS_CONTEXT"
    if any(x in text for x in ("health status", "diagnosed", "ever told", "symptom", "mental health", "anxiety", "depression")):
        return "CLINICAL_CONTEXT"
    if any(x in text for x in ("marital", "citizen", "born in", "education", "race", "hispanic", "sex of", "age of")):
        return "SOCIODEMOGRAPHIC"
    if any(x in text for x in ("region", "urban-rural", "household", "family")):
        return "LIVING_CONTEXT"
    return "OTHER_SUBSTANTIVE_CONTEXT"


def cluster_for(name: str) -> str:
    if name in DISABILITY_CLUSTER:
        return "DISAB3_WASHINGTON_GROUP_COMPONENTS"
    if name in NATIVITY_CLUSTER:
        return "NATIVITY_CITIZENSHIP"
    if name.startswith("FDS"):
        return "FOOD_SECURITY"
    if name.startswith("EMP"):
        return "EMPLOYMENT"
    if name in {"URBRRL", "URBRRL23", "REGION"}:
        return "GEOGRAPHIC_CONTEXT_NON_EQUIVALENT_SCHEMES"
    return ""


def classify(
    name: str,
    fields: dict[str, str],
    names_by_year: dict[int, set[str]],
) -> tuple[str, str]:
    section = fields.get("Section", "").lower()
    description = fields.get("Description", "").lower()
    files = fields.get("File(s)", "").lower()
    universe_desc = fields.get("Universe Description", "").lower()
    if name in DESIGN_NAMES or section.startswith("idn") or section.startswith("ucf"):
        return "EXCLUDED_BY_PREDECLARED_RULE", "IDENTIFIER_WEIGHT_STRATA_OR_PSU"
    if name in OUTCOME_PROXIMAL_NAMES:
        return "EXCLUDED_BY_PREDECLARED_RULE", "OUTCOME_OR_COST_BARRIER_PROXIMAL"
    if name.endswith("_FLG") or section.startswith("flg") or name in {"IMPNUM_A"}:
        return "EXCLUDED_BY_PREDECLARED_RULE", "PURE_ADMINISTRATIVE_OR_PROCESSING_FIELD"
    if any(x in description for x in ("prediction error", "model prediction", "residual score")):
        return "EXCLUDED_BY_PREDECLARED_RULE", "PREDICTION_OR_ERROR_DERIVED_FIELD"
    if "adult" not in files or ("sample child" in universe_desc and "sample adult" not in universe_desc):
        return "EXCLUDED_BY_PREDECLARED_RULE", "NO_ADULT_APPLICABILITY_DOMAIN"
    required = ["Description", "File(s)", "Data Type", "Universe"]
    if any(not fields.get(item, "").strip() for item in required):
        return "UNRESOLVED_METADATA", "REQUIRED_CODEBOOK_METADATA_MISSING"
    same_2022 = name in names_by_year[2022]
    same_2023 = name in names_by_year[2023]
    if name == "URBRRL23" and "URBRRL" in names_by_year[2022] and "URBRRL" in names_by_year[2023]:
        return "UNRESOLVED_METADATA", "URBAN_RURAL_SCHEME_VERSION_TRANSITION"
    if not same_2022 or not same_2023:
        return "UNRESOLVED_METADATA", "CROSS_YEAR_CONSTRUCT_NOT_ESTABLISHED"
    return "ELIGIBLE_FOR_HUMAN_REVIEW", "NONE_MACHINE_ELIGIBLE_NOT_FORMALLY_SELECTED"


def source_entry(path: Path, year: int, kind: str) -> dict[str, Any]:
    info = pdfinfo(path)
    url = URLS[year][kind]
    return {
        "year": year,
        "source_kind": "sample_adult_frequency_codebook" if kind == "codebook" else "public_use_file_list_and_checksum",
        "official_url": url,
        "http_final_url": url,
        "http_status": 200,
        "redirect_count": 0,
        "retrieved_at_utc": RETRIEVED_UTC[(year, kind)],
        "local_path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "pdf_pages": int(info["Pages"]),
        "pdf_title": info.get("Title", ""),
        "pdf_author": info.get("Author", ""),
        "embedded_or_header_year": year,
        "publisher_checksum_for_this_pdf": None,
        "publisher_checksum_algorithm": "NOT_STATED_AND_NO_CODEBOOK_PDF_CHECKSUM_LISTED",
    }


def checksum_entries(pdf_path: Path, year: int) -> list[dict[str, str]]:
    target = WORK / f"checksum-filelist-{year}.txt"
    subprocess.run(["pdftotext", "-layout", str(pdf_path), str(target)], check=True)
    text = target.read_text(encoding="utf-8", errors="replace")
    found: list[dict[str, str]] = []
    for file_name, checksum in re.findall(r"\b([a-z]+(?:inc)?\d{2}\.(?:csv|dat))\s+([0-9a-f]{32})\b", text, flags=re.I):
        found.append({
            "file_name": file_name,
            "checksum": checksum.lower(),
            "algorithm": "UNSPECIFIED_IN_PUBLISHER_DOCUMENT_32_HEX",
        })
    return found


def write_manifest(parse_stats: dict[int, dict[str, Any]], retrieved_2022: str) -> None:
    RETRIEVED_UTC[(2022, "codebook")] = retrieved_2022
    sources = []
    for year in (2022, 2023):
        sources.append(source_entry(CODEBOOKS[year], year, "codebook"))
        entry = source_entry(CHECKSUMS[year], year, "checksum")
        entry["listed_data_file_checksums"] = checksum_entries(CHECKSUMS[year], year)
        sources.append(entry)
    info_2024 = pdfinfo(CODEBOOKS[2024])
    sources.append({
        "year": 2024,
        "source_kind": "sample_adult_frequency_codebook",
        "official_url": None,
        "http_final_url": None,
        "http_status": None,
        "redirect_count": None,
        "retrieved_at_utc": None,
        "local_path": str(CODEBOOKS[2024].relative_to(ROOT)),
        "bytes": CODEBOOKS[2024].stat().st_size,
        "sha256": sha256(CODEBOOKS[2024]),
        "pdf_pages": int(info_2024["Pages"]),
        "pdf_title": info_2024.get("Title", ""),
        "pdf_author": info_2024.get("Author", ""),
        "embedded_or_header_year": 2024,
        "header_version": "18 June 2025",
        "publisher_checksum_for_this_pdf": None,
        "publisher_checksum_algorithm": "NOT_AVAILABLE_IN_THIS_GATE",
        "provenance_note": "Pre-existing local official NCHS/DHIS public-use codebook; not newly downloaded in this gate.",
    })
    entries_2022 = checksum_entries(CHECKSUMS[2022], 2022)
    entries_2023 = checksum_entries(CHECKSUMS[2023], 2023)
    manifest = {
        "schema_version": "nhis-official-codebook-source-manifest-v3-1.0",
        "gate": "A1_METADATA_ELIGIBILITY_V3",
        "status": "EVIDENCE_CAPTURED_PENDING_INDEPENDENT_REVIEW",
        "network_allowlist_observed": [URLS[y][k] for y in (2022, 2023) for k in ("codebook", "checksum")],
        "sources": sources,
        "parse_inventory": {str(year): parse_stats[year] for year in (2022, 2023, 2024)},
        "publisher_checksum_interpretation": {
            "algorithm_claim": "The publisher PDFs label the field only as Checksum; the 32-hex values are recorded without asserting MD5.",
            "codebook_pdf_checksum_listed": False,
            "2023_document_anomaly": (
                "The official URL and page header say 2023, but the body lists the same adult22/child22/paradata22 "
                "filenames, record counts, sizes, and checksum strings as the 2022 document. It is retained as official raw evidence, "
                "but is not treated as a valid 2023 Adult-codebook checksum or as evidence for adult23 data-file identity."
            ),
            "2022_and_2023_listed_entries_identical": entries_2022 == entries_2023,
        },
        "scientific_boundary": "Metadata provenance only; no microdata, predictions, performance, or 2025 content was accessed.",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build(retrieved_2022: str) -> None:
    for path in [*CODEBOOKS.values(), *CHECKSUMS.values(), ATLAS_PATH, FEATURES_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)
    parsed: dict[int, list[dict[str, Any]]] = {}
    parse_stats: dict[int, dict[str, Any]] = {}
    for year in (2022, 2023, 2024):
        parsed[year], parse_stats[year] = parse_codebook(CODEBOOKS[year], year)
    by_year = {year: {r["variable"]: r for r in records} for year, records in parsed.items()}
    names_by_year = {year: set(records) for year, records in by_year.items()}

    features = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))["feature_lists"]
    original_x = {x.upper() for x in features["primary_core_features"]}
    with ATLAS_PATH.open(newline="", encoding="utf-8") as fh:
        atlas_rows = list(csv.DictReader(fh))
    atlas_names = {row["variable"] for row in atlas_rows}

    ledger_fields = [
        "variable", "module", "section", "official_label_description", "official_page_2024",
        "data_type", "universe", "universe_description_routing", "recode", "construct_role",
        "in_original_X", "is_outcome_Y", "is_cost_barrier_proximal", "is_design_id_weight_psu_strata",
        "is_public_use", "candidate_name_2022", "candidate_name_2023", "nested_or_related_cluster",
        "admission_stage", "exclusion_or_unresolved_reason", "source_sha256_2024",
    ]
    ledger_rows: list[dict[str, Any]] = []
    source_hash_2024 = sha256(CODEBOOKS[2024])
    for record in parsed[2024]:
        name, fields = record["variable"], record["fields"]
        status, reason = classify(name, fields, names_by_year)
        candidate_2022 = name if name in names_by_year[2022] else ("URBRRL" if name == "URBRRL23" and "URBRRL" in names_by_year[2022] else "")
        candidate_2023 = name if name in names_by_year[2023] else ("URBRRL" if name == "URBRRL23" and "URBRRL" in names_by_year[2023] else "")
        row = {
            "variable": name,
            "module": fields.get("Module", ""),
            "section": fields.get("Section", ""),
            "official_label_description": fields.get("Description", ""),
            "official_page_2024": record["page"],
            "data_type": fields.get("Data Type", ""),
            "universe": fields.get("Universe", ""),
            "universe_description_routing": fields.get("Universe Description", ""),
            "recode": fields.get("Recode", ""),
            "construct_role": role_for(name, fields),
            "in_original_X": str(name in original_x).lower(),
            "is_outcome_Y": str(name in OUTCOME_NAMES).lower(),
            "is_cost_barrier_proximal": str(name in OUTCOME_PROXIMAL_NAMES).lower(),
            "is_design_id_weight_psu_strata": str(name in DESIGN_NAMES or status == "EXCLUDED_BY_PREDECLARED_RULE" and reason == "IDENTIFIER_WEIGHT_STRATA_OR_PSU").lower(),
            "is_public_use": "true",
            "candidate_name_2022": candidate_2022,
            "candidate_name_2023": candidate_2023,
            "nested_or_related_cluster": cluster_for(name),
            "admission_stage": status,
            "exclusion_or_unresolved_reason": reason,
            "source_sha256_2024": source_hash_2024,
        }
        assert row["admission_stage"] in ALLOWED_STATUSES
        ledger_rows.append(row)
    with LEDGER_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ledger_fields)
        writer.writeheader()
        writer.writerows(ledger_rows)

    eligible_names = {r["variable"] for r in ledger_rows if r["admission_stage"] == "ELIGIBLE_FOR_HUMAN_REVIEW"}
    canonical_names = sorted(atlas_names | eligible_names)
    ledger_by_name = {r["variable"]: r for r in ledger_rows}
    source_hashes = {year: sha256(CODEBOOKS[year]) for year in (2022, 2023, 2024)}
    matrix_fields = [
        "canonical_variable", "seed_atlas", "eligibility_status_2024", "year", "year_variable",
        "official_label_description", "substantive_codes", "missing_or_niu_codes", "universe",
        "universe_description_routing", "module", "section", "data_type", "recode", "official_page",
        "source_sha256", "name_match_status", "construct_version", "mapping_decision",
    ]
    matrix_rows: list[dict[str, Any]] = []
    for canonical in canonical_names:
        for year in (2022, 2023, 2024):
            year_name = canonical
            transition = canonical == "URBRRL23" and year in (2022, 2023)
            if transition:
                year_name = "URBRRL"
            record = by_year[year].get(year_name)
            if record is None:
                matrix_rows.append({
                    "canonical_variable": canonical,
                    "seed_atlas": str(canonical in atlas_names).lower(),
                    "eligibility_status_2024": ledger_by_name.get(canonical, {}).get("admission_stage", "ATLAS_SEED_NOT_IN_2024_LEDGER_ERROR"),
                    "year": year,
                    "year_variable": "",
                    "official_label_description": "",
                    "substantive_codes": "",
                    "missing_or_niu_codes": "",
                    "universe": "",
                    "universe_description_routing": "",
                    "module": "",
                    "section": "",
                    "data_type": "",
                    "recode": "",
                    "official_page": "",
                    "source_sha256": source_hashes[year],
                    "name_match_status": "NO_NAME_CANDIDATE",
                    "construct_version": "UNRESOLVED",
                    "mapping_decision": "UNRESOLVED_NO_OFFICIAL_ENTRY_MATCHED",
                })
                continue
            fields = record["fields"]
            matrix_rows.append({
                "canonical_variable": canonical,
                "seed_atlas": str(canonical in atlas_names).lower(),
                "eligibility_status_2024": ledger_by_name[canonical]["admission_stage"],
                "year": year,
                "year_variable": year_name,
                "official_label_description": fields.get("Description", ""),
                "substantive_codes": record["substantive_codes"],
                "missing_or_niu_codes": record["missing_niu_codes"],
                "universe": fields.get("Universe", ""),
                "universe_description_routing": fields.get("Universe Description", ""),
                "module": fields.get("Module", ""),
                "section": fields.get("Section", ""),
                "data_type": fields.get("Data Type", ""),
                "recode": fields.get("Recode", ""),
                "official_page": record["page"],
                "source_sha256": source_hashes[year],
                "name_match_status": "VERSIONED_NAME_CANDIDATE" if transition else "EXACT_NAME_MATCH",
                "construct_version": "PRE_2023_URBAN_RURAL_SCHEME" if transition else ("2023_NCHS_URBAN_RURAL_SCHEME" if canonical == "URBRRL23" else f"{year}_OFFICIAL_METADATA"),
                "mapping_decision": "UNRESOLVED_VERSION_TRANSITION_DO_NOT_POOL" if transition or canonical == "URBRRL23" else "OFFICIAL_METADATA_RECORDED_EQUIVALENCE_NOT_ASSUMED",
            })
    with MATRIX_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=matrix_fields)
        writer.writeheader()
        writer.writerows(matrix_rows)

    write_manifest(parse_stats, retrieved_2022)
    status_counts = Counter(row["admission_stage"] for row in ledger_rows)
    print(json.dumps({
        "ledger_rows": len(ledger_rows),
        "status_counts": status_counts,
        "harmonization_variables": len(canonical_names),
        "harmonization_rows": len(matrix_rows),
        "atlas_variables": len(atlas_names),
        "source_hashes": source_hashes,
    }, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieved-2022-codebook-utc", required=True)
    args = parser.parse_args()
    build(args.retrieved_2022_codebook_utc)
