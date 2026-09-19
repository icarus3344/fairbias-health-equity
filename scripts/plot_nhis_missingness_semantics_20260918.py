#!/usr/bin/env python3
"""Plot aggregate NHIS missingness semantics for the manuscript bundle.

Only the sealed v2 aggregate CSV is read.  No raw source, prepared bundle,
model, prediction, or identifier is accessed by this plotting script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = pathlib.Path(__file__).resolve()
INPUT = REPO / "docs/paper/manuscript_readiness_20260918/missingness/v2/missingness_by_variable.csv"
OUT = REPO / "docs/paper/manuscript_readiness_20260918/missingness/figures_v3"
PARTITIONS = ("F", "C", "S", "T")
VARIABLES = (
    "agep_a", "educp_a", "region", "pcnt18uptc", "pcntlt18tc", "ratcat_a",
    "empwrklsw1_a", "empwrkft1_a", "notcov_a", "phstat_a", "hypev_a",
    "chlev_a", "dibev_a", "asev_a", "visiondf_a", "hearingdf_a", "diff_a",
    "comdiff_a", "uppslfcr_a", "cogmemdff_a", "usualpl_a",
)
STATES = ("observed", "structural_niu", "item_nonresponse", "raw_null")
STATE_LABELS = {
    "observed": "Observed",
    "structural_niu": "Structural NIU",
    "item_nonresponse": "Item nonresponse",
    "raw_null": "Raw null",
}
COLORS = {
    "observed": "#4477AA",
    "structural_niu": "#CC6677",
    "item_nonresponse": "#DDCC77",
    "raw_null": "#66CCEE",
}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_aggregate(path: pathlib.Path) -> pd.DataFrame:
    require(path.is_file(), f"Missing aggregate input: {path}")
    frame = pd.read_csv(path)
    require(set(frame["arm_id"]) == {"arm_001", "arm_002", "arm_003", "arm_004"}, "unexpected arm coverage")
    arm = frame[(frame["arm_id"] == "arm_001") & (frame["status"] == "VALID")].copy()
    require(set(arm["partition"]) == set(PARTITIONS), "Arm001 partition coverage mismatch")
    require(tuple(dict.fromkeys(arm["variable"])) == VARIABLES, "Arm001 variable order/coverage mismatch")
    require(len(arm) == 84, f"expected 84 included Arm001 rows, got {len(arm)}")
    require(not arm.duplicated(["variable", "partition"]).any(), "duplicate Arm001 variable/partition rows")
    require(arm[["variable", "partition"]].drop_duplicates().shape == (84, 2), "variable/partition uniqueness mismatch")
    for state in STATES:
        for suffix in ("fraction", "weighted_fraction"):
            col = f"{state}_{suffix}"
            require(col in arm.columns, f"missing state column: {col}")
            values = pd.to_numeric(arm[col], errors="coerce").to_numpy(dtype=float)
            require(np.isfinite(values).all(), f"non-finite state fractions in {col}")
            require(((values >= 0.0) & (values <= 1.0)).all(), f"state fractions outside [0, 1] in {col}")
    for suffix in ("fraction", "weighted_fraction"):
        state_total = sum(arm[f"{state}_{suffix}"] for state in STATES)
        require(np.allclose(state_total.to_numpy(dtype=float), 1.0, atol=1e-9, rtol=0.0), f"four-state sum mismatch for {suffix}")
    require((arm["unknown_unmapped_fraction"] == 0.0).all(), "unexpected unknown_unmapped fraction")
    require((arm["unknown_unmapped_weighted_fraction"] == 0.0).all(), "unexpected weighted unknown_unmapped fraction")
    require((arm["preprocessing_missing_weighted_fraction"].notna()).all(), "missing weighted preprocessing fractions")
    return arm


def make_figure(arm: pd.DataFrame, output: pathlib.Path) -> None:
    pivot = arm.set_index(["variable", "partition"])
    fig = plt.figure(figsize=(13.5, 11.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=(1.0, 2.35))
    ax_state = fig.add_subplot(grid[0, 0])
    x = np.arange(len(PARTITIONS))
    bottoms = np.zeros(len(PARTITIONS), dtype=float)
    for state in STATES:
        values = np.array([
            float(pivot.loc[("empwrkft1_a", partition), f"{state}_weighted_fraction"]) * 100.0
            for partition in PARTITIONS
        ])
        ax_state.bar(
            x,
            values,
            bottom=bottoms,
            color=COLORS[state],
            edgecolor="white",
            linewidth=0.5,
            label=STATE_LABELS[state],
        )
        bottoms += values
    ax_state.set_ylim(0, 100)
    ax_state.set_xticks(x, PARTITIONS)
    ax_state.set_ylabel("Share of Arm001 domain (%)")
    ax_state.set_title("A. EMPWRKFT1_A input-state composition (survey-weighted)", loc="left", fontweight="bold")
    ax_state.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    ax_state.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax_state.set_axisbelow(True)
    ax_state.text(
        0.99,
        0.96,
        "Composition; not an item-missing rate",
        transform=ax_state.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#444444",
    )

    ax_heat = fig.add_subplot(grid[1, 0])
    matrix = np.array([
        [float(pivot.loc[(variable, partition), "item_nonresponse_weighted_fraction"]) * 100.0 for partition in PARTITIONS]
        for variable in VARIABLES
    ])
    image = ax_heat.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(1.0, float(np.nanmax(matrix))))
    ax_heat.set_xticks(np.arange(len(PARTITIONS)), PARTITIONS)
    ax_heat.set_yticks(np.arange(len(VARIABLES)), VARIABLES)
    ax_heat.set_xlabel("Analysis domain")
    ax_heat.set_ylabel("Core predictor")
    ax_heat.set_title("B. Item-nonresponse fraction by predictor and domain (survey-weighted)", loc="left", fontweight="bold")
    ax_heat.tick_params(axis="y", labelsize=8)
    ax_heat.tick_params(axis="x", labelsize=9)
    colorbar = fig.colorbar(image, ax=ax_heat, pad=0.015, fraction=0.025)
    colorbar.set_label("Item nonresponse (%)")
    for i in range(len(VARIABLES)):
        for j in range(len(PARTITIONS)):
            value = matrix[i, j]
            color = "white" if value > 0.55 * float(np.nanmax(matrix)) else "black"
            ax_heat.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color=color)
    ax_heat.set_xticks(np.arange(-0.5, len(PARTITIONS), 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, len(VARIABLES), 1), minor=True)
    ax_heat.grid(which="minor", color="white", linewidth=0.8)
    ax_heat.tick_params(which="minor", bottom=False, left=False)

    fig.suptitle("NHIS FairBias core-predictor missingness semantics", fontsize=15, fontweight="bold")
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "nhis_missingness_semantics_arm001.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "nhis_missingness_semantics_arm001.pdf", bbox_inches="tight")
    fig.savefig(output / "nhis_missingness_semantics_arm001.svg", bbox_inches="tight")
    plt.close(fig)


def write_caption(output: pathlib.Path) -> None:
    caption = """# Figure caption

**Figure. NHIS FairBias core-predictor missingness semantics for Arm001.** (A) Survey-weighted composition of the `EMPWRKFT1_A` input states in the F, C, S, and T analysis domains. The categories are shown separately: observed, structural not-in-universe (NIU), official item nonresponse, and raw null. The four displayed states together account for the whole domain. In T, combining the three non-observed states (structural NIU, item nonresponse and raw null) gives 44.31% **UNWEIGHTED** and 38.39% **WEIGHTED** non-observed-or-routed records; neither is an item-missing rate. (B) Survey-weighted official item-nonresponse percentages for all 21 registered core predictors across the same four domains. F/C are 2022 fitting/calibration, S is 2023 selection, and T is the permitted descriptive 2024 domain; each domain is shown separately without an equal-average or pooled-across-partitions interpretation.

Employment semantics follow the public NHIS source routing. `EMPWRKLSW1_A` covers employment situations including temporary absence, seasonal/contract work, and unpaid work; the figure should not be read as a literal “worked last week” prevalence plot. In the frozen preprocessing contract, only source routing `EMPWRKLSW1_A = 2` defines the `EMPWRKFT1_A` structural-NIU state. Official refusal/don't-know codes, raw blanks, and unexpected source codes remain distinct in the aggregate audit.

Source: `../v2/missingness_by_variable.csv`; all plotted values are aggregate-only, and no individual records or identifiers are exported.
"""
    (output / "FIGURE_CAPTION.md").write_text(caption, encoding="utf-8")


def write_hash_manifest(output: pathlib.Path, input_path: pathlib.Path) -> None:
    outputs = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "hash_manifest.json"
    }
    manifest = {
        "schema_version": "nhis-missingness-figure-v3",
        "scope": "aggregate_only",
        "input": {"path": str(input_path), "sha256": sha256_file(input_path)},
        "script": {"path": str(SCRIPT), "sha256": sha256_file(SCRIPT)},
        "outputs": outputs,
    }
    (output / "hash_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(input_path: pathlib.Path = INPUT, output: pathlib.Path = OUT) -> pathlib.Path:
    require(not output.exists(), f"Refusing to overwrite existing figure directory: {output}")
    arm = load_aggregate(input_path)
    make_figure(arm, output)
    write_caption(output)
    write_hash_manifest(output, input_path)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(INPUT), help=argparse.SUPPRESS)
    parser.add_argument("--output", default=str(OUT), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    path = run(pathlib.Path(args.input).resolve(), pathlib.Path(args.output).resolve())
    print(f"Figure export complete: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
