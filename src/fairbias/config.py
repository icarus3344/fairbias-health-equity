"""Configuration dataclasses and validation for the FairBias framework."""

from __future__ import annotations

import dataclasses
from typing import Any, Optional, Sequence

# ---------------------------------------------------------------------
# Algorithm modes (Round 4.1; renamed twice — the second rename is the
# Round 4.1 REPAIR-2 verdict of 2026-08-30)
#
# ``official_code_derived_monotone_cursor_unweighted``: an
# OFFICIAL-CODE-DERIVED VARIANT WITH A TERMINATION-SAFETY EXTENSION.
# It is derived from the behavior of the official code repository
# (github.com/zftang/MachineClassifer_BiasMitigation_beta, designated
# by the paper's Data Availability statement) but is NOT claimed to be
# behaviorally equivalent to it, and is NOT a "strict reproduction of
# the paper method":
#   - MDS embedding dimension FIXED AT 2, inherited from the official
#     code (official source comment: "here the MDS dimensition is fixed
#     at 2"); the paper TEXT (p. 5) instead prescribes selecting the
#     optimal MDS dimension via an elbow plot.  A paper-text mode
#     (elbow-plot dimension selection) is NOT implemented in this
#     codebase.
#   - power search over the OFFICIAL interleaved stream
#     [3, 1/3, 5, 1/5, ..., 1999, 1/1999] in the official (interleaved,
#     NOT ascending) order — inherited from the official code.
#   - DELIBERATE DEVIATION (termination-safety extension): a MONOTONE
#     per-attribute stream cursor (Round 4.1 REPAIR) that consumes each
#     searched position exactly once, so revisits advance forward-only
#     and powers are never reused.  The official implementation instead
#     RESTARTS the power search from the head of the stream on every
#     revisit, so powers rejected earlier can be retried after other
#     attributes change.  The cursor permanently excludes them, so this
#     mode can diverge from official-code behavior in general; the
#     unchanged COMPAS/Credit trajectories are an empirical
#     observation, NOT an equivalence guarantee.  A true
#     official-behavior reference mode (state-cycle detection with
#     fail-closed stop, unaltered candidate stream) may be added later
#     as a SEPARATE mode; it is not implemented here.
#   - NO finite iteration budget: the greedy loop terminates via the
#     epsilon ball, per-attribute exhaustion of the (finite) official
#     stream under the monotone cursor, or bounded categorical merge
#     chains — the cursor, not the stream's mere finiteness, is what
#     guarantees termination.
#   - NO validation-Pareto rollback: the greedy termination state is the
#     sole reported state.
#
# ``engineering_bounded``: the pre-existing bounded configuration
# (automatic stress-elbow MDS dimension, six-value ascending power grid,
# finite ``max_iterations`` budget, validation-Pareto checkpoint as an
# explicitly named engineering extension).  This mode makes NO
# paper-alignment claim.
#
# Mode resolution (see ``FairBiasConfig.resolved``) happens once, at
# pipeline entry, so every downstream component (evaluator, mitigation
# engine, output manifest) sees the concrete effective configuration.
# ---------------------------------------------------------------------
ALGORITHM_MODE_OFFICIAL = "official_code_derived_monotone_cursor_unweighted"
ALGORITHM_MODE_ENGINEERING = "engineering_bounded"
ALGORITHM_MODE_PAPER_FAITHFUL = "tang2024_paper_faithful"
ALGORITHM_MODES = (
    ALGORITHM_MODE_OFFICIAL,
    ALGORITHM_MODE_ENGINEERING,
    ALGORITHM_MODE_PAPER_FAITHFUL,
)

# The official implementation fixes the MDS embedding dimension at 2.
OFFICIAL_FIXED_MDS_DIM = 2

# Official interleaved power stream upper bound: the official code uses
# ``np.array([[i, 1/i] for i in range(3, 2000, 2)]).reshape(-1)``
# (MachineClassifer_BiasMitigation.py, line 198).
OFFICIAL_POWER_STREAM_MAX = 1999


def official_power_stream(up_to: int = OFFICIAL_POWER_STREAM_MAX) -> tuple:
    """Official interleaved power search stream ``[3, 1/3, 5, 1/5, ..., up_to, 1/up_to]``.

    Mirrors the official repository's
    ``np.array([[i, 1/i] for i in range(3, 2000, 2)]).reshape(-1)``
    exactly, INCLUDING the interleaved order (odd integer, then its
    reciprocal).  The order matters: the official implementation searches
    the stream in this order per attribute, which differs from an
    ascending sort.
    """
    stream = []
    for i in range(3, int(up_to) + 1, 2):
        stream.append(float(i))
        stream.append(1.0 / float(i))
    return tuple(stream)


@dataclasses.dataclass(frozen=True)
class FairBiasConfig:
    """Immutable configuration container for FairBias benchmarking runs."""

    # Algorithm mode (Round 4.1; renamed twice — see the module-level
    # mode contract above): "official_code_derived_monotone_cursor_unweighted"
    # (official-code-derived variant with a termination-safety extension;
    # neither an official-code equivalence claim nor a paper-text method
    # reproduction) or "engineering_bounded".
    algorithm_mode: str = ALGORITHM_MODE_ENGINEERING
    # When not None, the MDS embedding dimension is FIXED at this value and
    # the stress-elbow search is skipped entirely.  The official mode
    # forces 2 (official implementation behavior); None (default) keeps
    # the automatic elbow selection of the engineering mode.
    mds_fixed_components: Optional[int] = None

    dataset_name: str = "compas"
    dataset_path: str = "data_COMPAS.csv"
    label_Y: str = "two_year_recid"
    label_O: tuple[str, ...] = ("sex",)
    
    # Paper split: training 64% / validation 16% / test 20% (Eq. "Implementation details")
    test_size: float = 0.20
    val_size: float = 0.16
    random_seed: int = 0
    stratify_split: bool = True
    
    classifier: str = "LR"  # LR, DT, RF, GBDT, XGB, LGBM, CatBoost
    eval_norm: str = "min-max"  # min-max, z-score, none
    
    max_iterations: int = 5
    threshold_epsilon: float = 0.5
    threshold_accuracy: float = 0.01
    accuracy_tolerance_tau: float = 0.01  # Pareto constraint: ACC >= initial_ACC - tau
    selection_metric: str = "EO"  # EO or SP for best-iteration selection
    
    use_bias_mitigation: bool = True
    use_accuracy_enhancement: bool = False
    
    # Feature transform bounds
    transform_n_bins: int = 10
    transform_log_epsilon: float = 1e-5
    # None = NO additional magnitude guard: the only bound on power
    # transforms is the paper's numpy.float32 overflow rule (≈3.4e38 ->
    # attribute dropped).  Setting a numeric value re-enables a NON-PAPER
    # engineering guard and must be reported as such.
    transform_x_max: Optional[float] = None
    
    adaptive_threshold_method: str = "kmeans_125"  # "kmeans_125" or "ratio"
    h_order: int = 1  # Level-H exclusion order (Eq. 4/5): near-full companion contexts

    # Paper-level d_phi computation (bias_metric, Eqs. 1-6)
    mds_max_components: int = 15
    mds_slope_threshold: float = 0.01
    eval_divergence_num: str = "num-a"  # Eq. 2 numerical: centroid distance after min-max normalization
    eval_divergence_cat: str = "cat-a"  # Eq. 2 categorical: mean absolute frequency gap over K categories

    # Mitigation acceptance criteria
    phi_threshold: float = 100.0  # NMI information-loss gate (baseline PARAMS_MAIN_THRESHOLD_PHI)
    # CONFIGURED candidate grid (finite implementation budget).  The
    # paper's main text lists odd integers (3, 5, 7) and odd fractions
    # (1/3, 1/5, 1/7) as EXAMPLES ("e.g.") and prescribes an
    # increasing-order search with no stated finite upper bound; the
    # official code repository (zftang/MachineClassifer_BiasMitigation_beta)
    # uses an interleaved stream [3, 1/3, 5, 1/5, ..., 1999, 1/1999].
    # The Supplementary Materials (Algorithm 1) were not accessible for a
    # definitive bound, so this six-value grid is an implementation
    # budget: exhausting it is recorded as "candidate_grid_exhausted",
    # NOT as paper-level algorithmic non-convergence.
    transform_poly_exponents: tuple = (1 / 7, 1 / 5, 1 / 3, 3.0, 5.0, 7.0)

    # Failure semantics for the greedy mitigation search:
    #   "stop" (default): when the CURRENT highest-d_phi attribute's
    #       CONFIGURED candidate-grid search cannot reach the
    #       epsilon ball, the run records the failure (search_scope=
    #       "configured_grid") and terminates (the paper keeps operating
    #       on the highest attribute).  Note: this is "configured grid
    #       exhausted", not a paper-level non-convergence claim.
    #   "next" (explicitly named ENGINEERING extension): record the failure
    #       keyed by (protected attribute, feature) and try the next-ranked
    #       attribute instead.
    failed_attribute_mode: str = "stop"

    verbose: bool = False
    output_dir: str = "runs"

    def __post_init__(self) -> None:
        if self.algorithm_mode not in ALGORITHM_MODES:
            raise ValueError(
                f"algorithm_mode must be one of {ALGORITHM_MODES}, "
                f"got {self.algorithm_mode!r}"
            )
        if self.mds_fixed_components is not None and int(self.mds_fixed_components) < 1:
            raise ValueError(
                "mds_fixed_components must be a positive integer or None, "
                f"got {self.mds_fixed_components!r}"
            )

    def resolved(self) -> "FairBiasConfig":
        """Return the EFFECTIVE configuration with the mode concretized.

        ``official_code_derived_monotone_cursor_unweighted``
        (official-code-derived variant with a termination-safety
        extension) resolves to:
          - ``mds_fixed_components = 2`` (official fixed embedding dim,
            inherited from the official code — not the paper text's
            elbow-plot selection);
          - ``transform_poly_exponents`` = the official interleaved stream
            (order preserved; a custom non-official grid is rejected);
          - accuracy enhancement and the "next" failed-attribute mode are
            rejected (both are named engineering extensions and are
            incompatible with the official-code-derived mode).

        ``engineering_bounded`` resolves to itself unchanged.
        """
        if self.algorithm_mode == ALGORITHM_MODE_ENGINEERING:
            return self

        if self.algorithm_mode == ALGORITHM_MODE_PAPER_FAITHFUL:
            cfg = self
            if cfg.use_accuracy_enhancement:
                raise ValueError(
                    "use_accuracy_enhancement is not part of Tang et al. (2024) paper-faithful baseline "
                    "and cannot be combined with algorithm_mode='tang2024_paper_faithful'"
                )
            if cfg.failed_attribute_mode != "stop":
                raise ValueError(
                    "failed_attribute_mode='next' is a named ENGINEERING "
                    "extension and cannot be combined with algorithm_mode="
                    "'tang2024_paper_faithful'"
                )
            stream = official_power_stream()
            current_grid = tuple(float(p) for p in cfg.transform_poly_exponents)
            if current_grid != stream:
                default_grid = (1 / 7, 1 / 5, 1 / 3, 3.0, 5.0, 7.0)
                if tuple(sorted(current_grid)) == tuple(sorted(default_grid)):
                    cfg = dataclasses.replace(cfg, transform_poly_exponents=stream)
            # In paper-faithful mode, mds_fixed_components retains its configured value
            # (None for paper-described stress elbow plot selection, or int if explicitly specified).
            return cfg

        cfg = self
        if cfg.use_accuracy_enhancement:
            raise ValueError(
                "use_accuracy_enhancement is a named ENGINEERING extension "
                "and cannot be combined with algorithm_mode="
                "'official_code_derived_monotone_cursor_unweighted'"
            )
        if cfg.failed_attribute_mode != "stop":
            raise ValueError(
                "failed_attribute_mode='next' is a named ENGINEERING "
                "extension and cannot be combined with algorithm_mode="
                "'official_code_derived_monotone_cursor_unweighted'"
            )

        if cfg.mds_fixed_components is None:
            cfg = dataclasses.replace(
                cfg, mds_fixed_components=OFFICIAL_FIXED_MDS_DIM
            )
        elif int(cfg.mds_fixed_components) != OFFICIAL_FIXED_MDS_DIM:
            raise ValueError(
                "algorithm_mode='official_code_derived_monotone_cursor_"
                "unweighted' fixes the MDS embedding dimension at "
                f"{OFFICIAL_FIXED_MDS_DIM} "
                "(inherited official-code behavior, NOT the paper text's "
                "elbow-plot selection); got "
                f"mds_fixed_components={cfg.mds_fixed_components!r}. Use the "
                "engineering_bounded mode for other dimensions."
            )

        stream = official_power_stream()
        current_grid = tuple(float(p) for p in cfg.transform_poly_exponents)
        if current_grid != stream:
            default_grid = (1 / 7, 1 / 5, 1 / 3, 3.0, 5.0, 7.0)
            if tuple(sorted(current_grid)) == tuple(sorted(default_grid)):
                # Default six-value grid: silently concretize to the
                # official stream.
                cfg = dataclasses.replace(
                    cfg, transform_poly_exponents=stream
                )
            else:
                raise ValueError(
                    "algorithm_mode='official_code_derived_monotone_cursor_"
                    "unweighted' must use the official interleaved power "
                    "stream [3, 1/3, 5, 1/5, ..., 1999, 1/1999]; got a "
                    "custom grid. Use the engineering_bounded mode for "
                    "custom grids."
                )
        return cfg

    @classmethod
    def compas_default(cls, **overrides: Any) -> FairBiasConfig:
        """Standard preset for COMPAS dataset.

        ``mode`` is accepted as a convenience alias for ``algorithm_mode``
        ("official" -> official_code_derived_monotone_cursor_unweighted,
        "engineering" -> engineering_bounded).
        """
        params = {
            "dataset_name": "compas",
            "dataset_path": "data_COMPAS.csv",
            "label_Y": "two_year_recid",
            "label_O": ("sex",),
            "test_size": 0.20,
            "val_size": 0.16,
            "random_seed": 0,
            "classifier": "LR",
            "eval_norm": "min-max",
            "max_iterations": 5,
            "use_bias_mitigation": True,
            "use_accuracy_enhancement": False,
        }
        params.update(overrides)
        mode = params.pop("mode", None)
        if mode is not None:
            if mode == "official":
                params["algorithm_mode"] = ALGORITHM_MODE_OFFICIAL
            elif mode == "engineering":
                params["algorithm_mode"] = ALGORITHM_MODE_ENGINEERING
            elif mode in ("paper", "paper_faithful", "tang2024_paper_faithful"):
                params["algorithm_mode"] = ALGORITHM_MODE_PAPER_FAITHFUL
            else:
                raise ValueError(
                    "mode must be 'official', 'engineering', or 'paper', got "
                    f"{mode!r}"
                )
        if isinstance(params.get("label_O"), (list, set)):
            params["label_O"] = tuple(params["label_O"])
        return cls(**params)

    @classmethod
    def credit_default(cls, **overrides: Any) -> FairBiasConfig:
        """Standard preset for Taiwan Credit Card dataset.

        ``mode`` is accepted as a convenience alias for ``algorithm_mode``
        ("official" -> official_code_derived_monotone_cursor_unweighted,
        "engineering" -> engineering_bounded,
        "paper" -> tang2024_paper_faithful).
        """
        params = {
            "dataset_name": "credit",
            "dataset_path": "data_Credit_Card.csv",
            "label_Y": "default payment next month",
            "label_O": ("SEX",),
            "test_size": 0.20,
            "val_size": 0.16,
            "random_seed": 0,
            "classifier": "LR",
            "eval_norm": "min-max",
            "max_iterations": 5,
            "use_bias_mitigation": True,
            "use_accuracy_enhancement": False,
        }
        params.update(overrides)
        mode = params.pop("mode", None)
        if mode is not None:
            if mode == "official":
                params["algorithm_mode"] = ALGORITHM_MODE_OFFICIAL
            elif mode == "engineering":
                params["algorithm_mode"] = ALGORITHM_MODE_ENGINEERING
            elif mode in ("paper", "paper_faithful", "tang2024_paper_faithful"):
                params["algorithm_mode"] = ALGORITHM_MODE_PAPER_FAITHFUL
            else:
                raise ValueError(
                    "mode must be 'official', 'engineering', or 'paper', got "
                    f"{mode!r}"
                )
        if isinstance(params.get("label_O"), (list, set)):
            params["label_O"] = tuple(params["label_O"])
        return cls(**params)


if __name__ == "__main__":
    # Smoke: official mode resolution must produce the fixed dim and stream.
    cfg = FairBiasConfig.compas_default(mode="official").resolved()
    assert cfg.mds_fixed_components == OFFICIAL_FIXED_MDS_DIM
    assert cfg.transform_poly_exponents == official_power_stream()
    print("config smoke OK")
