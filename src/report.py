# src/report.py
#
# LaTeX table generation for the DIME paper.
#
# All public functions return a ready-to-paste LaTeX string.
# Print to stdout or redirect to a .tex file — no dependencies beyond pandas.
#
# ── Key concepts ──────────────────────────────────────────────────────────────
#
#   model      — key from config.MODEL_TO_HF     e.g. "contriever"
#   collection — key from config.COLLECTIONS     e.g. "dl19"
#   filter_tag — DimeFilter.tag                  e.g. "prf-k2", "oracular"
#   selector   — DimSelector.tag                 e.g. "top-alpha", "rdime"
#   alpha      — float (top-alpha) or "rdime" string (RDIME single-shot)
#
# ── Results files on disk ─────────────────────────────────────────────────────
#
#   Baseline:  data/results/{collection}/{model}.csv
#              columns: query_id, measure, value
#
#   DIME run:  data/results/{collection}/{model}__{filter}__{selector}.csv
#              columns: alpha, query_id, measure, value [, retained_frac]
#
# ── Four table types ──────────────────────────────────────────────────────────
#
#   performance  — fixed selector + fixed alpha, multiple measures as columns.
#                  Answers: "how good is method X?"
#
#   comparison   — Top-k alphas vs RDIME, single measure, delta(%) column.
#                  Answers: "does RDIME match the best fixed alpha?"
#
#   sweep        — alpha as column groups, one or more measures per row.
#                  Answers: "where does performance saturate?"
#
#   retained     — no measures, just mean retained dimension fractions.
#                  Answers: "how many dims does RDIME actually keep?"
#
# ── CLI usage ─────────────────────────────────────────────────────────────────
#
#   python src/report.py --table performance \
#       --models ance contriever tasb \
#       --collections dl19 dl20 dlhard \
#       --filters prf-k2 oracular \
#       --selector top-alpha --alpha 0.8 \
#       --measures nDCG@10 AP
#
#   python src/report.py --table comparison \
#       --models ance contriever tasb \
#       --collections dl19 dl20 dlhard \
#       --filters prf-k2 oracular \
#       --topk-alphas 0.4 0.6 0.8 \
#       --measure nDCG@10
#
#   python src/report.py --table sweep \
#       --models ance contriever tasb \
#       --collections dl19 dl20 \
#       --filter prf-k2 --selector top-alpha \
#       --alphas 0.2 0.4 0.6 0.8 1.0 \
#       --measures nDCG@10 AP
#
#   python src/report.py --table retained \
#       --models ance contriever tasb \
#       --collections dl19 dl20 dlhard \
#       --filters prf-k2 oracular
#
# Each command prints the LaTeX block to stdout.
# Redirect to a file:  python src/report.py --table performance ... > table.tex

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

import sys
sys.path.append('/hdd4/giuder/Documents/projects/dime')

from src.config import DATA_DIR


# ── Display name mappings — edit here when adding models / collections ─────────

MODEL_DISPLAY = {
    "ance":        "ANCE",
    "contriever":  "Contriever",
    "cocondenser": "CoCondenser",
    "tasb":        "TAS-B",
}

COLLECTION_DISPLAY = {
    "dl19":     "DL '19",
    "dl20":     "DL '20",
    "dlhard":   "DL HD",
    "robust04": "RB '04",
}

# LaTeX math names for filters — used as row labels
FILTER_DISPLAY = {
    "prf-k2":   r"$u^{PRF}$",
    "prf-k10":  r"$u^{PRF}$",
    "oracular": r"$u^{Oracle}$",
    "swc-k10":  r"$u^{SWC}$",
    "llm":      r"$u^{LLM}$",
}


# ── Data loading ───────────────────────────────────────────────────────────────

def _results_dir(collection: str) -> Path:
    return DATA_DIR / "results" / collection


def load_baseline_means(
    collection: str,
    model: str,
    measures: Sequence[str],
) -> dict[str, float | None]:
    """
    Load mean metric values for the full-dimensional baseline run.
    Returns a dict measure -> mean, with None for any missing measure.
    """
    path = _results_dir(collection) / f"{model}.csv"
    if not path.exists():
        return {m: None for m in measures}
    df = pd.read_csv(path, dtype={"query_id": str})
    return {
        m: float(df.loc[df["measure"] == m, "value"].mean())
        if not df.loc[df["measure"] == m].empty else None
        for m in measures
    }


def load_sweep_means(
    collection: str,
    model: str,
    filter_tag: str,
    selector: str,
    measures: Sequence[str],
) -> dict[str, dict[str | float, float]]:
    """
    Load mean metric values from a DIME sweep results CSV.

    Returns a nested dict:  measure -> { alpha -> mean_value }
    Alpha keys are floats for top-alpha sweeps, "rdime" for RDIME single-shot.
    Returns empty inner dicts for any missing file or measure.
    """
    path = _results_dir(collection) / f"{model}__{filter_tag}__{selector}.csv"
    if not path.exists():
        return {m: {} for m in measures}
    df = pd.read_csv(path, dtype={"query_id": str})
    return {
        m: df.loc[df["measure"] == m].groupby("alpha")["value"].mean().to_dict()
        for m in measures
    }


def load_retained_frac(
    collection: str,
    model: str,
    filter_tag: str,
    selector: str = "rdime",
) -> float | None:
    """
    Load the mean retained_frac across all queries for an RDIME run.
    Returns None if the column is absent or the file does not exist.
    """
    path = _results_dir(collection) / f"{model}__{filter_tag}__{selector}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"query_id": str})
    if "retained_frac" not in df.columns:
        return None
    return float(df["retained_frac"].dropna().mean())


# ── LaTeX primitives ───────────────────────────────────────────────────────────

def _bold(s: str) -> str:
    return r"\textbf{" + s + "}"


def _fmt(value: float | None, decimals: int = 3) -> str:
    """Format a float to fixed decimals; '--' for missing values."""
    if value is None:
        return "--"
    return f"{value:.{decimals}f}"


def _delta_pct(new: float | None, ref: float | None) -> str:
    """
    Relative change of `new` vs `ref` in percent, with sign.
    Positive means `new` is better than `ref`.
    """
    if new is None or ref is None or ref == 0:
        return "--"
    return f"{(new - ref) / ref * 100:+.2f}"


def _multicolumn(n: int, align: str, content: str) -> str:
    return rf"\multicolumn{{{n}}}{{{align}}}{{{content}}}"


def _multirow(n: int, content: str) -> str:
    return rf"\multirow{{{n}}}{{*}}{{{content}}}"


def _cmidrule(a: int, b: int) -> str:
    return rf"\cmidrule(lr){{{a}-{b}}}"


def _table_wrap(
    body_lines: list[str],
    col_spec: str,
    caption: str,
    label: str,
    wide: bool = False,
) -> str:
    """Wrap tabular body lines in a full table / table* environment."""
    env = "table*" if wide else "table"
    return "\n".join([
        rf"\begin{{{env}}}[t]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{{col_spec}}}",
        *body_lines,
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\end{{{env}}}",
    ])


def _best_indices(values: list[float | None]) -> set[int]:
    """Return the indices of the maximum value(s) in a list, ignoring Nones."""
    valid = [(v, i) for i, v in enumerate(values) if v is not None]
    if not valid:
        return set()
    best = max(v for v, _ in valid)
    return {i for v, i in valid if abs(v - best) < 1e-9}


# ── Table: performance ─────────────────────────────────────────────────────────

def table_performance(
    models: Sequence[str],
    collections: Sequence[str],
    filter_tags: Sequence[str],
    selector: str,
    alpha: float | str,
    measures: Sequence[str] = ("nDCG@10", "AP", "R@1000", "RR@10"),
    include_baseline: bool = True,
    bold_best: bool = True,
    caption: str | None = None,
    label: str | None = None,
) -> str:
    """
    Performance table: fixed selector and alpha, multiple measures as columns.

    Shows how good a given (filter, selector, alpha) configuration is across
    all models and collections. Multiple measures are shown side by side so
    the reader can assess the method across different evaluation criteria at
    a glance.

    Layout:
      - Row groups:    one per model
      - Sub-rows:      one per filter, plus an optional Baseline row
      - Column groups: one per collection
      - Columns:       one per measure, repeated across collections

    Args:
        models:           model keys to include
        collections:      collection keys to include
        filter_tags:      filter tags to include as sub-rows
        selector:         selector tag ("top-alpha" or "rdime")
        alpha:            alpha value (float) or "rdime" for RDIME single-shot
        measures:         IR measures to show as columns
        include_baseline: add a full-dim baseline sub-row per model
        bold_best:        bold the best value per (collection, measure) column
        caption / label:  LaTeX caption and label strings
    """
    n_meas    = len(measures)
    n_coll    = len(collections)
    n_subrows = len(filter_tags) + (1 if include_baseline else 0)

    # column spec: ll | (c...c) per collection
    col_spec = "ll" + "".join(["|" + "c" * n_meas] * n_coll)

    # ── pre-load all data ──────────────────────────────────────────────────────
    # data[collection][model][filter_tag][measure] = mean_value | None
    data: dict = {}
    for collection in collections:
        data[collection] = {}
        for model in models:
            data[collection][model] = {}
            for ft in filter_tags:
                sweep = load_sweep_means(collection, model, ft, selector, measures)
                data[collection][model][ft] = {m: sweep[m].get(alpha) for m in measures}
            if include_baseline:
                data[collection][model]["__baseline__"] = load_baseline_means(
                    collection, model, measures
                )

    # ── find best value per (collection, measure) across all rows ──────────────
    best: dict[tuple, float] = {}
    if bold_best:
        all_row_keys = list(filter_tags) + (["__baseline__"] if include_baseline else [])
        for collection in collections:
            for m in measures:
                vals = [
                    data[collection][model][rk][m]
                    for model in models
                    for rk in all_row_keys
                    if data[collection][model][rk].get(m) is not None
                ]
                if vals:
                    best[(collection, m)] = max(vals)

    def _cell(val: float | None, collection: str, measure: str) -> str:
        s = _fmt(val)
        if (
            bold_best
            and val is not None
            and (collection, measure) in best
            and abs(val - best[(collection, measure)]) < 1e-9
        ):
            s = _bold(s)
        return s

    # ── build lines ───────────────────────────────────────────────────────────
    lines: list[str] = []
    lines.append(r"\toprule")

    # header row 1 — collection names
    coll_headers = " & ".join(
        _multicolumn(n_meas, "c", COLLECTION_DISPLAY.get(c, c))
        for c in collections
    )
    lines.append(rf"Model & Filter & {coll_headers} \\")

    # header row 2 — measure names repeated per collection, with cmidrules
    cmidrules = " ".join(
        _cmidrule(3 + i * n_meas, 2 + (i + 1) * n_meas)
        for i in range(n_coll)
    )
    lines.append(cmidrules)
    meas_header = " & ".join(list(measures) * n_coll)
    lines.append(rf" & & {meas_header} \\")
    lines.append(r"\midrule")

    # body — one row-group per model
    for m_idx, model in enumerate(models):
        model_display = MODEL_DISPLAY.get(model, model)

        for f_idx, ft in enumerate(filter_tags):
            filter_display = FILTER_DISPLAY.get(ft, ft)
            model_cell = _multirow(n_subrows, model_display) if f_idx == 0 else ""

            row_cells = [model_cell, filter_display]
            for collection in collections:
                for measure in measures:
                    row_cells.append(_cell(data[collection][model][ft].get(measure), collection, measure))
            lines.append(" & ".join(row_cells) + r" \\")

        if include_baseline:
            bl_cells = ["", "Baseline"]
            for collection in collections:
                for measure in measures:
                    bl_cells.append(_cell(data[collection][model]["__baseline__"].get(measure), collection, measure))
            lines.append(" & ".join(bl_cells) + r" \\")

        if m_idx < len(models) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")

    alpha_str = str(alpha)
    cap = caption or (
        rf"Retrieval performance ({', '.join(measures)}) at $\alpha$={alpha_str}, "
        rf"selector={selector}."
    )
    lbl   = label or "tab:performance"
    wide  = n_coll * n_meas > 6

    return _table_wrap(lines, col_spec, cap, lbl, wide=wide)


# ── Table: comparison ─────────────────────────────────────────────────────────

def table_comparison(
    models: Sequence[str],
    collections: Sequence[str],
    filter_tags: Sequence[str],
    measure: str = "nDCG@10",
    topk_alphas: Sequence[float] = (0.4, 0.6, 0.8),
    selector_sweep: str = "top-alpha",
    selector_rdime: str = "rdime",
    include_baseline: bool = True,
    caption: str | None = None,
    label: str | None = None,
) -> str:
    """
    Comparison table: Top-k fixed alphas vs RDIME, with delta(%) column.

    Shows whether RDIME's adaptive threshold matches or beats the best
    fixed-alpha configuration without needing to tune alpha at all.
    The delta(%) column is RDIME relative to the best Top-k alpha shown.
    The RDIME cell includes the mean retained fraction in parentheses.

    Layout:
      - Row groups:    one per model
      - Sub-rows:      one per filter, plus an optional Baseline row
      - Column groups: one per collection
      - Columns:       [alpha_1 ... alpha_n | RDIME (frac) | delta(%)]

    Args:
        models:          model keys to include
        collections:     collection keys to include
        filter_tags:     filter tags to include as sub-rows
        measure:         single IR measure to display (e.g. "nDCG@10")
        topk_alphas:     fixed alpha values shown as Top-k columns
        selector_sweep:  selector tag for top-alpha sweep
        selector_rdime:  selector tag for RDIME
        include_baseline: add a full-dim baseline sub-row per model
        caption / label: LaTeX caption and label strings
    """
    n_topk     = len(topk_alphas)
    n_cols_per = n_topk + 2          # Top-k cols + RDIME col + delta(%) col
    n_coll     = len(collections)
    n_subrows  = len(filter_tags) + (1 if include_baseline else 0)

    col_spec = "ll" + "".join(["|" + "c" * n_cols_per] * n_coll)

    # ── pre-load ───────────────────────────────────────────────────────────────
    topk_data:  dict = {}   # [collection][model][filter] -> {alpha: value}
    rdime_data: dict = {}   # [collection][model][filter] -> value | None
    rdime_frac: dict = {}   # [collection][model][filter] -> frac  | None
    bl_data:    dict = {}   # [collection][model]         -> value | None

    for collection in collections:
        topk_data[collection]  = {}
        rdime_data[collection] = {}
        rdime_frac[collection] = {}
        bl_data[collection]    = {}
        for model in models:
            topk_data[collection][model]  = {}
            rdime_data[collection][model] = {}
            rdime_frac[collection][model] = {}
            for ft in filter_tags:
                sweep = load_sweep_means(collection, model, ft, selector_sweep, [measure])
                topk_data[collection][model][ft] = {a: sweep[measure].get(a) for a in topk_alphas}

                rdime_sweep = load_sweep_means(collection, model, ft, selector_rdime, [measure])
                rdime_data[collection][model][ft] = rdime_sweep[measure].get("rdime")
                rdime_frac[collection][model][ft] = load_retained_frac(
                    collection, model, ft, selector_rdime
                )
            bl_data[collection][model] = load_baseline_means(collection, model, [measure])[measure]

    # ── build lines ───────────────────────────────────────────────────────────
    lines: list[str] = []
    lines.append(r"\toprule")

    # header row 1 — collection names
    coll_headers = " & ".join(
        _multicolumn(n_cols_per, "c", COLLECTION_DISPLAY.get(c, c))
        for c in collections
    )
    lines.append(rf"Model & Filter & {coll_headers} \\")

    # header row 2 — alpha columns + RDIME + delta(%)
    cmidrules = " ".join(
        _cmidrule(3 + i * n_cols_per, 2 + (i + 1) * n_cols_per)
        for i in range(n_coll)
    )
    lines.append(cmidrules)
    per_coll_header = (
        " & ".join(str(a) for a in topk_alphas)
        + r" & RDIME & $\Delta$(\%)"
    )
    lines.append(rf" & & {' & '.join([per_coll_header] * n_coll)} \\")
    lines.append(r"\midrule")

    # body
    for m_idx, model in enumerate(models):
        model_display = MODEL_DISPLAY.get(model, model)

        for f_idx, ft in enumerate(filter_tags):
            filter_display = FILTER_DISPLAY.get(ft, ft)
            model_cell = _multirow(n_subrows, model_display) if f_idx == 0 else ""

            row_cells = [model_cell, filter_display]
            for collection in collections:
                # top-k cells — bold the best among shown alphas
                topk_vals    = [topk_data[collection][model][ft].get(a) for a in topk_alphas]
                best_indices = _best_indices(topk_vals)
                for i, v in enumerate(topk_vals):
                    s = _fmt(v)
                    row_cells.append(_bold(s) if i in best_indices else s)

                # RDIME cell — value + superscript star + (retained frac)
                rdime_val = rdime_data[collection][model][ft]
                frac      = rdime_frac[collection][model][ft]
                if rdime_val is not None:
                    frac_str   = f"({frac:.2f})" if frac is not None else ""
                    rdime_cell = rf"{_fmt(rdime_val)}$^\star$ {frac_str}"
                else:
                    rdime_cell = "--"
                row_cells.append(rdime_cell)

                # delta(%) — RDIME vs best top-k shown
                best_topk = max((v for v in topk_vals if v is not None), default=None)
                row_cells.append(_delta_pct(rdime_val, best_topk))

            lines.append(" & ".join(row_cells) + r" \\")

        if include_baseline:
            bl_cells = ["", "Baseline"]
            for collection in collections:
                bl_val = bl_data[collection][model]
                bl_cells += ["--"] * n_topk
                frac_str = "(1.00)"
                bl_cells.append(f"{_fmt(bl_val)} {frac_str}" if bl_val is not None else "--")
                bl_cells.append("--")
            lines.append(" & ".join(bl_cells) + r" \\")

        if m_idx < len(models) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")

    cap = caption or (
        rf"Comparison of {measure} between RDIME and fixed Top-$k$ thresholds. "
        r"$\star$ denotes RDIME; parentheses show mean fraction of retained dimensions. "
        r"$\Delta$(\%) is relative to the best Top-$k$ column shown."
    )
    lbl = label or "tab:comparison"

    return _table_wrap(lines, col_spec, cap, lbl, wide=True)


# ── Table: sweep ──────────────────────────────────────────────────────────────

def table_sweep(
    models: Sequence[str],
    collections: Sequence[str],
    filter_tag: str,
    selector: str = "top-alpha",
    alphas: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    measures: Sequence[str] = ("nDCG@10", "AP"),
    include_baseline: bool = True,
    caption: str | None = None,
    label: str | None = None,
) -> str:
    """
    Sweep table: alpha values as column groups, measures as sub-columns.

    Shows how retrieval performance evolves as more embedding dimensions are
    activated. Reveals where performance saturates and whether different
    measures agree on the optimal alpha.

    Layout:
      - Row groups:    one per model
      - Sub-rows:      one per collection
      - Column groups: one per alpha value (+ optional Baseline group)
      - Columns:       one per measure within each alpha group
      - Bold:          best alpha per (model, collection, measure)

    Args:
        models:           model keys to include
        collections:      collection keys to include
        filter_tag:       single filter tag (one sweep at a time)
        selector:         selector tag (typically "top-alpha")
        alphas:           alpha values to show as column groups
        measures:         IR measures shown within each alpha group
        include_baseline: add a full-dim baseline column group at the right
        caption / label:  LaTeX caption and label strings
    """
    n_meas  = len(measures)
    n_alpha = len(alphas)
    n_extra = n_meas if include_baseline else 0

    # column spec: ll | (c...c per alpha) | (c...c baseline)
    col_spec = (
        "ll"
        + "|" + "c" * (n_alpha * n_meas)
        + ("|" + "c" * n_extra if include_baseline else "")
    )

    # ── pre-load ───────────────────────────────────────────────────────────────
    data: dict = {}   # [collection][model][measure][alpha] = mean_value
    bl:   dict = {}   # [collection][model][measure]        = mean_value
    for collection in collections:
        data[collection] = {}
        bl[collection]   = {}
        for model in models:
            sweep = load_sweep_means(collection, model, filter_tag, selector, measures)
            data[collection][model] = sweep
            bl[collection][model]   = load_baseline_means(collection, model, measures)

    # ── build lines ───────────────────────────────────────────────────────────
    lines: list[str] = []
    lines.append(r"\toprule")

    # header row 1 — alpha group labels + optional Baseline
    alpha_headers = " & ".join(
        _multicolumn(n_meas, "c", str(a)) for a in alphas
    )
    bl_header = " & " + _multicolumn(n_meas, "c", "Baseline") if include_baseline else ""
    lines.append(rf"Model & Collection & {alpha_headers}{bl_header} \\")

    # cmidrules under each alpha group (and baseline)
    cmidrule_parts = [_cmidrule(3 + i * n_meas, 2 + (i + 1) * n_meas) for i in range(n_alpha)]
    if include_baseline:
        start = 3 + n_alpha * n_meas
        cmidrule_parts.append(_cmidrule(start, start + n_meas - 1))
    lines.append(" ".join(cmidrule_parts))

    # header row 2 — measure names repeated per alpha group (and baseline)
    n_groups  = n_alpha + (1 if include_baseline else 0)
    meas_str  = " & ".join(list(measures) * n_groups)
    lines.append(rf" & & {meas_str} \\")
    lines.append(r"\midrule")

    # body
    for m_idx, model in enumerate(models):
        model_display = MODEL_DISPLAY.get(model, model)
        n_coll        = len(collections)

        for c_idx, collection in enumerate(collections):
            coll_display = COLLECTION_DISPLAY.get(collection, collection)
            model_cell   = _multirow(n_coll, model_display) if c_idx == 0 else ""

            # precompute best alpha index per measure for this row
            best_per_measure = {
                m: _best_indices([data[collection][model][m].get(a) for a in alphas])
                for m in measures
            }

            row_cells = [model_cell, coll_display]
            for a_idx, a in enumerate(alphas):
                for measure in measures:
                    v = data[collection][model][measure].get(a)
                    s = _fmt(v)
                    row_cells.append(_bold(s) if a_idx in best_per_measure[measure] else s)

            if include_baseline:
                for measure in measures:
                    row_cells.append(_fmt(bl[collection][model].get(measure)))

            lines.append(" & ".join(row_cells) + r" \\")

        if m_idx < len(models) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")

    filter_display = FILTER_DISPLAY.get(filter_tag, filter_tag)
    cap = caption or (
        rf"Alpha sweep for {filter_display} with {selector} selector "
        rf"({', '.join(measures)}). Bold: best $\alpha$ per row."
    )
    lbl  = label or "tab:sweep"
    wide = n_alpha * n_meas > 8

    return _table_wrap(lines, col_spec, cap, lbl, wide=wide)


# ── Table: retained ───────────────────────────────────────────────────────────

def table_retained(
    models: Sequence[str],
    collections: Sequence[str],
    filter_tags: Sequence[str],
    selector: str = "rdime",
    caption: str | None = None,
    label: str | None = None,
) -> str:
    """
    Retained dimensions table: mean fraction of dims kept by RDIME per query.

    Shows how aggressively RDIME prunes dimensions across models, filters and
    collections. Complements the boxplot figure with exact means.
    Lower values indicate that RDIME concentrates signal in fewer dimensions.

    Layout:
      - Row groups:  one per model
      - Sub-rows:    one per filter
      - Columns:     one per collection

    Args:
        models:          model keys to include
        collections:     collection keys to include
        filter_tags:     filter tags to include as sub-rows
        selector:        selector tag (should be "rdime")
        caption / label: LaTeX caption and label strings
    """
    col_spec = "ll" + "c" * len(collections)

    lines: list[str] = []
    lines.append(r"\toprule")

    coll_cols = " & ".join(COLLECTION_DISPLAY.get(c, c) for c in collections)
    lines.append(rf"Model & Filter & {coll_cols} \\")
    lines.append(r"\midrule")

    for m_idx, model in enumerate(models):
        model_display = MODEL_DISPLAY.get(model, model)
        n_subrows     = len(filter_tags)

        for f_idx, ft in enumerate(filter_tags):
            filter_display = FILTER_DISPLAY.get(ft, ft)
            model_cell     = _multirow(n_subrows, model_display) if f_idx == 0 else ""

            frac_cells = [
                _fmt(load_retained_frac(c, model, ft, selector), decimals=2)
                for c in collections
            ]
            lines.append(f"{model_cell} & {filter_display} & " + " & ".join(frac_cells) + r" \\")

        if m_idx < len(models) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")

    cap = caption or (
        r"Mean fraction of embedding dimensions retained by RDIME per query "
        r"(averaged over all queries in each collection)."
    )
    lbl = label or "tab:retained"

    return _table_wrap(lines, col_spec, cap, lbl, wide=False)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables for the DIME paper.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--table", required=True,
        choices=["performance", "comparison", "sweep", "retained"],
        help=(
            "performance  -> fixed selector + alpha, multiple measures as columns\n"
            "comparison   -> Top-k alphas vs RDIME, single measure + delta(%%)\n"
            "sweep        -> alpha as column groups, one or more measures\n"
            "retained     -> mean retained dimension fractions under RDIME\n"
        ),
    )

    # shared
    parser.add_argument("--models",      nargs="+", default=["ance", "contriever", "tasb"],
                        help="Model keys  (default: ance contriever tasb)")
    parser.add_argument("--collections", nargs="+", default=["dl19", "dl20", "dlhard"],
                        help="Collection keys  (default: dl19 dl20 dlhard)")
    parser.add_argument("--caption",     default=None, help="Override LaTeX caption")
    parser.add_argument("--label",       default=None, help="Override LaTeX label")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Omit the full-dim baseline row / column group")

    # performance + comparison + retained share --filters
    parser.add_argument("--filters",  nargs="+", default=["prf-k2", "oracular"],
                        help="[performance/comparison/retained] Filter tags")

    # performance
    parser.add_argument("--selector", default="top-alpha",
                        help="[performance/sweep] Selector tag  (default: top-alpha)")
    parser.add_argument("--alpha",    default="0.8",
                        help="[performance] Alpha value or 'rdime'  (default: 0.8)")
    parser.add_argument("--measures", nargs="+", default=["nDCG@10", "AP"],
                        help="[performance/sweep] IR measures  (default: nDCG@10 AP)")

    # comparison
    parser.add_argument("--measure",        default="nDCG@10",
                        help="[comparison] Single IR measure  (default: nDCG@10)")
    parser.add_argument("--topk-alphas",    nargs="+", type=float, default=[0.4, 0.6, 0.8],
                        help="[comparison] Fixed alpha values shown as Top-k columns")
    parser.add_argument("--selector-sweep", default="top-alpha",
                        help="[comparison] Selector for the Top-k sweep  (default: top-alpha)")
    parser.add_argument("--selector-rdime", default="rdime",
                        help="[comparison] Selector for RDIME  (default: rdime)")

    # sweep
    parser.add_argument("--filter",  default="prf-k2",
                        help="[sweep] Single filter tag  (default: prf-k2)")
    parser.add_argument("--alphas",  nargs="+", type=float,
                        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        help="[sweep] Alpha values shown as column groups")

    args = parser.parse_args()

    include_baseline = not args.no_baseline

    # parse --alpha as float if possible, keep as string otherwise (e.g. "rdime")
    try:
        alpha_val: float | str = float(args.alpha)
    except ValueError:
        alpha_val = args.alpha

    if args.table == "performance":
        tex = table_performance(
            models=args.models,
            collections=args.collections,
            filter_tags=args.filters,
            selector=args.selector,
            alpha=alpha_val,
            measures=args.measures,
            include_baseline=include_baseline,
            caption=args.caption,
            label=args.label,
        )

    elif args.table == "comparison":
        tex = table_comparison(
            models=args.models,
            collections=args.collections,
            filter_tags=args.filters,
            measure=args.measure,
            topk_alphas=args.topk_alphas,
            selector_sweep=args.selector_sweep,
            selector_rdime=args.selector_rdime,
            include_baseline=include_baseline,
            caption=args.caption,
            label=args.label,
        )

    elif args.table == "sweep":
        tex = table_sweep(
            models=args.models,
            collections=args.collections,
            filter_tag=args.filter,
            selector=args.selector,
            alphas=args.alphas,
            measures=args.measures,
            include_baseline=include_baseline,
            caption=args.caption,
            label=args.label,
        )

    elif args.table == "retained":
        tex = table_retained(
            models=args.models,
            collections=args.collections,
            filter_tags=args.filters,
            caption=args.caption,
            label=args.label,
        )

    print(tex)