"""Make one Kepler KOI catalog from Lissauer+2024 and the DR22, DR24, DR25 tables.

The KOIs and values are those of Lissauer+2024 (Table 1), converted to DR
columns (LISSAUER_TO_DR). The ephemeris of a KOI in ALIASES replaces the
Lissauer+2024 ephemeris.

The DR tables are used only for checks: each DR entry is compared with the
Lissauer+2024 entry, and inconsistencies are written to notes/kepler.md.

Added column:
    alderaan_disp  FP = false positive, PC = planet candidate (transit to
                 model; includes Lissauer+2024 M, mass > 13 M_Jup), A =
                 ambiguous. From the Lissauer+2024 disposition (F -> FP,
                 P or M -> PC, S or R -> A).
Differences from the DR25 supplemental disposition are written to
notes/kepler.md.

Usage (from the repo root):
    python scripts/merge_kepler.py
"""

from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CATALOG_DIR = ROOT / "reference" / "kepler"  # published tables (inputs)
ALDERAAN_DIR = ROOT / "reference" / "alderaan"  # tables made in this repo
OUTPUT = ALDERAAN_DIR / "kepler_merged_alderaan.csv"
REPORT = ROOT / "notes" / "kepler.md"
# text from this line to the end of REPORT is written by hand and kept on rerun
MANUAL_MARKER = "<!-- hand-written notes below; kept by scripts/merge_kepler.py -->"

# most recent first
CATALOGS = {
    "dr25": "kepler_koi_dr25_thompson_2018.csv",
    "dr24": "kepler_koi_dr24_coughlin_2016.csv",
    "dr22": "kepler_koi_dr22_mullally_2015.csv",
}

# A difference is an inconsistency only if it exceeds BOTH the fractional
# limit and N_SIGMA combined uncertainties (fractional limit only if an
# uncertainty is missing). Depth and duration limits are wide because
# pipeline changes between data releases routinely shift them by tens of percent.
N_SIGMA = 5.0
FRAC_LIMIT = {"koi_period": 1e-3, "koi_depth": 0.5, "koi_duration": 0.5}
EPOCH_LIMIT = 0.5  # epoch offset, in units of transit duration
ALIAS_RATIOS = [2, 3, 4, 5, 1 / 2, 1 / 3, 1 / 4, 1 / 5]
ALIAS_TOL = 0.01  # fractional tolerance on the period ratio

# Lissauer+2024 Table 1 (see notes/kepler.md)
LISSAUER = "kepler_candidates_lissauer_2024.txt"
# Lissauer+2024 Table 1 field (first and last byte, from its ReadMe) for each DR
# column; "-" makes the positive lower uncertainty a negative err2
LISSAUER_TO_DR = {
    "kepid": (1, 8, ""),
    "kepler_name": (18, 31, ""),
    "koi_period": (33, 46, ""),  # negative: mono-transit estimate (note 2); abs taken
    "koi_period_err1": (48, 57, ""),
    "koi_period_err2": (48, 57, "-"),
    "koi_time0bk": (59, 69, ""),  # BJD - 2454900; converted below
    "koi_time0bk_err1": (71, 78, ""),
    "koi_time0bk_err2": (71, 78, "-"),
    "koi_ror": (80, 88, ""),
    "koi_ror_err1": (90, 98, ""),
    "koi_ror_err2": (100, 108, "-"),
    "koi_impact": (110, 116, ""),
    "koi_impact_err1": (118, 124, ""),
    "koi_impact_err2": (126, 132, "-"),
    "koi_srho": (134, 143, ""),
    "koi_srho_err1": (145, 153, ""),
    "koi_srho_err2": (155, 163, "-"),
    "koi_ldm_coeff1": (165, 170, ""),
    "koi_ldm_coeff2": (172, 178, ""),
    "koi_num_transits": (185, 188, ""),
    "koi_depth": (195, 202, ""),
    "koi_depth_err1": (204, 211, ""),
    "koi_depth_err2": (204, 211, "-"),
    "koi_duration": (213, 222, ""),
    "koi_duration_err1": (224, 232, ""),
    "koi_duration_err2": (224, 232, "-"),
    "koi_model_snr": (255, 260, ""),
    "koi_dor": (299, 307, ""),
    "koi_dor_err1": (309, 317, ""),
    "koi_dor_err2": (319, 327, "-"),
    "koi_incl": (329, 335, ""),
    "koi_incl_err1": (337, 343, ""),
    "koi_incl_err2": (345, 351, "-"),
    "koi_prad": (353, 374, ""),
    "koi_prad_err1": (376, 385, ""),
    "koi_prad_err2": (387, 396, "-"),
    "koi_insol": (398, 408, ""),
    "koi_insol_err1": (410, 418, ""),
    "koi_insol_err2": (420, 428, "-"),
    "koi_kepmag": (430, 442, ""),
    "koi_steff": (469, 473, ""),
    "koi_steff_err1": (475, 478, ""),
    "koi_steff_err2": (475, 478, "-"),
    "koi_srad": (480, 485, ""),
    "koi_srad_err1": (487, 491, ""),
    "koi_srad_err2": (493, 498, "-"),
    "koi_smass": (500, 503, ""),
    "koi_smass_err1": (505, 508, ""),
    "koi_smass_err2": (510, 513, "-"),
    "koi_slogg": (515, 519, ""),
    "koi_slogg_err1": (521, 525, ""),
    "koi_slogg_err2": (527, 531, "-"),
    "koi_smet": (533, 538, ""),
    "koi_smet_err1": (540, 544, ""),
    "koi_smet_err2": (540, 544, "-"),
}
LISSAUER_T0_OFFSET = 67  # Lissauer T0 (BJD - 2454900) -> koi_time0bk (BJD - 2454833)
# ephemerides that override Lissauer+2024 (period aliases), with sources
ALIASES = ALDERAAN_DIR / "kepler_period_aliases_alderaan.csv"
# Lissauer+2024 status letter -> alderaan_disp
LISSAUER_DISP = {"F": "FP", "M": "PC", "P": "PC", "S": "A", "R": "A"}


def load_catalogs():
    """Return {source: table indexed by kepoi_name} and the common column order."""
    tables, header = {}, None
    for source, filename in CATALOGS.items():
        # read as text so that the merged rows are verbatim copies of the source rows
        df = pd.read_csv(CATALOG_DIR / filename, dtype=str, keep_default_na=False)
        if header is None:
            header = list(df.columns)
        if list(df.columns) != header:
            raise ValueError(f"{filename}: columns differ from {CATALOGS['dr25']}")
        if df["kepoi_name"].duplicated().any():
            raise ValueError(f"{filename}: duplicate kepoi_name")
        tables[source] = df.set_index("kepoi_name")
    return tables, header


def load_lissauer(header):
    """Lissauer+2024 Table 1 in DR columns, and the status flags (note 14)."""
    rows, status = {}, {}
    with open(CATALOG_DIR / LISSAUER) as f:
        for line in f:
            if not line[:8].strip().isdigit():
                continue
            k = koi_name(line[9:16])
            if k in rows:
                raise ValueError(f"{LISSAUER}: duplicate KOI {k}")
            row = {}
            for col, (first, last, sign) in LISSAUER_TO_DR.items():
                text = line[first - 1 : last].strip()
                if sign and text:
                    if text.startswith("-"):
                        raise ValueError(f"{LISSAUER}: {k}: negative lower error")
                    text = sign + text
                row[col] = text
            t0 = Decimal(row["koi_time0bk"]) + LISSAUER_T0_OFFSET
            row["koi_time0bk"] = str(t0)
            # negative period: mono-transit estimate; ALDERAAN needs a positive one
            row["koi_period"] = row["koi_period"].removeprefix("-")
            rows[k] = row
            status[k] = line[547:551]
    columns = [c for c in header if c != "kepoi_name"]
    missing = set(LISSAUER_TO_DR) - set(columns)
    if missing:
        raise ValueError(f"DR columns missing: {sorted(missing)}")
    table = pd.DataFrame.from_dict(rows, orient="index", columns=columns).fillna("")
    return table, pd.Series(status)


def merge(lissauer):
    """Lissauer+2024 values, with the ephemerides in ALIASES."""
    overrides = pd.read_csv(ALIASES, dtype=str, keep_default_na=False)
    overrides = overrides.set_index("kepoi_name").drop(columns="comment")
    if overrides.index.duplicated().any():
        raise ValueError(f"{ALIASES.name}: duplicate KOI")
    missing = overrides.index.difference(lissauer.index)
    if len(missing):
        raise ValueError(f"{ALIASES.name}: KOIs not in {LISSAUER}: {list(missing)}")
    merged = lissauer.copy()
    merged.loc[overrides.index, overrides.columns] = overrides
    # every KOI needs an ephemeris, depth, and duration
    needed = ["koi_period", "koi_time0bk", "koi_depth", "koi_duration"]
    empty = merged.index[(merged[needed] == "").any(axis=1)]
    if len(empty):
        raise ValueError(f"missing ephemeris, depth, or duration: {list(empty)}")
    return merged.sort_index()


def koi_name(number):
    """'1.01' -> 'K00001.01'"""
    return f"K{float(number):08.2f}"


def designate(merged, status):
    """Add alderaan_disp; return the DR25supp changes counts."""
    status = status.reindex(merged.index)
    supp, lissauer = status.str[1], status.str[0]
    if not lissauer.isin(LISSAUER_DISP).all():
        raise ValueError(f"{LISSAUER}: unexpected status letter")
    merged["alderaan_disp"] = lissauer.map(LISSAUER_DISP)

    # (DR25supp, Lissauer) -> number of KOIs, for KOIs where they differ
    changes = pd.Series(list(zip(supp, lissauer)))[(supp != lissauer).values]
    return changes.value_counts().sort_index()


def numeric(df):
    """Compared columns and their uncertainties as floats (empty -> NaN)."""
    cols = [
        c + s
        for c in ["koi_period", "koi_time0bk", "koi_depth", "koi_duration"]
        for s in ["", "_err1", "_err2"]
    ]
    return df[cols].replace("", np.nan).astype(float)


def sigma(df, col):
    """Symmetric 1-sigma uncertainty: mean of |err1| and |err2|."""
    return 0.5 * (df[col + "_err1"].abs() + df[col + "_err2"].abs())


def exceeds(frac, nsig, frac_limit):
    return (frac > frac_limit) & ((nsig > N_SIGMA) | nsig.isna())


def compare(tables, lissauer):
    """Compare each DR entry of a KOI with its Lissauer+2024 entry."""
    kepid_rows, alias_rows, param_rows = [], [], []
    for source, old in tables.items():
        names = old.index.intersection(lissauer.index)
        new = lissauer.loc[names]

        bad = old.loc[names, "kepid"] != new.loc[names, "kepid"]
        for k in names[bad]:
            kepid_rows.append((k, source, old.at[k, "kepid"], new.at[k, "kepid"]))

        a, b = numeric(old.loc[names]), numeric(new.loc[names])

        ratio = a["koi_period"] / b["koi_period"]
        alias = np.zeros(len(names), dtype=bool)
        for r in ALIAS_RATIOS:
            alias |= (ratio / r - 1).abs().to_numpy() < ALIAS_TOL
        for k in names[alias]:
            alias_rows.append(
                (
                    k,
                    source,
                    a.at[k, "koi_period"],
                    b.at[k, "koi_period"],
                    ratio[k],
                )
            )

        # skip aliases: epoch, depth, and duration are not comparable
        a, b = a[~alias], b[~alias]
        for col in FRAC_LIMIT:
            diff = (a[col] - b[col]).abs()
            frac = diff / b[col].abs()
            nsig = diff / np.hypot(sigma(a, col), sigma(b, col))
            flag = exceeds(frac, nsig, FRAC_LIMIT[col])
            for k in a.index[flag]:
                param_rows.append(
                    (
                        k,
                        source,
                        col,
                        a.at[k, col],
                        b.at[k, col],
                        frac[k],
                        nsig[k],
                    )
                )

        # project the DR epoch to the transit nearest the Lissauer+2024 epoch
        period = b["koi_period"]
        n = np.round((b["koi_time0bk"] - a["koi_time0bk"]) / period)
        dt = (a["koi_time0bk"] + n * period - b["koi_time0bk"]).abs()
        frac = dt / (b["koi_duration"] / 24.0)
        nsig = dt / np.sqrt(
            sigma(a, "koi_time0bk") ** 2
            + sigma(b, "koi_time0bk") ** 2
            + (n * sigma(b, "koi_period")) ** 2
        )
        flag = exceeds(frac, nsig, EPOCH_LIMIT)
        for k in a.index[flag]:
            param_rows.append(
                (
                    k,
                    source,
                    "koi_time0bk",
                    a.at[k, "koi_time0bk"],
                    b.at[k, "koi_time0bk"],
                    frac[k],
                    nsig[k],
                )
            )

    return kepid_rows, sorted(alias_rows), sorted(param_rows)


def fmt(x):
    return "" if pd.isna(x) else f"{x:.6g}"


def write_report(tables, lissauer, merged, kepid_rows, alias_rows, param_rows, changes):
    n_multi = sum(
        c > 1
        for c in pd.concat(
            [t.index.to_series() for t in tables.values()]
        ).value_counts()
    )
    lines = [
        "# Kepler KOI catalog merge",
        "",
        "Generated by `scripts/merge_kepler.py`; do not edit above the",
        "hand-written notes marker.",
        "",
        "## Summary",
        "",
        "- Input: Lissauer+2024 "
        + f"{len(lissauer)}, "
        + ", ".join(f"{s.upper()} {len(t)}" for s, t in tables.items()),
        f"- Output: `{OUTPUT.relative_to(ROOT)}`, {len(merged)} KOIs; values from",
        f"  Lissauer+2024, ephemerides overridden by `{ALIASES.relative_to(ROOT)}`",
        f"- KOIs in more than one DR catalog: {n_multi}",
        "",
        "Each DR entry of a KOI is compared with its Lissauer+2024 entry. A",
        "difference is reported only if it exceeds both a fractional",
        f"limit and {N_SIGMA:g} combined sigma (sigma = mean of |err1| and |err2|;",
        "fractional limit only if an uncertainty is missing).",
        "",
        f"- period: {FRAC_LIMIT['koi_period']:g} (fractional)",
        f"- depth: {FRAC_LIMIT['koi_depth']:g} (fractional)",
        f"- duration: {FRAC_LIMIT['koi_duration']:g} (fractional)",
        f"- epoch: {EPOCH_LIMIT:g} transit durations, after projection to the",
        "  transit nearest the Lissauer+2024 epoch",
        "",
        "Changes of disposition are expected and are not reported.",
        "",
        f"## Different kepid ({len(kepid_rows)})",
        "",
    ]
    if kepid_rows:
        lines += [
            "| KOI | catalog | kepid | Lissauer kepid |",
            "|---|---|---|---|",
        ]
        lines += [f"| {k} | {s} | {a} | {b} |" for k, s, a, b in kepid_rows]
    else:
        lines.append("None.")
    catalogs = {}
    for k, source, *_ in alias_rows:
        catalogs.setdefault(k, []).append(source)
    lines += [
        "",
        f"## Period aliases ({len(catalogs)} KOIs)",
        "",
        f"DR period within {ALIAS_TOL:g} of 2, 3, 4, 5 or 1/2, 1/3, 1/4, 1/5 times the",
        "Lissauer+2024 period. Other parameters are not compared for these",
        "entries. The output period is the Lissauer+2024 period, except where",
        f"`{ALIASES.relative_to(ROOT)}` overrides the ephemeris (sources in its",
        "`comment` column).",
        "",
        "| KOI | aliased catalogs | Lissauer period | output period |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {k} | {', '.join(c)} | {lissauer.at[k, 'koi_period']} | "
        f"{merged.at[k, 'koi_period']} |"
        for k, c in catalogs.items()
    ]
    # one line per KOI; catalogs with the same values are combined
    by_koi = {}
    for k, s, c, a, b, f, n in param_rows:
        key = (c.removeprefix("koi_"), fmt(a), fmt(b), fmt(f), fmt(n))
        by_koi.setdefault(k, {}).setdefault(key, []).append(s)
    lines += ["", f"## Parameter inconsistencies ({len(by_koi)} KOIs)", ""]
    counts = pd.Series(
        [key[0] for d in by_koi.values() for key in d], dtype=object
    ).value_counts()
    lines += [
        f"- `{c}`: {counts.get(c, 0)} KOIs"
        for c in ["period", "time0bk", "depth", "duration"]
    ]
    lines += [
        "",
        "Each difference: parameter (catalogs): DR value -> Lissauer+2024 value",
        "[frac, nsigma]. `frac`: fractional difference (epoch: offset / duration).",
        "`nsigma`: difference in combined sigma (blank if an uncertainty is missing).",
        "",
    ]
    if by_koi:
        lines += ["| KOI | differences |", "|---|---|"]
        for k, d in by_koi.items():
            diffs = "; ".join(
                f"{c} ({', '.join(s)}): {a} -> {b} [{f}, {n}]"
                for (c, a, b, f, n), s in d.items()
            )
            lines.append(f"| {k} | {diffs} |")
    disp = merged["alderaan_disp"]
    lines += [
        "",
        "## ALDERAAN disposition",
        "",
        f"- `alderaan_disp`: FP {(disp == 'FP').sum()}, PC (incl. M) "
        f"{(disp == 'PC').sum()}, A (ambiguous) {(disp == 'A').sum()}",
        "",
        "`alderaan_disp` is from the Lissauer+2024 disposition (first letter of",
        "`status`): F -> FP, P or M -> PC, S or R -> A (ambiguous). Letters:",
        "F false positive, P planet/candidate, M mass > 13 M_Jup, S S/N < 7.1,",
        "R radius too large.",
        "",
        f"### Lissauer+2024 changes from DR25supp ({changes.sum()})",
        "",
        "DR25supp: second letter of `status` (N: not in DR25supp).",
        "",
        "| DR25supp | Lissauer+2024 | alderaan_disp | KOIs |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {s} | {lis} | {LISSAUER_DISP[lis]} | {n} |"
        for (s, lis), n in changes.items()
    ]
    old = REPORT.read_text() if REPORT.exists() else ""
    manual = (
        old[old.index(MANUAL_MARKER) :]
        if MANUAL_MARKER in old
        else MANUAL_MARKER + "\n"
    )
    REPORT.write_text("\n".join(lines) + "\n\n" + manual)


def main():
    tables, header = load_catalogs()
    lissauer, status = load_lissauer(header)
    kepid_rows, alias_rows, param_rows = compare(tables, lissauer)
    merged = merge(lissauer)
    changes = designate(merged, status)
    # DR columns that Lissauer+2024 provides
    columns = [c for c in header if c in LISSAUER_TO_DR or c == "kepoi_name"]
    columns += ["alderaan_disp"]
    merged.reset_index(names="kepoi_name")[columns].to_csv(OUTPUT, index=False)
    write_report(tables, lissauer, merged, kepid_rows, alias_rows, param_rows, changes)
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({len(merged)} KOIs)")
    print(f"wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
