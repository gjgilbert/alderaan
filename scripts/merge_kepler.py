"""Make one Kepler KOI catalog from Lissauer+2024 and the DR22, DR24, DR25 tables.

The KOIs and values are those of Lissauer+2024 (Table 1), converted to DR
columns (LISSAUER_TO_DR). Depth, duration, and impact (SHAPE_COLUMNS) are from
the first of DR25 > DR24 > DR22 with the same period (within SHAPE_PERIOD_TOL)
and positive depth and duration; else from Lissauer+2024. Non-empty values in
OVERRIDES (manual corrections, with reasons) replace all of these. False
positives (alderaan_disp = FP, including FP in OVERRIDES) are dropped.

Each DR entry is also compared with the Lissauer+2024 entry, and
inconsistencies are written to notes/kepler.md.

Disposition (alderaan_disp): FP = false positive, PC = planet candidate
(transit to model; includes Lissauer+2024 M, mass > 13 M_Jup), A = ambiguous.
From the Lissauer+2024 disposition (F -> FP, P or M -> PC, S or R -> A).
Differences from the DR25 supplemental disposition are written to
notes/kepler.md.

Stellar parameters (Teff [K], logg [cgs], [Fe/H], mass, radius, density) are
all from the first source with all six values: Berger+2023 > Berger+2020 >
CKS (Fulton & Petigura 2018) > DR25 > Mathur+2017 (STAR_SOURCES).

The output has the columns of the ALDERAAN KOI catalog (OUTPUT_COLUMNS; see
tests/catalogs/kepler_dr25_gaia_dr2_crossmatch.csv), without ror and dor:
    disposition  alderaan_disp
    koi_id       KOI host star (e.g. K00001); npl = number of output KOIs on it
    logrho       log10(density / solar density)
    limbdark_*   empty
    snr          koi_model_snr (Lissauer+2024)

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
STARS_DIR = ROOT / "reference" / "stars"
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
# depth, duration, impact: from a DR catalog with the same period (see merge)
SHAPE_COLUMNS = [
    c + s
    for c in ["koi_depth", "koi_duration", "koi_impact"]
    for s in ["", "_err1", "_err2"]
]
SHAPE_PERIOD_TOL = 0.01  # fractional; excludes period aliases
# manual corrections of catalog values (empty: not corrected), with reasons
OVERRIDES = ALDERAAN_DIR / "kepler_overrides_alderaan.csv"
# Lissauer+2024 status letter -> alderaan_disp
LISSAUER_DISP = {"F": "FP", "M": "PC", "P": "PC", "S": "A", "R": "A"}

# Berger+2023 Gaia-Kepler-TESS host stars, output (Table 4)
BERGER_OUTPUT = "gaia_stars_berger_2023.csv"
# Berger+2020 Gaia-Kepler stars, output (Table 2); iso_rho is log10(rho / rho_sun)
BERGER_2020 = "gaia_stars_berger_2020.txt"
# stellar parameters, in the order of BERGER_OUTPUT; rho in solar units
STAR_PARAMS = ["teff", "logg", "feh", "mass", "rad", "rho"]
STAR_COLUMNS = [f"star_{p}{s}" for p in STAR_PARAMS for s in ["", "_err1", "_err2"]]
# stellar parameter sources, in order of preference
STAR_SOURCES = [
    "berger2023",
    "berger2020",
    "cks",
    "dr25",
    "mathur2017",
]
RHO_SUN = 1.408  # g/cm3
# CKS (Fulton & Petigura 2018) Table 2, one row per KOI host star
CKS = "cks_stars_fulton_petigura_2018.txt"
# CKS fields (first and last byte, from its ReadMe): value, upper, lower error
CKS_FIELDS = {
    "teff": [(8, 11), (13, 14), (16, 17)],
    "feh": [(19, 23), (25, 28), (30, 33)],
    "mass": [(80, 84), (86, 90), (92, 96)],  # Miso
    "rad": [(98, 103), (105, 109), (111, 115)],  # Riso
    "rho": [(117, 120), (122, 125), (127, 130)],  # rhoiso, g/cm3
}
LOGG_SUN = 4.438  # cgs
# output columns: (output name, merged column)
OUTPUT_COLUMNS = [
    ("planet_name", "kepoi_name"),
    ("disposition", "alderaan_disp"),
    ("kic_id", "kepid"),
    ("koi_id", "koi_id"),
    ("npl", "npl"),
    ("kep_mag", "koi_kepmag"),
    *[
        (name + s, col + s)
        for name, col in [
            ("Mstar", "star_mass"),
            ("Rstar", "star_rad"),
            ("logrho", "logrho"),
            ("Teff", "star_teff"),
            ("FeH", "star_feh"),
            ("logg", "star_logg"),
        ]
        for s in ["", "_err1", "_err2"]
    ],
    ("limbdark_1", "limbdark_1"),
    ("limbdark_2", "limbdark_2"),
    *[
        (name + s, col + s)
        for name, col in [
            ("period", "koi_period"),
            ("epoch", "koi_time0bk"),
            ("depth", "koi_depth"),
            ("duration", "koi_duration"),
            ("impact", "koi_impact"),
        ]
        for s in ["", "_err1", "_err2"]
    ],
    ("snr", "koi_model_snr"),
]
# Mathur+2017 DR25 stellar properties (Table 4), all Kepler targets
MATHUR = "kepler_stars_mathur_2017.txt"
# Mathur+2017 fields (first and last byte, from its ReadMe): value, upper, lower error
MATHUR_FIELDS = {
    "teff": [(10, 14), (16, 19), (21, 24)],
    "logg": [(26, 31), (33, 37), (39, 43)],
    "feh": [(45, 50), (52, 56), (58, 62)],
    "mass": [(87, 91), (93, 97), (99, 103)],
    "rad": [(64, 70), (72, 77), (79, 85)],
    "rho": [(105, 113), (115, 123), (125, 133)],  # g/cm3
}


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


def merge(lissauer, tables):
    """Lissauer+2024 values, with SHAPE_COLUMNS from DR25 > DR24 > DR22 and
    OVERRIDES (see module docstring); shape_source records the source of
    SHAPE_COLUMNS."""
    overrides = pd.read_csv(OVERRIDES, dtype=str, keep_default_na=False)
    overrides = overrides.set_index("kepoi_name").drop(columns="comment")
    if overrides.index.duplicated().any():
        raise ValueError(f"{OVERRIDES.name}: duplicate KOI")
    missing = overrides.index.difference(lissauer.index)
    if len(missing):
        raise ValueError(f"{OVERRIDES.name}: KOIs not in {LISSAUER}: {list(missing)}")
    disp = overrides.get("alderaan_disp", pd.Series(dtype=str))
    if not disp.isin(["", *set(LISSAUER_DISP.values())]).all():
        raise ValueError(f"{OVERRIDES.name}: unexpected alderaan_disp")
    merged = lissauer.copy()

    def override():
        for col in overrides.columns:
            v = overrides.loc[overrides[col] != "", col]
            merged.loc[v.index, col] = v

    override()  # first, so that DR periods are matched to corrected periods
    merged["shape_source"] = "lissauer"
    period = merged["koi_period"].astype(float)
    for source, t in tables.items():
        t = t.reindex(merged.index)
        dr = t[["koi_period", "koi_depth", "koi_duration"]].replace("", np.nan)
        dr = dr.astype(float)
        use = (
            (merged["shape_source"] == "lissauer")
            & ((dr["koi_period"] / period - 1).abs() < SHAPE_PERIOD_TOL)
            & (dr["koi_depth"] > 0)
            & (dr["koi_duration"] > 0)
        )
        merged.loc[use, SHAPE_COLUMNS] = t.loc[use, SHAPE_COLUMNS]
        merged.loc[use, "shape_source"] = source
    override()
    shape = overrides[[c for c in SHAPE_COLUMNS if c in overrides]]
    merged.loc[shape.index[(shape != "").any(axis=1)], "shape_source"] = "override"
    # every KOI needs an ephemeris, depth, and duration
    needed = ["koi_period", "koi_time0bk", "koi_depth", "koi_duration"]
    empty = merged.index[(merged[needed] == "").any(axis=1)]
    if len(empty):
        raise ValueError(f"missing ephemeris, depth, or duration: {list(empty)}")
    return merged.sort_index()


def load_berger_2023():
    """Berger+2023 stellar parameters, indexed by kepid."""
    t = pd.read_csv(STARS_DIR / BERGER_OUTPUT)
    if t["id_starname"].duplicated().any():
        raise ValueError(f"{BERGER_OUTPUT}: duplicate id_starname")
    t = t[t["id_starname"].str.startswith("kic")]
    t = t.set_index(t["id_starname"].str.removeprefix("kic"))
    stars = t[[c.replace("star_", "iso_") for c in STAR_COLUMNS]]
    stars.columns = STAR_COLUMNS
    return stars


def load_berger_2020():
    """Berger+2020 stellar parameters, indexed by kepid; rho in solar units."""
    t = pd.read_csv(STARS_DIR / BERGER_2020, sep="&", usecols=range(19))
    if t["KIC"].duplicated().any():
        raise ValueError(f"{BERGER_2020}: duplicate KIC")
    t = t.set_index(t["KIC"].astype(str))
    log = t["iso_rho"]
    t["iso_rho"] = 10**log
    for s in ["_err1", "_err2"]:
        t["iso_rho" + s] = 10 ** (log + t["iso_rho" + s]) - t["iso_rho"]
    stars = t[[c.replace("star_", "iso_") for c in STAR_COLUMNS]]
    stars.columns = STAR_COLUMNS
    return stars


def load_cks(merged):
    """CKS stellar parameters, indexed by kepid; logg from Miso and Riso."""
    rows = {}
    with open(STARS_DIR / CKS) as f:
        for line in f:
            if not line.startswith("K"):
                continue
            row = {}
            for p, fields in CKS_FIELDS.items():
                v = [float(line[a - 1 : b]) for a, b in fields]
                row[f"star_{p}"], row[f"star_{p}_err1"] = v[0], v[1]
                row[f"star_{p}_err2"] = -v[2]  # lower error, positive in the table
            rows[line[:6]] = row
    stars = pd.DataFrame.from_dict(rows, orient="index")
    # no isochrone fit: Miso, Riso, rhoiso are 0
    iso = ["star_mass", "star_rad", "star_rho"]
    stars.loc[(stars[iso] == 0).any(axis=1), iso] = np.nan
    for s in ["", "_err1", "_err2"]:
        stars["star_rho" + s] /= RHO_SUN
    m, r = stars["star_mass"], stars["star_rad"]
    stars["star_logg"] = LOGG_SUN + np.log10(m) - 2 * np.log10(r)
    # first-order propagation; upper error from the mass upper and radius lower errors
    for s, o, sign in [("_err1", "_err2", 1), ("_err2", "_err1", -1)]:
        frac = np.hypot(stars["star_mass" + s] / m, 2 * stars["star_rad" + o] / r)
        stars["star_logg" + s] = sign * frac / np.log(10)
    # KOI host star (e.g. K00001) -> kepid
    host = merged["kepid"].groupby(merged.index.str[:6]).first()
    missing = stars.index.difference(host.index)
    if len(missing):
        raise ValueError(f"{CKS}: KOIs not in {LISSAUER}: {list(missing)}")
    return stars.set_index(host[stars.index].to_numpy())[STAR_COLUMNS]


def load_mathur():
    """Mathur+2017 stellar parameters, indexed by kepid; rho in solar units."""
    with open(STARS_DIR / MATHUR) as f:
        lines = f.readlines()
    # data follow the last dashed line of the ReadMe
    start = max(i for i, line in enumerate(lines) if line.startswith("-----")) + 1
    fields = [(1, 8)] + [ab for v in MATHUR_FIELDS.values() for ab in v]
    t = pd.read_fwf(
        STARS_DIR / MATHUR,
        colspecs=[(a - 1, b) for a, b in fields],
        names=["kepid", *STAR_COLUMNS],
        skiprows=start,
        header=None,
    )
    if t["kepid"].duplicated().any():
        raise ValueError(f"{MATHUR}: duplicate KIC")
    stars = t.set_index(t["kepid"].astype(str))[STAR_COLUMNS]
    for p in STAR_PARAMS:
        stars[f"star_{p}_err2"] *= -1  # lower error, positive in the table
    for s in ["", "_err1", "_err2"]:
        stars["star_rho" + s] /= RHO_SUN
    return stars


def dr_stars(table):
    """Stellar parameters of a DR KOI table, indexed by kepid; rho = M / R^3.

    Solar placeholders (koi_sparprov = Solar) are left out.
    """
    t = table[table["koi_sparprov"] != "Solar"]
    t = t.mask(t == "").drop_duplicates("kepid").set_index("kepid")
    dr = {
        "teff": "steff",
        "logg": "slogg",
        "feh": "smet",
        "mass": "smass",
        "rad": "srad",
    }
    stars = pd.DataFrame(index=t.index)
    for p, c in dr.items():
        for s in ["", "_err1", "_err2"]:
            stars[f"star_{p}{s}"] = t[f"koi_{c}{s}"].astype(float)
    m, r = stars["star_mass"], stars["star_rad"]
    stars["star_rho"] = m / r**3
    # first-order propagation; upper error from the mass upper and radius lower errors
    for s, o, sign in [("_err1", "_err2", 1), ("_err2", "_err1", -1)]:
        frac = np.hypot(stars["star_mass" + s] / m, 3 * stars["star_rad" + o] / r)
        stars["star_rho" + s] = sign * stars["star_rho"] * frac
    return stars[STAR_COLUMNS]


def add_stars(merged, tables):
    """Add stellar parameters (STAR_COLUMNS).

    Each star takes all its parameters from the first source in STAR_SOURCES
    that has all six values; star_source records it (empty: no source).
    """
    sources = {
        "berger2023": load_berger_2023(),
        "berger2020": load_berger_2020(),
        "cks": load_cks(merged),
        "dr25": dr_stars(tables["dr25"].reset_index()),
        "mathur2017": load_mathur(),
    }
    assert list(sources) == STAR_SOURCES
    kepid = merged["kepid"]
    merged["star_source"] = ""
    for col in STAR_COLUMNS:
        merged[col] = np.nan
    for name, stars in sources.items():
        values = stars.dropna(subset=[f"star_{p}" for p in STAR_PARAMS])
        use = (merged["star_source"] == "") & kepid.isin(values.index)
        merged.loc[use, STAR_COLUMNS] = values.loc[kepid[use], STAR_COLUMNS].to_numpy()
        merged.loc[use, "star_source"] = name


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
    out = merged[merged["alderaan_disp"] != "FP"]  # rows of OUTPUT
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
        f"- Output: `{OUTPUT.relative_to(ROOT)}`, {len(out)} KOIs (FP dropped);",
        "  values from Lissauer+2024, corrections from",
        f"  `{OVERRIDES.relative_to(ROOT)}`",
        "- Source of output depth, duration, impact: "
        + ", ".join(
            f"{s.upper()} {(out['shape_source'] == s).sum()}"
            for s in [*CATALOGS, "lissauer", "override"]
        )
        + "; zero depth or duration: "
        + (
            ", ".join(
                out.index[
                    (out["koi_depth"].astype(float) == 0)
                    | (out["koi_duration"].astype(float) == 0)
                ]
            )
            or "none"
        ),
        f"- KOIs in more than one DR catalog: {n_multi}",
        f"- Output stars: {out['kepid'].nunique()}; source of stellar parameters: "
        + ", ".join(
            f"{s or 'none'} {out.loc[out['star_source'] == s, 'kepid'].nunique()}"
            for s in [*STAR_SOURCES, ""]
        ),
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
        f"`{OVERRIDES.relative_to(ROOT)}` overrides the ephemeris (sources in its",
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


def output_catalog(merged):
    """The output catalog (OUTPUT_COLUMNS): PC and A KOIs, sorted by KOI."""
    out = merged[merged["alderaan_disp"] != "FP"].reset_index(names="kepoi_name")
    out["koi_id"] = out["kepoi_name"].str[:6]
    out["npl"] = out.groupby("koi_id")["kepoi_name"].transform("size")
    rho = out["star_rho"]
    out["logrho"] = np.log10(rho)
    for s in ["_err1", "_err2"]:
        out["logrho" + s] = np.log10(rho + out["star_rho" + s]) - out["logrho"]
    out["limbdark_1"] = out["limbdark_2"] = np.nan
    # computed values to 3 decimals; others as in their source tables
    computed = [c for c in out.columns if c.startswith(("star_", "logrho"))]
    out[computed] = out[computed].round(3)
    out = out[[col for _, col in OUTPUT_COLUMNS]]
    out.columns = [name for name, _ in OUTPUT_COLUMNS]
    return out.sort_values("planet_name").reset_index(drop=True)


def main():
    tables, header = load_catalogs()
    lissauer, status = load_lissauer(header)
    kepid_rows, alias_rows, param_rows = compare(tables, lissauer)
    changes = designate(lissauer, status)
    merged = merge(lissauer, tables)
    add_stars(merged, tables)
    output = output_catalog(merged)
    output.to_csv(OUTPUT)
    write_report(tables, lissauer, merged, kepid_rows, alias_rows, param_rows, changes)
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({len(output)} KOIs)")
    print(f"wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
