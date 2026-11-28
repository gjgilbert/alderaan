"""Migrate ALDERAAN output FITS headers, in place, to the current card schema.

Usage::

    python scripts/migrate_fits_headers.py <dir>

Every *_results.fits and *_detrended.fits under <dir> (recursively) is updated:
old card names are renamed (RENAME), cards that old files omit for a missing
value are added blank (BLANK), SAMPLES gets the provenance cards of PRIMARY
(PROVENANCE), and every known card gets its comment (COMMENTS). Data are not
changed. A migrated file is unchanged by a second run. Keep a backup until the
migrated files load.
"""

import glob
import os
import re
import sys

from astropy.io import fits

RENAME = {
    "T_START": "TSTART",
    "T_END": "TSTOP",
    "RUN_ID": "RUNID",
    "GIT_HASH": "GITHASH",
    "KOI_ID": "KOIID",
    "KIC_ID": "KICID",
    "LN_Z": "LNZ",
    "LN_ZERR": "LNZERR",
    "LD_U1": "LDU1",
    "LD_U2": "LDU2",
}

BLANK = {"TTIMES": ("OMCMODEL", "OMCDOF", "TTVSCAT"), "SAMPLES": ("LDU1", "LDU2")}

PROVENANCE = ("RUNID", "VERSION", "GITHASH")

# indexed cards (KEY<i>) are looked up by KEY and formatted with i
COMMENTS = {
    "MISSION": "observing mission",
    "TARGET": "target ID",
    "NPL": "number of planets",
    "KOIID": "KOI ID",
    "KICID": "Kepler Input Catalog ID",
    "CATALOG": "planet catalog",
    "VISITS": "Kepler quarters loaded",
    "TIMESYS": "BJD - 2454833",
    "RUNID": "ALDERAAN run ID",
    "VERSION": "ALDERAAN version",
    "GITHASH": "ALDERAAN git commit",
    "DTKERNEL": "detrending GP kernel",
    "DTNOMP": "[d] detrending nominal period",
    "RMSRAW": "visit {i} raw RMS out of transit",
    "RMSDET": "visit {i} detrended RMS out of transit",
    "PLNAME": "planet {i} name",
    "CHI2RED": "reduced chi2 of the median model",
    "NITER": "dynesty iterations",
    "EFF": "[%] dynesty sampling efficiency",
    "NCALL": "likelihood calls",
    "NEFF": "effective sample size",
    "LNZ": "ln evidence",
    "LNZERR": "ln evidence error",
    "INFO": "[nat] information",
    "SEED": "dynesty random seed",
    "BOUND": "dynesty bounding method",
    "SAMPLE": "dynesty sampling method",
    "LDPRIOR": "limb darkening prior on LDU1, LDU2",
    "LDU1": "limb darkening prior u1",
    "LDU2": "limb darkening prior u2",
    "TSTART": "[d] start of data span",
    "TSTOP": "[d] end of data span",
    "NOBSMODE": "number of obsmodes",
    "OBSMODE": "obsmode {i}",
    "SUPSAMP": "model supersampling of obsmode {i}",
    "OMCMODEL": "selected O-C model",
    "OMCDOF": "O-C model degrees of freedom",
    "TTVSCAT": "[d] TTV scatter about O-C model",
}


def comment(keyword):
    """Comment for a card keyword, or None if the keyword is not known"""
    if keyword in COMMENTS:
        return COMMENTS[keyword]
    key, i = re.fullmatch(r"(.*?)(\d*)", keyword).groups()
    return COMMENTS[key].format(i=i) if i and key in COMMENTS else None


def migrate(path):
    """Migrate the headers of one file in place"""
    with fits.open(path, mode="update") as hdul:
        primary = hdul[0].header
        for hdu in hdul:
            h = hdu.header
            for old, new in RENAME.items():
                if old in h:
                    h.rename_keyword(old, new)
            for k in BLANK.get(hdu.name.split("_")[0], ()):
                if k not in h:
                    h[k] = None
            if hdu.name == "SAMPLES":
                for k in PROVENANCE:
                    if k not in h and k in primary:
                        h[k] = primary[k]
            for k in list(h.keys()):
                c = comment(k)
                if c is not None:
                    h.comments[k] = c


def main(argv):
    (root,) = argv
    pattern = ("*_results.fits", "*_detrended.fits")
    paths = sorted(
        p
        for pat in pattern
        for p in glob.glob(os.path.join(root, "**", pat), recursive=True)
    )
    for path in paths:
        migrate(path)
        print(f"migrated {path}")
    print(f"{len(paths)} files")


if __name__ == "__main__":
    main(sys.argv[1:])
