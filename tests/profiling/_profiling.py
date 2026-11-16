"""Shared harness for the ALDERAAN "tallest tentpole" profiling suite.

This is **not** a test module (no ``test_`` prefix) and pytest does not collect
it. It is imported by the standalone ``tests/profiling/profile_*.py`` harnesses,
run on demand via ``make profile*`` (see the ``## Profiling`` section of
CLAUDE.md). It runs fully standalone with no interactive input.

Strategy
--------
We only care about the most critical bottlenecks ("tentpoles"). Each harness:

1. Runs the target under :mod:`cProfile` (pass 1) and charges every function's own
   time to the nearest enclosing **ALDERAAN function** (attribution), so the
   ranking points at the function to optimize rather than the library leaf it
   bottoms out in.
2. Flags the **hotspots**: functions with >= :data:`HOTSPOT_FRACTION` attributed
   time AND >= :data:`HOTSPOT_MIN_SECONDS`. "No hotspot" is a fine outcome.
3. Drills into each hotspot line-by-line with :mod:`line_profiler` (pass 2).
4. Writes a Markdown report to stdout and to the gitignored :data:`REPORTS_DIR`.

Inputs are real K00137 (Kepler-18) photometry, quarters 4-5 (see :data:`VISITS`).
Each algorithm harness profiles every invocation of that algorithm in the
pipeline, starting from the pipeline state that precedes it (:data:`STEPS`
mirrors ``alderaan.pipeline._run``); building that state is excluded from the
profile. ``run_profile`` returns the profiled wall-clock seconds.

Harness-only sampler settings (:func:`patch_samplers`): dynesty stops after
:data:`DYNESTY_MAXITER` iterations, and pymc runs its NUTS chains in-process
(``cores=1``), because cProfile cannot see forked worker processes.
"""

import ast
import copy
import cProfile
import functools
import inspect
import io
import os
import pstats
import sys
import textwrap
import time
from collections import namedtuple
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from line_profiler import LineProfiler  # noqa: E402

import alderaan  # noqa: E402
from alderaan.algorithms import (  # noqa: E402
    ClipOutliers,
    DetrendVisits,
    FlagBadTransits,
    QuickFitTransitShape,
    QuickFitTransitTimes,
    RegularizeOMC,
    SampleTransitModel,
)

# --- inputs ------------------------------------------------------------------

TESTS_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = TESTS_DIR / "data" / "MAST_downloads"
CATALOG_DIR = TESTS_DIR / "catalogs"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"

TARGET = "K00137"
MISSION = "Kepler"
# Two visits: the fewest that give planet d (P ~ 15 d) >= 8 transits, so
# RegularizeOMC runs its full model selection (NUTS) for every planet.
VISITS = [4, 5]

# --- thresholds (fractions are of the total own-time budget) -----------------

HOTSPOT_FRACTION = 0.20  # >= this share of attributed time AND ...
HOTSPOT_MIN_SECONDS = 1.0  # ... >= this many seconds -> a hotspot
TOP_FUNCTION_MIN_FRACTION = 0.02  # >= this share -> listed in the ranking

# --- harness-only sampler settings -------------------------------------------

DYNESTY_MAXITER = 5000

# --- pipeline steps (mirrors alderaan.pipeline._run, minus plots and I/O) ----

Step = namedtuple("Step", "name cls kwargs reset")
STEPS = [
    Step("omc_smooth", RegularizeOMC, {"method": "smooth"}, False),
    Step("clip_local_trend", ClipOutliers, {"method": "local_trend"}, False),
    Step("detrend_wide_mask", DetrendVisits, {"method": "wide_mask"}, True),
    Step("flag_pass1", FlagBadTransits, {}, False),
    Step("quick_fit_shape", QuickFitTransitShape, {}, False),
    Step("quick_fit_times", QuickFitTransitTimes, {}, False),
    Step("omc_select", RegularizeOMC, {"method": "select"}, False),
    Step("clip_transit_model", ClipOutliers, {"method": "transit_model"}, False),
    Step("detrend_tight_mask", DetrendVisits, {"method": "tight_mask"}, True),
    Step("flag_pass2", FlagBadTransits, {}, False),
    Step("sample", SampleTransitModel, {}, False),
]

# --- recipe-report labels ----------------------------------------------------

IO_LABEL = "I/O (load + write)"
REMAINDER_LABEL = "(plots + orchestration)"


def patch_samplers():
    """Cap dynesty iterations and run NUTS chains in-process (harness only)."""
    import dynesty
    import pymc

    run_nested = dynesty.DynamicNestedSampler.run_nested

    @functools.wraps(run_nested)
    def capped(self, *args, **kwargs):
        kwargs.setdefault("maxiter", DYNESTY_MAXITER)
        return run_nested(self, *args, **kwargs)

    dynesty.DynamicNestedSampler.run_nested = capped
    pymc.sample = functools.partial(pymc.sample, cores=1)


def require_data():
    """Exit 0 with a clear message when the K00137 photometry is absent."""
    if not DATA_DIR.exists():
        print(f"[skip] {DATA_DIR} not present; profiling needs real photometry.")
        raise SystemExit(0)


# --- pipeline state ----------------------------------------------------------


@functools.cache
def _load():
    from alderaan.utils.kepler import load_kepler_data

    star, planets, litecurve = load_kepler_data(
        TARGET, str(DATA_DIR), str(CATALOG_DIR), visits=VISITS
    )
    return star, planets, litecurve, copy.deepcopy(litecurve)


_STATES = {}  # step name -> state after that step (built once per process)


def run_step(step, state):
    """Apply one pipeline step to state = (star, planets, litecurve, raw)."""
    star, planets, litecurve, raw = state
    if step.reset:
        litecurve.reset(raw)  # detrend raw photometry for surviving cadences
    result = step.cls(star, planets, litecurve).perform(**step.kwargs)
    if step.cls is SampleTransitModel:
        return result
    planets, litecurve = result
    return star, planets, litecurve, raw


def state_before(name):
    """Fresh deep copy of the pipeline state entering step ``name``."""
    state = _load()
    for step in STEPS:
        if step.name == name:
            return copy.deepcopy(state)
        if step.name not in _STATES:
            _STATES[step.name] = run_step(step, copy.deepcopy(state))
        state = _STATES[step.name]
    raise KeyError(name)


def profile_algorithm(cls, title, report_name):
    """Profile every pipeline invocation of algorithm ``cls``.

    Each profiled step's output is kept as the state entering the next step,
    so a later harness in the same process never reruns it as setup.
    """
    steps = [s for s in STEPS if s.cls is cls]

    def setup():
        return [(s, state_before(s.name)) for s in steps]

    def call(jobs):
        for step, state in jobs:
            _STATES.setdefault(step.name, run_step(step, state))

    return run_profile(title, report_name, setup, call)


# --- attribution -------------------------------------------------------------

_PKG_DIR = os.path.dirname(os.path.abspath(alderaan.__file__))


def _is_alderaan(filename):
    """True for ALDERAAN source (anchored to the package dir; the conda env is
    also named ``alderaan``, so a substring test would match site-packages)."""
    return filename.startswith(_PKG_DIR + os.sep)


def _module_label(filename):
    """``.../alderaan/algorithms/x.py`` -> ``algorithms/x.py``"""
    return filename[len(_PKG_DIR) + 1 :] if _is_alderaan(filename) else filename


def _attributed(stats):
    """Charge each function's own time to the nearest enclosing ALDERAAN function.

    cProfile aggregates by function, so a bottleneck surfaces as a library *leaf*
    with no hint of which ALDERAAN function drives it. Push every non-ALDERAAN
    function's own time up the caller graph, weighted by per-caller cumulative
    time, until it lands on an ALDERAAN function. Recursive callers (cycles in
    the caller graph, e.g. pytensor graph rewrites) are skipped and the
    remaining weights renormalized, so time is not lost at a cycle. Returns
    ``{func_key: attributed_seconds}`` (``func_key`` is cProfile's
    ``(filename, lineno, funcname)``).
    """
    raw = stats.stats  # {func: (cc, nc, tt, ct, {caller: (cc, nc, tt, ct)})}
    cache = {}

    def lift(func, stack):
        if _is_alderaan(func[0]):
            return {func: 1.0}
        if func in cache:
            return cache[func]
        if func not in raw:
            return {}
        stack = stack | {func}
        callers = {c: v for c, v in raw[func][4].items() if c not in stack}
        # a result that skipped a non-self caller depends on the path; don't cache
        cacheable = all(c == func or c in callers for c in raw[func][4])
        weights = {c: v[3] for c, v in callers.items() if v[3] > 0}
        if not weights:
            weights = {c: v[1] for c, v in callers.items() if v[1] > 0}
        dist = {}
        for c, wt in weights.items():
            for m, mw in lift(c, stack).items():
                dist[m] = dist.get(m, 0.0) + mw * wt
        tot = sum(dist.values())
        dist = {m: w / tot for m, w in dist.items()} if tot > 0 else {}
        if cacheable:
            cache[func] = dist
        return dist

    attributed = {}
    for func, vals in raw.items():
        if vals[2] <= 0:
            continue
        for m, wt in lift(func, frozenset()).items():
            attributed[m] = attributed.get(m, 0.0) + vals[2] * wt
    return attributed


def _rows(stats):
    """ALDERAAN-function rows ranked by attributed time."""
    total = stats.total_tt or 1e-12
    rows = []
    for func, secs in _attributed(stats).items():
        fn, ln, name = func
        _cc, nc, tt, ct, _callers = stats.stats[func]
        rows.append(
            {
                "file": fn,
                "line": ln,
                "name": name,
                "ncalls": nc,
                "own_s": tt,
                "cumtime": ct,
                "attr_s": secs,
                "attr_frac": secs / total,
            }
        )
    rows.sort(key=lambda r: r["attr_s"], reverse=True)
    return rows, total


def _is_hotspot(r):
    return r["attr_frac"] >= HOTSPOT_FRACTION and r["attr_s"] >= HOTSPOT_MIN_SECONDS


def _note(r):
    """Generic, rule-based optimization hint for a hotspot."""
    if r["attr_s"] - r["own_s"] >= 0.5 * r["attr_s"]:
        return (
            "cost is mostly in library calls it makes; see the drill-down for the "
            "hot line, then cut the call count or batch the array op"
        )
    if r["ncalls"] >= 100_000:
        return "very high call count; push the Python loop into a vectorized op"
    return "own-time hotspot in Python; vectorize the inner loop or cache work"


# --- recipe summary ----------------------------------------------------------


def _recipe_summary(stats, wall):
    """Partition pipeline wall-clock into algorithm stages, I/O, and the rest.

    Each algorithm row is the cumulative time of its ``perform`` (both passes
    combined); I/O is ``load_kepler_data`` plus the outermost ``to_fits`` calls.
    Plots are not separable: under Python 3.12, cProfile drops every frame that
    is active while matplotlib renders a figure (``_run`` and the plot methods
    themselves), so plots and orchestration are the wall-clock remainder.
    """
    buckets = {}

    def add(label, secs):
        buckets[label] = buckets.get(label, 0.0) + secs

    for func, (_cc, _nc, _tt, ct, callers) in stats.stats.items():
        label = _module_label(func[0])
        if func[2] == "perform" and label.startswith("algorithms/"):
            add(label.removeprefix("algorithms/").removesuffix(".py"), ct)
        elif func[2] == "load_kepler_data" or (
            func[2] == "to_fits" and not any(c[2] == "to_fits" for c in callers)
        ):
            add(IO_LABEL, ct)
    add(REMAINDER_LABEL, max(wall - sum(buckets.values()), 0.0))

    rows = sorted(
        ((lbl, s, s / wall) for lbl, s in buckets.items()),
        key=lambda e: e[1],
        reverse=True,
    )
    return {"rows": rows, "denominator": wall}


# --- line-level drill-down ---------------------------------------------------


def _own_functions():
    """Map (filename, first_lineno) -> function for every loaded ALDERAAN function.

    Methods are unwrapped (``BaseAlg`` wraps ``perform``; static/class methods
    and properties are descriptors) so line_profiler gets the real code object.
    """
    registry = {}

    def add(obj):
        if isinstance(obj, (staticmethod, classmethod)):
            obj = obj.__func__
        elif isinstance(obj, property):
            obj = obj.fget
        if not callable(obj):
            return
        obj = inspect.unwrap(obj)
        code = getattr(obj, "__code__", None)
        if code is not None and _is_alderaan(code.co_filename):
            registry[(code.co_filename, code.co_firstlineno)] = obj

    for name, mod in list(sys.modules.items()):
        if not name.startswith("alderaan"):
            continue
        for obj in vars(mod).values():
            if inspect.isfunction(obj):
                add(obj)
            elif inspect.isclass(obj):
                for member in vars(obj).values():
                    add(member)
    return registry


def _drilldown_function(registry, r):
    """The function to line-profile for hotspot row ``r``; a nested function
    (closure) is not addressable, so fall back to the function enclosing it."""
    key = (r["file"], r["line"])
    if key in registry:
        return registry[key]
    best = None
    for (fn, ln), f in registry.items():
        if fn != r["file"] or ln > r["line"]:
            continue
        nlines = len(inspect.getsourcelines(f)[0])
        if r["line"] < ln + nlines and (best is None or ln > best[0]):
            best = (ln, f)
    return best[1] if best else None


def _docstring_lines(funcs):
    """``(filename, lineno)`` pairs occupied by each function's docstring."""
    pairs = set()
    for f in funcs:
        try:
            src_lines, start = inspect.getsourcelines(f)
            node = ast.parse(textwrap.dedent("".join(src_lines))).body[0]
        except (OSError, TypeError, SyntaxError, IndexError):
            continue
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if ast.get_docstring(node, clean=False) is None:
            continue
        doc = node.body[0]
        for rel in range(doc.lineno, (doc.end_lineno or doc.lineno) + 1):
            pairs.add((f.__code__.co_filename, start + rel - 1))
    return pairs


def _strip_docstrings(text, funcs):
    """Remove docstring rows from line-profiler output (never executed)."""
    doc = _docstring_lines(funcs)
    kept = []
    cur_file = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("File:"):
            cur_file = stripped[len("File:") :].strip()
        head = line.split(maxsplit=1)
        if cur_file and head and head[0].isdigit():
            if (cur_file, int(head[0])) in doc:
                continue
        kept.append(line)
    return "\n".join(kept)


def _line_drilldown(call, args, funcs):
    lp = LineProfiler()
    for f in funcs:
        lp.add_function(f)
    lp.runcall(call, *args)
    buf = io.StringIO()
    lp.print_stats(stream=buf, output_unit=1e-3)
    return _strip_docstrings(buf.getvalue(), funcs)


# --- rendering ---------------------------------------------------------------

_SAMPLERS_NOTE = (
    f"Harness-only sampler settings: dynesty stops at {DYNESTY_MAXITER} "
    "iterations; NUTS chains run in-process (`cores=1`)."
)


def _fmt_func(r):
    return f"`{_module_label(r['file'])}:{r['name']}`"


def _render(title, total, rows, flags, line_text):
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    out = []
    w = out.append

    w(f"# Profiling report: {title}")
    w("")
    w(f"_Generated {ts} • total profiled own-time budget: {total:.3f} s_")
    w("")
    w(
        "> Auto-generated by `tests/profiling/_profiling.py`. Each ALDERAAN "
        "function is charged the own time of the library calls it drives "
        "(**attributed time**). A **hotspot** has "
        f"≥ {HOTSPOT_FRACTION:.0%} attributed time and ≥ {HOTSPOT_MIN_SECONDS:.0f} s; "
        f"the line-level drill-down shows where inside it the time goes. "
        f"{_SAMPLERS_NOTE}"
    )
    w("")

    w(
        f"## Recommended actions (≥ {HOTSPOT_FRACTION:.0%} attributed time "
        f"and ≥ {HOTSPOT_MIN_SECONDS:.0f} s)"
    )
    w("")
    if flags:
        for r in flags:
            w(
                f"- {_fmt_func(r)} — {r['attr_frac']:.1%} "
                f"({r['attr_s']:.3f} s): {_note(r)}"
            )
    else:
        w("_Nothing over threshold — **no action needed**._")
    w("")

    w(
        f"## ALDERAAN functions ≥ {TOP_FUNCTION_MIN_FRACTION:.0%} of runtime "
        "(attributed)"
    )
    w("")
    w("| function | attributed % | attributed s | own s | calls |")
    w("|---|---:|---:|---:|---:|")
    for r in rows:
        if r["attr_frac"] < TOP_FUNCTION_MIN_FRACTION:
            continue
        w(
            f"| {_fmt_func(r)} | {r['attr_frac']:.1%} | {r['attr_s']:.3f} | "
            f"{r['own_s']:.3f} | {r['ncalls']} |"
        )
    w("")
    covered = sum(r["attr_s"] for r in rows)
    w(
        f"_ALDERAAN functions account for {covered:.3f} s ({covered / total:.0%}) "
        f"of the {total:.3f} s budget; the rest has no ALDERAAN caller "
        "(e.g. harness glue, import-time work)._"
    )
    w("")

    if line_text:
        w("## Line-level drill-down (hotspots)")
        w("")
        w("```")
        w(line_text.rstrip())
        w("```")
        w("")

    return "\n".join(out)


def _render_recipe(title, summary):
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    out = []
    w = out.append

    w(f"# Profiling report: {title}")
    w("")
    w(f"_Generated {ts} • pipeline wall-clock: {summary['denominator']:.3f} s_")
    w("")
    w(
        "> Auto-generated by `tests/profiling/_profiling.py`. Cumulative "
        "wall-clock per pipeline stage (each includes every function it calls, "
        "both passes combined), partitioned so the rows sum to the runtime. "
        "Plots cannot be timed separately (Python 3.12 cProfile drops frames "
        "active during matplotlib rendering), so they fall in the remainder. "
        f"Function-level detail is in the per-algorithm reports. {_SAMPLERS_NOTE}"
    )
    w("")
    w("## High-level summary")
    w("")
    w("| stage / bucket | % wall | s |")
    w("|---|---:|---:|")
    for lbl, secs, frac in summary["rows"]:
        w(f"| `{lbl}` | {frac:.1%} | {secs:.3f} |")
    w("")
    return "\n".join(out)


# --- driver ------------------------------------------------------------------


def run_profile(title, report_name, setup, call, line_pass=True, recipe=False):
    """Profile ``call(*setup())`` and write a tentpole report.

    Parameters
    ----------
    title : str
        Human-readable report title.
    report_name : str
        Base filename for the Markdown report under :data:`REPORTS_DIR`.
    setup : callable
        Zero-arg builder returning the args for ``call`` (a single value or a
        tuple). Called once per profiling pass so each pass gets fresh input;
        its work is excluded from the measured call.
    call : callable
        The function actually profiled, invoked as ``call(*args)``.
    line_pass : bool, default True
        Run the second :mod:`line_profiler` pass over the hotspots.
    recipe : bool, default False
        End-to-end layout: cumulative wall-clock per pipeline stage, no
        function-level sections (those live in the per-algorithm reports).
    """
    require_data()
    patch_samplers()
    print(f"\n=== profiling: {title} ===\n")

    def as_args(value):
        return value if isinstance(value, tuple) else (value,)

    args = as_args(setup())
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    call(*args)
    pr.disable()
    wall = time.perf_counter() - t0
    stats = pstats.Stats(pr)

    if recipe:
        report = _render_recipe(title, _recipe_summary(stats, wall))
        print(report)
        _write_report(report, report_name)
        return wall

    rows, total = _rows(stats)
    flags = [r for r in rows if _is_hotspot(r)]

    line_text = ""
    if line_pass and flags:
        registry = _own_functions()
        funcs = [_drilldown_function(registry, r) for r in flags]
        funcs = list(dict.fromkeys(f for f in funcs if f is not None))
        if funcs:
            line_text = _line_drilldown(call, as_args(setup()), funcs)

    report = _render(title, total, rows, flags, line_text)
    print(report)
    _write_report(report, report_name)
    return wall


def _write_report(report, report_name):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"{report_name}.md"
    out_path.write_text(report + "\n")
    print(f"\n[report written] {out_path}")
