# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ALDERAAN: an astronomy pipeline for analyzing transit photometry. The pipeline was originally developed for using data from the Kepler Space Telescope. Current development objectives include extending the pipeline to work with data from the K2 and TESS missions as well. We are in the process of refactoring the pipeline to be more modular.

**The scientific priority is accurate inference of transit parameters.** Everything should serve this goal: improving photometric detrending, improving transit timing variation models, improving transit model sampling, making failure modes explicit, and enabling systematic regression testing. It is important to prioritize speed of algorithm execution.

### A Simple Astronomy-Style Pipeline

We are building explicit Python modules with file-in/file-out execution. No workflow framework, no service infrastructure, no database dependency, no orchestration engine, no premature architecture. The goal is clarity, reproducibility, and stability — not elegance.

### Refactoring Philosophy

The previous pipeline included many hard-coded references to the Kepler mission. The algorithms have been thoroughly tested, but the code became difficult to modify. Our current refactoring efforts aim to retain the high quality of the original pipeline but with a code structure that more easily allows for the incorporation of new features.

### Guardrails

Avoid: rebuilding a hidden framework unintentionally, adding implicit global state, embedding database dependencies in science code, introducing silent retries, overengineering abstractions.

## Development Environment

- **Python 3.12**
- **Conda env**: `alderaan` — set up via `conda env create -f environment.yml`
- **Install package**: `pip install -e .` from the repo root (editable install)

## Commands

```bash
# All test commands run in the `alderaan` conda env; the make targets wrap conda run.
# Run from the repo root.

make test        # full suite, including @pytest.mark.slow
make test-fast   # fast subset; excludes slow (real data, samplers); the default
make test-debug  # iteration loop: rerun only last-failed (--lf), halt at first failure (-x)

# Single file / class / test — append ::Class or ::Class::test_name
conda run -n alderaan python -m pytest tests/regression/test_transit_model.py::TestTransitModel -v

# Formatting and linting (Ruff; config in pyproject.toml [tool.ruff])
conda run -n alderaan ruff format alderaan/ tests/
conda run -n alderaan ruff check --fix alderaan/ tests/

# Pipeline CLI (alderaan.cli); --help on each subcommand lists all options
alderaan run -t K00137 -m Kepler --input_dir <dir> --output_dir <dir> --catalog <csv>
alderaan batch -m Kepler --input_dir <dir> --output_dir <dir> --catalog <csv> --jobs N
```

## Testing

No git hook runs tests (pre-commit is ruff-only), so *you* decide when, and the default is
**not yet**. Run tests when meaningful change has accumulated and reached a natural
checkpoint, not after every edit. Match scope to blast radius:

- **Mid-change / trivial edits** (a few lines, a rename, a docstring): don't run.
- **A coherent unit of work is done, or before you commit**: run the targeted file(s)
  for what you touched (`pytest tests/regression/test_<area>.py`), or `make test-fast`
  if the change spans the pipeline core.
- **Wide blast radius only** (shared module, cross-cutting refactor, science-path
  change): `make test` (full).
- **Chasing a failure**: `make test-debug`.
- **End-to-end mechanics** (plots, I/O, logging, CLI, batch; not science): a smoke run.
  A scratch script patches `dynesty.DynamicNestedSampler.run_nested` to `maxiter=300`
  (as `tests/profiling/_profiling.py` does), then calls
  `alderaan.cli.main(["run", "-t", "K00137", "-m", "Kepler", "--input_dir",
  "tests/data/MAST_downloads", "--catalog", "reference/alderaan/kepler_merged_alderaan.csv",
  "--visits", "4", "5", "6", "--output_dir", <fresh tmp dir>])`. Pass: exit 0, expected
  files exist, no `Traceback` in the logs. Never run the full sampler for this.

Layout: `tests/regression/test_<module>.py`, one file per module, named for the module
(`test_litecurve.py`, `test_regularize_omc.py`, `test_base_alg.py`, `test_utils_stats.py`, ...).
`tests/conftest.py` registers the `slow` marker and provides the `k00137` fixture
(Kepler-18, quarters 4-6, from `tests/data/MAST_downloads` and `reference/`) and the
`make_posteriors` factory. Fast tests use small
synthetic objects built in the test file; anything loading real photometry or running
a sampler is `@pytest.mark.slow` (reading a small catalog is fast). A known bug is a
`@pytest.mark.xfail(strict=True, reason=...)` test; when it XPASSes, remove the marker.

Tests check behavior required by the design, not edge cases or development history.
Revise tests when production code changes: code design drives test design, never the
reverse.

Never brute-force diagnostics or exploration (random fuzzing, parameter sweeps, repeated
full runs). Read the code first, list the conditions that matter, then make the smallest
set of targeted calls on the smallest data that still answers the question.

## Profiling

A "tallest tentpole" suite: only the biggest bottlenecks matter, optimization must never
cost scientific accuracy, and "no action needed" is a fine result.

```bash
make profile                    # every per-algorithm report + summary.md ranking them
make profile-pipeline           # full pipeline run: stages, I/O, plots (repeats all work)
make profile-detrend_visits     # one algorithm; names mirror tests/regression/test_<x>.py
```

- Harnesses in `tests/profiling/profile_<x>.py` are not pytest tests; shared logic lives
  in `_profiling.py`. Each profiles every pipeline invocation of one algorithm on K00137
  Q4-5 (`VISITS`; two visits are the fewest that give every planet ≥ 8 transits, so
  RegularizeOMC runs its full model selection), starting from the state that precedes
  it (`STEPS` mirrors `pipeline._run`). `make profile` runs them in one process (`profile_all.py`) so that
  state is built once; a standalone `make profile-<x>` rebuilds everything ahead of its
  steps.
- Library time is attributed to the nearest enclosing ALDERAAN function; a **hotspot**
  has ≥ 20% attributed time and ≥ 1 s, and gets a `line_profiler` drill-down.
- Harness-only sampler setting: the transit-model dynesty run is capped at
  `DYNESTY_MAXITER`.
- Reports print to stdout and write to the gitignored `tests/profiling/reports/`.


## Design Principles

From the project charter — these guide all implementation decisions:

1. No hidden state
2. No implicit assumptions
3. Deterministic behavior
4. Fail loudly
6. Prefer clarity over cleverness
7. Implement everything in the simplest possible way.

Every major change must: preserve deterministic behavior and document impact on transit posteriors. Perform quick local validation that changes to the pipeline do not produce unexpected or large changes to scientific output: compare the K00137 Q4-6 posteriors with the values in `reference/`. What counts as a large change is context specific. Unless the definition of "large change" is clear from context (e.g. within machine precision), ask the user rather than guessing.

The user will run full verification through end-to-end batch processing on real data on the remote machine.

- Do not modify this file without explicit instructions from the user.
