.PHONY: test test-fast test-debug profile profile-pipeline

# All targets run through `conda run -n alderaan`.

# Full suite, including @pytest.mark.slow (real data, pytensor compiles, samplers).
test:
	conda run -n alderaan python -m pytest tests/

# Fast pre-commit subset: excludes @pytest.mark.slow.
test-fast:
	conda run -n alderaan python -m pytest tests/ -m "not slow"

# Debug loop: rerun only last-failed (--lf) and halt at the first failure (-x).
# Run `make test` once to seed the failure cache, then iterate on this.
test-debug:
	conda run -n alderaan python -m pytest --lf -x

# Profiling ("tallest tentpole" suite). Standalone harnesses in tests/profiling/
# (not pytest tests); reports land in tests/profiling/reports/ (gitignored).
# See the "## Profiling" section of CLAUDE.md.
#
# `make profile` runs every per-algorithm harness in one process (profile_all.py) so
# pipeline state is built once; `make profile-pipeline` (full run, I/O and plots) is
# separate because it repeats all of that work.
# Per-algorithm: `make profile-detrend_visits`, `make profile-regularize_omc`, etc.
# Names mirror the test files (test_<x>.py <-> profile_<x>.py).
profile: profile-all

profile-pipeline:
	conda run -n alderaan python -m tests.profiling.profile_pipeline

# Generic per-algorithm rule: profile-<x> -> tests/profiling/profile_<x>.py
profile-%:
	conda run -n alderaan python -m tests.profiling.profile_$*
