.PHONY: test test-fast test-debug profile profile-pipeline

# All targets run through `conda run -n alderaan`.

# Full suite, including @pytest.mark.slow (real data, samplers).
test:
	conda run -n alderaan python -m pytest tests/

# Fast pre-commit subset: excludes @pytest.mark.slow.
test-fast:
	conda run -n alderaan python -m pytest tests/ -m "not slow"

# Rerun only last-failed (--lf), halt at first failure (-x).
# Seed the failure cache with `make test` first.
test-debug:
	conda run -n alderaan python -m pytest --lf -x

# Profiling: standalone harnesses in tests/profiling/ (not pytest tests).
# See "## Profiling" in CLAUDE.md.
#
# `make profile` runs every per-algorithm harness in one process (profile_all.py)
# so pipeline state is built once. `make profile-pipeline` (full run, I/O, plots)
# is separate because it repeats all of that work.
# Per-algorithm: `make profile-<x>`; names mirror tests/regression/test_<x>.py.
profile: profile-all

profile-pipeline:
	conda run -n alderaan python -m tests.profiling.profile_pipeline

# profile-<x> -> tests/profiling/profile_<x>.py
profile-%:
	conda run -n alderaan python -m tests.profiling.profile_$*
