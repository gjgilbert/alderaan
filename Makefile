.PHONY: test test-fast test-debug

# All targets run through `conda run -n alderaan`.

# Full suite, including @pytest.mark.slow (real data, theano compiles, samplers).
test:
	conda run -n alderaan python -m pytest tests/

# Fast pre-commit subset: excludes @pytest.mark.slow.
test-fast:
	conda run -n alderaan python -m pytest tests/ -m "not slow"

# Debug loop: rerun only last-failed (--lf) and halt at the first failure (-x).
# Run `make test` once to seed the failure cache, then iterate on this.
test-debug:
	conda run -n alderaan python -m pytest --lf -x
