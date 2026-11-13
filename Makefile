.PHONY: test test-debug

# All targets run through `conda run -n alderaan`.

# Full suite.
test:
	conda run -n alderaan python -m pytest tests/

# Debug loop: rerun only last-failed (--lf) and halt at the first failure (-x).
# Run `make test` once to seed the failure cache, then iterate on this.
test-debug:
	conda run -n alderaan python -m pytest --lf -x
