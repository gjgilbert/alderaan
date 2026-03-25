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

- **Python>=3.9**
- **Conda env**: `alderaan` — set up via `conda env create -f environment.yml`
- **Install package**: `pip install -e alderaan` (editable install)


## Design Principles

From the project charter — these guide all implementation decisions:

1. No hidden state
2. No implicit calibration assumptions
3. Deterministic stacking and wavelength solution
4. Fail loudly
6. Prefer clarity over cleverness
7. Implement everything in the simplest possible way.

Every major change must: preserve deterministic behavior, run on the truth dataset, and document impact on transit posteriors.

- Keep this file updated as you make mistakes, learn lessons, and learn more efficient ways of coding or answering prompts.
- Use this file as your long-term memory.