# Contributing to ProxyAudit

Thanks for your interest! ProxyAudit is a research codebase; contributions that
improve correctness, reproducibility, or clarity are very welcome.

## Development setup
```bash
git clone https://github.com/your-org/proxyaudit
cd proxyaudit
python -m venv .venv && source .venv/bin/activate
make dev            # editable install + dev/app extras
make test           # run pytest
```

## Ground rules
- **No fabricated results.** Every reported number must come from
  `scripts/run_synth.py` on the reproducible testbed, from a user-run on a
  licensed real dataset, or from a *cited published* baseline. Synthetic twins
  are allowed only when clearly disclosed as such.
- **Tests for new metrics.** Any new fairness / faithfulness / counterfactual
  metric needs a unit test (see `tests/`), ideally with a closed-form check.
- **Keep the core dependency-light.** The package must run without `shap`,
  `xgboost`, `streamlit`, or `datasets`; those stay optional.
- Run `make lint` before opening a PR.

## Submitting
Open a pull request describing the change and how you verified it. For research
claims, point to the script and seed that reproduce the figure or number.
