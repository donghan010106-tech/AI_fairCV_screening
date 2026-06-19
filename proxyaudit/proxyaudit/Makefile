.PHONY: install dev synth app paper test lint clean

install:        ## editable install (core deps)
	pip install -e .

dev:            ## editable install with dev + app extras
	pip install -e ".[dev,app]"

synth:          ## reproduce all measured results + figures
	python scripts/run_synth.py

app:            ## launch the interactive auditor
	streamlit run app/streamlit_app.py

paper:          ## build the PDF (XeLaTeX)
	cd paper && xelatex -interaction=nonstopmode main.tex && bibtex main && \
	xelatex -interaction=nonstopmode main.tex && xelatex -interaction=nonstopmode main.tex

test:           ## run the test suite
	pytest

lint:           ## static checks
	ruff check src scripts tests

clean:          ## remove caches and LaTeX aux
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.bbl paper/*.blg paper/*.synctex.gz
	rm -rf .pytest_cache .ruff_cache *.egg-info build dist
