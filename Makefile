PYTHON ?= python3

.PHONY: validate pdfs clean catalog

validate:
	$(PYTHON) tools/validate_scaffold.py

pdfs: validate
	$(PYTHON) tools/build_pdfs.py

catalog:
	$(PYTHON) tools/print_catalog.py

clean:
	$(PYTHON) tools/clean_build.py
