PYTHON_EXEC = uv run python3
UV = uv
SRC = src

all: install run

install:
	@echo 'install and sync dependancies'
	$(UV) sync

run:
	@echo 'lunch R.A.G'
	$(PYTHON_EXEC) -m $(SRC)

clean:
	@echo 'remove artefact files'
	rm -rf .venv .uv
	rm -rf $(SRC)/__pycache__
	rm -rf data/output/search_results
	rm -rf data/output/searchSingleQuery
	rm -rf data/Output_SingleQuery
	rm -rf data/processed

debug:
	uv run python -m pdb -m ${SRC}

lint:
	@echo 'check quality code (mypy/flake8) norme'
	flake8 src --exclude .venv
	mypy . --strict --warn-return-any \
	--warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs \
	--check-untyped-defs