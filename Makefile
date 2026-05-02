PYTHON_EXEC = uv run python3
UV = uv
MAIN_FILE = main.py

all: install run

install:
	@echo 'install and sync dependancies'
	$(UV) sync

run:
	@echo 'lunch R.A.G'
	$(PYTHON_EXEC) $(MAIN_FILE)


clean:
	@echo 'remove artefact files'
	rm -rf __pycache__ .venv .uv

debug:
	uv run python -m pdb -m

lint:
	@echo 'check quality code (mypy/flake8) norme'
	flake8 . --exclude .venv
	mypy . --strict --warn-return-any \
	--warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs \
	--check-untyped-defs