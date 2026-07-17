.PHONY: install test lint run api docs build-docs sample

install:
	python -m pip install -e .[dev,docs,hub]

test:
	pytest -q

lint:
	ruff check .

run:
	python app.py

api:
	uvicorn api:app --reload --port 8000

docs:
	mkdocs serve

build-docs:
	mkdocs build --strict

sample:
	python -m fraudforge --output artifacts/sample --rows 900 --rounds 7 --seed 42
