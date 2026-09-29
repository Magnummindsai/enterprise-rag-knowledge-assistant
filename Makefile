.PHONY: install run test lint eval index docker-up docker-postgres

install:        ## create venv deps (activate your venv first)
	pip install -r requirements-dev.txt

run:            ## start the API + UI on http://localhost:8000
	uvicorn app.main:app --reload --port 8000

test:
	pytest

lint:
	ruff check .

eval:           ## offline metrics; adds RAGAS when OPENAI_API_KEY is set
	python -m eval.run_ragas

index:
	python scripts/build_index.py

docker-up:
	docker compose up --build

docker-postgres:
	docker compose --profile postgres up --build
