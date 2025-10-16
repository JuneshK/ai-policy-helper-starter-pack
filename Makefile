.PHONY: dev test fmt
dev:
	docker compose up --build

test:
	docker compose run --rm backend pytest -q

fmt:
	docker compose run --rm backend black app
