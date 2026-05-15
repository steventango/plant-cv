docker compose down pipeline && docker compose up -d pipeline && sleep 10 && docker compose run --rm pipeline uv run python -m pytest tests/test_tracking.py -vs
