docker compose down pipeline \
	&& docker compose up -d pipeline \
	&& curl --retry 30 --retry-all-errors --retry-delay 1 --silent --show-error http://localhost:8800/health \
	&& curl --retry 120 --retry-all-errors --retry-delay 2 --max-time 5 --silent --show-error --output /dev/null http://localhost:8805/ \
	&& curl --retry 120 --retry-all-errors --retry-delay 2 --max-time 5 --silent --show-error --output /dev/null http://localhost:8803/ \
	&& docker compose run --rm pipeline uv run python -m pytest tests/test_benchmark_parallel.py -vs
