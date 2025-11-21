FROM python:3.11

COPY --from=ghcr.io/astral-sh/uv:0.7.14 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
RUN --mount=type=cache,target=/root/.cache/ uv sync --dev

COPY . .

ENV LIGHTNING_USER_ID=abc
ENV LIGHTNING_API_KEY=123

CMD ["uv", "run", "lightning", "deploy", "app/main.py"]
