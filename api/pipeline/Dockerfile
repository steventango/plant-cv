FROM python:3.11

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

COPY --from=ghcr.io/astral-sh/uv:0.7.14 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml pyproject.toml
RUN --mount=type=cache,target=/root/.cache/ uv sync --no-dev

COPY . .

CMD ["uv", "run", "flask", "--app", "app/main.py", "run", "--host", "0.0.0.0", "--port", "8000"]
