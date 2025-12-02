FROM python:3.11

RUN apt update && apt install -y ffmpeg libsm6 libxext6 

COPY --from=ghcr.io/astral-sh/uv:0.7.14 /uv /uvx /bin/

ARG USERNAME=docker
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

WORKDIR /app

CMD ["uv", "run", "flask", "--app", "app/main.py", "run", "--host", "0.0.0.0", "--port", "8800"]
