FROM python:3.12

# git is needed to install the native sam3 package from its pinned git rev.
# The package is pure-Python (no CUDA build), so the torch wheel's bundled CUDA
# runtime is sufficient — no CUDA devel image required.
RUN apt update && apt install -y ffmpeg libsm6 libxext6 git

COPY --from=ghcr.io/astral-sh/uv:0.7.14 /uv /uvx /bin/

# Use the image's Python 3.12 (which has prebuilt wheels for numpy<2 etc.) instead of
# letting uv fetch a newer managed Python (3.13+) that would force source builds.
ENV UV_PYTHON=3.12 \
    UV_PYTHON_PREFERENCE=only-system \
    UV_LINK_MODE=copy

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

CMD ["uv", "run", "python", "app/main.py"]
