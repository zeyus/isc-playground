FROM ghcr.io/astral-sh/uv:python3.13-alpine

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"


EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/isc/_stcore/health

ENTRYPOINT []

CMD ["uv", "run", "streamlit", "run", "isc-playground.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.baseUrlPath=isc"]
