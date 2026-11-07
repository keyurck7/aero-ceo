# AERO-CEO v2 — one image definition, extras chosen per service.
#   docker build --build-arg EXTRAS=postgres,mcp      -t aero-evidence-mcp .
#   docker build --build-arg EXTRAS=postgres,extract  -t aero-worker .
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS build
ARG EXTRAS=postgres,mcp
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install ".[${EXTRAS}]"

FROM python:${PYTHON_VERSION}-slim AS runtime
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    AERO_LOG_JSON=true AERO_SEEDS_DIR=/app/seeds AERO_MCP_HOST=0.0.0.0
RUN groupadd --system aero && useradd --system --gid aero --home /app aero
WORKDIR /app
COPY --from=build /opt/venv /opt/venv
COPY seeds ./seeds
USER aero
EXPOSE 8101
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8101/healthz', timeout=4).status == 200 else 1)"
ENTRYPOINT ["aero"]
CMD ["serve", "evidence"]
