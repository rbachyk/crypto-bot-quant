# Single application image used by every Python service (Appendix B.3/B.12).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# uv for fast, reproducible installs (Appendix C).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first for layer caching.
COPY pyproject.toml uv.lock* README.md ./
RUN uv sync --frozen --no-dev || uv sync --no-dev

COPY . .

# Runtime data directories that are ALSO shared named-volume mount points (the data lake,
# artifacts, backups, reports). They must exist and be owned by the runtime user before the
# volume is mounted, so the mount inherits writable ownership (else a root-owned mount blocks
# the non-root user from writing the downloaded lake — making every reader find an empty lake).
RUN mkdir -p /app/var/datalake /app/var/artifacts /app/var/backups /app/reports

# Non-root runtime user.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
