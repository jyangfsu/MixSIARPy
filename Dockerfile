FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# A non-root user is required for a safer public deployment.  Its home
# directory is writable by Streamlit and matplotlib at runtime.
RUN useradd --create-home --uid 1000 appuser

COPY . /app
RUN python -m pip install --upgrade pip && \
    python -m pip install ".[bayes,gui]" && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/_stcore/health', timeout=3)" || exit 1

CMD ["streamlit", "run", "mixsiarpy/gui/app.py", "--server.address=0.0.0.0", "--server.port=7860", "--server.fileWatcherType=none"]
