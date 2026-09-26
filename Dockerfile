FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
COPY pyproject.toml uv.lock ./
COPY hermes_trading ./hermes_trading
COPY state ./state
COPY state ./state_seed
RUN uv sync --frozen
ENV HERMES_TRADING_MODE=paper
# Reflect in-process every 30 min via Gemini (free tier); needs LLM_API_KEY set on the service
ENV HERMES_REFLECT=llm
CMD ["uv", "run", "python", "-m", "hermes_trading.run"]
