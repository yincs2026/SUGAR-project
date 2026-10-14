FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e '.[analysis]'
COPY configs ./configs
COPY tests ./tests
COPY scripts ./scripts
ENTRYPOINT ["sugar"]
CMD ["--help"]
