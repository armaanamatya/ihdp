FROM python:3.10-slim
WORKDIR /app
COPY serve/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY serve/ serve/
COPY artifacts/dragonnet.onnx artifacts/meta.json artifacts/
ENV ARTIFACT_DIR=/app/artifacts
EXPOSE 8000
USER nobody
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
