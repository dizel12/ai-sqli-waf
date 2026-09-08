FROM python:3.11-slim
WORKDIR /app
COPY inference_svc/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ml ./ml
COPY inference_svc ./inference_svc
ENV ARTIFACTS_DIR=/app/ml/artifacts ACTIVE_MODEL=cnn BLOCK_THRESHOLD=0.5 INFERENCE_EAGER=1
CMD ["uvicorn", "inference_svc.app:app", "--host", "0.0.0.0", "--port", "9000"]
