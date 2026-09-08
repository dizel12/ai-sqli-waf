FROM python:3.11-slim
WORKDIR /app
COPY waf_proxy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ml/data ./ml/data
COPY ml/__init__.py ./ml/__init__.py
COPY waf_proxy ./waf_proxy
ENV EVENTS_DB=/data/events.db
CMD ["uvicorn", "waf_proxy.app:app", "--host", "0.0.0.0", "--port", "8080"]
