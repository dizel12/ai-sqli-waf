FROM python:3.11-slim
WORKDIR /app
COPY log_ui/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY waf_proxy/events.py ./waf_proxy/events.py
COPY waf_proxy/__init__.py ./waf_proxy/__init__.py
COPY log_ui ./log_ui
ENV EVENTS_DB=/data/events.db LOGUI_EAGER=1
CMD ["uvicorn", "log_ui.app:app", "--host", "0.0.0.0", "--port", "8081"]
