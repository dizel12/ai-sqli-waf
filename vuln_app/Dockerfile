FROM python:3.11-slim
WORKDIR /app
COPY vuln_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY vuln_app ./vuln_app
ENV APP_DB=/app/app.db SECURE_MODE=0
RUN python -m vuln_app.seed /app/app.db
CMD ["gunicorn", "-b", "0.0.0.0:8000", "vuln_app.app:app"]
