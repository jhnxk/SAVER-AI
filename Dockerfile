FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt "pandas>=2.2,<3" "numpy>=1.26,<3" gunicorn==26.2.0
COPY ai_server.py serve_submission.py server.ai.html ./
COPY national_hospital_db_departments_hira_updated.xlsx national_hospital_db_realtime_updated.xlsx national_hospital_master.xlsx saver_current_hospital_db.xlsx ./
COPY css/ css/
COPY js/ js/
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --worker-class gthread --threads 8 --no-control-socket serve_submission:app"]
