FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY eval/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY eval /app/eval

EXPOSE 8002

CMD ["uvicorn", "eval.app:app", "--host", "0.0.0.0", "--port", "8002"]
