FROM python:3.12.3-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# Use a reliable mirror, increase timeout, and retries
RUN pip install --no-cache-dir --default-timeout=100 --retries=5 \
    --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "app.service_recruiter_agent:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
