FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Use the Aliyun mirror to bypass network restrictions
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

COPY . .
RUN pip install -e . -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

EXPOSE 8000

CMD ["uvicorn", "llm_validator.api:app", "--host", "0.0.0.0", "--port", "8000"]
