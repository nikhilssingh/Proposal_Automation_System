# railway/fastapi.Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
ENV OPENAI_API_KEY=${OPENAI_API_KEY}
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
