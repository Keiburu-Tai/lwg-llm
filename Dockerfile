# 
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# 
WORKDIR /app

COPY . .

RUN pip install transformers fastapi uvicorn accelerate

EXPOSE 8000
# 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]