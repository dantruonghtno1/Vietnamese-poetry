FROM python:3.8-slim

# WORKDIR /app

COPY ./ ./

RUN pip install --upgrade pip

RUN pip install torch==1.7.1 Flask flask-cors transformers

CMD ["python3", "app.py"]