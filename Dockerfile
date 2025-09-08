FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
COPY app.py .
COPY models/ ./models/
COPY reports/ ./reports/

RUN apt-get update && apt-get install -y     libgl1-mesa-glx     libglib2.0-0     && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]