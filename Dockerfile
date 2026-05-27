FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predict.py .

# Model artifacts (*.joblib) are NOT baked into the image.
# Mount them at runtime or COPY them here after training:
#   COPY *.joblib .
# Example runtime mount:
#   docker run -v /path/to/models:/app -p 8000:8000 <image>

EXPOSE 8000

ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]
