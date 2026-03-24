FROM python:3.9

WORKDIR /app

COPY predict.py .
COPY label_encoders.pkl .

RUN pip install mlflow fastapi uvicorn pandas numpy scikit-learn joblib

ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5050
ENV MLFLOW_REGISTRY_URI=http://host.docker.internal:5050

EXPOSE 8000

ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]