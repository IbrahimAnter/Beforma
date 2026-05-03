FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MEAL_FEEDBACK_PATH=/data/meal_feedback.json

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /data

EXPOSE 8000
CMD ["uvicorn", "beforma_nutrition_api:app", "--host", "0.0.0.0", "--port", "8000"]
