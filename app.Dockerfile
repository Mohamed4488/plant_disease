FROM plant_disease_app:latest

WORKDIR /app

COPY . .

CMD ["streamlit", "run", "app.py"]