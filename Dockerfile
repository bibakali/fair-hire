# Image de base Python légère
FROM python:3.12-slim

# Répertoire de travail dans le conteneur
WORKDIR /app

# Copie des dépendances en premier (optimisation cache Docker)
COPY requirements.txt .

# Installation des dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copie du reste du projet
COPY . .

# Port exposé par Streamlit
EXPOSE 8501

# Variables d'environnement
ENV PYTHONUNBUFFERED=1

# Commande de lancement
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]