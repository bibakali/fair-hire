pipeline {
    agent any

    environment {
        IMAGE_NAME = "fair-hire-app"
        IMAGE_TAG  = "latest"
    }

    stages {

        stage('Checkout') {
            steps {
                echo '📥 Récupération du code...'
                checkout scm
            }
        }

        stage('Setup Python') {
            steps {
                echo '🐍 Installation des dépendances...'
                sh '''
                    python3 -m venv env
                    . env/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Tests') {
            steps {
                echo '🧪 Lancement des tests...'
                sh '''
                    . env/bin/activate
                    pytest tests/ -v
                '''
            }
        }

        stage('Build Docker') {
            steps {
                echo '🐳 Build de l image Docker...'
                sh "docker build -t ${IMAGE_NAME}:${IMAGE_TAG} ."
            }
        }

        stage('Deploy') {
            steps {
                echo '🚀 Déploiement...'
                sh '''
                    docker-compose down || true
                    docker-compose up -d
                '''
            }
        }
    }

    post {
        success {
            echo '✅ Pipeline terminé avec succès'
        }
        failure {
            echo '❌ Pipeline échoué — vérifier les logs'
        }
    }
}