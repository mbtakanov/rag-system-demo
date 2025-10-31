import os

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

if ENVIRONMENT == "production":
    API_URL = os.getenv("API_URL", "https://rag-system-production-2613.up.railway.app")
else:
    API_URL = os.getenv("API_URL", "http://localhost:8000")
