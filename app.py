from api.main import app

# This entrypoint exposes the FastAPI app instance from the api package
# so that uvicorn app:app --reload can resolve and run it correctly.