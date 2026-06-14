from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="App Store Review Analyzer",
    description="Collects, analyzes, and provides insights from App Store reviews.",
    version="1.0.0",
)

app.include_router(router)