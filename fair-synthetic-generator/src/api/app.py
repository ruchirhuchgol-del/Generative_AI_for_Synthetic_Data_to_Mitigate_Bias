from fastapi import FastAPI
from src.api.routes import generation, evaluation, health

app = FastAPI(
    title="Fair Synthetic Generator API",
    description="API for generating fair synthetic data and evaluating its quality.",
    version="0.1.0"
)

app.include_router(generation.router, prefix="/generate", tags=["generation"])
app.include_router(health.router, prefix="/health", tags=["health"])

@app.get("/")
async def root():
    return {"message": "Welcome to Fair Synthetic Generator API"}
