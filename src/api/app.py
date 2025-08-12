import logging
import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)

from src.api.routers import comprehensive, region_management, statistics, websocket
from src.services import detection_service, region_service, websocket_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="人体行为检测系统 API",
    description="基于深度学习的实时人体行为检测与分析系统",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.on_event("startup")
async def startup_event():
    logger.info("Starting up the application...")
    # Initialize services
    detection_service.initialize_detection_services()
    app.state.optimized_pipeline = detection_service.optimized_pipeline
    app.state.hairnet_pipeline = detection_service.hairnet_pipeline
    region_service.initialize_region_service(
        os.path.join(project_root, "config", "regions.json")
    )


# Include routers
app.include_router(comprehensive.router, prefix="/api/v1/detect", tags=["Detection"])
app.include_router(
    region_management.router, prefix="/api/v1/management", tags=["Region Management"]
)
app.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])
app.include_router(statistics.router, prefix="/api/v1", tags=["Statistics"])

from fastapi.responses import RedirectResponse


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/frontend/index.html")


# Mount frontend static files
frontend_path = os.path.join(project_root, "frontend")
if os.path.exists(frontend_path):
    app.mount("/frontend", StaticFiles(directory=frontend_path), name="frontend")
    logger.info(f"Static file directory mounted: {frontend_path} to /frontend")
