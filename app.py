import logging
import os
import sys

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse

from config import HOST, PORT, LOG_PATH
from database.db import init_db
from routes.api import router


# =====================================================
# LOGGING
# =====================================================
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger(__name__)


# =====================================================
# APP
# =====================================================
app = FastAPI(
    title="Air Quality AI",
    description="ESP32 → FastAPI → SQLite → AI",
    version="1.0.0",
)


# =====================================================
# STATIC + TEMPLATES
# =====================================================
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

templates = Jinja2Templates(directory="templates")


# =====================================================
# ROUTES
# =====================================================
app.include_router(router)


# =====================================================
# DASHBOARD PAGE
# =====================================================
@app.get("/", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


# =====================================================
# STARTUP
# =====================================================
@app.on_event("startup")
async def startup():
    logger.info("Starting Air Quality AI server...")
    init_db()
    logger.info("Database initialized.")
    logger.info(f"Server ready at http://{HOST}:{PORT}")
    logger.info("Docs at http://localhost:8000/docs")


# =====================================================
# RUN
# reload_dirs: chỉ watch folder chứa code Python
# KHÔNG watch database/, saved_models/, logs/, data/
# tránh reload loop khi ESP32 ghi dữ liệu liên tục
# =====================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=HOST,
        port=PORT,
        reload=True,
        reload_dirs=[
            "routes",
            "services",
            "ml",
            "models",
            "utils",
            "templates",
            "static",
        ],
        log_level="info"
    )