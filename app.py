import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse

from config import HOST, PORT, LOG_PATH
from database.db import init_db
from routes.api import router
import mqtt_client


# =====================================================
# LOGGING
# =====================================================
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logger = logging.getLogger(__name__)


# =====================================================
# LIFESPAN  (thay thế on_event deprecated)
# =====================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    logger.info("Starting Air Quality AI server...")
    init_db()
    logger.info("Database initialized.")
    mqtt_client.start()
    logger.info(f"Server ready at http://{HOST}:{PORT}")
    logger.info("Docs at http://localhost:8000/docs")

    yield  # server đang chạy

    # --- SHUTDOWN ---
    logger.info("Shutting down Air Quality AI server...")
    mqtt_client.stop()


# =====================================================
# APP
# =====================================================
app = FastAPI(
    title="Air Quality AI",
    description="ESP32 → MQTT → FastAPI → SQLite → AI",
    version="1.0.0",
    lifespan=lifespan,
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
# RUN
# reload_dirs: chỉ watch folder chứa code Python
# KHÔNG watch database/, saved_models/, logs/, data/
# tránh reload loop khi ESP32 ghi dữ liệu liên tục
# =====================================================
if __name__ == "__main__":
    import uvicorn

    # Luôn tắt reload — tránh vòng lặp khi logs/database thay đổi liên tục.
    # Nếu cần dev reload: uvicorn app:app --reload --reload-exclude 'logs/*' ...
    uvicorn.run(
        "app:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info",
    )