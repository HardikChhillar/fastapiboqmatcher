from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
from pathlib import Path

# Import all routes from main0
from backup_main import (
    app as main_app,
    get_dataset_info,
    get_token_usage,
    clear_cache,
    update_dataset,
    query_rate
)

# Create the main application
app = FastAPI(title="Rate Matching System")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the absolute path to the frontend directory
frontend_dir = Path(__file__).parent.parent / "frontend"

# Mount static files first (but not at root)
app.mount("/static", StaticFiles(directory=str(frontend_dir), html=True), name="static")

# Include all routes from main0
app.get("/dataset-info")(get_dataset_info)
app.get("/token-usage")(get_token_usage)
app.post("/clear-cache")(clear_cache)
app.post("/update-dataset")(update_dataset)
app.post("/query-rate")(query_rate)

# Serve index.html at root
@app.get("/")
async def serve_index():
    index_path = frontend_dir / "index copy.html"
    if index_path.exists():
        with open(index_path) as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return HTMLResponse(content="Frontend not found", status_code=404)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 