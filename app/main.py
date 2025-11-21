import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse, FileResponse
from fastapi.templating import Jinja2Templates

from app.llm_router import router as llm_router
from app.qdrant_init import (
    init_qdrant_with_materials,
    list_material_files,
    read_material_content,
)

MATERIALS_DIR = os.path.join(os.path.dirname(__file__), "materials")

app = FastAPI(title="RAG Demo App")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

app.include_router(llm_router, prefix="/llm", tags=["llm"])


@app.on_event("startup")
def startup_event():
    init_qdrant_with_materials()


@app.get("/")
async def index(request: Request):
    files = list_material_files()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "files": files},
    )


@app.get("/materials")
async def get_materials():
    return {"files": list_material_files()}


@app.get("/materials/{filename}", response_class=PlainTextResponse)
async def get_material_content(filename: str):
    files = list_material_files()
    if filename not in files:
        raise HTTPException(status_code=404, detail="File not found")
    ext = filename.lower().split(".")[-1]
    if ext == "pdf":
        return FileResponse(os.path.join(MATERIALS_DIR, filename), media_type="application/pdf")
    else:
        content = read_material_content(filename)
        return content


