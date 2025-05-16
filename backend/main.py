from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from backend.api import routes_papers, routes_qa

app = FastAPI(title="RAG Research Paper Chatbot")

# Mount static files
app.mount("/static", StaticFiles(directory="backend/static"), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory="backend/templates")

# Optional homepage redirect to /papers
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("search_form.html", {"request": request})

# Include routers
app.include_router(routes_papers.router, prefix="/papers")
app.include_router(routes_qa.router, prefix="/qa")
