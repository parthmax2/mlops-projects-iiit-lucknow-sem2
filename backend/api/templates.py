from fastapi.templating import Jinja2Templates
from fastapi import Request

templates = Jinja2Templates(directory="backend/templates")

def render_template(template_name: str, request: Request, context: dict = {}):
    return templates.TemplateResponse(template_name, {"request": request, **context})
