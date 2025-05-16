from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from backend.services.arxiv_service import search_arxiv, download_pdf
from backend.services.openalex_service import get_citation_count
from backend.api.templates import render_template  # helper for rendering Jinja2 templates

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    """Render the initial paper search form."""
    return render_template("search_form.html", request)


@router.post("/search", response_class=HTMLResponse)
async def search_papers(request: Request, query: str = Form(...)):
    """Search ArXiv papers and display ranked results with citation counts."""
    papers = search_arxiv(query, max_results=10)

    # Add citation counts from OpenAlex
    for paper in papers:
        paper["citations"] = get_citation_count(paper["title"])

    # Sort by citation count descending
    sorted_papers = sorted(papers, key=lambda p: p["citations"], reverse=True)

    return render_template("search_results.html", request, {"papers": sorted_papers})


@router.post("/download", response_class=RedirectResponse)
async def download_selected_paper(
    request: Request,
    title: str = Form(...),
    pdf_url: str = Form(...)
):
    """Download the selected paper and redirect to QA page."""
    saved_path = download_pdf(pdf_url, title)

    if not saved_path:
        # Redirect back to homepage if download fails
        return RedirectResponse(url="/", status_code=302)

    return RedirectResponse(url=f"/qa?title={title}", status_code=302)
