
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from backend.api.fact_check import fact_check_claim  
from backend.api.claims import classify_claim
from backend.api.tone_intent import detect_tone_and_intent
import json

app = FastAPI()



# Jinja2 templates for rendering HTML pages
templates = Jinja2Templates(directory="frontend")
from pathlib import Path
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR.parent / "frontend" / "assets"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process_claim")
async def process_claim(claim: str = Form(...)):
    # Classify the claim
    classification = classify_claim(claim)
    classification_type = classification.get("category", "Unknown")

    # Detect the tone and intent of the claim
    tone_intent = detect_tone_and_intent(claim)
    tone = tone_intent.get("tone", "Unknown")
    intent = tone_intent.get("intent", "Unknown")

    # fact-check the claim if it's factual and tone is positive
    if classification_type in ["Factual Claim", "Misleading Claim", "Factoid"]  and tone in ["Neutral", "Persuasive", "Humorous"]:
        fact_check_result = fact_check_claim(claim)
        verdict = fact_check_result.get("fact_check_result", "Unknown")
        evidence = fact_check_result.get("evidence", "No evidence available.")
        sources = fact_check_result.get("sources", [])
        reasoning = fact_check_result.get("reasoning", "No reasoning available.")
    else:
        verdict = "Not Fact-Checked"
        evidence = "Claim is either not factual or has a non-positive tone."
        sources = []
        reasoning = "The claim either isn’t factual or has a misleading tone, hence skipped from fact-checking."

    return {
        "claim": claim,
        "classification": classification_type,
        "tone": tone,
        "intent": intent,
        "fact_check_result": verdict,
        "evidence": evidence,
        "sources": sources,
        "reasoning": reasoning
    }
