import os
import json
import requests
from dotenv import load_dotenv
from backend.langchain_tools import llm, deepseek_tool 
import re
from backend.api.claims import classify_claim
from backend.api.tone_intent import detect_tone_and_intent
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
import wikipedia
import logging
from bs4 import BeautifulSoup
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
USER_AGENT = {"User-Agent": "Mozilla/5.0"}

# Initialising requests session to reuse connections
session = requests.Session()
session.headers.update(USER_AGENT)

# nltk data for tokenization once
nltk.download('punkt')

logging.basicConfig(level=logging.INFO)

fact_check_prompt = PromptTemplate.from_template("""
You are an expert fact-checker. Your task is to determine whether the following claim is true or false based on the information available from external sources (Wikipedia and Google search). The claim has already been classified into categories and analyzed for tone/intent.

Claim: "{claim}"
Classification: "{classification}"
Tone: "{tone}"
Intent: "{intent}"
Wikipedia Evidence: "{wikipedia_evidence}"
Serper Evidence: "{serper_evidence}"

Fact-checking Task:
- Based on the evidence from Wikipedia and Serper search results, classify the claim as:
  1. True
  2. False
  3. Uncertain (when there is insufficient evidence)

Respond in JSON format:
{{
  "claim": "{claim}",
  "classification": "{classification}",
  "tone": "{tone}",
  "intent": "{intent}",
  "fact_check_result": "<True/False/Uncertain/Misleading/Exaggerated/Partially True/Inconclusive:: >",
  "evidence": "<evidence supporting or contradicting the claim>",
  "sources": [
    {{
      "source": "Wikipedia",
      "url": "<url>"
    }},
    {{
      "source": "Search result",
      "url": "<serper_url>"
    }},
    {{
      "source": "LLM",
      "url": "<llm_inferred_url>"
    }},
    ...
  ],
  "reasoning": "<reasoning for the fact-checking result>"
}}
""")

fact_check_chain = fact_check_prompt | llm

# Wikipedia Search Function
def search_wikipedia(query):
    try:
        results = wikipedia.search(query, results=5)
        if not results:
            return {"error": "No results found on Wikipedia."}
        
        summaries = []
        for title in results:
            try:
                page = wikipedia.page(title)
                summaries.append({
                    "title": title, 
                    "summary": page.summary, 
                    "url": page.url
                })
            except wikipedia.exceptions.DisambiguationError as e:
                summaries.append({"error": f"Disambiguation: {str(e)}"})
            except Exception as e:
                summaries.append({"error": str(e)})
        return summaries
    except Exception as e:
        return {"error": str(e)}

# Fetch search results from Serper API
def fetch_search_results(query, sentences_count=3):
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY}
    data = {"q": query, "num": 10}

    try:
        response = session.post(url, json=data, headers=headers)
        response.raise_for_status()
        results = response.json()

        if not results.get("organic"):
            return "Error: No search results found."

        sources = []
        for result in results.get("organic", [])[:5]:  
            source_url = result.get("url") or result.get("link")
            sources.append(source_url)
        
        summary = fetch_and_summarize(sources[0], sentences_count) 
        return summary, sources
    except requests.exceptions.RequestException as e:
        return f"Error fetching search results: {e}"

# Summarize content from a Webpage
def fetch_and_summarize(url, sentences_count=3):
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = "\n".join([p.get_text() for p in paragraphs])

        if not text.strip():
            return "Error: No readable content found on the page."

        return summarize_text(text, sentences_count)
    except requests.exceptions.RequestException as e:
        return f"Error: Failed to fetch webpage ({str(e)})"
    except Exception as e:
        return f"Error: {str(e)}"

# Summarize text using LsaSummarizer
def summarize_text(text, sentences_count=3):
    if not text.strip():
        return "Error: No text provided for summarization."

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count)

    return " ".join(str(sentence) for sentence in summary)



def fact_check_claim(claim_text: str) -> dict:
    try:
        classification = classify_claim(claim_text)
        tone_intent = detect_tone_and_intent(claim_text)

        classification_type = classification.get("category", "Unknown")
        tone = tone_intent.get("tone", "Unknown")
        intent = tone_intent.get("intent", "Unknown")

        wikipedia_results = search_wikipedia(claim_text)
        serper_summary, serper_sources = fetch_search_results(claim_text)

        wikipedia_evidence = " ".join([r["summary"] for r in wikipedia_results if "summary" in r])

        sources = []

        for r in wikipedia_results:
            if "url" in r:
                sources.append({"source": "Wikipedia", "url": r["url"]})
        for url in serper_sources:
            sources.append({"source": "Serper", "url": url})

        try:
            fact_check_result = fact_check_chain.invoke({
                "claim": claim_text,
                "classification": classification_type,
                "tone": tone,
                "intent": intent,
                "wikipedia_evidence": wikipedia_evidence,
                "serper_evidence": serper_summary
            })

            result = json.loads(fact_check_result.content.strip())
            result["sources"] = sources
            return result

        except Exception as primary_error:
            logging.warning(f"Primary LLM failed: {primary_error}. Falling back to DeepSeek.")

            deepseek_prompt = fact_check_prompt.template.format(
                claim=claim_text,
                classification=classification_type,
                tone=tone,
                intent=intent,
                wikipedia_evidence=wikipedia_evidence,
                serper_evidence=serper_summary
            )

            deepseek_result = deepseek_tool.invoke({"input": deepseek_prompt})
            logging.info(f"Raw DeepSeek Output: {deepseek_result}")

            cleaned_output = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", deepseek_result.strip())

            result = json.loads(cleaned_output)
            result["sources"] = sources
            return result

    except Exception as final_error:
        logging.error(f"Total failure in fact_check_claim: {final_error}")
        return {"error": f"Fact-checking failed due to: {str(final_error)}"}
