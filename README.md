Sure! Here's a polished and detailed README for your Falcon project, incorporating the fallback model Deepseek and emphasizing the Gen AI nature:

---

# Falcon: Fake News Analysis and Language Comprehension for Online Neutrality

**Falcon** is an advanced Gen AI-powered system designed to analyze, verify, and classify online claims to promote digital content neutrality. Leveraging state-of-the-art language models, it classifies user-submitted text, detects tone and intent, verifies facts using multiple data sources, and generates an informed verdict about claim credibility.

---

## Features

* **Claim Classification**: Classifies input text into categories like Factual Claim, Opinion, Irrelevant Talk, or Vague/Incomplete.
* **Tone and Intent Detection**: Analyzes the emotional tone and intent behind the claim (e.g., neutral, persuasive, humorous).
* **Fact Verification**: Verifies factual claims by searching reliable sources like Google (via Serper API) and Wikipedia.
* **Verdict Generation**: Produces a clear verdict on the claim’s truthfulness based on gathered evidence and reasoning.
* **Fallback Language Model**: Uses OpenAI’s GPT-4 as the primary LLM and automatically switches to **Deepseek** as a fallback model to ensure high availability and robustness.
* **Simple Web Interface**: Easy-to-use frontend using HTML, CSS, and JavaScript for seamless user interaction.
* **FastAPI Backend**: Fast, lightweight API server handling all processing with asynchronous endpoints.

---

## Tech Stack

* **Backend**: FastAPI (Python)
* **Frontend**: HTML, CSS, JavaScript
* **Language Models**: OpenAI GPT-4 (primary), Deepseek (fallback)
* **APIs**: Serper API (Google Search), Wikipedia API
* **Prompt Engineering**: Carefully crafted templates for accurate and reliable results
* **Deployment**: Docker containerization (optional)

---

## Project Structure

```
falcon/
├── backend/
│   ├── api/
│   │   ├── claims.py           # Claim classification logic
│   │   ├── fact_check.py       # Fact verification logic
│   │   ├── tone_intent.py      # Tone and intent detection logic
│   ├── langchain_tools.py      # LangChain prompt templates and LLM wrappers
│   ├── config.py               # API keys and config variables
│   └── main.py                 # FastAPI app entry point
├── frontend/
│   ├── index.html              # Main UI page
│   ├── assets/                 # CSS, JS, images
├── Dockerfile                  # Docker container definition (optional)
└── README.md                   # This file
```

---

## Setup Instructions

### Prerequisites

* Python 3.9+
* FastAPI
* Uvicorn (ASGI server)
* OpenAI API key
* Serper API key
* Deepseek API key (for fallback)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/falcon.git
cd falcon
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure API keys in `backend/config.py`:

```python
OPENAI_API_KEY = "your_openai_api_key"
SERPER_API_KEY = "your_serper_api_key"
DEEPSEEK_API_KEY = "your_deepseek_api_key"
```

5. Run the FastAPI server:

```bash
uvicorn backend.main:app --port 8000
```

6. Open your browser and visit `http://localhost:8000` to use the Falcon app.

---

## Usage

* Enter a claim or statement in the input form.
* The system classifies the claim, detects tone and intent, performs fact verification if applicable, and returns a detailed verdict with supporting evidence.
* The fallback model Deepseek is automatically used if GPT-4 is unavailable, ensuring consistent performance.

---

## Contribution

Contributions, bug reports, and feature requests are welcome! Feel free to open issues or pull requests.
