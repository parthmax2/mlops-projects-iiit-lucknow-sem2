from langchain_openai import ChatOpenAI  
from langchain.agents import Tool
from langchain_community.utilities import WikipediaAPIWrapper
import requests
import os
from backend.config import OPENAI_API_KEY, SERPER_API_KEY

llm = ChatOpenAI(
    model="gpt-3.5-turbo",                
    temperature=0,                
    openai_api_key=OPENAI_API_KEY  
)


import wikipedia
from langchain.tools import Tool

def search_wikipedia(query):
    try:
        results = wikipedia.search(query, results=10)  # Fetch top 10 results
        if not results:
            return {"error": "No results found on Wikipedia."}
        
        summaries = []
        for title in results:
            try:
                page = wikipedia.page(title)
                summary = page.summary
                url = page.url
                summaries.append({"title": title, "summary": summary, "url": url})
            except wikipedia.exceptions.DisambiguationError as e:
                summaries.append({"error": f"Disambiguation: {str(e)}"})
            except wikipedia.exceptions.HTTPTimeoutError:
                summaries.append({"error": "Wikipedia request timed out."})
            except Exception as e:
                summaries.append({"error": str(e)})
        return summaries
    except Exception as e:
        return {"error": str(e)}

# Define the Wikipedia Tool
wiki_tool = Tool.from_function(
    func=search_wikipedia,
    name="WikipediaAPI",
    description="Fetch summaries from Wikipedia based on search query"
)


from langchain.tools import Tool

def fetch_search_results(query):
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    
    data = {
        "q": query,
        "num": 10  
    }

    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Serper API error: {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}

# Define the Serper Tool
serper_tool = Tool.from_function(
    func=fetch_search_results,
    name="SerperAPI",
    description="Search the web using the Serper API"
)


from together import Together
from backend.config import TOGETHER_API_KEY  

client = Together(api_key=TOGETHER_API_KEY)

def deepseek_v3_query(query):
    try:
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",  # DeepSeek-V3 model from Together
            messages=[{"role": "user", "content": query}],
            stream=True
        )

        result = ""
        for token in response:
            if hasattr(token, 'choices'):
                result += token.choices[0].delta.content
        return result
    except Exception as e:
        return {"error": f"DeepSeek-V3 API error: {str(e)}"}

# Define the DeepSeek-V3 Tool
deepseek_tool = Tool.from_function(
    func=deepseek_v3_query,
    name="DeepSeekV3",
    description="Query DeepSeek-V3 AI model from Together for advanced text completions"
)


# 5. Group tools together
search_tools = [wiki_tool, serper_tool, deepseek_tool]
