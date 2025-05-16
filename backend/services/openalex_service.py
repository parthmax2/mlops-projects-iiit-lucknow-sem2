# backend/services/openalex_service.py

import requests
from typing import Optional

OPENALEX_API_URL = "https://api.openalex.org/works"


def get_citation_count(title: str) -> Optional[int]:
    """
    Get citation count for a paper using OpenAlex by searching with the title.

    Args:
        title (str): The title of the paper.

    Returns:
        Optional[int]: The citation count if found, else 0.
    """
    try:
        url = f"{OPENALEX_API_URL}?search={requests.utils.quote(title)}&per_page=1"
        response = requests.get(url)
        data = response.json()
        return data["results"][0].get("cited_by_count", 0) if data["results"] else 0
    except:
        return 0
