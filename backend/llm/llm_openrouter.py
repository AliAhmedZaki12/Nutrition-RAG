import os
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "deepseek/deepseek-chat"  
)

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


def generate_answer(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.1,
) -> str:

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://nutrition-rag--alizaki6797.replit.app",  
        "X-Title": "Nutrition RAG",                                       
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": "Answer ONLY using the provided context."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    r = requests.post(BASE_URL, headers=headers, json=payload, timeout=60)

    #  Debug 
    if r.status_code != 200:
        print("Status:", r.status_code)
        print("Response:", r.text)

    r.raise_for_status()

    return r.json()["choices"][0]["message"]["content"]
