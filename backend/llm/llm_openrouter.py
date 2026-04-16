import os
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3-0324:free")
BASE_URL         = "https://openrouter.ai/api/v1/chat/completions"


def generate_answer(
    prompt: str,
    max_tokens: int    = 512,
    temperature: float = 0.1,
) -> str:
    # ✅ يُبنى هنا — بعد تحميل .env بشكل مضمون
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(" OPENROUTER_API_KEY is not set in environment")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": "Answer ONLY using the provided context."},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens":  max_tokens,
        "temperature": temperature,
    }
    r = requests.post(BASE_URL, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]
