import os
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# 🎯 Primary model (your choice)
PRIMARY_MODEL = "meta-llama/llama-3.2-3b-instruct:free"

# 🔁 Fallback models (important for stability)
FALLBACK_MODELS = [
    "mistralai/mistral-small-2603",
    "deepseek/deepseek-chat",
]


def call_openrouter(model: str, prompt: str, api_key: str) -> str:
    """
    Single model call to OpenRouter with safe error handling
    """

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://nutrition-rag--alizaki6797.replit.app",
        "X-Title": "RAG API",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Answer ONLY using the provided context. If not found, say you don't know."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "max_tokens": 512
    }

    try:
        r = requests.post(BASE_URL, headers=headers, json=payload, timeout=30)

        # 🔍 Debug if something goes wrong
        if r.status_code != 200:
            print(f"[ERROR] Model: {model}")
            print("Status:", r.status_code)
            print("Response:", r.text)
            return None

        return r.json()["choices"][0]["message"]["content"]

    except Exception as e:
        print(f"[EXCEPTION] Model: {model} -> {e}")
        return None


def generate_answer(prompt: str) -> str:
    """
    Main function with fallback system (production-safe)
    """

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing")

    # 🎯 Try primary model first
    response = call_openrouter(PRIMARY_MODEL, prompt, api_key)
    if response:
        return response

    # 🔁 Try fallback models
    for model in FALLBACK_MODELS:
        response = call_openrouter(model, prompt, api_key)
        if response:
            return response

    # ❌ If everything fails
    return "Sorry, all models failed. Please try again later."
