import os
import requests
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────
# ✅ FIX — HEADERS كانت تُبنى على module level
#
# المشكلة القديمة:
#   OPENROUTER_API_KEY = os.getenv(...)   ← ممكن يرجع None
#   HEADERS = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
#   → الـ header بيتبنى لما الملف يُستورد مش لما الدالة تتنفذ
#   → لو الـ env مش محمّل وقت الـ import → "Bearer None" → 401 دايمًا
#
# الحل:
#   نقل بناء الـ HEADERS جوه الدالة بحيث يتبنى وقت الاستدعاء
#   وبعد تحميل الـ .env بشكل مضمون
# ─────────────────────────────────────────────────────

OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nex-agi/deepseek-v3.1-nex-n1:free")
BASE_URL         = "https://openrouter.ai/api/v1/chat/completions"


def generate_answer(
    prompt: str,
    max_tokens: int    = 512,
    temperature: float = 0.1,
) -> str:
    # ✅ يُبنى هنا — بعد تحميل .env بشكل مضمون
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("❌ OPENROUTER_API_KEY is not set in environment")

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
