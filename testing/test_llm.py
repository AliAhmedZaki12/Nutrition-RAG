import os
import requests
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("OPENROUTER_API_KEY")

url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {key}",
    "X-API-Key": key,
    "HTTP-Referer": "http://localhost",
    "User-Agent": "TestingScript/1.0",
    "Content-Type": "application/json"
}

payload = {
    "model": "nex-agi/deepseek-v3.1-nex-n1:free",
    "messages": [
        {"role": "user", "content": "hello from python"}
    ]
}

print("Sending request...\n")

response = requests.post(url, headers=headers, json=payload)
print("Status:", response.status_code)
print("Response:", response.text)
