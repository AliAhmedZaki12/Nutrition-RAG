# debug_env_path.py
import os
from dotenv import find_dotenv, load_dotenv

path = find_dotenv()
print("Loaded .env from:", path)

load_dotenv()
print("Loaded OPENROUTER_API_KEY:", os.getenv("OPENROUTER_API_KEY"))
