# config.py
import os
from dotenv import load_dotenv
load_dotenv()

# Gemini model used by every agent. Set GEMINI_MODEL in .env to change it;
# the default applies if .env doesn't set one.
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
