# smoke_test.py
from google import genai
from google.genai import types
from dotenv import load_dotenv
from config import MODEL
load_dotenv()
 
client = genai.Client()  # reads GEMINI_API_KEY from your .env
response = client.models.generate_content(
    model=MODEL,
    contents="Say hello in one sentence.",
    config=types.GenerateContentConfig(max_output_tokens=250),
)
print(response.text)
print(f"Input tokens: {response.usage_metadata.prompt_token_count}")
print(f"Output tokens: {response.usage_metadata.candidates_token_count}")