import asyncio
from google import genai
from google.genai import types
import os
from pathlib import Path
from dotenv import load_dotenv

# Load GEMINI_API_KEY from the .env file sitting next to this script.
# Explicit path so it works no matter which directory you run from.
load_dotenv(Path(__file__).with_name(".env"))

if not os.environ.get("GEMINI_API_KEY"):
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Add it to the .env file next to this script "
        "(GEMINI_API_KEY=your-key-here), or set it as an environment variable."
    )
client = genai.Client()

print("=" * 60)
print("Available Google GenAI Models:")
print("=" * 60)
for m in client.models.list():
    actions = getattr(m, 'supported_actions', [])
    print(f"  {m.name}")
    if actions:
        print(f"    Supported actions: {actions}")
print("=" * 60)
print()

MODEL = "gemini-3.8-flash"

def test_streaming_commentary():
    """Test: send a text prompt and stream back a response."""
    print(f"Testing streaming with model: {MODEL}")

    response = client.models.generate_content_stream(
        model=MODEL,
        contents=["You are a live sports commentator. Give a 3-sentence example commentary."]
    )

    print("Streaming response:")
    for chunk in response:
        if chunk.text:
            print(chunk.text, end="", flush=True)
    print("\nDone.")

test_streaming_commentary()
