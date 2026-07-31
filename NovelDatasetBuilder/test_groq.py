"""Quick Groq API connectivity test."""
import os, sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path('.env'))

from openai import OpenAI
key = os.getenv('GROQ_API_KEY', '')
if not key:
    print('ERROR: GROQ_API_KEY not found in .env')
    sys.exit(1)

client = OpenAI(api_key=key, base_url='https://api.groq.com/openai/v1')
print('Testing Groq API with llama-3.3-70b-versatile ...')

resp = client.chat.completions.create(
    model='llama-3.3-70b-versatile',
    messages=[
        {
            'role': 'system',
            'content': 'You are an expert literature professor. Output ONLY valid JSON arrays.'
        },
        {
            'role': 'user',
            'content': (
                'Generate 2 instruction-response pairs about Sophocles Antigone.\n'
                'Format: [{"instruction":"...","response":"..."},{"instruction":"...","response":"..."}]\n'
                'Output ONLY the JSON array.'
            )
        }
    ],
    max_tokens=600,
    temperature=0.7
)

content = resp.choices[0].message.content or ''
print('Response received:')
print(content[:800])
print()

# Try to parse it
import json
try:
    import re
    arr_str = content[content.find('['):content.rfind(']')+1]
    data = json.loads(arr_str)
    print(f'[OK] Groq API works! Parsed {len(data)} pairs successfully.')
    for i, p in enumerate(data, 1):
        print(f'  Pair {i}: {p.get("instruction","")[:60]}...')
except Exception as e:
    print(f'[WARN] Parsing issue: {e} — but API connection itself works.')
