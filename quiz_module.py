import sys
_extra = r"D:\PyPackages\Python313\site-packages"
if _extra not in sys.path:
    sys.path.insert(0, _extra)

import google.genai as genai
import re
import json

def clean_json_block(text):
    # Remove Markdown ```json code fences
    return re.sub(r"```(?:json)?\n(.*?)```", r"\1", text, flags=re.DOTALL).strip()

def generate_quiz(text: str, api_key: str, model: str = "gemini-2.0-flash") -> list:
    try:
        client = genai.Client(api_key=api_key)

        prompt = f"""
You are a quiz generator.

From the following passage, create 3 multiple-choice questions. Each question should include:
- A "question"
- A list of 4 "options"
- A correct "answer" that must exactly match one of the options.

Format your output as **valid JSON**, like this:
[
  {{
    "question": "What is ...?",
    "options": ["A", "B", "C", "D"],
    "answer": "A"
  }}
]

Passage:
{text}
"""
        response = client.models.generate_content(
            model=model,
            contents=prompt
        )
        quiz_text = response.text.strip()

        # Clean markdown code blocks if any
        cleaned_text = clean_json_block(quiz_text)
        return json.loads(cleaned_text)

    except Exception as e:
        return [{"error": f"Quiz generation failed: {e}"}]
