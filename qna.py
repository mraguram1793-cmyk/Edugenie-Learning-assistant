import sys
_extra = r"D:\PyPackages\Python313\site-packages"
if _extra not in sys.path:
    sys.path.insert(0, _extra)

import google.genai as genai

def answer_question_with_gemini(question: str, api_key: str, model: str = "gemini-3-flash-preview") -> str:
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=question
        )
        return response.text.strip()
    except Exception as e:
        return f"[ERROR] QnA failed: {e}"
