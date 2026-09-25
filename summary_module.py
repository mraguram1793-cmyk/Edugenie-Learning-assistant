import sys
_extra = r"D:\PyPackages\Python313\site-packages"
if _extra not in sys.path:
    sys.path.insert(0, _extra)

import google.genai as genai

def summarize_text(text: str, api_key: str, model: str = "gemini-2.0-flash") -> str:
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"Summarize the following text in simple language:\n\n{text}"
        response = client.models.generate_content(
            model=model,
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"[ERROR] Summary failed: {e}"
