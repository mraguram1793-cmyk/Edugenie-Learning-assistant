import sys
_extra = r"D:\PyPackages\Python313\site-packages"
if _extra not in sys.path:
    sys.path.insert(0, _extra)

import google.genai as genai

def get_learning_recommendations(topic: str, api_key: str, model: str = "gemini-2.0-flash") -> str:
    try:
        client = genai.Client(api_key=api_key)

        prompt = f"""
You are an AI tutor. The student wants to learn about: {topic}.
Suggest a structured and adaptive learning path including key topics, order of learning, and resources (videos, articles, books).
Include beginner, intermediate, and advanced levels if needed.
"""
        response = client.models.generate_content(
            model=model,
            contents=prompt
        )

        if response and response.text:
            return response.text.strip()
        else:
            return "Could not extract content from Gemini response."

    except Exception as e:
        return f"[ERROR] Learning path generation failed: {e}"
