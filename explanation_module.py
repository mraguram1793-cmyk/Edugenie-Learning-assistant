import sys
import os
_extra = r"D:\PyPackages\Python313\site-packages"
if _extra not in sys.path:
    sys.path.insert(0, _extra)

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    _LOCAL_MODEL_ID = os.getenv("LOCAL_EXPLANATION_MODEL", "MBZUAI/LaMini-Flan-T5-783M")
    explain_tokenizer = AutoTokenizer.from_pretrained(_LOCAL_MODEL_ID)
    explain_model = AutoModelForSeq2SeqLM.from_pretrained(_LOCAL_MODEL_ID)
    _model_loaded = True
except Exception as _e:
    print(f"[WARNING] Local explanation model not available: {_e}")
    _model_loaded = False

def explain_topic(topic: str, api_key: str, model: str = "gemini-2.0-flash") -> str:
    if not _model_loaded:
        # Fallback to Gemini when local model is unavailable
        import google.genai as genai
        try:
            client = genai.Client(api_key=api_key)
            prompt = f"Explain the concept of '{topic}' in a simple and clear way for a school student."
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            return f"[ERROR] Explanation failed: {e}"

    # Use local model
    input_text = f"Explain the concept of '{topic}' in a simple and clear way for a school student."
    inputs = explain_tokenizer(input_text, return_tensors="pt")
    outputs = explain_model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )
    explanation = explain_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return explanation
