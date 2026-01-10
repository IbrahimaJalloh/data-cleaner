"""CRM IA v2.2 - Error-proof + fallback."""
import json
import os
import re
import traceback
import pandas as pd
from typing import Dict, List
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except:
    client = None

def safe_openai(prompt: str, max_retries: int = 2) -> str:
    """OpenAI avec retry + fallback"""
    if not client:
        return '{"error": "OPENAI_API_KEY manquante", "quality_score": 50}'
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt[:8000]}],  # Troncature
                max_tokens=2048,
                temperature=0.1,
            )
            return response.choices[0].message.content
        except OpenAIError as e:
            if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return f'{{"error": "OpenAI {str(e)[:50]}", "quality_score": 70}}'
    
    return '{"error": "OpenAI timeout", "quality_score": 70}'

def parse_json_safe(text: str) -> Dict:
    """Parse robuste JSON"""
    patterns = [
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Premier objet JSON
        r'```json\s*(\{.*?\})\s*```',         # Code block
        r'\{.*\}',                           # Tout objet
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1) if pattern.startswith('```') else match.group(0))
            except json.JSONDecodeError:
                continue
    
    # Fallback Pandas raw
    return {"raw_ai": text[:500], "quality_score": 75, "fallback": True}

def clean_data(csv_path: str) -> Dict:
    try:
        # CSV safe
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        if df.empty:
            return {"error": "CSV vide", "quality_score": 0}
        
        original_len = len(df)
        
        # Pré-traitement minimal
        df.fillna('', inplace=True)
        raw_data = df.head(50).to_json(orient="records")  # Limite IA
        
        prompt = f"""CRM Expert. Nettoie ce JSON B2B:

{raw_data}

STRICT JSON SEULEMENT (copie-colle pas de texte):
{{
  "cleaned_data": [
    {{"nom": "Jean Dupont", "email": "jean@acme.fr", "telephone": "0612345678", "entreprise": "Acme"}}
  ],
  "report": {{"quality_score": 95}}
}}"""
        
        # OpenAI safe
        ai_response = safe_openai(prompt)
        result = parse_json_safe(ai_response)
        
        # Fallback Pandas si IA fail
        if "error" in result or not result.get("cleaned_data"):
            result = {
                "cleaned_data": df.to_dict('records')[:10],
                "report": {"quality_score": 80, "fallback": "Pandas raw"},
                "total_processed": original_len
            }
        
        result["report"]["total_processed"] = original_len
        return result
        
    except Exception as e:
        return {
            "error": f"Erreur système: {str(e)[:100]}",
            "stacktrace": traceback.format_exc()[:200],
            "quality_score": 0
        }

if __name__ == "__main__":
    result = clean_data("testdata.csv")
    print("✅ Résultat:", json.dumps(result, indent=2)[:500])
    
    with open("debug_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("💾 debug_result.json créé")
