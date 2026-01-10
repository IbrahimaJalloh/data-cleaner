"""Nettoyage données CRM avec OpenAI GPT-4o-mini."""

import json
import os
from typing import Dict

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI  # Remplace anthropic

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Nouvelle var env


def clean_data(csv_path: str) -> Dict:
    """Nettoie CSV: déduplique, normalise, rapport qualité."""
    df = pd.read_csv(csv_path)
    raw_data = df.to_json(orient="records")
    # Exemple JSON échappé
    example = '{"cleaned_data": [...], "report": {"duplicates_found": 2, "filled_fields": 3}, "quality_score": 95}'
    prompt = f"""Tu es expert nettoyage B2B. JSON brut: {raw_data[:4000]}
Tâches: 1. Doublons (email/tel/nom). 2. Normalise formats. 3. Remplis vides.
Retour STRICT JSON comme: {example}"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Moins cher, rapide
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
        temperature=0.1,  # Déterministe pour JSON
    )
    response_text = response.choices[0].message.content
    try:
        # OpenAI retourne souvent JSON direct, cherche premier {}
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start != -1:
            result = json.loads(response_text[start:end])
        else:
            result = {"raw": response_text}
    except (json.JSONDecodeError, ValueError) as e:
        result = {"error": str(e), "raw": response_text}
    return result


if __name__ == "__main__":
    if not os.path.exists("testdata.csv"):
        print("Crée testdata.csv d'abord !")
        exit(1)
    result = clean_data("testdata.csv")
    with open("cleaned_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("Nettoyage OK ! Rapport:", result.get("report", "N/A"))
    print("Score qualité:", result.get("quality_score", "N/A"))
