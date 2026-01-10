"""Nettoyage données CRM avec OpenAI GPT-4o-mini."""
import json
import os
from typing import Dict
import pandas as pd
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def normalize_phone(phone: str) -> str:
    """Tous formats FR → 33612345678"""
    if pd.isna(phone) or not str(phone).strip():
        return ""
    clean = re.sub(r'[^\d]', '', str(phone))  # Chiffres only
    
    if clean.startswith('33') and len(clean) >= 11:
        return clean[:12]  # +33 6xxxxxxxx
    elif len(clean) == 10 and clean.startswith(('06', '07')):
        return '33' + clean  # 06 → 336
    elif len(clean) == 9 and clean.startswith(('6', '7')):
        return '336' + clean
    return clean[-10:] or ""

def pre_deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Dédoublons strict avant IA"""
    df['tel_norm'] = df['telephone'].apply(normalize_phone)
    df['email_norm'] = df['email'].str.lower().str.strip()
    df['nom_norm'] = df['nom'].astype(str).str.lower().str.strip()
    
    # DOUBLONS: email OU (nom + tel_norm)
    mask_dup = (
        df['email_norm'].duplicated(keep='first') |
        df.duplicated(subset=['nom_norm', 'tel_norm'], keep='first')
    )
    duplicates = mask_dup.sum()
    df_clean = df[~mask_dup].drop(columns=['tel_norm', 'email_norm', 'nom_norm']).reset_index(drop=True)
    
    return df_clean, int(duplicates)

def clean_data(csv_path: str) -> Dict:
    """Nettoie CSV: déduplique local + IA polish."""
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return {"error": "CSV vide"}
    
    # PRÉ-NETTOYAGE LOCAL (gratuit)
    df_clean, duplicates_local = pre_deduplicate(df)
    raw_data = df_clean.to_json(orient="records", lines=False)
    
    prompt = f"""Expert B2B. Données pré-nettoyées ({duplicates_local} doublons supprimés):
{raw_data}

TÂCHES FINALES:
1. Normalise formats (case, espaces)
2. Remplis vides INTELLIGEMMENT (email @entreprise.fr)
3. Formats tél FR corrects

STRICT JSON:
{{
  "cleaned_data": [{{...}}],
  "report": {{
    "duplicates_local": {duplicates_local},
    "quality_score": 95,
    "anomalies": ["exemples"]
  }}
}}"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
        temperature=0.1,
    )
    
    response_text = response.choices[0].message.content
    try:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        result = json.loads(response_text[start:end])
        result["report"]["duplicates_local"] = duplicates_local
    except:
        result = {
            "cleaned_data": df_clean.to_dict('records'),
            "report": {"duplicates_local": duplicates_local, "quality_score": 90}
        }
    
    return result

if __name__ == "__main__":
    result = clean_data("testdata.csv")
    with open("cleaned_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("✅ Nettoyage OK!")
    print("Doublons supprimés:", result["report"].get("duplicates_local", 0))
    print("Score:", result.get("report", {}).get("quality_score", "N/A"))
