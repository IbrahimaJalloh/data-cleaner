"""Nettoyage CRM IA v2.1 - FIX regex + IA agressive."""
import json
import os
import re
import pandas as pd
from typing import Dict, List
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def normalize_phone(phone: str) -> str:
    """FIX: Tous formats → 33612345678"""
    if pd.isna(phone) or not str(phone).strip():
        return ""
    clean = re.sub(r'[^\d]', '', str(phone))
    if clean.startswith('33') and len(clean) >= 11:
        return clean[:12]
    elif len(clean) == 10 and clean.startswith(('06', '07')):
        return '33' + clean
    elif len(clean) == 9 and clean.startswith(('6', '7')):
        return '336' + clean
    return clean[-10:] or ""

def is_valid_fr_phone(phone: str) -> bool:
    """FIX regex: 06xxxxxxxx OU 336xxxxxxxx"""
    if not phone:
        return False
    norm = normalize_phone(phone)
    return bool(re.match(r'^336\d{8}$', norm))

def calculate_quality_score(df: pd.DataFrame, original_len: int) -> Dict[str, int]:
    """Score DYNAMIQUE fixe regex"""
    if len(df) == 0:
        return {"score": 0, "breakdown": {}}
    
    breakdown = {}
    
    # Récupération
    recovery = min((len(df) / original_len) * 25, 25)
    breakdown["recovery"] = int(recovery)
    
    # Emails
    valid_email = df['email'].str.contains(r'@[\w\.-]+\.\w+', na=False, regex=True).sum()
    breakdown["emails"] = int(min((valid_email / len(df)) * 25, 25))
    
    # Tél FIX regex
    valid_tel = df['telephone'].apply(is_valid_fr_phone).sum()
    breakdown["tel"] = int(min((valid_tel / len(df)) * 25, 25))
    
    # Complétude
    complete = df[['nom', 'email', 'telephone', 'entreprise']].notna().all(axis=1).sum()
    breakdown["complete"] = int(min((complete / len(df)) * 25, 25))
    
    breakdown["score"] = sum(breakdown.values())
    return breakdown

def pre_deduplicate(df: pd.DataFrame) -> tuple:
    df = df.copy()
    df['tel_norm'] = df['telephone'].apply(normalize_phone)
    df['email_norm'] = df['email'].str.lower().str.strip()
    
    mask_dup = df['email_norm'].duplicated(keep='first')
    duplicates = mask_dup.sum()
    df_clean = df[~mask_dup].drop(columns=['tel_norm', 'email_norm'])
    
    return df_clean.reset_index(drop=True), int(duplicates)

def clean_data(csv_path: str) -> Dict:
    try:
        df = pd.read_csv(csv_path)
        original_len = len(df)
        
        # PRÉ-NETTOYAGE
        df_clean, duplicates_local = pre_deduplicate(df)
        quality = calculate_quality_score(df_clean, original_len)
        
        raw_data = df_clean.to_json(orient="records")
        
        prompt = f"""CRM Expert. Données pré-traitées:

{original_len} → {len(df_clean)} lignes ({duplicates_local} doublons)
Score base: {quality['score']}/100

JSON:
{raw_data}

FINITION AGRESSIVE:
- REMPLIS TOUS champs vides (@entreprise.fr, 06xxxxxxxx)
- Formats TÉL parfaits
- Normalise case/espaces

STRICT JSON (quality_score = base ±5):
{{
  "cleaned_data": [{{...}}],
  "report": {{
    "duplicates_local": {duplicates_local},
    "quality_score": {quality['score']},
    "quality_breakdown": {json.dumps(quality['breakdown'])}
  }}
}}"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.0,  # Déterministe
        )
        
        response_text = response.choices[0].message.content
        start = max(response_text.find("{"), 0)
        end = response_text.rfind("}") + 1
        result = json.loads(response_text[start:end])
        
        result["report"]["duplicates_local"] = duplicates_local
        result["report"]["quality_score"] = quality["score"]
        result["report"]["quality_breakdown"] = quality["breakdown"]
        result["report"]["total_processed"] = original_len
        
        return result
        
    except Exception as e:
        return {"error": str(e), "quality_score": 0}

if __name__ == "__main__":
    result = clean_data("testdata.csv")
    print("✅ Résultats:")
    print(f"Input: {result.get('report', {}).get('total_processed', 0)} lignes")
    print(f"Doublons: {result.get('report', {}).get('duplicates_local', 0)}")
    print(f"Score: {result.get('report', {}).get('quality_score', 0)}/100")
    print("\n📊 Breakdown:", result.get('report', {}).get('quality_breakdown', {}))
    
    with open("cleaned_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
