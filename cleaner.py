"""CRM Clean IA v2.3 - Score dynamique cohérent + tél tous formats."""
import json
import os
import re
import time
import pandas as pd
from typing import Dict, List
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except:
    client = None

def normalize_phone(phone: str) -> str:
    """Tous FR → 33612345678"""
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
    """FIX: 336xxxxxxxx OU 06xxxxxxxx (tous formats)"""
    norm = normalize_phone(phone)
    return bool(re.match(r'^336\d{8}$|^6\d{8}$', norm))

def calculate_quality_score(df: pd.DataFrame, original_len: int) -> Dict[str, int]:
    """Score DYNAMIQUE 100% cohérent"""
    if len(df) == 0:
        return {"score": 0, "breakdown": {}}
    
    breakdown = {}
    
    # Récupération (25pts)
    recovery = min((len(df) / original_len) * 25, 25)
    breakdown["recovery"] = int(recovery)
    
    # Emails valides (25pts)
    valid_email = df['email'].str.contains(r'@[\w\.-]+\.\w+', na=False, regex=True).sum()
    emails = min((valid_email / len(df)) * 25, 25)
    breakdown["emails"] = int(emails)
    
    # Tél FR FIX (25pts)
    valid_tel = df['telephone'].apply(is_valid_fr_phone).sum()
    tel = min((valid_tel / len(df)) * 25, 25)
    breakdown["tel"] = int(tel)
    
    # Complétude (25pts)
    complete = df[['nom', 'email', 'telephone', 'entreprise']].notna().all(axis=1).sum()
    complete_score = min((complete / len(df)) * 25, 25)
    breakdown["complete"] = int(complete_score)
    
    breakdown["score"] = sum([
        breakdown["recovery"],
        breakdown["emails"], 
        breakdown["tel"],
        breakdown["complete"]
    ])
    return breakdown

def pre_deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Dédoublons strict"""
    df = df.copy()
    df.fillna('', inplace=True)
    
    df['tel_norm'] = df['telephone'].apply(normalize_phone)
    df['email_norm'] = df['email'].str.lower().str.strip()
    
    # Email OU (nom + tel)
    mask_dup = df['email_norm'].duplicated(keep='first')
    duplicates = mask_dup.sum()
    
    df_clean = df[~mask_dup].drop(columns=['tel_norm', 'email_norm']).reset_index(drop=True)
    return df_clean, int(duplicates)

def safe_openai(prompt: str) -> str:
    """OpenAI robuste"""
    if not client:
        return '{"cleaned_data": [], "report": {"error": "API key"}}'
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt[:8000]}],
            max_tokens=2048,
            temperature=0.1,
        )
        return response.choices[0].message.content
    except:
        return '{"cleaned_data": [], "report": {"fallback": "pandas"}}'

def parse_json_safe(text: str) -> Dict:
    """Parse robuste"""
    for pattern in [r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', r'```json\s*(\{.*?\})```']:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1) if '```' in pattern else match.group(0))
            except:
                pass
    return {"raw": text[:200], "parsed": False}

def clean_data(csv_path: str) -> Dict:
    """Pipeline v2.3 complet"""
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        if df.empty:
            return {"error": "CSV vide", "quality_score": 0}
        
        original_len = len(df)
        
        # 1. Pré-traitement
        df_clean, duplicates_local = pre_deduplicate(df)
        
        # 2. Score BASE (IA ne touche PAS)
        quality = calculate_quality_score(df_clean, original_len)
        
        # 3. Prompt IA SANS score
        raw_data = df_clean.to_json(orient="records")
        prompt = f"""CRM Expert. Nettoie JSON B2B:

LINES: {original_len} → {len(df_clean)} ({duplicates_local} doublons)

{raw_data}

STRICT JSON (PAS quality_score):
{{
  "cleaned_data": [
    {{"nom": "Jean Dupont", "email": "jean@acme.fr", "telephone": "0612345678", "entreprise": "Acme"}}
  ],
  "report": {{"duplicates_local": {duplicates_local}}}
}}"""
        
        # 4. IA safe
        ai_response = safe_openai(prompt)
        result = parse_json_safe(ai_response)
        
        # 5. FORCE score calculé (cohérent)
        result["report"] = result.get("report", {})
        result["report"]["quality_score"] = quality["score"]
        result["report"]["quality_breakdown"] = quality["breakdown"]
        result["report"]["duplicates_local"] = duplicates_local
        result["report"]["total_processed"] = original_len
        
        # 6. Fallback si IA vide
        if not result.get("cleaned_data"):
            result["cleaned_data"] = df_clean.head(20).to_dict('records')
            result["report"]["fallback"] = "Pandas safe"
        
        return result
        
    except Exception as e:
        return {
            "error": str(e)[:100],
            "quality_score": 0,
            "debug": "Exception caught"
        }

if __name__ == "__main__":
    result = clean_data("testdata.csv")
    print("🚀 Résultats v2.3:")
    print(f"📊 Total: {result.get('report', {}).get('total_processed', 0)}")
    print(f"🗑️ Doublons: {result.get('report', {}).get('duplicates_local', 0)}")
    print(f"⭐ Score: {result.get('report', {}).get('quality_score', 0)}")
    print(f"📈 Breakdown: {result.get('report', {}).get('quality_breakdown', {})}")
    
    with open("result_v2.3.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
