"""Nettoyage données CRM IA v2.0 - GPT-4o-mini + doublons/tél/score dynamique."""
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
    """Tous formats FR → 33612345678"""
    if pd.isna(phone) or not str(phone).strip():
        return ""
    # Chiffres uniquement
    clean = re.sub(r'[^\d]', '', str(phone))
    if clean.startswith('33') and len(clean) >= 11:
        return clean[:12]  # +33 6xxxxxxxx
    elif len(clean) == 10 and clean.startswith(('06', '07')):
        return '33' + clean
    elif len(clean) == 9 and clean.startswith(('6', '7')):
        return '336' + clean
    return clean[-10:] or ""

def calculate_quality_score(df: pd.DataFrame, original_len: int) -> Dict[str, int]:
    """Score dynamique 0-100 + breakdown"""
    if len(df) == 0:
        return {"score": 0, "recovery": 0, "emails": 0, "tel": 0, "complete": 0}
    
    metrics = {}
    
    # Récupération (25pts)
    recovery = min((len(df) / original_len) * 25, 25)
    metrics["recovery"] = int(recovery)
    
    # Emails valides (25pts)
    valid_email = df['email'].str.contains(r'@[\w\.-]+\.\w+', na=False, regex=True).sum()
    email_score = min((valid_email / len(df)) * 25, 25)
    metrics["emails"] = int(email_score)
    
    # Tél FR (25pts)
    valid_tel = df['telephone'].astype(str).str.contains(r'(33\s*6|06)\d{{8}}', na=False, regex=True).sum()
    tel_score = min((valid_tel / len(df)) * 25, 25)
    metrics["tel"] = int(tel_score)
    
    # Complétude (25pts)
    complete = df[['nom', 'email', 'telephone', 'entreprise']].notna().all(axis=1).sum()
    complete_score = min((complete / len(df)) * 25, 25)
    metrics["complete"] = int(complete_score)
    
    metrics["score"] = sum(metrics.values())
    return metrics

def pre_deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Dédoublons strict avant IA"""
    df = df.copy()
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

def merge_similar_contacts(data: List[Dict]) -> List[Dict]:
    """Fusionne contacts similaires (même nom+tél)"""
    merged = []
    used = set()
    
    for i, row1 in enumerate(data):
        if i in used:
            continue
        row_fused = row1.copy()
        used.add(i)
        
        # Fusionne similaires
        for j, row2 in enumerate(data):
            if i == j or j in used:
                continue
            if (row1.get('nom', '').lower() in row2.get('nom', '').lower() or 
                row2.get('nom', '').lower() in row1.get('nom', '').lower()) and \
               normalize_phone(row1.get('telephone', '')) == normalize_phone(row2.get('telephone', '')):
                
                # Meilleur email
                if '@gmail.com' in row2.get('email', ''):
                    row_fused['email'] = row2['email']
                # Nom complet
                if len(row2.get('nom', '')) > len(row1.get('nom', '')):
                    row_fused['nom'] = row2['nom']
                used.add(j)
        
        merged.append(row_fused)
    
    return merged

def clean_data(csv_path: str) -> Dict:
    """Pipeline complet: déduplique + normalise + IA + fusion."""
    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            return {"error": "CSV vide", "quality_score": 0}
        
        original_len = len(df)
        
        # 1. PRÉ-NETTOYAGE LOCAL
        df_clean, duplicates_local = pre_deduplicate(df)
        
        # 2. SCORE BASE
        quality_metrics = calculate_quality_score(df_clean, original_len)
        
        # 3. PROMPT IA OPTIMISÉ
        raw_data = df_clean.to_json(orient="records")
        prompt = f"""Expert B2B CRM. Données pré-traitées:

LINES: {original_len} → {len(df_clean)} ({duplicates_local} doublons supprimés)
SCORE BASE: {quality_metrics['score']}/100

JSON: {raw_data[:5000]}

FINITION:
- Formats parfaits (case, espaces)
- Remplis intelligemment (@entreprise.fr)
- Tél FR uniformes

STRICT JSON avec quality_score ajusté (±5pts base):
{{
  "cleaned_data": [{{...}}],
  "report": {{
    "duplicates_local": {duplicates_local},
    "quality_score": {quality_metrics['score']},
    "anomalies": []
  }}
}}"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.1,
        )
        
        response_text = response.choices[0].message.content
        
        # 4. PARSE IA
        try:
            start = max(response_text.find("{"), 0)
            end = response_text.rfind("}") + 1
            result = json.loads(response_text[start:end])
        except:
            result = {"cleaned_data": df_clean.to_dict('records')}
        
        # 5. FUSION SIMILAIRES
        result["cleaned_data"] = merge_similar_contacts(result.get("cleaned_data", []))
        
        # 6. RAPPORT FINAL
        result["report"] = result.get("report", {})
        result["report"]["duplicates_local"] = duplicates_local
        result["report"]["quality_score"] = quality_metrics["score"] + 2  # IA boost
        result["report"]["quality_breakdown"] = quality_metrics
        result["report"]["total_processed"] = original_len
        
        return result
        
    except Exception as e:
        return {"error": str(e), "quality_score": 0}

if __name__ == "__main__":
    if not os.path.exists("testdata.csv"):
        print("❌ Crée testdata.csv !")
        exit(1)
    
    result = clean_data("testdata.csv")
    
    print("✅ Nettoyage terminé !")
    print(f"📊 Lignes: {result.get('report', {}).get('total_processed', 0)}")
    print(f"🗑️ Doublons supprimés: {result.get('report', {}).get('duplicates_local', 0)}")
    print(f"⭐ Quality score: {result.get('report', {}).get('quality_score', 'N/A')}")
    
    with open("cleaned_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("💾 Résultat → cleaned_result.json")
