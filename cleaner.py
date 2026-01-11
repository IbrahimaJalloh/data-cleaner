"""
CRM Clean IA v3.0 - Production Ready
Améliorations majeures :
- Validation téléphone stricte (10 chiffres exacts)
- Dédoublonnage composite (email + nom+tel + tel seul)
- Parsing JSON multi-stratégies avec fallback intelligent
- Logs progressifs pour gros volumes
- Gestion erreurs granulaire
- Tests unitaires intégrés
- Export CSV + JSON
- Métriques détaillées
"""

import json
import os
import re
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# ========== CONFIGURATION ==========
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crm_clean.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Validation API OpenAI
API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not API_KEY or API_KEY == "your-api-key-here":
    logger.warning("⚠️ OPENAI_API_KEY manquante ou invalide - Mode fallback Pandas activé")
    client = None
else:
    try:
        client = OpenAI(api_key=API_KEY)
        # Test connexion
        client.models.list()
        logger.info("✅ OpenAI connecté")
    except Exception as e:
        logger.error(f"❌ Erreur OpenAI : {e}")
        client = None

# ========== NORMALISATION TÉLÉPHONE ==========
def normalize_phone(phone: str) -> str:
    """
    Normalisation stricte téléphone français.
    Formats acceptés :
    - 06 12 34 56 78 → 0612345678
    - +33 6 12 34 56 78 → 0612345678
    - 33612345678 → 0612345678
    
    Returns:
        str: Numéro normalisé 10 chiffres (0612345678) ou chaîne vide
    """
    if pd.isna(phone) or not str(phone).strip():
        return ""
    
    # Supprime tout sauf chiffres
    clean = re.sub(r'[^\d]', '', str(phone))
    
    # Cas +33 ou 33 en préfixe
    if clean.startswith('33') and len(clean) >= 11:
        clean = '0' + clean[2:11]  # Garde 10 chiffres après 33
    
    # Validation stricte : 10 chiffres commençant par 06 ou 07
    if len(clean) == 10 and clean[0] == '0' and clean[1] in ['6', '7']:
        return clean
    
    return ""

def is_valid_fr_phone(phone: str) -> bool:
    """
    Validation stricte mobile français.
    
    Returns:
        bool: True si format exact 06XXXXXXXX ou 07XXXXXXXX
    """
    norm = normalize_phone(phone)
    # EXACTEMENT 10 chiffres commençant par 06 ou 07
    return bool(re.fullmatch(r'0[67]\d{8}', norm))

# ========== VALIDATION EMAIL ==========
def is_valid_email(email: str) -> bool:
    """Validation email stricte RFC 5322 simplifié"""
    if pd.isna(email) or not str(email).strip():
        return False
    pattern = r'^[a-zA-Z0-9][a-zA-Z0-9._%+-]*@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email).strip().lower()))

# ========== SCORE DE QUALITÉ ==========
def calculate_quality_score(df: pd.DataFrame, original_len: int) -> Dict[str, int]:
    """
    Score qualité sur 100 points :
    - 25pts : Taux de récupération (lignes valides / total)
    - 25pts : Emails valides
    - 25pts : Téléphones FR valides
    - 25pts : Complétude (4 champs remplis)
    
    Returns:
        Dict avec score total et breakdown détaillé
    """
    if len(df) == 0:
        return {
            "score": 0,
            "recovery": 0,
            "emails": 0,
            "phones": 0,
            "complete": 0
        }
    
    breakdown = {}
    
    # 1. Récupération (25pts)
    recovery_rate = len(df) / max(original_len, 1)
    breakdown["recovery"] = min(int(recovery_rate * 25), 25)
    
    # 2. Emails valides (25pts)
    valid_emails = df['email'].apply(is_valid_email).sum()
    email_rate = valid_emails / len(df)
    breakdown["emails"] = min(int(email_rate * 25), 25)
    
    # 3. Téléphones FR valides (25pts)
    valid_phones = df['telephone'].apply(is_valid_fr_phone).sum()
    phone_rate = valid_phones / len(df)
    breakdown["phones"] = min(int(phone_rate * 25), 25)
    
    # 4. Complétude (25pts)
    required_cols = ['nom', 'email', 'telephone', 'entreprise']
    complete_rows = df[required_cols].notna().all(axis=1).sum()
    complete_rate = complete_rows / len(df)
    breakdown["complete"] = min(int(complete_rate * 25), 25)
    
    breakdown["score"] = sum([
        breakdown["recovery"],
        breakdown["emails"],
        breakdown["phones"],
        breakdown["complete"]
    ])
    
    return breakdown

# ========== DÉDOUBLONNAGE INTELLIGENT ==========
def advanced_deduplicate(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Dédoublonnage en 3 passes :
    1. Email exact (priorité haute)
    2. Nom + Téléphone normalisé
    3. Téléphone seul (si valide)
    
    Returns:
        Tuple[DataFrame nettoyé, Dict statistiques]
    """
    df = df.copy()
    initial_count = len(df)
    
    # Normalisation
    df['email_norm'] = df['email'].apply(lambda x: str(x).lower().strip() if pd.notna(x) else '')
    df['tel_norm'] = df['telephone'].apply(normalize_phone)
    df['nom_norm'] = df['nom'].apply(lambda x: str(x).lower().strip() if pd.notna(x) else '')
    
    stats = {
        "duplicates_email": 0,
        "duplicates_name_phone": 0,
        "duplicates_phone": 0
    }
    
    # Passe 1 : Email exact
    mask_email = (df['email_norm'] != '') & df['email_norm'].duplicated(keep='first')
    stats["duplicates_email"] = mask_email.sum()
    df = df[~mask_email]
    
    # Passe 2 : Nom + Téléphone
    df['composite_key'] = df['nom_norm'] + '|' + df['tel_norm']
    mask_composite = (
        (df['nom_norm'] != '') & 
        (df['tel_norm'] != '') & 
        df['composite_key'].duplicated(keep='first')
    )
    stats["duplicates_name_phone"] = mask_composite.sum()
    df = df[~mask_composite]
    
    # Passe 3 : Téléphone seul (si valide FR)
    mask_phone = (
        df['tel_norm'].apply(lambda x: is_valid_fr_phone(x)) &
        df['tel_norm'].duplicated(keep='first')
    )
    stats["duplicates_phone"] = mask_phone.sum()
    df = df[~mask_phone]
    
    # Nettoyage colonnes temporaires
    df_clean = df.drop(columns=['email_norm', 'tel_norm', 'nom_norm', 'composite_key'])
    df_clean = df_clean.reset_index(drop=True)
    
    stats["total_duplicates"] = initial_count - len(df_clean)
    
    logger.info(f"📊 Dédoublonnage : {initial_count} → {len(df_clean)} ({stats['total_duplicates']} doublons)")
    
    return df_clean, stats

# ========== APPEL OPENAI ROBUSTE ==========
def safe_openai_call(prompt: str, max_retries: int = 3) -> str:
    """
    Appel OpenAI avec retry exponentiel.
    
    Args:
        prompt: Texte à envoyer (auto-tronqué à 12000 chars)
        max_retries: Nombre de tentatives
    
    Returns:
        str: Réponse JSON ou message d'erreur
    """
    if not client:
        logger.warning("⚠️ Client OpenAI non initialisé - Fallback Pandas")
        return '{"cleaned_data": [], "report": {"error": "API key manquante"}}'
    
    # Troncature sécurisée
    safe_prompt = prompt[:12000]
    
    for attempt in range(max_retries):
        try:
            logger.info(f"🤖 Appel OpenAI (tentative {attempt + 1}/{max_retries})")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es un expert CRM B2B. Réponds UNIQUEMENT en JSON valide, sans markdown."
                    },
                    {
                        "role": "user",
                        "content": safe_prompt
                    }
                ],
                max_tokens=3000,
                temperature=0.1,
            )
            return response.choices[0].message.content
        
        except OpenAIError as e:
            wait_time = 2 ** attempt
            logger.warning(f"⚠️ Erreur OpenAI (tentative {attempt + 1}) : {e}")
            if attempt < max_retries - 1:
                logger.info(f"⏳ Retry dans {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error("❌ Échec OpenAI après 3 tentatives")
                return '{"cleaned_data": [], "report": {"error": "OpenAI timeout"}}'
        
        except Exception as e:
            logger.error(f"❌ Erreur inattendue : {e}")
            return '{"cleaned_data": [], "report": {"error": "Exception interne"}}'

# ========== PARSING JSON MULTI-STRATÉGIES ==========
def parse_json_safe(text: str) -> Dict:
    """
    Parse JSON avec 5 stratégies de fallback :
    1. JSON direct
    2. Extraction entre ```json...```
    3. Extraction premier objet {...}
    4. Extraction array [...]
    5. Regex personnalisée
    
    Returns:
        Dict parsé ou {"error": "parse failed", "raw": extrait}
    """
    if not text or not text.strip():
        return {"error": "Réponse vide", "raw": ""}
    
    strategies = [
        # Stratégie 1 : JSON direct
        lambda t: json.loads(t),
        
        # Stratégie 2 : Markdown code block
        lambda t: json.loads(re.search(r'```json\s*(\{.*?\})\s*```', t, re.DOTALL).group(1)),
        
        # Stratégie 3 : Premier objet {}
        lambda t: json.loads(re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', t, re.DOTALL).group(0)),
        
        # Stratégie 4 : Array [...]
        lambda t: {"cleaned_data": json.loads(re.search(r'\[.*?\]', t, re.DOTALL).group(0)), "report": {}},
        
        # Stratégie 5 : Extraction manuelle clés
        lambda t: {
            "cleaned_data": json.loads(re.search(r'"cleaned_data"\s*:\s*(\[.*?\])', t, re.DOTALL).group(1)),
            "report": {}
        }
    ]
    
    for i, strategy in enumerate(strategies, 1):
        try:
            result = strategy(text)
            logger.info(f"✅ JSON parsé (stratégie {i})")
            return result
        except (json.JSONDecodeError, AttributeError, TypeError):
            continue
    
    # Échec total
    logger.error("❌ Échec parsing JSON - Toutes stratégies échouées")
    return {
        "error": "Parse failed",
        "raw": text[:500],
        "cleaned_data": [],
        "report": {}
    }

# ========== PIPELINE PRINCIPAL ==========
def clean_data(csv_path: str, output_dir: str = "output") -> Dict:
    """
    Pipeline complet de nettoyage CRM.
    
    Args:
        csv_path: Chemin CSV source
        output_dir: Dossier de sortie (créé si inexistant)
    
    Returns:
        Dict avec cleaned_data, report et métriques
    """
    start_time = time.time()
    logger.info(f"🚀 Début nettoyage : {csv_path}")
    
    # Création dossier output
    Path(output_dir).mkdir(exist_ok=True)
    
    try:
        # ========== 1. CHARGEMENT ==========
        logger.info("📂 Chargement CSV...")
        df = pd.read_csv(csv_path, on_bad_lines='skip', encoding='utf-8')
        
        if df.empty:
            return {
                "error": "CSV vide ou illisible",
                "quality_score": 0,
                "report": {}
            }
        
        original_len = len(df)
        logger.info(f"📊 {original_len} lignes chargées")
        
        # Colonnes standardisées (tolère variations casse)
        column_mapping = {
            col.lower().strip(): col for col in df.columns
        }
        required = ['nom', 'email', 'telephone', 'entreprise']
        for req in required:
            if req not in column_mapping:
                # Crée colonne vide si manquante
                df[req] = ""
                logger.warning(f"⚠️ Colonne '{req}' manquante - Créée vide")
        
        # ========== 2. DÉDOUBLONNAGE ==========
        logger.info("🔍 Dédoublonnage...")
        df_clean, dedup_stats = advanced_deduplicate(df)
        
        # ========== 3. NORMALISATION ==========
        logger.info("🧹 Normalisation téléphones...")
        df_clean['telephone'] = df_clean['telephone'].apply(normalize_phone)
        
        # ========== 4. SCORE DE QUALITÉ ==========
        logger.info("📈 Calcul score qualité...")
        quality = calculate_quality_score(df_clean, original_len)
        
        # ========== 5. PRÉPARATION PROMPT IA ==========
        # Limite à 200 lignes max pour éviter troncature
        sample_size = min(len(df_clean), 200)
        df_sample = df_clean.head(sample_size)
        
        raw_data = df_sample.to_json(orient="records", force_ascii=False)
        
        prompt = f"""Tu es un expert CRM B2B. Nettoie ces données selon ces règles STRICTES :

STATISTIQUES :
- Total initial : {original_len} lignes
- Après dédoublonnage : {len(df_clean)} lignes
- Échantillon traité : {sample_size} lignes
- Doublons retirés : {dedup_stats['total_duplicates']}

DONNÉES JSON :
{raw_data}

RÈGLES DE NETTOYAGE :
1. Email : Valider format RFC 5322 (ex: nom@domaine.fr)
2. Téléphone : Format 06XXXXXXXX ou 07XXXXXXXX (10 chiffres)
3. Nom : Capitaliser (ex: "jean dupont" → "Jean Dupont")
4. Entreprise : Retirer "SAS", "SARL" sauf si partie intégrante du nom
5. Retirer lignes avec email ET téléphone invalides

RÉPONSE ATTENDUE (JSON STRICT, SANS MARKDOWN) :
{{
  "cleaned_data": [
    {{
      "nom": "Jean Dupont",
      "email": "jean.dupont@acme.fr",
      "telephone": "0612345678",
      "entreprise": "Acme Corp"
    }}
  ],
  "report": {{
    "corrections_applied": 42,
    "invalid_removed": 5
  }}
}}

RÉPONDS UNIQUEMENT EN JSON, SANS TEXTE AVANT/APRÈS."""
        
        # ========== 6. APPEL OPENAI ==========
        ai_response = safe_openai_call(prompt)
        result = parse_json_safe(ai_response)
        
        # ========== 7. FUSION RÉSULTATS ==========
        # Force le score calculé (cohérent)
        result["report"] = result.get("report", {})
        result["report"]["quality_score"] = quality["score"]
        result["report"]["quality_breakdown"] = quality
        result["report"]["deduplication"] = dedup_stats
        result["report"]["total_processed"] = original_len
        result["report"]["final_count"] = len(df_clean)
        result["report"]["processing_time"] = round(time.time() - start_time, 2)
        
        # Fallback si IA échoue
        if not result.get("cleaned_data"):
            logger.warning("⚠️ IA n'a pas retourné de données - Fallback Pandas")
            result["cleaned_data"] = df_clean.head(100).to_dict('records')
            result["report"]["fallback"] = "Pandas (IA échec)"
        
        # ========== 8. EXPORT ==========
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON
        json_path = f"{output_dir}/crm_clean_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ JSON exporté : {json_path}")
        
        # CSV
        if result["cleaned_data"]:
            df_export = pd.DataFrame(result["cleaned_data"])
            csv_path_out = f"{output_dir}/crm_clean_{timestamp}.csv"
            df_export.to_csv(csv_path_out, index=False, encoding='utf-8')
            logger.info(f"✅ CSV exporté : {csv_path_out}")
        
        logger.info(f"✅ Nettoyage terminé en {result['report']['processing_time']}s")
        return result
        
    except FileNotFoundError:
        logger.error(f"❌ Fichier introuvable : {csv_path}")
        return {"error": "Fichier CSV introuvable", "quality_score": 0}
    
    except pd.errors.EmptyDataError:
        logger.error("❌ CSV vide ou corrompu")
        return {"error": "CSV vide", "quality_score": 0}
    
    except Exception as e:
        logger.error(f"❌ Erreur critique : {e}", exc_info=True)
        return {
            "error": f"Exception : {str(e)[:200]}",
            "quality_score": 0,
            "traceback": str(e)
        }

# ========== TESTS UNITAIRES ==========
def run_tests():
    """Tests de validation des fonctions critiques"""
    logger.info("🧪 Lancement tests unitaires...")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1 : Normalisation téléphone
    tests_total += 1
    test_cases = [
        ("06 12 34 56 78", "0612345678"),
        ("+33 6 12 34 56 78", "0612345678"),
        ("33612345678", "0612345678"),
        ("0712345678", "0712345678"),
        ("123", ""),  # Invalide
        ("", "")
    ]
    for input_val, expected in test_cases:
        if normalize_phone(input_val) == expected:
            tests_passed += 1
        else:
            logger.error(f"❌ Test normalize_phone échoué : {input_val} → attendu {expected}, obtenu {normalize_phone(input_val)}")
    
    # Test 2 : Validation téléphone
    tests_total += 1
    if is_valid_fr_phone("0612345678") and not is_valid_fr_phone("123"):
        tests_passed += 1
    else:
        logger.error("❌ Test is_valid_fr_phone échoué")
    
    # Test 3 : Validation email
    tests_total += 1
    if is_valid_email("test@example.com") and not is_valid_email("invalid"):
        tests_passed += 1
    else:
        logger.error("❌ Test is_valid_email échoué")
    
    # Test 4 : Score qualité
    tests_total += 1
    df_test = pd.DataFrame({
        'nom': ['Test'],
        'email': ['test@test.fr'],
        'telephone': ['0612345678'],
        'entreprise': ['TestCorp']
    })
    score = calculate_quality_score(df_test, 1)
    if score['score'] == 100:
        tests_passed += 1
    else:
        logger.error(f"❌ Test score échoué : attendu 100, obtenu {score['score']}")
    
    logger.info(f"✅ Tests : {tests_passed}/{tests_total} réussis")
    return tests_passed == tests_total

# ========== POINT D'ENTRÉE ==========
if __name__ == "__main__":
    # Tests
    if run_tests():
        logger.info("✅ Tous les tests passent - Démarrage traitement")
    else:
        logger.warning("⚠️ Certains tests échouent - Vérifiez le code")
    
    # Traitement
    result = clean_data("testdata.csv")
    
    # Affichage résultats
    print("\n" + "="*60)
    print("🎯 RÉSULTATS CRM CLEAN IA V3.0")
    print("="*60)
    print(f"📊 Total traité      : {result.get('report', {}).get('total_processed', 0)}")
    print(f"📈 Lignes finales    : {result.get('report', {}).get('final_count', 0)}")
    print(f"🗑️  Doublons retirés : {result.get('report', {}).get('deduplication', {}).get('total_duplicates', 0)}")
    print(f"⭐ Score qualité    : {result.get('report', {}).get('quality_score', 0)}/100")
    print(f"⏱️  Temps traitement : {result.get('report', {}).get('processing_time', 0)}s")
    print("\n📈 Détail score :")
    breakdown = result.get('report', {}).get('quality_breakdown', {})
    print(f"  - Récupération : {breakdown.get('recovery', 0)}/25")
    print(f"  - Emails       : {breakdown.get('emails', 0)}/25")
    print(f"  - Téléphones   : {breakdown.get('phones', 0)}/25")
    print(f"  - Complétude   : {breakdown.get('complete', 0)}/25")
    print("="*60 + "\n")