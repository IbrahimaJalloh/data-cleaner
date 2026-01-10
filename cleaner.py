"""CRM Clean IA v2.4 - Score dynamique cohérent + tél tous formats."""

import os
import json
import re
import logging
import traceback
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Gestion OpenAI avec fallback
try:
    from openai import OpenAI, APIError, APIConnectionError
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        client = OpenAI(api_key=api_key)
    else:
        client = None
except:
    client = None

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════
# LOGGING WINDOWS COMPATIBLE (UTF-8 force)
# ═══════════════════════════════════════════════════════════════════════

class UTF8StreamHandler(logging.StreamHandler):
    """Handler qui force UTF-8 sur Windows"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self.stream, 'reconfigure'):
            try:
                self.stream.reconfigure(encoding='utf-8')
            except:
                pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler('cleaner.log', encoding='utf-8'),
        UTF8StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("[INIT] CRM Cleaner v2.5 - Windows Compatible")


# ═══════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CleaningMetrics:
    input_rows: int
    output_rows: int
    duplicates_removed: int
    empty_rows_removed: int
    invalid_emails: int
    invalid_phones: int
    accents_normalized: int
    whitespace_trimmed: int
    execution_time: float
    quality_score: float
    error_message: Optional[str] = None


@dataclass
class Record:
    nom: str
    email: str
    telephone: str
    entreprise: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════
# VALIDATEURS
# ═══════════════════════════════════════════════════════════════════════

class Validator:
    """Validation emails & telephones"""
    
    EMAIL_PATTERN = re.compile(
        r'^[a-zA-Z0-9.!#$%&\'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$'
    )
    
    PHONE_PATTERN = re.compile(
        r'^(\+?33|0|(\+[0-9]{1,3}))?[\s\-]?[0-9]{1,4}[\s\-]?[0-9]{1,4}[\s\-]?[0-9]{1,4}$'
    )
    
    @staticmethod
    def is_valid_email(email: str) -> bool:
        if not email or not isinstance(email, str):
            return False
        email = email.strip().lower()
        return bool(Validator.EMAIL_PATTERN.match(email))
    
    @staticmethod
    def is_valid_phone(phone: str) -> bool:
        if not phone or not isinstance(phone, str):
            return False
        phone_clean = re.sub(r'[\s\-\(\)]', '', phone)
        return bool(Validator.PHONE_PATTERN.match(phone_clean))
    
    @staticmethod
    def is_empty_value(value: str) -> bool:
        if pd.isna(value):
            return True
        if isinstance(value, str) and value.strip() == '':
            return True
        return False


# ═══════════════════════════════════════════════════════════════════════
# NORMALISATEUR TEXTE
# ═══════════════════════════════════════════════════════════════════════

class TextNormalizer:
    """Normalisation texte + accents"""
    
    ACCENT_MAP = {
        'à': 'a', 'â': 'a', 'ä': 'a', 'á': 'a',
        'è': 'e', 'ê': 'e', 'ë': 'e', 'é': 'e',
        'ì': 'i', 'î': 'i', 'ï': 'i', 'í': 'i',
        'ò': 'o', 'ô': 'o', 'ö': 'o', 'ó': 'o',
        'ù': 'u', 'û': 'u', 'ü': 'u', 'ú': 'u',
        'ç': 'c', 'ñ': 'n',
        'À': 'A', 'Â': 'A', 'Ä': 'A', 'Á': 'A',
        'È': 'E', 'Ê': 'E', 'Ë': 'E', 'É': 'E',
        'Ì': 'I', 'Î': 'I', 'Ï': 'I', 'Í': 'I',
        'Ò': 'O', 'Ô': 'O', 'Ö': 'O', 'Ó': 'O',
        'Ù': 'U', 'Û': 'U', 'Ü': 'U', 'Ú': 'U',
        'Ç': 'C', 'Ñ': 'N',
    }
    
    @staticmethod
    def remove_accents(text: str) -> Tuple[str, bool]:
        if not isinstance(text, str):
            return str(text), False
        
        has_accent = False
        result = []
        for char in text:
            if char in TextNormalizer.ACCENT_MAP:
                result.append(TextNormalizer.ACCENT_MAP[char])
                has_accent = True
            else:
                result.append(char)
        
        return ''.join(result), has_accent
    
    @staticmethod
    def normalize_text(text: str) -> Tuple[str, int]:
        if not isinstance(text, str):
            return str(text), 0
        
        changes = 0
        
        # Trim
        text_before = text
        text = text.strip()
        if text != text_before:
            changes += 1
        
        # Accents
        text_before = text
        text, has_accent = TextNormalizer.remove_accents(text)
        if has_accent:
            changes += 1
        
        # Spaces multiples
        text_before = text
        text = re.sub(r'\s+', ' ', text)
        if text != text_before:
            changes += 1
        
        return text, changes
    
    @staticmethod
    def normalize_email(email: str) -> str:
        if not isinstance(email, str):
            return ''
        return email.strip().lower()
    
    @staticmethod
    def normalize_phone(phone: str) -> str:
        if not isinstance(phone, str):
            return ''
        
        phone = re.sub(r'[\s\-\(\)]', '', phone.strip())
        
        if phone.startswith('+33'):
            phone = '0' + phone[3:]
        
        return phone


# ═══════════════════════════════════════════════════════════════════════
# NETTOYEUR PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════

class DataCleaner:
    """Engine de nettoyage"""
    
    def __init__(self):
        self.metrics = None
        self.records = []
        self.validation = Validator()
        self.normalizer = TextNormalizer()
    
    def load_csv(self, filepath: str) -> pd.DataFrame:
        """Charge CSV"""
        logger.info(f"[LOAD] Fichier: {filepath}")
        
        try:
            df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')
            logger.info(f"[OK] {len(df)} lignes chargees (UTF-8)")
            return df
        except UnicodeDecodeError:
            logger.warning("[WARN] UTF-8 fail, essai latin-1...")
            df = pd.read_csv(filepath, encoding='latin-1', on_bad_lines='skip')
            logger.info(f"[OK] {len(df)} lignes chargees (latin-1)")
            return df
    
    def validate_columns(self, df: pd.DataFrame) -> bool:
        """Valide colonnes"""
        required = {'nom', 'email', 'telephone', 'entreprise'}
        cols = set(df.columns)
        
        if not required.issubset(cols):
            missing = required - cols
            logger.error(f"[ERROR] Colonnes manquantes: {missing}")
            return False
        
        logger.info(f"[OK] Colonnes validees: {required}")
        return True
    
    def clean_record(self, row: pd.Series) -> Tuple[Optional[Record], Dict]:
        """Nettoie UN enregistrement"""
        stats = {
            'skipped': False,
            'skip_reason': None,
            'changes': 0,
            'invalid_email': False,
            'invalid_phone': False,
        }
        
        # Extraction
        nom_raw = str(row.get('nom', '')).strip()
        email_raw = str(row.get('email', '')).strip()
        phone_raw = str(row.get('telephone', '')).strip()
        entreprise_raw = str(row.get('entreprise', '')).strip()
        
        # Validations
        if not nom_raw or nom_raw.lower() == 'nan':
            stats['skipped'] = True
            stats['skip_reason'] = 'nom_vide'
            return None, stats
        
        if not email_raw or email_raw.lower() == 'nan':
            stats['skipped'] = True
            stats['skip_reason'] = 'email_vide'
            return None, stats
        
        # Normalisation
        nom_norm, nom_changes = self.normalizer.normalize_text(nom_raw)
        email_norm = self.normalizer.normalize_email(email_raw)
        phone_norm = self.normalizer.normalize_phone(phone_raw)
        entreprise_norm, ent_changes = self.normalizer.normalize_text(entreprise_raw)
        
        stats['changes'] += nom_changes + ent_changes
        
        # Validation
        if not self.validation.is_valid_email(email_norm):
            stats['invalid_email'] = True
            stats['skipped'] = True
            stats['skip_reason'] = 'email_invalide'
            return None, stats
        
        if phone_norm and not self.validation.is_valid_phone(phone_norm):
            stats['invalid_phone'] = True
            phone_norm = ''
        
        record = Record(
            nom=nom_norm,
            email=email_norm,
            telephone=phone_norm,
            entreprise=entreprise_norm if entreprise_norm else 'N/A'
        )
        
        return record, stats
    
    def clean(self, filepath: str, remove_duplicates: bool = True) -> Dict:
        """Pipeline COMPLET"""
        start_time = datetime.now()
        
        try:
            # 1. Charge
            df = self.load_csv(filepath)
            if df.empty:
                raise ValueError("CSV vide")
            
            input_rows = len(df)
            
            # 2. Valide colonnes
            if not self.validate_columns(df):
                raise ValueError("Colonnes invalides")
            
            # 3. Nettoie
            records = []
            empty_rows = 0
            invalid_emails = 0
            invalid_phones = 0
            
            for idx, row in df.iterrows():
                record, stats = self.clean_record(row)
                
                if stats['skipped']:
                    empty_rows += 1
                    if stats['skip_reason'] == 'email_invalide':
                        invalid_emails += 1
                else:
                    records.append(record)
                
                if stats['invalid_phone']:
                    invalid_phones += 1
            
            # 4. Dédoublons
            duplicates = 0
            if remove_duplicates:
                before = len(records)
                unique_emails = set()
                unique_records = []
                for rec in records:
                    if rec.email not in unique_emails:
                        unique_emails.add(rec.email)
                        unique_records.append(rec)
                duplicates = before - len(unique_records)
                records = unique_records
                logger.info(f"[DEDUP] -{duplicates} doublons")
            
            output_rows = len(records)
            
            # 5. Score
            quality_score = self._calculate_quality_score(
                input_rows, output_rows, invalid_emails, invalid_phones
            )
            
            # 6. Métriques
            exec_time = (datetime.now() - start_time).total_seconds()
            self.metrics = CleaningMetrics(
                input_rows=input_rows,
                output_rows=output_rows,
                duplicates_removed=duplicates,
                empty_rows_removed=empty_rows,
                invalid_emails=invalid_emails,
                invalid_phones=invalid_phones,
                accents_normalized=0,
                whitespace_trimmed=0,
                execution_time=exec_time,
                quality_score=quality_score,
                error_message=None
            )
            
            logger.info(f"[SUCCESS] Nettoyage OK ({exec_time:.3f}s)")
            logger.info(f"[RESULT] {input_rows} -> {output_rows} | Score: {quality_score}/100")
            
            return {
                "success": True,
                "data": [rec.to_dict() for rec in records],
                "metrics": asdict(self.metrics)
            }
        
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"[FATAL] {error_msg}")
            
            exec_time = (datetime.now() - start_time).total_seconds()
            self.metrics = CleaningMetrics(
                input_rows=0, output_rows=0, duplicates_removed=0,
                empty_rows_removed=0, invalid_emails=0, invalid_phones=0,
                accents_normalized=0, whitespace_trimmed=0,
                execution_time=exec_time, quality_score=0,
                error_message=error_msg
            )
            
            return {
                "success": False,
                "error": error_msg,
                "metrics": asdict(self.metrics)
            }
    
    def _calculate_quality_score(
        self, input_rows: int, output_rows: int,
        invalid_emails: int, invalid_phones: int
    ) -> float:
        """Calcule score (0-100)"""
        if input_rows == 0:
            return 0.0
        
        retention = output_rows / input_rows
        base = retention * 100
        
        email_penalty = (invalid_emails / input_rows) * 20
        phone_penalty = (invalid_phones / input_rows) * 10
        
        final = max(0, base - email_penalty - phone_penalty)
        final = min(100, final)
        
        return round(final, 2)


# ═══════════════════════════════════════════════════════════════════════
# EXPORTERS
# ═══════════════════════════════════════════════════════════════════════

class Exporter:
    """Export JSON/CSV"""
    
    @staticmethod
    def to_json(result: Dict, output_path: str = 'output.json') -> str:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"[EXPORT] JSON: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"[ERROR] Export JSON: {e}")
            return None
    
    @staticmethod
    def to_csv(result: Dict, output_path: str = 'output.csv') -> str:
        try:
            if not result.get('success'):
                logger.warning("[WARN] Pas de donnees a exporter")
                return None
            
            df = pd.DataFrame(result['data'])
            df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"[EXPORT] CSV: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"[ERROR] Export CSV: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════
# AUTO-DETECTION FICHIERS TEST
# ═══════════════════════════════════════════════════════════════════════

def find_test_file() -> Optional[str]:
    """Auto-detect fichier de test"""
    test_files = [
        '1_test_extreme.csv',
        '2_test_accents.csv',
        '3_test_international.csv',
        '4_test_vides.csv',
        '5_test_erreurs.csv',
        '6_test_1000lignes.csv',
    ]
    
    for fname in test_files:
        if Path(fname).exists():
            logger.info(f"[DETECT] Fichier trouve: {fname}")
            return fname
    
    logger.warning("[WARN] Pas de fichier de test detecte")
    return None


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main(csv_file: Optional[str] = None, output_format: str = 'both') -> Dict:
    """Orchestrateur principal"""
    
    # Auto-detect fichier
    if not csv_file:
        csv_file = find_test_file()
    
    if not csv_file or not Path(csv_file).exists():
        logger.error(f"[FATAL] Fichier CSV introuvable: {csv_file}")
        return {
            "success": False,
            "error": f"Fichier CSV introuvable: {csv_file}",
            "metrics": asdict(CleaningMetrics(
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                f"Fichier introuvable: {csv_file}"
            ))
        }
    
    logger.info("=" * 70)
    logger.info("[START] CRM CLEANER v2.5")
    logger.info(f"[INPUT] Fichier: {csv_file}")
    logger.info("=" * 70)
    
    # Nettoie
    cleaner = DataCleaner()
    result = cleaner.clean(csv_file, remove_duplicates=True)
    
    # Exporte
    Path('output').mkdir(exist_ok=True)
    
    json_path = None
    csv_path = None
    
    if output_format in ['json', 'both']:
        json_path = Exporter.to_json(
            result,
            f"output/cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
    
    if output_format in ['csv', 'both']:
        csv_path = Exporter.to_csv(
            result,
            f"output/cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
    
    # Resume
    if result['success']:
        m = result['metrics']
        logger.info("=" * 70)
        logger.info("[STATS] Nettoyage termine")
        logger.info(f"  Input:      {m['input_rows']} lignes")
        logger.info(f"  Output:     {m['output_rows']} lignes")
        logger.info(f"  Doublons:   -{m['duplicates_removed']}")
        logger.info(f"  Erreurs:    {m['invalid_emails']} emails + {m['invalid_phones']} phones")
        logger.info(f"  Score:      {m['quality_score']}/100")
        logger.info(f"  Temps:      {m['execution_time']:.3f}s")
        logger.info("=" * 70)
    else:
        logger.error(f"[FATAL] {result['error']}")
    
    return result


if __name__ == '__main__':
    result = main()
    print(json.dumps(result, indent=2, ensure_ascii=False))
