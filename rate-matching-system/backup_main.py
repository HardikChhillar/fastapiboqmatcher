# backup_main.py - COMPLETE FIXED FastAPI Rate Matching Application
"""
Dependencies to install:
pip install fastapi uvicorn pandas openai scikit-learn nltk sentence-transformers pint python-multipart python-dotenv
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import logging
import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Set
import json
import os
import pickle
from datetime import datetime, timedelta
import openai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import time
from pint import UnitRegistry
import re
from pathlib import Path
import tempfile
from fastapi.middleware.cors import CORSMiddleware
import hashlib
from io import BytesIO
from pandas import Index
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import itertools
from scipy import sparse
from rapidfuzz import fuzz
import torch
from typing import Dict, Set, Tuple, Optional
import faiss
import ast

# Add construction-specific synonym dictionary
SYNONYM_DICT = {
    # Existing synonyms
    "P/L": "providing and laying",
    "laying": "fixing",
    "fix": "lay",
    "tile": "tiles",
    "wall": "vertical surface",
    "flooring": "floor",
    "RM": "running meter",
    "Rmt": "running meter",
    "sqm": "square meter",
    "mm": "millimeter",
    "thk": "thickness",
    "with adhesive": "using cement paste",
    "plastering": "applying plaster",
    "providing": "p/f",
    "p/f": "providing and fixing",
    "supplying": "supply and",
    "installing": "installation of",
    "dia": "diameter",
    "thick": "thickness",
    "nos": "numbers",
    "no.": "number",
    "m2": "sqm",
    "m³": "cum",
    "cum": "cubic meter",
    "rft": "running feet",
    "sft": "square feet",
    "including": "incl.",
    "excluding": "excl.",
    "complete": "completed",
    
    # Additional synonyms for better rephrasing support
    "providing and fixing": "p/f",
    "supply and fix": "p/f",
    "supply and install": "p/f",
    "supply & fix": "p/f",
    "supply & install": "p/f",
    "supply and laying": "p/l",
    "supply & laying": "p/l",
    "providing & laying": "p/l",
    "provide and lay": "p/l",
    "provide & lay": "p/l",
    "installation": "fixing",
    "installed": "fixed",
    "mounting": "fixing",
    "mounted": "fixed",
    "erecting": "fixing",
    "erection": "fixing",
    "erected": "fixed",
    "placing": "fixing",
    "placed": "fixed",
    "fitting": "fixing",
    "fitted": "fixed",
    "applying": "fixing",
    "applied": "fixed",
    "construction": "constructing",
    "constructed": "constructing",
    "built": "constructing",
    "building": "constructing",
    "fabrication": "fabricating",
    "fabricated": "fabricating",
    "manufacturing": "fabricating",
    "manufactured": "fabricating",
    "made": "fabricating",
    "making": "fabricating",
    "square metres": "sqm",
    "square meters": "sqm",
    "sq. metres": "sqm",
    "sq. meters": "sqm",
    "cubic metres": "cum",
    "cubic meters": "cum",
    "cu. metres": "cum",
    "cu. meters": "cum",
    "running metres": "rmt",
    "running meters": "rmt",
    "r. metres": "rmt",
    "r. meters": "rmt",
    "millimetres": "mm",
    "millimeters": "mm",
    "centimetres": "cm",
    "centimeters": "cm",
    "metres": "m",
    "meters": "m",
    "numbers": "nos",
    "pieces": "nos",
    "units": "nos",
    "quantity": "nos",
    "pcs": "nos",
    "pc": "nos",
}

# Add unit conversion mapping
UNIT_CONVERSION_MAP = {
    "sqm": {"square meter", "m2", "sq.m", "sq m"},
    "cum": {"cubic meter", "m3", "cu.m", "cu m"},
    "rmt": {"running meter", "rm", "r.m", "meter"},
    "rft": {"running feet", "rf", "r.f", "feet"},
    "mm": {"millimeter", "millimetre"},
    "kg": {"kilogram", "kgs", "kilos"},
    "nos": {"numbers", "no.", "no", "pieces", "pcs"},
    "ls": {"lumpsum", "l.s.", "lot"},
}

class MatchResult:
    """Class to store match results with metadata"""
    def __init__(self, index: int, description: str, score: float, unit: str):
        self.index = index
        self.description = description
        self.score = score
        self.unit = unit
        self.unit_compatible = False
        self.unit_message = ""
        self.timestamp = datetime.now()

def normalize_description(desc: str) -> str:
    """
    Normalize description using synonym dictionary and basic text processing
    """
    try:
        if not isinstance(desc, str):
            return ""
        
        # Convert to lowercase and normalize whitespace
        normalized = desc.lower().strip()
        
        # Replace multiple spaces with single space
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Replace synonyms (do multiple passes to catch nested synonyms)
        for _ in range(2):  # Two passes to catch nested replacements
            for k, v in SYNONYM_DICT.items():
                normalized = re.sub(rf"\b{k.lower()}\b", v.lower(), normalized)
        
        # Normalize numbers with units (e.g., "300 x 300" to "300x300")
        normalized = re.sub(r'(\d+)\s*[xX]\s*(\d+)', r'\1x\2', normalized)
        
        # Normalize units
        for std_unit, variants in UNIT_CONVERSION_MAP.items():
            for variant in variants:
                normalized = re.sub(rf"\b{variant}\b", std_unit, normalized)
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
        
    except Exception as e:
        logger.error(f"Error in normalize_description: {e}")
        return desc if isinstance(desc, str) else ""

def get_fuzzy_score(text1: str, text2: str) -> float:
    """
    Get fuzzy matching score between two texts
    """
    try:
        return fuzz.token_sort_ratio(text1, text2) / 100.0
    except Exception as e:
        logger.error(f"Error in fuzzy matching: {e}")
        return 0.0

def get_unit_compatibility(unit1: str, unit2: str) -> Tuple[bool, str]:
    """
    Check if two units are compatible
    Returns (is_compatible, explanation)
    """
    try:
        unit1 = unit1.lower().strip()
        unit2 = unit2.lower().strip()
        
        # Direct match
        if unit1 == unit2:
            return True, "Units match exactly"
            
        # Check in conversion map
        for std_unit, variants in UNIT_CONVERSION_MAP.items():
            if unit1 in variants and unit2 in variants:
                return True, f"Units are compatible variants of {std_unit}"
                
        # Try pint conversion as fallback
        try:
            result = function2_convert_rate_by_unit(1.0, unit1, unit2)
            if result is not None:
                return True, "Units are convertible"
        except:
            pass
            
        return False, "Units are not compatible"
        
    except Exception as e:
        logger.error(f"Error in unit compatibility check: {e}")
        return False, str(e)

# Create FastAPI app
app = FastAPI(title="Rate Matching System")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]  # Add this to expose custom headers
)

# Global variables for caching
sentence_model = None
ureg = UnitRegistry()
dataset_df = None
match_logs = []  # Initialize without limit
MAX_LOGS = 1000  # Add this constant

# Enhanced caching system
matching_cache = {}
cache_file = "rate-matching-system/backend/matching_cache.pkl"
token_usage_log = []

# Vector index for fast similarity search
vector_index = None
vector_descriptions = []
last_index_update = None

# FIXED CONFIGURATION - Adjusted thresholds for better matching
EMBEDDING_SIMILARITY_THRESHOLD = 0.90  # Lowered threshold for more matches
MAX_OPENAI_CANDIDATES = 15  # Increased number of candidates
CACHE_EXPIRY_DAYS = 30
MAX_TOKENS_PER_DAY = 5000000  # More reasonable daily limit
current_day_tokens = 0
current_day = datetime.now().date()

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# OpenAI configuration (set your API key as environment variable)
openai.api_key = os.getenv("OPENAI_API_KEY")

# FIXED: Dataset Rate Conversion Logic
DATASET_CONVERSION_FACTORS = {
    # Linear conversions (Meter ↔ Feet)
    ('mtr', 'feet'): 3.37,
    ('feet', 'mtr'): 3.37,
    ('rmt', 'rft'): 3.37,
    ('rft', 'rmt'): 3.37,
    
    # Area conversions (Square metre ↔ Square feet)
    ('sqm', 'sqft'): 10.76,
    ('SQM','sft'): 10.76,
    ('sqft', 'sqm'): 10.76,
    ('sqmtr', 'sft'): 10.76,   # Critical: Your dataset uses "Sqmtr" and query uses "sft"
    ('sft', 'sqmtr'): 10.76,
    
    # Volume conversions (Cubic metre ↔ Cubic feet)
    ('cum', 'cuft'): 35.31,
    ('cuft', 'cum'): 35.31,
}

# Enhanced unit aliases to normalize unit names
ENHANCED_UNIT_ALIASES = {
    # Linear units
    'mtr': 'mtr',
    'mtr.': 'mtr',
    'm': 'mtr',
    'meter': 'mtr',
    'metre': 'mtr',
    'rmt': 'mtr',
    'rft': 'feet',
    'rft.': 'feet',
    'feet': 'feet',
    'foot': 'feet',
    'ft': 'feet',
    
    # Area units - CRITICAL: Map all variations to standard forms
    'sqm': 'sqm',
    'sq.m': 'sqm',
    'sqmt': 'sqm',
    'sqmtr': 'sqmtr',  # Keep this separate as it's common in your dataset
    'square meter': 'sqm',
    'square metre': 'sqm',
    'sqft': 'sqft',
    'sft': 'sft',      # Keep this separate as it's common in queries
    'sft.': 'sft',
    'sq.ft': 'sqft',
    'square feet': 'sqft',
    'square foot': 'sqft',
    
    # Volume units
    'cum': 'cum',
    'cu.m': 'cum',
    'cubic meter': 'cum',
    'cubic metre': 'cum',
    'cuft': 'cuft',
    'cu.ft': 'cuft',
    'cubic feet': 'cuft',
    'cubic foot': 'cuft',
    'cft': 'cuft',
    
    # Weight/Count units
    'kg': 'kg',
    'kilogram': 'kg',
    'nos': 'nos',
    'nos.': 'nos',
    'no.': 'nos',
    'number': 'nos',
    'each': 'nos',
    'no': 'nos',
    
    # Other units from your dataset
    'lumsum': 'lumsum',
    'lumpsum': 'lumsum',
    'ls': 'lumsum',
    'l.s': 'lumsum',
    'job': 'job',
    'trip': 'trip',
    'trolley': 'trolley',
    'pair': 'pair',
    'mt': 'mt',
    'pt': 'pt',
}

def normalize_unit_name(unit: str) -> str:
    """Normalize unit name using enhanced aliases"""
    if not unit:
        return ""
    unit_clean = unit.lower().strip()
    return ENHANCED_UNIT_ALIASES.get(unit_clean, unit_clean)

def parse_dataset_description(desc_str: str) -> List[str]:
    """
    Parse dataset description string which might be in list format
    
    Args:
        desc_str: Description string from dataset (might be like "['description text']")
    
    Returns:
        List of description strings
    """
    try:
        if isinstance(desc_str, str):
            # Try to parse as Python list literal
            if desc_str.startswith('[') and desc_str.endswith(']'):
                try:
                    parsed = ast.literal_eval(desc_str)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed]
                except:
                    pass
            # If not a list format, treat as single string
            return [desc_str.strip()]
        elif isinstance(desc_str, list):
            return [str(item).strip() for item in desc_str]
        else:
            return [str(desc_str).strip()]
    except Exception as e:
        logger.error(f"Error parsing dataset description: {e}")
        return [str(desc_str)]

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class InputItem(BaseModel):
    description: str
    unit: str
    rate: float

class DatasetItem(BaseModel):
    description: List[str]
    unit: str
    rate: float

class QueryItem(BaseModel):
    description: str
    unit: str
    rate: Optional[float] = None

class MatchLog(BaseModel):
    timestamp: datetime
    query_description: str
    matched_description: str
    query_unit: str
    matched_unit: str
    rate: float
    status: str
    unit_mismatch: bool

class RateUpdateResponse(BaseModel):
    """Response model for rate updates that need manual intervention"""
    description: str
    dataset_rate: float
    input_rate: float
    dataset_unit: str
    input_unit: str
    needs_manual_rate: bool
    reason: str

# ============================================================================
# CACHING UTILITIES
# ============================================================================

def load_cache():
    """Load matching cache from disk"""
    global matching_cache
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                # Filter out expired entries
                current_time = datetime.now()
                matching_cache = {
                    k: v for k, v in cache_data.items()
                    if current_time - v['timestamp'] < timedelta(days=CACHE_EXPIRY_DAYS)
                }
            logger.info(f"Loaded {len(matching_cache)} cached matches")
    except Exception as e:
        logger.error(f"Error loading cache: {e}")
        matching_cache = {}

def save_cache():
    """Save matching cache to disk"""
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(matching_cache, f)
        logger.info(f"Saved {len(matching_cache)} cached matches")
    except Exception as e:
        logger.error(f"Error saving cache: {e}")

def get_cache_key(input_desc: str, candidates: List[str]) -> str:
    """Generate a cache key for the input and candidates"""
    candidates_str = "|".join(sorted(candidates))
    cache_string = f"{input_desc}:{candidates_str}"
    return hashlib.md5(cache_string.encode()).hexdigest()

def log_token_usage(input_tokens: int, output_tokens: int, operation: str):
    """Log token usage for monitoring"""
    global current_day_tokens, current_day
    
    today = datetime.now().date()
    if today != current_day:
        current_day = today
        current_day_tokens = 0
    
    total_tokens = input_tokens + output_tokens
    current_day_tokens += total_tokens
    
    token_usage_log.append({
        'timestamp': datetime.now(),
        'operation': operation,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': total_tokens,
        'daily_total': current_day_tokens
    })
    
    logger.info(f"Token usage - Operation: {operation}, Tokens: {total_tokens}, Daily total: {current_day_tokens}")

def check_token_limit() -> bool:
    """Check if daily token limit has been reached"""
    global current_day_tokens, current_day
    
    today = datetime.now().date()
    if today != current_day:
        current_day = today
        current_day_tokens = 0
    
    return current_day_tokens < MAX_TOKENS_PER_DAY

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def initialize_model():
    """Initialize the sentence transformer model"""
    global sentence_model
    if sentence_model is None:
        try:
            sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model: {e}")
            raise HTTPException(status_code=500, detail="Failed to initialize ML model")
    return sentence_model

def load_dataset():
    """FIXED: Load the internal dataset from CSV with proper description parsing"""
    global dataset_df
    dataset_path = "dataset.csv"
    
    if os.path.exists(dataset_path):
        try:
            dataset_df = pd.read_csv(dataset_path)
            
            # FIXED: Parse descriptions that are in string format like "['description']"
            if 'description' in dataset_df.columns:
                dataset_df['description'] = dataset_df['description'].apply(parse_dataset_description)
            
            logger.info(f"Loaded dataset with {len(dataset_df)} entries")
            
            # Log some examples for debugging
            for i in range(min(3, len(dataset_df))):
                desc = dataset_df.iloc[i]['description']
                unit = dataset_df.iloc[i]['unit']
                rate = dataset_df.iloc[i]['rate']
                logger.info(f"Dataset entry {i}: {desc[0][:50] if isinstance(desc, list) else str(desc)[:50]}... - {unit} - {rate}")
                
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            dataset_df = pd.DataFrame({
                'description': pd.Series(dtype='object'),
                'unit': pd.Series(dtype='str'),
                'rate': pd.Series(dtype='float64')
            })
    else:
        dataset_df = pd.DataFrame({
            'description': pd.Series(dtype='object'),
            'unit': pd.Series(dtype='str'),
            'rate': pd.Series(dtype='float64')
        })
        logger.info("Created new empty dataset")
    
    return dataset_df

def save_dataset():
    """Save the dataset to CSV"""
    global dataset_df
    if dataset_df is not None:
        try:
            # Convert list descriptions to string representation for CSV storage
            save_df = dataset_df.copy()
            save_df['description'] = save_df['description'].apply(str)
            save_df.to_csv("dataset.csv", index=False)
            logger.info("Dataset saved successfully")
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")

def load_important_terms() -> set:
    """Load important terms from the JSON file"""
    try:
        terms_file = "important_terms.json"
        if os.path.exists(terms_file):
            with open(terms_file, 'r') as f:
                terms = set(json.load(f))
            return terms
        return set()
    except Exception as e:
        logger.error(f"Error loading important terms: {e}")
        return set()

def get_embeddings(texts: List[str]) -> np.ndarray:
    """Get sentence embeddings using sentence-transformers"""
    model = initialize_model()
    try:
        embeddings = model.encode(texts)
        return np.array(embeddings)  # Convert to numpy array explicitly
    except Exception as e:
        logger.error(f"Error getting embeddings: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate embeddings")

def estimate_tokens(text: str) -> int:
    """Estimate token count for text (rough approximation)"""
    return int(len(text.split()) * 1.3)  # Rough estimate: 1.3 tokens per word

# FIXED: Simplified quota tracker without aggressive limits
quota_tracker = {"daily_usage": 0, "last_reset": datetime.now().date()}

def reset_quota_if_needed():
    today = datetime.now().date()
    if quota_tracker["last_reset"] != today:
        quota_tracker["daily_usage"] = 0
        quota_tracker["last_reset"] = today

DEFAULT_GPT_MODEL = "gpt-3.5-turbo"

def call_openai_for_matching(input_description: str, candidates: List[str], model: str = DEFAULT_GPT_MODEL) -> tuple:
    """
    FIXED: Use GPT model to select the best match from candidates with proper error handling.
    """
    
    if not openai.api_key:
        logger.warning("OpenAI API key not set, using fallback matching")
        return "no match", 0, 0
   
    # Reset quota if new day
    reset_quota_if_needed()
   
    # FIXED: Removed overly aggressive quota limits
    if quota_tracker["daily_usage"] > MAX_TOKENS_PER_DAY * 0.9:  # 90% of daily limit
        logger.warning("Daily usage limit approaching, using fallback matching")
        return "no match", 0, 0
    
    # Check cache first
    cache_key = get_cache_key(input_description, candidates)
    if cache_key in matching_cache:
        logger.info("Using cached result for matching")
        return matching_cache[cache_key]['result'], 0, 0
    
    try:
        # Limit candidates to reduce token usage
        limited_candidates = candidates[:MAX_OPENAI_CANDIDATES]
        
        candidates_text = ""
        for i, candidate in enumerate(limited_candidates, 1):
            candidates_text += f"{i}. \"{candidate}\"\n"

        # Simplified prompt to reduce token usage
        prompt = f"""You are an expert in construction industry terminology and activity matching.

Your task is to match a human-written QUERY activity to one of the candidate construction activity descriptions below. The candidate that most closely **represents the same core activity** as the QUERY should be selected.

✅ Consider:
- Rephrased wording (e.g., "fixing" vs "installation")
- Synonyms (e.g., "tile" vs "ceramic surface")
- Human spelling errors
- Unit differences, as long as the **meaning** is the same
- Brand names may or may not match (ignore if core task is same)
- If both descriptions are refering same activity then it is a match.

❌ Do NOT match if:
- The core construction task is **different**
- The activity describes a **different trade** (e.g., plumbing vs masonry)
- The activity includes different **materials or purposes** (e.g., paint vs primer)
- The activity specific dimensions and appearance are different

QUERY:
\"\"\"{input_description}\"\"\"

CANDIDATES:
{candidates_text}

Give ONLY one exact candidate from the list above that best matches the QUERY, or return exactly:
no match
"""

        # FIXED: Proper retry logic with exponential backoff
        max_retries = 3
        base_wait_time = 1
        
        for attempt in range(max_retries):
            try:
                response = openai.chat.completions.create(
                   model=model,
                   messages=[
                     {
                         "role": "system",
                         "content": "You are a construction domain expert. You can match descriptions even if phrased differently, spelled wrong, or partially detailed."
                     },
                     {
                         "role": "user",
                         "content": prompt
                     }
                    ],
                     max_tokens=15000,
                     temperature=0.2,
                     timeout=30
                )

                break
            except openai.RateLimitError as e:
                logger.warning(f"Rate limit error: {e}")
                # FIXED: Proper exponential backoff
                wait_time = base_wait_time * (2 ** attempt) + np.random.uniform(0, 1)
                if attempt < max_retries - 1:
                    logger.info(f"Waiting {wait_time:.1f} seconds before retry {attempt + 1}")
                    time.sleep(wait_time)
                else:
                    logger.error("Max retries exceeded due to rate limiting")
                    return "no match", 0, 0
            except openai.APIError as e:
                logger.error(f"OpenAI API error: {e}")
                if "insufficient_quota" in str(e).lower():
                    logger.error("OpenAI quota exceeded - please check billing")
                    return "no match", 0, 0
                if attempt < max_retries - 1:
                    time.sleep(base_wait_time)
                else:
                    return "no match", 0, 0
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(base_wait_time)
                else:
                    return "no match", 0, 0
        
        if response and response.choices and response.choices[0].message:
            result = response.choices[0].message.content
            if result:
                result = result.strip()
                
                # Clean up result
                if result.startswith('"') and result.endswith('"'):
                    result = result[1:-1]
                
                # Log token usage if usage exists
                if response.usage:
                    input_tokens = response.usage.prompt_tokens
                    output_tokens = response.usage.completion_tokens
                    quota_tracker["daily_usage"] += input_tokens + output_tokens
                    log_token_usage(input_tokens, output_tokens, "description_matching")
                
                # Cache the result
                matching_cache[cache_key] = {
                    'result': result,
                    'timestamp': datetime.now()
                }
                
                logger.info(f"OpenAI matching result: {result}")
                return result, input_tokens if 'input_tokens' in locals() else 0, output_tokens if 'output_tokens' in locals() else 0
        
        return "no match", 0, 0
        
    except Exception as e:
        logger.error(f"OpenAI unexpected error: {e}")
        return "no match", 0, 0

def function1_match_description_with_openai_enhanced(input_description: str, dataset_descriptions: List[List[str]], query_unit: Optional[str] = None) -> tuple:
    """
    Enhanced matching using OpenAI for all potential matches
    """
    if not dataset_descriptions or dataset_df is None:
        logger.info("No dataset descriptions provided or dataset not loaded")
        return -1, None
    
    try:
        # Generate cache key
        cache_key = hashlib.md5(f"{input_description}:{query_unit}".encode()).hexdigest()
        
        # Check cache first
        if cache_key in matching_cache:
            cached = matching_cache[cache_key]
            if datetime.now() - cached['timestamp'] < timedelta(days=CACHE_EXPIRY_DAYS):
                logger.info("Using cached match result")
                return cached['index'], cached['description']
        
        # Normalize input description
        normalized_input = normalize_description(input_description)
        logger.info(f"Normalized input: {normalized_input}")
        
        # Get input embedding
        input_embedding = get_embeddings([normalized_input])[0]
        
        # Get embeddings for all descriptions
        all_candidates = []
        candidate_to_index = []
        
        for idx, desc_list in enumerate(dataset_descriptions):
            for desc in desc_list:
                normalized_desc = normalize_description(desc)
                all_candidates.append(normalized_desc)
                candidate_to_index.append(idx)
        
        if not all_candidates:
            return -1, None
        
        # Get embeddings and similarities
        candidate_embeddings = get_embeddings(all_candidates)
        similarities = cosine_similarity(input_embedding.reshape(1, -1), candidate_embeddings)[0]
        
        # Get fuzzy match scores
        fuzzy_scores = np.array([
            get_fuzzy_score(normalized_input, cand)
            for cand in all_candidates
        ])
        
        # Combine scores
        combined_scores = (0.7 * similarities) + (0.3 * fuzzy_scores)
        
        # Get top candidates for OpenAI
        top_k = min(MAX_OPENAI_CANDIDATES, len(all_candidates))
        top_indices = np.argsort(combined_scores)[-top_k:][::-1]
        top_candidates = [all_candidates[i] for i in top_indices]
        top_original_indices = [candidate_to_index[i] for i in top_indices]
        
        # Try OpenAI for validation with enhanced prompt
        try:
            system_msg = """You are a construction estimation expert. Your task is to find exact matches or very close matches between construction descriptions.

Key matching rules:
1. Match descriptions that refer to the same work/activity even if worded differently
2. Consider synonyms and abbreviations (e.g., 'P/F' = 'Providing and Fixing')
3. Match if only minor details differ but core work is identical
4. Match if units or measurements are written differently but mean the same
5. Match if only the formatting or word order is different

Do NOT match if:
1. Different materials are specified
2. Different methods of construction are specified
3. Different locations or applications are specified
4. Different quality grades or standards are specified"""

            user_msg = f"""Compare this input description:
"{input_description}"

With these potential matches:
{chr(10).join(f"{i+1}. {desc}" for i, desc in enumerate(top_candidates))}

Return ONLY the number (1-{len(top_candidates)}) of the best matching description, or 0 if none match.
Just return the number, no explanation."""
            
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            if response and response.choices and response.choices[0].message:
                result = response.choices[0].message.content.strip()
                try:
                    selected_idx = int(result)
                    if 1 <= selected_idx <= len(top_candidates):
                        matched_idx = top_original_indices[selected_idx - 1]
                        matched_desc = dataset_descriptions[matched_idx][0]  # Get original description
                        
                        # Cache the result
                        matching_cache[cache_key] = {
                            'index': matched_idx,
                            'description': matched_desc,
                            'timestamp': datetime.now()
                        }
                        
                        return matched_idx, matched_desc
                except ValueError:
                    logger.warning(f"Invalid OpenAI response format: {result}")
        
        except Exception as e:
            logger.error(f"OpenAI matching error: {e}")
        
        return -1, None
        
    except Exception as e:
        logger.error(f"Error in matching function: {e}")
        return -1, None

def function2_convert_rate_by_unit(rate: float, from_unit: str, to_unit: str) -> Optional[float]:
    """
    FIXED: Convert rate from one unit to another using dataset conversion logic.
    
    This function handles rate conversions based on your conversion table:
    - 350 Sqmtr → 32.53 sft (divide by 10.76)
    - 900 Sqmtr → 83.64 sft (divide by 10.76)
    
    Args:
        rate: The rate value to convert
        from_unit: Source unit (e.g., 'Sqmtr', 'sft')
        to_unit: Target unit (e.g., 'sft', 'Sqmtr')
    
    Returns:
        Converted rate value or None if conversion not possible
    """
    try:
        # Normalize unit names
        from_unit_norm = normalize_unit_name(from_unit)
        to_unit_norm = normalize_unit_name(to_unit)
        
        logger.info(f"Converting rate: {rate} from '{from_unit}' ({from_unit_norm}) to '{to_unit}' ({to_unit_norm})")
        
        # If same unit, no conversion needed
        if from_unit_norm == to_unit_norm:
            logger.info(f"Same units, no conversion needed: {rate}")
            return float(rate)
        
        # Check if conversion exists in our dataset conversion factors
        conversion_key = (from_unit_norm, to_unit_norm)
        
        if conversion_key in DATASET_CONVERSION_FACTORS:
            factor = DATASET_CONVERSION_FACTORS[conversion_key]
            logger.info(f"Found conversion factor: {factor} for {conversion_key}")
            
            # Apply conversion logic based on your table
            if from_unit_norm in ('sqmt','sqmtr','sqm','SQM')  and to_unit_norm in ('sft','sqft'):
                # Sqmtr → sft: Divide by 10.76 (350 Sqmtr → 32.53 sft)
                converted_rate = rate / factor
                logger.info(f"Sqmtr to sft conversion: {rate} / {factor} = {converted_rate}")
                
            elif from_unit_norm in ('sft','sqft') and to_unit_norm in ('sqmtr','sqm','SQM','sqmt'):
                # sft → Sqmtr: Multiply by 10.76 (32.53 sft → 350 Sqmtr)
                converted_rate = rate * factor
                logger.info(f"sft to Sqmtr conversion: {rate} * {factor} = {converted_rate}")
                
            elif from_unit_norm in ('mtr','meter') and to_unit_norm in ('feet','ft'):
                # Mtr → Feet: Divide by 3.37
                converted_rate = rate / factor
                logger.info(f"Mtr to feet conversion: {rate} / {factor} = {converted_rate}")
                
            elif from_unit_norm in ('feet','ft') and to_unit_norm in ('mtr','meter'):
                # Feet → Mtr: Multiply by 3.37
                converted_rate = rate * factor
                logger.info(f"Feet to Mtr conversion: {rate} * {factor} = {converted_rate}")
                
            elif from_unit_norm in ('cum','cumtr') and to_unit_norm in ('cuft','cubic feet'):
                # Cum → Cuft: Divide by 35.31
                converted_rate = rate / factor
                logger.info(f"Cum to cuft conversion: {rate} / {factor} = {converted_rate}")
                
            elif from_unit_norm in ('cuft','cubic feet') and to_unit_norm in ('cum','cumtr'):
                # Cuft → Cum: Multiply by 35.31
                converted_rate = rate * factor
                logger.info(f"Cuft to cum conversion: {rate} * {factor} = {converted_rate}")
            
            else:
                logger.warning(f"Conversion key found but no logic defined for {conversion_key}")
                return None
                
            return float(converted_rate)
        
        # If no conversion found, log warning and return None
        logger.warning(f"No conversion available from {from_unit} to {to_unit}")
        return None
        
    except Exception as e:
        logger.error(f"Unit conversion error from {from_unit} to {to_unit}: {e}")
        return None

def function2_convert_rate_by_unit_with_openai(rate: float, from_unit: str, to_unit: str) -> Optional[tuple[float, str]]:
    """
    Convert rate from one unit to another using OpenAI with construction industry context
    Returns tuple of (converted_rate, explanation) or None if conversion not possible
    """
    if not openai.api_key:
        logger.warning("OpenAI API key not set, cannot perform unit conversion")
        return None
    
    try:
        prompt = f"""In the construction industry, I need to convert a rate from one unit to another.
Rate: {rate}
From Unit: {from_unit}
To Unit: {to_unit}

If these units are convertible in construction context, return the converted rate and explanation.
If they are not convertible or if conversion doesn't make sense in construction context, return "NOT_CONVERTIBLE" and explain why.

Return ONLY in this format:
RATE: <converted_rate or NOT_CONVERTIBLE>
EXPLANATION: <brief explanation>"""

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a construction industry expert who understands unit conversions in practical context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.2
        )

        if response and hasattr(response, 'choices') and response.choices and hasattr(response.choices[0], 'message'):
            result = response.choices[0].message.content
            if result is None:
                return None
                
            result = result.strip()
            
            # Parse the response
            rate_line = None
            explanation_line = None
            for line in result.split('\n'):
                if line.startswith('RATE:'):
                    rate_line = line[5:].strip()
                elif line.startswith('EXPLANATION:'):
                    explanation_line = line[12:].strip()
            
            if rate_line and explanation_line:
                if rate_line == "NOT_CONVERTIBLE":
                    return None
                try:
                    converted_rate = float(rate_line)
                    return (converted_rate, explanation_line)
                except ValueError:
                    logger.error(f"OpenAI returned invalid rate format: {rate_line}")
                    return None
        
        return None
        
    except Exception as e:
        logger.error(f"OpenAI unit conversion error: {e}")
        return None

# ============================================================================
# FASTAPI ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize models, dataset, and vector index on startup"""
    initialize_model()
    load_dataset()
    load_cache()
    initialize_vector_index()

@app.on_event("shutdown")
async def shutdown_event():
    """Save cache on shutdown"""
    save_cache()

@app.post("/update-dataset/")
async def update_dataset(file: UploadFile = File(...)):
    """Update the internal dataset with new items from Excel file"""
    global dataset_df
    
    # Validate file
    if not file or not hasattr(file, 'filename'):
        raise HTTPException(status_code=400, detail="Invalid file upload")
        
    filename = str(file.filename)
    if not any(filename.endswith(ext) for ext in ['.xlsx', '.xls']):
        raise HTTPException(status_code=400, detail="File must be Excel format (.xlsx or .xls)")
    
    try:
        # Initialize dataset if needed
        if dataset_df is None:
            dataset_df = load_dataset()
        if not isinstance(dataset_df, pd.DataFrame):
            # Initialize empty DataFrame with proper column types
            dataset_df = pd.DataFrame({
                'description': pd.Series(dtype='object'),
                'unit': pd.Series(dtype='str'),
                'rate': pd.Series(dtype='float64')
            })
        
        # Read uploaded Excel file
        contents = await file.read()
        input_df = pd.read_excel(BytesIO(contents))
        
        # Validate required columns
        required_cols = ['description', 'unit', 'rate']
        missing_cols = [col for col in required_cols if col not in input_df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Excel missing required columns: {missing_cols}")
        
        updates_made = 0
        new_entries = 0
        manual_updates_needed = []  # List to store entries needing manual update
        
        for idx in range(len(input_df)):
            row = input_df.iloc[idx]
            input_desc = str(row['description'])
            
            # Validate unit using pd.isna
            if pd.isna(row['unit']) or str(row['unit']).strip() == '':
                logger.warning(f"Skipping row {idx + 1} with missing unit")
                continue
            
            input_unit = str(row['unit']).strip()
            
            # Skip if description is missing/invalid
            if not input_desc or input_desc == 'nan' or not input_unit:
                logger.warning(f"Skipping row {idx + 1} with missing description or unit")
                continue
            
            try:
                input_rate = float(row['rate'])
            except (ValueError, TypeError):
                logger.warning(f"Skipping row {idx + 1} with invalid rate: {row['rate']}")
                continue
            
            # Apply matching function
            dataset_descriptions = dataset_df['description'].tolist() if len(dataset_df.index) > 0 else []
            matched_idx, matched_desc = function1_match_description_with_openai_enhanced(input_desc, dataset_descriptions)
            
            if matched_idx >= 0:
                # Match found - get dataset values safely
                try:
                    dataset_unit = str(dataset_df.iloc[matched_idx]['unit'])
                    dataset_rate = float(dataset_df.iloc[matched_idx]['rate'])
                    
                    if dataset_unit.lower() == input_unit.lower():
                        # Units match - check rate difference
                        rate_diff_percent = abs(dataset_rate - input_rate) / dataset_rate * 100
                        
                        if rate_diff_percent > 20:  # More than 20% difference
                            manual_updates_needed.append(RateUpdateResponse(
                                description=input_desc,
                                dataset_rate=dataset_rate,
                                input_rate=input_rate,
                                dataset_unit=dataset_unit,
                                input_unit=input_unit,
                                needs_manual_rate=True,
                                reason=f"Rate difference of {rate_diff_percent:.1f}% exceeds 20% threshold"
                            ))
                            continue
                        else:
                            # Update with average rate as difference is acceptable
                            new_rate = (dataset_rate + input_rate) / 2
                            dataset_df.at[matched_idx, 'rate'] = new_rate
                            updates_made += 1
                    else:
                        # Units don't match - needs manual rate entry
                        manual_updates_needed.append(RateUpdateResponse(
                            description=input_desc,
                            dataset_rate=dataset_rate,
                            input_rate=input_rate,
                            dataset_unit=dataset_unit,
                            input_unit=input_unit,
                            needs_manual_rate=True,
                            reason=f"Unit mismatch: {input_unit} vs {dataset_unit}"
                        ))
                        continue
                except (IndexError, ValueError, TypeError) as e:
                    logger.error(f"Error processing matched entry: {e}")
                    continue
            else:
                # No match found - add new entry
                new_entry = pd.DataFrame({
                    'description': [[input_desc]],
                    'unit': [input_unit],
                    'rate': [input_rate]
                })
                dataset_df = pd.concat([dataset_df, new_entry], ignore_index=True)
                new_entries += 1
        
        # Save updated dataset and cache
        save_dataset()
        save_cache()
        
        return {
            "message": "Dataset update processed",
            "updates_made": updates_made,
            "new_entries": new_entries,
            "manual_updates_needed": [update.dict() for update in manual_updates_needed],
            "total_dataset_size": len(dataset_df.index)
        }
        
    except Exception as e:
        logger.error(f"Error updating dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query-rate/")
async def query_rate(file: UploadFile = File(...)):
    """FIXED: Fill missing rates in query Excel file using dataset matches and return Excel file"""
    global dataset_df, match_logs
    
    # Validate file
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if not hasattr(file, 'filename'):
        raise HTTPException(status_code=400, detail="Invalid file - missing filename")
    
    filename = getattr(file, 'filename', None)
    if not isinstance(filename, str) or not filename:
        raise HTTPException(status_code=400, detail="Invalid file - filename must be a non-empty string")
    
    filename_lower = str(filename).lower()
    valid_extensions = ['.xlsx', '.xls']
    is_valid_extension = False
    
    try:
        for ext in valid_extensions:
            if isinstance(ext, str):
                try:
                    if isinstance(filename_lower, str) and filename_lower and len(filename_lower) >= len(ext):
                        if filename_lower[-len(ext):] == ext:
                            is_valid_extension = True
                            break
                except (AttributeError, TypeError, IndexError):
                    continue
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid file - filename must be a string")
    
    if not is_valid_extension:
        raise HTTPException(status_code=400, detail="File must be Excel format (.xlsx or .xls)")
    
    try:
        # Read uploaded Excel file
        contents = await file.read()
        query_df = pd.read_excel(BytesIO(contents))
        
        # Validate required columns
        required_cols = {'description', 'unit'}
        existing_cols = set(query_df.columns)
        missing_cols = required_cols - existing_cols
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Excel must contain columns: {list(required_cols)}")
        
        # Ensure rate column exists and add dataset_unit column
        if 'rate' not in existing_cols:
            query_df['rate'] = np.nan
        if 'dataset_unit' not in existing_cols:
            query_df['dataset_unit'] = ''
        if 'matched_description' not in existing_cols:
            query_df['matched_description'] = ''
        if 'match_score' not in existing_cols:
            query_df['match_score'] = 0.0
        
        # Load current dataset
        if dataset_df is None:
            dataset_df = load_dataset()
        
        if not isinstance(dataset_df, pd.DataFrame):
            raise HTTPException(status_code=500, detail="Internal error - dataset not properly loaded")
        
        if dataset_df.shape[0] == 0:
            raise HTTPException(status_code=400, detail="No dataset available. Please update dataset first.")
        
        rates_filled = 0
        conversions_made = 0
        processing_log = []
        
        # Prepare dataset descriptions for matching
        dataset_descriptions = []
        description_to_index = {}
        
        for idx, row in dataset_df.iterrows():
            desc_list = row['description']
            if isinstance(desc_list, list):
                for desc in desc_list:
                    if desc and str(desc).strip():
                        normalized_desc = normalize_description(desc)
                        dataset_descriptions.append([normalized_desc])
                        description_to_index[normalized_desc] = idx
            else:
                if desc_list and str(desc_list).strip():
                    normalized_desc = normalize_description(str(desc_list))
                    dataset_descriptions.append([normalized_desc])
                    description_to_index[normalized_desc] = idx
        
        logger.info(f"Prepared {len(dataset_descriptions)} descriptions for matching")
        
        # FIXED: Process each row with proper rate conversion
        for idx in range(len(query_df)):
            try:
                row = query_df.iloc[idx]
                query_desc = str(row['description'])
                query_unit = str(row['unit'])
                
                # Skip if rate already exists and is not NaN
                existing_rate = row.get('rate', np.nan)
                if isinstance(existing_rate, (int, float)) and not pd.isna(existing_rate):
                    processing_log.append(f"Row {idx + 1}: Rate already exists, skipping")
                    continue
                
                # Apply matching function
                matched_idx, matched_desc = function1_match_description_with_openai_enhanced(
                    query_desc, 
                    dataset_descriptions,
                    query_unit
                )
                
                if matched_idx >= 0:
                    # Get the original dataset index
                    normalized_matched = normalize_description(matched_desc)
                    original_idx = description_to_index.get(normalized_matched)
                    
                    if original_idx is not None:
                        # Match found - get dataset values from original index
                        dataset_row = dataset_df.iloc[original_idx]
                        dataset_unit = str(dataset_row['unit'])
                        dataset_rate = dataset_row['rate']
                        
                        # FIXED: Skip if dataset rate is NaN or empty
                        if pd.isna(dataset_rate) or dataset_rate == '' or dataset_rate == 0:
                            processing_log.append(f"Row {idx + 1}: Dataset rate is empty/zero, skipping")
                            continue
                        
                        dataset_rate = float(dataset_rate)
                        
                        # FIXED: Always fill dataset_unit and matched_description
                        query_df.at[idx, 'dataset_unit'] = dataset_unit
                        query_df.at[idx, 'matched_description'] = matched_desc
                        
                        # FIXED: Check unit compatibility and convert rates
                        query_unit_norm = normalize_unit_name(query_unit)
                        dataset_unit_norm = normalize_unit_name(dataset_unit)
                        
                        logger.info(f"Row {idx + 1}: Comparing units - Query: {query_unit} ({query_unit_norm}) vs Dataset: {dataset_unit} ({dataset_unit_norm})")
                        
                        if query_unit_norm == dataset_unit_norm:
                            # Units match exactly - direct assignment
                            query_df.at[idx, 'rate'] = dataset_rate
                            processing_log.append(f"Row {idx + 1}: Direct match, rate={dataset_rate}")
                        else:
                            # Units don't match - try conversion
                            converted_rate = function2_convert_rate_by_unit(dataset_rate, dataset_unit, query_unit)
                            
                            if converted_rate is not None:
                                # Conversion successful
                                query_df.at[idx, 'rate'] = converted_rate
                                processing_log.append(f"Row {idx + 1}: Rate converted from {dataset_unit} to {query_unit}, original={dataset_rate}, converted={converted_rate}")
                                conversions_made += 1
                            else:
                                # Conversion failed - keep original rate but note the issue
                                query_df.at[idx, 'rate'] = dataset_rate
                                processing_log.append(f"Row {idx + 1}: Unit conversion failed from {dataset_unit} to {query_unit}, kept original rate={dataset_rate}")
                        
                        rates_filled += 1
                        
                        # Create match log entry
                        match_log = {
                            "timestamp": datetime.now(),
                            "query_description": query_desc,
                            "matched_description": matched_desc,
                            "query_unit": query_unit,
                            "matched_unit": dataset_unit,
                            "rate": dataset_rate,
                            "converted_rate": query_df.at[idx, 'rate'],
                            "status": "MATCHED",
                            "unit_mismatch": query_unit_norm != dataset_unit_norm
                        }
                        match_logs.append(match_log)
                    else:
                        logger.error(f"Row {idx + 1}: Could not find original index for matched description")
                        processing_log.append(f"Row {idx + 1}: Error - Could not find original index")
                else:
                    processing_log.append(f"Row {idx + 1}: No matching description found")
                    match_logs.append({
                        "timestamp": datetime.now(),
                        "query_description": query_desc,
                        "matched_description": "No match found",
                        "query_unit": query_unit,
                        "matched_unit": "",
                        "rate": 0.0,
                        "status": "NO_MATCH",
                        "unit_mismatch": False
                    })
            except Exception as e:
                logger.error(f"Error processing row {idx + 1}: {e}")
                processing_log.append(f"Row {idx + 1}: Error - {str(e)}")
                continue
        
        # Log processing summary
        logger.info(f"Query processing complete: {rates_filled} rates filled, {conversions_made} conversions made out of {len(query_df)} rows")
        for log_entry in processing_log:
            logger.info(log_entry)
        
        # Create temporary file for Excel output
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
        temp_file_path = temp_file.name
        temp_file.close()
        
        # Write Excel file to temporary location
        with pd.ExcelWriter(temp_file_path, engine='openpyxl') as writer:
            query_df.to_excel(writer, sheet_name='Filled_Rates', index=False)
            
            # Auto-adjust column widths
            workbook = writer.book
            worksheet = writer.sheets['Filled_Rates']
            
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        # Create filename safely
        original_name = os.path.splitext(filename)[0] if filename else "output"
        output_filename = f"{original_name}_filled_rates.xlsx"
        
        # Define cleanup function
        async def cleanup():
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")
        
        # Fix the background task issue
        from starlette.background import BackgroundTask
        
        response = FileResponse(
            path=temp_file_path,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=output_filename,
            background=BackgroundTask(cleanup),
            headers={
                "Content-Disposition": f"attachment; filename={output_filename}",
                "X-Rates-Filled": str(rates_filled),
                "X-Total-Rows": str(len(query_df)),
                "X-Conversions-Made": str(conversions_made),
                "X-Daily-Tokens-Used": str(quota_tracker["daily_usage"]),
                "X-Token-Limit": str(MAX_TOKENS_PER_DAY),
                "X-Embedding-Threshold": str(EMBEDDING_SIMILARITY_THRESHOLD)
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/dataset-info/")
async def get_dataset_info():
    """Get information about the dataset"""
    try:
        if dataset_df is None:
            load_dataset()
            
        total_entries = 0
        total_descriptions = 0
        avg_descriptions = 0
        
        if dataset_df is not None and not dataset_df.empty:
            total_entries = len(dataset_df.index)
            
            # Count total descriptions safely
            desc_series = dataset_df['description']
            total_descriptions = 0
            
            for desc in desc_series:
                if isinstance(desc, list):
                    total_descriptions += len(desc)
                elif pd.notna(desc):  # Handle single string descriptions
                    total_descriptions += 1
            
            avg_descriptions = round(total_descriptions / total_entries, 2) if total_entries > 0 else 0
            
        return {
            "total_entries": total_entries,
            "total_descriptions": total_descriptions,
            "average_descriptions_per_entry": avg_descriptions
        }
    except Exception as e:
        logger.error(f"Error getting dataset info: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/token-usage/")
async def get_token_usage():
    """Get information about token usage"""
    try:
        return {
            "daily_tokens_used": quota_tracker.get("daily_usage", 0),
            "daily_token_limit": MAX_TOKENS_PER_DAY,
            "cached_matches": len(matching_cache) if matching_cache else 0
        }
    except Exception as e:
        logger.error(f"Error getting token usage: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/clear-cache/")
async def clear_cache():
    """Clear the matching cache"""
    global matching_cache
    matching_cache = {}
    save_cache()
    return {"message": "Cache cleared successfully"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Fixed Rate Matching API",
        "version": "2.0.0",
        "fixes": [
            "Fixed rate conversion logic (350 Sqmtr → 32.53 sft)",
            "Added Sqmtr ↔ sft unit recognition",
            "Proper dataset description parsing",
            "Enhanced unit normalization",
            "Skip empty dataset rates"
        ],
        "configuration": {
            "embedding_similarity_threshold": EMBEDDING_SIMILARITY_THRESHOLD,
            "max_tokens_per_day": MAX_TOKENS_PER_DAY,
            "openai_usage": "Used for most matches (similarity < 0.9)"
        },
        "endpoints": {
            "/update-dataset/": "POST - Upload Excel to update internal dataset",
            "/query-rate/": "POST - Upload Excel to fill missing rates",
            "/dataset-info/": "GET - View current dataset information",
            "/token-usage/": "GET - View token usage statistics",
            "/clear-cache/": "POST - Clear matching cache"
        }
    }

@app.get("/dataset-entries/")
async def get_dataset_entries():
    """Get all entries from the dataset"""
    global dataset_df
    
    if dataset_df is None:
        dataset_df = load_dataset()
    
    entries = []
    if isinstance(dataset_df, pd.DataFrame) and len(dataset_df.index) > 0:
        for idx in dataset_df.index:
            try:
                # Get values safely using .at accessor
                description = dataset_df.at[idx, 'description']
                unit = dataset_df.at[idx, 'unit']
                rate = dataset_df.at[idx, 'rate']
                
                # Convert values safely using pandas isna
                is_unit_valid = isinstance(unit, (str, int, float)) and not pd.isna(unit)
                is_rate_valid = isinstance(rate, (int, float)) and not pd.isna(rate)
                
                # Handle description as string
                if pd.notna(description):
                    if isinstance(description, list):
                        desc_str = str(description[0]) if description else ""
                    else:
                        desc_str = str(description).strip()
                else:
                    desc_str = ""
                
                # Only add entry if we have valid data
                if desc_str and is_unit_valid:
                    entries.append({
                        "description": desc_str,
                        "unit": str(unit) if is_unit_valid else "",
                        "rate": float(rate) if is_rate_valid else None
                    })
            except Exception as e:
                logger.warning(f"Error processing dataset entry: {e}")
                continue
    
    return {"entries": entries}

@app.get("/match-logs/")
async def get_match_logs():
    """Get all match logs"""
    global match_logs
    # Filter out unmatched entries and return only successful matches
    filtered_logs = [
        log for log in match_logs 
        if log["status"] in ["MATCHED", "CONVERTED"] and 
        log["matched_description"] != "No match found"
    ]
    return filtered_logs

@app.post("/select-unit/")
async def select_unit(data: dict):
    """Handle unit selection for rate conversion"""
    try:
        # Try OpenAI conversion first
        conversion_result = function2_convert_rate_by_unit_with_openai(
            data['rate'],
            data['matched_unit'],
            data['selected_unit']
        )
        
        if conversion_result:
            converted_rate, explanation = conversion_result
            # Update the match log status
            for log in match_logs:
                if (log['query_description'] == data['query_description'] and
                    log['matched_description'] == data['matched_description']):
                    log['status'] = 'CONVERTED'
                    log['rate'] = converted_rate
                    log['conversion_explanation'] = explanation
                    break
            
            return {
                "success": True,
                "converted_rate": converted_rate,
                "explanation": explanation
            }
        
        return {
            "success": False,
            "detail": "Could not convert between units"
        }
    except Exception as e:
        logger.error(f"Error in unit selection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def get_construction_synonyms(description: str) -> List[str]:
    """Get construction industry synonyms for a description using GPT-3"""
    try:
        response = openai.chat.completions.create(
            model=DEFAULT_GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are a construction industry expert. Generate synonyms and alternative descriptions for construction items."},
                {"role": "user", "content": f"Generate 3-5 alternative descriptions for this construction item, one per line: {description}"}
            ],
            temperature=0.7,
            max_tokens=150
        )
        
        if response and response.choices and response.choices[0].message and response.choices[0].message.content:
            synonyms = [
                syn.strip() for syn in response.choices[0].message.content.split('\n')
                if syn.strip() and syn.strip() != description
            ]
            return synonyms
        else:
            logger.warning(f"Invalid response format from OpenAI for description: {description}")
            return []
            
    except Exception as e:
        logger.error(f"Error getting synonyms for {description}: {e}")
        return []

def analyze_descriptions_for_terms(descriptions: List[str]) -> Set[str]:
    """
    Analyze descriptions using TF-IDF and construction domain rules to identify important terms.
    Returns a set of important terms that should be preserved during text cleaning.
    """
    try:
        # Initialize TF-IDF vectorizer with custom parameters
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),  # Consider both unigrams and bigrams
            max_features=1000,
            stop_words='english',
            min_df=2,  # Term must appear in at least 2 documents
            max_df=0.9  # Ignore terms that appear in more than 90% of documents
        )
        
        # Fit TF-IDF
        tfidf_matrix = vectorizer.fit_transform(descriptions)
        feature_names = vectorizer.get_feature_names_out()
        
        # Calculate mean TF-IDF scores for each term using sparse operations
        mean_tfidf = np.asarray(sparse.csr_matrix.mean(tfidf_matrix, axis=0)).flatten()
        
        # Get terms with high TF-IDF scores
        important_terms = set()
        for term_idx in np.argsort(mean_tfidf)[-100:]:  # Top 100 terms
            term = feature_names[term_idx]
            if mean_tfidf[term_idx] > 0.1:  # Minimum threshold
                important_terms.add(term)
        
        # Add terms that match construction-specific patterns
        for desc in descriptions:
            # Find measurement patterns (e.g., "100 mm", "50 sqm")
            measurement_matches = re.finditer(r'\d+\s*([a-zA-Z]+)', desc.lower())
            for match in measurement_matches:
                unit = match.group(1)
                if len(unit) <= 5:  # Reasonable unit length
                    important_terms.add(unit)
            
            # Find material specifications
            material_matches = re.finditer(r'(grade|type|class|mark)\s+([a-zA-Z0-9-]+)', desc.lower())
            for match in material_matches:
                spec = match.group(2)
                important_terms.add(spec)
        
        # Add common construction terms that co-occur frequently
        doc_terms = [set(desc.lower().split()) for desc in descriptions]
        term_pairs = Counter()
        
        for terms in doc_terms:
            pairs = list(itertools.combinations(terms, 2))
            term_pairs.update(pairs)
        
        # Add frequently co-occurring terms
        for (term1, term2), count in term_pairs.most_common(50):
            if count >= len(descriptions) * 0.1:  # Appears in at least 10% of descriptions
                important_terms.add(term1)
                important_terms.add(term2)
        
        # Filter out very short terms and common English words
        stop_words = set(stopwords.words('english'))
        important_terms = {
            term for term in important_terms 
            if len(term) > 2 
            and term not in stop_words
            and not term.isdigit()
        }
        
        logger.info(f"Extracted {len(important_terms)} important terms from descriptions")
        return important_terms
        
    except Exception as e:
        logger.error(f"Error analyzing descriptions for terms: {e}")
        return set()

def refresh_dataset_descriptions() -> int:
    """Analyze descriptions to identify and save important terms for text cleaning"""
    try:
        global dataset_df
        if dataset_df is None:
            dataset_df = pd.DataFrame()
            
        if dataset_df.empty:
            logger.warning("Dataset is empty")
            return 0
            
        # Collect all descriptions for analysis
        all_descriptions = []
        for idx in range(len(dataset_df)):
            # Get current description
            current_description = str(dataset_df.iloc[idx]['description']).strip()
            if not current_description:
                continue
            all_descriptions.append(current_description)
            
        # Analyze descriptions to get important terms
        extracted_terms = analyze_descriptions_for_terms(all_descriptions)
        
        # Update the important terms in clean_text function
        important_construction_terms = {
            'with', 'without', 'including', 'per', 'in', 'at', 'to', 'from',
            'mm', 'cm', 'm', 'sqm', 'cum', 'kg', 'ton', 'ml', 'l',
            'thick', 'width', 'depth', 'height', 'dia', 'gauge'
        }
        important_construction_terms.update(extracted_terms)
        
        # Store the updated terms in a file for persistence
        terms_file = "important_terms.json"
        with open(terms_file, 'w') as f:
            json.dump(list(important_construction_terms), f, indent=2)
        
        logger.info(f"Updated important terms list with {len(extracted_terms)} new terms")
        return len(extracted_terms)
        
    except Exception as e:
        logger.error(f"Error refreshing descriptions: {e}")
        return 0

@app.post("/refresh-descriptions/")
async def refresh_descriptions():
    """Endpoint to refresh all descriptions with GPT-generated synonyms"""
    try:
        refresh_dataset_descriptions()
        if isinstance(dataset_df, pd.DataFrame):
            return {"message": "Successfully refreshed descriptions", "total_entries": len(dataset_df.index)}
        else:
            return {"message": "No dataset available", "total_entries": 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def initialize_vector_index():
    """Initialize FAISS vector index for fast similarity search"""
    global vector_index, vector_descriptions, last_index_update, dataset_df
    try:
        if dataset_df is None or dataset_df.empty:
            return
            
        # Get all descriptions
        all_descs = []
        for desc_list in dataset_df['description']:
            if isinstance(desc_list, list):
                all_descs.extend(desc_list)
            else:
                all_descs.append(str(desc_list))
        
        # Normalize descriptions
        normalized_descs = [normalize_description(desc) for desc in all_descs]
        vector_descriptions = normalized_descs
        
        # Get embeddings
        embeddings = get_embeddings(normalized_descs)
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        vector_index = faiss.IndexFlatL2(dimension)
        if vector_index is not None:
            vector_index.add(embeddings_array)  # type: ignore
            last_index_update = datetime.now()
            logger.info(f"Initialized vector index with {len(normalized_descs)} descriptions")
        
    except Exception as e:
        logger.error(f"Error initializing vector index: {e}")
        vector_index = None

def get_similar_vectors(query_embedding: np.ndarray, k: int = MAX_OPENAI_CANDIDATES) -> List[int]:
    """Get indices of k most similar vectors using FAISS"""
    global vector_index
    try:
        if vector_index is None:
            initialize_vector_index()
        if vector_index is None:
            return []
            
        # Reshape query embedding for FAISS
        query_array = query_embedding.reshape(1, -1).astype('float32')
        
        # Search index
        D, I = vector_index.search(x=query_array, k=k)  # type: ignore
        return I[0].tolist() if I is not None else []
        
    except Exception as e:
        logger.error(f"Error in vector search: {e}")
        return []

def validate_and_convert_match(match_result: MatchResult, query_unit: str) -> MatchResult:
    """Validate units and update match result"""
    try:
        # Check unit compatibility
        is_compatible, message = get_unit_compatibility(match_result.unit, query_unit)
        match_result.unit_compatible = is_compatible
        match_result.unit_message = message
        
        # Log the validation
        logger.info(f"Unit validation: {query_unit} vs {match_result.unit} - {message}")
        
        return match_result
    except Exception as e:
        logger.error(f"Error in match validation: {e}")
        return match_result

# Test function to verify the fixes work with actual data
def test_conversion_with_actual_data():
    """Test the fixed conversion logic with actual data from your dataset"""
    print("Testing Rate Conversions with Actual Dataset:")
    print("=" * 60)
    
    # Test cases based on your actual dataset and expected output
    test_cases = [
        # (rate, from_unit, to_unit, expected_result)
        (350, 'Sqmtr', 'sft', 32.53),   # From your dataset
        (900, 'Sqmtr', 'sft', 83.64),   # From your dataset
        (320, 'Sqmtr', 'sft', 29.74),   # From your dataset
        (250, 'Sqmtr', 'sft', 23.24),   # From your dataset
        (1300, 'Sqmtr', 'sft', 120.82), # From your dataset
        # Reverse conversions
        (32.53, 'sft', 'Sqmtr', 350),
        (83.64, 'sft', 'Sqmtr', 900),
    ]
    
    for rate, from_unit, to_unit, expected in test_cases:
        result = function2_convert_rate_by_unit(rate, from_unit, to_unit)
        if result is not None:
            print(f"✅ {rate} {from_unit} → {result:.2f} {to_unit}")
            print(f"   Expected: {expected:.2f}, Got: {result:.2f}")
            print(f"   Match: {'✅' if abs(result - expected) < 1 else '❌'}")
        else:
            print(f"❌ Failed: {rate} {from_unit} → {to_unit}")
        print()

# ============================================================================
# RUN THE APPLICATION
# ============================================================================

if __name__ == "__main__":
    # Test the conversion function
    test_conversion_with_actual_data()
    
    # Run the FastAPI application
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)