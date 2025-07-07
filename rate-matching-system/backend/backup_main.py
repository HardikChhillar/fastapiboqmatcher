# main.py - Fixed FastAPI Rate Matching Application
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

# FIXED CONFIGURATION - Removed overly aggressive limits
EMBEDDING_SIMILARITY_THRESHOLD = 1  # Balanced threshold
MAX_OPENAI_CANDIDATES = 8  # Reasonable number
CACHE_EXPIRY_DAYS = 30
MAX_TOKENS_PER_DAY = 5000000
  # More reasonable daily limit
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
    """Load the internal dataset from CSV or create empty one"""
    global dataset_df
    dataset_path = "dataset.csv"  # Changed to be relative to current directory
    
    if os.path.exists(dataset_path):
        try:
            dataset_df = pd.read_csv(dataset_path)
            # Convert string representation of lists back to actual lists
            if 'description' in dataset_df.columns:
                dataset_df['description'] = dataset_df['description'].apply(
                    lambda x: [str(item).strip() for item in eval(x)] if isinstance(x, str) and x.startswith('[') 
                    else [str(x).strip()]
                )
            logger.info(f"Loaded dataset with {len(dataset_df)} entries")
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
            save_df.to_csv("dataset.csv", index=False)  # Changed to be relative to current directory
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

def clean_text(text: str) -> str:
    """Clean and tokenize text while preserving important terms"""
    try:
        if not isinstance(text, str) or not text.strip():
            return ""
            
        # Convert to lowercase and normalize whitespace
        text = ' '.join(text.lower().split())
        
        # Load important terms
        important_terms = load_important_terms()
        if not important_terms:
            important_terms = {
                'with', 'without', 'including', 'per', 'in', 'at', 'to', 'from',
                'mm', 'cm', 'm', 'sqm', 'cum', 'kg', 'ton', 'ml', 'l',
                'thick', 'width', 'depth', 'height', 'dia', 'gauge'
            }
        
        # First, protect numbers with units
        protected_terms = []
        number_unit_pattern = r'\b(\d+(?:\.\d+)?)\s*(mm|cm|m|sqm|cum|kg|ton|ml|l)\b'
        text = re.sub(number_unit_pattern, 
                     lambda m: f"__NUM_{len(protected_terms)}__" if protected_terms.append(m.group(0)) else "",
                     text)
        
        # Tokenize using NLTK
        words = word_tokenize(text)
        
        # Process words
        cleaned_words = []
        i = 0
        while i < len(words):
            # Check for protected terms
            current = words[i]
            if current.startswith('__NUM_'):
                idx = int(current.split('_')[2].rstrip('_'))
                if idx < len(protected_terms):
                    cleaned_words.append(protected_terms[idx])
                i += 1
                continue
            
            # Try multi-word terms
            matched = False
            for length in range(3, 0, -1):
                if i + length <= len(words):
                    phrase = ' '.join(words[i:i+length])
                    if phrase in important_terms:
                        cleaned_words.append(phrase)
                        i += length
                        matched = True
                        break
            
            if not matched:
                word = words[i]
                # Keep numbers, important terms, and meaningful words
                if (word in important_terms or 
                    any(c.isdigit() for c in word) or 
                    (len(word) > 1 and word.replace('.', '').replace('-', '').isalnum())):
                    cleaned_words.append(word)
                i += 1
        
        # Join words and restore protected terms
        cleaned_text = ' '.join(cleaned_words)
        
        logger.info(f"Cleaned text tokens: {cleaned_words}")
        return cleaned_text
        
    except Exception as e:
        logger.error(f"Error in clean_text: {e}")
        return text if isinstance(text, str) else ""

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
        prompt = f"""Match the query activity to the best candidate.

QUERY: "{input_description}"

CANDIDATES:
{candidates_text}

Return ONLY the exact matched candidate text or "no match" if none are similar."""
        
        # FIXED: Proper retry logic with exponential backoff
        max_retries = 3
        base_wait_time = 1
        
        for attempt in range(max_retries):
            try:
                response = openai.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert activity matcher. Return only the exact matched text or 'no match'."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=100,
                    temperature=0.1,
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


def function1_match_description_with_openai_enhanced(input_description: str, dataset_descriptions: List[List[str]]) -> tuple:
    """
    Enhanced matching with numerical context preservation and mandatory OpenAI validation
    """
    if not dataset_descriptions:
        logger.info("No dataset descriptions provided")
        return -1, None
    
    try:
        # Clean input description
        clean_input = clean_text(input_description)
        logger.info(f"Cleaned input: {clean_input}")
        
        # Prepare all candidate descriptions
        all_candidates = []
        candidate_to_index = []
        
        for idx, desc_list in enumerate(dataset_descriptions):
            for desc in desc_list:
                clean_desc = clean_text(desc)
                all_candidates.append(clean_desc)
                candidate_to_index.append(idx)
                logger.info(f"Candidate {len(all_candidates)}: {clean_desc} (Index: {idx})")
        
        if not all_candidates:
            return -1, None
        
        # Get embeddings for similarity comparison
        all_texts = [clean_input] + all_candidates
        embeddings = get_embeddings(all_texts)
        
        input_embedding = embeddings[0:1]
        candidate_embeddings = embeddings[1:]
        
        similarities = cosine_similarity(input_embedding, candidate_embeddings)[0]
        best_similarity_idx = np.argmax(similarities)
        best_similarity = similarities[best_similarity_idx]
        
        logger.info(f"Best similarity score: {best_similarity:.3f}")
        
        # Perfect or near-perfect match (similarity >= 0.95)
        if best_similarity >= 0.95:
            matched_index = candidate_to_index[best_similarity_idx]
            matched_desc = all_candidates[best_similarity_idx]
            logger.info("Very high similarity match found")
            return matched_index, matched_desc
        
        # High similarity (>= 0.40) - validate with OpenAI
        if best_similarity >= 0.40:
            logger.info("Moderate to high similarity found, validating with OpenAI")
            # Get top candidates for OpenAI validation
            top_indices = np.argsort(similarities)[-MAX_OPENAI_CANDIDATES:][::-1]
            top_candidates = [all_candidates[i] for i in top_indices]
            
            # Try OpenAI matching
            gpt_result, input_tokens, output_tokens = call_openai_for_matching(
                input_description, top_candidates
            )
            
            if gpt_result != "no match":
                # Find which candidate matched
                for i, candidate in enumerate(top_candidates):
                    if gpt_result == candidate or any(gpt_result in desc for desc in dataset_descriptions[candidate_to_index[top_indices[i]]]):
                        matched_index = candidate_to_index[top_indices[i]]
                        logger.info(f"OpenAI validated match: {gpt_result}")
                        return matched_index, gpt_result
            
            # If OpenAI didn't validate but similarity is still high
            if best_similarity >= 0.80:
                matched_index = candidate_to_index[best_similarity_idx]
                matched_desc = all_candidates[best_similarity_idx]
                logger.info(f"Using high similarity match despite OpenAI rejection (similarity: {best_similarity:.3f})")
                return matched_index, matched_desc
        
        # No good match found
        logger.info(f"No good match found (best similarity: {best_similarity:.3f})")
        return -1, None
        
    except Exception as e:
        logger.error(f"Error in matching function: {e}")
        return -1, None

def function2_convert_rate_by_unit(rate: float, from_unit: str, to_unit: str) -> Optional[float]:
    """
    Convert rate from one unit to another using pint with comprehensive unit aliases
    """
    try:
        # Standard unit aliases - using most common civil industry terms
        unit_aliases = {
            # Area units - Square feet (sqft is most common in civil industry)
            'sft.': 'sqft',
            'sft': 'sqft',
            'sq.ft': 'sqft',
            'sq. ft.': 'sqft',
            'sq. ft..': 'sqft',
            'sq.ft': 'sqft',
            'sqft ': 'sqft',
            
            # Area units - Square meters (sqm is standard)
            'sq.m': 'sqm',
            'sqmt': 'sqm',
            
            # Linear units - Running feet (rft is standard)
            'rft.': 'rft',
            'running feet': 'rft',
            'rf': 'rft',
            
            # Linear units - Running meters (rmt is standard)
            'running meter': 'rmt',
            'rm': 'rmt',
            'metre': 'rmt',
            'mtr': 'rmt',
            'mtr.': 'rmt',
            'm': 'rmt',
            
            # Volume units - Cubic meters (cum is standard)
            'cu m': 'cum',
            'cuft': 'cum',  # assuming conversion to standard
            'cu ft': 'cum',
            
            # Weight/Mass units (kg is standard)
            # kg stays as kg
            
            # Volume units - Liters (ltr is standard)
            # ltr stays as ltr
            
            # Count/Number units - Numbers (nos is most common in civil industry)
            'nos.': 'nos',
            'no.': 'nos',
            'no.s': 'nos',
            'no\'s': 'nos',
            'no': 'nos',
            'num': 'nos',
            'each': 'nos',
            
            # Lump sum units (ls is standard)
            'l.s': 'ls',
            'lot': 'ls',
            
            # Work/Service units - keeping as individual standards
            # man-days, trips, job, set, packet, pack stay as themselves
        }
        
        # Standard unit to pint unit mapping
        standard_to_pint = {
            'sqft': 'foot**2',
            'sqm': 'meter**2',
            'rft': 'foot',
            'rmt': 'meter',
            'cum': 'meter**3',
            'kg': 'kilogram',
            'ltr': 'liter',
            'nos': 'dimensionless',
            'ls': 'dimensionless',
            'man-days': 'dimensionless',
            'trips': 'dimensionless',
            'job': 'dimensionless',
            'set': 'dimensionless',
            'packet': 'dimensionless',
            'pack': 'dimensionless',
        }
        
        # Normalize unit names (convert to lowercase and strip whitespace)
        from_unit_clean = from_unit.lower().strip()
        to_unit_clean = to_unit.lower().strip()
        
        # First normalize to standard civil industry units
        from_unit_standard = unit_aliases.get(from_unit_clean, from_unit_clean)
        to_unit_standard = unit_aliases.get(to_unit_clean, to_unit_clean)
        
        # Then convert to pint units
        from_unit_pint = standard_to_pint.get(from_unit_standard, from_unit_standard)
        to_unit_pint = standard_to_pint.get(to_unit_standard, to_unit_standard)
        
        # Handle dimensionless units specially
        if from_unit_pint == 'dimensionless' and to_unit_pint == 'dimensionless':
            return float(rate)  # No conversion needed for dimensionless to dimensionless
        
        # If one unit is dimensionless and the other isn't, return None (incompatible)
        if (from_unit_pint == 'dimensionless') != (to_unit_pint == 'dimensionless'):
            logger.warning(f"Cannot convert between dimensionless and dimensional units: {from_unit} to {to_unit}")
            return None
        
        # Create quantities and convert
        quantity = ureg.Quantity(rate, from_unit_pint)
        converted = quantity.to(to_unit_pint)
        
        return float(converted.magnitude)
        
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
            temperature=0.1
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
    """Initialize models and load dataset on startup"""
    initialize_model()
    load_dataset()
    load_cache()

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
    """Fill missing rates in query Excel file using dataset matches and return Excel file"""
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
        
        # Validate required columns using sets for efficient membership testing
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
        
        # Load current dataset
        if dataset_df is None:
            dataset_df = load_dataset()
        
        if not isinstance(dataset_df, pd.DataFrame):
            raise HTTPException(status_code=500, detail="Internal error - dataset not properly loaded")
        
        if dataset_df.shape[0] == 0:
            raise HTTPException(status_code=400, detail="No dataset available. Please update dataset first.")
        
        rates_filled = 0
        processing_log = []
        
        # Convert index to list of integers for safe iteration
        indices = [int(x) for x in query_df.index]
        
        for idx in indices:
            row = query_df.iloc[idx]
            query_desc = str(row['description'])
            query_unit = str(row['unit'])
            
            # Skip if rate already exists and is not NaN
            existing_rate = row.get('rate', np.nan)
            if isinstance(existing_rate, (int, float)) and not np.isnan(existing_rate):
                processing_log.append(f"Row {idx + 1}: Rate already exists, skipping")
                continue
            
            # Apply matching function
            dataset_descriptions = dataset_df['description'].tolist()
            matched_idx, matched_desc = function1_match_description_with_openai_enhanced(query_desc, dataset_descriptions)
            
            # Create match log entry regardless of match status
            match_log = {
                "timestamp": datetime.now(),
                "query_description": query_desc,
                "matched_description": matched_desc if matched_idx >= 0 else "No match found",
                "query_unit": query_unit,
                "matched_unit": dataset_df.iloc[matched_idx]['unit'] if matched_idx >= 0 else "",
                "rate": float(dataset_df.iloc[matched_idx]['rate']) if matched_idx >= 0 else 0.0,
                "status": "MATCHED" if matched_idx >= 0 else "NO_MATCH",
                "unit_mismatch": False if matched_idx < 0 else query_unit.lower() != dataset_df.iloc[matched_idx]['unit'].lower()
            }
            
            # Add to match logs without limit
            match_logs.append(match_log)
            
            if matched_idx >= 0:
                # Match found
                dataset_unit = dataset_df.iloc[matched_idx]['unit']
                dataset_rate = dataset_df.iloc[matched_idx]['rate']
                
                # Convert to string safely
                query_unit_str = str(query_unit) if isinstance(query_unit, (str, int, float)) and not isinstance(query_unit, bool) and not pd.isna(query_unit) else ""
                dataset_unit_str = str(dataset_unit) if isinstance(dataset_unit, (str, int, float)) and not isinstance(dataset_unit, bool) and not pd.isna(dataset_unit) else ""
                
                if not query_unit_str or not dataset_unit_str or query_unit_str == 'nan' or dataset_unit_str == 'nan':
                    processing_log.append(f"Row {idx + 1}: Match found but unit information is missing")
                    continue
                
                if query_unit_str.lower() == dataset_unit_str.lower():
                    # Units match - use rate directly
                    query_df.iloc[idx, query_df.columns.get_loc('rate')] = dataset_rate
                    processing_log.append(f"Row {idx + 1}: Direct match found, rate={dataset_rate}")
                else:
                    # Units don't match - try OpenAI conversion
                    conversion_result = function2_convert_rate_by_unit_with_openai(dataset_rate, dataset_unit_str, query_unit_str)
                    if conversion_result:
                        converted_rate, explanation = conversion_result
                        query_df.iloc[idx, query_df.columns.get_loc('rate')] = converted_rate
                        processing_log.append(f"Row {idx + 1}: Match found with unit conversion, rate={converted_rate:.4f} ({explanation})")
                    else:
                        # Units not convertible - store original rate and dataset unit
                        query_df.iloc[idx, query_df.columns.get_loc('rate')] = dataset_rate
                        query_df.iloc[idx, query_df.columns.get_loc('dataset_unit')] = dataset_unit_str
                        processing_log.append(f"Row {idx + 1}: Match found but units not convertible. Stored original rate={dataset_rate} with unit={dataset_unit_str}")
                
                rates_filled += 1
            else:
                processing_log.append(f"Row {idx + 1}: No matching description found")
        
        # Save cache after processing
        save_cache()
        
        # Log processing summary
        logger.info(f"Query processing complete: {rates_filled} rates filled out of {len(query_df)} rows")
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
        "message": "Optimized Rate Matching API",
        "version": "1.0.0",
        "optimizations": [
            "GPT-3.5-turbo for cost efficiency",
            "Intelligent caching system",
            "Embedding-based filtering",
            "Daily token limits",
            "Reduced candidate pool"
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

# ============================================================================
# RUN THE APPLICATION
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)