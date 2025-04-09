import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Language codes (as specified in Spec 9.5.2)
SUPPORTED_LANGUAGES = [
    "frFR",  # French
    "deDE",  # German
    "ptBR",  # Brazilian Portuguese
    "esLA",  # Latin American Spanish
    "jaJP",  # Japanese
    "zhCN",  # Chinese
    "plPL",  # Polish
    "ukUA",  # Ukrainian
    "ruRU",  # Russian
    "trTR"   # Turkish
]

# File paths
RULESET_DIR = os.environ.get("RULESET_DIR", "./rulesets")
# Spec uses ./rulesets/global_ruleset.md, adjusting based on dir structure in Spec 10.1
GLOBAL_RULESET_PATH = os.environ.get("GLOBAL_RULESET_PATH", os.path.join(RULESET_DIR, "smite2/global_ruleset.md"))
TRANSLATION_MEMORY_DB = os.environ.get("TRANSLATION_MEMORY_DB", "./data/translation_memory.db")
ERROR_LOG_FILE = os.environ.get("ERROR_LOG_FILE", "./logs/translation_errors.log")

# Translation settings
DEFAULT_CONFIDENCE_THRESHOLD = float(os.environ.get("DEFAULT_CONFIDENCE_THRESHOLD", "0.9"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.environ.get("RETRY_DELAY", "1.0"))  # seconds

# Quality assessment settings
QUALITY_THRESHOLD = float(os.environ.get("QUALITY_THRESHOLD", "80.0"))  # minimum acceptable quality score
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "3"))  # maximum number of improvement iterations

# System settings
PARALLEL_WORKERS = int(os.environ.get("PARALLEL_WORKERS", "5"))  # number of parallel workers for batch processing

# API settings
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "gpt-4o")
FALLBACK_MODEL = os.environ.get("FALLBACK_MODEL", "gpt-3.5-turbo")

# Ensure base directories exist (optional, can be handled by specific components)
DATA_DIR = os.path.dirname(TRANSLATION_MEMORY_DB)
LOG_DIR = os.path.dirname(ERROR_LOG_FILE)

# Example check, could be expanded or moved to main application logic
# if DATA_DIR:
#     os.makedirs(DATA_DIR, exist_ok=True)
# if LOG_DIR:
#     os.makedirs(LOG_DIR, exist_ok=True) 