import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DOC_DIR = Path(os.getenv("DOC_DIR", "./dataDocs"))
DB_PATH = os.getenv("DB_PATH", "./vectordb")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:4141/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "dummy")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5-mini")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))
TOP_K = int(os.getenv("TOP_K", "3"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))

ANALYTICAL_KEYWORDS = {
    "top", "highest", "lowest", "average", "avg",
    "total", "sum", "count", "rank", "minimum",
    "maximum", "max", "min", "trend", "compare",
    "best", "worst", "most", "least", "revenue",
    "sales", "quantity", "orders", "product"
}

INJECTION_PATTERNS = [
    r"ignore (all |previous |above )?instructions",
    r"you are now",
    r"forget (everything|your instructions)",
    r"system\s*:",
    r"<\s*/?system\s*>",
    r"disregard",
]

AMBIGUOUS_PATTERNS = [
    (r"\bthe report\b",              "Which report are you referring to?"),
    (r"\bthe process\b",             "Which process do you mean?"),
    (r"\bbad\b|\bissue\b|\bproblem\b","Can you be more specific about what aspect concerns you?"),
    (r"\brecently\b|\blast\b",       "What time period are you asking about?"),
    (r"\bthey\b|\bthem\b|\bit\b",    "Could you clarify who or what you are referring to?"),
]