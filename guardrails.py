import re
from config import INJECTION_PATTERNS, AMBIGUOUS_PATTERNS


def detect_injection(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in INJECTION_PATTERNS)


def detect_ambiguity(query: str) -> str | None:
    return next((q for p, q in AMBIGUOUS_PATTERNS if re.search(p, query.lower())), None)