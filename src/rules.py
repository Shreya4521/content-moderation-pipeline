"""
rules.py
Rule-based compliance checks that run alongside the ML model.
These catch things a text classifier often misses: PII leaks, spam links,
and explicit profanity — useful for GDPR/DPDP-style "personal data exposure" flags.
"""

import re

# --- Profanity list (small demo list; extend with a larger wordlist for production) ---
PROFANITY_WORDS = [
    "idiot", "stupid", "moron", "trash", "garbage", "worthless",
    "shut up", "hate you", "kill yourself", "die", "hurt you"
]

# --- Regex patterns for PII ---
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\d{10}\b")
CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d[ -]*?){13,16}\b")

# --- Spam indicators ---
URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+|\S+\.(com|xyz|link|bit\.ly)\S*)", re.IGNORECASE)
SPAM_KEYWORDS = [
    "click here", "free money", "claim now", "won $", "winner",
    "work from home", "earn $", "dm me", "subscribe now", "limited offer"
]


def check_profanity(text: str) -> dict:
    text_lower = text.lower()
    found = [w for w in PROFANITY_WORDS if w in text_lower]
    return {"flagged": len(found) > 0, "matches": found}


def check_pii(text: str) -> dict:
    emails = EMAIL_PATTERN.findall(text)
    phones = PHONE_PATTERN.findall(text)
    cards = CREDIT_CARD_PATTERN.findall(text)
    flagged = bool(emails or phones or cards)
    return {
        "flagged": flagged,
        "emails_found": len(emails),
        "phones_found": len(phones),
        "cards_found": len(cards),
    }


def check_spam(text: str) -> dict:
    text_lower = text.lower()
    urls = URL_PATTERN.findall(text)
    keyword_hits = [kw for kw in SPAM_KEYWORDS if kw in text_lower]
    flagged = bool(urls) or len(keyword_hits) > 0
    return {"flagged": flagged, "urls_found": len(urls), "keyword_matches": keyword_hits}


def run_all_rules(text: str) -> dict:
    """Runs every rule-based check and returns a consolidated result."""
    profanity = check_profanity(text)
    pii = check_pii(text)
    spam = check_spam(text)

    any_flag = profanity["flagged"] or pii["flagged"] or spam["flagged"]

    return {
        "rule_flagged": any_flag,
        "profanity": profanity,
        "pii": pii,
        "spam": spam,
    }
