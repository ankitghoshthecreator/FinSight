"""Security, PII redaction, and compliance guardrails for enterprise financial RAG.
Target pattern: JPMC / Goldman Sachs / MUFG audit standards.
"""

import re
from typing import Tuple, List, Dict


class PIIRedactor:
    """Redacts Personally Identifiable Information (PII) before vector indexing.
    
    Compliance patterns:
    - Social Security Numbers (SSN)
    - Credit Card numbers (PAN)
    - Bank Account / Routing numbers
    - Email addresses
    - Phone numbers
    """

    PATTERNS: Dict[str, re.Pattern] = {
        "SSN": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
        "CREDIT_CARD": re.compile(r'\b(?:\d{4}[- ]?){3}\d{4}\b'),
        "EMAIL": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        "PHONE": re.compile(r'\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b'),
        "BANK_ACCOUNT": re.compile(r'\b(?:account|acct|iban)[\s:#]+([0-9A-Z]{8,24})\b', re.IGNORECASE),
    }

    def redact(self, text: str) -> Tuple[str, List[str]]:
        """Redacts sensitive PII entities from text and returns cleaned text with detected types."""
        redacted_types = []
        cleaned_text = text

        for pii_type, pattern in self.PATTERNS.items():
            if pattern.search(cleaned_text):
                redacted_types.append(pii_type)
                cleaned_text = pattern.sub(f"[REDACTED_{pii_type}]", cleaned_text)

        return cleaned_text, redacted_types
