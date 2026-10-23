"""Financial query rewriter and query expansion module.
Target pattern: JPMC / Goldman Sachs / MUFG domain query expansion.

Expands financial acronyms, normalizes regulatory identifiers, and generates
sub-queries / HyDE-style expansions to optimize hybrid sparse and dense retrieval.
"""

import re
from typing import Dict, List, Set


class FinancialQueryRewriter:
    """Expands regulatory acronyms and generates search variants for financial queries."""

    # Curated financial domain taxonomy and acronym mapping
    FINANCIAL_ACRONYMS: Dict[str, str] = {
        "SLR": "Supplementary Leverage Ratio",
        "CET1": "Common Equity Tier 1 capital ratio",
        "LCR": "Liquidity Coverage Ratio",
        "NSFR": "Net Stable Funding Ratio",
        "CCAR": "Comprehensive Capital Analysis and Review stress test",
        "DFAST": "Dodd-Frank Act Stress Testing",
        "VAR": "Value at Risk market risk",
        "SOFR": "Secured Overnight Financing Rate benchmark",
        "LIBOR": "London Interbank Offered Rate benchmark",
        "ISDA": "International Swaps and Derivatives Association master agreement",
        "RWA": "Risk-Weighted Assets",
        "RWAS": "Risk-Weighted Assets",
        "TLAC": "Total Loss-Absorbing Capacity",
        "G-SIB": "Global Systemically Important Bank surcharge",
        "GSIB": "Global Systemically Important Bank surcharge",
        "EPS": "Earnings Per Share diluted",
        "EBITDA": "Earnings Before Interest Taxes Depreciation Amortization",
        "KYC": "Know Your Customer customer due diligence",
        "AML": "Anti-Money Laundering BSA compliance",
        "SEC": "Securities and Exchange Commission regulatory filing",
        "10-K": "annual report Form 10-K SEC",
        "10-Q": "quarterly report Form 10-Q SEC",
        "8-K": "current report Form 8-K SEC",
    }

    def __init__(self, expand_acronyms: bool = True):
        self.expand_acronyms = expand_acronyms

    def rewrite(self, query: str) -> str:
        """Enriches the query with expanded regulatory terms while preserving the original query terms.
        
        Example:
            'What is Goldman Sachs SLR?'
            -> 'What is Goldman Sachs SLR Supplementary Leverage Ratio?'
        """
        if not query or not query.strip():
            return query

        if not self.expand_acronyms:
            return query.strip()

        words = re.findall(r'\b[A-Za-z0-9\-]+\b', query)
        added_expansions: Set[str] = set()

        for w in words:
            upper_w = w.upper()
            if upper_w in self.FINANCIAL_ACRONYMS and upper_w not in added_expansions:
                expansion = self.FINANCIAL_ACRONYMS[upper_w]
                # Only add if the expansion isn't already present in query
                if expansion.lower() not in query.lower():
                    added_expansions.add(expansion)

        if added_expansions:
            enriched = f"{query.strip()} ({' '.join(added_expansions)})"
            return enriched

        return query.strip()

    def generate_query_variants(self, query: str) -> List[str]:
        """Generates multi-query search variants to improve recall across lexical and semantic layers."""
        variants = [query.strip()]
        rewritten = self.rewrite(query)
        if rewritten != query.strip():
            variants.append(rewritten)

        # Generate a keyword-focused formulation by stripping common conversational interrogatives
        cleaned = re.sub(
            r'^(what is|what are|explain|describe|tell me about|how does|can you find|where is)\s+',
            '',
            query.strip(),
            flags=re.IGNORECASE,
        ).rstrip('?')

        if cleaned and cleaned.lower() != query.strip().lower():
            variants.append(cleaned)

        return list(dict.fromkeys(variants))
