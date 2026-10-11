"""Financial QA prompt templates with few-shot audit-grade examples."""

SYSTEM_PROMPT = """You are FinSight AI, a document intelligence engine designed for Tier-1 financial institutions (e.g. JPMorgan Chase, Goldman Sachs, UBS).
Your task is to answer the user's financial or regulatory question strictly and solely based on the provided context passages.

CRITICAL INSTRUCTIONS:
1. ONLY make claims directly substantiated by the Context.
2. For EVERY factual claim or figure, cite the source document and page number in brackets: [source: <doc_name>, page <N>].
3. If the provided context does not contain sufficient facts to answer the question with high certainty, state clearly: "Insufficient context: the provided documents do not contain information regarding [topic]." Do not extrapolate or fabricate figures.
4. Output cleanly in the requested JSON structure.
"""

FEW_SHOT_EXAMPLES = """
[Example 1]
Context:
[Doc: Q3_2024_Earnings.pdf, Page 14]
In the third quarter of 2024, our Common Equity Tier 1 (CET1) ratio reached 15.3%, up 40 bps year-over-year. Net Interest Income was $22.7B.

Query: What was the company's CET1 ratio in Q3 2024?
Response JSON:
{
  "answer": "In Q3 2024, the Common Equity Tier 1 (CET1) ratio was 15.3%, representing a 40 bps year-over-year increase [source: Q3_2024_Earnings.pdf, page 14].",
  "confidence": 0.98,
  "escalate": false
}

[Example 2]
Context:
[Doc: Compliance_Policy_v4.pdf, Page 3]
All employee personal account trades in covered securities must be pre-cleared via the automated compliance portal.

Query: What is the maximum daily trading limit for non-executive employees?
Response JSON:
{
  "answer": "Insufficient context: the provided documents do not state the maximum daily trading limit for non-executive employees.",
  "confidence": 0.15,
  "escalate": true
}
"""

def build_qa_prompt(query: str, context_chunks: list) -> str:
    """Builds prompt containing formatted contexts and user question."""
    formatted_contexts = []
    for c in context_chunks:
        header = f"[Doc: {c.metadata.filename}, Page {c.metadata.page_number}]"
        formatted_contexts.append(f"{header}\n{c.content}")

    context_block = "\n\n".join(formatted_contexts) if formatted_contexts else "NO CONTEXT AVAILABLE."

    prompt = f"""{SYSTEM_PROMPT}

{FEW_SHOT_EXAMPLES}

---
CURRENT CONTEXT:
{context_block}

USER QUERY:
{query}

Provide your response in strictly valid JSON format:
{{
  "answer": "<grounded response with [source: doc_name, page N] citations>",
  "confidence": <float between 0.0 and 1.0>,
  "escalate": <true if insufficient context or low confidence, false otherwise>
}}
"""
    return prompt
