"""Tests for RAGAS evaluation metrics calculation."""

import pytest
from eval.ragas_eval import FinancialRAGEvaluator


def test_ragas_evaluator_grounded_sample():
    evaluator = FinancialRAGEvaluator()
    sample = {
        "query": "What was the CET1 ratio for Q3 2024?",
        "answer": "The CET1 ratio for Q3 2024 was 15.3% [source: report.pdf, page 2].",
        "contexts": ["In Q3 2024, our Common Equity Tier 1 (CET1) ratio reached 15.3%."],
        "ground_truth": "The CET1 ratio reached 15.3% in Q3 2024."
    }
    metrics = evaluator.evaluate_sample(
        query=sample["query"],
        answer=sample["answer"],
        retrieved_contexts=sample["contexts"],
        ground_truth=sample["ground_truth"]
    )
    assert metrics["faithfulness"] >= 0.7
    assert metrics["answer_relevancy"] >= 0.7
    assert metrics["context_recall"] >= 0.7
    assert not metrics["hallucination_detected"]


def test_ragas_evaluator_hallucinated_sample():
    evaluator = FinancialRAGEvaluator()
    sample = {
        "query": "What was the dividend yield?",
        "answer": "The dividend yield was 8.4% paid quarterly.",
        "contexts": ["The firm reported net earnings of $4B and did not distribute dividend."],
        "ground_truth": "No dividend was paid."
    }
    metrics = evaluator.evaluate_sample(
        query=sample["query"],
        answer=sample["answer"],
        retrieved_contexts=sample["contexts"],
        ground_truth=sample["ground_truth"]
    )
    assert metrics["faithfulness"] < 0.6


def test_ragas_evaluator_benchmark():
    evaluator = FinancialRAGEvaluator()
    dataset = [
        {
            "query": "What is the Tier 1 leverage ratio?",
            "answer": "Tier 1 leverage ratio was 5.4% [source: filings.pdf, page 1].",
            "contexts": ["Tier 1 leverage ratio was 5.4%."],
            "ground_truth": "Tier 1 leverage ratio is 5.4%."
        },
        {
            "query": "What is the risk-free rate assumed in Black-Scholes?",
            "answer": "Insufficient context: not found.",
            "contexts": ["Equities portfolio value was $100M."],
            "ground_truth": "Not available."
        }
    ]
    benchmark_res = evaluator.evaluate_benchmark(dataset)
    assert benchmark_res["faithfulness"] >= 0.8
    assert benchmark_res["answer_relevancy"] >= 0.6
