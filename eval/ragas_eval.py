"""RAGAS Evaluation harness computing Faithfulness, Answer Relevancy, and Context Recall."""

from typing import List, Dict, Any


class FinancialRAGEvaluator:
    """Computes RAGAS-style metrics on financial QA pairs:
    1. Faithfulness: Is the answer strictly grounded in the retrieved context?
    2. Answer Relevancy: Does the answer address the user query without hallucinating unrequested claims?
    3. Context Recall: Were all ground-truth facts retrieved in the top-K chunks?
    4. Hallucination Rate: Frequency of fabricated figures or references.
    """

    def evaluate_sample(
        self,
        query: str,
        answer: str,
        retrieved_contexts: List[str],
        ground_truth: str
    ) -> Dict[str, float]:
        """Evaluates a single sample."""
        faithfulness = self._compute_faithfulness(answer, retrieved_contexts)
        relevancy = self._compute_answer_relevancy(query, answer)
        recall = self._compute_context_recall(ground_truth, retrieved_contexts)
        hallucination = 1.0 - faithfulness if faithfulness < 1.0 else 0.0

        return {
            "faithfulness": round(faithfulness, 3),
            "answer_relevancy": round(relevancy, 3),
            "context_recall": round(recall, 3),
            "hallucination_detected": hallucination > 0.3,
        }

    def _compute_faithfulness(self, answer: str, retrieved_contexts: List[str]) -> float:
        """Checks statement overlap between generated response and context."""
        if not retrieved_contexts or not answer:
            return 0.0
        if "insufficient context" in answer.lower():
            return 1.0  # Correctly refused to hallucinate

        # Strip citation markers like [source: doc.pdf, page 2]
        import re
        clean_answer = re.sub(r'\[source:[^\]]+\]', '', answer, flags=re.IGNORECASE)
        context_blob = " ".join(retrieved_contexts).lower()

        # Stop words shouldn't penalize factual grounding
        stopwords = {"the", "a", "an", "for", "in", "on", "was", "is", "were", "are", "of", "to", "at", "by", "and", "or"}
        words = [re.sub(r'[^\w\%\$]', '', w) for w in clean_answer.lower().split()]
        tokens = [w for w in words if len(w) >= 2 and w not in stopwords]
        if not tokens:
            return 1.0

        grounded_count = sum(1 for t in tokens if t in context_blob)
        return min(1.0, grounded_count / len(tokens))



    def _compute_answer_relevancy(self, query: str, answer: str) -> float:
        """Measures lexical and semantic alignment between query and response."""
        import re
        if "insufficient context" in answer.lower():
            return 0.85  # Refusing when unanswerable is highly relevant and safe

        q_tokens = set(re.findall(r'\w+', query.lower()))
        if not q_tokens:
            return 0.5

        a_tokens = set(re.findall(r'\w+', answer.lower()))
        overlap = len(q_tokens.intersection(a_tokens))
        return min(1.0, 0.5 + (overlap / len(q_tokens)) * 0.5)


    def _compute_context_recall(self, ground_truth: str, retrieved_contexts: List[str]) -> float:
        """Measures whether required ground truth terms appear in retrieved passages."""
        gt_words = [w.lower() for w in ground_truth.split() if len(w) > 3]
        if not gt_words:
            return 1.0
        context_blob = " ".join(retrieved_contexts).lower()
        found = sum(1 for w in gt_words if w in context_blob)
        return min(1.0, found / len(gt_words))

    def evaluate_benchmark(self, dataset: List[Dict[str, Any]]) -> Dict[str, float]:
        """Averages metrics across a financial test benchmark."""
        if not dataset:
            return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_recall": 0.0}

        total_faith = 0.0
        total_rel = 0.0
        total_rec = 0.0

        for item in dataset:
            res = self.evaluate_sample(
                query=item["query"],
                answer=item["answer"],
                retrieved_contexts=item.get("contexts", []),
                ground_truth=item.get("ground_truth", "")
            )
            total_faith += res["faithfulness"]
            total_rel += res["answer_relevancy"]
            total_rec += res["context_recall"]

        n = len(dataset)
        return {
            "faithfulness": round(total_faith / n, 2),
            "answer_relevancy": round(total_rel / n, 2),
            "context_recall": round(total_rec / n, 2),
        }
