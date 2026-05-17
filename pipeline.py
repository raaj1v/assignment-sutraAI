from __future__ import annotations

import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer

import config
from analytics import is_analytical, run_analytical_query
from guardrails import detect_ambiguity, detect_injection
from parsers import load_corpus
from vector_store import build_vector_store


class RAGPipeline:
    def __init__(
        self,
        doc_dir=config.DOC_DIR,
        db_path=config.DB_PATH,
    ):
        self.conversation_history: list[dict] = []

        print("Loading embedding model…")
        self.embed_model = SentenceTransformer(config.EMBED_MODEL_NAME)

        print("Loading corpus…")
        all_chunks, self.dataframes = load_corpus(doc_dir)

        print("Building vector store…")
        self.collection = build_vector_store(all_chunks, db_path, self.embed_model)

        self.llm = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
        print("Pipeline ready.\n")

    def ask(self, question: str) -> dict:
        if detect_injection(question):
            return dict(answer="Your query contains patterns that look like prompt injection. Please rephrase.", confidence=0.0, sources=[], flag="injection_detected")

        clarification = detect_ambiguity(question)
        if clarification:
            return dict(answer=f"Your question seems ambiguous. {clarification}", confidence=None, sources=[], flag="clarification_needed")

        if is_analytical(question) and self.dataframes:
            analytical_result = run_analytical_query(question, self.dataframes)
            if analytical_result:
                self._update_history(question, analytical_result)
                return dict(answer=analytical_result, confidence=1.0, sources=[{"source_file": f} for f in self.dataframes], flag="analytical")

        search_query = self._rewrite_query(question)
        retrieved_docs = self._retrieve(search_query)
        top_confidence = self._confidence(retrieved_docs[0]["distance"])

        if top_confidence < config.CONFIDENCE_THRESHOLD:
            return dict(answer=f"I could not find sufficient evidence in the documents to answer this. Best match confidence: {top_confidence:.2f}. Try rephrasing or check that the relevant document is loaded.", confidence=top_confidence, sources=[], flag="low_confidence")

        prompt = self._build_prompt(question, retrieved_docs)
        answer = self._generate(prompt)
        self._update_history(question, answer)

        sources = [
            {
                "source_file": r["metadata"]["source_file"],
                "page":        r["metadata"].get("page"),
                "row":         r["metadata"].get("row"),
                "snippet":     r["text"][:200],
                "confidence":  round(self._confidence(r["distance"]), 2),
            }
            for r in retrieved_docs
        ]
        return dict(answer=answer, confidence=round(top_confidence, 2), sources=sources, flag="ok")

    def evaluate(self, test_cases: list[dict]) -> list[dict]:
        import numpy as np

        print(f"\n{'='*60}")
        print(f"{'EVALUATION REPORT':^60}")
        print(f"{'='*60}")

        scores = []
        for tc in test_cases:
            result = self.ask(tc["question"])
            answer = result["answer"].lower()
            keywords = tc.get("expected_keywords", [])
            kw_hits = sum(1 for kw in keywords if kw.lower() in answer)
            kw_score = kw_hits / len(keywords) if keywords else 1.0

            source_match = None
            if tc.get("expected_source"):
                cited = any(
                    tc["expected_source"] in str(s.get("source_file", ""))
                    for s in result.get("sources", [])
                )
                source_match = int(cited)

            scores.append({
                "question":     tc["question"][:60],
                "keyword_score": round(kw_score, 2),
                "source_match": source_match,
                "confidence":   result.get("confidence"),
                "flag":         result.get("flag"),
            })

            print(f"\nQ : {tc['question'][:60]}")
            print(f"  keyword_score : {kw_score:.2f} ({kw_hits}/{len(keywords)} keywords found)")
            if source_match is not None:
                print(f"  source_match  : {'yes' if source_match else 'NO'}")
            print(f"  confidence    : {result.get('confidence')}")
            print(f"  flag          : {result.get('flag')}")

        avg_kw = np.mean([s["keyword_score"] for s in scores])
        print(f"\n{'─'*60}")
        print(f"Average keyword score : {avg_kw:.2f}")
        print(f"{'='*60}\n")
        return scores

    def _retrieve(self, query: str) -> list[dict]:
        q_emb = self.embed_model.encode(
            query, normalize_embeddings=True
        ).tolist()
        results = self.collection.query(
            query_embeddings=[q_emb], n_results=config.TOP_K
        )
        return [
            {"text": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def _rewrite_query(self, question: str) -> str:
        """Skip LLM rewrite when there's no history or no anaphora."""
        ANAPHORA = {"it", "that", "they", "them", "this", "he", "she",
                    "above", "previous", "earlier", "step", "those"}
        tokens = set(question.lower().split())
        if not self.conversation_history or not (tokens & ANAPHORA):
            return question   # fast path — no rewrite needed

        recent = self.conversation_history[-4:]
        history_text = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in recent
        )
        prompt = (
            f"Given this conversation history:\n{history_text}\n\n"
            f"Rewrite the following question as a fully self-contained search query.\n"
            f"Output ONLY the rewritten query, nothing else.\n\n"
            f"Question: {question}\nRewritten query:"
        )
        try:
            resp = self.llm.chat.completions.create(
                model=config.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[WARN] Query rewrite failed: {e}. Using original.")
            return question

    def _build_prompt(self, question: str, retrieved_docs: list[dict]) -> str:
        context = "".join(
            f"\nSOURCE: {r['metadata']['source_file']}\n{r['text']}\n"
            for r in retrieved_docs
        )
        history_text = ""
        if self.conversation_history:
            history_text = "\nCONVERSATION SO FAR:\n" + "".join(
                f"{m['role'].capitalize()}: {m['content']}\n"
                for m in self.conversation_history[-(config.MAX_HISTORY_TURNS * 2):]
            )
        return (
            "You are a grounded enterprise assistant.\n"
            "Rules:\n"
            "- Answer ONLY from the provided context below.\n"
            "- If the context does not contain enough information, say so explicitly.\n"
            "- Always cite the source file(s) you used.\n"
            "- Never fabricate data, names, or figures.\n"
            "- Use the conversation history to understand follow-up questions.\n"
            f"{history_text}\n"
            f"CONTEXT:\n{context}\n"
            f"QUESTION:\n{question}\n\n"
            "ANSWER:"
        )

    def _generate(self, prompt: str) -> str:
        try:
            resp = self.llm.chat.completions.create(
                model=config.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"[ERROR] LLM call failed: {e}"

    def _update_history(self, question: str, answer: str) -> None:
        self.conversation_history.append({"role": "user",      "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})
        # Keep a rolling window: MAX_HISTORY_TURNS * 2 messages
        max_msgs = config.MAX_HISTORY_TURNS * 2
        if len(self.conversation_history) > max_msgs:
            self.conversation_history = self.conversation_history[-max_msgs:]

    @staticmethod
    def _confidence(distance: float) -> float:
        return 1.0 - (distance / 2.0)