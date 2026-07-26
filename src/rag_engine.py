"""
RAG engine: hybrid retrieval (dense embeddings + BM25 keyword search, fused
with Reciprocal Rank Fusion) followed by grounded answer generation with
inline source citations.

Why hybrid: dense embeddings catch semantic/paraphrased matches ("bugs eating
my olives" -> olive fruit fly) while BM25 catches exact keyword/jargon matches
(a specific chemical name, a disease name) that embeddings sometimes miss.
Fusing both is a standard production RAG pattern, not just an academic one.
"""

import os
import json
import logging

from dotenv import load_dotenv
load_dotenv()

from embeddings import get_embedder, DenseEmbedder, TfidfEmbedder
from rank_bm25 import BM25Okapi
import chromadb
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
BM25_CORPUS_PATH = os.path.join(DB_DIR, "bm25_corpus.json")
COLLECTION_NAME = "farm_advisor_kb"


def _tokenize(text: str):
    return text.lower().split()


class RAGEngine:
    def __init__(self, top_k: int = 4):
        self.top_k = top_k

        if not os.path.exists(BM25_CORPUS_PATH):
            raise FileNotFoundError(
                "No ingested corpus found. Run `python src/ingest.py` first."
            )

        with open(BM25_CORPUS_PATH, "r", encoding="utf-8") as f:
            self.corpus = json.load(f)

        self.chunk_id_to_idx = {c["chunk_id"]: i for i, c in enumerate(self.corpus)}
        tokenized_corpus = [_tokenize(c["text"]) for c in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

        client = chromadb.PersistentClient(path=DB_DIR)
        self.collection = client.get_collection(COLLECTION_NAME)

        with open(os.path.join(DB_DIR, "embedder_backend.json")) as f:
            backend_info = json.load(f)
        self.embedder = get_embedder()
        if self.embedder.backend != backend_info["backend"]:
            logger.warning(
                "Embedder backend mismatch: index was built with '%s' but "
                "query time loaded '%s'. Re-run ingest.py to fix.",
                backend_info["backend"], self.embedder.backend,
            )
        if isinstance(self.embedder, TfidfEmbedder):
            # TF-IDF vectorizer must be refit on the same corpus it indexed with.
            self.embedder.fit([c["text"] for c in self.corpus])

    def _dense_search(self, query: str, k: int):
        query_emb = self.embedder.encode([query])[0]
        results = self.collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=k,
        )
        return results["ids"][0]  # list of chunk_ids, ranked

    def _sparse_search(self, query: str, k: int):
        scores = self.bm25.get_scores(_tokenize(query))
        ranked_idx = np.argsort(scores)[::-1][:k]
        return [self.corpus[i]["chunk_id"] for i in ranked_idx]

    def retrieve(self, query: str, k: int = None):
        """Hybrid retrieval via Reciprocal Rank Fusion (RRF)."""
        k = k or self.top_k
        pool_size = max(k * 3, 10)

        dense_ranked = self._dense_search(query, pool_size)
        sparse_ranked = self._sparse_search(query, pool_size)

        rrf_scores = {}
        RRF_K = 60  # standard constant from the RRF literature
        for rank, chunk_id in enumerate(dense_ranked):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (RRF_K + rank + 1)
        for rank, chunk_id in enumerate(sparse_ranked):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (RRF_K + rank + 1)

        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [self.corpus[self.chunk_id_to_idx[chunk_id]] for chunk_id, _ in fused]

    def _build_prompt(self, query: str, chunks):
        context_blocks = []
        for i, c in enumerate(chunks, 1):
            context_blocks.append(f"[Source {i}: {c['source']} — {c['section']}]\n{c['text']}")
        context = "\n\n".join(context_blocks)

        prompt = f"""You are an agricultural advisory assistant. Answer the farmer's question
using ONLY the information in the sources below. If the sources don't contain
the answer, say so honestly rather than guessing. Cite sources inline like [Source 1].

SOURCES:
{context}

QUESTION: {query}

ANSWER:"""
        return prompt

    def answer(self, query: str, k: int = None):
        chunks = self.retrieve(query, k)
        prompt = self._build_prompt(query, chunks)

        answer_text = self._generate(prompt)
        return {
            "answer": answer_text,
            "sources": [{"source": c["source"], "section": c["section"]} for c in chunks],
        }

    def _generate(self, prompt: str) -> str:
        groq_key = os.environ.get("GROQ_API_KEY")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

        if groq_key:
            return self._generate_groq(prompt, groq_key)
        elif anthropic_key:
            return self._generate_anthropic(prompt, anthropic_key)
        else:
            return (
                "[No GROQ_API_KEY or ANTHROPIC_API_KEY set — showing retrieved context only. "
                "Get a free Groq API key at https://console.groq.com/keys to enable generated answers.]\n\n"
                + prompt.split("QUESTION:")[0].replace("SOURCES:\n", "")
            )

    def _generate_groq(self, prompt: str, api_key: str) -> str:
        """Free tier, no credit card required. https://console.groq.com/keys"""
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    def _generate_anthropic(self, prompt: str, api_key: str) -> str:
        """Paid, but higher quality. Optional alternative to Groq."""
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in response.content if block.type == "text"
        )
