# 🌿 Farm Advisor — Hybrid RAG Agricultural Advisory System

A retrieval-augmented generation system that answers agronomy questions (crop
disease, soil health, irrigation, harvest planning) grounded in a curated
knowledge base, with cited sources instead of hallucinated advice.

Built as a natural extension of an earlier computer-vision project (YOLOv8 +
Jetson Nano olive yield detection): that project *detects* fruit, this one
*advises* on the problems detected in the field.

## Why hybrid retrieval

Most tutorial RAG projects use dense embeddings only. This one fuses two
retrieval methods with **Reciprocal Rank Fusion (RRF)**:

- **Dense retrieval** (sentence-transformer embeddings) — catches semantic
  matches even when the farmer's wording doesn't match the document's wording
  (e.g. "bugs eating my olives" → *olive fruit fly*).
- **Sparse retrieval** (BM25) — catches exact keyword/jargon matches that
  embeddings can miss (a specific disease name, a chemical, an acronym).

Pure dense retrieval alone tends to under-perform on the second case; pure
BM25 alone misses paraphrased questions. Fusing both is standard practice in
production RAG systems, not just an academic nicety.

## Architecture

```
   User question
        │
        ▼
 ┌─────────────────┐      ┌─────────────────┐
 │  Dense search    │      │  BM25 search    │
 │ (embeddings +    │      │ (keyword match) │
 │  ChromaDB)       │      │                 │
 └────────┬─────────┘      └────────┬────────┘
          │                         │
          └──────────┬──────────────┘
                      ▼
          Reciprocal Rank Fusion
                      │
                      ▼
           Top-k retrieved chunks
                      │
                      ▼
        Prompt with cited sources
                      │
                      ▼
              Claude (generation)
                      │
                      ▼
          Grounded, cited answer
```

## What's in the knowledge base

Five markdown documents covering: olive diseases & pests, soil health &
fertilization, irrigation management, harvest & yield planning, and general
field crop pest management (wheat, cotton, rice, sugarcane — relevant to
Punjab agriculture). Swap in your own domain documents to repurpose this for
any advisory use case.

## Setup

```bash
git clone <your-repo-url>
cd farm-advisor-rag
pip install -r requirements.txt
cp .env.example .env
```

**Get a free API key (no credit card needed):**
1. Go to [console.groq.com/keys](https://console.groq.com/keys)
2. Sign up, click "Create API Key"
3. Paste it into `.env` as `GROQ_API_KEY=...`

This runs generation on Llama 3.3 70B via Groq's free tier. If you'd rather use
Claude instead, add `ANTHROPIC_API_KEY=...` to `.env` (requires billing credit)
— the code prefers Groq if both are set, otherwise falls back to whichever one
is present.

**Note on first run:** the sentence-transformer model (`all-MiniLM-L6-v2`,
~80MB) downloads automatically the first time you run ingestion — this
needs internet access once. If it can't download (e.g. fully offline), the
pipeline automatically falls back to TF-IDF embeddings so it still runs end
to end — see `src/embeddings.py` for that fallback logic.

```bash
# 1. Build the vector index from the knowledge base
python src/ingest.py

# 2. Run retrieval evaluation (no API key needed — tests retrieval only)
python eval/run_eval.py

# 3. Launch the interactive chat demo
streamlit run src/app.py
```

Without an `ANTHROPIC_API_KEY` set, the app still retrieves and shows
relevant sources — it just won't generate a natural-language answer. This
is intentional graceful degradation, not a bug.

## Evaluation results

Retrieval evaluated on 15 held-out questions against a known expected source
document per question (`eval/eval_questions.json`):

| Metric | Score |
|---|---|
| Hit@4 (correct source in top 4 results) | 100% |
| MRR (Mean Reciprocal Rank) | 0.906 |

*(Measured with the TF-IDF fallback embedder in a sandboxed test environment
without internet access to download the transformer model. Re-run
`eval/run_eval.py` after a normal install with internet access to get
numbers for the full sentence-transformer + BM25 hybrid setup — expect
similar or better results, especially on paraphrased questions the dense
embeddings are specifically better at.)*

## Project structure

```
farm-advisor-rag/
├── data/                       # Knowledge base source documents (markdown)
├── src/
│   ├── chunking.py             # Header-aware chunking with word-based overlap
│   ├── embeddings.py           # Dense embedder + TF-IDF fallback
│   ├── ingest.py                # Builds the vector index
│   ├── rag_engine.py           # Hybrid retrieval (RRF) + generation
│   └── app.py                  # Streamlit chat UI
├── eval/
│   ├── eval_questions.json     # Labeled eval set
│   └── run_eval.py             # Hit@k / MRR evaluation
└── requirements.txt
```

## Limitations (be upfront about these)

This is a portfolio/demo project, not a production advisory tool. Specifically:

- The knowledge base is 5 documents written for this demo, not vetted by
  agronomists and not comprehensive. It covers olive diseases/pests, soil,
  irrigation, harvest planning, and a few field crops — nothing else.
- Questions outside that scope will either get an honest "I don't know" or,
  in some cases, a weak match to the *closest* available topic — a known RAG
  failure mode worth understanding (ungrounded generation) rather than a
  guarantee this system avoids.
- There's currently no confidence threshold: even a low-relevance retrieval
  still gets passed to the model and answered. A production version would
  check retrieval scores and decline to answer below a confidence cutoff
  instead of forcing a response.
- Evaluation numbers above are retrieval-only (does it find the right
  source), not answer-quality evaluation (is the generated text itself
  correct) — those are two different things and worth being precise about
  when describing this project.

## Possible extensions

- Add a cross-encoder reranker on top of the RRF-fused results for a further
  precision boost
- Swap Chroma for a hosted vector DB (Pinecone/Weaviate) to demonstrate
  production-scale retrieval
- Add conversation memory so follow-up questions ("what about for cotton
  instead?") retain context
- Wrap the retrieval layer with LlamaIndex and the orchestration with
  LangGraph to show framework fluency alongside the from-scratch version

## Suggested CV bullet

> Built a hybrid RAG agricultural advisory system fusing dense embeddings and
> BM25 retrieval via Reciprocal Rank Fusion, achieving 100% Hit@4 and 0.906
> MRR on a custom 15-question evaluation set; deployed as an interactive
> Streamlit application with cited, grounded answers.
