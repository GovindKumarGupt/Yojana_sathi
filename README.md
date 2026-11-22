# Yojana Saathi – Government Scheme RAG Chatbot

A RAG chatbot that answers questions about Indian government schemes (starting with PM-KISAN)
in English or Hindi, using **only** the official guideline PDFs, with source citations,
guardrails and an evaluation suite.

## Features
- **RAG**: PDF → clean → chunk → OpenAI embeddings → Chroma vector DB → top-k retrieval filtered by scheme
- **Conversation memory**: follow-up questions ("what documents do they need?") are rewritten into standalone questions
- **Hindi support**: Hindi questions are translated for search, answers come back in Hindi
- **Citations**: every answer lists the source file and page
- **Guardrails**
  - Input: personal-data masking (Aadhaar, PAN, phone, bank account, email), prompt-injection blocking, off-topic filter, length limit
  - Output: low-relevance fallback (no guessing), LLM groundedness check before showing an answer
- **Evaluation**: retrieval recall@k and MRR, LLM-as-judge answer accuracy, refusal accuracy, groundedness rate, latency
- **Logging**: every query logged to `logs/queries.jsonl` (masked question, flags, latency)

## Architecture
```
question
  → input guardrails (PII mask · injection · off-topic)
  → memory (rewrite follow-up → standalone English question)
  → retriever (Chroma, filtered by scheme, top 4)
  → relevance check (nothing relevant → "I don't know")
  → generator (GPT-4o-mini, answer only from context, cite sources)
  → groundedness check (unsupported → "I don't know")
  → answer + sources
```

## Project structure
```
scheme-rag-chatbot/
├── data/raw/pm_kisan/        # official PDFs (one sub-folder per scheme)
├── data/processed/           # chunks.json (generated)
├── vectorstore/              # Chroma DB (generated)
├── logs/                     # queries.jsonl (generated)
├── config.py                 # all settings
├── ingest.py                 # extract, clean, chunk
├── embed_store.py            # embed + store in Chroma
├── retriever.py              # similarity search
├── guardrails.py             # input + output safety checks
├── generator.py              # LLM answer with citations
├── memory.py                 # follow-up question rewriting
├── rag_pipeline.py           # connects everything (+ terminal chat)
├── app.py                    # Streamlit UI
└── eval/
    ├── test_questions.json   # test set (direct, tricky, Hindi, not-in-docs, off-topic, injection)
    ├── test_guardrails.py    # unit tests, no API needed
    ├── test_retrieval.py     # recall@k, MRR
    └── evaluate.py           # end-to-end accuracy, refusal, groundedness, latency
```

## Setup
```bash
python -m venv venv
venv\Scripts\activate            # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env           # Mac/Linux: cp .env.example .env  → then add your OpenAI key
```
Download the PM-KISAN operational guidelines and FAQ PDFs from https://pmkisan.gov.in
and put them in `data/raw/pm_kisan/`.

## Run
```bash
python ingest.py                     # 1. PDFs → chunks.json
python embed_store.py                # 2. chunks → Chroma
python retriever.py                  # 3. check the search results look right
python rag_pipeline.py               # 4. chat in the terminal
python eval/test_guardrails.py       # 5a. guardrail unit tests (no API)
python eval/test_retrieval.py        # 5b. retrieval score
python eval/evaluate.py              # 5c. full evaluation
streamlit run app.py                 # 6. web app
```

**Before running the evals:** open your PDFs and fix `expected_file` / `expected_page` /
`expected` in `eval/test_questions.json` to match your files. Grow the set to 25–30 questions.

## Results
| Metric | Score |
|---|---|
| Retrieval recall@4 | 95% (18/19) |
| Answer accuracy (LLM judge) | 65% (17/26) |
| Refusal accuracy | 81% |
| Groundedness rate | 100% (of 18 checked answers) |
| Avg latency | 5.0 s |

_Measured on 26 test questions (direct, tricky, Hindi, follow-up, not-in-docs, off-topic, injection). First working version, not yet tuned._

## Deploy (Streamlit Community Cloud)
Push to GitHub (without `.env`), create an app on share.streamlit.io, and add
`OPENAI_API_KEY` under **Secrets**. Because `vectorstore/` is git-ignored, either commit it
or run `ingest.py` and `embed_store.py` once at startup.

## Roadmap (agent phase)
Document photo reading (vision) → mock application portal → Playwright form filling →
LangGraph workflow with OTP pause (human-in-the-loop) → FastAPI backend + mobile app.
