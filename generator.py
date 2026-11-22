"""Step 4: write the answer from the retrieved chunks."""
from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_BASE_URL, CHAT_MODEL, FALLBACK_ANSWER
from retriever import retrieve

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

SYSTEM_PROMPT = f"""You are "Yojana Saathi", a helpful assistant for Indian government schemes.

Rules:
1. Answer ONLY using the CONTEXT. Never use outside knowledge, even if you know the answer
   (for example amounts or rules of a different scheme that is only mentioned in passing).
2. If the CONTEXT does not contain the answer, reply with exactly: "{FALLBACK_ANSWER}"
   and nothing else (no "Source:" line).
   Read rules carefully: if a group is listed as an exception (for example "excluding ..."),
   do not say that group is covered by the rule.
3. Use simple words that a farmer can understand. Use short bullet points for lists.
4. Reply in the same language as the user's ORIGINAL question (Hindi -> Hindi, English -> English).
5. End with a line "Source:" listing the file name and page number(s) you used.
6. Never ask for or repeat Aadhaar, bank or phone numbers."""


def build_context(chunks):
    """Join chunks into one block, each labelled with its source."""
    return "\n\n".join(
        f"[Source: {c['file_name']}, page {c['page']}]\n{c['text']}" for c in chunks
    )


def generate_answer(question, scheme, original_question=None, chunks=None):
    """Retrieve (if chunks not given) and ask the LLM.

    Returns {"answer": str, "chunks": list, "context": str}.
    """
    if chunks is None:
        chunks = retrieve(question, scheme)
    context = build_context(chunks)

    user_msg = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    if original_question and original_question != question:
        user_msg += f"\n\nORIGINAL QUESTION (reply in this language): {original_question}"

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
    )
    return {
        "answer": response.choices[0].message.content.strip(),
        "chunks": chunks,
        "context": context,
    }


if __name__ == "__main__":
    result = generate_answer("Who is not eligible for PM-KISAN?", "PM-KMY")
    print(result["answer"])
