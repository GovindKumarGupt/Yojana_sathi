"""Chat memory: turn follow-up questions into full standalone English questions.

Example:
  history: "Who can apply for PM-KISAN?" -> (answer)
  new:     "uske liye kaunse documents chahiye?"
  rewrite: "What documents are needed to apply for PM-KISAN?"
"""
from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_BASE_URL, CHAT_MODEL, HISTORY_LIMIT

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


def format_history(history, limit=HISTORY_LIMIT):
    """Keep only the last few messages and turn them into plain text."""
    recent = history[-limit:]
    lines = []
    for m in recent:
        content = m["content"]
        if m["role"] == "assistant" and len(content) > 400:
            content = content[:400] + "..."        # long answers: keep it short
        lines.append(f"{m['role']}: {content}")
    return "\n".join(lines)


def rewrite_question(question, history, scheme):
    """Return a standalone English question. Also translates Hindi -> English."""
    chat = format_history(history) or "(no earlier messages)"
    prompt = f"""The user is asking about the government scheme: {scheme}.

Chat so far:
{chat}

New question: {question}

Rewrite the new question as ONE complete question in English that makes sense
on its own (replace words like "it", "they", "this" with what they refer to).
If it is already complete and in English, return it unchanged.
Return only the rewritten question."""
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip().strip('"')


if __name__ == "__main__":
    history = [
        {"role": "user", "content": "Who can apply for PM-KISAN?"},
        {"role": "assistant", "content": "Small and marginal landholding farmer families..."},
    ]
    print(rewrite_question("uske liye kaunse documents chahiye?", history, "PM-KISAN"))
