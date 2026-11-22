"""All settings in one place. Change a value here and every file picks it up."""
import os
from dotenv import load_dotenv

load_dotenv()

# --- OpenAI ---
# We use AICredits, an OpenAI-compatible gateway, so we also set a base URL.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://aicredits.in/v1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# --- Folders and files ---
RAW_FOLDER = "data/raw"                       # put PDFs here, one sub-folder per scheme
CHUNKS_FILE = "data/processed/chunks.json"
VECTOR_FOLDER = "vectorstore"  #It's the folder where the app saves its Chroma vector database. 
COLLECTION_NAME = "schemes" #collection inside the Chroma database. A collection is like a table. The chunks are stored and searched under this name, so ingestion and the chatbot must use the same one.
LOG_FILE = "logs/queries.jsonl"

# --- Schemes (sub-folder name in data/raw -> display name) ---
SCHEMES = {
    "pm_kisan": "PM-KMY",   # folder name stays pm_kisan; the PDF is the PM-KMY guidelines
    # "pmfby": "PMFBY",   # add more schemes later: create data/raw/pmfby/ and uncomment
}
DEFAULT_SCHEME = "PM-KMY"

# --- Chunking and retrieval ---
CHUNK_SIZE = 2000        # characters (~400 words)
CHUNK_OVERLAP = 200
TOP_K = 4

# Some PDFs are half English, half Hindi, and the Hindi pages come out garbled.
# Pages after this number are ignored (file name -> last page to keep).
LAST_PAGE_TO_KEEP = {"pm_kisan.pdf": 15}
HISTORY_LIMIT = 6        # last 6 messages = last 3 question/answer pairs

# --- Guardrails ---
MAX_QUESTION_CHARS = 500
MIN_RELEVANCE = 0.25     # chunks less similar than this count as "nothing found" (0..1)
GROUNDEDNESS_CHECK = True

FALLBACK_ANSWER = (
    "I don't know based on the official documents. "
    "Please check the official scheme website or your nearest CSC centre."
)
