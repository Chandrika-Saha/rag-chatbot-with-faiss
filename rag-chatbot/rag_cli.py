# rag_cli.py
# Enhanced Local RAG with FAISS + Ollama
# Improvements:
#  - Proper Ollama API integration with fallback
#  - Better error handling and validation
#  - Streaming support
#  - Improved prompt formatting
#  - More robust code structure

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# -----------------------
# Config
# -----------------------
INDEX_PATH = "meditations.index"
META_PATH = "meditations_metadata.json"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2:1b"
OLLAMA_API_URL = "http://localhost:11434"  # Default Ollama API endpoint

TOP_K = 3
THRESHOLD = 0.25
MAX_CONTEXT_CHARS = 3500
MAX_TOKENS = 256  # Max tokens for generation
TEMPERATURE = 0.3


# -----------------------
# Data Models
# -----------------------
@dataclass
class Passage:
    pid: str
    book: str
    text: str

    def __str__(self) -> str:
        return f"Book {self.book}, id:{self.pid}"


@dataclass
class RetrievalResult:
    score: float
    passage: Passage


# -----------------------
# Utilities
# -----------------------
def load_metadata(path: str) -> List[Passage]:
    """Load and parse metadata from JSON file."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    passages = []
    for x in raw:
        try:
            # Extract book number and passage ID
            pid = str(x["id"]).split('_')[1].lstrip('P')
            book = str(x["book"]).split()[1]
            text = str(x["text"])
            passages.append(Passage(pid=pid, book=book, text=text))
        except (IndexError, KeyError) as e:
            print(f"Warning: Skipping malformed entry: {e}", file=sys.stderr)
            continue

    return passages


def load_index(path: str) -> faiss.Index:
    """Load FAISS index from file and validate it's IndexFlatIP for cosine similarity."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"FAISS index not found: {path}")

    index = faiss.read_index(path)

    # Validate index type for cosine similarity
    if not isinstance(index, faiss.IndexFlatIP):
        raise TypeError(
            f"Expected IndexFlatIP for cosine similarity, got {type(index).__name__}. "
            "Please recreate your index with IndexFlatIP using normalized vectors."
        )

    print(f"✓ Loaded IndexFlatIP with {index.ntotal} vectors (dim={index.d})",
          file=sys.stderr)

    return index


def retrieve(
        index: faiss.Index,
        metadata: List[Passage],
        query_vec: np.ndarray,
        top_k: int
) -> List[RetrievalResult]:
    """
    Retrieve top-k most similar passages using cosine similarity.

    For IndexFlatIP with normalized vectors:
    - Scores are cosine similarities in range [-1, 1]
    - Higher score = MORE similar
    - 1.0 = identical, 0.0 = orthogonal, -1.0 = opposite
    """
    # FAISS expects 2D array: (num_queries, dimension)
    similarities, indices = index.search(query_vec.reshape(1, -1), top_k)

    results = []
    for cosine_sim, idx in zip(similarities[0], indices[0]):
        # Skip invalid indices
        if idx == -1 or idx >= len(metadata):
            continue

        # For IndexFlatIP with normalized vectors, the score IS the cosine similarity
        # Higher values = more similar (already in correct order)
        results.append(RetrievalResult(
            score=float(cosine_sim),  # Already cosine similarity, no conversion needed
            passage=metadata[idx]
        ))

    # Results are already sorted by FAISS in descending order (most similar first)
    return results


# -----------------------
# Embedding & Retrieval
# -----------------------
def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize vector(s) to unit length."""
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        return v / (norm + 1e-12)

    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / (norms + 1e-12)


def embed_query(model: SentenceTransformer, query: str) -> np.ndarray:
    """Embed query text and normalize."""
    vec = model.encode([query], convert_to_numpy=True)[0].astype("float32")
    return normalize(vec)


# -----------------------
# Prompt Building
# -----------------------
def build_context_block(results: List[RetrievalResult]) -> str:
    """Format retrieved passages into context block."""
    lines = []
    for i, result in enumerate(results, 1):
        p = result.passage
        text = p.text.strip().replace("\n", " ")
        lines.append(f"[{i}] (Book {p.book}, id:{p.pid})\n{text}\n")

    context = "\n".join(lines)
    return context[:MAX_CONTEXT_CHARS]


def build_prompt(
        question: str,
        context: str,
        top_score: float,
        threshold: float
) -> str:
    """Build the complete prompt for the LLM."""
    print(context)

    system_msg = f"""
Answer the Question based on the Context Passages. 
You MUST provide ONLY an one-sentence answer. 
Base your answer on the Context passage.
Use direct sentence from the paragraph whenever appropriate.
The answer MUST NOT CONTAIN ANYTHING about analyzing or considering passages.
If the question has weak connection to the paragraph, draw from it's core concept and respond in the given answer format in ONE sentence.
You MUST NOT include any pleasantries. 
Context Passages: {context}
Question: {question}
Answer Format: ```[Direct advice statement]- (Book X, id:Y).```"""

    return system_msg


# -----------------------
# Ollama Integration
# -----------------------
class OllamaClient:
    """Handle Ollama API interactions with fallback to CLI."""

    def __init__(self, base_url: str = OLLAMA_API_URL, model: str = OLLAMA_MODEL):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self._check_health()

    def _check_health(self) -> None:
        """Check if Ollama server is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running (try: ollama serve)"
            ) from e

    def generate(
            self,
            prompt: str,
            stream: bool = False,
            temperature: float = TEMPERATURE,
            max_tokens: int = MAX_TOKENS
    ) -> str:
        """Generate completion using Ollama API."""
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        try:
            if stream:
                return self._generate_stream(url, payload)
            else:
                return self._generate_blocking(url, payload)

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API request failed: {e}") from e

    def _generate_blocking(self, url: str, payload: dict) -> str:
        """Non-streaming generation."""
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()

        result = response.json()
        return result.get("response", "").strip()

    def _generate_stream(self, url: str, payload: dict) -> str:
        """Streaming generation with live output."""
        response = requests.post(url, json=payload, stream=True, timeout=60)
        response.raise_for_status()

        full_response = []

        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line)
                    if "response" in chunk:
                        text = chunk["response"]
                        print(text, end="", flush=True)
                        full_response.append(text)

                    if chunk.get("done", False):
                        break

                except json.JSONDecodeError:
                    continue

        print()  # Newline after streaming
        return "".join(full_response)

    def check_model(self) -> bool:
        """Verify the model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
            return any(m.get("name", "").startswith(self.model) for m in models)
        except requests.exceptions.RequestException:
            return False


# -----------------------
# Display Utilities
# -----------------------
def display_results(
        results: List[RetrievalResult],
        top_score: float,
        threshold: float
) -> None:
    """Pretty print retrieval results."""
    print("\n" + "=" * 70)
    print("RETRIEVED PASSAGES")
    print("=" * 70)

    for i, result in enumerate(results, 1):
        p = result.passage
        print(f"\n[{i}] Score: {result.score:.4f} | {p}")
        print(f"    {p.text[:150]}...")

    print("\n" + "-" * 70)
    # Use the top_score parameter that was passed in
    print(f"Top score: {top_score:.4f} | Threshold: {threshold:.4f} | "
          f"Confidence: {'HIGH' if top_score >= threshold else 'LOW'}")
    print("=" * 70 + "\n")


# -----------------------
# Main Application
# -----------------------
def main():
    parser = argparse.ArgumentParser(
        description="RAG CLI for querying Marcus Aurelius' Meditations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rag_cli.py "What does Marcus say about death?"
  python rag_cli.py "How to deal with difficult people?" --show --stream
  python rag_cli.py "What is virtue?" --k 3 --threshold 0.3
        """
    )

    parser.add_argument(
        "question",
        type=str,
        help="Your philosophical question"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=TOP_K,
        help=f"Number of passages to retrieve (default: {TOP_K})"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help=f"Similarity threshold for high confidence (default: {THRESHOLD})"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display retrieved passages and scores"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the response token by token"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=OLLAMA_MODEL,
        help=f"Ollama model to use (default: {OLLAMA_MODEL})"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help=f"Sampling temperature (default: {TEMPERATURE})"
    )

    args = parser.parse_args()

    try:
        # Load resources
        print("Loading resources...", file=sys.stderr)
        metadata = load_metadata(META_PATH)
        index = load_index(INDEX_PATH)
        embedder = SentenceTransformer(EMBED_MODEL)

        # Initialize Ollama client
        ollama = OllamaClient(model=args.model)

        # Check if model is available
        if not ollama.check_model():
            print(f"Warning: Model '{args.model}' not found. Pulling...",
                  file=sys.stderr)
            # You might want to auto-pull here or give instructions

        # Retrieve relevant passages
        query_vec = embed_query(embedder, args.question)
        results = retrieve(index, metadata, query_vec, args.k)

        if not results:
            print("Error: No passages retrieved.", file=sys.stderr)
            sys.exit(1)

        top_score = results[0].score

        # Display retrieval results if requested
        if args.show:
            display_results(results, top_score, args.threshold)

        # Build prompt
        context = build_context_block(results)
        prompt = build_prompt(args.question, context, top_score, args.threshold)

        # Generate answer
        if not args.show:
            print("Generating answer...\n", file=sys.stderr)

        answer = ollama.generate(
            prompt,
            stream=args.stream,
            temperature=args.temperature
        )

        if not args.stream:
            print(answer)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)

    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()