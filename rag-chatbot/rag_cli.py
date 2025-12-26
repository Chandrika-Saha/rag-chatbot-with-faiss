# rag_cli.py
# Project 2.1 — Local RAG with FAISS + TinyLlama (Ollama)
# Modes:
#  A) Grounded answer if similarity >= THRESHOLD
#  B) "Related reflection" fallback if similarity < THRESHOLD (still cites passages)

import argparse
import json
import subprocess
from dataclasses import dataclass
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# -----------------------
# Config
# -----------------------
INDEX_PATH = "meditations.index"
META_PATH = "metadata.json"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "tinyllama"

TOP_K = 5

# For normalized cosine similarity with IndexFlatIP:
# scores ~ [-1..1], typical "somewhat related" often ~0.25-0.45 depending on data.
# Start here and tune with 20 queries.
THRESHOLD = 0.33

MAX_CONTEXT_CHARS = 3500  # keep small for tiny models + speed


# -----------------------
# Data
# -----------------------
@dataclass
class Passage:
    pid: str
    book: str
    text: str

def load_metadata(path: str) -> List[Passage]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Expect raw is a list aligned with FAISS order.
    # Each entry should include at least: id, book, text
    return [Passage(pid=str(x["id"]), book=str(x["book"]), text=str(x["text"])) for x in raw]


# -----------------------
# Embedding + Retrieval
# -----------------------
def normalize(v: np.ndarray) -> np.ndarray:
    # v: (d,) or (n,d)
    if v.ndim == 1:
        denom = np.linalg.norm(v) + 1e-12
        return v / denom
    denom = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / denom

def embed_query(model: SentenceTransformer, q: str) -> np.ndarray:
    vec = model.encode([q], convert_to_numpy=True)[0].astype("float32")
    vec = normalize(vec)
    return vec

def retrieve(index, meta: List[Passage], qvec: np.ndarray, top_k: int) -> List[Tuple[float, Passage]]:
    # FAISS expects shape (n, d)
    D, I = index.search(qvec.reshape(1, -1), top_k)
    scores = D[0].tolist()
    idxs = I[0].tolist()

    results = []
    for s, i in zip(scores, idxs):
        if i == -1:
            continue
        results.append((float(s), meta[i]))
    return results


# -----------------------
# Prompting
# -----------------------
def build_context_block(results: List[Tuple[float, Passage]]) -> str:
    # Provide compact, citation-ready context
    lines = []
    for score, p in results:
        # Keep each passage short-ish
        txt = p.text.strip().replace("\n", " ")
        lines.append(f"- (Book {p.book}, id:{p.pid}) {txt}")
    ctx = "\n".join(lines)
    return ctx[:MAX_CONTEXT_CHARS]

def build_prompt(question: str, ctx: str, mode: str) -> str:
    # Mode: "grounded" or "reflection"
    if mode == "grounded":
        system = (
            "You are a retrieval-augmented assistant.\n"
            "Answer the question using ONLY the provided CONTEXT.\n"
            "If the answer is not supported by CONTEXT, say exactly:\n"
            "\"I cannot answer based on the provided passages.\"\n"
            "Cite supporting passages in the form (Book X, id:Y).\n"
            "Keep the answer to 2–4 sentences.\n"
        )
        instruction = (
            "Return only the answer text (no preamble).\n"
        )
    else:
        system = (
            "You are a reflective assistant grounded in the provided CONTEXT.\n"
            "The CONTEXT may not directly answer the question.\n"
            "Use it as a lens to offer a related reflection WITHOUT claiming it directly answers the question.\n"
            "Give 1–2 practical takeaways.\n"
            "Always cite the passage(s) you used in the form (Book X, id:Y).\n"
            "Keep it to 2–4 sentences.\n"
        )
        instruction = (
            "Start with: \"Related reflection:\" then give your response.\n"
        )

    prompt = (
        f"{system}\n"
        f"CONTEXT:\n{ctx}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"{instruction}"
    )
    return prompt


# -----------------------
# Ollama call
# -----------------------
def ollama_generate(model: str, prompt: str) -> str:
    # Uses: ollama run <model> -p <prompt>
    # This avoids needing extra Python deps.
    try:
        out = subprocess.check_output(
            ["ollama", "run", model, "-p", prompt],
            stderr=subprocess.STDOUT,
            text=True,
        )
        return out.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Ollama failed:\n{e.output}") from e


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str, help="User question to ask the RAG system.")
    parser.add_argument("--k", type=int, default=TOP_K, help="Top-k passages to retrieve.")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Similarity threshold for grounded vs reflection mode.")
    parser.add_argument("--show", action="store_true", help="Print retrieved passages + scores.")
    args = parser.parse_args()

    # Load components
    meta = load_metadata(META_PATH)
    index = faiss.read_index(INDEX_PATH)
    embedder = SentenceTransformer(EMBED_MODEL)

    # Retrieve
    qvec = embed_query(embedder, args.question)
    results = retrieve(index, meta, qvec, args.k)

    if not results:
        print("I cannot answer based on the provided passages.")
        return

    top_score = results[0][0]
    mode = "grounded" if top_score >= args.threshold else "reflection"

    if args.show:
        print("\n[Retrieval]")
        for s, p in results:
            print(f"  score={s:.3f}  (Book {p.book}, id:{p.pid}) {p.text[:90].strip()}...")
        print(f"\n[Mode] {mode} (top_score={top_score:.3f}, threshold={args.threshold:.3f})\n")

    ctx = build_context_block(results)
    prompt = build_prompt(args.question, ctx, mode)
    answer = ollama_generate(OLLAMA_MODEL, prompt)

    print(answer)


if __name__ == "__main__":
    main()
