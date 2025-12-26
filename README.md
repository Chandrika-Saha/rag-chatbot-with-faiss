# 🏛️ Marcus Aurelius AI Companion

A Retrieval-Augmented Generation (RAG) chatbot that brings the wisdom of Marcus Aurelius' *Meditations* to life. Ask philosophical questions and receive Stoic advice backed by actual passages from the ancient text.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Ollama](https://img.shields.io/badge/Ollama-000000?logo=ollama&logoColor=white)](https://ollama.ai)

---

## ✨ Features

- 🔍 **Semantic Search** - Uses FAISS and sentence transformers for intelligent passage retrieval
- 🤖 **Local LLM** - Powered by Ollama (completely private, runs on your machine)
- 📚 **Source Citations** - Every answer includes references to specific passages
- 🎯 **Confidence Scoring** - Cosine similarity scores show retrieval quality
- 💬 **Dual Interfaces** - CLI for quick queries, Streamlit for rich UI
- ⚙️ **Highly Configurable** - Adjust retrieval parameters, model settings, and more

---

## 🎬 Demo
### Main Chat Interface
![Marcus Aurelius AI Chat](Assets/First_Look.png)


### Streamlit Interface
![Streamlit Demo](docs/streamlit_demo.png)

### CLI Interface
```bash
$ python rag_cli.py "How should I deal with difficult people?" --show

[Retrieval]
  score=0.403  (Book ELEVEN, id:p4) Have I done something for the general interest?...
  score=0.322  (Book SIX, id:p7) Take pleasure in one thing and rest in it...
  score=0.295  (Book THREE, id:p1) We ought to consider not only that our life...

[Confidence] top_score=0.403, threshold=0.250

When you wake up in the morning, tell yourself: the people I deal with today will be 
meddling, ungrateful, arrogant, dishonest, jealous and surly - (Book TWO, id:p1).
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai) installed and running
- At least 4GB RAM for embedding model + LLM

### Installation

1. Clone the repository and navigate to the project directory
2. Install required Python packages from requirements.txt
3. Pull an Ollama model (e.g., llama3.2:1b or llama3.2:3b)
4. Start the Ollama server if not already running
5. Ensure you have `meditations.index` and `meditations_metadata.json` in the project root

### Running the Application

**Streamlit Interface (Recommended):**

Run the Streamlit app and open your browser to http://localhost:8501

**Command Line Interface:**

Use the CLI for quick queries with various options like --show, --k, --threshold, etc.

---

