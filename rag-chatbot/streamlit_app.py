# streamlit_app.py
# Streamlit UI for Marcus Aurelius Meditations RAG Chatbot

import streamlit as st
import sys
from pathlib import Path
from typing import List, Tuple
import time

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Import from your rag_cli.py
from rag_cli import (
    load_metadata,
    load_index,
    embed_query,
    retrieve,
    build_context_block,
    OllamaClient,
    RetrievalResult,
    Passage,
    INDEX_PATH,
    META_PATH,
    EMBED_MODEL,
    OLLAMA_MODEL,
    TOP_K,
    THRESHOLD,
    TEMPERATURE,
    MAX_TOKENS
)

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="Marcus Aurelius - AI Companion",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------
# Custom CSS
# -----------------------
st.markdown("""
<style>
    /* Main chat container */
    .main-chat {
        max-width: 900px;
        margin: 0 auto;
    }

    /* User message bubble */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 18px 18px 5px 18px;
        margin: 10px 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }

    /* Assistant message bubble */
    .assistant-message {
        background: #f0f2f6;
        color: #1f1f1f;
        padding: 15px 20px;
        border-radius: 18px 18px 18px 5px;
        margin: 10px 0;
        max-width: 80%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }

    /* Citation badge */
    .citation {
        display: inline-block;
        background: #e8eaf6;
        color: #3f51b5;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: 500;
        margin-left: 8px;
    }

    /* Confidence indicator */
    .confidence-high {
        color: #4caf50;
        font-weight: bold;
    }

    .confidence-low {
        color: #ff9800;
        font-weight: bold;
    }

    /* Passage card */
    .passage-card {
        background: white;
        border-left: 4px solid #667eea;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    .passage-header {
        font-weight: 600;
        color: #667eea;
        margin-bottom: 8px;
    }

    .passage-text {
        color: #424242;
        line-height: 1.6;
    }

    /* Score badge */
    .score-badge {
        background: #667eea;
        color: white;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.8em;
        margin-left: 8px;
    }

    /* Header styling */
    .app-header {
        text-align: center;
        padding: 20px 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }

    /* Sidebar styling */
    .sidebar-section {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Initialize Session State
# -----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "metadata" not in st.session_state:
    st.session_state.metadata = None

if "index" not in st.session_state:
    st.session_state.index = None

if "embedder" not in st.session_state:
    st.session_state.embedder = None

if "ollama_client" not in st.session_state:
    st.session_state.ollama_client = None

if "initialized" not in st.session_state:
    st.session_state.initialized = False

# NEW: Separate variable for pending questions
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# -----------------------
# Helper Functions
# -----------------------
@st.cache_resource
def initialize_resources():
    """Load all resources once and cache them."""
    try:
        metadata = load_metadata(META_PATH)
        index = load_index(INDEX_PATH)
        embedder = SentenceTransformer(EMBED_MODEL)
        return metadata, index, embedder, None
    except Exception as e:
        return None, None, None, str(e)


def build_llm_prompt(question: str, context: str) -> str:
    """Build prompt for LLM."""
    return f"""Answer the Question based on the Context Passages. 
You MUST provide ONLY a one-sentence answer. 
Base your answer on the Context passage.
Use direct sentence from the paragraph whenever appropriate.
The answer MUST NOT CONTAIN ANYTHING about analyzing or considering passages.
If the question has weak connection to the paragraph, draw from its core concept and respond in the given answer format in ONE sentence.
You MUST NOT include any pleasantries. 

Context Passages: {context}

Question: {question}

Answer Format: ```[Direct advice statement]- (Book X, id:Y).```"""


def extract_citation(answer: str) -> Tuple[str, str]:
    """Extract citation from answer."""
    import re
    citation_pattern = r'$Book\s+(\w+),\s+id:(\w+)$'
    match = re.search(citation_pattern, answer)
    if match:
        return match.group(1), match.group(2)
    return None, None


def format_passage_card(result: RetrievalResult, index: int) -> str:
    """Format a passage as an HTML card."""
    p = result.passage
    return f"""
    <div class="passage-card">
        <div class="passage-header">
            [{index}] Book {p.book}, Passage {p.pid}
            <span class="score-badge">{result.score:.3f}</span>
        </div>
        <div class="passage-text">{p.text}</div>
    </div>
    """


def process_question(question: str, top_k: int, threshold: float,
                     temperature: float, max_tokens: int, model_name: str):
    """Process a question and generate a response."""
    try:
        with st.spinner("Consulting the Meditations..."):
            query_vec = embed_query(st.session_state.embedder, question)
            results = retrieve(
                st.session_state.index,
                st.session_state.metadata,
                query_vec,
                top_k
            )

            if not results:
                st.error("No relevant passages found.")
                return

            top_score = results[0].score
            context = build_context_block(results)

            # Build prompt and generate
            prompt = build_llm_prompt(question, context)

            # Update Ollama client with current settings
            st.session_state.ollama_client.model = model_name

            answer = st.session_state.ollama_client.generate(
                prompt,
                stream=False,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Add assistant message to chat
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "passages": results,
                "confidence": {
                    "score": top_score,
                    "is_high": top_score >= threshold,
                    "level": "HIGH" if top_score >= threshold else "LOW"
                }
            })

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


# -----------------------
# Sidebar Configuration
# -----------------------
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # Model settings
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**Model Settings**")
    model_name = st.text_input("Ollama Model", value=OLLAMA_MODEL)
    temperature = st.slider("Temperature", 0.0, 1.0, TEMPERATURE, 0.1)
    max_tokens = st.slider("Max Tokens", 50, 500, MAX_TOKENS, 50)
    st.markdown('</div>', unsafe_allow_html=True)

    # Retrieval settings
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**Retrieval Settings**")
    top_k = st.slider("Top K Passages", 1, 10, TOP_K)
    threshold = st.slider("Confidence Threshold", 0.0, 1.0, THRESHOLD, 0.05)
    st.markdown('</div>', unsafe_allow_html=True)

    # Display settings
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**Display Settings**")
    show_passages = st.checkbox("Show Retrieved Passages", value=True)
    show_scores = st.checkbox("Show Confidence Scores", value=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Actions
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # About
    st.markdown("---")
    st.markdown("""
    ### About
    This AI companion helps you explore **Marcus Aurelius' Meditations** 
    using RAG (Retrieval Augmented Generation).

    Ask philosophical questions and receive wisdom from the Stoic emperor.
    """)

# -----------------------
# Main App
# -----------------------

# Header
st.markdown("""
<div class="app-header">
    <h1>🏛️ Marcus Aurelius - AI Companion</h1>
    <p>Explore Stoic wisdom through the Meditations</p>
</div>
""", unsafe_allow_html=True)

# Initialize resources
if not st.session_state.initialized:
    with st.spinner("Loading Marcus Aurelius' wisdom..."):
        metadata, index, embedder, error = initialize_resources()

        if error:
            st.error(f"❌ Failed to load resources: {error}")
            st.stop()

        st.session_state.metadata = metadata
        st.session_state.index = index
        st.session_state.embedder = embedder

        # Initialize Ollama client
        try:
            st.session_state.ollama_client = OllamaClient(model=model_name)
            st.session_state.initialized = True
        except Exception as e:
            st.error(f"❌ Failed to connect to Ollama: {e}")
            st.info("Make sure Ollama is running: `ollama serve`")
            st.stop()

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="user-message">
            <strong>You:</strong><br>{message["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-message">
            <strong>Marcus:</strong><br>{message["content"]}
        </div>
        """, unsafe_allow_html=True)

        # Show retrieved passages if available
        if show_passages and "passages" in message:
            with st.expander("📚 View Retrieved Passages", expanded=False):
                for i, result in enumerate(message["passages"], 1):
                    st.markdown(format_passage_card(result, i), unsafe_allow_html=True)

        # Show confidence score
        if show_scores and "confidence" in message:
            conf_class = "confidence-high" if message["confidence"]["is_high"] else "confidence-low"
            st.markdown(f"""
            <div style="text-align: right; margin-top: 5px;">
                <small>
                    Confidence: <span class="{conf_class}">{message["confidence"]["score"]:.3f}</span>
                    ({message["confidence"]["level"]})
                </small>
            </div>
            """, unsafe_allow_html=True)

# Chat input
st.markdown("---")

# Check if there's a pending question from example buttons
if st.session_state.pending_question:
    user_question = st.session_state.pending_question
    st.session_state.pending_question = None  # Clear it

    # Add user message to chat
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    # Process the question
    process_question(user_question, top_k, threshold, temperature, max_tokens, model_name)
    st.rerun()

# Regular input
col1, col2 = st.columns([6, 1])

with col1:
    user_question = st.text_input(
        "Ask Marcus Aurelius...",
        placeholder="e.g., How should I deal with difficult people?",
        label_visibility="collapsed",
        key="user_input"
    )

with col2:
    send_button = st.button("Send 📤", use_container_width=True)

# Process user input from text box
if send_button and user_question:
    # Add user message to chat
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    # Process the question
    process_question(user_question, top_k, threshold, temperature, max_tokens, model_name)
    st.rerun()

# Example questions (shown when chat is empty)
if len(st.session_state.messages) == 0:
    st.markdown("### 💭 Example Questions")

    example_col1, example_col2 = st.columns(2)

    with example_col1:
        if st.button("🤔 How should I deal with difficult people?", use_container_width=True):
            st.session_state.pending_question = "How should I deal with difficult people?"
            st.rerun()

        if st.button("⏰ What does Marcus say about using time wisely?", use_container_width=True):
            st.session_state.pending_question = "What does Marcus say about using time wisely?"
            st.rerun()

    with example_col2:
        if st.button("😌 How can I find inner peace?", use_container_width=True):
            st.session_state.pending_question = "How can I find inner peace?"
            st.rerun()

        if st.button("💪 What is the Stoic view on adversity?", use_container_width=True):
            st.session_state.pending_question = "What is the Stoic view on adversity?"
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>Powered by RAG + Ollama | Data: Marcus Aurelius' Meditations</p>
</div>
""", unsafe_allow_html=True)
# # streamlit_app.py
# # Streamlit UI for Marcus Aurelius Meditations RAG Chatbot
#
# import streamlit as st
# import sys
# from pathlib import Path
# from typing import List, Tuple
# import time
#
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
#
# # Import from your rag_cli.py
# from rag_cli import (
#     load_metadata,
#     load_index,
#     embed_query,
#     retrieve,
#     build_context_block,
#     OllamaClient,
#     RetrievalResult,
#     Passage,
#     INDEX_PATH,
#     META_PATH,
#     EMBED_MODEL,
#     OLLAMA_MODEL,
#     TOP_K,
#     THRESHOLD,
#     TEMPERATURE,
#     MAX_TOKENS
# )
#
# # -----------------------
# # Page Config
# # -----------------------
# st.set_page_config(
#     page_title="Marcus Aurelius - AI Companion",
#     page_icon="🏛️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )
#
# # -----------------------
# # Custom CSS
# # -----------------------
# st.markdown("""
# <style>
#     /* Main chat container */
#     .main-chat {
#         max-width: 900px;
#         margin: 0 auto;
#     }
#
#     /* User message bubble */
#     .user-message {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         padding: 15px 20px;
#         border-radius: 18px 18px 5px 18px;
#         margin: 10px 0;
#         max-width: 80%;
#         margin-left: auto;
#         box-shadow: 0 2px 5px rgba(0,0,0,0.1);
#     }
#
#     /* Assistant message bubble */
#     .assistant-message {
#         background: #f0f2f6;
#         color: #1f1f1f;
#         padding: 15px 20px;
#         border-radius: 18px 18px 18px 5px;
#         margin: 10px 0;
#         max-width: 80%;
#         box-shadow: 0 2px 5px rgba(0,0,0,0.1);
#     }
#
#     /* Citation badge */
#     .citation {
#         display: inline-block;
#         background: #e8eaf6;
#         color: #3f51b5;
#         padding: 3px 10px;
#         border-radius: 12px;
#         font-size: 0.85em;
#         font-weight: 500;
#         margin-left: 8px;
#     }
#
#     /* Confidence indicator */
#     .confidence-high {
#         color: #4caf50;
#         font-weight: bold;
#     }
#
#     .confidence-low {
#         color: #ff9800;
#         font-weight: bold;
#     }
#
#     /* Passage card */
#     .passage-card {
#         background: white;
#         border-left: 4px solid #667eea;
#         padding: 12px 16px;
#         margin: 8px 0;
#         border-radius: 4px;
#         box-shadow: 0 1px 3px rgba(0,0,0,0.1);
#     }
#
#     .passage-header {
#         font-weight: 600;
#         color: #667eea;
#         margin-bottom: 8px;
#     }
#
#     .passage-text {
#         color: #424242;
#         line-height: 1.6;
#     }
#
#     /* Score badge */
#     .score-badge {
#         background: #667eea;
#         color: white;
#         padding: 2px 8px;
#         border-radius: 10px;
#         font-size: 0.8em;
#         margin-left: 8px;
#     }
#
#     /* Header styling */
#     .app-header {
#         text-align: center;
#         padding: 20px 0;
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         border-radius: 10px;
#         margin-bottom: 30px;
#     }
#
#     /* Sidebar styling */
#     .sidebar-section {
#         background: #f8f9fa;
#         padding: 15px;
#         border-radius: 8px;
#         margin-bottom: 15px;
#     }
# </style>
# """, unsafe_allow_html=True)
#
# # -----------------------
# # Initialize Session State
# # -----------------------
# if "messages" not in st.session_state:
#     st.session_state.messages = []
#
# if "metadata" not in st.session_state:
#     st.session_state.metadata = None
#
# if "index" not in st.session_state:
#     st.session_state.index = None
#
# if "embedder" not in st.session_state:
#     st.session_state.embedder = None
#
# if "ollama_client" not in st.session_state:
#     st.session_state.ollama_client = None
#
# if "initialized" not in st.session_state:
#     st.session_state.initialized = False
#
#
# # -----------------------
# # Helper Functions
# # -----------------------
# @st.cache_resource
# def initialize_resources():
#     """Load all resources once and cache them."""
#     try:
#         metadata = load_metadata(META_PATH)
#         index = load_index(INDEX_PATH)
#         embedder = SentenceTransformer(EMBED_MODEL)
#         return metadata, index, embedder, None
#     except Exception as e:
#         return None, None, None, str(e)
#
#
# def build_llm_prompt(question: str, context: str) -> str:
#     """Build prompt for LLM."""
#     return f"""Answer the Question based on the Context Passages.
# You MUST provide ONLY a one-sentence answer.
# Base your answer on the Context passage.
# Use direct sentence from the paragraph whenever appropriate.
# The answer MUST NOT CONTAIN ANYTHING about analyzing or considering passages.
# If the question has weak connection to the paragraph, draw from its core concept and respond in the given answer format in ONE sentence.
# You MUST NOT include any pleasantries.
#
# Context Passages: {context}
#
# Question: {question}
#
# Answer Format: ```[Direct advice statement]- (Book X, id:Y).```"""
#
#
# def extract_citation(answer: str) -> Tuple[str, str]:
#     """Extract citation from answer."""
#     import re
#     citation_pattern = r'$Book\s+(\w+),\s+id:(\w+)$'
#     match = re.search(citation_pattern, answer)
#     if match:
#         return match.group(1), match.group(2)
#     return None, None
#
#
# def format_passage_card(result: RetrievalResult, index: int) -> str:
#     """Format a passage as an HTML card."""
#     p = result.passage
#     return f"""
#     <div class="passage-card">
#         <div class="passage-header">
#             [{index}] Book {p.book}, Passage {p.pid}
#             <span class="score-badge">{result.score:.3f}</span>
#         </div>
#         <div class="passage-text">{p.text}</div>
#     </div>
#     """
#
#
# # -----------------------
# # Sidebar Configuration
# # -----------------------
# with st.sidebar:
#     st.markdown("### ⚙️ Configuration")
#
#     # Model settings
#     st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
#     st.markdown("**Model Settings**")
#     model_name = st.text_input("Ollama Model", value=OLLAMA_MODEL)
#     temperature = st.slider("Temperature", 0.0, 1.0, TEMPERATURE, 0.1)
#     max_tokens = st.slider("Max Tokens", 50, 500, MAX_TOKENS, 50)
#     st.markdown('</div>', unsafe_allow_html=True)
#
#     # Retrieval settings
#     st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
#     st.markdown("**Retrieval Settings**")
#     top_k = st.slider("Top K Passages", 1, 10, TOP_K)
#     threshold = st.slider("Confidence Threshold", 0.0, 1.0, THRESHOLD, 0.05)
#     st.markdown('</div>', unsafe_allow_html=True)
#
#     # Display settings
#     st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
#     st.markdown("**Display Settings**")
#     show_passages = st.checkbox("Show Retrieved Passages", value=True)
#     show_scores = st.checkbox("Show Confidence Scores", value=True)
#     st.markdown('</div>', unsafe_allow_html=True)
#
#     # Actions
#     st.markdown("---")
#     if st.button("🗑️ Clear Chat History", use_container_width=True):
#         st.session_state.messages = []
#         st.rerun()
#
#     # About
#     st.markdown("---")
#     st.markdown("""
#     ### About
#     This AI companion helps you explore **Marcus Aurelius' Meditations**
#     using RAG (Retrieval Augmented Generation).
#
#     Ask philosophical questions and receive wisdom from the Stoic emperor.
#     """)
#
# # -----------------------
# # Main App
# # -----------------------
#
# # Header
# st.markdown("""
# <div class="app-header">
#     <h1>🏛️ Marcus Aurelius - AI Companion</h1>
#     <p>Explore Stoic wisdom through the Meditations</p>
# </div>
# """, unsafe_allow_html=True)
#
# # Initialize resources
# if not st.session_state.initialized:
#     with st.spinner("Loading Marcus Aurelius' wisdom..."):
#         metadata, index, embedder, error = initialize_resources()
#
#         if error:
#             st.error(f"❌ Failed to load resources: {error}")
#             st.stop()
#
#         st.session_state.metadata = metadata
#         st.session_state.index = index
#         st.session_state.embedder = embedder
#
#         # Initialize Ollama client
#         try:
#             st.session_state.ollama_client = OllamaClient(model=model_name)
#             st.session_state.initialized = True
#         except Exception as e:
#             st.error(f"❌ Failed to connect to Ollama: {e}")
#             st.info("Make sure Ollama is running: `ollama serve`")
#             st.stop()
#
# # Display chat messages
# for message in st.session_state.messages:
#     if message["role"] == "user":
#         st.markdown(f"""
#         <div class="user-message">
#             <strong>You:</strong><br>{message["content"]}
#         </div>
#         """, unsafe_allow_html=True)
#     else:
#         st.markdown(f"""
#         <div class="assistant-message">
#             <strong>Marcus:</strong><br>{message["content"]}
#         </div>
#         """, unsafe_allow_html=True)
#
#         # Show retrieved passages if available
#         if show_passages and "passages" in message:
#             with st.expander("📚 View Retrieved Passages", expanded=False):
#                 for i, result in enumerate(message["passages"], 1):
#                     st.markdown(format_passage_card(result, i), unsafe_allow_html=True)
#
#         # Show confidence score
#         if show_scores and "confidence" in message:
#             conf_class = "confidence-high" if message["confidence"]["is_high"] else "confidence-low"
#             st.markdown(f"""
#             <div style="text-align: right; margin-top: 5px;">
#                 <small>
#                     Confidence: <span class="{conf_class}">{message["confidence"]["score"]:.3f}</span>
#                     ({message["confidence"]["level"]})
#                 </small>
#             </div>
#             """, unsafe_allow_html=True)
#
# # Chat input
# st.markdown("---")
# col1, col2 = st.columns([6, 1])
#
# with col1:
#     user_question = st.text_input(
#         "Ask Marcus Aurelius...",
#         placeholder="e.g., How should I deal with difficult people?",
#         label_visibility="collapsed",
#         key="user_input"
#     )
#
# with col2:
#     send_button = st.button("Send 📤", use_container_width=True)
#
# # Process user input
# if send_button and user_question:
#     # Add user message to chat
#     st.session_state.messages.append({
#         "role": "user",
#         "content": user_question
#     })
#
#     # Retrieve relevant passages
#     try:
#         with st.spinner("Consulting the Meditations..."):
#             query_vec = embed_query(st.session_state.embedder, user_question)
#             results = retrieve(
#                 st.session_state.index,
#                 st.session_state.metadata,
#                 query_vec,
#                 top_k
#             )
#
#             if not results:
#                 st.error("No relevant passages found.")
#                 st.stop()
#
#             top_score = results[0].score
#             context = build_context_block(results)
#
#             # Build prompt and generate
#             prompt = build_llm_prompt(user_question, context)
#
#             # Update Ollama client with current settings
#             st.session_state.ollama_client.model = model_name
#
#             answer = st.session_state.ollama_client.generate(
#                 prompt,
#                 stream=False,
#                 temperature=temperature,
#                 max_tokens=max_tokens
#             )
#
#             # Add assistant message to chat
#             st.session_state.messages.append({
#                 "role": "assistant",
#                 "content": answer,
#                 "passages": results,
#                 "confidence": {
#                     "score": top_score,
#                     "is_high": top_score >= threshold,
#                     "level": "HIGH" if top_score >= threshold else "LOW"
#                 }
#             })
#
#             # Rerun to display new messages
#             st.rerun()
#
#     except Exception as e:
#         st.error(f"❌ Error: {str(e)}")
#
# # Example questions (shown when chat is empty)
# if len(st.session_state.messages) == 0:
#     st.markdown("### 💭 Example Questions")
#
#     example_col1, example_col2 = st.columns(2)
#
#     with example_col1:
#         if st.button("🤔 How should I deal with difficult people?", use_container_width=True):
#             st.session_state.user_input = "How should I deal with difficult people?"
#             st.rerun()
#
#         if st.button("⏰ What does Marcus say about using time wisely?", use_container_width=True):
#             st.session_state.user_input = "What does Marcus say about using time wisely?"
#             st.rerun()
#
#     with example_col2:
#         if st.button("😌 How can I find inner peace?", use_container_width=True):
#             st.session_state.user_input = "How can I find inner peace?"
#             st.rerun()
#
#         if st.button("💪 What is the Stoic view on adversity?", use_container_width=True):
#             st.session_state.user_input = "What is the Stoic view on adversity?"
#             st.rerun()
#
# # Footer
# st.markdown("---")
# st.markdown("""
# <div style="text-align: center; color: #666; font-size: 0.9em;">
#     <p>Powered by RAG + Ollama | Data: Marcus Aurelius' Meditations</p>
# </div>
# """, unsafe_allow_html=True)