import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def load_faiss_index(index_file='meditations.index',
                     metadata_file='meditations_metadata.json',
                     model_name='all-MiniLM-L6-v2'):
    """
    Load existing FAISS index and metadata

    Args:
        index_file: FAISS index file path
        metadata_file: Metadata JSON file path
        model_name: Sentence transformer model name

    Returns:
        index, model, metadata tuple
    """
    print("📂 Loading FAISS index...")
    index = faiss.read_index(index_file)

    print("📂 Loading metadata...")
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    print("🤖 Loading model...")
    model = SentenceTransformer(model_name)

    print(f"✓ Loaded index with {index.ntotal} vectors")
    print(f"✓ Loaded {len(metadata)} metadata entries")

    return index, model, metadata


def search_meditations(query,
                       model,
                       index,
                       metadata,
                       top_k=5):
    """
    Search for similar paragraphs using FAISS

    Args:
        query: Search query text
        model: SentenceTransformer model
        index: FAISS index
        metadata: List of metadata dicts
        top_k: Number of results to return

    Returns:
        List of result dictionaries
    """
    print(f"\n🔍 Searching for: '{query}'")
    print(f"{'=' * 60}")

    # Generate embedding for query
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    # Search in FAISS index
    scores, indices = index.search(query_embedding.astype('float32'), top_k)

    # Collect and display results
    results = []

    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
        item = metadata[idx]

        result = {
            'rank': rank,
            'score': float(score),
            'id': item['id'],
            'book': item['book'],
            'text': item['text']
        }
        results.append(result)

        print(f"\n[{rank}] 📊 Score: {score:.4f}")
        print(f"    🆔 ID: {item['id']}")
        print(f"    📖 Book: {item['book']}")
        print(f"    📝 Text: {item['text'][:150]}{'...' if len(item['text']) > 150 else ''}")

    print(f"\n{'=' * 60}\n")
    return results


def batch_search(queries, model, index, metadata, top_k=5):
    """
    Search multiple queries at once

    Args:
        queries: List of query strings
        model: SentenceTransformer model
        index: FAISS index
        metadata: List of metadata dicts
        top_k: Number of results per query

    Returns:
        Dictionary mapping queries to their results
    """
    all_results = {}

    for query in queries:
        results = search_meditations(query, model, index, metadata, top_k)
        all_results[query] = results

    return all_results


def get_embedding_stats(index_file='meditations.index'):
    """Get statistics about the FAISS index"""
    index = faiss.read_index(index_file)

    print("📊 FAISS Index Statistics:")
    print(f"   Total vectors: {index.ntotal}")
    print(f"   Dimension: {index.d}")
    print(f"   Index type: {type(index).__name__}")

    return {
        'total_vectors': index.ntotal,
        'dimension': index.d,
        'index_type': type(index).__name__
    }


# Main execution
if __name__ == '__main__':
    index, model, metadata = load_faiss_index('meditations.index',
                                              'meditations_metadata.json',
                                              'all-MiniLM-L6-v2')
    # Multiple searches
    queries = [
        "What is the nature of death?",
        "The importance of virtue",
        "Finding inner peace"
    ]

    # query_categories = {
    #     "Emotional Regulation": [
    #         "I'm constantly worrying about things I can't control. How do I stop?",
    #         "I'm anxious about what people think of me. How can I care less about others' opinions?",
    #         "How do I stop overthinking everything?"
    #     ],
    #
    #     "Anger Management": [
    #         "Someone at work keeps undermining me. How do I not get angry?",
    #         "I get so frustrated when things don't go my way. How can I be more patient?",
    #         "How do I respond to people who are rude or disrespectful?"
    #     ],
    #
    #     "Purpose & Meaning": [
    #         "I feel like my work is meaningless. How do I find purpose?",
    #         "I'm procrastinating on important tasks. How do I motivate myself?",
    #         "Should I quit my job to pursue my passion?"
    #     ],
    #
    #     "Mortality & Loss": [
    #         "I'm afraid of dying. How do I make peace with mortality?",
    #         "Someone close to me died and I can't stop grieving. How do I cope?",
    #         "I feel like I'm wasting my life. How do I make the most of my time?"
    #     ],
    #
    #     "Self-Development": [
    #         "How do I become a better person?",
    #         "I keep making the same mistakes. How do I change?",
    #         "How do I stay humble when I'm successful?"
    #     ],
    #
    #     "Resilience": [
    #         "Everything is going wrong in my life right now. How do I keep going?",
    #         "I failed at something important. How do I deal with failure?",
    #         "How do I stay calm when everything feels chaotic?"
    #     ],
    #
    #     "Relationships": [
    #         "How do I deal with toxic family members?",
    #         "Why should I be kind to people who don't deserve it?"
    #     ]
    # }

    all_queries = [
        "I'm constantly worrying about things I can't control. How do I stop?",
        "I'm anxious about what people think of me. How can I care less about others' opinions?",
        "How do I stop overthinking everything?",
        "Someone at work keeps undermining me. How do I not get angry?",
        "I get so frustrated when things don't go my way. How can I be more patient?",
        "How do I respond to people who are rude or disrespectful?",
        "I feel like my work is meaningless. How do I find purpose?",
        "I'm procrastinating on important tasks. How do I motivate myself?",
        "Should I quit my job to pursue my passion?",
        "I'm afraid of dying. How do I make peace with mortality?",
        "Someone close to me died and I can't stop grieving. How do I cope?",
        "I feel like I'm wasting my life. How do I make the most of my time?",
        "How do I become a better person?",
        "I keep making the same mistakes. How do I change?",
        "How do I stay humble when I'm successful?",
        "Everything is going wrong in my life right now. How do I keep going?",
        "I failed at something important. How do I deal with failure?",
        "How do I stay calm when everything feels chaotic?",
        "How do I deal with toxic family members?",
        "Why should I be kind to people who don't deserve it?"
    ]

    print(f"Total queries: {len(all_queries)}")


    all_results = batch_search(all_queries, model, index, metadata, top_k=3)

    # Get index stats
    stats = get_embedding_stats('meditations.index')