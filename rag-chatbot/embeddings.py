import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def create_faiss_index(json_file='quotes_flat.json',
                       model_name='all-MiniLM-L6-v2',
                       index_file='meditations.index',
                       metadata_file='meditations_metadata.json'):
    """
    Create FAISS index from flattened meditations data

    Args:
        json_file: Input JSON file with flattened structure
        model_name: Sentence transformer model to use
        index_file: Output FAISS index file
        metadata_file: Output metadata file
    """

    # Load the flattened data
    print("📖 Loading meditations data...")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"✓ Loaded {len(data)} paragraphs")

    # Extract texts for embedding
    texts = [item['text'] for item in data]

    # Load the embedding model
    print(f"🤖 Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)

    # Generate embeddings
    print("🔄 Generating embeddings (this may take a moment)...")
    embeddings = model.encode(texts,
                              show_progress_bar=True,
                              convert_to_numpy=True,
                              normalize_embeddings=True,
                              batch_size=32)

    print(f"✓ Generated embeddings with shape: {embeddings.shape}")
    print(f"  - Number of vectors: {embeddings.shape[0]}")
    print(f"  - Embedding dimension: {embeddings.shape[1]}")

    # Create FAISS index
    print("🏗️  Creating FAISS index...")
    dimension = embeddings.shape[1]

    # Using IndexFlatIP for exact cosine similarity search
    index = faiss.IndexFlatIP(dimension)

    # Add embeddings to index
    index.add(embeddings.astype('float32'))

    print(f"✓ Added {index.ntotal} vectors to FAISS index")

    # Save the index
    print(f"💾 Saving FAISS index to {index_file}...")
    faiss.write_index(index, index_file)

    # Save metadata (the original data structure is perfect as-is)
    print(f"💾 Saving metadata to {metadata_file}...")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("✅ Done! Index created successfully.")
    print(f"   📁 FAISS index: {index_file}")
    print(f"   📁 Metadata: {metadata_file}")

    return index, model, data

# Main execution
if __name__ == '__main__':
    # Option 1: Create new index
    print("Creating FAISS index from flattened JSON...\n")
    index, model, metadata = create_faiss_index(
        json_file='quotes_flat.json',
        model_name='all-MiniLM-L6-v2',
        index_file='meditations.index',
        metadata_file='meditations_metadata.json'
    )
