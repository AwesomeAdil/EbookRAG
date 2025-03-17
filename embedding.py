import faiss
import pickle
import os
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from book_load import load_and_chunk_book
from tqdm import tqdm  # For progress bars

def create_chunk_mapping(embedding_chunks, reading_chunks):
    """
    Creates a mapping from embedding chunks to reading chunks based on text similarity.
    Returns a dictionary where keys are embedding chunk indices and values are reading chunk indices.
    """
    print("Creating chunk mapping...")
    chunk_mapping = {}
    
    # Use tqdm for a progress bar
    for emb_idx, emb_chunk in tqdm(enumerate(embedding_chunks), total=len(embedding_chunks)):
        # First try exact substring matching
        for read_idx, read_chunk in enumerate(reading_chunks):
            if emb_chunk in read_chunk:
                chunk_mapping[emb_idx] = read_idx
                break
        
        # If not found exactly, use text similarity
        if emb_idx not in chunk_mapping:
            best_match = -1
            best_score = 0
            
            # Split into words for comparison
            emb_words = set(emb_chunk.lower().split())
            
            for read_idx, read_chunk in enumerate(reading_chunks):
                read_words = set(read_chunk.lower().split())
                
                # Calculate Jaccard similarity (intersection over union)
                intersection = len(emb_words & read_words)
                union = len(emb_words | read_words)
                
                # Avoid division by zero
                if union > 0:
                    score = intersection / union
                    
                    # Boost score if there's significant content overlap
                    content_overlap = sum(1 for word in emb_words if word in read_words) / len(emb_words) if emb_words else 0
                    score = 0.3 * score + 0.7 * content_overlap
                    
                    if score > best_score:
                        best_score = score
                        best_match = read_idx
            
            # Only map if we found a reasonable match
            if best_match >= 0:
                chunk_mapping[emb_idx] = best_match
            else:
                # Fallback: map to the middle of the book if no good match
                chunk_mapping[emb_idx] = len(reading_chunks) // 2
    
    return chunk_mapping

def main():
    # Configuration
    file_path = "The Scarlet Letter Images.epub"  # Replace with actual EPUB path
    output_dir = "./"  # Directory to save output files
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing book: {file_path}")
    
    # Load and chunk book with different sizes for reading and embeddings
    reading_chunks, embedding_chunks = load_and_chunk_book(file_path)
    
    print(f"Created {len(reading_chunks)} reading chunks")
    print(f"Created {len(embedding_chunks)} embedding chunks")
    
    # Save reading chunks for app.py
    reading_chunks_path = os.path.join(output_dir, "reading_chunks.pkl")
    with open(reading_chunks_path, "wb") as f:
        pickle.dump(reading_chunks, f)
    print(f"Reading chunks saved to {reading_chunks_path}")
    
    # Create and save mapping between embedding chunks and reading chunks
    chunk_mapping = create_chunk_mapping(embedding_chunks, reading_chunks)
    
    mapping_path = os.path.join(output_dir, "chunk_mapping.pkl")
    with open(mapping_path, "wb") as f:
        pickle.dump(chunk_mapping, f)
    print(f"Chunk mapping saved to {mapping_path}")
    
    # Initialize embeddings
    print(f"Initializing embeddings with model: {model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    # Create FAISS vector store with metadata
    print("Creating FAISS vector store...")
    texts = []
    metadatas = []
    
    for i, chunk in enumerate(embedding_chunks):
        texts.append(chunk)
        
        # Include the reading chunk index in metadata if available
        read_idx = chunk_mapping.get(i, i)  # Default to same index if not mapped
        metadatas.append({
            "emb_index": i,
            "read_index": read_idx,
            "text_preview": chunk[:100] + "..."  # Preview for debugging
        })
    
    # Create the vector store
    vector_db = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    
    # Save FAISS vector database
    db_path = os.path.join(output_dir, "vector_db.pkl")
    with open(db_path, "wb") as f:
        pickle.dump(vector_db, f)
    print(f"FAISS vector store saved to {db_path}")
    
    # Create empty feedback file if it doesn't exist
    feedback_path = os.path.join(output_dir, "feedback.pkl")
    if not os.path.exists(feedback_path):
        with open(feedback_path, "wb") as f:
            pickle.dump({}, f)
        print(f"Empty feedback file created at {feedback_path}")
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
