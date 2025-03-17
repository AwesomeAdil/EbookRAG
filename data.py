import pickle
from reranker import FeedbackBasedReranker
def load_data():
    # Load FAISS vector database
    db_path = "vector_db.pkl"
    with open(db_path, "rb") as f:
        vector_db = pickle.load(f)

    # Load the reading chunks for displaying like an eBook
    reading_chunks_path = "reading_chunks.pkl"
    with open(reading_chunks_path, "rb") as f:
        reading_chunks = pickle.load(f)

    # Load the mapping from embedding chunks to reading chunks
    mapping_path = "chunk_mapping.pkl"
    try:
        with open(mapping_path, "rb") as f:
            chunk_mapping = pickle.load(f)
    except FileNotFoundError:
        chunk_mapping = {}

    # Load or initialize feedback storage
    feedback_path = "feedback_data.pkl"
    try:
        with open(feedback_path, "rb") as f:
            feedback_data = pickle.load(f)
    except FileNotFoundError:
        feedback_data = {}

    reranker = FeedbackBasedReranker("vector_db.pkl", "feedback_data.pkl")
    return vector_db, reading_chunks, chunk_mapping, feedback_data, reranker
