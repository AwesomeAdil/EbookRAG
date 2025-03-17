import ebooklib
from ebooklib import epub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import re

def extract_text_from_epub(file_path):
    """Extracts and cleans text from an EPUB file, fixing spacing issues."""
    book = epub.read_epub(file_path)
    text = ""

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(item.get_content(), "html.parser")

            # Extract only meaningful text
            raw_text = soup.get_text(separator=" ", strip=True)  # Use space instead of \n

            # Remove excessive spaces and fix formatting issues
            cleaned_text = re.sub(r'\s+', ' ', raw_text).strip()

            # Append to book text
            text += cleaned_text + "\n\n"  # Add double newlines for paragraph separation

    return text


def find_last_chapter(chapters):
    """Finds the last valid chapter and returns its index."""
    chapter_titles = ["acknowledgments", "references", "about the author", "appendix", "summary", "other books"]
    
    for i, chapter in enumerate(chapters):
        chapter_lower = chapter.lower()
        if any(title in chapter_lower for title in chapter_titles):
            return i  # Stop at the first non-story chapter
    
    return len(chapters)  # If no unwanted chapters are found, include all


def clean_chunk_boundaries(chunks):
    """Clean up chunk boundaries to ensure proper sentence endings and beginnings."""
    cleaned_chunks = []
    
    for i, chunk in enumerate(chunks):
        # Remove leading periods that might have been duplicated from previous chunk
        chunk = re.sub(r'^\.+\s*', '', chunk)
        
        # Ensure chunks end with proper punctuation
        if i < len(chunks) - 1 and not re.search(r'[.!?]\s*$', chunk):
            # If chunk doesn't end with punctuation, try to end at the last complete sentence
            last_period = max(chunk.rfind('. '), chunk.rfind('! '), chunk.rfind('? '))
            if last_period != -1:
                # Move incomplete sentence to the next chunk
                next_part = chunk[last_period + 2:]
                chunks[i+1] = next_part + ' ' + chunks[i+1]
                chunk = chunk[:last_period + 1]  # Keep the period
        
        cleaned_chunks.append(chunk)
    
    return cleaned_chunks


def load_and_chunk_book(file_path):
    """Loads book and creates two different chunk sets: reading chunks and embedding chunks."""
    book_text = extract_text_from_epub(file_path)
    
    # Create larger chunks for reading (with paragraph preservation)
    reading_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # Larger size for better reading experience
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]  # Prioritize paragraph breaks
    )
    reading_chunks = reading_splitter.split_text(book_text)
    reading_chunks = clean_chunk_boundaries(reading_chunks)
    
    # Create smaller chunks for embeddings/RAG
    embedding_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Smaller for more precise information retrieval
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    embedding_chunks = embedding_splitter.split_text(book_text)
    embedding_chunks = clean_chunk_boundaries(embedding_chunks)
    
    print(f"Reading Chunks: {len(reading_chunks)}")
    print(f"Embedding Chunks: {len(embedding_chunks)}")
    
    return reading_chunks, embedding_chunks
