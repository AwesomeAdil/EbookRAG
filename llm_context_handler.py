import pickle
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

# Import the BookSummary class from book_summary_generator
from book_summary_generator import BookSummary, generate_book_summary, load_book_summary, save_book_summary

class LLMContextHandler:
    def __init__(
        self,
        reading_chunks_path: str,
        vector_db_path: str,
        book_summary: Optional[BookSummary] = None,
        book_summary_path: Optional[str] = None,
        book_metadata: Dict[str, Any] = None
    ):
        """Initialize the LLM Context Handler with book chunks and summary.
        
        Args:
            reading_chunks_path: Path to the reading chunks pickle file
            vector_db_path: Path to the vector database pickle file
            book_summary: Optional pre-generated BookSummary object
            book_summary_path: Optional path to load/save book summary
            book_metadata: Optional dictionary with book metadata
        """
        # Load reading chunks and vector DB
        with open(reading_chunks_path, "rb") as f:
            self.reading_chunks = pickle.load(f)
        
        with open(vector_db_path, "rb") as f:
            self.vector_db = pickle.load(f)
        
        self.book_metadata = book_metadata or {}
        
        # Load or generate book summary
        if book_summary:
            # Use provided summary
            self.book_summary = book_summary
        elif book_summary_path and os.path.exists(book_summary_path):
            # Load from file
            self.book_summary = load_book_summary(book_summary_path)
            print(f"Loaded book summary from {book_summary_path}")
        else:
            # Generate new summary
            self.book_summary = generate_book_summary(
                reading_chunks=self.reading_chunks,
                book_metadata=self.book_metadata
            )
            print("Generated new book summary")
            
            # Save summary if path provided
            if book_summary_path:
                save_book_summary(self.book_summary, book_summary_path)
                print(f"Saved book summary to {book_summary_path}")
    
    def format_context_for_llm(
        self, 
        query: str, 
        relevant_chunks: List[Dict[str, Any]],
        include_summary: bool = True,
        max_tokens: int = 4000
    ) -> str:
        """Format the context from relevant chunks and book summary for the LLM.
        
        Args:
            query: The user's query
            relevant_chunks: List of relevant chunks with their content and metadata
            include_summary: Whether to include book summary in context
            max_tokens: Maximum token count for context (approximate)
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Add book summary if requested
        if include_summary and self.book_summary:
            summary_context = f"""
BOOK INFORMATION:
Title: {self.book_summary.title}
Author: {self.book_summary.author}

SUMMARY:
{self.book_summary.summary}

KEY THEMES:
{', '.join(self.book_summary.key_themes[:5])}

IMPORTANT CHARACTERS:
"""
            # Add characters with descriptions
            for char_name, char_desc in self.book_summary.important_characters.items():
                summary_context += f"- {char_name}: {char_desc}\n"
            
            context_parts.append(summary_context)
        
        # Add relevant chunks (with ranking info)
        chunks_context = "RELEVANT SECTIONS FROM THE BOOK:\n\n"
        
        for i, chunk in enumerate(relevant_chunks):
            # Extract content and any available metadata
            content = chunk.get("content", "")
            relevance = chunk.get("relevance_score", 0.0)
            position = chunk.get("position", "unknown")
            
            # Add chunk with metadata
            chunk_text = f"[SECTION {i+1} | Relevance: {relevance:.2f} | Position: {position}]\n{content}\n\n"
            chunks_context += chunk_text
        
        context_parts.append(chunks_context)
        
        # Add query-focused instructions
        instruction = f"""
QUESTION: {query}

Based on the information provided from the book, please answer this question.
If the answer isn't fully contained in the provided sections, use your understanding
of the book's summary and themes to provide the best possible answer.
Indicate clearly when you're inferring information rather than directly quoting.
"""
        context_parts.append(instruction)
        
        # Combine context parts, respecting max token limit (approximate)
        combined_context = ""
        for part in context_parts:
            # Very rough estimation: 1 token ≈ 4 characters
            if len(combined_context) + len(part) < max_tokens * 4:
                combined_context += part
            else:
                # Truncate last part to fit
                remaining_chars = max_tokens * 4 - len(combined_context)
                if remaining_chars > 100:  # Only add if we can include a meaningful amount
                    combined_context += part[:remaining_chars] + "\n[Context truncated due to length]"
                break
        
        return combined_context
    
    def prepare_query_response(
        self,
        query: str,
        search_results: List,
        reranking_info: Optional[Dict] = None,
        include_summary: bool = True
    ) -> Dict[str, Any]:
        """Prepare the context and format for LLM query response.
        
        Args:
            query: User query string
            search_results: List of document objects from vector search
            reranking_info: Optional dict with reranking scores and strategy
            include_summary: Whether to include book summary
            
        Returns:
            Dictionary with formatted context and query info
        """
        # Process search results into a consistent format
        relevant_chunks = []
        
        for i, doc in enumerate(search_results):
            # Extract content and metadata
            content = doc.page_content
            metadata = doc.metadata
            
            # Get reading position info
            read_idx = metadata.get("read_index", i)
            total_chunks = len(self.reading_chunks)
            position_ratio = read_idx / total_chunks if total_chunks > 0 else 0
            
            # Convert position ratio to a descriptive string
            if position_ratio < 0.25:
                position = "beginning"
            elif position_ratio < 0.75:
                position = "middle"
            else:
                position = "end"
            
            # Get relevance score if available in reranking info
            relevance_score = 1.0  # Default score
            if reranking_info and "scores" in reranking_info:
                if i < len(reranking_info["scores"]):
                    relevance_score = reranking_info["scores"][i]
            
            # Add to relevant chunks
            relevant_chunks.append({
                "content": content,
                "metadata": metadata,
                "relevance_score": relevance_score,
                "position": position,
                "read_index": read_idx
            })
        
        # Format context for LLM
        formatted_context = self.format_context_for_llm(
            query=query,
            relevant_chunks=relevant_chunks,
            include_summary=include_summary
        )
        
        # Prepare response package
        response_package = {
            "query": query,
            "formatted_context": formatted_context,
            "relevant_chunks": relevant_chunks,
            "reranking_info": reranking_info or {},
            "book_summary": self.book_summary.__dict__ if include_summary else None
        }
        
        return response_package
    
    def update_book_summary(self, updated_summary: BookSummary, summary_path: str = None):
        """Update the book summary with new information.
        
        Args:
            updated_summary: New BookSummary object
            summary_path: Optional path to save updated summary
        """
        self.book_summary = updated_summary
        
        # Save updated summary if path provided
        if summary_path:
            save_book_summary(self.book_summary, summary_path)
            print(f"Updated book summary saved to {summary_path}")
