import os
import pickle
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import re
import json

@dataclass
class BookSummary:
    """Generic data structure for storing book summary information."""
    title: str
    author: str = "Unknown Author"
    summary: str = ""
    key_themes: List[str] = field(default_factory=list)
    important_characters: Dict[str, str] = field(default_factory=dict)
    structure: Dict[str, str] = field(default_factory=dict)
    genre: str = ""
    settings: List[str] = field(default_factory=list)
    # Add any additional fields that might be useful
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BookSummary':
        """Create BookSummary from dictionary."""
        return cls(**data)


def generate_book_summary(
    reading_chunks: List[str], 
    book_metadata: Dict[str, Any] = None,
    llm_generator: Optional[Any] = None,
    llm_generate_func: Optional[callable] = None
) -> BookSummary:
    """Generate a book summary using sample chunks and optional LLM.
    
    Args:
        reading_chunks: List of text chunks from the book
        book_metadata: Dictionary with book metadata (title, author, etc.)
        llm_generator: Optional LLM instance to use for summary generation
        llm_generate_func: Function to call LLM if custom implementation needed
        
    Returns:
        BookSummary object with extracted/generated information
    """
    # Use provided metadata or extract from content
    metadata = book_metadata or {}
    title = metadata.get("title", "Unknown Title")
    author = metadata.get("author", "Unknown Author")
    
    # Extract sample chunks for analysis
    samples = extract_sample_chunks(reading_chunks)
    
    # If an LLM generator is provided, use it to create the summary
    if llm_generator and llm_generate_func:
        return generate_summary_with_llm(
            samples, title, author, llm_generator, llm_generate_func
        )
    
    # Otherwise, use basic text analysis to extract information
    return generate_basic_summary(samples, title, author)


def extract_sample_chunks(reading_chunks: List[str], sample_count: int = 10) -> List[str]:
    """Extract representative sample chunks from the book for analysis.
    
    Args:
        reading_chunks: List of text chunks from the book
        sample_count: Number of samples to extract
        
    Returns:
        List of sample text chunks
    """
    chunk_count = len(reading_chunks)
    if chunk_count == 0:
        return []
    
    samples = []
    
    # Get beginning chunks (first 10%)
    beginning_idx = max(1, int(chunk_count * 0.1))
    beginning_chunks = reading_chunks[:beginning_idx]
    
    # Get middle chunks (middle 10%)
    middle_start = int(chunk_count * 0.45)
    middle_end = min(chunk_count - 1, int(chunk_count * 0.55))
    middle_chunks = reading_chunks[middle_start:middle_end]
    
    # Get end chunks (last 10%)
    end_idx = int(chunk_count * 0.9)
    end_chunks = reading_chunks[end_idx:]
    
    # Take samples from each section
    sections = [
        ("beginning", beginning_chunks),
        ("middle", middle_chunks),
        ("end", end_chunks)
    ]
    
    for section_name, chunk_list in sections:
        # Take up to 3 evenly spaced samples from each section
        if chunk_list:
            step = max(1, len(chunk_list) // 3)
            for i in range(0, len(chunk_list), step):
                if len(samples) < sample_count and i < len(chunk_list):
                    samples.append(chunk_list[i])
    
    return samples


def generate_basic_summary(samples: List[str], title: str, author: str) -> BookSummary:
    """Generate a basic book summary using text analysis without an LLM.
    
    Args:
        samples: List of sample text chunks
        title: Book title
        author: Book author
        
    Returns:
        BookSummary object with extracted information
    """
    # Combine samples for analysis
    all_text = " ".join(samples).lower()
    
    # Extract potential character names (capitalized words)
    character_pattern = r'\b([A-Z][a-z]+)\b'
    potential_characters = {}
    
    for sample in samples:
        matches = re.findall(character_pattern, sample)
        for name in matches:
            if len(name) > 2 and name not in ['The', 'And', 'But', 'For', 'With']:
                # Count occurrences and find context
                if name not in potential_characters:
                    context_pattern = r'(.{0,50}' + re.escape(name) + r'.{0,50})'
                    context_match = re.search(context_pattern, sample)
                    context = context_match.group(1) if context_match else ""
                    potential_characters[name] = context
    
    # Sort by frequency in samples
    character_dict = {}
    for name in sorted(list(potential_characters.keys()), 
                      key=lambda x: all_text.count(x.lower()), 
                      reverse=True)[:5]:
        character_dict[name] = potential_characters[name]
    
    # Extract potential themes (common significant words)
    words = re.findall(r'\b[a-z]{5,}\b', all_text)
    word_counts = {}
    for word in words:
        if word not in ['which', 'there', 'their', 'about', 'would', 'could']:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Get top words as themes
    themes = [word for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:8]]
    
    # Create a basic summary from samples
    summary_text = f"This book appears to be titled '{title}' by {author}. "
    
    # Add a bit about the beginning
    if beginning_idx := min(1, len(samples) - 1):
        summary_text += f"It begins with: \"{samples[beginning_idx][:100]}...\". "
    
    # Add something about the themes
    if themes:
        summary_text += f"The text contains recurring themes related to {', '.join(themes[:3])}. "
    
    # Add something about characters
    if character_dict:
        top_character = list(character_dict.keys())[0]
        summary_text += f"A notable character appears to be {top_character}. "
    
    # Create structure information
    structure = {
        "beginning": "Introduction of the setting and characters",
        "middle": "Development of main conflicts and story arcs",
        "end": "Resolution of the main storylines"
    }
    
    # Create and return the summary object
    return BookSummary(
        title=title,
        author=author,
        summary=summary_text,
        key_themes=themes,
        important_characters=character_dict,
        structure=structure
    )


def generate_summary_with_llm(
    samples: List[str], 
    title: str, 
    author: str, 
    llm_generator: Any,
    llm_generate_func: callable
) -> BookSummary:
    """Generate a comprehensive book summary using an LLM.
    
    Args:
        samples: List of sample text chunks
        title: Book title
        author: Book author
        llm_generator: LLM instance to use
        llm_generate_func: Function to call the LLM
        
    Returns:
        BookSummary object with LLM-generated information
    """
    # Create the prompt for the LLM
    prompt = f"""
Based on the following excerpts from the book "{title}" by {author}, please create a comprehensive summary of the book. 
Include the following information:
1. A brief summary of the plot and main concepts
2. Key themes and ideas present in the text
3. Important characters and their roles (if fiction) or key concepts (if non-fiction)
4. The structure and flow of the book (beginning, middle, end)
5. The setting and atmosphere (if relevant)

EXCERPTS FROM THE BOOK:

"""
    
    # Add samples with position markers
    for i, sample in enumerate(samples):
        position = "BEGINNING" if i < len(samples) // 3 else "MIDDLE" if i < 2 * len(samples) // 3 else "END"
        prompt += f"\n{position} EXCERPT {i+1}:\n{sample[:500]}...\n"
    
    prompt += "\nPlease format your response as JSON with the following structure:"
    prompt += """
{
  "summary": "Overall book summary",
  "key_themes": ["Theme 1", "Theme 2", ...],
  "important_characters": {
    "Character Name": "Brief description",
    ...
  },
  "structure": {
    "beginning": "Description of beginning",
    "middle": "Description of middle",
    "end": "Description of end"
  },
  "settings": ["Setting 1", "Setting 2", ...],
  "genre": "Identified genre"
}
"""
    
    # Call the LLM using the provided function
    llm_response = llm_generate_func(llm_generator, prompt)
    
    # Try to extract JSON from the response
    try:
        # Find JSON block in response
        json_match = re.search(r'({[\s\S]*})', llm_response)
        if json_match:
            json_str = json_match.group(1)
            data = json.loads(json_str)
            
            # Create BookSummary from the JSON data
            summary = BookSummary(
                title=title,
                author=author,
                summary=data.get("summary", ""),
                key_themes=data.get("key_themes", []),
                important_characters=data.get("important_characters", {}),
                structure=data.get("structure", {}),
                genre=data.get("genre", ""),
                settings=data.get("settings", [])
            )
            return summary
    except (json.JSONDecodeError, AttributeError):
        # If JSON parsing fails, extract information manually
        pass
    
    # Fallback to basic summary if JSON parsing fails
    return generate_basic_summary(samples, title, author)


def save_book_summary(summary: BookSummary, file_path: str) -> None:
    """Save a BookSummary object to disk.
    
    Args:
        summary: BookSummary object to save
        file_path: Path to save the summary
    """
    with open(file_path, 'wb') as f:
        pickle.dump(summary, f)


def load_book_summary(file_path: str) -> Optional[BookSummary]:
    """Load a BookSummary object from disk.
    
    Args:
        file_path: Path to the saved summary
        
    Returns:
        BookSummary object or None if file doesn't exist
    """
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return None
