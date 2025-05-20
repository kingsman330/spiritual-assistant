import os
import re
import tiktoken
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
import unicodedata
import string
from datetime import datetime
import time
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))
pdf_folder = os.getenv("PDF_FOLDER", "./pdfs")

# Constants
EMBED_MODEL = "text-embedding-ada-002"
CHUNK_SIZE = 500  # Reduced from 1000 to 500 tokens
OVERLAP = 100     # Reduced from 200 to 100 tokens
MAX_TOKENS = 8000 # Maximum tokens for the model

PDF_TAGS = {
    "Aetheral Expansion Thoughts and Discovery collection 1": {
        "type": "exploration",
        "theme": "aetheral",
        "dimension": "celestial",
        "tone": "philosophical"
    },
    "Ascension Theory": {
        "type": "doctrine",
        "topic": "ascension",
        "tone": "teaching",
        "dimension": "soul"
    },
    "In the vast tapestry of existence, the journey of creation and refinement is a process that began long before we were aware of our place in the universe": {
        "type": "reflection",
        "theme": "creation",
        "tone": "contemplative",
        "dimension": "origin"
    },
    "Laws of Creation Framework - thoughts": {
        "type": "law_matrix",
        "law": "multiple",
        "tone": "scriptural",
        "secondary_tone": "teaching"
    },
    "Master Compilation Bring the World His Truth": {
        "type": "doctrine",
        "theme": "truth",
        "dimension": "mortal",
        "tone": "prophetic"
    },
    "Matt the Trauma baby": {
        "type": "testimony",
        "theme": "trauma",
        "tone": "personal",
        "dimension": "mortal"
    },
    "Our freedom to Choose, the law of Choice and the refinement of Truths": {
        "type": "law",
        "law": "Law of Choice",
        "tone": "explanatory",
        "dimension": "moral"
    },
    "received my reward,": {
        "type": "reflection",
        "theme": "reward",
        "tone": "personal",
        "dimension": "celestial"
    },
    "Wow girl I really don't know where to start -": {
        "type": "dialogue",
        "tone": "conversational",
        "dimension": "emotional"
    }
}

def to_ascii_id(text):
    # Normalize to NFKD and encode to ASCII, ignore errors (removes accents, smart quotes, etc.)
    ascii_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode()
    # Replace spaces with underscores
    ascii_text = ascii_text.replace(' ', '_')
    # Remove all non-alphanumeric and non-underscore/dash chars
    allowed = string.ascii_letters + string.digits + "_-"
    ascii_text = ''.join(c if c in allowed else '' for c in ascii_text)
    return ascii_text

def clean_text(raw_text):
    # Remove literal '/n' and '\\n' strings
    cleaned = re.sub(r'/n', ' ', raw_text)
    cleaned = cleaned.replace('\\n', '\n')

    # Replace 3+ newlines with just two
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    # Remove page numbers (e.g., "Page 12", "12", "- 12 -", etc.)
    cleaned = re.sub(r'\b(page\s*)?\d+\b', '', cleaned, flags=re.IGNORECASE)

    # Remove common headers/footers or generic repetitive lines
    cleaned = re.sub(r'(Table of Contents|Continued on next page)', '', cleaned, flags=re.IGNORECASE)

    # Remove lines with only whitespace or special characters
    cleaned = re.sub(r'^[\s\W_]+$', '', cleaned, flags=re.MULTILINE)

    # Remove multiple blank lines again (final sweep)
    cleaned = re.sub(r'\n\s*\n+', '\n\n', cleaned)

    # Strip leading/trailing whitespace
    cleaned = cleaned.strip()
    return cleaned

def count_tokens(text):
    """Count the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-4")
    return len(encoding.encode(text))

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Split text into overlapping chunks of approximately chunk_size tokens."""
    encoding = tiktoken.encoding_for_model("gpt-4")
    tokens = encoding.encode(text)
    chunks = []
    
    i = 0
    while i < len(tokens):
        # Get chunk of tokens
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = encoding.decode(chunk_tokens)
        
        # Verify token count
        token_count = len(chunk_tokens)
        if token_count > MAX_TOKENS:
            print(f"Warning: Chunk exceeds maximum token limit ({token_count} > {MAX_TOKENS})")
            # If chunk is too large, try to split at a sentence boundary
            sentences = chunk_text.split('. ')
            reduced_chunk = '. '.join(sentences[:len(sentences)//2])
            chunk_text = reduced_chunk
            token_count = count_tokens(chunk_text)
        
        chunks.append(chunk_text)
        
        # Move forward, accounting for overlap
        i += (chunk_size - overlap)
    
    return chunks

def get_tag_from_filename(filename):
    for base_name, tags in PDF_TAGS.items():
        if filename.lower().startswith(base_name.lower()):
            return tags
    return {"source": filename}

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def get_embedding(text):
    """Get embedding for a text string."""
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def upsert_to_pinecone(vectors):
    """Upload vectors to Pinecone with retry logic."""
    try:
        index.upsert(vectors=vectors)
    except Exception as e:
        print(f"Error uploading to Pinecone: {str(e)}")
        raise

def process_pdf(pdf_path, metadata=None):
    """Process a PDF file and upload its chunks to Pinecone."""
    try:
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            print(f"Warning: No text extracted from {pdf_path}")
            return
        
        # Clean the text
        text = clean_text(text)
        
        # Split into chunks
        chunks = chunk_text(text)
        if not chunks:
            print(f"Warning: No chunks created from {pdf_path}")
            return
        
        print(f"Created {len(chunks)} chunks from {pdf_path}")
        
        # Prepare metadata
        base_metadata = {
            "source": os.path.basename(pdf_path),
            "type": "doctrine",  # or "law" or "glossary" based on content
            "timestamp": datetime.utcnow().isoformat(),
        }
        if metadata:
            base_metadata.update(metadata)
        
        # Process chunks in smaller batches
        batch_size = 25  # Reduced from 50 to 25
        vectors_batch = []
        
        for i, chunk in enumerate(chunks):
            try:
                # Verify token count before embedding
                token_count = count_tokens(chunk)
                if token_count > MAX_TOKENS:
                    print(f"Warning: Chunk {i} exceeds maximum token limit ({token_count} > {MAX_TOKENS})")
                    continue
                
                # Get embedding
                embedding = get_embedding(chunk)
                
                # Prepare chunk metadata
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    "chunk_index": i,
                    "text": chunk,
                    "token_count": token_count
                })
                
                # Add to batch
                vectors_batch.append({
                    "id": f"{to_ascii_id(os.path.basename(pdf_path))}-{i}",
                    "values": embedding,
                    "metadata": chunk_metadata
                })
                
                # Upload batch if it reaches batch_size
                if len(vectors_batch) >= batch_size:
                    upsert_to_pinecone(vectors_batch)
                    print(f"Uploaded batch of {len(vectors_batch)} chunks from {os.path.basename(pdf_path)}")
                    vectors_batch = []
                    time.sleep(1)  # Add delay between batches
                
            except Exception as e:
                print(f"Error processing chunk {i} from {pdf_path}: {str(e)}")
                continue
        
        # Upload remaining vectors
        if vectors_batch:
            upsert_to_pinecone(vectors_batch)
            print(f"Uploaded final batch of {len(vectors_batch)} chunks from {os.path.basename(pdf_path)}")
        
        print(f"Completed processing {pdf_path}")
        
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")

def process_directory(directory_path, metadata=None):
    """Process all PDFs in a directory."""
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return
    
    pdf_files = [f for f in os.listdir(directory_path) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in {directory_path}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for filename in pdf_files:
        pdf_path = os.path.join(directory_path, filename)
        print(f"\nProcessing {filename}...")
        process_pdf(pdf_path, metadata)
        time.sleep(2)  # Add delay between files

def main():
    # Example usage
    pdf_directory = os.getenv("PDF_FOLDER", "pdfs")
    
    # Create directory if it doesn't exist
    if not os.path.exists(pdf_directory):
        os.makedirs(pdf_directory)
        print(f"Created directory: {pdf_directory}")
        print("Please add your PDF files to this directory and run the script again.")
        return
    
    try:
        # Process all PDFs in the directory
        process_directory(pdf_directory)
        print("\nProcessing complete!")
    except Exception as e:
        print(f"Error in main process: {str(e)}")

if __name__ == "__main__":
    main()
