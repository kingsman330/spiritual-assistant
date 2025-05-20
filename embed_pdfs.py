import os
import re
import tiktoken
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))
pdf_folder = os.getenv("PDF_FOLDER", "./pdfs")

EMBED_MODEL = "text-embedding-ada-002"
CHUNK_SIZE = 300

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
    "Wow girl I really don’t know where to start -": {
        "type": "dialogue",
        "tone": "conversational",
        "dimension": "emotional"
    }
}

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

def chunk_text(text, chunk_size=CHUNK_SIZE):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]
    decoded_chunks = [enc.decode(chunk) for chunk in chunks]
    return decoded_chunks

def get_tag_from_filename(filename):
    for base_name, tags in PDF_TAGS.items():
        if filename.lower().startswith(base_name.lower()):
            return tags
    return {"source": filename}

def process_pdf(pdf_path, filename):
    try:
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += "\n" + text
        return full_text.strip()
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return ""

def embed_and_upsert(filename, text):
    tags = get_tag_from_filename(filename)
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        try:
            response = client.embeddings.create(model=EMBED_MODEL, input=chunk)
            vector = response.data[0].embedding
            meta = dict(tags)
            meta.update({
                "source_file": filename,
                "chunk_index": i,
                "text": chunk
            })
            index.upsert([{
                "id": f"{filename}_{i}",
                "values": vector,
                "metadata": meta
            }])
            print(f"Uploaded: {filename} chunk {i}")
        except Exception as e:
            print(f"Error embedding/uploading chunk {i} of {filename}: {e}")

def main():
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            print(f"Processing {filename}...")
            text = process_pdf(pdf_path, filename.replace(".pdf", ""))
            if text:
                cleaned_text = clean_text(text)
                embed_and_upsert(filename.replace(".pdf", ""), cleaned_text)

if __name__ == "__main__":
    main()
