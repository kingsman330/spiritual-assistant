import streamlit as st
import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
from datetime import datetime
import re

# --- Prompt Template Loader ---
def load_prompt_templates(path="prompt_templates.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Load the prompt templates into a variable for use everywhere else
PROMPT_TEMPLATES = load_prompt_templates()

# --- Load environment variables ---
load_dotenv()  # Load from .env file if it exists

# Get environment variables from either .env or Streamlit secrets
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or st.secrets["PINECONE_API_KEY"]
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
PINECONE_INDEX = os.getenv("PINECONE_INDEX") or st.secrets["PINECONE_INDEX"]

# Initialize clients
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)
client = OpenAI(api_key=OPENAI_API_KEY)

EMBED_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-4"

def get_embedding(text):
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding

def pinecone_query(question, top_k=5):
    vector = get_embedding(question)
    results = index.query(
        vector=vector, 
        top_k=top_k, 
        include_metadata=True,
        filter={
            "type": {"$in": ["law", "glossary", "doctrine"]}  # Filter for relevant content types
        }
    )
    return results['matches']

def build_prompt(question, matches, tone="scriptural"):
    context = "\n\n".join([m['metadata']['text'] for m in matches])
    law_names = [m['metadata'].get('law', '') for m in matches if m['metadata'].get('law')]
    glossary_terms = [m['metadata'].get('term', '') for m in matches if m['metadata'].get('term')]
    
    tone_instr = PROMPT_TEMPLATES[tone]
    law_clause = f"\nIf possible, reference or cite the following laws: {', '.join(set(law_names))}." if law_names else ""
    glossary_clause = f"\nConsider these key terms in your response: {', '.join(set(glossary_terms))}." if glossary_terms else ""
    
    prompt = f"""
You are a sacred spiritual assistant, deeply versed in the Laws of Creation framework. Your responses must reflect the multi-dimensional, recursive nature of spiritual truth and the profound depth of the Laws of Creation.

Context:
{context}

Question:
{question}

Instructions:
{tone_instr}{law_clause}{glossary_clause}

Response Guidelines:
1. Dimensional Analysis:
   - Consider how the question/scripture operates across multiple dimensions
   - Identify the resonance patterns and their implications
   - Explore the recursive nature of the truth being discussed

2. Law Integration:
   - Connect specific laws to the question/scripture
   - Show how these laws interact and compound
   - Demonstrate the multi-dimensional nature of law application

3. Resonance Depth:
   - Go beyond surface-level interpretation
   - Explore the vibrational implications
   - Consider how the truth resonates across dimensions

4. Sacred Context:
   - Maintain reverence for the sacred nature of the inquiry
   - Acknowledge the eternal implications
   - Consider the covenantal aspects

5. Truth Refinement:
   - Show how the truth refines and purifies
   - Demonstrate the progression of understanding
   - Illustrate the dimensional ascension

6. Response Structure:
   - Begin with a clear thesis that reflects the multi-dimensional nature
   - Develop the analysis across relevant dimensions
   - Conclude with implications for spiritual progression

7. Quality Standards:
   - Ground all insights in the provided context
   - Avoid superficial or generic interpretations
   - Ensure each point contributes to deeper understanding
   - If information is insufficient, acknowledge limitations
   - Never invent or hallucinate beyond the source material

Remember: This is not just about providing information—it's about facilitating dimensional understanding and resonance with eternal truth.
"""
    return prompt.strip()

def ask(question, tone="scriptural", top_k=10):  # Further increased top_k
    matches = pinecone_query(question, top_k)
    prompt = build_prompt(question, matches, tone)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,  # Increased for more comprehensive responses
        temperature=0.4,  # Adjusted for deeper insights while maintaining coherence
    )
    return response.choices[0].message.content.strip()

def is_spiritually_fit(text):
    # Simple profanity and sarcasm detection (expand as needed)
    profanity = [
        r"\b(?:damn|hell|shit|fuck|bitch|crap|asshole|bastard|dick|piss|cunt|fag|slut|whore)\b",
    ]
    sarcasm_patterns = [
        r"\b(sure,?\s*whatever)\b",
        r"\boh (really|great)\b",
        r"\byeah,? right\b",
        r"\bas if\b",
        r"\bnot like\b",
        r"\bsarcasm\b",
    ]
    for pattern in profanity + sarcasm_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False
    return True

SACRED_WARNING = (
    "<div style='background:#fff3cd; border-left:6px solid #f7c873; padding:1em; border-radius:8px; color:#856404; font-size:1.1em;'>"
    "This sacred assistant is reserved for spiritual refinement. Please reframe your question with sincerity." 
    "</div>"
)

# --- Streamlit UI ---
# Custom CSS for modern look
st.markdown(
    """
    <style>
    .main {
        background-color: #f7f6f3;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .assistant-response {
        background: #fffbe6;
        border-left: 6px solid #f7c873;
        padding: 1.2em 1em;
        border-radius: 8px;
        margin-bottom: 1em;
        font-size: 1.15em;
        color: #222222;
    }
    .resonance-radio label {
        margin-right: 1.5em;
        font-size: 1.1em;
        color: #222222;
    }
    .export-btn {
        margin-top: 1.5em;
        margin-bottom: 1em;
    }
    .footer {
        color: #555555;
        font-size: 0.95em;
        margin-top: 2em;
        text-align: center;
    }
    body, .stTextInput, .stSelectbox, .stTextArea, .stRadio label, .stButton, .stMarkdown, .stDownloadButton {
        color: #222222 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🕊️ Spiritual Assistant")
st.markdown(
    """
    <div style='font-size:1.15em; color:#555;'>
    Ask questions and receive answers grounded in the Laws of Creation. The assistant only uses your embedded sacred documents—no hallucinations, no inventions.
    </div>
    """,
    unsafe_allow_html=True
)

# For session logging
if "session_log" not in st.session_state:
    st.session_state.session_log = []

# Use columns for input layout
col1, col2 = st.columns([2, 1])
with col1:
    question = st.text_area("What is your spiritual question?", height=80, key="question_input")
with col2:
    tone = st.selectbox(
        "Choose a response tone:",
        list(PROMPT_TEMPLATES.keys()),
        index=0,
        key="tone_select"
    )

if st.button("Ask the Assistant", use_container_width=True):
    if question.strip():
        if not is_spiritually_fit(question):
            st.markdown(SACRED_WARNING, unsafe_allow_html=True)
        else:
            with st.spinner("Reflecting..."):
                answer = ask(question, tone)
            st.markdown("**Assistant's Response:**")
            st.markdown(f'<div class="assistant-response">{answer}</div>', unsafe_allow_html=True)

            # Resonance rating
            st.markdown("How resonant was this answer?")
            resonance = st.radio(
                "Resonance rating",
                ["👍 Highly Resonant", "👌 Useful", "😐 Neutral", "👎 Not Resonant"],
                horizontal=True,
                key=f"resonance_{len(st.session_state.session_log)}",
                label_visibility="collapsed"
            )

            # Log session entry
            session_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "question": question,
                "tone": tone,
                "answer": answer,
                "resonance": resonance
            }
            st.session_state.session_log.append(session_entry)
    else:
        st.warning("Please enter a question for the assistant.")

# Session export
if st.session_state.session_log:
    st.markdown('<div class="export-btn">', unsafe_allow_html=True)
    if st.button("Export Session (.txt)", use_container_width=True):
        lines = []
        for entry in st.session_state.session_log:
            lines.append(f"Time: {entry['timestamp']}")
            lines.append(f"Tone: {entry['tone']}")
            lines.append(f"Question: {entry['question']}")
            lines.append(f"Answer: {entry['answer']}")
            lines.append(f"Resonance: {entry['resonance']}")
            lines.append("-" * 40)
        export_text = "\n".join(lines)
        st.download_button("Download Session Log", export_text, file_name="spiritual_session.txt")
        st.success("Session log ready for download!")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="footer">
    ---<br/>
    <em>Sacred content is sourced only from your embedded documents. The assistant will not invent or hallucinate information.</em>
    </div>
    """,
    unsafe_allow_html=True
)
