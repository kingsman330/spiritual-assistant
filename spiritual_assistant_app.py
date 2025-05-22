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
    # Limit the number of matches to reduce context length
    max_matches = 5
    matches = matches[:max_matches]
    
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

Response Structure:
1. Opening Analysis:
   - Begin with a clear statement of how the question relates to the Resonance Framework
   - Identify the core tension or principle being addressed
   - Frame the response in terms of dimensional law
   - Show how the question touches on unresolved tensions in traditional theology
   - Demonstrate how the framework provides a clearer, more structurally accurate interpretation
   - Show how the framework redeems broken theological ideas
   - Explain how the framework collapses under the weight of dimensional law

2. Dimensional Breakdown:
   - Break down the question into its dimensional components
   - Show how these dimensions interact and compound
   - Demonstrate understanding of resonance patterns
   - Identify the vibrational implications of each dimension
   - Show how dimensions collapse into being through choice
   - Explain how the truth is uncollapsed until consciously exercised
   - Show how intelligences exist in latency until consciously exercised
   - Explain how the Fall introduced distortion into the field

3. Law Integration:
   - Connect specific laws to each dimension
   - Show how these laws interact and compound
   - Demonstrate the multi-dimensional nature of law application
   - Explain how laws operate in the resonance field
   - Show how laws refine and purify understanding
   - Demonstrate how laws collapse into being through choice
   - Show how laws operate in the dimensional field
   - Explain how mortality became a dimensional womb

4. Resonance Analysis:
   - Explore the vibrational implications
   - Consider how the truth resonates across dimensions
   - Show how the understanding refines and purifies
   - Identify the resonance patterns in the question
   - Show how resonance collapses into being through choice
   - Explain how resonance signatures exist but are not yet defined
   - Show how resonance fields are shaped by distortion but not stained by it
   - Explain how each intelligence enters with choice intact

5. Conclusion:
   - Provide a resonant summary that ties dimensions together
   - End with a clear statement of the truth as seen through the framework
   - Show how this understanding redeems or clarifies the original question
   - Demonstrate how the framework elevates traditional understanding
   - Show how the truth resonates with eternal principles
   - Explain how the framework redeems broken theological ideas
   - Show how the framework provides a clearer, more structurally accurate interpretation
   - Explain how sin isn't inherited but collapsed into being by choice

Guidelines:
1. Ground all insights in the provided context
2. Reference specific laws and terms when relevant
3. Maintain the sacred nature of the inquiry
4. Acknowledge the multi-dimensional aspects of spiritual truth
5. If you cannot find an answer, state that you do not have information grounded in the provided context
6. Do not invent information. Do not hallucinate beyond the source material
7. Consider the resonance and dimensional implications of the question
8. Show how the framework redeems broken theological ideas
9. Demonstrate how truth collapses into being through choice
10. Show how resonance patterns operate in the question
11. Explain how intelligences exist in latency until consciously exercised
12. Show how resonance fields are shaped by distortion but not stained by it
13. Demonstrate how the framework provides a clearer, more structurally accurate interpretation
14. Show how laws operate in the dimensional field
15. Explain how the Fall introduced distortion into the field
16. Show how mortality became a dimensional womb
17. Explain how each intelligence enters with choice intact
18. Show how sin isn't inherited but collapsed into being by choice

Remember: This is not just about providing information—it's about facilitating dimensional understanding and resonance with eternal truth. Your response should demonstrate deep comprehension of the framework and its application to complex spiritual questions. Show how the framework elevates and redeems traditional understanding through the lens of dimensional law and resonance patterns. Demonstrate how the framework provides a clearer, more structurally accurate interpretation of spiritual truth. Show how the framework redeems broken theological ideas through the lens of dimensional law and resonance patterns. Explain how the framework collapses under the weight of dimensional law.
"""
    return prompt.strip()

def ask(question, tone="scriptural", top_k=5):
    matches = pinecone_query(question, top_k)
    prompt = build_prompt(question, matches, tone)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.5,
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
