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
load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-4"

def get_embedding(text):
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding

def pinecone_query(question, top_k=5):
    vector = get_embedding(question)
    results = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return results['matches']

def build_prompt(question, matches, tone="scriptural"):
    context = "\n\n".join([m['metadata']['text'] for m in matches])
    law_names = [m['metadata'].get('law', '') for m in matches if m['metadata'].get('law')]
    tone_instr = PROMPT_TEMPLATES[tone]
    law_clause = f"\nIf possible, reference or cite the following laws: {', '.join(set(law_names))}." if law_names else ""
    prompt = f"""
You are a sacred spiritual assistant. Respond to the user's question based on the content provided below, and always reflect the Laws of Creation framework.

Context:
{context}

Question:
{question}

Instructions:
{tone_instr}{law_clause}
If you cannot find an answer, state that you do not have information grounded in the provided context.
Do not invent information. Do not hallucinate beyond the source material.
"""
    return prompt.strip()

def ask(question, tone="scriptural", top_k=5):
    matches = pinecone_query(question, top_k)
    prompt = build_prompt(question, matches, tone)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()

# --- Sacred Mode Input Validation ---
def validate_sacred_input(text):
    # Common profanity patterns (simplified for example)
    profanity_patterns = [
        r'\b(fuck|shit|damn|hell|ass)\b',
        r'\b(omg|wtf|fml)\b',
        r'[!]{2,}',  # Multiple exclamation marks
        r'[?]{2,}',  # Multiple question marks
    ]
    
    # Check for sarcasm indicators
    sarcasm_indicators = [
        r'\b(yeah right|sure|whatever)\b',
        r'\b(duh|obviously|clearly)\b',
        r'[?]{2,}',  # Multiple question marks
    ]
    
    # Check for profanity
    for pattern in profanity_patterns:
        if re.search(pattern, text.lower()):
            return False, "This sacred assistant is reserved for spiritual refinement. Please reframe your question with sincerity."
    
    # Check for sarcasm
    for pattern in sarcasm_indicators:
        if re.search(pattern, text.lower()):
            return False, "This sacred assistant is reserved for spiritual refinement. Please reframe your question with sincerity."
    
    # Check for minimum length and meaningful content
    if len(text.strip()) < 5:
        return False, "Please provide a more detailed question to receive a meaningful response."
    
    return True, ""

# --- Streamlit UI ---
st.set_page_config(page_title="Spiritual Assistant", layout="centered")

# Custom CSS for better formatting
st.markdown("""
<style>
    .stTextArea textarea {
        font-size: 16px;
    }
    .response-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .user-question {
        font-weight: 500;
        color: #1f1f1f;
        margin-bottom: 10px;
    }
    .assistant-response {
        color: #1f1f1f;
        line-height: 1.6;
    }
    .resonance-section {
        margin-top: 15px;
        padding-top: 15px;
        border-top: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# For session logging and chat history
if "session_log" not in st.session_state:
    st.session_state.session_log = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "input_key" not in st.session_state:
    st.session_state.input_key = 0

# Main chat interface
st.title("Spiritual Assistant")
st.markdown("_A sacred space for spiritual inquiry and growth_")

# Display chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="user-question">
            <strong>You:</strong><br>
            {message["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="response-box">
            <div class="assistant-response">
                {message["content"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

# Input area
question = st.text_area("What is your spiritual question?", height=80, key=f"question_input_{st.session_state.input_key}")
tone = st.selectbox(
    "Choose a response tone:",
    list(PROMPT_TEMPLATES.keys()),
    index=0
)

if st.button("Ask the Assistant"):
    if question.strip():
        # Validate input through sacred mode
        is_valid, message = validate_sacred_input(question)
        
        if not is_valid:
            st.warning(message)
        else:
            # Add user question to chat history
            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })
            
            with st.spinner("Reflecting..."):
                answer = ask(question, tone)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })
            
            # Log session entry
            session_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "question": question,
                "tone": tone,
                "answer": answer,
                "resonance": "Not Rated"  # Default value
            }
            st.session_state.session_log.append(session_entry)
            
            # Increment input key to clear the input
            st.session_state.input_key += 1
            st.experimental_rerun()

# Session export
if st.session_state.session_log:
    st.markdown("---")
    if st.button("Export Session"):
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

st.markdown("---\n_Sacred content is sourced only from your embedded documents. The assistant will not invent or hallucinate information._")
