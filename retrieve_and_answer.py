import streamlit as st
import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
from datetime import datetime

# --- Prompt Template Loader ---
def load_prompt_templates(path="prompt_templates.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

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

# --- Streamlit UI ---
st.set_page_config(page_title="Spiritual Assistant", layout="centered")
st.title("🌟 Spiritual Assistant: Laws of Creation")

# For session logging
if "session_log" not in st.session_state:
    st.session_state.session_log = []

question = st.text_area("What is your spiritual question?", height=80)
tone = st.selectbox(
    "Choose a response tone:",
    list(PROMPT_TEMPLATES.keys()),
    index=0
)

if st.button("Ask the Assistant"):
    if question.strip():
        with st.spinner("Reflecting..."):
            answer = ask(question, tone)
        st.markdown("**Assistant’s Response:**")
        st.success(answer)

        # Resonance rating
        st.markdown("How resonant was this answer?")
        resonance = st.radio(
            "Resonance rating",
            ["👍 Highly Resonant", "👌 Useful", "😐 Neutral", "👎 Not Resonant"],
            horizontal=True,
            key=f"resonance_{len(st.session_state.session_log)}"
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
    if st.button("Export Session (.txt)"):
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
