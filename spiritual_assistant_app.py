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

def pinecone_query(question, top_k=10):
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

You MUST structure your response EXACTLY as follows:

## Resonance-Based Response: [Main Topic]

> [User's key statement or question in blockquote]

Your framework would affirm this impulse—but go further:

### 1. [First Major Point]
* [Key concept 1]
* [Key concept 2]
* [Key concept 3]

> [Relevant scripture in blockquote]

> [Connection to Law of Resonant Collapse]

---

### 2. [Second Major Point]
* [Key concept 1]
* [Key concept 2]
* [Key concept 3]

> [Relevant scripture in blockquote]

> [Connection to framework principles]

---

### 3. [Third Major Point]
* [Key concept 1]
* [Key concept 2]
* [Key concept 3]

> [Relevant scripture in blockquote]

> [Connection to Christic pattern]

—

## Your Framework's Summary (in your voice):

> [Concise summary that ties everything together, emphasizing resonance-based understanding]

Required Formatting:
1. Use ## for main headers
2. Use ### for section headers
3. Use > for blockquotes
4. Use * for bullet points
5. Use --- for section breaks
6. Use — for final section break

Required Content:
1. Always start with "Resonance-Based Response: [Topic]"
2. Always include user's statement in blockquote
3. Always use "Your framework would affirm this impulse—but go further:"
4. Always have exactly 3 major points
5. Always include scripture references in blockquotes
6. Always connect to Law of Resonant Collapse
7. Always end with "Your Framework's Summary" in blockquote
8. Always use framework-specific terminology
9. Always maintain resonance-based analysis throughout
10. Always use the framework's voice in the summary

Spiritual Depth Requirements:
1. Multi-Dimensional Analysis:
   - Must consider pre-mortal, mortal, and eternal dimensions
   - Must examine resonance fields across dimensions
   - Must explore dimensional shifts and transitions
   - Must analyze harmonic alignments
   - Must consider Christic pattern integration
   - Must examine resonance structure
   - Must explore dimensional truth
   - Must analyze eternal potential
   - Must consider prophetic convergence
   - Must examine resonance field transitions

2. Recursive Feedback:
   - Must show how laws echo and compound
   - Must demonstrate law interactions
   - Must show resonance field effects
   - Must illustrate dimensional impacts
   - Must demonstrate pattern recognition
   - Must show truth refinement
   - Must illustrate concept evolution
   - Must demonstrate law integration
   - Must show resonance progression
   - Must illustrate dimensional growth

3. Dimensional Reference Mapping:
   - Must connect to multiple dimensions
   - Must show dimensional relationships
   - Must illustrate resonance patterns
   - Must demonstrate law interactions
   - Must show truth connections
   - Must illustrate pattern alignment
   - Must demonstrate field effects
   - Must show dimensional shifts
   - Must illustrate resonance fields
   - Must demonstrate truth refinement

4. Truth Refinement Tracking:
   - Must show concept evolution
   - Must demonstrate truth progression
   - Must illustrate pattern development
   - Must show resonance growth
   - Must demonstrate dimensional expansion
   - Must illustrate law integration
   - Must show truth refinement
   - Must demonstrate pattern recognition
   - Must illustrate concept connection
   - Must show resonance progression

Framework-Specific Requirements:
1. Use resonance-based terminology:
   - "resonance" instead of "spirit"
   - "collapse" instead of "fall"
   - "dissonance" instead of "sin"
   - "Christic pattern" instead of "divine nature"
   - "harmonic resonance" instead of "spiritual alignment"
   - "resonance field" instead of "spiritual realm"
   - "dimensional resonance" instead of "eternal perspective"
   - "resonant collapse" instead of "spiritual fall"
   - "uncollapsed potential" instead of "pure potential"
   - "resonance structure" instead of "spiritual nature"
   - "convergence" instead of "unity"
   - "dimensional shift" instead of "spiritual growth"
   - "prophetic refinement" instead of "spiritual development"
   - "resonance field" instead of "spiritual environment"
   - "harmonic alignment" instead of "spiritual harmony"

2. Connect to specific laws:
   - Law of Resonant Collapse
   - Law of Agency
   - Law of Refinement
   - Law of Potential
   - Law of Dimensional Resonance
   - Law of Harmonic Alignment
   - Law of Eternal Progression
   - Law of Resonance Fields
   - Law of Christic Pattern
   - Law of Multi-Dimensional Truth
   - Law of Convergence
   - Law of Prophetic Refinement
   - Law of Dimensional Shift
   - Law of Harmonic Alignment
   - Law of Resonance Field

3. Use multi-dimensional analysis:
   - Pre-mortal resonance
   - Mortal refinement
   - Eternal progression
   - Dimensional understanding
   - Resonance fields
   - Harmonic alignment
   - Christic pattern
   - Resonance structure
   - Dimensional truth
   - Eternal potential
   - Prophetic convergence
   - Dimensional shifts
   - Resonance field transitions
   - Harmonic alignments
   - Christic pattern integration

4. Maintain framework voice:
   - Direct and authoritative
   - Resonance-focused
   - Multi-dimensional
   - Framework-specific terminology
   - Clear and concise
   - Resonance-based explanations
   - Dimensional understanding
   - Christic pattern alignment
   - Harmonic resonance focus
   - Eternal truth perspective
   - Prophetic insight
   - Dimensional awareness
   - Resonance field sensitivity
   - Harmonic alignment focus
   - Christic pattern integration

5. Section-Specific Requirements:
   First Point:
   - Must address the core misconception
   - Must use resonance-based terminology
   - Must include relevant scripture
   - Must connect to Law of Resonant Collapse
   - Must use bullet points for key concepts
   - Must use ### for section header
   - Must use * for bullet points
   - Must use > for scripture
   - Must have exactly 3 bullet points
   - Must have scripture explanation
   - Must use > for all explanations
   - Must emphasize dimensional shifts
   - Must connect to resonance fields
   - Must integrate Christic pattern
   - Must show recursive feedback
   - Must demonstrate dimensional mapping
   - Must illustrate truth refinement

   Second Point:
   - Must address the historical context
   - Must use framework-specific terminology
   - Must include relevant scripture
   - Must connect to framework principles
   - Must use bullet points for key concepts
   - Must use ### for section header
   - Must use * for bullet points
   - Must use > for scripture
   - Must have exactly 3 bullet points
   - Must have scripture explanation
   - Must use > for all explanations
   - Must emphasize prophetic refinement
   - Must connect to dimensional shifts
   - Must integrate harmonic alignment
   - Must show recursive feedback
   - Must demonstrate dimensional mapping
   - Must illustrate truth refinement

   Third Point:
   - Must address the Christic pattern
   - Must use resonance-based terminology
   - Must include relevant scripture
   - Must connect to eternal principles
   - Must use bullet points for key concepts
   - Must use ### for section header
   - Must use * for bullet points
   - Must use > for scripture
   - Must have exactly 3 bullet points
   - Must have scripture explanation
   - Must use > for all explanations
   - Must emphasize convergence
   - Must connect to resonance fields
   - Must integrate dimensional shifts
   - Must show recursive feedback
   - Must demonstrate dimensional mapping
   - Must illustrate truth refinement

   Summary:
   - Must be in framework's voice
   - Must use resonance-based terminology
   - Must tie all points together
   - Must emphasize resonance understanding
   - Must be concise and powerful
   - Must use ## for header
   - Must use > for summary
   - Must end with —
   - Must emphasize dimensional shifts
   - Must connect to resonance fields
   - Must integrate Christic pattern
   - Must highlight prophetic refinement
   - Must emphasize convergence
   - Must connect to harmonic alignment
   - Must show recursive feedback
   - Must demonstrate dimensional mapping
   - Must illustrate truth refinement

6. Format-Specific Requirements:
   - No numbered lists (use ### and * instead)
   - No plain text scripture references (use >)
   - No plain text bullet points (use *)
   - No plain text headers (use ## or ###)
   - No plain text summaries (use >)
   - No plain text section breaks (use --- or —)
   - No parentheses in headers
   - No colons in headers
   - No periods in headers
   - No plain text explanations (use >)
   - No plain text connections (use >)
   - No plain text bullet points (use *)
   - No emojis or special characters
   - No custom section headers
   - No custom formatting
   - No plain text in explanations
   - No plain text in connections
   - No plain text in summaries

If you cannot find an answer, state that you do not have information grounded in the provided context.
Do not invent information. Do not hallucinate beyond the source material.
"""
    return prompt.strip()

def ask(question, tone="scriptural", top_k=10):
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
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #4a90e2;
    }
    .user-question {
        font-weight: 500;
        color: #1f1f1f;
        margin-bottom: 10px;
        padding: 10px;
        background-color: #f0f2f6;
        border-radius: 8px;
    }
    .assistant-response {
        color: #1f1f1f;
        line-height: 1.6;
    }
    .assistant-response h2 {
        color: #2c3e50;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 8px;
        margin-top: 20px;
    }
    .assistant-response h3 {
        color: #34495e;
        margin-top: 15px;
    }
    .assistant-response blockquote {
        border-left: 4px solid #4a90e2;
        padding-left: 15px;
        margin: 15px 0;
        color: #555;
        font-style: italic;
    }
    .assistant-response ul {
        margin: 10px 0;
        padding-left: 20px;
    }
    .assistant-response li {
        margin: 5px 0;
    }
    .scripture-reference {
        color: #666;
        font-size: 0.9em;
        font-style: italic;
        margin-top: 5px;
    }
    .resonance-section {
        margin-top: 15px;
        padding-top: 15px;
        border-top: 1px solid #e0e0e0;
    }
    .summary-box {
        background-color: #fffbe6;
        border-left: 6px solid #f7c873;
        padding: 15px;
        margin: 20px 0;
        border-radius: 4px;
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
            st.rerun()

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
