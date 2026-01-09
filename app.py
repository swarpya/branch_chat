import streamlit as st
from groq import Groq
import uuid

# --- Configuration ---
MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct"

try:
    if "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]
    else:
        st.error("GROQ_API_KEY not found in secrets.toml")
        st.stop()
        
    client = Groq(api_key=api_key)
except Exception as e:
    st.error(f"Error initializing Groq client: {e}")
    st.stop()

# --- System Prompt ---
SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are a helpful assistant. You must provide answers in bullet points. Use a maximum of 3 bullet points."
}

# --- State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "threads" not in st.session_state:
    st.session_state.threads = {}

if "active_thread_id" not in st.session_state:
    st.session_state.active_thread_id = None

# --- Helper Functions ---

def parse_groq_stream(stream):
    """Yields clean text content from the raw Groq response objects."""
    for chunk in stream:
        if chunk.choices:
            content = chunk.choices[0].delta.content
            if content:
                yield content

def generate_response(user_query, context_text=None):
    """
    Generates response. 
    If context_text is provided (for threads), it is embedded in the user prompt.
    """
    
    # Construct the message payload
    if context_text:
        # Combine context + query for the LLM so it understands what we are talking about
        combined_prompt = f"Context from previous conversation:\n{context_text}\n\nCurrent Question:\n{user_query}"
        messages = [
            SYSTEM_PROMPT,
            {"role": "user", "content": combined_prompt}
        ]
    else:
        # Standard stateless request
        messages = [
            SYSTEM_PROMPT,
            {"role": "user", "content": user_query}
        ]

    try:
        stream = client.chat.completions.create(
            messages=messages,
            model=MODEL_ID,
            temperature=0.5,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
        )
        return stream
    except Exception as e:
        return f"Error: {str(e)}"

def enter_thread(msg_id):
    st.session_state.active_thread_id = msg_id

def exit_thread():
    st.session_state.active_thread_id = None

# --- UI Layout ---

st.title("Groq Chat: Stateless & Branching")

# LOGIC BRANCH: Are we in Main View or Thread View?
if st.session_state.active_thread_id is None:
    # ==========================
    # MAIN CONVERSATION VIEW
    # ==========================
    
    st.subheader("Main Conversation")

    # Display History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Add "Reply/Follow-up" button for User messages
            if msg["role"] == "user":
                thread_count = len(st.session_state.threads.get(msg["id"], []))
                label = "Open Thread" if thread_count == 0 else f"View Thread ({thread_count})"
                st.button(label, key=f"btn_{msg['id']}", on_click=enter_thread, args=(msg["id"],))

    # Main Input
    if prompt := st.chat_input("Start a new query..."):
        # 1. Generate ID immediately (so we can use it for the button)
        new_msg_id = str(uuid.uuid4())

        # 2. Display User Message AND The Button immediately
        with st.chat_message("user"):
            st.markdown(prompt)
            # FIX: Manually add the button here for the new message so it appears instantly
            st.button("Open Thread", key=f"btn_{new_msg_id}", on_click=enter_thread, args=(new_msg_id,))
        
        # 3. Add User Message to State
        st.session_state.messages.append({
            "id": new_msg_id,
            "role": "user",
            "content": prompt
        })

        # 4. Generate & Stream Response (No Context)
        with st.chat_message("assistant"):
            raw_stream = generate_response(prompt)
            response_text = st.write_stream(parse_groq_stream(raw_stream))
        
        # 5. Add Assistant Message to State
        st.session_state.messages.append({
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": response_text
        })

else:
    # ==========================
    # THREAD / FOLLOW-UP VIEW
    # ==========================
    
    parent_id = st.session_state.active_thread_id
    parent_msg = next((m for m in st.session_state.messages if m["id"] == parent_id), None)
    
    if not parent_msg:
        st.error("Error: Parent message not found.")
        if st.button("Return to Main"):
            exit_thread()
            st.rerun()
    else:
        # Header
        col1, col2 = st.columns([1, 5])
        with col1:
            st.button("← Back", on_click=exit_thread)
        with col2:
            st.info(f"Viewing Follow-up Thread for: **{parent_msg['content'][:50]}...**")

        st.divider()

        # Display Parent Context
        st.caption("Original Query:")
        with st.chat_message(parent_msg["role"]):
            st.markdown(parent_msg["content"])
        st.divider()

        # Initialize Thread Storage
        if parent_id not in st.session_state.threads:
            st.session_state.threads[parent_id] = []

        # Display Thread History
        for t_msg in st.session_state.threads[parent_id]:
            with st.chat_message(t_msg["role"]):
                st.markdown(t_msg["content"])

        # Thread Input
        if thread_prompt := st.chat_input("Ask a follow-up..."):
            # Display User Message
            with st.chat_message("user"):
                st.markdown(thread_prompt)
            
            # Save to Thread State
            st.session_state.threads[parent_id].append({
                "role": "user",
                "content": thread_prompt
            })

            # Generate Response WITH CONTEXT
            with st.chat_message("assistant"):
                # We pass parent message content so the AI has context
                raw_stream = generate_response(thread_prompt, context_text=parent_msg["content"]) 
                response_text = st.write_stream(parse_groq_stream(raw_stream))

            # Save Response to Thread State
            st.session_state.threads[parent_id].append({
                "role": "assistant",
                "content": response_text
            })