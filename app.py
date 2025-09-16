import os
import time
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder
import tempfile

# ------------------ Setup ------------------
load_dotenv()

# Session state for API key
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

# ------------------ Sidebar Settings ------------------
st.sidebar.header("⚙️ Settings")

# API Key Input (manual override)
manual_api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
if manual_api_key:
    st.session_state.api_key = manual_api_key.strip()

# Handle missing API key
if not st.session_state.api_key:
    st.error(
        "🚫 **API Key Missing**\n\n"
        "- No OpenAI API key found.\n"
        "- Please set it in your `.env` file, Streamlit Secrets, or enter it manually in the sidebar."
    )
    st.stop()

# Create OpenAI client
client = OpenAI(api_key=st.session_state.api_key)

# Validate API Key before using
try:
    _ = client.models.list()
except Exception as e:
    error_msg = str(e)
    display_msg = "❌ Invalid or non-working OpenAI API Key."
    if "401" in error_msg or "invalid_api_key" in error_msg:
        display_msg = (
            "🚫 **Invalid API Key**\n\n"
            "- Double-check your API key.\n"
            "- Get/create a new key at: [OpenAI API Keys](https://platform.openai.com/account/api-keys)"
        )
    st.error(display_msg)
    st.stop()

# ------------------ App UI ------------------
st.set_page_config(page_title="AI Agent by SD", page_icon="🎙️")
st.title("🎙️ AI Agent by SD")
st.markdown("Ask me via **microphone** or **text**. I'll answer like ChatGPT.")

# ------------------ Session State ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None

if "is_replying" not in st.session_state:
    st.session_state.is_replying = False

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# ------------------ Model selector (dynamic & filtered) ------------------
try:
    all_models = client.models.list().data
    # Only chat-capable GPT models (exclude audio-only or unsupported)
    available_models = [
        m.id for m in all_models
        if "gpt" in m.id.lower()
        and all(x not in m.id.lower() for x in [
            "instruct", "embedding", "codex", "davinci", "babbage", "curie", "ada", "audio"
        ])
    ]
    if not available_models:
        raise ValueError("No GPT chat models found, using defaults.")
except Exception as e:
    st.warning(f"Could not fetch models dynamically. Using default list. Error: {e}")
    available_models = [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-3.5-turbo"
    ]

# Ensure selected_model is valid
if st.session_state.selected_model not in available_models:
    st.session_state.selected_model = available_models[0]

# ------------------ Model selection with confirmation ------------------
new_model = st.sidebar.selectbox(
    "Choose AI Model",
    available_models,
    index=available_models.index(st.session_state.selected_model)
)

if new_model != st.session_state.selected_model:
    if st.sidebar.checkbox(f"Confirm switch to {new_model}? This will reset the conversation.", value=False):
        st.session_state.selected_model = new_model
        st.session_state.messages = []
        st.success(f"Model changed to {new_model} and conversation reset.")
    else:
        st.sidebar.info("Model change cancelled. Conversation preserved.")

# ------------------ Display active model below instructions ------------------
st.markdown(f"**Active Model:** {st.session_state.selected_model}")

# ------------------ Ensure system message exists ------------------
if not any(msg["role"] == "system" for msg in st.session_state.messages):
    st.session_state.messages.insert(
        0,
        {
            "role": "system",
            "content": f"You are ChatGPT. Behave like ChatGPT and mention your model {st.session_state.selected_model} when asked."
        }
    )

# ------------------ Conversation Container ------------------
conversation_container = st.container()

def render_conversation():
    with conversation_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f"<div style='margin-bottom:1.5em;'><strong>👤 You:</strong> {msg['content']}</div>",
                    unsafe_allow_html=True
                )
            elif msg["role"] == "assistant":
                st.markdown(
                    f"""
                    <div style='margin-bottom:0.2em;'><strong>🤖 Assistant:</strong></div>
                    <div style="
                        padding: 1em;
                        border: 1px solid #ddd;
                        border-radius: 10px;
                        background-color: #f5f5f5;
                        margin-bottom: 2em;">
                        {msg['content']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# ------------------ Stream assistant reply with error handling ------------------
def stream_reply():
    st.session_state.is_replying = True
    reply_placeholder = conversation_container.empty()
    full_reply = ""

    # Add empty assistant message first for history
    st.session_state.messages.append({"role": "assistant", "content": ""})

    try:
        # API call
        response = client.chat.completions.create(
            model=st.session_state.selected_model,
            messages=st.session_state.messages[:-1]  # exclude empty placeholder
        )
        reply_text = response.choices[0].message.content
        st.session_state.messages[-1]["content"] = reply_text

        # Stream character by character
        for char in reply_text:
            full_reply += char
            reply_placeholder.markdown(
                f"""
                <div style='margin-bottom:0.2em;'><strong>🤖 Assistant:</strong></div>
                <div style="
                    padding: 1em;
                    border: 1px solid #ddd;
                    border-radius: 10px;
                    background-color: #f5f5f5;
                    margin-bottom: 2em;">
                    {full_reply}▌
                </div>
                """,
                unsafe_allow_html=True
            )
            time.sleep(0.01)

        # Final render
        reply_placeholder.markdown(
            f"""
            <div style='margin-bottom:0.2em;'><strong>🤖 Assistant:</strong></div>
            <div style="
                padding: 1em;
                border: 1px solid #ddd;
                border-radius: 10px;
                background-color: #f5f5f5;
                margin-bottom: 2em;">
                {full_reply}
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:
        # Decorate error nicely
        st.error(
            f"❌ Error with model **{st.session_state.selected_model}**.\n\n"
            f"Details: {str(e)}\n\n"
            f"⚡ Make sure the model supports chat text input. Non-chat or audio-only models cannot be used here."
        )
        st.session_state.messages.pop()  # remove empty placeholder

    st.session_state.is_replying = False

# ------------------ Render past conversation ------------------
render_conversation()

# ------------------ Text Input ------------------
if st.session_state.is_replying:
    st.info("🤖 Assistant is typing... Please wait...")
else:
    user_text = st.chat_input("Type a message and press Enter...")
    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text.strip()})
        with conversation_container:
            st.markdown(
                f"<div style='margin-bottom:1.5em;'><strong>👤 You:</strong> {user_text.strip()}</div>",
                unsafe_allow_html=True
            )
        stream_reply()

# ------------------ Audio Input ------------------
st.markdown("### 🎤 Or speak")
if st.session_state.is_replying:
    st.info("🤖 Assistant is typing... Recording disabled.")
else:
    audio = mic_recorder(
        start_prompt="🎙️ Start Recording",
        stop_prompt="⏹️ Stop Recording",
        just_once=False,
        use_container_width=True,
        format="wav",
        key="recorder"
    )

    if audio and audio.get("bytes"):
        audio_id = hash(audio["bytes"])
        if st.session_state.last_audio_id != audio_id:
            st.session_state.last_audio_id = audio_id

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio["bytes"])
                audio_path = f.name

            with open(audio_path, "rb") as f:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f
                )
            text = transcript.text.strip()
            if text:
                st.session_state.messages.append({"role": "user", "content": text})
                with conversation_container:
                    st.markdown(
                        f"<div style='margin-bottom:1.5em;'><strong>👤 You:</strong> {text}</div>",
                        unsafe_allow_html=True
                    )
                stream_reply()
