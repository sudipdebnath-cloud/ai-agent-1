import os
import time
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder
import tempfile

# ------------------ Setup ------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY not found. Add it to .env (local) or Streamlit Secrets (cloud).")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

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

# Model selector state
available_models = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-3.5-turbo"
]
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gpt-4o-mini"

# ------------------ Sidebar Settings ------------------
st.sidebar.header("⚙️ Settings")
st.session_state.selected_model = st.sidebar.selectbox(
    "Choose AI Model",
    available_models,
    index=available_models.index(st.session_state.selected_model)
)

# ------------------ Conversation Container ------------------
conversation_container = st.container()


def render_conversation():
    """Render all messages with assistant replies in styled boxes."""
    with conversation_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"**👤 You:** {msg['content']}")
            else:
                st.markdown("**🤖 Assistant:**")  # label outside box
                st.markdown(
                    f"""
                    <div style="
                        padding: 0.8em;
                        border: 1px solid #ddd;
                        border-radius: 8px;
                        background-color: #f9f9f9;
                        margin-bottom: 0.5em;">
                        {msg['content']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ------------------ Stream assistant reply ------------------
def stream_reply():
    """Stream assistant reply inside styled box with label outside."""
    st.session_state.is_replying = True
    label_placeholder = conversation_container.empty()
    reply_placeholder = conversation_container.empty()
    full_reply = ""

    # Inject system message so the assistant knows which model is active
    messages_with_model = [
        {"role": "system", "content": f"You are ChatGPT running on the {st.session_state.selected_model} model."}
    ] + st.session_state.messages

    # Show assistant label first
    label_placeholder.markdown("**🤖 Assistant:**")

    # Get response with selected model
    response = client.chat.completions.create(
        model=st.session_state.selected_model,
        messages=messages_with_model
    )
    reply = response.choices[0].message.content

    # Typing effect inside styled box
    for char in reply:
        full_reply += char
        reply_placeholder.markdown(
            f"""
            <div style="
                padding: 0.8em;
                border: 1px solid #ddd;
                border-radius: 8px;
                background-color: #f9f9f9;
                margin-bottom: 0.5em;">
                {full_reply}▌
            </div>
            """,
            unsafe_allow_html=True,
        )
        time.sleep(0.01)

    # Final reply (without cursor)
    reply_placeholder.markdown(
        f"""
        <div style="
            padding: 0.8em;
            border: 1px solid #ddd;
            border-radius: 8px;
            background-color: #f9f9f9;
            margin-bottom: 0.5em;">
            {full_reply}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.session_state.messages.append({"role": "assistant", "content": full_reply})
    st.session_state.is_replying = False


# ------------------ Display past conversation ------------------
render_conversation()

# ------------------ Text Input ------------------
if st.session_state.is_replying:
    st.info("🤖 Assistant is typing... Please wait before sending new messages or recording.")
    user_text = None
else:
    user_text = st.chat_input("Type a message and press Enter...")
    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})
        render_conversation()
        stream_reply()

# ------------------ Audio Input ------------------
st.markdown("### 🎤 Or speak")
if st.session_state.is_replying:
    st.info("🤖 Assistant is typing... Recording disabled.")
    audio = None
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

        # Save audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio["bytes"])
            audio_path = f.name

        # Transcribe
        with open(audio_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
        text = transcript.text

        st.session_state.messages.append({"role": "user", "content": text})
        render_conversation()
        stream_reply()
