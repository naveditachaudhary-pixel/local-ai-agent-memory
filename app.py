"""Streamlit chat UI for the agent.  Run with:  streamlit run app.py

Locally it uses Ollama. When a GROQ_API_KEY is configured (for example in Streamlit Cloud secrets),
it switches to demo mode: a hosted model, and memory that is private to each visitor's session.
"""

import json
import os
import tempfile
from pathlib import Path

import streamlit as st

from agent import Agent, Settings
from agent.memory import EXCHANGE, FACT

st.set_page_config(page_title="Local AI Agent with Memory", page_icon="🧠", layout="wide")


def groq_key() -> str:
    try:
        val = st.secrets.get("GROQ_API_KEY", "") or os.environ.get("GROQ_API_KEY", "")
        if val:
            return val
    except Exception:  # no secrets file
        pass
    if os.environ.get("GROQ_API_KEY"):
        return os.environ.get("GROQ_API_KEY")
    if "key" in st.query_params and st.query_params["key"]:
        return st.query_params["key"]
    if "groq_api_key" in st.query_params and st.query_params["groq_api_key"]:
        return st.query_params["groq_api_key"]
    return st.session_state.get("custom_groq_key", "")


DEMO = bool(groq_key())


@st.cache_resource
def get_local_agent() -> Agent:
    return Agent()


def get_demo_agent() -> Agent:
    """One agent per visitor, with memory in a private temporary folder."""
    if "agent" not in st.session_state:
        os.environ["GROQ_API_KEY"] = groq_key()
        folder = tempfile.mkdtemp(prefix="agent-demo-")
        settings = Settings(backend="groq", data_dir=Path(folder),
                            max_distance=float(os.environ.get("AGENT_MAX_DISTANCE", "0.65")))
        st.session_state.agent = Agent(settings)
    return st.session_state.agent


agent = get_demo_agent() if DEMO else get_local_agent()

# ---------- model check ----------
ok, status = agent.llm.status() if hasattr(agent.llm, "status") else (True, "custom model")
if not ok:
    st.title("🧠 Local AI Agent with Memory")
    if not DEMO:
        st.info(
            "💡 **Running online?** Enter a free [Groq API key](https://console.groq.com) below to try the hosted demo right now, or run Ollama on your local machine."
        )
        col1, col2 = st.columns([4, 1])
        user_key = col1.text_input("Groq API Key", type="password", placeholder="gsk_...", label_visibility="collapsed")
        if col2.button("Start Demo", use_container_width=True) and user_key:
            st.session_state.custom_groq_key = user_key.strip()
            st.rerun()
    st.error(status)
    st.markdown(
        """**Setup (one time):**
1. Install Ollama from [ollama.com](https://ollama.com) and start it.
2. Pull the two models:
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```
3. Refresh this page."""
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "sent" not in st.session_state:
    st.session_state.sent = 0

# ---------- sidebar: what the agent remembers ----------
with st.sidebar:
    st.subheader("🧠 Memory")
    models = agent.llm.chat_models() if hasattr(agent.llm, "chat_models") else []
    if len(models) > 1:
        current = agent.settings.chat_model
        index = models.index(current) if current in models else 0
        chosen = st.selectbox("Chat model", models, index=index, help="Any chat model you've pulled with `ollama pull`.")
        if chosen != current:
            agent.settings.chat_model = chosen
            st.rerun()
        st.caption(f"Embeddings: {agent.settings.embed_model}")
    else:
        st.caption(f"Model: {status}")

    facts = agent.memory.all(FACT)
    c1, c2, c3 = st.columns(3)
    c1.metric("Facts", len(facts))
    c2.metric("Past chats", agent.memory.count(EXCHANGE))
    c3.metric("Turns", agent.summaries.turns)

    if st.button("Start a new session", icon=":material/restart_alt:", use_container_width=True,
                 disabled=not st.session_state.messages,
                 help="Clears the chat, like closing and reopening the app. Memory is kept."):
        agent.new_session()
        st.session_state.messages = []
        st.session_state.new_session = True
        st.rerun()

    st.markdown("**Summary**")
    st.caption(agent.summaries.summary or "No summary yet. It builds up as you chat.")
    if st.button("Refresh summary now", use_container_width=True, disabled=not agent.history):
        with st.spinner("Summarizing…"):
            agent.refresh_summary()
        st.rerun()

    st.markdown("**What I remember about you**")
    if not facts:
        st.caption("Nothing yet. Tell me about yourself!")
    for fact in facts:
        col_text, col_btn = st.columns([5, 1])
        col_text.write(fact.text)
        if col_btn.button("", icon=":material/close:", key=f"forget-{fact.id}", help="Forget this"):
            agent.memory.forget(fact.id)
            st.rerun()

    with st.expander("Export / import"):
        st.download_button("Download memory (JSON)", json.dumps(agent.export_memory(), indent=2),
                           file_name="agent-memory.json", mime="application/json",
                           icon=":material/download:", use_container_width=True)
        upload = st.file_uploader("Import a memory file", type="json", key=f"upload-{st.session_state.get('uploads', 0)}")
        if upload is not None and st.button("Import", icon=":material/upload:", use_container_width=True):
            try:
                added = agent.import_memory(json.loads(upload.getvalue()))
            except (ValueError, json.JSONDecodeError) as err:
                st.error(str(err) if isinstance(err, ValueError) else "That file isn't valid JSON.")
            else:
                st.session_state.uploads = st.session_state.get("uploads", 0) + 1  # clears the uploader
                st.session_state.imported = added
                st.rerun()
        if "imported" in st.session_state:
            st.success(f"Imported {st.session_state.pop('imported')} memories.")

    with st.expander("Danger zone"):
        sure = st.checkbox("I understand this deletes all memory")
        if st.button("Clear all memory", type="primary", disabled=not sure, use_container_width=True):
            agent.reset_memory()
            st.session_state.messages = []
            st.rerun()

# ---------- chat ----------
st.title("🧠 Local AI Agent with Memory")
if DEMO:
    st.info(
        "**Online demo.** Replies come from a hosted model so you can try it without installing anything. "
        "Your memory is private to this browser tab and is deleted when you close it. "
        "The full version runs entirely on your own computer: "
        "[see the code on GitHub](https://github.com/naveditachaudhary-pixel/local-ai-agent-memory).",
        icon=":material/info:",
    )
else:
    st.caption("Runs entirely on your machine. Remembers facts and past conversations between sessions.")

if st.session_state.pop("new_session", False):
    st.success("New session started. The chat is empty, but the memory isn't. "
               "Ask me what I remember about you.", icon=":material/restart_alt:")
elif not st.session_state.messages and not agent.memory.count():
    st.markdown(
        "**Try this:** tell me a few things about yourself, for example *“I'm Sam, I work as a nurse "
        "and I'm training for a half marathon.”* Then press **Start a new session** in the sidebar "
        "and ask *“What do you know about me?”*"
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("recalled"):
            st.caption("Recalled: " + " · ".join(msg["recalled"]))

limit = agent.settings.demo_message_limit
if DEMO and st.session_state.sent >= limit:
    st.warning(f"That's the {limit}-message limit for the online demo. "
               "To keep going, run it on your own computer (instructions on GitHub).")
    st.stop()

if prompt := st.chat_input("Say something…"):
    st.session_state.sent += 1
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(agent.stream(prompt))
        except Exception as err:  # e.g. the hosted model is busy or rate-limited
            st.session_state.messages.pop()
            st.error(f"The model didn't respond ({type(err).__name__}). Please try again in a moment.")
            st.stop()
        recalled = [m.text for m in agent.last_recalled if m.kind == FACT]
        if recalled:
            st.caption("Recalled: " + " · ".join(recalled))

    st.session_state.messages.append({"role": "assistant", "content": reply, "recalled": recalled})
    replaced = {new for _, new in agent.last_updated}
    for old, new in agent.last_updated:
        st.toast(f"Updated: {new} (was: {old})", icon="🔄")
    for fact in agent.last_learned:
        if fact not in replaced:
            st.toast(f"Remembered: {fact}", icon="🧠")
    st.rerun()  # refresh the sidebar counters and facts after every turn
