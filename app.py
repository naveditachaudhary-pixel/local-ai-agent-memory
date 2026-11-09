"""Streamlit chat UI for the local agent.  Run with:  streamlit run app.py"""

import streamlit as st

from agent import Agent
from agent.memory import EXCHANGE, FACT

st.set_page_config(page_title="Local AI Agent with Memory", page_icon="🧠", layout="wide")


@st.cache_resource
def get_agent() -> Agent:
    return Agent()


agent = get_agent()

# ---------- model check ----------
ok, status = agent.llm.status() if hasattr(agent.llm, "status") else (True, "custom model")
if not ok:
    st.title("🧠 Local AI Agent with Memory")
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

# ---------- sidebar: what the agent remembers ----------
with st.sidebar:
    st.subheader("🧠 Memory")
    st.caption(f"Model: {status}")

    facts = agent.memory.all(FACT)
    c1, c2, c3 = st.columns(3)
    c1.metric("Facts", len(facts))
    c2.metric("Past chats", agent.memory.count(EXCHANGE))
    c3.metric("Turns", agent.summaries.turns)

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

    with st.expander("Danger zone"):
        sure = st.checkbox("I understand this deletes all memory")
        if st.button("Clear all memory", type="primary", disabled=not sure, use_container_width=True):
            agent.reset_memory()
            st.session_state.messages = []
            st.rerun()

# ---------- chat ----------
st.title("🧠 Local AI Agent with Memory")
st.caption("Runs entirely on your machine. Remembers facts and past conversations between sessions.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("recalled"):
            st.caption("Recalled: " + " · ".join(msg["recalled"]))

if prompt := st.chat_input("Say something…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        reply = st.write_stream(agent.stream(prompt))
        recalled = [m.text for m in agent.last_recalled if m.kind == FACT]
        if recalled:
            st.caption("Recalled: " + " · ".join(recalled))

    st.session_state.messages.append({"role": "assistant", "content": reply, "recalled": recalled})
    for fact in agent.last_learned:
        st.toast(f"Remembered: {fact}", icon="🧠")
    st.rerun()  # refresh the sidebar counters and facts after every turn
