## Streamlit UI. Run with: streamlit run main.py
import logging

import streamlit as st
from graph import build_graph, new_state
from config import MEMORY_TURNS

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Financial News Assistant")
st.title("Financial News Assistant")
st.caption("Simple summaries of stock prices, company news, and market updates.")
st.info("This assistant focuses on financial topics and does not provide investment advice.")


@st.cache_resource
def get_graph():
    return build_graph()


graph = get_graph()

if "chat" not in st.session_state:
    st.session_state.chat = []

if st.button("Clear conversation"):
    st.session_state.chat = []

for message in st.session_state.chat:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("trace"):
            with st.expander("Agent activity"):
                for step in message["trace"]:
                    st.write(step)

st.caption("Try a question")
columns = st.columns(3)
suggested_query = None
if columns[0].button("Tesla price and news"):
    suggested_query = "What is Tesla's latest stock price and recent news?"
if columns[1].button("Recent market news"):
    suggested_query = "What are the latest financial market headlines?"
if columns[2].button("Compare two stocks"):
    suggested_query = "Compare the recent stock performance of Apple and Microsoft."

typed_query = st.chat_input("Ask about a stock, company, or market...")
query = typed_query or suggested_query

if query:
    chat = st.session_state.chat
    history = [
        (chat[index]["content"], chat[index + 1]["content"])
        for index in range(0, len(chat) - 1, 2)
    ][-MEMORY_TURNS:]

    st.session_state.chat.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        trace, answer = [], ""
        status = None
        request_failed = False
        try:
            with st.status("Working on your question...", expanded=True) as status:
                activity = st.empty()
                for update in graph.stream(new_state(query, history), stream_mode="updates"):
                    for output in update.values():
                        for line in output.get("trace", []):
                            trace.append(line)
                        answer = output.get("answer", answer)
                    if trace:
                        activity.markdown("\n".join(f"- {step}" for step in trace))
                status.update(label="Answer ready", state="complete")
        except Exception:
            logger.exception("Could not complete the chat request")
            request_failed = True
            answer = (
                "I couldn't complete that request. Please try again or rephrase it "
                "as a question about a company, stock, or market."
            )
            trace = ["System: The request could not be completed."]
            if status is not None:
                status.update(label="Could not complete the request", state="error")
            st.error(answer)
        if not request_failed:
            st.markdown(answer)

    st.session_state.chat.append({"role": "assistant", "content": answer, "trace": trace})
