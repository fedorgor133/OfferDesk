import streamlit as st

from src.core.rag_agent import RAGAgent


st.set_page_config(page_title="Offer Desk Agent", page_icon="💬", layout="centered")

st.title("Offer Desk Agent")
st.caption("Policy-based guidance for Account Executives")


@st.cache_resource
def get_agent() -> RAGAgent:
    agent = RAGAgent(local_mode=True)
    agent.load_documents()
    agent.initialize()
    return agent


agent = get_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []
    # Show welcoming message only once
    welcome_msg = """
    Welcome! 👋

    **Offer Desk AI** is an internal guidance tool for account executives sales teams, designed to provide clear commercial solutions, boundaries and recommendations for complex Core and XL deals. It draws on defined rules and playbooks to help AEs know what they can offer without unnecessary internal friction.

    **What can I help you with?** Ask me about:
    - Pricing rules and discounts
    - Multi-year commitments and caps
    - Seasonal billing and seat management
    - Commercial exceptions and POC approvals
    - Deal structures and policy boundaries
    """
    st.info(welcome_msg)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask Offer Desk...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = agent.query(user_input)
            answer = result.get("answer", "No answer returned.")
            sources = result.get("sources", [])

        st.markdown(answer)

        if sources:
            with st.expander("Sources"):
                for src in sources:
                    section = src.get("section", "N/A")
                    conv_id = src.get("conversation_id", "N/A")
                    source = src.get("source", "Unknown")
                    st.markdown(f"- Section {section} | Conversation {conv_id} | {source}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
