import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="centered")

st.title("PDF Chatbot")
st.write("Upload a PDF and ask questions about it in plain English.")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "Grok API key",
        type="password",
        placeholder="xai-...",
        help="Get your key from console.x.ai",
    )
    st.caption("Your key is never stored — it only lives in this session.")
    st.divider()
    st.markdown("**How to get a Grok API key**")
    st.markdown(
        "1. Go to **console.x.ai** \n"
        "2. Sign in with your X (Twitter) account \n"
        "3. Click API Keys → Create key \n"
        "4. Paste it above"
    )
    st.divider()
    st.markdown("**How it works**")
    st.markdown(
        "1. Upload a PDF \n"
        "2. Text is split into chunks \n"
        "3. Chunks embedded locally (free, no API) \n"
        "4. Your question finds relevant chunks \n"
        "5. Grok answers using those chunks"
    )


def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


def build_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    return FAISS.from_texts(chunks, embedding=get_embeddings())


def get_answer(question, vectorstore, api_key, message_history):
    """Retrieve relevant chunks and ask Grok — manually managing history."""

    # 1. find relevant PDF chunks for this question
    docs = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join(doc.page_content for doc in docs)

    # 2. build message list manually — no LangChain chain needed
    llm = ChatOpenAI(
        model_name="grok-3",
        temperature=0,
        openai_api_key=api_key,
        openai_api_base="https://api.x.ai/v1",
    )

    messages = [
        SystemMessage(content=(
            "You are a helpful assistant that answers questions based only on "
            "the PDF context provided below. If the answer is not in the context, "
            "say so clearly.\n\nContext:\n" + context
        ))
    ]

    # add previous conversation turns so Grok remembers context
    messages.extend(message_history)

    # add current question
    messages.append(HumanMessage(content=question))

    response = llm.invoke(messages)
    return response.content, docs


# ── Session state ─────────────────────────────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []       # for display
if "message_history" not in st.session_state:
    st.session_state.message_history = []    # LangChain message objects
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# ── File upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and api_key:
    if st.session_state.pdf_name != uploaded_file.name:
        with st.spinner("Reading and processing PDF..."):
            raw_text = extract_text(uploaded_file)
            if not raw_text.strip():
                st.error("Could not extract text — PDF may be scanned or image-based.")
                st.stop()
            try:
                st.session_state.vectorstore = build_vectorstore(raw_text)
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
                st.stop()
            st.session_state.chat_history = []
            st.session_state.message_history = []
            st.session_state.pdf_name = uploaded_file.name
        st.success(f"Ready! Ask me anything about **{uploaded_file.name}**")

elif uploaded_file and not api_key:
    st.warning("Please enter your Grok API key in the sidebar.")
elif not uploaded_file:
    st.info("Upload a PDF to get started.")

# ── Chat interface ────────────────────────────────────────────────────────────
if st.session_state.vectorstore and api_key:

    # show chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_question = st.chat_input("Ask a question about the PDF...")

    if user_question:
        # show user message
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)

        # get answer from Grok
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, source_docs = get_answer(
                        user_question,
                        st.session_state.vectorstore,
                        api_key,
                        st.session_state.message_history,
                    )
                    st.write(answer)

                    with st.expander("View source excerpts from PDF"):
                        for i, doc in enumerate(source_docs, 1):
                            st.markdown(f"**Excerpt {i}:**")
                            st.caption(doc.page_content[:400] + "...")

                    # update message history for next turn
                    st.session_state.message_history.append(HumanMessage(content=user_question))
                    st.session_state.message_history.append(AIMessage(content=answer))
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})

                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.chat_history:
        if st.button("Clear chat history"):
            st.session_state.chat_history = []
            st.session_state.message_history = []
            st.rerun()

st.divider()
st.caption("Built with Streamlit, LangChain, FAISS, HuggingFace Embeddings, and Grok (xAI)")