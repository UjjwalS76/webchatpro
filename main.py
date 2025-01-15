import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


# Configure Streamlit
st.set_page_config(page_title="Website Chatbot", layout="wide")
st.title("🗣️ Chat with Website Content")

# Sidebar configuration
st.sidebar.header("🔧 Configuration")

# User inputs website URL (no default)
url = st.sidebar.text_input(
    "Enter a valid website URL (including http:// or https://)",
    value=""
)

def load_website(website_url: str):
    """
    Loads text content from a specified URL using WebBaseLoader.
    Raises an exception if the URL is invalid or unreachable.
    """
    if not (website_url.startswith("http://") or website_url.startswith("https://")):
        raise ValueError("Please enter a valid URL with http:// or https://.")
    loader = WebBaseLoader(website_url)
    data = loader.load()
    return data


def process_data(data):
    """
    Splits documents into manageable chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(data)
    return splits


def get_google_api_key():
    """
    Retrieve the Google API key from Streamlit's secrets.
    If not set, show an error message and stop execution.
    """
    if "GOOGLE_API_KEY" not in st.secrets:
        st.error("❌ Google API key not found in Streamlit secrets.")
        st.stop()
    return st.secrets["GOOGLE_API_KEY"]


def get_embeddings(api_key):
    """
    Initialize GoogleGenerativeAIEmbeddings.
    """
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        api_key=api_key
    )


def get_llm(api_key):
    """
    Initialize the ChatGoogleGenerativeAI LLM with Gemini model.
    """
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-002",
        api_key=api_key
    )


def get_vectorstore(docs, embedding_fn):
    """
    Create a Chroma vector store from documents.
    """
    vectorstore = Chroma.from_documents(docs, embedding_fn)
    return vectorstore


# Main logic
if url:
    # Try loading the website data if the user has entered a URL
    try:
        with st.spinner("🔍 Loading website data..."):
            website_data = load_website(url)
    except Exception as e:
        st.error(f"Error loading website: {e}")
        st.stop()

    # Process the data
    with st.spinner("🛠️ Processing website data..."):
        splits = process_data(website_data)

    # Get API key and set up embeddings
    google_api_key = get_google_api_key()
    embeddings = get_embeddings(google_api_key)

    with st.spinner("📦 Creating vector store..."):
        vectorstore = get_vectorstore(splits, embeddings)

    # Initialize the LLM
    llm = get_llm(google_api_key)

    # Conversation memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create Conversational Retrieval Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    st.header("💬 Ask Questions About the Website:")
    user_input = st.text_input("You:", key="input")

    if st.button("Send") and user_input.strip():
        with st.spinner("🧠 Generating response..."):
            try:
                response = qa_chain({"question": user_input})
                answer = response['answer']
                st.session_state.setdefault('questions', []).append(user_input)
                st.session_state.setdefault('responses', []).append(answer)
            except Exception as e:
                st.error(f"Error generating response: {e}")

    # Display chat history
    if 'responses' in st.session_state and st.session_state.responses:
        st.markdown("### 🗨️ Chat History")
        for i in range(len(st.session_state.responses)):
            st.markdown(f"**You:** {st.session_state.questions[i]}")
            st.markdown(f"**Bot:** {st.session_state.responses[i]}")
else:
    st.info("🔄 Please enter a valid website URL (including http:// or https://) to get started.")
