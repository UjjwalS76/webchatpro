import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Initialize Streamlit app
st.set_page_config(page_title="Website Chatbot", layout="wide")
st.title("Chat with Website Content")

# Sidebar for API Key input and other configurations
st.sidebar.header("Configuration")
default_url = "https://www.vedabase.io/"

# Input for website URL
url = st.sidebar.text_input("Enter Website URL", value=default_url)

# Function to load website data
@st.cache_data(show_spinner=False)
def website_loader(url):
    loader = WebBaseLoader(url)
    data = loader.load()
    return data

# Function to process website data
@st.cache_data(show_spinner=False)
def process_website(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(data)
    return splits

# Function to initialize embeddings
def initialize_embeddings(api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=api_key)
    return embeddings

# Function to initialize LLM
def initialize_llm(api_key):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-002", api_key=api_key)
    return llm

# Load and process website data
if url:
    with st.spinner("Loading website data..."):
        website_data = website_loader(url)
    with st.spinner("Processing website data..."):
        splits = process_website(website_data)
else:
    st.warning("Please enter a website URL.")

# Initialize embeddings and LLM using API keys from secrets
if "google_api_key" not in st.secrets:
    st.error("Google API key not found in secrets.")
    st.stop()

google_api_key = st.secrets["GOOGLE_API_KEY"]
embeddings = initialize_embeddings(google_api_key)

llm = initialize_llm(google_api_key)

# Create or load vector store
@st.cache_resource(show_spinner=False)
def get_vectorstore(splits, embeddings):
    vectorstore = Chroma.from_documents(splits, embeddings)
    return vectorstore

with st.spinner("Creating vector store..."):
    vectorstore = get_vectorstore(splits, embeddings)

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create Conversational Retrieval Chain
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

# Chat interface
def chat_with_website(query):
    answer = qa({"question": query})
    return answer['answer']

# Streamlit chat interface
if "responses" not in st.session_state:
    st.session_state.responses = []
if "questions" not in st.session_state:
    st.session_state.questions = []

st.header("Ask questions about the website:")
user_input = st.text_input("You:", key="input")

if st.button("Send") and user_input:
    with st.spinner("Generating response..."):
        response = chat_with_website(user_input)
    st.session_state.questions.append(user_input)
    st.session_state.responses.append(response)

# Display chat history
if st.session_state.responses:
    st.markdown("### Chat History")
    for i in range(len(st.session_state.responses)):
        st.markdown(f"**You:** {st.session_state.questions[i]}")
        st.markdown(f"**Bot:** {st.session_state.responses[i]}")
