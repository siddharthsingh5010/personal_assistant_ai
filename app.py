import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# ================= LOAD ENV =================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY not found in .env file")
    st.stop()


# ================= CONFIG =================
PDF_FOLDER = "./documents"
FAISS_PATH = "faiss_index"


# ================= UI =================
st.set_page_config(page_title="📄 Your Personal AI Assistant", layout="wide")
st.title("📄 Your Personal AI Assistant")
st.write("Ask questions from your documents")

st.write("🚀 App started...")  # Debug


# ================= LOAD DOCUMENTS =================
def load_documents():
    docs = []

    for file in Path(PDF_FOLDER).glob("*.pdf"):
        loader = PyPDFLoader(str(file))
        pages = loader.load()

        for page in pages:
            page.metadata["source"] = file.name

        docs.extend(pages)

    return docs


# ================= SPLIT =================
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(docs)


# ================= VECTOR STORE =================
@st.cache_resource
def get_vectorstore():
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Create new FAISS index
    st.write("📄 Loading Documents...")

    docs = load_documents()
    print(len(docs))

    if not docs:
        st.error("❌ No PDFs found in ./documents folder")
        st.stop()

    chunks = split_documents(docs)
    # st.write(f"✂️ Total chunks: {len(chunks)}")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    # vectorstore.save_local(FAISS_PATH)

    st.success("✅ Documents loaded processed successfully.")

    return vectorstore


# ================= RAG CHAIN =================
def get_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer ONLY from the provided context.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
""")

    def format_docs(docs):
        return "\n\n".join(
            f"{doc.page_content}\n(Source: {doc.metadata.get('source', '')})"
            for doc in docs
        )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ================= INIT =================
with st.spinner("🔄 Setting up RAG system..."):
    vectorstore = get_vectorstore()
    rag_chain = get_rag_chain(vectorstore)

st.success("✅ Ready! Ask your questions below")


# ================= CHAT =================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
query = st.chat_input("Ask a question...")

if query:
    retrieved_docs = vectorstore.similarity_search(query, k=4)

    # st.write("🔍 Retrieved Chunks:")
    # for i, doc in enumerate(retrieved_docs):
    #     st.write(f"--- Chunk {i+1} ---")
    #     st.write(doc.page_content[:500])

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # st.write("📚 Final Context Sent to LLM:")
    # st.write(context[:1000])

    response = rag_chain.invoke(query)

    st.write("🤖 Response:")
    st.write(response)