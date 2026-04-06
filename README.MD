
📄 RAG-based Personal Assistant Chatbot

A simple Retrieval-Augmented Generation (RAG) chatbot built using LangChain, FAISS, OpenAI, and Streamlit.
This app allows you to ask questions from your own PDF documents and get context-aware answers.

⸻

🚀 Features
	•	📂 Load and process PDF documents
	•	✂️ Automatic text chunking
	•	🧠 Semantic search using FAISS vector store
	•	🤖 OpenAI-powered responses
	•	💬 Chat-style interface using Streamlit
	•	📌 Answers grounded in your documents (RAG)

⸻

🏗️ Tech Stack
	•	Python
	•	LangChain (latest LCEL)
	•	FAISS (vector database)
	•	OpenAI Embeddings + LLM
	•	Streamlit (UI)

⸻

📁 Project Structure

.
├── app.py               # Streamlit app
├── documents/           # Your PDF files
├── .env                 # API keys
└── README.md


⸻

⚙️ Setup Instructions

1️⃣ Clone the repository

git clone <your-repo-url>
cd <your-repo-folder>


⸻

2️⃣ Create virtual environment

python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\scripts\activate      # Windows


⸻

3️⃣ Install dependencies

pip install -U \
  streamlit \
  python-dotenv \
  langchain \
  langchain-community \
  langchain-openai \
  langchain-text-splitters \
  faiss-cpu \
  pypdf


⸻

4️⃣ Add OpenAI API Key

Create a .env file:

OPENAI_API_KEY=your_openai_key_here


⸻

5️⃣ Add your PDFs

Place your documents inside:

./documents/


⸻

6️⃣ Run the app

python3 -m streamlit run app.py

⸻

💬 How It Works
	1.	PDFs are loaded and split into chunks
	2.	Chunks are converted into embeddings
	3.	Stored in FAISS vector database
	4.	User query → relevant chunks retrieved
	5.	LLM generates answer using retrieved context

⸻

⚠️ Notes
	•	Ensure PDFs contain clean, readable text
	•	First run may take time due to embedding creation

⸻

📜 License

This project is open-source and available under the MIT License.


Siddharth Singh