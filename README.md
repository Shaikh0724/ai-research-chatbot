🔬 AI Research Assistant: What, Why, and How
A professional Retrieval-Augmented Generation (RAG) chatbot built to explore the world of Artificial Intelligence. This project indexes 10+ groundbreaking research papers to provide expert-level answers about AI mechanisms, tools, and ethics.

🚀 Features
Intelligent Retrieval: Uses LangChain and ChromaDB to fetch relevant context from academic papers.

Expert Knowledge: Deep insights into GPT-3, RAG, Chain-of-Thought, and AI Safety.

Source Attribution: Every answer includes citations (file name and page number) for transparency.

Chat History: Remembers the context of your conversation for a seamless experience.

Professional UI: Clean and minimal interface built with Streamlit.

📚 Key Research Papers Indexed
The system is built on 10 pillar papers, including:

Language Models are Few-Shot Learners (GPT-3) - Explaining AI's scale.

Retrieval-Augmented Generation (RAG) - Explaining the mechanism.

Chain-of-Thought Prompting - Explaining AI reasoning.

On the Dangers of Stochastic Parrots - Covering AI risks and ethics.

InstructGPT - How AI is aligned with human values.

🛠️ Tech Stack
Frontend: Streamlit

LLM: OpenAI GPT-4o-mini

Orchestration: LangChain

Vector Database: ChromaDB

Embeddings: OpenAI text-embedding-3-small

⚙️ Setup Instructions
1. Clone the repository

git clone https://github.com/Shaikh0724/ai-research-chatbot.git
cd ai-research-chatbot
2. Install dependencies

pip install -r requirements.txt
3. Environment Variables
Create a .env file in the root directory and add your OpenAI API key:

Code snippet

OPENAI_API_KEY=your_actual_api_key_here
4. Ingest Documents
Place your research PDFs in the data_folder/ and run:



python ingestion.py
5. Run the Application

streamlit run app.py
