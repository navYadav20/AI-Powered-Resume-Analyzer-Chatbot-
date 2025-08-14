# AI-Powered-Resume-Analyzer-Chatbot-AI-Powered Resume Analyzer Chatbot (RAG + Gemini)

`Overview`

An interactive Streamlit app that lets you:

- Upload multiple PDF resumes and chat with their content (RAG over FAISS index), automatic chunking with page-level metadata.

- See cited sources (filename + page) for answers.

- Compare resumes to a pasted Job Description and get:

   - Final recommendation

    - Candidate scorecard with % match

    - Strengths/weaknesses per candidate

- JD analysis: LLM-generated recommendation and scorecard using relevant chunks per resume.
- Conversational Q&A over uploaded resumes


`Tech stack`

- Python, Streamlit

- LangChain, FAISS, HuggingFace Embeddings (all-MiniLM-L6-v2)

- Google Gemini (via langchain-google-genai)

- PyPDF2



`Project structure`
- app.py – main Streamlit app

- htmlTemplates.py – chat CSS + HTML templates

- requirements.txt – dependencies

- (local only, not in repo) .env, virtual env folders, .idea


`Getting started`

- Prerequisites

    - Python 3.9(recommended)

    -  API key

- Clone and install

    - git clone <your-repo-url>

    - python -m venv .venv

    - Activate the venv:

    - install requiremets

- Environment variables
    - Create a .env file in the project root: Copy API key 

``
Note: .env is intentionally not committed.
``


- Run

    - streamlit run app.py


