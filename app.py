import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template


# -------- PDF TEXT EXTRACTION WITH PAGE METADATA --------
def get_pdf_text_with_metadata(pdf_docs):
    docs = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                docs.append({
                    "text": page_text,
                    "metadata": {"page": page_num, "source": pdf.name}
                })
    return docs


# -------- CHUNKING WITH METADATA --------
def get_text_chunks_with_metadata(docs):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunked_docs = []
    for doc in docs:
        chunks = text_splitter.split_text(doc["text"])
        for chunk in chunks:
            chunked_docs.append({
                "text": chunk,
                "metadata": doc["metadata"]
            })
    return chunked_docs


# -------- VECTOR STORE --------
def get_vectorstore(chunked_docs):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    texts = [doc["text"] for doc in chunked_docs]
    metadatas = [doc["metadata"] for doc in chunked_docs]
    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )
    return vectorstore


# -------- CONVERSATION CHAIN WITH GEMINI & SOURCE DOCS --------
def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # MODIFIED: Corrected model name
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.3,
        convert_system_message_to_human=True
    )
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key="answer"
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )
    return conversation_chain


# -------- HANDLE USER QUESTION WITH CITATIONS --------
def handle_userinput(user_question):
    if not st.session_state.conversation:
        st.warning("Please upload and process at least one resume.")
        return

    with st.spinner("Thinking..."):
        response = st.session_state.conversation({'question': user_question})

    st.session_state.chat_history = response['chat_history']

    # Display chat history from the top
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.markdown(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.markdown(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            # Display sources only for the latest bot message
            if i == len(st.session_state.chat_history) - 1:
                if "source_documents" in response:
                    sources = response["source_documents"]
                    if sources:
                        st.markdown("📚 **Sources:**")
                        # Using a set to avoid duplicate source links
                        unique_sources = set()
                        for src in sources:
                            page = src.metadata.get("page", "?")
                            filename = src.metadata.get("source", "Unknown file")
                            unique_sources.add(f"- {filename} (Page {page})")
                        for usrc in sorted(list(unique_sources)):
                            st.markdown(usrc)


# -------- NEW: FUNCTION TO HANDLE JD ANALYSIS --------
def handle_jd_analysis(job_description, vectorstore):
    """Compares all resumes against the JD and returns the analysis."""
    st.markdown("---")
    st.subheader("🤖 AI-Powered Analysis & Recommendation")

    with st.spinner("Analyzing resumes against the job description... This may take a moment."):
        # 1. Get unique resume filenames from the vector store
        if not vectorstore.index_to_docstore_id:
            st.error("Vector store is empty. Please process resumes first.")
            return

        docstore = vectorstore.docstore._dict
        unique_filenames = set(doc.metadata['source'] for doc in docstore.values())

        # 2. Create the context for the prompt
        resume_contexts = ""
        for filename in unique_filenames:
            # Find relevant chunks for THIS resume using metadata filtering
            retriever = vectorstore.as_retriever(
                search_kwargs={'k': 5, 'filter': {'source': filename}}
            )
            relevant_docs = retriever.get_relevant_documents(job_description)
            # Concatenate the text of the relevant chunks
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            resume_contexts += f"\n\n--- Resume: {filename} ---\n{context}"

        # 3. Create the prompt for the LLM
        prompt = f"""
        You are an expert HR Analyst. Your task is to analyze the provided resume excerpts against the following job description.

        **Job Description:**
        {job_description}

        **Relevant Resume Excerpts:**
        {resume_contexts}

        **Your Analysis:**
        Please provide a detailed, structured comparison. Your output must include the following sections:
        1.  **Final Recommendation:** Start with a clear statement recommending the single best candidate for the role.
        2.  **Candidate Scorecard:** Provide a percentage match score for each resume (e.g., "JohnDoe_Resume.pdf": 90%, "JaneSmith_Resume.pdf": 75%).
        3.  **Detailed Breakdown:** For each candidate, write a concise paragraph explaining their strengths and weaknesses in relation to the job description, citing specific skills or experiences from their resume excerpts.
        """

        # 4. Get the analysis from Gemini
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.2  # Lower temperature for more factual analysis
        )
        analysis_result = llm.invoke(prompt).content

        # 5. Display the result
        st.markdown(analysis_result)
    st.success("Analysis complete!")


# -------- MAIN APP --------
def main():
    load_dotenv()
    st.set_page_config(page_title="Resume Analyzer", page_icon="🤖")
    st.write(css, unsafe_allow_html=True)

    # Session state initialization
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "vectorstore" not in st.session_state:  # NEW: Store vectorstore in session
        st.session_state.vectorstore = None

    # --- SIDEBAR ---
    with st.sidebar:
        st.subheader("Configuration")

        # MODIFIED: File uploader with limit
        pdf_docs = st.file_uploader(
            "Upload Resumes (Max 5 files)",
            accept_multiple_files=True,
            type="pdf"
        )

        if st.button("Process Documents"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            elif len(pdf_docs) > 5:
                st.warning("You can upload a maximum of 5 files.")
            else:
                with st.spinner("Processing documents..."):
                    docs = get_pdf_text_with_metadata(pdf_docs)
                    chunked_docs = get_text_chunks_with_metadata(docs)
                    st.session_state.vectorstore = get_vectorstore(chunked_docs)
                    st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)
                st.success("✅ Documents processed!")

        st.markdown("---")

        # NEW: Job Description Input
        st.subheader("Analyze Against Job Description")
        job_description = st.text_area("Paste the Job Description here", height=200)

        # NEW: Analysis Button
        if st.button("Analyze & Suggest Best Match"):
            if not st.session_state.vectorstore:
                st.warning("Please process the documents first.")
            elif not job_description:
                st.warning("Please paste a job description.")
            else:
                # Trigger the analysis in the main panel
                st.session_state.run_analysis = True
                st.session_state.jd_text = job_description

    # --- MAIN CHAT PANEL ---
    st.header("📄 Resume Analysis Chatbot")

    # MODIFIED: General Q&A input at the top
    st.subheader("General Q&A")
    st.write("Ask any question about the content of the uploaded resumes.")
    user_question = st.text_input("Your question:", key="user_q")
    if user_question:
        handle_userinput(user_question)

    # NEW: Trigger analysis if the button was clicked
    if st.session_state.get("run_analysis", False):
        handle_jd_analysis(st.session_state.jd_text, st.session_state.vectorstore)
        st.session_state.run_analysis = False  # Reset the flag


if __name__ == '__main__':
    main()