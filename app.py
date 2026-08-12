import streamlit as st
import os
from dotenv import load_dotenv
from utils import process_pdf

# OpenRouter
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    st.error("OpenRouter API Key not found")
    st.stop()

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

st.set_page_config(page_title="AI Multi App")
st.title("🤖 AI Multi-Tool App")

# Sidebar
option = st.sidebar.selectbox(
    "Choose Feature",
    ["💬 Chatbot", "📄 PDF Q&A (RAG)", "✍️ Summarizer"]
)
# ==============================
# 1. CHATBOT
# ==============================
if option == "💬 Chatbot":

    st.subheader("Chat with AI")

    user_input = st.text_input("Ask anything")

    if user_input:

        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": user_input
                }
            ]
        )

        st.write(
            "🤖",
            response.choices[0].message.content
        )
# ==============================
# 2. PDF Q&A (RAG)
# ==============================
elif option == "📄 PDF Q&A (RAG)":

    st.subheader("Ask Questions from PDF")

    pdf = st.file_uploader("Upload PDF", type="pdf")

    if pdf:

        with open("temp.pdf", "wb") as f:
            f.write(pdf.read())

        db = process_pdf("temp.pdf")

        retriever = db.as_retriever()

        question = st.text_input("Ask question from PDF")

        if question:

            docs = retriever.get_relevant_documents(question)

            context = " ".join(
                [d.page_content for d in docs]
            )

            prompt = f"""
            Answer based on context only.

            Context:
            {context}

            Question:
            {question}
            """

            response = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            st.write(
                "📘 Answer:",
                response.choices[0].message.content
            )
# ==============================
# 3. SUMMARIZER
# ==============================
elif option == "✍️ Summarizer":

    st.subheader("Text Summarizer")

    text = st.text_area("Enter text")

    if st.button("Summarize"):

        prompt = f"Summarize this text:\n{text}"

        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        st.write(
            "📝 Summary:",
            response.choices[0].message.content
        )