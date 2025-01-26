import os
from openai import RateLimitError
from functions import *
import streamlit as st
from dotenv import load_dotenv
import io
from langchain_community.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def main():
    """
    Función para ejecutar la aplicación Streamlit
    """
    try:
            
        model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0.1)
        
        st.title("🔍 DocuGenius Assistant ")
        
        # Agregar mensaje predefinido
        if 'message_history' not in st.session_state:
            st.session_state.message_history = [{"type": "ai", "content": "How can I help you?"}]

        with st.sidebar:
            st.subheader("Your PDFs")
            pdf_doc = st.file_uploader("Upload PDF and click process", type="pdf", accept_multiple_files=False) # solo se guardara si es pdf

        # retriever = None
        if pdf_doc and 'retriever' not in st.session_state: 
            pdf_bytes = io.BytesIO(pdf_doc.getvalue())
            
            texts,images = process_pdf(pdf_bytes)

            st.session_state.retriever = create_retriever(texts,images)

            st.sidebar.success("PDF processed successfully!")

        # Formulario de entrada
        user_input = st.chat_input("Write here your message")

        if user_input and 'retriever' in st.session_state:
            # RAG pipeline
            chain = (
                {
                    "context": st.session_state.retriever | RunnableLambda(split_image_text_types),
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(prompt_func)
                | model
                | StrOutputParser()
            )
            response = chain.invoke(user_input)
            st.session_state.message_history.append({"type": "user", "content": user_input})
            st.session_state.message_history.append({"type": "ai", "content": response})  
        
        if pdf_doc is None and 'retriever' in st.session_state: # manejar cuando se quita un archivo luego de haberlo subido
            del st.session_state.retriever

        if user_input and 'retriever' not in st.session_state:
            prompt = PromptTemplate(
                template=(
                """You are an AI assistant designed to answer questions about a document. 
                The user will upload a PDF file, and you should analyze its content text and images. 
                If the user asks a question without uploading a PDF, respond based on your general knowledge and 
                ask him if he wants to upload a PDF and help him understand.\n"""
                "User  question: {user_question}\n"
                ),
                input_variables=["user_question"]
            )
            chain = prompt | model | StrOutputParser()
            response = chain.invoke({"user_question": user_input})
            st.session_state.message_history.append({"type": "user", "content": user_input})
            st.session_state.message_history.append({"type": "ai", "content": response})  
    
        # Mostrar los mensajes actualizados
        for msg in st.session_state.message_history:
            st.chat_message(msg["type"]).write(msg["content"])
      
            
    except RateLimitError:
        st.warning("Request too large for the model assistant.")

if __name__ == "__main__":
    main()
