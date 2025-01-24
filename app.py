import os
import fitz
from openai import OpenAI
import base64
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_image_description(image):
    """
    Obtiene una descripción para la imagen utilizando la API de OpenAI.
    
    Args:
        image (str): La imagen en formato base64.
    
    Returns:
        str: Descripción de la imagen generada por el modelo GPT-4.
    """
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image and give it an appropriate name. The output will be in format:\n imagename: description"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
                    ],
                }
            ],
        )
        return response.choices[0].message.content  
    except Exception as e:
        st.error(f"Error while generating image description: {str(e)}") 

def process_pdf(pdf_bytes):
    """
    Procesa un archivo PDF para extraer texto e imágenes, y genera descripciones para las imágenes.
    
    Args:
        pdf_bytes (bytes): Bytes del archivo PDF a procesar.
    
    Returns:
        tuple: Contiene el texto extraído del PDF y una lista con las descripciones de las imágenes.
    """
    try:
        # Extraer texto completo
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        
        # Extraer imgs 
        images = []
        image_descriptions = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Guardar la imagen
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                images.append(base64_image)
                description = get_image_description(base64_image)
                image_descriptions.append(description)

        doc.close()

        return text, images
    except Exception as e:
        raise Exception(f"Error while processing PDF: {str(e)}")

def main():
    """
    Función para ejecutar la aplicación Streamlit
    """

    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0.6)

    st.title("🔍 DocuGenius Assistant ")

    with st.sidebar:
        st.subheader("Your PDFs")
        pdf_doc = st.file_uploader("Upload PDF and click process", type="pdf", accept_multiple_files=False) # solo se guardara si es pdf

    # Initializar historial de mensajes
    message_history = StreamlitChatMessageHistory(key="chat_messages")

    # Agregar mensaje predefinido
    if not message_history.messages:
        message_history.add_ai_message("How can I help you?")

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", """You are an AI assistant designed to answer questions about a document. The user will upload a PDF file, and you should analyze its content. 
            When the user asks a question, search the document for relevant information and provide answers based only on the document's content.  If the question cannot be answered from the document, state that you cannot answer the question based on the document.  Do not use any external knowledge. 
            In case the document havn't the information, ask the user if he want you to respond using external knowledge and do that if he allows it."""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "The user has uploaded a document with the following content: {context}"),
            ("human", "{question}"),
        ]
    )

    # Se usa el chain con historial para conservar contexto visible de la conversacion
    chain_with_history = RunnableWithMessageHistory(
        prompt_template | model,
        lambda session_id: message_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    # Mostrar msjs
    for msg in message_history.messages:
        st.chat_message(msg.type).write(msg.content)


    # Formulario de entrada, columna a campo de entrada de texto, b boton de envio, c boton de archivo adjunto
    user_input = st.chat_input("Write here your message")
    # Subida de archivo
    if user_input:
        st.chat_message("human").write(user_input)

        # Verificar si hay datos del PDF procesados
        pdf_text = st.session_state.get("pdf_text", "")
        pdf_images = st.session_state.get("images", "")

        response = chain_with_history.invoke(
            {"question": user_input, "context":pdf_text+pdf_images}, 
            {"configurable": {"session_id": "any"}}
        )
        
        st.chat_message("ai").write(response.content)

    if pdf_doc: 
        pdf_bytes = pdf_doc.getvalue()
        
        text_chunks, images = process_pdf(pdf_bytes)
        
        st.sidebar.success("PDF processed successfully!")

        # Guardar los datos procesados en el estado de la sesión
        st.session_state["pdf_text"] = text_chunks
        st.session_state["images"] = "\nDescription of the images present in the PDF:\n".join(images)

if __name__ == "__main__":
    main()
