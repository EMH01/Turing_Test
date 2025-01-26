import os
from dotenv import load_dotenv
from openai import OpenAI
import fitz #pymupdf
import pymupdf4llm
import base64
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
def get_image_description(image):
    client = OpenAI()
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

def summarize(md_text):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You are an expert assistant in summarizing text. Provides a concise and coherent summary of the following text in markdown.
                Start the answer by explaining what the file is about..or a similar beginning"""
            },
            {
                "role": "user",
                "content": md_text
            }
        ],
    )
    return response.choices[0].message.content

def process_pdf(pdf_bytes):
    # Extraer texto completo
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    # Extraer texto como markdown
    md_text = pymupdf4llm.to_markdown(
        doc=doc,
        page_chunks=False,
        write_images=False
    )
    headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    md_text_splits = markdown_splitter.split_text(md_text)
    summary = summarize(md_text)
    summary_doc = Document(page_content=summary, metadata={"type": "summary file"})
    md_text_splits.insert(0, summary_doc)

    # Extraer imgs y obtener descripciones
    images = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images()
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            # Guardar la imagen
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            description = get_image_description(base64_image)

            image_doc = Document(page_content=description, metadata={"image/base64": base64_image})
            images.append(image_doc)

    doc.close()
    return md_text_splits,images

def create_retriever(texts, images):
    vector_store = InMemoryVectorStore(OpenAIEmbeddings())
    vector_store.add_documents(documents=texts)
    vector_store.add_documents(documents=images)
    retriever = vector_store.as_retriever()
    return retriever

def split_image_text_types(docs):
    texts = []
    images = []
    for doc in docs:
        if "image/base64" in doc.metadata:
            images.append(doc.metadata['image/base64'])
        else:
            texts.append(doc.page_content)
    return {"texts": texts, "images": images}

from langchain.schema import HumanMessage

def prompt_func(data_dict):
    # Joining the context texts into a single string
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    # Adding image(s) to the messages if present
    if data_dict["context"]["images"]:
        image_message = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{data_dict['context']['images'][0]}"
            },
        }
        messages.append(image_message)

    # Adding the text message for analysis
    text_message = {
        "type": "text",
        "text": (
            """You are an AI assistant designed to answer questions about a document. The user will upload a PDF file, and you should analyze its content text and images.
         When the user asks a question, search the document for relevant information and provide answers based on the document's content. 
         If the question cannot be answered from the document or it isn't related with the content, state that you cannot answer the question based on the document.  Do not use any external knowledge.\n"""
           f"User question: {data_dict['question']}\n\n"
            "Text and / or tables:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)

    return [HumanMessage(content=messages)]


