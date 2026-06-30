# ingest_data.py
# Pipeline de ingestão de documentos PDF para o Azure AI Search.
#
# Este script é executado UMA VEZ (ou sempre que novos documentos forem adicionados)
# para popular o índice. Ele realiza três etapas principais:
#   1. Extração  — lê o texto bruto de cada PDF
#   2. Chunking  — divide o texto em pedaços menores (chunks) para não exceder
#                  o limite de tokens do modelo de embedding
#   3. Indexação — gera o embedding de cada chunk e envia tudo ao Azure AI Search

import os
import uuid
import fitz  # PyMuPDF: biblioteca para leitura de PDFs
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from retriever import openai_client  # Reutiliza o cliente OpenAI já configurado

load_dotenv()

# Cliente para upload de documentos no índice do Azure AI Search
search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name="meu-indice-chat",
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
)

def get_embedding(text):
    """Gera o vetor de embedding para um chunk de texto."""
    return openai_client.embeddings.create(
        input=[text],
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    ).data[0].embedding

def extract_text_from_pdf(pdf_path):
    """Extrai todo o texto de um PDF, página a página, usando PyMuPDF."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def ingest_pdfs(directory_path):
    """Processa todos os PDFs de um diretório e os indexa no Azure AI Search."""

    # RecursiveCharacterTextSplitter divide o texto tentando preservar parágrafos,
    # frases e palavras (nessa ordem). chunk_overlap garante continuidade de contexto
    # entre chunks vizinhos, evitando perda de informação nas bordas.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents_to_upload = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            print(f"Processando: {filename}")
            full_path = os.path.join(directory_path, filename)

            # Etapa 1: extrai o texto bruto do PDF
            raw_text = extract_text_from_pdf(full_path)

            # Etapa 2: divide o texto em chunks menores
            chunks = text_splitter.split_text(raw_text)

            # Etapa 3: para cada chunk, gera o embedding e monta o documento
            for chunk in chunks:
                doc = {
                    "id": str(uuid.uuid4()).replace("-", ""),  # IDs do Azure Search não aceitam hífens
                    "content": chunk,                          # Texto original do chunk
                    "content_vector": get_embedding(chunk)     # Vetor semântico do chunk
                }
                documents_to_upload.append(doc)

    # Envia todos os documentos de uma vez (batch upload) para melhor performance
    if documents_to_upload:
        search_client.upload_documents(documents=documents_to_upload)
        print(f"Sucesso! {len(documents_to_upload)} chunks de PDFs indexados.")

if __name__ == "__main__":
    # Coloque os arquivos PDF na pasta ./documents antes de rodar este script
    ingest_pdfs("./documents")