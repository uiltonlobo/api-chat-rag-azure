# retriever.py
# Implementa o "R" do RAG (Retrieval-Augmented Generation): a etapa de recuperação.
#
# Fluxo da busca vetorial (semantic search):
#   1. Converte a pergunta do usuário em embedding (vetor) via Azure OpenAI.
#   2. Envia esse vetor ao Azure AI Search para encontrar os chunks
#      de texto com maior similaridade cossenoidal.
#   3. Retorna os trechos de texto relevantes que servirão de contexto para o GPT.
#
# Nota: este módulo também exporta `openai_client` para ser reaproveitado
# pelo ingest_data.py, evitando instanciar o cliente múltiplas vezes.

import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# SearchClient: usado para buscar e inserir documentos no índice
search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name="meu-indice-chat",
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
)

# AzureOpenAI: usado para gerar o embedding da pergunta antes de buscar
openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def search_documents(query_text):
    """Busca os chunks mais relevantes para a pergunta usando busca vetorial."""

    # 1. Converte a pergunta em vetor para poder compará-la com os vetores do índice
    embedding = openai_client.embeddings.create(
        input=[query_text],
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    ).data[0].embedding

    # 2. Monta a query vetorial: busca os 3 vizinhos mais próximos (k_nearest_neighbors=3)
    #    no campo "content_vector" do índice
    vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=3, fields="content_vector")

    # 3. Executa a busca pura por vetor (search_text=None desativa a busca textual tradicional).
    #    Para busca híbrida (texto + vetor), bastaria passar um search_text aqui.
    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=["content"]  # Retorna apenas o campo de texto, sem o vetor
    )

    # Extrai e retorna apenas o conteúdo textual dos top-3 resultados
    return [res["content"] for res in results]