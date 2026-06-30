# create_index.py
# Responsável por criar (ou recriar) o índice no Azure AI Search.
# Em RAG, o índice é o "banco de dados" onde os chunks de texto e seus
# vetores (embeddings) ficam armazenados para busca semântica posterior.

import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SearchField, SearchFieldDataType, SimpleField,
    SearchableField, VectorSearch, VectorSearchProfile,
    HnswAlgorithmConfiguration
)

load_dotenv()

# Lê as credenciais do Azure AI Search a partir do .env
endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
key = os.getenv("AZURE_SEARCH_KEY")
index_name = "meu-indice-chat"

# SearchIndexClient gerencia a estrutura do índice (criação, deleção, etc.)
# É diferente do SearchClient, que é usado para fazer buscas e uploads de documentos.
client = SearchIndexClient(endpoint, AzureKeyCredential(key))

# Definição dos campos do índice — equivale ao schema de uma tabela.
# Cada documento indexado terá estes três campos:
fields = [
    # Campo-chave obrigatório: identifica unicamente cada documento
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    # Campo de texto: armazena o chunk de texto original; permite busca textual (full-text)
    SearchableField(name="content", type=SearchFieldDataType.String, sortable=True),
    # Campo vetorial: armazena o embedding do chunk.
    # vector_search_dimensions=1536 corresponde ao tamanho do vetor gerado pelo
    # modelo text-embedding-ada-002 da Azure OpenAI.
    SearchField(
        name="content_vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        vector_search_dimensions=1536,
        vector_search_profile_name="my-vector-profile"
    )
]

# Configuração do algoritmo de busca vetorial.
# HNSW (Hierarchical Navigable Small World) é um algoritmo de busca aproximada
# de vizinhos mais próximos (ANN), muito eficiente para grandes volumes de vetores.
vector_search = VectorSearch(
    profiles=[VectorSearchProfile(name="my-vector-profile", algorithm_configuration_name="my-hnsw-config")],
    algorithms=[HnswAlgorithmConfiguration(name="my-hnsw-config")]
)

# Cria ou atualiza o índice (idempotente: seguro de rodar mais de uma vez)
index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
result = client.create_or_update_index(index)

print(f"Índice '{result.name}' criado com sucesso!")