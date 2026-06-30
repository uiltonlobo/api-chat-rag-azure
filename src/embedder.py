# embedder.py
# Módulo utilitário para geração de embeddings (vetores numéricos) via Azure OpenAI.
#
# O que é um embedding?
# É uma representação numérica de um texto — uma lista de floats (ex.: 1536 valores)
# que captura o "significado semântico" do texto. Textos com significado parecido
# ficam próximos no espaço vetorial, o que permite busca por similaridade.
#
# Em RAG, embeddings são usados em dois momentos:
#   1. Na ingestão: cada chunk de documento é convertido em vetor e salvo no índice.
#   2. Na busca: a pergunta do usuário também é convertida em vetor para encontrar
#      os chunks mais relevantes por proximidade (busca vetorial).

import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# Inicializa o cliente Azure OpenAI para chamadas à API de embeddings
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def get_embedding(text):
    """Converte um texto em vetor de embedding usando o modelo configurado no .env."""
    return client.embeddings.create(
        input=[text],
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")  # Ex.: text-embedding-ada-002
    ).data[0].embedding

# Teste rápido para validar que a conexão e o deployment estão funcionando
text_to_embed = "O Azure AI Search é incrível para RAG."
embedding = get_embedding(text_to_embed)
print(f"Vetor gerado com sucesso! Tamanho do vetor: {len(embedding)}")