# generator.py
# Implementa o "G" do RAG (Retrieval-Augmented Generation): a etapa de geração.
#
# Responsabilidade: receber o histórico da conversa + os chunks de contexto
# recuperados pelo retriever e enviar tudo ao GPT para gerar uma resposta fundamentada.
#
# Diferença em relação a um chat comum:
#   - Um chat simples envia apenas o histórico ao GPT.
#   - No RAG, injetamos no system prompt os trechos relevantes dos documentos,
#     dando ao modelo uma "base de conhecimento" específica para responder.

import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from typing import List

from pydantic import BaseModel

load_dotenv()

class Message(BaseModel):
    role: str  # "user" ou "assistant"
    content: str

# Inicializa o cliente Azure OpenAI para chamadas ao modelo de chat (GPT)
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def generate_answer(history: List[Message], context):
    """Gera uma resposta usando o histórico da conversa e o contexto recuperado."""

    # O system prompt instrui o modelo a usar apenas o contexto fornecido.
    # Os chunks recuperados pelo retriever são concatenados e injetados aqui.
    # Isso "ancora" o modelo nos documentos indexados, reduzindo alucinações.
    messages = [
        {"role": "system", "content": f"Use este contexto para responder: {chr(10).join(context)}"}
    ]

    # Adiciona o histórico completo da conversa (mantém a "memória" do chat)
    # O último item do histórico é a pergunta atual do usuário.
    for msg in history:
        messages.append({"role": msg.role, "content": msg.content})

    # Envia ao modelo de chat configurado no Azure OpenAI.
    # temperature=0.3: valor baixo torna as respostas mais determinísticas e precisas,
    # adequado para Q&A sobre documentos (evita respostas criativas/inventadas).
    response = client.chat.completions.create(
        model="chat-rag-01-gpt-5.4-mini",
        messages=messages,
        temperature=0.3
    )
    return response.choices[0].message.content