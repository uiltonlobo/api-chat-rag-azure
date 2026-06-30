# main.py
# Ponto de entrada da API REST construída com FastAPI.
#
# Este arquivo orquestra o pipeline RAG completo em um único endpoint POST /chat:
#   Requisição HTTP → Retrieval (Azure AI Search) → Generation (Azure OpenAI) → Resposta
#
# Para iniciar o servidor: uvicorn src.main:app --reload
# Documentação interativa disponível em: http://localhost:8000/docs

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv

# Importa os dois módulos principais do pipeline RAG
from retriever import search_documents   # Etapa R: busca no Azure AI Search
from generator import generate_answer    # Etapa G: geração com Azure OpenAI

load_dotenv()

app = FastAPI(title="Backend Chat RAG Pro")

# --- Modelos de dados (contrato da API) ---

class Message(BaseModel):
    role: str      # "user" ou "assistant"
    content: str

class ChatRequest(BaseModel):
    # O cliente envia o histórico completo a cada requisição.
    # A última mensagem deve ser do role "user" (a pergunta atual).
    history: List[Message]

# --- Endpoints ---

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # 1. Extrai a pergunta mais recente do usuário (última mensagem do histórico)
        user_message = request.history[-1].content

        # 2. RETRIEVAL — busca os chunks de texto mais relevantes no Azure AI Search
        #    usando busca vetorial (semântica) sobre a pergunta atual
        contexto = search_documents(user_message)

        # 3. GENERATION — envia histórico completo + contexto recuperado ao GPT.
        #    O histórico garante memória da conversa; o contexto fundamenta a resposta.
        answer = generate_answer(request.history, contexto)
        print(f"Resposta gerada: {answer}")

        return {
            "response": answer,
            "context_used": len(contexto) > 0  # Indica se algum contexto foi encontrado
        }

    except Exception as e:
        print(f"Erro no processamento: {e}")
        raise HTTPException(
            status_code=500,
            detail="Ocorreu um erro ao processar sua pergunta. Tente novamente mais tarde."
        )

if __name__ == "__main__":
    import uvicorn
    # Roda o servidor localmente na porta 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)