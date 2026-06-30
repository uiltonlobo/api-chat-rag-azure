# Chat RAG com Azure AI

Este projeto implementa um sistema de **Retrieval-Augmented Generation (RAG)** que permite consultar seus próprios documentos (PDFs/TXTs) utilizando a inteligência do Azure OpenAI e a velocidade de busca do Azure AI Search.

## Arquitetura Resumida
1. **Ingestão**: Converte documentos em vetores semânticos.
2. **Retrieval**: Busca trechos relevantes no Azure AI Search baseado na pergunta do usuário.
3. **Generation**: O GPT-4o gera uma resposta contextualizada baseada nos trechos encontrados.



## Pré-requisitos
* Conta no [Azure Portal](https://portal.azure.com/).
* Python 3.10+ instalado.
* Conhecimento básico em terminais de comando.

## 1. Configuração do Ambiente
Clone este repositório e crie um ambiente virtual para isolar as dependências:

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar o ambiente (Windows)
.\venv\Scripts\activate
# Ativar o ambiente (Linux/Mac)
source venv/bin/activate

# Instalar dependências
pip install openai azure-search-documents azure-identity python-dotenv fastapi uvicorn langchain pymupdf

## 2. Variáveis de Ambiente
Crie um arquivo .env na raiz do projeto com as seguintes credenciais:

```plaintext
AZURE_OPENAI_ENDPOINT=https://<SEU_RECURSO>[.openai.azure.com/](https://.openai.azure.com/)
AZURE_OPENAI_KEY=<SUA_CHAVE>
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_SEARCH_ENDPOINT=https://<SEU_SERVICO>.search.windows.net
AZURE_SEARCH_KEY=<SUA_CHAVE_SEARCH>
```

## 3. Provisão de Recursos no Azure

### Azure OpenAI
1. No Portal do Azure, crie um recurso **Azure OpenAI**.
2. Acesse o **Azure OpenAI Studio**.
3. Em **Deployments**, crie duas implantações:
   - Modelo `gpt-4o` (ou `gpt-4`) para geração de texto.
   - Modelo `text-embedding-3-small` para criação de vetores.

### Azure AI Search
1. Crie um recurso **Azure AI Search**.
2. Na aba **Chaves**, copie a URL e a Chave de Administração para o seu arquivo `.env`.

## 4. Criação do Índice Vetorial
O índice é onde seus dados residirão de forma otimizada para busca semântica. Execute o script de criação:

```bash
python create_index.py
```

Este comando criará o índice meu-indice-chat no seu serviço de busca.

## 5. Ingestão de Documentos (PDF/TXT)
Para popular o índice com o conteúdo dos seus documentos:

1. Coloque seus arquivos na pasta ./meus_documentos/.
2. Execute o script de ingestão:

```bash
python ingest_data.py
```

O script realizará o "chunking" (divisão de texto) e enviará os vetores automaticamente para o Azure AI Search.

## 6. Execução da API
Com os dados indexados, inicie o servidor da API:

```bash
python main.py
```

A API estará disponível localmente em: http://127.0.0.1:8000.

## 7. Testes via Swagger
Para testar a aplicação, acesse http://127.0.0.1:8000/docs no seu navegador.

### Exemplo de Teste com Histórico
Utilize o endpoint POST /chat com o seguinte formato JSON para manter o contexto da conversa:

```json
{
  "history": [
    {
      "role": "user",
      "content": "Qual é a política de reembolso da empresa?"
    },
    {
      "role": "assistant",
      "content": "A política permite reembolso em até 30 dias após a compra."
    },
    {
      "role": "user",
      "content": "E após esse prazo, o que acontece?"
    }
  ]
}
```

## 8. Notas de Produção
Para levar esta aplicação a um ambiente de produção (Azure App Service, Docker, etc.):

* Segurança: Nunca publique seu arquivo .env. Utilize as "Configurações de Aplicativo" (App Settings) no Azure para gerenciar as chaves de forma segura.
* Economia de Tokens: Implemente uma estratégia de "janela deslizante" (enviar apenas as últimas mensagens do histórico) para evitar custos excessivos com a OpenAI.
* Monitoramento: Considere integrar o Azure Application Insights para rastrear latência e erros em tempo real.
* Persistência: Para chats mais longos, utilize um banco de dados (ex: Redis ou Cosmos DB) para gerenciar o histórico via session_id, evitando a necessidade de reenviar todo o histórico a cada chamada.

### Licença
Este projeto foi desenvolvido como uma implementação de referência de RAG utilizando o ecossistema Azure.