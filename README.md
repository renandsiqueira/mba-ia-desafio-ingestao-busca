# Desafio MBA Engenharia de Software com IA - Full Cycle

Este projeto usa Python, um ambiente virtual e PostgreSQL com a extensão `pgvector`. Realiza ingestão de PDFs em dois vetores (OpenAI e Google Gemini) e permite busca e chat com RAG.

## Pré-requisitos

- Python 3.12 ou compatível com o ambiente virtual do projeto (Python 3.14 não funcionou com algumas dependências)
- `pip` e suporte ao módulo `venv`
- Docker e Docker Compose

## 1. Criar e ativar o ambiente virtual

Se o suporte ao `venv` ainda não estiver instalado no seu sistema:

```bash
sudo apt update && sudo apt install python3-full -y
```

Depois, crie o ambiente virtual na raiz do projeto:

```bash
python3.12 -m venv .venv
```

Ative o ambiente virtual:

```bash
source .venv/bin/activate
```

## 2. Instalar as dependências

Com o ambiente virtual ativo, instale os pacotes do projeto:

```bash
pip install -r requirements.txt
```

## 3. Configurar as variáveis de ambiente

Crie um arquivo `.env` a partir do `.env.example` e preencha os valores necessários:

```bash
cp .env.example .env
```

As variáveis usadas pelo código são:

| Variável | Descrição | Padrão |
|---|---|---|
| `PDF_PATH` | Caminho para o arquivo PDF a ser ingerido | `./document.pdf` |
| `DATABASE_URL` | URL de conexão com o PostgreSQL | `postgresql://postgres:postgres@localhost:5432/rag` |
| `PG_VECTOR_COLLECTION` | Nome base das coleções no pgvector | `my-embedding-documents` |
| `GOOGLE_API_KEY` | Chave de API do Google | — |
| `GOOGLE_EMBEDDING_MODEL` | Modelo de embedding do Google | `models/gemini-embedding-001` |
| `OPENAI_API_KEY` | Chave de API da OpenAI | — |
| `OPENAI_EMBEDDING_MODEL` | Modelo de embedding da OpenAI | `text-embedding-3-small` |

> **Atenção:** o modelo `models/embedding-001` foi descontinuado pelo Google. O projeto já usa o modelo atual `models/gemini-embedding-001`.

## 4. Subir o banco de dados

O projeto inclui um `docker-compose.yml` com PostgreSQL e `pgvector`.

```bash
docker compose up -d
```

Isso inicia o banco em `localhost:5432` com as credenciais definidas no compose.

## 5. Executar os scripts

### Ingestão do PDF

Carrega o PDF, divide em chunks e armazena embeddings em duas coleções no pgvector:
- `{PG_VECTOR_COLLECTION}-gemini` — gerada com Google Gemini (com rate limit: 5 docs/lote, 10s de espera)
- `{PG_VECTOR_COLLECTION}-openai` — gerada com OpenAI (sem rate limit)

```bash
python3 src/ingest.py
```

### Chat

Interface de chat com RAG sobre o conteúdo do PDF:

```bash
python3 src/chat.py
```

## Observações

- Se algum script retornar erro de inicialização, verifique as variáveis de ambiente e a conexão com o banco.
- As coleções no banco são nomeadas automaticamente com sufixo `-gemini` e `-openai` a partir do valor de `PG_VECTOR_COLLECTION`.